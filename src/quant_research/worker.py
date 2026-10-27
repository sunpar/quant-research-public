from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import select
import subprocess
import tarfile
import tempfile
import threading
import time
from concurrent.futures import Future, wait
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from quant_research import datasets
from quant_research.access import RunContext
from quant_research.catalog import Catalog
from quant_research.records import (
    PackageError,
    append_log,
    canonical_bytes,
    canonical_json,
    format_time,
    parse_digest,
    parse_fields,
    parse_uuid,
    read_log,
    unique_keys,
)

IMAGE = "quant-research-worker:2"
LABEL = "quant_research.worker"

# tar adds headers and pads to 10 KiB records. The output tmpfs and the stream bound exceed the
# signal limit by this much, so an oversized signal reaches the host's own size check.
_FRAMING = 2**16

type Reason = (
    datasets.Reason
    | Literal[
        "timeout",
        "memory",
        "failed",
        "missing_output",
        "extra_output",
        "oversized",
        "malformed",
        "non_finite",
        "wrong_index",
    ]
)

_MESSAGES: dict[str, str] = {
    "timeout": "The candidate ran past the wall-clock limit",
    "memory": "The candidate exceeded the memory limit",
    "failed": "The candidate exited with an error",
    "missing_output": "The candidate wrote no signal file",
    "extra_output": "The output directory holds something other than the signal file",
    "oversized": "The output exceeds the size limit",
    "malformed": "The signal is not a JSON object of numbers",
    "non_finite": "The signal holds a NaN or infinite value",
    "wrong_index": "The signal keys differ from the slice keys",
}


class WorkerError(RuntimeError):
    """Report a Docker failure, a missing worker image, or a run log that cannot be replayed."""


@dataclass(frozen=True)
class Limits:
    """Resource limits for one container. Sizes are bytes and the timeout is wall-clock seconds."""

    timeout: float = 60.0
    memory: int = 512 * 2**20
    cpus: float = 1.0
    pids: int = 64
    scratch: int = 64 * 2**20
    output: int = 2**20


LIMITS = Limits()


@dataclass(frozen=True)
class RunRecord:
    """Commit one run to its inputs, so it can be repeated.

    image is the Docker image ID and source the SHA-256 of the UTF-8 candidate source. read
    replays the slice from the dataset bytes.
    """

    image: str
    source: str
    read: datasets.ReadRecord
    limits: Limits

    def to_bytes(self) -> bytes:
        """Return canonical UTF-8 JSON with the schema version and every field."""
        return canonical_bytes(
            {
                "image": self.image,
                "limits": asdict(self.limits),
                "read": json.loads(self.read.to_bytes()),
                "schema_version": 1,
                "source": self.source,
            }
        )


@dataclass(frozen=True)
class Signal:
    """A validated candidate output: one finite value for each slice key, in slice order."""

    values: dict[str, float]
    record: RunRecord


@dataclass(frozen=True)
class Refused:
    """A run with no accepted output. record is None when the dataset read was refused."""

    reason: Reason
    record: RunRecord | None = None

    @property
    def message(self) -> str:
        return _MESSAGES.get(self.reason) or datasets.Refused(self.reason).message


@dataclass(frozen=True)
class LoggedRun:
    """One container run as a RunLog keeps it.

    outcome is "signal" or the refusal reason. output is the SHA-256 of the signal values'
    canonical JSON for a signal and None for a refusal.
    """

    run_id: UUID
    record: RunRecord
    outcome: Literal["signal"] | Reason
    output: str | None

    def to_bytes(self) -> bytes:
        """Return the canonical UTF-8 JSON run log entry, with the schema version."""
        return canonical_bytes(
            {
                "run": {
                    "outcome": self.outcome,
                    "output": self.output,
                    "record": json.loads(self.record.to_bytes()),
                    "run_id": str(self.run_id),
                },
                "schema_version": 1,
            }
        )


def run(
    source: str,
    catalog: Catalog,
    store: Path,
    dataset: str,
    context: RunContext,
    limits: Limits = LIMITS,
) -> Signal | Refused:
    """Run generated Python over an approved dataset slice in a locked-down container.

    The slice is what `datasets.read` supplies to the context. The container sees the slice at
    /job/slice.json and the source at /job/candidate.py, both read-only, and must write one JSON
    object mapping every slice key to a finite number to /output/signal.json. It has no network,
    no host environment, and only a tmpfs scratch directory at /tmp.

    Args:
        source (str): Candidate Python source.
        catalog (Catalog): Catalog that must approve the dataset for the adapter.
        store (Path): Dataset store, as `datasets.save` writes it.
        dataset (str): Dataset name.
        context (RunContext): Trusted run authority.
        limits (Limits, optional): Resource limits for the container.

    Returns:
        (Signal | Refused) The validated signal, or the reason there is none.

    Raises:
        WorkerError: If Docker fails or the worker image is not built.
    """
    read = datasets.read(catalog, store, dataset, context)
    if isinstance(read, datasets.Refused):
        return Refused(read.reason)
    try:
        image = _docker("image", "inspect", "--format", "{{.Id}}", IMAGE)
    except WorkerError as error:
        raise WorkerError(f"Build {IMAGE} with make worker-image") from error
    code = source.encode()
    record = RunRecord(image, hashlib.sha256(code).hexdigest(), read.record, limits)
    rows = [
        {
            "arrived_at": format_time(row.arrived_at),
            "key": row.key,
            "published_at": format_time(row.published_at),
            "revision": row.revision,
            "values": row.values,
        }
        for row in read.rows
    ]
    with tempfile.TemporaryDirectory() as directory:
        job = Path(directory)
        job.chmod(0o755)
        for name, data in (
            ("candidate.py", code),
            ("slice.json", canonical_bytes({"dataset": dataset, "rows": rows})),
        ):
            (job / name).write_bytes(data)
            (job / name).chmod(0o644)
        stream = _execute(job, record)
    if isinstance(stream, str):
        return Refused(stream, record)
    values = _validate(stream, [row.key for row in read.rows], limits.output)
    return Refused(values, record) if isinstance(values, str) else Signal(values, record)


class RunLog:
    """Keep each container run as a numbered canonical JSON file in an existing directory.

    Run i is the file i.json, read and written through records.read_log and append_log. Every
    operation first replays the stored runs, so logs opened on one directory never act on stale
    state.
    """

    def __init__(self, directory: Path) -> None:
        """Open a run log and verify every stored run.

        Args:
            directory (Path): Existing directory that holds the numbered run files.
        """
        self._directory = directory
        self.load()

    def load(self) -> tuple[LoggedRun, ...]:
        """Return every stored run in order, rechecking each one.

        Returns:
            (tuple[LoggedRun, ...]) The stored runs in log order.
        """
        runs: dict[UUID, LoggedRun] = {}
        try:
            for index, value in enumerate(read_log(self._directory, "Run log entry", "Run files")):
                logged = _logged(value)
                if logged.run_id in runs:
                    raise PackageError(f"Run log entry {index}.json repeats a run")
                runs[logged.run_id] = logged
        except (OSError, ValueError, RecursionError) as error:
            raise WorkerError(f"Cannot replay run log {self._directory}: {error}") from error
        return tuple(runs.values())

    def append(self, result: Signal | Refused) -> LoggedRun:
        """Store a container run under a new run ID.

        Args:
            result (Signal | Refused): What `run` returned for a run that started a container.

        Returns:
            (LoggedRun) The stored run.

        Raises:
            WorkerError: If the result is a refused read, which started no container, if load
                would refuse the run, or if the log cannot be replayed or written.
        """
        match result:
            case Signal(values, record):
                outcome, output = "signal", hashlib.sha256(canonical_bytes(values)).hexdigest()
            case Refused(reason, RunRecord() as record):
                outcome, output = reason, None
            case _:
                raise WorkerError("A refused read started no container, so it has no run to log")
        logged = LoggedRun(uuid4(), record, outcome, output)
        try:
            content = logged.to_bytes()
            # Parse with load's own check, so a run that load refuses never locks the log.
            _logged(canonical_json(content, "Run log entry"))
        except ValueError as error:
            raise WorkerError(f"Cannot log this run: {error}") from error
        try:
            return append_log(self._directory, lambda: (len(self.load()), logged, content))
        except OSError as error:
            raise WorkerError(f"Cannot write run log {self._directory}: {error}") from error


def _docker(*args: str) -> str:
    # The daemon finishes a request it has received even after its CLI dies, and an exception
    # inside subprocess.run kills or loses that CLI. Python runs signal handlers only on the main
    # thread, and a terminal Ctrl-C signals the whole foreground process group.
    future: Future[subprocess.CompletedProcess[str]] = Future()

    def call() -> None:
        if future.set_running_or_notify_cancel():
            try:
                future.set_result(
                    subprocess.run(
                        ["docker", *args], capture_output=True, text=True, start_new_session=True
                    )
                )
            except BaseException as error:
                future.set_exception(error)

    try:
        threading.Thread(target=call, daemon=True).start()
        wait([future])
    except BaseException:
        if not future.cancel():
            wait([future])
        raise
    try:
        result = future.result()
    except OSError as error:
        raise WorkerError(f"Cannot run docker: {error}") from error
    if result.returncode:
        raise WorkerError(f"docker {args[0]} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _execute(job: Path, record: RunRecord) -> bytes | Reason:
    limits = record.limits
    container = f"quant-research-worker-{uuid4().hex}"
    try:
        _docker(
            "create",
            f"--name={container}",
            "--pull=never",
            "--attach=stdout",
            "--log-driver=none",
            f"--label={LABEL}={job}",
            "--network=none",
            "--ipc=none",
            "--read-only",
            f"--mount=type=bind,source={job},target=/job,readonly",
            f"--tmpfs=/tmp:rw,nosuid,nodev,noexec,size={limits.scratch},mode=1777",
            f"--tmpfs=/output:rw,nosuid,nodev,noexec,size={limits.output + _FRAMING},mode=1777",
            "--tmpfs=/dev/mqueue:ro",
            "--user=65534:65534",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            f"--pids-limit={limits.pids}",
            f"--memory={limits.memory}",
            f"--memory-swap={limits.memory}",
            f"--cpus={limits.cpus}",
            record.image,
        )
        with subprocess.Popen(
            ["docker", "start", "--attach", container],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ) as process:
            try:
                return _collect(process, container, limits)
            finally:
                # Killing the attached client never stops the container, and lets Popen exit.
                process.kill()
    finally:
        _docker("rm", "--force", "--volumes", container)


def _collect(process: subprocess.Popen, container: str, limits: Limits) -> bytes | Reason:
    deadline = time.monotonic() + limits.timeout
    chunks: list[bytes] = []
    size = 0
    while (remaining := deadline - time.monotonic()) > 0:
        if not select.select([process.stdout], [], [], remaining)[0]:
            continue
        chunk = os.read(process.stdout.fileno(), 2**16)
        if not chunk:
            break
        size += len(chunk)
        if size > limits.output + 2 * _FRAMING:
            return "oversized"
        chunks.append(chunk)
    else:
        return "timeout"
    try:
        process.wait(max(deadline - time.monotonic(), 0))
    except subprocess.TimeoutExpired:
        return "timeout"
    state = json.loads(_docker("inspect", "--format", "{{json .State}}", container))
    if state["Error"]:
        raise WorkerError(f"Container did not run: {state['Error']}")
    if state["OOMKilled"]:
        return "memory"
    if state["ExitCode"]:
        return "failed"
    return b"".join(chunks)


def _validate(stream: bytes, keys: list[str], limit: int) -> dict[str, float] | Reason:
    try:
        with tarfile.open(fileobj=io.BytesIO(stream), mode="r:") as archive:
            members = sorted((member.name, member.isfile()) for member in archive.getmembers())
            # Bytes after the end marker would be a second archive that tarfile never reads.
            if stream[archive.offset :].strip(b"\0"):
                return "malformed"
            if members == [(".", False)]:
                return "missing_output"
            if members != [(".", False), ("./signal.json", True)]:
                return "extra_output"
            member = archive.getmember("./signal.json")
            if member.size > limit:
                return "oversized"
            data = archive.extractfile(member).read()
    except tarfile.TarError:
        return "malformed"
    try:
        value = json.loads(data.decode(), parse_int=float, object_pairs_hook=unique_keys)
    except (ValueError, RecursionError):
        return "malformed"
    if not isinstance(value, dict) or any(type(item) is not float for item in value.values()):
        return "malformed"
    if not all(map(math.isfinite, value.values())):
        return "non_finite"
    if value.keys() != set(keys):
        return "wrong_index"
    return {key: value[key] for key in keys}


def _logged(raw: object) -> LoggedRun:
    value = parse_fields(raw, "run schema_version", "run log entry")
    _version(value["schema_version"])
    entry = parse_fields(value["run"], "outcome output record run_id", "run")
    match entry["outcome"], entry["output"]:
        case "signal", output:
            parse_digest(output)
        case str() as reason, None if reason in _MESSAGES:
            pass
        case outcome, output:
            raise PackageError(f"Invalid outcome {outcome!r} with output {output!r}")
    return LoggedRun(
        parse_uuid(entry["run_id"]), _record(entry["record"]), entry["outcome"], entry["output"]
    )


def _record(raw: object) -> RunRecord:
    value = parse_fields(raw, "image limits read schema_version source", "run record")
    _version(value["schema_version"])
    image = value["image"]
    if not isinstance(image, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise PackageError(f"Invalid image ID {image!r}")
    limits = parse_fields(value["limits"], " ".join(item.name for item in fields(Limits)), "limits")
    for name, limit in limits.items():
        if type(limit) not in (int, float):
            raise PackageError(f"Invalid limit {name} {limit!r}")
    return RunRecord(
        image,
        parse_digest(value["source"]),
        datasets.ReadRecord.from_bytes(canonical_bytes(value["read"])),
        Limits(**limits),
    )


def _version(value: object) -> None:
    if type(value) is not int or value != 1:
        raise PackageError(f"Unsupported schema version {value!r}")
