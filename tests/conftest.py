import asyncio
import hashlib
import shutil
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path

import pytest

from quant_research import datasets
from quant_research.access import issue_context
from quant_research.catalog import Approved, Catalog, Documented, Evidence, Validated, Vintage
from quant_research.records import canonical_bytes
from quant_research.worker import IMAGE

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cli():
    def run(directory: Path, *arguments: str, success=True):
        bootstrap = (
            f"import runpy, sys; sys.path.append({str(ROOT / 'src')!r}); "
            f"sys.argv = {['quant_research', *arguments]!r}; "
            "runpy.run_module('quant_research', run_name='__main__')"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-I", "-S", "-c", bootstrap],
            cwd=directory,
            text=True,
            capture_output=True,
        )
        assert (result.returncode == 0) == success, result.stdout + result.stderr
        return result

    return run


@pytest.fixture(scope="session")
def running():
    from quant_research import store

    @contextmanager
    def start(directory: Path):
        if shutil.which(store.SERVER_EXECUTABLE) is None:
            pytest.fail(
                f"SurrealDB {store.SERVER_VERSION} is not installed. Run: {store.INSTALL_COMMAND}"
            )
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            server = store.LocalServer(directory / "db", probe.getsockname()[1])
        log = directory / "server.log"
        with log.open("wb") as output:
            process = subprocess.Popen(
                server.start_command(), stdout=output, stderr=subprocess.STDOUT
            )
        try:
            deadline = time.monotonic() + 30
            while True:
                with (
                    suppress(OSError),
                    urllib.request.urlopen(f"{server.endpoint}/health", timeout=1) as response,
                ):
                    if response.status == 200:
                        break
                if process.poll() is not None or time.monotonic() > deadline:
                    pytest.fail(
                        f"SurrealDB at {server.endpoint} did not become healthy "
                        f"(exit status {process.poll()}):\n{log.read_text()}"
                    )
                time.sleep(0.05)
            yield server
        finally:
            process.terminate()
            process.wait()

    return start


@pytest.fixture(scope="session")
def server(running, tmp_path_factory):
    with running(tmp_path_factory.mktemp("surrealdb")) as started:
        yield started


QUALIFIED_CLI = "temporal version 1.9.1 (Server 1.32.0, UI 2.54.1)"
CLI_RELEASE = "https://github.com/temporalio/cli/releases/tag/v1.9.1"


@pytest.fixture(scope="session")
def temporal(tmp_path_factory):
    """Start the qualified Temporal dev server once per session and return its address."""
    from temporalio.api.workflowservice.v1 import DescribeNamespaceRequest
    from temporalio.client import Client
    from temporalio.service import RPCError

    async def answered(address: str, log: Path) -> None:
        deadline = time.monotonic() + 30
        while True:
            try:
                client = await Client.connect(address)
                await client.workflow_service.describe_namespace(
                    DescribeNamespaceRequest(namespace="default")
                )
                return
            except (RuntimeError, RPCError):
                if time.monotonic() > deadline:
                    pytest.fail(f"Temporal dev server at {address} did not answer; see {log}")
                await asyncio.sleep(0.1)

    cli = shutil.which("temporal")
    if cli is None:
        pytest.fail(f"Temporal CLI not found; run `brew install temporal` or see {CLI_RELEASE}")
    found = subprocess.run([cli, "--version"], capture_output=True, text=True, check=True)
    if found.stdout.strip() != QUALIFIED_CLI:
        pytest.fail(
            f"Found {found.stdout.strip()!r} but docs/decisions.md qualified {QUALIFIED_CLI!r} "
            f"({CLI_RELEASE}); requalify Temporal before running on another version"
        )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    log = tmp_path_factory.mktemp("temporal") / "server.log"
    with log.open("wb") as output:
        process = subprocess.Popen(
            [cli, "server", "start-dev", "--headless", "--ip", "127.0.0.1", "--port", str(port)],
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    address = f"127.0.0.1:{port}"
    try:
        asyncio.run(answered(address, log))
        yield address
    finally:
        process.terminate()
        process.wait()


@pytest.fixture
def reachable_text():
    """Return a function that joins an error's traceback and every attribute along its chain."""

    def join(error: BaseException) -> str:
        parts, pending, seen = ["".join(traceback.format_exception(error))], [error], set()
        while pending:
            node = pending.pop()
            if node is None or id(node) in seen:
                continue
            seen.add(id(node))
            parts += [repr(getattr(node, name)) for name in dir(node) if not name.startswith("__")]
            pending += [node.__cause__, node.__context__]
        return "".join(parts)

    return join


@pytest.fixture(scope="session")
def image() -> str:
    """Return the worker image ID, skipping only when no docker executable is on PATH."""
    if shutil.which("docker") is None:
        pytest.skip("Docker is not installed")
    found = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE], capture_output=True, text=True
    )
    if found.returncode:
        pytest.fail(f"Run make worker-image; docker cannot supply {IMAGE}: {found.stderr.strip()}")
    return found.stdout.strip()


class Probe:
    """Store and approve the synthetic dataset that the worker and runner tests read."""

    name = "vendor.probe"
    context = issue_context(
        clearances=(),
        cutoff=datetime(2015, 7, 1, tzinfo=UTC),
        generation="synthetic/v1",
        providers=(),
    )

    def catalog(self, *access: str, vintage: str = "0" * 64) -> Catalog:
        evidence = Evidence(hashlib.sha256(b"synthetic").hexdigest(), "Synthetic evidence")
        vintages = tuple(
            Vintage(aspect, evidence) for aspect in ("publication", "revision", "arrival")
        )
        at = datetime(2026, 10, 21, 12, tzinfo=UTC)
        return Catalog(
            (
                Documented(
                    self.name, "researcher", at, "Data", ("close",), access, vintage, (), vintages
                ),
                Validated(self.name, "reviewer", at, evidence),
                Approved(self.name, "lead", at, datasets.ADAPTER),
            )
        )

    @staticmethod
    def version(key: str, revision: int = 1, at: str = "2015-06-01T20:00:00Z", **values) -> dict:
        return {
            "arrived_at": at,
            "key": key,
            "published_at": at,
            "revision": revision,
            "values": values or {"close": "1"},
        }

    def save(self, store: Path, rows: list[str | dict]) -> str:
        """Store rows, where a bare key stands for its first version, and return the vintage."""
        rows = [self.version(row) if isinstance(row, str) else row for row in rows]
        document = {"dataset": self.name, "rows": rows, "schema_version": 1}
        return datasets.save(store, self.name, canonical_bytes(document))


@pytest.fixture(scope="session")
def probe() -> Probe:
    return Probe()
