from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import re
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from uuid import UUID

from surrealdb import BlockingWsSurrealConnection, QueryError, RecordID

from quant_research import store
from quant_research.findings import Event, Ledger, event_value, parse_event
from quant_research.invalidation import Invalidation, parse_invalidation
from quant_research.invalidation import catch_up as catch_up_ledger
from quant_research.packages import VerifiedPackage, open_package
from quant_research.records import PackageError, canonical_bytes
from quant_research.worker import RunLog

PUBLISH_ATTEMPTS = 5

SCHEMA = """
DEFINE TABLE IF NOT EXISTS object SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS size ON object TYPE int;

DEFINE TABLE IF NOT EXISTS specification SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS question ON specification TYPE string;
DEFINE FIELD IF NOT EXISTS signal_profile ON specification TYPE string;
DEFINE FIELD IF NOT EXISTS evaluation_profile ON specification TYPE string;
DEFINE FIELD IF NOT EXISTS data_version ON specification TYPE string;
DEFINE FIELD IF NOT EXISTS implementation_version ON specification TYPE string;
DEFINE FIELD IF NOT EXISTS inputs ON specification TYPE array<record<object>>;
DEFINE FIELD IF NOT EXISTS data_contract ON specification TYPE record<object>;
DEFINE FIELD IF NOT EXISTS implementation ON specification TYPE record<object>;
DEFINE FIELD IF NOT EXISTS environment ON specification TYPE {
    implementation: string, python_version: string, abi: string, platform: string,
    uv_version: string, lock: record<object>
};

DEFINE TABLE IF NOT EXISTS trial SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS lineage ON trial TYPE uuid;
DEFINE FIELD IF NOT EXISTS specification ON trial TYPE record<specification>;
DEFINE FIELD IF NOT EXISTS links ON trial TYPE array<{
    trial: record<trial>, specification: record<specification>, relation: string, reason: string
}>;

DEFINE TABLE IF NOT EXISTS attempt SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS trial ON attempt TYPE record<trial>;
DEFINE FIELD IF NOT EXISTS run ON attempt TYPE uuid;
DEFINE FIELD IF NOT EXISTS predecessor ON attempt TYPE option<record<attempt>>;
DEFINE FIELD IF NOT EXISTS started_at ON attempt TYPE datetime;
DEFINE FIELD IF NOT EXISTS finished_at ON attempt TYPE datetime;
DEFINE FIELD IF NOT EXISTS outcome ON attempt TYPE {
    status: string, detail: string, result_artifacts: array<string>, limitations: array<string>
};
DEFINE FIELD IF NOT EXISTS produced ON attempt TYPE array<record<object>>;

DEFINE TABLE IF NOT EXISTS exposure SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS actor ON exposure TYPE string;
DEFINE FIELD IF NOT EXISTS at ON exposure TYPE datetime;
DEFINE FIELD IF NOT EXISTS material ON exposure TYPE record<object>;
DEFINE FIELD IF NOT EXISTS dataset_id ON exposure TYPE string;
DEFINE FIELD IF NOT EXISTS interval_start ON exposure TYPE datetime;
DEFINE FIELD IF NOT EXISTS interval_end ON exposure TYPE datetime;
DEFINE FIELD IF NOT EXISTS kind ON exposure TYPE string;
DEFINE FIELD IF NOT EXISTS reason ON exposure TYPE string;

DEFINE TABLE IF NOT EXISTS package SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS manifest ON package TYPE record<object>;
DEFINE FIELD IF NOT EXISTS trial ON package TYPE option<record<trial>>;
DEFINE FIELD IF NOT EXISTS trials ON package TYPE array<record<trial>>;
DEFINE FIELD IF NOT EXISTS attempts ON package TYPE array<record<attempt>>;
DEFINE FIELD IF NOT EXISTS exposures ON package TYPE array<record<exposure>>;
DEFINE FIELD IF NOT EXISTS artifacts ON package TYPE array<{
    name: string, media_type: string, role: string, object: record<object>
}>;
DEFINE FIELD IF NOT EXISTS disclosure_coverage ON package TYPE string;
DEFINE FIELD IF NOT EXISTS scope ON package TYPE string;
DEFINE FIELD IF NOT EXISTS confirmation ON package TYPE string;

DEFINE TABLE IF NOT EXISTS invalidation SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS evidence_id ON invalidation TYPE string;
DEFINE FIELD IF NOT EXISTS kind ON invalidation TYPE string;
DEFINE FIELD IF NOT EXISTS actor ON invalidation TYPE string;
DEFINE FIELD IF NOT EXISTS at ON invalidation TYPE datetime;
DEFINE FIELD IF NOT EXISTS reason ON invalidation TYPE string;

DEFINE TABLE IF NOT EXISTS review_event SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS event ON review_event TYPE object FLEXIBLE;

DEFINE TABLE IF NOT EXISTS container_run SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS run ON container_run TYPE uuid;
DEFINE FIELD IF NOT EXISTS entry ON container_run TYPE bytes;
"""

PACKAGE_TRIAL_MIGRATION = """
IF (INFO FOR TABLE package).fields.trial CONTAINS 'TYPE record<trial>' {
    ALTER FIELD trial ON package TYPE option<record<trial>>
};
"""


class RegistryError(ValueError):
    """Report an unknown package, a conflicting immutable identity, or a corrupt object."""


def _rows(package: VerifiedPackage, contents: dict[str, bytes]) -> list[dict]:
    record = package.record
    digests = {artifact.name: artifact.digest for artifact in package.artifact_records}
    rows = [
        {"id": RecordID("object", digest), "size": len(data)} for digest, data in contents.items()
    ]
    for trial in record.trials:
        spec = trial.specification
        environment = spec.environment
        rows.append(
            {
                "id": RecordID("specification", spec.spec_id),
                "question": spec.question,
                "signal_profile": spec.signal_profile,
                "evaluation_profile": spec.evaluation_profile,
                "data_version": spec.data_version,
                "implementation_version": spec.implementation_version,
                "inputs": [RecordID("object", digest) for digest in spec.input_digests],
                "data_contract": RecordID("object", spec.data_contract_digest),
                "implementation": RecordID("object", spec.implementation_digest),
                "environment": {
                    "implementation": environment.implementation,
                    "python_version": environment.python_version,
                    "abi": environment.abi,
                    "platform": environment.platform,
                    "uv_version": environment.uv_version,
                    "lock": RecordID("object", environment.lock_digest),
                },
            }
        )
        rows.append(
            {
                "id": RecordID("trial", trial.trial_id),
                "lineage": trial.lineage_id,
                "specification": RecordID("specification", spec.spec_id),
                "links": [
                    {
                        "trial": RecordID("trial", link.trial_id),
                        "specification": RecordID("specification", link.spec_id),
                        "relation": link.relation,
                        "reason": link.reason,
                    }
                    for link in trial.links
                ],
            }
        )
        for attempt in trial.attempts:
            outcome = attempt.outcome
            row = {
                "id": RecordID("attempt", attempt.id),
                "trial": RecordID("trial", trial.trial_id),
                "run": attempt.run_id,
                "started_at": attempt.started_at,
                "finished_at": attempt.finished_at,
                "outcome": {
                    "status": outcome.status,
                    "detail": outcome.detail,
                    "result_artifacts": list(outcome.result_artifacts),
                    "limitations": list(outcome.limitations),
                },
                "produced": [
                    RecordID("object", digests[name]) for name in outcome.result_artifacts
                ],
            }
            # Python None writes NONE and the field reads back absent, so a first attempt omits it.
            if attempt.predecessor is not None:
                row["predecessor"] = RecordID("attempt", attempt.predecessor)
            rows.append(row)
    rows.extend(
        {
            "id": RecordID("exposure", exposure.id),
            "actor": exposure.actor,
            "at": exposure.at,
            "material": RecordID("object", exposure.material_digest),
            "dataset_id": exposure.dataset_id,
            "interval_start": exposure.interval_start,
            "interval_end": exposure.interval_end,
            "kind": exposure.kind,
            "reason": exposure.reason,
        }
        for exposure in record.exposures
    )
    row = {
        "id": RecordID("package", package.digest),
        "manifest": RecordID("object", package.digest),
        "trials": [RecordID("trial", trial.trial_id) for trial in record.trials],
        "attempts": [
            RecordID("attempt", attempt.id) for trial in record.trials for attempt in trial.attempts
        ],
        "exposures": [RecordID("exposure", exposure.id) for exposure in record.exposures],
        "artifacts": [
            {
                "name": artifact.name,
                "media_type": artifact.media_type,
                "role": artifact.role,
                "object": RecordID("object", artifact.digest),
            }
            for artifact in package.artifact_records
        ],
        "disclosure_coverage": record.disclosure_coverage,
        "scope": record.scope,
        "confirmation": record.confirmation,
    }
    if record.trial_id is not None:
        row["trial"] = RecordID("trial", record.trial_id)
    rows.append(row)
    return rows


def _by_id(rows: list[dict]) -> dict[tuple[str, object], dict]:
    return {(row["id"].table_name, row["id"].id): row for row in rows}


def _read_object(objects: Path, digest: str) -> bytes:
    try:
        content = (objects / digest).read_bytes()
    except OSError as error:
        raise RegistryError(f"Cannot read stored object {digest}: {error}") from error
    if hashlib.sha256(content).hexdigest() != digest:
        raise RegistryError(f"Stored object {digest} does not match its digest")
    return content


def _write_objects(objects: Path, directory: int, contents: dict[str, bytes]) -> None:
    try:
        for digest, content in contents.items():
            _write_object(objects, digest, content)
        os.fsync(directory)
    except OSError as error:
        raise RegistryError(f"Cannot write objects to {objects}: {error}") from error


def _write_object(objects: Path, digest: str, content: bytes) -> None:
    descriptor, staged = tempfile.mkstemp(prefix=".object-", dir=objects)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, objects / digest)
    except BaseException:
        os.unlink(staged)
        raise


@contextmanager
def _locked(objects: Path, operation: int) -> Iterator[int]:
    directory = os.open(objects, os.O_RDONLY)
    try:
        fcntl.flock(directory, operation)
        yield directory
    finally:
        os.close(directory)


def _transact[T](
    connection: BlockingWsSurrealConnection, action: str, body: Callable[[UUID], T]
) -> T:
    for _ in range(PUBLISH_ATTEMPTS):
        transaction = connection.begin()
        try:
            result = body(transaction)
        except BaseException:
            connection.cancel(transaction)
            raise
        # The server discards a transaction whose commit fails, so there is nothing to cancel.
        try:
            connection.commit(transaction)
            return result
        except QueryError as error:
            if error.details != {"kind": "TransactionConflict"}:
                raise
    raise RegistryError(f"{action} conflicted with concurrent writes {PUBLISH_ATTEMPTS} times")


def _publish(
    connection: BlockingWsSurrealConnection,
    rows: list[dict],
    label: str,
    write: Callable[[], None] = lambda: None,
) -> None:
    expected = _by_id(rows)

    def compare_and_insert(transaction: UUID) -> None:
        *_, found = store.query(
            connection,
            f"{SCHEMA}{PACKAGE_TRIAL_MIGRATION}SELECT * FROM $ids",
            {"ids": [row["id"] for row in rows]},
            txn=transaction,
        )
        stored = _by_id(found)
        for key, row in stored.items():
            if row != expected[key]:
                raise RegistryError(f"Existing record {row['id']} differs from {label}")
        write()
        store.query(
            connection,
            "INSERT $rows",
            {"rows": [row for key, row in expected.items() if key not in stored]},
            txn=transaction,
        )

    _transact(connection, f"Importing {label}", compare_and_insert)


def import_package(
    connection: BlockingWsSurrealConnection, objects: Path, package: VerifiedPackage
) -> str:
    """Store a verified package's original bytes and records under their original identities.

    Args:
        connection (BlockingWsSurrealConnection): Registry database connection.
        objects (Path): Content-addressed object directory, keyed by SHA-256 digest.
        package (VerifiedPackage): Opened package whose bytes are rechecked before any write.

    Returns:
        (str) The package digest, which identifies the registered package.

    Raises:
        PackageError: If the package bytes changed after opening. Nothing is written.
        RegistryError: If an existing record differs from this package, the object directory
            or an object cannot be written, or the commit conflicts with concurrent writes
            five times.
    """
    try:
        manifest = (package.path / "manifest.json").read_bytes()
    except OSError as error:
        raise PackageError(f"Cannot read package manifest: {error}") from error
    if hashlib.sha256(manifest).hexdigest() != package.digest:
        raise PackageError("Package manifest changed after opening")
    contents = {package.digest: manifest} | {
        artifact.digest: item.data
        for artifact, item in zip(package.artifact_records, package.artifacts(), strict=True)
    }
    try:
        objects.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise RegistryError(f"Cannot write objects to {objects}: {error}") from error
    # Until its rows commit, this import's objects look unreferenced to a cleanup.
    with _locked(objects, fcntl.LOCK_SH) as directory:
        _publish(
            connection,
            _rows(package, contents),
            f"package {package.digest}",
            lambda: _write_objects(objects, directory, contents),
        )
    return package.digest


def import_invalidation(
    connection: BlockingWsSurrealConnection, invalidation: Invalidation
) -> None:
    """Store an invalidation under the identity that its suspensions carry in a ledger.

    Its evidence ID, kind, actor, and reason identify it, so another actor or reason for the
    same evidence is another invalidation, and importing the same one again writes nothing.

    Args:
        connection (BlockingWsSurrealConnection): Registry database connection.
        invalidation (Invalidation): Invalidation in force. Its time is stored in UTC.

    Raises:
        InvalidationError: If the invalidation is malformed. Nothing is written.
        RegistryError: If a stored invalidation with this identity has another time, or the
            commit conflicts with concurrent writes five times.
    """
    row = asdict(parse_invalidation(invalidation))
    identity = {name: value for name, value in row.items() if name != "at"}
    key = hashlib.sha256(canonical_bytes(identity)).hexdigest()
    _publish(
        connection,
        [{"id": RecordID("invalidation", key), **row}],
        f"an invalidation of {row['evidence_id']}",
    )


def export_package(
    connection: BlockingWsSurrealConnection, objects: Path, digest: str, destination: Path
) -> VerifiedPackage:
    """Rebuild a registered package directory from its original stored bytes.

    Args:
        connection (BlockingWsSurrealConnection): Registry database connection.
        objects (Path): Content-addressed object directory, keyed by SHA-256 digest.
        digest (str): Registered package digest.
        destination (Path): New package directory, or an existing copy of the same package.

    Returns:
        (VerifiedPackage) The exported package, verified like any other package.

    Raises:
        RegistryError: If the package is unknown, a stored object is corrupt, or the
            destination holds anything else or cannot be written. A failed export creates no
            destination.
    """
    # A single record from an undefined table raises; an array of records returns no rows.
    [found] = store.query(
        connection, "SELECT artifacts FROM $ids", {"ids": [RecordID("package", digest)]}
    )
    if not found:
        raise RegistryError(f"Unknown package {digest}")
    try:
        if not (destination.exists() or destination.is_symlink()):
            manifest = _read_object(objects, digest)
            [package] = found
            artifacts = {
                key: _read_object(objects, key)
                for key in {artifact["object"].id for artifact in package["artifacts"]}
            }
            with tempfile.TemporaryDirectory(
                prefix=".package-", dir=destination.parent
            ) as directory:
                stage = Path(directory)
                (stage / "artifacts").mkdir()
                (stage / "manifest.json").write_bytes(manifest)
                (stage / "manifest.sha256").write_text(f"{digest}\n", encoding="ascii")
                for key, content in artifacts.items():
                    (stage / "artifacts" / key).write_bytes(content)
                try:
                    stage.rename(destination)
                except OSError as error:
                    if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                        raise
    except OSError as error:
        raise RegistryError(f"Cannot export package to {destination}: {error}") from error
    try:
        exported = open_package(destination)
    except PackageError as error:
        raise RegistryError(f"{destination} does not hold package {digest}: {error}") from error
    if exported.digest != digest:
        raise RegistryError(f"{destination} holds a different package")
    return exported


def import_ledger(connection: BlockingWsSurrealConnection, ledger: Ledger) -> None:
    """Store each event of a review log under its position in the log.

    Importing the same log again writes nothing, and a log that extends the stored one adds only
    its new events.

    Args:
        connection (BlockingWsSurrealConnection): Registry database connection.
        ledger (Ledger): Review log that extends the stored log or is a prefix of it.

    Raises:
        RegistryError: If a stored event differs from the ledger's event at its position, or the
            commit conflicts with concurrent writes five times. Nothing is written.
    """
    _publish(connection, _event_rows(ledger.events, 0), "the review log")


def load_ledger(connection: BlockingWsSurrealConnection, *, at: datetime) -> Ledger:
    """Store the suspensions that stored invalidations call for and return the stored review log.

    One transaction reads the stored events and invalidations and, from the package rows, the
    inputs of each registered package whose trials list invalidated evidence. It replays the
    invalidations through `invalidation.catch_up` and stores the suspensions it appends after the
    stored events, so loading again appends nothing.

    Args:
        connection (BlockingWsSurrealConnection): Registry database connection.
        at (datetime): Time of every appended suspension, no earlier than the last stored event.

    Returns:
        (Ledger) The stored log with the appended suspensions, empty when nothing is stored.

    Raises:
        RegistryError: If the stored positions are not numbered from 0 without gaps, or the
            commit conflicts with concurrent writes five times.
        ReviewError: If a stored event is malformed or illegal, or a suspension is due and at
            precedes the last stored event. Nothing is written.
    """

    def catch_up_stored(transaction: UUID) -> Ledger:
        *_, rows, invalidations = store.query(
            connection,
            f"{SCHEMA}SELECT * FROM review_event ORDER BY id;"
            " SELECT evidence_id, kind, actor, at, reason FROM invalidation",
            txn=transaction,
        )
        if [row["id"].id for row in rows] != list(range(len(rows))):
            raise RegistryError("Stored review events must be numbered from 0 without gaps")
        ledger = Ledger(tuple(parse_event(row["event"]) for row in rows))
        stored = [Invalidation(**row) for row in invalidations]
        [readers] = store.query(
            connection,
            "SELECT VALUE [record::id(id), array::flatten(trials.specification.inputs)]"
            " FROM package WHERE array::flatten(trials.specification.inputs) CONTAINSANY $ids",
            {"ids": [RecordID("object", item.evidence_id) for item in stored]},
            txn=transaction,
        )
        caught = catch_up_ledger(
            ledger,
            {digest: {link.id for link in inputs} for digest, inputs in readers},
            stored,
            at=at,
        )
        store.query(
            connection,
            "INSERT $rows",
            {"rows": _event_rows(caught.events, len(rows))},
            txn=transaction,
        )
        return caught

    return _transact(connection, "Loading the review log", catch_up_stored)


def _event_rows(events: tuple[Event, ...], start: int) -> list[dict]:
    return [
        {"id": RecordID("review_event", position), "event": event_value(event)}
        for position, event in enumerate(events[start:], start)
    ]


def import_run_log(connection: BlockingWsSurrealConnection, log: RunLog) -> None:
    """Store each entry of a worker run log under its position in the log.

    Importing the same log again writes nothing, and a log that extends the stored one adds only
    its new entries.

    Args:
        connection (BlockingWsSurrealConnection): Registry database connection.
        log (RunLog): Run log that extends the stored log or is a prefix of it.

    Raises:
        WorkerError: If the run log cannot be replayed. No table is defined.
        RegistryError: If a stored entry differs from the log's entry at its position, or the
            commit conflicts with concurrent writes five times. Nothing is written.
    """
    _publish(
        connection,
        [
            {
                "id": RecordID("container_run", position),
                "run": logged.run_id,
                "entry": logged.to_bytes(),
            }
            for position, logged in enumerate(log.load())
        ],
        "the run log",
    )


def load_run_log(connection: BlockingWsSurrealConnection) -> tuple[bytes, ...]:
    """Return the stored entries of the worker run log in order.

    Writing entry i as the file i.json of an empty directory rebuilds the imported log.

    Args:
        connection (BlockingWsSurrealConnection): Registry database connection.

    Returns:
        (tuple[bytes, ...]) The canonical JSON bytes of each stored entry, empty when nothing is
        stored.

    Raises:
        RegistryError: If the stored positions are not numbered from 0 without gaps.
    """
    *_, rows = store.query(connection, f"{SCHEMA}SELECT * FROM container_run ORDER BY id")
    if [row["id"].id for row in rows] != list(range(len(rows))):
        raise RegistryError("Stored container runs must be numbered from 0 without gaps")
    return tuple(row["entry"] for row in rows)


def remove_unreferenced_objects(
    connection: BlockingWsSurrealConnection, objects: Path
) -> list[str]:
    """Remove object files and interrupted writes that no registry row references.

    Args:
        connection (BlockingWsSurrealConnection): Registry database connection.
        objects (Path): Content-addressed object directory, keyed by SHA-256 digest.

    Returns:
        (list[str]) The removed file names, sorted. Names the registry never writes stay.

    Raises:
        RegistryError: If the object directory cannot be read or a file cannot be removed.
    """
    try:
        # Running imports hold a shared lock until their rows commit.
        with _locked(objects, fcntl.LOCK_EX):
            # An import interrupted before its first commit leaves objects but no table.
            *_, referenced = store.query(
                connection, f"{SCHEMA}SELECT VALUE record::id(id) FROM object"
            )
            written = {
                path.name
                for path in objects.iterdir()
                if path.is_file() and re.fullmatch(r"[0-9a-f]{64}|\.object-.+", path.name)
            }
            removed = sorted(written - set(referenced))
            for name in removed:
                (objects / name).unlink()
    except OSError as error:
        raise RegistryError(f"Cannot remove objects from {objects}: {error}") from error
    return removed
