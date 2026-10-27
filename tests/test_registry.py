from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from surrealdb import QueryError, RecordID

from quant_research import datasets, registry, store
from quant_research.access import INTERNAL_CODE, Provider, Restrictions
from quant_research.findings import (
    Cites,
    Claim,
    Declaration,
    External,
    Finding,
    Ledger,
    Lesson,
    Proposal,
    Review,
    ReviewError,
    Scope,
    Supersession,
    event_value,
    record_id,
)
from quant_research.invalidation import Invalidation, InvalidationError
from quant_research.packages import (
    ArtifactInput,
    EnvironmentRecord,
    ExecutionAttempt,
    ExposureEvent,
    LineageLink,
    Outcome,
    PackageRecord,
    Specification,
    TrialRecord,
    open_package,
    save_package,
)
from quant_research.records import PackageError, canonical_bytes
from quant_research.registry import (
    RegistryError,
    export_package,
    import_invalidation,
    import_ledger,
    import_package,
    import_run_log,
    load_ledger,
    load_run_log,
)
from quant_research.worker import Limits, Refused, RunLog, RunRecord, Signal, WorkerError

NAMESPACE = "registry"
TABLES = (
    "object",
    "specification",
    "trial",
    "attempt",
    "exposure",
    "package",
    "invalidation",
    "review_event",
    "container_run",
)
TIME = datetime(2026, 10, 21, 12, 0, 0, 123456, tzinfo=UTC)
INVALIDATION = Invalidation(
    hashlib.sha256(b"momentum backtest package").hexdigest(),
    "invalid_package",
    "auditor",
    TIME + timedelta(days=1),
    "Look-ahead prices",
)
SCOPE = Scope(
    "US large-cap equities",
    "one month",
    datetime(2010, 1, 1, tzinfo=UTC),
    datetime(2024, 12, 31, tzinfo=UTC),
)
RESULT = b"named result, no scientific interpretation\n"


class Interrupted(Exception):
    pass


@pytest.fixture
def db(server, request):
    with server.connect(NAMESPACE, request.node.name) as connection:
        yield connection


@pytest.fixture
def history():
    artifacts = tuple(
        ArtifactInput(name, data, "text/plain", role)
        for name, data, role in (
            ("input", b"synthetic input\n", "input"),
            ("contract", b"synthetic only, version 1\n", "data contract"),
            ("code", b"hand-authored evidence\n", "implementation"),
            ("lock", b"locked environment\n", "environment"),
            ("result", RESULT, "result"),
            ("result-copy", RESULT, "result"),
            ("failure", b"supplied execution failure\n", "failure detail"),
        )
    )
    digests = {item.name: sha256(item.data) for item in artifacts}
    spec = Specification(
        "Preserve a declared development result",
        "hand-authored/v1",
        "generic-result/v1",
        "synthetic/v1",
        "test-authoring/v1",
        (digests["input"],),
        digests["contract"],
        digests["code"],
        EnvironmentRecord(
            "cpython", "3.12.13", "cpython-312", "test-platform", "uv test", digests["lock"]
        ),
    )
    failed = ExecutionAttempt(
        uuid4(),
        uuid4(),
        None,
        TIME,
        TIME + timedelta(seconds=1),
        Outcome("failed", "Executor failed", ("failure",)),
    )
    retry = ExecutionAttempt(
        uuid4(),
        failed.run_id,
        failed.id,
        TIME + timedelta(seconds=2),
        TIME + timedelta(seconds=3),
        Outcome("successful", "Supplied result", ("result", "result-copy"), ("Synthetic only",)),
    )
    repeat = replace(
        retry,
        id=uuid4(),
        run_id=uuid4(),
        predecessor=None,
        started_at=TIME + timedelta(seconds=4),
        finished_at=TIME + timedelta(seconds=5),
    )
    fork = replace(
        retry,
        id=uuid4(),
        started_at=TIME + timedelta(seconds=6),
        finished_at=TIME + timedelta(seconds=7),
        outcome=Outcome("inconclusive", "A second retry of the same failure"),
    )
    first = TrialRecord(uuid4(), uuid4(), spec, (failed, retry, fork))
    second = TrialRecord(
        uuid4(),
        first.lineage_id,
        spec,
        (repeat,),
        (LineageLink(first.trial_id, spec.spec_id, "repeats", "Same specification again"),),
    )
    exposure = ExposureEvent(
        uuid4(),
        "researcher",
        TIME,
        digests["input"],
        "synthetic/v1",
        TIME,
        TIME + timedelta(days=1),
        "data",
        "Supplied prior data exposure",
    )
    record = PackageRecord(second.trial_id, (first, second), (exposure,), "declared_inputs")
    return record, artifacts


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def files(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


def snapshot(db) -> list:
    return store.query(db, "; ".join(f"SELECT * FROM {table} ORDER BY id" for table in TABLES))


def stored_objects(package) -> set[str]:
    return {package.digest} | {artifact.digest for artifact in package.artifact_records}


def test_round_trip_exports_original_bytes_under_original_identities(db, tmp_path, history, cli):
    record, artifacts = history
    original = save_package(tmp_path / "packages", record, artifacts)
    objects = tmp_path / "objects"
    assert import_package(db, objects, original) == original.digest
    exported = export_package(db, objects, original.digest, tmp_path / "exported")
    assert exported.digest == original.digest
    assert files(exported.path) == files(original.path)
    assert original.digest in cli(tmp_path, "verify", str(exported.path)).stdout
    assert export_package(db, objects, original.digest, exported.path).digest == original.digest
    first, second = record.trials
    failed, retry, fork = first.attempts
    trials, specifications, object_ids, retried, package = store.query(
        db,
        "SELECT VALUE record::id(id) FROM trial;"
        " SELECT VALUE record::id(id) FROM specification;"
        " SELECT VALUE record::id(id) FROM object;"
        " SELECT predecessor, produced, started_at FROM $retries;"
        " SELECT VALUE [trial, trials, attempts] FROM $package",
        {
            "retries": [RecordID("attempt", retry.id), RecordID("attempt", fork.id)],
            "package": RecordID("package", original.digest),
        },
    )
    assert set(trials) == {first.trial_id, second.trial_id}
    assert specifications == [first.specification.spec_id]
    assert set(object_ids) == {original.digest} | {sha256(item.data) for item in artifacts}
    assert retried == [
        {
            "predecessor": RecordID("attempt", failed.id),
            "produced": [RecordID("object", sha256(RESULT))] * 2,
            "started_at": TIME + timedelta(seconds=2),
        },
        {
            "predecessor": RecordID("attempt", failed.id),
            "produced": [],
            "started_at": TIME + timedelta(seconds=6),
        },
    ]
    assert package == [
        [
            RecordID("trial", second.trial_id),
            [RecordID("trial", first.trial_id), RecordID("trial", second.trial_id)],
            [RecordID("attempt", attempt.id) for attempt in (*first.attempts, *second.attempts)],
        ]
    ]


def test_importing_the_same_package_twice_changes_nothing(db, tmp_path, history):
    package = save_package(tmp_path / "packages", *history)
    objects = tmp_path / "objects"
    import_package(db, objects, package)
    before = snapshot(db), sorted(objects.iterdir())
    assert import_package(db, objects, package) == package.digest
    assert (snapshot(db), sorted(objects.iterdir())) == before


def commit_together(monkeypatch, connection, barrier, commits):
    original = connection.commit

    def commit(transaction):
        commits.append(connection)
        if commits.count(connection) == 1:
            barrier.wait(timeout=30)
        original(transaction)

    monkeypatch.setattr(connection, "commit", commit)


def test_concurrent_imports_of_one_package_retry_and_store_one_copy(
    server, db, request, tmp_path, history, monkeypatch
):
    package = save_package(tmp_path / "packages", *history)
    objects = tmp_path / "objects"
    barrier = threading.Barrier(2)
    commits = []
    with server.connect(NAMESPACE, request.node.name) as other:
        for connection in (db, other):
            commit_together(monkeypatch, connection, barrier, commits)
        with ThreadPoolExecutor(max_workers=2) as workers:
            digests = list(
                workers.map(
                    lambda connection: import_package(connection, objects, package), (db, other)
                )
            )
    assert digests == [package.digest, package.digest]
    assert len(commits) == 3
    assert {path.name for path in objects.iterdir()} == stored_objects(package)
    assert store.query(db, "SELECT VALUE record::id(id) FROM package") == [[package.digest]]
    exported = export_package(db, objects, package.digest, tmp_path / "exported")
    assert files(exported.path) == files(package.path)


def test_import_gives_up_after_five_commit_conflicts_without_rows(
    db, tmp_path, history, monkeypatch
):
    package = save_package(tmp_path / "packages", *history)
    commits = []

    def conflicting(transaction):
        commits.append(transaction)
        db.cancel(transaction)
        raise QueryError("Query", "conflict", details={"kind": "TransactionConflict"})

    monkeypatch.setattr(db, "commit", conflicting)
    with pytest.raises(RegistryError, match="5 times"):
        import_package(db, tmp_path / "objects", package)
    assert len(commits) == 5
    assert store.query(db, "INFO FOR DB")[0]["tables"] == {}


FIELD_BEFORE_D060 = "DEFINE FIELD trial ON package TYPE record<trial> PERMISSIONS FULL"
FIELD_AFTER_D060 = "DEFINE FIELD trial ON package TYPE none | record<trial> PERMISSIONS FULL"


def schema_before_d060() -> str:
    old, new = (
        f"DEFINE FIELD IF NOT EXISTS trial ON package TYPE {kind};"
        for kind in ("record<trial>", "option<record<trial>>")
    )
    assert registry.SCHEMA.count(new) == 1
    return registry.SCHEMA.replace(new, old)


def package_trial_field(db) -> str:
    return store.query(db, "INFO FOR TABLE package")[0]["fields"]["trial"]


@pytest.mark.parametrize("supplied", [True, False], ids=["trials", "no-trials"])
def test_a_package_without_a_selected_trial_round_trips_and_stores_no_trial(
    db, tmp_path, history, cli, supplied
):
    record, artifacts = history
    unselected = replace(record, trial_id=None, trials=record.trials if supplied else ())
    package = save_package(tmp_path / "packages", unselected, artifacts)
    objects = tmp_path / "objects"
    assert import_package(db, objects, package) == package.digest
    exported = export_package(db, objects, package.digest, tmp_path / "exported")
    assert files(exported.path) == files(package.path)
    assert package.digest in cli(tmp_path, "verify", str(exported.path)).stdout
    assert store.query(
        db, "SELECT VALUE trial FROM $package", {"package": RecordID("package", package.digest)}
    ) == [[None]]
    before = snapshot(db), sorted(objects.iterdir())
    assert import_package(db, objects, package) == package.digest
    assert (snapshot(db), sorted(objects.iterdir())) == before
    assert load_ledger(db, at=TIME).events == ()


def test_a_registry_created_before_d060_migrates_once_and_keeps_its_selected_package(
    db, tmp_path, history, monkeypatch
):
    record, artifacts = history
    selected = save_package(tmp_path / "packages", record, artifacts)
    unselected = save_package(tmp_path / "packages", replace(record, trial_id=None), artifacts)
    objects = tmp_path / "objects"
    monkeypatch.setattr(registry, "SCHEMA", schema_before_d060())
    monkeypatch.setattr(registry, "PACKAGE_TRIAL_MIGRATION", "")
    import_package(db, objects, selected)
    assert package_trial_field(db) == FIELD_BEFORE_D060
    monkeypatch.undo()
    import_package(db, objects, unselected)
    assert package_trial_field(db) == FIELD_AFTER_D060
    before = snapshot(db), sorted(objects.iterdir())
    assert import_package(db, objects, selected) == selected.digest
    assert (snapshot(db), sorted(objects.iterdir())) == before
    for package in (selected, unselected):
        exported = export_package(db, objects, package.digest, tmp_path / package.digest)
        assert files(exported.path) == files(package.path)
        assert open_package(exported.path).record == package.record


def test_concurrent_first_imports_into_a_registry_created_before_d060_both_migrate(
    server, db, request, tmp_path, history, monkeypatch
):
    record, artifacts = history
    packages = [
        save_package(
            tmp_path / "packages", replace(record, trial_id=None, trials=trials), artifacts
        )
        for trials in (record.trials, ())
    ]
    objects = tmp_path / "objects"
    store.query(db, schema_before_d060())
    assert package_trial_field(db) == FIELD_BEFORE_D060
    barrier = threading.Barrier(2)
    commits = []
    with server.connect(NAMESPACE, request.node.name) as other:
        for connection in (db, other):
            commit_together(monkeypatch, connection, barrier, commits)
        with ThreadPoolExecutor(max_workers=2) as workers:
            digests = list(
                workers.map(
                    lambda connection, package: import_package(connection, objects, package),
                    (db, other),
                    packages,
                )
            )
    assert digests == [package.digest for package in packages]
    assert package_trial_field(db) == FIELD_AFTER_D060
    for package in packages:
        exported = export_package(db, objects, package.digest, tmp_path / package.digest)
        assert files(exported.path) == files(package.path)
        assert open_package(exported.path).record == package.record


def test_packages_sharing_an_artifact_store_its_bytes_once(db, tmp_path, history):
    record, artifacts = history
    exposure = replace(record.exposures[0], id=uuid4(), reason="Another supplied exposure")
    packages = (
        save_package(tmp_path / "packages", record, artifacts),
        save_package(
            tmp_path / "packages",
            replace(record, exposures=(*record.exposures, exposure)),
            artifacts,
        ),
    )
    objects = tmp_path / "objects"
    for package in packages:
        import_package(db, objects, package)
    stored = stored_objects(packages[0]) | stored_objects(packages[1])
    assert len(stored) == len({item.data for item in artifacts}) + 2
    assert {path.name for path in objects.iterdir()} == stored
    [rows] = store.query(db, "SELECT VALUE record::id(id) FROM object")
    assert sorted(rows) == sorted(stored)
    for package in packages:
        exported = export_package(db, objects, package.digest, tmp_path / package.digest)
        assert files(exported.path) == files(package.path)


def append_space(path: Path) -> None:
    path.write_bytes(path.read_bytes() + b" ")


@pytest.mark.parametrize(
    "tamper",
    [
        lambda path: append_space(next((path / "artifacts").iterdir())),
        lambda path: append_space(path / "manifest.json"),
        shutil.rmtree,
    ],
    ids=["artifact", "manifest", "removed"],
)
def test_tampered_package_is_refused_before_anything_is_written(db, tmp_path, history, tamper):
    package = save_package(tmp_path / "packages", *history)
    tamper(package.path)
    with pytest.raises(PackageError):
        import_package(db, tmp_path / "objects", package)
    assert not (tmp_path / "objects").exists()
    assert store.query(db, "INFO FOR DB")[0]["tables"] == {}


@pytest.mark.parametrize("reused", ["trial", "attempt"])
def test_reused_identity_with_different_content_is_refused_without_writes(
    db, tmp_path, history, reused
):
    record, artifacts = history
    objects = tmp_path / "objects"
    import_package(db, objects, save_package(tmp_path / "packages", record, artifacts))
    before = snapshot(db), sorted(objects.iterdir())
    first = record.trials[0]
    if reused == "trial":
        changed = replace(first, specification=replace(first.specification, question="Changed"))
        identity = first.trial_id
    else:
        attempt = replace(first.attempts[0], outcome=Outcome("null", "A different outcome"))
        changed = replace(first, trial_id=uuid4(), lineage_id=uuid4(), attempts=(attempt,))
        identity = attempt.id
    notes = ArtifactInput("notes", b"only in the conflicting package\n", "text/plain", "notes")
    conflicting = save_package(
        tmp_path / "packages", PackageRecord(changed.trial_id, (changed,)), (*artifacts, notes)
    )
    with pytest.raises(RegistryError, match=str(identity)):
        import_package(db, objects, conflicting)
    assert (snapshot(db), sorted(objects.iterdir())) == before


def fail_second_rename(monkeypatch, connection):
    rename = os.replace
    targets = []

    def interrupted(source, target):
        targets.append(target)
        if len(targets) > 1:
            raise Interrupted
        rename(source, target)

    monkeypatch.setattr(registry.os, "replace", interrupted)


def fail_commit(monkeypatch, connection):
    def interrupted(transaction):
        raise Interrupted

    monkeypatch.setattr(connection, "commit", interrupted)


@pytest.mark.parametrize("interrupt", [fail_second_rename, fail_commit])
def test_interrupted_import_leaves_only_unreferenced_objects_and_a_reopened_retry_completes(
    server, db, request, tmp_path, history, monkeypatch, interrupt
):
    package = save_package(tmp_path / "packages", *history)
    objects = tmp_path / "objects"
    interrupt(monkeypatch, db)
    with pytest.raises(Interrupted):
        import_package(db, objects, package)
    identities = [RecordID("package", package.digest)] + [
        RecordID("trial", trial.trial_id) for trial in package.record.trials
    ]
    assert store.query(db, "SELECT * FROM $ids", {"ids": identities}) == [[]]
    written = sorted(objects.iterdir())
    assert set() < {path.name for path in written} <= stored_objects(package)
    # A power loss can cut an unflushed object short.
    written[0].write_bytes(written[0].read_bytes()[:-1])
    monkeypatch.undo()
    with server.connect(NAMESPACE, request.node.name) as reopened:
        assert import_package(reopened, objects, package) == package.digest
        exported = export_package(reopened, objects, package.digest, tmp_path / "exported")
    assert files(exported.path) == files(package.path)


def test_import_refuses_an_object_path_that_is_a_directory_without_rows(db, tmp_path, history):
    package = save_package(tmp_path / "packages", *history)
    objects = tmp_path / "objects"
    blocked = objects / package.artifact_records[0].digest
    blocked.mkdir(parents=True)
    with pytest.raises(RegistryError, match=blocked.name):
        import_package(db, objects, package)
    assert store.query(db, "INFO FOR DB")[0]["tables"] == {}
    assert not [path for path in objects.iterdir() if path.name.startswith(".object-")]


def test_import_refuses_a_store_path_that_is_a_file_without_rows(db, tmp_path, history):
    package = save_package(tmp_path / "packages", *history)
    objects = tmp_path / "objects"
    objects.write_bytes(b"not an object store\n")
    with pytest.raises(RegistryError):
        import_package(db, objects, package)
    assert objects.read_bytes() == b"not an object store\n"
    assert store.query(db, "INFO FOR DB")[0]["tables"] == {}


def test_import_flushes_each_object_and_the_store_directory_before_commit(
    db, tmp_path, history, monkeypatch
):
    package = save_package(tmp_path / "packages", *history)
    objects = tmp_path / "objects"
    fsync, commit = os.fsync, db.commit
    flushed, committed = set(), []

    def recorded_fsync(descriptor):
        fsync(descriptor)
        flushed.add(os.fstat(descriptor).st_ino)

    def recorded_commit(transaction):
        committed.append(set(flushed))
        commit(transaction)

    monkeypatch.setattr(registry.os, "fsync", recorded_fsync)
    monkeypatch.setattr(db, "commit", recorded_commit)
    import_package(db, objects, package)
    [at_commit] = committed
    assert {path.stat().st_ino for path in (objects, *objects.iterdir())} <= at_commit


def test_reimport_repairs_a_corrupt_object_for_every_package_that_shares_it(db, tmp_path, history):
    record, artifacts = history
    notes = ArtifactInput("notes", b"shared by two packages\n", "text/plain", "notes")
    exposure = replace(record.exposures[0], id=uuid4(), reason="Another supplied exposure")
    root = tmp_path / "packages"
    sharing = (
        save_package(root, record, (*artifacts, notes)),
        save_package(
            root, replace(record, exposures=(*record.exposures, exposure)), (*artifacts, notes)
        ),
    )
    unaffected = save_package(root, record, artifacts)
    objects = tmp_path / "objects"
    for package in (*sharing, unaffected):
        import_package(db, objects, package)
    corrupt = sha256(notes.data)
    (objects / corrupt).write_bytes(b"corrupted\n")
    before = snapshot(db)
    exported = export_package(db, objects, unaffected.digest, tmp_path / "unaffected")
    assert files(exported.path) == files(unaffected.path)
    for package in sharing:
        with pytest.raises(RegistryError, match=corrupt):
            export_package(db, objects, package.digest, tmp_path / package.digest)
    assert import_package(db, objects, sharing[1]) == sharing[1].digest
    assert snapshot(db) == before
    for package in sharing:
        exported = export_package(db, objects, package.digest, tmp_path / package.digest)
        assert files(exported.path) == files(package.path)


def test_export_refuses_unknown_packages_and_corrupt_objects_without_a_destination(
    db, tmp_path, history
):
    package = save_package(tmp_path / "packages", *history)
    objects = tmp_path / "objects"
    destination = tmp_path / "exported"
    with pytest.raises(RegistryError):
        export_package(db, objects, package.digest, destination)
    import_package(db, objects, package)
    with pytest.raises(RegistryError):
        export_package(db, objects, "0" * 64, destination)
    (objects / package.artifact_records[0].digest).write_bytes(b"corrupted\n")
    with pytest.raises(RegistryError):
        export_package(db, objects, package.digest, destination)
    (objects / package.artifact_records[0].digest).unlink()
    with pytest.raises(RegistryError):
        export_package(db, objects, package.digest, destination)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["objects", "packages"]


@pytest.mark.parametrize(
    "occupy",
    [
        lambda path: path.mkdir(),
        lambda path: path.write_bytes(b"not a package\n"),
        lambda path: path.symlink_to(path.with_name("missing")),
        lambda path: path.parent.rmdir(),
        lambda path: path.parent.chmod(0o600),
    ],
    ids=["empty-directory", "file", "dangling-symlink", "missing-parent", "unsearchable-parent"],
)
def test_export_refuses_a_destination_that_is_not_this_package(db, tmp_path, history, occupy):
    package = save_package(tmp_path / "packages", *history)
    objects = tmp_path / "objects"
    import_package(db, objects, package)
    exports = tmp_path / "exports"
    exports.mkdir()
    occupy(exports / "exported")
    before = files(exports), sorted(exports.rglob("*"))
    with pytest.raises(RegistryError):
        export_package(db, objects, package.digest, exports / "exported")
    assert (files(exports), sorted(exports.rglob("*"))) == before


def test_export_refuses_a_destination_holding_another_registered_package(db, tmp_path, history):
    record, artifacts = history
    exposure = replace(record.exposures[0], id=uuid4(), reason="Another supplied exposure")
    root = tmp_path / "packages"
    package = save_package(root, record, artifacts)
    extended = save_package(
        root, replace(record, exposures=(*record.exposures, exposure)), artifacts
    )
    objects = tmp_path / "objects"
    for item in (package, extended):
        import_package(db, objects, item)
    destination = tmp_path / "exported"
    export_package(db, objects, extended.digest, destination)
    before = files(destination)
    with pytest.raises(RegistryError):
        export_package(db, objects, package.digest, destination)
    assert files(destination) == before


def test_exports_racing_to_one_new_destination_both_return_the_package(
    server, db, request, tmp_path, history, monkeypatch
):
    package = save_package(tmp_path / "packages", *history)
    objects = tmp_path / "objects"
    import_package(db, objects, package)
    barrier = threading.Barrier(2)
    rename = Path.rename

    def together(stage, target):
        barrier.wait(timeout=30)
        return rename(stage, target)

    monkeypatch.setattr(Path, "rename", together)
    destination = tmp_path / "exported"
    with server.connect(NAMESPACE, request.node.name) as other:
        with ThreadPoolExecutor(max_workers=2) as workers:
            exported = list(
                workers.map(
                    lambda connection: export_package(
                        connection, objects, package.digest, destination
                    ),
                    (db, other),
                )
            )
    assert [item.digest for item in exported] == [package.digest, package.digest]
    assert files(destination) == files(package.path)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["exported", "objects", "packages"]


def test_export_refuses_a_directory_that_appears_before_its_rename(
    db, tmp_path, history, monkeypatch
):
    package = save_package(tmp_path / "packages", *history)
    objects = tmp_path / "objects"
    import_package(db, objects, package)
    rename = Path.rename

    def occupied_first(stage, target):
        (target / "other").mkdir(parents=True)
        return rename(stage, target)

    monkeypatch.setattr(Path, "rename", occupied_first)
    with pytest.raises(RegistryError):
        export_package(db, objects, package.digest, tmp_path / "exported")
    assert [path.name for path in (tmp_path / "exported").rglob("*")] == ["other"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["exported", "objects", "packages"]


def test_cleanup_removes_objects_and_interrupted_writes_that_no_row_references(
    server, db, request, tmp_path, history, monkeypatch
):
    package = save_package(tmp_path / "packages", *history)
    objects = tmp_path / "objects"
    fail_commit(monkeypatch, db)
    with pytest.raises(Interrupted):
        import_package(db, objects, package)
    monkeypatch.undo()
    (objects / ".object-killed").write_bytes(b"cut short by a kill during rename")
    (objects / "notes.txt").write_bytes(b"not written by the registry\n")
    (objects / ("f" * 64)).mkdir()
    with server.connect(NAMESPACE, request.node.name) as reopened:
        assert registry.remove_unreferenced_objects(reopened, objects) == sorted(
            {*stored_objects(package), ".object-killed"}
        )
        import_package(reopened, objects, package)
        (objects / ".object-later").write_bytes(b"left by a later kill")
        assert registry.remove_unreferenced_objects(reopened, objects) == [".object-later"]
        exported = export_package(reopened, objects, package.digest, tmp_path / "exported")
    assert {path.name for path in objects.iterdir()} == stored_objects(package) | {
        "notes.txt",
        "f" * 64,
    }
    assert files(exported.path) == files(package.path)


def test_cleanup_waits_for_a_running_import_and_keeps_its_objects(
    server, db, request, tmp_path, history, monkeypatch
):
    package = save_package(tmp_path / "packages", *history)
    objects = tmp_path / "objects"
    commit = db.commit
    cleanups = []
    with (
        server.connect(NAMESPACE, request.node.name) as other,
        ThreadPoolExecutor(max_workers=1) as worker,
    ):

        def commit_after_cleanup(transaction):
            cleanups.append(worker.submit(registry.remove_unreferenced_objects, other, objects))
            # An unlocked cleanup finishes within this wait and removes the uncommitted objects.
            wait(cleanups, timeout=1)
            commit(transaction)

        monkeypatch.setattr(db, "commit", commit_after_cleanup)
        import_package(db, objects, package)
        assert [cleanup.result(timeout=30) for cleanup in cleanups] == [[]]
    assert {path.name for path in objects.iterdir()} == stored_objects(package)
    exported = export_package(db, objects, package.digest, tmp_path / "exported")
    assert files(exported.path) == files(package.path)


def stored_invalidations(db) -> list[Invalidation]:
    [rows] = store.query(db, "SELECT evidence_id, kind, actor, at, reason FROM invalidation")
    return [Invalidation(**row) for row in rows]


@pytest.mark.parametrize(
    "change",
    [{"actor": "second auditor"}, {"reason": "Survivorship bias"}],
    ids=["actor", "reason"],
)
def test_importing_an_invalidation_twice_changes_nothing_and_another_cause_adds_a_row(db, change):
    import_invalidation(db, INVALIDATION)
    before = snapshot(db)
    import_invalidation(db, INVALIDATION)
    assert snapshot(db) == before
    other = replace(INVALIDATION, **change)
    import_invalidation(db, other)
    stored = stored_invalidations(db)
    assert len(stored) == 2
    assert set(stored) == {INVALIDATION, other}


def test_an_invalidation_identity_with_another_time_is_refused_without_writes(db):
    import_invalidation(db, INVALIDATION)
    before = snapshot(db)
    with pytest.raises(RegistryError):
        import_invalidation(db, replace(INVALIDATION, at=INVALIDATION.at + timedelta(hours=1)))
    assert snapshot(db) == before


@pytest.mark.parametrize(
    "change",
    [{"at": TIME.replace(tzinfo=None)}, {"evidence_id": "artifact:momentum-backtest"}],
    ids=["naive-time", "not-digest"],
)
def test_malformed_invalidation_is_refused_before_any_table(db, change):
    with pytest.raises(InvalidationError):
        import_invalidation(db, replace(INVALIDATION, **change))
    assert store.query(db, "INFO FOR DB")[0]["tables"] == {}


def test_concurrent_imports_of_one_invalidation_retry_and_store_one_row(
    server, db, request, monkeypatch
):
    barrier = threading.Barrier(2)
    commits = []
    with server.connect(NAMESPACE, request.node.name) as other:
        for connection in (db, other):
            commit_together(monkeypatch, connection, barrier, commits)
        with ThreadPoolExecutor(max_workers=2) as workers:
            results = list(
                workers.map(
                    lambda connection: import_invalidation(connection, INVALIDATION), (db, other)
                )
            )
    assert results == [None, None]
    assert len(commits) == 3
    assert stored_invalidations(db) == [INVALIDATION]


def supports(*items) -> tuple:
    return tuple(
        External(item, "supports") if isinstance(item, str) else Cites(record_id(item), "supports")
        for item in items
    )


def finding(statement: str, *support) -> Finding:
    return Finding(
        statement, "walk-forward/v1", "supports_idea", "researcher", SCOPE, supports(*support)
    )


def lesson(statement: str, *support) -> Lesson:
    return Lesson(statement, "agent", SCOPE, supports(*support))


def accepted(at: datetime, *records) -> list:
    events = []
    for record in records:
        actions = ("review", "admit") if isinstance(record, Lesson) else ("review",)
        events.append(Proposal(record, at))
        events.extend(
            Review(record_id(record), action, "reviewer", at, "Checked") for action in actions
        )
    return events


def test_loading_stores_the_suspensions_that_stored_invalidations_call_for(
    server, db, request, tmp_path, history
):
    record, artifacts = history
    first = record.trials[0]
    other_input = ArtifactInput("other-input", b"another vintage\n", "text/plain", "input")
    unread = replace(
        first,
        trial_id=uuid4(),
        lineage_id=uuid4(),
        specification=replace(first.specification, input_digests=(sha256(other_input.data),)),
        attempts=(replace(first.attempts[0], id=uuid4()),),
    )
    root = tmp_path / "packages"
    reader = save_package(root, record, artifacts)
    unrelated = save_package(
        root, PackageRecord(unread.trial_id, (unread,)), (*artifacts, other_input)
    )
    objects = tmp_path / "objects"
    for package in (reader, unrelated):
        import_package(db, objects, package)
    (vintage,) = first.specification.input_digests
    source = sha256(b"momentum survey source generation")
    corrected = Invalidation(
        vintage, "corrected_vintage", "auditor", TIME + timedelta(days=1), "Restated closes"
    )
    retracted = Invalidation(
        source, "retracted_source", "auditor", TIME + timedelta(days=1, hours=1), "Retracted"
    )
    for item in (retracted, corrected):
        import_invalidation(db, item)
    result = finding("Momentum earns a premium on the corrected vintage", reader.digest)
    unaffected = finding("Momentum earns a premium on another vintage", unrelated.digest)
    hedged = lesson("Momentum holds on either vintage", result, unaffected)
    rule = lesson("Momentum holds on vendor prices", result)
    sourced = Claim("Momentum profits reverse", "extracted", "agent", SCOPE, supports(source))
    late = finding("Momentum earns a premium in the rerun", reader.digest)
    ledger = Ledger(
        (
            *accepted(TIME, result, unaffected, hedged, rule, sourced),
            *accepted(TIME + timedelta(days=2), late),
        )
    )
    import_ledger(db, ledger)
    now = TIME + timedelta(days=3)
    with server.connect(NAMESPACE, request.node.name) as reopened:
        caught = load_ledger(reopened, at=now)
        stored = snapshot(reopened)
        again = load_ledger(reopened, at=now + timedelta(days=1))
        assert snapshot(reopened) == stored
    corrected_reason = f"corrected_vintage {vintage}: Restated closes"
    assert caught.events[: len(ledger.events)] == ledger.events
    assert caught.events[len(ledger.events) :] == (
        Review(record_id(result), "suspend", "auditor", now, corrected_reason),
        Review(record_id(rule), "suspend", "auditor", now, corrected_reason),
        Review(record_id(late), "suspend", "auditor", now, corrected_reason),
        Review(
            record_id(sourced), "suspend", "auditor", now, f"retracted_source {source}: Retracted"
        ),
    )
    assert stored_log(db) == logged(caught)
    assert caught.eligible_lessons() == (record_id(hedged),)
    assert again.events == caught.events


def cite_corrected_vintage(
    db, reader, trial: TrialRecord, now: datetime
) -> tuple[Invalidation, Ledger, Review]:
    (vintage,) = trial.specification.input_digests
    corrected = Invalidation(
        vintage, "corrected_vintage", "auditor", TIME + timedelta(days=1), "Restated closes"
    )
    result = finding("Momentum earns a premium on the corrected vintage", reader.digest)
    reason = f"corrected_vintage {vintage}: Restated closes"
    ledger = Ledger(tuple(accepted(TIME, result)))
    import_ledger(db, ledger)
    return corrected, ledger, Review(record_id(result), "suspend", "auditor", now, reason)


@pytest.mark.parametrize("corrupt", ["result", "manifest"])
def test_loading_suspends_through_a_reader_whose_stored_object_is_corrupt(
    db, tmp_path, history, corrupt
):
    record, artifacts = history
    reader = save_package(tmp_path / "packages", record, artifacts)
    objects = tmp_path / "objects"
    import_package(db, objects, reader)
    now = TIME + timedelta(days=3)
    corrected, ledger, suspension = cite_corrected_vintage(db, reader, record.trials[0], now)
    import_invalidation(db, corrected)
    digest = {"result": sha256(RESULT), "manifest": reader.digest}[corrupt]
    (objects / digest).write_bytes(b"corrupted\n")
    caught = load_ledger(db, at=now)
    assert caught.events[len(ledger.events) :] == (suspension,)
    assert stored_log(db) == logged(caught)
    with pytest.raises(RegistryError, match=digest):
        export_package(db, objects, reader.digest, tmp_path / "exported")


def test_loading_withdraws_a_package_when_any_of_its_trials_reads_stored_evidence(
    db, tmp_path, history
):
    record, artifacts = history
    first = record.trials[0]
    other_input = ArtifactInput("other-input", b"another vintage\n", "text/plain", "input")
    unread = replace(
        first,
        trial_id=uuid4(),
        lineage_id=uuid4(),
        specification=replace(first.specification, input_digests=(sha256(other_input.data),)),
        attempts=(replace(first.attempts[0], id=uuid4(), run_id=uuid4()),),
    )
    root = tmp_path / "packages"
    reader = save_package(
        root, PackageRecord(unread.trial_id, (unread, first)), (*artifacts, other_input)
    )
    unrelated = save_package(
        root, PackageRecord(unread.trial_id, (unread,)), (*artifacts, other_input)
    )
    objects = tmp_path / "objects"
    for package in (reader, unrelated):
        import_package(db, objects, package)
    now = TIME + timedelta(days=3)
    corrected, ledger, suspension = cite_corrected_vintage(db, reader, first, now)
    import_invalidation(db, corrected)
    unaffected = finding("Momentum earns a premium on another vintage", unrelated.digest)
    ledger = Ledger((*ledger.events, *accepted(TIME, unaffected)))
    import_ledger(db, ledger)
    caught = load_ledger(db, at=now)
    assert caught.events[len(ledger.events) :] == (suspension,)


def test_a_reader_package_imported_while_loading_waits_for_the_next_load(
    server, db, request, tmp_path, history, monkeypatch
):
    record, artifacts = history
    reader = save_package(tmp_path / "packages", record, artifacts)
    now = TIME + timedelta(days=3)
    corrected, ledger, suspension = cite_corrected_vintage(db, reader, record.trials[0], now)
    import_invalidation(db, corrected)
    begin = db.begin
    with server.connect(NAMESPACE, request.node.name) as other:

        def begin_then_import():
            transaction = begin()
            import_package(other, tmp_path / "objects", reader)
            return transaction

        monkeypatch.setattr(db, "begin", begin_then_import)
        loaded = load_ledger(db, at=now)
    monkeypatch.undo()
    assert loaded.events == ledger.events
    caught = load_ledger(db, at=now)
    assert caught.events[len(ledger.events) :] == (suspension,)


def test_an_invalidation_imported_while_loading_waits_for_the_next_load(
    server, db, request, tmp_path, history, monkeypatch
):
    record, artifacts = history
    reader = save_package(tmp_path / "packages", record, artifacts)
    import_package(db, tmp_path / "objects", reader)
    now = TIME + timedelta(days=3)
    corrected, ledger, suspension = cite_corrected_vintage(db, reader, record.trials[0], now)
    begin = db.begin
    with server.connect(NAMESPACE, request.node.name) as other:

        def begin_then_import():
            transaction = begin()
            import_invalidation(other, corrected)
            return transaction

        monkeypatch.setattr(db, "begin", begin_then_import)
        loaded = load_ledger(db, at=now)
    monkeypatch.undo()
    assert loaded.events == ledger.events
    caught = load_ledger(db, at=now)
    assert caught.events[len(ledger.events) :] == (suspension,)


def test_loading_without_stored_invalidations_appends_nothing(db):
    ledger = Ledger(tuple(accepted(TIME, finding("Momentum earns a premium", "0" * 64))))
    import_ledger(db, ledger)
    before = snapshot(db)
    loaded = load_ledger(db, at=TIME + timedelta(days=1))
    assert loaded.events == ledger.events
    assert snapshot(db) == before


def test_loading_an_empty_registry_returns_an_empty_ledger(db):
    assert load_ledger(db, at=TIME).events == ()


def review_log() -> Ledger:
    result = finding("Momentum earns a premium", INVALIDATION.evidence_id)
    rival = finding("Momentum earns a premium net of costs", sha256(b"rival backtest package"))
    rule, robust = lesson("Momentum holds", result), lesson("Momentum survives costs", rival)
    later = TIME + timedelta(hours=1)
    declared = Restrictions(
        available_at=TIME,
        labels=frozenset({INTERNAL_CODE}),
        approved_for=frozenset({Provider("local-model")}),
    )
    return Ledger(
        (
            Declaration(INVALIDATION.evidence_id, declared, "ingestion", TIME, "Store labels"),
            *accepted(TIME, result, rule, rival, robust),
            Review(record_id(rule), "suspend", "reviewer", later, "Rechecked\x00costs, é"),
            Supersession(record_id(result), record_id(rival), "reviewer", later, "Replaced"),
        )
    )


LOG = review_log()


def stored_log(db) -> list:
    [rows] = store.query(db, "SELECT * FROM review_event ORDER BY id")
    return [(row["id"].id, row["event"]) for row in rows]


def logged(ledger: Ledger) -> list:
    return list(enumerate(map(event_value, ledger.events)))


def answers(ledger: Ledger) -> tuple:
    identities = [record_id(event.record) for event in ledger.events if isinstance(event, Proposal)]
    return {identity: ledger.status(identity) for identity in identities}, ledger.eligible_lessons()


def test_an_imported_ledger_loads_from_a_reopened_connection_with_the_same_answers(
    server, db, request
):
    import_ledger(db, LOG)
    with server.connect(NAMESPACE, request.node.name) as reopened:
        loaded = load_ledger(reopened, at=TIME + timedelta(days=1))
    assert loaded.events == LOG.events
    assert loaded.head == LOG.head
    assert answers(loaded) == answers(LOG)
    assert stored_log(db) == logged(LOG)


def test_reimporting_a_ledger_writes_nothing_and_an_extension_adds_its_new_events(db):
    prefix = Ledger(LOG.events[:3])
    import_ledger(db, prefix)
    before = snapshot(db)
    import_ledger(db, prefix)
    assert snapshot(db) == before
    import_ledger(db, LOG)
    assert stored_log(db) == logged(LOG)
    after = snapshot(db)
    import_ledger(db, prefix)
    assert snapshot(db) == after


def test_a_ledger_that_differs_from_the_stored_log_is_refused_without_writes(db):
    import_ledger(db, LOG)
    before = snapshot(db)
    *kept, last = LOG.events
    with pytest.raises(RegistryError):
        import_ledger(db, Ledger((*kept, replace(last, reason="Replaced by a rerun"))))
    assert snapshot(db) == before


def test_concurrent_imports_of_one_ledger_retry_and_store_one_copy(
    server, db, request, monkeypatch
):
    barrier = threading.Barrier(2)
    commits = []
    with server.connect(NAMESPACE, request.node.name) as other:
        for connection in (db, other):
            commit_together(monkeypatch, connection, barrier, commits)
        with ThreadPoolExecutor(max_workers=2) as workers:
            results = list(
                workers.map(lambda connection: import_ledger(connection, LOG), (db, other))
            )
    assert results == [None, None]
    assert len(commits) == 3
    assert stored_log(db) == logged(LOG)


def test_concurrent_loads_at_different_times_store_one_log(server, db, request, monkeypatch):
    result = finding("Momentum earns a premium", INVALIDATION.evidence_id)
    import_ledger(db, Ledger(tuple(accepted(TIME, result))))
    import_invalidation(db, INVALIDATION)
    times = (TIME + timedelta(days=2), TIME + timedelta(days=3))
    barrier = threading.Barrier(2)
    commits = []
    with server.connect(NAMESPACE, request.node.name) as other:
        for connection in (db, other):
            commit_together(monkeypatch, connection, barrier, commits)
        with ThreadPoolExecutor(max_workers=2) as workers:
            first, second = workers.map(
                lambda connection, at: load_ledger(connection, at=at), (db, other), times
            )
    assert len(commits) == 3
    assert first.events == second.events
    assert stored_log(db) == logged(first)
    [suspension] = first.events[2:]
    assert (suspension.record_id, suspension.action) == (record_id(result), "suspend")
    assert suspension.at in times


@pytest.mark.parametrize(
    "surql, variables",
    [
        ("DELETE $id", {"id": RecordID("review_event", 0)}),
        (
            "INSERT $rows",
            {
                "rows": [
                    {
                        "id": RecordID("review_event", str(len(LOG.events))),
                        "event": event_value(replace(LOG.events[0], at=TIME + timedelta(hours=2))),
                    }
                ]
            },
        ),
    ],
    ids=["gap", "stray"],
)
def test_a_stored_log_with_a_gap_or_a_stray_id_is_refused_without_writes(db, surql, variables):
    import_ledger(db, LOG)
    store.query(db, surql, variables)
    before = snapshot(db)
    with pytest.raises(RegistryError):
        load_ledger(db, at=TIME + timedelta(days=1))
    assert snapshot(db) == before


def test_loading_before_the_last_event_refuses_a_due_suspension_without_writes(db):
    ledger = Ledger(
        tuple(accepted(TIME + timedelta(days=2), finding("Momentum", INVALIDATION.evidence_id)))
    )
    import_ledger(db, ledger)
    early = TIME + timedelta(days=1)
    assert load_ledger(db, at=early).events == ledger.events
    import_invalidation(db, INVALIDATION)
    before = snapshot(db)
    with pytest.raises(ReviewError):
        load_ledger(db, at=early)
    assert snapshot(db) == before
    assert len(load_ledger(db, at=TIME + timedelta(days=3)).events) == 3


@pytest.fixture
def run_record(probe, tmp_path) -> RunRecord:
    data = tmp_path / "datasets"
    vintage = probe.save(data, ["A", "B"])
    read = datasets.read(probe.catalog(vintage=vintage), data, probe.name, probe.context)
    return RunRecord(f"sha256:{'0' * 64}", sha256(b"pass"), read.record, Limits())


@pytest.fixture
def runs(tmp_path, run_record) -> Path:
    directory = tmp_path / "runs"
    directory.mkdir()
    log = RunLog(directory)
    log.append(Signal({"A": 1.0, "B": 2.0}, run_record))
    log.append(Refused("timeout", run_record))
    return directory


def stored_runs(db) -> list:
    [rows] = store.query(db, "SELECT * FROM container_run ORDER BY id")
    return [(row["id"].id, row["run"], row["entry"]) for row in rows]


def logged_runs(directory: Path) -> list:
    return [
        (position, logged.run_id, (directory / f"{position}.json").read_bytes())
        for position, logged in enumerate(RunLog(directory).load())
    ]


def test_an_imported_run_log_loads_from_a_reopened_connection_with_the_same_entries(
    server, db, request, tmp_path, runs
):
    import_run_log(db, RunLog(runs))
    with server.connect(NAMESPACE, request.node.name) as reopened:
        loaded = load_run_log(reopened)
    assert loaded == ((runs / "0.json").read_bytes(), (runs / "1.json").read_bytes())
    rebuilt = tmp_path / "rebuilt"
    rebuilt.mkdir()
    for position, entry in enumerate(loaded):
        (rebuilt / f"{position}.json").write_bytes(entry)
    assert RunLog(rebuilt).load() == RunLog(runs).load()
    assert stored_runs(db) == logged_runs(runs)


def test_reimporting_a_run_log_writes_nothing_and_an_extension_adds_its_new_entries(
    db, tmp_path, runs, run_record
):
    import_run_log(db, RunLog(runs))
    before = snapshot(db)
    import_run_log(db, RunLog(runs))
    assert snapshot(db) == before
    prefix = tmp_path / "prefix"
    shutil.copytree(runs, prefix)
    RunLog(runs).append(Refused("memory", run_record))
    import_run_log(db, RunLog(runs))
    assert stored_runs(db) == logged_runs(runs)
    after = snapshot(db)
    import_run_log(db, RunLog(prefix))
    assert snapshot(db) == after


@pytest.mark.parametrize("conflict", ["same-run", "another-log"])
def test_a_run_log_that_differs_from_the_stored_log_is_refused_without_writes(
    db, tmp_path, runs, run_record, conflict
):
    import_run_log(db, RunLog(runs))
    before = snapshot(db)
    other = tmp_path / "other"
    if conflict == "same-run":
        shutil.copytree(runs, other)
        changed = other / "1.json"
        entry = json.loads(changed.read_bytes())
        entry["run"]["outcome"] = "memory"
        changed.write_bytes(canonical_bytes(entry))
    else:
        other.mkdir()
        RunLog(other).append(Signal({"A": 1.0, "B": 2.0}, run_record))
    with pytest.raises(RegistryError):
        import_run_log(db, RunLog(other))
    assert snapshot(db) == before


def test_a_run_log_that_repeats_a_run_is_refused_before_any_table(db, runs):
    log = RunLog(runs)
    shutil.copyfile(runs / "0.json", runs / "2.json")
    with pytest.raises(WorkerError, match="repeats a run"):
        import_run_log(db, log)
    assert store.query(db, "INFO FOR DB")[0]["tables"] == {}


@pytest.mark.parametrize(
    "surql, variables",
    [
        ("DELETE $id", {"id": RecordID("container_run", 0)}),
        (
            "INSERT $rows",
            {"rows": [{"id": RecordID("container_run", "2"), "run": uuid4(), "entry": b"{}\n"}]},
        ),
    ],
    ids=["gap", "stray"],
)
def test_a_stored_run_log_with_a_gap_or_a_stray_id_is_refused(db, runs, surql, variables):
    import_run_log(db, RunLog(runs))
    store.query(db, surql, variables)
    with pytest.raises(RegistryError):
        load_run_log(db)


def test_loading_an_empty_registry_returns_an_empty_run_log(db):
    assert load_run_log(db) == ()


def test_campaign_artifacts_round_trip_after_importing_the_original_trial_rows(
    db, tmp_path, history
):
    record, artifacts = history
    original = save_package(tmp_path / "packages", record, artifacts)
    objects = tmp_path / "objects"
    import_package(db, objects, original)
    campaign = save_package(
        tmp_path / "packages",
        replace(record, trial_id=None),
        (
            *artifacts,
            ArtifactInput(
                "campaign-original-report",
                b'{"ending":{"outcome":"cancelled"}}\n',
                "application/json",
                "campaign-original-report",
            ),
            ArtifactInput(
                "campaign-call-0",
                b'{"schema_version":3,"preserved":"original bytes"}\n',
                "application/json",
                "campaign-call",
            ),
            ArtifactInput(
                "campaign-transcript",
                b'"\\u0065xact historical reply"',
                "application/json",
                "campaign-transcript",
            ),
        ),
    )
    assert import_package(db, objects, campaign) == campaign.digest
    assert import_package(db, objects, campaign) == campaign.digest
    exported = export_package(db, objects, campaign.digest, tmp_path / "exported")
    assert files(exported.path) == files(campaign.path)
    assert exported.record.trials == original.record.trials
    assert exported.record.exposures == original.record.exposures
    for item in campaign.artifact_records:
        assert exported.read_artifact(item.name) == campaign.read_artifact(item.name)
