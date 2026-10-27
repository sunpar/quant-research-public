from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

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
from quant_research.records import PackageError

ROOT = Path(__file__).resolve().parents[1]
TIME = datetime(2026, 10, 21, 12, tzinfo=UTC)


@pytest.fixture
def evidence():
    artifacts = tuple(
        ArtifactInput(name, data, "text/plain", role)
        for name, data, role in (
            ("input", b"synthetic input\n", "input"),
            ("contract", b"synthetic only, version 1\n", "data contract"),
            ("code", b"hand-authored evidence\n", "implementation"),
            ("lock", b"locked environment\n", "environment"),
            ("result", b"named result, no scientific interpretation\n", "result"),
            ("failure", b"supplied execution failure\n", "failure detail"),
        )
    )
    digests = {item.name: hashlib.sha256(item.data).hexdigest() for item in artifacts}
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
    attempt = ExecutionAttempt(
        uuid4(),
        uuid4(),
        None,
        TIME,
        TIME + timedelta(seconds=1),
        Outcome("successful", "Supplied generic outcome", ("result",), ("Synthetic data only",)),
    )
    trial = TrialRecord(uuid4(), uuid4(), spec, (attempt,))
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
    return PackageRecord(trial.trial_id, (trial,), (exposure,)), artifacts


def rewrite_manifest(path: Path, value: dict | bytes) -> None:
    content = (
        value
        if isinstance(value, bytes)
        else (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
    )
    (path / "manifest.json").write_bytes(content)
    (path / "manifest.sha256").write_text(hashlib.sha256(content).hexdigest() + "\n")


@pytest.mark.smoke
def test_portable_copy_recovers_every_artifact_and_inspects_without_origin(tmp_path, evidence, cli):
    record, artifacts = evidence
    origin = tmp_path / "origin"
    saved = save_package(origin, record, artifacts)
    copied = tmp_path / "unrelated directory" / "renamed package"
    shutil.copytree(saved.path, copied)
    shutil.rmtree(origin)
    reopened = open_package(copied)
    assert reopened.record == record
    assert reopened.digest == saved.digest
    assert {item.name: reopened.read_artifact(item.name) for item in artifacts} == {
        item.name: item.data for item in artifacts
    }
    assert reopened.artifacts() == artifacts
    exported = save_package(tmp_path / "another destination", reopened.record, reopened.artifacts())
    assert exported.digest == saved.digest
    assert exported.path.parent.name == str(record.trial_id)
    assert exported.path.name == saved.digest
    result = json.loads(cli(tmp_path, "inspect", str(copied)).stdout)
    assert result["lineage_id"] == str(record.selected_trial.lineage_id)
    assert result["record"]["exposures"][0]["id"] == str(record.exposures[0].id)
    assert (
        result["record"]["trials"][0]["specification"]["environment"]["python_version"] == "3.12.13"
    )
    assert result["record"]["trials"][0]["attempts"][0]["outcome"]["status"] == "successful"
    assert set(item["name"] for item in result["artifacts"]) == {item.name for item in artifacts}
    assert saved.digest in cli(tmp_path, "verify", str(copied)).stdout


@pytest.mark.parametrize("supplied", [True, False], ids=["trials", "no-trials"])
def test_a_package_without_a_selected_trial_saves_as_version_3_and_reopens(
    tmp_path, evidence, cli, supplied
):
    record, artifacts = evidence
    unselected = replace(record, trial_id=None, trials=record.trials if supplied else ())
    saved = save_package(tmp_path / "origin", unselected, artifacts)
    manifest = json.loads((saved.path / "manifest.json").read_bytes())
    assert (manifest["schema_version"], manifest["record"]["trial_id"]) == (3, None)
    assert saved.path.relative_to(tmp_path / "origin") == Path("unselected", saved.digest)
    reopened = open_package(saved.path)
    assert reopened.record == unselected
    resaved = save_package(tmp_path / "elsewhere", reopened.record, reopened.artifacts())
    assert resaved.digest == saved.digest
    assert saved.digest in cli(tmp_path, "verify", str(saved.path)).stdout
    inspected = json.loads(cli(tmp_path, "inspect", str(saved.path)).stdout)
    assert (inspected["lineage_id"], inspected["record"]["trial_id"]) == (None, None)


def test_selected_trial_of_a_package_without_one_raises_a_named_error(tmp_path, evidence):
    record, artifacts = evidence
    saved = save_package(tmp_path, replace(record, trial_id=None), artifacts)
    with pytest.raises(PackageError, match="The package selects no trial"):
        _ = open_package(saved.path).record.selected_trial


@pytest.mark.parametrize("status", ["successful", "rejected", "failed", "null", "inconclusive"])
def test_generic_statuses_and_named_results_survive_save(tmp_path, evidence, status):
    record, artifacts = evidence
    trial = record.selected_trial
    attempt = replace(trial.attempts[0], outcome=Outcome(status, "Supplied outcome", ("result",)))
    updated = replace(record, trials=(replace(trial, attempts=(attempt,)),))
    assert save_package(tmp_path, updated, artifacts).record == updated


@pytest.mark.smoke
def test_failed_retry_repeat_and_changed_spec_keep_complete_supplied_history(
    tmp_path, evidence, cli
):
    record, artifacts = evidence
    first = record.selected_trial
    failed = replace(first.attempts[0], outcome=Outcome("failed", "Executor failed", ("failure",)))
    first = replace(first, attempts=(failed,))
    original = save_package(tmp_path, replace(record, trials=(first,)), artifacts)
    retry = replace(
        failed,
        id=uuid4(),
        predecessor=failed.id,
        started_at=TIME + timedelta(seconds=2),
        finished_at=TIME + timedelta(seconds=3),
        outcome=Outcome("null", "Supplied null result", ("result",)),
    )
    repeat = replace(
        retry,
        id=uuid4(),
        run_id=uuid4(),
        predecessor=None,
        started_at=TIME + timedelta(seconds=4),
        finished_at=TIME + timedelta(seconds=5),
    )
    current = replace(first, attempts=(failed, retry, repeat))
    retried = save_package(tmp_path, replace(record, trials=(current,)), artifacts)
    assert original.record.selected_trial.attempts == (failed,)
    assert retried.record.selected_trial.attempts == (failed, retry, repeat)
    assert failed.run_id == retry.run_id != repeat.run_id
    assert len({failed.id, retry.id, repeat.id}) == 3
    assert retried.read_artifact("failure") == next(
        item.data for item in artifacts if item.name == "failure"
    )
    changed = replace(
        current, specification=replace(current.specification, question="A new question")
    )
    with pytest.raises(PackageError, match="conflicting trial ID"):
        save_package(tmp_path, replace(record, trials=(current, changed)), artifacts)
    changed = replace(changed, trial_id=uuid4(), attempts=())
    with pytest.raises(PackageError, match="requires a prior trial link"):
        save_package(
            tmp_path,
            replace(record, trial_id=changed.trial_id, trials=(current, changed)),
            artifacts,
        )
    linked = replace(
        changed,
        links=(
            LineageLink(
                current.trial_id,
                current.specification.spec_id,
                "derived_from",
                "Changed scientific question",
            ),
        ),
    )
    history = replace(record, trial_id=linked.trial_id, trials=(current, linked))
    saved = save_package(tmp_path, history, artifacts)
    assert saved.record == history
    assert saved.record.trials[0].attempts == (failed, retry, repeat)
    assert saved.record.selected_trial.lineage_id == first.lineage_id
    assert saved.record.trials[0].specification.spec_id != linked.specification.spec_id
    inspected = json.loads(cli(tmp_path, "inspect", str(saved.path)).stdout)
    assert len(inspected["record"]["trials"][0]["attempts"]) == 3
    assert len(inspected["record"]["exposures"]) == 1


def test_related_lineages_and_declared_exposures_survive_composition(tmp_path, evidence):
    record, artifacts = evidence
    old = record.selected_trial
    related = replace(old, trial_id=uuid4(), lineage_id=uuid4(), attempts=())
    exposure = replace(record.exposures[0], id=uuid4(), actor="second researcher", kind="feedback")
    combined = replace(
        record,
        trial_id=related.trial_id,
        trials=(old, related),
        exposures=(*record.exposures, exposure),
    )
    saved = save_package(tmp_path, combined, artifacts)
    assert open_package(saved.path).record == combined
    assert tuple(trial.lineage_id for trial in saved.record.trials) == (
        old.lineage_id,
        related.lineage_id,
    )
    assert saved.record.exposures == (*record.exposures, exposure)


def test_identical_saves_and_separate_concurrent_histories_remain_addressable(tmp_path, evidence):
    record, artifacts = evidence
    second = replace(
        record,
        exposures=(
            *record.exposures,
            replace(record.exposures[0], id=uuid4(), reason="Another supplied exposure"),
        ),
    )
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(
            workers.map(
                lambda item: save_package(tmp_path, item, artifacts),
                (record, record, second, second),
            )
        )
    assert results[0].digest == results[1].digest
    assert results[2].digest == results[3].digest
    assert results[0].digest != results[2].digest
    assert open_package(results[0].path).record == record
    assert open_package(results[2].path).record == second
    assert not list(tmp_path.rglob(".package-*"))


def test_empty_history_still_declares_coverage_and_unavailable_confirmation(
    tmp_path, evidence, cli
):
    record, artifacts = evidence
    empty = replace(record, trials=(replace(record.selected_trial, attempts=()),), exposures=())
    saved = save_package(tmp_path, empty, artifacts)
    result = json.loads(cli(tmp_path, "inspect", str(saved.path)).stdout)["record"]
    assert result["trials"][0]["attempts"] == result["exposures"] == []
    assert result["disclosure_coverage"] == "unknown"
    assert result["scope"] == "development"
    assert result["confirmation"] == "unavailable"


@pytest.mark.parametrize("coverage", ["unknown", "declared_inputs"])
def test_disclosure_coverage_survives_save_open_verify_and_inspect(
    tmp_path, evidence, cli, coverage
):
    record, artifacts = evidence
    declared = replace(record, disclosure_coverage=coverage)
    saved = save_package(tmp_path, declared, artifacts)
    assert json.loads((saved.path / "manifest.json").read_bytes())["schema_version"] == 2
    opened = open_package(saved.path)
    assert opened.record.disclosure_coverage == coverage
    assert opened.record == declared
    assert saved.digest in cli(tmp_path, "verify", str(saved.path)).stdout
    inspected = json.loads(cli(tmp_path, "inspect", str(saved.path)).stdout)
    assert inspected["record"]["disclosure_coverage"] == coverage


@pytest.mark.parametrize(
    "version, coverage",
    [
        (2, "caller_declared"),
        (2, ["declared_inputs"]),
        (1, "unknown"),
        (1, "declared_inputs"),
        (3, "caller_declared"),
    ],
)
def test_coverage_outside_the_manifest_version_vocabulary_is_refused(
    tmp_path, evidence, version, coverage
):
    saved = save_package(tmp_path, *evidence)
    value = json.loads((saved.path / "manifest.json").read_bytes())
    value["schema_version"] = version
    value["record"]["disclosure_coverage"] = coverage
    if version == 3:
        value["record"]["trial_id"] = None
    rewrite_manifest(saved.path, value)
    with pytest.raises(PackageError, match="Unsupported disclosure coverage"):
        open_package(saved.path)


def test_committed_schema_one_package_opens_and_verifies_with_unknown_coverage(tmp_path, cli):
    source = ROOT / "examples" / "development-package"
    assert json.loads((source / "manifest.json").read_bytes())["schema_version"] == 1
    package = open_package(source)
    assert package.record.disclosure_coverage == "unknown"
    assert package.digest in cli(tmp_path, "verify", str(source)).stdout
    inspected = json.loads(cli(tmp_path, "inspect", str(source)).stdout)
    assert inspected["record"]["disclosure_coverage"] == "unknown"


@pytest.mark.parametrize("damage", ["tamper", "last-artifact", "missing", "manifest"])
def test_open_cli_and_saved_reader_reject_changed_or_missing_bytes(tmp_path, evidence, damage, cli):
    saved = save_package(tmp_path, *evidence)
    artifact = saved.artifact_records[-1 if damage == "last-artifact" else 0]
    path = saved.path / "artifacts" / artifact.digest
    if damage in ("tamper", "last-artifact"):
        path.write_bytes(b"changed bytes")
    elif damage == "missing":
        path.unlink()
    else:
        (saved.path / "manifest.json").write_bytes(
            (saved.path / "manifest.json").read_bytes() + b" "
        )
    with pytest.raises(PackageError):
        open_package(saved.path)
    with pytest.raises(PackageError):
        saved.read_artifact(artifact.name)
    with pytest.raises(PackageError):
        saved.artifacts()
    assert (
        "Package verification failed"
        in cli(tmp_path, "verify", str(saved.path), success=False).stderr
    )


def test_saved_reader_detects_a_replaced_manifest_even_with_recomputed_checksum(tmp_path, evidence):
    saved = save_package(tmp_path, *evidence)
    value = json.loads((saved.path / "manifest.json").read_bytes())
    value["record"]["exposures"] = []
    rewrite_manifest(saved.path, value)
    with pytest.raises(PackageError, match="manifest changed"):
        saved.read_artifact("result")
    with pytest.raises(PackageError, match="manifest changed"):
        saved.artifacts()


@pytest.mark.parametrize(
    "component", ["root", "artifacts", "manifest.json", "manifest.sha256", "artifact"]
)
def test_package_boundaries_reject_symlinks_even_to_identical_bytes(tmp_path, evidence, component):
    saved = save_package(tmp_path / "saved", *evidence)
    path = saved.path
    if component == "root":
        original = path
        path = tmp_path / "linked package"
        path.symlink_to(original, target_is_directory=True)
    else:
        target = path / (
            f"artifacts/{saved.artifact_records[0].digest}"
            if component == "artifact"
            else component
        )
        outside = tmp_path / "outside"
        target.rename(outside)
        target.symlink_to(outside, target_is_directory=outside.is_dir())
    with pytest.raises(PackageError):
        open_package(path)
    reader = replace(saved, path=path)
    with pytest.raises(PackageError):
        reader.read_artifact(saved.artifact_records[0].name)
    with pytest.raises(PackageError):
        reader.artifacts()


def test_nonregular_artifact_and_undeclared_files_fail_without_blocking(tmp_path, evidence):
    saved = save_package(tmp_path, *evidence)
    path = saved.path / "artifacts" / saved.artifact_records[0].digest
    path.unlink()
    os.mkfifo(path)
    with pytest.raises(PackageError, match="regular file"):
        open_package(saved.path)
    path.unlink()
    path.write_bytes(evidence[1][0].data)
    (saved.path / "artifacts" / "unlisted").symlink_to(tmp_path)
    with pytest.raises(PackageError, match="do not match"):
        open_package(saved.path)


@pytest.mark.parametrize(
    "damage, message",
    [
        ("version", "Unsupported schema"),
        ("boolean-version", "Unsupported schema"),
        ("unknown-fields", "unknown 'unexpected'"),
        ("typo-field", "missing 'lineage_id'; unknown 'lineage_di'"),
        ("trial-id", "UUID"),
        ("selected-trial", "Selected trial"),
        ("selection-in-version-3", "Schema version 3 records no selected trial"),
        ("no-selection-in-version-2", "package trial_id UUID"),
        ("duplicate-trial", "conflicting trial"),
        ("duplicate-attempt", "conflicting attempt"),
        ("duplicate-exposure", "conflicting exposure"),
        ("duplicate-artifact", "conflicting artifact"),
        ("spec-id", "Specification ID"),
        ("missing-digest", "unavailable artifact digest"),
        ("result-reference", "unavailable result artifact"),
        ("predecessor", "Retry predecessor"),
        ("timestamp", "started_at timestamp"),
        ("reversed-attempt", "finishes before"),
        ("reversed-exposure", "interval ends"),
        ("unsafe-name", "logical artifact name"),
        ("unsafe-digest", "artifact digest"),
        ("boolean-size", "size"),
        ("confirmation", "unavailable"),
        ("coverage", "Unsupported disclosure coverage"),
        ("scope", "development"),
        ("status", "Unsupported development outcome"),
    ],
)
def test_malformed_external_records_are_rejected_at_open(tmp_path, evidence, damage, message):
    saved = save_package(tmp_path, *evidence)
    value = json.loads((saved.path / "manifest.json").read_bytes())
    record = value["record"]
    trial = record["trials"][0]
    attempt = trial["attempts"][0]
    match damage:
        case "version":
            value["schema_version"] = 4
        case "boolean-version":
            value["schema_version"] = True
        case "unknown-fields":
            trial["unexpected"] = "field"
        case "typo-field":
            trial["lineage_di"] = trial.pop("lineage_id")
        case "trial-id":
            trial["trial_id"] = "../unsafe"
        case "selected-trial":
            record["trial_id"] = str(uuid4())
        case "selection-in-version-3":
            value["schema_version"] = 3
        case "no-selection-in-version-2":
            record["trial_id"] = None
        case "duplicate-trial":
            record["trials"].append(trial)
        case "duplicate-attempt":
            trial["attempts"].append(attempt)
        case "duplicate-exposure":
            record["exposures"].append(record["exposures"][0])
        case "duplicate-artifact":
            value["artifacts"].append(value["artifacts"][0])
        case "spec-id":
            trial["specification"]["spec_id"] = "0" * 64
        case "missing-digest":
            record["exposures"][0]["material_digest"] = "0" * 64
        case "result-reference":
            attempt["outcome"]["result_artifacts"] = ["missing"]
        case "predecessor":
            attempt["predecessor"] = str(uuid4())
        case "timestamp":
            attempt["started_at"] = "2026-10-21T12:00:00"
        case "reversed-attempt":
            attempt["finished_at"] = "2026-10-20T12:00:00Z"
        case "reversed-exposure":
            record["exposures"][0]["interval_end"] = "2026-10-20T12:00:00Z"
        case "unsafe-name":
            value["artifacts"][0]["name"] = "../outside"
        case "unsafe-digest":
            value["artifacts"][0]["digest"] = "../outside"
        case "boolean-size":
            value["artifacts"][0]["size"] = True
        case "confirmation":
            record["confirmation"] = "available"
        case "coverage":
            record["disclosure_coverage"] = "complete"
        case "scope":
            record["scope"] = "confirmation"
        case "status":
            attempt["outcome"]["status"] = "confirmed"
    rewrite_manifest(saved.path, value)
    with pytest.raises(PackageError, match=message):
        open_package(saved.path)


@pytest.mark.parametrize(
    "damage",
    [
        "duplicate-key",
        "nan",
        "infinity",
        "negative-infinity",
        "encoding",
        "noncanonical",
        "deep-json",
        "large-integer",
    ],
)
def test_external_json_requires_unique_keys_and_strict_canonical_encoding(
    tmp_path, evidence, damage, cli
):
    saved = save_package(tmp_path, *evidence)
    raw = (saved.path / "manifest.json").read_bytes()
    match damage:
        case "duplicate-key":
            raw = raw.replace(b'"schema_version":2', b'"schema_version":2,"schema_version":2')
        case "nan" | "infinity" | "negative-infinity":
            raw = raw.replace(
                b'"unknown"',
                {"nan": b"NaN", "infinity": b"Infinity", "negative-infinity": b"-Infinity"}[damage],
            )
        case "encoding":
            raw = b"\xff"
        case "noncanonical":
            raw += b" "
        case "deep-json":
            raw = b"[" * 20000 + b"0" + b"]" * 20000
        case "large-integer":
            raw = b"1" * 5000
    rewrite_manifest(saved.path, raw)
    with pytest.raises(PackageError):
        open_package(saved.path)
    result = cli(tmp_path, "verify", str(saved.path), success=False)
    assert "Package verification failed" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    "damage", ["repeat-run", "retry-run", "foreign-trial", "link-spec", "link-trial"]
)
def test_save_rejects_inconsistent_trial_run_attempt_and_link_references(
    tmp_path, evidence, damage
):
    record, artifacts = evidence
    first = record.selected_trial
    attempt = first.attempts[0]
    later = replace(
        attempt,
        id=uuid4(),
        started_at=TIME + timedelta(seconds=2),
        finished_at=TIME + timedelta(seconds=3),
    )
    trials = (first,)
    match damage:
        case "repeat-run":
            trials = (replace(first, attempts=(attempt, later)),)
        case "retry-run":
            trials = (
                replace(
                    first,
                    attempts=(attempt, replace(later, predecessor=attempt.id, run_id=uuid4())),
                ),
            )
        case "foreign-trial":
            other = replace(
                first,
                trial_id=uuid4(),
                lineage_id=uuid4(),
                attempts=(replace(later, predecessor=attempt.id),),
            )
            trials = (first, other)
        case "link-spec" | "link-trial":
            link = LineageLink(
                first.trial_id if damage == "link-spec" else uuid4(),
                "0" * 64,
                "derived_from",
                "Changed question",
            )
            trials = (first, replace(first, trial_id=uuid4(), attempts=(), links=(link,)))
    with pytest.raises(PackageError):
        save_package(tmp_path, replace(record, trials=trials), artifacts)
    assert not list(tmp_path.iterdir())


def test_public_save_rejects_invalid_typed_values_and_symlink_destinations(tmp_path, evidence):
    record, artifacts = evidence
    naive = replace(record.selected_trial.attempts[0], started_at=TIME.replace(tzinfo=None))
    with pytest.raises(PackageError, match="timezone"):
        save_package(
            tmp_path,
            replace(record, trials=(replace(record.selected_trial, attempts=(naive,)),)),
            artifacts,
        )
    with pytest.raises(PackageError, match="content must be bytes"):
        save_package(tmp_path, record, (replace(artifacts[0], data="text"),))
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "link"
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(PackageError, match="must not be a symlink"):
        save_package(linked, record, artifacts)
    assert not list(target.iterdir())


@pytest.mark.parametrize("text", ["\ud800", "\udfff"])
@pytest.mark.parametrize("field", ["question", "environment", "outcome", "exposure", "artifact"])
def test_public_save_rejects_lone_surrogates_before_publication(tmp_path, evidence, text, field):
    record, artifacts = evidence
    trial = record.selected_trial
    match field:
        case "question":
            trial = replace(trial, specification=replace(trial.specification, question=text))
        case "environment":
            spec = trial.specification
            trial = replace(
                trial,
                specification=replace(spec, environment=replace(spec.environment, platform=text)),
            )
        case "outcome":
            attempt = trial.attempts[0]
            trial = replace(
                trial, attempts=(replace(attempt, outcome=replace(attempt.outcome, detail=text)),)
            )
        case "exposure":
            record = replace(record, exposures=(replace(record.exposures[0], actor=text),))
        case "artifact":
            artifacts = (replace(artifacts[0], media_type=text), *artifacts[1:])
    record = replace(record, trials=(trial,))
    supplied = asdict(record), tuple(asdict(item) for item in artifacts)
    with pytest.raises(PackageError, match="UTF-8"):
        save_package(tmp_path, record, artifacts)
    assert not list(tmp_path.iterdir())
    assert (asdict(record), tuple(asdict(item) for item in artifacts)) == supplied


@pytest.mark.parametrize("error_type", [TypeError, AttributeError, ValueError])
def test_save_preserves_internal_programming_errors(tmp_path, evidence, monkeypatch, error_type):
    error = error_type("Internal serialization failure")

    def fail_record_data(record):
        raise error

    monkeypatch.setattr("quant_research.packages._record_data", fail_record_data)
    with pytest.raises(error_type) as caught:
        save_package(tmp_path, *evidence)
    assert caught.value is error
    assert not list(tmp_path.iterdir())


def test_hash_bearing_records_survive_git_checkout_with_autocrlf_and_copy(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    source = ROOT / "examples" / "development-package"
    shutil.copytree(source, repository / "examples" / "development-package")
    protected = {
        path.relative_to(ROOT): path.read_bytes() for path in source.rglob("*") if path.is_file()
    }
    shutil.copy2(ROOT / ".gitattributes", repository / ".gitattributes")
    control = repository / "checkout-control.txt"
    control.write_bytes(b"one\ntwo\n")
    for arguments in (
        ("init", "--quiet"),
        (
            "-c",
            "core.autocrlf=false",
            "add",
            "--",
            ".gitattributes",
            "examples",
            control.name,
        ),
    ):
        subprocess.run(["git", "-C", str(repository), *arguments], check=True, capture_output=True)
    shutil.rmtree(repository / "examples")
    control.unlink()
    subprocess.run(
        ["git", "-C", str(repository), "-c", "core.autocrlf=true", "checkout-index", "--all"],
        check=True,
        capture_output=True,
    )
    checked_out = repository / "examples" / "development-package"
    assert control.read_bytes() == b"one\r\ntwo\r\n"
    assert {path: (repository / path).read_bytes() for path in protected} == protected
    copied = tmp_path / "copied package"
    shutil.copytree(checked_out, copied)
    original = open_package(source)
    reopened = open_package(copied)
    assert reopened.digest == original.digest
    assert reopened.record == original.record
    assert reopened.artifacts() == original.artifacts()


def test_new_authoring_contexts_have_distinct_identities_and_composable_history(tmp_path):
    authored = []
    started = datetime.now(UTC)
    for number in (1, 2):
        lock = tmp_path / f"context-{number}.lock"
        lock.write_bytes(f"supplied lock context {number}\n".encode())
        arguments = ["examples.save_development", str(tmp_path / "packages"), "--lock", str(lock)]
        bootstrap = (
            f"import runpy, sys; sys.path.extend({[str(ROOT), str(ROOT / 'src')]!r}); "
            f"sys.argv = {arguments!r}; "
            "runpy.run_module('examples.save_development', run_name='__main__')"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-I", "-S", "-c", bootstrap],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        )
        authored.append(open_package(Path(result.stdout.strip())))
    finished = datetime.now(UTC)
    trials = tuple(package.record.selected_trial for package in authored)
    assert trials[0].specification.spec_id != trials[1].specification.spec_id
    identities = [
        identity
        for trial in trials
        for identity in (
            trial.trial_id,
            trial.lineage_id,
            trial.attempts[0].id,
            trial.attempts[0].run_id,
        )
    ]
    assert len(set(identities)) == 8
    for trial in trials:
        attempt = trial.attempts[0]
        assert started <= attempt.started_at <= attempt.finished_at <= finished
        assert attempt.started_at.utcoffset() == timedelta(0)
    first, second = authored
    artifacts = {item.name: item for item in first.artifacts()}
    for artifact in second.artifacts():
        if artifact != artifacts.get(artifact.name):
            renamed = replace(artifact, name=f"second_{artifact.name}")
            artifacts[renamed.name] = renamed
    combined = save_package(
        tmp_path / "composed",
        PackageRecord(trials[-1].trial_id, trials),
        tuple(artifacts.values()),
    )
    reopened = open_package(combined.path)
    assert reopened.record.trials == trials
    assert reopened.record.disclosure_coverage == "unknown"
    assert reopened.read_artifact("uv_lock") == first.read_artifact("uv_lock")
    assert reopened.read_artifact("second_uv_lock") == second.read_artifact("uv_lock")


@pytest.mark.parametrize("kind", ["lineage", "run", "attempt", "exposure"])
def test_identity_kinds_cannot_reuse_the_same_uuid(tmp_path, evidence, kind):
    record, artifacts = evidence
    trial = record.selected_trial
    match kind:
        case "lineage":
            record = replace(record, trials=(replace(trial, lineage_id=trial.trial_id),))
        case "run" | "attempt":
            attempt = replace(
                trial.attempts[0], **{f"{kind}_id" if kind == "run" else "id": trial.trial_id}
            )
            record = replace(record, trials=(replace(trial, attempts=(attempt,)),))
        case "exposure":
            record = replace(record, exposures=(replace(record.exposures[0], id=trial.trial_id),))
    with pytest.raises(PackageError, match="distinct from other record kinds"):
        save_package(tmp_path, record, artifacts)
