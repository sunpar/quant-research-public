from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import unicodedata
from collections import Counter
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import get_args
from uuid import UUID, uuid4

import pytest

from quant_research.evaluation import Evaluation, Interval, Invalid, Profile
from quant_research.packages import (
    ArtifactInput,
    EnvironmentRecord,
    ExecutionAttempt,
    ExposureEvent,
    LineageLink,
    Outcome,
    PackageRecord,
    Specification,
    Status,
    TrialRecord,
    VerifiedPackage,
    open_package,
    save_package,
)
from quant_research.records import PackageError
from quant_research.reference import EVALUATION_ROLE, encode_evaluation
from quant_research.report import render_report

ROOT = Path(__file__).resolve().parents[1]
TIME = datetime(2026, 10, 21, 12, tzinfo=UTC)
SUPPORTED = Evaluation(
    "above_minimum",
    3,
    0.9916666666666667,
    Interval(0.012345678901234568, 0.0021, 0.0071, 0.0175),
    Interval(0.0042, 0.0011, 0.0015, 0.0069),
    (0.011, 0.0135, 0.0098, 0.0151),
    Profile(0.001, 4, 0.05),
)
UNBENCHMARKED = Evaluation(
    "spans_minimum",
    1,
    0.975,
    Interval(0.002, 0.0015, -0.001, 0.005),
    None,
    (0.003, 0.001),
    Profile(0.0025, 2, 0.05),
)
INVALID = Invalid("zero_variance", "Returns are *constant*\nin every month", Profile(0.001, 2, 0.1))
HEADINGS = [
    "## Limits",
    "## Question",
    "## Outcomes",
    "## Trials",
    "## Exposure history",
    "## Artifacts",
]


def attempt(
    minute: int, outcome: Outcome, run_id: UUID | None = None, predecessor: UUID | None = None
) -> ExecutionAttempt:
    return ExecutionAttempt(
        uuid4(),
        run_id or uuid4(),
        predecessor,
        TIME + timedelta(minutes=minute),
        TIME + timedelta(minutes=minute, seconds=30),
        outcome,
    )


def build(destination: Path, note: str = "") -> VerifiedPackage:
    artifacts = tuple(
        ArtifactInput(name, data, f"{media_type}{note}", f"{role}{note}")
        for name, data, media_type, role in (
            ("input", b"synthetic prices\n", "text/csv", "synthetic input"),
            ("input_copy", b"synthetic prices\n", "text/csv", "duplicate input"),
            ("contract", b"price contract\n", "text/plain", "data contract"),
            ("code", b"signal code\n", "text/x-python", "implementation"),
            ("lock", b"locked environment\n", "text/plain", "environment lock"),
            ("result", b"decay table\n", "text/csv", "result table"),
            ("failure", b"executor log\n", "text/plain", "failure log"),
        )
    )
    digests = {item.name: hashlib.sha256(item.data).hexdigest() for item in artifacts}
    spec = Specification(
        f"Measure decay of the declared signal{note}",
        f"declared signal/v1{note}",
        f"decay evaluation/v1{note}",
        f"synthetic/v1{note}",
        f"signal code/v1{note}",
        (digests["input"],),
        digests["contract"],
        digests["code"],
        EnvironmentRecord(
            f"cpython{note}",
            "3.12.13",
            f"cpython-312{note}",
            f"test-platform{note}",
            f"uv test{note}",
            digests["lock"],
        ),
    )

    def outcome(status, detail, limitation, results=("result",)):
        return Outcome(status, f"{detail}{note}", results, (f"{limitation}{note}",))

    failed = attempt(
        0, outcome("failed", "Executor ran out of memory", "Partial log", ("failure",))
    )
    first = TrialRecord(
        uuid4(),
        uuid4(),
        spec,
        (
            failed,
            attempt(
                1,
                outcome("successful", "Completed after a memory increase", "Single random seed"),
                failed.run_id,
                failed.id,
            ),
            attempt(2, outcome("null", "Repeat run found no decay", "Short sample window")),
        ),
    )
    second = TrialRecord(
        uuid4(),
        first.lineage_id,
        replace(
            spec,
            question=f"Measure decay on revised synthetic data{note}",
            data_version=f"synthetic/v2{note}",
        ),
        (
            attempt(3, outcome("rejected", "Decay reversed sign", "Costs held fixed")),
            attempt(4, outcome("null", "Second run found no decay", "Same short window")),
            attempt(5, outcome("inconclusive", "Revised sample too short", "Wide interval", ())),
        ),
        (
            LineageLink(
                first.trial_id, spec.spec_id, f"revises{note}", f"Revised data version{note}"
            ),
        ),
    )
    exposures = (
        ExposureEvent(
            uuid4(),
            f"steward{note}",
            TIME - timedelta(days=1),
            digests["input"],
            f"synthetic/v1{note}",
            TIME - timedelta(days=30),
            TIME - timedelta(days=2),
            "data",
            f"Loaded prices before the first trial{note}",
        ),
        ExposureEvent(
            uuid4(),
            f"reviewing researcher Zoë{note}",
            TIME + timedelta(hours=1),
            digests["result"],
            f"decay results{note}",
            TIME,
            TIME + timedelta(minutes=3),
            "feedback",
            f"Read first trial results before revising the sample{note}",
        ),
    )
    record = PackageRecord(second.trial_id, (first, second), exposures)
    return open_package(save_package(destination, record, artifacts).path)


def evaluated(destination: Path, results: dict[str, bytes]) -> VerifiedPackage:
    """Save the fixture with its last attempt also producing results as pilot evaluations."""
    package = build(destination / "plain")
    first, second = package.record.trials
    *earlier, last = second.attempts
    last = replace(last, outcome=replace(last.outcome, result_artifacts=("result", *results)))
    record = replace(package.record, trials=(first, replace(second, attempts=(*earlier, last))))
    produced = tuple(
        ArtifactInput(name, data, "application/json", EVALUATION_ROLE)
        for name, data in results.items()
    )
    saved = save_package(destination / "evaluated", record, (*package.artifacts(), *produced))
    return open_package(saved.path)


def section(text: str, level: int, marker: str) -> str:
    (body,) = [
        part
        for part in re.split(rf"(?m)^(?=#{{1,{level}}} )", text)
        if part.startswith(f"{'#' * level} ") and marker in part.splitlines()[0]
    ]
    return body


def headings(text: str, level: int) -> list[str]:
    return [line for line in text.splitlines() if line.startswith(f"{'#' * level} ")]


def bullets(body: str) -> list[tuple[str, str]]:
    return [tuple(line[2:].split(": ", 1)) for line in body.splitlines() if line.startswith("- ")]


def structure(text: str) -> list[str]:
    return [line.split("`")[0].split(":")[0] for line in text.splitlines()]


def strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from strings(item)


@pytest.mark.smoke
def test_each_attempt_block_carries_its_own_status_detail_limitations_and_run(tmp_path):
    package = build(tmp_path)
    text = render_report(package)
    trials = package.record.trials
    attempts = [attempt for trial in trials for attempt in trial.attempts]
    for attempt, heading in zip(attempts, headings(text, 4), strict=True):
        assert f"`{attempt.id}`" in heading
    for attempt in attempts:
        body = section(text, 4, f"`{attempt.id}`")
        outcome = attempt.outcome
        assert outcome.status in body.splitlines()[0]
        assert {outcome.detail, *outcome.limitations} <= {value for _, value in bullets(body)}
        for value in (
            attempt.run_id,
            attempt.started_at.isoformat(),
            attempt.finished_at.isoformat(),
            *outcome.result_artifacts,
        ):
            assert f"`{value}`" in body
        assert not any(other.outcome.detail in body for other in attempts if other != attempt)
    failed, retry, repeat = trials[0].attempts
    assert f"retry of attempt `{failed.id}`" in section(text, 4, f"`{retry.id}`")
    for first in (failed, repeat, *trials[1].attempts):
        assert "first attempt in run" in section(text, 4, f"`{first.id}`")
    counts = Counter(attempt.outcome.status for attempt in attempts)
    assert dict(bullets(section(text, 2, "Outcomes"))) == {
        "Trials": str(len(trials)),
        "Attempts": str(len(attempts)),
        **{status: str(counts[status]) for status in get_args(Status.__value__)},
    }


def test_selected_trial_shows_its_question_specification_provenance_and_link(tmp_path):
    package = build(tmp_path)
    text = render_report(package)
    first, selected = package.record.trials
    spec = selected.specification
    environment = spec.environment
    (link,) = selected.links
    intro = text.split("\n## ", 1)[0]
    for value in (package.digest, selected.trial_id, selected.lineage_id, "quant-research verify"):
        assert str(value) in intro
    assert [value for _, value in bullets(section(text, 2, "Question"))] == [spec.question]
    body = section(text, 3, f"`{selected.trial_id}`")
    assert "selected" in body.splitlines()[0]
    assert "selected" not in section(text, 3, f"`{first.trial_id}`").splitlines()[0]
    assert {
        spec.question,
        spec.signal_profile,
        spec.evaluation_profile,
        spec.data_version,
        spec.implementation_version,
        environment.implementation,
        environment.python_version,
        environment.abi,
        environment.platform,
        environment.uv_version,
        link.relation,
        link.reason,
    } <= {value for _, value in bullets(body)}
    for value in (spec.spec_id, selected.lineage_id, link.trial_id, link.spec_id):
        assert f"`{value}`" in body
    for digest, names in (
        (spec.input_digests[0], ("input", "input_copy")),
        (spec.data_contract_digest, ("contract",)),
        (spec.implementation_digest, ("code",)),
        (environment.lock_digest, ("lock",)),
    ):
        (line,) = [line for line in body.splitlines() if f"`{digest}`" in line]
        assert all(f"`{name}`" in line for name in names)
    assert first.specification.question not in body
    assert first.specification.data_version not in {value for _, value in bullets(body)}


def test_report_orders_sections_lists_exposures_and_artifacts_and_states_limits(tmp_path):
    package = build(tmp_path)
    text = render_report(package)
    exposures = package.record.exposures
    assert headings(text, 2) == HEADINGS
    markers = [
        *(trial.trial_id for trial in package.record.trials),
        *(event.id for event in exposures),
        *(artifact.name for artifact in package.artifact_records),
    ]
    for marker, heading in zip(markers, headings(text, 3), strict=True):
        assert f"`{marker}`" in heading
    for event in exposures:
        body = section(text, 3, f"`{event.id}`")
        assert {event.kind, event.actor, event.dataset_id, event.reason} <= {
            value for _, value in bullets(body)
        }
        for value in (event.at, event.interval_start, event.interval_end):
            assert f"`{value.isoformat()}`" in body
        assert f"`{event.material_digest}`" in body
        assert not any(other.reason in body for other in exposures if other != event)
    (material,) = [
        line
        for line in section(text, 3, f"`{exposures[0].id}`").splitlines()
        if exposures[0].material_digest in line
    ]
    assert "`input`" in material and "`input_copy`" in material
    for artifact in package.artifact_records:
        body = section(text, 3, f"`{artifact.name}`")
        assert {artifact.role, artifact.media_type, str(artifact.size)} <= {
            value for _, value in bullets(body)
        }
        assert f"`{artifact.digest}`" in body
    limits = section(text, 2, "Limits").lower()
    for claim in ("coverage: unknown", "complete historical disclosure", "coordinated rewrite"):
        assert claim in limits
    (confirmation,) = [line for line in limits.splitlines() if "confirmation" in line]
    assert "unavailable" in confirmation
    (investment,) = [line for line in limits.splitlines() if "investment" in line]
    assert "no claim" in investment
    assert not re.search(r"\b(final|winning|accepted|best)\b", text, re.IGNORECASE)


def test_report_states_the_package_disclosure_coverage(tmp_path):
    package = build(tmp_path)
    declared = save_package(
        tmp_path / "declared",
        replace(package.record, disclosure_coverage="declared_inputs"),
        package.artifacts(),
    )
    for verified, coverage in ((package, "unknown"), (declared, "declared inputs")):
        (line,) = [
            line
            for line in section(render_report(verified), 2, "Limits").splitlines()
            if "Disclosure coverage" in line
        ]
        assert line.startswith(f"- Disclosure coverage: {coverage}.")


def test_report_of_a_package_without_a_selected_trial_states_it_and_marks_no_trial(tmp_path, cli):
    package = build(tmp_path / "selected")
    unselected = save_package(
        tmp_path / "unselected", replace(package.record, trial_id=None), package.artifacts()
    )
    text = render_report(unselected)
    assert headings(text, 2) == HEADINGS
    assert "The package selects no trial." in text.split("\n## ", 1)[0].splitlines()
    assert bullets(section(text, 2, "Question")) == [
        ("Question", "none, because the package selects no trial")
    ]
    assert [line for line in headings(text, 3) if line.startswith("### Trial ")] == [
        f"### Trial {number}: `{trial.trial_id}`"
        for number, trial in enumerate(unselected.record.trials, 1)
    ]
    assert cli(tmp_path, "report", str(unselected.path)).stdout == text
    assert unselected.digest in cli(tmp_path, "verify", str(unselected.path)).stdout


def test_report_of_a_package_without_trials_says_none_were_supplied(tmp_path):
    package = build(tmp_path / "selected")
    empty = save_package(
        tmp_path / "empty",
        replace(package.record, trial_id=None, trials=()),
        package.artifacts(),
    )
    text = render_report(empty)
    assert headings(text, 2) == HEADINGS
    assert section(text, 2, "Trials").strip().splitlines() == [
        "## Trials",
        "",
        "No trials were supplied. This does not establish that no trial ran.",
    ]
    assert dict(bullets(section(text, 2, "Outcomes"))) == {
        "Trials": "0",
        "Attempts": "0",
        **dict.fromkeys(get_args(Status.__value__), "0"),
    }


def test_report_bytes_match_across_renders_copies_stdout_and_output_file(
    tmp_path, cli, monkeypatch
):
    package = build(tmp_path / "saved")
    text = render_report(package)
    assert text.endswith("\n") and not text.endswith("\n\n")
    assert render_report(package) == text
    copied = tmp_path / "elsewhere" / "renamed copy"
    shutil.copytree(package.path, copied)
    assert render_report(open_package(copied)) == text
    assert cli(tmp_path, "report", str(copied)).stdout == text
    output = tmp_path / "report.md"
    assert cli(tmp_path, "report", str(copied), "--output", str(output)).stdout == ""
    assert output.read_bytes() == text.encode()
    monkeypatch.setenv("LC_ALL", "en_US.ISO8859-1")
    assert cli(tmp_path, "report", str(copied)).stdout == text


@pytest.mark.smoke
@pytest.mark.parametrize("damage", ["artifact-byte", "rewritten-manifest"])
def test_damaged_package_fails_report_without_creating_output(tmp_path, cli, damage):
    package = build(tmp_path / "saved")
    copied = tmp_path / "damaged"
    shutil.copytree(package.path, copied)
    match damage:
        case "artifact-byte":
            path = copied / "artifacts" / package.artifact_records[0].digest
            content = bytearray(path.read_bytes())
            content[0] ^= 1
            path.write_bytes(content)
        case "rewritten-manifest":
            value = json.loads((copied / "manifest.json").read_bytes())
            value["record"]["trials"][-1]["specification"]["question"] = "Rewritten question"
            content = (
                json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
            (copied / "manifest.json").write_bytes(content)
            (copied / "manifest.sha256").write_text(hashlib.sha256(content).hexdigest() + "\n")
    with pytest.raises(PackageError):
        open_package(copied)
    output = tmp_path / "report.md"
    result = cli(tmp_path, "report", str(copied), "--output", str(output), success=False)
    assert result.returncode == 1
    assert "Package verification failed" in result.stderr
    assert result.stdout == ""
    assert not output.exists()


def test_each_pilot_evaluation_an_attempt_produced_renders_after_its_limitations(tmp_path):
    results = {"evaluation-a": SUPPORTED, "evaluation-b": UNBENCHMARKED, "invalid": INVALID}
    package = evaluated(
        tmp_path, {name: encode_evaluation(result) for name, result in results.items()}
    )
    text = render_report(package)
    attempt = package.record.trials[-1].attempts[-1]
    lines = section(text, 4, f"`{attempt.id}`").splitlines()
    assert [line for line in lines[lines.index("- Limitation: Wide interval") + 1 :] if line] == [
        "- Evaluation procedure: pilot-evaluation/v1 (artifact `evaluation-a`)",
        "- Evaluation parameters: minimum effect 0.001, subperiods 4, alpha 0.05",
        r"- Evaluation verdict: supported (above\_minimum)",
        "- Trials counted: 3, interval confidence 0.9916666666666667",
        "- Mean return: estimate 0.012345678901234568, standard error 0.0021, lower 0.0071, "
        "upper 0.0175",
        "- Incremental return over the baseline: estimate 0.0042, standard error 0.0011, "
        "lower 0.0015, upper 0.0069",
        "- Subperiod 1 estimate: 0.011",
        "- Subperiod 2 estimate: 0.0135",
        "- Subperiod 3 estimate: 0.0098",
        "- Subperiod 4 estimate: 0.0151",
        "- Evaluation procedure: pilot-evaluation/v1 (artifact `evaluation-b`)",
        "- Evaluation parameters: minimum effect 0.0025, subperiods 2, alpha 0.05",
        r"- Evaluation verdict: inconclusive (spans\_minimum)",
        "- Trials counted: 1, interval confidence 0.975",
        "- Mean return: estimate 0.002, standard error 0.0015, lower -0.001, upper 0.005",
        "- Incremental return over the baseline: none",
        "- Subperiod 1 estimate: 0.003",
        "- Subperiod 2 estimate: 0.001",
        "- Evaluation procedure: pilot-evaluation/v1 (artifact `invalid`)",
        "- Evaluation parameters: minimum effect 0.001, subperiods 2, alpha 0.1",
        r"- Invalid input reason: zero\_variance",
        r"- Invalid input detail: Returns are \*constant\* in every month",
    ]
    assert text.count("- Evaluation procedure: ") == len(results)


def test_a_malformed_or_changed_pilot_evaluation_fails_the_report_before_any_output(tmp_path, cli):
    malformed = evaluated(tmp_path / "malformed", {"evaluation": b"{}\n"})
    with pytest.raises(PackageError):
        render_report(malformed)
    output = tmp_path / "report.md"
    result = cli(tmp_path, "report", str(malformed.path), "--output", str(output), success=False)
    assert result.returncode == 1
    assert result.stdout == ""
    assert "Traceback" not in result.stderr
    assert not output.exists()
    data = encode_evaluation(SUPPORTED)
    changed = evaluated(tmp_path / "changed", {"evaluation": data})
    stored = changed.path / "artifacts" / hashlib.sha256(data).hexdigest()
    stored.write_bytes(encode_evaluation(UNBENCHMARKED))
    with pytest.raises(PackageError):
        render_report(changed)


def test_report_output_inside_the_package_is_rejected_and_the_package_still_verifies(tmp_path, cli):
    package = build(tmp_path / "saved")
    manifest = (package.path / "manifest.json").read_bytes()
    alias = tmp_path / "alias"
    alias.symlink_to(package.path)
    variant = package.path.with_name(package.path.name.upper())
    for output in (
        package.path / "report.md",
        package.path / "manifest.json",
        package.path / "artifacts" / "report.md",
        alias / "report.md",
        *([variant / "manifest.json"] if variant.exists() else []),
    ):
        result = cli(tmp_path, "report", str(package.path), "--output", str(output), success=False)
        assert "outside the package" in result.stderr
        assert result.stdout == ""
    linked = tmp_path / "linked.md"
    linked.hardlink_to(package.path / "manifest.json")
    assert cli(tmp_path, "report", str(package.path), "--output", str(linked)).stdout == ""
    assert linked.read_bytes() == render_report(package).encode()
    assert (package.path / "manifest.json").read_bytes() == manifest
    assert open_package(package.path).digest == package.digest
    directory = tmp_path / "directory"
    directory.mkdir()
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    # An open reader keeps a write into the FIFO from blocking the test.
    reader = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    locked = tmp_path / "locked"
    locked.mkdir(mode=0o500)
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    dangling = tmp_path / "dangling"
    dangling.symlink_to("newdir/")
    for output in (
        tmp_path / "missing" / "report.md",
        directory,
        f"{directory}/",
        f"{tmp_path / 'missing'}/",
        f"{tmp_path / 'missing'}/.",
        f"{tmp_path / 'missing'}/sub/..",
        f"{tmp_path / 'missing'}/../report.md",
        f"{linked}/../report.md",
        dangling,
        fifo,
        locked / "report.md",
        loop / "report.md",
        tmp_path / ("a" * 300),
        tmp_path / "null\0byte",
    ):
        result = cli(tmp_path, "report", str(package.path), "--output", str(output), success=False)
        assert "Cannot write report" in result.stderr
        assert "Traceback" not in result.stderr
    os.close(reader)
    assert fifo.is_fifo()
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "alias",
        "dangling",
        "directory",
        "fifo",
        "linked.md",
        "locked",
        "loop",
        "saved",
    ]


def test_report_output_errors_name_the_path_as_typed(tmp_path, cli):
    package = build(tmp_path / "saved")
    (tmp_path / "file.md").write_text("kept\n")
    (tmp_path / "locked").mkdir(mode=0o500)
    for output, problem in (("file.md/x", "Not a directory"), ("locked/r.md", "Permission denied")):
        result = cli(tmp_path, "report", str(package.path), "--output", output, success=False)
        assert result.stderr == f"Cannot write report to {output!r}: {problem}\n"
    assert (tmp_path / "file.md").read_text() == "kept\n"
    assert list((tmp_path / "locked").iterdir()) == []
    assert sorted(path.name for path in tmp_path.iterdir()) == ["file.md", "locked", "saved"]


def test_package_text_renders_inline_and_cannot_forge_report_structure(tmp_path):
    specials = "\\ ` * _ [ ] < > & | ~ $ \x1b \x07 \u202e"
    note = f"\n## Limits\n- Confirmation: available\r\n{specials}"
    plain = render_report(build(tmp_path / "plain"))
    package = build(tmp_path / "injected", note)
    text = render_report(package)
    assert structure(text) == structure(plain)
    assert not [
        char for char in text if unicodedata.category(char) in ("Cc", "Cf") and char != "\n"
    ]
    lines = text.splitlines()
    assert [line for line in lines if line.startswith("## Limits")] == ["## Limits"]
    assert not any(line.startswith("- Confirmation: available") for line in lines)
    assert f"available {specials}" not in text
    supplied = {
        value.removesuffix(note) for value in strings(asdict(package)) if value.endswith(note)
    }
    assert package.record.selected_trial.specification.question.removesuffix(note) in supplied
    escaped = r"\\ \` \* \_ \[ \] \< \> \& \| \~ \$ \x1b \x07 \u202e"
    for value in supplied:
        assert f"{value} ## Limits - Confirmation: available {escaped}" in text


def test_example_package_report_shows_its_inconclusive_outcome_and_no_supplied_exposures():
    package = open_package(ROOT / "examples" / "development-package")
    text = render_report(package)
    (attempt,) = package.record.selected_trial.attempts
    body = section(text, 4, f"`{attempt.id}`")
    assert "inconclusive" in body.splitlines()[0]
    assert {attempt.outcome.detail, *attempt.outcome.limitations} <= {
        value for _, value in bullets(body)
    }
    outcomes = dict(bullets(section(text, 2, "Outcomes")))
    assert {status: outcomes[status] for status in get_args(Status.__value__)} == {
        "successful": "0",
        "rejected": "0",
        "failed": "0",
        "null": "0",
        "inconclusive": "1",
    }
    exposures = section(text, 2, "Exposure history")
    assert "no exposure events were supplied" in exposures.lower()
    assert "coverage: unknown" in section(text, 2, "Limits").lower()


def test_campaign_report_renders_original_outcome_decisions_and_cutoff_without_temporal(
    tmp_path, cli
):
    from quant_research.records import canonical_bytes
    from quant_research.report import CAMPAIGN_ROLE

    snapshot = {
        "key": "campaign/run/final",
        "question": "What did the campaign ask?",
        "report": {
            "workflow_id": "campaign",
            "run_id": "run",
            "revision": "a" * 64,
            "steps": [],
            "ending": {"outcome": "cancelled", "reason": "researcher", "final": None},
            "assessment": None,
            "unrecorded": None,
            "steering": json.dumps(
                {"events": [{"actor": "sundeep", "action": "cancel", "reason": "Stop <now>"}]}
            ),
        },
        "original_head": "b" * 64,
        "recovered_head": "c" * 64,
        "acceptance": "Supplied acceptance log preserved verbatim",
        "acceptance_decisions": [
            {
                "attempt": "accepted-attempt",
                "actor": "reviewer",
                "reason": "Reviewed result",
                "at": "2026-10-23T19:00:00Z",
                "supersedes": None,
            },
            {
                "attempt": "replacement-attempt",
                "actor": "reviewer",
                "reason": "Reviewed replacement",
                "at": "2026-10-23T20:00:00Z",
                "supersedes": "accepted-attempt",
            },
        ],
        "cutoff_checks": [{"period": "held_back", "sentence": "The model cutoff is unknown."}],
    }
    package = save_package(
        tmp_path / "packages",
        PackageRecord(None, ()),
        (
            ArtifactInput(
                "campaign-snapshot", canonical_bytes(snapshot), "application/json", CAMPAIGN_ROLE
            ),
        ),
    )
    text = render_report(package)
    assert "Research ending: cancelled" in text
    assert "Ending reason: researcher" in text
    assert "cancel by sundeep: Stop \\<now\\>" in text
    assert "Human acceptance: Supplied acceptance log preserved verbatim" in text
    assert "Accepted attempt `accepted-attempt` by reviewer: Reviewed result" in text
    assert (
        "Accepted attempt `replacement-attempt` by reviewer: Reviewed replacement, "
        "at `2026-10-23T20:00:00Z`, supersedes attempt `accepted-attempt`"
    ) in text
    assert text.index("Accepted attempt `accepted-attempt`") < text.index(
        "Accepted attempt `replacement-attempt`"
    )
    assert "held\\_back: The model cutoff is unknown." in text
    assert "Original numerical head: " + "b" * 64 in text
    assert "Recovery snapshot head: " + "c" * 64 in text
    assert "confirmation unavailable" in text
    assert cli(tmp_path, "report", str(package.path)).stdout == text
