from __future__ import annotations

import hashlib
import shutil
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from uuid import uuid4

import pytest

from quant_research.compare import CompareError, Metric, Row, compare_packages, render_comparison
from quant_research.packages import (
    ArtifactInput,
    EnvironmentRecord,
    ExecutionAttempt,
    LineageLink,
    Outcome,
    PackageRecord,
    Specification,
    TrialRecord,
    save_package,
)
from quant_research.replay import EVALUATION_PROFILE, NewTrial, Repeat, replay_fixture

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/causal"
LOCK = (ROOT / "uv.lock").read_bytes()
QUESTION = "Describe the synthetic level-difference association"
TIME = datetime(2026, 10, 21, 12, tzinfo=UTC)
INDENT = " " * 15
ARTIFACTS = tuple(
    ArtifactInput(name, data, "text/plain", role)
    for name, data, role in (
        ("first", b"first input\n", "input"),
        ("second", b"second input\n", "input"),
        ("contract", b"synthetic only\n", "data contract"),
        ("code", b"hand-authored\n", "implementation"),
        ("lock", b"locked environment\n", "environment"),
        ("result", b'{"pairs": 3}\n', "diagnostic"),
    )
)
DIGESTS = {item.name: hashlib.sha256(item.data).hexdigest() for item in ARTIFACTS}
SPEC = Specification(
    "Preserve a declared development result",
    "hand-authored/v1",
    "generic-result/v1",
    "synthetic/v1",
    "test-authoring/v1",
    (DIGESTS["first"],),
    DIGESTS["contract"],
    DIGESTS["code"],
    EnvironmentRecord(
        "cpython", "3.12.13", "cpython-312", "test-platform", "uv test", DIGESTS["lock"]
    ),
)


def replay(tmp_path: Path, fixture="known-answer", intent=None):
    return replay_fixture(
        FIXTURES / f"{fixture}.json",
        tmp_path / "packages",
        intent=intent or NewTrial(QUESTION),
        lock=LOCK,
    )


def trial(spec=SPEC, links=(), statuses=("successful",)):
    attempts = tuple(
        ExecutionAttempt(
            uuid4(), uuid4(), None, TIME, TIME, Outcome(status, f"Supplied {status}", ("result",))
        )
        for status in statuses
    )
    return TrialRecord(uuid4(), uuid4(), spec, attempts, links)


def save(tmp_path: Path, *trials: TrialRecord, artifacts=ARTIFACTS):
    record = PackageRecord(trials[-1].trial_id, trials)
    return save_package(tmp_path / "packages", record, artifacts)


def sections(stdout: str) -> dict[str, list[str]]:
    blocks = stdout.removesuffix("\n").split("\n\n")
    return {lines[0]: lines[1:] for lines in (block.split("\n") for block in blocks)}


@pytest.mark.smoke
def test_package_against_itself_marks_every_row_same(tmp_path, cli):
    package = replay(tmp_path)
    comparison = compare_packages(package, package)
    assert comparison.differences == ()
    assert (comparison.relation, comparison.shared_trials) == (
        "same lineage",
        (package.record.trial_id,),
    )
    stdout = cli(tmp_path, "compare", str(package.path), str(package.path)).stdout
    assert "\nDifferences: none\n" in stdout
    shown = sections(stdout)
    rows = [
        line
        for title in ("Specification", "Inputs", "Versions", "Outcome", "Metrics")
        for line in shown[title]
    ]
    assert len(rows) == 3 + 4 + 7 + 7 + 7
    assert all(line.startswith("  same       ") for line in rows)


def test_known_answer_against_null_differs_in_exactly_the_changed_fields(tmp_path, cli):
    a, b = replay(tmp_path, "known-answer"), replay(tmp_path, "null")
    comparison = compare_packages(a, b)
    assert {row.name: row.mark for row in comparison.differences} == dict.fromkeys(
        (
            "data version",
            "input 1",
            "data contract",
            "status",
            "detail",
            "coefficient",
            "centered_cross",
            "label_variance",
        ),
        "different",
    )
    assert {
        metric.name: (metric.a, metric.b, metric.difference) for metric in comparison.metrics
    } == {
        "decisions": (3, 3, 0),
        "signals": (3, 3, 0),
        "pairs": (3, 3, 0),
        "coefficient": (1, 0, -1),
        "centered_cross": (2, 0, -2),
        "signal_variance": (2, 2, 0),
        "label_variance": (2, 6, 4),
    }
    assert (comparison.relation, comparison.shared_trials) == ("unrelated", ())
    stdout = cli(tmp_path, "compare", str(a.path), str(b.path)).stdout
    assert "\nDifferences: 8\n" in stdout
    assert f"  different  status\n{INDENT}A: successful\n{INDENT}B: null\n" in stdout
    assert (
        f"  different  coefficient\n{INDENT}A: 1\n{INDENT}B: 0\n{INDENT}difference: -1\n"
    ) in stdout
    assert "  same       signal_variance: 2\n" in stdout
    assert sections(stdout)["Lineage"] == ["  relation: unrelated", "  shared trials: none"]


def test_output_is_byte_identical_across_runs_copies_and_paths(tmp_path, cli):
    a, b = replay(tmp_path, "known-answer"), replay(tmp_path, "null")
    copy = tmp_path / "elsewhere" / "renamed"
    shutil.copytree(a.path, copy)
    relative = str(a.path.relative_to(tmp_path))
    outputs = {
        cli(tmp_path, "compare", first, str(b.path)).stdout
        for first in (str(a.path), str(a.path), str(copy), relative)
    }
    assert outputs == {render_comparison(compare_packages(a, b))}
    (stdout,) = outputs
    assert str(tmp_path) not in stdout
    assert relative not in stdout


@pytest.mark.parametrize("corrupt", [0, 1])
def test_corrupt_package_in_either_position_prints_nothing(tmp_path, cli, corrupt):
    paths = [replay(tmp_path, "known-answer").path, replay(tmp_path, "null").path]
    (paths[corrupt] / "manifest.sha256").write_text("0" * 64 + "\n")
    result = cli(tmp_path, "compare", *map(str, paths), success=False)
    assert (result.returncode, result.stdout) == (1, "")
    assert str(paths[corrupt]) in result.stderr
    assert "Manifest hash does not match manifest.sha256" in result.stderr


def test_repeat_differs_from_its_original_only_in_attempt_count(tmp_path):
    original = replay(tmp_path)
    repeated = replay(tmp_path, intent=Repeat(original))
    comparison = compare_packages(original, repeated)
    assert comparison.a.package != comparison.b.package
    assert [(row.name, row.a, row.b) for row in comparison.differences] == [("attempts", "1", "2")]
    assert (comparison.relation, comparison.shared_trials) == (
        "same lineage",
        (original.record.trial_id,),
    )


def test_linked_trial_in_another_lineage_with_an_added_input(tmp_path, cli):
    first = trial()
    second = trial(
        replace(SPEC, input_digests=(DIGESTS["first"], DIGESTS["second"])),
        (LineageLink(first.trial_id, SPEC.spec_id, "follows", "Adds a second input"),),
    )
    a, b = save(tmp_path, first), save(tmp_path, first, second)
    comparison = compare_packages(a, b)
    assert (comparison.relation, comparison.shared_trials) == ("linked", (first.trial_id,))
    assert [(row.name, row.mark) for row in comparison.differences] == [("input 2", "only in B")]
    assert [(row.name, row.mark) for row in compare_packages(b, a).differences] == [
        ("input 2", "only in A")
    ]
    assert comparison.metrics == ()
    shown = sections(cli(tmp_path, "compare", str(a.path), str(b.path)).stdout)
    assert shown["Inputs"][-2:] == ["  only in B  input 2", f"{INDENT}B: {DIGESTS['second']}"]
    assert shown["Metrics"] == ["  none"]
    assert shown["Lineage"] == ["  relation: linked", f"  shared trials: {first.trial_id}"]


def test_free_text_renders_on_one_line_with_control_characters_escaped(tmp_path, cli):
    a = save(tmp_path, trial())
    b = save(tmp_path, trial(replace(SPEC, question="Does the level\r\ndifference\nhold?\x1b")))
    stdout = cli(tmp_path, "compare", str(a.path), str(b.path)).stdout
    assert sections(stdout)["Specification"][:3] == [
        "  different  question",
        f"{INDENT}A: {SPEC.question}",
        f"{INDENT}B: Does the level difference hold?\\x1b",
    ]


def test_trial_without_attempts_shows_only_its_attempt_count(tmp_path):
    attempted = trial(replace(SPEC, evaluation_profile=EVALUATION_PROFILE))
    planned = replace(attempted, trial_id=uuid4(), lineage_id=uuid4(), attempts=())
    comparison = compare_packages(save(tmp_path, attempted), save(tmp_path, planned))
    assert comparison.outcome[0] == Row("attempts", "1", "0")
    assert [(row.name, row.mark) for row in comparison.differences] == [
        ("attempts", "different"),
        ("status", "only in A"),
        ("detail", "only in A"),
        ("pairs", "only in A"),
    ]


def test_outcome_uses_the_last_supplied_attempt(tmp_path):
    a = save(tmp_path, trial(statuses=("failed", "successful")))
    b = save(tmp_path, trial())
    differences = compare_packages(a, b).differences
    assert [(row.name, row.a, row.b) for row in differences] == [("attempts", "2", "1")]


@pytest.mark.parametrize("side", ["A", "B"])
def test_a_package_without_a_selected_trial_is_refused_naming_its_side(tmp_path, cli, side):
    selected = save(tmp_path, trial())
    unselected = save_package(
        tmp_path / "unselected", replace(selected.record, trial_id=None), ARTIFACTS
    )
    a, b = (unselected, selected) if side == "A" else (selected, unselected)
    with pytest.raises(CompareError, match=f"^Package {side} at .* selects no trial"):
        compare_packages(a, b)
    result = cli(tmp_path, "compare", str(a.path), str(b.path), success=False)
    assert (result.returncode, result.stdout) == (1, "")
    assert result.stderr.startswith(f"Comparison failed: Package {side} at {unselected.path} ")
    assert "selects no trial" in result.stderr


def test_metric_difference_is_exact_beyond_default_decimal_precision():
    a = Decimal("0.000000000000000000000000000000001")
    b = Decimal("1234567890123456789012345678901234")
    assert Fraction(Metric("coefficient", a, b).difference) == Fraction(b) - Fraction(a)


@pytest.mark.parametrize(
    "content, reason",
    [
        (b"not json\n", "is not JSON"),
        (b"[" * 100_000 + b"]" * 100_000, "is not JSON"),
        (b"[3]\n", "is not a JSON object"),
        (b'{"pairs": "abc"}\n', "has pairs 'abc'"),
        (b'{"coefficient": "NaN"}\n', "has coefficient 'NaN'"),
        (b'{"pairs": 0.5}\n', "has pairs 0.5"),
    ],
)
def test_diagnostic_outside_its_profile_is_refused_with_the_reason(tmp_path, cli, content, reason):
    artifacts = (*ARTIFACTS[:-1], replace(ARTIFACTS[-1], data=content))
    declared = trial(replace(SPEC, evaluation_profile=EVALUATION_PROFILE))
    package = save(tmp_path, declared, artifacts=artifacts)
    with pytest.raises(CompareError, match=reason):
        compare_packages(package, package)
    result = cli(tmp_path, "compare", str(package.path), str(package.path), success=False)
    assert (result.returncode, result.stdout) == (1, "")
    assert result.stderr.startswith("Comparison failed: Diagnostic 'result' in ")
    assert reason in result.stderr
