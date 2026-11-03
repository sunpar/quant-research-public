from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from quant_research.compare import (
    CompareError,
    Comparison,
    Identity,
    Metric,
    Row,
    compare_packages,
    render_comparison,
    render_comparison_json,
)
from quant_research.evaluation import VERSION, Evaluation, Interval, Invalid, Profile
from quant_research.packages import (
    ArtifactInput,
    EnvironmentRecord,
    ExecutionAttempt,
    LineageLink,
    Outcome,
    PackageRecord,
    Specification,
    TrialRecord,
    open_package,
    save_package,
)
from quant_research.records import PackageError, canonical_bytes
from quant_research.reference import EVALUATION_ROLE, encode_evaluation
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
PILOT = Evaluation(
    "above_minimum",
    7,
    0.975,
    Interval(0.1, 1e-30, -0.2, 0.30000000000000004),
    Interval(0.01, 0.002, -0.003, 0.04),
    (0.12, 0.13),
    Profile(0.005, 2, 0.05),
)


def replay(tmp_path: Path, fixture="known-answer", intent=None):
    return replay_fixture(
        FIXTURES / f"{fixture}.json",
        tmp_path / "packages",
        intent=intent or NewTrial(QUESTION),
        lock=LOCK,
    )


def trial(spec=SPEC, links=(), statuses=("successful",), *, results=("result",)):
    attempts = tuple(
        ExecutionAttempt(
            uuid4(), uuid4(), None, TIME, TIME, Outcome(status, f"Supplied {status}", results)
        )
        for status in statuses
    )
    return TrialRecord(uuid4(), uuid4(), spec, attempts, links)


def save(tmp_path: Path, *trials: TrialRecord, artifacts=ARTIFACTS):
    record = PackageRecord(trials[-1].trial_id, trials)
    return save_package(tmp_path / "packages", record, artifacts)


@pytest.fixture
def linked_history():
    trials, artifacts = [], list(ARTIFACTS)
    for name, pairs, inputs in (
        ("ancestor", 11, (DIGESTS["first"],)),
        ("middle", 22, (DIGESTS["second"],)),
        ("latest", 33, (DIGESTS["first"], DIGESTS["second"])),
    ):
        links = ()
        if trials:
            parent = trials[-1]
            links = (LineageLink(parent.trial_id, parent.specification.spec_id, "follows", name),)
        supplied = trial(
            replace(
                SPEC,
                question=f"{name} question",
                evaluation_profile=EVALUATION_PROFILE,
                implementation_version=f"{name}/v1",
                input_digests=inputs,
            ),
            links,
            statuses=("failed", "successful") if not trials else ("successful",),
        )
        last = replace(
            supplied.attempts[-1], outcome=Outcome("successful", f"{name} outcome", (name,))
        )
        trials.append(replace(supplied, attempts=(*supplied.attempts[:-1], last)))
        artifacts.append(
            ArtifactInput(
                name, f'{{"pairs": {pairs}}}\n'.encode(), "application/json", "diagnostic"
            )
        )
    return tuple(trials), tuple(artifacts)


@pytest.fixture
def pilot_results():
    second = replace(
        PILOT,
        reason="below_minimum",
        trials=11,
        confidence=0.99,
        mean=Interval(0.30000000000000004, 1e30, -0.1, 0.4),
        incremental=Interval(0.02, 0.003, -0.002, 0.05),
        subperiods=(0.22, 0.23),
        profile=Profile(0.02, 2, 0.1),
    )
    return tuple(
        ArtifactInput(name, encode_evaluation(result), "application/json", EVALUATION_ROLE)
        for name, result in (
            ("pilot-a", PILOT),
            ("pilot-b", second),
            ("unbenchmarked", replace(PILOT, incremental=None)),
            ("invalid", Invalid("zero_variance", "Constant\nreturns\x1b", PILOT.profile)),
        )
    )


def sections(stdout: str) -> dict[str, list[str]]:
    blocks = stdout.removesuffix("\n").split("\n\n")
    return {lines[0]: lines[1:] for lines in (block.split("\n") for block in blocks)}


def test_json_schema_preserves_exact_values_types_marks_and_order():
    first, second = UUID(int=1), UUID(int=2)
    raw = 'line\r\n\t\x1b\x00é "quoted" \\n'
    comparison = Comparison(
        Identity("a" * 64, first, second, "c" * 64),
        Identity("b" * 64, second, first, "d" * 64),
        specification=(Row("zeta", raw, raw), Row("alpha", "old", "new")),
        inputs=(Row("left", "", None), Row("right", None, "B")),
        versions=(Row("version", "v1", "v1"),),
        outcome=(Row("detail\nraw", "failed", "successful"),),
        metrics=(
            Metric("count", Decimal(3), Decimal(7)),
            Metric("scale", Decimal("1.00"), Decimal("1.0")),
            Metric("precision", Decimal("1E-31"), Decimal("1234567890123456789012345678901234")),
            Metric("only A", Decimal("0.30000000000000004"), None),
            Metric("only B", None, Decimal("1E-30")),
            Metric("exponent", Decimal("1E+30"), Decimal("2E+30")),
            Metric("signed zero", Decimal("-0.00"), Decimal("-0.0")),
        ),
        relation="linked",
        shared_trials=(second, first),
    )
    rendered = render_comparison_json(comparison)
    document = json.loads(rendered)
    assert set(document) == {
        "schema_version",
        "a",
        "b",
        "specification",
        "inputs",
        "versions",
        "outcome",
        "metrics",
        "difference_count",
        "relation",
        "shared_trials",
    }
    assert type(document["schema_version"]) is int and document["schema_version"] == 1
    assert type(document["difference_count"]) is int and document["difference_count"] == 9
    assert document["a"] == {
        "package": "a" * 64,
        "trial_id": "00000000-0000-0000-0000-000000000001",
        "lineage_id": "00000000-0000-0000-0000-000000000002",
        "spec_id": "c" * 64,
    }
    assert document["b"] == {
        "package": "b" * 64,
        "trial_id": "00000000-0000-0000-0000-000000000002",
        "lineage_id": "00000000-0000-0000-0000-000000000001",
        "spec_id": "d" * 64,
    }
    assert document["relation"] == "linked"
    assert document["shared_trials"] == [
        "00000000-0000-0000-0000-000000000002",
        "00000000-0000-0000-0000-000000000001",
    ]
    assert document["specification"] == [
        {"name": "zeta", "a": raw, "b": raw, "mark": "same"},
        {"name": "alpha", "a": "old", "b": "new", "mark": "different"},
    ]
    assert document["inputs"] == [
        {"name": "left", "a": "", "b": None, "mark": "only in A"},
        {"name": "right", "a": None, "b": "B", "mark": "only in B"},
    ]
    assert document["versions"] == [{"name": "version", "a": "v1", "b": "v1", "mark": "same"}]
    assert document["outcome"] == [
        {"name": "detail\nraw", "a": "failed", "b": "successful", "mark": "different"}
    ]
    assert [
        (row["name"], row["a"], row["b"], row["mark"], row["difference"])
        for row in document["metrics"]
    ] == [
        ("count", "3", "7", "different", "4"),
        ("scale", "1.00", "1.0", "same", "0.00"),
        (
            "precision",
            "1E-31",
            "1234567890123456789012345678901234",
            "different",
            "1234567890123456789012345678901233.9999999999999999999999999999999",
        ),
        ("only A", "0.30000000000000004", None, "only in A", None),
        ("only B", None, "1E-30", "only in B", None),
        ("exponent", "1E+30", "2E+30", "different", "1E+30"),
        ("signed zero", "-0.00", "-0.0", "same", "0.00"),
    ]
    for section in ("specification", "inputs", "versions", "outcome", "metrics"):
        assert type(document[section]) is list
        for row in document[section]:
            assert set(row) == {"name", "a", "b", "mark"} | (
                {"difference"} if section == "metrics" else set()
            )
            assert type(row["name"]) is type(row["mark"]) is str
            assert row["a"] is None or type(row["a"]) is str
            assert row["b"] is None or type(row["b"]) is str
    for row in document["metrics"]:
        difference = row["difference"]
        assert difference is None or type(difference) is str
        if difference is not None:
            assert Fraction(difference) == Fraction(row["b"]) - Fraction(row["a"])
    assert rendered == render_comparison_json(comparison)
    assert rendered == json.dumps(document, indent=2, sort_keys=True) + "\n"
    assert rendered.endswith("}\n") and not rendered.endswith("\n\n")
    assert "é" not in rendered and "\\u00e9" in rendered


def test_json_zero_difference_count_does_not_assert_package_identity(tmp_path, cli):
    a, b = save(tmp_path, trial()), save(tmp_path, trial())
    result = cli(tmp_path, "compare", str(a.path), str(b.path), "--format", "json")
    document = json.loads(result.stdout)
    assert document["difference_count"] == 0
    assert document["a"]["package"] != document["b"]["package"]
    assert document["a"]["trial_id"] != document["b"]["trial_id"]
    assert document["a"]["lineage_id"] != document["b"]["lineage_id"]
    assert document["relation"] == "unrelated" and document["shared_trials"] == []
    assert document["metrics"] == []
    default = cli(tmp_path, "compare", str(a.path), str(b.path), text=False)
    explicit = cli(tmp_path, "compare", str(a.path), str(b.path), "--format", "text", text=False)
    assert default.stdout == explicit.stdout == render_comparison(compare_packages(a, b)).encode()
    assert result.stderr == "" and default.stderr == explicit.stderr == b""


def test_compare_format_help_and_invalid_choice_precede_package_reads(tmp_path, cli):
    assert "--format {text,json}" in cli(tmp_path, "compare", "--help").stdout
    result = cli(tmp_path, "compare", "missing-a", "missing-b", "--format", "yaml", success=False)
    assert (result.returncode, result.stdout) == (2, "")
    assert "--format" in result.stderr and "invalid choice: 'yaml'" in result.stderr
    assert "Package verification failed" not in result.stderr


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


@pytest.mark.parametrize("selected", [True, False])
@pytest.mark.parametrize("format_args", [(), ("--format", "text"), ("--format", "json")])
def test_output_is_byte_identical_across_runs_copies_and_paths(
    tmp_path, cli, linked_history, pilot_results, selected, format_args
):
    trials, artifacts = linked_history
    amended = []
    for supplied, names in zip(trials[:2], (("pilot-a", "invalid"), ("pilot-b",)), strict=True):
        attempt = supplied.attempts[-1]
        attempt = replace(
            attempt,
            outcome=replace(
                attempt.outcome, result_artifacts=(*attempt.outcome.result_artifacts, *names)
            ),
        )
        amended.append(replace(supplied, attempts=(*supplied.attempts[:-1], attempt)))
    trials = (*amended, trials[-1])
    artifacts = (*artifacts, *pilot_results)
    selection = trials[-1].trial_id if selected else None
    package = save_package(tmp_path / "packages", PackageRecord(selection, trials), artifacts)
    record, digest = package.record, package.digest
    before = {
        path.relative_to(package.path): path.read_bytes()
        for path in package.path.rglob("*")
        if path.is_file()
    }
    copy = tmp_path / "elsewhere" / "renamed"
    shutil.copytree(package.path, copy)
    relative = str(package.path.relative_to(tmp_path))
    outputs = {
        cli(
            tmp_path,
            "compare",
            first,
            str(package.path),
            "--trial-a",
            str(trials[0].trial_id),
            "--trial-b",
            str(trials[1].trial_id),
            "--evaluation-a",
            "pilot-a",
            "--evaluation-b",
            "pilot-b",
            *format_args,
            text=False,
        ).stdout
        for first in (str(package.path), str(package.path), str(copy), relative)
    }
    comparison = compare_packages(
        package,
        package,
        trial_a=trials[0].trial_id,
        trial_b=trials[1].trial_id,
        evaluation_a="pilot-a",
        evaluation_b="pilot-b",
    )
    assert comparison == compare_packages(
        package,
        package,
        trial_a=trials[0].trial_id,
        trial_b=trials[1].trial_id,
        evaluation_a="pilot-a",
    )
    assert comparison.metrics[0] == Metric("pairs", Decimal(11), Decimal(22))
    assert comparison.metrics[1] == Metric("pilot trial count", Decimal(7), Decimal(11))
    assert Row("pilot artifact", "pilot-a", "pilot-b") in comparison.outcome
    assert (comparison.relation, comparison.shared_trials) == ("linked", (trials[0].trial_id,))
    renderer = render_comparison_json if format_args == ("--format", "json") else render_comparison
    assert outputs == {renderer(comparison).encode()}
    if format_args == ("--format", "json"):
        document = json.loads(next(iter(outputs)))
        assert document["a"]["trial_id"] == str(trials[0].trial_id)
        assert document["b"]["trial_id"] == str(trials[1].trial_id)
        assert document["metrics"][:2] == [
            {"name": "pairs", "a": "11", "b": "22", "mark": "different", "difference": "11"},
            {
                "name": "pilot trial count",
                "a": "7",
                "b": "11",
                "mark": "different",
                "difference": "4",
            },
        ]
        assert document["metrics"][3]["difference"] == "0.20000000000000004"
    assert (comparison.a.trial_id, comparison.b.trial_id) == (
        trials[0].trial_id,
        trials[1].trial_id,
    )
    (stdout,) = outputs
    assert str(tmp_path).encode() not in stdout
    assert relative.encode() not in stdout
    assert (package.record, package.digest) == (record, digest)
    for path in (package.path, copy):
        reopened = open_package(path)
        assert (reopened.record, reopened.digest) == (record, digest)
        assert reopened.record.trial_id == selection
        assert {
            file.relative_to(path): file.read_bytes() for file in path.rglob("*") if file.is_file()
        } == before


def test_explicit_trials_control_identity_rows_metrics_and_ancestry(tmp_path, cli, linked_history):
    trials, artifacts = linked_history
    package = save(tmp_path, *trials, artifacts=artifacts)
    ancestor, middle, _ = trials
    comparison = compare_packages(
        package, package, trial_a=ancestor.trial_id, trial_b=middle.trial_id
    )
    assert comparison.a.package == comparison.b.package == package.digest
    assert (comparison.a.trial_id, comparison.a.lineage_id, comparison.a.spec_id) == (
        ancestor.trial_id,
        ancestor.lineage_id,
        ancestor.specification.spec_id,
    )
    assert (comparison.b.trial_id, comparison.b.lineage_id, comparison.b.spec_id) == (
        middle.trial_id,
        middle.lineage_id,
        middle.specification.spec_id,
    )
    assert comparison.specification[0] == Row("question", "ancestor question", "middle question")
    assert comparison.inputs[-1] == Row("input 1", DIGESTS["first"], DIGESTS["second"])
    assert comparison.versions[0] == Row("implementation version", "ancestor/v1", "middle/v1")
    assert comparison.outcome == (
        Row("attempts", "2", "1"),
        Row("status", "successful", "successful"),
        Row("detail", "ancestor outcome", "middle outcome"),
    )
    assert comparison.metrics == (Metric("pairs", Decimal(11), Decimal(22)),)
    assert (comparison.relation, comparison.shared_trials) == ("linked", (ancestor.trial_id,))
    result = cli(
        tmp_path,
        "compare",
        str(package.path),
        str(package.path),
        "--trial-a",
        str(ancestor.trial_id),
        "--trial-b",
        str(middle.trial_id),
    )
    assert result.stdout == render_comparison(comparison)


def test_explicit_middle_trial_keeps_all_of_its_ancestors(tmp_path, linked_history):
    trials, artifacts = linked_history
    package = save(tmp_path, *trials, artifacts=artifacts)
    ancestor, middle, latest = trials
    comparison = compare_packages(
        package, package, trial_a=middle.trial_id, trial_b=latest.trial_id
    )
    assert (comparison.relation, comparison.shared_trials) == (
        "linked",
        tuple(sorted((ancestor.trial_id, middle.trial_id), key=str)),
    )


@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize("output_format", ["text", "json"])
def test_one_explicit_trial_keeps_the_other_sides_selection(
    tmp_path, cli, linked_history, side, output_format
):
    trials, artifacts = linked_history
    package = save(tmp_path, *trials, artifacts=artifacts)
    comparison = compare_packages(package, package, **{f"trial_{side}": trials[0].trial_id})
    expected = (trials[0].trial_id, trials[2].trial_id)
    values = (Decimal(11), Decimal(33))
    if side == "b":
        expected, values = expected[::-1], values[::-1]
    assert (comparison.a.trial_id, comparison.b.trial_id) == expected
    assert comparison.metrics == (Metric("pairs", *values),)
    assert comparison.shared_trials == (trials[0].trial_id,)
    result = cli(
        tmp_path,
        "compare",
        str(package.path),
        str(package.path),
        f"--trial-{side}",
        str(trials[0].trial_id),
        "--format",
        output_format,
    )
    renderer = render_comparison_json if output_format == "json" else render_comparison
    assert result.stdout == renderer(comparison)


@pytest.mark.parametrize("corrupt, format_args", [(0, ()), (1, ()), (0, ("--format", "json"))])
def test_corrupt_package_in_either_position_prints_nothing(tmp_path, cli, corrupt, format_args):
    paths = [replay(tmp_path, "known-answer").path, replay(tmp_path, "null").path]
    (paths[corrupt] / "manifest.sha256").write_text("0" * 64 + "\n")
    result = cli(tmp_path, "compare", *map(str, paths), *format_args, success=False)
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
    document = json.loads(
        cli(tmp_path, "compare", str(a.path), str(b.path), "--format", "json").stdout
    )
    assert document["specification"][0]["b"] == "Does the level\r\ndifference\nhold?\x1b"


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


@pytest.mark.parametrize("side, format_args", [("A", ()), ("B", ()), ("B", ("--format", "json"))])
def test_a_package_without_a_selected_trial_is_refused_naming_its_side(
    tmp_path, cli, side, format_args
):
    selected = save(tmp_path, trial())
    unselected = save_package(
        tmp_path / "unselected", replace(selected.record, trial_id=None), ARTIFACTS
    )
    a, b = (unselected, selected) if side == "A" else (selected, unselected)
    reason = (
        f"Package {side} at {unselected.path} selects no trial; "
        "compare needs a selected trial in each package"
    )
    with pytest.raises(CompareError) as error:
        compare_packages(a, b)
    assert str(error.value) == reason
    with pytest.raises(CompareError) as error:
        compare_packages(a, b, **{f"trial_{'b' if side == 'A' else 'a'}": selected.record.trial_id})
    assert str(error.value) == reason
    result = cli(tmp_path, "compare", str(a.path), str(b.path), *format_args, success=False)
    assert (result.returncode, result.stdout) == (1, "")
    assert result.stderr == f"Comparison failed: {reason}\n"


@pytest.mark.parametrize("side, format_args", [("a", ()), ("b", ()), ("a", ("--format", "json"))])
def test_explicit_uuid_must_belong_to_its_own_side(
    tmp_path, cli, linked_history, side, format_args
):
    trials, artifacts = linked_history
    history = save(tmp_path, *trials, artifacts=artifacts)
    other = save(tmp_path, trial())
    a, b = (other, history) if side == "a" else (history, other)
    requested = trials[0].trial_id
    reason = f"Package {side.upper()} at {other.path} has no trial {requested}"
    with pytest.raises(CompareError) as error:
        compare_packages(a, b, **{f"trial_{side}": requested})
    assert str(error.value) == reason
    result = cli(
        tmp_path,
        "compare",
        str(a.path),
        str(b.path),
        f"--trial-{side}",
        str(requested),
        *format_args,
        success=False,
    )
    assert (result.returncode, result.stdout, result.stderr) == (
        1,
        "",
        f"Comparison failed: {reason}\n",
    )


def test_empty_history_refuses_selection_and_resolves_side_a_first(tmp_path, cli):
    package = save_package(tmp_path / "packages", PackageRecord(None, ()), ())
    with pytest.raises(CompareError) as error:
        compare_packages(package, package)
    assert str(error.value) == (
        f"Package A at {package.path} selects no trial; "
        "compare needs a selected trial in each package"
    )
    requested = uuid4()
    reason = f"Package A at {package.path} has no trial {requested}"
    with pytest.raises(CompareError) as error:
        compare_packages(package, package, trial_a=requested, trial_b=uuid4())
    assert str(error.value) == reason
    result = cli(
        tmp_path,
        "compare",
        str(package.path),
        str(package.path),
        "--trial-a",
        str(requested),
        success=False,
    )
    assert (result.returncode, result.stdout, result.stderr) == (
        1,
        "",
        f"Comparison failed: {reason}\n",
    )


@pytest.mark.parametrize("option", ["--trial-a", "--trial-b"])
def test_malformed_trial_uuid_is_rejected_before_opening_packages(tmp_path, cli, option):
    result = cli(tmp_path, "compare", "missing-a", "missing-b", option, "not-a-uuid", success=False)
    assert (result.returncode, result.stdout) == (2, "")
    assert option in result.stderr
    assert "invalid UUID value" in result.stderr
    assert "not-a-uuid" in result.stderr
    assert "Package verification failed" not in result.stderr


def test_metric_difference_is_exact_beyond_default_decimal_precision():
    a = Decimal("0.000000000000000000000000000000001")
    b = Decimal("1234567890123456789012345678901234")
    assert Fraction(Metric("coefficient", a, b).difference) == Fraction(b) - Fraction(a)


@pytest.mark.parametrize(
    "content, reason, format_args",
    [
        (b"not json\n", "is not JSON", ()),
        (b"[" * 100_000 + b"]" * 100_000, "is not JSON", ()),
        (b"[3]\n", "is not a JSON object", ()),
        (b'{"pairs": "abc"}\n', "has pairs 'abc'", ()),
        (b'{"coefficient": "NaN"}\n', "has coefficient 'NaN'", ()),
        (b'{"pairs": 0.5}\n', "has pairs 0.5", ()),
        (b'{"pairs": 0.5}\n', "has pairs 0.5", ("--format", "json")),
    ],
)
def test_diagnostic_outside_its_profile_is_refused_with_the_reason(
    tmp_path, cli, content, reason, format_args
):
    artifacts = (*ARTIFACTS[:-1], replace(ARTIFACTS[-1], data=content))
    declared = trial(replace(SPEC, evaluation_profile=EVALUATION_PROFILE))
    package = save(tmp_path, declared, artifacts=artifacts)
    with pytest.raises(CompareError, match=reason):
        compare_packages(package, package)
    result = cli(
        tmp_path, "compare", str(package.path), str(package.path), *format_args, success=False
    )
    assert (result.returncode, result.stdout) == (1, "")
    assert result.stderr.startswith("Comparison failed: Diagnostic 'result' in ")
    assert reason in result.stderr


@pytest.mark.parametrize("profile", [VERSION, SPEC.evaluation_profile])
def test_singleton_pilots_show_recorded_fields_and_exact_decimals(
    tmp_path, cli, pilot_results, profile
):
    spec = replace(SPEC, evaluation_profile=profile)
    artifacts = (*ARTIFACTS, *pilot_results)
    a = save(tmp_path, trial(spec, results=("pilot-a",)), artifacts=artifacts)
    b = save(tmp_path, trial(spec, results=("pilot-b",)), artifacts=artifacts)
    comparison = compare_packages(a, b)
    assert comparison.outcome[3:] == (
        Row("pilot artifact", "pilot-a", "pilot-b"),
        Row("pilot kind", "evaluation", "evaluation"),
        Row("pilot procedure", "pilot-evaluation/v1", "pilot-evaluation/v1"),
        Row("pilot minimum effect", "0.005", "0.02"),
        Row("pilot subperiod count", "2", "2"),
        Row("pilot alpha", "0.05", "0.1"),
        Row("pilot verdict", "supported", "null"),
        Row("pilot reason", "above_minimum", "below_minimum"),
        Row("pilot incremental interval", "present", "present"),
        Row("pilot subperiod estimates by recorded position", "(0.12, 0.13)", "(0.22, 0.23)"),
    )
    assert comparison.metrics == tuple(
        Metric(name, Decimal(a), Decimal(b))
        for name, a, b in (
            ("pilot trial count", "7", "11"),
            ("pilot confidence", "0.975", "0.99"),
            ("pilot mean estimate", "0.1", "0.30000000000000004"),
            ("pilot mean standard error", "1e-30", "1e30"),
            ("pilot mean lower", "-0.2", "-0.1"),
            ("pilot mean upper", "0.30000000000000004", "0.4"),
            ("pilot incremental estimate", "0.01", "0.02"),
            ("pilot incremental standard error", "0.002", "0.003"),
            ("pilot incremental lower", "-0.003", "-0.002"),
            ("pilot incremental upper", "0.04", "0.05"),
        )
    )
    for metric in comparison.metrics:
        assert Fraction(metric.difference) == Fraction(metric.b) - Fraction(metric.a)
    assert comparison.metrics[2].difference == Decimal("0.20000000000000004")
    result = cli(tmp_path, "compare", str(a.path), str(b.path))
    assert result.stdout == render_comparison(comparison)
    assert (
        "pilot subperiod estimates by recorded position" in sections(result.stdout)["Outcome"][-3]
    )
    help_text = cli(tmp_path, "compare", "--help").stdout
    assert "--evaluation-a NAME" in help_text and "--evaluation-b NAME" in help_text


@pytest.mark.parametrize("side", ["a", "b"])
def test_pilot_ambiguity_is_sorted_and_explicit_choice_ignores_malformed_siblings(
    tmp_path, cli, pilot_results, side
):
    malformed = ArtifactInput("z-malformed", b"{}\n", "application/json", EVALUATION_ROLE)
    artifacts = (*ARTIFACTS, *pilot_results, malformed)
    ambiguous = trial(results=("z-malformed", "pilot-a", "invalid"))
    other = save(tmp_path, trial(results=("pilot-b",)), artifacts=artifacts)
    projections = []
    for reverse in (False, True):
        attempt = ambiguous.attempts[-1]
        if reverse:
            attempt = replace(
                attempt,
                outcome=replace(
                    attempt.outcome, result_artifacts=("invalid", "pilot-a", "z-malformed")
                ),
            )
        package = save(
            tmp_path,
            replace(ambiguous, attempts=(attempt,)),
            artifacts=artifacts[::-1] if reverse else artifacts,
        )
        a, b = (package, other) if side == "a" else (other, package)
        reason = (
            f"Package {side.upper()} at {package.path} trial {ambiguous.trial_id} has multiple "
            "pilot-evaluation result artifacts in its last attempt; "
            f"choose --evaluation-{side} NAME from: 'invalid', 'pilot-a', 'z-malformed'"
        )
        with pytest.raises(CompareError) as error:
            compare_packages(a, b)
        assert str(error.value) == reason
        output_format = "json" if side == "b" else "text"
        refused = cli(
            tmp_path, "compare", str(a.path), str(b.path), "--format", output_format, success=False
        )
        assert (refused.returncode, refused.stdout, refused.stderr) == (
            1,
            "",
            f"Comparison failed: {reason}\n",
        )
        comparison = compare_packages(a, b, **{f"evaluation_{side}": "pilot-a"})
        projections.append((comparison.outcome, comparison.metrics))
        result = cli(
            tmp_path,
            "compare",
            str(a.path),
            str(b.path),
            f"--evaluation-{side}",
            "pilot-a",
            "--format",
            output_format,
        )
        renderer = render_comparison_json if output_format == "json" else render_comparison
        assert result.stdout == renderer(comparison)
    assert projections[0] == projections[1]


@pytest.mark.parametrize("side", ["a", "b"])
def test_pilot_choices_must_belong_to_the_chosen_trials_last_attempt(
    tmp_path, cli, pilot_results, side
):
    artifacts = (*ARTIFACTS, *pilot_results, replace(pilot_results[0], name="earlier"))
    earlier = trial(results=("earlier",))
    chosen = trial(results=("pilot-a", "result"))
    chosen = replace(chosen, attempts=(*earlier.attempts, *chosen.attempts))
    other_trial = trial(results=("pilot-b",))
    package = save(tmp_path, chosen, other_trial, artifacts=artifacts)
    other = save(tmp_path, trial(results=("pilot-b",)), artifacts=artifacts)
    a, b = (package, other) if side == "a" else (other, package)
    for name in ("missing", "result", "earlier", "pilot-b", ""):
        reason = (
            f"Package {side.upper()} at {package.path} trial {chosen.trial_id} has no eligible "
            f"pilot-evaluation result artifact {name!r} in its last attempt; "
            "eligible artifacts: 'pilot-a'"
        )
        with pytest.raises(CompareError) as error:
            compare_packages(a, b, **{f"trial_{side}": chosen.trial_id, f"evaluation_{side}": name})
        assert str(error.value) == reason
        refused = cli(
            tmp_path,
            "compare",
            str(a.path),
            str(b.path),
            f"--trial-{side}",
            str(chosen.trial_id),
            f"--evaluation-{side}",
            name,
            "--format",
            "json" if side == "b" else "text",
            success=False,
        )
        assert (refused.returncode, refused.stdout, refused.stderr) == (
            1,
            "",
            f"Comparison failed: {reason}\n",
        )
    planned = replace(chosen, attempts=())
    empty = save(tmp_path, planned, artifacts=artifacts)
    a, b = (empty, other) if side == "a" else (other, empty)
    with pytest.raises(CompareError) as error:
        compare_packages(a, b, **{f"evaluation_{side}": "pilot-a"})
    assert str(error.value) == (
        f"Package {side.upper()} at {empty.path} trial {planned.trial_id} has no eligible "
        "pilot-evaluation result artifact 'pilot-a' in its last attempt; eligible artifacts: none"
    )
    refused = cli(
        tmp_path,
        "compare",
        str(a.path),
        str(b.path),
        f"--evaluation-{side}",
        "pilot-a",
        success=False,
    )
    assert (refused.returncode, refused.stdout, refused.stderr) == (
        1,
        "",
        f"Comparison failed: {error.value}\n",
    )


def test_pilot_absence_invalid_and_missing_incremental_keep_numbers_absent(
    tmp_path, cli, pilot_results
):
    artifacts = (*ARTIFACTS, *pilot_results)
    valid = save(tmp_path, trial(results=("pilot-a",)), artifacts=artifacts)
    unbenchmarked = save(tmp_path, trial(results=("unbenchmarked",)), artifacts=artifacts)
    invalid = save(tmp_path, trial(results=("invalid",)), artifacts=artifacts)
    earlier = trial(results=("pilot-a",))
    failed = trial(statuses=("failed",))
    absent = save(
        tmp_path,
        replace(earlier, attempts=(*earlier.attempts, *failed.attempts)),
        artifacts=artifacts,
    )
    comparison = compare_packages(valid, unbenchmarked)
    assert Row("pilot incremental interval", "present", "absent") in comparison.outcome
    assert all(metric.b is None and metric.difference is None for metric in comparison.metrics[-4:])
    assert comparison.metrics[2] == Metric("pilot mean estimate", Decimal("0.1"), Decimal("0.1"))
    for a, b in ((valid, invalid), (invalid, valid), (valid, absent), (absent, valid)):
        comparison = compare_packages(a, b)
        assert len(comparison.metrics) == 10
        assert all(metric.difference is None for metric in comparison.metrics)
        assert all((metric.a is None) != (metric.b is None) for metric in comparison.metrics)
    invalid_only = compare_packages(invalid, invalid)
    assert invalid_only.metrics == ()
    assert invalid_only.outcome[3:] == (
        Row("pilot artifact", "invalid", "invalid"),
        Row("pilot kind", "invalid", "invalid"),
        Row("pilot procedure", "pilot-evaluation/v1", "pilot-evaluation/v1"),
        Row("pilot minimum effect", "0.005", "0.005"),
        Row("pilot subperiod count", "2", "2"),
        Row("pilot alpha", "0.05", "0.05"),
        Row("pilot reason", "zero_variance", "zero_variance"),
        Row("pilot invalid detail", "Constant\nreturns\x1b", "Constant\nreturns\x1b"),
    )
    stdout = cli(tmp_path, "compare", str(invalid.path), str(invalid.path)).stdout
    assert "  same       pilot invalid detail: Constant returns\\x1b\n" in stdout
    no_pilot = compare_packages(absent, absent)
    assert not any(row.name.startswith("pilot ") for row in no_pilot.outcome)
    assert no_pilot.metrics == ()
    last_failed = save(
        tmp_path, trial(statuses=("failed",), results=("pilot-a",)), artifacts=artifacts
    )
    assert compare_packages(valid, last_failed).metrics == compare_packages(valid, valid).metrics


def test_selected_pilot_decode_and_post_open_integrity_errors_remain_package_errors(
    tmp_path, cli, pilot_results
):
    malformed = ArtifactInput(
        "malformed",
        canonical_bytes({"kind": "evaluation", "result": {}, "schema_version": 1}),
        "application/json",
        EVALUATION_ROLE,
    )
    package = save(tmp_path, trial(results=("malformed",)), artifacts=(*ARTIFACTS, malformed))
    with pytest.raises(PackageError, match="evaluation") as error:
        compare_packages(package, package)
    for format_args in ((), ("--format", "json")):
        result = cli(
            tmp_path, "compare", str(package.path), str(package.path), *format_args, success=False
        )
        assert (result.returncode, result.stdout, result.stderr) == (
            1,
            "",
            f"Package verification failed: {error.value}\n",
        )
    for changed in ("artifact", "manifest"):
        package = save(
            tmp_path / changed, trial(results=("pilot-a",)), artifacts=(*ARTIFACTS, *pilot_results)
        )
        path = package.path / "manifest.json"
        if changed == "artifact":
            artifact = next(item for item in package.artifact_records if item.name == "pilot-a")
            path = package.path / "artifacts" / artifact.digest
        path.write_bytes(path.read_bytes() + b"\n")
        with pytest.raises(PackageError):
            compare_packages(package, package)
        result = cli(tmp_path, "compare", str(package.path), str(package.path), success=False)
        assert (result.returncode, result.stdout) == (1, "")
        assert result.stderr.startswith("Package verification failed: ")
        assert "Traceback" not in result.stderr
