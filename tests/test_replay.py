from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import (
    ROUND_DOWN,
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DefaultContext,
    Inexact,
    InvalidOperation,
    getcontext,
    localcontext,
)
from fractions import Fraction
from pathlib import Path
from uuid import UUID, uuid4

import pytest

import quant_research.replay as replay_module
from quant_research.packages import ArtifactInput, open_package, save_package
from quant_research.records import PackageError
from quant_research.replay import FixtureError, NewTrial, Repeat, ReplayError, Retry, replay_fixture

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/causal"
LOCK = (ROOT / "uv.lock").read_bytes()
QUESTION = "Describe the synthetic level-difference association"
TIME_KEYS = {
    "occurred_at",
    "evaluation_cutoff",
    "known_at",
    "published_at",
    "arrived_at",
    "at",
    "decision_at",
    "available_at",
}


def timestamps(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in TIME_KEYS:
                yield datetime.fromisoformat(item)
            else:
                yield from timestamps(item)
    elif isinstance(value, list):
        for item in value:
            yield from timestamps(item)


def replay(tmp_path: Path, fixture="known-answer", *, intent=None, lock=LOCK):
    if isinstance(fixture, dict):
        source = tmp_path / f"fixture-{uuid4()}.json"
        source.write_text(json.dumps(fixture) + "\n")
    else:
        source = FIXTURES / f"{fixture}.json"
    return replay_fixture(
        source, tmp_path / "packages", intent=intent or NewTrial(QUESTION), lock=lock
    )


def output(package, role: str) -> bytes:
    result_names = package.record.selected_trial.attempts[-1].outcome.result_artifacts
    name = next(
        item.name
        for item in package.artifact_records
        if item.role == role and item.name in result_names
    )
    return package.read_artifact(name)


def data(package, role: str):
    return json.loads(output(package, role))


@pytest.fixture
def document():
    return json.loads((FIXTURES / "known-answer.json").read_bytes())


@pytest.mark.smoke
@pytest.mark.parametrize(
    "fixture, labels, expected_cross, expected_y_variance, coefficient, status",
    [
        ("known-answer", (-1, 0, 1), 2, 2, 1, "successful"),
        ("null", (1, -2, 1), 0, 6, 0, "null"),
    ],
)
def test_full_fixture_to_portable_diagnostic_matches_independent_terms(
    tmp_path, cli, fixture, labels, expected_cross, expected_y_variance, coefficient, status
):
    saved = replay(tmp_path, fixture)
    package = open_package(saved.path)
    signals = data(package, "signals")
    assert [Decimal(row["value"]) for row in signals] == [-1, 0, 1]
    assert signals[0]["current"] == {"entity_id": "A", "session_id": "s1", "revision": 2}
    assert signals[0]["previous"] == {"entity_id": "A", "session_id": "s0", "revision": 1}
    x, y = tuple(map(Fraction, (-1, 0, 1))), tuple(map(Fraction, labels))
    centered_x = tuple(value - sum(x) / len(x) for value in x)
    centered_y = tuple(value - sum(y) / len(y) for value in y)
    cross = sum(a * b for a, b in zip(centered_x, centered_y, strict=True))
    variance_x = sum(value * value for value in centered_x)
    variance_y = sum(value * value for value in centered_y)
    assert (cross, variance_x, variance_y) == (expected_cross, 2, expected_y_variance)
    diagnostic = data(package, "diagnostic")
    assert diagnostic["pairs"] == diagnostic["signals"] == diagnostic["decisions"] == 3
    assert Decimal(diagnostic["centered_cross"]) == cross
    assert Decimal(diagnostic["signal_variance"]) == variance_x
    assert Decimal(diagnostic["label_variance"]) == variance_y
    assert Decimal(diagnostic["coefficient"]) == coefficient
    assert (
        diagnostic["status"] == package.record.selected_trial.attempts[-1].outcome.status == status
    )
    assert [Fraction(pair["label"]) for pair in data(package, "evaluation")["pairs"]] == list(y)
    inspected = json.loads(cli(tmp_path, "inspect", str(package.path)).stdout)
    assert inspected["record"]["trials"][-1]["attempts"][-1]["outcome"]["status"] == status
    assert "Verified" in cli(tmp_path, "verify", str(package.path)).stdout


@pytest.mark.parametrize("labels", [(1, 0, -1), (-2, 0, 1)])
def test_nonzero_descriptive_coefficient_uses_fraction_oracle(tmp_path, document, labels):
    for row, label in zip(document["labels"], labels, strict=True):
        row["value"] = str(label)
    package = replay(tmp_path, document)
    x = tuple(map(Fraction, (-1, 0, 1)))
    y = tuple(map(Fraction, labels))
    dx = tuple(value - sum(x) / 3 for value in x)
    dy = tuple(value - sum(y) / 3 for value in y)
    cross = sum(a * b for a, b in zip(dx, dy, strict=True))
    exact_squared = cross**2 / (sum(value**2 for value in dx) * sum(value**2 for value in dy))
    diagnostic = data(package, "diagnostic")
    with localcontext() as context:
        context.prec = 80
        coefficient = Decimal(diagnostic["coefficient"])
        expected = Decimal(exact_squared.numerator) / Decimal(exact_squared.denominator)
        assert abs(coefficient**2 - expected) < Decimal("1e-32")
    assert (coefficient < 0) == (cross < 0)
    assert diagnostic["status"] == "successful"


@pytest.mark.parametrize("case", ["signal", "coefficient", "rejection"])
def test_decimal_profile_ignores_callers_arithmetic_context(tmp_path, document, case):
    if case == "coefficient":
        for row, value in zip(document["labels"], ("1", "-2", "1." + "0" * 32 + "1"), strict=True):
            row["value"] = value
    else:
        document["observations"][0]["level"] = (
            "10.00000000000000000000000000000001" if case == "signal" else "nonsense"
        )
    baseline = replay(tmp_path, document)
    with localcontext() as context:
        context.prec = 3
        context.rounding = ROUND_DOWN
        context.capitals = 0
        context.Emin, context.Emax = -9, 9
        context.clamp = 1
        context.traps[Inexact] = True
        context.traps[InvalidOperation] = False
        changed_context = replay(tmp_path, document)
    if case == "signal":
        assert data(baseline, "signals")[0]["value"] == "-1E-32"
    elif case == "coefficient":
        assert "E-" in data(baseline, "diagnostic")["coefficient"]
    else:
        assert data(baseline, "diagnostic")["code"] == "invalid_number"
    for role in ("signals", "provenance", "evaluation", "diagnostic"):
        assert output(baseline, role) == output(changed_context, role)
    assert (
        baseline.record.selected_trial.specification
        == changed_context.record.selected_trial.specification
    )
    assert (
        baseline.record.selected_trial.attempts[-1].outcome.detail
        == changed_context.record.selected_trial.attempts[-1].outcome.detail
    )


@pytest.mark.parametrize(
    "fixture, reason, pairs",
    [("zero-variance", "zero_variance", 3), ("insufficient-pairs", "insufficient_pairs", 2)],
)
def test_inconclusive_cases_have_distinct_reasons_and_no_coefficient(
    tmp_path, fixture, reason, pairs
):
    package = replay(tmp_path, fixture)
    diagnostic = data(package, "diagnostic")
    assert diagnostic["status"] == "inconclusive"
    assert diagnostic["reason"] == reason
    assert diagnostic["pairs"] == pairs
    assert "coefficient" not in diagnostic
    assert package.record.selected_trial.attempts[-1].outcome.status == "inconclusive"


def test_zero_signal_variance_is_inconclusive(tmp_path, document):
    for row in document["observations"]:
        row["level"] = "10"
    package = replay(tmp_path, document)
    assert [Decimal(row["value"]) for row in data(package, "signals")] == [0, 0, 0]
    assert data(package, "diagnostic")["reason"] == "zero_variance"
    assert "coefficient" not in data(package, "diagnostic")


def test_large_constant_labels_do_not_acquire_variance_from_rounding(tmp_path, document):
    for row in document["labels"]:
        row["value"] = "9999999999999999999999999999999999"
    package = replay(tmp_path, document)
    assert data(package, "diagnostic")["reason"] == "zero_variance"
    assert "coefficient" not in data(package, "diagnostic")


def test_large_shared_level_offset_preserves_small_known_differences(tmp_path, document):
    for row in document["observations"]:
        with localcontext() as context:
            context.prec = 80
            row["level"] = str(Decimal(row["level"]) + Decimal("100000000000000000000000000000000"))
    package = replay(tmp_path, document)
    assert [Decimal(row["value"]) for row in data(package, "signals")] == [-1, 0, 1]
    assert Decimal(data(package, "diagnostic")["coefficient"]) == 1


@pytest.mark.parametrize("sign", [1, -1])
def test_near_perfect_correlation_stays_within_pearson_range(tmp_path, document, sign):
    levels = ("-234653", "-1340689", "677661.8571428571428571428571428571")
    for row in document["observations"]:
        if row["session_id"] == "s0":
            row["level"] = "0"
        elif row["session_id"] == "s1":
            row["level"] = levels[document["entities"].index(row["entity_id"])]
    with localcontext(Context(prec=34, rounding=ROUND_HALF_EVEN)):
        for row, level in zip(document["labels"], levels, strict=True):
            row["value"] = str(
                sign * Decimal(level) * Decimal("74.76923076923076923076923076923077")
            )
    package = replay(tmp_path, document)
    assert [Decimal(row["value"]) for row in data(package, "signals")] == list(map(Decimal, levels))
    pairs = data(package, "evaluation")["pairs"]
    x, y = (
        tuple(Fraction(pair["signal"]) for pair in pairs),
        tuple(Fraction(pair["label"]) for pair in pairs),
    )
    cross = 3 * sum(a * b for a, b in zip(x, y, strict=True)) - sum(x) * sum(y)
    variance_x = 3 * sum(value**2 for value in x) - sum(x) ** 2
    variance_y = 3 * sum(value**2 for value in y) - sum(y) ** 2
    expected_squared = cross**2 / (variance_x * variance_y)
    diagnostic = data(package, "diagnostic")
    coefficient = Decimal(diagnostic["coefficient"])
    assert 0 < coefficient * sign <= 1
    with localcontext() as context:
        context.prec = 80
        expected = Decimal(expected_squared.numerator) / Decimal(expected_squared.denominator)
        assert abs(coefficient**2 - expected) < Decimal("1e-32")
    assert diagnostic["status"] == "successful"


def test_extreme_decimal_width_preserves_exact_null(tmp_path, document):
    levels = {"A": "-1e100", "B": "0", "C": "1e100"}
    for row in document["observations"]:
        if row["session_id"] == "s0":
            row["level"] = "0"
        elif row["session_id"] == "s1":
            row["level"] = levels[row["entity_id"]]
    for row, value in zip(document["labels"], ("1e-100", "-2e-100", "1e-100"), strict=True):
        row["value"] = value
    diagnostic = data(replay(tmp_path, document), "diagnostic")
    assert diagnostic["status"] == "null"
    assert Decimal(diagnostic["coefficient"]) == Decimal(diagnostic["centered_cross"]) == 0
    assert Decimal(diagnostic["signal_variance"]) == Decimal("2e200")
    assert Decimal(diagnostic["label_variance"]) == Decimal("6e-200")


def test_default_decimal_template_cannot_change_declared_profile(tmp_path, document, monkeypatch):
    for row, value in zip(document["labels"], ("-2", "0", "1"), strict=True):
        row["value"] = value
    baseline = replay(tmp_path, document)
    monkeypatch.setattr(DefaultContext, "rounding", ROUND_DOWN)
    monkeypatch.setattr(DefaultContext, "prec", 2)
    monkeypatch.setattr(DefaultContext, "Emax", 9)
    monkeypatch.setitem(DefaultContext.traps, Inexact, True)

    def run_with_fresh_context():
        context = getcontext()
        assert context.prec == 2 and context.rounding == ROUND_DOWN
        assert context.Emax == 9 and context.traps[Inexact]
        return replay(tmp_path, document)

    with ThreadPoolExecutor(max_workers=1) as executor:
        changed = executor.submit(run_with_fresh_context).result()
    assert output(baseline, "signals") == output(changed, "signals")
    assert output(baseline, "diagnostic") == output(changed, "diagnostic")


def test_future_information_cannot_change_prior_signals_or_selected_versions(tmp_path, document):
    baseline = replay(tmp_path, document)
    for row in document["observations"]:
        if row["arrived_at"] > "2026-01-06T17:00:00Z":
            row["level"] = "-9999"
    document["observations"].append(
        dict(
            document["observations"][2],
            revision=20,
            published_at="2026-01-08T12:00:00Z",
            level="8000",
        )
    )
    document["memberships"][-1]["included"] = True
    for row in document["labels"]:
        row["value"] = "8000"
    changed = replay(tmp_path, document)
    assert output(baseline, "signals") == output(changed, "signals")
    assert output(baseline, "provenance") == output(changed, "provenance")
    assert data(changed, "diagnostic")["reason"] == "zero_variance"


@pytest.mark.parametrize("fixture", sorted(path.stem for path in FIXTURES.glob("*.json")))
def test_exposures_span_raw_fixture_times_unless_rows_are_rejected(tmp_path, fixture):
    raw = (FIXTURES / f"{fixture}.json").read_bytes()
    document = json.loads(raw)
    package = replay(tmp_path, fixture)
    if fixture == "invalid":
        assert data(package, "diagnostic")["status"] == "rejected"
        times, coverage = list(timestamps(document["contract"])), "unknown"
    else:
        times, coverage = list(timestamps(document)), "declared_inputs"
    assert [
        (event.kind, event.interval_start, event.interval_end) for event in package.record.exposures
    ] == [("data", min(times), max(times)), ("result", min(times), max(times))]
    assert package.record.exposures[0].material_digest == hashlib.sha256(raw).hexdigest()
    assert package.record.disclosure_coverage == coverage


@pytest.mark.parametrize("row", ["revision", "membership", "label"])
def test_exposure_interval_covers_rows_the_replay_never_selects(tmp_path, document, row):
    baseline = replay(tmp_path, document)
    late = "2026-01-09T12:00:00Z"
    match row:
        case "revision":
            document["observations"].append(
                dict(document["observations"][2], revision=4, published_at=late, arrived_at=late)
            )
        case "membership":
            document["memberships"].append(
                dict(document["memberships"][1], known_at=late, included=False)
            )
        case "label":
            document["labels"][1]["available_at"] = late
    changed = replay(tmp_path, document)
    assert {(event.interval_start, event.interval_end) for event in changed.record.exposures} == {
        (datetime(2026, 1, 4, 12, tzinfo=UTC), datetime.fromisoformat(late))
    }
    assert output(baseline, "signals") == output(changed, "signals")
    assert output(baseline, "provenance") == output(changed, "provenance")


def test_history_without_declared_input_coverage_leaves_coverage_unknown(tmp_path):
    first = replay(tmp_path)
    unknown = save_package(
        tmp_path / "unknown",
        replace(first.record, disclosure_coverage="unknown"),
        first.artifacts(),
    )
    linked = replay(tmp_path, intent=NewTrial(QUESTION, (first,)))
    assert linked.record.disclosure_coverage == "declared_inputs"
    for intent in (NewTrial(QUESTION, (first, unknown)), Repeat(unknown)):
        assert replay(tmp_path, intent=intent).record.disclosure_coverage == "unknown"


@pytest.mark.parametrize("field", ["arrived_at", "published_at"])
def test_moving_availability_across_decision_changes_only_eligible_signal(
    tmp_path, document, field
):
    row = document["observations"][3]
    row["published_at"] = row["arrived_at"] = "2026-01-06T17:00:00Z"
    row[field] = "2026-01-06T17:00:00.000001Z"
    before = replay(tmp_path, document)
    row[field] = "2026-01-06T17:00:00Z"
    after = replay(tmp_path, document)
    assert Decimal(data(before, "signals")[0]["value"]) == -1
    assert Decimal(data(after, "signals")[0]["value"]) == 889
    assert data(after, "signals")[0]["current"]["revision"] == 3
    assert data(before, "signals")[1:] == data(after, "signals")[1:]


def test_late_older_revision_cannot_replace_higher_revision(tmp_path, document):
    document["observations"][1]["arrived_at"] = "2026-01-06T17:00:00.000001Z"
    before = replay(tmp_path, document)
    document["observations"][1]["arrived_at"] = "2026-01-06T17:00:00Z"
    after = replay(tmp_path, document)
    assert output(before, "signals") == output(after, "signals")
    assert output(before, "provenance") == output(after, "provenance")
    assert data(after, "signals")[0]["current"]["revision"] == 2


@pytest.mark.parametrize(
    "change, reason",
    [
        ("known_at", "membership_unavailable"),
        ("effective", "membership_missing"),
        ("excluded", "membership_excluded"),
    ],
)
def test_membership_requires_effective_session_and_known_at(tmp_path, document, change, reason):
    row = document["memberships"][2]
    if change == "known_at":
        row["known_at"] = "2026-01-06T17:00:00.000001Z"
    elif change == "effective":
        row["first_session"] = "s2"
    else:
        row["included"] = False
    package = replay(tmp_path, document)
    assert [row["decision"]["entity_id"] for row in data(package, "signals")] == ["A", "B"]
    assert data(package, "provenance")[2]["reasons"] == [reason]
    assert data(package, "diagnostic")["reason"] == "insufficient_pairs"


def test_latest_known_effective_membership_wins_including_exact_cutoff(tmp_path, document):
    document["memberships"][-1]["known_at"] = "2026-01-06T17:00:00Z"
    package = replay(tmp_path, document)
    assert data(package, "provenance")[0]["reasons"] == ["membership_excluded"]
    assert data(package, "provenance")[0]["membership"]["known_at"] == "2026-01-06T17:00:00Z"


A_BASELINE = ([], ("2026-01-04T12:00:00Z", "s0"), ("s1", 2), ("s0", 1), -1)
SUPERSEDED = (["membership_superseded"], None, None, None, None)


@pytest.mark.parametrize(
    "rows, decisions, expected",
    [
        pytest.param(
            [("s0", "s0", "2026-01-07T12:00:00Z", True)],
            [("s1", "2026-01-07T13:00:00Z")],
            {"2026-01-06T17:00:00Z": A_BASELINE, "2026-01-07T13:00:00Z": SUPERSEDED},
            id="shortened-vintage",
        ),
        pytest.param(
            [
                ("s0", "s0", "2026-01-07T12:00:00Z", True),
                ("s2", "s2", "2026-01-07T12:00:00Z", True),
            ],
            [("s1", "2026-01-07T13:00:00Z"), ("s2", "2026-01-07T17:30:00Z")],
            {
                "2026-01-06T17:00:00Z": A_BASELINE,
                "2026-01-07T13:00:00Z": SUPERSEDED,
                "2026-01-07T17:30:00Z": (
                    [],
                    ("2026-01-07T12:00:00Z", "s2"),
                    ("s2", 1),
                    ("s1", 3),
                    -400,
                ),
            },
            id="vintage-with-gap",
        ),
        pytest.param(
            [("s1", "s2", "2026-01-07T12:00:00Z", False)],
            [("s0", "2026-01-07T13:00:00Z"), ("s1", "2026-01-07T13:30:00Z")],
            {
                "2026-01-06T17:00:00Z": A_BASELINE,
                "2026-01-07T13:00:00Z": SUPERSEDED,
                "2026-01-07T13:30:00Z": (
                    ["membership_excluded"],
                    ("2026-01-07T12:00:00Z", "s1"),
                    None,
                    None,
                    None,
                ),
            },
            id="exclusion-only-vintage",
        ),
        pytest.param(
            [
                ("s0", "s0", "2026-01-07T12:00:00Z", True),
                ("s1", "s2", "2026-01-07T12:00:00Z", False),
            ],
            [("s0", "2026-01-07T13:00:00Z"), ("s1", "2026-01-07T13:30:00Z")],
            {
                "2026-01-06T17:00:00Z": A_BASELINE,
                "2026-01-07T13:00:00Z": (
                    ["previous_session_missing"],
                    ("2026-01-07T12:00:00Z", "s0"),
                    ("s0", 1),
                    None,
                    None,
                ),
                "2026-01-07T13:30:00Z": (
                    ["membership_excluded"],
                    ("2026-01-07T12:00:00Z", "s1"),
                    None,
                    None,
                    None,
                ),
            },
            id="vintage-member-beside-exclusion",
        ),
        pytest.param(
            [
                ("s0", "s0", "2026-01-06T12:00:00Z", True),
                ("s0", "s2", "2026-01-07T12:00:00Z", True),
            ],
            [],
            {"2026-01-06T17:00:00Z": SUPERSEDED},
            id="coverage-returns-after-decision",
        ),
    ],
)
def test_latest_known_vintage_supersedes_older_session_coverage(
    tmp_path, document, rows, decisions, expected
):
    document["memberships"][-1:] = [
        {
            "entity_id": "A",
            "first_session": first,
            "last_session": last,
            "known_at": known_at,
            "included": included,
        }
        for first, last, known_at, included in rows
    ]
    document["decisions"] += [
        {"entity_id": "A", "session_id": session, "at": at} for session, at in decisions
    ]
    package = replay(tmp_path, document)
    signals = {
        (row["decision"]["entity_id"], row["decision"]["at"]): Decimal(row["value"])
        for row in data(package, "signals")
    }
    outcomes = {
        row["decision"]["at"]: (
            row["reasons"],
            row["membership"]
            and (row["membership"]["known_at"], row["membership"]["first_session"]),
            row["current"] and (row["current"]["session_id"], row["current"]["revision"]),
            row["previous"] and (row["previous"]["session_id"], row["previous"]["revision"]),
            signals.get(("A", row["decision"]["at"])),
        )
        for row in data(package, "provenance")
        if row["decision"]["entity_id"] == "A"
    }
    assert outcomes == expected


def test_every_selection_names_the_vintage_and_a_superseded_one_names_what_dropped_it(
    tmp_path, document
):
    document["memberships"][-1]["first_session"] = "s2"
    late = {"entity_id": "A", "session_id": "s1", "at": "2026-01-07T13:00:00Z"}
    document["decisions"].append(late)
    rows = data(replay(tmp_path, document), "provenance")
    superseded = next(row for row in rows if row["decision"] == late)
    assert superseded["reasons"] == ["membership_superseded"]
    assert (superseded["membership"], superseded["vintage"]) == (None, "2026-01-07T12:00:00Z")
    members = [row for row in rows if row is not superseded]
    assert [row["vintage"] for row in members] == [row["membership"]["known_at"] for row in members]


def test_missing_predecessor_never_falls_back_to_older_session(tmp_path, document):
    document["contract"]["sessions"].insert(
        0, {"id": "older", "occurred_at": "2026-01-04T16:00:00Z"}
    )
    document["observations"][0]["session_id"] = "older"
    package = replay(tmp_path, document)
    selected = data(package, "provenance")[0]
    assert selected["current"]["revision"] == 2
    assert selected["previous"] is None
    assert selected["reasons"] == ["previous_missing"]
    assert [row["decision"]["entity_id"] for row in data(package, "signals")] == ["B", "C"]


@pytest.mark.parametrize(
    "session, reason", [("s0", "previous_session_missing"), ("s2", "current_session_not_occurred")]
)
def test_session_must_have_occurred_and_have_declared_predecessor(
    tmp_path, document, session, reason
):
    document["decisions"].append(
        {"entity_id": "C", "session_id": session, "at": "2026-01-06T16:50:00Z"}
    )
    document["observations"].append(dict(document["observations"][7], session_id="s2", level="200"))
    package = replay(tmp_path, document)
    selected = next(
        row
        for row in data(package, "provenance")
        if row["decision"]["at"] == "2026-01-06T16:50:00Z"
    )
    assert reason in selected["reasons"]
    assert len(data(package, "signals")) == 3


def test_observations_can_be_missing_or_unavailable_without_invalidating_fixture(
    tmp_path, document
):
    document["observations"][0]["arrived_at"] = "2026-01-06T17:00:00.000001Z"
    document["observations"] = [
        row
        for row in document["observations"]
        if not (row["entity_id"] == "B" and row["session_id"] == "s1")
    ]
    package = replay(tmp_path, document)
    assert data(package, "provenance")[0]["reasons"] == ["previous_unavailable"]
    assert data(package, "provenance")[1]["reasons"] == ["current_missing"]
    assert data(package, "diagnostic")["pairs"] == 1
    assert data(package, "diagnostic")["status"] == "inconclusive"


def test_input_row_order_cannot_change_computation_outputs(tmp_path, document):
    baseline = replay(tmp_path, document)
    generator = random.Random(40)
    for name in ("entities", "memberships", "observations", "decisions", "labels"):
        generator.shuffle(document[name])
    generator.shuffle(document["contract"]["sessions"])
    reordered = replay(tmp_path, document)
    for role in ("signals", "provenance", "evaluation", "diagnostic"):
        assert output(baseline, role) == output(reordered, role)


def test_labels_join_by_entity_decision_and_only_at_evaluation_cutoff(tmp_path, document):
    baseline = replay(tmp_path, document)
    document["labels"].reverse()
    document["labels"][0]["available_at"] = "2026-01-07T18:00:00.000001Z"
    document["labels"][1]["available_at"] = "2026-01-07T18:00:00Z"
    package = replay(tmp_path, document)
    assert output(baseline, "signals") == output(package, "signals")
    assert output(baseline, "provenance") == output(package, "provenance")
    evaluation = data(package, "evaluation")
    assert [pair["label"] for pair in evaluation["pairs"]] == ["-1", "0"]
    assert evaluation["labels"][2]["reason"] == "label_unavailable"
    assert data(package, "diagnostic")["pairs"] == 2


@pytest.mark.parametrize(
    "damage, code",
    [
        ("unknown-entity", "unknown_entity"),
        ("unknown-session", "unknown_session"),
        ("duplicate-entity", "duplicate_entity"),
        ("duplicate-observation", "duplicate_observation"),
        ("duplicate-decision", "duplicate_decision"),
        ("duplicate-label", "duplicate_label"),
        ("ambiguous-membership", "ambiguous_membership"),
        ("membership-type", "invalid_membership"),
        ("reversed-membership", "invalid_membership"),
        ("unit", "invalid_unit"),
        ("label-unit", "invalid_unit"),
        ("time", "invalid_time"),
        ("future-decision", "invalid_decision_time"),
        ("label-decision", "unknown_decision"),
        ("label-at-decision", "invalid_label_time"),
        ("revision", "invalid_revision"),
        ("boolean-revision", "invalid_revision"),
        ("number-type", "invalid_number"),
        ("array", "invalid_array"),
        ("fields", "invalid_fields"),
    ],
)
def test_invalid_rows_preserve_rejected_attempt_and_raw_bytes(tmp_path, document, damage, code):
    match damage:
        case "unknown-entity":
            document["observations"][0]["entity_id"] = "unknown"
        case "unknown-session":
            document["observations"][0]["session_id"] = "unknown"
        case "duplicate-entity":
            document["entities"].append("A")
        case "duplicate-observation":
            document["observations"].append(document["observations"][0])
        case "duplicate-decision":
            document["decisions"].append(document["decisions"][0])
        case "duplicate-label":
            document["labels"].append(document["labels"][0])
        case "ambiguous-membership":
            document["memberships"].append(document["memberships"][0])
        case "membership-type":
            document["memberships"][0]["included"] = "true"
        case "reversed-membership":
            document["memberships"][0]["first_session"] = "s2"
            document["memberships"][0]["last_session"] = "s0"
        case "unit":
            document["observations"][0]["unit"] = "USD"
        case "label-unit":
            document["labels"][0]["unit"] = "USD"
        case "time":
            document["observations"][0]["arrived_at"] = "2026-01-05T16:05:00"
        case "future-decision":
            document["decisions"][0]["at"] = "2026-01-10T12:00:00Z"
        case "label-decision":
            document["labels"][0]["decision_at"] = "2026-01-06T16:00:00Z"
        case "label-at-decision":
            document["labels"][0]["available_at"] = document["labels"][0]["decision_at"]
        case "revision":
            document["observations"][0]["revision"] = 0
        case "boolean-revision":
            document["observations"][0]["revision"] = True
        case "number-type":
            document["observations"][0]["level"] = 10.0
        case "array":
            document["decisions"] = None
        case "fields":
            document["unknown"] = True
    source = tmp_path / "invalid.json"
    raw = (json.dumps(document, indent=4) + "\n").encode()
    source.write_bytes(raw)
    package = replay_fixture(source, tmp_path / "packages", intent=NewTrial(QUESTION), lock=LOCK)
    diagnostic = data(package, "diagnostic")
    assert diagnostic["status"] == "rejected"
    assert diagnostic["code"] == code
    assert "coefficient" not in diagnostic
    assert data(package, "signals") == []
    assert data(package, "provenance") == []
    assert next(item.data for item in package.artifacts() if item.role == "input") == raw
    assert package.record.selected_trial.attempts[-1].outcome.status == "rejected"


@pytest.mark.parametrize(
    "value, code",
    [
        *[
            (value, "invalid_number")
            for value in ("+1", "01", ".5", " 1", "1_0", "1٢", "0.٥", "1e٢")
        ],
        *[(value, "nonfinite_number") for value in ("NaN", "sNaN", "-Infinity")],
        *[
            (value, "number_out_of_profile")
            for value in ("1e101", "1e-101", "12e100", "0e101", "1" * 35)
        ],
    ],
)
def test_invalid_numbers_keep_syntax_finiteness_and_bounds_distinct(
    tmp_path, document, value, code
):
    document["observations"][0]["level"] = value
    package = replay(tmp_path, document)
    diagnostic = data(package, "diagnostic")
    assert diagnostic["status"] == "rejected" and diagnostic["code"] == code
    assert "coefficient" not in diagnostic
    assert data(package, "signals") == []
    raw = next(item.data for item in package.artifacts() if item.role == "input")
    assert json.loads(raw)["observations"][0]["level"] == value


@pytest.mark.parametrize(
    "raw", [b"\xff", b"{", b"[]", b"{}", b'{"contract":{},"contract":{}}', b'{"contract":NaN}']
)
def test_unestablished_contract_is_boundary_error_without_trial(tmp_path, raw):
    source = tmp_path / "invalid.json"
    source.write_bytes(raw)
    destination = tmp_path / "packages"
    with pytest.raises(FixtureError):
        replay_fixture(source, destination, intent=NewTrial(QUESTION), lock=LOCK)
    assert not destination.exists()


@pytest.mark.parametrize("row", ['"level": NaN', '"level": "10", "level": "11"'])
def test_non_json_rows_cannot_establish_a_contract(tmp_path, document, row):
    document["observations"][0]["level"] = "replace-me"
    raw = json.dumps(document).replace('"level": "replace-me"', row).encode()
    source = tmp_path / "invalid.json"
    source.write_bytes(raw)
    destination = tmp_path / "packages"
    with pytest.raises(FixtureError) as error:
        replay_fixture(source, destination, intent=NewTrial(QUESTION), lock=LOCK)
    assert error.value.code == "invalid_json"
    assert not destination.exists()


@pytest.mark.parametrize(
    "change", ["nonsynthetic", "version", "precision", "units", "time", "calendar"]
)
def test_invalid_declared_contract_cannot_invent_trial(tmp_path, document, change):
    contract = document["contract"]
    match change:
        case "nonsynthetic":
            contract["designation"] = "production"
        case "version":
            contract["version"] = True
        case "precision":
            contract["decimal_precision"] = 28
        case "units":
            contract["level_unit"] = "USD"
        case "time":
            contract["evaluation_cutoff"] = "2026-01-07T18:00:00-05:00"
        case "calendar":
            contract["sessions"].append(contract["sessions"][0])
    with pytest.raises(FixtureError):
        replay(tmp_path, document)
    assert not (tmp_path / "packages").exists()


@pytest.fixture
def execution_clock(monkeypatch):
    class Clock(datetime):
        current = datetime(2026, 10, 21, 12, tzinfo=UTC)

        @classmethod
        def now(cls, tz=None):
            return cls.current.astimezone(tz)

    read_bytes = Path.read_bytes
    diagnostic = replay_module._diagnostic

    def read_fixture(path):
        raw = read_bytes(path)
        if path == FIXTURES / "known-answer.json":
            Clock.current += timedelta(minutes=1)
        return raw

    def compute_diagnostic(pairs):
        result = diagnostic(pairs)
        Clock.current += timedelta(minutes=1)
        return result

    monkeypatch.setattr(replay_module, "datetime", Clock)
    monkeypatch.setattr(Path, "read_bytes", read_fixture)
    monkeypatch.setattr(replay_module, "_diagnostic", compute_diagnostic)
    return Clock


def test_attempt_and_exposures_preserve_read_and_computation_times(tmp_path, execution_clock):
    started_at = execution_clock.current
    package = replay(tmp_path)
    attempt = package.record.selected_trial.attempts[-1]
    exposures = {event.kind: event for event in package.record.exposures}
    assert attempt.started_at == started_at
    assert exposures["data"].at == started_at + timedelta(minutes=1)
    assert attempt.finished_at == execution_clock.current == started_at + timedelta(minutes=2)
    assert exposures["result"].at == attempt.finished_at
    assert attempt.started_at < exposures["data"].at < exposures["result"].at


def test_retry_predecessor_must_finish_before_start_even_if_read_finishes_later(
    tmp_path, execution_clock
):
    original = replay(tmp_path)
    predecessor = original.record.selected_trial.attempts[-1]
    started_at = execution_clock.current = predecessor.finished_at - timedelta(seconds=30)
    with pytest.raises(PackageError, match="predecessor finishes"):
        replay(tmp_path, intent=Retry(original, predecessor.id))
    assert started_at < predecessor.finished_at < execution_clock.current
    assert open_package(original.path).record == original.record
    assert len(list((tmp_path / "packages").glob("*/*/manifest.json"))) == 1


def test_repeat_retry_and_linked_new_trial_keep_complete_history_and_old_artifacts(
    tmp_path, document
):
    first = replay(tmp_path)
    repeated = replay(tmp_path, intent=Repeat(first))
    retried = replay(tmp_path, intent=Retry(repeated, first.record.selected_trial.attempts[0].id))
    first_attempt, second_attempt, third_attempt = retried.record.selected_trial.attempts
    assert first_attempt == first.record.selected_trial.attempts[0]
    assert first_attempt.run_id == third_attempt.run_id != second_attempt.run_id
    assert third_attempt.predecessor == first_attempt.id
    assert len({attempt.id for attempt in retried.record.selected_trial.attempts}) == 3
    assert first.record.trial_id == repeated.record.trial_id == retried.record.trial_id
    assert first.record.selected_trial.specification == retried.record.selected_trial.specification
    for role in ("signals", "provenance", "evaluation", "diagnostic"):
        assert output(first, role) == output(repeated, role) == output(retried, role)
    other = replay(tmp_path, "null")
    disclosed = replace(
        first.record.exposures[0],
        id=uuid4(),
        actor="supplied researcher",
        reason="Supplied prior exposure",
    )
    document["labels"][0]["value"] = "2"
    changed = replay(
        tmp_path,
        document,
        intent=NewTrial(
            "A linked question", (first, retried, other), (disclosed,), "Changed synthetic target"
        ),
    )
    assert len(changed.record.trials) == 3
    assert changed.record.trials[0] == retried.record.selected_trial
    assert changed.record.trials[1] == other.record.selected_trial
    assert (
        changed.record.selected_trial.specification.spec_id
        != first.record.selected_trial.specification.spec_id
    )
    assert changed.record.selected_trial.lineage_id == first.record.selected_trial.lineage_id
    assert {link.trial_id for link in changed.record.selected_trial.links} == {
        first.record.trial_id,
        other.record.trial_id,
    }
    assert set(retried.record.exposures + other.record.exposures + (disclosed,)) <= set(
        changed.record.exposures
    )
    for old in (first, repeated, retried, other):
        for artifact in old.artifacts():
            assert changed.read_artifact(artifact.name) == artifact.data
    assert len({item.name for item in changed.artifact_records}) == len(changed.artifact_records)
    assert changed.record.disclosure_coverage == "declared_inputs"
    assert changed.record.confirmation == "unavailable"


def test_rejected_attempt_can_be_retried_without_replacing_it(tmp_path):
    rejected = replay(tmp_path, "invalid")
    attempt = rejected.record.selected_trial.attempts[-1]
    retried = replay(tmp_path, "invalid", intent=Retry(rejected, attempt.id))
    assert retried.record.selected_trial.attempts[0] == attempt
    assert retried.record.selected_trial.attempts[1].run_id == attempt.run_id
    assert retried.record.selected_trial.attempts[1].outcome.status == "rejected"


@pytest.mark.parametrize("operation", ["repeat", "retry"])
@pytest.mark.parametrize("change", ["fixture", "lock"])
def test_repeat_and_retry_refuse_specification_changes(tmp_path, operation, change):
    original = replay(tmp_path)
    intent = (
        Repeat(original)
        if operation == "repeat"
        else Retry(original, original.record.selected_trial.attempts[0].id)
    )
    with pytest.raises(ReplayError, match="identical specification"):
        replay(
            tmp_path,
            "null" if change == "fixture" else "known-answer",
            intent=intent,
            lock=b"changed lock" if change == "lock" else LOCK,
        )
    assert len(list((tmp_path / "packages").glob("*/*/manifest.json"))) == 1


@pytest.mark.parametrize("operation", ["repeat", "retry"])
def test_repeat_and_retry_refuse_a_package_without_a_selected_trial(tmp_path, cli, operation):
    original = replay(tmp_path)
    unselected = save_package(
        tmp_path / "unselected", replace(original.record, trial_id=None), original.artifacts()
    )
    attempt_id = original.record.selected_trial.attempts[0].id
    intent = Repeat(unselected) if operation == "repeat" else Retry(unselected, attempt_id)
    with pytest.raises(PackageError, match="selects no trial"):
        replay(tmp_path, intent=intent)
    assert len(list((tmp_path / "packages").glob("*/*/manifest.json"))) == 1
    flags = (
        ["--repeat", str(unselected.path)]
        if operation == "repeat"
        else ["--retry", str(unselected.path), "--attempt-id", str(attempt_id)]
    )
    result = cli(
        tmp_path,
        "replay",
        str(FIXTURES / "known-answer.json"),
        "--output",
        str(tmp_path / "runs"),
        "--lock",
        str(ROOT / "uv.lock"),
        *flags,
        success=False,
    )
    assert result.stderr == "Replay failed: The package selects no trial\n"
    assert not (tmp_path / "runs").exists()


def test_new_trial_carries_unselected_history_without_linking_to_it(tmp_path):
    first = replay(tmp_path)
    unselected = save_package(
        tmp_path / "unselected", replace(first.record, trial_id=None), first.artifacts()
    )
    other = replay(tmp_path, "null")
    question = "A question after a campaign that selected nothing"
    linked = replay(tmp_path, intent=NewTrial(question, (unselected, other)))
    alone = replay(tmp_path, intent=NewTrial(question, (unselected,)))
    (link,) = linked.record.selected_trial.links
    assert link.trial_id == other.record.trial_id
    assert linked.record.selected_trial.lineage_id == other.record.selected_trial.lineage_id
    assert alone.record.selected_trial.links == ()
    assert alone.record.selected_trial.lineage_id not in {
        trial.lineage_id for trial in unselected.record.trials
    }
    for package in (linked, alone):
        assert set(unselected.record.trials) <= set(package.record.trials)
        assert set(unselected.record.exposures) <= set(package.record.exposures)
        for artifact in unselected.artifacts():
            assert package.read_artifact(artifact.name) == artifact.data


def test_invalid_intents_do_not_create_additional_trials(tmp_path):
    original = replay(tmp_path)
    with pytest.raises(ReplayError, match="existing attempt"):
        replay(tmp_path, intent=Retry(original, uuid4()))
    with pytest.raises(PackageError, match="question must be nonempty text"):
        replay(tmp_path, intent=NewTrial(""))
    with pytest.raises(ReplayError, match="lock bytes"):
        replay(tmp_path, lock=b"")
    with pytest.raises(ReplayError, match="explicit"):
        replay_fixture(FIXTURES / "known-answer.json", tmp_path / "bad", intent="retry", lock=LOCK)
    assert len(list((tmp_path / "packages").glob("*/*/manifest.json"))) == 1


@pytest.mark.parametrize("conflict", ["trial", "attempt", "exposure", "artifact"])
def test_conflicting_supplied_history_is_never_overwritten(tmp_path, conflict):
    first = replay(tmp_path)
    record = first.record
    trial = record.selected_trial
    artifacts = first.artifacts()
    match conflict:
        case "trial":
            trial = replace(
                trial, specification=replace(trial.specification, question="Conflicting question")
            )
            record = replace(record, trials=(trial,))
        case "attempt":
            attempt = trial.attempts[0]
            attempt = replace(
                attempt, outcome=replace(attempt.outcome, detail="Conflicting result")
            )
            record = replace(record, trials=(replace(trial, attempts=(attempt,)),))
        case "exposure":
            record = replace(
                record,
                exposures=(
                    replace(record.exposures[0], reason="Conflicting exposure"),
                    *record.exposures[1:],
                ),
            )
        case "artifact":
            first = save_package(
                tmp_path / "first-extra",
                record,
                (*artifacts, ArtifactInput("shared", b"first", "text/plain", "extra")),
            )
            artifacts = (*artifacts, ArtifactInput("shared", b"second", "text/plain", "extra"))
    conflict_package = save_package(tmp_path / "conflict", record, artifacts)
    with pytest.raises(ReplayError, match="Conflicting"):
        replay(tmp_path, intent=NewTrial(QUESTION, (first, conflict_package)))
    assert len(list((tmp_path / "packages").glob("*/*/manifest.json"))) == 1


def test_shared_exposure_conflict_is_not_hidden_by_a_new_disclosure(tmp_path):
    original = replay(tmp_path)
    changed = replace(original.record.exposures[0], reason="A conflicting disclosure")
    with pytest.raises(ReplayError, match="Conflicting exposure"):
        replay(tmp_path, intent=NewTrial(QUESTION, (original,), (changed,)))


def test_export_keeps_raw_inputs_actual_sources_lock_and_software_exposure(tmp_path):
    saved = replay(tmp_path)
    copied_path = tmp_path / "copied"
    shutil.copytree(saved.path, copied_path)
    shutil.rmtree(tmp_path / "packages")
    package = open_package(copied_path)
    artifact_bytes = {item.role: item.data for item in package.artifacts()}
    assert artifact_bytes["input"] == (FIXTURES / "known-answer.json").read_bytes()
    assert artifact_bytes["lock"] == LOCK
    implementation = json.loads(artifact_bytes["implementation"])
    source_root = ROOT / "src"
    assert implementation["sources"] == {
        path.relative_to(source_root).as_posix(): path.read_text()
        for path in (source_root / "quant_research").rglob("*.py")
    }
    spec = package.record.selected_trial.specification
    assert spec.input_digests == (hashlib.sha256(artifact_bytes["input"]).hexdigest(),)
    assert spec.environment.lock_digest == hashlib.sha256(LOCK).hexdigest()
    assert spec.environment.uv_version == "not observed; caller-supplied lock"
    assert "unverified" in " ".join(package.record.selected_trial.attempts[-1].outcome.limitations)
    assert {event.kind for event in package.record.exposures} == {"data", "result"}
    assert {event.actor for event in package.record.exposures} == {
        "software:quant_research.replay/v1"
    }
    assert all("human exposure is unknown" in event.reason for event in package.record.exposures)
    restored_input = tmp_path / "input.json"
    restored_input.write_bytes(artifact_bytes["input"])
    repeated = replay_fixture(
        restored_input, tmp_path / "restored", intent=Repeat(package), lock=artifact_bytes["lock"]
    )
    assert repeated.record.selected_trial.attempts[0] == package.record.selected_trial.attempts[0]
    assert output(repeated, "diagnostic") == output(package, "diagnostic")


def test_runtime_patch_ignores_build_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "version", sys.version.replace(sys.version.split()[0], "3.12.13+"))
    package = replay(tmp_path)
    assert package.record.selected_trial.specification.environment.python_version == ".".join(
        str(part) for part in sys.version_info[:3]
    )


def test_nested_module_changes_require_a_new_trial(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "models").mkdir(parents=True)
    (source / "__init__.py").write_text("")
    (source / "models" / "signal.py").write_text("VALUE = 1\n")
    (source / "notes.txt").write_text("not source\n")
    monkeypatch.setattr(replay_module.resources, "files", lambda name: source)
    saved = replay(tmp_path)
    implementation = next(item for item in saved.artifacts() if item.role == "implementation")
    assert json.loads(implementation.data)["sources"] == {
        "quant_research/__init__.py": "",
        "quant_research/models/signal.py": "VALUE = 1\n",
    }
    (source / "models" / "signal.py").write_text("VALUE = 2\n")
    with pytest.raises(ReplayError, match="identical specification"):
        replay(tmp_path, intent=Repeat(saved))


def test_supplied_history_uses_records_read_from_disk(tmp_path):
    original = replay(tmp_path)
    forged = replace(original, record=replace(original.record, exposures=()))
    repeated = replay(tmp_path, intent=Repeat(forged))
    assert {event.id for event in original.record.exposures} <= {
        event.id for event in repeated.record.exposures
    }


def test_mutated_supplied_package_is_rejected_before_replay(tmp_path):
    original = replay(tmp_path)
    (original.path / "manifest.json").write_bytes(b"{}")
    with pytest.raises(PackageError):
        replay(tmp_path, intent=Repeat(original))


def test_cli_exposes_new_repeat_retry_and_linked_history(tmp_path, cli):
    base = [
        "replay",
        str(FIXTURES / "known-answer.json"),
        "--output",
        str(tmp_path / "runs"),
        "--lock",
        str(ROOT / "uv.lock"),
    ]
    first = json.loads(cli(tmp_path, *base, "--new", QUESTION).stdout)
    assert first["status"] == "successful"
    assert "unverified" in " ".join(first["limitations"])
    repeated = json.loads(cli(tmp_path, *base, "--repeat", first["package_path"]).stdout)
    retried = json.loads(
        cli(
            tmp_path,
            *base,
            "--retry",
            repeated["package_path"],
            "--attempt-id",
            first["attempt_id"],
        ).stdout
    )
    assert UUID(first["run_id"]) == UUID(retried["run_id"]) != UUID(repeated["run_id"])
    changed = json.loads(
        cli(
            tmp_path,
            *base,
            "--new",
            "Linked question",
            "--history",
            retried["package_path"],
            "--reason",
            "Changed question",
        ).stdout
    )
    package = open_package(Path(changed["package_path"]))
    assert len(package.record.trials) == 2
    assert len(package.record.trials[0].attempts) == 3
    assert package.record.selected_trial.links[0].trial_id == UUID(first["trial_id"])
    assert (
        "--attempt-id"
        in cli(tmp_path, *base, "--retry", first["package_path"], success=False).stderr
    )
    assert (
        "--history"
        in cli(
            tmp_path,
            *base,
            "--repeat",
            first["package_path"],
            "--history",
            first["package_path"],
            success=False,
        ).stderr
    )
    assert "Replay failed" in cli(tmp_path, *base, "--new", "", success=False).stderr
    assert (
        "--reason"
        in cli(
            tmp_path,
            *base,
            "--repeat",
            first["package_path"],
            "--reason",
            "This cannot change a repeat",
            success=False,
        ).stderr
    )
    invalid = json.loads(
        cli(
            tmp_path,
            "replay",
            str(FIXTURES / "invalid.json"),
            "--output",
            str(tmp_path / "runs"),
            "--lock",
            str(ROOT / "uv.lock"),
            "--new",
            QUESTION,
        ).stdout
    )
    assert invalid["status"] == "rejected"


@pytest.mark.parametrize("reason", ["why", ""])
def test_cli_reason_requires_history_before_reading_inputs(tmp_path, cli, reason):
    result = cli(
        tmp_path,
        "replay",
        str(tmp_path / "missing-fixture.json"),
        "--output",
        str(tmp_path / "runs"),
        "--lock",
        str(tmp_path / "missing.lock"),
        "--new",
        QUESTION,
        "--reason",
        reason,
        success=False,
    )
    assert result.returncode == 2
    assert "--reason requires --history" in result.stderr
    assert not (tmp_path / "runs").exists()
