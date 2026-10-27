from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from datetime import UTC, date, datetime
from fractions import Fraction
from pathlib import Path

import pytest

from quant_research import datasets
from quant_research.access import issue_context
from quant_research.packages import open_package, save_package
from quant_research.public_data import approve, ingest, load_catalog, pull, validate
from quant_research.reconcile import (
    METRICS,
    SAHM_REALTIME,
    ReconcileError,
    reconcile,
    reconcile_series,
)
from quant_research.records import PackageError
from quant_research.replay import NewTrial, replay_fixture
from quant_research.signals import SAHM, decide

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/causal"
REFERENCE = ROOT / "tests/fixtures/reconcile/known-answer.json"
KNOWN_ANSWER = json.loads(REFERENCE.read_bytes())["metrics"]
QUESTION = "Describe the synthetic level-difference association"
FIELDS = ("recorded", "expected", "tolerance", "difference", "status")


def replay(directory: Path, fixture: str | dict = "known-answer") -> Path:
    if isinstance(fixture, dict):
        source = directory / "fixture.json"
        source.write_text(json.dumps(fixture))
    else:
        source = FIXTURES / f"{fixture}.json"
    return replay_fixture(
        source,
        directory / "packages",
        intent=NewTrial(QUESTION),
        lock=(ROOT / "uv.lock").read_bytes(),
    ).path


def write_reference(directory: Path, metrics: dict) -> Path:
    path = directory / "reference.json"
    path.write_text(json.dumps({"version": 1, "metrics": metrics}))
    return path


def run(cli, directory: Path, package: Path, reference: Path, *, success=True):
    arguments = ("reconcile", "package", str(package), "--reference", str(reference))
    return cli(directory, *arguments, success=success)


def row(*values) -> dict:
    return dict(zip(FIELDS, values, strict=True))


def statuses(rows: dict) -> set[str]:
    return {item["status"] for item in rows.values()}


@pytest.fixture(scope="module")
def known_answer(tmp_path_factory) -> Path:
    return replay(tmp_path_factory.mktemp("known-answer"))


@pytest.mark.smoke
def test_known_answer_package_agrees_with_reference_and_recomputation(tmp_path, cli, known_answer):
    result = run(cli, tmp_path, known_answer, REFERENCE)
    record = json.loads(result.stdout)
    package = open_package(known_answer)
    assert record["agreed"] is True
    assert record["package_digest"] == package.digest
    assert record["attempt_id"] == str(package.record.selected_trial.attempts[-1].id)
    assert record["reference_digest"] == hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    values = {"decisions": "3", "signals": "3", "pairs": "3", "coefficient": "1"}
    values |= dict.fromkeys(("centered_cross", "signal_variance", "label_variance"), "2")
    for side in ("reference", "recomputation"):
        assert {name: item["recorded"] for name, item in record[side].items()} == values
        assert {name: item["expected"] for name, item in record[side].items()} == values
        assert statuses(record[side]) == {"agreed"}
    assert record["reference"]["coefficient"] == row("1", "1", "1E-30", "0", "agreed")
    assert {item["tolerance"] for item in record["recomputation"].values()} == {"0"}
    assert reconcile(package, REFERENCE).record().decode() == result.stdout


def test_record_is_byte_identical_for_a_copied_package(tmp_path, cli, known_answer):
    copied = tmp_path / "elsewhere" / "copied"
    shutil.copytree(known_answer, copied)
    outputs = {
        run(cli, location.parent, location, REFERENCE).stdout
        for location in (known_answer, known_answer, copied)
    }
    assert len(outputs) == 1


@pytest.mark.parametrize(
    "change, name, expected",
    [
        (
            {"coefficient": {"value": "0.9", "tolerance": "0.05"}},
            "coefficient",
            row("1", "0.9", "0.05", "0.1", "disagreed"),
        ),
        (
            {"coefficient": {"value": "1.05", "tolerance": "0.05"}},
            "coefficient",
            row("1", "1.05", "0.05", "-0.05", "agreed"),
        ),
        ({"pairs": None}, "pairs", row("3", None, None, None, "missing_from_expected")),
        (
            {"turnover": {"value": 0, "tolerance": 0}},
            "turnover",
            row(None, "0", "0", None, "missing_from_package"),
        ),
    ],
)
def test_reference_rows_report_tolerance_and_missing_metrics(
    tmp_path, cli, known_answer, change, name, expected
):
    metrics = {key: value for key, value in (KNOWN_ANSWER | change).items() if value is not None}
    agreed = expected["status"] == "agreed"
    result = run(cli, tmp_path, known_answer, write_reference(tmp_path, metrics), success=agreed)
    record = json.loads(result.stdout)
    assert record["agreed"] is agreed
    assert result.stderr == ("" if agreed else "Reconciliation found disagreements\n")
    assert record["reference"].pop(name) == expected
    assert statuses(record["reference"]) == statuses(record["recomputation"]) == {"agreed"}


def test_inconclusive_package_reports_coefficient_missing_from_package(tmp_path, cli):
    counts = {"decisions": 3, "signals": 3, "pairs": 2}
    reference = write_reference(
        tmp_path,
        {name: {"value": value, "tolerance": 0} for name, value in counts.items()}
        | {"coefficient": {"value": "0", "tolerance": "1"}},
    )
    package = replay(tmp_path, "insufficient-pairs")
    record = json.loads(run(cli, tmp_path, package, reference, success=False).stdout)
    assert record["reference"].pop("coefficient") == row(
        None, "0", "1", None, "missing_from_package"
    )
    assert statuses(record["reference"]) == {"agreed"}
    assert {name: item["expected"] for name, item in record["recomputation"].items()} == {
        name: str(value) for name, value in counts.items()
    }
    assert statuses(record["recomputation"]) == {"agreed"}


def test_a_package_without_a_selected_trial_is_refused_by_name(tmp_path, cli, known_answer):
    package = open_package(known_answer)
    unselected = save_package(
        tmp_path / "unselected", replace(package.record, trial_id=None), package.artifacts()
    )
    with pytest.raises(PackageError, match="selects no trial"):
        reconcile(unselected, REFERENCE)
    result = run(cli, tmp_path, unselected.path, REFERENCE, success=False)
    assert (result.returncode, result.stdout, result.stderr) == (
        1,
        "",
        "Reconciliation failed: The package selects no trial\n",
    )


@pytest.mark.parametrize(
    "metrics, reason",
    [
        (KNOWN_ANSWER, "The selected trial has no attempts to reconcile"),
        ({}, "Reference metrics must be a nonempty object"),
    ],
    ids=("no-attempts", "malformed-reference-first"),
)
def test_a_selected_trial_without_attempts_is_refused_after_reference_validation(
    tmp_path, cli, known_answer, metrics, reason
):
    package = open_package(known_answer)
    saved = save_package(
        tmp_path / "empty",
        replace(
            package.record,
            trials=(replace(package.record.selected_trial, attempts=()),),
            exposures=(),
        ),
        package.artifacts(),
    )
    opened = open_package(saved.path)
    reference = write_reference(tmp_path, metrics)
    with pytest.raises(ReconcileError, match=reason):
        reconcile(opened, reference)
    result = run(cli, tmp_path, opened.path, reference, success=False)
    assert (result.returncode, result.stdout, result.stderr) == (
        1,
        "",
        f"Reconciliation failed: {reason}\n",
    )


def test_recomputation_catches_a_recorded_metric_the_reference_accepts(tmp_path, cli, known_answer):
    package = open_package(known_answer)
    outcome = package.record.selected_trial.attempts[-1].outcome
    name = next(
        item.name
        for item in package.artifact_records
        if item.role == "diagnostic" and item.name in outcome.result_artifacts
    )
    original = package.read_artifact(name)
    edited = json.dumps(json.loads(original) | {"coefficient": "0.5"}).encode()
    original_digest, edited_digest = (
        hashlib.sha256(data).hexdigest() for data in (original, edited)
    )
    exposures = tuple(
        replace(event, material_digest=edited_digest)
        if event.material_digest == original_digest
        else event
        for event in package.record.exposures
    )
    saved = save_package(
        tmp_path / "rewritten",
        replace(package.record, exposures=exposures),
        tuple(
            replace(item, data=edited) if item.name == name else item
            for item in package.artifacts()
        ),
    )
    assert "Verified" in cli(tmp_path, "verify", str(saved.path)).stdout
    reference = write_reference(
        tmp_path, KNOWN_ANSWER | {"coefficient": {"value": "0.5", "tolerance": "0"}}
    )
    record = json.loads(run(cli, tmp_path, saved.path, reference, success=False).stdout)
    assert statuses(record["reference"]) == {"agreed"}
    assert record["recomputation"].pop("coefficient") == row("0.5", "1", "0", "-0.5", "disagreed")
    assert statuses(record["recomputation"]) == {"agreed"}


def negative_labels_rounded_twice(document: dict) -> None:
    for label, value in zip(document["labels"], ("0", "2", "-1"), strict=True):
        label["value"] = value


def signal_beyond_precision(document: dict) -> None:
    # A's signal -1 - 6e-34 needs 35 significant digits, so the saved signal is rounded.
    document["observations"][0]["level"] = "6e-34"
    document["observations"][2]["level"] = "-1"


def selection_rules(document: dict) -> None:
    for entity, at, level in (
        ("B", "2026-01-07T14:00:00Z", "12"),
        ("C", "2026-01-07T16:05:00Z", "13"),
    ):
        document["observations"].append(
            {
                "entity_id": entity,
                "session_id": "s2",
                "revision": 1,
                "published_at": at,
                "arrived_at": at,
                "level": level,
                "unit": "synthetic_level",
            }
        )
    document["decisions"] += [
        {"entity_id": entity, "session_id": session, "at": at}
        for entity, session, at in (
            ("A", "s0", "2026-01-07T17:30:00Z"),  # the first session has no predecessor
            ("B", "s2", "2026-01-07T15:00:00Z"),  # s2 has not occurred
            ("A", "s2", "2026-01-07T17:00:00Z"),  # the latest known membership excludes A
            ("C", "s2", "2026-01-07T17:00:00Z"),  # a signal whose label arrives after the cutoff
            ("B", "s2", "2026-01-07T17:00:00Z"),  # B's latest vintage leaves out s2
        )
    ]
    document["memberships"].append(
        {
            "entity_id": "B",
            "first_session": "s0",
            "last_session": "s1",
            "known_at": "2026-01-07T16:30:00Z",
            "included": True,
        }
    )
    document["labels"].append(
        {
            "entity_id": "C",
            "decision_at": "2026-01-07T17:00:00Z",
            "available_at": "2026-01-07T19:00:00Z",
            "value": "5",
            "unit": "synthetic_target",
        }
    )


@pytest.mark.parametrize(
    "edit, expected",
    [
        (
            negative_labels_rounded_twice,
            {
                "centered_cross": Fraction(-1),
                "signal_variance": Fraction(2),
                "label_variance": Fraction("4.666666666666666666666666666666667"),
                "coefficient": Fraction("-0.3273268353539885718991462281234291"),
            },
        ),
        (
            signal_beyond_precision,
            {
                "centered_cross": Fraction("2.000000000000000000000000000000001"),
                "signal_variance": Fraction("2.000000000000000000000000000000002"),
                "label_variance": Fraction(2),
                "coefficient": Fraction(1),
            },
        ),
        (selection_rules, {"decisions": Fraction(8), "signals": Fraction(4), "pairs": Fraction(3)}),
    ],
)
def test_recomputation_matches_replay_at_zero_tolerance(tmp_path, edit, expected):
    document = json.loads((FIXTURES / "known-answer.json").read_bytes())
    edit(document)
    result = reconcile(open_package(replay(tmp_path, document)), REFERENCE)
    assert [item.name for item in result.recomputation] == sorted(METRICS)
    assert {(item.status, item.tolerance) for item in result.recomputation} == {("agreed", 0)}
    recorded = {item.name: item.recorded for item in result.recomputation}
    assert {name: recorded[name] for name in expected} == expected


def reference_text(entry: object = None, **fields) -> str:
    entry = {"value": "1", "tolerance": "0"} if entry is None else entry
    return json.dumps({"version": 1, "metrics": {"coefficient": entry}} | fields)


REFUSED_REFERENCES = {
    "no-tolerance": reference_text({"value": "1"}),
    "extra-entry-field": reference_text({"value": "1", "tolerance": "0", "unit": "x"}),
    "entry-not-object": reference_text("1"),
    "float-value": reference_text({"value": 1.5, "tolerance": "0"}),
    "nan-constant": reference_text({"value": float("nan"), "tolerance": "0"}),
    "bool-value": reference_text({"value": True, "tolerance": "0"}),
    "null-value": reference_text({"value": None, "tolerance": "0"}),
    "leading-zero": reference_text({"value": "01", "tolerance": "0"}),
    "plus-sign": reference_text({"value": "+1", "tolerance": "0"}),
    "trailing-point": reference_text({"value": "1.", "tolerance": "0"}),
    "nan-string": reference_text({"value": "NaN", "tolerance": "0"}),
    "negative-tolerance": reference_text({"value": "1", "tolerance": "-1e-30"}),
    "huge-exponent": reference_text({"value": "1e5000", "tolerance": "0"}),
    "tiny-exponent": reference_text({"value": "1", "tolerance": "1e-5000"}),
    "long-digits": reference_text({"value": "1" * 5000, "tolerance": "0"}),
    "endless-exponent": reference_text({"value": "1e9999999999999999999", "tolerance": "0"}),
    "surrogate-name": '{"version": 1, "metrics": {"\\ud800": {"value": "1", "tolerance": "0"}}}',
    "empty-metrics": reference_text(metrics={}),
    "metrics-array": reference_text(metrics=["coefficient"]),
    "string-version": reference_text(version="1"),
    "bool-version": reference_text(version=True),
    "version-2": reference_text(version=2),
    "extra-field": reference_text(note=""),
    "duplicate-key": '{"version": 1, ' + reference_text()[1:],
    "array": "[]",
    "truncated": reference_text()[:-1],
}


@pytest.mark.parametrize(
    "package, reference, error",
    [
        *(
            pytest.param("known-answer", text, ReconcileError, id=label)
            for label, text in REFUSED_REFERENCES.items()
        ),
        pytest.param("invalid", REFERENCE.read_text(), ReconcileError, id="rejected-attempt"),
        pytest.param("example", REFERENCE.read_text(), ReconcileError, id="no-diagnostic"),
        pytest.param("known-answer", None, OSError, id="unreadable-reference"),
    ],
)
def test_unreconcilable_reference_or_package_is_refused(
    tmp_path, cli, known_answer, package, reference, error
):
    paths = {"known-answer": known_answer, "example": ROOT / "examples/development-package"}
    path = paths[package] if package in paths else replay(tmp_path, package)
    reference_path = tmp_path / "reference.json"
    if reference is not None:
        reference_path.write_text(reference)
    result = run(cli, tmp_path, path, reference_path, success=False)
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.startswith("Reconciliation failed: ")
    with pytest.raises(error):
        reconcile(open_package(path), reference_path)


def test_reference_numbers_at_the_bounds_print_an_exact_record(tmp_path, cli, known_answer):
    largest = "9" * 1000 + "e1"
    smallest = "0." + "0" * 999 + "1" * 1000
    reference = write_reference(
        tmp_path, {name: {"value": largest, "tolerance": smallest} for name in METRICS}
    )
    result = run(cli, tmp_path, known_answer, reference, success=False)
    rows = json.loads(result.stdout)["reference"]
    assert statuses(rows) == {"disagreed"}
    assert Fraction(rows["coefficient"]["difference"]) == 1 - Fraction(largest)


SIGNALS = ROOT / "tests/fixtures/signals"
# Each ALFRED response is an excerpt of a real download, with count set to the excerpt's length.
EXCERPTS = {
    name: (SIGNALS / f"alfred-{name}.json").read_bytes()
    for name in ("unrate-2024", "sahmrealtime-2025", "unrate-1960", "unrate-2021")
}
NOW = datetime(2026, 10, 22, 12, tzinfo=UTC)
# FRED's SAHMREALTIME for December 1960 and March 2021, from the 22 October 2026 download.
FRED_1960_12, FRED_2021_03 = Fraction("1.30"), Fraction("2.33")
UNOBSERVED_NULL = "FRED has no value and the program cannot compute the month"
CONTEXT = issue_context(clearances=(), cutoff=NOW, generation=SAHM_REALTIME.name, providers=())


def month(text: str) -> date:
    return date.fromisoformat(f"{text}-01")


def approved_root(root: Path, responses: dict[str, bytes]) -> Path:
    for name, data in responses.items():
        pull(root, name, now=NOW, api_key="k", fetch=lambda url, data=data: data)
        ingest(root, name, now=NOW)
        validate(root, name, actor="researcher", evidence=b"notes", summary="checked", now=NOW)
        approve(root, name, actor="researcher", now=NOW)
    return root


def series_root(root: Path, reference: bytes = EXCERPTS["sahmrealtime-2025"]) -> Path:
    unrate = EXCERPTS["unrate-2024"]
    return approved_root(root, {"alfred.unrate": unrate, "alfred.sahmrealtime": reference})


def reconcile_root(root: Path, profile=SAHM_REALTIME):
    return reconcile_series(profile, load_catalog(root), root / "datasets", CONTEXT)


def decide_root(root: Path, months: list[date]):
    return decide(SAHM, load_catalog(root), root / "datasets", CONTEXT, months)


@pytest.fixture(scope="module")
def sahm_root(tmp_path_factory) -> Path:
    return series_root(tmp_path_factory.mktemp("sahm"))


@pytest.mark.smoke
def test_series_reconciliation_matches_fred_and_names_each_exclusion(tmp_path, cli, sahm_root):
    record = json.loads(cli(tmp_path, "reconcile", "series", "--root", str(sahm_root)).stdout)
    catalog = load_catalog(sahm_root)
    assert record["agreed"] is True
    assert record["profile"] == "sahm-realtime/v1"
    assert record["window"] == {"start": "2025-08-01", "end": "2026-01-01"}
    assert record["matched"] == 5
    assert record["excluded"] == [
        {"month": "2025-10-01", "reason": UNOBSERVED_NULL, "reference": None, "recomputed": []}
    ]
    assert (record["mismatched"], record["before_window"]) == ([], [])
    assert record["inputs"] == {
        name: catalog.entry(name).documentation.vintage
        for name in ("alfred.unrate", "alfred.sahmrealtime")
    }
    assert record["catalog"] == catalog.digest
    recorded = reconcile_root(sahm_root).record()
    copied = shutil.copytree(sahm_root, tmp_path / "elsewhere" / "copied")
    assert reconcile_root(copied).record() == recorded
    function = json.loads(recorded)
    assert function.pop("cutoff") == "2026-10-22T12:00:00Z"
    record.pop("cutoff")
    assert record == function


def test_a_cutoff_before_the_reference_is_released_is_refused(sahm_root):
    # FRED's first excerpt month, August 2025, is available from 2025-09-07.
    early = CONTEXT.narrow(cutoff=datetime(2025, 9, 1, tzinfo=UTC))
    with pytest.raises(ReconcileError, match="'alfred.sahmrealtime' has no rows available by"):
        reconcile_series(SAHM_REALTIME, load_catalog(sahm_root), sahm_root / "datasets", early)


def test_a_series_too_short_for_the_program_is_refused(tmp_path):
    short = (ROOT / "tests/fixtures/public_data/alfred-unrate.json").read_bytes()
    root = approved_root(
        tmp_path, {"alfred.unrate": short, "alfred.sahmrealtime": EXCERPTS["sahmrealtime-2025"]}
    )
    with pytest.raises(ReconcileError, match="can compute sahm-realtime/v1"):
        reconcile_root(root)


def sahm_reference(change) -> bytes:
    response = json.loads(EXCERPTS["sahmrealtime-2025"])
    change(response["observations"])
    response["count"] = len(response["observations"])
    return json.dumps(response).encode()


def test_a_reference_that_ends_before_the_first_observation_is_refused(tmp_path):
    def earlier(observations):
        period = {"realtime_start": "2024-09-06", "realtime_end": "9999-12-31", "value": "0.1"}
        observations[:] = [period | {"date": day} for day in ("2024-06-01", "2024-07-01")]

    root = series_root(tmp_path, sahm_reference(earlier))
    with pytest.raises(ReconcileError, match="ends at 2024-07, before the first observed month"):
        reconcile_root(root)


def set_value(observations: list[dict], day: str, value: str) -> None:
    (item,) = [item for item in observations if item["date"] == day]
    item["value"] = value


def test_a_tampered_reference_month_mismatches_unless_declared(tmp_path, cli):
    reference = sahm_reference(lambda observations: set_value(observations, "2025-11-01", "0.50"))
    root = series_root(tmp_path / "data", reference)
    month_record = {"month": "2025-11-01", "reference": "1/2", "recomputed": ["13/30"]}
    result = reconcile_root(root)
    assert result.agreed is False
    assert json.loads(result.record())["mismatched"] == [month_record]
    output = cli(tmp_path, "reconcile", "series", "--root", str(root), success=False)
    assert json.loads(output.stdout)["agreed"] is False
    assert output.stderr == "Reconciliation found disagreements\n"
    # A declared month that matches still counts as matched.
    exclusions = {month("2025-11"): "Declared for this test", month("2025-12"): "Unused"}
    record = json.loads(
        reconcile_root(root, replace(SAHM_REALTIME, exclusions=exclusions)).record()
    )
    assert (record["agreed"], record["matched"], record["mismatched"]) == (True, 4, [])
    assert record["excluded"] == [
        {"month": "2025-10-01", "reason": UNOBSERVED_NULL, "reference": None, "recomputed": []},
        month_record | {"reason": "Declared for this test"},
    ]


def skipped_and_unreleased(observations: list[dict]) -> None:
    set_value(observations, "2025-10-01", "0.40")
    observations.append(
        {
            "realtime_start": "2026-03-06",
            "realtime_end": "9999-12-31",
            "date": "2026-02-01",
            "value": "0.30",
        }
    )


def test_a_month_no_decision_observes_names_the_decision_that_passed_it(tmp_path):
    root = series_root(tmp_path, sahm_reference(skipped_and_unreleased))
    record = json.loads(reconcile_root(root).record())
    assert (record["agreed"], record["matched"]) == (True, 5)
    assert record["window"] == {"start": "2025-08-01", "end": "2026-02-01"}
    assert record["excluded"] == [
        {
            "month": "2025-10-01",
            "reason": "no decision observes it; the decision for 2026-01 already observes 2025-11",
            "reference": "2/5",
            "recomputed": [],
        },
        {
            "month": "2026-02-01",
            "reason": "no decision before the run's cutoff observes it",
            "reference": "3/10",
            "recomputed": [],
        },
    ]


def test_fred_december_1960_is_the_indicator_after_a_later_revision(tmp_path):
    root = approved_root(tmp_path, {"alfred.unrate": EXCERPTS["unrate-1960"]})
    february, march = decide_root(root, [month("1961-02"), month("1961-03")])
    assert (february.observation, february.value) == (month("1960-12"), Fraction(43, 30))
    assert abs(february.value - FRED_1960_12) > SAHM_REALTIME.tolerance
    # The 1961-02-09 release, available the next midnight UTC (D052), still ends at December 1960.
    assert march.released_at == datetime(1961, 2, 10, tzinfo=UTC)
    assert (march.observation, march.value) == (month("1960-12"), Fraction(13, 10))
    assert abs(march.value - FRED_1960_12) <= SAHM_REALTIME.tolerance
    assert "1961-02-09 release" in SAHM_REALTIME.exclusions[month("1960-12")]


def test_fred_march_2021_needs_january_2020_before_its_revision(tmp_path):
    root = approved_root(tmp_path, {"alfred.unrate": EXCERPTS["unrate-2021"]})
    (may,) = decide_root(root, [month("2021-05")])
    assert (may.observation, may.value) == (month("2021-03"), Fraction(71, 30))
    assert abs(may.value - FRED_2021_03) > SAHM_REALTIME.tolerance
    narrowed = CONTEXT.narrow(cutoff=datetime(2021, 5, 1, tzinfo=UTC))
    read = datasets.read(load_catalog(root), root / "datasets", "alfred.unrate", narrowed)
    unrate = {date.fromisoformat(row.key): row.values for row in read.rows}
    unrate[month("2020-01")] = {"value": "3.6"}
    value = SAHM.evaluate({"alfred.unrate": unrate}, month("2021-03"))
    assert value == Fraction(7, 3)
    assert abs(value - FRED_2021_03) <= SAHM_REALTIME.tolerance
    assert month("2021-03") in SAHM_REALTIME.exclusions
