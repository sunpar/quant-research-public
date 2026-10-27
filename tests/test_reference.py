from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from fractions import Fraction
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from quant_research import datasets, reference, signals
from quant_research.access import (
    AccessError,
    Provider,
    Restrictions,
    derive,
    issue_context,
    licensed,
)
from quant_research.catalog import Catalog, Documented
from quant_research.datasets import ReadRecord, Refused
from quant_research.evaluation import Invalid, Profile
from quant_research.packages import (
    ArtifactInput,
    LineageLink,
    VerifiedPackage,
    open_package,
    save_package,
)
from quant_research.providers import Material, dump_material, load_material
from quant_research.public_data import PILOT, approve, ingest, load_catalog, pull, validate
from quant_research.records import PackageError, canonical_bytes
from quant_research.reference import FACTORS
from quant_research.signals import (
    SAHM,
    SAHM_RULE,
    Decision,
    Lag,
    Series,
    Subtract,
    Threshold,
    canonical,
    decide,
    shift,
)

# numba's default cache writes into site-packages, which the prepared environment fingerprints.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "quant-research-numba"))

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 22, 12, tzinfo=UTC)
CONTEXT = issue_context(clearances=(), cutoff=NOW, generation=reference.GENERATION, providers=())
LOCK = b"synthetic lock\n"
CASES = {case.name: case for case in reference.CASES}
REFERENCE, CURRENT, FIXED, NEVER = CASES


def month(text: str) -> date:
    return date.fromisoformat(f"{text}-01")


def midnight(day: date) -> datetime:
    return datetime.combine(day, time(), UTC)


def span(first: str, last: str) -> list[date]:
    start, end = month(first), month(last)
    count = (end.year - start.year) * 12 + end.month - start.month + 1
    return [shift(start, offset) for offset in range(count)]


MONTHS = span("2019-01", "2024-02")
# Unemployment rises through 2022, so the rule holds T-bills from June to December 2022.
LEVELS = {
    month(key): value
    for key, value in {
        "2022-01": "4.2",
        "2022-02": "4.4",
        "2022-03": "4.6",
        "2022-04": "4.8",
        "2022-05": "5.0",
        "2022-06": "5.0",
        "2022-07": "4.8",
        "2022-08": "4.6",
        "2022-09": "4.4",
        "2022-10": ".",
        "2022-11": "4.2",
    }.items()
}
# BLS's 2025 delays, moved to 2022: September out on 20 November, and the uncollected October
# released as null with November on 16 December. Each is available the next midnight UTC.
SLIPPED = {month("2022-09"): date(2022, 11, 20), month("2022-10"): date(2022, 12, 16)}
SLIPPED[month("2022-11")] = date(2022, 12, 16)
# The January 2023 release revises March 2022 from 4.6 to 4.9, which lifts the indicator May
# 2022 observes from 0.40 to 0.50, so the current vintage holds T-bills a month early.
REVISED, REVISION, REVISED_ON = month("2022-03"), "4.9", date(2023, 1, 13)
BILLS = span("2022-06", "2022-12")


def release(observed: date) -> date:
    following = shift(observed, 1)
    return SLIPPED.get(observed, date(following.year, following.month, 5))


def alfred(rows: list[tuple[date, date, date, str]]) -> bytes:
    observations = [
        {
            "date": observed.isoformat(),
            "realtime_end": end.isoformat(),
            "realtime_start": start.isoformat(),
            "value": value,
        }
        for observed, start, end, value in rows
    ]
    return json.dumps(
        {
            "count": len(observations),
            "observations": observations,
            "realtime_end": "9999-12-31",
            "realtime_start": "1776-07-04",
        }
    ).encode()


def unrate() -> bytes:
    rows = []
    for observed in MONTHS:
        value = LEVELS.get(observed, "4.0")
        if observed == REVISED:
            rows.append((observed, release(observed), REVISED_ON - timedelta(days=1), value))
            rows.append((observed, REVISED_ON, date.max, REVISION))
        else:
            rows.append((observed, release(observed), date.max, value))
    return alfred(rows)


def sahm_realtime(root: Path) -> bytes:
    """FRED's series as the causal indicator at two decimals, released with each UNRATE month."""
    catalog = load_catalog(root)
    decisions = decide(SAHM, catalog, root / "datasets", CONTEXT, span("2020-04", "2024-03"))
    observed = {item.observation: item.value for item in decisions if item.observation}
    return alfred(
        [
            (key, release(key), date.max, f"{observed[key]:.2f}" if key in observed else ".")
            for key in span(f"{min(observed):%Y-%m}", f"{max(observed):%Y-%m}")
        ]
    )


def french() -> bytes:
    rows = [
        f"{key:%Y%m},{(index * 37 % 23 * 50 - 550) / 100:7.2f},   0.10,  -0.10,"
        f"{(10 + index % 7 * 2) / 100:7.2f}"
        for index, key in enumerate(MONTHS)
    ]
    text = "\n".join(
        [
            "Synthetic monthly factors for tests",
            "",
            ",Mkt-RF,SMB,HML,RF",
            *rows,
            "",
            " Annual Factors: January-December ",
            ",Mkt-RF,SMB,HML,RF",
            "",
        ]
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("F-F_Research_Data_Factors.csv", text)
    return buffer.getvalue()


def approved(root: Path, responses: dict[str, bytes]) -> Path:
    for name, data in responses.items():
        pull(root, name, now=NOW, api_key="k", fetch=lambda url, data=data: data)
        ingest(root, name, now=NOW)
        validate(root, name, actor="researcher", evidence=b"notes", summary="checked", now=NOW)
        approve(root, name, actor="researcher", now=NOW)
    return root


def experiment_bytes(**changes) -> bytes:
    rules = {
        "alpha": "0.05",
        "minimum_effect": "0.001",
        "subperiods": 4,
        "version": "pilot-evaluation/v1",
    }
    document = {
        "baseline": "buy_and_hold_market",
        "cutoff": "2024-03-15T00:00:00Z",
        "evaluation": rules,
        "schema_version": 1,
        "window": {"end": "2023-12-01", "start": "2021-01-01"},
    }
    for key, value in changes.items():
        target = (
            rules if key in rules else document["window"] if key in ("start", "end") else document
        )
        target[key] = value
    return canonical_bytes(document)


EXPERIMENT = reference.load_experiment(experiment_bytes())


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    data = approved(tmp_path_factory.mktemp("data"), {"alfred.unrate": unrate()})
    responses = {"alfred.sahmrealtime": sahm_realtime(data), "french.ff3_monthly": french()}
    return approved(data, responses)


def run(case, root: Path, destination: Path, experiment=EXPERIMENT) -> VerifiedPackage:
    return reference.run(
        case,
        experiment,
        catalog=load_catalog(root),
        store=root / "datasets",
        context=CONTEXT,
        lock=LOCK,
        destination=destination,
    )


@pytest.fixture(scope="module")
def saved(root, tmp_path_factory) -> dict[str, VerifiedPackage]:
    destination = tmp_path_factory.mktemp("packages")
    return {case.name: run(case, root, destination) for case in reference.CASES}


def outcome(package: VerifiedPackage):
    (attempt,) = package.record.selected_trial.attempts
    return attempt.outcome


def results(package: VerifiedPackage) -> dict[str, bytes]:
    roles = {item.name: item.role for item in package.artifact_records}
    return {roles[name]: package.read_artifact(name) for name in outcome(package).result_artifacts}


def breaches(package: VerifiedPackage) -> list[tuple[str, str]]:
    found = json.loads(results(package)["audit"])["breaches"]
    return [(item["month"], item["seen"]) for item in found]


def test_both_flawed_variants_are_rejected_before_any_return(saved):
    for name in (CURRENT, FIXED):
        assert outcome(saved[name]).status == "rejected"
        assert set(results(saved[name])) == {"decisions", "audit"}
    # Fixed lag waits for each slipped release and reads it the midnight after it came out.
    assert breaches(saved[FIXED]) == [
        ("2022-11-01", "2022-11-21T00:00:00Z"),
        ("2022-12-01", "2022-12-17T00:00:00Z"),
    ]
    # The current vintage is one read at the experiment's cutoff, after every decision month.
    assert breaches(saved[CURRENT]) == [
        (key.isoformat(), "2024-03-15T00:00:00Z") for key in EXPERIMENT.months
    ]


@pytest.mark.parametrize("name", list(CASES))
def test_a_decision_rerun_at_its_own_month_changes_only_months_the_audit_flags(root, name):
    case = CASES[name]
    catalog, store = load_catalog(root), root / "datasets"
    authority = CONTEXT.narrow(cutoff=EXPERIMENT.cutoff)
    decisions = case.decide(case.program, catalog, store, authority, EXPERIMENT.months)
    changed = set()
    for item in decisions:
        narrowed = authority.narrow(cutoff=midnight(item.month))
        (rerun,) = case.decide(case.program, catalog, store, narrowed, [item.month])
        if (rerun.observation, rerun.value) != (item.observation, item.value):
            changed.add(item.month)
    assert changed <= {breach.month for breach in reference.audit(decisions)}
    expected = {FIXED: {month("2022-11"), month("2022-12")}, CURRENT: {month("2022-05")}}
    assert changed == expected.get(name, set())


def test_the_audit_flags_a_missing_read_a_late_read_and_a_late_release():
    def decision(key: str, read_at: datetime | None, released_at: datetime | None) -> Decision:
        reads = (
            () if read_at is None else (ReadRecord("d", "0" * 64, "1" * 64, read_at, "2" * 64, ()),)
        )
        return Decision(month(key), None, None, released_at, reads)

    start = midnight(month("2022-05"))
    later = start + timedelta(microseconds=1)
    decisions = [
        decision("2022-05", start, start),
        decision("2022-05", None, None),
        decision("2022-05", later, None),
        decision("2022-05", start, later),
    ]
    assert reference.audit(decisions) == (
        reference.Breach(month("2022-05"), None),
        reference.Breach(month("2022-05"), later),
        reference.Breach(month("2022-05"), later),
    )


def test_the_never_bills_control_records_null_with_a_zero_incremental_return(saved):
    package = saved[NEVER]
    assert outcome(package).status == "null"
    produced = results(package)
    result = reference.decode_evaluation(produced["pilot-evaluation"])
    assert (result.verdict, result.incremental.estimate, result.incremental.upper) == (
        "null",
        0.0,
        0.0,
    )
    rows = json.loads(produced["returns"])["months"]
    assert [row["month"] for row in rows] == [key.isoformat() for key in EXPERIMENT.months]
    assert all(row["holding"] == "market" and row["strategy"] == row["baseline"] for row in rows)


def test_the_reference_case_is_evaluated_on_the_approved_vintages(root, saved):
    package = open_package(saved[REFERENCE].path)
    assert outcome(package).status in CASES[REFERENCE].expected
    catalog = load_catalog(root)
    vintages = {catalog.entry(name).documentation.vintage: name for name in PILOT}
    spec = package.record.selected_trial.specification
    assert spec.input_digests == tuple(sorted(vintages))
    assert spec.data_contract_digest == EXPERIMENT.digest
    assert spec.evaluation_profile == "pilot-evaluation/v1"
    for item in package.artifact_records:
        if item.role == "input":
            stored = root / "datasets" / vintages[item.digest] / f"{item.digest}.json"
            assert package.read_artifact(item.name) == stored.read_bytes()
    produced = results(package)
    assert set(produced) == {
        "decisions",
        "audit",
        "reconciliation",
        "returns",
        "engine-agreement",
        "pilot-evaluation",
    }
    assert json.loads(produced["audit"])["breaches"] == []
    assert json.loads(produced["reconciliation"])["agreed"] is True
    assert json.loads(produced["engine-agreement"])["agreed"] is True
    rows = json.loads(produced["returns"])["months"]
    assert [month(row["month"][:7]) for row in rows if row["holding"] == "bills"] == BILLS
    exposures = {(event.kind, event.dataset_id) for event in package.record.exposures}
    assert exposures == {("data", name) for name in PILOT} | {("result", "french.ff3_monthly")}
    assert package.record.disclosure_coverage == "declared_inputs"


def test_a_fresh_run_reproduces_every_result_artifact_and_specification(root, saved, tmp_path):
    for case in reference.CASES:
        first, second = saved[case.name], run(case, root, tmp_path / "again")
        assert results(second) == results(first)
        spec_ids = [
            package.record.selected_trial.specification.spec_id for package in (first, second)
        ]
        assert spec_ids[0] == spec_ids[1]
        opened = open_package(first.path)
        resaved = save_package(tmp_path / "resaved", opened.record, opened.artifacts())
        assert resaved.digest == first.digest


def test_a_changed_evaluation_parameter_changes_the_specification(root, saved, tmp_path):
    changed = reference.load_experiment(experiment_bytes(minimum_effect="0.002"))
    package = run(CASES[CURRENT], root, tmp_path, changed)
    specs = [item.record.selected_trial.specification for item in (saved[CURRENT], package)]
    assert specs[1].data_contract_digest == hashlib.sha256(changed.data).hexdigest()
    assert specs[0].spec_id != specs[1].spec_id


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(json.dumps(json.loads(experiment_bytes()), indent=1).encode(), id="pretty"),
        pytest.param(experiment_bytes(note="frozen"), id="unknown field"),
        pytest.param(experiment_bytes(schema_version=2), id="schema version"),
        pytest.param(experiment_bytes(baseline="buy_and_hold_bills"), id="other baseline"),
        pytest.param(experiment_bytes(version="pilot-evaluation/v2"), id="other procedure"),
        pytest.param(experiment_bytes(start="2021-01-15"), id="mid-month start"),
        pytest.param(experiment_bytes(end="2023-05-01"), id="29-month window"),
        pytest.param(experiment_bytes(cutoff="2023-12-31T00:00:00Z"), id="cutoff before returns"),
        pytest.param(experiment_bytes(alpha="1.5"), id="alpha above 1"),
        pytest.param(experiment_bytes(subperiods=1), id="one subperiod"),
        pytest.param(experiment_bytes(minimum_effect="0.0010"), id="trailing zero"),
        pytest.param(experiment_bytes(minimum_effect=0.001), id="number not text"),
    ],
)
def test_an_experiment_outside_the_frozen_format_is_refused(data):
    with pytest.raises(reference.ReferencePipelineError):
        reference.load_experiment(data)


def test_a_window_of_exactly_the_needed_months_ending_at_its_cutoff_loads():
    experiment = reference.load_experiment(
        experiment_bytes(end="2023-06-01", cutoff="2023-07-01T00:00:00Z")
    )
    assert len(experiment.months) == 30


def test_the_committed_experiment_loads_and_its_digest_is_pinned():
    experiment = reference.frozen()
    assert experiment.digest == "af4fafb3df64b497d9a02e0c2122902a72a8968092d1b1460cc4d563d5d0c8b9"
    assert experiment.profile == Profile(0.001, 4, 0.05)


PACKAGE_RULES = ROOT / "src" / "quant_research" / "rules"
PILOT_RULES = "6387e6b8c71e4b825295ab7c92cbca7d3230afabc564788fea32b62b4bb3b8b0"
SEARCH_WINDOW = experiment_bytes(
    start="2020-07-01", end="2022-12-01", cutoff="2023-01-01T00:00:00Z", subperiods=2
)
HELD_BACK_WINDOW = experiment_bytes(
    start="2023-01-01", end="2025-06-01", cutoff="2025-07-01T00:00:00Z", subperiods=2
)


def rules_bytes(search: bytes = SEARCH_WINDOW, held_back: bytes = HELD_BACK_WINDOW, **changes):
    document = {
        "held_back": json.loads(held_back),
        "schema_version": 1,
        "search": json.loads(search),
    }
    return canonical_bytes(document | changes)


def test_the_committed_pilot_rules_load_and_each_rules_file_is_named_by_its_digest():
    pilot = reference.rules(PACKAGE_RULES, PILOT_RULES)
    assert pilot.digest == PILOT_RULES
    assert [path.stem for path in PACKAGE_RULES.iterdir()] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in PACKAGE_RULES.iterdir()
    ]
    search, held_back = pilot.search, pilot.held_back
    assert (search.start, search.end, len(search.months)) == (
        month("1963-01"),
        month("2004-12"),
        504,
    )
    assert search.cutoff == datetime(2005, 1, 1, tzinfo=UTC)
    assert (held_back.start, held_back.end, len(held_back.months)) == (
        month("2005-01"),
        month("2026-08"),
        260,
    )
    assert held_back.cutoff == datetime(2026, 10, 23, tzinfo=UTC)
    assert (search.profile, held_back.profile) == (Profile(0.001, 4, 0.05), Profile(0.001, 2, 0.05))
    assert search.digest == "6727ff18343b2ca8dc5469e6dfc00e71f0302fb4512566dffbc73928c76735be"


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(json.dumps(json.loads(rules_bytes()), indent=1).encode(), id="pretty"),
        pytest.param(rules_bytes(method="search-then-held-back/v1"), id="unknown field"),
        pytest.param(
            canonical_bytes({"schema_version": 1, "search": json.loads(SEARCH_WINDOW)}),
            id="no held-back window",
        ),
        pytest.param(rules_bytes(schema_version=2), id="schema version"),
        pytest.param(rules_bytes(search=experiment_bytes(end="2023-05-01")), id="29-month window"),
        pytest.param(
            rules_bytes(
                held_back=experiment_bytes(
                    start="2023-02-01", end="2025-07-01", cutoff="2025-08-01T00:00:00Z"
                )
            ),
            id="gap between windows",
        ),
        pytest.param(
            rules_bytes(
                held_back=experiment_bytes(
                    start="2022-12-01", end="2025-06-01", cutoff="2025-07-01T00:00:00Z"
                )
            ),
            id="overlapping windows",
        ),
        pytest.param(
            rules_bytes(
                search=experiment_bytes(
                    start="2020-07-01", end="2022-12-01", cutoff="2023-01-01T00:00:01Z"
                )
            ),
            id="search cutoff past the held-back start",
        ),
        pytest.param(
            rules_bytes(search=HELD_BACK_WINDOW, held_back=SEARCH_WINDOW), id="held back first"
        ),
    ],
)
def test_rules_outside_the_frozen_format_are_refused(data):
    with pytest.raises(reference.ReferencePipelineError):
        reference.load_rules(data)


def test_rules_are_found_by_digest_and_a_missing_renamed_or_invalid_file_is_refused(tmp_path):
    data = rules_bytes()
    digest = hashlib.sha256(data).hexdigest()
    (tmp_path / f"{digest}.json").write_bytes(data)
    found = reference.rules(tmp_path, digest)
    assert (found.data, found.search.digest) == (data, hashlib.sha256(SEARCH_WINDOW).hexdigest())
    other = rules_bytes(search=SEARCH_WINDOW.replace(b'"0.001"', b'"0.002"'))
    (tmp_path / f"{hashlib.sha256(other).hexdigest()}.json").write_bytes(data)
    invalid = json.dumps(json.loads(data), indent=1).encode()
    (tmp_path / f"{hashlib.sha256(invalid).hexdigest()}.json").write_bytes(invalid)
    for name in (
        "0" * 64,
        hashlib.sha256(other).hexdigest(),
        hashlib.sha256(invalid).hexdigest(),
        f"../{digest}",
    ):
        with pytest.raises(reference.ReferencePipelineError):
            reference.rules(tmp_path, name)


SEARCHED = reference.load_rules(rules_bytes())
LINEAGE = UUID("6b2f8a54-21c9-4e0e-9a43-6f1c2d7e9b10")
CODEX = Provider("codex-subscription")
UNRATE = "alfred.unrate"
SAHM_TEXT = canonical(SAHM_RULE)
RISE_TEXT = canonical(Threshold(Subtract(Series(UNRATE), Lag(Series(UNRATE), 12)), Fraction(1, 2)))
LEAKY_TEXT = canonical(Threshold(Lag(Series(UNRATE), 1), Fraction(5))).replace(
    b'"months":1', b'"months":-1'
)
EVALUATED = {"successful", "null", "inconclusive"}


def searching(root: Path, tmp_path: Path, scope=(UNRATE, FACTORS)) -> reference.Search:
    """Return a search over the module root whose context ends at the search window's cutoff."""
    context = issue_context(
        clearances=(),
        cutoff=NOW,
        generation=reference.GENERATION,
        providers=(CODEX,),
        scope=scope,
    )
    return reference.Search(
        "Does a rise in unemployment time the market?",
        LINEAGE,
        SEARCHED,
        load_catalog(root),
        root / "datasets",
        context.narrow(cutoff=SEARCHED.search.cutoff),
        LOCK,
        tmp_path / str(LINEAGE),
    )


def floor(search: reference.Search) -> Restrictions:
    return Restrictions(
        available_at=search.context.cutoff,
        labels=frozenset(),
        approved_for=search.context.providers,
    )


def roles(package: VerifiedPackage, trial) -> dict[str, bytes]:
    """Return the bytes of each role a trial's specification or outcome names, by role."""
    named = set(trial.attempts[0].outcome.result_artifacts)
    spec = trial.specification
    digests = {*spec.input_digests, spec.data_contract_digest, spec.implementation_digest}
    digests.add(spec.environment.lock_digest)
    program = spec.signal_profile.removeprefix(f"{reference.SEARCH_PROFILE} ")
    return {
        item.role: package.read_artifact(item.name)
        for item in package.artifact_records
        if item.name in named or item.digest in digests or item.digest == program
    }


def test_extend_and_forfeit_grow_one_lineage_whose_trials_follow_each_other(root, tmp_path):
    search = searching(root, tmp_path)
    ids = [uuid4() for _ in range(4)]
    first = reference.extend(search, None, ids[0], SAHM_TEXT)
    second = reference.extend(search, first.package.digest, ids[1], LEAKY_TEXT)
    ended = NOW + timedelta(minutes=5)
    third = reference.forfeit(search, second.package.digest, ids[2], RISE_TEXT, NOW, ended)
    fourth = reference.extend(search, third, ids[3], RISE_TEXT)
    package = reference.open_lineage(search, fourth.package.digest)
    record = package.record
    assert package.path == tmp_path / str(LINEAGE) / "unselected" / package.digest
    assert record.trial_id is None
    assert [trial.trial_id for trial in record.trials] == ids
    assert {trial.lineage_id for trial in record.trials} == {LINEAGE}
    assert [trial.links for trial in record.trials] == [
        (),
        *(
            (LineageLink(prior.trial_id, prior.specification.spec_id, "follows", reference.LINK),)
            for prior in record.trials[:-1]
        ),
    ]
    outcomes = [trial.attempts[0].outcome for trial in record.trials]
    assert {outcomes[0].status, outcomes[3].status} <= EVALUATED
    assert [outcome.status for outcome in outcomes[1:3]] == ["rejected", "failed"]
    assert outcomes[1].detail.startswith("Rejected (negative_lag): ")
    assert outcomes[2].detail == reference.UNFINISHED
    forfeited = record.trials[2].attempts[0]
    assert (forfeited.started_at, forfeited.finished_at) == (NOW, ended)
    counted = [
        reference.decode_evaluation(roles(package, record.trials[index])["pilot-evaluation"])
        for index in (0, 3)
    ]
    assert [result.trials for result in counted] == [1, 4]
    for trial in record.trials[1:3]:
        assert trial.specification.input_digests == ()
        assert set(roles(package, trial)) == {"contract", "implementation", "lock", "program"}
    for head, count in ((first.package.digest, 1), (second.package.digest, 2), (third, 3)):
        assert reference.open_lineage(search, head).record.trials == record.trials[:count]
    # The forfeited run may have read what no exposure records, and later heads keep that.
    covered = [
        reference.open_lineage(search, head).record.disclosure_coverage
        for head in (first.package.digest, second.package.digest, third)
    ]
    assert [*covered, record.disclosure_coverage] == ["declared_inputs"] * 2 + ["unknown"] * 2
    assert open_package(package.path).digest == package.digest


def test_each_candidate_names_its_program_by_digest_and_one_program_has_one_specification(
    root, tmp_path
):
    search = searching(root, tmp_path)
    spelled = SAHM_TEXT.replace(b'"1/2"', b'"0.5"')
    first = reference.extend(search, None, uuid4(), SAHM_TEXT)
    second = reference.extend(search, first.package.digest, uuid4(), spelled)
    third = reference.extend(search, second.package.digest, uuid4(), RISE_TEXT)
    package = third.package
    trials = package.record.trials
    specs = [trial.specification for trial in trials]
    assert specs[0].signal_profile == f"search/v1 {signals.digest(SAHM_RULE)}"
    assert specs[0].spec_id == specs[1].spec_id != specs[2].spec_id
    assert specs[2].signal_profile == f"search/v1 {hashlib.sha256(RISE_TEXT).hexdigest()}"
    assert [roles(package, trial)["program"] for trial in trials] == [
        SAHM_TEXT,
        SAHM_TEXT,
        RISE_TEXT,
    ]
    assert specs[0].input_digests == tuple(
        sorted(load_catalog(root).entry(name).documentation.vintage for name in (UNRATE, FACTORS))
    )


@pytest.mark.parametrize(
    ("program", "reason", "scope"),
    [
        pytest.param(LEAKY_TEXT, "negative_lag", (UNRATE, FACTORS), id="negative lag"),
        pytest.param(
            canonical(Threshold(Series(UNRATE, vintage=1), Fraction(5))).replace(
                b'"vintage":1', b'"vintage":-1'
            ),
            "reads_past_vintage",
            (UNRATE, FACTORS),
            id="past vintage",
        ),
        pytest.param(
            canonical(Threshold(Series("alfred.sahmrealtime"), Fraction(1, 2))),
            "unapproved_dataset",
            (UNRATE, FACTORS, "alfred.sahmrealtime"),
            id="outside the universe",
        ),
        pytest.param(SAHM_TEXT, "unapproved_dataset", (FACTORS,), id="outside the scope"),
        pytest.param(canonical(Series(UNRATE)), "wrong_type", (UNRATE, FACTORS), id="a number"),
        pytest.param(b"momentum", "malformed_json", (UNRATE, FACTORS), id="not json"),
    ],
)
def test_a_refused_program_is_a_rejected_trial_that_read_nothing(
    root, tmp_path, program, reason, scope
):
    search = searching(root, tmp_path, scope)
    trial = uuid4()
    extended = reference.extend(search, None, trial, program)
    package = extended.package
    [recorded] = package.record.trials
    outcome = recorded.attempts[0].outcome
    assert extended.status == outcome.status == "rejected"
    assert outcome.detail == f"Rejected ({reason}): {signals.Rejected(reason).message}"
    assert recorded.specification.input_digests == ()
    assert package.record.exposures == ()
    stored = roles(package, recorded)
    assert set(stored) == {"contract", "implementation", "lock", "program"}
    assert stored["contract"] == SEARCHED.search.data
    digest = hashlib.sha256(stored["program"]).hexdigest()
    assert recorded.specification.signal_profile == f"search/v1 {digest}"
    assert json.loads(extended.summary) == {
        "detail": outcome.detail,
        "evaluation": None,
        "status": "rejected",
        "trial": str(trial),
    }
    assert extended.restrictions == floor(search)


def test_the_summary_holds_the_outcome_and_evaluation_under_the_restrictions_of_every_read(
    root, tmp_path
):
    search = searching(root, tmp_path)
    trial = uuid4()
    extended = reference.extend(search, None, trial, SAHM_TEXT)
    package = extended.package
    [recorded] = package.record.trials
    stored = roles(package, recorded)
    summary = json.loads(extended.summary)
    assert canonical_bytes(summary).decode() == extended.summary
    assert summary == {
        "detail": recorded.attempts[0].outcome.detail,
        "evaluation": json.loads(stored["pilot-evaluation"]),
        "status": extended.status,
        "trial": str(trial),
    }
    assert extended.status in EVALUATED
    assert len(extended.summary) < 8 * 2**10
    for item in package.artifacts():
        if item.role in ("input", "decisions", "returns"):
            assert item.data.decode()[:200] not in extended.summary
    catalog, store = load_catalog(root), root / "datasets"
    read = [datasets.read(catalog, store, name, search.context) for name in (UNRATE, FACTORS)]
    assert extended.restrictions == derive(floor(search), *(item.restrictions for item in read))
    assert extended.restrictions.available_at == SEARCHED.search.cutoff
    assert extended.restrictions.approved_for == frozenset() != floor(search).approved_for


def test_a_search_trial_reads_nothing_after_the_search_cutoff_and_cannot_decide_past_it(
    root, tmp_path
):
    search = searching(root, tmp_path)
    package = reference.extend(search, None, uuid4(), SAHM_TEXT).package
    [recorded] = package.record.trials
    decisions = json.loads(roles(package, recorded)["decisions"])["decisions"]
    cutoffs = {read["cutoff"] for item in decisions for read in item["reads"]}
    assert max(cutoffs) <= "2023-01-01T00:00:00Z"
    assert all(
        event.interval_end <= midnight(month("2022-12")) for event in package.record.exposures
    )
    catalog, store = load_catalog(root), root / "datasets"
    supplied = datasets.read(catalog, store, UNRATE, search.context)
    assert max(row.available_at for row in supplied.rows) <= SEARCHED.search.cutoff
    assert (
        max(row.key for row in supplied.rows)
        < "2023-01-01"
        < max(row.key for row in datasets.read(catalog, store, UNRATE, CONTEXT).rows)
    )
    with pytest.raises(AccessError):
        decide(SAHM_RULE, catalog, store, search.context, [month("2023-02")])


def test_a_search_that_cannot_read_the_factors_fails_each_trial_before_any_read(root, tmp_path):
    search = searching(root, tmp_path, scope=(UNRATE,))
    first = reference.extend(search, None, uuid4(), SAHM_TEXT)
    assert first.status == "failed"
    [recorded] = first.package.record.trials
    assert recorded.attempts[0].outcome.detail == (
        "The search cannot read french.ff3_monthly; nothing was read"
    )
    assert recorded.specification.input_digests == ()
    assert first.package.record.exposures == ()
    second = reference.forfeit(search, first.package.digest, uuid4(), b"momentum", NOW, NOW)
    assert len(reference.open_lineage(search, second).record.trials) == 2


def test_a_frozen_shortlist_appends_real_held_back_trials_without_changing_search(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(globals(), "MONTHS", span("2019-01", "2025-06"))
    root = approved(tmp_path / "expanded", {UNRATE: unrate(), FACTORS: french()})
    search = searching(root, tmp_path)
    first = reference.extend(search, None, uuid4(), SAHM_TEXT)
    second = reference.extend(search, first.package.digest, uuid4(), RISE_TEXT)
    originals = second.package.record.trials
    pairs = tuple((trial.trial_id, uuid4()) for trial in originals)
    assessed = reference.assess(search, CONTEXT, second.package.digest, pairs, adjustment_trials=2)
    reopened = reference.open_lineage(search, assessed.package)
    assert reopened.record.trial_id is None
    assert reopened.record.trials[:2] == originals
    assert len(reopened.record.trials) == 4 > 2
    assert {trial.lineage_id for trial in reopened.record.trials} == {LINEAGE}
    summaries = json.loads(assessed.summary)
    assert [(item["source"], item["trial"]) for item in summaries] == [
        (str(source), str(new)) for source, new in pairs
    ]
    for original, measurement, summary in zip(
        originals, reopened.record.trials[2:], summaries, strict=True
    ):
        assert measurement.trial_id != original.trial_id
        assert measurement.specification.signal_profile == original.specification.signal_profile
        assert measurement.specification.data_contract_digest == SEARCHED.held_back.digest
        assert measurement.links[-1] == LineageLink(
            original.trial_id,
            original.specification.spec_id,
            "held_back_of",
            "Frozen shortlist candidate",
        )
        material = roles(reopened, measurement)
        assert material["program"] == roles(second.package, original)["program"]
        assert {
            "audit",
            "decisions",
            "returns",
            "engine-agreement",
            "implementation",
            "lock",
            "input",
        } <= material.keys()
        assert summary["evaluation"]["result"]["trials"] == 2
        assert summary["evaluation"]["result"]["profile"]["alpha"] == 0.05
        assert json.loads(material["pilot-evaluation"]) == summary["evaluation"]
    assert reference.open_lineage(search, second.package.digest).record.trials == originals
    assert any(
        event.kind == "result" and event.interval_start == midnight(SEARCHED.held_back.start)
        for event in reopened.record.exposures
    )
    assert assessed.restrictions.available_at == SEARCHED.held_back.cutoff
    recovery = reference.assessment_reference(
        search.lineage, search.rules.digest, second.package.digest, pairs, adjustment_trials=2
    )
    stale = first.package.path / "artifacts" / first.package.artifact_records[0].digest
    stale.write_bytes(b"unrelated damaged package")
    opened = []
    original_open = reference.open_package

    def opening(path):
        opened.append(path.name)
        return original_open(path)

    monkeypatch.setattr(reference, "open_package", opening)
    assert reference.recover_assessment(search, recovery) == assessed
    assert first.package.digest not in opened
    assert opened.count(second.package.digest) == 1


@pytest.mark.parametrize(
    ("profile", "error"),
    [("other/v1", reference.ReferencePipelineError), (f"search/v1 {'0' * 64}", KeyError)],
)
def test_an_invalid_second_shortlist_source_publishes_no_held_back_measurement(
    root, tmp_path, profile, error
):
    search = searching(root, tmp_path)
    first = reference.forfeit(search, None, uuid4(), SAHM_TEXT, NOW, NOW)
    second = reference.forfeit(search, first, uuid4(), RISE_TEXT, NOW, NOW)
    package = reference.open_lineage(search, second)
    original, invalid = package.record.trials
    changed = replace(invalid, specification=replace(invalid.specification, signal_profile=profile))
    frozen = save_package(
        search.destination,
        replace(package.record, trials=(original, changed)),
        package.artifacts(),
    )
    before = set((search.destination / "unselected").iterdir())
    pairs = tuple((trial.trial_id, uuid4()) for trial in frozen.record.trials)
    with pytest.raises(error):
        reference.assess(search, CONTEXT, frozen.digest, pairs, adjustment_trials=2)
    assert set((search.destination / "unselected").iterdir()) == before


@pytest.mark.parametrize("failure", ["second_measurement", "published_return"])
def test_published_held_back_evidence_survives_a_failed_or_lost_return(
    tmp_path, monkeypatch, failure
):
    monkeypatch.setitem(globals(), "MONTHS", span("2019-01", "2025-06"))
    root = approved(tmp_path / "expanded", {UNRATE: unrate(), FACTORS: french()})
    search = searching(root, tmp_path)
    label = licensed("saved-policy")
    context = issue_context(
        clearances=(label,), cutoff=NOW, generation=reference.GENERATION, providers=(CODEX,)
    )
    search = replace(
        search,
        catalog=Catalog(
            tuple(
                replace(event, access=(label,), approved_for=(CODEX,))
                if isinstance(event, Documented)
                else event
                for event in search.catalog.events
            )
        ),
        context=context.narrow(cutoff=search.rules.search.cutoff),
    )
    first = reference.extend(search, None, uuid4(), SAHM_TEXT)
    second = reference.extend(search, first.package.digest, uuid4(), RISE_TEXT)
    pairs = tuple((trial.trial_id, uuid4()) for trial in second.package.record.trials)
    recovery = reference.assessment_reference(
        search.lineage, search.rules.digest, second.package.digest, pairs, adjustment_trials=2
    )
    examine, append = reference._examine, reference._append
    measured, published = [], []

    def measure(*args, **kwargs):
        measured.append(args)
        if len(measured) == 2:
            raise RuntimeError("second measurement failed")
        return examine(*args, **kwargs)

    def publish(*args, **kwargs):
        package = append(*args, **kwargs)
        published.append(package)
        if failure == "published_return":
            raise RuntimeError("published return lost")
        return package

    monkeypatch.setattr(reference, "_examine", measure)
    monkeypatch.setattr(reference, "_append", publish)
    with pytest.raises(RuntimeError, match="second measurement failed|published return lost"):
        reference.assess(search, context, second.package.digest, pairs, adjustment_trials=2)

    def unexpected(*args, **kwargs):
        pytest.fail("Recovery must only read published packages")

    monkeypatch.setattr(reference, "_examine", unexpected)
    monkeypatch.setattr(reference, "_read", unexpected)
    monkeypatch.setattr(reference, "_append", unexpected)
    recovered = reference.recover_assessment(
        replace(search, catalog=Catalog(), store=tmp_path / "missing"), recovery
    )
    assert recovered == reference.recover_assessment(search, recovery)
    assert recovered.package == published[0].digest
    assert recovered.restrictions == Restrictions(
        available_at=SEARCHED.held_back.cutoff, labels={label}, approved_for={CODEX}
    )
    assert published[0].record.trials[:2] == second.package.record.trials
    [summary] = json.loads(recovered.summary)
    assert (summary["source"], summary["trial"]) == tuple(map(str, pairs[0]))
    assert summary["evaluation"] == json.loads(
        roles(published[0], published[0].record.trials[-1])["pilot-evaluation"]
    )
    assert len(measured) == (2 if failure == "second_measurement" else 1)
    missing = reference.assessment_reference(
        search.lineage, search.rules.digest, second.package.digest, pairs, adjustment_trials=3
    )
    assert reference.recover_assessment(search, missing) is None


@pytest.mark.parametrize("changed", ["request", "summary", "source", "conflicting_head"])
def test_recovery_refuses_a_receipt_that_conflicts_with_its_published_measurement(
    root, tmp_path, changed
):
    search = searching(root, tmp_path)
    candidate = reference.extend(search, None, uuid4(), LEAKY_TEXT)
    pairs = ((candidate.package.record.trials[0].trial_id, uuid4()),)
    assessed = reference.assess(
        search, CONTEXT, candidate.package.digest, pairs, adjustment_trials=2
    )
    recovery = reference.assessment_reference(
        search.lineage, search.rules.digest, candidate.package.digest, pairs, adjustment_trials=2
    )
    package = reference.open_lineage(search, assessed.package)
    artifacts, record = list(package.artifacts()), package.record
    if changed in {"request", "summary"}:
        index = next(i for i, item in enumerate(artifacts) if item.name == f"held-back-{recovery}")
        receipt = json.loads(artifacts[index].data)
        if changed == "request":
            receipt["request"]["trials"][0][1] = str(uuid4())
        else:
            material = load_material(receipt["material"])
            summary = json.loads(material.text)
            summary[0]["status"] = "null"
            receipt["material"] = dump_material(
                Material(canonical_bytes(summary).decode(), material.restrictions)
            )
        artifacts[index] = replace(artifacts[index], data=canonical_bytes(receipt))
    elif changed == "source":
        trial = record.trials[-1]
        record = replace(
            record,
            trials=(
                *record.trials[:-1],
                replace(trial, links=(replace(trial.links[0], relation="follows"),)),
            ),
        )
    else:
        artifacts.append(ArtifactInput("conflict", b"different head", "text/plain", "note"))
    save_package(search.destination, record, tuple(artifacts))
    with pytest.raises(
        reference.ReferencePipelineError, match="request|summary|source|Conflicting"
    ):
        reference.recover_assessment(search, recovery)


def test_an_empty_held_back_shortlist_reads_nothing_and_saves_no_package(
    root, tmp_path, monkeypatch
):
    def unexpected(*args, **kwargs):
        pytest.fail("An empty shortlist must not read a dataset")

    search = searching(root, tmp_path)
    monkeypatch.setattr(reference, "_read", unexpected)
    assessed = reference.assess(search, CONTEXT, None, (), adjustment_trials=5)
    assert (assessed.package, json.loads(assessed.summary)) == ("", [])


def test_rejected_shortlist_candidates_have_no_verdict_and_unknown_ids_are_refused(root, tmp_path):
    search = searching(root, tmp_path)
    rejected = reference.extend(search, None, uuid4(), LEAKY_TEXT)
    source = rejected.package.record.trials[0].trial_id
    assessed = reference.assess(
        search, CONTEXT, rejected.package.digest, ((source, uuid4()),), adjustment_trials=5
    )
    [summary] = json.loads(assessed.summary)
    assert summary["status"] == "rejected"
    assert summary["evaluation"] is None
    for pairs in (
        ((uuid4(), uuid4()),),
        ((source, uuid4()), (source, uuid4())),
        ((source, source),),
    ):
        with pytest.raises(reference.ReferencePipelineError, match="unique known"):
            reference.assess(search, CONTEXT, rejected.package.digest, pairs, adjustment_trials=5)


def test_the_evaluation_codec_round_trips_bytes_and_refuses_another_procedure(saved):
    data = results(saved[REFERENCE])["pilot-evaluation"]
    assert reference.encode_evaluation(reference.decode_evaluation(data)) == data
    invalid = Invalid("zero_variance", "Returns are constant", Profile(0.001, 2, 0.05))
    encoded = reference.encode_evaluation(invalid)
    assert reference.decode_evaluation(encoded) == invalid
    assert reference.encode_evaluation(reference.decode_evaluation(encoded)) == encoded
    document = json.loads(data)
    document["result"]["profile"]["version"] = "pilot-evaluation/v2"
    with pytest.raises(PackageError):
        reference.decode_evaluation(canonical_bytes(document))


def command(tmp_path: Path, root: Path) -> subprocess.CompletedProcess[str]:
    """Run reference run on root and the test experiment, with the site-packages vectorbt needs."""
    (tmp_path / "experiment.json").write_bytes(experiment_bytes())
    (tmp_path / "lock.txt").write_bytes(LOCK)
    arguments = [
        *("reference", "run", "--root", str(root), "--output", str(tmp_path / "packages")),
        *("--lock", str(tmp_path / "lock.txt"), "--experiment", str(tmp_path / "experiment.json")),
    ]
    bootstrap = (
        f"import runpy, sys; sys.path.append({str(ROOT / 'src')!r}); "
        f"sys.argv = {['quant_research', *arguments]!r}; "
        "runpy.run_module('quant_research', run_name='__main__')"
    )
    return subprocess.run(
        [sys.executable, "-B", "-I", "-c", bootstrap], cwd=tmp_path, text=True, capture_output=True
    )


def test_the_reference_command_saves_every_case_in_packages_that_verify_and_report(
    root, saved, tmp_path, cli
):
    result = command(tmp_path, root)
    assert result.returncode == 0, result.stderr
    rows = json.loads(result.stdout)
    assert [row["case"] for row in rows] == list(CASES)
    statuses = {row["case"]: row["status"] for row in rows}
    assert {name: statuses[name] for name in (CURRENT, FIXED, NEVER)} == {
        CURRENT: "rejected",
        FIXED: "rejected",
        NEVER: "null",
    }
    for row in rows:
        case, package = CASES[row["case"]], open_package(Path(row["package_path"]))
        assert row["status"] in case.expected
        assert (row["expected"], row["as_expected"]) == (sorted(case.expected), True)
        assert (row["package_digest"], row["status"], row["detail"]) == (
            package.digest,
            outcome(package).status,
            outcome(package).detail,
        )
        assert results(package) == results(saved[case.name])
    path = next(row["package_path"] for row in rows if row["case"] == REFERENCE)
    package = open_package(Path(path))
    assert cli(tmp_path, "verify", path).stdout == f"Verified {package.digest} at {package.path}\n"
    verdict = reference.decode_evaluation(results(package)["pilot-evaluation"]).verdict
    assert f"\n- Evaluation verdict: {verdict} (" in cli(tmp_path, "report", path).stdout


def test_the_reference_command_exits_with_the_adapter_refusal_for_unapproved_data(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    pull(data, "alfred.unrate", now=NOW, api_key="k", fetch=lambda url: unrate())
    ingest(data, "alfred.unrate", now=NOW)
    validate(
        data, "alfred.unrate", actor="researcher", evidence=b"notes", summary="checked", now=NOW
    )
    result = command(tmp_path, data)
    assert result.returncode == 1
    assert result.stdout == ""
    assert Refused("validated").message in result.stderr
