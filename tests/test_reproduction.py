from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from quant_research.access import issue_context, licensed
from quant_research.locators import MARKDOWN_LOCATOR, MARKDOWN_LOCATOR_V2, publish_locations
from quant_research.records import canonical_bytes
from quant_research.reproduction import (
    Choice,
    Classification,
    Locator,
    LocatorPin,
    OpenQuestion,
    Reproduction,
    ReproductionError,
    Result,
    Specification,
    Target,
    Tolerance,
    Verdict,
    freeze,
)
from quant_research.sources import (
    TEXT_PIPELINE,
    SourceDeclaration,
    publish_generation,
    register_source,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PAPER = FIXTURES / "reproduction" / "paper.md"
LOCATED_PAPER = FIXTURES / "locators" / "paper.md"
SIGNAL = ("universe", "lookback", "skip_month", "portfolios")
P1 = Locator("passage", "paper.md#p1")
P2 = Locator("passage", "paper.md#p2")
P3 = Locator("passage", "paper.md#p3")
P4 = Locator("passage", "paper.md#p4")
P5 = Locator("passage", "paper.md#p5")
EQ1 = Locator("equation", "paper.md#eq1")
TABLE1 = Locator("table", "paper.md#table1")
TABLE2 = Locator("table", "paper.md#table2")
APPENDIX_B = Locator("passage", "paper.md#appendix-b")
CODE = Locator(
    "code", "reference-code@9c1f4e2b7d3a5f6e8b0c1d2e3f4a5b6c7d8e9f01:signal.py#trailing_spread"
)
TIME = datetime(2026, 10, 21, 12, tzinfo=UTC)
DECLARATION = SourceDeclaration(
    media_type="text/markdown",
    access_domain="synthetic",
    location="tests/fixtures/locators/paper.md",
    revision="1",
    acquisition="hand-authored test fixture",
    rights="synthetic, unrestricted",
    labels=frozenset(),
    approved_for=frozenset(),
    published_at=TIME - timedelta(days=1),
    source_available_at=TIME,
    availability_evidence="authored for these tests",
    received_at=TIME,
)
VENDOR = licensed("synthetic-vendor")


def replace_first(items, **changes):
    return (replace(items[0], **changes), *items[1:])


def run(*clearances, cutoff=TIME):
    return issue_context(
        clearances=clearances, cutoff=cutoff, generation="synthetic/v1", providers=()
    )


def publish_paper(store, declaration=DECLARATION):
    """Register the locator fixture paper and publish its version 1 locations."""
    occurrence = register_source(store, LOCATED_PAPER.read_bytes(), declaration)
    publish_generation(store, occurrence.id, TEXT_PIPELINE)
    return publish_locations(store, occurrence.id, MARKDOWN_LOCATOR)


def cite(generation, kind):
    location = next(item for item in generation.locations if item.kind == kind)
    return Locator(kind, generation.ref(location))


def pinned_specification(generation):
    """Cite an equation, table, claim, and figure of the generation beside opaque refs."""
    equation, table, claim, figure = (
        cite(generation, kind) for kind in ("equation", "table", "claim", "figure")
    )
    signal = ("model", "portfolios")
    return Specification(
        "synthetic-paper:momentum@generation-1",
        "synthetic-exchange/v1",
        "matched_inputs",
        (
            Choice("model", "r_t = alpha + beta m_t", (equation, CODE)),
            Choice("portfolios", "10 decile portfolios", (P1,)),
        ),
        (
            OpenQuestion(
                "q-top-decile",
                "contradictory",
                "The text reports 1.2% a month for the top decile and the table reports 1.6%.",
                (claim, table),
            ),
        ),
        (
            Target("outperformance", 0.03, Tolerance(0.005), (figure,), depends_on=signal),
            Target(
                "top_decile_mean",
                None,
                Tolerance(0.001),
                question="q-top-decile",
                depends_on=signal,
            ),
        ),
        LocatorPin(generation.id, generation.digest),
    )


@pytest.fixture
def specification():
    return Specification(
        "synthetic-paper:trailing-spread@generation-1",
        "synthetic-exchange/v1",
        "matched_inputs",
        (
            Choice("universe", "40 synthetic assets traded monthly, months 1 to 120", (P1,)),
            Choice("lookback", "12 months", (EQ1,)),
            Choice("skip_month", "skip month t-1", (P3, EQ1, CODE)),
            Choice("portfolios", "5 equal-weighted portfolios, long top, short bottom", (P2,)),
            Choice("newey_west_lags", None, question="q-lags"),
            Choice("trading_costs", None, question="q-costs"),
        ),
        (
            OpenQuestion(
                "q-lags",
                "ambiguous",
                "The paper reports Newey-West t-statistics without a lag count.",
                (P4,),
            ),
            OpenQuestion(
                "q-costs",
                "missing",
                "Net returns deduct trading costs the paper never states.",
                (P4,),
            ),
            OpenQuestion(
                "q-turnover",
                "contradictory",
                "The text reports 31% top-quintile turnover and Table 2 reports 0.13.",
                (P5, TABLE2),
            ),
            OpenQuestion(
                "q-subperiods",
                "restricted",
                "Subperiod results are available to subscribers only.",
                (APPENDIX_B,),
            ),
        ),
        (
            Target("gross_mean", 0.0050, Tolerance(0.0005), (TABLE1,), depends_on=SIGNAL),
            Target(
                "gross_tstat",
                2.40,
                Tolerance(0.0, 0.1),
                (TABLE1,),
                depends_on=(*SIGNAL, "newey_west_lags"),
            ),
            Target(
                "net_mean",
                0.0038,
                Tolerance(0.0005),
                (TABLE2,),
                depends_on=(*SIGNAL, "trading_costs"),
            ),
            Target("turnover", None, Tolerance(0.01), question="q-turnover", depends_on=SIGNAL),
            Target(
                "subperiod_mean",
                None,
                Tolerance(0.0005),
                question="q-subperiods",
                depends_on=SIGNAL,
            ),
        ),
    )


def test_freezing_is_deterministic_and_identity_covers_tolerances_and_basis(specification):
    frozen = freeze(specification)
    tightened = replace(
        specification,
        targets=replace_first(specification.targets, tolerance=Tolerance(0.0004)),
    )
    assert frozen.specification == specification
    assert re.fullmatch(r"[0-9a-f]{64}", frozen.spec_id)
    assert freeze(copy.deepcopy(specification)).spec_id == frozen.spec_id
    assert freeze(tightened).spec_id != frozen.spec_id
    assert freeze(replace(specification, basis="adapted")).spec_id != frozen.spec_id
    assert frozen.results == ()
    assert frozen.classify() == ()


def test_revision_is_refused_once_a_result_exists(specification):
    frozen = freeze(specification)
    widened = replace(
        specification,
        targets=replace_first(specification.targets, tolerance=Tolerance(0.0010)),
    )
    revised = frozen.revise(widened)
    assert revised.specification == widened
    assert revised.spec_id == freeze(widened).spec_id
    assert revised.spec_id != frozen.spec_id
    assert revised.results == ()
    recorded = frozen.record("attempt-1", {"gross_mean": 0.0058})
    with pytest.raises(ReproductionError):
        recorded.revise(widened)
    assert recorded.classify() == (
        Classification(
            "attempt-1",
            frozen.spec_id,
            (
                Verdict("gross_mean", "matched_inputs", "outside_tolerance"),
                Verdict("gross_tstat", "matched_inputs", "open_question", ("q-lags",)),
                Verdict("net_mean", "matched_inputs", "open_question", ("q-costs",)),
                Verdict("turnover", "matched_inputs", "open_question", ("q-turnover",)),
                Verdict("subperiod_mean", "matched_inputs", "open_question", ("q-subperiods",)),
            ),
        ),
    )
    (widened_classification,) = revised.record("attempt-1", {"gross_mean": 0.0058}).classify()
    assert widened_classification.verdicts[0] == Verdict(
        "gross_mean", "matched_inputs", "within_tolerance"
    )


def test_results_stay_bound_to_the_specification_they_were_recorded_under(specification):
    widened = replace(
        specification,
        targets=replace_first(specification.targets, tolerance=Tolerance(0.0010)),
    )
    recorded = freeze(specification).record("attempt-1", {"gross_mean": 0.0058})
    with pytest.raises(ReproductionError):
        replace(recorded, specification=widened).classify()
    targets = list(specification.targets)
    mutable = freeze(replace(specification, targets=targets))
    mutable = mutable.record("attempt-1", {"gross_mean": 0.0058})
    targets[0] = replace(targets[0], tolerance=Tolerance(0.0010))
    with pytest.raises(ReproductionError):
        mutable.classify()
    unlocated = replace(specification, choices=replace_first(specification.choices, locators=()))
    with pytest.raises(ReproductionError):
        Reproduction(unlocated).record("attempt-1", {"gross_mean": 0.0053})


def test_recorded_observations_ignore_later_changes_to_the_callers_dict(specification):
    observations = {"gross_mean": 0.0053}
    reproduction = freeze(specification).record("attempt-1", observations)
    observations["gross_mean"] = 0.0060
    assert reproduction.results == (
        Result("attempt-1", reproduction.spec_id, (("gross_mean", 0.0053),)),
    )
    assert reproduction.classify() == (
        Classification(
            "attempt-1",
            reproduction.spec_id,
            (
                Verdict("gross_mean", "matched_inputs", "within_tolerance"),
                Verdict("gross_tstat", "matched_inputs", "open_question", ("q-lags",)),
                Verdict("net_mean", "matched_inputs", "open_question", ("q-costs",)),
                Verdict("turnover", "matched_inputs", "open_question", ("q-turnover",)),
                Verdict("subperiod_mean", "matched_inputs", "open_question", ("q-subperiods",)),
            ),
        ),
    )


@pytest.mark.smoke
def test_unresolved_inputs_leave_targets_inconclusive_whatever_their_values(specification):
    reproduction = (
        freeze(specification)
        .record(
            "attempt-1",
            {"gross_mean": 0.0053, "gross_tstat": 2.41, "net_mean": 0.0040, "turnover": 0.13},
        )
        .record(
            "attempt-2",
            {"gross_mean": 0.0060, "gross_tstat": 2.41, "net_mean": 0.0040, "turnover": 0.13},
        )
        .record("attempt-3", {"gross_mean": 0.0053, "net_mean": 0.5})
    )
    reproduced = Verdict("gross_mean", "matched_inputs", "within_tolerance")
    blocked = (
        Verdict("gross_tstat", "matched_inputs", "open_question", ("q-lags",)),
        Verdict("net_mean", "matched_inputs", "open_question", ("q-costs",)),
        Verdict("turnover", "matched_inputs", "open_question", ("q-turnover",)),
        Verdict("subperiod_mean", "matched_inputs", "open_question", ("q-subperiods",)),
    )
    classifications = reproduction.classify()
    assert classifications == (
        Classification("attempt-1", reproduction.spec_id, (reproduced, *blocked)),
        Classification(
            "attempt-2",
            reproduction.spec_id,
            (Verdict("gross_mean", "matched_inputs", "outside_tolerance"), *blocked),
        ),
        Classification("attempt-3", reproduction.spec_id, (reproduced, *blocked)),
    )
    first, second, _ = classifications
    assert [verdict.outcome for verdict in first.verdicts] == ["reproduced"] + ["inconclusive"] * 4
    assert [verdict.outcome for verdict in second.verdicts] == (
        ["not_reproduced"] + ["inconclusive"] * 4
    )
    assert [classification.outcome for classification in classifications] == [
        "inconclusive",
        "not_reproduced",
        "inconclusive",
    ]


def test_blocking_questions_list_the_targets_own_first_then_each_choice_question_once():
    spec = Specification(
        "synthetic-paper:trailing-spread@generation-1",
        "synthetic-exchange/v1",
        "matched_inputs",
        (
            Choice("lookback", "12 months", (EQ1,), question="q-lookback"),
            Choice("trading_costs", None, question="q-costs"),
            Choice("newey_west_lags", None, question="q-lags"),
        ),
        (
            OpenQuestion("q-lookback", "contradictory", "Sources disagree on the lookback.", (P3,)),
            OpenQuestion("q-costs", "missing", "The deducted trading costs are not stated.", (P4,)),
            OpenQuestion("q-lags", "ambiguous", "The Newey-West lag count is not stated.", (P4,)),
        ),
        (
            Target(
                "net_tstat",
                1.9,
                Tolerance(0.0, 0.1),
                (TABLE2,),
                question="q-lags",
                depends_on=("trading_costs", "lookback", "newey_west_lags"),
            ),
        ),
    )
    (classification,) = freeze(spec).record("attempt-1", {"net_tstat": 1.9}).classify()
    assert classification.verdicts == (
        Verdict(
            "net_tstat", "matched_inputs", "open_question", ("q-lags", "q-costs", "q-lookback")
        ),
    )
    assert classification.outcome == "inconclusive"


def test_adapted_data_never_reads_as_reproduction(specification):
    adapted = replace(specification, basis="adapted", data="synthetic-exchange-b/v1")
    reproduction = (
        freeze(adapted)
        .record("attempt-1", {"gross_mean": 0.0053})
        .record("attempt-2", {"gross_mean": 0.0060})
    )
    blocked = (
        Verdict("gross_tstat", "adapted", "open_question", ("q-lags",)),
        Verdict("net_mean", "adapted", "open_question", ("q-costs",)),
        Verdict("turnover", "adapted", "open_question", ("q-turnover",)),
        Verdict("subperiod_mean", "adapted", "open_question", ("q-subperiods",)),
    )
    within, outside = reproduction.classify()
    assert within.verdicts == (Verdict("gross_mean", "adapted", "within_tolerance"), *blocked)
    assert outside.verdicts == (Verdict("gross_mean", "adapted", "outside_tolerance"), *blocked)
    assert [verdict.outcome for verdict in within.verdicts] == (
        ["reproduced_on_adapted_data"] + ["inconclusive"] * 4
    )
    assert [verdict.outcome for verdict in outside.verdicts] == (
        ["not_reproduced_on_adapted_data"] + ["inconclusive"] * 4
    )
    assert (within.outcome, outside.outcome) == ("inconclusive", "not_reproduced_on_adapted_data")
    assert reproduction.spec_id != freeze(specification).spec_id


@pytest.mark.parametrize(
    ("basis", "positive", "negative"),
    [
        ("matched_inputs", "reproduced", "not_reproduced"),
        ("adapted", "reproduced_on_adapted_data", "not_reproduced_on_adapted_data"),
    ],
)
def test_tolerance_bounds_are_inclusive_and_unusable_values_are_inconclusive(
    basis, positive, negative
):
    targets = (
        *(
            Target(name, 0.5, Tolerance(0.25), (TABLE1,), depends_on=("signal",))
            for name in ("inside", "boundary", "outside", "missing", "nan", "inf")
        ),
        Target("relative", -2.0, Tolerance(0.0, 0.25), (TABLE1,), depends_on=("signal",)),
    )
    spec = Specification(
        "synthetic-paper:tolerances@generation-1",
        "synthetic-exchange/v1",
        basis,
        (Choice("signal", "trailing spread", (EQ1,)),),
        (),
        targets,
    )
    observations = {
        "inside": 0.625,
        "boundary": 0.75,
        "outside": 0.875,
        "nan": math.nan,
        "inf": math.inf,
        "relative": -1.5,
    }
    (mixed,) = freeze(spec).record("attempt-1", observations).classify()
    assert mixed.verdicts == (
        Verdict("inside", basis, "within_tolerance"),
        Verdict("boundary", basis, "within_tolerance"),
        Verdict("outside", basis, "outside_tolerance"),
        Verdict("missing", basis, "missing"),
        Verdict("nan", basis, "non_finite"),
        Verdict("inf", basis, "non_finite"),
        Verdict("relative", basis, "within_tolerance"),
    )
    assert [verdict.outcome for verdict in mixed.verdicts] == [
        positive,
        positive,
        negative,
        "inconclusive",
        "inconclusive",
        "inconclusive",
        positive,
    ]
    assert mixed.outcome == negative
    (agreeing,) = (
        freeze(replace(spec, targets=targets[:2]))
        .record("attempt-2", {"inside": 0.625, "boundary": 0.75})
        .classify()
    )
    assert agreeing.outcome == positive


def test_tolerance_bounds_compare_the_decimal_values_as_written():
    targets = (
        Target("absolute", 0.3, Tolerance(0.1), (TABLE1,), depends_on=("signal",)),
        Target("relative", 0.3, Tolerance(0.0, 0.5), (TABLE1,), depends_on=("signal",)),
        Target("beyond", 0.3, Tolerance(0.1), (TABLE1,), depends_on=("signal",)),
    )
    spec = Specification(
        "synthetic-paper:tolerances@generation-1",
        "synthetic-exchange/v1",
        "matched_inputs",
        (Choice("signal", "trailing spread", (EQ1,)),),
        (),
        targets,
    )
    observations = {"absolute": 0.4, "relative": 0.45, "beyond": 0.4000000000000001}
    (classification,) = freeze(spec).record("attempt-1", observations).classify()
    assert classification.verdicts == (
        Verdict("absolute", "matched_inputs", "within_tolerance"),
        Verdict("relative", "matched_inputs", "within_tolerance"),
        Verdict("beyond", "matched_inputs", "outside_tolerance"),
    )


@pytest.mark.parametrize(
    "damage",
    [
        "choice-value-without-locator",
        "choice-without-value-or-question",
        "target-expected-without-locator",
        "undeclared-question",
        "unreferenced-question",
        "choice-without-dependent-target",
        "unknown-dependency",
        "duplicate-choice",
        "duplicate-target",
        "duplicate-question",
        "no-targets",
        "negative-tolerance",
        "nan-tolerance",
        "nan-expected",
        "unknown-locator-kind",
        "malformed-pin",
        "unknown-question-kind",
        "unknown-basis",
        "empty-ref",
        "nul-in-text",
        "lone-surrogate",
        "blank-data",
        "blank-question-detail",
        "blank-choice-value",
        "blank-target-name",
    ],
)
def test_freeze_rejects_incomplete_or_inconsistent_specifications(specification, damage):
    spec = specification
    match damage:
        case "choice-value-without-locator":
            spec = replace(spec, choices=replace_first(spec.choices, locators=()))
        case "choice-without-value-or-question":
            spec = replace(spec, choices=replace_first(spec.choices, value=None))
        case "target-expected-without-locator":
            spec = replace(spec, targets=replace_first(spec.targets, locators=()))
        case "undeclared-question":
            spec = replace(spec, choices=replace_first(spec.choices, question="q-unknown"))
        case "unreferenced-question":
            unused = OpenQuestion("q-unused", "ambiguous", "No choice or target names this.")
            spec = replace(spec, questions=(*spec.questions, unused))
        case "choice-without-dependent-target":
            spec = replace(spec, choices=(*spec.choices, Choice("rebalancing", "monthly", (P2,))))
        case "unknown-dependency":
            depends_on = (*SIGNAL, "volatility_filter")
            spec = replace(spec, targets=replace_first(spec.targets, depends_on=depends_on))
        case "duplicate-choice":
            spec = replace(spec, choices=(*spec.choices, spec.choices[0]))
        case "duplicate-target":
            spec = replace(spec, targets=(*spec.targets, spec.targets[0]))
        case "duplicate-question":
            spec = replace(spec, questions=(*spec.questions, spec.questions[0]))
        case "no-targets":
            spec = replace(spec, choices=(), questions=(), targets=())
        case "negative-tolerance":
            tolerance = Tolerance(-0.0005)
            spec = replace(spec, targets=replace_first(spec.targets, tolerance=tolerance))
        case "nan-tolerance":
            tolerance = Tolerance(0.0005, math.nan)
            spec = replace(spec, targets=replace_first(spec.targets, tolerance=tolerance))
        case "nan-expected":
            spec = replace(spec, targets=replace_first(spec.targets, expected=math.nan))
        case "unknown-locator-kind":
            locators = (Locator("theorem", "paper.md#p1"),)
            spec = replace(spec, choices=replace_first(spec.choices, locators=locators))
        case "malformed-pin":
            spec = replace(spec, pin=LocatorPin("0" * 64, "g" * 64))
        case "unknown-question-kind":
            spec = replace(spec, questions=replace_first(spec.questions, kind="speculative"))
        case "unknown-basis":
            spec = replace(spec, basis="partial")
        case "empty-ref":
            locators = (Locator("passage", " "),)
            spec = replace(spec, choices=replace_first(spec.choices, locators=locators))
        case "nul-in-text":
            spec = replace(spec, paper="synthetic-paper\x00trailing-spread")
        case "lone-surrogate":
            spec = replace(spec, paper="synthetic-paper\ud800trailing-spread")
        case "blank-data":
            spec = replace(spec, data=" ")
        case "blank-question-detail":
            spec = replace(spec, questions=replace_first(spec.questions, detail=" "))
        case "blank-choice-value":
            spec = replace(spec, choices=replace_first(spec.choices, value=" "))
        case "blank-target-name":
            spec = replace(spec, targets=replace_first(spec.targets, name=" "))
    with pytest.raises(ReproductionError):
        freeze(spec)
    with pytest.raises(ReproductionError):
        Reproduction(spec).record("attempt-1", {})


def test_an_unpinned_specification_keeps_the_identity_it_had_before_pins(specification):
    golden = "ca73f3b62a0f7407a27be6ff99964213de3c99d940fd8896273b98657d99e537"
    frozen = freeze(specification)
    assert frozen.specification.pin is None
    assert frozen.spec_id == golden
    (classification,) = frozen.record("attempt-1", {"gross_mean": 0.0053}).classify()
    assert (classification.spec_id, classification.verdicts[0]) == (
        golden,
        Verdict("gross_mean", "matched_inputs", "within_tolerance"),
    )


def test_a_pinned_specification_freezes_once_its_located_refs_resolve(tmp_path):
    generation = publish_paper(tmp_path)
    revised = publish_locations(tmp_path, generation.occurrence.id, MARKDOWN_LOCATOR_V2)
    spec = pinned_specification(generation)
    frozen = freeze(spec, store=tmp_path, context=run())
    unpinned = freeze(replace(spec, pin=None))
    assert frozen.specification == spec
    assert frozen.spec_id != unpinned.spec_id
    assert unpinned.revise(spec, store=tmp_path, context=run()).spec_id == frozen.spec_id
    identity = canonical_bytes(asdict(frozen.specification))
    assert hashlib.sha256(identity).hexdigest() == frozen.spec_id
    document = json.loads(identity)
    assert document["pin"] == {"generation": generation.id, "digest": generation.digest}
    rebuilt = replace(spec, pin=LocatorPin(**document["pin"]))
    assert freeze(rebuilt, store=tmp_path, context=run()).spec_id == frozen.spec_id
    recorded = frozen.record("attempt-1", {"outperformance": 0.031})
    assert recorded.classify() == (
        Classification(
            "attempt-1",
            frozen.spec_id,
            (
                Verdict("outperformance", "matched_inputs", "within_tolerance"),
                Verdict("top_decile_mean", "matched_inputs", "open_question", ("q-top-decile",)),
            ),
        ),
    )
    repinned = replace(spec, pin=LocatorPin(revised.id, revised.digest))
    assert hashlib.sha256(canonical_bytes(asdict(repinned))).hexdigest() != frozen.spec_id
    with pytest.raises(ReproductionError):
        replace(recorded, specification=repinned).classify()


def test_a_later_locator_version_resolves_only_its_own_refs(tmp_path):
    generation = publish_paper(tmp_path)
    revised = publish_locations(tmp_path, generation.occurrence.id, MARKDOWN_LOCATOR_V2)
    stale = replace(pinned_specification(generation), pin=LocatorPin(revised.id, revised.digest))
    with pytest.raises(ReproductionError):
        freeze(stale, store=tmp_path, context=run())
    current = pinned_specification(revised)
    assert freeze(current, store=tmp_path, context=run()).specification == current


@pytest.mark.parametrize(
    "refusal",
    [
        "unknown-ref",
        "wrong-kind",
        "other-version-ref",
        "unknown-equation-in-a-later-choice",
        "wrong-kind-equation-in-the-last-target",
        "other-version-equation-after-good-refs",
        "unpublished-generation",
        "stale-digest",
        "unreadable-by-label",
        "unreadable-by-cutoff",
        "missing-context",
    ],
)
def test_freezing_a_pinned_specification_refuses_what_it_cannot_resolve(tmp_path, refusal):
    labels = frozenset({VENDOR}) if refusal == "unreadable-by-label" else frozenset()
    generation = publish_paper(tmp_path, replace(DECLARATION, labels=labels))
    spec = pinned_specification(generation)
    context = run()
    match refusal:
        case "unknown-ref":
            stray = Locator("claim", f"{generation.id}:claim:0-1")
            spec = replace(spec, choices=replace_first(spec.choices, locators=(stray,)))
        case "wrong-kind":
            mislabeled = Locator("table", cite(generation, "equation").ref)
            spec = replace(spec, questions=replace_first(spec.questions, locators=(mislabeled,)))
        case "other-version-ref":
            revised = publish_locations(tmp_path, generation.occurrence.id, MARKDOWN_LOCATOR_V2)
            later = (cite(revised, "figure"),)
            spec = replace(spec, targets=replace_first(spec.targets, locators=later))
        case "unknown-equation-in-a-later-choice":
            stray = Locator("equation", f"{generation.id}:equation:0-1")
            first, second = spec.choices
            spec = replace(spec, choices=(first, replace(second, locators=(stray, P1))))
        case "wrong-kind-equation-in-the-last-target":
            mislabeled = Locator("equation", cite(generation, "table").ref)
            *earlier, last = spec.targets
            located = replace(last, locators=(P5, mislabeled))
            spec = replace(spec, targets=(*earlier, located))
        case "other-version-equation-after-good-refs":
            revised = publish_locations(tmp_path, generation.occurrence.id, MARKDOWN_LOCATOR_V2)
            (question,) = spec.questions
            later = (*question.locators, cite(revised, "equation"))
            spec = replace(spec, questions=(replace(question, locators=later),))
        case "unpublished-generation":
            unpublished = replace(generation, locator=MARKDOWN_LOCATOR_V2.locator)
            spec = replace(spec, pin=LocatorPin(unpublished.id, generation.digest))
        case "stale-digest":
            spec = replace(spec, pin=LocatorPin(generation.id, "0" * 64))
        case "unreadable-by-cutoff":
            context = run().narrow(cutoff=TIME - timedelta(microseconds=1))
        case "missing-context":
            context = None
    with pytest.raises(ReproductionError) as raised:
        freeze(spec, store=tmp_path, context=context)
    with pytest.raises(ReproductionError):
        freeze(replace(spec, pin=None)).revise(spec, store=tmp_path, context=context)
    if refusal == "unreadable-by-label":
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ReproductionError) as unpublished:
            freeze(spec, store=empty, context=run(VENDOR))
        assert str(raised.value) == str(unpublished.value)
        assert freeze(spec, store=tmp_path, context=run(VENDOR)).specification == spec


def test_a_reproduction_built_or_copied_outside_freeze_still_checks_its_refs(tmp_path):
    generation = publish_paper(tmp_path)
    revised = publish_locations(tmp_path, generation.occurrence.id, MARKDOWN_LOCATOR_V2)
    spec = pinned_specification(generation)
    frozen = freeze(spec, store=tmp_path, context=run())
    stray = Locator("equation", f"{generation.id}:equation:0-1")
    unresolved = replace(spec, choices=replace_first(spec.choices, locators=(stray,)))
    unpinned = replace(spec, pin=None)
    for bypass in (
        lambda: Reproduction(spec),
        lambda: Reproduction(unresolved, generation=generation),
        lambda: replace(frozen, specification=unresolved),
        lambda: Reproduction(spec, generation=revised),
        lambda: Reproduction(unpinned, generation=generation),
    ):
        with pytest.raises(ReproductionError):
            bypass().record("attempt-1", {"outperformance": 0.031})
    assert Reproduction(spec, generation=generation) == frozen
    recorded = Reproduction(unpinned).record("attempt-1", {"outperformance": 0.031})
    assert recorded.classify()[0].spec_id == freeze(unpinned).spec_id


def test_version_1_generations_and_the_specifications_pinned_to_them_keep_their_bytes(tmp_path):
    generation = publish_paper(tmp_path)
    publish_locations(tmp_path, generation.occurrence.id, MARKDOWN_LOCATOR_V2)
    frozen = freeze(pinned_specification(generation), store=tmp_path, context=run())
    assert (generation.id, generation.digest, frozen.spec_id) == (
        "8268aac811eb0481db2e9ecb9e7b3a5cc8c9f10d12454f8023dd347844c6cc8d",
        "b4f5ff6d0e90b50d9b413a856103f6456825ddc2fb1d3b1c85c76f809bd315e6",
        "a52c5a886f83224b736d53843aa1c630fd3458c41cbd27a3bdb645be9a623bcf",
    )


@pytest.mark.parametrize(
    ("run", "observations"),
    [
        pytest.param(
            "attempt-1", {"gross_mean": 0.0053, "sharpe_ratio": 0.8}, id="undeclared-target"
        ),
        pytest.param(" ", {"gross_mean": 0.0053}, id="blank-run"),
    ],
)
def test_record_rejects_undeclared_targets_and_blank_runs(specification, run, observations):
    with pytest.raises(ReproductionError):
        freeze(specification).record(run, observations)


def test_record_rejects_a_run_label_already_recorded(specification):
    reproduction = freeze(specification).record("attempt-1", {"gross_mean": 0.0053})
    with pytest.raises(ReproductionError):
        reproduction.record("attempt-1", {"gross_mean": 0.0060})
    assert len(reproduction.record("attempt-2", {"gross_mean": 0.0060}).results) == 2


def test_fixture_locators_name_labels_in_the_synthetic_paper(specification):
    paper = PAPER.read_text(encoding="utf-8")
    labels = {
        locator.ref.removeprefix("paper.md#")
        for part in (*specification.choices, *specification.questions, *specification.targets)
        for locator in part.locators
        if locator.ref.startswith("paper.md#")
    }
    assert labels
    assert {label for label in labels if f"[{label}]" not in paper} == set()
