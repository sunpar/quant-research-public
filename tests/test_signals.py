from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, time, timedelta
from fractions import Fraction
from functools import reduce
from pathlib import Path

import pytest

from quant_research import datasets
from quant_research.access import issue_context
from quant_research.datasets import Refused
from quant_research.public_data import (
    SEARCH_UNIVERSE,
    approve,
    ingest,
    load_catalog,
    pull,
    validate,
)
from quant_research.signals import (
    SAHM,
    SAHM_RULE,
    And,
    Lag,
    Not,
    Or,
    ProgramError,
    Ratio,
    Rejected,
    Series,
    Subtract,
    Threshold,
    TrailingMean,
    TrailingMin,
    canonical,
    decide,
    digest,
    observe,
    parse,
    shift,
)

# Each ALFRED response is an excerpt of a real download, with count set to the excerpt's length.
UNRATE = (Path(__file__).parent / "fixtures" / "signals" / "alfred-unrate-2024.json").read_bytes()
NOW = datetime(2026, 10, 22, 12, tzinfo=UTC)
CONTEXT = issue_context(clearances=(), cutoff=NOW, generation="signals/v1", providers=())
X = Series("x")
RATE = Series("alfred.unrate")
RATE_JSON = {"op": "Series", "dataset": "alfred.unrate"}
SAHM_RULE_JSON = (
    b'{"level":"1/2","op":"Threshold","source":{"left":{"months":3,"op":"TrailingMean",'
    b'"source":{"dataset":"alfred.unrate","op":"Series"}},"op":"Subtract","right":{"months":12,'
    b'"op":"TrailingMin","source":{"months":3,"op":"TrailingMean","source":{"dataset":'
    b'"alfred.unrate","op":"Series"}}}}}\n'
)


def month(text: str) -> date:
    return date.fromisoformat(f"{text}-01")


def history(*values: str | None, start: str = "2024-01") -> dict:
    return {"x": {shift(month(start), index): {"value": v} for index, v in enumerate(values)}}


def approved_root(root: Path, responses: dict[str, bytes], now: datetime = NOW) -> Path:
    for name, data in responses.items():
        pull(root, name, now=now, api_key="k", fetch=lambda url, data=data: data)
        ingest(root, name, now=now)
        validate(root, name, actor="researcher", evidence=b"notes", summary="checked", now=now)
        approve(root, name, actor="researcher", now=now)
    return root


def decisions(root: Path, months: list[date]) -> tuple:
    return decide(SAHM, load_catalog(root), root / "datasets", CONTEXT, months)


@pytest.fixture(scope="module")
def catalog(tmp_path_factory):
    return load_catalog(
        approved_root(tmp_path_factory.mktemp("catalog"), {"alfred.unrate": UNRATE})
    )


def node(op: str, **fields) -> dict:
    return {"op": op, **fields}


def as_json(value: object) -> object:
    if isinstance(value, Fraction):
        return str(value)
    return value if isinstance(value, int | float | str) else json.loads(canonical(value))


def parsed(raw: object, catalog, universe=SEARCH_UNIVERSE):
    return parse(json.dumps(raw).encode(), catalog, universe)


def assert_both_paths(catalog, reason, cls, **arguments):
    """Assert the constructor and parse both accept the program, or both report reason."""
    raw = node(cls.__name__, **{name: as_json(value) for name, value in arguments.items()})
    if reason is None:
        assert parsed(raw, catalog) == cls(**arguments)
        return
    with pytest.raises(ProgramError) as raised:
        cls(**arguments)
    assert (raised.value.reason, parsed(raw, catalog)) == (reason, Rejected(reason))


def nested(depth: int) -> bytes:
    """Return a program of depth operators, Nots around a threshold on a series."""
    inner = json.dumps(node("Threshold", source=RATE_JSON, level="4"))
    return ('{"op":"Not","condition":' * (depth - 2) + inner + "}" * (depth - 2)).encode()


def test_operators_match_hand_computed_values():
    values = history("4.2", "5.0", "6.1", "3.3", "1.4")
    may = month("2024-05")
    mean, prior_low = TrailingMean(X, 3), TrailingMin(X, 3)
    assert X.evaluate(values, may) == Fraction(7, 5)
    assert mean.evaluate(values, may) == Fraction(18, 5)
    # May's 1.4 is the lowest value, and the prior minimum leaves it out.
    assert prior_low.evaluate(values, may) == Fraction(33, 10)
    assert Subtract(mean, prior_low).evaluate(values, may) == Fraction(3, 10)
    assert Threshold(Subtract(mean, prior_low), Fraction(3, 10)).evaluate(values, may) is True
    assert Threshold(Subtract(mean, prior_low), Fraction(301, 1000)).evaluate(values, may) is False


@pytest.mark.parametrize(
    "program, reach",
    [
        (SAHM, {"alfred.unrate": 14}),
        (Threshold(TrailingMin(TrailingMean(X, 3), 4), Fraction(0)), {"x": 6}),
        (Subtract(Series("a"), TrailingMean(Series("b"), 3)), {"a": 0, "b": 2}),
        (Subtract(TrailingMean(Series("a"), 3), TrailingMin(Series("a"), 4)), {"a": 4}),
        (Lag(TrailingMean(X, 3), 2), {"x": 4}),
        (Subtract(Series("a"), Series("a", vintage=1)), {"a": 0, "a@1": 0}),
    ],
)
def test_reach_is_the_months_each_dataset_is_read_back(program, reach):
    assert program.reach == reach


def test_null_months_are_skipped_in_windows_and_not_computable_themselves():
    mean = TrailingMean(X, 3)
    assert mean.evaluate(history("4.4", None, "4.6"), month("2024-03")) == Fraction(9, 2)
    assert mean.evaluate(history("4.4", "4.6", None), month("2024-03")) is None
    values = history("4.2", "4.4", "4.6", None)
    assert observe(mean, values) == (month("2024-03"), Fraction(22, 5))
    assert observe(TrailingMin(X, 2), history(None, None, "4.6")) is None


def test_a_window_reaching_before_the_first_month_is_not_computable():
    mean = TrailingMean(X, 3)
    assert observe(mean, history("4.2", "4.4", "4.6")) == (month("2024-03"), Fraction(22, 5))
    assert observe(mean, history("4.4", "4.6", start="2024-02")) is None


@pytest.mark.smoke
def test_decisions_read_the_vintage_before_each_month_and_reproduce_fred(tmp_path):
    root = approved_root(tmp_path, {"alfred.unrate": UNRATE})
    months = [month(text) for text in ("2025-10", "2025-11", "2025-12", "2026-01", "2026-02")]
    result = decisions(root, months)
    # ALFRED releases on 2025-09-05, 2025-11-20, 2025-12-16, and 2026-01-09 are available at the
    # next midnight UTC (D052). The 2025 shutdown released nothing in October.
    released = [datetime(*day, tzinfo=UTC) for day in ((2025, 9, 6), (2025, 11, 21))]
    released += [datetime(*day, tzinfo=UTC) for day in ((2025, 12, 17), (2026, 1, 10))]
    assert [(item.month, item.observation, item.released_at) for item in result] == [
        (months[0], month("2025-08"), released[0]),
        (months[1], month("2025-08"), released[0]),
        (months[2], month("2025-09"), released[1]),
        (months[3], month("2025-11"), released[2]),
        (months[4], month("2025-12"), released[3]),
    ]
    assert [result[3].value, result[4].value] == [Fraction(13, 30), Fraction(7, 20)]
    fred = [Fraction(text) for text in ("0.13", "0.13", "0.23", "0.43", "0.35")]
    differences = [item.value - value for item, value in zip(result, fred, strict=True)]
    assert all(abs(difference) <= Fraction(1, 200) for difference in differences)
    catalog = load_catalog(root)
    for item in result:
        narrowed = CONTEXT.narrow(cutoff=datetime.combine(item.month, time(), UTC))
        read = datasets.read(catalog, root / "datasets", "alfred.unrate", narrowed)
        assert item.reads == (read.record,)


def later_vintage(data: bytes) -> bytes:
    response = json.loads(data)
    observations = response["observations"]
    (latest,) = [item for item in observations if item["date"] == "2026-01-01"]
    latest["realtime_end"] = "2026-03-05"
    observations += [
        {"realtime_start": "2026-03-06", "realtime_end": "9999-12-31", "date": day, "value": "4.4"}
        for day in ("2026-01-01", "2026-02-01")
    ]
    response["count"] = len(observations)
    return json.dumps(response).encode()


def test_a_later_vintage_leaves_every_earlier_decision_unchanged(tmp_path):
    root = approved_root(tmp_path, {"alfred.unrate": UNRATE})
    months = [shift(month("2025-09"), offset) for offset in range(8)]
    before = decisions(root, months)
    approved_root(root, {"alfred.unrate": later_vintage(UNRATE)}, now=NOW + timedelta(days=1))
    after = decisions(root, months)

    def seen(item):
        supplied = [read.supplied for read in item.reads]
        return item.month, item.observation, item.value, item.released_at, supplied

    assert after[0].reads[0].vintage != before[0].reads[0].vintage
    assert [seen(item) for item in after[:-1]] == [seen(item) for item in before[:-1]]
    assert (before[-1].observation, after[-1].observation) == (month("2026-01"), month("2026-02"))


def test_an_unapproved_dataset_is_refused(tmp_path):
    pull(tmp_path, "alfred.unrate", now=NOW, api_key="k", fetch=lambda url: UNRATE)
    ingest(tmp_path, "alfred.unrate", now=NOW)
    assert decisions(tmp_path, [month("2026-01")]) == Refused("documented")


def test_lag_and_ratio_match_hand_computed_values():
    values = history("4.2", "5.0", "6.1", "3.3", "1.4")
    may = month("2024-05")
    assert Lag(X, 0).evaluate(values, may) == Fraction(7, 5)
    assert Lag(X, 2).evaluate(values, may) == Fraction(61, 10)
    assert Ratio(X, Lag(X, 1)).evaluate(values, may) == Fraction(14, 33)
    february, march = month("2024-02"), month("2024-03")
    zero = history("0", "2.5")
    assert Ratio(Lag(X, 1), X).evaluate(zero, february) == Fraction(0)
    assert Ratio(X, Lag(X, 1)).evaluate(zero, february) is None
    gap = history("4.2", None, "6.1")
    assert Lag(X, 1).evaluate(gap, march) is None
    assert Ratio(X, Lag(X, 1)).evaluate(gap, march) is None
    assert Ratio(Lag(X, 1), X).evaluate(gap, march) is None


def test_not_and_or_match_hand_computed_values_and_any_none_operand_gives_none():
    values = history("4.2", None)
    january, february = month("2024-01"), month("2024-02")
    high, low = Threshold(X, Fraction(4)), Threshold(X, Fraction(5))
    conditions = [Not(high), Not(low), And(high, low), And(high, Not(low)), Or(low, high)]
    assert [item.evaluate(values, january) for item in conditions] == [
        False,
        True,
        False,
        True,
        True,
    ]
    assert Or(low, low).evaluate(values, january) is False
    # In February the earlier month's 4.2 is known and the month itself is null.
    known, missing = Threshold(Lag(X, 1), Fraction(4)), Threshold(X, Fraction(4))
    assert known.evaluate(values, february) is True
    assert Not(missing).evaluate(values, february) is None
    assert And(Not(known), missing).evaluate(values, february) is None
    assert Or(known, missing).evaluate(values, february) is None


def test_each_leak_reports_exactly_one_reason_on_both_paths(catalog):
    assert_both_paths(catalog, "reads_past_vintage", Series, dataset="alfred.unrate", vintage=-1)
    assert_both_paths(catalog, None, Series, dataset="alfred.unrate", vintage=0)
    assert_both_paths(catalog, "negative_lag", Lag, source=RATE, months=-1)
    assert_both_paths(catalog, None, Lag, source=RATE, months=0)
    assert_both_paths(catalog, "unapproved_dataset", Series, dataset="ALFRED.PAYEMS")


def test_a_dataset_the_catalog_or_universe_does_not_approve_is_rejected(tmp_path):
    root = approved_root(tmp_path, {"alfred.unrate": UNRATE, "alfred.sahmrealtime": UNRATE})
    pull(root, "alfred.payems", now=NOW, api_key="k", fetch=lambda url: UNRATE)
    ingest(root, "alfred.payems", now=NOW)
    payems = node("Series", dataset="alfred.payems")
    assert parsed(payems, load_catalog(root)) == Rejected("unapproved_dataset")
    validate(root, "alfred.payems", actor="researcher", evidence=b"notes", summary="ok", now=NOW)
    approve(root, "alfred.payems", actor="researcher", now=NOW)
    assert parsed(payems, load_catalog(root)) == Series("alfred.payems")
    outside = node("Series", dataset="alfred.sahmrealtime")
    assert parsed(outside, load_catalog(root)) == Rejected("unapproved_dataset")
    widened = (*SEARCH_UNIVERSE, "alfred.sahmrealtime")
    assert parsed(outside, load_catalog(root), widened) == Series("alfred.sahmrealtime")


@pytest.mark.parametrize(
    ("cls", "arguments", "reason"),
    [
        (Series, {"vintage": 0}, None),
        (Series, {"vintage": 12}, None),
        (Series, {"vintage": 13}, "out_of_range"),
        (Lag, {"months": 0}, None),
        (Lag, {"months": 24}, None),
        (Lag, {"months": 25}, "out_of_range"),
        (TrailingMean, {"months": 1}, None),
        (TrailingMean, {"months": 24}, None),
        (TrailingMean, {"months": 0}, "out_of_range"),
        (TrailingMean, {"months": 25}, "out_of_range"),
        (TrailingMin, {"months": 1}, None),
        (TrailingMin, {"months": 24}, None),
        (TrailingMin, {"months": 0}, "out_of_range"),
        (TrailingMin, {"months": 25}, "out_of_range"),
        (Threshold, {"level": Fraction(-1_000_000)}, None),
        (Threshold, {"level": Fraction(1_000_000)}, None),
        (Threshold, {"level": Fraction(-10_000_000_001, 10_000)}, "out_of_range"),
        (Threshold, {"level": Fraction(10_000_000_001, 10_000)}, "out_of_range"),
        (Threshold, {"level": Fraction(1, 10_000)}, None),
        (Threshold, {"level": Fraction(1, 20_000)}, "out_of_range"),
    ],
)
def test_each_bound_admits_its_floor_and_ceiling_and_reports_a_miss(
    catalog, cls, arguments, reason
):
    base = {"dataset": "alfred.unrate"} if cls is Series else {"source": RATE}
    assert_both_paths(catalog, reason, cls, **base, **arguments)


@pytest.mark.parametrize(
    ("cls", "arguments"),
    [
        (TrailingMean, {"source": RATE, "months": True}),
        (Series, {"dataset": "alfred.unrate", "vintage": True}),
        (Series, {"dataset": "alfred.unrate", "field": 1}),
        (Lag, {"source": Threshold(RATE, Fraction(4)), "months": 1}),
        (Subtract, {"left": RATE, "right": Threshold(RATE, Fraction(4))}),
        (Not, {"condition": RATE}),
        (And, {"left": Threshold(RATE, Fraction(4)), "right": RATE}),
        (Threshold, {"source": RATE, "level": 4}),
        (Threshold, {"source": RATE, "level": 0.5}),
    ],
)
def test_a_value_of_another_type_is_wrong_type(catalog, cls, arguments):
    assert_both_paths(catalog, "wrong_type", cls, **arguments)


def test_sahm_costs_54_evaluations_per_month():
    # Each mean takes 1 + 3 series values = 4, the minimum 1 + 12 means = 49, and the
    # difference 1 + 4 + 49.
    assert SAHM.cost == 54


def test_a_program_over_its_cost_or_read_limit_is_too_large(catalog):
    # 1 + 15 * (1 + 16) = 256 evaluations, then 1 + 16 * (1 + 15) = 257.
    assert_both_paths(catalog, None, TrailingMean, source=TrailingMean(RATE, 16), months=15)
    assert_both_paths(catalog, "too_large", TrailingMean, source=TrailingMean(RATE, 15), months=16)
    reads = [Series("alfred.unrate", vintage=vintage) for vintage in range(5)]
    assert_both_paths(catalog, None, Subtract, left=reduce(Subtract, reads[:3]), right=reads[3])
    assert_both_paths(
        catalog, "too_large", Subtract, left=reduce(Subtract, reads[:4]), right=reads[4]
    )


def test_nesting_past_32_operators_is_too_large_on_both_paths(catalog):
    deepest = parse(nested(32), catalog, SEARCH_UNIVERSE)
    assert parse(canonical(deepest), catalog, SEARCH_UNIVERSE) == deepest
    with pytest.raises(ProgramError) as raised:
        Not(deepest)
    assert raised.value.reason == "too_large"
    assert parse(nested(33), catalog, SEARCH_UNIVERSE) == Rejected("too_large")
    assert parse(nested(1000), catalog, SEARCH_UNIVERSE) == Rejected("too_large")
    assert parse(nested(100_000), catalog, SEARCH_UNIVERSE) == Rejected("too_large")


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ([RATE_JSON], "wrong_type"),
        (node("Lag", source="alfred.unrate", months=1), "wrong_type"),
        (node("Threshold", source=RATE_JSON, level="half"), "wrong_type"),
        (node("Threshold", source=RATE_JSON, level="1/0"), "wrong_type"),
        (node("Threshold", source=RATE_JSON, level="9" * 5000), "wrong_type"),
        (node("Maximum", source=RATE_JSON), "unknown_operator"),
        (node(3, source=RATE_JSON), "unknown_operator"),
        ({"dataset": "alfred.unrate"}, "missing_field"),
        (node("Lag", source=RATE_JSON), "missing_field"),
        (node("Series", dataset="alfred.unrate", window=3), "extra_field"),
        (node("Series", dataset="alfred.unrate", field="close"), "unknown_field"),
    ],
)
def test_a_malformed_program_is_rejected_with_its_reason(catalog, raw, reason):
    assert parsed(raw, catalog) == Rejected(reason)


@pytest.mark.parametrize(
    "data",
    [
        b'{"op":"Series","dataset":"alfred.unrate","dataset":"alfred.payems"}',
        b'{"op":"Lag","source":{"op":"Series","dataset":"alfred.unrate"},"months":NaN}',
        b'{"op":"Series","dataset":"alfred.unrate"',
        b"\xff",
    ],
)
def test_text_that_is_not_strict_json_is_malformed(catalog, data):
    assert parse(data, catalog, SEARCH_UNIVERSE) == Rejected("malformed_json")


def test_the_sahm_rule_has_a_pinned_canonical_form_and_every_program_round_trips(catalog):
    assert canonical(SAHM_RULE) == SAHM_RULE_JSON
    assert digest(SAHM_RULE) == hashlib.sha256(SAHM_RULE_JSON).hexdigest()
    every = Or(
        And(
            Threshold(
                Ratio(Lag(RATE, 1), TrailingMean(Series("alfred.unrate", vintage=1), 3)),
                Fraction(-3, 4),
            ),
            Not(Threshold(TrailingMin(Series("alfred.unrate", field="value"), 2), Fraction(5))),
        ),
        SAHM_RULE,
    )
    for program in (SAHM, SAHM_RULE, every):
        assert parse(canonical(program), catalog, SEARCH_UNIVERSE) == program


def test_decimal_and_fraction_levels_give_one_program_and_one_digest(catalog):
    source = json.loads(canonical(SAHM))
    programs = [
        parsed(node("Threshold", source=source, level=text), catalog)
        for text in ("0.5", "1/2", "2/4", "0.50")
    ]
    assert programs == [SAHM_RULE] * 4
    assert {digest(program) for program in programs} == {digest(SAHM_RULE)}


def test_a_revision_program_reads_this_and_last_months_vintages(tmp_path):
    root = approved_root(tmp_path, {"alfred.unrate": UNRATE})
    catalog = load_catalog(root)
    revision = Subtract(RATE, Series("alfred.unrate", vintage=1))
    months = [month("2026-01"), month("2026-02")]
    result = decide(revision, catalog, root / "datasets", CONTEXT, months)
    # The vintage read at 2025-12-01 ends at September (4.4), before the 2025-12-16 release.
    # November read 4.6 at 2026-01-01 and 4.5 after the 2026-01-09 revision.
    assert [(item.observation, item.value) for item in result] == [
        (month("2025-09"), Fraction(0)),
        (month("2025-11"), Fraction(-1, 10)),
    ]
    for item in result:
        cutoffs = (item.month, shift(item.month, -1))
        direct = tuple(
            datasets.read(
                catalog,
                root / "datasets",
                "alfred.unrate",
                CONTEXT.narrow(cutoff=datetime.combine(cutoff, time(), UTC)),
            ).record
            for cutoff in cutoffs
        )
        assert item.reads == direct
