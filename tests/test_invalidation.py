from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from quant_research.access import Restrictions, issue_context
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
    record_id,
)
from quant_research.invalidation import (
    Effect,
    Impact,
    Invalidation,
    InvalidationError,
    assess,
    catch_up,
    suspend,
)
from quant_research.records import canonical_bytes

T0 = datetime(2026, 10, 21, 12, tzinfo=UTC)
LATER = T0 + timedelta(days=1)
SCOPE = Scope(
    "US large-cap equities",
    "one month",
    datetime(2010, 1, 1, tzinfo=UTC),
    datetime(2024, 12, 31, tzinfo=UTC),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


PACKAGE = sha256(b"momentum backtest package")
CLEAN = sha256(b"international momentum backtest package")
SOURCE = sha256(b"momentum survey source generation")


def supports(*items):
    return tuple(
        External(item, "supports") if isinstance(item, str) else Cites(record_id(item), "supports")
        for item in items
    )


def finding(statement, *support, outcome="supports_idea"):
    return Finding(statement, "walk-forward/v1", outcome, "researcher", SCOPE, supports(*support))


def claim(statement, *support):
    return Claim(statement, "extracted", "agent", SCOPE, supports(*support))


def lesson(statement, *support):
    return Lesson(statement, "agent", SCOPE, supports(*support))


def act(record, action, at=T0):
    return Review(record_id(record), action, "reviewer", at, "Checked the cited evidence")


def accepted(*records):
    events = []
    for record in records:
        events.extend((Proposal(record, T0), act(record, "review")))
        if isinstance(record, Lesson):
            events.append(act(record, "admit"))
    return Ledger(tuple(events))


def invalidation(evidence_id, kind="invalid_package", at=LATER, reason="Look-ahead prices"):
    return Invalidation(evidence_id, kind, "auditor", at, reason)


RESULT = finding("Twelve-month momentum earns a premium after costs", PACKAGE)
RULE = lesson("Momentum is a durable cross-sectional effect", RESULT)
REGIONAL = finding("Twelve-month momentum earns a premium in international equities", CLEAN)
HEDGED = lesson("Momentum survives in at least one region", RESULT, REGIONAL)


@pytest.mark.smoke
def test_invalid_package_suspends_every_record_resting_only_on_it():
    pending = claim("Momentum also appears in sector portfolios", PACKAGE)
    retracted = claim("Momentum also appears in commodities", PACKAGE)
    ledger = Ledger(
        (
            *accepted(RESULT, RULE).events,
            Proposal(pending, T0),
            Proposal(retracted, T0),
            act(retracted, "retract"),
        )
    )
    impact = assess(ledger, {}, invalidation(PACKAGE, reason="Look-ahead prices in the signal"))
    assert impact.withdrawn == (PACKAGE,)
    assert impact.effects == (
        Effect(record_id(RESULT), "finding", "supports_idea", (PACKAGE,), False),
        Effect(record_id(RULE), "lesson", None, (record_id(RESULT),), False),
        Effect(record_id(pending), "claim", None, (PACKAGE,), False),
        Effect(record_id(retracted), "claim", None, (PACKAGE,), False),
    )
    suspended = suspend(ledger, impact)
    assert ledger.eligible_lessons() == (record_id(RULE),)
    assert suspended.eligible_lessons() == ()
    assert suspended.events[: len(ledger.events)] == ledger.events
    assert suspended.status(record_id(retracted)) == "retracted"
    assert suspended.history(record_id(retracted)) == ledger.history(record_id(retracted))
    for record in (RESULT, RULE, pending):
        identity = record_id(record)
        assert suspended.status(identity) == "suspended"
        assert suspended.record(identity) == record
        last = suspended.history(identity)[-1]
        assert (last.action, last.reviewer, last.at) == ("suspend", "auditor", LATER)
        for part in ("invalid_package", PACKAGE, "Look-ahead prices in the signal"):
            assert part in last.reason


def test_a_suspension_reaches_admissible_lessons_only_through_a_later_memory_pin():
    declared = Ledger(
        (
            *accepted(RESULT, RULE).events,
            Declaration(
                PACKAGE,
                Restrictions(available_at=T0, labels=frozenset()),
                "ingestion",
                T0,
                "Package inputs are public",
            ),
        )
    )
    impact = assess(declared, {}, invalidation(PACKAGE))
    assert impact == assess(accepted(RESULT, RULE), {}, invalidation(PACKAGE))
    suspended = suspend(declared, impact)
    assert suspended.status(record_id(RULE)) == "suspended"
    for pinned, expected in ((declared, (record_id(RULE),)), (suspended, ())):
        context = issue_context(
            clearances=(), cutoff=T0, generation="0" * 64, providers=(), memory=pinned.head
        )
        assert suspended.admissible_lessons(context) == expected


def test_lesson_with_independent_support_stays_eligible_and_is_reported():
    ledger = accepted(RESULT, REGIONAL, HEDGED)
    impact = assess(ledger, {}, invalidation(PACKAGE))
    assert impact.effects == (
        Effect(record_id(RESULT), "finding", "supports_idea", (PACKAGE,), False),
        Effect(record_id(HEDGED), "lesson", None, (record_id(RESULT),), True),
    )
    suspended = suspend(ledger, impact)
    assert suspended.eligible_lessons() == (record_id(HEDGED),)
    assert suspended.status(record_id(HEDGED)) == "admitted"
    assert suspended.history(record_id(HEDGED)) == ledger.history(record_id(HEDGED))


@pytest.mark.smoke
@pytest.mark.parametrize("kind", ["corrected_vintage", "revoked_vintage"])
def test_vintage_invalidation_reaches_a_lesson_through_the_package_that_read_it(kind):
    vintage = sha256(b"prices vintage 2026-09\n")
    october = sha256(b"prices vintage 2026-10\n")
    reader = sha256(b"September and October momentum package")
    unrelated = sha256(b"October momentum package")
    package_inputs = {reader: {vintage, october}, unrelated: {october}}
    result = finding("Momentum earns a premium on September prices", reader)
    rule = lesson("Momentum holds on vendor prices", result)
    unaffected = finding("Momentum earns a premium on October prices", unrelated)
    ledger = accepted(result, unaffected, rule)
    impact = assess(ledger, package_inputs, invalidation(vintage, kind))
    assert impact.withdrawn == (vintage, reader)
    assert impact.effects == (
        Effect(record_id(result), "finding", "supports_idea", (reader,), False),
        Effect(record_id(rule), "lesson", None, (record_id(result),), False),
    )
    assert suspend(ledger, impact).eligible_lessons() == ()
    literal = assess(ledger, package_inputs, invalidation(vintage))
    assert (literal.withdrawn, literal.effects) == ((vintage,), ())


def test_rereview_after_suspension_restores_eligibility():
    ledger = accepted(RESULT, RULE)
    suspended = suspend(ledger, assess(ledger, {}, invalidation(PACKAGE)))
    after = LATER + timedelta(hours=1)
    restored = Ledger(
        (
            *suspended.events,
            act(RESULT, "review", after),
            act(RULE, "review", after),
            act(RULE, "admit", after),
        )
    )
    assert restored.eligible_lessons() == (record_id(RULE),)
    history = restored.history(record_id(RULE))
    assert history[0] == Proposal(RULE, T0)
    assert [event.action for event in history[1:]] == [
        "review",
        "admit",
        "suspend",
        "review",
        "admit",
    ]


def test_invalid_implementation_and_valid_negative_result_read_differently():
    broken = finding("Short-term reversal loses money after costs", PACKAGE, outcome="rejects_idea")
    negative = finding("Low volatility loses money after costs", CLEAN, outcome="rejects_idea")
    avoid_reversal = lesson("Do not pursue short-term reversal", broken)
    avoid_low_volatility = lesson("Do not pursue low volatility", negative)
    ledger = accepted(broken, negative, avoid_reversal, avoid_low_volatility)
    impact = assess(ledger, {}, invalidation(PACKAGE))
    assert impact.invalidation.kind == "invalid_package"
    assert impact.effects == (
        Effect(record_id(broken), "finding", "rejects_idea", (PACKAGE,), False),
        Effect(record_id(avoid_reversal), "lesson", None, (record_id(broken),), False),
    )
    suspended = suspend(ledger, impact)
    assert suspended.status(record_id(negative)) == "reviewed"
    assert suspended.eligible_lessons() == (record_id(avoid_low_volatility),)
    assert suspended.status(record_id(broken)) == "suspended"
    assert "invalid_package" in suspended.history(record_id(broken))[-1].reason


def hedged_impact():
    return assess(accepted(RESULT, REGIONAL, HEDGED), {}, invalidation(PACKAGE))


def test_impact_bytes_round_trip_canonically_and_reject_other_encodings():
    impact = hedged_impact()
    data = impact.to_bytes()
    assert Impact.from_bytes(data) == impact
    assert Impact.from_bytes(data).to_bytes() == data
    with pytest.raises(InvalidationError):
        Impact.from_bytes(json.dumps(json.loads(data), indent=2, sort_keys=True).encode())


@pytest.mark.parametrize(
    "damage",
    [
        lambda value: value.update(schema_version=2),
        lambda value: value.update(note="extra"),
        lambda value: value["withdrawn"].append("artifact:momentum"),
        lambda value: value["invalidation"].update(at="2026-10-22T12:00:00"),
        lambda value: value["invalidation"].update(kind="stale_vintage"),
        lambda value: value["effects"][0].update(kind="note"),
        lambda value: value["effects"][0].update(outcome=None),
        lambda value: value["effects"][1].update(outcome="rejects_idea"),
        lambda value: value["effects"][1].update(supported="true"),
        lambda value: value["effects"][1]["via"].append(""),
    ],
    ids=[
        "schema-version",
        "extra-field",
        "withdrawn-not-digest",
        "naive-time",
        "unknown-kind",
        "unknown-record-kind",
        "finding-without-outcome",
        "lesson-with-outcome",
        "support-not-boolean",
        "via-not-digest",
    ],
)
def test_impact_bytes_reject_malformed_fields(damage):
    value = json.loads(hedged_impact().to_bytes())
    damage(value)
    with pytest.raises(InvalidationError):
        Impact.from_bytes(canonical_bytes(value))


def test_suspending_twice_appends_nothing_until_a_new_invalidation_arrives():
    ledger = accepted(RESULT, RULE)
    once = suspend(ledger, assess(ledger, {}, invalidation(PACKAGE)))
    assert suspend(once, assess(ledger, {}, invalidation(PACKAGE))).events == once.events
    second = invalidation(PACKAGE, at=LATER + timedelta(hours=1), reason="Survivorship bias")
    twice = suspend(once, assess(once, {}, second))
    appended = twice.events[len(once.events) :]
    assert [event.record_id for event in appended] == [record_id(RESULT), record_id(RULE)]
    assert all("Survivorship bias" in event.reason for event in appended)
    assert twice.status(record_id(RULE)) == "suspended"


def test_suspension_keeps_a_retraction_beneath_it_in_force():
    retracted = claim("Momentum also appears in sector portfolios", SOURCE)
    sectoral = finding("Momentum earns a premium in sector portfolios", PACKAGE, retracted)
    rule = lesson("Momentum holds at home and across sectors", sectoral, REGIONAL)
    ledger = Ledger(
        (*accepted(retracted, sectoral, REGIONAL, rule).events, act(retracted, "retract"))
    )
    assert ledger.eligible_lessons() == ()
    impact = assess(ledger, {}, invalidation(PACKAGE))
    assert impact.effects == (
        Effect(record_id(sectoral), "finding", "supports_idea", (PACKAGE,), False),
        Effect(record_id(rule), "lesson", None, (record_id(sectoral),), True),
    )
    suspended = suspend(ledger, impact)
    assert suspended.status(record_id(sectoral)) == "suspended"
    assert suspended.status(record_id(rule)) == "admitted"
    assert suspended.eligible_lessons() == ()


@pytest.mark.parametrize(
    "rule, via",
    [
        (lesson("Momentum holds in both backtest packages", PACKAGE, CLEAN), (CLEAN,)),
        (lesson("Momentum holds at home or abroad", PACKAGE, REGIONAL), (record_id(REGIONAL),)),
    ],
    ids=["two-packages", "package-and-finding"],
)
def test_invalidations_accumulate_until_no_support_remains(rule, via):
    ledger = accepted(RESULT, REGIONAL, rule)
    first = invalidation(PACKAGE)
    once = suspend(ledger, assess(ledger, {}, first))
    assert once.eligible_lessons() == (record_id(rule),)
    later = invalidation(CLEAN, at=LATER + timedelta(hours=1), reason="Survivorship bias")
    second = assess(once, {}, later, earlier=(first,))
    assert second.effects == (
        Effect(record_id(REGIONAL), "finding", "supports_idea", (CLEAN,), False),
        Effect(record_id(rule), "lesson", None, via, False),
    )
    twice = suspend(once, second)
    assert twice.status(record_id(rule)) == "suspended"
    assert twice.eligible_lessons() == ()


def test_earlier_losses_count_until_a_rereview_restores_them():
    rule = lesson("Momentum holds in the backtest and abroad", RESULT, CLEAN)
    ledger = accepted(RESULT, rule)
    first = invalidation(PACKAGE)
    later = invalidation(CLEAN, at=LATER + timedelta(hours=1), reason="Survivorship bias")
    second = assess(ledger, {}, later, earlier=(first,))
    assert second.effects == (Effect(record_id(rule), "lesson", None, (CLEAN,), False),)
    once = suspend(ledger, assess(ledger, {}, first))
    assert suspend(once, second).eligible_lessons() == ()
    restored = once.apply(act(RESULT, "review", LATER + timedelta(minutes=30)))
    after_review = assess(restored, {}, later, earlier=(first,))
    assert after_review.effects == (Effect(record_id(rule), "lesson", None, (CLEAN,), True),)


def test_earlier_invalidations_count_whether_applied_or_in_order():
    sourced = claim("Momentum profits reverse after five years", SOURCE)
    rule = lesson("Momentum holds on the backtest, the survey, and abroad", RESULT, sourced, CLEAN)
    study = lesson("Momentum deserves more study", CLEAN)
    ledger = accepted(RESULT, sourced, rule).apply(Proposal(study, LATER + timedelta(hours=2)))
    retraction = invalidation(SOURCE, "retracted_source", at=LATER + timedelta(hours=1))
    last = invalidation(CLEAN, at=LATER + timedelta(hours=3), reason="Survivorship bias")
    impact = assess(ledger, {}, last, earlier=(retraction, invalidation(PACKAGE)))
    assert impact.effects == (
        Effect(record_id(rule), "lesson", None, (CLEAN,), False),
        Effect(record_id(study), "lesson", None, (CLEAN,), False),
    )


def test_invalidations_at_the_same_time_replay_in_one_order():
    both = finding("Momentum earns a premium in both backtests", PACKAGE, CLEAN)
    rule = lesson("Momentum holds in the backtests and the survey", both, SOURCE)
    ledger = accepted(both, rule)
    first = invalidation(PACKAGE, reason="Look-ahead prices")
    second = invalidation(CLEAN, reason="Survivorship bias")
    ledger = suspend(ledger, assess(ledger, {}, first))
    ledger = suspend(ledger, assess(ledger, {}, second, earlier=(first,)))
    ledger = ledger.apply(act(both, "review", LATER + timedelta(hours=1)))
    last = invalidation(SOURCE, "retracted_source", at=LATER + timedelta(hours=2))
    forward = assess(ledger, {}, last, earlier=(first, second))
    assert assess(ledger, {}, last, earlier=(second, first)) == forward


def test_records_proposed_after_an_invalidation_count_once_it_is_replayed():
    first = invalidation(PACKAGE)
    grown = suspend(accepted(RESULT), assess(accepted(RESULT), {}, first))
    late = finding("Momentum earns a premium in the rerun backtest", PACKAGE)
    hedged = lesson("Momentum holds in the rerun and abroad", late, CLEAN)
    alone = lesson("Momentum holds in the rerun", late)
    after = LATER + timedelta(hours=1)
    for record in (late, hedged, alone):
        grown = grown.apply(Proposal(record, after)).apply(act(record, "review", after))
    grown = grown.apply(act(hedged, "admit", after)).apply(act(alone, "admit", after))
    assert grown.eligible_lessons() == (record_id(hedged), record_id(alone))
    later = invalidation(CLEAN, at=LATER + timedelta(hours=2), reason="Survivorship bias")
    second = assess(grown, {}, later, earlier=(first,))
    assert second.effects == (Effect(record_id(hedged), "lesson", None, (CLEAN,), False),)
    assert suspend(grown, second).eligible_lessons() == (record_id(alone),)
    now = LATER + timedelta(hours=3)
    caught_up = suspend(grown, assess(grown, {}, first), at=now)
    reason = f"invalid_package {PACKAGE}: Look-ahead prices"
    assert caught_up.events[len(grown.events) :] == tuple(
        Review(record_id(record), "suspend", "auditor", now, reason) for record in (late, alone)
    )
    assert caught_up.eligible_lessons() == (record_id(hedged),)
    replayed = assess(caught_up, {}, later, earlier=(first,))
    assert suspend(caught_up, replayed, at=now).eligible_lessons() == ()


def test_catch_up_names_every_cause_whatever_order_or_iterables_it_is_given():
    august = sha256(b"prices vintage 2026-08\n")
    september = sha256(b"prices vintage 2026-09\n")
    august_reader = sha256(b"August momentum package")
    september_reader = sha256(b"September momentum package")
    first = finding("Momentum earns a premium on August prices", august_reader)
    second = finding("Momentum earns a premium on September prices", september_reader)
    rule = lesson("Momentum holds on either vintage", first, second)
    ledger = accepted(first, second, rule)
    corrected = invalidation(august, "corrected_vintage", reason="Restated closes")
    revoked = invalidation(
        september, "revoked_vintage", at=LATER + timedelta(hours=1), reason="Withdrawn"
    )
    now = LATER + timedelta(hours=2)
    readers = {august_reader: {august}, september_reader: {september}}
    forward = catch_up(ledger, readers, [corrected, revoked], at=now)
    backward = catch_up(ledger, dict(reversed(readers.items())), iter((revoked, corrected)), at=now)
    corrected_reason = f"corrected_vintage {august}: Restated closes"
    revoked_reason = f"revoked_vintage {september}: Withdrawn"
    assert forward.events[len(ledger.events) :] == (
        Review(record_id(first), "suspend", "auditor", now, corrected_reason),
        Review(record_id(second), "suspend", "auditor", now, revoked_reason),
        Review(record_id(rule), "suspend", "auditor", now, revoked_reason),
        Review(record_id(rule), "suspend", "auditor", now, corrected_reason),
    )
    assert backward.events == forward.events
    assert catch_up(forward, readers, (revoked, corrected), at=now).events == forward.events


def test_retracted_source_suspends_an_extracted_claim_and_its_lesson():
    extracted = claim("Momentum profits reverse after five years", SOURCE)
    rule = lesson("Hold momentum positions for under five years", extracted)
    ledger = accepted(extracted, rule)
    impact = assess(ledger, {}, invalidation(SOURCE, "retracted_source", reason="Retracted"))
    assert impact.withdrawn == (SOURCE,)
    suspended = suspend(ledger, impact)
    assert suspended.status(record_id(extracted)) == "suspended"
    assert suspended.status(record_id(rule)) == "suspended"
    assert suspended.eligible_lessons() == ()


@pytest.mark.parametrize(
    "field, value",
    [
        ("evidence_id", "artifact:momentum-backtest"),
        ("evidence_id", PACKAGE.upper()),
        ("kind", "stale_vintage"),
        ("at", datetime(2026, 10, 22, 12)),
        ("actor", ""),
        ("reason", "   "),
    ],
    ids=["not-digest", "uppercase-digest", "unknown-kind", "naive-time", "no-actor", "blank"],
)
def test_malformed_invalidation_is_refused(field, value):
    with pytest.raises(InvalidationError):
        assess(accepted(RESULT), {}, replace(invalidation(PACKAGE), **{field: value}))


def test_suspension_dated_before_the_log_is_refused_by_the_ledger():
    ledger = accepted(RESULT, RULE)
    impact = assess(ledger, {}, invalidation(PACKAGE, at=T0 - timedelta(days=1)))
    with pytest.raises(ReviewError):
        suspend(ledger, impact)
