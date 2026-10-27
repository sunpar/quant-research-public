from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from itertools import permutations
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from quant_research.access import (
    INTERNAL_CODE,
    NO_MEMORY,
    MemoryPin,
    Provider,
    Restrictions,
    issue_context,
    licensed,
)
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
    Supersession,
    event_value,
    parse_event,
    record_id,
)

T0 = datetime(2026, 10, 21, 12, tzinfo=UTC)
SCOPE = Scope(
    "US large-cap equities",
    "one month",
    datetime(2010, 1, 1, tzinfo=UTC),
    datetime(2024, 12, 31, tzinfo=UTC),
    ("monthly rebalance", "10 bps one-way costs"),
)
BACKTEST = (External("artifact:momentum-backtest", "supports"),)
VENDOR = licensed("synthetic-vendor")
LOCAL = Provider("local-model")
HOSTED = Provider("hosted-model")


def finding(statement, outcome="supports_idea", evidence=BACKTEST):
    return Finding(statement, "walk-forward/v1", outcome, "researcher", SCOPE, evidence)


def claim(statement, evidence, origin="hypothesis"):
    return Claim(statement, origin, "agent", SCOPE, evidence)


def lesson(statement, evidence):
    return Lesson(statement, "agent", SCOPE, evidence)


def supports(*records):
    return tuple(Cites(record_id(record), "supports") for record in records)


def propose(record, at=T0):
    return Proposal(record, at)


def act(record, action, at=T0):
    return Review(record_id(record), action, "reviewer", at, "Checked the cited evidence")


def supersede(record, replacement, at=T0):
    return Supersession(record_id(record), record_id(replacement), "reviewer", at, "Replaced")


def admit(*lessons):
    return tuple(
        event
        for item in lessons
        for event in (propose(item), act(item, "review"), act(item, "admit"))
    )


def restrictions(*labels, available_at=T0, approved_for=()):
    return Restrictions(
        available_at=available_at, labels=frozenset(labels), approved_for=frozenset(approved_for)
    )


def declare(evidence_id, *labels, available_at=T0, approved_for=(), at=T0):
    declared = restrictions(*labels, available_at=available_at, approved_for=approved_for)
    return Declaration(evidence_id, declared, "ingestion", at, "Labels from the evidence's store")


def run(memory, *clearances, cutoff=T0):
    return issue_context(
        clearances=clearances, cutoff=cutoff, generation="synthetic/v1", providers=(), memory=memory
    )


RESULT = finding("Twelve-month momentum earns a premium after costs")
RULE = lesson("Momentum is a durable cross-sectional effect", supports(RESULT))
REVIEWED = (propose(RESULT), act(RESULT, "review"))
SECTOR = claim("Momentum also appears in sector portfolios", BACKTEST, "extracted")
RIVAL = finding("The momentum premium disappears after 2009")
CONTRARIAN = lesson(
    "Momentum crashes after market rebounds", (Cites(record_id(RESULT), "contradicts"),)
)
SUCCESSOR = lesson("Momentum persists across regions", supports(RESULT))
FALLBACK = lesson("Momentum persists across decades", supports(RESULT))
REPLICATION = finding("Twelve-month momentum earns a premium in international equities")
HYPOTHESIS = claim("Momentum reflects slow diffusion of news", supports(RESULT))
INDUSTRY = claim("Momentum also appears in industry portfolios", BACKTEST, "extracted")
CORROBORATED = claim(
    "News diffusion explains momentum in two samples", supports(HYPOTHESIS, REPLICATION)
)


def test_record_ids_keep_the_content_addresses_that_stored_logs_name():
    assert record_id(RESULT) == "f4bfc0d7bb49b79a2d2e309d007c39f64fe1ef158598501082f268bd91109fac"
    assert record_id(RULE) == "fe5ca8001d6e76f04d2a00285d409b0549375e1b6c4af8269355010e7e983267"


@pytest.mark.smoke
def test_only_admitted_lessons_with_standing_support_are_eligible():
    proposed = lesson("Momentum works in small caps", supports(RESULT))
    reviewed = lesson("Momentum works in emerging markets", supports(RESULT))
    archived = lesson("Momentum holds in the archived backtest", BACKTEST)
    ledger = Ledger(
        (
            *REVIEWED,
            propose(proposed),
            propose(reviewed),
            act(reviewed, "review"),
            propose(RULE),
            act(RULE, "review"),
            act(RULE, "admit"),
            propose(archived),
            act(archived, "review"),
            act(archived, "admit"),
        )
    )
    assert ledger.eligible_lessons() == (record_id(RULE), record_id(archived))
    assert ledger.status(record_id(proposed)) == "proposed"
    assert ledger.status(record_id(reviewed)) == "reviewed"


@pytest.mark.parametrize(
    "base, event",
    [
        pytest.param((*REVIEWED, propose(RULE)), act(RULE, "admit"), id="admit-from-proposed"),
        pytest.param(REVIEWED, act(RESULT, "review"), id="review-twice"),
        pytest.param(
            (propose(SECTOR), act(SECTOR, "review")), act(SECTOR, "admit"), id="admit-claim"
        ),
        pytest.param(REVIEWED, act(RESULT, "admit"), id="admit-finding"),
        pytest.param(
            (*REVIEWED, propose(CONTRARIAN), act(CONTRARIAN, "review")),
            act(CONTRARIAN, "admit"),
            id="admit-without-supporting-link",
        ),
        pytest.param(
            (propose(RESULT), propose(RULE), act(RULE, "review")),
            act(RULE, "admit"),
            id="admit-on-unreviewed-support",
        ),
        pytest.param((), act(RESULT, "review"), id="review-unknown-record"),
        pytest.param((), propose(RULE), id="cite-unknown-record"),
        pytest.param(REVIEWED, supersede(RESULT, RIVAL), id="supersede-with-unknown-replacement"),
        pytest.param((propose(RESULT),), propose(RESULT), id="duplicate-proposal"),
        pytest.param(
            (),
            propose(claim("Momentum studies agree", BACKTEST, "summary")),
            id="summary-citing-no-record",
        ),
        pytest.param(
            (),
            propose(replace(RESULT, scope=replace(SCOPE, start=SCOPE.end, end=SCOPE.start))),
            id="period-ends-before-start",
        ),
        pytest.param(
            (propose(RESULT, T0 + timedelta(hours=1)),),
            act(RESULT, "review", T0),
            id="event-earlier-than-previous",
        ),
        pytest.param(
            (*REVIEWED, propose(SECTOR)), supersede(RESULT, SECTOR), id="supersede-with-other-kind"
        ),
        pytest.param(
            (*REVIEWED, propose(RIVAL), act(RIVAL, "retract")),
            supersede(RESULT, RIVAL),
            id="supersede-with-retracted-replacement",
        ),
        pytest.param(
            (*REVIEWED, propose(RIVAL), act(RIVAL, "suspend")),
            supersede(RESULT, RIVAL),
            id="supersede-with-suspended-replacement",
        ),
        pytest.param(
            (*REVIEWED, act(RESULT, "suspend"), propose(RULE), act(RULE, "review")),
            act(RULE, "admit"),
            id="admit-on-suspended-support",
        ),
        pytest.param(REVIEWED, supersede(RESULT, RESULT), id="supersede-with-itself"),
        pytest.param((), propose(RESULT, datetime(2026, 10, 21, 12)), id="naive-event-time"),
        pytest.param(
            REVIEWED,
            Review(record_id(RESULT), "supersede", "reviewer", T0, "Replaced"),
            id="supersede-as-a-review-action",
        ),
        pytest.param(
            REVIEWED,
            propose(lesson("Momentum is robust", (Cites(record_id(RESULT), "Supports"),))),
            id="unknown-evidence-relation",
        ),
        pytest.param((), propose(finding("Momentum failed", "invalid")), id="unknown-outcome"),
        pytest.param((), propose(claim("Studies agree", BACKTEST, "Summary")), id="unknown-origin"),
        pytest.param((), declare(" "), id="blank-declared-evidence"),
    ],
)
def test_illegal_events_are_rejected_with_a_reason(base, event):
    ledger = Ledger(base)
    with pytest.raises(ReviewError) as rejected:
        ledger.apply(event)
    assert str(rejected.value)
    with pytest.raises(ReviewError):
        Ledger((*base, event))


@pytest.mark.parametrize(
    "ending", [act(RULE, "retract"), supersede(RULE, SUCCESSOR)], ids=["retracted", "superseded"]
)
@pytest.mark.parametrize(
    "steps, event",
    [
        ((), act(RULE, "review")),
        ((act(RULE, "review"),), act(RULE, "admit")),
        ((), act(RULE, "retract")),
        ((), supersede(RULE, FALLBACK)),
        ((), act(RULE, "suspend")),
        ((act(RULE, "review"),), act(RULE, "suspend")),
        ((act(RULE, "review"), act(RULE, "admit")), act(RULE, "suspend")),
        ((act(RULE, "suspend"),), act(RULE, "suspend")),
        ((act(RULE, "suspend"),), act(RULE, "review")),
        ((act(RULE, "suspend"),), act(RULE, "retract")),
        ((act(RULE, "suspend"),), supersede(RULE, FALLBACK)),
    ],
    ids=[
        "review",
        "admit",
        "retract",
        "supersede",
        "suspend-proposed",
        "suspend-reviewed",
        "suspend-admitted",
        "suspend-suspended",
        "review-suspended",
        "retract-suspended",
        "supersede-suspended",
    ],
)
def test_retracted_and_superseded_records_accept_no_further_action(ending, steps, event):
    live = Ledger((*REVIEWED, propose(RULE), *steps, propose(SUCCESSOR), propose(FALLBACK)))
    assert live.apply(event).history(record_id(RULE))[-1] == event
    ended = live.apply(ending)
    with pytest.raises(ReviewError):
        ended.apply(event)


@pytest.mark.smoke
def test_transitions_append_events_and_earlier_states_stay_queryable():
    hours = [T0 + timedelta(hours=hour) for hour in range(6)]
    events = (
        propose(RESULT, hours[0]),
        act(RESULT, "review", hours[1]),
        propose(RULE, hours[2]),
        act(RULE, "review", hours[3]),
        act(RULE, "admit", hours[4]),
    )
    admitted = Ledger(events)
    retraction = act(RESULT, "retract", hours[5])
    retracted = admitted.apply(retraction)
    assert retracted.events == (*events, retraction)
    assert retracted.history(record_id(RESULT)) == (events[0], events[1], retraction)
    assert retracted.history(record_id(RULE)) == events[2:]
    assert retracted.status(record_id(RESULT)) == "retracted"
    assert retracted.eligible_lessons() == ()
    assert admitted.events == events
    with pytest.raises(AttributeError):
        admitted.events = ()
    stored = list(events)
    rebuilt = Ledger(stored)
    stored.append(retraction)
    assert rebuilt.events == events
    assert admitted.status(record_id(RESULT)) == "reviewed"
    assert admitted.eligible_lessons() == (record_id(RULE),)
    assert retracted.as_of(hours[4]).status(record_id(RESULT)) == "reviewed"
    assert retracted.as_of(hours[4]).eligible_lessons() == (record_id(RULE),)
    assert retracted.as_of(hours[3]).status(record_id(RULE)) == "reviewed"
    assert retracted.as_of(hours[3]).eligible_lessons() == ()


@pytest.mark.smoke
@pytest.mark.parametrize(
    "withdrawal",
    [act(RESULT, "retract"), supersede(RESULT, RIVAL)],
    ids=["retracted", "superseded"],
)
def test_withdrawing_evidence_two_hops_below_a_lesson_makes_it_ineligible(withdrawal):
    hypothesis = claim("Momentum reflects slow diffusion of news", supports(RESULT))
    rule = lesson("Favor momentum where news diffuses slowly", supports(hypothesis))
    ledger = Ledger(
        (
            *REVIEWED,
            propose(hypothesis),
            act(hypothesis, "review"),
            propose(rule),
            act(rule, "review"),
            act(rule, "admit"),
            propose(RIVAL),
        )
    )
    assert ledger.eligible_lessons() == (record_id(rule),)
    withdrawn = ledger.apply(withdrawal)
    assert withdrawn.eligible_lessons() == ()
    assert withdrawn.status(record_id(rule)) == "admitted"
    assert withdrawn.status(record_id(hypothesis)) == "reviewed"
    assert withdrawn.history(record_id(rule)) == ledger.history(record_id(rule))


def test_summary_citing_a_lesson_before_it_is_proposed_is_rejected_as_unknown():
    summary = claim("Momentum lessons agree", supports(RULE), "summary")
    ledger = Ledger(REVIEWED)
    with pytest.raises(ReviewError, match="(?i)unknown"):
        ledger.apply(propose(summary))
    recorded = ledger.apply(propose(RULE)).apply(propose(summary))
    assert recorded.status(record_id(summary)) == "proposed"


def test_lesson_cannot_be_superseded_by_one_resting_on_its_own_summary():
    summary = claim("Momentum lessons agree", supports(RULE), "summary")
    laundered = lesson("Momentum is a reliable effect", supports(summary))
    direct = lesson("Momentum is a reliable effect in large caps", supports(RESULT))
    ledger = Ledger(
        (
            *REVIEWED,
            propose(RULE),
            act(RULE, "review"),
            act(RULE, "admit"),
            propose(summary),
            act(summary, "review"),
            propose(laundered),
            propose(direct),
        )
    )
    with pytest.raises(ReviewError):
        ledger.apply(supersede(RULE, laundered))
    assert ledger.apply(supersede(RULE, direct)).status(record_id(RULE)) == "superseded"


def test_invalid_implementation_supersedes_a_valid_rejection_without_erasing_it():
    rejection = finding("Short-term reversal loses money after costs", "rejects_idea")
    avoid = lesson("Do not pursue short-term reversal", supports(rejection))
    invalid = finding(
        "The reversal backtest used look-ahead prices",
        "invalid_implementation",
        (Cites(record_id(rejection), "contradicts"),),
    )
    ledger = Ledger(
        (
            propose(rejection),
            act(rejection, "review"),
            propose(avoid),
            act(avoid, "review"),
            act(avoid, "admit"),
            propose(invalid),
            act(invalid, "review"),
        )
    )
    assert ledger.eligible_lessons() == (record_id(avoid),)
    corrected = ledger.apply(supersede(rejection, invalid))
    assert corrected.eligible_lessons() == ()
    assert corrected.status(record_id(rejection)) == "superseded"
    assert corrected.history(record_id(rejection))[0].record.outcome == "rejects_idea"
    assert corrected.history(record_id(rejection))[-1].replacement == record_id(invalid)
    assert corrected.record(record_id(invalid)).outcome == "invalid_implementation"


def test_reviewed_lesson_cannot_support_another_until_admitted():
    derived = lesson("Momentum deserves a standing allocation", supports(RULE))
    ledger = Ledger(
        (*REVIEWED, propose(RULE), act(RULE, "review"), propose(derived), act(derived, "review"))
    )
    with pytest.raises(ReviewError):
        ledger.apply(act(derived, "admit"))
    admitted = ledger.apply(act(RULE, "admit")).apply(act(derived, "admit"))
    assert admitted.eligible_lessons() == (record_id(RULE), record_id(derived))


def test_rereviewed_suspended_lesson_needs_admission_again():
    admitted = (*REVIEWED, propose(RULE), act(RULE, "review"), act(RULE, "admit"))
    suspended = Ledger((*admitted, act(RULE, "suspend")))
    assert suspended.status(record_id(RULE)) == "suspended"
    assert suspended.eligible_lessons() == ()
    rereviewed = suspended.apply(act(RULE, "review"))
    assert rereviewed.status(record_id(RULE)) == "reviewed"
    assert rereviewed.eligible_lessons() == ()
    readmitted = rereviewed.apply(act(RULE, "admit"))
    assert readmitted.eligible_lessons() == (record_id(RULE),)
    assert [event.action for event in readmitted.history(record_id(RULE))[1:]] == [
        "review",
        "admit",
        "suspend",
        "review",
        "admit",
    ]


@pytest.mark.smoke
def test_suspended_support_withdraws_nothing_while_other_support_remains():
    hedged = lesson("Momentum survives in at least one region", supports(RESULT, REPLICATION))
    ledger = Ledger(
        (
            *REVIEWED,
            propose(REPLICATION),
            act(REPLICATION, "review"),
            propose(RULE),
            act(RULE, "review"),
            act(RULE, "admit"),
            propose(hedged),
            act(hedged, "review"),
            act(hedged, "admit"),
        )
    )
    suspended = ledger.apply(act(RESULT, "suspend"))
    assert suspended.eligible_lessons() == (record_id(hedged),)
    assert suspended.status(record_id(RULE)) == "admitted"
    assert suspended.history(record_id(RULE)) == ledger.history(record_id(RULE))
    assert suspended.apply(act(REPLICATION, "retract")).eligible_lessons() == ()


@pytest.mark.parametrize(
    "withdrawal",
    [act(SECTOR, "retract"), supersede(SECTOR, INDUSTRY)],
    ids=["retracted", "superseded"],
)
def test_suspension_does_not_hide_a_withdrawal_beneath_it(withdrawal):
    sectoral = finding(
        "Momentum earns a premium in sector portfolios", evidence=(*BACKTEST, *supports(SECTOR))
    )
    rule = lesson("Momentum holds across portfolio sorts", supports(sectoral, REPLICATION))
    admitted = Ledger(
        (
            propose(SECTOR),
            act(SECTOR, "review"),
            propose(INDUSTRY),
            propose(sectoral),
            act(sectoral, "review"),
            propose(REPLICATION),
            act(REPLICATION, "review"),
            propose(rule),
            act(rule, "review"),
            act(rule, "admit"),
        )
    )
    assert admitted.eligible_lessons() == (record_id(rule),)
    withdrawn = admitted.apply(withdrawal)
    assert withdrawn.eligible_lessons() == ()
    assert withdrawn.apply(act(sectoral, "suspend")).eligible_lessons() == ()


SUSPENDED_HYPOTHESIS = (
    propose(HYPOTHESIS),
    act(HYPOTHESIS, "review"),
    propose(REPLICATION),
    act(REPLICATION, "review"),
    propose(CORROBORATED),
    act(CORROBORATED, "review"),
    act(HYPOTHESIS, "suspend"),
)


@pytest.mark.parametrize(
    "steps, record, expected",
    [
        ((), RESULT, True),
        ((act(RESULT, "suspend"),), RESULT, False),
        ((propose(HYPOTHESIS),), HYPOTHESIS, False),
        ((propose(HYPOTHESIS), act(HYPOTHESIS, "review")), HYPOTHESIS, True),
        (
            (propose(HYPOTHESIS), act(HYPOTHESIS, "review"), act(RESULT, "suspend")),
            HYPOTHESIS,
            False,
        ),
        ((propose(RULE), act(RULE, "review")), RULE, False),
        ((propose(RULE), act(RULE, "review"), act(RULE, "admit")), RULE, True),
        (
            (propose(RULE), act(RULE, "review"), act(RULE, "admit"), act(RESULT, "retract")),
            RULE,
            False,
        ),
        (SUSPENDED_HYPOTHESIS, CORROBORATED, True),
        ((*SUSPENDED_HYPOTHESIS, act(RESULT, "retract")), CORROBORATED, False),
    ],
    ids=[
        "reviewed-finding",
        "suspended-finding",
        "proposed-claim",
        "reviewed-claim",
        "claim-on-suspended-support",
        "reviewed-lesson",
        "admitted-lesson",
        "lesson-on-retracted-support",
        "claim-on-suspended-and-other-support",
        "claim-on-retraction-beneath-suspension",
    ],
)
def test_record_stands_with_its_kind_status_and_holding_support(steps, record, expected):
    assert Ledger((*REVIEWED, *steps)).stands(record_id(record)) is expected


def test_restrictions_derive_from_every_external_item_through_both_relations():
    rival = finding(
        "The momentum premium disappears after 2009",
        evidence=(External("artifact:rival-backtest", "supports"),),
    )
    diffusion = claim(
        "Momentum reflects slow diffusion of news",
        (*supports(RESULT), Cites(record_id(rival), "contradicts")),
    )
    rule = lesson("Favor momentum where news diffuses slowly", supports(diffusion))
    proposals = tuple(propose(record) for record in (RESULT, rival, diffusion, rule))
    declarations = (
        declare(
            "artifact:momentum-backtest",
            VENDOR,
            available_at=T0 - timedelta(days=2),
            approved_for=(LOCAL, HOSTED),
        ),
        declare(
            "artifact:rival-backtest",
            INTERNAL_CODE,
            available_at=T0 - timedelta(days=1),
            approved_for=(LOCAL,),
        ),
    )
    expected = restrictions(
        VENDOR, INTERNAL_CODE, available_at=T0 - timedelta(days=1), approved_for=(LOCAL,)
    )
    for order in permutations(declarations):
        ledger = Ledger((*proposals, *order))
        assert ledger.restrictions(record_id(rule)) == expected
    assert ledger.apply(act(rival, "retract")).restrictions(record_id(rule)) == expected
    assert ledger.restrictions(record_id(RESULT)) == declarations[0].restrictions


def test_restrictions_are_unknown_while_an_item_is_undeclared_or_a_record_has_no_evidence():
    guess = claim("Momentum reflects investor overconfidence", ())
    behavioral = lesson("Momentum is a behavioral effect", (*BACKTEST, *supports(guess)))
    declared = Ledger(
        (
            *(propose(record) for record in (RESULT, RULE, guess, behavioral)),
            declare("artifact:momentum-backtest"),
        )
    )
    assert declared.restrictions(record_id(RULE)) == restrictions()
    assert declared.restrictions(record_id(guess)) is None
    assert declared.restrictions(record_id(behavioral)) is None
    assert Ledger(declared.events[:-1]).restrictions(record_id(RULE)) is None
    with pytest.raises(ReviewError):
        declared.restrictions("f" * 64)


def test_repeated_declarations_only_tighten_and_an_identical_repeat_changes_nothing():
    first = declare(
        "artifact:momentum-backtest",
        VENDOR,
        available_at=T0 - timedelta(days=1),
        approved_for=(LOCAL, HOSTED),
    )
    looser = declare("artifact:momentum-backtest", available_at=T0 - timedelta(days=2))
    tighter = declare("artifact:momentum-backtest", INTERNAL_CODE, approved_for=(LOCAL,))
    declared = Ledger((propose(RESULT), first))
    identity = record_id(RESULT)
    assert declared.restrictions(identity) == first.restrictions
    assert declared.apply(first).restrictions(identity) == first.restrictions
    assert declared.apply(looser).restrictions(identity) == restrictions(
        VENDOR, available_at=T0 - timedelta(days=1)
    )
    expected = restrictions(VENDOR, INTERNAL_CODE, approved_for=(LOCAL,))
    assert declared.apply(tighter).restrictions(identity) == expected
    assert Ledger((propose(RESULT), tighter, first)).restrictions(identity) == expected


def test_a_head_names_one_exact_log_and_an_earlier_head_names_its_prefix():
    hours = [T0 + timedelta(hours=hour) for hour in range(3)]
    events = (
        propose(RESULT, hours[0]),
        act(RESULT, "review", hours[1]),
        declare("artifact:momentum-backtest", at=hours[2]),
    )
    ledger = Ledger(events)
    assert Ledger().head == NO_MEMORY
    assert Ledger(events).head == ledger.head
    zone = ZoneInfo("America/New_York")
    elsewhere = tuple(replace(event, at=event.at.astimezone(zone)) for event in events)
    assert Ledger(elsewhere).head == ledger.head
    assert len({Ledger(events[:length]).head for length in range(len(events) + 1)}) == 4
    assert ledger.as_of(hours[1]).head == Ledger(events[:2]).head
    relabelled = (*events[:2], declare("artifact:momentum-backtest", VENDOR, at=hours[2]))
    assert Ledger(relabelled).head != ledger.head


def test_admissible_lessons_are_eligible_at_the_pin_and_readable_by_the_run():
    open_rule = lesson("Momentum holds in the open backtest", BACKTEST)
    vendor_rule = lesson(
        "Momentum holds in the vendor backtest", (External("artifact:vendor", "supports"),)
    )
    later_rule = lesson(
        "Momentum holds in the later backtest", (External("artifact:later", "supports"),)
    )
    withdrawn = lesson("Momentum holds in the open backtest after costs", BACKTEST)
    kept = lesson("Momentum holds in the open backtest before costs", BACKTEST)
    fresh = lesson("Momentum holds in the open backtest across regions", BACKTEST)
    pinned = Ledger(
        (
            declare("artifact:momentum-backtest"),
            declare("artifact:vendor", VENDOR),
            declare("artifact:later", available_at=T0 + timedelta(days=1)),
            *admit(open_rule, vendor_rule, later_rule, withdrawn, kept),
            act(withdrawn, "suspend"),
        )
    )
    ledger = Ledger((*pinned.events, act(kept, "suspend"), *admit(fresh)))
    ids = {item: record_id(item) for item in (open_rule, vendor_rule, later_rule, kept, fresh)}
    assert ledger.admissible_lessons(run(pinned.head)) == (ids[open_rule], ids[kept])
    assert ledger.admissible_lesson_restrictions(run(pinned.head)) == (
        (ids[open_rule], restrictions()),
        (ids[kept], restrictions()),
    )
    assert ledger.admissible_lessons(run(pinned.head, VENDOR)) == (
        ids[open_rule],
        ids[vendor_rule],
        ids[kept],
    )
    assert ledger.admissible_lessons(run(pinned.head, cutoff=T0 + timedelta(days=1))) == (
        ids[open_rule],
        ids[later_rule],
        ids[kept],
    )
    assert ledger.admissible_lessons(run(ledger.head)) == (ids[open_rule], ids[fresh])
    assert ledger.admissible_lessons(run(NO_MEMORY)) == ()
    assert ledger.admissible_lesson_restrictions(run(NO_MEMORY)) == ()
    for stranger in (
        run(MemoryPin("f" * 64)),
        SimpleNamespace(memory=ledger.head, read_denials=lambda item: ()),
    ):
        with pytest.raises(ReviewError):
            ledger.admissible_lessons(stranger)
        with pytest.raises(ReviewError):
            ledger.admissible_lesson_restrictions(stranger)


def test_a_later_declaration_does_not_admit_undeclared_evidence_at_an_older_pin():
    pinned = Ledger((*REVIEWED, *admit(RULE)))
    ledger = pinned.apply(declare("artifact:momentum-backtest"))
    identity = record_id(RULE)
    assert ledger.eligible_lessons() == (identity,)
    assert ledger.admissible_lesson_restrictions(run(pinned.head)) == ()
    assert ledger.admissible_lessons(run(pinned.head)) == ()
    assert ledger.admissible_lesson_restrictions(run(ledger.head)) == ((identity, restrictions()),)


def test_every_event_kind_round_trips_through_its_canonical_value():
    zone = ZoneInfo("America/New_York")
    hedged = Claim(
        "Momentum reflects slow diffusion of news",
        "hypothesis",
        "agent",
        replace(SCOPE, start=SCOPE.start.astimezone(zone)),
        (*BACKTEST, *supports(RESULT)),
    )
    events = (
        declare("artifact:momentum-backtest", VENDOR, INTERNAL_CODE, approved_for=(LOCAL, HOSTED)),
        *REVIEWED,
        propose(hedged),
        *admit(RULE),
        act(RULE, "suspend"),
        act(RULE, "retract"),
        propose(RIVAL),
        supersede(RESULT, RIVAL),
    )
    ledger = Ledger(tuple(replace(event, at=event.at.astimezone(zone)) for event in events))
    decoded = tuple(parse_event(event_value(event)) for event in ledger.events)
    assert decoded == ledger.events
    assert Ledger(decoded).head == ledger.head


DECLARED = declare("artifact:momentum-backtest", VENDOR, INTERNAL_CODE, approved_for=(LOCAL,))


def relabel(value):
    restrictions = value["restrictions"]
    return {**value, "restrictions": {**restrictions, "labels": restrictions["labels"][::-1]}}


@pytest.mark.parametrize(
    "event, change",
    [
        (DECLARED, lambda value: {**value, "kind": "comment"}),
        (DECLARED, lambda value: {key: item for key, item in value.items() if key != "reason"}),
        (DECLARED, lambda value: {**value, "note": "Unreviewed"}),
        (DECLARED, lambda value: {**value, "at": "2026-10-21T12:00:00+00:00"}),
        (DECLARED, lambda value: {**value, "declarer": 7}),
        (DECLARED, relabel),
        (propose(RULE), lambda value: {**value, "record": {**value["record"], "kind": "memo"}}),
        (
            propose(RULE),
            lambda value: {
                **value,
                "record": {**value["record"], "evidence": [{"record_id": "f" * 64}]},
            },
        ),
    ],
    ids=[
        "unknown-kind",
        "missing-key",
        "extra-key",
        "noncanonical-time",
        "text-not-a-string",
        "unsorted-labels",
        "unknown-record-kind",
        "evidence-missing-relation",
    ],
)
def test_parse_event_refuses_a_value_that_is_not_a_canonical_event(event, change):
    value = event_value(event)
    assert parse_event(value) == event
    with pytest.raises(ReviewError):
        parse_event(change(value))


def test_times_in_a_repeated_hour_are_ordered_by_instant():
    zone = ZoneInfo("America/New_York")
    daylight = datetime(2026, 11, 1, 1, 50, tzinfo=zone)
    standard = datetime(2026, 11, 1, 1, 10, fold=1, tzinfo=zone)
    assert daylight.astimezone(UTC) < standard.astimezone(UTC)
    ledger = Ledger((propose(RESULT, daylight), act(RESULT, "review", standard)))
    decoded = Ledger(tuple(parse_event(event_value(event)) for event in ledger.events))
    assert decoded.head == ledger.head
    assert ledger.as_of(datetime(2026, 11, 1, 1, 20, fold=1, tzinfo=zone)).events == ledger.events
    assert ledger.as_of(datetime(2026, 11, 1, 1, 55, tzinfo=zone)).events == ledger.events[:1]
    with pytest.raises(ReviewError, match="must not decrease"):
        Ledger((propose(RESULT, standard), act(RESULT, "review", daylight)))
    reversed_scope = replace(RESULT, scope=replace(SCOPE, start=standard, end=daylight))
    with pytest.raises(ReviewError, match="ends before it starts"):
        Ledger((propose(reversed_scope),))
