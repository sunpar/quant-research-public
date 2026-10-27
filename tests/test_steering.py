from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from quant_research.access import Provider, Restrictions, licensed
from quant_research.campaigns import (
    Campaign,
    CampaignRevision,
    CampaignSpecification,
    DataPermission,
    EvaluationRules,
    ResourceLimits,
    Source,
    TrialReference,
    open_campaign,
    revise,
    save_revision,
    start_campaign,
)
from quant_research.steering import (
    Control,
    Steer,
    SteeringError,
    SteeringLog,
    dump_declaration,
    load_declaration,
    log_bytes,
    read_log,
)

TIME = datetime(2026, 10, 21, 12, tzinfo=UTC)
START = datetime(2000, 1, 1, tzinfo=UTC)
END = datetime(2019, 12, 31, tzinfo=UTC)
PAPER = hashlib.sha256(b"synthetic paper\n").hexdigest()
SPEC = CampaignSpecification(
    "Does 12-1 momentum add value beyond book-to-market?",
    (Source("momentum-paper", "https://example.org/momentum.pdf", PAPER, "paper"),),
    (DataPermission("synthetic/v1", START, END),),
    EvaluationRules("pilot-evaluation/v1", hashlib.sha256(b"pilot rules\n").hexdigest()),
    ResourceLimits(20, 60, 7200),
)
WITH_COSTS = EvaluationRules(
    "pilot-evaluation/v2", hashlib.sha256(b"rules with costs\n").hexdigest()
)
PRIOR_TRIAL = TrialReference(uuid4(), uuid4())
NEW_TRIAL = TrialReference(uuid4(), uuid4())
DECLARED = Restrictions(
    available_at=datetime(2025, 1, 1, tzinfo=UTC),
    labels=frozenset({licensed("vendor-b"), licensed("vendor-a")}),
    approved_for=frozenset({Provider("codex-subscription")}),
)
# Written by main's steering.py at 4b06fe4, before events could declare restrictions.
MAIN_BYTES = (
    '{"campaign_id":"6f1c0e0a-3b7d-4c55-9a7e-2d4f8b1c9e01","events":[{"action":"pause",'
    '"actor":"researcher","at":"2026-10-21T12:00:00Z","reason":"Check the inputs"},'
    '{"action":"annotate","actor":"researcher","at":"2026-10-21T12:01:00Z",'
    '"reason":"Z\u00fcrich data arrives late"},{"action":"steer","actor":"reviewer",'
    f'"at":"2026-10-22T12:00:00Z","parent":"{"a" * 64}","reason":"Add costs after review",'
    f'"revision":"{"b" * 64}"}},{{"action":"resume","actor":"researcher",'
    '"at":"2026-10-23T12:00:00Z","reason":"Inputs checked"},{"action":"cancel",'
    '"actor":"runner","at":"2026-10-23T12:01:00Z","reason":"Stop the campaign"}],'
    '"schema_version":1}\n'
)


def canonical(value: dict) -> bytes:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (text + "\n").encode()


def run(log: SteeringLog, *actions: str, start: datetime = TIME) -> SteeringLog:
    for minute, action in enumerate(actions):
        log = log.record(
            action,
            actor="researcher",
            at=start + timedelta(minutes=minute),
            reason=f"Record {action} {minute}",
        )
    return log


def costs(parent: CampaignRevision, **arguments) -> CampaignRevision:
    defaults = {
        "author": "reviewer",
        "at": parent.created_at + timedelta(days=1),
        "reason": "Add costs after review",
    }
    spec = replace(parent.specification, evaluation=WITH_COSTS)
    return revise(parent, spec, **(defaults | arguments))


def saved(directory: Path, *revisions: CampaignRevision) -> Campaign:
    for revision in revisions:
        save_revision(directory, revision)
    return open_campaign(directory / str(revisions[0].campaign_id))


@pytest.fixture
def first():
    return start_campaign(
        SPEC, author="researcher", at=TIME, reason="Initial protocol", trials=(PRIOR_TRIAL,)
    )


@pytest.fixture
def second(first):
    return costs(first, trials=(NEW_TRIAL,))


@pytest.mark.parametrize(
    "actions, status",
    [
        ((), "running"),
        (("pause",), "paused"),
        (("pause", "resume"), "running"),
        (("cancel",), "cancelled"),
        (("pause", "cancel"), "cancelled"),
        (("pause", "pause"), "paused"),
        (("resume",), "running"),
        (("cancel", "cancel"), "cancelled"),
        (("annotate",), "running"),
        (("pause", "annotate"), "paused"),
        (("cancel", "annotate"), "cancelled"),
    ],
)
def test_recorded_commands_fold_into_the_campaign_status(actions, status):
    log = run(SteeringLog(uuid4()), *actions)
    assert [event.action for event in log.events] == list(actions)
    assert log.status == status


def test_an_exact_retry_appends_nothing_even_after_later_events():
    pause = {"actor": "researcher", "at": TIME, "reason": "Wait for the vendor fix"}
    paused = SteeringLog(uuid4()).record("pause", **pause)
    assert paused.record("pause", **pause) == paused
    eastern = TIME.astimezone(timezone(timedelta(hours=-5)))
    assert paused.record("pause", **(pause | {"at": eastern})) == paused
    resumed = paused.record(
        "resume", actor="researcher", at=TIME + timedelta(hours=1), reason="Vendor fixed it"
    )
    assert resumed.record("pause", **pause) == resumed
    assert (len(resumed.events), resumed.status) == (2, "running")


@pytest.mark.parametrize(
    "actions, action, at, named",
    [
        (("cancel",), "resume", TIME + timedelta(hours=1), "resume a cancelled"),
        (("cancel",), "pause", TIME + timedelta(hours=1), "pause a cancelled"),
        (("pause",), "resume", TIME - timedelta(seconds=1), "dated before"),
        ((), "steer", TIME, "steer(revision)"),
        ((), "stop", TIME, "'stop'"),
        ((), "pause", TIME.replace(tzinfo=None), "timezone"),
    ],
)
def test_record_refuses_an_event_the_log_cannot_take(actions, action, at, named):
    log = run(SteeringLog(uuid4()), *actions)
    with pytest.raises(SteeringError, match=re.escape(named)):
        log.record(action, actor="researcher", at=at, reason="Refused")


def test_a_steer_links_parent_to_child_and_leaves_the_status_alone(first, second):
    at = TIME + timedelta(days=1, hours=3)
    log = SteeringLog(first.campaign_id).steer(second, at=at)
    assert log.events == (
        Steer("reviewer", at, "Add costs after review", first.revision_id, second.revision_id),
    )
    assert log.status == "running"
    assert log.steer(second, at=at) == log
    paused = run(SteeringLog(first.campaign_id), "pause").steer(second, at=at)
    assert [event.action for event in paused.events] == ["pause", "steer"]
    assert paused.status == "paused"


def test_a_revision_created_before_a_pause_can_be_steered_after_it(tmp_path, first, second):
    pause = second.created_at + timedelta(hours=1)
    log = run(SteeringLog(first.campaign_id), "pause", start=pause).steer(
        second, at=pause + timedelta(hours=1)
    )
    assert [(event.action, event.at) for event in log.events] == [
        ("pause", pause),
        ("steer", pause + timedelta(hours=1)),
    ]
    assert log.status == "paused"
    assert read_log(log_bytes(log), saved(tmp_path, first, second)) == log


@pytest.mark.parametrize(
    "refusal",
    ["unchanged", "root", "other-campaign", "cancelled", "before-revision", "steered", "naive"],
)
def test_steer_refuses_a_steer_the_log_cannot_take(first, second, refusal):
    log = SteeringLog(first.campaign_id)
    revision, at = second, second.created_at
    match refusal:
        case "unchanged":
            revision = revise(
                first, SPEC, author="reviewer", at=TIME + timedelta(days=1), reason="Muse"
            )
            named = "annotation"
        case "root":
            revision, named = first, "root"
        case "other-campaign":
            other = start_campaign(SPEC, author="researcher", at=TIME, reason="Initial protocol")
            revision, named = costs(other), str(other.campaign_id)
        case "cancelled":
            log, named = run(log, "cancel"), "steer a cancelled"
        case "before-revision":
            at, named = at - timedelta(seconds=1), f"dated before revision {second.revision_id}"
        case "steered":
            log, at = log.steer(second, at=at), at + timedelta(hours=1)
            named = f"Duplicate steer to revision '{second.revision_id}'"
        case "naive":
            at, named = at.replace(tzinfo=None), "timezone"
    with pytest.raises(SteeringError, match=re.escape(named)):
        log.steer(revision, at=at)


@pytest.mark.smoke
def test_a_steered_campaign_keeps_earlier_results_after_saving_and_reading(tmp_path, first, second):
    log = run(SteeringLog(first.campaign_id), "pause").steer(second, at=second.created_at)
    log = run(log, "resume", start=TIME + timedelta(days=2))
    campaign = saved(tmp_path, first, second)
    stored = read_log(log_bytes(log), campaign)
    assert stored == log
    assert stored.status == "running"
    steer = stored.events[1]
    revisions = {revision.revision_id: revision for revision in campaign.revisions}
    assert revisions[steer.parent].trials == (PRIOR_TRIAL,)
    assert revisions[steer.revision].trials == (PRIOR_TRIAL, NEW_TRIAL)


def test_steering_onto_a_branch_keeps_both_lines_steerable(tmp_path, first):
    overlay = revise(
        first,
        replace(SPEC, question="Does momentum add value beyond a value overlay?"),
        author="reviewer",
        at=TIME + timedelta(days=1),
        reason="Try a value overlay",
        branch="value-overlay",
    )
    main = costs(first, at=TIME + timedelta(days=2))
    widened = revise(
        overlay,
        replace(overlay.specification, limits=ResourceLimits(40, 120, 7200)),
        author="reviewer",
        at=TIME + timedelta(days=3),
        reason="Widen the overlay search",
    )
    log = (
        SteeringLog(first.campaign_id)
        .steer(main, at=main.created_at)
        .steer(overlay, at=main.created_at + timedelta(hours=1))
        .steer(widened, at=widened.created_at)
    )
    campaign = saved(tmp_path, first, overlay, main, widened)
    assert (campaign.head("main"), campaign.head("value-overlay")) == (main, widened)
    assert main.trials == widened.trials == first.trials
    stored = read_log(log_bytes(log), campaign)
    assert [(event.parent, event.revision) for event in stored.events] == [
        (first.revision_id, main.revision_id),
        (first.revision_id, overlay.revision_id),
        (overlay.revision_id, widened.revision_id),
    ]


def test_log_bytes_are_sorted_json_with_utc_times_and_events_in_log_order(first, second):
    eastern = TIME.astimezone(timezone(timedelta(hours=-5))) + timedelta(hours=1)
    log = SteeringLog(first.campaign_id).record(
        "annotate", actor="researcher", at=eastern, reason="Zürich data arrives late"
    )
    assert log_bytes(log.steer(second, at=TIME + timedelta(days=1, hours=2))) == canonical(
        {
            "schema_version": 1,
            "campaign_id": str(first.campaign_id),
            "events": [
                {
                    "action": "annotate",
                    "actor": "researcher",
                    "at": "2026-10-21T13:00:00Z",
                    "reason": "Zürich data arrives late",
                },
                {
                    "action": "steer",
                    "actor": "reviewer",
                    "at": "2026-10-22T14:00:00Z",
                    "reason": "Add costs after review",
                    "parent": first.revision_id,
                    "revision": second.revision_id,
                },
            ],
        }
    )


def test_a_log_with_every_action_reads_back_byte_identically(tmp_path, first, second):
    log = run(SteeringLog(first.campaign_id), "pause", "annotate").steer(
        second, at=second.created_at
    )
    log = run(log, "resume", "cancel", "annotate", "cancel", start=TIME + timedelta(days=2))
    content = log_bytes(log)
    stored = read_log(content, saved(tmp_path, first, second))
    assert stored == log
    assert log_bytes(stored) == content


@pytest.mark.parametrize(
    "damage",
    [
        "indented",
        "schema-version",
        "unknown-field",
        "unknown-action",
        "earlier-time",
        "duplicate",
        "resume-after-cancel",
        "absent-revision",
        "steer-reason",
        "steer-before-revision",
        "steer-twice",
        "other-campaign",
    ],
)
def test_read_log_rejects_a_stored_log_that_breaks_a_rule(tmp_path, first, second, damage):
    log = run(SteeringLog(first.campaign_id), "pause").steer(second, at=second.created_at)
    log = run(log, "cancel", start=TIME + timedelta(days=2))
    campaign = saved(tmp_path, first, second)
    value = json.loads(log_bytes(log))
    assert read_log(canonical(value), campaign) == log
    pause, steer, cancel = value["events"]
    absent = hashlib.sha256(b"unsaved revision\n").hexdigest()
    other = str(uuid4())
    named = {
        "indented": "canonical",
        "schema-version": "schema version 2",
        "unknown-field": "unknown 'priority'",
        "unknown-action": "'stop'",
        "earlier-time": "dated before",
        "duplicate": "Duplicate event",
        "resume-after-cancel": "resume a cancelled",
        "absent-revision": absent,
        "steer-reason": second.revision_id,
        "steer-before-revision": f"dated before revision {second.revision_id}",
        "steer-twice": "Duplicate steer to revision",
        "other-campaign": other,
    }[damage]
    match damage:
        case "schema-version":
            value["schema_version"] = 2
        case "unknown-field":
            pause["priority"] = "high"
        case "unknown-action":
            pause["action"] = "stop"
        case "earlier-time":
            cancel["at"] = "2026-10-21T11:00:00Z"
        case "duplicate":
            value["events"].append(cancel)
        case "resume-after-cancel":
            value["events"].append(cancel | {"action": "resume"})
        case "absent-revision":
            steer["revision"] = absent
        case "steer-reason":
            steer["reason"] = "A different account of the decision"
        case "steer-before-revision":
            steer["at"] = "2026-10-21T13:00:00Z"
        case "steer-twice":
            value["events"] = [pause, steer, steer | {"at": "2026-10-22T13:00:00Z"}, cancel]
        case "other-campaign":
            value |= {"campaign_id": other, "events": [pause, cancel]}
    content = (
        json.dumps(value, sort_keys=True, indent=2).encode()
        if damage == "indented"
        else canonical(value)
    )
    with pytest.raises(SteeringError, match=re.escape(named)):
        read_log(content, campaign)


def test_a_log_without_declarations_keeps_the_bytes_main_wrote():
    events = (
        Control("pause", "researcher", TIME, "Check the inputs"),
        Control("annotate", "researcher", TIME + timedelta(minutes=1), "Zürich data arrives late"),
        Steer("reviewer", TIME + timedelta(days=1), "Add costs after review", "a" * 64, "b" * 64),
        Control("resume", "researcher", TIME + timedelta(days=2), "Inputs checked"),
        Control("cancel", "runner", TIME + timedelta(days=2, minutes=1), "Stop the campaign"),
    )
    log = SteeringLog(UUID("6f1c0e0a-3b7d-4c55-9a7e-2d4f8b1c9e01"), events)
    assert log_bytes(log) == MAIN_BYTES.encode()
    later = log.record("annotate", actor="researcher", at=TIME + timedelta(days=3), reason="Done")
    assert later.events[:-1] == events
    assert "restrictions" not in log_bytes(later).decode()


def test_a_declared_control_and_steer_read_back_with_their_restrictions(tmp_path, first, second):
    log = run(SteeringLog(first.campaign_id), "pause")
    log = log.record(
        "annotate", actor="researcher", at=TIME, reason="Skip 2008", restrictions=DECLARED
    )
    log = log.steer(second, at=second.created_at, restrictions=DECLARED)
    content = log_bytes(log)
    stored = read_log(content, saved(tmp_path, first, second))
    assert stored == log
    assert log_bytes(stored) == content
    pause, annotate, steer = json.loads(content)["events"]
    assert "restrictions" not in pause
    assert (
        annotate["restrictions"]
        == steer["restrictions"]
        == {
            "available_at": "2025-01-01T00:00:00Z",
            "labels": ["licensed:vendor-a", "licensed:vendor-b"],
            "approved_for": ["codex-subscription"],
        }
    )
    assert json.loads(content)["schema_version"] == 1
    assert load_declaration(dump_declaration(DECLARED)) == DECLARED


def test_declared_returns_only_the_declared_events_from_an_index(first, second):
    log = SteeringLog(first.campaign_id).record(
        "annotate", actor="researcher", at=TIME, reason="First note", restrictions=DECLARED
    )
    log = run(log, "pause", "resume")
    log = log.steer(second, at=second.created_at, restrictions=DECLARED)
    note, _, _, steer = log.events
    assert log.declared() == ((note, DECLARED), (steer, DECLARED))
    assert log.declared(1) == ((steer, DECLARED),)
    assert run(SteeringLog(first.campaign_id), "pause", "annotate").declared() == ()


def test_an_identical_declared_event_collapses_and_another_declaration_does_not(first):
    note = {"actor": "researcher", "at": TIME, "reason": "Skip 2008"}
    log = SteeringLog(first.campaign_id).record("annotate", **note, restrictions=DECLARED)
    assert log.record("annotate", **note, restrictions=DECLARED) == log
    narrower = Restrictions(available_at=TIME, labels=frozenset(), approved_for=frozenset())
    assert len(log.record("annotate", **note, restrictions=narrower).events) == 2


@pytest.mark.parametrize(
    "damage",
    ["unsorted", "repeated", "offset-time", "unknown-key", "missing-key", "null", "number-label"],
)
def test_a_damaged_declaration_is_refused(tmp_path, first, damage):
    log = SteeringLog(first.campaign_id).record(
        "annotate", actor="researcher", at=TIME, reason="Skip 2008", restrictions=DECLARED
    )
    value = json.loads(log_bytes(log))
    declared = value["events"][0]["restrictions"]
    match damage:
        case "unsorted":
            declared["labels"].reverse()
        case "repeated":
            declared["labels"].append(declared["labels"][0])
        case "offset-time":
            declared["available_at"] = "2025-01-01T00:00:00+00:00"
        case "unknown-key":
            declared["priority"] = "high"
        case "missing-key":
            del declared["approved_for"]
        case "null":
            value["events"][0]["restrictions"] = declared = None
        case "number-label":
            declared["labels"] = [7]
    with pytest.raises(SteeringError):
        read_log(canonical(value), saved(tmp_path, first))
    with pytest.raises(SteeringError):
        load_declaration(json.dumps(declared))


@pytest.mark.parametrize("text", ["a", "é", "😀", '"'])
def test_steer_revision_size_counts_utf8_bytes_at_the_exact_limit(text):
    from quant_research.steering import REVISION_BYTES, check_revision_size

    unit = text.encode()
    content = unit * (REVISION_BYTES // len(unit))
    assert len(content) == REVISION_BYTES
    check_revision_size(content)
    with pytest.raises(SteeringError, match="65536 UTF-8 bytes"):
        check_revision_size(content + b"x")
