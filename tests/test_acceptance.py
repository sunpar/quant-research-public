from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from quant_research.acceptance import (
    AcceptanceError,
    AcceptanceLog,
    Accepted,
    Attempt,
    Completed,
    Execution,
    Submitted,
    dump_log,
    load_log,
)
from quant_research.packages import Outcome

AT = datetime(2026, 1, 2, tzinfo=UTC)
LATER = AT + timedelta(hours=1)
SUCCESS = Outcome("successful", "Signal matches the reference outputs")
FAILURE = Outcome("failed", "Worker lost its lease")
TRIAL, RUN, ATTEMPT, RETRY, OTHER_RUN, OTHER = (uuid4() for _ in range(6))


def _accept(attempt, supersedes=None, reason="Matches the reference outputs"):
    return Accepted(attempt, supersedes, "researcher", reason, LATER)


BASE = (
    Submitted(TRIAL, RUN, ATTEMPT, None, AT),
    Completed(ATTEMPT, SUCCESS, LATER),
    Submitted(TRIAL, OTHER_RUN, OTHER, None, AT),
)
HISTORY = (
    *BASE[:2],
    Submitted(TRIAL, RUN, RETRY, ATTEMPT, LATER),
    Completed(
        RETRY,
        Outcome("successful", "Rerun on the corrected vintage", ("returns",), ("Synthetic",)),
        LATER,
    ),
    _accept(ATTEMPT),
    _accept(RETRY, ATTEMPT, "The first run read a stale data vintage"),
)


def _canonical_json(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{text}\n".encode()


@pytest.mark.smoke
def test_successful_attempt_is_unaccepted_until_an_explicit_decision():
    log = AcceptanceLog(BASE[:2])
    assert log.executions() == {RUN: Execution(TRIAL, RUN, (Attempt(*BASE[:2]),), ())}
    assert log.executions()[RUN].accepted is None
    log.record(_accept(ATTEMPT))
    assert log.executions()[RUN].accepted == Attempt(*BASE[:2])


@pytest.mark.parametrize(
    "outcome, finished_at",
    [(FAILURE, LATER), (SUCCESS, LATER + timedelta(minutes=5))],
    ids=["different-status", "different-finish-time"],
)
def test_redelivered_events_record_once_and_a_different_completion_is_refused(outcome, finished_at):
    log = AcceptanceLog([*BASE[:2], *BASE[:2]])
    assert log.events == BASE[:2]
    recorded = f"already completed as 'successful' at {LATER.isoformat()}"
    with pytest.raises(AcceptanceError, match=re.escape(recorded)):
        log.record(Completed(ATTEMPT, outcome, finished_at))
    assert log.events == BASE[:2]


def test_ambiguous_submission_keeps_every_attempt_and_one_accepted_result():
    trial, run, first, retry = (uuid4() for _ in range(4))
    events = (
        Submitted(trial, run, first, None, AT),
        Submitted(trial, run, retry, first, AT + timedelta(minutes=10)),
        Completed(retry, SUCCESS, AT + timedelta(minutes=20)),
        _accept(retry),
        Completed(first, SUCCESS, LATER + timedelta(hours=1)),
    )
    log = AcceptanceLog(events)
    with pytest.raises(
        AcceptanceError, match=f"already accepted attempt {retry}; a new acceptance"
    ):
        log.record(_accept(first))
    execution = log.executions()[run]
    assert execution.attempts == (Attempt(events[0], events[4]), Attempt(events[1], events[2]))
    assert execution.decisions == (events[3],)
    assert execution.accepted == Attempt(events[1], events[2])
    assert log.events == events


def test_interrupted_attempt_cannot_be_accepted_and_stays_visible():
    log = AcceptanceLog(BASE)
    with pytest.raises(AcceptanceError, match="no completion; an interrupted attempt cannot be"):
        log.record(_accept(OTHER))
    assert log.executions()[OTHER_RUN] == Execution(TRIAL, OTHER_RUN, (Attempt(BASE[2], None),), ())
    assert log.events == BASE


def test_acceptance_changes_only_by_superseding_the_current_one():
    first, supersession = HISTORY[4:]
    log = AcceptanceLog(HISTORY[:4])
    with pytest.raises(AcceptanceError, match=f"Run {RUN} has no accepted attempt to supersede"):
        log.record(supersession)
    log.record(first)
    with pytest.raises(AcceptanceError, match=f"accepted attempt {ATTEMPT}; a new acceptance must"):
        log.record(_accept(RETRY))
    log.record(supersession)
    with pytest.raises(AcceptanceError, match=f"currently accepts attempt {RETRY}, not {ATTEMPT}"):
        log.record(_accept(RETRY, ATTEMPT, "A concurrent review of the same vintage"))
    log.record(first)
    execution = log.executions()[RUN]
    assert execution.decisions == (first, supersession)
    assert execution.accepted == Attempt(*HISTORY[2:4])
    assert log.events == HISTORY


def test_superseding_the_accepted_attempt_itself_records_a_new_reason():
    first = _accept(ATTEMPT)
    restated = _accept(ATTEMPT, ATTEMPT, "An independent recomputation matched the outputs")
    execution = AcceptanceLog([*BASE[:2], first, restated]).executions()[RUN]
    assert execution.decisions == (first, restated)
    assert execution.accepted == Attempt(*BASE[:2])


def test_repeat_is_a_separate_execution_with_its_own_acceptance():
    repeat, second = uuid4(), uuid4()
    events = (
        *BASE[:2],
        _accept(ATTEMPT),
        Submitted(TRIAL, repeat, second, None, LATER),
        Completed(second, FAILURE, LATER),
        _accept(second, reason="The repeat records the lost worker"),
    )
    executions = list(AcceptanceLog(events).executions().values())
    assert executions == [
        Execution(TRIAL, RUN, (Attempt(*BASE[:2]),), (events[2],)),
        Execution(TRIAL, repeat, (Attempt(*events[3:5]),), (events[5],)),
    ]
    assert [execution.accepted for execution in executions] == [
        Attempt(*BASE[:2]),
        Attempt(*events[3:5]),
    ]


@pytest.mark.parametrize(
    "event, message",
    [
        pytest.param(
            Submitted(TRIAL, RUN, uuid4(), None, LATER),
            "A repeat requires a new run ID; a retry needs a predecessor",
            id="second-attempt-without-predecessor",
        ),
        pytest.param(
            Submitted(TRIAL, RUN, uuid4(), OTHER, LATER),
            f"Retry predecessor {OTHER} must be a submitted attempt in run {RUN}",
            id="predecessor-from-another-run",
        ),
        pytest.param(
            Submitted(uuid4(), RUN, uuid4(), ATTEMPT, LATER),
            f"ID {RUN} already names another kind or parent",
            id="run-under-another-trial",
        ),
        pytest.param(
            Submitted(TRIAL, ATTEMPT, uuid4(), None, LATER),
            f"ID {ATTEMPT} already names another kind or parent",
            id="attempt-id-as-run-id",
        ),
        pytest.param(
            Submitted(TRIAL, RUN, ATTEMPT, None, LATER),
            f"Attempt {ATTEMPT} is already submitted differently",
            id="attempt-resubmitted-differently",
        ),
        pytest.param(
            Completed(uuid4(), SUCCESS, LATER), "is not submitted", id="unknown-completion"
        ),
        pytest.param(_accept(uuid4()), "is not submitted", id="unknown-acceptance"),
        pytest.param(
            Submitted(TRIAL, uuid4(), UUID(int=0), None, AT),
            "Invalid canonical UUID",
            id="nil-attempt-id",
        ),
        pytest.param(
            Accepted(ATTEMPT, None, " ", "Matches", LATER),
            "actor must be nonempty text",
            id="blank-actor",
        ),
        pytest.param(
            Accepted(ATTEMPT, None, "researcher", " ", LATER),
            "acceptance reason must be nonempty text",
            id="blank-reason",
        ),
        pytest.param(
            Accepted(ATTEMPT, None, "researcher", "Matches", datetime(2026, 1, 2)),
            "Timestamps must include a timezone",
            id="naive-timestamp",
        ),
        pytest.param(
            Completed(OTHER, Outcome("confirmed", "Not a development outcome"), LATER),
            "Unsupported development outcome status",
            id="unknown-status",
        ),
    ],
)
def test_invalid_or_conflicting_events_raise_and_record_nothing(event, message):
    log = AcceptanceLog(BASE)
    with pytest.raises(AcceptanceError, match=message):
        log.record(event)
    assert log.events == BASE


def test_four_thousand_retries_of_one_run_record_and_dump_within_seconds():
    attempts = 4000
    trial, run = uuid4(), uuid4()
    log = AcceptanceLog()
    predecessor = None
    start = time.process_time()
    for minute in range(attempts):
        attempt = uuid4()
        at = AT + timedelta(minutes=minute)
        log.record(Submitted(trial, run, attempt, predecessor, at))
        log.record(Completed(attempt, FAILURE, at))
        predecessor = attempt
    content = dump_log(log)
    # A worker restart replays every attempt inside one 10-second Temporal workflow task.
    # CPU time ignores how busy the shared test machine is.
    assert time.process_time() - start < 7
    assert len(json.loads(content)["events"]) == 2 * attempts


def test_log_round_trips_through_canonical_bytes():
    log = AcceptanceLog([*HISTORY, BASE[2]])
    content = dump_log(log)
    loaded = load_log(content)
    assert dump_log(loaded) == content
    assert loaded.events == log.events
    assert loaded.executions() == log.executions()
    assert [event["kind"] for event in json.loads(content)["events"]] == [
        *("submitted", "completed") * 2,
        *("accepted", "accepted", "submitted"),
    ]
    with pytest.raises(AcceptanceError, match="not the canonical encoding"):
        load_log(content.replace(b'":', b'": '))


@pytest.mark.parametrize(
    "change, message",
    [
        pytest.param(
            lambda log: log["events"].append(log["events"][1]),
            "not the canonical encoding of distinct recorded events",
            id="duplicated-event",
        ),
        pytest.param(
            lambda log: log["events"][5].update(supersedes=None),
            "a new acceptance must supersede it",
            id="two-plain-acceptances",
        ),
        pytest.param(
            lambda log: log["events"][0].update(kind="cancelled"),
            "Unknown event kind 'cancelled'",
            id="unknown-kind",
        ),
        pytest.param(
            lambda log: log.update(schema_version=2),
            "Unsupported schema version 2",
            id="schema-version",
        ),
    ],
)
def test_load_rejects_canonical_logs_that_record_would_not_produce(change, message):
    content = dump_log(AcceptanceLog(HISTORY))
    value = json.loads(content)
    assert _canonical_json(value) == content
    change(value)
    with pytest.raises(AcceptanceError, match=message):
        load_log(_canonical_json(value))
