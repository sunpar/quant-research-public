from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from uuid import UUID, uuid4

import pytest

from quant_research.access import (
    INTERNAL_CODE,
    AccessError,
    Provider,
    Restrictions,
    issue_context,
    licensed,
)
from quant_research.accounting import Usage
from quant_research.confirmation import Period, freeze, run_confirmation
from quant_research.model_provenance import UNKNOWN, ModelRecord, RoleAssignment
from quant_research.packages import Outcome
from quant_research.providers import (
    Answer,
    CallLog,
    CallRecord,
    Cancelled,
    Cleared,
    Closed,
    Dispatched,
    Failed,
    Fault,
    Gateway,
    Lease,
    Material,
    ProviderError,
    ProviderPolicy,
    RateLimited,
    Receipt,
    Request,
    ScriptedProvider,
    Served,
    Unanswered,
    dump_material,
    dump_receipts,
    dump_result,
    load_material,
    load_receipts,
    load_result,
)
from quant_research.records import PackageError

CUTOFF = datetime(2026, 10, 21, 12, tzinfo=UTC)
FAKE_A = Provider("fake-a")
FAKE_B = Provider("fake-b")
VENDOR = licensed("vendor")
WIDE = ProviderPolicy(provider=FAKE_A, labels=frozenset({INTERNAL_CODE, VENDOR}), retains=False)
NARROW = ProviderPolicy(provider=FAKE_B, labels=frozenset({INTERNAL_CODE}), retains=True)
MODEL_A = ModelRecord("fake-a", "model-a", "2026-05-01", date(2026, 1, 31))
MODEL_B = ModelRecord("fake-b", "model-b", UNKNOWN, UNKNOWN)
CONTEXT = issue_context(
    clearances=(INTERNAL_CODE, VENDOR),
    cutoff=CUTOFF,
    generation="synthetic/v1",
    providers=(FAKE_A, FAKE_B),
)
PROMPT = "Propose a momentum signal"
SCORE = (("score", float), ("count", int))
PROPOSER = RoleAssignment("proposer", MODEL_A)
FALLING_BACK = RoleAssignment("proposer", MODEL_A, (MODEL_B,))
SEALED = Usage(model_calls=1, unreported=frozenset({"input_tokens", "output_tokens", "cost_usd"}))


V3_DISPATCH = """
{
  "dispatch": {
    "attempt_id": "00000000-0000-0000-0000-000000000001",
    "call_id": "00000000-0000-0000-0000-000000000002",
    "fence": 1,
    "model": {
      "knowledge_cutoff": "2026-01-31",
      "model": "model-a",
      "provider": "fake-a",
      "version": "2026-05-01"
    },
    "policy_version": "1f6e97636b35eb4815f356976068ef7d315153b578eedaf3cde43828a72a8a33",
    "request_digest": "3cc7577dc5b82d47dc320586884f6e13c1ce021ce5b73c52f4a9fe9a0d6611ce",
    "role": "proposer",
    "runtime_version": "scripted",
    "subscription": false,
    "token": "30259d1c-0a3a-4fd0-bafd-84feda32b380"
  },
  "schema_version": 3
}
"""
V3_CALL = """
{
  "call": {
    "attempt_id": "00000000-0000-0000-0000-000000000001",
    "call_id": "00000000-0000-0000-0000-000000000002",
    "fence": 1,
    "model": {
      "knowledge_cutoff": "2026-01-31",
      "model": "model-a",
      "provider": "fake-a",
      "version": "2026-05-01"
    },
    "outcome": "completed",
    "policy_version": "1f6e97636b35eb4815f356976068ef7d315153b578eedaf3cde43828a72a8a33",
    "reasons": [],
    "reply_digest": "2b9b16b47cc5668b7a049b84452bd4d5098ded50493ef9c5edfd7d001bc27a8b",
    "request_digest": "3cc7577dc5b82d47dc320586884f6e13c1ce021ce5b73c52f4a9fe9a0d6611ce",
    "role": "proposer",
    "served": {
      "resolved_model": "model-a",
      "runtime_version": "scripted"
    },
    "usage": {
      "compute_seconds": "0",
      "cost_usd": "0.000012",
      "input_tokens": 4,
      "model_calls": 1,
      "output_tokens": 2,
      "unreported": [],
      "wall_seconds": "0"
    }
  },
  "schema_version": 3
}
"""


def material(text, *labels):
    restrictions = Restrictions(
        available_at=CUTOFF - timedelta(days=1),
        labels=frozenset(labels),
        approved_for=frozenset({FAKE_A, FAKE_B}),
    )
    return Material(text, restrictions)


def usage(sent, received):
    return Usage(
        model_calls=1,
        input_tokens=sent,
        output_tokens=received,
        cost_usd=Decimal("0.000002") * (sent + received),
    )


def gateway(directory, fake_a, fake_b=None):
    (directory / "calls").mkdir(exist_ok=True)
    routes = [(WIDE, fake_a)] if fake_b is None else [(WIDE, fake_a), (NARROW, fake_b)]
    return Gateway(CallLog(directory / "calls"), routes)


def leased(fence=1):
    return Lease(uuid4(), (uuid4(), uuid4(), uuid4()), fence)


def send(router, assignment, *materials, schema=None, cancel=None, lease=None):
    return router.call(
        CONTEXT,
        assignment,
        Request(materials, schema),
        lease=lease or leased(),
        cancel=cancel or threading.Event(),
    )


def crash_on(monkeypatch, kind):
    """Make the log raise Crash, as a dying process would, instead of storing a kind of entry."""
    append = CallLog.append

    def crash(log, entry):
        if isinstance(entry, kind):
            raise Crash
        return append(log, entry)

    monkeypatch.setattr(CallLog, "append", crash)


def named(content):
    return hashlib.sha256(content).hexdigest()


def transcripts(directory):
    return directory / "calls.transcripts"


class Crash(Exception):
    """A process dying between two writes."""


@pytest.mark.smoke
def test_content_above_a_provider_policy_never_reaches_its_adapter(tmp_path):
    fake_a, fake_b = ScriptedProvider([]), ScriptedProvider(["Looks fine"])
    router = gateway(tmp_path, fake_a, fake_b)
    reviewer = RoleAssignment("reviewer", MODEL_B)
    refused = send(router, reviewer, material("vendor excerpt", INTERNAL_CODE, VENDOR))
    assert isinstance(refused, Unanswered)
    assert [(r.outcome, r.reasons, r.usage) for r in refused.records] == [
        ("refused", ("outside_provider_policy",), Usage())
    ]
    assert fake_b.calls == []
    answered = send(router, reviewer, material("internal diff", INTERNAL_CODE))
    assert isinstance(answered, Answer) and answered.material.text == "Looks fine"
    assert [(call.model, call.texts) for call in fake_b.calls] == [(MODEL_B, ("internal diff",))]


def test_fallback_with_a_narrower_policy_is_refused_the_same_content(tmp_path):
    fake_a, fake_b = ScriptedProvider([Fault.FAIL]), ScriptedProvider(["unused"])
    router = gateway(tmp_path, fake_a, fake_b)
    result = send(router, FALLING_BACK, material("vendor excerpt", VENDOR))
    assert isinstance(result, Unanswered)
    assert [(r.model, r.outcome, r.reasons, r.usage) for r in result.records] == [
        (MODEL_A, "provider_error", (), usage(2, 0)),
        (MODEL_B, "refused", ("outside_provider_policy",), Usage()),
    ]
    assert len(fake_a.calls) == 1 and fake_b.calls == []


def test_reply_claiming_a_clearance_keeps_the_restrictions_of_its_request(tmp_path):
    claim = '{"clearance": "public"}'
    fake_a, fake_b = ScriptedProvider([claim]), ScriptedProvider(["unused"])
    router = gateway(tmp_path, fake_a, fake_b)
    excerpt = material("vendor excerpt", VENDOR)
    answer = send(
        router, RoleAssignment("summarizer", MODEL_A), excerpt, schema=(("clearance", str),)
    )
    assert isinstance(answer, Answer) and answer.values == {"clearance": "public"}
    assert answer.material == Material(claim, excerpt.restrictions)
    resent = send(router, RoleAssignment("critic", MODEL_B), answer.material)
    assert isinstance(resent, Unanswered)
    assert resent.records[0].reasons == ("outside_provider_policy",)
    assert fake_b.calls == []


@pytest.mark.parametrize(
    ("reply", "words"),
    [
        pytest.param("score 0.5 count 2", 4, id="invalid-json"),
        pytest.param("[0.5, 2]", 2, id="not-an-object"),
        pytest.param('{"score": 0.5}', 2, id="missing-key"),
        pytest.param('{"score": 0.5, "count": 2, "note": "x"}', 6, id="extra-key"),
        pytest.param('{"score": 0.5, "count": 2, "count": 3}', 6, id="duplicate-key"),
        pytest.param('{"score": 1, "count": 2}', 4, id="int-for-float"),
        pytest.param('{"score": 0.5, "count": true}', 4, id="bool-for-int"),
        pytest.param('{"score": NaN, "count": 2}', 4, id="nan"),
        pytest.param('{"score": 1e400, "count": 2}', 4, id="overflow"),
    ],
)
def test_malformed_structured_output_fails_the_call_and_records_its_usage(tmp_path, reply, words):
    router = gateway(tmp_path, ScriptedProvider([reply]))
    result = send(router, RoleAssignment("scorer", MODEL_A), material(PROMPT), schema=SCORE)
    assert isinstance(result, Unanswered)
    log = CallLog(tmp_path / "calls")
    assert [(r.outcome, r.usage, log.reply(r)) for r in result.records] == [
        ("malformed_output", usage(4, words), reply)
    ]


@pytest.mark.parametrize(
    ("script", "preset"),
    [
        pytest.param(["unused"], True, id="set-before"),
        pytest.param([Fault.CANCEL], False, id="set-during"),
    ],
)
def test_cancelled_call_is_recorded_and_ends_the_chain(tmp_path, script, preset):
    fake_a, fake_b = ScriptedProvider(script), ScriptedProvider(["unused"])
    router = gateway(tmp_path, fake_a, fake_b)
    cancel = threading.Event()
    if preset:
        cancel.set()
    result = send(router, FALLING_BACK, material(PROMPT, INTERNAL_CODE), cancel=cancel)
    assert isinstance(result, Unanswered)
    assert [(r.outcome, r.usage) for r in result.records] == [("cancelled", usage(4, 0))]
    assert cancel.is_set() and fake_b.calls == []


class FailsLate:
    """An adapter that misses the event set during its call and then reports a failure."""

    subscription = False
    runtime_version = "late"

    def complete(self, request, cancel):
        cancel.set()
        return Failed(usage(4, 0))


def test_a_failure_reported_after_the_event_is_set_ends_the_chain(tmp_path):
    fake_b = ScriptedProvider(["unused"])
    router = gateway(tmp_path, FailsLate(), fake_b)
    result = send(router, FALLING_BACK, material(PROMPT, INTERNAL_CODE))
    assert isinstance(result, Unanswered)
    assert [r.outcome for r in result.records] == ["provider_error"]
    assert fake_b.calls == []


def test_a_rate_limited_call_is_recorded_and_the_next_fallback_is_tried(tmp_path):
    fake_a, fake_b = ScriptedProvider([Fault.RATE_LIMIT]), ScriptedProvider(["Looks fine"])
    router = gateway(tmp_path, fake_a, fake_b)
    held = leased()
    answer = send(router, FALLING_BACK, material(PROMPT, INTERNAL_CODE), lease=held)
    assert isinstance(answer, Answer) and answer.material.text == "Looks fine"
    assert [(r.call_id, r.model, r.outcome, r.usage) for r in router.records] == [
        (held.dispatches[0], MODEL_A, "rate_limited", usage(4, 0)),
        (held.dispatches[1], MODEL_B, "completed", usage(4, 2)),
    ]


def test_a_subscription_records_no_cash_cost_and_its_tokens_as_unreported(tmp_path, monkeypatch):
    subscription = ScriptedProvider(["Looks fine", "Lost reply"], usd_per_token=None)
    router = gateway(tmp_path, subscription)
    tokens = frozenset({"input_tokens", "output_tokens"})
    answer = send(router, PROPOSER, material(PROMPT))
    assert isinstance(answer, Answer)
    assert answer.record.usage == Usage(
        model_calls=1, input_tokens=4, output_tokens=2, unreported=tokens
    )
    held = leased()
    crash_on(monkeypatch, CallRecord)
    with pytest.raises(Crash):
        send(router, PROPOSER, material(PROMPT), lease=held)
    monkeypatch.undo()
    [receipt] = router.close(held)
    assert (receipt.outcome, receipt.usage) == ("unknown", Usage(model_calls=1, unreported=tokens))


def test_a_claim_sealed_by_a_subscription_route_keeps_the_billing_it_was_sent_under(
    tmp_path, monkeypatch
):
    sending, held = gateway(tmp_path, ScriptedProvider(["Lost reply"])), leased()
    crash_on(monkeypatch, CallRecord)
    with pytest.raises(Crash):
        send(sending, PROPOSER, material(PROMPT), lease=held)
    monkeypatch.undo()
    resealing = gateway(tmp_path, ScriptedProvider([], usd_per_token=None))
    [receipt] = resealing.close(held)
    assert (receipt.outcome, receipt.usage) == ("unknown", SEALED)


@pytest.mark.smoke
def test_completed_call_records_its_provenance_under_the_leased_id(tmp_path):
    reply = '{"score": 0.75, "count": 20}'
    fake_a = ScriptedProvider([reply])
    router = gateway(tmp_path, fake_a)
    parts = (material(PROMPT, INTERNAL_CODE), material("vendor excerpt", VENDOR))
    held = leased()
    answer = send(router, PROPOSER, *parts, schema=SCORE, lease=held)
    assert isinstance(answer, Answer) and answer.values == {"score": 0.75, "count": 20}
    assert answer.material == material(reply, INTERNAL_CODE, VENDOR)
    terms = b'{"labels":["internal_code","licensed:vendor"],"provider":"fake-a","retains":false}\n'
    assert answer.record == CallRecord(
        held.dispatches[0],
        held.attempt_id,
        1,
        "proposer",
        MODEL_A,
        hashlib.sha256(terms).hexdigest(),
        answer.record.request_digest,
        "completed",
        (),
        usage(6, 4),
        named(json.dumps(reply).encode()),
        Served("scripted", "model-a"),
    )
    assert CallLog(tmp_path / "calls").reply(answer.record) == reply
    assert router.records == (answer.record,)
    assert [(call.texts, call.schema) for call in fake_a.calls] == [
        ((PROMPT, "vendor excerpt"), SCORE)
    ]


def test_a_resume_after_a_crash_before_the_claim_sends_the_call_once(tmp_path, monkeypatch):
    fake_a = ScriptedProvider(["Looks fine"])
    router = gateway(tmp_path, fake_a)
    held = leased()
    crash_on(monkeypatch, Dispatched)
    with pytest.raises(Crash):
        send(router, PROPOSER, material(PROMPT), lease=held)
    monkeypatch.undo()
    assert CallLog(tmp_path / "calls").load() == () and fake_a.calls == []
    answer = send(router, PROPOSER, material(PROMPT), lease=replace(held, fence=2))
    assert isinstance(answer, Answer) and len(fake_a.calls) == 1
    claim, record = CallLog(tmp_path / "calls").load()
    assert (claim.call_id, claim.fence, record) == (held.dispatches[0], 2, answer.record)


def test_a_crash_after_the_claim_is_sealed_as_unknown_and_never_resent(tmp_path, monkeypatch):
    fake_a = ScriptedProvider(["Looks fine"])
    router = gateway(tmp_path, fake_a)
    held = leased()
    crash_on(monkeypatch, CallRecord)
    with pytest.raises(Crash):
        send(router, PROPOSER, material(PROMPT), lease=held)
    monkeypatch.undo()
    (claim,) = CallLog(tmp_path / "calls").load()
    assert (claim.call_id, claim.fence) == (held.dispatches[0], 1)
    resumed = send(router, PROPOSER, material(PROMPT), lease=replace(held, fence=2))
    assert isinstance(resumed, Unanswered)
    [sealed] = resumed.records
    assert (sealed.call_id, sealed.fence, sealed.outcome, sealed.usage) == (
        claim.call_id,
        1,
        "unknown",
        SEALED,
    )
    assert (sealed.reply_digest, sealed.served) == (None, Served("scripted", None))
    assert len(fake_a.calls) == 1


def test_a_crash_after_the_record_reuses_the_reply_without_sending_again(tmp_path):
    fake_a = ScriptedProvider(['{"score": 0.75, "count": 20}'])
    router = gateway(tmp_path, fake_a)
    held = leased()
    first = send(router, PROPOSER, material(PROMPT), schema=SCORE, lease=held)
    resumed = send(router, PROPOSER, material(PROMPT), schema=SCORE, lease=replace(held, fence=2))
    assert isinstance(first, Answer) and resumed == first
    assert resumed.values == {"score": 0.75, "count": 20}
    assert len(fake_a.calls) == 1 and len(router.records) == 1


def test_a_resume_after_the_body_returned_reuses_every_record_of_the_chain(tmp_path):
    fake_a, fake_b = ScriptedProvider([Fault.FAIL]), ScriptedProvider(["Looks fine"])
    router = gateway(tmp_path, fake_a, fake_b)
    held = leased()
    first = send(router, FALLING_BACK, material(PROMPT, INTERNAL_CODE), lease=held)
    stored = CallLog(tmp_path / "calls").load()
    resumed = send(
        router, FALLING_BACK, material(PROMPT, INTERNAL_CODE), lease=replace(held, fence=2)
    )
    assert isinstance(first, Answer) and resumed == first
    assert (len(fake_a.calls), len(fake_b.calls)) == (1, 1)
    assert CallLog(tmp_path / "calls").load() == stored


def test_the_sent_transcript_is_kept_before_the_claim_and_holds_the_request(tmp_path, monkeypatch):
    fake_a = ScriptedProvider(["Looks fine"])
    router = gateway(tmp_path, fake_a)
    request = Request((material(PROMPT),))
    crash_on(monkeypatch, Dispatched)
    with pytest.raises(Crash):
        send(router, PROPOSER, material(PROMPT))
    monkeypatch.undo()
    assert CallLog(tmp_path / "calls").load() == () and fake_a.calls == []
    assert [path.name for path in transcripts(tmp_path).iterdir()] == [request.digest]
    answer = send(router, PROPOSER, material(PROMPT))
    assert answer.record.request_digest == request.digest
    assert CallLog(tmp_path / "calls").sent(answer.record) == request


def test_the_reply_transcript_is_kept_before_the_record_that_names_it(tmp_path, monkeypatch):
    router, held = gateway(tmp_path, ScriptedProvider(["Lost reply"])), leased()
    crash_on(monkeypatch, CallRecord)
    with pytest.raises(Crash):
        send(router, PROPOSER, material(PROMPT), lease=held)
    monkeypatch.undo()
    assert sorted(path.name for path in transcripts(tmp_path).iterdir()) == sorted(
        [Request((material(PROMPT),)).digest, named(b'"Lost reply"')]
    )
    [receipt] = router.close(held)
    assert (receipt.outcome, receipt.usage) == ("unknown", SEALED)


@pytest.mark.parametrize("reply", ["Looks fine", "café \ud800", "", "two\nlines"])
def test_any_reply_text_is_kept_and_read_back_from_its_transcript(tmp_path, reply):
    router = gateway(tmp_path, ScriptedProvider([reply]))
    answer = send(router, PROPOSER, material(PROMPT))
    assert isinstance(answer, Answer) and answer.material.text == reply
    content = (transcripts(tmp_path) / answer.record.reply_digest).read_bytes()
    assert named(content) == answer.record.reply_digest and json.loads(content) == reply


def test_a_refused_call_stores_no_transcript(tmp_path):
    router = gateway(tmp_path, ScriptedProvider([]), ScriptedProvider([]))
    refused = send(router, RoleAssignment("reviewer", MODEL_B), material("vendor excerpt", VENDOR))
    [record] = refused.records
    assert (record.outcome, record.reply_digest, record.served) == ("refused", None, None)
    assert not transcripts(tmp_path).exists()


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("missing", "Cannot read transcript"),
        ("other-bytes", "holds other bytes"),
        ("truncated", "holds other bytes"),
    ],
)
def test_a_reply_whose_transcript_is_missing_or_damaged_is_never_sent_again(
    tmp_path, damage, message
):
    fake_a = ScriptedProvider(["Looks fine", "unused"])
    router, held = gateway(tmp_path, fake_a), leased()
    first = send(router, PROPOSER, material(PROMPT), lease=held)
    assert send(router, PROPOSER, material(PROMPT), lease=replace(held, fence=2)) == first
    path = transcripts(tmp_path) / first.record.reply_digest
    match damage:
        case "missing":
            path.unlink()
        case "other-bytes":
            path.write_bytes(b'"Looks fine!"')
        case "truncated":
            path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ProviderError, match=message):
        send(router, PROPOSER, material(PROMPT), lease=replace(held, fence=3))
    with pytest.raises(ProviderError, match=message):
        CallLog(tmp_path / "calls").reply(first.record)
    assert len(fake_a.calls) == 1 and len(router.records) == 1


def test_a_transcript_name_holding_other_bytes_stops_the_call_before_its_claim(tmp_path):
    fake_a = ScriptedProvider(["unused"])
    router = gateway(tmp_path, fake_a)
    transcripts(tmp_path).mkdir()
    (transcripts(tmp_path) / Request((material(PROMPT),)).digest).write_bytes(b"other")
    with pytest.raises(ProviderError, match="holds other bytes"):
        send(router, PROPOSER, material(PROMPT))
    assert CallLog(tmp_path / "calls").load() == () and fake_a.calls == []


def test_each_record_says_what_served_it_and_a_refused_one_names_nothing(tmp_path, monkeypatch):
    steps = [
        (Fault.FAIL, None),
        ("not json", SCORE),
        (Fault.RATE_LIMIT, None),
        (Fault.CANCEL, None),
        ("Looks fine", None),
        ("Lost reply", None),
    ]
    router = gateway(tmp_path, ScriptedProvider([step for step, _ in steps]), ScriptedProvider([]))
    for _, schema in steps[:-1]:
        send(router, PROPOSER, material(PROMPT), schema=schema)
    held = leased()
    crash_on(monkeypatch, CallRecord)
    with pytest.raises(Crash):
        send(router, PROPOSER, material(PROMPT), lease=held)
    monkeypatch.undo()
    router.close(held)
    send(router, RoleAssignment("reviewer", MODEL_B), material("vendor excerpt", VENDOR))
    served = Served("scripted", "model-a")
    assert [(r.outcome, r.served) for r in router.records] == [
        ("provider_error", served),
        ("malformed_output", served),
        ("rate_limited", served),
        ("cancelled", served),
        ("completed", served),
        ("unknown", Served("scripted", None)),
        ("refused", None),
    ]


def test_a_blocked_call_holds_its_claim_until_the_event_is_set_and_ends_the_chain(tmp_path):
    fake_a, fake_b = ScriptedProvider([Fault.BLOCK]), ScriptedProvider(["unused"])
    router, cancel = gateway(tmp_path, fake_a, fake_b), threading.Event()
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(send, router, FALLING_BACK, material(PROMPT), cancel=cancel)
        assert fake_a.blocking.wait(5)
        [claim] = CallLog(tmp_path / "calls").load()
        assert isinstance(claim, Dispatched) and not pending.done()
        cancel.set()
        result = pending.result(5)
    assert isinstance(result, Unanswered)
    assert [(r.outcome, r.usage) for r in result.records] == [("cancelled", usage(4, 0))]
    assert fake_b.calls == []


def test_reuse_is_refused_when_the_request_differs_from_the_stored_one(tmp_path):
    fake_a = ScriptedProvider(["Looks fine", "unused"])
    router = gateway(tmp_path, fake_a)
    held = leased()
    send(router, PROPOSER, material(PROMPT), lease=held)
    with pytest.raises(ProviderError, match="another request"):
        send(router, PROPOSER, material("A different prompt"), lease=replace(held, fence=2))
    with pytest.raises(ProviderError, match="another request"):
        send(router, PROPOSER, material(PROMPT), schema=SCORE, lease=replace(held, fence=2))
    stored = material(PROMPT).restrictions
    for restrictions in (
        replace(stored, labels=frozenset({VENDOR})),
        replace(stored, approved_for=frozenset({FAKE_A})),
        replace(stored, available_at=CUTOFF - timedelta(days=2)),
    ):
        with pytest.raises(ProviderError, match="another request"):
            send(router, PROPOSER, Material(PROMPT, restrictions), lease=replace(held, fence=2))
    assert len(fake_a.calls) == 1


def test_a_straggler_that_meets_a_later_claim_stops_before_sending(tmp_path, monkeypatch):
    fake_a = ScriptedProvider(["Looks fine"])
    router = gateway(tmp_path, fake_a)
    held = leased()
    complete = fake_a.complete

    def straggler_wakes_during_the_resume(request, cancel):
        with pytest.raises(ProviderError, match="Activity 2 holds the lease"):
            send(router, PROPOSER, material(PROMPT), lease=held)
        return complete(request, cancel)

    monkeypatch.setattr(fake_a, "complete", straggler_wakes_during_the_resume)
    answer = send(router, PROPOSER, material(PROMPT), lease=replace(held, fence=2))
    assert isinstance(answer, Answer) and len(fake_a.calls) == 1
    assert send(router, PROPOSER, material(PROMPT), lease=held) == answer


def test_a_straggler_reply_after_a_resume_sealed_its_claim_is_refused(tmp_path, monkeypatch):
    fake_a = ScriptedProvider(["Looks fine"])
    router = gateway(tmp_path, fake_a)
    held = leased()
    complete = fake_a.complete
    resumed = []

    def resume_seals_during_the_send(request, cancel):
        resumed.append(send(router, PROPOSER, material(PROMPT), lease=replace(held, fence=2)))
        return complete(request, cancel)

    monkeypatch.setattr(fake_a, "complete", resume_seals_during_the_send)
    straggler = send(router, PROPOSER, material(PROMPT), lease=held)
    assert straggler == resumed[0] == Unanswered(router.records)
    [sealed] = router.records
    assert (sealed.outcome, sealed.fence, sealed.usage) == ("unknown", 1, SEALED)
    assert len(fake_a.calls) == 1


def test_a_straggler_after_close_keeps_its_records_and_sends_no_new_call(tmp_path, monkeypatch):
    fake_a, fake_b = ScriptedProvider(["Looks fine"]), ScriptedProvider(["unused"])
    router = gateway(tmp_path, fake_a, fake_b)
    held = leased()
    complete = fake_a.complete
    closed = []

    def the_lease_closes_during_the_send(request, cancel):
        closed.append(router.close(held))
        return complete(request, cancel)

    monkeypatch.setattr(fake_a, "complete", the_lease_closes_during_the_send)
    with pytest.raises(ProviderError, match="is closed"):
        send(router, FALLING_BACK, material(PROMPT, INTERNAL_CODE), lease=held)
    assert fake_b.calls == []
    [sealed] = router.records
    assert closed == [(Receipt(held.dispatches[0], "proposer", "unknown", SEALED),)]
    assert router.close(held) == closed[0]


def test_close_seals_each_unanswered_claim_and_returns_receipts_in_chain_order(
    tmp_path, monkeypatch
):
    fake_a, fake_b = ScriptedProvider([Fault.FAIL]), ScriptedProvider(["unused"])
    router = gateway(tmp_path, fake_a, fake_b)
    held = leased()
    append = CallLog.append

    def crash_before_the_fallback_reply(log, entry):
        if isinstance(entry, CallRecord) and entry.model == MODEL_B:
            raise Crash
        return append(log, entry)

    monkeypatch.setattr(CallLog, "append", crash_before_the_fallback_reply)
    with pytest.raises(Crash):
        send(router, FALLING_BACK, material(PROMPT, INTERNAL_CODE), lease=held)
    monkeypatch.undo()
    receipts = router.close(held)
    assert receipts == (
        Receipt(held.dispatches[0], "proposer", "provider_error", usage(4, 0)),
        Receipt(held.dispatches[1], "proposer", "unknown", SEALED),
    )
    assert CallLog(tmp_path / "calls").load()[-2:] == (Closed(held.attempt_id), router.records[-1])
    files = sorted((tmp_path / "calls").iterdir())
    assert router.close(held) == receipts
    assert sorted((tmp_path / "calls").iterdir()) == files
    assert load_receipts(dump_receipts(receipts)) == receipts
    refused = leased()
    send(
        router, RoleAssignment("critic", MODEL_B), material("vendor excerpt", VENDOR), lease=refused
    )
    assert router.close(refused) == ()


@pytest.mark.parametrize("order", ["loser-before-the-reply", "loser-after-the-reply"])
def test_two_holders_of_one_fence_that_read_before_either_writes_send_once(
    tmp_path, monkeypatch, order
):
    fake_a = ScriptedProvider(["Looks fine"])
    routers = {name: gateway(tmp_path, fake_a) for name in ("winner", "loser")}
    held = leased()
    both_read = threading.Barrier(2, timeout=5)
    claimed, done = threading.Event(), {name: threading.Event() for name in routers}
    append, complete = CallLog.append, fake_a.complete

    def interleaved(log, entry):
        if isinstance(entry, Dispatched):
            both_read.wait()
            if threading.current_thread().name == "loser":
                (claimed if order == "loser-before-the-reply" else done["winner"]).wait(5)
        stored = append(log, entry)
        if isinstance(entry, Dispatched) and threading.current_thread().name == "winner":
            claimed.set()
        return stored

    def reply_after_the_loser(request, cancel):
        if order == "loser-before-the-reply":
            done["loser"].wait(5)
        return complete(request, cancel)

    monkeypatch.setattr(CallLog, "append", interleaved)
    monkeypatch.setattr(fake_a, "complete", reply_after_the_loser)
    results = {}

    def hold(name):
        try:
            results[name] = send(routers[name], PROPOSER, material(PROMPT), lease=held)
        except ProviderError as error:
            results[name] = error
        finally:
            done[name].set()

    threads = [threading.Thread(target=hold, args=(name,), name=name) for name in routers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert len(fake_a.calls) == 1
    assert isinstance(results["winner"], Answer)
    if order == "loser-before-the-reply":
        assert "already being sent" in str(results["loser"])
    else:
        assert results["loser"] == results["winner"]
    claims = [e for e in CallLog(tmp_path / "calls").load() if isinstance(e, Dispatched)]
    assert len(claims) == 1


def test_a_chain_longer_than_its_lease_raises_before_any_adapter_runs(tmp_path):
    fake_a, fake_b = ScriptedProvider(["unused"]), ScriptedProvider(["unused"])
    router = gateway(tmp_path, fake_a, fake_b)
    short = Lease(uuid4(), (uuid4(),), 1)
    with pytest.raises(ProviderError, match="needs as many call IDs"):
        send(router, FALLING_BACK, material(PROMPT), lease=short)
    assert fake_a.calls == [] and router.records == ()


def logged_calls(directory):
    fake_a = ScriptedProvider([Fault.FAIL, "A summary of the diff", Fault.CANCEL])
    fake_b = ScriptedProvider([])
    router = gateway(directory, fake_a, fake_b)
    vendor = material("vendor excerpt", VENDOR)
    send(router, RoleAssignment("proposer", MODEL_B, (MODEL_A,)), vendor)
    send(router, RoleAssignment("summarizer", MODEL_A, (MODEL_B,)), vendor)
    send(router, RoleAssignment("critic", MODEL_A), material(PROMPT, INTERNAL_CODE))
    return router.records


CLAIMED = "call_id attempt_id fence role model policy_version request_digest".split()


def test_reopened_log_reproduces_every_entry_and_its_bytes(tmp_path):
    records = logged_calls(tmp_path)
    assert [r.outcome for r in records] == [
        "refused",
        "provider_error",
        "completed",
        "cancelled",
    ]
    files = [tmp_path / "calls" / f"{index}.json" for index in range(7)]
    contents = [path.read_bytes() for path in files]
    log = CallLog(tmp_path / "calls")
    entries = log.load()
    refused, failed, completed, cancelled = records
    assert entries[::2] == records
    for claim, record in zip(entries[1::2], (failed, completed, cancelled), strict=True):
        assert isinstance(claim, Dispatched)
        assert [getattr(claim, name) for name in CLAIMED] == [
            getattr(record, name) for name in CLAIMED
        ]
    assert [log.append(entry) for entry in entries] == list(entries)
    assert sorted((tmp_path / "calls").iterdir()) == files
    assert [path.read_bytes() for path in files] == contents
    stored = [json.loads(content) for content in contents]
    assert stored[1]["dispatch"]["call_id"] == str(failed.call_id)
    assert stored[0]["call"]["model"] == {
        "provider": "fake-b",
        "model": "model-b",
        "version": None,
        "knowledge_cutoff": None,
    }
    assert stored[4]["call"]["usage"]["cost_usd"] == "0.000014"
    assert stored[4]["call"]["reply_digest"] == named(b'"A summary of the diff"')
    assert stored[4]["call"]["served"] == {
        "runtime_version": "scripted",
        "resolved_model": "model-a",
    }
    assert stored[0]["call"]["served"] is None


def claim_of(call_id, attempt_id=None, fence=1):
    return Dispatched(
        call_id,
        attempt_id or uuid4(),
        fence,
        "proposer",
        MODEL_A,
        WIDE.version,
        "d" * 64,
        False,
        "scripted",
        uuid4(),
    )


def test_racing_logs_store_each_entry_once_and_one_claim_per_call(tmp_path):
    logs = [CallLog(tmp_path) for _ in range(8)]
    distinct = [claim_of(uuid4()) for _ in logs]
    contested = uuid4()
    rivals = [claim_of(contested, fence=index) for index, _ in enumerate(logs)]
    start = threading.Barrier(len(logs))

    def append(log, claim, rival):
        start.wait()
        stored = (log.append(claim), log.append(claim), log.append(rival))
        return stored

    with ThreadPoolExecutor(len(logs)) as pool:
        results = list(pool.map(append, logs, distinct, rivals))
    assert [(first, again) for first, again, _ in results] == [(c, c) for c in distinct]
    [winner] = {won for _, _, won in results}
    assert winner in rivals
    stored = CallLog(tmp_path).load()
    assert sorted(entry.call_id for entry in stored) == sorted(
        [claim.call_id for claim in distinct] + [contested]
    )


def old_log(directory, outcome="completed", *, subscription=False, closed=False):
    calls = directory / "calls"
    calls.mkdir()
    dispatch, call = json.loads(V3_DISPATCH), json.loads(V3_CALL)
    dispatch["dispatch"]["subscription"] = subscription
    record = call["call"]
    record["outcome"] = outcome
    if outcome != "completed":
        record["reply_digest"] = None
        record["usage"].update(output_tokens=0, cost_usd="0.000008")
    if outcome == "unknown":
        record["served"]["resolved_model"] = None
        record["usage"].update(
            input_tokens=0,
            cost_usd="0",
            unreported=["input_tokens", "output_tokens"]
            if subscription
            else ["cost_usd", "input_tokens", "output_tokens"],
        )
    entries = [dispatch] if outcome is None else [dispatch, call]
    if closed:
        entries.append({"schema_version": 3, "close": {"attempt_id": record["attempt_id"]}})
    for index, entry in enumerate(entries):
        (calls / f"{index}.json").write_bytes(canonical(entry))
    request = Request((material(PROMPT),))
    assert request.digest == record["request_digest"]
    transcripts(directory).mkdir()
    (transcripts(directory) / request.digest).write_bytes(request.to_bytes())
    if outcome == "completed":
        (transcripts(directory) / record["reply_digest"]).write_bytes(b'"Looks fine"')
    held = Lease(UUID(record["attempt_id"]), (UUID(record["call_id"]),), 2)
    return held, request, record


@pytest.mark.parametrize("outcome", ["completed", "provider_error", "cancelled", "rate_limited"])
def test_v3_records_and_activity_results_resume_without_rewriting_or_resending(tmp_path, outcome):
    held, request, raw = old_log(tmp_path, outcome)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    text = "Looks fine" if outcome == "completed" else None
    historical = json.dumps({"records": [raw], "text": text}, sort_keys=True)
    expected = load_result(historical, request)
    fake = ScriptedProvider(["new call"])
    router = gateway(tmp_path, fake)
    result = send(router, PROPOSER, *request.materials, lease=held)
    assert result == expected
    assert router.records[0].reason is None
    assert fake.calls == []
    assert all(path.read_bytes() == content for path, content in before.items())
    (receipt,) = router.close(held)
    assert set(json.loads(dump_receipts((receipt,)))[0]) == {"call_id", "role", "outcome", "usage"}
    send(router, PROPOSER, material("A fresh question"))
    assert len(fake.calls) == 1
    assert [
        json.loads((tmp_path / "calls" / f"{i}.json").read_bytes())["schema_version"]
        for i in range(5)
    ] == [3, 3, 4, 4, 4]
    assert gateway(tmp_path, ScriptedProvider([])).records == router.records
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("subscription", [False, True])
@pytest.mark.parametrize("closed", [False, True])
def test_closing_a_v3_claim_seals_once_using_the_original_subscription(
    tmp_path, subscription, closed
):
    held, _, _ = old_log(tmp_path, None, subscription=subscription, closed=closed)
    before = {path: path.read_bytes() for path in (tmp_path / "calls").iterdir()}
    fake = ScriptedProvider([], usd_per_token=Decimal(1) if subscription else None)
    router = gateway(tmp_path, fake)
    receipts = router.close(held)
    [record] = router.records
    assert record.outcome == "unknown" and record.reason is None
    assert ("cost_usd" not in record.usage.unreported) == subscription
    assert record.served == Served("scripted", None)
    sealed = {path: path.read_bytes() for path in (tmp_path / "calls").iterdir()}
    assert gateway(tmp_path, fake).close(replace(held, fence=3)) == receipts
    assert sealed == {path: path.read_bytes() for path in (tmp_path / "calls").iterdir()}
    assert all(path.read_bytes() == content for path, content in before.items())
    assert fake.calls == []


@pytest.mark.parametrize("subscription", [False, True])
def test_v3_unknown_seals_require_exact_original_billing(tmp_path, subscription):
    held, _, _ = old_log(tmp_path, "unknown", subscription=subscription, closed=True)
    router = gateway(tmp_path, ScriptedProvider([]))
    assert router.close(held)[0].outcome == "unknown"
    assert router.records[0].reason is None
    path = tmp_path / "calls" / "1.json"
    forged = (
        ["cost_usd", "input_tokens", "output_tokens"]
        if subscription
        else ["input_tokens", "output_tokens"]
    )
    rewrite(path, lambda call: call["usage"].update(unreported=forged))
    with pytest.raises(ProviderError, match="sealed unlike its claim"):
        CallLog(tmp_path / "calls")


def test_corrupt_v3_reply_refuses_resume_without_resending(tmp_path):
    held, request, raw = old_log(tmp_path)
    (transcripts(tmp_path) / raw["reply_digest"]).write_bytes(b"corrupt")
    fake = ScriptedProvider(["must not send"])
    with pytest.raises(ProviderError, match="holds other bytes"):
        send(gateway(tmp_path, fake), PROPOSER, *request.materials, lease=held)
    assert fake.calls == []


@pytest.mark.parametrize(
    ("version", "reason", "extra"),
    [(3, None, False), (4, "missing", False), (3, "missing", True), (4, None, True)],
)
def test_log_versions_require_their_exact_call_fields(tmp_path, version, reason, extra):
    old_log(tmp_path)
    path = tmp_path / "calls" / "1.json"
    value = json.loads(path.read_bytes())
    value["schema_version"] = version
    if reason != "missing":
        value["call"]["reason"] = reason
    if extra:
        value["call"]["unexpected"] = None
    path.write_bytes(canonical(value))
    with pytest.raises(ProviderError, match="Invalid fields"):
        CallLog(tmp_path / "calls")


@pytest.mark.parametrize("reason", ["secret-sentinel", ["secret-sentinel"], 42])
def test_persistence_rejects_unsupported_failure_reasons_without_storing_them(tmp_path, reason):
    old_log(tmp_path, "provider_error")
    log = CallLog(tmp_path / "calls")
    record = log.load()[1]
    object.__setattr__(record, "reason", reason)
    before = {path: path.read_bytes() for path in (tmp_path / "calls").iterdir()}
    with pytest.raises(ProviderError, match="Unsupported failure reason") as caught:
        log.append(record)
    assert "secret-sentinel" not in str(caught.value)
    assert before == {path: path.read_bytes() for path in (tmp_path / "calls").iterdir()}


@pytest.mark.parametrize("outcome", ["completed", "unknown"])
def test_an_answer_or_unknown_seal_cannot_acquire_a_failure_reason(tmp_path, outcome):
    _, request, raw = old_log(tmp_path, outcome)
    value = json.loads((tmp_path / "calls" / "1.json").read_bytes())
    value["schema_version"] = 4
    value["call"]["reason"] = "provider_error"
    (tmp_path / "calls" / "1.json").write_bytes(canonical(value))
    with pytest.raises(ProviderError, match="Only failed"):
        CallLog(tmp_path / "calls")
    with pytest.raises(ProviderError, match="Only failed"):
        load_result(
            json.dumps({"text": None, "records": [raw | {"reason": "provider_error"}]}), request
        )


@pytest.mark.parametrize("reason", [False, True])
def test_activity_results_accept_only_the_exact_legacy_or_current_record_shape(tmp_path, reason):
    _, request, raw = old_log(tmp_path, "provider_error")
    if reason:
        raw["reason"] = "deadline"
    raw["diagnostic"] = None
    with pytest.raises(PackageError, match="Invalid fields"):
        load_result(json.dumps({"text": None, "records": [raw]}), request)


@pytest.mark.parametrize("kind", [Failed, Cancelled, RateLimited])
def test_unsuccessful_replies_keep_positional_model_and_default_absent_reason(kind):
    reply = kind(Usage(input_tokens=2), "served-model")
    assert reply.resolved_model == "served-model" and reply.reason is None
    assert kind(reply.usage, reply.resolved_model, "deadline").reason == "deadline"


def canonical(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{text}\n".encode()


def rewrite(path, change):
    entry = json.loads(path.read_bytes())
    change(entry["call"])
    path.write_bytes(canonical(entry))


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("indent", "not canonical"),
        ("no-final-newline", "not canonical"),
        ("version-2", "schema_version 3"),
        ("version-text", "schema_version 3"),
        ("gap", "without gaps"),
        ("stray-file", "without gaps"),
        ("decimal-not-rendered-canonically", "Invalid decimal"),
        ("repeated-entry", "repeats an entry"),
        ("repeated-call-id", "conflicts with an entry"),
        ("unknown-reason", "Unsupported reasons"),
        ("refused-with-usage", "no usage"),
        ("conflicting-claim", "conflicts with an entry"),
        ("record-unlike-its-claim", "does not match its claim"),
        ("sent-record-without-a-claim", "does not match its claim"),
        ("refused-record-of-a-claimed-call", "refused after its claim"),
        ("claim-after-its-record", "conflicts with an entry"),
        ("claim-after-its-attempt-closed", "claimed after its attempt closed"),
        ("reply-on-an-unreplied-call", "names a reply exactly when"),
        ("reply-not-a-digest", "Invalid reply digest"),
        ("served-on-a-refused-call", "served it exactly when"),
        ("sent-without-served", "served it exactly when"),
        ("unknown-with-a-resolved-model", "sealed unlike its claim"),
        ("unknown-with-forged-usage", "sealed unlike its claim"),
        ("runtime-unlike-its-claim", "does not match its claim"),
        ("blank-runtime", "runtime version must be nonempty"),
        ("negative-fence", "fence"),
        ("subscription-text", "subscription is true or false"),
    ],
)
def test_damaged_log_fails_to_open(tmp_path, case, message):
    logged_calls(tmp_path)
    refused, failed, completed, cancelled = (tmp_path / "calls" / f"{i}.json" for i in (0, 2, 4, 6))
    later, last = tmp_path / "calls" / "7.json", tmp_path / "calls" / "8.json"
    stored = json.loads(refused.read_bytes())
    claim = json.loads((tmp_path / "calls" / "1.json").read_bytes())
    match case:
        case "indent":
            refused.write_text(json.dumps(stored, indent=2))
        case "no-final-newline":
            refused.write_bytes(refused.read_bytes().rstrip(b"\n"))
        case "version-2":
            refused.write_bytes(canonical(stored | {"schema_version": 2}))
        case "version-text":
            refused.write_bytes(canonical(stored | {"schema_version": "3"}))
        case "gap":
            cancelled.rename(later)
        case "stray-file":
            (tmp_path / "calls" / "notes.txt").write_text("")
        case "decimal-not-rendered-canonically":
            rewrite(completed, lambda call: call["usage"].update(cost_usd="1.4E-5"))
        case "repeated-entry":
            later.write_bytes(failed.read_bytes())
        case "repeated-call-id":
            rewrite(failed, lambda call: call.update(call_id=stored["call"]["call_id"]))
        case "unknown-reason":
            rewrite(refused, lambda call: call.update(reasons=["uncleared", "ignored"]))
        case "refused-with-usage":
            usage = json.loads(failed.read_bytes())["call"]["usage"]
            rewrite(refused, lambda call: call.update(usage=usage))
        case "conflicting-claim":
            claim["dispatch"]["role"] = "critic"
            later.write_bytes(canonical(claim))
        case "record-unlike-its-claim":
            rewrite(failed, lambda call: call.update(model=stored["call"]["model"]))
        case "sent-record-without-a-claim":
            rewrite(failed, lambda call: call.update(call_id=str(uuid4())))
        case "refused-record-of-a-claimed-call":
            refusal = {
                name: stored["call"][name] for name in ("outcome", "reasons", "usage", "served")
            }
            rewrite(failed, lambda call: call.update(refusal))
        case "claim-after-its-record":
            claim["dispatch"]["call_id"] = stored["call"]["call_id"]
            later.write_bytes(canonical(claim))
        case "claim-after-its-attempt-closed":
            attempt = claim["dispatch"]["attempt_id"]
            later.write_bytes(canonical({"schema_version": 3, "close": {"attempt_id": attempt}}))
            claim["dispatch"]["call_id"] = str(uuid4())
            last.write_bytes(canonical(claim))
        case "reply-on-an-unreplied-call":
            rewrite(failed, lambda call: call.update(reply_digest="e" * 64))
        case "reply-not-a-digest":
            rewrite(completed, lambda call: call.update(reply_digest=7))
        case "served-on-a-refused-call":
            served = json.loads(failed.read_bytes())["call"]["served"]
            rewrite(refused, lambda call: call.update(served=served))
        case "sent-without-served":
            rewrite(failed, lambda call: call.update(served=None))
        case "unknown-with-a-resolved-model":
            rewrite(failed, lambda call: call.update(outcome="unknown"))
        case "unknown-with-forged-usage":
            served = {"resolved_model": None, "runtime_version": "scripted"}
            rewrite(failed, lambda call: call.update(outcome="unknown", served=served))
        case "runtime-unlike-its-claim":
            rewrite(failed, lambda call: call["served"].update(runtime_version="other"))
        case "blank-runtime":
            claim["dispatch"]["runtime_version"] = " "
            (tmp_path / "calls" / "1.json").write_bytes(canonical(claim))
        case "negative-fence":
            rewrite(refused, lambda call: call.update(fence=-1))
        case "subscription-text":
            claim["dispatch"]["subscription"] = "false"
            (tmp_path / "calls" / "1.json").write_bytes(canonical(claim))
    with pytest.raises(ProviderError, match=message):
        CallLog(tmp_path / "calls")


def test_unrouted_model_anywhere_in_the_chain_raises_before_any_adapter_runs(tmp_path):
    fake_a = ScriptedProvider(["unused"])
    router = gateway(tmp_path, fake_a)
    with pytest.raises(ProviderError):
        send(router, FALLING_BACK, material(PROMPT))
    assert fake_a.calls == [] and router.records == ()


def test_blank_role_raises_before_any_adapter_runs(tmp_path):
    fake_a = ScriptedProvider(["unused"])
    router = gateway(tmp_path, fake_a)
    with pytest.raises(PackageError):
        send(router, RoleAssignment(" ", MODEL_A), material(PROMPT))
    assert fake_a.calls == [] and router.records == ()


def test_role_the_log_cannot_store_raises_before_any_adapter_runs(tmp_path):
    fake_a = ScriptedProvider(["unused"])
    router = gateway(tmp_path, fake_a)
    with pytest.raises(ProviderError, match="cannot store"):
        send(router, RoleAssignment("\ud800", MODEL_A), material(PROMPT))
    assert fake_a.calls == [] and router.records == ()


@pytest.mark.parametrize(
    "smuggle",
    [
        lambda report: Dispatched(
            uuid4(),
            uuid4(),
            1,
            "proposer",
            ModelRecord("fake-a", "model-a", "v2", report),
            "0" * 64,
            "d" * 64,
            False,
            "scripted",
            uuid4(),
        ),
        lambda report: CallRecord(
            uuid4(),
            uuid4(),
            1,
            "proposer",
            MODEL_A,
            WIDE.version,
            "d" * 64,
            report,
            (),
            Usage(),
            None,
            Served("scripted", None),
        ),
        lambda report: CallRecord(
            uuid4(),
            uuid4(),
            1,
            "proposer",
            MODEL_A,
            WIDE.version,
            "d" * 64,
            "provider_error",
            (),
            {**asdict(Usage(model_calls=1)), "cost_usd": report},
            None,
            Served("scripted", None),
        ),
    ],
    ids=["model-cutoff", "outcome", "usage-cost"],
)
def test_log_refuses_a_report_it_could_not_reopen_before_writing_it(
    tmp_path, smuggle, reachable_text
):
    candidate = "1" * 64
    report = run_confirmation(
        freeze((candidate,), "2" * 64, "3" * 64, "4" * 64, "5" * 64),
        Period("synthetic/v1", CUTOFF - timedelta(days=30), CUTOFF),
        (),
        lambda *_: {candidate: Outcome("successful", "Restricted detail: candidate held")},
        "researcher",
        CUTOFF,
    )
    with pytest.raises(ProviderError) as caught:
        CallLog(tmp_path).append(smuggle(report))
    assert list(tmp_path.iterdir()) == []
    assert "Restricted detail" not in reachable_text(caught.value)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: Request(()), id="no-materials"),
        pytest.param(lambda: Request((material(PROMPT),), ()), id="empty-schema"),
        pytest.param(lambda: Request((material(PROMPT),), (("", str),)), id="unnamed-field"),
        pytest.param(lambda: Request((material(PROMPT),), SCORE[:1] * 2), id="repeated-field"),
        pytest.param(lambda: Request((material(PROMPT),), (("items", list),)), id="list-field"),
        pytest.param(lambda: Cleared(MODEL_A, (PROMPT,), None), id="uncleared-content"),
    ],
)
def test_invalid_configuration_raises(build):
    with pytest.raises(ProviderError):
        build()


FIRST = Restrictions(
    available_at=CUTOFF - timedelta(days=1, microseconds=250),
    labels=frozenset({INTERNAL_CODE}),
    approved_for=frozenset({FAKE_A, FAKE_B}),
)
SECOND = Restrictions(
    available_at=datetime(2026, 10, 20, 9, 30, 0, 123456, tzinfo=timezone(timedelta(hours=-4))),
    labels=frozenset({VENDOR}),
    approved_for=frozenset({FAKE_A}),
)


@pytest.mark.parametrize(
    ("sent", "digest"),
    [
        pytest.param(
            Request(
                (Material(PROMPT, FIRST), Material("vendor excerpt", SECOND)),
                (("s", str), ("i", int), ("f", float), ("b", bool)),
            ),
            "e13856b7896296f215afaeee1fd2a92307f9a8384c8711277340ab4ba3b2b8a7",
            id="mixed-restrictions-every-kind",
        ),
        pytest.param(
            Request((Material("café \ud800", FIRST),)),
            "197d942897a993f66a6fab55b58951c0d2fe73cb50bda5419875de5f56beace4",
            id="lone-surrogate",
        ),
    ],
)
def test_a_request_round_trips_through_its_bytes_under_the_digest_main_stored(sent, digest):
    decoded = Request.from_bytes(sent.to_bytes())
    assert sent.digest == decoded.digest == digest
    assert decoded.to_bytes() == sent.to_bytes() and decoded.schema == sent.schema
    assert [m.text for m in decoded.materials] == [m.text for m in sent.materials]
    assert {m.restrictions for m in decoded.materials} == {sent.restrictions}


CANONICAL = Request((material(PROMPT, INTERNAL_CODE),), SCORE).to_bytes()


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b"{", id="not-json"),
        pytest.param(CANONICAL.replace(b'"texts": ', b'"texts":'), id="spacing"),
        pytest.param(CANONICAL[:-1] + b', "extra": 1}', id="extra-key"),
        pytest.param(CANONICAL[:-1] + b', "texts": []}', id="repeated-key"),
        pytest.param(CANONICAL.replace(b'"texts": [', b'"texts": [5, '), id="non-text"),
        pytest.param(CANONICAL.replace(b'["internal_code"]', b"[1]"), id="non-text-label"),
        pytest.param(CANONICAL.replace(b'"int"', b'"list"'), id="unknown-kind"),
        pytest.param(CANONICAL.replace(b'["count", "int"]', b'["count"]'), id="short-field"),
        pytest.param(CANONICAL.replace(b'00Z"', b'00+00:00"'), id="noncanonical-time"),
        pytest.param(
            CANONICAL.replace(b'["internal_code"]', b'["internal_code", "internal_code"]'),
            id="repeated-label",
        ),
        pytest.param(
            CANONICAL.replace(b'["fake-a", "fake-b"]', b'["fake-b", "fake-a"]'),
            id="unsorted-approvals",
        ),
        pytest.param(CANONICAL.replace(b"momentum", "mómentum".encode()), id="non-ascii"),
        pytest.param(CANONICAL.replace(b'"texts": [', b'"texts": [[], '), id="nested-text"),
    ],
)
def test_bytes_that_to_bytes_would_not_write_are_refused_without_their_content(
    content, reachable_text
):
    with pytest.raises(ProviderError) as caught:
        Request.from_bytes(content)
    assert "mentum" not in reachable_text(caught.value)


def test_a_result_crosses_as_text_and_loads_only_for_its_request(tmp_path):
    fake_a = ScriptedProvider(['{"score": 0.75, "count": 20}', "café \ud800", Fault.FAIL])
    router = gateway(tmp_path, fake_a, ScriptedProvider([Fault.RATE_LIMIT]))
    scored = Request((material(PROMPT, INTERNAL_CODE),), SCORE)
    free = Request((material(PROMPT, INTERNAL_CODE),))
    results = [
        (router.call(CONTEXT, assignment, sent, lease=leased(), cancel=threading.Event()), sent)
        for assignment, sent in ((PROPOSER, scored), (PROPOSER, free), (FALLING_BACK, free))
    ]
    (answer, _), (surrogate, _), (unanswered, _) = results
    assert isinstance(answer, Answer) and isinstance(surrogate, Answer)
    assert [r.outcome for r in unanswered.records] == ["provider_error", "rate_limited"]
    for result, sent in results:
        assert load_result(dump_result(result), sent) == result
    with pytest.raises(ProviderError, match="another request"):
        load_result(dump_result(answer), free)
    with pytest.raises(ProviderError, match="another request"):
        load_result(dump_result(unanswered), scored)
    swapped = replace(answer, material=Material('{"score": 1.5, "count": 20}', scored.restrictions))
    with pytest.raises(ProviderError, match="record naming it"):
        load_result(dump_result(swapped), scored)


def test_a_material_crosses_as_text_with_its_restrictions():
    item = Material("café \ud800", replace(SECOND, labels=frozenset({VENDOR, INTERNAL_CODE})))
    assert load_material(dump_material(item)) == item


def test_provider_routed_twice_raises(tmp_path):
    with pytest.raises(ProviderError):
        Gateway(CallLog(tmp_path), [(WIDE, None), (WIDE, None)])


def test_policy_labels_must_be_a_collection():
    with pytest.raises(AccessError):
        ProviderPolicy(provider=FAKE_A, labels="internal_code", retains=False)
