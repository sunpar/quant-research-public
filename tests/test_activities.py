from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from temporalio import activity, workflow
from temporalio.api.enums.v1 import EventType, TimeoutType
from temporalio.api.failure.v1 import Failure
from temporalio.client import Client, WorkflowHistory
from temporalio.converter import DefaultFailureConverter, DefaultPayloadConverter
from temporalio.exceptions import ActivityError, ApplicationError
from temporalio.testing import ActivityEnvironment
from temporalio.worker import Replayer, Worker

from quant_research import activities

# The sandbox imports this module to run Asking and Using, so imports pass through as in
# execution.py, and activities is imported outside the block to prove it loads in the sandbox.
with workflow.unsafe.imports_passed_through():
    import pytest

    from quant_research.access import INTERNAL_CODE, Provider, Restrictions, issue_context
    from quant_research.accounting import Usage
    from quant_research.agent_tools import Toolset
    from quant_research.campaigns import (
        CampaignRevision,
        CampaignSpecification,
        EvaluationRules,
        ResourceLimits,
        file_bytes,
        parse_file,
        revise,
        start_campaign,
    )
    from quant_research.catalog import Catalog
    from quant_research.findings import Ledger
    from quant_research.model_provenance import (
        UNKNOWN,
        ModelRecord,
        RoleAssignment,
        dump_assignment,
    )
    from quant_research.providers import (
        CallLog,
        CallRecord,
        Closed,
        Dispatched,
        Fault,
        Gateway,
        Lease,
        Material,
        ProviderError,
        ProviderPolicy,
        Receipt,
        Replied,
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

TIME = datetime(2026, 10, 22, 12, tzinfo=UTC)
PROVIDER = Provider("fake-a")
POLICY = ProviderPolicy(provider=PROVIDER, labels=frozenset({INTERNAL_CODE}), retains=False)
PRIMARY = ModelRecord("fake-a", "model-a", "v1", date(2026, 1, 31))
FALLBACK = ModelRecord("fake-a", "model-b", UNKNOWN, UNKNOWN)
PROPOSER = RoleAssignment("proposer", PRIMARY)
CONTEXT = issue_context(
    clearances=(INTERNAL_CODE,), cutoff=TIME, generation="synthetic/v1", providers=(PROVIDER,)
)
CLEARED = Restrictions(
    available_at=TIME - timedelta(days=1),
    labels=frozenset({INTERNAL_CODE}),
    approved_for=frozenset({PROVIDER}),
)
REQUEST = Request((Material("Propose a momentum signal", CLEARED),))
SEALED = Usage(model_calls=1, unreported=frozenset({"input_tokens", "output_tokens", "cost_usd"}))
# Well under every activity timeout below, since the local deadline is the timeout minus this.
GRACE = timedelta(seconds=0.5)
SEARCH = json.dumps({"mode": "lexical", "text": "momentum"})


def campaign() -> CampaignRevision:
    specification = CampaignSpecification(
        "Describe the synthetic level-difference association",
        (),
        (),
        EvaluationRules("pilot-evaluation/v1", "a" * 64),
        ResourceLimits(5, 1, 100),
    )
    return start_campaign(specification, author="researcher", at=TIME, reason="Ask the models")


def toolset(tmp_path: Path, context=CONTEXT) -> Toolset:
    """Return tools whose corpus is missing, so a search raises a SourceError naming its path."""
    missing = tmp_path / "corpus-secret"
    return Toolset(context, missing, missing, Catalog(), Ledger())


def authority(tools: Toolset) -> SimpleNamespace:
    """Return the two parts of an Authority the activities read: a send context and the tools."""
    return SimpleNamespace(context=tools.context, toolset=tools)


def bound(tmp_path, adapter, *, resolve=None, grace=GRACE) -> tuple[activities.Activities, CallLog]:
    calls = tmp_path / "calls"
    calls.mkdir(exist_ok=True)
    log = CallLog(calls)
    tools = authority(toolset(tmp_path))
    gateway = Gateway(log, [(POLICY, adapter)])
    return activities.Activities(gateway, resolve or (lambda revision: tools), grace=grace), log


def lease(models: int = 1) -> Lease:
    return Lease(uuid4(), tuple(uuid4() for _ in range(models)), 1)


def asked(held, *, request=REQUEST, assignment=PROPOSER, revision=None) -> activities._Ask:
    """Return the model activity's input as ask_model builds it."""
    return activities._Ask(
        file_bytes(revision or campaign()).decode(),
        dump_assignment(assignment),
        request.to_bytes().decode(),
        held,
    )


def used(tool: str, arguments: str, revision=None) -> activities._Use:
    return activities._Use(file_bytes(revision or campaign()).decode(), tool, arguments)


def environment(seconds: float = 30) -> ActivityEnvironment:
    """Return an activity environment whose deadline is seconds from now, not from the epoch."""
    env = ActivityEnvironment()
    now, limit = datetime.now(UTC), timedelta(seconds=seconds)
    env.info = replace(
        env.info,
        scheduled_time=now,
        started_time=now,
        current_attempt_scheduled_time=now,
        schedule_to_close_timeout=limit,
        start_to_close_timeout=limit,
        heartbeat_timeout=activities.HEARTBEAT,
    )
    return env


def execute(defn, argument, env=None):
    return asyncio.run((env or environment()).run(defn, argument))


def records(log: CallLog) -> list[CallRecord]:
    return [entry for entry in log.load() if isinstance(entry, CallRecord)]


class Stubborn:
    """An adapter that ignores its event until the test releases it."""

    subscription = False
    runtime_version = "stubborn"

    def __init__(self) -> None:
        self.started, self.release = threading.Event(), threading.Event()

    def complete(self, request, cancel):
        self.started.set()
        self.release.wait(10)
        return Replied("late", SEALED)


class Raising:
    """An adapter that raises with backing text instead of reporting a failure."""

    subscription = False
    runtime_version = "raising"

    def complete(self, request, cancel):
        raise RuntimeError("claude exited: token secret-7f3a")


@pytest.mark.smoke
def test_a_model_call_returns_its_answer_and_keeps_both_transcripts(tmp_path):
    acts, log = bound(tmp_path, ScriptedProvider(["Looks fine"]))
    held = lease()
    answer = load_result(execute(acts.model, asked(held)), REQUEST)
    assert answer.material == Material("Looks fine", REQUEST.restrictions)
    [record] = records(log)
    assert answer.record == record
    assert (record.call_id, record.outcome) == (held.dispatches[0], "completed")
    assert record.served == Served("scripted", "model-a")
    assert log.sent(record).to_bytes() == REQUEST.to_bytes()
    assert log.reply(record) == "Looks fine"


@pytest.mark.smoke
def test_a_provider_failure_is_recorded_and_the_fallback_answers(tmp_path):
    acts, log = bound(tmp_path, ScriptedProvider([Fault.FAIL, "Looks fine"]))
    held = lease(2)
    assignment = RoleAssignment("proposer", PRIMARY, (FALLBACK,))
    answer = load_result(execute(acts.model, asked(held, assignment=assignment)), REQUEST)
    assert answer.material.text == "Looks fine"
    assert [(r.call_id, r.model, r.outcome) for r in records(log)] == [
        (held.dispatches[0], PRIMARY, "provider_error"),
        (held.dispatches[1], FALLBACK, "completed"),
    ]


@pytest.mark.smoke
def test_a_cancel_sets_the_adapters_event_and_the_call_records_cancelled(tmp_path):
    provider = ScriptedProvider([Fault.BLOCK])
    acts, log = bound(tmp_path, provider)
    held = lease()
    env = environment()

    async def cancelled() -> None:
        running = asyncio.ensure_future(env.run(acts.model, asked(held)))
        assert await asyncio.to_thread(provider.blocking.wait, 5)
        env.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running

    asyncio.run(cancelled())
    [record] = records(log)
    assert (record.call_id, record.outcome) == (held.dispatches[0], "cancelled")
    assert load_receipts(execute(acts.close, held)) == (
        Receipt(record.call_id, "proposer", "cancelled", record.usage),
    )


@pytest.mark.smoke
@pytest.mark.parametrize("owner", ["activities", "execution"])
def test_a_bound_close_keeps_its_wire_contract_and_settles_once(tmp_path, owner):
    from quant_research import execution

    calls = tmp_path / "calls"
    calls.mkdir()
    log = CallLog(calls)
    gateway = Gateway(log, [(POLICY, ScriptedProvider(["answer"]))])
    acts = (
        activities.Activities(gateway, lambda revision: pytest.fail("close resolved authority"))
        if owner == "activities"
        else execution._Attempts({}, gateway)
    )
    held = lease()
    answer = gateway.call(CONTEXT, PROPOSER, REQUEST, lease=held, cancel=threading.Event())
    definition = activity._Definition.from_callable(acts.close)
    assert (definition.name, definition.arg_types, definition.ret_type) == ("close", [Lease], str)
    receipt = Receipt(answer.record.call_id, "proposer", "completed", answer.record.usage)
    assert execute(acts.close, held) == dump_receipts((receipt,))
    assert execute(acts.close, held) == dump_receipts((receipt,))
    assert len([entry for entry in log.load() if isinstance(entry, Closed)]) == 1


@pytest.mark.smoke
def test_a_compute_worker_has_no_gateway_to_close():
    from quant_research import execution

    with pytest.raises(ProviderError, match="This worker has no gateway"):
        execute(execution._Attempts({}, None).close, lease())


def test_each_worker_registers_one_close_activity(temporal, tmp_path):
    from quant_research import execution, research

    gateway = Gateway(CallLog(tmp_path), [(POLICY, ScriptedProvider([]))])

    async def body(client, name):
        for worker in (
            execution.worker(client, name, {}, gateway),
            research.worker(
                client,
                f"{name}-research",
                gateway,
                lambda revision: authority(toolset(tmp_path)),
                archive=research.CampaignArchive(tmp_path / "archive", tmp_path),
            ),
        ):
            definitions = [
                activity._Definition.from_callable(definition)
                for definition in worker.config()["activities"]
            ]
            assert [definition.name for definition in definitions].count("close") == 1

    connected(temporal, body)


@pytest.mark.smoke
def test_the_local_deadline_stops_a_blocked_call_before_the_temporal_deadline(tmp_path):
    acts, log = bound(tmp_path, ScriptedProvider([Fault.BLOCK]), grace=timedelta(seconds=1))
    began = time.monotonic()
    result = load_result(execute(acts.model, asked(lease()), environment(1.5)), REQUEST)
    assert time.monotonic() - began < 1.5
    assert result == Unanswered(tuple(records(log)))
    assert [record.outcome for record in result.records] == ["cancelled"]


@pytest.mark.smoke
def test_a_deadline_already_inside_the_grace_cancels_the_call_before_it_starts(tmp_path):
    acts, log = bound(tmp_path, ScriptedProvider(["Looks fine"]), grace=timedelta(seconds=1))
    result = load_result(execute(acts.model, asked(lease()), environment(0.5)), REQUEST)
    assert result == Unanswered(tuple(records(log)))
    assert [record.outcome for record in result.records] == ["cancelled"]


@pytest.mark.smoke
def test_an_adapter_that_ignores_its_event_is_abandoned_after_the_grace_and_logged(
    tmp_path, caplog
):
    adapter = Stubborn()
    acts, log = bound(tmp_path, adapter, grace=timedelta(seconds=0.2))
    held = lease()
    env = environment()

    async def abandoned() -> tuple[float, str]:
        running = asyncio.ensure_future(env.run(acts.model, asked(held)))
        assert await asyncio.to_thread(adapter.started.wait, 5)
        env.cancel()
        began = time.monotonic()
        with pytest.raises(asyncio.CancelledError):
            await running
        waited, logged = time.monotonic() - began, caplog.text
        adapter.release.set()
        deadline = time.monotonic() + 5
        while "abandoned model call" not in caplog.text:
            assert time.monotonic() < deadline, "the abandoned call's end was never logged"
            await asyncio.sleep(0.05)
        return waited, logged

    waited, logged = asyncio.run(abandoned())
    assert waited < 1
    assert f"model call of attempt {held.attempt_id} under fence 1 outlives" in logged
    assert "abandoned model call" not in logged
    assert [record.outcome for record in records(log)] == ["completed"]


@pytest.mark.smoke
@pytest.mark.parametrize("kind", ["Model", "Tool"])
def test_a_failure_crosses_temporal_as_a_fixed_error_without_its_backing_text(
    tmp_path, caplog, reachable_text, kind
):
    acts, _ = bound(tmp_path, Raising())
    work = (
        (acts.model, asked(lease()))
        if kind == "Model"
        else (acts.tool, used("search_corpus", SEARCH))
    )
    with pytest.raises(ApplicationError) as raised:
        execute(*work)
    error = raised.value
    assert (error.type, error.non_retryable) == (f"{kind}Failed", True)
    failure = Failure()
    DefaultFailureConverter().to_failure(error, DefaultPayloadConverter(), failure)
    assert b"secret" not in failure.SerializeToString()
    assert "secret" not in reachable_text(error)
    assert "secret" in caplog.text


@pytest.mark.smoke
def test_each_activity_runs_under_the_revision_its_input_names(tmp_path):
    root = campaign()
    child = revise(
        root,
        replace(root.specification, question="Does the association survive a one-day delay?"),
        author="researcher",
        at=TIME + timedelta(days=1),
        reason="Ask about the delay",
    )
    later = issue_context(
        clearances=(INTERNAL_CODE,),
        cutoff=TIME + timedelta(days=1),
        generation="synthetic/v1",
        providers=(),
    )
    contexts = {root.revision_id: CONTEXT, child.revision_id: later}
    seen = []

    def resolve(revision: CampaignRevision) -> SimpleNamespace:
        seen.append(revision.revision_id)
        return authority(toolset(tmp_path, contexts[revision.revision_id]))

    provider = ScriptedProvider(["Looks fine"])
    acts, _ = bound(tmp_path, provider, resolve=resolve)
    answered = load_result(execute(acts.model, asked(lease(), revision=root)), REQUEST)
    [refused] = load_result(execute(acts.model, asked(lease(), revision=child)), REQUEST).records
    lessons = [
        load_material(execute(acts.tool, used("read_lessons", "{}", revision)))
        for revision in (root, child)
    ]
    assert answered.material.text == "Looks fine"
    assert (refused.outcome, refused.reasons) == ("refused", ("provider_not_permitted",))
    assert len(provider.calls) == 1
    assert [material.restrictions.available_at for material in lessons] == [
        TIME,
        TIME + timedelta(days=1),
    ]
    assert seen == [root.revision_id, child.revision_id] * 2


@pytest.mark.smoke
def test_a_request_keeps_its_digest_across_a_crash_a_seal_and_a_reuse(tmp_path, monkeypatch):
    primary = replace(PRIMARY, version="unknown")
    assignment = RoleAssignment("proposer", primary, (FALLBACK,))
    request = Request(
        (
            Material("Propose a signal \ud800", CLEARED),
            Material("Prices", replace(CLEARED, available_at=TIME - timedelta(microseconds=1))),
        )
    )
    provider = ScriptedProvider(["first", "second"])
    acts, log = bound(tmp_path, provider)
    held = lease(2)
    ask = asked(held, request=request, assignment=assignment)
    append = CallLog.append

    def crash(self, entry):
        if isinstance(entry, CallRecord):
            raise RuntimeError("the worker died before the reply was stored")
        return append(self, entry)

    monkeypatch.setattr(CallLog, "append", crash)
    with pytest.raises(ApplicationError):
        execute(acts.model, ask)
    monkeypatch.undo()
    sealed, reused = (
        load_result(execute(acts.model, replace(ask, lease=replace(held, fence=fence))), request)
        for fence in (2, 3)
    )
    assert (sealed.material.text, sealed.record.fence) == ("second", 2)
    assert reused == sealed
    assert len(provider.calls) == 2
    assert [(r.model, r.fence, r.outcome) for r in records(log)] == [
        (primary, 1, "unknown"),
        (FALLBACK, 2, "completed"),
    ]
    entries = [entry for entry in log.load() if not isinstance(entry, Closed)]
    assert {entry.request_digest for entry in entries} == {request.digest}


@pytest.mark.smoke
@pytest.mark.parametrize("tool", ["read_lessons", "no_such_tool"])
def test_a_tool_call_returns_the_material_its_toolset_renders(tmp_path, tool):
    acts, _ = bound(tmp_path, ScriptedProvider([]))
    material = load_material(execute(acts.tool, used(tool, "{}")))
    assert material == toolset(tmp_path).call(tool, "{}").material


@pytest.mark.smoke
@pytest.mark.parametrize("failed", [False, True], ids=["completed", "failed"])
def test_a_stop_that_arrives_with_the_activitys_result_returns_cancelled(monkeypatch, failed):
    # One workflow task can carry both the activity's result and the stop signal.
    async def met(condition) -> None:
        assert condition()

    async def run():
        settled = asyncio.get_running_loop().create_future()
        if failed:
            settled.set_exception(
                ActivityError(
                    "failed",
                    scheduled_event_id=1,
                    started_event_id=2,
                    identity="worker",
                    activity_type="tool",
                    activity_id="1",
                    retry_state=None,
                )
            )
        else:
            settled.set_result(dump_material(REQUEST.materials[0]))
        monkeypatch.setattr(activities.workflow, "start_activity_method", lambda *_, **__: settled)
        return await activities.call_tool(
            campaign(), "read_lessons", "{}", within=timedelta(seconds=10), stop=lambda: True
        )

    monkeypatch.setattr(activities.workflow, "wait_condition", met)
    assert asyncio.run(run()) == "cancelled"


@dataclass(frozen=True)
class Plan:
    revision: str
    within: int


@dataclass(frozen=True)
class Reported:
    result: str
    receipts: str
    elapsed: float
    activities: tuple[activities.ModelActivity, ...]
    close_elapsed: float
    interrupted_by: str | None


@workflow.defn
class Asking:
    """Stand in for P12: one model attempt through ask_model, charging nothing."""

    def __init__(self) -> None:
        self._stopped = False

    @workflow.signal
    def stop(self) -> None:
        self._stopped = True

    @workflow.run
    async def run(self, plan: Plan) -> Reported:
        asked = await activities.ask_model(
            parse_file(plan.revision.encode()),
            PROPOSER,
            REQUEST,
            workflow.uuid4(),
            within=timedelta(seconds=plan.within),
            stop=lambda: self._stopped,
        )
        result = asked.result if isinstance(asked.result, str) else dump_result(asked.result)
        return Reported(
            result,
            dump_receipts(asked.receipts),
            asked.elapsed.total_seconds(),
            asked.activities,
            asked.close_elapsed.total_seconds(),
            asked.interrupted_by,
        )


@dataclass(frozen=True)
class ToolPlan:
    revision: str
    tool: str
    arguments: str


@workflow.defn
class Using:
    @workflow.run
    async def run(self, plan: ToolPlan) -> str:
        result = await activities.call_tool(
            parse_file(plan.revision.encode()),
            plan.tool,
            plan.arguments,
            within=timedelta(seconds=10),
            stop=lambda: False,
        )
        return result if isinstance(result, str) else dump_material(result)


def serve(address: str, queue: str, calls: str) -> None:
    """Serve the model activity over a provider that blocks, until the test kills this process."""
    gateway = Gateway(CallLog(Path(calls)), [(POLICY, ScriptedProvider([Fault.BLOCK]))])
    tools = authority(toolset(Path(calls).parent))
    served = activities.Activities(gateway, lambda revision: tools, grace=GRACE)

    async def serving() -> None:
        client = await Client.connect(address)
        await Worker(client, task_queue=queue, activities=served.definitions).run()

    asyncio.run(serving())


def connected(address: str, body) -> None:
    async def run() -> None:
        await body(await Client.connect(address), str(uuid4()))

    asyncio.run(asyncio.wait_for(run(), 60))


def outcomes(history: WorkflowHistory) -> list[tuple[str, str]]:
    activity_ids, closed = {}, []
    for event in history.events:
        kind = EventType.Name(event.event_type).removeprefix("EVENT_TYPE_ACTIVITY_TASK_")
        if kind == "SCHEDULED":
            activity_ids[event.event_id] = (
                event.activity_task_scheduled_event_attributes.activity_id
            )
        elif kind in {"COMPLETED", "FAILED", "TIMED_OUT", "CANCELED"}:
            attributes = getattr(event, event.WhichOneof("attributes"))
            closed.append((activity_ids[attributes.scheduled_event_id], kind))
    return closed


@pytest.mark.parametrize("how", ["cancel", "stop"])
def test_a_cancel_or_stop_during_a_blocked_model_call_records_cancelled(temporal, tmp_path, how):
    provider = ScriptedProvider([Fault.BLOCK])
    acts, log = bound(tmp_path, provider)

    async def body(client, name):
        async with Worker(client, task_queue=name, workflows=[Asking], activities=acts.definitions):
            handle = await client.start_workflow(
                Asking.run, Plan(file_bytes(campaign()).decode(), 30), id=name, task_queue=name
            )
            assert await asyncio.to_thread(provider.blocking.wait, 10)
            await (handle.cancel() if how == "cancel" else handle.signal(Asking.stop))
            reported = await handle.result()
        claim, record, closed = log.load()
        assert (type(claim), record.outcome, closed) == (
            Dispatched,
            "cancelled",
            Closed(claim.attempt_id),
        )
        assert reported.result == "cancelled"
        assert [(item.fence, item.ended) for item in reported.activities] == [(1, "cancelled")]
        assert reported.interrupted_by == ("workflow" if how == "cancel" else "stop")
        assert load_receipts(reported.receipts) == (
            Receipt(record.call_id, "proposer", "cancelled", record.usage),
        )
        assert outcomes(await handle.fetch_history()) == [
            (f"model/{claim.attempt_id}/1", "CANCELED"),
            (f"close/{claim.attempt_id}", "COMPLETED"),
        ]
        await Replayer(workflows=[Asking]).replay_workflow(await handle.fetch_history())

    connected(temporal, body)


def test_a_model_attempt_that_runs_out_of_its_time_records_cancelled(temporal, tmp_path):
    acts, log = bound(tmp_path, ScriptedProvider([Fault.BLOCK]), grace=timedelta(seconds=1))

    async def body(client, name):
        async with Worker(client, task_queue=name, workflows=[Asking], activities=acts.definitions):
            handle = await client.start_workflow(
                Asking.run, Plan(file_bytes(campaign()).decode(), 3), id=name, task_queue=name
            )
            reported = await handle.result()
        [record] = load_result(reported.result, REQUEST).records
        assert record.outcome == "cancelled"
        assert records(log) == [record]
        assert load_receipts(reported.receipts) == (
            Receipt(record.call_id, "proposer", "cancelled", record.usage),
        )
        assert reported.elapsed < 3
        assert [(item.fence, item.ended) for item in reported.activities] == [(1, "completed")]
        assert reported.interrupted_by is None
        assert outcomes(await handle.fetch_history()) == [
            (f"model/{record.attempt_id}/1", "COMPLETED"),
            (f"close/{record.attempt_id}", "COMPLETED"),
        ]

    connected(temporal, body)


@pytest.mark.smoke
def test_repeated_cancellation_keeps_one_close_alive_until_its_receipts_arrive(monkeypatch):
    async def run():
        closing = asyncio.get_running_loop().create_future()
        started = []

        def schedule(*args, **kwargs):
            started.append(kwargs["activity_id"])
            return closing

        monkeypatch.setattr(activities.workflow, "start_activity_method", schedule)
        held = lease()
        task = asyncio.create_task(activities._closed(held))
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and not closing.cancelled()
        closing.set_result(dump_receipts(()))
        assert await task == ((), True)
        assert started == [f"close/{held.attempt_id}"]

    asyncio.run(run())


@workflow.defn
class LargeInput:
    @workflow.run
    async def run(self, revision: str, quotes: bool) -> str:
        request = Request((Material(('"' if quotes else "x") * 614400, CLEARED),))
        asked = await activities.ask_model(
            parse_file(revision.encode()),
            PROPOSER,
            request,
            workflow.uuid4(),
            within=timedelta(seconds=30),
            stop=lambda: False,
        )
        return asked.failure_reason or (
            "answered" if isinstance(asked.result, activities.Answer) else "failed"
        )


@pytest.mark.parametrize("quotes", [False, True])
def test_the_model_input_uses_the_measured_transport_limit_and_full_json_envelope(
    temporal, tmp_path, quotes
):
    provider = ScriptedProvider(["answer"])
    bound_activities, log = bound(tmp_path, provider)

    async def body(client, name):
        async with Worker(
            client, task_queue=name, workflows=[LargeInput], activities=bound_activities.definitions
        ):
            handle = await client.start_workflow(
                LargeInput.run,
                args=[file_bytes(campaign()).decode(), quotes],
                id=name,
                task_queue=name,
            )
            assert await handle.result() == ("invalid_payload" if quotes else "answered")
        history = await handle.fetch_history()
        assert not any(
            event.event_type == EventType.EVENT_TYPE_WORKFLOW_TASK_FAILED
            for event in history.events
        )
        assert len(provider.calls) == (0 if quotes else 1)
        assert len(outcomes(history)) == (0 if quotes else 2)
        if not quotes:
            assert log.sent(records(log)[0]).materials[0].text == "x" * 614400

    connected(temporal, body)


def test_a_worker_killed_mid_send_resumes_under_the_next_fence_and_returns_an_unknown_receipt(
    temporal, tmp_path
):
    calls = tmp_path / "calls"
    calls.mkdir()
    restarted = ScriptedProvider([])
    acts, log = bound(tmp_path, restarted)

    async def body(client, name):
        # The verify gate seals its environment, so the child must not write bytecode into it.
        bootstrap = (
            f"import runpy, sys; sys.path[:] = {sys.path!r}; "
            f"runpy.run_path({__file__!r})['serve']({temporal!r}, {name!r}, {str(calls)!r})"
        )
        async with Worker(client, task_queue=name, workflows=[Asking]):
            killed = subprocess.Popen([sys.executable, "-B", "-I", "-S", "-c", bootstrap])
            try:
                handle = await client.start_workflow(
                    Asking.run, Plan(file_bytes(campaign()).decode(), 30), id=name, task_queue=name
                )
                deadline = time.monotonic() + 30
                while not any(isinstance(entry, Dispatched) for entry in log.load()):
                    assert time.monotonic() < deadline, "the killed worker never claimed the call"
                    await asyncio.sleep(0.1)
            finally:
                killed.kill()
                killed.wait()
            async with Worker(client, task_queue=name, activities=acts.definitions):
                reported = await handle.result()
        history = await handle.fetch_history()
        claim, sealed, _ = log.load()
        assert (claim.fence, sealed.fence, sealed.outcome) == (1, 1, "unknown")
        assert sealed.served == Served("scripted", None)
        assert load_result(reported.result, REQUEST) == Unanswered((sealed,))
        assert load_receipts(reported.receipts) == (
            Receipt(sealed.call_id, "proposer", "unknown", SEALED),
        )
        assert restarted.calls == []
        attempt = claim.attempt_id
        assert outcomes(history) == [
            (f"model/{attempt}/1", "TIMED_OUT"),
            (f"model/{attempt}/2", "COMPLETED"),
            (f"close/{attempt}", "COMPLETED"),
        ]
        [timeout] = [
            event.activity_task_timed_out_event_attributes
            for event in history.events
            if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_TIMED_OUT
        ]
        assert (
            timeout.failure.timeout_failure_info.timeout_type == TimeoutType.TIMEOUT_TYPE_HEARTBEAT
        )

    connected(temporal, body)


def test_a_tool_call_through_a_workflow_returns_its_material_or_failed(temporal, tmp_path):
    acts, _ = bound(tmp_path, ScriptedProvider([]))
    revision = file_bytes(campaign()).decode()

    async def body(client, name):
        async with Worker(client, task_queue=name, workflows=[Using], activities=acts.definitions):
            lessons = await client.execute_workflow(
                Using.run, ToolPlan(revision, "read_lessons", "{}"), id=name, task_queue=name
            )
            searching = await client.start_workflow(
                Using.run,
                ToolPlan(revision, "search_corpus", SEARCH),
                id=f"{name}-search",
                task_queue=name,
            )
            searched = await searching.result()
        assert load_material(lessons) == toolset(tmp_path).call("read_lessons", "{}").material
        assert searched == "failed"
        assert "secret" not in (await searching.fetch_history()).to_json()

    connected(temporal, body)
