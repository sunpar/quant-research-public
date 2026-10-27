from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from temporalio import activity, workflow
from temporalio.api.common.v1 import Payload, Payloads
from temporalio.api.enums.v1 import EventType, PendingActivityState, TimeoutType
from temporalio.client import (
    Client,
    WorkflowFailureError,
    WorkflowHandle,
    WorkflowHistory,
    WorkflowUpdateFailedError,
)
from temporalio.common import RawValue, RetryPolicy
from temporalio.converter import DataConverter
from temporalio.exceptions import ApplicationError, WorkflowAlreadyStartedError
from temporalio.service import RPCError, RPCStatusCode
from temporalio.testing import ActivityEnvironment
from temporalio.worker import (
    Interceptor,
    Replayer,
    UnsandboxedWorkflowRunner,
    Worker,
    WorkflowInboundInterceptor,
)

from quant_research import activities, execution
from quant_research.acceptance import Accepted, Execution, load_log
from quant_research.access import INTERNAL_CODE, Provider, Restrictions, issue_context, licensed
from quant_research.accounting import Charged, Released, Reserved, Usage, load_ledger
from quant_research.campaigns import (
    CampaignRevision,
    CampaignSpecification,
    EvaluationRules,
    ResourceLimits,
    file_bytes,
    revise,
    start_campaign,
)
from quant_research.execution import (
    CampaignRun,
    Job,
    ModelBody,
    ModelJob,
    ModelWork,
    Record,
    Report,
    RunnerError,
    Summary,
)
from quant_research.model_provenance import ModelRecord, RoleAssignment
from quant_research.packages import Outcome, open_package
from quant_research.providers import (
    Answer,
    CallLog,
    CallRecord,
    Dispatched,
    Fault,
    Gateway,
    Lease,
    Material,
    ProviderPolicy,
    Replied,
    Request,
    ScriptedProvider,
)
from quant_research.records import PackageError, canonical_bytes
from quant_research.worker import Limits, RunLog
from tests import test_research

research_setup = test_research.setup

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/causal/known-answer.json"
RULES = hashlib.sha256(b"pilot evaluation rules\n").hexdigest()
QUESTION = "Describe the synthetic level-difference association"
STEERED = "Does the level-difference association survive a one-day delay?"
TIME = datetime(2026, 10, 22, 12, tzinfo=UTC)
PROVIDER = Provider("fake-a")
POLICY = ProviderPolicy(provider=PROVIDER, labels=frozenset({INTERNAL_CODE}), retains=False)
PROPOSER = RoleAssignment("proposer", ModelRecord("fake-a", "model-a", "v1", date(2026, 1, 31)))
CONTEXT = issue_context(
    clearances=(INTERNAL_CODE,), cutoff=TIME, generation="synthetic/v1", providers=(PROVIDER,)
)
CLEARED = Restrictions(
    available_at=TIME - timedelta(days=1),
    labels=frozenset({INTERNAL_CODE}),
    approved_for=frozenset({PROVIDER}),
)
PROMPT = "Propose a momentum signal"
SEALED = Usage(model_calls=1, unreported=frozenset({"input_tokens", "output_tokens", "cost_usd"}))
HISTORICAL = ROOT / "tests/fixtures/execution/dcd7597"
ROLE_PROMPT = "Follow the pinned role instructions"


def model_authority(revision):
    return SimpleNamespace(
        revision=revision.revision_id,
        context=CONTEXT,
        roles=SimpleNamespace(assignments=(PROPOSER,)),
        prompts={"proposer": Material(ROLE_PROMPT, CLEARED)},
    )


async def prepare(work: ModelWork) -> tuple[Material, ...]:
    return (Material(work.argument, CLEARED),)


async def interpret(work: ModelWork, request, result) -> Outcome:
    return answered_by(result)


async def prepare_slowly(work: ModelWork) -> tuple[Material, ...]:
    await asyncio.sleep(1.5)
    return await prepare(work)


MODELS = {"ask": ModelBody(prepare, interpret), "ask_slowly": ModelBody(prepare_slowly, interpret)}


def serve_model(address: str, queue: str, calls: str, boundary: str) -> None:
    async def serve():
        provider = ScriptedProvider([Fault.BLOCK if boundary == "claim" else "answer"])
        gateway = Gateway(CallLog(Path(calls)), [(POLICY, provider)])
        bound = activities.Activities(gateway, model_authority)

        @activity.defn(name="model")
        async def lost(ask: activities._Ask) -> str:
            result = await bound.model(ask)
            await asyncio.sleep(60)
            return result

        publication = Path(calls).parent / "publication.json"

        async def preparing(work):
            if boundary == "prepare":
                publication.write_text(str(work.attempt_id))
                await asyncio.sleep(60)
            return await prepare(work)

        async def interpreting(work, request, result):
            publication.write_bytes(
                canonical_bytes(
                    {"attempt": str(work.attempt_id), "request": request.to_bytes().decode()}
                )
            )
            await asyncio.sleep(60)
            return answered_by(result)

        bindings = {
            "ask": ModelBody(preparing, interpreting if boundary == "interpret" else interpret)
        }
        bodies = execution._ModelBodies(bindings, model_authority)
        await Worker(
            await Client.connect(address),
            task_queue=queue,
            workflows=[CampaignRun],
            activities=[
                bodies.prepare_model,
                lost if boundary in {"claim", "reply"} else bound.model,
                bodies.interpret_model,
                bound.close,
            ],
        ).run()

    asyncio.run(serve())


@pytest.fixture(scope="module")
def server(temporal):
    return temporal


def run(address: str, body: Callable[[Client, str], Awaitable[None]]) -> None:
    async def connected() -> None:
        await body(await Client.connect(address), str(uuid4()))

    asyncio.run(asyncio.wait_for(connected(), 60))


async def started(handle: WorkflowHandle, activity_id: str) -> None:
    while True:
        pending = (await handle.describe()).raw_description.pending_activities
        if any(
            item.activity_id == activity_id
            and item.state == PendingActivityState.PENDING_ACTIVITY_STATE_STARTED
            for item in pending
        ):
            return
        await asyncio.sleep(0.1)


def events(history: WorkflowHistory, kind: str) -> list:
    event_type = EventType.Value(f"EVENT_TYPE_{kind}")
    return [
        getattr(event, event.WhichOneof("attributes"))
        for event in history.events
        if event.event_type == event_type
    ]


def scheduled(history: WorkflowHistory) -> list[str]:
    return [attributes.activity_id for attributes in events(history, "ACTIVITY_TASK_SCHEDULED")]


def outcomes(history: WorkflowHistory) -> list[tuple[str, str]]:
    activity_ids = {}
    closed = []
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


def day(number: int) -> datetime:
    return TIME + timedelta(days=number)


def campaign(
    *,
    trials: int = 5,
    attempts: int = 1,
    seconds: int = 100,
    generated_code: bool = False,
    **limits: int,
) -> CampaignRevision:
    specification = CampaignSpecification(
        QUESTION,
        (),
        (),
        EvaluationRules("pilot-evaluation/v1", RULES),
        ResourceLimits(trials, attempts, seconds, **limits),
        generated_code=generated_code,
    )
    return start_campaign(specification, author="researcher", at=TIME, reason="Run the jobs")


def executions(record: Record) -> list[Execution]:
    return list(load_log(record.acceptance.encode()).executions().values())


def links(record: Record) -> list[tuple[str, str | None]]:
    return [(link.activity_id, link.ended) for link in record.attempts]


def report(detail: str) -> Report:
    return Report(Outcome("successful", detail), datetime.now(UTC))


async def answer(revision: CampaignRevision, argument: str, dispatch) -> Outcome:
    await asyncio.sleep(float(argument))
    return Outcome("successful", revision.specification.question)


async def crash(revision: CampaignRevision, argument: str, dispatch) -> Outcome:
    raise RuntimeError(argument)


async def garbled(revision: CampaignRevision, argument: str, dispatch) -> Outcome:
    return Outcome("successful", "wrote \udcff.csv")


class Unrecordable(Exception):
    def __repr__(self) -> str:
        return "<file prices-\udcff.csv>"


async def unrecordable(revision: CampaignRevision, argument: str, dispatch) -> Outcome:
    raise Unrecordable


async def hand_off(revision: CampaignRevision, argument: str, dispatch) -> Outcome:
    activity.raise_complete_async()


async def bulky(revision: CampaignRevision, argument: str, dispatch) -> Outcome:
    return Outcome("successful", "x" * int(argument))


def answered_by(result) -> Outcome:
    if isinstance(result, Answer):
        return Outcome("successful", result.material.text)
    return Outcome("failed", ", ".join(record.outcome for record in result.records))


async def ask(revision: CampaignRevision, argument: str, dispatch) -> Outcome:
    request = Request((Material(argument, CLEARED),))
    result = await asyncio.to_thread(dispatch, CONTEXT, PROPOSER, request, cancel=threading.Event())
    return answered_by(result)


BODIES = {
    "answer": answer,
    "bulky": bulky,
    "garbled": garbled,
    "crash": crash,
    "hand_off": hand_off,
    "unrecordable": unrecordable,
    "replay": execution.replay,
}


def attempt_activities(history: WorkflowHistory) -> list[str]:
    return [item for item in scheduled(history) if not item.startswith("close/")]


def charges(record: Record) -> list[Charged]:
    return [e for e in load_ledger(record.ledger.encode()).events if isinstance(e, Charged)]


@pytest.mark.smoke
def test_frozen_campaign_histories_replay_with_exact_records_and_memo_digests():
    manifest = json.loads((HISTORICAL / "manifest.json").read_text())
    for item in manifest["files"]:
        data = (HISTORICAL / item["path"]).read_bytes()
        assert len(data) == item["bytes"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"]

    class Capture(Interceptor):
        def __init__(self):
            self.records = []
            self.summaries = []

        def workflow_interceptor_class(self, input):
            records, summaries = self.records, self.summaries

            class ReadClosed(WorkflowInboundInterceptor):
                async def execute_workflow(self, input):
                    result = await super().execute_workflow(input)
                    instance = workflow.instance()
                    parts = []
                    for offset in itertools.count(0, execution.PART):
                        part = instance.part(offset)
                        parts.append(part)
                        if len(part) < execution.PART:
                            break
                    records.append(b"".join(parts))
                    summaries.append(
                        DataConverter.default.payload_converter.to_payload(instance.summary()).data
                    )
                    return result

            return ReadClosed

    async def replay():
        for name, digest in (
            (
                "reply-before-report-loss",
                "d6da5ac5a13f6b97dfc59e409312b263c66c159412e413cf422f3aa067662ade",
            ),
            ("steering-cancel", "cadf0950709fca50d2fc8d61c48cfcfb461f3c41e4272234fc76ef672fffe1ed"),
        ):
            case = HISTORICAL / name
            plan = json.loads((case / "plan.payload.data").read_bytes())
            assert hashlib.sha256(canonical_bytes(plan)).hexdigest() == digest
            expected = (case / "expected-record.payload.data").read_bytes()
            history = WorkflowHistory.from_json(
                json.loads(expected)["workflow_id"], (case / "closed/history.json").read_text()
            )
            capture = Capture()
            converter = replace(DataConverter.default, failure_converter_class=execution._Failures)
            await Replayer(
                workflows=[CampaignRun], interceptors=[capture], data_converter=converter
            ).replay_workflow(history)
            assert capture.records == [expected]
            assert capture.summaries == [(case / "expected-summary.payload.data").read_bytes()]

    asyncio.run(replay())


def test_frozen_pending_task_reuses_its_reply_on_an_upgraded_worker(tmp_path):
    pending = HISTORICAL / "reply-before-report-loss/before-kill"
    for name in ("calls", "calls.transcripts"):
        shutil.copytree(pending / name, tmp_path / name)
    converter = DataConverter.default.payload_converter
    task = converter.from_payload(
        Payload(
            metadata={"encoding": b"json/plain"},
            data=(pending / "activity-5-input-0.payload.data").read_bytes(),
        ),
        execution.Task,
    )
    provider = ScriptedProvider([])
    gateway = Gateway(CallLog(tmp_path / "calls"), [(POLICY, provider)])
    attempts = execution._Attempts({"ask": ask}, gateway)
    result = asyncio.run(ActivityEnvironment().run(attempts.attempt, task))
    assert isinstance(result, Report)
    assert result.outcome == Outcome("successful", "answer")
    assert provider.calls == []
    receipts = asyncio.run(ActivityEnvironment().run(attempts.close, task.lease))
    assert len(json.loads(receipts)) == 1
    for path in (pending / "calls.transcripts").iterdir():
        assert (tmp_path / "calls.transcripts" / path.name).read_bytes() == path.read_bytes()


def test_raw_fresh_legacy_model_input_fails_before_reserving_or_dispatching(server, tmp_path):
    async def body(client, name):
        revision = campaign(max_model_calls=1)
        provider = ScriptedProvider([])
        gateway = Gateway(CallLog(tmp_path), [(POLICY, provider)])
        async with execution.worker(client, name, BODIES, gateway):
            handle = await client.start_workflow(
                CampaignRun.run,
                execution.Plan(
                    file_bytes(revision).decode(), (Job("q", "ask", PROMPT, 0, dispatches=1),)
                ),
                id=str(revision.campaign_id),
                task_queue=name,
            )
            with pytest.raises(WorkflowFailureError) as failure:
                await handle.result()
            assert failure.value.cause.type == "LegacyModelProtocol"
            assert failure.value.cause.non_retryable
            raw = await handle.query(CampaignRun.part, 0)
        assert json.loads(raw)["attempts"] == []
        assert load_ledger(json.loads(raw)["ledger"].encode()).events == ()
        assert scheduled(await handle.fetch_history()) == []
        assert provider.calls == []

    run(server, body)


def test_prepared_model_body_uses_pinned_authority_and_returns_a_domain_outcome(server, tmp_path):
    async def body(client, name):
        seen = []

        async def rejected(work, request, result):
            seen.append((work, request.to_bytes()))
            return Outcome("rejected", result.material.text, limitations=("synthetic",))

        provider = ScriptedProvider(["domain rejection"])
        gateway = Gateway(CallLog(tmp_path), [(POLICY, provider)])
        revision = campaign(max_model_calls=1)
        job = ModelJob("q", "ask", PROMPT, 2, "proposer", 1)
        plan = execution._plan_v2(file_bytes(revision).decode(), (job,))
        payload = DataConverter.default.payload_converter.to_payload(plan)
        assert isinstance(execution._read_plan(RawValue(payload)), execution._PlanV2)
        with pytest.raises(TypeError):
            DataConverter.default.payload_converter.from_payload(payload, execution.Plan)
        async with execution.worker(
            client,
            name,
            {},
            gateway,
            models={"ask": ModelBody(prepare, rejected)},
            resolve=model_authority,
        ):
            handle = await execution.start(client, revision, (job,), name)
            record = await execution.record(handle)
        assert record.schema_version == 2
        assert [(link.phase, link.ended) for link in record.attempts] == [
            ("prepare", "completed"),
            ("model", "completed"),
            ("interpret", "completed"),
        ]
        assert record.interruptions == ()
        [accepted] = executions(record)
        assert accepted.accepted.completion.outcome == Outcome(
            "rejected", "domain rejection", limitations=("synthetic",)
        )
        assert seen[0][0].attempt_id == accepted.accepted.submission.attempt_id
        assert seen[0][0].revision == revision
        assert (
            seen[0][1]
            == Request((Material(ROLE_PROMPT, CLEARED), Material(PROMPT, CLEARED))).to_bytes()
        )
        assert provider.calls[0].texts == (ROLE_PROMPT, PROMPT)
        history = await handle.fetch_history()
        assert len([item for item in scheduled(history) if item.startswith("close/")]) == 1
        assert len([charge for charge in charges(record) if charge.usage.model_calls]) == 1
        assert not events(history, "WORKFLOW_TASK_FAILED")
        await Replayer(workflows=[CampaignRun]).replay_workflow(history)

    run(server, body)


@pytest.mark.parametrize("how", ["steering", "temporal"])
@pytest.mark.parametrize("phase", ["prepare", "model", "interpret", "close"])
def test_cancellation_settles_each_model_phase_before_releasing(server, tmp_path, how, phase):
    async def body(client, name):
        ready, release = threading.Event(), threading.Event()
        interpreted = []

        async def preparing(work):
            if phase == "prepare":
                ready.set()
                await asyncio.sleep(60)
            return await prepare(work)

        async def interpreting(work, request, result):
            interpreted.append(work.attempt_id)
            if phase == "interpret":
                ready.set()
                await asyncio.sleep(60)
            return answered_by(result)

        provider = ScriptedProvider([Fault.BLOCK if phase == "model" else "answer"])

        class ClosingGateway(Gateway):
            def close(self, lease):
                if phase == "close":
                    ready.set()
                    assert release.wait(10)
                return super().close(lease)

        gateway = ClosingGateway(CallLog(tmp_path), [(POLICY, provider)])
        async with execution.worker(
            client,
            name,
            {},
            gateway,
            models={"ask": ModelBody(preparing, interpreting)},
            resolve=model_authority,
        ):
            handle = await execution.start(
                client,
                campaign(max_model_calls=1),
                (ModelJob("q", "ask", PROMPT, 0, "proposer", 1),),
                name,
            )
            assert await asyncio.to_thread(
                (provider.blocking if phase == "model" else ready).wait, 10
            )
            if how == "temporal":
                await handle.cancel()
            else:
                await execution.control(
                    handle, "cancel", actor="researcher", at=day(1), reason="Stop this attempt"
                )
            release.set()
            result = await handle.result()
            record = await execution.record(handle)
        assert result.status == record.status == "cancelled"
        [run_record] = executions(record)
        assert run_record.accepted is None
        assert run_record.attempts[0].completion is None
        assert record.interruptions[-1].reason == (
            "temporal_cancel" if how == "temporal" else "steering_cancel"
        )
        ledger = load_ledger(record.ledger.encode())
        assert isinstance(ledger.events[-1], Released)
        model_charges = [charge for charge in charges(record) if charge.usage.model_calls]
        assert len(model_charges) == (0 if phase == "prepare" else 1)
        history = await handle.fetch_history()
        assert len(events(history, "WORKFLOW_EXECUTION_COMPLETED")) == 1
        assert not events(history, "WORKFLOW_TASK_FAILED")
        if phase == "model":
            assert gateway.records[0].outcome == "cancelled"
            terminal = outcomes(history)
            assert terminal[-2][0].startswith("model/") and terminal[-2][1] == "CANCELED"
            assert terminal[-1][0].startswith("close/")
        if phase in {"prepare", "model", "close"}:
            assert interpreted == []

    run(server, body)


@pytest.mark.parametrize("phase", ["prepare", "model", "interpret"])
def test_same_task_stop_keeps_a_delivered_report_without_accepting_it(
    server, tmp_path, monkeypatch, phase
):
    original = execution.workflow.wait_condition

    async def racing(condition, **kwargs):
        await original(condition, **kwargs)
        instance = workflow.instance()
        if isinstance(instance, CampaignRun) and instance._activities:
            current = instance._activities[-1].phase
            if (
                phase == current
                or phase == "model"
                and current == "prepare"
                and len(instance._activities) == 1
                and condition.__module__ == activities.__name__
            ):
                instance.control(
                    "cancel",
                    "researcher",
                    day(1).isoformat().replace("+00:00", "Z"),
                    "Stop with result",
                )

    monkeypatch.setattr(execution.workflow, "wait_condition", racing)

    async def body(client, name):
        provider = ScriptedProvider(["answer"])
        gateway = Gateway(CallLog(tmp_path), [(POLICY, provider)])
        bindings = execution._ModelBodies(MODELS, model_authority)
        model = activities.Activities(gateway, model_authority)
        async with Worker(
            client,
            task_queue=name,
            workflows=[CampaignRun],
            workflow_runner=UnsandboxedWorkflowRunner(),
            activities=[bindings.prepare_model, model.model, bindings.interpret_model, model.close],
        ):
            handle = await execution.start(
                client,
                campaign(max_model_calls=1),
                (ModelJob("q", "ask", PROMPT, 0, "proposer", 1),),
                name,
            )
            record = await execution.record(handle)
        assert record.status == "cancelled"
        [run_record] = executions(record)
        [attempt] = run_record.attempts
        assert run_record.accepted is None
        assert (attempt.completion is not None) == (phase == "interpret")
        assert record.attempts[-1].ended == "completed"
        if attempt.completion:
            reported = events(await handle.fetch_history(), "ACTIVITY_TASK_COMPLETED")[
                -1
            ].result.payloads[0]
            raw = DataConverter.default.payload_converter.from_payload(reported)
            assert attempt.completion.finished_at == datetime.fromisoformat(raw["finished_at"])

    run(server, body)


@pytest.mark.parametrize("phase", ["prepare", "model", "interpret"])
@pytest.mark.parametrize("payload", ["unknown", "missing"])
def test_malformed_phase_results_fail_only_the_attempt_and_close_sent_calls(
    server, tmp_path, phase, payload
):
    async def body(client, name):
        provider = ScriptedProvider(["answer"])
        gateway = Gateway(CallLog(tmp_path), [(POLICY, provider)])
        bindings = execution._ModelBodies(MODELS, model_authority)
        models = activities.Activities(gateway, model_authority)
        legacy = execution._Attempts(BODIES, gateway)

        @activity.defn(name=f"{phase}_model" if phase != "model" else "model")
        async def malformed(raw: RawValue) -> RawValue:
            if phase == "model":
                ask = DataConverter.default.payload_converter.from_payload(
                    raw.payload, activities._Ask
                )
                await models.model(ask)
            if payload == "missing":
                activity.raise_complete_async()
            return RawValue(Payload(metadata={"encoding": b"binary/unknown"}, data=b"?"))

        definitions = [
            bindings.prepare_model,
            models.model,
            bindings.interpret_model,
            legacy.attempt,
            legacy.close,
        ]
        name_of_phase = f"{phase}_model" if phase != "model" else "model"
        definitions = [
            malformed if activity._Definition.from_callable(item).name == name_of_phase else item
            for item in definitions
        ]
        converter = replace(client.data_converter, failure_converter_class=execution._Failures)
        async with Worker(
            Client(**client.config() | {"data_converter": converter}),
            task_queue=name,
            workflows=[CampaignRun],
            activities=definitions,
        ):
            jobs = (ModelJob("q", "ask", PROMPT, 0, "proposer", 1), Job("next", "answer", "0", 1))
            handle = await execution.start(client, campaign(max_model_calls=2), jobs, name)
            if payload == "missing":
                while True:
                    pending = (await handle.describe()).raw_description.pending_activities
                    found = next(
                        (
                            item
                            for item in pending
                            if item.activity_type.name == name_of_phase
                            and item.state == PendingActivityState.PENDING_ACTIVITY_STATE_STARTED
                        ),
                        None,
                    )
                    if found is not None:
                        break
                    await asyncio.sleep(0.02)
                await client.get_async_activity_handle(
                    workflow_id=handle.id, run_id=handle.run_id, activity_id=found.activity_id
                ).complete()
            record = await execution.record(handle)
        failed, next_job = executions(record)
        assert failed.attempts[0].completion is None
        assert failed.accepted is None and next_job.accepted is not None
        assert record.interruptions[0].reason == (
            "invalid_preparation" if phase == "prepare" else "invalid_result"
        )
        assert len([charge for charge in charges(record) if charge.usage.model_calls]) == (
            0 if phase == "prepare" else 1
        )
        history = await handle.fetch_history()
        assert not events(history, "WORKFLOW_TASK_FAILED")
        assert len([item for item in scheduled(history) if item.startswith("close/")]) == (
            0 if phase == "prepare" else 1
        )

    run(server, body)


@pytest.mark.parametrize("phase", ["prepare", "interpret"])
def test_a_lost_callback_keeps_the_attempt_and_never_repeats_a_finished_model(
    server, tmp_path, phase
):
    calls, publication = tmp_path / "calls", tmp_path / "publication.json"
    calls.mkdir()

    async def body(client, name):
        bootstrap = (
            f"import runpy, sys; sys.path[:] = {sys.path!r}; "
            f"runpy.run_path({__file__!r})['serve_model']"
            f"({server!r}, {name!r}, {str(calls)!r}, {phase!r})"
        )
        killed = subprocess.Popen([sys.executable, "-B", "-I", "-S", "-c", bootstrap])
        try:
            handle = await execution.start(
                client,
                campaign(max_model_calls=1),
                (ModelJob("q", "ask", PROMPT, 0, "proposer", 1),),
                name,
            )
            deadline = time.monotonic() + 30
            while not publication.exists():
                assert killed.poll() is None
                assert time.monotonic() < deadline
                await asyncio.sleep(0.05)
        finally:
            killed.kill()
            killed.wait()
        frozen = publication.read_bytes()
        provider = ScriptedProvider(["answer"] if phase == "prepare" else [])
        gateway = Gateway(CallLog(calls), [(POLICY, provider)])
        seen = []

        async def finishing(work, request, result):
            seen.append((str(work.attempt_id), request.to_bytes()))
            return Outcome("inconclusive", result.material.text, ("result.json",), ("synthetic",))

        async with execution.worker(
            client,
            name,
            {},
            gateway,
            models={"ask": ModelBody(prepare, finishing)},
            resolve=model_authority,
        ):
            record = await execution.record(handle)
        assert publication.read_bytes() == frozen
        assert len(provider.calls) == (1 if phase == "prepare" else 0)
        assert len({link.attempt_id for link in record.attempts}) == 1
        assert [link.ended for link in record.attempts if link.phase == phase] == [
            "timed_out",
            "completed",
        ]
        assert len([link for link in record.attempts if link.phase == "model"]) == 1
        if phase == "prepare":
            assert seen[0][0] == frozen.decode()
        else:
            previous = json.loads(frozen)
            assert seen[0] == (previous["attempt"], previous["request"].encode())
        [run_record] = executions(record)
        assert run_record.accepted.completion.outcome == Outcome(
            "inconclusive", "answer", ("result.json",), ("synthetic",)
        )
        assert len([charge for charge in charges(record) if charge.usage.model_calls]) == 1
        await Replayer(workflows=[CampaignRun]).replay_workflow(await handle.fetch_history())

    run(server, body)


def test_model_wall_occupancy_includes_worker_phases_and_excludes_close(server, tmp_path):
    async def body(client, name):
        async def preparing(work):
            await asyncio.sleep(0.4)
            return await prepare(work)

        async def interpreting(work, request, result):
            await asyncio.sleep(0.4)
            return answered_by(result)

        class SlowClose(Gateway):
            def close(self, lease):
                time.sleep(1.5)
                return super().close(lease)

        gateway = SlowClose(CallLog(tmp_path), [(POLICY, ScriptedProvider(["answer"]))])
        async with execution.worker(
            client,
            name,
            {},
            gateway,
            models={"ask": ModelBody(preparing, interpreting)},
            resolve=model_authority,
        ):
            handle = await execution.start(
                client,
                campaign(max_model_calls=1),
                (ModelJob("q", "ask", PROMPT, 0, "proposer", 1),),
                name,
            )
            record = await execution.record(handle)
        [wall] = [
            charge.usage.wall_seconds for charge in charges(record) if charge.usage.wall_seconds
        ]
        assert Decimal("0.8") <= wall < Decimal("1.5")
        history = await handle.fetch_history()
        assert len(events(history, "ACTIVITY_TASK_SCHEDULED")) == 4
        assert events(history, "WORKFLOW_EXECUTION_COMPLETED")

    run(server, body)


@pytest.mark.parametrize("boundary", ["model_result", "interpret_input"])
def test_oversized_stored_model_reply_still_closes_and_charges_without_resending(
    server, tmp_path, boundary
):
    async def body(client, name):
        reply = '"' * 614400 if boundary == "model_result" else "x" * 2095500
        provider = ScriptedProvider([reply])
        log = CallLog(tmp_path)
        gateway = Gateway(log, [(POLICY, provider)])
        async with execution.worker(
            client, name, {}, gateway, models=MODELS, resolve=model_authority
        ):
            handle = await execution.start(
                client,
                campaign(max_model_calls=1),
                (ModelJob("q", "ask", PROMPT, 0, "proposer", 1),),
                name,
            )
            record = await execution.record(handle)
        assert len(provider.calls) == 1
        assert log.reply(gateway.records[0]) == reply
        assert len([charge for charge in charges(record) if charge.usage.model_calls]) == 1
        assert executions(record)[0].attempts[0].completion is None
        assert record.interruptions[-1].reason == (
            "activity_failed" if boundary == "model_result" else "invalid_payload"
        )
        history = await handle.fetch_history()
        assert len([item for item in scheduled(history) if item.startswith("close/")]) == 1
        assert not events(history, "WORKFLOW_TASK_FAILED")

    run(server, body)


def test_a_large_steer_and_argument_refuse_only_the_oversized_preparation_input(server, tmp_path):
    async def body(client, name):
        revision = campaign(max_model_calls=1)
        child = revise(
            revision,
            replace(revision.specification, question="x" * 1100000),
            author="researcher",
            at=day(1),
            reason="Use the larger question",
        )
        job = ModelJob("q", "ask", "x" * 1100000, 0, "proposer", 1)
        jobs = (Job("hold", "hand_off", "", 0, external=True), job)
        converter = DataConverter.default.payload_converter
        plan = execution._plan_v2(file_bytes(revision).decode(), jobs)
        assert Payloads(payloads=[converter.to_payload(plan)]).ByteSize() < 2097152
        assert (
            Payloads(
                payloads=[
                    converter.to_payload(file_bytes(child).decode()),
                    converter.to_payload(day(2).isoformat().replace("+00:00", "Z")),
                ]
            ).ByteSize()
            < 2097152
        )
        task = execution._PrepareTask(
            uuid4(), job.body, job.argument, file_bytes(child).decode(), job.role
        )
        assert Payloads(payloads=[converter.to_payload(task)]).ByteSize() > 2097152
        provider = ScriptedProvider([])
        gateway = Gateway(CallLog(tmp_path), [(POLICY, provider)])
        async with execution.worker(
            client, name, BODIES, gateway, models=MODELS, resolve=model_authority
        ):
            handle = await execution.start(client, revision, jobs, name)
            await started(handle, "hold/1")
            await execution.steer(handle, child, at=day(2))
            await client.get_async_activity_handle(
                workflow_id=handle.id, run_id=handle.run_id, activity_id="hold/1"
            ).complete(report("Ready"))
            record = await execution.record(handle)
        assert record.status == "completed"
        assert [(item.phase, item.reason) for item in record.interruptions] == [
            ("prepare", "invalid_payload")
        ]
        assert provider.calls == []
        history = await handle.fetch_history()
        assert scheduled(history) == ["hold/1"]
        assert not events(history, "WORKFLOW_TASK_FAILED")
        ledger = load_ledger(record.ledger.encode())
        assert len([event for event in ledger.events if isinstance(event, Reserved)]) == 2
        assert len([event for event in ledger.events if isinstance(event, Released)]) == 2
        assert all(line.outcome != "open" for line in ledger.report())
        assert executions(record)[1].attempts[0].completion is None

    run(server, body)


def test_a_worker_killed_mid_attempt_recovers_on_another_worker(server):
    async def body(client, name):
        # The verify gate seals its environment, so the child must not write bytecode into it.
        bootstrap = (
            f"import runpy, sys; sys.path[:] = {sys.path!r}; "
            "from quant_research.execution import run_worker; "
            f"run_worker({server!r}, {name!r}, runpy.run_path({__file__!r})['BODIES'])"
        )
        killed = subprocess.Popen([sys.executable, "-B", "-I", "-S", "-c", bootstrap])
        try:
            handle = await execution.start(
                client, campaign(attempts=2), (Job("slow", "answer", "4", 3),), name
            )
            await started(handle, "slow/1")
        finally:
            killed.kill()
            killed.wait()
        async with execution.worker(client, name, BODIES):
            record = await execution.record(handle)
        history = await handle.fetch_history()
        assert record.status == "completed"
        assert links(record) == [("slow/1", "timed_out"), ("slow/2", "completed")]
        assert record.spent == 6
        [slow] = executions(record)
        first, second = slow.attempts
        assert first.completion is None
        assert second.submission.predecessor == first.submission.attempt_id
        assert slow.accepted == second
        assert [attempt.submission.attempt_id for attempt in slow.attempts] == [
            link.attempt_id for link in record.attempts
        ]
        assert outcomes(history) == [("slow/1", "TIMED_OUT"), ("slow/2", "COMPLETED")]
        [timeout] = events(history, "ACTIVITY_TASK_TIMED_OUT")
        assert (
            timeout.failure.timeout_failure_info.timeout_type == TimeoutType.TIMEOUT_TYPE_HEARTBEAT
        )
        assert len(events(history, "WORKFLOW_EXECUTION_COMPLETED")) == 1
        assert record.run_id == handle.result_run_id == (await handle.describe()).run_id

    run(server, body)


def test_a_dispatch_limit_of_three_refuses_the_fourth_dispatch_before_it_runs(server, tmp_path):
    async def body(client, name):
        provider = ScriptedProvider(["one", "two", "three", "four"])
        router = Gateway(CallLog(tmp_path), [(POLICY, provider)])
        jobs = tuple(ModelJob(f"q{number}", "ask", PROMPT, 0, "proposer", 1) for number in range(4))
        async with execution.worker(
            client, name, BODIES, router, models=MODELS, resolve=model_authority
        ):
            handle = await execution.start(client, campaign(max_model_calls=3), jobs, name)
            record = await execution.record(handle)
        assert record.status == "budget_exhausted"
        assert [call.texts for call in provider.calls] == [(ROLE_PROMPT, PROMPT)] * 3
        dispatched = [charge for charge in charges(record) if charge.usage.model_calls]
        assert [(charge.id, charge.usage.model_calls) for charge in dispatched] == [
            (sent.call_id, 1) for sent in router.records
        ]
        assert len(dispatched) == 3
        assert [link.phase for link in record.attempts] == ["prepare", "model", "interpret"] * 3
        assert [run.accepted.completion.outcome.detail for run in executions(record)] == [
            "one",
            "two",
            "three",
        ]

    run(server, body)


@pytest.mark.parametrize("boundary", ["reply", "claim"])
def test_a_lost_model_activity_recovers_its_reply_or_claim_without_resending(
    server, tmp_path, boundary
):
    calls = tmp_path / "calls"
    calls.mkdir()

    async def body(client, name):
        bootstrap = (
            f"import runpy, sys; sys.path[:] = {sys.path!r}; "
            f"runpy.run_path({__file__!r})['serve_model']"
            f"({server!r}, {name!r}, {str(calls)!r}, {boundary!r})"
        )
        killed = subprocess.Popen([sys.executable, "-B", "-I", "-S", "-c", bootstrap])
        try:
            job = ModelJob("q", "ask", PROMPT, 0, "proposer", 1)
            handle = await execution.start(client, campaign(max_model_calls=1), (job,), name)
            kind = CallRecord if boundary == "reply" else Dispatched
            deadline = time.monotonic() + 30
            while not any(isinstance(entry, kind) for entry in CallLog(calls).load()):
                assert time.monotonic() < deadline, "the killed worker never stored its boundary"
                await asyncio.sleep(0.05)
        finally:
            killed.kill()
            killed.wait()
        restarted = ScriptedProvider([])
        gateway = Gateway(CallLog(calls), [(POLICY, restarted)])
        async with execution.worker(
            client, name, BODIES, gateway, models=MODELS, resolve=model_authority
        ):
            record = await execution.record(handle)
        assert restarted.calls == []
        assert [(link.phase, link.ended) for link in record.attempts] == [
            ("prepare", "completed"),
            ("model", "timed_out"),
            ("model", "completed"),
            ("interpret", "completed"),
        ]
        assert len({link.attempt_id for link in record.attempts}) == 1
        [claim] = [entry for entry in CallLog(calls).load() if isinstance(entry, Dispatched)]
        [saved] = gateway.records
        assert claim.fence == saved.fence == 1
        assert saved.outcome == ("completed" if boundary == "reply" else "unknown")
        assert [(c.id, c.usage) for c in charges(record) if c.usage.model_calls] == [
            (saved.call_id, saved.usage)
        ]
        [q] = executions(record)
        if boundary == "reply":
            assert q.accepted.completion.outcome == Outcome("successful", "answer")
        else:
            assert q.accepted is None
            assert q.attempts[0].completion.outcome == Outcome("failed", "unknown")
            assert saved.usage == SEALED
        await Replayer(workflows=[CampaignRun]).replay_workflow(await handle.fetch_history())

    run(server, body)


def test_a_rate_limited_attempt_spends_no_failure_slot_and_retries_under_new_ids(server, tmp_path):
    async def body(client, name):
        router = Gateway(CallLog(tmp_path), [(POLICY, ScriptedProvider([Fault.RATE_LIMIT, "ok"]))])
        jobs = (ModelJob("q", "ask", PROMPT, 0, "proposer", 1),)
        revision = campaign(attempts=1, max_model_calls=2)
        async with execution.worker(
            client, name, BODIES, router, models=MODELS, resolve=model_authority
        ):
            record = await execution.record(await execution.start(client, revision, jobs, name))
        [q] = executions(record)
        limited, answered = q.attempts
        assert limited.completion.outcome == Outcome("inconclusive", "rate_limited: rate_limited")
        assert q.accepted == answered
        first, second = router.records
        assert (first.outcome, second.outcome) == ("rate_limited", "completed")
        assert first.call_id != second.call_id and first.attempt_id != second.attempt_id
        ledger = load_ledger(record.ledger.encode())
        released = [event.outcome for event in ledger.events if isinstance(event, Released)]
        assert released == ["inconclusive", "successful"]

    run(server, body)


def test_wall_time_is_reserved_per_model_attempt_and_charged_as_it_elapses(server, tmp_path):
    async def body(client, name):
        router = Gateway(CallLog(tmp_path), [(POLICY, ScriptedProvider(["one", "two"]))])
        jobs = (
            ModelJob("slow", "ask_slowly", PROMPT, 0, "proposer", 1),
            ModelJob("next", "ask", PROMPT, 0, "proposer", 1),
        )
        revision = campaign(max_model_calls=2, max_wall_seconds=601)
        async with execution.worker(
            client, name, BODIES, router, models=MODELS, resolve=model_authority
        ):
            handle = await execution.start(client, revision, jobs, name)
            record = await execution.record(handle)
        assert record.status == "budget_exhausted"
        ledger = load_ledger(record.ledger.encode())
        [reserved] = [event for event in ledger.events if isinstance(event, Reserved)]
        assert reserved.estimate == Usage(wall_seconds=Decimal(600), model_calls=1)
        [wall] = [
            charge.usage.wall_seconds for charge in charges(record) if charge.usage.wall_seconds
        ]
        assert Decimal("1.5") <= wall < Decimal(600)
        assert [link.phase for link in record.attempts] == ["prepare", "model", "interpret"]

    run(server, body)


def test_start_refuses_dispatches_that_the_revision_cannot_hold(server):
    async def body(client, name):
        asking = Job("q", "ask", PROMPT, 0, dispatches=1)
        cases = [
            (campaign(), asking, "campaign's 0 model calls"),
            (campaign(max_model_calls=1), replace(asking, dispatches=2), "campaign's 1 model"),
            (campaign(max_model_calls=1), replace(asking, dispatches=True), "nonnegative"),
            (campaign(max_model_calls=1), replace(asking, external=True), "external job"),
            (campaign(max_model_calls=1, max_wall_seconds=599), asking, "max_wall_seconds"),
            (campaign(max_model_calls=17), replace(asking, dispatches=17), "at most 16 models"),
        ]
        for revision, job, refusal in cases:
            with pytest.raises(RunnerError, match=refusal):
                await execution.start(client, revision, (job,), name)

    run(server, body)


@pytest.mark.parametrize("kind", ["compute", "chain"])
def test_unreserved_model_sends_are_refused_before_claiming_a_call(server, tmp_path, kind):
    async def body(client, name):
        provider = ScriptedProvider([])
        gateway = Gateway(CallLog(tmp_path), [(POLICY, provider)])

        def resolve(revision):
            result = model_authority(revision)
            result.roles = SimpleNamespace(
                assignments=(replace(PROPOSER, fallbacks=(PROPOSER.primary,)),)
            )
            return result

        job = (
            Job("q", "legacy", PROMPT, 0)
            if kind == "compute"
            else ModelJob("q", "ask", PROMPT, 0, "proposer", 1)
        )
        async with execution.worker(
            client, name, {"legacy": ask}, gateway, models=MODELS, resolve=resolve
        ):
            handle = await execution.start(client, campaign(max_model_calls=1), (job,), name)
            record = await execution.record(handle)
        assert provider.calls == []
        assert CallLog(tmp_path).load() == ()
        assert not any(charge.usage.model_calls for charge in charges(record))
        assert not any(
            item.startswith("close/") for item in scheduled(await handle.fetch_history())
        )
        assert executions(record)[0].accepted is None
        if kind == "chain":
            assert record.interruptions[0].reason == "invalid_preparation"

    run(server, body)


def test_a_body_cannot_send_under_a_lease_of_its_own(tmp_path):
    async def smuggle(revision: CampaignRevision, argument: str, dispatch) -> Outcome:
        request = Request((Material(argument, CLEARED),))
        stray = Lease(uuid4(), (uuid4(),), 0)
        return answered_by(
            dispatch(CONTEXT, PROPOSER, request, lease=stray, cancel=threading.Event())
        )

    provider = ScriptedProvider(["sent under the stray lease"])
    gateway = Gateway(CallLog(tmp_path), [(POLICY, provider)])
    attempts = execution._Attempts({"smuggle": smuggle}, gateway)
    lease = Lease(uuid4(), (uuid4(),), 0)
    task = execution.Task("smuggle", PROMPT, file_bytes(campaign()).decode(), lease)
    report = asyncio.run(ActivityEnvironment().run(attempts.attempt, task))
    assert report.outcome.status == "failed"
    assert report.outcome.detail.startswith("TypeError(")
    assert provider.calls == []


def test_exact_legacy_joins_and_new_model_identity_refuse_conflicting_inputs(server):
    async def body(client, name):
        revision = campaign(max_model_calls=1)
        old = Job("q", "ask", PROMPT, 0, dispatches=1)
        plan = execution.Plan(file_bytes(revision).decode(), (old,))
        digest = hashlib.sha256(
            canonical_bytes({"revision": plan.revision, "jobs": [old.__dict__]})
        ).hexdigest()
        original = await client.start_workflow(
            CampaignRun.run,
            plan,
            id=str(revision.campaign_id),
            task_queue=name,
            memo={"plan": digest},
        )
        joined = await execution.start(client, revision, (old,), name)
        assert joined.run_id == original.result_run_id
        with pytest.raises(RunnerError, match="different plan"):
            await execution.start(client, revision, (replace(old, argument="changed"),), name)
        with pytest.raises(RunnerError, match="ModelJob"):
            await execution.start(
                client, revision, (old, ModelJob("new", "ask", PROMPT, 0, "proposer", 1)), name
            )
        await original.terminate("Identity test complete")
        with pytest.raises(RunnerError, match="closed"):
            await execution.start(client, revision, (old,), name)
        fresh = campaign(max_model_calls=1)
        with pytest.raises(RunnerError, match="ModelJob"):
            await execution.start(client, fresh, (old,), name)
        missing = await client.start_workflow(
            CampaignRun.run,
            execution.Plan(file_bytes(fresh).decode(), (old,)),
            id=str(fresh.campaign_id),
            task_queue=name,
        )
        with pytest.raises(RunnerError, match="different plan"):
            await execution.start(client, fresh, (old,), name)
        await missing.terminate("Identity test complete")
        revision = campaign(max_model_calls=1)
        job = ModelJob("q", "ask", PROMPT, 0, "proposer", 1)
        first = await execution.start(client, revision, (job,), name)
        same = await execution.start(client, revision, (job,), name)
        assert same.run_id == first.run_id
        for changed in (
            replace(job, body="other.v1"),
            replace(job, argument="other"),
            replace(job, role="critic"),
            replace(job, schema=(("answer", str),)),
        ):
            with pytest.raises(RunnerError, match="different plan"):
                await execution.start(client, revision, (changed,), name)
        await first.terminate("Identity test complete")

    run(server, body)


@pytest.mark.parametrize("mode", ["fallback", "subscription", "inexact"])
def test_model_receipts_keep_fallbacks_subscription_usage_and_inexact_costs(server, tmp_path, mode):
    async def body(client, name):
        class Costly(ScriptedProvider):
            def complete(self, request, cancel):
                self.calls.append(request)
                return Replied("answer", Usage(cost_usd=Decimal("9E+999999")))

        assignment = (
            replace(PROPOSER, fallbacks=(replace(PROPOSER.primary, model="model-b"),))
            if mode == "fallback"
            else PROPOSER
        )

        def resolve(revision):
            result = model_authority(revision)
            result.roles = SimpleNamespace(assignments=(assignment,))
            return result

        provider = (
            Costly([])
            if mode == "inexact"
            else ScriptedProvider(
                [Fault.RATE_LIMIT, "answer"] if mode == "fallback" else ["answer"],
                usd_per_token=None if mode == "subscription" else Decimal("0.000002"),
            )
        )
        gateway = Gateway(CallLog(tmp_path), [(POLICY, provider)])
        jobs = tuple(
            ModelJob(f"q{number}", "ask", PROMPT, 0, "proposer", 2 if mode == "fallback" else 1)
            for number in range(2 if mode == "inexact" else 1)
        )
        async with execution.worker(client, name, {}, gateway, models=MODELS, resolve=resolve):
            handle = await execution.start(client, campaign(max_model_calls=2), jobs, name)
            record = await execution.record(handle)
        receipts = [charge for charge in charges(record) if charge.usage.model_calls]
        assert len(receipts) == (1 if mode == "subscription" else 2)
        assert all(item.accepted is not None for item in executions(record))
        if mode == "fallback":
            assert [item.outcome for item in gateway.records] == ["rate_limited", "completed"]
            assert [charge.id for charge in receipts] == [item.call_id for item in gateway.records]
        elif mode == "subscription":
            assert receipts[0].usage.cost_usd == 0
            assert receipts[0].usage.unreported == frozenset({"input_tokens", "output_tokens"})
            assert all(
                item.subscription
                for item in CallLog(tmp_path).load()
                if isinstance(item, Dispatched)
            )
        else:
            assert receipts[0].usage.cost_usd == Decimal("9E+999999")
            assert receipts[1].usage.cost_usd == 0
            assert receipts[1].usage.unreported == frozenset({"cost_usd"})
            assert gateway.records[1].usage.cost_usd == Decimal("9E+999999")
        assert not events(await handle.fetch_history(), "WORKFLOW_TASK_FAILED")

    run(server, body)


def test_steering_pins_current_model_callbacks_and_changes_later_role_prompts(server, tmp_path):
    async def body(client, name):
        first = campaign(max_model_calls=2)
        child = revise(
            first,
            replace(first.specification, question=STEERED),
            author="researcher",
            at=day(1),
            reason="Change the next role",
        )
        prepared, proceed = asyncio.Event(), asyncio.Event()
        seen = []

        def resolve(revision):
            result = model_authority(revision)
            result.roles = SimpleNamespace(
                assignments=(
                    replace(
                        PROPOSER,
                        primary=replace(
                            PROPOSER.primary,
                            model="old" if revision.revision_id == first.revision_id else "new",
                        ),
                    ),
                )
            )
            result.prompts = {"proposer": Material(revision.specification.question, CLEARED)}
            return result

        async def preparing(work):
            if work.argument == "first":
                prepared.set()
                await proceed.wait()
            seen.append(("prepare", work.revision.revision_id))
            return await prepare(work)

        async def interpreting(work, request, result):
            seen.append(("interpret", work.revision.revision_id))
            return answered_by(result)

        provider = ScriptedProvider(["first", "second"])
        gateway = Gateway(CallLog(tmp_path), [(POLICY, provider)])
        async with execution.worker(
            client,
            name,
            {},
            gateway,
            models={"ask": ModelBody(preparing, interpreting)},
            resolve=resolve,
        ):
            jobs = tuple(
                ModelJob(value, "ask", value, 0, "proposer", 1) for value in ("first", "second")
            )
            handle = await execution.start(client, first, jobs, name)
            await prepared.wait()
            await execution.steer(handle, child, at=day(2))
            proceed.set()
            record = await execution.record(handle)
        assert seen == [
            ("prepare", first.revision_id),
            ("interpret", first.revision_id),
            ("prepare", child.revision_id),
            ("interpret", child.revision_id),
        ]
        assert [call.texts for call in provider.calls] == [(QUESTION, "first"), (STEERED, "second")]
        assert [item.model.model for item in gateway.records] == ["old", "new"]
        assert [link.revision for link in record.attempts] == [first.revision_id] * 3 + [
            child.revision_id
        ] * 3

    run(server, body)


def test_a_repeated_start_joins_the_running_campaign_and_a_closed_one_refuses(server, tmp_path):
    async def body(client, name):
        revision = campaign(trials=1)
        with pytest.raises(RunnerError, match="exceed"):
            await execution.start(
                client, revision, (Job("a", "answer", "0", 1), Job("b", "answer", "0", 1)), name
            )
        for jobs in ((Job("a", " ", "0", 1),), (Job("a" * 101, "answer", "0", 1),)):
            with pytest.raises(RunnerError, match="must match"):
                await execution.start(client, revision, jobs, name)
        with pytest.raises(PackageError, match="UTF-8"):
            await execution.start(client, revision, (Job("a", "answer", "\udcff", 1),), name)
        with pytest.raises(PackageError, match="reason"):
            await execution.start(
                client, replace(revision, reason=" "), (Job("a", "answer", "0", 1),), name
            )
        for cost in (-1, 0.5, True):
            with pytest.raises(RunnerError, match="nonnegative integer"):
                await execution.start(client, revision, (Job("a", "answer", "0", cost),), name)
        argument = {
            "fixture": str(FIXTURE),
            "destination": str(tmp_path / "packages"),
            "lock": "synthetic lock\n",
        }
        jobs = (Job("replay", "replay", json.dumps(argument), 1),)
        first = await execution.start(client, revision, jobs, name)
        second = await execution.start(client, revision, jobs, name)
        assert second.result_run_id == first.result_run_id
        with pytest.raises(RunnerError, match="different plan"):
            await execution.start(client, revision, (Job("other", "answer", "0", 1),), name)
        async with execution.worker(client, name, BODIES):
            record = await execution.record(first)
        with pytest.raises(WorkflowAlreadyStartedError):
            await execution.start(client, revision, jobs, name)
        assert record.status == "completed"
        assert record.workflow_id == str(revision.campaign_id)
        [replayed] = executions(record)
        outcome = replayed.accepted.completion.outcome
        package = open_package(Path(outcome.detail.rsplit(" at ", 1)[1]))
        assert f"; package {package.digest} at " in outcome.detail
        assert package.record.selected_trial.specification.question == QUESTION
        assert outcome == replace(
            package.record.selected_trial.attempts[-1].outcome, detail=outcome.detail
        )
        assert outcome.status == "successful"

    run(server, body)


def test_start_refuses_a_running_research_loop_and_leaves_its_owner_running(
    temporal, tmp_path, research_setup
):
    from quant_research import research

    revision = test_research.pilot(research_setup, "checkpoint before conclude")

    async def body(run):
        first = await research.start(run.client, revision, run.queue)
        await test_research.reaches(first, "paused")
        with pytest.raises(RunnerError, match="different plan"):
            await execution.start(run.client, revision, (), run.queue)
        again = await research.start(run.client, revision, run.queue)
        assert again.result_run_id == first.result_run_id
        assert (await research.status(first)).state == "paused"
        await research.control(first, "resume", actor="researcher", reason="Continue the loop")
        assert (await first.result()).outcome == "stopped"

    test_research.serve(temporal, tmp_path, research_setup, [test_research.STOP], body)


def test_a_duplicate_or_late_completion_report_is_not_accepted_twice(server):
    async def body(client, name):
        jobs = (Job("batch", "hand_off", "", 1, external=True),)
        async with execution.worker(client, name, BODIES):
            handle = await execution.start(client, campaign(attempts=4), jobs, name)
            batch = {
                number: client.get_async_activity_handle(
                    workflow_id=handle.id,
                    run_id=handle.result_run_id,
                    activity_id=f"batch/{number}",
                )
                for number in (1, 2, 3, 4)
            }
            await started(handle, "batch/1")
            await batch[1].fail(ApplicationError("node lost"))
            await started(handle, "batch/2")
            west = timezone(-timedelta(hours=1))
            await batch[2].complete(
                Report(Outcome("successful", "x"), datetime.max.replace(tzinfo=west))
            )
            await started(handle, "batch/3")
            digits = b'{"outcome":' + b"9" * 5001 + b"}"
            await batch[3].complete(
                RawValue(Payload(metadata={"encoding": b"json/plain"}, data=digits))
            )
            await started(handle, "batch/4")
            await batch[4].complete(report("first report"))
            with pytest.raises(RPCError) as late:
                await batch[1].complete(report("late report"))
            with pytest.raises(RPCError) as repeated:
                await batch[4].complete(report("second report"))
            record = await execution.record(handle)
        assert late.value.status == RPCStatusCode.NOT_FOUND
        assert repeated.value.status == RPCStatusCode.NOT_FOUND
        assert links(record) == [
            ("batch/1", "failed"),
            ("batch/2", "completed"),
            ("batch/3", "completed"),
            ("batch/4", "completed"),
        ]
        log = load_log(record.acceptance.encode())
        [accepted] = [event for event in log.events if isinstance(event, Accepted)]
        [batch_run] = log.executions().values()
        lost, *malformed, reported = batch_run.attempts
        assert lost.completion is None
        for refused in malformed:
            assert refused.completion is None
        assert [item.reason for item in record.interruptions].count("invalid_result") == 2
        assert accepted.attempt_id == reported.submission.attempt_id
        assert reported.completion.outcome == Outcome("successful", "first report")
        assert outcomes(await handle.fetch_history()) == [
            ("batch/1", "FAILED"),
            ("batch/2", "COMPLETED"),
            ("batch/3", "COMPLETED"),
            ("batch/4", "COMPLETED"),
        ]

    run(server, body)


def test_a_failure_report_with_undecodable_details_fails_only_its_attempt(server):
    async def body(client, name):
        jobs = (Job("batch", "hand_off", "", 1, external=True),)
        async with execution.worker(client, name, BODIES):
            handle = await execution.start(client, campaign(attempts=2), jobs, name)
            batch = {
                number: client.get_async_activity_handle(
                    workflow_id=handle.id,
                    run_id=handle.result_run_id,
                    activity_id=f"batch/{number}",
                )
                for number in (1, 2)
            }
            unknown = RawValue(Payload(metadata={"encoding": b"binary/unknown"}, data=b"?"))
            lost = ApplicationError("node lost", unknown)
            lost.__cause__ = ApplicationError("disk failed", unknown)
            await started(handle, "batch/1")
            await batch[1].fail(lost)
            await started(handle, "batch/2")
            await batch[2].complete(report("second report"))
            record = await execution.record(handle)
        assert links(record) == [("batch/1", "failed"), ("batch/2", "completed")]
        [batch_run] = executions(record)
        lost_attempt, reported = batch_run.attempts
        assert lost_attempt.completion is None
        assert batch_run.accepted == reported

    run(server, body)


def test_a_record_over_the_payload_limit_reads_back_in_parts(server):
    async def body(client, name):
        jobs = tuple(Job(f"bulk{number}", "bulky", "60000", 1) for number in range(40))
        async with execution.worker(client, name, BODIES):
            handle = await execution.start(client, campaign(trials=40), jobs, name)
            summary = await handle.result()
            record = await execution.record(handle)
        history = await handle.fetch_history()
        assert summary == Summary("completed", 40)
        assert len(record.acceptance.encode()) > 2**21
        assert [run.accepted.completion.outcome.detail for run in executions(record)] == [
            "x" * 60000
        ] * 40
        assert max(event.ByteSize() for event in history.events) < 2**17

    run(server, body)


def test_pause_annotate_and_steer_apply_to_the_attempts_scheduled_after_them(server):
    async def body(client, name):
        root = campaign()
        child = revise(
            root,
            replace(root.specification, question=STEERED),
            author="researcher",
            at=day(1),
            reason="Ask about the delayed association",
        )
        wider = revise(
            root,
            replace(root.specification, limits=ResourceLimits(5, 1, 1000)),
            author="researcher",
            at=day(1),
            reason="Raise the compute limit",
        )
        jobs = (Job("a", "answer", "3", 1), Job("b", "answer", "0", 1))
        async with execution.worker(client, name, BODIES):
            handle = await execution.start(client, root, jobs, name)
            await started(handle, "a/1")
            paused = await execution.control(
                handle, "pause", actor="researcher", at=day(2), reason="Check the inputs"
            )
            assert paused == "paused"
            await execution.control(
                handle, "annotate", actor="researcher", at=day(3), reason="Inputs look complete"
            )
            assert await execution.steer(handle, child, at=day(4)) == "paused"
            with pytest.raises(WorkflowUpdateFailedError):
                await execution.steer(handle, wider, at=day(4))
            while not outcomes(await handle.fetch_history()):
                await asyncio.sleep(0.1)
            assert (await handle.query(CampaignRun.summary)).status == "paused"
            assert scheduled(await handle.fetch_history()) == ["a/1"]
            await execution.control(
                handle, "resume", actor="researcher", at=day(5), reason="Inputs checked"
            )
            record = await execution.record(handle)
        assert record.status == "completed"
        assert [(link.activity_id, link.revision) for link in record.attempts] == [
            ("a/1", root.revision_id),
            ("b/1", child.revision_id),
        ]
        a, b = executions(record)
        assert a.accepted.completion.outcome.detail == QUESTION
        assert b.accepted.completion.outcome.detail == STEERED
        steering = json.loads(record.steering)["events"]
        assert [event["action"] for event in steering] == ["pause", "annotate", "steer", "resume"]
        assert steering[2]["revision"] == child.revision_id

    run(server, body)


def test_cancel_stops_the_running_attempt_and_is_recorded(server):
    async def body(client, name):
        async with execution.worker(client, name, BODIES):
            handle = await execution.start(
                client, campaign(), (Job("long", "answer", "60", 3),), name
            )
            await started(handle, "long/1")
            assert await handle.query(CampaignRun.summary) == Summary("running", 3)
            cancelled = await execution.control(
                handle, "cancel", actor="researcher", at=day(1), reason="Stop the campaign"
            )
            assert cancelled == "cancelled"
            record = await execution.record(handle)
        history = await handle.fetch_history()
        assert record.status == "cancelled"
        assert links(record) == [("long/1", "cancelled")]
        assert outcomes(history) == [("long/1", "CANCELED")]
        assert len(events(history, "ACTIVITY_TASK_CANCEL_REQUESTED")) == 1
        assert len(events(history, "WORKFLOW_EXECUTION_COMPLETED")) == 1
        assert json.loads(record.steering)["events"][-1]["action"] == "cancel"
        [long] = executions(record)
        [attempt] = long.attempts
        assert attempt.completion is None
        assert long.decisions == ()

    run(server, body)


def test_a_cancel_does_not_wait_for_an_external_reporter(server):
    async def body(client, name):
        jobs = (Job("batch", "hand_off", "", 1, external=True),)
        async with execution.worker(client, name, BODIES):
            handle = await execution.start(client, campaign(), jobs, name)
            await started(handle, "batch/1")
            await execution.control(
                handle, "cancel", actor="researcher", at=day(1), reason="Stop the campaign"
            )
            record = await asyncio.wait_for(execution.record(handle), 10)
        assert record.status == "cancelled"
        assert links(record) == [("batch/1", None)]
        assert record.attempts[0].cancel_requested_at is not None
        with pytest.raises(RPCError) as late:
            await client.get_async_activity_handle(
                workflow_id=handle.id, run_id=handle.result_run_id, activity_id="batch/1"
            ).complete(report("late report"))
        assert late.value.status == RPCStatusCode.NOT_FOUND

    run(server, body)


def test_the_budget_is_charged_before_scheduling_and_failed_attempts_are_charged(server):
    async def body(client, name):
        jobs = (
            Job("a", "answer", "0", 2),
            Job("b", "crash", "disk\x00full", 2),
            Job("c", "answer", "0", 2),
        )
        async with execution.worker(client, name, BODIES):
            handle = await execution.start(client, campaign(attempts=2, seconds=5), jobs, name)
            record = await execution.record(handle)
        assert record.status == "budget_exhausted"
        assert record.spent == 4
        assert scheduled(await handle.fetch_history()) == ["a/1", "b/1"]
        a, b = executions(record)
        assert a.accepted is not None
        [failed] = b.attempts
        assert failed.completion.outcome == Outcome("failed", "RuntimeError('disk\\x00full')")
        assert b.decisions == ()

    run(server, body)


def test_a_job_that_uses_its_attempts_stays_unaccepted_and_the_campaign_continues(server):
    async def body(client, name):
        jobs = (
            Job("garbled", "garbled", "", 1),
            Job("unrecordable", "unrecordable", "", 1),
            Job("good", "answer", "0", 1),
        )
        async with execution.worker(client, name, BODIES):
            handle = await execution.start(client, campaign(attempts=2), jobs, name)
            record = await execution.record(handle)
        assert record.status == "completed"
        assert scheduled(await handle.fetch_history()) == [
            "garbled/1",
            "garbled/2",
            "unrecordable/1",
            "unrecordable/2",
            "good/1",
        ]
        *bad, good = executions(record)
        for exhausted in bad:
            assert all(attempt.completion is None for attempt in exhausted.attempts)
            assert exhausted.decisions == ()
        assert [item.reason for item in record.interruptions] == ["invalid_result"] * 4
        assert good.accepted is not None

    run(server, body)


def signal_job(probe, name: str, body: str, source: str) -> Job:
    prelude = (
        "import json\n"
        "KEYS = [row['key'] for row in json.load(open('/job/slice.json'))['rows']]\n"
        "def write(text):\n"
        "    open('/output/signal.json', 'w').write(text)\n"
    )
    return Job(name, body, json.dumps({"source": prelude + source, "dataset": probe.name}), 1)


SIGNAL = "write(json.dumps({k: float(len(k)) for k in KEYS}))\n"


def test_generated_code_stays_off_unless_the_specification_enables_it(server, tmp_path, probe):
    async def body(client, name):
        revision = campaign(attempts=2)
        bodies = {
            "signal": execution.generated(
                probe.catalog(), tmp_path, probe.context, RunLog(tmp_path), tmp_path
            )
        }
        plan = (signal_job(probe, "signal", "signal", "write('{}')\n"),)
        async with execution.worker(client, name, bodies):
            handle = await execution.start(client, revision, plan, name)
            record = await execution.record(handle)
        [signal] = executions(record)
        refusal = f"RunnerError('Revision {revision.revision_id} does not enable generated code')"
        assert [attempt.completion.outcome for attempt in signal.attempts] == [
            Outcome("failed", refusal)
        ] * 2
        assert signal.decisions == ()
        assert list(tmp_path.iterdir()) == []

    run(server, body)


def test_a_refused_read_fails_the_attempt_and_logs_no_run(server, tmp_path, probe):
    async def body(client, name):
        revision = campaign(attempts=2, generated_code=True)
        catalog = probe.catalog(licensed("vendor"))
        bodies = {
            "signal": execution.generated(
                catalog, tmp_path, probe.context, RunLog(tmp_path), tmp_path
            )
        }
        plan = (signal_job(probe, "signal", "signal", "write('{}')\n"),)
        async with execution.worker(client, name, bodies):
            handle = await execution.start(client, revision, plan, name)
            record = await execution.record(handle)
        [signal] = executions(record)
        refusal = Outcome("failed", "unknown_dataset: No dataset this run may read has this name")
        assert [attempt.completion.outcome for attempt in signal.attempts] == [refusal] * 2
        assert signal.decisions == ()
        assert list(tmp_path.iterdir()) == []

    run(server, body)


def test_generated_code_runs_in_the_worker_and_its_refusals_are_typed(
    server, tmp_path, image, probe
):
    store = tmp_path / "store"
    stored = probe.save(store, ["A", "BB"])
    signals = tmp_path / "signals"
    signals.mkdir()
    runs = tmp_path / "runs"
    runs.mkdir()
    log = RunLog(runs)

    async def body(client, name):
        catalog = probe.catalog(vintage=stored)
        bodies = {
            "signal": execution.generated(catalog, store, probe.context, log, signals),
            "brief": execution.generated(
                catalog, store, probe.context, log, signals, Limits(timeout=1.0)
            ),
        }
        plan = (
            signal_job(probe, "good", "signal", SIGNAL),
            signal_job(probe, "malformed", "signal", "write('not json')\n"),
            signal_job(probe, "slow", "brief", "import time\ntime.sleep(30)\n"),
        )
        revision = campaign(attempts=2, generated_code=True)
        async with execution.worker(client, name, bodies):
            handle = await execution.start(client, revision, plan, name)
            record = await execution.record(handle)
        good, malformed, slow = executions(record)
        runs = log.load()
        assert {run.record.image for run in runs} == {image}
        signal, rejected = (
            next(run for run in runs if run.outcome == kind) for kind in ("signal", "malformed")
        )
        [path] = signals.iterdir()
        assert path.name == f"{signal.output}.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == signal.output
        assert json.loads(path.read_bytes()) == {"A": 1.0, "BB": 2.0}
        assert good.accepted.completion.outcome == Outcome(
            "successful", f"signal {signal.output} at {path}; run {signal.run_id}"
        )
        malformed_message = "The signal is not a JSON object of numbers"
        assert malformed.accepted.completion.outcome == Outcome(
            "rejected", f"malformed: {malformed_message}; run {rejected.run_id}"
        )
        timeout_message = "The candidate ran past the wall-clock limit"
        assert sorted(attempt.completion.outcome.detail for attempt in slow.attempts) == sorted(
            f"timeout: {timeout_message}; run {run.run_id}"
            for run in runs
            if run.outcome == "timeout"
        )
        assert {attempt.completion.outcome.status for attempt in slow.attempts} == {"failed"}
        assert len(runs) == 4
        assert slow.decisions == ()

    run(server, body)


def test_a_steer_that_enables_generated_code_runs_it_in_later_attempts(
    server, tmp_path, image, probe
):
    store, signals, runs = tmp_path / "store", tmp_path / "signals", tmp_path / "runs"
    stored = probe.save(store, ["A", "BB"])
    signals.mkdir()
    runs.mkdir()
    log = RunLog(runs)
    root = campaign()
    child = revise(
        root,
        replace(root.specification, generated_code=True),
        author="researcher",
        at=day(1),
        reason="Run generated signal code",
    )

    async def body(client, name):
        signal = execution.generated(
            probe.catalog(vintage=stored), store, probe.context, log, signals
        )
        jobs = (
            signal_job(probe, "before", "signal", SIGNAL),
            Job("wait", "answer", "3", 1),
            signal_job(probe, "after", "signal", SIGNAL),
        )
        async with execution.worker(client, name, {"answer": answer, "signal": signal}):
            handle = await execution.start(client, root, jobs, name)
            await started(handle, "wait/1")
            await execution.steer(handle, child, at=day(2))
            record = await execution.record(handle)
        assert [(link.activity_id, link.revision) for link in record.attempts] == [
            ("before/1", root.revision_id),
            ("wait/1", root.revision_id),
            ("after/1", child.revision_id),
        ]
        before, _, after = executions(record)
        refusal = f"RunnerError('Revision {root.revision_id} does not enable generated code')"
        assert [attempt.completion.outcome for attempt in before.attempts] == [
            Outcome("failed", refusal)
        ]
        [logged] = log.load()
        path = signals / f"{logged.output}.json"
        assert json.loads(path.read_bytes()) == {"A": 1.0, "BB": 2.0}
        assert after.accepted.completion.outcome == Outcome(
            "successful", f"signal {logged.output} at {path}; run {logged.run_id}"
        )

    run(server, body)


@activity.defn
async def fail_before_third_attempt() -> int:
    attempt = activity.info().attempt
    if attempt < 3:
        raise ApplicationError(f"server attempt {attempt} failed")
    return attempt


@workflow.defn(sandboxed=False)
class ServerRetry:
    @workflow.run
    async def run(self) -> int:
        return await workflow.execute_activity(
            fail_before_third_attempt,
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(milliseconds=100), maximum_attempts=3
            ),
        )


def test_server_retries_leave_only_the_last_attempt_in_history(server):
    async def body(client, name):
        async with Worker(
            client, task_queue=name, workflows=[ServerRetry], activities=[fail_before_third_attempt]
        ):
            assert await client.execute_workflow(ServerRetry.run, id=name, task_queue=name) == 3
        history = await client.get_workflow_handle(name).fetch_history()
        attempts = events(history, "ACTIVITY_TASK_STARTED")
        assert [(item.attempt, item.last_failure.message) for item in attempts] == [
            (3, "server attempt 2 failed")
        ]
        kinds = [EventType.Name(event.event_type) for event in history.events]
        retries = kinds[kinds.index("EVENT_TYPE_ACTIVITY_TASK_SCHEDULED") :]
        assert retries[: retries.index("EVENT_TYPE_ACTIVITY_TASK_COMPLETED")] == [
            "EVENT_TYPE_ACTIVITY_TASK_SCHEDULED",
            "EVENT_TYPE_ACTIVITY_TASK_STARTED",
        ]
        text = history.to_json()
        assert "server attempt 2 failed" in text
        assert "server attempt 1 failed" not in text

    run(server, body)
