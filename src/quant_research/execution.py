"""Run a campaign's jobs in one Temporal workflow that owns its three logs.

ModelJob selects a pinned role and versioned ModelBody. Worker preparation returns restricted
materials. The workflow constructs the request, calls ask_model, charges its settled receipts,
and schedules worker interpretation into an Outcome. Both callbacks receive the same attempt
identity through ModelWork and may execute on different workers.

Job runs compute or external work through attempt(Task) -> Report. Its dispatching form remains
only for retained execution. One entry patch preserves those historical commands and Record
bytes. New runs expose phase observations and interruptions in RecordV2. Closed records are
read in bounded parts; Summary remains the workflow result and status query.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import re
import threading
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Literal
from uuid import UUID

from temporalio import activity, exceptions, workflow
from temporalio.api.common.v1 import Payload
from temporalio.api.failure.v1 import Failure
from temporalio.client import Client, WorkflowExecutionStatus, WorkflowHandle
from temporalio.common import (
    RawValue,
    RetryPolicy,
    WorkflowIDConflictPolicy,
    WorkflowIDReusePolicy,
)
from temporalio.converter import (
    DataConverter,
    DefaultFailureConverter,
    DefaultPayloadConverter,
    PayloadConverter,
)
from temporalio.exceptions import ActivityError, ApplicationError, is_cancelled_exception
from temporalio.service import RPCError, RPCStatusCode
from temporalio.worker import Worker
from temporalio.workflow import ActivityCancellationType

with workflow.unsafe.imports_passed_through():
    from quant_research import worker as container
    from quant_research.acceptance import AcceptanceLog, Accepted, Completed, Submitted, dump_log
    from quant_research.access import RunContext
    from quant_research.accounting import Budget, InexactUsage, Ledger, Usage, Work, dump_ledger
    from quant_research.activities import (
        Activities,
        Closing,
        ask_model,
        input_fits,
    )
    from quant_research.campaigns import (
        Authority,
        CampaignRevision,
        check_link,
        file_bytes,
        parse_file,
    )
    from quant_research.catalog import Catalog
    from quant_research.model_provenance import RoleAssignment, dump_assignment, load_assignment
    from quant_research.packages import Outcome, parse_outcome
    from quant_research.providers import (
        Answer,
        Gateway,
        Lease,
        Material,
        ProviderError,
        Receipt,
        Request,
        Schema,
        Unanswered,
        dump_material,
        dump_result,
        load_material,
        load_receipts,
        load_result,
    )
    from quant_research.records import canonical_bytes, format_time, parse_time
    from quant_research.replay import NewTrial, replay_fixture
    from quant_research.steering import SteeringLog, log_bytes

# Temporal's JSON converter cannot decode a PEP 695 alias, so these stay plain assignments.
Status = Literal["running", "paused", "cancelled", "completed", "budget_exhausted"]
Ending = Literal["completed", "failed", "timed_out", "cancelled"]

# A gateway call bound to the attempt's lease: dispatch(context, assignment, request, cancel=...).
type Dispatch = Callable[..., Answer | Unanswered]
# A body builds its requests only from its revision and argument, so a resumed activity sends the
# requests that the stored claims name and reuses their replies.
type Body = Callable[[CampaignRevision, str, Dispatch], Awaitable[Outcome]]

ACTOR = "runner"
# A model attempt reserves this much wall time and every activity of the attempt shares it. A
# compute attempt's activity runs for at most this long.
TIMEOUT = timedelta(minutes=10)
CLOSE = timedelta(seconds=30)
# The close activity returns every receipt of an attempt in one payload, so a chain stays short.
CHAIN = 16
# Temporal refuses a payload over 2 MiB and warns over 512 KiB, while a record grows with attempts.
PART = 2**18
# Names enter activity IDs, ledger roles, and acceptance reasons, which refuse other text.
NAME = re.compile(r"[A-Za-z0-9._-]{1,100}")
# Worker reasons that judge what a finished candidate wrote. The others mean a refused read or a
# candidate that never finished.
REJECTED = frozenset(
    {"missing_output", "extra_output", "oversized", "malformed", "non_finite", "wrong_index"}
)


class RunnerError(ValueError):
    """Report a refused campaign start or steering request."""


@dataclass(frozen=True)
class Job:
    """Declare one logical execution.

    body names a worker callable, and cost is the compute seconds charged per attempt.
    An external body delegates completion to an outside reporter. New jobs use zero dispatches.
    Nonzero dispatches retain the historical wire contract and may only join an exact active
    legacy run. Declare new model work with ModelJob.
    """

    name: str
    body: str
    argument: str
    cost: int
    external: bool = False
    dispatches: int = 0


@dataclass(frozen=True)
class Plan:
    revision: str
    jobs: tuple[Job, ...]


@dataclass(frozen=True)
class ModelJob:
    """Declare one prepared model request and its domain interpretation.

    body names immutable ModelBody callbacks. Their bindings must remain available for
    retained work. dispatches reserves the ceiling of the pinned role's fallback chain.
    """

    name: str
    body: str
    argument: str
    cost: int
    role: str
    dispatches: int
    schema: Schema | None = None


@dataclass(frozen=True)
class ModelWork:
    """Supply worker resources for one attempt; callbacks may run on different workers."""

    attempt_id: UUID
    revision: CampaignRevision
    argument: str
    authority: Authority


@dataclass(frozen=True)
class ModelBody:
    """Prepare restricted materials and interpret the exact recorded request and result.

    Preparation must be repeatable. Interpretation must publish idempotently under attempt_id
    because a lost activity result can repeat that callback without repeating the model call.
    """

    prepare: Callable[[ModelWork], Awaitable[tuple[Material, ...]]]
    interpret: Callable[[ModelWork, Request, Answer | Unanswered], Awaitable[Outcome]]


@dataclass(frozen=True)
class _ComputeJob:
    kind: Literal["compute"]
    name: str
    body: str
    argument: str
    cost: int
    external: bool


@dataclass(frozen=True)
class _ModelJob:
    kind: Literal["model"]
    name: str
    body: str
    argument: str
    cost: int
    role: str
    dispatches: int
    schema: tuple[tuple[str, str], ...] | None


@dataclass(frozen=True)
class _PlanV2:
    schema_version: Literal[2]
    revision: str
    entries: tuple[_ComputeJob | _ModelJob, ...]


@dataclass(frozen=True)
class _PrepareTask:
    attempt_id: UUID
    body: str
    argument: str
    revision: str
    role: str


@dataclass(frozen=True)
class _Prepared:
    assignment: str
    prompt: str
    materials: tuple[str, ...]


@dataclass(frozen=True)
class _InterpretTask:
    work: _PrepareTask
    request: str
    result: str


@dataclass(frozen=True)
class Task:
    body: str
    argument: str
    revision: str
    lease: Lease


@dataclass(frozen=True)
class Report:
    """Carry an attempt's outcome and the time the attempt itself finished."""

    outcome: Outcome
    finished_at: datetime


@dataclass(frozen=True)
class AttemptLink:
    """Name the Temporal activity of an acceptance attempt and the revision it ran under."""

    activity_id: str
    attempt_id: UUID
    revision: str
    ended: Ending | None = None


@dataclass(frozen=True)
class Summary:
    status: Status
    spent: int


@dataclass(frozen=True)
class Record:
    workflow_id: str
    run_id: str
    status: Status
    spent: int
    attempts: tuple[AttemptLink, ...]
    acceptance: str
    steering: str
    ledger: str


Phase = Literal["compute", "prepare", "model", "interpret"]
InterruptionReason = Literal[
    "steering_cancel",
    "temporal_cancel",
    "deadline",
    "activity_failed",
    "activity_cancelled",
    "invalid_result",
    "invalid_preparation",
    "invalid_payload",
]


@dataclass(frozen=True)
class ActivityLink:
    activity_id: str
    attempt_id: UUID
    revision: str
    phase: Phase
    ended: Ending | None = None
    cancel_requested_at: datetime | None = None


@dataclass(frozen=True)
class Interruption:
    attempt_id: UUID
    phase: Phase
    reason: InterruptionReason
    observed_at: datetime


@dataclass(frozen=True)
class RecordV2:
    schema_version: Literal[2]
    workflow_id: str
    run_id: str
    status: Status
    spent: int
    attempts: tuple[ActivityLink, ...]
    interruptions: tuple[Interruption, ...]
    acceptance: str
    steering: str
    ledger: str


def _read_plan(raw: RawValue) -> Plan | _PlanV2:
    converter = DataConverter.default.payload_converter
    value = converter.from_payload(raw.payload)
    if "entries" not in value:
        return converter.from_payload(raw.payload, Plan)
    if value.get("schema_version") != 2 or "jobs" in value:
        raise ValueError("Unsupported campaign input")
    entries = []
    for entry in value["entries"]:
        kind = {"compute": _ComputeJob, "model": _ModelJob}[entry["kind"]]
        entries.append(converter.from_payload(converter.to_payload(entry), kind))
    return _PlanV2(2, value["revision"], tuple(entries))


def _plan_v2(revision: str, jobs: tuple[Job | ModelJob, ...]) -> _PlanV2:
    entries = []
    for job in jobs:
        if isinstance(job, ModelJob):
            schema = None if job.schema is None else tuple((n, t.__name__) for n, t in job.schema)
            entries.append(
                _ModelJob(
                    "model",
                    job.name,
                    job.body,
                    job.argument,
                    job.cost,
                    job.role,
                    job.dispatches,
                    schema,
                )
            )
        else:
            entries.append(
                _ComputeJob("compute", job.name, job.body, job.argument, job.cost, job.external)
            )
    return _PlanV2(2, revision, tuple(entries))


@workflow.defn
class CampaignRun:
    @workflow.init
    def __init__(self, raw: RawValue) -> None:
        plan = _read_plan(raw)
        self._plan = plan
        revision = parse_file(plan.revision.encode())
        self._revisions: dict[str, CampaignRevision] = {revision.revision_id: revision}
        self._revision = revision
        limits = revision.specification.limits
        self._ledger = Ledger(
            {
                revision.campaign_id: Budget(
                    compute_seconds=Decimal(limits.max_compute_seconds),
                    wall_seconds=None
                    if limits.max_wall_seconds is None
                    else Decimal(limits.max_wall_seconds),
                    model_calls=limits.max_model_calls,
                    input_tokens=limits.max_input_tokens,
                    output_tokens=limits.max_output_tokens,
                    failures_per_run=limits.max_attempts,
                )
            }
        )
        self._acceptance = AcceptanceLog()
        self._steering = SteeringLog(revision.campaign_id)
        self._attempts: list[AttemptLink] = []
        self._activities: list[ActivityLink] = []
        self._interruptions: list[Interruption] = []
        self._current = False
        self._ended: Literal["completed", "budget_exhausted", "cancelled"] | None = None

    @workflow.run
    async def run(self, raw: RawValue) -> Summary:
        self._current = workflow.patched("campaignrun-model-v2")
        plan = self._plan
        if not self._current:
            return await self._legacy(plan)
        if isinstance(plan, Plan):
            if any(job.dispatches for job in plan.jobs):
                raise ApplicationError(
                    "New model runs require ModelJob",
                    type="LegacyModelProtocol",
                    non_retryable=True,
                )
            plan = _plan_v2(plan.revision, plan.jobs)
        try:
            return await self._run_v2(plan)
        except asyncio.CancelledError:
            self._ended = "cancelled"
            return self.summary()

    async def _legacy(self, plan: Plan) -> Summary:
        campaign_id = self._revision.campaign_id
        for job in plan.jobs:
            trial_id, run_id = workflow.uuid4(), workflow.uuid4()
            predecessor = None
            numbers = itertools.count(1)
            while True:
                await workflow.wait_condition(lambda: self._steering.status != "paused")
                if self._steering.status == "cancelled":
                    return self.summary()
                attempt_id = workflow.uuid4()
                lease = Lease(attempt_id, tuple(workflow.uuid4() for _ in range(job.dispatches)), 0)
                cost = Usage(compute_seconds=Decimal(job.cost))
                held = replace(cost, wall_seconds=_seconds(TIMEOUT), model_calls=job.dispatches)
                refusals = self._ledger.reserve(
                    Work(campaign_id, trial_id, run_id, attempt_id),
                    held if job.dispatches else cost,
                ).refusals
                if any(refusal.limit != "failures_per_run" for refusal in refusals):
                    self._ended = "budget_exhausted"
                    return self.summary()
                if refusals:
                    break
                self._ledger.charge(workflow.uuid4(), attempt_id, job.body, cost)
                self._acceptance.record(
                    Submitted(trial_id, run_id, attempt_id, predecessor, workflow.now())
                )
                task = Task(job.body, job.argument, file_bytes(self._revision).decode(), lease)
                raw, limited = await self._attempt(job, task, self._revision.revision_id, numbers)
                completion = None if raw is None else self._complete(attempt_id, raw, limited)
                if completion is not None:
                    status = completion.outcome.status
                else:
                    status = "inconclusive" if limited else "failed"
                self._ledger.release(attempt_id, status)
                if completion is not None and status != "failed" and not limited:
                    self._acceptance.record(
                        Accepted(
                            attempt_id,
                            None,
                            ACTOR,
                            f"First attempt of {job.name} with a non-failed outcome",
                            workflow.now(),
                        )
                    )
                    break
                predecessor = attempt_id
        self._ended = "completed"
        return self.summary()

    async def _run_v2(self, plan: _PlanV2) -> Summary:
        for job in plan.entries:
            trial_id, run_id = workflow.uuid4(), workflow.uuid4()
            predecessor = None
            numbers = itertools.count(1)
            while True:
                await workflow.wait_condition(lambda: self._steering.status != "paused")
                if (self._ended or self._steering.status) == "cancelled":
                    return self.summary()
                revision = self._revision
                attempt_id = workflow.uuid4()
                cost = Usage(compute_seconds=Decimal(job.cost))
                model = isinstance(job, _ModelJob)
                held = (
                    replace(cost, wall_seconds=_seconds(TIMEOUT), model_calls=job.dispatches)
                    if model
                    else cost
                )
                refusals = self._ledger.reserve(
                    Work(revision.campaign_id, trial_id, run_id, attempt_id), held
                ).refusals
                if any(refusal.limit != "failures_per_run" for refusal in refusals):
                    self._ended = "budget_exhausted"
                    return self.summary()
                if refusals:
                    break
                self._ledger.charge(workflow.uuid4(), attempt_id, job.body, cost)
                self._acceptance.record(
                    Submitted(trial_id, run_id, attempt_id, predecessor, workflow.now())
                )
                if model:
                    raw, limited = await self._model_attempt(job, attempt_id, revision)
                    phase = "interpret"
                else:
                    task = Task(
                        job.body,
                        job.argument,
                        file_bytes(revision).decode(),
                        Lease(attempt_id, (), 0),
                    )
                    raw = await self._worker_phase(
                        "compute",
                        task,
                        attempt_id,
                        revision.revision_id,
                        job.name,
                        numbers,
                        workflow.now() + TIMEOUT,
                        external=job.external,
                    )
                    limited, phase = False, "compute"
                completion = None
                if raw is not None:
                    try:
                        report = workflow.payload_converter().from_payload(raw.payload)
                        completion = Completed(
                            attempt_id,
                            _limited(parse_outcome(report["outcome"]), limited),
                            datetime.fromisoformat(report["finished_at"]),
                        )
                        self._acceptance.record(completion)
                    except Exception:
                        self._interrupt(attempt_id, phase, "invalid_result")
                        completion = None
                status = (
                    completion.outcome.status
                    if completion
                    else "inconclusive"
                    if limited
                    else "failed"
                )
                self._ledger.release(attempt_id, status)
                if (self._ended or self._steering.status) == "cancelled":
                    return self.summary()
                if completion is not None and status != "failed" and not limited:
                    self._acceptance.record(
                        Accepted(
                            attempt_id,
                            None,
                            ACTOR,
                            f"First attempt of {job.name} with a non-failed outcome",
                            workflow.now(),
                        )
                    )
                    break
                predecessor = attempt_id
        self._ended = "completed"
        return self.summary()

    def _interrupt(self, attempt_id: UUID, phase: Phase, reason: InterruptionReason) -> None:
        self._interruptions.append(Interruption(attempt_id, phase, reason, workflow.now()))
        if reason in {"steering_cancel", "temporal_cancel"}:
            self._ended = "cancelled"

    async def _worker_phase(
        self,
        phase: Phase,
        task: Task | _PrepareTask | _InterpretTask,
        attempt_id: UUID,
        revision_id: str,
        prefix: str,
        numbers: Iterator[int],
        deadline: datetime,
        *,
        external: bool = False,
    ) -> RawValue | None:
        while True:
            left = deadline - workflow.now()
            if left < timedelta(seconds=1):
                self._interrupt(attempt_id, phase, "deadline")
                return None
            activity_id = f"{prefix}/{next(numbers)}"
            index = len(self._activities)
            self._activities.append(ActivityLink(activity_id, attempt_id, revision_id, phase))
            handle = workflow.start_activity(
                "attempt" if phase == "compute" else f"{phase}_model",
                task,
                activity_id=activity_id,
                result_type=RawValue,
                start_to_close_timeout=TIMEOUT if phase == "compute" else None,
                schedule_to_close_timeout=None
                if phase == "compute"
                else timedelta(seconds=left // timedelta(seconds=1)),
                heartbeat_timeout=None if external else timedelta(seconds=2),
                retry_policy=RetryPolicy(maximum_attempts=1),
                cancellation_type=ActivityCancellationType.TRY_CANCEL
                if external
                else ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
            )
            interrupted = None
            try:
                await workflow.wait_condition(
                    lambda handle=handle: handle.done() or self._steering.status == "cancelled"
                )
                if self._steering.status == "cancelled":
                    interrupted = "steering_cancel"
            except asyncio.CancelledError:
                interrupted = "temporal_cancel"
            requested_at = None
            if not handle.done():
                requested_at = workflow.now()
                handle.cancel()
            while True:
                try:
                    raw = await asyncio.shield(handle)
                    ending = "completed"
                    break
                except asyncio.CancelledError:
                    interrupted = "temporal_cancel"
                except ActivityError as error:
                    ending = (
                        "cancelled"
                        if is_cancelled_exception(error)
                        else "timed_out"
                        if isinstance(error.cause, exceptions.TimeoutError)
                        else "failed"
                    )
                    raw = None
                    break
            self._activities[index] = replace(
                self._activities[index],
                ended=None
                if external and requested_at is not None and ending == "cancelled"
                else ending,
                cancel_requested_at=requested_at,
            )
            if interrupted:
                self._interrupt(attempt_id, phase, interrupted)
                return raw
            if (
                ending == "timed_out"
                and phase != "compute"
                and deadline - workflow.now() >= timedelta(seconds=1)
            ):
                continue
            if raw is None:
                self._interrupt(
                    attempt_id,
                    phase,
                    {
                        "timed_out": "deadline",
                        "cancelled": "activity_cancelled",
                        "failed": "activity_failed",
                        "completed": "invalid_preparation"
                        if phase == "prepare"
                        else "invalid_result",
                    }[ending],
                )
            return raw

    async def _model_attempt(
        self, job: _ModelJob, attempt_id: UUID, revision: CampaignRevision
    ) -> tuple[RawValue | None, bool]:
        started = workflow.now()
        deadline = started + TIMEOUT
        close_elapsed = timedelta(0)
        task = _PrepareTask(
            attempt_id, job.body, job.argument, file_bytes(revision).decode(), job.role
        )
        try:
            if not input_fits(task):
                self._interrupt(attempt_id, "prepare", "invalid_payload")
                return None, False
            raw = await self._worker_phase(
                "prepare",
                task,
                attempt_id,
                revision.revision_id,
                f"prepare/{attempt_id}",
                itertools.count(1),
                deadline,
            )
            if raw is None or (self._ended or self._steering.status) == "cancelled":
                return None, False
            try:
                prepared = DataConverter.default.payload_converter.from_payload(
                    raw.payload, _Prepared
                )
                assignment = load_assignment(prepared.assignment)
                if assignment.role != job.role or 1 + len(assignment.fallbacks) > job.dispatches:
                    raise ValueError("The pinned role exceeds the declared chain")
                kinds = {kind.__name__: kind for kind in (str, int, float, bool)}
                schema = (
                    None
                    if job.schema is None
                    else tuple((name, kinds[kind]) for name, kind in job.schema)
                )
                request = Request(
                    (
                        load_material(prepared.prompt),
                        *(load_material(item) for item in prepared.materials),
                    ),
                    schema,
                )
            except Exception:
                self._interrupt(attempt_id, "prepare", "invalid_preparation")
                return None, False
            within = deadline - workflow.now()
            if within < timedelta(seconds=1):
                self._interrupt(attempt_id, "model", "deadline")
                return None, False
            asked = await ask_model(
                revision,
                assignment,
                request,
                attempt_id,
                within=within,
                stop=lambda: self._steering.status == "cancelled",
            )
            close_elapsed = asked.close_elapsed
            self._activities.extend(
                ActivityLink(
                    item.activity_id, attempt_id, revision.revision_id, "model", item.ended
                )
                for item in asked.activities
            )
            for receipt in asked.receipts:
                self._charge(attempt_id, receipt)
            outcomes = {receipt.outcome for receipt in asked.receipts}
            limited = "rate_limited" in outcomes and "completed" not in outcomes
            if asked.interrupted_by or self._steering.status == "cancelled":
                self._interrupt(
                    attempt_id,
                    "model",
                    "temporal_cancel" if asked.interrupted_by == "workflow" else "steering_cancel",
                )
                return None, limited
            if isinstance(asked.result, str):
                self._interrupt(
                    attempt_id,
                    "model",
                    {
                        "timed_out": "deadline",
                        "cancelled": "activity_cancelled",
                        "failed": asked.failure_reason or "activity_failed",
                    }[asked.result],
                )
                return None, limited
            interpreting = _InterpretTask(
                task, request.to_bytes().decode(), dump_result(asked.result)
            )
            if not input_fits(interpreting):
                self._interrupt(attempt_id, "interpret", "invalid_payload")
                return None, limited
            raw = await self._worker_phase(
                "interpret",
                interpreting,
                attempt_id,
                revision.revision_id,
                f"interpret/{attempt_id}",
                itertools.count(1),
                deadline + close_elapsed,
            )
            return raw, limited
        finally:
            elapsed = min(workflow.now() - started - close_elapsed, TIMEOUT)
            self._ledger.charge(
                workflow.uuid4(), attempt_id, job.body, Usage(wall_seconds=_seconds(elapsed))
            )

    async def _attempt(
        self, job: Job, task: Task, revision_id: str, numbers: Iterator[int]
    ) -> tuple[RawValue | None, bool]:
        """Run an attempt's activities, then settle a model attempt's dispatches.

        Returns the last activity's report payload, or None when it ended without one, and
        whether the attempt was rate-limited: a receipt says rate_limited and none completed.

        A model attempt whose report is lost resumes under the same lease in a new activity with
        the attempt's remaining TIMEOUT, so a stored reply is reused instead of sent again, and
        no activity of the attempt outlives its wall reservation. A compute attempt has nothing
        to reuse, so a lost report ends it as D042 describes. A model attempt then charges its
        wall time, closes its lease, and charges every receipt before the caller releases it.
        """
        started = workflow.now()
        while True:
            number = next(numbers)
            activity_id = f"{job.name}/{number}"
            self._attempts.append(AttemptLink(activity_id, task.lease.attempt_id, revision_id))
            left = TIMEOUT - (workflow.now() - started)
            handle = workflow.start_activity(
                "attempt",
                replace(task, lease=replace(task.lease, fence=number)),
                activity_id=activity_id,
                result_type=RawValue,
                start_to_close_timeout=None if job.dispatches else TIMEOUT,
                schedule_to_close_timeout=timedelta(seconds=left // timedelta(seconds=1))
                if job.dispatches
                else None,
                heartbeat_timeout=None if job.external else timedelta(seconds=2),
                # Server retries hide earlier attempts from history and from the budget.
                retry_policy=RetryPolicy(maximum_attempts=1),
                # An external attempt sends no heartbeat that could deliver the cancel.
                cancellation_type=ActivityCancellationType.TRY_CANCEL
                if job.external
                else ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
            )
            ending, raw = await self._settled(handle)
            self._attempts[-1] = replace(self._attempts[-1], ended=ending)
            elapsed = workflow.now() - started
            if (
                ending != "timed_out"
                or not job.dispatches
                or self._steering.status == "cancelled"
                or TIMEOUT - elapsed < timedelta(seconds=1)
            ):
                break
        if not job.dispatches:
            return raw, False
        wall = Usage(wall_seconds=_seconds(min(elapsed, TIMEOUT)))
        self._ledger.charge(workflow.uuid4(), task.lease.attempt_id, job.body, wall)
        receipts = load_receipts(
            await workflow.execute_activity(
                "close",
                task.lease,
                activity_id=f"close/{task.lease.attempt_id}",
                result_type=str,
                start_to_close_timeout=CLOSE,
            )
        )
        for receipt in receipts:
            self._charge(task.lease.attempt_id, receipt)
        outcomes = {receipt.outcome for receipt in receipts}
        return raw, "rate_limited" in outcomes and "completed" not in outcomes

    async def _settled(
        self, handle: workflow.ActivityHandle[RawValue]
    ) -> tuple[Ending, RawValue | None]:
        await workflow.wait_condition(lambda: handle.done() or self._steering.status == "cancelled")
        if not handle.done():
            handle.cancel()
        try:
            return "completed", await handle
        except ActivityError as error:
            if is_cancelled_exception(error):
                return "cancelled", None
            if isinstance(error.cause, exceptions.TimeoutError):
                return "timed_out", None
            return "failed", None

    def _charge(self, attempt_id: UUID, receipt: Receipt) -> None:
        """Charge one receipt under its call ID.

        A cost that the campaign's exact dollar sum cannot hold is charged as unreported
        instead, because Temporal cannot refuse an activity result and a raise here would fail
        every workflow task. The call log keeps the cost the provider reported.
        """
        try:
            self._ledger.charge(receipt.call_id, attempt_id, receipt.role, receipt.usage)
        except InexactUsage:
            unreported = receipt.usage.unreported | {"cost_usd"}
            usage = replace(receipt.usage, cost_usd=Decimal(0), unreported=unreported)
            self._ledger.charge(receipt.call_id, attempt_id, receipt.role, usage)

    def _complete(self, attempt_id: UUID, raw: RawValue, limited: bool) -> Completed:
        """Record the reported outcome; a rate-limited attempt's becomes inconclusive.

        An inconclusive release spends no failure slot, and the runner never accepts a
        rate-limited attempt, so the next attempt dispatches under new call IDs.
        """
        try:
            report = workflow.payload_converter().from_payload(raw.payload)
            # Temporal writes datetimes as isoformat text, which records.parse_time rejects.
            finished_at = datetime.fromisoformat(report["finished_at"])
            outcome = parse_outcome(report["outcome"])
            completion = Completed(attempt_id, _limited(outcome, limited), finished_at)
            self._acceptance.record(completion)
        # Temporal cannot refuse a report, so any report this block cannot record, however it
        # fails, becomes the attempt's failed outcome instead of failing every workflow task.
        except Exception as error:
            failed = Outcome("failed", f"The report cannot be recorded: {str(error)!r}")
            completion = Completed(attempt_id, _limited(failed, limited), workflow.now())
            self._acceptance.record(completion)
        return completion

    # The sandbox hides datetime behind a proxy the converter cannot decode to, so update
    # times cross as canonical UTC text.
    @workflow.update
    def control(self, action: str, actor: str, at: str, reason: str) -> Status:
        self._steering = self._steering.record(
            action, actor=actor, at=parse_time(at), reason=reason
        )
        return self.summary().status

    @control.validator
    def _check_control(self, action: str, actor: str, at: str, reason: str) -> None:
        self._steering.record(action, actor=actor, at=parse_time(at), reason=reason)

    @workflow.update
    def steer(self, revision: str, at: str) -> Status:
        child, self._steering = self._steered(revision, at)
        self._revisions[child.revision_id] = child
        self._revision = child
        return self.summary().status

    @steer.validator
    def _check_steer(self, revision: str, at: str) -> None:
        self._steered(revision, at)

    def _steered(self, text: str, at: str) -> tuple[CampaignRevision, SteeringLog]:
        child = parse_file(text.encode())
        parent = self._revisions.get(child.parent)
        if parent is None:
            raise RunnerError(
                f"Revision {child.revision_id} revises a revision this run does not hold"
            )
        check_link(parent, child)
        if "limits" in child.changes:
            raise RunnerError(
                f"Revision {child.revision_id} changes limits; a running campaign keeps its "
                "starting limits"
            )
        return child, self._steering.steer(child, at=parse_time(at))

    @workflow.query
    def summary(self) -> Summary:
        spent = sum(line.usage.compute_seconds for line in self._ledger.report())
        return Summary(self._ended or self._steering.status, int(spent))

    @workflow.query
    def part(self, start: int) -> bytes:
        info = workflow.info()
        summary = self.summary()
        fields = (
            info.workflow_id,
            info.run_id,
            summary.status,
            summary.spent,
        )
        logs = (
            dump_log(self._acceptance).decode(),
            log_bytes(self._steering).decode(),
            dump_ledger(self._ledger).decode(),
        )
        record = (
            RecordV2(2, *fields, tuple(self._activities), tuple(self._interruptions), *logs)
            if self._current
            else Record(*fields, tuple(self._attempts), *logs)
        )
        return DataConverter.default.payload_converter.to_payload(record).data[start : start + PART]


def _limited(outcome: Outcome, limited: bool) -> Outcome:
    if not limited:
        return outcome
    return replace(outcome, status="inconclusive", detail=f"rate_limited: {outcome.detail}")


def _seconds(delta: timedelta) -> Decimal:
    return Decimal(delta // timedelta(microseconds=1)).scaleb(-6)


def _no_gateway(*_: object, **__: object) -> Answer | Unanswered:
    raise ProviderError("This worker has no gateway")


class _Attempts(Closing):
    def __init__(self, bodies: Mapping[str, Body], gateway: Gateway | None) -> None:
        self._bodies = dict(bodies)
        super().__init__(gateway)

    @activity.defn(name="attempt")
    async def attempt(self, task: Task) -> Report:
        body = self._bodies[task.body]
        calling = _no_gateway if self._gateway is None else self._gateway.call

        def dispatch(
            context: RunContext,
            assignment: RoleAssignment,
            request: Request,
            *,
            cancel: threading.Event,
        ) -> Answer | Unanswered:
            return calling(context, assignment, request, lease=task.lease, cancel=cancel)

        beating = asyncio.create_task(_heartbeat())
        try:
            outcome = await body(parse_file(task.revision.encode()), task.argument, dispatch)
        # The body's failure is its failed outcome. raise_complete_async and cancellation are
        # BaseExceptions and pass through to Temporal.
        except Exception as error:
            outcome = Outcome("failed", repr(error))
        finally:
            beating.cancel()
        return Report(outcome, datetime.now(UTC))


async def _heartbeat() -> None:
    while True:
        activity.heartbeat()
        await asyncio.sleep(0.2)


class _ModelBodies:
    def __init__(
        self, models: Mapping[str, ModelBody], resolve: Callable[[CampaignRevision], Authority]
    ) -> None:
        self._models = dict(models)
        self._resolve = resolve

    def _work(self, task: _PrepareTask) -> ModelWork:
        revision = parse_file(task.revision.encode())
        return ModelWork(task.attempt_id, revision, task.argument, self._resolve(revision))

    @activity.defn(name="prepare_model")
    async def prepare_model(self, task: _PrepareTask) -> _Prepared:
        beating = asyncio.create_task(_heartbeat())
        try:
            work = self._work(task)
            assignment = next(
                item for item in work.authority.roles.assignments if item.role == task.role
            )
            materials = await self._models[task.body].prepare(work)
            return _Prepared(
                dump_assignment(assignment),
                dump_material(work.authority.prompts[task.role]),
                tuple(dump_material(item) for item in materials),
            )
        except Exception:
            activity.logger.exception("Model preparation failed")
        finally:
            beating.cancel()
        raise ApplicationError(
            "Model preparation failed", type="PreparationFailed", non_retryable=True
        )

    @activity.defn(name="interpret_model")
    async def interpret_model(self, task: _InterpretTask) -> Report:
        beating = asyncio.create_task(_heartbeat())
        try:
            work = self._work(task.work)
            request = Request.from_bytes(task.request.encode())
            result = load_result(task.result, request)
            outcome = await self._models[task.work.body].interpret(work, request, result)
            return Report(outcome, datetime.now(UTC))
        except Exception:
            activity.logger.exception("Model interpretation failed")
        finally:
            beating.cancel()
        raise ApplicationError(
            "Model interpretation failed", type="InterpretationFailed", non_retryable=True
        )


async def start(
    client: Client, revision: CampaignRevision, jobs: tuple[Job | ModelJob, ...], task_queue: str
) -> WorkflowHandle[CampaignRun, Summary]:
    """Start the campaign's only run, or return it while it is still running.

    The workflow ID is the campaign ID, which the research record must keep single-use.
    New model work requires ModelJob. An exact legacy request can join its active run, bound to
    that run ID. The memo comparison applies to callers of this helper; raw USE_EXISTING input
    is not delivered to an already running workflow.

    Args:
        client (Client): Connected Temporal client.
        revision (CampaignRevision): Revision the first attempts run under.
        jobs (tuple[Job | ModelJob, ...]): Compute, external, or model executions in run order,
            with distinct names and nonnegative costs.
        task_queue (str): Queue that a worker with the jobs' bodies polls.

    Returns:
        (WorkflowHandle) The campaign's run.

    Raises:
        RunnerError: A job or body name is not a NAME, job names repeat, jobs exceed
            max_trials, a cost or dispatch count is not a nonnegative int, a job dispatches
            more than CHAIN models or more than max_model_calls (undeclared reads as none) or is
            external, a dispatching campaign's max_wall_seconds is below TIMEOUT, or the
            campaign is already running a different plan or its workflow ID belongs to
            another runner.
        CampaignError | PackageError: The revision does not parse, or a job argument is not
            valid UTF-8.
    """
    content = file_bytes(revision)
    parse_file(content)
    if any(NAME.fullmatch(text) is None for job in jobs for text in (job.name, job.body)):
        raise RunnerError(f"Job and body names must match {NAME.pattern}")
    names = [job.name for job in jobs]
    if len(set(names)) != len(names):
        raise RunnerError(f"Job names must be distinct, not {names}")
    max_trials = revision.specification.limits.max_trials
    if len(jobs) > max_trials:
        raise RunnerError(f"{len(jobs)} jobs exceed the campaign's {max_trials} trials")
    if any(type(job.cost) is not int or job.cost < 0 for job in jobs):
        raise RunnerError("Job costs must be nonnegative integer compute seconds")
    if any(type(job.dispatches) is not int or job.dispatches < 0 for job in jobs):
        raise RunnerError("Job dispatches must be nonnegative integers")
    if any(job.dispatches > CHAIN for job in jobs):
        raise RunnerError(f"A job dispatches at most {CHAIN} models")
    limits = revision.specification.limits
    calls = limits.max_model_calls or 0
    if any(job.dispatches > calls for job in jobs):
        raise RunnerError(f"A job dispatches more models than the campaign's {calls} model calls")
    if any(isinstance(job, Job) and job.external and job.dispatches for job in jobs):
        raise RunnerError("An external job cannot dispatch models")
    if any(
        isinstance(job, ModelJob) and (NAME.fullmatch(job.role) is None or not job.dispatches)
        for job in jobs
    ):
        raise RunnerError("A model job needs a named role and at least one dispatch")
    wall = limits.max_wall_seconds
    if any(job.dispatches for job in jobs) and wall is not None and wall < TIMEOUT.total_seconds():
        raise RunnerError(f"Dispatching jobs need max_wall_seconds of at least {TIMEOUT}")
    legacy = Plan(content.decode(), jobs) if all(isinstance(job, Job) for job in jobs) else None
    legacy_digest = (
        None if legacy is None else hashlib.sha256(canonical_bytes(asdict(legacy))).hexdigest()
    )
    if any(isinstance(job, Job) and job.dispatches for job in jobs):
        if legacy is None:
            raise RunnerError("New model runs require ModelJob")
        existing = client.get_workflow_handle(str(revision.campaign_id), result_type=Summary)
        try:
            described = await existing.describe()
        except RPCError as error:
            if error.status != RPCStatusCode.NOT_FOUND:
                raise
            raise RunnerError("New model runs require ModelJob") from error
        if (
            described.workflow_type != "CampaignRun"
            or await described.memo_value("plan", None) != legacy_digest
        ):
            raise RunnerError(f"Campaign {revision.campaign_id} is running a different plan")
        if described.status != WorkflowExecutionStatus.RUNNING:
            raise RunnerError("The retained campaign is closed; new model runs require ModelJob")
        return client.get_workflow_handle(existing.id, run_id=described.run_id, result_type=Summary)
    plan = _plan_v2(content.decode(), jobs)
    digest = hashlib.sha256(canonical_bytes(asdict(plan))).hexdigest()
    handle = await client.start_workflow(
        CampaignRun.run,
        plan,
        id=str(revision.campaign_id),
        task_queue=task_queue,
        id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
        id_reuse_policy=WorkflowIDReusePolicy.REJECT_DUPLICATE,
        memo={"plan": digest},
    )
    # USE_EXISTING returns the running run without delivering this plan.
    described = await handle.describe()
    if described.workflow_type != "CampaignRun" or await described.memo_value("plan", None) not in {
        digest,
        legacy_digest,
    } - {None}:
        raise RunnerError(f"Campaign {revision.campaign_id} is running a different plan")
    return client.get_workflow_handle(handle.id, run_id=described.run_id, result_type=Summary)


async def control(
    handle: WorkflowHandle[CampaignRun, Summary],
    action: str,
    *,
    actor: str,
    at: datetime,
    reason: str,
) -> Status:
    """Record a pause, resume, cancel, or annotation in the run's steering log.

    Args:
        handle (WorkflowHandle): Campaign run.
        action (str): pause, resume, cancel, or annotate.
        actor (str): Person who issues the command.
        at (datetime): Timezone-aware time, no earlier than the log's last event.
        reason (str): Why the command is issued, or the note itself for an annotation.

    Returns:
        (Status) The run's status after the command.
    """
    return await handle.execute_update(
        CampaignRun.control, args=[action, actor, format_time(at), reason]
    )


async def steer(
    handle: WorkflowHandle[CampaignRun, Summary], revision: CampaignRevision, *, at: datetime
) -> Status:
    """Steer the run to a child revision; attempts scheduled after it run under the child.

    Args:
        handle (WorkflowHandle): Campaign run.
        revision (CampaignRevision): Child of a revision the run holds that keeps its limits.
        at (datetime): Timezone-aware time, no earlier than the child's creation or the log's
            last event.

    Returns:
        (Status) The run's status after the steer.
    """
    return await handle.execute_update(
        CampaignRun.steer, args=[file_bytes(revision).decode(), format_time(at)]
    )


async def record(handle: WorkflowHandle[CampaignRun, Summary]) -> Record | RecordV2:
    """Wait for the run to close and read its record, PART bytes per query.

    Args:
        handle (WorkflowHandle): Campaign run.

    Returns:
        (Record | RecordV2) The retained or versioned record, decoded after joining its parts.
            New direct cancellation returns a cancelled domain record after settlement.
    """
    await handle.result()
    parts = [await handle.query(CampaignRun.part, 0)]
    while len(parts[-1]) == PART:
        parts.append(await handle.query(CampaignRun.part, PART * len(parts)))
    payload = Payload(metadata={"encoding": b"json/plain"}, data=b"".join(parts))
    kind = RecordV2 if json.loads(payload.data).get("schema_version") == 2 else Record
    return DataConverter.default.payload_converter.from_payload(payload, kind)


class _Undecoded(DefaultPayloadConverter):
    def from_payloads(self, payloads, type_hints=None) -> list[RawValue]:
        return [RawValue(payload) for payload in payloads]


class _Failures(DefaultFailureConverter):
    """Convert a failure without decoding its details, which workflow code never reads.

    Temporal converts a reported failure before workflow code runs, so a detail this converter
    could not decode would fail every workflow task.
    """

    def from_failure(self, failure: Failure, payload_converter: PayloadConverter) -> BaseException:
        return super().from_failure(failure, _Undecoded())


def worker(
    client: Client,
    task_queue: str,
    bodies: Mapping[str, Body],
    gateway: Gateway | None = None,
    *,
    models: Mapping[str, ModelBody] | None = None,
    resolve: Callable[[CampaignRevision], Authority] | None = None,
) -> Worker:
    """Bind compute bodies, optional model callbacks, and one shared close activity.

    Every worker on the queue opens the same gateway call log. Immutable body versions and
    operator bindings must remain available for retained work. Compute-only workers need no
    gateway or resolver. The new input-size checks qualify the default converter without a
    payload codec.

    Args:
        client (Client): Connected Temporal client.
        task_queue (str): Queue to poll.
        bodies (Mapping[str, Body]): Compute and retained legacy bodies.
        gateway (Gateway | None, optional): Gateway shared by model activities and close.
        models (Mapping[str, ModelBody] | None, optional): Versioned preparation and interpretation
            callbacks. Publications must be idempotent under ModelWork.attempt_id.
        resolve (Callable[[CampaignRevision], Authority] | None, optional): Resolve each callback
            and model activity's pinned revision. Required with gateway when models are bound.

    Returns:
        (Worker) The unstarted worker.
    """
    converter = replace(client.data_converter, failure_converter_class=_Failures)
    attempts = _Attempts(bodies, gateway)
    definitions = [attempts.attempt, attempts.close]
    if models:
        if gateway is None or resolve is None:
            raise RunnerError("Model bodies require a gateway and authority resolver")
        model_bodies = _ModelBodies(models, resolve)
        definitions.extend(
            [
                model_bodies.prepare_model,
                Activities(gateway, resolve).model,
                model_bodies.interpret_model,
            ]
        )
    return Worker(
        Client(**client.config() | {"data_converter": converter}),
        task_queue=task_queue,
        workflows=[CampaignRun],
        activities=definitions,
    )


def run_worker(
    address: str,
    task_queue: str,
    bodies: Mapping[str, Body],
    gateway: Gateway | None = None,
    *,
    models: Mapping[str, ModelBody] | None = None,
    resolve: Callable[[CampaignRevision], Authority] | None = None,
) -> None:
    async def serve() -> None:
        await worker(
            await Client.connect(address),
            task_queue,
            bodies,
            gateway,
            models=models,
            resolve=resolve,
        ).run()

    asyncio.run(serve())


async def replay(revision: CampaignRevision, argument: str, dispatch: Dispatch) -> Outcome:
    """Replay a synthetic fixture as a new trial of the revision's question.

    argument is JSON {"fixture": path, "destination": directory, "lock": text}. The outcome is
    the package's new attempt outcome, and its detail names the package digest and path.
    """
    values = json.loads(argument)
    package = await asyncio.to_thread(
        replay_fixture,
        Path(values["fixture"]),
        Path(values["destination"]),
        intent=NewTrial(revision.specification.question),
        lock=values["lock"].encode(),
    )
    outcome = package.record.selected_trial.attempts[-1].outcome
    return replace(outcome, detail=f"{outcome.detail}; package {package.digest} at {package.path}")


def generated(
    catalog: Catalog,
    store: Path,
    context: RunContext,
    log: container.RunLog,
    destination: Path,
    limits: container.Limits = container.LIMITS,
) -> Body:
    """Return a body that runs a job's generated signal code in the isolated worker.

    argument is JSON {"source": text, "dataset": name}. The catalog, store, run context, run log,
    output directory, and limits come from the operator, so no argument can widen them. The body
    starts no container unless the revision the attempt runs under enables generated code. Each
    container run is appended to log, and its outcome detail names the logged run ID. A validated
    signal's values are written to destination as <digest>.json under the logged output digest. A
    REJECTED refusal rejects the candidate, and any other refusal fails the attempt; both details
    name the worker's reason.
    """

    async def body(revision: CampaignRevision, argument: str, dispatch: Dispatch) -> Outcome:
        if not revision.specification.generated_code:
            raise RunnerError(f"Revision {revision.revision_id} does not enable generated code")
        values = json.loads(argument)
        result = await asyncio.to_thread(
            container.run, values["source"], catalog, store, values["dataset"], context, limits
        )
        if result.record is None:
            return Outcome("failed", f"{result.reason}: {result.message}")
        logged = await asyncio.to_thread(log.append, result)
        if isinstance(result, container.Refused):
            status = "rejected" if result.reason in REJECTED else "failed"
            return Outcome(status, f"{result.reason}: {result.message}; run {logged.run_id}")
        path = destination / f"{logged.output}.json"
        path.write_bytes(canonical_bytes(result.values))
        return Outcome("successful", f"signal {logged.output} at {path}; run {logged.run_id}")

    return body
