"""Run model and tool calls as Temporal activities that stop their adapters, and schedule them.

    worker side    Activities(gateway, resolve).definitions: the model, tool, and close
                   activities, bound to the operator's gateway and authority resolver
    workflow side  ask_model: one model attempt from its lease to its close, with its receipts
                   call_tool: one tool call and its rendered material

ask_model and call_tool set every option the activities depend on: the heartbeat timeout that
lets a cancel reach a running call, server retries off so that each activity number is one
fence, the cancellation type, the resume of a timed-out model activity, and a close that a
workflow cancel cannot interrupt. The activity inputs are private, so no workflow schedules
these activities any other way.
"""

from __future__ import annotations

import asyncio
import itertools
import threading
from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import Literal
from uuid import UUID

from temporalio import activity, exceptions, workflow
from temporalio.api.common.v1 import Payloads
from temporalio.common import RawValue, RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError, is_cancelled_exception
from temporalio.workflow import ActivityCancellationType

with workflow.unsafe.imports_passed_through():
    from quant_research.campaigns import Authority, CampaignRevision, file_bytes, parse_file
    from quant_research.model_provenance import RoleAssignment, dump_assignment, load_assignment
    from quant_research.providers import (
        Answer,
        Gateway,
        Lease,
        Material,
        ProviderError,
        Receipt,
        Request,
        Unanswered,
        dump_material,
        dump_receipts,
        dump_result,
        load_material,
        load_receipts,
        load_result,
    )

# Temporal's JSON converter cannot decode a PEP 695 alias, so this stays a plain assignment.
Unreported = Literal["failed", "timed_out", "cancelled"]

# A cancel reaches a heartbeating activity about 0.8 times its heartbeat timeout after the
# request (measured), so the timeout is short and an activity beats five times within it.
HEARTBEAT = timedelta(seconds=1)
BEAT = 0.2
GRACE = timedelta(seconds=5)
CLOSE = timedelta(seconds=30)
INPUT_LIMIT = 2**21
_FAILURES = {
    "ModelFailed": "The model call failed; its cause is in the worker log",
    "ModelDeadline": "The model call passed its deadline and its adapter did not stop",
    "ToolFailed": "The tool call failed; its cause is in the worker log",
}


@dataclass(frozen=True)
class ModelActivity:
    activity_id: str
    fence: int
    ended: Literal["completed", "failed", "timed_out", "cancelled"]


@dataclass(frozen=True)
class Asked:
    """One model attempt after its lease closed.

    result is the last activity's result, or why it reported none. It is "cancelled" whenever
    stop() was true as an activity settled or ask_model caught a workflow cancel, whatever the
    activity reported. A call that its local deadline stopped records cancelled, as a call its
    provider cancelled does, and only elapsed tells them apart. receipts are the attempt's sent
    calls in chain order. elapsed runs from the first schedule to the last settle, at most within.
    """

    result: Answer | Unanswered | Unreported
    receipts: tuple[Receipt, ...]
    elapsed: timedelta
    activities: tuple[ModelActivity, ...] = ()
    close_elapsed: timedelta = timedelta(0)
    interrupted_by: Literal["stop", "workflow"] | None = None
    failure_reason: (
        Literal["activity_failed", "activity_cancelled", "invalid_result", "invalid_payload"] | None
    ) = None


def input_fits(value: object) -> bool:
    """Check a complete activity input against the qualified default-codec transport limit."""
    payload = workflow.payload_converter().to_payload(value)
    return Payloads(payloads=[payload]).ByteSize() <= INPUT_LIMIT


async def ask_model(
    revision: CampaignRevision,
    assignment: RoleAssignment,
    request: Request,
    attempt_id: UUID,
    *,
    within: timedelta,
    stop: Callable[[], bool],
) -> Asked:
    """Run one model attempt from its lease to its close. Call only from workflow code.

    The caller reserves 1 + len(assignment.fallbacks) model calls and within of wall time under
    attempt_id first, then charges elapsed and every receipt; ask_model charges nothing. It
    mints one call ID per model. Activity n holds the lease under fence n, with the rest of
    within, floored to whole seconds, as its schedule-to-close timeout, so within is at least a
    second. A timed-out activity resumes under the next fence while a second remains and stop()
    is false, so a stored reply is reused and a claim without a record is sealed as unknown.
    When stop() turns true or the workflow is cancelled, the running activity is cancelled and
    waited for, so the gateway records its call before the lease closes. The close runs even
    under a workflow cancel.

    Args:
        revision (CampaignRevision): Revision whose authority every activity of the attempt
            resolves, resumes included.
        assignment (RoleAssignment): Role and its models in the order to try them.
        request (Request): Materials to send and the schema of a structured reply.
        attempt_id (UUID): Attempt the caller reserved under, minted with workflow.uuid4().
        within (timedelta): The attempt's reserved wall time.
        stop (Callable[[], bool]): The caller's cancel condition, such as its steering status.

    Returns:
        (Asked) The attempt's result, receipts, and wall time.
    """
    chain = 1 + len(assignment.fallbacks)
    lease = Lease(attempt_id, tuple(workflow.uuid4() for _ in range(chain)), 0)
    ask = _Ask(
        file_bytes(revision).decode(),
        dump_assignment(assignment),
        request.to_bytes().decode(),
        lease,
    )
    started = workflow.now()
    observations = []
    interrupted_by = None
    failure_reason = None
    for fence in itertools.count(1):
        left = within - (workflow.now() - started)
        activity_id = f"model/{attempt_id}/{fence}"
        current = replace(ask, lease=replace(lease, fence=fence))
        if not input_fits(current):
            result, failure_reason = "failed", "invalid_payload"
            elapsed = workflow.now() - started
            break
        handle = workflow.start_activity(
            "model",
            current,
            activity_id=activity_id,
            result_type=RawValue,
            schedule_to_close_timeout=timedelta(seconds=left // timedelta(seconds=1)),
            heartbeat_timeout=HEARTBEAT,
            # A server retry would repeat the fence, and the gateway refuses a second holder.
            retry_policy=RetryPolicy(maximum_attempts=1),
            # The close must not seal the claim before the gateway records the cancel.
            cancellation_type=ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
        )
        try:
            await workflow.wait_condition(lambda handle=handle: handle.done() or stop())
            if stop():
                interrupted_by = "stop"
        except asyncio.CancelledError:
            interrupted_by = "workflow"
        if not handle.done():
            handle.cancel()
        while True:
            try:
                raw = await asyncio.shield(handle)
                ending = "completed"
                break
            except asyncio.CancelledError:
                interrupted_by = "workflow"
            except ActivityError as error:
                ending = (
                    "cancelled"
                    if is_cancelled_exception(error)
                    else "timed_out"
                    if isinstance(error.cause, exceptions.TimeoutError)
                    else "failed"
                )
                raw = None
                if ending == "failed":
                    failure_reason = "activity_failed"
                elif ending == "cancelled":
                    failure_reason = "activity_cancelled"
                break
        observations.append(ModelActivity(activity_id, fence, ending))
        result = "failed" if ending == "cancelled" else ending
        if ending == "completed" and raw is None:
            result, failure_reason = "failed", "invalid_result"
        if raw is not None:
            try:
                result = load_result(
                    workflow.payload_converter().from_payload(raw.payload), request
                )
            except Exception:
                result = "failed"
                failure_reason = "invalid_result"
        if interrupted_by is not None:
            result = "cancelled"
        elapsed = workflow.now() - started
        if result != "timed_out" or within - elapsed < timedelta(seconds=1):
            break
    closing_at = workflow.now()
    receipts, interrupted = await _closed(lease) if observations else ((), False)
    if interrupted:
        interrupted_by = "workflow"
    return Asked(
        "cancelled" if interrupted_by else result,
        receipts,
        min(elapsed, within),
        tuple(observations),
        workflow.now() - closing_at,
        interrupted_by,
        failure_reason,
    )


async def call_tool(
    revision: CampaignRevision,
    tool: str,
    arguments: str,
    *,
    within: timedelta,
    stop: Callable[[], bool],
) -> Material | Unreported:
    """Run one model tool call under the revision's authority. Call only from workflow code.

    A failure reports only "failed". The activity's error carries a fixed message and its cause
    stays in the worker log, so nothing this returns can relay backing text to a model, and the
    caller must not turn "failed" into material either. A stop or a workflow cancel abandons
    the call without waiting, because Toolset.call cannot be interrupted, and the worker logs
    how its thread ends.

    Args:
        revision (CampaignRevision): Revision whose Toolset the worker resolves.
        tool (str): Tool name as the model sent it.
        arguments (str): JSON text of the arguments as the model sent it.
        within (timedelta): Start-to-close timeout of the tool activity.
        stop (Callable[[], bool]): The caller's cancel condition.

    Returns:
        (Material | Unreported) The rendered result under its restrictions, or why there is none.
    """
    handle = workflow.start_activity_method(
        Activities.tool,
        _Use(file_bytes(revision).decode(), tool, arguments),
        start_to_close_timeout=within,
        heartbeat_timeout=HEARTBEAT,
        # run_signal logs a container run per execution, so a retry would run the source again.
        retry_policy=RetryPolicy(maximum_attempts=1),
        cancellation_type=ActivityCancellationType.TRY_CANCEL,
    )
    return await _settled(handle, stop, load_material)


async def _settled[T](
    handle: workflow.ActivityHandle[str], stop: Callable[[], bool], decode: Callable[[str], T]
) -> T | Unreported:
    """Wait for an activity, cancelling it when stop() turns true or the workflow is cancelled.

    A caught workflow cancel ends here. It, or a stop() true when the wait ends, returns
    "cancelled" whatever the activity reported, since one workflow task can carry both the
    activity's result and the stop. A workflow that catches its cancel can still await the
    cancelled activity and schedule more.
    """
    try:
        await workflow.wait_condition(lambda: handle.done() or stop())
        cancelled = stop()
    except asyncio.CancelledError:
        cancelled = True
    if not handle.done():
        handle.cancel()
        cancelled = True
    try:
        text = await handle
    except ActivityError as error:
        if cancelled:
            return "cancelled"
        return "timed_out" if isinstance(error.cause, exceptions.TimeoutError) else "failed"
    return "cancelled" if cancelled else decode(text)


async def _closed(lease: Lease) -> tuple[tuple[Receipt, ...], bool]:
    """Close the lease and return its receipts, and whether a workflow cancel arrived meanwhile.

    Under asyncio.shield a workflow cancel interrupts only the wait, so the close completes and
    awaiting it again returns the receipts.
    """
    closing = workflow.start_activity_method(
        Activities.close,
        lease,
        activity_id=f"close/{lease.attempt_id}",
        start_to_close_timeout=CLOSE,
    )
    interrupted = False
    while True:
        try:
            return load_receipts(await asyncio.shield(closing)), interrupted
        except asyncio.CancelledError:
            interrupted = True


@dataclass(frozen=True)
class _Ask:
    """A model activity's input: canonical text and a Lease, which Temporal carries exactly.

    RunContext cannot cross a payload, so the worker resolves authority from the revision.
    """

    revision: str
    assignment: str
    request: str
    lease: Lease


@dataclass(frozen=True)
class _Use:
    revision: str
    tool: str
    arguments: str


class Closing:
    def __init__(self, gateway: Gateway | None) -> None:
        self._gateway = gateway

    @activity.defn(name="close")
    async def close(self, lease: Lease) -> str:
        """End the lease and return its receipts as dump_receipts text."""
        if self._gateway is None:
            raise ProviderError("This worker has no gateway")
        return dump_receipts(await asyncio.to_thread(self._gateway.close, lease))


class Activities(Closing):
    """The model, tool, and close activities of one worker.

    Every worker that polls a queue whose workflows call ask_model must bind a gateway on the
    same call log directory, because a resume or a close can run on any of them (D056). resolve
    returns the Authority of a pinned revision: a model call sends under its context, and a tool
    call runs under its Toolset, whose context ends the search window, so one resolver is the
    only source of a run's authority.
    """

    def __init__(
        self,
        gateway: Gateway,
        resolve: Callable[[CampaignRevision], Authority],
        *,
        grace: timedelta = GRACE,
    ) -> None:
        """Bind the operator's gateway and authority resolver.

        Args:
            gateway (Gateway): The gateway every model call of this worker uses.
            resolve (Callable[[CampaignRevision], Authority]): A revision's run authority.
            grace (timedelta, optional): How long an adapter whose event is set has to return.
                A model activity sets the event at its own deadline minus grace, so an activity
                with no more than grace left sets it at once.
        """
        super().__init__(gateway)
        self._resolve = resolve
        self._grace = grace

    @property
    def definitions(self) -> tuple[Callable[..., Awaitable[str]], ...]:
        """Return the activities to register: Worker(..., activities=bound.definitions)."""
        return (self.model, self.tool, self.close)

    @activity.defn(name="model")
    async def model(self, ask: _Ask) -> str:
        """Send one request under ask.lease and return the gateway's result as dump_result text.

        The call runs under the authority of the revision ask names, never a later one. A cancel
        or the activity's local deadline sets the adapter's event, and the gateway then records
        the call as cancelled unless a record came first. Any failure but a cancel raises one
        fixed, non-retryable error, and its cause goes only to the worker log.
        """
        event = threading.Event()

        def send() -> str:
            context = self._resolve(parse_file(ask.revision.encode())).context
            assignment = load_assignment(ask.assignment)
            request = Request.from_bytes(ask.request.encode())
            result = self._gateway.call(context, assignment, request, lease=ask.lease, cancel=event)
            return dump_result(result)

        label = f"model call of attempt {ask.lease.attempt_id} under fence {ask.lease.fence}"
        return await _contained("Model", _flown(send, event, label, self._grace))

    @activity.defn(name="tool")
    async def tool(self, use: _Use) -> str:
        """Run one tool call under the revision's Toolset and return its material as text.

        Refusals are material, as Toolset.call renders them. Any exception raises one fixed,
        non-retryable error, and its cause goes only to the worker log.
        """

        def run() -> str:
            toolset = self._resolve(parse_file(use.revision.encode())).toolset
            return dump_material(toolset.call(use.tool, use.arguments).material)

        return await _contained("Tool", _flown(run, None, f"tool call {use.tool!r}", self._grace))


async def _contained(kind: Literal["Model", "Tool"], flight: Awaitable[str]) -> str:
    """Await flight, turning any failure but a cancel into one fixed, non-retryable error.

    The error is raised after the handler, so it has no cause or context, and the original goes
    only to the worker log. No backing text, such as a path that a SourceError quotes, reaches
    the workflow's failure, so the workflow has none to hand to a model.
    """
    try:
        return await flight
    except _Deadline:
        failure = f"{kind}Deadline"
    except Exception:
        activity.logger.exception("The %s call failed", kind.lower())
        failure = f"{kind}Failed"
    raise ApplicationError(_FAILURES[failure], type=failure, non_retryable=True)


async def _flown[T](
    work: Callable[[], T], event: threading.Event | None, label: str, grace: timedelta
) -> T:
    """Run work on a daemon thread and heartbeat until it ends or the activity gives up on it.

    The flight is running until a cancel arrives or, when work has an event, the local deadline
    passes: the activity's own deadline minus grace, so a live worker stores the adapter's
    cancelled record before Temporal times the activity out. Stopping sets the event and keeps
    heartbeating for grace, or for no time when work has no event. The flight is settled when
    work returns in time and abandoned otherwise, and the worker log names an abandoned flight
    and how it ends. The thread is a daemon, so an abandoned one never blocks worker shutdown,
    as an asyncio.to_thread executor thread would.

    Raises:
        asyncio.CancelledError: A cancel arrived, and work has settled or been abandoned.
        _Deadline: The local deadline stopped work, and work outlived the grace.
    """
    future: Future[T] = Future()

    def run() -> None:
        try:
            future.set_result(work())
        except BaseException as error:
            future.set_exception(error)

    stop_at = None if event is None else _deadline(activity.info()) - grace
    # Set before the thread starts, so the adapter's first check sees a deadline already past.
    if stop_at is not None and datetime.now(UTC) >= stop_at:
        event.set()
    threading.Thread(target=run, name=label, daemon=True).start()
    waiting = asyncio.wrap_future(future)
    give_up: datetime | None = None
    cancel: asyncio.CancelledError | None = None
    while not waiting.done():
        now = datetime.now(UTC)
        if give_up is None and (cancel is not None or (stop_at is not None and now >= stop_at)):
            if event is not None:
                event.set()
            give_up = now + (timedelta(0) if event is None else grace)
        if give_up is not None and now >= give_up:
            break
        activity.heartbeat()
        until = give_up or stop_at
        timeout = BEAT if until is None else min(BEAT, (until - now).total_seconds())
        try:
            await asyncio.wait({waiting}, timeout=timeout)
        except asyncio.CancelledError as error:
            cancel = error
    if not waiting.done():
        activity.logger.error("The %s outlives its activity", label)
        waiting.add_done_callback(partial(_ended, label))
        if cancel is None:
            raise _Deadline
    elif cancel is not None and (error := waiting.exception()) is not None:
        activity.logger.error("The cancelled %s failed", label, exc_info=error)
    if cancel is not None:
        raise cancel
    return waiting.result()


def _deadline(info: activity.Info) -> datetime:
    """Return the earliest of the activity's schedule-to-close and start-to-close deadlines."""
    return min(
        start + limit
        for start, limit in (
            (info.scheduled_time, info.schedule_to_close_timeout),
            (info.started_time, info.start_to_close_timeout),
        )
        if limit
    )


def _ended(label: str, waiting: asyncio.Future) -> None:
    activity.logger.error("The abandoned %s ended", label, exc_info=waiting.exception())


class _Deadline(Exception):
    """The local deadline set the event, and work outlived the grace."""
