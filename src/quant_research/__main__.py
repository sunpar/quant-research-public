from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from collections.abc import Set
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal, get_args
from uuid import UUID, uuid4

from quant_research import prospective, public_data, reference
from quant_research.access import AccessError, Restrictions, issue_context
from quant_research.accounting import Charged, Ledger, Released, Reserved, load_ledger
from quant_research.accounting import json_default as ledger_json
from quant_research.backtest import BacktestError
from quant_research.campaigns import CampaignRevision, parse_file
from quant_research.catalog import CatalogError
from quant_research.compare import CompareError, compare_packages, render_comparison
from quant_research.confirmation import Period
from quant_research.datasets import DatasetError, ReadRecord, Refused
from quant_research.packages import open_package, parse_outcome
from quant_research.providers import CallLog, CallRecord, Closed, Dispatched, Entry
from quant_research.reconcile import SAHM_REALTIME, ReconcileError, reconcile, reconcile_series
from quant_research.records import (
    PackageError,
    json_default,
    parse_array,
    parse_digest,
    parse_fields,
    parse_text,
    parse_time,
)
from quant_research.replay import NewTrial, Repeat, ReplayError, Retry, replay_fixture
from quant_research.report import render_report
from quant_research.steering import check_revision_size


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay and inspect development experiment packages"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "verify"):
        command = commands.add_parser(name)
        command.add_argument("path", type=Path)
    compare = commands.add_parser("compare", help="Compare two verified packages")
    compare.add_argument("a", type=Path, metavar="package-a")
    compare.add_argument("b", type=Path, metavar="package-b")
    replay = commands.add_parser("replay", help="Run the fixed synthetic development fixture")
    replay.add_argument("path", type=Path)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--lock", type=Path, required=True, help="Caller-supplied, unverified lock")
    intent = replay.add_mutually_exclusive_group(required=True)
    intent.add_argument("--new", metavar="QUESTION")
    intent.add_argument("--repeat", type=Path, metavar="PACKAGE")
    intent.add_argument("--retry", type=Path, metavar="PACKAGE")
    replay.add_argument("--attempt-id", type=UUID)
    replay.add_argument("--history", type=Path, action="append", default=[])
    replay.add_argument("--reason", help="Reason for a new trial's links to supplied history")
    report = commands.add_parser("report", help="Write a Markdown report of a verified package")
    report.add_argument("path", type=Path)
    report.add_argument("--output")
    prospective_commands = commands.add_parser(
        "prospective", help="Register terminal packages and verify their prospective log"
    ).add_subparsers(dest="step", required=True)
    registration = prospective_commands.add_parser("register")
    registration.add_argument("path", type=Path)
    registration.add_argument("--root", type=Path, required=True)
    prospective_report = prospective_commands.add_parser("report")
    prospective_report.add_argument("--root", type=Path, required=True)
    observation = commands.add_parser(
        "observe", help="Record a caller-supplied outcome with original read evidence"
    )
    observation.add_argument("registration")
    observation.add_argument("--root", type=Path, required=True)
    observation.add_argument("--evidence", type=Path, required=True)
    observation.add_argument("--outcome", type=Path, required=True)
    observation.add_argument("--observed-at", required=True)
    reconciliation = commands.add_parser(
        "reconcile", help="Compare a package or a recomputed series with its reference"
    )
    targets = reconciliation.add_subparsers(dest="target", required=True)
    package_target = targets.add_parser(
        "package", help="Check a package's diagnostic against a reference and its raw input"
    )
    package_target.add_argument("path", type=Path)
    package_target.add_argument("--reference", type=Path, required=True)
    series = targets.add_parser(
        "series", help="Compare the recomputed real-time Sahm series with FRED's"
    )
    series.add_argument("--root", type=Path, required=True)
    data = commands.add_parser("data", help="Pull, ingest, validate, and approve public data")
    steps = data.add_subparsers(dest="step", required=True)
    for name, help_text in (
        ("pull", "Download each dataset and keep its original bytes"),
        ("ingest", "Store each latest pull as a vintage and document it in the catalog"),
    ):
        step = steps.add_parser(name, help=help_text)
        step.add_argument("datasets", nargs="*", default=list(public_data.PILOT))
        step.add_argument("--root", type=Path, required=True)
    steps.choices["ingest"].add_argument(
        "--provider", action="append", default=[], help="Provider approved to receive the data"
    )
    validate = steps.add_parser("validate", help="Record a manual validation attestation")
    validate.add_argument("--evidence", type=Path, required=True)
    validate.add_argument("--summary", required=True)
    approve = steps.add_parser("approve", help="Approve a validated dataset for agent reads")
    for step in (validate, approve):
        step.add_argument("dataset")
        step.add_argument("--root", type=Path, required=True)
        step.add_argument("--actor", required=True)
    pilot = commands.add_parser("reference", help="Run the frozen reference experiment")
    pilot_run = pilot.add_subparsers(dest="step", required=True).add_parser(
        "run", help="Run every reference case and save one package per case"
    )
    pilot_run.add_argument("--root", type=Path, required=True)
    pilot_run.add_argument("--output", type=Path, required=True)
    pilot_run.add_argument(
        "--lock", type=Path, required=True, help="Caller-supplied, unverified lock"
    )
    pilot_run.add_argument(
        "--experiment", type=Path, help="Experiment file to run instead of the committed one"
    )
    connected = argparse.ArgumentParser(add_help=False)
    connected.add_argument(
        "--address", default="127.0.0.1:7233", help="Temporal frontend of the research workers"
    )
    declaring = argparse.ArgumentParser(add_help=False)
    declaration = declaring.add_argument_group(
        "declaration", "Restrictions of the command's text, which reaches a model only if declared"
    )
    declaration.add_argument(
        "--available-at",
        metavar="TIME",
        help="UTC time like 2025-06-01T00:00:00Z when the text's facts were available",
    )
    declaration.add_argument(
        "--label", action="append", default=[], help="Restriction label of the text; repeatable"
    )
    declaration.add_argument(
        "--approved-for",
        action="append",
        default=[],
        metavar="PROVIDER",
        help="Provider the text is approved for; repeatable",
    )
    campaign = commands.add_parser(
        "campaign", help="Run a long-running worker and start, inspect, or control research loops"
    )
    campaign_steps = campaign.add_subparsers(dest="step", required=True)
    campaign_worker = campaign_steps.add_parser(
        "worker", help="Run a research worker from a JSON deployment configuration"
    )
    campaign_worker.add_argument("config", type=Path, metavar="CONFIG_FILE")
    campaign_start = campaign_steps.add_parser(
        "start", parents=[connected], help="Start the campaign's research loop under a revision"
    )
    campaign_start.add_argument(
        "revision", type=Path, help="Revision file as campaigns.save_revision writes it"
    )
    campaign_start.add_argument("--task-queue", required=True, help="Queue a research worker polls")
    campaign_status = campaign_steps.add_parser(
        "status",
        parents=[connected],
        help="Print the loop's state, steps, limits, spend, ledger, and receipts",
    )
    campaign_status.add_argument("campaign_id", type=UUID, metavar="CAMPAIGN")
    campaign_status.add_argument(
        "--call-log",
        type=Path,
        required=True,
        metavar="DIR",
        help="Call log directory that every research worker of the queue writes",
    )
    for action in CONTROLS:
        control = campaign_steps.add_parser(
            action,
            parents=[connected, declaring],
            help=f"{action.capitalize()} the loop, recording the actor and the reason",
        )
        control.add_argument("campaign_id", type=UUID, metavar="CAMPAIGN")
        control.add_argument("--actor", required=True)
        control.add_argument("--reason", required=True)
    campaign_steer = campaign_steps.add_parser(
        "steer", parents=[connected, declaring], help="Run every later ask under a child revision"
    )
    campaign_steer.add_argument(
        "revision",
        type=Path,
        help="Child revision file; its reason is the text, its author the actor",
    )
    args = parser.parse_args()
    if args.command in ("prospective", "observe"):
        return _prospective(parser, args)
    if args.command == "campaign":
        return _campaign(parser, args)
    if args.command == "data":
        now = datetime.now(UTC).replace(microsecond=0)
        try:
            if args.step == "pull":
                key = os.environ.get("FRED_API_KEY")
                results = [
                    asdict(public_data.pull(args.root, name, now=now, api_key=key))
                    for name in args.datasets
                ]
            elif args.step == "ingest":
                results = [
                    asdict(public_data.ingest(args.root, name, now=now, approved_for=args.provider))
                    for name in args.datasets
                ]
            elif args.step == "validate":
                status = public_data.validate(
                    args.root,
                    args.dataset,
                    actor=args.actor,
                    evidence=args.evidence.read_bytes(),
                    summary=args.summary,
                    now=now,
                )
                results = [{"dataset": args.dataset, "status": status}]
            else:
                status = public_data.approve(args.root, args.dataset, actor=args.actor, now=now)
                results = [{"dataset": args.dataset, "status": status}]
        except (public_data.PublicDataError, CatalogError, DatasetError, OSError) as error:
            parser.exit(1, f"Data {args.step} failed: {error}\n")
        print(json.dumps(results, default=json_default, indent=2, sort_keys=True))
        return
    if args.command == "reference":
        context = issue_context(
            clearances=(),
            cutoff=datetime.now(UTC).replace(microsecond=0),
            generation=reference.GENERATION,
            providers=(),
        )
        rows = []
        try:
            experiment = (
                reference.frozen()
                if args.experiment is None
                else reference.load_experiment(args.experiment.read_bytes())
            )
            catalog, lock = public_data.load_catalog(args.root), args.lock.read_bytes()
            for case in reference.CASES:
                package = reference.run(
                    case,
                    experiment,
                    catalog=catalog,
                    store=args.root / "datasets",
                    context=context,
                    lock=lock,
                    destination=args.output,
                )
                if isinstance(package, Refused):
                    parser.exit(1, f"Reference run of {case.name} refused: {package.message}\n")
                outcome = package.record.selected_trial.attempts[-1].outcome
                rows.append(
                    {
                        "case": case.name,
                        "status": outcome.status,
                        "expected": sorted(case.expected),
                        "as_expected": outcome.status in case.expected,
                        "package_path": str(package.path),
                        "package_digest": package.digest,
                        "detail": outcome.detail,
                    }
                )
        except (
            reference.ReferencePipelineError,
            DatasetError,
            CatalogError,
            PackageError,
            ReconcileError,
            BacktestError,
            AccessError,
            OSError,
        ) as error:
            parser.exit(1, f"Reference run failed: {error}\n")
        print(json.dumps(rows, indent=2, sort_keys=True))
        if not all(row["as_expected"] for row in rows):
            parser.exit(1, "A reference case ended outside its expected statuses\n")
        return
    if args.command == "replay":
        if (args.retry is not None) != (args.attempt_id is not None):
            parser.error("--retry requires --attempt-id, which is only valid for a retry")
        if (args.history or args.reason is not None) and args.new is None:
            parser.error("--history and --reason are only valid for a new trial")
        if args.reason is not None and not args.history:
            parser.error("--reason requires --history")
        try:
            if args.new is not None:
                run_intent = NewTrial(
                    args.new,
                    tuple(open_package(path) for path in args.history),
                    reason=(args.reason if args.reason is not None else NewTrial.reason),
                )
            elif args.repeat is not None:
                run_intent = Repeat(open_package(args.repeat))
            else:
                run_intent = Retry(open_package(args.retry), args.attempt_id)
            package = replay_fixture(
                args.path, args.output, intent=run_intent, lock=args.lock.read_bytes()
            )
        except (ReplayError, PackageError, OSError) as error:
            parser.exit(1, f"Replay failed: {error}\n")
        attempt = package.record.selected_trial.attempts[-1]
        print(
            json.dumps(
                {
                    "package_path": str(package.path),
                    "package_digest": package.digest,
                    "trial_id": str(package.record.trial_id),
                    "attempt_id": str(attempt.id),
                    "run_id": str(attempt.run_id),
                    **asdict(attempt.outcome),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "reconcile" and args.target == "series":
        context = issue_context(
            clearances=(),
            cutoff=datetime.now(UTC).replace(microsecond=0),
            generation=SAHM_REALTIME.name,
            providers=(),
        )
        try:
            result = reconcile_series(
                SAHM_REALTIME, public_data.load_catalog(args.root), args.root / "datasets", context
            )
        except (ReconcileError, DatasetError, CatalogError, PackageError, OSError) as error:
            parser.exit(1, f"Reconciliation failed: {error}\n")
        sys.stdout.buffer.write(result.record())
        if not result.agreed:
            parser.exit(1, "Reconciliation found disagreements\n")
        return
    if args.command == "compare":
        try:
            comparison = compare_packages(open_package(args.a), open_package(args.b))
        except PackageError as error:
            parser.exit(1, f"Package verification failed: {error}\n")
        except CompareError as error:
            parser.exit(1, f"Comparison failed: {error}\n")
        sys.stdout.buffer.write(render_comparison(comparison).encode())
        return
    try:
        package = open_package(args.path)
    except PackageError as error:
        parser.exit(1, f"Package verification failed: {error}\n")
    if args.command == "report":
        try:
            report = render_report(package)
        except PackageError as error:
            parser.exit(1, f"Cannot render report: {error}\n")
        if args.output is None:
            sys.stdout.buffer.write(report.encode())
            return
        try:
            requested = Path(args.output)
            output = requested.resolve()
            if any(
                path.exists() and path.samefile(package.path) for path in (output, *output.parents)
            ):
                parser.exit(1, "Report output must be outside the package directory\n")
            # resolve() collapses .. lexically; the OS walk raises ENOENT or ENOTDIR instead.
            requested.parent.stat()
            if os.path.basename(args.output) in ("", ".", "..") or (
                requested.exists(follow_symlinks=False) and not requested.is_file()
            ):
                parser.exit(1, f"Cannot write report: {args.output!r} must name a regular file\n")
            staged = output.parent / f".{uuid4()}"
            try:
                staged.write_bytes(report.encode())
                staged.replace(output)
            finally:
                staged.unlink(missing_ok=True)
        except OSError as error:
            parser.exit(1, f"Cannot write report to {args.output!r}: {error.strerror}\n")
        except (RuntimeError, ValueError) as error:
            parser.exit(1, f"Cannot write report: {error}\n")
        return
    if args.command == "reconcile":
        try:
            result = reconcile(package, args.reference)
        except (ReconcileError, PackageError, OSError) as error:
            parser.exit(1, f"Reconciliation failed: {error}\n")
        sys.stdout.buffer.write(result.record())
        if not result.agreed:
            parser.exit(1, "Reconciliation found disagreements\n")
        return
    if args.command == "verify":
        print(f"Verified {package.digest} at {package.path}")
        return
    print(
        json.dumps(
            {
                "package_digest": package.digest,
                "lineage_id": None
                if package.record.trial_id is None
                else str(package.record.selected_trial.lineage_id),
                "record": asdict(package.record),
                "spec_ids": {
                    str(trial.trial_id): trial.specification.spec_id
                    for trial in package.record.trials
                },
                "artifacts": [asdict(artifact) for artifact in package.artifact_records],
            },
            default=str,
            indent=2,
            sort_keys=True,
        )
    )


type _Action = Literal["pause", "resume", "cancel"]
CONTROLS = get_args(_Action.__value__)


@dataclass(frozen=True)
class _Start:
    revision: CampaignRevision
    task_queue: str

    @property
    def campaign_id(self) -> UUID:
        return self.revision.campaign_id


@dataclass(frozen=True)
class _Control:
    """A pause, resume, or cancel, whose reason reaches a model only under restrictions (D063)."""

    campaign_id: UUID
    action: _Action
    actor: str
    reason: str
    restrictions: Restrictions | None


@dataclass(frozen=True)
class _Steer:
    """A steer to a child revision, whose reason is the text restrictions declare."""

    revision: CampaignRevision
    restrictions: Restrictions | None

    @property
    def campaign_id(self) -> UUID:
        return self.revision.campaign_id


@dataclass(frozen=True)
class _Status:
    campaign_id: UUID
    call_log: CallLog


type Command = _Start | _Control | _Steer | _Status


@dataclass(frozen=True)
class CallCheck:
    call_id: str
    attempt_id: str
    state: Literal[
        "matched",
        "cost_unreported",
        "refused",
        "pending",
        "differs",
        "uncharged",
        "unrecorded",
        "unsealed",
    ]
    outcome: str | None


@dataclass(frozen=True)
class LiveReceipts:
    """Per-call evidence at an observation; agreement may include explicitly pending calls.

    reserved counts all captured reservations; settled counts completed model asks only.
    """

    reserved: int
    settled: int
    calls: tuple[CallCheck, ...]
    unclosed: tuple[str, ...]
    agreed: bool


@dataclass(frozen=True)
class LedgerReceipts:
    """A closed loop's sent calls, joined by call ID with its ledger's model-call charges.

    The ledger reserved every attempt the loop made, so the log's entries under those attempts
    are the campaign's, whatever else the log holds. A sent call and its charge are matched when
    the charge repeats the call's attempt, role, and usage, cost_unreported when it differs only
    as D056's fallback charges a cost the ledger cannot hold exactly, and differs otherwise.
    uncharged names sent calls without a charge, unrecorded names charges without a sent call,
    and unsealed names claims without a record, so the log is not the queue's or is partial.
    """

    reserved: int
    sent: int
    charged: int
    matched: int
    refused: int
    unknown: int
    cost_unreported: tuple[str, ...]
    differs: tuple[str, ...]
    uncharged: tuple[str, ...]
    unrecorded: tuple[str, ...]
    unsealed: tuple[str, ...]
    agreed: bool


def _prospective(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    try:
        if args.command == "observe":
            evidence = parse_fields(
                json.loads(args.evidence.read_bytes()), "procedure window reads", "evidence"
            )
            window = parse_fields(evidence["window"], "dataset_id start end", "window")
            reads = []
            for raw in parse_array(evidence["reads"], "reads"):
                value = parse_fields(raw, "record vintage scored", "read evidence")
                identities = [
                    parse_fields(item, "key revision", "scored identity")
                    for item in parse_array(value["scored"], "scored")
                ]
                reads.append(
                    prospective.ReadEvidence(
                        ReadRecord.from_bytes(
                            (
                                args.evidence.parent / parse_text(value["record"], "record path")
                            ).read_bytes()
                        ),
                        (
                            args.evidence.parent / parse_text(value["vintage"], "vintage path")
                        ).read_bytes(),
                        tuple((item["key"], item["revision"]) for item in identities),
                    )
                )
            event = prospective.observe(
                args.root,
                args.registration,
                procedure=parse_digest(evidence["procedure"]),
                window=Period(
                    window["dataset_id"], parse_time(window["start"]), parse_time(window["end"])
                ),
                reads=tuple(reads),
                outcome=parse_outcome(json.loads(args.outcome.read_bytes())),
                observed_at=parse_time(args.observed_at),
            )
            result = asdict(event)
        elif args.step == "register":
            registration = prospective.register_package(args.root, open_package(args.path))
            result = {"registration_id": registration.registration_id, **asdict(registration)}
        else:
            result = [asdict(entry) for entry in prospective.load_log(args.root).report()]
    except (ValueError, OSError, RecursionError) as error:
        parser.exit(1, f"Prospective command failed: {error}\n")
    print(json.dumps(result, default=prospective.json_default, indent=2, sort_keys=True))


def _campaign(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate and run a long-running worker or a command on a campaign's loop."""
    try:
        if args.step == "worker":
            from quant_research.research_worker import load_worker

            configured = load_worker(args.config)
            configured.run()
            return
        command = _command(args)
    except (OSError, ValueError, RecursionError) as error:
        parser.exit(1, f"Campaign {args.step} failed: {error}\n")
    # Temporal loads only after the checks, which therefore run where it is not installed.
    from temporalio.exceptions import FailureError, TemporalError

    try:
        output, agreed = asyncio.run(_drive(args.address, command))
    except (OSError, ValueError, TemporalError) as error:
        # A refused update or a failed run arrives wrapped, and its cause holds the reason.
        cause = error.cause if isinstance(error, TemporalError) else None
        reason = cause.message if isinstance(cause, FailureError) else error
        parser.exit(1, f"Campaign {args.step} failed: {reason}\n")
    print(json.dumps(output, default=ledger_json, indent=2, sort_keys=True))
    if not agreed:
        parser.exit(1, "Receipt reconciliation found disagreements\n")


def _command(args: argparse.Namespace) -> Command:
    """Return the command a campaign command line asks for, with its files read and checked.

    Raises:
        OSError: A revision file cannot be read.
        ValueError: A revision is not canonical, a declaration lacks its time or the time is not
            the UTC text that records.format_time writes, or the call log does not replay.
        RecursionError: A revision nests too deeply to decode.
    """
    match args.step:
        case "start":
            return _Start(parse_file(args.revision.read_bytes()), args.task_queue)
        case "steer":
            content = args.revision.read_bytes()
            check_revision_size(content)
            return _Steer(parse_file(content), _declared(args))
        case "status":
            return _Status(args.campaign_id, CallLog(args.call_log))
        case action:
            return _Control(args.campaign_id, action, args.actor, args.reason, _declared(args))


def _declared(args: argparse.Namespace) -> Restrictions | None:
    """Return the restrictions a command line declares for its text, or None for none."""
    if args.available_at is not None:
        return Restrictions(
            available_at=parse_time(args.available_at, "--available-at time"),
            labels=frozenset(args.label),
            approved_for=frozenset(args.approved_for),
        )
    if args.label or args.approved_for:
        raise ValueError("--label and --approved-for declare text only beside --available-at")
    return None


async def _drive(address: str, command: Command) -> tuple[dict[str, object], bool]:
    """Carry out a checked command on its campaign's loop.

    Returns:
        (tuple[dict[str, object], bool]) The command's JSON output, and whether its receipts
            agree, which only status can deny.
    """
    from temporalio.client import Client, WorkflowExecutionStatus

    from quant_research import research

    try:
        client = await Client.connect(address)
    except RuntimeError as error:
        raise ConnectionError(f"Cannot connect to Temporal at {address}: {error}") from None
    campaign_id = str(command.campaign_id)
    # The workflow ID is the campaign ID (D063).
    handle = client.get_workflow_handle_for(research.ResearchLoop.run, campaign_id)
    match command:
        case _Start(revision, task_queue):
            started = await research.start(client, revision, task_queue)
            return {
                "campaign_id": campaign_id,
                "revision": revision.revision_id,
                "run_id": started.result_run_id,
            }, True
        case _Control(_, action, actor, reason, restrictions):
            state = await research.control(
                handle, action, actor=actor, reason=reason, restrictions=restrictions
            )
            return {"campaign_id": campaign_id, "state": state}, True
        case _Steer(revision, restrictions):
            state = await research.steer(handle, revision, restrictions=restrictions)
            return {
                "campaign_id": campaign_id,
                "revision": revision.revision_id,
                "state": state,
            }, True
        case _Status(_, call_log):
            # Each branch reads the log after the report, so the log holds every call it counts.
            description = await handle.describe()
            if description.status == WorkflowExecutionStatus.RUNNING:
                observed = await research.observe(handle, client=client, description=description)
                report, ledger = observed.report, observed.ledger
                receipts = live_receipts(ledger, call_log.load(), settled=observed.settled)
            else:
                report = await research.record(handle)
                ledger = load_ledger(report.ledger.encode())
                receipts = ledger_receipts(ledger, call_log.load())
            lines = [asdict(line) for line in ledger.report()]
            shown = {key: value for key, value in asdict(report).items() if key != "steering"}
            return shown | {"ledger": lines, "receipts": asdict(receipts)}, receipts.agreed


def live_receipts(
    ledger: Ledger, entries: tuple[Entry, ...], *, settled: frozenset[UUID]
) -> LiveReceipts:
    """Join captured reservations and charges with later call evidence, allowing unsettled work."""
    attempts = {event.work.attempt_id for event in ledger.events if isinstance(event, Reserved)}
    finished = settled | {
        event.attempt_id for event in ledger.events if isinstance(event, Released)
    }
    charges = {
        event.id: event
        for event in ledger.events
        if isinstance(event, Charged) and event.usage.model_calls
    }
    records = {
        entry.call_id: entry
        for entry in entries
        if isinstance(entry, CallRecord)
        and (entry.attempt_id in attempts or entry.call_id in charges)
    }
    claims = {
        entry.call_id: entry
        for entry in entries
        if isinstance(entry, Dispatched) and entry.attempt_id in attempts
    }
    checks = []
    for call_id in sorted(records.keys() | claims.keys() | charges.keys()):
        record, charge = records.get(call_id), charges.get(call_id)
        if record is not None:
            attempt, outcome = record.attempt_id, record.outcome
            if record.outcome == "refused":
                state = "refused" if charge is None else "differs"
            elif charge is not None:
                state = _verdict(record, charge)
            else:
                state = "uncharged" if attempt in finished else "pending"
        elif charge is not None:
            attempt, outcome, state = charge.attempt_id, None, "unrecorded"
        else:
            attempt, outcome = claims[call_id].attempt_id, None
            state = "unsealed" if attempt in finished else "pending"
        checks.append(CallCheck(str(call_id), str(attempt), state, outcome))
    closed = {entry.attempt_id for entry in entries if isinstance(entry, Closed)}
    unclosed = tuple(map(str, sorted((settled & attempts) - closed)))
    return LiveReceipts(
        reserved=len(attempts),
        settled=len(settled & attempts),
        calls=tuple(checks),
        unclosed=unclosed,
        agreed=not unclosed
        and all(
            check.state in {"matched", "cost_unreported", "refused", "pending"} for check in checks
        ),
    )


def ledger_receipts(ledger: Ledger, entries: tuple[Entry, ...]) -> LedgerReceipts:
    """Join a closed loop's sent calls with its ledger's model-call charges by call ID.

    Args:
        ledger (Ledger): The closed loop's ledger.
        entries (tuple[Entry, ...]): The call log.

    Returns:
        (LedgerReceipts) Each sent call and charge, by how they agree.
    """
    attempts = {event.work.attempt_id for event in ledger.events if isinstance(event, Reserved)}
    sent, refused, unsealed = _logged(entries, attempts)
    # Wall time and tool compute are charged without model calls, and every receipt with one.
    charges = {
        event.id: event
        for event in ledger.events
        if isinstance(event, Charged) and event.usage.model_calls
    }
    verdicts: defaultdict[str, list[str]] = defaultdict(list)
    for call_id in sorted(sent):
        verdicts[_verdict(sent[call_id], charges.get(call_id))].append(str(call_id))
    differs, uncharged = tuple(verdicts["differs"]), tuple(verdicts["uncharged"])
    unrecorded = tuple(map(str, sorted(charges.keys() - sent.keys())))
    return LedgerReceipts(
        reserved=len(attempts),
        sent=len(sent),
        charged=len(charges),
        matched=len(verdicts["matched"]),
        refused=refused,
        unknown=sum(record.outcome == "unknown" for record in sent.values()),
        cost_unreported=tuple(verdicts["cost_unreported"]),
        differs=differs,
        uncharged=uncharged,
        unrecorded=unrecorded,
        unsealed=unsealed,
        agreed=not (differs or uncharged or unrecorded or unsealed),
    )


def _logged(
    entries: tuple[Entry, ...], attempts: Set[UUID]
) -> tuple[dict[UUID, CallRecord], int, tuple[str, ...]]:
    """Return the attempts' sent records by call ID, refused count, and claims never recorded."""
    records = {
        entry.call_id: entry
        for entry in entries
        if isinstance(entry, CallRecord) and entry.attempt_id in attempts
    }
    sent = {call_id: record for call_id, record in records.items() if record.outcome != "refused"}
    claims = (
        entry.call_id
        for entry in entries
        if isinstance(entry, Dispatched) and entry.attempt_id in attempts
    )
    unsealed = tuple(map(str, sorted(call_id for call_id in claims if call_id not in records)))
    return sent, len(records) - len(sent), unsealed


def _verdict(
    record: CallRecord, charge: Charged | None
) -> Literal["matched", "cost_unreported", "differs", "uncharged"]:
    """Return how the charge under a sent call's ID accounts for the call."""
    if charge is None:
        return "uncharged"
    if (charge.attempt_id, charge.role) != (record.attempt_id, record.role):
        return "differs"
    if charge.usage == record.usage:
        return "matched"
    usage = record.usage
    fallback = replace(usage, cost_usd=Decimal(0), unreported=usage.unreported | {"cost_usd"})
    return "cost_unreported" if charge.usage == fallback else "differs"


if __name__ == "__main__":
    sys.exit(main())
