"""Run the platform-owned research loop for one campaign as one durable Temporal workflow (D063).

    researcher  start, control, steer, status, record: the client surface
    operator    worker(client, queue, gateway, authorize, archive=archive), or run_worker in
                its own process: ResearchLoop, the model, tool, close, and archive activities,
                and the loop's own activities, on a queue of their own
    ResearchLoop
                one workflow per campaign: asks the researcher role for a step, carries out its
                move, and records it, until a conclusion, a stop, a cancel, or a limit ends it;
                the single writer of its ledger, steering log, step records, and transcript,
                all in memory and rebuilt by replay (D042)
    Authorized  the activities of this module, each resolving a revision on the worker:
                opening returns its role's prompt, assignment, authority digest, floor, and the
                protocol, as text; experiment runs one candidate as the next trial of the
                campaign's lineage (D065); forfeit records a candidate whose experiment ended
                without a result as a failed trial; evaluate measures one frozen shortlist on
                the held-back window (D066)

    Move        a completed reply's schema values, parsed: UseTool | Experiment | Conclude | Stop
    Checkpoint  one of the revision's interventions, parsed: Every | Before
    Role        what the opening activity resolved for one revision, decoded
    Transcript  what the model reads after its role prompt: whole entries, kept per revision
    StepRecord  one ask as the researcher sees it, with no model text
    Ending      how the loop stopped, and the workflow result
    Unrecorded  a candidate the loop ended without recording in its lineage, for recovery
    Assessment  the frozen shortlist, held-back measurements, conclusion, and critique
    Report      the status query, and the closed record with both logs

Every ordinary ask is one ledger attempt on one shared run. Three standing attempts reserve
one closing selection chain and two single-model interpretations before search starts. A
refused reservation, the step cap, or the history guard still ends through a closing selection.
Every worker shares the archive root, call-log directory, and numerical publication storage.
Bind Gateway(CallLog(calls), ...) and CampaignArchive(root, calls) to the same calls path.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Literal, NamedTuple
from uuid import UUID

from temporalio import activity, workflow
from temporalio.api.common.v1 import Payload
from temporalio.client import (
    Client,
    WorkflowExecutionDescription,
    WorkflowExecutionStatus,
    WorkflowHandle,
    WorkflowHistory,
)
from temporalio.common import RetryPolicy, WorkflowIDConflictPolicy, WorkflowIDReusePolicy
from temporalio.converter import DataConverter
from temporalio.exceptions import ActivityError, ApplicationError, WorkflowAlreadyStartedError
from temporalio.worker import Worker
from temporalio.workflow import ActivityCancellationType

with workflow.unsafe.imports_passed_through():
    from quant_research import acceptance, datasets, evaluation, reference, signals
    from quant_research.access import Restrictions, RunContext, derive
    from quant_research.accounting import (
        Budget,
        InexactUsage,
        Ledger,
        Reserved,
        Usage,
        Work,
        dump_ledger,
        load_ledger,
    )
    from quant_research.activities import BEAT, HEARTBEAT, Activities, Asked, ask_model, call_tool
    from quant_research.agent_tools import TOOLS
    from quant_research.campaigns import (
        Authority,
        CampaignRevision,
        ResourceLimits,
        RulesError,
        check_link,
        file_bytes,
        parse_file,
    )
    from quant_research.catalog import Catalog
    from quant_research.locators import published_locations
    from quant_research.model_provenance import (
        CutoffCheck,
        ModelRecord,
        RoleAssignment,
        json_default,
        load_assignment,
        parse_model_record,
    )
    from quant_research.packages import (
        ArtifactInput,
        PackageRecord,
        VerifiedPackage,
        open_package,
        save_package,
    )
    from quant_research.providers import (
        Answer,
        CallLog,
        CallRecord,
        Gateway,
        Material,
        Receipt,
        Request,
        Value,
        dump_material,
        load_material,
    )
    from quant_research.records import (
        canonical_bytes,
        format_time,
        keep_object,
        parse_array,
        parse_fields,
        parse_text,
        parse_time,
        unique_keys,
        write_once,
    )
    from quant_research.report import CAMPAIGN_ROLE
    from quant_research.sources import open_generation
    from quant_research.steering import (
        Event,
        Steer,
        SteeringLog,
        check_revision_size,
        dump_declaration,
        load_declaration,
        log_bytes,
    )

type Asking = Literal["answered", "rate_limited", "failed", "cancelled"]
# Temporal's JSON converter cannot decode a PEP 695 alias, so these stay plain assignments.
State = Literal["running", "paused", "cancelled", "concluded", "stopped", "inconclusive"]
Outcome = Literal["concluded", "stopped", "inconclusive", "cancelled"]
Classification = Literal["supported", "null", "inconclusive", "no_result"]
Result = Literal[
    "ran",
    "rejected",
    "refused",
    "concluded",
    "stopped",
    "set_aside",
    "rate_limited",
    "failed",
    "cancelled",
]

ROLE = "researcher"
# The actor of the loop's own steering events. control refuses it, so only a researcher resumes.
ACTOR = "runner"
# The reply schema, whose order enters every request digest.
SCHEMA = (("action", str), ("arguments", str), ("rationale", str))
# Copied from execution.py, which is P09's to change: the wall time one ask reserves, and the
# bytes of one part query, under Temporal's 512 KiB payload warning.
TIMEOUT = timedelta(minutes=10)
PART = 2**18
TOOL_TIMEOUT = timedelta(minutes=5)
OPENING_TIMEOUT = timedelta(minutes=1)
EXPERIMENT_TIMEOUT = timedelta(minutes=5)
FORFEIT_TIMEOUT = timedelta(minutes=1)
# Ordinary asks per loop, failed and rate-limited ones included. The closing ask is one more.
STEP_CAP = 100
# Temporal terminates a run past 50 MB of history, so the loop closes out first.
HISTORY_BYTES = 32 * 2**20
# Bytes of Request.to_bytes. The fixed part of a request (role prompt, protocol, notices) must fit
# ROLE_BYTES, so the rest of the bound holds a dataset read of MAX_RESULT_BYTES and its reply.
BOUND = 384 * 2**10
ROLE_BYTES = 64 * 2**10
# Updates while paused add history that no step check sees, so researcher commands are bounded.
NOTE_BYTES = 4 * 2**10
COMMANDS = 256

CLOSING = (
    "The session's budget is spent, so this is the last step. Reply with the action conclude, "
    'with arguments exactly {"trials":["search trial UUID", ...]}, or stop.'
)
SCOPE = "Provisional development evidence only; no confirmation or investment claim"
CONCLUSION = (
    'Interpret the held-back evidence. Reply with exactly {"classification":"supported",'
    '"citations":["held-back trial UUID", ...],"explanation":"text"}. '
    "classification must be supported, null, inconclusive, or no_result. "
    "A supported claim must cite supported held-back evidence. Null requires every shortlisted "
    "result to be null. Cite only evaluated held-back IDs. Do not request tools or experiments."
)
CRITIQUE = (
    "Critique the conclusion against all held-back evidence. Reply with exactly "
    '{"decision":"accept","citations":["held-back trial UUID", ...],'
    '"explanation":"text"}. decision must be accept or reject. '
    "Reject claims the evidence does not justify. "
    "Do not request tools or experiments."
)
OMITTED = "Earlier steps are omitted."
WITHHELD = "This step is larger than the prompt can hold, so it is withheld."
TRIALS_SPENT = "The campaign has run its {limit} trials, so this experiment did not run."
SET_ASIDE = (
    "The researcher steered the session while this reply waited, so it is set aside. "
    "Take the next step."
)
_WORDS = ("experiment", "conclude", "stop")
_RELEASED = {
    "answered": "successful",
    "rate_limited": "inconclusive",
    "failed": "failed",
    "cancelled": "failed",
}
_UNOPENED = {
    "OpeningFailed": "The revision cannot be opened; its cause is in the worker log",
    "RulesUnavailable": "The revision's rules cannot be resolved; its cause is in the worker log",
    "PromptTooLarge": (
        f"The fixed role prompt exceeds {ROLE_BYTES} bytes or the opening exceeds {BOUND} bytes"
    ),
}
# The Ending reason of an opening failure, "role" for any type not named here.
_UNADMITTED = {"PromptTooLarge": "prompt", "RulesUnavailable": "rules"}

_EVERY = re.compile(r"checkpoint every ([1-9][0-9]{0,5}) steps")
_BEFORE = "checkpoint before conclude"


class ResearchError(ValueError):
    """Report a refused start, control, or steer of a research loop."""


@dataclass(frozen=True)
class UseTool:
    """Any action that is not experiment, conclude, or stop.

    tool and arguments go to Toolset.call as the model sent them, which renders an unknown tool
    or bad arguments as refusal material, so agent_tools alone decides which tools exist.
    """

    tool: str
    arguments: str


@dataclass(frozen=True)
class Experiment:
    """A request to run the program in arguments, the program's own JSON document."""

    program: str


@dataclass(frozen=True)
class Conclude:
    """The frozen shortlist as JSON text with exactly the trials field."""

    finding: str


@dataclass(frozen=True)
class Stop:
    """The model ends the session without a finding."""


type Move = UseTool | Experiment | Conclude | Stop


def parse_move(values: Mapping[str, Value]) -> Move:
    """Return the move that a completed reply's schema values name.

    The gateway completes a reply only when its values hold exactly SCHEMA's text fields, so
    this cannot fail.

    Args:
        values (Mapping[str, Value]): A completed reply's values.

    Returns:
        (Move) The move its action names.
    """
    match values["action"]:
        case "experiment":
            return Experiment(values["arguments"])
        case "conclude":
            return Conclude(values["arguments"])
        case "stop":
            return Stop()
        case tool:
            return UseTool(tool, values["arguments"])


@dataclass(frozen=True)
class Every:
    """checkpoint every N steps: pause before the ask that follows every Nth ask."""

    steps: int


@dataclass(frozen=True)
class Before:
    """checkpoint before conclude: hold a conclude or stop move before it ends the loop."""


type Checkpoint = Every | Before


def checkpoints(revision: CampaignRevision) -> tuple[Checkpoint, ...]:
    """Parse each of the revision's interventions as a checkpoint.

    Args:
        revision (CampaignRevision): Revision whose interventions to parse.

    Returns:
        (tuple[Checkpoint, ...]) One checkpoint per intervention, in order.

    Raises:
        ResearchError: An intervention is neither "checkpoint every N steps", with N from 1 to
            STEP_CAP, nor "checkpoint before conclude".
    """
    parsed: list[Checkpoint] = []
    for text in revision.specification.interventions:
        every = _EVERY.fullmatch(text)
        if text == _BEFORE:
            parsed.append(Before())
        elif every is not None and int(every[1]) <= STEP_CAP:
            parsed.append(Every(int(every[1])))
        else:
            raise ResearchError(
                f"Intervention {text!r} is neither 'checkpoint every N steps' with N from 1 to "
                f"{STEP_CAP} nor {_BEFORE!r}"
            )
    return tuple(parsed)


@dataclass(frozen=True)
class Role:
    """What the opening activity resolved for one revision, decoded.

    prompt is the first material of every request (D062), and protocol is runner text under the
    prompt's restrictions. floor is what a declared steering note must fit to reach every model
    of the chain: available_at is the run's cutoff, labels the labels every chain provider's
    policy admits within the run's clearances, and approved_for the chain's providers. tools
    names the tools the protocol describes.
    """

    prompt: Material
    protocol: Material
    assignment: RoleAssignment
    authority: str
    floor: Restrictions
    tools: frozenset[str]
    checkpoints: tuple[Checkpoint, ...]
    critics: tuple[tuple[ModelRecord, Material, Restrictions], ...] = ()

    @property
    def chain(self) -> int:
        """Return how many models one ask may send to."""
        return 1 + len(self.assignment.fallbacks)


def check_declaration(declared: Restrictions, floor: Restrictions) -> None:
    """Raise ResearchError unless text under declared may reach every model a floor names.

    Args:
        declared (Restrictions): Restrictions a researcher declared for steering text.
        floor (Restrictions): A Role's floor.
    """
    if declared.available_at > floor.available_at:
        raise ResearchError(
            f"Declared text available at {format_time(declared.available_at)} is after the "
            f"run's cutoff {format_time(floor.available_at)}"
        )
    if missing := sorted(floor.approved_for - declared.approved_for):
        raise ResearchError(f"Declared text must be approved for the role's providers {missing}")
    if extra := sorted(declared.labels - floor.labels):
        raise ResearchError(f"Declared labels {extra} are more than the role's providers receive")


class _Entry(NamedTuple):
    material: Material
    cost: int


class Transcript:
    """What the model reads after its role prompt: whole entries per revision, oldest first.

    An entry is a step's reply and result, or a declared steering note, as one Material. A
    revision's requests show only its own entries, so text read under a parent's authority
    never reaches a request under its child. An entry that could not fit beside a ROLE_BYTES
    fixed part is withheld whole. prompt() drops entries from the oldest end, for good, until
    the request fits BOUND or none is left, so the transcript is a pure function of the calls
    made on it, which replay repeats. Once one of a revision's entries is dropped, its requests
    carry OMITTED under the restrictions of the prompt and of everything dropped, so a
    request's restrictions never fall below those of an entry an earlier request showed.
    """

    def __init__(self) -> None:
        self._entries: defaultdict[str, deque[_Entry]] = defaultdict(deque)
        self._omitted: dict[str, Restrictions] = {}

    def add_step(self, revision: str, number: int, reply: Material, result: Material) -> str:
        """Append a step under the revision its ask ran under.

        Args:
            revision (str): ID of the revision the step's ask ran under.
            number (int): The step's number.
            reply (Material): The step's reply.
            result (Material): What the step's move produced, tool material or runner text.

        Returns:
            (str) SHA-256 of the entry's text as one ASCII JSON string, as the call log names
                a reply.
        """
        text = f"Step {number}: {reply.text}\nResult: {result.text}"
        return self._add(revision, Material(text, derive(reply.restrictions, result.restrictions)))

    def add_note(self, revision: str, event: Event, declared: Restrictions) -> None:
        """Append a researcher's steering text under its declared restrictions.

        Args:
            revision (str): ID of the head revision when the note arrived.
            event (Event): Steering event whose reason is the note.
            declared (Restrictions): The event's declared restrictions.
        """
        self._add(revision, Material(f"Researcher {event.action}: {event.reason}", declared))

    def prompt(self, revision: str, role: Role, *, closing: bool = False) -> Request | None:
        """Return the next request under a revision, dropping its oldest entries until it fits.

        Args:
            revision (str): ID of the revision the ask runs under.
            role (Role): That revision's role.
            closing (bool, optional): Whether the request asks for the closing reply.

        Returns:
            (Request | None) The role prompt, the protocol, OMITTED once anything is dropped, the
                entries left, and CLOSING on a closing request, or None when the restrictions
                of the revision's entries alone pass BOUND.
        """
        shown = self._entries[revision]
        while True:
            request = self._request(revision, role, closing)
            excess = len(request.to_bytes()) - BOUND
            if excess <= 0:
                return request
            if not shown:
                return None
            while excess > 0 and shown:
                dropped = shown.popleft()
                excess -= dropped.cost
                self._omitted[revision] = derive(
                    self._omitted.get(revision, role.prompt.restrictions),
                    dropped.material.restrictions,
                )

    def _add(self, revision: str, material: Material) -> str:
        cost = len(json.dumps(material.text)) + 2
        if cost > BOUND - ROLE_BYTES:
            material = Material(WITHHELD, material.restrictions)
            cost = len(json.dumps(WITHHELD)) + 2
        self._entries[revision].append(_Entry(material, cost))
        return hashlib.sha256(json.dumps(material.text).encode()).hexdigest()

    def _request(self, revision: str, role: Role, closing: bool) -> Request:
        omitted = self._omitted.get(revision)
        return Request(
            (
                role.prompt,
                role.protocol,
                *(() if omitted is None else (Material(OMITTED, omitted),)),
                *(entry.material for entry in self._entries[revision]),
                *((Material(CLOSING, role.prompt.restrictions),) if closing else ()),
            ),
            SCHEMA,
        )


@dataclass(frozen=True)
class Opened:
    """The opening activity's result, as text, so a request rebuilt on replay reads history.

    A tool added to agent_tools therefore cannot change a request of an open campaign.
    """

    prompt: str
    protocol: str
    tools: tuple[str, ...]
    assignment: str
    authority: str
    floor: str
    critics: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Trying:
    """An experiment's input, as text, so no RunContext crosses a payload.

    package is the head digest the loop last recorded, "" before the first trial, and trial
    the ID the loop minted for this candidate.
    """

    revision: str
    package: str
    trial: str
    program: str

    @property
    def data(self) -> bytes:
        # A JSON escape such as \ud800 reaches a reply as a lone surrogate, which strict UTF-8
        # cannot encode; these bytes make it a malformed_json trial instead of a failed activity.
        return self.program.encode("utf-8", "surrogatepass")


@dataclass(frozen=True)
class Tried:
    """An experiment's result, as text: the new head's digest, the trial's status, its summary.

    material is dump_material of the summary under its restrictions: a few KiB, never package
    bytes.
    """

    package: str
    status: str
    material: str


@dataclass(frozen=True)
class _Evaluating:
    revision: str
    package: str
    trials: tuple[tuple[str, str], ...]
    adjustment_trials: int


@dataclass(frozen=True)
class Evaluated:
    package: str
    material: str


@dataclass(frozen=True)
class ClosedCampaign:
    """The original closed record, admitted revisions, and finite interaction history."""

    report: bytes
    revisions: tuple[bytes, ...]
    history: bytes
    failure: bytes | None = None
    acceptance: bytes | None = None


@dataclass(frozen=True)
class AuthorityEvidence:
    """Opening-time bytes and the numerical publication directory they authorize."""

    revision: str
    artifacts: tuple[ArtifactInput, ...]
    destination: Path


class CampaignArchive:
    """Publish one immutable campaign snapshot per explicit export key."""

    def __init__(
        self,
        root: Path,
        calls: Path,
        *,
        importer: Callable[[VerifiedPackage], object] | None = None,
    ) -> None:
        self.root, self.calls, self.importer = root, calls, importer
        for name in ("objects", "pins", "selections", "receipts", "bases"):
            (root / name).mkdir(parents=True, exist_ok=True)

    def _keep(self, artifacts: tuple[ArtifactInput, ...]) -> list[dict]:
        return [
            {
                "name": item.name,
                "digest": keep_object(self.root / "objects", item.data),
                "media_type": item.media_type,
                "role": item.role,
            }
            for item in artifacts
        ]

    def _read(self, records: list[dict]) -> tuple[ArtifactInput, ...]:
        artifacts = []
        for item in records:
            data = (self.root / "objects" / item["digest"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != item["digest"]:
                raise ResearchError("An archive object differs from its selected digest")
            artifacts.append(ArtifactInput(item["name"], data, item["media_type"], item["role"]))
        return tuple(artifacts)

    def pin(self, authority: Authority) -> AuthorityEvidence:
        """Preserve resolved inputs before opening returns, independently of later resolution."""
        prefix = f"campaign-{authority.revision}"
        artifacts = [
            ArtifactInput(
                f"{prefix}-authority", authority.record(), "application/json", "campaign-authority"
            ),
            ArtifactInput(
                f"{prefix}-catalog",
                authority.search.catalog.to_bytes(),
                "application/json",
                "campaign-catalog",
            ),
            ArtifactInput(
                f"{prefix}-context",
                authority.context.to_bytes(),
                "application/json",
                "campaign-context",
            ),
            ArtifactInput(
                f"{prefix}-search-context",
                authority.search.context.to_bytes(),
                "application/json",
                "campaign-search-context",
            ),
            ArtifactInput(
                f"{prefix}-rules", authority.search.rules.data, "application/json", "campaign-rules"
            ),
            ArtifactInput(f"{prefix}-lock", authority.search.lock, "text/plain", "campaign-lock"),
            ArtifactInput(
                f"{prefix}-models",
                json.dumps(asdict(authority.roles), default=json_default, sort_keys=True).encode(),
                "application/json",
                "campaign-models",
            ),
        ]
        for role, prompt in sorted(authority.prompts.items()):
            artifacts.append(
                ArtifactInput(
                    f"{prefix}-prompt-{role}",
                    dump_material(prompt).encode(),
                    "application/json",
                    "campaign-prompt",
                )
            )
        for entry in authority.search.catalog.search():
            if authority.context.permits(entry.name):
                supplied = datasets.read(
                    authority.search.catalog, authority.search.store, entry.name, authority.context
                )
                if isinstance(supplied, datasets.Supplied):
                    data = (
                        authority.search.store / entry.name / f"{supplied.record.vintage}.json"
                    ).read_bytes()
                    supplied.record.replay(data)
                    if authority.context.dataset_periods is not None:
                        binding = hashlib.sha256(
                            f"{authority.revision}:{entry.name}".encode()
                        ).hexdigest()
                        artifacts.append(
                            ArtifactInput(
                                f"campaign-read-{binding}",
                                supplied.record.to_bytes(),
                                "application/json",
                                "campaign-read",
                            )
                        )
                    artifacts.append(
                        ArtifactInput(
                            f"campaign-vintage-{supplied.record.vintage}",
                            data,
                            "application/json",
                            "campaign-vintage",
                        )
                    )
        source_map = {}
        paths = [authority.toolset.sources / "corpora" / f"{authority.corpus.id}.json"]
        extractions = set()
        for identity, digest in sorted(authority.generations.items()):
            generation = open_generation(authority.toolset.sources, identity)
            if generation.digest != digest:
                raise ResearchError("A pinned source generation differs from its authority")
            extractions.add(generation.extraction_id)
            occurrence = generation.occurrence
            paths.extend(
                [
                    authority.toolset.sources / "occurrences" / f"{occurrence.id}.json",
                    authority.toolset.sources / "blobs" / occurrence.blob_digest,
                    authority.toolset.sources / "extractions" / f"{generation.extraction_id}.json",
                    *sorted((authority.toolset.sources / "generations" / identity).iterdir()),
                ]
            )
        for locator in published_locations(authority.toolset.sources, authority.toolset.context):
            if locator.extraction_id in extractions:
                paths.extend(
                    sorted((authority.toolset.sources / "locations" / locator.id).iterdir())
                )
        for path in paths:
            if not path.name.startswith("."):
                relative = str(path.relative_to(authority.toolset.sources))
                data = path.read_bytes()
                name = f"campaign-source-{hashlib.sha256(data).hexdigest()}"
                source_map[relative] = name
                artifacts.append(
                    ArtifactInput(name, data, "application/octet-stream", "campaign-source")
                )
        artifacts.append(
            ArtifactInput(
                f"{prefix}-source-map",
                canonical_bytes(source_map),
                "application/json",
                "campaign-source-map",
            )
        )
        evidence = AuthorityEvidence(
            authority.revision,
            tuple({item.name: item for item in artifacts}.values()),
            authority.search.destination,
        )
        record = canonical_bytes(
            {
                "destination": str(evidence.destination.absolute()),
                "artifacts": self._keep(evidence.artifacts),
            }
        )
        if not write_once(self.root / "pins" / authority.revision, record):
            raise ResearchError("A revision's opening authority changed after it was pinned")
        return evidence

    def publish(
        self, closed: ClosedCampaign, *, key: str, authorities: tuple[AuthorityEvidence, ...] = ()
    ) -> VerifiedPackage:
        """Publish or reopen the first complete evidence selection for key, then import it."""
        identity = hashlib.sha256(key.encode()).hexdigest()
        selected = self.root / "selections" / identity
        if not selected.exists():
            selection = self._select(closed, key, authorities)
            write_once(selected, canonical_bytes(selection))
        selection = json.loads(selected.read_bytes())
        base = (
            None
            if selection["base"] is None
            else open_package(self.root / "bases" / "unselected" / selection["base"])
        )
        package = save_package(
            self.root / "packages",
            PackageRecord(None, (), ()) if base is None else base.record,
            self._read(selection["artifacts"]),
        )
        receipt = self.root / "receipts" / identity
        if not write_once(receipt, package.digest.encode()):
            raise ResearchError("An archive key published conflicting packages")
        from quant_research.prospective import register_package

        register_package(self.root / "prospective", package)
        if self.importer is not None:
            self.importer(package)
        return package

    def _select(
        self, closed: ClosedCampaign, key: str, authorities: tuple[AuthorityEvidence, ...]
    ) -> dict:
        from quant_research.__main__ import ledger_receipts

        report = json.loads(closed.report)
        revisions = tuple(parse_file(data) for data in closed.revisions)
        head_revision = next(
            revision for revision in revisions if revision.revision_id == report["revision"]
        )
        pinned = {item.revision: item for item in authorities}
        for revision in revisions:
            path = self.root / "pins" / revision.revision_id
            if revision.revision_id not in pinned and path.exists():
                raw = json.loads(path.read_bytes())
                pinned[revision.revision_id] = AuthorityEvidence(
                    revision.revision_id, self._read(raw["artifacts"]), Path(raw["destination"])
                )
        artifacts = [
            ArtifactInput(
                "campaign-original-report",
                closed.report,
                "application/json",
                "campaign-original-report",
            ),
            ArtifactInput(
                "campaign-original-ending",
                canonical_bytes(report["ending"]),
                "application/json",
                "campaign-original-ending",
            ),
            ArtifactInput(
                "campaign-history", closed.history, "application/json", "campaign-history"
            ),
            ArtifactInput(
                "campaign-steering",
                report["steering"].encode(),
                "application/json",
                "campaign-steering",
            ),
            ArtifactInput(
                "campaign-ledger", report["ledger"].encode(), "application/json", "campaign-ledger"
            ),
            ArtifactInput(
                "campaign-acceptance",
                closed.acceptance
                if closed.acceptance is not None
                else canonical_bytes(
                    {
                        "owner": "ResearchLoop",
                        "events": [],
                        "reason": "ResearchLoop records no acceptance log",
                    }
                ),
                "application/json",
                "campaign-acceptance",
            ),
        ]
        if closed.failure is not None:
            artifacts.append(
                ArtifactInput(
                    "campaign-operational-failure",
                    closed.failure,
                    "application/json",
                    "campaign-operational-failure",
                )
            )
        for data, revision in zip(closed.revisions, revisions, strict=True):
            artifacts.append(
                ArtifactInput(
                    f"campaign-revision-{revision.revision_id}",
                    data,
                    "application/json",
                    "campaign-revision",
                )
            )
            if revision.revision_id in pinned:
                artifacts.extend(pinned[revision.revision_id].artifacts)
            else:
                artifacts.append(
                    ArtifactInput(
                        f"campaign-unavailable-{revision.revision_id}",
                        canonical_bytes(
                            {
                                "revision": revision.revision_id,
                                "reason": "Opening did not pin a resolved authority",
                            }
                        ),
                        "application/json",
                        "campaign-unavailable",
                    )
                )
        ledger = load_ledger(report["ledger"].encode())
        attempts = {event.work.attempt_id for event in ledger.events if isinstance(event, Reserved)}
        calls = CallLog(self.calls)
        all_entries = calls.load()
        transcripts = set()
        for index, entry in enumerate(all_entries):
            if entry.attempt_id not in attempts:
                continue
            artifacts.append(
                ArtifactInput(
                    f"campaign-call-{index}",
                    (self.calls / f"{index}.json").read_bytes(),
                    "application/json",
                    "campaign-call",
                )
            )
            if isinstance(entry, CallRecord):
                if entry.outcome != "refused":
                    calls.sent(entry)
                    transcripts.add(entry.request_digest)
                if entry.reply_digest is not None:
                    calls.reply(entry)
                    transcripts.add(entry.reply_digest)
            elif hasattr(entry, "request_digest"):
                transcripts.add(entry.request_digest)
        transcript_root = self.calls.resolve().with_name(f"{self.calls.resolve().name}.transcripts")
        for digest in sorted(transcripts):
            data = (transcript_root / digest).read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ResearchError("A call transcript differs from its recorded digest")
            artifacts.append(
                ArtifactInput(
                    f"campaign-transcript-{digest}", data, "application/json", "campaign-transcript"
                )
            )
        artifacts.append(
            ArtifactInput(
                "campaign-receipts",
                canonical_bytes(asdict(ledger_receipts(ledger, all_entries))),
                "application/json",
                "campaign-receipts",
            )
        )
        assessment = report.get("assessment")
        if assessment is not None:
            for name in ("shortlist", "evidence", "conclusion", "critique"):
                if assessment[name] is not None:
                    artifacts.append(
                        ArtifactInput(
                            f"campaign-original-{name}",
                            assessment[name].encode(),
                            "application/json",
                            f"campaign-original-{name}",
                        )
                    )
        original = report["package"]
        recovered = None
        evidence = pinned.get(report["revision"])
        if evidence is not None and assessment is not None and assessment.get("recovery"):
            by_role = {item.role: item.data for item in evidence.artifacts}
            search = reference.Search(
                head_revision.specification.question,
                head_revision.campaign_id,
                reference.load_rules(by_role["campaign-rules"]),
                Catalog.from_bytes(by_role["campaign-catalog"]),
                Path(),
                RunContext.from_bytes(by_role["campaign-search-context"]),
                by_role["campaign-lock"],
                evidence.destination,
            )
            recovered = reference.recover_assessment(search, assessment["recovery"])
        head = original if recovered is None else recovered.package
        base = None
        if head:
            if evidence is None:
                raise ResearchError("Numerical evidence needs its pinned publication directory")
            base = open_package(evidence.destination / "unselected" / head)
            if original:
                original_package = (
                    base
                    if original == head
                    else open_package(evidence.destination / "unselected" / original)
                )
                artifacts.append(
                    ArtifactInput(
                        "campaign-original-manifest",
                        (original_package.path / "manifest.json").read_bytes(),
                        "application/json",
                        "campaign-original-manifest",
                    )
                )
            base_artifacts = base.artifacts()
            base = save_package(self.root / "bases", base.record, base_artifacts)
            artifacts.extend(base_artifacts)
        checks = []
        for evidence in pinned.values():
            by_role = {item.role: item.data for item in evidence.artifacts}
            models = {
                json.dumps(raw, sort_keys=True): parse_model_record(raw)
                for assignment in json.loads(by_role["campaign-models"])["assignments"]
                for raw in (assignment["primary"], *assignment["fallbacks"])
            }
            rules = reference.load_rules(by_role["campaign-rules"])
            for model in models.values():
                for name, period in (("search", rules.search), ("held_back", rules.held_back)):
                    check = CutoffCheck(model, period.start, period.end)
                    checks.append(
                        {
                            "revision": evidence.revision,
                            "period": name,
                            "model": json.loads(json.dumps(asdict(model), default=json_default)),
                            "start": period.start.isoformat(),
                            "end": period.end.isoformat(),
                            "status": check.status,
                            "sentence": check.sentence,
                        }
                    )
        snapshot = {
            "schema_version": 1,
            "key": key,
            "report": report,
            "question": head_revision.specification.question,
            "original_head": original,
            "recovered_head": None if recovered is None else recovered.package,
            "cutoff_checks": checks,
            "acceptance": "ResearchLoop records no acceptance log"
            if closed.acceptance is None
            else "Supplied acceptance log preserved verbatim",
            "acceptance_decisions": []
            if closed.acceptance is None
            else [
                {
                    "attempt": str(event.attempt_id),
                    "supersedes": None if event.supersedes is None else str(event.supersedes),
                    "actor": event.actor,
                    "reason": event.reason,
                    "at": format_time(event.at),
                }
                for event in acceptance.load_log(closed.acceptance).events
                if isinstance(event, acceptance.Accepted)
            ],
            "history": (
                "Supplied finite Temporal history; automatic exports stop before terminal "
                "archive scheduling"
            ),
            "failure": closed.failure is not None,
        }
        artifacts.append(
            ArtifactInput(
                "campaign-snapshot", canonical_bytes(snapshot), "application/json", CAMPAIGN_ROLE
            )
        )
        return {
            "base": None if base is None else base.digest,
            "artifacts": self._keep(
                tuple(
                    sorted(
                        {item.name: item for item in artifacts}.values(), key=lambda item: item.name
                    )
                )
            ),
        }

    @activity.defn(name="archive_campaign")
    async def archive(self) -> str:
        info = activity.info()
        handle = activity.client().get_workflow_handle(
            info.workflow_id, run_id=info.workflow_run_id
        )
        parts = [await handle.query(ResearchLoop.archive_part, 0)]
        while len(parts[-1]) == PART:
            parts.append(await handle.query(ResearchLoop.archive_part, PART * len(parts)))
        raw = json.loads(b"".join(parts))
        history = await handle.fetch_history()
        boundary = next(
            event.event_id
            for event in history.events
            if event.HasField("activity_task_scheduled_event_attributes")
            and event.activity_task_scheduled_event_attributes.activity_id == info.activity_id
        )
        prefix = (
            WorkflowHistory(
                info.workflow_id,
                tuple(event for event in history.events if event.event_id < boundary),
            )
            .to_json()
            .encode()
        )
        closed = ClosedCampaign(
            raw["report"].encode(),
            tuple(text.encode() for text in raw["revisions"]),
            prefix,
            None if raw["failure"] is None else raw["failure"].encode(),
        )
        result = await asyncio.to_thread(
            self.publish, closed, key=f"{info.workflow_id}/{info.workflow_run_id}/final"
        )
        return result.digest


class Authorized:
    """Register the loop and archive activities beside Activities(...).definitions.

    Bind the archive to the gateway's call-log path and every queue worker's shared storage.
    """

    def __init__(
        self, authorize: Callable[[CampaignRevision], Authority], archive: CampaignArchive
    ) -> None:
        """Bind the operator's authority resolver and required archive.

        Args:
            authorize (Callable[[CampaignRevision], Authority]): A revision's run authority,
                functools.partial(campaigns.resolve, setup=setup) in production.
            archive (CampaignArchive): Shared archive using the gateway's call-log directory.
        """
        self._authorize = authorize
        self._archive = archive

    @property
    def definitions(self) -> tuple[Callable[..., Awaitable[object]], ...]:
        """Return the activities to register: Worker(..., activities=authorized.definitions)."""
        return (
            self.opening,
            self.experiment,
            self.forfeit,
            self.evaluate,
            self.recover,
            self._archive.archive,
        )

    @activity.defn(name="opening")
    async def opening(self, revision: str) -> Opened:
        """Resolve a revision and return what its ROLE needs, as text.

        Any failure raises one fixed, non-retryable error, and its cause goes only to the worker
        log, so no backing text reaches the workflow.
        """
        try:
            parsed = parse_file(revision.encode())
            authority = await asyncio.to_thread(self._authorize, parsed)
            await asyncio.to_thread(self._archive.pin, authority)
            [assignment] = [item for item in authority.roles.assignments if item.role == ROLE]
            policies = {policy.provider: policy for policy in authority.policies}
            providers = {model.provider for model in (assignment.primary, *assignment.fallbacks)}
            floor = Restrictions(
                available_at=authority.context.cutoff,
                labels=authority.context.clearances.intersection(
                    *(policies[provider].labels for provider in providers)
                ),
                approved_for=providers,
            )
            prompt = authority.prompts[ROLE]
            critics = tuple(
                json.dumps(
                    {
                        "models": [
                            {
                                "model": asdict(model),
                                "floor": dump_declaration(
                                    Restrictions(
                                        available_at=authority.context.cutoff,
                                        labels=authority.context.clearances
                                        & policies[model.provider].labels,
                                        approved_for={model.provider},
                                    )
                                ),
                            }
                            for model in (item.primary, *item.fallbacks)
                        ],
                        "prompt": None
                        if item.role == ROLE
                        else dump_material(authority.prompts[item.role]),
                    },
                    default=json_default,
                    sort_keys=True,
                )
                for item in sorted(
                    authority.roles.assignments, key=lambda item: item.role != "critic"
                )
                if item.role in (ROLE, "critic")
            )
            opened = Opened(
                dump_material(prompt),
                _protocol(authority.search, parsed.specification.limits.max_trials),
                tuple(tool.name for tool in TOOLS),
                json.dumps(asdict(assignment), default=json_default, sort_keys=True),
                hashlib.sha256(authority.record()).hexdigest(),
                dump_declaration(floor),
                critics,
            )
            notices = (
                Material(text, prompt.restrictions) for text in (opened.protocol, OMITTED, CLOSING)
            )
            size = len(Request((prompt, *notices), SCHEMA).to_bytes())
            payload_size = len(DataConverter.default.payload_converter.to_payload(opened).data)
        except RulesError:
            activity.logger.exception("The rules of a research revision cannot be resolved")
            failure = "RulesUnavailable"
        except Exception:
            activity.logger.exception("The opening of a research revision failed")
            failure = "OpeningFailed"
        else:
            if size <= ROLE_BYTES and payload_size <= BOUND:
                return opened
            # Checked here, so an oversized prompt never crosses a payload.
            failure = "PromptTooLarge"
        raise ApplicationError(_UNOPENED[failure], type=failure, non_retryable=True)

    @activity.defn(name="experiment")
    async def experiment(self, trying: _Trying) -> Tried:
        """Run one candidate as the next trial of the campaign's lineage, on its search window.

        The run takes a thread while the activity heartbeats, so a dead worker surfaces within
        HEARTBEAT. Any exception raises one fixed, non-retryable error, and its cause goes only
        to the worker log, so no backing text reaches the workflow.
        """

        def run() -> Tried:
            search = self._authorize(parse_file(trying.revision.encode())).search
            extended = reference.extend(
                search, trying.package or None, UUID(trying.trial), trying.data
            )
            material = Material(extended.summary, extended.restrictions)
            return Tried(extended.package.digest, extended.status, dump_material(material))

        running = asyncio.ensure_future(asyncio.to_thread(run))
        try:
            while not running.done():
                activity.heartbeat()
                await asyncio.wait({running}, timeout=BEAT)
            return running.result()
        except Exception:
            activity.logger.exception("The experiment of trial %s failed", trying.trial)
        raise ApplicationError(
            "The experiment failed; its cause is in the worker log",
            type="ExperimentFailed",
            non_retryable=True,
        )

    @activity.defn(name="forfeit")
    async def forfeit(self, trying: _Trying, started: str, ended: str) -> str:
        """Append a candidate whose experiment ended without a result as a failed trial.

        A retry only saves another immutable head, so the error it raises is retryable; its
        cause goes only to the worker log.

        Returns:
            (str) The new head's digest.
        """

        def run() -> str:
            search = self._authorize(parse_file(trying.revision.encode())).search
            return reference.forfeit(
                search,
                trying.package or None,
                UUID(trying.trial),
                trying.data,
                parse_time(started),
                parse_time(ended),
            )

        try:
            return await asyncio.to_thread(run)
        except Exception:
            activity.logger.exception("The forfeit of trial %s failed", trying.trial)
        raise ApplicationError(
            "The forfeit failed; its cause is in the worker log", type="ForfeitFailed"
        )

    @activity.defn(name="evaluate")
    async def evaluate(self, closing: _Evaluating) -> Evaluated:
        """Evaluate the entire frozen shortlist in one non-retried activity."""
        cancelled = threading.Event()

        def run() -> Evaluated:
            authority = self._authorize(parse_file(closing.revision.encode()))
            assessed = reference.assess(
                authority.search,
                authority.context,
                closing.package or None,
                tuple((UUID(source), UUID(trial)) for source, trial in closing.trials),
                adjustment_trials=closing.adjustment_trials,
                stop=cancelled.is_set,
            )
            result = Evaluated(
                assessed.package, dump_material(Material(assessed.summary, assessed.restrictions))
            )
            if len(DataConverter.default.payload_converter.to_payload(result).data) > BOUND:
                raise ValueError("The held-back evidence exceeds the activity payload bound")
            return result

        running = asyncio.ensure_future(asyncio.to_thread(run))
        try:
            while not running.done():
                activity.heartbeat()
                await asyncio.wait({running}, timeout=BEAT)
            return running.result()
        except Exception:
            activity.logger.exception("The held-back evaluation failed")
            raise ApplicationError(
                "The held-back evaluation failed; its cause is in the worker log",
                type="EvaluationFailed",
                non_retryable=True,
            ) from None
        finally:
            cancelled.set()

    @activity.defn(name="recover_assessment")
    async def recover(self, revision: str, recovery: str) -> Evaluated | None:
        """Read one snapshot of already published held-back evidence without measuring."""

        def read() -> Evaluated | None:
            search = self._authorize(parse_file(revision.encode())).search
            assessed = reference.recover_assessment(search, recovery)
            return (
                None
                if assessed is None
                else Evaluated(
                    assessed.package,
                    dump_material(Material(assessed.summary, assessed.restrictions)),
                )
            )

        try:
            result = await asyncio.to_thread(read)
            if len(DataConverter.default.payload_converter.to_payload(result).data) > BOUND:
                raise ValueError("The recovered evidence exceeds the activity payload bound")
            return result
        except Exception:
            activity.logger.exception("The held-back recovery snapshot failed")
            raise ApplicationError(
                "The held-back recovery failed; its cause is in the worker log",
                type="AssessmentRecoveryFailed",
                non_retryable=True,
            ) from None


def _protocol(search: reference.Search, trials: int) -> str:
    tools = "\n".join(
        f"- {tool.name}: {tool.description} Its arguments are JSON text matching "
        f"{json.dumps(tool.input_schema, sort_keys=True)}."
        for tool in TOOLS
    )
    window = search.rules.search
    return (
        "Reply with one JSON object whose keys are exactly action, arguments, and rationale, "
        "each a string. rationale says why you take the step.\n"
        f"Set action to a tool's name to call it with arguments:\n{tools}\n"
        "Set action to experiment to run the signal program in arguments, one JSON document in "
        f"the search grammar such as {signals.canonical(signals.SAHM_RULE).decode().strip()}, "
        f"on decisions from {window.start:%B %Y} to {window.end:%B %Y}. Tools and experiments "
        f"see only data available by {format_time(window.cutoff)}. Programs may read "
        f"{', '.join(search.universe)}. Every experiment is a trial, rejected and failed ones "
        f"included, and the campaign runs at most {trials}.\n"
        'Set action to conclude with arguments exactly {"trials":["search trial UUID", ...]} '
        "to freeze unique known trials for one held-back evaluation. An empty list is valid.\n"
        "Set action to stop to end the session without a finding."
    )


def _role(opened: Opened, revision: CampaignRevision) -> Role | str:
    """Decode the opening's text, or return why the loop cannot take the revision."""
    try:
        prompt = load_material(opened.prompt)
        assignment = load_assignment(opened.assignment)
        critics = []
        for raw in opened.critics:
            candidate = json.loads(raw)
            instructions = (
                prompt if candidate["prompt"] is None else load_material(candidate["prompt"])
            )
            critics.extend(
                (parse_model_record(item["model"]), instructions, load_declaration(item["floor"]))
                for item in candidate["models"]
            )
        role = Role(
            prompt,
            Material(opened.protocol, prompt.restrictions),
            assignment,
            opened.authority,
            load_declaration(opened.floor),
            frozenset(opened.tools),
            checkpoints(revision),
            tuple(critics),
        )
    except ResearchError:
        return "checkpoint"
    except ValueError:
        return "role"
    return role


@dataclass(frozen=True)
class StepRecord:
    """One ask and what its move did.

    at is when the record was made, and attempt the ledger and call-log attempt of the ask, so
    the reply is found from it. action is a tool name the protocol describes, experiment,
    conclude, or stop, "unknown" for any other action, and "" when no reply completed. digest
    names the entry the step added to the transcript, "" when it added none. trial is the ID
    the loop gave the candidate of an experiment it scheduled, so the records alone count the
    campaign's trials, and None on every other step.
    """

    number: int
    at: str
    revision: str
    attempt: str
    action: str
    result: Result
    digest: str
    trial: str | None = None


@dataclass(frozen=True)
class Ending:
    """How the loop stopped, and the workflow result.

    reason is "" for a conclusion or a stop. An inconclusive ending names the ledger limit that
    refused a reservation, "steps", "history", "prompt" when no request fits BOUND, "lineage"
    when a failed experiment's candidate could not be forfeited, or, when the first revision
    could not be taken, "role", "prompt", "rules", or "checkpoint". A cancelled one names
    "researcher" or "workflow".
    Finalization failures also name "shortlist", "evaluation", "interpretation", "failed", or
    "rate_limited". final is dump_material of the fresh conclusion when delivered, otherwise
    the selection or stop reply.
    """

    outcome: Outcome
    reason: str
    steps: int
    final: str | None


@dataclass(frozen=True)
class Unrecorded:
    """A candidate the loop ended without recording in its lineage, kept so it can be recovered.

    revision is the ID the experiment ran under, program the move's arguments as the model sent
    them, and started and ended when the loop scheduled the experiment and gave up on it.
    """

    revision: str
    trial: str
    program: str
    started: str
    ended: str


@dataclass(frozen=True)
class HeldBack:
    source: str
    trial: str
    status: str
    evaluation: str | None


@dataclass(frozen=True)
class Assessment:
    """The frozen selection, its held-back evidence, and both fresh interpretations."""

    revision: str
    reason: str
    search_package: str
    package: str
    trials: tuple[HeldBack, ...]
    shortlist: str
    evidence: str | None = None
    conclusion: str | None = None
    critique: str | None = None
    proposer: str = ""
    critic: str = ""
    independent: bool = False
    conclusion_calls: tuple[str, ...] = ()
    critique_calls: tuple[str, ...] = ()
    classification: Classification = "inconclusive"
    detail: str = "Finalization has not completed"
    scope: str = SCOPE
    recovery: str | None = None


@dataclass(frozen=True)
class Finalization:
    revision: str
    proposer: ModelRecord
    selection: Material
    trials: tuple[tuple[str, str], ...]
    reason: str


@dataclass(frozen=True)
class Report:
    """The loop as the researcher reads it, while it runs and after it closes.

    paused_by is "<actor>: <reason>" of the latest pause while the loop is paused. authority is
    the head revision's authority digest, "" before its opening returns. spent sums every
    charge by usage field as exact decimal text, with "unreported" naming the fields some
    charge left unreported. package is the digest of the lineage head the loop last recorded,
    "" before its first trial, and unrecorded the candidate an experiment left out of it when the
    loop ended during that experiment. steering and ledger are the logs' canonical text in a
    record and "" in a status.
    """

    workflow_id: str
    run_id: str
    state: State
    paused_by: str | None
    revision: str
    authority: str
    cap: int
    steps: tuple[StepRecord, ...]
    limits: dict[str, int | None]
    spent: dict[str, str]
    ending: Ending | None
    package: str
    unrecorded: Unrecorded | None
    steering: str
    ledger: str
    assessment: Assessment | None = None


@dataclass(frozen=True)
class Observation:
    """One live summary with its exact ledger prefix and model asks completed by _settle.

    Released unused reservations, including evaluation-only holds, are not settled asks.
    """

    report: Report
    ledger: Ledger
    settled: frozenset[UUID]


@dataclass(frozen=True)
class _Observation:
    report: Report
    event_count: int
    settled: tuple[str, ...]


@workflow.defn
class ResearchLoop:
    @workflow.init
    def __init__(self, revision: str) -> None:
        head = parse_file(revision.encode())
        self._revisions: dict[str, CampaignRevision] = {head.revision_id: head}
        self._roles: dict[str, Role] = {}
        self._head = head.revision_id
        self._ledger = Ledger({head.campaign_id: _budget(head.specification.limits)})
        self._settled: set[UUID] = set()
        self._steering = SteeringLog(head.campaign_id)
        self._transcript = Transcript()
        self._steps: list[StepRecord] = []
        self._hold: Work | None = None
        self._final_holds: list[Work] = []
        self._chain = 0
        self._closing = False
        self._ending: Ending | None = None
        self._gated = 0
        self._package = ""
        self._unrecorded: Unrecorded | None = None
        self._assessment: Assessment | None = None
        self._frozen = b""
        self._archive_view = b""
        self._failure: str | None = None

    @workflow.run
    async def run(self, revision: str) -> Ending:
        try:
            closing = await self._loop()
            ending = await self._finalize(closing) if isinstance(closing, Finalization) else closing
        except asyncio.CancelledError:
            ending = self._cancel()
        except Exception as error:
            self._failure = canonical_bytes(
                {
                    "type": type(error).__name__,
                    "reason": "Unexpected research implementation failure",
                }
            ).decode()
            await self._archive_closed()
            raise ApplicationError(
                "Unexpected research implementation failure",
                type=type(error).__name__,
                non_retryable=True,
            ) from None
        await self._archive_closed()
        return ending

    async def _archive_closed(self) -> None:
        async def finish() -> None:
            await workflow.wait_condition(workflow.all_handlers_finished)
            report = self._report(
                log_bytes(self._steering).decode(), dump_ledger(self._ledger).decode()
            )
            self._frozen = DataConverter.default.payload_converter.to_payload(report).data
            self._archive_view = canonical_bytes(
                {
                    "report": self._frozen.decode(),
                    "revisions": [
                        file_bytes(revision).decode() for revision in self._revisions.values()
                    ],
                    "failure": self._failure,
                }
            )
            await workflow.execute_activity(
                "archive_campaign",
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=RetryPolicy(initial_interval=timedelta(seconds=1)),
            )

        cleanup = asyncio.create_task(finish())
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                pass
        cleanup.result()

    async def _loop(self) -> Ending | Finalization:
        first = self._revisions[self._head]
        trial, working = workflow.uuid4(), workflow.uuid4()
        role = await self._admit(first)
        if isinstance(role, str):
            return self._cancel() if self._cancelled() else self._finish("inconclusive", role)
        self._roles[first.revision_id] = role
        hold = Work(first.campaign_id, trial, workflow.uuid4(), workflow.uuid4())
        if refusals := self._ledger.reserve(hold, _asking(role.chain)).refusals:
            return self._finish("inconclusive", refusals[0].limit)
        self._hold, self._chain = hold, role.chain
        for _ in range(2):
            final = Work(first.campaign_id, trial, workflow.uuid4(), workflow.uuid4())
            if refusals := self._ledger.reserve(final, _asking(1)).refusals:
                return self._finish("inconclusive", refusals[0].limit)
            self._final_holds.append(final)
        while True:
            await workflow.wait_condition(lambda: self._steering.status != "paused")
            if self._cancelled():
                return self._cancel()
            revision, role = self._revisions[self._head], self._roles[self._head]
            if self._checkpoint(role):
                continue
            attempt = Work(first.campaign_id, trial, working, workflow.uuid4())
            reason = self._capped()
            if reason is None:
                refusals = self._ledger.reserve(attempt, _asking(role.chain)).refusals
                reason = refusals[0].limit if refusals else None
            if reason is not None:
                return await self._close_out(reason)
            if (ending := await self._step(revision, role, attempt.attempt_id)) is not None:
                return ending

    async def _step(
        self, revision: CampaignRevision, role: Role, attempt: UUID
    ) -> Ending | Finalization | None:
        """Ask for one step and return an ending or frozen finalization when search ends."""
        number = len(self._steps) + 1
        request = self._transcript.prompt(revision.revision_id, role)
        if request is None:
            self._ledger.release(attempt, "inconclusive")
            return self._finish("inconclusive", "prompt")
        asked = await ask_model(
            revision, role.assignment, request, attempt, within=TIMEOUT, stop=self._cancelled
        )
        kind = self._settle(attempt, asked)
        if kind != "answered":
            self._ledger.release(attempt, _RELEASED[kind])
            self._record(number, revision, attempt, "", kind)
            if kind == "cancelled":
                return self._cancel()
            if kind == "rate_limited":
                self._pause(f"Rate limited on step {number}")
            return None
        reply = asked.result
        action = _label(reply.values["action"], role)
        match parse_move(reply.values):
            case UseTool(tool, arguments):
                started = workflow.now()
                used = await call_tool(
                    revision, tool, arguments, within=TOOL_TIMEOUT, stop=self._cancelled
                )
                # Tool compute is observed, so a campaign that calls no tool needs no compute.
                spent = Usage(compute_seconds=_seconds(workflow.now() - started))
                self._ledger.charge(workflow.uuid4(), attempt, "tools", spent)
                if not isinstance(used, Material):
                    kind = "cancelled" if used == "cancelled" else "failed"
                    self._ledger.release(attempt, "failed")
                    self._record(number, revision, attempt, action, kind)
                    return self._cancel() if kind == "cancelled" else None
                result, shown = "ran", used
            case Experiment(program):
                return await self._experiment(number, revision, role, attempt, reply, program)
            case Conclude() | Stop() as move:
                self._ledger.release(attempt, "successful")
                return await self._conclude(number, revision, role, attempt, reply, action, move)
        self._ledger.release(attempt, "successful")
        digest = self._transcript.add_step(revision.revision_id, number, reply.material, shown)
        self._record(number, revision, attempt, action, result, digest)
        return None

    async def _experiment(
        self,
        number: int,
        revision: CampaignRevision,
        role: Role,
        attempt: UUID,
        reply: Answer,
        program: str,
    ) -> Ending | None:
        """Run a candidate as the lineage's next trial, or refuse it once the trials are spent.

        The trial ID is minted before the activity is scheduled. An experiment that fails or
        times out adds no entry, and its candidate is forfeited as a failed trial, so every
        candidate the loop schedules is counted. A cancel during the experiment, or a forfeit
        that fails, ends the loop with the candidate in Report.unrecorded.
        """
        limit = revision.specification.limits.max_trials
        if sum(step.trial is not None for step in self._steps) >= limit:
            refusal = Material(TRIALS_SPENT.format(limit=limit), role.prompt.restrictions)
            self._ledger.release(attempt, "successful")
            digest = self._transcript.add_step(
                revision.revision_id, number, reply.material, refusal
            )
            self._record(number, revision, attempt, "experiment", "refused", digest)
            return None
        trying = _Trying(
            file_bytes(revision).decode(), self._package, str(workflow.uuid4()), program
        )
        started = workflow.now()
        running = workflow.start_activity_method(
            Authorized.experiment,
            trying,
            start_to_close_timeout=EXPERIMENT_TIMEOUT,
            heartbeat_timeout=HEARTBEAT,
            # A retry would run the candidate as a second trial under the same ID.
            retry_policy=RetryPolicy(maximum_attempts=1),
            cancellation_type=ActivityCancellationType.TRY_CANCEL,
        )
        # As activities._settled waits, which is private: a caught workflow cancel ends here.
        try:
            await workflow.wait_condition(lambda: running.done() or self._cancelled())
            cancelled = self._cancelled()
        except asyncio.CancelledError:
            cancelled = True
        if not running.done():
            running.cancel()
        try:
            tried: Tried | None = await running
        except ActivityError:
            tried = None
        ended = workflow.now()
        spent = Usage(compute_seconds=_seconds(ended - started))
        self._ledger.charge(workflow.uuid4(), attempt, "experiments", spent)
        if tried is None:
            self._ledger.release(attempt, "failed")
            failed = "cancelled" if cancelled else "failed"
            self._record(number, revision, attempt, "experiment", failed, trial=trying.trial)
            candidate = Unrecorded(
                revision.revision_id,
                trying.trial,
                program,
                format_time(started),
                format_time(ended),
            )
            if cancelled:
                self._unrecorded = candidate
                return self._cancel()
            return await self._forfeit(trying, candidate)
        # Kept even when a cancel came with the result, since the head holds the trial.
        self._package = tried.package
        result = "rejected" if tried.status == "rejected" else "ran"
        shown = load_material(tried.material)
        self._ledger.release(attempt, "successful")
        digest = self._transcript.add_step(revision.revision_id, number, reply.material, shown)
        self._record(number, revision, attempt, "experiment", result, digest, trying.trial)
        return self._cancel() if cancelled else None

    async def _forfeit(self, trying: _Trying, candidate: Unrecorded) -> Ending | None:
        """Record a candidate whose experiment ended without a result as a failed trial.

        A search whose record cannot be kept complete must stop, so a forfeit that fails ends the
        loop inconclusive, and a cancel during it ends the loop cancelled, each keeping the
        candidate in Report.unrecorded.
        """
        try:
            self._package = await workflow.execute_activity_method(
                Authorized.forfeit,
                args=[trying, candidate.started, candidate.ended],
                start_to_close_timeout=FORFEIT_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
        except ActivityError:
            self._unrecorded = candidate
            return self._cancel() if self._cancelled() else self._finish("inconclusive", "lineage")
        except asyncio.CancelledError:
            self._unrecorded = candidate
            return self._cancel()
        return None

    async def _conclude(
        self,
        number: int,
        revision: CampaignRevision,
        role: Role,
        attempt: UUID,
        reply: Answer,
        action: str,
        move: Conclude | Stop,
    ) -> Ending | Finalization | None:
        """End the loop on a conclusion or a stop, holding it first at a before checkpoint.

        A held move is set aside, and the loop asks again, when a steer or declared steering
        text arrived during the hold, because the reply could not have read it.
        """
        outcome = "concluded" if isinstance(move, Conclude) else "stopped"
        if Before() in role.checkpoints:
            arrived = len(self._steering.events)
            self._pause(f"Checkpoint before conclude holds step {number}")
            try:
                await workflow.wait_condition(lambda: self._steering.status != "paused")
                cancelled = self._cancelled()
            except asyncio.CancelledError:
                cancelled = True
            if cancelled:
                self._record(number, revision, attempt, action, "cancelled")
                return self._cancel()
            if any(
                isinstance(event, Steer) or event.restrictions is not None
                for event in self._steering.events[arrived:]
            ):
                aside = Material(SET_ASIDE, role.prompt.restrictions)
                digest = self._transcript.add_step(
                    revision.revision_id, number, reply.material, aside
                )
                self._record(number, revision, attempt, action, "set_aside", digest)
                return None
        self._record(number, revision, attempt, action, outcome)
        if isinstance(move, Stop):
            self._assessment = Assessment(
                revision.revision_id,
                "",
                self._package,
                self._package,
                (),
                dump_material(reply.material),
                classification="no_result",
                detail="The researcher stopped without a finding",
            )
            return self._finish(outcome, "", reply.material)
        return self._freeze(revision, reply, move, "")

    async def _close_out(self, reason: str) -> Ending | Finalization:
        """Ask once under the closing hold, then end or freeze a finalization as inconclusive.

        The budget ended the session, so any closing selection and later conclusion retain
        the exhausted limit and inconclusive outcome.
        """
        self._closing = True
        revision, role, hold = self._revisions[self._head], self._roles[self._head], self._hold
        number = len(self._steps) + 1
        request = self._transcript.prompt(self._head, role, closing=True)
        if request is None:
            return self._finish("inconclusive", "prompt")
        asked = await ask_model(
            revision,
            role.assignment,
            request,
            hold.attempt_id,
            within=TIMEOUT,
            stop=self._cancelled,
        )
        kind = self._settle(hold.attempt_id, asked)
        self._ledger.release(hold.attempt_id, _RELEASED[kind])
        self._hold = None
        if kind != "answered":
            self._record(number, revision, hold.attempt_id, "", kind)
            return self._cancel() if kind == "cancelled" else self._finish("inconclusive", reason)
        move = parse_move(asked.result.values)
        final = isinstance(move, Conclude | Stop)
        result = ("concluded" if isinstance(move, Conclude) else "stopped") if final else "refused"
        action = _label(asked.result.values["action"], role)
        self._record(number, revision, hold.attempt_id, action, result)
        if isinstance(move, Conclude):
            return self._freeze(revision, asked.result, move, reason)
        if isinstance(move, Stop):
            self._assessment = Assessment(
                revision.revision_id,
                reason,
                self._package,
                self._package,
                (),
                dump_material(asked.result.material),
                detail=f"The campaign exhausted {reason} and stopped without a finding",
            )
        return self._finish("inconclusive", reason, asked.result.material if final else None)

    def _freeze(
        self, revision: CampaignRevision, reply: Answer, move: Conclude, reason: str
    ) -> Ending | Finalization:
        self._closing = True
        selection = dump_material(reply.material)
        self._assessment = Assessment(
            revision.revision_id,
            reason,
            self._package,
            self._package,
            (),
            selection,
            proposer=json.dumps(asdict(reply.record.model), default=json_default, sort_keys=True),
        )
        try:
            value = parse_fields(
                json.loads(move.finding, object_pairs_hook=unique_keys), "trials", "shortlist"
            )
            trials = parse_array(value["trials"], "trials")
            known = {step.trial for step in self._steps if step.trial is not None}
            if any(type(trial) is not str or trial not in known for trial in trials) or len(
                set(trials)
            ) != len(trials):
                raise ValueError("The shortlist needs unique known trial IDs")
        except (ValueError, RecursionError):
            self._assessment = replace(self._assessment, detail="Invalid frozen shortlist")
            return self._finish("inconclusive", reason or "shortlist", reply.material)
        if self._hold is not None:
            self._ledger.release(self._hold.attempt_id, "inconclusive")
            self._hold = None
        pairs = tuple((trial, str(workflow.uuid4())) for trial in trials)
        self._assessment = replace(
            self._assessment,
            trials=tuple(HeldBack(source, trial, "pending", None) for source, trial in pairs),
        )
        return Finalization(revision.revision_id, reply.record.model, reply.material, pairs, reason)

    async def _finalize(self, closing: Finalization) -> Ending:
        await workflow.wait_condition(lambda: self._steering.status != "paused")
        if self._cancelled():
            return self._cancel()
        revision, role = self._revisions[closing.revision], self._roles[closing.revision]
        self._assessment = replace(
            self._assessment,
            recovery=reference.assessment_reference(
                revision.campaign_id,
                revision.specification.evaluation.digest,
                self._package or None,
                tuple((UUID(source), UUID(trial)) for source, trial in closing.trials),
                adjustment_trials=revision.specification.limits.max_trials,
            ),
        )
        started = workflow.now()
        running = workflow.start_activity_method(
            Authorized.evaluate,
            _Evaluating(
                file_bytes(revision).decode(),
                self._package,
                closing.trials,
                revision.specification.limits.max_trials,
            ),
            start_to_close_timeout=EXPERIMENT_TIMEOUT * max(1, len(closing.trials)),
            heartbeat_timeout=HEARTBEAT,
            retry_policy=RetryPolicy(maximum_attempts=1),
            cancellation_type=ActivityCancellationType.TRY_CANCEL,
        )
        try:
            await workflow.wait_condition(lambda: running.done() or self._cancelled())
            cancelled = self._cancelled()
        except asyncio.CancelledError:
            cancelled = True
        if not running.done():
            running.cancel()
        try:
            evaluated = await running
        except ActivityError:
            evaluated = None
        failed = evaluated is None
        self._ledger.charge(
            workflow.uuid4(),
            self._final_holds[0].attempt_id,
            "evaluation",
            Usage(compute_seconds=_seconds(workflow.now() - started)),
        )
        if failed and not cancelled and not self._cancelled():
            recovering = workflow.start_activity_method(
                Authorized.recover,
                args=[file_bytes(revision).decode(), self._assessment.recovery],
                start_to_close_timeout=FORFEIT_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=1),
                cancellation_type=ActivityCancellationType.TRY_CANCEL,
            )
            try:
                await workflow.wait_condition(lambda: recovering.done() or self._cancelled())
                cancelled = self._cancelled()
            except asyncio.CancelledError:
                cancelled = True
            if not recovering.done():
                recovering.cancel()
            try:
                evaluated = await recovering
            except ActivityError:
                evaluated = None
        if evaluated is not None:
            evidence = load_material(evaluated.material)
            completed = {
                item["trial"]: HeldBack(
                    item["source"],
                    item["trial"],
                    item["status"],
                    None
                    if item["evaluation"] is None
                    else json.dumps(item["evaluation"], sort_keys=True),
                )
                for item in json.loads(evidence.text)
            }
            self._package = evaluated.package
            self._assessment = replace(
                self._assessment,
                package=self._package,
                trials=tuple(completed.get(item.trial, item) for item in self._assessment.trials),
                evidence=evaluated.material,
            )
        if cancelled:
            return self._cancel()
        if failed:
            self._assessment = replace(self._assessment, detail="Held-back evaluation failed")
            return self._finish("inconclusive", closing.reason or "evaluation", closing.selection)
        shared = (
            Material(
                f"Question: {revision.specification.question}\n{SCOPE}", role.prompt.restrictions
            ),
            closing.selection,
            evidence,
        )
        conclusion = await self._interpret(
            revision,
            RoleAssignment(ROLE, closing.proposer),
            (role.prompt, *shared, Material(CONCLUSION, role.prompt.restrictions)),
            "conclusion",
        )
        if conclusion == "cancelled":
            return self._cancel()
        if not isinstance(conclusion, Material):
            return self._finish("inconclusive", closing.reason or conclusion, closing.selection)
        if self._cancelled():
            return self._cancel()
        eligible = []
        for model, prompt, floor in role.critics:
            restrictions = derive(
                prompt.restrictions, conclusion.restrictions, *(m.restrictions for m in shared)
            )
            if (
                model.provider in restrictions.approved_for
                and restrictions.labels <= floor.labels
                and restrictions.available_at <= floor.available_at
            ):
                eligible.append((model, prompt))
        critic, prompt = sorted(
            eligible, key=lambda item: item[0].provider == closing.proposer.provider
        )[0]
        self._assessment = replace(
            self._assessment,
            critic=json.dumps(asdict(critic), default=json_default, sort_keys=True),
            independent=critic.provider != closing.proposer.provider,
        )
        critique = await self._interpret(
            revision,
            RoleAssignment("critic", critic),
            (prompt, *shared, conclusion, Material(CRITIQUE, prompt.restrictions)),
            "critique",
        )
        if critique == "cancelled" or self._cancelled():
            return self._cancel()
        if not isinstance(critique, Material):
            return self._finish("inconclusive", closing.reason or critique, conclusion)
        classification, detail = _classify(self._assessment)
        if closing.reason:
            classification, detail = "inconclusive", f"The campaign exhausted {closing.reason}"
        self._assessment = replace(self._assessment, classification=classification, detail=detail)
        outcome = "inconclusive" if classification == "inconclusive" else "concluded"
        return self._finish(
            outcome,
            closing.reason or ("interpretation" if outcome == "inconclusive" else ""),
            conclusion,
        )

    async def _interpret(
        self,
        revision: CampaignRevision,
        assignment: RoleAssignment,
        materials: tuple[Material, ...],
        stage: str,
    ) -> Material | str:
        await workflow.wait_condition(lambda: self._steering.status != "paused")
        if self._cancelled():
            return "cancelled"
        request = Request(materials)
        if len(request.to_bytes()) > BOUND:
            self._assessment = replace(
                self._assessment, detail=f"Mandatory {stage} materials exceed the prompt bound"
            )
            return "prompt"
        hold = self._final_holds.pop(0)
        asked = await ask_model(
            revision, assignment, request, hold.attempt_id, within=TIMEOUT, stop=self._cancelled
        )
        kind = self._settle(hold.attempt_id, asked, assignment.role)
        self._ledger.release(hold.attempt_id, _RELEASED[kind])
        updates = {f"{stage}_calls": tuple(str(receipt.call_id) for receipt in asked.receipts)}
        if isinstance(asked.result, Answer):
            updates[stage] = dump_material(asked.result.material)
        else:
            updates["detail"] = f"The fresh {stage} call ended {kind}"
        self._assessment = replace(self._assessment, **updates)
        return asked.result.material if isinstance(asked.result, Answer) else kind

    def _settle(self, attempt: UUID, asked: Asked, role: str = ROLE) -> Asking:
        """Charge an ask's wall time and every receipt once, and say how it ended."""
        self._ledger.charge(
            workflow.uuid4(), attempt, role, Usage(wall_seconds=_seconds(asked.elapsed))
        )
        for receipt in asked.receipts:
            self._charge(attempt, receipt)
        self._settled.add(attempt)
        outcomes = {receipt.outcome for receipt in asked.receipts}
        if asked.result == "cancelled":
            return "cancelled"
        if "rate_limited" in outcomes and "completed" not in outcomes:
            return "rate_limited"
        return "answered" if isinstance(asked.result, Answer) else "failed"

    def _charge(self, attempt: UUID, receipt: Receipt) -> None:
        """Charge one receipt under its call ID, as execution.CampaignRun._charge does.

        Temporal cannot refuse an activity result, so a cost the exact sum cannot hold is
        charged as unreported instead of failing every workflow task.
        """
        try:
            self._ledger.charge(receipt.call_id, attempt, receipt.role, receipt.usage)
        except InexactUsage:
            unreported = receipt.usage.unreported | {"cost_usd"}
            usage = replace(receipt.usage, cost_usd=Decimal(0), unreported=unreported)
            self._ledger.charge(receipt.call_id, attempt, receipt.role, usage)

    def _finish(self, outcome: Outcome, reason: str, final: Material | None = None) -> Ending:
        if self._hold is not None:
            self._ledger.release(self._hold.attempt_id, "inconclusive")
            self._hold = None
        for hold in self._final_holds:
            self._ledger.release(hold.attempt_id, "inconclusive")
        self._final_holds.clear()
        self._ending = Ending(
            outcome, reason, len(self._steps), None if final is None else dump_material(final)
        )
        return self._ending

    def _cancel(self) -> Ending:
        """End the loop cancelled, recording the runner's cancel when Temporal's came first."""
        if self._cancelled():
            return self._finish("cancelled", "researcher")
        self._steering = self._steering.record(
            "cancel",
            actor=ACTOR,
            at=self._at(),
            reason=f"The workflow was cancelled after step {len(self._steps)}",
        )
        return self._finish("cancelled", "workflow")

    def _checkpoint(self, role: Role) -> bool:
        """Pause as the runner when an every checkpoint falls due; return whether it did."""
        done = len(self._steps)
        due = [item.steps for item in role.checkpoints if isinstance(item, Every)]
        due = [steps for steps in due if done % steps == 0]
        if not due or done in (0, self._gated):
            return False
        self._gated = done
        self._pause(f"Checkpoint every {due[0]} steps after step {done}")
        return True

    def _capped(self) -> str | None:
        if len(self._steps) >= STEP_CAP:
            return "steps"
        if workflow.info().get_current_history_size() >= HISTORY_BYTES:
            return "history"
        return None

    def _pause(self, reason: str) -> None:
        """Record a runner pause, whose reason names its step so no two collapse into one."""
        if not self._cancelled():
            self._steering = self._steering.record(
                "pause", actor=ACTOR, at=self._at(), reason=reason
            )

    def _record(
        self,
        number: int,
        revision: CampaignRevision,
        attempt: UUID,
        action: str,
        result: Result,
        digest: str = "",
        trial: str | None = None,
    ) -> None:
        self._steps.append(
            StepRecord(
                number,
                format_time(workflow.now()),
                revision.revision_id,
                str(attempt),
                action,
                result,
                digest,
                trial,
            )
        )

    def _at(self) -> datetime:
        """Return the later of now and the last event, so no event is dated before another."""
        events = self._steering.events
        return max(workflow.now(), events[-1].at) if events else workflow.now()

    def _cancelled(self) -> bool:
        return self._steering.status == "cancelled"

    def _state(self) -> State:
        return self._steering.status if self._ending is None else self._ending.outcome

    async def _admit(self, revision: CampaignRevision) -> Role | str:
        """Run the revision's opening and decode it, or return why the loop cannot take it.

        A researcher's cancel or the loop's end abandons the opening, so one that hangs holds
        neither the loop nor a steer.
        """
        opening = workflow.start_activity_method(
            Authorized.opening,
            file_bytes(revision).decode(),
            start_to_close_timeout=OPENING_TIMEOUT,
        )
        try:
            await workflow.wait_condition(
                lambda: (
                    opening.done()
                    or self._cancelled()
                    or self._ending is not None
                    or self._failure is not None
                )
            )
        finally:
            if not opening.done():
                opening.cancel()
        try:
            opened = await opening
        except ActivityError as error:
            cause = error.cause
            return _UNADMITTED.get(
                cause.type if isinstance(cause, ApplicationError) else "", "role"
            )
        return _role(opened, revision)

    def _absorb(self, after: int) -> None:
        """Add the declared text of the events from index after to the head's transcript."""
        for event, declared in self._steering.declared(after):
            self._transcript.add_note(self._head, event, declared)

    @workflow.update
    def control(self, action: str, actor: str, reason: str, declared: str | None) -> State:
        after = len(self._steering.events)
        self._steering = self._controlled(action, actor, reason, declared)
        self._absorb(after)
        return self._state()

    @control.validator
    def _check_control(self, action: str, actor: str, reason: str, declared: str | None) -> None:
        self._controlled(action, actor, reason, declared)

    def _controlled(
        self, action: str, actor: str, reason: str, declared: str | None
    ) -> SteeringLog:
        """Return the log with a researcher's command, mutating nothing.

        The validator runs it first, so the handler, which runs it again on the same state,
        cannot raise.
        """
        if actor == ACTOR:
            raise ResearchError(f"The actor {ACTOR!r} is reserved for the research loop")
        self._admissible(actor, reason, declared)
        restrictions = None
        if declared is not None:
            role = self._roles.get(self._head)
            if role is None:
                raise ResearchError("Declared text waits until the loop opens its revision")
            restrictions = load_declaration(declared)
            check_declaration(restrictions, role.floor)
        return self._steering.record(
            action, actor=actor, at=self._at(), reason=reason, restrictions=restrictions
        )

    @workflow.update
    async def steer(self, revision: str, declared: str | None) -> State:
        """Adopt a child revision for every later ask, after its opening returns."""
        role = await self._admit(parse_file(revision.encode()))
        try:
            if isinstance(role, str):
                raise ResearchError(f"The child revision cannot be opened: {role}")
            child, steering = self._steered(revision, declared, role)
        except ValueError as error:
            # Raised after validation, any other exception would fail the workflow task.
            raise ApplicationError(str(error), non_retryable=True) from None
        after = len(self._steering.events)
        self._steering = steering
        self._revisions[child.revision_id] = child
        self._roles[child.revision_id] = role
        self._head = child.revision_id
        self._absorb(after)
        return self._state()

    @steer.validator
    def _check_steer(self, revision: str, declared: str | None) -> None:
        check_revision_size(revision.encode())
        self._steered(revision, declared, None)

    def _steered(
        self, text: str, declared: str | None, role: Role | None
    ) -> tuple[CampaignRevision, SteeringLog]:
        """Return the child and the log with its steer, mutating nothing.

        The validator runs it before the child's opening, and the handler again with the child's
        role, whose chain and floor are known only then.
        """
        child = parse_file(text.encode())
        parent = self._revisions.get(child.parent)
        if parent is None:
            raise ResearchError(f"Revision {child.revision_id} revises a revision the loop lacks")
        if self._hold is None or self._closing:
            raise ResearchError("The loop takes a steer only between its first and closing asks")
        check_link(parent, child)
        if changed := sorted({"limits", "evaluation"} & set(child.changes)):
            raise ResearchError(f"A running campaign keeps its starting {changed}")
        checkpoints(child)
        self._admissible(child.author, child.reason, declared)
        restrictions = None if declared is None else load_declaration(declared)
        if role is not None and role.chain > self._chain:
            raise ResearchError(f"The child's chain of {role.chain} passes the {self._chain} held")
        if role is not None and restrictions is not None:
            check_declaration(restrictions, role.floor)
        steered = self._steering.steer(
            child, at=max(self._at(), child.created_at), restrictions=restrictions
        )
        return child, steered

    def _admissible(self, actor: str, reason: str, declared: str | None) -> None:
        if self._ending is not None or self._failure is not None:
            raise ResearchError("The research loop has ended")
        if max(len(text.encode()) for text in (actor, reason, declared or "")) > NOTE_BYTES:
            raise ResearchError(
                f"An actor, reason, or declaration holds at most {NOTE_BYTES} bytes"
            )
        if sum(isinstance(e, Steer) or e.actor != ACTOR for e in self._steering.events) >= COMMANDS:
            raise ResearchError(f"A campaign takes at most {COMMANDS} researcher commands")

    @workflow.query
    def status(self) -> Report:
        return self._report("", "")

    @workflow.query
    def observation(self) -> _Observation:
        if self._failure is not None:
            raise ResearchError("Unexpected research implementation failure")
        report = self._report("", "", include_spent=False)
        summary = replace(
            report,
            assessment=None
            if report.assessment is None
            else replace(
                report.assessment,
                trials=tuple(replace(trial, evaluation=None) for trial in report.assessment.trials),
            ),
            ending=None if report.ending is None else replace(report.ending, final=None),
            unrecorded=None
            if report.unrecorded is None
            else replace(report.unrecorded, program=""),
        )
        events = self._ledger.events
        return _Observation(summary, len(events), tuple(map(str, sorted(self._settled))))

    @workflow.query
    def ledger_part(self, event_count: int, start: int) -> bytes:
        if type(start) is not int or start < 0:
            raise ResearchError("A ledger offset must be a nonnegative integer")
        return dump_ledger(self._ledger, event_count=event_count)[start : start + PART]

    @workflow.query
    def part(self, start: int) -> bytes:
        """Return PART bytes of the closed loop's full Report from start; read only once closed."""
        if self._frozen:
            return self._frozen[start : start + PART]
        report = self._report(
            log_bytes(self._steering).decode(), dump_ledger(self._ledger).decode()
        )
        return DataConverter.default.payload_converter.to_payload(report).data[start : start + PART]

    @workflow.query
    def archive_part(self, start: int) -> bytes:
        return self._archive_view[start : start + PART]

    def _report(self, steering: str, ledger: str, *, include_spent: bool = True) -> Report:
        info = workflow.info()
        head = self._roles.get(self._head)
        pauses = [event for event in self._steering.events if event.action == "pause"]
        paused_by = (
            f"{pauses[-1].actor}: {pauses[-1].reason}" if self._state() == "paused" else None
        )
        return Report(
            info.workflow_id,
            info.run_id,
            self._state(),
            paused_by,
            self._head,
            "" if head is None else head.authority,
            STEP_CAP,
            tuple(self._steps),
            asdict(self._revisions[self._head].specification.limits),
            _spent(self._ledger) if include_spent else {},
            self._ending
            if ledger or self._assessment is None or self._ending is None
            else replace(self._ending, final=None),
            self._package,
            self._unrecorded,
            steering,
            ledger,
            self._assessment
            if ledger or self._assessment is None
            else replace(
                self._assessment, shortlist="", evidence=None, conclusion=None, critique=None
            ),
        )


def _spent(ledger: Ledger) -> dict[str, str]:
    usage = sum((line.usage for line in ledger.report()), Usage())
    return {name: str(value) for name, value in asdict(usage).items() if name != "unreported"} | {
        "unreported": " ".join(sorted(usage.unreported))
    }


def _classify(assessment: Assessment) -> tuple[str, str]:
    verdicts = {}
    for trial in assessment.trials:
        result = (
            None
            if trial.evaluation is None
            else reference.decode_evaluation(
                reference.canonical_bytes(json.loads(trial.evaluation))
            )
        )
        verdicts[trial.trial] = (
            result.verdict if isinstance(result, evaluation.Evaluation) else None
        )
    try:
        conclusion = parse_fields(
            json.loads(load_material(assessment.conclusion).text, object_pairs_hook=unique_keys),
            "classification citations explanation",
            "conclusion",
        )
        critique = parse_fields(
            json.loads(load_material(assessment.critique).text, object_pairs_hook=unique_keys),
            "decision citations explanation",
            "critique",
        )
        classification = parse_text(conclusion["classification"], "classification")
        decision = parse_text(critique["decision"], "decision")
        if classification not in {
            "supported",
            "null",
            "inconclusive",
            "no_result",
        } or decision not in {"accept", "reject"}:
            raise ValueError("Unknown interpretation")
        cited = []
        for value in (conclusion, critique):
            parse_text(value["explanation"], "explanation")
            citations = parse_array(value["citations"], "citations")
            if any(type(item) is not str or item not in verdicts for item in citations) or len(
                set(citations)
            ) != len(citations):
                raise ValueError("Unknown or repeated citation")
            cited.append(citations)
    except (ValueError, RecursionError):
        return "inconclusive", "Malformed interpretation or invalid held-back citations"
    if decision != "accept":
        return "inconclusive", "The critic rejected the conclusion"
    if classification == "supported":
        if any(verdicts[trial] == "supported" for trial in cited[0]):
            return "supported", "The accepted conclusion cites supported held-back evidence"
        return "inconclusive", "The positive conclusion has no cited supported held-back evidence"
    if classification == "null":
        if (
            verdicts
            and set(cited[0]) == set(verdicts)
            and all(verdict == "null" for verdict in verdicts.values())
        ):
            return "null", "Every shortlisted held-back result is null and both models agree"
        return "inconclusive", "The shortlist does not establish a complete held-back null"
    if not verdicts:
        return "no_result", "The frozen shortlist is empty"
    return "inconclusive", "The conclusion makes no supported or complete null claim"


def _label(action: str, role: Role) -> str:
    return action if action in role.tools or action in _WORDS else "unknown"


def _budget(limits: ResourceLimits) -> Budget:
    # The mapping CampaignRun.__init__ makes inline, in a workflow this module must not import.
    return Budget(
        compute_seconds=Decimal(limits.max_compute_seconds),
        wall_seconds=None if limits.max_wall_seconds is None else Decimal(limits.max_wall_seconds),
        model_calls=limits.max_model_calls,
        input_tokens=limits.max_input_tokens,
        output_tokens=limits.max_output_tokens,
        failures_per_run=limits.max_attempts,
    )


def _asking(chain: int) -> Usage:
    return Usage(wall_seconds=_seconds(TIMEOUT), model_calls=chain)


def _seconds(delta: timedelta) -> Decimal:
    return Decimal(delta // timedelta(microseconds=1)).scaleb(-6)


async def start(
    client: Client, revision: CampaignRevision, task_queue: str
) -> WorkflowHandle[ResearchLoop, Ending]:
    """Start the campaign's research loop, or return it while it runs the same revision.

    The workflow ID is the campaign ID, as execution.start uses, so one campaign has one runner
    of either kind. The memo pins the revision and its frozen evaluation rules.

    Args:
        client (Client): Connected Temporal client.
        revision (CampaignRevision): Revision the first asks run under.
        task_queue (str): Queue that a research worker polls.

    Returns:
        (WorkflowHandle) The campaign's loop.

    Raises:
        ResearchError: An intervention is not a checkpoint, the revision has no active researcher
            role, max_model_calls is undeclared or below two chains plus two calls,
            max_wall_seconds is below four times TIMEOUT, or the campaign has already run or
            runs another revision, other
            rules, or another runner.
        CampaignError: The revision does not parse.
    """
    content = file_bytes(revision)
    parse_file(content)
    checkpoints(revision)
    roles = [
        role
        for role in revision.specification.model_roles
        if (role.role, role.status) == (ROLE, "active")
    ]
    if not roles:
        raise ResearchError(f"The revision has no active {ROLE} role")
    chain = 1 + len(roles[0].fallbacks)
    limits = revision.specification.limits
    if limits.max_model_calls is None or limits.max_model_calls < 2 * chain + 2:
        raise ResearchError(f"The loop needs max_model_calls of at least {2 * chain + 2}")
    wall = 4 * TIMEOUT // timedelta(seconds=1)
    if limits.max_wall_seconds is not None and limits.max_wall_seconds < wall:
        raise ResearchError(f"The loop needs max_wall_seconds of at least {wall}")
    memo = {"revision": revision.revision_id, "rules": revision.specification.evaluation.digest}
    try:
        handle = await client.start_workflow(
            ResearchLoop.run,
            content.decode(),
            id=str(revision.campaign_id),
            task_queue=task_queue,
            id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
            id_reuse_policy=WorkflowIDReusePolicy.REJECT_DUPLICATE,
            memo=memo,
        )
    except WorkflowAlreadyStartedError as error:
        raise ResearchError(f"Campaign {revision.campaign_id} has already run") from error
    # USE_EXISTING returns the running workflow without delivering this revision.
    description = await handle.describe()
    if {key: await description.memo_value(key, None) for key in memo} != memo:
        raise ResearchError(
            f"Campaign {revision.campaign_id} runs another revision, other rules, or another runner"
        )
    return handle


async def control(
    handle: WorkflowHandle[ResearchLoop, Ending],
    action: str,
    *,
    actor: str,
    reason: str,
    restrictions: Restrictions | None = None,
) -> State:
    """Record a pause, resume, cancel, or annotation in the loop's steering log.

    The workflow dates the event. Only a declared reason can reach a later request.

    Args:
        handle (WorkflowHandle): Campaign loop.
        action (str): pause, resume, cancel, or annotate.
        actor (str): Person who issues the command, never ACTOR.
        reason (str): Why the command is issued, or the note itself for an annotation.
        restrictions (Restrictions | None, optional): Declared restrictions of reason, which
            must fit the head revision's floor.

    Returns:
        (State) The loop's state after the command.
    """
    declared = None if restrictions is None else dump_declaration(restrictions)
    return await handle.execute_update(ResearchLoop.control, args=[action, actor, reason, declared])


async def steer(
    handle: WorkflowHandle[ResearchLoop, Ending],
    revision: CampaignRevision,
    *,
    restrictions: Restrictions | None = None,
) -> State:
    """Steer the loop to a child revision, under which every later ask runs.

    Args:
        handle (WorkflowHandle): Campaign loop.
        revision (CampaignRevision): Child of a revision the loop holds that keeps its limits
            and evaluation rules and whose chain is no longer than the first revision's.
        restrictions (Restrictions | None, optional): Declared restrictions of the child's
            reason, which must fit the child's floor.

    Returns:
        (State) The loop's state after the steer.
    """
    content = file_bytes(revision)
    check_revision_size(content)
    declared = None if restrictions is None else dump_declaration(restrictions)
    return await handle.execute_update(ResearchLoop.steer, args=[content.decode(), declared])


async def status(handle: WorkflowHandle[ResearchLoop, Ending]) -> Report:
    """Return the loop's Report without its logs, while it runs or after it closes."""
    return await handle.query(ResearchLoop.status)


async def observe(
    handle: WorkflowHandle[ResearchLoop, Ending],
    *,
    client: Client,
    description: WorkflowExecutionDescription | None = None,
) -> Observation:
    """Read one run's bounded summary and ledger prefix, reusing description when supplied."""
    if description is None:
        description = await handle.describe()
    pinned = client.get_workflow_handle_for(ResearchLoop.run, handle.id, run_id=description.run_id)
    if description.status != WorkflowExecutionStatus.RUNNING:
        await pinned.result()
    head = await pinned.query(ResearchLoop.observation)
    parts = []
    while True:
        part = await pinned.query(
            ResearchLoop.ledger_part,
            args=[head.event_count, len(parts) * PART],
        )
        parts.append(part)
        if len(part) < PART:
            break
    ledger = load_ledger(b"".join(parts))
    report = replace(head.report, spent=_spent(ledger))
    return Observation(report, ledger, frozenset(map(UUID, head.settled)))


async def record(handle: WorkflowHandle[ResearchLoop, Ending]) -> Report:
    """Wait for the loop to close and read its Report with both logs, PART bytes per query.

    Args:
        handle (WorkflowHandle): Campaign loop.

    Returns:
        (Report) The closed loop's report with both logs, which researcher commands can grow
            past one Temporal payload.
    """
    await handle.result()
    parts = [await handle.query(ResearchLoop.part, 0)]
    while len(parts[-1]) == PART:
        parts.append(await handle.query(ResearchLoop.part, PART * len(parts)))
    payload = Payload(metadata={"encoding": b"json/plain"}, data=b"".join(parts))
    return DataConverter.default.payload_converter.from_payload(payload, Report)


def worker(
    client: Client,
    task_queue: str,
    gateway: Gateway,
    authorize: Callable[[CampaignRevision], Authority],
    *,
    archive: CampaignArchive,
) -> Worker:
    """Return a worker for the research loop, its model, tool, and close activities, and its own.

    The queue is the loop's own, because execution.worker serves CampaignRun and both register
    an activity named close. A resume or a close may run on any worker of the queue, so every
    one must share the archive root, call-log directory, and numerical publication storage.
    Bind Gateway(CallLog(calls), ...) and CampaignArchive(root, calls) to the same calls path.

    Args:
        client (Client): Connected Temporal client.
        task_queue (str): Queue to poll.
        gateway (Gateway): The gateway every model call of this worker uses.
        authorize (Callable[[CampaignRevision], Authority]): A revision's run authority,
            functools.partial(campaigns.resolve, setup=setup) in production.
        archive (CampaignArchive): Required archive using the gateway's call-log directory.

    Returns:
        (Worker) The unstarted worker.
    """
    return Worker(
        client,
        task_queue=task_queue,
        workflows=[ResearchLoop],
        activities=[
            *Activities(gateway, authorize).definitions,
            *Authorized(authorize, archive).definitions,
        ],
    )


def run_worker(
    address: str,
    task_queue: str,
    gateway: Gateway,
    authorize: Callable[[CampaignRevision], Authority],
    *,
    archive: CampaignArchive,
) -> None:
    async def serve() -> None:
        try:
            client = await Client.connect(address)
        except RuntimeError as error:
            raise ConnectionError(f"Cannot connect to Temporal at {address}: {error}") from None
        await worker(client, task_queue, gateway, authorize, archive=archive).run()

    asyncio.run(serve())
