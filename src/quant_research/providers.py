from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Iterable, Mapping
from dataclasses import InitVar, asdict, dataclass, replace
from decimal import Decimal, localcontext
from enum import Enum
from pathlib import Path
from typing import Literal, Protocol, get_args
from uuid import UUID, uuid4

from quant_research import access
from quant_research.access import (
    Provider,
    RestrictionLabel,
    Restrictions,
    RunContext,
    derive,
    frozen,
)
from quant_research.accounting import EXACT, REPORTABLE, Usage, json_default, parse_usage
from quant_research.model_provenance import ModelRecord, RoleAssignment, parse_model_record
from quant_research.model_provenance import json_default as _model_json_default
from quant_research.records import (
    ObjectConflict,
    append_log,
    canonical_bytes,
    format_time,
    keep_object,
    parse_array,
    parse_digest,
    parse_fields,
    parse_text,
    parse_time,
    parse_uuid,
    read_log,
    unique_keys,
)

type Value = str | int | float | bool
type Schema = tuple[tuple[str, type], ...]
type Reason = access.Reason | Literal["outside_provider_policy"]
type FailureReason = Literal[
    "api_error",
    "api_key_source",
    "cancelled",
    "config",
    "deadline",
    "exit",
    "features",
    "gate_turn",
    "hooks",
    "http_connection_failed",
    "interrupted",
    "inventory",
    "local_turn",
    "mcp_servers",
    "model",
    "model_consent_fallback",
    "model_fallback",
    "model_refusal_fallback",
    "no_answer",
    "output_style",
    "plugins",
    "protocol",
    "provider_error",
    "rate_limit",
    "rerouted",
    "server_request",
    "signed_out",
    "skills",
    "spawn",
    "thread",
    "tools",
    "usage_limit",
    "version",
]
type Sent = Literal[
    "completed", "provider_error", "malformed_output", "cancelled", "rate_limited", "unknown"
]
type Outcome = Sent | Literal["refused"]

_VALUE_TYPES = get_args(Value.__value__)
_SENT = get_args(Sent.__value__)
_OUTCOMES = (*_SENT, "refused")
_REASONS = (*get_args(access.Reason.__value__), "outside_provider_policy")
_FAILURE_REASONS = get_args(FailureReason.__value__)

# Only Gateway passes this, so an adapter receives content only after its policy check.
_ISSUER = object()


class ProviderError(ValueError):
    """Report a misconfigured gateway, an invalid request or lease, or a bad log."""


@dataclass(frozen=True, kw_only=True)
class ProviderPolicy:
    """One provider's terms for outbound content, from operator configuration.

    labels is the ceiling of restriction labels the provider may receive. The material's own
    approval for the provider and the run's clearances still apply. retains is recorded through
    version; no rule reads it yet.
    """

    provider: Provider
    labels: frozenset[RestrictionLabel]
    retains: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", frozen(self.labels, "labels"))

    @property
    def version(self) -> str:
        """Return the SHA-256 digest of the canonical policy terms."""
        terms = {"provider": self.provider, "labels": sorted(self.labels), "retains": self.retains}
        return hashlib.sha256(canonical_bytes(terms)).hexdigest()

    def denials(self, context: RunContext, item: Restrictions) -> tuple[Reason, ...]:
        """Return the run's send denials, then outside_provider_policy if a label exceeds it."""
        denials = context.send_denials(item, self.provider)
        return denials if item.labels <= self.labels else (*denials, "outside_provider_policy")


@dataclass(frozen=True)
class Material:
    """Text with the restrictions that ingestion or derivation gave it, never read from the text."""

    text: str
    restrictions: Restrictions


@dataclass(frozen=True)
class Request:
    """Materials to send in order, and the schema of a structured reply or None for free text.

    to_bytes is the request's one canonical form: the bytes its digest hashes, the sent
    transcript the call log keeps, and the text a model activity's input carries.
    """

    materials: tuple[Material, ...]
    schema: Schema | None = None

    def __post_init__(self) -> None:
        if not self.materials:
            raise ProviderError("A request needs at least one material")
        if self.schema is None:
            return
        names = [name for name, _ in self.schema]
        if (
            not names
            or not all(names)
            or len(set(names)) != len(names)
            or any(kind not in _VALUE_TYPES for _, kind in self.schema)
        ):
            raise ProviderError(
                "A schema needs unique nonempty names typed str, int, float, or bool"
            )

    @property
    def restrictions(self) -> Restrictions:
        """Return the restrictions derived over every material."""
        return derive(*(material.restrictions for material in self.materials))

    @property
    def digest(self) -> str:
        """Return the SHA-256 digest of to_bytes, which every claim and record of it names."""
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_bytes(self) -> bytes:
        """Return the texts, the schema, and the derived restrictions as ASCII JSON, keys sorted."""
        schema = None if self.schema is None else [[n, kind.__name__] for n, kind in self.schema]
        terms = {
            "texts": [material.text for material in self.materials],
            "schema": schema,
            "restrictions": _restrictions(self.restrictions),
        }
        # ASCII escapes let any text, lone surrogates included, enter the digest. The restrictions
        # enter it too, since a reused reply takes the restrictions of the request it answers.
        return json.dumps(terms, sort_keys=True).encode()

    @classmethod
    def from_bytes(cls, content: bytes) -> Request:
        """Return the request whose to_bytes is content, raising ProviderError for other bytes.

        Each material carries the request's derived restrictions, which is all the gateway reads,
        so the result has the policy denials and the digest of the request that wrote content.

        Args:
            content (bytes): Bytes that to_bytes wrote.

        Returns:
            (Request) The decoded request.
        """
        kinds = {kind.__name__: kind for kind in _VALUE_TYPES}
        try:
            terms = parse_fields(
                json.loads(content.decode("ascii"), object_pairs_hook=unique_keys),
                "texts schema restrictions",
                "request",
            )
            schema = terms["schema"]
            if schema is not None:
                fields = [_texts(field, "schema field") for field in parse_array(schema, "schema")]
                schema = tuple((name, kinds.get(kind)) for name, kind in fields)
            item = _parse_restrictions(terms["restrictions"])
            request = cls(
                tuple(Material(text, item) for text in _texts(terms["texts"], "texts")), schema
            )
        except (ValueError, RecursionError) as error:
            failure = f"Invalid request bytes: {error}"
        else:
            if request.to_bytes() == content:
                return request
            failure = "Request bytes are not the form that to_bytes writes"
        # Raising after the handler chains nothing, since decode errors keep the whole content.
        raise ProviderError(failure)


def _restrictions(item: Restrictions) -> dict[str, object]:
    return {
        "available_at": format_time(item.available_at),
        "labels": sorted(item.labels),
        "approved_for": sorted(item.approved_for),
    }


def _parse_restrictions(raw: object) -> Restrictions:
    value = parse_fields(raw, "available_at labels approved_for", "restrictions")
    return Restrictions(
        available_at=parse_time(value["available_at"]),
        labels=frozenset(_texts(value["labels"], "labels")),
        approved_for=frozenset(_texts(value["approved_for"], "approved_for")),
    )


def _texts(value: object, label: str) -> list[str]:
    values = parse_array(value, label)
    if not all(type(item) is str for item in values):
        raise ProviderError(f"{label} must hold only text")
    return values


@dataclass(frozen=True)
class Cleared:
    """Content that passed one model's policy check. Only Gateway creates one."""

    model: ModelRecord
    texts: tuple[str, ...]
    schema: Schema | None
    issuer: InitVar[object] = None

    def __post_init__(self, issuer: object) -> None:
        if issuer is not _ISSUER:
            raise ProviderError("Only Gateway clears content for an adapter")


@dataclass(frozen=True)
class Replied:
    text: str
    usage: Usage
    resolved_model: str | None = None


@dataclass(frozen=True)
class Failed:
    usage: Usage
    resolved_model: str | None = None
    reason: FailureReason | None = None


@dataclass(frozen=True)
class Cancelled:
    usage: Usage
    resolved_model: str | None = None
    reason: FailureReason | None = None


@dataclass(frozen=True)
class RateLimited:
    """A provider's refusal of a call for a usage cap."""

    usage: Usage
    resolved_model: str | None = None
    reason: FailureReason | None = None


type Reply = Replied | Failed | Cancelled | RateLimited

_UNREPLIED = {Failed: "provider_error", Cancelled: "cancelled", RateLimited: "rate_limited"}


class Adapter(Protocol):
    subscription: bool
    """True when calls draw on a flat subscription, so the gateway records no cash cost."""

    runtime_version: str
    """The pinned runtime the adapter accepts, such as "claude 2.1.301", stored on each claim."""

    def complete(self, request: Cleared, cancel: threading.Event) -> Reply:
        """Send cleared content to one model and report the usage it incurred.

        Report provider trouble as Failed and a usage cap as RateLimited rather than raising,
        mark each field the runtime does not fully report as unreported, report the model the
        provider says it served as nonblank resolved_model text when it names one, and check
        cancel.is_set() at each step. An unsuccessful reply's reason must be a FailureReason
        or None.
        """


@dataclass(frozen=True)
class Lease:
    """Call IDs that the workflow minted and reserved for one ledger attempt.

    dispatches holds one call ID per model the attempt may try, in chain order, fallbacks
    included, and the gateway sends model i only under dispatches[i]. fence is the number of the
    activity that holds the lease, so a holder that meets a later activity's claim stops.
    """

    attempt_id: UUID
    dispatches: tuple[UUID, ...]
    fence: int


@dataclass(frozen=True)
class Dispatched:
    """The claim on a call ID, stored before its adapter runs.

    The log therefore names every model sent content. The adapter runs only for the claim the
    log stored first, and token is minted for each claim, so no two claims hold the same bytes.
    A claim without a record is a call whose outcome and usage are unknown until a seal records
    them as such. subscription and runtime_version are the billing and the runtime of the
    adapter that sends it, so any worker that seals the claim records the same cost and runtime.
    """

    call_id: UUID
    attempt_id: UUID
    fence: int
    role: str
    model: ModelRecord
    policy_version: str
    request_digest: str
    subscription: bool
    runtime_version: str
    token: UUID


@dataclass(frozen=True)
class Served:
    """What served one sent call.

    runtime_version is the runtime of the adapter that sent it. resolved_model is the model the
    provider reported serving, or None when it reported none or the call was sealed unanswered.
    """

    runtime_version: str
    resolved_model: str | None


@dataclass(frozen=True)
class CallRecord:
    """The final record of one call ID: refused before sending, sent, or sealed as unknown.

    A sent record repeats the fields of its claim and says what served it. reply_digest names
    the transcript of a completed or malformed reply, so a later holder of the lease reuses it
    instead of sending again. A refused record has neither.

    reason is a fixed diagnosis for a failed, cancelled, or rate-limited reply. It is None for
    older records, an absent or ambiguous diagnosis, and unknown seals. reasons independently
    names policy denials for a refused call.
    """

    call_id: UUID
    attempt_id: UUID
    fence: int
    role: str
    model: ModelRecord
    policy_version: str
    request_digest: str
    outcome: Outcome
    reasons: tuple[Reason, ...]
    usage: Usage
    reply_digest: str | None
    served: Served | None
    reason: FailureReason | None = None

    def __post_init__(self) -> None:
        if self.reason is not None:
            if type(self.reason) is not str or self.reason not in _FAILURE_REASONS:
                raise ProviderError("Unsupported failure reason")
            if self.outcome not in _UNREPLIED.values():
                raise ProviderError("Only failed, cancelled, or rate-limited calls have a reason")
        if (self.outcome == "refused") != bool(self.reasons):
            raise ProviderError("A call record has reasons exactly when it was refused")
        if self.outcome == "refused" and self.usage != Usage():
            raise ProviderError("A refused call sent nothing, so it has no usage")
        if (self.outcome == "refused") != (self.served is None):
            raise ProviderError("A call record says what served it exactly when it was sent")
        if (self.outcome in ("completed", "malformed_output")) != (self.reply_digest is not None):
            raise ProviderError("A call record names a reply exactly when it holds one")


@dataclass(frozen=True)
class Closed:
    """The end of an attempt's lease: no later claim may name the attempt."""

    attempt_id: UUID


type Entry = Dispatched | CallRecord | Closed


@dataclass(frozen=True)
class Receipt:
    """One sent call of a closed lease, as the workflow charges it."""

    call_id: UUID
    role: str
    outcome: Sent
    usage: Usage


@dataclass(frozen=True)
class Answer:
    """A completed reply, restricted as the request it answers, and the call that produced it."""

    material: Material
    values: Mapping[str, Value] | None
    record: CallRecord


@dataclass(frozen=True)
class Unanswered:
    """Every model tried, in order, when none completed."""

    records: tuple[CallRecord, ...]


class Gateway:
    """Send role requests only through provider policies, under call IDs the workflow minted.

    The gateway is the only writer of its call log and mints no call ID. A call ID moves from
    unclaimed to claimed to recorded, or from unclaimed to refused, and Closed ends its
    attempt's lease. The first record stored for an ID is final, so each ID is sent at most
    once and a stored reply is reused instead of sent again.
    """

    def __init__(self, log: CallLog, routes: Iterable[tuple[ProviderPolicy, Adapter]]) -> None:
        """Bind each provider's policy to its adapter.

        Args:
            log (CallLog): Store of every claim, call record, and close.
            routes (Iterable[tuple[ProviderPolicy, Adapter]]): One policy and adapter per provider.
        """
        self._log = log
        self._routes: dict[Provider, tuple[ProviderPolicy, Adapter]] = {}
        for policy, adapter in routes:
            if policy.provider in self._routes:
                raise ProviderError(f"Provider {policy.provider!r} is routed twice")
            self._routes[policy.provider] = (policy, adapter)

    @property
    def records(self) -> tuple[CallRecord, ...]:
        return tuple(entry for entry in self._log.load() if isinstance(entry, CallRecord))

    def call(
        self,
        context: RunContext,
        assignment: RoleAssignment,
        request: Request,
        *,
        lease: Lease,
        cancel: threading.Event,
    ) -> Answer | Unanswered:
        """Send a request to the role's primary model, then to each fallback until one completes.

        Args:
            context (RunContext): Run authority whose send denials every model must pass.
            assignment (RoleAssignment): Role, and its models in the order to try them.
            request (Request): Materials to send and the schema of a structured reply.
            lease (Lease): The attempt's call IDs, one per model, and the holding activity's fence.
            cancel (threading.Event): Event that stops the current call and every later fallback.

        Returns:
            (Answer | Unanswered) The first completed reply, or every record when none completed.

        Notes:
            A provider without a route or a chain longer than the lease raises ProviderError
            before any model runs, and a blank role raises PackageError. Model i takes the ID
            lease.dispatches[i], and the gateway decides from a fresh read of the log, again
            after each write:
            - A stored record is reused and nothing is sent. A record stored for another
              attempt, role, model, or request raises ProviderError.
            - A claim without a record under a later fence means a later activity holds the
              lease, and one under this fence means another call of this activity is sending
              it, so both raise ProviderError. One under an earlier fence belongs to an activity
              that will never use its reply, so it is sealed as unknown.
            - An unclaimed ID of a closed attempt raises ProviderError.
            - Otherwise the policy check runs. A denied model is recorded as refused, and an
              admitted one is sent only when the log stores this holder's claim first.
            A provider error, malformed output, or rate limit moves to the next model unless
            the event is set, and a cancellation ends the chain. The request's bytes are kept
            as its sent transcript before its claim, and a reply's before the record that names
            it. A completed reply, sent now or reused, is read back from its transcript, so a
            missing or damaged one raises ProviderError and is never sent again.
        """
        role = parse_text(assignment.role, "role")
        chain = (assignment.primary, *assignment.fallbacks)
        unrouted = sorted({model.provider for model in chain} - self._routes.keys())
        if unrouted:
            raise ProviderError(f"No policy and adapter for providers {unrouted}")
        if len(chain) > len(lease.dispatches):
            raise ProviderError(
                f"A chain of {len(chain)} models needs as many call IDs, "
                f"not the lease's {len(lease.dispatches)}"
            )
        item, sent, digest = request.restrictions, request.to_bytes(), request.digest
        texts = tuple(material.text for material in request.materials)
        records: list[CallRecord] = []
        for model, call_id in zip(chain, lease.dispatches, strict=False):
            policy, _ = self._routes[Provider(model.provider)]
            claimed = (call_id, lease.attempt_id, lease.fence, role, model, policy.version, digest)
            cleared = Cleared(model, texts, request.schema, issuer=_ISSUER)
            record = self._settle(claimed, cleared, sent, context, item, cancel)
            records.append(record)
            if record.outcome == "completed":
                text = self._log.reply(record)
                return Answer(Material(text, item), _judge(text, request.schema)[1], record)
            if record.outcome == "cancelled" or cancel.is_set():
                break
        return Unanswered(tuple(records))

    def close(self, lease: Lease) -> tuple[Receipt, ...]:
        """End the lease, seal each claim without a record, and return the lease's sent calls.

        Args:
            lease (Lease): Lease of the attempt to close.

        Returns:
            (tuple[Receipt, ...]) One receipt per sent call, in chain order. Closing again
            stores nothing and returns the same receipts.
        """
        self._log.append(Closed(lease.attempt_id))
        held = self._log._held()
        for call_id in lease.dispatches:
            claim = held.get((Dispatched, call_id))
            if claim is not None and (CallRecord, call_id) not in held:
                self._log.append(_sealed(claim))
        held = self._log._held()
        records = (held.get((CallRecord, call_id)) for call_id in lease.dispatches)
        return tuple(
            Receipt(record.call_id, record.role, record.outcome, record.usage)
            for record in records
            if record is not None and record.outcome != "refused"
        )

    def _settle(
        self,
        claimed: tuple,
        cleared: Cleared,
        sent: bytes,
        context: RunContext,
        item: Restrictions,
        cancel: threading.Event,
    ) -> CallRecord:
        """Return the final record of one call ID, sending it only if this holder claims it."""
        call_id, attempt_id, fence, role, model, _, digest = claimed
        policy, adapter = self._routes[Provider(model.provider)]
        while True:
            held = self._log._held()
            record, claim = held.get((CallRecord, call_id)), held.get((Dispatched, call_id))
            if record is not None:
                stored = (record.attempt_id, record.role, record.model, record.request_digest)
                if stored != (attempt_id, role, model, digest):
                    raise ProviderError(f"Call {call_id} is recorded for another request")
                return record
            if claim is not None and claim.fence > fence:
                raise ProviderError(f"Activity {claim.fence} holds the lease of {attempt_id}")
            if claim is not None and claim.fence == fence:
                raise ProviderError(f"Call {call_id} is already being sent by this activity")
            if claim is not None:
                self._log.append(_sealed(claim))
            elif (Closed, attempt_id) in held:
                raise ProviderError(f"Attempt {attempt_id} is closed")
            elif reasons := policy.denials(context, item):
                self._log.append(CallRecord(*claimed, "refused", reasons, Usage(), None, None))
            else:
                # Kept before the claim, so a store that cannot hold it stops the call unsent.
                self._log._keep(sent)
                mine = Dispatched(*claimed, adapter.subscription, adapter.runtime_version, uuid4())
                if self._log.append(mine) == mine:
                    reply = adapter.complete(cleared, cancel)
                    kept = (
                        self._log._keep(_reply_bytes(reply.text))
                        if isinstance(reply, Replied)
                        else None
                    )
                    self._log.append(_replied(mine, reply, cleared.schema, kept))


def _sealed(claim: Dispatched) -> CallRecord:
    """Return the record of a claim whose reply never arrived: one call, nothing else known."""
    known = {"cost_usd"} if claim.subscription else set()
    usage = Usage(model_calls=1, unreported=frozenset(REPORTABLE) - known)
    served = Served(claim.runtime_version, None)
    return CallRecord(*_claimed(claim), "unknown", (), usage, None, served)


def _replied(
    claim: Dispatched, reply: Reply, schema: Schema | None, reply_digest: str | None
) -> CallRecord:
    """Return a sent call's record. The gateway counts one call and a subscription costs nothing."""
    usage = replace(reply.usage, model_calls=1)
    if claim.subscription:
        usage = replace(usage, cost_usd=Decimal(0), unreported=usage.unreported - {"cost_usd"})
    served = Served(claim.runtime_version, reply.resolved_model)
    outcome = (
        _judge(reply.text, schema)[0] if isinstance(reply, Replied) else _UNREPLIED[type(reply)]
    )
    reason = None if isinstance(reply, Replied) else reply.reason
    return CallRecord(*_claimed(claim), outcome, (), usage, reply_digest, served, reason)


def _reply_bytes(text: str) -> bytes:
    """Return a reply transcript: the text as one ASCII JSON string, so any str encodes."""
    return json.dumps(text).encode()


def _judge(text: str, schema: Schema | None) -> tuple[Outcome, dict[str, Value] | None]:
    if schema is None:
        return "completed", None
    try:
        values = json.loads(text, object_pairs_hook=unique_keys)
    except (ValueError, RecursionError):
        return "malformed_output", None
    exact = (
        isinstance(values, dict)
        and values.keys() == {name for name, _ in schema}
        and all(type(values[name]) is kind for name, kind in schema)
        and all(math.isfinite(value) for value in values.values() if type(value) is float)
    )
    return ("completed", values) if exact else ("malformed_output", None)


def _json_default(value: object) -> object:
    return (
        json_default(value)
        if isinstance(value, Decimal | UUID | frozenset)
        else _model_json_default(value)
    )


def _encoded(value: object) -> object:
    # The ledger's context pins the exponent letter that str() writes for a Decimal.
    with localcontext(EXACT):
        return json.loads(json.dumps(value, default=_json_default))


class CallLog:
    """Keep a gateway's claims, call records, and closes as numbered JSON files.

    Entry i is the file i.json in an existing directory. Every operation first replays the
    stored entries, so logs opened on one directory never act on stale state. Transcripts live
    in the sibling directory <name>.transcripts, because read_log refuses any other file in the
    log directory. Each is named by the SHA-256 digest of its bytes. A claim's request_digest
    names the request's to_bytes, and a record's reply_digest names its reply as a JSON string.
    """

    def __init__(self, directory: Path) -> None:
        """Open a call log and verify every stored entry.

        Args:
            directory (Path): Existing directory that holds the numbered entry files.
        """
        self._directory = directory
        resolved = directory.resolve()
        self._transcripts = resolved.with_name(f"{resolved.name}.transcripts")
        self.load()

    def load(self) -> tuple[Entry, ...]:
        """Return every stored entry in order, rechecking each one.

        Returns:
            (tuple[Entry, ...]) Claims, call records, and closes in log order.
        """
        return tuple(self._held().values())

    def _held(self) -> dict[tuple[type, UUID], Entry]:
        """Return every stored entry under its key, in log order, rechecking each one.

        Returns:
            (dict[tuple[type, UUID], Entry]) Claims and records keyed by their type and call ID,
            and closes by Closed and attempt ID.
        """
        held: dict[tuple[type, UUID], Entry] = {}
        try:
            for index, value in enumerate(
                read_log(self._directory, "Call log entry", "Entry files")
            ):
                entry = _entry(value)
                if (stored := _admit(held, entry)) is not entry:
                    problem = "repeats an entry" if stored == entry else "conflicts with an entry"
                    raise ProviderError(f"Call log entry {index}.json {problem}")
        except (OSError, ValueError, RecursionError) as error:
            raise ProviderError(f"Cannot replay call log {self._directory}: {error}") from error
        return held

    def append(self, entry: Entry) -> Entry:
        """Store entry unless its key already holds one, and return the entry the key holds.

        The first writer of a key wins and every writer learns which entry won. A claim of an ID
        that already has a record returns the record. The entry is read back as replay reads it
        before anything is written, so an entry that would fail the next open raises
        ProviderError instead.

        Args:
            entry (Entry): A claim, a call record, or the close of an attempt.

        Returns:
            (Entry) The entry stored under the key, equal to entry when this call stored it.
        """
        try:
            raw = _encoded({"schema_version": _SCHEMA_VERSION, _KINDS[type(entry)]: asdict(entry)})
            content = canonical_bytes(raw)
            entry = _entry(raw)
        except ValueError as error:
            raise ProviderError(f"Call log cannot store entry: {error}") from error

        def attempt() -> tuple[int, Entry, bytes | None]:
            held = self._held()
            count = len(held)
            stored = _admit(held, entry)
            return count, stored, content if stored is entry else None

        try:
            return append_log(self._directory, attempt)
        except OSError as error:
            raise ProviderError(f"Cannot write call log {self._directory}: {error}") from error

    def sent(self, record: CallRecord) -> Request:
        """Return the request that a sent record's call sent, read from its transcript.

        Args:
            record (CallRecord): A record of this log whose outcome is not refused.

        Returns:
            (Request) The decoded request, each material under the derived restrictions.
        """
        return Request.from_bytes(self._read(record.request_digest))

    def reply(self, record: CallRecord) -> str:
        """Return the reply that a completed or malformed record names, read from its transcript.

        Args:
            record (CallRecord): A record of this log that holds a reply.

        Returns:
            (str) The reply text, whose transcript's bytes hash to the record's reply_digest.
        """
        if record.reply_digest is None:
            raise ProviderError(f"Call {record.call_id} holds no reply")
        text = json.loads(self._read(record.reply_digest))
        if type(text) is not str:
            raise ProviderError(f"Transcript {record.reply_digest} is not a reply")
        return text

    def _keep(self, content: bytes) -> str:
        """Store content under its SHA-256 digest unless it is there already; return the digest."""
        try:
            self._transcripts.mkdir(exist_ok=True)
            return keep_object(self._transcripts, content)
        except ObjectConflict as error:
            raise ProviderError(f"Transcript {error} holds other bytes") from error
        except OSError as error:
            raise ProviderError(f"Cannot write transcripts {self._transcripts}: {error}") from error

    def _read(self, digest: str) -> bytes:
        """Return the transcript stored under digest, raising unless its bytes hash to digest."""
        path = self._transcripts / parse_digest(digest, "transcript digest")
        try:
            content = path.read_bytes()
        except OSError as error:
            raise ProviderError(f"Cannot read transcript {digest}: {error}") from error
        if hashlib.sha256(content).hexdigest() != digest:
            raise ProviderError(f"Transcript {digest} holds other bytes")
        return content


_SCHEMA_VERSION = 4
_KINDS = {Dispatched: "dispatch", CallRecord: "call", Closed: "close"}
_CLAIM = "call_id attempt_id fence role model policy_version request_digest"


def _claimed(entry: Dispatched | CallRecord) -> tuple:
    return tuple(getattr(entry, name) for name in _CLAIM.split())


def _admit(held: dict[tuple[type, UUID], Entry], entry: Entry) -> Entry:
    """Store entry under its key unless the key holds one, and return the entry the key holds.

    Raises ProviderError for an entry that no order of writers stores: a sent record unlike its
    claim or naming another runtime, an unknown record unlike its claim's seal, a refused record
    of a claimed call, or a claim of a closed attempt.
    """
    key = (type(entry), entry.attempt_id if isinstance(entry, Closed) else entry.call_id)
    if key in held:
        return held[key]
    match entry:
        case Dispatched() if (CallRecord, entry.call_id) in held:
            return held[CallRecord, entry.call_id]
        case Dispatched() if (Closed, entry.attempt_id) in held:
            raise ProviderError(f"Call {entry.call_id} is claimed after its attempt closed")
        case CallRecord(outcome="refused") if (Dispatched, entry.call_id) in held:
            raise ProviderError(f"Call {entry.call_id} is refused after its claim")
        case CallRecord(served=Served(runtime_version=runtime)) if (
            claim := held.get((Dispatched, entry.call_id))
        ) is None or (_claimed(entry), runtime) != (_claimed(claim), claim.runtime_version):
            raise ProviderError(f"Call {entry.call_id} does not match its claim")
        case CallRecord(outcome="unknown") if entry != _sealed(held[Dispatched, entry.call_id]):
            raise ProviderError(f"Call {entry.call_id} is sealed unlike its claim")
    held[key] = entry
    return entry


def _entry(raw: object) -> Entry:
    if (
        not isinstance(raw, dict)
        or len(raw) != 2
        or type(raw.get("schema_version")) is not int
        or raw["schema_version"] not in (3, 4)
    ):
        raise ProviderError("A call log entry holds schema_version 3 or 4 and one entry")
    match raw:
        case {"dispatch": value}:
            value = parse_fields(value, f"{_CLAIM} subscription runtime_version token", "dispatch")
            if type(value["subscription"]) is not bool:
                raise ProviderError("A claim's subscription is true or false")
            return Dispatched(
                *_claim(value),
                value["subscription"],
                parse_text(value["runtime_version"], "runtime version"),
                parse_uuid(value["token"]),
            )
        case {"call": value}:
            return _record(value, raw["schema_version"])
        case {"close": value}:
            return Closed(parse_uuid(parse_fields(value, "attempt_id", "close")["attempt_id"]))
    raise ProviderError(f"Unknown call log entry {sorted(raw)!r}")


def _record(raw: object, version: int | None = None) -> CallRecord:
    if version is None:
        version = 4 if isinstance(raw, dict) and "reason" in raw else 3
    names = f"{_CLAIM} outcome reasons usage reply_digest served"
    value = parse_fields(raw, names + (" reason" if version == 4 else ""), "call record")
    reasons = tuple(parse_array(value["reasons"], "reasons"))
    if value["outcome"] not in _OUTCOMES:
        raise ProviderError("Unsupported outcome")
    if any(reason not in _REASONS for reason in reasons):
        raise ProviderError("Unsupported reasons")
    usage = parse_usage(value["usage"])
    if value["outcome"] != "refused" and usage.model_calls != 1:
        raise ProviderError("A sent call is one model call")
    reply_digest, served = value["reply_digest"], value["served"]
    return CallRecord(
        *_claim(value),
        value["outcome"],
        reasons,
        usage,
        None if reply_digest is None else parse_digest(reply_digest, "reply digest"),
        None if served is None else _served(served),
        value.get("reason"),
    )


def _served(raw: object) -> Served:
    value = parse_fields(raw, "runtime_version resolved_model", "served")
    model = value["resolved_model"]
    return Served(
        parse_text(value["runtime_version"], "runtime version"),
        None if model is None else parse_text(model, "resolved model"),
    )


def _claim(value: dict) -> tuple[UUID, UUID, int, str, ModelRecord, str, str]:
    if type(value["fence"]) is not int or value["fence"] < 0:
        raise ProviderError("A fence is a nonnegative integer")
    return (
        parse_uuid(value["call_id"]),
        parse_uuid(value["attempt_id"]),
        value["fence"],
        parse_text(value["role"], "role"),
        parse_model_record(value["model"]),
        parse_digest(value["policy_version"]),
        parse_digest(value["request_digest"], "request digest"),
    )


def dump_receipts(receipts: tuple[Receipt, ...]) -> str:
    """Return receipts as canonical JSON text, the form in which they leave the close activity."""
    return canonical_bytes(_encoded([asdict(receipt) for receipt in receipts])).decode()


def load_receipts(text: str) -> tuple[Receipt, ...]:
    """Return the receipts that dump_receipts wrote."""
    receipts = []
    for raw in parse_array(json.loads(text), "receipts"):
        value = parse_fields(raw, "call_id role outcome usage", "receipt")
        if value["outcome"] not in _SENT:
            raise ProviderError("Unsupported receipt outcome")
        receipts.append(
            Receipt(
                parse_uuid(value["call_id"]),
                parse_text(value["role"], "role"),
                value["outcome"],
                parse_usage(value["usage"]),
            )
        )
    return tuple(receipts)


def dump_result(result: Answer | Unanswered) -> str:
    """Return a gateway result as ASCII JSON text, the form in which it leaves the model activity.

    The text holds the completed reply, or null, and each record in the call log's own form. An
    answer's restrictions and values are left out, since both derive from its request.
    """
    text, records = (
        (result.material.text, (result.record,))
        if isinstance(result, Answer)
        else (None, result.records)
    )
    value = {"text": text, "records": [asdict(record) for record in records]}
    return json.dumps(_encoded(value), sort_keys=True)


def load_result(text: str, request: Request) -> Answer | Unanswered:
    """Return the result that dump_result wrote for a call of request.

    Args:
        text (str): Text that dump_result wrote.
        request (Request): The request the call sent, whose restrictions and schema an answer takes.

    Returns:
        (Answer | Unanswered) The result. Every record must name request's digest, and an
        answer's one completed record must name its reply's transcript.
    """
    value = parse_fields(json.loads(text, object_pairs_hook=unique_keys), "text records", "result")
    records = tuple(_record(raw) for raw in parse_array(value["records"], "records"))
    digest = request.digest
    if any(record.request_digest != digest for record in records):
        raise ProviderError("A result holds a record of another request")
    reply = value["text"]
    if reply is None:
        if not records or any(record.outcome == "completed" for record in records):
            raise ProviderError("An unanswered result holds records, and none completed")
        return Unanswered(records)
    if type(reply) is not str or [(r.outcome, r.reply_digest) for r in records] != [
        ("completed", hashlib.sha256(_reply_bytes(reply)).hexdigest())
    ]:
        raise ProviderError("An answer holds its reply and the one completed record naming it")
    outcome, values = _judge(reply, request.schema)
    if outcome != "completed":
        raise ProviderError("An answer's reply does not fit its request's schema")
    return Answer(Material(reply, request.restrictions), values, records[0])


def dump_material(material: Material) -> str:
    """Return a material as ASCII JSON text, the form in which it leaves the tool activity."""
    value = {"text": material.text, "restrictions": _restrictions(material.restrictions)}
    return json.dumps(value, sort_keys=True)


def load_material(text: str) -> Material:
    """Return the material that dump_material wrote."""
    value = parse_fields(
        json.loads(text, object_pairs_hook=unique_keys), "text restrictions", "material"
    )
    if type(value["text"]) is not str:
        raise ProviderError("A material's text must be text")
    return Material(value["text"], _parse_restrictions(value["restrictions"]))


class Fault(Enum):
    FAIL = "fail"
    CANCEL = "cancel"
    RATE_LIMIT = "rate_limit"
    BLOCK = "block"


class ScriptedProvider:
    """Answer each call with the next step of a script, offline and deterministically.

    A text step replies with that text, FAIL reports a provider failure, RATE_LIMIT a usage cap,
    and CANCEL sets the event as a caller cancelling mid-call would. BLOCK sets blocking, then
    holds the call until the caller sets the event and reports Cancelled. blocking is one event
    per instance, so give each blocking call its own provider. Every reply names the requested
    model as the one served. Usage is one call, with whitespace-separated words in and out as
    tokens, priced at usd_per_token each. usd_per_token=None scripts a subscription, which
    reports its word counts as lower bounds of unreported tokens.
    """

    runtime_version = "scripted"

    def __init__(
        self, script: Iterable[str | Fault], *, usd_per_token: Decimal | None = Decimal("0.000002")
    ) -> None:
        self.calls: list[Cleared] = []
        self.blocking = threading.Event()
        self.subscription = usd_per_token is None
        self._script = iter(script)
        self._usd_per_token = usd_per_token

    def complete(self, request: Cleared, cancel: threading.Event) -> Reply:
        """Record the request, then take the next step unless the call is already cancelled."""
        self.calls.append(request)
        sent = sum(len(text.split()) for text in request.texts)
        served = request.model.model
        if cancel.is_set():
            return Cancelled(self._usage(sent, 0), served)
        match next(self._script):
            case Fault.FAIL:
                return Failed(self._usage(sent, 0), served)
            case Fault.RATE_LIMIT:
                return RateLimited(self._usage(sent, 0), served)
            case Fault.CANCEL:
                cancel.set()
                return Cancelled(self._usage(sent, 0), served)
            case Fault.BLOCK:
                self.blocking.set()
                cancel.wait()
                return Cancelled(self._usage(sent, 0), served)
            case str(text):
                return Replied(text, self._usage(sent, len(text.split())), served)

    def _usage(self, sent: int, received: int) -> Usage:
        if self.subscription:
            unreported = frozenset({"input_tokens", "output_tokens"})
            return Usage(
                model_calls=1, input_tokens=sent, output_tokens=received, unreported=unreported
            )
        return Usage(
            model_calls=1,
            input_tokens=sent,
            output_tokens=received,
            cost_usd=self._usd_per_token * (sent + received),
        )
