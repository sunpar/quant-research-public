"""Register terminal research procedures and record prospective observations in one log.

Schema 1 preserves the original availability-only records. Schema 2 records completion metadata,
original rows, and replayable reads. Scored rows and new ancillary inputs require publication
after registration. Exact registered initialization can precede it. The caller supplies the
evaluation outcome, and the registry verifies evidence eligibility rather than numerical work.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from itertools import pairwise
from pathlib import Path
from typing import Literal
from uuid import UUID

from quant_research import datasets, reference
from quant_research.campaigns import parse_file
from quant_research.confirmation import FrozenShortlist, Period, freeze
from quant_research.model_provenance import UNKNOWN, ModelRecord, Unknown, parse_model_record
from quant_research.model_provenance import json_default as _model_json_default
from quant_research.packages import Outcome, VerifiedPackage, parse_outcome
from quant_research.records import (
    append_log,
    canonical_bytes,
    format_time,
    keep_object,
    parse_array,
    parse_digest,
    parse_fields,
    parse_name,
    parse_text,
    parse_time,
    parse_uuid,
    read_log,
    unique_keys,
)


class ProspectiveError(ValueError):
    """Report an event or stored log that the prospective registry refuses."""


@dataclass(frozen=True)
class Horizon:
    """The monthly holding duration of the reference experiment's returns."""

    months: int = 1

    def __post_init__(self) -> None:
        if type(self.months) is not int or self.months != 1:
            raise ProspectiveError("The reference evaluation has a one-month horizon")

    def mature_at(self, last_month: datetime) -> datetime:
        month = reference.shift(last_month.date().replace(day=1), self.months)
        return datetime(month.year, month.month, 1, tzinfo=UTC)


@dataclass(frozen=True)
class RowEvidence:
    dataset: str
    row: datasets.Row

    @property
    def identity(self) -> tuple[str, str, int]:
        return self.dataset, self.row.key, self.row.revision


@dataclass(frozen=True)
class ReadEvidence:
    """Replay original bytes, with an explicit subset of scored row identities."""

    record: datasets.ReadRecord
    vintage: bytes
    scored: tuple[tuple[str, int], ...] = ()

    def rows(self) -> tuple[RowEvidence, ...]:
        return tuple(
            RowEvidence(self.record.dataset, row) for row in self.record.replay(self.vintage)
        )


@dataclass(frozen=True)
class Completion:
    campaign: UUID
    revision: str
    specification: str
    selection: Literal["frozen", "unfrozen", "invalid", "incomplete"]
    reason: str
    configured_models: tuple[tuple[str, str, tuple[str, ...]], ...]
    horizon: Horizon | Unknown


@dataclass(frozen=True)
class Procedure:
    shortlist: FrozenShortlist | None
    models: tuple[ModelRecord, ...] = ()
    completion: Completion | None = None

    @property
    def identity(self) -> str:
        """Return the identity of the frozen procedure, preserving legacy hashes."""
        shortlist = (
            None if self.shortlist is None else hashlib.sha256(self.shortlist.manifest).hexdigest()
        )
        value = {"shortlist": shortlist, "models": [asdict(model) for model in self.models]}
        if self.completion is not None:
            completion = asdict(self.completion)
            del completion["reason"]
            value.update(schema_version=2, completion=completion)
        return hashlib.sha256(canonical_bytes(_plain(value))).hexdigest()


@dataclass(frozen=True)
class Registration:
    procedure: Procedure
    registered_at: datetime
    package: str | None = None
    initialization: tuple[RowEvidence, ...] | None = None
    reads: tuple[datasets.ReadRecord, ...] | None = None

    @property
    def registration_id(self) -> str:
        """Return the SHA-256 digest of the stored form of this registration."""
        return hashlib.sha256(canonical_bytes(_encoded(self))).hexdigest()


@dataclass(frozen=True)
class Observation:
    """Record the caller's outcome with original scored and ancillary input evidence."""

    registration: str
    procedure: str
    window: Period
    available_at: datetime
    observed_at: datetime
    outcome: Outcome
    inputs: tuple[RowEvidence, ...] | None = None
    evaluated: tuple[RowEvidence, ...] | None = None
    reads: tuple[datasets.ReadRecord, ...] | None = None


type Event = Registration | Observation


@dataclass(frozen=True)
class Entry:
    registration_id: str
    registration: Registration
    observations: tuple[Observation, ...]


def json_default(value: datetime | date | Unknown | UUID) -> str | None:
    if isinstance(value, UUID):
        return str(value)
    return format_time(value) if isinstance(value, datetime) else _model_json_default(value)


def _plain(value: object) -> object:
    return json.loads(json.dumps(value, default=json_default, allow_nan=False))


def _encoded(event: Event) -> dict:
    value = {"kind": type(event).__name__.lower(), **asdict(event)}
    if isinstance(event, Registration) and event.procedure.completion is None:
        for name in ("package", "initialization", "reads"):
            if value[name] is not None:
                raise ProspectiveError("Legacy registration cannot carry partial evidence")
            del value[name]
        del value["procedure"]["completion"]
    elif isinstance(event, Observation) and event.inputs is None:
        if event.evaluated is not None or event.reads is not None:
            raise ProspectiveError("Observation evidence must be complete")
        for name in ("inputs", "evaluated", "reads"):
            del value[name]
    else:
        value["schema_version"] = 2
    return _plain(value)


def _shortlist(raw: object) -> FrozenShortlist:
    value = parse_fields(
        raw, "candidates preprocessing training portfolio_rules evaluation identity", "shortlist"
    )
    shortlist = freeze(
        tuple(parse_array(value["candidates"], "candidates")),
        value["preprocessing"],
        value["training"],
        value["portfolio_rules"],
        value["evaluation"],
    )
    if shortlist.identity != value["identity"]:
        raise ProspectiveError("Shortlist changed after freezing")
    return shortlist


def _rows(raw: object) -> tuple[RowEvidence, ...]:
    result = []
    for item in parse_array(raw, "rows"):
        evidence = parse_fields(item, "dataset row", "row evidence")
        row = parse_fields(evidence["row"], "key revision published_at arrived_at values", "row")
        if type(row["revision"]) is not int or row["revision"] < 1:
            raise ProspectiveError("Row revision must be a positive integer")
        if not isinstance(row["values"], dict):
            raise ProspectiveError("Row values must be an object")
        result.append(
            RowEvidence(
                parse_name(evidence["dataset"]),
                datasets.Row(
                    parse_text(row["key"], "row key"),
                    row["revision"],
                    parse_time(row["published_at"]),
                    parse_time(row["arrived_at"]),
                    datasets.Values(row["values"]),
                ),
            )
        )
    if len({item.identity for item in result}) != len(result):
        raise ProspectiveError("Duplicate row evidence")
    return tuple(result)


def _reads(raw: object) -> tuple[datasets.ReadRecord, ...]:
    result = []
    for item in parse_array(raw, "reads"):
        value = parse_fields(item, "dataset vintage catalog cutoff context supplied", "read")
        value = {
            **value,
            "schema_version": 1,
            "supplied": [{"key": key, "revision": revision} for key, revision in value["supplied"]],
        }
        result.append(datasets.ReadRecord.from_bytes(canonical_bytes(value)))
    return tuple(result)


def _completion(raw: object) -> Completion:
    value = parse_fields(
        raw,
        "campaign revision specification selection reason configured_models horizon",
        "completion",
    )
    if value["selection"] not in ("frozen", "unfrozen", "invalid", "incomplete"):
        raise ProspectiveError("Unsupported selection disposition")
    horizon = value["horizon"]
    if horizon is not None:
        horizon = Horizon(parse_fields(horizon, "months", "horizon")["months"])
    return Completion(
        parse_uuid(value["campaign"]),
        parse_digest(value["revision"]),
        parse_digest(value["specification"]),
        value["selection"],
        parse_text(value["reason"], "reason"),
        tuple(
            (
                parse_text(role, "role"),
                parse_text(model, "model"),
                tuple(parse_text(item, "fallback") for item in parse_array(fallbacks, "fallbacks")),
            )
            for role, model, fallbacks in parse_array(
                value["configured_models"], "configured_models"
            )
        ),
        UNKNOWN if horizon is None else horizon,
    )


def _event(raw: object) -> Event:
    kind = raw.get("kind") if isinstance(raw, dict) else None
    versioned = isinstance(raw, dict) and "schema_version" in raw
    extra = " schema_version" if versioned else ""
    if versioned and (type(raw["schema_version"]) is not int or raw["schema_version"] != 2):
        raise ProspectiveError("Unsupported event schema version")
    if kind == "registration":
        value = parse_fields(
            raw,
            "kind procedure registered_at"
            + extra
            + (" package initialization reads" if versioned else ""),
            "registration",
        )
        procedure = parse_fields(
            value["procedure"],
            "shortlist models" + (" completion" if versioned else ""),
            "procedure",
        )
        models = tuple(
            parse_model_record(item) for item in parse_array(procedure["models"], "models")
        )
        if len(set(models)) != len(models):
            raise ProspectiveError("Duplicate model records")
        completion = _completion(procedure["completion"]) if versioned else None
        shortlist = (
            None
            if versioned and procedure["shortlist"] is None
            else _shortlist(procedure["shortlist"])
        )
        if completion is not None and (completion.selection == "frozen") != (shortlist is not None):
            raise ProspectiveError("Only a validated frozen selection can carry a shortlist")
        return Registration(
            Procedure(shortlist, models, completion),
            parse_time(value["registered_at"]),
            parse_digest(value["package"]) if versioned else None,
            _rows(value["initialization"]) if versioned else None,
            _reads(value["reads"]) if versioned else None,
        )
    if kind == "observation":
        value = parse_fields(
            raw,
            "kind registration procedure window available_at observed_at outcome"
            + extra
            + (" inputs evaluated reads" if versioned else ""),
            "observation",
        )
        window = parse_fields(value["window"], "dataset_id start end", "window")
        return Observation(
            parse_digest(value["registration"]),
            parse_digest(value["procedure"]),
            Period(window["dataset_id"], parse_time(window["start"]), parse_time(window["end"])),
            parse_time(value["available_at"]),
            parse_time(value["observed_at"]),
            parse_outcome(value["outcome"]),
            _rows(value["inputs"]) if versioned else None,
            _rows(value["evaluated"]) if versioned else None,
            _reads(value["reads"]) if versioned else None,
        )
    raise ProspectiveError(f"Unsupported event kind {kind!r}")


def _evidence(event: Event) -> tuple[RowEvidence, ...]:
    if isinstance(event, Registration):
        return event.initialization or ()
    return (*(event.inputs or ()), *(event.evaluated or ()))


def _read_coverage(event: Event) -> None:
    rows = _evidence(event)
    declared = {item.identity: item.row.available_at for item in rows}
    supplied = {
        (read.dataset, key, revision)
        for read in event.reads or ()
        for key, revision in read.supplied
    }
    if declared.keys() != supplied or len(declared) != len(rows):
        raise ProspectiveError("Read records must cover exactly the declared original rows")
    for read in event.reads or ():
        if any(
            declared[read.dataset, key, revision] > read.cutoff for key, revision in read.supplied
        ):
            raise ProspectiveError("Read predates a supplied row's original availability")


def _at(event: Event) -> datetime:
    return event.registered_at if isinstance(event, Registration) else event.observed_at


class Registry:
    """Hold an append-only log of registrations and observations and the report folded from it."""

    def __init__(self, events: tuple[Event, ...] = ()) -> None:
        """Fold a new or stored log, raising ProspectiveError at the first event that breaks a rule.

        An invalid field value raises PackageError, as it does in freeze and Period.

        Args:
            events (tuple): Registrations and observations in non-decreasing time order.
        """
        self._events = tuple(
            _event(json.loads(canonical_bytes(_encoded(event)))) for event in events
        )
        if any(_at(later) < _at(earlier) for earlier, later in pairwise(self._events)):
            raise ProspectiveError("Event times must not decrease")
        if len({canonical_bytes(_encoded(event)) for event in self._events}) != len(self._events):
            raise ProspectiveError("An event is already recorded")
        self._entries: dict[str, Entry] = {}
        known: dict[tuple[str, str, int], RowEvidence] = {}
        for event in self._events:
            for item in _evidence(event):
                if item.identity in known and known[item.identity] != item:
                    raise ProspectiveError("A recorded row changed its original values or dates")
                known[item.identity] = item
            if event.reads is not None:
                _read_coverage(event)
            if isinstance(event, Registration):
                if any(
                    item.row.available_at > event.registered_at
                    for item in event.initialization or ()
                ):
                    raise ProspectiveError("Initialization was unavailable at registration")
                self._entries[event.registration_id] = Entry(event.registration_id, event, ())
            else:
                self._observe(event)

    @property
    def events(self) -> tuple[Event, ...]:
        """Return the log that the report derives from."""
        return self._events

    def append(self, event: Event) -> Registry:
        """Return a registry with the event appended, or this registry if it is already recorded."""
        return self if event in self._events else Registry((*self._events, event))

    def report(self) -> tuple[Entry, ...]:
        """Return each registration with its observations, both in log order."""
        return tuple(self._entries.values())

    def _observe(self, observation: Observation) -> None:
        entry = self._entries.get(observation.registration)
        if entry is None:
            raise ProspectiveError(f"Unknown registration {observation.registration}")
        registration = entry.registration
        if observation.procedure != registration.procedure.identity:
            raise ProspectiveError(
                f"Observation ran procedure {observation.procedure}, but registration "
                f"{entry.registration_id} froze procedure {registration.procedure.identity}"
            )
        completion = registration.procedure.completion
        if completion is not None:
            if (
                observation.inputs is None
                or observation.evaluated is None
                or observation.reads is None
            ):
                raise ProspectiveError(
                    "An evidence-bearing registration requires original read evidence"
                )
            if completion.selection != "frozen" or not isinstance(completion.horizon, Horizon):
                raise ProspectiveError(
                    "The registration has no complete frozen procedure and horizon"
                )
            if not observation.evaluated:
                raise ProspectiveError("The scored evidence must be nonempty")
            if any(
                instant != datetime(instant.year, instant.month, 1, tzinfo=UTC)
                for instant in (observation.window.start, observation.window.end)
            ):
                raise ProspectiveError("Monthly windows must start and end on a month's first day")
            if observation.observed_at < completion.horizon.mature_at(observation.window.end):
                raise ProspectiveError("The final monthly outcome has not matured")
            frozen = {item.identity: item for item in registration.initialization}
            evaluated = {item.identity for item in observation.evaluated}
            for item in _evidence(observation):
                scored_window = _in_window(item, observation.window)
                if item.identity in evaluated:
                    if not scored_window:
                        raise ProspectiveError(
                            "Scored rows must match the declared dataset and window"
                        )
                elif scored_window:
                    raise ProspectiveError("An in-window outcome cannot be an initialization input")
                elif frozen.get(item.identity) == item:
                    continue
                if item.row.published_at <= registration.registered_at:
                    raise ProspectiveError(
                        "New evidence must be published strictly after registration"
                    )
            if any(
                item.row.available_at > observation.observed_at for item in _evidence(observation)
            ):
                raise ProspectiveError("Observation is dated before an input became available")
            if observation.available_at != min(
                item.row.available_at for item in observation.evaluated
            ):
                raise ProspectiveError("Availability must be the earliest scored row availability")
        elif observation.inputs is not None:
            raise ProspectiveError("Original evidence requires an evidence-bearing registration")
        if observation.available_at <= registration.registered_at:
            raise ProspectiveError(
                f"Observation uses data available at {format_time(observation.available_at)}, "
                f"at or before registration at {format_time(registration.registered_at)}"
            )
        if observation.observed_at < observation.available_at:
            raise ProspectiveError("Observation is dated before its data became available")
        self._entries[entry.registration_id] = replace(
            entry, observations=(*entry.observations, observation)
        )


def dump_registry(registry: Registry) -> bytes:
    """Return the canonical stored form of a registry's log.

    Args:
        registry (Registry): Registry to store.

    Returns:
        (bytes) Canonical UTF-8 JSON of the schema version and every event in log order.
    """
    events = [_encoded(event) for event in registry.events]
    return canonical_bytes(
        {
            "schema_version": 2 if any("schema_version" in event for event in events) else 1,
            "events": events,
        }
    )


def load_registry(content: bytes) -> Registry:
    """Parse a stored log and fold it again, rechecking every rule.

    Args:
        content (bytes): Canonical bytes written by dump_registry.

    Returns:
        (Registry) The registry folded from the stored events.
    """
    try:
        value = parse_fields(
            json.loads(content.decode("utf-8")), "schema_version events", "registry"
        )
        if type(value["schema_version"]) is not int or value["schema_version"] not in (1, 2):
            raise ProspectiveError(f"Unsupported schema version {value['schema_version']!r}")
        if canonical_bytes(value) != content:
            raise ProspectiveError("Registry is not canonical UTF-8 JSON")
        events = parse_array(value["events"], "events")
        version = (
            2 if any(isinstance(item, dict) and "schema_version" in item for item in events) else 1
        )
        if version != value["schema_version"]:
            raise ProspectiveError("Registry schema version differs from its event shapes")
        return Registry(tuple(_event(item) for item in events))
    except (ValueError, RecursionError) as error:
        raise ProspectiveError(f"Cannot load registry: {error}") from error


def _in_window(item: RowEvidence, window: Period) -> bool:
    if item.dataset != window.dataset_id:
        return False
    try:
        month = date.fromisoformat(item.row.key)
    except ValueError as error:
        raise ProspectiveError("Scored dataset keys must name calendar months") from error
    if month.day != 1 or month.isoformat() != item.row.key:
        raise ProspectiveError("Scored dataset keys must name calendar months")
    return window.start.date() <= month <= window.end.date()


def load_log(root: Path) -> Registry:
    """Load durable events and verify every read against its persisted original vintage."""
    directory = root / "events"
    if not directory.exists():
        return Registry()
    registry = Registry(
        tuple(_event(raw) for raw in read_log(directory, "Prospective event", "Prospective events"))
    )
    for event in registry.events:
        replayed = {}
        for read in event.reads or ():
            content = (root / "evidence" / read.vintage).read_bytes()
            for row in read.replay(content):
                item = RowEvidence(read.dataset, row)
                if item.identity in replayed and replayed[item.identity] != item:
                    raise ProspectiveError("Persisted reads disagree on an original row")
                replayed[item.identity] = item
        if event.reads is not None and replayed != {
            item.identity: item for item in _evidence(event)
        }:
            raise ProspectiveError("Persisted vintage evidence differs from the event's rows")
    return registry


def _keep_reads(root: Path, reads: tuple[ReadEvidence, ...]) -> None:
    (root / "events").mkdir(parents=True, exist_ok=True)
    (root / "evidence").mkdir(parents=True, exist_ok=True)
    for evidence in reads:
        keep_object(root / "evidence", evidence.vintage)


def _package_procedure(package: VerifiedPackage) -> tuple[Procedure, tuple[ReadEvidence, ...]]:
    snapshot = json.loads(package.read_artifact("campaign-snapshot"))
    report = snapshot["report"]
    revision = parse_file(package.read_artifact(f"campaign-revision-{report['revision']}"))
    prefix = f"campaign-{revision.revision_id}"
    names = {artifact.name for artifact in package.artifact_records}
    specification = hashlib.sha256(
        canonical_bytes(_plain(asdict(revision.specification)))
    ).hexdigest()
    configured = tuple(
        (role.role, role.model, role.fallbacks) for role in revision.specification.model_roles
    )
    horizon = (
        Horizon() if revision.specification.evaluation.profile == "pilot-evaluation/v1" else UNKNOWN
    )
    models = ()
    reads = []
    rules = None
    if f"{prefix}-rules" in names:
        rules = reference.load_rules(package.read_artifact(f"{prefix}-rules"))
        if rules.digest != revision.specification.evaluation.digest:
            raise ProspectiveError("Pinned rules differ from the campaign revision")
        horizon = Horizon()
        model_roles = json.loads(package.read_artifact(f"{prefix}-models"))
        models = tuple(
            dict.fromkeys(
                parse_model_record(raw)
                for assignment in model_roles["assignments"]
                for raw in (assignment["primary"], *assignment["fallbacks"])
            )
        )
    authority = (
        json.loads(package.read_artifact(f"{prefix}-authority"))
        if f"{prefix}-authority" in names
        else {}
    )
    if (
        rules is not None
        or authority.get("schema_version") == 3
        or (
            f"{prefix}-context" in names
            and "dataset_periods" in json.loads(package.read_artifact(f"{prefix}-context"))
        )
    ):
        from quant_research.access import RunContext
        from quant_research.catalog import Catalog

        context = RunContext.from_bytes(package.read_artifact(f"{prefix}-context"))
        catalog = Catalog.from_bytes(package.read_artifact(f"{prefix}-catalog"))
        if context.dataset_periods is not None or authority.get("schema_version") == 3:
            if (
                context.dataset_periods is None
                or authority.get("schema_version") != 3
                or authority.get("context") != context.identity
                or authority.get("catalog") != catalog.digest
                or authority.get("dataset_periods")
                != [item.record() for item in context.dataset_periods]
            ):
                raise ProspectiveError("Archived context differs from its pinned period authority")
        admitted = (
            {item.dataset for item in datasets.listing(catalog, context).datasets}
            if context.dataset_periods is not None
            else set()
        )
        for entry in catalog.search():
            vintage = entry.documentation.vintage
            name = f"campaign-vintage-{vintage}"
            if context.dataset_periods is not None:
                if entry.name not in admitted:
                    continue
                binding = hashlib.sha256(
                    f"{revision.revision_id}:{entry.name}".encode()
                ).hexdigest()
                read_name = f"campaign-read-{binding}"
                if name not in names or read_name not in names:
                    raise ProspectiveError(
                        "A period-aware archive is missing its dataset read evidence"
                    )
                bound = datasets.ReadRecord.from_bytes(package.read_artifact(read_name))
                if bound.dataset != entry.name:
                    raise ProspectiveError("Archived read differs from its pinned authority")
                content = package.read_artifact(name)
                try:
                    bound.validate(content, catalog=catalog, context=context)
                except datasets.DatasetError as error:
                    raise ProspectiveError(
                        "Archived read differs from its pinned authority"
                    ) from error
                reads.append(ReadEvidence(bound, content))
                continue
            if name not in names or not context.permits(entry.name):
                continue
            content = package.read_artifact(name)
            identities = tuple((row["key"], row["revision"]) for row in json.loads(content)["rows"])
            bound = datasets.ReadRecord(
                entry.name, vintage, catalog.digest, context.cutoff, context.identity, identities
            )
            supplied = tuple(
                (row.key, row.revision)
                for row in bound.replay(content)
                if row.available_at <= context.cutoff
            )
            reads.append(ReadEvidence(replace(bound, supplied=supplied), content))
    selection, reason, shortlist = "unfrozen", "No frozen selection was recorded", None
    assessment = report.get("assessment")
    if assessment is not None:
        try:
            material = json.loads(assessment["shortlist"])
            reply = json.loads(material["text"])
            if reply["action"] == "conclude":
                value = parse_fields(
                    json.loads(reply["arguments"], object_pairs_hook=unique_keys),
                    "trials",
                    "shortlist selection",
                )
                sources = parse_array(value["trials"], "selected trials")
                trials = {str(trial.trial_id): trial for trial in package.record.trials}
                known = {step["trial"] for step in report["steps"] if step.get("trial") is not None}
                if len(set(sources)) != len(sources) or any(
                    type(source) is not str or source not in known for source in sources
                ):
                    raise ProspectiveError(
                        "The selection does not name unique recorded search trials"
                    )
                pairs = assessment["trials"]
                if [pair["source"] for pair in pairs] != sources:
                    raise ProspectiveError(
                        "The assessment does not bind the selected source trials"
                    )
                selection, reason = (
                    "incomplete",
                    "The frozen selection has incomplete authority or source evidence",
                )
                complete = (
                    rules is not None
                    and assessment["revision"] == revision.revision_id
                    and all(source in trials for source in sources)
                )
                for pair in pairs:
                    source, measured = trials.get(pair["source"]), trials.get(pair["trial"])
                    if (
                        source is not None
                        and measured is not None
                        and (
                            source.specification.signal_profile
                            != measured.specification.signal_profile
                            or not any(
                                str(link.trial_id) == pair["source"]
                                and link.spec_id == source.specification.spec_id
                                for link in measured.links
                            )
                        )
                    ):
                        raise ProspectiveError(
                            "Held-back measurement does not bind its source specification"
                        )
                if complete:
                    shortlist = freeze(
                        tuple(trials[source].specification.spec_id for source in sources),
                        rules.digest,
                        specification,
                        rules.held_back.digest,
                        revision.specification.evaluation.digest,
                    )
                    selection, reason = (
                        "frozen",
                        "Validated original selection and source specifications",
                    )
            elif reply["action"] != "stop":
                raise ProspectiveError("The selection contains no terminal action")
        except (ValueError, KeyError, TypeError, RecursionError):
            selection, reason, shortlist = (
                "invalid",
                "The original selection or its source bindings are invalid",
                None,
            )
    if rules is None:
        selection, reason, shortlist = (
            "incomplete",
            "Opening did not pin a resolved authority",
            None,
        )
    completion = Completion(
        revision.campaign_id,
        revision.revision_id,
        specification,
        selection,
        reason,
        configured,
        horizon,
    )
    return Procedure(shortlist, models, completion), tuple(reads)


def register_package(root: Path, package: VerifiedPackage) -> Registration:
    """Register a terminal package once per campaign, revision, and semantic procedure."""
    procedure, reads = _package_procedure(package)
    replayed = tuple((read.record, read.rows()) for read in reads)
    _keep_reads(root, reads)

    def attempt() -> tuple[int, Registration, bytes | None]:
        registry = load_log(root)
        for entry in registry.report():
            original = entry.registration
            if original.procedure.identity == procedure.identity:
                return len(registry.events), original, None
        registered_at = datetime.now(UTC)
        rows, records = {}, []
        for record, supplied in replayed:
            initialization = tuple(
                item for item in supplied if item.row.available_at <= registered_at
            )
            rows.update((item.identity, item) for item in initialization)
            records.append(
                replace(
                    record,
                    supplied=tuple((item.row.key, item.row.revision) for item in initialization),
                )
            )
        event = Registration(
            procedure,
            registered_at,
            package.digest,
            tuple(rows.values()),
            tuple(records),
        )
        registry.append(event)
        return len(registry.events), event, canonical_bytes(_encoded(event))

    return append_log(root / "events", attempt)


def observe(
    root: Path,
    registration: str,
    *,
    procedure: str,
    window: Period,
    reads: tuple[ReadEvidence, ...],
    outcome: Outcome,
    observed_at: datetime,
) -> Observation:
    """Record one caller-supplied evaluation after replaying all its declared input evidence."""
    procedure = parse_digest(procedure)
    rows, evaluated = {}, {}
    for read in reads:
        supplied = read.rows()
        scored = set(read.scored)
        if len(scored) != len(read.scored) or not scored <= set(read.record.supplied):
            raise ProspectiveError("Scored identities must be a unique subset of their read")
        for item in supplied:
            if item.identity in rows and rows[item.identity] != item:
                raise ProspectiveError("Reads disagree on an original row")
            rows[item.identity] = item
            if (item.row.key, item.row.revision) in scored:
                evaluated[item.identity] = item
    if not evaluated:
        raise ProspectiveError("The scored evidence must be nonempty")
    registry = load_log(root)
    event = Observation(
        registration,
        procedure,
        window,
        min(item.row.available_at for item in evaluated.values()),
        observed_at,
        outcome,
        tuple(item for identity, item in rows.items() if identity not in evaluated),
        tuple(evaluated.values()),
        tuple(read.record for read in reads),
    )
    registry.append(event)
    _keep_reads(root, reads)

    def attempt() -> tuple[int, Observation, bytes | None]:
        current = load_log(root)
        appended = current.append(event)
        return (
            len(current.events),
            event,
            None if appended is current else canonical_bytes(_encoded(event)),
        )

    return append_log(root / "events", attempt)
