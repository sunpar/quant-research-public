from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import stat
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Literal
from uuid import UUID, uuid4

from quant_research import reference
from quant_research.access import (
    NO_MEMORY,
    DatasetPeriods,
    Interval,
    MemoryPin,
    RestrictionLabel,
    RunContext,
    issue_context,
)
from quant_research.agent_tools import Generated, Toolset
from quant_research.catalog import Catalog
from quant_research.datasets import (
    DatasetError,
    contract_of,
    listing,
    period_contract,
    publish_contract,
)
from quant_research.findings import Ledger
from quant_research.model_provenance import MODELS, Activation, RoleAssignment, RoleVersion
from quant_research.packages import ExposureEvent, parse_exposure
from quant_research.providers import Material, ProviderPolicy
from quant_research.records import (
    canonical_bytes,
    format_time,
    json_default,
    parse_array,
    parse_digest,
    parse_fields,
    parse_name,
    parse_text,
    parse_time,
    parse_uuid,
)
from quant_research.retrieval import Corpus, RetrievalError, publish_corpus
from quant_research.sources import (
    TEXT_PIPELINE,
    SourceDeclaration,
    extracted_text,
    open_generation,
    publish_generation,
    register_source,
)

type Section = Literal[
    "question",
    "sources",
    "data",
    "evaluation",
    "limits",
    "interventions",
    "model_roles",
    "generated_code",
]


class CampaignError(ValueError):
    """Report an invalid campaign record, revision history, or filesystem failure."""


class RulesError(CampaignError):
    """Report evaluation rules that name no usable frozen rules file for a setup."""


@dataclass(frozen=True)
class Source:
    name: str
    locator: str
    digest: str
    role: str


@dataclass(frozen=True)
class DataPermission:
    dataset_id: str
    interval_start: datetime
    interval_end: datetime


@dataclass(frozen=True)
class EvaluationRules:
    profile: str
    digest: str


@dataclass(frozen=True)
class ResourceLimits:
    """Declare a campaign's limits; None leaves a limit added in schema_version 2 undeclared.

    The runner reserves model calls and wall seconds before each attempt, so those limits are
    hard. Token limits are observed: a subscription may not report tokens, so they refuse only
    once the reported sum passes them.
    """

    max_trials: int
    max_attempts: int
    max_compute_seconds: int
    max_model_calls: int | None = None
    max_wall_seconds: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None


PROMPT = "prompt"
CONTRACT = "contract"


@dataclass(frozen=True)
class ModelRole:
    """Name the model for a role. resolve activates an active role on an accepted provider only.

    model and each fallback are "<provider>/<model>", a key of model_provenance.MODELS.
    """

    role: str
    model: str
    fallbacks: tuple[str, ...] = ()
    status: Activation = "inactive"


@dataclass(frozen=True)
class CampaignSpecification:
    """generated_code lets the campaign's jobs run generated Python in the isolated worker."""

    question: str
    sources: tuple[Source, ...]
    data: tuple[DataPermission, ...]
    evaluation: EvaluationRules
    limits: ResourceLimits
    interventions: tuple[str, ...] = ()
    model_roles: tuple[ModelRole, ...] = ()
    generated_code: bool = False


_SECTIONS = tuple(field.name for field in fields(CampaignSpecification))
_LIMITS = tuple(field.name for field in fields(ResourceLimits))
_ADDED = _LIMITS[3:]


@dataclass(frozen=True)
class TrialReference:
    lineage_id: UUID
    trial_id: UUID


@dataclass(frozen=True)
class CampaignRevision:
    """Record one methodology snapshot and every trial and exposure observed before it.

    A child's trials and exposures begin with its parent's, so a revision cannot make
    observed data untouched.
    """

    campaign_id: UUID
    branch: str
    parent: str | None
    author: str
    created_at: datetime
    reason: str
    specification: CampaignSpecification
    trials: tuple[TrialReference, ...]
    exposures: tuple[ExposureEvent, ...]
    changes: tuple[Section, ...]

    @property
    def revision_id(self) -> str:
        return hashlib.sha256(file_bytes(self)).hexdigest()


@dataclass(frozen=True)
class Campaign:
    """Hold one campaign's validated revisions, parents first."""

    campaign_id: UUID
    revisions: tuple[CampaignRevision, ...]

    def head(self, branch: str) -> CampaignRevision:
        """Return the latest revision on a branch.

        Args:
            branch (str): Branch name.

        Returns:
            (CampaignRevision) The last revision in the branch's chain.
        """
        revision = next((item for item in reversed(self.revisions) if item.branch == branch), None)
        if revision is None:
            raise CampaignError(f"Unknown branch {branch!r}")
        return revision


def unique(values: Iterable, label: str) -> None:
    """Raise CampaignError for the first value that repeats an earlier one."""
    seen = set()
    for value in values:
        if value in seen:
            raise CampaignError(f"Duplicate {label} {value!r}")
        seen.add(value)


def _source(raw: object) -> Source:
    value = parse_fields(raw, "name locator digest role", "source")
    return Source(
        parse_name(value["name"], "source"),
        parse_text(value["locator"], "source locator"),
        parse_digest(value["digest"]),
        parse_text(value["role"], "source role"),
    )


def _permission(raw: object) -> DataPermission:
    value = parse_fields(raw, "dataset_id interval_start interval_end", "data permission")
    result = DataPermission(
        parse_text(value["dataset_id"], "dataset_id"),
        parse_time(value["interval_start"]),
        parse_time(value["interval_end"]),
    )
    if result.interval_end < result.interval_start:
        raise CampaignError(f"Data permission {result.dataset_id!r} ends before it starts")
    return result


def _role(raw: object) -> ModelRole:
    value = parse_fields(raw, "role model fallbacks status", "model role")
    if value["status"] not in ("inactive", "active"):
        raise CampaignError(
            f"Model role status must be 'inactive' or 'active', not {value['status']!r}"
        )
    return ModelRole(
        parse_text(value["role"], "model role"),
        parse_text(value["model"], "model"),
        tuple(
            parse_text(item, "model fallback")
            for item in parse_array(value["fallbacks"], "fallbacks")
        ),
        value["status"],
    )


def _limits(raw: object) -> ResourceLimits:
    present = (name for name in _ADDED if isinstance(raw, dict) and name in raw)
    limits = parse_fields(raw, " ".join((*_LIMITS[:3], *present)), "limits")
    for key, limit in limits.items():
        if type(limit) is not int or limit <= 0:
            raise CampaignError(f"Limit {key} must be a positive integer, not {limit!r}")
    return ResourceLimits(**limits)


def _specification(raw: object) -> CampaignSpecification:
    enabled = isinstance(raw, dict) and "generated_code" in raw
    sections = (section for section in _SECTIONS if enabled or section != "generated_code")
    value = parse_fields(raw, " ".join(sections), "specification")
    evaluation = parse_fields(value["evaluation"], "profile digest", "evaluation")
    result = CampaignSpecification(
        parse_text(value["question"], "question"),
        tuple(_source(item) for item in parse_array(value["sources"], "sources")),
        tuple(_permission(item) for item in parse_array(value["data"], "data")),
        EvaluationRules(
            parse_text(evaluation["profile"], "evaluation profile"),
            parse_digest(evaluation["digest"]),
        ),
        _limits(value["limits"]),
        tuple(
            parse_text(item, "intervention")
            for item in parse_array(value["interventions"], "interventions")
        ),
        tuple(_role(item) for item in parse_array(value["model_roles"], "model_roles")),
        enabled,
    )
    unique((source.name for source in result.sources), "source name")
    unique(result.data, "data permission")
    unique(result.interventions, "intervention")
    unique((role.role for role in result.model_roles), "model role")
    return result


def _trial_reference(raw: object) -> TrialReference:
    value = parse_fields(raw, "lineage_id trial_id", "trial reference")
    return TrialReference(parse_uuid(value["lineage_id"]), parse_uuid(value["trial_id"]))


def _revision(raw: object) -> CampaignRevision:
    value = parse_fields(
        raw,
        "campaign_id branch parent author created_at reason specification trials exposures changes",
        "revision",
    )
    changes = tuple(parse_array(value["changes"], "changes"))
    for section in changes:
        if section not in _SECTIONS:
            raise CampaignError(f"Unknown change section {section!r}")
    unique(changes, "change")
    result = CampaignRevision(
        parse_uuid(value["campaign_id"]),
        parse_name(value["branch"], "branch"),
        None if value["parent"] is None else parse_digest(value["parent"]),
        parse_text(value["author"], "author"),
        parse_time(value["created_at"]),
        parse_text(value["reason"], "reason"),
        _specification(value["specification"]),
        tuple(_trial_reference(item) for item in parse_array(value["trials"], "trials")),
        tuple(parse_exposure(item) for item in parse_array(value["exposures"], "exposures")),
        changes,
    )
    unique((trial.trial_id for trial in result.trials), "trial ID")
    unique((exposure.id for exposure in result.exposures), "exposure ID")
    for exposure in result.exposures:
        if exposure.at > result.created_at:
            raise CampaignError(f"Exposure {exposure.id} is dated after its revision")
    return result


def file_bytes(revision: CampaignRevision) -> bytes:
    """Return the canonical stored bytes of a campaign revision.

    A revision that sets any limit added in schema_version 2 is stored under version 2, and
    any other under version 1, so each revision has one stored byte string.
    """
    data = json.loads(json.dumps(asdict(revision), default=json_default, allow_nan=False))
    # Revisions written before generated code or the added limits existed keep their bytes and IDs.
    if revision.specification.generated_code is False:
        del data["specification"]["generated_code"]
    limits = data["specification"]["limits"]
    for name in _ADDED:
        if limits[name] is None:
            del limits[name]
    version = 2 if any(name in limits for name in _ADDED) else 1
    return canonical_bytes({"schema_version": version, "revision": data})


def parse_file(content: bytes) -> CampaignRevision:
    """Return the revision whose canonical stored bytes are exactly content."""
    value = parse_fields(
        json.loads(content.decode("utf-8")), "schema_version revision", "revision file"
    )
    if type(value["schema_version"]) is not int or value["schema_version"] not in (1, 2):
        raise CampaignError(f"Unsupported schema version {value['schema_version']!r}")
    if canonical_bytes(value) != content:
        raise CampaignError("Revision file is not canonical UTF-8 JSON")
    revision = _revision(value["revision"])
    if file_bytes(revision) != content:
        raise CampaignError("Revision file is not the stored form of its revision")
    return revision


def _revisions(directory: Path) -> dict[str, CampaignRevision]:
    revisions = {}
    for path in sorted(directory.iterdir()):
        if path.name.startswith("."):
            continue
        try:
            if not stat.S_ISREG(path.lstat().st_mode):
                raise CampaignError("Entry is not a regular file")
            content = path.read_bytes()
            if path.name != f"{hashlib.sha256(content).hexdigest()}.json":
                raise CampaignError("File name must be the SHA-256 of its bytes plus .json")
            revisions[path.stem] = parse_file(content)
        except (OSError, ValueError, RecursionError) as error:
            raise CampaignError(f"Invalid campaign entry {path.name!r}: {error}") from error
    return revisions


def _changes(parent: CampaignSpecification, child: CampaignSpecification) -> tuple[Section, ...]:
    return tuple(
        section for section in _SECTIONS if getattr(parent, section) != getattr(child, section)
    )


def _extend(recorded: tuple, observed: Iterable, kind: type, key: str, label: str) -> tuple:
    merged = {getattr(item, key): item for item in recorded}
    for item in observed:
        if not isinstance(item, kind):
            raise CampaignError(f"{label} must be given as {kind.__name__}, not {item!r}")
        if merged.setdefault(getattr(item, key), item) != item:
            raise CampaignError(f"{label} ID {getattr(item, key)} is already recorded differently")
    return tuple(merged.values())


def _observation_id(item: TrialReference | ExposureEvent) -> str:
    return str(item.trial_id if isinstance(item, TrialReference) else item.id)


def _observed(revision: CampaignRevision, kind: str) -> set[str]:
    return {_observation_id(item) for item in getattr(revision, kind)}


def check_link(parent: CampaignRevision, child: CampaignRevision) -> None:
    """Raise CampaignError unless child keeps its parent's campaign, records, and listed changes."""
    parent_id = parent.revision_id
    if child.campaign_id != parent.campaign_id:
        raise CampaignError(
            f"Revision {child.revision_id} belongs to campaign {child.campaign_id}, "
            f"not its parent's campaign {parent.campaign_id}"
        )
    if child.created_at < parent.created_at:
        raise CampaignError(f"Revision {child.revision_id} is dated before its parent {parent_id}")
    if child.trials[: len(parent.trials)] != parent.trials:
        raise CampaignError(
            f"Revision {child.revision_id} drops or changes trials of its parent {parent_id}"
        )
    if child.exposures[: len(parent.exposures)] != parent.exposures:
        raise CampaignError(
            f"Revision {child.revision_id} drops or changes exposures of its parent {parent_id}"
        )
    changes = _changes(parent.specification, child.specification)
    if child.changes != changes:
        raise CampaignError(
            f"Revision {child.revision_id} lists changes {list(child.changes)}, "
            f"but its specification changes {list(changes)}"
        )


def _history(revisions: dict[str, CampaignRevision], new: str | None = None) -> Campaign:
    children = defaultdict(list)
    for revision_id, revision in revisions.items():
        if revision.parent is not None:
            if revision.parent not in revisions:
                raise CampaignError(
                    f"Revision {revision_id} names missing parent {revision.parent}"
                )
            check_link(revisions[revision.parent], revision)
            children[revision.parent].append(revision_id)
    campaigns = sorted({str(revision.campaign_id) for revision in revisions.values()})
    if len(campaigns) != 1:
        raise CampaignError(
            f"Expected revisions of one campaign, found {', '.join(campaigns) or 'none'}"
        )
    roots = [revision_id for revision_id, revision in revisions.items() if revision.parent is None]
    if len(roots) != 1:
        raise CampaignError(f"Expected one root revision, found {', '.join(roots)}")
    root = revisions[roots[0]]
    if root.changes:
        raise CampaignError(f"Root revision {roots[0]} must not list changes")
    starts = set()
    continued = set()
    for revision in revisions.values():
        parent = revisions.get(revision.parent)
        if parent is not None and parent.branch == revision.branch:
            if revision.parent in continued:
                raise CampaignError(
                    f"Revision {revision.parent} already has a child on branch "
                    f"{revision.branch!r}; revise that branch's head or start a new branch"
                )
            continued.add(revision.parent)
        elif revision.branch in starts:
            raise CampaignError(f"Branch {revision.branch!r} already exists; choose a new name")
        else:
            starts.add(revision.branch)
    # Observation is campaign-wide, so an ID names one record and a revision also carries what
    # other branches recorded earlier.
    records = {}
    for revision_id, revision in revisions.items():
        for kind in ("trials", "exposures"):
            for item in getattr(revision, kind):
                key = _observation_id(item)
                known, by = records.setdefault((kind, key), (item, revision_id))
                if known != item:
                    raise CampaignError(
                        f"Revisions {by} and {revision_id} record different {kind} under ID {key}"
                    )
    earlier = {"trials": {}, "exposures": {}}
    by_time = sorted(revisions.items(), key=lambda entry: entry[1].created_at)
    for _, group in itertools.groupby(by_time, key=lambda entry: entry[1].created_at):
        group = tuple(group)
        for revision_id, revision in group:
            for kind, recorded in earlier.items():
                observed = _observed(revision, kind)
                missing = {key: by for key, by in recorded.items() if key not in observed}
                ids = ", ".join(sorted(missing))
                if new in missing.values():
                    latest = max(
                        other.created_at
                        for other in revisions.values()
                        if any(
                            not _observed(revisions[new], k) <= _observed(other, k) for k in earlier
                        )
                    )
                    raise CampaignError(
                        f"Revision {new} records {kind} {ids}, which the later revision "
                        f"{revision_id} omits; date it no earlier than {format_time(latest)}"
                    )
                if missing:
                    raise CampaignError(
                        f"Revision {revision_id} omits {kind} recorded earlier in the campaign: "
                        f"{ids}; include them"
                    )
        for revision_id, revision in group:
            for kind, recorded in earlier.items():
                for key in _observed(revision, kind):
                    recorded.setdefault(key, revision_id)
    # IDs cover parent IDs, so links cannot cycle and walking from the root reaches every revision.
    ordered = []
    ready = [(root.created_at, roots[0])]
    while ready:
        _, revision_id = heapq.heappop(ready)
        ordered.append(revisions[revision_id])
        for child_id in children[revision_id]:
            heapq.heappush(ready, (revisions[child_id].created_at, child_id))
    return Campaign(root.campaign_id, tuple(ordered))


def start_campaign(
    specification: CampaignSpecification,
    *,
    author: str,
    at: datetime,
    reason: str,
    branch: str = "main",
    trials: tuple[TrialReference, ...] = (),
    exposures: tuple[ExposureEvent, ...] = (),
) -> CampaignRevision:
    """Return the root revision of a new campaign.

    Args:
        specification (CampaignSpecification): Initial methodology.
        author (str): Person who records the revision.
        at (datetime): Timezone-aware creation time.
        reason (str): Decision the revision records.
        branch (str, optional): Name of the first branch.
        trials (tuple, optional): Trials observed before the campaign starts.
        exposures (tuple, optional): Exposure events observed before the campaign starts.

    Returns:
        (CampaignRevision) The validated root revision under a new campaign ID.
    """
    try:
        return parse_file(
            file_bytes(
                CampaignRevision(
                    uuid4(), branch, None, author, at, reason, specification, trials, exposures, ()
                )
            )
        )
    except (OSError, TypeError, ValueError, RecursionError) as error:
        raise CampaignError(f"Cannot start campaign: {error}") from error


def revise(
    parent: CampaignRevision,
    specification: CampaignSpecification,
    *,
    author: str,
    at: datetime,
    reason: str,
    branch: str | None = None,
    trials: tuple[TrialReference, ...] = (),
    exposures: tuple[ExposureEvent, ...] = (),
) -> CampaignRevision:
    """Return a child revision that keeps every trial and exposure of its parent.

    Args:
        parent (CampaignRevision): Revision to continue or branch from, left unchanged.
        specification (CampaignSpecification): Complete methodology of the child.
        author (str): Person who records the revision.
        at (datetime): Timezone-aware creation time, no earlier than the parent's.
        reason (str): Decision the revision records.
        branch (str | None, optional): New branch name, or None to continue the parent's.
        trials (tuple, optional): Trials observed since the parent; recorded ones are kept once.
        exposures (tuple, optional): Exposure events observed since the parent; recorded ones
            are kept once.

    Returns:
        (CampaignRevision) The validated child, listing the sections that differ from its parent.
    """
    try:
        if branch == parent.branch:
            raise CampaignError(
                f"Branch {branch!r} is already the parent's branch; omit branch to continue it"
            )
        parsed = parse_file(
            file_bytes(
                CampaignRevision(
                    parent.campaign_id,
                    parent.branch if branch is None else branch,
                    parent.revision_id,
                    author,
                    at,
                    reason,
                    specification,
                    _extend(parent.trials, trials, TrialReference, "trial_id", "Trial"),
                    _extend(parent.exposures, exposures, ExposureEvent, "id", "Exposure"),
                    (),
                )
            )
        )
        # Compare parsed specifications so a list supplied for a tuple is not a change.
        child = replace(parsed, changes=_changes(parent.specification, parsed.specification))
        check_link(parent, child)
        return child
    except (OSError, TypeError, ValueError, RecursionError) as error:
        raise CampaignError(f"Cannot revise campaign: {error}") from error


def save_revision(destination: Path, revision: CampaignRevision) -> Path:
    """Write a revision file after validating the campaign history it joins.

    Args:
        destination (Path): Parent directory of campaign directories.
        revision (CampaignRevision): Revision to store in its campaign's directory.

    Returns:
        (Path) The new revision file, or the identical file already saved.
    """
    try:
        content = file_bytes(revision)
        parsed = parse_file(content)
        directory = destination / str(parsed.campaign_id)
        final = directory / f"{hashlib.sha256(content).hexdigest()}.json"
        stored = _revisions(directory) if directory.exists() else {}
        _history(stored | {final.stem: parsed}, final.stem)
        if final.stem not in stored:
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".campaign-", dir=destination) as stage:
                staged = Path(stage, final.name)
                staged.write_bytes(content)
                staged.rename(final)
        return final
    except (OSError, TypeError, ValueError, RecursionError) as error:
        raise CampaignError(f"Cannot save campaign revision: {error}") from error


def open_campaign(path: Path) -> Campaign:
    """Verify and open the complete revision history in a campaign directory.

    Args:
        path (Path): Campaign directory, which may have been copied or renamed.

    Returns:
        (Campaign) Every revision, parents first, with a head for each branch.
    """
    try:
        return _history(_revisions(path))
    except (OSError, ValueError, RecursionError) as error:
        raise CampaignError(f"Cannot open campaign {path}: {error}") from error


def register(
    store: Path, *, name: str, role: str, data: bytes, declaration: SourceDeclaration
) -> Source:
    """Register source bytes, publish their text generation, and return the Source that pins it.

    Args:
        store (Path): Existing source store directory.
        name (str): Source name in the campaign.
        role (str): Source role, such as PROMPT for a model role's instructions.
        data (bytes): Original source bytes.
        declaration (SourceDeclaration): Provenance, restrictions, and availability of the bytes.

    Returns:
        (Source) The source, located by the id of its published TEXT_PIPELINE generation and
            pinned by that generation's manifest digest, equal on every repeat.

    Raises:
        SourceError: If the bytes cannot be registered or published.
    """
    occurrence = register_source(store, data, declaration)
    generation = publish_generation(store, occurrence.id, TEXT_PIPELINE)
    return Source(name, generation.id, generation.digest, role)


def contract_sources(store: Path, catalog: Catalog, datasets: Iterable[str]) -> tuple[Source, ...]:
    """Publish the contract of each named dataset and return one contract source per part.

    Pass the names a specification's data section permits, so its sources pin exactly the
    permitted contracts. A steer that changes the permitted datasets runs this again.

    Args:
        store (Path): Existing source store directory.
        catalog (Catalog): Catalog that documents each dataset.
        datasets (Iterable[str]): Permitted dataset names, in any order and with repeats.

    Returns:
        (tuple) Sources named <dataset>.contract.<part>, in name and part order.

    Raises:
        DatasetError: If a name is not a catalog entry, or its contract file is malformed.
        SourceError: If a part cannot be registered or published.
    """
    return tuple(
        Source(f"{name}.contract.{index}", generation.id, generation.digest, CONTRACT)
        for name in sorted(set(datasets))
        for index, generation in enumerate(publish_contract(store, catalog, name))
    )


@dataclass(frozen=True, kw_only=True)
class Setup:
    """Operator configuration that, with a campaign revision, fixes a run's authority.

    Nothing here comes from model output. sources holds the revision's source generations and
    the corpus resolve publishes, and datasets the dataset files. policies holds the operator's
    terms for each provider it routes, so build the Gateway from the same tuple. generated is
    used only when the revision enables generated code. rules is a directory of frozen rules
    files named <sha256>.json, the package's own by default. packages is where each campaign's
    lineage packages are saved, shared by every worker of a queue, and lock is the environment
    lock each trial records.
    """

    sources: Path
    datasets: Path
    catalog: Catalog
    ledger: Ledger
    clearances: frozenset[RestrictionLabel]
    cutoff: datetime
    policies: tuple[ProviderPolicy, ...]
    rules: Path = Path(str(resources.files("quant_research").joinpath("rules")))
    packages: Path
    lock: bytes
    memory: MemoryPin = NO_MEMORY
    generated: Generated | None = None


@dataclass(frozen=True)
class Authority:
    """What one campaign revision lets a run read, search, and send.

    generations pins every resolved revision source, including prompts outside the corpus.
    context.scope is the dataset names data permits, context.generation is corpus.id, and
    context.providers is the providers of the active roles' models, each with its policy in
    policies. prompts maps each active role to its instructions. context keeps the setup's
    cutoff: model sends, the prompt check, the steering floor, and the final evaluation use it.
    search.context is context narrowed once to the search cutoff of the revision's rules, and
    toolset is bound to it, so every tool read and every experiment ends at the search window.
    """

    revision: str
    data: tuple[DataPermission, ...]
    context: RunContext
    corpus: Corpus
    generations: Mapping[str, str]
    roles: RoleVersion
    policies: tuple[ProviderPolicy, ...]
    prompts: Mapping[str, Material]
    toolset: Toolset
    search: reference.Search

    def record(self) -> bytes:
        """Return canonical UTF-8 JSON naming what the authority pins, each prompt by digest."""
        return canonical_bytes(
            {
                "schema_version": 2 if self.context.dataset_periods is None else 3,
                **(
                    {"dataset_periods": [item.record() for item in self.context.dataset_periods]}
                    if self.context.dataset_periods is not None
                    else {}
                ),
                "revision": self.revision,
                "catalog": self.toolset.catalog.digest,
                "context": self.context.identity,
                "search": self.search.context.identity,
                "rules": self.search.rules.digest,
                "scope": sorted(self.context.scope),
                "data": [
                    {
                        "dataset_id": permission.dataset_id,
                        "interval_start": format_time(permission.interval_start),
                        "interval_end": format_time(permission.interval_end),
                    }
                    for permission in self.data
                ],
                "corpus": {"id": self.corpus.id, "generations": dict(self.corpus.generations)},
                "roles": {
                    "version": self.roles.version_id,
                    "activation": self.roles.activation,
                    "assignments": [
                        {
                            "role": assignment.role,
                            "models": [
                                f"{model.provider}/{model.model}"
                                for model in (assignment.primary, *assignment.fallbacks)
                            ],
                        }
                        for assignment in self.roles.assignments
                    ],
                },
                "policies": [
                    {"provider": policy.provider, "version": policy.version}
                    for policy in self.policies
                ],
                "prompts": {
                    role: hashlib.sha256(prompt.text.encode()).hexdigest()
                    for role, prompt in self.prompts.items()
                },
                "generated": None
                if self.toolset.generated is None
                else asdict(self.toolset.generated.limits),
            }
        )


_ROLES = "The campaign revision's active model roles"


def resolve(revision: CampaignRevision, setup: Setup) -> Authority:
    """Resolve a frozen campaign revision into its run authority.

    Pure apart from the write-once publication of its corpus, so resolving the same inputs again
    returns an equal authority, and resolving a steered child returns the child's. Nothing is
    cached. No check here asks whether each corpus member or dataset may go to each active
    provider; the gateway refuses at send.

    Args:
        revision (CampaignRevision): Revision whose data, sources, and model roles to resolve.
        setup (Setup): Stores, catalog, review log, clearances, cutoff, policies, and memory pin.

    Returns:
        (Authority) The scoped run context, corpus, active role version, policies, prompts,
            the search that the revision's rules fix, and the Toolset bound to its context.

    Raises:
        RulesError: If no rules file in setup.rules is named by the revision's evaluation digest,
            the file's bytes hash to another digest or load_rules refuses them, the revision's
            evaluation profile is not the windows' procedure, or the setup's cutoff is before
            the held-back window's.
        CampaignError: If an active role names an unknown model or a provider that has not
            passed a live check, an active provider has no policy or two, a permitted dataset
            is not in the catalog, the revision enables generated code and the setup has no
            Generated, a source's locator names no published generation or its digest is not
            that generation's manifest digest, the corpus cannot be published, a source at a
            contract location does not name its part, a source is the contract of a dataset the
            run cannot read under the restrictions the contract carries, or an active role has
            no prompt the run can read or one whose extracted text does not match its
            generation's text digest.
    """
    spec = revision.specification
    try:
        rules = reference.rules(setup.rules, spec.evaluation.digest)
    except reference.ReferencePipelineError as error:
        raise RulesError(f"No frozen rules file for {spec.evaluation.digest}") from error
    if spec.evaluation.profile != rules.search.profile.version:
        raise RulesError(
            f"The revision's evaluation profile {spec.evaluation.profile!r} is not its rules' "
            f"procedure {rules.search.profile.version!r}"
        )
    # The final evaluation reads the held-back window under context, so a setup that cannot
    # reach it would spend the campaign's trials on a method it cannot finish.
    if setup.cutoff < rules.held_back.cutoff:
        raise RulesError("The setup's cutoff is before the held-back window's cutoff")
    active = [role for role in spec.model_roles if role.status == "active"]
    named = {model for role in active for model in (role.model, *role.fallbacks)}
    if unknown := sorted(named - MODELS.keys()):
        raise CampaignError(f"Unknown models {unknown}")
    assignments = tuple(
        RoleAssignment(
            role.role, MODELS[role.model], tuple(MODELS[item] for item in role.fallbacks)
        )
        for role in active
    )
    try:
        roles = RoleVersion(
            str(revision.campaign_id),
            None,
            assignments,
            _ROLES,
            "active" if assignments else "inactive",
        )
    except ValueError as error:
        raise CampaignError(f"Cannot activate the model roles: {error}") from error
    providers = sorted(
        {
            model.provider
            for assignment in assignments
            for model in (assignment.primary, *assignment.fallbacks)
        }
    )
    unique((policy.provider for policy in setup.policies), "provider policy")
    terms = {policy.provider: policy for policy in setup.policies}
    if missing := [provider for provider in providers if provider not in terms]:
        raise CampaignError(f"No provider policy for {missing}")
    names = sorted({permission.dataset_id for permission in spec.data})
    catalogued = {entry.name for entry in setup.catalog.search()}
    if uncatalogued := [name for name in names if name not in catalogued]:
        raise CampaignError(f"Permitted datasets {uncatalogued} are not in the catalog")
    if spec.generated_code and setup.generated is None:
        raise CampaignError("The revision enables generated code, but the setup has no Generated")
    generations = {}
    for source in spec.sources:
        try:
            generation = open_generation(setup.sources, parse_digest(source.locator))
        except (OSError, ValueError) as error:
            raise CampaignError(f"Source {source.name!r} names no published generation") from error
        if generation.digest != source.digest:
            raise CampaignError(
                f"Source {source.name!r} pins manifest digest {source.digest}, but its "
                f"generation's manifest digest is {generation.digest}"
            )
        generations[source.name] = generation
    members = [source for source in spec.sources if source.role != PROMPT]
    mappings = {}
    for generation in generations.values():
        try:
            mapping = period_contract(generation)
        except DatasetError as error:
            raise CampaignError("A pinned source has an invalid dataset period mapping") from error
        if mapping is not None:
            if mapping.dataset in mappings and mappings[mapping.dataset] != mapping:
                raise CampaignError(f"Competing period mappings for {mapping.dataset!r}")
            mappings[mapping.dataset] = mapping
    periods = None
    if mappings:
        if extra := sorted(set(mappings) - set(names)):
            raise CampaignError(
                f"A pinned source is the contract of {extra[0]!r}, which the run does not permit"
            )
        if missing := sorted(set(names) - set(mappings)):
            raise CampaignError(f"No period mapping for permitted datasets {missing}")
        periods = tuple(
            DatasetPeriods(
                name,
                mappings[name].generation,
                mappings[name].manifest,
                mappings[name].mapping,
                tuple(
                    Interval(item.interval_start, item.interval_end)
                    for item in spec.data
                    if item.dataset_id == name
                ),
            )
            for name in names
        )
    try:
        corpus = publish_corpus(setup.sources, [source.locator for source in members])
    except RetrievalError as error:
        raise CampaignError(f"Cannot publish the revision's corpus: {error}") from error
    context = issue_context(
        clearances=setup.clearances,
        cutoff=setup.cutoff,
        generation=corpus.id,
        providers=providers,
        memory=setup.memory,
        scope=names,
        dataset_periods=periods,
    )
    readable = {
        item.dataset: item.restrictions for item in listing(setup.catalog, context).datasets
    }
    for name, generation in generations.items():
        try:
            dataset = contract_of(generation)
        except DatasetError as error:
            raise CampaignError(
                f"Source {name!r} is at a contract location, but its revision names no part"
            ) from error
        if dataset is None:
            continue
        current = readable.get(dataset)
        declared = generation.occurrence.declaration.restrictions
        if current is None or not (
            current.labels <= declared.labels and declared.approved_for <= current.approved_for
        ):
            raise CampaignError(
                f"Source {name!r} is the contract of {dataset!r}, which this run cannot read "
                "under the restrictions the contract carries"
            )
    instructions = {
        source.name: generations[source.name] for source in spec.sources if source.role == PROMPT
    }
    prompts = {}
    for assignment in assignments:
        generation = instructions.get(assignment.role)
        if generation is None:
            raise CampaignError(f"Active role {assignment.role!r} has no prompt source")
        restrictions = generation.occurrence.declaration.restrictions
        if denials := context.read_denials(restrictions):
            raise CampaignError(
                f"The run cannot read the prompt of role {assignment.role!r}: {list(denials)}"
            )
        text = extracted_text(setup.sources, generation.occurrence, TEXT_PIPELINE.extract)
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != generation.text_digest:
            raise CampaignError(
                f"The extracted prompt of role {assignment.role!r} does not match the text "
                "digest its generation pins"
            )
        prompts[assignment.role] = Material(text, restrictions)
    search = context.narrow(cutoff=rules.search.cutoff)
    return Authority(
        revision.revision_id,
        spec.data,
        context,
        corpus,
        MappingProxyType({generation.id: generation.digest for generation in generations.values()}),
        roles,
        tuple(terms[provider] for provider in providers),
        MappingProxyType(prompts),
        Toolset(
            search,
            setup.sources,
            setup.datasets,
            setup.catalog,
            setup.ledger,
            setup.generated if spec.generated_code else None,
        ),
        reference.Search(
            spec.question,
            revision.campaign_id,
            rules,
            setup.catalog,
            setup.datasets,
            search,
            setup.lock,
            setup.packages / str(revision.campaign_id),
        ),
    )
