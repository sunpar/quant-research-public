from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date
from enum import Enum
from types import MappingProxyType
from typing import Literal

from quant_research.records import (
    PackageError,
    canonical_bytes,
    parse_array,
    parse_fields,
    parse_text,
)

type CutoffStatus = Literal[
    "period_after_cutoff", "period_before_or_overlapping_cutoff", "cutoff_unknown"
]
type Activation = Literal["inactive", "active"]

ACCEPTED: frozenset[str] = frozenset({"codex-subscription", "claude-subscription"})


class Unknown(Enum):
    UNKNOWN = "unknown"


UNKNOWN = Unknown.UNKNOWN


@dataclass(frozen=True)
class ModelRecord:
    """Identify one provider model and the knowledge cutoff its provider published.

    knowledge_cutoff is the last date the model's training information may include. Record a
    month-level statement as that month's last day, and record UNKNOWN when the provider
    published none.
    """

    provider: str
    model: str
    version: str | Unknown
    knowledge_cutoff: date | Unknown


MODELS: Mapping[str, ModelRecord] = MappingProxyType(
    {
        f"{record.provider}/{record.model}": record
        for record in (
            ModelRecord("codex-subscription", "gpt-6.1-sol", UNKNOWN, UNKNOWN),
            ModelRecord("claude-subscription", "claude-opus-5-5", UNKNOWN, UNKNOWN),
        )
    }
)


@dataclass(frozen=True)
class RoleAssignment:
    role: str
    primary: ModelRecord
    fallbacks: tuple[ModelRecord, ...] = ()


@dataclass(frozen=True)
class RoleVersion:
    """A snapshot of a campaign's model roles. An active version names only accepted providers."""

    campaign_id: str
    previous: str | None
    assignments: tuple[RoleAssignment, ...]
    reason: str
    activation: Activation = "inactive"

    def __post_init__(self) -> None:
        roles = [assignment.role for assignment in self.assignments]
        if len(set(roles)) != len(roles):
            raise ValueError("Duplicate role assignments")
        named = {
            model.provider
            for assignment in self.assignments
            for model in (assignment.primary, *assignment.fallbacks)
        }
        if self.activation == "active" and (unaccepted := sorted(named - ACCEPTED)):
            raise ValueError(
                f"Providers {unaccepted} have not passed a live check, so their roles cannot "
                "activate"
            )

    @property
    def version_id(self) -> str:
        record = json.loads(json.dumps(asdict(self), default=json_default, allow_nan=False))
        return hashlib.sha256(canonical_bytes(record)).hexdigest()

    def revise(self, assignments: tuple[RoleAssignment, ...], reason: str) -> RoleVersion:
        """Return the next version, linked to this one, with a complete replacement role set."""
        return RoleVersion(self.campaign_id, self.version_id, assignments, reason, self.activation)


@dataclass(frozen=True)
class CutoffCheck:
    """Classify one inclusive evaluation period against a model's knowledge cutoff."""

    model: ModelRecord
    start: date
    end: date

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError("Evaluation period ends before it starts")

    @property
    def status(self) -> CutoffStatus:
        cutoff = self.model.knowledge_cutoff
        if cutoff is UNKNOWN:
            return "cutoff_unknown"
        if self.start > cutoff:
            return "period_after_cutoff"
        return "period_before_or_overlapping_cutoff"

    @property
    def sentence(self) -> str:
        version = self.model.version
        return _SENTENCES[self.status].format(
            start=self.start,
            end=self.end,
            cutoff=self.model.knowledge_cutoff,
            version="an unknown version" if version is UNKNOWN else f"version {version}",
            model=self.model.model,
            provider=self.model.provider,
        )


_SENTENCES: dict[CutoffStatus, str] = {
    "period_after_cutoff": (
        "The period from {start} through {end} begins after {cutoff}, the recorded knowledge "
        "cutoff for {version} of {model} from {provider}."
    ),
    "period_before_or_overlapping_cutoff": (
        "The period from {start} through {end} begins on or before {cutoff}, the recorded "
        "knowledge cutoff for {version} of {model} from {provider}, so the model may already "
        "contain knowledge of the period even when the platform withheld its results, and the "
        "historical result is qualified confirmation, not untouched evidence."
    ),
    "cutoff_unknown": (
        "The knowledge cutoff for {version} of {model} from {provider} is unknown, so the model "
        "may already contain knowledge of the period from {start} through {end} even when the "
        "platform withheld its results, and the historical result is qualified confirmation, "
        "not untouched evidence."
    ),
}


def json_default(value: date | Unknown) -> str | None:
    """Encode a date as ISO 8601 text and UNKNOWN as null in a model record."""
    if isinstance(value, date):
        return value.isoformat()
    if value is UNKNOWN:
        return None
    raise TypeError(f"Unsupported record value {value!r}")


def parse_model_record(raw: object) -> ModelRecord:
    """Return the model record stored as plain JSON, raising PackageError for any invalid field."""
    value = parse_fields(raw, "provider model version knowledge_cutoff", "model record")
    return ModelRecord(
        parse_text(value["provider"], "provider"),
        parse_text(value["model"], "model"),
        UNKNOWN if value["version"] is None else parse_text(value["version"], "model version"),
        _cutoff(value["knowledge_cutoff"]),
    )


def dump_assignment(assignment: RoleAssignment) -> str:
    """Return the assignment as ASCII JSON, with UNKNOWN as null, apart from the text "unknown"."""
    return json.dumps(asdict(assignment), default=json_default, sort_keys=True)


def load_assignment(text: str) -> RoleAssignment:
    value = parse_fields(json.loads(text), "role primary fallbacks", "role assignment")
    return RoleAssignment(
        parse_text(value["role"], "role"),
        parse_model_record(value["primary"]),
        tuple(parse_model_record(raw) for raw in parse_array(value["fallbacks"], "fallbacks")),
    )


def _cutoff(value: object) -> date | Unknown:
    if value is None:
        return UNKNOWN
    try:
        result = date.fromisoformat(value)
    except (TypeError, ValueError):
        # Raising after the handler chains nothing, since the parse error quotes the value.
        failure = "Invalid knowledge cutoff"
    else:
        if result.isoformat() == value:
            return result
        failure = "Invalid canonical knowledge cutoff"
    raise PackageError(failure)
