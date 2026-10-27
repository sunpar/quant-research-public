from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import InitVar, dataclass, replace
from datetime import UTC, datetime
from typing import Literal, NewType

from quant_research.records import (
    PackageError,
    canonical_bytes,
    format_time,
    parse_array,
    parse_digest,
    parse_fields,
    parse_name,
    parse_time,
    unique_keys,
)

RestrictionLabel = NewType("RestrictionLabel", str)
Provider = NewType("Provider", str)
MemoryPin = NewType("MemoryPin", str)

CONFIRMATION = RestrictionLabel("confirmation")
INTERNAL_CODE = RestrictionLabel("internal_code")
# The head of the empty review log, so a run pinned here admits no memory.
NO_MEMORY = MemoryPin(hashlib.sha256(b"").hexdigest())
type PeriodMapping = Literal["iso-month-start/closed-label/v1"]
PERIOD_MAPPING: PeriodMapping = "iso-month-start/closed-label/v1"

type Reason = Literal[
    "uncleared", "after_cutoff", "provider_not_permitted", "provider_not_approved"
]

# Only issue_context and narrow pass this, so RunContext(...) and dataclasses.replace raise.
_ISSUER = object()


class AccessError(ValueError):
    """Report invalid restrictions or an attempt to create or widen a run context."""


def licensed(name: str) -> RestrictionLabel:
    return RestrictionLabel(f"licensed:{name}")


# Datetimes sharing one tzinfo compare by wall clock, which misorders a repeated hour.
def _utc(value: datetime, name: str) -> datetime:
    if value.utcoffset() is None:
        raise AccessError(f"{name} must include a timezone")
    return value.astimezone(UTC)


# A string would iterate as characters and silently change what is cleared or approved.
def frozen[T](values: Iterable[T], name: str) -> frozenset[T]:
    if isinstance(values, str):
        raise AccessError(f"{name} must be a collection, not a string")
    return frozenset(values)


@dataclass(frozen=True, kw_only=True)
class Restrictions:
    """What access control knows about one piece of material."""

    available_at: datetime
    labels: frozenset[RestrictionLabel]
    approved_for: frozenset[Provider] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "available_at", _utc(self.available_at, "available_at"))
        object.__setattr__(self, "labels", frozen(self.labels, "labels"))
        object.__setattr__(self, "approved_for", frozen(self.approved_for, "approved_for"))


def derive(first: Restrictions, *rest: Restrictions) -> Restrictions:
    """Return the restrictions of material derived from every input."""
    # Requires one input: the empty intersection of approvals would approve every provider.
    return Restrictions(
        available_at=max(item.available_at for item in (first, *rest)),
        labels=first.labels.union(*(item.labels for item in rest)),
        approved_for=first.approved_for.intersection(*(item.approved_for for item in rest)),
    )


@dataclass(frozen=True, order=True)
class Interval:
    start: datetime
    end: datetime


@dataclass(frozen=True)
class DatasetPeriods:
    """One pinned interpretation of a dataset's keys and its original interval union."""

    dataset: str
    generation: str
    manifest: str
    mapping: PeriodMapping
    intervals: tuple[Interval, ...]

    def record(self) -> dict:
        return {
            "dataset": self.dataset,
            "generation": self.generation,
            "manifest": self.manifest,
            "mapping": self.mapping,
            "intervals": [
                {"start": format_time(interval.start), "end": format_time(interval.end)}
                for interval in self.intervals
            ],
        }


@dataclass(frozen=True, kw_only=True)
class RunContext:
    """A run's trusted authority. Only issue_context creates one.

    scope names the only datasets the run may read. None leaves the run unscoped, the authority
    of operator commands outside any campaign.
    """

    clearances: frozenset[RestrictionLabel]
    cutoff: datetime
    generation: str
    providers: frozenset[Provider]
    memory: MemoryPin = NO_MEMORY
    scope: frozenset[str] | None = None
    dataset_periods: tuple[DatasetPeriods, ...] | None = None
    issuer: InitVar[object] = None

    def __post_init__(self, issuer: object) -> None:
        if issuer is not _ISSUER:
            raise AccessError("Create a run context with issue_context")
        object.__setattr__(self, "cutoff", _utc(self.cutoff, "cutoff"))
        if CONFIRMATION in self.clearances:
            raise AccessError("No run context may clear confirmation material")

    @property
    def identity(self) -> str:
        """Return the SHA-256 of the canonical authority, so equal authority has one identity."""
        authority = {
            "clearances": sorted(self.clearances),
            "cutoff": format_time(self.cutoff),
            "generation": self.generation,
            "providers": sorted(self.providers),
        }
        if self.memory != NO_MEMORY:
            authority["memory"] = self.memory
        if self.scope is not None:
            authority["scope"] = sorted(self.scope)
        if self.dataset_periods is not None:
            authority["dataset_periods"] = [item.record() for item in self.dataset_periods]
        return hashlib.sha256(canonical_bytes(authority)).hexdigest()

    def to_bytes(self) -> bytes:
        """Encode archived authority while preserving the historical absent-field form."""
        value = {
            "clearances": sorted(self.clearances),
            "cutoff": format_time(self.cutoff),
            "generation": self.generation,
            "providers": sorted(self.providers),
            "memory": self.memory,
            "scope": None if self.scope is None else sorted(self.scope),
        }
        if self.dataset_periods is not None:
            value["dataset_periods"] = [item.record() for item in self.dataset_periods]
        return canonical_bytes(value)

    @classmethod
    def from_bytes(cls, data: bytes) -> RunContext:
        """Restore authority from an archived context at the trusted input boundary."""
        try:
            raw = json.loads(data, object_pairs_hook=unique_keys)
            optional = " ".join(
                name for name in ("memory", "scope", "dataset_periods") if name in raw
            )
            value = parse_fields(
                raw, f"clearances cutoff generation providers {optional}", "context"
            )
            value["cutoff"] = parse_time(value["cutoff"])
            if "dataset_periods" in value:
                bindings = []
                for item in parse_array(value["dataset_periods"], "dataset_periods"):
                    item = parse_fields(
                        item, "dataset generation manifest mapping intervals", "dataset periods"
                    )
                    intervals = [
                        parse_fields(interval, "start end", "interval")
                        for interval in parse_array(item["intervals"], "intervals")
                    ]
                    bindings.append(
                        DatasetPeriods(
                            item["dataset"],
                            item["generation"],
                            item["manifest"],
                            item["mapping"],
                            tuple(
                                Interval(parse_time(interval["start"]), parse_time(interval["end"]))
                                for interval in intervals
                            ),
                        )
                    )
                value["dataset_periods"] = bindings
            return issue_context(**value)
        except (ValueError, TypeError, RecursionError) as error:
            raise AccessError("Invalid archived run context") from error

    def permits(self, dataset: str) -> bool:
        """Return whether the run's scope names the dataset. An unscoped run permits every one."""
        return self.scope is None or dataset in self.scope

    def read_denials(self, item: Restrictions) -> tuple[Reason, ...]:
        """Return every reason this run may not read the item; empty means allowed."""
        denials: list[Reason] = []
        if not item.labels <= self.clearances:
            denials.append("uncleared")
        if item.available_at > self.cutoff:
            denials.append("after_cutoff")
        return tuple(denials)

    def send_denials(self, item: Restrictions, provider: Provider) -> tuple[Reason, ...]:
        """Return the read denials, then every reason the item may not go to the provider."""
        denials = list(self.read_denials(item))
        if provider not in self.providers:
            denials.append("provider_not_permitted")
        if provider not in item.approved_for:
            denials.append("provider_not_approved")
        return tuple(denials)

    def narrow(self, *, cutoff: datetime) -> RunContext:
        """Return this context with an earlier or equal cutoff."""
        narrowed = replace(self, cutoff=cutoff, issuer=_ISSUER)
        if narrowed.cutoff > self.cutoff:
            raise AccessError("A narrowed cutoff cannot be after the issued cutoff")
        return narrowed


def issue_context(
    *,
    clearances: Iterable[RestrictionLabel],
    cutoff: datetime,
    generation: str,
    providers: Iterable[Provider],
    memory: MemoryPin = NO_MEMORY,
    scope: Iterable[str] | None = None,
    dataset_periods: Iterable[DatasetPeriods] | None = None,
) -> RunContext:
    """Create a run's authority from operator configuration, never from model output."""
    cleared, permitted = frozen(clearances, "clearances"), frozen(providers, "providers")
    scoped = None if scope is None else frozen(scope, "scope")
    if not all(
        isinstance(value, str)
        for value in (generation, memory, *cleared, *permitted, *(scoped or ()))
    ):
        raise AccessError(
            "A run context's generation, memory, clearances, providers, and scope must be text"
        )
    periods = None
    if dataset_periods is not None:
        normalized = []
        for item in dataset_periods:
            if item.mapping != PERIOD_MAPPING:
                raise AccessError("Unsupported dataset period mapping")
            intervals = tuple(
                sorted(
                    Interval(
                        _utc(interval.start, "interval start"), _utc(interval.end, "interval end")
                    )
                    for interval in item.intervals
                )
            )
            if not intervals or any(interval.end < interval.start for interval in intervals):
                raise AccessError("Dataset periods require nonempty ordered intervals")
            try:
                dataset = parse_name(item.dataset)
                period_generation = parse_digest(item.generation)
                manifest = parse_digest(item.manifest)
            except PackageError as error:
                raise AccessError(str(error)) from error
            normalized.append(
                DatasetPeriods(dataset, period_generation, manifest, item.mapping, intervals)
            )
        periods = tuple(sorted(normalized, key=lambda item: item.dataset))
        names = [item.dataset for item in periods]
        if scoped is None or set(names) != scoped or len(names) != len(set(names)):
            raise AccessError("Dataset periods must cover exactly the scoped datasets once")
    return RunContext(
        clearances=cleared,
        cutoff=cutoff,
        generation=generation,
        providers=permitted,
        memory=memory,
        scope=scoped,
        dataset_periods=periods,
        issuer=_ISSUER,
    )
