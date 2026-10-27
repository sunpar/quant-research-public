from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from uuid import uuid4

from quant_research.packages import ExposureEvent, Outcome
from quant_research.records import (
    PackageError,
    canonical_bytes,
    format_time,
    parse_digest,
    parse_text,
)


class ConfirmationError(ValueError):
    """Report a refused confirmation run."""


@dataclass(frozen=True)
class Period:
    dataset_id: str
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        parse_text(self.dataset_id, "dataset_id")
        format_time(self.start)
        format_time(self.end)
        if self.end < self.start:
            raise PackageError("Period ends before it starts")


@dataclass(frozen=True)
class FrozenShortlist:
    candidates: tuple[str, ...]
    preprocessing: str
    training: str
    portfolio_rules: str
    evaluation: str
    identity: str

    @property
    def manifest(self) -> bytes:
        """Return the canonical frozen items whose SHA-256 digest is the identity."""
        items = asdict(self)
        del items["identity"]
        return canonical_bytes(items)


@dataclass(frozen=True)
class NoConfirmationNeeded:
    shortlist: str


@dataclass(frozen=True)
class RestrictedConfirmationReport:
    """Hold restricted confirmation outcomes for authorized review and audit.

    Ordinary package exports and reusable memory must not include this report or its
    results. Append only its exposure, which carries no outcomes, to the lineage history.
    """

    shortlist: str
    period: Period
    exposure: ExposureEvent
    results: tuple[tuple[str, Outcome], ...] = field(repr=False)


def _touching(history: tuple[ExposureEvent, ...], period: Period) -> list[ExposureEvent]:
    return [
        event
        for event in history
        if event.dataset_id == period.dataset_id
        and event.interval_start <= period.end
        and period.start <= event.interval_end
    ]


def eligible_periods(
    history: tuple[ExposureEvent, ...], periods: tuple[Period, ...]
) -> tuple[Period, ...]:
    """Return the periods that no supplied exposure touches.

    Eligibility depends only on exposure history, so a new campaign or protocol revision
    cannot restore an observed period.

    Args:
        history (tuple): Supplied exposure history of the research lineage.
        periods (tuple): Candidate confirmation periods.

    Returns:
        (tuple) Untouched periods in their supplied order.
    """
    return tuple(period for period in periods if not _touching(history, period))


def freeze(
    candidates: tuple[str, ...],
    preprocessing: str,
    training: str,
    portfolio_rules: str,
    evaluation: str,
) -> FrozenShortlist:
    """Freeze a shortlist and its procedure references under a content identity.

    Args:
        candidates (tuple): Distinct candidate specification IDs, possibly none.
        preprocessing (str): Preprocessing artifact digest.
        training (str): Training artifact digest.
        portfolio_rules (str): Portfolio rules artifact digest.
        evaluation (str): Evaluation procedure artifact digest.

    Returns:
        (FrozenShortlist) The shortlist identified by the SHA-256 digest of its manifest.
    """
    for digest in (*candidates, preprocessing, training, portfolio_rules, evaluation):
        parse_digest(digest)
    if len(set(candidates)) != len(candidates):
        raise PackageError("Duplicate shortlist candidates")
    shortlist = FrozenShortlist(
        candidates, preprocessing, training, portfolio_rules, evaluation, identity=""
    )
    return replace(shortlist, identity=hashlib.sha256(shortlist.manifest).hexdigest())


def run_confirmation(
    shortlist: FrozenShortlist,
    period: Period | None,
    history: tuple[ExposureEvent, ...],
    evaluate: Callable[[FrozenShortlist, Period], Mapping[str, Outcome]],
    actor: str,
    at: datetime,
) -> NoConfirmationNeeded | RestrictedConfirmationReport:
    """Run the frozen evaluation once on a confirmation period no exposure has touched.

    Args:
        shortlist (FrozenShortlist): Shortlist whose items must still match its identity.
        period (Period | None): Confirmation period to evaluate, or None when none exists.
        history (tuple): Supplied exposure history of the research lineage.
        evaluate (Callable): Frozen procedure that receives the shortlist and the approved
            period and returns one outcome per candidate.
        actor (str): Person or agent who observes the confirmation results.
        at (datetime): Time of that observation.

    Returns:
        (NoConfirmationNeeded | RestrictedConfirmationReport) No run for an empty shortlist,
            otherwise the restricted outcomes and the exposure to append to the history.

    Raises:
        ConfirmationError: If the shortlist changed after freezing, no period exists, an
            exposure touches the period, or the outcomes do not cover exactly the candidates.
        PackageError: If the actor or time cannot be recorded as an exposure.
    """
    if hashlib.sha256(shortlist.manifest).hexdigest() != shortlist.identity:
        raise ConfirmationError("Shortlist changed after freezing")
    if not shortlist.candidates:
        return NoConfirmationNeeded(shortlist.identity)
    if period is None:
        raise ConfirmationError("Confirmation needs a period")
    if blocking := _touching(history, period):
        exposures = ", ".join(str(event.id) for event in blocking)
        raise ConfirmationError(f"Confirmation period already exposed by {exposures}")
    parse_text(actor, "actor")
    format_time(at)
    outcomes = evaluate(shortlist, period)
    if set(outcomes) != set(shortlist.candidates):
        raise ConfirmationError("Confirmation needs exactly one outcome per candidate")
    return RestrictedConfirmationReport(
        shortlist.identity,
        period,
        ExposureEvent(
            uuid4(),
            actor,
            at,
            shortlist.identity,
            period.dataset_id,
            period.start,
            period.end,
            "result",
            "Observed final confirmation results",
        ),
        tuple((candidate, outcomes[candidate]) for candidate in shortlist.candidates),
    )
