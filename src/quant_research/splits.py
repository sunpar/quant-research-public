"""Split a strictly increasing index of decision times into expanding walk-forward folds.

Position p's features use observations through p, and its label reads observations through
p + horizon. Rows and observations are sequences aligned to the index by position, and every
window is a range of positions. Evaluation windows of `length` positions tile forward from
`start`, and training stops `horizon + embargo` positions before each one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence, Set, Sized
from dataclasses import dataclass, fields, is_dataclass
from datetime import date
from itertools import pairwise

_MISSING = object()


class LeakageError(ValueError):
    """Report a fit or label computation that read outside its window."""


@dataclass(frozen=True)
class WalkForward:
    start: int
    length: int
    horizon: int
    embargo: int = 0

    def __post_init__(self) -> None:
        if (
            self.length < 1
            or self.horizon < 0
            or self.embargo < 0
            or self.start <= self.horizon + self.embargo
        ):
            raise ValueError(
                f"{self} needs length >= 1, horizon >= 0, embargo >= 0, "
                "and start > horizon + embargo"
            )


@dataclass(frozen=True)
class Fold:
    training: range
    evaluation: range
    horizon: int


class _Window[V](Mapping[int, V]):
    def __init__(self, values: Sequence[V], window: range) -> None:
        self._values = values
        self._window = window

    def __getitem__(self, position: int) -> V:
        if position not in self._window:
            raise LeakageError(f"Position {position} is outside the readable window {self._window}")
        return self._values[position]

    def __iter__(self) -> Iterator[int]:
        return iter(self._window)

    def __len__(self) -> int:
        return len(self._window)

    def __contains__(self, position: object) -> bool:
        return position in self._window

    def get[D](self, position: int, default: D = None) -> V | D:
        return self._values[position] if position in self._window else default


def _nan(label: object) -> bool:
    differs = label != label
    if isinstance(differs, Sized):
        raise LeakageError(
            f"{type(label).__name__} labels compare elementwise, so the check cannot match them"
        )
    return bool(differs)


def _same(early: object, full: object) -> bool:
    """Return whether two labels match, NaN matching NaN at any depth.

    Containers compare by structure, so a subclass's own equality does not count.
    """
    if isinstance(early, list | tuple) and isinstance(full, list | tuple):
        return (
            isinstance(early, list) == isinstance(full, list)
            and len(early) == len(full)
            and all(map(_same, early, full))
        )
    if isinstance(early, Mapping) and isinstance(full, Mapping):
        return early.keys() == full.keys() and all(_same(early[key], full[key]) for key in early)
    if isinstance(early, Set) and isinstance(full, Set):
        unmatched = list(full)
        for item in early:
            found = next(
                (index for index, other in enumerate(unmatched) if _same(item, other)), None
            )
            if found is None:
                return False
            del unmatched[found]
        return not unmatched
    if all(is_dataclass(label) and not isinstance(label, type) for label in (early, full)):
        return type(early) is type(full) and all(
            _same(getattr(early, field.name), getattr(full, field.name))
            for field in fields(early)
            if field.compare
        )
    if any(nan := (_nan(early), _nan(full))):
        return all(nan)
    differs = early != full
    # An elementwise result, as a numpy scalar against a list gives, means different shapes.
    return not (isinstance(differs, Sized) or differs)


def walk_forward(index: Sequence[date], config: WalkForward) -> tuple[Fold, ...]:
    """Return folds tiled forward from start whose evaluation labels the index can realize."""
    for earlier, later in pairwise(index):
        if later <= earlier:
            raise ValueError(f"Index must be strictly increasing, but {later} follows {earlier}")
    gap = config.horizon + config.embargo
    starts = range(config.start, len(index) - config.horizon - config.length + 1, config.length)
    return tuple(
        Fold(range(start - gap), range(start, start + config.length), config.horizon)
        for start in starts
    )


def fit[R, P](fold: Fold, rows: Sequence[R], step: Callable[[Mapping[int, R]], P]) -> P:
    """Return step's result on a read-only view of the training rows, keyed by index position."""
    return step(_Window(rows, fold.training))


def labels[V, L](
    folds: Sequence[Fold],
    observations: Sequence[V],
    labeler: Callable[[Mapping[int, V]], Mapping[int, L]],
) -> Mapping[int, L]:
    """Return the full-sample labels after checking no training label reads past its horizon."""
    full = labeler(_Window(observations, range(len(observations))))
    for fold in folds:
        cutoff = fold.training.stop + fold.horizon
        early = labeler(_Window(observations, range(cutoff)))
        for position in fold.training:
            if not _same(early.get(position, _MISSING), full.get(position, _MISSING)):
                raise LeakageError(
                    f"Label at position {position} changes when positions from {cutoff} are "
                    f"withheld, so it reads past its declared horizon {fold.horizon}"
                )
    return full
