from __future__ import annotations

import math
from collections import namedtuple
from dataclasses import dataclass, field
from datetime import date

import pytest

from quant_research.splits import Fold, LeakageError, WalkForward, fit, labels, walk_forward

CONFIG = WalkForward(start=7, length=2, horizon=2, embargo=1)
DAYS = tuple(date(2026, 1, day) for day in range(1, 16))
SQUARES = tuple(position * position for position in range(15))


def mean(view):
    return sum(view.values()) / len(view)


def forward_change(view):
    return {p: view[p + 2] - view[p] for p in view if p + 2 in view}


def forward_change_by_get(view):
    later = {p: view.get(p + 2) for p in view}
    return {p: value - view[p] for p, value in later.items() if value is not None}


def forward_change_with_gap(view):
    # A new NaN per call, as computed NaN is, so containers cannot match it by identity.
    return {p: float("nan") if p == 1 else view[p + 2] - view[p] for p in view if p + 2 in view}


def three_step_change(view):
    return {p: view[p + 3] - view[p] for p in view if p + 3 in view}


def three_step_change_or_nan(view):
    return {p: view.get(p + 3, math.nan) - view[p] for p in view if p + 2 in view}


def demeaned_change(view):
    return {p: view[p + 2] - view[p] - mean(view) for p in view if p + 2 in view}


def peeking_change(view):
    return {p: view[p + 3 if p == 3 else p + 2] - view[p] for p in view if p + 2 in view}


def nested_three_step_change_or_nan(view):
    return {p: {"changes": (change,)} for p, change in three_step_change_or_nan(view).items()}


def elementwise_change(view):
    return {p: Elementwise(change) for p, change in forward_change(view).items()}


class Ambiguous(list):
    def __bool__(self):
        raise ValueError("The truth value of an array with more than one element is ambiguous")


class Elementwise:
    """Compare elementwise, as numpy arrays do."""

    def __init__(self, *values):
        self.values = values

    def __eq__(self, other):
        return Ambiguous(a == b for a, b in zip(self.values, other.values, strict=True))

    def __ne__(self, other):
        return Ambiguous(a != b for a, b in zip(self.values, other.values, strict=True))


class Scalar:
    """Compare as one value, but elementwise against a list, as numpy scalars do."""

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        if isinstance(other, list):
            return Ambiguous(self.value == item for item in other)
        return isinstance(other, Scalar) and self.value == other.value

    def __ne__(self, other):
        if isinstance(other, list):
            return Ambiguous(self.value != item for item in other)
        return not (isinstance(other, Scalar) and self.value == other.value)


@dataclass(frozen=True)
class Change:
    value: float


@dataclass(frozen=True)
class Weighted:
    value: float
    weights: list = field(hash=False)


def scalar_or_list_change(view):
    return {p: Scalar(c) if p + 3 not in view else [c] for p, c in forward_change(view).items()}


def nested_scalar_or_list_change(view):
    return {p: (label,) for p, label in scalar_or_list_change(view).items()}


def set_three_step_change_or_nan(view):
    return {p: {change} for p, change in three_step_change_or_nan(view).items()}


def dataclass_three_step_change_or_nan(view):
    return {p: Change(change) for p, change in three_step_change_or_nan(view).items()}


def nan_count_change(view):
    # Each float("nan") is a distinct set member, so the set grows by one when p + 3 is readable.
    return {
        p: {float("nan") for _ in range(2 if p + 3 in view else 1)} for p in forward_change(view)
    }


def masked_change(view):
    changes = forward_change(view)
    return {p: math.nan if change < mean(view) else change for p, change in changes.items()}


@pytest.mark.smoke
@pytest.mark.parametrize(
    "config, count, expected",
    [
        (CONFIG, 13, (Fold(range(0, 4), range(7, 9), 2), Fold(range(0, 6), range(9, 11), 2))),
        (
            CONFIG,
            15,
            (
                Fold(range(0, 4), range(7, 9), 2),
                Fold(range(0, 6), range(9, 11), 2),
                Fold(range(0, 8), range(11, 13), 2),
            ),
        ),
        (CONFIG, 10, ()),
        (
            WalkForward(start=3, length=3, horizon=1),
            10,
            (Fold(range(0, 2), range(3, 6), 1), Fold(range(0, 5), range(6, 9), 1)),
        ),
        (
            WalkForward(start=2, length=2, horizon=0),
            6,
            (Fold(range(0, 2), range(2, 4), 0), Fold(range(0, 4), range(4, 6), 0)),
        ),
    ],
    ids=["embargo", "embargo-appended", "embargo-too-short", "no-embargo", "no-horizon"],
)
def test_folds_end_every_training_label_before_evaluation(config, count, expected):
    folds = walk_forward(DAYS[:count], config)
    assert folds == expected
    for fold in folds:
        assert max(fold.training) + fold.horizon < fold.evaluation.start
        assert not set(fold.training) & set(fold.evaluation)


def test_appending_observations_only_adds_folds():
    short = walk_forward(DAYS[:13], CONFIG)
    extended = walk_forward(DAYS, CONFIG)
    assert extended[:2] == short
    assert fit(short[0], SQUARES[:13], mean) == fit(extended[0], SQUARES, mean) == 3.5
    before = labels(short, SQUARES[:13], forward_change)
    after = labels(extended, SQUARES, forward_change)
    positions = {p for fold in short for p in (*fold.training, *fold.evaluation)}
    assert {p: after[p] for p in positions} == {p: before[p] for p in positions}


@pytest.mark.smoke
def test_fit_sees_only_training_rows_keyed_by_position():
    fold = walk_forward(DAYS[:13], CONFIG)[0]
    items = fit(fold, SQUARES[:13], lambda view: tuple(view.items()))
    assert items == ((0, 0), (1, 1), (2, 4), (3, 9))
    membership = fit(
        fold, SQUARES[:13], lambda view: (len(view), 3 in view, 7 in view, view.get(7, "hidden"))
    )
    assert membership == (4, True, False, "hidden")


@pytest.mark.smoke
@pytest.mark.parametrize("position", [4, 5, 6, 7, 12, -1])
def test_fit_rejects_reads_outside_training(position):
    fold = walk_forward(DAYS[:13], CONFIG)[0]
    with pytest.raises(LeakageError, match=rf"Position {position} .*range\(0, 4\)"):
        fit(fold, SQUARES[:13], lambda view: view[position])


@pytest.mark.parametrize("labeler", [forward_change, forward_change_by_get])
def test_honest_labels_return_full_sample_values(labeler):
    expected = {0: 4, 1: 8, 2: 12, 3: 16, 4: 20, 5: 24, 6: 28, 7: 32, 8: 36, 9: 40, 10: 44}
    assert labels(walk_forward(DAYS[:13], CONFIG), SQUARES[:13], labeler) == expected


def test_nan_labels_that_recompute_as_nan_return_full_sample_values():
    folds = walk_forward(DAYS[:13], CONFIG)
    gapped = labels(folds, SQUARES[:13], forward_change_with_gap)
    honest = labels(folds, SQUARES[:13], forward_change)
    assert math.isnan(gapped[1])
    assert {**gapped, 1: honest[1]} == honest


@pytest.mark.parametrize(
    "wrap",
    [
        lambda change: (change, 0),
        lambda change: [change],
        lambda change: {"change": change},
        lambda change: ({"changes": [change]},),
        lambda change: namedtuple("Change", "value")(change),
        lambda change: {change},
        lambda change: frozenset({change, 0}),
        lambda change: Change(change),
        lambda change: {Change(change)},
        lambda change: {Weighted(change, [change])},
    ],
    ids=[
        "tuple",
        "list",
        "dict",
        "nested",
        "namedtuple-per-call",
        "set",
        "frozenset",
        "dataclass",
        "dataclass-in-set",
        "unhashable-field-in-set",
    ],
)
def test_container_labels_holding_nan_that_recompute_as_nan_return_full_sample_values(wrap):
    folds = walk_forward(DAYS[:13], CONFIG)
    gapped = labels(
        folds,
        SQUARES[:13],
        lambda view: {p: wrap(change) for p, change in forward_change_with_gap(view).items()},
    )
    honest = {p: wrap(change) for p, change in labels(folds, SQUARES[:13], forward_change).items()}
    assert {**gapped, 1: honest[1]} == honest


def test_labels_that_compare_elementwise_raise():
    with pytest.raises(LeakageError, match="Elementwise labels compare elementwise"):
        labels(walk_forward(DAYS[:13], CONFIG), SQUARES[:13], elementwise_change)


@pytest.mark.parametrize(
    "labeler, message",
    [
        (three_step_change, "position 3 .*from 6 .*horizon 2"),
        (demeaned_change, "position 0 .*from 6 .*horizon 2"),
        (peeking_change, r"Position 6 is outside .*range\(0, 6\)"),
        (three_step_change_or_nan, "position 3 .*from 6 .*horizon 2"),
        (nested_three_step_change_or_nan, "position 3 .*from 6 .*horizon 2"),
        (masked_change, "position 2 .*from 6 .*horizon 2"),
        (set_three_step_change_or_nan, "position 3 .*from 6 .*horizon 2"),
        (dataclass_three_step_change_or_nan, "position 3 .*from 6 .*horizon 2"),
        (nan_count_change, "position 3 .*from 6 .*horizon 2"),
        (scalar_or_list_change, "position 3 .*from 6 .*horizon 2"),
        (nested_scalar_or_list_change, "position 3 .*from 6 .*horizon 2"),
    ],
    ids=[
        "three-step",
        "full-sample-mean",
        "explicit-read",
        "nan-withheld",
        "nan-withheld-in-container",
        "nan-full-sample",
        "nan-withheld-in-set",
        "nan-withheld-in-dataclass",
        "nan-count-in-set",
        "scalar-becomes-list",
        "nested-scalar-becomes-list",
    ],
)
def test_labels_that_read_past_their_horizon_raise(labeler, message):
    with pytest.raises(LeakageError, match=message) as caught:
        labels(walk_forward(DAYS[:13], CONFIG), SQUARES[:13], labeler)
    assert caught.type is LeakageError


@pytest.mark.parametrize(
    "arguments",
    [
        {"start": 3, "length": 0, "horizon": 1},
        {"start": 3, "length": 1, "horizon": -1},
        {"start": 3, "length": 1, "horizon": 1, "embargo": -1},
        {"start": 3, "length": 1, "horizon": 2, "embargo": 1},
    ],
    ids=["empty-window", "negative-horizon", "negative-embargo", "empty-training"],
)
def test_invalid_configuration_raises(arguments):
    with pytest.raises(ValueError) as caught:
        WalkForward(**arguments)
    assert caught.type is ValueError


@pytest.mark.parametrize(
    "index",
    [(DAYS[0], DAYS[1], DAYS[1]), (DAYS[1], DAYS[0], DAYS[2])],
    ids=["duplicate", "reversed"],
)
def test_non_increasing_index_raises(index):
    with pytest.raises(ValueError) as caught:
        walk_forward(index, WalkForward(start=1, length=1, horizon=0))
    assert caught.type is ValueError
