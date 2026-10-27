from __future__ import annotations

import hashlib
import json
import re
import typing
from collections.abc import Collection, Iterable, Iterator, Mapping
from dataclasses import MISSING, dataclass, fields
from datetime import UTC, date, datetime, time
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Literal

from quant_research import datasets
from quant_research.access import RunContext
from quant_research.catalog import Catalog
from quant_research.datasets import ReadRecord, Refused
from quant_research.records import PackageError, canonical_bytes, parse_name, unique_keys

type Reason = Literal[
    "malformed_json",
    "unknown_operator",
    "missing_field",
    "extra_field",
    "wrong_type",
    "out_of_range",
    "negative_lag",
    "reads_past_vintage",
    "too_large",
    "unapproved_dataset",
    "unknown_field",
]
type Number = Series | Lag | TrailingMean | TrailingMin | Subtract | Ratio
type Condition = Threshold | Not | And | Or
type Program = Number | Condition
type History = Mapping[str, Mapping[date, Mapping[str, object]]]
type DatasetName = str
type _Kind = Literal["number", "condition", "dataset", "text", "count", "level"]

_MESSAGES: dict[Reason, str] = {
    "malformed_json": "The program is not valid JSON with unique keys",
    "unknown_operator": "No operator has this name",
    "missing_field": "An operator lacks a required field",
    "extra_field": "An operator has a field it does not declare",
    "wrong_type": "A program or field holds a value of another type",
    "out_of_range": "A parameter is outside its declared range",
    "negative_lag": "A lag reads an observation later than the one the decision reports",
    "reads_past_vintage": "A series reads a vintage released after the decision",
    "too_large": "The program takes more evaluations, reads, or nesting than the size limit",
    "unapproved_dataset": "A series names a dataset the catalog or the universe does not approve",
    "unknown_field": "A series names a field its dataset does not document",
}


class Read(str):
    """Name one adapter read per decision, a dataset as the decision vintage months earlier saw it.

    The text is the dataset name at vintage 0 and `<dataset>@<vintage>` otherwise, so a
    vintage-0 read keys a history exactly as the bare dataset name does. Series checks its dataset
    with records.parse_name when built, so no dataset holds '@', the text names one read, and
    dataset and vintage derive from it.
    """

    def __new__(cls, dataset: str, vintage: int = 0) -> Read:
        return super().__new__(cls, f"{dataset}@{vintage}" if vintage else dataset)

    @property
    def dataset(self) -> str:
        return self.partition("@")[0]

    @property
    def vintage(self) -> int:
        return int(self.partition("@")[2] or 0)


@dataclass(frozen=True)
class Bound:
    """A parameter's declared range, and the reason a value below low reports.

    A value must lie in [low, high] and be a whole multiple of step. Only the floor has its own
    reason, because both leak reasons are floors. A lag below 0 and a vintage below 0 leak.
    """

    low: int | Fraction
    high: int | Fraction
    below: Reason = "out_of_range"
    step: int | Fraction = 1

    def miss(self, value: int | Fraction) -> Reason | None:
        """Return the reason value falls outside this range, or None when it is inside."""
        if value < self.low:
            return self.below
        if value > self.high or value % self.step:
            return "out_of_range"
        return None


WINDOW = Bound(1, 24)
LAG = Bound(0, 24, below="negative_lag")
VINTAGE = Bound(0, 12, below="reads_past_vintage")
# Covers payrolls in thousands (~160,000) and ratios near 1 to four decimal places.
LEVEL = Bound(Fraction(-(10**6)), Fraction(10**6), step=Fraction(1, 10**4))
# Evaluations one month's value takes. Nodes do not bound work: a window multiplies its source.
MAX_COST = 256
# Each distinct read parses the whole stored file once per decision (D053).
MAX_READS = 4
# Operators on the longest path. parse stops at it before building, so untrusted nesting never
# recurses past it.
MAX_DEPTH = 32


class ProgramError(ValueError):
    """Report a program a researcher built in Python that breaks the grammar."""

    def __init__(self, reason: Reason) -> None:
        super().__init__(_MESSAGES[reason])
        self.reason = reason


@dataclass(frozen=True)
class Rejected:
    """A program text the grammar, the catalog, or the universe refuses, named by one reason."""

    reason: Reason

    @property
    def message(self) -> str:
        return _MESSAGES[self.reason]


def shift(month: date, months: int) -> date:
    """Return the first day of the month that lies months after month, or before when negative."""
    index = month.year * 12 + month.month - 1 + months
    return date(index // 12, index % 12 + 1, 1)


def _merged(*sources: Program) -> dict[Read, int]:
    reach: dict[Read, int] = {}
    for source in sources:
        for read, back in source.reach.items():
            reach[read] = max(reach.get(read, 0), back)
    return reach


class _Operator:
    """Check every operator against the grammar when it is built, from Python or from JSON.

    An operator reads what its children read and costs one evaluation plus theirs, unless it
    overrides reach or cost.
    """

    def __post_init__(self) -> None:
        _check(self)

    @property
    def reach(self) -> dict[Read, int]:
        """Return each read and how many months before the evaluated month it reaches."""
        return _merged(*_children(self))

    @property
    def cost(self) -> int:
        """Return the evaluations one month's value takes."""
        return 1 + sum(child.cost for child in _children(self))

    @property
    def depth(self) -> int:
        """Return the operators on the longest path from this one to a series."""
        return 1 + max((child.depth for child in _children(self)), default=0)


@dataclass(frozen=True)
class Series(_Operator):
    """A dataset field's value for the month, None where the dataset holds null.

    vintage k reads the vintage the decision k months earlier read, so a revision is
    Subtract(Series(d), Series(d, vintage=1)).
    """

    dataset: DatasetName
    field: str = "value"
    vintage: Annotated[int, VINTAGE] = 0

    @property
    def read(self) -> Read:
        return Read(self.dataset, self.vintage)

    @property
    def reach(self) -> dict[Read, int]:
        return {self.read: 0}

    def evaluate(self, history: History, month: date) -> Fraction | None:
        text = history[self.read][month][self.field]
        return None if text is None else Fraction(text)


@dataclass(frozen=True)
class Lag(_Operator):
    """The source's value months before the month, None when that is None."""

    source: Number
    months: Annotated[int, LAG]

    @property
    def reach(self) -> dict[Read, int]:
        return {read: back + self.months for read, back in self.source.reach.items()}

    def evaluate(self, history: History, month: date) -> Fraction | None:
        return self.source.evaluate(history, shift(month, -self.months))


@dataclass(frozen=True)
class TrailingMean(_Operator):
    """The mean of the source over months ending at the month, skipping null months.

    None when the source is None at the month itself.
    """

    source: Number
    months: Annotated[int, WINDOW]

    @property
    def reach(self) -> dict[Read, int]:
        return {read: back + self.months - 1 for read, back in self.source.reach.items()}

    @property
    def cost(self) -> int:
        return 1 + self.months * self.source.cost

    def evaluate(self, history: History, month: date) -> Fraction | None:
        values = [self.source.evaluate(history, shift(month, -back)) for back in range(self.months)]
        if values[0] is None:
            return None
        present = [value for value in values if value is not None]
        return sum(present, Fraction(0)) / len(present)


@dataclass(frozen=True)
class TrailingMin(_Operator):
    """The least source value over the months before the month, None when all are None."""

    source: Number
    months: Annotated[int, WINDOW]

    @property
    def reach(self) -> dict[Read, int]:
        return {read: back + self.months for read, back in self.source.reach.items()}

    @property
    def cost(self) -> int:
        return 1 + self.months * self.source.cost

    def evaluate(self, history: History, month: date) -> Fraction | None:
        values = (
            self.source.evaluate(history, shift(month, -back)) for back in range(1, self.months + 1)
        )
        return min((value for value in values if value is not None), default=None)


@dataclass(frozen=True)
class Subtract(_Operator):
    """The left value minus the right value, None when either is None."""

    left: Number
    right: Number

    def evaluate(self, history: History, month: date) -> Fraction | None:
        left, right = self.left.evaluate(history, month), self.right.evaluate(history, month)
        return None if left is None or right is None else left - right


@dataclass(frozen=True)
class Ratio(_Operator):
    """The numerator over the denominator, None when either is None or the denominator is 0."""

    numerator: Number
    denominator: Number

    def evaluate(self, history: History, month: date) -> Fraction | None:
        top = self.numerator.evaluate(history, month)
        bottom = self.denominator.evaluate(history, month)
        return None if top is None or not bottom else top / bottom


@dataclass(frozen=True)
class Threshold(_Operator):
    """Whether the source is at or above level, None when the source is None."""

    source: Number
    level: Annotated[Fraction, LEVEL]

    def evaluate(self, history: History, month: date) -> bool | None:
        value = self.source.evaluate(history, month)
        return None if value is None else value >= self.level


@dataclass(frozen=True)
class Not(_Operator):
    """Whether the condition is false, None when it is None. Not(Threshold) is strictly below."""

    condition: Condition

    def evaluate(self, history: History, month: date) -> bool | None:
        value = self.condition.evaluate(history, month)
        return None if value is None else not value


@dataclass(frozen=True)
class And(_Operator):
    """Whether both conditions hold, None when either is None."""

    left: Condition
    right: Condition

    def evaluate(self, history: History, month: date) -> bool | None:
        left, right = self.left.evaluate(history, month), self.right.evaluate(history, month)
        return None if left is None or right is None else left and right


@dataclass(frozen=True)
class Or(_Operator):
    """Whether either condition holds, None when either is None."""

    left: Condition
    right: Condition

    def evaluate(self, history: History, month: date) -> bool | None:
        left, right = self.left.evaluate(history, month), self.right.evaluate(history, month)
        return None if left is None or right is None else left or right


@dataclass(frozen=True)
class _Param:
    name: str
    kind: _Kind
    bound: Bound | None
    default: object


def _params(cls: type) -> tuple[_Param, ...]:
    """Return an operator's parameters, each with the kind and range its annotation declares.

    Raises:
        TypeError: If a field is a bare int or Fraction, or any type outside the grammar, so a
            new operator without a declared range fails at import.
    """
    hints = typing.get_type_hints(cls, include_extras=True)
    params = []
    for item in fields(cls):
        hint = hints[item.name]
        bound = None
        if hint is Number:
            kind = "number"
        elif hint is Condition:
            kind = "condition"
        elif hint is DatasetName:
            kind = "dataset"
        elif hint is str:
            kind = "text"
        elif typing.get_origin(hint) is Annotated and hint.__origin__ in (int, Fraction):
            kind, bound = ("count" if hint.__origin__ is int else "level"), hint.__metadata__[0]
        else:
            raise TypeError(f"{cls.__name__}.{item.name} declares no kind and range")
        params.append(_Param(item.name, kind, bound, item.default))
    return tuple(params)


_NUMBERS = typing.get_args(Number.__value__)
_CONDITIONS = typing.get_args(Condition.__value__)
# Built after the last operator and before SAHM, whose constructors check against it.
_GRAMMAR = {cls: _params(cls) for cls in (*_NUMBERS, *_CONDITIONS)}
# The class name is the JSON tag, so the JSON form is the constructor's keyword form.
_OPERATORS = {cls.__name__: cls for cls in _GRAMMAR}
_TYPES: dict[_Kind, tuple[type, ...]] = {
    "number": _NUMBERS,
    "condition": _CONDITIONS,
    "dataset": (str,),
    "text": (str,),
    "count": (int,),
    "level": (Fraction,),
}


def _children(program: Program) -> Iterator[Program]:
    for param in _GRAMMAR[type(program)]:
        if param.kind in ("number", "condition"):
            yield getattr(program, param.name)


def _check(program: Program) -> None:
    """Raise when one operator breaks the grammar. Its children were checked when built.

    Raises:
        ProgramError: wrong_type when a field's value is not its kind (a bool is never a count,
            and a condition is never a number); the bound's reason when a count or level misses
            its range; unapproved_dataset when a dataset is not a logical name; too_large over
            MAX_COST evaluations, MAX_READS distinct reads, or MAX_DEPTH operators deep.
    """
    for param in _GRAMMAR[type(program)]:
        value = getattr(program, param.name)
        if not isinstance(value, _TYPES[param.kind]) or isinstance(value, bool):
            raise ProgramError("wrong_type")
        if param.bound is not None and (reason := param.bound.miss(value)):
            raise ProgramError(reason)
        if param.kind == "dataset":
            try:
                parse_name(value, "dataset")
            except PackageError:
                raise ProgramError("unapproved_dataset") from None
    if program.cost > MAX_COST or len(program.reach) > MAX_READS or program.depth > MAX_DEPTH:
        raise ProgramError("too_large")


SAHM = Subtract(
    TrailingMean(Series("alfred.unrate"), 3),
    TrailingMin(TrailingMean(Series("alfred.unrate"), 3), 12),
)
SAHM_RULE = Threshold(SAHM, Fraction(1, 2))


def _encode(program: Program) -> dict[str, object]:
    data: dict[str, object] = {"op": type(program).__name__}
    for param in _GRAMMAR[type(program)]:
        value = getattr(program, param.name)
        if value == param.default:
            continue
        if param.kind in ("number", "condition"):
            data[param.name] = _encode(value)
        elif param.kind == "level":
            data[param.name] = str(value)
        else:
            data[param.name] = value
    return data


def canonical(program: Program) -> bytes:
    """Return a program's canonical JSON, the bytes its digest names.

    Each operator is an object whose "op" is its class name and whose other keys are its fields.
    A field at its default is left out, so a later defaulted field leaves every earlier digest
    unchanged. Levels are `str(Fraction)` text, and keys sort as `records.canonical_bytes` sorts.

    Args:
        program (Program): Program to encode.

    Returns:
        (bytes) Sorted, compact UTF-8 JSON with a trailing newline.
    """
    return canonical_bytes(_encode(program))


def digest(program: Program) -> str:
    """Return the SHA-256 of a program's canonical JSON, which enters a trial's specification."""
    return hashlib.sha256(canonical(program)).hexdigest()


# Decimal text as publishers write it, or the `str(Fraction)` text canonical writes. The digit
# caps keep Fraction below Python's integer text limit, far above any level LEVEL allows.
_LEVEL_TEXT = re.compile(r"-?[0-9]{1,16}(\.[0-9]{1,16}|/[1-9][0-9]{0,15})?")


def _constant(text: str) -> object:
    raise ValueError(f"{text} is not JSON")


def _build(raw: object, depth: int) -> Program:
    """Return the operator one JSON object names, building its children first.

    Raises:
        ProgramError: The first reason, in order: too_large past MAX_DEPTH, wrong_type for a
            non-object, missing_field for no "op", unknown_operator, missing_field, extra_field,
            wrong_type for level text, then whatever a child or the constructor raises.
    """
    if depth > MAX_DEPTH:
        raise ProgramError("too_large")
    if not isinstance(raw, dict):
        raise ProgramError("wrong_type")
    if "op" not in raw:
        raise ProgramError("missing_field")
    op = raw["op"]
    cls = _OPERATORS.get(op) if isinstance(op, str) else None
    if cls is None:
        raise ProgramError("unknown_operator")
    params = _GRAMMAR[cls]
    if any(param.default is MISSING and param.name not in raw for param in params):
        raise ProgramError("missing_field")
    if raw.keys() - {param.name for param in params} - {"op"}:
        raise ProgramError("extra_field")
    arguments = {}
    for param in params:
        if param.name not in raw:
            continue
        value = raw[param.name]
        if param.kind in ("number", "condition"):
            value = _build(value, depth + 1)
        elif param.kind == "level":
            if not isinstance(value, str) or not _LEVEL_TEXT.fullmatch(value):
                raise ProgramError("wrong_type")
            value = Fraction(value)
        arguments[param.name] = value
    return cls(**arguments)


def _leaves(program: Program) -> Iterator[Series]:
    if isinstance(program, Series):
        yield program
    for child in _children(program):
        yield from _leaves(child)


def parse(data: bytes, catalog: Catalog, universe: Collection[str]) -> Program | Rejected:
    """Return the program a JSON text names, or the first reason the grammar or catalog refuses it.

    This is the one boundary for untrusted programs. It decodes strictly, builds each operator
    through its constructor so a parsed program passes the same checks as one built in Python,
    and then requires every series, in pre-order, to name a dataset the universe holds and the
    catalog approves, and a field its documentation lists. A level may be decimal text or
    fraction text, so "0.5" and "1/2" give one program and one digest.

    Args:
        data (bytes): The program's own JSON document.
        catalog (Catalog): Catalog whose current state must approve every dataset the program
            reads.
        universe (Collection[str]): Datasets a search may read, such as
            public_data.SEARCH_UNIVERSE.

    Returns:
        (Program | Rejected) The program, or the rejection with its reason.
    """
    try:
        raw = json.loads(data, object_pairs_hook=unique_keys, parse_constant=_constant)
    except ValueError:
        return Rejected("malformed_json")
    except RecursionError:
        return Rejected("too_large")
    try:
        program = _build(raw, 1)
    except ProgramError as error:
        return Rejected(error.reason)
    approved = {
        entry.name: entry for entry in catalog.search(status="approved") if entry.name in universe
    }
    for series in _leaves(program):
        entry = approved.get(series.dataset)
        if entry is None:
            return Rejected("unapproved_dataset")
        if series.field not in entry.documentation.fields:
            return Rejected("unknown_field")
    return program


@dataclass(frozen=True)
class Decision:
    """What a program observed for one decision month from the vintages available before it.

    observation is the latest computable month, and released_at is when the newest vintage read
    was released: the latest `available_at` among the supplied rows. All three are None when
    nothing is computable. reads holds one record per read the program reaches, sorted by read
    text, so a vintage-1 read follows the vintage-0 read of the same dataset.
    """

    month: date
    observation: date | None
    value: Fraction | bool | None
    released_at: datetime | None
    reads: tuple[ReadRecord, ...]


def observe(program: Program, history: History) -> tuple[date, Fraction | bool] | None:
    """Return the latest computable month of a history and the program's value there.

    A month is computable when every month within the program's reach is a key of each read's
    history, and the program's value there is not None. With reads at two vintages, that is the
    latest month both vintages hold. A month absent from a vintage is outside the series; a
    month present with a null value is skipped inside windows and is not computable itself, so
    a None at the latest month falls back to an earlier month.

    Args:
        program (Program): Program to evaluate.
        history (History): Read text -> {month: row values}, holding every read in reach.

    Returns:
        (tuple[date, Fraction | bool] | None) The month and value, or None when no month is
            computable.
    """
    reach = program.reach
    for month in sorted({key for months in history.values() for key in months}, reverse=True):
        if all(
            shift(month, -back) in history[read]
            for read, depth in reach.items()
            for back in range(depth + 1)
        ):
            value = program.evaluate(history, month)
            if value is not None:
                return month, value
    return None


def decide(
    program: Program,
    catalog: Catalog,
    store: Path,
    context: RunContext,
    months: Iterable[date],
) -> tuple[Decision, ...] | Refused:
    """Evaluate a program causally for each decision month.

    The decision for month m narrows the context to the start of m, the end of month m-1. Each
    read at vintage k narrows that decision's context again, to the start of shift(m, -k), the
    cutoff the decision k months earlier used, and reads through the dataset adapter. At k = 0
    that is P02's cutoff, so a program with no vintage above 0 reads exactly what P02 read.

    Args:
        program (Program): Program to evaluate.
        catalog (Catalog): Catalog that must approve every dataset the program reaches.
        store (Path): Dataset store directory.
        context (RunContext): Trusted run authority whose cutoff is at or after every month.
        months (Iterable[date]): Decision months, each the first day of its month.

    Returns:
        (tuple[Decision, ...] | Refused) One decision per month, or the first refused read.

    Raises:
        DatasetError: If a documented vintage is unreadable or malformed.
        AccessError: If a decision month starts after the context's cutoff.
    """
    reads = sorted(program.reach)
    decisions = []
    for month in months:
        decision = context.narrow(cutoff=datetime(month.year, month.month, 1, tzinfo=UTC))
        supplied = []
        for read in reads:
            start = datetime.combine(shift(month, -read.vintage), time(), UTC)
            result = datasets.read(catalog, store, read.dataset, decision.narrow(cutoff=start))
            if isinstance(result, Refused):
                return result
            supplied.append(result)
        history = {
            read: {date.fromisoformat(row.key): row.values for row in result.rows}
            for read, result in zip(reads, supplied, strict=True)
        }
        observed = observe(program, history)
        if observed is None:
            observation = value = released_at = None
        else:
            observation, value = observed
            released_at = max(row.available_at for result in supplied for row in result.rows)
        records = tuple(result.record for result in supplied)
        decisions.append(Decision(month, observation, value, released_at, records))
    return tuple(decisions)
