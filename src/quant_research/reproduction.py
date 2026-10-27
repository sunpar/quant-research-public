from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Literal, get_args

from quant_research.access import RunContext
from quant_research.locators import Kind, LocatorGeneration, published_locations
from quant_research.records import PackageError, canonical_bytes, parse_digest
from quant_research.sources import SourceError

type Basis = Literal["matched_inputs", "adapted"]
type Reason = Literal[
    "within_tolerance", "outside_tolerance", "open_question", "missing", "non_finite"
]
type Outcome = Literal[
    "reproduced",
    "not_reproduced",
    "reproduced_on_adapted_data",
    "not_reproduced_on_adapted_data",
    "inconclusive",
]

_OUTCOMES: dict[Basis, dict[Reason, Outcome]] = {
    "matched_inputs": {"within_tolerance": "reproduced", "outside_tolerance": "not_reproduced"},
    "adapted": {
        "within_tolerance": "reproduced_on_adapted_data",
        "outside_tolerance": "not_reproduced_on_adapted_data",
    },
}

_LOCATED = get_args(Kind.__value__)


class ReproductionError(ValueError):
    """Report an incomplete or unresolved specification, a late revision, or a bad result."""


@dataclass(frozen=True)
class Locator:
    """Reference to a passage or code revision, opaque here, or to a located paper element."""

    kind: Literal["passage", "code"] | Kind
    ref: str


@dataclass(frozen=True)
class LocatorPin:
    """A published locator generation's id and digest; the digest covers every location's text."""

    generation: str
    digest: str


@dataclass(frozen=True)
class OpenQuestion:
    id: str
    kind: Literal["ambiguous", "contradictory", "missing", "restricted"]
    detail: str
    locators: tuple[Locator, ...] = ()


@dataclass(frozen=True)
class Choice:
    """An implementation choice. A value needs a locator; no value needs an open question."""

    name: str
    value: str | None
    locators: tuple[Locator, ...] = ()
    question: str | None = None


@dataclass(frozen=True)
class Tolerance:
    """Accept a distance up to the larger of absolute and relative times the expected size."""

    absolute: float
    relative: float = 0.0


@dataclass(frozen=True)
class Target:
    """A reported result. An expected value needs a locator; no value needs an open question."""

    name: str
    expected: float | None
    tolerance: Tolerance
    locators: tuple[Locator, ...] = ()
    question: str | None = None
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class Specification:
    paper: str
    data: str
    basis: Basis
    choices: tuple[Choice, ...]
    questions: tuple[OpenQuestion, ...]
    targets: tuple[Target, ...]
    pin: LocatorPin | None = None


@dataclass(frozen=True)
class Verdict:
    target: str
    basis: Basis
    reason: Reason
    questions: tuple[str, ...] = ()

    @property
    def outcome(self) -> Outcome:
        return _OUTCOMES[self.basis].get(self.reason, "inconclusive")


@dataclass(frozen=True)
class Classification:
    run: str
    spec_id: str
    verdicts: tuple[Verdict, ...]

    @property
    def outcome(self) -> Outcome:
        outcomes = {verdict.outcome for verdict in self.verdicts}
        for outcome in ("not_reproduced", "not_reproduced_on_adapted_data", "inconclusive"):
            if outcome in outcomes:
                return outcome
        return self.verdicts[0].outcome


@dataclass(frozen=True)
class Result:
    run: str
    spec_id: str
    observations: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class Reproduction:
    """A frozen specification and the results recorded against it, in order.

    A pinned reproduction holds the generation its specification pins. Every construction,
    including a copy, checks the equation, table, figure, and claim refs against it.
    """

    specification: Specification
    results: tuple[Result, ...] = ()
    generation: LocatorGeneration | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        _identity(self.specification)
        pin, generation = self.specification.pin, self.generation
        if pin is None and generation is None:
            return
        if pin is None or generation is None or generation.id != pin.generation:
            raise ReproductionError("A reproduction holds exactly the generation its pin names")
        if generation.digest != pin.digest:
            raise ReproductionError(f"Locator generation {pin.generation} has another digest")
        spec = self.specification
        for part in (*spec.questions, *spec.choices, *spec.targets):
            for locator in part.locators:
                if locator.kind not in _LOCATED:
                    continue
                location = generation.lookup(locator.ref)
                if location is None:
                    raise ReproductionError(f"Locator {locator.ref!r} names no location")
                if location.kind != locator.kind:
                    raise ReproductionError(
                        f"Locator {locator.ref!r} names kind {location.kind}, not {locator.kind}"
                    )

    @property
    def spec_id(self) -> str:
        return _identity(self.specification)

    def revise(
        self,
        specification: Specification,
        *,
        store: Path | None = None,
        context: RunContext | None = None,
    ) -> Reproduction:
        """Freeze a changed specification, which is allowed only before any result."""
        if self.results:
            raise ReproductionError("A specification is fixed once a result exists")
        return freeze(specification, store=store, context=context)

    def record(self, run: str, observations: Mapping[str, float]) -> Reproduction:
        """Return a new reproduction that adds a snapshot of one run's observations."""
        _text(run, "Run")
        if run in {result.run for result in self.results}:
            raise ReproductionError(f"Run {run!r} is already recorded")
        if unknown := set(observations) - {target.name for target in self.specification.targets}:
            raise ReproductionError(f"Observations name undeclared targets {sorted(unknown)}")
        result = Result(run, self.spec_id, tuple(sorted(observations.items())))
        return replace(self, results=(*self.results, result))

    def classify(self) -> tuple[Classification, ...]:
        """Return one classification per recorded result, in recording order."""
        spec = self.specification
        spec_id = self.spec_id
        choices = {choice.name: choice for choice in spec.choices}
        classifications = []
        for result in self.results:
            if result.spec_id != spec_id:
                raise ReproductionError(
                    f"Run {result.run!r} was recorded under another specification"
                )
            observed = dict(result.observations)
            verdicts = tuple(
                _verdict(target, spec.basis, choices, observed.get(target.name))
                for target in spec.targets
            )
            classifications.append(Classification(result.run, spec_id, verdicts))
        return tuple(classifications)


def _text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ReproductionError(f"{label} must be nonempty text")


def _names(label: str, names: list[str]) -> set[str]:
    for name in names:
        _text(name, label)
    if len(set(names)) != len(names):
        raise ReproductionError(f"Duplicate {label}s")
    return set(names)


def _verdict(
    target: Target, basis: Basis, choices: Mapping[str, Choice], observed: float | None
) -> Verdict:
    questions = tuple(
        dict.fromkeys(
            question
            for question in (
                target.question,
                *(choices[name].question for name in target.depends_on),
            )
            if question is not None
        )
    )
    if questions:
        return Verdict(target.name, basis, "open_question", questions)
    if observed is None:
        return Verdict(target.name, basis, "missing")
    if not math.isfinite(observed):
        return Verdict(target.name, basis, "non_finite")
    # Compare the shortest decimal forms so a value written on the boundary stays inside it.
    value, expected, absolute, relative = (
        Fraction(str(number))
        for number in (
            observed,
            target.expected,
            target.tolerance.absolute,
            target.tolerance.relative,
        )
    )
    if abs(value - expected) <= max(absolute, relative * abs(expected)):
        return Verdict(target.name, basis, "within_tolerance")
    return Verdict(target.name, basis, "outside_tolerance")


def freeze(
    specification: Specification,
    *,
    store: Path | None = None,
    context: RunContext | None = None,
) -> Reproduction:
    """Validate a paper reproduction specification and fix its identity before any result.

    Args:
        specification (Specification): Located choices, open questions, reported targets,
            comparison basis, tolerances, and an optional locator pin.
        store (Path, optional): Source store that holds the pinned locator generation.
        context (RunContext, optional): Trusted run authority that reads that generation.

    Returns:
        (Reproduction) The specification under its SHA-256 content identity, with no results.

    Raises:
        ReproductionError: If the specification is incomplete or inconsistent. For a pinned
            specification, also if the store or context is missing or invalid, the pinned
            generation is unpublished, unreadable under the context, or has another digest,
            or an equation, table, figure, or claim locator names no location of its kind
            in that generation.

    Notes:
        An unpinned specification is not resolved, and store and context are not read.
        Passage and code refs stay opaque, because chunk generations and the code index
        own them.
    """
    _identity(specification)
    if (pin := specification.pin) is None:
        return Reproduction(specification)
    if store is None or context is None:
        raise ReproductionError("A pinned specification needs a store and a run context")
    try:
        generations = published_locations(store, context)
    except SourceError as error:
        raise ReproductionError(str(error)) from error
    generation = next((item for item in generations if item.id == pin.generation), None)
    if generation is None:
        # Unpublished and unreadable read alike, so the refusal leaks no label or availability.
        raise ReproductionError(
            f"Locator generation {pin.generation} is unpublished or unreadable in this run"
        )
    return Reproduction(specification, generation=generation)


def _identity(specification: Specification) -> str:
    spec = specification
    if spec.basis not in ("matched_inputs", "adapted"):
        raise ReproductionError(f"Unknown comparison basis {spec.basis!r}")
    _text(spec.paper, "Paper")
    _text(spec.data, "Data")
    questions = _names("question id", [question.id for question in spec.questions])
    choices = _names("choice name", [choice.name for choice in spec.choices])
    _names("target name", [target.name for target in spec.targets])
    if not spec.targets:
        raise ReproductionError("A specification needs at least one target")
    for question in spec.questions:
        _text(question.detail, f"Question {question.id!r} detail")
        if question.kind not in ("ambiguous", "contradictory", "missing", "restricted"):
            raise ReproductionError(f"Unknown question kind {question.kind!r}")
    for part in (*spec.questions, *spec.choices, *spec.targets):
        for locator in part.locators:
            _text(locator.ref, "Locator ref")
            if locator.kind not in ("passage", "code", *_LOCATED):
                raise ReproductionError(f"Unknown locator kind {locator.kind!r}")
    for part, value in (
        *((choice, choice.value) for choice in spec.choices),
        *((target, target.expected) for target in spec.targets),
    ):
        if value is not None and not part.locators:
            raise ReproductionError(f"{part.name!r} needs a locator for its value")
        if value is None and part.question is None:
            raise ReproductionError(f"{part.name!r} needs a located value or an open question")
        if part.question is not None and part.question not in questions:
            raise ReproductionError(f"{part.name!r} names undeclared question {part.question!r}")
    for choice in spec.choices:
        if choice.value is not None:
            _text(choice.value, f"Choice {choice.name!r} value")
    for target in spec.targets:
        if target.expected is not None and not math.isfinite(target.expected):
            raise ReproductionError(f"Target {target.name!r} expects a non-finite value")
        bounds = (target.tolerance.absolute, target.tolerance.relative)
        if not all(math.isfinite(bound) and bound >= 0 for bound in bounds):
            raise ReproductionError(f"Target {target.name!r} needs finite nonnegative tolerances")
        if unknown := set(target.depends_on) - choices:
            raise ReproductionError(
                f"{target.name!r} depends on undeclared choices {sorted(unknown)}"
            )
    if unused := choices - {name for target in spec.targets for name in target.depends_on}:
        raise ReproductionError(f"No target depends on choices {sorted(unused)}")
    if unused := questions - {part.question for part in (*spec.choices, *spec.targets)}:
        raise ReproductionError(f"No choice or target names questions {sorted(unused)}")
    fields = asdict(spec)
    try:
        if spec.pin is None:
            # Leaving the key out keeps every identity derived before pins existed.
            del fields["pin"]
        else:
            parse_digest(spec.pin.generation)
            parse_digest(spec.pin.digest)
        document = canonical_bytes(fields)
    except PackageError as error:
        raise ReproductionError(str(error)) from error
    return hashlib.sha256(document).hexdigest()
