from __future__ import annotations

import hashlib
import json
import re
import sys
import sysconfig
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from fractions import Fraction
from importlib import resources
from pathlib import Path
from typing import get_args, get_type_hints
from uuid import UUID, uuid4

from quant_research import backtest, datasets, signals
from quant_research.access import Restrictions, RunContext, derive
from quant_research.backtest import Holding
from quant_research.catalog import Catalog
from quant_research.datasets import ReadRecord, Refused, Supplied
from quant_research.evaluation import (
    MINIMUM_OBSERVATIONS,
    VERSION,
    Evaluation,
    Interval,
    Invalid,
    Profile,
    Reason,
    evaluate,
)
from quant_research.packages import (
    ArtifactInput,
    EnvironmentRecord,
    ExecutionAttempt,
    ExposureEvent,
    LineageLink,
    Outcome,
    PackageRecord,
    Specification,
    Status,
    TrialRecord,
    VerifiedPackage,
    open_package,
    save_package,
)
from quant_research.providers import Material, dump_material, load_material
from quant_research.public_data import SEARCH_UNIVERSE
from quant_research.reconcile import SAHM_REALTIME, SeriesProfile, reconcile_series
from quant_research.records import (
    PackageError,
    canonical_bytes,
    canonical_json,
    format_time,
    parse_array,
    parse_digest,
    parse_fields,
    parse_text,
    parse_time,
)
from quant_research.replay import _sources
from quant_research.signals import SAHM_RULE, Decision, Program, Series, Subtract, Threshold, shift

GENERATION = "reference/v1"
IMPLEMENTATION_VERSION = "reference-pipeline/v1"
ACTOR = "software:quant_research.reference/v1"
BASELINE = "buy_and_hold_market"
EVALUATION_ROLE = "pilot-evaluation"
FACTORS = "french.ff3_monthly"
LAG = 2
# A search candidate's signal profile is this and the SHA-256 of its program artifact.
SEARCH_PROFILE = "search/v1"
ENTRYPOINT = "quant_research.reference:extend"
LINK = "Next candidate of the campaign's search"
UNFINISHED = "The experiment ended without a result; nothing was evaluated"
LIMITATIONS = (
    "Development result on public data that the researcher and the models have seen; no "
    "investment claim",
    "Research returns from the factor file, with no slippage, taxes, or financing beyond the "
    "declared switch cost",
    "The evaluation profile assumes independent, identically distributed errors (D015), so its "
    "inference is provisional",
    "Caller-supplied environment lock; installation and dependency reconciliation unverified",
    "Input artifacts hold whole vintage files, including rows after the run's cutoff; exposure "
    "events record what the software read",
    "Software exposure events do not establish human exposure or complete disclosure",
    "An agent chose the frozen experiment file under the user's autonomy grant",
)

# One spelling per value, so equal parameters always give equal file digests.
_DECIMAL = re.compile(r"(0|[1-9][0-9]*)(\.[0-9]*[1-9])?", re.ASCII)
_REASONS = get_args(Reason.__value__)
_INVALID_REASONS = get_args(get_type_hints(Invalid)["reason"])
_CONDITIONS = get_args(signals.Condition.__value__)
# Every reason a validated experiment can reach. too_few_observations, length_mismatch, and
# nonfinite_value are absent: the window length, one shared sample, and exact returns rule them
# out, so a KeyError here is a broken invariant rather than a research outcome.
_STATUS: dict[str, Status] = {
    "above_minimum": "successful",
    "below_minimum": "null",
    "spans_minimum": "inconclusive",
    "unstable_subperiods": "inconclusive",
    "zero_variance": "inconclusive",
}


class ReferencePipelineError(ValueError):
    """Report invalid reference inputs, refused reads, or conflicting execution evidence."""


@dataclass(frozen=True)
class Experiment:
    """A frozen experiment file, parsed, with the exact bytes that were parsed.

    start and end are the first and last decision months, which are also the first and last
    return months. cutoff is the information horizon every run narrows its authority to. data
    is canonical JSON, so its SHA-256 is the trial's data contract digest, which binds the
    window, cutoff, baseline, and evaluation parameters to the specification ID.
    """

    start: date
    end: date
    cutoff: datetime
    profile: Profile
    data: bytes

    @property
    def months(self) -> tuple[date, ...]:
        count = (self.end.year - self.start.year) * 12 + self.end.month - self.start.month + 1
        return tuple(shift(self.start, offset) for offset in range(count))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


def _midnight(day: date) -> datetime:
    return datetime.combine(day, time(), UTC)


def _month(text: object) -> date:
    month = date.fromisoformat(text) if isinstance(text, str) else None
    if month is None or month.isoformat() != text or month.day != 1:
        raise ValueError(f"{text!r} is not the first day of a month as YYYY-MM-DD")
    return month


def _decimal(text: object) -> float:
    if not isinstance(text, str) or not _DECIMAL.fullmatch(text):
        raise ValueError(f"{text!r} is not decimal text such as '0.05'")
    return float(Decimal(text))


def load_experiment(data: bytes) -> Experiment:
    """Parse a frozen experiment file and refuse anything it does not fully state.

    The file is canonical JSON with exactly the fields baseline, cutoff, evaluation (alpha,
    minimum_effect, subperiods, version), schema_version, and window (end, start). alpha and
    minimum_effect are decimal text without redundant zeros, and subperiods is an integer.

    Args:
        data (bytes): The file's bytes.

    Returns:
        (Experiment) The parsed experiment, holding data unchanged.

    Raises:
        ReferencePipelineError: If the bytes are not canonical JSON, a field is missing or
            unknown, the baseline is not buy_and_hold_market, the procedure version is not
            evaluation.VERSION, a window date is not the first of a month, the parameters are
            not a valid evaluation.Profile, the window has fewer months than the procedure
            needs, or the cutoff is before the last month's returns are available.
    """
    try:
        value = parse_fields(
            canonical_json(data, "Experiment"),
            "baseline cutoff evaluation schema_version window",
            "experiment",
        )
        window = parse_fields(value["window"], "end start", "window")
        rules = parse_fields(
            value["evaluation"], "alpha minimum_effect subperiods version", "evaluation"
        )
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError(f"Unsupported schema version {value['schema_version']!r}")
        if value["baseline"] != BASELINE:
            raise ValueError(f"The baseline must be {BASELINE!r}")
        if rules["version"] != VERSION:
            raise ValueError(f"The evaluation procedure must be {VERSION!r}")
        experiment = Experiment(
            _month(window["start"]),
            _month(window["end"]),
            parse_time(value["cutoff"], "cutoff"),
            Profile(
                _decimal(rules["minimum_effect"]), rules["subperiods"], _decimal(rules["alpha"])
            ),
            data,
        )
        needed = max(MINIMUM_OBSERVATIONS, 2 * experiment.profile.subperiods)
        if len(experiment.months) < needed:
            raise ValueError(f"The window needs at least {needed} months")
        if experiment.cutoff < _midnight(shift(experiment.end, 1)):
            raise ValueError("The cutoff is before the last month's returns are available")
    except ValueError as error:
        raise ReferencePipelineError(f"Invalid experiment: {error}") from error
    return experiment


def frozen() -> Experiment:
    """Return the committed experiment, reference.json beside this module."""
    return load_experiment(
        resources.files("quant_research").joinpath("reference.json").read_bytes()
    )


@dataclass(frozen=True)
class Rules:
    """A frozen repeated-search method: a search window and the held-back window after it (D065).

    Search results return to the model; only a final evaluation reads held_back. The windows are
    contiguous, search first, and search.cutoff is midnight of held_back's first month, so a read
    at the search cutoff supplies the search window's last return and no held-back return. data
    is the canonical file, so digest is what a campaign's EvaluationRules.digest names.
    """

    search: Experiment
    held_back: Experiment
    data: bytes

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


def load_rules(data: bytes) -> Rules:
    """Parse a rules file and refuse anything it does not fully state.

    The file is canonical JSON with exactly the fields held_back, schema_version (1), and search.
    Each window is one document in reference.json's schema, which load_experiment parses.

    Args:
        data (bytes): The file's bytes.

    Returns:
        (Rules) The two windows, holding data unchanged.

    Raises:
        ReferencePipelineError: If the bytes are not canonical JSON, a field is missing or
            unknown, the schema version is not 1, load_experiment refuses a window, the held-back
            window does not start the month after the search window ends, or the search
            cutoff is not midnight of the held-back window's first month.
    """
    try:
        value = parse_fields(
            canonical_json(data, "Rules"), "held_back schema_version search", "rules"
        )
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError(f"Unsupported schema version {value['schema_version']!r}")
        search, held_back = (
            load_experiment(canonical_bytes(value[window])) for window in ("search", "held_back")
        )
        if held_back.start != shift(search.end, 1):
            raise ValueError("The held-back window must start the month after the search window")
        if search.cutoff != _midnight(held_back.start):
            raise ValueError("The search cutoff must be midnight of the held-back window's start")
    except ValueError as error:
        raise ReferencePipelineError(f"Invalid rules: {error}") from error
    return Rules(search, held_back, data)


def rules(directory: Path, digest: str) -> Rules:
    """Return the rules file named <digest>.json in directory, whose bytes must hash to digest.

    Args:
        directory (Path): Directory of rules files, each named by its SHA-256.
        digest (str): The SHA-256 a campaign's EvaluationRules.digest names.

    Returns:
        (Rules) The parsed file.

    Raises:
        ReferencePipelineError: If digest is not a SHA-256, the file is missing or unreadable,
            its bytes hash to another digest, or load_rules refuses it.
    """
    try:
        data = (directory / f"{parse_digest(digest)}.json").read_bytes()
    except (OSError, ValueError) as error:
        raise ReferencePipelineError(f"No rules file is named {digest!r}") from error
    if hashlib.sha256(data).hexdigest() != digest:
        raise ReferencePipelineError(f"The rules file named {digest} holds other bytes")
    return load_rules(data)


@dataclass(frozen=True)
class Search:
    """What every trial of one campaign's search shares. campaigns.resolve builds it.

    context is the search authority: the campaign's scoped context narrowed once to
    rules.search.cutoff, the context its Toolset holds. lineage is the campaign ID, since one
    campaign's search is one lineage, and destination is that lineage's directory, which every
    worker of the queue shares. question is the revision's, so a steered child's trials carry
    the question in force.
    """

    question: str
    lineage: UUID
    rules: Rules
    catalog: Catalog
    store: Path
    context: RunContext
    lock: bytes
    destination: Path

    @property
    def readable(self) -> frozenset[str]:
        """Return every dataset that a read under context would supply rather than refuse."""
        return frozenset(
            entry.dataset for entry in datasets.listing(self.catalog, self.context).datasets
        )

    @property
    def universe(self) -> tuple[str, ...]:
        """Return the datasets a candidate may read: the search universe that context can read."""
        readable = self.readable
        return tuple(name for name in SEARCH_UNIVERSE if name in readable)


type Decider = Callable[
    [Program, Catalog, Path, RunContext, Sequence[date]], tuple[Decision, ...] | Refused
]


def _read(
    catalog: Catalog, store: Path, names: Sequence[str], context: RunContext
) -> dict[str, Supplied] | Refused:
    supplied = {}
    for name in names:
        result = datasets.read(catalog, store, name, context)
        if isinstance(result, Refused):
            return result
        supplied[name] = result
    return supplied


def _history(supplied: dict[str, Supplied]) -> signals.History:
    return {
        name: {date.fromisoformat(row.key): row.values for row in result.rows}
        for name, result in supplied.items()
    }


def _released(supplied: dict[str, Supplied]) -> datetime | None:
    return max(
        (row.available_at for result in supplied.values() for row in result.rows), default=None
    )


def current_vintage(
    program: Program, catalog: Catalog, store: Path, context: RunContext, months: Sequence[date]
) -> tuple[Decision, ...] | Refused:
    """Decide each month with values from the vintage current at the context's cutoff.

    Deliberately flawed. Observation months come from signals.decide, so the only flaw is the
    vintage: one read at the context's cutoff, made once and reused for every past decision,
    supplies each value. Evaluating the program at an observation reads no month after it.
    Every decision reports that read, and a decision with a value reports its release time.

    Args:
        program (Program): Condition to evaluate.
        catalog (Catalog): Catalog that must approve every dataset the program reaches.
        store (Path): Dataset store directory.
        context (RunContext): Trusted run authority whose cutoff is at or after every month.
        months (Sequence[date]): Decision months, each the first day of its month.

    Returns:
        (tuple[Decision, ...] | Refused) One decision per month, or the first refused read.
    """
    causal = signals.decide(program, catalog, store, context, months)
    if isinstance(causal, Refused):
        return causal
    current = _read(catalog, store, sorted(program.reach), context)
    if isinstance(current, Refused):
        return current
    history = _history(current)
    reads = tuple(result.record for result in current.values())
    return tuple(
        replace(decision, reads=reads)
        if decision.observation is None
        else replace(
            decision,
            value=program.evaluate(history, decision.observation),
            released_at=_released(current),
            reads=reads,
        )
        for decision in causal
    )


def _appearance(
    catalog: Catalog, store: Path, names: Sequence[str], context: RunContext, key: str
) -> dict[str, Supplied] | Refused | None:
    """Return the earliest read under context whose every dataset holds key, None if none does.

    The adapter supplies only a key's latest revision at a cutoff, so each walk steps back to
    just before the revision it found until the key is absent. The last revision found is the
    key's first.
    """
    first = []
    for name in names:
        cutoff, appeared = context.cutoff, None
        while True:
            result = datasets.read(catalog, store, name, context.narrow(cutoff=cutoff))
            if isinstance(result, Refused):
                return result
            row = next((row for row in result.rows if row.key == key), None)
            if row is None:
                break
            appeared = row.available_at
            cutoff = appeared - timedelta(microseconds=1)
        if appeared is None:
            return None
        first.append(appeared)
    return _read(catalog, store, names, context.narrow(cutoff=max(first)))


def fixed_lag(
    program: Program, catalog: Catalog, store: Path, context: RunContext, months: Sequence[date]
) -> tuple[Decision, ...] | Refused:
    """Decide each month m from observation m-2, read when month m-2 first appears.

    Deliberately flawed. When every vintage read at the start of m already holds m-2, the
    decision is that causal read. When the release slipped, it reads at the time key m-2 first
    becomes available, after m began. A key appears once any revision of it is available, even
    one whose value is null. Waiting for a value instead would never see a month such as
    October 2025, which has none, so its slipped release would leave no trace. Reads never pass
    the context's cutoff, and an m-2 that has not appeared by then leaves m with no value and
    its causal read.

    Args:
        program (Program): Condition to evaluate.
        catalog (Catalog): Catalog that must approve every dataset the program reaches.
        store (Path): Dataset store directory.
        context (RunContext): Trusted run authority whose cutoff is at or after every month.
        months (Sequence[date]): Decision months, each the first day of its month.

    Returns:
        (tuple[Decision, ...] | Refused) One decision per month, or the first refused read.
    """
    names = sorted(program.reach)
    decisions = []
    for month in months:
        observation = shift(month, -LAG)
        supplied = _read(catalog, store, names, context.narrow(cutoff=_midnight(month)))
        if isinstance(supplied, Refused):
            return supplied
        if any(observation not in rows for rows in _history(supplied).values()):
            slipped = _appearance(catalog, store, names, context, observation.isoformat())
            if isinstance(slipped, Refused):
                return slipped
            if slipped is None:
                reads = tuple(result.record for result in supplied.values())
                decisions.append(Decision(month, None, None, None, reads))
                continue
            supplied = slipped
        decisions.append(
            Decision(
                month,
                observation,
                program.evaluate(_history(supplied), observation),
                _released(supplied),
                tuple(result.record for result in supplied.values()),
            )
        )
    return tuple(decisions)


@dataclass(frozen=True)
class Case:
    """A rule to test, the decider that reads data for it, and the statuses its run expects.

    program is a condition: True holds one-month T-bills and False holds the market. name is
    the specification's signal profile. known_answer, when set, names a published series the
    program's indicator must reproduce before any return is computed.
    """

    name: str
    question: str
    program: Program
    decide: Decider
    known_answer: SeriesProfile | None
    expected: frozenset[Status]


CASES = (
    Case(
        "sahm-rule/v1",
        "Does holding one-month T-bills while the real-time Sahm indicator is at or above 0.50, "
        "and the market otherwise, earn a monthly return that exposure to buy-and-hold market "
        "returns does not explain?",
        SAHM_RULE,
        signals.decide,
        SAHM_REALTIME,
        frozenset({"successful", "null", "inconclusive"}),
    ),
    Case(
        "sahm-rule-current-vintage/v1",
        "Control: does the causality audit reject the Sahm rule computed from the current UNRATE "
        "vintage?",
        SAHM_RULE,
        current_vintage,
        None,
        frozenset({"rejected"}),
    ),
    Case(
        "sahm-rule-fixed-lag/v1",
        "Control: does the causality audit reject the Sahm rule observed a fixed two months back?",
        SAHM_RULE,
        fixed_lag,
        None,
        frozenset({"rejected"}),
    ),
    # u - u is 0 in every month, below 1/2, so the rule always holds the market and its returns
    # are the baseline's: the incremental interval is [0, 0] before any data is read.
    Case(
        "never-bills/v1",
        "Control: does a rule that never leaves the market record null against buy-and-hold?",
        Threshold(Subtract(Series("alfred.unrate"), Series("alfred.unrate")), Fraction(1, 2)),
        signals.decide,
        None,
        frozenset({"null"}),
    ),
)


@dataclass(frozen=True)
class Breach:
    """A decision whose record could rest on data from after its month began.

    seen is the latest read cutoff or release time the decision names, None when it names no
    read.
    """

    month: date
    seen: datetime | None


def audit(decisions: Sequence[Decision]) -> tuple[Breach, ...]:
    """Return every decision whose record could have seen past the start of its month.

    A decision breaches when it names no read, when a read's cutoff is after 00:00 UTC on the
    first of its month, or when its release time is. The adapter mints each read's cutoff from
    the context it was given and supplies only rows available by then, so the saved records
    are enough to repeat this check.

    Args:
        decisions (Sequence[Decision]): Decisions to check, each with the reads it names.

    Returns:
        (tuple[Breach, ...]) The breaching decisions, in their supplied order.
    """
    breaches = []
    for decision in decisions:
        seen = [read.cutoff for read in decision.reads]
        if decision.released_at is not None:
            seen.append(decision.released_at)
        latest = max(seen, default=None)
        if not decision.reads or latest > _midnight(decision.month):
            breaches.append(Breach(decision.month, latest))
    return tuple(breaches)


def allocate(decisions: Sequence[Decision]) -> dict[date, Holding] | None:
    """Return each decision month's holding, or None when the first decision has no value.

    True holds T-bills and False holds the market. A decision without a value keeps the
    previous holding, so only an undecided first month leaves the allocation undefined.

    Args:
        decisions (Sequence[Decision]): Decisions in month order.

    Returns:
        (dict[date, Holding] | None) The holding per month, or None.

    Raises:
        ReferencePipelineError: If a decision's value is a number rather than a condition's.
    """
    holdings: dict[date, Holding] = {}
    holding = None
    for decision in decisions:
        match decision.value:
            case True:
                holding = Holding.BILLS
            case False:
                holding = Holding.MARKET
            case None if holding is None:
                return None
            case None:
                pass
            case value:
                raise ReferencePipelineError(
                    f"The decision for {decision.month:%Y-%m} is {value}, not a condition"
                )
        holdings[decision.month] = holding
    return holdings


def encode_evaluation(result: Evaluation | Invalid) -> bytes:
    """Return the pilot-evaluation artifact bytes of an evaluation result.

    The bytes are canonical JSON of every field of the result, its procedure version and
    parameters included. Floats keep their shortest round-trip text, so decode_evaluation and
    then encode_evaluation return the same bytes.

    Args:
        result (Evaluation | Invalid): What evaluation.evaluate returned.

    Returns:
        (bytes) Canonical UTF-8 JSON.
    """
    kind = "evaluation" if isinstance(result, Evaluation) else "invalid"
    return canonical_bytes({"kind": kind, "result": asdict(result), "schema_version": 1})


def _float(value: object, label: str) -> float:
    if type(value) is not float:
        raise PackageError(f"{label} must be a JSON number with a fraction or exponent")
    return value


def _interval(raw: object) -> Interval:
    value = parse_fields(raw, "estimate lower standard_error upper", "interval")
    return Interval(
        *(_float(value[key], key) for key in ("estimate", "standard_error", "lower", "upper"))
    )


def _profile(raw: object) -> Profile:
    value = parse_fields(raw, "alpha minimum_effect subperiods version", "profile")
    if value["version"] != VERSION:
        raise PackageError(f"Unsupported evaluation procedure {value['version']!r}")
    minimum_effect, alpha = (
        _float(value["minimum_effect"], "minimum_effect"),
        _float(value["alpha"], "alpha"),
    )
    try:
        return Profile(minimum_effect, value["subperiods"], alpha)
    except ValueError as error:
        raise PackageError(f"Invalid evaluation parameters: {error}") from error


def decode_evaluation(data: bytes) -> Evaluation | Invalid:
    """Parse pilot-evaluation artifact bytes back into the evaluation's own types.

    Args:
        data (bytes): Bytes that encode_evaluation wrote.

    Returns:
        (Evaluation | Invalid) The result, with its Profile rebuilt.

    Raises:
        PackageError: If the bytes are not canonical JSON, a field is missing, unknown, or of
            another JSON type, a reason is not one the procedure returns, the subperiod
            estimates do not match the subperiod count, or the procedure version is not
            evaluation.VERSION.
    """
    value = parse_fields(
        canonical_json(data, "Pilot evaluation"), "kind result schema_version", "pilot evaluation"
    )
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise PackageError(f"Unsupported schema version {value['schema_version']!r}")
    match value["kind"]:
        case "evaluation":
            raw = parse_fields(
                value["result"],
                "confidence incremental mean profile reason subperiods trials",
                "evaluation",
            )
            profile = _profile(raw["profile"])
            estimates = tuple(
                _float(item, "subperiod estimate")
                for item in parse_array(raw["subperiods"], "subperiods")
            )
            trials = raw["trials"]
            if raw["reason"] not in _REASONS:
                raise PackageError("Unsupported evaluation reason")
            if type(trials) is not int or trials < 1:
                raise PackageError("trials must be a positive integer")
            if len(estimates) != profile.subperiods:
                raise PackageError("Subperiod estimates do not match the subperiod count")
            return Evaluation(
                raw["reason"],
                trials,
                _float(raw["confidence"], "confidence"),
                _interval(raw["mean"]),
                None if raw["incremental"] is None else _interval(raw["incremental"]),
                estimates,
                profile,
            )
        case "invalid":
            raw = parse_fields(value["result"], "detail profile reason", "invalid")
            if raw["reason"] not in _INVALID_REASONS:
                raise PackageError("Unsupported invalid-input reason")
            return Invalid(
                raw["reason"], parse_text(raw["detail"], "detail"), _profile(raw["profile"])
            )
    raise PackageError("Pilot evaluation kind must be evaluation or invalid")


def _encode_decisions(decisions: Sequence[Decision]) -> bytes:
    return canonical_bytes(
        {
            "decisions": [
                {
                    "month": decision.month.isoformat(),
                    "observation": None
                    if decision.observation is None
                    else decision.observation.isoformat(),
                    "value": str(decision.value)
                    if isinstance(decision.value, Fraction)
                    else decision.value,
                    "released_at": None
                    if decision.released_at is None
                    else format_time(decision.released_at),
                    "reads": [
                        {
                            "catalog": read.catalog,
                            "context": read.context,
                            "cutoff": format_time(read.cutoff),
                            "dataset": read.dataset,
                            "record_digest": hashlib.sha256(read.to_bytes()).hexdigest(),
                            "supplied": len(read.supplied),
                            "vintage": read.vintage,
                        }
                        for read in decision.reads
                    ],
                }
                for decision in decisions
            ],
            "schema_version": 1,
        }
    )


def _encode_audit(decisions: Sequence[Decision], breaches: Sequence[Breach]) -> bytes:
    return canonical_bytes(
        {
            "breaches": [
                {
                    "month": breach.month.isoformat(),
                    "seen": None if breach.seen is None else format_time(breach.seen),
                }
                for breach in breaches
            ],
            "decisions": len(decisions),
            "schema_version": 1,
        }
    )


@dataclass(frozen=True)
class _Finding:
    status: Status
    detail: str


@dataclass(frozen=True)
class _Scored:
    strategy: tuple[Fraction, ...]
    baseline: tuple[Fraction, ...]


@dataclass(frozen=True)
class _Examined:
    """Every read record a case's checks made, their result artifacts, and where they ended."""

    records: tuple[ReadRecord, ...]
    results: dict[str, bytes]
    outcome: _Finding | _Scored


def _examine(
    case: Case, experiment: Experiment, catalog: Catalog, store: Path, authority: RunContext
) -> _Examined | Refused:
    decisions = case.decide(case.program, catalog, store, authority, experiment.months)
    if isinstance(decisions, Refused):
        return decisions
    records = [read for decision in decisions for read in decision.reads]
    breaches = audit(decisions)
    results = {
        "decisions": _encode_decisions(decisions),
        "audit": _encode_audit(decisions, breaches),
    }

    def found(status: Status, detail: str) -> _Examined:
        return _Examined(tuple(records), results, _Finding(status, detail))

    if breaches:
        return found(
            "rejected",
            f"Look-ahead: {len(breaches)} of {len(decisions)} decisions rest on data from after "
            f"their month began, first {breaches[0].month:%Y-%m}",
        )
    if case.known_answer is not None:
        # reconcile_series raises on a refused read, and a refusal never depends on the cutoff,
        # so reading the published series first returns any refusal as Refused.
        known = _read(catalog, store, [case.known_answer.reference], authority)
        if isinstance(known, Refused):
            return known
        records.extend(result.record for result in known.values())
        answer = reconcile_series(case.known_answer, catalog, store, authority)
        results["reconciliation"] = answer.record()
        if not answer.agreed:
            return found(
                "rejected",
                f"Known answer {answer.profile} mismatches in {len(answer.mismatched)} months, "
                f"first {answer.mismatched[0].month:%Y-%m}",
            )
    holdings = allocate(decisions)
    if holdings is None:
        return found(
            "rejected",
            f"The first decision, {experiment.start:%Y-%m}, has no value, so no initial "
            "allocation exists",
        )
    market = backtest.read(catalog, store, authority)
    if isinstance(market, Refused):
        return market
    records.append(market.record)
    sample = market.sample(holdings)
    agreement = backtest.compare(sample)
    baseline = backtest.plain(market.sample(dict.fromkeys(holdings, Holding.MARKET)))
    results["returns"] = canonical_bytes(
        {
            "months": [
                {
                    "baseline": str(base),
                    "holding": period.holding.name.lower(),
                    "month": period.month.isoformat(),
                    "strategy": str(strategy),
                }
                for period, strategy, base in zip(
                    sample.periods, agreement.reference, baseline, strict=True
                )
            ],
            "schema_version": 1,
        }
    )
    results["engine-agreement"] = canonical_bytes(
        {
            "agreed": agreement.agreed,
            "difference": str(agreement.difference),
            "engine": "vectorbt",
            "month": agreement.month.isoformat(),
            "schema_version": 1,
            "tolerance": str(backtest.TOLERANCE),
        }
    )
    if not agreement.agreed:
        return found(
            "failed",
            f"vectorbt differs from the plain engine by {float(agreement.difference):.3g} in "
            f"{agreement.month:%Y-%m}, above the tolerance {float(backtest.TOLERANCE):g}",
        )
    return _Examined(tuple(records), results, _Scored(agreement.reference, baseline))


def _summary(result: Evaluation | Invalid) -> str:
    if isinstance(result, Invalid):
        return f"Pilot evaluation is inconclusive ({result.reason}): {result.detail}"
    tested = result.incremental
    return (
        f"Pilot evaluation verdict {result.verdict} ({result.reason}): incremental return "
        f"{tested.estimate:.6g} per month, interval {tested.lower:.6g} to {tested.upper:.6g} at "
        f"confidence {result.confidence:.6g}, minimum effect {result.profile.minimum_effect:g}"
    )


def _artifact(
    label: str, data: bytes, role: str, media_type: str = "application/json"
) -> ArtifactInput:
    return ArtifactInput(f"{label}-{hashlib.sha256(data).hexdigest()[:16]}", data, media_type, role)


def run(
    case: Case,
    experiment: Experiment,
    *,
    catalog: Catalog,
    store: Path,
    context: RunContext,
    lock: bytes,
    destination: Path,
) -> VerifiedPackage | Refused:
    """Decide, audit, allocate, back test on two engines, evaluate, and save one package.

    The context is narrowed to the experiment's cutoff before any read, so every read record,
    and with them every result artifact, is the same whatever later cutoff the caller holds.
    A breach of the causality audit, a known answer that mismatches, or an undecided first month
    rejects the trial before any return is computed. Engines that disagree beyond
    backtest.TOLERANCE fail it. Otherwise the evaluation's reason sets the status. The package
    holds one trial with one attempt, so the evaluation counts one trial.

    Args:
        case (Case): Rule, decider, optional known answer, and expected statuses.
        experiment (Experiment): Frozen window, cutoff, and evaluation parameters.
        catalog (Catalog): Catalog that must approve every dataset the case reads.
        store (Path): Dataset store directory.
        context (RunContext): Trusted run authority whose cutoff is at or after the experiment's.
        lock (bytes): Caller-supplied environment lock. The run does not install or verify it.
        destination (Path): Parent directory for the package.

    Returns:
        (VerifiedPackage | Refused) The saved package, or the first refused read.

    Raises:
        AccessError: If the context's cutoff is before the experiment's.
        BacktestError: If the factors lack a window month.
        DatasetError: If a documented vintage is unreadable or malformed.
        ReconcileError: If the known answer cannot be reconciled.
        PackageError: If the package cannot be saved.
    """
    started = datetime.now(UTC)
    authority = context.narrow(cutoff=experiment.cutoff)
    examined = _examine(case, experiment, catalog, store, authority)
    if isinstance(examined, Refused):
        return examined
    built = _trial(
        examined,
        experiment,
        authority,
        catalog=catalog,
        store=store,
        lock=lock,
        question=case.question,
        signal_profile=case.name,
        entrypoint="quant_research.reference:run",
        program=(),
        trial=uuid4(),
        lineage=uuid4(),
        history=(),
        started=started,
    )
    if isinstance(built, Refused):
        return built
    record = PackageRecord(built.trial.trial_id, (built.trial,), built.exposures, "declared_inputs")
    return save_package(destination, record, built.artifacts)


@dataclass(frozen=True)
class _Built:
    """One trial with its one attempt, and everything its package and its reader need.

    bounds holds the read at the run's authority of each dataset the trial read, empty for a
    trial that read nothing. evaluation is the pilot-evaluation artifact, None without one.
    """

    trial: TrialRecord
    artifacts: tuple[ArtifactInput, ...]
    exposures: tuple[ExposureEvent, ...]
    bounds: dict[str, Supplied]
    evaluation: bytes | None


def _trial(
    examined: _Examined,
    experiment: Experiment,
    authority: RunContext,
    *,
    catalog: Catalog,
    store: Path,
    lock: bytes,
    question: str,
    signal_profile: str,
    entrypoint: str,
    program: tuple[ArtifactInput, ...],
    trial: UUID,
    lineage: UUID,
    history: tuple[TrialRecord, ...],
    started: datetime,
    finished: datetime | None = None,
    adjustment_trials: int | None = None,
) -> _Built | Refused:
    """Record what _examine found as one trial evaluated against history, the trials before it.

    The trial follows the last trial of history. The evaluation counts history and this trial
    unless adjustment_trials fixes a candidate family independently of measurement count.
    finished is when the attempt ended, None for the moment its evaluation does.
    """
    # Deciders and reconcile_series read more than their decisions report. A read at the
    # authority supplies every key any narrower read did, so it bounds what the run saw.
    bounds = _read(catalog, store, sorted({read.dataset for read in examined.records}), authority)
    if isinstance(bounds, Refused):
        return bounds
    read_at = datetime.now(UTC)
    inputs = {}
    for record in (*examined.records, *(result.record for result in bounds.values())):
        if record.vintage not in inputs:
            data = (store / record.dataset / f"{record.vintage}.json").read_bytes()
            record.replay(data)
            inputs[record.vintage] = _artifact(record.dataset, data, "input")
    implementation = canonical_bytes(
        {
            "entrypoint": entrypoint,
            "sources": dict(_sources(resources.files("quant_research"), "quant_research/")),
        }
    )
    named = (
        _artifact("contract", experiment.data, "contract"),
        _artifact("implementation", implementation, "implementation"),
        _artifact("lock", lock, "lock", "text/plain"),
    )
    contract, code, environment = (hashlib.sha256(item.data).hexdigest() for item in named)
    record = TrialRecord(
        trial,
        lineage,
        Specification(
            question,
            signal_profile,
            VERSION,
            datasets.ADAPTER,
            IMPLEMENTATION_VERSION,
            tuple(sorted(inputs)),
            contract,
            code,
            EnvironmentRecord(
                sys.implementation.name,
                ".".join(str(part) for part in sys.version_info[:3]),
                sysconfig.get_config_var("SOABI"),
                sysconfig.get_platform(),
                "not observed; caller-supplied lock",
                environment,
            ),
        ),
        (),
        tuple(
            LineageLink(prior.trial_id, prior.specification.spec_id, "follows", LINK)
            for prior in history[-1:]
        ),
    )
    results = dict(examined.results)
    if isinstance(examined.outcome, _Finding):
        status, detail = examined.outcome.status, examined.outcome.detail
    else:
        profile = experiment.profile
        result = evaluate(
            [float(value) for value in examined.outcome.strategy],
            PackageRecord(record.trial_id, (*history, record)),
            [float(value) for value in examined.outcome.baseline],
            minimum_effect=profile.minimum_effect,
            subperiods=profile.subperiods,
            alpha=profile.alpha,
            adjustment_trials=adjustment_trials,
        )
        results[EVALUATION_ROLE] = encode_evaluation(result)
        status, detail = _STATUS[result.reason], _summary(result)
    produced = tuple(_artifact(role, data, role) for role, data in results.items())
    attempt = ExecutionAttempt(
        uuid4(),
        uuid4(),
        None,
        started,
        datetime.now(UTC) if finished is None else finished,
        Outcome(status, detail, tuple(item.name for item in produced), LIMITATIONS),
    )
    exposures = [
        ExposureEvent(
            uuid4(),
            ACTOR,
            read_at,
            supplied.record.vintage,
            name,
            _midnight(date.fromisoformat(supplied.rows[0].key)),
            _midnight(date.fromisoformat(supplied.rows[-1].key)),
            "data",
            "Software read this vintage through the dataset adapter; the interval spans the keys "
            "a read at the run's cutoff supplied, which bound every read the run made; human "
            "exposure is unknown",
        )
        for name, supplied in bounds.items()
        if supplied.rows
    ]
    if EVALUATION_ROLE in results:
        exposures.append(
            ExposureEvent(
                uuid4(),
                ACTOR,
                attempt.finished_at,
                hashlib.sha256(results[EVALUATION_ROLE]).hexdigest(),
                FACTORS,
                _midnight(experiment.start),
                _midnight(experiment.end),
                "result",
                "Software produced the pilot evaluation of the window's returns; the interval "
                "spans the evaluated months; human exposure is unknown",
            )
        )
    return _Built(
        replace(record, attempts=(attempt,)),
        (*inputs.values(), *named, *program, *produced),
        tuple(exposures),
        bounds,
        results.get(EVALUATION_ROLE),
    )


@dataclass(frozen=True)
class Extended:
    """The lineage's new head, its newest trial's status, and what the model reads of it.

    summary is canonical JSON text with exactly the keys detail, evaluation, status, and trial,
    built from the outcome and the pilot-evaluation artifact, never from inputs or any other
    package artifact. restrictions derive over the search floor and every dataset the trial
    read at the search authority.
    """

    package: VerifiedPackage
    status: Status
    summary: str
    restrictions: Restrictions


def extend(search: Search, head: str | None, trial: UUID, program: bytes) -> Extended:
    """Run one candidate on the search window and save the lineage with it appended (D065).

    Every candidate becomes a trial. A program the grammar, catalog, or universe refuses, or one
    that is a number rather than a condition, is rejected before any read. A candidate fails
    before any read when the search cannot read the factors. Any other program runs through
    run's pipeline, whose reads the universe and that check admit, so none is refused. The
    evaluation counts the lineage so far, this trial included, and the head selects no trial.

    Args:
        search (Search): The campaign's search, from its Authority.
        head (str | None): Digest of the head the workflow last recorded, None before the
            first trial.
        trial (UUID): The workflow's ID for this candidate.
        program (bytes): The move's arguments exactly as the model sent them.

    Returns:
        (Extended) The new head, the trial's status, its summary, and its restrictions.

    Raises:
        PackageError: If the head cannot be opened or the new head cannot be saved.
        BacktestError: If the factors lack a window month.
        DatasetError: If a documented vintage is unreadable or malformed.
        ReferencePipelineError: If a read the search can make is refused.
    """
    started = datetime.now(UTC)
    prior = None if head is None else open_lineage(search, head)
    identity, candidate = _candidate(search, program)
    if isinstance(candidate, str):
        reason = signals.Rejected(candidate)
        examined = _unrun("rejected", f"Rejected ({candidate}): {reason.message}")
    elif backtest.FACTORS not in search.readable:
        # Checked before any read: a refusal after the signal's reads would lose their records.
        examined = _unrun("failed", f"The search cannot read {backtest.FACTORS}; nothing was read")
    else:
        case = Case(
            _signal_profile(identity), search.question, candidate, signals.decide, None, frozenset()
        )
        examined = _examine(case, search.rules.search, search.catalog, search.store, search.context)
    made = examined
    if not isinstance(examined, Refused):
        made = _searched(search, prior, examined, identity, trial, started)
    if isinstance(made, Refused):
        raise ReferencePipelineError(f"A read the search admitted was refused: {made.message}")
    outcome = made.trial.attempts[0].outcome
    summary = {
        "detail": outcome.detail,
        "evaluation": None if made.evaluation is None else json.loads(made.evaluation),
        "status": outcome.status,
        "trial": str(trial),
    }
    # agent_tools._floor, which is private: tool results reach the model under the same floor.
    floor = Restrictions(
        available_at=search.context.cutoff,
        labels=frozenset(),
        approved_for=search.context.providers,
    )
    return Extended(
        _append(search, prior, made, recorded=True),
        outcome.status,
        canonical_bytes(summary).decode(),
        derive(floor, *(supplied.restrictions for supplied in made.bounds.values())),
    )


@dataclass(frozen=True)
class Assessed:
    """The immutable held-back head and its compact, restricted evidence."""

    package: str
    summary: str
    restrictions: Restrictions


def _assessment_request(
    lineage: UUID,
    rules_digest: str,
    head: str | None,
    trials: tuple[tuple[UUID, UUID], ...],
    *,
    adjustment_trials: int,
) -> bytes:
    if type(adjustment_trials) is not int or adjustment_trials < 1:
        raise ReferencePipelineError("adjustment_trials must be a positive integer")
    return canonical_bytes(
        {
            "schema_version": 1,
            "lineage": str(lineage),
            "rules": rules_digest,
            "head": head,
            "trials": [[str(source), str(trial)] for source, trial in trials],
            "adjustment_trials": adjustment_trials,
        }
    )


def assessment_reference(
    lineage: UUID,
    rules_digest: str,
    head: str | None,
    trials: tuple[tuple[UUID, UUID], ...],
    *,
    adjustment_trials: int,
) -> str:
    """Identify one frozen held-back request, including its ordered measurement IDs."""
    return hashlib.sha256(
        _assessment_request(
            lineage, rules_digest, head, trials, adjustment_trials=adjustment_trials
        )
    ).hexdigest()


def recover_assessment(search: Search, recovery: str) -> Assessed | None:
    """Read the longest atomically published prefix of one frozen held-back request."""
    parse_digest(recovery, "assessment reference")
    name = f"held-back-{recovery}"
    recovered, previous = None, None
    bases = {}
    for path in (search.destination / "unselected").glob("*"):
        if re.fullmatch(r"[0-9a-f]{64}", path.name) is None:
            continue
        if f'"name":"{name}"'.encode() not in (path / "manifest.json").read_bytes():
            continue
        package = open_package(path)
        receipt = next((item for item in package.artifact_records if item.name == name), None)
        if receipt is None:
            continue
        value = parse_fields(
            canonical_json(package.read_artifact(name), "Held-back receipt"),
            "request material",
            "held-back receipt",
        )
        request = parse_fields(
            value["request"],
            "schema_version lineage rules head trials adjustment_trials",
            "held-back request",
        )
        pairs = tuple(
            (UUID(source), UUID(trial))
            for source, trial in parse_array(request["trials"], "held-back pairs")
        )
        canonical = _assessment_request(
            UUID(request["lineage"]),
            request["rules"],
            request["head"],
            pairs,
            adjustment_trials=request["adjustment_trials"],
        )
        if (
            canonical != canonical_bytes(request)
            or hashlib.sha256(canonical).hexdigest() != recovery
            or request["lineage"] != str(search.lineage)
            or request["rules"] != search.rules.digest
            or receipt.role != "held-back-assessment"
        ):
            raise ReferencePipelineError("Held-back receipt does not match its frozen request")
        head = request["head"]
        if head not in bases:
            bases[head] = None if head is None else open_lineage(search, head)
        base = bases[head]
        originals = () if base is None else base.record.trials
        measured = package.record.trials[len(originals) :]
        sources = {item.trial_id: item for item in originals}
        material = load_material(value["material"])
        summaries = parse_array(json.loads(material.text), "held-back summary")
        if (
            package.record.trial_id is not None
            or package.record.trials[: len(originals)] != originals
            or (
                base is not None
                and package.record.exposures[: len(base.record.exposures)] != base.record.exposures
            )
            or not 0 < len(measured) == len(summaries) <= len(pairs)
            or len({source for source, _ in pairs}) != len(pairs)
            or len({trial for _, trial in pairs}) != len(pairs)
            or any(source not in sources or trial in sources for source, trial in pairs)
        ):
            raise ReferencePipelineError(
                "Held-back package does not extend the frozen search prefix"
            )
        for (source, identity), trial, summary in zip(
            pairs[: len(measured)], measured, summaries, strict=True
        ):
            original = sources[source]
            if (
                trial.trial_id != identity
                or trial.lineage_id != search.lineage
                or original.lineage_id != search.lineage
                or trial.specification.signal_profile != original.specification.signal_profile
                or trial.specification.data_contract_digest != search.rules.held_back.digest
                or trial.links
                != (
                    LineageLink(
                        source,
                        original.specification.spec_id,
                        "held_back_of",
                        "Frozen shortlist candidate",
                    ),
                )
                or len(trial.attempts) != 1
            ):
                raise ReferencePipelineError("Held-back measurement does not match its source")
            outcome = trial.attempts[0].outcome
            evaluations = [
                package.read_artifact(item.name)
                for item in package.artifact_records
                if item.name in outcome.result_artifacts and item.role == EVALUATION_ROLE
            ]
            if len(evaluations) > 1:
                raise ReferencePipelineError("Held-back measurement has conflicting evaluations")
            result = None if not evaluations else decode_evaluation(evaluations[0])
            if isinstance(result, Evaluation) and result.trials != request["adjustment_trials"]:
                raise ReferencePipelineError(
                    "Held-back evaluation changed the frozen candidate budget"
                )
            if summary != {
                "source": str(source),
                "trial": str(identity),
                "status": outcome.status,
                "evaluation": None if not evaluations else json.loads(evaluations[0]),
            }:
                raise ReferencePipelineError(
                    "Held-back summary does not match its saved evaluation"
                )
        if previous is not None:
            common = min(len(previous.record.trials), len(package.record.trials))
            if previous.record.trials[:common] != package.record.trials[:common] or (
                len(previous.record.trials) == len(package.record.trials)
                and previous.digest != package.digest
            ):
                raise ReferencePipelineError("Conflicting published held-back prefixes")
        if previous is None or len(package.record.trials) > len(previous.record.trials):
            previous = package
            recovered = Assessed(package.digest, material.text, material.restrictions)
    return recovered


def assess(
    search: Search,
    context: RunContext,
    head: str | None,
    trials: tuple[tuple[UUID, UUID], ...],
    *,
    adjustment_trials: int,
    stop: Callable[[], bool] = lambda: False,
) -> Assessed:
    """Evaluate a frozen shortlist once and append new measurements to its search lineage.

    Args:
        search (Search): Frozen campaign search and package destination.
        context (RunContext): Campaign authority, narrowed to the held-back cutoff here.
        head (str | None): Frozen search package, absent only for an empty shortlist.
        trials (tuple): Source and new held-back UUID pairs minted by the workflow.
        adjustment_trials (int): Frozen candidate budget used for every measurement.
        stop (Callable): Cancellation condition checked before each candidate.

    Returns:
        (Assessed) The unselected package digest and evidence linking source and held-back IDs.
    """
    request = _assessment_request(
        search.lineage, search.rules.digest, head, trials, adjustment_trials=adjustment_trials
    )
    recovery = hashlib.sha256(request).hexdigest()
    prior = None if head is None else open_lineage(search, head)
    originals = {} if prior is None else {item.trial_id: item for item in prior.record.trials}
    sources, identities = zip(*trials, strict=True) if trials else ((), ())
    if (
        len(set(sources)) != len(sources)
        or len(set(identities)) != len(identities)
        or any(source not in originals for source in sources)
        or any(identity in originals for identity in identities)
    ):
        raise ReferencePipelineError(
            "The shortlist needs unique known sources and new held-back IDs"
        )
    authority = context.narrow(cutoff=search.rules.held_back.cutoff)
    floor = Restrictions(
        available_at=authority.cutoff, labels=frozenset(), approved_for=authority.providers
    )
    restrictions, summaries = floor, []
    programs = {
        item.digest: item.name
        for item in (() if prior is None else prior.artifact_records)
        if item.role == "program"
    }
    selected = []
    for source, identity in trials:
        original = originals[source]
        profile = original.specification.signal_profile
        if original.lineage_id != search.lineage or not profile.startswith(f"{SEARCH_PROFILE} "):
            raise ReferencePipelineError(
                "The shortlist must name this campaign's search candidates"
            )
        program = prior.read_artifact(programs[profile.removeprefix(f"{SEARCH_PROFILE} ")])
        selected.append((source, identity, original, profile, program))
    for source, identity, original, profile, program in selected:
        if stop():
            break
        started = datetime.now(UTC)
        artifact, candidate = _candidate(search, program)
        if isinstance(candidate, str):
            reason = signals.Rejected(candidate)
            examined = _unrun("rejected", f"Rejected ({candidate}): {reason.message}")
        elif FACTORS not in search.readable:
            examined = _unrun("failed", f"The search cannot read {FACTORS}; nothing was read")
        else:
            case = Case(
                profile,
                original.specification.question,
                candidate,
                signals.decide,
                None,
                frozenset(),
            )
            examined = _examine(
                case, search.rules.held_back, search.catalog, search.store, authority
            )
        if isinstance(examined, Refused):
            raise ReferencePipelineError(f"A held-back read was refused: {examined.message}")
        made = _trial(
            examined,
            search.rules.held_back,
            authority,
            catalog=search.catalog,
            store=search.store,
            lock=search.lock,
            question=original.specification.question,
            signal_profile=profile,
            entrypoint="quant_research.reference:assess",
            program=(artifact,),
            trial=identity,
            lineage=search.lineage,
            history=prior.record.trials,
            started=started,
            adjustment_trials=adjustment_trials,
        )
        if isinstance(made, Refused):
            raise ReferencePipelineError(f"A held-back read was refused: {made.message}")
        link = LineageLink(
            source, original.specification.spec_id, "held_back_of", "Frozen shortlist candidate"
        )
        made = replace(made, trial=replace(made.trial, links=(link,)))
        outcome = made.trial.attempts[0].outcome
        summaries.append(
            {
                "source": str(source),
                "trial": str(identity),
                "status": outcome.status,
                "evaluation": None if made.evaluation is None else json.loads(made.evaluation),
            }
        )
        restrictions = derive(restrictions, *(item.restrictions for item in made.bounds.values()))
        receipt = ArtifactInput(
            f"held-back-{recovery}",
            canonical_bytes(
                {
                    "request": json.loads(request),
                    "material": dump_material(
                        Material(canonical_bytes(summaries).decode(), restrictions)
                    ),
                }
            ),
            "application/json",
            "held-back-assessment",
        )
        prior = _append(
            search, prior, replace(made, artifacts=(*made.artifacts, receipt)), recorded=True
        )
    return Assessed(
        "" if prior is None else prior.digest, canonical_bytes(summaries).decode(), restrictions
    )


def forfeit(
    search: Search,
    head: str | None,
    trial: UUID,
    program: bytes,
    started: datetime,
    ended: datetime,
) -> str:
    """Append a candidate whose experiment ended without a result to the lineage as failed.

    The trial records no read and holds the workflow's start and end times, so a crashed or
    timed-out candidate still counts, as D015 counts failed trials. Its run may have read data
    before it ended, so the lineage's disclosure coverage is unknown from this head on.

    Args:
        search (Search): The campaign's search, from its Authority.
        head (str | None): Digest of the head the workflow last recorded, None before the
            first trial.
        trial (UUID): The ID the workflow gave the candidate.
        program (bytes): The move's arguments exactly as the model sent them.
        started (datetime): When the workflow scheduled the experiment.
        ended (datetime): When the workflow saw it end.

    Returns:
        (str) The new head's digest.

    Raises:
        PackageError: If the head cannot be opened or the new head cannot be saved.
    """
    prior = None if head is None else open_lineage(search, head)
    identity, _ = _candidate(search, program)
    made = _searched(search, prior, _unrun("failed", UNFINISHED), identity, trial, started, ended)
    return _append(search, prior, made, recorded=False).digest


def open_lineage(search: Search, head: str) -> VerifiedPackage:
    """Open the lineage package whose digest a campaign's loop recorded as its head.

    The one place that knows save_package keeps a package that selects no trial under
    unselected/<digest>.
    """
    return open_package(search.destination / "unselected" / head)


def _searched(
    search: Search,
    prior: VerifiedPackage | None,
    examined: _Examined,
    identity: ArtifactInput,
    trial: UUID,
    started: datetime,
    finished: datetime | None = None,
) -> _Built | Refused:
    """Record a search candidate as the trial after prior's, on the search window."""
    return _trial(
        examined,
        search.rules.search,
        search.context,
        catalog=search.catalog,
        store=search.store,
        lock=search.lock,
        question=search.question,
        signal_profile=_signal_profile(identity),
        entrypoint=ENTRYPOINT,
        program=(identity,),
        trial=trial,
        lineage=search.lineage,
        history=() if prior is None else prior.record.trials,
        started=started,
        finished=finished,
    )


def _append(
    search: Search, prior: VerifiedPackage | None, made: _Built, recorded: bool
) -> VerifiedPackage:
    """Save prior's trials, exposures, and artifacts with made's, as a package selecting none.

    recorded is whether made's exposures cover what its run read. Once one trial's do not, the
    lineage's coverage is unknown.
    """
    trials, exposures, artifacts = (
        ((), (), ())
        if prior is None
        else (prior.record.trials, prior.record.exposures, prior.artifacts())
    )
    merged = {item.name: item for item in (*artifacts, *made.artifacts)}
    known = recorded and (prior is None or prior.record.disclosure_coverage == "declared_inputs")
    record = PackageRecord(
        None,
        (*trials, made.trial),
        (*exposures, *made.exposures),
        "declared_inputs" if known else "unknown",
    )
    return save_package(search.destination, record, tuple(merged.values()))


def _candidate(search: Search, program: bytes) -> tuple[ArtifactInput, Program | signals.Reason]:
    """Return a candidate's program artifact, and its condition or why the search refuses it.

    The artifact holds signals.canonical of a parsed program and the request's own bytes
    otherwise, so its SHA-256 is signals.digest for a program. parse accepts a number, which
    allocate cannot hold, so the search rejects one here.
    """
    parsed = signals.parse(program, search.catalog, search.universe)
    if isinstance(parsed, signals.Rejected):
        return _artifact("program", program, "program", "text/plain"), parsed.reason
    identity = _artifact("program", signals.canonical(parsed), "program")
    return identity, parsed if isinstance(parsed, _CONDITIONS) else "wrong_type"


def _signal_profile(identity: ArtifactInput) -> str:
    return f"{SEARCH_PROFILE} {hashlib.sha256(identity.data).hexdigest()}"


def _unrun(status: Status, detail: str) -> _Examined:
    """Return the finding of a trial that read nothing."""
    return _Examined((), {}, _Finding(status, detail))
