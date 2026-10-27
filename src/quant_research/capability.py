from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import asdict, dataclass, field
from functools import partial
from itertools import product
from pathlib import Path
from time import monotonic
from typing import Literal, get_args
from uuid import UUID, uuid4

from quant_research.packages import Status
from quant_research.records import (
    append_log,
    canonical_bytes,
    parse_array,
    parse_digest,
    parse_fields,
    parse_text,
    read_log,
    unique_keys,
)

type TaskKind = Literal["known_answer", "null", "flawed", "paper_reproduction"]
type Procedure = Callable[[dict], Response]
type AsyncProcedure = Callable[[dict], Awaitable[ProcedureResult]]

_STATUSES = get_args(Status.__value__)
_ACCEPTED: dict[TaskKind, tuple[Status, ...]] = {
    "known_answer": ("successful",),
    "paper_reproduction": ("successful",),
    "null": ("null", "inconclusive"),
    "flawed": ("rejected",),
}
# run and disclose enumerate the whole grid, so a count no evaluation could run fails at freeze.
_MAX_REPEATS = 1000


class CapabilityError(ValueError):
    """Report an invalid task file, answer key, or evaluation request."""


def _identity(value: object, label: str) -> str:
    try:
        return hashlib.sha256(canonical_bytes(value)).hexdigest()
    except (TypeError, ValueError) as error:
        raise CapabilityError(f"{label} must be canonical JSON: {error}") from error


@dataclass(frozen=True)
class TaskSet:
    version: str
    tasks: dict[str, dict]

    @property
    def set_id(self) -> str:
        return _identity(asdict(self), "Task set")

    @property
    def digests(self) -> dict[str, str]:
        return {name: _identity(inputs, f"Task {name!r}") for name, inputs in self.tasks.items()}


@dataclass(frozen=True)
class Answer:
    kind: TaskKind
    estimates: dict[str, float]
    tolerance: float


@dataclass(frozen=True)
class AnswerKey:
    task_set_id: str
    answers: dict[str, Answer]

    @property
    def key_id(self) -> str:
        return _identity(asdict(self), "Answer key")


@dataclass(frozen=True)
class Configuration:
    name: str
    implementation: str
    model_versions: tuple[str, ...]
    starting_memory: str
    resources: str
    human_steering: str

    @property
    def config_id(self) -> str:
        return _identity(asdict(self), "Configuration")


@dataclass(frozen=True)
class Comparison:
    task_set_id: str
    answer_key_id: str
    tasks: dict[str, str]
    configurations: tuple[Configuration, ...]
    repeats: int

    @property
    def comparison_id(self) -> str:
        return _identity(asdict(self), "Comparison")


@dataclass(frozen=True)
class Response:
    status: Status
    estimates: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Run:
    configuration: str
    task: str
    repeat: int
    response: Response


@dataclass(frozen=True)
class Results:
    comparison: Comparison
    runs: tuple[Run, ...]


@dataclass(frozen=True)
class Score:
    configuration: str
    task: str
    repeat: int
    response: Response
    success: bool


@dataclass(frozen=True)
class Development:
    comparison: Comparison
    known_answer_success_share: float

    @property
    def development_id(self) -> str:
        return _identity(asdict(self), "Development")


@dataclass(frozen=True)
class ProcedureResult:
    response: Response
    evidence: dict


@dataclass(frozen=True)
class DevelopmentResults:
    development: Development
    execution_id: str
    results: Results
    evidence: dict[str, dict]


@dataclass(frozen=True)
class DevelopmentDisclosure:
    execution_id: str
    scores: tuple[Score, ...]
    known_answer_shares: dict[str, dict[str, float]]
    accepted: dict[str, dict[str, bool]]
    passed: bool


@dataclass
class _History:
    """Fold ledger events in order; live operations pass the same transitions before writing."""

    frozen: dict[str, Comparison] = field(default_factory=dict)
    exposed: set[str] = field(default_factory=set)
    bound: dict[str, str] = field(default_factory=dict)
    answer_keys: dict[str, str] = field(default_factory=dict)
    developments: dict[str, Development] = field(default_factory=dict)
    executions: dict[str, str] = field(default_factory=dict)
    disclosed: set[str] = field(default_factory=set)

    def apply(self, value: object) -> None:
        if isinstance(value, dict) and value.get("schema_version") == 2:
            event = parse_fields(value, "schema_version action value", "development event")
            if type(event["schema_version"]) is not int:
                raise CapabilityError("Unsupported schema version")
            match event["action"]:
                case "freeze_development":
                    if not self.freeze_development(_development(event["value"])):
                        raise CapabilityError("A development is frozen twice")
                case "begin_development" | "disclose_development":
                    fields = "development execution"
                    if event["action"] == "disclose_development":
                        fields += " results"
                    item = parse_fields(event["value"], fields, "development execution")
                    identity = parse_digest(item["development"])
                    execution = _execution(item["execution"])
                    if event["action"] == "begin_development":
                        self.begin_development(identity, execution)
                    else:
                        parse_digest(item["results"])
                        self.disclose_development(identity, execution)
                case _:
                    raise CapabilityError("Unknown development event")
            return
        if isinstance(value, dict) and "freeze" in value:
            if not self.freeze(_comparison(_versioned(value, "freeze")["freeze"])):
                raise CapabilityError("A comparison is frozen twice")
        else:
            event = _versioned(value, "disclose results")
            parse_digest(event["results"])
            self.disclose(parse_digest(event["disclose"]))

    def freeze(self, comparison: Comparison) -> bool:
        if not self.exposed.isdisjoint(comparison.tasks.values()):
            raise CapabilityError("A comparison was frozen over tasks already disclosed")
        key = comparison.answer_key_id
        if any(self.answer_keys.get(task, key) != key for task in comparison.tasks.values()):
            raise CapabilityError("A held-out task is frozen under another answer key")
        self._check_bindings(comparison.configurations)
        new = comparison.comparison_id not in self.frozen
        self.frozen[comparison.comparison_id] = comparison
        self.answer_keys |= dict.fromkeys(comparison.tasks.values(), key)
        return new

    def disclose(self, identity: str) -> Comparison:
        if identity not in self.frozen:
            raise CapabilityError("Results do not belong to a frozen comparison")
        comparison = self.frozen[identity]
        if not self.exposed.isdisjoint(comparison.tasks.values()):
            raise CapabilityError("Tasks in this comparison were already disclosed")
        self._check_bindings(comparison.configurations)
        self.exposed |= set(comparison.tasks.values())
        self.bound |= {item.name: item.config_id for item in comparison.configurations}
        return comparison

    def _check_bindings(self, configurations: tuple[Configuration, ...]) -> None:
        for item in configurations:
            if self.bound.get(item.name, item.config_id) != item.config_id:
                raise CapabilityError(f"Configuration {item.name!r} changed after disclosure")

    def freeze_development(self, development: Development) -> bool:
        comparison = development.comparison
        key = comparison.answer_key_id
        if any(self.answer_keys.get(task, key) != key for task in comparison.tasks.values()):
            raise CapabilityError("A task is frozen under another answer key")
        self._check_bindings(comparison.configurations)
        new = development.development_id not in self.developments
        self.developments[development.development_id] = development
        self.answer_keys |= dict.fromkeys(comparison.tasks.values(), key)
        return new

    def begin_development(self, identity: str, execution: str) -> None:
        if identity not in self.developments:
            raise CapabilityError("Execution does not belong to a frozen development")
        if execution in self.executions:
            raise CapabilityError("A development execution began twice")
        comparison = self.developments[identity].comparison
        self._check_bindings(comparison.configurations)
        self.executions[execution] = identity
        self.exposed |= set(comparison.tasks.values())

    def disclose_development(self, identity: str, execution: str) -> Development:
        if self.executions.get(execution) != identity:
            raise CapabilityError("Results do not belong to a begun development execution")
        if execution in self.disclosed:
            raise CapabilityError("A development execution was already disclosed")
        development = self.developments[identity]
        self._check_bindings(development.comparison.configurations)
        self.bound |= {item.name: item.config_id for item in development.comparison.configurations}
        self.disclosed.add(execution)
        return development


class ExposureLedger:
    """Freeze comparisons, disclose each one once, and retire its tasks, in an append-only store.

    Event i is the canonical JSON file i.json. Every operation replays all events first, so
    ledgers opened on one directory never act on stale state.
    """

    def __init__(self, store: Path) -> None:
        """Open the ledger kept in an existing directory and verify every recorded event.

        Args:
            store (Path): Existing directory that holds the numbered event files.
        """
        self._store = store
        self._replay()

    def freeze(
        self,
        tasks: TaskSet,
        answer_key_id: str,
        configurations: tuple[Configuration, ...],
        repeats: int,
    ) -> Comparison:
        """Fix the answer key, configurations, and held-out tasks before any outcome.

        Args:
            tasks (TaskSet): Task-set version whose undisclosed tasks the comparison covers.
            answer_key_id (str): Digest of the sealed answer key that disclosure must use.
            configurations (tuple): Uniquely named configurations to compare.
            repeats (int): Runs per configuration and task.

        Returns:
            (Comparison) The frozen comparison, equal for an identical request.
        """

        def attempt() -> tuple[int, Comparison, bytes | None]:
            count, history = self._replay()
            held_out = {
                name: digest
                for name, digest in sorted(tasks.digests.items())
                if digest not in history.exposed
            }
            if not held_out:
                raise CapabilityError("No held-out tasks remain in this task set")
            proposed = Comparison(tasks.set_id, answer_key_id, held_out, configurations, repeats)
            try:
                comparison = _comparison(json.loads(canonical_bytes(asdict(proposed))))
            except (TypeError, ValueError) as error:
                raise CapabilityError(f"Invalid comparison: {error}") from error
            record = canonical_bytes({"schema_version": 1, "freeze": asdict(comparison)})
            return count, comparison, record if history.freeze(comparison) else None

        return self._append(attempt)

    def disclose(self, results: Results, answers: AnswerKey) -> tuple[Score, ...]:
        """Score every run of a frozen comparison once and retire the comparison's tasks.

        Args:
            results (Results): Complete runs of one frozen comparison.
            answers (AnswerKey): The sealed answer key whose digest the comparison committed.

        Returns:
            (tuple) One Score per configuration, task, and repeat.
        """
        identity = results.comparison.comparison_id
        # A results digest makes a concurrent disclosure of other results conflict.
        digest = _identity(asdict(results), "Results")
        record = canonical_bytes({"schema_version": 1, "disclose": identity, "results": digest})

        def attempt() -> tuple[int, tuple[Score, ...], bytes]:
            count, history = self._replay()
            return count, _score(history.disclose(identity), results, answers), record

        return self._append(attempt)

    def _replay(self) -> tuple[int, _History]:
        history, count = _History(), 0
        try:
            for value in read_log(self._store, "Ledger event", "Event files"):
                history.apply(value)
                count += 1
        except (OSError, ValueError, RecursionError) as error:
            raise CapabilityError(f"Cannot read exposure ledger {self._store}: {error}") from error
        return count, history

    def freeze_development(
        self,
        tasks: TaskSet,
        answer_key_id: str,
        configurations: tuple[Configuration, ...],
        repeats: int,
        *,
        known_answer_success_share: float,
    ) -> Development:
        """Freeze a reusable development definition, including already exposed inputs."""
        proposed = Development(
            Comparison(
                tasks.set_id,
                answer_key_id,
                dict(sorted(tasks.digests.items())),
                configurations,
                repeats,
            ),
            known_answer_success_share,
        )
        development = _development(json.loads(canonical_bytes(asdict(proposed))))

        def attempt() -> tuple[int, Development, bytes | None]:
            count, history = self._replay()
            record = _development_event("freeze_development", asdict(development))
            return count, development, record if history.freeze_development(development) else None

        return self._append(attempt)

    def _begin_development(self, development: Development, execution: str) -> None:
        def attempt() -> tuple[int, None, bytes]:
            count, history = self._replay()
            history.begin_development(development.development_id, _execution(execution))
            return (
                count,
                None,
                _development_event(
                    "begin_development",
                    {"development": development.development_id, "execution": execution},
                ),
            )

        self._append(attempt)

    def disclose_development(
        self, results: DevelopmentResults, answers: AnswerKey
    ) -> DevelopmentDisclosure:
        """Disclose a complete begun execution once and retain acceptance for every case."""
        identity = results.development.development_id
        execution = _execution(results.execution_id)
        digest = _identity(asdict(results), "Development results")

        def attempt() -> tuple[int, DevelopmentDisclosure, bytes]:
            count, history = self._replay()
            development = history.disclose_development(identity, execution)
            comparison = development.comparison
            if results.results.comparison != comparison:
                raise CapabilityError("Results do not match the development comparison")
            expected = {_cell(*cell) for cell in _grid(comparison)}
            if results.evidence.keys() != expected:
                raise CapabilityError("Evidence must cover every development cell exactly once")
            for value in results.evidence.values():
                canonical_bytes(_plain(value, "evidence"))
            runs = tuple(
                Run(item.configuration, item.task, item.repeat, _response(item.response))
                for item in results.results.runs
            )
            scores = _score(comparison, Results(comparison, runs), answers)
            successes = Counter((item.configuration, item.task) for item in scores if item.success)
            shares, accepted = {}, {}
            for configuration in comparison.configurations:
                shares[configuration.name], accepted[configuration.name] = {}, {}
                for task in comparison.tasks:
                    known = answers.answers[task].kind in {"known_answer", "paper_reproduction"}
                    share = successes[configuration.name, task] / comparison.repeats
                    if known:
                        shares[configuration.name][task] = share
                    accepted[configuration.name][task] = (
                        share >= development.known_answer_success_share
                        if known
                        else successes[configuration.name, task] == comparison.repeats
                    )
            disclosure = DevelopmentDisclosure(
                execution,
                scores,
                shares,
                accepted,
                all(passed for cases in accepted.values() for passed in cases.values()),
            )
            return (
                count,
                disclosure,
                _development_event(
                    "disclose_development",
                    {
                        "development": identity,
                        "execution": execution,
                        "results": digest,
                    },
                ),
            )

        return self._append(attempt)

    def _append[T](self, attempt: Callable[[], tuple[int, T, bytes | None]]) -> T:
        try:
            return append_log(self._store, attempt)
        except OSError as error:
            raise CapabilityError(f"Cannot write exposure ledger {self._store}: {error}") from error


def _comparison(raw: object) -> Comparison:
    value = parse_fields(
        raw, "task_set_id answer_key_id tasks configurations repeats", "comparison"
    )

    if not isinstance(value["tasks"], dict) or not value["tasks"]:
        raise CapabilityError("A comparison needs one or more held-out tasks")
    configurations = tuple(
        _configuration(item) for item in parse_array(value["configurations"], "configurations")
    )
    if not configurations or len({item.name for item in configurations}) != len(configurations):
        raise CapabilityError("A comparison needs one or more uniquely named configurations")
    if type(value["repeats"]) is not int or not 1 <= value["repeats"] <= _MAX_REPEATS:
        raise CapabilityError(f"Repeats must be a positive integer of at most {_MAX_REPEATS}")
    return Comparison(
        parse_digest(value["task_set_id"]),
        parse_digest(value["answer_key_id"]),
        {name: parse_digest(digest) for name, digest in value["tasks"].items()},
        configurations,
        value["repeats"],
    )


def _development(raw: object) -> Development:
    value = parse_fields(raw, "comparison known_answer_success_share", "development")
    share = value["known_answer_success_share"]
    if type(share) not in (int, float) or not math.isfinite(share) or not 0 < share <= 1:
        raise CapabilityError(
            "Known-answer success share must be greater than zero and at most one"
        )
    return Development(_comparison(value["comparison"]), float(share))


def _execution(value: object) -> str:
    if type(value) is not str or str(UUID(value)) != value:
        raise CapabilityError("Execution must be a canonical UUID")
    return value


def _development_event(action: str, value: dict) -> bytes:
    return canonical_bytes({"schema_version": 2, "action": action, "value": value})


def _cell(configuration: str, task: str, repeat: int) -> str:
    return canonical_bytes([configuration, task, repeat]).decode()


def _configuration(raw: object) -> Configuration:
    value = parse_fields(
        raw,
        "name implementation model_versions starting_memory resources human_steering",
        "configuration",
    )
    return Configuration(
        parse_text(value["name"], "name"),
        parse_text(value["implementation"], "implementation"),
        tuple(
            parse_text(item, "model version")
            for item in parse_array(value["model_versions"], "model_versions")
        ),
        parse_text(value["starting_memory"], "starting_memory"),
        parse_text(value["resources"], "resources"),
        parse_text(value["human_steering"], "human_steering"),
    )


def _score(comparison: Comparison, results: Results, answers: AnswerKey) -> tuple[Score, ...]:
    if answers.key_id != comparison.answer_key_id:
        raise CapabilityError("Answer key is not the one committed at freeze")
    if answers.task_set_id != comparison.task_set_id or not (
        comparison.tasks.keys() <= answers.answers.keys()
    ):
        raise CapabilityError("Answer key does not cover the comparison's task set")
    cells = [(item.configuration, item.task, item.repeat) for item in results.runs]
    expected = set(_grid(comparison))
    if len(cells) != len(expected) or set(cells) != expected:
        raise CapabilityError("Results must hold each configuration, task, and repeat exactly once")
    scores = []
    for item in results.runs:
        answer = answers.answers[item.task]
        reported = item.response.estimates
        success = item.response.status in _ACCEPTED[answer.kind] and all(
            key in reported and abs(reported[key] - value) <= answer.tolerance
            for key, value in answer.estimates.items()
        )
        scores.append(Score(item.configuration, item.task, item.repeat, item.response, success))
    return tuple(scores)


def _grid(comparison: Comparison) -> Iterator[tuple[str, str, int]]:
    return product(
        (item.name for item in comparison.configurations),
        comparison.tasks,
        range(comparison.repeats),
    )


def run(comparison: Comparison, tasks: TaskSet, procedures: dict[str, Procedure]) -> Results:
    """Run each configuration's procedure on every task and repeat of a comparison.

    Args:
        comparison (Comparison): Frozen comparison to execute.
        tasks (TaskSet): Task-set version the comparison was frozen from.
        procedures (dict): One callable per configuration name, called with task inputs only.

    Returns:
        (Results) One Run per configuration, task, and repeat, produced without answers.
    """
    if tasks.set_id != comparison.task_set_id:
        raise CapabilityError("Task set does not match the comparison")
    if procedures.keys() != {item.name for item in comparison.configurations}:
        raise CapabilityError("Supply exactly one procedure per configuration")
    runs = []
    for name, task, repeat in _grid(comparison):
        # A fresh JSON copy per call keeps one run's edits out of later runs.
        returned = procedures[name](json.loads(canonical_bytes(tasks.tasks[task])))
        try:
            response = _response(returned)
        except (TypeError, ValueError, OverflowError) as error:
            raise CapabilityError(
                f"Procedure {name!r} returned an invalid response: {error}"
            ) from error
        runs.append(Run(name, task, repeat, response))
    return Results(comparison, tuple(runs))


async def run_development(
    ledger: ExposureLedger,
    development: Development,
    tasks: TaskSet,
    procedures: dict[str, AsyncProcedure],
) -> DevelopmentResults:
    """Begin durable exposure before delivering fresh inputs to each async procedure."""
    development = _development(json.loads(canonical_bytes(asdict(development))))
    comparison = development.comparison
    if tasks.set_id != comparison.task_set_id or tasks.digests != comparison.tasks:
        raise CapabilityError("Task set does not match the development")
    if procedures.keys() != {item.name for item in comparison.configurations}:
        raise CapabilityError("Supply exactly one procedure per configuration")
    execution, runs, evidence = str(uuid4()), [], {}
    ledger._begin_development(development, execution)
    for name, task, repeat in _grid(comparison):
        returned = await procedures[name](json.loads(canonical_bytes(tasks.tasks[task])))
        try:
            if type(returned) is not ProcedureResult:
                raise TypeError("expected a ProcedureResult")
            response = _response(returned.response)
            saved = json.loads(canonical_bytes(_plain(returned.evidence, "evidence")))
        except (TypeError, ValueError, OverflowError) as error:
            raise CapabilityError(
                f"Procedure {name!r} returned invalid development evidence: {error}"
            ) from error
        runs.append(Run(name, task, repeat, response))
        evidence[_cell(name, task, repeat)] = saved
    return DevelopmentResults(development, execution, Results(comparison, tuple(runs)), evidence)


class ResearchProcedure:
    """Evaluate one supplied candidate through a fresh archived ResearchLoop campaign."""

    def __init__(
        self,
        client,
        prepare: Callable,
        gateway_factory: Callable,
        artifacts: Path,
        *,
        max_seconds: float = 1200,
    ) -> None:
        self.client, self.prepare, self.gateway_factory = client, prepare, gateway_factory
        self.artifacts, self.max_seconds = artifacts, max_seconds

    async def __call__(self, inputs: dict) -> ProcedureResult:
        from quant_research import campaigns, research
        from quant_research.packages import open_package

        started = monotonic()
        directory = self.artifacts / str(uuid4())
        directory.mkdir(parents=True)
        evidence = {
            "directory": str(directory),
            "input_digest": _identity(inputs, "Inputs"),
            "calls_path": str(directory / "calls"),
            "archive_path": str(directory / "archive"),
        }
        report, package, error, stopped = None, None, None, None
        if self.max_seconds <= 0:
            evidence |= {"elapsed_seconds": monotonic() - started, "failure": "deadline"}
            (directory / "evidence.json").write_bytes(canonical_bytes(evidence))
            return ProcedureResult(Response("failed"), evidence)
        try:
            revision, setup = self.prepare(inputs, directory)
            (directory / "revision.json").write_bytes(campaigns.file_bytes(revision))
            calls = directory / "calls"
            calls.mkdir()
            gateway = self.gateway_factory(calls)
            archive = research.CampaignArchive(directory / "archive", calls)
            queue = str(uuid4())
            async with research.worker(
                self.client,
                queue,
                gateway,
                partial(campaigns.resolve, setup=setup),
                archive=archive,
            ):
                if monotonic() - started >= self.max_seconds:
                    raise TimeoutError
                handle = await research.start(self.client, revision, queue)
                evidence |= {
                    "campaign": str(revision.campaign_id),
                    "revision": revision.revision_id,
                }
                result = asyncio.create_task(handle.result())
                while not result.done():
                    remaining = self.max_seconds - (monotonic() - started)
                    if remaining <= 0 or any(
                        item.outcome == "rate_limited" for item in gateway.records
                    ):
                        stopped = "deadline" if remaining <= 0 else "rate_limited"
                        await research.control(handle, "cancel", actor="capability", reason=stopped)
                        break
                    await asyncio.wait({result}, timeout=min(0.2, remaining))
                try:
                    await result
                    report = await research.record(handle)
                except Exception as failure:
                    error = {"type": type(failure).__name__}
                    report = await research.status(handle)
                (directory / "report.json").write_bytes(canonical_bytes(asdict(report)))
                history = await handle.fetch_history()
                (directory / "history.json").write_text(history.to_json())
                evidence["calls"] = [
                    {
                        "call_id": str(item.call_id),
                        "outcome": item.outcome,
                        "served": None if item.served is None else asdict(item.served),
                        "model": {"provider": item.model.provider, "model": item.model.model},
                        "model_calls": item.usage.model_calls,
                        "unreported": sorted(item.usage.unreported),
                    }
                    for item in gateway.records
                ]
                receipts = sorted((directory / "archive" / "receipts").iterdir())
                evidence["archives"] = []
                for receipt in receipts:
                    digest = parse_digest(receipt.read_text())
                    archived = open_package(
                        directory / "archive" / "packages" / "unselected" / digest
                    )
                    evidence["archives"].append(
                        {"digest": archived.digest, "path": str(archived.path)}
                    )
                if report.package:
                    package = open_package(
                        setup.packages / str(revision.campaign_id) / "unselected" / report.package
                    )
                    evidence["numerical_package"] = {
                        "digest": package.digest,
                        "path": str(package.path),
                    }
                evidence["report"] = str(directory / "report.json")
                evidence["history"] = str(directory / "history.json")
        except Exception as failure:
            error = {"type": type(failure).__name__}
            (directory / "failure.json").write_bytes(canonical_bytes(error))
        elapsed = monotonic() - started
        evidence["elapsed_seconds"] = elapsed
        if stopped or elapsed > self.max_seconds:
            evidence["failure"] = stopped or "deadline"
        if error is not None:
            evidence["error"] = error
        response = Response("failed")
        if (
            report is not None
            and package is not None
            and evidence.get("archives")
            and not (error or evidence.get("failure"))
        ):
            response = _research_response(report, package, canonical_bytes(inputs["candidate"]))
        evidence["response"] = asdict(response)
        (directory / "evidence.json").write_bytes(canonical_bytes(evidence))
        return ProcedureResult(response, evidence)


def _research_response(report, package, supplied_program: bytes) -> Response:
    from quant_research import evaluation, reference
    from quant_research.providers import load_material

    failed = Response("failed")
    assessment = report.assessment
    if report.ending is None or assessment is None or report.unrecorded is not None:
        return failed
    if report.ending.outcome not in {"concluded", "inconclusive"} or report.ending.reason not in {
        "",
        "interpretation",
    }:
        return failed
    try:
        conclusion = parse_fields(
            json.loads(load_material(assessment.conclusion).text, object_pairs_hook=unique_keys),
            "classification citations explanation",
            "conclusion",
        )
        critique = parse_fields(
            json.loads(load_material(assessment.critique).text, object_pairs_hook=unique_keys),
            "decision citations explanation",
            "critique",
        )
        interpreted = assessment.classification
        if critique["decision"] != "accept":
            return failed
        for value in (conclusion, critique):
            parse_text(value["explanation"], "explanation")
            citations = parse_array(value["citations"], "citations")
            if any(type(item) is not str for item in citations) or len(set(citations)) != len(
                citations
            ):
                return failed
        if conclusion["classification"] == "supported" and interpreted != "supported":
            return failed
        attempted = [item.trial for item in report.steps if item.action == "experiment"]
        if len(attempted) != 1:
            return failed
        source = next(
            trial for trial in package.record.trials if str(trial.trial_id) == attempted[0]
        )
        profile = source.specification.signal_profile.split()[-1]
        program = next(
            item
            for item in package.artifact_records
            if item.role == "program" and item.digest == profile
        )
        if canonical_bytes(json.loads(package.read_artifact(program.name))) != canonical_bytes(
            json.loads(supplied_program)
        ):
            return failed
        outcome = source.attempts[-1].outcome
        if outcome.status == "rejected":
            if (
                assessment.trials
                or interpreted != "no_result"
                or conclusion["classification"] != "no_result"
            ):
                return failed
            reason = re.fullmatch(r"Rejected \(([^)]+)\): .+", outcome.detail)
            return (
                Response("rejected")
                if reason and reason[1] in {"negative_lag", "reads_past_vintage"}
                else failed
            )
        measured = {}
        artifacts = {item.name: item for item in package.artifact_records}
        for item in assessment.trials:
            trial = next(
                trial for trial in package.record.trials if str(trial.trial_id) == item.trial
            )
            saved = trial.attempts[-1].outcome
            if (
                item.source != attempted[0]
                or item.status != saved.status
                or item.evaluation is None
            ):
                return failed
            raw = canonical_bytes(json.loads(item.evaluation))
            if not any(
                artifacts[name].role == reference.EVALUATION_ROLE
                and package.read_artifact(name) == raw
                for name in saved.result_artifacts
            ):
                return failed
            numeric = reference.decode_evaluation(raw)
            if not isinstance(numeric, evaluation.Evaluation):
                return failed
            measured[item.trial] = numeric
        if not measured:
            return failed
        if any(
            trial not in measured
            for value in (conclusion, critique)
            for trial in value["citations"]
        ):
            return failed
        if interpreted == "supported":
            supported = [
                measured[trial]
                for trial in conclusion["citations"]
                if measured[trial].verdict == "supported"
            ]
            if len(supported) != 1:
                return failed
            return Response("successful", {"incremental_return": supported[0].incremental.estimate})
        if (
            interpreted == "null"
            and conclusion["classification"] == "null"
            and (
                set(conclusion["citations"]) == set(measured)
                and all(value.verdict == "null" for value in measured.values())
            )
        ):
            return Response("null")
        if interpreted == "inconclusive" and conclusion["classification"] == "inconclusive":
            return Response("inconclusive")
    except (ValueError, TypeError, KeyError, StopIteration, AttributeError):
        return failed
    return failed


def _response(value: object) -> Response:
    # Checking exact built-in types on snapshots makes an accepted response plain data, and the
    # snapshots keep a procedure from editing its recorded response before disclosure.
    fields = _plain(vars(value) if type(value) is Response else None, "a Response")
    if fields.keys() != {"status", "estimates"}:
        raise TypeError("expected a Response")
    status, estimates = fields["status"], _plain(fields["estimates"], "estimates")
    if type(status) is not str or status not in _STATUSES:
        raise ValueError("status must be a known outcome")
    if not all(type(item) is int or type(item) is float for item in estimates.values()):
        raise ValueError("estimates must be numbers")
    canonical_bytes(estimates)
    return Response(status, {key: float(item) for key, item in estimates.items()})


def _plain(value: object, label: str) -> dict:
    pairs = list(value.items()) if type(value) is dict else None
    if pairs is None or not all(type(key) is str for key, _ in pairs):
        raise TypeError(f"expected {label} with text keys")
    return dict(pairs)


def _document(path: Path, fields: str) -> dict:
    value = json.loads(path.read_bytes(), object_pairs_hook=unique_keys)
    # Identities hash canonical JSON, so NaN or lone surrogates fail here, not at freeze.
    canonical_bytes(value)
    return _versioned(value, fields)


def _versioned(value: object, fields: str) -> dict:
    if not isinstance(value, dict) or value.keys() != {"schema_version", *fields.split()}:
        raise CapabilityError(f"Expected fields schema_version {fields}")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise CapabilityError(f"Unsupported schema version {value['schema_version']!r}")
    return value


def _objects(value: object, label: str) -> dict[str, dict]:
    if (
        not isinstance(value, dict)
        or not value
        or not all(isinstance(item, dict) for item in value.values())
    ):
        raise CapabilityError(f"{label} must map one or more names to JSON objects")
    return value


def load_tasks(path: Path) -> TaskSet:
    """Read the task inputs of one task-set version, which procedures may see.

    Args:
        path (Path): Tasks JSON file.

    Returns:
        (TaskSet) Task inputs keyed by neutral task name.
    """
    try:
        value = _document(path, "version tasks")
        if not isinstance(value["version"], str) or not value["version"].strip():
            raise CapabilityError("Task-set version must be nonempty text")
        result = TaskSet(value["version"], _objects(value["tasks"], "tasks"))
    except (OSError, ValueError, RecursionError) as error:
        raise CapabilityError(f"Cannot load tasks {path}: {error}") from error
    return result


def load_answers(path: Path) -> AnswerKey:
    """Read a sealed answer key, which only disclosure needs.

    Args:
        path (Path): Answers JSON file bound to one task-set version.

    Returns:
        (AnswerKey) Each task's kind, expected estimates, and tolerance.
    """
    try:
        value = _document(path, "task_set_id answers")
        task_set_id = value["task_set_id"]
        if not isinstance(task_set_id, str) or not re.fullmatch(r"[0-9a-f]{64}", task_set_id):
            raise CapabilityError(f"Invalid task set ID {task_set_id!r}")
        answers = {}
        for name, raw in _objects(value["answers"], "answers").items():
            if raw.keys() != {"kind", "estimates", "tolerance"}:
                raise CapabilityError(f"Answer {name!r} needs kind, estimates, and tolerance")
            kind, estimates, tolerance = raw["kind"], raw["estimates"], raw["tolerance"]
            if not isinstance(kind, str) or kind not in _ACCEPTED:
                raise CapabilityError(f"Unknown task kind {kind!r}")
            positive = "successful" in _ACCEPTED[kind]
            if not isinstance(estimates, dict) or bool(estimates) != positive:
                raise CapabilityError(
                    f"A {kind} answer must {'carry' if positive else 'omit'} estimates"
                )
            numbers = (*estimates.values(), tolerance)
            if (
                not all(type(item) in (int, float) and math.isfinite(item) for item in numbers)
                or tolerance < 0
            ):
                raise CapabilityError(
                    f"Answer {name!r} needs finite estimates and a nonnegative tolerance"
                )
            answers[name] = Answer(kind, estimates, tolerance)
        result = AnswerKey(task_set_id, answers)
    except (OSError, ValueError, OverflowError, RecursionError) as error:
        raise CapabilityError(f"Cannot load answers {path}: {error}") from error
    return result
