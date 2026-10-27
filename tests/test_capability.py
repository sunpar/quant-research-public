from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import threading
from collections import Counter
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from functools import partial
from itertools import count
from pathlib import Path
from statistics import linear_regression

import pytest

from quant_research import records
from quant_research.capability import (
    AnswerKey,
    CapabilityError,
    Configuration,
    ExposureLedger,
    ProcedureResult,
    ResearchProcedure,
    Response,
    TaskSet,
    load_answers,
    load_tasks,
    run,
    run_development,
)
from quant_research.records import write_once

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "capability"
KINDS = {
    "task-1": "flawed",
    "task-2": "paper_reproduction",
    "task-3": "known_answer",
    "task-4": "null",
}


def reference(inputs: dict, no_effect: str = "null", offset: float = 0.0) -> Response:
    if inputs["signal_lag"] < 1:
        return Response("rejected")
    beta = linear_regression(inputs["signal"], inputs["returns"]).slope
    if abs(beta) < 1e-9:
        return Response(no_effect)
    return Response("successful", {"beta": beta + offset})


def credulous(inputs: dict) -> Response:
    beta = linear_regression(inputs["signal"], inputs["returns"]).slope
    return Response("successful", {"beta": beta})


PROCEDURES = {
    "honest": reference,
    "hedged": partial(reference, no_effect="inconclusive", offset=0.1),
    "credulous": credulous,
}


class EqualsAnything:
    def __eq__(self, other: object) -> bool:
        return True


@dataclass(frozen=True)
class ExtendedResponse(Response):
    extra: int = 0


class FieldlessResponse(Response):
    __dataclass_fields__ = {}


class Uncopyable:
    def __deepcopy__(self, memo: dict) -> Uncopyable:
        raise RuntimeError("cannot copy")

    def __repr__(self) -> str:
        raise RuntimeError("cannot print")


class RaisingDict(dict):
    def keys(self):
        raise KeyError("keys")

    def items(self):
        raise KeyError("items")


def nested(depth: int) -> object:
    value = 1.0
    for _ in range(depth):
        value = [value]
    return value


class LenientType(type):
    __hash__ = type.__hash__

    def __eq__(cls, other: object) -> bool:
        return True


class MisreportedFloat(float, metaclass=LenientType):
    def __float__(self) -> float:
        return 42.0


class RaisingName(str):
    __hash__ = str.__hash__

    def __eq__(self, other: object) -> bool:
        raise RuntimeError("compared")


def replaced_attributes(attributes: dict) -> Response:
    response = Response("successful")
    object.__setattr__(response, "__dict__", attributes)
    return response


FAKE_RESPONSES = {
    "uninitialized": lambda: object.__new__(Response),
    "unprintable-status": lambda: Response(Uncopyable()),
    "subclass-with-field": lambda: ExtendedResponse("successful", {"beta": 1.0}, 3),
    "subclass-without-fields": lambda: FieldlessResponse("successful", {"beta": 1.0}),
    "replaced-attributes": lambda: replaced_attributes(
        RaisingDict(status="successful", estimates={})
    ),
    "subclass-attribute-names": lambda: replaced_attributes(
        {RaisingName("status"): "successful", RaisingName("estimates"): {}}
    ),
    "nested-estimate": lambda: Response("successful", {"beta": nested(5000)}),
    "uncopyable-estimate": lambda: Response("successful", {"beta": Uncopyable()}),
    "dict-subclass-estimates": lambda: Response("successful", RaisingDict(beta=1.0)),
    "surrogate-key": lambda: Response("successful", {"\ud800": 1.0}),
    "boolean-estimate": lambda: Response("successful", {"beta": True}),
    "misreported-float": lambda: Response("successful", {"beta": MisreportedFloat(1.0)}),
}


def configuration(name: str) -> Configuration:
    return Configuration(name, f"tests/{name}", ("model-a",), "empty", "local", "none")


def execute(comparison, tasks, procedures=PROCEDURES):
    return run(
        comparison, tasks, {item.name: procedures[item.name] for item in comparison.configurations}
    )


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


@pytest.fixture
def tasks():
    return load_tasks(FIXTURES / "tasks.json")


@pytest.fixture
def answers():
    return load_answers(FIXTURES / "answers.json")


@pytest.fixture
def ledger(tmp_path):
    return ExposureLedger(tmp_path)


def test_procedures_receive_only_a_fresh_copy_of_each_task_input(tasks, answers, ledger):
    entries = json.loads((FIXTURES / "tasks.json").read_text())["tasks"]
    received = Counter()

    def spy(*args, **kwargs):
        assert not kwargs and len(args) == 1 and type(args[0]) is dict
        received.update(name for name, entry in entries.items() if entry == args[0])
        args[0].clear()
        return Response("inconclusive")

    run(ledger.freeze(tasks, answers.key_id, (configuration("spy"),), 3), tasks, {"spy": spy})
    assert received == {name: 3 for name in entries}
    assert tasks == load_tasks(FIXTURES / "tasks.json")


def test_runs_keep_each_response_as_returned(tasks, answers, ledger):
    shared = {"beta": 0}

    def drifting(inputs: dict) -> Response:
        shared["beta"] += 1
        return Response("successful", shared)

    comparison = ledger.freeze(tasks, answers.key_id, (configuration("drifting"),), 2)
    results = run(comparison, tasks, {"drifting": drifting})
    assert [item.response.estimates["beta"] for item in results.runs] == [*map(float, range(1, 9))]


@pytest.mark.smoke
def test_disclosure_scores_each_repeat_against_its_task_kind(tasks, answers, ledger):
    assert answers.task_set_id == tasks.set_id
    calls = count()
    procedures = PROCEDURES | {
        "flaky": lambda inputs: Response("failed") if next(calls) % 2 else reference(inputs)
    }
    succeeds_on = {
        "honest": set(KINDS.values()),
        "hedged": {"null", "flawed"},
        "credulous": {"known_answer", "paper_reproduction"},
        "flaky": set(KINDS.values()),
    }
    comparison = ledger.freeze(tasks, answers.key_id, tuple(map(configuration, procedures)), 3)
    results = execute(comparison, tasks, procedures)
    responses = {
        (item.configuration, item.task, item.repeat): item.response for item in results.runs
    }
    assert len(responses) == len(results.runs)
    assert responses.keys() == {
        (name, task, repeat) for name in procedures for task in KINDS for repeat in range(3)
    }
    scores = ledger.disclose(results, answers)
    assert len(scores) == len(responses)
    assert {
        (item.configuration, item.task, item.repeat): (item.response, item.success)
        for item in scores
    } == {
        (name, task, repeat): (
            response,
            KINDS[task] in succeeds_on[name] and response.status != "failed",
        )
        for (name, task, repeat), response in responses.items()
    }
    assert {item.success for item in scores if item.configuration == "flaky"} == {True, False}


def test_disclosure_needs_the_frozen_complete_results_once_and_retires_tasks(
    tasks, answers, ledger
):
    baseline = (configuration("honest"),)
    comparison = ledger.freeze(tasks, answers.key_id, baseline, 2)
    assert ledger.freeze(tasks, answers.key_id, baseline, 2) == comparison
    overlapping = ledger.freeze(tasks, answers.key_id, (configuration("credulous"),), 1)
    foreign = replace(answers, task_set_id="0" * 64)
    stray = TaskSet("synthetic-stray", {"task-7": tasks.tasks["task-1"] | {"signal_lag": 3}})
    misbound = execute(ledger.freeze(stray, foreign.key_id, baseline, 2), stray)
    results = execute(comparison, tasks)
    changed = replace(
        comparison, configurations=(replace(baseline[0], model_versions=("model-b",)),)
    )
    runs = results.runs
    unscorable = replace(runs[0], response=Response("successful", {"beta": float("nan")}))
    for damaged, key, message in (
        (replace(results, runs=(unscorable, *runs[1:])), answers, "canonical JSON"),
        (replace(results, comparison=changed), answers, "frozen comparison"),
        (replace(results, runs=runs[:-1]), answers, "exactly once"),
        (replace(results, runs=(*runs, runs[0])), answers, "exactly once"),
        (results, foreign, "committed at freeze"),
        (misbound, foreign, "does not cover"),
    ):
        with pytest.raises(CapabilityError, match=message):
            ledger.disclose(damaged, key)
    assert len(ledger.disclose(results, answers)) == 8
    for disclosed in (results, execute(overlapping, tasks)):
        with pytest.raises(CapabilityError, match="already disclosed"):
            ledger.disclose(disclosed, answers)
    comparison.tasks.clear()
    with pytest.raises(CapabilityError, match="No held-out tasks"):
        ledger.freeze(tasks, answers.key_id, baseline, 2)
    revised = TaskSet(
        "synthetic-2",
        tasks.tasks
        | {
            "task-3": tasks.tasks["task-3"] | {"returns": [1, 2, 3, 4]},
            "task-5": tasks.tasks["task-4"] | {"signal_lag": 2},
            "task-6": tasks.tasks["task-1"],
        },
    )
    assert set(ledger.freeze(revised, answers.key_id, baseline, 2).tasks) == {"task-3", "task-5"}


def test_disclosure_needs_the_answer_key_committed_at_freeze(tasks, answers, ledger):
    results = execute(ledger.freeze(tasks, answers.key_id, (configuration("hedged"),), 1), tasks)
    known = answers.answers["task-3"]
    loosened = replace(answers, answers=answers.answers | {"task-3": replace(known, tolerance=1)})
    unhashable = replace(
        answers, answers=answers.answers | {"task-3": replace(known, tolerance=float("nan"))}
    )
    for key, message in ((loosened, "committed at freeze"), (unhashable, "canonical JSON")):
        with pytest.raises(CapabilityError, match=message):
            ledger.disclose(results, key)
    scores = {item.task: item.success for item in ledger.disclose(results, answers)}
    assert scores == {"task-1": True, "task-2": False, "task-3": False, "task-4": True}


def test_held_out_tasks_keep_the_answer_key_of_their_first_freeze(tasks, answers, ledger):
    baseline = (configuration("honest"),)
    comparison = ledger.freeze(tasks, answers.key_id, baseline, 1)
    loosened = answers.answers | {"task-3": replace(answers.answers["task-3"], tolerance=1)}
    for task_set in (tasks, TaskSet("synthetic-copy", tasks.tasks)):
        key = AnswerKey(task_set.set_id, loosened)
        with pytest.raises(CapabilityError, match="another answer key"):
            ledger.freeze(task_set, key.key_id, baseline, 1)
    assert ledger.freeze(tasks, answers.key_id, baseline, 1) == comparison
    widened = ledger.freeze(tasks, answers.key_id, (*baseline, configuration("credulous")), 1)
    assert widened.tasks == comparison.tasks


def test_disclosed_configuration_names_keep_their_identity(tasks, answers, ledger):
    baseline = configuration("honest")
    changed = replace(baseline, model_versions=("model-b",))
    fresh = TaskSet("synthetic-2", {"task-5": tasks.tasks["task-4"] | {"signal_lag": 2}})
    key = AnswerKey(fresh.set_id, {"task-5": answers.answers["task-4"]})
    rival = ledger.freeze(fresh, key.key_id, (changed,), 1)
    ledger.disclose(execute(ledger.freeze(tasks, answers.key_id, (baseline,), 1), tasks), answers)
    with pytest.raises(CapabilityError, match="changed after disclosure"):
        ledger.disclose(execute(rival, fresh), key)
    with pytest.raises(CapabilityError, match="changed after disclosure"):
        ledger.freeze(fresh, key.key_id, (changed,), 1)
    assert set(ledger.freeze(fresh, key.key_id, (baseline, configuration("hedged")), 1).tasks) == {
        "task-5"
    }


def test_a_reopened_ledger_keeps_disclosures_and_frozen_identities(
    tmp_path, tasks, answers, ledger
):
    baseline = configuration("honest")
    fresh = TaskSet("synthetic-2", {"task-5": tasks.tasks["task-4"] | {"signal_lag": 2}})
    key = AnswerKey(fresh.set_id, {"task-5": answers.answers["task-4"]})
    pending = ledger.freeze(fresh, key.key_id, (baseline,), 1)
    results = execute(ledger.freeze(tasks, answers.key_id, (baseline,), 1), tasks)
    ledger.disclose(results, answers)
    reopened = ExposureLedger(tmp_path)
    with pytest.raises(CapabilityError, match="already disclosed"):
        reopened.disclose(results, answers)
    combined = TaskSet("synthetic-3", tasks.tasks | fresh.tasks)
    assert set(reopened.freeze(combined, key.key_id, (baseline,), 1).tasks) == {"task-5"}
    with pytest.raises(CapabilityError, match="changed after disclosure"):
        reopened.freeze(fresh, key.key_id, (replace(baseline, model_versions=("model-b",)),), 1)
    assert [item.success for item in reopened.disclose(execute(pending, fresh), key)] == [True]


def test_a_ledger_opened_before_a_disclosure_cannot_disclose_overlapping_tasks(
    tmp_path, tasks, answers, ledger
):
    stale = ExposureLedger(tmp_path)
    honest = ledger.freeze(tasks, answers.key_id, (configuration("honest"),), 1)
    credulous = stale.freeze(tasks, answers.key_id, (configuration("credulous"),), 1)
    ledger.disclose(execute(honest, tasks), answers)
    with pytest.raises(CapabilityError, match="already disclosed"):
        stale.disclose(execute(credulous, tasks), answers)


def test_a_writer_that_loses_an_event_file_replays_the_winner_and_retries(
    tmp_path, tasks, answers, ledger, monkeypatch
):
    rival = ExposureLedger(tmp_path)
    fresh = TaskSet("synthetic-2", {"task-5": tasks.tasks["task-4"] | {"signal_lag": 2}})
    key = AnswerKey(fresh.set_id, {"task-5": answers.answers["task-4"]})
    honest, credulous = (configuration("honest"),), (configuration("credulous"),)
    winners = []

    def lose_race(path, content):
        while winners:
            winners.pop()()
        return write_once(path, content)

    monkeypatch.setattr(records, "write_once", lose_race)
    winners.append(lambda: rival.freeze(fresh, key.key_id, honest, 1))
    comparison = ledger.freeze(tasks, answers.key_id, honest, 1)
    pending = ledger.freeze(tasks, answers.key_id, credulous, 1)
    winners.append(lambda: rival.disclose(execute(comparison, tasks), answers))
    with pytest.raises(CapabilityError, match="already disclosed"):
        ledger.disclose(execute(pending, tasks), answers)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        f"{index}.json" for index in range(4)
    ]


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "non-canonical",
        "unknown-field",
        "boolean-version",
        "gap",
        "stray-file",
        "unknown-disclosure",
        "edited-comparison",
        "repeated-freeze",
        "second-key",
        "repeated-disclosure",
        "freeze-over-disclosed",
        "invalid-results-digest",
    ],
)
def test_opening_a_damaged_store_fails(tmp_path, tasks, answers, ledger, damage):
    comparison = ledger.freeze(tasks, answers.key_id, (configuration("honest"),), 1)
    ledger.disclose(execute(comparison, tasks), answers)
    files = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert files.keys() == {"0.json", "1.json"}
    events = {name: json.loads(content) for name, content in files.items()}
    assert {name: canonical(event) for name, event in events.items()} == files
    frozen, disclosed = events["0.json"], events["1.json"]
    match damage:
        case "unknown-field":
            frozen["note"] = "edited"
        case "boolean-version":
            disclosed["schema_version"] = True
        case "gap":
            events["2.json"] = events.pop("1.json")
        case "stray-file":
            events["notes.json"] = disclosed
        case "unknown-disclosure":
            disclosed["disclose"] = "0" * 64
        case "edited-comparison":
            frozen["freeze"]["repeats"] = 2
        case "repeated-freeze":
            events["1.json"], events["2.json"] = frozen, disclosed
        case "second-key":
            rekeyed = frozen | {"freeze": frozen["freeze"] | {"answer_key_id": "0" * 64}}
            events["1.json"], events["2.json"] = rekeyed, disclosed
        case "repeated-disclosure":
            events["2.json"] = disclosed
        case "freeze-over-disclosed":
            events["2.json"] = frozen | {"freeze": frozen["freeze"] | {"repeats": 2}}
        case "invalid-results-digest":
            disclosed["results"] = "x"
    for path in tmp_path.iterdir():
        path.unlink()
    for name, event in events.items():
        (tmp_path / name).write_bytes(
            json.dumps(event, indent=2).encode() if damage == "non-canonical" else canonical(event)
        )
    with pytest.raises(CapabilityError):
        ExposureLedger(tmp_path / "missing" if damage == "missing" else tmp_path)


def test_freeze_and_run_reject_invalid_requests(tasks, answers, ledger):
    honest = configuration("honest")
    key = answers.key_id
    for key_id, configurations, repeats, message in (
        (key, (), 1, "uniquely named"),
        (key, (honest, replace(honest, implementation="tests/other")), 1, "uniquely named"),
        (key, (honest,), 0, "positive integer"),
        (key, (honest,), 1001, "at most 1000"),
        (key, (honest,), 10**300, "at most 1000"),
        (key[:-1], (honest,), 1, "SHA-256"),
        (key, (replace(honest, model_versions="model-a"),), 1, "must be an array"),
        (key, (replace(honest, model_versions={"model-a"}),), 1, "Invalid comparison"),
    ):
        with pytest.raises(CapabilityError, match=message):
            ledger.freeze(tasks, key_id, configurations, repeats)
    unhashable = TaskSet("synthetic-nan", {"task-1": {"signal": [float("nan")]}})
    with pytest.raises(CapabilityError, match="canonical JSON"):
        ledger.freeze(unhashable, key, (honest,), 1)
    assert ledger.freeze(tasks, key, (honest,), 1000).repeats == 1000
    comparison = ledger.freeze(tasks, answers.key_id, (honest,), 1)
    for task_set, procedures, message in (
        (replace(tasks, version="synthetic-2"), {"honest": reference}, "does not match"),
        (unhashable, {"honest": reference}, "canonical JSON"),
        (tasks, {}, "one procedure per configuration"),
        (tasks, {"honest": reference, "other": reference}, "one procedure per configuration"),
        (tasks, {"honest": lambda inputs: Response("confirmed")}, "invalid response"),
        (tasks, {"honest": lambda inputs: {"status": "successful"}}, "invalid response"),
        (tasks, {"honest": lambda inputs: Response("successful", None)}, "invalid response"),
        (tasks, {"honest": lambda inputs: Response("successful", {"beta": "1"})}, "invalid"),
        (tasks, {"honest": lambda inputs: Response(EqualsAnything())}, "invalid response"),
    ):
        with pytest.raises(CapabilityError, match=message):
            run(comparison, task_set, procedures)


@pytest.mark.parametrize("fake", FAKE_RESPONSES.values(), ids=list(FAKE_RESPONSES))
def test_run_rejects_responses_that_are_not_plain_data(tasks, answers, ledger, fake):
    comparison = ledger.freeze(tasks, answers.key_id, (configuration("fake"),), 1)
    with pytest.raises(CapabilityError, match="invalid response"):
        run(comparison, tasks, {"fake": lambda inputs: fake()})


@pytest.mark.parametrize(
    "name, damage",
    [
        ("tasks.json", "missing"),
        ("tasks.json", "syntax"),
        ("tasks.json", "boolean-version"),
        ("tasks.json", "unknown-field"),
        ("tasks.json", "empty-tasks"),
        ("tasks.json", "array-inputs"),
        ("tasks.json", "nan-inputs"),
        ("tasks.json", "duplicate-key"),
        ("answers.json", "digest"),
        ("answers.json", "kind"),
        ("answers.json", "null-estimates"),
        ("answers.json", "missing-estimates"),
        ("answers.json", "infinite-tolerance"),
        ("answers.json", "oversized-tolerance"),
        ("answers.json", "duplicate-key"),
        ("answers.json", "surrogate-key"),
    ],
)
def test_loaders_reject_malformed_files(tmp_path, name, damage):
    document = json.loads((FIXTURES / name).read_text())
    match damage:
        case "boolean-version":
            document["schema_version"] = True
        case "unknown-field":
            document["answers"] = {}
        case "empty-tasks":
            document["tasks"] = {}
        case "array-inputs":
            document["tasks"]["task-1"] = []
        case "nan-inputs":
            document["tasks"]["task-1"]["signal"] = [float("nan")]
        case "digest":
            document["task_set_id"] = "../tasks.json"
        case "kind":
            document["answers"]["task-4"]["kind"] = "confirmed"
        case "null-estimates":
            document["answers"]["task-4"]["estimates"] = {"beta": 0.0}
        case "missing-estimates":
            document["answers"]["task-3"]["estimates"] = {}
        case "infinite-tolerance":
            document["answers"]["task-3"]["tolerance"] = float("inf")
        case "oversized-tolerance":
            document["answers"]["task-3"]["tolerance"] = 10**400
        case "surrogate-key":
            document["answers"]["task-3"]["estimates"] = {"\ud800": 0.5}
    text = json.dumps(document)
    if damage == "duplicate-key":
        text = text.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1')
    path = tmp_path / name
    if damage != "missing":
        path.write_text("{" if damage == "syntax" else text)
    with pytest.raises(CapabilityError):
        (load_tasks if name == "tasks.json" else load_answers)(path)


@pytest.fixture
def pilot():
    spec = importlib.util.spec_from_file_location(
        "capability_pilot", FIXTURES / "pilot" / "pilot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def development(ledger, tasks, answers, *, names=("honest",), repeats=3, share=2 / 3):
    return ledger.freeze_development(
        tasks,
        answers.key_id,
        tuple(map(configuration, names)),
        repeats,
        known_answer_success_share=share,
    )


def test_development_reruns_reopen_and_keep_historical_bytes(tasks, answers, ledger, tmp_path):
    comparison = ledger.freeze(tasks, answers.key_id, (configuration("honest"),), 3)
    assert (
        comparison.comparison_id
        == "6df45a67b3b2ee20ec28adfaf159999cdadf11fa317dbae7640fb0154863c009"
    )
    assert (tmp_path / "0.json").read_bytes() == records.canonical_bytes(
        {"schema_version": 1, "freeze": asdict(comparison)}
    )
    frozen = development(ledger, tasks, answers)
    assert development(ledger, tasks, answers) == frozen
    changed = development(ledger, tasks, answers, share=1)
    assert changed.development_id != frozen.development_id
    assert changed.comparison == frozen.comparison
    shared = {"received": []}
    calls = Counter()

    async def procedure(inputs):
        calls.update(name for name, value in tasks.tasks.items() if value == inputs)
        response = reference(inputs)
        inputs.clear()
        shared["received"].append(1)
        return ProcedureResult(response, shared)

    first = asyncio.run(run_development(ledger, frozen, tasks, {"honest": procedure}))
    first_scores = ledger.disclose_development(first, answers)
    reopened = ExposureLedger(tmp_path)
    second = asyncio.run(run_development(reopened, frozen, tasks, {"honest": procedure}))
    second_scores = reopened.disclose_development(second, answers)
    assert first.execution_id != second.execution_id
    assert first.results == second.results
    assert first_scores.scores == second_scores.scores
    assert first_scores.passed and second_scores.passed
    assert len(first_scores.scores) == 12
    assert first_scores.known_answer_shares == {"honest": {"task-2": 1.0, "task-3": 1.0}}
    assert [len(item["received"]) for item in first.evidence.values()] == list(range(1, 13))
    assert calls == {task: 6 for task in tasks.tasks}
    with pytest.raises(CapabilityError, match="already disclosed"):
        reopened.disclose_development(first, answers)
    with pytest.raises(CapabilityError, match="already disclosed"):
        reopened.disclose(execute(comparison, tasks), answers)


def test_development_begin_retires_crashed_and_renamed_tasks(tasks, answers, ledger, tmp_path):
    pending = ledger.freeze(tasks, answers.key_id, (configuration("honest"),), 1)
    pending_results = execute(pending, tasks)
    frozen = development(ledger, tasks, answers)

    async def crash(inputs):
        assert len(list(tmp_path.glob("*.json"))) == 3
        raise RuntimeError("crashed after durable exposure")

    with pytest.raises(RuntimeError, match="durable exposure"):
        asyncio.run(run_development(ledger, frozen, tasks, {"honest": crash}))
    stale = ExposureLedger(tmp_path)
    renamed = TaskSet("renamed", {f"renamed-{name}": value for name, value in tasks.tasks.items()})
    with pytest.raises(CapabilityError, match="No held-out"):
        stale.freeze(renamed, answers.key_id, (configuration("honest"),), 1)
    with pytest.raises(CapabilityError, match="already disclosed"):
        stale.disclose(pending_results, answers)
    assert development(stale, tasks, answers) == frozen


def test_development_shares_are_per_configuration_and_case(tasks, answers, ledger):
    frozen = development(ledger, tasks, answers, names=("honest", "bad"))
    attempts = Counter()

    async def procedure(inputs):
        task = next(name for name, value in tasks.tasks.items() if value == inputs)
        attempts[task] += 1
        return ProcedureResult(Response("failed") if attempts[task] == 3 else reference(inputs), {})

    async def bad(inputs):
        return ProcedureResult(reference(inputs, offset=0.1), {})

    result = asyncio.run(run_development(ledger, frozen, tasks, {"honest": procedure, "bad": bad}))
    disclosure = ledger.disclose_development(result, answers)
    assert disclosure.known_answer_shares["honest"] == {"task-2": 2 / 3, "task-3": 2 / 3}
    assert disclosure.known_answer_shares["bad"] == {"task-2": 0, "task-3": 0}
    assert disclosure.accepted["honest"] == {
        "task-1": False,
        "task-2": True,
        "task-3": True,
        "task-4": False,
    }
    assert not disclosure.passed


@pytest.mark.parametrize(
    "damage", ["key", "grid", "evidence", "execution", "comparison", "response"]
)
def test_development_disclosure_rejects_uncommitted_or_incomplete_evidence(
    tasks,
    answers,
    ledger,
    tmp_path,
    damage,
):
    frozen = development(ledger, tasks, answers, repeats=1)

    async def procedure(inputs):
        return ProcedureResult(reference(inputs), {"saved": True})

    result = asyncio.run(run_development(ledger, frozen, tasks, {"honest": procedure}))
    match damage:
        case "key":
            answers = replace(answers, answers={})
        case "grid":
            result = replace(result, results=replace(result.results, runs=result.results.runs[:-1]))
        case "evidence":
            result = replace(result, evidence={})
        case "execution":
            result = replace(result, execution_id="00000000-0000-0000-0000-000000000001")
        case "comparison":
            result = replace(
                result,
                results=replace(result.results, comparison=replace(frozen.comparison, repeats=2)),
            )
        case "response":
            runs = (
                replace(result.results.runs[0], response=Response("successful", {"beta": True})),
                *result.results.runs[1:],
            )
            result = replace(result, results=replace(result.results, runs=runs))
    count = len(list(tmp_path.glob("*.json")))
    with pytest.raises((CapabilityError, ValueError)):
        ledger.disclose_development(result, answers)
    assert len(list(tmp_path.glob("*.json"))) == count


def test_development_bindings_and_begin_append_race(tasks, answers, ledger, tmp_path, monkeypatch):
    frozen = development(ledger, tasks, answers, repeats=1)
    stale = ExposureLedger(tmp_path)
    original = records.write_once
    injected = False

    def compete(path, data):
        nonlocal injected
        if path.name == "1.json" and not injected:
            injected = True
            stale._begin_development(frozen, "00000000-0000-0000-0000-000000000001")
        return original(path, data)

    monkeypatch.setattr(records, "write_once", compete)

    async def procedure(inputs):
        return ProcedureResult(reference(inputs), {})

    result = asyncio.run(run_development(ledger, frozen, tasks, {"honest": procedure}))
    assert injected and len(list(tmp_path.glob("*.json"))) == 3
    with pytest.raises(CapabilityError, match="began twice"):
        stale._begin_development(frozen, result.execution_id)
    ledger.disclose_development(result, answers)
    with pytest.raises(CapabilityError, match="another answer key"):
        development(ledger, tasks, replace(answers, task_set_id="0" * 64))
    with pytest.raises(CapabilityError, match="changed after disclosure"):
        ledger.freeze_development(
            tasks,
            answers.key_id,
            (replace(configuration("honest"), implementation="changed"),),
            1,
            known_answer_success_share=2 / 3,
        )


@pytest.mark.parametrize("value", [True, 0, -1, 1.01, float("inf"), "0.5"])
def test_development_freeze_rejects_invalid_acceptance(tasks, answers, ledger, value):
    with pytest.raises(ValueError):
        development(ledger, tasks, answers, share=value)


@pytest.mark.parametrize(
    "damage",
    ["version", "unknown-action", "unknown-study", "duplicate-begin", "duplicate-disclose"],
)
def test_development_event_replay_rejects_invalid_transitions(
    tasks, answers, ledger, tmp_path, damage
):
    frozen = development(ledger, tasks, answers, repeats=1)
    ledger._begin_development(frozen, "00000000-0000-0000-0000-000000000001")
    event = json.loads((tmp_path / "1.json").read_bytes())
    match damage:
        case "version":
            event["schema_version"] = 2.0
        case "unknown-action":
            event["action"] = "unknown"
        case "unknown-study":
            event["value"]["development"] = "0" * 64
        case "duplicate-begin":
            pass
        case "duplicate-disclose":
            event["action"] = "disclose_development"
            event["value"]["results"] = "0" * 64
            (tmp_path / "2.json").write_bytes(records.canonical_bytes(event))
    index = 3 if damage == "duplicate-disclose" else 2
    (tmp_path / f"{index}.json").write_bytes(records.canonical_bytes(event))
    with pytest.raises(CapabilityError):
        ExposureLedger(tmp_path)


def test_pilot_oracle_matches_causal_inputs_and_actual_numerical_pipeline(pilot, tmp_path):
    from uuid import uuid4

    from quant_research import campaigns, evaluation, signals
    from quant_research import reference as numerical

    tasks = load_tasks(FIXTURES / "pilot" / "tasks.json")
    answers = load_answers(FIXTURES / "pilot" / "answers.json")
    assert tasks.set_id == answers.task_set_id
    switches = (Fraction(0), *(Fraction(-1, 1000), Fraction(979, 20000)) * 14, Fraction(979, 20000))
    assert len(switches) == 30
    expected = sum(switches) / 30
    assert expected == Fraction(2881, 120000)
    assert answers.answers["case-1"].estimates == {"incremental_return": float(expected)}
    assert answers.answers["case-1"].tolerance == 1e-12
    for name, inputs in tasks.tasks.items():
        directory = tmp_path / name
        directory.mkdir()
        revision, setup = pilot.prepare(inputs, directory)
        authority = campaigns.resolve(revision, setup=setup)
        source = uuid4()
        extended = numerical.extend(
            authority.search, None, source, records.canonical_bytes(inputs["candidate"])
        )
        if answers.answers[name].kind == "flawed":
            assert extended.status == "rejected"
            reason = "negative_lag" if name == "case-3" else "reads_past_vintage"
            assert f"({reason})" in json.loads(extended.summary)["detail"]
            continue
        held = numerical.assess(
            authority.search,
            authority.context,
            extended.package.digest,
            ((source, uuid4()),),
            adjustment_trials=1,
        )
        [item] = json.loads(held.summary)
        result = numerical.decode_evaluation(records.canonical_bytes(item["evaluation"]))
        assert isinstance(result, evaluation.Evaluation)
        assert result.incremental.estimate == pytest.approx(
            float(expected) if name == "case-1" else 0, abs=1e-12
        )
        if name == "case-1":
            program = signals.parse(
                records.canonical_bytes(inputs["candidate"]),
                setup.catalog,
                authority.search.universe,
            )
            decisions = signals.decide(
                program,
                setup.catalog,
                setup.datasets,
                authority.context,
                authority.search.rules.search.months,
            )
            assert [item.value for item in decisions] == [index % 2 == 0 for index in range(30)]
            assert all(item.observation == signals.shift(item.month, -2) for item in decisions)
            assert all(value > 0 for value in result.subperiods)


@pytest.mark.parametrize(
    "case,mode,expected",
    [
        ("case-1", "honest", "successful"),
        ("case-2", "honest", "null"),
        ("case-3", "honest", "rejected"),
        ("case-4", "honest", "rejected"),
        ("case-2", "unsupported-positive", "failed"),
        ("case-2", "malformed-conclusion", "failed"),
        ("case-2", "budget", "failed"),
        ("case-1", "critic-reject", "failed"),
        ("case-2", "honest-inconclusive", "inconclusive"),
        ("case-3", "empty-shortlist", "failed"),
        ("case-1", "unrelated-rejection", "failed"),
        ("case-3", "repair", "failed"),
        ("case-2", "cancel", "failed"),
        ("case-2", "rate-limit", "failed"),
        ("case-2", "worker-failure", "failed"),
        ("case-2", "deadline", "failed"),
    ],
)
def test_research_procedure_runs_real_loop_and_never_turns_missing_support_into_null(
    temporal,
    tmp_path,
    pilot,
    case,
    mode,
    expected,
):
    from temporalio.client import Client

    from quant_research import reference as numerical
    from quant_research.providers import CallLog, Fault, Gateway, ScriptedProvider

    tasks = load_tasks(FIXTURES / "pilot" / "tasks.json")
    inputs = tasks.tasks[case]
    state = {}

    def action(name, arguments):
        return json.dumps(
            {"action": name, "arguments": json.dumps(arguments), "rationale": "Supplied protocol"}
        )

    def script():
        if mode in {"cancel", "rate-limit", "worker-failure", "deadline"}:
            fault = {
                "cancel": Fault.CANCEL,
                "rate-limit": Fault.RATE_LIMIT,
                "worker-failure": Fault.FAIL,
                "deadline": Fault.BLOCK,
            }[mode]
            while True:
                yield fault
        if mode == "empty-shortlist":
            yield action("conclude", {"trials": []})
        elif mode == "budget":
            yield action("read_dataset", {"dataset": pilot.UNRATE})
            yield action("read_dataset", {"dataset": pilot.UNRATE})
            yield action("conclude", {"trials": []})
        else:
            candidate = (
                tasks.tasks["case-3"]["candidate"]
                if mode == "unrelated-rejection"
                else tasks.tasks["case-1"]["candidate"]
                if mode == "repair"
                else inputs["candidate"]
            )
            yield action("experiment", candidate)
            summary = json.loads(state["provider"].calls[-1].texts[-1].split("\nResult: ")[1])
            trials = [] if summary["status"] == "rejected" else [summary["trial"]]
            yield action("conclude", {"trials": trials})
        evidence = json.loads(state["provider"].calls[-1].texts[-2])
        classification = "no_result"
        citations = []
        if evidence:
            [trial] = evidence
            result = numerical.decode_evaluation(records.canonical_bytes(trial["evaluation"]))
            classification, citations = result.verdict, [trial["trial"]]
        if mode == "unsupported-positive":
            classification = "supported"
        if mode == "honest-inconclusive":
            classification = "inconclusive"
        if mode == "malformed-conclusion":
            yield json.dumps({"classification": "supported"})
            yield json.dumps({"decision": "accept", "citations": [], "explanation": "Reviewed"})
            return
        yield json.dumps(
            {
                "classification": classification,
                "citations": citations,
                "explanation": "The recorded numerical evidence determines this claim",
            }
        )
        yield json.dumps(
            {
                "decision": "reject" if mode == "critic-reject" else "accept",
                "citations": [],
                "explanation": "Evidence reviewed",
            }
        )

    budget = pilot.DispatchBudget(tmp_path / "budget")

    def gateway(calls):
        provider = ScriptedProvider(script())
        state["provider"] = provider
        counted = budget.adapter(
            provider, hashlib.sha256(records.canonical_bytes(inputs)).hexdigest(), calls.parent.name
        )
        return Gateway(CallLog(calls), [(policy, counted) for policy in pilot.POLICIES])

    async def execute():
        client = await Client.connect(temporal)
        procedure = ResearchProcedure(
            client,
            pilot.prepare,
            gateway,
            tmp_path / "cells",
            max_seconds=2 if mode == "deadline" else 60,
        )
        return await procedure(inputs)

    result = asyncio.run(execute())
    assert result.response.status == expected, result.evidence
    assert "error" not in result.evidence, result.evidence
    assert result.evidence["elapsed_seconds"] > 0
    assert Path(result.evidence["report"]).exists()
    assert Path(result.evidence["history"]).exists()
    assert result.evidence["archives"]
    if mode in {
        "honest",
        "unsupported-positive",
        "critic-reject",
        "honest-inconclusive",
        "unrelated-rejection",
        "repair",
    }:
        assert budget.snapshot()["total"] == 4
        assert len(result.evidence["calls"]) == 4
    if expected == "successful":
        assert result.response.estimates["incremental_return"] == pytest.approx(
            float(Fraction(2881, 120000)), abs=1e-12
        )
    if mode == "unsupported-positive":
        report = json.loads(Path(result.evidence["report"]).read_bytes())
        assert report["assessment"]["classification"] == "inconclusive"
    if mode == "rate-limit":
        assert budget.stopped and budget.retry_at == "unknown"
        assert budget.snapshot()["total"] == 1
    if mode == "deadline":
        assert state["provider"].blocking.is_set()
        report = json.loads(Path(result.evidence["report"]).read_bytes())
        assert report["ending"]["outcome"] == "cancelled"
        assert result.evidence["failure"] == "deadline"
    if mode == "budget":
        report = json.loads(Path(result.evidence["report"]).read_bytes())
        assert report["ending"]["reason"] == "model_calls"


@pytest.mark.parametrize("limit", ["campaign", "case", "effort"])
def test_dispatch_budget_counts_before_calls_reopens_and_refuses_over_limit(pilot, tmp_path, limit):

    limits = {"campaign": {"per_campaign": 1}, "case": {"per_case": 1}, "effort": {"max_calls": 1}}
    budget = pilot.DispatchBudget(tmp_path, **limits[limit])

    class Adapter:
        subscription = True
        runtime_version = "offline-budget-probe"
        calls = 0

        def complete(self, request, cancel):
            assert pilot.DispatchBudget(tmp_path, **limits[limit]).snapshot()["total"] == 1
            self.calls += 1
            return "reply"

    provider = Adapter()
    counted = budget.adapter(provider, "a", "first")
    assert counted.complete("request", threading.Event()) == "reply"
    reopened = pilot.DispatchBudget(tmp_path, **limits[limit])
    case = "a" if limit == "case" else "b"
    campaign = "first" if limit == "campaign" else "second"
    cancel = threading.Event()
    reopened.adapter(provider, case, campaign).complete("request", cancel)
    assert provider.calls == 1 and cancel.is_set()
    assert reopened.stopped and reopened.snapshot()["total"] == 1
    with pytest.raises(ValueError, match="policy changed"):
        pilot.DispatchBudget(tmp_path, max_calls=2)


def test_dispatch_rate_limit_stops_all_later_adapters_durably(pilot, tmp_path):
    from quant_research.accounting import Usage
    from quant_research.providers import RateLimited

    class Adapter:
        subscription = True
        runtime_version = "offline-rate-probe"
        calls = 0

        def complete(self, request, cancel):
            self.calls += 1
            return RateLimited(Usage())

    provider = Adapter()
    budget = pilot.DispatchBudget(tmp_path)
    assert isinstance(
        budget.adapter(provider, "one", "first").complete(None, threading.Event()), RateLimited
    )
    reopened = pilot.DispatchBudget(tmp_path)
    reopened.adapter(provider, "two", "second").complete(None, threading.Event())
    assert reopened.stopped and reopened.retry_at == "unknown"
    assert provider.calls == reopened.snapshot()["total"] == 1
