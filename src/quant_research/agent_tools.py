from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, get_args

from quant_research import datasets, worker
from quant_research.access import MemoryPin, Restrictions, RunContext, derive
from quant_research.catalog import Catalog
from quant_research.datasets import Description, Listing, Supplied
from quant_research.findings import Ledger, Lesson
from quant_research.locators import Kind, Location, published_locations
from quant_research.providers import Material
from quant_research.records import canonical_bytes, json_default, parse_fields, unique_keys
from quant_research.retrieval import (
    Mode,
    Query,
    Retrieval,
    RetrievalError,
    corpus_members,
    eligible_restrictions,
    retrieval_bytes,
    retrieve,
)
from quant_research.worker import LoggedRun, RunRecord

type Reason = (
    Literal["unknown_tool", "invalid_arguments", "unknown_ref", "generated_code_off"]
    | datasets.Reason
)

_MESSAGES: dict[Reason, str] = {
    "unknown_tool": "No tool has this name",
    "invalid_arguments": "The arguments do not match the tool's input schema",
    "unknown_ref": "No location this run may read has this ref",
    "generated_code_off": "This run does not enable generated code",
    **{reason: datasets.Refused(reason).message for reason in get_args(datasets.Reason.__value__)},
}

_KINDS = get_args(Kind.__value__)
# ref() writes offsets into a str, so they never exceed 19 digits. The bound also keeps int()
# under Python's digit limit, which is configurable per process and would break replay.
_OFFSET = "(?:0|[1-9][0-9]{0,18})"
_REF = re.compile(
    rf"(?P<id>[0-9a-f]{{64}}):(?P<kind>{'|'.join(_KINDS)}):(?P<start>{_OFFSET})-(?P<end>{_OFFSET})"
)

MAX_UNITS = 20
MAX_RESULT_BYTES = 2**18
_WITHHELD = canonical_bytes(
    {
        "refusal": "result_too_large",
        "message": f"The result is larger than {MAX_RESULT_BYTES} bytes, so it is withheld",
    }
).decode()


class ToolError(ValueError):
    """Report a record that is malformed, names another context, or replays differently."""


@dataclass(frozen=True)
class Tool:
    """What a model sees of one tool: its name, what it does, and its arguments' JSON Schema."""

    name: str
    description: str
    input_schema: Mapping[str, object]


@dataclass(frozen=True)
class Refusal:
    """A call that ran no tool or was not allowed. Its message quotes no argument or content."""

    reason: Reason

    @property
    def message(self) -> str:
        return _MESSAGES[self.reason]


@dataclass(frozen=True)
class Located:
    """One located element of the run's pinned corpus, under the ref that names it.

    digest is the content digest of the locator generation that holds the location.
    """

    ref: str
    occurrence_id: str
    digest: str
    location: Location


@dataclass(frozen=True)
class Recalled:
    """One lesson a run may read, under the record ID that a later proposal cites it by."""

    record_id: str
    lesson: Lesson


@dataclass(frozen=True)
class Lessons:
    """The lessons a run may read at its memory pin, in proposal order."""

    memory: MemoryPin
    lessons: tuple[Recalled, ...]


@dataclass(frozen=True)
class Generated:
    """Where a run that enables generated code logs its container runs and keeps their signals.

    signals is an existing directory. A valid signal's values are written to it as
    <digest>.json under the logged output digest.
    """

    log: worker.RunLog
    signals: Path
    limits: worker.Limits = worker.LIMITS


@dataclass(frozen=True)
class _Candidate:
    source: str
    read: Supplied


type Outcome = (
    Retrieval | Supplied | Listing | Description | Located | Lessons | LoggedRun | Refusal
)
type _Result = tuple[Outcome | _Candidate, tuple[Restrictions, ...]]


@dataclass(frozen=True)
class Call:
    """One tool call as the model sent it, the identity of the run context, and its outcome.

    restrictions derive over the run's floor and every readable item the tool read, returned or
    not. Toolset derives them again on replay, so the record leaves them out.
    """

    tool: str
    arguments: str
    context: str
    outcome: Outcome
    restrictions: Restrictions

    @property
    def material(self) -> Material:
        """Return the outcome as the JSON text a model reads, under the call's restrictions.

        A text larger than MAX_RESULT_BYTES of UTF-8 is replaced whole by a fixed notice under
        the same restrictions, so withholding never loosens them.
        """
        text = _render(self.outcome)
        if len(text.encode()) > MAX_RESULT_BYTES:
            text = _WITHHELD
        return Material(text, self.restrictions)

    def to_bytes(self) -> bytes:
        """Return canonical UTF-8 JSON with the schema version and every field.

        A read records row identities and the fixture vintage, never row values. A listing
        records the listed names under the catalog digest, which pins their documentation. A run
        records its run log entry, which holds the signal's digest, never its values.
        """
        match self.outcome:
            case Refusal(reason):
                outcome: dict[str, object] = {"refusal": reason}
            case Retrieval() as retrieval:
                outcome = {"retrieval": json.loads(retrieval_bytes(retrieval))}
            case Supplied(record=record):
                outcome = {"read": json.loads(record.to_bytes())}
            case Listing(catalog, listed):
                outcome = {
                    "listing": {"catalog": catalog, "datasets": [item.dataset for item in listed]}
                }
            case Description(catalog, dataset, units):
                outcome = {
                    "description": {
                        "catalog": catalog,
                        "dataset": dataset,
                        "units": [asdict(unit) for unit in units],
                    }
                }
            case Located(ref, occurrence_id, digest, location):
                outcome = {
                    "location": {
                        "ref": ref,
                        "occurrence_id": occurrence_id,
                        "digest": digest,
                        **asdict(location),
                    }
                }
            case LoggedRun() as run:
                outcome = {"run": json.loads(run.to_bytes())}
            case Lessons(memory, lessons):
                outcome = {"lessons": {"memory": memory, "lessons": _recalled(lessons)}}
        return canonical_bytes(
            {
                "schema_version": 1,
                "tool": self.tool,
                "arguments": self.arguments,
                "context": self.context,
                "outcome": outcome,
            }
        )


@dataclass(frozen=True)
class Toolset:
    """The tools of one run, bound to its trusted context, stores, catalog, and review log.

    sources holds the corpus, its generations, and their locator generations. datasets holds
    the dataset fixtures that the catalog approves. ledger holds the review log whose prefix
    the context's memory pin names. generated is None unless the run enables generated code.
    """

    context: RunContext
    sources: Path
    datasets: Path
    catalog: Catalog
    ledger: Ledger
    generated: Generated | None = None

    def call(self, tool: str, arguments: str) -> Call:
        """Run one model tool call under this run's authority.

        The name and arguments are checked before any store, catalog, review log, or file
        access. No argument names a cutoff, corpus, clearance, provider, or memory pin, so none
        can widen them.

        Args:
            tool (str): Tool name as the model sent it.
            arguments (str): JSON text of the arguments as the model sent it.

        Returns:
            (Call) The call with its result or with a refusal, under the restrictions of the
            run's floor and of every readable item the tool read.

        Raises:
            RetrievalError: If the corpus is unpublished, damaged, or differs from its pins.
            SourceError: If a published generation or locator generation fails verification.
            DatasetError: If a dataset fixture is unreadable or malformed.
            ReviewError: If the review log has no prefix at the context's memory pin.
            WorkerError: If Docker fails, the worker image is not built, or the run log cannot
                be replayed or written.
            OSError: If a logged signal cannot be written to the signal directory.
        """
        outcome, inputs = self._outcome(tool, arguments)
        if isinstance(outcome, _Candidate):
            outcome = self._contain(outcome)
        restrictions = derive(_floor(self.context), *inputs)
        return Call(tool, arguments, self.context.identity, outcome, restrictions)

    def _contain(self, candidate: _Candidate) -> LoggedRun:
        generated = self.generated
        result = worker.run(
            candidate.source,
            self.catalog,
            self.datasets,
            candidate.read.record.dataset,
            self.context,
            generated.limits,
        )
        logged = generated.log.append(result)
        if isinstance(result, worker.Signal):
            path = generated.signals / f"{logged.output}.json"
            path.write_bytes(canonical_bytes(result.values))
        return logged

    def _outcome(self, tool: str, arguments: str) -> _Result:
        if tool not in _TOOLS:
            return Refusal("unknown_tool"), ()
        spec, handler = _TOOLS[tool]
        try:
            values = parse_fields(
                json.loads(arguments, object_pairs_hook=unique_keys),
                " ".join(spec.input_schema["properties"]),
                "arguments",
            )
            if any(type(value) is not str for value in values.values()):
                return Refusal("invalid_arguments"), ()
            for value in values.values():
                # A \ud800 escape decodes to a lone surrogate, which no record can encode.
                value.encode("utf-8")
        except (ValueError, RecursionError):
            return Refusal("invalid_arguments"), ()
        return handler(self, **values)

    def replay(self, record: bytes) -> Call:
        """Re-run a recorded call under this run and require byte-identical record bytes.

        A recorded container run is not run again. Its read is repeated, and the bound run log
        must hold that run of the recorded source over the same read under the bound limits.

        Args:
            record (bytes): Record bytes, as `Call.to_bytes` returns them.

        Returns:
            (Call) The re-run call, whose bytes equal the record.

        Raises:
            ToolError: If the record is malformed, names another run context, or the re-run
                call differs from it.
            RetrievalError: As `call` raises it.
            SourceError: As `call` raises it.
            DatasetError: As `call` raises it.
            ReviewError: As `call` raises it.
            WorkerError: If the run log cannot be replayed.
        """
        try:
            value = parse_fields(
                json.loads(record.decode("utf-8")),
                "schema_version tool arguments context outcome",
                "tool call record",
            )
        except (ValueError, RecursionError) as error:
            raise ToolError("Tool call record is malformed") from error
        if type(value["tool"]) is not str or type(value["arguments"]) is not str:
            raise ToolError("Tool call record is malformed")
        if value["context"] != self.context.identity:
            raise ToolError("Tool call record names another run context")
        outcome, inputs = self._outcome(value["tool"], value["arguments"])
        if isinstance(outcome, _Candidate):
            outcome = self._recorded(outcome, value["outcome"])
        restrictions = derive(_floor(self.context), *inputs)
        call = Call(value["tool"], value["arguments"], self.context.identity, outcome, restrictions)
        if call.to_bytes() != record:
            raise ToolError("Re-executed tool call differs from the record")
        return call

    def _recorded(self, candidate: _Candidate, recorded: object) -> LoggedRun:
        generated = self.generated
        source = hashlib.sha256(candidate.source.encode()).hexdigest()
        for run in generated.log.load():
            commitment = RunRecord(
                run.record.image, source, candidate.read.record, generated.limits
            )
            if run.record == commitment and {"run": json.loads(run.to_bytes())} == recorded:
                return run
        raise ToolError("The run log holds no run of this call")


def _floor(context: RunContext) -> Restrictions:
    return Restrictions(
        available_at=context.cutoff, labels=frozenset(), approved_for=context.providers
    )


def _recalled(lessons: tuple[Recalled, ...]) -> list[object]:
    recalled = [{"record_id": item.record_id, **asdict(item.lesson)} for item in lessons]
    return json.loads(json.dumps(recalled, default=json_default))


def _render(outcome: Outcome) -> str:
    match outcome:
        case Refusal() as refusal:
            value: object = {"refusal": refusal.reason, "message": refusal.message}
        case Retrieval(units=units):
            value = {"units": [asdict(unit) for unit in units]}
        case Supplied(rows=rows, record=record):
            value = {"dataset": record.dataset, "rows": [asdict(row) for row in rows]}
        case Listing(datasets=listed):
            value = {
                "datasets": [
                    {"dataset": item.dataset, "owner": item.owner, "fields": list(item.fields)}
                    for item in listed
                ]
            }
        case Description(dataset=dataset, units=units):
            value = {"dataset": dataset, "units": [asdict(unit) for unit in units]}
        case Located(ref, occurrence_id, _, location):
            value = {"ref": ref, "occurrence_id": occurrence_id, **asdict(location)}
        case Lessons(lessons=lessons):
            value = {"lessons": _recalled(lessons)}
        case LoggedRun(run_id=run_id, outcome="signal", output=output):
            value = {"run_id": run_id, "signal": output}
        case LoggedRun(run_id=run_id, outcome=reason):
            value = {"run_id": run_id, "reason": reason, "message": worker.Refused(reason).message}
    return canonical_bytes(json.loads(json.dumps(value, default=json_default))).decode()


def _search_corpus(toolset: Toolset, mode: str, text: str) -> _Result:
    try:
        query = Query(mode, text)
    except RetrievalError:
        return Refusal("invalid_arguments"), ()
    retrieval = retrieve(toolset.sources, toolset.context, query)
    return (
        replace(retrieval, units=retrieval.units[:MAX_UNITS]),
        eligible_restrictions(toolset.sources, toolset.context),
    )


def _list_datasets(toolset: Toolset) -> _Result:
    listed = datasets.listing(toolset.catalog, toolset.context)
    return listed, tuple(item.restrictions for item in listed.datasets)


def _describe_dataset(toolset: Toolset, dataset: str) -> _Result:
    result = datasets.describe(toolset.catalog, toolset.sources, dataset, toolset.context)
    if isinstance(result, datasets.Refused):
        return Refusal(result.reason), ()
    return result, (result.restrictions,)


def _read_dataset(toolset: Toolset, dataset: str) -> _Result:
    result = datasets.read(toolset.catalog, toolset.datasets, dataset, toolset.context)
    if isinstance(result, datasets.Refused):
        return Refusal(result.reason), ()
    return result, (result.restrictions,)


def _lookup_location(toolset: Toolset, ref: str) -> _Result:
    match = _REF.fullmatch(ref)
    if match is None or int(match["start"]) >= int(match["end"]):
        return Refusal("invalid_arguments"), ()
    extractions = {
        generation.extraction_id for generation in corpus_members(toolset.sources, toolset.context)
    }
    wanted = (match["kind"], int(match["start"]), int(match["end"]))
    for generation in published_locations(toolset.sources, toolset.context):
        if generation.id == match["id"] and generation.extraction_id in extractions:
            for location in generation.locations:
                if (location.kind, location.start, location.end) == wanted:
                    return (
                        Located(ref, generation.occurrence.id, generation.digest, location),
                        (generation.occurrence.declaration.restrictions,),
                    )
    # Restricted, later, outside the corpus, and nonexistent refs share one refusal.
    return Refusal("unknown_ref"), ()


def _read_lessons(toolset: Toolset) -> _Result:
    ledger, context = toolset.ledger, toolset.context
    admissible = ledger.admissible_lesson_restrictions(context)
    return (
        Lessons(
            context.memory,
            tuple(Recalled(identity, ledger.record(identity)) for identity, _ in admissible),
        ),
        tuple(restrictions for _, restrictions in admissible),
    )


def _run_signal(toolset: Toolset, source: str, dataset: str) -> _Result:
    if toolset.generated is None:
        return Refusal("generated_code_off"), ()
    read, inputs = _read_dataset(toolset, dataset)
    return (_Candidate(source, read) if isinstance(read, Supplied) else read), inputs


def _schema(**properties: dict[str, object]) -> dict[str, object]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_TOOLS: dict[str, tuple[Tool, Callable[..., _Result]]] = {
    tool.name: (tool, handler)
    for tool, handler in (
        (
            Tool(
                "search_corpus",
                "Search the source corpus pinned for this run. Returns the matching text chunks "
                f"this run may read as of its cutoff, at most {MAX_UNITS} and first in the mode's "
                "order, each with its source occurrence id, generation id, and code-point span. "
                "Chunks this run may not read never appear.",
                _schema(
                    mode={
                        "type": "string",
                        "enum": list(get_args(Mode.__value__)),
                        "description": "identifier returns, in corpus order, the chunks that "
                        "contain the exact, case-sensitive text with no word character on either "
                        "side. lexical ranks chunks by BM25 over the words of the text.",
                    },
                    text={"type": "string", "description": "Text to find. It must not be blank."},
                ),
            ),
            _search_corpus,
        ),
        (
            Tool(
                "list_datasets",
                "Return the datasets this run may read, in name order, each with its owner and "
                "fields. A dataset this run may not read never appears.",
                _schema(),
            ),
            _list_datasets,
        ),
        (
            Tool(
                "describe_dataset",
                "Return the contract of a dataset this run may read: the text of each contract "
                "part available by this run's cutoff, in order of availability so the structure "
                "comes before later conclusions, each chunk with its source occurrence id, "
                "generation id, and code-point span. A dataset this run may not read gets the "
                "same refusal as a name that does not exist.",
                _schema(
                    dataset={"type": "string", "description": "Lowercase catalog dataset name."}
                ),
            ),
            _describe_dataset,
        ),
        (
            Tool(
                "read_dataset",
                "Read a dataset approved for agent use, as of this run's cutoff. Returns, for "
                "each key, the latest revision available by the cutoff, or a refusal that says "
                f"why the dataset cannot be read. A result larger than {MAX_RESULT_BYTES} bytes "
                "is withheld whole, with a notice that says so.",
                _schema(
                    dataset={"type": "string", "description": "Lowercase catalog dataset name."}
                ),
            ),
            _read_dataset,
        ),
        (
            Tool(
                "lookup_location",
                "Return one located element of a source in the corpus pinned for this run, with "
                "its kind, text, and code-point span. A ref this run may not read gets the same "
                "refusal as a ref that names nothing.",
                _schema(
                    ref={
                        "type": "string",
                        "description": "Location ref <generation>:<kind>:<start>-<end>, where "
                        "generation is a 64-character lowercase hex locator generation id, kind "
                        f"is one of {', '.join(_KINDS)}, and start and end are code-point "
                        "offsets.",
                    }
                ),
            ),
            _lookup_location,
        ),
        (
            Tool(
                "read_lessons",
                "Return the reviewed lessons in the memory pinned for this run that this run may "
                "read as of its cutoff, in proposal order, each with its record id, statement, "
                "author, scope, and evidence links. Lessons this run may not read never appear.",
                _schema(),
            ),
            _read_lessons,
        ),
        (
            Tool(
                "run_signal",
                "Run Python source over a dataset approved for agent use, as of this run's "
                "cutoff, in an isolated container with Python 3.12, numpy, and pandas, no "
                "network, and limits on time, memory, and processes. The source reads "
                "/job/slice.json, a JSON object whose rows list, for each key, the latest "
                "revision this run may read with its key, revision, published_at, arrived_at, "
                "and values. It must write only /output/signal.json, one JSON object within the "
                "output size limit that maps every row key to a finite number and repeats no key. "
                "Returns the logged run id with the signal's digest or the reason the run has no "
                "signal. Nothing the source prints is returned.",
                _schema(
                    source={"type": "string", "description": "Python source to run."},
                    dataset={"type": "string", "description": "Lowercase catalog dataset name."},
                ),
            ),
            _run_signal,
        ),
    )
}

TOOLS: tuple[Tool, ...] = tuple(tool for tool, _ in _TOOLS.values())
