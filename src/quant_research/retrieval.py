from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from quant_research.access import Restrictions, RunContext
from quant_research.records import (
    PackageError,
    canonical_bytes,
    format_time,
    parse_array,
    parse_digest,
    parse_fields,
    parse_text,
    parse_time,
    write_once,
)
from quant_research.sources import Generation, SourceError, open_generation, published_generations

type Mode = Literal["identifier", "lexical"]


class RetrievalError(ValueError):
    """Report an invalid corpus, query, or retrieval record, or a replay that differs."""


@dataclass(frozen=True)
class Corpus:
    """Published generations admitted together, at most one per source occurrence.

    generations maps each admitted generation id to its manifest digest, which pins content.
    """

    generations: Mapping[str, str]

    @property
    def id(self) -> str:
        return hashlib.sha256(_corpus_record(self)).hexdigest()


@dataclass(frozen=True)
class Query:
    """A search request; mode names the matcher."""

    mode: Mode
    text: str

    def __post_init__(self) -> None:
        # A blank identifier matches at every boundary of every unit, read_retrieval refuses
        # NUL in query text, and retrieval_bytes cannot encode a lone surrogate as UTF-8.
        if (
            self.mode not in _MATCHERS
            or not self.text.strip()
            or "\x00" in self.text
            or re.search(r"[\ud800-\udfff]", self.text)
        ):
            raise RetrievalError(
                f"A query needs a mode in {sorted(_MATCHERS)} and nonblank UTF-8 text without "
                f"NUL, not {self.mode!r} and {self.text!r}"
            )


@dataclass(frozen=True)
class Unit:
    """One chunk of an admitted generation, located by code-point offsets."""

    occurrence_id: str
    generation_id: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Retrieval:
    """Exactly what one query supplied, under the run context whose identity it records."""

    query: Query
    context: str
    generation: str
    cutoff: datetime
    units: tuple[Unit, ...]


def _corpus_record(corpus: Corpus) -> bytes:
    return canonical_bytes({"schema_version": 1, "generations": dict(corpus.generations)})


def _corpus(raw: object) -> Corpus:
    value = parse_fields(raw, "schema_version generations", "corpus record")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise RetrievalError(f"Unsupported corpus schema version {value['schema_version']!r}")
    pins = value["generations"]
    if not isinstance(pins, dict):
        raise RetrievalError("Corpus generations must be a JSON object")
    return Corpus({parse_digest(name): parse_digest(digest) for name, digest in pins.items()})


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def _identifier(units: tuple[Unit, ...], text: str) -> tuple[Unit, ...]:
    pattern = re.compile(rf"(?<!\w){re.escape(text)}(?!\w)")
    return tuple(unit for unit in units if pattern.search(unit.text))


def _lexical(units: tuple[Unit, ...], text: str) -> tuple[Unit, ...]:
    k1, b = 1.2, 0.75
    documents = [Counter(_tokens(unit.text)) for unit in units]
    # Summing in query order, never set order, keeps float rounding and so ties fixed across
    # processes, which replay relies on.
    idfs = {
        term: math.log(1 + (len(units) - n + 0.5) / (n + 0.5))
        for term in dict.fromkeys(_tokens(text))
        if (n := sum(term in counts for counts in documents))
    }
    if not idfs:
        return ()
    average = sum(counts.total() for counts in documents) / len(units)
    scored = []
    for unit, counts in zip(units, documents, strict=True):
        norm = k1 * (1 - b + b * counts.total() / average)
        score = sum(
            idf * counts[term] * (k1 + 1) / (counts[term] + norm)
            for term, idf in idfs.items()
            if counts[term]
        )
        if score > 0:
            scored.append((score, unit))
    return tuple(unit for _, unit in sorted(scored, key=lambda pair: -pair[0]))


_MATCHERS: dict[Mode, Callable[[tuple[Unit, ...], str], tuple[Unit, ...]]] = {
    "identifier": _identifier,
    "lexical": _lexical,
}


def publish_corpus(store: Path, generation_ids: Iterable[str]) -> Corpus:
    """Pin published generations into a corpus a run context can name, recorded once.

    Args:
        store (Path): Store that holds the published generations.
        generation_ids (Iterable[str]): Ids of published generations to admit.

    Returns:
        (Corpus) The corpus, equal on every repeat; its id is the context generation.

    Raises:
        RetrievalError: If an id names no published generation, a generation fails
            verification, two ids process one source occurrence, or file access fails.
    """
    try:
        admitted: dict[str, Generation] = {}
        for generation_id in generation_ids:
            generation = open_generation(store, parse_digest(generation_id))
            first = admitted.setdefault(generation.occurrence.id, generation)
            if first.id != generation.id:
                raise RetrievalError(
                    f"Generations {first.id} and {generation.id} both process occurrence "
                    f"{generation.occurrence.id}"
                )
        corpus = Corpus({generation.id: generation.digest for generation in admitted.values()})
        (store / "corpora").mkdir(exist_ok=True)
        if not write_once(store / "corpora" / f"{corpus.id}.json", _corpus_record(corpus)):
            raise RetrievalError(f"Existing corpus record {corpus.id} does not match its id")
    except (OSError, PackageError, SourceError) as error:
        raise RetrievalError(f"Cannot publish a corpus in {store}: {error}") from error
    return corpus


def corpus_members(store: Path, context: RunContext) -> tuple[Generation, ...]:
    """Return every generation of the context's corpus the run may read, in corpus order.

    Args:
        store (Path): Store that holds the corpus and its generations.
        context (RunContext): Trusted run authority whose generation names a published corpus.

    Returns:
        (tuple) Readable member generations, ordered by published_generations.

    Raises:
        RetrievalError: If the corpus is unpublished or damaged, or a member generation is
            missing or differs from its pinned digest.
        SourceError: If context is not a RunContext or any published generation fails
            verification.
    """
    readable = published_generations(store, context)
    try:
        content = (store / "corpora" / f"{parse_digest(context.generation)}.json").read_bytes()
        corpus = _corpus(json.loads(content.decode("utf-8")))
    except (OSError, ValueError, RecursionError) as error:
        raise RetrievalError(f"Cannot open corpus {context.generation!r}: {error}") from error
    # Comparing the parsed corpus, not the stored bytes, refuses a record whose duplicate
    # key hides pins its hash covers.
    if corpus.id != context.generation:
        raise RetrievalError(f"Corpus record does not derive its id {context.generation}")
    pins = corpus.generations
    if missing := sorted(name for name in pins if not (store / "generations" / name).is_dir()):
        raise RetrievalError(f"Corpus {context.generation} members are missing: {missing}")
    members = tuple(generation for generation in readable if generation.id in pins)
    for generation in members:
        if generation.digest != pins[generation.id]:
            raise RetrievalError(f"Generation {generation.id} differs from its corpus pin")
    return members


def eligible_units(store: Path, context: RunContext) -> tuple[Unit, ...]:
    """Return every unit of the context's corpus the run may read, in corpus order.

    Args:
        store (Path): Store that holds the corpus and its generations.
        context (RunContext): Trusted run authority whose generation names a published corpus.

    Returns:
        (tuple) Units ordered by published_generations, then by offset within a generation.

    Raises:
        RetrievalError: As corpus_members raises it.
        SourceError: As corpus_members raises it.
    """
    return tuple(
        Unit(generation.occurrence.id, generation.id, chunk.start, chunk.end, chunk.text)
        for generation in corpus_members(store, context)
        for chunk in generation.chunks
    )


def eligible_restrictions(store: Path, context: RunContext) -> tuple[Restrictions, ...]:
    """Return the occurrence restrictions of every corpus member that holds an eligible unit.

    Args:
        store (Path): Store that holds the corpus and its generations.
        context (RunContext): Trusted run authority whose generation names a published corpus.

    Returns:
        (tuple) Restrictions of the readable members with a chunk, in corpus order.

    Raises:
        RetrievalError: As corpus_members raises it.
        SourceError: As corpus_members raises it.
    """
    return tuple(
        generation.occurrence.declaration.restrictions
        for generation in corpus_members(store, context)
        if generation.chunks
    )


def retrieve(store: Path, context: RunContext, query: Query) -> Retrieval:
    """Match a query against the exact eligible corpus of a run.

    Args:
        store (Path): Store that holds the corpus and its generations.
        context (RunContext): Trusted run authority whose generation names a published corpus.
        query (Query): Identifier query, matched exactly in corpus order, or lexical query,
            ranked by BM25 over the eligible units alone.

    Returns:
        (Retrieval) The matching units under the context's identity, cutoff, and corpus.

    Raises:
        RetrievalError: If the corpus is unpublished, damaged, or no longer matches its pins.
        SourceError: If context is not a RunContext or any published generation fails
            verification.
    """
    units = eligible_units(store, context)
    return Retrieval(
        query,
        context.identity,
        context.generation,
        context.cutoff,
        _MATCHERS[query.mode](units, query.text),
    )


def replay(store: Path, context: RunContext, retrieval: Retrieval) -> Retrieval:
    """Re-execute a recorded retrieval and require the identical result.

    Args:
        store (Path): Store that holds the corpus and its generations.
        context (RunContext): Run authority whose identity the record names.
        retrieval (Retrieval): Recorded retrieval, the authority on what was supplied.

    Returns:
        (Retrieval) The re-executed retrieval, equal to the record.

    Raises:
        RetrievalError: If the context is not the recorded one or the re-executed retrieval
            differs from the record in any field or unit order.
        SourceError: If context is not a RunContext or any published generation fails
            verification.
    """
    if context.identity != retrieval.context:
        raise RetrievalError(f"Replay context is not the recorded context {retrieval.context}")
    replayed = retrieve(store, context, retrieval.query)
    if replayed != retrieval:
        raise RetrievalError("Re-executed retrieval differs from the record")
    return replayed


def retrieval_bytes(retrieval: Retrieval) -> bytes:
    """Return the canonical stored form of a retrieval.

    Args:
        retrieval (Retrieval): Retrieval to serialize.

    Returns:
        (bytes) Sorted-key UTF-8 JSON with the units in order and a trailing newline.
    """
    return canonical_bytes(
        {"schema_version": 1, **asdict(retrieval), "cutoff": format_time(retrieval.cutoff)}
    )


def _unit(raw: object) -> Unit:
    value = parse_fields(raw, "occurrence_id generation_id start end text", "unit")
    start, end, text = value["start"], value["end"], value["text"]
    if not (
        type(start) is int
        and type(end) is int
        and 0 <= start < end
        and isinstance(text, str)
        and len(text) == end - start
    ):
        raise RetrievalError(f"Unit text {text!r} does not fill a valid span ({start!r}, {end!r})")
    return Unit(
        parse_digest(value["occurrence_id"]), parse_digest(value["generation_id"]), start, end, text
    )


def read_retrieval(content: bytes) -> Retrieval:
    """Verify and open a stored retrieval.

    Args:
        content (bytes): Stored retrieval, as retrieval_bytes writes it.

    Returns:
        (Retrieval) The recorded retrieval.

    Raises:
        RetrievalError: If the bytes are not a canonical, well-formed retrieval record.
    """
    try:
        value = parse_fields(
            json.loads(content.decode("utf-8")),
            "schema_version query context generation cutoff units",
            "retrieval record",
        )
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise RetrievalError(f"Unsupported schema version {value['schema_version']!r}")
        if canonical_bytes(value) != content:
            raise RetrievalError("Retrieval record is not canonical UTF-8 JSON")
        query = parse_fields(value["query"], "mode text", "query")
        return Retrieval(
            Query(parse_text(query["mode"], "query mode"), parse_text(query["text"], "query text")),
            parse_digest(value["context"]),
            parse_digest(value["generation"]),
            parse_time(value["cutoff"]),
            tuple(_unit(item) for item in parse_array(value["units"], "units")),
        )
    except (ValueError, RecursionError) as error:
        raise RetrievalError(f"Cannot read retrieval record: {error}") from error
