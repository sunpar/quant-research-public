from __future__ import annotations

import errno
import hashlib
import json
import tempfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from quant_research.access import AccessError, Provider, RestrictionLabel, Restrictions, RunContext
from quant_research.records import (
    ObjectConflict,
    PackageError,
    canonical_bytes,
    canonical_json,
    format_time,
    keep_object,
    parse_array,
    parse_digest,
    parse_fields,
    parse_text,
    parse_time,
    write_once,
)


class SourceError(ValueError):
    """Report invalid source records, processing output, or store contents."""


class IdentityConflict(SourceError):
    """Report a declared identity already bound to different content."""


@dataclass(frozen=True)
class SourceDeclaration:
    """Caller-declared provenance, restrictions, and timing, never inferred from the bytes.

    labels name the clearances a reader needs, and approved_for the providers the material
    may reach. published_at is the publisher's claimed release, source_available_at the
    earliest supported availability backed by availability_evidence, and received_at the
    actual arrival in this system. Only restrictions decides who may read an occurrence.
    """

    media_type: str
    access_domain: str
    location: str
    revision: str
    acquisition: str
    rights: str
    labels: frozenset[RestrictionLabel]
    approved_for: frozenset[Provider]
    published_at: datetime | None
    source_available_at: datetime
    availability_evidence: str
    received_at: datetime

    @property
    def restrictions(self) -> Restrictions:
        return Restrictions(
            available_at=self.source_available_at,
            labels=self.labels,
            approved_for=self.approved_for,
        )


@dataclass(frozen=True)
class SourceOccurrence:
    """Original bytes under one declaration; its id hashes the canonical declaration."""

    blob_digest: str
    declaration: SourceDeclaration

    @property
    def id(self) -> str:
        record = _occurrence_data(self)
        del record["blob_digest"]
        return hashlib.sha256(canonical_bytes(record)).hexdigest()


@dataclass(frozen=True)
class Step:
    """Declared identity of an extractor or chunker; version names code and configuration."""

    name: str
    version: str


@dataclass(frozen=True)
class Pipeline:
    """Declared steps and their implementations; split yields (start, end) code-point spans."""

    extractor: Step
    extract: Callable[[bytes, str], str]
    chunker: Step
    split: Callable[[str], Iterable[tuple[int, int]]]


@dataclass(frozen=True)
class Chunk:
    """Text located by Unicode code-point offsets into the extracted text."""

    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Generation:
    """Complete output of one occurrence under one declared extractor and chunker.

    recorded_at is the UTC clock reading of the first publication, outside id and digest.
    """

    occurrence: SourceOccurrence
    extractor: Step
    chunker: Step
    chunks: tuple[Chunk, ...]
    text_digest: str
    length: int
    recorded_at: datetime

    @property
    def extraction_id(self) -> str:
        return identify_extraction(self.occurrence, self.extractor)

    @property
    def id(self) -> str:
        """Return the declared identity, which names the published directory."""
        return hashlib.sha256(
            canonical_bytes({"extraction_id": self.extraction_id, "chunker": asdict(self.chunker)})
        ).hexdigest()

    @property
    def digest(self) -> str:
        """Return the content identity, the SHA-256 of the canonical manifest."""
        return hashlib.sha256(canonical_bytes(_manifest_data(self))).hexdigest()


def _decode(data: bytes, media_type: str) -> str:
    if media_type not in ("text/plain", "text/markdown"):
        raise SourceError(f"Unsupported media type {media_type!r} for text extraction")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        # Raising after the handler chains nothing, since the decode error keeps the bytes.
        reason = str(error)
    raise SourceError(f"Source bytes are not valid UTF-8: {reason}")


def paragraphs(text: str) -> Iterator[tuple[int, int]]:
    """Yield the span of each run of nonblank lines, without its outer whitespace."""
    start = None
    offset = 0
    for line in text.splitlines(keepends=True):
        if line.strip():
            if start is None:
                start = offset + len(line) - len(line.lstrip())
            end = offset + len(line.rstrip())
        elif start is not None:
            yield start, end
            start = None
        offset += len(line)
    if start is not None:
        yield start, end


TEXT_PIPELINE = Pipeline(Step("utf-8-text", "1"), _decode, Step("paragraph", "1"), paragraphs)


def _occurrence_data(occurrence: SourceOccurrence) -> dict:
    record = {"schema_version": 1, **asdict(occurrence)}
    restrictions = occurrence.declaration.restrictions
    record["declaration"]["labels"] = sorted(restrictions.labels)
    record["declaration"]["approved_for"] = sorted(restrictions.approved_for)
    return json.loads(json.dumps(record, default=format_time))


def _occurrence(raw: object) -> SourceOccurrence:
    value = parse_fields(raw, "schema_version blob_digest declaration", "occurrence record")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise SourceError(f"Unsupported occurrence schema version {value['schema_version']!r}")
    declaration = parse_fields(
        value["declaration"],
        "media_type access_domain location revision acquisition rights labels approved_for "
        "published_at source_available_at availability_evidence received_at",
        "source declaration",
    )
    return SourceOccurrence(
        parse_digest(value["blob_digest"]),
        SourceDeclaration(
            *(
                parse_text(declaration[key], key)
                for key in (
                    "media_type",
                    "access_domain",
                    "location",
                    "revision",
                    "acquisition",
                    "rights",
                )
            ),
            *(
                frozenset(
                    parse_text(item, f"{key} entry") for item in parse_array(declaration[key], key)
                )
                for key in ("labels", "approved_for")
            ),
            None
            if declaration["published_at"] is None
            else parse_time(declaration["published_at"]),
            parse_time(declaration["source_available_at"]),
            parse_text(declaration["availability_evidence"], "availability_evidence"),
            parse_time(declaration["received_at"]),
        ),
    )


def parse_step(raw: object, label: str) -> Step:
    """Return the named, versioned step stored as plain JSON."""
    value = parse_fields(raw, "name version", label)
    return Step(
        parse_text(value["name"], f"{label} name"), parse_text(value["version"], f"{label} version")
    )


def _span(span: object, length: int, chunks: list[Chunk]) -> tuple[int, int]:
    match span:
        case (start, end) if (
            type(start) is int
            and type(end) is int
            and 0 <= start < end <= length
            and (not chunks or (chunks[-1].start, chunks[-1].end) < (start, end))
        ):
            return start, end
    raise SourceError(
        f"Invalid chunk span at index {len(chunks)}; spans must be integer pairs that strictly "
        f"increase within {length} code points"
    )


def _manifest_data(generation: Generation) -> dict:
    return {
        "schema_version": 1,
        "occurrence_id": generation.occurrence.id,
        "extraction": {
            "extractor": asdict(generation.extractor),
            "text_digest": generation.text_digest,
            "length": generation.length,
        },
        "chunker": asdict(generation.chunker),
        "chunks": [asdict(chunk) for chunk in generation.chunks],
    }


def identify_extraction(occurrence: SourceOccurrence, extractor: Step) -> str:
    """Return the SHA-256 identity of one extractor applied to one occurrence's bytes."""
    return hashlib.sha256(
        canonical_bytes(
            {
                "occurrence_id": occurrence.id,
                "blob_digest": occurrence.blob_digest,
                "extractor": asdict(extractor),
            }
        )
    ).hexdigest()


def extraction_record(length: int, text_digest: str) -> bytes:
    """Return the canonical binding of an extraction to its text length and digest."""
    return canonical_bytes({"length": length, "text_digest": text_digest})


def load_occurrence(store: Path, occurrence_id: str) -> SourceOccurrence:
    """Return the stored occurrence whose record hashes to occurrence_id."""
    path = store / "occurrences" / f"{parse_digest(occurrence_id)}.json"
    occurrence = _occurrence(canonical_json(path.read_bytes(), f"Occurrence {occurrence_id}"))
    if occurrence.id != occurrence_id:
        raise SourceError(f"Occurrence record does not hash to {occurrence_id}")
    return occurrence


def extracted_text(
    store: Path, occurrence: SourceOccurrence, extract: Callable[[bytes, str], str]
) -> str:
    """Return the extracted text of an occurrence's blob after checking the blob's digest."""
    data = (store / "blobs" / occurrence.blob_digest).read_bytes()
    if hashlib.sha256(data).hexdigest() != occurrence.blob_digest:
        raise SourceError(f"Blob {occurrence.blob_digest} does not match its digest")
    text = extract(data, occurrence.declaration.media_type)
    # A str subclass could override the methods that check, hash, slice, and store the text.
    if type(text) is not str:
        raise SourceError(f"Extractor for occurrence {occurrence.id} did not return str")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as error:
        # Raising after the handler chains nothing, since the encode error keeps the text.
        reason = error.reason
    else:
        return text
    raise SourceError(f"Extracted text of occurrence {occurrence.id} cannot encode: {reason}")


def read_manifest(directory: Path, kind: str, fields: str) -> tuple[dict, datetime]:
    """Return a published manifest's checked fields and its first publication time.

    fields names the kind's own fields beside schema_version, occurrence_id, and extraction.
    The returned extraction length is a nonnegative integer.
    """
    label = f"{kind.capitalize()} {directory.name}"
    content = (directory / "manifest.json").read_bytes()
    checksum = f"{hashlib.sha256(content).hexdigest()}\n".encode()
    if (directory / "manifest.sha256").read_bytes() != checksum:
        raise SourceError(f"{label} manifest does not match manifest.sha256")
    value = parse_fields(
        canonical_json(content, f"{label} manifest"),
        f"schema_version occurrence_id extraction {fields}",
        f"{kind} manifest",
    )
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise SourceError(f"Unsupported {kind} schema version {value['schema_version']!r}")
    extraction = parse_fields(value["extraction"], "extractor text_digest length", "extraction")
    length = extraction["length"]
    if type(length) is not int or length < 0:
        raise SourceError(f"Extracted text length {length!r} must be a nonnegative integer")
    publication = parse_fields(
        canonical_json(
            (directory / "publication.json").read_bytes(), f"{label} publication record"
        ),
        "recorded_at",
        "publication record",
    )
    return value, parse_time(publication["recorded_at"])


def open_generation(store: Path, generation_id: str) -> Generation:
    """Return the published generation after checking its manifest and extraction binding."""
    value, recorded_at = read_manifest(
        store / "generations" / generation_id, "generation", "chunker chunks"
    )
    extraction = value["extraction"]
    chunks: list[Chunk] = []
    for item in parse_array(value["chunks"], "chunks"):
        chunk = parse_fields(item, "start end text", "chunk")
        start, end = _span((chunk["start"], chunk["end"]), extraction["length"], chunks)
        if not isinstance(chunk["text"], str) or len(chunk["text"]) != end - start:
            raise SourceError(f"Chunk text does not fill its span ({start}, {end})")
        chunks.append(Chunk(start, end, chunk["text"]))
    generation = Generation(
        load_occurrence(store, value["occurrence_id"]),
        parse_step(extraction["extractor"], "extractor"),
        parse_step(value["chunker"], "chunker"),
        tuple(chunks),
        parse_digest(extraction["text_digest"]),
        extraction["length"],
        recorded_at,
    )
    if generation.id != generation_id:
        raise SourceError(f"Generation manifest does not derive its directory {generation_id}")
    binding = store / "extractions" / f"{generation.extraction_id}.json"
    if binding.read_bytes() != extraction_record(generation.length, generation.text_digest):
        raise SourceError(f"Generation {generation_id} contradicts its extraction binding")
    return generation


def publish_manifest(directory: Path, content: bytes, recorded_at: datetime) -> None:
    """Stage a complete manifest privately and rename it into place unless one is there."""
    directory.parent.mkdir(exist_ok=True)
    if directory.exists():
        return
    with tempfile.TemporaryDirectory(prefix=".stage-", dir=directory.parent) as staging:
        stage = Path(staging)
        (stage / "manifest.json").write_bytes(content)
        (stage / "manifest.sha256").write_text(
            f"{hashlib.sha256(content).hexdigest()}\n", encoding="ascii"
        )
        (stage / "publication.json").write_bytes(
            canonical_bytes({"recorded_at": format_time(recorded_at)})
        )
        try:
            stage.rename(directory)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise


def readable[T](
    store: Path, context: RunContext, kind: str, open_published: Callable[[Path, str], T]
) -> tuple[T, ...]:
    """Open every publication under store / kind, then keep those the context may read."""
    # A stand-in whose read_denials returns () would make every generation readable.
    if not isinstance(context, RunContext):
        raise SourceError(f"context must be a RunContext, not {type(context).__name__}")
    published = store / kind
    try:
        if not store.is_dir():
            raise SourceError(f"Source store {store} is not an existing directory")
        names = [path.name for path in published.iterdir()] if published.exists() else []
        opened = [
            open_published(store, parse_digest(name)) for name in names if not name.startswith(".")
        ]
    except (OSError, PackageError) as error:
        raise SourceError(f"Cannot read published {kind} in {store}: {error}") from error
    return tuple(
        sorted(
            (
                item
                for item in opened
                if not context.read_denials(item.occurrence.declaration.restrictions)
            ),
            key=lambda item: (item.occurrence.declaration.source_available_at, item.id),
        )
    )


def register_source(store: Path, data: bytes, declaration: SourceDeclaration) -> SourceOccurrence:
    """Record source bytes once and each declared occurrence under content-derived names.

    Args:
        store (Path): Existing store directory.
        data (bytes): Original source bytes.
        declaration (SourceDeclaration): Caller-declared provenance, rights, and availability.

    Returns:
        (SourceOccurrence) The registered occurrence, equal on every repeat.

    Raises:
        IdentityConflict: If the same declaration was registered with different bytes.
        SourceError: If the declaration is invalid, a stored blob differs, or file access fails.
    """
    try:
        occurrence = _occurrence(
            _occurrence_data(SourceOccurrence(hashlib.sha256(data).hexdigest(), declaration))
        )
        record = canonical_bytes(_occurrence_data(occurrence))
        for directory in ("blobs", "occurrences"):
            (store / directory).mkdir(exist_ok=True)
        if not write_once(store / "occurrences" / f"{occurrence.id}.json", record):
            raise IdentityConflict(f"Occurrence {occurrence.id} is registered with other bytes")
        keep_object(store / "blobs", data)
    except ObjectConflict as error:
        raise SourceError(f"Existing blob {error} does not match its digest") from error
    except (OSError, PackageError, AccessError) as error:
        raise SourceError(f"Cannot register source in {store}: {error}") from error
    return occurrence


def publish_generation(store: Path, occurrence_id: str, pipeline: Pipeline) -> Generation:
    """Process a registered occurrence in memory, then publish the complete generation.

    Args:
        store (Path): Store that holds the registered occurrence.
        occurrence_id (str): Identity returned by register_source.
        pipeline (Pipeline): Declared extractor and chunker with their implementations.

    Returns:
        (Generation) The generation as first published, including its recorded_at.

    Raises:
        IdentityConflict: If its extraction or generation was bound to different content.
        SourceError: If records, processing output, or file access are invalid.

    Notes:
        A failed run publishes nothing.
    """
    try:
        occurrence = load_occurrence(store, occurrence_id)
        text = extracted_text(store, occurrence, pipeline.extract)
        chunks: list[Chunk] = []
        for span in pipeline.split(text):
            start, end = _span(span, len(text), chunks)
            chunks.append(Chunk(start, end, text[start:end]))
        generation = Generation(
            occurrence,
            parse_step(asdict(pipeline.extractor), "extractor"),
            parse_step(asdict(pipeline.chunker), "chunker"),
            tuple(chunks),
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
            len(text),
            datetime.now(UTC),
        )
        (store / "extractions").mkdir(exist_ok=True)
        if not write_once(
            store / "extractions" / f"{generation.extraction_id}.json",
            extraction_record(generation.length, generation.text_digest),
        ):
            raise IdentityConflict(
                f"Extraction {generation.extraction_id} is already bound to different text"
            )
        publish_manifest(
            store / "generations" / generation.id,
            canonical_bytes(_manifest_data(generation)),
            generation.recorded_at,
        )
        published = open_generation(store, generation.id)
    except (OSError, PackageError, UnicodeEncodeError) as error:
        raise SourceError(f"Cannot publish a generation of {occurrence_id}: {error}") from error
    if published.digest != generation.digest:
        raise IdentityConflict(
            f"Generation {generation.id} is already published with different content"
        )
    return published


def published_generations(store: Path, context: RunContext) -> tuple[Generation, ...]:
    """Return verified published generations whose source occurrences a run may read.

    Args:
        store (Path): Existing store directory.
        context (RunContext): Trusted run authority from issue_context or narrow.

    Returns:
        (tuple) Readable generations sorted by source_available_at, then generation id.

    Raises:
        SourceError: If context is not a RunContext, or any published generation fails
            verification, including one the context may not read.
    """
    return readable(store, context, "generations", open_generation)
