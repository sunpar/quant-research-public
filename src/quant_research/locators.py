from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import cached_property, partial
from pathlib import Path
from typing import Literal, get_args

from quant_research.access import RunContext
from quant_research.records import (
    PackageError,
    canonical_bytes,
    parse_array,
    parse_digest,
    parse_fields,
)
from quant_research.sources import (
    TEXT_PIPELINE,
    IdentityConflict,
    SourceError,
    SourceOccurrence,
    Step,
    extracted_text,
    extraction_record,
    identify_extraction,
    load_occurrence,
    paragraphs,
    parse_step,
    publish_manifest,
    read_manifest,
    readable,
)

type Kind = Literal["equation", "table", "figure", "claim"]

_KINDS = get_args(Kind.__value__)


@dataclass(frozen=True)
class Location:
    """One located element, by Unicode code-point offsets into the extracted text."""

    kind: Kind
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class LocatorPipeline:
    """Declared steps and their implementations; locate yields (kind, start, end) spans."""

    extractor: Step
    extract: Callable[[bytes, str], str]
    locator: Step
    locate: Callable[[str], Iterable[tuple[Kind, int, int]]]


@dataclass(frozen=True)
class LocatorGeneration:
    """Every element one locator version finds in one published extraction.

    recorded_at is the UTC clock reading of the first publication, outside id and digest.
    """

    occurrence: SourceOccurrence
    extractor: Step
    locator: Step
    locations: tuple[Location, ...]
    text_digest: str
    length: int
    recorded_at: datetime

    @property
    def extraction_id(self) -> str:
        return identify_extraction(self.occurrence, self.extractor)

    @cached_property
    def id(self) -> str:
        """Return the declared identity, which names the published directory."""
        return hashlib.sha256(
            canonical_bytes({"extraction_id": self.extraction_id, "locator": asdict(self.locator)})
        ).hexdigest()

    @cached_property
    def digest(self) -> str:
        """Return the content identity, the SHA-256 of the canonical manifest."""
        return hashlib.sha256(canonical_bytes(_manifest_data(self))).hexdigest()

    @cached_property
    def _by_ref(self) -> dict[str, Location]:
        return {
            f"{self.id}:{location.kind}:{location.start}-{location.end}": location
            for location in self.locations
        }

    def lookup(self, ref: str) -> Location | None:
        """Return the location that ref names in this generation, or None."""
        return self._by_ref.get(ref)

    def ref(self, location: Location) -> str:
        """Name one located element; the id already carries the extraction and locator version.

        Raises:
            SourceError: If the location is not in this generation.
        """
        ref = f"{self.id}:{location.kind}:{location.start}-{location.end}"
        if self.lookup(ref) != location:
            raise SourceError(
                f"Location {location.kind} {location.start}-{location.end} is not in "
                f"locator generation {self.id}"
            )
        return ref


type _Rule = tuple[Kind | None, Callable[[str], Iterable[tuple[int, int]]], str]

_BLANK, _OPAQUE, _BREAK = " ", "\x00", "\n"
_BOM = re.compile(r"\A\ufeff")
_WITHIN_PARAGRAPH = r"(?:(?!\n\s*\n).)"
_FENCED_CODE = re.compile(
    r"^[ ]{0,3}(?P<fence>(?P<mark>[`~])(?P=mark){2,})(?!(?<=`)[^\n]*`)[^\n]*$.*?"
    r"(?:^[ ]{0,3}(?P=fence)(?P=mark)*[ \t]*\r?$|\Z)",
    re.M | re.S,
)
_CODE_SPAN = re.compile(
    rf"(?<!`)(?P<ticks>`+)(?!`){_WITHIN_PARAGRAPH}+?(?<!`)(?P=ticks)(?!`)", re.S
)
_DISPLAY_MATH = re.compile(
    rf"\$\${_WITHIN_PARAGRAPH}+?\$\$"
    rf"|\\\[{_WITHIN_PARAGRAPH}+?\\\]"
    r"|\\begin\{(?P<environment>(?:equation|align|gather|multline)\*?)\}"
    rf"{_WITHIN_PARAGRAPH}*?\\end\{{(?P=environment)\}}",
    re.S,
)
# Pandoc's rule: $ opens before a non-space and closes after one, never before a digit.
_INLINE_MATH = re.compile(
    rf"\\\({_WITHIN_PARAGRAPH}+?\\\)"
    r"|(?<![\\$])\$(?![\s$])(?:\\.|[^\\$\n]|\n(?!\s*\n))+?(?<![\s\\])\$(?!\d)",
    re.S,
)
_ALT = r"!\[(?P<alt>(?:[^\[\]\n]|\[[^\[\]\n]*\])*)\]"
_TITLE = r"""(?:[ \t]+(?:"[^"\n]*"|'[^'\n]*'|\([^()\n]*\)))?"""
_IMAGE = re.compile(rf"{_ALT}\((?:<[^<>\n]*>|(?:[^()\s]|\([^()\s]*\))+){_TITLE}\)")
_REFERENCE_IMAGE = re.compile(rf"{_ALT}(?:\[(?P<label>[^\[\]\n]*)\])?")
_DEFINITION = re.compile(
    rf"^[ ]{{0,3}}\[(?!\^)(?P<label>[^\S\n]*[^\[\]\s][^\[\]\n]*)\]:"
    rf"[ \t]*(?:<[^<>\n]*>|\S+){_TITLE}[ \t]*$",
    re.M,
)
_HTML_COMMENT = re.compile(r"<!--(?:-?>|.*?-->)", re.S)
_INDENTED = re.compile(r"[ ]{4}|[ ]{0,3}\t")
_LIST_ITEM = re.compile(r"[ ]{0,3}(?:[-*+]|\d{1,9}[.)])(?:[ \t]|$)")


def _table(prefix: str) -> re.Pattern[str]:
    return re.compile(
        rf"^{prefix}(?P<table>[^\n|]*+\|[^\n]*+\n"
        rf"{prefix}[ \t]*+\|?(?:[ \t]*:?-+:?[ \t]*\|)++(?:[ \t]*:?-+:?)?[ \t]*\r?$"
        rf"(?:\n{prefix}[^\n]*\|[^\n]*)*)",
        re.M,
    )


_TABLE = _table("")
_QUOTED_TABLE = _table(r"(?:[ ]{0,3}>[ ]?)++")
_HEADING = re.compile(r"^[ ]{0,3}#{1,6}(?:[ \t][^\n]*)?\r?$", re.M)
# Starting only after a blank line keeps the scan linear and leaves a lazy continuation alone.
_SETEXT_HEADING = re.compile(
    rf"(?:\A|^[^\S\n]*\n)(?:(?!{_LIST_ITEM.pattern}|[ ]{{0,3}}>)[^\S\n]*\S[^\n]*\n)+"
    r"[ ]{0,3}(?:=+|-+)[ \t]*$",
    re.M,
)
_MARKER = re.compile(r"^[ \t]*(?:(?:[-*+]|\d{1,9}[.)])[ \t]+|>[ \t]*)+", re.M)
_CAPTION = re.compile(r"(?:Figure|Fig\.)[ \t]+\d+[.:]")
_SENTENCE_END = re.compile(r"[.!?](?=\s|\Z)")
_ABBREVIATIONS = ("e.g.", "i.e.", "et al.", "Fig.", "Eq.", "cf.", "vs.")
_CLAIM_CUES = (
    r"\d\s?\\?%",
    r"\d\s+(?:percent|basis points)\b",
    r"\bt-stat",
    r"\bp-value",
    r"\bsignifican(?:t|ce)",
    r"\boutperform",
    r"\bwe (?:find|show|document|estimate|report)\b",
    r"\bresults (?:show|suggest|indicate)\b",
)
_ABBREVIATION = re.compile(rf"(?<![\w.])(?:{'|'.join(map(re.escape, _ABBREVIATIONS))})\Z")
_CLAIM_CUE = re.compile("|".join(_CLAIM_CUES), re.IGNORECASE)


def _spans(
    pattern: re.Pattern[str], group: int | str = 0
) -> Callable[[str], Iterator[tuple[int, int]]]:
    return lambda text: (match.span(group) for match in pattern.finditer(text))


def _label(name: str) -> str:
    return " ".join(name.split()).casefold()


def _html_comments(text: str) -> Iterator[tuple[int, int]]:
    # Nothing after an unclosed opener can close, so stopping there keeps the scan linear.
    start = text.find("<!--")
    while start >= 0 and (match := _HTML_COMMENT.match(text, start)):
        yield match.span()
        start = text.find("<!--", match.end())


def _reference_images(text: str) -> Iterator[tuple[int, int]]:
    defined = {_label(match["label"]) for match in _DEFINITION.finditer(text)}
    for match in _REFERENCE_IMAGE.finditer(text):
        if _label(match["label"] or match["alt"]) in defined:
            yield match.span()


def _indented_code(text: str) -> Iterator[tuple[int, int]]:
    # An indented run after a list item continues the item, as pandoc reads it.
    code, after_blank, listed, offset = False, True, False, 0
    for line in text.split("\n"):
        content = line.strip()
        if content and _INDENTED.match(line):
            code = code or (after_blank and not listed)
            if code:
                yield offset, offset + len(line)
        elif content:
            code, listed = False, bool(_LIST_ITEM.match(line))
        after_blank = not content
        offset += len(line) + 1


_UNREAD: tuple[_Rule, ...] = (
    (None, _spans(_BOM), _BLANK),
    (None, _spans(_FENCED_CODE), _BLANK),
    (None, _spans(_CODE_SPAN), _OPAQUE),
    ("equation", _spans(_DISPLAY_MATH), _OPAQUE),
    ("figure", _spans(_IMAGE), _BLANK),
)
_STRUCTURE: tuple[_Rule, ...] = (
    ("equation", _spans(_INLINE_MATH), _OPAQUE),
    ("table", _spans(_TABLE), _BLANK),
    (None, _spans(_HEADING), _BLANK),
    (None, _spans(_MARKER), _BREAK),
)
_UNREAD_V2: tuple[_Rule, ...] = (
    (None, _spans(re.compile(r"\r(?!\n)")), _BREAK),
    (None, _spans(re.compile(r"\r")), _BLANK),
    (None, _spans(_BOM), _BLANK),
    (None, _spans(_FENCED_CODE), _BLANK),
    (None, _indented_code, _BLANK),
    (None, _spans(_CODE_SPAN), _OPAQUE),
    (None, _html_comments, _BLANK),
    ("equation", _spans(_DISPLAY_MATH), _OPAQUE),
    ("figure", _spans(_IMAGE), _BLANK),
    ("figure", _reference_images, _BLANK),
    (None, _spans(_DEFINITION), _BLANK),
)
_STRUCTURE_V2: tuple[_Rule, ...] = (
    ("equation", _spans(_INLINE_MATH), _OPAQUE),
    ("table", _spans(_TABLE), _BLANK),
    ("table", _spans(_QUOTED_TABLE, "table"), _BLANK),
    (None, _spans(_HEADING), _BLANK),
    (None, _spans(_SETEXT_HEADING), _BLANK),
    (None, _spans(_MARKER), _BREAK),
)


def _trimmed(text: str, start: int, end: int) -> tuple[int, int]:
    piece = text[start:end]
    return start + len(piece) - len(piece.lstrip()), start + len(piece.rstrip())


def _hide(text: str, rules: tuple[_Rule, ...]) -> tuple[str, list[tuple[Kind, int, int]]]:
    found: list[tuple[Kind, int, int]] = []
    for kind, find, fill in rules:
        hidden = list(text)
        for start, end in find(text):
            if kind is not None:
                found.append((kind, *_trimmed(text, start, end)))
            hidden[start:end] = re.sub(r"[^\n]", fill, text[start:end])
        text = "".join(hidden)
    return text, found


def _sentences(text: str, start: int, end: int) -> Iterator[tuple[int, int]]:
    begin = start
    for match in _SENTENCE_END.finditer(text, start, end):
        if not _ABBREVIATION.search(text, begin, match.end()):
            yield _trimmed(text, begin, match.end())
            begin = match.end()
    if text[begin:end].strip():
        yield _trimmed(text, begin, end)


def _markdown_elements(
    unread: tuple[_Rule, ...], structure: tuple[_Rule, ...], text: str
) -> list[tuple[Kind, int, int]]:
    prose, found = _hide(text, unread)
    masked, structural = _hide(prose, structure)
    found.extend(structural)
    for start, end in paragraphs(masked):
        if _CAPTION.match(masked, start):
            found.append(("figure", start, end))
        else:
            found.extend(
                ("claim", *span)
                for span in _sentences(masked, start, end)
                if _CLAIM_CUE.search(prose, *span)
            )
    return sorted(found, key=lambda item: (item[1], item[2], item[0]))


MARKDOWN_LOCATOR = LocatorPipeline(
    TEXT_PIPELINE.extractor,
    TEXT_PIPELINE.extract,
    Step("markdown-elements", "1"),
    partial(_markdown_elements, _UNREAD, _STRUCTURE),
)
MARKDOWN_LOCATOR_V2 = LocatorPipeline(
    TEXT_PIPELINE.extractor,
    TEXT_PIPELINE.extract,
    Step("markdown-elements", "2"),
    partial(_markdown_elements, _UNREAD_V2, _STRUCTURE_V2),
)


def _location(item: object, length: int, locations: list[Location]) -> tuple[Kind, int, int]:
    match item:
        case (kind, start, end) if (
            type(kind) is str
            and kind in _KINDS
            and type(start) is int
            and type(end) is int
            and 0 <= start < end <= length
            and (
                not locations
                or (locations[-1].start, locations[-1].end, locations[-1].kind) < (start, end, kind)
            )
        ):
            return kind, start, end
    raise SourceError(
        f"Invalid location at index {len(locations)}; locations must be (kind, start, end) "
        f"triples that strictly increase by (start, end, kind) within {length} code points"
    )


def _manifest_data(generation: LocatorGeneration) -> dict:
    return {
        "schema_version": 1,
        "occurrence_id": generation.occurrence.id,
        "extraction": {
            "extractor": asdict(generation.extractor),
            "text_digest": generation.text_digest,
            "length": generation.length,
        },
        "locator": asdict(generation.locator),
        "locations": [asdict(location) for location in generation.locations],
    }


def _open_locations(store: Path, generation_id: str) -> LocatorGeneration:
    value, recorded_at = read_manifest(
        store / "locations" / generation_id, "locator generation", "locator locations"
    )
    extraction = value["extraction"]
    locations: list[Location] = []
    for item in parse_array(value["locations"], "locations"):
        location = parse_fields(item, "kind start end text", "location")
        kind, start, end = _location(
            (location["kind"], location["start"], location["end"]),
            extraction["length"],
            locations,
        )
        if not isinstance(location["text"], str) or len(location["text"]) != end - start:
            raise SourceError(f"Location text does not fill its span ({start}, {end})")
        locations.append(Location(kind, start, end, location["text"]))
    generation = LocatorGeneration(
        load_occurrence(store, value["occurrence_id"]),
        parse_step(extraction["extractor"], "extractor"),
        parse_step(value["locator"], "locator"),
        tuple(locations),
        parse_digest(extraction["text_digest"]),
        extraction["length"],
        recorded_at,
    )
    if generation.id != generation_id:
        raise SourceError(
            f"Locator generation manifest does not derive its directory {generation_id}"
        )
    binding = store / "extractions" / f"{generation.extraction_id}.json"
    if binding.read_bytes() != extraction_record(generation.length, generation.text_digest):
        raise SourceError(f"Locator generation {generation_id} contradicts its extraction binding")
    return generation


def publish_locations(
    store: Path, occurrence_id: str, pipeline: LocatorPipeline
) -> LocatorGeneration:
    """Locate elements in a published extraction, then publish the complete generation.

    Args:
        store (Path): Store that holds the occurrence and its published extraction.
        occurrence_id (str): Identity returned by register_source.
        pipeline (LocatorPipeline): Declared extractor and locator with their implementations.

    Returns:
        (LocatorGeneration) The generation as first published, including its recorded_at.

    Raises:
        IdentityConflict: If the extractor's text differs from the published extraction, or
            this generation was published with different content.
        SourceError: If the extraction is unpublished, or records, locate output, or file
            access are invalid.

    Notes:
        publish_generation publishes the extraction. A failed run publishes nothing.
    """
    try:
        occurrence = load_occurrence(store, occurrence_id)
        extractor = parse_step(asdict(pipeline.extractor), "extractor")
        text = extracted_text(store, occurrence, pipeline.extract)
        text_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        extraction_id = identify_extraction(occurrence, extractor)
        binding = store / "extractions" / f"{extraction_id}.json"
        if binding.read_bytes() != extraction_record(len(text), text_digest):
            raise IdentityConflict(f"Extraction {extraction_id} is bound to different text")
        locations: list[Location] = []
        for item in pipeline.locate(text):
            kind, start, end = _location(item, len(text), locations)
            locations.append(Location(kind, start, end, text[start:end]))
        generation = LocatorGeneration(
            occurrence,
            extractor,
            parse_step(asdict(pipeline.locator), "locator"),
            tuple(locations),
            text_digest,
            len(text),
            datetime.now(UTC),
        )
        publish_manifest(
            store / "locations" / generation.id,
            canonical_bytes(_manifest_data(generation)),
            generation.recorded_at,
        )
        published = _open_locations(store, generation.id)
    except (OSError, PackageError, UnicodeEncodeError) as error:
        raise SourceError(f"Cannot publish locations of {occurrence_id}: {error}") from error
    if published.digest != generation.digest:
        raise IdentityConflict(
            f"Locator generation {generation.id} is already published with different content"
        )
    return published


def published_locations(store: Path, context: RunContext) -> tuple[LocatorGeneration, ...]:
    """Return verified published locator generations whose source occurrences a run may read.

    Args:
        store (Path): Existing store directory.
        context (RunContext): Trusted run authority from issue_context or narrow.

    Returns:
        (tuple) Readable locator generations sorted by source_available_at, then id.

    Raises:
        SourceError: If context is not a RunContext, or any published locator generation
            fails verification, including one the context may not read.
    """
    return readable(store, context, "locations", _open_locations)
