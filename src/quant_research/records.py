from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID


class PackageError(ValueError):
    """Report failed record validation, filesystem access, or package integrity."""


class ObjectConflict(ValueError):
    """Name a content digest whose stored object contains different bytes."""


def canonical_bytes(value: object) -> bytes:
    """Return a value as sorted, compact UTF-8 JSON bytes with a trailing newline."""
    content = (
        json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        + "\n"
    )
    try:
        return content.encode("utf-8")
    except UnicodeEncodeError:
        # Raising after the handler chains nothing, since the encode error keeps the whole text.
        pass
    raise PackageError("Canonical JSON text must be valid UTF-8")


def canonical_json(content: bytes, label: str) -> object:
    """Return the JSON value whose canonical bytes are exactly content."""
    try:
        value = json.loads(content.decode("utf-8"))
        canonical = canonical_bytes(value)
    except (ValueError, RecursionError) as error:
        # Raising after the handler chains nothing, since decode errors keep the whole content.
        failure = f"{label} is not valid JSON: {error}"
    else:
        if canonical == content:
            return value
        failure = f"{label} is not canonical UTF-8 JSON"
    raise PackageError(failure)


def write_once(path: Path, content: bytes) -> bool:
    """Create path exclusively and report whether its stored bytes equal content."""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".") as stream:
        stream.write(content)
        stream.flush()
        try:
            os.link(stream.name, path)
        except FileExistsError:
            return path.read_bytes() == content
    return True


def keep_object(directory: Path, content: bytes) -> str:
    """Store original bytes under their SHA-256 in an existing directory; return the digest."""
    digest = hashlib.sha256(content).hexdigest()
    if not write_once(directory / digest, content):
        raise ObjectConflict(digest)
    return digest


def read_log(directory: Path, label: str, files: str) -> Iterator[object]:
    """Yield the canonical JSON value of each file 0.json, 1.json, and so on, in order.

    Hidden files, such as the temporary files an interrupted write leaves, are skipped. An error
    names a bad file after label, and a gap or a stray file after files.
    """
    names = {path.name for path in directory.iterdir() if not path.name.startswith(".")}
    if names != {f"{index}.json" for index in range(len(names))}:
        raise PackageError(f"{files} must be numbered from 0.json without gaps")
    for index in range(len(names)):
        yield canonical_json((directory / f"{index}.json").read_bytes(), f"{label} {index}.json")


def append_log[T](directory: Path, attempt: Callable[[], tuple[int, T, bytes | None]]) -> T:
    """Store the next file of a read_log directory and return the result of its attempt.

    attempt replays the stored files and returns how many there were, its result, and the
    bytes of the next file, or None to store nothing. When another writer has already taken
    that number with other bytes, attempt runs again on the longer log.
    """
    while True:
        count, result, content = attempt()
        if content is None or write_once(directory / f"{count}.json", content):
            return result


def json_default(value: UUID | datetime) -> str:
    """Encode a UUID or timezone-aware datetime as its canonical record text."""
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return format_time(value)
    raise TypeError(f"Unsupported record value {value!r}")


def format_time(value: datetime) -> str:
    """Return a timezone-aware datetime as UTC ISO 8601 text ending in Z."""
    if value.utcoffset() is None:
        raise PackageError("Timestamps must include a timezone")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def parse_fields(value: object, names: str, label: str) -> dict:
    """Return a JSON object whose keys are exactly the space-separated names."""
    if not isinstance(value, dict):
        raise PackageError(f"{label} must be a JSON object")
    expected = set(names.split())
    problems = [
        f"{kind} {', '.join(map(repr, sorted(keys)))}"
        for kind, keys in (
            ("missing", expected - value.keys()),
            ("unknown", value.keys() - expected),
        )
        if keys
    ]
    if problems:
        raise PackageError(f"Invalid fields for {label}; {'; '.join(problems)}")
    return value


def parse_text(value: object, label: str) -> str:
    """Return nonblank text without NUL characters."""
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise PackageError(f"{label} must be nonempty text")
    return value


def parse_digest(value: object, label: str = "SHA-256 digest") -> str:
    """Return a lowercase hexadecimal SHA-256 digest; errors name label, never the value."""
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise PackageError(f"Invalid {label}")
    return value


def parse_name(value: object, label: str = "logical artifact") -> str:
    """Return a lowercase logical name of at most 128 characters; errors omit the value."""
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_.-]{0,127}", value):
        raise PackageError(f"Invalid {label} name")
    return value


def parse_uuid(value: object, label: str = "UUID") -> UUID:
    """Return a nonzero canonical lowercase hyphenated UUID; errors name label, not the value."""
    if not isinstance(value, str):
        raise PackageError(f"Invalid {label}")
    try:
        result = UUID(value)
    except ValueError:
        # Raising after the handler chains nothing, since the parse error can quote the value.
        failure = f"Invalid {label}"
    else:
        if str(result) == value and result.int != 0:
            return result
        failure = f"Invalid canonical {label}"
    raise PackageError(failure)


def parse_time(value: object, label: str = "UTC timestamp") -> datetime:
    """Return a UTC datetime from the text that format_time writes; errors name label."""
    if not isinstance(value, str):
        raise PackageError(f"Invalid {label}")
    try:
        result = datetime.fromisoformat(value)
    except ValueError:
        # Raising after the handler chains nothing, since the parse error quotes the value.
        failure = f"Invalid {label}"
    else:
        if result.utcoffset() == timedelta(0) and format_time(result) == value:
            return result
        failure = f"Invalid canonical {label}"
    raise PackageError(failure)


def parse_array(value: object, label: str) -> list:
    """Return a JSON array."""
    if not isinstance(value, list):
        raise PackageError(f"{label} must be an array")
    return value


def unique_keys(pairs: list[tuple[str, object]]) -> dict:
    """Return JSON object pairs as a dict, raising ValueError for a repeated key."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result
