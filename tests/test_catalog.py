from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from quant_research.catalog import (
    VINTAGE_GAPS,
    Approved,
    Catalog,
    CatalogError,
    Documented,
    Entry,
    Evidence,
    Revoked,
    Validated,
    Vintage,
)

TIME = datetime(2026, 10, 21, 12, tzinfo=UTC)
NAME = "vendor.fundamentals"
ADAPTER = "timing.fundamentals-pit"
ASPECTS = ("publication", "revision", "arrival")
KNOWN_GAPS = ("Restatements before 2001 overwrite first releases", "No coverage before 1990")


def evidence(material: str) -> Evidence:
    return Evidence(hashlib.sha256(material.encode()).hexdigest(), f"Synthetic {material}")


VINTAGES = tuple(Vintage(aspect, evidence(f"{aspect} snapshots")) for aspect in ASPECTS)


def documented(
    name=NAME,
    *,
    owner="Data Engineering",
    fields=("gvkey", "rdq", "epspxq"),
    vintages=(),
    approved_for=(),
):
    return Documented(
        name,
        "researcher",
        TIME,
        owner,
        fields,
        ("licensed-internal",),
        evidence("dataset file").digest,
        KNOWN_GAPS,
        vintages,
        approved_for,
    )


def validated(name=NAME):
    return Validated(name, "reviewer", TIME, evidence("reconciliation report"))


def approved(name=NAME):
    return Approved(name, "lead", TIME, ADAPTER)


def revoked(name=NAME):
    return Revoked(name, "lead", TIME, "Adapter leaked restatements")


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def names(entries: tuple[Entry, ...]) -> tuple[str, ...]:
    return tuple(entry.name for entry in entries)


@pytest.fixture
def lifecycle():
    return Catalog(
        (
            documented(vintages=VINTAGES, approved_for=("firm-proxy",)),
            documented("crsp.daily", owner="Équipe données", fields=("permno", "date", "ret")),
            validated(),
            approved(),
            revoked(),
        )
    )


def test_registration_is_documented_with_known_and_vintage_gaps_and_no_adapter():
    registration = documented()
    entry = Catalog().append(registration).entry(NAME)
    assert entry.status == "documented"
    assert entry.history == (registration,)
    assert tuple(VINTAGE_GAPS) == ASPECTS
    assert all(aspect in VINTAGE_GAPS[aspect] for aspect in ASPECTS)
    with pytest.raises(TypeError):
        del VINTAGE_GAPS["arrival"]
    assert entry.gaps == (
        *KNOWN_GAPS,
        VINTAGE_GAPS["publication"],
        VINTAGE_GAPS["revision"],
        VINTAGE_GAPS["arrival"],
    )
    assert entry.adapter is None


def test_documented_dataset_cannot_be_approved_even_with_every_vintage_and_an_adapter():
    catalog = Catalog().append(documented(vintages=VINTAGES))
    with pytest.raises(CatalogError):
        catalog.append(approved())
    assert catalog.append(validated()).append(approved()).entry(NAME).status == "approved"


@pytest.mark.parametrize("missing", ASPECTS)
def test_missing_vintage_aspect_blocks_approval_until_revoked_amended_and_revalidated(missing):
    partial = documented(vintages=tuple(item for item in VINTAGES if item.aspect != missing))
    catalog = Catalog().append(partial).append(validated())
    assert catalog.entry(NAME).status == "validated"
    assert catalog.entry(NAME).gaps == (*KNOWN_GAPS, VINTAGE_GAPS[missing])
    with pytest.raises(CatalogError):
        catalog.append(approved())
    amendment = replace(partial, vintages=VINTAGES)
    with pytest.raises(CatalogError):
        catalog.append(amendment)
    catalog = catalog.append(Revoked(NAME, "lead", TIME, f"No {missing} evidence"))
    with pytest.raises(CatalogError):
        catalog.append(validated())
    catalog = catalog.append(amendment)
    assert catalog.entry(NAME).gaps == KNOWN_GAPS
    entry = catalog.append(validated()).append(approved()).entry(NAME)
    assert entry.status == "approved"
    assert entry.adapter == ADAPTER


def test_validated_dataset_with_every_vintage_and_an_adapter_is_approved():
    catalog = Catalog().append(documented(vintages=VINTAGES)).append(validated())
    entry = catalog.append(approved()).entry(NAME)
    assert entry.status == "approved"
    assert entry.adapter == ADAPTER
    assert entry.gaps == KNOWN_GAPS
    for adapter in ("", "  "):
        with pytest.raises(CatalogError):
            catalog.append(replace(approved(), adapter=adapter))


@pytest.mark.smoke
def test_lifecycle_keeps_a_revoked_approval_visible_through_a_bytes_round_trip():
    events = (documented(vintages=VINTAGES), validated(), approved(), revoked())
    catalog = Catalog()
    statuses = []
    for event in events:
        catalog = catalog.append(event)
        statuses.append(catalog.entry(NAME).status)
    assert statuses == ["documented", "validated", "approved", "revoked"]
    entry = catalog.entry(NAME)
    assert entry.history == events
    assert entry.adapter is None
    assert catalog.search(status="revoked") == (entry,)
    assert catalog.search(status="approved") == ()
    reloaded = Catalog.from_bytes(catalog.to_bytes())
    assert reloaded.entry(NAME) == entry
    assert reloaded.search(status="revoked") == (entry,)


def test_append_returns_a_new_catalog_and_keeps_append_order():
    first, other, validation = documented(), documented("crsp.daily"), validated()
    original = Catalog().append(first)
    extended = original.append(other).append(validation)
    assert original.events == (first,)
    assert names(original.search()) == (NAME,)
    assert original.entry(NAME).history == (first,)
    assert extended.events == (first, other, validation)
    assert extended.entry(NAME).history == (first, validation)
    assert extended.entry("crsp.daily").history == (other,)


def test_bytes_round_trip_keeps_the_catalog_its_entries_and_its_digest(lifecycle):
    data = lifecycle.to_bytes()
    reloaded = Catalog.from_bytes(data)
    assert reloaded == lifecycle
    assert reloaded.search() == lifecycle.search()
    assert reloaded.digest == lifecycle.digest == hashlib.sha256(data).hexdigest()
    assert lifecycle.append(documented("ibes.estimates")).digest != lifecycle.digest


@pytest.mark.parametrize(
    "damage",
    [
        "trailing-space",
        "duplicate-key",
        "truncated",
        "encoding",
        "version",
        "extra-field",
        "unknown-kind",
        "invalid-digest",
        "deep-json",
        "large-integer",
    ],
)
def test_from_bytes_rejects_noncanonical_or_malformed_bytes(lifecycle, damage):
    data = lifecycle.to_bytes()
    match damage:
        case "trailing-space":
            data += b" "
        case "duplicate-key":
            data = data.replace(b'"schema_version":1', b'"schema_version":1,"schema_version":1')
        case "truncated":
            data = data[: len(data) // 2]
        case "encoding":
            data = b"\xff" + data
        case "version":
            data = data.replace(b'"schema_version":1', b'"schema_version":2')
        case "extra-field":
            data = data.replace(b'{"events":', b'{"comment":"unexpected","events":', 1)
        case "unknown-kind":
            data = data.replace(b'"kind":"validated"', b'"kind":"verified"')
        case "invalid-digest":
            data = data.replace(b'"digest":"', b'"digest":"x', 1)
        case "deep-json":
            data = b"[" * 20000 + b"0" + b"]" * 20000
        case "large-integer":
            data = b"1" * 5000
    with pytest.raises(CatalogError):
        Catalog.from_bytes(data)


def test_hand_written_or_directly_built_approval_without_validation_is_refused():
    registration = documented(vintages=VINTAGES)
    catalog = Catalog((registration, validated(), approved()))
    value = json.loads(catalog.to_bytes())
    assert Catalog.from_bytes(canonical(value)) == catalog
    value["events"] = [event for event in value["events"] if event["kind"] != "validated"]
    with pytest.raises(CatalogError):
        Catalog.from_bytes(canonical(value))
    with pytest.raises(CatalogError):
        Catalog((registration, approved()))


@pytest.mark.parametrize(
    "event",
    [
        replace(validated(), evidence=Evidence("0" * 63, "Truncated digest")),
        documented(vintages=(Vintage("arrival", Evidence("Z" * 64, "Not hexadecimal")),)),
        documented(vintages=(Vintage("delivery", evidence("delivery snapshots")),)),
        replace(documented(), vintage="Z" * 64),
        replace(documented(), approved_for=(" ",)),
        replace(validated(), at=TIME.replace(tzinfo=None)),
        validated("unregistered.dataset"),
        approved("unregistered.dataset"),
        revoked("unregistered.dataset"),
    ],
    ids=[
        "validation-digest",
        "vintage-digest",
        "vintage-aspect",
        "dataset-vintage",
        "provider-approval",
        "naive-time",
        "unregistered-validation",
        "unregistered-approval",
        "unregistered-revocation",
    ],
)
def test_malformed_or_unregistered_events_are_refused(event):
    catalog = Catalog().append(documented())
    with pytest.raises(CatalogError):
        catalog.append(event)


def test_an_event_dated_before_the_latest_event_of_its_dataset_is_refused():
    catalog = Catalog().append(documented(vintages=VINTAGES))
    with pytest.raises(CatalogError):
        catalog.append(replace(validated(), at=TIME - timedelta(days=1)))
    later = catalog.append(replace(validated(), at=TIME + timedelta(days=1)))
    assert later.entry(NAME).status == "validated"
    with pytest.raises(CatalogError):
        later.append(approved())
    other = replace(documented("crsp.daily"), at=TIME - timedelta(days=1))
    assert later.append(other).entry("crsp.daily").status == "documented"


def test_entry_of_an_unregistered_name_is_refused():
    catalog = Catalog().append(documented())
    with pytest.raises(CatalogError):
        catalog.entry("unregistered.dataset")


def test_search_matches_every_term_against_name_field_or_owner_in_name_order():
    registrations = (
        documented(vintages=VINTAGES),
        documented("crsp.daily", owner="Research Data", fields=("permno", "date", "ret")),
        documented("ibes.estimates", fields=("ticker", "fpedats", "meanest")),
    )
    forward = Catalog((*registrations, validated()))
    backward = Catalog((*reversed(registrations), validated()))
    expected = {
        ("", None): ("crsp.daily", "ibes.estimates", NAME),
        ("CRSP", None): ("crsp.daily",),
        ("PermNo", None): ("crsp.daily",),
        ("engineering", None): ("ibes.estimates", NAME),
        ("engineering GVKEY", None): (NAME,),
        ("engineering permno", None): (),
        ("", "validated"): (NAME,),
        ("data", "documented"): ("crsp.daily", "ibes.estimates"),
    }
    for (text, status), result in expected.items():
        assert names(forward.search(text, status=status)) == result
        assert backward.search(text, status=status) == forward.search(text, status=status)
