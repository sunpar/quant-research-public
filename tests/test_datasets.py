from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from importlib import resources
from types import SimpleNamespace

import pytest

from quant_research.access import (
    CONFIRMATION,
    INTERNAL_CODE,
    Provider,
    Restrictions,
    issue_context,
    licensed,
)
from quant_research.catalog import (
    Approved,
    Catalog,
    Documented,
    Evidence,
    Revoked,
    Validated,
    Vintage,
)
from quant_research.datasets import (
    ADAPTER,
    DatasetError,
    Listed,
    ReadRecord,
    Refused,
    Row,
    Supplied,
    contract_of,
    describe,
    listing,
    period_contract,
    publish_contract,
    read,
    save,
)
from quant_research.public_data import PILOT, SEARCH_UNIVERSE
from quant_research.records import canonical_bytes
from quant_research.retrieval import Unit, publish_corpus

GOVERNANCE = datetime(2026, 10, 21, 12, tzinfo=UTC)
CUTOFF = datetime(2015, 7, 1, tzinfo=UTC)
NAME = "vendor.prices"
VENDOR = licensed("synthetic-vendor")
ASPECTS = ("publication", "revision", "arrival")
VINTAGES = tuple(
    Vintage(aspect, Evidence(hashlib.sha256(aspect.encode()).hexdigest(), f"Synthetic {aspect}"))
    for aspect in ASPECTS
)


def lifecycle(
    *steps,
    access=(),
    adapter=ADAPTER,
    name=NAME,
    owner="Data Engineering",
    data=b"",
    approved_for=(),
):
    vintage = hashlib.sha256(data).hexdigest()
    events = {
        "documented": Documented(
            name,
            "researcher",
            GOVERNANCE,
            owner,
            ("close",),
            access,
            vintage,
            (),
            VINTAGES,
            approved_for,
        ),
        "validated": Validated(name, "reviewer", GOVERNANCE, VINTAGES[0].evidence),
        "approved": Approved(name, "lead", GOVERNANCE, adapter),
        "revoked": Revoked(name, "lead", GOVERNANCE, "Adapter leaked restatements"),
    }
    return Catalog(tuple(events[step] for step in steps))


def approved(**options):
    return lifecycle("documented", "validated", "approved", **options)


def run(*clearances, cutoff=CUTOFF, scope=None):
    return issue_context(
        clearances=clearances,
        cutoff=cutoff,
        generation="synthetic/v1",
        providers=(),
        scope=scope,
    )


def at(day: int, hour: int = 0) -> str:
    return f"2015-06-{day:02d}T{hour:02d}:00:00Z"


def version(
    key="A", revision=1, published="2015-06-01T20:00:00Z", arrived="2015-06-01T21:00:00Z", **values
):
    return {
        "arrived_at": arrived,
        "key": key,
        "published_at": published,
        "revision": revision,
        "values": values or {"close": "11"},
    }


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def document(rows, *, dataset=NAME, **extra):
    return {"dataset": dataset, "rows": rows, "schema_version": 1, **extra}


@pytest.fixture
def store(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    return root


@pytest.fixture
def publish(store):
    def write(rows=(), *, name=NAME, data=None):
        data = canonical(document(list(rows), dataset=name)) if data is None else data
        (store / name).mkdir(exist_ok=True)
        (store / name / f"{hashlib.sha256(data).hexdigest()}.json").write_bytes(data)
        return data

    return write


def supplied(result) -> Supplied:
    assert isinstance(result, Supplied), result
    return result


def keys(result) -> list[tuple[str, int]]:
    return [(row.key, row.revision) for row in supplied(result).rows]


def test_written_fixture_reads_back_with_its_record(store, publish):
    data = publish([version(close="11", volume="5")])
    catalog = approved(data=data)
    context = run()
    result = supplied(read(catalog, store, NAME, context))
    assert result.rows == (
        Row(
            "A",
            1,
            datetime(2015, 6, 1, 20, tzinfo=UTC),
            datetime(2015, 6, 1, 21, tzinfo=UTC),
            {"close": "11", "volume": "5"},
        ),
    )
    assert result.rows[0].available_at == datetime(2015, 6, 1, 21, tzinfo=UTC)
    assert result.record == ReadRecord(
        NAME,
        hashlib.sha256(data).hexdigest(),
        catalog.digest,
        CUTOFF,
        context.identity,
        (("A", 1),),
    )


def test_a_repeated_read_supplies_the_rows_and_record_a_fresh_parse_gives(store, publish):
    data = publish(REVISIONS)
    catalog = approved(data=data)
    for cutoff in (datetime(2015, 6, 19, tzinfo=UTC), datetime(2015, 6, 30, tzinfo=UTC)):
        first, again = (supplied(read(catalog, store, NAME, run(cutoff=cutoff))) for _ in "12")
        assert again.rows == first.rows == first.record.replay(data)
        assert again.record.to_bytes() == first.record.to_bytes()


@pytest.mark.parametrize(
    "change",
    [
        lambda values: values.__setitem__("close", "12"),
        lambda values: values.update(close="12"),
        lambda values: values.pop("close"),
        lambda values: values.clear(),
        lambda values: values.__delitem__("close"),
    ],
)
def test_a_supplied_rows_values_refuse_change_so_a_later_read_sees_the_file(store, publish, change):
    catalog = approved(data=publish([version(close="11")]))
    [row] = supplied(read(catalog, store, NAME, run())).rows
    with pytest.raises(TypeError):
        change(row.values)
    assert supplied(read(catalog, store, NAME, run())).rows[0].values == {"close": "11"}


def test_every_read_checks_the_stored_bytes_against_the_vintage(store, publish):
    data = publish([version()])
    catalog = approved(data=data)
    supplied(read(catalog, store, NAME, run()))
    (store / NAME / f"{hashlib.sha256(data).hexdigest()}.json").write_bytes(
        canonical(document([version(close="99")]))
    )
    with pytest.raises(DatasetError, match="not its documented vintage"):
        read(catalog, store, NAME, run())


@pytest.mark.parametrize(
    "data",
    [
        canonical(document([version()], labels=[])),
        canonical(document([{**version(), "labels": []}])),
        canonical(document([{k: v for k, v in version().items() if k != "arrived_at"}])),
        canonical({"dataset": NAME, "rows": []}),
        canonical(document([], dataset="vendor.other")),
        canonical({**document([]), "schema_version": 2}),
        canonical({**document([]), "schema_version": True}),
        canonical(document({})),
        canonical(document([version(), version(close="12")])),
        canonical(document([version(revision=1), version(revision=1, arrived=at(2))])),
        canonical(document([version(revision=0)])),
        canonical(document([version(revision=-1)])),
        canonical(document([version(revision=True)])),
        canonical(document([version(revision=1.0)])),
        canonical(document([version(revision="1")])),
        canonical(document([{**version(), "values": []}])),
        canonical(document([version(key=" ")])),
        canonical(document([version(key=7)])),
        canonical(document([version(published="2015-06-01T20:00:00+00:00")])),
        canonical(document([version(arrived="2015-06-01 21:00:00")])),
        canonical(document([version(arrived=None)])),
        canonical(document([])).rstrip(b"\n"),
        json.dumps(document([]), sort_keys=True).encode() + b"\n",
        json.dumps(document([version()]), sort_keys=True, indent=1).encode() + b"\n",
        canonical(document([])).replace(b'"rows":[]', b'"rows":[],"rows":[]'),
        b'{"dataset":"vendor.prices","rows":[],"schema_version":1,"x":NaN}\n',
        b"\xef\xbb\xbf" + canonical(document([])),
        b"\xff",
        b"garbage",
        b"",
    ],
)
def test_malformed_fixtures_raise(store, publish, data):
    publish(data=data)
    with pytest.raises(DatasetError):
        read(approved(data=data), store, NAME, run())


def test_missing_fixture_raises(store):
    with pytest.raises(DatasetError):
        read(approved(), store, NAME, run())


def test_malformed_row_raises_even_when_the_cutoff_excludes_it(store, publish):
    data = publish([version(), version(key="B", revision=0, published=at(29))])
    with pytest.raises(DatasetError):
        read(approved(data=data), store, NAME, run(cutoff=datetime(2015, 6, 2, tzinfo=UTC)))


@pytest.mark.parametrize(
    "row",
    [
        version(key="B", published="2099-SECRET"),
        {**version(key="B"), "secret_field": 1},
        version(key="B", revision="SECRET"),
    ],
)
def test_malformed_fixture_error_quotes_no_fixture_text(store, publish, row):
    data = publish([version(), row])
    with pytest.raises(DatasetError) as raised:
        read(approved(data=data), store, NAME, run(cutoff=datetime(2015, 6, 2, tzinfo=UTC)))
    assert "SECRET" not in str(raised.value).upper()


@pytest.mark.parametrize("content", [None, b"garbage", canonical(document([], dataset="other"))])
@pytest.mark.parametrize(
    ("catalog", "reason", "words"),
    [
        (lifecycle("documented"), "documented", ("documented", "not approved for agent use")),
        (lifecycle("documented", "validated"), "validated", ("validated", "not approved")),
        (
            lifecycle("documented", "validated", "approved", "revoked"),
            "revoked",
            ("revoked", "not approved"),
        ),
        (lifecycle("documented", "validated", "revoked"), "revoked", ("revoked", "not approved")),
        (
            lifecycle("documented", "validated", "revoked", "documented"),
            "documented",
            ("documented", "not approved"),
        ),
        (approved(adapter="other.adapter-v1"), "other_adapter", ("different adapter",)),
        (approved(access=(VENDOR,)), "unknown_dataset", ("No dataset",)),
        (Catalog(), "unknown_dataset", ("No dataset",)),
        (approved(name="vendor.other"), "unknown_dataset", ("No dataset",)),
    ],
)
def test_refusals_come_before_any_row_is_read(store, publish, catalog, reason, words, content):
    if content is not None:
        publish(data=content)
    result = read(catalog, store, NAME, run())
    assert result == Refused(reason)
    assert all(word in result.message for word in words)


@pytest.mark.parametrize("dataset", ["", "../vendor.prices", "Vendor.Prices", "vendor/prices", " "])
def test_dataset_ids_that_are_not_logical_names_are_refused_even_if_cataloged(
    store, publish, dataset
):
    publish(data=b"garbage")
    catalog = approved(name=dataset) if dataset.strip() else Catalog()
    assert read(catalog, store, dataset, run()) == Refused("unknown_dataset")


def test_approval_is_the_catalogs_current_state_and_a_2026_approval_reads_a_2015_cutoff(
    store, publish
):
    data = publish([version()])
    assert approved().entry(NAME).history[-1].at.year == 2026
    assert keys(
        read(approved(data=data), store, NAME, run(cutoff=datetime(2015, 6, 2, tzinfo=UTC)))
    ) == [("A", 1)]
    assert read(
        lifecycle("documented", "validated", "approved", "revoked"), store, NAME, run()
    ) == Refused("revoked")


@pytest.mark.parametrize(
    ("published", "arrived", "cutoff", "visible"),
    [
        (at(1, 20), at(1, 21), at(1, 21), True),
        (at(1, 20), at(1, 21), at(1, 22), True),
        (at(1, 20), at(1, 21), at(1, 20), False),
        (at(1, 20), at(1, 21), at(1, 19), False),
        (at(1, 21), at(1, 20), at(1, 20), False),
        (at(1, 21), at(1, 20), at(1, 21), True),
        (at(1, 20), at(3, 0), at(2, 0), False),
        (at(3, 0), at(1, 20), at(2, 0), False),
    ],
)
def test_a_version_is_read_only_once_both_publication_and_arrival_have_passed(
    store, publish, published, arrived, cutoff, visible
):
    data = publish([version(published=published, arrived=arrived)])
    result = supplied(
        read(approved(data=data), store, NAME, run(cutoff=datetime.fromisoformat(cutoff)))
    )
    assert keys(result) == ([("A", 1)] if visible else [])
    assert result.record.supplied == tuple(keys(result))


REVISIONS = [
    version("B", 2, at(2, 0), at(2, 0), close="b2"),
    version("A", 3, at(20, 0), at(20, 0), close="a3"),
    version("A", 1, at(1, 0), at(1, 0), close="a1"),
    version("B", 1, at(1, 0), at(20, 0), close="b1"),
    version("A", 2, at(5, 0), at(18, 0), close="a2"),
    version("C", 1, at(12, 0), at(12, 0), close="c1"),
]


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (30, [("A", 3), ("B", 2), ("C", 1)]),
        (19, [("A", 2), ("B", 2), ("C", 1)]),
        (15, [("A", 1), ("B", 2), ("C", 1)]),
        (10, [("A", 1), ("B", 2)]),
        (1, [("A", 1)]),
    ],
)
def test_each_key_gets_its_latest_revision_available_by_the_cutoff(store, publish, day, expected):
    data = publish(REVISIONS)
    result = read(
        approved(data=data), store, NAME, run(cutoff=datetime(2015, 6, day, 12, tzinfo=UTC))
    )
    assert keys(result) == expected


def test_a_cutoff_before_every_version_supplies_nothing_without_a_reason(store, publish):
    data = publish(REVISIONS)
    result = supplied(
        read(approved(data=data), store, NAME, run(cutoff=datetime(2015, 5, 1, tzinfo=UTC)))
    )
    assert result.rows == ()
    assert result.record.supplied == ()
    assert result.record.replay(data) == ()


def test_a_narrowed_context_reads_the_earlier_revision(store, publish):
    catalog = approved(data=publish(REVISIONS))
    wide = run(cutoff=datetime(2015, 6, 30, tzinfo=UTC))
    narrow = wide.narrow(cutoff=datetime(2015, 6, 15, tzinfo=UTC))
    assert keys(read(catalog, store, NAME, wide))[0] == ("A", 3)
    result = supplied(read(catalog, store, NAME, narrow))
    assert keys(result)[0] == ("A", 1)
    assert result.record.context == narrow.identity != wide.identity


def test_restricted_dataset_is_refused_for_an_uncleared_context_and_supplied_for_a_cleared_one(
    store, publish
):
    catalog = approved(access=(VENDOR, INTERNAL_CODE), data=publish([version()]))
    for context in (run(), run(VENDOR), run(INTERNAL_CODE), run(licensed("other"), INTERNAL_CODE)):
        result = read(catalog, store, NAME, context)
        assert result == read(Catalog(), store, NAME, context) == Refused("unknown_dataset")
        assert "vendor" not in result.message and "2015" not in result.message
    assert keys(read(catalog, store, NAME, run(VENDOR, INTERNAL_CODE))) == [("A", 1)]
    assert read(approved(access=(CONFIRMATION,)), store, NAME, run()) == Refused("unknown_dataset")


def test_a_dataset_outside_the_scope_is_refused_as_a_missing_name_before_any_file_opens(
    store, publish
):
    data = publish(data=b"garbage")
    for catalog in (approved(data=data), lifecycle("documented", data=data)):
        for scope in ((), ("vendor.other",)):
            result = read(catalog, store, NAME, run(scope=scope))
            assert result == read(Catalog(), store, NAME, run()) == Refused("unknown_dataset")
            assert result.message == "No dataset this run may read has this name"
    with pytest.raises(DatasetError):
        read(approved(data=data), store, NAME, run(scope=(NAME, "vendor.other")))


@pytest.mark.parametrize(
    ("catalog", "reason"),
    [
        (lifecycle("documented", access=(VENDOR,)), "documented"),
        (lifecycle("documented", "validated", access=(VENDOR,)), "validated"),
        (lifecycle("documented", "validated", "approved", "revoked", access=(VENDOR,)), "revoked"),
        (approved(access=(VENDOR,), adapter="other.adapter-v1"), "other_adapter"),
    ],
)
def test_a_status_refusal_reaches_only_a_run_that_may_see_the_dataset(
    store, publish, catalog, reason
):
    publish(data=b"garbage")
    assert read(catalog, store, NAME, run(VENDOR)) == Refused(reason)
    assert read(catalog, store, NAME, run(VENDOR, scope=(NAME,))) == Refused(reason)
    for hidden in (run(), run(VENDOR, scope=()), run(VENDOR, scope=("vendor.other",))):
        assert read(catalog, store, NAME, hidden) == Refused("unknown_dataset")


def test_a_read_carries_the_documented_labels_and_approvals_at_the_cutoff(store, publish):
    catalog = approved(access=(VENDOR,), approved_for=("firm-proxy",), data=publish(REVISIONS))
    expected = Restrictions(
        available_at=CUTOFF,
        labels=frozenset({VENDOR}),
        approved_for=frozenset({Provider("firm-proxy")}),
    )
    assert supplied(read(catalog, store, NAME, run(VENDOR))).restrictions == expected
    early = datetime(2015, 5, 1, tzinfo=UTC)
    empty = supplied(read(catalog, store, NAME, run(VENDOR, cutoff=early)))
    assert empty.rows == ()
    assert empty.restrictions == replace(expected, available_at=early)
    undocumented = supplied(read(approved(data=publish(REVISIONS)), store, NAME, run()))
    assert undocumented.restrictions == Restrictions(available_at=CUTOFF, labels=frozenset())


def test_values_cannot_claim_a_clearance_or_add_a_restriction(store, publish):
    claims = {"access": [], "clearances": [VENDOR], "labels": [], "restrictions": []}
    data = publish([version(**claims)])
    assert read(approved(access=(VENDOR,), data=data), store, NAME, run()) == Refused(
        "unknown_dataset"
    )
    result = supplied(read(approved(data=data), store, NAME, run()))
    assert result.rows[0].values == claims
    data = publish([version(access=[VENDOR], labels=[VENDOR])])
    assert keys(read(approved(data=data), store, NAME, run())) == [("A", 1)]


def test_a_stand_in_context_raises_before_any_gate_or_file(store, publish):
    publish(data=b"garbage")
    stand_in = SimpleNamespace(
        clearances=frozenset({VENDOR}), cutoff=CUTOFF, read_denials=lambda item: ()
    )
    for catalog in (approved(access=(VENDOR,)), lifecycle("documented"), Catalog()):
        with pytest.raises(DatasetError):
            read(catalog, store, NAME, stand_in)


@pytest.mark.smoke
def test_replay_returns_the_supplied_rows_in_supplied_order_from_the_bytes_alone(store, publish):
    data = publish(REVISIONS)
    result = supplied(
        read(approved(data=data), store, NAME, run(cutoff=datetime(2015, 6, 19, tzinfo=UTC)))
    )
    (store / NAME / f"{result.record.vintage}.json").unlink()
    assert result.record.replay(data) == result.rows
    restored = ReadRecord.from_bytes(result.record.to_bytes())
    assert restored.replay(data) == result.rows
    assert [row.key for row in restored.replay(data)] == ["A", "B", "C"]


def test_each_read_opens_the_vintage_its_documentation_names(store, publish):
    old = publish(REVISIONS)
    new = publish([*REVISIONS, version("D", close="new")])
    earlier = supplied(read(approved(data=old), store, NAME, run()))
    later = supplied(read(approved(data=new), store, NAME, run()))
    assert keys(earlier) == [("A", 3), ("B", 2), ("C", 1)]
    assert keys(later) == [*keys(earlier), ("D", 1)]
    assert earlier.record.vintage == hashlib.sha256(old).hexdigest() != later.record.vintage
    assert earlier.record.replay(old) == earlier.rows


def test_bytes_stored_under_another_vintage_raise(store, publish):
    data = publish(REVISIONS)
    (store / NAME / f"{hashlib.sha256(data).hexdigest()}.json").write_bytes(
        canonical(document([version()]))
    )
    with pytest.raises(DatasetError):
        read(approved(data=data), store, NAME, run())


def test_save_keeps_every_vintage_under_its_digest(store):
    first, second = canonical(document(REVISIONS)), canonical(document([version()]))
    vintages = [save(store, NAME, data) for data in (first, second, first)]
    assert vintages == [hashlib.sha256(data).hexdigest() for data in (first, second, first)]
    assert sorted(path.name for path in (store / NAME).iterdir()) == sorted(
        f"{vintage}.json" for vintage in set(vintages)
    )
    assert (store / NAME / f"{vintages[0]}.json").read_bytes() == first


@pytest.mark.parametrize(
    ("dataset", "data"),
    [
        (NAME, b"garbage"),
        (NAME, canonical(document([], dataset="vendor.other"))),
        ("../outside", canonical(document([], dataset="../outside"))),
        ("Vendor.Prices", canonical(document([], dataset="Vendor.Prices"))),
    ],
)
def test_save_refuses_a_malformed_file_or_name_and_stores_nothing(store, dataset, data):
    with pytest.raises(DatasetError):
        save(store, dataset, data)
    assert list(store.rglob("*")) == []


def test_replay_refuses_changed_bytes_and_forged_identities(store, publish):
    data = publish(REVISIONS)
    record = supplied(read(approved(data=data), store, NAME, run())).record
    changed = canonical(document([*REVISIONS, version("D", close="new")]))
    for forged in (
        data.replace(b"a3", b"zz"),
        changed,
        canonical(document(REVISIONS, dataset="vendor.other")),
        b"garbage",
        b"",
    ):
        with pytest.raises(DatasetError):
            record.replay(forged)
    for identity in (("A", 9), ("Z", 1), ("C", 2), ("b", 2)):
        with pytest.raises(DatasetError):
            replace(record, supplied=(*record.supplied, identity)).replay(data)
    with pytest.raises(DatasetError):
        replace(record, vintage="0" * 64).replay(data)
    with pytest.raises(DatasetError):
        replace(record, dataset="vendor.other").replay(data)


def test_replay_follows_recorded_order_whatever_the_file_order(store, publish):
    data = publish(REVISIONS)
    record = supplied(read(approved(data=data), store, NAME, run())).record
    reordered = replace(record, supplied=tuple(reversed(record.supplied)))
    assert [(row.key, row.revision) for row in reordered.replay(data)] == [
        ("C", 1),
        ("B", 2),
        ("A", 3),
    ]


def test_read_validation_checks_admission_and_documented_bytes(store, publish):
    data = publish(REVISIONS)
    catalog, context = approved(data=data), run()
    record = supplied(read(catalog, store, NAME, context)).record
    assert record.validate(data, catalog=catalog, context=context) is None
    with pytest.raises(DatasetError, match="No dataset this run may read"):
        record.validate(b"garbage", catalog=catalog, context=run(scope=()))
    with pytest.raises(DatasetError, match="documented vintage"):
        record.validate(b"garbage", catalog=catalog, context=context)
    with pytest.raises(DatasetError, match="Invalid dataset"):
        record.validate(b"garbage", catalog=approved(data=b"garbage"), context=context)
    reordered = replace(record, supplied=tuple(reversed(record.supplied)))
    assert reordered.replay(data) == tuple(reversed(record.replay(data)))
    with pytest.raises(DatasetError, match="differs from its pinned authority"):
        reordered.validate(data, catalog=catalog, context=context)


def test_record_binds_the_catalog_revision_and_the_authority_of_the_run(store, publish):
    data = publish([version()])
    records = [
        supplied(read(catalog, store, NAME, context)).record
        for catalog, context in (
            (approved(data=data), run()),
            (approved(owner="Other Team", data=data), run()),
            (approved(data=data), run(INTERNAL_CODE)),
            (approved(data=data), run(cutoff=CUTOFF + timedelta(days=1))),
        )
    ]
    assert len({record.to_bytes() for record in records}) == len(records)
    assert records[0].catalog == approved(data=data).digest != records[1].catalog
    assert records[0].context == run().identity != records[2].context
    assert records[0].cutoff == CUTOFF != records[3].cutoff


def test_record_bytes_round_trip(store, publish):
    data = publish(REVISIONS)
    record = supplied(read(approved(data=data), store, NAME, run())).record
    encoded = record.to_bytes()
    assert ReadRecord.from_bytes(encoded) == record
    assert ReadRecord.from_bytes(encoded).to_bytes() == encoded
    assert encoded == canonical(json.loads(encoded))


def record_fields():
    return json.loads(
        ReadRecord(NAME, "a" * 64, "b" * 64, CUTOFF, "c" * 64, (("A", 1), ("B", 2))).to_bytes()
    )


@pytest.mark.parametrize(
    "data",
    [
        canonical({**record_fields(), "extra": 1}),
        canonical({k: v for k, v in record_fields().items() if k != "context"}),
        canonical({**record_fields(), "schema_version": 2}),
        canonical({**record_fields(), "dataset": "Vendor Prices"}),
        canonical({**record_fields(), "vintage": "A" * 64}),
        canonical({**record_fields(), "catalog": "b" * 63}),
        canonical({**record_fields(), "context": None}),
        canonical({**record_fields(), "cutoff": "2015-07-01T00:00:00+00:00"}),
        canonical({**record_fields(), "cutoff": "2015-07-01"}),
        canonical({**record_fields(), "supplied": {}}),
        canonical({**record_fields(), "supplied": [["A", 1]]}),
        canonical({**record_fields(), "supplied": [{"key": "A", "revision": 0}]}),
        canonical({**record_fields(), "supplied": [{"key": "A", "revision": True}]}),
        canonical({**record_fields(), "supplied": [{"key": "", "revision": 1}]}),
        canonical({**record_fields(), "supplied": [{"key": "A", "revision": 1, "x": 1}]}),
        canonical(record_fields()).rstrip(b"\n"),
        json.dumps(record_fields(), sort_keys=True).encode() + b"\n",
        json.dumps(record_fields(), sort_keys=True, indent=1).encode() + b"\n",
        b"[]\n",
        b"garbage",
        b"\xff",
        b"",
    ],
)
def test_noncanonical_or_malformed_record_bytes_raise(data):
    with pytest.raises(DatasetError):
        ReadRecord.from_bytes(data)


def test_listing_names_exactly_the_datasets_read_would_supply(store, publish):
    def catalog_of(*catalogs):
        return Catalog(tuple(event for catalog in catalogs for event in catalog.events))

    stored = {
        name: publish([version()], name=name)
        for name in ("vendor.prices", "vendor.quotes", "vendor.restricted", "vendor.unscoped")
    }
    catalog = catalog_of(
        approved(name="vendor.prices", data=stored["vendor.prices"], approved_for=("firm-proxy",)),
        approved(name="vendor.quotes", owner="Trading Desk", data=stored["vendor.quotes"]),
        approved(name="vendor.restricted", access=(VENDOR,), data=stored["vendor.restricted"]),
        approved(name="vendor.unscoped", data=stored["vendor.unscoped"]),
        lifecycle("documented", name="vendor.documented"),
        lifecycle("documented", "validated", "approved", "revoked", name="vendor.revoked"),
        approved(name="vendor.adapter", adapter="other.adapter-v1"),
    )
    scope = ("vendor.quotes", "vendor.prices", "vendor.documented", "zq.missing")
    for context, names in (
        (run(), ["vendor.prices", "vendor.quotes", "vendor.unscoped"]),
        (run(VENDOR), ["vendor.prices", "vendor.quotes", "vendor.restricted", "vendor.unscoped"]),
        (run(VENDOR, scope=scope), ["vendor.prices", "vendor.quotes"]),
        (run(scope=()), []),
    ):
        result = listing(catalog, context)
        assert result.catalog == catalog.digest
        assert [listed.dataset for listed in result.datasets] == names
        reads = {
            entry.name: read(catalog, store, entry.name, context) for entry in catalog.search()
        }
        assert names == [name for name, outcome in reads.items() if isinstance(outcome, Supplied)]
        for listed in result.datasets:
            owner = "Trading Desk" if listed.dataset == "vendor.quotes" else "Data Engineering"
            assert listed == Listed(
                listed.dataset, owner, ("close",), reads[listed.dataset].restrictions
            )


UNRATE = "alfred.unrate"
STRUCTURE_AT = datetime(1960, 3, 16, tzinfo=UTC)
OCTOBER_AT = datetime(2025, 12, 17, tzinfo=UTC)
CONTRACT_RECEIVED_AT = datetime(2026, 10, 24, 4, 31, 57, tzinfo=UTC)
FIRM, SUBSCRIPTION = Provider("firm-proxy"), Provider("claude-subscription")


@pytest.fixture
def sources(tmp_path):
    root = tmp_path / "sources"
    root.mkdir()
    return root


def files(directory) -> list:
    return sorted(path.relative_to(directory) for path in directory.rglob("*"))


def unrate(**options):
    return approved(name=UNRATE, approved_for=(FIRM, SUBSCRIPTION), **options)


def test_a_contract_publishes_one_occurrence_per_part_under_the_documentation(sources):
    generations = publish_contract(sources, unrate(access=(VENDOR,)), UNRATE)
    declarations = [generation.occurrence.declaration for generation in generations]
    assert [item.source_available_at for item in declarations] == [STRUCTURE_AT, OCTOBER_AT]
    assert [item.received_at for item in declarations] == [
        CONTRACT_RECEIVED_AT,
        datetime(2026, 10, 23, tzinfo=UTC),
    ]
    for item in declarations:
        assert item.restrictions == Restrictions(
            available_at=item.source_available_at,
            labels=frozenset({VENDOR}),
            approved_for=frozenset({FIRM, SUBSCRIPTION}),
        )
        assert item.published_at is None
    assert [contract_of(generation) for generation in generations] == [UNRATE, UNRATE]
    assert "2025-10-01" in generations[1].chunks[0].text
    assert all("2025-10-01" not in chunk.text for chunk in generations[0].chunks)
    written = files(sources)
    assert publish_contract(sources, unrate(access=(VENDOR,)), UNRATE) == generations
    assert files(sources) == written
    relabeled = publish_contract(sources, unrate(), UNRATE)
    assert {generation.occurrence.id for generation in relabeled}.isdisjoint(
        generation.occurrence.id for generation in generations
    )


def test_a_dataset_without_a_contract_publishes_nothing_and_an_uncatalogued_one_raises(sources):
    assert publish_contract(sources, approved(), NAME) == ()
    for catalog, dataset in ((approved(), UNRATE), (unrate(), "Alfred.Unrate")):
        with pytest.raises(DatasetError):
            publish_contract(sources, catalog, dataset)
    assert files(sources) == []


@pytest.mark.parametrize("dataset", sorted({*PILOT, *SEARCH_UNIVERSE}))
def test_every_pilot_dataset_ships_a_canonical_contract_whose_structure_comes_first(
    sources, dataset
):
    path = resources.files("quant_research").joinpath("contracts", f"{dataset}.json")
    data = path.read_bytes()
    assert canonical_bytes(json.loads(data)) == data
    generations = publish_contract(sources, approved(name=dataset), dataset)
    available = [
        generation.occurrence.declaration.source_available_at for generation in generations
    ]
    assert available[0] == min(available)
    assert f"`{dataset}`" in generations[0].chunks[0].text
    assert {contract_of(generation) for generation in generations} == {dataset}
    assert len({generation.occurrence.id for generation in generations}) == len(generations)
    assert period_contract(generations[0]).mapping == "iso-month-start/closed-label/v1"
    assert all(period_contract(generation) is None for generation in generations[1:])
    assert generations[0].occurrence.declaration.received_at == CONTRACT_RECEIVED_AT
    assert all(
        generation.occurrence.declaration.received_at == datetime(2026, 10, 23, tzinfo=UTC)
        for generation in generations[1:]
    )


@pytest.mark.parametrize(
    "change",
    [
        {"period_mapping": None},
        {"period_mapping": "unknown/v1"},
        {"parts": []},
        {"schema_version": 3},
        {"schema_version": True},
    ],
)
def test_contract_publication_rejects_incomplete_or_unknown_period_mappings(
    sources, tmp_path, monkeypatch, change
):
    path = resources.files("quant_research").joinpath("contracts", f"{UNRATE}.json")
    value = {**json.loads(path.read_bytes()), **change}
    directory = tmp_path / "definitions"
    (directory / "contracts").mkdir(parents=True)
    (directory / "contracts" / f"{UNRATE}.json").write_bytes(canonical_bytes(value))
    monkeypatch.setattr(resources, "files", lambda _: directory)
    with pytest.raises(DatasetError, match="Invalid contract"):
        publish_contract(sources, unrate(), UNRATE)
    assert files(sources) == []


@pytest.mark.parametrize(
    "revision",
    (
        "Draft 3, period mapping appendix",
        "Part 0, period mapping",
        "Part 0, period mapping unknown/v1, text SHA-256 {digest}",
    ),
)
def test_period_contract_ignores_ordinary_sources_and_refuses_bad_contracts(sources, revision):
    generation = publish_contract(sources, unrate(), UNRATE)[0]
    declaration = replace(
        generation.occurrence.declaration,
        revision=revision.format(digest=generation.occurrence.blob_digest),
    )
    malformed = replace(
        generation, occurrence=replace(generation.occurrence, declaration=declaration)
    )
    with pytest.raises(DatasetError):
        period_contract(malformed)
    ordinary = replace(declaration, location="https://example.test/paper")
    assert (
        period_contract(
            replace(generation, occurrence=replace(generation.occurrence, declaration=ordinary))
        )
        is None
    )


def test_no_other_contract_ships_beside_the_pilot_datasets():
    shipped = resources.files("quant_research").joinpath("contracts").iterdir()
    assert sorted(path.name for path in shipped) == sorted(
        f"{dataset}.json" for dataset in {*PILOT, *SEARCH_UNIVERSE}
    )


def contract_run(corpus, *clearances, cutoff=OCTOBER_AT, scope=None):
    return issue_context(
        clearances=clearances,
        cutoff=cutoff,
        generation=corpus.id,
        providers=(),
        scope=scope,
    )


def units(generations) -> tuple[Unit, ...]:
    return tuple(
        Unit(generation.occurrence.id, generation.id, chunk.start, chunk.end, chunk.text)
        for generation in generations
        for chunk in generation.chunks
    )


def test_describe_hides_a_contract_conclusion_the_cutoff_does_not_admit(sources):
    catalog = unrate()
    structure, october = publish_contract(sources, catalog, UNRATE)
    corpus = publish_corpus(sources, [october.id, structure.id])
    for cutoff, visible in (
        (OCTOBER_AT - timedelta(microseconds=1), (structure,)),
        (OCTOBER_AT, (structure, october)),
        (STRUCTURE_AT - timedelta(microseconds=1), ()),
    ):
        result = describe(catalog, sources, UNRATE, contract_run(corpus, cutoff=cutoff))
        assert (result.catalog, result.dataset) == (catalog.digest, UNRATE)
        assert result.units == units(visible)


def test_describe_returns_equally_available_parts_in_contract_order(sources):
    dataset = "french.ff3_monthly"
    catalog = approved(name=dataset)
    published = publish_contract(sources, catalog, dataset)
    corpus = publish_corpus(sources, [generation.id for generation in published])
    context = contract_run(corpus, cutoff=datetime(2026, 10, 22, tzinfo=UTC))
    assert describe(catalog, sources, dataset, context).units == units(published)


def test_describe_refuses_what_read_refuses_and_omits_parts_outside_the_corpus(sources):
    structure, october = publish_contract(sources, unrate(access=(VENDOR,)), UNRATE)
    corpus = publish_corpus(sources, [structure.id, october.id])
    for catalog, context, reason in (
        (unrate(access=(VENDOR,)), contract_run(corpus), "unknown_dataset"),
        (unrate(access=(VENDOR,)), contract_run(corpus, VENDOR, scope=(NAME,)), "unknown_dataset"),
        (approved(), contract_run(corpus, VENDOR), "unknown_dataset"),
        (
            lifecycle("documented", name=UNRATE, access=(VENDOR,)),
            contract_run(corpus, VENDOR),
            "documented",
        ),
    ):
        assert describe(catalog, sources, UNRATE, context) == Refused(reason)
    catalog = unrate(access=(VENDOR,))
    for members, visible in (([structure.id], (structure,)), ([], ())):
        result = describe(
            catalog, sources, UNRATE, contract_run(publish_corpus(sources, members), VENDOR)
        )
        assert result.units == units(visible)


def test_a_description_carries_the_reads_restrictions_and_each_returned_parts(sources):
    published = publish_contract(sources, unrate(access=(VENDOR,)), UNRATE)
    corpus = publish_corpus(sources, [generation.id for generation in published])
    context = contract_run(corpus, VENDOR, INTERNAL_CODE)
    catalog = approved(name=UNRATE, access=(INTERNAL_CODE,), approved_for=(FIRM,))
    result = describe(catalog, sources, UNRATE, context)
    assert result.units == units(published)
    assert result.restrictions == Restrictions(
        available_at=OCTOBER_AT,
        labels=frozenset({VENDOR, INTERNAL_CODE}),
        approved_for=frozenset({FIRM}),
    )
