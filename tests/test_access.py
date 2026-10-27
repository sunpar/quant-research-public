from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta, timezone
from itertools import permutations
from zoneinfo import ZoneInfo

import pytest

from quant_research.access import (
    CONFIRMATION,
    INTERNAL_CODE,
    NO_MEMORY,
    PERIOD_MAPPING,
    AccessError,
    DatasetPeriods,
    Interval,
    MemoryPin,
    Provider,
    Restrictions,
    RunContext,
    derive,
    issue_context,
    licensed,
)
from quant_research.records import canonical_bytes

CUTOFF = datetime(2026, 10, 21, 12, tzinfo=UTC)
EARLIER = CUTOFF - timedelta(days=1)
LATER = CUTOFF + timedelta(days=1)
LOCAL = Provider("local-model")
HOSTED = Provider("hosted-model")
VENDOR = licensed("synthetic-vendor")
PIN = MemoryPin("a" * 64)


def context(
    *,
    clearances=(INTERNAL_CODE,),
    cutoff=CUTOFF,
    generation="synthetic/v1",
    providers=(LOCAL,),
    memory=NO_MEMORY,
    scope=None,
    dataset_periods=None,
):
    return issue_context(
        clearances=clearances,
        cutoff=cutoff,
        generation=generation,
        providers=providers,
        memory=memory,
        scope=scope,
        dataset_periods=dataset_periods,
    )


def material(*labels, available_at=EARLIER, approved_for=(LOCAL,)):
    return Restrictions(
        available_at=available_at, labels=frozenset(labels), approved_for=frozenset(approved_for)
    )


def test_derive_keeps_the_most_restrictive_value_whatever_the_order_or_grouping():
    latest = material(INTERNAL_CODE, available_at=EARLIER, approved_for=(LOCAL, HOSTED))
    vendor = material(
        VENDOR, available_at=EARLIER - timedelta(days=2), approved_for=(LOCAL, HOSTED)
    )
    local_only = material(available_at=EARLIER - timedelta(days=1), approved_for=(LOCAL,))
    expected = material(INTERNAL_CODE, VENDOR, available_at=EARLIER, approved_for=(LOCAL,))
    for order in permutations((latest, vendor, local_only)):
        assert derive(*order) == expected
    assert derive(derive(latest, vendor), local_only) == expected
    assert derive(latest, derive(vendor, local_only)) == expected
    for item in (latest, vendor, local_only):
        assert derive(item) == derive(item, item) == item


def test_derived_material_inherits_the_restrictions_of_each_input():
    run = context()
    approved = material(INTERNAL_CODE)
    assert run.send_denials(approved, LOCAL) == ()
    unapproved = material(approved_for=())
    assert run.send_denials(derive(approved, unapproved), LOCAL) == ("provider_not_approved",)
    assert run.read_denials(derive(approved, material(VENDOR))) == ("uncleared",)


@pytest.mark.parametrize(
    ("item", "provider", "read", "send"),
    [
        (material(INTERNAL_CODE), LOCAL, (), ()),
        (material(VENDOR), LOCAL, ("uncleared",), ("uncleared",)),
        (material(available_at=LATER), LOCAL, ("after_cutoff",), ("after_cutoff",)),
        (material(approved_for=(LOCAL, HOSTED)), HOSTED, (), ("provider_not_permitted",)),
        (material(approved_for=(HOSTED,)), LOCAL, (), ("provider_not_approved",)),
        (material(), HOSTED, (), ("provider_not_permitted", "provider_not_approved")),
        (
            material(VENDOR, available_at=LATER),
            HOSTED,
            ("uncleared", "after_cutoff"),
            ("uncleared", "after_cutoff", "provider_not_permitted", "provider_not_approved"),
        ),
    ],
    ids=["allowed", "uncleared", "after-cutoff", "not-permitted", "not-approved", "both", "all"],
)
def test_denials_name_every_failed_rule_in_a_fixed_order(item, provider, read, send):
    run = context()
    assert run.read_denials(item) == read
    assert run.send_denials(item, provider) == send


def test_restrictions_keep_their_own_copy_of_labels_and_approvals():
    labels, approved = {INTERNAL_CODE, VENDOR}, {LOCAL}
    item = Restrictions(available_at=EARLIER, labels=labels, approved_for=approved)
    labels.clear()
    approved.add(HOSTED)
    assert context(providers=(LOCAL, HOSTED)).send_denials(item, HOSTED) == (
        "uncleared",
        "provider_not_approved",
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda: Restrictions(available_at=EARLIER, labels=CONFIRMATION),
        lambda: Restrictions(available_at=EARLIER, labels=(), approved_for="local-model-zdr"),
        lambda: context(clearances=CONFIRMATION),
        lambda: context(providers="local-model-zdr"),
        lambda: context(scope="alfred.unrate"),
    ],
    ids=["labels", "approved-for", "clearances", "providers", "scope"],
)
def test_a_string_is_not_a_collection_of_labels_or_providers(build):
    with pytest.raises(AccessError):
        build()


@pytest.mark.parametrize(
    "authority",
    [
        {"generation": 123},
        {"generation": None},
        {"clearances": [1]},
        {"clearances": [licensed("a"), b"x"]},
        {"providers": [None]},
        {"memory": None},
        {"memory": PIN.encode()},
        {"scope": [1]},
        {"scope": ["alfred.unrate", b"alfred.payems"]},
    ],
    ids=[
        "generation-int",
        "generation-none",
        "clearance-int",
        "clearance-bytes",
        "provider-none",
        "memory-none",
        "memory-bytes",
        "scope-int",
        "scope-bytes",
    ],
)
def test_an_issued_context_holds_only_text_authority(authority):
    with pytest.raises(AccessError):
        context(**authority)


@pytest.mark.parametrize(
    ("available_at", "expected"),
    [
        (CUTOFF, ()),
        (CUTOFF.astimezone(timezone(timedelta(hours=5))), ()),
        (CUTOFF + timedelta(microseconds=1), ("after_cutoff",)),
    ],
    ids=["at-cutoff", "same-instant-elsewhere", "one-microsecond-later"],
)
def test_cutoff_is_inclusive_and_compares_instants(available_at, expected):
    assert context().read_denials(material(available_at=available_at)) == expected


def test_the_repeated_daylight_saving_hour_is_ordered_by_instant():
    zone = ZoneInfo("America/New_York")
    first = datetime(2026, 11, 1, 1, 30, tzinfo=zone)
    repeated = datetime(2026, 11, 1, 1, 30, fold=1, tzinfo=zone)
    run = context(cutoff=first)
    assert run.read_denials(material(available_at=repeated)) == ("after_cutoff",)
    with pytest.raises(AccessError):
        run.narrow(cutoff=repeated)
    inputs = (material(available_at=first), material(available_at=repeated))
    assert derive(*inputs) == derive(*reversed(inputs)) == material(available_at=repeated)
    assert derive(*inputs).available_at == repeated.astimezone(UTC)


def test_narrow_moves_only_the_cutoff_earlier():
    original = context()
    item = material(available_at=CUTOFF)
    earlier = CUTOFF - timedelta(microseconds=1)
    narrowed = original.narrow(cutoff=earlier)
    assert narrowed == context(cutoff=earlier)
    assert narrowed.read_denials(item) == ("after_cutoff",)
    assert original.read_denials(item) == ()
    assert original.narrow(cutoff=CUTOFF) == original
    assert context(memory=PIN).narrow(cutoff=earlier).memory == PIN
    scoped = context(scope=("alfred.unrate",)).narrow(cutoff=earlier)
    assert scoped == context(cutoff=earlier, scope=("alfred.unrate",))
    assert (scoped.permits("alfred.unrate"), scoped.permits("alfred.payems")) == (True, False)


@pytest.mark.smoke
def test_no_context_can_read_or_send_confirmation_material():
    widest = context(
        clearances=(INTERNAL_CODE, VENDOR),
        cutoff=datetime(9999, 1, 1, tzinfo=UTC),
        providers=(LOCAL, HOSTED),
    )
    ordinary = material(INTERNAL_CODE, VENDOR, approved_for=(LOCAL, HOSTED))
    confirmation = material(CONFIRMATION, approved_for=(LOCAL, HOSTED))
    assert widest.send_denials(ordinary, HOSTED) == ()
    for item in (confirmation, derive(ordinary, confirmation)):
        assert widest.read_denials(item) == ("uncleared",)
        for provider in (LOCAL, HOSTED):
            assert widest.send_denials(item, provider) == ("uncleared",)
    with pytest.raises(AccessError):
        context(clearances=(CONFIRMATION, INTERNAL_CODE, VENDOR))


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("attempt", "error"),
    [
        (
            lambda run: RunContext(
                clearances=run.clearances,
                cutoff=LATER,
                generation=run.generation,
                providers=run.providers,
            ),
            AccessError,
        ),
        (lambda run: replace(run, cutoff=LATER), AccessError),
        (lambda run: replace(run, providers=run.providers | {HOSTED}), AccessError),
        (lambda run: replace(run, clearances=run.clearances | {VENDOR}), AccessError),
        (lambda run: replace(run, memory=PIN), AccessError),
        (lambda run: replace(run, scope=None), AccessError),
        (lambda run: run.narrow(cutoff=LATER), AccessError),
        (lambda run: setattr(run, "cutoff", LATER), FrozenInstanceError),
    ],
    ids=[
        "construct",
        "replace-cutoff",
        "replace-providers",
        "replace-clearances",
        "replace-memory",
        "replace-scope",
        "narrow",
        "assign",
    ],
)
def test_an_issued_context_cannot_be_widened(attempt, error):
    run = context(scope=("alfred.unrate",))
    with pytest.raises(error):
        attempt(run)
    assert run == context(scope=("alfred.unrate",))


def test_wider_authority_requires_a_new_issued_context():
    original = context()
    wider = context(clearances=(INTERNAL_CODE, VENDOR), cutoff=LATER, providers=(LOCAL, HOSTED))
    item = material(VENDOR, available_at=LATER, approved_for=(HOSTED,))
    assert wider != original
    assert wider.send_denials(item, HOSTED) == ()
    assert original.send_denials(item, HOSTED) == (
        "uncleared",
        "after_cutoff",
        "provider_not_permitted",
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda naive: Restrictions(available_at=naive, labels=()),
        lambda naive: context(cutoff=naive),
        lambda naive: context().narrow(cutoff=naive),
    ],
    ids=["restrictions", "issue-context", "narrow"],
)
def test_naive_times_raise(build):
    with pytest.raises(AccessError):
        build(CUTOFF.replace(tzinfo=None))


def test_identity_is_equal_for_equal_authority_and_differs_when_any_part_differs():
    base = context(clearances=(INTERNAL_CODE, VENDOR), providers=(LOCAL, HOSTED))
    same = context(
        clearances=(VENDOR, INTERNAL_CODE),
        cutoff=CUTOFF.astimezone(ZoneInfo("America/New_York")),
        providers=(HOSTED, LOCAL),
    )
    assert base.identity == same.identity
    authority = {
        "clearances": sorted((INTERNAL_CODE, VENDOR)),
        "cutoff": "2026-10-21T12:00:00Z",
        "generation": "synthetic/v1",
        "providers": sorted((LOCAL, HOSTED)),
    }
    assert base.identity == hashlib.sha256(canonical_bytes(authority)).hexdigest()
    assert base.identity == "6c1fa1f6311a0489dd02b808a1e379f27c816133a9bee2a4a128a445243d0477"
    narrowed = base.narrow(cutoff=EARLIER)
    issued = context(clearances=(INTERNAL_CODE, VENDOR), cutoff=EARLIER, providers=(LOCAL, HOSTED))
    assert narrowed.identity == issued.identity
    other_generation = issue_context(
        clearances=(INTERNAL_CODE, VENDOR),
        cutoff=CUTOFF,
        generation="synthetic/v2",
        providers=(LOCAL, HOSTED),
    )
    different = [
        context(clearances=(INTERNAL_CODE,), providers=(LOCAL, HOSTED)),
        narrowed,
        other_generation,
        context(clearances=(INTERNAL_CODE, VENDOR), providers=(LOCAL,)),
        context(clearances=(INTERNAL_CODE, VENDOR), providers=(LOCAL, HOSTED), memory=PIN),
    ]
    identities = {base.identity, *(item.identity for item in different)}
    assert len(identities) == 6


def test_a_scope_enters_the_identity_and_an_empty_scope_differs_from_none():
    unscoped = context()
    assert unscoped.scope is None
    assert unscoped.identity == context(scope=None).identity
    assert unscoped.identity == ("46b2033864241d952f8de5d1b1f2f5b7d450d16e78988a9e99d7dd765b13b26a")
    scopes = [(), ("alfred.unrate",), ("alfred.unrate", "alfred.payems")]
    identities = {unscoped.identity, *(context(scope=names).identity for names in scopes)}
    assert len(identities) == 4
    assert context(scope=("alfred.payems", "alfred.unrate")).identity == (
        context(scope=["alfred.unrate", "alfred.payems", "alfred.unrate"]).identity
    )
    authority = {
        "clearances": [INTERNAL_CODE],
        "cutoff": "2026-10-21T12:00:00Z",
        "generation": "synthetic/v1",
        "providers": [LOCAL],
        "scope": ["alfred.payems", "alfred.unrate"],
    }
    assert context(scope=("alfred.unrate", "alfred.payems")).identity == (
        hashlib.sha256(canonical_bytes(authority)).hexdigest()
    )


def test_permits_reads_the_scope_and_an_unscoped_run_permits_every_name():
    names = ("alfred.unrate", "alfred.payems", "vendor.prices")
    assert all(context().permits(name) for name in names)
    scoped = context(scope=("alfred.unrate", "vendor.prices"))
    assert [scoped.permits(name) for name in names] == [True, False, True]
    assert not any(context(scope=()).permits(name) for name in names)


def test_period_authority_normalizes_original_intervals_and_survives_narrowing_and_archive():
    interval = Interval(EARLIER, CUTOFF)
    other = Interval(CUTOFF, LATER)
    periods = DatasetPeriods("alfred.unrate", "a" * 64, "b" * 64, PERIOD_MAPPING, (other, interval))
    issued = context(scope=(periods.dataset,), dataset_periods=(periods,))
    normalized = replace(periods, intervals=(interval, other))
    assert issued.dataset_periods == (normalized,)
    same = context(scope=(periods.dataset,), dataset_periods=(normalized,))
    assert same.identity == issued.identity
    assert RunContext.from_bytes(issued.to_bytes()) == issued
    narrowed = issued.narrow(cutoff=EARLIER)
    assert narrowed.dataset_periods == issued.dataset_periods
    assert RunContext.from_bytes(narrowed.to_bytes()) == narrowed
    local = timezone(timedelta(hours=-4))
    shifted = replace(
        periods,
        intervals=tuple(
            Interval(item.start.astimezone(local), item.end.astimezone(local))
            for item in periods.intervals
        ),
    )
    assert context(scope=(periods.dataset,), dataset_periods=(shifted,)).identity == issued.identity
    changed = replace(periods, intervals=(Interval(EARLIER, CUTOFF + timedelta(seconds=1)), other))
    assert context(scope=(periods.dataset,), dataset_periods=(changed,)).identity != issued.identity
    assert (
        context(
            scope=(periods.dataset,), dataset_periods=(replace(periods, generation="c" * 64),)
        ).identity
        != issued.identity
    )
    assert RunContext.from_bytes(context().to_bytes()) == context()
    assert "dataset_periods" not in json.loads(context().to_bytes())


@pytest.mark.parametrize(
    "change",
    [
        {"dataset": "other"},
        {"dataset": "bad/name"},
        {"mapping": "unknown"},
        {"manifest": "bad"},
        {"generation": "bad"},
        {"intervals": ()},
        {"intervals": (Interval(LATER, EARLIER),)},
        {"intervals": (Interval(CUTOFF.replace(tzinfo=None), LATER),)},
    ],
)
def test_invalid_period_authority_fails_issuance(change):
    periods = DatasetPeriods(
        "alfred.unrate", "a" * 64, "b" * 64, PERIOD_MAPPING, (Interval(EARLIER, CUTOFF),)
    )
    with pytest.raises(AccessError):
        context(scope=(periods.dataset,), dataset_periods=(replace(periods, **change),))


def test_period_authority_requires_one_binding_per_scoped_dataset():
    periods = DatasetPeriods(
        "alfred.unrate", "a" * 64, "b" * 64, PERIOD_MAPPING, (Interval(EARLIER, CUTOFF),)
    )
    for scope, bindings in [
        (None, (periods,)),
        ((periods.dataset,), ()),
        ((periods.dataset,), (periods, periods)),
    ]:
        with pytest.raises(AccessError):
            context(scope=scope, dataset_periods=bindings)
    value = json.loads(context(scope=(periods.dataset,), dataset_periods=(periods,)).to_bytes())
    value["dataset_periods"][0]["mapping"] = "unknown/v2"
    with pytest.raises(AccessError, match="archived"):
        RunContext.from_bytes(canonical_bytes(value))
