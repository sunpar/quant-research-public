from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import pytest

from quant_research.confirmation import Period, freeze
from quant_research.model_provenance import UNKNOWN, ModelRecord
from quant_research.packages import Outcome
from quant_research.prospective import (
    Entry,
    Observation,
    Procedure,
    ProspectiveError,
    Registration,
    Registry,
    dump_registry,
    load_registry,
)
from quant_research.records import PackageError

REGISTERED = datetime(2026, 11, 2, 9, tzinfo=UTC)
DAY = timedelta(days=1)
INSTANT = timedelta(microseconds=1)


def reference(name):
    return hashlib.sha256(name.encode()).hexdigest()


def canonical(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{text}\n".encode()


REFERENCES = {
    name: reference(f"{name}/v1")
    for name in ("preprocessing", "training", "portfolio_rules", "evaluation")
}
SHORTLIST = freeze((reference("candidate 1"), reference("candidate 2")), **REFERENCES)
MODELS = (
    ModelRecord("provider-a", "model-a", "2026-05-01", date(2026, 1, 31)),
    ModelRecord("provider-b", "model-b", UNKNOWN, UNKNOWN),
)
PROCEDURE = Procedure(SHORTLIST, MODELS)
EMPTY = Procedure(freeze((), **REFERENCES))


def observation(registration, available, *, procedure=None, observed=None, status="null"):
    window = Period("synthetic/v1", available, available + 30 * DAY)
    return Observation(
        registration.registration_id,
        (procedure or registration.procedure).identity,
        window,
        available,
        window.end + DAY if observed is None else observed,
        Outcome(status, f"Supplied {status} outcome"),
    )


def record(*events):
    registry = Registry()
    for event in events:
        registry = registry.append(event)
    return registry


@pytest.mark.smoke
def test_report_lists_each_registration_with_its_observations_in_append_order():
    empty = Registration(EMPTY, REGISTERED)
    full = Registration(PROCEDURE, REGISTERED + DAY)
    later_window = observation(full, REGISTERED + 40 * DAY)
    earlier_window = observation(full, REGISTERED + 2 * DAY, observed=later_window.observed_at)
    registry = record(empty, full, later_window, earlier_window)
    assert empty.registration_id != full.registration_id
    assert registry.report() == (
        Entry(empty.registration_id, empty, ()),
        Entry(full.registration_id, full, (later_window, earlier_window)),
    )


@pytest.mark.parametrize(
    ("available", "available_text"),
    [
        pytest.param(REGISTERED - DAY, "2026-11-01T09:00:00Z", id="earlier"),
        pytest.param(REGISTERED, "2026-11-02T09:00:00Z", id="equal"),
    ],
)
def test_observation_reading_data_available_by_registration_is_refused(available, available_text):
    registration = Registration(PROCEDURE, REGISTERED)
    with pytest.raises(ProspectiveError) as refused:
        record(registration, observation(registration, available))
    assert available_text in str(refused.value)
    assert "2026-11-02T09:00:00Z" in str(refused.value)


def test_observation_reading_data_available_one_microsecond_after_registration_is_accepted():
    registration = Registration(PROCEDURE, REGISTERED)
    event = observation(registration, REGISTERED + INSTANT)
    assert record(registration, event).report() == (
        Entry(registration.registration_id, registration, (event,)),
    )


@pytest.mark.parametrize(
    "changed",
    [
        pytest.param(
            Procedure(
                freeze(
                    SHORTLIST.candidates, **{**REFERENCES, "evaluation": reference("evaluation/v2")}
                ),
                MODELS,
            ),
            id="evaluation",
        ),
        pytest.param(
            Procedure(SHORTLIST, (replace(MODELS[0], version="2026-09-01"), MODELS[1])),
            id="model-version",
        ),
    ],
)
def test_changed_procedure_is_observed_only_under_its_own_registration(changed):
    assert changed.identity != PROCEDURE.identity
    original = Registration(PROCEDURE, REGISTERED)
    revised = Registration(changed, REGISTERED + DAY)
    kept = observation(original, REGISTERED + 2 * DAY)
    registry = record(original, revised, kept)
    for registration, ran in ((original, changed), (revised, PROCEDURE)):
        with pytest.raises(ProspectiveError) as refused:
            registry.append(observation(registration, REGISTERED + 2 * DAY, procedure=ran))
        assert ran.identity in str(refused.value)
        assert registration.procedure.identity in str(refused.value)
    accepted = observation(revised, REGISTERED + 2 * DAY)
    assert registry.append(accepted).report() == (
        Entry(original.registration_id, original, (kept,)),
        Entry(revised.registration_id, revised, (accepted,)),
    )


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda last, at: Registration(EMPTY, at), id="registration"),
        pytest.param(
            lambda last, at: replace(last, observed_at=at, outcome=Outcome("failed", "Retry")),
            id="observation",
        ),
    ],
)
def test_event_dated_before_the_last_event_is_refused(build):
    registration = Registration(PROCEDURE, REGISTERED)
    last = observation(registration, REGISTERED + DAY)
    registry = record(registration, last)
    same_time = build(last, last.observed_at)
    assert registry.append(same_time).events == (registration, last, same_time)
    with pytest.raises(ProspectiveError):
        registry.append(build(last, last.observed_at - INSTANT))


def test_appending_a_recorded_event_leaves_the_log_and_its_bytes_unchanged():
    registration = Registration(PROCEDURE, REGISTERED)
    event = observation(registration, REGISTERED + DAY)
    once = record(registration, event)
    again = once.append(event).append(registration)
    assert again.events == once.events == (registration, event)
    assert dump_registry(again) == dump_registry(once)


@pytest.mark.parametrize("index", [0, 1], ids=["registration", "observation"])
def test_load_refuses_a_stored_log_that_repeats_an_event(index):
    registration = Registration(PROCEDURE, REGISTERED)
    content = dump_registry(record(registration, observation(registration, REGISTERED + DAY)))
    stored = json.loads(content)
    assert canonical(stored) == content
    stored["events"].insert(index, stored["events"][index])
    with pytest.raises(ProspectiveError):
        load_registry(canonical(stored))


@pytest.mark.smoke
def test_dump_and_load_reproduce_the_stored_bytes_and_report():
    empty = Registration(EMPTY, REGISTERED)
    full = Registration(PROCEDURE, REGISTERED + DAY)
    failed = observation(full, REGISTERED + 2 * DAY, status="failed")
    retried = observation(full, REGISTERED + 2 * DAY, status="successful")
    later = observation(full, REGISTERED + 40 * DAY, status="inconclusive")
    registry = record(empty, full, failed, retried, later)
    content = dump_registry(registry)
    loaded = load_registry(content)
    assert dump_registry(loaded) == content
    assert loaded.report() == registry.report()
    assert loaded.report()[1].observations == (failed, retried, later)
    assert json.loads(content)["events"][1]["procedure"]["models"] == [
        {
            "provider": "provider-a",
            "model": "model-a",
            "version": "2026-05-01",
            "knowledge_cutoff": "2026-01-31",
        },
        {"provider": "provider-b", "model": "model-b", "version": None, "knowledge_cutoff": None},
    ]


@pytest.mark.parametrize(
    "rewrite",
    [
        pytest.param(
            lambda content: json.dumps(json.loads(content), indent=2).encode(), id="indent"
        ),
        pytest.param(lambda content: content.rstrip(b"\n"), id="no-final-newline"),
    ],
)
def test_load_refuses_non_canonical_bytes(rewrite):
    registration = Registration(PROCEDURE, REGISTERED)
    content = dump_registry(record(registration, observation(registration, REGISTERED + DAY)))
    with pytest.raises(ProspectiveError):
        load_registry(rewrite(content))


def test_load_refuses_a_stored_shortlist_whose_identity_does_not_match_its_fields():
    content = dump_registry(record(Registration(PROCEDURE, REGISTERED)))
    stored = json.loads(content)
    assert canonical(stored) == content
    stored["events"][0]["procedure"]["shortlist"]["evaluation"] = reference("evaluation/v2")
    with pytest.raises(ProspectiveError):
        load_registry(canonical(stored))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        pytest.param("candidates", SHORTLIST.candidates[:1], id="candidates"),
        pytest.param("candidates", (), id="emptied-candidates"),
        *(pytest.param(name, reference(f"{name}/v2"), id=name) for name in REFERENCES),
        pytest.param("identity", reference("forged identity"), id="identity"),
    ],
)
def test_registration_refuses_a_shortlist_changed_after_freezing(name, value):
    changed = Procedure(replace(SHORTLIST, **{name: value}), MODELS)
    with pytest.raises(ProspectiveError):
        Registry().append(Registration(changed, REGISTERED))


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda registration: observation(Registration(EMPTY, REGISTERED), REGISTERED + DAY),
            id="unknown-registration",
        ),
        pytest.param(
            lambda registration: observation(
                registration, REGISTERED + DAY, observed=REGISTERED + DAY - INSTANT
            ),
            id="observed-before-available",
        ),
        pytest.param(
            lambda registration: Registration(Procedure(SHORTLIST, MODELS[:1] * 2), REGISTERED),
            id="duplicate-models",
        ),
        pytest.param(
            lambda registration: observation(
                registration,
                REGISTERED + DAY,
                procedure=Procedure(
                    replace(SHORTLIST, candidates=SHORTLIST.candidates[:1]), MODELS
                ),
            ),
            id="shortlist-changed-after-registration",
        ),
    ],
)
def test_registry_refuses_an_event_that_breaks_a_registration_rule(build):
    registration = Registration(PROCEDURE, REGISTERED)
    registry = record(registration)
    with pytest.raises(ProspectiveError):
        registry.append(build(registration))


def test_append_refuses_an_event_that_cannot_be_stored():
    registration = Registration(PROCEDURE, REGISTERED)
    event = replace(
        observation(registration, REGISTERED + DAY),
        outcome=Outcome("null", "lone \ud800 surrogate"),
    )
    with pytest.raises(PackageError):
        record(registration, event)


def test_legacy_bytes_and_ids_are_pinned():
    registration = Registration(PROCEDURE, REGISTERED)
    assert PROCEDURE.identity == "b7e75560399fbbd1a7af595f8f11b941f9bf0e349be176a51a211d127a4c387a"
    assert (
        registration.registration_id
        == "2b7ea443241ca3a661a8b5592054264031575c376894951f480b82349f78f934"
    )
    assert hashlib.sha256(dump_registry(Registry((registration,)))).hexdigest() == (
        "7cfbcc088dde23e90bec915a9c5ac4927d07d38601fda1ed904bf1916121b205"
    )


def row(key, published, *, arrived=None, revision=1, value=1):
    from quant_research.datasets import Row, Values

    return Row(key, revision, published, arrived or published, Values(value=value))


def read_evidence(*rows, dataset="monthly", scored=()):
    from dataclasses import asdict

    from quant_research.datasets import ReadRecord
    from quant_research.prospective import ReadEvidence, json_default

    content = canonical(
        json.loads(
            json.dumps(
                {"dataset": dataset, "rows": [asdict(item) for item in rows], "schema_version": 1},
                default=json_default,
            )
        )
    )
    record = ReadRecord(
        dataset,
        hashlib.sha256(content).hexdigest(),
        reference("catalog"),
        max(item.available_at for item in rows),
        reference("context"),
        tuple((item.key, item.revision) for item in rows),
    )
    return ReadEvidence(record, content, scored)


def stored_registration(root):
    from uuid import UUID

    from quant_research.prospective import Completion, Horizon, RowEvidence

    history = read_evidence(row("2026-07-01", REGISTERED - 30 * DAY))
    completion = Completion(
        UUID(int=1),
        reference("revision"),
        reference("configuration"),
        "frozen",
        "Synthetic frozen selection",
        (),
        Horizon(),
    )
    registration = Registration(
        Procedure(SHORTLIST, MODELS, completion),
        REGISTERED,
        reference("package"),
        tuple(
            RowEvidence(history.record.dataset, item)
            for item in history.record.replay(history.vintage)
        ),
        (history.record,),
    )
    (root / "events").mkdir()
    (root / "evidence").mkdir()
    (root / "evidence" / history.record.vintage).write_bytes(history.vintage)
    event = json.loads(dump_registry(Registry((registration,))))["events"][0]
    (root / "events" / "0.json").write_bytes(canonical(event))
    return registration, history


def test_original_evidence_roundtrips_and_reloads_persisted_vintages(tmp_path):
    from quant_research.prospective import load_log, observe

    registration, history = stored_registration(tmp_path)
    scored = read_evidence(
        row("2026-11-01", datetime(2026, 12, 2, tzinfo=UTC)), scored=(("2026-11-01", 1),)
    )
    event = observe(
        tmp_path,
        registration.registration_id,
        procedure=registration.procedure.identity,
        window=Period(
            "monthly", datetime(2026, 11, 1, tzinfo=UTC), datetime(2026, 11, 1, tzinfo=UTC)
        ),
        reads=(history, scored),
        outcome=Outcome("null", "Supplied fixture outcome"),
        observed_at=datetime(2026, 12, 3, tzinfo=UTC),
    )
    assert event.inputs == history.rows()
    assert event.available_at == scored.rows()[0].row.available_at
    assert load_log(tmp_path).report()[0].observations == (event,)
    content = dump_registry(load_log(tmp_path))
    assert dump_registry(load_registry(content)) == content
    assert (
        observe(
            tmp_path,
            registration.registration_id,
            procedure=registration.procedure.identity,
            window=event.window,
            reads=(history, scored),
            outcome=event.outcome,
            observed_at=event.observed_at,
        )
        == event
    )
    assert len(load_log(tmp_path).events) == 2
    (tmp_path / "evidence" / scored.record.vintage).write_bytes(b"changed original vintage")
    with pytest.raises(ValueError, match="recorded vintage"):
        load_log(tmp_path)


@pytest.mark.parametrize(
    "case",
    [
        "mixed",
        "old_publication",
        "restamped",
        "changed_value",
        "unregistered_old_input",
        "in_window_input",
        "immature",
        "before_release",
        "wrong_dataset",
        "empty",
        "wrong_bytes",
        "monthly_window",
    ],
)
def test_ineligible_evidence_never_appends_an_event(tmp_path, case):
    from quant_research.prospective import observe

    registration, history = stored_registration(tmp_path)
    new = row("2026-11-01", datetime(2026, 12, 2, tzinfo=UTC))
    scored = read_evidence(new, scored=((new.key, new.revision),))
    reads = (history, scored)
    window = Period("monthly", datetime(2026, 11, 1, tzinfo=UTC), datetime(2026, 11, 1, tzinfo=UTC))
    at = datetime(2026, 12, 3, tzinfo=UTC)
    if case == "mixed":
        old = row("2026-10-01", REGISTERED - DAY)
        scored = read_evidence(old, new, scored=((old.key, 1), (new.key, 1)))
        reads = (history, scored)
        window = replace(window, start=datetime(2026, 10, 1, tzinfo=UTC))
    elif case == "old_publication":
        reads = (
            read_evidence(replace(new, published_at=REGISTERED - DAY), scored=((new.key, 1),)),
        )
    elif case in ("restamped", "changed_value"):
        old = history.rows()[0].row
        changed = (
            replace(old, published_at=REGISTERED + DAY, arrived_at=REGISTERED + DAY)
            if case == "restamped"
            else row(old.key, old.published_at, value=99)
        )
        reads = (read_evidence(changed), scored)
    elif case == "unregistered_old_input":
        reads = (read_evidence(row("2026-06-01", REGISTERED - DAY), dataset="feature"), scored)
    elif case == "in_window_input":
        reads = (read_evidence(row("2026-11-01", REGISTERED - DAY, revision=2)), scored)
    elif case == "immature":
        reads = (
            read_evidence(
                replace(new, published_at=REGISTERED + DAY, arrived_at=REGISTERED + DAY),
                scored=((new.key, 1),),
            ),
        )
        at = datetime(2026, 11, 30, 23, 59, tzinfo=UTC)
    elif case == "before_release":
        at = new.available_at - INSTANT
    elif case == "wrong_dataset":
        window = replace(window, dataset_id="other")
    elif case == "empty":
        reads = (history,)
    elif case == "wrong_bytes":
        reads = (replace(scored, vintage=history.vintage),)
    elif case == "monthly_window":
        window = replace(window, start=window.start + DAY, end=window.end + DAY)
    before = (tmp_path / "events" / "0.json").read_bytes()
    with pytest.raises(ValueError):
        observe(
            tmp_path,
            registration.registration_id,
            procedure=registration.procedure.identity,
            window=window,
            reads=reads,
            outcome=Outcome("null", "Fixture"),
            observed_at=at,
        )
    assert list((tmp_path / "events").iterdir()) == [tmp_path / "events" / "0.json"]
    assert (tmp_path / "events" / "0.json").read_bytes() == before


def test_newly_published_old_calendar_outcome_and_future_feature_inputs_qualify(tmp_path):
    from quant_research.prospective import load_log, observe

    registration, history = stored_registration(tmp_path)
    old_month = row("2026-09-01", REGISTERED + DAY)
    released = read_evidence(old_month, scored=((old_month.key, 1),))
    feature = read_evidence(row("2026-06-01", REGISTERED + 2 * DAY), dataset="feature")
    window = Period("monthly", datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 9, 1, tzinfo=UTC))
    first = observe(
        tmp_path,
        registration.registration_id,
        procedure=registration.procedure.identity,
        window=window,
        reads=(history, released, feature),
        outcome=Outcome("null", "Older month first published later"),
        observed_at=REGISTERED + 3 * DAY,
    )
    future = row("2026-11-01", datetime(2026, 12, 2, tzinfo=UTC))
    second = observe(
        tmp_path,
        registration.registration_id,
        procedure=registration.procedure.identity,
        window=Period(
            "monthly", datetime(2026, 11, 1, tzinfo=UTC), datetime(2026, 11, 1, tzinfo=UTC)
        ),
        reads=(replace(released, scored=()), read_evidence(future, scored=((future.key, 1),))),
        outcome=Outcome("null", "Earlier accepted outcome as rolling history"),
        observed_at=datetime(2026, 12, 3, tzinfo=UTC),
    )
    assert load_log(tmp_path).report()[0].observations == (first, second)


def test_new_registration_cannot_bypass_evidence_in_the_reducer(tmp_path):
    registration, _ = stored_registration(tmp_path)
    with pytest.raises(ProspectiveError, match="requires original read evidence"):
        Registry((registration,)).append(observation(registration, REGISTERED + DAY))
    with pytest.raises(ValueError):
        Registry((replace(registration, reads=None),))
    legacy = Registration(PROCEDURE, REGISTERED - DAY)
    registry = Registry((legacy, registration))
    assert load_registry(dump_registry(registry)).events == registry.events
    malformed = json.loads(dump_registry(registry))
    malformed["schema_version"] = 1
    with pytest.raises(ProspectiveError, match="event shapes"):
        load_registry(canonical(malformed))


@pytest.fixture
def terminal_package(tmp_path):
    from dataclasses import asdict
    from uuid import uuid4

    from quant_research import reference as pilot_reference
    from quant_research.campaigns import (
        CampaignSpecification,
        EvaluationRules,
        ModelRole,
        ResourceLimits,
        file_bytes,
        start_campaign,
    )
    from quant_research.catalog import Catalog
    from quant_research.packages import (
        ArtifactInput,
        EnvironmentRecord,
        ExecutionAttempt,
        LineageLink,
        PackageRecord,
        Specification,
        TrialRecord,
        save_package,
    )
    from quant_research.prospective import json_default

    def experiment(start, end, cutoff):
        return {
            "schema_version": 1,
            "window": {"start": start, "end": end},
            "cutoff": cutoff,
            "baseline": "buy_and_hold_market",
            "evaluation": {
                "version": "pilot-evaluation/v1",
                "alpha": "0.05",
                "minimum_effect": "0.001",
                "subperiods": 2,
            },
        }

    rules = pilot_reference.load_rules(
        canonical(
            {
                "schema_version": 1,
                "search": experiment("2017-01-01", "2019-12-01", "2020-01-01T00:00:00Z"),
                "held_back": experiment("2020-01-01", "2022-12-01", "2023-01-01T00:00:00Z"),
            }
        )
    )
    revision = start_campaign(
        CampaignSpecification(
            "A synthetic question",
            (),
            (),
            EvaluationRules("pilot-evaluation/v1", rules.digest),
            ResourceLimits(1, 1, 1),
            model_roles=(ModelRole("researcher", "provider-a/model-a"),),
        ),
        author="fixture",
        at=REGISTERED - DAY,
        reason="Fixture",
    )
    artifacts = tuple(
        ArtifactInput(name, name.encode(), "text/plain", name)
        for name in ("input", "contract", "implementation", "lock")
    )
    digests = {item.name: hashlib.sha256(item.data).hexdigest() for item in artifacts}
    spec = Specification(
        "A candidate",
        "synthetic/v1",
        "pilot-evaluation/v1",
        "fixture/v1",
        "fixture/v1",
        (digests["input"],),
        digests["contract"],
        digests["implementation"],
        EnvironmentRecord("cpython", "3.12.13", "test", "test", "test", digests["lock"]),
    )
    source = TrialRecord(
        uuid4(),
        uuid4(),
        spec,
        (
            ExecutionAttempt(
                uuid4(),
                uuid4(),
                None,
                REGISTERED - DAY,
                REGISTERED - DAY,
                Outcome("null", "Fixture source"),
            ),
        ),
    )
    measured = TrialRecord(
        uuid4(),
        source.lineage_id,
        replace(spec, data_version="held-back/v1"),
        (
            ExecutionAttempt(
                uuid4(),
                uuid4(),
                None,
                REGISTERED - DAY,
                REGISTERED - DAY,
                Outcome("null", "Fixture measurement"),
            ),
        ),
        (
            LineageLink(
                source.trial_id, source.specification.spec_id, "adaptation", "Frozen source"
            ),
        ),
    )

    def make(kind="full", *, key="first"):
        selected = (
            [str(source.trial_id)]
            if kind in ("full", "pending", "missing_source", "wrong_binding")
            else []
        )
        if kind == "invalid":
            selected = ["unknown trial"]
        reply = {
            "action": "stop" if kind == "stop" else "conclude",
            "arguments": json.dumps({"trials": selected}),
        }
        assessment = {
            "revision": revision.revision_id,
            "shortlist": json.dumps({"text": json.dumps(reply)}),
            "trials": [
                {
                    "source": str(source.trial_id),
                    "trial": str(measured.trial_id),
                    "status": "pending" if kind == "pending" else "null",
                }
            ]
            if selected and kind != "invalid"
            else [],
        }
        report = {
            "revision": revision.revision_id,
            "steps": [{"trial": str(source.trial_id)}],
            "assessment": None if kind == "unfrozen" else assessment,
        }
        items = [
            *artifacts,
            ArtifactInput(
                "campaign-snapshot",
                canonical({"key": key, "report": report}),
                "application/json",
                "campaign-snapshot",
            ),
            ArtifactInput(
                f"campaign-revision-{revision.revision_id}",
                file_bytes(revision),
                "application/json",
                "campaign-revision",
            ),
        ]
        if kind != "no_authority":
            prefix = f"campaign-{revision.revision_id}"
            pinned = {
                "rules": rules.data,
                "models": json.dumps(
                    {"assignments": [{"primary": asdict(MODELS[1]), "fallbacks": []}]},
                    default=json_default,
                ).encode(),
                "context": canonical(
                    {
                        "clearances": [],
                        "cutoff": "2022-01-01T00:00:00Z",
                        "generation": "fixture",
                        "providers": [],
                    }
                ),
                "catalog": Catalog(()).to_bytes(),
            }
            items.extend(
                ArtifactInput(f"{prefix}-{name}", data, "application/json", f"campaign-{name}")
                for name, data in pinned.items()
            )
        trials = (
            ()
            if kind == "missing_source"
            else (source,)
            if kind == "pending"
            else (source, measured)
        )
        if kind == "wrong_binding":
            trials = (
                source,
                replace(
                    measured,
                    specification=replace(measured.specification, signal_profile="other/v1"),
                ),
            )
        return save_package(tmp_path / "packages", PackageRecord(None, trials), tuple(items))

    return make, source


@pytest.mark.parametrize(
    ("kind", "disposition"),
    [
        ("full", "frozen"),
        ("empty", "frozen"),
        ("stop", "unfrozen"),
        ("unfrozen", "unfrozen"),
        ("invalid", "invalid"),
        ("no_authority", "incomplete"),
        ("pending", "frozen"),
        ("missing_source", "incomplete"),
        ("wrong_binding", "invalid"),
    ],
)
def test_terminal_packages_preserve_selection_and_known_metadata(
    tmp_path, terminal_package, kind, disposition
):
    from quant_research.prospective import Horizon, load_log, register_package

    make, source = terminal_package
    registration = register_package(tmp_path / "prospective", make(kind))
    procedure = registration.procedure
    assert procedure.completion.selection == disposition
    assert procedure.completion.horizon == Horizon(1)
    assert procedure.completion.configured_models == (("researcher", "provider-a/model-a", ()),)
    assert procedure.models == (() if kind == "no_authority" else (MODELS[1],))
    if disposition == "frozen":
        assert procedure.shortlist.candidates == (
            () if kind == "empty" else (source.specification.spec_id,)
        )
    else:
        assert procedure.shortlist is None
    assert load_log(tmp_path / "prospective").report()[0].registration == registration


def test_semantic_registration_reuses_first_publication_across_export_keys_and_races(
    tmp_path, terminal_package, monkeypatch
):
    from quant_research import records
    from quant_research.prospective import load_log, register_package

    make, _ = terminal_package
    first, second = make(key="first"), make(key="second")
    root = tmp_path / "prospective"
    write_once = records.write_once
    winner = []

    def compete(path, content):
        if path == root / "events" / "0.json" and not winner:
            winner.append(None)
            winner[0] = register_package(root, second)
        return write_once(path, content)

    monkeypatch.setattr(records, "write_once", compete)
    registration = register_package(root, first)
    assert registration == winner[0]
    assert registration.package == second.digest
    assert register_package(root, first) == registration
    assert len(load_log(root).events) == 1


def test_registration_freezes_only_available_rows_from_a_pinned_future_vintage(
    tmp_path, terminal_package, monkeypatch
):
    from quant_research import prospective

    make, _ = terminal_package
    package = make("empty")
    extract = prospective._package_procedure
    now = datetime.now(UTC)
    old = row("2025-01-01", now - DAY)
    future_month = date(now.year + 1, 1, 1)
    released = datetime(future_month.year, 2, 2, tzinfo=UTC)
    future = row(future_month.isoformat(), released)
    evidence = read_evidence(old, future)

    def pinned(package):
        procedure, _ = extract(package)
        return procedure, (evidence,)

    monkeypatch.setattr(prospective, "_package_procedure", pinned)
    root = tmp_path / "prospective"
    registration = prospective.register_package(root, package)
    assert registration.initialization == (prospective.RowEvidence("monthly", old),)
    assert registration.reads[0].supplied == ((old.key, old.revision),)
    assert (root / "evidence" / evidence.record.vintage).read_bytes() == evidence.vintage
    assert prospective.register_package(root, package) == registration
    window_at = datetime(future_month.year, future_month.month, 1, tzinfo=UTC)
    observation = prospective.observe(
        root,
        registration.registration_id,
        procedure=registration.procedure.identity,
        window=Period("monthly", window_at, window_at),
        reads=(read_evidence(future, scored=((future.key, future.revision),)),),
        outcome=Outcome("null", "Future row was first published after registration"),
        observed_at=released,
    )
    assert prospective.load_log(root).report()[0].observations == (observation,)


def test_top_level_observe_cli_records_and_refuses_evidence_without_append(tmp_path, cli):
    from dataclasses import asdict

    from quant_research.prospective import json_default, load_log

    root = tmp_path / "log"
    root.mkdir()
    registration, history = stored_registration(root)
    new = row("2026-11-01", datetime(2026, 12, 2, tzinfo=UTC))
    scored = read_evidence(new, scored=((new.key, 1),))
    for index, read in enumerate((history, scored)):
        (tmp_path / f"read-{index}.json").write_bytes(read.record.to_bytes())
        (tmp_path / f"vintage-{index}.json").write_bytes(read.vintage)
    evidence = {
        "procedure": registration.procedure.identity,
        "window": {
            "dataset_id": "monthly",
            "start": "2026-11-01T00:00:00Z",
            "end": "2026-11-01T00:00:00Z",
        },
        "reads": [
            {
                "record": f"read-{index}.json",
                "vintage": f"vintage-{index}.json",
                "scored": [{"key": key, "revision": revision} for key, revision in read.scored],
            }
            for index, read in enumerate((history, scored))
        ],
    }
    (tmp_path / "evidence.json").write_bytes(canonical(evidence))
    (tmp_path / "outcome.json").write_bytes(
        canonical(asdict(Outcome("null", "Caller-supplied fixture")))
    )
    arguments = (
        "observe",
        registration.registration_id,
        "--root",
        str(root),
        "--evidence",
        str(tmp_path / "evidence.json"),
        "--outcome",
        str(tmp_path / "outcome.json"),
        "--observed-at",
        "2026-12-03T00:00:00Z",
    )
    result = cli(tmp_path, *arguments)
    assert json.loads(result.stdout)["outcome"]["status"] == "null"
    assert len(load_log(root).events) == 2
    for value in (reference("wrong procedure"), None):
        if value is None:
            evidence["reads"][0]["record"] = None
        else:
            evidence["procedure"] = value
        (tmp_path / "evidence.json").write_bytes(canonical(evidence))
        refusal = cli(tmp_path, *arguments, success=False)
        assert "Prospective command failed" in refusal.stderr
        assert "Traceback" not in refusal.stderr
        assert len(load_log(root).events) == 2
    report = cli(tmp_path, "prospective", "report", "--root", str(root))
    assert json.loads(report.stdout)[0]["registration_id"] == registration.registration_id
    assert json_default(REGISTERED) == "2026-11-02T09:00:00Z"
