from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, dataclass, field, replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from quant_research.confirmation import (
    ConfirmationError,
    FrozenShortlist,
    NoConfirmationNeeded,
    Period,
    RestrictedConfirmationReport,
    eligible_periods,
    freeze,
    run_confirmation,
)
from quant_research.packages import (
    ArtifactInput,
    EnvironmentRecord,
    ExposureEvent,
    LineageLink,
    Outcome,
    PackageRecord,
    Specification,
    TrialRecord,
    open_package,
    save_package,
)
from quant_research.records import PackageError

TIME = datetime(2026, 10, 21, 12, tzinfo=UTC)
INSTANT = timedelta(microseconds=1)
DATASET = "synthetic/v1"
CONFIRMATION = Period(DATASET, datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 6, 30, tzinfo=UTC))


def reference(name):
    return hashlib.sha256(name.encode()).hexdigest()


SPEC = Specification(
    "Does the shortlisted signal hold on confirmation data",
    "hand-authored/v1",
    "generic-result/v1",
    DATASET,
    "test-authoring/v1",
    (reference("input"),),
    reference("contract"),
    reference("code"),
    EnvironmentRecord(
        "cpython", "3.12.13", "cpython-312", "test-platform", "uv test", reference("lock")
    ),
)
CANDIDATES = tuple(replace(SPEC, question=f"Candidate signal {n}").spec_id for n in (1, 2, 3))
REFERENCES = {
    name: reference(f"{name}/v1")
    for name in ("preprocessing", "training", "portfolio_rules", "evaluation")
}
OUTCOMES = dict(
    zip(
        CANDIDATES,
        (
            Outcome("successful", "Restricted detail: candidate one held"),
            Outcome("null", "Restricted detail: candidate two showed no effect"),
            Outcome("rejected", "Restricted detail: candidate three reversed"),
        ),
        strict=True,
    )
)


@dataclass
class Recorder:
    outcomes: Mapping[str, Outcome]
    calls: list[tuple[FrozenShortlist, Period]] = field(default_factory=list)

    def __call__(self, shortlist: FrozenShortlist, period: Period) -> Mapping[str, Outcome]:
        self.calls.append((shortlist, period))
        return self.outcomes


def exposure(start, end, kind="data", actor="researcher"):
    return ExposureEvent(
        uuid4(), actor, TIME, reference("input"), DATASET, start, end, kind, "Supplied exposure"
    )


@pytest.fixture
def development():
    artifacts = tuple(
        ArtifactInput(name, name.encode(), "text/plain", role)
        for name, role in (
            ("input", "input"),
            ("contract", "data contract"),
            ("code", "implementation"),
            ("lock", "environment"),
        )
    )
    trial = TrialRecord(uuid4(), uuid4(), SPEC, ())
    seen = exposure(datetime(2025, 1, 1, tzinfo=UTC), CONFIRMATION.start - INSTANT)
    return PackageRecord(trial.trial_id, (trial,), (seen,)), artifacts


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("kind", "actor"),
    [("data", "researcher"), ("result", "research agent"), ("feedback", "reviewer")],
)
def test_every_exposure_kind_and_actor_blocks_each_period_it_touches(kind, actor):
    start, end = datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 6, 30, tzinfo=UTC)
    day = timedelta(days=1)
    before = Period(DATASET, start - day, start - INSTANT)
    after = Period(DATASET, end + INSTANT, end + day)
    elsewhere = Period("other/v1", start, end)
    touched = (
        Period(DATASET, start - day, start),
        Period(DATASET, end, end + day),
        Period(DATASET, end, end),
        Period(DATASET, start + day, end - day),
        Period(DATASET, start - day, end + day),
    )
    history = (exposure(start, end, kind, actor),)
    assert eligible_periods(history, (before, *touched, after, elsewhere)) == (
        before,
        after,
        elsewhere,
    )


def test_new_campaign_or_revised_protocol_cannot_restore_an_exposed_period(tmp_path, development):
    record, artifacts = development
    peek = exposure(
        CONFIRMATION.start + timedelta(days=60), CONFIRMATION.start + timedelta(days=90), "result"
    )
    saved = save_package(tmp_path, replace(record, exposures=(*record.exposures, peek)), artifacts)
    campaign_a = open_package(saved.path).record
    prior = campaign_a.selected_trial
    successor = TrialRecord(
        uuid4(),
        prior.lineage_id,
        replace(SPEC, question="Revised protocol question"),
        (),
        (LineageLink(prior.trial_id, prior.specification.spec_id, "derived_from", "New campaign"),),
    )
    campaign_b = open_package(
        save_package(
            tmp_path, PackageRecord(successor.trial_id, (*campaign_a.trials, successor)), artifacts
        ).path
    ).record
    history = (*campaign_a.exposures, *campaign_b.exposures)
    assert eligible_periods(history, (CONFIRMATION,)) == ()
    revised = freeze(CANDIDATES, **{**REFERENCES, "evaluation": reference("evaluation/v2")})
    assert revised.identity != freeze(CANDIDATES, **REFERENCES).identity
    recorder = Recorder(OUTCOMES)
    with pytest.raises(ConfirmationError, match=str(peek.id)):
        run_confirmation(revised, CONFIRMATION, history, recorder, "researcher", TIME)
    assert recorder.calls == []


def test_freeze_identifies_exactly_the_frozen_content():
    shortlist = freeze(CANDIDATES, **REFERENCES)
    assert shortlist.identity == hashlib.sha256(shortlist.manifest).hexdigest()
    assert json.loads(shortlist.manifest) == {"candidates": list(CANDIDATES), **REFERENCES}
    assert freeze(CANDIDATES, **REFERENCES) == shortlist
    variants = (
        freeze(CANDIDATES[:-1], **REFERENCES),
        freeze(CANDIDATES[::-1], **REFERENCES),
        *(
            freeze(CANDIDATES, **{**REFERENCES, name: reference(f"{name}/v2")})
            for name in REFERENCES
        ),
    )
    identities = {shortlist.identity, *(variant.identity for variant in variants)}
    assert len(identities) == 1 + len(variants)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        pytest.param("candidates", CANDIDATES[:-1], id="candidates"),
        pytest.param("candidates", (), id="emptied-candidates"),
        *(pytest.param(name, reference(f"{name}/v2"), id=name) for name in REFERENCES),
    ],
)
def test_changing_a_frozen_item_is_refused_before_evaluation(name, value):
    shortlist = freeze(CANDIDATES, **REFERENCES)
    with pytest.raises(FrozenInstanceError):
        setattr(shortlist, name, value)
    changed = replace(shortlist, **{name: value})
    recorder = Recorder({candidate: OUTCOMES[candidate] for candidate in changed.candidates})
    with pytest.raises(ConfirmationError):
        run_confirmation(changed, CONFIRMATION, (), recorder, "researcher", TIME)
    assert recorder.calls == []


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: freeze((*CANDIDATES, "candidate"), **REFERENCES), id="candidate"),
        pytest.param(
            lambda: freeze(CANDIDATES, **{**REFERENCES, "training": "training/v1"}),
            id="reference",
        ),
        pytest.param(lambda: freeze((*CANDIDATES, CANDIDATES[0]), **REFERENCES), id="duplicate"),
        pytest.param(lambda: Period(DATASET, TIME, TIME - INSTANT), id="reversed-period"),
        pytest.param(lambda: Period(DATASET, TIME.replace(tzinfo=None), TIME), id="naive-start"),
        pytest.param(lambda: Period(DATASET, TIME, TIME.replace(tzinfo=None)), id="naive-end"),
        pytest.param(lambda: Period(" ", TIME, TIME), id="blank-dataset"),
    ],
)
def test_freeze_and_period_reject_malformed_input(build):
    with pytest.raises(PackageError):
        build()


def test_empty_shortlist_needs_no_confirmation_run_even_on_an_exposed_period():
    shortlist = freeze((), **REFERENCES)
    history = (exposure(CONFIRMATION.start, CONFIRMATION.end),)
    recorder = Recorder({})
    assert eligible_periods(history, (CONFIRMATION,)) == ()
    for period in (CONFIRMATION, None):
        result = run_confirmation(shortlist, period, history, recorder, "researcher", TIME)
        assert result == NoConfirmationNeeded(shortlist.identity)
    assert recorder.calls == []


@pytest.mark.parametrize(
    ("period", "actor", "at", "error"),
    [
        pytest.param(None, "researcher", TIME, ConfirmationError, id="no-period"),
        pytest.param(CONFIRMATION, " ", TIME, PackageError, id="blank-actor"),
        pytest.param(
            CONFIRMATION, "researcher", TIME.replace(tzinfo=None), PackageError, id="naive-at"
        ),
    ],
)
def test_confirmation_refuses_missing_period_or_unrecordable_exposure_before_evaluation(
    period, actor, at, error
):
    recorder = Recorder(OUTCOMES)
    later = exposure(CONFIRMATION.end + INSTANT, CONFIRMATION.end + timedelta(days=1))
    with pytest.raises(error):
        run_confirmation(freeze(CANDIDATES, **REFERENCES), period, (later,), recorder, actor, at)
    assert recorder.calls == []


@pytest.mark.smoke
def test_eligible_frozen_shortlist_yields_a_restricted_report_with_its_exposure():
    shortlist = freeze(CANDIDATES, **REFERENCES)
    recorder = Recorder(dict(reversed(OUTCOMES.items())))
    history = (exposure(CONFIRMATION.start - timedelta(days=30), CONFIRMATION.start - INSTANT),)
    at = TIME + timedelta(days=1)
    report = run_confirmation(shortlist, CONFIRMATION, history, recorder, "reviewer", at)
    assert recorder.calls == [(shortlist, CONFIRMATION)]
    assert isinstance(report, RestrictedConfirmationReport)
    assert (report.shortlist, report.period) == (shortlist.identity, CONFIRMATION)
    assert report.results == tuple((candidate, OUTCOMES[candidate]) for candidate in CANDIDATES)
    event = report.exposure
    assert (event.actor, event.at, event.material_digest, event.kind) == (
        "reviewer",
        at,
        shortlist.identity,
        "result",
    )
    assert (event.dataset_id, event.interval_start, event.interval_end) == (
        CONFIRMATION.dataset_id,
        CONFIRMATION.start,
        CONFIRMATION.end,
    )
    assert "Restricted detail" not in repr(report)


@pytest.mark.parametrize(
    "candidates",
    [
        pytest.param(CANDIDATES[:-1], id="missing"),
        pytest.param((*CANDIDATES, reference("unlisted")), id="extra"),
    ],
)
def test_evaluation_must_return_exactly_one_outcome_per_candidate(candidates):
    recorder = Recorder(
        {candidate: Outcome("null", "Supplied outcome") for candidate in candidates}
    )
    with pytest.raises(ConfirmationError):
        run_confirmation(
            freeze(CANDIDATES, **REFERENCES), CONFIRMATION, (), recorder, "researcher", TIME
        )


def test_confirmation_exposure_makes_its_own_period_ineligible():
    shortlist = freeze(CANDIDATES, **REFERENCES)
    report = run_confirmation(shortlist, CONFIRMATION, (), Recorder(OUTCOMES), "researcher", TIME)
    history = (report.exposure,)
    assert eligible_periods(history, (CONFIRMATION,)) == ()
    recorder = Recorder(OUTCOMES)
    with pytest.raises(ConfirmationError, match=str(report.exposure.id)):
        run_confirmation(shortlist, CONFIRMATION, history, recorder, "researcher", TIME)
    assert recorder.calls == []


def test_ordinary_export_refuses_the_report_but_keeps_its_exposure(tmp_path, development):
    record, artifacts = development
    shortlist = freeze(CANDIDATES, **REFERENCES)
    report = run_confirmation(
        shortlist, CONFIRMATION, record.exposures, Recorder(OUTCOMES), "researcher", TIME
    )
    with pytest.raises((PackageError, TypeError, AttributeError)):
        save_package(tmp_path, report, ())
    smuggled = ArtifactInput("confirmation", report, "application/json", "result")
    with pytest.raises(PackageError):
        save_package(tmp_path, record, (*artifacts, smuggled))
    with pytest.raises(PackageError):
        save_package(tmp_path, replace(record, confirmation=report), artifacts)
    assert not list(tmp_path.iterdir())
    lineage = replace(record, exposures=(*record.exposures, report.exposure))
    frozen = ArtifactInput("shortlist", shortlist.manifest, "application/json", "frozen shortlist")
    saved = save_package(tmp_path, lineage, (*artifacts, frozen))
    assert open_package(saved.path).record.exposures == (*record.exposures, report.exposure)
    assert b"Restricted detail" not in (saved.path / "manifest.json").read_bytes()
