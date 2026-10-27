from __future__ import annotations

import hashlib
import itertools
import json
import re
import shutil
import threading
import traceback
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from quant_research.access import (
    CONFIRMATION,
    INTERNAL_CODE,
    AccessError,
    Restrictions,
    issue_context,
    licensed,
)
from quant_research.confirmation import Period, eligible_periods, freeze, run_confirmation
from quant_research.confirmation_review import (
    Deposited,
    ForCampaign,
    ForReport,
    Granted,
    Opened,
    OpenRefused,
    Refused,
    ReportStoreError,
    Revoked,
    deposit_report,
    grant_review,
    open_report,
    read_audit,
    revoke_review,
)
from quant_research.findings import (
    External,
    Ledger,
    Lesson,
    Proposal,
    Review,
    ReviewError,
    Scope,
    Supersession,
    record_id,
)
from quant_research.packages import (
    ArtifactInput,
    EnvironmentRecord,
    ExecutionAttempt,
    ExposureEvent,
    Outcome,
    PackageRecord,
    Specification,
    TrialRecord,
    open_package,
    save_package,
)
from quant_research.records import PackageError, canonical_bytes
from quant_research.sources import SourceDeclaration, SourceError, register_source

TIME = datetime(2026, 10, 21, 12, tzinfo=UTC)
LATER = TIME + timedelta(hours=1)
INSTANT = timedelta(microseconds=1)
DAY = timedelta(days=1)
DATASET = "synthetic/v1"
PERIOD = Period(DATASET, datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 6, 30, tzinfo=UTC))
CAMPAIGN = uuid4()
OTHER_CAMPAIGN = uuid4()
SECRET = "Restricted detail"


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
ARTIFACTS = tuple(
    ArtifactInput(name, name.encode(), "text/plain", role)
    for name, role in (
        ("input", "input"),
        ("contract", "data contract"),
        ("code", "implementation"),
        ("lock", "environment"),
    )
)
CANDIDATES = (reference("candidate/1"), reference("candidate/2"))
SHORTLIST = freeze(
    CANDIDATES,
    reference("preprocessing/v1"),
    reference("training/v1"),
    reference("portfolio_rules/v1"),
    reference("evaluation/v1"),
)
DECLARATION = SourceDeclaration(
    media_type="text/plain",
    access_domain="synthetic",
    location="tests/review-note.txt",
    revision="1",
    acquisition="hand-authored test fixture",
    rights="synthetic, unrestricted",
    labels=frozenset(),
    approved_for=frozenset(),
    published_at=None,
    source_available_at=TIME,
    availability_evidence="authored for these tests",
    received_at=TIME,
)
LESSON_SCOPE = Scope("US large-cap equities", "one month", PERIOD.start, PERIOD.end)
BACKTEST = (External("artifact:momentum-backtest", "supports"),)
OUTCOMES = {
    CANDIDATES[0]: Outcome("successful", f"{SECRET}: candidate one held"),
    CANDIDATES[1]: Outcome("rejected", f"{SECRET}: candidate two reversed"),
}


def confirm():
    return run_confirmation(
        SHORTLIST, PERIOD, (), lambda shortlist, period: OUTCOMES, "researcher", TIME
    )


def deposit(store, report, campaign=CAMPAIGN, at=TIME):
    return deposit_report(store, report, campaign=campaign, actor="researcher", at=at)


def grant(store, scope, reviewer="alice", purpose="human_review", expires_at=TIME + DAY, at=TIME):
    return grant_review(
        store,
        scope=scope,
        reviewer=reviewer,
        purpose=purpose,
        expires_at=expires_at,
        actor="lead",
        at=at,
    )


def revoke(store, grant_id, at=TIME):
    revoke_review(store, grant_id, actor="lead", at=at, reason="Review ended")


def read(store, report_id, reviewer="alice", purpose="human_review", at=LATER):
    return open_report(store, report_id, reviewer=reviewer, purpose=purpose, at=at)


def entries(store):
    log = store / "log"
    return {path.name: path.read_bytes() for path in log.iterdir() if not path.name.startswith(".")}


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "review"
    path.mkdir()
    return path


@pytest.fixture
def report():
    return confirm()


@pytest.fixture
def ours(store, report):
    return deposit(store, report)


@pytest.mark.smoke
def test_opened_report_redeposits_to_identical_canonical_bytes(tmp_path, store, report, ours):
    grant(store, ForReport(ours))
    opened = read(store, ours).report
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    assert deposit(fresh, opened) == ours == report.exposure.id
    blob = f"{ours}.json"
    stored = (store / "reports" / blob).read_bytes()
    assert (fresh / "reports" / blob).read_bytes() == stored
    digests = [
        event.digest
        for path in (store, fresh)
        for event in read_audit(path).events
        if isinstance(event, Deposited)
    ]
    assert digests == [hashlib.sha256(stored).hexdigest()] * 2
    files = [*store.rglob("*.json"), *fresh.rglob("*.json")]
    assert len(files) == 2 + len(read_audit(store).events) + len(read_audit(fresh).events)
    for path in files:
        content = path.read_bytes()
        assert canonical_bytes(json.loads(content)) == content


def test_repeat_deposit_records_nothing_and_a_conflicting_one_is_refused(store, report, ours):
    before = read_audit(store).events
    assert deposit(store, report, at=LATER) == ours
    altered = replace(report, results=report.results[:1])
    for conflict in ((report, OTHER_CAMPAIGN), (altered, CAMPAIGN)):
        with pytest.raises(ReportStoreError):
            deposit(store, *conflict, at=LATER)
    assert read_audit(store).events == before
    (deposited,) = before
    assert (deposited.report, deposited.campaign, deposited.actor, deposited.at) == (
        ours,
        CAMPAIGN,
        "researcher",
        TIME,
    )


@pytest.mark.parametrize(
    "changes",
    [{"actor": ""}, {"actor": None}, {"actor": "\ud800"}, {"campaign": "campaign"}],
    ids=["blank-actor", "missing-actor", "unencodable-actor", "malformed-campaign"],
)
def test_refused_deposit_writes_no_report(store, report, changes):
    with pytest.raises(ReportStoreError):
        deposit_report(
            store, report, **{"campaign": CAMPAIGN, "actor": "researcher", "at": TIME, **changes}
        )
    assert entries(store) == {}
    assert not (store / "reports").exists()


def no_grant(store, ours, other):
    return ours


def grant_for_another_report(store, ours, other):
    grant(store, ForReport(other))
    return ours


def grant_to_another_reviewer(store, ours, other):
    grant(store, ForReport(ours), reviewer="bob")
    return ours


def grant_for_another_purpose(store, ours, other):
    grant(store, ForReport(ours), purpose="audit")
    return ours


def grant_expiring_at_the_open(store, ours, other):
    grant(store, ForReport(ours), expires_at=LATER)
    return ours


def revoked_grant(store, ours, other):
    revoke(store, grant(store, ForReport(ours)))
    return ours


def expired_and_revoked_grants(store, ours, other):
    grant_expiring_at_the_open(store, ours, other)
    return revoked_grant(store, ours, other)


def unknown_report(store, ours, other):
    grant(store, ForCampaign(CAMPAIGN))
    return uuid4()


@pytest.mark.parametrize(
    ("setup", "reasons"),
    [
        (no_grant, ("no_grant",)),
        (grant_for_another_report, ("no_grant",)),
        (grant_to_another_reviewer, ("no_grant",)),
        (grant_for_another_purpose, ("wrong_purpose",)),
        (grant_expiring_at_the_open, ("expired",)),
        (revoked_grant, ("revoked",)),
        (expired_and_revoked_grants, ("expired", "revoked")),
        (unknown_report, ("no_grant",)),
    ],
    ids=lambda value: getattr(value, "__name__", None),
)
def test_unauthorized_open_is_refused_after_its_refusal_is_recorded(store, ours, setup, reasons):
    other = deposit(store, confirm())
    target = setup(store, ours, other)
    before = read_audit(store).events
    with pytest.raises(OpenRefused) as refused:
        read(store, target)
    assert refused.value.reasons == reasons
    after = read_audit(store).events
    assert after[:-1] == before
    event = after[-1]
    assert isinstance(event, Refused)
    assert (event.actor, event.at, event.report, event.purpose, event.reasons) == (
        "alice",
        LATER,
        target,
        "human_review",
        reasons,
    )


def test_grant_opens_until_the_instant_before_it_expires(store, report, ours):
    grant(store, ForReport(ours), expires_at=LATER)
    assert read(store, ours, at=LATER - INSTANT).report == report
    with pytest.raises(OpenRefused) as refused:
        read(store, ours, at=LATER)
    assert refused.value.reasons == ("expired",)


def test_campaign_grant_opens_later_deposits_and_no_other_campaigns(store, report, ours):
    campaign_grant = grant(store, ForCampaign(CAMPAIGN))
    later_report = confirm()
    later = deposit(store, later_report, at=LATER)
    elsewhere = deposit(store, confirm(), campaign=OTHER_CAMPAIGN, at=LATER)
    assert read(store, ours).report == report
    assert read(store, later).report == later_report
    with pytest.raises(OpenRefused) as refused:
        read(store, elsewhere)
    assert refused.value.reasons == ("no_grant",)
    opens = [
        (event.report, event.grant)
        for event in read_audit(store).events
        if isinstance(event, Opened)
    ]
    assert opens == [(ours, campaign_grant), (later, campaign_grant)]


@pytest.mark.smoke
def test_authorized_open_returns_the_stored_report_and_records_its_exposure(store, report, ours):
    review = grant(store, ForReport(ours))
    first = read(store, ours)
    second = read(store, ours, purpose="human_review", at=LATER + INSTANT)
    assert first.report == second.report == report
    audit = read_audit(store)
    opened = audit.events[-1]
    assert isinstance(opened, Opened)
    assert (opened.actor, opened.at, opened.report, opened.purpose, opened.grant) == (
        "alice",
        LATER + INSTANT,
        ours,
        "human_review",
        review,
    )
    assert audit.exposures == (first.exposure, second.exposure)
    assert [exposure.id for exposure in audit.exposures] == [
        event.id for event in audit.events if isinstance(event, Opened)
    ]
    exposure = first.exposure
    assert (
        exposure.actor,
        exposure.at,
        exposure.material_digest,
        exposure.dataset_id,
        exposure.interval_start,
        exposure.interval_end,
        exposure.kind,
    ) == ("alice", LATER, SHORTLIST.identity, DATASET, PERIOD.start, PERIOD.end, "result")
    assert SECRET not in exposure.reason
    assert all(SECRET.encode() not in content for content in entries(store).values())


def test_review_exposure_saves_beside_the_shortlist_and_retires_the_period(
    tmp_path, store, report, ours
):
    grant(store, ForReport(ours))
    exposure = read(store, ours).exposure
    trial = TrialRecord(uuid4(), uuid4(), SPEC, ())
    record = PackageRecord(trial.trial_id, (trial,), (report.exposure, exposure))
    frozen = ArtifactInput("shortlist", SHORTLIST.manifest, "application/json", "frozen shortlist")
    saved = save_package(tmp_path / "packages", record, (*ARTIFACTS, frozen))
    assert open_package(saved.path).record.exposures == (report.exposure, exposure)
    assert SECRET.encode() not in (saved.path / "manifest.json").read_bytes()
    assert eligible_periods((exposure,), (PERIOD,)) == ()


def test_grant_changes_are_recorded_once_with_who_and_when(store, ours):
    review = grant(store, ForReport(ours))
    assert grant(store, ForReport(ours), at=LATER) == review
    revoke(store, review, at=LATER)
    revoke(store, review, at=LATER + INSTANT)
    renewed = grant(store, ForReport(ours), at=LATER + INSTANT)
    assert renewed != review
    _, granted, revoked, regranted = read_audit(store).events
    assert isinstance(granted, Granted) and isinstance(regranted, Granted)
    assert (granted.id, granted.actor, granted.at, granted.scope, granted.reviewer) == (
        review,
        "lead",
        TIME,
        ForReport(ours),
        "alice",
    )
    assert (granted.purpose, granted.expires_at) == ("human_review", TIME + DAY)
    assert isinstance(revoked, Revoked)
    assert (revoked.grant, revoked.actor, revoked.at, revoked.reason) == (
        review,
        "lead",
        LATER,
        "Review ended",
    )
    assert regranted.id == renewed


@pytest.mark.parametrize(
    "request_",
    [
        pytest.param(
            lambda store, ours: grant(store, ForReport(ours), at=TIME.replace(tzinfo=None)),
            id="naive-time",
        ),
        pytest.param(
            lambda store, ours: grant(
                store, ForReport(ours), expires_at=LATER.replace(tzinfo=None)
            ),
            id="naive-expiry",
        ),
        pytest.param(
            lambda store, ours: grant(store, ForReport(ours), at=TIME - INSTANT),
            id="before-newest-event",
        ),
        pytest.param(
            lambda store, ours: grant(store, ForReport(ours), reviewer=" "), id="blank-reviewer"
        ),
        pytest.param(
            lambda store, ours: grant(store, ForReport(ours), purpose="curiosity"),
            id="unknown-purpose",
        ),
        pytest.param(
            lambda store, ours: grant(store, ForReport(ours), expires_at=TIME),
            id="expiry-not-after-issue",
        ),
        pytest.param(lambda store, ours: grant(store, ForReport(uuid4())), id="undeposited-report"),
        pytest.param(lambda store, ours: revoke(store, uuid4()), id="unknown-grant"),
        pytest.param(lambda store, ours: read(store, ours, reviewer=" "), id="blank-open-reviewer"),
        pytest.param(
            lambda store, ours: read(store, ours, purpose="curiosity"), id="unknown-open-purpose"
        ),
        pytest.param(
            lambda store, ours: read(store, ours, at=TIME - INSTANT), id="open-before-newest-event"
        ),
        pytest.param(lambda store, ours: deposit(store, "report"), id="not-a-report"),
        pytest.param(
            lambda store, ours: deposit_report(
                store, confirm(), campaign=CAMPAIGN, actor=" ", at=TIME
            ),
            id="blank-depositor",
        ),
    ],
)
def test_invalid_requests_raise_and_record_nothing(store, ours, request_):
    before = entries(store)
    with pytest.raises(ReportStoreError) as raised:
        request_(store, ours)
    assert type(raised.value) is ReportStoreError
    assert entries(store) == before


@pytest.fixture
def history(store, ours):
    review = grant(store, ForReport(ours))
    read(store, ours)
    with pytest.raises(OpenRefused):
        read(store, ours, reviewer="bob")
    revoke(store, review, at=LATER)
    return review


def rewrite(path, change):
    value = json.loads(path.read_bytes())
    change(value)
    path.write_bytes(canonical_bytes(value))


def swap(first, second):
    first_bytes = first.read_bytes()
    first.write_bytes(second.read_bytes())
    second.write_bytes(first_bytes)


def drop_and_rechain(log, index):
    contents = [
        (log / f"{position}.json").read_bytes() for position in range(len(entries(log.parent)))
    ]
    for position in range(len(contents)):
        (log / f"{position}.json").unlink()
    del contents[index]
    previous = None
    for position, content in enumerate(contents):
        data = canonical_bytes({**json.loads(content), "previous": previous})
        (log / f"{position}.json").write_bytes(data)
        previous = hashlib.sha256(data).hexdigest()


def append_forged(store, event):
    log = store / "log"
    names = entries(store)
    previous = hashlib.sha256((log / f"{len(names) - 1}.json").read_bytes()).hexdigest()
    entry = {"schema_version": 1, "previous": previous, "event": event}
    (log / f"{len(names)}.json").write_bytes(canonical_bytes(entry))


@pytest.mark.parametrize(
    "tamper",
    [
        pytest.param(
            lambda log: rewrite(log / "1.json", lambda value: value["event"].update(actor="eve")),
            id="edit-middle",
        ),
        pytest.param(lambda log: (log / "2.json").unlink(), id="delete-middle"),
        pytest.param(lambda log: swap(log / "1.json", log / "2.json"), id="swap"),
        pytest.param(lambda log: (log / "notes.txt").write_text("note"), id="stray-file"),
        pytest.param(
            lambda log: (log / "1.json").write_bytes(
                json.dumps(json.loads((log / "1.json").read_bytes()), indent=1).encode()
            ),
            id="non-canonical",
        ),
    ],
)
def test_tampered_log_raises_on_read_and_open_and_records_nothing(store, ours, history, tamper):
    tamper(store / "log")
    tampered = entries(store)
    for attempt in (lambda: read_audit(store), lambda: read(store, ours)):
        with pytest.raises(ReportStoreError) as raised:
            attempt()
        assert type(raised.value) is ReportStoreError
    assert entries(store) == tampered


def test_rechained_open_without_an_active_grant_fails_verification(store, ours, history):
    stamp = {"actor": "alice", "at": "2026-10-21T14:00:00Z", "purpose": "human_review"}
    refusal = {"kind": "refused", "reasons": ["revoked"], "report": str(ours), **stamp}
    append_forged(store, {**refusal, "id": str(uuid4())})
    assert isinstance(read_audit(store).events[-1], Refused)
    opening = {"kind": "opened", "grant": str(history), "report": str(ours), **stamp}
    append_forged(store, {**opening, "id": str(uuid4())})
    with pytest.raises(ReportStoreError):
        read_audit(store)


@pytest.mark.parametrize(
    "tamper",
    [
        pytest.param(
            lambda log: rewrite(log / "4.json", lambda value: value["event"].update(reason="None")),
            id="edit-newest",
        ),
        pytest.param(lambda log: (log / "4.json").unlink(), id="delete-tail"),
        pytest.param(lambda log: drop_and_rechain(log, 3), id="drop-refusal-and-rechain"),
    ],
)
def test_rewrites_a_plain_read_accepts_are_detected_against_an_earlier_head(store, history, tamper):
    audit = read_audit(store)
    assert isinstance(audit.events[-1], Revoked)
    assert read_audit(store, audit.head).events == audit.events
    tamper(store / "log")
    assert read_audit(store).head != audit.head
    with pytest.raises(ReportStoreError):
        read_audit(store, audit.head)


PINNED_FILES = {
    "log/0.json": "f2de21d7fd8b0f6f6e02f442139fe54d0c4faf0942fba1e0a83a83d8a0a19e67",
    "log/1.json": "971cb85a9d8d4312aaf87b2dd112f010853d1018c3f4fe758a26e724fa180410",
    "log/2.json": "c3fde034bf1a9418199233e27509fe61145077b26fb5cc04c5457db62c824e6e",
    "log/3.json": "4805b5073ea7d07b963d012befea2272c612360f0c55d33bb9f1cb7f5e5e224a",
    "log/4.json": "d6f9216fadec255abcca61de0002347be3d562e735b14c97da1636f2e96130cc",
    "reports/00000000-0000-0000-0000-000000000001.json": (
        "111e898c45365574d7667ee569ef680746afec0cd5c815c55b119c2ca0786fd1"
    ),
}


def on_read_and_write(message):
    return tuple(
        f"ReportStoreError: Cannot {verb} review store <dir>/review: {message}"
        for verb in ("read", "update")
    )


UNNUMBERED = on_read_and_write("Log entries must be numbered from 0.json without gaps")
UNLINKED = on_read_and_write("Log entry 1 does not follow the entry before it")
LOG_REFUSALS = {
    "hidden-file": (None, None),
    "gap": UNNUMBERED,
    "stray-file": UNNUMBERED,
    "stray-then-noncanonical": UNNUMBERED,
    "noncanonical": on_read_and_write("Log entry 1.json is not canonical UTF-8 JSON"),
    "invalid-json": on_read_and_write(
        "Log entry 1.json is not valid JSON: Expecting property name enclosed in double quotes: "
        "line 1 column 2 (char 1)"
    ),
    "broken-link": UNLINKED,
    "broken-link-then-noncanonical": UNLINKED,
    "subdirectory": on_read_and_write("[Errno 21] Is a directory: '<dir>/review/log/5.json'"),
    "missing-log": (None, None),
    "missing-store": (
        None,
        "ReportStoreError: Cannot update review store <dir>/review/missing: "
        "[Errno 2] No such file or directory: '<dir>/review/missing/log'",
    ),
    "read-only": (
        None,
        "ReportStoreError: Cannot update review store <dir>/review: "
        "[Errno 13] Permission denied: '<dir>/review/log/.<temp>'",
    ),
}


@pytest.fixture
def pinned(store, monkeypatch):
    ids = (UUID(int=index) for index in itertools.count(1))
    for module in ("confirmation", "confirmation_review"):
        monkeypatch.setattr(f"quant_research.{module}.uuid4", lambda: next(ids))
    ours = deposit(store, confirm(), campaign=UUID(int=100))
    review = grant(store, ForReport(ours))
    read(store, ours)
    with pytest.raises(OpenRefused):
        read(store, ours, reviewer="bob")
    revoke(store, review, at=LATER)


def make_noncanonical(path):
    path.write_bytes(path.read_bytes().replace(b",", b", ", 1))


def test_review_store_writes_the_pinned_bytes(store, pinned):
    files = sorted(path for path in store.rglob("*") if path.is_file())
    digests = {
        path.relative_to(store).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    assert digests == PINNED_FILES


@pytest.mark.parametrize("case", LOG_REFUSALS)
def test_review_log_refuses_the_pinned_damage(tmp_path, store, pinned, case):
    log = store / "log"
    match case:
        case "hidden-file":
            (log / ".tmp-killed-writer").write_bytes(b"{")
        case "gap":
            (log / "4.json").rename(log / "5.json")
        case "stray-file" | "stray-then-noncanonical":
            (log / "notes.txt").write_text("")
            if case == "stray-then-noncanonical":
                make_noncanonical(log / "1.json")
        case "noncanonical":
            make_noncanonical(log / "1.json")
        case "invalid-json":
            (log / "1.json").write_bytes(b"{")
        case "broken-link" | "broken-link-then-noncanonical":
            rewrite(log / "1.json", lambda value: value.update(previous="0" * 64))
            if case == "broken-link-then-noncanonical":
                make_noncanonical(log / "3.json")
        case "subdirectory":
            (log / "5.json").mkdir()
        case "missing-log":
            shutil.rmtree(log)
        case "missing-store":
            store = store / "missing"
        case "read-only":
            log.chmod(0o500)
    refusals = []
    try:
        for attempt in (read_audit, lambda path: grant(path, ForCampaign(CAMPAIGN), at=LATER)):
            try:
                attempt(store)
            except ReportStoreError as error:
                text = str(error).replace(str(tmp_path), "<dir>")
                text = re.sub(r"/\.\w+'", "/.<temp>'", text)
                refusals.append(f"{type(error).__name__}: {text}")
            else:
                refusals.append(None)
    finally:
        log.chmod(0o700)
    assert tuple(refusals) == LOG_REFUSALS[case]


def test_review_log_starts_empty_and_appends_past_hidden_files(store):
    audit = read_audit(store)
    assert (audit.events, audit.head) == ((), None)
    grant(store, ForCampaign(CAMPAIGN))
    (store / "log" / ".tmp-killed-writer").write_bytes(b"{")
    grant(store, ForCampaign(OTHER_CAMPAIGN))
    assert sorted(entries(store)) == ["0.json", "1.json"]


def nested(depth):
    value = SECRET
    for _ in range(depth):
        value = (value,)
    return value


@pytest.mark.parametrize(
    "outcome",
    [
        Outcome("successful", object()),
        Outcome(SECRET, "Unsupported status"),
        Outcome("successful", "Held", limitations=(nested(5000),)),
    ],
    ids=["non-json-detail", "unknown-status", "deep-limitations"],
)
def test_unstorable_report_is_refused_without_quoting_its_outcomes(store, ours, outcome):
    report = replace(confirm(), results=((CANDIDATES[0], outcome),))
    before = entries(store)
    with pytest.raises(ReportStoreError) as raised:
        deposit(store, report, at=LATER)
    assert SECRET not in "".join(traceback.format_exception(raised.value))
    assert entries(store) == before
    assert not (store / "reports" / f"{report.exposure.id}.json").exists()


def test_edited_report_raises_on_an_authorized_open_and_records_no_open(store, ours):
    grant(store, ForReport(ours))
    rewrite(
        store / "reports" / f"{ours}.json",
        lambda value: value["results"][0][1].update(detail="Edited detail"),
    )
    before = entries(store)
    with pytest.raises(ReportStoreError) as raised:
        read(store, ours)
    assert type(raised.value) is ReportStoreError
    assert entries(store) == before


def test_log_only_appends(store, ours):
    review = grant(store, ForReport(ours))
    earlier, files = read_audit(store), entries(store)
    read(store, ours)
    with pytest.raises(OpenRefused):
        read(store, ours, purpose="audit")
    revoke(store, review, at=LATER)
    later = read_audit(store)
    assert len(later.events) == len(earlier.events) + 3
    assert later.events[: len(earlier.events)] == earlier.events
    assert read_audit(store, earlier.head).events == later.events
    assert {name: entries(store)[name] for name in files} == files


def test_opens_racing_a_revocation_are_refused_once_it_is_recorded(store, ours):
    review = grant(store, ForReport(ours))
    readers, attempts = 8, 3
    start = threading.Barrier(readers + 1)
    results = []

    def reader():
        start.wait()
        for _ in range(attempts):
            try:
                results.append(read(store, ours))
            except OpenRefused as refused:
                results.append(refused)

    def revoker():
        start.wait()
        revoke(store, review, at=LATER)

    threads = [threading.Thread(target=reader) for _ in range(readers)]
    threads.append(threading.Thread(target=revoker))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    decisions = read_audit(store).events[2:]
    assert len(results) == len(decisions) - 1 == readers * attempts
    cut = next(index for index, event in enumerate(decisions) if isinstance(event, Revoked))
    assert all(isinstance(event, Opened) for event in decisions[:cut])
    assert all(
        isinstance(event, Refused) and event.reasons == ("revoked",)
        for event in decisions[cut + 1 :]
    )
    assert {item.exposure.id for item in results if not isinstance(item, OpenRefused)} == {
        event.id for event in decisions[:cut]
    }
    assert sum(isinstance(item, OpenRefused) for item in results) == len(decisions) - cut - 1


@pytest.fixture
def opening(store, ours):
    grant(store, ForReport(ours))
    return read(store, ours)


def test_opened_report_cannot_enter_a_package(tmp_path, opening):
    packages = tmp_path / "packages"
    trial = TrialRecord(uuid4(), uuid4(), SPEC, ())
    record = PackageRecord(trial.trial_id, (trial,))
    with pytest.raises((PackageError, TypeError, AttributeError)):
        save_package(packages, opening.report, ARTIFACTS)
    smuggled = ArtifactInput("confirmation", opening.report, "application/json", "result")
    with pytest.raises(PackageError):
        save_package(packages, record, (*ARTIFACTS, smuggled))
    with pytest.raises(PackageError):
        save_package(packages, replace(record, confirmation=opening.report), ARTIFACTS)
    assert not packages.exists()
    assert save_package(packages, record, ARTIFACTS).record.trial_id == record.trial_id


@pytest.mark.parametrize(
    "field, message",
    [
        ("trial_id", "Invalid package trial_id UUID"),
        ("disclosure_coverage", "Unsupported disclosure coverage"),
        ("status", "Unsupported development outcome status"),
        ("at", "Invalid exposure timestamp"),
        ("material_digest", "Invalid material_digest"),
        ("kind", "Unsupported exposure kind"),
        ("name", "Invalid logical artifact name"),
    ],
)
def test_package_errors_name_the_field_holding_a_report_without_quoting_it(
    tmp_path, report, field, message
):
    attempt = ExecutionAttempt(uuid4(), uuid4(), None, TIME, TIME, Outcome("successful", "Held"))
    trial = TrialRecord(uuid4(), uuid4(), SPEC, (attempt,))
    exposure = ExposureEvent(
        uuid4(), "researcher", TIME, reference("input"), DATASET, TIME, TIME, "data", "Read"
    )
    record = PackageRecord(trial.trial_id, (trial,), (exposure,))
    artifacts = ARTIFACTS
    match field:
        case "trial_id" | "disclosure_coverage":
            record = replace(record, **{field: report})
        case "status":
            attempt = replace(attempt, outcome=Outcome(report, "Held"))
            record = replace(record, trials=(replace(trial, attempts=(attempt,)),))
        case "at" | "material_digest" | "kind":
            record = replace(record, exposures=(replace(exposure, **{field: report}),))
        case "name":
            artifacts = (*ARTIFACTS, ArtifactInput(report, b"report", "text/plain", "result"))
    with pytest.raises(PackageError, match=message) as raised:
        save_package(tmp_path / "packages", record, artifacts)
    assert SECRET not in "".join(traceback.format_exception(raised.value))
    assert not (tmp_path / "packages").exists()


def test_opened_report_cannot_enter_a_source_generation(tmp_path, opening):
    sources = tmp_path / "sources"
    sources.mkdir()
    attempts = [
        (opening.report, DECLARATION),
        *(
            (b"Review note", replace(DECLARATION, **{name: opening.report}))
            for name in ("location", "rights", "availability_evidence")
        ),
    ]
    for data, declaration in attempts:
        with pytest.raises((SourceError, TypeError, AttributeError)):
            register_source(sources, data, declaration)
    assert not any(sources.iterdir())
    occurrence = register_source(sources, b"Review note", DECLARATION)
    assert occurrence.blob_digest == hashlib.sha256(b"Review note").hexdigest()


LESSONS = {
    "statement": lambda value: Lesson(value, "agent", LESSON_SCOPE, BACKTEST),
    "evidence-id": lambda value: Lesson(
        "Momentum is durable", "agent", LESSON_SCOPE, (External(value, "supports"),)
    ),
    "scope-universe": lambda value: Lesson(
        "Momentum is durable", "agent", replace(LESSON_SCOPE, universe=value), BACKTEST
    ),
}


@pytest.mark.parametrize("field", LESSONS)
def test_opened_report_cannot_enter_a_lesson(opening, field):
    build = LESSONS[field]
    accepted = build("Momentum holds in large caps")
    assert Ledger((Proposal(accepted, TIME),)).status(record_id(accepted)) == "proposed"
    for value in (opening.report, " "):
        with pytest.raises(ReviewError):
            Ledger((Proposal(build(value), TIME),))


@pytest.mark.parametrize("field", ["reviewer", "reason"])
def test_opened_report_cannot_enter_a_review_or_supersession(opening, field):
    lesson, successor = (
        LESSONS["statement"](text) for text in ("Momentum is durable", "Momentum persists")
    )
    proposals = (Proposal(lesson, TIME), Proposal(successor, TIME))

    def events(value):
        texts = {"reviewer": "reviewer", "reason": "Checked the evidence", field: value}
        return (
            Review(record_id(lesson), "review", at=TIME, **texts),
            Supersession(record_id(lesson), record_id(successor), at=TIME, **texts),
        )

    for event in events("Checked text"):
        assert Ledger((*proposals, event)).events[-1] == event
    for value in (opening.report, " "):
        for event in events(value):
            with pytest.raises(ReviewError):
                Ledger((*proposals, event))


def test_reviewing_a_report_grants_no_run_context_clearance(store, ours):
    material = Restrictions(available_at=TIME, labels=frozenset({CONFIRMATION}))

    def check():
        with pytest.raises(AccessError):
            issue_context(
                clearances={CONFIRMATION}, cutoff=LATER, generation="review", providers=()
            )
        research = issue_context(
            clearances={INTERNAL_CODE, licensed("vendor")},
            cutoff=LATER,
            generation="alice-research",
            providers=(),
        )
        assert research.read_denials(material) == ("uncleared",)

    check()
    grant(store, ForReport(ours))
    read(store, ours)
    check()
