from __future__ import annotations

import hashlib
import itertools
import json
import re
import shutil
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from quant_research import datasets, reference
from quant_research.access import Provider, Restrictions, RunContext, licensed
from quant_research.agent_tools import Generated, Refusal, ToolError
from quant_research.campaigns import (
    CONTRACT,
    PROMPT,
    CampaignError,
    CampaignRevision,
    CampaignSpecification,
    DataPermission,
    EvaluationRules,
    ModelRole,
    ResourceLimits,
    RulesError,
    Setup,
    Source,
    TrialReference,
    contract_sources,
    file_bytes,
    open_campaign,
    parse_file,
    register,
    resolve,
    revise,
    save_revision,
    start_campaign,
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
from quant_research.datasets import DatasetError, Supplied, contract_of
from quant_research.findings import Ledger
from quant_research.model_provenance import MODELS, UNKNOWN, ModelRecord, RoleAssignment
from quant_research.packages import (
    EnvironmentRecord,
    ExposureEvent,
    PackageRecord,
    Specification,
    TrialRecord,
)
from quant_research.providers import Material, ProviderPolicy
from quant_research.records import canonical_bytes
from quant_research.sources import (
    TEXT_PIPELINE,
    SourceDeclaration,
    Step,
    open_generation,
    publish_generation,
    register_source,
)
from quant_research.worker import Limits, RunLog

TIME = datetime(2026, 10, 21, 12, tzinfo=UTC)
START = datetime(2000, 1, 1, tzinfo=UTC)
END = datetime(2019, 12, 31, tzinfo=UTC)
PAPER = hashlib.sha256(b"synthetic paper\n").hexdigest()


def window(start: str, end: str, cutoff: str) -> dict:
    """Return one window of a rules file, in reference.json's schema."""
    return {
        "baseline": "buy_and_hold_market",
        "cutoff": cutoff,
        "evaluation": {
            "alpha": "0.05",
            "minimum_effect": "0.001",
            "subperiods": 2,
            "version": "pilot-evaluation/v1",
        },
        "schema_version": 1,
        "window": {"end": end, "start": start},
    }


# Thirty months each, ending before the setup's cutoff, so a 2025 declaration is after the search.
FROZEN = reference.load_rules(
    canonical_bytes(
        {
            "held_back": window("2020-01-01", "2022-06-01", "2022-07-01T00:00:00Z"),
            "schema_version": 1,
            "search": window("2017-07-01", "2019-12-01", "2020-01-01T00:00:00Z"),
        }
    )
)
RULES = hashlib.sha256(b"pilot evaluation rules\n").hexdigest()
COSTS = hashlib.sha256(b"pilot evaluation rules with costs\n").hexdigest()
PRIOR_TRIAL = TrialReference(uuid4(), uuid4())
PRIOR = ExposureEvent(
    uuid4(),
    "researcher",
    TIME - timedelta(days=30),
    PAPER,
    "synthetic/v1",
    START,
    END,
    "data",
    "Explored the data before the campaign",
)
CHANGED = {
    "question": "Does 12-1 momentum survive transaction costs?",
    "sources": (
        Source("momentum-paper", "https://example.org/momentum.pdf", PAPER, "paper"),
        Source("momentum-code", "https://example.org/momentum.git@v1", RULES, "implementation"),
    ),
    "data": (DataPermission("synthetic/v2", START, END),),
    "evaluation": EvaluationRules("pilot-evaluation/v2", COSTS),
    "limits": ResourceLimits(20, 60, 3600),
    "interventions": ("before_trial",),
    "model_roles": (),
    "generated_code": True,
}


def specification(**changes) -> CampaignSpecification:
    return replace(
        CampaignSpecification(
            "Does 12-1 momentum add value beyond book-to-market in Zürich listings?",
            (Source("momentum-paper", "https://example.org/momentum.pdf", PAPER, "paper"),),
            (DataPermission("synthetic/v1", START, END),),
            EvaluationRules("pilot-evaluation/v1", RULES),
            ResourceLimits(20, 60, 7200),
            ("before_trial", "methodology_change"),
            (ModelRole("critic", "provider/model-x", ("provider/model-y",)),),
        ),
        **changes,
    )


def child(parent: CampaignRevision, spec: CampaignSpecification, **arguments) -> CampaignRevision:
    defaults = {
        "author": "researcher",
        "at": parent.created_at + timedelta(days=1),
        "reason": "Revise the protocol",
    }
    return revise(parent, spec, **(defaults | arguments))


def canonical(value: dict) -> bytes:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (text + "\n").encode()


def write_revision(directory: Path, content: bytes) -> Path:
    path = directory / f"{hashlib.sha256(content).hexdigest()}.json"
    path.write_bytes(content)
    return path


@pytest.fixture
def first():
    return start_campaign(
        specification(),
        author="researcher",
        at=TIME,
        reason="Initial protocol",
        trials=(PRIOR_TRIAL,),
        exposures=(PRIOR,),
    )


@pytest.fixture
def package():
    digest = hashlib.sha256(b"synthetic input\n").hexdigest()
    spec = Specification(
        "Preserve a declared development result",
        "hand-authored/v1",
        "generic-result/v1",
        "synthetic/v1",
        "test-authoring/v1",
        (digest,),
        digest,
        digest,
        EnvironmentRecord("cpython", "3.12.13", "cpython-312", "test-platform", "uv test", digest),
    )
    trial = TrialRecord(uuid4(), uuid4(), spec, ())
    exposure = ExposureEvent(
        uuid4(), "researcher", TIME, digest, "synthetic/v1", START, END, "result", "Read a result"
    )
    return PackageRecord(trial.trial_id, (trial,), (exposure,))


@pytest.mark.smoke
def test_saved_campaign_reopens_identically_from_a_copied_and_renamed_directory(tmp_path, first):
    assert (first.parent, first.branch, first.changes) == (None, "main", ())
    assert first.specification == specification()
    assert (first.trials, first.exposures) == ((PRIOR_TRIAL,), (PRIOR,))
    second = child(first, specification(evaluation=CHANGED["evaluation"]))
    origin = tmp_path / "origin"
    directory = save_revision(origin, first).parent
    save_revision(origin, second)
    campaign = open_campaign(directory)
    assert campaign.campaign_id == first.campaign_id
    assert campaign.revisions == (first, second)
    copied = tmp_path / "unrelated directory" / "renamed campaign"
    shutil.copytree(directory, copied)
    shutil.rmtree(origin)
    assert open_campaign(copied).revisions == (first, second)


def test_only_an_enabled_generated_code_section_enters_the_revision_bytes():
    revision = CampaignRevision(
        UUID("00000000-0000-4000-8000-000000000001"),
        "main",
        None,
        "researcher",
        TIME,
        "Initial protocol",
        specification(),
        (),
        (),
        (),
    )
    assert revision.revision_id == (
        "86f1ed5a379b5cf1dd96b586e635cd64c341f198ede0592df79a0339a9933170"
    )
    assert b"generated_code" not in file_bytes(revision)
    assert parse_file(file_bytes(revision)) == revision
    enabled = replace(revision, specification=specification(generated_code=True))
    content = file_bytes(enabled)
    assert json.loads(content)["revision"]["specification"]["generated_code"] is True
    assert parse_file(content) == enabled
    with pytest.raises(CampaignError, match="not the stored form of its revision"):
        parse_file(content.replace(b'"generated_code":true', b'"generated_code":false'))


def test_a_revision_graph_saved_before_p05_opens_with_its_original_ids_and_bytes(tmp_path):
    root = {
        "campaign_id": "00000000-0000-4000-8000-000000000001",
        "branch": "main",
        "parent": None,
        "author": "researcher",
        "created_at": "2026-10-21T12:00:00Z",
        "reason": "Initial protocol",
        "specification": {
            "question": "Does 12-1 momentum add value beyond book-to-market in Zürich listings?",
            "sources": [
                {
                    "name": "momentum-paper",
                    "locator": "https://example.org/momentum.pdf",
                    "digest": PAPER,
                    "role": "paper",
                }
            ],
            "data": [
                {
                    "dataset_id": "synthetic/v1",
                    "interval_start": "2000-01-01T00:00:00Z",
                    "interval_end": "2019-12-31T00:00:00Z",
                }
            ],
            "evaluation": {"profile": "pilot-evaluation/v1", "digest": RULES},
            "limits": {"max_trials": 20, "max_attempts": 60, "max_compute_seconds": 7200},
            "interventions": [],
            "model_roles": [],
        },
        "trials": [],
        "exposures": [],
        "changes": [],
    }
    # The IDs main's campaigns.py gave this root and child before the added limits existed.
    ids = [
        "1856708a4cead513eeb728ebe0560c1f9f1ad8d5734ba61ac87bb1655abc0b25",
        "c4f73d9a937be5d61a48ff788909f9323a3f17269cf1ecb7705168ba75e66f56",
    ]
    tightened = {"max_trials": 4, "max_attempts": 2, "max_compute_seconds": 100}
    second = root | {
        "parent": ids[0],
        "created_at": "2026-10-22T12:00:00Z",
        "reason": "Tighten the limits",
        "specification": root["specification"] | {"limits": tightened},
        "changes": ["limits"],
    }
    contents = [canonical({"schema_version": 1, "revision": value}) for value in (root, second)]
    for content in contents:
        write_revision(tmp_path, content)
    revisions = open_campaign(tmp_path).revisions
    assert [revision.revision_id for revision in revisions] == ids
    assert [file_bytes(revision) for revision in revisions] == contents


def test_a_schema_2_child_of_a_schema_1_parent_saves_and_lists_limits_as_its_change(
    tmp_path, first
):
    limits = replace(first.specification.limits, max_model_calls=3, max_wall_seconds=3600)
    second = child(first, replace(first.specification, limits=limits))
    assert second.changes == ("limits",)
    parent = save_revision(tmp_path, first)
    path = save_revision(tmp_path, second)
    assert [json.loads(item.read_bytes())["schema_version"] for item in (parent, path)] == [1, 2]
    assert open_campaign(path.parent).revisions == (first, second)


ADDED = {
    "max_model_calls": 3,
    "max_wall_seconds": 3600,
    "max_input_tokens": 50_000,
    "max_output_tokens": 20_000,
}


@pytest.mark.parametrize(
    "names", [names for size in range(5) for names in itertools.combinations(ADDED, size)]
)
def test_each_subset_of_the_added_limits_round_trips_under_one_schema_version(first, names):
    limits = replace(first.specification.limits, **{name: ADDED[name] for name in names})
    revision = replace(first, specification=replace(first.specification, limits=limits))
    content = file_bytes(revision)
    value = json.loads(content)
    assert value["schema_version"] == (2 if names else 1)
    stored = value["revision"]["specification"]["limits"]
    assert stored.keys() == {"max_trials", "max_attempts", "max_compute_seconds", *names}
    assert parse_file(content) == revision


@pytest.mark.parametrize(
    "damage, named",
    [
        ("v1-with-an-added-limit", "not the stored form of its revision"),
        ("v2-without-added-limits", "not the stored form of its revision"),
        ("null-limit", "max_model_calls must be a positive integer, not None"),
        ("zero-limit", "max_wall_seconds must be a positive integer, not 0"),
    ],
)
def test_open_refuses_added_limits_that_do_not_match_the_schema_version(
    tmp_path, first, damage, named
):
    limits = replace(first.specification.limits, max_model_calls=3)
    revision = replace(first, specification=replace(first.specification, limits=limits))
    path = save_revision(tmp_path, revision)
    value = json.loads(path.read_bytes())
    stored = value["revision"]["specification"]["limits"]
    match damage:
        case "v1-with-an-added-limit":
            value["schema_version"] = 1
        case "v2-without-added-limits":
            del stored["max_model_calls"]
        case "null-limit":
            stored["max_model_calls"] = None
        case "zero-limit":
            stored["max_wall_seconds"] = 0
    path.unlink()
    write_revision(path.parent, canonical(value))
    with pytest.raises(CampaignError, match=re.escape(named)):
        open_campaign(path.parent)


def test_revision_file_is_named_by_the_sha256_of_its_canonical_bytes(tmp_path, first):
    twin = CampaignRevision(
        first.campaign_id,
        "main",
        None,
        "researcher",
        TIME,
        "Initial protocol",
        specification(),
        (TrialReference(PRIOR_TRIAL.lineage_id, PRIOR_TRIAL.trial_id),),
        (replace(PRIOR),),
        (),
    )
    interval = {"interval_start": "2000-01-01T00:00:00Z", "interval_end": "2019-12-31T00:00:00Z"}
    expected = canonical(
        {
            "schema_version": 1,
            "revision": {
                "campaign_id": str(first.campaign_id),
                "branch": "main",
                "parent": None,
                "author": "researcher",
                "created_at": "2026-10-21T12:00:00Z",
                "reason": "Initial protocol",
                "specification": {
                    "question": (
                        "Does 12-1 momentum add value beyond book-to-market in Zürich listings?"
                    ),
                    "sources": [
                        {
                            "name": "momentum-paper",
                            "locator": "https://example.org/momentum.pdf",
                            "digest": PAPER,
                            "role": "paper",
                        }
                    ],
                    "data": [{"dataset_id": "synthetic/v1", **interval}],
                    "evaluation": {"profile": "pilot-evaluation/v1", "digest": RULES},
                    "limits": {"max_trials": 20, "max_attempts": 60, "max_compute_seconds": 7200},
                    "interventions": ["before_trial", "methodology_change"],
                    "model_roles": [
                        {
                            "role": "critic",
                            "model": "provider/model-x",
                            "fallbacks": ["provider/model-y"],
                            "status": "inactive",
                        }
                    ],
                },
                "trials": [
                    {
                        "lineage_id": str(PRIOR_TRIAL.lineage_id),
                        "trial_id": str(PRIOR_TRIAL.trial_id),
                    }
                ],
                "exposures": [
                    {
                        "id": str(PRIOR.id),
                        "actor": "researcher",
                        "at": "2026-09-21T12:00:00Z",
                        "material_digest": PAPER,
                        "dataset_id": "synthetic/v1",
                        **interval,
                        "kind": "data",
                        "reason": "Explored the data before the campaign",
                    }
                ],
                "changes": [],
            },
        }
    )
    digest = hashlib.sha256(expected).hexdigest()
    assert first.revision_id == twin.revision_id == digest
    path = save_revision(tmp_path, first)
    assert path == tmp_path / str(first.campaign_id) / f"{digest}.json"
    assert path.read_bytes() == expected


@pytest.mark.parametrize("section", list(CHANGED))
def test_changing_one_section_changes_the_identity_and_is_the_listed_change(first, section):
    changed = specification(**{section: CHANGED[section]})
    assert replace(first, specification=changed).revision_id != first.revision_id
    assert child(first, changed).changes == (section,)


@pytest.mark.parametrize("layout", ["shuffled-keys", "indented"])
def test_open_rejects_a_rehashed_revision_file_that_is_not_canonical(tmp_path, first, layout):
    path = save_revision(tmp_path, first)
    value = json.loads(path.read_bytes())
    if layout == "shuffled-keys":
        reordered = {"schema_version": 1, "revision": dict(reversed(value["revision"].items()))}
        text = json.dumps(reordered, separators=(",", ":"), ensure_ascii=False)
    else:
        text = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False)
    path.unlink()
    rewritten = write_revision(path.parent, (text + "\n").encode())
    with pytest.raises(CampaignError, match=re.escape(rewritten.name)):
        open_campaign(path.parent)


@pytest.mark.parametrize("name", ["0" * 64 + ".json", "initial-protocol.json"])
def test_open_rejects_a_renamed_revision_file(tmp_path, first, name):
    path = save_revision(tmp_path, first)
    path.rename(path.with_name(name))
    with pytest.raises(CampaignError, match=re.escape(name)):
        open_campaign(path.parent)


def test_saving_again_returns_the_same_file_and_never_rewrites_a_differing_one(tmp_path, first):
    path = save_revision(tmp_path, first)
    content = path.read_bytes()
    assert save_revision(tmp_path, first) == path
    assert path.read_bytes() == content
    assert [item.name for item in tmp_path.iterdir()] == [str(first.campaign_id)]
    assert [item.name for item in path.parent.iterdir()] == [path.name]
    altered = content.replace(b"Initial protocol", b"Altered protocol")
    path.write_bytes(altered)
    with pytest.raises(CampaignError, match=re.escape(path.name)):
        save_revision(tmp_path, first)
    assert path.read_bytes() == altered


def test_revising_evaluation_keeps_the_parent_file_and_moves_the_main_head(tmp_path, first):
    first_path = save_revision(tmp_path, first)
    first_bytes = first_path.read_bytes()
    second = revise(
        first,
        replace(first.specification, evaluation=CHANGED["evaluation"]),
        author="reviewer",
        at=TIME + timedelta(days=1),
        reason="Add costs after review",
    )
    second_path = save_revision(tmp_path, second)
    assert (second.parent, second.branch, second.changes) == (
        first.revision_id,
        "main",
        ("evaluation",),
    )
    assert first_path.read_bytes() == first_bytes
    assert sorted(first_path.parent.iterdir()) == sorted([first_path, second_path])
    campaign = open_campaign(first_path.parent)
    assert campaign.revisions == (first, second)
    assert campaign.head("main") == second
    both = replace(second.specification, limits=CHANGED["limits"], question=CHANGED["question"])
    assert child(second, both).changes == ("question", "limits")
    with pytest.raises(CampaignError, match=first.revision_id):
        child(first, specification(), at=TIME - timedelta(seconds=1))


def test_branches_keep_separate_heads_and_reject_forks_and_reused_names(tmp_path, first):
    directory = save_revision(tmp_path, first).parent
    second = child(first, specification(evaluation=CHANGED["evaluation"]))
    side = child(first, specification(question=CHANGED["question"]), branch="value-overlay")
    save_revision(tmp_path, second)
    save_revision(tmp_path, side)
    campaign = open_campaign(directory)
    assert campaign.head("main") == second
    assert campaign.head("value-overlay") == side
    assert (side.parent, side.branch) == (first.revision_id, "value-overlay")
    with pytest.raises(CampaignError, match="momentum-only"):
        campaign.head("momentum-only")
    with pytest.raises(CampaignError, match="'main'"):
        child(first, specification(limits=CHANGED["limits"]), branch="main")
    saved = sorted(directory.iterdir())
    rival = child(first, specification(limits=CHANGED["limits"]))
    with pytest.raises(CampaignError, match=first.revision_id):
        save_revision(tmp_path, rival)
    reused = child(second, specification(limits=CHANGED["limits"]), branch="value-overlay")
    with pytest.raises(CampaignError, match="value-overlay"):
        save_revision(tmp_path, reused)
    assert sorted(directory.iterdir()) == saved


def test_observations_carry_forward_and_a_hand_written_child_cannot_drop_them(
    tmp_path, first, package
):
    observed = tuple(TrialReference(trial.lineage_id, trial.trial_id) for trial in package.trials)
    second = child(
        first,
        specification(data=CHANGED["data"]),
        trials=observed,
        exposures=package.exposures,
    )
    assert second.changes == ("data",)
    assert second.trials == (PRIOR_TRIAL, *observed)
    assert second.exposures == (PRIOR, *package.exposures)
    third = child(second, replace(second.specification, question=CHANGED["question"]))
    assert (third.trials, third.exposures) == (second.trials, second.exposures)
    repeated = child(second, second.specification, trials=observed, exposures=package.exposures)
    assert (repeated.trials, repeated.exposures) == (second.trials, second.exposures)
    conflicting = replace(package.exposures[0], reason="A different account of the event")
    with pytest.raises(CampaignError, match=str(conflicting.id)):
        child(second, second.specification, exposures=(conflicting,))
    directory = save_revision(tmp_path, first).parent
    save_revision(tmp_path, second)
    save_revision(tmp_path, third)
    sibling = json.loads((directory / f"{second.revision_id}.json").read_bytes())["revision"] | {
        "parent": second.revision_id,
        "branch": "kept-exposure",
        "created_at": "2026-10-25T12:00:00Z",
        "changes": [],
    }
    write_revision(directory, canonical({"schema_version": 1, "revision": sibling}))
    assert open_campaign(directory).head("kept-exposure").exposures == second.exposures
    dropped = sibling | {"branch": "dropped-exposure", "exposures": sibling["exposures"][1:]}
    path = write_revision(directory, canonical({"schema_version": 1, "revision": dropped}))
    with pytest.raises(CampaignError, match=path.stem):
        open_campaign(directory)


def test_a_later_revision_must_carry_what_another_branch_already_observed(tmp_path, first, package):
    observed = tuple(TrialReference(trial.lineage_id, trial.trial_id) for trial in package.trials)
    side = child(
        first,
        specification(question=CHANGED["question"]),
        branch="value-overlay",
        trials=observed,
        exposures=package.exposures,
    )
    directory = save_revision(tmp_path, first).parent
    save_revision(tmp_path, side)
    later = {"at": side.created_at + timedelta(days=1)}
    unaware = child(first, specification(evaluation=CHANGED["evaluation"]), **later)
    with pytest.raises(CampaignError, match=str(observed[0].trial_id)):
        save_revision(tmp_path, unaware)
    trials_only = child(
        first, specification(evaluation=CHANGED["evaluation"]), trials=observed, **later
    )
    with pytest.raises(CampaignError, match=str(package.exposures[0].id)):
        save_revision(tmp_path, trials_only)
    aware = child(
        first,
        specification(evaluation=CHANGED["evaluation"]),
        trials=observed,
        exposures=package.exposures,
        **later,
    )
    save_revision(tmp_path, aware)
    head = open_campaign(directory).head("main")
    assert (head.trials, head.exposures) == ((PRIOR_TRIAL, *observed), (PRIOR, *package.exposures))
    concurrent = child(first, specification(limits=CHANGED["limits"]), branch="limits")
    assert concurrent.created_at == side.created_at
    save_revision(tmp_path, concurrent)


@pytest.mark.parametrize("hours", [-1, 0, 1])
@pytest.mark.parametrize("kind", ["trials", "exposures"])
def test_saving_rejects_a_branch_that_records_an_id_differently(
    tmp_path, first, package, kind, hours
):
    trial = TrialReference(package.trials[0].lineage_id, package.trials[0].trial_id)
    exposure = package.exposures[0]
    recorded, rival, conflict = {
        "trials": (trial, replace(trial, lineage_id=uuid4()), trial.trial_id),
        "exposures": (exposure, replace(exposure, reason="Another account"), exposure.id),
    }[kind]
    north = child(
        first, specification(question=CHANGED["question"]), branch="north", **{kind: (recorded,)}
    )
    south = child(
        first,
        specification(limits=CHANGED["limits"]),
        branch="south",
        at=north.created_at + timedelta(hours=hours),
        **{kind: (rival,)},
    )
    directory = save_revision(tmp_path, first).parent
    save_revision(tmp_path, north)
    saved = sorted(directory.iterdir())
    with pytest.raises(CampaignError, match=f"record different {kind} under ID {conflict}"):
        save_revision(tmp_path, south)
    assert sorted(directory.iterdir()) == saved
    save_revision(tmp_path, child(north, north.specification))


def test_saving_names_the_new_revision_and_a_date_that_works_when_later_ones_lack_its_exposure(
    tmp_path, first, package
):
    main = child(
        first, specification(evaluation=CHANGED["evaluation"]), at=TIME + timedelta(hours=2)
    )
    head = child(main, main.specification, at=TIME + timedelta(hours=3))
    side = child(
        first,
        specification(question=CHANGED["question"]),
        branch="value-overlay",
        at=TIME + timedelta(hours=1),
        exposures=package.exposures,
    )
    for revision in (first, main, head):
        save_revision(tmp_path, revision)
    with pytest.raises(
        CampaignError, match=f"Revision {side.revision_id} records exposures"
    ) as error:
        save_revision(tmp_path, side)
    assert str(error.value).endswith("date it no earlier than 2026-10-21T15:00:00Z")
    later = child(
        first,
        side.specification,
        branch="value-overlay",
        at=head.created_at,
        exposures=package.exposures,
    )
    save_revision(tmp_path, later)


def test_a_revision_that_later_ones_lack_on_both_kinds_gets_one_date_that_saves(tmp_path, first):
    trial = TrialReference(uuid4(), uuid4())
    exposure = replace(PRIOR, id=uuid4(), at=TIME)
    north = child(first, first.specification, branch="north", at=TIME + timedelta(hours=3))
    south = child(
        first, first.specification, branch="south", at=TIME + timedelta(hours=5), trials=(trial,)
    )
    for revision in (first, north, south):
        save_revision(tmp_path, revision)

    def side(at: datetime) -> CampaignRevision:
        return child(
            first, first.specification, branch="side", at=at, trials=(trial,), exposures=(exposure,)
        )

    with pytest.raises(CampaignError, match="date it no earlier than") as error:
        save_revision(tmp_path, side(TIME + timedelta(hours=1)))
    suggested = datetime.fromisoformat(str(error.value).rsplit(" ", 1)[1])
    assert suggested == south.created_at
    save_revision(tmp_path, side(suggested))


def test_open_and_save_skip_dot_files_beside_the_revisions(tmp_path, first):
    directory = save_revision(tmp_path, first).parent
    (directory / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")
    assert open_campaign(directory).revisions == (first,)
    revised = child(first, first.specification)
    save_revision(tmp_path, revised)
    assert open_campaign(directory).head("main") == revised


@pytest.mark.parametrize(
    "damage",
    [
        "dropped-trial",
        "unlisted-change",
        "earlier-date",
        "other-campaign",
        "second-root",
        "foreign-root",
    ],
)
def test_open_rejects_a_hand_written_revision_that_breaks_the_history(tmp_path, first, damage):
    directory = save_revision(tmp_path, first).parent
    revision = json.loads((directory / f"{first.revision_id}.json").read_bytes())["revision"] | {
        "parent": first.revision_id,
        "branch": "hand-written",
        "created_at": "2026-10-22T12:00:00Z",
    }
    control = write_revision(directory, canonical({"schema_version": 1, "revision": revision}))
    assert open_campaign(directory).head("hand-written").parent == first.revision_id
    control.unlink()
    match damage:
        case "dropped-trial":
            revision["trials"] = []
        case "unlisted-change":
            revision["specification"]["question"] = "Does a quietly changed question count?"
        case "earlier-date":
            revision["created_at"] = "2026-10-20T12:00:00Z"
        case "other-campaign":
            revision["campaign_id"] = str(uuid4())
        case "second-root":
            revision["parent"] = None
        case "foreign-root":
            revision |= {"parent": None, "campaign_id": str(uuid4())}
    path = write_revision(directory, canonical({"schema_version": 1, "revision": revision}))
    named = revision["campaign_id"] if damage == "foreign-root" else path.stem
    with pytest.raises(CampaignError, match=named):
        open_campaign(directory)


def test_open_rejects_a_symlinked_entry_even_to_identical_bytes(tmp_path, first):
    path = save_revision(tmp_path / "campaigns", first)
    outside = tmp_path / "outside.json"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(CampaignError, match=re.escape(path.name)):
        open_campaign(path.parent)


def test_revisions_list_parents_first_then_by_time_and_identity(tmp_path, first):
    tied = next(
        revision
        for revision in (
            child(first, first.specification, at=TIME, reason=f"Record decision {number}")
            for number in itertools.count()
        )
        if revision.revision_id < first.revision_id
    )
    later = [child(first, first.specification, branch=name) for name in ("north", "south")]
    for revision in (first, tied, *later):
        save_revision(tmp_path, revision)
    campaign = open_campaign(tmp_path / str(first.campaign_id))
    assert campaign.revisions == (first, tied, *sorted(later, key=lambda item: item.revision_id))
    assert campaign.head("main") == tied


@pytest.mark.parametrize(
    "damage, named",
    [
        ("zero-limit", "max_trials"),
        ("boolean-limit", "max_trials"),
        ("zero-model-calls", "max_model_calls"),
        ("integer-gate", "not the stored form of its revision"),
        ("bad-digest", "Invalid SHA-256 digest"),
        ("unknown-status", "'paused'"),
        ("reversed-interval", "synthetic/v1"),
        ("duplicate-source", "momentum-paper"),
        ("duplicate-permission", "synthetic/v1"),
        ("duplicate-role", "critic"),
        ("duplicate-intervention", "before_trial"),
        ("duplicate-trial", str(PRIOR_TRIAL.trial_id)),
        ("duplicate-exposure", str(PRIOR.id)),
        ("naive-time", "timezone"),
        ("branch-name", "Invalid branch name"),
        ("source-name", "Invalid source name"),
        ("date-time", "datetime.date(2026, 10, 21)"),
        ("future-exposure", str(PRIOR.id)),
    ],
)
def test_authoring_rejects_invalid_input_and_names_the_offending_value(damage, named):
    spec = specification()
    arguments = {"author": "researcher", "at": TIME, "reason": "Initial protocol"}
    match damage:
        case "zero-limit":
            spec = specification(limits=ResourceLimits(0, 60, 7200))
        case "boolean-limit":
            spec = specification(limits=ResourceLimits(True, 60, 7200))
        case "zero-model-calls":
            spec = specification(limits=ResourceLimits(20, 60, 7200, max_model_calls=0))
        case "integer-gate":
            spec = specification(generated_code=1)
        case "bad-digest":
            spec = specification(evaluation=EvaluationRules("pilot-evaluation/v1", "not-a-digest"))
        case "unknown-status":
            spec = specification(
                model_roles=(ModelRole("critic", "provider/model-x", (), "paused"),)
            )
        case "reversed-interval":
            spec = specification(data=(DataPermission("synthetic/v1", END, START),))
        case "duplicate-source":
            spec = specification(sources=spec.sources * 2)
        case "duplicate-permission":
            spec = specification(data=spec.data * 2)
        case "duplicate-role":
            spec = specification(model_roles=spec.model_roles * 2)
        case "duplicate-intervention":
            spec = specification(interventions=("before_trial", "before_trial"))
        case "duplicate-trial":
            arguments["trials"] = (PRIOR_TRIAL, PRIOR_TRIAL)
        case "duplicate-exposure":
            arguments["exposures"] = (PRIOR, PRIOR)
        case "naive-time":
            arguments["at"] = TIME.replace(tzinfo=None)
        case "date-time":
            arguments["at"] = TIME.date()
        case "branch-name":
            arguments["branch"] = "Main Branch"
        case "source-name":
            spec = specification(sources=(replace(spec.sources[0], name="Momentum Paper"),))
        case "future-exposure":
            arguments["exposures"] = (replace(PRIOR, at=TIME + timedelta(seconds=1)),)
    with pytest.raises(CampaignError, match=re.escape(named)):
        start_campaign(spec, **arguments)


@pytest.mark.parametrize(
    "observed",
    [{"trials": (object(),)}, {"trials": (PRIOR,)}, {"exposures": (set(),)}],
)
def test_revise_rejects_observations_that_are_not_records(first, observed):
    with pytest.raises(CampaignError, match="Cannot revise campaign"):
        child(first, first.specification, **observed)


@pytest.mark.parametrize(
    "damage, named",
    [
        ("unknown-field", "unknown 'max_gpus'"),
        ("missing-field", "missing 'reason'"),
        ("unknown-section", "'budget'"),
        ("duplicate-change", "'data'"),
        ("root-changes", "changes"),
        ("boolean-version", "schema version"),
    ],
)
def test_open_rejects_hand_edited_revision_files_and_names_the_problem(
    tmp_path, first, damage, named
):
    path = save_revision(tmp_path, first)
    value = json.loads(path.read_bytes())
    revision = value["revision"]
    match damage:
        case "unknown-field":
            revision["specification"]["limits"]["max_gpus"] = 8
        case "missing-field":
            del revision["reason"]
        case "unknown-section":
            revision["changes"] = ["budget"]
        case "duplicate-change":
            revision["changes"] = ["data", "data"]
        case "root-changes":
            revision["changes"] = ["data"]
        case "boolean-version":
            value["schema_version"] = True
    path.unlink()
    write_revision(path.parent, canonical(value))
    with pytest.raises(CampaignError, match=re.escape(named)):
        open_campaign(path.parent)


def test_a_revision_whose_parent_is_missing_is_rejected_at_save_and_open(tmp_path, first):
    second = child(first, specification(question=CHANGED["question"]))
    with pytest.raises(CampaignError, match=first.revision_id):
        save_revision(tmp_path / "fresh", second)
    assert not (tmp_path / "fresh").exists()
    directory = save_revision(tmp_path, first).parent
    save_revision(tmp_path, second)
    (directory / f"{first.revision_id}.json").unlink()
    with pytest.raises(CampaignError, match=first.revision_id):
        open_campaign(directory)


CODEX = "codex-subscription/gpt-6.1-sol"
CODEX_POLICY = ProviderPolicy(
    provider=Provider("codex-subscription"), labels=frozenset(), retains=True
)
CLAUDE_POLICY = ProviderPolicy(
    provider=Provider("claude-subscription"), labels=frozenset(), retains=True
)
VENDOR = licensed("synthetic-vendor")
CUTOFF = datetime(2025, 6, 1, tzinfo=UTC)
OCTOBER = datetime(2025, 12, 17, tzinfo=UTC)
WRITTEN = datetime(2025, 1, 1, tzinfo=UTC)
BRIEF = "The pilot asks whether the real-time Sahm rule times the market.\n"
INSTRUCTIONS = "You are the researcher. Propose one signal program at a time.\n"
DECLARATION = SourceDeclaration(
    media_type="text/markdown",
    access_domain="repository",
    location="docs/phase0-pilot.md",
    revision="1",
    acquisition="hand-authored test fixture",
    rights="synthetic, unrestricted",
    labels=frozenset(),
    approved_for=frozenset({Provider("codex-subscription")}),
    published_at=None,
    source_available_at=WRITTEN,
    availability_evidence="authored for these tests",
    received_at=WRITTEN,
)
ROWS = [
    {
        "arrived_at": "2025-02-08T00:00:00Z",
        "key": "2025-01-01",
        "published_at": "2025-02-08T00:00:00Z",
        "revision": 1,
        "values": {"value": "4.0"},
    }
]
STORED = ("alfred.unrate", "alfred.sahmrealtime", "alfred.payems", "french.ff3_monthly")


def dataset_file(name: str) -> bytes:
    return canonical_bytes({"dataset": name, "rows": ROWS, "schema_version": 1})


def documented(name, *steps, access=()):
    vintages = tuple(
        Vintage(
            aspect, Evidence(hashlib.sha256(aspect.encode()).hexdigest(), f"Synthetic {aspect}")
        )
        for aspect in ("publication", "revision", "arrival")
    )
    vintage = hashlib.sha256(dataset_file(name)).hexdigest()
    events = {
        "documented": Documented(
            name,
            "researcher",
            TIME,
            "Data Engineering",
            ("value",),
            access,
            vintage,
            (),
            vintages,
            ("codex-subscription",),
        ),
        "validated": Validated(name, "reviewer", TIME, vintages[0].evidence),
        "approved": Approved(name, "lead", TIME, datasets.ADAPTER),
    }
    return tuple(events[step] for step in ("documented", *steps))


CATALOG = Catalog(
    (
        *(event for name in STORED for event in documented(name, "validated", "approved")),
        *documented("vendor.documented"),
        *documented("vendor.restricted", "validated", "approved", access=(VENDOR,)),
    )
)


@pytest.fixture
def setup(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    store = tmp_path / "datasets"
    for name in STORED:
        datasets.save(store, name, dataset_file(name))
    rules = tmp_path / "rules"
    rules.mkdir()
    (rules / f"{FROZEN.digest}.json").write_bytes(FROZEN.data)
    return Setup(
        sources=sources,
        datasets=store,
        catalog=CATALOG,
        ledger=Ledger(),
        clearances=frozenset(),
        cutoff=CUTOFF,
        policies=(CODEX_POLICY, CLAUDE_POLICY),
        rules=rules,
        packages=tmp_path / "packages",
        lock=b"synthetic lock\n",
    )


def brief(setup) -> Source:
    return register(
        setup.sources,
        name="pilot-brief",
        role="brief",
        data=BRIEF.encode(),
        declaration=DECLARATION,
    )


def prompt(setup, role="researcher", **declared) -> Source:
    declaration = replace(DECLARATION, location=f"prompts/{role}.md", **declared)
    return register(
        setup.sources, name=role, role=PROMPT, data=INSTRUCTIONS.encode(), declaration=declaration
    )


RESEARCHER = ModelRole("researcher", CODEX, (), "active")


def pilot(setup, names=("alfred.unrate", "alfred.sahmrealtime"), roles=(RESEARCHER,), **changes):
    sources = (brief(setup), prompt(setup), *contract_sources(setup.sources, setup.catalog, names))
    spec = specification(
        sources=sources,
        data=tuple(DataPermission(name, START, END) for name in names),
        evaluation=EvaluationRules("pilot-evaluation/v1", FROZEN.digest),
        model_roles=roles,
    )
    return start_campaign(
        replace(spec, **changes), author="researcher", at=TIME, reason="Pilot protocol"
    )


def call(authority, tool, **arguments):
    return authority.toolset.call(tool, json.dumps(arguments))


def test_an_active_role_round_trips_through_its_revision_file(first):
    revision = child(first, specification(model_roles=(RESEARCHER,)))
    stored = json.loads(file_bytes(revision))["revision"]["specification"]["model_roles"]
    assert stored == [{"role": "researcher", "model": CODEX, "fallbacks": [], "status": "active"}]
    assert parse_file(file_bytes(revision)) == revision


def test_register_returns_a_source_pinned_by_its_text_generation(setup):
    source = brief(setup)
    generation = open_generation(setup.sources, source.locator)
    assert source == Source("pilot-brief", generation.id, generation.digest, "brief")
    assert generation.occurrence.declaration == DECLARATION
    assert "".join(chunk.text for chunk in generation.chunks) == BRIEF.strip()
    assert brief(setup) == source


def test_contract_sources_pin_each_part_of_each_named_contract_once(setup):
    sources = contract_sources(
        setup.sources, setup.catalog, ["alfred.unrate", "vendor.documented", "alfred.unrate"]
    )
    assert [(source.name, source.role) for source in sources] == [
        ("alfred.unrate.contract.0", CONTRACT),
        ("alfred.unrate.contract.1", CONTRACT),
    ]
    for source in sources:
        generation = open_generation(setup.sources, source.locator)
        assert contract_of(generation) == "alfred.unrate"
        assert source.digest == generation.digest
    with pytest.raises(DatasetError):
        contract_sources(setup.sources, setup.catalog, ["zq.missing"])


def test_resolve_scopes_the_context_to_exactly_the_permitted_datasets(setup):
    data = (
        DataPermission("alfred.unrate", START, END),
        DataPermission("alfred.unrate", END, END + timedelta(days=365)),
        DataPermission("french.ff3_monthly", START, END),
    )
    revision = pilot(setup, ("alfred.unrate", "french.ff3_monthly"), data=data)
    authority = resolve(revision, setup)
    assert authority.revision == revision.revision_id
    assert authority.data == data
    assert authority.context.scope == {"alfred.unrate", "french.ff3_monthly"}
    assert authority.context.cutoff == CUTOFF
    assert authority.search.context == authority.context.narrow(cutoff=FROZEN.search.cutoff)
    assert authority.toolset.context is authority.search.context


def test_a_tool_read_of_an_approved_dataset_the_campaign_does_not_permit_is_refused(setup):
    authority = resolve(pilot(setup), setup)
    assert isinstance(call(authority, "read_dataset", dataset="alfred.unrate").outcome, Supplied)
    refused = call(authority, "read_dataset", dataset="alfred.payems")
    missing = call(authority, "read_dataset", dataset="zq.missing")
    assert refused.outcome == missing.outcome == Refusal("unknown_dataset")
    assert refused.material == missing.material


def test_list_datasets_names_exactly_the_campaigns_readable_datasets(setup):
    names = ("alfred.unrate", "french.ff3_monthly", "vendor.documented", "vendor.restricted")
    revision = pilot(setup, names, sources=(brief(setup), prompt(setup)))
    listed = json.loads(call(resolve(revision, setup), "list_datasets").material.text)
    assert [item["dataset"] for item in listed["datasets"]] == [
        "alfred.unrate",
        "french.ff3_monthly",
    ]


def test_the_corpus_is_every_source_but_the_prompts(setup):
    revision = pilot(setup)
    authority = resolve(revision, setup)
    members = [source for source in revision.specification.sources if source.role != PROMPT]
    assert authority.corpus.generations == {source.locator: source.digest for source in members}
    assert [source.name for source in members] == [
        "pilot-brief",
        "alfred.sahmrealtime.contract.0",
        "alfred.sahmrealtime.contract.1",
        "alfred.unrate.contract.0",
        "alfred.unrate.contract.1",
    ]
    assert authority.context.generation == authority.corpus.id


def test_resolve_refuses_the_contract_of_a_dataset_the_revision_does_not_permit(setup):
    revision = pilot(setup, ("alfred.unrate",))
    payems = contract_sources(setup.sources, setup.catalog, ["alfred.payems"])
    spec = revision.specification
    widened = child(revision, replace(spec, sources=(*spec.sources, *payems)))
    with pytest.raises(CampaignError, match="alfred.payems"):
        resolve(widened, setup)


def test_resolve_refuses_an_unpermitted_contract_pinned_as_a_prompt(setup):
    revision = pilot(setup, ("alfred.unrate",))
    part = contract_sources(setup.sources, setup.catalog, ["alfred.payems"])[0]
    researcher = replace(part, name="researcher", role=PROMPT)
    spec = revision.specification
    sources = tuple(
        researcher if source.name == "researcher" else source for source in spec.sources
    )
    with pytest.raises(CampaignError, match="contract of 'alfred.payems'"):
        resolve(child(revision, replace(spec, sources=sources)), setup)


def test_resolve_refuses_a_source_at_a_contract_location_whose_revision_names_no_part(setup):
    revision = pilot(setup, ("alfred.unrate",))
    location = f"{datasets.ADAPTER}/contract/alfred.unrate"
    draft = register(
        setup.sources,
        name="draft",
        role=CONTRACT,
        data=b"A draft conclusion about the unemployment rate.\n",
        declaration=replace(DECLARATION, location=location, revision="draft"),
    )
    spec = revision.specification
    with pytest.raises(CampaignError, match="'draft' is at a contract location"):
        resolve(child(revision, replace(spec, sources=(*spec.sources, draft))), setup)


REVOKED = Revoked("alfred.unrate", "lead", TIME, "Withdrawn for review")


def redocumented(**changes):
    documentation, *steps = documented("alfred.unrate", "validated", "approved")
    return (REVOKED, replace(documentation, **changes), *steps)


@pytest.mark.parametrize(
    ("events", "clearances"),
    [
        ((REVOKED,), frozenset()),
        (redocumented(access=(VENDOR,)), frozenset({VENDOR})),
        (redocumented(approved_for=()), frozenset()),
    ],
    ids=["revoked", "stricter-labels", "fewer-approvals"],
)
def test_resolve_refuses_a_contract_the_catalog_now_restricts_further(setup, events, clearances):
    revision = pilot(setup, ("alfred.unrate",))
    changed = replace(setup, catalog=Catalog((*CATALOG.events, *events)), clearances=clearances)
    with pytest.raises(CampaignError, match="contract of 'alfred.unrate'"):
        resolve(revision, changed)


def test_tools_search_and_describe_as_of_the_search_cutoff_whatever_the_setup_cutoff(setup):
    revision = pilot(setup, ("alfred.unrate",))
    authority = resolve(revision, setup)
    # The brief is declared in 2025, after the search cutoff, so only the contract is found.
    for text, found in (("Sahm rule", False), ("alfred.unrate", True), ("signal program", False)):
        result = call(authority, "search_corpus", mode="identifier", text=text)
        assert bool(json.loads(result.material.text)["units"]) is found
    later = resolve(revision, replace(setup, cutoff=OCTOBER))
    for resolved in (authority, later):
        described = call(resolved, "describe_dataset", dataset="alfred.unrate").material.text
        assert "2025-10-01" not in described
    wide = datasets.describe(setup.catalog, setup.sources, "alfred.unrate", later.context)
    assert "2025-10-01" in "".join(unit.text for unit in wide.units)
    refused = call(authority, "describe_dataset", dataset="alfred.payems")
    assert refused.outcome == Refusal("unknown_dataset")


def test_a_revision_whose_rules_are_missing_renamed_or_invalid_refuses_to_resolve(setup):
    other = FROZEN.data.replace(b'"0.001"', b'"0.002"')
    (setup.rules / f"{hashlib.sha256(other).hexdigest()}.json").write_bytes(FROZEN.data)
    invalid = json.dumps(json.loads(FROZEN.data), indent=1).encode()
    (setup.rules / f"{hashlib.sha256(invalid).hexdigest()}.json").write_bytes(invalid)
    for data in (b"unwritten rules\n", other, invalid):
        named = EvaluationRules("pilot-evaluation/v1", hashlib.sha256(data).hexdigest())
        with pytest.raises(RulesError, match="No frozen rules file"):
            resolve(pilot(setup, evaluation=named), setup)


def test_a_revision_of_another_procedure_or_a_setup_short_of_the_held_back_cutoff_is_refused(
    setup,
):
    other = EvaluationRules("pilot-evaluation/v2", FROZEN.digest)
    with pytest.raises(RulesError, match="'pilot-evaluation/v2'"):
        resolve(pilot(setup, evaluation=other), setup)
    short = replace(setup, cutoff=FROZEN.held_back.cutoff - timedelta(seconds=1))
    with pytest.raises(RulesError, match="held-back window's cutoff"):
        resolve(pilot(setup), short)
    assert issubclass(RulesError, CampaignError)


CLAUDE = "claude-subscription/claude-opus-5-5"


@pytest.mark.parametrize(
    ("role", "assignment", "policies"),
    [
        (RESEARCHER, RoleAssignment("researcher", MODELS[CODEX]), (CODEX_POLICY,)),
        (
            ModelRole("researcher", CLAUDE, (), "active"),
            RoleAssignment("researcher", MODELS[CLAUDE]),
            (CLAUDE_POLICY,),
        ),
        (
            ModelRole("researcher", CODEX, (CLAUDE,), "active"),
            RoleAssignment("researcher", MODELS[CODEX], (MODELS[CLAUDE],)),
            (CLAUDE_POLICY, CODEX_POLICY),
        ),
    ],
    ids=["codex", "claude", "claude-fallback"],
)
def test_an_active_accepted_role_resolves_to_an_active_role_version(
    setup, role, assignment, policies
):
    authority = resolve(pilot(setup, roles=(role,)), setup)
    assert authority.roles.activation == "active"
    assert authority.roles.assignments == (assignment,)
    assert authority.context.providers == {policy.provider for policy in policies}
    assert authority.policies == policies


def test_inactive_roles_are_skipped_without_a_model_lookup_and_their_prompts_ignored(setup):
    unreadable = prompt(setup, "critic", source_available_at=OCTOBER)
    critic = ModelRole("critic", "provider/model-x", ("provider/model-y",))
    revision = pilot(setup, roles=(critic, replace(RESEARCHER, status="inactive")))
    spec = revision.specification
    revision = child(revision, replace(spec, sources=(*spec.sources, unreadable)))
    authority = resolve(revision, setup)
    assert (authority.roles.activation, authority.roles.assignments) == ("inactive", ())
    assert (authority.context.providers, authority.policies) == (frozenset(), ())
    assert authority.prompts == {}
    assert unreadable.locator not in authority.corpus.generations


UNCHECKED = "unchecked-provider/model-x"


@pytest.mark.parametrize(
    "role",
    [
        ModelRole("researcher", UNCHECKED, (), "active"),
        ModelRole("researcher", CODEX, (UNCHECKED,), "active"),
    ],
    ids=["primary", "fallback"],
)
def test_an_active_role_on_an_unaccepted_provider_refuses_to_resolve(setup, role, monkeypatch):
    record = ModelRecord("unchecked-provider", "model-x", UNKNOWN, UNKNOWN)
    monkeypatch.setattr("quant_research.campaigns.MODELS", {**MODELS, UNCHECKED: record})
    unaccepted = re.escape("Providers ['unchecked-provider'] have not passed a live check")
    with pytest.raises(CampaignError, match=unaccepted):
        resolve(pilot(setup, roles=(role,)), setup)


@pytest.mark.parametrize(
    ("roles", "policies", "named"),
    [
        ((ModelRole("researcher", "codex-subscription/gpt-9", (), "active"),), None, "gpt-9"),
        ((RESEARCHER,), (CLAUDE_POLICY,), "No provider policy for ['codex-subscription']"),
        ((RESEARCHER,), (CODEX_POLICY, CODEX_POLICY), "Duplicate provider policy"),
        ((RESEARCHER, ModelRole("critic", CODEX, (), "active")), None, "'critic' has no prompt"),
    ],
    ids=["unknown-model", "missing-policy", "duplicate-policy", "missing-prompt"],
)
def test_resolve_refuses_an_unknown_model_a_missing_policy_or_prompt(setup, roles, policies, named):
    configured = setup if policies is None else replace(setup, policies=policies)
    with pytest.raises(CampaignError, match=re.escape(named)):
        resolve(pilot(setup, roles=roles), configured)


def test_resolve_refuses_an_uncatalogued_dataset_or_an_unpublished_source(setup):
    revision = pilot(setup)
    spec = revision.specification
    for changed, named in (
        (
            replace(spec, data=(*spec.data, DataPermission("synthetic/v1", START, END))),
            "synthetic/v1",
        ),
        (
            replace(spec, sources=(*spec.sources, Source("lost", "0" * 64, "0" * 64, "paper"))),
            "lost",
        ),
    ):
        with pytest.raises(CampaignError, match=re.escape(named)):
            resolve(child(revision, changed), setup)


def test_resolve_refuses_a_source_pinned_to_another_stores_manifest_of_its_generation(
    setup, tmp_path
):
    revision = pilot(setup)
    spec = revision.specification
    other = tmp_path / "other"
    other.mkdir()
    occurrence = register_source(other, BRIEF.encode(), DECLARATION)
    unchunked = replace(TEXT_PIPELINE, split=lambda text: ())
    elsewhere = publish_generation(other, occurrence.id, unchunked)
    assert elsewhere.id == brief(setup).locator
    moved = Source("pilot-brief", elsewhere.id, elsewhere.digest, "brief")
    sources = tuple(moved if source.name == "pilot-brief" else source for source in spec.sources)
    with pytest.raises(CampaignError, match="'pilot-brief' pins manifest digest"):
        resolve(child(revision, replace(spec, sources=sources)), setup)


@pytest.mark.parametrize(
    "declared",
    [{"source_available_at": OCTOBER}, {"labels": frozenset({VENDOR})}],
    ids=["after-cutoff", "uncleared"],
)
def test_a_prompt_the_run_cannot_read_refuses_to_resolve(setup, declared):
    revision = pilot(setup)
    spec = revision.specification
    hidden = prompt(setup, **declared)
    sources = tuple(hidden if source.name == "researcher" else source for source in spec.sources)
    with pytest.raises(CampaignError, match="prompt of role 'researcher'"):
        resolve(child(revision, replace(spec, sources=sources)), setup)


def test_a_prompt_carries_its_whole_text_under_its_declared_restrictions(setup):
    authority = resolve(pilot(setup), setup)
    restrictions = replace(DECLARATION, location="prompts/researcher.md").restrictions
    assert authority.prompts == {"researcher": Material(INSTRUCTIONS, restrictions)}
    assert restrictions == Restrictions(
        available_at=WRITTEN, labels=frozenset(), approved_for={"codex-subscription"}
    )


def test_a_prompt_whose_generation_pins_other_text_refuses_to_resolve(setup):
    revision = pilot(setup)
    spec = revision.specification
    occurrence = open_generation(setup.sources, prompt(setup).locator).occurrence
    redacting = replace(
        TEXT_PIPELINE, extractor=Step("redact", "1"), extract=lambda data, media_type: "REDACTED"
    )
    redacted = publish_generation(setup.sources, occurrence.id, redacting)
    sources = tuple(
        replace(source, locator=redacted.id, digest=redacted.digest)
        if source.name == "researcher"
        else source
        for source in spec.sources
    )
    with pytest.raises(CampaignError, match="'researcher' does not match the text digest"):
        resolve(child(revision, replace(spec, sources=sources)), setup)


def test_generated_code_follows_the_revision(setup, tmp_path):
    (tmp_path / "runs").mkdir()
    bound = replace(setup, generated=Generated(RunLog(tmp_path / "runs"), tmp_path))
    assert resolve(pilot(setup), bound).toolset.generated is None
    revision = pilot(setup, generated_code=True)
    assert resolve(revision, bound).toolset.generated is bound.generated
    with pytest.raises(CampaignError, match="enables generated code"):
        resolve(revision, setup)
    tight = replace(bound, generated=replace(bound.generated, limits=Limits(timeout=5.0)))
    assert resolve(revision, tight).record() != resolve(revision, bound).record()


def test_resolving_again_returns_an_equal_authority_and_writes_nothing_new(setup):
    revision = pilot(setup)
    first_authority = resolve(revision, setup)
    written = sorted(setup.sources.rglob("*"))
    again = resolve(revision, setup)
    assert again == first_authority
    assert again.record() == first_authority.record()
    assert sorted(setup.sources.rglob("*")) == written


def test_a_steered_child_resolves_to_its_own_authority(setup):
    parent = pilot(setup, roles=(replace(RESEARCHER, status="inactive"),))
    names = ("alfred.unrate", "french.ff3_monthly")
    sources = (brief(setup), prompt(setup), *contract_sources(setup.sources, setup.catalog, names))
    steered = replace(
        parent.specification,
        sources=sources,
        data=tuple(DataPermission(name, START, END) for name in names),
        model_roles=(RESEARCHER,),
    )
    steered_child = child(parent, steered)
    assert steered_child.changes == ("sources", "data", "model_roles")
    before = resolve(parent, setup)
    after = resolve(steered_child, setup)
    assert before.context.scope == {"alfred.unrate", "alfred.sahmrealtime"}
    assert after.context.scope == {"alfred.unrate", "french.ff3_monthly"}
    assert before.corpus != after.corpus
    assert set(after.corpus.generations) == {
        source.locator for source in sources if source.role != PROMPT
    }
    assert (before.roles.activation, after.roles.activation) == ("inactive", "active")
    assert (before.prompts, list(after.prompts)) == ({}, ["researcher"])
    assert resolve(parent, setup) == before
    assert call(after, "read_dataset", dataset="alfred.sahmrealtime").outcome == Refusal(
        "unknown_dataset"
    )


def test_the_authority_record_names_exactly_the_datasets_corpus_and_active_roles(setup):
    critic = ModelRole("critic", "provider/model-x", ("provider/model-y",))
    revision = pilot(setup, roles=(critic, RESEARCHER))
    authority = resolve(revision, setup)
    content = authority.record()
    record = json.loads(content)
    assert canonical_bytes(record) == content
    assert record == {
        "schema_version": 3,
        "dataset_periods": [item.record() for item in authority.context.dataset_periods],
        "revision": revision.revision_id,
        "catalog": CATALOG.digest,
        "context": authority.context.identity,
        "search": authority.search.context.identity,
        "rules": FROZEN.digest,
        "scope": ["alfred.sahmrealtime", "alfred.unrate"],
        "data": [
            {
                "dataset_id": name,
                "interval_start": "2000-01-01T00:00:00Z",
                "interval_end": "2019-12-31T00:00:00Z",
            }
            for name in ("alfred.unrate", "alfred.sahmrealtime")
        ],
        "corpus": {"id": authority.corpus.id, "generations": dict(authority.corpus.generations)},
        "roles": {
            "version": authority.roles.version_id,
            "activation": "active",
            "assignments": [{"role": "researcher", "models": [CODEX]}],
        },
        "policies": [{"provider": "codex-subscription", "version": CODEX_POLICY.version}],
        "prompts": {"researcher": hashlib.sha256(INSTRUCTIONS.encode()).hexdigest()},
        "generated": None,
    }
    assert len(record["corpus"]["generations"]) == 5
    assert record["search"] != record["context"]


def test_frozen_legacy_authority_reads_tools_descriptions_and_contract_publication_stay_exact(
    setup, tmp_path, monkeypatch
):
    frozen = json.loads(
        (Path(__file__).parent / "fixtures/legacy-period-authority.json").read_bytes()
    )
    root = tmp_path / "frozen"
    for name, data in frozen.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.encode())
    for name, digest in json.loads(frozen["result.json"])["files"].items():
        assert hashlib.sha256(frozen[name].encode()).hexdigest() == digest
    setup = replace(
        setup,
        sources=root / "store/sources",
        datasets=root / "store/datasets",
        rules=root / "store/rules",
        catalog=Catalog.from_bytes(frozen["catalog.json"].encode()),
    )
    revision = parse_file(frozen["revision.json"].encode())
    assert file_bytes(revision) == frozen["revision.json"].encode()
    authority = resolve(revision, setup)
    assert authority.record() == frozen["authority.json"].encode()
    for context, name in (
        (authority.context, "context.json"),
        (authority.search.context, "search-context.json"),
    ):
        assert context.dataset_periods is None
        assert context.to_bytes() == frozen[name].encode()
        assert RunContext.from_bytes(frozen[name].encode()) == context
    supplied = datasets.read(setup.catalog, setup.datasets, "alfred.unrate", authority.context)
    assert supplied.record.to_bytes() == frozen["read.json"].encode()
    assert [row.key for row in supplied.rows] == ["1989-10-01"]
    vintage = (setup.datasets / "alfred.unrate" / f"{supplied.record.vintage}.json").read_bytes()
    assert supplied.record.replay(vintage) == supplied.rows
    for tool, record, material in (
        ("read_dataset", "tool-call.json", "tool-material.json"),
        ("describe_dataset", "description-call.json", "description-material.json"),
    ):
        supplied_call = call(authority, tool, dataset="alfred.unrate")
        assert supplied_call.to_bytes() == frozen[record].encode()
        assert supplied_call.material.text.encode() == frozen[material].encode()
        assert (
            authority.toolset.replay(frozen[record].encode()).to_bytes() == frozen[record].encode()
        )
    mapped = contract_sources(setup.sources, setup.catalog, ("alfred.unrate",))
    revised = child(
        revision,
        replace(
            revision.specification,
            sources=(
                *revision.specification.sources,
                *tuple(
                    replace(source, name=f"mapped-{index}") for index, source in enumerate(mapped)
                ),
            ),
        ),
    )
    updated = resolve(revised, setup)
    assert call(updated, "read_dataset", dataset="alfred.unrate").outcome.rows == ()
    with pytest.raises(ToolError):
        updated.toolset.replay(frozen["tool-call.json"].encode())
    assert resolve(revision, setup).record() == frozen["authority.json"].encode()
    assert (
        resolve(revision, setup).toolset.replay(frozen["tool-call.json"].encode()).to_bytes()
        == frozen["tool-call.json"].encode()
    )
    (root / "contracts").mkdir()
    (root / "contracts/alfred.unrate.json").write_text(frozen["contract.json"])
    monkeypatch.setattr(datasets.resources, "files", lambda _: root)
    republished = contract_sources(setup.sources, setup.catalog, ("alfred.unrate",))
    assert republished == tuple(
        source for source in revision.specification.sources if source.role == CONTRACT
    )


@pytest.fixture
def monthly_setup(setup):
    rows = [
        {
            "key": key,
            "revision": 1,
            "published_at": "1990-02-01T00:00:00Z",
            "arrived_at": "1990-02-01T00:00:00Z",
            "values": {"value": str(index)},
        }
        for index, key in enumerate(
            (
                "1989-07-01",
                "1989-08-01",
                "1989-09-01",
                "1989-10-01",
                "1989-11-01",
                "1989-12-01",
                "1990-01-01",
            )
        )
    ]
    rows.extend(
        [
            {
                **rows[3],
                "revision": 2,
                "published_at": "1994-01-01T00:00:00Z",
                "arrived_at": "1994-01-01T00:00:00Z",
            },
            {
                **rows[3],
                "revision": 3,
                "published_at": "2022-01-01T00:00:00Z",
                "arrived_at": "2022-01-01T00:00:00Z",
            },
        ]
    )
    rows[5]["arrived_at"] = "1995-01-01T00:00:00Z"
    data = canonical_bytes({"dataset": "alfred.unrate", "schema_version": 1, "rows": rows})
    vintage = datasets.save(setup.datasets, "alfred.unrate", data)
    catalog = Catalog(
        tuple(
            replace(event, vintage=vintage)
            if isinstance(event, Documented) and event.dataset == "alfred.unrate"
            else event
            for event in setup.catalog.events
        )
    )
    return replace(setup, catalog=catalog)


def monthly_permission(start, end):
    return DataPermission(
        "alfred.unrate",
        datetime.fromisoformat(start).replace(tzinfo=UTC),
        datetime.fromisoformat(end).replace(tzinfo=UTC),
    )


def test_mapped_union_selects_latest_available_rows_in_direct_tools_and_replay(monthly_setup):
    permissions = (
        monthly_permission("1989-08-01", "1989-10-01"),
        monthly_permission("1989-12-01", "1989-12-01"),
    )
    revision = pilot(monthly_setup, ("alfred.unrate",), data=permissions)
    authority = resolve(revision, monthly_setup)
    result = datasets.read(
        monthly_setup.catalog, monthly_setup.datasets, "alfred.unrate", authority.search.context
    )
    assert result.record.supplied == (
        ("1989-08-01", 1),
        ("1989-09-01", 1),
        ("1989-10-01", 2),
        ("1989-12-01", 1),
    )
    supplied_call = call(authority, "read_dataset", dataset="alfred.unrate")
    assert supplied_call.outcome == result
    assert (
        resolve(revision, monthly_setup).toolset.replay(supplied_call.to_bytes()).to_bytes()
        == supplied_call.to_bytes()
    )
    assert datasets.read(
        monthly_setup.catalog, monthly_setup.datasets, "alfred.unrate", authority.context
    ).record.supplied[2] == ("1989-10-01", 3)
    earlier = authority.context.narrow(cutoff=datetime(1994, 6, 1, tzinfo=UTC))
    assert (
        datasets.read(
            monthly_setup.catalog, monthly_setup.datasets, "alfred.unrate", earlier
        ).record.supplied
        == result.record.supplied[:-1]
    )
    original = (
        monthly_setup.datasets / "alfred.unrate" / f"{result.record.vintage}.json"
    ).read_bytes()
    assert datasets.ReadRecord.from_bytes(result.record.to_bytes()).replay(original) == result.rows
    changed = child(
        revision,
        replace(
            revision.specification,
            data=(monthly_permission("1989-08-02", "1989-10-01"), *permissions[1:]),
        ),
    )
    with pytest.raises(ToolError):
        resolve(changed, monthly_setup).toolset.replay(supplied_call.to_bytes())
    assert resolve(revision, monthly_setup).record() == authority.record()


@pytest.mark.parametrize(
    ("start", "end", "keys"),
    [
        ("1989-10-01", "1989-10-01", ("1989-10-01",)),
        ("1989-10-15", "1989-12-15", ("1989-11-01", "1989-12-01")),
        ("1989-09-30T23:59:59", "1989-10-01", ("1989-10-01",)),
        ("2000-01-01", "2000-01-01", ()),
    ],
)
def test_month_labels_use_closed_endpoints_and_keep_empty_datasets_visible(
    monthly_setup, start, end, keys
):
    authority = resolve(
        pilot(monthly_setup, ("alfred.unrate",), data=(monthly_permission(start, end),)),
        monthly_setup,
    )
    assert (
        tuple(
            row.key for row in call(authority, "read_dataset", dataset="alfred.unrate").outcome.rows
        )
        == keys
    )
    assert [
        item.dataset for item in datasets.listing(monthly_setup.catalog, authority.context).datasets
    ] == ["alfred.unrate"]
    assert datasets.describe(
        monthly_setup.catalog, monthly_setup.sources, "alfred.unrate", authority.context
    ).units


def test_mapped_authority_requires_complete_unambiguous_pins_and_deduplicates_aliases(setup):
    revision = pilot(setup)
    spec = revision.specification
    missing = tuple(
        source for source in spec.sources if source.name != "alfred.sahmrealtime.contract.0"
    )
    with pytest.raises(CampaignError, match="No period mapping"):
        resolve(child(revision, replace(spec, sources=missing)), setup)
    [source] = [source for source in spec.sources if source.name == "alfred.unrate.contract.0"]
    alias = replace(source, name="alias")
    assert (
        resolve(
            child(revision, replace(spec, sources=(*spec.sources, alias))), setup
        ).context.identity
        == resolve(revision, setup).context.identity
    )
    generation = open_generation(setup.sources, source.locator)
    different = register(
        setup.sources,
        name="competing",
        role=CONTRACT,
        data=(setup.sources / "blobs" / generation.occurrence.blob_digest).read_bytes(),
        declaration=replace(
            generation.occurrence.declaration, rights="Another supported declaration"
        ),
    )
    with pytest.raises(CampaignError, match="Competing period mappings"):
        resolve(child(revision, replace(spec, sources=(*spec.sources, different))), setup)
    unsupported = register(
        setup.sources,
        name="unsupported",
        role=CONTRACT,
        data=(setup.sources / "blobs" / generation.occurrence.blob_digest).read_bytes(),
        declaration=replace(
            generation.occurrence.declaration,
            revision=generation.occurrence.declaration.revision.replace(
                "closed-label/v1", "closed-label/v2"
            ),
        ),
    )
    with pytest.raises(CampaignError, match="invalid dataset period mapping"):
        resolve(child(revision, replace(spec, sources=(*spec.sources, unsupported))), setup)


def test_period_mapping_survives_a_cutoff_before_its_description_is_visible(monthly_setup):
    revision = pilot(
        monthly_setup, ("alfred.unrate",), data=(monthly_permission("1989-10-01", "1989-10-01"),)
    )
    original = next(
        source
        for source in revision.specification.sources
        if source.name == "alfred.unrate.contract.0"
    )
    generation = open_generation(monthly_setup.sources, original.locator)
    delayed = register(
        monthly_setup.sources,
        name=original.name,
        role=original.role,
        data=(monthly_setup.sources / "blobs" / generation.occurrence.blob_digest).read_bytes(),
        declaration=replace(
            generation.occurrence.declaration, source_available_at=datetime(2025, 1, 1, tzinfo=UTC)
        ),
    )
    revision = child(
        revision,
        replace(
            revision.specification,
            sources=tuple(
                delayed if source == original else source
                for source in revision.specification.sources
            ),
        ),
    )
    authority = resolve(revision, monthly_setup)
    assert (
        datasets.describe(
            monthly_setup.catalog, monthly_setup.sources, "alfred.unrate", authority.search.context
        ).units
        == ()
    )
    assert call(authority, "read_dataset", dataset="alfred.unrate").outcome.record.supplied == (
        ("1989-10-01", 2),
    )


@pytest.mark.parametrize(
    "key", ["1989-10-15", "1989-13-01", "19891001", "1989-10-01T00:00:00Z", "opaque"]
)
def test_mapped_reads_reject_invalid_keys_even_when_excluded_or_unavailable(monthly_setup, key):
    raw = {
        "dataset": "alfred.unrate",
        "schema_version": 1,
        "rows": [
            {
                "key": key,
                "revision": 1,
                "published_at": "2030-01-01T00:00:00Z",
                "arrived_at": "2030-01-01T00:00:00Z",
                "values": {"value": "1"},
            }
        ],
    }
    vintage = datasets.save(monthly_setup.datasets, "alfred.unrate", canonical_bytes(raw))
    setup = replace(
        monthly_setup,
        catalog=Catalog(
            tuple(
                replace(event, vintage=vintage)
                if isinstance(event, Documented) and event.dataset == "alfred.unrate"
                else event
                for event in monthly_setup.catalog.events
            )
        ),
    )
    authority = resolve(pilot(setup, ("alfred.unrate",)), setup)
    with pytest.raises(DatasetError, match="^Invalid dataset 'alfred.unrate'$"):
        datasets.read(setup.catalog, setup.datasets, "alfred.unrate", authority.context)
    assert call(authority, "read_dataset", dataset="alfred.payems").outcome == Refusal(
        "unknown_dataset"
    )


def test_monthly_signal_requires_explicit_warmup_and_keeps_the_observed_month(monthly_setup):
    from fractions import Fraction

    from quant_research.signals import Series, TrailingMean, decide

    program = TrailingMean(Series("alfred.unrate"), 3)
    first = pilot(
        monthly_setup, ("alfred.unrate",), data=(monthly_permission("1989-10-01", "1989-10-01"),)
    )
    without = decide(
        program,
        monthly_setup.catalog,
        monthly_setup.datasets,
        resolve(first, monthly_setup).context,
        (date(2020, 1, 1),),
    )[0]
    assert (without.observation, without.value) == (None, None)
    assert without.reads[0].supplied == (("1989-10-01", 2),)
    next_revision = child(
        first, replace(first.specification, data=(monthly_permission("1989-08-01", "1989-10-01"),))
    )
    assert next_revision.specification.evaluation == first.specification.evaluation
    with_warmup = decide(
        program,
        monthly_setup.catalog,
        monthly_setup.datasets,
        resolve(next_revision, monthly_setup).context,
        (date(2020, 1, 1),),
    )[0]
    assert (with_warmup.observation, with_warmup.value) == (date(1989, 10, 1), Fraction(2))
    assert with_warmup.reads[0].supplied == (
        ("1989-08-01", 1),
        ("1989-09-01", 1),
        ("1989-10-01", 2),
    )


@pytest.mark.parametrize("include_rules", (True, False))
def test_period_archive_retains_each_revision_context_and_actual_reads(
    monthly_setup, tmp_path, include_rules
):
    from quant_research import prospective, research
    from quant_research.packages import ArtifactInput, save_package

    vintage = monthly_setup.catalog.entry("alfred.unrate").documentation.vintage
    raw = json.loads((monthly_setup.datasets / "alfred.unrate" / f"{vintage}.json").read_bytes())
    raw["rows"].append({**raw["rows"][-1], "revision": 4, "published_at": "2027-01-01T00:00:00Z"})
    vintage = datasets.save(monthly_setup.datasets, "alfred.unrate", canonical_bytes(raw))
    monthly_setup = replace(
        monthly_setup,
        catalog=Catalog(
            tuple(
                replace(event, vintage=vintage)
                if isinstance(event, Documented) and event.dataset == "alfred.unrate"
                else event
                for event in monthly_setup.catalog.events
            )
        ),
    )
    archive = research.CampaignArchive(tmp_path / "archive", tmp_path / "calls")
    parent = pilot(
        monthly_setup, ("alfred.unrate",), data=(monthly_permission("1989-09-01", "1989-10-01"),)
    )
    child_revision = child(
        parent,
        replace(parent.specification, data=(monthly_permission("2000-01-01", "2000-01-01"),)),
    )
    for revision, expected in (
        (parent, (("1989-09-01", 1), ("1989-10-01", 3))),
        (child_revision, ()),
    ):
        authority = resolve(revision, monthly_setup)
        evidence = archive.pin(authority)
        original = datasets.read(
            monthly_setup.catalog, monthly_setup.datasets, "alfred.unrate", authority.context
        )
        contexts = [item for item in evidence.artifacts if item.role == "campaign-context"]
        assert RunContext.from_bytes(contexts[0].data) == authority.context
        [read] = [item for item in evidence.artifacts if item.role == "campaign-read"]
        assert read.data == original.record.to_bytes()
        artifacts = (
            *evidence.artifacts,
            ArtifactInput(
                "campaign-snapshot",
                canonical_bytes(
                    {"report": {"revision": revision.revision_id, "steps": [], "assessment": None}}
                ),
                "application/json",
                "campaign-snapshot",
            ),
            ArtifactInput(
                f"campaign-revision-{revision.revision_id}",
                file_bytes(revision),
                "application/json",
                "campaign-revision",
            ),
        )
        artifacts = tuple(
            item for item in artifacts if include_rules or item.role != "campaign-rules"
        )
        package = save_package(tmp_path / "packages", PackageRecord(None, (), ()), artifacts)
        procedure, reads = prospective._package_procedure(package)
        assert procedure.completion.selection == ("unfrozen" if include_rules else "incomplete")
        assert len(reads) == 1
        assert reads[0].record.supplied == expected
        assert reads[0].record.to_bytes() == original.record.to_bytes()
        assert len(json.loads(reads[0].vintage)["rows"]) == 10
        assert tuple(item.row for item in reads[0].rows()) == original.rows
        for missing in (
            ("campaign-read",),
            ("campaign-vintage",),
            ("campaign-read", "campaign-vintage"),
        ):
            damaged = save_package(
                tmp_path / "packages",
                package.record,
                tuple(item for item in artifacts if item.role not in missing),
            )
            with pytest.raises(
                prospective.ProspectiveError, match="missing its dataset read evidence"
            ):
                prospective._package_procedure(damaged)
        context_without_periods = json.loads(contexts[0].data)
        del context_without_periods["dataset_periods"]
        damaged = save_package(
            tmp_path / "packages",
            package.record,
            tuple(
                replace(item, data=canonical_bytes(context_without_periods))
                if item == contexts[0]
                else item
                for item in artifacts
            ),
        )
        with pytest.raises(prospective.ProspectiveError, match="pinned period authority"):
            prospective._package_procedure(damaged)
        if expected:
            for supplied in (
                (("1989-07-01", 1),),
                (("1989-09-01", 1), ("1989-10-01", 4)),
                (("1989-09-01", 1), ("1989-10-01", 1)),
                (),
                expected[:1],
                (*expected, expected[0]),
                tuple(reversed(expected)),
            ):
                damaged = save_package(
                    tmp_path / "packages",
                    package.record,
                    tuple(
                        replace(item, data=replace(original.record, supplied=supplied).to_bytes())
                        if item == read
                        else item
                        for item in artifacts
                    ),
                )
                with pytest.raises(
                    prospective.ProspectiveError, match="differs from its pinned authority"
                ):
                    prospective._package_procedure(damaged)
        for field, value in (
            ("dataset", "other"),
            ("vintage", "e" * 64),
            ("catalog", "f" * 64),
            ("cutoff", datetime(2020, 1, 1, tzinfo=UTC)),
            ("context", "c" * 64),
        ):
            changed_read = replace(original.record, **{field: value})
            damaged = save_package(
                tmp_path / "packages",
                package.record,
                tuple(
                    replace(item, data=changed_read.to_bytes()) if item == read else item
                    for item in artifacts
                ),
            )
            with pytest.raises(
                prospective.ProspectiveError, match="differs from its pinned authority"
            ):
                prospective._package_procedure(damaged)
    assert (
        archive.pin(resolve(parent, monthly_setup)).artifacts
        == archive.pin(resolve(parent, monthly_setup)).artifacts
    )
