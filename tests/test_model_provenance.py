from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from quant_research.model_provenance import (
    MODELS,
    UNKNOWN,
    CutoffCheck,
    ModelRecord,
    RoleAssignment,
    RoleVersion,
    dump_assignment,
    load_assignment,
    parse_model_record,
)
from quant_research.records import PackageError

ROOT = Path(__file__).resolve().parents[1]
MODEL = ModelRecord("provider-a", "model-one", "v2", date(2025, 6, 30))
BACKUP = ModelRecord("provider-b", "model-two", "v1", UNKNOWN)
NETWORK_PROBE = """
import sys
sys.path.append(sys.argv[1])
from datetime import date
from quant_research.model_provenance import CutoffCheck, ModelRecord, RoleAssignment, RoleVersion
model = ModelRecord("provider-a", "model-one", "v2", date(2025, 6, 30))
roles = RoleVersion("campaign-a", None, (RoleAssignment("proposer", model),), "Initial roles")
roles.revise(roles.assignments, "Recheck roles").version_id
check = CutoffCheck(model, date(2025, 7, 1), date(2025, 7, 31))
check.status, check.sentence
print(sorted({"socket", "ssl", "http"} & set(sys.modules)))
"""


@pytest.mark.smoke
def test_assignment_codec_keeps_the_historical_activity_bytes():
    expected = (
        '{"fallbacks": [{"knowledge_cutoff": null, "model": "model-two", '
        '"provider": "provider-b", "version": "unknown"}], "primary": '
        '{"knowledge_cutoff": "2025-06-30", "model": "model-one", '
        '"provider": "provider-a", "version": "v2"}, "role": "proposer"}'
    )
    assignment = RoleAssignment("proposer", MODEL, (replace(BACKUP, version="unknown"),))
    assert dump_assignment(assignment) == expected
    assert load_assignment(expected) == assignment


def test_unknown_cutoff_qualifies_every_period_and_cannot_be_omitted():
    model = ModelRecord("provider-a", "model-one", UNKNOWN, UNKNOWN)
    for year in (1990, 2100):
        check = CutoffCheck(model, date(year, 1, 1), date(year, 12, 31))
        assert (check.status, check.sentence) == (
            "cutoff_unknown",
            "The knowledge cutoff for an unknown version of model-one from provider-a is unknown, "
            f"so the model may already contain knowledge of the period from {year}-01-01 through "
            f"{year}-12-31 even when the platform withheld its results, and the historical result "
            "is qualified confirmation, not untouched evidence.",
        )
    with pytest.raises(TypeError):
        ModelRecord("provider-a", "model-one", "v2")


@pytest.mark.parametrize(
    ("start", "end", "status", "sentence"),
    [
        (
            date(2025, 7, 1),
            date(2025, 12, 31),
            "period_after_cutoff",
            "The period from 2025-07-01 through 2025-12-31 begins after 2025-06-30, the recorded "
            "knowledge cutoff for version v2 of model-one from provider-a.",
        ),
        (
            date(2025, 6, 30),
            date(2025, 12, 31),
            "period_before_or_overlapping_cutoff",
            "The period from 2025-06-30 through 2025-12-31 begins on or before 2025-06-30, the "
            "recorded knowledge cutoff for version v2 of model-one from provider-a, so the "
            "model may already contain knowledge of the period even when the platform withheld "
            "its results, and the historical result is qualified confirmation, not untouched "
            "evidence.",
        ),
        (
            date(2024, 1, 1),
            date(2024, 12, 31),
            "period_before_or_overlapping_cutoff",
            "The period from 2024-01-01 through 2024-12-31 begins on or before 2025-06-30, the "
            "recorded knowledge cutoff for version v2 of model-one from provider-a, so the "
            "model may already contain knowledge of the period even when the platform withheld "
            "its results, and the historical result is qualified confirmation, not untouched "
            "evidence.",
        ),
    ],
    ids=["day-after-cutoff", "starts-on-cutoff", "ends-before-cutoff"],
)
def test_only_a_period_starting_after_the_cutoff_is_unqualified(start, end, status, sentence):
    check = CutoffCheck(MODEL, start, end)
    assert (check.status, check.sentence) == (status, sentence)


@pytest.mark.smoke
def test_revisions_link_the_prior_version_and_leave_it_unchanged():
    assignments = (RoleAssignment("proposer", MODEL, (BACKUP,)), RoleAssignment("critic", BACKUP))
    prior = RoleVersion("campaign-a", None, assignments, "Initial roles")
    prior_id = prior.version_id
    upgrade = prior.revise(
        (RoleAssignment("proposer", replace(MODEL, version="v3"), (BACKUP,)), assignments[1]),
        "Upgrade model-one to v3",
    )
    fallback_change = prior.revise(
        (RoleAssignment("proposer", MODEL), assignments[1]), "Drop the proposer fallback"
    )
    for revision in (upgrade, fallback_change):
        assert (revision.campaign_id, revision.previous) == ("campaign-a", prior_id)
    assert len({prior_id, upgrade.version_id, fallback_change.version_id}) == 3
    assert prior == RoleVersion("campaign-a", None, assignments, "Initial roles")
    assert prior.version_id == prior_id


def test_version_ids_follow_content_and_keep_unknown_apart_from_text():
    def roles(model):
        return RoleVersion("campaign-a", None, (RoleAssignment("proposer", model),), "Roles")

    rebuilt = roles(ModelRecord("provider-a", "model-one", "v2", date(2025, 6, 30)))
    assert rebuilt.version_id == roles(MODEL).version_id
    assert rebuilt.revise(rebuilt.assignments, rebuilt.reason).version_id != rebuilt.version_id
    variants = (
        MODEL,
        replace(MODEL, knowledge_cutoff=UNKNOWN),
        replace(MODEL, version=UNKNOWN),
        replace(MODEL, version="unknown"),
    )
    assert len({roles(model).version_id for model in variants}) == len(variants)


def test_duplicate_roles_and_reversed_periods_are_rejected():
    critics = (RoleAssignment("critic", MODEL), RoleAssignment("critic", BACKUP))
    with pytest.raises(ValueError):
        RoleVersion("campaign-a", None, critics, "Two critics")
    with pytest.raises(ValueError):
        CutoffCheck(MODEL, date(2025, 3, 2), date(2025, 3, 1))
    single_day = CutoffCheck(MODEL, date(2025, 3, 1), date(2025, 3, 1))
    assert single_day.status == "period_before_or_overlapping_cutoff"


def test_role_records_are_inactive_data_that_load_no_network_modules(tmp_path):
    roles = RoleVersion("campaign-a", None, (RoleAssignment("proposer", MODEL),), "Initial roles")
    assert roles.activation == "inactive"
    assert roles.revise(roles.assignments, "Recheck roles").activation == "inactive"
    result = subprocess.run(
        [sys.executable, "-B", "-I", "-S", "-c", NETWORK_PROBE, str(ROOT / "src")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert (result.returncode, result.stdout) == (0, "[]\n"), result.stderr


CODEX = ModelRecord("codex-subscription", "gpt-6.1-sol", UNKNOWN, UNKNOWN)
CLAUDE = ModelRecord("claude-subscription", "claude-opus-5-5", UNKNOWN, UNKNOWN)


def test_an_inactive_version_keeps_the_id_it_had_before_roles_could_activate():
    roles = RoleVersion("campaign-a", None, (RoleAssignment("proposer", MODEL),), "Initial roles")
    assert roles.version_id == "e8fa4e6a5f82e2dff1ddee727c0df75e9d861cbfb5954840751e1fbbe7c72f8a"
    codex = replace(roles, assignments=(RoleAssignment("proposer", CODEX),))
    assert codex.version_id != replace(codex, activation="active").version_id


@pytest.mark.parametrize(
    "assignment",
    [RoleAssignment("proposer", MODEL), RoleAssignment("proposer", CODEX, (MODEL,))],
    ids=["unaccepted-primary", "unaccepted-fallback"],
)
def test_only_providers_that_passed_a_live_check_can_activate(assignment):
    roles = (RoleAssignment("critic", CODEX, (CLAUDE,)), assignment)
    assert RoleVersion("campaign-a", None, roles, "Inactive roles").activation == "inactive"
    with pytest.raises(ValueError, match=MODEL.provider):
        RoleVersion("campaign-a", None, roles, "Activate the roles", "active")
    active = RoleVersion("campaign-a", None, roles[:1], "Activate Codex and Claude", "active")
    assert active.activation == "active"
    revised = active.revise((RoleAssignment("critic", CLAUDE, (CODEX,)),), "Swap the models")
    assert (revised.previous, revised.activation) == (active.version_id, "active")
    with pytest.raises(ValueError, match=MODEL.provider):
        active.revise(roles, "Add an unaccepted role")


def test_each_model_is_keyed_by_its_provider_and_model():
    assert MODELS == {
        "codex-subscription/gpt-6.1-sol": CODEX,
        "claude-subscription/claude-opus-5-5": CLAUDE,
    }
    assert all(key == f"{model.provider}/{model.model}" for key, model in MODELS.items())
    with pytest.raises(TypeError):
        MODELS["provider-a/model-one"] = MODEL


RESTRICTED = "top decile sharpe"


@pytest.mark.parametrize("cutoff", [RESTRICTED, {"outcome": RESTRICTED}], ids=["text", "object"])
def test_cutoff_errors_keep_the_value_out_of_their_exception_chains(cutoff, reachable_text):
    raw = {
        "provider": "provider-a",
        "model": "model-one",
        "version": "v2",
        "knowledge_cutoff": cutoff,
    }
    with pytest.raises(PackageError, match="Invalid knowledge cutoff") as caught:
        parse_model_record(raw)
    assert RESTRICTED not in reachable_text(caught.value)
