from __future__ import annotations

from dataclasses import replace

import pytest

from quant_research.workflows import (
    TEMPLATES,
    Branch,
    Checkpoint,
    Exchange,
    Role,
    Step,
    WorkflowError,
    WorkflowTemplate,
    differences,
    revise,
    validate,
)

BASE = WorkflowTemplate(
    name="local",
    version=1,
    previous=None,
    roles=(Role("worker", ("read-evidence",)),),
    branches=(
        Branch("lead", max_instances=1, max_iterations=1, steps=(Step("frame", "worker"),)),
        Branch(
            "search",
            max_instances=2,
            max_iterations=2,
            steps=(Step("gather", "worker", depends_on=("frame",)),),
        ),
    ),
    exchanges=(Exchange("lead", "search", "search scope"),),
    checkpoints=(Checkpoint("gather", "Review the gathered evidence"),),
)


@pytest.mark.smoke
@pytest.mark.parametrize("name", TEMPLATES)
def test_every_catalog_template_validates(name):
    validate(TEMPLATES[name])


def test_catalog_holds_the_baseline_and_four_patterns_keyed_by_name():
    assert set(TEMPLATES) == {
        "single-branch",
        "competing-hypotheses",
        "independent-replication",
        "focused-critiques",
        "parallel-evidence-searches",
    }
    assert all(template.name == name for name, template in TEMPLATES.items())


def test_baseline_runs_one_instance_and_replicators_never_share_with_each_other():
    baseline = TEMPLATES["single-branch"]
    assert [branch.max_instances for branch in baseline.branches] == [1]
    assert baseline.exchanges == ()
    routes = {
        name: {(exchange.source, exchange.target) for exchange in template.exchanges}
        for name, template in TEMPLATES.items()
    }
    assert ("replicator", "replicator") not in routes["independent-replication"]
    assert ("search", "search") in routes["parallel-evidence-searches"]


@pytest.mark.parametrize(
    "damage, token",
    [
        ("cycle", "frame -> check -> frame|check -> frame -> check"),
        ("self-dependency", "frame -> frame"),
        ("unknown-dependency", "ghost"),
        ("missing-instances", "max_instances"),
        ("zero-instances", "max_instances"),
        ("boolean-instances", "max_instances"),
        ("missing-iterations", "max_iterations"),
        ("negative-iterations", "max_iterations"),
        ("empty-role", "gather"),
        ("undeclared-role", "gather"),
        ("undeclared-target", "ghost"),
        ("undeclared-source", "ghost"),
        pytest.param("uncovered-dependency", "gather", marks=pytest.mark.smoke),
        ("reversed-exchange", "gather"),
        ("missing-checkpoint-step", "ghost"),
        ("duplicate-step", "frame"),
        ("unlinked-revision", "previous"),
        ("linked-first-version", "previous"),
        ("malformed-previous", "not-a-digest"),
        ("empty-branch", "idle"),
        ("no-branches", "branches"),
        ("invalid-name", "Local"),
        ("duplicate-tool", "read-evidence"),
        ("duplicate-resource", "paper-corpus"),
        ("blank-evidence", "evidence"),
        ("blank-reason", "reason"),
        ("list-roles", "roles"),
        ("bare-string-tools", "tools"),
        ("lone-surrogate", "identity"),
        ("duplicate-dependency", "step 'gather' dependency name 'frame'"),
        ("duplicate-exchange", "exchange from branch 'lead' to 'search'"),
        ("duplicate-checkpoint", "checkpoint step name 'gather'"),
        ("nul-evidence", "evidence"),
        ("nul-reason", "reason"),
        ("list-role", r"step 'gather' role name \['worker'\]"),
        ("list-dependency", r"step 'gather' dependency name \['frame'\]"),
        ("list-exchange-end", r"exchange branch name \['lead'\]"),
        ("list-checkpoint-step", r"checkpoint step name \['gather'\]"),
        ("string-role", "roles must hold Role entries, got str"),
        ("missing-role", "roles must hold Role entries, got NoneType"),
        ("string-branch", "branches must hold Branch entries, got str"),
        ("missing-step", "Branch 'search' steps must hold Step entries, got NoneType"),
        ("string-exchange", "exchanges must hold Exchange entries, got str"),
        ("missing-checkpoint", "checkpoints must hold Checkpoint entries, got NoneType"),
    ],
)
def test_validate_rejects_and_names_the_offending_element(damage, token):
    lead, search = BASE.branches
    frame, gather = lead.steps[0], search.steps[0]
    changes = {}
    match damage:
        case "cycle":
            check = Step("check", "worker", depends_on=("frame",))
            lead = replace(lead, steps=(replace(frame, depends_on=("check",)), check))
        case "self-dependency":
            lead = replace(lead, steps=(replace(frame, depends_on=("frame",)),))
        case "unknown-dependency":
            search = replace(search, steps=(replace(gather, depends_on=("ghost",)),))
        case "missing-instances":
            search = replace(search, max_instances=None)
        case "zero-instances":
            search = replace(search, max_instances=0)
        case "boolean-instances":
            search = replace(search, max_instances=True)
        case "missing-iterations":
            search = replace(search, max_iterations=None)
        case "negative-iterations":
            search = replace(search, max_iterations=-1)
        case "empty-role":
            search = replace(search, steps=(replace(gather, role=""),))
        case "undeclared-role":
            search = replace(search, steps=(replace(gather, role="ghost"),))
        case "undeclared-target":
            changes["exchanges"] = (*BASE.exchanges, Exchange("lead", "ghost", "search scope"))
        case "undeclared-source":
            changes["exchanges"] = (*BASE.exchanges, Exchange("ghost", "search", "search scope"))
        case "uncovered-dependency":
            changes["exchanges"] = ()
        case "reversed-exchange":
            changes["exchanges"] = (Exchange("search", "lead", "evidence summaries"),)
        case "missing-checkpoint-step":
            changes["checkpoints"] = (Checkpoint("ghost", "Review the gathered evidence"),)
        case "duplicate-step":
            search = replace(search, steps=(gather, Step("frame", "worker")))
        case "unlinked-revision":
            changes["version"] = 2
        case "linked-first-version":
            changes["previous"] = "0" * 64
        case "malformed-previous":
            changes |= {"version": 2, "previous": "not-a-digest"}
        case "empty-branch":
            idle = Branch("idle", max_instances=1, max_iterations=1, steps=())
            changes["branches"] = (lead, search, idle)
        case "no-branches":
            changes |= {"branches": (), "exchanges": (), "checkpoints": ()}
        case "invalid-name":
            changes["name"] = "Local"
        case "duplicate-tool":
            changes["roles"] = (Role("worker", ("read-evidence", "read-evidence")),)
        case "duplicate-resource":
            changes["shared_resources"] = ("paper-corpus", "paper-corpus")
        case "blank-evidence":
            changes["exchanges"] = (Exchange("lead", "search", " "),)
        case "blank-reason":
            changes["checkpoints"] = (Checkpoint("gather", ""),)
        case "list-roles":
            changes["roles"] = list(BASE.roles)
        case "bare-string-tools":
            changes["roles"] = (Role("worker", ("read-evidence")),)
        case "lone-surrogate":
            changes["exchanges"] = (Exchange("lead", "search", "\ud800"),)
        case "duplicate-dependency":
            search = replace(search, steps=(replace(gather, depends_on=("frame", "frame")),))
        case "duplicate-exchange":
            changes["exchanges"] = (*BASE.exchanges, Exchange("lead", "search", "search notes"))
        case "duplicate-checkpoint":
            changes["checkpoints"] = (*BASE.checkpoints, Checkpoint("gather", "Review it again"))
        case "nul-evidence":
            changes["exchanges"] = (Exchange("lead", "search", "search\x00scope"),)
        case "nul-reason":
            changes["checkpoints"] = (Checkpoint("gather", "Review\x00"),)
        case "list-role":
            search = replace(search, steps=(replace(gather, role=["worker"]),))
        case "list-dependency":
            search = replace(search, steps=(replace(gather, depends_on=(["frame"],)),))
        case "list-exchange-end":
            changes["exchanges"] = (Exchange(["lead"], "search", "search scope"),)
        case "list-checkpoint-step":
            changes["checkpoints"] = (Checkpoint(["gather"], "Review the gathered evidence"),)
        case "string-role":
            changes["roles"] = ("worker",)
        case "missing-role":
            changes["roles"] = (None,)
        case "string-branch":
            changes["branches"] = ("lead", "search")
        case "missing-step":
            search = replace(search, steps=(None,))
        case "string-exchange":
            changes["exchanges"] = ("lead",)
        case "missing-checkpoint":
            changes["checkpoints"] = (None,)
    with pytest.raises(WorkflowError, match=token):
        validate(replace(BASE, **{"branches": (lead, search), **changes}))


def test_revise_links_each_version_to_the_template_it_replaces():
    lead, search = BASE.branches
    revised = revise(BASE, branches=(lead, replace(search, max_instances=3)))
    assert (revised.version, revised.previous) == (2, BASE.template_id)
    assert revised.template_id != BASE.template_id
    assert differences(BASE, revised) == ("version", "previous", "branches")
    again = revise(revised, shared_resources=("paper-corpus",))
    assert (again.version, again.previous) == (3, revised.template_id)


def test_revise_requires_a_change_that_keeps_the_template_valid():
    with pytest.raises(WorkflowError):
        revise(BASE)
    with pytest.raises(WorkflowError):
        revise(BASE, name=BASE.name)
    with pytest.raises(WorkflowError, match="ghost"):
        revise(BASE, checkpoints=(Checkpoint("ghost", "Review the gathered evidence"),))
    with pytest.raises(WorkflowError, match="roles"):
        revise(BASE, roles=list(BASE.roles))
    with pytest.raises(WorkflowError, match="roles"):
        revise(BASE, roles=(role for role in BASE.roles))
    with pytest.raises(WorkflowError, match="roles"):
        revise(replace(BASE, roles=list(BASE.roles)), roles=BASE.roles)
    with pytest.raises(WorkflowError, match="identity"):
        revise(replace(BASE, exchanges=(Exchange("lead", "search", "\ud800"),)), name="other")
    with pytest.raises(WorkflowError, match="gather"):
        revise(BASE, checkpoints=(Checkpoint(["gather"], "Review the gathered evidence"),))

    class Subclassed(Role):
        pass

    with pytest.raises(WorkflowError, match="changes nothing"):
        revise(BASE, roles=tuple(Subclassed(role.name, role.tools) for role in BASE.roles))


@pytest.mark.parametrize("field", ["owner", "template_id", "version", "previous"])
def test_revise_rejects_fields_it_cannot_replace(field):
    with pytest.raises(WorkflowError, match=f"cannot replace {field}"):
        revise(BASE, **{field: 2})


def test_differences_shows_how_a_multi_branch_pattern_departs_from_the_baseline():
    changed = differences(TEMPLATES["single-branch"], TEMPLATES["competing-hypotheses"])
    assert {"branches", "exchanges"} <= set(changed)


def test_equal_content_gives_an_equal_template_id():
    rebuilt = WorkflowTemplate(
        name="single-branch",
        version=1,
        previous=None,
        roles=(
            Role("proposer", ("read-evidence",)),
            Role("implementer", ("compose-signal",)),
            Role("evaluator", ("run-evaluation",)),
            Role("critic", ("read-evidence",)),
            Role("interpreter", ("read-evidence", "record-finding")),
        ),
        branches=(
            Branch(
                "main",
                max_instances=1,
                max_iterations=5,
                steps=(
                    Step("propose", "proposer"),
                    Step("construct", "implementer", depends_on=("propose",)),
                    Step("evaluate", "evaluator", depends_on=("construct",)),
                    Step("critique", "critic", depends_on=("evaluate",)),
                    Step("interpret", "interpreter", depends_on=("critique",)),
                ),
            ),
        ),
        shared_resources=("development-data", "evaluation-budget"),
        checkpoints=(Checkpoint("interpret", "Review conclusions before revising or stopping"),),
    )
    assert rebuilt.template_id == TEMPLATES["single-branch"].template_id
