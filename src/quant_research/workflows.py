from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields, replace
from graphlib import CycleError, TopologicalSorter

from quant_research.records import PackageError, canonical_bytes


class WorkflowError(ValueError):
    """Report an invalid workflow template or a revision that changes nothing."""


@dataclass(frozen=True)
class Role:
    name: str
    tools: tuple[str, ...]


@dataclass(frozen=True)
class Step:
    name: str
    role: str
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class Branch:
    name: str
    max_instances: int
    max_iterations: int
    steps: tuple[Step, ...]


@dataclass(frozen=True)
class Exchange:
    source: str
    target: str
    evidence: str


@dataclass(frozen=True)
class Checkpoint:
    after: str
    reason: str


@dataclass(frozen=True)
class WorkflowTemplate:
    name: str
    version: int
    previous: str | None
    roles: tuple[Role, ...]
    branches: tuple[Branch, ...]
    shared_resources: tuple[str, ...] = ()
    exchanges: tuple[Exchange, ...] = ()
    checkpoints: tuple[Checkpoint, ...] = ()

    @property
    def template_id(self) -> str:
        return hashlib.sha256(canonical_bytes(asdict(self))).hexdigest()


def _name(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", value):
        raise WorkflowError(f"Invalid {label} name {value!r}")
    return value


def _unique(values: Iterable[object], label: str) -> set[str]:
    names = set()
    for value in values:
        name = _name(value, label)
        if name in names:
            raise WorkflowError(f"Duplicate {label} name {name!r}")
        names.add(name)
    return names


def _tuple(value: object, label: str, kind: type = object) -> tuple:
    if type(value) is not tuple:
        raise WorkflowError(f"{label} must be a tuple, got {type(value).__name__}")
    for item in value:
        if not isinstance(item, kind):
            raise WorkflowError(
                f"{label} must hold {kind.__name__} entries, got {type(item).__name__}"
            )
    return value


def _text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise WorkflowError(f"{label} must be nonempty text")


def validate(template: WorkflowTemplate) -> None:
    """Check a workflow template against every structural rule.

    Args:
        template (WorkflowTemplate): Template to check.

    Raises:
        WorkflowError: If the template breaks a rule; the reason names the offending element.
    """
    _name(template.name, "template")
    if type(template.version) is not int or template.version < 1:
        raise WorkflowError(f"Invalid template version {template.version!r}")
    if (template.previous is None) != (template.version == 1):
        raise WorkflowError(
            f"Version {template.version} conflicts with previous {template.previous!r}"
        )
    if template.previous is not None and not (
        isinstance(template.previous, str) and re.fullmatch(r"[0-9a-f]{64}", template.previous)
    ):
        raise WorkflowError(f"Invalid previous template ID {template.previous!r}")
    roles = _unique((role.name for role in _tuple(template.roles, "roles", Role)), "role")
    for role in template.roles:
        _unique(_tuple(role.tools, f"Role {role.name!r} tools"), f"role {role.name!r} tool")
    if not _tuple(template.branches, "branches", Branch):
        raise WorkflowError(f"Template {template.name!r} has no branches")
    branches = _unique((branch.name for branch in template.branches), "branch")
    for branch in template.branches:
        for limit in ("max_instances", "max_iterations"):
            value = getattr(branch, limit)
            if type(value) is not int or value < 1:
                raise WorkflowError(f"Branch {branch.name!r} {limit} must be a positive integer")
        if not _tuple(branch.steps, f"Branch {branch.name!r} steps", Step):
            raise WorkflowError(f"Branch {branch.name!r} has no steps")
    steps = [step for branch in template.branches for step in branch.steps]
    _unique((step.name for step in steps), "step")
    owner = {step.name: branch.name for branch in template.branches for step in branch.steps}
    for step in steps:
        if _name(step.role, f"step {step.name!r} role") not in roles:
            raise WorkflowError(f"Step {step.name!r} has undeclared role {step.role!r}")
        dependencies = _tuple(step.depends_on, f"Step {step.name!r} depends_on")
        _unique(dependencies, f"step {step.name!r} dependency")
        for dependency in dependencies:
            if dependency not in owner:
                raise WorkflowError(f"Step {step.name!r} depends on unknown step {dependency!r}")
    try:
        TopologicalSorter({step.name: step.depends_on for step in steps}).prepare()
    except CycleError as error:
        raise WorkflowError(f"Dependency cycle {' -> '.join(error.args[1])}") from error
    _unique(_tuple(template.shared_resources, "shared_resources"), "shared resource")
    routes = set()
    for exchange in _tuple(template.exchanges, "exchanges", Exchange):
        for end in (exchange.source, exchange.target):
            if _name(end, "exchange branch") not in branches:
                raise WorkflowError(f"Exchange names undeclared branch {end!r}")
        route = (exchange.source, exchange.target)
        if route in routes:
            raise WorkflowError(
                f"Duplicate exchange from branch {exchange.source!r} to {exchange.target!r}"
            )
        routes.add(route)
        _text(exchange.evidence, f"Exchange {exchange.source!r} to {exchange.target!r} evidence")
    for step in steps:
        for dependency in step.depends_on:
            source, target = owner[dependency], owner[step.name]
            if source != target and (source, target) not in routes:
                raise WorkflowError(
                    f"Step {step.name!r} needs an exchange from branch {source!r} to {target!r}"
                )
    checkpoints = _tuple(template.checkpoints, "checkpoints", Checkpoint)
    _unique((checkpoint.after for checkpoint in checkpoints), "checkpoint step")
    for checkpoint in checkpoints:
        if checkpoint.after not in owner:
            raise WorkflowError(f"Checkpoint after unknown step {checkpoint.after!r}")
        _text(checkpoint.reason, f"Checkpoint after {checkpoint.after!r} reason")
    try:
        canonical_bytes(asdict(template))
    except PackageError as error:
        raise WorkflowError(f"Template {template.name!r} has no canonical identity") from error


def revise(template: WorkflowTemplate, **changes: object) -> WorkflowTemplate:
    """Create the next version of a template from replacement field values.

    Args:
        template (WorkflowTemplate): Version to revise.
        **changes (object): Replacement values for fields other than version and previous.

    Returns:
        (WorkflowTemplate) The validated next version, linked to the ID of the one it replaces.

    Raises:
        WorkflowError: If either version breaks a rule or the changes leave it unchanged.
    """
    validate(template)
    revisable = {field.name for field in fields(WorkflowTemplate)} - {"version", "previous"}
    if unknown := changes.keys() - revisable:
        raise WorkflowError(
            f"Revision of {template.name!r} cannot replace {', '.join(sorted(unknown))}"
        )
    revised = replace(
        template, **changes, version=template.version + 1, previous=template.template_id
    )
    validate(revised)
    unchanged = replace(revised, version=template.version, previous=template.previous)
    if asdict(unchanged) == asdict(template):
        raise WorkflowError(f"Revision of {template.name!r} changes nothing")
    return revised


def differences(a: WorkflowTemplate, b: WorkflowTemplate) -> tuple[str, ...]:
    """Name the top-level fields whose values differ between two templates.

    Args:
        a (WorkflowTemplate): One template.
        b (WorkflowTemplate): The template to compare it with.

    Returns:
        (tuple[str, ...]) Differing field names in declaration order.
    """
    return tuple(
        field.name
        for field in fields(WorkflowTemplate)
        if getattr(a, field.name) != getattr(b, field.name)
    )


TEMPLATES: dict[str, WorkflowTemplate] = {
    template.name: template
    for template in (
        WorkflowTemplate(
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
            checkpoints=(
                Checkpoint("interpret", "Review conclusions before revising or stopping"),
            ),
        ),
        WorkflowTemplate(
            name="competing-hypotheses",
            version=1,
            previous=None,
            roles=(
                Role("lead", ("read-evidence", "record-finding")),
                Role("proposer", ("read-evidence",)),
                Role("implementer", ("compose-signal",)),
                Role("evaluator", ("run-evaluation",)),
            ),
            branches=(
                Branch(
                    "lead",
                    max_instances=1,
                    max_iterations=1,
                    steps=(
                        Step("frame", "lead"),
                        Step("compare", "lead", depends_on=("evaluate",)),
                    ),
                ),
                Branch(
                    "hypothesis",
                    max_instances=4,
                    max_iterations=3,
                    steps=(
                        Step("propose", "proposer", depends_on=("frame",)),
                        Step("construct", "implementer", depends_on=("propose",)),
                        Step("evaluate", "evaluator", depends_on=("construct",)),
                    ),
                ),
            ),
            shared_resources=("development-data", "evaluation-budget"),
            exchanges=(
                Exchange("lead", "hypothesis", "research framing"),
                Exchange("hypothesis", "lead", "evaluation results"),
            ),
            checkpoints=(
                Checkpoint("frame", "Approve the framing before hypotheses branch"),
                Checkpoint("compare", "Review the comparison of hypotheses"),
            ),
        ),
        WorkflowTemplate(
            name="independent-replication",
            version=1,
            previous=None,
            roles=(
                Role("lead", ("read-evidence", "record-finding")),
                Role("replicator", ("compose-signal", "run-evaluation")),
            ),
            branches=(
                Branch(
                    "lead",
                    max_instances=1,
                    max_iterations=1,
                    steps=(
                        Step("specify", "lead"),
                        Step("reconcile", "lead", depends_on=("run",)),
                    ),
                ),
                Branch(
                    "replicator",
                    max_instances=2,
                    max_iterations=2,
                    steps=(
                        Step("implement", "replicator", depends_on=("specify",)),
                        Step("run", "replicator", depends_on=("implement",)),
                    ),
                ),
            ),
            shared_resources=("development-data", "evaluation-budget"),
            exchanges=(
                Exchange("lead", "replicator", "frozen specification"),
                Exchange("replicator", "lead", "replication results"),
            ),
            checkpoints=(Checkpoint("reconcile", "Review agreement between replications"),),
        ),
        WorkflowTemplate(
            name="focused-critiques",
            version=1,
            previous=None,
            roles=(
                Role("proposer", ("read-evidence",)),
                Role("implementer", ("compose-signal",)),
                Role("evaluator", ("run-evaluation",)),
                Role("critic", ("read-evidence",)),
            ),
            branches=(
                Branch(
                    "author",
                    max_instances=1,
                    max_iterations=3,
                    steps=(
                        Step("propose", "proposer"),
                        Step("construct", "implementer", depends_on=("propose",)),
                        Step("evaluate", "evaluator", depends_on=("construct",)),
                        Step("revise", "implementer", depends_on=("critique",)),
                    ),
                ),
                Branch(
                    "critic",
                    max_instances=3,
                    max_iterations=3,
                    steps=(Step("critique", "critic", depends_on=("evaluate",)),),
                ),
            ),
            shared_resources=("development-data", "evaluation-budget"),
            exchanges=(
                Exchange("author", "critic", "candidate and evaluation evidence"),
                Exchange("critic", "author", "critiques"),
            ),
            checkpoints=(Checkpoint("critique", "Adjudicate critiques before revision"),),
        ),
        WorkflowTemplate(
            name="parallel-evidence-searches",
            version=1,
            previous=None,
            roles=(
                Role("lead", ("read-evidence", "record-finding")),
                Role("searcher", ("search-corpus", "read-evidence")),
            ),
            branches=(
                Branch(
                    "lead",
                    max_instances=1,
                    max_iterations=1,
                    steps=(
                        Step("scope", "lead"),
                        Step("synthesize", "lead", depends_on=("gather",)),
                    ),
                ),
                Branch(
                    "search",
                    max_instances=3,
                    max_iterations=2,
                    steps=(Step("gather", "searcher", depends_on=("scope",)),),
                ),
            ),
            shared_resources=("paper-corpus", "code-index", "data-catalog"),
            exchanges=(
                Exchange("lead", "search", "search scope"),
                Exchange("search", "search", "discovered sources"),
                Exchange("search", "lead", "evidence summaries"),
            ),
            checkpoints=(Checkpoint("synthesize", "Review the synthesized evidence"),),
        ),
    )
}
