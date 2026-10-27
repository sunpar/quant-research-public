from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from typing import get_args

from quant_research.evaluation import Evaluation, Interval, Invalid
from quant_research.packages import (
    Artifact,
    Coverage,
    ExecutionAttempt,
    ExposureEvent,
    Status,
    TrialRecord,
    VerifiedPackage,
)
from quant_research.reference import EVALUATION_ROLE, decode_evaluation

_MARKDOWN = re.compile(r"[\\`*_\[\]<>&|~$]")
CAMPAIGN_ROLE = "campaign-snapshot"

_COVERAGE: dict[Coverage, str] = {
    "unknown": "- Disclosure coverage: unknown. The package holds the trials and exposures its "
    "author supplied. It does not establish complete historical disclosure.",
    "declared_inputs": "- Disclosure coverage: declared inputs. The author states that the "
    "exposures cover everything it read from the inputs its trials declare. It does not "
    "establish complete historical disclosure.",
}


def _limits(coverage: Coverage) -> list[str]:
    return [
        "## Limits",
        "",
        "- Scope: development. The package records development research only.",
        _COVERAGE[coverage],
        "- Confirmation: unavailable. No confirmation outcome appears in this package or "
        "this report.",
        "- Integrity: checksums detect ordinary changes to the manifest and artifacts. They do not "
        "detect a coordinated rewrite of both content and checksums.",
        "- Ranking: none. Statuses, details, and limitations are supplied by the package author. "
        "This report does not rank trials or select an attempt.",
        "- Negative results: a rejected, null, or inconclusive outcome is a recorded research "
        "result.",
        "- Investment usefulness: not assessed. This report makes no claim about it.",
    ]


def single_line(value: str) -> str:
    """Return package text on one line with control and format characters escaped."""
    return "".join(
        char.encode("unicode_escape").decode()
        if unicodedata.category(char) in ("Cc", "Cf")
        else char
        for char in " ".join(value.splitlines())
    )


def _inline(value: str) -> str:
    return single_line(_MARKDOWN.sub(r"\\\g<0>", value))


def _named(digest: str, artifacts: tuple[Artifact, ...]) -> str:
    names = [f"`{item.name}`" for item in artifacts if item.digest == digest]
    return f"`{digest}` ({'artifact' if len(names) == 1 else 'artifacts'} {', '.join(names)})"


def _trial(number: int, trial: TrialRecord, package: VerifiedPackage) -> list[str]:
    spec = trial.specification
    environment = spec.environment
    artifacts = package.artifact_records
    selected = " (selected)" if trial.trial_id == package.record.trial_id else ""
    return [
        f"### Trial {number}: `{trial.trial_id}`{selected}",
        "",
        f"- Lineage: `{trial.lineage_id}`",
        f"- Specification ID: `{spec.spec_id}`",
        *(
            [
                line
                for link in trial.links
                for line in (
                    f"- Lineage link to trial `{link.trial_id}` with specification "
                    f"`{link.spec_id}`, relation: {_inline(link.relation)}",
                    f"- Link reason: {_inline(link.reason)}",
                )
            ]
            or ["- Lineage links: none"]
        ),
        f"- Question: {_inline(spec.question)}",
        f"- Signal profile: {_inline(spec.signal_profile)}",
        f"- Evaluation profile: {_inline(spec.evaluation_profile)}",
        f"- Data version: {_inline(spec.data_version)}",
        f"- Implementation version: {_inline(spec.implementation_version)}",
        *(
            [f"- Input digest: {_named(digest, artifacts)}" for digest in spec.input_digests]
            or ["- Input digests: none"]
        ),
        f"- Data contract digest: {_named(spec.data_contract_digest, artifacts)}",
        f"- Implementation digest: {_named(spec.implementation_digest, artifacts)}",
        f"- Python implementation: {_inline(environment.implementation)}",
        f"- Python version: {_inline(environment.python_version)}",
        f"- ABI: {_inline(environment.abi)}",
        f"- Platform: {_inline(environment.platform)}",
        f"- uv version: {_inline(environment.uv_version)}",
        f"- Lock digest: {_named(environment.lock_digest, artifacts)}",
        f"- Attempts: {len(trial.attempts)}",
    ]


def _interval(interval: Interval) -> str:
    return (
        f"estimate {interval.estimate!r}, standard error {interval.standard_error!r}, "
        f"lower {interval.lower!r}, upper {interval.upper!r}"
    )


def _evaluation(name: str, result: Evaluation | Invalid) -> list[str]:
    profile = result.profile
    lines = [
        f"- Evaluation procedure: {_inline(profile.version)} (artifact `{name}`)",
        f"- Evaluation parameters: minimum effect {profile.minimum_effect!r}, subperiods "
        f"{profile.subperiods}, alpha {profile.alpha!r}",
    ]
    if isinstance(result, Invalid):
        return [
            *lines,
            f"- Invalid input reason: {_inline(result.reason)}",
            f"- Invalid input detail: {_inline(result.detail)}",
        ]
    return [
        *lines,
        f"- Evaluation verdict: {_inline(result.verdict)} ({_inline(result.reason)})",
        f"- Trials counted: {result.trials}, interval confidence {result.confidence!r}",
        f"- Mean return: {_interval(result.mean)}",
        "- Incremental return over the baseline: "
        + ("none" if result.incremental is None else _interval(result.incremental)),
        *(
            f"- Subperiod {number} estimate: {estimate!r}"
            for number, estimate in enumerate(result.subperiods, 1)
        ),
    ]


def _attempt(attempt: ExecutionAttempt, package: VerifiedPackage) -> list[str]:
    outcome = attempt.outcome
    order = (
        "first attempt in run"
        if attempt.predecessor is None
        else f"retry of attempt `{attempt.predecessor}`"
    )
    results = ", ".join(f"`{name}`" for name in outcome.result_artifacts) or "none"
    roles = {item.name: item.role for item in package.artifact_records}
    return [
        f"#### Attempt `{attempt.id}`: {outcome.status}",
        "",
        f"- Run: `{attempt.run_id}`, {order}",
        f"- Started: `{attempt.started_at.isoformat()}`",
        f"- Finished: `{attempt.finished_at.isoformat()}`",
        f"- Detail: {_inline(outcome.detail)}",
        f"- Result artifacts: {results}",
        *(
            [f"- Limitation: {_inline(item)}" for item in outcome.limitations]
            or ["- Limitations: none recorded"]
        ),
        *(
            line
            for name in outcome.result_artifacts
            if roles[name] == EVALUATION_ROLE
            for line in _evaluation(name, decode_evaluation(package.read_artifact(name)))
        ),
    ]


def _exposure(event: ExposureEvent, artifacts: tuple[Artifact, ...]) -> list[str]:
    return [
        f"### Exposure `{event.id}`",
        "",
        f"- Kind: {event.kind}",
        f"- Actor: {_inline(event.actor)}",
        f"- At: `{event.at.isoformat()}`",
        f"- Dataset: {_inline(event.dataset_id)}",
        f"- Interval start: `{event.interval_start.isoformat()}`",
        f"- Interval end: `{event.interval_end.isoformat()}`",
        f"- Material digest: {_named(event.material_digest, artifacts)}",
        f"- Reason: {_inline(event.reason)}",
    ]


def _artifact(artifact: Artifact) -> list[str]:
    return [
        f"### Artifact `{artifact.name}`",
        "",
        f"- Role: {_inline(artifact.role)}",
        f"- Media type: {_inline(artifact.media_type)}",
        f"- Size in bytes: {artifact.size}",
        f"- Digest: `{artifact.digest}`",
    ]


def campaign_snapshot(package: VerifiedPackage) -> dict | None:
    """Read the campaign writer's frozen JSON evidence without importing its workflow runtime."""
    artifact = next((item for item in package.artifact_records if item.role == CAMPAIGN_ROLE), None)
    return None if artifact is None else json.loads(package.read_artifact(artifact.name))


def _campaign(snapshot: dict) -> list[str]:
    report = snapshot["report"]
    ending, assessment = report["ending"], report["assessment"]
    outcome = "unavailable after operational failure" if ending is None else ending["outcome"]
    reason = "unavailable" if ending is None else ending["reason"] or "none"
    lines = [
        "## Campaign",
        "",
        f"- Workflow: `{_inline(report['workflow_id'])}`, run `{_inline(report['run_id'])}`",
        f"- Revision: `{report['revision']}`",
        f"- Campaign question: {_inline(snapshot['question'])}",
        f"- Snapshot: {_inline(snapshot['key'])}",
        f"- Research ending: {_inline(outcome)}",
        f"- Ending reason: {_inline(reason)}",
        f"- Steps: {len(report['steps'])}",
        f"- Original numerical head: {snapshot['original_head'] or 'none'}",
        f"- Recovery snapshot head: {snapshot['recovered_head'] or 'none'}",
        f"- Human acceptance: {_inline(snapshot['acceptance'])}",
        "- Scope: provisional development evidence only; confirmation unavailable",
    ]
    if report["unrecorded"] is not None:
        lines.append(f"- Pending unrecorded candidate: `{report['unrecorded']['trial']}`")
    if assessment is not None:
        lines.extend(
            [
                f"- Original classification: {_inline(assessment['classification'])}",
                f"- Assessment detail: {_inline(assessment['detail'])}",
                f"- Independent critic: {str(assessment['independent']).lower()}",
            ]
        )
        for name in ("shortlist", "evidence", "conclusion", "critique"):
            material = assessment[name]
            lines.append(
                f"- Original {name}: "
                + ("unavailable" if material is None else _inline(json.loads(material)["text"]))
            )
    elif ending is not None and ending["final"] is not None:
        lines.append(f"- Original final text: {_inline(json.loads(ending['final'])['text'])}")
    lines.extend(
        ["", "### Recorded steering decisions", "", "Actors are recorded by the steering log.", ""]
    )
    events = [
        event for event in json.loads(report["steering"])["events"] if event["actor"] != "runner"
    ]
    lines.extend(
        f"- {_inline(event['action'])} by {_inline(event['actor'])}: {_inline(event['reason'])}"
        for event in events
    )
    decisions = snapshot["acceptance_decisions"]
    lines.extend(
        f"- Accepted attempt `{decision['attempt']}` by {_inline(decision['actor'])}: "
        f"{_inline(decision['reason'])}, at `{decision['at']}`"
        + (
            ""
            if decision["supersedes"] is None
            else f", supersedes attempt `{decision['supersedes']}`"
        )
        for decision in decisions
    )
    if not events and not decisions:
        lines.append("No steering decisions from actors other than runner were recorded.")
    lines.extend(["", "### Model cutoff checks", ""])
    lines.extend(
        f"- {_inline(check['period'])}: {_inline(check['sentence'])}"
        for check in snapshot["cutoff_checks"]
    )
    if not snapshot["cutoff_checks"]:
        lines.append("No model and frozen evaluation period pair was available.")
    return lines


def render_report(package: VerifiedPackage) -> str:
    """Render a verified development package as a deterministic Markdown report.

    Each attempt's pilot-evaluation result artifacts are read hash-verified and rendered as
    further bullets of its block, after its limitations. A package that selects no trial says so
    in its intro and question.

    Args:
        package (VerifiedPackage): Package returned by open_package.

    Returns:
        (str) Markdown ending in one newline, identical for every copy of the package.

    Raises:
        PackageError: If a pilot-evaluation artifact changed since opening or does not decode.
    """
    record = package.record
    if record.trial_id is None:
        selection = "The package selects no trial."
        question = "none, because the package selects no trial"
    else:
        selected = record.selected_trial
        selection = (
            f"The selected trial is `{selected.trial_id}` in lineage `{selected.lineage_id}`, "
            f"trial {record.trials.index(selected) + 1} of {len(record.trials)} "
            "in the supplied history."
        )
        question = _inline(selected.specification.question)
    counts = Counter(
        attempt.outcome.status for trial in record.trials for attempt in trial.attempts
    )
    blocks = [
        [
            "# Development research report",
            "",
            f"Rendered from verified package `{package.digest}`.",
            selection,
            "To recheck the package, run `quant-research verify <package-dir>`.",
        ],
        _limits(record.disclosure_coverage),
        ["## Question", "", f"- Question: {question}"],
        [
            "## Outcomes",
            "",
            "Counts cover every attempt in every supplied trial.",
            "",
            f"- Trials: {len(record.trials)}",
            f"- Attempts: {counts.total()}",
            *(f"- {status}: {counts[status]}" for status in get_args(Status.__value__)),
        ],
        ["## Trials"],
    ]
    if not record.trials:
        blocks.append(["No trials were supplied. This does not establish that no trial ran."])
    for number, trial in enumerate(record.trials, 1):
        blocks.append(_trial(number, trial, package))
        blocks.extend(_attempt(attempt, package) for attempt in trial.attempts)
    if snapshot := campaign_snapshot(package):
        blocks.append(_campaign(snapshot))
    blocks.append(["## Exposure history"])
    blocks.extend(
        [_exposure(event, package.artifact_records) for event in record.exposures]
        or [
            ["No exposure events were supplied. This does not establish that no exposure occurred."]
        ]
    )
    blocks.append(["## Artifacts", "", "The report lists each artifact without its contents."])
    blocks.extend(_artifact(artifact) for artifact in package.artifact_records)
    return "\n\n".join("\n".join(block) for block in blocks) + "\n"
