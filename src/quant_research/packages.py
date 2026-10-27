from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

from quant_research.records import (
    PackageError,
    canonical_bytes,
    json_default,
    parse_array,
    parse_digest,
    parse_fields,
    parse_name,
    parse_text,
    parse_time,
    parse_uuid,
)

type Status = Literal["successful", "rejected", "failed", "null", "inconclusive"]
type Coverage = Literal["unknown", "declared_inputs"]

_DECLARED: dict[str, Coverage] = {"unknown": "unknown", "declared_inputs": "declared_inputs"}
_COVERAGE: dict[int, dict[str, Coverage]] = {
    1: {"caller_declared": "unknown"},
    2: _DECLARED,
    3: _DECLARED,
}


@dataclass(frozen=True)
class ArtifactInput:
    name: str
    data: bytes
    media_type: str
    role: str


@dataclass(frozen=True)
class Artifact:
    name: str
    digest: str
    size: int
    media_type: str
    role: str


@dataclass(frozen=True)
class EnvironmentRecord:
    implementation: str
    python_version: str
    abi: str
    platform: str
    uv_version: str
    lock_digest: str


@dataclass(frozen=True)
class Specification:
    question: str
    signal_profile: str
    evaluation_profile: str
    data_version: str
    implementation_version: str
    input_digests: tuple[str, ...]
    data_contract_digest: str
    implementation_digest: str
    environment: EnvironmentRecord

    @property
    def spec_id(self) -> str:
        return hashlib.sha256(canonical_bytes(asdict(self))).hexdigest()


@dataclass(frozen=True)
class Outcome:
    status: Status
    detail: str
    result_artifacts: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionAttempt:
    id: UUID
    run_id: UUID
    predecessor: UUID | None
    started_at: datetime
    finished_at: datetime
    outcome: Outcome


@dataclass(frozen=True)
class LineageLink:
    trial_id: UUID
    spec_id: str
    relation: str
    reason: str


@dataclass(frozen=True)
class TrialRecord:
    trial_id: UUID
    lineage_id: UUID
    specification: Specification
    attempts: tuple[ExecutionAttempt, ...]
    links: tuple[LineageLink, ...] = ()


@dataclass(frozen=True)
class ExposureEvent:
    id: UUID
    actor: str
    at: datetime
    material_digest: str
    dataset_id: str
    interval_start: datetime
    interval_end: datetime
    kind: Literal["data", "result", "feedback"]
    reason: str


@dataclass(frozen=True)
class PackageRecord:
    """Retain supplied trials and exposures without asserting historical completeness.

    A package selects one supplied trial, or none (trial_id None) when its research ended
    without a selection; it may then supply no trials.
    Supply ancestral-first trials and predecessor-first attempts within each trial.
    Explicit predecessor links may preserve retry forks in a run.
    Coverage "declared_inputs" asserts the exposures cover everything the producer read
    from the inputs its trials declare; "unknown" makes no completeness claim.
    """

    trial_id: UUID | None
    trials: tuple[TrialRecord, ...]
    exposures: tuple[ExposureEvent, ...] = ()
    disclosure_coverage: Coverage = "unknown"
    scope: Literal["development"] = "development"
    confirmation: Literal["unavailable"] = "unavailable"

    @property
    def selected_trial(self) -> TrialRecord:
        """Return the selected trial, for an operation that needs one.

        Raises:
            PackageError: If the package selects no trial.
        """
        if self.trial_id is None:
            raise PackageError("The package selects no trial")
        return next(trial for trial in self.trials if trial.trial_id == self.trial_id)


@dataclass(frozen=True)
class VerifiedPackage:
    """Retain the opened identity and recheck the manifest and requested artifact bytes."""

    path: Path
    digest: str
    record: PackageRecord
    artifact_records: tuple[Artifact, ...]

    def read_artifact(self, name: str) -> bytes:
        """Return a named artifact after checking its bytes and the manifest.

        Args:
            name (str): Logical artifact name in this package.

        Returns:
            (bytes) The hash-verified artifact content.
        """
        artifact = next((item for item in self.artifact_records if item.name == name), None)
        if artifact is None:
            raise PackageError(f"Unknown artifact {name!r}")
        try:
            with _package_files(self.path) as (root, files):
                _, digest = _manifest_bytes(root)
                if digest != self.digest:
                    raise PackageError("Package manifest changed after opening")
                return _artifact_bytes(files, artifact)
        except OSError as error:
            raise PackageError(
                f"Cannot read package artifact {artifact.name!r}: {error}"
            ) from error

    def artifacts(self) -> tuple[ArtifactInput, ...]:
        """Return all named artifacts with freshly verified bytes."""
        try:
            with _package_files(self.path) as (root, files):
                _, digest = _manifest_bytes(root)
                if digest != self.digest:
                    raise PackageError("Package manifest changed after opening")
                return tuple(
                    ArtifactInput(
                        item.name, _artifact_bytes(files, item), item.media_type, item.role
                    )
                    for item in self.artifact_records
                )
        except OSError as error:
            raise PackageError(f"Cannot read package artifacts: {error}") from error


def _specification(raw: object) -> Specification:
    value = parse_fields(
        raw,
        "spec_id question signal_profile evaluation_profile data_version implementation_version "
        "input_digests data_contract_digest implementation_digest environment",
        "specification",
    )
    environment = parse_fields(
        value["environment"],
        "implementation python_version abi platform uv_version lock_digest",
        "environment",
    )
    if not re.fullmatch(
        r"\d+\.\d+\.\d+", parse_text(environment["python_version"], "Python version")
    ):
        raise PackageError("Python version must include an exact patch version")
    result = Specification(
        *(
            parse_text(value[key], key)
            for key in (
                "question",
                "signal_profile",
                "evaluation_profile",
                "data_version",
                "implementation_version",
            )
        ),
        tuple(
            parse_digest(item, "input digest")
            for item in parse_array(value["input_digests"], "input_digests")
        ),
        parse_digest(value["data_contract_digest"], "data_contract_digest"),
        parse_digest(value["implementation_digest"], "implementation_digest"),
        EnvironmentRecord(
            *(
                parse_text(environment[key], key)
                for key in ("implementation", "python_version", "abi", "platform", "uv_version")
            ),
            parse_digest(environment["lock_digest"], "lock_digest"),
        ),
    )
    if len(set(result.input_digests)) != len(result.input_digests):
        raise PackageError("Duplicate specification input digests")
    if parse_digest(value["spec_id"], "spec_id digest") != result.spec_id:
        raise PackageError("Specification ID does not match its content")
    return result


def parse_outcome(raw: object) -> Outcome:
    """Return the outcome that a package record stores as JSON."""
    value = parse_fields(raw, "status detail result_artifacts limitations", "outcome")
    if value["status"] not in ("successful", "rejected", "failed", "null", "inconclusive"):
        raise PackageError("Unsupported development outcome status")
    result = Outcome(
        value["status"],
        parse_text(value["detail"], "outcome detail"),
        tuple(
            parse_name(item, "result artifact")
            for item in parse_array(value["result_artifacts"], "result_artifacts")
        ),
        tuple(
            parse_text(item, "limitation")
            for item in parse_array(value["limitations"], "limitations")
        ),
    )
    if len(set(result.result_artifacts)) != len(result.result_artifacts):
        raise PackageError("Duplicate result artifact names")
    return result


def _attempt(raw: object) -> ExecutionAttempt:
    value = parse_fields(raw, "id run_id predecessor started_at finished_at outcome", "attempt")
    result = ExecutionAttempt(
        parse_uuid(value["id"], "attempt id UUID"),
        parse_uuid(value["run_id"], "run_id UUID"),
        parse_uuid(value["predecessor"], "predecessor UUID")
        if value["predecessor"] is not None
        else None,
        parse_time(value["started_at"], "started_at timestamp"),
        parse_time(value["finished_at"], "finished_at timestamp"),
        parse_outcome(value["outcome"]),
    )
    if result.finished_at < result.started_at:
        raise PackageError("Attempt finishes before it starts")
    return result


def _trial(raw: object) -> TrialRecord:
    value = parse_fields(raw, "trial_id lineage_id specification attempts links", "trial")
    links = []
    for raw_link in parse_array(value["links"], "links"):
        link = parse_fields(raw_link, "trial_id spec_id relation reason", "lineage link")
        links.append(
            LineageLink(
                parse_uuid(link["trial_id"], "link trial_id UUID"),
                parse_digest(link["spec_id"], "link spec_id digest"),
                parse_text(link["relation"], "link relation"),
                parse_text(link["reason"], "link reason"),
            )
        )
    return TrialRecord(
        parse_uuid(value["trial_id"], "trial_id UUID"),
        parse_uuid(value["lineage_id"], "lineage_id UUID"),
        _specification(value["specification"]),
        tuple(_attempt(item) for item in parse_array(value["attempts"], "attempts")),
        tuple(links),
    )


def parse_exposure(raw: object) -> ExposureEvent:
    """Return the exposure event that a package record stores as JSON."""
    value = parse_fields(
        raw,
        "id actor at material_digest dataset_id interval_start interval_end kind reason",
        "exposure",
    )
    if value["kind"] not in ("data", "result", "feedback"):
        raise PackageError("Unsupported exposure kind")
    result = ExposureEvent(
        parse_uuid(value["id"], "exposure id UUID"),
        parse_text(value["actor"], "actor"),
        parse_time(value["at"], "exposure timestamp"),
        parse_digest(value["material_digest"], "material_digest"),
        parse_text(value["dataset_id"], "dataset_id"),
        parse_time(value["interval_start"], "interval_start timestamp"),
        parse_time(value["interval_end"], "interval_end timestamp"),
        value["kind"],
        parse_text(value["reason"], "exposure reason"),
    )
    if result.interval_end < result.interval_start:
        raise PackageError("Exposure interval ends before it starts")
    return result


def _record(raw: object, version: int) -> PackageRecord:
    value = parse_fields(
        raw, "trial_id trials exposures disclosure_coverage scope confirmation", "package record"
    )
    for key, expected in (("scope", "development"), ("confirmation", "unavailable")):
        if value[key] != expected:
            raise PackageError(f"{key} must be {expected!r}")
    coverage = value["disclosure_coverage"]
    if not isinstance(coverage, str) or coverage not in _COVERAGE[version]:
        raise PackageError(f"Unsupported disclosure coverage for schema version {version}")
    if version != 3:
        trial_id = parse_uuid(value["trial_id"], "package trial_id UUID")
    elif value["trial_id"] is None:
        trial_id = None
    else:
        raise PackageError(
            "Schema version 3 records no selected trial, so its trial_id must be null"
        )
    return PackageRecord(
        trial_id,
        tuple(_trial(item) for item in parse_array(value["trials"], "trials")),
        tuple(parse_exposure(item) for item in parse_array(value["exposures"], "exposures")),
        _COVERAGE[version][coverage],
    )


def _record_data(record: PackageRecord) -> dict:
    value = asdict(record)
    for trial, encoded in zip(record.trials, value["trials"], strict=True):
        encoded["specification"]["spec_id"] = trial.specification.spec_id
    return value


def _validate_references(record: PackageRecord, artifacts: tuple[Artifact, ...]) -> None:
    identities = set()
    for kind, identifiers in (
        ("lineage", {trial.lineage_id for trial in record.trials}),
        ("trial", {trial.trial_id for trial in record.trials}),
        ("run", {attempt.run_id for trial in record.trials for attempt in trial.attempts}),
        ("attempt", {attempt.id for trial in record.trials for attempt in trial.attempts}),
        ("exposure", {event.id for event in record.exposures}),
    ):
        if identities & identifiers:
            raise PackageError(f"{kind} IDs must be distinct from other record kinds")
        identities.update(identifiers)
    trials = {}
    runs = {}
    attempts = {}
    lineages = set()
    artifact_names = {artifact.name for artifact in artifacts}
    required_digests = set()
    for trial in record.trials:
        if trial.trial_id in trials:
            raise PackageError(f"Duplicate or conflicting trial ID {trial.trial_id}")
        prior_trials = set()
        for link in trial.links:
            if link.trial_id not in trials or link.trial_id in prior_trials:
                raise PackageError("Lineage links must name distinct supplied prior trials")
            if link.spec_id != trials[link.trial_id].specification.spec_id:
                raise PackageError("Lineage link specification does not match its trial")
            prior_trials.add(link.trial_id)
        if trial.lineage_id in lineages and not any(
            trials[prior].lineage_id == trial.lineage_id for prior in prior_trials
        ):
            raise PackageError("A new trial in an existing lineage requires a prior trial link")
        spec = trial.specification
        required_digests.update(
            (
                *spec.input_digests,
                spec.data_contract_digest,
                spec.implementation_digest,
                spec.environment.lock_digest,
            )
        )
        for attempt in trial.attempts:
            if attempt.id in attempts:
                raise PackageError(f"Duplicate or conflicting attempt ID {attempt.id}")
            if attempt.predecessor is None:
                if attempt.run_id in runs:
                    raise PackageError(
                        "A repeat requires a new run ID; a retry needs a predecessor"
                    )
            else:
                prior = attempts.get(attempt.predecessor)
                if (
                    prior is None
                    or prior.run_id != attempt.run_id
                    or runs[attempt.run_id] != trial.trial_id
                ):
                    raise PackageError(
                        "Retry predecessor must be a supplied attempt in the same trial/run"
                    )
                if prior.finished_at > attempt.started_at:
                    raise PackageError("Retry starts before its predecessor finishes")
            if not set(attempt.outcome.result_artifacts) <= artifact_names:
                raise PackageError("Outcome references an unavailable result artifact")
            runs[attempt.run_id] = trial.trial_id
            attempts[attempt.id] = attempt
        trials[trial.trial_id] = trial
        lineages.add(trial.lineage_id)
    if record.trial_id is not None and record.trial_id not in trials:
        raise PackageError("Selected trial ID is not in supplied history")
    exposures = set()
    for exposure in record.exposures:
        if exposure.id in exposures:
            raise PackageError(f"Duplicate or conflicting exposure ID {exposure.id}")
        exposures.add(exposure.id)
        required_digests.add(exposure.material_digest)
    if not required_digests <= {artifact.digest for artifact in artifacts}:
        raise PackageError("Record references an unavailable artifact digest")


def _artifacts(raw: object) -> tuple[Artifact, ...]:
    artifacts = []
    names = set()
    for item in parse_array(raw, "artifacts"):
        value = parse_fields(item, "name digest size media_type role", "artifact")
        name = parse_name(value["name"])
        if name in names:
            raise PackageError(f"Duplicate or conflicting artifact name {name!r}")
        if type(value["size"]) is not int or value["size"] < 0:
            raise PackageError("Artifact size must be a nonnegative integer")
        artifacts.append(
            Artifact(
                name,
                parse_digest(value["digest"], "artifact digest"),
                value["size"],
                parse_text(value["media_type"], "media_type"),
                parse_text(value["role"], "role"),
            )
        )
        names.add(name)
    return tuple(artifacts)


@contextmanager
def _package_files(path: Path):
    root = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if set(os.listdir(root)) != {"manifest.json", "manifest.sha256", "artifacts"}:
            raise PackageError("Package must contain manifest.json, manifest.sha256 and artifacts")
        files = os.open("artifacts", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root)
        try:
            yield root, files
        finally:
            os.close(files)
    finally:
        os.close(root)


def _read_regular(directory: int, name: str) -> bytes:
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise PackageError(f"Package file {name!r} must be a regular file")
        return stream.read()


def _manifest_bytes(root: int) -> tuple[bytes, str]:
    content = _read_regular(root, "manifest.json")
    digest = hashlib.sha256(content).hexdigest()
    if _read_regular(root, "manifest.sha256") != f"{digest}\n".encode():
        raise PackageError("Manifest hash does not match manifest.sha256")
    return content, digest


def _artifact_bytes(files: int, artifact: Artifact) -> bytes:
    content = _read_regular(files, artifact.digest)
    if len(content) != artifact.size or hashlib.sha256(content).hexdigest() != artifact.digest:
        raise PackageError(f"Artifact {artifact.name!r} hash or size does not match")
    return content


def open_package(path: Path) -> VerifiedPackage:
    """Verify and open a self-contained development package directory.

    Args:
        path (Path): Package directory, which may have been copied or renamed.

    Returns:
        (VerifiedPackage) Parsed records and access to every verified artifact.
    """
    try:
        with _package_files(path) as (root, files):
            content, digest = _manifest_bytes(root)
            raw = json.loads(content.decode("utf-8"))
            value = parse_fields(raw, "schema_version record artifacts", "manifest")
            version = value["schema_version"]
            if type(version) is not int or version not in _COVERAGE:
                raise PackageError(f"Unsupported schema version {version!r}")
            if canonical_bytes(value) != content:
                raise PackageError("Manifest is not canonical UTF-8 JSON")
            record = _record(value["record"], version)
            artifacts = _artifacts(value["artifacts"])
            _validate_references(record, artifacts)
            if set(os.listdir(files)) != {artifact.digest for artifact in artifacts}:
                raise PackageError("Package artifact files do not match the manifest")
            for artifact in artifacts:
                _artifact_bytes(files, artifact)
        return VerifiedPackage(path.absolute(), digest, record, artifacts)
    except (OSError, ValueError, RecursionError) as error:
        raise PackageError(f"Cannot open package {path}: {error}") from error


def save_package(
    destination: Path, record: PackageRecord, artifacts: tuple[ArtifactInput, ...]
) -> VerifiedPackage:
    """Publish a complete hash-verified revision using a private staging directory.

    Args:
        destination (Path): Parent directory for revisions grouped by selected trial ID, or
            under `unselected` for a package that selects no trial.
        record (PackageRecord): Complete supplied trial and disclosure history.
        artifacts (tuple): All referenced bytes under distinct logical names.

    Returns:
        (VerifiedPackage) Published revision, or the verified identical existing revision.

    Raises:
        PackageError: If explicit record checks or filesystem operations fail.

    Notes:
        Use the declared Path, PackageRecord, and tuple types. Other Python exceptions
        propagate unchanged.
    """
    descriptions = []
    for item in artifacts:
        if not isinstance(item.data, bytes):
            raise PackageError("Artifact content must be bytes")
        descriptions.append(
            asdict(
                Artifact(
                    item.name,
                    hashlib.sha256(item.data).hexdigest(),
                    len(item.data),
                    item.media_type,
                    item.role,
                )
            )
        )
    manifest = json.loads(
        json.dumps(
            {
                "schema_version": 2 if record.trial_id is not None else 3,
                "record": _record_data(record),
                "artifacts": descriptions,
            },
            default=json_default,
            allow_nan=False,
        )
    )
    parsed = _record(manifest["record"], manifest["schema_version"])
    metadata = _artifacts(descriptions)
    _validate_references(parsed, metadata)
    content = canonical_bytes(manifest)
    digest = hashlib.sha256(content).hexdigest()
    try:
        destination = destination.absolute()
        if destination.is_symlink():
            raise PackageError("Package destination must not be a symlink")
        parent = destination / ("unselected" if parsed.trial_id is None else str(parsed.trial_id))
        parent.mkdir(parents=True, exist_ok=True)
        if parent.is_symlink():
            raise PackageError("Trial directory must not be a symlink")
        final = parent / digest
        if final.exists() or final.is_symlink():
            existing = open_package(final)
            if existing.digest != digest:
                raise PackageError("Existing revision has a different manifest")
            return existing
        with tempfile.TemporaryDirectory(prefix=".package-", dir=parent) as directory:
            stage = Path(directory)
            (stage / "artifacts").mkdir()
            blobs = {
                artifact.digest: item.data
                for item, artifact in zip(artifacts, metadata, strict=True)
            }
            for blob_digest, data in blobs.items():
                (stage / "artifacts" / blob_digest).write_bytes(data)
            (stage / "manifest.json").write_bytes(content)
            (stage / "manifest.sha256").write_text(f"{digest}\n", encoding="ascii")
            try:
                stage.rename(final)
            except OSError as error:
                if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
        saved = open_package(final)
        if saved.digest != digest:
            raise PackageError("Published revision has a different manifest")
        return saved
    except OSError as error:
        raise PackageError(f"Cannot save package: {error}") from error
