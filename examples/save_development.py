"""Save a hand-authored synthetic example with no computed research conclusion."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import sysconfig
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from quant_research.packages import (
    ArtifactInput,
    EnvironmentRecord,
    ExecutionAttempt,
    Outcome,
    PackageRecord,
    Specification,
    TrialRecord,
    save_package,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    artifacts = (
        ArtifactInput("input", b"entity,value\nsynthetic-a,1\n", "text/csv", "synthetic input"),
        ArtifactInput(
            "contract",
            b"Hand-authored synthetic development example, version 1.\n",
            "text/plain",
            "data contract",
        ),
        ArtifactInput(
            "implementation", Path(__file__).read_bytes(), "text/x-python", "authoring code"
        ),
        ArtifactInput(
            "uv_lock",
            args.lock.read_bytes(),
            "text/plain",
            "authoring environment lock",
        ),
        ArtifactInput(
            "result", b"No research computation was performed.\n", "text/plain", "result"
        ),
    )
    digests = {item.name: hashlib.sha256(item.data).hexdigest() for item in artifacts}
    specification = Specification(
        question="Demonstrate a portable synthetic development record",
        signal_profile="hand-authored/v1",
        evaluation_profile="unavailable/v1",
        data_version="synthetic-example/v1",
        implementation_version="authoring-example/v1",
        input_digests=(digests["input"],),
        data_contract_digest=digests["contract"],
        implementation_digest=digests["implementation"],
        environment=EnvironmentRecord(
            sys.implementation.name,
            ".".join(map(str, sys.version_info[:3])),
            sysconfig.get_config_var("SOABI"),
            sysconfig.get_platform(),
            subprocess.check_output(["uv", "--version"], text=True).strip(),
            digests["uv_lock"],
        ),
    )
    trial_id = uuid4()
    authored_at = datetime.now(UTC)
    record = PackageRecord(
        trial_id,
        (
            TrialRecord(
                trial_id,
                uuid4(),
                specification,
                (
                    ExecutionAttempt(
                        uuid4(),
                        uuid4(),
                        None,
                        authored_at,
                        authored_at,
                        Outcome(
                            "inconclusive",
                            "Hand-authored example without a research computation",
                            ("result",),
                            ("Synthetic data only", "No scientific conclusion"),
                        ),
                    ),
                ),
            ),
        ),
    )
    saved = save_package(args.destination, record, artifacts)
    print(saved.path)


if __name__ == "__main__":
    main()
