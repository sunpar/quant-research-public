import contextlib
import io
import json
import sys
from importlib.metadata import distribution
from pathlib import Path

import quant_research.packages as packages
from quant_research.__main__ import main
from quant_research.report import render_report

wheel = Path(sys.argv[1])
assert packages.__file__.startswith(str(wheel) + "/")
entry = next(
    item
    for item in distribution("quant-research").entry_points
    if item.group == "console_scripts" and item.name == "quant-research"
)
assert entry.value == "quant_research.__main__:main"
saved = packages.open_package(Path("package"))
assert "- Disclosure coverage: unknown." in render_report(saved)
artifacts = saved.artifacts()
assert artifacts
for artifact in artifacts:
    assert saved.read_artifact(artifact.name) == artifact.data
copied = packages.save_package(Path("copies"), saved.record, artifacts)
assert packages.save_package(Path("copies"), copied.record, artifacts).digest == copied.digest
assert packages.open_package(copied.path).record == saved.record
print("Wheel import and portable save/open consumer passed")

sessions = {"s0": "2026-01-05T16:00:00Z", "s1": "2026-01-06T16:00:00Z"}
levels = {"A": ("11", "10"), "B": ("10", "10"), "C": ("9", "10")}
labels = {"A": "-1", "B": "0", "C": "1"}
fixture = {
    "contract": {
        "version": 1,
        "designation": "synthetic",
        "dataset_id": "wheel-consumer",
        "sessions": [{"id": key, "occurred_at": at} for key, at in sessions.items()],
        "level_unit": "synthetic_level",
        "label_unit": "synthetic_target",
        "evaluation_cutoff": "2026-01-07T18:00:00Z",
        "signal_profile": "level-difference/v1",
        "evaluation_profile": "descriptive-pearson/v1",
        "decimal_precision": 34,
    },
    "entities": list(levels),
    "memberships": [
        {
            "entity_id": entity,
            "first_session": "s0",
            "last_session": "s1",
            "known_at": "2026-01-04T12:00:00Z",
            "included": True,
        }
        for entity in levels
    ],
    "observations": [
        {
            "entity_id": entity,
            "session_id": session,
            "revision": 1,
            "published_at": sessions[session],
            "arrived_at": sessions[session],
            "level": level,
            "unit": "synthetic_level",
        }
        for entity, values in levels.items()
        for session, level in zip(sessions, values, strict=True)
    ],
    "decisions": [
        {"entity_id": entity, "session_id": "s1", "at": "2026-01-06T17:00:00Z"} for entity in levels
    ],
    "labels": [
        {
            "entity_id": entity,
            "decision_at": "2026-01-06T17:00:00Z",
            "available_at": "2026-01-07T17:00:00Z",
            "value": value,
            "unit": "synthetic_target",
        }
        for entity, value in labels.items()
    ],
}
Path("fixture.json").write_text(json.dumps(fixture))
Path("consumer.lock").write_text("wheel consumer lock\n")
sys.argv = [
    "quant-research",
    "replay",
    "fixture.json",
    "--output",
    "replays",
    "--lock",
    "consumer.lock",
    "--new",
    "Check the synthetic wheel diagnostic",
]
with contextlib.redirect_stdout(io.StringIO()) as stdout:
    main()
receipt = json.loads(stdout.getvalue())
replayed = packages.open_package(Path(receipt["package_path"]))
assert replayed.digest == receipt["package_digest"]
assert receipt["status"] == "successful"
outputs = {
    item.role: json.loads(replayed.read_artifact(item.name))
    for item in replayed.artifact_records
    if item.name in receipt["result_artifacts"]
}
assert [row["value"] for row in outputs["signals"]] == ["-1", "0", "1"]
assert outputs["diagnostic"]["coefficient"] == "1"
implementation = next(item for item in replayed.artifacts() if item.role == "implementation")
replay_source = json.loads(implementation.data)["sources"]["quant_research/replay.py"]
assert "def replay_fixture(" in replay_source
print("Wheel causal replay CLI consumer passed")
