import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# A change under .claude/ merges on the smoke suite alone, so the hook's tests belong to it.
pytestmark = pytest.mark.smoke


def run_hook(command: str) -> subprocess.CompletedProcess:
    (entry,) = json.loads((ROOT / ".claude/settings.json").read_text())["hooks"]["PreToolUse"]
    assert entry["matcher"] == "Bash"
    (hook,) = entry["hooks"]
    return subprocess.run(
        hook["command"],
        shell=True,
        input=json.dumps(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Bash",
                "tool_input": {"command": command, "description": "Run a command"},
            }
        ),
        env={"PATH": os.environ["PATH"], "CLAUDE_PROJECT_DIR": str(ROOT)},
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "command",
    [
        "gh pr merge 12 --merge",
        "make verify TARGET=HEAD",
        "FOO=1 make coverage",
        "cd /tmp && make -j2 verify",
        "make smoke-index; make verify",
        "(make verify)",
        "result=$(make verify)",
        "echo ok | gh pr merge 3",
        "true\nmake verify",
        "make verify 2>&1 &",
    ],
)
def test_hook_blocks_direct_merges_and_full_runs(command):
    result = run_hook(command)
    assert result.returncode == 2, result.stderr
    assert "make merge PR=<n>" in result.stderr
    assert "pytest tests/<file>.py -k <name>" in result.stderr


@pytest.mark.parametrize(
    "command",
    [
        "make merge PR=12",
        "make smoke-index",
        "grep 'make verify' AGENTS.md",
        "git commit -m 'Explain why make verify moved'",
        'git commit -m "Drop make verify\nand gh pr merge"',
        "gh pr view 12 --json state",
        "PYTHONPATH=src uv run python -m pytest tests/test_repo_gates.py -k merge",
        "echo done # make verify",
        "cat <<'EOF'\nit's here\nmake verify\nEOF",
    ],
)
def test_hook_allows_everything_else(command):
    result = run_hook(command)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
