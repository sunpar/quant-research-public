import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

state_file = Path(os.environ["FAKE_GH_STATE"])
state = json.loads(state_file.read_text())
arguments = sys.argv[1:]
host = os.environ.get("GH_HOST", "github.com")
if "--hostname" in arguments:
    host = arguments[arguments.index("--hostname") + 1]
elif "--repo" in arguments:
    repository = arguments[arguments.index("--repo") + 1]
    if repository.count("/") == 2:
        host = repository.split("/", 1)[0]
elif arguments[:2] == ["repo", "view"] and arguments[2].count("/") == 2:
    host = arguments[2].split("/", 1)[0]
entry = {"args": arguments, "host": host}
if arguments[:2] == ["pr", "create"]:
    entry["body"] = Path(arguments[arguments.index("--body-file") + 1]).read_text()
with state_file.with_suffix(".calls").open("a") as stream:
    stream.write(json.dumps(entry) + "\n")
if state.get("enforce_host") and host != "github.com":
    sys.exit(f"Wrong GitHub host {host}")


def git(*args):
    return subprocess.check_output(["git", "--git-dir", state["remote"], *args], text=True).strip()


action = arguments[:2]
if action == ["auth", "status"]:
    stage = "auth"
elif action == ["repo", "view"]:
    stage = "repository"
elif arguments[0] == "api":
    if arguments[1].endswith("/pulls"):
        stage = "pr-list"
        assert arguments[arguments.index("--method") + 1] == "GET"
        assert "state=open" in arguments
        assert "head=example:feature" in arguments
    else:
        branch = unquote(arguments[1].split("/branches/", 1)[1])
        stage = "base" if branch == "main" else "remote-head"
else:
    stage = "-".join(action)

if state.get("fail") == stage:
    sys.exit(f"Injected {stage} failure")

if stage in state.get("outputs", {}):
    print(state["outputs"][stage])
    sys.exit()

if stage == "repository":
    print(
        json.dumps(
            {
                "nameWithOwner": state.get("repository", "example/research"),
                "defaultBranchRef": {"name": "main"},
            }
        )
    )
elif stage in ("base", "remote-head"):
    if stage == "remote-head" and state.get("mode") == "move-before-create":
        git("update-ref", "refs/heads/feature", state["other"])
    print(
        json.dumps(
            {
                "name": branch,
                "commit": {"sha": git("rev-parse", "--verify", f"refs/heads/{branch}")},
            }
        )
    )
elif stage == "pr-list":
    if state.get("mode") == "move-local":
        subprocess.run(["git", "update-ref", "refs/heads/feature", state["other"]], check=True)
    if state.get("mode") == "mutate-metadata":
        Path(state["title"]).write_text("Changed title")
        Path(state["body"]).write_text("Changed body")
    if state.get("mode") == "move-pushurl":
        subprocess.run(
            ["git", "remote", "set-url", "--push", "origin", state["push_url"]], check=True
        )
    entries = (
        [
            {
                "html_url": "https://github.com/example/research/pull/9",
                "base": {"ref": state.get("existing_base", "main")},
                "head": {
                    "ref": "feature",
                    "repo": {"full_name": state.get("existing_head", "example/research")},
                },
            }
        ]
        if state.get("existing")
        else []
    )
    if state.get("created"):
        entries.append(state["created"])
    if state.get("create_attempts") and state.get("recovery_failure"):
        sys.exit("Injected recovery failure")
    if state.get("create_attempts") and "recovery_output" in state:
        print(state["recovery_output"])
        sys.exit()
    pages = state.get("pages", [entries])
    if "--paginate" not in arguments:
        pages = pages[:1]
    print(json.dumps(pages if "--slurp" in arguments else pages[0]))
elif stage == "pr-create":
    state["create_attempts"] = state.get("create_attempts", 0) + 1
    if state.get("mode") == "move-after-create":
        git("update-ref", "refs/heads/feature", state["other"])
    url = "https://github.com/example/research/pull/10"
    if state.get("create_persist", True):
        state["created"] = {
            "html_url": url,
            "head": {
                "ref": arguments[arguments.index("--head") + 1],
                "sha": git("rev-parse", "refs/heads/feature"),
                "repo": {"full_name": "example/research"},
            },
            "base": {"ref": arguments[arguments.index("--base") + 1]},
        }
    state_file.write_text(json.dumps(state))
    print(state.get("create_output", url))
    if "--assignee" in arguments and state.get("fail") == "pr-edit":
        sys.exit("Injected pr-edit failure after creation")
    if state.get("create_status"):
        sys.exit("Injected create failure after publication")
elif stage == "pr-view" and "pull" in state:
    pull = state["pull"] | {
        "headRefOid": git("rev-parse", f"refs/heads/{state['pull']['headRefName']}")
    }
    print(
        json.dumps(
            {field: pull[field] for field in arguments[arguments.index("--json") + 1].split(",")}
        )
    )
elif stage == "pr-view":
    created = state["created"]
    print(
        json.dumps(
            {
                "url": created["html_url"],
                "headRefOid": created["head"]["sha"],
                "headRefName": created["head"]["ref"],
                "headRepository": {"nameWithOwner": created["head"]["repo"]["full_name"]},
                "baseRefName": created["base"]["ref"],
            }
        )
    )
elif stage == "pr-edit":
    assert arguments[2] == state["created"]["html_url"]
    assert arguments[arguments.index("--add-assignee") + 1] == "@me"
    state["assigned"] = True
    state_file.write_text(json.dumps(state))
elif stage == "pr-comment":
    assert arguments[2] == state["pull"]["url"]
elif stage == "pr-merge":
    pull = state["pull"]
    assert arguments[2] == pull["url"]
    assert "--merge" in arguments
    head = git("rev-parse", f"refs/heads/{pull['headRefName']}")
    if arguments[arguments.index("--match-head-commit") + 1] != head:
        sys.exit("Head branch was modified")
    base = git("rev-parse", f"refs/heads/{pull['baseRefName']}")
    merge = git(
        "-c",
        "user.name=Fake GitHub",
        "-c",
        "user.email=github@example.invalid",
        "commit-tree",
        git("merge-tree", "--write-tree", base, head),
        "-p",
        base,
        "-p",
        head,
        "-m",
        f"Merge pull request #{pull['number']}",
    )
    git("update-ref", f"refs/heads/{pull['baseRefName']}", merge, base)
    pull |= {"state": "MERGED", "mergeCommit": {"oid": merge}}
    state_file.write_text(json.dumps(state))
