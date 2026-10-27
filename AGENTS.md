# Working in quant-research

## Start with poteto mode and the project record

When changing code, fixtures, or documents, or creating or updating a PR, read [poteto mode](.agents/pstack/skills/poteto-mode/SKILL.md) in full, open the matching playbook in [`playbooks/`](.agents/pstack/skills/poteto-mode/playbooks), and copy its steps verbatim into the task list before the first edit. Follow narrower human instructions when they override a skill.

Read [the project state](docs/project-state.md) before choosing or planning work. It links the [architecture](docs/architecture.md), [decisions](docs/decisions.md), [open questions](docs/open-questions.md), [to dos](docs/todos.md), [master plan](docs/master-plan.md), and [initial PR plan](docs/initial-prs.md). Keep one authoritative home for each fact and link to it instead of copying. Skip both reads for questions, reviews, and setup tasks that need neither.

## Keep changes narrow

Do what was asked. Search with `rg` before creating helpers. Prefer deleting broken or duplicate logic, then replacing it, before adding another layer. State the motivation for any defensive behavior and obtain human approval.

Before choosing an approach, and after two failed attempts at the same problem, use the installed `codex-advisor` or `fable-advisor` for a second opinion. Read the `python-guidelines` skill before Python changes. Do not create unrelated documentation.

The repository tooling in `tools/repo.py` is finished. Change it only to fix a failure that blocks research work or to make a workflow change a human asks for, not to cover another boundary case.

## Test what you changed

While iterating, run only the tests that cover the change:

```sh
PYTHONPATH=src uv run python -m pytest tests/<file>.py -k <name>
```

The smoke suite runs on its own. The commit hook checks the staged tree, the merge-commit hook checks a merge result, and a rebase checks the rebased head. Do not run `make verify` or `make coverage`. The full suite runs once, inside `make merge`, on the exact commit that lands, and `make merge` posts its `Passed` line to the PR. A PR whose diff against main touches only `docs/**/*.md`, top-level Markdown files, or `.claude/` runs the smoke suite there instead. When an earlier full run passed the same gate inputs, the output also shows a `Skipped tests/test_repo_gates.py` line ([D072](docs/decisions.md#d072-skip-gate-tests-whose-exact-inputs-already-passed)).

## Merge

- `make merge PR=<n>` is the only way to merge. It waits for a lock that every worktree of this clone shares, merges newer main in, runs the suite, and merges the commit that passed ([D074](docs/decisions.md#d074-run-the-full-suite-only-inside-make-merge))
- Run `make merge` in the background and watch it; failing tests print as they fail. Any fix needs a new `make merge`
- When several PRs are open, give each a fixed D number above main's highest up front; keep numeric order, gaps are fine
- Several sessions share this machine; stop only your own `make merge` run, never `pkill` by pattern

## Update the record when facts change

When a PR changes capabilities, decisions, open questions, or next steps, update the affected page in the same PR. Distinguish proposed, implemented, and verified behavior. Never mark real-framework acceptance complete using synthetic fixtures. Do not leave a decision or prerequisite only in chat or an agent's memory.

## Tools and conventions

Keep pstack preferences in [.agents/pstack-models.md](.agents/pstack-models.md) for Codex and [.claude/rules/pstack-models.md](.claude/rules/pstack-models.md) for Claude Code. Do not read or write a home-directory fallback for this project. Keep the plugin disables in `.claude/settings.json`; pstack loads from `.agents/pstack`, so leave the pstack plugin off.

Use `uv` for Python, `pathlib` for paths, and `gh` for GitHub. PR titles start with a capitalized verb and have no type prefix. Self-assign created PRs. Create pending reviews only. Verify external citations against their actual publications before adding or changing claims.
