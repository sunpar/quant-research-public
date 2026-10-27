# Build the local autonomous research pilot

Plan dated 22 October 2026, against `origin/main` at `728702c`. Return to [project state](project-state.md) for the index.

This plan takes the repository from the selected [Phase 0 local pilot](phase0-pilot.md) to a bounded autonomous research session on this Mac. The researcher's Claude Code and Codex subscriptions supply the inference, as tool-disabled completions behind the provider gateway ([D051](decisions.md#d051-own-the-research-loop-and-call-subscriptions-as-tool-disabled-completions)). The platform owns the loop, the tools, the stopping rule, and the record. The plan has 20 pull requests on the critical path, P01 to P20, and two optional paper-case PRs, P21 and P22.

The plan reaches a scoped local autonomous session, in the way [D050](decisions.md#d050-run-the-phase-0-pilot-locally-on-public-us-macro-data) scopes Phase 1. It skips the Phase 2 research console, keeps campaign logs in Temporal history and the final package instead of SurrealDB, and leaves the Phase 3 paper output on the optional track. It does not complete the master plan's Phase 4 gate. That gate runs the single-branch procedure across a held-out capability-evaluation set, and a held-out set needs answer keys from someone outside development, which a one-person project cannot supply. Phase 4 stays open with that comparison as a named prerequisite, unless the researcher records an approved change to the gate.

P06 and P07 ran the first live checks of subscription behavior, and both recorded a go ([D057](decisions.md#d057-call-claude-code-as-a-tool-disabled-completion-behind-a-local-first-turn-gate), [D058](decisions.md#d058-call-codex-through-a-fresh-app-server-in-an-isolated-home)). Statistical claims from the autonomous search stay provisional development evidence, because all of the pilot's history is development data.

## How to read this

One box is one unit of work. Every box names the evidence that checks it. Check a box only when its evidence exists: a file, a log line, a test run, or a SHA. The body is a how-to. The appendices explain and record.

Each PR follows [AGENTS.md](../AGENTS.md). Run targeted tests while iterating, and merge with `make merge PR=<n>`, which runs the full suite once on the commit that lands. The **Verify, live** block lists commands that the PR's owner runs locally against the real data, servers, or subscriptions, each with its pass condition. This plan does not use the ten cloud-VM lanes of the pstack plan template, because the surfaces here are a CLI and a local Temporal workflow, and the repository's verification rules govern. Every subscription call in a live block is opt-in and capped.

Tests alone are not sufficient verification. A PR is verified only when its unit and live boxes are checked, and its perf box when it has one.

## Program checklist

### Hold the human gates

- [ ] P04 stops twice. The researcher accepts the evaluation profile and its frozen parameters before the real returns are evaluated. Then the researcher reviews the reconciliation and evaluation evidence and accepts or rejects the scoped Phase 1. Acceptance closes [Q003](open-questions.md#q003-what-independent-baseline-defines-agreement). An agent made both calls under the user's autonomy grant. The researcher accepted the file and the scoped Phase 1 on 23 October 2026, after the run and the merge, so neither stop was met as written.
- [ ] P06 and P07 stop at merge-ready. The researcher reviews the isolation evidence before any repository content goes to a personal subscription. Neither gate passed before its merge. The researcher approved P06's isolation after its merge, on 23 October 2026 by accepting the coordinating agent's recommendation ([D057](decisions.md#d057-call-claude-code-as-a-tool-disabled-completion-behind-a-local-first-turn-gate)).
- [ ] P14 asks the researcher to choose the search universe before it merges. This gate did not pass before the merge. An agent chose the universe, and the researcher approved it after the merge, on 23 October 2026.
- [ ] P15 asks the researcher to accept the repeated-search method before it merges. An agent decided the method under the user's autonomy grant on 23 October 2026, per [D065](decisions.md#d065-run-experiments-inside-the-loop-under-a-frozen-search-method), and the researcher has not reviewed it.
- [ ] P20 is the first autonomous session. The researcher reviews its package and report before the record calls the scoped local session complete. Phase 4 itself stays open until a held-out capability comparison exists.
- [ ] No PR marks Phase 0, Phase 1, or Phase 4 complete without the evidence its master-plan gate names.

### Follow the dependency graph

- [ ] Wave 1 starts from `main` in parallel. It holds P01 and P05.
- [ ] Wave 2 holds P02, P03, and P08 after P01, and P06 and P07 after P05.
- [ ] Wave 3 holds P04 after P02 and P03, P10 after P05 and P08, P11 after a go from P06 or P07, and P14 after P02. A no-go from both stops the program for a replan.
- [ ] Wave 4 holds P09 after P04 and P05, and P12 after P10 and P11.
- [ ] Wave 5 holds P13 after P12, and P15 after P04, P12, and P14.
- [ ] Wave 6 holds P16 after P15.
- [x] Wave 7 completes P17 after P09 and P16, per [D067](decisions.md#d067-archive-each-research-ending-as-a-complete-campaign-snapshot).
- [x] Wave 8 completed P18 and P19 after P17 in [PR 157](https://github.com/sunpar/quant-research/pull/157) and [PR 158](https://github.com/sunpar/quant-research/pull/158), with their live checks passed before P20.
- [x] Wave 9 ran P20 after P13, P18, and P19 merged, per the [live record](phase0-pilot.md#review-the-first-autonomous-local-sessions). Researcher review remains pending.
- [ ] The paper track runs off the critical path. P21 follows P01, and P22 follows P21.

### Hold the file boundaries

- [ ] In wave 2, P02 alone edits `reconcile.py` and `__main__.py`, P03 alone edits `pyproject.toml` and `uv.lock`, and P08 alone edits `agent_tools.py`, `datasets.py`, `retrieval.py`, and `sources.py`.
- [ ] In wave 3, P10 alone edits `providers.py`, and P11 alone edits `campaigns.py`, `agent_tools.py`, and `datasets.py`.
- [ ] In wave 4, P09 owns `execution.py`, and P12 adds `research.py` without editing `execution.py`.
- [ ] Every PR that changes a rule appends its decision to `docs/decisions.md` under the number this plan assigns. Resolve an append conflict by keeping both entries in numeric order.
- [ ] Before each PR opens, check `main` and the open PRs for its decision number. When this plan was written, main's highest was D051. This plan assigns D052 to D071 in PR order, and decisions outside this plan take numbers above D071.

### Apply the PR mechanics to every PR

- [ ] Write the PR title as a capitalized verb with no type prefix, and assign it to `@me`.
- [ ] Commit with a typed subject, because the commit hook requires one.
- [ ] Edit `docs/*.md` through a script, not the Edit tool, because the formatter hook rewrites paired asterisks.
- [ ] Update the affected record pages in the same PR: project state, architecture, decisions, open questions, and to dos.
- [ ] Triage every Codex GitHub review finding before merge.
- [ ] Run `git merge-base --is-ancestor origin/main HEAD` right before merge, then merge with `gh pr merge --merge --match-head-commit`.

## Ingest the pilot's public data with every vintage kept (P01)

**Depends on.** None. **Decision.** D052.

**Files.**

- [x] Create `src/quant_research/public_data.py`, `tests/test_public_data.py`, and recorded API responses under `tests/fixtures/public_data/`.
- [x] Edit `src/quant_research/datasets.py`, `catalog.py`, `__main__.py`, and their tests.

**Build.**

- [x] Add provider approvals to catalog documentation, so P08 can carry them into tool results.
- [x] Store each converted dataset file under its vintage digest, and let a read record name the vintage it read. Today the store keeps one file per dataset, so a later download breaks replay of every earlier record.
- [x] Add `data pull` and `data ingest`. Download ALFRED `UNRATE` vintages, FRED `SAHMREALTIME`, and the Ken French monthly 3-factor file. Read the FRED API key from the environment and never write it. Keep each download's original bytes under its SHA-256, with the retrieval date.
- [x] Convert each series to the D045 row format under names such as `alfred.unrate`. Store values as decimal strings and missing values as `null`. Number ALFRED revisions per observation month in `realtime_start` order.
- [x] Set `published_at` and `arrived_at` from `realtime_start` under a convention the decision records, because a download time would hide every row from every historical cutoff. Give each Ken French month an availability time at or after that month's end under a recorded convention, because the file has no per-row dates and P15's search cutoff must hide later returns.
- [x] Add `data validate`, which records the researcher's manual validation attestation, and `data approve`, which approves a validated dataset for agent use. `data ingest` leaves a dataset documented only. Download digests alone do not validate a dataset.

**You see.**

- [x] `quant-research data ingest` prints each dataset's name, row count, vintage digest, and catalog status `documented`. After `data validate` and `data approve`, the status is `approved`.

**Verify, unit.**

- [x] `tests/test_public_data.py` converts the recorded responses, rejects a malformed response, reads at a 1990 cutoff only the vintages released by then, and replays an old read record after a newer vintage is stored. Run `PYTHONPATH=src uv run python -m pytest tests/test_public_data.py tests/test_datasets.py tests/test_catalog.py`.

**Verify, live.**

- [x] With `FRED_API_KEY` set, run `quant-research data pull` and `data ingest`, then repeat both. Pass when the second run leaves every digest unchanged and an agent read of each dataset is refused as unapproved.
- [x] Run `data validate` and `data approve` for each dataset. Pass when an agent read then succeeds.

**Verify, perf.** None.

**Review gate.** None. P01 is not review-gated beyond its validation attestation.

## Recompute the real-time Sahm series and reconcile it with FRED (P02)

**Depends on.** P01. **Decision.** D053.

**Files.**

- [x] Create `src/quant_research/signals.py` and `tests/test_signals.py`.
- [x] Edit `src/quant_research/reconcile.py` and `__main__.py`, and their tests.

**Build.**

- [x] Define the first approved signal operators: a dataset series, a trailing mean, a trailing minimum over prior months, subtraction, and a threshold. Express the Sahm indicator as a program in these operators, so the pipeline that Phase 1 accepts is the one the autonomous loop later searches.
- [x] Evaluate a program causally. The decision for month m reads the vintage available at the end of month m-1 and uses the latest computable observation. It records the observation month and the release date. Choose between narrowing the run context per decision and a read that returns one vintage.
- [x] Compute in `Fraction`. Find the missing-month convention that reproduces FRED's November and December 2025 values, and record it in the pilot brief.
- [x] Add a series profile to `reconcile.py`. Add `quant-research reconcile series`, which compares the recomputed series with the pinned `SAHMREALTIME` over the declared window and lists each excluded month with its reason.
- [ ] Let `reconcile.py` choose its recorded metrics and recomputation by profile. The package path's choice waits for P04, which first records a series in a package ([D053](decisions.md#d053-evaluate-signal-programs-per-decision-and-reconcile-the-sahm-series-with-fred)).

**You see.**

- [x] `quant-research reconcile series` prints the window, the number of matching months, and the excluded months with reasons.

**Verify, unit.**

- [x] `tests/test_signals.py` checks the operators against hand-computed values, shows that appending a later vintage leaves every earlier decision unchanged, and covers the missing-month convention. Run `PYTHONPATH=src uv run python -m pytest tests/test_signals.py tests/test_reconcile.py`.

**Verify, live.**

- [x] Run `quant-research reconcile series` over the P01 data and attach its output to the PR. Pass when no month in the window has an unexplained mismatch.

**Verify, perf.**

- [x] Metric. Wall time to recompute and reconcile the full series. Rule. Fail above 60 seconds on this Mac.

**Review gate.** None. P02 is not review-gated.

## Backtest allocations with two independent engines (P03)

**Depends on.** P01, for the live check. **Decision.** D054.

**Files.**

- [x] Create `src/quant_research/backtest.py` and `tests/test_backtest.py`.
- [x] Edit `pyproject.toml` and `uv.lock` to add vectorbt 1.2.0.

**Build.**

- [x] Define one backtest interface with two implementations, because the pilot substitutes one engine for the other. The plain implementation computes `RF + w × (Mkt-RF)` and the switch costs in `Fraction` and is the reference. The vectorbt implementation is the stand-in for the firm backtester.
- [x] Charge 10 basis points per allocation switch in the month the new allocation starts, including the last month.
- [x] Import vectorbt lazily, because the CLI tests run `python -I -S`.
- [x] If vectorbt does not resolve on Python 3.12, name a replacement independent engine in the PR, or leave Phase 1 blocked until the researcher approves a scope change. The plain engine alone does not meet D050's independent recomputation. vectorbt 1.2.0 resolves on Python 3.12, so no replacement is needed ([D054](decisions.md#d054-backtest-allocations-with-a-plain-reference-and-a-vectorbt-engine)).

**You see.**

- [x] Both engines return the same monthly returns for the hand-calculated cases, within the declared tolerance.

**Verify, unit.**

- [x] `tests/test_backtest.py` checks both engines against hand-calculated cash flows for a first month that carries no initial fee, a switch in a middle month, and a switch in the last month. Run `PYTHONPATH=src uv run python -m pytest tests/test_backtest.py`.

**Verify, live.**

- [x] Run both engines over the full Ken French history from P01 with a fixed allocation path. Pass when the largest monthly difference is within the declared tolerance.

**Verify, perf.**

- [x] Metric. The plain engine's and vectorbt's full-history run times, with vectorbt's import time recorded. Rule. Fail above 2 seconds for the plain engine or 10 seconds for a vectorbt run.

**Review gate.** None. P03 is not review-gated.

## Run the reference experiment as a scoped Phase 1 (P04)

**Depends on.** P02 and P03. **Decision.** D055.

**Files.**

- [x] Create `src/quant_research/reference.py` and `tests/test_reference.py`.
- [x] Edit `src/quant_research/report.py` and `__main__.py`, and their tests.
- [x] Edit `docs/phase0-pilot.md`, `docs/open-questions.md`, and `docs/todos.md` with the evidence.

**Build.**

- [x] Freeze the evaluation profile's parameters, the subperiod count, and the buy-and-hold baseline in a file. Commit `411fabc` froze `reference.json` before any real return was evaluated.
- [ ] Stop for the researcher's acceptance of that file before the real returns are evaluated. An agent made this call under the user's autonomy grant on 22 October 2026. The researcher accepted the unchanged file on 23 October 2026, after seeing the results, so this stop was not met as written.
- [x] Read the approved datasets through D045 with a trusted run context, run the Sahm program, allocate, backtest, and evaluate.
- [x] Save a package whose input digests are the dataset vintage digests, so [invalidation](decisions.md#d046-suspend-records-whose-support-an-invalidation-removes) can find it. Write the evaluation parameters into the trial's specification.
- [x] Add the two flawed variants, the current-vintage read and the fixed two-month lag, and a controlled null fixture with a known answer.
- [x] Show the verdict, intervals, subperiods, and parameters in `report`. Add `quant-research reference run`.

**You see.**

- [x] `quant-research reference run` writes a package that `verify` accepts, and `report` shows its verdict, intervals, subperiods, and parameters.

**Verify, unit.**

- [x] `tests/test_reference.py` shows that both flawed variants fail their checks, that the null fixture gives a null record, and that a fresh run with the same inputs and specification gives identical numerical artifacts. Re-exporting one record gives identical bytes. Run `PYTHONPATH=src uv run python -m pytest tests/test_reference.py tests/test_report.py`.

**Verify, live.**

- [x] Run the reference experiment, the reconciliation, and the report on the P01 data, and attach the report. Pass when the package verifies, the reconciliation passes, both flawed variants fail, and the null fixture records null.

**Verify, perf.** None.

**Review gate.** The researcher reviews twice.

- [ ] Before the real run, the researcher accepts the frozen evaluation file. An agent decided it under the user's autonomy grant before the run. The researcher accepted it on 23 October 2026, after seeing the results, so this gate was not met as written, per [D055](decisions.md#d055-run-the-reference-experiment-through-one-audited-pipeline-against-a-frozen-experiment-file).
- [ ] After the run, post the report, the reconciliation output, and the flawed and null results in the PR. Stop at merge-ready. Merge after the researcher accepts the scoped Phase 1, and record that acceptance in the pilot brief. An agent accepted the result under the user's autonomy grant and merged before the researcher's review. The researcher accepted the scoped Phase 1 on 23 October 2026, after the merge, and the pilot brief records that acceptance.

## Make the workflow the single accounting owner with durable call receipts (P05)

**Depends on.** None. **Decision.** D056.

**Files.**

- [x] Edit `src/quant_research/accounting.py`, `execution.py`, `providers.py`, `campaigns.py`, and their tests.

**Build.**

- [x] Keep the workflow as the only writer of the ledger, as [D042](decisions.md#d042-run-each-campaign-as-one-temporal-workflow-that-owns-its-logs) requires. The workflow mints a stable identity for each model dispatch, including each fallback, and reserves it before the activity runs.
- [x] Replace the gateway's direct `LedgerStore` charge with an attempt lease and a durable dispatch receipt. The gateway writes the receipt before it sends and records the reply and usage after. The workflow charges from the receipts.
- [x] Recover after a crash. A dispatch with a stored reply is charged and reused, not resent. A dispatch with a receipt and no reply is ambiguous, is charged as unknown usage, and is never assumed unsent.
- [x] Represent unreported usage as unknown, not zero, and record no cash cost for a subscription route.
- [x] Add model-dispatch, token, and wall-time limits to `ResourceLimits`. Dispatches and wall time are hard limits that the workflow reserves against. Tokens are observed, because a subscription may not report them.
- [x] Version the campaign revision schema for the new limits. A stored schema-version-1 revision must still parse with its original identity, so its child links hold.
- [x] Add a `rate_limited` reply that the gateway records without spending a failure slot, so the loop can pause instead of failing.
- [x] Export the full ledger from the workflow query, not only the compute spend.

**You see.**

- [x] A campaign with a dispatch limit of three refuses a fourth dispatch before it runs, and its ledger shows three charged dispatches.

**Verify, unit.**

- [x] Tests in `tests/test_accounting.py`, `tests/test_providers.py`, and `tests/test_execution.py` cover crashes before dispatch, after the provider replies, after the receipt is stored, and before the activity completes, plus the limits, the `rate_limited` reply, and opening a revision graph saved before P05. Run `PYTHONPATH=src uv run python -m pytest tests/test_accounting.py tests/test_providers.py tests/test_execution.py tests/test_campaigns.py`.

**Verify, live.**

- [x] Run a local campaign with `ScriptedProvider`, kill the worker after a reply is stored, and restart it. Pass when the ledger charges that dispatch once and the provider sees no second request.

**Verify, perf.** None.

**Review gate.** None. P05 is not review-gated.

## Call Claude Code as a tool-disabled completion (P06)

**Depends on.** P05. **Decision.** D057.

**Files.**

- [x] Create `src/quant_research/claude_adapter.py` and `tests/test_claude_adapter.py`.

**Build.**

- [x] Implement the D041 `Adapter` with the installed `claude` executable, using `-p`, `--tools ""`, `--safe-mode`, `--strict-mcp-config`, `--setting-sources ""`, `--disable-slash-commands`, `--no-session-persistence`, `--system-prompt`, and stream-json input and output. Do not use `--bare`, because it never reads the subscription sign-in. Do not use `--json-schema`, because it adds a `StructuredOutput` tool ([D057](decisions.md#d057-call-claude-code-as-a-tool-disabled-completion-behind-a-local-first-turn-gate)).
- [x] Run in an empty temporary directory, remove the environment variables quarry blanks, and set `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`.
- [x] Check `claude auth status` before each session, and check the version that the session's `system/init` reports before any request. Quarry pins 2.1.299, and the installed version is 2.1.301, so re-probe every control at the installed version and pin it ([D057](decisions.md#d057-call-claude-code-as-a-tool-disabled-completion-behind-a-local-first-turn-gate)).
- [x] Read the `system/init` tool and MCP inventory, which only `stream-json` reports, before any repository content is sent. Try a stream-json input session that sends a fixed harmless first turn, checks the inventory, and only then sends the real prompt. If that fails, D051 needs an amendment, and the PR stops for the researcher.
- [x] Map a usage-cap response to `rate_limited` and an expired sign-in to `Failed` with its reason, with bounded retries. [D077](decisions.md#d077-align-recovery-records-and-pinned-lesson-authority) adds the fixed-token reason; the adapter still makes no retry beyond the CLI's existing bound.
- [x] Define the provider's policy as public labels only, with `retains` set from the account's retention and training setting.

**You see.**

- [x] A live call returns one JSON object that the gateway accepts, and the record shows the resolved model, the CLI version, and the reported usage.

**Verify, unit.**

- [x] `tests/test_claude_adapter.py` drives a fake executable for success, malformed output, a failed run, cancellation, a rate limit, an expired sign-in, and a nonempty inventory. Run `PYTHONPATH=src uv run python -m pytest tests/test_claude_adapter.py`.

**Verify, live.** Opt-in, capped at ten calls.

- [x] Run on the researcher's configured machine. Pass when the inventory lists no tools and no MCP servers, a structured reply arrives, the usage is reported or marked unknown, and a cancellation stops the process group.
- [x] Add a user-level MCP server in a temporary configuration home and run again. Pass when the server stays out of `system/init`, and when a wrapper that drops the three isolation flags makes the adapter refuse before sending real content ([D057](decisions.md#d057-call-claude-code-as-a-tool-disabled-completion-behind-a-local-first-turn-gate)).
- [x] Write the go or no-go into D057. P06 is a go ([D057](decisions.md#d057-call-claude-code-as-a-tool-disabled-completion-behind-a-local-first-turn-gate)).

**Verify, perf.**

- [x] Metric. Added latency over a bare `claude -p` call. Rule. Fail above 2 seconds per call.

**Review gate.** The researcher reviews the isolation evidence before merge. This gate did not pass before the merge. An agent decided the review under the user's autonomy grant on 23 October 2026, and the researcher approved it after the merge, on 23 October 2026 by accepting the coordinating agent's recommendation ([D057](decisions.md#d057-call-claude-code-as-a-tool-disabled-completion-behind-a-local-first-turn-gate)).

## Call Codex as a tool-disabled completion (P07)

**Depends on.** P05. **Decision.** D058.

**Files.**

- [x] Create `src/quant_research/codex_adapter.py` and `tests/test_codex_adapter.py`.

**Build.**

- [x] Use quarry's app-server route, pinned to Codex 0.162.1, because only it reports configuration and MCP status before a turn starts. `codex exec` reports nothing before the turn, so it cannot meet D051's check before sending.
- [x] Start an ephemeral thread with `environments: []` and no dynamic tools. Override the model catalog to disable the shell tool, and disable browser use, computer use, apps, web search, and project documents. Force the ChatGPT login method, and remove `OPENAI_API_KEY`, `CODEX_API_KEY`, and `OPENAI_BASE_URL`.
- [x] Verify the ChatGPT account, `config/read`, and `mcpServerStatus/list` before `turn/start`, and give each turn a deadline.
- [x] Map replies, failures, rate limits, cancellation, and usage as in P06. Define a public-only policy with `retains` set from the account's setting. Record in D058 that OpenAI recommends API keys for automation. App-server does not report the account's data controls, so the campaign setup states `retains`, per [D058](decisions.md#d058-call-codex-through-a-fresh-app-server-in-an-isolated-home).

**You see.**

- [x] A live call returns one JSON object, and the pre-turn checks show no tool available.

**Verify, unit.**

- [x] `tests/test_codex_adapter.py` drives a fake app-server for the same cases as P06. Run `PYTHONPATH=src uv run python -m pytest tests/test_codex_adapter.py`.

**Verify, live.** Opt-in, capped at ten calls.

- [x] Pass when the pre-turn checks show no shell, browser, computer-use, apps, web search, or MCP tool, a structured reply arrives, the usage is reported or marked unknown, and a cancellation stops the process group.
- [x] Write the go or no-go into D058. A no-go leaves the program on one provider, and P16 uses its single-provider fallback.

**Verify, perf.**

- [x] Metric. Added latency over a bare app-server turn. Rule. Fail above 2 seconds per call.

**Review gate.** The researcher reviews the isolation evidence before merge. This gate did not pass before the merge. An agent decided the review under the user's autonomy grant on 23 October 2026, and the researcher accepted the isolation after the merge, the same day ([D058](decisions.md#d058-call-codex-through-a-fresh-app-server-in-an-isolated-home)).

## Give tool results the restrictions the gateway checks (P08)

**Depends on.** P01. **Decision.** D059.

**Files.**

- [x] Edit `src/quant_research/agent_tools.py`, `datasets.py`, `retrieval.py`, `sources.py`, and their tests.

**Build.**

- [x] Carry a dataset read's labels and provider approvals in `Supplied`, and each retrieval unit's occurrence restrictions in its result.
- [x] Render each tool `Call` as gateway `Material` with derived labels, approvals, and availability. Define the restrictions of an empty result and of a fixed refusal, because `derive` refuses empty input.
- [x] Cap the units a search returns and the bytes a rendered result holds.

**You see.**

- [x] A dataset read approved only for the firm proxy renders as material that the gateway refuses to send to a subscription route.

**Verify, unit.**

- [x] Tests cover every rendered outcome kind, a context with mixed provider approvals, and the gateway refusal. Run `PYTHONPATH=src uv run python -m pytest tests/test_agent_tools.py tests/test_datasets.py tests/test_retrieval.py`.

**Verify, live.**

- [x] Render a read of the P01 data and a search of a small published corpus. Pass when each rendered material names the expected labels and approvals and the gateway refuses the unapproved route.

**Verify, perf.** None.

**Review gate.** None. P08 is not review-gated.

## Let a package record no selected trial (P09)

**Depends on.** P04 and P05. **Decision.** D060.

**Files.**

- [x] Edit `src/quant_research/packages.py`, `compare.py`, `reconcile.py`, `replay.py`, `execution.py`, `registry.py`, `report.py`, `__main__.py`, and their tests. `execution.py` needed no change, because its replay job always selects its new trial.

**Build.**

- [x] Allow a package with no selected trial, including one with no trials. This amends [D006](decisions.md#d006-keep-development-packages-self-contained).
- [x] Migrate every consumer that assumes a selection. `report` renders such a package, and `compare`, `reconcile`, and `inspect` refuse it with a named reason or handle it.
- [x] Change the registry schema so it works against an existing database.

**You see.**

- [x] `quant-research verify` and `report` accept a package with no selection, and `compare` on it prints its reason.

**Verify, unit.**

- [x] Tests cover a package with no selection and no trials, each consumer, and a registry round trip on a database created before the change. Run `PYTHONPATH=src uv run python -m pytest tests/test_packages.py tests/test_compare.py tests/test_reconcile.py tests/test_registry.py tests/test_report.py`.

**Verify, live.**

- [x] Import the P04 package and a no-selection package into a local registry and export both. Pass when both exported packages verify. Done on 23 October 2026 against a registry created by the code from before the change, as [D060](decisions.md#d060-let-a-package-record-no-selected-trial) records.

**Verify, perf.** None.

**Review gate.** None. P09 is not review-gated.

## Run model calls and tool calls as runner activities (P10)

**Depends on.** P05 and P08. **Decision.** D061.

**Files.**

- [x] Create `src/quant_research/activities.py` and `tests/test_activities.py`.
- [x] Edit `src/quant_research/providers.py` and its tests.

**Build.**

- [x] Run `Gateway.call` in a worker thread inside an activity with its own timeout, heartbeat while it runs, and bridge Temporal cancellation to the adapter's `threading.Event`.
- [x] Run `Toolset.call` in an activity and return the rendered material from P08. Never relay a backing exception to a model.
- [x] Keep each dispatch's sent texts and reply as content-addressed transcript files, and add their digests to `CallRecord`.
- [x] Record the configured model, the resolved model, and the CLI version on each call.

**You see.**

- [x] Cancelling a workflow during a model call ends the call as `Cancelled`, and transcript files exist for every completed call.

**Verify, unit.**

- [x] `tests/test_activities.py` covers success, provider failure, cancellation, a timeout, and transcript digests. Run `PYTHONPATH=src uv run python -m pytest tests/test_activities.py`.

**Verify, live.**

- [x] Run a scripted provider whose call blocks, cancel the workflow, and kill a worker during a second call. Pass when the first call records `Cancelled` and the second recovers through its P05 receipt. The 23 October 2026 run passed, per [D061](decisions.md#d061-run-model-calls-and-tool-calls-as-activities-that-stop-their-adapters).

**Verify, perf.**

- [x] Metric. Activity overhead per model call with `ScriptedProvider`. Rule. Fail above 300 milliseconds per call. The median was 85 milliseconds, per [D061](decisions.md#d061-run-model-calls-and-tool-calls-as-activities-that-stop-their-adapters).

**Review gate.** None. P10 is not review-gated.

## Resolve a campaign into its run authority (P11)

**Depends on.** A go from P06 or P07, and P08. If both return no-go, the program stops and the researcher replans before P11 starts. **Decision.** D062.

**Files.**

- [x] Edit `src/quant_research/campaigns.py`, `access.py`, `model_provenance.py`, `agent_tools.py`, `datasets.py`, and their tests.

**Build.**

- [x] Resolve a frozen campaign revision into its permitted datasets and intervals, its corpus, its provider policies, its active role versions, and its prompt templates. Today the run context checks catalog approval and labels, not the campaign's own dataset list.
- [x] Activate model roles for the providers that passed P06 or P07, as recorded role versions. Campaign roles parse only as inactive today.
- [x] Register the pilot brief and the pilot datasets' contracts as the session's corpus, so `search_corpus` has a published corpus to search.
- [x] Add `list_datasets` and `describe_dataset` to the agent tools. Each returns only the campaign's permitted datasets that the run may read, and `describe_dataset` assembles the contract occurrences that the run's cutoff admits.
- [x] Check clearance before approval status in `datasets.read`, and return `unknown_dataset` for a dataset that the run cannot see, so a refusal never shows that restricted data exists. This amends [D045](decisions.md#d045-read-approved-datasets-through-a-timing-and-access-adapter).
- [x] Rebuild the authority after a steer, from the steered revision.

**You see.**

- [x] A tool read of an approved dataset that the campaign does not permit is refused.
- [x] `list_datasets` names exactly the campaign's readable datasets, and a read of an uncleared dataset gets the same refusal as a name that does not exist.

**Verify, unit.**

- [x] Tests cover the out-of-scope dataset, the corpus, role activation, the rebuild after a steer, both new tools, a contract conclusion that the cutoff hides, and the uncleared refusal. Run `PYTHONPATH=src uv run python -m pytest tests/test_campaigns.py tests/test_access.py tests/test_agent_tools.py tests/test_datasets.py`.

**Verify, live.**

- [x] Resolve a pilot campaign and print its authority. Pass when it names exactly the campaign's datasets, corpus, and active roles.

**Verify, perf.** None.

**Review gate.** None. P11 is not review-gated.

## Run the platform-owned research loop (P12)

**Depends on.** P10 and P11. **Decision.** D063.

**Files.**

- [x] Create `src/quant_research/research.py` and `tests/test_research.py`.
- [x] Edit `src/quant_research/steering.py` and its tests.

**Build.**

- [x] Add a workflow that asks a model for the next step and runs it. A step is a tool call, an experiment request, a conclusion, or a stop. The proposal uses a flat schema of string fields: the action, its arguments as JSON text, and a rationale. Experiment requests are refused until P15.
- [x] Let the workflow grow its own step list, and pin its memo to the campaign revision and the frozen rules digest instead of a job list.
- [x] Derive the restrictions of each prompt from everything in it, including earlier replies, tool results, and steering text. Steering records carry only an actor, a time, and free text today, so add declared restrictions to steering events, and withhold from every prompt any steering text whose restrictions are not declared. Bound the bytes of each prompt.
- [x] Enforce the P05 limits before each dispatch, and keep enough budget for a conclusion. When the budget runs out first, end with an inconclusive result.
- [x] Record each researcher checkpoint as a runner pause in the steering log, which the researcher's `resume` ends.
- [x] Pause the campaign on a `rate_limited` reply.
- [x] Cap attempts well below the history limit D073 records, instead of continuing as new, and measure the workflow task time at the cap.
- [x] Rename the single-branch template's tools in `workflows.py` to match `agent_tools`, or record that the templates stay unwired. [D063](decisions.md#d063-run-the-platform-owned-research-loop-as-one-temporal-workflow-per-campaign) records that they stay unwired.

**You see.**

- [x] A scripted campaign runs search and read steps, pauses at a checkpoint, resumes, and stops at its dispatch limit with an inconclusive result.

**Verify, unit.**

- [x] `tests/test_research.py` covers each action, each limit, the checkpoint pause, the rate-limit pause, prompt restrictions, and withheld steering text without declared restrictions. Run `PYTHONPATH=src uv run python -m pytest tests/test_research.py`.

**Verify, live.**

- [x] Kill the worker during a step and restart it. Pass when the campaign resumes and no dispatch with a stored reply runs twice. The 23 October 2026 run passed, per [D063](decisions.md#d063-run-the-platform-owned-research-loop-as-one-temporal-workflow-per-campaign).

**Verify, perf.**

- [x] Metric. Workflow task time at the attempt cap, and overhead per step. Rule. Fail above 2 seconds per workflow task or 1 second per step. A fresh worker replayed a capped history in at most 0.50 seconds, and the overhead was a median of 75 to 86 milliseconds per step, per [D063](decisions.md#d063-run-the-platform-owned-research-loop-as-one-temporal-workflow-per-campaign).

**Review gate.** None. P12 is not review-gated.

## Drive campaigns from the CLI (P13)

**Depends on.** P12. **Decision.** None assigned. The PR records [D076](decisions.md#d076-drive-the-research-loop-from-the-cli-and-reconcile-its-receipts-in-status).

**Files.**

- [x] Edit `src/quant_research/__main__.py`, and create `tests/test_campaign_cli.py`.

**Build.**

- [x] Add `campaign start`, `status`, `pause`, `resume`, `cancel`, and `steer`, with lazy imports of Temporal and SurrealDB. Commands that take text also take its restriction labels.
- [x] Show the ledger and the receipt reconciliation in `campaign status`.

**You see.**

- [x] `quant-research campaign status` prints the campaign's state, steps, limits, and spend.

**Verify, unit.**

- [x] Create `tests/test_campaign_cli.py`, which uses the shared `cli` fixture in `tests/conftest.py` to cover each command's arguments and errors. Run `PYTHONPATH=src uv run python -m pytest tests/test_campaign_cli.py`.

**Verify, live.**

- [x] Start a scripted campaign, then pause, resume, steer, and cancel it from the CLI. Pass when each command changes the state that `status` shows. The 23 October 2026 run passed, per [D076](decisions.md#d076-drive-the-research-loop-from-the-cli-and-reconcile-its-receipts-in-status).

**Verify, perf.** None.

**Review gate.** None. P13 is not review-gated.

## Define the search grammar and its universe (P14)

**Depends on.** P02. **Decision.** D064.

**Files.**

- [x] Edit `src/quant_research/signals.py`, `public_data.py`, and their tests.

**Build.**

- [x] Extend the operators with bounded windows, differences, ratios, comparisons, and logical combinations. Every parameter has a declared range, and every program has a size limit.
- [x] Reject a program that reads past its decision's vintage, uses a negative lag, or names an unapproved dataset. These rejections are the leaky-candidate test that Phase 4 requires.
- [x] Give each program a canonical JSON form whose digest enters the trial's specification.
- [x] Ask the researcher which ALFRED and FRED series form the search universe, then ingest and approve them through P01's commands. An agent chose the universe under the user's autonomy grant, per [D064](decisions.md#d064-search-signal-programs-in-a-bounded-grammar-over-a-chosen-alfred-universe). The researcher approved it and its series in the researcher's data root on 23 October 2026 ([pilot brief](phase0-pilot.md#pin-the-data)).

**You see.**

- [x] The Sahm program from P02 validates unchanged, and each leaky program fails with a named reason.

**Verify, unit.**

- [x] `tests/test_signals.py` gains one rejection case per leak type, a range check per operator, and the size limit. Run `PYTHONPATH=src uv run python -m pytest tests/test_signals.py`.

**Verify, live.**

- [x] Ingest the chosen series and run three hand-written programs over them. Pass when each program evaluates and the Sahm program reproduces P02's series.

**Verify, perf.** None.

**Review gate.** The researcher chooses the search universe before merge.

## Run experiments inside the loop under a frozen search method (P15)

**Depends on.** P04, P12, and P14. **Decision.** D065.

**Files.**

- [x] Edit `src/quant_research/research.py`, `reference.py`, `campaigns.py`, and their tests. P15 also edits `activities.py`, `datasets.py`, and `backtest.py` and their tests, for the reasons [D065](decisions.md#d065-run-experiments-inside-the-loop-under-a-frozen-search-method) gives.

**Build.**

- [x] Resolve the campaign's evaluation-rules digest to a frozen rules file before the first proposal. The loop refuses to start without it.
- [ ] Fix the repeated-search method in that file. [D015](decisions.md#d015-evaluate-pilot-signals-against-their-recorded-trials) excludes outcome-dependent search, so counting trials is not enough. The proposed method splits history into a search window, whose results return to the model, and a held-back window, which only the final evaluation reads. The final evaluation adjusts for the declared maximum trial count. The researcher accepts the method before merge. An agent fixed the method in [its rules file](../src/quant_research/rules/6387e6b8c71e4b825295ab7c92cbca7d3230afabc564788fea32b62b4bb3b8b0.json) under the user's autonomy grant on 23 October 2026, per [D065](decisions.md#d065-run-experiments-inside-the-loop-under-a-frozen-search-method). The researcher has not reviewed it.
- [x] Run the search under a run authority whose cutoff ends the search window, so tool reads and experiments cannot reach held-back data. The final evaluation uses a separate authority.
- [x] Validate each experiment request as a P14 program, then run it through P04's pipeline as one activity, on the search window only.
- [x] Record every candidate as a trial in one package lineage, including rejected and failed ones.
- [x] Derive the restrictions of each experiment result.

**You see.**

- [x] A scripted campaign that proposes three programs, one of them leaky, records three trials, rejects the leaky one, and has its read of a held-back month refused. The read supplies no held-back row instead of refusing one, because `read_dataset` names no month ([D065](decisions.md#d065-run-experiments-inside-the-loop-under-a-frozen-search-method)).

**Verify, unit.**

- [x] `tests/test_research.py` covers the rules file, the window split, the rejection, and the trial record. Run `PYTHONPATH=src uv run python -m pytest tests/test_research.py tests/test_reference.py`.

**Verify, live.**

- [x] Run a scripted campaign over the real pilot data that tries a `read_dataset` call and an experiment reaching into the held-back window. Pass when every trial is recorded and both attempts are refused. The read passes by supplying no held-back row, as [D065](decisions.md#d065-run-experiments-inside-the-loop-under-a-frozen-search-method) records.

**Verify, perf.**

- [x] Metric. Wall time per experiment on the search window. Rule. Fail above 30 seconds per candidate.

**Review gate.** The researcher accepts the repeated-search method before merge.

## Interpret results under critique (P16)

**Depends on.** P15, and the go or no-go of P06 and P07. **Decision.** D066.

**Files.**

- [x] Edit `src/quant_research/research.py` and `tests/test_research.py`.

**Build.**

- [x] At the end of the search, evaluate the shortlist once on the held-back window. Keep each result provisional development evidence.
- [x] Ask the proposing provider for a conclusion that cites trial IDs, and ask the other provider to critique it. With one provider, use a fresh completion from the same provider and record that the critique is not independent.
- [x] Classify the campaign's conclusion as supported, null, inconclusive, or no result. Only evaluation verdicts produce supported or null. A failed call, a rejected candidate, an empty shortlist, or a disagreeing critic is not evidence of a null effect.
- [x] Refuse a positive claim that lacks a supported held-back verdict, whatever the critique says.

**You see.**

- [x] A scripted campaign whose model claims success without a supported trial ends inconclusive, with both texts in its transcript.

**Verify, unit.**

- [x] `tests/test_research.py` covers each conclusion class, a refused claim, a critique rejection, an empty shortlist, and the single-provider fallback. Run `PYTHONPATH=src uv run python -m pytest tests/test_research.py`.

**Verify, live.** Opt-in, capped at 20 subscription calls.

- [x] Run a small real campaign with Claude proposing and Codex critiquing, or the fallback. Pass when the conclusion cites recorded trials and its class matches the held-back verdicts. The development live check passed on 23 October 2026, per [D066](decisions.md#d066-interpret-held-back-results-with-a-separate-critique).

**Verify, perf.** None.

**Review gate.** None. P16 is not review-gated.

Implemented on 23 October 2026. [D066](decisions.md#d066-interpret-held-back-results-with-a-separate-critique) records the finalization protocol, completed checks, and live verification. The reference pipeline and evaluation count also change because the loop had no held-back entry point.

## Save a campaign as one complete research package (P17)

**Depends on.** P09 and P16. **Decision.** D067.

**Files.**

- [x] Edit `src/quant_research/research.py`, `report.py`, and their tests, and verify the existing `registry.py` import/export contract without a schema or implementation change.

**Build.**

- [x] Write the campaign's package when the campaign ends for any reason: a conclusion, a cancel, an exhausted budget, or a failure.
- [x] Include as artifacts every trial, the exact dataset vintages read, the frozen rules, the campaign revisions, the steering and acceptance logs, the ledger, the call receipts, the transcripts, the conclusion, the critique, and the exposure events.
- [x] Add each model's cutoff qualification with `model_provenance.CutoffCheck` to the package and the report.
- [x] Extend the report with each trial's evaluation, the conclusion and its class, the critique, the human decisions, and the stopping reason.
- [x] Import the package into the registry.

**You see.**

- [x] `quant-research report` on a campaign package lists every trial with its status and shows the conclusion, the critique, the human decisions, and the cutoff qualifications.

**Verify, unit.**

- [x] Tests cover a package from each ending, the artifact list, and the report sections. Run `PYTHONPATH=src uv run python -m pytest tests/test_research.py tests/test_report.py tests/test_registry.py`.

**Verify, live.**

- [x] Export the P16 campaign's package, make the original data and transcript stores unavailable, and rerun its trials from the package alone. Pass when the rerun gives identical numerical artifacts.

**Verify, perf.** None.

**Review gate.** None. P17 is not review-gated.

Implemented and live-verified on 23 October 2026. [D067](decisions.md#d067-archive-each-research-ending-as-a-complete-campaign-snapshot) records the immutable campaign snapshot, completed checks, package-only numerical reproduction, and operational limits. The Phase 4 capability gate stays open.

## Register the frozen procedure for prospective observation (P18)

**Depends on.** P17. **Decision.** D068, which amends D032 if its rules change.

**Files.**

- [x] Edit `src/quant_research/prospective.py`, `research.py`, `__main__.py`, and their tests.

**Build.**

- [x] Register each finished campaign's frozen procedure and shortlist, including an empty one, with its model records and its outcome horizon.
- [x] Separate the procedure's historical initialization data from the evaluation data released after registration. `Observation.available_at` today names the earliest data the observation read, which older observations always precede. Keep each original availability date, so a new download never refreshes one.
- [x] Keep the registration log in a local file store, and add `quant-research observe`.

**You see.**

- [x] `observe` on a fixture with a vintage released after registration records one eligible observation.

**Verify, unit.**

- [x] Tests cover registration of an empty shortlist, the separation of initialization from evaluation data, and refusal of evaluation data released before registration. Run `PYTHONPATH=src uv run python -m pytest tests/test_prospective.py`.

**Verify, live.**

- [x] Register the P16 campaign's procedure. Pass when the registration log verifies. The first real observation waits for the next monthly release and is not on the merge path.

**Verify, perf.** None.

**Review gate.** None. P18 is not review-gated.

Implemented and live-verified on 23 October 2026. [D068](decisions.md#d068-register-campaign-procedures-before-later-released-outcomes) records the durable log, release eligibility, actual P16 registration, and limits. The first real prospective observation remains pending.

## Record the single-branch baseline on development cases (P19)

**Depends on.** P17. **Decision.** D069.

**Files.**

- [x] Create task files under `tests/fixtures/capability/pilot/`, and edit `src/quant_research/capability.py` and its tests.

**Build.**

- [x] Add development semantics to the capability ledger. Fresh execution identities permit reruns with one task-set version and event directory while development exposure excludes later held-out use.
- [x] Write known-answer, null, and flawed cases. Leaky candidates are `flawed` cases, and inconclusive outcomes are scored under `null`.
- [x] Wrap the loop as a capability procedure, and freeze one configuration with its model records, resources, starting memory, steering, and the share of repeats in which a known-answer case must succeed. Record unknown model versions as unknown.
- [x] Record that these development cases do not establish the master plan's held-out capability comparison.

**You see.**

- [x] The disclosure lists each case's score per repeat.

**Verify, unit.**

- [x] Tests cover the procedure wrapper with a scripted provider and the development semantics. Run `PYTHONPATH=src uv run python -m pytest tests/test_capability.py`.

**Verify, live.** Opt-in, capped at 150 subscription calls in total.

- [x] Run the frozen configuration with three repeats and save the disclosure. Pass when every flawed case is rejected in every repeat, no null case yields a positive claim in any repeat, and the known-answer cases meet the frozen success share.

**Verify, perf.**

- [x] Metric. Subscription calls and wall time per case. Rule. Fail above 15 calls or 20 minutes per case.

**Review gate.** None. P19 is not review-gated.

Implemented and live-verified on 23 October 2026. [D069](decisions.md#d069-record-a-reusable-development-baseline-through-the-research-loop) records the sealed configuration, all twelve repeat scores, cumulative calls and wall time, and the supplied-candidate scope. The researcher has not reviewed the agent decision, and the Phase 4 held-out gate stays open.

## Run the first autonomous research session (P20)

**Depends on.** P13 and P18. **Decision.** None.

**Files.**

- [x] Edit `docs/phase0-pilot.md`, `docs/project-state.md`, and `docs/todos.md` with the evidence.

**Build.**

- [ ] Write a campaign question with the researcher, for example whether real-time labor-market series improve on the Sahm rule for timing equity exposure. Set the budget, the checkpoints, and the frozen rules.
- [x] Run it. Pause and steer it once, kill the worker once, and let it stop at a limit or a conclusion.
- [x] Run a second, smaller session that deliberately exhausts its dispatch limit.

**You see.**

- [x] A verified package, its report, a registered procedure, and a ledger within budget, for each session.

**Verify, unit.**

- [x] None. P20 changes no code. Its evidence is the live runs below.

**Verify, live.** Capped by each campaign's dispatch limit.

- [x] Pass when the first session survives the worker kill, records the steer as a revision, stops at its limit or conclusion, and reruns its trials from the exported package with identical numerical artifacts.
- [x] Pass when the second session stops at its dispatch limit and still writes a complete package with an inconclusive or no-result conclusion.

**Verify, perf.** None.

**Evidence.** Both live checks passed on 23 October 2026, per the [P20 live record](phase0-pilot.md#review-the-first-autonomous-local-sessions). The agent chose and froze the questions, rules, budgets, and checkpoints under the autonomy grant. The original requirement to write the question with the researcher remains unchecked because no researcher coauthorship occurred.

**Review gate.** An agent reviewed both packages and reports together with P19's baseline under the autonomy grant and accepted the operational evidence. The researcher has not reviewed them, so the original researcher-review gate remains unmet. The master-plan Phase 4 gate stays open for the independent held-out capability comparison.

## Register the paper case (P21)

**Depends on.** P01. Off the critical path. **Decision.** D070.

**Files.**

- [x] Edit `src/quant_research/public_data.py` and its tests, and add the converted paper under the source store.

**Build.**

- [x] Convert Estrella and Mishkin (1998) to Markdown outside the store, register it, and publish its chunk and locator generations.
- [x] Ingest FRED Treasury yields and NBER recession dates through P01's commands.

**Verify, unit.**

- [x] Tests cover the registered source, its locations, and the new series. Run `PYTHONPATH=src uv run python -m pytest tests/test_public_data.py tests/test_locators.py`.

**Verify, live.**

- [x] Look up a table location in the paper. Pass when the located text matches the paper.

**Review gate.** None. P21 is not review-gated.

## Specify and run the paper reproduction (P22)

**Depends on.** P21. Off the critical path. **Decision.** D071, which amends D020 only if its rules change.

**Files.**

- [x] Create `src/quant_research/probit.py` and its tests, and add a dependency only if the PR justifies it.

**Build.**

- [x] Choose one table and model, and freeze a reproduction specification as adapted data unless matched inputs are established. The operator grammar cannot express a probit, so this PR adds its own estimator.
- [x] Run the estimator and classify each reported target.

**Verify, unit.**

- [x] Tests check the estimator against a hand-computed probit. Run `PYTHONPATH=src uv run python -m pytest tests/test_probit.py tests/test_reproduction.py`.

**Verify, live.**

- [x] Run the reproduction. Pass when each target has a classification with its tolerance.

**Review gate.** The researcher reviews the comparison basis before the run. This gate did not pass before the run. An agent decided the basis under the user's autonomy grant on 22 October 2026, and the researcher approved it after the run, on 23 October 2026, by accepting the coordinating agent's recommendation ([D071](decisions.md#d071-reproduce-table-2s-spread-probit-on-adapted-fred-data)).

## Close the program

- [ ] Every box above is checked with its evidence. P20's operational checks passed, but original researcher gates and unaudited historical wave or ownership claims remain unchecked. This close-out does not retroactively satisfy them.
- [x] The [P20 live record](phase0-pilot.md#review-the-first-autonomous-local-sessions) names the passed operational gates, scoped skips, pending researcher review, independent held-out Phase 4 prerequisite, and firm-port requirements.

## Appendix A. What is proven and what is not

When this plan was written, nothing in it had run. `claude --help`, `codex exec --help`, and `codex features list` show the flags and features that P06 and P07 rely on. `codex features list` reports the shell tool, browser use, computer use, and apps enabled on this machine. P07's live check showed structured output, the checks before sending, usage reporting, cancellation, and isolation from user configuration for Codex 0.162.1 ([D058](decisions.md#d058-call-codex-through-a-fresh-app-server-in-an-isolated-home)). P06's live check showed the same for Claude Code 2.1.301 ([D057](decisions.md#d057-call-claude-code-as-a-tool-disabled-completion-behind-a-local-first-turn-gate)). vectorbt 1.2.0 declares support for Python 3.12 on PyPI, but it has not been installed here.

## Appendix B. Alternatives rejected

- Native agent loops in the Claude Agent SDK or Codex app-server run tool calls inside the runtime, outside the gateway. D051 records why the platform owns the loop instead.
- `codex exec` was the first Codex route considered. It reports no configuration or tool inventory before the turn, so P07 uses the app-server route.
- Continuing the workflow as new would carry the whole logs across the boundary as one input payload, which D073 does not bound. The bounded pilot caps attempts instead.
- Generated Python in the D047 worker is a separately gated extension in the master plan. The first pilot searches a grammar of approved operators.
- Plan-included API credits need the researcher's plan details and consent, which D051 leaves open.
- Persisting campaign logs in SurrealDB and the research console are Phase 2 work that the scoped local session skips. The final package holds the logs.

## Appendix C. Risks

- P07 may find that the app-server route cannot disable every tool. The program then has one provider, P16 uses its single-provider fallback, and two-provider support becomes a prerequisite. If P06 also fails, no subscription route exists, and the program stops for a replan.
- P06 may find no way to read the inventory before sending real content. D051 then needs an amendment that the researcher approves.
- P03's vectorbt may not resolve on Python 3.12. P03 then names a replacement engine, or Phase 1 waits for a scope change.
- P15's held-back window is still known to the researcher and to the models, so even its results are development evidence.
- Subscription limits assume ordinary individual use, so P19 and P20 may hit usage caps. Their call caps bound the spend, and a `rate_limited` reply pauses the campaign.
- P02's missing-month convention may not reproduce FRED's values. P02 then excludes the affected months with recorded reasons, and the researcher decides in P04 whether the narrower window is acceptable.
- Runner tests run only in the full suite, and `tests/test_execution.py`'s `server` fixture shadows the SurrealDB fixture of the same name. P05, P10, and P12 must use `running` directly when they need both servers.
- Every code PR changes replay's implementation digest, because it hashes every module under `quant_research`. No test pins that digest today.

## Appendix D. Reviews and reading list

- Claude Fable 5.1 and GPT-6 Astra at high effort reviewed the first draft of this plan, and both returned "agree with changes". This revision applies their findings.
- [Master plan](master-plan.md), [Phase 0 pilot brief](phase0-pilot.md), and [D050](decisions.md#d050-run-the-phase-0-pilot-locally-on-public-us-macro-data) to [D051](decisions.md#d051-own-the-research-loop-and-call-subscriptions-as-tool-disabled-completions).
- The runner in [D042](decisions.md#d042-run-each-campaign-as-one-temporal-workflow-that-owns-its-logs), the gateway in [D041](decisions.md#d041-check-outbound-provider-content-against-the-run-and-a-versioned-provider-policy), the agent tools in [D049](decisions.md#d049-expose-evidence-to-agents-through-typed-tools-bound-to-the-run-context), and the dataset adapter in [D045](decisions.md#d045-read-approved-datasets-through-a-timing-and-access-adapter).
- Quarry's subscription runtimes in `sunpar/quarry`, `src/quarry/agent/claude_runtime.py` and `codex_runtime.py`.
