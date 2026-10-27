# Resolve the open questions

These questions identify the facts and ownership needed for real integration. Return to [project state](project-state.md) for the index. See [architecture](architecture.md) for the affected boundaries, [decisions](decisions.md) for selected defaults, and [to dos](todos.md) for dependent work.

## Q001. Which numerical frameworks are the integration target?

Status is selected for the local pilot and open for the firm. The firm's backtester and signal evaluator are the production target, and their interfaces get inspected when the code moves to the firm. Until then, the [local pilot](phase0-pilot.md) uses vectorbt 1.2.0 for portfolio construction and backtests and the [pilot evaluation profile](decisions.md#d015-evaluate-pilot-signals-against-their-recorded-trials) for evaluation, per [D050](decisions.md#d050-run-the-phase-0-pilot-locally-on-public-us-macro-data).

[D054](decisions.md#d054-backtest-allocations-with-a-plain-reference-and-a-vectorbt-engine) ran vectorbt 1.2.0 on Python 3.12 and checked its fee and cash handling against hand-calculated cash flows, which closes the local part. Close the firm part with the firm systems' inspected calls, supported Python versions, and a bounded runnable example.

## Q002. Which data can the pilot use?

Status is selected for the local pilot and open for the firm. The local pilot uses ALFRED vintages of `UNRATE`, FRED's `SAHMREALTIME`, and Ken French monthly factor returns, and the paper case adds ALFRED's `GS10`, `TB3MS`, and `USRECQ` for Treasury yields and NBER recession dates ([D070](decisions.md#d070-register-the-estrella-and-mishkin-working-paper-and-its-fred-series)). The [pilot brief](phase0-pilot.md#pin-the-data) records each series' known limits, including the missing October 2025 observation and ALFRED's vintage start in March 1960.

`quant-research data pull` and `data ingest` now pin each download and ingest it into the [dataset adapter](decisions.md#d045-read-approved-datasets-through-a-timing-and-access-adapter)'s rows with its revision, publication, and arrival times, per [D052](decisions.md#d052-ingest-the-pilots-public-data-into-a-vintage-addressed-store). Close the local part when the researcher validates and approves the pilot's own data root. The firm's data sources must expose the same times, or the adapter cannot enforce cutoffs there. Database timestamps alone cannot prove information availability. For firm data, the required evidence stays identifiers, units, calendars, historical universe membership, publication and revision history, firm-arrival history, licensing, and access restrictions.

## Q003. What independent baseline defines agreement?

Status is closed. The researcher accepted the answer on 23 October 2026. The researcher owner and the methodological reviewer are the same person, so review is not independent. The reference signal is the real-time Sahm rule, its known answer is FRED's `SAHMREALTIME`, and the [pilot brief](phase0-pilot.md#accept-the-local-pilot-as-a-scoped-phase-1) declares the costs, the evaluation profile, and the reconciliation rules.

The answer is FRED's pinned `SAHMREALTIME` download, compared month by month with the recomputed series within 1/200 over the declared window. A month leaves the window only with a recorded reason. `quant-research reconcile series` produces the recomputed series and its agreement, per [D053](decisions.md#d053-evaluate-signal-programs-per-decision-and-reconcile-the-sahm-series-with-fred). P04 saves them in the reference package with the pinned `SAHMREALTIME` vintage, per [D055](decisions.md#d055-run-the-reference-experiment-through-one-audited-pipeline-against-a-frozen-experiment-file). Its live run matched 780 of 800 months and excluded 20 with recorded reasons. It found no mismatch. The researcher accepted the exclusions, the frozen evaluation file, and the scoped Phase 1 result on 23 October 2026. The answer covers the local pilot on public data. The firm frameworks need their own acceptance.

## Q004. What has already been observed?

Status is answered for the local pilot. The researcher has seen the Sahm rule's historical performance, including its 2024 trigger, and knows the macro history of the sample, as the models do. All history is development data, and confirmation is prospective only, per the [pilot brief](phase0-pilot.md#record-exposure-and-confirmation). The search method of [D065](decisions.md#d065-run-experiments-inside-the-loop-under-a-frozen-search-method) keeps data available after 1 January 2005 from a search's tools and experiments and holds it back for the final evaluation, but the researcher and the models know that window too, so its verdicts are still development evidence.

Keep restricted confirmation outcomes outside ordinary development exports. An empty local history does not establish confirmation eligibility.

## Q005. Which infrastructure and provider assumptions are correct?

Status is answered for the local pilot and open for the firm. The local pilot runs on one Mac with a local SurrealDB server, a local Temporal dev server, and the researcher's Claude Code and Codex subscriptions. Models receive public material only, through tool-disabled completions under [D051](decisions.md#d051-own-the-research-loop-and-call-subscriptions-as-tool-disabled-completions). OpenAI recommends API keys over a ChatGPT sign-in for automated Codex use, as [D058](decisions.md#d058-call-codex-through-a-fresh-app-server-in-an-isolated-home) records, and the researcher keeps the ChatGPT sign-in.

The firm will provide a model proxy, a research database, and its data services. The proxy's transport and terms, the database version, and whether the firm runs Temporal or a compute scheduler are unknown. [D027](decisions.md#d027-qualify-the-pinned-surrealdb-server-and-client-over-websocket) pins a local SurrealDB server and client for development only. The hosted backend, backup expectations, and credentials for each access boundary remain open, as the [SurrealDB plan](surrealdb-plan.md#12-remaining-implementation-inputs) lists.

Routine verification uses fixtures with no production credentials, paid APIs, cloud services, or model calls. Subscription probes are explicit and opt-in.

The gateway records no cash cost for a subscription call, but a Claude plan with usage credits turned on, or a usage-based Enterprise plan, bills Claude Code calls at API rates. [D057](decisions.md#d057-call-claude-code-as-a-tool-disabled-completion-behind-a-local-first-turn-gate) records what the CLI reports about it. On 23 October 2026 the researcher reported a $200-a-month Claude subscription with usage credits turned on and none bought. Later that day the researcher decided not to buy usage credits and confirmed that auto-reload is off. Anthropic's [usage credits page](https://support.claude.com/en/articles/12429409-manage-usage-credits-for-paid-claude-plans) says that usage past an included limit continues until the monthly spend limit, or until the credit balance runs out with auto-reload off. With no balance and auto-reload off, a call past the limit cannot be billed, so the gateway's zero cost for a subscription call holds. Buying credits or turning on auto-reload would reopen the question.

## How to close a question

Retain its identifier, record the answer and supporting evidence, and update the affected architecture and tasks in the same PR. Put a resulting choice in [decisions](decisions.md). If the answer creates another prerequisite, record it here instead of hiding it in chat.
