# Choose the next work

This list tracks capabilities and dependencies for the current revision. Return to [project state](project-state.md) for the index. [The initial PR plan](initial-prs.md) defines acceptance details, [architecture](architecture.md) defines boundaries, and [open questions](open-questions.md) records prerequisites. GitHub pull requests own live review and merge state.

## Completed foundations

- [x] Import the project material and record source provenance and limits.
- [x] Configure project-local pstack models and disable project Superpowers use.
- [x] Write and independently review the master plan.
- [x] Merge [the master plan PR](https://github.com/sunpar/quant-research/pull/1).

## Merged. Foundation and portable packages

- [x] Implement the locked Python foundation, repository runner, hooks, and shared commands in [PR 2](https://github.com/sunpar/quant-research/pull/2).
- [x] Implement portable experiment packages, the inspect and verify CLI, and the authoring example in [PR 3](https://github.com/sunpar/quant-research/pull/3).

## Merged. Faster verification and a leaner plan

- [x] Run the smoke suite on push, drop the PR wrapper's full run, trim the plan, and record SurrealDB as the planned store in [PR 4](https://github.com/sunpar/quant-research/pull/4).
- [x] Shard the full suite into four isolated pytest processes, keep gate fixtures off the hooks, and keep three gate checks in the smoke suite, per [D044](decisions.md#d044-shard-the-full-suite-and-keep-the-smoke-suite-small).
- [x] Skip the gate tests in `make verify` and the hooks when a full suite already passed their exact inputs, per [D072](decisions.md#d072-skip-gate-tests-whose-exact-inputs-already-passed).
- [x] Run the full suite only inside `make merge`, one merge at a time, and replace the push hook with merge-commit and rebase hooks, per [D074](decisions.md#d074-run-the-full-suite-only-inside-make-merge).

## Merged. Causal development replay

- [x] Implement one declared synthetic signal and diagnostic profile.
- [x] Select inputs by their actual fixture availability and preserve selected-version provenance.
- [x] Verify known-answer, null, inconclusive, and deliberately flawed cases with independent expected values.
- [x] Save reproducible packages and check late arrivals, revisions, membership, and input ordering.
- [x] Replay a fixture from the built wheel.
- [x] Run the full suite on the final commit and merge [PR 5](https://github.com/sunpar/quant-research/pull/5).
- [x] Remove replay checks that `save_package` already enforces, and hash every module in the package as the implementation.

## Pilot evaluation profile

- [x] Evaluate uncertainty, selection effects, subperiod robustness, and incremental value with independently derived expected values in [PR 23](https://github.com/sunpar/quant-research/pull/23), per [D015](decisions.md#d015-evaluate-pilot-signals-against-their-recorded-trials).
- [x] Record the declared minimum effect, subperiod count, and alpha on each evaluation result, and keep replay's descriptive diagnostic, per [D028](decisions.md#d028-keep-replay-descriptive-and-record-evaluation-parameters-with-each-result).

## Merged. Qualify SurrealDB storage

- [x] Pin the SurrealDB 3.3.4 server and Python SDK 2.0.0 with a local server configuration.
- [x] Probe decoding, statement errors, identities, time, optional values, decimals, transactions, conflicts, uniqueness, backup and restore, and filtered vector search against a real server ([D027](decisions.md#d027-qualify-the-pinned-surrealdb-server-and-client-over-websocket)).
- [x] Run the full suite on the final commit and merge [PR 17](https://github.com/sunpar/quant-research/pull/17).

## Experiment comparison

- [x] Compare the selected trials of two verified packages with `quant-research compare`, per [D034](decisions.md#d034-compare-the-selected-trials-of-two-verified-packages).

## Follow-ups from causal replay

- [x] Represent unknown exposure coverage in the package format, then make replay's data exposure cover every raw row it reads. This changes [packages](../src/quant_research/packages.py), and [D010](decisions.md#d010-replay-one-declared-synthetic-causal-profile) records the rule.
- [x] Remove the duplicate `cli` helper from [the package tests](../tests/test_packages.py) in favor of the [shared fixture](../tests/conftest.py).

## Research package report

- [x] Render a deterministic Markdown report from a verified package with `quant-research report`.

## Real integration after the local batch

- [x] Run both bounded P20 sessions, verify steering and paused-state worker recovery, reconcile their calls, register their actual shortlists, and reproduce the first export's numerical evidence, per the [P20 live record](phase0-pilot.md#review-the-first-autonomous-local-sessions). Agent review under autonomy passed the operational checks; the researcher has not reviewed the packages.
- [ ] Obtain the researcher's review of both P20 packages and reports together with P19's baseline. Preserve the open independent held-out Phase 4 comparison and the [firm-port prerequisites](phase0-pilot.md#list-what-changes-at-the-firm-port).

The [autonomous pilot PR plan](autonomous-pilot-prs.md) orders the local work below into pull requests P01 to P22.

- [x] Select the Phase 0 local pilot and record its brief, roles, exposure, and provider policy in the [pilot brief](phase0-pilot.md), per [D050](decisions.md#d050-run-the-phase-0-pilot-locally-on-public-us-macro-data). The firm parts of Q001, Q002, and Q005 stay open, and [Q003](open-questions.md#q003-what-independent-baseline-defines-agreement) closed on 23 October 2026.
- [x] Reconcile the recomputed real-time Sahm series with the pinned `SAHMREALTIME` download over the declared window, and record the missing-month convention in the pilot brief, per [D053](decisions.md#d053-evaluate-signal-programs-per-decision-and-reconcile-the-sahm-series-with-fred). The researcher accepted its excluded months on 23 October 2026, per [D055](decisions.md#d055-run-the-reference-experiment-through-one-audited-pipeline-against-a-frozen-experiment-file).
- [x] Bound the signal operators, reject leaky programs with named reasons, give each program a canonical digest, and choose the search universe, per [D064](decisions.md#d064-search-signal-programs-in-a-bounded-grammar-over-a-chosen-alfred-universe). An agent chose the universe under the user's autonomy grant.
- [x] Review the search universe, then pull, ingest, validate, and approve its new series in the researcher's own data root. The researcher approved both on 23 October 2026 ([pilot brief](phase0-pilot.md#pin-the-data)).
- [x] Cache each stored vintage's parse in the dataset adapter, so a search candidate that reads `INDPRO` fits P15's 30-second rule ([D064](decisions.md#d064-search-signal-programs-in-a-bounded-grammar-over-a-chosen-alfred-universe)). [D065](decisions.md#d065-run-experiments-inside-the-loop-under-a-frozen-search-method) caches it.
- [x] Run vectorbt 1.2.0 on Python 3.12 and check its fee and cash handling against hand-calculated cash flows before declaring the recomputation tolerance, per [D054](decisions.md#d054-backtest-allocations-with-a-plain-reference-and-a-vectorbt-engine). The live check agreed within 3.4e-16 over 1202 months, and the declared tolerance is 1e-12 per month.
- [x] Probe tool-disabled subscription completions at Claude Code 2.1.301 for structured output, reported usage, cancellation, and isolation from user-level configuration, per [D057](decisions.md#d057-call-claude-code-as-a-tool-disabled-completion-behind-a-local-first-turn-gate). The live check passed on 23 October 2026 after `claude auth login` restored the sign-in, including a signed-in session that kept out an injected MCP server, and the Claude route is a go.
- [x] Review the Claude route's isolation evidence in [D057](decisions.md#d057-call-claude-code-as-a-tool-disabled-completion-behind-a-local-first-turn-gate). An agent decided the review under the user's autonomy grant, and the researcher approved it on 23 October 2026 by accepting the coordinating agent's recommendation.
- [x] Probe tool-disabled Codex 0.162.1 completions for structured output, reported usage, cancellation, and isolation from user-level configuration, per [D058](decisions.md#d058-call-codex-through-a-fresh-app-server-in-an-isolated-home). The live check passed, and the Codex route is a go.
- [x] Review the Codex route's isolation evidence in [D058](decisions.md#d058-call-codex-through-a-fresh-app-server-in-an-isolated-home). The researcher accepted it on 23 October 2026.
- [x] Add `claude-subscription` to `model_provenance.ACCEPTED` and its model record to `MODELS` ([D062](decisions.md#d062-resolve-a-campaign-revision-into-its-run-authority)). The route was activated after the researcher's approval. The researcher then decided not to buy usage credits and confirmed that auto-reload is off, which answers the billing part of [Q005](open-questions.md#q005-which-infrastructure-and-provider-assumptions-are-correct).
- [x] Resolve a campaign revision into a run authority with a dataset scope, a pinned corpus of the pilot brief and dataset contracts, active roles on the Codex route, prompts, and `list_datasets` and `describe_dataset`, per [D062](decisions.md#d062-resolve-a-campaign-revision-into-its-run-authority). An agent decided the design under the user's autonomy grant.
- [ ] Review the fields of the pilot dataset contracts in `src/quant_research/contracts` and the availability dates of their conclusions, so that a reviewed claim can enter a contract version ([D062](decisions.md#d062-resolve-a-campaign-revision-into-its-run-authority)). The researcher approved publishing them on 23 October 2026 by accepting the coordinating agent's recommendation, which does not review their fields.
- [x] Apply a campaign's permitted intervals at read time under pinned monthly contracts, preserving legacy authority and archived reads, per [D081](decisions.md#d081-enforce-permitted-months-through-pinned-dataset-contracts).
- [x] Run the platform-owned research loop as one Temporal workflow per campaign, with checkpoints, rate-limit pauses, a held conclusion budget, and steering text that reaches a prompt only when its restrictions are declared, per [D063](decisions.md#d063-run-the-platform-owned-research-loop-as-one-temporal-workflow-per-campaign). An agent decided the design under the user's autonomy grant.
- [x] Review the research loop's protocol, checkpoint grammar, and closing rule in [D063](decisions.md#d063-run-the-platform-owned-research-loop-as-one-temporal-workflow-per-campaign). The researcher approved them on 23 October 2026 by accepting the coordinating agent's recommendation.
- [x] Run experiments inside the research loop under a frozen search method, with a search window whose verdicts return to the model, a held-back window for the final evaluation, and every candidate a trial of one lineage, per [D065](decisions.md#d065-run-experiments-inside-the-loop-under-a-frozen-search-method). An agent decided the method under the user's autonomy grant.
- [ ] Review the repeated-search method, its windows, and its rules file in [D065](decisions.md#d065-run-experiments-inside-the-loop-under-a-frozen-search-method). An agent decided them under the user's autonomy grant on 23 October 2026, and the researcher has not reviewed them.
- [x] Evaluate a frozen shortlist once on the held-back window, obtain a trial-citing conclusion and a separate critique, and refuse unsupported positive claims, per [D066](decisions.md#d066-interpret-held-back-results-with-a-separate-critique). An agent decided the protocol under the user's autonomy grant.
- [x] Archive every supported campaign ending, preserve complete recorded evidence, and reproduce P16 numerical artifacts from its exported package alone, per [D067](decisions.md#d067-archive-each-research-ending-as-a-complete-campaign-snapshot).
- [ ] Review the campaign snapshot and late-evidence policy in [D067](decisions.md#d067-archive-each-research-ending-as-a-complete-campaign-snapshot). An agent chose them under the autonomy grant, and the researcher has not reviewed them.
- [x] Drive the research loop from the CLI with `campaign start`, `status`, `pause`, `resume`, `cancel`, and `steer`, and reconcile the ledger with the call log's receipts in `status`, per [D076](decisions.md#d076-drive-the-research-loop-from-the-cli-and-reconcile-its-receipts-in-status). An agent decided the design under the user's autonomy grant.
- [x] Review the campaign CLI's declaration flags and receipt reconciliation in [D076](decisions.md#d076-drive-the-research-loop-from-the-cli-and-reconcile-its-receipts-in-status). The researcher approved them on 23 October 2026 by accepting the coordinating agent's recommendation.
- [x] Run a research worker from a command and a configuration file that names its provider routes, call log, and pilot operator `Setup`, per [D078](decisions.md#d078-run-campaign-workers-and-reconcile-live-receipts-under-bounded-steers).
- [x] Let a running loop's status carry a coherent ledger prefix and reconcile each call by ID before the loop closes, per [D078](decisions.md#d078-run-campaign-workers-and-reconcile-live-receipts-under-bounded-steers).
- [x] Share the `close` activity and refuse a campaign run whose campaign already has an open research loop, per [D077](decisions.md#d077-align-recovery-records-and-pinned-lesson-authority).
- [x] Limit each steer's revision text and verify the existing encoded opening bound before it enters history, per [D078](decisions.md#d078-run-campaign-workers-and-reconcile-live-receipts-under-bounded-steers). Repeated failed or concurrent legal steers can still grow aggregate paused history.
- [x] Give each rendered tool result derived labels, provider approvals, and availability, so the gateway can check it before a provider adapter sends it, per [D059](decisions.md#d059-render-each-tool-call-as-gateway-material-under-the-restrictions-of-what-it-read).
- [x] Choose one accounting owner for the campaign runner and the provider gateway, with shared attempt IDs, a usage mapping that keeps unreported usage unknown, and reconciliation after a crash or cancellation, per [D056](decisions.md#d056-make-the-workflow-the-single-accounting-owner-with-durable-call-receipts). The workflow owns the ledger and charges each call from a durable receipt.
- [x] Let a package record no selected trial, including one with no trials, and migrate its readers and the registry schema, per [D060](decisions.md#d060-let-a-package-record-no-selected-trial).
- [x] Run model calls and tool calls as Temporal activities that stop their adapters on a cancel or a deadline, with content-addressed transcripts and the served runtime and model on each call record, per [D061](decisions.md#d061-run-model-calls-and-tool-calls-as-activities-that-stop-their-adapters).
- [x] Run new `CampaignRun` model attempts through `ask_model`, with worker preparation and interpretation, settled cancellation, and retained-history compatibility, per [D080](decisions.md#d080-run-new-campaign-model-attempts-through-workflow-owned-settlement). Retained legacy executions keep their original cancellation behavior.
- [x] Share content-addressed writes through `records.keep_object`, preserving each store's directory and error rules, per [D077](decisions.md#d077-align-recovery-records-and-pinned-lesson-authority).
- [x] Record fixed-token failure reasons from both adapters in call log schema 4, preserving version 3 logs and recorded activity results, per [D077](decisions.md#d077-align-recovery-records-and-pinned-lesson-authority).
- [x] Register Estrella and Mishkin's Research Paper 9609 with its chunk and locator generations, and ingest `GS10`, `TB3MS`, and `USRECQ`, per [D070](decisions.md#d070-register-the-estrella-and-mishkin-working-paper-and-its-fred-series). An agent chose the working-paper version under the user's autonomy grant.
- [x] Choose one Estrella and Mishkin table and model, freeze its comparison basis in a reproduction specification, and classify a live run, per [D071](decisions.md#d071-reproduce-table-2s-spread-probit-on-adapted-fred-data). An agent chose Table 2's SPREAD row, the adapted basis, and the tolerances under the user's autonomy grant.
- [x] Review the paper reproduction's comparison basis, choices, and tolerances ([D071](decisions.md#d071-reproduce-table-2s-spread-probit-on-adapted-fred-data)). The researcher approved them on 23 October 2026 by accepting the coordinating agent's recommendation.
- [ ] Compare Research Paper 9609's tables with the published 1998 article once a copy is available ([D070](decisions.md#d070-register-the-estrella-and-mishkin-working-paper-and-its-fred-series)).
- [ ] Resolve the inconclusive cells in appendix tables A1 to A8 before a run uses those cells ([D070](decisions.md#d070-register-the-estrella-and-mishkin-working-paper-and-its-fred-series)).
- [ ] Find a source that states the Newey-West lag that Estrella and Mishkin used, then freeze a new specification that compares the eight t-statistics ([D071](decisions.md#d071-reproduce-table-2s-spread-probit-on-adapted-fred-data)).
- [x] Register the paper and pull, ingest, validate, and approve the paper case's series in the researcher's own data root ([pilot brief](phase0-pilot.md#pin-the-data)).
- [ ] Supply the agreed prior-exposure record as confirmation history before choosing a confirmation period, per [D021](decisions.md#d021-gate-final-confirmation-on-exposure-history-and-frozen-shortlists).
- [x] Store restricted confirmation reports for review under expiring grants, with every open, refusal, and grant change in a hash-chained log, per [D048](decisions.md#d048-review-restricted-confirmation-reports-through-audited-grants). Real reviewer identities and a trusted clock wait for Phase 0.
- [x] Register the P16 pilot's frozen procedure in the verified local prospective log, with empty-shortlist support and separate initialization evidence, per [D068](decisions.md#d068-register-campaign-procedures-before-later-released-outcomes).
- [ ] Record the first real prospective observation after a future monthly release. Registration and the synthetic CLI fixture do not supply that observation ([D068](decisions.md#d068-register-campaign-procedures-before-later-released-outcomes)).
- [ ] Inspect the firm's framework and data interfaces before designing a firm adapter.
- [x] Version campaign specifications with revisions, branches, and carried exposure, per [D013](decisions.md#d013-version-campaign-specifications-as-content-addressed-revisions). Registry import waits for the SurrealDB registry.
- [x] Record campaign pauses, resumes, cancels, annotations, and methodology steers in an append-only log, per [D031](decisions.md#d031-record-campaign-steering-in-an-append-only-log). The campaign runner enforces it, and the registry must persist it.
- [x] Register sources and publish complete, versioned chunk generations in a local file store for later SurrealDB import ([D018](decisions.md#d018-publish-complete-source-generations-from-a-file-store)).
- [x] Return only the source generations that a trusted run context may read ([D036](decisions.md#d036-filter-source-reads-through-the-trusted-run-context)).
- [x] Locate equations, tables, figures, and claim sentences in published extractions as versioned locator generations ([D043](decisions.md#d043-locate-paper-elements-in-published-extractions)).
- [x] Resolve a pinned reproduction specification's equation, table, figure, and claim refs against its published locator generation when it freezes ([D020](decisions.md#d020-fix-the-reproduction-comparison-before-results)).
- [x] Search a pinned corpus of published generations exactly, by identifier and BM25, and replay the record of supplied units ([D040](decisions.md#d040-retrieve-exactly-over-a-pinned-eligible-corpus)).
- [x] Build the timing-and-access adapter that agent approval names, against synthetic fixtures ([D045](decisions.md#d045-read-approved-datasets-through-a-timing-and-access-adapter)).
- [x] Run generated signal code in a locked-down container and return only validated output ([D047](decisions.md#d047-run-generated-signal-code-in-a-locked-down-container)).
- [x] Keep the worker's container runs in a local run log, and add a hash-locked numpy and pandas layer to its image ([D047](decisions.md#d047-run-generated-signal-code-in-a-locked-down-container)).
- [x] Pull, ingest, and document the Phase 0 pilot's public datasets in a vintage-addressed store and the [catalog](../src/quant_research/catalog.py) with their evidence, under [D052](decisions.md#d052-ingest-the-pilots-public-data-into-a-vintage-addressed-store)'s timing conventions.
- [x] Validate and approve the pilot datasets in the researcher's own data root with a manual validation attestation ([pilot brief](phase0-pilot.md#pin-the-data)).
- [x] Expose corpus search, dataset reads, and paper location lookup to agents as typed tools bound to the trusted run context, per [D049](decisions.md#d049-expose-evidence-to-agents-through-typed-tools-bound-to-the-run-context). The campaign runner and a provider adapter must still call them.
- [x] Give lessons restrictions derived from their cited evidence, pin a review-log head in the run context, and let agents read admissible lessons, per [D049](decisions.md#d049-expose-evidence-to-agents-through-typed-tools-bound-to-the-run-context).
- [ ] Declare evidence restrictions in the review log from the source, dataset, and package stores, so real lessons become readable to agents.
- [x] Give `read_lessons` the restrictions from the exact review-log prefix that admits each lesson, per [D077](decisions.md#d077-align-recovery-records-and-pinned-lesson-authority). `Ledger.admissible_lessons` still returns IDs.
- [ ] Enforce [D025](decisions.md#d025-carry-restrictions-with-material-and-authority-with-the-run) restrictions and run contexts in SurrealDB retrieval, and replace the test policies of retrieval, the dataset adapter, and agent tools with Phase 0 grants.
- [x] Check outbound provider content against the run context and a versioned per-provider data policy, store each call, and charge each sent call's usage, per [D041](decisions.md#d041-check-outbound-provider-content-against-the-run-and-a-versioned-provider-policy). Real adapters and Phase 0 policies remain.
- [x] Record result acceptance as explicit events with one accepted attempt per run, per [D030](decisions.md#d030-accept-results-through-explicit-events). Persisting the log in SurrealDB remains, and packages cannot yet hold a retry that overlaps its predecessor.
- [x] Build the first durable campaign runner on the Temporal semantics in [D026](decisions.md#d026-qualify-temporal-for-durable-campaign-execution), per [D042](decisions.md#d042-run-each-campaign-as-one-temporal-workflow-that-owns-its-logs).
- [ ] Run the campaign runner's jobs through the real compute scheduler with measured costs and scheduler cancellation once [Q005](open-questions.md#q005-which-infrastructure-and-provider-assumptions-are-correct) settles it, and persist the runner's logs in SurrealDB.
- [ ] Continue a campaign run as new before its history reaches Temporal's event or byte limit, once its logs live outside workflow state ([D073](decisions.md#d073-bound-the-campaign-runners-per-event-cost-and-payloads)). New model attempts add preparation and interpretation activities, so the old estimate of about 8,500 attempts does not apply to them ([D080](decisions.md#d080-run-new-campaign-model-attempts-through-workflow-owned-settlement)).
- [x] Probe SurrealDB permissions and record-level access, which [D027](decisions.md#d027-qualify-the-pinned-surrealdb-server-and-client-over-websocket) left untested. [D038](decisions.md#d038-restrict-reads-through-record-access-and-make-denied-writes-throw) records the semantics and gaps.
- [x] Install `surreal-db` and the Temporal CLI in the manual hosted workflow for its full and coverage suites. [D074](decisions.md#d074-run-the-full-suite-only-inside-make-merge) later deleted that workflow.
- [x] Drop the hosted run of the full and coverage suites that was to confirm the install step and the 45-minute job timeout, because [D074](decisions.md#d074-run-the-full-suite-only-inside-make-merge) deleted the workflow.
- [x] Drop building the worker image in the hosted workflow, because [D074](decisions.md#d074-run-the-full-suite-only-inside-make-merge) deleted the workflow.
- [x] Wait for each docker command the worker has started, so a host exception or a terminal Ctrl-C during `docker create` or `docker rm` leaves no container ([D047](decisions.md#d047-run-generated-signal-code-in-a-locked-down-container)).
- [x] Register portable packages through SurrealDB with a byte-identical round trip, per increment 2 of the [SurrealDB plan](surrealdb-plan.md#9-delivery-increments) and [D039](decisions.md#d039-register-verified-packages-in-surrealdb-and-keep-their-original-bytes-in-an-object-store).
- [x] Find and remove object-store files that no registry row references, per [D039](decisions.md#d039-register-verified-packages-in-surrealdb-and-keep-their-original-bytes-in-an-object-store).
- [ ] Replace the local object store with S3, with its own rule for removing unreferenced objects while an import runs.
- [x] Persist the [resource ledger](../src/quant_research/accounting.py) as numbered canonical JSON events replayed through its own checks, per [D016](decisions.md#d016-account-for-resources-in-an-append-only-ledger). Registry import waits for the SurrealDB registry.
- [x] Persist the [review-lifecycle events](decisions.md#d019-admit-lessons-only-through-recorded-review-events) in SurrealDB, rebuild statuses and lesson eligibility from them, and store the suspensions that stored invalidations call for at each load, per [D039](decisions.md#d039-register-verified-packages-in-surrealdb-and-keep-their-original-bytes-in-an-object-store).
- [x] Trace a corrected or revoked dataset vintage, an invalid package, or a retracted source to the records that rest on it, and suspend each record left without support, per [D046](decisions.md#d046-suspend-records-whose-support-an-invalidation-removes). The registry stores invalidations and the suspensions that each load of the review log appends.
- [x] Reconcile a replay package against declared reference tolerances and an exact independent recomputation, per [D035](decisions.md#d035-reconcile-packages-against-declared-tolerances-and-an-exact-recomputation).
- [x] Reconcile the first real experiment against independent outputs and agreed tolerances, per [D055](decisions.md#d055-run-the-reference-experiment-through-one-audited-pipeline-against-a-frozen-experiment-file). The researcher accepted the result on 23 October 2026.
- [x] Refuse `reconcile package` with `ReconcileError` when the selected trial has no attempts, per [D077](decisions.md#d077-align-recovery-records-and-pinned-lesson-authority).
- [x] Archive a campaign with no trials without evaluating it, per [D067](decisions.md#d067-archive-each-research-ending-as-a-complete-campaign-snapshot).
- [x] Record the single-branch development baseline through the actual research loop on known-answer, null, and flawed synthetic cases, with three repeats and frozen resources, per [D069](decisions.md#d069-record-a-reusable-development-baseline-through-the-research-loop).
- [ ] Author independent held-out capability tasks and answers under the Phase 0 policy, and import the [D011](decisions.md#d011-seal-capability-evaluation-answers-and-retire-disclosed-tasks) exposure-ledger events from their local file store into the registry.
- [x] Attach model provenance and cutoff qualifications to campaign packages and reports, per [D067](decisions.md#d067-archive-each-research-ending-as-a-complete-campaign-snapshot).
- [ ] Fit the reference experiment's preprocessing and check its labels through [walk-forward splits](../src/quant_research/splits.py) once its data and label horizon are known.
- [x] Review the P04 frozen evaluation file, the 20 excluded months, and the scoped Phase 1 result in [D055](decisions.md#d055-run-the-reference-experiment-through-one-audited-pipeline-against-a-frozen-experiment-file). The researcher accepted them on 23 October 2026, which closed [Q003](open-questions.md#q003-what-independent-baseline-defines-agreement).
- [x] Rename `reference.ReferenceError` to `reference.ReferencePipelineError`, preserving the builtin's distinct meaning, per [D079](decisions.md#d079-name-the-reference-pipelines-domain-error).
- [x] Evaluate the first real experiment's period returns with the [pilot evaluation profile](decisions.md#d015-evaluate-pilot-signals-against-their-recorded-trials), save the result with its parameters, and write those parameters into the trial's specification. Do so only after the [Q003](open-questions.md#q003-what-independent-baseline-defines-agreement) reviewer accepts the profile. P04 did so in [D055](decisions.md#d055-run-the-reference-experiment-through-one-audited-pipeline-against-a-frozen-experiment-file), with the profile and its parameters accepted by an agent under the user's autonomy grant. The researcher accepted them on 23 October 2026, after the run. Replay's fixtures yield no period return series, per [D028](decisions.md#d028-keep-replay-descriptive-and-record-evaluation-parameters-with-each-result).
- [x] Run generated signal code from the campaign runner through the [isolated worker](../src/quant_research/worker.py) when the campaign specification enables it, and append each container run to its run log ([D047](decisions.md#d047-run-generated-signal-code-in-a-locked-down-container)).
- [x] Call the isolated worker from the agent tools and append each container run to its run log ([D047](decisions.md#d047-run-generated-signal-code-in-a-locked-down-container), [D049](decisions.md#d049-expose-evidence-to-agents-through-typed-tools-bound-to-the-run-context)).
- [x] Persist the isolated worker's run records in the registry ([D039](decisions.md#d039-register-verified-packages-in-surrealdb-and-keep-their-original-bytes-in-an-object-store), [D047](decisions.md#d047-run-generated-signal-code-in-a-locked-down-container)).
- [ ] Execute the validated [workflow templates](decisions.md#d017-define-research-workflows-as-validated-versioned-templates) through the Phase 4 orchestrator and compare them with the single-branch baseline.
- [x] Specify and classify the live paper case with [reproduction specifications](decisions.md#d020-fix-the-reproduction-comparison-before-results) once pilot data exist, pinning its [locator generation](decisions.md#d043-locate-paper-elements-in-published-extractions), per [D071](decisions.md#d071-reproduce-table-2s-spread-probit-on-adapted-fred-data). It has no code refs. Its passage refs stay unchecked until chunk generations have a ref form.
- [ ] Index the Phase 3 reference implementation with the [code index](../src/quant_research/code_index.py) once it is selected. Synthetic Git fixtures do not establish that acceptance.

Do not mark the Phase 0 or Phase 1 gate complete until its [master-plan acceptance criteria](master-plan.md) have evidence. Preserve unsuccessful and inconclusive work as research history.
