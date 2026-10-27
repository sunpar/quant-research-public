# Quant research platform master plan

Proposed development roadmap, 21 October 2026, revised 22 October 2026.

For the current implementation, decisions, open questions, and tasks, read [project state](project-state.md). This roadmap defines the longer-term phase gates.

Build a platform that takes a research question through source analysis, signal construction, experiments, verification, and a reproducible research report. Researchers can direct the work, delegate bounded campaigns, and eventually configure swarms that reuse earlier findings. This overview sets the development sequence and completion criteria. Detailed designs, implementation tasks, estimates, and release dates follow when each phase is planned.

## The product we are building

The central unit is a **research campaign**. It binds a question to its sources, permitted data, evaluation rules, models, budget, and human intervention policy. A campaign preserves the history of its hypotheses, experiments, and decisions.

A researcher should be able to ask the platform to reproduce an idea from a paper, compare it with an existing signal, explore extensions, and explain the evidence. The same workflow should accept a manually written signal through a Python or notebook interface.

Every completed campaign produces a development research package containing:

- An answer supported by identifiable sources and experiments.
- The implementation, input versions, and instructions needed to reproduce its results.
- Signal diagnostics, backtests, assumptions, and limitations.
- Successful, rejected, failed, and inconclusive experiments, with reasons for their status.
- Human decisions and proposed follow-up work.

Final confirmation produces a separate restricted report for authorized human review and audit. Ordinary package exports and reusable memory exclude that report and its outcomes. An honest finding of no improvement is a successful platform outcome. Investment usefulness is a separate judgment from whether the platform executed the research correctly.

Related research retains a shared history of trials and information exposure, including human decisions, branch exchanges, and retrieved memory. Confirmation eligibility follows this history. Starting a campaign or revising a protocol cannot make observed data untouched. These rules apply to researcher-operated work from Phase 1 and extend to agents as later phases introduce them.

## Starting point and planning assumptions

The repository holds the project tooling and the local work that [project state](project-state.md) describes. The firm integrations remain to be built.

The explicit requirements are configurable swarms and loops, paper and code understanding, memory and knowledge graphs, human steering, autonomy, multiple model providers, verification, existing quant frameworks, and heterogeneous data access. The imported discussions assume an internal Python environment with firm databases, object storage, a compute scheduler, and an internal model SDK. Those infrastructure assumptions and the actual framework interfaces still need inspection.

The latest stack recommendation supplies the working architecture. These are proposed choices, subject to the Phase 0 integration review:

| Responsibility | Working direction |
| --- | --- |
| Research application | Python backend with FastAPI and a React/TypeScript console, plus a Python interface for researchers |
| Campaign execution | Temporal owns scheduling, recovery, waits, and intervention, on its own supported persistence backend |
| Agent interactions | PydanticAI for typed agent behavior, with a thin adapter to the existing model SDK for provider access and accounting |
| Research record | SurrealDB stores campaigns, trials and attempts, artifact references, claims, findings, review and disclosure events, and admitted memory |
| Source and result artifacts | S3 stores original packages, versioned documents, code snapshots, numerical outputs, and reports; original bytes remain the hash authority |
| Retrieval and knowledge graph | SurrealDB full-text, vector, and graph queries over published evidence generations |
| Numerical work | Adapters invoke the existing data, signal, portfolio, evaluation, and backtesting frameworks on approved compute |

The first baseline run can use current tools. One modular application can serve the initial product, with separate processes and permissions for candidate execution and trusted evaluation.

SurrealDB is the proposed application database. Before the platform relies on it, a short qualification probe pins the server and Python SDK and checks transactions and conflicts, permissions, filtered vector search, and backup and restore. The database preserves imported package identities and never regenerates packages from decoded records. Temporal keeps a separate supported persistence backend; research runs reference its workflow identities without becoming a second execution log. Database progress does not complete the Phase 0 or Phase 1 research gates. The [SurrealDB plan](surrealdb-plan.md) has the detailed design.

## Delivery sequence

Each phase ends with a capability that a researcher or operator can demonstrate. Completion depends on evidence against agreed criteria, rather than a fixed number of weeks.

| Phase | Capability delivered | Depends on |
| --- | --- | --- |
| 0 | An agreed reference campaign and verified integration assumptions | Current research material |
| 1 | A limited Python or notebook pilot that reproduces a trusted experiment | 0 |
| 2 | A durable campaign runner, followed by a research console | 1 |
| 3 | Evidence-backed paper reproduction and context for the pilot's data and code | 1; integrates with 2 |
| 4 | A bounded autonomous research pilot | 2 and the pilot-specific outputs of 3 |
| 5 | Reusable, scoped research memory across campaigns | 4 |
| 6 | Configurable multi-agent research and measured coordination benefits | 4; memory-aware workflows also require 5 |
| 7 | Broader team adoption and sustained research programs | 2 for researcher-operated use; 4 for autonomy; 5 and 6 for their respective features |

Phases 2 and 3 can progress in parallel after their shared experiment records exist. Phases 5 and 6 can also progress in parallel, with memory-aware swarms evaluated after both are ready. Permissions, evaluation integrity, and operational visibility develop throughout the sequence.

The first limited release arrives in Phase 1, the durable workspace in Phase 2, and the autonomous pilot in Phase 4. Team adoption and operational hardening grow with these releases. They do not wait for memory or swarms to demonstrate a benefit.

### Run the local pilot ahead of the phase order

On 22 October 2026 the researcher approved a local pilot track that departs from the dependency table above. [D050](decisions.md#d050-run-the-phase-0-pilot-locally-on-public-us-macro-data) scopes Phase 0 and Phase 1 to public data on one machine, because the firm's systems are not available yet. The [autonomous pilot PR plan](autonomous-pilot-prs.md) then reaches a bounded autonomous session before the Phase 2 console exists and before Phase 3 accepts two providers. That plan lists what the track skips, including SurrealDB persistence of campaign logs and the paper case.

The track changes the order of the work, not the gates. Each phase completes only on the evidence its exit gate names. The local session does not complete Phase 4, whose gate needs a held-out capability comparison that a one-person project cannot supply. Acceptance on public data and local services does not accept the firm frameworks, and the firm port is a separate acceptance. [D075](decisions.md#d075-run-the-local-pilot-track-ahead-of-the-master-plan-phase-order) records the approval.

On 23 October 2026 the researcher accepted the local pilot as Phase 1 on public data and vectorbt, per [D055](decisions.md#d055-run-the-reference-experiment-through-one-audited-pipeline-against-a-frozen-experiment-file). The firm frameworks' Phase 1 acceptance stays open.

## Phase 0. Define the reference campaign

Choose a bounded data domain and two linked reference cases. A researcher-owned signal with known local outputs tests framework integration in Phase 1. A paper and its reference code, where available, test interpretation and reproduction in Phase 3. Establish each baseline independently before autonomous extension so failures remain attributable.

This phase resolves the assumptions that could change the architecture:

- Inspect the actual framework interfaces, data access, historical vintages, compute environment, and model providers.
- Inventory known data sources and prioritize the subset needed for the pilot. Record owners, access constraints, and gaps.
- Record prior exposure from existing research and published results before choosing development and confirmation periods. Agree on costs, numerical tolerances, and acceptance evidence.
- Define development fixtures and a separate set of capability-evaluation tasks spanning known-answer, null, flawed, and paper-reproduction cases. Pilot reference cases are development fixtures.
- Establish which data, internal code, and research outputs each provider may receive, including licensing, retention, and fallback constraints.
- Select a researcher owner, a methodological reviewer, and integration owners. Define which decisions require researcher input.
- Check the proposed orchestration and model integration against representative interfaces before committing to those libraries.

The exit gate is an agreed pilot brief, reference artifacts, prior-exposure record, development and final confirmation policy, provider data policy, and feasible integration path. The evaluation policy identifies eligible periods or records that only development evaluation is currently feasible. Missing data history or framework access must result in a narrower pilot or a named prerequisite. Estimates follow this discovery work.

## Phase 1. Reproduce a trusted experiment

Make the known-answer signal run through the actual data, construction, evaluation, portfolio, and backtest path. A researcher can rerun it through Python or a notebook and reconcile its outputs. This limited release establishes useful reproduction before the campaign console exists.

The core work covers:

- Approved data access with explicit identifiers, units, calendars, historical universe membership, and information availability.
- Stable framework contracts, with clear distinctions between signal quality, portfolio construction, and simulated execution.
- Versioned experiment specifications and artifacts, including code, data, environments, costs, and evaluator versions.
- Portable trial and disclosure records keyed to the research lineage. They distinguish scientific trials from execution retries and can be imported into the Phase 2 registry.
- Trusted metric computation, preprocessing fitted within training windows, and checks that future observations cannot change past signals.
- A pilot evaluation profile covering uncertainty, robustness, incremental value, and selection effects, proportionate to the research performed.

Enforce the shared disclosure rules from the first experiment. Baseline reconciliation uses development data, and the records identify any other observed period. Historical publication, revision, and firm-arrival information need evidence in the underlying data; database timestamps alone are insufficient.

Implement the initial development fixtures and separate capability-evaluation cases in this phase. The exit gate is baseline agreement within declared tolerances, independent recomputation of selected metrics, and rejection of deliberately flawed fixtures. A null signal and an inconclusive result must also produce understandable records. Reproducibility applies to saved inputs and implementations; regenerating identical model responses is a different requirement.

## Phase 2. Deliver a durable research workspace

Deliver the durable campaign runner and registry through the Phase 1 Python interface, then add a focused console. Researchers can submit, inspect, compare, and repeat campaigns as these capabilities become available.

The workspace adds:

- A SurrealDB registry whose first milestone is a portable-package round trip: import a verified package, store its objects and records idempotently, and export bytes that pass the existing verifier. Result acceptance is an explicit event, never the latest or successful attempt by default.
- A versioned campaign specification covering sources, data, evaluation, resource limits, and intervention points, importing Phase 1's research records. Model roles activate only after Phase 3 provider acceptance; their assignments, fallbacks, and upgrades become recorded research-process versions.
- Durable job execution, recovery after interruption, and reconciliation with the compute scheduler.
- Experiment comparisons, evidence navigation, costs, progress, and an initial research package export.
- Pause, cancel, annotate, and recorded steering. Methodology changes create revisions or branches and retain prior results.
- Runtime resource accounting, access controls, diagnostics, backup and restore, and upgrades that preserve research records.
- Provisional findings with source links, scope, and review status. Capture lessons early; only admitted evidence enters reusable research context.

The exit gate is a researcher completing the reference workflow without reconstructing state from logs. Interrupted execution, ambiguous job submission, and repeated completion events preserve one accepted result per logical execution, with every attempt visible. Cancellation and spending limits work through the real compute path.

The console should emphasize questions, evidence, experiments, and decisions. Execution logs support diagnosis.

## Phase 3. Ground research in papers, code, and data

Evaluate the selected paper or reference implementation using the pilot's approved data. Agree on the comparison basis and tolerances before running it, distinguishing reproduction on matched inputs from adaptation to different data. A researcher can inspect the interpretation and trace implementation choices to passages, code revisions, and data definitions. Broader ingestion can continue after this pilot case passes.

This phase adds:

- Paper ingestion that retains originals and locates equations, tables, figures, and claims within them.
- Code understanding tied to repository revisions, relevant symbols, tests, dependencies, and framework usage.
- A searchable pilot data catalog that distinguishes documented datasets, manually validated datasets, and datasets approved for agent use. Agent use requires an adapter that enforces timing and access rules.
- Ingestion that publishes complete, versioned evidence generations. Partially processed output never reaches agent retrieval, and a new parser, chunker, or embedding model creates a new generation instead of rewriting old evidence.
- Retrieval in SurrealDB across exact identifiers, lexical and vector search, and bounded graph expansion, each measured against an exact eligible-corpus baseline. Each source claim remains distinguishable from an experimentally supported finding.
- Approved provider adapters and role configuration, exercised with at least two providers before multi-provider support is accepted.

The exit gate has two independently accepted outputs. The first is approved data and provider adapters with verified policy enforcement and usable context for the selected pilot. The second is the reviewed paper specification, its result against the agreed comparison basis, and source-question checks. The first output can enable autonomous extension of the already validated researcher-owned signal while the paper work continues.

Add separate paper tasks to the capability-evaluation set. Keep the live paper case among development fixtures. Source checks cover ambiguity, contradictions, missing information, and restricted material; the platform must expose uncertainty instead of inventing an input or citation.

Late in this phase, compare SurrealDB Agent Memory with the project-owned schema on a labeled fixture set covering aliases, identifier changes, restatements, late arrivals, contradictions, and failed replications. The service stays optional and off the registry's critical path. It cannot admit lessons, accept outcomes, or receive restricted confirmation material, and a written go or no-go decision closes the comparison.

Point-in-time eligibility applies to every retrieval path, including graph edges, aliases, summaries, caches, and exports. Strict replay uses a pinned corpus and retrieval generation and preserves the context actually supplied. Source ingestion and retrieval enforce the same data permissions as execution. Supplied documents and repositories cannot grant tools or access. The provider review covers tool behavior, structured outputs, failures, cancellation, usage accounting, and permitted outbound content. Source assertions and provisional lessons retain their status until reviewed for reuse.

## Phase 4. Complete a bounded autonomous campaign

Connect the proven components into one research loop that proposes, constructs, evaluates, interprets, revises, and stops. Its selected reference case, data context, and provider adapters must pass their gates. The other reference case and broader ingestion can continue in parallel. Begin with a single branch and approved signal operations, with independent critique supplementing executable verification. The local pilot implements one held-back shortlist evaluation and a separate provider critique, with a fresh same-provider fallback, per [D066](decisions.md#d066-interpret-held-back-results-with-a-separate-critique). These results remain development evidence. The local pilot also archives each supported campaign ending and reproduces the P16 numerical artifacts from an exported package alone, per [D067](decisions.md#d067-archive-each-research-ending-as-a-complete-campaign-snapshot). The supplied-candidate development baseline passed P19, per [D069](decisions.md#d069-record-a-reusable-development-baseline-through-the-research-loop). The Phase 4 gate stays open until an independent held-out capability comparison exists.

Agents reach evidence through a small set of typed tools. Authorization, cutoff, corpus generation, and admissible memory come from the trusted run context, so a model cannot widen them. Researchers can choose checkpoints or allow the campaign to continue within its configured permissions and limits. Fix development feedback and repeated-search evaluation rules before search begins. The runtime owns stopping and resource allocation. Scope or methodology changes become recorded decisions.

The pilot exit uses approved operators. Generated Python is a separately gated extension of this phase, developed when the research requires it. Its workers receive limited inputs and no evaluator or control-plane credentials. Enable that extension only after isolation and output-validation checks pass; it does not delay the first pilot.

The exit gate is a complete development research package, including bounded extensions, unsuccessful trials, and supported conclusions. The pilot must survive interruption, accept steering, stop at its limits, and reproduce its results. Run the single-branch procedure across the capability-evaluation set to establish the baseline for Phases 5 and 6. It must meet Phase 0's acceptance criteria, including rejection of deliberately leaky candidates and recognition of null or inconclusive cases without unsupported positive claims.

Register the frozen research procedure for prospective observation, including when it found no selected candidates. Retain agent transcripts for diagnosis. The local pilot now registers terminal procedures and separates historical initialization from later released scored observations, per [D068](decisions.md#d068-register-campaign-procedures-before-later-released-outcomes). The first real observation remains pending, and registration does not close the independent held-out capability gate.

Final confirmation runs only when an eligible shortlist and confirmation period exist. Freeze the shortlist, preprocessing, training, portfolio rules, and evaluation procedure, then issue the separate restricted report. An empty shortlist remains a valid campaign result and requires no final confirmation run.

Informed follow-up requires a revised protocol with eligible confirmation data.

Record available model-version and training-date information, including unknowns. Reports qualify historical confirmation because a pretrained model may already contain later knowledge even when the platform withheld results. Prospective observation evaluates the frozen procedure on subsequent data.

## Phase 5. Turn completed research into reusable knowledge

Expand the provisional evidence records and reviewed lessons from Phases 2 and 3 into systematic learning across campaigns. Reuse preserves the conditions under which a finding held.

Memory includes source interpretations, hypotheses, experimental findings, failed approaches, and tested procedures. Each entry records its supporting evidence, market or universe, horizon, period, assumptions, and status. An invalid implementation must remain distinguishable from a valid experiment that rejects an idea.

Admitted lessons, corrections, and supersession records live in SurrealDB beside the evidence they cite. Extend admission and review to correction, contradiction, and supersession. Retrieval respects permissions and records which earlier evidence informed a campaign in the shared trial and disclosure history. A corrected dataset or invalidated experiment identifies affected conclusions and removes unsupported lessons from eligible retrieval while preserving history.

Begin comparison once separate completed tasks supply a usable corpus of eligible development evidence. The exit gate is a later campaign retrieving relevant prior attempts, recognizing their scope, and avoiding a known failure. Compare memory enabled and disabled using the capability-evaluation protocol below. Keep policies that improve evidence quality or research effort against agreed criteria.

Researchers can inspect why a lesson was retrieved and challenge it. Automated summaries do not become independent evidence for themselves.

## Phase 6. Add configurable swarms and research workflows

Let researchers configure multiple branches and specialist roles through versioned templates. Initial patterns include competing hypotheses, independent replication, focused critiques, and parallel evidence searches.

Workflow configuration covers model roles, tools, dependencies, branch limits, iteration limits, shared resources, and human checkpoints. The orchestrator authorizes additional work and records how branches exchange evidence. Branch histories connect to the shared research-program history, including changes in models and information access.

The console adds branch comparisons and lineage, with controls to stop, fork, prioritize, and review work. A general visual workflow editor can follow demonstrated demand; validated templates are enough for this phase.

The exit gate is a comparison against the Phase 4 single-branch baseline using the capability-evaluation protocol below. Assess valid findings, replication, duplicate work, researcher effort, elapsed time, and cost. Multi-agent configurations become recommended defaults where the comparison supports them.

Memory-aware workflows also pass Phase 5's checks. A neutral or negative comparison preserves the simpler default and does not block team adoption.

## Phase 7. Broaden adoption and operate sustained programs

Expand the working platform around real research demand, starting with Phase 2's researcher-operated workflow and adding autonomy after Phase 4. Introduce memory and swarms as they pass their own gates. Add data domains, frameworks, and templates in priority order. Track each source's documentation, validation, agent-use approval, and access status.

Team operation adds:

- Shared projects, ownership, review queues, permission changes, and established support responsibilities.
- Deployment, upgrades, disaster recovery, capacity management, and service objectives tested against the intended workload.
- Capability evaluations for model and workflow upgrades, cost reporting, and controls for repeated failures or stalled jobs.
- Recurring research triggered by schedules, new data, revised sources, or signal deterioration. Related runs retain trial and disclosure history, with confirmation eligibility reviewed before reuse.
- Evaluation of the research process across repeated campaigns and prospective observations of frozen procedures.

Recurrence is a scheduling choice. Scheduled campaigns can still use interactive, supervised, or bounded autonomous policies.

The exit gate applies to the capabilities each team adopts: representative programs meet their operational objectives, recovery works, and each dependency has an owner. Researcher-operated adoption can pass after Phase 2. Teams adopting autonomy also continue the prospective observations registered in Phase 4. Those conclusions require subsequent data and cannot be promised on an engineering release date.

## How progress will be judged

The platform needs separate measures for execution reliability, research quality, and researcher usefulness. A strong backtest alone cannot establish that all three improved.

| Dimension | Evidence to collect from the pilot onward |
| --- | --- |
| Reproducibility | Baseline agreement, rerun consistency, complete input and implementation records |
| Scientific integrity | Known-error detection, independent numerical checks, supported conclusions, complete trial and disclosure history |
| Research usefulness | Researcher review effort, useful negative findings, replication, incremental value under the agreed evaluation profile |
| System behavior | Recovery, cancellation, permission enforcement, time to complete, cost, and duplicate work |
| Added capabilities | Controlled comparisons for memory, providers, templates, and swarms |

Phase 0 establishes initial acceptance criteria. The capability-evaluation protocol uses separate, versioned tasks with repeated runs and controlled model versions, starting memory, resources, and human steering. Freeze compared configurations before disclosing evaluation outcomes. Keep task answers and artifacts outside live research and memory construction, and retire exposed tasks from future held-out comparisons.

Phase 1 implements initial cases, Phase 3 adds paper cases, and Phase 4 records the single-branch baseline procedure and initial results. Phases 5 and 6 rerun that frozen baseline alongside their additions on the same eligible tasks, using fresh cases when earlier outcomes have informed development. Disclose unavoidable differences. The research owner accepts usefulness and methodological scope; engineering owners demonstrate execution and operational behavior.

## Decisions reserved for detailed planning

The next planning work should resolve the reference campaign and Phase 1 implementation. It needs the actual framework names and interfaces, suitable datasets and vintages, approved compute and providers, evaluation ownership, and the pilot researchers.

Later designs settle exact schemas, framework adapters, model choices, candidate languages, statistical procedures, deployment sizing, and workflow authoring. Each phase plan should specify its smallest useful release, dependencies, verification evidence, rollout, and recovery. Development runs targeted tests while iterating and the full suite once before merge, using local fixtures; external integration checks remain explicit.

The largest schedule risks are missing historical data, undocumented framework behavior, provider restrictions, and unclear evaluation authority. Resolve those before expanding autonomy. Broader data coverage follows a prioritized inventory rather than delaying the pilot until every source is connected.

A new enterprise catalog, adaptive research allocation, an incremental indexing framework, and extra distributed infrastructure remain optional. Introduce them when measured retrieval, coordination, or workload needs justify them. The existing quant frameworks remain the first integration target.

## Source basis

This roadmap synthesizes the imported platform discussion, architecture request, research report, and later stack proposal. This public copy leaves those ChatGPT exports out.

The [SurrealDB plan](surrealdb-plan.md), dated 22 October 2026, replaces the earlier PostgreSQL storage proposal.

The principal sequencing change is to establish evaluation integrity and experimental provenance before autonomous search. Specific research-paper performance claims in the imported material are not acceptance evidence for this platform. Phase designs must verify any external claim on which an implementation choice depends.
