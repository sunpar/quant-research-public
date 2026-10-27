# Revised SurrealDB plan for quant-research

Date: 22 October 2026  
Status: proposed architecture and implementation plan; no repository or database changes made in this review.

## Decision

Use core SurrealDB as the primary application database for the research registry, evidence metadata, native provenance graph, search indexes, review and disclosure events, and admitted research memory. Preserve S3/object storage for original packages and larger artifacts, and preserve the existing numerical data and computation interfaces.

Evaluate SurrealDB Agent Memory as an optional service that proposes and retrieves derived research context. It will not decide package identity, accepted experimental outcomes, memory admission, or access to restricted confirmation material.

The supplied report strengthens this direction. Its examples establish useful design patterns, but they do not establish correctness, performance, or production readiness for this repository's workload. The implementation must earn those conclusions through the gates below.

## Repository basis and scope

This plan uses the repository snapshot already reviewed, especially [project-state.md at 100a517](https://github.com/sunpar/quant-research/blob/100a5172c792c77d755fc439aa6f32aa9581460d/docs/project-state.md), [packages.py](https://github.com/sunpar/quant-research/blob/100a5172c792c77d755fc439aa6f32aa9581460d/src/quant_research/packages.py), and [replay.py](https://github.com/sunpar/quant-research/blob/100a5172c792c77d755fc439aa6f32aa9581460d/src/quant_research/replay.py). The merged [master plan](https://github.com/sunpar/quant-research/blob/2d3bd78fcbf58e87797876eefa97b36a7f8bb577/docs/master-plan.md) defines the phase order.

At that snapshot the implemented application consists of self-contained portable development packages and fixed synthetic causal replay. The database, deployed services, real quant-framework integration, accepted reference experiment, and autonomous runtime remain proposed. Selecting SurrealDB therefore changes the planned storage implementation; it is not a claim that an existing PostgreSQL deployment has been migrated.

The existing Phase 0 and Phase 1 gates remain prerequisites for claims about real research validity. Storage design and a small compatibility probe can proceed independently, but database progress does not complete the real-framework baseline.

The additional input is the user-supplied report, “SurrealDB for Autonomous Research, Agentic Quant Research, and Agentic Knowledge Graphs,” attached as Pasted text.txt. Material product claims were checked against primary documentation on 22 October 2026.

## Changes from the previous proposal

| Area | Revised decision |
| --- | --- |
| Memory product | Add a time-bounded Agent Memory comparison during Phase 3, once the evidence baseline exists. Production cross-campaign reuse still belongs in Phase 5. |
| Historical information | Separate source availability, availability inside a replay, actual recording time, and material disclosed to the agent. A single known_at field is insufficient. |
| Ingestion | Publish complete, versioned evidence generations. Keep partially processed output outside agent retrieval. |
| Identity | Separate original-byte identity, source occurrence, processing version, embedding space, and existing trial/run/attempt identities. |
| Retrieval | Give agents a small set of typed evidence tools with a trusted policy context. Evaluate exact, lexical, vector, and graph stages independently. |
| Integration choice | Keep Python/PydanticAI and the planned Temporal boundary. Treat adapters as optional implementation choices. |
| Qualification | Add a focused SDK/server, transaction, permissions, restore, and filtered-search gate before relying on the storage layer. |

## 1. Responsibilities and authority

| Component | Responsibility | Authority boundary |
| --- | --- | --- |
| Core SurrealDB | Campaign registry, trials and attempts, artifact references, typed relationships, claims, findings, review/admission events, disclosure events, search projections | Authoritative for native application events; preserves imported package identities and provenance |
| Original package and artifact storage | Canonical manifests, exact supplied bytes, logs, retained extraction outputs, larger numerical artifacts | Original evidence remains independently verifiable |
| Existing quant data and computation interfaces | Historical datasets, numerical calculations, causal replay, backtesting and framework-native execution | Numerical results originate from deterministic tools with declared data/code/environment provenance |
| Temporal | Workflow execution, durable waits, retries, intervention and recovery | Owns workflow history through its supported backend |
| Optional Agent Memory service | Extraction/reconciliation experiments and derived retrieval proposals | Cannot independently promote a claim into reviewed memory or modify authoritative evidence |

Temporal's supported persistence systems do not include SurrealDB. The application database choice therefore does not remove Temporal's own persistence dependency. Use a supported service/backend when Temporal is introduced. [Temporal persistence][S1]

SurrealDB application state and Temporal execution history have different owners. A research run may reference a workflow identity and observed status; the application must not create a second scheduler by interpreting those records as an independent execution log.

Similarly, an Agent Memory Context is a separate service/database boundary. Synchronizing selected evidence into it is not part of the core registry transaction. Use a versioned export and a replayable delivery record when the pilot needs synchronization.

## 2. Preserve the evidence contract

The first production storage increment is a package round trip:

1. Open and verify the existing portable package.
2. Preserve its exact canonical manifest and artifact bytes.
3. Upload and verify stored objects.
4. Commit registry metadata and required provenance relationships idempotently.
5. Retrieve the package through its registry identity.
6. Export the original representation and pass the existing verifier.

Preserve trial, lineage, run, attempt, exposure, and artifact identities exactly as the current contract requires. Do not substitute a specification hash for a trial identity. Keep repeat executions distinct from retries, retain explicit retry predecessors, and preserve legitimate retry forks.

The registry must not select an accepted result merely because an attempt was the latest or was successful. Result acceptance is a separate, explicit application event. Rejected, failed, null, and inconclusive outcomes remain queryable.

The original canonical byte stream remains the hash authority. SurrealDB's decoded JSON, record IDs, datetime representation, or floating-point conversions must not be used to regenerate a supposedly identical package. Preserve decimal strings and ordering where the existing package format makes them significant.

Checksums demonstrate integrity relative to the recorded digest. They do not establish immutable storage, truthful source claims, complete disclosure, or a valid research finding. Imported caller-declared exposure coverage stays caller-declared.

Creating a new campaign does not erase prior exposure or restore an already observed confirmation set to an unexposed state.

Native review decisions, admission/retraction history, newly observed disclosures, and retained model-produced extraction outputs are additional authoritative records. They cannot all be rebuilt from the original packages. Back them up along with the registry.

## 3. Domain model

Use typed tables and relations for the semantic core. Allow flexible source-specific metadata only where the application has no stronger contract. These are proposed logical records, not a requirement to create every table in the first increment.

| Record family | Purpose |
| --- | --- |
| Campaign, trial, run, execution attempt | Research organization and the existing execution identities |
| Package version and artifact reference | Original manifest/object digest, storage locator, verification evidence, publication state |
| Source occurrence and source revision | Where bytes came from, source identity, acquisition, rights, revision, and availability evidence |
| Processing generation | Parser, chunker, extractor, model, prompt, configuration and input identities; retained outputs |
| Chunk and embedding revision | Source offsets/page/span, exact text, embedding space and vector version |
| Entity and identifier assignment | Canonical security/company/dataset/code identity; aliases with source and temporal qualification |
| Claim | A proposition attributed to evidence or an agent, including its provenance and uncertainty |
| Finding | A result produced and assessed under a specified evaluation protocol |
| Lesson and admission event | A reviewed, scoped statement eligible for later research reuse |
| Retrieval/context manifest and disclosure event | What was considered, selected, actually supplied, and recorded for a particular run |

Initial relations should answer a small set of concrete questions: which attempt used which data/code; which artifact an attempt produced; which evidence supports or contradicts a claim; which finding supersedes or fails to replicate another; and which reviewed lesson derives from those records.

Claims benefit from explicit records because multiple sources, contradictory evidence, and review history need a shared identity. SurrealDB relation records can themselves carry metadata, so this design is a domain choice rather than a claim that edges cannot express provenance.

An extracted sentence, an agent hypothesis, a numerical result, and an admitted lesson are distinct states. Model confidence is not review approval, statistical confidence, or a calibrated probability that a signal will work.

## 4. Time and causal eligibility

A useful starting vocabulary is:

| Field or event | Meaning |
| --- | --- |
| event_time | When an economic or research event occurred |
| published_at | When the publisher says the source was released |
| source_available_at, with availability evidence | Earliest supported availability under the declared data-access model |
| received_at / recorded_at | Actual arrival and recording in this system |
| valid_from / valid_until | Period for which an assertion is asserted to hold |
| generation and replay admission | Exact version admitted to a corpus or simulation state |
| disclosed_at / submission event | Exact material supplied to a researcher, model, or tool |

The names may adapt to the existing replay contract. Their meanings must stay separate. A future event can be announced today, so event_time is not a universal exclusion criterion. A document can have an old publication date and arrive late. An alias, summary, or claim inferred later must not silently acquire the source's historical publication date as its own observed creation time.

Two supported modes need different guarantees:

- **Reconstruction of an actual run:** use its stored input/context manifests, original evidence versions, and disclosure events. Reproducing the supplied context does not promise identical stochastic model output.
- **Historical simulation:** use an explicit availability policy and a pinned corpus/input generation. Parsing or computing eligible historical inputs today may be allowed by the simulation contract, but the output must be labeled as a replay transformation, with its actual production time retained.

This avoids two errors: excluding every historical backfill because it was ingested today, and pretending a claim generated today was actually known years ago.

Point-in-time eligibility must apply to exact lookups, lexical/vector candidates, graph edges and endpoint versions, aliases, summaries, caches, comparisons, admitted memory, and final exports. Restricted material carries its restriction through all derived representations.

Agent Memory's documented historical graph cutoff does not automatically provide the equivalent passage-level historical view. Its API also allows a caller-supplied observation time for derived facts. The platform therefore needs its own validated availability policy; it cannot treat a memory timestamp as proof of historical access. [Temporal validity][S2]

Historical row filtering also does not guarantee reproducible ranking. A changing corpus can change lexical statistics or approximate search behavior. For strict replay, use the pinned eligible corpus/retrieval generation or a deterministic reference retriever. Always preserve the actual context supplied, so replay does not depend on reconstructing it from today's live index.

No database policy removes knowledge embedded in a pretrained model. Claims about genuinely prospective performance still require the repository's prospective evaluation boundary.

## 5. Ingestion, identity, and publication

### Layered identity

Deduplicate bytes without merging distinct provenance or access rights.

| Identity | Inputs that distinguish it |
| --- | --- |
| Blob | Digest of original bytes |
| Source occurrence | Access domain, source identity/location/revision, acquisition, rights and availability provenance |
| Extraction generation | Source revision, parser/OCR version, configuration and original input digest |
| Chunk generation | Extraction generation, chunker version/configuration, locator and text digest |
| Embedding generation | Chunk input, model/revision, dimensions and normalization configuration |
| Assertion generation | Evidence inputs, extractor/model, prompt/schema version and retained output |
| Publication | Complete output manifest, pipeline version, parent generation and admission event |

The same text at two locations can have two source occurrences pointing to one blob. A changed chunker or embedding model creates a new processing generation. A retry with the same declared identity and different immutable content must fail explicitly.

### Short publication transactions

Download, parse, embed and extract outside the database transaction. Preserve the produced outputs before retrying the final commit. SurrealDB uses snapshot isolation and commit-time conflict detection; named-record guards can protect selected invariants, but there is no blanket serializable predicate isolation. Use uniqueness constraints and explicit revision/guard records where needed, and retry the entire short transaction. [Transactions][S3]

For a normal package, keep required registry metadata and lineage relationships in one transaction. S3 and SurrealDB do not share an atomic transaction. Object verification precedes database visibility; reconciliation handles orphaned objects or interrupted work.

For document pipelines spanning multiple batches:

1. Allocate an ingestion generation and an expected output manifest.
2. Write chunks, vectors, extracted assertions and relationships to that generation in bounded idempotent batches.
3. Validate required output identities, digests, endpoints and completion evidence.
4. Publish the manifest and advance the relevant active-generation pointer in a short guarded transaction.
5. Make every retrieval tool resolve a published generation.

Publication applies to an appropriate document or bounded ingestion batch; it need not rebuild the entire research universe for every new document.

CocoIndex is an optional adapter for incremental derived output. Its upstream connector applies transactions in batches of up to 500 actions per table, not as one transaction across related tables. Its reconciliation can also remove no-longer-declared output. Consequently it must not own historical evidence or the application's publication invariant. Keep schema migrations and retention rules under project control. [CocoIndex connector][S4]

Kreuzberg or another parser can be introduced behind an extraction interface when actual documents require it. Choose the parser for formats, offsets and extraction fidelity; avoid adopting a generic connector's schema as the domain model.

## 6. Retrieval and the agent tool contract

Start with exact identifiers and lexical retrieval, add a vector reference baseline, then measure hybrid retrieval. Use HNSW for the initial approximate-search candidate when the measured workload justifies it. Evaluate DiskANN only when memory/corpus size makes it useful.

The proposed agent-facing operations are:

| Tool | Purpose |
| --- | --- |
| lookup_entity | Resolve an exact identifier or qualified alias |
| search_evidence | Return ranked, eligible evidence excerpts with provenance |
| trace_provenance | Explain how a result or claim derives from data, code and evidence |
| expand_evidence | Follow a small allowlist of relation types with depth and result budgets |
| compare_trials | Compare explicitly selected trial/attempt results under compatible protocols |
| propose_claim | Store a proposed assertion and its evidence links for validation/review |
| run_experiment | Invoke the existing bounded numerical tool contract |

These are application interfaces, not claims about existing repository functions. Typed Python results should carry source occurrence, evidence version, offsets, scope/restriction, availability basis, corpus generation, scores and relationship path where relevant.

Authorization, campaign protocol, cutoff, corpus generation and admissible memory version come from trusted run context. The model cannot broaden these by supplying a different tenant, future cutoff, or raw SurrealQL expression.

For hybrid retrieval:

1. Build eligible lexical and vector candidate lists separately.
2. Explicitly order them by relevance and apply stable tie handling where required.
3. Fuse ranks.
4. Expand only selected evidence through useful relation types, initially one or two hops.
5. Recheck edge, endpoint and final-payload eligibility.
6. Persist the selected payload and provenance before recording the model submission.

Current SurrealDB documentation supports filtered ANN search and index-based prefiltering. Verify the actual query plan, but use adversarial fixtures and an exact eligible-corpus baseline to establish correctness and recall. An EXPLAIN result alone is not sufficient. [Filtered similarity search][S5]

Record the distinction between database candidates, reranker inputs, context selected for the model, and the payload actually submitted. A candidate the model never saw is not automatically an exposure to that model. A failed request with an uncertain submission outcome must preserve that uncertainty.

Retrieval failures must be explicit. Do not convert an unexpected SDK result envelope or query error into an empty evidence list.

## 7. Agent Memory pilot

Run this pilot after the Phase 3 evidence/retrieval baseline exists, in parallel with remaining work. Do not put it on the package registry's critical path.

Agent Memory was publicly introduced on 14 October 2026 and its documentation still identifies pre-GA behavior. The product's own announcement calls out temporal event-ordering weaknesses and benchmark-methodology differences. Treat the pilot as a workload-specific comparison. [Launch][S6] [Contexts][S7]

### Pilot structure

- Seed a clean Context with a finite, versioned export of permitted development evidence.
- Use an application-owned adapter to submit/query the service.
- Retain input/output identities and error traces.
- Accept proposed entities, claims, contradictions and summaries as candidate output.
- Compare that output with the simple project-owned schema on the same questions and source set.
- Route any reusable lesson through the existing review/admission contract.
- Keep production use disabled until the acceptance criteria are met.

Separate Contexts have separate database/key boundaries. Scopes, lenses and grants serve different roles within a Context; profiles, cache and other processing can still be Context-wide. Use clean Contexts for evaluation variants and hard isolation boundaries, and do not send restricted confirmation material to this pilot. [Contexts][S7]

The current product documentation adds concrete compatibility constraints:

| Constraint observed in current docs | Plan response |
| --- | --- |
| Raw passage and structured graph temporal behavior differ | Test both output families; enforce application eligibility |
| Entity matching uses normalized type/name and does not automatically merge aliases | Own financial identifier resolution outside memory |
| Embeddings are currently fixed to Gemini's documented embedding model and dimension | Use only approved inputs/providers; retain provider choice in the core path |
| Python memory client uses the prerelease SDK distribution/extra | Isolate the pilot dependency choice from the core registry |
| Unknown REST fields can be ignored, and field casing differs by route | Validate returned scope/time behavior, not just successful HTTP responses |

Sources: [temporal behavior][S2], [reconciliation][S8], [model settings][S9], [Python memory client][S10], [REST API][S11].

### Evaluation

Use a labeled fixture set with ambiguous tickers, identifier changes, multiple source names for one entity, restatements, late arrivals, contradictory evidence, failed replications, null outcomes, and claims derived after the source date.

Compare four progressively richer approaches: exact/lexical baseline, hybrid retrieval, hybrid plus bounded graph expansion, and the optional memory service. Separate retrieval quality from generation quality by fixing eligible inputs, questions and evaluation rules.

Measure entity resolution, source attribution, contradiction retention, supersession accuracy, temporal exclusion, restricted-data exclusion, useful evidence recall, latency, cost and reviewer correction effort. Record model/provider differences instead of treating unlike configurations as a fair single-number leaderboard.

A memory service that cannot satisfy provider approval, isolation or temporal controls remains an optional research result. Core storage and retrieval continue without it.

## 8. Framework and version choices

Keep Python 3.12 and the planned PydanticAI application model. SurrealDB's PydanticAI example uses a custom retrieval tool, which fits the proposed service boundary. The documented first-party LangGraph adapter is for TypeScript/JavaScript; its presence is not a reason to change this repository's runtime. [PydanticAI integration][S12] [LangGraph integration][S13]

The initial compatibility candidate is SurrealDB 3.3.4 and the documented stable Python SDK 2.0.0. Confirm the installed packages and their actual behavior before pinning the implementation. Current query documentation uses an explicit execution builder, so the report's simplified Python snippet is not production-ready source. [Python SDK][S14] [Query execution][S15]

The probe should cover query execution, result decoding, per-statement errors, UUID/record identities, timestamp and optional-value mapping, decimal preservation, transaction rollback/conflict behavior, and the selected network transport. Run the meaningful integration checks against the actual server/backend choice; an embedded smoke test alone does not establish transport parity.

Increment 1 installed exactly these versions and probed them over WebSocket against a local RocksDB server. [D027](decisions.md#d027-qualify-the-pinned-surrealdb-server-and-client-over-websocket) records the observed behavior and its gaps. [D038](decisions.md#d038-restrict-reads-through-record-access-and-make-denied-writes-throw) records the later permission, record-access, and non-root probes.

Do not copy the report's schema and ingestion sample wholesale. Its per-chunk commits, ID/unique-key mismatch, mutable timestamp upserts and permissive response normalizer do not implement the repository's evidence contract.

## 9. Delivery increments

These are dependency-ordered increments, not calendar estimates. Increment 5 is an optional parallel branch; it does not gate core retrieval or the bounded agent in increment 6.

| Increment | Scope | Acceptance gate |
| --- | --- | --- |
| 0. Record the decision | Revise living architecture/roadmap docs; declare component authority and temporal vocabulary | Planned versus implemented state is explicit; original archive and first-three-increment boundaries remain intact |
| 1. Qualify core storage | Pin candidate server/SDK, migrations and local server configuration; execute a small compatibility/concurrency/restore probe | Actual installed versions and observed semantics recorded |
| 2. Register portable packages | Canonical import/export, original UUIDs, attempts, provenance, native events, object publication | Byte-preserving round trip; duplicate/conflicting/concurrent import; retry forks; interruption recovery |
| 3. Ingest and retrieve evidence | Versioned source occurrences and processing generations; exact/BM25/vector/hybrid retrieval; context manifests | Correct eligible output, traceable excerpts, explicit failure handling, incomplete generations hidden |
| 4. Add targeted knowledge relations | Claims, supporting/contradicting evidence, experiment lineage and bounded traversal | Useful relation questions improve over the retrieval baseline; future/restricted paths stay inaccessible |
| 5. Compare Agent Memory | Isolated, finite pilot alongside late Phase 3 work | Written go/no-go based on temporal correctness, provenance, entity resolution, provider fit, cost and maintenance |
| 6. Integrate the bounded agent | PydanticAI tools and planned Temporal orchestration; durable failures and resource accounting | Recovery and intervention preserve evidence identity, budgets and disclosure history |
| 7. Admit reusable memory | Reviewed lessons, correction/retraction/supersession and cross-campaign retrieval | Memory-enabled versus disabled evaluation under the same research protocol |

The phase mapping stays: Phase 2 registry, Phase 3 retrieval plus optional pilot, Phase 4 bounded autonomous loop, Phase 5 reviewed reuse, Phase 6 measured swarm experiments. Phase 0/1 framework validation remains the research-validity priority.

Operational checks accompany each increment. They are not deferred until all features are built.

## 10. Decisive acceptance cases

| Property | Minimum meaningful case |
| --- | --- |
| Integrity | Existing package survives register/export with identical canonical manifest and artifact bytes |
| Idempotency | Sequential and concurrent duplicate imports converge; conflicting immutable identities are rejected |
| Attempts | Two retry forks survive independently; success does not create implicit acceptance |
| Publication | A failure after object upload or midway through processing exposes no partial published generation |
| Provenance | Identical bytes from distinct sources retain both occurrences and their access/availability evidence |
| Processing revisions | A new chunker, prompt or embedding model cannot silently rewrite an old run's evidence |
| Time | Late publication, delayed arrival, corrected source, future relation and later-derived summary behave according to the declared replay mode |
| Restriction | Exact ID, lexical/vector search, graph expansion, cache and export all reject unauthorized/restricted material |
| Replay | Appending later or unauthorized material does not change retrieval results or assembled context for a replay pinned to an earlier corpus/retrieval generation; the originally submitted payload also remains preserved |
| Disclosure | Retrieved-but-unused, truncated, submitted and uncertain-submission material remain distinguishable |
| Search quality | Filtered ANN is compared with exact eligible-corpus results; narrow scopes retain useful recall |
| Errors | Invalid SDK envelopes and query failures cannot appear as valid empty search results |
| Recovery | Restore preserves native events, UUIDs, edge identities, representative traversals and verified object references |
| Memory | Contradiction, retraction and failed replication remain visible without making an unreviewed lesson authoritative |

For bulk vector loads, separate completeness from steady-state performance. Current index docs describe a compaction period in which pending vectors can still be searched but queries are slower. Benchmark after build/compaction stabilizes, and also measure ingestion-time degradation. Do not block ordinary incremental publication merely because background compaction is briefly active. Pin a tested patch with the documented duplicate-vector and permission-filter fixes. [Index behavior][S16]

No arbitrary throughput or latency pass marks are invented here. Choose the representative corpus, concurrent workload, latency target and budget before measuring, and preserve the measurements with exact versions.

## 11. Repository documentation changes

The follow-on repository change should update the living record together:

- docs/master-plan.md: replace planned PostgreSQL registry/FTS/pgvector/relationship storage with SurrealDB; preserve phase gates and numerical/artifact boundaries.
- docs/architecture.md: component authority, evidence lifecycle, source/generation identities, trusted retrieval context and optional memory adapter.
- docs/decisions.md: add the next unused decision recording this choice and its limits; D010 was next at the inspected snapshot.
- docs/open-questions.md: workload/deployment constraints, provider permissions, availability evidence, memory-pilot criteria and numerical-interface prerequisites.
- docs/todos.md: record the ordered increments with explicit acceptance evidence.
- docs/project-state.md: state that SurrealDB is selected/proposed until implementation and verification actually land.
- docs/initial-prs.md: add a follow-on pointer if useful; retain the original database-free scope of the first three increments.

Preserve imported source material, prior decision history and completed review records. Apply the repository's pstack workflow, model settings and normal verification/review gates to any implementation change.

## 12. Remaining implementation inputs

The design can proceed without choosing every operational detail. Before production qualification, resolve:

1. Real framework/data access and the accepted reference campaign.
2. Availability semantics for each source type and the evidence supporting its timestamps.
3. Representative corpus, vector dimensions, write rate, concurrent runs and latency/cost targets.
4. Approved embedding/extraction providers, including whether the optional memory service is permissible.
5. Hosting/backend, backup/restore expectations and credentials for each hard access boundary.

The immediate deliverable after this plan is the architecture decision and a small core-storage qualification exercise. The first functional database milestone remains the portable-package registry round trip.

## Primary sources

[S1]: https://docs.temporal.io/temporal-service/persistence
[S2]: https://surrealdb.com/docs/agent-memory/reasoning/temporal-validity
[S3]: https://surrealdb.com/docs/learn/querying/concepts-and-guides/transactions
[S4]: https://cocoindex.io/docs/connectors/surrealdb/
[S5]: https://surrealdb.com/docs/learn/data-models/vector-search/similarity-search
[S6]: https://surrealdb.com/blog/introducing-surrealdb-agent-memory
[S7]: https://surrealdb.com/docs/agent-memory/mental-model/contexts-and-scope
[S8]: https://surrealdb.com/docs/agent-memory/reasoning/reconciliation-and-supersession
[S9]: https://surrealdb.com/docs/agent-memory/tuning/models-per-stage
[S10]: https://surrealdb.com/docs/agent-memory/integrations/sdks/python
[S11]: https://surrealdb.com/docs/agent-memory/reference/rest-api
[S12]: https://surrealdb.com/docs/build/integrations/ai-frameworks/pydantic-ai
[S13]: https://surrealdb.com/docs/build/integrations/ai-frameworks/langgraph
[S14]: https://surrealdb.com/docs/reference/python
[S15]: https://surrealdb.com/docs/reference/python/concepts/executing-queries
[S16]: https://surrealdb.com/docs/reference/query-language/statements/define/indexes

- [Temporal persistence][S1]
- [Agent Memory temporal validity][S2]
- [SurrealDB transaction semantics][S3]
- [CocoIndex SurrealDB connector][S4]
- [Filtered similarity search][S5]
- [Agent Memory introduction][S6]
- [Agent Memory Contexts and scope][S7]
- [Reconciliation and supersession][S8]
- [Agent Memory model configuration][S9]
- [Agent Memory Python client][S10]
- [Agent Memory REST contract][S11]
- [PydanticAI integration][S12]
- [LangGraph integration scope][S13]
- [Python SDK compatibility][S14]
- [Python query execution][S15]
- [Index correctness and compaction][S16]

