# Build the first local research workflow

Plan dated 21 October 2026. This plan turns the [master plan](master-plan.md) into three independently verifiable pull requests. Return to [project state](project-state.md) for architecture, decisions, open questions, and tasks.

## What this batch must demonstrate

An experiment package can explain its inputs, implementation, results, execution attempts, and disclosure history. A deterministic fixture can demonstrate information availability and honest negative or inconclusive outcomes.

These are preparation for Phase 1. They do not complete Phase 0 or Phase 1. Real integration requires inspected framework interfaces, approved historical data, a researcher-owned reference signal, independently established outputs, and agreed numerical tolerances.

## Selected architecture

| Owner | Responsibility |
| --- | --- |
| `quant_research.packages` | Versioned records, artifact integrity, publication, and supplied history |
| `quant_research.replay` | The fixed synthetic timing rules, signal, and diagnostic profile |
| Application CLI | Arguments and presentation through those owners |

The package API has two operations, `save_package` and `open_package`. A saved revision contains all referenced artifacts and supplied prior history. Keep a stable lineage ID alongside distinct trial, run, and attempt IDs. Record that disclosure coverage is caller-declared and confirmation is unavailable. Package records preserve generic outcomes and named artifacts; fixture-specific numerical meaning belongs to replay. Keep no placeholder engine or future adapter framework.

## PR 1. Set up the Python project

Use Python 3.12, `uv`, a dependency lock, pytest, and ruff. This PR also adds a repository runner, `tools/repo.py`, with Git hooks, snapshot verification, and a guarded PR command. [Architecture](architecture.md) describes it. Apart from the verification-speed changes below, the runner is complete; later PRs build research behavior, not more repository tooling.

## PR 2. Preserve and inspect an experiment package

Give a researcher a Python interface and a local CLI to save, inspect, and verify a concrete versioned development package. Start with JSON records and artifact bytes on disk. Save hashes so verification detects missing or changed artifacts. Describe this as versioned, hash-verified evidence; filesystem storage does not make it immutable.

Keep research lineage, scientific trial, and execution attempt identities distinct. A retry retains its trial and earlier attempts. Record data and code versions, environment and evaluator versions, result status, limitations, and prior information exposure. Lineage and disclosure records travel with an exported package so a new campaign cannot silently discard earlier exposure.

Accept successful, rejected, failed, null, and inconclusive records without implying investment usefulness. Inspecting a package must explain its provenance and status. Tests cover round trips, artifact tampering, retries, lineage and exposure preservation, and malformed external records. Keep final confirmation outcomes outside this ordinary development package.

Do not introduce a database, event bus, general registry framework, or empty adapters for unknown numerical engines. Derive the initial schema from this executable consumer and the next fixture replay.

## PR 3. Replay a causal development fixture

Give a researcher one local command that selects inputs available at a declared cutoff, applies a bounded signal operation, computes independently checkable diagnostics, and saves the experiment package from PR 2. Use synthetic data whose expected outputs can be derived independently of the implementation.

Represent observation time, publication or revision time, and arrival time where the fixture needs them. Include a late arrival and a revision. Appending later information must not change a past signal whose available inputs are unchanged. Deliberately flawed or leaky inputs must fail the relevant check.

Include known-answer, null, and inconclusive outcomes. Tests must distinguish invalid execution from a valid experiment that supports no positive conclusion. This numerical profile does not claim portfolio construction, simulated execution, real historical-vintage coverage, or a statistical alpha evaluation.

Use these cases as visible development fixtures. They are not held-out capability-evaluation tasks. Separate evaluation tasks and their answers require the policy and ownership established in Phase 0.

## How the work will proceed

Implement one PR at a time and merge in dependency order. Change the design when an actual probe disproves an assumption, and record material deviations in [decisions](decisions.md).

Test the change being made. While iterating, run the affected test file or selection. Run the full suite once on the final commit before merge, not after every fix or review round. Missing real framework facts remain prerequisites for the later integration PR, rather than guessed interfaces in this batch.

## Make verification fast

Run fewer full suites before making each one faster. In priority order:

1. **Run the full suite once per PR.** Commits and pushes run the smoke suite, and `make pr` validates and pushes without a full run of its own. The one full run is `make verify TARGET=HEAD` on the final commit before merge. Hosted checks stay manual. Nothing enforces that final run, so the PR description records it. Implemented.
2. **Batch changes before that run.** Finish code, fixtures, and documents first. A later production change invalidates a completed run.
3. **Cut fixture setup.** Each repository fixture copies the session template and points `core.hooksPath` at the copied gates instead of reinstalling them, saving about half a second per test. Tests that exercise installation still run `make hooks`, and corruption and isolation tests keep separate writable copies. Fixture commits and pushes skip the hooks they do not test. Implemented.
4. **Parallelize within one suite.** The full suite counts its tests once, then runs four shards that each take every fourth collected test, each started through the same isolated bootstrap. That sidesteps the plugin-autoload and import-path blockers that ruled out pytest-xdist workers, and nested fixture gates still run their own suites. Implemented, per [D044](decisions.md#d044-shard-the-full-suite-and-keep-the-smoke-suite-small).
5. **Skip gate tests that already passed.** The full and smoke suites skip the repository gate tests when an earlier full suite passed their exact inputs. Implemented, per [D072](decisions.md#d072-skip-gate-tests-whose-exact-inputs-already-passed).

## What the real integration needs next

Supply the signal, evaluation, portfolio, and backtesting framework names and accessible code. Identify an approved dataset with historical-vintage evidence, the researcher-owned signal, its expected outputs, the numerical tolerances, and the methodological reviewer. Inspect those inputs before choosing the next adapter and evaluation profile.

The broader service stack remains the master plan's proposed direction. FastAPI, Temporal, model providers, cloud execution, and a console should enter when an inspected use case needs their behavior. These three increments stay database-free; the [SurrealDB registry](surrealdb-plan.md) begins with a package round trip after them.
