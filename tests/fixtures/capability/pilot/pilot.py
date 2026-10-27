from __future__ import annotations

import hashlib
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path

from quant_research import campaigns, datasets, signals
from quant_research.access import Provider
from quant_research.accounting import Usage
from quant_research.capability import Configuration
from quant_research.catalog import Approved, Catalog, Documented, Evidence, Validated, Vintage
from quant_research.findings import Ledger
from quant_research.providers import Cancelled, ProviderPolicy, RateLimited
from quant_research.records import (
    append_log,
    canonical_bytes,
    format_time,
    parse_fields,
    read_log,
    write_once,
)
from quant_research.sources import SourceDeclaration

PROVIDERS = (Provider("codex-subscription"), Provider("claude-subscription"))
POLICIES = tuple(
    ProviderPolicy(provider=name, labels=frozenset(), retains=True) for name in PROVIDERS
)
AT = datetime(2026, 10, 23, tzinfo=UTC)
WRITTEN = datetime(2025, 1, 1, tzinfo=UTC)
UNRATE, FACTORS = "alfred.unrate", "french.ff3_monthly"
DECLARATION = SourceDeclaration(
    media_type="text/markdown",
    access_domain="public",
    location="agent-authored-capability-development-pilot",
    revision="1",
    acquisition="Agent-authored synthetic development fixture",
    rights="synthetic, unrestricted",
    labels=frozenset(),
    approved_for=frozenset(PROVIDERS),
    published_at=None,
    source_available_at=WRITTEN,
    availability_evidence="Authored synthetic fixture",
    received_at=WRITTEN,
)
PROTOCOL = """You are evaluating one supplied candidate on agent-authored synthetic data.
During search, submit exactly the supplied candidate JSON unchanged as your first experiment.
Run that candidate once. Do not repair, replace, or add candidates, tools, or data.
After its result, conclude immediately. If it was evaluated, cite its actual trial ID in
the conclude trials array. If it was rejected, conclude with an empty trials array.
Do not invent a trial ID.
When asked for a final conclusion, use the final conclusion schema and interpret the
actual evidence. Use no_result for an empty shortlist and cite no trials. Otherwise
cite the actual held-back trial IDs supplied in the evidence.
When asked to critique, use the critique schema and judge the actual evidence.
These synthetic catalog checks are agent checks, not human researcher data validation.
Supplied candidate follows.
"""


def prepare(inputs: dict, directory: Path):
    """Build isolated source, dataset, and numerical stores with a fresh campaign."""
    sources, store, rules = (directory / name for name in ("sources", "datasets", "rules"))
    sources.mkdir()
    rules.mkdir()
    generator = inputs["generator"]
    first = date.fromisoformat(generator["start"])
    events = []
    for name, day, columns in (
        (UNRATE, 6, ("value",)),
        (FACTORS, 1, ("mkt_rf", "smb", "hml", "rf")),
    ):
        rows = []
        for index in range(generator["months"]):
            month = signals.shift(first, index)
            following = signals.shift(month, 1)
            available = format_time(datetime(following.year, following.month, day, tzinfo=UTC))
            values = (
                {"value": generator["predictor"][index % 2]}
                if name == UNRATE
                else {
                    "mkt_rf": generator["market_excess_percent"][index % 2],
                    "smb": "0.00",
                    "hml": "0.00",
                    "rf": "0.00",
                }
            )
            rows.append(
                {
                    "arrived_at": available,
                    "key": month.isoformat(),
                    "published_at": available,
                    "revision": 1,
                    "values": values,
                }
            )
        data = canonical_bytes({"dataset": name, "rows": rows, "schema_version": 1})
        datasets.save(store, name, data)
        checked = Evidence(
            hashlib.sha256(data).hexdigest(),
            "Agent-only synthetic fixture check, no human acceptance",
        )
        vintages = tuple(
            Vintage(aspect, checked) for aspect in ("publication", "revision", "arrival")
        )
        events.extend(
            (
                Documented(
                    name,
                    "agent",
                    AT,
                    "Agent-authored synthetic fixture",
                    columns,
                    (),
                    hashlib.sha256(data).hexdigest(),
                    (),
                    vintages,
                    PROVIDERS,
                ),
                Validated(name, "agent", AT, checked),
                Approved(name, "agent", AT, datasets.ADAPTER),
            )
        )
    rules_data = canonical_bytes(inputs["rules"])
    digest = hashlib.sha256(rules_data).hexdigest()
    (rules / f"{digest}.json").write_bytes(rules_data)
    setup = campaigns.Setup(
        sources=sources,
        datasets=store,
        catalog=Catalog(tuple(events)),
        ledger=Ledger(),
        clearances=frozenset(),
        cutoff=datetime(2025, 6, 1, tzinfo=UTC),
        policies=POLICIES,
        rules=rules,
        packages=directory / "packages",
        lock=Path(__file__).resolve().parents[4].joinpath("uv.lock").read_bytes(),
    )
    candidate = canonical_bytes(inputs["candidate"]).decode()
    prompts = tuple(
        campaigns.register(
            sources,
            name=role,
            role=campaigns.PROMPT,
            data=(PROTOCOL + candidate).encode(),
            declaration=DECLARATION,
        )
        for role in ("researcher", "critic")
    )
    brief = campaigns.register(
        sources,
        name="development-input",
        role="brief",
        data=(PROTOCOL + candidate).encode(),
        declaration=DECLARATION,
    )
    specification = campaigns.CampaignSpecification(
        "What does this supplied candidate establish under the frozen numerical rules?",
        (brief, *prompts, *campaigns.contract_sources(sources, setup.catalog, (UNRATE, FACTORS))),
        tuple(
            campaigns.DataPermission(
                name, datetime(2000, 1, 1, tzinfo=UTC), datetime(2023, 12, 31, tzinfo=UTC)
            )
            for name in (UNRATE, FACTORS)
        ),
        campaigns.EvaluationRules("pilot-evaluation/v1", digest),
        campaigns.ResourceLimits(1, 2, 600, max_model_calls=5, max_wall_seconds=3600),
        (),
        (
            campaigns.ModelRole("researcher", "claude-subscription/claude-opus-5-5", (), "active"),
            campaigns.ModelRole("critic", "codex-subscription/gpt-6.1-sol", (), "active"),
        ),
    )
    return campaigns.start_campaign(
        specification, author="agent", at=AT, reason="Frozen synthetic development protocol"
    ), setup


def configuration(*, implementation_revision: str, runtimes: dict[str, str]) -> Configuration:
    """Freeze source identities, actual role declarations, isolation, and resource policy."""
    from quant_research import capability

    source = Path(capability.__file__)
    digests = {
        str(path.name): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            source,
            Path(__file__),
            Path(__file__).with_name("tasks.json"),
            Path(__file__).resolve().parents[4] / "uv.lock",
        )
    }
    models = tuple(
        canonical_bytes(
            {
                "provider": provider,
                "model": model,
                "version": "unknown",
                "knowledge_cutoff": "unknown",
                "runtime": runtimes[provider],
                "role": role,
                "fallbacks": [],
            }
        ).decode()
        for provider, model, role in (
            ("claude-subscription", "claude-opus-5-5", "researcher"),
            ("codex-subscription", "gpt-6.1-sol", "critic"),
        )
    )
    return Configuration(
        "supplied-candidate-development-v1",
        canonical_bytes({"revision": implementation_revision, "digests": digests}).decode(),
        models,
        "empty; fresh campaign, source, dataset, findings and package stores per cell",
        "one supplied candidate; trial admission 1; "
        "5 dispatches/campaign; 15/case across 3 repeats; 150/effort; no paid fallback; "
        "1200 measured seconds/case including setup and cleanup; external cancel; "
        "campaign wall admission 3600 seconds; cleanup can exceed deadline and fails acceptance",
        "frozen supplied candidate unchanged once; conclude actual trial or empty on rejection; "
        "no repair, replacement, extra candidates, or researcher steering",
    )


class DispatchBudget:
    """Admit actual adapter dispatches durably before they reach either subscription."""

    def __init__(self, path: Path, *, max_calls=150, per_case=15, per_campaign=5):
        self.path = path
        self.policy = {"max_calls": max_calls, "per_case": per_case, "per_campaign": per_campaign}
        path.mkdir(parents=True, exist_ok=True)
        if not write_once(path / ".policy", canonical_bytes(self.policy)):
            raise ValueError("Dispatch budget policy changed")
        self.snapshot()

    def snapshot(self) -> dict:
        events = list(read_log(self.path, "Dispatch event", "Dispatch events"))
        admissions, stopped, retry_at = [], None, None
        for event in events:
            if event.get("action") == "admit":
                value = parse_fields(event, "action case campaign", "dispatch admission")
                admissions.append(value)
            else:
                value = parse_fields(event, "action reason retry_at", "dispatch stop")
                if value["action"] != "stop":
                    raise ValueError("Unknown dispatch event")
                stopped, retry_at = value["reason"], value["retry_at"]
        return {
            "events": events,
            "admissions": admissions,
            "total": len(admissions),
            "cases": dict(Counter(item["case"] for item in admissions)),
            "campaigns": dict(Counter(item["campaign"] for item in admissions)),
            "stopped": stopped,
            "retry_at": retry_at,
        }

    @property
    def stopped(self) -> bool:
        return self.snapshot()["stopped"] is not None

    @property
    def retry_at(self) -> str | None:
        return self.snapshot()["retry_at"]

    def admit(self, case: str, campaign: str) -> bool:
        def attempt():
            state = self.snapshot()
            if state["stopped"] is not None:
                return len(state["events"]), False, None
            reason = None
            if state["total"] >= self.policy["max_calls"]:
                reason = "effort_dispatch_limit"
            elif state["cases"].get(case, 0) >= self.policy["per_case"]:
                reason = "case_dispatch_limit"
            elif state["campaigns"].get(campaign, 0) >= self.policy["per_campaign"]:
                reason = "campaign_dispatch_limit"
            event = (
                {"action": "admit", "case": case, "campaign": campaign}
                if reason is None
                else {"action": "stop", "reason": reason, "retry_at": None}
            )
            return len(state["events"]), reason is None, canonical_bytes(event)

        return append_log(self.path, attempt)

    def stop(self, reason: str, retry_at: str | None = None) -> None:
        def attempt():
            state = self.snapshot()
            event = {"action": "stop", "reason": reason, "retry_at": retry_at}
            return len(state["events"]), None, None if state["stopped"] else canonical_bytes(event)

        append_log(self.path, attempt)

    def adapter(self, adapter, case: str, campaign: str):
        return Counted(adapter, self, case, campaign)


class Counted:
    def __init__(self, adapter, budget: DispatchBudget, case: str, campaign: str):
        self.wrapped, self.budget, self.case, self.campaign = adapter, budget, case, campaign
        self.subscription, self.runtime_version = adapter.subscription, adapter.runtime_version

    def complete(self, request, cancel):
        if not self.budget.admit(self.case, self.campaign):
            cancel.set()
            return Cancelled(Usage())
        reply = self.wrapped.complete(request, cancel)
        if isinstance(reply, RateLimited):
            self.budget.stop("rate_limited", "unknown")
        return reply
