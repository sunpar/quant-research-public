from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from quant_research import campaigns
from quant_research.access import NO_MEMORY
from quant_research.catalog import Catalog
from quant_research.claude_adapter import POLICY, ClaudeAdapter
from quant_research.codex_adapter import CodexAdapter, policy
from quant_research.findings import Ledger
from quant_research.providers import CallLog, Gateway
from quant_research.records import parse_array, parse_fields, parse_text, parse_time, unique_keys


@dataclass(frozen=True)
class ResearchWorker:
    address: str
    task_queue: str
    setup: campaigns.Setup
    gateway: Gateway
    calls: Path
    archive: Path

    def run(self) -> None:
        """Run the validated configuration on its dedicated Temporal queue."""
        from quant_research import research

        self.setup.packages.mkdir(parents=True, exist_ok=True)
        research.run_worker(
            self.address,
            self.task_queue,
            self.gateway,
            partial(campaigns.resolve, setup=self.setup),
            archive=research.CampaignArchive(self.archive, self.calls),
        )


def _fields(value: object, required: str, optional: str, label: str) -> dict:
    names = set(value) & set(optional.split()) if isinstance(value, dict) else set()
    return parse_fields(value, required + " " + " ".join(names), label)


def _path(base: Path, value: object, label: str, *, directory: bool | None = None) -> Path:
    path = (base / parse_text(value, label)).resolve()
    if directory is True and not path.is_dir():
        raise ValueError(f"{label} must name an existing directory: {path}")
    if directory is False and not path.is_file():
        raise ValueError(f"{label} must name an existing file: {path}")
    return path


def load_worker(path: Path) -> ResearchWorker:
    """Validate JSON v1 and existing inputs before importing Temporal or creating outputs.

    Paths use the configuration's parent; bare executables keep PATH lookup. Version one uses
    an empty findings ledger, NO_MEMORY, no generated code, and no registry importer.
    The calls root must be disjoint from archive and packages after resolving paths.
    """
    path = path.resolve()
    base = path.parent
    value = parse_fields(
        json.loads(path.read_bytes(), object_pairs_hook=unique_keys),
        "schema_version temporal calls archive setup routes",
        "research worker",
    )
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("Unsupported research worker schema version")
    temporal = parse_fields(value["temporal"], "address task_queue", "temporal")
    address = parse_text(temporal["address"], "Temporal address")
    task_queue = parse_text(temporal["task_queue"], "task queue")
    options = _fields(
        value["setup"], "sources datasets catalog packages lock cutoff clearances", "rules", "setup"
    )
    sources = _path(base, options["sources"], "sources", directory=True)
    datasets = _path(base, options["datasets"], "datasets", directory=True)
    catalog = Catalog.from_bytes(_path(base, options["catalog"], "catalog").read_bytes())
    lock = _path(base, options["lock"], "lock").read_bytes()
    packages = _path(base, options["packages"], "packages")
    archive = _path(base, value["archive"], "archive")
    calls = _path(base, value["calls"], "calls", directory=True)
    for name, output in (("archive", archive), ("packages", packages)):
        if calls.is_relative_to(output) or output.is_relative_to(calls):
            raise ValueError(f"calls and {name} roots must be disjoint")
    cutoff = parse_time(options["cutoff"], "cutoff")
    clearances = frozenset(
        parse_text(label, "clearance") for label in parse_array(options["clearances"], "clearances")
    )
    rules = (
        {"rules": _path(base, options["rules"], "rules", directory=True)}
        if "rules" in options
        else {}
    )
    routes = []
    for raw in parse_array(value["routes"], "routes"):
        provider = raw.get("provider") if isinstance(raw, dict) else None
        match provider:
            case "claude-subscription":
                route = _fields(raw, "provider", "executable", "Claude route")
            case "codex-subscription":
                route = _fields(raw, "provider retains", "executable auth", "Codex route")
                if type(route["retains"]) is not bool:
                    raise ValueError("Codex retains must be a boolean")
            case _:
                raise ValueError("Each route must name claude-subscription or codex-subscription")
        executable = parse_text(route.get("executable", provider.split("-")[0]), "executable")
        if "/" in executable:
            executable = str(_path(base, executable, "executable", directory=False))
        if provider == "claude-subscription":
            routes.append((POLICY, ClaudeAdapter(Path(executable))))
        else:
            auth = (
                {"auth": _path(base, route["auth"], "auth", directory=False)}
                if "auth" in route
                else {}
            )
            routes.append((policy(retains=route["retains"]), CodexAdapter(executable, **auth)))
    if not routes:
        raise ValueError("A research worker needs at least one route")
    gateway = Gateway(CallLog(calls), routes)
    setup = campaigns.Setup(
        sources=sources,
        datasets=datasets,
        catalog=catalog,
        ledger=Ledger(),
        clearances=clearances,
        cutoff=cutoff,
        policies=tuple(terms for terms, _ in routes),
        packages=packages,
        lock=lock,
        memory=NO_MEMORY,
        generated=None,
        **rules,
    )
    return ResearchWorker(address, task_queue, setup, gateway, calls, archive)
