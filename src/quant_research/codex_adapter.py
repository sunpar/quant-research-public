"""Call Codex 0.162.1 as a tool-disabled completion through its app-server (D058)."""

from __future__ import annotations

import contextlib
import itertools
import json
import logging
import os
import select
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

from quant_research.access import Provider
from quant_research.accounting import Usage
from quant_research.providers import (
    Cancelled,
    Cleared,
    Failed,
    FailureReason,
    ProviderPolicy,
    RateLimited,
    Replied,
    Reply,
)

_VERSION = "0.162.1"
_CLIENT = {"name": "quant_research", "title": "quant-research", "version": "1"}
_READY_SECONDS = 30.0
_POLL_SECONDS = 0.05
_CLOSE_SECONDS = 2.0
_TERM_SECONDS = 1.0
_KILL_SECONDS = 5.0
_CHATGPT_URL = "https://chatgpt.com/backend-api"
# The gateway takes a float field only as a JSON number with a decimal point.
_INSTRUCTIONS = (
    "Answer the user's request from its text alone. You have no tools, files, network access, "
    "or memory. When a JSON schema constrains the reply, reply with exactly one JSON object "
    "that matches it and nothing else. Write every number with a decimal point, for example "
    "1.0, except in a field the schema types as an integer."
)
# Each name goes to argv as --disable, which fails on an unknown name, and
# experimentalFeature/list must report none of them enabled.
_DISABLED = (
    "shell_tool",
    "stable_environment_tools",
    "multi_agent",
    "multi_agent_v2",
    "code_mode",
    "code_mode_only",
    "code_mode_host",
    "hooks",
    "plugins",
    "apps",
    "connectors",
    "image_generation",
    "imagegenext",
    "browser_use",
    "browser_use_external",
    "computer_use",
    "in_app_browser",
    "memories",
    "memory_tool",
    "goals",
    "agent_message_board",
    "skill_search",
    "deferred_executor",
    "deferred_tool_world_state",
    "standalone_web_search",
    "token_budget",
    "current_time_reminder",
    "sleep_tool",
    "send_message_to_user_async",
    "shell_snapshot",
    "workspace_dependencies",
    "codex_git_commit",
)
# Each override goes to argv as -c, and config/read must echo it back.
_OVERRIDES = (
    ("model_provider", "openai"),
    ("forced_login_method", "chatgpt"),
    # The keyring and auto stores delete auth.json after saving, which would unlink the link.
    ("cli_auth_credentials_store", "file"),
    ("agents.enabled", False),
    ("notify", []),
    ("web_search", "disabled"),
    ("features.skip_host_skill_discovery", True),
    ("project_doc_max_bytes", 0),
    ("developer_instructions", ""),
    ("skills.include_instructions", False),
    ("include_permissions_instructions", False),
    ("include_apps_instructions", False),
    ("include_collaboration_mode_instructions", False),
    ("include_environment_context", False),
)
# config/read in 0.162.1 leaves tools.*.enabled out of its reply, so these go unchecked.
_UNECHOED = (
    ("tools.experimental_request_user_input.enabled", False),
    ("tools.update_plan.enabled", False),
)
_UNSET = ("openai_base_url", "model_providers.openai", "mcp_servers")
_SPOKEN = frozenset({"agentMessage", "reasoning", "userMessage"})
_ERRORS: dict[str, tuple[type[Failed | RateLimited], FailureReason]] = {
    "usageLimitExceeded": (RateLimited, "usage_limit"),
    "rateLimitExceeded": (RateLimited, "rate_limit"),
    "unauthorized": (Failed, "signed_out"),
    "httpConnectionFailed": (Failed, "http_connection_failed"),
}
_STATUSES: dict[str, FailureReason] = {"interrupted": "interrupted"}
_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}
_TOKENS = frozenset({"input_tokens", "output_tokens"})
_LOG = logging.getLogger(__name__)


def policy(*, retains: bool) -> ProviderPolicy:
    """Return the codex-subscription policy, which admits public material only.

    Args:
        retains (bool): Whether the ChatGPT account keeps what it is sent. App-server cannot
            report it, so the campaign setup states it.

    Returns:
        (ProviderPolicy) The policy of provider codex-subscription, with no labels.
    """
    return ProviderPolicy(
        provider=Provider("codex-subscription"), labels=frozenset(), retains=retains
    )


def _signed_in() -> Path:
    """Return Codex's own sign-in file: $CODEX_HOME/auth.json, else ~/.codex/auth.json."""
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "auth.json"


@dataclass(frozen=True)
class CodexAdapter:
    """Send cleared texts as one turn of a fresh, tool-free Codex app-server.

    Each call owns its process, temporary CODEX_HOME, and deadlines, so calls may run
    concurrently. They share only the sign-in file, which Codex refreshes in place.

    Attributes:
        codex (str): Executable name or path of codex-cli 0.162.1.
        auth (Path): Sign-in file that each call's home links to, the only file of the real
            Codex home a call touches.
        turn_seconds (float): Longest wait from turn/start to turn/completed.
        subscription (bool): True, since calls draw on a ChatGPT plan.
        runtime_version (str): The pinned Codex runtime, for provenance.
    """

    codex: str = "codex"
    auth: Path = field(default_factory=_signed_in)
    turn_seconds: float = 600.0
    subscription = True
    runtime_version = f"codex {_VERSION}"

    def complete(self, request: Cleared, cancel: threading.Event) -> Reply:
        """Send the request as one turn, after proving that nothing but the model is reachable.

        Args:
            request (Cleared): Texts to send as one input each, and the reply's schema.
            cancel (threading.Event): Event that ends the call as Cancelled once set.

        Returns:
            (Reply) Replied with the final answer's text unchanged, or Failed, Cancelled, or
            RateLimited. Tokens are unreported lower bounds, and a call that ends before
            turn/start reports no usage. Each call without a reply logs its cause.
        """
        slug = request.model.model
        with tempfile.TemporaryDirectory(prefix="quant-research-codex-") as name:
            # Resolved, so config/read echoes the catalog path exactly: macOS /var is /private/var.
            root = Path(name).resolve()
            try:
                process = _spawn(self, root, slug, cancel)
            except _End as end:
                return _ended(end, _Turn())
            channel = _Channel(process, cancel)
            try:
                thread = _preflight(channel, root, slug, time.monotonic() + _READY_SECONDS)
                # Set before the write, since a partial write may already carry material.
                channel.turn = replace(channel.turn, sent=True)
                deadline = time.monotonic() + self.turn_seconds
                channel.call("turn/start", _turn_params(thread, request), deadline)
                while channel.turn.status is None:
                    channel.receive(deadline)
                # thread/start reported this slug, and a reroute fails the call.
                return replace(_finish(channel.turn), resolved_model=slug)
            except _End as end:
                return _ended(end, channel.turn)
            finally:
                _stop(process)


class _End(Exception):
    """End a call without a reply. Raised anywhere below complete and caught only there."""

    def __init__(
        self, kind: type[Failed | Cancelled | RateLimited], cause: FailureReason, detail: str = ""
    ) -> None:
        super().__init__(kind.__name__, cause, detail)
        self.kind, self.cause, self.detail = kind, cause, detail


@dataclass(frozen=True)
class _Turn:
    """What one call's turn has reported so far, parsed from the wire.

    Attributes:
        sent (bool): turn/start was written, so material may have reached the model.
        final (str | None): Text of the last agentMessage in phase final_answer.
        plain (str | None): Text of the last agentMessage without a phase. Commentary is
            dropped.
        input_tokens (int): Input tokens of the last usage total.
        output_tokens (int): Output tokens of the last usage total.
        status (str | None): Status from turn/completed, None until it arrives.
        error (object): codexErrorInfo of turn/completed, else of the last unretried error.
    """

    sent: bool = False
    final: str | None = None
    plain: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    status: str | None = None
    error: object = None


def _fold(turn: _Turn, message: dict) -> _Turn:
    """Return turn after one notification, or raise _End for a tool or a rerouted model."""
    match message:
        case {
            "method": "item/completed",
            "params": {
                "item": {"type": "agentMessage", "phase": "final_answer", "text": str(text)}
            },
        }:
            return replace(turn, final=text)
        case {
            "method": "item/completed",
            "params": {"item": {"type": "agentMessage", "phase": None, "text": str(text)}},
        }:
            return replace(turn, plain=text)
        case {
            "method": "item/started" | "item/completed",
            "params": {"item": {"type": str(kind)}},
        } if kind not in _SPOKEN:
            raise _End(Failed, "tools", kind)
        case {"method": "model/rerouted"}:
            raise _End(Failed, "rerouted", json.dumps(message.get("params")))
        case {
            "method": "thread/tokenUsage/updated",
            "params": {
                "tokenUsage": {"total": {"inputTokens": int(sent), "outputTokens": int(got)}}
            },
        }:
            return replace(turn, input_tokens=sent, output_tokens=got)
        case {"method": "error", "params": {"error": {"codexErrorInfo": info}, "willRetry": False}}:
            return replace(turn, error=info)
        case {
            "method": "turn/completed",
            "params": {"turn": {"status": str(status), "error": {"codexErrorInfo": info}}},
        } if info is not None:
            return replace(turn, status=status, error=info)
        case {"method": "turn/completed", "params": {"turn": {"status": str(status)}}}:
            return replace(turn, status=status)
    return turn


def _finish(turn: _Turn) -> Replied:
    """Return the reply of a completed turn, or raise _End for any other end."""
    match turn:
        case (
            _Turn(status="completed", final=str(text)) | _Turn(status="completed", plain=str(text))
        ):
            return Replied(text, _usage(turn))
        case _Turn(status="completed"):
            raise _End(Failed, "no_answer")
        case _Turn(status="failed"):
            raise _End(*_classify(turn.error))
    raise _End(Failed, _STATUSES.get(turn.status, "provider_error"), str(turn.status))


def _classify(info: object) -> tuple[type[Failed | RateLimited], FailureReason, str]:
    """Map a codexErrorInfo, from a turn, an error notification, or an RPC error's data."""
    match info:
        case str():
            return *_ERRORS.get(info, (Failed, "provider_error")), info
        case dict() if len(info) == 1:
            name = next(iter(info))
            _, reason = _ERRORS.get(name, (Failed, "provider_error"))
            return Failed, reason, name
    return Failed, "provider_error", "unknown codexErrorInfo"


def _usage(turn: _Turn) -> Usage:
    """Return the turn's tokens as unreported lower bounds, or no usage for a turn never sent.

    Subscription usage undercounts the inferences behind it (D051), as ScriptedProvider's
    subscription mode models.
    """
    if not turn.sent:
        return Usage()
    return Usage(
        input_tokens=turn.input_tokens, output_tokens=turn.output_tokens, unreported=_TOKENS
    )


def _ended(end: _End, turn: _Turn) -> Failed | Cancelled | RateLimited:
    """Log why a call ended without a reply, and return that outcome with the turn's usage."""
    level = logging.INFO if end.kind is Cancelled else logging.WARNING
    _LOG.log(level, "Codex call %s, cause %s: %s", end.kind.__name__, end.cause, end.detail)
    return end.kind(_usage(turn), reason=end.cause)


def _preflight(channel: _Channel, root: Path, slug: str, deadline: float) -> str:
    """Prove that nothing but the model is reachable, and return the thread ID.

    Sends no material. A reply of an unexpected shape fails its check.
    """
    hello = {"clientInfo": _CLIENT, "capabilities": {"experimentalApi": True}}
    match channel.call("initialize", hello, deadline):
        case {"userAgent": str(agent)} if agent.startswith(f"{_CLIENT['name']}/{_VERSION} "):
            pass
        case reply:
            raise _End(Failed, "version", json.dumps(_at(reply, "userAgent")))
    channel.send({"method": "initialized"}, deadline)
    account = channel.call("account/read", {"refreshToken": False}, deadline)
    if _at(account, "account.type") != "chatgpt":
        raise _End(Failed, "signed_out", json.dumps(_at(account, "account.type")))
    reply = channel.call(
        "config/read", {"cwd": str(root / "work"), "includeLayers": False}, deadline
    )
    if wrong := _unsafe(_at(reply, "config"), root):
        raise _End(Failed, "config", ", ".join(wrong))
    for feature in _pages(channel, "experimentalFeature/list", {}, deadline):
        if _at(feature, "name") in _DISABLED and _at(feature, "enabled") is not False:
            raise _End(Failed, "features", str(_at(feature, "name")))
    started = {
        "model": slug,
        "modelProvider": "openai",
        "ephemeral": True,
        "cwd": str(root / "work"),
        "environments": [],
        "dynamicTools": [],
        "allowProviderModelFallback": False,
        "baseInstructions": _INSTRUCTIONS,
        "developerInstructions": "",
        "approvalPolicy": "never",
        "sandbox": "read-only",
    }
    match channel.call("thread/start", started, deadline):
        case {
            "thread": {"id": str(thread), "environments": []},
            "model": model,
            "modelProvider": "openai",
            "instructionSources": [],
        } if model == slug:
            pass
        case reply:
            raise _End(Failed, "thread", json.dumps(reply))
    inventory = {"threadId": thread, "detail": "toolsAndAuthOnly"}
    for server in _pages(channel, "mcpServerStatus/list", inventory, deadline):
        raise _End(Failed, "inventory", json.dumps(_at(server, "name")))
    return thread


def _unsafe(config: object, root: Path) -> list[str]:
    """Return every key of config/read's config that the isolation rules out."""
    # JSON text compares by type as well, since in Python False == 0.
    wrong = [
        key for key, value in _overrides(root) if json.dumps(_at(config, key)) != json.dumps(value)
    ]
    wrong += [key for key in _UNSET if _at(config, key) not in (None, {}, [])]
    url = _at(config, "chatgpt_base_url")
    if not (isinstance(url, str) and url.rstrip("/") == _CHATGPT_URL):
        wrong.append("chatgpt_base_url")
    return wrong


def _at(value: object, path: str) -> object:
    """Return the value at a dotted path through nested objects, or None where a part is missing."""
    for part in path.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def _pages(channel: _Channel, method: str, params: dict, deadline: float) -> Iterator[object]:
    """Yield every item of a paged list, following nextCursor until it is null."""
    cursor = None
    while True:
        match channel.call(method, params | {"cursor": cursor}, deadline):
            case {"data": list(items), "nextCursor": None}:
                yield from items
                return
            case {"data": list(items), "nextCursor": str(cursor)}:
                yield from items
            case reply:
                raise _End(Failed, "protocol", f"{method}: {json.dumps(reply)}")


def _turn_params(thread: str, request: Cleared) -> dict:
    """Return turn/start's params: one text input per cleared text, in order, and the schema.

    Codex sends outputSchema as a strict json_schema format, which takes only a closed object
    whose every property is required.
    """
    params = {
        "threadId": thread,
        "input": [{"type": "text", "text": text, "text_elements": []} for text in request.texts],
    }
    if request.schema is None:
        return params
    schema = {
        "type": "object",
        "properties": {name: {"type": _JSON_TYPES[kind]} for name, kind in request.schema},
        "required": [name for name, _ in request.schema],
        "additionalProperties": False,
    }
    return params | {"outputSchema": schema}


def _spawn(
    adapter: CodexAdapter, root: Path, slug: str, cancel: threading.Event
) -> subprocess.Popen:
    """Lay out a fresh home under root and start one app-server in it.

    root holds home/, whose only entry auth.json links to adapter.auth; catalog.json; and
    work/, the empty cwd. Codex 0.162.1 saves auth.json by truncate and write, so a token
    refresh writes through the link, and a logout unlinks only the link.
    """
    if cancel.is_set():
        raise _End(Cancelled, "cancelled")
    (root / "home").mkdir()
    (root / "home" / "auth.json").symlink_to(adapter.auth.absolute())
    (root / "work").mkdir()
    (root / "catalog.json").write_text(json.dumps(_catalog(slug)))
    env = {
        key: value for key, value in os.environ.items() if not key.startswith(("OPENAI_", "CODEX_"))
    }
    try:
        process = subprocess.Popen(
            _argv(adapter.codex, root),
            bufsize=0,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=root / "work",
            env=env | {"CODEX_HOME": str(root / "home")},
            start_new_session=True,
        )
    except OSError as error:
        raise _End(Failed, "spawn", str(error)) from error
    os.set_blocking(process.stdin.fileno(), False)
    return process


def _overrides(root: Path) -> tuple[tuple[str, object], ...]:
    """Return _OVERRIDES and the catalog path, the one override that varies by call."""
    return (*_OVERRIDES, ("model_catalog_json", str(root / "catalog.json")))


def _argv(codex: str, root: Path) -> list[str]:
    """Return app-server's argv: -c key=value per override, and --disable per feature.

    json.dumps writes a valid TOML value for each type the overrides use: str, bool, int, [].
    """
    overrides = (*_overrides(root), *_UNECHOED)
    # Popen runs in the empty work directory, so a relative path must be made absolute first.
    return [
        str(Path(codex).absolute()) if os.sep in codex else codex,
        "app-server",
        "--listen",
        "stdio://",
        *(arg for key, value in overrides for arg in ("-c", f"{key}={json.dumps(value)}")),
        *(arg for name in _DISABLED for arg in ("--disable", name)),
    ]


def _catalog(slug: str) -> dict:
    """Return a one-model catalog for slug.

    The bundled entry's tool metadata overrides the feature flags, and its base_instructions is
    Codex's agent prompt, so this entry replaces both.
    """
    entry = {
        "slug": slug,
        "display_name": slug,
        "base_instructions": _INSTRUCTIONS,
        "supported_reasoning_levels": [],
        "shell_type": "disabled",
        "visibility": "list",
        "supported_in_api": True,
        "priority": 0,
        "support_verbosity": False,
        "truncation_policy": {"mode": "bytes", "limit": 10000},
        "experimental_supported_tools": [],
        "tool_mode": "direct",
        "include_apps_usage_instructions": False,
    }
    return {"models": [entry]}


def _stop(process: subprocess.Popen) -> None:
    """End the app-server by closing its stdin, and signal its group only if it stays.

    Codex rewrites the linked auth.json by truncate and write, so a signal must never be the
    usual end of a call. Nothing signals after _exited reaps the leader: until then its pid is
    the group's ID and cannot pass to another process.
    """
    process.stdin.close()
    if not _exited(process, _CLOSE_SECONDS):
        # macOS returns EPERM when the unreaped leader is the group's last member.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGTERM)
        if not _exited(process, _TERM_SECONDS):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(_KILL_SECONDS)
    process.stdout.close()


def _exited(process: subprocess.Popen, seconds: float) -> bool:
    """Return whether stdout reached EOF and the leader exited within seconds, reaping it."""
    deadline = time.monotonic() + seconds
    stdout = process.stdout.fileno()
    while (left := deadline - time.monotonic()) > 0 and select.select([stdout], [], [], left)[0]:
        if not os.read(stdout, 1 << 16):
            try:
                process.wait(max(deadline - time.monotonic(), 0))
            except subprocess.TimeoutExpired:
                return False
            return True
    return False


class _Channel:
    """One app-server's JSONL stdio, read and written under a deadline and the caller's cancel.

    Every notification folds into turn as it is read, including one that arrives while a
    request awaits its reply, so what the turn reported survives any end. Requests are
    sequential, each with the next integer ID. A server request ends the call, since a
    tool-free completion answers none.
    """

    def __init__(self, process: subprocess.Popen, cancel: threading.Event) -> None:
        self.turn = _Turn()
        self._stdin = process.stdin.fileno()
        self._stdout = process.stdout.fileno()
        self._cancel = cancel
        self._ids = itertools.count(1)
        self._buffer = b""

    def call(self, method: str, params: dict, deadline: float) -> object:
        """Send a request and return its result. An error reply ends the call."""
        ident = next(self._ids)
        self.send({"id": ident, "method": method, "params": params}, deadline)
        while True:
            match self.receive(deadline, ident):
                case None:
                    continue
                case {"result": result}:
                    return result
                case {"error": dict(error)}:
                    kind, cause, name = _classify(error.get("data"))
                    raise _End(kind, cause, f"{method}: {error.get('message')}, {name}")
                case reply:
                    raise _End(Failed, "protocol", f"{method}: {json.dumps(reply)}")

    def send(self, message: dict, deadline: float) -> None:
        """Write one JSON line to the non-blocking stdin, waiting for room under the deadline."""
        self._check(deadline)
        data = json.dumps(message).encode() + b"\n"
        while data:
            self._wait([], [self._stdin], deadline)
            try:
                data = data[os.write(self._stdin, data) :]
            except BrokenPipeError as error:
                raise _End(Failed, "exit", str(error)) from error

    def receive(self, deadline: float, awaiting: int | None = None) -> dict | None:
        """Consume one message: fold a notification, or return the reply to request awaiting."""
        message = self._message(deadline)
        match message:
            case {"id": _, "method": method}:
                raise _End(Failed, "server_request", str(method))
            case {"method": _}:
                self.turn = _fold(self.turn, message)
                return None
            case {"id": ident} if ident == awaiting:
                return message
        raise _End(Failed, "protocol", json.dumps(message))

    def _message(self, deadline: float) -> object:
        """Return the next line as JSON, checking cancel and the deadline before each one."""
        self._check(deadline)
        while b"\n" not in self._buffer:
            self._wait([self._stdout], [], deadline)
            chunk = os.read(self._stdout, 1 << 16)
            if not chunk:
                raise _End(Failed, "exit")
            self._buffer += chunk
        line, self._buffer = self._buffer.split(b"\n", 1)
        try:
            return json.loads(line)
        except (ValueError, RecursionError) as error:
            raise _End(Failed, "protocol", str(error)) from error

    def _wait(self, reads: list[int], writes: list[int], deadline: float) -> None:
        """Block until a descriptor is ready, checking cancel and the deadline every poll."""
        while not any(select.select(reads, writes, [], _POLL_SECONDS)[:2]):
            self._check(deadline)

    def _check(self, deadline: float) -> None:
        if self._cancel.is_set():
            raise _End(Cancelled, "cancelled")
        if time.monotonic() > deadline:
            raise _End(Failed, "deadline")
