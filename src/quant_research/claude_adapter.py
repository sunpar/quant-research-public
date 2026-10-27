"""Call the installed Claude Code CLI as a tool-disabled completion behind the gateway (D057)."""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import KW_ONLY, dataclass, fields
from operator import itemgetter
from pathlib import Path
from types import NoneType
from typing import Any, ClassVar

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
    Schema,
)
from quant_research.records import unique_keys

VERSION = "2.1.301"
PROVIDER = Provider("claude-subscription")
# Consumer plans keep Claude Code data 30 days with model improvement off and 5 years with it
# on (D057), so the route retains content under either setting.
POLICY = ProviderPolicy(provider=PROVIDER, labels=frozenset(), retains=True)

# Without --disable-slash-commands a material that starts with / would run a CLI command.
_FLAGS = (
    "-p",
    "--input-format",
    "stream-json",
    "--output-format",
    "stream-json",
    "--verbose",
    "--tools",
    "",
    "--safe-mode",
    "--strict-mcp-config",
    "--setting-sources",
    "",
    "--disable-slash-commands",
    "--no-session-persistence",
)
_INHERITED = ("HOME", "PATH", "USER", "LOGNAME", "SHELL", "TMPDIR")
# Without CLAUDE_CODE_DISABLE_FILE_MENTIONS the CLI would read each @path in a material from the
# host and attach the file.
_FORCED = {
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_MAX_RETRIES": "2",
    "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "DISABLE_AUTOUPDATER": "1",
    "CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK": "1",
    "CLAUDE_CODE_NO_MODEL_FALLBACK": "1",
    "CLAUDE_CODE_DISABLE_FILE_MENTIONS": "1",
}
# The binary's built-in plugins, the only ones left when no settings source is read.
_PLUGINS = (
    ("cc-plugin-agents-md", "builtin", "cc-plugin-agents-md@builtin"),
    ("cc-plugin-plugin-authoring", "builtin", "cc-plugin-plugin-authoring@builtin"),
)
# init does not list hooks, and a configured hook would run commands that see the request.
_HOOKS = {
    "type": "control_request",
    "request_id": "hooks",
    "request": {"subtype": "get_hooks_listing"},
}
# The CLI answers this turn itself, so it reports the session's inventory before any content
# is written and never reaches a model.
_GATE = {"type": "user", "message": {"role": "user", "content": "/cost"}}
_ANSWER = "Answer the user's message. You have no tools."
_JSON_TYPES = {
    str: "a JSON string",
    int: "a JSON integer",
    float: "a JSON number written with a decimal point, for example 2.0",
    bool: "true or false",
}
_ERRORS: dict[str, tuple[type[Failed | RateLimited], FailureReason]] = {
    "rate_limit": (RateLimited, "rate_limit"),
    "authentication_failed": (Failed, "signed_out"),
    "api_error": (Failed, "api_error"),
}
_INVENTORY_REASONS: dict[str, FailureReason] = {
    "tools": "tools",
    "mcp_servers": "mcp_servers",
    "skills": "skills",
    "plugins": "plugins",
    "api_key_source": "api_key_source",
    "output_style": "output_style",
    "version": "version",
    "model": "model",
}
_COUNTS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
)
_TOKENS = frozenset({"input_tokens", "output_tokens"})
_TICK = 0.05
_GRACE = 2.0


@dataclass(frozen=True)
class ClaudeAdapter:
    """Send cleared content through the installed Claude Code CLI under its subscription sign-in.

    Each call checks claude auth status, then runs one CLI session in an empty temporary
    directory with an allowlisted environment. The session must list no hooks, its first turn
    is /cost, and the request is written only when every system/init event reports exactly the
    expected version, model, and inventory: no tools, MCP servers, or skills, only the built-in
    plugins, and the default output style, since managed settings apply even with no settings
    source. A turn that falls back to another model fails.
    """

    executable: Path = Path("claude")
    _: KW_ONLY
    timeout: float = 300.0
    subscription: ClassVar[bool] = True
    runtime_version: ClassVar[str] = f"claude {VERSION}"

    def complete(self, request: Cleared, cancel: threading.Event) -> Reply:
        """Run one gated CLI session for the request and return its reply.

        The usage is empty until the request is queued and marks its tokens unreported from then
        on, so it never undercounts. A finished real turn's tokens are lower bounds.
        """
        usage = Usage()
        try:
            if cancel.is_set():
                raise _Stop(Cancelled, "cancelled")
            deadline = time.monotonic() + self.timeout
            expected = _Inventory(
                tools=(),
                mcp_servers=(),
                skills=(),
                plugins=_PLUGINS,
                api_key_source="none",
                output_style="default",
                version=VERSION,
                model=request.model.model,
            )
            argv = [
                str(self.executable),
                *_FLAGS,
                "--model",
                request.model.model,
                "--system-prompt",
                _system_prompt(request.schema),
            ]
            with tempfile.TemporaryDirectory() as cwd:
                _require_sign_in(self.executable, cwd, deadline, cancel)
                with _Process(argv, cwd, deadline, cancel) as process:
                    events = process.events()
                    process.queue(_HOOKS)
                    _require_no_hooks(events)
                    process.queue(_GATE)
                    _, gate = _turn(events, expected)
                    if _field(gate, "num_turns", int) != 0:
                        raise _Stop(Failed, "gate_turn")
                    if cancel.is_set():
                        raise _Stop(Cancelled, "cancelled")
                    usage = Usage(unreported=_TOKENS)
                    content = [{"type": "text", "text": text} for text in request.texts]
                    process.queue({"type": "user", "message": {"role": "user", "content": content}})
                    error, result = _turn(events, expected)
                    usage = _usage(result)
                    # Every system/init event reported the requested model, and a fallback fails.
                    return Replied(_verdict(error, result), usage, request.model.model)
        except _Stop as stop:
            return stop.kind(usage, reason=stop.reason)


class _Stop(Exception):
    def __init__(self, kind: type[Failed | RateLimited | Cancelled], reason: FailureReason) -> None:
        super().__init__(reason)
        self.kind, self.reason = kind, reason


@dataclass(frozen=True, kw_only=True)
class _Inventory:
    tools: tuple[str, ...]
    mcp_servers: tuple[str, ...]
    skills: tuple[str, ...]
    plugins: tuple[tuple[str, str, str], ...]
    api_key_source: str
    output_style: str
    version: str
    model: str


class _Process:
    def __init__(self, argv: list[str], cwd: str, deadline: float, cancel: threading.Event) -> None:
        try:
            self._popen = subprocess.Popen(
                argv,
                cwd=cwd,
                env={name: os.environ[name] for name in _INHERITED if name in os.environ} | _FORCED,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            raise _Stop(Failed, "spawn") from None
        os.set_blocking(self._popen.stdin.fileno(), False)
        self._deadline, self._cancel, self._pending = deadline, cancel, b""

    def __enter__(self) -> _Process:
        return self

    def __exit__(self, kind: type[BaseException] | None, *_: object) -> None:
        self._popen.stdin.close()
        if kind is None:
            end = min(time.monotonic() + _GRACE, self._deadline)
            while (
                self._popen.poll() is None
                and time.monotonic() < end
                and not self._cancel.wait(_TICK)
            ):
                pass
        # macOS answers EPERM rather than ESRCH when every member of the group has exited.
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(self._popen.pid, signal.SIGKILL)
        self._popen.wait()
        self._popen.stdout.close()

    def queue(self, message: dict) -> None:
        self._pending += json.dumps(message).encode() + b"\n"

    def read(self) -> bytes:
        stdin, stdout = self._popen.stdin, self._popen.stdout
        while True:
            writing = [stdin] if self._pending else []
            readable, writable, _ = select.select([stdout], writing, [], self._tick())
            if writable:
                try:
                    written = os.write(stdin.fileno(), self._pending)
                except BrokenPipeError:
                    raise _Stop(Failed, "exit") from None
                self._pending = self._pending[written:]
            if readable:
                return os.read(stdout.fileno(), 2**16)

    def events(self) -> Iterator[object]:
        buffer = b""
        while chunk := self.read():
            *lines, buffer = (buffer + chunk).split(b"\n")
            yield from map(_parse, lines)

    def wait(self) -> int:
        while (code := self._popen.poll()) is None:
            time.sleep(self._tick())
        return code

    def _tick(self) -> float:
        if self._cancel.is_set():
            raise _Stop(Cancelled, "cancelled")
        if (remaining := self._deadline - time.monotonic()) <= 0:
            raise _Stop(Failed, "deadline")
        return min(remaining, _TICK)


def _require_sign_in(executable: Path, cwd: str, deadline: float, cancel: threading.Event) -> None:
    with _Process([str(executable), "auth", "status"], cwd, deadline, cancel) as process:
        output = b"".join(iter(process.read, b""))
        code = process.wait()
    with suppress(ValueError, RecursionError):
        match code, json.loads(output, object_pairs_hook=unique_keys):
            case 0, {"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty"}:
                return
    raise _Stop(Failed, "signed_out")


def _system_prompt(schema: Schema | None) -> str:
    if schema is None:
        return _ANSWER
    keys = ", ".join(f"{json.dumps(name)} ({_JSON_TYPES[kind]})" for name, kind in schema)
    return (
        f"{_ANSWER} Reply with exactly one JSON object and nothing else, with no prose and no "
        f"code fence. The object has exactly these keys, in this order: {keys}."
    )


def _parse(line: bytes) -> object:
    try:
        return json.loads(line, object_pairs_hook=unique_keys)
    except (ValueError, RecursionError):
        raise _Stop(Failed, "protocol") from None


def _require_no_hooks(events: Iterator[object]) -> None:
    for event in events:
        match event:
            case {
                "type": "control_response",
                "response": {
                    "subtype": "success",
                    "request_id": "hooks",
                    "response": {"hooks": list() as hooks},
                },
            }:
                if hooks:
                    raise _Stop(Failed, "hooks")
                return
            case {"type": "control_response"}:
                raise _Stop(Failed, "protocol")
            case {"type": str()}:
                pass
            case _:
                raise _Stop(Failed, "protocol")
    raise _Stop(Failed, "exit")


def _turn(events: Iterator[object], expected: _Inventory) -> tuple[str | None, dict]:
    initialized, error = False, None
    for event in events:
        match event:
            case {"type": "system", "subtype": "init"}:
                if (found := _inventory(event)) != expected:
                    differs = (
                        item.name
                        for item in fields(found)
                        if getattr(found, item.name) != getattr(expected, item.name)
                    )
                    raise _Stop(Failed, _INVENTORY_REASONS[next(differs)])
                initialized = True
            # A fallback reruns the turn on a model the call record does not name.
            case {
                "type": "system",
                "subtype": (
                    "model_fallback" | "model_refusal_fallback" | "model_consent_fallback"
                ) as subtype,
            }:
                raise _Stop(Failed, subtype)
            case {"type": "assistant"}:
                error = error or _field(event, "error", str, NoneType)
            case {"type": "result"}:
                if not initialized:
                    raise _Stop(Failed, "protocol")
                return error, event
            case {"type": str()}:
                pass
            case _:
                raise _Stop(Failed, "protocol")
    raise _Stop(Failed, "exit")


def _inventory(event: dict) -> _Inventory:
    try:
        return _Inventory(
            tools=_names(event["tools"]),
            mcp_servers=_names(event["mcp_servers"], "name"),
            skills=_names(event["skills"]),
            plugins=_names(event["plugins"], "name", "path", "source"),
            api_key_source=event["apiKeySource"],
            output_style=event["output_style"],
            version=event["claude_code_version"],
            model=event["model"],
        )
    except (KeyError, TypeError):
        raise _Stop(Failed, "protocol") from None


def _names(items: object, *keys: str) -> tuple:
    if not isinstance(items, list):
        raise _Stop(Failed, "protocol")
    return tuple(sorted(itemgetter(*keys)(item) if keys else item for item in items))


def _verdict(error: str | None, result: dict) -> str:
    # An assistant error decides first, since the CLI also marks that turn's result as an error.
    if error is not None:
        raise _Stop(*_ERRORS.get(error, (Failed, "provider_error")))
    if _field(result, "num_turns", int) == 0:
        raise _Stop(Failed, "local_turn")
    if _field(result, "is_error", bool):
        reason = _field(result, "terminal_reason", str, NoneType)
        _, reason = _ERRORS.get(
            reason or _field(result, "subtype", str), (Failed, "provider_error")
        )
        raise _Stop(Failed, reason)
    return _field(result, "result", str)


def _usage(result: dict) -> Usage:
    usage = _field(result, "usage", dict)
    *inputs, output = (_field(usage, name, int, NoneType) or 0 for name in _COUNTS)
    if min(*inputs, output) < 0:
        raise _Stop(Failed, "protocol")
    return Usage(input_tokens=sum(inputs), output_tokens=output, unreported=_TOKENS)


def _field(event: dict, name: str, *kinds: type) -> Any:
    if type(value := event.get(name)) not in kinds:
        raise _Stop(Failed, "protocol")
    return value
