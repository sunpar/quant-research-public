from __future__ import annotations

import json
import re
import shlex
import sys

SEPARATORS = set(";&|()\n")
ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
REASON = (
    "Merge with `make merge PR=<n>`, which runs the full suite once on the exact commit "
    "that lands. While iterating, run targeted tests: "
    "PYTHONPATH=src uv run python -m pytest tests/<file>.py -k <name>"
)


def forbidden(words: list[str]) -> bool:
    while words and ASSIGNMENT.match(words[0]):
        words = words[1:]
    if words[:3] == ["gh", "pr", "merge"]:
        return True
    return words[:1] == ["make"] and not {"verify", "coverage"}.isdisjoint(words[1:])


def main() -> None:
    lexer = shlex.shlex(
        json.load(sys.stdin)["tool_input"]["command"],
        posix=True,
        punctuation_chars=";&|()<>\n",
    )
    lexer.whitespace_split = True
    lexer.whitespace = " \t\r"
    try:
        tokens = [*lexer, ";"]
    except ValueError:
        # The shell, not this guard, reports unbalanced quotes; heredoc bodies with
        # apostrophes land here too, so they run instead of failing the hook.
        return
    words: list[str] = []
    for token in tokens:
        if token and set(token) <= SEPARATORS:
            if forbidden(words):
                print(f"Blocked: {REASON}", file=sys.stderr)
                sys.exit(2)
            words = []
        else:
            words.append(token)


main()
