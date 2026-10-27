from __future__ import annotations

import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from uuid import UUID

from surrealdb import BlockingWsSurrealConnection, Value
from surrealdb.errors import parse_query_error

SERVER_EXECUTABLE = "surreal-db"
SERVER_VERSION = "3.3.4"
INSTALL_COMMAND = "brew install surrealdb/tap/surreal-db"
ROOT_USERNAME = "root"


@dataclass(frozen=True)
class LocalServer:
    storage: Path
    port: int
    password: str = field(default_factory=lambda: secrets.token_hex(16))

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/rpc"

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start_command(self) -> list[str]:
        return [
            SERVER_EXECUTABLE,
            "start",
            "--bind",
            f"127.0.0.1:{self.port}",
            "--username",
            ROOT_USERNAME,
            "--password",
            self.password,
            "--no-banner",
            "--log",
            "warn",
            f"rocksdb://{self.storage}",
        ]

    def transfer_command(
        self, operation: Literal["export", "import"], namespace: str, database: str, path: Path
    ) -> list[str]:
        return [
            SERVER_EXECUTABLE,
            operation,
            "--endpoint",
            self.endpoint,
            "--username",
            ROOT_USERNAME,
            "--password",
            self.password,
            "--namespace",
            namespace,
            "--database",
            database,
            str(path),
        ]

    @contextmanager
    def connect(self, namespace: str, database: str) -> Iterator[BlockingWsSurrealConnection]:
        # The SDK's HTTP client rejects client-side transactions, so connect over WebSocket.
        with BlockingWsSurrealConnection(self.url) as connection:
            connection.signin({"username": ROOT_USERNAME, "password": self.password})
            connection.use(namespace, database)
            yield connection


def query(
    connection: BlockingWsSurrealConnection,
    surql: str,
    variables: dict[str, Value] | None = None,
    txn: UUID | None = None,
) -> list[Value]:
    """Return every statement result, raising all failed statements together.

    The SDK's own query() returns only the first statement's result and drops later errors.
    """
    response = connection.query_raw(surql, variables, txn_id=txn)
    connection.check_response_for_error(response, "query")
    connection.check_response_for_result(response, "query")
    statements = response["result"]
    errors = []
    for number, statement in enumerate(statements, 1):
        if statement.get("status") != "OK":
            error = parse_query_error(statement)
            error.add_note(f"SurrealQL statement {number}")
            errors.append(error)
    if errors:
        raise ExceptionGroup("SurrealQL statements failed", errors)
    return [statement["result"] for statement in statements]
