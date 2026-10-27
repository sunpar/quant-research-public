from __future__ import annotations

import importlib.metadata
import math
import subprocess
import time
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

import pytest
from surrealdb import (
    AlreadyExistsError,
    BlockingWsSurrealConnection,
    InternalError,
    NotAllowedError,
    NotFoundError,
    QueryError,
    RecordID,
    ThrownError,
    ValidationError,
)

from quant_research import store

NAMESPACE = "probe"


@pytest.fixture
def db(server, request):
    with server.connect(NAMESPACE, request.node.name) as connection:
        yield connection


def test_server_and_sdk_report_the_pinned_versions(db):
    assert db.version() == f"surrealdb-{store.SERVER_VERSION}"
    assert importlib.metadata.version("surrealdb") == "2.0.0"


def test_query_returns_one_decoded_result_per_statement(db):
    statements = "RETURN 42; RETURN 'text'; RETURN [true, NULL]; RETURN {ratio: 0.5}"
    assert store.query(db, statements) == [42, "text", [True, None], {"ratio": 0.5}]


def test_statement_failures_raise_instead_of_returning_partial_or_empty_results(db):
    assert db.query("RETURN 1; THROW 'boom'") == 1
    with pytest.RaisesGroup(pytest.RaisesExc(ThrownError, match="statement 2")):
        store.query(db, "RETURN 1; THROW 'boom'")
    with pytest.RaisesGroup(NotFoundError):
        store.query(db, "SELECT * FROM undefined_table")
    with pytest.raises(ValidationError):
        store.query(db, "SELEC 1")


def test_statements_outside_a_transaction_stay_written_when_a_later_one_fails(db):
    store.query(db, "DEFINE TABLE thing SCHEMALESS")
    with pytest.RaisesGroup(ThrownError):
        store.query(db, "CREATE thing:kept; THROW 'late'")
    assert store.query(db, "SELECT VALUE record::id(id) FROM thing") == [["kept"]]


def test_uuid_record_ids_round_trip_and_differ_from_their_string_form(db):
    trial = uuid4()
    created = store.query(
        db,
        "CREATE $id SET trial = $trial RETURN id, trial",
        {"id": RecordID("trial", trial), "trial": trial},
    )
    selected = store.query(
        db, "SELECT id, trial FROM type::record('trial', $trial)", {"trial": trial}
    )
    assert created == selected == [[{"id": RecordID("trial", trial), "trial": trial}]]
    assert store.query(db, "RETURN record::id($id)", {"id": RecordID("trial", trial)}) == [trial]
    store.query(db, "CREATE type::record('trial', $key)", {"key": str(trial)})
    [ids] = store.query(db, "SELECT VALUE id FROM trial")
    assert len(ids) == 2
    assert RecordID("trial", trial) in ids
    assert RecordID("trial", str(trial)) in ids


@pytest.fixture
def new_york_local_time(monkeypatch):
    # On a UTC host, a naive datetime read as UTC and one read as local time look the same.
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_datetimes_return_as_the_same_utc_instant_at_microsecond_precision(db, new_york_local_time):
    instant = datetime(2026, 10, 22, 12, 0, 0, 123456, tzinfo=UTC)
    returned = store.query(
        db,
        "RETURN $aware; RETURN $offset; RETURN $naive; RETURN d'2026-10-22T12:00:00.123456789Z';"
        " RETURN <string> d'2026-10-22T12:00:00.123456789Z'",
        {
            "aware": instant,
            "offset": instant.astimezone(timezone(timedelta(hours=-4))),
            "naive": instant.replace(tzinfo=None),
        },
    )
    assert returned == [instant, instant, instant, instant, "2026-10-22T12:00:00.123456789Z"]
    assert {value.tzinfo for value in returned[:4]} == {UTC}


def test_python_none_writes_none_and_option_fields_reject_null(db):
    store.query(db, "DEFINE TABLE note SCHEMAFULL; DEFINE FIELD label ON note TYPE option<string>")
    store.query(db, "CREATE note:a SET label = $label", {"label": None})
    with pytest.RaisesGroup(InternalError):
        store.query(db, "CREATE note:b SET label = NULL")
    assert store.query(db, "SELECT * FROM note") == [[{"id": RecordID("note", "a")}]]
    assert store.query(db, "CREATE loose:a SET label = NULL") == [
        [{"id": RecordID("loose", "a"), "label": None}]
    ]


def test_decimal_strings_keep_their_scale_while_decimal_parameters_normalize(db):
    store.query(db, "CREATE price:a SET amount = $amount", {"amount": "1.2300"})
    assert store.query(db, "SELECT VALUE amount FROM price:a") == [["1.2300"]]
    [normalized] = store.query(db, "RETURN $amount", {"amount": Decimal("1.2300")})
    assert type(normalized) is Decimal
    assert str(normalized) == "1.23"
    assert store.query(db, "RETURN <decimal> '0.10000000000000000000001'") == [
        Decimal("0.10000000000000000000001")
    ]


def test_failed_and_cancelled_transactions_leave_no_record(server, db, request):
    store.query(db, "DEFINE TABLE attempt SCHEMALESS")
    with pytest.RaisesGroup(QueryError, ThrownError, QueryError):
        store.query(db, "BEGIN; CREATE attempt:a; THROW 'stop'; COMMIT")
    assert store.query(db, "SELECT * FROM attempt") == [[]]
    transaction = db.begin()
    store.query(db, "CREATE attempt:b", txn=transaction)
    assert store.query(db, "SELECT VALUE id FROM attempt", txn=transaction) == [
        [RecordID("attempt", "b")]
    ]
    with server.connect(NAMESPACE, request.node.name) as other:
        assert store.query(other, "SELECT * FROM attempt") == [[]]
    db.cancel(transaction)
    assert store.query(db, "SELECT * FROM attempt") == [[]]


def test_concurrent_updates_conflict_at_the_second_commit(server, db, request):
    store.query(db, "UPSERT counter:x SET n = 0")
    with server.connect(NAMESPACE, request.node.name) as other:
        first, second = db.begin(), other.begin()
        assert store.query(db, "SELECT VALUE n FROM counter:x", txn=first) == [[0]]
        assert store.query(other, "SELECT VALUE n FROM counter:x", txn=second) == [[0]]
        store.query(db, "UPDATE counter:x SET n += 1", txn=first)
        store.query(other, "UPDATE counter:x SET n += 10", txn=second)
        db.commit(first)
        with pytest.raises(QueryError) as conflict:
            other.commit(second)
    assert conflict.value.details == {"kind": "TransactionConflict"}
    assert store.query(db, "SELECT VALUE n FROM counter:x") == [[1]]


def test_snapshot_isolation_allows_write_skew_that_a_guard_record_prevents(server, db, request):
    store.query(db, "DEFINE TABLE acceptance SCHEMALESS; DEFINE TABLE guard SCHEMALESS")
    count = "RETURN count(SELECT * FROM acceptance)"
    with server.connect(NAMESPACE, request.node.name) as other:
        first, second = db.begin(), other.begin()
        assert store.query(db, count, txn=first) == store.query(other, count, txn=second) == [0]
        store.query(db, "CREATE acceptance:a", txn=first)
        store.query(other, "CREATE acceptance:b", txn=second)
        db.commit(first)
        other.commit(second)
        first, second = db.begin(), other.begin()
        store.query(db, "UPSERT guard:trial SET attempt = 'a'", txn=first)
        store.query(other, "UPSERT guard:trial SET attempt = 'b'", txn=second)
        db.commit(first)
        with pytest.raises(QueryError) as conflict:
            other.commit(second)
    assert conflict.value.details == {"kind": "TransactionConflict"}
    assert store.query(
        db,
        "SELECT VALUE record::id(id) FROM acceptance ORDER BY id; SELECT VALUE attempt FROM guard",
    ) == [["a", "b"], ["a"]]


def test_unique_index_rejects_sequential_and_concurrent_duplicates(server, db, request):
    store.query(
        db,
        "DEFINE TABLE artifact SCHEMALESS;"
        " DEFINE INDEX artifact_digest ON artifact FIELDS digest UNIQUE",
    )
    store.query(db, "CREATE artifact:one SET digest = 'first'")
    with pytest.RaisesGroup(pytest.RaisesExc(InternalError, match="artifact_digest")):
        store.query(db, "CREATE artifact:two SET digest = 'first'")
    with pytest.RaisesGroup(AlreadyExistsError):
        store.query(db, "CREATE artifact:one SET digest = 'other'")
    with server.connect(NAMESPACE, request.node.name) as other:
        first, second = db.begin(), other.begin()
        store.query(db, "CREATE artifact:three SET digest = 'second'", txn=first)
        store.query(other, "CREATE artifact:four SET digest = 'second'", txn=second)
        db.commit(first)
        with pytest.raises(QueryError) as conflict:
            other.commit(second)
    assert conflict.value.details == {"kind": "TransactionConflict"}
    assert store.query(db, "SELECT VALUE record::id(id) FROM artifact ORDER BY id") == [
        ["one", "three"]
    ]


def test_export_and_import_preserve_records_edges_traversals_and_unique_index(
    server, db, request, tmp_path, running
):
    store.query(
        db,
        """
        DEFINE TABLE attempt SCHEMAFULL;
        DEFINE FIELD started ON attempt TYPE datetime;
        DEFINE FIELD cost ON attempt TYPE string;
        DEFINE TABLE artifact SCHEMAFULL;
        DEFINE FIELD digest ON artifact TYPE string;
        DEFINE INDEX artifact_digest ON artifact FIELDS digest UNIQUE;
        DEFINE TABLE produced TYPE RELATION IN attempt OUT artifact;
        """,
    )
    store.query(
        db,
        """
        CREATE type::record('attempt', $first)
            SET started = d'2026-10-22T12:00:00.123456789Z', cost = '1.2300';
        CREATE type::record('attempt', $second)
            SET started = d'2026-10-22T12:00:01.000000001Z', cost = '0.0100';
        CREATE artifact:result SET digest = 'sha256:aa';
        CREATE artifact:log SET digest = 'sha256:bb';
        RELATE (type::record('attempt', $first))->produced->artifact:result;
        RELATE (type::record('attempt', $first))->produced->artifact:log;
        RELATE (type::record('attempt', $second))->produced->artifact:result;
        """,
        {"first": uuid4(), "second": uuid4()},
    )
    snapshot = """
        SELECT * FROM attempt ORDER BY id;
        SELECT * FROM artifact ORDER BY id;
        SELECT * FROM produced ORDER BY id;
        SELECT id, ->produced->artifact AS outputs FROM attempt ORDER BY id;
        SELECT VALUE [<string> started, cost] FROM attempt ORDER BY started;
    """
    original = store.query(db, snapshot)
    assert original[4] == [
        ["2026-10-22T12:00:00.123456789Z", "1.2300"],
        ["2026-10-22T12:00:01.000000001Z", "0.0100"],
    ]
    backup = tmp_path / "backup.surql"
    subprocess.run(
        server.transfer_command("export", NAMESPACE, request.node.name, backup), check=True
    )
    missing = tmp_path / "missing.surql"
    subprocess.run(server.transfer_command("export", NAMESPACE, "misspelled", missing), check=True)
    assert "INSERT" not in missing.read_text()
    with running(tmp_path) as fresh:
        subprocess.run(
            fresh.transfer_command("import", NAMESPACE, request.node.name, backup), check=True
        )
        with fresh.connect(NAMESPACE, request.node.name) as restored:
            assert store.query(restored, snapshot) == original
            with pytest.RaisesGroup(pytest.RaisesExc(InternalError, match="artifact_digest")):
                store.query(restored, "CREATE artifact:copy SET digest = 'sha256:aa'")


def test_filtered_hnsw_search_matches_the_exact_eligible_baseline(db):
    store.query(
        db,
        """
        DEFINE TABLE chunk SCHEMAFULL;
        DEFINE FIELD scope ON chunk TYPE string;
        DEFINE FIELD v ON chunk TYPE array<float, 3>;
        DEFINE INDEX chunk_v ON chunk FIELDS v HNSW DIMENSION 3 DIST EUCLIDEAN;
        """,
    )
    chunks = [(f"restricted{i:02}", "restricted", [0.01 * (i + 1), 0.0, 0.0]) for i in range(20)]
    chunks += [(name, "public", [1.0, 1.0, 0.0]) for name in ("public_twin_a", "public_twin_b")]
    chunks += [(f"public{i}", "public", [1.0 + 0.1 * i, 1.0, 0.0]) for i in range(1, 6)]
    for key, scope, vector in chunks:
        store.query(
            db,
            "CREATE type::record('chunk', $key) SET scope = $scope, v = $v",
            {"key": key, "scope": scope, "v": vector},
        )
    origin = [0.0, 0.0, 0.0]
    exact = sorted(
        (math.dist(vector, origin), key) for key, scope, vector in chunks if scope == "public"
    )
    search = (
        "SELECT record::id(id) AS id, vector::distance::knn() AS distance FROM chunk"
        " WHERE scope = 'public' AND v <|{k},{ef}|> $origin ORDER BY distance, id"
    )
    # ef=5 is below the 20 nearer restricted vectors, so filtering only after the search fails.
    for k, ef in ((3, 5), (40, 40)):
        [found] = store.query(db, search.format(k=k, ef=ef), {"origin": origin})
        assert [row["id"] for row in found] == [key for _, key in exact[:k]]
        assert [row["distance"] for row in found] == pytest.approx(
            [distance for distance, _ in exact[:k]], rel=1e-6
        )
    [plan] = store.query(db, "EXPLAIN " + search.format(k=3, ef=5), {"origin": origin})
    assert "KnnScan" in plan
    [unfiltered] = store.query(
        db, "SELECT VALUE record::id(id) FROM chunk WHERE v <|3,40|> $origin", {"origin": origin}
    )
    assert sorted(unfiltered) == ["restricted00", "restricted01", "restricted02"]
    store.query(db, "REMOVE INDEX chunk_v ON chunk")
    assert store.query(db, search.format(k=3, ef=5), {"origin": origin}) == [[]]


READER_ACCESS = """
    DEFINE ACCESS reader ON DATABASE TYPE RECORD
        SIGNIN (SELECT * FROM account WHERE name = $name AND crypto::argon2::compare(pass, $pass))
        DURATION FOR TOKEN 1h, FOR SESSION 1h
"""

RESTRICTED = f"""
    DEFINE TABLE account SCHEMAFULL PERMISSIONS NONE;
    DEFINE FIELD name ON account TYPE string;
    DEFINE FIELD pass ON account TYPE string;
    DEFINE FIELD clearance ON account TYPE string;
    {READER_ACCESS};
    DEFINE TABLE attempt SCHEMAFULL
        PERMISSIONS FOR select WHERE scope = 'public' OR $auth.clearance = 'restricted';
    DEFINE FIELD scope ON attempt TYPE string;
    DEFINE TABLE artifact SCHEMAFULL PERMISSIONS
        FOR select WHERE scope = 'public' OR $auth.clearance = 'restricted'
        FOR create WHERE scope = 'public';
    DEFINE FIELD scope ON artifact TYPE string;
    DEFINE FIELD digest ON artifact TYPE string;
    DEFINE FIELD source ON artifact TYPE option<record<attempt>>;
    DEFINE FIELD note ON artifact TYPE option<string>
        PERMISSIONS FOR select WHERE $auth.clearance = 'restricted';
    DEFINE INDEX artifact_digest ON artifact FIELDS digest UNIQUE;
    CREATE account:bob
        SET name = 'bob', pass = crypto::argon2::generate('bob-pass'), clearance = 'public';
    CREATE account:carol
        SET name = 'carol', pass = crypto::argon2::generate('carol-pass'), clearance = 'restricted';
    CREATE attempt:open SET scope = 'public';
    CREATE attempt:hidden SET scope = 'restricted';
    CREATE artifact:result
        SET scope = 'public', digest = 'sha256:aa', source = attempt:hidden, note = 'internal';
    CREATE artifact:secret SET scope = 'restricted', digest = 'sha256:bb', source = attempt:open;
"""


@pytest.fixture
def restricted(db):
    store.query(db, RESTRICTED)
    return db


def reader(database, name):
    return {
        "namespace": NAMESPACE,
        "database": database,
        "access": "reader",
        "variables": {"name": name, "pass": f"{name}-pass"},
    }


def database_user(database, name):
    return {
        "namespace": NAMESPACE,
        "database": database,
        "username": name,
        "password": f"{name}-pass",
    }


@contextmanager
def signed_in(server, credentials):
    with BlockingWsSurrealConnection(server.url) as connection:
        connection.signin(credentials)
        yield connection


def transfer(server, operation, database, path, *options):
    return subprocess.run(
        [
            store.SERVER_EXECUTABLE,
            operation,
            "--endpoint",
            server.endpoint,
            *options,
            "--namespace",
            NAMESPACE,
            "--database",
            database,
            str(path),
        ],
        capture_output=True,
        text=True,
    )


def test_record_access_signs_in_without_saying_which_credential_was_wrong(
    server, restricted, request
):
    name = request.node.name
    with BlockingWsSurrealConnection(server.url) as anonymous:
        anonymous.use(NAMESPACE, name)
        with pytest.raises(NotAllowedError, match="Anonymous access not allowed"):
            store.query(anonymous, "SELECT * FROM artifact")
        for variables in ({"name": "bob", "pass": "wrong"}, {"name": "nobody", "pass": "bob-pass"}):
            with pytest.raises(NotFoundError, match="No record was returned"):
                anonymous.signin({**reader(name, "bob"), "variables": variables})
        with pytest.raises(InternalError, match="access method does not exist"):
            anonymous.signin({**reader(name, "bob"), "access": "missing"})
    with signed_in(server, reader(name, "bob")) as bob:
        # A statement reads $auth's fields under the user's own permissions; clauses do not.
        assert store.query(
            bob, "RETURN $auth; RETURN $auth.id; RETURN $access; RETURN session::db()"
        ) == [RecordID("account", "bob"), None, "reader", name]
        with pytest.RaisesGroup(NotAllowedError, NotAllowedError):
            store.query(bob, "INFO FOR DB; DEFINE TABLE forged SCHEMALESS")


def test_namespace_and_database_users_stay_within_their_level_and_role(server, restricted, request):
    name = request.node.name
    store.query(
        restricted,
        """
        DEFINE USER viewer ON DATABASE PASSWORD 'viewer-pass' ROLES VIEWER;
        DEFINE USER editor ON DATABASE PASSWORD 'editor-pass' ROLES EDITOR;
        DEFINE USER namespace_owner ON NAMESPACE PASSWORD 'namespace_owner-pass' ROLES OWNER;
        """,
    )
    with BlockingWsSurrealConnection(server.url) as connection:
        for wrong_level in (
            {"namespace": NAMESPACE, "username": "viewer", "password": "viewer-pass"},
            database_user("other", "viewer"),
        ):
            with pytest.raises(NotAllowedError, match="problem with authentication"):
                connection.signin(wrong_level)
    with signed_in(server, database_user(name, "viewer")) as viewer:
        assert store.query(
            viewer,
            "RETURN $auth; RETURN session::db();"
            " SELECT VALUE record::id(id) FROM artifact ORDER BY id",
        ) == [None, name, ["result", "secret"]]
        [info] = store.query(viewer, "INFO FOR DB")
        assert "PASSHASH '[REDACTED]'" in info["users"]["editor"]
        with pytest.RaisesGroup(NotAllowedError, NotAllowedError):
            store.query(viewer, "DEFINE TABLE forged SCHEMALESS; USE DB other")
    with signed_in(server, database_user(name, "editor")) as editor:
        assert store.query(editor, "DEFINE TABLE drafted SCHEMALESS") == [None]
        with pytest.RaisesGroup(NotAllowedError):
            store.query(editor, "DEFINE USER promoted ON DATABASE PASSWORD 'x' ROLES OWNER")
    namespace_owner = {
        "namespace": NAMESPACE,
        "username": "namespace_owner",
        "password": "namespace_owner-pass",
    }
    with signed_in(server, namespace_owner) as owner:
        owner.use(NAMESPACE, name)
        assert store.query(
            owner, "DEFINE USER appointed ON DATABASE PASSWORD 'x' ROLES VIEWER"
        ) == [None]
        with pytest.RaisesGroup(NotAllowedError):
            store.query(owner, "USE NS other")


def test_viewer_writes_are_checked_against_table_permissions_with_no_auth_record(
    server, db, request
):
    store.query(
        db,
        """
        DEFINE TABLE closed SCHEMALESS PERMISSIONS FOR create NONE;
        DEFINE TABLE bare SCHEMALESS;
        DEFINE TABLE open SCHEMALESS PERMISSIONS FOR create FULL;
        DEFINE TABLE owned SCHEMALESS PERMISSIONS FOR create WHERE owner = $auth.id;
        DEFINE USER viewer ON DATABASE PASSWORD 'viewer-pass' ROLES VIEWER;
        DEFINE USER editor ON DATABASE PASSWORD 'editor-pass' ROLES EDITOR;
        """,
    )
    tables = ("closed", "bare", "open", "owned")
    creates = "; ".join(f"CREATE {table}:written" for table in tables)
    with signed_in(server, database_user(request.node.name, "viewer")) as viewer:
        assert store.query(viewer, creates) == [
            [],
            [],
            [{"id": RecordID("open", "written")}],
            [{"id": RecordID("owned", "written")}],
        ]
    store.query(db, "; ".join(f"DELETE {table}" for table in tables))
    with signed_in(server, database_user(request.node.name, "editor")) as editor:
        assert store.query(editor, creates) == [
            [{"id": RecordID(table, "written")}] for table in tables
        ]


def test_select_permissions_hide_restricted_records_fields_and_existence(
    server, restricted, request
):
    with signed_in(server, reader(request.node.name, "bob")) as bob:
        assert store.query(
            bob,
            """
            SELECT * FROM artifact;
            SELECT * FROM artifact:secret;
            RETURN artifact:secret.digest;
            RETURN record::exists(artifact:secret);
            RETURN count(SELECT * FROM artifact);
            """,
        ) == [
            [
                {
                    "id": RecordID("artifact", "result"),
                    "scope": "public",
                    "digest": "sha256:aa",
                    "source": RecordID("attempt", "hidden"),
                }
            ],
            [],
            None,
            False,
            1,
        ]
    with signed_in(server, reader(request.node.name, "carol")) as carol:
        assert store.query(
            carol, "SELECT VALUE [record::id(id), note] FROM artifact ORDER BY id"
        ) == [[["result", "internal"], ["secret", None]]]


def test_traversals_skip_restricted_records_but_edges_and_links_expose_their_ids(
    server, restricted, request
):
    store.query(
        restricted,
        """
        DEFINE TABLE produced TYPE RELATION IN attempt OUT artifact;
        DEFINE TABLE linked TYPE RELATION IN attempt OUT artifact PERMISSIONS FOR select FULL;
        DEFINE TABLE scoped TYPE RELATION IN attempt OUT artifact PERMISSIONS FOR select
            WHERE (in.scope = 'public' AND out.scope = 'public') OR $auth.clearance = 'restricted';
        """,
    )
    for edge in ("produced", "linked", "scoped"):
        store.query(
            restricted,
            f"""
            RELATE attempt:open->{edge}->artifact:result;
            RELATE attempt:open->{edge}->artifact:secret;
            RELATE attempt:hidden->{edge}->artifact:result;
            """,
        )
    with signed_in(server, reader(request.node.name, "bob")) as bob:
        assert store.query(
            bob,
            """
            RETURN attempt:open->produced->artifact;
            RETURN attempt:open->linked->artifact;
            RETURN artifact:result<-linked<-attempt;
            RETURN attempt:open->scoped->artifact;
            SELECT VALUE [in, out] FROM scoped;
            SELECT VALUE out FROM linked WHERE in = attempt:open ORDER BY out;
            SELECT VALUE in FROM linked WHERE out = artifact:result ORDER BY in;
            RETURN artifact:result.source;
            RETURN artifact:result.source.*;
            """,
        ) == [
            [],
            [RecordID("artifact", "result")],
            [RecordID("attempt", "open")],
            [RecordID("artifact", "result")],
            [[RecordID("attempt", "open"), RecordID("artifact", "result")]],
            [RecordID("artifact", "result"), RecordID("artifact", "secret")],
            [RecordID("attempt", "hidden"), RecordID("attempt", "open")],
            RecordID("attempt", "hidden"),
            None,
        ]
    with signed_in(server, reader(request.node.name, "carol")) as carol:
        assert store.query(carol, "RETURN array::sort(attempt:open->scoped->artifact)") == [
            [RecordID("artifact", "result"), RecordID("artifact", "secret")]
        ]


def test_hnsw_search_applies_select_permissions_within_the_search(server, restricted, request):
    store.query(
        restricted,
        """
        DEFINE TABLE chunk SCHEMAFULL PERMISSIONS FOR select WHERE scope = 'public';
        DEFINE FIELD scope ON chunk TYPE string;
        DEFINE FIELD v ON chunk TYPE array<float, 3>;
        DEFINE INDEX chunk_v ON chunk FIELDS v HNSW DIMENSION 3 DIST EUCLIDEAN;
        """,
    )
    chunks = [(f"restricted{i:02}", "restricted", [0.01 * (i + 1), 0.0, 0.0]) for i in range(20)]
    chunks += [(f"public{i}", "public", [1.0 + 0.1 * i, 1.0, 0.0]) for i in range(1, 6)]
    for key, scope, vector in chunks:
        store.query(
            restricted,
            "CREATE type::record('chunk', $key) SET scope = $scope, v = $v",
            {"key": key, "scope": scope, "v": vector},
        )
    origin = [0.0, 0.0, 0.0]
    exact = sorted(
        (math.dist(vector, origin), key) for key, scope, vector in chunks if scope == "public"
    )
    search = (
        "SELECT record::id(id) AS id, vector::distance::knn() AS distance FROM chunk"
        " WHERE v <|{k},{ef}|> $origin ORDER BY distance, id"
    )
    [nearest] = store.query(restricted, search.format(k=3, ef=40), {"origin": origin})
    assert [row["id"] for row in nearest] == ["restricted00", "restricted01", "restricted02"]
    with signed_in(server, reader(request.node.name, "bob")) as bob:
        # ef=5 is below the 20 nearer restricted vectors, so a check only after the search fails.
        for k, ef in ((3, 5), (40, 40)):
            [found] = store.query(bob, search.format(k=k, ef=ef), {"origin": origin})
            assert [row["id"] for row in found] == [key for _, key in exact[:k]]
            assert [row["distance"] for row in found] == pytest.approx(
                [distance for distance, _ in exact[:k]], rel=1e-6
            )


def test_denied_writes_return_no_rows_unless_the_permission_throws(server, restricted, request):
    store.query(
        restricted,
        """
        DEFINE TABLE ledger SCHEMAFULL PERMISSIONS
            FOR select FULL
            FOR create WHERE IF writer = $auth { true } ELSE { THROW 'create denied' }
            FOR update WHERE IF writer = $auth { true } ELSE { THROW 'update denied' }
            FOR delete WHERE THROW 'delete denied';
        DEFINE FIELD writer ON ledger TYPE record<account>;
        DEFINE FIELD n ON ledger TYPE int DEFAULT 0;
        DEFINE TABLE entry SCHEMAFULL PERMISSIONS FOR select, create FULL;
        DEFINE FIELD writer ON entry TYPE record<account> ASSERT $value = $auth;
        CREATE ledger:theirs SET writer = account:carol;
        """,
    )
    artifacts = "SELECT * FROM artifact ORDER BY id"
    before = store.query(restricted, artifacts)
    with signed_in(server, reader(request.node.name, "bob")) as bob:
        assert store.query(
            bob,
            """
            CREATE artifact:forged SET scope = 'restricted', digest = 'sha256:cc';
            INSERT INTO artifact {id: artifact:inserted, scope: 'restricted', digest: 'sha256:dd'};
            UPDATE artifact:result SET scope = 'restricted';
            UPSERT artifact:result SET digest = 'sha256:ee';
            DELETE artifact:result;
            """,
        ) == [[], [], [], [], []]
        assert store.query(
            bob,
            "CREATE ledger:mine SET writer = $auth; UPDATE ledger:mine SET n = 1;"
            " CREATE entry:mine SET writer = $auth RETURN VALUE record::id(id)",
        ) == [
            [{"id": RecordID("ledger", "mine"), "writer": RecordID("account", "bob"), "n": 0}],
            [{"id": RecordID("ledger", "mine"), "writer": RecordID("account", "bob"), "n": 1}],
            ["mine"],
        ]
        with pytest.RaisesGroup(
            pytest.RaisesExc(ThrownError, match="create denied"),
            pytest.RaisesExc(ThrownError, match="create denied"),
            pytest.RaisesExc(ThrownError, match="update denied"),
            pytest.RaisesExc(ThrownError, match="delete denied"),
            pytest.RaisesExc(InternalError, match=r"must conform to: \$value = \$auth"),
        ):
            store.query(
                bob,
                """
                CREATE ledger:forged SET writer = account:carol;
                INSERT INTO ledger {id: ledger:inserted, writer: account:carol};
                UPDATE ledger:theirs SET n = 5;
                DELETE ledger:mine;
                CREATE entry:forged SET writer = account:carol;
                """,
            )
    assert store.query(restricted, artifacts) == before
    assert store.query(
        restricted,
        "SELECT VALUE [record::id(id), record::id(writer), n] FROM ledger ORDER BY id;"
        " SELECT VALUE record::id(id) FROM entry",
    ) == [[["mine", "bob", 1], ["theirs", "carol", 0]], ["mine"]]


def test_write_errors_reveal_restricted_record_ids_and_unique_values(server, restricted, request):
    with (
        signed_in(server, reader(request.node.name, "bob")) as bob,
        pytest.RaisesGroup(
            pytest.RaisesExc(
                InternalError, match="already contains 'sha256:bb', with record `artifact:secret`"
            ),
            pytest.RaisesExc(AlreadyExistsError, match="`artifact:secret` already exists"),
        ),
    ):
        store.query(
            bob,
            "CREATE artifact:copy SET scope = 'public', digest = 'sha256:bb';"
            " CREATE artifact:secret SET scope = 'public', digest = 'sha256:cc'",
        )


def test_exports_need_a_system_user_and_ignore_select_permissions(
    server, restricted, request, tmp_path
):
    name = request.node.name
    store.query(
        restricted,
        "DEFINE USER viewer ON DATABASE PASSWORD 'viewer-pass' ROLES VIEWER;"
        " DEFINE USER curator ON DATABASE PASSWORD 'curator-pass' ROLES OWNER",
    )
    with BlockingWsSurrealConnection(server.url) as bob:
        token = bob.signin(reader(name, "bob")).access
    refused = transfer(server, "export", name, tmp_path / "record.surql", "--token", token)
    assert refused.returncode == 1
    assert "403 Forbidden" in refused.stderr
    dump = tmp_path / "viewer.surql"
    viewer = ("--username", "viewer", "--password", "viewer-pass", "--auth-level", "database")
    assert transfer(server, "export", name, dump, *viewer).returncode == 0
    lines = dump.read_text().splitlines()
    assert any("id: artifact:secret" in line and "sha256:bb" in line for line in lines)
    users = {line.split()[2]: line for line in lines if line.startswith("DEFINE USER")}
    assert set(users) == {"viewer", "curator"}
    assert all("PASSHASH '$argon2id$" in line for line in users.values())
    [access] = [line for line in lines if line.startswith("DEFINE ACCESS")]
    assert "KEY '[REDACTED]'" in access


def test_restoring_record_access_needs_a_redefinition_and_keeps_permissions(
    server, restricted, request, tmp_path, running
):
    name = request.node.name
    full, without_access = tmp_path / "full.surql", tmp_path / "without-access.surql"
    root = ("--username", store.ROOT_USERNAME, "--password", server.password)
    assert transfer(server, "export", name, full, *root).returncode == 0
    assert (
        transfer(server, "export", name, without_access, *root, "--accesses", "false").returncode
        == 0
    )
    with BlockingWsSurrealConnection(server.url) as bob:
        token = bob.signin(reader(name, "bob")).access
    with running(tmp_path) as fresh:
        fresh_root = ("--username", store.ROOT_USERNAME, "--password", fresh.password)
        partial = transfer(fresh, "import", "partial", full, *fresh_root)
        assert partial.returncode == 1
        assert "redaction placeholder" in partial.stderr
        with fresh.connect(NAMESPACE, "partial") as restored:
            [info] = store.query(restored, "INFO FOR DB")
            assert info["accesses"] == {}
            assert store.query(
                restored, "SELECT VALUE record::id(id) FROM artifact ORDER BY id"
            ) == [["result", "secret"]]
        with fresh.connect(NAMESPACE, name) as restored:
            store.query(
                restored, "DEFINE USER restorer ON DATABASE PASSWORD 'restorer-pass' ROLES OWNER"
            )
        restorer = (
            "--username",
            "restorer",
            "--password",
            "restorer-pass",
            "--auth-level",
            "database",
        )
        assert transfer(fresh, "import", name, without_access, *restorer).returncode == 0
        with BlockingWsSurrealConnection(fresh.url) as stale:
            with pytest.raises(InternalError, match="access method 'reader' does not exist"):
                stale.authenticate(token)
            with fresh.connect(NAMESPACE, name) as restored:
                store.query(restored, READER_ACCESS)
            with pytest.raises(NotAllowedError, match="problem with authentication"):
                stale.authenticate(token)
        with signed_in(fresh, reader(name, "bob")) as restored_bob:
            assert store.query(restored_bob, "SELECT VALUE record::id(id) FROM artifact") == [
                ["result"]
            ]
