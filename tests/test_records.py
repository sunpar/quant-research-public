from __future__ import annotations

import hashlib
import re
import shutil
from datetime import date
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest

from quant_research.accounting import Usage
from quant_research.capability import (
    Configuration,
    ExposureLedger,
    Response,
    load_answers,
    load_tasks,
    run,
)
from quant_research.model_provenance import ModelRecord
from quant_research.providers import CallLog, CallRecord, Closed, Dispatched, Served
from quant_research.records import (
    ObjectConflict,
    PackageError,
    canonical_bytes,
    keep_object,
    parse_time,
    parse_uuid,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "capability"
MODEL = ModelRecord("fake-a", "model-a", "2026-05-01", date(2026, 1, 31))
CLAIM = Dispatched(
    UUID(int=11),
    UUID(int=4),
    1,
    "proposer",
    MODEL,
    "a" * 64,
    "b" * 64,
    False,
    "scripted",
    UUID(int=13),
)
SENT = CallRecord(
    CLAIM.call_id,
    CLAIM.attempt_id,
    CLAIM.fence,
    CLAIM.role,
    CLAIM.model,
    CLAIM.policy_version,
    CLAIM.request_digest,
    "completed",
    (),
    Usage(model_calls=1, input_tokens=4, output_tokens=2, cost_usd=Decimal("0.000012")),
    hashlib.sha256(b'"Looks fine"').hexdigest(),
    Served("scripted", "model-a"),
)


def _exposure_steps():
    tasks, answers = load_tasks(FIXTURES / "tasks.json"), load_answers(FIXTURES / "answers.json")

    def freeze(ledger, name):
        configuration = Configuration(name, f"tests/{name}", ("model-a",), "empty", "local", "none")
        return ledger.freeze(tasks, answers.key_id, (configuration,), 1)

    def disclose(ledger):
        results = run(freeze(ledger, "honest"), tasks, {"honest": lambda _: Response("null")})
        ledger.disclose(results, answers)

    return (
        lambda ledger: freeze(ledger, "honest"),
        lambda ledger: freeze(ledger, "hedged"),
        disclose,
    )


STORES = {
    "exposure-ledger": (ExposureLedger, _exposure_steps()),
    "call-log": (
        CallLog,
        (
            lambda log: log.append(CLAIM),
            lambda log: log.append(SENT),
            lambda log: log.append(Closed(CLAIM.attempt_id)),
        ),
    ),
}
DIGESTS = {
    "exposure-ledger": [
        "5a4c1588416c6ad670a29f562102bc560c9cd6e1371875c1e6e5161767c6db95",
        "166b3316e04312037a5ecc7903c7ac10a57f2f03a25fd3bacc4b1cdd77ac691e",
        "e248cd38457a192038b5f43bef532700afd5db156eae1d858a21f5ff92dfeec4",
    ],
    "call-log": [
        "4db4adb4c9e9430ebff80464eec34647ee8801508b4149228f7b7aeee8eee48d",
        "7f81a17b7c1522abed9cd840518fe9b24a05ff1624ed36f3dad9c427fee1e5b9",
        "215949b9dfd705f5263cffbd5a835d924ce80aafd0ee16ca29318681115b62a3",
    ],
}
CASES = (
    "hidden-file",
    "gap",
    "stray-file",
    "noncanonical",
    "invalid-json",
    "repeated",
    "repeated-then-invalid",
    "subdirectory",
    "missing",
    "read-only",
)
REFUSALS = {
    ("exposure-ledger", "gap"): (
        "CapabilityError",
        "Cannot read exposure ledger <dir>: Event files must be numbered from 0.json without gaps",
    ),
    ("exposure-ledger", "stray-file"): (
        "CapabilityError",
        "Cannot read exposure ledger <dir>: Event files must be numbered from 0.json without gaps",
    ),
    ("exposure-ledger", "noncanonical"): (
        "CapabilityError",
        "Cannot read exposure ledger <dir>: Ledger event 1.json is not canonical UTF-8 JSON",
    ),
    ("exposure-ledger", "invalid-json"): (
        "CapabilityError",
        (
            "Cannot read exposure ledger <dir>: Ledger event 1.json is not valid JSON: Expecting "
            "property name enclosed in double quotes: line 1 column 2 (char 1)"
        ),
    ),
    ("exposure-ledger", "repeated"): (
        "CapabilityError",
        "Cannot read exposure ledger <dir>: A comparison was frozen over tasks already disclosed",
    ),
    ("exposure-ledger", "repeated-then-invalid"): (
        "CapabilityError",
        "Cannot read exposure ledger <dir>: A comparison was frozen over tasks already disclosed",
    ),
    ("exposure-ledger", "subdirectory"): (
        "CapabilityError",
        "Cannot read exposure ledger <dir>: [Errno 21] Is a directory: '<dir>/3.json'",
    ),
    ("exposure-ledger", "missing"): (
        "CapabilityError",
        (
            "Cannot read exposure ledger <dir>/missing: [Errno 2] No such file or directory: "
            "'<dir>/missing'"
        ),
    ),
    ("exposure-ledger", "read-only"): (
        "CapabilityError",
        "Cannot write exposure ledger <dir>: [Errno 13] Permission denied: '<dir>/.<temp>'",
    ),
    ("call-log", "gap"): (
        "ProviderError",
        "Cannot replay call log <dir>: Entry files must be numbered from 0.json without gaps",
    ),
    ("call-log", "stray-file"): (
        "ProviderError",
        "Cannot replay call log <dir>: Entry files must be numbered from 0.json without gaps",
    ),
    ("call-log", "noncanonical"): (
        "ProviderError",
        "Cannot replay call log <dir>: Call log entry 1.json is not canonical UTF-8 JSON",
    ),
    ("call-log", "invalid-json"): (
        "ProviderError",
        (
            "Cannot replay call log <dir>: Call log entry 1.json is not valid JSON: Expecting "
            "property name enclosed in double quotes: line 1 column 2 (char 1)"
        ),
    ),
    ("call-log", "repeated"): (
        "ProviderError",
        "Cannot replay call log <dir>: Call log entry 3.json repeats an entry",
    ),
    ("call-log", "repeated-then-invalid"): (
        "ProviderError",
        "Cannot replay call log <dir>: Call log entry 3.json repeats an entry",
    ),
    ("call-log", "subdirectory"): (
        "ProviderError",
        "Cannot replay call log <dir>: [Errno 21] Is a directory: '<dir>/3.json'",
    ),
    ("call-log", "missing"): (
        "ProviderError",
        (
            "Cannot replay call log <dir>/missing: [Errno 2] No such file or "
            "directory: '<dir>/missing'"
        ),
    ),
    ("call-log", "read-only"): (
        "ProviderError",
        "Cannot write call log <dir>: [Errno 13] Permission denied: '<dir>/.<temp>'",
    ),
}


def _filled(directory, name, steps=3):
    open_store, actions = STORES[name]
    store = open_store(directory)
    for action in actions[:steps]:
        action(store)
    return store


def _refusal(error, directory):
    text = str(error).replace(str(directory), "<dir>")
    return type(error).__name__, re.sub(r"<dir>/\.\w+'", "<dir>/.<temp>'", text)


def test_byte_objects_preserve_bytes_and_refuse_corruption_without_creating_directories(tmp_path):
    content = b'raw\xff\r\n{"unformatted": true}'
    directory = tmp_path / "objects"
    with pytest.raises(FileNotFoundError):
        keep_object(directory, content)
    assert not directory.exists()
    directory.mkdir()
    digest = keep_object(directory, content)
    assert digest == hashlib.sha256(content).hexdigest()
    assert keep_object(directory, content) == digest
    assert list(directory.iterdir()) == [directory / digest]
    assert (directory / digest).read_bytes() == content
    (directory / digest).write_bytes(b"corrupt")
    with pytest.raises(ObjectConflict, match=digest):
        keep_object(directory, content)
    assert (directory / digest).read_bytes() == b"corrupt"


@pytest.mark.parametrize("name", STORES)
def test_each_store_writes_the_pinned_bytes(tmp_path, name):
    _filled(tmp_path, name)
    files = [tmp_path / f"{index}.json" for index in range(3)]
    assert sorted(tmp_path.iterdir()) == files
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in files] == DIGESTS[name]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("name", STORES)
def test_each_store_refuses_the_pinned_damage(tmp_path, name, case):
    directory = tmp_path / "store"
    directory.mkdir()
    open_store, actions = STORES[name]
    _filled(directory, name, 2 if case == "read-only" else 3)
    files = [directory / f"{index}.json" for index in range(3)]
    match case:
        case "hidden-file":
            (directory / ".tmp-killed-writer").write_bytes(b"{")
        case "gap":
            files[2].rename(directory / "3.json")
        case "stray-file":
            (directory / "notes.txt").write_text("")
        case "noncanonical":
            files[1].write_bytes(files[1].read_bytes().replace(b",", b", ", 1))
        case "invalid-json":
            files[1].write_bytes(b"{")
        case "repeated" | "repeated-then-invalid":
            shutil.copyfile(files[0], directory / "3.json")
            if case == "repeated-then-invalid":
                (directory / "4.json").write_bytes(b"{")
        case "subdirectory":
            (directory / "3.json").mkdir()
        case "missing":
            directory = directory / "missing"
        case "read-only":
            directory.chmod(0o500)
    try:
        if case == "read-only":
            actions[2](open_store(directory))
        else:
            open_store(directory)
    except ValueError as error:
        refusal = _refusal(error, tmp_path / "store")
    else:
        refusal = None
    finally:
        (tmp_path / "store").chmod(0o700)
    assert refusal == REFUSALS.get((name, case))


RESTRICTED = "top decile sharpe"


@pytest.mark.parametrize(
    "attempt",
    [
        # UUID parses 32 characters as hexadecimal, and that error quotes them.
        lambda: parse_uuid(RESTRICTED.ljust(32)),
        lambda: parse_time(RESTRICTED),
        lambda: canonical_bytes({"note": f"{RESTRICTED}\ud800"}),
    ],
    ids=["uuid", "time", "canonical-bytes"],
)
def test_record_errors_keep_the_value_out_of_their_exception_chains(attempt, reachable_text):
    with pytest.raises(PackageError) as caught:
        attempt()
    assert RESTRICTED not in reachable_text(caught.value)
