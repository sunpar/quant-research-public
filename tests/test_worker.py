from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import pytest

from quant_research import datasets
from quant_research.access import licensed
from quant_research.records import canonical_bytes
from quant_research.worker import (
    IMAGE,
    LABEL,
    Limits,
    Refused,
    RunLog,
    RunRecord,
    Signal,
    WorkerError,
    run,
)

ROOT = Path(__file__).resolve().parents[1]
LIMITS = Limits(timeout=20.0, memory=128 * 2**20, pids=32, scratch=8 * 2**20, output=2**16)
PRELUDE = """\
import json

KEYS = [row["key"] for row in json.load(open("/job/slice.json"))["rows"]]


def write(text):
    with open("/output/signal.json", "w") as output:
        output.write(text)


def emit(values):
    write(json.dumps(values))

"""


def labelled(jobs: Path) -> list[str]:
    found = subprocess.run(
        ["docker", "ps", "--all", f"--filter=label={LABEL}", f'--format={{{{.Label "{LABEL}"}}}}'],
        capture_output=True,
        text=True,
        check=True,
    )
    return [job for job in found.stdout.splitlines() if Path(job).parent == jobs]


def pinned() -> dict[str, str]:
    """Return the locked version of each image dependency, checking that each names its hashes."""
    text = re.sub(r"#.*", "", (ROOT / "worker" / "requirements.txt").read_text())
    pins = {}
    for line in filter(str.strip, text.replace("\\\n", " ").splitlines()):
        pin, *hashes = line.split()
        assert re.fullmatch(r"[a-z0-9-]+==[0-9a-z.]+", pin), line
        assert hashes, line
        assert all(re.fullmatch(r"--hash=sha256:[0-9a-f]{64}", item) for item in hashes), line
        name, version = pin.split("==")
        pins[name] = version
    return pins


@pytest.fixture
def jobs(tmp_path, monkeypatch) -> Path:
    directory = tmp_path / "jobs"
    directory.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(directory))
    return directory


@pytest.fixture
def worker(tmp_path, image, jobs, probe):
    store = tmp_path / "store"

    def execute(source, rows, limits=LIMITS):
        stored = probe.save(store, rows)
        result = run(
            source, probe.catalog(vintage=stored), store, probe.name, probe.context, limits
        )
        assert labelled(jobs) == [], "a container outlived its run"
        assert list(jobs.iterdir()) == [], "a job directory outlived its run"
        return result

    execute.store = store
    return execute


def values(result) -> dict[str, float]:
    assert isinstance(result, Signal), result
    return result.values


def run_record(probe, tmp_path, limits=LIMITS) -> RunRecord:
    store = tmp_path / "store"
    stored = probe.save(store, ["A", "B"])
    read = datasets.read(probe.catalog(vintage=stored), store, probe.name, probe.context)
    return RunRecord(f"sha256:{'0' * 64}", hashlib.sha256(b"pass").hexdigest(), read.record, limits)


def test_refused_read_starts_no_container_and_has_no_run_to_log(tmp_path, probe):
    (tmp_path / probe.name).mkdir()
    (tmp_path / probe.name / f"{'0' * 64}.json").write_bytes(b"unread")
    catalog = probe.catalog(licensed("vendor"))
    result = run("raise SystemExit(1)", catalog, tmp_path, probe.name, probe.context)
    assert result == Refused("unknown_dataset")
    assert result.message == "No dataset this run may read has this name"
    (tmp_path / "runs").mkdir()
    with pytest.raises(WorkerError, match="started no container"):
        RunLog(tmp_path / "runs").append(result)
    assert list((tmp_path / "runs").iterdir()) == []


def test_image_build_pins_the_base_and_every_dependency_by_hash():
    assert f"name={IMAGE}," in (ROOT / "Makefile").read_text()
    dockerfile = (ROOT / "worker" / "Dockerfile").read_text()
    assert re.search(r"^FROM [^\s@]+@sha256:[0-9a-f]{64}$", dockerfile, re.MULTILINE)
    assert "--require-hashes --no-deps --only-binary=:all:" in dockerfile
    assert pinned().keys() == {"numpy", "pandas", "python-dateutil", "six"}


def test_container_tests_fail_with_dockers_reason_when_docker_cannot_supply_the_image(tmp_path):
    docker = tmp_path / "docker"
    docker.write_text("#!/bin/sh\necho 'Cannot connect to the Docker daemon' >&2\nexit 1\n")
    docker.chmod(0o755)
    arguments = [
        "-q",
        "-p",
        "no:cacheprovider",
        f"--basetemp={tmp_path / 'run'}",
        f"{__file__}::test_candidate_has_no_network",
    ]
    bootstrap = (
        f"import sys; sys.path[:0] = {sys.path!r}; import pytest; "
        f"sys.exit(pytest.main({arguments!r}))"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-I", "-S", "-c", bootstrap],
        cwd=ROOT,
        env=os.environ | {"PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"docker cannot supply {IMAGE}: Cannot connect to the Docker daemon" in result.stdout
    assert "skipped" not in result.stdout


def test_record_names_the_image_source_and_read_and_repeats(worker, image, probe):
    source = PRELUDE + "emit({key: float(len(key)) for key in KEYS})\n"
    first = worker(source, ["A", "BB"])
    second = worker(source, ["A", "BB"])
    assert values(first) == {"A": 1.0, "BB": 2.0}
    assert first == second
    assert first.record.to_bytes() == second.record.to_bytes()
    assert first.record.image == image
    assert first.record.source == hashlib.sha256(source.encode()).hexdigest()
    assert first.record.limits == LIMITS
    stored = worker.store / probe.name / f"{first.record.read.vintage}.json"
    replayed = first.record.read.replay(stored.read_bytes())
    assert [(row.key, row.revision) for row in replayed] == [("A", 1), ("BB", 1)]
    assert json.loads(first.record.to_bytes())["image"] == image


def test_candidate_imports_the_pinned_dependencies_on_one_blas_thread(worker):
    pins = pinned()
    source = PRELUDE + (
        "import os\n"
        "from importlib.metadata import version\n"
        "import numpy, pandas\n"
        f"PINS = {pins!r}\n"
        "product = numpy.ones((256, 256)) @ numpy.ones((256, 256))\n"
        "emit({name: float(version(name) == pin) for name, pin in PINS.items()} | {\n"
        "    'rolling': float(pandas.Series(range(10)).rolling(3).mean().iloc[-1]),\n"
        "    'product': float(product[0, 0]),\n"
        "    'threads': float(len(os.listdir('/proc/self/task'))),\n"
        "})\n"
    )
    assert values(worker(source, [*pins, "rolling", "product", "threads"])) == {
        **dict.fromkeys(pins, 1.0),
        "rolling": 8.0,
        "product": 256.0,
        "threads": 1.0,
    }


def test_reopened_log_keeps_each_container_run(worker, image, tmp_path):
    emitted = worker(PRELUDE + "emit({key: 1 for key in KEYS})\n", ["A", "B"])
    refused = worker(PRELUDE + "raise SystemExit(4)\n", ["A", "B"])
    (tmp_path / "runs").mkdir()
    log = RunLog(tmp_path / "runs")
    stored = [log.append(result) for result in (emitted, refused, emitted)]
    assert RunLog(tmp_path / "runs").load() == tuple(stored)
    assert sorted((tmp_path / "runs").iterdir()) == [
        tmp_path / "runs" / f"{i}.json" for i in range(3)
    ]
    output = hashlib.sha256(canonical_bytes({"A": 1.0, "B": 1.0})).hexdigest()
    assert [(run.record, run.outcome, run.output) for run in stored] == [
        (emitted.record, "signal", output),
        (refused.record, "failed", None),
        (emitted.record, "signal", output),
    ]
    assert len({run.run_id for run in stored}) == 3
    assert stored[0].record.image == image


def test_candidate_sees_only_the_adapter_slice_and_its_source(worker, probe):
    source = PRELUDE + (
        "import os\n"
        "assert sorted(os.listdir('/job')) == ['candidate.py', 'slice.json']\n"
        "rows = json.load(open('/job/slice.json'))['rows']\n"
        "emit({row['key']: float(row['values']['close']) * 10 + row['revision'] for row in rows})\n"
    )
    rows = [
        probe.version("A", close="11"),
        probe.version("A", revision=2, at="2015-07-02T20:00:00Z", close="12"),
        probe.version("B", close="5"),
    ]
    assert values(worker(source, rows)) == {"A": 111.0, "B": 51.0}


def test_candidate_has_no_network(worker):
    source = PRELUDE + (
        "import socket\n"
        "def attempt(action):\n"
        "    try:\n"
        "        action()\n"
        "    except OSError:\n"
        "        return 0.0\n"
        "    return 1.0\n"
        "def udp():\n"
        "    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:\n"
        "        probe.sendto(b'probe', ('8.8.8.8', 53))\n"
        "emit({\n"
        "    'dns': attempt(lambda: socket.getaddrinfo('example.com', 443)),\n"
        "    'tcp': attempt(lambda: socket.create_connection(('1.1.1.1', 443), 3).close()),\n"
        "    'tcp6': attempt(\n"
        "        lambda: socket.create_connection(('2606:4700:4700::1111', 443), 3).close()\n"
        "    ),\n"
        "    'udp': attempt(udp),\n"
        "})\n"
    )
    assert values(worker(source, ["dns", "tcp", "tcp6", "udp"])) == {
        "dns": 0.0,
        "tcp": 0.0,
        "tcp6": 0.0,
        "udp": 0.0,
    }


def test_candidate_reads_no_host_credentials(worker, tmp_path, monkeypatch, probe):
    credential = f"qr-credential-{uuid4().hex}"
    control = f"qr-control-{uuid4().hex}"
    monkeypatch.setenv("QR_WORKER_CREDENTIAL", credential)
    (tmp_path / "credentials").write_text(credential)
    # The source holds both markers reversed, so only a leak or the slice can supply them.
    source = PRELUDE + (
        "import os, stat\n"
        f"CREDENTIAL = {credential[::-1]!r}[::-1].encode()\n"
        f"CONTROL = {control[::-1]!r}[::-1].encode()\n"
        "found = set()\n"
        "def scan(data):\n"
        "    found.update(name for name, marker in (('credential', CREDENTIAL), "
        "('control', CONTROL)) if marker in data)\n"
        "scan(json.dumps(dict(os.environ)).encode())\n"
        "for pid in filter(str.isdigit, os.listdir('/proc')):\n"
        "    for name in ('environ', 'cmdline'):\n"
        "        try:\n"
        "            scan(open(f'/proc/{pid}/{name}', 'rb').read())\n"
        "        except OSError:\n"
        "            pass\n"
        "for root, directories, files in os.walk('/'):\n"
        "    if root == '/':\n"
        "        directories[:] = [d for d in directories if d not in ('proc', 'sys', 'dev')]\n"
        "    for name in files:\n"
        "        path = os.path.join(root, name)\n"
        "        try:\n"
        "            mode = os.lstat(path).st_mode\n"
        "            if stat.S_ISSOCK(mode):\n"
        "                found.add('socket')\n"
        "            elif stat.S_ISREG(mode):\n"
        "                scan(open(path, 'rb').read())\n"
        "        except OSError:\n"
        "            pass\n"
        "emit({\n"
        "    'control': float('control' in found),\n"
        "    'credential': float('credential' in found),\n"
        "    'socket': float('socket' in found),\n"
        "    'environment': float(\n"
        "        set(os.environ) <= {'PATH', 'PWD', 'LC_CTYPE', 'OPENBLAS_NUM_THREADS'}\n"
        "    ),\n"
        "})\n"
    )
    rows = [probe.version("control", note=control), "credential", "socket", "environment"]
    assert values(worker(source, rows)) == {
        "control": 1.0,
        "credential": 0.0,
        "socket": 0.0,
        "environment": 1.0,
    }


def test_runtime_is_locked_down(worker):
    source = PRELUDE + (
        "import os\n"
        "status = dict(line.split(':', 1) for line in open('/proc/self/status'))\n"
        "def cgroup(name):\n"
        "    return open(f'/sys/fs/cgroup/{name}').read().split()\n"
        "def writes(path, mode='w'):\n"
        "    try:\n"
        "        with open(path, mode):\n"
        "            return 1.0\n"
        "    except OSError:\n"
        "        return 0.0\n"
        "outside = ['/', '/job', '/etc', '/usr/local/lib', '/var/tmp', '/home', '/root', '/run',\n"
        "           '/opt', '/dev', '/dev/mqueue', '/sys/fs/cgroup']\n"
        "quota, period = cgroup('cpu.max')\n"
        "emit({\n"
        "    'root': float(0 in (os.getuid(), os.geteuid(), os.getgid())),\n"
        "    'capabilities': float(int(status['CapEff'], 16)),\n"
        "    'no_new_privs': float(status['NoNewPrivs']),\n"
        "    'seccomp': float(status['Seccomp']),\n"
        "    'outside': sum(writes(f'{path}/probe') for path in outside),\n"
        "    'inputs': writes('/job/candidate.py', 'a') + writes('/job/slice.json', 'a'),\n"
        "    'scratch': writes('/tmp/probe'),\n"
        "    'cpus': int(quota) / int(period),\n"
        "    'memory': float(cgroup('memory.max')[0]),\n"
        "    'pids': float(cgroup('pids.max')[0]),\n"
        "})\n"
    )
    keys = "root capabilities no_new_privs seccomp outside inputs scratch cpus memory pids"
    assert values(worker(source, keys.split())) == {
        "root": 0.0,
        "capabilities": 0.0,
        "no_new_privs": 1.0,
        "seccomp": 2.0,
        "outside": 0.0,
        "inputs": 0.0,
        "scratch": 1.0,
        "cpus": LIMITS.cpus,
        "memory": float(LIMITS.memory),
        "pids": float(LIMITS.pids),
    }


def test_fork_loop_stops_at_the_pids_limit(worker):
    source = PRELUDE + (
        "import os, signal\n"
        "children = []\n"
        "try:\n"
        "    while len(children) < 1000:\n"
        "        pid = os.fork()\n"
        "        if pid == 0:\n"
        "            signal.pause()\n"
        "            os._exit(0)\n"
        "        children.append(pid)\n"
        "except BlockingIOError:\n"
        "    pass\n"
        "for pid in children:\n"
        "    os.kill(pid, signal.SIGKILL)\n"
        "    os.waitpid(pid, 0)\n"
        "emit({'forks': float(len(children))})\n"
    )
    forks = values(worker(source, ["forks"]))["forks"]
    assert 0 < forks < LIMITS.pids


def test_allocation_loop_dies_at_the_memory_cap(worker):
    source = "chunks = []\nwhile True:\n    chunks.append(bytearray(16 * 2**20))\n"
    assert worker(source, ["A"]).reason == "memory"


def test_sleep_loop_is_killed_at_the_timeout(worker):
    source = "import time\nwhile True:\n    time.sleep(1)\n"
    started = time.monotonic()
    result = worker(source, ["A"], Limits(timeout=2.0, memory=LIMITS.memory, pids=LIMITS.pids))
    assert result.reason == "timeout"
    assert time.monotonic() - started < 10


def test_host_exception_removes_the_container_and_job_directory(image, jobs, tmp_path, probe):
    stored = probe.save(tmp_path, ["A"])
    source = "import time\nwhile True:\n    time.sleep(0.5)\n"

    def interrupt(signum, frame):
        raise RuntimeError("host interrupted the run")

    previous = signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, 1.5)
    try:
        with pytest.raises(RuntimeError, match="host interrupted"):
            run(source, probe.catalog(vintage=stored), tmp_path, probe.name, probe.context, LIMITS)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert labelled(jobs) == []
    assert list(jobs.iterdir()) == []


def interrupt(signum, frame):
    raise RuntimeError("host interrupted docker")


# A terminal Ctrl-C reaches every process in the host's process group.
CTRL_C = 'kill -INT $PPID\n  [ "$(ps -o pgid= -p $PPID)" = "$(ps -o pgid= -p $$)" ] && kill -INT $$'


@pytest.mark.parametrize(
    ("signum", "handler", "error", "send", "starting"),
    [
        (signal.SIGALRM, interrupt, RuntimeError, "kill -ALRM $PPID", False),
        (signal.SIGINT, signal.default_int_handler, KeyboardInterrupt, CTRL_C, False),
        (signal.SIGALRM, interrupt, RuntimeError, ":", True),
    ],
    ids=["exception", "ctrl-c", "cli-start"],
)
def test_host_interrupt_during_create_leaves_no_container(
    worker, jobs, tmp_path, monkeypatch, signum, handler, error, send, starting
):
    docker = shutil.which("docker")
    created, removed = tmp_path / "created", tmp_path / "removed"
    shim = tmp_path / "bin" / "docker"
    shim.parent.mkdir()
    # The background job is the daemon, which finishes the create after the host's removal, or
    # after a second while the host waits for the CLI. Docker refuses a missing bind source, so
    # the removal waits for the create while the job directory still exists.
    shim.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = create ]; then\n'
        f'  (for _ in 1 2 3 4 5 6 7 8 9 10; do [ -e "{removed}" ] && break; sleep 0.1; done\n'
        f'   "{docker}" "$@" > "{created}.part" && mv "{created}.part" "{created}") \\\n'
        "    > /dev/null 2>&1 &\n"
        f"  {send}\n"
        "  wait\n"
        "  exit\n"
        "fi\n"
        f'"{docker}" "$@"\n'
        "status=$?\n"
        'if [ "$1" = rm ]; then\n'
        f'  touch "{removed}"\n'
        f'  for _ in $(seq 100); do [ -e "{created}" ] && break; sleep 0.1; done\n'
        "fi\n"
        "exit $status\n"
    )
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{shim.parent}{os.pathsep}{os.environ['PATH']}")
    if starting:
        start = subprocess.Popen.__init__

        def interrupt_on_start(self, args, *rest, **options):
            start(self, args, *rest, **options)
            if args[1:2] == ["create"]:
                signal.pthread_kill(threading.main_thread().ident, signum)

        monkeypatch.setattr(subprocess.Popen, "__init__", interrupt_on_start)
    previous = signal.signal(signum, handler)
    try:
        with pytest.raises(error):
            worker("pass", ["A"])
    finally:
        signal.signal(signum, previous)
    assert created.read_text(), "the daemon never created the container"
    assert labelled(jobs) == []
    assert list(jobs.iterdir()) == []


def test_host_exception_during_removal_still_removes_the_container(
    worker, jobs, tmp_path, monkeypatch
):
    docker = shutil.which("docker")
    shim = tmp_path / "bin" / "docker"
    shim.parent.mkdir()
    # A removal killed during the sleep never reaches the daemon.
    shim.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = rm ]; then\n'
        "  kill -ALRM $PPID\n"
        "  sleep 1\n"
        "fi\n"
        f'exec "{docker}" "$@"\n'
    )
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{shim.parent}{os.pathsep}{os.environ['PATH']}")
    previous = signal.signal(signal.SIGALRM, interrupt)
    try:
        with pytest.raises(RuntimeError, match="host interrupted"):
            worker("pass", ["A"])
    finally:
        signal.signal(signal.SIGALRM, previous)
    assert labelled(jobs) == []
    assert list(jobs.iterdir()) == []


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("emit({'A': 1, 'B': 2})\nopen('/output/extra', 'w').close()\n", "extra_output"),
        ("emit({'A': 1, 'B': 2})\nimport os\nos.mkdir('/output/more')\n", "extra_output"),
        ("import os\nos.symlink('/job/slice.json', '/output/signal.json')\n", "extra_output"),
        ("pass\n", "missing_output"),
        ("raise SystemExit(3)\n", "failed"),
        ("write('{\"A\": 1, \"B\": 2' + ' ' * 2**16 + '}')\n", "oversized"),
        ('write(\'{"A": NaN, "B": 2}\')\n', "non_finite"),
        ('write(\'{"A": -Infinity, "B": 2}\')\n', "non_finite"),
        ('write(\'{"A": 1e999, "B": 2}\')\n', "non_finite"),
        ("emit({'A': 1})\n", "wrong_index"),
        ("emit({'A': 1, 'B': 2, 'C': 3})\n", "wrong_index"),
        ("emit({'A': '1', 'B': 2})\n", "malformed"),
        ("emit({'A': True, 'B': 2})\n", "malformed"),
        ("emit([1, 2])\n", "malformed"),
        ('write(\'{"A": 1, "A": 2, "B": 2}\')\n', "malformed"),
        ('write(\'{"A": "1", "A": 1, "B": 2}\')\n', "malformed"),
        ('write(\'{"A": 1, "A": 1.0, "B": 2}\')\n', "malformed"),
        ("write('A=1')\n", "malformed"),
        (
            "emit({'A': 1, 'B': 2})\nimport os\nfor fd in os.listdir('/proc/1/fd'):\n"
            "    if os.readlink(f'/proc/1/fd/{fd}').startswith('pipe:'):\n"
            "        open(f'/proc/1/fd/{fd}', 'wb').write(b'forged')\n",
            "malformed",
        ),
    ],
)
def test_output_outside_the_contract_is_refused(worker, body, reason):
    result = worker(PRELUDE + body, ["A", "B"])
    assert isinstance(result, Refused)
    assert (result.reason, result.record.limits) == (reason, LIMITS)


def test_racing_logs_store_every_identical_run(tmp_path, probe):
    result = Signal({"A": 1.0, "B": 2.0}, run_record(probe, tmp_path))
    (tmp_path / "runs").mkdir()
    logs = [RunLog(tmp_path / "runs") for _ in range(8)]
    start = threading.Barrier(len(logs))

    def append(log):
        start.wait()
        return log.append(result)

    with ThreadPoolExecutor(len(logs)) as pool:
        stored = list(pool.map(append, logs))
    reopened = RunLog(tmp_path / "runs").load()
    assert sorted(run.run_id for run in reopened) == sorted(run.run_id for run in stored)


def test_log_reopens_a_run_whose_limits_mix_ints_and_floats(tmp_path, probe):
    record = run_record(probe, tmp_path, Limits(timeout=20, memory=float(LIMITS.memory)))
    (tmp_path / "runs").mkdir()
    stored = RunLog(tmp_path / "runs").append(Signal({"A": 1.0, "B": 2.0}, record))
    assert RunLog(tmp_path / "runs").load() == (stored,)


@pytest.mark.parametrize(
    ("limits", "reason", "message"),
    [
        (Limits(timeout=True), "timeout", "Invalid limit timeout"),
        (Limits(timeout=math.nan), "timeout", "not JSON compliant"),
        (LIMITS, "uncleared", "Invalid outcome"),
    ],
)
def test_append_refuses_a_run_that_load_would_refuse(tmp_path, probe, limits, reason, message):
    record = run_record(probe, tmp_path, limits)
    (tmp_path / "runs").mkdir()
    with pytest.raises(WorkerError, match=message):
        RunLog(tmp_path / "runs").append(Refused(reason, record))
    assert RunLog(tmp_path / "runs").load() == ()


def canonical(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{text}\n".encode()


def rewrite(path, change):
    entry = json.loads(path.read_bytes())
    change(entry["run"])
    path.write_bytes(canonical(entry))


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("indent", "not canonical"),
        ("no-final-newline", "not canonical"),
        ("version-2", "schema version 2"),
        ("record-version-text", "schema version '1'"),
        ("gap", "without gaps"),
        ("stray-file", "without gaps"),
        ("repeated-run", "repeats a run"),
        ("unknown-field", "unknown 'note'"),
        ("image-tag", "Invalid image ID"),
        ("unknown-outcome", "Invalid outcome 'crashed'"),
        ("signal-without-output", "Invalid SHA-256 digest"),
        ("refusal-with-output", "Invalid outcome 'timeout'"),
        ("text-pids", "Invalid limit pids '32'"),
        ("source-not-digest", "Invalid SHA-256 digest"),
        ("damaged-read", "Invalid read record"),
    ],
)
def test_damaged_log_fails_to_open(tmp_path, probe, case, message):
    record = run_record(probe, tmp_path)
    directory = tmp_path / "runs"
    directory.mkdir()
    log = RunLog(directory)
    log.append(Signal({"A": 1.0, "B": 2.0}, record))
    log.append(Refused("timeout", record))
    emitted, refused = directory / "0.json", directory / "1.json"
    stored = json.loads(emitted.read_bytes())
    match case:
        case "indent":
            emitted.write_text(json.dumps(stored, indent=2))
        case "no-final-newline":
            emitted.write_bytes(emitted.read_bytes().rstrip(b"\n"))
        case "version-2":
            emitted.write_bytes(canonical(stored | {"schema_version": 2}))
        case "record-version-text":
            rewrite(emitted, lambda run: run["record"].update(schema_version="1"))
        case "gap":
            refused.rename(directory / "2.json")
        case "stray-file":
            (directory / "notes.txt").write_text("")
        case "repeated-run":
            (directory / "2.json").write_bytes(emitted.read_bytes())
        case "unknown-field":
            rewrite(emitted, lambda run: run.update(note="extra"))
        case "image-tag":
            rewrite(emitted, lambda run: run["record"].update(image=IMAGE))
        case "unknown-outcome":
            rewrite(refused, lambda run: run.update(outcome="crashed"))
        case "signal-without-output":
            rewrite(emitted, lambda run: run.update(output=None))
        case "refusal-with-output":
            rewrite(refused, lambda run: run.update(output=stored["run"]["output"]))
        case "text-pids":
            rewrite(emitted, lambda run: run["record"]["limits"].update(pids="32"))
        case "source-not-digest":
            rewrite(emitted, lambda run: run["record"].update(source="candidate.py"))
        case "damaged-read":
            rewrite(emitted, lambda run: run["record"]["read"].update(vintage="unread"))
    with pytest.raises(WorkerError, match=message):
        RunLog(directory)
