from pathlib import Path

import pytest

failures: Path


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--repo-shard", help="index/count/total: run every count-th selected test")
    parser.addoption("--repo-failures", help="file that receives one line per failed test phase")


def pytest_configure(config: pytest.Config) -> None:
    global failures
    failures = Path(config.getoption("repo_failures"))


# Run after -k, -m, and other deselection, so positions match the counted collection.
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    index, count, total = map(int, config.getoption("repo_shard").split("/"))
    # Parametrize IDs may differ between processes, so shards select by position, not node ID.
    if len(items) != total:
        raise pytest.UsageError(
            f"Shard {index} collected {len(items)} tests; the suite collected {total}"
        )
    config.hook.pytest_deselected(
        items=[item for position, item in enumerate(items) if position % count != index]
    )
    items[:] = items[index::count]


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.failed:
        line = (
            f"FAILED {report.nodeid}"
            if report.when == "call"
            else f"ERROR {report.nodeid} during {report.when}"
        )
        with failures.open("a") as stream:
            stream.write(line + "\n")
