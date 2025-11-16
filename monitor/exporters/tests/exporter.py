import json
import pathlib
import time

from prometheus_client import Gauge, start_http_server

STATE_FILE = pathlib.Path("/data/polymarket/tests_state.json")

last_run_start = Gauge(
    "polymorph_tests_last_run_start_timestamp",
    "UNIX timestamp of last test run start",
)

last_run_end = Gauge(
    "polymorph_tests_last_run_end_timestamp",
    "UNIX timestamp of last test run end",
)

tests_total = Gauge(
    "polymorph_tests_total",
    "Total tests in last run",
)

tests_failed = Gauge(
    "polymorph_tests_failed_total",
    "Failed tests in last run",
)

tests_skipped = Gauge(
    "polymorph_tests_skipped_total",
    "Skipped tests in last run",
)


def read_state():
    if not STATE_FILE.exists():
        return
    try:
        data = json.loads(STATE_FILE.read_text())
    except Exception:
        return

    if "last_run_start" in data:
        last_run_start.set(data["last_run_start"])
    if "last_run_end" in data:
        last_run_end.set(data["last_run_end"])
    if "tests_total" in data:
        tests_total.set(data["tests_total"])
    if "tests_failed" in data:
        tests_failed.set(data["tests_failed"])
    if "tests_skipped" in data:
        tests_skipped.set(data["tests_skipped"])


def main():
    start_http_server(9401)
    while True:
        read_state()
        time.sleep(10)


if __name__ == "__main__":
    main()
