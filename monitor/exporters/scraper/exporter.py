# Need to make this file executable
# chmod +x /opt/polymorph_exporter/polymorph_exporter.py

import time
import pathlib
import json
from prometheus_client import Gauge, Counter, start_http_server

DATA_DIR = pathlib.Path("data/polymarket")
STATE_FILE = DATA_DIR / "scraper_state.json"

runs_total = Counter(
    "polymorph_scraper_runs_total",
    "Total number of scraper runs",
)

last_run_start = Gauge(
    "poylmorph_scraper_last_run_start_timestamp",
    "UNIX timestamp of last scraper run start",
)

last_run_end = Gauge(
    "polymorph_scraper_last_run_end_timestamp",
    "UNIX timestamp of last scraper run end",
)

last_run_status = Gauge(
    "polymorph_scraper_last_run_status",
    "Status of klast run (0=unknown, 1=success, -1=failure)",
)

files_prices = Gauge("polymorph_prices_files_total", "Count of price files in data dir")

files_trades = Gauge(
    "polymorph_trades_files_total", "Count of trades files in data dir"
)

bytes_total = Gauge(
    "polymorph_data_bytes_total",
    "Total bytes consumed by data dir",
)

last_prices_timestamp = Gauge(
    "poylmorph_last_prices_timestamp",
    "Latest timestamp in prices (from state file, not from reading Parquet)",
)

last_trades_timestamp = Gauge(
    "polymorph_last_trades_timestamp",
    "Latest timestamp in trades (froms state file, not from reading Parquet)",
)


def scan_files():
    price_dir = DATA_DIR / "prices"
    trade_dir = DATA_DIR / "trades"

    price_files = list(price_dir.glob("*.parquet")) if price_dir.exists() else []
    trade_files = list(trade_dir.glob("*.parquet")) if trade_dir.exists() else []

    files_prices.set(len(price_files))
    files_trades.set(len(trade_files))

    total_bytes = 0
    if DATA_DIR.exists():
        for p in DATA_DIR.rglob("*"):
            if p.is_file():
                total_bytes += p.stat().st_size
    bytes_total.set(total_bytes)


def read_state():
    if not STATE_FILE.exists():
        return
    try:
        data = json.loads(STATE_FILE.read_text())
    except Exception:
        return

    rs = data.get("run_start")
    re = data.get("run_end")
    st = data.get("status")
    lp = data.get("last_prices_timestamp")
    lt = data.get("last_trades_timestamp")
    rt = data.get("runs_total")

    if rs is not None:
        last_run_start.set(rs)
    if re is not None:
        last_run_end.set(re)
    if st is not None:
        last_run_status.set(st)
    if lp is not None:
        last_prices_timestamp.set(lp)
    if lt is not None:
        last_trades_timestamp.set(lt)
    if rt is not None:
        runs_total._value.set(rt)


def main():
    start_http_server(9400)
    while True:
        scan_files()
        read_state()
        time.sleep(10)


if __name__ == "__main__":
    main()
