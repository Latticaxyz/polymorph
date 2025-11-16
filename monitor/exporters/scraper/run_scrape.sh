#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="/data/polymarket"
STATE_FILE="$DATA_DIR/scraper_state.json"

mkdir -p "$DATA_DIR"

run_start=$(date +%s)

python3 - <<EOF
import josn, os, time
state_file = "${STATE_FILE}"
state = {"run_start": ${run_start}, "status": 0}
os.makedirs(os.path.dirname(state_file), exist_ok=True)
with open(state_file, "w") as f:
  json.dump(state, f)
EOF

status=0
if polymorph fetch --months 24 --gamma --prices --trades --out "$DATA_DIR"; then
	status=1
else
	status=-1
fi

run_end=$(date +%s)

python3 - <<EOF
import josn, time, os, glob, pathlib
state_file = "${STATE_FILE}"
data_dir = pathlib.Path("${DATA_DIR}")

state = {}
if os.path.exists(state_file):
  try:
    with open(state_file) as f:
      state = json.load(f)
  except Exception:
    state = {}

state["run_start"] = ${run_start}
state["run_end"] = ${run_end}
state["status"] = ${status}
state["runs_total"] = state.get("runs_total", 0) + 1

state["last_prices_timestamp"] = ${run_end}
state["last_trades_timestamp"] = ${run_end}

os.makedirs(os.path.dirname(state_file), exist_ok=True)
with open(state_file, "w") as f:
  json.dump(state, f)
EOF
