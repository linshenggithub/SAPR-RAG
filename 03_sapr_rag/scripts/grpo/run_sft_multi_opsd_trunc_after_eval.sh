#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

TWOWIKI_ROOT="$PROJ_ROOT/data/eval_results/2wikimultihopqa/sft_opsd_ckpt1000_full12576_20260903"
MUSIQUE_ROOT="$PROJ_ROOT/data/eval_results/musique/sft_opsd_ckpt1000_full2417_20260903"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-28800}"
started="$(date +%s)"

while true; do
    if grep -q '^completed_at=' "$TWOWIKI_ROOT/config_full.txt" 2>/dev/null \
            && grep -q '^completed_at=' "$MUSIQUE_ROOT/config_full.txt" 2>/dev/null; then
        break
    fi
    if (( $(date +%s) - started >= WAIT_TIMEOUT_SECONDS )); then
        echo "ERROR: timed out waiting for full 2Wiki and MuSiQue evaluations" >&2
        exit 2
    fi
    sleep 60
done

python - \
    "$TWOWIKI_ROOT/full/checkpoint-1000/2wikimultihopqa/metrics.json" 12576 \
    "$MUSIQUE_ROOT/full/checkpoint-1000/musique/metrics.json" 2417 <<'PY'
import json
import sys

for path, expected in zip(sys.argv[1::2], sys.argv[2::2]):
    with open(path) as f:
        metrics = json.load(f)
    actual = int(metrics["n_total"])
    if actual != int(expected):
        raise SystemExit(f"ERROR: {path}: n_total={actual}, expected={expected}")
    print(f"[eval-check] {path}: n_total={actual}")
PY

exec env \
    NO_PROXY=127.0.0.1,localhost \
    no_proxy=127.0.0.1,localhost \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.183.06 \
    bash "$SCRIPT_DIR/run_sft_multi_opsd_trunc_s1500.sh"
