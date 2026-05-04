#!/usr/bin/env bash
set -euo pipefail

cd ~/distributed_prng_analysis

CAP="${1:-100000}"
TRIALS="${2:-5}"

case "$CAP" in
  100000|150000|200000) ;;
  *)
    echo "ERROR: CAP must be one of: 100000, 150000, 200000"
    exit 1
    ;;
esac

echo "=== S170 Stability Curve Run ==="
echo "CAP=$CAP  TRIALS=$TRIALS"

echo "--- git state ---"
git branch --show-current && git rev-parse --short HEAD

echo "--- cleanup ---"
rm -f logs/pwc_startup_diag_simple.jsonl optimal_window_config.json
mkdir -p logs
truncate -s 0 logs/netconsole_all_rigs.log

echo "--- launch ---"
source ~/venvs/torch/bin/activate
PRNG_PWC_STARTUP_DIAG=1 \
PRNG_PWC_FIRST_ASSIGN_JITTER_SEC=3 \
PRNG_PWC_PER_WORKER_MIN_GAP_SEC=0.02 \
PYTHONPATH=. nohup python3 agents/watcher_agent.py \
  --run-pipeline --start-step 1 --end-step 1 --force-step 1 \
  --params "{\"min_workers\": 24, \"seed_cap_amd\": $CAP, \"window_trials\": $TRIALS}" \
  > logs/stability_cap_${CAP}_t${TRIALS}_$(date +%H%M).log 2>&1 & echo PID: $!
