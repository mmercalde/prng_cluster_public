#!/usr/bin/env bash
# ============================================================================
# TB Step 1: FT=0.73 Deterministic Reproduction (TB-corrected v2)
# ----------------------------------------------------------------------------
# Goal:  Reproduce 2026-05-03 crash at W21_O66_FT0.73 / pool=8 / chunk=50k.
# Hyp:   Failure deterministic at this config (~325-350 chunks per worker).
# Notes: - Calls window_optimizer.py DIRECTLY because agents/watcher_agent.py
#          strips warm_start_* args (lines 1486-1492).
#        - --max-seeds 425M chosen so 425e6/50k/24 ~= 354 chunks/worker,
#          landing in observed failure band.
#        - Preflight guard: fails fast if any required CLI flag is missing.
# ============================================================================
set -euo pipefail

cd ~/distributed_prng_analysis

echo "=== TB Step 1: FT=0.73 Deterministic Reproduction ==="
echo "Config: W=21 O=66 FT=0.73 RT=0.31 skip=10-209 pool=8 chunk=50k trials=1"
echo "Seed budget: 425M => ~327-354 chunks/worker/pass"
echo

echo "--- git state ---"
git branch --show-current
git rev-parse --short HEAD
echo

echo "--- preflight: required CLI flags ---"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source ~/venvs/torch/bin/activate
python3 window_optimizer.py --help > /tmp/window_optimizer_help.txt 2>&1
for flag in \
  "--pwc-transport" \
  "--min-workers" \
  "--worker-pool-size" \
  "--seed-cap-amd" \
  "--seed-cap-nvidia" \
  "--max-seeds" \
  "--warm-start-window" \
  "--warm-start-offset" \
  "--warm-start-skip-min" \
  "--warm-start-skip-max" \
  "--warm-start-fwd-thresh" \
  "--warm-start-rev-thresh" \
  "--warm-start-session-idx"
do
  if ! grep -q -- "$flag" /tmp/window_optimizer_help.txt; then
    echo "MISSING required CLI flag: $flag"
    exit 2
  fi
done
echo "Required CLI flags present"
echo

echo "--- cleanup ---"
mkdir -p logs results
rm -f logs/pwc_startup_diag_simple.jsonl
rm -f optimal_window_config.json window_optimization_results.json
truncate -s 0 logs/netconsole_all_rigs.log 2>/dev/null || true

echo "--- launch ---"
echo "Invariant: pool=8 chunk=50k max_seeds=425000000 expected_chunks_per_amd_worker~354"
PRNG_PWC_STARTUP_DIAG=1 \
PRNG_PWC_FIRST_ASSIGN_JITTER_SEC=3 \
PRNG_PWC_PER_WORKER_MIN_GAP_SEC=0.02 \
S163_MEM_DEBUG=1 \
PYTHONPATH=. python3 window_optimizer.py \
  --strategy bayesian \
  --lottery-file daily3.json \
  --trials 1 \
  --output optimal_window_config.json \
  --prng-type java_lcg \
  --use-persistent-workers \
  --pwc-transport tcp \
  --min-workers 24 \
  --worker-pool-size 8 \
  --seed-cap-amd 50000 \
  --seed-cap-nvidia 50000 \
  --max-seeds 425000000 \
  --seed-start 0 \
  --warm-start-window 21 \
  --warm-start-offset 66 \
  --warm-start-skip-min 10 \
  --warm-start-skip-max 209 \
  --warm-start-fwd-thresh 0.73 \
  --warm-start-rev-thresh 0.31 \
  --warm-start-session-idx 0
