#!/usr/bin/env bash
# ============================================================================
# S174 Gate Validation — verify ready-gate hard fix before D1
# ----------------------------------------------------------------------------
# THIS IS NOT A CRASH REPRODUCTION.
# THIS IS A GATE-PATCH VERIFICATION RUN.
# ----------------------------------------------------------------------------
# pool=8 / chunk=25k / 5M seeds (tiny — completes in <30s)
# Mode: open Optuna (1 trial)
# ----------------------------------------------------------------------------
# TB acceptance criteria:
#   1. Log emits "READY GATE PASSED" with ready >= 24
#   2. First job_assign happens AFTER "READY GATE PASSED"
#   3. (Negative case tested separately by setting min-workers > 26)
#   4. No "N ready worker(s) — dispatching" with N < min_workers
# ============================================================================
set -euo pipefail

cd ~/distributed_prng_analysis

RUN_ID="s174_gate_validation_$(date +%Y%m%d_%H%M%S)"
POOL=8
CHUNK=25000
MAX_SEEDS=5000000
MIN_READY_WORKERS=24
RUN_LOG="logs/${RUN_ID}.log"
SUMMARY_FILE="logs/${RUN_ID}_summary.txt"

mkdir -p logs

{
  echo "=== S174 Gate Validation (NOT CRASH REPRO — PATCH VERIFICATION) ==="
  echo "RUN_ID=$RUN_ID"
  echo "git_sha=$(git rev-parse --short HEAD)"
  echo "started_at=$(date -Is)"
  echo "pool=$POOL"
  echo "chunk_cap=$CHUNK"
  echo "max_seeds=$MAX_SEEDS"
  echo "expected_chunks_total=$((MAX_SEEDS / CHUNK))"
  echo "min_workers=$MIN_READY_WORKERS"
  echo "purpose=verify S174 ready-gate hard fix"
  echo
} | tee "$SUMMARY_FILE"

echo "--- preflight ---"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source ~/venvs/torch/bin/activate

echo "--- cleanup ---"
rm -f logs/pwc_startup_diag_simple.jsonl
rm -f logs/s173_job_assignment_ledger.jsonl
rm -f optimal_window_config.json window_optimization_results.json
truncate -s 0 logs/netconsole_all_rigs.log 2>/dev/null || true

echo "--- launch (foreground, no watchdog needed — patch enforces gate in code) ---"

set +e
PRNG_PWC_STARTUP_DIAG=1 \
PYTHONPATH=. python3 window_optimizer.py \
  --strategy bayesian \
  --lottery-file daily3.json \
  --trials 1 \
  --output optimal_window_config.json \
  --prng-type java_lcg \
  --use-persistent-workers \
  --pwc-transport tcp \
  --min-workers $MIN_READY_WORKERS \
  --worker-pool-size $POOL \
  --seed-cap-amd $CHUNK \
  --seed-cap-nvidia $CHUNK \
  --max-seeds $MAX_SEEDS \
  --seed-start 0 \
  > "$RUN_LOG" 2>&1
PYTHON_EXIT=$?
set -e

{
  echo
  echo "=== Result ==="
  echo "python_exit_code=$PYTHON_EXIT"
  echo "completed_at=$(date -Is)"
} | tee -a "$SUMMARY_FILE"

# ===========================================================================
# Acceptance criteria checks
# ===========================================================================
{
  echo
  echo "=== TB Acceptance Criteria ==="

  # 1. READY GATE PASSED with ready >= 24
  GATE_LINE=$(grep -E "READY GATE PASSED" "$RUN_LOG" | head -1 || true)
  if [ -n "$GATE_LINE" ]; then
    READY_AT_GATE=$(echo "$GATE_LINE" | grep -oE "[0-9]+/[0-9]+ ready" | head -1 | cut -d/ -f1 || true)
    if [ "${READY_AT_GATE:-0}" -ge "$MIN_READY_WORKERS" ]; then
      echo "  [PASS] Criterion 1: READY GATE PASSED with ready=$READY_AT_GATE >= $MIN_READY_WORKERS"
    else
      echo "  [FAIL] Criterion 1: READY GATE PASSED but ready=$READY_AT_GATE < $MIN_READY_WORKERS"
    fi
  else
    GATE_FAIL=$(grep -E "READY GATE FAILED" "$RUN_LOG" | head -1 || true)
    if [ -n "$GATE_FAIL" ]; then
      echo "  [INFO] READY GATE FAILED triggered (gate is enforcing — good if intended)"
      echo "         Line: $GATE_FAIL"
    else
      echo "  [FAIL] Criterion 1: no READY GATE PASSED line found in log"
    fi
  fi

  # 2. First job_assign happens AFTER READY GATE PASSED
  if [ -n "$GATE_LINE" ] && [ -f logs/s173_job_assignment_ledger.jsonl ]; then
    GATE_TS=$(grep -E "READY GATE PASSED" "$RUN_LOG" | head -1 | grep -oE "[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}" | head -1 || true)
    FIRST_JOB_TS=$(head -1 logs/s173_job_assignment_ledger.jsonl 2>/dev/null | python3 -c 'import sys,json; print(json.loads(sys.stdin.read()).get("ts",""))' 2>/dev/null || true)
    echo "  [INFO] Criterion 2: gate_ts=$GATE_TS  first_job_ts=$FIRST_JOB_TS"
    echo "         (manual verification — first_job_ts should be >= gate_ts)"
  else
    echo "  [SKIP] Criterion 2: cannot compare (gate or ledger missing)"
  fi

  # 3. No "N ready worker(s) — dispatching" with N < min_workers
  STALE_DISPATCH=$(grep -E "ready worker\(s\) — dispatching" "$RUN_LOG" || true)
  if [ -z "$STALE_DISPATCH" ]; then
    echo "  [PASS] Criterion 3: no legacy 'N ready worker(s) — dispatching' line"
  else
    UNSAFE=$(echo "$STALE_DISPATCH" | grep -vE "($MIN_READY_WORKERS|2[5-9]|[3-9][0-9]) ready worker" || true)
    if [ -z "$UNSAFE" ]; then
      echo "  [PASS] Criterion 3: legacy line present but all dispatches were >= $MIN_READY_WORKERS"
    else
      echo "  [FAIL] Criterion 3: legacy dispatch with N < $MIN_READY_WORKERS:"
      echo "$UNSAFE" | sed 's/^/         /'
    fi
  fi

  # 4. dispatch confirmed line shows defense-in-depth
  DISPATCH_CONFIRMED=$(grep -E "dispatch confirmed:" "$RUN_LOG" | head -1 || true)
  if [ -n "$DISPATCH_CONFIRMED" ]; then
    echo "  [PASS] Criterion 4: defense-in-depth fired — $DISPATCH_CONFIRMED"
  else
    echo "  [INFO] Criterion 4: no 'dispatch confirmed' line (may be okay if gate already raised)"
  fi

  # Job count
  JOB_COUNT=$(wc -l < logs/s173_job_assignment_ledger.jsonl 2>/dev/null | tr -d ' ' || echo 0)
  echo
  echo "=== Run stats ==="
  echo "job_assignments_in_ledger: $JOB_COUNT"
  echo "expected_chunks: $((MAX_SEEDS / CHUNK))"

} | tee -a "$SUMMARY_FILE"

echo
echo "--- summary written ---"
echo "Summary: $SUMMARY_FILE"
echo "Run log: $RUN_LOG"
echo
echo "Inspect:"
echo "  grep -E 'READY GATE|DISPATCH BLOCKED|dispatch confirmed' $RUN_LOG"
