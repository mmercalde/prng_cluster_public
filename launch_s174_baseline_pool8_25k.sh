#!/usr/bin/env bash
# ============================================================================
# S174 Baseline (TB conditional approval — 5 conditions implemented)
# ----------------------------------------------------------------------------
# THIS IS NOT A CRASH REPRODUCTION.
# THIS IS A POST-RESET INSTRUMENTATION AND HEALTHY-CLUSTER BASELINE.
# ----------------------------------------------------------------------------
# pool=8 / chunk=25k / 213M seeds (Mode 1 equal chunk-count target ~354/worker)
# Mode: open Optuna (no forced warm-start)
# ----------------------------------------------------------------------------
# TB Conditions (Section refs from S174 spec):
#   1. Labeled as instrumentation/baseline, NOT crash repro                  ✓
#   2. INVALID if READY workers < 24 before dispatch (post-launch watchdog)  ✓
#   3. 10-min post-completion observation window                             ✓
#   4. Forensic bundle captured even if clean                                ✓
#   5. Organic Optuna config recorded in run summary                         ✓
# ============================================================================
set -euo pipefail

cd ~/distributed_prng_analysis

RUN_ID="s174_baseline_pool8_25k_$(date +%Y%m%d_%H%M%S)"
POOL=8
CHUNK=25000
MAX_SEEDS=213000000
MIN_READY_WORKERS=24
RUN_LOG="logs/${RUN_ID}.log"
SUMMARY_FILE="logs/${RUN_ID}_summary.txt"
BUNDLE_DIR="logs/${RUN_ID}_bundle"

mkdir -p logs results "$BUNDLE_DIR"

# ===========================================================================
# Provenance (TB Section 9 — print before Python starts)
# ===========================================================================
{
  echo "=== S174 Baseline (NOT CRASH REPRO — INSTRUMENTATION VALIDATION) ==="
  echo "RUN_ID=$RUN_ID"
  echo "git_sha=$(git rev-parse --short HEAD)"
  echo "started_at=$(date -Is)"
  echo "pool=$POOL"
  echo "chunk_cap=$CHUNK"
  echo "max_seeds=$MAX_SEEDS"
  echo "expected_chunks_total=$((MAX_SEEDS / CHUNK))"
  echo "expected_chunks_per_amd_worker=$((MAX_SEEDS / CHUNK / 24))"
  echo "min_ready_workers_required=$MIN_READY_WORKERS"
  echo "mode=open_optuna (no forced warm-start)"
  echo "purpose=post-reset instrumentation validation, healthy-cluster baseline"
  echo
} | tee "$SUMMARY_FILE"

# ===========================================================================
# Preflight
# ===========================================================================
echo "--- preflight: required CLI flags ---"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source ~/venvs/torch/bin/activate
python3 window_optimizer.py --help > /tmp/window_optimizer_help.txt 2>&1
for flag in \
  "--pwc-transport" "--min-workers" "--worker-pool-size" \
  "--seed-cap-amd" "--seed-cap-nvidia" "--max-seeds"
do
  if ! grep -q -- "$flag" /tmp/window_optimizer_help.txt; then
    echo "MISSING required CLI flag: $flag" | tee -a "$SUMMARY_FILE"
    exit 2
  fi
done
echo "Required CLI flags present" | tee -a "$SUMMARY_FILE"

# ===========================================================================
# Per-run cleanup (S173 instrumentation files only)
# ===========================================================================
echo "--- cleanup ---"
rm -f logs/pwc_startup_diag_simple.jsonl
rm -f logs/s173_job_assignment_ledger.jsonl
rm -f optimal_window_config.json window_optimization_results.json
truncate -s 0 logs/netconsole_all_rigs.log 2>/dev/null || true

# ===========================================================================
# Launch (background, watchdog runs in foreground)
# ===========================================================================
echo "--- launch ---"
echo "RUN_ID=$RUN_ID launch_at=$(date -Is)" | tee -a "$SUMMARY_FILE"

PRNG_PWC_STARTUP_DIAG=1 \
PRNG_PWC_FIRST_ASSIGN_JITTER_SEC=3 \
PRNG_PWC_PER_WORKER_MIN_GAP_SEC=0.02 \
S163_MEM_DEBUG=1 \
PYTHONPATH=. nohup python3 window_optimizer.py \
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
  > "$RUN_LOG" 2>&1 &

PYTHON_PID=$!
echo "python_pid=$PYTHON_PID" | tee -a "$SUMMARY_FILE"
echo "run_log=$RUN_LOG" | tee -a "$SUMMARY_FILE"

# ===========================================================================
# Condition 2: Ready-worker gate (TB Section 5)
# ===========================================================================
# Coordinator's _tcp_wait_ready timeout is 180s. We give it 240s safety margin.
# Watch for either:
#   "[PWC-TCP] N/M workers ready — dispatching"  (success)
#   "[PWC-TCP] ready timeout: N/M workers ready" (continued anyway — abort)
# ===========================================================================
echo "--- ready-worker gate (waiting up to 240s) ---"
GATE_DEADLINE=$(($(date +%s) + 240))
GATE_RESULT=""
READY_COUNT=""
ONLINE_COUNT=""

while [ "$(date +%s)" -lt "$GATE_DEADLINE" ]; do
  # Coordinator dead?
  if ! kill -0 $PYTHON_PID 2>/dev/null; then
    GATE_RESULT="DEAD_BEFORE_GATE"
    break
  fi
  # Dispatching? Match all known variants:
  #   "10/26 workers ready — dispatching"
  #   "[PWC-TCP] 24/26 workers ready — dispatching"
  #   "10 ready worker(s) — dispatching"
  #   "[PWC-TCP] N ready workers — dispatching"
  if grep -qE "(workers? ready|ready workers?|ready worker\(s\)).*dispatching" "$RUN_LOG" 2>/dev/null; then
    READY_LINE=$(grep -E "(workers? ready|ready workers?|ready worker\(s\)).*dispatching" "$RUN_LOG" | tail -1)
    # Try N/M form first; fall back to bare leading int
    READY_COUNT=$(echo "$READY_LINE" | grep -oE "[0-9]+/[0-9]+" | head -1 | cut -d/ -f1)
    if [ -z "$READY_COUNT" ]; then
      READY_COUNT=$(echo "$READY_LINE" | grep -oE "[0-9]+ ready" | head -1 | grep -oE "^[0-9]+")
    fi
    GATE_RESULT="DISPATCHING"
    break
  fi
  # Coordinator timeout? Match variants:
  #   "[PWC-TCP] ready timeout: 10/26 workers ready after 180s"
  #   "ready timeout: ..."
  if grep -qE "ready timeout" "$RUN_LOG" 2>/dev/null; then
    READY_LINE=$(grep -E "ready timeout" "$RUN_LOG" | tail -1)
    READY_COUNT=$(echo "$READY_LINE" | grep -oE "[0-9]+/[0-9]+" | head -1 | cut -d/ -f1)
    if [ -z "$READY_COUNT" ]; then
      READY_COUNT=$(echo "$READY_LINE" | grep -oE "[0-9]+ ready" | head -1 | grep -oE "^[0-9]+")
    fi
    GATE_RESULT="COORDINATOR_TIMEOUT"
    break
  fi
  sleep 2
done

# Capture online count from startup-complete line. Match variants:
#   "[PWC-TCP] startup complete: 26 online, 10 ready"
#   "startup complete: 26 online, 24 ready"
ONLINE_LINE=$(grep -E "startup complete" "$RUN_LOG" 2>/dev/null | tail -1 || true)
if [ -n "$ONLINE_LINE" ]; then
  ONLINE_COUNT=$(echo "$ONLINE_LINE" | grep -oE "[0-9]+ online" | grep -oE "^[0-9]+")
fi

echo "ready_workers_before_dispatch=${READY_COUNT:-unknown}" | tee -a "$SUMMARY_FILE"
echo "online_workers_before_dispatch=${ONLINE_COUNT:-unknown}" | tee -a "$SUMMARY_FILE"
echo "gate_result=$GATE_RESULT" | tee -a "$SUMMARY_FILE"

if [ -z "$GATE_RESULT" ]; then
  echo "INVALID: gate watchdog timed out after 240s with no decision" | tee -a "$SUMMARY_FILE"
  kill -9 $PYTHON_PID 2>/dev/null || true
  echo "INVALID_RUN" > "$BUNDLE_DIR/INVALID_RUN.tag"
  exit 3
fi

if [ "$GATE_RESULT" = "DEAD_BEFORE_GATE" ]; then
  echo "INVALID: window_optimizer.py died before reaching dispatch gate" | tee -a "$SUMMARY_FILE"
  echo "INVALID_RUN" > "$BUNDLE_DIR/INVALID_RUN.tag"
  exit 3
fi

if [ "${READY_COUNT:-0}" -lt "$MIN_READY_WORKERS" ]; then
  echo "INVALID: ready_count=$READY_COUNT < min_required=$MIN_READY_WORKERS" | tee -a "$SUMMARY_FILE"
  kill -TERM $PYTHON_PID 2>/dev/null || true
  sleep 5
  kill -9 $PYTHON_PID 2>/dev/null || true
  echo "INVALID_RUN" > "$BUNDLE_DIR/INVALID_RUN.tag"
  exit 3
fi

echo "Gate passed: $READY_COUNT ready workers — run is VALID" | tee -a "$SUMMARY_FILE"

# ===========================================================================
# Wait for run to complete
# ===========================================================================
echo "--- waiting for trial completion ---"
set +e
wait $PYTHON_PID
PYTHON_EXIT=$?
set -e
echo "python_exit_code=$PYTHON_EXIT" | tee -a "$SUMMARY_FILE"
echo "trial_completed_at=$(date -Is)" | tee -a "$SUMMARY_FILE"

# ===========================================================================
# Condition 3: 10-minute post-completion observation window (TB Section 4, 7)
# ===========================================================================
echo "--- 10-min post-completion observation window ---"
OBS_START=$(date +%s)
OBS_END=$((OBS_START + 600))

while [ "$(date +%s)" -lt "$OBS_END" ]; do
  echo "  [obs] $(date -Is)" >> "$BUNDLE_DIR/observation_window.log"
  for rig in 192.168.3.120 192.168.3.154 192.168.3.162; do
    if timeout 5 ssh -o ConnectTimeout=5 "$rig" 'true' >/dev/null 2>&1; then
      REACH="REACHABLE"
    else
      REACH="UNREACHABLE"
    fi
    echo "    $rig: $REACH" >> "$BUNDLE_DIR/observation_window.log"
  done
  sleep 60
done
echo "post_completion_observation_minutes=10" | tee -a "$SUMMARY_FILE"

# ===========================================================================
# Condition 4: Forensic bundle (capture even if clean)
# ===========================================================================
echo "--- capturing forensic bundle ---"

# Run log
cp "$RUN_LOG" "$BUNDLE_DIR/" 2>/dev/null || true

# Netconsole
cp logs/netconsole_all_rigs.log "$BUNDLE_DIR/" 2>/dev/null || true

# PWC startup diag
cp logs/pwc_startup_diag_simple.jsonl "$BUNDLE_DIR/" 2>/dev/null || true

# PWC assignment ledger (S173 post-dispatch ledger)
cp logs/s173_job_assignment_ledger.jsonl "$BUNDLE_DIR/" 2>/dev/null || true

# Per-rig: rocm-smi, ps, active-job state, GPU bus map, worker logs
for rig in 192.168.3.120 192.168.3.154 192.168.3.162; do
  RIG_DIR="$BUNDLE_DIR/${rig}"
  mkdir -p "$RIG_DIR"
  if timeout 5 ssh -o ConnectTimeout=5 "$rig" 'true' >/dev/null 2>&1; then
    REACH="REACHABLE"
  else
    REACH="UNREACHABLE"
  fi
  echo "$REACH" > "$RIG_DIR/reachability.txt"

  if [ "$REACH" = "REACHABLE" ]; then
    timeout 15 ssh "$rig" 'rocm-smi'                        > "$RIG_DIR/rocm-smi.txt" 2>&1 || true
    timeout 10 ssh "$rig" 'ps auxf | head -200'             > "$RIG_DIR/ps.txt" 2>&1 || true
    timeout 10 ssh "$rig" 'cat /tmp/prng_active_worker_gpu*.json 2>/dev/null'   > "$RIG_DIR/active_worker.txt" 2>&1 || true
    timeout 10 ssh "$rig" 'cat /tmp/prng_gpu_bus_map_gpu*.json 2>/dev/null'     > "$RIG_DIR/gpu_bus_map.txt" 2>&1 || true
    timeout 30 scp "$rig:/tmp/pwc_tcp_worker_*.log" "$RIG_DIR/" 2>/dev/null || true
  fi
done

# rocm-smi worker counts
echo "--- rig health post-run ---" >> "$SUMMARY_FILE"
ROCM_HEALTH=""
ALL_REACHABLE="yes"
for rig in 192.168.3.120 192.168.3.154 192.168.3.162; do
  COUNT=$(timeout 10 ssh -o ConnectTimeout=5 "$rig" 'rocm-smi --showid 2>/dev/null | grep -c "Device Name"' 2>/dev/null || echo "UNREACHABLE")
  echo "$rig: $COUNT GPUs" | tee -a "$SUMMARY_FILE"
  if [ "$COUNT" = "UNREACHABLE" ]; then
    ALL_REACHABLE="no"
  fi
  ROCM_HEALTH="${ROCM_HEALTH}${rig}=${COUNT} "
done

# ===========================================================================
# Condition 5: Record organic Optuna config (TB Section: chosen_config)
# ===========================================================================
CHOSEN_CONFIG=""
if [ -f optimal_window_config.json ]; then
  CHOSEN_CONFIG=$(python3 -c 'import json; c=json.load(open("optimal_window_config.json")); print(f"W{c.get(\"window_size\",\"?\")}_O{c.get(\"offset\",\"?\")}_S{c.get(\"skip_min\",\"?\")}-{c.get(\"skip_max\",\"?\")}_FT{c.get(\"forward_threshold\",\"?\")}_RT{c.get(\"reverse_threshold\",\"?\")}")' 2>/dev/null || echo "PARSE_FAILED")
  cp optimal_window_config.json "$BUNDLE_DIR/" 2>/dev/null || true
fi
echo "chosen_config=$CHOSEN_CONFIG" | tee -a "$SUMMARY_FILE"

# ===========================================================================
# Final summary fields (TB exact-output requirements)
# ===========================================================================
CHUNKS_COMPLETED=$(grep -cE "Chunk [0-9]+: [0-9,]+ seeds" "$RUN_LOG" 2>/dev/null || echo 0)
ELAPSED_FROM_LOG=$(grep -E "Bayesian optimization complete|OPTIMIZATION COMPLETE" "$RUN_LOG" 2>/dev/null | tail -1 || echo "INCOMPLETE")
NETCON_FAULTS=$(grep -cE "GCVM_L2_PROTECTION_FAULT|TransferTableSmu2Dram|qcm fence wait loop" logs/netconsole_all_rigs.log 2>/dev/null || echo 0)
MANIFEST_COUNT=$(ls -d ~/crash_dumps/${RUN_ID}* 2>/dev/null | wc -l || echo 0)

{
  echo
  echo "=== TB Required Summary Fields ==="
  echo "RUN_ID: $RUN_ID"
  echo "git_sha: $(git rev-parse --short HEAD)"
  echo "ready_workers_before_dispatch: ${READY_COUNT:-unknown}"
  echo "online_workers_before_dispatch: ${ONLINE_COUNT:-unknown}"
  echo "chosen_config: $CHOSEN_CONFIG"
  echo "chunks_completed: $CHUNKS_COMPLETED"
  echo "elapsed: see run log final lines"
  echo "fault_manifests_count: $MANIFEST_COUNT"
  echo "netconsole_fault_count: $NETCON_FAULTS"
  echo "post_completion_observation_minutes: 10"
  echo "rocm-smi after run: $ROCM_HEALTH"
  echo "all rigs reachable after run: $ALL_REACHABLE"
} | tee -a "$SUMMARY_FILE"

echo
echo "Bundle: $BUNDLE_DIR"
echo "Summary: $SUMMARY_FILE"
echo "DONE: $RUN_ID"
