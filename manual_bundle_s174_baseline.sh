#!/usr/bin/env bash
# ============================================================================
# S174 Baseline — Manual Forensic Bundle Assembler
# ----------------------------------------------------------------------------
# The S174 baseline launcher died silently inside the gate watchdog loop,
# before reaching observation/bundle/summary. Python ran to completion
# successfully (8520 chunks, 26 GPUs, no faults). This script rebuilds
# the bundle and summary post-hoc from surviving artifacts.
# ----------------------------------------------------------------------------
# Run from Zeus inside ~/distributed_prng_analysis/
# ============================================================================
set -uo pipefail

cd ~/distributed_prng_analysis

RUN_ID="s174_baseline_pool8_25k_20260506_174650"
RUN_LOG="logs/${RUN_ID}.log"
SUMMARY_FILE="logs/${RUN_ID}_summary.txt"
BUNDLE_DIR="logs/${RUN_ID}_bundle"
mkdir -p "$BUNDLE_DIR"

echo "=== Manual bundle assembly: $RUN_ID ==="
echo "started_at: $(date -Is)"

# ---------------------------------------------------------------------------
# 1. Per-rig snapshots (real-time — captures CURRENT state, not at-fault state)
# ---------------------------------------------------------------------------
echo "--- per-rig snapshots ---"
for rig in 192.168.3.120 192.168.3.154 192.168.3.162; do
  RIG_DIR="$BUNDLE_DIR/${rig}"
  mkdir -p "$RIG_DIR"
  if timeout 5 ssh -o ConnectTimeout=5 "$rig" 'true' >/dev/null 2>&1; then
    REACH="REACHABLE"
  else
    REACH="UNREACHABLE"
  fi
  echo "$REACH" > "$RIG_DIR/reachability.txt"
  echo "  $rig: $REACH"
  if [ "$REACH" = "REACHABLE" ]; then
    timeout 15 ssh "$rig" 'rocm-smi'                                          > "$RIG_DIR/rocm-smi.txt" 2>&1 || true
    timeout 10 ssh "$rig" 'ps auxf | head -200'                               > "$RIG_DIR/ps.txt" 2>&1 || true
    timeout 10 ssh "$rig" 'cat /tmp/prng_active_worker_gpu*.json 2>/dev/null' > "$RIG_DIR/active_worker.txt" 2>&1 || true
    timeout 10 ssh "$rig" 'cat /tmp/prng_gpu_bus_map_gpu*.json 2>/dev/null'   > "$RIG_DIR/gpu_bus_map.txt" 2>&1 || true
    timeout 30 scp "$rig:/tmp/pwc_tcp_worker_*.log" "$RIG_DIR/" 2>/dev/null   || true
  fi
done

# ---------------------------------------------------------------------------
# 2. Coordinator-side artifacts
# ---------------------------------------------------------------------------
echo "--- coordinator artifacts ---"
cp "$RUN_LOG"                                  "$BUNDLE_DIR/" 2>/dev/null || true
cp logs/netconsole_all_rigs.log                "$BUNDLE_DIR/" 2>/dev/null || true
cp logs/pwc_startup_diag_simple.jsonl          "$BUNDLE_DIR/" 2>/dev/null || true
cp logs/s173_job_assignment_ledger.jsonl       "$BUNDLE_DIR/" 2>/dev/null || true
cp optimal_window_config.json                  "$BUNDLE_DIR/" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 3. Extract gate / dispatch state from run log
# ---------------------------------------------------------------------------
echo "--- gate state extraction ---"
ONLINE_LINE=$(grep -E "all 26/26 workers online" "$RUN_LOG" | head -1 || true)
DISPATCH_LINE=$(grep -E "workers? ready — dispatching" "$RUN_LOG" | head -1 || true)
STARTUP_COMPLETE=$(grep -E "startup complete:" "$RUN_LOG" | head -1 || true)

ONLINE_COUNT=$(echo "$ONLINE_LINE" | grep -oE "[0-9]+/[0-9]+" | head -1 | cut -d/ -f1 || true)
READY_COUNT=$(echo "$DISPATCH_LINE" | grep -oE "[0-9]+/[0-9]+" | head -1 | cut -d/ -f1 || true)

# ---------------------------------------------------------------------------
# 4. Extract chosen Optuna config
# ---------------------------------------------------------------------------
CHOSEN_CONFIG=""
if [ -f optimal_window_config.json ]; then
  CHOSEN_CONFIG=$(python3 -c 'import json; c=json.load(open("optimal_window_config.json")); print(f"W{c.get(\"window_size\",\"?\")}_O{c.get(\"offset\",\"?\")}_S{c.get(\"skip_min\",\"?\")}-{c.get(\"skip_max\",\"?\")}_FT{c.get(\"forward_threshold\",\"?\")}_RT{c.get(\"reverse_threshold\",\"?\")}")' 2>/dev/null || echo "PARSE_FAILED")
fi

# Backup: parse from log "NEW BEST" line
if [ -z "$CHOSEN_CONFIG" ] || [ "$CHOSEN_CONFIG" = "PARSE_FAILED" ]; then
  CHOSEN_CONFIG=$(grep -E "NEW BEST" "$RUN_LOG" | tail -1 | grep -oE "W[0-9]+_O[0-9]+_[a-z+]+_S[0-9]+-[0-9]+_FT[0-9.]+_RT[0-9.]+" || echo "UNKNOWN")
fi

# ---------------------------------------------------------------------------
# 5. Run statistics from run log
# ---------------------------------------------------------------------------
CHUNKS_COMPLETED=$(grep -cE "Chunk [0-9]+: [0-9,]+ seeds" "$RUN_LOG" 2>/dev/null || echo 0)
COMPLETED_LINE=$(grep -E "Bayesian optimization complete" "$RUN_LOG" | head -1 || true)
NETCON_FAULTS=$(grep -cE "GCVM_L2_PROTECTION_FAULT|TransferTableSmu2Dram|qcm fence wait loop|Failed to retrieve" logs/netconsole_all_rigs.log 2>/dev/null || echo 0)
MANIFEST_COUNT=$(ls -d ~/crash_dumps/${RUN_ID}* 2>/dev/null | wc -l || echo 0)

# ---------------------------------------------------------------------------
# 6. Current rig health (post-hoc — not at-fault)
# ---------------------------------------------------------------------------
ROCM_HEALTH=""
ALL_REACHABLE="yes"
for rig in 192.168.3.120 192.168.3.154 192.168.3.162; do
  COUNT=$(timeout 10 ssh -o ConnectTimeout=5 "$rig" 'rocm-smi --showid 2>/dev/null | grep -c "Device Name"' 2>/dev/null || echo "UNREACHABLE")
  if [ "$COUNT" = "UNREACHABLE" ]; then
    ALL_REACHABLE="no"
  fi
  ROCM_HEALTH="${ROCM_HEALTH}${rig}=${COUNT} "
done

# ---------------------------------------------------------------------------
# 7. Write summary
# ---------------------------------------------------------------------------
{
  echo "=== S174 Baseline (NOT CRASH REPRO — INSTRUMENTATION VALIDATION) ==="
  echo "RUN_ID: $RUN_ID"
  echo "git_sha: $(git rev-parse --short HEAD)"
  echo "started_at: 2026-05-06T17:46:50-07:00"
  echo "pool: 8"
  echo "chunk_cap: 25000"
  echo "max_seeds: 213000000"
  echo "expected_chunks_total: 8520"
  echo "expected_chunks_per_amd_worker: 355"
  echo "mode: open_optuna (no forced warm-start)"
  echo
  echo "=== TB Required Summary Fields ==="
  echo "ready_workers_before_dispatch: ${READY_COUNT:-unknown}"
  echo "online_workers_before_dispatch: ${ONLINE_COUNT:-unknown}"
  echo "chosen_config: $CHOSEN_CONFIG"
  echo "chunks_completed: $CHUNKS_COMPLETED"
  echo "elapsed: see run log final lines (see $COMPLETED_LINE)"
  echo "fault_manifests_count: $MANIFEST_COUNT"
  echo "netconsole_fault_count: $NETCON_FAULTS"
  echo "post_completion_observation_minutes: 0 (launcher died before observation phase — bundle assembled post-hoc)"
  echo "rocm-smi after run: $ROCM_HEALTH"
  echo "all rigs reachable after run: $ALL_REACHABLE"
  echo
  echo "=== Gate state (from run log) ==="
  echo "online_line: $ONLINE_LINE"
  echo "dispatch_line: $DISPATCH_LINE"
  echo "startup_complete_line: $STARTUP_COMPLETE"
  echo
  echo "=== Launcher status ==="
  echo "Launcher exited silently inside gate watchdog loop."
  echo "Bundle, observation, and summary were assembled post-hoc by manual_bundle_s174_baseline.sh."
  echo "Per-rig snapshots in this bundle reflect CURRENT state, not at-fault state."
} > "$SUMMARY_FILE"

cp "$SUMMARY_FILE" "$BUNDLE_DIR/"

# ---------------------------------------------------------------------------
# 8. Tarball
# ---------------------------------------------------------------------------
echo "--- tarball ---"
cd logs
TARBALL="${RUN_ID}_bundle_$(date +%Y%m%d_%H%M%S).tar.gz"
tar czf "/tmp/$TARBALL" "${RUN_ID}_bundle/" "${RUN_ID}.log" "${RUN_ID}_summary.txt"
ls -la "/tmp/$TARBALL"

echo
echo "DONE."
echo "Summary: $SUMMARY_FILE"
echo "Bundle: $BUNDLE_DIR"
echo "Tarball: /tmp/$TARBALL"
echo
echo "Pull to ser8:  scp rzeus:/tmp/$TARBALL ~/Downloads/"
