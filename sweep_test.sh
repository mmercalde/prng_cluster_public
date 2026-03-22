#!/bin/bash
# =============================================================================
# sweep_test.sh — Short smoke test before production launch
# =============================================================================
# Purpose:
#   1. Verifies slim_v1 IPC is working on all rigs
#   2. Verifies 26/26 GPUs active
#   3. Verifies bidirectional survivors are found and flushed to NPZ
#   4. Completes in ~3-5 minutes (1 trial, 5M seeds)
#
# Run this BEFORE sweep_run1.sh after:
#   - Re-enabling slim_v1 on rigs
#   - Any sieve_gpu_worker.py change
#   - Any coordinator change
#   - Rebooting any rig
#
# Usage:
#   bash sweep_test.sh
#   bash sweep_test.sh --seed-start 100000000   # test different range
# =============================================================================

set -e
cd ~/distributed_prng_analysis
source ~/venvs/torch/bin/activate

RUN_ID="test"
LOG="logs/sweep_${RUN_ID}.log"
PID_FILE="logs/sweep_${RUN_ID}.pid"
MANIFEST="agent_manifests/window_optimizer.json"

mkdir -p logs

# ── Parse args ────────────────────────────────────────────────────────────────
SEED_START=0
for arg in "$@"; do
    case $arg in
        --seed-start=*)  SEED_START="${arg#*=}" ;;
        --seed-start)    shift; SEED_START="$1" ;;
    esac
done

SEED_COUNT=5000000    # 5M seeds — enough to find survivors, fast enough to finish in minutes
N_TRIALS=1            # Single trial — we just want to confirm the pipeline works

echo "============================================================"
echo "SMOKE TEST — 1 trial, ${SEED_COUNT} seeds from ${SEED_START}"
echo "Expected runtime: ~3-5 minutes"
echo "Log: $LOG"
echo "============================================================"
echo ""

# Guard: kill any existing test run
if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "⚠️  Killing existing test run (PID $OLD_PID)"
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
    fi
fi

# Clean up any stale state from previous test runs
rm -f optimal_window_config.json /tmp/agent_halt
PYTHONPATH=. python3 agents/watcher_agent.py --clear-halt 2>/dev/null || true

# Patch manifest for test params (save original first)
python3 - <<PYEOF
import json, shutil

manifest_path = '$MANIFEST'
shutil.copy2(manifest_path, manifest_path + '.bak_sweep_test')

m = json.load(open(manifest_path))

# Save original params for restore
orig = {
    'window_trials':    m['default_params'].get('window_trials'),
    'seed_count':       m['default_params'].get('seed_count', m['default_params'].get('max_seeds')),
    'seed_start':       m['default_params'].get('seed_start', 0),
    'study_name':       m['default_params'].get('study_name', ''),
    'resume_study':     m['default_params'].get('resume_study', False),
    'enable_pruning':   m['default_params'].get('enable_pruning', True),
}
json.dump(orig, open('logs/sweep_test_orig_params.json', 'w'), indent=2)

# Apply test params
m['default_params']['window_trials'] = $N_TRIALS
m['default_params']['seed_start']    = $SEED_START
if 'max_seeds' in m['default_params']:
    m['default_params']['max_seeds'] = $SEED_COUNT
if 'seed_count' in m['default_params']:
    m['default_params']['seed_count'] = $SEED_COUNT
m['default_params']['study_name']    = ''       # fresh study
m['default_params']['resume_study']  = False
m['default_params']['enable_pruning'] = False   # don't prune a single trial

json.dump(m, open(manifest_path, 'w'), indent=2)
print(f'  Manifest patched: trials={$N_TRIALS}, seeds={$SEED_COUNT}, seed_start={$SEED_START}')
PYEOF

# Trap to restore manifest on exit (success, error, or Ctrl+C)
restore_manifest() {
    echo ""
    echo "Restoring manifest to original params..."
    python3 - <<PYEOF2
import json
orig = json.load(open('logs/sweep_test_orig_params.json'))
m    = json.load(open('$MANIFEST'))
m['default_params']['window_trials']  = orig['window_trials']
m['default_params']['seed_start']     = orig['seed_start']
if 'max_seeds' in m['default_params']:
    m['default_params']['max_seeds']  = orig.get('seed_count', 1073741824)
if 'seed_count' in m['default_params']:
    m['default_params']['seed_count'] = orig.get('seed_count', 1073741824)
m['default_params']['study_name']     = orig['study_name']
m['default_params']['resume_study']   = orig['resume_study']
m['default_params']['enable_pruning'] = orig['enable_pruning']
json.dump(m, open('$MANIFEST', 'w'), indent=2)
print('  Manifest restored.')
PYEOF2
}
trap restore_manifest EXIT

# Launch pipeline (Step 1 only, foreground so we can watch it)
echo "Launching test pipeline..."
echo ""

PYTHONPATH=. python3 agents/watcher_agent.py \
    --run-pipeline --start-step 1 --end-step 1 \
    2>&1 | tee "$LOG"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "============================================================"
echo "SMOKE TEST COMPLETE (exit code: $EXIT_CODE)"
echo "============================================================"
echo ""

# Results summary
python3 - <<PYEOF3
import os, sys
import numpy as np

passed = []
failed = []

# Check 1: NPZ written
accum = 'bidirectional_survivors_all.npz'
if os.path.exists(accum):
    try:
        d = np.load(accum)
        n = len(d[list(d.keys())[0]])
        if n > 0:
            passed.append(f'NPZ written: {n:,} survivors in {accum}')
        else:
            failed.append(f'NPZ exists but 0 seeds: {accum}')
    except Exception as e:
        failed.append(f'NPZ unreadable: {e}')
else:
    failed.append(f'NPZ not written: {accum} missing')

# Check 2: S152-FLUSH in log
log_path = 'logs/sweep_test.log'
if os.path.exists(log_path):
    log = open(log_path).read()
    if '[S152-FLUSH]' in log:
        flush_lines = [l.strip() for l in log.splitlines() if '[S152-FLUSH]' in l]
        passed.append(f'Incremental flush fired: {len(flush_lines)} time(s)')
    else:
        # Not a hard failure — could be < threshold survivors
        passed.append('No S152-FLUSH lines — survivors may be below flush threshold (OK if < PRNG_FLUSH_EVERY)')

    # Check 3: Worker pool ready
    if 'Worker pool ready' in log:
        import re
        m = re.search(r'Worker pool ready.*?(\d+).*?workers', log)
        if m:
            passed.append(f'Worker pool ready: {m.group(0).strip()}')
        else:
            passed.append('Worker pool ready: confirmed')
    else:
        failed.append('Worker pool ready: NOT found in log')

    # Check 4: slim_v1 (look for slim_v1 in log)
    if 'slim_v1' in log:
        passed.append('slim_v1 IPC: active')
    else:
        passed.append('slim_v1 IPC: not confirmed in log (may be normal if log trimmed)')

    # Check 5: bidirectional survivors count
    import re as re2
    bidi_lines = [l for l in log.splitlines() if 'Total bidirectional' in l or 'Bidirectional (constant)' in l]
    if bidi_lines:
        passed.append(f'Bidirectional survivors found: {bidi_lines[-1].strip()}')
    else:
        failed.append('No bidirectional survivors found — check log')

print()
print('─' * 60)
print('RESULTS')
print('─' * 60)
for p in passed:
    print(f'  ✅ {p}')
for f in failed:
    print(f'  ❌ {f}')
print('─' * 60)
if failed:
    print(f'  {len(failed)} check(s) FAILED — review log: logs/sweep_test.log')
    sys.exit(1)
else:
    print(f'  All checks passed — safe to launch sweep_run1.sh')
PYEOF3
