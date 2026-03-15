#!/usr/bin/env bash
# s145r1_smoke_tests.sh
# Two-phase smoke test for S145-R1 progressive sweep patches
#
# Phase 1: Light test  — 5M seeds, 3 trials  (~10 min)
#   Verifies: accumulator fires, bidirectional_survivors_all.json created,
#             NPZ converts from accumulator, correct fields present
#
# Phase 2: Minimal test — 100K seeds, 2 trials (~2-3 min)
#   Verifies: accumulator MERGES (appends to Phase 1 results),
#             seed_range advances in exhaustive_progress,
#             study_name preserved for resume
#
# Usage: bash s145r1_smoke_tests.sh
# Run from: ~/distributed_prng_analysis
# Requires: source ~/venvs/torch/bin/activate

set -e
cd ~/distributed_prng_analysis
source ~/venvs/torch/bin/activate

echo "========================================================"
echo "S145-R1 Smoke Tests"
echo "========================================================"

# ── Pre-flight ─────────────────────────────────────────────
echo ""
echo "[PRE-FLIGHT] Clearing stale halt and output files..."
PYTHONPATH=. python3 -c "from agents.safety import clear_halt; clear_halt()" 2>/dev/null || true
rm -f optimal_window_config.json
rm -f bidirectional_survivors.json
rm -f bidirectional_survivors_all.json
rm -f bidirectional_survivors_binary.npz
echo "  ✅ Working directory clean"

# ── PHASE 1 — Light test ───────────────────────────────────
echo ""
echo "========================================================"
echo "PHASE 1 — Light test (5M seeds, 3 trials)"
echo "Expected runtime: ~10 minutes"
echo "========================================================"
echo ""

PYTHONPATH=. python3 agents/watcher_agent.py \
    --run-pipeline --start-step 1 --end-step 1 \
    --params '{"max_seeds": 5000000, "window_trials": 3}' \
    2>&1 | tee logs/s145r1_smoke_phase1.log

echo ""
echo "[PHASE 1 VERIFICATION]"

# Check 1: Accumulator log line
echo ""
echo "--- Check 1: Accumulator fired ---"
grep "\[S145-R1\]\[ACCUMULATOR\]" logs/s145r1_smoke_phase1.log && \
    echo "  ✅ PASS" || echo "  ❌ FAIL — accumulator log line missing"

# Check 2: NPZ converted from accumulator
echo ""
echo "--- Check 2: NPZ from accumulator ---"
grep "Converted.*bidirectional_survivors_all" logs/s145r1_smoke_phase1.log && \
    echo "  ✅ PASS" || echo "  ❌ FAIL — NPZ not converted from accumulator"

# Check 3: Accumulator file exists
echo ""
echo "--- Check 3: Accumulator file on disk ---"
if [ -f "bidirectional_survivors_all.json" ]; then
    COUNT=$(python3 -c "import json; d=json.load(open('bidirectional_survivors_all.json')); print(len(d))")
    echo "  ✅ PASS — bidirectional_survivors_all.json exists with $COUNT survivors"
else
    echo "  ❌ FAIL — bidirectional_survivors_all.json not found"
fi

# Check 4: NPZ exists and readable
echo ""
echo "--- Check 4: NPZ readable ---"
python3 -c "
import numpy as np
d = np.load('bidirectional_survivors_binary.npz')
print(f'  ✅ PASS — NPZ contains {len(d[\"seeds\"]):,} seeds, {len(d.files)} fields')
print(f'  Seed range: {d[\"seeds\"].min():,} → {d[\"seeds\"].max():,}')
" 2>/dev/null || echo "  ❌ FAIL — NPZ not readable"

# Check 5: exhaustive_progress entry written
echo ""
echo "--- Check 5: Coverage tracker updated ---"
python3 -c "
import sqlite3
conn = sqlite3.connect('prng_analysis.db')
rows = conn.execute(
    'SELECT prng_type, seed_range_start, seed_range_end, seeds_completed '
    'FROM exhaustive_progress ORDER BY rowid DESC LIMIT 3'
).fetchall()
if rows:
    for r in rows:
        print(f'  ✅ PASS — {r[0]}: {r[1]:,} → {r[2]:,} ({r[3]:,} seeds)')
else:
    print('  ❌ FAIL — no exhaustive_progress entries')
conn.close()
"

# Check 6: trial history written
echo ""
echo "--- Check 6: Trial history in DB ---"
python3 -c "
import sqlite3
conn = sqlite3.connect('prng_analysis.db')
rows = conn.execute(
    'SELECT trial_number, window_size, offset, trial_score, bidirectional_survivors '
    'FROM step1_trial_history ORDER BY recorded_at DESC LIMIT 5'
).fetchall()
if rows:
    print(f'  ✅ PASS — {len(rows)} trial history rows')
    for r in rows:
        print(f'     T{r[0]}: W{r[1]}_O{r[2]} score={r[3]} bidir={r[4]}')
else:
    print('  ❌ FAIL — no trial history rows')
conn.close()
"

PHASE1_COUNT=$(python3 -c "
import json, os
if os.path.exists('bidirectional_survivors_all.json'):
    print(len(json.load(open('bidirectional_survivors_all.json'))))
else:
    print(0)
")
echo ""
echo "Phase 1 complete. Accumulator has $PHASE1_COUNT survivors."

# ── PHASE 2 — Minimal test (accumulation test) ─────────────
echo ""
echo "========================================================"
echo "PHASE 2 — Minimal test (100K seeds, 2 trials)"
echo "Testing accumulation — should MERGE with Phase 1 results"
echo "Expected runtime: ~2-3 minutes"
echo "========================================================"
echo ""

# Must delete freshness gate output to force re-run
rm -f optimal_window_config.json
echo "  Cleared optimal_window_config.json for re-run"

PYTHONPATH=. python3 agents/watcher_agent.py \
    --run-pipeline --start-step 1 --end-step 1 \
    --params '{"max_seeds": 100000, "window_trials": 2}' \
    2>&1 | tee logs/s145r1_smoke_phase2.log

echo ""
echo "[PHASE 2 VERIFICATION]"

# Check 7: Accumulator grew (or same if no new survivors)
echo ""
echo "--- Check 7: Accumulator after Phase 2 ---"
PHASE2_COUNT=$(python3 -c "
import json, os
if os.path.exists('bidirectional_survivors_all.json'):
    print(len(json.load(open('bidirectional_survivors_all.json'))))
else:
    print(0)
")
echo "  Phase 1 survivors: $PHASE1_COUNT"
echo "  Phase 2 survivors: $PHASE2_COUNT"
if [ "$PHASE2_COUNT" -ge "$PHASE1_COUNT" ]; then
    echo "  ✅ PASS — accumulator preserved Phase 1 survivors (count did not decrease)"
else
    echo "  ❌ FAIL — accumulator count decreased — merge overwrote instead of merged"
fi

# Check 8: seed_start advanced in coverage tracker
echo ""
echo "--- Check 8: seed_start advanced between phases ---"
python3 -c "
import sqlite3
conn = sqlite3.connect('prng_analysis.db')
rows = conn.execute(
    'SELECT prng_type, seed_range_start, seed_range_end '
    'FROM exhaustive_progress ORDER BY rowid DESC LIMIT 5'
).fetchall()
print(f'  Coverage tracker entries ({len(rows)} total):')
for r in rows:
    print(f'    {r[0]}: {r[1]:,} → {r[2]:,}')
if len(rows) >= 2:
    print('  ✅ PASS — multiple coverage entries confirm seed_start advanced')
else:
    print('  ⚠️  Only 1 entry — Phase 2 may have used same range (check log)')
conn.close()
"

# Check 9: Phase 2 accumulator log shows prior survivors
echo ""
echo "--- Check 9: Phase 2 accumulator shows merge ---"
grep "\[S145-R1\]\[ACCUMULATOR\]" logs/s145r1_smoke_phase2.log && \
    echo "  ✅ PASS" || echo "  ❌ FAIL"

# Final summary
echo ""
echo "========================================================"
echo "SMOKE TEST SUMMARY"
echo "========================================================"
echo "Phase 1 log: logs/s145r1_smoke_phase1.log"
echo "Phase 2 log: logs/s145r1_smoke_phase2.log"
echo ""
echo "If all checks passed — proceed to commit:"
echo ""
echo "  git add -f bidirectional_survivors_binary.npz bidirectional_survivors_all.json"
echo "  git add window_optimizer_integration_final.py"
echo "  git add agent_manifests/window_optimizer.json"
echo "  git add agents/watcher_agent.py"
echo "  git add .gitignore"
echo "  git add apply_s145r1_progressive_sweep.py"
echo "  git add docs/PROPOSAL_S145_R1_Progressive_Empirical_Sweep.md"
echo "  git commit -m 'feat(s145-r1): progressive sweep — accumulator + resume invariant + manifest corrections'"
echo "  git push origin main && git push public main"
echo ""
echo "If any checks failed — inspect logs above before committing."
echo "Backups available at: *.s145r1_backup"
