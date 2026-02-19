# SESSION CHANGELOG — S93
**Date:** 2026-02-15
**Focus:** Fix WATCHER Retry-Without-Rerun Bug (Bug A) + Bug B + Bug C

---

## Summary

Fixed three bugs identified by Team Beta post-run analysis in S92. Bug A was the
CRITICAL defect explaining why the NN retry-learn-retry loop has been non-functional
for multiple sessions — the NN was only training once but the skip counter reached 3.

## Bugs Fixed

### Bug A: WATCHER Retry-Without-Rerun (CRITICAL)

**Root Cause:** When health check returns RETRY in `run_pipeline()`, the loop sets
`self.current_step = 5` and continues. But `run_step(5)` hits the freshness gate
(`check_output_freshness()`) which sees the Step 5 output is newer than its inputs
(it was just produced!) and returns `{"skipped": True}`. Step 5 never re-executes.
Meanwhile, the health check runs again on the same stale diagnostics, calling
`_check_skip_registry()` which increments `consecutive_critical`. Counter reaches 3
after only 1 real training run.

**Fix (Team Beta Option 1):** Added `_invalidate_step_freshness(step)` method that
touches the primary input file to make the output appear stale. Called in the RETRY
branch before setting `current_step = 5`. This forces `check_output_freshness()` to
return STALE on the next iteration, so `run_step(5)` actually re-executes.

**Patches:** A1 (add method), A2 (call on retry), A3 (skip-registry real-attempts guard)

A3 is Team Beta's defense-in-depth suggestion: the health check block in `run_pipeline()`
now checks if `run_step(5)` returned `skipped: True`. If so, health check is skipped
entirely — stale diagnostics have no new information, so the skip counter should not
increment. This makes the system resilient even if freshness invalidation ever regresses.

### Bug B: Health Check Model Mismatch

**Root Cause:** After `--compare-models` selects catboost as winner, the health check
reads `model_type` from the diagnostics file which may still say `neural_net` (the
last model evaluated). This triggers NN-specific retries on a catboost winner.

**Fix:** Before evaluating single-model vs multi-model format, check for
`compare_models_summary.json` sidecar. If it exists and has a winner, override
the `model_type` in the diagnostics dict.

**Patch:** B1 (winner override)

### Bug C: diagnostics_llm_analyzer.py History Type Guard

**Root Cause:** History loop in `build_diagnostics_prompt()` catches `JSONDecodeError`
+ `KeyError` but not `AttributeError`. When a history file contains a bare string
instead of a dict, `hist.get()` crashes with `AttributeError`.

**Fix:** Added `isinstance(hist, dict)` guard before accessing `.get()`. Broadened
exception catch to include `AttributeError` and `TypeError`.

**Patch:** C1 (type guard)

---

## Files Modified

| File | Change |
|------|--------|
| `agents/watcher_agent.py` | +1 method (`_invalidate_step_freshness` with mtime logging), +2 lines RETRY branch, +3 lines skipped-execution guard |
| `training_health_check.py` | +15 lines: compare_models_summary winner override |
| `diagnostics_llm_analyzer.py` | +5 lines: isinstance guard + broadened exception |

## Backups Created

| File | Purpose |
|------|---------|
| `agents/watcher_agent.py.pre_s93_bug_fixes` | Pre-patch safety |
| `training_health_check.py.pre_s93_bug_fixes` | Pre-patch safety |
| `diagnostics_llm_analyzer.py.pre_s93_bug_fixes` | Pre-patch safety |

---

## Deployment

```bash
scp ~/Downloads/apply_s93_bug_a_b_c_fixes.py rzeus:~/distributed_prng_analysis/
scp ~/Downloads/SESSION_CHANGELOG_20260215_S93.md rzeus:~/distributed_prng_analysis/docs/

ssh rzeus
cd ~/distributed_prng_analysis
source ~/venvs/torch/bin/activate
python3 apply_s93_bug_a_b_c_fixes.py
```

## Verification

```bash
# 1. Reset skip registry to 0 before testing
python3 -c "
import json
reg = {'neural_net': {'consecutive_critical': 0, 'last_critical': None}}
with open('diagnostics_outputs/model_skip_registry.json', 'w') as f:
    json.dump(reg, f, indent=2)
print('Skip registry reset')
"

# 2. Run pipeline — watch for [S93][BUG-A] log lines
PYTHONPATH=. python3 agents/watcher_agent.py \
  --run-pipeline --start-step 5 --end-step 6 \
  --params '{"trials":1,"enable_diagnostics":true}'

# 3. Verify freshness invalidation occurred
grep "BUG-A" /tmp/watcher*.log 2>/dev/null || echo "Check terminal output for [S93][BUG-A]"

# 4. Check skip registry — should NOT hit 3 from 1 real run
python3 training_health_check.py --status
```

## Git

```bash
git add agents/watcher_agent.py training_health_check.py diagnostics_llm_analyzer.py
git add docs/SESSION_CHANGELOG_20260215_S93.md
git commit -m "fix(s93): Bug A/B/C — freshness invalidation on RETRY, winner model override, history type guard

Bug A (CRITICAL): Step 5 freshness gate blocked re-execution on RETRY.
consecutive_critical reached 3 from 1 real training run.
Fix: _invalidate_step_freshness() touches primary input before retry loop-back.
Also: log before/after mtime on invalidation; skip health-check entirely
when Step 5 was freshness-skipped (A3 defense-in-depth).

Bug B: Health check evaluated CLI model_type, not compare-models winner.
Fix: Read winner from compare_models_summary.json sidecar.

Bug C: History loop crashed on non-dict JSON entries (AttributeError).
Fix: isinstance guard + broadened exception catch."
git push origin main && git push public main
```

---

## Next Session Priorities

1. **Priority 2: Increase NN Optuna Trials** — Bump from 1 to 15 in reinforcement.json
2. **Priority 5: Recalibrate NN Diagnostic Thresholds** post-Category-B
3. **Priority 6: Deferred Backlog** — regression diagnostics, dead code audit, 27 stale files

---

*Session 93 — Team Alpha*
