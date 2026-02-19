# SESSION CHANGELOG — S97
**Date:** 2026-02-19
**Session:** 97
**Focus:** S96B Acceptance Testing, Root Cause Analysis, Data Quality Investigation

---

## Summary

Completed S96B acceptance testing, discovered and fixed manifest param-routing bug,
identified that persistent GPU workers are working but tree models are now the new
bottleneck, and uncovered the root cause of weak ML signal: Step 5 is training on
a stale test file (`test_survivors_10k.json`) via a symlink rather than real Step 3
output. Extensive architectural discussion on holdout_hits, signal quality, and what
the models are actually predicting.

---

## Part 1: S96B Acceptance Tests — Results

### Tests 0–3: PASSED (prior session)
- Syntax clean, markers verified, worker IPC smoke test passed
- Test 3 (S96A baseline, workers OFF): **8m55s**

### Test 4 Issue: Workers Spawned But No Speedup

**Root cause found and fixed during session:**

WATCHER's step-scoped param filter (`allowed_params = set(default_params.keys())`)
silently dropped `persistent_workers: true` because it was not declared in
`agent_manifests/reinforcement.json` → `default_params`.

```
grep result: zero hits on "persistent_workers" in watcher_agent.py → confirmed
```

**Fix applied:** Added `persistent_workers: false` to `reinforcement.json`
`default_params` AND `parameter_bounds`. Manifest bumped to v1.7.0.

```python
# agent_manifests/reinforcement.json
"default_params": {
    ...
    "persistent_workers": false   # ← ADDED
},
"parameter_bounds": {
    ...
    "persistent_workers": {       # ← ADDED
        "type": "bool",
        "default": false,
        "description": "[S96B] Persistent GPU workers for NN trials (5-10x speedup)"
    }
}
```

### Test 4 Re-run: Workers Confirmed Spawning

Screenshots confirmed:
```
[S96B] Spawning 2 persistent GPU workers: [0, 1]
[S96B] GPU-0 worker ready: cuda:0 (NVIDIA GeForce RTX 3080 Ti)
[S96B] GPU-1 worker ready: cuda:0 (NVIDIA GeForce RTX 3080 Ti)
[S96B] 2 workers ready
CLI param overrides: {'compare_models': True, 'trials': 20, 'persistent_workers': True}
```

### S96B Speedup Reality

| Run | Pipeline Time | Wall Clock | Workers |
|-----|--------------|------------|---------|
| Test 3 (S96A baseline) | 8m55s | ~27m18s | OFF |
| Test 4 (S96B) | 8m50s | 26m49s | ON |

**Pipeline speedup: ~0.9% — effectively zero.**

**Why:** Optuna picked up existing study at **trial 181**. By trial 181, TPE is
heavily exploiting (not exploring), so each NN trial is already fast regardless
of S96B. The real bottleneck is now **tree models (CatBoost/LightGBM/XGBoost)**
which run sequentially and are not accelerated by the persistent worker architecture.

**Irony noted:** S96B solved the NN subprocess overhead problem, but in doing so
revealed that tree models are now the new bottleneck. To measure true S96B NN
speedup, need a fresh Optuna study from trial 0.

### S96B Verdict
✅ **Architecturally correct and working.** NN workers spawn, process jobs, shut
down cleanly. The speedup benefit will be measurable once tree model timing is
addressed or a clean study baseline is established.

---

## Part 2: What the Models Are Actually Predicting

### Discovery: Stale Symlink

```bash
ls -lh survivors_with_scores.json
# lrwxrwxrwx → test_survivors_10k.json  (created Feb 14 23:27)
```

Step 5 has been training on `test_survivors_10k.json` — a test file, not real
Step 3 output from the bidirectional survivor pool.

### Test File Analysis

| Metric | Value |
|--------|-------|
| Count | 98,783 survivors |
| prng_type | `'?'` — **field missing entirely** |
| Max holdout_hits | 0.007 |
| Mean holdout_hits | **0.001000** |
| Random baseline | 0.001000 (1/1000) |

**Mean = exact random baseline.** This file has zero signal above random chance.
The model has been learning to rank pure noise against pure noise. Step 6 pools
generated from this are essentially random selections.

### Architectural Clarification (Important)

Corrected understanding of the system goals:

- Bidirectional survivors are **not** noise candidates hoping to contain a true
  seed — they are mathematically validated PRNG state machines that passed three
  independent filters simultaneously (forward sieve, reverse sieve, lane mod CRT).
  Passing all three is combinatorially near-impossible for a random seed.

- `holdout_hits` measures **future predictive power** among these validated
  survivors, not whether a seed is "real." It answers: which of our mathematically
  valid survivors will continue to predict future draws?

- `holdout_hits = 0.007` = 7 exact matches out of 1000 holdout draws = 7× above
  random. Weak but not zero. The issue is there's **no variance** between survivors
  to differentiate them — they all cluster around 0.001.

- The `✅ Model generalizes well to test set!` message is misleading — it means
  no overfitting detected, but says nothing about predictive power. Flagged for
  future UI fix.

### Real bidirectional_survivors.json

```
-rw-rw-r-- Feb 8 20:16  bidirectional_survivors.json  (59MB)
Keys: seed, window_size, offset, skip_min, skip_max, skip_range, sessions,
      trial_number, prng_base, skip_mode, prng_type, forward_count,
      reverse_count, bidirectional_count, bidirectional_selectivity,
      score, intersection_count, intersection_ratio, etc.
```

This file has proper `prng_type` and all survivor metadata. Step 3 has never
been run against it to produce a real `survivors_with_scores.json`.

---

## Part 3: Errors & Warnings Identified

### Error 1: Bash errors at end of Test 4 run
```
Command 'Watch' not found
[S96B]: command not found
-bash: syntax error near unexpected token `('
```
**Cause:** Claude's response text was pasted into terminal as commands instead
of just the test command. Not a code bug — user error from copy-paste. No fix needed.

### Warning 1: LightGBM feature names mismatch
```
sklearn/utils/validation.py: UserWarning: X does not have valid feature names,
but LGBMRegressor was fitted with feature names
```
**Fix:** Pass DataFrame with column names at prediction time, or train on numpy
directly. Low priority — cosmetic, results correct.

### Warning 2: XGBoost device mismatch (fires 3×)
```
xgboost/core.py: Falling back to prediction using DMatrix due to mismatched devices
```
**Fix:** Ensure `device="cuda"` consistent between training and inference.
Low priority — cosmetic, results correct.

### Warning 3: Misleading "Model generalizes well" message
```
✅ Model generalizes well to test set!
Signal Quality: weak
```
**Fix:** Change success message to distinguish overfitting check from signal quality:
```
✅ No overfitting detected (train/test gap within bounds)
⚠️  Signal quality: weak — R² near zero, limited predictive power
```

---

## TODO — Priority Order

### P1 — Critical (blocks real predictions)
- [ ] **Fix survivors_with_scores.json symlink** — remove stale symlink, run real
      Step 3 against `bidirectional_survivors.json`, relink to real output
- [ ] **Run Steps 3→4→5→6** on real survivor data with proper prng_type and features

### P2 — S96B Completion
- [ ] **Measure true S96B NN speedup** — reset Optuna study to trial 0, run
      NN-only compare with `persistent_workers: true` vs false
- [ ] **Git commit S96B** with manifest fix included
      (`reinforcement.json` v1.7.0 not yet committed)

### P3 — Warning Fixes
- [ ] **LightGBM feature names** — pass DataFrame at prediction, not raw numpy
- [ ] **XGBoost device mismatch** — pin `device="cuda"` consistently
- [ ] **Misleading "Model generalizes well" message** — fix display text

### P4 — Phase 3 (Concurrent Trial Batching)
- [ ] **Phase 3A: NN concurrent batching** — Pack 3–5 SurvivorQualityNet instances
      simultaneously per GPU via CUDA streams. Add `train_batch` IPC command to
      nn_gpu_worker.py. Requires architecture bucketing + VRAM budget enforcer.
      Target: 20 NN trials < 30 seconds. Medium priority (NN already fast at ~2 min).
- [ ] **Phase 3B: Tree parallel workers** — Run 2 tree trials simultaneously by
      assigning one subprocess per GPU with pinned CUDA_VISIBLE_DEVICES. GPU
      isolation invariant fully preserved. Target: trees ~12 min (down from ~24 min),
      total compare_models ~13 min (down from ~26 min). HIGH priority — trees are
      93% of current wall clock.
- [ ] **Team Beta review** required for both Phase 3A and 3B before implementation
- [ ] **Benchmark on real data only** — after symlink fix and Step 3 re-run

### P5 — Deferred / Backlog
- [ ] Remove 27 stale project files from Claude project (identified S85/S86)
- [ ] Feature names into `best_model.meta.json` at training time (proper fix vs
      current fallback extraction)
- [ ] Regression diagnostics synthetic data for gate=True activation (S86/S87)
- [ ] Phase 9B.3 policy proposal heuristics
- [ ] Dead code audit (MultiModelTrainer inline path) — comment out, never delete
- [ ] Web dashboard

---

## Key Learnings This Session

1. **Manifest param-routing is a silent failure mode.** Any new CLI flag added to
   a script must ALSO be declared in the step's manifest `default_params` or WATCHER
   will silently drop it. No error, no warning — just the old behavior.

2. **Solving one bottleneck reveals the next.** S96B removed subprocess spawn
   overhead from NN → tree models became the new bottleneck. Classic Amdahl's Law.

3. **Test symlinks left in production are dangerous.** `survivors_with_scores.json`
   pointing to a test file with no prng_type and noise-level signal means every
   pipeline run since Feb 14 has been training on garbage data. Always verify
   data provenance before interpreting ML results.

4. **holdout_hits is a per-seed score, not a pool membership criterion.**
   It measures future predictive power (hits/total_holdout_draws) not sieve quality.
   Sieve quality is already guaranteed by passing the bidirectional filters.

---

## Files Modified This Session

| File | Change |
|------|--------|
| `agent_manifests/reinforcement.json` | Added `persistent_workers` to `default_params` and `parameter_bounds`, bumped to v1.7.0 |

## Files Created This Session

| File | Purpose |
|------|---------|
| `docs/SESSION_CHANGELOG_20260219_S97.md` | This changelog |
| `scripts/s97_backup_and_fix.sh` | Backup + symlink fix + manifest commit |
| `scripts/s97_fix_warnings.py` | LightGBM/XGBoost/message warning patches |
| `scripts/s97_run_step3_real_data.sh` | Step 3 runner against real bidirectional pool |
| `docs/PROPOSAL_PHASE3_CONCURRENT_TRIAL_BATCHING_v1_0.md` | Phase 3A (NN CUDA streams) + 3B (tree parallel workers) |

---

## Git Status

⚠️ `reinforcement.json` manifest change NOT yet committed. Must commit before
next session or the persistent_workers fix will be lost.

Recommended commit:
```bash
cd ~/distributed_prng_analysis
git add agent_manifests/reinforcement.json
git commit -m "fix(s97): add persistent_workers to reinforcement manifest — was silently dropped by WATCHER param filter"
git push origin main && git push public main
```

---

## Next Session Start Checklist

1. Commit manifest fix (above)
2. Verify `bidirectional_survivors.json` still intact (59MB, Feb 8)
3. Check `holdout_history.json` exists and is current (6.8KB, Feb 8)
4. Run Step 3 → real `survivors_with_scores.json`
5. Run Steps 4→5→6
6. Compare new holdout_hits distribution vs current noise baseline

---

*Session S97 — Team Alpha*
*S96B persistent workers: architecturally complete, bottleneck shifted to tree models*
*Critical finding: survivors_with_scores.json is a stale test symlink — fix before any further ML work*
