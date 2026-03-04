# CHAPTER 1 PATCH — S114
## Window Optimizer: Warm-Start Enqueue + Study Resume Flag
**Version:** v3.2.0 (patch over v3.1)
**Session:** S114 — 2026-03-02
**Status:** Implemented and validated

---

## Summary of Changes

Two new capabilities added to the Bayesian optimization path in S114:

1. **Warm-Start Enqueue** — Seeds Optuna TPE with known-good config on trial 0
2. **`--resume-study` Flag** — Enables pause/resume of Optuna studies across sessions

---

## Section 4 Addendum: Bayesian Optimization — Warm-Start Enqueue

### Problem Solved
Optuna TPE requires positive signal to guide search. With a sparse signal space
(W8_O43 found in only 1 of 50 trials in S112), TPE may run 20-40 blind trials
before randomly discovering a survivor-producing configuration. Each blind trial
at 10M seeds costs ~67 seconds of cluster time.

### Implementation
After `study = optuna.create_study(...)` in `window_optimizer_bayesian.py`:

```python
# Warm-start: enqueue known-good S112 config as trial 0
# Gives TPE immediate positive signal, guiding search toward small windows
# Skipped automatically on --resume-study (trial already exists in DB)
study.enqueue_trial({
    'window_size': 8,
    'offset': 43,
    'skip_min': 5,
    'skip_max': 56,
    'forward_threshold': 0.49,
    'reverse_threshold': 0.49
})
```

### Behavior
- Trial 0 always tests W8_O43_S5-56 on fresh study starts
- TPE model immediately learns: small windows near 8, offset near 43 → survivors
- Subsequent trials (1-99) guided toward that neighborhood
- Automatically skipped when `resume_study=True` (trial 0 already in DB)
- Per-run only: each fresh study gets the enqueue; resume studies do not

### Validated Results (S114)
```
Trial 0: W8_O43_S5-56 → 43 bidirectional survivors (67 seconds)
Trial 3: W3_O0_S10-28 → 143,959 bidirectional survivors
```
TPE immediately guided search toward small windows after trial 0 positive signal.

---

## Section 10 Addendum: CLI Interface — --resume-study Flag

### New Parameter
```
--resume-study    Resume most recent incomplete Optuna study DB.
                  Default: False (fresh study every run)
                  Type: flag (boolean, store_true)
```

### Full CLI Reference (Updated)
```bash
# Standard fresh run (default — no flag needed)
python3 window_optimizer.py \
    --strategy bayesian \
    --lottery-file daily3.json \
    --trials 100 \
    --max-seeds 10000000 \
    --prng-type java_lcg \
    --test-both-modes

# Resume most recent incomplete study
python3 window_optimizer.py \
    --strategy bayesian \
    --lottery-file daily3.json \
    --trials 100 \
    --resume-study

# Via WATCHER (fresh run — default)
--params '{"lottery_file": "daily3.json"}'

# Via WATCHER (resume)
--params '{"lottery_file": "daily3.json", "resume_study": true}'
```

### Resume Logic (window_optimizer_bayesian.py)
```
1. Scan optuna_studies/window_opt_*.db sorted by mtime (newest first)
2. Load most recent DB
3. Check: completed_trials > 0 AND completed_trials < max_iterations
4. If resumable:
   - load_if_exists=True
   - Skip warm-start enqueue (trial 0 already in DB)
   - _trials_to_run = max_iterations - completed_trials
5. If not resumable (empty or already finished):
   - Fall back to fresh study
   - Enqueue warm-start as trial 0
   - _trials_to_run = max_iterations
```

### When to Use --resume-study

| Scenario | Use Flag? | Reason |
|----------|-----------|--------|
| Normal pipeline run | ❌ No | Fresh study = clean audit artifact |
| Session interrupted mid-run | ✅ Yes | Continue from checkpoint |
| Extending trial count on good study | ✅ Yes | Leverage existing TPE model |
| Changed PRNG type | ❌ No | Old trials would corrupt TPE |
| Changed lottery data | ❌ No | Old trials from different dataset |
| Changed thresholds significantly | ❌ No | Old trials not comparable |
| Debugging / isolation testing | ❌ No | Clean slate preferred |

### Design Rationale
Auto-resume was considered and rejected:
- Risk of loading stale studies from weeks ago
- PRNG type, data, or threshold changes would corrupt TPE model silently
- "Each run is a clean audit artifact" principle
- Consistent with WATCHER explicit-params architecture

Flag-based opt-in ensures resume is always a deliberate, documented decision.

---

## Section 11 Addendum: Manifest Parameters (window_optimizer.json)

### New Parameters Added

```json
"search_bounds": {
    "resume_study": {
        "type": "bool",
        "default": false,
        "description": "Resume most recent incomplete Optuna study DB. False = fresh study every run (default, safe). True = continue from last checkpoint."
    }
}

"default_params": {
    "resume_study": false,
    "trials": 100
}
```

Note: `trials` default updated from 50 to 100 in S114.

---

## Key Discovery: Discrete PRNG Regime Structure (S114)

During S114 Step 1 validation, two distinct survivor regimes were discovered:

| Window | Offset | Survivors | Interpretation |
|--------|--------|-----------|----------------|
| 3 | 0 | 143,959 | Short reseed cycle regime |
| 8 | 43 | 43-53 | Long persistence regime |
| 31+ | any | 0 | No signal |

The signal is **discrete, not continuous** — intermediate windows (31, 243, 489)
produce zero survivors. This suggests the lottery PRNG reseeds at specific draw
intervals (3 and 8), not a smooth temporal decay.

This is consistent with Java LCG behavior and provides two independent validation
points for the PRNG hypothesis.

---

## Files Modified (S114 Patch)

| File | Change |
|------|--------|
| `window_optimizer.py` | Added `--resume-study` CLI argument |
| `window_optimizer_bayesian.py` | Added warm-start enqueue + resume logic |
| `agent_manifests/window_optimizer.json` | Added resume_study param, trials 50→100 |

**Patcher:** `apply_resume_patch_v2.py`

---

*Chapter 1 Patch S114 — 2026-03-02*
*Applies over CHAPTER_1_WINDOW_OPTIMIZER.md v3.1*
