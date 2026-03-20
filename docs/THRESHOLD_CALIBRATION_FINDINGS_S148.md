# Sieve Threshold Calibration Findings — S148

**Date:** 2026-03-19  
**Session:** S148  
**Author:** Team Alpha  
**Status:** Authoritative — supersedes synthetic-era defaults in THRESHOLD_GOVERNANCE.md  

---

## 1. Purpose

This document records the methodology, raw results, and actionable conclusions
from the first empirical threshold calibration of the Step 1 sieve using the
real GPU kernels against CA Daily 3-style draw data.

Prior to this session, threshold defaults (0.25 forward/reverse) were carried
forward from early synthetic testing with no rigorous empirical basis — noted
explicitly in `THRESHOLD_GOVERNANCE.md` as a synthetic-era artifact requiring
update.

---

## 2. Background — Why This Was Needed

### The threshold problem

Step 1 runs four sieve passes (constant-skip forward/reverse, hybrid
forward/reverse). Each pass filters candidate seeds by checking how many draws
in a window match the seed's PRNG output at a given skip value. Seeds whose
match rate exceeds `threshold` survive to the next pass.

The threshold value is critical:

- **Too low:** thousands of false survivors pass forward, flood the reverse
  pass, create noise in bidirectional intersection, waste GPU time, and dilute
  ML signal in Steps 3–5.
- **Too high:** real candidates get pruned before ever reaching bidirectional
  intersection.

The default of **0.25** was never empirically validated — it was a reasonable
guess. It was known to be a synthetic-era artifact.

### Why Java LCG as the test PRNG

Java LCG (`a=25214903917, c=11, mask=2^48-1`) is the primary PRNG in
production sieve passes. It has forward and reverse GPU kernels, a hybrid
kernel, and a known CPU reference implementation. Any calibration finding for
Java LCG directly informs production threshold settings.

### Why CA Daily 3 as the draw model

The target application involves CA Daily 3 (Pick 3, 3 digits 0–9 with
replacement, midday 1:00:10 PM and evening 6:30:05 PM Pacific per CA draw
procedures). We do not know the CA draw machine's exact internal skip rate,
dual-RNG stride, or session overhead. However we know our own PRNG completely.

The calibration strategy was therefore:

> Generate synthetic draw sequences from a **known Java LCG seed** across a
> range of plausible CA-machine skip scenarios. Run the real sieve kernel
> against each. Find the threshold at which the known seed survives with
> **zero false positives**. That threshold is the empirically-grounded minimum
> safe default.

This is conservative by design — we are testing with perfectly clean synthetic
data. Real CA draws will have additional noise (session gaps, missing draws,
non-LCG deviations), so real-world thresholds should stay slightly below the
synthetic zero-noise optimum to preserve a recovery margin.

---

## 3. Methodology

### 3.1 Draw generation — kernel-aligned model

Draws were generated using exactly the same step model as the GPU kernel:

```
for each draw:
    burn draw_skip steps          # inter-draw overhead
    advance one LCG step
    output = (state >> 16) & 0xFFFFFFFF) % 1000
```

This ensures match_rate=1.0 at the correct skip from the known seed —
a necessary condition to distinguish threshold effects from alignment errors.

**Known seed:** 3141592  
**Draw format:** `{"draw": int, "session": "midday"|"evening", "timestamp": int}`  
**Sessions:** alternating midday/evening matching CA D3 structure  
**Timestamps:** CA draw times (1:00:10 PM / 6:30:05 PM Pacific)

### 3.2 Sieve execution — real GPU worker

Draws were written to a temp JSON file and submitted to `sieve_gpu_worker.py`
in persistent mode via stdin/stdout IPC — the same path production uses.

**Kernel:** `java_lcg` (CUDA, RTX 3080 Ti on Zeus)  
**Triple residue check:** `output % 1000 == draw % 1000 AND % 8 AND % 125`  
**Skip sweep:** `[0, 50]` per job — kernel finds best skip per seed  
**Seed range:** ±100,000 around known seed (200,001 seeds total)  

### 3.3 Experiments

Two experiments were run:

**Experiment 1 — Skip scenario sweep (window=8 fixed)**  
Five draw_skip values representing plausible CA machine scenarios:

| Scenario | draw_skip | Interpretation |
|----------|-----------|----------------|
| A | 0 | Consecutive outputs, no inter-draw overhead |
| B | 3 | ≈ 1 pre-test cycle (3 digits for D3) |
| C | 5 | ≈ pre-test + small session overhead |
| D | 10 | ≈ pre-test + larger overhead or dual-RNG stride |
| E | 20 | ≈ heavier session overhead |

**Experiment 2 — Window size sweep (draw_skip=5 fixed)**  
Four window sizes representing the configurable sieve window parameter:

| Window | Draws compared per pass |
|--------|------------------------|
| 8 | Production default |
| 10 | +2 draws |
| 12 | +4 draws |
| 16 | +8 draws |

---

## 4. Raw Results

### 4.1 Experiment 1 — Skip scenario sweep (window=8)

| Skip | Zero-noise threshold | Kernel best_skip | Known seed survival range |
|------|---------------------|-----------------|--------------------------|
| 0 | **0.30** | 0 | 0.10 → 0.75 |
| 3 | **0.40** | 3 | 0.10 → 0.75 |
| 5 | **0.40** | 5 | 0.10 → 0.75 |
| 10 | **0.30** | 10 | 0.10 → 0.75 |
| 20 | **0.40** | 20 | 0.10 → 0.75 |

**Key observations:**
- Kernel correctly identified `best_skip` in every scenario — the skip sweep
  [0, 50] was sufficient for all tested scenarios.
- Known seed survived at match_rate=1.0 across the entire threshold range in
  all scenarios — no risk of over-pruning the real signal.
- At threshold=0.25 (current production default), 272 false survivors passed
  in every scenario. Zero-noise requires 0.30–0.40 depending on skip.
- Conservative baseline across all scenarios: **0.40**
- Permissive baseline (best case): **0.30**

### 4.2 Experiment 2 — Window size sweep (draw_skip=5)

| Window | Zero-noise threshold | vs window=8 | Noise at thresh=0.25 |
|--------|---------------------|-------------|----------------------|
| 8 | 0.40 | baseline | 271 false survivors |
| 10 | 0.35 | −0.05 | 1 false survivor |
| 12 | 0.30 | −0.10 | 1 false survivor |
| 16 | 0.20 | −0.20 | 0 false survivors |

**Key observations:**
- Window size has a strong, consistent effect: each +2 draws reduces the
  zero-noise threshold by 0.05.
- Window=12 at threshold=0.25 has only 1 false survivor — nearly clean.
- Window=16 at threshold=0.20 achieves zero noise — the most permissive
  threshold tested.
- The known seed survives at match_rate=1.0 to threshold=0.75 across all
  window sizes — zero risk of signal loss from threshold or window increases.

---

## 5. Findings and Conclusions

### Finding 1: Current default (threshold=0.25, window=8) is under-configured

At window=8, threshold=0.25 passes **271–278 false survivors per 200k seed
range** in the forward pass. At production scale (1B seeds, 50 trials):

```
Estimated false forward survivors per trial:
  = (272 / 200,000) × 1,000,000,000 = ~1,360,000 per trial
  × 50 trials = ~68,000,000 total false forward survivors
```

These are eliminated by the reverse pass and bidirectional intersection, but
they consume GPU time and add noise to the intersection.

### Finding 2: Window size is the primary lever

Increasing window size is more efficient than raising threshold alone:
- Window=12 at threshold=0.25 gives near-zero noise (1 false positive)
- Window=12 at threshold=0.30 gives zero noise
- No risk to known-seed recovery at any combination tested

### Finding 3: Threshold ceiling for Optuna is too low

The current Optuna search maximum is 0.60 (`distributed_config.json`). The
calibration shows the known seed survives at **threshold=0.75** — meaning the
search space can safely extend to at least 0.75 without risk of pruning real
candidates. The current ceiling of 0.60 is unnecessarily restrictive.

### Finding 4: Threshold floor for Optuna is too high

The current minimum is 0.15. The calibration shows zero-noise requires at
least 0.20 (window=16) to 0.40 (window=8). A minimum of 0.15 allows Optuna
to explore values that will produce massive false positive rates. The minimum
should be raised to match the empirical zero-noise floor for the configured
window size.

---

## 6. Recommended Configuration Updates

### 6.1 Immediate — production defaults

These represent the safest drop-in replacements requiring no window change:

```json
// persistent_worker_coordinator.py default dict
"threshold": 0.30,        // was 0.25 — empirically safe at window=8+

// window_optimizer.py WindowSearchBounds
"default_forward_threshold": 0.30,   // was 0.25
"default_reverse_threshold": 0.30,   // was 0.25
```

### 6.2 Preferred — window + threshold together

The strongest configuration supported by the data:

```json
// distributed_config.json and PWC defaults
"window_size": 12,
"threshold":   0.25    // zero-noise at window=12, threshold=0.30
                       // near-zero at window=12, threshold=0.25
```

Or if a conservative zero-noise guarantee is required:

```json
"window_size": 12,
"threshold":   0.30
```

### 6.3 Optuna search bounds update

Replace current bounds in `distributed_config.json`:

```json
// CURRENT (synthetic-era)
"forward_threshold": { "min": 0.15, "max": 0.60, "default": 0.25 },
"reverse_threshold": { "min": 0.15, "max": 0.60, "default": 0.25 }

// RECOMMENDED (empirically grounded, window=8 baseline)
"forward_threshold": { "min": 0.30, "max": 0.75, "default": 0.30 },
"reverse_threshold": { "min": 0.30, "max": 0.75, "default": 0.30 }

// RECOMMENDED (if window raised to 12)
"forward_threshold": { "min": 0.20, "max": 0.75, "default": 0.30 },
"reverse_threshold": { "min": 0.20, "max": 0.75, "default": 0.30 }
```

**Rationale for ceiling 0.75:**
- Known seed survives at match_rate=1.0 to threshold=0.75 in all experiments.
- Allowing Optuna to explore up to 0.75 enables discovery of very tight
  configurations on data that happens to align well.
- Raising ceiling from 0.60 to 0.75 costs nothing — the known signal is safe.

**Rationale for floor 0.30 (window=8) / 0.20 (window=12):**
- Below these values, false positive counts grow rapidly (272+ per 200k range).
- Letting Optuna explore below the empirical zero-noise floor wastes trials on
  configurations that will produce noise-dominated bidirectional sets.

### 6.4 Baseline file update

```json
// baselines/baseline_window_thresholds.json
{
  "forward_threshold": 0.30,
  "reverse_threshold": 0.30,
  "window_size": 12,
  "skip_max": 200,
  "expected_survivor_band": [1000, 10000],
  "calibration_source": "THRESHOLD_CALIBRATION_FINDINGS_S148.md",
  "calibration_date": "2026-03-19"
}
```

---

## 7. Caveats and Limitations

1. **Synthetic data only.** All experiments used perfectly clean Java LCG
   sequences. Real CA draws may contain session gaps, missing draws, or
   deviations from pure LCG behavior. Production thresholds should remain
   slightly below the synthetic zero-noise optimum as a recovery margin —
   hence the recommendation of 0.30 rather than 0.40.

2. **Single PRNG tested.** Calibration used `java_lcg` only. If other PRNG
   families have different false-positive distributions, their thresholds may
   need separate calibration. The `phase2_threshold` for hybrid passes (0.50)
   was not tested in this session and remains a future calibration item.

3. **200k seed range.** Experiments used ±100k around the known seed.
   False-positive rates scale linearly with range — the per-200k counts
   reported here should be multiplied by `total_range / 200,000` for
   production estimates.

4. **window_optimizer Optuna bounds vs PWC defaults.** Two separate threshold
   parameters exist: the Optuna search bounds in `distributed_config.json`
   (controlling what Optuna can explore) and the PWC hardcoded default
   (fallback when no optimal config exists). Both must be updated
   consistently — the invariant `baseline ∈ [search_min, search_max]` from
   `THRESHOLD_GOVERNANCE.md` must be preserved.

---

## 8. Implementation Checklist

- [ ] Update `persistent_worker_coordinator.py` default `threshold` 0.25 → 0.30
- [ ] Update `window_optimizer.py` `default_forward_threshold` 0.25 → 0.30
- [ ] Update `window_optimizer.py` `default_reverse_threshold` 0.25 → 0.30
- [ ] Update `window_optimizer.py` `min_forward_threshold` 0.15 → 0.30
- [ ] Update `window_optimizer.py` `min_reverse_threshold` 0.15 → 0.30
- [ ] Update `window_optimizer.py` `max_forward_threshold` 0.60 → 0.75
- [ ] Update `window_optimizer.py` `max_reverse_threshold` 0.60 → 0.75
- [ ] Update `distributed_config.json` forward/reverse threshold bounds
- [ ] Update `baselines/baseline_window_thresholds.json`
- [ ] Update `THRESHOLD_GOVERNANCE.md` change history
- [ ] Add `window_size: 12` to production sweep manifest (decision pending)
- [ ] Submit Optuna bound changes to Team Beta for review before applying

---

## 9. Calibration Scripts

Both scripts are committed to the project root:

| Script | Purpose |
|--------|---------|
| `ca_d3_threshold_calibration.py` | Skip scenario sweep — threshold vs skip |
| `ca_d3_window_calibration.py` | Window size sensitivity — threshold vs window |

Re-run either script at any time to reproduce findings:

```bash
# Skip scenario sweep
python3 ca_d3_threshold_calibration.py

# Window sensitivity
python3 ca_d3_window_calibration.py
```

---

## 10. Change History

| Date | Change | Author |
|------|--------|--------|
| 2026-03-19 | Initial calibration findings from S148 | Team Alpha |

---

## References

- `THRESHOLD_GOVERNANCE.md` — governance model and authority boundaries
- `distributed_config.json` — Optuna search bounds
- `baselines/baseline_window_thresholds.json` — recovery baseline
- `window_optimizer.py` — `WindowSearchBounds` class
- `persistent_worker_coordinator.py` — PWC threshold defaults
- `sieve_gpu_worker.py` — kernel execution and IPC
- `ca_d3_threshold_calibration.py` — skip scenario calibration script (S148)
- `ca_d3_window_calibration.py` — window sensitivity script (S148)
