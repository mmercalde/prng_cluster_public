# SESSION CHANGELOG — March 19, 2026 (S148)

**Focus:** Empirical threshold calibration — synthetic-era defaults replaced with GPU-verified values  
**Outcome:** 5-file threshold patch deployed and verified. Manifest informational fields updated.
Both remotes synced at `<commit_after_push>`.

---

## Summary

S148 performed the first empirical calibration of sieve thresholds using real GPU kernels
against CA Daily 3-style draw sequences. Prior defaults (threshold=0.25, bounds=[0.15,0.60])
were synthetic-era artifacts with no empirical basis. Two calibration scripts were run on
Zeus RTX 3080 Ti hardware to determine zero-noise thresholds across skip scenarios and window
sizes. Results informed a 5-file patch replacing defaults system-wide.

---

## Calibration Experiments (pre-session, committed as `e051ee2`)

**Reference doc:** `THRESHOLD_CALIBRATION_FINDINGS_S148.md`

### Experiment 1 — Skip scenario sweep (window=8)
Known seed: 3141592. Seed range: ±100k (200,001 total).
Five draw_skip values (0, 3, 5, 10, 20) tested against real `sieve_gpu_worker.py`.

| Skip | Zero-noise threshold | False positives at 0.25 |
|------|---------------------|------------------------|
| 0    | 0.30                | 272                    |
| 3    | 0.40                | 272                    |
| 5    | 0.40                | 272                    |
| 10   | 0.30                | 272                    |
| 20   | 0.40                | 272                    |

Known seed survived at match_rate=1.0 to threshold=0.75 in ALL scenarios.

### Experiment 2 — Window size sweep (draw_skip=5)

| Window | Zero-noise threshold | Noise at thresh=0.25 |
|--------|---------------------|-----------------------|
| 8      | 0.40                | 271 false survivors   |
| 10     | 0.35                | 1 false survivor      |
| 12     | 0.30                | 1 false survivor      |
| 16     | 0.20                | 0 false survivors     |

**Key finding:** Window=12 + threshold=0.30 = near-zero noise. Preferred production config.

### Production impact at old defaults
Estimated false forward survivors with threshold=0.25, window=8 across 1B seeds:
`(272 / 200,000) × 1,000,000,000 = ~1,360,000 per trial × 50 trials = ~68,000,000 total`

---

## Patch Deployed — apply_s148_threshold_calibration.py

Applied via `apply_s148_threshold_calibration.py`. All 5 files verified green (15/15 checks).

### File changes

| File | Change |
|------|--------|
| `persistent_worker_coordinator.py` | `threshold=0.25→0.30` (example call + smoke test) |
| `window_optimizer.py` | `WindowSearchBounds`: min 0.15→0.30, max 0.60→0.75, default 0.25→0.30 (×2) |
| `distributed_config.json` | `search_bounds` fwd/rev: min 0.15→0.30, max 0.60→0.75, default 0.25→0.30 |
| `baselines/baseline_window_thresholds.json` | **NEW** — empirically calibrated recovery baseline |
| `THRESHOLD_GOVERNANCE.md` | **NEW** — governance file created with S148 change history |

### Manifest informational update

`agent_manifests/window_optimizer.json` `parameter_bounds` informational fields updated:
- `forward_threshold.default`: 0.25 → 0.30
- `reverse_threshold.default`: 0.25 → 0.30  
- `_bounds_reference`: `[0.15, 0.60]` → `[0.30, 0.75]` (with S148 annotation)

Applied via `apply_s148_manifest_update.py`.

### Invariant preserved
`baseline ∈ [search_min, search_max]`: 0.30 ∈ [0.30, 0.75] ✓

---

## Architecture Notes

### Why threshold=0.30 not 0.40
0.40 is the zero-noise optimum for window=8 (strictest skip scenarios). However
production data will have session gaps and real-world noise the synthetic calibration
doesn't model. Using 0.30 preserves a recovery margin while eliminating the bulk of
false positives vs the old 0.25 default.

### Optuna ceiling raised to 0.75
Known seed survives at match_rate=1.0 to threshold=0.75 across all scenarios tested.
The old ceiling of 0.60 unnecessarily restricted Optuna from finding tight configurations
when data is clean. Raising to 0.75 costs nothing — signal is preserved.

### Preferred config for next sweep (decision pending)
Window=12 + threshold=0.30 recorded in `baselines/baseline_window_thresholds.json`.
Production manifest still uses window=8. Update manifest before Run 1 relaunch if window
upgrade desired — see Implementation Checklist in calibration doc.

---

## Git Commits (S148)

| Commit | Description |
|--------|-------------|
| `e051ee2` | docs(S148): empirical threshold calibration findings — window/threshold sweep |
| `<next>` | fix(s148): empirical threshold calibration — 0.25→0.30, max 0.60→0.75 |

Both remotes must be synced after commit.

---

## Files Created This Session

| File | Purpose |
|------|---------|
| `apply_s148_threshold_calibration.py` | Main patch script — 5 files |
| `apply_s148_manifest_update.py` | Manifest informational field update |
| `THRESHOLD_GOVERNANCE.md` | NEW — threshold governance + change history |
| `baselines/baseline_window_thresholds.json` | NEW — empirically calibrated baseline |

---

**END OF SESSION S148**

---

## S148 Run-1 Ruling — W12 Promotion

**Ruling (verbatim):**
> S148 Run-1 decision: promote `window_size` from 8 to 12 for production relaunch; retain `threshold=0.30`; preserve Optuna search bounds `[0.30, 0.75]`. Rationale: orders-of-magnitude forward-noise reduction with empirically validated signal retention.

**Patch:** `apply_s148_w12_promotion.py` — 3 files, 10/10 checks green.

| File | Change |
|------|--------|
| `distributed_config.json` | `search_bounds.window_size.default` (new field) = 12 + calibration note |
| `agent_manifests/window_optimizer.json` | `parameter_bounds.window_size.default` 2 → 12 (informational) |
| `baselines/baseline_window_thresholds.json` | `window_size` confirmed 12, `run1_ruling` annotation added |

**Threshold regression confirmed green** (all 6 threshold fields verified after W12 patch).

### Run 1 configuration — final
| Parameter | Value | Source |
|-----------|-------|--------|
| `window_size` | **12** | S148 Run-1 ruling |
| `threshold` (fwd/rev) | **0.30** | S148 empirical calibration |
| Optuna `min` (fwd/rev) | 0.30 | S148 empirical calibration |
| Optuna `max` (fwd/rev) | 0.75 | S148 empirical calibration |
| `max_seeds` | 1,073,741,824 | production manifest |
| `window_trials` | 50 | production manifest |
| `use_persistent_workers` | true | PWC architecture |

**This is the first production sweep under empirical threshold governance.**
