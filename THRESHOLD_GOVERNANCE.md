# THRESHOLD_GOVERNANCE.md

Threshold governance model for the distributed PRNG sieve.

---

## S148 Change History Entry — 2026-03-19

**Session:** S148  
**Author:** Team Alpha  
**Change:** Empirical threshold calibration — synthetic-era defaults replaced.

### What changed
| Parameter | Old | New | Source |
|-----------|-----|-----|--------|
| PWC default threshold | 0.25 | 0.30 | S148 calibration, window=8 skip sweep |
| window_optimizer default_forward_threshold | 0.25 | 0.30 | same |
| window_optimizer default_reverse_threshold | 0.25 | 0.30 | same |
| window_optimizer min_forward_threshold | 0.15 | 0.30 | empirical zero-noise floor |
| window_optimizer min_reverse_threshold | 0.15 | 0.30 | same |
| window_optimizer max_forward_threshold | 0.60 | 0.75 | known seed survives to 0.75 |
| window_optimizer max_reverse_threshold | 0.60 | 0.75 | same |
| distributed_config forward_threshold.min | 0.15 | 0.30 | same as min bounds |
| distributed_config reverse_threshold.min | 0.15 | 0.30 | same |
| distributed_config forward_threshold.max | 0.60 | 0.75 | same as max bounds |
| distributed_config reverse_threshold.max | 0.60 | 0.75 | same |
| distributed_config *.default | 0.25 | 0.30 | same |

### Reference
See `THRESHOLD_CALIBRATION_FINDINGS_S148.md` for full methodology, raw data,
and rationale. The window=12 + threshold=0.30 preferred configuration is
recorded in `baselines/baseline_window_thresholds.json`.

### Invariant preserved
`baseline ∈ [search_min, search_max]`: 0.30 ∈ [0.30, 0.75] ✓
