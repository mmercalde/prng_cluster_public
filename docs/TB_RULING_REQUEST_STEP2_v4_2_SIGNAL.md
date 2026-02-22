# TB Ruling Request — Step 2 v4.2: Quality Signal Replacement
**Session:** S107  
**Date:** 2026-02-22  
**From:** Team Alpha  
**Status:** BLOCKING — smoke test failed, Step 2 cannot optimize

---

## What Happened

v4.1 deployed cleanly (19/19 checks, AST OK). Smoke test ran. Objective varies
across trials (-0.357, -0.354, -1.0) so the landscape is technically non-flat —
but for the wrong reason. `sel_score = 0.0000` on every passing trial.

### Smoke test output (3 trials, sample_size=500):

```
Trial 0: subset_n=14  keep=0.028  sel_mean=1.00993  sel_score=0.0000  objective=-0.357
Trial 1: subset_n=11  keep=0.022  sel_mean=1.00993  sel_score=0.0000  objective=-0.354
Trial 2: subset_n=9   keep=0.018  too_small -> -1.0
```

Objective only varies because `size_penalty` and `coverage` differ.
`sel_score` is dead — contributing zero to every trial.

---

## Root Cause

`bidirectional_selectivity` is not a continuous signal in this NPZ:

```
min    : 1.0099
p25    : 1.0099
median : 1.0099
p75    : 1.0099
p90    : 1.0099
max    : 2.4711
mean   : 1.0222
% at floor (<=1.011): 98.8%
```

98.8% of the 6,739 survivors sit at the minimum value. Only ~81 seeds
have any selectivity above the floor. The residue filter — which picks
subsets of ~10-30 seeds from a 500-seed sample — has essentially zero
probability of catching those 81 rare seeds. Result: sel_mean is always
1.00993, sel_score is always 0.0.

`bidirectional_selectivity` cannot serve as the primary quality signal
for this dataset.

---

## Full NPZ Field Inventory (live stats from Zeus)

| Field | std | min | max | Notes |
|-------|-----|-----|-----|-------|
| `bidirectional_count` | 722.7 | 6 | 6702 | **High variance — survival frequency** |
| `forward_count` | 1488.5 | 299 | 14951 | High variance |
| `reverse_count` | 1542.7 | 121 | 14804 | High variance |
| `intersection_count` | 722.7 | 6 | 6702 | Same as bidirectional_count |
| `intersection_ratio` | 0.028 | 0.014 | 0.291 | Real spread |
| `intersection_weight` | 0.021 | 0.014 | 0.225 | Real spread |
| `survivor_overlap_ratio` | 0.043 | 0.020 | 0.448 | Real spread |
| `forward_matches` | 0.032 | 0.25 | 0.75 | Real spread |
| `reverse_matches` | 0.030 | 0.25 | 0.75 | Real spread |
| `score` | 0.030 | 0.25 | 0.75 | Same as fwd/rev |
| `skip_range` | 2.71 | 87 | 178 | Skip behavior spread |
| `bidirectional_selectivity` | 0.112 | 1.010 | 2.471 | **98.8% at floor — unusable** |
| `prng_type` | 0.0 | 0 | 0 | Constant — skip |
| `window_size` | 0.060 | 4 | 6 | Near-constant — skip |

---

## Proposed Replacement: `bidirectional_count`

**Semantic meaning:** how many times this seed appeared in the bidirectional
intersection across all Optuna trials in Step 1. A seed that survived 6,702
intersections is far more reliable than one that survived only 6.
This is a direct measure of confidence — not an artifact of the scoring scale.

**Why it works as an objective signal:**
- Real continuous variance (std=722, range 6→6702)
- Independent of the residue filter (filter uses seed values mod rm,
  not bidirectional_count)
- Semantically correct: we want the filter to select the most consistently
  surviving seeds, not random survivors

**Proposed percentile-rank formula (same structure as v4.1):**

```python
bc_subset  = bidirectional_count[mask]
bc_mean    = float(bc_subset.mean())
# Global percentile: what fraction of ALL survivors have lower bc than this subset?
bc_score   = float(np.mean(npz_bidirectional_count < bc_mean))  # in [0, 1]
```

Normalization is identical to v4.1 sel_score — just swapping the field.
Rest of the objective formula unchanged.

---

## Alternative Candidate: `intersection_ratio`

If TB prefers a ratio-based signal over a raw count:

```
intersection_ratio = intersection_count / forward_count  (approx)
std=0.028, range 0.014→0.291
```

Less dominated by absolute trial count, more about what fraction of
forward survivors also appeared in reverse. Semantically: how "tight"
the bidirectional agreement is for this seed.

---

## Questions for TB

**Q1 — Primary signal replacement:**
Replace `bidirectional_selectivity` with `bidirectional_count` as the
primary quality signal? Or prefer `intersection_ratio`?

**Q2 — Normalization:**
Percentile-rank (same as v4.1) appropriate for `bidirectional_count`?
Given the large range (6→6702), percentile-rank is more robust than
min-max normalization. Confirm or specify alternative.

**Q3 — Secondary signals:**
Should `intersection_ratio` or `survivor_overlap_ratio` be added as a
secondary bonus term (small weight, like coverage), or keep the formula
structure clean and stick to one primary signal?

**Q4 — `bidirectional_selectivity` — keep or drop:**
Given it's 98.8% at floor, should it be dropped from consideration
entirely, or retained as a very-minor tie-breaker?

---

## Impact if Approved

- Change is surgical: swap `npz_selectivity` → `npz_bidirectional_count`
  in `load_data()` and `run_trial()`
- Global stats recomputed at load time (already the pattern from v4.1)
- No structural changes to objective formula or patcher approach
- Smoke test re-run with new signal — expect `bc_score` to vary across
  trials since bidirectional_count has real continuous spread

---

## What Does NOT Change

- k-of-3 mask logic
- Balance bonus (bal)
- Temporal coverage bonus (tw_weight * coverage)
- Size penalty and keep-rate band
- All degenerate guards
- _log_trial_metrics structure (just rename sel_* → bc_*)

---

*Team Alpha — S107*
*Blocking issue. Awaiting TB ruling before v4.2 patch.*
