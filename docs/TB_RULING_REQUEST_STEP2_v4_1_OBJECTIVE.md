# TB Ruling Request — Step 2 Objective Redesign (v4.1)

**Session:** S107  
**Date:** 2026-02-22  
**From:** Team Alpha  
**Re:** WSI v4.0 tautology — scoring formula is structurally self-correlated

---

## The Problem

Smoke test on Zeus produced WSI = 0.9997 on the first trial:

```
Parameters: rm1=10, rm2=100, rm3=500, max_offset=5, temporal_window_size=50
Parametric scoring: wf=0.016  wr=0.164  w3=0.820  tw=0.250  wi=0.333
WSI = 0.999673  (cov=0.001441  std_s=0.0458  std_q=0.0315  quality_mean=0.0666)
```

**Root cause:** The scoring formula contains `quality` as its dominant term.

```python
quality = fwd * rev                          # what WSI measures against
w3      = rm3 / (rm1 + rm2 + rm3)           # = 500/610 = 0.820

scores  = wf*fwd + wr*rev + w3*(fwd*rev) + ...
        ≈ 0.820 * (fwd*rev) + small_terms
        ≈ 0.820 * quality   + small_terms
```

So `corr(scores, quality) ≈ 1.0` regardless of params, because `scores` is
structurally a scaled copy of `quality`. WSI will be near 1.0 for any param
set where rm3 dominates (which happens whenever rm3 >> rm1 + rm2).

Optuna cannot optimize a landscape that is flat at ~1.0.

This is the "IQR tautological" TODO that was deferred — now confirmed with
live evidence.

---

## What We Need From Step 2

Step 2 (scorer_meta_optimizer) uses Optuna TPE to find params that produce
**good survivors** for downstream ML (Steps 3-6). The question is: what does
"good" mean without draw history?

The NPZ gives us per-survivor quality signals already computed by the sieve:
- `forward_matches`  — how well this seed matched the forward sieve
- `reverse_matches`  — how well this seed matched the reverse sieve
- `intersection_ratio`, `intersection_weight`, `bidirectional_selectivity`

These are fixed — they don't change with params. So any objective that is
purely a function of NPZ fields will produce the same value for every trial.

**The real question:** What do the Step 2 params (`rm1`, `rm2`, `rm3`,
`max_offset`, `temporal_window_size`) actually control?

Looking at the CHAPTER_3 / Step 2 architecture — these params define the
**residue filter** applied to seeds before passing to the ML scorer:
- `residue_mod_1/2/3` — modular arithmetic filters (seed % mod)
- `max_offset` — offset tolerance
- `temporal_window_size` — temporal grouping window

**The params define a FILTER, not a score.** Different params select different
subsets of survivors. The objective should measure the QUALITY of the selected
subset, not score all survivors.

---

## Proposed Fix: Subset Selection Objective

Instead of scoring all survivors, use params to SELECT a subset, then measure
the intrinsic quality of that subset using NPZ fields that are independent of
the selection criterion.

```python
# Step 1: Use params to define a residue filter (select subset)
mask_1 = (seeds % int(rm1)) < int(max_offset)
mask_2 = (seeds % int(rm2)) < int(max_offset)
mask_3 = (seeds % int(rm3)) < int(max_offset)
mask   = mask_1 | mask_2 | mask_3       # union

# Step 2: If subset too small, return penalty
if mask.sum() < 10:
    return -1.0

# Step 3: Measure quality of selected subset using NPZ signals
#         Use fields DIFFERENT from the selection criterion
selected_selectivity = npz_selectivity[mask]    # bidirectional_selectivity
selected_fwd         = npz_forward_matches[mask]
selected_rev         = npz_reverse_matches[mask]

# Quality = mean bidirectional selectivity of selected subset
#           (higher = more discriminating survivors)
subset_quality = float(selected_selectivity.mean())

# Bonus: reward subsets where fwd/rev are balanced (tight intersection)
fwd_rev_balance = 1.0 - abs(selected_fwd.mean() - selected_rev.mean())

# WSI = composite quality score, bounded
wsi = np.clip(subset_quality * fwd_rev_balance, -1.0, 1.0)
```

**Why this works:**
- Different param combinations select DIFFERENT subsets of survivors
- Each subset has its own intrinsic quality (selectivity, balance)
- The objective is not tautologically correlated with the selection criterion
- Optuna can now find params that select the highest-quality survivor subsets
- `temporal_window_size` can control a temporal diversity bonus

**Why `bidirectional_selectivity`:**
From the NPZ stats: min=1.01, max=2.47, mean=1.022
This is NOT flat — there is real variance to optimize against.

---

## Alternative: Pure Diversity Objective

If TB prefers to avoid residue filters entirely, an alternative is to use
params to weight NPZ fields and maximize **survivor diversity** rather than
subset quality:

```python
# Params define weights for a composite quality score
w1 = rm1 / w_sum
w2 = rm2 / w_sum  
w3 = rm3 / w_sum

# Composite using DIFFERENT NPZ fields (not fwd*rev which = quality)
composite = (w1 * npz_selectivity / 2.47        # normalize to [0,1]
           + w2 * npz_intersection_ratio / 0.291
           + w3 * npz_survivor_overlap_ratio)

# WSI = correlation of composite with fwd*rev quality
# Now composite ≠ quality, so correlation is non-trivial
```

---

## TB Questions

1. **Is the subset-selection approach architecturally correct?** Do rm1/rm2/rm3
   function as residue filters in the original Step 2 design, or do they
   serve a different purpose?

2. **Which NPZ fields are appropriate quality signals?** Suggested:
   `bidirectional_selectivity`, `intersection_ratio`, `survivor_overlap_ratio`
   (all have real variance and are independent of fwd*rev).

3. **Should temporal_window_size participate?** If so, how — as a diversity
   metric (spread of trial_number values in subset) or a count threshold?

4. **Preferred approach:** subset-selection or diversity-weighted composite?

---

## What Happens If We Don't Fix This

Optuna runs 100 trials all returning WSI ≈ 0.999. TPE cannot distinguish
between param sets. `optimal_scorer_config.json` will contain effectively
random params. Steps 3-6 proceed with no meaningful guidance from Step 2.

---

*Team Alpha — S107*
*Awaiting TB ruling before implementing v4.1*
