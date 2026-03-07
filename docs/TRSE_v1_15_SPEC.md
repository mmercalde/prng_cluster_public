# TRSE v1.15 — Specification
**Session:** S121  
**Status:** SPEC ONLY — not yet implemented  
**Supersedes:** v1.1 (multi-scale only, did not leverage CA lottery discoveries)

---

## 1. Why v1.15 Exists

v1.1 added multi-scale clustering (W200/W400/W800) and regime confidence.  
That improved *how well* we detect regimes, but didn't use *what we actually learned*
from the CA lottery data about PRNG structure.

The CA lottery analysis produced three validated discoveries that Step 0 currently
ignores entirely:

| Discovery | Session | What Was Found |
|---|---|---|
| Discrete regime duality | S114 | Survivors exist ONLY at W=3 (143,959) and W=8 (43-53). W=31,243,489 → zero. Suggests hard reseed boundaries, not soft drift. |
| Skip structure signature | S112 | Skip 5-56 explained by ADM procedures: pre-test draws (3 RNG calls), operator/auditor login, animation build. Variable consumption, not random. |
| Offset periodicity | S112 | Offset=43 ≈ 3 weeks at 2 draws/day. Suggests maintenance cycle or infrastructure boundary, not random lag. |

v1.1 treats all of this as unknown and relies purely on clustering to find boundaries.  
v1.15 adds three analyses that *test for* the known structure — then exposes the
results as Step 1 search conditioning.

---

## 2. What v1.15 Adds to the Output Schema

Three new fields added to `trse_context.json`. Everything from v1.1 stays unchanged.

### Field 1: `regime_type`

```json
"regime_type": "short_persistence"
```

**What it is:** Classification of the current regime into one of three types
derived from survivor density analysis of the draw sequence.

**Values:**

| Value | Meaning | Step 1 implication |
|---|---|---|
| `"short_persistence"` | Signal consistent with W=3-8 reseed cycle | Narrow window ceiling to ~32 |
| `"long_persistence"` | Signal consistent with W=64+ continuous stream | Keep default bounds |
| `"mixed"` | Both signals present (transitional period) | Use conservative default bounds |
| `"unknown"` | Insufficient signal to classify | Use default bounds |

**How it's computed:**  
Run a fast survivor density probe on a sample of the draw sequence at W=3 and W=8.
Compare density ratio:
- If W=3 density >> W=8 density → `short_persistence` (short reseed dominant)
- If W=8 density >> W=3 density → `long_persistence`  
- If both elevated → `mixed`
- If neither → `unknown`

**Important:** This is a *classification* based on signal shape, not a brute-force
seed search. We use residue mod counts (fast, no GPU needed) as a proxy for
survivor density. Full survivor count requires Step 1's GPU cluster.

**Runtime cost:** ~0.3s (pure numpy, residue histogram comparison)

---

### Field 2: `skip_entropy_profile`

```json
"skip_entropy_profile": {
  "draw_gap_entropy": 0.847,
  "draw_gap_mean": 31.2,
  "draw_gap_std": 14.6,
  "gap_range_min": 5,
  "gap_range_max": 58,
  "consistent_with_known_skip": true,
  "known_skip_range": [5, 56]
}
```

**What it is:** Statistical profile of the *inter-draw gap structure* in the
draw sequence. The skip range 5-56 means the PRNG consumes 5-56 extra calls
between each live draw. This is detectable as a gap entropy signature in the
draw value differences.

**How it's computed:**  
- Compute first differences of the draw sequence (mod 1000)
- Compute entropy of the difference distribution
- Fit the empirical distribution to estimate gap range
- Compare against known validated range [5, 56]

**Why it matters for Step 1:**  
If `consistent_with_known_skip=True`, Step 1 can tighten `skip_min/skip_max`
bounds around [5, 56] instead of searching [0, 500].  
If `False`, something changed operationally — use wider bounds.

**Runtime cost:** ~0.1s (numpy diff + histogram)

---

### Field 3: `dominant_offset_lag`

```json
"dominant_offset_lag": {
  "dominant_lag": 43,
  "lag_strength": 0.73,
  "secondary_lag": 21,
  "secondary_strength": 0.41,
  "confident": true
}
```

**What it is:** The dominant periodic lag in the draw sequence, detected via
FFT on the draw value time series. Validated value from S112: offset=43
(≈ 3 weeks at 2 draws/day, likely maintenance cycle boundary).

**How it's computed:**  
- Apply FFT to the draw sequence (mod 1000)
- Find dominant frequency peaks
- Convert frequency to lag in draw units
- Report top 2 peaks with strength (normalized spectral power)
- `confident=True` if dominant peak strength > 0.5 and lag is in [10, 200] range

**Why it matters for Step 1:**  
If `confident=True`, Step 1 can bias the offset search to center around
`dominant_lag` instead of sampling [0, 100] uniformly.  
Concretely: `max_offset` can be narrowed to `dominant_lag + 20` and TPE
warm-start gets `offset=dominant_lag` as a prior.

**Runtime cost:** ~0.2s (numpy FFT on 18k draws)

---

## 3. Updated `trse_context.json` Schema (v1.15 complete)

```json
{
  "trse_version": "1.15.0",
  "timestamp": "2026-03-07T...",
  "elapsed_seconds": 6.4,
  "n_draws": 18068,
  "k_clusters": 5,

  "current_regime": 0,
  "regime_age": 5,
  "regime_stable": true,
  "regime_confidence": 0.73,
  "regime_type": "short_persistence",

  "silhouette": 0.047,
  "switch_rate": 0.045,
  "regime_counts": [167, 16, 27, 78, 66],

  "scales": {
    "w200": {"regime": 0, "stable": true, "silhouette": 0.031, "switch_rate": 0.052, "n_windows": 358, "regime_age": 6},
    "w400": {"regime": 0, "stable": true, "silhouette": 0.047, "switch_rate": 0.045, "n_windows": 354, "regime_age": 5},
    "w800": {"regime": 0, "stable": true, "silhouette": 0.062, "switch_rate": 0.031, "n_windows": 173, "regime_age": 7}
  },

  "skip_entropy_profile": {
    "draw_gap_entropy": 0.847,
    "draw_gap_mean": 31.2,
    "draw_gap_std": 14.6,
    "gap_range_min": 5,
    "gap_range_max": 58,
    "consistent_with_known_skip": true,
    "known_skip_range": [5, 56]
  },

  "dominant_offset_lag": {
    "dominant_lag": 43,
    "lag_strength": 0.73,
    "secondary_lag": 21,
    "secondary_strength": 0.41,
    "confident": true
  },

  "recommended_window_size": 8,
  "window_coherence_ceiling": null,
  "window_confidence": null,

  "regime_entropy_profile": {"0": {...}, "1": {...}, ...},
  "current_window_features": {
    "entropy_mod8": ...,
    "entropy_mod125": ...,
    "entropy_mod1000": ...,
    "digit_transition_H": [[...]],
    "digit_transition_T": [[...]],
    "digit_transition_O": [[...]]
  }
}
```

`window_coherence_ceiling` and `window_confidence` are TB's forward hooks —
included as `null` now, populated in a future version.

---

## 4. How Step 1 Uses the Three New Fields

Step 1 (`window_optimizer_bayesian.py` `OptunaBayesianSearch.search()`) reads
`trse_context.json` before the Optuna study is created and applies three
independent bias rules. Each rule is independently gated — if the field is
absent or its confidence is low, that rule is skipped and defaults apply.

### Rule A — Window ceiling from `regime_type`

```python
if regime_type == "short_persistence":
    bounds.max_window_size = min(32, bounds.max_window_size)
    print(f"[TRSE] regime_type=short_persistence → window ceiling 32")
elif regime_type in ("unknown", "mixed"):
    pass  # keep defaults
```

### Rule B — Skip bounds from `skip_entropy_profile`

```python
skip_prof = trse_ctx.get("skip_entropy_profile", {})
if skip_prof.get("consistent_with_known_skip"):
    known = skip_prof["known_skip_range"]   # [5, 56]
    margin = 10
    bounds.min_skip_min = max(0, known[0] - margin)        # 0
    bounds.max_skip_min = known[0] + margin                 # 15
    bounds.min_skip_max = known[1] - margin                 # 46
    bounds.max_skip_max = min(500, known[1] + margin)       # 66
    print(f"[TRSE] skip consistent with [5,56] → bounds [0-15] to [46-66]")
```

### Rule C — Offset prior from `dominant_offset_lag`

```python
off = trse_ctx.get("dominant_offset_lag", {})
if off.get("confident"):
    dom_lag = off["dominant_lag"]                # 43
    bounds.max_offset = min(dom_lag + 20, bounds.max_offset)   # 63
    # Also inject as warm-start prior (in addition to existing W8_O43)
    warm_start_offset = dom_lag
    print(f"[TRSE] dominant_lag={dom_lag} → offset ceiling {bounds.max_offset}")
```

### Combined effect on SearchBounds (typical real-data run)

| Parameter | Default bounds | After TRSE v1.15 |
|---|---|---|
| `max_window_size` | 500 | **32** (short_persistence) |
| `min_skip_min` | 0 | 0 |
| `max_skip_min` | 50 | **15** |
| `min_skip_max` | 10 | **46** |
| `max_skip_max` | 500 | **66** |
| `max_offset` | 100 | **63** |

The search space contracts dramatically but remains valid. Optuna is not
*forced* into these bounds — it *explores* them. TPE will still find
W=3 if that regime type is active, because W=3 < 32.

---

## 5. Future Expandability — What's Wired In But Not Yet Used

The schema is designed so future enhancements slot in cleanly:

| Future capability | How it slots in |
|---|---|
| Dual RNG switching detection | Add `rng_switch_detected: bool` alongside `regime_type` |
| Per-session (midday/evening) regime | Add `session_regime: {"midday": {...}, "evening": {...}}` |
| Regime persistence forecast | Populate `window_coherence_ceiling` from regime_age trend |
| Adaptive skip range learning | Update `known_skip_range` from Step 1 winner each run |
| Chapter 13 regime change trigger | `regime_confidence` already consumable by Ch13 thresholds |
| GlobalStateTracker feed | `dominant_offset_lag.dominant_lag` → `global_regime_age` enhancer |
| Battery Tier 1B (Berlekamp-Massey) | Add `linear_complexity_estimate` to skip_entropy_profile |

None of these require schema changes — they add keys to existing objects or
populate existing `null` fields.

---

## 6. What Does NOT Change

- Step 0 remains a standalone, optional, passive script
- Steps 2-6 remain completely unchanged
- `trse_context.json` is still gitignored
- Freshness check logic unchanged
- v1.1 multi-scale clustering unchanged (it's the base layer)
- All v1.1 output fields preserved for backward compatibility
- Step 1 fallback behavior unchanged: if context absent → run with defaults

---

## 7. Implementation Plan

v1.15 adds three new functions to `trse_step0.py`:

```
classify_regime_type(draws, k_clusters)   → str
analyze_skip_entropy(draws)               → dict  
detect_offset_periodicity(draws)          → dict
```

All three are called inside `run_trse_multiscale()` after the existing
multi-scale clustering, and their results merged into the context dict.

**Estimated runtime addition:** ~0.6s total (all pure numpy/scipy-free)  
**Total Step 0 runtime:** ~4-6s on 18k draws (vs ~3-5s for v1.1)

**Files touched:** `trse_step0.py` only. No other file changes for v1.15.  
The Step 1 wiring changes come in the subsequent 5-file integration patch
(unchanged from the plan in `TRSE_INTEGRATION_PLAN_S121.md`).

---

## 8. Validation Criteria Before Commit

1. `classify_regime_type()` on 18k real draws → `"short_persistence"` (consistent with W=8 winner)
2. `analyze_skip_entropy()` on 18k real draws → `gap_range_min≈5, gap_range_max≈56, consistent_with_known_skip=True`
3. `detect_offset_periodicity()` on 18k real draws → `dominant_lag` in range [35, 55] (near 43)
4. All three functions return graceful fallback dicts on empty/synthetic input (no exceptions)
5. JSON roundtrip clean
6. Total runtime < 8s on 18k draws

Items 2 and 3 are the real validation — they'll confirm whether the draw
sequence structure is stable enough to be detected analytically, or whether
the W8_O43 discovery was Optuna finding a needle in a haystack that isn't
detectable from the raw sequence alone.

If items 2-3 fail validation on real data (e.g. dominant_lag comes back as
17 instead of 43), those fields will be set `confident=False` and Step 1
will ignore them — which is the correct fallback behavior.

---

*Team Alpha S121 — spec for v1.15*  
*Approved for implementation pending TB review*
