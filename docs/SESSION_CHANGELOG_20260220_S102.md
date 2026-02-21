# SESSION CHANGELOG — February 20, 2026 (S102)

**Focus:** Step 2 Zero-Score Investigation — Root Cause Analysis & v3.6 Patch  
**Outcome:** Two bugs identified and partially fixed. Architectural question raised for Team Beta. Step 2 not yet producing valid scores — awaiting Team Beta direction before proceeding.

---

## Summary

Investigated why Step 2 (Scorer Meta-Optimizer) returns `accuracy=-1.0` for all 100 trials across all 26 GPUs. Traced the failure through three layers: objective function (fixed in S101/v3.5), prng_type resolution (fixed this session/v3.6), and seed values (root cause — requires Team Beta architectural ruling).

---

## Work Completed

| Item | Status |
|------|--------|
| Pre-patch baseline committed to both repos (6082188) | ✅ Complete |
| v3.5 patch applied and distributed to all rigs | ✅ Complete |
| v3.5 checksums verified identical across cluster | ✅ Complete |
| Step 2 test run launched (100 trials, 1hr) | ✅ Complete |
| Zero-score root cause investigation | ✅ Complete |
| Confirmed `pass` block predates S102 (commit 4e340de) | ✅ Complete |
| v3.6 patch written and applied to Zeus | ✅ Complete |
| v3.6 distributed to rigs (120, 154) | ✅ Complete |
| v3.6 verified firing (`Pipeline config: prng_type=java_lcg`) | ✅ Complete |
| scorer_trial_results/ cleared | ✅ Complete |
| Git push + session changelog | ✅ This document |

---

## Bugs Found & Fixed

### Bug 1 — Pre-existing NPZ prng_type `pass` block (v3.6 fix)

| Field | Detail |
|-------|--------|
| File | `scorer_trial_worker.py` |
| Present since | commit `4e340de` (pre-S102, NOT introduced by us) |
| Location | `load_data()` NPZ branch, line ~175 |
| Bug | `pass  # Keep defaults, NPZ doesn't store per-survivor prng_type` |
| Reality | `optimal_window_config.json` IS the canonical source per Chapter 1 and scorer_meta.json manifest |
| Fix | Read `prng_type` and `mod` from `optimal_window_config.json` |
| Version | v3.5 → v3.6 |
| Verified | `Pipeline config: prng_type=java_lcg, mod=None (from optimal_window_config.json)` ✅ |

**Note:** `mod=None` because `optimal_window_config.json` does not contain a `mod` field. Code correctly falls back to `mod=1000`. This is acceptable but `mod` should be added to the config in a future session.

---

## Root Cause — Still Unresolved

### Why scores are still all 0.0 after v3.6

The scorer generates PRNG sequences from each survivor seed and checks:
```python
matches = (predictions == hist_expanded)  # literal value comparison
```

**Current test setup (correct per functional mimicry protocol):**
- Input: `synthetic_lottery.json` — generated from known seed(s)
- Step 1 searched seed space: 0 to 99,999 (`seed_count=100,000`)
- Survivors: 37,846 seeds in range 0–37,999
- `train_history.json`: 4,000 draws, first draw=243

**The problem:** Seeds 0–37,999 are valid Java LCG states but produce sequences with effectively zero literal match rate against the synthetic lottery draw history under the scorer's evaluation method. The degenerate guard fires (`std < 1e-12`) → `accuracy = -1.0` for all trials.

**Two interpretations — requires Team Beta ruling:**

**Interpretation A — Wrong seed space:**
The synthetic data was generated from a seed OUTSIDE the 0–99,999 range. Expand seed search space in Step 1 to cover the true seed, then re-run. The scorer architecture is correct, just needs survivors that actually match the data.

**Interpretation B — Wrong scoring approach:**
Survivors already passed the most stringent possible filter (triple-lane residue matching, bidirectional sieve). Re-checking them against draws with literal equality is redundant and wrong. Step 2 should instead optimize ranking of already-validated survivors using intrinsic NPZ properties:
- `forward_matches`, `reverse_matches`
- `bidirectional_selectivity`, `intersection_ratio`
- `survivor_overlap_ratio`, `score`

---

## Team Beta Questions

1. **What seed was used to generate `synthetic_lottery.json`?**  
   If it's in range 0–99,999 → the scorer has a different evaluation problem (Interpretation B).  
   If it's outside that range → expand seed search space in Step 1 (Interpretation A).

2. **Architectural ruling on Step 2 objective:**  
   Should Step 2 score by literal draw matching (current design), or rank survivors by intrinsic NPZ sieve properties (user's proposed redesign)?

3. **`mod` field missing from `optimal_window_config.json`:**  
   Should Step 1 write `mod` to the config so Step 2 can read it cleanly? Currently defaults to 1000.

---

## Current System State

| Item | State |
|------|-------|
| `scorer_trial_worker.py` | v3.6 on Zeus + both rigs |
| `scorer_trial_results/` | Empty — cleared |
| `bidirectional_survivors_binary.npz` | Touched (fresh timestamp) |
| Step 2 | NOT running — awaiting Team Beta direction |
| `optimal_window_config.json` | `prng_type=java_lcg`, `seed_count=100000`, synthetic input |
| `bidirectional_survivors.json` | 37,846 survivors, seeds 0–37,999 |

---

## Files Modified This Session

| File | Type | Purpose |
|------|------|---------|
| `scorer_trial_worker.py` | MODIFIED | v3.5 → v3.6, NPZ prng_type fix |
| `scorer_trial_worker.py.bak_20260220_*` | NEW | Auto-backup from patch |
| `apply_s102_scorer_worker_v3_6.py` | NEW | Patch script (for record) |
| `SESSION_CHANGELOG_20260220_S102.md` | NEW | This document |

---

## Git Commit

```bash
cd ~/distributed_prng_analysis
git add docs/SESSION_CHANGELOG_20260220_S102.md scorer_trial_worker.py apply_s102_scorer_worker_v3_6.py
git commit -m "S102: scorer_trial_worker.py v3.6 — fix NPZ prng_type resolution

BUG FIX (pre-existing since 4e340de):
  load_data() NPZ branch had 'pass' — never read prng_type from config.
  Comment claimed 'NPZ doesn't store per-survivor prng_type' — incorrect.
  optimal_window_config.json is canonical source per Chapter 1 / manifest.

FIX:
  Read prng_type and mod from optimal_window_config.json.
  No hardcoded strings — configurable per project design principles.
  Warns if config missing, then falls back to java_lcg / 1000.

VERIFIED: 'Pipeline config: prng_type=java_lcg' logged on single trial test.

STATUS: Scores still 0.0 — second bug identified (seed space vs scorer
evaluation method). Awaiting Team Beta architectural ruling before proceeding.

See SESSION_CHANGELOG_20260220_S102.md for full analysis."
git push origin main
git push public main
```

---

## Next Session Starting Point

1. Receive Team Beta ruling on Interpretation A vs B
2. If A: Re-run Step 1 with expanded seed space covering the known synthetic seed
3. If B: Redesign Step 2 objective to rank by intrinsic NPZ properties
4. Add `mod` field to `optimal_window_config.json` output from Step 1
5. Re-run Step 2 clean and verify non-zero scores

---

**END OF SESSION S102**
