# SESSION_CHANGELOG_20260224_S110.md

## Session 110 — February 24, 2026 (Updated)

### Focus: Phase 0 Baseline + Battery-Inspired Statistical Features Proposal

**Note:** This changelog was created retroactively in S111 and updated in S112
to include the battery-inspired statistical features proposal (v1.0→v1.3 FINAL)
which was developed across multiple S110/S111 chat sessions but originated here.

---

## Starting Point (from Session 109)

- Pipeline Steps 1-6 + Selfplay A1-A5 validated
- Docs patched (Ch1 v3.1, Ch3 v4.2, Guide v2.1)
- 200-trial run best = 17,247 bidirectional survivors
- 58 stray docs moved to docs/
- 884 root files identified for cleanup

---

## Work Performed

### 1. Phase 0 Baseline Run (holdout_hits target)

Ran full pipeline Steps 3→5→6 to establish baseline metrics for the existing
`holdout_hits` ML target before the S111 holdout validation redesign.

**Phase 0 Baseline Results:**
| Metric | Value |
|--------|-------|
| ML target | `holdout_hits` |
| Target distribution | ~8 discrete values (Poisson λ≈1) |
| Target variance | ~1.0e-06 |
| Best R² | 0.000155 |
| Feature alignment | 0/6 scoring dimensions |
| Skip semantics | Uniform 4000 (hardcoded) |

**Diagnosis:** The `holdout_hits` target is fundamentally broken — it counts
integer draw matches (Poisson noise), providing no gradient signal for ML
models to learn from. This motivated the v1.1 holdout validation redesign
proposal (approved by Team Beta, implemented in S111).

### 2. Battery-Inspired Statistical Features Proposal (v1.0→v1.3 FINAL)

Developed a comprehensive proposal for PRNG-agnostic statistical features
inspired by NIST SP 800-22 and Diehard battery tests. The proposal evolved
through 4 revisions with Team Beta review at each stage:

**Evolution:**
- **v1.0** — LCG-centric, SciPy dependencies, variable column counts
- **v1.1** — PRNG-agnostic reframe, still had SciPy in workers
- **v1.2** — SciPy-free (numpy-only for Tier 1), runtime budget added,
  leakage guardrail (`seq` invariant assertion)
- **v1.3 FINAL** — Fixed-width autocorr columns (10 lags always, zero-filled),
  popcount optimization, all TB nits resolved

**Tier 1A Features (23 fixed columns, <0.2ms/survivor):**

| ID | Feature Group | Columns | Description |
|----|---------------|---------|-------------|
| F1 | Spectral (FFT + diff) | 5 | peak_magnitude, secondary_peak, spectral_concentration, diff_peak, diff_concentration |
| F5 | Autocorrelation | 12 | 10 lag values (zero-filled) + decay_rate + significant_lag_count |
| F7 | Cumulative Sum | 3 | max_excursion, mean_excursion, zero_crossings |
| F6 | Bit Frequency (32-bit) | 3 | hamming_mean, hamming_std, popcount_bias |

**Tier 1B Features (6 columns, after 1A validation):**

| ID | Feature Group | Columns |
|----|---------------|---------|
| F3 | Runs Analysis (SciPy-free) | 3 |
| F4 | Linear Complexity (LSB, cap=256) | 3 |

**Tier 2+ (gated behind `enable_expensive_features`):**

| ID | Feature Group | Columns |
|----|---------------|---------|
| F2 | Approximate Entropy (cap=64, Numba) | 2 |

**Key Design Principles (TB requirements):**
- Battery-inspired, NOT verbatim NIST tests (operates on skip-subsampled, mod-reduced sequences)
- Features computed ONLY from `seq`, never from `lottery_history` (leakage guardrail)
- Fixed column count regardless of tuning params (prevents dimension drift)
- No SciPy in workers (numpy only for Tier 1)
- Seq invariant assertions in both CPU and GPU paths
- Runtime budget: Tier 1A < 5ms/survivor on CPU

**Deliverable:** `PROPOSAL_BATTERY_STATISTICAL_FEATURES_v1_3_FINAL.md`

**Status:** TB approved, awaiting implementation after Phase 1 baseline.

### 3. Root Cleanup Assessment

Identified 884 files in project root needing organization.
**Status:** Assessment only — cleanup deferred to future session.

---

## Files Changed

| File | Type | Change |
|------|------|--------|
| PROPOSAL_BATTERY_STATISTICAL_FEATURES_v1_3_FINAL.md | New | Battery features proposal |
| (no code changes) | — | Baseline run + proposal only |

---

## Key Findings

1. **R² = 0.000155** confirms holdout_hits is noise, not signal
2. All 4 model types (neural_net, xgboost, lightgbm, catboost) show near-zero R²
3. The consistent scoring rule violation (training features ≠ holdout evaluation)
   is the root cause — features measure residue coherence, lane agreement, etc.
   but target counts raw integer matches
4. **Battery-inspired features** identified as the primary lever for LCG pattern
   detection after holdout validation redesign

---

## TODOs (from S110)

1. ~~S111: Implement holdout_quality redesign~~ → DONE (S111)
2. S110 root cleanup (884 files) — deferred
3. sklearn warnings Step 5 — deferred
4. Remove CSV writer from coordinator.py (dead weight) — deferred
5. Regression diag gate=True — deferred
6. S103 Part2 — deferred
7. Phase 9B.3 — deferred
8. **Battery features Tier 1A implementation** — after Phase 1 baseline

---

## Git Commits

*(No commits this session — baseline run + proposal only, no code changes)*

---

## Next Session (S111)

Implement v1.1 holdout validation redesign:
- New module: `holdout_quality.py`
- Step 3: Add holdout feature extraction per survivor
- Step 5: Switch target from `holdout_hits` → `holdout_quality`
- Expected: R² improvement from 0.000155 to 0.01-0.10+ (100×+ lift)

---

*Session 110 — Retroactive changelog updated during S112*
*Phase 0 baseline established. Battery features proposal v1.3 FINAL delivered.*
*Holdout validation redesign approved.*
