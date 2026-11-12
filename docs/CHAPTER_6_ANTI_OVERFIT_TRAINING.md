# Chapter 6: Anti-Overfit Training (Step 5)

> **What TFM is.** TFM filters candidate generators against observed draws in two directions,
> keeps a population of survivors on purpose, ranks them, and lets the top candidates vote their
> next value into a prediction pool. A survivor is a scored candidate, not a verdict. The
> generator family is a substrate for mimicry, not a claim about the machine. The target is a
> pool hit rate. TFM does not recover a seed or reconstruct state, and an objection that assumes
> it does is out of scope.

**Correction-pass status (2026-11-11):** **CORRECTED AGAINST CODE at `a15b524`** — not closed.
The v3.1.0 text (December 2025) is retained in full. Where it disagrees with the code, an inline
**"Corrected 2026-11-11"** note or a **SUPERSEDED (2026-11-11)** marker says what the code at
`a15b524` does, with an anchor. **Read §12–§17 (end of file) first:** §12 is what Step 5 does
today, §13 what RANGE-MINER / the 22-array NPZ / the window anchor changed, §14 the known gaps,
§15 open items, §16 the VIR-1…6 declaration, §17 the closure statement and completion sentinel.
**Authority for this pass:** `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md` (REV1).
**Base:** `a15b5247336255801de8cf67d52653701b1e8c80` (`a15b524`) on VM 101. HEAD at the time of this
pass is `741822b`, which differs from `a15b524` only under `docs/`, so the code is the same.
**Audit:** `docs/CHAPTER_6_AUDIT.md` (60 claims: 21 accurate, 8 BROKEN, 2 SWITCHED-OFF-BY-RULING,
21 DESCRIPTION-ONLY, 2 UNVERIFIED, 6 absorbed).
**Numbering:** bare "Step N" in this chapter means the **executable** scheme
(`docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.1): Step 5 = `meta_prediction_optimizer_anti_overfit.py`
(`agents/watcher_agent.py:387-395` `STEP_SCRIPTS[5]`; `agent_manifests/reinforcement.json:4`
`"pipeline_step": 5`). The one exception is the §3.2 diagram, whose "STEP 2: SIEVES" is the
conceptual scheme — the sieve runs inside executable Step 1.

### How this chapter was produced (2026-11-11 pass)

1. Every substantive claim was checked against the code at `a15b524`, by symbol, and labelled once:
   BROKEN, SWITCHED-OFF-BY-RULING or DESCRIPTION-ONLY (`docs/CHAPTER_6_AUDIT.md` §4–§6). The
   governance trail (`docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/TEAM_ALPHA_*`) was searched before
   any item was called new.
2. Host state on VM 101 was read, never changed: the Step-5 sidecar and checkpoint under
   `models/reinforcement/` (key tree; the checkpoint by pickle disassembly, nothing unpickled), four
   WATCHER Step-5 logs, `agent_manifests/definitions.json` (gitignored), the train/holdout files.
   Nothing was trained or run.
3. The chapter was then corrected **in place, section by section**. No section was deleted; stale
   sections carry SUPERSEDED markers and keep their text.
4. Claim scope: what the chapter says against what the code at `a15b524` does. Whether the design is
   right is not judged here.
5. WATCHER anchors are from the committed blob (`git show a15b524:agents/watcher_agent.py`); during
   this pass the working tree carried another session's uncommitted edits to that file.

## PRNG Analysis Pipeline — Complete Operating Guide

**Version:** 3.1.0 (Holdout Integration) · 3.2.0 correction overlay 2026-11-11  
**File:** `meta_prediction_optimizer_anti_overfit.py`  
**Lines:** ~900  
**Purpose:** ML model training with anti-overfitting measures and multi-model comparison

> **Corrected 2026-11-11 (DESCRIPTION-ONLY, D1):** at `a15b524` the file is 3,234 lines and
> describes itself as v3.3 / "v3.4" (`meta_prediction_optimizer_anti_overfit.py:3`, `:33`, `:1655`).
> Step 5 also uses `train_single_trial.py` (NN training subprocess), `holdout_quality.py` (the target,
> computed in Step 3) and `models/wrappers/neural_net_wrapper.py` (`SurvivorQualityNet`).

---

## Table of Contents

1. [Why ML Matters for Survivors](#1-why-ml-matters-for-survivors)
2. [The Training Target Problem](#2-the-training-target-problem)
3. [Correct Training Architecture](#3-correct-training-architecture)
4. [Feature Categories](#4-feature-categories)
5. [Multi-Model Architecture](#5-multi-model-architecture)
6. [AntiOverfitMetaOptimizer Class](#6-antioverfit-metaoptimizer-class)
7. [K-Fold Cross-Validation](#7-k-fold-cross-validation)
8. [Hyperparameter Sampling](#8-hyperparameter-sampling)
9. [Sidecar Metadata](#9-sidecar-metadata)
10. [CLI Interface](#10-cli-interface)
11. [Integration Points](#11-integration-points)
12. [What Step 5 does at a15b524](#12-what-step-5-does-at-a15b524) *(added 2026-11-11)*
13. [RANGE-MINER, the 22-array NPZ and the window anchor — what changed for Step 5](#13-range-miner-the-22-array-npz-and-the-window-anchor--what-changed-for-step-5) *(added 2026-11-11)*
14. [Known gaps at a15b524](#14-known-gaps-at-a15b524) *(added 2026-11-11)*
15. [Open items](#15-open-items) *(added 2026-11-11)*
16. [Verification declaration (VIR-1…6)](#16-verification-declaration-vir-16) *(added 2026-11-11)*
17. [Closure statement](#17-closure-statement) *(added 2026-11-11)*

---

## 1. Why ML Matters for Survivors

> **SUPERSEDED (2026-11-11) — framing; DESCRIPTION-ONLY, D2.** §1.1–§1.3 below present the
> τ = 1 bound (P ≈ 10⁻¹¹⁹¹) as the sieve's operating false-positive rate and read every survivor as
> a seed identity ("true seed", "the sieves already proved that"). Both readings are corrected in
> `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` §1.3 (the 10⁻¹¹⁹¹ table is the τ = 1 limit, which TFM
> deliberately does not run at) and §4.3 ("A survivor is a scored candidate, not a verdict"; loose
> thresholds admit a manifold of near-consistent seeds on purpose). **Replacement reading:** the
> survivor population is a deliberately non-trivial set of candidates; Step 5 learns which
> *feature patterns* go with a high holdout-quality score, so that Step 6 can rank candidates for the
> prediction pool. It decides nothing about which seed is "real". The text is kept as history.

### 1.1 Survivors Are Mathematically Significant

The bidirectional sieves eliminate noise with astronomical precision:

```
P(random seed survives bidirectional) ≈ 10⁻¹¹⁹¹

Expected false positives from 4.3 billion seeds ≈ 0
```

**Every bidirectional survivor is significant** — they exist because they actually match PRNG behavior, not by chance.

### 1.2 But Survivors Aren't All Equal

Real-world complications mean survivors may represent:

| Scenario | Implication |
|----------|-------------|
| True seed | Will continue to predict correctly |
| One of multiple seeds | Lottery uses different seeds for different sessions |
| Partial match | Valid before a reseed event, may fail afterward |

### 1.3 What ML Learns

ML doesn't determine IF a survivor is significant — the sieves already proved that.

**ML learns WHICH survivors will continue to perform** based on:
- Feature patterns that indicate robustness
- Temporal stability across different windows
- Quality of bidirectional agreement
- Skip hypothesis consistency

---

## 2. The Training Target Problem

### 2.1 The Bug (CRITICAL)

The original implementation had a **circular training target**:

```python
# BROKEN: y is computed from features that define y
X = [all 62 features including exact_matches, residue_1000_match_rate]
y = score  # Which IS exact_matches / total × 100
```

**What this means:**

```python
# survivor_scorer.py line 198
base_score = matches.float().mean().item()
features = {
    'score': base_score * 100,
    'exact_matches': matches.sum().item(),
    'residue_1000_match_rate': matches.sum().item() / total,
    ...
}
```

The score IS the exact_matches / total — mathematically equivalent to `residue_1000_match_rate × 100`.

> **Corrected 2026-11-11.** *Accurate as history:* before `holdout_hits` the target was the training
> score (`git show 7c981fb^:meta_prediction_optimizer_anti_overfit.py`, `:185-188`, `quality =
> survivor['score']`). *Accurate:* the equivalence holds at `mod = 1000` because predictions are
> already reduced mod 1000 (`survivor_scorer.py:125`), so `score = 100 × match rate`
> (`survivor_scorer.py:365-369`) equals `100 × residue_1000_match_rate` (`:380`, `:388`).
> *DESCRIPTION-ONLY, D3:* the code is not at "line 198"; it is `survivor_scorer.py:365-372` (single
> seed) and `:703` (batch path `extract_ml_features_batch`). *DESCRIPTION-ONLY, D4:* "62 features"
> — at `a15b524` Step 3 extracts **91** per-survivor features and Step 5 trains **89** (§12.3).

### 2.2 What the Model Learned (Wrong)

```
Feature Importance (BROKEN):
  residue_1000_match_rate:  65.01%
  exact_matches:            34.99%
  all other 60 features:     0.00%  ← COMPLETELY IGNORED
```

The model learned a **tautology**:
```
y ≈ 0.65 × residue_1000_match_rate + 0.35 × exact_matches
```

It predicts the score from the components that **define** the score. Circular. Useless.

> **[UNVERIFIED] (2026-11-11, audit U1):** the 65.01 % / 34.99 % split above is also quoted in
> `docs/PROPOSAL_ML_Architecture_Remediation_v2_0.md:227` (same date, same author), but no run log,
> model file or importance dump behind it was found in the repo or on VM 101. It is kept as recorded
> history, not as a measured fact.

### 2.3 Why This Defeats the Purpose

Per the whitepapers:
> "ML models learn optimal weighting across features to maximize Hit@K metrics"

The current ML **cannot generalize** because:
- It doesn't use structural features (intersection, lane agreement, temporal)
- It predicts **training** performance, not **future** performance
- It's mathematically circular — no learning required

---

## 3. Correct Training Architecture

### 3.1 The Fix

| Aspect | BROKEN (Current) | CORRECT (Fixed) |
|--------|------------------|-----------------|
| **y (target)** | `score` (training match rate) | `holdout_hits` (holdout Hit@K) |
| **Learning** | Tautology | Generalization |

> **Corrected 2026-11-11 — SWITCHED-OFF-BY-RULING, S1.** The "CORRECT (Fixed)" target in this table
> is no longer the target. Since S111 Step 5 trains on **`holdout_quality`**
> (`meta_prediction_optimizer_anti_overfit.py:1479-1483`; loader default `:612`; read from the
> record's top level `:657-658`), a composite in [0, 1] computed in Step 3 (`holdout_quality.py:99-125`,
> written at `full_scoring_worker.py:423`): 0.50 × CRT match (0.40 `residue_1000_match_rate` + 0.20
> `lane_agreement_8` + 0.20 `lane_agreement_125` + 0.20 `lane_consistency`) + 0.30 × coherence
> (0.34 / 0.33 / 0.33 at mod 1000 / 8 / 125) + 0.20 × `temporal_stability_mean`, all measured on the
> holdout draws. `holdout_hits` is still computed and stored (`full_scoring_worker.py:364-367`, `:407`)
> but is excluded from `X` and not used as `y` (`meta_prediction_optimizer_anti_overfit.py:1482`).
> Recorded authority: `docs/S111_IMPLEMENTATION_PLAN_FINAL.md:6` ("v1.1 proposal APPROVED by Team
> Beta"), `:445`; commits `aaca35d`, `1cb90aa`. **What lifting it would enable:** training again on the
> exact holdout hit rate measured by continuing each seed past the training draws (§3.2.1); S111 records
> why it was set aside — near-discrete values and R² 0.000155 (`S111_IMPLEMENTATION_PLAN_FINAL.md:477-480`).
> No recommendation is made.

### 3.2 Correct Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│ STEP 2: SIEVES                                               │
│   Input: Seed space + lottery history                        │
│   Output: Bidirectional survivors (P ≈ 10⁻¹¹⁹¹)              │
├──────────────────────────────────────────────────────────────┤
│ STEP 3: FEATURE EXTRACTION                                   │
│   Input: Survivors + TRAINING history                        │
│   Output: 47 features characterizing WHY each survived       │
│                                                              │
│   ALSO: Compute HOLDOUT performance                          │
│   ┌────────────────────────────────────────────────────────┐ │
│   │ offset = len(train_history)  ← DERIVED, NOT CONFIGURED │ │
│   │ predictions = prng_func(seed, n_holdout, skip=offset)  │ │
│   │ holdout_hits = matches / n_holdout                     │ │
│   └────────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────────┤
│ STEP 5: ML TRAINING                                          │
│   X = structural features (exclude circular ones)            │
│   y = holdout_hits (NOT training score!)                     │
│   Learn: which feature patterns → FUTURE success             │
├──────────────────────────────────────────────────────────────┤
│ STEP 6: PREDICTION                                           │
│   Rank new survivors by predicted holdout quality            │
│   Top-K form prediction pool                                 │
└──────────────────────────────────────────────────────────────┘
```

> **Corrected 2026-11-11.**
> - *DESCRIPTION-ONLY, D21:* "STEP 2: SIEVES" is the conceptual scheme. In the executable scheme the
>   sieve runs inside Step 1 (`window_optimizer.py`; `agents/watcher_agent.py:387-395`), and executable
>   Step 2 is the scorer meta-optimizer. "Input: Seed space + lottery history" — the sieve reads the
>   observed draw history.
> - *DESCRIPTION-ONLY, D4:* "47 features" — Step 3 emits 91 per-survivor features (§12.3).
> - *Accurate:* the `holdout_hits` box — `offset = len(train_history)` derived, predictions
>   `prng_func(seed, n_holdout, skip=offset)`, `holdout_hits = matches / n_holdout`
>   (`full_scoring_worker.py:297-308`, offset passed at `:367`).
> - *SWITCHED-OFF-BY-RULING, S1:* the Step-5 box's `y = holdout_hits` is now `y = holdout_quality`
>   (§3.1 note). The `holdout_quality` features are **not** computed at `offset = len(train_history)`;
>   see the §3.2.1 note.
> - *Corrected (P≈10⁻¹¹⁹¹):* see the §1 SUPERSEDED note. Step 6 ranks survivors by the model's predicted
>   `holdout_quality`, not "holdout quality" in the `holdout_hits` sense.

### 3.2.1 Why Offset Must Be Derived (CRITICAL)

Per Team Beta's guidance:

> "Remove offset as a choice. Keep offset as a law."

```python
# CORRECT (immutable):
offset = len(train_history)

# WRONG (configurable):
offset = args.holdout_offset  # ❌ NEVER DO THIS
```

**Rationale:**
- PRNGs are state machines
- Holdout is contiguous future data
- Validation requires continuing from post-training state
- Any manual offset introduces ambiguity and defeats verification

The offset is recorded in metadata for debugging but NEVER configurable:
```json
"holdout_eval": {
  "train_len": 4000,
  "holdout_len": 1000,
  "offset": 4000,
  "derived_from": "train_history_length"
}
```

> **Corrected 2026-11-11.**
> - *Accurate:* the offset is derived and not configurable for `holdout_hits`: "OFFSET DERIVED - THIS IS
>   THE LAW" (`full_scoring_worker.py:300-302`), `train_history_len=len(train_history)` (`:367`); no
>   `holdout_offset` argument exists in any tracked `*.py`. Team Beta has since named this the
>   **consumer continuation law** and ruled it separate from the Step-1 window anchor
>   (`docs/TB_RULING_WINDOW_ANCHOR_SEQUENCING.md:19`, `:53`).
> - *Absorbed into S1:* the current target does not use this offset. `holdout_quality`'s holdout
>   features come from `extract_ml_features(seed, lottery_history=holdout_history,
>   skip=get_survivor_skip(meta))` (`full_scoring_worker.py:412-418`). `get_survivor_skip` returns
>   `skip_best`, else `skip_min`, else 0 (`holdout_quality.py:47`); no tracked code produces
>   `skip_best`, so it is `skip_min`. The CPU reference treats `skip` as a one-time burn before `n`
>   consecutive outputs (`prng_registry.py:176-177`, `java_lcg_cpu`). The target's holdout features
>   therefore compare outputs from position `skip_min` onward with the holdout draws, not from position
>   `len(train_history)`. S111 recorded this as intended ("Skip semantics | Uniform 4000 | Per-survivor
>   from NPZ", `docs/S111_IMPLEMENTATION_PLAN_FINAL.md:481`). Open item O1 (§15) records the tension
>   with the continuation-law ruling for the owner.
> - *BROKEN, B8:* the offset is not "recorded in metadata". No tracked `*.py` writes `holdout_eval`,
>   `derived_from` or `train_history_length`; Step 3 only logs it (`full_scoring_worker.py:302`). The
>   nearest sidecar field, `data_context.holdout_window`, is computed from `--holdout-draws`, default
>   1000 (`meta_prediction_optimizer_anti_overfit.py:3074`, `:476-479`), which WATCHER never passes
>   (`agent_manifests/reinforcement.json:173-184`); the host sidecar shows `draw_count 1000`. That
>   window also feeds the WATCHER loop-prevention fingerprint (`:486-492`).

### 3.3 Implementation Change

> **SUPERSEDED (2026-11-11) — SWITCHED-OFF-BY-RULING, S1.** Replaced by the S111 target; see the §3.1
> note and §12.2. The live code is `load_survivors_with_features(..., target_field='holdout_quality',
> exclude_features=['score', 'confidence', 'holdout_hits', 'holdout_quality'])`
> (`meta_prediction_optimizer_anti_overfit.py:1479-1483`). The loader's docstring still says "default:
> holdout_hits" (`:620`), as do `agent_manifests/reinforcement.json:17`, `:43`.

```python
# meta_prediction_optimizer_anti_overfit.py

# OLD (broken):
y = [s['features']['score'] for s in survivors]

# NEW (correct):
y = [s['holdout_hits'] for s in survivors]
```

### 3.4 Features to Consider Excluding from X

These features measure training performance and may dominate:

```python
POTENTIALLY_CIRCULAR = {
    'score',                    # The old target
    'confidence',               # Derived from score
    'exact_matches',            # Defines score
    'residue_1000_match_rate',  # Equivalent to score
}
```

**Recommendation:** Test both approaches:
1. Keep all features in X, just change y
2. Exclude circular features from X, change y

Compare which produces better feature distribution.

> **Corrected 2026-11-11 (DESCRIPTION-ONLY, D5).** The code took neither option fully: it excludes
> only `score` and `confidence` from `X` (plus the top-level `holdout_hits` / `holdout_quality`,
> `meta_prediction_optimizer_anti_overfit.py:1482`). `exact_matches` and `residue_1000_match_rate`
> are trained (host sidecar `feature_schema.feature_names`). Under the current target the relevant
> relation is a different one: `residue_1000_match_rate`, both lane agreements, `lane_consistency`,
> the three coherences and `temporal_stability_mean` in `X` are the **training-window** versions of the
> eight quantities the target composes on the **holdout window** (`holdout_quality.py:100-115`;
> committed `docs/CHAPTER_14_AUDIT.md` §3). The S111 check meant to tell persistence from scorer
> self-consistency (`compute_autocorrelation_diagnostics`, `holdout_quality.py:128`) has no pipeline
> caller (Chapter 14 audit Q2); one hand-run output exists on VM 101,
> `diagnostics_outputs/holdout_feature_autocorr.json` (2026-02-25, 49,882 survivors, median r 0.0088;
> `SESSION_CHANGELOG_20260225_S111.md:52`). The `actual_mean` / `actual_std` exclusion is Chapter 14
> audit S10, now an owner decision.

### 3.5 Expected Outcomes After Fix

**Before (BROKEN):**
```
Feature Importance:
  residue_1000_match_rate:  65%
  exact_matches:            35%
  [60 other features]:       0%
```

**After (FIXED):**
```
Feature Importance:
  intersection_weight:      ~15-20%
  lane_agreement_*:         ~10-15%
  skip_entropy:             ~10-15%
  temporal_stability_*:     ~10-15%
  residue_*_coherence:      ~5-10%
  forward_count:            ~5-10%
  [distributed across structural features]
```

### 3.6 Success Criteria

| Metric | BROKEN | TARGET |
|--------|--------|--------|
| Features with >1% importance | 2 | >10 |
| Top feature importance | 65% | <25% |
| Holdout Hit@K | Unknown | Measurable lift vs random |

> **Corrected 2026-11-11 (DESCRIPTION-ONLY, D6; and S2).** §3.5 and §3.6 are expectations, not code:
> no Step-5 code computes importance shares or Hit@K. Three of the six expected leaders cannot carry
> per-survivor signal at `a15b524`: `intersection_weight` and `forward_count` are identical for every
> survivor of a trial (`utils/canonical_records.py:234`, `:243`, copied per seed `:246-257`) and
> `skip_entropy` is a constant 0.0 (`survivor_scorer.py:784-791`) — SWITCHED-OFF-BY-RULING S2, §14.2.
> The importance picture measured after S111 (mod-8 residue ≈ 32 %, prediction residuals ≈ 20 %) is in
> `docs/CHAPTER_14_TRAINING_DIAGNOSTICS.md` "Feature Signal on Real Data" and
> `SESSION_CHANGELOG_20260225_S111.md:60-67`, measured on pre-RANGE-MINER survivors.

### 3.7 Verification Test (Synthetic Data)

Before running on real data, verify the pipeline with known ground truth:

**Test Setup:**
- PRNG: `java_lcg` (from `prng_registry`)
- TRUE_SEED: `12345`
- Training: `synthetic_train_v2.json` (4000 draws, positions 0-3999)
- Holdout: `synthetic_holdout_v2.json` (1000 draws, positions 4000-4999)

**Expected Result:**
```
Seed 12345 → holdout_hits = 1.0 (100%)
All other seeds → holdout_hits ≈ 0.001 (random chance)
```

**What This Proves:**
- ✅ Temporal offset logic works (`offset = len(train_history)`)
- ✅ PRNG registry alignment correct
- ✅ holdout_hits is a valid ML y-label
- ✅ Distributed pipeline works end-to-end

**Verification Command:**
```bash
python3 -c "
import json
from prng_registry import get_cpu_reference

with open('synthetic_holdout_v2.json') as f:
    holdout = json.load(f)

prng = get_cpu_reference('java_lcg')
outputs = prng(12345, 5000)
predicted = [v % 1000 for v in outputs[4000:]]  # offset = 4000

hits = sum(1 for p, a in zip(predicted, holdout) if p == a)
print(f'Seed 12345: {hits}/{len(holdout)} = {hits/len(holdout):.4f}')
# Expected: 1000/1000 = 1.0000
"
```

> **[UNVERIFIED] (2026-11-11, audit U2).** `synthetic_train_v2.json` and `synthetic_holdout_v2.json`
> exist on VM 101 (untracked, 2025-12-31); the check was not run in this read-only pass. It tests the
> `holdout_hits` path (S1, no longer the target). The "true seed" wording describes a **synthetic
> fixture with a known generator**, which is a legitimate parity test; it is not a statement about
> what TFM recovers on observed draws (§1 framing).

---

## 4. Feature Categories

### 4.1 Why Features Matter

Since every survivor passed the ~10⁻¹¹⁹¹ sieve, features characterize **WHY**:

> **Corrected 2026-11-11 — SWITCHED-OFF-BY-RULING S2, and DESCRIPTION-ONLY D7.** The "~10⁻¹¹⁹¹" premise
> is superseded (§1 note). More importantly, at `a15b524` most of the tables below do **not**
> characterise why one survivor survived rather than another:
> - **Trial-level (13):** every column in §4.2, `skip_min`/`skip_max`/`skip_range` in §4.3, plus
>   `forward_count`, `reverse_count`, `bidirectional_count`, `survivor_overlap_ratio` — built once per
>   trial (`utils/canonical_records.py:215-243`) and copied into every record (`:254`).
> - **Constant 0.0 (6):** `skip_mean`, `skip_std`, `skip_entropy` (§4.3), `survivor_velocity` (§4.6),
>   `velocity_acceleration` — no producer, defaulted (`survivor_scorer.py:784-791`) — and `best_offset`
>   (hardcoded, `:707`).
> - **Run-level (17):** the 14 `global_*` (§4.7; `full_scoring_worker.py:403-405`) plus
>   `total_predictions`, `actual_mean`, `actual_std`.
> - **Per-seed (53):** §4.4 lanes, §4.5 residue, §4.6 temporal stability, and the match, residual,
>   S119 digit and Battery Tier 1A columns the chapter does not list — all from Step 3's regeneration of
>   the seed's own sequence against the training draws (`survivor_scorer.py:592-766`).
>
> The per-seed sieve evidence (`forward_match_rate`, `reverse_match_rate`) is not in `X`
> (`survivor_scorer.py:774-779` merge list). Holding it out and keeping the skip-shape columns at zero
> are governed: committed `docs/CHAPTER_14_AUDIT.md` S5 and S6
> (`docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:167-171`, `:199-200`, `:268`, `:276`).
> **Lifting them would enable** the §4 "why it survived" reading: per-seed sieve match quality and skip
> shape would reach the model; it needs a feature-schema version, Step-3 merge-list changes,
> regenerated artifacts and retraining. Name corrections (D7): `global_regime_change` is
> `global_regime_change_detected`; the tables omit `temporal_stability_min/max`, `pred_*`,
> `residual_*`, the four S119 digit columns and the 23 `batt_*` columns. Grouping and counts as in the
> committed Chapter 14 §15.1.

### 4.2 Intersection Features

| Feature | Meaning | Predictive Value |
|---------|---------|------------------|
| `intersection_count` | Seeds in BOTH forward AND reverse | Higher = more robust |
| `intersection_ratio` | Quality of bidirectional overlap | Higher = better |
| `intersection_weight` | Strength of agreement | Higher = more confident |
| `forward_only_count` | Passed forward but not reverse | Lower = better |
| `reverse_only_count` | Passed reverse but not forward | Lower = better |
| `bidirectional_selectivity` | Precision of intersection | Higher = better |

### 4.3 Skip/Gap Features

| Feature | Meaning | Predictive Value |
|---------|---------|------------------|
| `skip_min` | Minimum gap that worked | Lower range = stronger |
| `skip_max` | Maximum gap that worked | Tighter = better |
| `skip_range` | Hypothesis flexibility | Smaller = more confident |
| `skip_entropy` | Distribution of successful gaps | Lower = more consistent |
| `skip_mean`, `skip_std` | Central tendency | Lower std = better |

**Tight skip range = stronger hypothesis** (only one gap pattern works)

### 4.4 Lane Agreement Features (CRT-derived)

| Feature | Meaning | Predictive Value |
|---------|---------|------------------|
| `lane_agreement_8` | Bit-level consistency | Higher = better |
| `lane_agreement_125` | Decimal structure consistency | Higher = better |
| `lane_consistency` | Overall CRT coherence | Higher = better |

### 4.5 Residue Features

| Feature | Meaning | Note |
|---------|---------|------|
| `residue_8_match_rate` | Direct matches at mod 8 | |
| `residue_125_match_rate` | Direct matches at mod 125 | |
| `residue_1000_match_rate` | Direct matches at mod 1000 | ⚠️ CIRCULAR with score |
| `residue_*_coherence` | Distribution similarity | Good predictor |
| `residue_*_kl_divergence` | Information-theoretic distance | Good predictor |

### 4.6 Temporal Features

| Feature | Meaning | Predictive Value |
|---------|---------|------------------|
| `temporal_stability_mean` | Consistency over time windows | Higher = more robust |
| `temporal_stability_std` | Variance in performance | Lower = better |
| `temporal_stability_trend` | Improving or degrading | Positive = better |
| `survivor_velocity` | Population change rate | Stable = better |

### 4.7 Global Features

| Feature | Meaning | Predictive Value |
|---------|---------|------------------|
| `global_residue_*_entropy` | Distribution uniformity | Signals PRNG type |
| `global_regime_change` | Lottery behavior shift | Warning flag |
| `global_marker_*_variance` | Tracking specific numbers | Bias detection |

---

## 5. Multi-Model Architecture

### 5.1 Supported Models

| Model | Library | Strengths |
|-------|---------|-----------|
| **neural_net** | PyTorch | Non-linear patterns, feature interactions |
| **xgboost** | XGBoost | High variance features, sparse data |
| **lightgbm** | LightGBM | Fast training, large datasets |
| **catboost** | CatBoost | Categorical features, robust defaults |
| **random_forest** | scikit-learn | Interpretability, feature importance |

> **Corrected 2026-11-11 (DESCRIPTION-ONLY, D8).** Step 5 supports **four** families:
> `--model-type` choices are `neural_net`, `xgboost`, `lightgbm`, `catboost`
> (`meta_prediction_optimizer_anti_overfit.py:3057-3059`) and `--compare-models` runs those four
> (`:148`), minus any family the S90 skip registry has excluded (`:150-185`). `random_forest` exists
> only in `train_single_trial.py:785` and in the manifest's bounds list
> (`agent_manifests/reinforcement.json:64`); Step 5 cannot select it.

### 5.2 Model Selection

```bash
# Train single model type
python3 meta_prediction_optimizer_anti_overfit.py \
    --model-type xgboost \
    --survivors survivors_with_scores.json

# Compare all model types
python3 meta_prediction_optimizer_anti_overfit.py \
    --compare-models \
    --survivors survivors_with_scores.json
```

### 5.3 Subprocess Isolation

Each model type runs in a **separate subprocess** to prevent GPU context conflicts:

```python
class SubprocessTrialCoordinator:
    """Coordinates model training in isolated subprocesses"""
    
    def train_model(self, model_type: str, config: dict):
        # Each model gets clean GPU context
        result = subprocess.run([
            'python3', 'train_single_trial.py',
            '--model-type', model_type,
            '--config', json.dumps(config)
        ])
        return result
```

This resolves OpenCL/CUDA conflicts when comparing PyTorch neural nets with XGBoost/LightGBM/CatBoost.

> **Corrected 2026-11-11 (DESCRIPTION-ONLY, D9).** The isolation is real; the mechanism is not the one
> shown. At `a15b524`:
> 1. `--compare-models` re-invokes `meta_prediction_optimizer_anti_overfit.py` once **per model type**
>    as a child process with `S88_COMPARE_MODELS_CHILD=1` (`_s88_run_compare_models`,
>    `meta_prediction_optimizer_anti_overfit.py:223-269`); LightGBM is forced to CPU there
>    (`:265-266`, `:1052`).
> 2. Inside a child, **neural_net** trials and the final NN model run in `train_single_trial.py`
>    subprocesses fed by an NPZ split (`:1873-1882`, `:2562-2602`); the parent never imports torch
>    (`:2555`, `:3131-3143`; design invariant `docs/DESIGN_INVARIANT_GPU_ISOLATION.md`).
> 3. **Tree models train inline** in their child (`:2111-2116` trials, `:1832-1836` final).
>
> `SubprocessTrialCoordinator` exists (`subprocess_trial_coordinator.py:104`, method `run_trial`
> `:211`, not `train_model`) but is used only by `run_subprocess_comparison` (`:758`), reachable only
> through `MultiModelTrainer.train_and_compare` with `--compare-models` set inside a child — which the
> child command never sets (`:230-236`). `train_single_trial.py` has no `--config` argument
> (`train_single_trial.py:845-874`).

---

## 6. AntiOverfitMetaOptimizer Class

### 6.1 Initialization

```python
class AntiOverfitMetaOptimizer:
    """ML optimizer with anti-overfitting measures"""
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,           # MUST be holdout_hits, NOT score!
        n_splits: int = 5,
        test_size: float = 0.2,
        study_name: str = 'anti_overfit',
        storage: str = None
    ):
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.test_size = test_size
        
        # Optuna study for hyperparameter optimization
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction='minimize',
            load_if_exists=True
        )
```

> **Corrected 2026-11-11.**
> - *DESCRIPTION-ONLY, D10:* the constructor is `__init__(survivors_file, lottery_history, k_folds=5,
>   test_holdout_pct=0.2, study_name=None, storage=None, model_type='catboost', compare_models=False,
>   output_dir='models/reinforcement', device='cuda:0', prng_type='java_lcg', mod=1000,
>   holdout_draws=1000, parent_run_id=None)` (`meta_prediction_optimizer_anti_overfit.py:1390-1406`).
>   It loads `X`, `y` itself (`_load_data`, `:1474-1507`). No study is created in `__init__`; the Optuna
>   study is created in `_run_optuna_optimization` with **`direction="maximize"`**, `TPESampler(seed=42)`
>   (`:1774-1780`). `lottery_history` is used only for its length (`:1499`).
> - *BROKEN, B1:* the 20 % "TRUE test set" is split off (`_create_splits`, `:1509-1519`; the
>   permutation is unseeded, `:1514`) and **never evaluated** — `X_test`/`y_test` are not read after
>   `:1530`. See the §7.3 note.
> - *SWITCHED-OFF-BY-RULING, S1:* the comment "MUST be holdout_hits" — `y` is `holdout_quality`.

### 6.2 Key Methods

| Method | Purpose |
|--------|---------|
| `optimize(n_trials)` | Run Optuna optimization |
| `_objective(trial)` | Single trial with K-fold CV |
| `_evaluate_fold(...)` | Train and evaluate one fold |
| `final_evaluation()` | Holdout test set evaluation |
| `save_best_model()` | Save model + sidecar metadata |

> **Corrected 2026-11-11 (DESCRIPTION-ONLY, D11; BROKEN, B1).** Methods at `a15b524`: `run(n_trials)`
> (`meta_prediction_optimizer_anti_overfit.py:1534`; degenerate-signal early exit `:1554-1589`, then
> compare or single-model), `_run_single_model` (`:2722`; `n_trials > 1` → `_run_optuna_optimization`
> `:1708`, else single-shot), `_optuna_objective(trial)` (`:2005`), `_sample_hyperparameters` (`:2460`),
> `_run_nn_via_subprocess` (`:2540`), `_compute_final_metrics` (`:2782`), `save_best_model` (`:2799`),
> `_save_existing_checkpoint_sidecar` (`:2857`), `_save_degenerate_sidecar` (`:2952`). There is no
> `optimize`, `_evaluate_fold` or `final_evaluation`; no code evaluates the held-out test set (B1).

---

## 7. K-Fold Cross-Validation

### 7.1 Why K-Fold

K-Fold prevents overfitting by:
- Training on K-1 folds, validating on 1 fold
- Rotating which fold is validation
- Averaging metrics across all folds

### 7.2 Implementation

```python
def _objective(self, trial: optuna.Trial) -> float:
    """Optuna objective with K-fold cross-validation"""
    
    # Sample hyperparameters
    params = self._sample_hyperparameters(trial)
    
    # K-fold cross-validation
    kf = KFold(n_splits=self.n_splits, shuffle=True)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train)):
        X_fold_train = self.X_train[train_idx]
        y_fold_train = self.y_train[train_idx]
        X_fold_val = self.X_train[val_idx]
        y_fold_val = self.y_train[val_idx]
        
        # Train and evaluate
        metrics = self._evaluate_fold(
            X_fold_train, y_fold_train,
            X_fold_val, y_fold_val,
            params
        )
        fold_scores.append(metrics.val_mae)
    
    return np.mean(fold_scores)
```

> **Corrected 2026-11-11.**
> - *Accurate:* K-fold over the train+val set with shuffling — `KFold(n_splits=..., shuffle=True,
>   random_state=42)` (`meta_prediction_optimizer_anti_overfit.py:2034`), folds `:2099-2117`; folds are
>   clamped to `n_samples` (S124, `:2026-2033`).
> - *DESCRIPTION-ONLY, D12:* the objective is the **mean fold R², maximised** (`:2117-2121`, direction
>   `:1776`), not mean `val_mae` minimised.
> - *BROKEN, B2:* the fold count WATCHER asks for does not reach compare-models training. WATCHER passes
>   `--k-folds 20` (`agent_manifests/reinforcement.json:183`; host log `logs/s120_step4_6.log` EXEC CMD)
>   and `--compare-models` (`:176`). The parent hands off to `_s88_run_compare_models` (`:3112-3130`),
>   whose child command carries only `--survivors --lottery-data --model-type --trials`
>   `[--enable-diagnostics]` and the NN flags (`:230-259`), so each child uses the default 5 (`:3047`).
>   Host: `logs/s112_step5_4models.log` 12 of 12 children and `logs/s120_step4_6.log` 3 of 3 log
>   "K-Fold CV: 5 folds"; the single-model run `logs/s121_ynorm_smoke.log` logs 20.
>   `--output-dir`, `--enable-vmap` and `--persistent-workers` are dropped the same way.
> - The final model is not chosen by CV: after the search it is refit on the first 80 % of train+val and
>   validated on the last 20 % (`:1818-1838`); compare mode then picks the family with the highest
>   `training_metrics.r2` from that single split (`:94-139`, `:288-311`).

### 7.3 Overfitting Detection

```python
def is_overfitting(metrics: ValidationMetrics) -> bool:
    """Detect if model is overfitting"""
    return (
        metrics.overfit_ratio > 1.5 or      # Test error 50% higher than train
        metrics.test_mae > metrics.val_mae * 1.3 or  # Test 30% higher than val
        metrics.p_value > 0.05              # Not statistically significant
    )
```

> **Corrected 2026-11-11 — BROKEN, B1.** *Accurate:* `ValidationMetrics.is_overfitting()` has exactly
> these thresholds (`meta_prediction_optimizer_anti_overfit.py:1352-1358`). *Broken:* its inputs are
> constants. `_compute_final_metrics` (`:2782-2797`) sets `train_variance = val_variance =
> test_variance = 0.0`, `test_mae = 0.0`, `overfit_ratio = 1.0`, `variance_consistency =
> temporal_stability = 1.0`, `p_value = 0.01`, `confidence_interval = (0.0, 1.0)`, and fills
> `train_mae` / `val_mae` with the **square root of MSE** (RMSE under an MAE name). So
> `is_overfitting()` is always False and `main` always prints "✅ Model generalizes well to test set!"
> (`:3219-3223`). `composite_score()` (`:1360-1371`) has no caller. The placeholders date from the
> initial commit (`git log -S'p_value=0.01'` → `0101306`); no governance document records them. **A
> repair would have to** score the saved model on `X_test`/`y_test` and compute these quantities from
> real errors, or withdraw the claim and the message.

---

## 8. Hyperparameter Sampling

### 8.1 Neural Network Parameters

```python
def _sample_neural_net_params(self, trial: optuna.Trial) -> dict:
    return {
        'hidden_layers': trial.suggest_int('hidden_layers', 1, 4),
        'hidden_size': trial.suggest_int('hidden_size', 32, 256),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'learning_rate': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'epochs': trial.suggest_int('epochs', 50, 200)
    }
```

> **Corrected 2026-11-11.**
> - *DESCRIPTION-ONLY, D13:* the NN space at `a15b524` (`_sample_hyperparameters`,
>   `meta_prediction_optimizer_anti_overfit.py:2462-2477`) is `n_layers` 2–4; `layer_0` 64–256 and
>   each later layer 32–(previous width); `dropout` 0.2–0.6; `learning_rate` 1e-5–1e-2 (log);
>   `batch_size` ∈ {256, 512, 1024}; `epochs` 50–200; `patience` 10–30. No `weight_decay` is sampled
>   (`train_single_trial.py:558` uses a fixed default 1e-5). Feature standardisation and LeakyReLU(0.01)
>   are always on for the NN (`:244-246`, `:1879-1880`, `:2568-2570`). A CLI `--dropout` overrides the
>   sampled value (`train_single_trial.py:524-533`).
> - *BROKEN, B3:* **the final NN model does not use the architecture or patience Optuna chose.** After
>   the search `self.best_config = dict(study.best_trial.params)` (`:1813`) — the flat study keys
>   `n_layers`, `layer_0…`, `patience` — is passed as `--params` (`:1826-1829`, `:2576`).
>   `train_single_trial.py` reads `hidden_layers` (default `[256, 128, 64]`, `:522`) and
>   `early_stopping_patience` (default 15, `:601`), which that dict does not contain. Trials are
>   unaffected (they receive the list form from `_sample_hyperparameters`). Host: the sidecar records
>   `n_layers 4`, `layer_0..3 = 84, 45, 35, 32`; the checkpoint `models/reinforcement/best_model.pth`
>   carries `hidden_layers [256, 128, 64]` and a first weight tensor of 89 × 256 floats (pickle
>   disassembly; nothing unpickled). Present since S89 (`9676bca`). **A repair would have to** rebuild
>   the training config from the best trial before the final fit and show the checkpoint's
>   `hidden_layers` equals the sidecar's.

### 8.2 XGBoost Parameters

```python
def _sample_xgboost_params(self, trial: optuna.Trial) -> dict:
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
    }
```

> **Corrected 2026-11-11.**
> - *DESCRIPTION-ONLY, D14:* the XGBoost space is `n_estimators` 100–500, `max_depth` 3–12,
>   `learning_rate` 0.01–0.3 (log), `subsample` 0.6–1.0, `colsample_bytree` 0.6–1.0
>   (`meta_prediction_optimizer_anti_overfit.py:2479-2486`); LightGBM samples the same five
>   (`:2488-2495`); CatBoost samples `iterations` 200–800, `depth` 3–10, `learning_rate`
>   (`:2497-2502`). No `min_child_weight`, `reg_alpha` or `reg_lambda`. Tree models in Step 5 train
>   inline through `MultiModelTrainer` (`:998-1140`).
> - *BROKEN, B4:* **CatBoost's sampled `iterations` and `depth` never reach the model.**
>   `_train_catboost` reads `params.get('n_estimators', 200)` and `params.get('max_depth', 8)`
>   (`:1105-1106`), so every CatBoost trial and the final model are 200 iterations at depth 8, with only
>   the learning rate searched; the sidecar then records the sampled values as the model's
>   hyperparameters (`:1813`, `:2843`). `train_single_trial.py:381-382` has the same mismatch. CatBoost is
>   the WATCHER default (`agent_manifests/reinforcement.json:180`). Present since S89 (`9676bca`).
>   **A repair would have to** use one key vocabulary between search space and trainer, and record the
>   parameters the model was built with.

---

## 9. Sidecar Metadata

### 9.1 Purpose

The sidecar file (`best_model.meta.json`) contains everything needed to:
- Load and use the model
- Reproduce the training
- Track provenance

### 9.2 Sidecar Format

```json
{
  "model_type": "xgboost",
  "checkpoint_path": "models/reinforcement/best_model.xgb",
  "feature_schema_hash": "a1b2c3d4...",
  "feature_count": 62,
  "feature_names": ["intersection_weight", "lane_agreement_8", ...],
  "training_metrics": {
    "train_mae": 0.15,
    "val_mae": 0.18,
    "test_mae": 0.19,
    "overfit_ratio": 1.05
  },
  "hyperparameters": {
    "n_estimators": 500,
    "max_depth": 6,
    ...
  },
  "provenance": {
    "training_target": "holdout_hits",
    "survivor_count": 831672,
    "train_history_draws": 18000,
    "holdout_history_draws": 225,
    "cli_args": "--model-type xgboost --trials 50",
    "git_commit": "abc123...",
    "timestamp": "2025-12-30T14:30:00Z"
  },
  "agent_metadata": {
    "run_id": "anti_overfit_20251230_143000",
    "parent_run_id": "full_scoring_20251230_120000",
    "pipeline_step": 5
  }
}
```

> **Corrected 2026-11-11 (DESCRIPTION-ONLY, D15; values in the example are illustrative).** The sidecar
> at `a15b524` is schema `3.4.0`, written by `MultiModelTrainer.save_model`
> (`meta_prediction_optimizer_anti_overfit.py:1286-1324`), `_save_existing_checkpoint_sidecar`
> (`:2898-2938`, the compare-models and NN path) or `_save_degenerate_sidecar` (`:2966-3019`). Top-level
> keys on VM 101's `models/reinforcement/best_model.meta.json` (2026-03-06): `schema_version`,
> `model_type`, `checkpoint_path`, `checkpoint_format`, `feature_schema {source_file, feature_count,
> feature_names, ordering, feature_schema_hash, excluded_features}`, `signal_quality {target_name,
> signal_status, signal_confidence, prediction_allowed, …}`, `data_context {fingerprint_hash,
> training_window, holdout_window, survivor_source, prng_hypothesis, …}`, `training_metrics {r2,
> train_mae, val_mae, test_mae, overfit_ratio, status}`, `hyperparameters`, `optuna`, `hardware`,
> `training_info`, `agent_metadata`, `provenance`. Differences from the example: the feature hash, count
> and names are **nested under `feature_schema`**; the target is recorded as
> `signal_quality.target_name = "holdout_quality"` (S1), not `provenance.training_target`; provenance
> carries no git commit, CLI args or draw counts (`:2873-2883`); `test_mae` and `overfit_ratio` are the
> constants of B1; an XGBoost checkpoint is `best_model.json`, not `.xgb` (`:1251-1256`). Accurate:
> `agent_metadata.pipeline_step = 5` (`:1312`, `:2920`), and a sidecar is written on every outcome.

### 9.3 Feature Schema Hash

The feature schema hash ensures the model receives features in the expected order:

```python
def compute_feature_hash(feature_names: List[str]) -> str:
    """Compute hash of feature schema for validation"""
    schema_str = ','.join(sorted(feature_names))
    return hashlib.sha256(schema_str.encode()).hexdigest()[:16]
```

Step 6 validates this hash before making predictions.

> **Corrected 2026-11-11.** *Accurate:* the hash is SHA-256 over the comma-joined sorted names, first
> 16 hex digits (`compute_feature_schema_hash`, `meta_prediction_optimizer_anti_overfit.py:600-603`;
> the schema reader hashes the already-sorted list the same way, `:586-588`).
> *BROKEN, B5:* **Step 6 does not validate the hash on its production path.** `prediction_generator.py:350-353`
> calls `load_model_from_sidecar(models_dir=..., device=...)` without `survivors_file`, and the hash and
> count checks run only `if survivors_file and strict` (`models/model_factory.py:334`); the count check
> also reads `total_features` / `per_seed_feature_count` (`:344`), which Step 5 does not write.
> `validate_feature_schema_hash` is imported (`prediction_generator.py:68`) and never called. What Step 6
> does instead is order its input columns by the sidecar's `feature_names` (`:366`) and fill any
> missing name with 0.0 (`:798`), so a renamed Step-3 feature becomes a silent zero column rather than a
> fatal error. `agent_manifests/definitions.json:77`, `:91-94` (gitignored, read by no code) still
> declares a fatal hash check. This is Chapter 7's surface; it is recorded here because this chapter makes
> the claim. **A repair would have to** compare the runtime schema with the sidecar before predicting and
> fail on mismatch.

---

## 10. CLI Interface

### 10.1 Basic Usage

```bash
# Train single model
python3 meta_prediction_optimizer_anti_overfit.py \
    --survivors survivors_with_scores.json \
    --lottery-data train_history.json \
    --model-type neural_net \
    --trials 50 \
    --k-folds 5 \
    --output-dir models/reinforcement

# Compare all models
python3 meta_prediction_optimizer_anti_overfit.py \
    --survivors survivors_with_scores.json \
    --lottery-data train_history.json \
    --compare-models \
    --trials 50 \
    --output-dir models/reinforcement
```

### 10.2 CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--survivors` | Yes | - | Path to survivors with features |
| `--lottery-data` | Yes | - | Path to training history |
| `--model-type` | No | neural_net | Model type to train |
| `--compare-models` | No | False | Train all 4 model types |
| `--trials` | No | 50 | Number of Optuna trials |
| `--k-folds` | No | 5 | K-fold cross-validation splits |
| `--output-dir` | No | models/reinforcement | Output directory |
| `--study-name` | No | anti_overfit | Optuna study name |
| `--save-all-models` | No | False | Save all 4 models for analysis |

> **Corrected 2026-11-11** (argparse at `meta_prediction_optimizer_anti_overfit.py:3037-3106`).
> - *Accurate:* `--survivors`, `--lottery-data` required (`:3041-3044`); `--compare-models` trains the four
>   families (`:3060`, `:148`); `--trials` default 50 (`:3045`); `--k-folds` default 5 (`:3047`);
>   `--output-dir` default `models/reinforcement` (`:3064`).
> - *DESCRIPTION-ONLY, D16:* `--model-type` defaults to **`catboost`** (`:3057`); choices are the four
>   families, no `random_forest`. WATCHER also passes `catboost` (`agent_manifests/reinforcement.json:180`),
>   while the manifest's own bounds list says `neural_net` (`:66`).
> - *DESCRIPTION-ONLY, D17:* `--save-all-models` does not exist; compare mode archives every family's
>   artifacts by default under `models/reinforcement/compare_models/<run_id>/<model>/` (`:193`,
>   `:274-278`). The table omits `--test-holdout`, `--storage`, `--device`, `--prng-type`, `--mod`,
>   `--holdout-draws`, `--parent-run-id`, `--enable-diagnostics`, `--normalize-features`,
>   `--use-leaky-relu`, `--dropout`, `--persistent-workers`, `--no-persistent-workers`, `--enable-vmap`,
>   `--allow-inline-nn-fallback` (`:3049-3104`).
> - *BROKEN, B6:* `--study-name` (and `--storage`) have no effect: stored as `self.study_name` /
>   `self.storage` (`:1440-1441`) and never read. Optuna names the study
>   `step5_{model}_{schema_hash[:10]}_{fingerprint[:10]}` (+ `_catb22` for the NN) and picks its own
>   SQLite or journal storage under `optuna_studies/` (`:1737-1767`).
> - *BROKEN, B2:* in compare mode the parent's `--k-folds`, `--output-dir`, `--enable-vmap` and
>   `--persistent-workers` are not forwarded to the per-family children (`:230-259`); see §7.2 note.
>
> **What WATCHER actually runs** (`agent_manifests/reinforcement.json:173-184` →
> `agents/watcher_agent.py:1815`, `:2141-2154` at `a15b524`; host log `logs/s120_step4_6.log`):
>
> ```
> python3 meta_prediction_optimizer_anti_overfit.py --survivors survivors_with_scores.json \
>     --lottery-data train_history.json --compare-models --trials 20 \
>     --output-dir models/reinforcement --enable-diagnostics --model-type catboost \
>     --enable-vmap --k-folds 20
> ```
>
> Only parameters declared in the manifest's `default_params` can be overridden (step-scoped filter,
> `agents/watcher_agent.py:1815` at `a15b524`) — so a WATCHER RETRY does not change Step-5 parameters
> (committed Chapter 14 audit S1).

---

## 11. Integration Points

### 11.1 Pipeline Position

```
Step 3: Full Scoring
    ↓ (survivors_with_scores.json)
Step 4: Adaptive Meta-Optimizer
    ↓ (reinforcement_engine_config.json)
Step 5: ANTI-OVERFIT TRAINING ← You are here
    ↓ (best_model.* + best_model.meta.json)
Step 6: Prediction Generator
```

### 11.2 Input Requirements

| File | Source | Contains |
|------|--------|----------|
| `survivors_with_scores.json` | Step 3 | 62 features per survivor + holdout_hits |
| `train_history.json` | Step 1 | Training lottery draws |
| `holdout_history.json` | Step 1 | Holdout lottery draws |
| `reinforcement_engine_config.json` | Step 4 | Derived training parameters |

> **Corrected 2026-11-11.**
> - *Accurate:* §11.1's order 3 → 4 → 5 → 6 (`agents/watcher_agent.py:387-395`). `train_history.json`
>   and `holdout_history.json` are written by Step 1 (`window_optimizer.py:1100-1105`).
> - *DESCRIPTION-ONLY, D4 / S1:* `survivors_with_scores.json` carries **91** features per survivor
>   under `features`, plus top-level `holdout_hits`, `holdout_quality` and `holdout_features`
>   (`full_scoring_worker.py:389-423`); Step 5 trains on 89 of the features with `y = holdout_quality`.
> - *DESCRIPTION-ONLY, D18:* Step 5 does not read `holdout_history.json`. Holdout information reaches it
>   only through `holdout_quality` (Step 3 reads the file, `full_scoring_worker.py:546`). Step 5 reads
>   `train_history.json` (`--lottery-data`) only for its length (`meta_prediction_optimizer_anti_overfit.py:1499`,
>   `:3157-3160`).
> - *BROKEN, B7:* **Step 5 does not read `reinforcement_engine_config.json`.** The string does not occur
>   in `meta_prediction_optimizer_anti_overfit.py` or `train_single_trial.py`; the manifest lists it as a
>   required input (`agent_manifests/reinforcement.json:9`), which WATCHER uses only to locate inputs for
>   its freshness/preflight checks (`agents/watcher_agent.py:783-800` at `a15b524`), and no flag carries
>   it to the script (`reinforcement.json:173-184`). Step 6 is its reader (`--config
>   reinforcement_engine_config.json`, host log `logs/s120_step4_6.log`). Chapter 5 §2.4 also names
>   Step 5 as the consumer. The file is absent from VM 101's working tree today. **A repair would have
>   to** give Step 5 a route for Step 4's capacity decisions, or record that Step 4 informs Step 6 only.

### 11.3 Output Files

```
models/reinforcement/
├── best_model.pth          # (if neural_net)
├── best_model.xgb          # (if xgboost)
├── best_model.lgb          # (if lightgbm)
├── best_model.cbm          # (if catboost)
└── best_model.meta.json    # Sidecar metadata (ALWAYS)
```

> **Corrected 2026-11-11 (DESCRIPTION-ONLY, D19).** Extensions are `.pth` (NN), **`.json`** (XGBoost),
> **`.txt`** (LightGBM), `.cbm` (CatBoost) (`meta_prediction_optimizer_anti_overfit.py:1251-1256`).
> *Accurate:* the sidecar is written on every outcome, including a degenerate target with no model
> (`:1568`, `:2811-2813`, `:2952-3019`). In compare mode every family's artifacts are archived under
> `models/reinforcement/compare_models/<run_id>/<model>/` and the winner's are copied back
> (`:193`, `:274-278`, `:313-321`), so `models/reinforcement/` holds stale checkpoints of the losing
> families next to the winner's (VM 101: `.cbm .json .pth .txt`); the sidecar's `checkpoint_path` names
> the one in use. A per-run summary goes to `diagnostics_outputs/compare_models_summary_<run_id>.json`
> (`:325`).

### 11.4 Consumed By

- **prediction_generator.py** — Uses model to rank survivors
- **watcher_agent.py** — Monitors for training completion

> **Corrected 2026-11-11.** *Accurate:* Step 6 loads the model through the sidecar
> (`prediction_generator.py:350-353`) after gating on `signal_quality.prediction_allowed` (`:308-347`).
> WATCHER's completion test is `evaluation_type: file_exists` on
> `models/reinforcement/best_model.meta.json` (`agent_manifests/reinforcement.json:166-168`). After
> Step 5, WATCHER also runs the Chapter 14 training health check; that path carries the committed
> Chapter 14 findings B1 (`best_round` vs `best_val_round`), B3 (tree final models get no
> diagnostics), B5 (`f{i}` feature names) and B18 (NN per-feature gradients never captured) —
> cross-referenced, not re-reported here.

---

## Summary: Getting ML Right

> **SUPERSEDED in part (2026-11-11).** "The Problem" restates the τ = 1 framing (§1 note, D2); "The
> Right Approach" names `y = holdout_hits`, replaced by `y = holdout_quality` (S1, §3.1 note); the
> "60/62 features" figure is D4 (91 extracted / 89 trained); the success metrics are expectations (D6)
> and the feature groups listed as learnable are partly trial-level or zero (S2). For what Step 5 does
> today read §12. The text is kept as history.

### The Problem

Bidirectional survivors are mathematically significant (~10⁻¹¹⁹¹ false positive rate), but:
- Multiple seeds may exist (different sessions)
- Reseeding events may invalidate some
- Need to predict which will **continue** to perform

### The Wrong Approach (BROKEN)

```python
y = training_score  # Circular: score = f(features that define score)
```

Model learns tautology, ignores 60/62 features.

### The Right Approach (FIXED)

```python
y = holdout_hits  # Generalization: predict future from structural features
```

Model learns which feature patterns predict future success:
- Intersection quality
- Skip hypothesis consistency
- Temporal stability
- Lane agreement strength

### Success Metrics

| Metric | BROKEN | FIXED |
|--------|--------|-------|
| Features with >1% importance | 2 | >10 |
| Top feature importance | 65% | <25% |
| Generalization | None | Measurable |

---

## 12. What Step 5 does at a15b524

*(Added 2026-11-11. Every anchor was opened in this pass. File abbreviations in this section:
`mpo` = `meta_prediction_optimizer_anti_overfit.py`, `tst` = `train_single_trial.py`. Nothing was run.)*

### 12.1 Entry and modes

1. **WATCHER** builds the command from `agent_manifests/reinforcement.json:173-184` (`compare_models
   true`, `trials 20`, `k_folds 20`, `model_type catboost`, `enable_diagnostics true`, `enable_vmap
   true`) — the exact argv is in the §10 note.
2. **Compare-models parent** (`--compare-models`, no `S88_COMPARE_MODELS_CHILD`): `_s88_run_compare_models`
   (`mpo:3112-3130` → `:141-331`). Drops any family the skip registry marks (`:150-185`), runs one child
   per family (`:226-269`), archives each, picks the winner by the highest `training_metrics.r2` in the
   child's sidecar (`:94-139`, `:288-311`), copies the winner back (`:313-321`), writes the summary
   (`:325`). The parent never loads data or a GPU library.
3. **Child / single-model run:** loads data (`:1474-1507`), splits 80/20 into train+val and test with an
   unseeded permutation (`:1509-1519`), exits early with a degenerate sidecar and exit code 2 if the
   target has no variance (`:1554-1589`, `:3194-3209`), then runs Optuna when `--trials > 1`
   (`:2726-2728`) or a single-shot fit otherwise (`:2730-2780`). In compare mode the child receives only
   `--survivors --lottery-data --model-type --trials`, `--enable-diagnostics` and the NN flags
   (`:230-259`) — B2.

### 12.2 The target

`y = holdout_quality`, read from each record's top level (`mpo:1479-1483`, `:657-658`). Step 3 computes it
per survivor on the holdout draws (`full_scoring_worker.py:408-423`) as the composite in §3.1's note
(`holdout_quality.py:99-125`). Its holdout features are generated with `skip = skip_min` applied as a
one-time burn (`full_scoring_worker.py:412-418`, `holdout_quality.py:47`, `prng_registry.py:176-177`), not by
continuing from `len(train_history)` (§3.2.1 note; open item O1). `holdout_hits` is still computed with the
continuation offset (`full_scoring_worker.py:297-308`, `:364-367`) and stored, but not trained on. The
signal-quality block that gates Step 6 is computed on this `y` (`compute_signal_quality`, `mpo:379-439`;
called `:1489`). Selfplay optimises a different objective (committed Chapter 13 audit B13).

### 12.3 The features

**91 extracted, 89 trained** — re-derived from the batch producer:

| block | n | anchor |
|---|---|---|
| match, statistics, lanes, S119 digits, temporal (incl. `score`, `confidence`) | 27 | `survivor_scorer.py:702-731` |
| residue mods 8 / 125 / 1000 × {match_rate, kl_divergence, coherence} | 9 | `survivor_scorer.py:624-667`, merged `:734` |
| Battery Tier 1A | 23 | `survivor_scorer.py:754-766` |
| Step-3 metadata merge list | 18 | `survivor_scorer.py:774-779` |
| `global_*` | 14 | `full_scoring_worker.py:403-405`; 14 names `models/global_state_tracker.py:366` |
| **extracted** | **91** | |
| minus `score`, `confidence` (`mpo:1482`) | **89** | VM 101 sidecar: `feature_count 89`, 89 names |

Names come from the first record's sorted `features` keys (`mpo:546-597`); each row is built by name with
0.0 for a missing key (`mpo:651`). Of the 89, 13 are trial-level, 17 run-level and 6 constant zero in a
single-generation survivor set (§4 note; Chapter 14 §15.1).

### 12.4 Neural-net specifics (S92 Category B, S121)

- Always standardised (`tst:486-492`; zero-variance column → scale 1.0) and LeakyReLU(0.01) (forced by
  `mpo:244-246`, `:1879-1880`, `:2568-2570`). Architecture `SurvivorQualityNet`: Linear → BatchNorm1d →
  activation → Dropout per layer (`models/wrappers/neural_net_wrapper.py:62-75`).
- **Y-normalisation (S121):** `y_mean`, `y_std` from the training split, guard `y_std < 1e-8 → 1.0`;
  both splits z-scored (`tst:497-512`). Predictions are inverse-transformed before MSE/R², so metrics are
  on the original scale (`tst:687-701`). `y_mean`/`y_std` go into the checkpoint (`tst:717-719`) and
  `NeuralNetWrapper.predict` applies the inverse at inference (`neural_net_wrapper.py:357-359`, loaded
  `:437-441`). Tree models train on raw `y`. VM 101's checkpoint: `y_mean 0.27418`, `y_std 0.00201`.
- Early stopping on validation loss with the best state restored (`tst:621-679`); full-batch when the
  data is under 200 MB (`tst:572-597`).
- The final NN fit ignores the Optuna architecture and patience — B3.

### 12.5 Search, final fit, selection

Optuna TPE (seed 42), **maximise** mean K-fold R² (`mpo:1774-1780`, `:2005-2121`); NN folds in `tst`
subprocesses, parallel across detected GPUs (`:1746-1755`); tree folds inline. The final model is refit on
the first 80 % of train+val and validated on the last 20 % (`:1818-1838`). In compare mode the family with
the highest single-split R² wins. **No step scores the held-out test set** (B1). CatBoost's structural
parameters are not searched (B4).

### 12.6 Outputs

`models/reinforcement/best_model.{pth|json|txt|cbm}` and `best_model.meta.json` (§9 and §11.3 notes).
Step 6 gates on `signal_quality.prediction_allowed`, loads through the sidecar and orders features by the
sidecar's names; it does not compare the schema hash (B5).

### 12.7 Diagnostics

With `--enable-diagnostics` (WATCHER default), only the NN final model captures training diagnostics
(`mpo:2586-2588` → `tst:607-660`); Optuna folds get none by design (TB Trim #1, `mpo:1891-1892`). Tree
final models get none — committed Chapter 14 audit B3. The health check WATCHER runs afterwards is
Chapter 14's subject.

---

## 13. RANGE-MINER, the 22-array NPZ and the window anchor — what changed for Step 5

*(Added 2026-11-11.)* **No Step-5 code path changed.** `meta_prediction_optimizer_anti_overfit.py`,
`train_single_trial.py` and `holdout_quality.py` contain no reference to the miner, the NPZ contract or the
window anchor. What changed is **what Step 5 is given**:

1. **Trial-level columns.** Under the canonical record the 13 NPZ-backed fields are built once per trial
   and copied into every record (`utils/canonical_records.py:215-243`, `:254`). Every §4.2 feature and the
   §3.5 leaders `intersection_weight` and `forward_count` therefore carry no seed-to-seed signal within a
   trial.
2. **Per-seed sieve evidence is not an input.** `forward_match_rate` / `reverse_match_rate` are set per seed
   (`utils/canonical_records.py:248-251`) but are not in Step 3's merge list (`survivor_scorer.py:774-779`).
   Held out by ruling (S2).
3. **Skip shape is zero.** The 22 arrays carry no skip-shape statistic; `skip_mean`, `skip_std`,
   `skip_entropy` default to 0.0 (`survivor_scorer.py:784-791`). Held at zero by ruling (S2).
4. **"Per-survivor" skip in the target is per-trial.** `get_survivor_skip` falls back to `skip_min`
   (`holdout_quality.py:47`), which the canonical record fills from the trial context
   (`utils/canonical_records.py:226`).
5. **`offset` now has three unrelated meanings.** (a) This chapter's §3.2.1 offset is Step 3's
   `holdout_hits` continuation, `len(train_history)` (`full_scoring_worker.py:300-302`) — the "consumer
   continuation law" (`docs/TB_RULING_WINDOW_ANCHOR_SEQUENCING.md:19`, `:53`). (b) Canonical array 4
   `offset` is, after Brief I, the **window anchor and never the generator phase**
   (`utils/canonical_records.py:217-225`; commits `48a8705`, `69ca910`); it is not a model input. (c)
   `best_offset` in `X` is a hardcoded 0.0 (`survivor_scorer.py:707`). Brief I is implemented and
   committed; its final acceptance is pending a completed production-shape run (`G-PROD-SHAPE` ran on
   2026-08-22 and failed, `docs/S172_BRIEF_I_PRODUCTION_SHAPE_CLASSIFICATION_REPORT.md`).
6. **Scale.** Every population number in this chapter (4.3 billion seeds, 831,672 survivors) predates the
   miner. VM 101's current sidecar was trained on 85 survivors (`data_context.survivor_source.survivor_count`).

---

## 14. Known gaps at a15b524

*(Added 2026-11-11.)* Each item carries exactly one label. Evidence: `docs/CHAPTER_6_AUDIT.md` §4–§6. None of
the BROKEN items was found in the governance trail.

### 14.1 BROKEN (8)

| # | Chapter says | Where the code stops | A repair would have to |
|---|---|---|---|
| B1 | A 20 % holdout test set guards against overfitting; `is_overfitting()` on test/val/train metrics (§6.1, §6.2, §7.3, §9.2) | `X_test` never read; `_compute_final_metrics` returns constants (`mpo:2782-2797`), so `is_overfitting()` is always False (`:1352-1358`, `:3219-3223`) | Score the saved model on the test split and compute the overfit quantities from real errors |
| B2 | `--k-folds` sets CV splits (§10.2); manifest asks 20 | Compare-mode child command omits it (`mpo:230-259`); host logs 15/15 children at 5 folds | Forward every Step-5 parameter to each child and show it in the child log |
| B3 | Optuna's best NN config is what gets saved (§6.2, §8.1) | `best_config = dict(study.best_trial.params)` (`mpo:1813`) has no `hidden_layers` / `early_stopping_patience`, which `tst:522`, `:601` read; host checkpoint 256-128-64 vs sidecar 84-45-35-32 | Rebuild the training config from the best trial; prove checkpoint = sidecar |
| B4 | Sampled hyperparameters drive training (§8) | CatBoost samples `iterations`/`depth` (`mpo:2497-2502`); trainer reads `n_estimators`/`max_depth` (`mpo:1105-1106`; `tst:381-382`) | One key vocabulary; sidecar records the parameters actually used |
| B5 | Step 6 validates the schema hash (§9.3) | `prediction_generator.py:350-353` passes no `survivors_file`; check gated at `models/model_factory.py:334` | Compare runtime schema with the sidecar before predicting; fail on mismatch |
| B6 | `--study-name` names the Optuna study (§10.2) | Stored (`mpo:1440-1441`), never read; study named at `:1742` | Honour the flags or remove them |
| B7 | Step 4's `reinforcement_engine_config.json` supplies training parameters (§11.2) | No reference in `mpo`/`tst`; manifest `reinforcement.json:9` used only for input checks | A route for Step 4's decisions into Step 5, or record that Step 6 is the consumer |
| B8 | The holdout offset is recorded in metadata (§3.2.1) | No writer of `holdout_eval`; sidecar holdout window from `--holdout-draws` default 1000 (`mpo:3074`, `:476-479`), never passed by WATCHER | Record train length, holdout length and offset from the files used |

### 14.2 SWITCHED-OFF-BY-RULING (2)

| # | What is held back | Recorded authority | What lifting it would enable |
|---|---|---|---|
| S1 | `y = holdout_hits` (the chapter's fix), with the continuation offset | `docs/S111_IMPLEMENTATION_PLAN_FINAL.md:6`, `:445`, `:481` (Team Beta approved v1.1); commits `aaca35d`, `1cb90aa`. Already recorded: behaviour model §7 D5; Chapter 14 audit Q2 | Step 5 training on the exact holdout hit rate measured by continuing each seed past the training draws |
| S2 | Per-seed sieve match rates in `X`; non-zero skip-shape columns | `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:167-171`, `:199-200`, `:268`, `:276` (committed Chapter 14 audit S5, S6) | The §4 "why it survived" reading: per-seed sieve evidence and skip shape in the model (needs a schema version, Step-3 merge changes, regenerated artifacts, retraining) |

Under the 2026-11-11 authority ruling both are recorded review positions the owner may keep or lift. No
recommendation is made.

### 14.3 DESCRIPTION-ONLY (21)

| # | Chapter says | Code / current fact |
|---|---|---|
| D1 | v3.1.0, ~900 lines | 3,234 lines, self-described v3.3 / v3.4 |
| D2 | P ≈ 10⁻¹¹⁹¹; survivors are seed identities | τ = 1 bound; corrected in Chapter 2 §1.3, §4.3 |
| D3 | Code at `survivor_scorer.py` line 198 | `:365-372`, `:703` |
| D4 | 47 / 62 features | 91 extracted / 89 trained |
| D5 | Potentially circular set excluded or tested | Only `score`, `confidence` excluded |
| D6 | Expected importance distribution, success criteria | Expectations; no Step-5 code measures them |
| D7 | Feature table names | `global_regime_change_detected`; several groups omitted |
| D8 | Five models incl. random_forest | Four |
| D9 | `SubprocessTrialCoordinator.train_model` isolation | Per-family `mpo` children + NN via `tst`; trees inline |
| D10 | Constructor `(X, y, …)`, direction minimize | `(survivors_file, lottery_history, …)`; maximize |
| D11 | `optimize`, `_objective`, `_evaluate_fold`, `final_evaluation` | `run`, `_optuna_objective`, … (§6.2 note) |
| D12 | CV metric `val_mae`, minimised | Mean fold R², maximised |
| D13 | NN search space | §8.1 note |
| D14 | XGBoost search space | §8.2 note |
| D15 | Sidecar format | Schema 3.4.0, nested `feature_schema`, more blocks |
| D16 | `--model-type` default `neural_net` | `catboost` |
| D17 | `--save-all-models`; nine-flag table | Flag absent (archiving is default); 15 flags undocumented |
| D18 | `holdout_history.json` is a Step-5 input | Not read by Step 5 |
| D19 | `.xgb`, `.lgb` checkpoints | `.json`, `.txt` |
| D20 | Footer "END OF CHAPTER 4" | Chapter 6 |
| D21 | "STEP 2: SIEVES" | Conceptual numbering; executable Step 1 |

### 14.4 Recorded elsewhere — cross-referenced, not counted here

- Committed `docs/CHAPTER_14_AUDIT.md`: **B1** (`best_round` vs `best_val_round`), **B3** (tree final models
  get no diagnostics), **B5** (`f{i}` / `feature_{i}` names), **B18** (NN per-feature gradients never
  captured), **S1** (step-scoped param filter: a RETRY does not change Step-5 params), **S10**
  (`actual_mean` / `actual_std` exclusion — an owner decision; see O2).
- Committed `docs/CHAPTER_13_AUDIT.md` **B13**: selfplay's objective is R² against `holdout_hits`, not
  Step 5's `holdout_quality`.

---

## 15. Open items

*(Added 2026-11-11.)*

- **O1 — Holdout-feature generation vs the continuation-law ruling (owner decision; both positions).**
  *Code / S111 position:* the target's holdout features use the trial's `skip_min` as a one-time burn and
  start comparing at that position (§12.2), as S111 approved ("Per-survivor from NPZ",
  `S111_IMPLEMENTATION_PLAN_FINAL.md:481`). *Ruling text:* `TB_RULING_WINDOW_ANCHOR_SEQUENCING.md:53`
  describes Step 3's holdout path as using `offset = train_history_len` "to continue the candidate PRNG
  into unseen history", and rules that any change to that consumer contract needs its own ruling. That
  description holds for `holdout_hits` only. Alpha's observation, not a ruling: the sieve's constant skip
  is a per-draw gap (Chapter 2 §5.2), while the CPU reference here applies `skip` once
  (`prng_registry.py:176-177`), so the holdout features do not reproduce the sieve's rhythm either. Not
  labelled as a chapter claim; recorded for Michael.
- **O2 — Status of S111 guardrails G1–G6.** `SESSION_CHANGELOG_20260225_S111.md:80` records "TB response:
  Approved all 6" and `:82` "Hold off on S112 patch". Committed Chapter 14 audit S10 says no answering ruling
  was found. G4, G5, G6 are absent from `mpo` (study name `:1742`; no importance or rank metric in the
  sidecar). For the owner; Chapter 14 not edited.
- **O3 — U1, U2** (§2.2 importance split; §3.7 synthetic check) remain UNVERIFIED.
- **O4 — `docs/TEAM_BETA_REVIEW_kfolds_S100.docx`** is binary and unread (catalog §7 gap 7); if it rules on
  compare-mode fold forwarding, B2's "no governance record" is limited to the readable surfaces.
- **O5 — Neighbouring chapters.** B5 is Chapter 7's surface; B7 contradicts Chapter 5 §2.4. Not edited here.
- **O6 — `agent_manifests/definitions.json`** (gitignored, read by no tracked code) still shows the
  `holdout_hits` target, `feature_count 62`, a `prng_type`-inclusive fingerprint and a fatal Step-6 hash
  check (`:77`, `:91-94`, `:157`, `:178`, `:185`).

---

## 16. Verification declaration (VIR-1…6)

**Execution proof.** Every `file:line` in the 2026-11-11 notes was opened in this pass on VM 101 at
`a15b524` (WATCHER from `git show a15b524:agents/watcher_agent.py`; every other cited file byte-identical
to `a15b524`). The files opened are listed in `docs/CHAPTER_6_AUDIT.md` §8.

**Clean control.** `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` was read, not edited.

**Fault-injection control.** Not applicable to a documentation pass. No detector, gate or harness was run
that could pass vacuously; none was invented.

**Completion sentinel.** §17.5.

**Unavailable-observer behaviour.** `.127`, the rigs, the unread `.docx`, the S111 v1.1 proposal text and
the absent `reinforcement_engine_config.json` are UNAVAILABLE, not clean. No runtime behaviour was
observed by execution; B2, B3 and B8 are corroborated by host artifacts, the rest by source.

**Audit claim scope.** What this chapter says against what the code at `a15b524` does — not whether the
design is right.

**Searched surfaces:** tracked repo; gitignored files (`git status --ignored --porcelain`;
`agent_manifests/definitions.json`; `models/reinforcement/`; Step-5 logs); git history (`git log -S` for each
BROKEN item's code); host state (read-only listings and file reads). **Governance trail searched:**
`docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/TEAM_ALPHA_*` (140 files), plus S100, S111, S121 records, fact file,
`TODO_CONSOLIDATED.md`, `PROJECT_FILE_CATALOG.md`. **Chapters searched:** 2, 5, 13 audit, 14 and its audit,
the behaviour model.

---

## 17. Closure statement

### 17.1 Verified against

Code base `a15b524` on VM 101 (`192.168.3.177`), working tree, 2026-11-11; HEAD `741822b` (docs-only
difference).

### 17.2 What this pass did

Audited 60 claims (21 accurate, 8 BROKEN, 2 SWITCHED-OFF-BY-RULING, 21 DESCRIPTION-ONLY, 2 UNVERIFIED, 6
absorbed; 21 + 8 + 2 + 21 + 2 + 6 = 60). Added the §1 framing block, status header, inline correction notes
in §1–§11 and the Summary, and §12–§17. Re-derived the target (`holdout_quality`), the S121
y-normalisation and the 91 / 89 feature contract from code.

### 17.3 What this chapter is NOT

Not a repair: no code, manifest or config was changed. Not a ruling: S1 and S2 are recorded, not
recommended for or against. Not a measurement: nothing was trained or run.

### 17.4 Length

The chapter grew from 738 to about 1,430 lines; nothing was removed. Every pre-existing heading is still
present. One pre-existing line was extended rather than replaced: the header's `**Version:** 3.1.0
(Holdout Integration)` now also names the 3.2.0 correction overlay (`git diff a15b524` shows it as the
only `-` line).

### 17.5 Completion sentinel

```
STATUS:  PASS
```

`PASS` for the declared scope: every substantive claim was checked and labelled, the chapter was corrected
in place with anchors, and the counts reconcile. It is not a claim that Step 5 is correct — §14 lists eight
BROKEN paths — and two claims remain UNVERIFIED (O3).

---

## Version History

```
Version 3.2.0 - November 11, 2026 (correction overlay, not a rewrite)
- Audited against code at a15b524 (docs/CHAPTER_6_AUDIT.md): 60 claims
- Added §1 framing block, status header, inline corrections, §12–§17
- Target re-derived: holdout_quality (S111), not holdout_hits (S1)
- Feature contract re-derived: 91 extracted / 89 trained
- Recorded 8 BROKEN paths (B1–B8); nothing deleted

Version 3.0.0 - December 30, 2025
- MAJOR: Documented training target bug
- MAJOR: Documented correct y = holdout_hits
- MAJOR: Added feature importance expectations
- MAJOR: Explained why ML matters for survivors

Version 2.0.0 - December 22, 2025
- NEW: Multi-model support (--model-type, --compare-models)
- NEW: Subprocess isolation for OpenCL/CUDA compatibility
- NEW: Sidecar metadata generation

Version 1.x - November 9, 2025
- Original implementation
- K-fold cross-validation
```

---

**END OF CHAPTER 4**

> **Corrected 2026-11-11 (DESCRIPTION-ONLY, D20):** this is the end of **Chapter 6**.
