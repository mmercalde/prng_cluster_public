# Chapter 11: Feature Importance & Visualization

> **What TFM is.** TFM filters candidate generators against observed draws in two directions,
> keeps a population of survivors on purpose, ranks them, and lets the top candidates vote their
> next value into a prediction pool. A survivor is a scored candidate, not a verdict. The
> generator family is a substrate for mimicry, not a claim about the machine. The target is a
> pool hit rate. TFM does not recover a seed or reconstruct state, and an objection that assumes
> it does is out of scope.

**Correction-pass status (2026-11-11):** **CORRECTED AGAINST CODE at `a15b524`**, not closed.
The original text below is kept in full. Where it disagrees with the code, an inline **"Corrected
2026-11-11"** note or a **SUPERSEDED (2026-11-11)** marker says what the code at `a15b524` does,
with an anchor. **Read §17–§24 (end of file) first:** §17 is what exists and who calls it, §18 the
feature contract (91 extracted / 89 trained), §19 how to read an importance value, §20 RANGE-MINER /
the 22-array NPZ / the window anchor, §21 the known gaps, §22 open items, §23 the VIR-1…6
declaration, §24 the closure statement and completion sentinel.
**Authority for this pass:** `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md` (REV1).
**Base:** `a15b5247336255801de8cf67d52653701b1e8c80` (`a15b524`) on VM 101. The working tree was at
`b6cb789`, which changed only `agents/watcher_agent.py`, `agents/watcher_dispatch.py` and
`tests/test_watcher_safety_repairs.py` outside `docs/` (`git diff --name-only a15b524 HEAD`). The one
`agents/watcher_agent.py` anchor here is taken from `git show a15b524:agents/watcher_agent.py`; every
other file cited is identical at `a15b524` and HEAD (`git diff --quiet a15b524 -- <file>`). None of the
`b6cb789` repairs (Ch12 B1, B3, B6) touches anything this chapter describes.
**Audit:** `docs/CHAPTER_11_AUDIT.md` (33 claims: 10 accurate, 4 BROKEN, 0 SWITCHED-OFF-BY-RULING,
12 DESCRIPTION-ONLY, 1 UNVERIFIED, 6 absorbed into items already counted in the Chapter 6, 13 and 14 audits).
**Numbering:** this is a **component/analysis chapter**; it documents no executable step. Bare
"Step N" means the **executable** scheme (`docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.1;
`git show a15b524:agents/watcher_agent.py:387-395`, Step 5 = `meta_prediction_optimizer_anti_overfit.py`
at `:393`). Chapter number 11 is not a step number. The file names `feature_importance_step4.json` /
`_step5.json` use the Dec-2025 numbering, in which "Step 4" was `meta_prediction_optimizer.py`; at
`a15b524` executable Step 4 is `adaptive_meta_optimizer.py` (`:392`).
**Canonical version:** the only Chapter 11 file ever committed (`git log --all -- docs/CHAPTER_11_*`).
Its LLM section was updated at `49e8ad4` (2026-01-20) and **reverted by "Doc sync 20260209_1704" `323b6fa`**;
the text below is the pre-update text (§10, §17.5).

### How this chapter was produced (2026-11-11 pass)

1. Every substantive claim was checked against the code at `a15b524`: `feature_importance.py` and
   `feature_importance_ai_interpreter.py` read in full; `feature_visualizer.py`, `feature_drift_tracker.py`,
   `feature_importance_interpreter.py`, the `/features` route of `web_dashboard.py`, `per_survivor_attribution.py`,
   Step 5's loader and CLI, Step 3's feature producers, and the Chapter 13 feature-diagnostics reader read in the
   relevant parts. Results are in `docs/CHAPTER_11_AUDIT.md`.
2. The governance trail was searched before anything was called a finding (`docs/TB_RULING_*`,
   `docs/PROPOSAL_*`, `docs/TEAM_ALPHA_*`; the approved `PROPOSAL_Addendum_H_Feature_Importance_Integration.md`
   read in full). Items already recorded in the Chapter 6, 13 and 14 audits are cross-referenced and not re-counted.
3. Git history was searched for every function, flag and file the chapter names (`git log --all -S`).
4. Host state was read, never changed: the three feature-importance JSON files in the project root, the first
   record's key set of `survivors_with_scores.json`, the keys of `models/reinforcement/best_model.meta.json` and of
   `diagnostics_outputs/`, `agent_manifests/definitions.json` (gitignored), `crontab -l`, `systemctl list-units`,
   `ss -ltn`. Nothing was run: no training, no dashboard, no web server, no LLM query, no pipeline step.
5. The chapter was then corrected **in place, section by section**. No section was deleted. Stale sections carry
   SUPERSEDED markers and keep their text. Every disagreement carries exactly one label (BROKEN,
   SWITCHED-OFF-BY-RULING or DESCRIPTION-ONLY); all are listed in §21.
6. Claim scope: what the chapter says against what the code at `a15b524` does. This pass does not judge whether
   the design is right.

## PRNG Analysis Pipeline — Complete Operating Guide

**Files:** `feature_importance.py`, `feature_visualizer.py`  
**Lines:** ~600 combined  
**Purpose:** ML interpretability, feature analysis, and 13-chart visualization system

> **Corrected 2026-11-11 (D1):** previously "two files, ~600 lines combined". At `a15b524` the subsystem
> is six pieces: `feature_importance.py` (724 lines), `feature_visualizer.py` (1422),
> `feature_drift_tracker.py` (663), `feature_importance_ai_interpreter.py` (332),
> `feature_importance_interpreter.py` (140), and the `/features` route in `web_dashboard.py`
> (`:2033-2088`) — `wc -l`. See §17 for what each does and who calls it. **No pipeline step calls any of
> them (B1, §17.2).**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture Position](#2-architecture-position)
3. [Feature System Recap](#3-feature-system-recap)
4. [Feature Importance Methods](#4-feature-importance-methods)
5. [Permutation Importance](#5-permutation-importance)
6. [Gradient-Based Importance](#6-gradient-based-importance)
7. [SHAP Analysis](#7-shap-analysis)
8. [Feature Visualizer](#8-feature-visualizer)
9. [13 Chart Types](#9-13-chart-types)
10. [AI-Powered Interpretation](#10-ai-powered-interpretation)
11. [Integration with Pipeline](#11-integration-with-pipeline)
12. [CLI Interface](#12-cli-interface)
13. [Output Formats](#13-output-formats)
14. [Troubleshooting](#14-troubleshooting)
15. [Complete Method Reference](#15-complete-method-reference)
16. Chapter Summary
17. What exists at a15b524, and who calls it *(added 2026-11-11)*
18. The feature contract: 91 extracted / 89 trained *(added 2026-11-11)*
19. How to read an importance value *(added 2026-11-11)*
20. RANGE-MINER, the 22-array NPZ and the window anchor *(added 2026-11-11)*
21. Known gaps at a15b524 *(added 2026-11-11)*
22. Open items *(added 2026-11-11)*
23. Verification declaration (VIR-1…6) *(added 2026-11-11)*
24. Closure statement *(added 2026-11-11)*

---

## 1. Overview

### 1.1 Why Feature Importance?

After training ML models (Step 5), we need to understand:

- **Which features matter?** — Identify predictive features
- **Are we overfitting?** — Detect circular/leaky features
- **What drives predictions?** — Explain model decisions
- **How to improve?** — Guide feature engineering

> **Corrected 2026-11-11 (B1, D12):** "after training ML models (Step 5)" describes the Dec-2025 design
> (`docs/PROPOSAL_Addendum_H_Feature_Importance_Integration.md:60-70`). At `a15b524` Step 5 does not call
> this subsystem: the Step 5 writer of `feature_importance_step5.json` was removed at `e0b55a1` and the
> import at `249052b`; `meta_prediction_optimizer_anti_overfit.py` has no reference to `feature_importance`
> (§17.2). Also: every question above is about the **mimicry model** — which inputs the fitted ranking model
> uses — not about the generator or the machine (§19).

### 1.2 The Two Files

| File | Purpose | Charts |
|------|---------|--------|
| `feature_importance.py` | Compute importance scores | N/A |
| `feature_visualizer.py` | Generate 13 chart types | 13 |

### 1.3 Key Features

| Feature | Description |
|---------|-------------|
| **3 Importance Methods** | Permutation, Gradient, SHAP |
| **4 Model Types** | Neural Net, XGBoost, LightGBM, CatBoost |
| **13 Chart Types** | Bar, radar, heatmap, waterfall, etc. |
| **AI Interpretation** | LLM-powered analysis via Qwen2.5-Math |
| **Export Formats** | PNG, HTML (interactive), JSON, CSV |

> **Corrected 2026-11-11 — row by row, against `a15b524`:**
> - **Methods (B2, D3).** The module implements **native, permutation, gradient** — not SHAP.
>   `_resolve_method` accepts `auto|native|permutation|gradient` and raises on anything else
>   (`feature_importance.py:261-295`). No tracked `*.py` imports `shap`. `auto` picks native for tree
>   models and permutation for PyTorch (`:272-279`).
> - **Model types (D3, U1).** Detection is by interface, not by family: `pytorch`, `xgboost`,
>   `sklearn_tree` (anything with `feature_importances_`), `sklearn_other` (`_detect_model_type`,
>   `:220-246`). Whether the objects Step 5 actually saves pass through it was not executed (audit U1).
>   Importance captured *during* training is a separate route and is broken for three of four families
>   (Chapter 14 audit B4, B18; keys `f{i}`, B5).
> - **13 charts (accurate count; D8).** Thirteen exist (`feature_visualizer.py:1283-1297`), but they are a
>   different set from §9 — see §17.3.
> - **AI interpretation (D2).** DeepSeek-R1-14B on `localhost:8080`, then the Claude CLI, then a template
>   (`feature_importance_ai_interpreter.py:3-13`, `:27`, `:172-205`) — not Qwen2.5-Math.
> - **Export (D4).** One HTML report, `feature_report.html` in the working directory
>   (`feature_visualizer.py:1272`, `:1414-1416`), plus the dashboard route. No PNG, no CSV. JSON only via
>   `FeatureImportanceResult.save` (`feature_importance.py:78-81`).

---

## 2. Architecture Position

### 2.1 In the Pipeline

```
Step 5: Anti-Overfit Training
    │
    ├── best_model.pth (or .xgb, .lgb, .cbm)
    ├── best_model.meta.json
    │
    └── Feature Importance Analysis ◄── THIS CHAPTER
            │
            ├── feature_importance.py
            │       ├── Permutation importance
            │       ├── Gradient importance (neural nets)
            │       └── SHAP values
            │
            └── feature_visualizer.py
                    └── 13 chart types
                            │
                            └── visualization_outputs/
                                    ├── importance_bar.png
                                    ├── correlation_matrix.png
                                    ├── shap_summary.html
                                    └── ... (10 more)
```

> **Corrected 2026-11-11 (B1, D4, D5; X2):** the arrow from Step 5 into "Feature Importance Analysis" does
> not exist at `a15b524` — no pipeline step calls `feature_importance.py` (§17.2). The visualizer reads
> `feature_importance_step4.json` / `_step5.json` / `feature_drift_step4_to_step5.json`; nothing in the
> tracked tree or its history writes them now, and the host copies are dated 2025-12-10 with 24 names, 4 of
> which exist in the live feature vector. Checkpoint extensions are `.pth`, `.json` (XGBoost), `.txt`
> (LightGBM), `.cbm` (`models/model_factory.py:29-32`), not `.xgb`/`.lgb`. There is no
> `visualization_outputs/` directory in any tracked code; the output is one `feature_report.html`. The
> sidecar `best_model.meta.json` carries **no** `feature_importance` key (host, keys read) — that is S111
> guardrail G5, TB-approved and owner-held (Chapter 14 audit Addendum 2; §21.2).

### 2.2 Why Post-Training?

```
Training gives us:
    - Model weights
    - Validation metrics
    - Best hyperparameters

Feature importance tells us:
    - WHY the model works
    - Which features to keep/remove
    - Whether training target is correct
```

---

## 3. Feature System Recap

### 3.1 62 Total Features

> **SUPERSEDED (2026-11-11) — replaced by §18.** The count at `a15b524` is **91 extracted / 89 trained**,
> re-derived from code (D6): 27 match/statistics keys (`survivor_scorer.py:702-731`) + 9 residue
> (`:74`, `:625-667`) + 23 battery (`:756`) + 18 NPZ-merged or defaulted fields (`:774-791`) + 14
> `global_*` (`full_scoring_worker.py:403-405`); Step 5 drops `score` and `confidence`
> (`meta_prediction_optimizer_anti_overfit.py:1479-1483`). The global names carry a `global_` prefix, and
> several names below (`marker_567_variance`, `regime_stability`, `entropy`) do not exist. The "62" is the
> Dec-2025 count (commit `e0b55a1`, "64 → 62 total"). The tree below is kept as history.

```
Combined Features (62 total):
│
├── Per-Seed Statistical (48)
│   ├── Residue Features (9)
│   │   ├── residue_8_match_rate
│   │   ├── residue_8_coherence
│   │   ├── residue_8_kl_divergence
│   │   ├── residue_125_match_rate
│   │   ├── residue_125_coherence
│   │   ├── residue_125_kl_divergence
│   │   ├── residue_1000_match_rate
│   │   ├── residue_1000_coherence
│   │   └── residue_1000_kl_divergence
│   │
│   ├── Temporal Features (5)
│   │   ├── temporal_stability_mean
│   │   ├── temporal_stability_std
│   │   ├── temporal_stability_min
│   │   ├── temporal_stability_max
│   │   └── temporal_stability_trend
│   │
│   ├── Lane Features (3)
│   │   ├── lane_agreement_8
│   │   ├── lane_agreement_125
│   │   └── lane_consistency
│   │
│   ├── Skip Features (6)
│   │   ├── skip_entropy
│   │   ├── skip_mean
│   │   ├── skip_std
│   │   ├── skip_range
│   │   ├── skip_min
│   │   └── skip_max
│   │
│   ├── Intersection Features (6)
│   │   ├── intersection_count
│   │   ├── intersection_ratio
│   │   ├── intersection_weight
│   │   ├── forward_only_count
│   │   ├── reverse_only_count
│   │   └── bidirectional_selectivity
│   │
│   └── Other Statistical (19)
│       ├── actual_mean, actual_std
│       ├── consecutive_matches
│       ├── exact_matches
│       └── ... (15 more)
│
└── Global State Features (14)
    ├── regime_change_detected
    ├── regime_stability
    ├── marker_390_variance
    ├── marker_567_variance
    ├── reseed_probability
    ├── entropy
    └── ... (8 more)
```

### 3.2 Feature Categories for Analysis

| Category | Count | Predictive Value |
|----------|-------|------------------|
| **Intersection** | 6 | HIGH — bidirectional robustness |
| **Skip** | 6 | HIGH — hypothesis consistency |
| **Lane** | 3 | HIGH — CRT coherence |
| **Temporal** | 5 | MEDIUM — time stability |
| **Residue** | 9 | MEDIUM — (beware circular features) |
| **Global** | 14 | LOW-MEDIUM — regime detection |
| **Statistical** | 19 | VARIES |

> **Corrected 2026-11-11 (B4; X6):** the code has **two** categories, not seven:
> `statistical_features` (46 names) and `global_state_features` (14 names), plus `other_features` for the
> rest (`feature_importance.py:95-119`, `_compute_category_importance` `:601-624`,
> `get_importance_summary_for_agent` `:679-705`). The global list is unprefixed (`regime_age`) while the
> live names are `global_regime_age` etc., so for any real model **`global_state_features` sums to 0 and 45
> of the 89 trained names fall into `other_features`** (BROKEN B4). The "Predictive Value" column is not a
> measured property: 13 of the intersection/skip columns are **trial-level** values copied onto every seed
> (Chapter 14 audit Q5; Chapter 6 audit S2), and `skip_entropy`, `skip_mean`, `skip_std` are 0.0
> placeholders (`survivor_scorer.py:783-791`). See §19.

### 3.3 Circular Features (Exclude from X)

```python
CIRCULAR_FEATURES = {
    'score',                    # The training target!
    'confidence',               # Derived from score
    'exact_matches',            # Defines score
    'residue_1000_match_rate',  # Equivalent to score
}
```

**Why?** These features measure the training target itself, leading to:
- 100% importance on 1-2 features
- 0% importance on 60 other features
- No generalization

> **Corrected 2026-11-11 (X4, X5 — counted in the Chapter 6 audit):** no `CIRCULAR_FEATURES` constant
> exists in tracked code. Step 5 excludes `score`, `confidence`, `holdout_hits`, `holdout_quality`
> (`meta_prediction_optimizer_anti_overfit.py:1482`); only the first two are inside `features`, so
> `exact_matches` and `residue_1000_match_rate` **are trained** (Chapter 6 audit D5). The premise has also
> moved: the target is `holdout_quality`, not `score` (`:1481`; Chapter 6 audit S1), so these columns are
> training-window statistics, not the target itself.

---

## 4. Feature Importance Methods

### 4.1 Three Methods

| Method | Works With | Strengths | Weaknesses |
|--------|------------|-----------|------------|
| **Permutation** | All models | Model-agnostic, reliable | Slow for large datasets |
| **Gradient** | Neural nets | Fast, captures interactions | Only for differentiable models |
| **SHAP** | All models | Explains individual predictions | Very slow, memory-intensive |

> **Corrected 2026-11-11 (B2, D3):** the methods at `a15b524` are **native** (tree models: XGBoost
> booster gain, `feature_importance.py:331-359`, or `feature_importances_`, `:302-328`), **permutation**
> (any model, `:362-462`) and **gradient** (PyTorch only, `:465-499`). The SHAP row describes nothing in
> this module (BROKEN B2). Permutation and gradient rows are accurate.

### 4.2 When to Use Each

```
Quick check (minutes):
    → Permutation importance

Deep analysis (hours):
    → SHAP values

Neural net debugging:
    → Gradient importance

Production monitoring:
    → Permutation (fast, reliable)
```

---

## 5. Permutation Importance

### 5.1 How It Works

```
For each feature f:
    1. Record baseline score S₀ on validation set
    2. Shuffle feature f (break its relationship with target)
    3. Record new score S₁
    4. Importance(f) = S₀ - S₁

If shuffling feature f hurts performance:
    → f is important
    
If shuffling has no effect:
    → f is not used by model
```

### 5.2 Implementation

```python
def compute_permutation_importance(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 10,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Compute permutation importance for any model.
    
    Args:
        model: Trained model with .predict() method
        X_val: Validation features [n_samples, n_features]
        y_val: Validation targets [n_samples]
        feature_names: List of feature names
        n_repeats: Number of shuffles per feature
        random_state: Random seed for reproducibility
    
    Returns:
        Dict mapping feature_name → importance_score
    """
    from sklearn.inspection import permutation_importance
    
    result = permutation_importance(
        model, X_val, y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring='neg_mean_absolute_error'
    )
    
    importance = {}
    for i, name in enumerate(feature_names):
        importance[name] = result.importances_mean[i]
    
    # Normalize to sum to 1
    total = sum(abs(v) for v in importance.values())
    if total > 0:
        importance = {k: abs(v)/total for k, v in importance.items()}
    
    return importance
```

> **Corrected 2026-11-11 (D7):** no `compute_permutation_importance()` exists. The public entry point is
> `get_feature_importance(model, X, y, feature_names, method='auto', n_repeats=10, device='cuda:0',
> normalize=True)` (`feature_importance.py:126-213`). For non-PyTorch models it calls sklearn
> `permutation_importance` with **`scoring='neg_mean_squared_error'`** (not MAE), `random_state=42`
> (`:387-392`). For PyTorch it runs its own GPU loop: baseline MSE, then per feature `n_repeats` shuffles by
> unseeded `torch.randperm`, importance = mean MSE increase (`:405-462`). Normalisation to a sum of 1 over
> absolute values matches (`:396-400`, `:456-460`).

### 5.3 Example Output

```
Permutation Importance (CORRECT training):
──────────────────────────────────────────
intersection_weight:       0.182  ████████████████████
skip_entropy:              0.156  █████████████████
lane_agreement_8:          0.134  ███████████████
temporal_stability_mean:   0.098  ███████████
bidirectional_selectivity: 0.087  ██████████
lane_consistency:          0.076  █████████
skip_range:                0.065  ███████
residue_8_coherence:       0.054  ██████
forward_only_count:        0.048  █████
... (distributed across 50+ features)
```

> **Corrected 2026-11-11 (X6 — counted in the Chapter 14 audit, Q5 and D11):** this example is
> illustrative, not a recorded run. It cannot occur at `a15b524`: `skip_entropy` is a constant 0.0
> placeholder (`survivor_scorer.py:783-791`), and permutation importance of a constant column is 0 by
> construction. `intersection_weight`, `bidirectional_selectivity`, `skip_range` and `forward_only_count`
> are trial-level values identical for every seed of a trial (Chapter 14 audit Q5), so their importance is
> a statement about trials, not seeds (§19). The one recorded importance in the repo (S111, LightGBM split
> counts) is led by `residue_8_coherence`, `pred_std`, `residue_125_kl_divergence`
> (`docs/S111_TEAM_BETA_BRIEFING.md:38-55`).

### 5.4 Red Flag: Circular Features

```
Permutation Importance (BROKEN training):
──────────────────────────────────────────
residue_1000_match_rate:   0.652  █████████████████████████████████████████
exact_matches:             0.348  ██████████████████████
[60 other features]:       0.000  

⚠️ WARNING: Only 2 features have importance!
   This indicates circular features in training.
   'residue_1000_match_rate' ≈ 'score' (the target)
```

---

## 6. Gradient-Based Importance

### 6.1 How It Works

```
For neural networks:
    1. Forward pass: compute predictions
    2. Backward pass: compute ∂prediction/∂input
    3. Importance(f) = mean(|gradient_f|) across samples

Large gradient → small change in feature causes big change in prediction
→ feature is important
```

### 6.2 Implementation

```python
def compute_gradient_importance(
    model: torch.nn.Module,
    X_val: torch.Tensor,
    feature_names: List[str],
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Compute gradient-based importance for PyTorch neural nets.
    
    Args:
        model: Trained PyTorch model
        X_val: Validation features as tensor
        feature_names: List of feature names
        device: 'cuda' or 'cpu'
    
    Returns:
        Dict mapping feature_name → importance_score
    """
    model.eval()
    X_val = X_val.to(device)
    X_val.requires_grad_(True)
    
    # Forward pass
    predictions = model(X_val)
    
    # Backward pass
    predictions.sum().backward()
    
    # Gradient magnitude per feature
    gradients = X_val.grad.abs().mean(dim=0)  # [n_features]
    
    importance = {}
    for i, name in enumerate(feature_names):
        importance[name] = gradients[i].item()
    
    # Normalize
    total = sum(importance.values())
    if total > 0:
        importance = {k: v/total for k, v in importance.items()}
    
    return importance
```

> **Verified 2026-11-11 (accurate in substance; name D7):** the algorithm matches
> `_extract_gradient_importance` (`feature_importance.py:465-499`): forward pass, `output.sum().backward()`,
> mean absolute input gradient per feature, normalised. It is private; it is reached through
> `get_feature_importance(..., method='gradient')`, which refuses non-PyTorch models (`:199-202`,
> `:286-289`). No pipeline step calls it (B1); the separate NN-gradient capture in training diagnostics
> is never populated (Chapter 14 audit B18).

### 6.3 Integrated Gradients (Advanced)

> **SUPERSEDED (2026-11-11) — not implemented (BROKEN B3).** No tracked `*.py` contains
> `integrated_gradients`; `git log --all -S compute_integrated_gradients` finds only the commit that added
> this chapter (`53a3829`). The code below was never in the code base. Kept as design text.

```python
def compute_integrated_gradients(
    model: torch.nn.Module,
    X_val: torch.Tensor,
    baseline: torch.Tensor = None,
    steps: int = 50
) -> torch.Tensor:
    """
    Integrated gradients for more accurate attribution.
    
    Integrates gradients along path from baseline to input.
    """
    if baseline is None:
        baseline = torch.zeros_like(X_val)
    
    # Interpolate between baseline and input
    alphas = torch.linspace(0, 1, steps).to(X_val.device)
    
    gradients = []
    for alpha in alphas:
        interpolated = baseline + alpha * (X_val - baseline)
        interpolated.requires_grad_(True)
        
        output = model(interpolated)
        output.sum().backward()
        
        gradients.append(interpolated.grad.clone())
        interpolated.grad.zero_()
    
    # Average gradients and multiply by (input - baseline)
    avg_gradients = torch.stack(gradients).mean(dim=0)
    integrated_grads = (X_val - baseline) * avg_gradients
    
    return integrated_grads
```

---

## 7. SHAP Analysis

> **SUPERSEDED (2026-11-11) — global SHAP is not implemented (BROKEN B2).** `compute_shap_values()` and
> `generate_shap_plots()` were never in the code base (`git log --all -S compute_shap_values` → `53a3829`,
> this chapter, only); no tracked `*.py` imports `shap`. **What does exist is per-survivor attribution**, a
> separate module: CatBoost `ShapValues` (`per_survivor_attribution.py:231-272`, call `:259`), XGBoost
> `pred_contribs` (`:134-177`), LightGBM `pred_contrib` (`:184-224`), NN input gradients (`:49-127`). It
> cannot execute today (Chapter 13 audit B15, four blockers plus a gate that never opens) and a real
> attribution run is ruled blocked (Chapter 14 audit S2); attribution-driven autonomy is not authorized
> (Chapter 14 audit S4). §7.1's "local explanations" bullet belongs to that module (audit X3). The text
> below is kept as design history.

### 7.1 What is SHAP?

**SHAP** (SHapley Additive exPlanations) provides:
- **Local explanations** — Why did THIS prediction happen?
- **Global explanations** — Which features matter overall?
- **Interaction effects** — How do features combine?

### 7.2 Implementation

```python
def compute_shap_values(
    model: Any,
    X_val: np.ndarray,
    feature_names: List[str],
    model_type: str = 'tree',
    max_samples: int = 1000
) -> Dict[str, Any]:
    """
    Compute SHAP values for model interpretability.
    
    Args:
        model: Trained model
        X_val: Validation features
        feature_names: List of feature names
        model_type: 'tree' for XGBoost/LightGBM, 'kernel' for neural nets
        max_samples: Max samples for computation (SHAP is slow!)
    
    Returns:
        Dict with shap_values, feature_importance, and plots
    """
    import shap
    
    # Sample if too many
    if len(X_val) > max_samples:
        idx = np.random.choice(len(X_val), max_samples, replace=False)
        X_sample = X_val[idx]
    else:
        X_sample = X_val
    
    # Choose explainer based on model type
    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(
            model.predict, 
            shap.sample(X_sample, 100)
        )
    
    shap_values = explainer.shap_values(X_sample)
    
    # Global importance (mean absolute SHAP)
    importance = {}
    for i, name in enumerate(feature_names):
        importance[name] = np.abs(shap_values[:, i]).mean()
    
    # Normalize
    total = sum(importance.values())
    if total > 0:
        importance = {k: v/total for k, v in importance.items()}
    
    return {
        'shap_values': shap_values,
        'feature_importance': importance,
        'explainer': explainer
    }
```

### 7.3 SHAP Visualizations

```python
def generate_shap_plots(shap_result: Dict, feature_names: List[str], output_dir: str):
    """Generate SHAP visualization plots."""
    import shap
    import matplotlib.pyplot as plt
    
    shap_values = shap_result['shap_values']
    
    # Summary plot (beeswarm)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    plt.savefig(f"{output_dir}/shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Bar plot (importance)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, feature_names=feature_names, 
                      plot_type='bar', show=False)
    plt.savefig(f"{output_dir}/shap_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Dependence plots for top features
    importance = shap_result['feature_importance']
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:5]
    
    for name, _ in top_features:
        idx = feature_names.index(name)
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(idx, shap_values, feature_names=feature_names, show=False)
        plt.savefig(f"{output_dir}/shap_dependence_{name}.png", dpi=150, bbox_inches='tight')
        plt.close()
```

---

## 8. Feature Visualizer

### 8.1 Overview

> **SUPERSEDED (2026-11-11) — replaced by §17.3 (D8).** There is no `FeatureVisualizer` class. At
> `a15b524` `feature_visualizer.py` is 13 module-level `generate_*` functions plus
> `get_feature_summary_html()` and a CLI (`feature_visualizer.py:42-1258`, `:1265-1423`). They take
> **importance dicts or importance files only** — no feature matrix `X`, no target `y` — and return
> Plotly HTML fragments. The class below is kept as design history.

`feature_visualizer.py` generates 13 chart types for comprehensive feature analysis:

```python
class FeatureVisualizer:
    """
    Comprehensive feature visualization system.
    
    Generates 13 chart types using Plotly, Seaborn, and Matplotlib.
    """
    
    def __init__(
        self,
        feature_importance: Dict[str, float],
        feature_names: List[str],
        X_data: np.ndarray,
        y_data: np.ndarray,
        output_dir: str = "visualization_outputs"
    ):
        self.importance = feature_importance
        self.feature_names = feature_names
        self.X = X_data
        self.y = y_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all(self):
        """Generate all 13 chart types."""
        self.importance_bar_chart()
        self.importance_radar_chart()
        self.correlation_matrix()
        self.feature_distribution()
        self.pairwise_scatter()
        self.importance_heatmap()
        self.cumulative_importance()
        self.feature_violin()
        self.top_features_boxplot()
        self.shap_waterfall()
        self.shap_force_plot()
        self.feature_cluster_dendrogram()
        self.interactive_dashboard()
```

---

## 9. 13 Chart Types

### 9.1 Chart Catalog

| # | Chart Type | Library | Purpose |
|---|------------|---------|---------|
| 1 | **Importance Bar** | Plotly | Ranked feature importance |
| 2 | **Importance Radar** | Plotly | Radial importance view |
| 3 | **Correlation Matrix** | Seaborn | Feature correlations |
| 4 | **Feature Distribution** | Seaborn | Histograms per feature |
| 5 | **Pairwise Scatter** | Seaborn | Feature relationships |
| 6 | **Importance Heatmap** | Plotly | Category-grouped importance |
| 7 | **Cumulative Importance** | Matplotlib | Pareto analysis |
| 8 | **Feature Violin** | Seaborn | Distribution by target |
| 9 | **Top Features Boxplot** | Seaborn | Spread of top features |
| 10 | **SHAP Waterfall** | SHAP | Individual prediction explain |
| 11 | **SHAP Force Plot** | SHAP | Feature contribution flow |
| 12 | **Feature Cluster Dendrogram** | Scipy | Feature groupings |
| 13 | **Interactive Dashboard** | Plotly | Combined HTML dashboard |

> **SUPERSEDED (2026-11-11) — the as-built catalogue is §17.3 (D8).** The count of 13 is right; the set is
> not. Of the thirteen above, only the bar chart and the radar chart exist as described. "Heatmap" exists
> but plots importance **over runs** (`feature_visualizer.py:846-937`), and "waterfall" exists but plots the
> **drift change** between two importance files (`:1013-1087`). Correlation matrix, feature-value
> distributions, pairwise scatter, cumulative/Pareto, violin, boxplot and dendrogram are not produced — the
> visualizer never reads feature values. SHAP waterfall and force plots are part of B2. The combined
> dashboard is the CLI's `feature_report.html` (`:1338-1416`) and the `/features` route of `web_dashboard.py`
> (`:2033-2088`), not a 2×2 subplot. §9.2–§9.6 are kept as design history.

### 9.2 Chart 1: Importance Bar Chart

```python
def importance_bar_chart(self, top_n: int = 20):
    """Horizontal bar chart of top N features by importance."""
    import plotly.express as px
    
    # Sort and select top N
    sorted_imp = sorted(self.importance.items(), key=lambda x: -x[1])[:top_n]
    names = [x[0] for x in sorted_imp]
    values = [x[1] for x in sorted_imp]
    
    fig = px.bar(
        x=values, y=names,
        orientation='h',
        title=f"Top {top_n} Feature Importance",
        labels={'x': 'Importance', 'y': 'Feature'}
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    fig.write_html(self.output_dir / "importance_bar.html")
    fig.write_image(self.output_dir / "importance_bar.png", scale=2)
```

### 9.3 Chart 2: Importance Radar Chart

```python
def importance_radar_chart(self, top_n: int = 10):
    """Radar/spider chart for top features."""
    import plotly.graph_objects as go
    
    sorted_imp = sorted(self.importance.items(), key=lambda x: -x[1])[:top_n]
    names = [x[0] for x in sorted_imp]
    values = [x[1] for x in sorted_imp]
    
    # Close the polygon
    names = names + [names[0]]
    values = values + [values[0]]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=names,
        fill='toself',
        name='Importance'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(values)*1.1])),
        title=f"Feature Importance Radar (Top {top_n})"
    )
    
    fig.write_html(self.output_dir / "importance_radar.html")
    fig.write_image(self.output_dir / "importance_radar.png", scale=2)
```

### 9.4 Chart 3: Correlation Matrix

```python
def correlation_matrix(self, method: str = 'pearson'):
    """Heatmap of feature correlations."""
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.DataFrame(self.X, columns=self.feature_names)
    corr = df.corr(method=method)
    
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr, mask=mask,
        cmap='RdBu_r', center=0,
        annot=False,  # Too many features for annotations
        square=True,
        linewidths=0.5
    )
    
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(self.output_dir / "correlation_matrix.png", dpi=150)
    plt.close()
```

### 9.5 Chart 7: Cumulative Importance (Pareto)

```python
def cumulative_importance(self):
    """Pareto chart showing cumulative importance."""
    import matplotlib.pyplot as plt
    
    sorted_imp = sorted(self.importance.items(), key=lambda x: -x[1])
    names = [x[0] for x in sorted_imp]
    values = [x[1] for x in sorted_imp]
    cumulative = np.cumsum(values)
    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Bar chart
    ax1.bar(range(len(names)), values, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Feature Rank')
    ax1.set_ylabel('Importance', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    
    # Cumulative line
    ax2 = ax1.twinx()
    ax2.plot(range(len(names)), cumulative, 'r-', linewidth=2, marker='o', markersize=3)
    ax2.set_ylabel('Cumulative Importance', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # 80% line
    ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='80% threshold')
    
    # Find how many features for 80%
    n_80 = np.searchsorted(cumulative, 0.8) + 1
    ax1.axvline(x=n_80, color='green', linestyle='--', alpha=0.7)
    
    plt.title(f"Cumulative Feature Importance (80% at {n_80} features)")
    plt.tight_layout()
    plt.savefig(self.output_dir / "cumulative_importance.png", dpi=150)
    plt.close()
```

### 9.6 Chart 13: Interactive Dashboard

```python
def interactive_dashboard(self):
    """Combined interactive HTML dashboard."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Feature Importance (Bar)",
            "Feature Importance (Radar)",
            "Top Feature Distributions",
            "Cumulative Importance"
        ),
        specs=[
            [{"type": "bar"}, {"type": "polar"}],
            [{"type": "histogram"}, {"type": "scatter"}]
        ]
    )
    
    # Add all subplots...
    # (Implementation details for each subplot)
    
    fig.update_layout(
        height=900,
        width=1400,
        title_text="Feature Importance Dashboard"
    )
    
    fig.write_html(self.output_dir / "dashboard.html")
```

---

## 10. AI-Powered Interpretation

> **SUPERSEDED (2026-11-11) — corrected (D2, D12).** At `a15b524` the interpreter is
> `feature_importance_ai_interpreter.py` v2.0.0. `interpret_with_ai(step5_path, drift_path)` (`:208`) builds a
> prompt from `feature_importance_step5.json` and the drift file and sends it through a fallback chain:
> **DeepSeek-R1-14B at `http://localhost:8080/completion`** (health check `:36-42`, query `:51-85`), then the
> **Claude CLI** (`claude -p`, `:88-113`), then a **template** (`:116-169`) (`query_llm`, `:172-205`). The
> server configuration agrees (`llm_services/llm_server_config.json` `primary.port` 8080,
> `backup.access_method` `claude_code_cli`). It prints to stdout; it writes no file. A template-only
> variant is `feature_importance_interpreter.py`. **History:** this section was updated to DeepSeek at
> `49e8ad4` (per `docs/PATCH_Chapter11_LLM_Update_v2.md`) and **reverted by `323b6fa` ("Doc sync
> 20260209_1704")**, a stale-copy overwrite of the same class as Chapter 2 §0.1. Like every consumer here it
> reads a file nothing writes (B1). **Framing:** "PRNG seed prediction model" below, and the code's own
> prompt "how the PRNG can be predicted" (`:260`), misdescribe the object: importance describes which
> inputs the **mimicry model** uses to rank survivors (§19). The code text is recorded, not changed. The
> original text is kept below.

### 10.1 LLM Integration

```python
def generate_ai_interpretation(
    importance: Dict[str, float],
    model_metrics: Dict[str, float],
    llm_endpoint: str = "http://localhost:8081/v1/completions"
) -> str:
    """
    Generate AI-powered interpretation using Qwen2.5-Math.
    
    Uses the Math LLM for statistical reasoning about features.
    """
    import requests
    
    # Prepare prompt
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:10]
    features_str = "\n".join([f"  {name}: {value:.4f}" for name, value in top_features])
    
    prompt = f"""Analyze this feature importance distribution from a PRNG seed prediction model:

Top 10 Features:
{features_str}

Model Metrics:
  Validation MAE: {model_metrics.get('val_mae', 'N/A')}
  Test MAE: {model_metrics.get('test_mae', 'N/A')}
  Overfit Ratio: {model_metrics.get('overfit_ratio', 'N/A')}

Questions to address:
1. Is the importance distribution healthy (spread across multiple features)?
2. Are there signs of circular/leaky features?
3. Which feature categories dominate (intersection, skip, lane, temporal)?
4. Recommendations for feature engineering?

Provide a concise statistical interpretation."""

    response = requests.post(
        llm_endpoint,
        json={
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.3
        }
    )
    
    return response.json()['choices'][0]['text']
```

### 10.2 Example AI Interpretation

```
AI Interpretation (Qwen2.5-Math-7B):
────────────────────────────────────
The feature importance distribution shows a HEALTHY pattern:

1. **Distribution Quality**: Top 10 features account for ~65% of importance,
   with remaining 35% spread across 52 features. This indicates the model
   is learning multiple predictive signals, not overfitting to one.

2. **No Circular Features Detected**: 'residue_1000_match_rate' is NOT in
   top 10, suggesting training target is correctly defined as holdout_hits
   rather than training score.

3. **Category Analysis**:
   - Intersection features (32%): Strongest predictors, indicating
     bidirectional sieve agreement is key
   - Skip features (24%): Second strongest, confirming skip hypothesis
     consistency matters
   - Lane features (18%): CRT validation adds predictive value
   - Temporal features (14%): Time stability provides moderate signal

4. **Recommendations**:
   - Consider combining intersection_weight and bidirectional_selectivity
     into a single "bidirectional_quality" meta-feature
   - skip_entropy and skip_range are highly correlated (r=0.87), may
     be redundant
   - Explore non-linear interactions between lane_agreement_* features
```

> **Corrected 2026-11-11 (X5, X6; D6):** an illustrative example, not a recorded run. "52 features" follows
> the superseded 62 count (§18). "Training target correctly defined as holdout_hits" is out of date: the
> target is `holdout_quality` (Chapter 6 audit S1). The category percentages could not come from this code:
> it has no intersection/skip/lane/temporal categories (§3.2 note, B4), and the intersection and skip-range
> columns are trial-level (Chapter 14 audit Q5).

---

## 11. Integration with Pipeline

### 11.1 After Step 5 Training

> **SUPERSEDED (2026-11-11) — there is no integration with the pipeline at `a15b524` (BROKEN B1; CLI D9).**
> Step 5 does not call `feature_importance.py` (§17.2). The three commands below do not run as written:
> Step 5's flag is `--trials`, not `--n-trials`, and `--lottery-data` is required
> (`meta_prediction_optimizer_anti_overfit.py:3043`, `:3045`); `feature_importance.py` has no CLI — its
> `__main__` is a self-test that prints library availability (`feature_importance.py:712-724`);
> `feature_visualizer.py` takes `--step4 --step5 --drift --output --top-n --charts`
> (`feature_visualizer.py:1268-1274`). What a hand run can do today is §17.4. Kept as design history.

```bash
# Train model (Step 5)
python3 meta_prediction_optimizer_anti_overfit.py \
    --survivors survivors_with_scores.json \
    --model-type neural_net \
    --n-trials 50

# Analyze feature importance
python3 feature_importance.py \
    --model-path models/reinforcement/best_model.pth \
    --survivors survivors_with_scores.json \
    --output-dir visualization_outputs

# Generate all visualizations
python3 feature_visualizer.py \
    --importance-json visualization_outputs/importance.json \
    --data-file survivors_with_scores.json \
    --output-dir visualization_outputs
```

### 11.2 With --save-all-models (TODO)

```bash
# Train all 4 models
python3 meta_prediction_optimizer_anti_overfit.py \
    --survivors survivors_with_scores.json \
    --compare-models \
    --save-all-models

# Compare importance across models
python3 feature_importance.py \
    --model-dir models/reinforcement/ \
    --compare-all \
    --output-dir visualization_outputs

# Generates:
#   importance_neural_net.json
#   importance_xgboost.json
#   importance_lightgbm.json
#   importance_catboost.json
#   importance_comparison.html
```

> **Verified 2026-11-11 (accurate as a TODO):** still not built. Step 5 has no `--save-all-models` flag
> (`meta_prediction_optimizer_anti_overfit.py` argparse `:3036-3105`); it survives only in
> `backups/step6_restoration_20251223_213936/meta_prediction_optimizer_anti_overfit.py:928`.
> `feature_importance.py` has no `--model-dir`/`--compare-all`; its `compare_importance()` compares two
> importance dicts for drift (`:640-676`), not models.

---

## 12. CLI Interface

> **SUPERSEDED (2026-11-11) — as-built CLIs in §17.4 (D9).** `feature_importance.py` has **no**
> command-line interface (`:712-724` is a self-test). `feature_visualizer.py`'s real flags are
> `--step4` (default `feature_importance_step4.json`), `--step5` (default `feature_importance_step5.json`),
> `--drift` (default `feature_drift_step4_to_step5.json`), `--output` (default `feature_report.html`),
> `--top-n` (default 20), `--charts` (`all|basic|comparison|<comma list>`) (`feature_visualizer.py:1268-1274`,
> `:1300-1307`). None of `--importance-json`, `--data-file`, `--format`, `--dpi`, `--output-dir`, `--method`,
> `--model-path`, `--json-only` exists. Kept as design history.

### 12.1 feature_importance.py

```bash
python3 feature_importance.py [options]

Options:
  --model-path PATH       Path to trained model file
  --model-type TYPE       Model type: neural_net, xgboost, lightgbm, catboost
  --survivors PATH        Path to survivors JSON with features
  --method METHOD         Importance method: permutation, gradient, shap, all
  --n-repeats INT         Repeats for permutation (default: 10)
  --max-samples INT       Max samples for SHAP (default: 1000)
  --output-dir DIR        Output directory (default: visualization_outputs)
  --json-only             Only output JSON, skip plots
  -h, --help              Show help
```

### 12.2 feature_visualizer.py

```bash
python3 feature_visualizer.py [options]

Options:
  --importance-json PATH  Path to importance JSON from feature_importance.py
  --data-file PATH        Path to survivors JSON with features
  --charts CHARTS         Comma-separated chart types (default: all)
  --top-n INT             Number of top features to show (default: 20)
  --output-dir DIR        Output directory (default: visualization_outputs)
  --format FORMAT         Output format: png, html, both (default: both)
  --dpi INT               DPI for PNG output (default: 150)
  -h, --help              Show help
```

### 12.3 Example Workflows

**Quick importance check:**
```bash
python3 feature_importance.py \
    --model-path best_model.pth \
    --model-type neural_net \
    --survivors survivors.json \
    --method permutation \
    --json-only
```

**Full analysis with all charts:**
```bash
python3 feature_importance.py \
    --model-path best_model.pth \
    --model-type neural_net \
    --survivors survivors.json \
    --method all \
    --output-dir analysis_run_001

python3 feature_visualizer.py \
    --importance-json analysis_run_001/importance.json \
    --data-file survivors.json \
    --output-dir analysis_run_001 \
    --format both
```

---

## 13. Output Formats

> **SUPERSEDED (2026-11-11) — corrected (D4, D10).** No `importance.json` and no `visualization_outputs/`
> are written by any tracked code. What exists:
> - `FeatureImportanceResult` (`feature_importance.py:62-88`) — fields `computation_method`, `model_version`,
>   `timestamp`, `total_features`, `importance_by_feature`, `importance_by_category`, `top_10_features`,
>   `bottom_10_features`, `model_type_detected`; `.save(path)` writes it as JSON. No caller in the pipeline.
> - The files the visualizer, dashboard and interpreters **read**: `feature_importance_step4.json`,
>   `feature_importance_step5.json` (a dict with `feature_importance`, or `importance_by_feature`, or a bare
>   dict — `feature_visualizer.py:85-91`), and `feature_drift_step4_to_step5.json`. Nothing writes them (B1).
> - The visualizer's only file output: one `feature_report.html` (`:1414-1416`). No PNG, no CSV; the AI
>   interpreter prints to stdout and writes no `ai_interpretation.txt`.
> Kept as design history.

### 13.1 importance.json

```json
{
    "method": "permutation",
    "model_type": "neural_net",
    "n_features": 62,
    "timestamp": "2025-12-30T14:23:45",
    "importance": {
        "intersection_weight": 0.182,
        "skip_entropy": 0.156,
        "lane_agreement_8": 0.134,
        "temporal_stability_mean": 0.098,
        "bidirectional_selectivity": 0.087,
        ...
    },
    "metadata": {
        "n_samples": 5000,
        "n_repeats": 10,
        "baseline_score": 0.0423,
        "computation_time_seconds": 127.4
    }
}
```

### 13.2 Generated Files

```
visualization_outputs/
├── importance.json              # Raw importance scores
├── importance_bar.html          # Interactive bar chart
├── importance_bar.png           # Static bar chart
├── importance_radar.html        # Interactive radar
├── importance_radar.png         # Static radar
├── correlation_matrix.png       # Feature correlations
├── feature_distributions.png    # Histograms
├── cumulative_importance.png    # Pareto chart
├── feature_violin.png           # Violin plots
├── top_features_boxplot.png     # Box plots
├── shap_summary.png             # SHAP beeswarm
├── shap_bar.png                 # SHAP importance
├── shap_dependence_*.png        # Per-feature SHAP
├── feature_dendrogram.png       # Hierarchical clustering
├── dashboard.html               # Combined interactive
└── ai_interpretation.txt        # LLM analysis
```

---

## 14. Troubleshooting

### 14.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "SHAP too slow" | Large dataset | Use `--max-samples 500` |
| "All importance = 0" | Model not trained | Verify model loaded correctly |
| "2 features = 100%" | Circular features | Check training target is holdout_hits |
| "Memory error" | Too many features | Use permutation instead of SHAP |
| "Plotly not found" | Missing dependency | `pip install plotly kaleido` |

> **Corrected 2026-11-11 (B2, D4, X5):** the SHAP row has no code to act on (B2). `kaleido` is not needed —
> nothing calls `write_image` (D4); the visualizer needs `plotly` only. "Check training target is
> holdout_hits": the target is `holdout_quality` (`meta_prediction_optimizer_anti_overfit.py:1481`;
> Chapter 6 audit S1). "Missing data" is the most likely real failure: every consumer reads files nothing
> writes (B1); the dashboard says "Run Step 4 or Step 5 to generate data" (`web_dashboard.py:2056`; also
> `feature_visualizer.py:1215`), and neither step generates them any more.

### 14.2 Verify Correct Training

```python
# Check if importance is healthy
importance = load_json("importance.json")["importance"]

top_10 = sorted(importance.items(), key=lambda x: -x[1])[:10]
top_10_sum = sum(v for _, v in top_10)

if top_10_sum > 0.95:
    print("⚠️  WARNING: Top 10 features = 95%+ importance")
    print("   This suggests circular features in training")
    print("   Check that y = holdout_hits, not score")
else:
    print(f"✅ Healthy distribution: Top 10 = {top_10_sum:.1%}")
```

> **Verified 2026-11-11 (accurate):** the same 95 % threshold is in code, in the template interpreter:
> above 0.95 it warns of "possible circular/leaky features", above 0.80 it reports moderate concentration
> (`feature_importance_ai_interpreter.py:135-145`). The "y = holdout_hits" line is out of date (X5,
> Chapter 6 audit S1). A concentrated distribution is evidence about the fitted model's inputs; it is not
> by itself evidence of circularity — 9 trained columns are constant and 13 are trial-level (Chapter 14
> audit Q3/Q5), which also concentrates importance.

---

## 15. Complete Method Reference

> **SUPERSEDED (2026-11-11) — the as-built reference is §17.1 and §17.3 (D7, D8; B2, B3).** None of the
> six functions in §15.1 exists under that name; `compute_shap_values`, `generate_shap_plots` and
> `compute_integrated_gradients` exist under no name. None of the thirteen methods in §15.2 exists (there is
> no class). Kept as design history.

### 15.1 feature_importance.py

| Function | Purpose |
|----------|---------|
| `compute_permutation_importance()` | Model-agnostic importance |
| `compute_gradient_importance()` | Neural net gradients |
| `compute_integrated_gradients()` | Advanced gradient attribution |
| `compute_shap_values()` | SHAP analysis |
| `generate_shap_plots()` | SHAP visualizations |
| `compare_model_importance()` | Cross-model comparison |

### 15.2 feature_visualizer.py

| Method | Chart Type |
|--------|------------|
| `importance_bar_chart()` | #1 Bar chart |
| `importance_radar_chart()` | #2 Radar chart |
| `correlation_matrix()` | #3 Heatmap |
| `feature_distribution()` | #4 Histograms |
| `pairwise_scatter()` | #5 Scatter matrix |
| `importance_heatmap()` | #6 Category heatmap |
| `cumulative_importance()` | #7 Pareto chart |
| `feature_violin()` | #8 Violin plots |
| `top_features_boxplot()` | #9 Box plots |
| `shap_waterfall()` | #10 SHAP waterfall |
| `shap_force_plot()` | #11 SHAP force |
| `feature_cluster_dendrogram()` | #12 Dendrogram |
| `interactive_dashboard()` | #13 HTML dashboard |

---

## 16. Chapter Summary

> **Corrected 2026-11-11 (D11, and the rows below):** this is **Chapter 11**, not Chapter 14 (Chapter 14 is
> Training Diagnostics). In the table: "Permutation" and "Gradient" are accurate for the module; "SHAP" is not
> implemented (B2); "13 Charts" is the right count of a different set (D8); "AI Interpretation" uses
> DeepSeek-R1-14B → Claude CLI → template (D2). In the key points: three methods = native, permutation,
> gradient; no circular-feature detection exists beyond the 95 % concentration warning (Chapter 6 audit D5);
> export is HTML (and JSON via `FeatureImportanceResult.save`), not PNG. The as-built summary is §17.

**Chapter 14: Feature Importance & Visualization** covers ML interpretability:

| Component | Purpose |
|-----------|---------|
| **Permutation** | Model-agnostic importance |
| **Gradient** | Neural net attributions |
| **SHAP** | Individual + global explanations |
| **13 Charts** | Comprehensive visualization |
| **AI Interpretation** | LLM-powered analysis |

**Key Points:**
- 3 importance methods for different use cases
- 13 chart types cover all visualization needs
- Circular feature detection prevents overfitting
- AI interpretation via Qwen2.5-Math
- Export to PNG, HTML (interactive), JSON

---

## 17. What exists at a15b524, and who calls it

*Added 2026-11-11. Every anchor opened in this pass; all files identical at `a15b524` and `b6cb789`.*

### 17.1 `feature_importance.py` (724 lines, v1.1.0)

| Symbol | What it does | Anchor |
|---|---|---|
| `get_feature_importance(model, X, y, feature_names, method='auto', n_repeats=10, device='cuda:0', normalize=True)` | Detect model type, resolve method, dispatch, sort by absolute value | `:126-213` |
| `_detect_model_type` | `pytorch` / `xgboost` / `sklearn_tree` / `sklearn_other` / `unknown`, by interface | `:220-246` |
| `_resolve_method` | `auto` → native for `xgboost`/`sklearn_tree`, permutation otherwise; refuses unknown methods | `:261-295` |
| `_extract_native_importance`, `_extract_xgboost_booster_importance` | `feature_importances_`, or booster `gain` with an `f{i}` fallback | `:302-359` |
| `_extract_permutation_importance` (+ `_pytorch`) | sklearn `neg_mean_squared_error`, `random_state=42`; or a GPU MSE loop | `:362-462` |
| `_extract_gradient_importance` | mean absolute input gradient | `:465-499` |
| `FeatureImportanceExtractor.extract` → `FeatureImportanceResult` | adds category sums, top/bottom 10, metadata | `:506-599`, `:62-88` |
| `compare_importance` | per-feature delta, mean absolute drift, threshold 0.15 | `:640-676` |
| `get_importance_summary_for_agent` | top 5, statistical/global weights | `:679-705` |
| `STATISTICAL_FEATURES` (46), `GLOBAL_STATE_FEATURES` (14) | the Dec-2025 60-name lists; stale (B4) | `:95-119` |

### 17.2 Callers — the null result

**No pipeline step calls this subsystem.** Denominator: every tracked and gitignored `*.py`, `*.sh`,
`*.json` (`/bin/grep -rn`), git history, host.

- `meta_prediction_optimizer_anti_overfit.py` (executable Step 5) has no reference to `feature_importance`.
  The writer of `feature_importance_step5.json` was removed at `e0b55a1` (2025-12-20) and the import at
  `249052b` (2025-12-22); neither commit message mentions it, and no governance document records the
  removal. The approved `docs/PROPOSAL_Addendum_H_Feature_Importance_Integration.md` still reads
  "IMPLEMENTED" (`:6`, `:60-70`). **BROKEN B1.**
- The only remaining code caller is `meta_prediction_optimizer.py:41`, `:195`. It is not in `STEP_SCRIPTS`
  (`git show a15b524:agents/watcher_agent.py:387-395`) and nothing invokes it.
- Tests: `test_feature_importance.py`, `test_feature_importance_dual_gpu.py`, `test_feature_importance_zeus.py`
  (root, not under `tests/`).
- No manifest, no `agent_manifests/definitions.json` entry, no cron entry, no systemd unit.

### 17.3 `feature_visualizer.py` (1422 lines, v1.0.0) — the 13 charts that exist

All Plotly, all over importance dicts or files; none reads feature values (`feature_visualizer.py:1283-1297`).

| CLI id | Function | What it shows | Inputs |
|---|---|---|---|
| `importance_bar` | `generate_feature_importance_chart` `:42` | top-N bar | step 5 file |
| `drift_comparison` | `generate_drift_comparison_chart` `:157` | step 4 vs step 5 bars | step 4 + 5 |
| `drift_gauge` | `generate_drift_indicator_chart` `:264` | drift score gauge | drift file |
| `top_movers` | `generate_top_movers_chart` `:328` | largest changes | drift file |
| `category_pie` | `generate_category_breakdown_chart` `:403` | statistical vs global (0 % global for live names, B4) | step 5 |
| `scatter` | `generate_importance_scatter_chart` `:472` | step 4 vs step 5 scatter | step 4 + 5 |
| `treemap` | `generate_importance_treemap` `:586` | prefix-based statistical/global split | step 5 |
| `radar` | `generate_radar_chart` `:669` | top-10 radar | step 5 |
| `dual_radar` | `generate_dual_radar_chart` `:747` | step 4 vs step 5 radar | step 4 + 5 |
| `heatmap` | `generate_importance_heatmap` `:846` | importance over runs (`feature_importance_history/`, else step 4/5) | history |
| `histogram` | `generate_distribution_histogram` `:939` | distribution of importance values | step 5 |
| `waterfall` | `generate_waterfall_chart` `:1013` | drift change waterfall | drift file |
| `sunburst` | `generate_sunburst_chart` `:1089` | substring-based groups (Lane, Residue, Skip/Temporal, Score/Confidence, Global State, Other) | step 5 |

Each returns an HTML fragment (`fig.to_html(full_html=False, include_plotlyjs="cdn")`). The same functions feed
`web_dashboard.py`'s `/features` route (import `:38-57`, route `:2033-2088`, server port 5002). The CLI
writes one `feature_report.html` (`:1338-1416`).

### 17.4 What a hand run can do today

Read-only description; nothing was run in this pass.

- `python3 feature_visualizer.py [--step4 F] [--step5 F] [--drift F] [--output F] [--top-n N] [--charts all|basic|comparison|ids]`
  renders whatever importance files it is given into one HTML report.
- `python3 feature_importance_ai_interpreter.py [--step5 F] [--drift F] [--test-llm]` interprets a step-5
  file through DeepSeek → Claude CLI → template, to stdout (`:302-332`).
- `python3 feature_importance.py` and `python3 feature_drift_tracker.py` run self-tests only (`:712-724`;
  `feature_drift_tracker.py:610-663`).
- To get importance for a trained model, a caller must load the model itself and call
  `get_feature_importance()` with the sidecar's 89 names in sidecar order (§18). No script in the tree does that.

**On the host today the inputs are not real.** `feature_importance_step4.json` and `_step5.json` (both
2025-12-10, `model_type` `neural_network`, `extraction_method` `permutation`) hold 24 names, **4 of 24**
present in the live 91 (`lane_agreement_8`, `skip_entropy`, `survivor_overlap_ratio`,
`temporal_stability_mean`); names such as `lane_agreement_16`, `window_coverage`, `bidirectional_score`
occur nowhere in the code or its history. Their key shape is not the removed writer's. The dashboard
would render them as if they were a model's importance.

### 17.5 `feature_drift_tracker.py`, the interpreters, and history

- `feature_drift_tracker.py` (663 lines): `FeatureDriftTracker` keeps a history file
  (`feature_importance_history.json`), a baseline, and a `DriftAnalysis` per comparison: mean absolute delta,
  alert above 0.15, top gainers/losers, `statistical_drift`/`global_drift` over the same stale lists
  (`:131-190`, `:228-460`; `compare_importance_files` `:513-557`). Its agent summary reports `status`
  `ALERT`/`stable` (`:602`); the visualizer colours `drift_detected`, which the tracker never emits
  (`feature_visualizer.py:285`, `:1228`) — colour only. No caller in the pipeline. The original chapter does
  not mention drift tracking at all (D1).
- `feature_importance_interpreter.py` (140 lines): template-only interpretation by substring category.
- **History of this chapter's §10:** `49e8ad4` applied `docs/PATCH_Chapter11_LLM_Update_v2.md`
  (Qwen → DeepSeek + Claude); `323b6fa` ("Doc sync 20260209_1704", 24 files) put the old 12 lines back.

---

## 18. The feature contract: 91 extracted / 89 trained

*Added 2026-11-11; re-derived from code, then checked against the host.* Replaces §3.1.

| Origin | Count | Anchor |
|---|---|---|
| Match/statistics keys built on the GPU (`score`, `confidence`, `exact_matches`, `total_predictions`, `best_offset`, 4 `pred_*`, 4 `residual_*`, 2 `actual_*`, 3 lane, 4 digit, 5 temporal) | 27 | `survivor_scorer.py:702-731` |
| Residue: `match_rate`, `kl_divergence`, `coherence` × mods 8, 125, 1000 | 9 | `:74`, `:625-667` |
| Battery Tier 1A (`batt_*`) | 23 | `:756` |
| NPZ-merged trial-level fields, with 0.0 defaults for the unproduced ones | 18 | merge `:774-779`, defaults `:783-791` |
| **Scorer total** | **77** | |
| Run-level `global_*` from `GlobalStateTracker` | 14 | `full_scoring_worker.py:403-405`, `:458-460` |
| **Extracted (`features` dict)** | **91** | |
| Excluded by Step 5 from `features`: `score`, `confidence` | −2 | `meta_prediction_optimizer_anti_overfit.py:1479-1483`, `:546-597` |
| **Trained** | **89** | sidecar `feature_schema.feature_names` (89 entries, host) |

`holdout_hits` and `holdout_quality` are also in the exclude list, but they are top-level record keys, not
in `features`. Order: `lexicographic_by_key` (`meta_prediction_optimizer_anti_overfit.py:594`). Host check: first record of
`survivors_with_scores.json` (2026-03-11) has 91 feature keys; 14 start with `global_`. This agrees with
`docs/S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md` §1.3–1.5 and `docs/PROJECT_FILE_CATALOG.md:933`.
Older documents say 62 (Dec 2025, after `e0b55a1`) or 60 (Addendum H). `agent_manifests/definitions.json:178`
still carries an example `"feature_count": 62`.

What sits inside the 89, for reading importance (Chapter 14 audit Q3/Q5):
- **13 trial-level** NPZ fields, identical for every seed of a trial.
- **14 run-level** `global_*`, identical for every seed of a run.
- **9 constant** columns: `skip_mean`, `skip_std`, `skip_entropy`, `survivor_velocity`,
  `velocity_acceleration` (0.0 placeholders), `best_offset` (hardcoded 0), `total_predictions`,
  `actual_mean`, `actual_std` (history-level).
- `forward_match_rate` / `reverse_match_rate` — the per-seed sieve match quality — are **not** in `X`
  (governed: Chapter 14 audit Q4/S5).

---

## 19. How to read an importance value

*Added 2026-11-11.*

1. **Importance describes the mimicry model, not the machine.** It says which columns the fitted ranking
   model leans on to order survivors by `holdout_quality`. It is not evidence about the generator family,
   a seed, or the draw machine (§1 framing). A high value for a residue or lane column means the model
   uses that column; it does not mean the machine "has" that structure.
2. **Trial-level and run-level columns are not per-seed evidence.** 13 NPZ fields are copied onto every
   seed of a trial and 14 `global_*` onto every seed of a run (Chapter 14 audit Q5). Their importance
   is about which **trial** a survivor came from. Within a single-trial survivor set they are constant.
3. **Constant columns score zero.** Permutation importance of a constant column is 0 by construction, and
   native tree importance never splits on one. So 9 columns (§18) can never rank, whatever the model.
4. **Names must come from the sidecar.** The module zips names to positions without validation. Use
   `feature_schema.feature_names` from `best_model.meta.json`, in its lexicographic order. Training-time
   importance is keyed `f{i}`/`feature_{i}` (Chapter 14 audit B5).
5. **Category weights are unreliable until B4 is repaired** (0 % global for live names).
6. **Concentration is not proof of circularity.** The 95 % top-10 warning is a heuristic; constant and
   trial-level columns concentrate importance too.

---

## 20. RANGE-MINER, the 22-array NPZ and the window anchor

*Added 2026-11-11.*

- **RANGE-MINER** changes nothing in this subsystem directly; no Chapter 11 module reads an NPZ. Its effect
  is upstream: the 22-array contract fixes which survivor metadata exists, and **13 of the 22 arrays become
  trained features**, all trial-level (`utils/canonical_records.py:215-244`, `record.update(metadata_base)`
  `:254`; Chapter 14 audit Q5). The per-seed arrays `forward_match_rate`/`reverse_match_rate` are not merged
  into `features` (`survivor_scorer.py:774-779`).
- **The window-anchor change (Brief I)** sets canonical array 4 `offset` to the window anchor
  (`utils/canonical_records.py:216-225`). `offset` is not in the Step 3 merge list, so it is **not a feature**
  and the change does not alter the 91/89 contract or any importance name. `window_anchor` and
  `generator_phase` occur in no Chapter 11 module (`/bin/grep -c` → 0).
- `best_offset`, a trained feature, is unrelated to either: it is hardcoded 0 (`survivor_scorer.py:707`,
  sequential path `:373`).

---

## 21. Known gaps at a15b524

*Added 2026-11-11. Full detail and evidence in `docs/CHAPTER_11_AUDIT.md` §4–§6.*

### 21.1 BROKEN (4)

| # | Gap | Anchor | A repair must |
|---|---|---|---|
| B1 | No pipeline step produces global feature importance; the visualizer, dashboard and interpreters read files nothing writes | Step 5 has no `feature_importance` reference; removals `e0b55a1`, `249052b`; readers `feature_visualizer.py:69-72`, `web_dashboard.py:2043-2045`, `feature_importance_ai_interpreter.py:209-210` | name one producer that writes the shape the readers expect, keyed by the sidecar's 89 names — or state that these are standalone tools over a hand-supplied file |
| B2 | Global SHAP importance is not implemented | `feature_importance.py:261-295`; no `import shap` in tracked `*.py` | implement it, or state that global importance is native/permutation/gradient only |
| B3 | Integrated gradients are not implemented | no `integrated_gradients` in tracked `*.py` | implement it, or state that gradient importance is plain saliency |
| B4 | Category accounting cannot place live names: global weight 0, 45 of 89 names in "other" | `feature_importance.py:95-119`, `:601-633`, `:679-705`; `feature_drift_tracker.py:197-221`, `:429-436` | derive the category lists from the sidecar or the Step 3 producer |

### 21.2 SWITCHED-OFF-BY-RULING (0 counted here; 3 governed holds bear on this chapter)

Counted in the Chapter 6 and 14 audits and cross-referenced here. No recommendation for or against.

- **G5, feature importance in the sidecar** — TB approved the six S111 guardrails; the owner held the patch
  "until clean baseline established" (`docs/SESSION_CHANGELOG_20260225_S111.md:80-84`;
  `docs/S111_TEAM_BETA_BRIEFING.md:127-128`; Chapter 14 audit Addendum 2). **Lifting it would enable:** every
  trained model to carry its own importance in `best_model.meta.json`, and Chapter 13's feature-drift
  diagnostic (`chapter_13_diagnostics.py:449`, which reads that key and today gets `{}`) to have an input.
- **Real per-survivor attribution run** — blocked until schema, scaler and prediction-parity gates pass
  (Chapter 14 audit S2); attribution-driven autonomy not authorized (S4). **Lifting it would enable:**
  per-seed attributions from the existing `per_survivor_attribution.py` backends, after P0-C (Chapter 13 audit B15).
- **`y = holdout_quality` replacing `holdout_hits`** (Chapter 6 audit S1). **Lifting it would enable:** Step 5
  to train on exact holdout hits again; the chapter's "check y = holdout_hits" advice would then apply.

Related deferral, not a prohibition: the `/training` dashboard and TensorBoard (Chapter 14 audit S9). The
`/features` route does exist.

### 21.3 DESCRIPTION-ONLY (12)

| # | Chapter said | Code at `a15b524` |
|---|---|---|
| D1 | two files, ~600 lines | six pieces, 2146 lines in the two named files; drift tracker unmentioned (§17) |
| D2 | Qwen2.5-Math on 8081; `generate_ai_interpretation()`; `ai_interpretation.txt` | DeepSeek-R1-14B on 8080 → Claude CLI → template; `interpret_with_ai()`; stdout (§10 note) |
| D3 | four model types; methods omit native | detection by interface; `auto` → native for trees (§1.3 note) |
| D4 | PNG/HTML/JSON/CSV into `visualization_outputs/`; kaleido | one `feature_report.html`; JSON via `FeatureImportanceResult.save`; no PNG/CSV (§13 note) |
| D5 | `.xgb`, `.lgb` checkpoints | `.json`, `.txt` (`models/model_factory.py:29-32`) |
| D6 | 62 features | 91 extracted / 89 trained (§18) |
| D7 | `compute_permutation_importance()` with MAE; `compute_gradient_importance()` | `get_feature_importance()`; MSE; private extractors (§5.2 note) |
| D8 | `FeatureVisualizer` class and its 13-chart catalogue | 13 module-level Plotly functions over importance only (§17.3) |
| D9 | CLIs for both modules; Step 5 `--n-trials` | `feature_importance.py` has none; visualizer flags differ; Step 5 `--trials`, `--lottery-data` required (§11.1, §12 notes) |
| D10 | `importance.json` shape | `FeatureImportanceResult` fields; readers accept three shapes (§13 note) |
| D11 | "Chapter 14", "Next: Chapter 16 Prediction Generator" | Chapter 11; Prediction Generator is Chapter 7 |
| D12 | "PRNG seed prediction model" | importance is about the mimicry model (§19); code prompt text recorded, not changed |

---

## 22. Open items

- **U1 (UNVERIFIED):** whether `get_feature_importance()` works on the objects Step 5 actually saves (raw
  `lgb.train`/`xgb.train` Boosters, the NN dict checkpoint, CatBoost). On reading, raw Boosters would route
  to sklearn `permutation_importance` (`feature_importance.py:239-246`, `:383-392`). Settles by loading each
  `best_model.*` and calling it — a run, outside this pass.
- **Origin of the 2025-12-10 host files** — no writer in code or history; UNAVAILABLE.
- **The `323b6fa` doc-sync revert** may have reverted other files it touched (24 files, including Chapters 3,
  6, 9, 12); not checked here.
- Items owned elsewhere: Chapter 14 audit B4, B5, B18 (training-time importance), Chapter 13 audit B15
  (attribution), G5 (owner hold), Chapter 6 audit S1/D5.

---

## 23. Verification declaration (VIR-1…6)

**execution proof (VIR-1).** The files opened in this pass are listed in `docs/CHAPTER_11_AUDIT.md` §12.
Every `file:line` here was read on VM 101 this session with `Read`, `sed -n` or `/bin/grep -n`; the one
`agents/watcher_agent.py` anchor came from `git show a15b524:`. Host JSON was read with `python3 -I`
one-liners (keys and name sets only for the sidecar and `diagnostics_outputs/`). The main session's
five-anchor spot-check is recorded by the main session.

**clean control (VIR-2).** `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` was opened as the model and not edited
(`git diff --quiet` clean); `python3 tests/test_chapter2_content_gate.py` after the edits: "12/12 checks green
(6 gates + 6 mutants)", `SENTINEL : PASS`.
Sections of this chapter verified correct as written: §4.1 permutation and gradient rows, §5.1, §6.1, §6.2
(algorithm), §11.2 (as a TODO), §14.2 (threshold).

**fault-injection control.** Not applicable to a documentation pass. Stated, not invented: no detector, gate
or harness was run that could pass vacuously.

**completion sentinel (VIR-3).** §24.5.

**unavailable-observer behaviour (VIR-5).** Nothing here is established by execution. Runtime behaviour of the
extraction module on real Step 5 objects is U1, not assumed. The dashboard and LLM servers were down and were
not started; their behaviour is read from source. The origin of the host sample files is UNAVAILABLE.

**audit claim scope (VIR-6).** What this chapter says against what the code at `a15b524` does — not whether the
design is right, and not whether any feature carries signal.

*Searched surfaces:* tracked repo; gitignored files (`git status --ignored`, `git ls-files -o -i --exclude-standard`,
`/bin/grep` so `.json` is included); git history (`git log --all`, `-S`, `git show`); host (`models/reinforcement/`,
`diagnostics_outputs/`, root feature JSON files, `survivors_with_scores.json`, `agent_manifests/definitions.json`,
`~/dashboard_work`, `crontab -l`, `systemctl list-units --all`, `ss -ltn`).
*Unavailable surfaces:* runtime execution (by scope), ser8 archives, the private remote, the rigs.
*Governance trail searched:* `docs/TB_RULING_*` (word-bounded grep: none name feature importance or SHAP),
`docs/PROPOSAL_*` (10 hits; Addendum H read in full; Unified Agent Context Framework v3.2.3/v3.2.4
feature-importance sections), `docs/TEAM_ALPHA_*` (none), the S111 briefing and changelog, the S172
attribution trace report.
*Chapters searched:* this chapter; Chapters 2 (model), 5, 6, 13, 14 and their audits; grep for Chapter 11
mentions in every `CHAPTER_*_AUDIT.md`, `CHAPTER_14_TRAINING_DIAGNOSTICS.md`, `CHAPTER_3_SCORER_META_OPTIMIZER.md`;
`PIPELINE_BEHAVIOUR_MODEL.md` §1.1.

---

## 24. Closure statement

### 24.1 Verified against

Code at `a15b524` (working tree `b6cb789`, identical for every file cited outside `agents/`), host state of
VM 101 on 2026-11-11.

### 24.2 What this pass did

Added the §1 framing block, the status header, "How this chapter was produced", inline corrections in §1–§16,
SUPERSEDED markers on §3.1, §6.3, §7, §8.1, §9.1, §10, §11.1, §12, §13, §15, and the new §17–§24. Counted 33
claims: 10 accurate, 4 BROKEN, 0 SWITCHED-OFF-BY-RULING, 12 DESCRIPTION-ONLY, 1 UNVERIFIED, 6 absorbed
(10 + 4 + 0 + 12 + 1 + 6 = 33).

### 24.3 What this chapter is NOT

Not a closure of Chapter 11. Not a repair: B1–B4 are recorded, not fixed. Not a statement that any feature
carries signal about the machine. Not a recommendation on G5, the attribution blocks, or the target choice.

### 24.4 Length

1099 lines at `a15b524`; 1688 after this pass (`git diff --numstat a15b524`: 589 added, 0 deleted). Nothing was
removed; every original heading is still present. The chapter only grew.

### 24.5 Completion sentinel

**`CHAPTER_11_CORRECTION: PASS`** — every substantive claim labelled and counted; counts reconcile; framing,
status header, known gaps, open items, VIR declaration and closure present; edits made in place with the
Edit tool. Not a closure sentinel: U1 and the items in §22 stay open.

---

## Version History

| Date | Change |
|---|---|
| 2026-01-08 | Added (`53a3829`) |
| 2026-01-20 | §1.3, §10, §16 LLM text updated to DeepSeek-R1-14B + Claude (`49e8ad4`) |
| 2026-02-09 | That update reverted by "Doc sync 20260209_1704" (`323b6fa`) |
| 2026-11-11 | Correction pass against `a15b524`: framing block, status header, inline corrections, SUPERSEDED markers, §17–§24 (`docs/CHAPTER_11_AUDIT.md`) |

---

## Next Chapter

> **Corrected 2026-11-11 (D11):** the Prediction Generator is **Chapter 7**
> (`docs/CHAPTER_7_PREDICTION_GENERATOR.md`); the chapter after this one is Chapter 12 (WATCHER Agent,
> `docs/CHAPTER_12_WATCHER_AGENT.md`). The footer's "Chapter 14" means this chapter, Chapter 11.

**Chapter 16: Prediction Generator** will cover:
- `prediction_generator.py` — Generate final predictions
- Model loading with schema validation
- Top-K survivor ranking
- Agent metadata for autonomous operation

---

*End of Chapter 14: Feature Importance & Visualization*
