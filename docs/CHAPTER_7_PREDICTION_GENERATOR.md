# Chapter 7: Prediction Generator (Step 6)

> **What TFM is.** TFM filters candidate generators against observed draws in two directions,
> keeps a population of survivors on purpose, ranks them, and lets the top candidates vote their
> next value into a prediction pool. A survivor is a scored candidate, not a verdict. The
> generator family is a substrate for mimicry, not a claim about the machine. The target is a
> pool hit rate. TFM does not recover a seed or reconstruct state, and an objection that assumes
> it does is out of scope.

**Correction-pass status (2026-11-11):** **CORRECTED AGAINST CODE at `a15b524`**, not closed.
The v1.0 text (December 2025) is kept in full. Where it disagrees with the code, an inline
**"Corrected 2026-11-11"** note or a **SUPERSEDED (2026-11-11)** marker says what the code at
`a15b524` does, with an anchor. **Read §14–§19 (end of file) first:**
- §14: what Step 6 does today, including how the pool is actually built.
- §15: what RANGE-MINER, the 22-array NPZ and the window anchor changed.
- §16: the known gaps.
- §17: open items.
- §18: the VIR-1…6 declaration.
- §19: the closure statement and completion sentinel.

**Authority for this pass:** `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md` (REV1).
**Base:** `a15b5247336255801de8cf67d52653701b1e8c80` (`a15b524`) on VM 101. HEAD during this pass is `b6cb789`.
It changed only `agents/watcher_agent.py`, `agents/watcher_dispatch.py` and
`tests/test_watcher_safety_repairs.py`, so every WATCHER anchor here is taken from
`git show a15b524:agents/watcher_agent.py` and marked "at `a15b524`". Every other cited file is identical at
`a15b524` and HEAD (`git diff --quiet a15b524 -- <file>`).
**Audit:** `docs/CHAPTER_7_AUDIT.md` (47 claims: 18 accurate, 3 BROKEN, 1 SWITCHED-OFF-BY-RULING,
16 DESCRIPTION-ONLY, 1 UNVERIFIED, 8 absorbed).
**Numbering:** bare "Step N" in this chapter means the **executable** scheme
(`docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.1). Step 6 = `prediction_generator.py`
(`agents/watcher_agent.py:394` `STEP_SCRIPTS[6]` at `a15b524`; `agent_manifests/prediction.json:4`
`"pipeline_step": 6`). The §1.3 diagram's "Step 2.5" is the old name for executable Step 2.

### How this chapter was produced (2026-11-11 pass)

1. Every substantive claim was checked against the code at `a15b524`, by symbol, and given exactly one
   label: BROKEN, SWITCHED-OFF-BY-RULING or DESCRIPTION-ONLY (`docs/CHAPTER_7_AUDIT.md` §4–§6). The
   governance trail (`docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/TEAM_ALPHA_*`) was searched first,
   then the corrected Chapters 3, 4, 5, 6, 9, 12, 13 and 14 and their audits, then the code. A break
   already recorded there is cross-referenced, not re-reported.
2. Host state on VM 101 was read and never changed:
   - `predictions/` and its 19 history files (keys and metadata only);
   - the Step 5 sidecar (keys);
   - the forward/reverse survivor files;
   - `agent_manifests/definitions.json` (gitignored);
   - WATCHER logs, crontab and systemd unit lists.

   Nothing was predicted, trained or run.
3. The chapter was then corrected **in place, section by section**. No section was deleted; stale
   sections carry SUPERSEDED markers and keep their text.
4. Claim scope: what the chapter says, checked against what the code at `a15b524` does. Whether the
   design is right is not judged here.

## PRNG Analysis Pipeline — Complete Operating Guide

**Version:** 1.0 · 1.1 correction overlay 2026-11-11  
**File:** `prediction_generator.py`  
**Lines:** ~400  
**Purpose:** Generate final predictions using trained ML model

> **Corrected 2026-11-11 (DESCRIPTION-ONLY, D1):** at `a15b524` the file has 1,095 lines. It calls itself
> v6.1 (`prediction_generator.py:15`); the CLI banner says v6.0 (`:962`). The purpose line is accurate:
> the trained model scores the survivors (`:828`). Step 6 also depends on `models/model_factory.py`
> (`load_model_from_sidecar`), `survivor_scorer.py` (sequence regeneration, dual-sieve intersection,
> on-the-fly features) and `integration/metadata_writer.py`.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture Position](#2-architecture-position)
3. [Model Loading with Sidecar](#3-model-loading-with-sidecar)
4. [Feature Schema Validation](#4-feature-schema-validation)
5. [Survivor Ranking](#5-survivor-ranking)
6. [Prediction Pool Generation](#6-prediction-pool-generation)
7. [Next-Draw Prediction](#7-next-draw-prediction)
8. [Output Formats](#8-output-formats)
9. [CLI Interface](#9-cli-interface)
10. [Agent Integration](#10-agent-integration)
11. [Troubleshooting](#11-troubleshooting)
12. [Complete Method Reference](#12-complete-method-reference)
13. [Chapter Summary](#13-chapter-summary)
14. [What Step 6 Does at a15b524](#14-what-step-6-does-at-a15b524) *(added 2026-11-11)*
15. [What RANGE-MINER, the 22-Array NPZ and the Window Anchor Changed](#15-what-range-miner-the-22-array-npz-and-the-window-anchor-changed) *(added 2026-11-11)*
16. [Known Gaps at a15b524](#16-known-gaps-at-a15b524) *(added 2026-11-11)*
17. [Open Items](#17-open-items) *(added 2026-11-11)*
18. [Verification Declaration (VIR-1…6)](#18-verification-declaration-vir-16) *(added 2026-11-11)*
19. [Closure Statement](#19-closure-statement) *(added 2026-11-11)*

---

## 1. Overview

### 1.1 What the Prediction Generator Does

The prediction generator is **Step 6** — the final step that produces actionable predictions:

- **Loads trained model** from Step 5 using sidecar metadata
- **Validates feature schema** to ensure compatibility
- **Ranks survivors** by predicted quality
- **Generates prediction pools** (Tight/Balanced/Wide)
- **Produces next-draw predictions** with confidence scores

> **Corrected 2026-11-11.** Accurate at `a15b524`:
> - Step 6 loads the Step 5 model through the sidecar (`prediction_generator.py:290-297`, `:350-353`).
> - It ranks survivors by predicted quality (`:828`, `:836`).
> - It emits per-prediction confidence scores (`:686-713`).
>
> Not accurate:
> - **It does not validate the feature schema hash.** Chapter 6 audit B5: `:350-353` passes no
>   `survivors_file`, so the check at `models/model_factory.py:334` never runs.
> - **It builds one pool, not three** (§6, superseded). That pool is one value per top-ranked survivor,
>   with no vote aggregation (§14.3; gap B3).
>
> Also missing from this list: **a signal-quality gate.** When Step 5 declares the signal unusable,
> Step 6 refuses to predict and exits 3 (§14.2).

### 1.2 Key Features

| Feature | Description |
|---------|-------------|
| **Sidecar-Only Loading** | Model type from meta.json ONLY (never from extension) |
| **Feature Hash Validation** | FATAL error if schema mismatch |
| **Multi-Model Support** | Works with all 4 model types |
| **Prediction Pools** | Tight (20), Balanced (100), Wide (300) |
| **Confidence Scores** | Per-seed and per-prediction quality metrics |

> **Corrected 2026-11-11.**
> - **Sidecar-Only Loading — accurate.** `load_model_from_sidecar` takes `model_type` from the sidecar
>   (`models/model_factory.py:308`) and calls `load_model(model_type, path)` (`:374`, `:98`). The file
>   extension is never read.
> - **Feature Hash Validation — not done** (Chapter 6 audit B5; this audit X1).
> - **Multi-Model Support — five types, not four** (D2): `neural_net`, `xgboost`, `lightgbm`, `catboost`,
>   `random_forest` (`models/model_factory.py:25`, `:54-67`).
> - **Prediction Pools — one pool, not three tiers** (X3, §6).
> - **Confidence Scores — accurate**, one per prediction (`prediction_generator.py:686-701`).

### 1.3 Pipeline Position

```
Step 1: Window Optimizer
    ↓
Step 2.5: Scorer Meta-Optimizer
    ↓
Step 3: Full Scoring → survivors_with_scores.json
    ↓
Step 4: Adaptive Meta-Optimizer
    ↓
Step 5: Anti-Overfit Training → best_model.* + best_model.meta.json
    ↓
Step 6: PREDICTION GENERATOR ◄── THIS FILE
    ↓
Output: ranked_predictions.json, prediction_pools.json
```

> **Corrected 2026-11-11 (DESCRIPTION-ONLY, D3; outputs absorbed into X3).**
> - **Step numbers.** In the executable scheme, Step 0 is `trse_step0.py`, Step 2 is
>   `run_scorer_meta_optimizer.sh` (not "Step 2.5"), and Step 4 is `adaptive_meta_optimizer.py`
>   (`agents/watcher_agent.py:387-395` at `a15b524`).
> - **Upstream outputs — accurate.** Step 3 writes `survivors_with_scores.json`; Step 5 writes `best_model.*`
>   and `best_model.meta.json` (`docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.3).
> - **Step 6's output.** The only output is `predictions/next_draw_prediction.json`, plus a timestamped copy in
>   `predictions/history/` (`prediction_generator.py:905-913`). No code writes `ranked_predictions.json` or
>   `prediction_pools.json` from Step 6. This is already recorded as `docs/PIPELINE_BEHAVIOUR_MODEL.md` D3
>   (governed as system-map Blocker 1) and Chapter 13 audit B1.
> - **What Step 6 does not read.** It does not read Step 3's `survivors_with_scores.json` (§14.1; gap B1).

---

## 2. Architecture Position

### 2.1 In the Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    STEP 6: PREDICTION GENERATOR                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INPUTS:                                                                 │
│  ├── models/reinforcement/best_model.*        (trained model)           │
│  ├── models/reinforcement/best_model.meta.json (sidecar metadata)       │
│  ├── forward_survivors.json                    (from Step 1)            │
│  └── lottery_history.json                      (recent draws)           │
│                                                                          │
│  PROCESSING:                                                             │
│  ├── Load model (type from meta.json ONLY)                              │
│  ├── Validate feature schema hash                                       │
│  ├── Extract features for forward survivors                             │
│  ├── Predict quality scores                                             │
│  ├── Rank by predicted quality                                          │
│  └── Build prediction pools                                             │
│                                                                          │
│  OUTPUTS:                                                                │
│  ├── ranked_predictions.json      (all survivors with predictions)      │
│  ├── prediction_pools.json        (Tight/Balanced/Wide pools)           │
│  └── next_draw_prediction.json    (aggregated next-draw forecast)       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

> **Corrected 2026-11-11.** What WATCHER passes at `a15b524` (`agent_manifests/prediction.json:176-184`;
> the command is built at `agents/watcher_agent.py:2104-2154` at `a15b524`):
>
> ```
> python3 prediction_generator.py --config reinforcement_engine_config.json \
>     --survivors-forward forward_survivors.json --survivors-reverse reverse_survivors.json \
>     --lottery-history <frozen absolute daily3 path> --k 20 --k 20 --models-dir models/reinforcement
> ```
>
> - **Model inputs — accurate.** The model and sidecar come from `models/reinforcement/`.
> - **Survivor inputs — BROKEN (B1).** `forward_survivors.json` and `reverse_survivors.json` have been
>   count-only stubs since S166 (`8ebdf99`, 2026-04-21; today `window_optimizer_integration_final.py:2945-2953`).
>   On that input the default dual-sieve path raises `ValueError` at `survivor_scorer.py:136`. See §14.1 and §15.
> - **History input — DESCRIPTION-ONLY (D4).** The history is `daily3.json` (`prediction.json:180`), not
>   `lottery_history.json`. P0.5 dataset authority rewrites it to the run's frozen absolute path
>   (`agents/watcher_agent.py:1862-1870` at `a15b524`).
> - **Duplicate `--k`.** `--k` appears twice because the manifest's `args_map` maps CLI `k` to the parameter
>   `pool_size` (`prediction.json:32`). `--pool-size` is therefore never sent, and `pool_size` stays at its default
>   20 (`prediction_generator.py:89`).
> - **Processing — "Validate feature schema hash."** Not done (Chapter 6 audit B5).
> - **Outputs.** Only the third output exists (X3).

### 2.2 Data Flow

> **SUPERSEDED (2026-11-11)** — the flow below shows three output files and a separate ranking stage. The as-built
> flow is in **§14.1–§14.4**. Text kept unchanged for history.

```
Step 5 Output                    Step 6 Processing                Step 6 Output
─────────────                    ─────────────────                ────────────
                                        
best_model.pth ─────┐                                            
                    │            ┌─────────────────┐             ranked_predictions.json
best_model.meta.json┼───────────►│  Model Loader   │                    │
                    │            │  (sidecar-only) │             ┌──────┴──────┐
forward_survivors ──┼───────────►├─────────────────┤             │             │
                    │            │ Feature Extract │──────►      │  Ranking    │
lottery_history ────┘            ├─────────────────┤             │             │
                                 │   Predictor     │──────►      └──────┬──────┘
                                 ├─────────────────┤                    │
                                 │   Pool Builder  │──────►      prediction_pools.json
                                 └─────────────────┘                    │
                                                                 next_draw_prediction.json
```

---

## 3. Model Loading with Sidecar

### 3.1 Critical Contract

**Model type is determined ONLY from `best_model.meta.json`. File extensions are NEVER used.**

```python
# CORRECT:
meta = load_json("best_model.meta.json")
model_type = meta["model_type"]  # "xgboost", "neural_net", etc.

# WRONG (FORBIDDEN):
if path.endswith(".pth"):
    model_type = "neural_net"  # NEVER DO THIS
```

> **Verified 2026-11-11 (accurate).** `models/model_factory.py:308` takes `model_type` from the sidecar and
> `:374` loads with it. No code in the Step 6 path branches on a file extension. The sidecar is read before
> anything else (`prediction_generator.py:290-297`).

### 3.2 load_best_model()

> **Corrected 2026-11-11.** This function does not exist at `a15b524`.
>
> - **What runs instead (DESCRIPTION-ONLY, D5).** Step 6 calls
>   `load_model_from_sidecar(models_dir, device, survivors_file=None, strict=True)`
>   (`models/model_factory.py:262`) at `prediction_generator.py:350-353`. That function uses
>   `load_model(model_type, path, device)` (`:98`), not `ModelFactory.load`. It takes `checkpoint_path`
>   as written in the sidecar (absolute on the host), and if that path is missing it tries
>   `models_dir/<basename>` (`:319-330`).
> - **Contracts 1 and 2 — accurate.**
>   - A missing sidecar raises `FileNotFoundError`. Step 6 checks first (`prediction_generator.py:292-294`),
>     and the factory checks again (`models/model_factory.py:291-296`).
>   - A missing `model_type` or `checkpoint_path` raises `ValueError` (`:302-315`).
> - **Contract 3 (schema hash) — not executed.** Step 6 passes no `survivors_file`, so the check gated at
>   `models/model_factory.py:334` is skipped. This is Chapter 6 audit B5, cross-referenced here as X1.
> - **"Fails fast on any validation error" — BROKEN (B2).**
>   - A missing checkpoint raises `FileNotFoundError` in the factory (`:319-330`).
>   - `prediction_generator.py:375-376` catches it and only logs a warning, so `self.model` stays `None`.
>   - Survivors are then ranked by `np.mean(X, axis=1)` (`:827-833`).
>   - The run is reported as `status: success` with `model_type: fallback_mean`, and the process exits 0.
>   - Exit code 2, "No model found" (`:32`), is never returned.
> - **Not in the code below, but real (§14.2):**
>   - The signal-quality gate (`:308-342`).
>   - A `ValueError` when `checkpoint_path` is null but prediction is allowed (`:345-347`).
>   - The NN scaler load (`:361-362`).
>   - A `ValueError` when the sidecar has no feature names and `require_feature_schema` is set (`:366-371`).

```python
from pathlib import Path
import json
from models import ModelFactory
from models.feature_schema import get_feature_schema_with_hash

def load_best_model(models_dir: str, survivors_file: str) -> tuple:
    """
    Load best model from Step 5 output with full validation.
    
    CRITICAL CONTRACTS:
    1. Model type from meta.json ONLY (never from file extension)
    2. Feature schema hash must match between training and runtime
    3. Fails fast on any validation error
    
    Args:
        models_dir: Directory containing best_model.meta.json
        survivors_file: Runtime survivors file for schema validation
    
    Returns:
        model: Loaded ModelInterface
        meta: Full metadata dict
    
    Raises:
        FileNotFoundError: If meta.json missing
        ValueError: If schema hash mismatch
    """
    meta_path = Path(models_dir) / "best_model.meta.json"
    
    # Contract 1: Meta file MUST exist
    if not meta_path.exists():
        raise FileNotFoundError(
            f"FATAL: Missing metadata sidecar: {meta_path}\n"
            f"Step 5 MUST generate best_model.meta.json.\n"
            f"Model type CANNOT be inferred from file extension."
        )
    
    with open(meta_path) as f:
        meta = json.load(f)
    
    # Contract 2: model_type field MUST exist
    if "model_type" not in meta:
        raise ValueError(
            f"FATAL: Missing 'model_type' in {meta_path}\n"
            f"Sidecar schema version may be outdated."
        )
    
    # Contract 3: Validate feature schema hash
    expected_hash = meta["feature_schema"]["feature_schema_hash"]
    runtime_schema = get_feature_schema_with_hash(survivors_file)
    runtime_hash = runtime_schema["feature_schema_hash"]
    
    if runtime_hash != expected_hash:
        raise ValueError(
            f"FATAL: Feature schema mismatch!\n"
            f"  Training hash: {expected_hash}\n"
            f"  Runtime hash:  {runtime_hash}\n"
            f"  Training file: {meta['feature_schema']['source_file']}\n"
            f"  Runtime file:  {survivors_file}\n"
            f"\n"
            f"Feature ordering has changed. Model predictions would be INVALID.\n"
            f"Retrain model or use matching survivors file."
        )
    
    # Load model (type from meta only)
    model_type = meta["model_type"]
    checkpoint_path = Path(models_dir) / meta["checkpoint_path"]
    
    model = ModelFactory.load(model_type, str(checkpoint_path))
    
    print(f"✅ Loaded {model_type} model from {checkpoint_path}")
    print(f"   Feature hash validated: {expected_hash}")
    
    return model, meta
```

### 3.3 Sidecar Metadata Structure

```json
{
  "schema_version": "3.1.2",
  "model_type": "xgboost",
  "checkpoint_path": "best_model.json",
  
  "feature_schema": {
    "source_file": "/absolute/path/to/survivors_with_scores.json",
    "feature_count": 50,
    "feature_names": ["actual_mean", "actual_std", "..."],
    "ordering": "lexicographic_by_key",
    "feature_schema_hash": "a1b2c3d4e5f67890"
  },
  
  "validation_metrics": {
    "mse": 0.0234,
    "mae": 0.1123,
    "r2": 0.847
  },
  
  "training_info": {
    "started_at": "2025-12-20T10:30:00Z",
    "completed_at": "2025-12-20T11:45:00Z",
    "k_folds": 5
  }
}
```

> **Corrected 2026-11-11.** Checked against the host sidecar `models/reinforcement/best_model.meta.json`
> (2026-03-06).
>
> - **Accurate.** It carries `model_type`, `checkpoint_path` and
>   `feature_schema{source_file, feature_count, feature_names, ordering, feature_schema_hash}`.
>   The ordering is `lexicographic_by_key` (`models/feature_schema.py:97-100`).
> - **The example differs (D6):**
>   - `checkpoint_path` is absolute.
>   - `feature_count` is 89, with 4 excluded.
>   - There is no `validation_metrics`; the metrics are under `training_metrics`.
>   - Other top-level keys: `checkpoint_format`, `signal_quality`, `data_context`, `hyperparameters`, `optuna`,
>     `hardware`, `training_info`, `agent_metadata` and `provenance`.
> - **Step 6 reads four of these:**
>   - `signal_quality` (the gate, §14.2);
>   - `agent_metadata.run_id` (lineage, `prediction_generator.py:302-303`);
>   - `feature_schema.per_seed_feature_names`, or `feature_names` (`:366`);
>   - `model_type` / `checkpoint_path` (through the factory).

---

## 4. Feature Schema Validation

### 4.1 Why Validation Matters

```
TRAINING:                           PREDICTION:
  Feature 0: actual_mean              Feature 0: actual_mean     ✅ Match
  Feature 1: actual_std               Feature 1: best_offset     ❌ MISMATCH!
  Feature 2: best_offset              Feature 2: actual_std      ❌ MISMATCH!
  ...                                 ...

Without validation, model receives wrong features in wrong positions.
Predictions would be MEANINGLESS.
```

> **Corrected 2026-11-11.**
> - **The rationale is accurate.** Step 6 places each column by the sidecar's feature names
>   (`prediction_generator.py:366`, `:797-798`).
> - **It does not detect a missing or renamed feature.**
>   - Any sidecar name absent from a survivor's feature dict becomes 0.0 (`:798`).
>   - No hash or name list is compared (Chapter 6 audit B5).
>   - So a drifted schema produces silent zero columns, not a stop.
> - **The dual-sieve path makes this worse.** It recomputes features through `extract_ml_features` (§14.1), which
>   produces zeros for two groups of sidecar names:
>   - The 18 sieve- and metadata-derived names are set to 0.0 placeholders (`survivor_scorer.py:455-461`;
>     Chapter 4 audit D9). Examples: `forward_count`, `intersection_ratio`, `skip_min`.
>   - The 14 `global_*` names are not produced at all, so `prediction_generator.py:798` fills them with 0.0.
>
>   Either way, those columns are 0.0 for every survivor.

### 4.2 Hash Computation

> **Verified 2026-11-11 (accurate in substance).** `models/feature_schema.py:185-188` hashes the comma-joined
> feature names, which `:99` has already sorted lexicographically, and keeps 16 hex characters.
> `validate_feature_schema_hash` sorts explicitly (`:207`). It is imported by Step 6
> (`prediction_generator.py:68`) and never called.

```python
import hashlib

def get_feature_schema_with_hash(survivors_file: str) -> dict:
    """
    Get feature schema with validation hash.
    
    The hash is computed from sorted feature names, ensuring
    any change in feature set or ordering is detected.
    """
    schema = get_feature_schema_from_data(survivors_file)
    
    # Canonical hash from sorted feature names
    names_str = ",".join(sorted(schema["feature_names"]))
    schema["feature_schema_hash"] = hashlib.sha256(
        names_str.encode('utf-8')
    ).hexdigest()[:16]
    
    return schema
```

### 4.3 Validation Flow

> **SUPERSEDED (2026-11-11)** — this flow is not executed at `a15b524`. Chapter 6 audit B5 / this audit X1:
> no hash comparison runs, and no "Feature schema mismatch" error or exit code 1 comes from it. What
> Step 6 actually checks before predicting:
> - the signal gate (§14.2);
> - a non-empty feature-name list, when `require_feature_schema` is set (`prediction_generator.py:366-371`,
>   `:770-775`).
>
> Text kept unchanged for history.

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE SCHEMA VALIDATION                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Load expected_hash from best_model.meta.json                │
│                                                                  │
│  2. Compute runtime_hash from survivors file                    │
│                                                                  │
│  3. Compare:                                                     │
│     ┌─────────────────────────────────────────────┐             │
│     │  expected_hash == runtime_hash?              │             │
│     └─────────────────┬───────────────────────────┘             │
│                       │                                          │
│              ┌────────┴────────┐                                │
│              │ YES             │ NO                              │
│              ▼                 ▼                                 │
│     ┌─────────────┐   ┌─────────────────────────────────┐       │
│     │ ✅ PROCEED  │   │ ❌ FATAL ERROR                   │       │
│     │ Load model  │   │ "Feature schema mismatch!"       │       │
│     └─────────────┘   │ "Model predictions INVALID"      │       │
│                       │ EXIT with error code 1           │       │
│                       └─────────────────────────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Survivor Ranking

### 5.1 Ranking Process

> **Corrected 2026-11-11 (DESCRIPTION-ONLY, D7; feature extraction absorbed into Chapter 6 B5 as X2).**
> - **No `rank_survivors` function.** Ranking is inline in `PredictionGenerator._build_prediction_pool`
>   (`prediction_generator.py:752-862`).
> - **Where the feature rows come from:**
>   - A `dict` survivor supplies `survivor["features"]` (`:782-784`).
>   - An `int` survivor gets `scorer.extract_ml_features(seed, lottery_history, skip=…)`, computed on the spot
>     (`:785-790`).
>   - Every survivor in the default dual-sieve path is an `int` (§14.1).
> - **Missing features become 0.0** (`:798`). The `KeyError` in the code below would not happen.
> - **Sort — accurate in effect.** `predicted_quality = self.model.predict(X)` (`:828`), then
>   `np.argsort(...)[::-1]` (`:836`). There is no tiebreak rule.
> - **No `rank` field** is written to any output.

```python
def rank_survivors(
    model: ModelInterface,
    survivors: List[Dict],
    feature_names: List[str]
) -> List[Dict]:
    """
    Rank survivors by predicted quality.
    
    Args:
        model: Loaded ML model
        survivors: List of survivor dicts with features
        feature_names: Ordered feature names from schema
    
    Returns:
        Survivors sorted by predicted_quality (descending)
    """
    # Extract features in correct order
    X = np.array([
        [s['features'][f] for f in feature_names]
        for s in survivors
    ])
    
    # Predict quality
    predictions = model.predict(X)
    
    # Attach predictions to survivors
    for i, survivor in enumerate(survivors):
        survivor['predicted_quality'] = float(predictions[i])
    
    # Sort by predicted quality (best first)
    ranked = sorted(survivors, key=lambda s: s['predicted_quality'], reverse=True)
    
    # Add rank
    for i, survivor in enumerate(ranked):
        survivor['rank'] = i + 1
    
    return ranked
```

### 5.2 Ranking Output

> **SUPERSEDED (2026-11-11)** — no ranked-survivor record is written at `a15b524`. The output keeps only the
> predicted values, their raw scores and confidences, in rank order. It records no seed, rank, skip or features
> (§14.4). This is Chapter 13 audit B15 Blocker 1 (no producer for a ranked predictions file), cross-referenced
> here as X4. `TB_RULING_S179_IMPLEMENTATION_AUTH.md` §2.5 specifies the record that is wanted. Text kept
> unchanged for history.

```json
[
  {
    "seed": 12345678,
    "rank": 1,
    "predicted_quality": 0.923,
    "features": {...},
    "prng_type": "java_lcg"
  },
  {
    "seed": 87654321,
    "rank": 2,
    "predicted_quality": 0.891,
    "features": {...},
    "prng_type": "java_lcg"
  }
]
```

---

## 6. Prediction Pool Generation

> **SUPERSEDED (2026-11-11)** — §6.1–§6.3 describe a three-tier, weighted-vote pool builder that was never in the
> code (`git log --all -S"build_prediction_pools"` finds documentation commits only). **The as-built pool is
> described in §14.3.**
>
> - **The three tiers and the pool file** are absorbed into `docs/PIPELINE_BEHAVIOUR_MODEL.md` D3 and Chapter 13
>   audit B1 (X3). The 20/100/300 tiers exist only in the separate backtest CLIs `build_pools.py` /
>   `evaluate_pools.py`.
> - **The weighted vote** is gap **B3**, governed by `TB_RULING_S179_IMPLEMENTATION_AUTH.md` §2.5.
> - **The hit-rate figures** are unverified (U1).
>
> Text kept unchanged for history.

### 6.1 Pool Definitions

| Pool | Size | Purpose | Expected Hit Rate |
|------|------|---------|-------------------|
| **Tight** | 20 | Highest confidence, maximum precision | Highest lift vs random |
| **Balanced** | 100 | Broader coverage, >60% weight | Hit@100 > 70% |
| **Wide** | 300 | Comprehensive coverage, >85% weight | Hit@300 > 90% |

> **Corrected 2026-11-11.**
> - **Pool size.** There is one pool, `pool_size` (default 20; `prediction_generator.py:89`). The manifest
>   bounds it at 5–100 (`agent_manifests/prediction.json` `parameter_bounds.pool_size`).
> - **Expected Hit Rate column — UNVERIFIED (U1).** No artifact measures these numbers, and no Step 6 output
>   has been graded on the host (Chapter 13 audit A.7). Beta records that Hit@100/300 cannot be derived from a
>   20-entry pool (`TB_RULING_S177_KPI_GOVERNANCE.md:325`).

### 6.2 build_prediction_pools()

```python
def build_prediction_pools(
    ranked_survivors: List[Dict],
    lottery_history: List[int],
    prng_type: str = 'java_lcg',
    mod: int = 1000
) -> Dict:
    """
    Build prediction pools from ranked survivors.
    
    Each pool contains:
    - Top-K survivors by predicted quality
    - Their next-draw predictions
    - Aggregated vote weights for each possible number
    """
    from survivor_scorer import SurvivorScorer
    
    scorer = SurvivorScorer(prng_type=prng_type, mod=mod)
    
    pools = {
        'tight': {'size': 20, 'survivors': [], 'predictions': {}},
        'balanced': {'size': 100, 'survivors': [], 'predictions': {}},
        'wide': {'size': 300, 'survivors': [], 'predictions': {}}
    }
    
    for pool_name, pool in pools.items():
        pool_survivors = ranked_survivors[:pool['size']]
        pool['survivors'] = pool_survivors
        
        # Aggregate predictions with weighted voting
        votes = {}  # number -> total weight
        
        for survivor in pool_survivors:
            seed = survivor['seed']
            weight = survivor['predicted_quality']
            
            # Get next prediction for this seed
            next_pred = scorer.predict_next(seed, lottery_history)
            
            if next_pred not in votes:
                votes[next_pred] = 0.0
            votes[next_pred] += weight
        
        # Normalize weights
        total_weight = sum(votes.values())
        pool['predictions'] = {
            num: weight / total_weight
            for num, weight in sorted(votes.items(), key=lambda x: -x[1])
        }
        
        # Summary stats
        pool['total_weight'] = total_weight
        pool['unique_predictions'] = len(votes)
        pool['top_prediction'] = max(votes.items(), key=lambda x: x[1])
    
    return pools
```

> **Corrected 2026-11-11.**
> - **BROKEN — B3, governed by S179 §2.5.** No per-value vote sum, normalisation, unique count or
>   `top_prediction` exists. Each of the first `pool_size` ranked survivors appends one value
>   (`prediction_generator.py:841-850`), and the list is cut to `k` (`:653-664`). Repeated values stay as
>   separate entries. On the host, 4 of 19 history files hold a repeat among 20.
> - **DESCRIPTION-ONLY — D8.** `scorer.predict_next(seed, history)` does not exist. The value is
>   `scorer._generate_sequence(seed, len(history)+1, skip)[len(history)]` (`:839-849`;
>   `survivor_scorer.py:119-125`).
> - **Accurate.** `SurvivorScorer(prng_type, mod)` is built at `:263-266`, and its defaults `java_lcg` / 1000 are
>   at `:107-108`.

### 6.3 Pool Output Format

```json
{
  "tight": {
    "size": 20,
    "survivors": [...],
    "predictions": {
      "347": 0.15,
      "891": 0.12,
      "234": 0.09
    },
    "total_weight": 18.45,
    "unique_predictions": 15,
    "top_prediction": ["347", 0.15]
  },
  "balanced": {
    "size": 100,
    "survivors": [...],
    "predictions": {...},
    "total_weight": 85.23
  },
  "wide": {
    "size": 300,
    "survivors": [...],
    "predictions": {...},
    "total_weight": 245.67
  }
}
```

---

## 7. Next-Draw Prediction

> **SUPERSEDED (2026-11-11)** — `generate_next_draw_prediction` and the cross-pool weights 0.5/0.3/0.2 do not
> exist. That is absorbed into B3 (X6). The written shape in §7.2 is wrong (D9). **The as-built output is §14.4.**
> Text kept unchanged for history.

### 7.1 Weighted Voting

```python
def generate_next_draw_prediction(
    pools: Dict,
    pool_weights: Dict[str, float] = None
) -> Dict:
    """
    Generate aggregated next-draw prediction from all pools.
    
    Args:
        pools: Output from build_prediction_pools()
        pool_weights: Relative weights for each pool
                     Default: {'tight': 0.5, 'balanced': 0.3, 'wide': 0.2}
    
    Returns:
        Next-draw prediction with confidence metrics
    """
    if pool_weights is None:
        pool_weights = {'tight': 0.5, 'balanced': 0.3, 'wide': 0.2}
    
    # Aggregate votes across pools
    combined_votes = {}
    
    for pool_name, pool in pools.items():
        pool_weight = pool_weights.get(pool_name, 0.0)
        
        for number, vote_weight in pool['predictions'].items():
            if number not in combined_votes:
                combined_votes[number] = 0.0
            combined_votes[number] += vote_weight * pool_weight
    
    # Sort by combined weight
    ranked_predictions = sorted(
        combined_votes.items(),
        key=lambda x: -x[1]
    )
    
    # Build output
    top_predictions = ranked_predictions[:20]
    total_weight = sum(v for _, v in combined_votes.items())
    
    return {
        'timestamp': datetime.now().isoformat(),
        'top_predictions': [
            {
                'number': num,
                'weight': weight,
                'confidence': weight / total_weight * 100
            }
            for num, weight in top_predictions
        ],
        'primary_prediction': {
            'number': ranked_predictions[0][0],
            'confidence': ranked_predictions[0][1] / total_weight * 100
        },
        'coverage': {
            'top_10_weight_pct': sum(v for _, v in ranked_predictions[:10]) / total_weight * 100,
            'top_50_weight_pct': sum(v for _, v in ranked_predictions[:50]) / total_weight * 100,
            'unique_numbers': len(combined_votes)
        },
        'pool_contributions': {
            pool_name: {
                'weight': pool_weights.get(pool_name, 0.0),
                'top_prediction': pool['top_prediction']
            }
            for pool_name, pool in pools.items()
        }
    }
```

### 7.2 Prediction Output

```json
{
  "timestamp": "2025-12-31T14:30:00",
  "top_predictions": [
    {"number": "347", "weight": 0.18, "confidence": 18.2},
    {"number": "891", "weight": 0.12, "confidence": 12.1},
    {"number": "234", "weight": 0.09, "confidence": 9.3}
  ],
  "primary_prediction": {
    "number": "347",
    "confidence": 18.2
  },
  "coverage": {
    "top_10_weight_pct": 65.3,
    "top_50_weight_pct": 89.2,
    "unique_numbers": 147
  },
  "pool_contributions": {
    "tight": {"weight": 0.5, "top_prediction": ["347", 0.15]},
    "balanced": {"weight": 0.3, "top_prediction": ["347", 0.11]},
    "wide": {"weight": 0.2, "top_prediction": ["891", 0.08]}
  }
}
```

> **Corrected 2026-11-11 (DESCRIPTION-ONLY, D9).** The real `predictions/next_draw_prediction.json` has these
> top-level keys:
>
> - `status`
> - `predictions` — a list of ints
> - `raw_scores`
> - `confidence_scores`
> - `confidence_scores_normalized`
> - `metadata`
> - `agent_metadata`
>
> These are built at `prediction_generator.py:709-733` and `:929-942`, and the host file of 2026-03-07 agrees.
> There is no `primary_prediction`, `coverage` or `pool_contributions`.

---

## 8. Output Formats

> **SUPERSEDED (2026-11-11)** — Step 6 writes one canonical file, `predictions/next_draw_prediction.json`
> (1,483 B on the host, not ~10 KB). It also writes a non-contractual copy,
> `predictions/history/predictions_<YYYYMMDD_HHMMSS>.json` (`prediction_generator.py:905-913`, `:944-952`).
> Neither `ranked_predictions.json` nor `prediction_pools.json` is written by Step 6 (X3, X4).
> `agent_manifests/definitions.json` (gitignored, read by no code) names a third set:
> `predictions/next_draw_prediction.json`, `predictions/prediction_pools.json`, `predictions/ranked_survivors.json`
> (`:80-84`). **Only the first exists on the host.** The `success_metrics` in §8.3 are U1. Text kept unchanged
> for history.

### 8.1 File Outputs

| File | Contents | Size |
|------|----------|------|
| `ranked_predictions.json` | All survivors with predicted quality | ~50MB |
| `prediction_pools.json` | Tight/Balanced/Wide pools | ~5MB |
| `next_draw_prediction.json` | Aggregated prediction | ~10KB |

### 8.2 ranked_predictions.json

```json
{
  "metadata": {
    "generated_at": "2025-12-31T14:30:00Z",
    "model_type": "xgboost",
    "model_r2": 0.847,
    "total_survivors": 395211,
    "feature_schema_hash": "a1b2c3d4e5f67890"
  },
  "rankings": [
    {
      "rank": 1,
      "seed": 12345678,
      "predicted_quality": 0.923,
      "prng_type": "java_lcg",
      "next_prediction": 347
    }
  ]
}
```

### 8.3 prediction_pools.json

```json
{
  "metadata": {
    "generated_at": "2025-12-31T14:30:00Z",
    "lottery_history_length": 5000
  },
  "pools": {
    "tight": {...},
    "balanced": {...},
    "wide": {...}
  },
  "success_metrics": {
    "expected_hit_at_20": 0.15,
    "expected_hit_at_100": 0.70,
    "expected_hit_at_300": 0.90,
    "lift_vs_random": 15.2
  }
}
```

---

## 9. CLI Interface

### 9.1 Arguments

```bash
python3 prediction_generator.py [options]

Required:
  --models-dir DIR           Directory with best_model.* and meta.json
  --survivors-forward FILE   Forward survivors from Step 1
  --lottery-history FILE     Recent lottery draws

Optional:
  --output-dir DIR           Output directory (default: predictions/)
  --pool-sizes SIZES         Pool sizes as "tight,balanced,wide" (default: 20,100,300)
  --prng-type TYPE           PRNG algorithm (default: java_lcg)
  --mod MOD                  Modulo value (default: 1000)
  -h, --help                 Show help
```

> **Corrected 2026-11-11 (DESCRIPTION-ONLY, D10; `--output-dir` is SWITCHED-OFF-BY-RULING, S1).**
>
> **The real flags** (`prediction_generator.py:965-986`):
>
> | Flag | Default / behaviour |
> |---|---|
> | `--config` | `prediction_generator_config.json` |
> | `--survivors-forward` | required to predict |
> | `--survivors-reverse` | optional |
> | `--lottery-history` | required to predict |
> | `--models-dir` | `models/reinforcement` |
> | `--parent-run-id` | optional |
> | `--no-require-schema` | — |
> | `--k` | number of predictions kept |
> | `--pool-size` | number of top survivors that vote |
> | `--output-dir` | accepted but ignored (S1) |
> | `--log-level` | — |
> | `--test` | — |
> | `--dry-run` | — |
>
> `--survivors-forward` and `--lottery-history` are required for a prediction run (`:1009-1010`, accurate).
>
> **Flags the chapter lists that do not exist:** `--pool-sizes`, `--prng-type`, `--mod` and `--verbose`.
> - PRNG type and mod come from the `--config` file's `prng` section. Failing that, they come from
>   `optimal_window_config.json`, and failing that from `java_lcg` / 1000 (`:150-165`).
> - Use `--log-level DEBUG` for verbose output.
>
> **Why `--output-dir` is ignored.** It sets `config.predictions_dir` (`:1022-1023`), which no code reads. The
> output path is fixed to `predictions/` (`:905-907`) under the Team Beta ruling of 2026-01-28 cited at `:902`
> and in commit `c6e3989`. The ruling text was not found as a file in `docs/`.

### 9.2 Usage Examples

> **Corrected 2026-11-11 (absorbed into D10, S1 and B1 — X8).**
> - The "Custom pool sizes" and "Different PRNG" examples would fail at argparse, because `--pool-sizes` and
>   `--prng-type` do not exist.
> - Every example passes `forward_survivors.json`. Since S166 that file is a count-only stub, and Step 6 cannot
>   consume it (B1, §15.1).

**Basic usage:**
```bash
python3 prediction_generator.py \
    --models-dir models/reinforcement \
    --survivors-forward forward_survivors.json \
    --lottery-history daily3.json
```

**Custom pool sizes:**
```bash
python3 prediction_generator.py \
    --models-dir models/reinforcement \
    --survivors-forward forward_survivors.json \
    --lottery-history daily3.json \
    --pool-sizes 10,50,200 \
    --output-dir my_predictions/
```

**Different PRNG:**
```bash
python3 prediction_generator.py \
    --models-dir models/reinforcement \
    --survivors-forward forward_survivors.json \
    --lottery-history daily3.json \
    --prng-type xorshift32
```

---

## 10. Agent Integration

### 10.1 Agent Manifest

> **Corrected 2026-11-11 (DESCRIPTION-ONLY, D11).** The live `agent_manifests/prediction.json` is version
> 1.5.0. Its top-level keys are `agent_name`, `description`, `pipeline_step: 6`, `version` and
> `required_inputs`. It also has `primary_output`, `outputs`, `actions[0]` (with `args_map`, and
> `distributed: false`), `follow_up_agents: []`, `success_condition`, `retry: 2`, `parameter_bounds`,
> `default_params`, and `evaluation_type: "file_exists"`.
>
> - **`required_inputs` (`:6-11`)** lists four files: `models/reinforcement/best_model.meta.json`,
>   `survivors_with_scores.json`, `forward_survivors.json` and `optimal_window_config.json`. WATCHER uses this list
>   only for its existence and freshness check (`agents/watcher_agent.py:803-834` at `a15b524`).
> - **`survivors_with_scores.json` is never opened by Step 6** (B1).
> - **Accurate:** there is no follow-up agent.

**agent_manifests/prediction.json:**
```json
{
  "agent_id": "prediction_agent",
  "step_id": 6,
  "step_name": "prediction_generation",
  "script": "prediction_generator.py",
  
  "inputs": [
    {"name": "models_dir", "type": "directory", "required": true},
    {"name": "survivors_forward", "type": "file", "required": true},
    {"name": "lottery_history", "type": "file", "required": true}
  ],
  
  "outputs": [
    {"name": "ranked_predictions.json", "type": "file"},
    {"name": "prediction_pools.json", "type": "file"},
    {"name": "next_draw_prediction.json", "type": "file"}
  ],
  
  "success_criteria": {
    "min_survivors_ranked": 1000,
    "min_pool_coverage": 0.8
  },
  
  "follow_up_agent": null,
  
  "llm_prompts": {
    "analyze": "Analyze these prediction results and identify confidence patterns..."
  }
}
```

### 10.2 Watcher Agent Integration

> **SUPERSEDED (2026-11-11)** — `handle_step6_completion` does not exist (`/bin/grep -n handle_step6` on
> `agents/watcher_agent.py` at `a15b524` → none). This is absorbed into Chapter 12 audit B1 (X5). What WATCHER does
> at `a15b524`:
>
> 1. **Before running.** It checks the four `required_inputs` exist and that the output is fresh
>    (`:803-834`).
> 2. **After running.** It evaluates the step by the file-exists branch for every manifest (`:946`).
> 3. **Results lookup.** It looks for Step 6 "results" in `prediction_pool.json` (`:2223`), which Step 6 never
>    writes.
> 4. **Completion.** It declares the pipeline complete (`:1390-1393`).
>
> No confidence threshold, and no LLM query on the prediction, exists.
>
> **Repaired at `b6cb789` (failure authority).** A non-zero exit now escalates as `step_failed`
> (`docs/CLAUDE_CODE_REPORT_WATCHER_SAFETY_REPAIRS.md`). That includes Step 6's exit 3, PREDICTION_SKIPPED: it is
> escalated like any other failure, not handled as a distinct state. Text kept unchanged for history.

```python
# In agents/watcher_agent.py

def handle_step6_completion(self, result_dir: str) -> AgentDecision:
    """Evaluate prediction generation results."""
    
    # Load prediction output
    pred_path = Path(result_dir) / 'next_draw_prediction.json'
    with open(pred_path) as f:
        prediction = json.load(f)
    
    # Check confidence
    top_confidence = prediction['primary_prediction']['confidence']
    
    if top_confidence < 5.0:
        return AgentDecision(
            decision='ESCALATE',
            confidence=0.8,
            reasoning=f'Low prediction confidence: {top_confidence:.1f}%'
        )
    
    # Query Math LLM for interpretation
    analysis = self.llm.calculate(f"""
        Analyze this prediction distribution:
        - Top prediction: {prediction['primary_prediction']}
        - Coverage: {prediction['coverage']}
        
        Is this a healthy prediction distribution?
    """)
    
    return AgentDecision(
        decision='PROCEED',
        confidence=0.9,
        reasoning=f'Prediction generated with {top_confidence:.1f}% top confidence'
    )
```

### 10.3 Agent Metadata Injection

```python
def generate_predictions_with_metadata(
    model, meta, survivors, lottery_history, output_dir
):
    """Generate predictions with agent metadata."""
    
    # Generate predictions
    ranked = rank_survivors(model, survivors, meta['feature_schema']['feature_names'])
    pools = build_prediction_pools(ranked, lottery_history)
    prediction = generate_next_draw_prediction(pools)
    
    # Inject agent metadata
    prediction['agent_metadata'] = {
        'pipeline_step': 6,
        'pipeline_step_name': 'prediction_generation',
        'run_id': f"step6_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'parent_run_id': meta.get('agent_metadata', {}).get('run_id'),
        'model_type': meta['model_type'],
        'model_r2': meta['validation_metrics']['r2'],
        'follow_up_agent': None  # Terminal step
    }
    
    # Save outputs
    save_json(prediction, output_dir / 'next_draw_prediction.json')
    
    return prediction
```

> **Corrected 2026-11-11.**
> - **Accurate.** Lineage and terminal-step handling are as described. `parent_run_id` comes from the Step 5
>   sidecar's `agent_metadata.run_id` (`prediction_generator.py:302-303`), or from `--parent-run-id`
>   (`:1063-1064`). `pipeline_step=6` and `follow_up_agent=None` are passed to `inject_agent_metadata`
>   (`:929-942`), which writes `follow_up_agent: null` for Step 6 (`integration/metadata_writer.py:156-158`).
> - **DESCRIPTION-ONLY (D12).** The fields differ from the code below:
>   - `pipeline_step_name` is `"prediction"` (`integration/metadata_writer.py:47`).
>   - There is no `run_id`, `model_type` or `model_r2`.
>   - There are `inputs`, `outputs`, `confidence`, `reasoning` and `timestamp_injected`.
>   - `inputs` always names `survivors_with_scores.json` (`prediction_generator.py:933`), although Step 6 never
>     reads it (B1).

---

## 11. Troubleshooting

### 11.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Missing metadata sidecar" | Step 5 didn't generate meta.json | Re-run Step 5 with latest code |
| "Feature schema mismatch" | Different survivors file | Use same file as training |
| "Model type missing" | Old sidecar format | Re-run Step 5 |
| "Empty predictions" | No survivors passed | Check input file |
| "Low confidence" | Model not trained well | Check Step 5 metrics |

> **Corrected 2026-11-11 (DESCRIPTION-ONLY, D13; row 2 absorbed into Chapter 6 B5).**
>
> | Row | What the code does at `a15b524` |
> |---|---|
> | Missing sidecar | Step 6 raises its own `Required sidecar missing: …` (`prediction_generator.py:294`) before the factory's "Missing metadata sidecar" message can appear. |
> | Feature schema mismatch | Never raised (Chapter 6 audit B5). |
> | Model type missing | Accurate (`models/model_factory.py:302-306`). |
> | Empty predictions | Step 6 returns without writing any file (`prediction_generator.py:631-640`), so the previous `next_draw_prediction.json` stays in place. Today's most likely cause is the stub survivor files (B1). |
>
> **Rows the table is missing:**
>
> | Symptom | What happens |
> |---|---|
> | `ValueError: invalid literal for int()` | Raised from `survivor_scorer.py:136` when the stub survivor files are fed to the dual-sieve path (B1). |
> | PREDICTION_SKIPPED, exit 3 | Step 5 declared the signal unusable (§14.2). |
> | `model_type: fallback_mean` in the output | The checkpoint was missing, and the run continued without a model (B2). |

### 11.2 Debug Mode

```bash
# Verbose output
python3 prediction_generator.py \
    --models-dir models/reinforcement \
    --survivors-forward forward_survivors.json \
    --lottery-history daily3.json \
    --verbose \
    2>&1 | tee prediction_debug.log
```

### 11.3 Verify Model Loading

> **Corrected 2026-11-11.**
> - **§11.2.** `--verbose` does not exist; use `--log-level DEBUG` (D10).
> - **§11.3 (DESCRIPTION-ONLY, D14).** The training hash was computed from Step 3's `survivors_with_scores.json`
>   (the sidecar's `feature_schema.source_file`), so that is the file to check against. `forward_survivors.json` is
>   a count-only stub with no feature record to read (`models/feature_schema.py:45-78`).

```bash
# Check sidecar contents
cat models/reinforcement/best_model.meta.json | jq .

# Verify feature hash
python3 -c "
from models.feature_schema import get_feature_schema_with_hash
schema = get_feature_schema_with_hash('forward_survivors.json')
print(f'Runtime hash: {schema[\"feature_schema_hash\"]}')
"
```

### 11.4 Validate Predictions

```bash
# Check prediction output
cat predictions/next_draw_prediction.json | jq '.primary_prediction'

# Verify pool sizes
cat predictions/prediction_pools.json | jq '.pools | keys'
```

> **Corrected 2026-11-11 (absorbed into D9 and X3 — X7).** `.primary_prediction` does not exist, and
> `predictions/prediction_pools.json` is not written. Working equivalents:
> `jq '{status, predictions, confidence_scores}' predictions/next_draw_prediction.json` and
> `jq '.metadata | {method, pool_size, k, intersection_count, model_type}' predictions/next_draw_prediction.json`.

---

## 12. Complete Method Reference

> **Corrected 2026-11-11 (DESCRIPTION-ONLY, D15).** Most names below do not exist. The real symbols at `a15b524`:
>
> **`prediction_generator.py`:**
>
> | Symbol | Line |
> |---|---|
> | `PredictionConfig` | `:81` |
> | `PredictionConfig.from_json` | `:142` |
> | `PredictionGenerator.__init__` | `:258` (sidecar, gate, model load) |
> | `_load_nn_scaler` | `:385` |
> | `_check_signal_gate` | `:465` |
> | `_get_survivor_skip` | `:530` |
> | `generate_predictions` | `:559` |
> | `_empty_pool_result` | `:740` |
> | `_build_prediction_pool` | `:752` (rank and vote) |
> | `extract_features_batch` | `:864` (no caller) |
> | `_save_predictions` | `:896` |
> | `main` | `:958` |
>
> **Model loading — `models/model_factory.py`:** `load_model_from_sidecar` (`:262`), `load_model` (`:98`).
>
> **Feature schema — `models/feature_schema.py`:**
> - `get_feature_schema_with_hash` (`:162`) exists.
> - `validate_feature_schema_hash` (`:193`) exists. Step 6 imports it and never calls it.
>
> **Sequence and intersection — `survivor_scorer.py`:** `_generate_sequence` (`:119`),
> `compute_dual_sieve_intersection` (`:139`).

### 12.1 Model Loading

| Function | Purpose |
|----------|---------|
| `load_best_model(models_dir, survivors_file)` | Load model with validation |
| `validate_feature_schema_hash(expected, runtime)` | Check schema compatibility |
| `get_feature_schema_with_hash(file)` | Compute schema hash |

### 12.2 Ranking

| Function | Purpose |
|----------|---------|
| `rank_survivors(model, survivors, features)` | Rank by predicted quality |
| `extract_features_ordered(survivor, feature_names)` | Extract in correct order |

### 12.3 Pool Generation

| Function | Purpose |
|----------|---------|
| `build_prediction_pools(ranked, history)` | Build Tight/Balanced/Wide pools |
| `aggregate_votes(survivors, scorer)` | Weighted vote aggregation |
| `generate_next_draw_prediction(pools)` | Combined prediction |

### 12.4 Output

| Function | Purpose |
|----------|---------|
| `save_ranked_predictions(ranked, path)` | Save full rankings |
| `save_prediction_pools(pools, path)` | Save pool details |
| `save_next_draw(prediction, path)` | Save aggregated prediction |

---

## 13. Chapter Summary

**Chapter 16: Prediction Generator (Step 6)** covers the final pipeline step:

| Component | Lines | Purpose |
|-----------|-------|---------|
| Model loading | ~80 | Sidecar-only loading with validation |
| Feature validation | ~40 | Hash-based schema verification |
| Survivor ranking | ~60 | ML-based quality prediction |
| Pool generation | ~100 | Tight/Balanced/Wide pools |
| Next-draw prediction | ~60 | Weighted vote aggregation |

**Key Points:**
- Model type from meta.json ONLY (never from extension)
- Feature schema hash MUST match (FATAL if mismatch)
- Three prediction pools: Tight (20), Balanced (100), Wide (300)
- Weighted voting aggregates predictions
- Terminal step (no follow-up agent)

**Success Metrics:**
- Hit@100 > 70%
- Hit@300 > 90%
- Lift vs random: 10-20x

> **Corrected 2026-11-11.**
> - **Self-name (DESCRIPTION-ONLY, D16).** This is Chapter 7, not "Chapter 16".
> - **Line counts (D1).** The component line counts are not checked against the code; the module has 1,095 lines.
> - **Key points.**
>   - Point 1 is accurate.
>   - Point 2 is not executed (Chapter 6 B5).
>   - Point 3 is X3.
>   - Point 4 is B3: there is no weighted vote.
>   - Point 5 is accurate (`agent_manifests/prediction.json:39`; `agents/watcher_agent.py:1390-1393` at `a15b524`).
> - **Success Metrics — UNVERIFIED (U1).** No conclusion is drawn about them here.

---

## Pipeline Complete

With Chapter 16, all 6 pipeline steps are now documented:

| Step | Chapter | Script |
|------|---------|--------|
| 1 | Chapter 1 | `window_optimizer.py` |
| 2.5 | Chapter 13 | `scorer_trial_worker.py` |
| 3 | Chapter 6, 8 | `survivor_scorer.py`, `full_scoring_worker.py` |
| 4 | Chapter 4 | `meta_prediction_optimizer.py` |
| 5 | Chapter 4, 10 | `meta_prediction_optimizer_anti_overfit.py` |
| **6** | **Chapter 16** | **`prediction_generator.py`** |

> **Corrected 2026-11-11 (DESCRIPTION-ONLY, D16).** The executable steps at `a15b524`
> (`agents/watcher_agent.py:387-395` at `a15b524`), with the chapter that documents each:
>
> | Step | Chapter | Script |
> |---|---|---|
> | 0 | — | `trse_step0.py` |
> | 1 | 1; the sieve inside it is Chapter 2 | `window_optimizer.py` |
> | 2 | 3 | `run_scorer_meta_optimizer.sh` |
> | 3 | 4 | `run_step3_full_scoring.sh` |
> | 4 | 5 | `adaptive_meta_optimizer.py` |
> | 5 | 6 | `meta_prediction_optimizer_anti_overfit.py` |
> | 6 | **7** | `prediction_generator.py` |
>
> The live feedback loop (Chapter 13) is not a WATCHER step.

---

## 14. What Step 6 Does at a15b524

*(Added 2026-11-11. Every anchor below was opened in this pass. Line numbers are offered for convenience; the
symbol names are the durable reference.)*

### 14.1 Inputs

- **Model.** The sidecar `models/reinforcement/best_model.meta.json` is read first (`PredictionGenerator.__init__`,
  `prediction_generator.py:290-297`). The model is then loaded by `load_model_from_sidecar`
  (`models/model_factory.py:262`), which takes its type from the sidecar only.
- **Survivors.** `--survivors-forward` and, optionally, `--survivors-reverse`.
  - `main` accepts a JSON list, or an object with a `survivors` key (`:1042-1050`).
  - In dual-sieve mode (`use_dual_sieve` defaults True, `:112`; WATCHER always sends a reverse file,
    `agent_manifests/prediction.json:179`), the working set is `SurvivorScorer.compute_dual_sieve_intersection`
    (`:606-616`).
  - That function coerces both lists to integer seeds (`survivor_scorer.py:128-137`) and returns them sorted
    (`:172`). **So every survivor that reaches the pool in this mode is a bare integer.**
  - **Today both files are count-only stubs** (§15.1), and this step raises `ValueError` (B1).
- **History.** `--lottery-history`: a list of ints, `{"draws": …}`, or `[{"draw": …}]` (`:1052-1056`). WATCHER sends
  the frozen absolute dataset path (§15.2).
- **Config.** `--config`; WATCHER sends `reinforcement_engine_config.json`.
  - `PredictionConfig.from_json` reads its `prediction` section, or the whole file if there is none (`:147`).
  - PRNG type and mod fall back to `optimal_window_config.json`, then to `java_lcg` / 1000 (`:150-165`).
  - If the file cannot be read, the defaults are used (`:1013-1016`).
- **Not read.** Step 3's `survivors_with_scores.json` is never opened, though the manifest requires it and
  `agent_metadata.inputs` names it (`:933`) (B1). The canonical NPZ is not read either.

### 14.2 Gates, in order

1. **Sidecar must exist.** Otherwise `FileNotFoundError`, exit 1 (`:292-294`).
2. **Signal-quality gate, before any model load** (`:308-342`).
   - If `signal_quality.prediction_allowed` is false, no model is loaded, and a `PREDICTION_SKIPPED` record
     (reason, `signal_quality`, `data_context`, `recovery_authority: WATCHER_AGENT`) is written to the canonical
     path (`:580-587`).
   - The process exits **3** (`:1074-1084`).
   - Design rule, quoted in code: *"Learning steps declare signal quality; execution steps act only on declared
     usable signals; control agents decide recovery"* (`:469-471`).
   - A second check, `_check_signal_gate` (`:465-524`), repeats the gate in `generate_predictions` (`:593-597`).
   - Host sidecar: `signal_status "weak"`, `prediction_allowed true`.
3. **Null checkpoint.** A null `checkpoint_path` while prediction is allowed raises `ValueError` (`:345-347`).
4. **Missing checkpoint file.** This does **not** stop the run. Ranking continues on the feature-row mean (B2).
5. **Feature names.** The sidecar must carry them when `require_feature_schema` is set (`:366-371`, `:770-775`).
   **No hash or name-list comparison** runs against the runtime data (Chapter 6 audit B5).

### 14.3 How the top candidates vote values into the pool

This is the mechanism the TFM target — a pool hit rate — is measured on.

1. **Feature rows.** One row per survivor, in the sidecar's feature-name order (`:797-798`).
   - A dict survivor supplies `features`.
   - An integer survivor's features are recomputed against the whole history (`:785-790`;
     `survivor_scorer.py:350-365`).
   - Any name not produced is 0.0.
   - For `neural_net` only, the checkpoint's stored scaler is applied (`:814-824`).
2. **Rank.** The model's prediction is the survivor's relevance: `predicted_quality = self.model.predict(X)`
   (`:828`). Survivors are ordered by `np.argsort(predicted_quality)[::-1]` (`:836`), with no tiebreak rule.
3. **Vote.** Each of the first `pool_size` ranked survivors (`:841`, default 20) casts **one** vote: the next value
   its own generator emits.
   - The seed's sequence is regenerated to length `len(history)+1`
     (`scorer._generate_sequence(seed, next_idx + 1, skip=survivor_skip)`, `:846`), reduced by `mod`
     (`survivor_scorer.py:119-125`).
   - The value at index `len(history)` is read off (`:839`, `:849`).
   - The vote carries the survivor's raw quality as its score (`:850`).
   - **Skip.** The survivor's own skip is used only when the survivor arrives as a dict (`_get_survivor_skip`,
     `:530-553`). In the dual-sieve path every survivor is an integer, so every vote uses `default_skip`
     (0 unless configured, `:109`). Host: 19 of 19 recorded runs were `dual_sieve` with `default_skip 0`.
4. **Pool.** The votes are kept as a list in rank order, **one entry per survivor, with no aggregation**.
   - There is no per-value tally or weighting, and repeated values stay as separate entries (`:848-850`).
   - The list is then cut to the first `k` (`:653-664`).
   - Each kept entry's confidence is a sigmoid of its raw score's z-score within the kept set, or 0.5 for all when
     the scores are constant (`:686-701`).
   - Host: 4 of 19 history files contain a repeated value among their 20 entries.
5. **What `pool_size` and `k` are under WATCHER.** The manifest maps CLI `--k` to the parameter `pool_size`, so
   WATCHER sends `--k 20` twice and never `--pool-size` (`agent_manifests/prediction.json:32`, `:181-182`;
   `agents/watcher_agent.py:2104-2154` at `a15b524`). The voting set therefore stays at the default 20, and 20 are
   kept.

`TB_RULING_S179_IMPLEMENTATION_AUTH.md` §2.5 records that the generator "ranks predictions and then truncates to
`pool_size`". It approves, but does not implement at `a15b524`, a ranked unique artifact with:
- deterministic ordering (quality descending, then stable survivor id, then source index);
- duplicate collapse (the first occurrence of an outcome takes the `unique_rank`);
- persisted `raw_rank` / `unique_rank` / survivor id / `duplicate-of` fields.

### 14.4 Output

`predictions/next_draw_prediction.json` (`:905-907`, written at `:944-947`) holds:

| Key | Contents |
|---|---|
| `status` | `"success"` |
| `predictions` | list of int |
| `raw_scores` | list |
| `confidence_scores` | list |
| `confidence_scores_normalized` | list |
| `metadata` | `method`, `pool_size`, `k`, forward / reverse / intersection counts, `prng_type`, `mod`, `default_skip`, `model_type`, `feature_count`, `score_stats`, timestamp, version (`:709-733`) |
| `agent_metadata` | `inject_agent_metadata`, `:929-942` |

- The same content is archived to `predictions/history/predictions_<timestamp>.json` (`:911-913`, `:950-952`).
- **No seed, rank, skip or feature value is written.**
- The output path ignores `--output-dir` (S1).
- An empty pool returns without writing (`:631-640`).
- Exit codes: 0 success; 1 uncaught exception; 3 signal gate. Exit 2 is documented (`:32`) and never returned (B2).

**Host state (point-in-time, read-only, 2026-11-11).**
- `predictions/next_draw_prediction.json` is dated 2026-03-07. It holds 2 predictions from a 2-seed intersection
  (`forward_count 293`, `reverse_count 173`), from a `neural_net` model with 89 features.
- `predictions/history/` holds 19 files, dated 2026-02-08 to 2026-03-07. All 19 are `dual_sieve`, `k 20`,
  `pool_size 20`.
- **None postdates 2026-04-21**, when the survivor files became stubs.

### 14.5 Downstream

- **Chapter 13** grades `prediction_pool.json`, not this file (Chapter 13 audit B1).
- **WATCHER.** At `a15b524` it validates Step 6 by file existence (Chapter 12 audit B1; repaired at `b6cb789` for
  failure authority). Its results lookup names `prediction_pool.json` (`agents/watcher_agent.py:2223` at
  `a15b524`).
- No consumer reads the `predictions` list as a ranked unique pool.

---

## 15. What RANGE-MINER, the 22-Array NPZ and the Window Anchor Changed

*(Added 2026-11-11.)*

### 15.1 The survivor input became a stub

- **History.** S166-ACCUM (`8ebdf99`, 2026-04-21) stopped retaining forward and reverse survivor objects, to
  bound RAM at fleet scale. The S172 D3.5 finalizer (`46a3828`, 2026-07-25) kept that behaviour.
- **What Step 1 writes now.** Both files are written as `{"survivor_count": N, "note": "Full survivors omitted —
  objects not retained; see …"}` (`window_optimizer_integration_final.py:2945-2953`). In `--config-file` mode they
  are `[]` (Chapter 1 audit C-5).
- **Where the survivors live instead.** The canonical carrier is the certified 22-array NPZ generation
  (`utils/canonical_arrays.py:98` `CANONICAL_ARRAY_CONTRACT`). Steps 2 and 3 consume it.
- **Step 6 was not moved to it.** It still reads the two JSON files.
- **The governance record says otherwise.** `PROPOSAL_S172_RANGE_MINER_v1_4_4.md:249` and `:320` list
  `forward_survivors.json` as "Channel D … summary stub … Step 1 → Step 6" and say "runtime tolerates summary stub".
  **At `a15b524` it does not:**
  - The dual-sieve path raises `ValueError` at `survivor_scorer.py:136`.
  - The forward-only and `[]` paths produce an empty pool and write nothing.

  This is gap **B1**.

### 15.2 The dataset path is frozen

P0.5 dataset authority rewrites Step 6's `lottery_history` parameter from bare `daily3.json` to the run's frozen
absolute immutable path before dispatch (`agents/watcher_agent.py:474` `_P05_DATASET_PARAM_KEYS`, `:1862-1870` at
`a15b524`). Step 6's own code is unchanged by this.

### 15.3 Window anchor and generator phase (Brief I)

- **No change reached Step 6.** `prediction_generator.py` and `survivor_scorer.py` contain neither
  `window_anchor` nor `generator_phase`.
- **How Step 6 aligns.** It reads its vote at generator output index `len(history)`, with generator phase 0 at
  history index 0 (`prediction_generator.py:839`, `:846`). The on-the-fly features use the same alignment
  (`survivor_scorer.py:354-365`).
- **No document addresses Step 6.** Searched: `docs/*WINDOW_ANCHOR*`, `docs/*BRIEF_I*`,
  `docs/TB_RULING_WINDOW_ANCHOR*` — nothing mentions Step 6 or `prediction_generator`. Whether this alignment is
  right for a survivor found at `window_anchor ≠ 0` is therefore not settled (open item O3).

### 15.4 RANGE-MINER execution

None. Step 6 is not distributed (`agent_manifests/prediction.json` `"distributed": false`; Chapter 9 audit row 22).

---

## 16. Known Gaps at a15b524

*(Added 2026-11-11. Full evidence in `docs/CHAPTER_7_AUDIT.md` §4–§7.)*

### 16.1 BROKEN

| # | Gap | Stops at | A repair would have to |
|---|---|---|---|
| **B1** | Step 6's survivor input has been a count-only stub since S166. The dual-sieve path raises `ValueError`; the forward-only path writes nothing. Step 6 never reads the population Step 5 trained on. Not recorded as a defect; the S172 proposal asserts tolerance. | `survivor_scorer.py:136` | Feed Step 6 the population Step 3 scored and Step 5 trained on, with the per-survivor fields the vote needs. Make a stub or empty input fail loudly. |
| **B2** | A missing checkpoint is caught; survivors are ranked by the feature-row mean, reported as success, exit 0. Exit 2 is never returned. | `prediction_generator.py:375` | Fail with a named non-zero exit, or record a governed decision that the fallback is intended and mark its output as not model-based. |
| **B3** | The pool is not a vote: one value per top survivor, no aggregation or weighting, repeats kept, cut to `k`. *(GOVERNED: `TB_RULING_S179_IMPLEMENTATION_AUTH.md` §2.5, approved and not implemented.)* | `prediction_generator.py:849` | What S179 §2.5 specifies (deterministic order, duplicate collapse, persisted survivor id), plus any weighting or tiering the owner chooses. |

**Cross-referenced, counted in other audits:**
- **Chapter 6 B5** — no schema-hash validation in Step 6.
- **Chapter 6 B3** — the checkpoint's architecture is not the sidecar's (`models/wrappers/neural_net_wrapper.py:406`
  loads the checkpoint's own `hidden_layers`).
- **Chapter 13 B1** — Chapter 13 grades a file Step 6 does not write.
- **Chapter 13 B15 Blocker 1** — no ranked-predictions producer.
- **Chapter 12 B1** — WATCHER validates by file existence; repaired at `b6cb789`.

### 16.2 SWITCHED-OFF-BY-RULING

- **S1 — `--output-dir` is accepted and ignored; the output path is fixed to `predictions/`.**
  - **Ruling.** The Team Beta ruling of 2026-01-28, cited at `prediction_generator.py:902` and in commit `c6e3989`
    (the ruling text was not found as a file).
  - **What lifting it would enable.** Step 6 could write where it is told. WATCHER's success condition and
    freshness check name only `predictions/next_draw_prediction.json` (`agent_manifests/prediction.json:12`,
    `:40-41`), so the manifest would have to move with it, or WATCHER would see a missing or stale output.

### 16.3 DESCRIPTION-ONLY

| # | Section | What is wrong |
|---|---|---|
| D1 | header, §13 | Version and line counts |
| D2 | §1.2 | Four model types; there are five |
| D3 | §1.3 | "Step 2.5" and the missing Step 0 |
| D4 | §2.1 | `lottery_history.json`; the file is `daily3.json`, frozen by P0.5 |
| D5 | §3.2 | `load_best_model` / `ModelFactory.load` names |
| D6 | §3.3 | The sidecar example |
| D7 | §5.1 | `rank_survivors` |
| D8 | §6.2 | `predict_next` |
| D9 | §7.2 | The output shape |
| D10 | §9.1, §11.2 | CLI flags |
| D11 | §10.1 | Manifest content |
| D12 | §10.3 | The metadata fields |
| D13 | §11.1 | Troubleshooting messages |
| D14 | §11.3 | Hash-check file |
| D15 | §12 | Method names |
| D16 | §13 and the step table | Chapter self-name and step map |

**UNVERIFIED (U1):** the §6.1 / §8.3 / §13 hit-rate and lift figures.

**Absorbed (X1–X8):** see `docs/CHAPTER_7_AUDIT.md` §7.

---

## 17. Open Items

| # | Item | Where it is tracked |
|---|---|---|
| O1 | Repairs for B1–B3. Documentation only here; the repairs are separate briefs. | This chapter §16.1; audit §4 |
| O2 | U1: no graded walk-forward Hit@K exists for any Step 6 output. | S179 §2.6; Chapter 13 audit A.7 |
| O3 | Whether the vote index `len(history)` at generator phase 0 is right for survivors found at `window_anchor ≠ 0`. No Brief I document addresses Step 6. | This chapter §15.3; audit §2 Q3 |
| O4 | N1: every vote in the dual-sieve path uses `default_skip`, not the survivor's own skip. The fact file states the dict-path behaviour as the production path. | Audit §7 N1 |
| O5 | N2/N3: fourteen parsed-but-unused configuration knobs; Step 4's config reaches Step 6 only through same-named keys. | Audit §7 N2, N3 |
| O6 | The text of the 2026-01-28 Beta ruling behind S1 was not found as a file. | Audit §5 |
| O7 | A PREDICTION_SKIPPED exit 3 is a distinct state in Step 6. At `a15b524` WATCHER does not tell it apart from success, because it judges by file existence; after `b6cb789` it treats it as a generic failure. No distinct WATCHER handling exists either way. | Chapter 12 audit B1; this chapter §10.2 |

---

## 18. Verification Declaration (VIR-1…6)

- **VIR-1 execution proof.** Every file opened in this pass is listed in `docs/CHAPTER_7_AUDIT.md` §8. Every anchor in
  §14–§16 was read in this session. The main session's five-anchor spot-check is recorded by the main session.
- **VIR-2 clean control.** `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` was read for shape only and not edited. This pass
  modified only this chapter and `docs/CHAPTER_7_AUDIT.md`.
- **VIR-3 fault-injection control.** Not applicable to a documentation pass. None was invented.
- **VIR-4 completion sentinel.** See §19.
- **VIR-5 unavailable-observer behaviour.** These were not read and are reported as **UNAVAILABLE**, not clean:
  - `.127`;
  - the rigs;
  - root and other users' crontabs;
  - the private remote;
  - pre-repository archives;
  - the text of the 2026-01-28 ruling.

  Nothing was executed, so B1's `ValueError` is traced by reading, not observed.
- **VIR-6 audit claim scope.** Each claim is what this chapter says, checked against what the code at `a15b524`
  does. Whether the design is right is not judged.
  - **Searched surfaces:**
    - the tracked tree;
    - gitignored files (`agent_manifests/definitions.json`, `predictions/`, the sidecar, survivor stubs, logs);
    - git history (`-S` searches listed in the audit §8);
    - host crontab and systemd.
  - **Governance trail searched:**
    - `docs/TB_RULING_S177_KPI_GOVERNANCE.md` and `docs/TB_RULING_S179_IMPLEMENTATION_AUTH.md`;
    - `docs/TB_RULING_*` and `docs/PROPOSAL_*` by grep;
    - `PROPOSAL_S172_RANGE_MINER_v1_4_4.md` and `PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md`;
    - `docs/TEAM_ALPHA_SKIP_SEMANTICS_SUBMISSION.md`.
  - **Chapters searched:** the audits of Chapters 1, 3, 4, 5, 6, 9, 12, 13 and 14, and
    `docs/PIPELINE_BEHAVIOUR_MODEL.md`.

---

## 19. Closure Statement

- **Verified against:** `a15b524` (code identical at HEAD `b6cb789` outside the WATCHER files, which were read at
  `a15b524`).
- **What changed in this pass:**
  - Added the §1 framing block, status header and production note.
  - Added inline corrections in §1–§13.
  - Marked §2.2, §4.3, §5.2, §6, §7, §8 and §10.2 **SUPERSEDED (2026-11-11)**, with their text kept.
  - Added §14–§19.
- **Size.** No section was removed and no heading was renamed. The chapter is longer than its 933-line starting
  point, so the §3.3 shrink rule does not apply.
- **Counts:** 47 claims —
  - 18 accurate;
  - 3 BROKEN (B1, B2, B3; B3 governed);
  - 1 SWITCHED-OFF-BY-RULING (S1);
  - 16 DESCRIPTION-ONLY;
  - 1 UNVERIFIED (U1);
  - 8 absorbed (X1–X8).

  18 + 3 + 1 + 16 + 1 + 8 = 47.
- **What this chapter is not:** a closed chapter, a repair, or a judgement on the design.

**Completion sentinel: PASS** — every claim in the chapter is labelled or placed, and every section is corrected or
marked. U1 stays UNVERIFIED because no artifact exists to measure it, not because the audit was left unfinished.

---

*End of Chapter 7: Prediction Generator (Step 6)* — corrected 2026-11-11; this line previously read "End of
Chapter 16: Prediction Generator (Step 6)" (D16).
