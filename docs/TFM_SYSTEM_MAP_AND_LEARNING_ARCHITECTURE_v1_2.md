# TFM System Map & Learning Architecture — Canonical Reference

**Version:** 1.2.0
**Date:** 2026-07-28
**Author:** Team Alpha (Claude)
**Status:** REFERENCE — v1.2 supersedes v1.1 and v1.0. Corrections in §7 are load-bearing; do not cite earlier versions.
**Primary evidence:** `docs/S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md` + **Team Beta review of that trace (2026-07-28), binding** (Claude Code, VM101 live source, read-only, 2026-07-28). Where this map and that report disagree, **the report wins** — it read live source; v1.0 of this map read a public clone.
**Purpose:** One document explaining how TFM actually learns, what is wired vs. stubbed, and how RANGE-MINER (S172) relates to the learning/attribution layer — so future sessions stop re-deriving it.

### What changed in v1.1
- Per-survivor attribution reclassified **[WIRED] → [IMPLEMENTED, INVOKED, UNREACHABLE]** (v1.0 was wrong; Team Beta's "not implemented" was also wrong — see §4).
- Feature vector corrected **~62 → 91** (89 trained).
- Three new defects recorded (§7.3): weak METADATA LOSS guardrail, sequential-fallback silent zero-fill, `forward_matches`/`reverse_matches` unused.
- RANGE-MINER question **closed with evidence** (§7.1).

### What changed in v1.2 (Team Beta review of the trace — binding)
- **Attribution blockers: 2 → 4.** Added (3) Chapter 13 bypasses the canonical NN loader and (4) NN attribution omits the training scaler. See §4.2.
- **v1.1's "two cheap unblocks" is WITHDRAWN.** TB ruling: *"One-line Chapter 13 feature-name patch alone — Not sufficient; do not land in isolation."* See §4.4.
- **New §9: the three feature namespaces** (survivor-local / run-global / dead) — governs REV2.1 search space and fold construction.
- **New §10: attribution must preserve signed contributions**, not only absolute shares.
- `forward_matches`/`reverse_matches` elevated: TB calls it possibly the most consequential finding; requires a governed schema decision (§7.3).
- Feature-contract + attribution-activation work elevated to **P0** with a new landing order (§11) and a new required brief.

---

## 1. The pipeline in one screen

```
Step 1  Window Optimizer (Optuna TPE)  → best window(s)
Step 2  Bidirectional Sieve            → survivors  [RANGE-MINER (S172) replaces this engine]
Step 3  Full Scoring (26 GPUs)         → scored survivors + 91-feature per-survivor vector
Step 5  Model Training (4 models)      → best model (trained on 89) + diagnostics
Step 6  Prediction Generation          → pools (20/100/300)
        ────────────────────────────────────────────────
Feedback  Chapter 13 daemon            → ingest draw → grade → (attribute) → decide → relearn
```

Two carrier objects: the **22-array NPZ survivor contract** (`[S172 Phase-5 D3.0]`) and the **prediction pool + coverage/lift score**.

---

## 2. The autonomous loop, as-built

Legend: **[WIRED]** running · **[PARTIAL]** present, incomplete · **[BLOCKED]** implemented but cannot execute · **[GAP]** absent/broken.

| Capability | Status | Anchor |
|---|---|---|
| Live draw ingestion + feedback daemon | **[WIRED]** | `chapter_13_orchestrator.py` (`new_draw.flag` → `run_cycle`) |
| Coverage / hit-rate at fixed pool size, lift vs random | **[WIRED]** | `evaluate_pools.py:28` (pools 20/100/300); Ch13 downstream score ~`:305` |
| Hit-regression detection | **[WIRED]** | `chapter_13_orchestrator.py:536-561`, called `:325` |
| Per-survivor attribution **engine** (4 model families) | **[WIRED]** | `per_survivor_attribution.py:49,134,184,231`; dispatcher `:289-316` |
| Per-survivor attribution **execution in production** | **[BLOCKED]** | **four** independent blockers — §4.2 |
| Consumer acting on attribution output | **[GAP]** | none exists; observational end-to-end (report §2.7) |
| Concentrate pool on strength; abstain on weak signal | **[WIRED]** | `prediction_generator.py` + Signal Quality Gate (exit 3) |
| Reinforce high-quality survivors | **[WIRED]** | `reinforcement_engine.py` (`SurvivorQualityNet`, `GlobalStateTracker`) |
| Retrain triggers | **[WIRED]** | `chapter_13_triggers.py` — all defensive (§5) |
| LLM advisory + acceptance + human gate | **[WIRED]** | `chapter_13_llm_advisor.py`, `chapter_13_acceptance.py` |
| WATCHER pipeline autonomy (~85%) | **[WIRED]** | `agents/watcher_agent.py` |
| Accumulate signal across runs | **[PARTIAL]** | DB downstream write-back; seed-coverage tracker (S140); trial history (S140b) |
| **Discover new strong heuristics (search)** | **[GAP]** | self-play is an evaluation harness; Phase 9B.3 never built |
| **Confirm strength by generation** | **[GAP]** | no forward synthetic-cycle generator |
| **Persist discovered policy through promotion** | **[GAP/BROKEN]** | `chapter_13_acceptance.py:224` `SelfplayCandidate` lacks `transforms`; `promote_candidate` `:818` writes without them |

---

## 3. Vision → code mapping

| Vision element | Status |
|---|---|
| Autonomous operation, ingest live draws | **built** |
| "Which heuristics drove THIS survivor" | **built but unreachable** (§4) |
| Leverage strength / concentrate pool | **built** |
| Reinforce confirmed strength | **built** |
| Coverage-at-fixed-pool-size lift vs random | **built** (per-draw; not yet a rolling 25-yr objective) |
| **Search for NEW strong heuristics** | **gap** — self-play engine (REV2.1) |
| **Confirm strength under generation** | **gap** — synthetic generator (new capability) |

---

## 4. Per-survivor attribution — the corrected status

### 4.1 Two different modules, two different answers

- `training_diagnostics.py:474` — `feat_grads = self._input_gradients.abs().mean(dim=0)`. Hook captures first layer only (`:429-431`); `grad_input[0]` is `(batch, n_features)`, so `mean(dim=0)` reduces **over the batch**. **Batch-averaged importance, no survivor identity available.** *Team Beta's finding on this file is CONFIRMED.* Its `top_10` output (`:479-485`) has **no reader**. As batch importance it is consumed via `spread_ratio` (`:538`, `:596`; thresholds `:77`).
- `per_survivor_attribution.py` — a **separate module** implementing genuine single-sample attribution: NN `grad`/`grad_x_input` (`:49-127`, single-sample at `:98`,`:108`), XGB `pred_contribs` (`:134-177`), LGB `pred_contrib` (`:184-224`), CatBoost native SHAP (`:231-273`), dispatcher (`:289-316`). Prior TB corrections already applied (`:83-88`, `:93-95`). Chapter 13 imports and calls it **with seed identity attached** (`chapter_13_orchestrator.py:57`, `:702`, `:710-711`, `:725`, `:733-734`).

So the capability **is** implemented — Team Beta generalized from the wrong file, and v1.0 of this map claimed a working path that cannot run.

### 4.2 Why it never executes (FOUR independent blockers)

**Blocker 1 — no producer for `predictions/ranked_predictions.json`.** Reader exists (`chapter_13_orchestrator.py:875-937`, called `:328`); **no writer exists anywhere in the repo**. `load_predictions_from_disk()` always returns `None` in production.

**Blocker 2 — wrong sidecar key.** `chapter_13_orchestrator.py:582` reads `feature_names` at the sidecar top level; it lives at `feature_schema.feature_names`. Returns `None` for neural_net.

**Blocker 3 — Chapter 13 bypasses the canonical NN loader.** *(Team Beta, §4 of their review.)* Ch13 executes `torch.load(model_path, map_location="cpu", weights_only=False)` then `.eval()`. But the S92/S121 training path saves a **checkpoint dictionary** (`state_dict`, architecture metadata, feature count, normalization flags, scaler mean/scale, target mean/std). `torch.load()` on that returns a **dict, not an instantiated model** — `.eval()` on it is not a valid load path. Step 6 already uses the model factory + sidecar loader. **Ruling: replace the duplicated loader with the canonical mechanism returning an explicit inference bundle** (`model`, `model_type`, `feature_names`, `feature_schema_hash`, `feature_count`, `normalize_features`, `scaler_mean`, `scaler_scale`, `checkpoint_hash`, `checkpoint_version`) — do **not** patch it defect-by-defect into a second drifting implementation.

**Blocker 4 — NN attribution omits the training scaler.** *(Team Beta, §5.)* S92 standardizes `X_train` and applies training-fold statistics downstream; Step 6 loads the stored scaler before inference. But `per_survivor_attribution_nn` passes its supplied vector directly into the model (`x = torch.tensor(features, ...)`). Unless that vector is already normalized exactly as in training, **attribution is computed at the wrong point in feature space** — changing prediction, activation state, gradient magnitude, **gradient direction**, dead-neuron behavior and feature rankings. Not cosmetic.

> **Gate A-NORM (release-blocking):** for the same survivor, `Step-6 inference prediction == attribution-path forward prediction` within a declared tolerance, proving identical feature ordering, exclusion, scaling, model state, checkpoint and device semantics. **No attribution result is valid until prediction parity passes.**

Consequently **every archived artifact on disk is synthetic** (`Column_27`/`Column_42` placeholders), and `compare_pool_tiers()` (`per_survivor_attribution.py:323-440`) averages survivor identity away (`:399-402`) into a file that is never produced — so LLM readers always receive `None` (`agents/watcher_agent.py:1762-1764`, `:1776-1778`).

### 4.3 Correct classification

> **Implemented, invoked, unreachable, unconsumed.**
> Team Beta's formulation: *"Algorithmically implemented, partially integrated, operationally disconnected and not yet behaviorally closed."*
> Not "designed-documented only" (the code is real and test-exercised). Not "wired" at system level (it cannot run).

### 4.4 WITHDRAWN: v1.1's "two cheap unblocks"

v1.1 advised two quick fixes. **That advice is withdrawn.** Team Beta ruling: *"One-line Chapter 13 feature-name patch alone — **Not sufficient; do not land in isolation**."* With Blockers 3 and 4, activation requires the canonical loader, the scaler restoration, the ranked-prediction artifact contract, and A-NORM parity — see §11 (P0-C). A real attribution run is **blocked until schema, scaler and prediction-parity gates pass.**

### 4.5 The positional-alignment prohibition

Every backend `enumerate`s `feature_names` against raw positions with **no validation** (`per_survivor_attribution.py:117,173,220,269`). Ch13 currently does `features=np.asarray(features, dtype=np.float32)` with no schema-hash, count or name check.

**Required ranked-prediction record contract** (TB §6): `seed`, `rank`, `prediction`, `draw_id`, `model_id`, `feature_schema_hash`, and a **name-keyed** `feature_values` map. Ch13 must build the ordered vector by iterating the sidecar's authoritative `feature_names`, then assert: record hash == model hash; value count == 89; every required name present; no unexpected model-input name; ordered length == model input dim; all values finite. A prebuilt `model_input_vector` may be included for efficiency but **cannot replace** the name-keyed map. **Silent positional zipping is prohibited.**

---

## 5. Attribution semantics — signed contributions required

*(Team Beta §13.)* The engine converts contributions to absolute values and normalizes to sum 1 (NN: `grads = (x.grad * x).squeeze().abs()`; tree backends similarly `abs()` before normalizing). That answers *"which features had the largest influence magnitude"* — it does **not** answer *"did this feature push the survivor's quality prediction up or down."*

**A strength-seeking discovery loop needs direction.** The revised artifact must preserve **both** `signed_contribution` and `absolute_share`, plus: raw model prediction; expected/baseline prediction where supported; attribution sum / completeness residual; attribution method; model+checkpoint hash; schema hash; preprocessing hash. The normalized-absolute output may remain as a summary but cannot be the only retained evidence.

---

## 6. The strategic gap: defense vs. offense


All triggers in `chapter_13_triggers.py` are degradation-based: `CONSECUTIVE_MISSES`, `CONFIDENCE_DRIFT`, `HIT_RATE_COLLAPSE`, `N_DRAWS`, window-decay. **No opportunity trigger** ("a heuristic shows durable edge — press it").

Nuance (Michael's correction, accepted): the **pipeline** does find working configurations (Step 1 Optuna + sieve + scoring). The gap is (a) the **feedback loop** is defensive — it recovers a decayed config, it does not seek edges; and (b) discovery operates at **config/window** level, not the **per-survivor-heuristic** level attribution exposes.

---

## 7. The data-contract spine — and RANGE-MINER (S172)

### 7.1 RANGE-MINER question: CLOSED

**The 22-array contract is sufficient and complete for RANGE-MINER.** The miner must emit **exactly the 22 arrays and nothing more.**

Live feature vector = **91** features (empirically verified on `full_scoring_results_20260311_170236/chunk_0000.json`, 84 records):
- **13** merged directly from NPZ arrays (`survivor_scorer.py:770-781`, list `:774-779`) — all verified non-zero and varying.
- **59** deterministically computed in Step 3 from the **regenerated PRNG sequence** (seeded by `seeds`) vs `train_history`: 27 match/stats (`survivor_scorer.py:702-731`), 9 residue (`:624-667`, mods `[8,125,1000]`), 23 battery S113 (`:745-757`).
- **19 category-(c)** — *not* from the 22 arrays: **14** `global_*` from `GlobalStateTracker(train_history)` (survivor-independent, draw-history-derived, identical for every survivor) and **5 dead placeholders** (`skip_mean`, `skip_std`, `skip_entropy`, `survivor_velocity`, `velocity_acceleration`) with **no producer anywhere in the repo**.

**Neither category obliges the miner to emit anything beyond the 22 arrays.** Michael's original requirement — *"the new range miner produces exactly the parameters PWC made"* — is therefore correct and sufficient.

Of the 22: `seeds` is identity + PRNG-regeneration key (drives all 59 computed features); 13 are merged as features; 8 are carried but unconsumed.

**Phase-6 acceptance (Team Beta confirmed + strengthened):** for **all four** declared backend paths, assert exactly 22 arrays; exact name set; **frozen order**; no missing/extra; exact **dtype**; exact **shape**; `np.array_equal` per array; identical row ordering; Step-2 load-back with `fallback_used=False`. `np.array_equal` alone is insufficient (equal values can differ in dtype); optionally `tobytes(order="C")` after verifying dtype/endianness/contiguity. Any single-array failure fails Phase 6.

### 7.2 Feature-count correction

**91 live / 89 trained.** The docs' "~62" traces to `feature_importance.py:95-119` (46 `STATISTICAL_FEATURES` + 14 `GLOBAL_STATE_FEATURES` = 60), which is **stale by 31** (23 battery, 4 digit S119, 4 metadata). Same stale list duplicated at `feature_drift_tracker.py:206-207`. Sidecar corroborates: `feature_schema.feature_count = 89`, `excluded_features = [score, confidence, holdout_hits, holdout_quality]`; 91 − 2 = 89 exactly. `full_scoring_worker.py` docstring/argparse still say **50** — also stale.

### 7.3 Three pre-existing defects (NOT caused by, and not fixable by, RANGE-MINER)

1. **METADATA LOSS guardrail is near-useless.** `generate_step3_scoring_jobs.py:95-100` raises if `len(result[0]) < 3` while the message claims "Expected 20+" (`:99`). Dropping 19 of 22 fields passes silently. Catches only total collapse. Also does **not** cover the JSON/list branch (`:104-117`).
2. **Sequential-fallback silent zero-fill.** Batch merges **18** metadata fields (`survivor_scorer.py:774-779`); sequential merges only **6** (`full_scoring_worker.py:453-454`). If the GPU batch throws (`:438-442` → `:443`), **seven** NPZ-backed features silently become 0.0 (`intersection_count`, `intersection_ratio`, `intersection_weight`, `bidirectional_selectivity`, `forward_only_count`, `reverse_only_count`, `survivor_overlap_ratio`). Same failure class as the Jan 23 2026 incident, on a path the guardrail does not cover; records remain structurally valid with 91 keys. **Team Beta classifies this P0**: the worker logs a fallback rather than failing, feature count is unchanged, downstream schema checks see all expected keys, and training may proceed on semantically corrupted data.

> **Gate F-PARITY (required):** on the same seeds/histories/metadata, `batch feature names == fallback feature names` and `batch feature values == fallback feature values` for every deterministic feature within declared tolerance; **all metadata-backed fields exactly equal**. Until F-PARITY passes, **sequential-fallback output must be marked invalid for model training**, not silently accepted.
3. **`forward_matches` / `reverse_matches` are produced but never become model features.** *(Team Beta: possibly the most consequential finding in the whole trace.)* The converter header (`convert_survivors_to_binary.py:16-20`) describes them as genuine **per-seed** quality signals — the v3.1.0 repair that stopped all quality fields being identical for every seed in a trial — yet they are absent from the Step-3 merge list (`survivor_scorer.py:774-779`). The two arrays specifically repaired to carry per-seed sieve quality are transported through NPZ and chunk layers and **never seen by the model**.

**Ruling (TB §8): do NOT silently add them during S172.** RANGE-MINER parity requires reproducing the current contract, not changing model semantics. Open a separately governed feature-schema decision:
- **Option A — intended features:** add under a new feature-schema version, run leakage + redundancy analysis, regenerate Step-3 artifacts, retrain all models.
- **Option B — intentionally excluded:** correct the converter documentation and record why the primary per-seed match rates must not enter training.

The present state — declared central ranking signals, silently unused — is unacceptable. **The miner must keep emitting both regardless.**

---

## 8. The three feature namespaces (governs REV2.1 search space)

*(Team Beta §10 — architecturally load-bearing; neither proposal had this.)* The 91 features are **not** one undifferentiated transform space:

```
13  NPZ-backed survivor metadata      ┐
59  seed/history-derived survivor     ┘ = 72 survivor-LOCAL
14  global_* run-context (identical for every survivor in a run)
 5  permanently-zero placeholders
─────
91  extracted   (− score, confidence = 89 trained)
```

### 8.1 Survivor-local (72) — the legitimate search space
May vary among survivors; can support ranking, masking, attribution and survivor conditioning.

### 8.2 Run-global context (14) — NOT an ordinary filter space
Computed once from draw history and stamped identically onto every survivor in the run. Therefore:
- **filtering survivors by a global field is meaningless** — it can only retain or remove the *whole run*;
- per-survivor attribution to a global feature is **contextual model dependence**, not evidence distinguishing survivors;
- **random row-level folds across multiple runs can leak run identity / regime context.**

**Governance:** globals belong in a separate `context_features.*` namespace. They may condition an outer policy or regime model but **must not be searchable as ordinary per-survivor filter thresholds**. When datasets combine multiple scoring runs, folds must be **grouped or temporally separated by run** — random row splits are insufficient.

### 8.3 Dead placeholders (5) — explicit lifecycle decision required
`skip_mean`, `skip_std`, `skip_entropy`, `survivor_velocity`, `velocity_acceleration`. Options: (1) remove under a versioned schema change + retrain; (2) retain as deprecated reserved fields guaranteed zero; (3) implement real producers under a new schema version + retrain. **They must not begin receiving nonzero values under the existing 89-feature schema** — that would be an unannounced distribution and semantic change.

---

## 9. Static feature registries are no longer authoritative

`feature_importance.py:95-119` omits **31 live features** (23 battery, 4 digit, 4 metadata); the same stale list is duplicated at `feature_drift_tracker.py:206-207`. Importance and drift tooling therefore under-observes the model.

**Required:** one canonical feature-schema producer, authority order `model sidecar feature schema → versioned extraction-schema manifest → runtime validation`. **No diagnostic module may maintain a manually duplicated feature-name list.** The canonical artifact classifies each feature by: stable name, dtype, origin, local/global status, deployability, model inclusion, transform eligibility, leakage classification, deprecation status, expected variance class.

---

## 10. `holdout_hits` — conditionally resolved

The trace establishes `holdout_hits` is computed from a holdout-history block, stored **outside** the feature dict, used as the Step-5 target, and is **not** one of the 91 inputs. Team Beta therefore resolves its prior Correction-B question:

> **Classification A — authorized offline outcome-derived supervised label.** Permitted as a training target; **forbidden as a filter, weight, mask, window or production-time feature.**

Conditions (all required): train/holdout intervals explicitly recorded and non-overlapping; feature generation uses train history only; never available to policy transforms; search folds and locked selection do not improperly reuse the same label block; all history hashes and temporal boundaries persisted. Provenance must be recorded in every selfplay study and model sidecar.

---

## 11. Root-cause classification stays observe-only

*(TB §14.)* Ch13 currently takes Top-20, splits hit/miss, keeps each survivor's top-3 feature names, compares name sets, and labels divergence > 0.5 a regime shift. For a single Pick-3 outcome the **hit cohort is frequently zero or one** — not enough evidence for an autonomous regime decision. Observe-only is correct.

Before attribution may influence search priors, the companion proposal must add: rolling multi-draw evidence; minimum hit/miss cohort sizes; attribution stability across draws; confidence intervals / bootstrap stability; model-family agreement; signed-contribution handling; temporal decay; abstention when evidence is sparse.

Also: replace **nested** Top-20/100/300 tiers with **disjoint rank bands** (1–20, 21–100, 101–300) — nested tiers recount the same high-ranked survivors and confound causal interpretation.

---

## 12. Intellectual-honesty log

1. **"Target is `score`, TB got it wrong."** Wrong — live path is `selfplay_orchestrator.py:933`, prefers `holdout_hits`. Read a secondary helper, not the live builder.
2. **"The accumulation/autonomy loop appears nowhere."** Wrong — scoped to 4 self-play files. It lives in Ch13 + WATCHER + reinforcement + prediction generator.
3. **"Coverage-at-fixed-pool-size lift is the missing keystone."** Wrong — already in `evaluate_pools.py`.
4. **"NN reinstatement is marginal."** Wrong rationale; but also **"NN earns its place through attribution, not R²"** was rejected by TB — correct position is *both* (Gate NN-Q quality **and** Gate NN-A attribution utility beyond tree SHAP). Trees do local attribution too (SHAP/`pred_contribs`).
5. **Did not read Chapter 14 before opining on the learning layer.**
6. **v1.0 of this map marked per-survivor attribution [WIRED].** Wrong — it is unreachable (§4). *Team Beta's counter-claim ("not implemented, only batch-averaged") was also wrong* — they generalized from `training_diagnostics.py` and missed `per_survivor_attribution.py`. Both errors corrected by live-source trace.
7. **Feature count "~62"** repeated from stale docs; real value is **91 / 89 trained**.
8. **v1.1 called attribution activation "two cheap unblocks."** Wrong — Team Beta found two further blockers (canonical-loader bypass; missing training scaler) and explicitly ruled the one-line patch *"not sufficient; do not land in isolation."* Error came from treating the blockers the trace happened to surface as the complete set. *(TB, in turn, had earlier generalized from `training_diagnostics.py` and withdrew that; both sides have now corrected each other from source.)*

**Meta-lesson:** every one of these came from reasoning about a component instead of tracing the live path end to end (invocation → artifact → consumer). The Claude Code VM101 trace resolved in one pass what four conversational rounds could not. **Prefer live-source traces over clone-based reasoning for any as-built claim.**

---

## 13. Required landing order (Team Beta, binding)

### P0-A — Freeze the real feature contract
1. Canonical **91-feature extraction manifest**; 2. canonical **89-feature model-input manifest**; 3. eliminate duplicated feature lists; 4. exact name/count/order/hash validation; 5. classify local / global / dead / label fields; 6. record schema version in every Step-3, Step-5, Step-6 and attribution artifact.

### P0-B — Repair Step-3 semantic parity
1. Replace the weak metadata-loss guard with exact required-field validation; 2. make batch and sequential metadata merges identical; 3. add **F-PARITY** tests; 4. fail closed on semantic schema loss; 5. decide the fate of `forward_matches` / `reverse_matches` (Option A or B, §7.3).

### P0-C — Activate attribution safely
1. Define the ranked-prediction artifact; 2. emit name-keyed model features + schema hash; 3. use the **canonical sidecar/model loader** in Ch13; 4. restore the exact NN scaler; 5. validate **A-NORM** prediction parity before attribution; 6. preserve signed + absolute contributions; 7. run one real-data **observe-only** attribution cycle; 8. prove artifact freshness and draw identity.

### P1 — Connect attribution to discovery
Aggregate evidence over multiple draws; separate survivor-local vs global-context signals; synthesize NN and tree attribution **without treating their raw scales as identical**; emit an advisory discovery-prior artifact; selfplay consumes it **as a prior only**; require locked evaluation before candidate emission; preserve Chapter 13 + human promotion authority.

---

## 14. Authorization status (Team Beta)

| Work item | Ruling |
|---|---|
| RANGE-MINER 22-array parity work | **Authorized; unchanged** |
| Expanding RANGE-MINER beyond 22 arrays | **Not authorized** |
| Canonical 91/89 feature-schema manifest | **Authorized** |
| Batch/fallback Step-3 parity repair | **Authorized, P0** |
| Static feature-list replacement | **Authorized** |
| One-line Ch13 feature-name patch alone | **Not sufficient; do not land in isolation** |
| Canonical Ch13 inference-bundle loader | **Authorized** |
| Ranked-prediction artifact design | **Authorized** |
| Real attribution run | **Blocked** until schema, scaler and prediction-parity gates pass |
| Forward/reverse match feature addition | **Requires separate feature-schema decision + retraining plan** |
| Attribution-driven autonomous parameter changes | **Not authorized** |
| Observe-only attribution→selfplay prior experiment | **Authorized after P0 activation** |

---

## 15. Document set

- `TODO_TFM_FEATURE_CONTRACT_AND_ATTRIBUTION_ACTIVATION_v1_0.md` — **new, owns P0-A/B/C; must exist before either broad proposal enters coding**
- `PROPOSAL_SELFPLAY_LEARNING_LOOP_REV2_1_ADDENDUM.md` — narrow search contract + dependencies: search requires the canonical feature registry; transform fields must be survivor-local and inference-time available; global context is not an ordinary per-survivor filter space; attribution priors unavailable until activation gates pass
- `PROPOSAL_TFM_ATTRIBUTION_DRIVEN_DISCOVERY_LOOP_v1_0.md` — scope is **activate / harden / aggregate / connect** the existing engine, **not** build attribution from scratch

**RANGE-MINER (S172) is upstream of all of this, correctly scoped, and blocks none of it. Finish D6.**

---

**END — TFM SYSTEM MAP v1.2**
