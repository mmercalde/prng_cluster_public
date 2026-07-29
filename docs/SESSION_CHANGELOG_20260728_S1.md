# SESSION CHANGELOG — 2026-07-28 S1

**Session:** S1 (2026-07-28)
**Team Alpha:** Claude (chat) + Claude Code (VM101 live-source trace)
**Reviewer:** Team Beta (two review rounds + trace review — all rulings binding)
**Scope:** Selfplay learning architecture investigation. **No production code modified. No pipeline launched. Docs-only commit.**
**Parallel work (untouched by this session):** S172 Phase-5 D6 threshold correction — see §7.

---

## 1. Why this session happened

Question raised: *does selfplay actually learn?* Investigation established it does not, then progressively established what TFM **does** do — ending in a live-source trace that corrected both Team Alpha and Team Beta on load-bearing facts.

Net outcome: the architecture was not wrong; **the map of it was.** This session replaces the map.

---

## 2. Artifacts added

| File | Purpose |
|---|---|
| `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md` | Canonical as-built map. Supersedes v1.0/v1.1. |
| `docs/S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md` | Claude Code read-only live-source trace on VM101 — **primary evidence** for every claim in the map. |
| `docs/TB_UPDATE_SELFPLAY_REFRAMING_2026-07-28.md` | Team Alpha → Team Beta reframing memo. |

**Removed:** `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_0.md` — superseded and **actively wrong** (claimed attribution `[WIRED]`, feature count ~62). Deleted to prevent two contradictory maps coexisting. v1.1 never left the sandbox.

---

## 3. Principal findings

### 3.1 Selfplay is a discovery front-end, not the learning system
The grade → attribute → concentrate → reinforce loop is **already built**: Ch13 feedback daemon (`new_draw.flag` → `run_cycle`), `evaluate_pools.py` (coverage + lift vs random, pools 20/100/300), `prediction_generator.py` (confidence weighting + Signal Quality Gate abstention), `reinforcement_engine.py`, WATCHER (~85% autonomy). Selfplay is the **missing discovery/proposal stage** feeding it. Team Beta accepted this reframing.

### 3.2 Feature contract: 91 extracted / 89 trained (not ~62)
Empirically verified. Decomposition: **13** NPZ-backed metadata + **59** seed/history-derived + **14** `global_*` run-context + **5** dead placeholders = 91; minus `score`, `confidence` = **89** trained. The "~62" traced to `feature_importance.py:95-119` (46+14=60), **stale by 31 features** (23 battery/S113, 4 digit/S119, 4 metadata), duplicated in `feature_drift_tracker.py:206-207`.

### 3.3 Per-survivor attribution: implemented, invoked, unreachable
`per_survivor_attribution.py` implements genuine single-survivor attribution for all four model families; Ch13 calls it with seed identity. It **cannot run in production** — four independent blockers:
1. no producer for `predictions/ranked_predictions.json`;
2. `chapter_13_orchestrator.py:582` reads `feature_names` at sidecar top level (lives at `feature_schema.feature_names`);
3. Ch13 bypasses the canonical NN loader — `torch.load()` returns a **dict** for the S92/S121 checkpoint format, so `.eval()` is not a valid load *(TB finding)*;
4. NN attribution omits the training scaler → attribution computed at the wrong point in feature space, changing gradient magnitude **and direction** *(TB finding)*.

All archived artifacts on disk are **synthetic**. No consumer acts on attribution output.

### 3.4 RANGE-MINER obligation: CLOSED
The frozen **22-array NPZ contract is sufficient and complete**. All 19 category-(c) features are either survivor-independent (`global_*`, draw-history-derived) or structurally dead (no producer anywhere). Michael's original requirement — *"produce exactly the parameters PWC made"* — is confirmed correct and sufficient. **No miner-side change required.**

### 3.5 Three defects found outside the original question
1. **METADATA LOSS guardrail is near-useless** — `generate_step3_scoring_jobs.py:95-100` raises if `len < 3` while the message claims "Expected 20+"; dropping 19 of 22 fields passes silently. Does not cover the JSON/list branch.
2. **Sequential-fallback silent corruption (P0)** — batch merges 18 metadata fields, sequential merges 6. On GPU batch failure, **seven NPZ-backed features silently become 0.0** in structurally valid 91-key records. Gate **F-PARITY** required; until it passes, fallback output is invalid for training.
3. **`forward_matches`/`reverse_matches` produced but never become model features** — the v3.1.0 repair that gave them per-seed variance is transported through NPZ and chunk layers and never reaches the model. Team Beta: possibly the most consequential finding of the trace. Requires a governed Option A (add + leakage/redundancy analysis + retrain) or Option B (correct docs + record rationale) decision. **Not to be silently "fixed" during S172.**

### 3.6 Three feature namespaces (new governance)
`global_*` values are stamped **identically** on every survivor in a run ⇒ filtering survivors on a global field is meaningless (retains/removes the whole run), and **random row-level folds across runs leak run identity/regime context**. Globals move to a `context_features.*` namespace, excluded from ordinary per-survivor filter search; multi-run datasets require run-grouped or temporally separated folds. Constrains REV2.1 search space and fold construction.

### 3.7 Attribution must preserve direction
The engine applies `abs()` then normalizes — answering "which feature mattered most" but not "did it push quality up or down." A strength-seeking loop needs sign. Revised artifact retains **both** `signed_contribution` and `absolute_share` plus prediction/baseline/completeness/method/checkpoint/schema/preprocessing hashes.

### 3.8 `holdout_hits` resolved
**Classification A — authorized offline outcome-derived supervised label.** Permitted as Step-5 target; **forbidden as filter, weight, mask, window or production-time feature.** Conditional on non-overlapping recorded train/holdout intervals, train-history-only feature generation, and persisted history hashes/temporal boundaries.

---

## 4. Corrections recorded (both directions)

**Team Alpha errors, corrected by source:**
- Claimed live selfplay target was `score`; it is `holdout_hits` (`selfplay_orchestrator.py:933`). Read a secondary helper, not the live builder.
- Claimed the accumulation/autonomy loop "appears nowhere" — scoped to 4 selfplay files; it lives in Ch13/WATCHER/reinforcement/prediction_generator.
- Claimed coverage-lift metric was missing — already in `evaluate_pools.py`.
- Claimed "NN earns its place through attribution, not R²" — TB corrected: **both** (Gate NN-Q quality *and* Gate NN-A attribution utility beyond tree SHAP).
- Marked attribution `[WIRED]` in map v1.0 — it is unreachable.
- Called activation "two cheap unblocks" — TB: *"one-line patch alone not sufficient; do not land in isolation."*
- Repeated "~62 features" from stale docs.

**Team Beta correction, withdrawn by them:** the earlier finding that only batch-aggregated attribution existed generalized from `training_diagnostics.py:474` (`mean(dim=0)` — correct about that file) and missed the separate `per_survivor_attribution.py` module.

**Meta-lesson:** every Alpha error came from reasoning about a component instead of tracing the live path end-to-end (invocation → artifact → consumer). One Claude Code live-source trace resolved in a single pass what four conversational rounds could not. **Prefer live-source traces over clone-based reasoning for any as-built claim.**

---

## 5. Team Beta rulings this session

1. Reframing **accepted** — selfplay is the discovery/proposal front-end.
2. Selfplay proposal **architecturally approved**; Optuna engine coding blocked pending REV2.1 corrections A–E.
3. Trace **accepted**; prior attribution assessment **corrected**; feature-contract + attribution-activation elevated to **P0**.
4. **S172 Phase-6 acceptance confirmed and strengthened:** for all four declared backend paths — exactly 22 arrays; exact name set; frozen order; no missing/extra; exact dtype; exact shape; `np.array_equal` per array; identical row ordering; Step-2 load-back with `fallback_used=False`. `np.array_equal` alone is insufficient (equal values can differ in dtype).
5. Companion proposal **directed** (not inline).

---

## 6. Open queue

**New required brief (before either broad proposal enters coding):**
- `TODO_TFM_FEATURE_CONTRACT_AND_ATTRIBUTION_ACTIVATION_v1_0.md` — owns P0-A (freeze feature contract), P0-B (Step-3 semantic parity), P0-C (safe attribution activation).

**Then:**
- `PROPOSAL_SELFPLAY_LEARNING_LOOP_REV2_1_ADDENDUM.md` — narrow search contract + TB corrections A–E + feature-registry/namespace dependencies.
- `PROPOSAL_TFM_ATTRIBUTION_DRIVEN_DISCOVERY_LOOP_v1_0.md` — scope is **activate/harden/aggregate/connect** the existing engine, not build attribution.

**Also outstanding:** promotion-seam repair (`chapter_13_acceptance.py:224` `SelfplayCandidate` lacks `transforms`; `promote_candidate:818` writes without them); rolling 25-year coverage-lift backtest as north-star objective; strength-seeking (opportunity) trigger; synthetic-cycle generator as durability tester.

---

## 7. S172 / RANGE-MINER status — unaffected

**RANGE-MINER 22-array parity work: authorized, unchanged.** Expanding the miner beyond 22 arrays: **not authorized.** D6 threshold-correction work proceeds independently in its own session; this session modified **no** miner, coordinator or pipeline code. GitHub pushes for D6 remain **held** until Phase 6 by Michael's instruction.

---

## 8. Commit contents

Docs only — no code paths touched, so this cannot disturb the in-flight D6 working tree.

```
A  docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md
A  docs/S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md
A  docs/TB_UPDATE_SELFPLAY_REFRAMING_2026-07-28.md
D  docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_0.md
A  docs/SESSION_CHANGELOG_20260728_S1.md
```

Dual-push: `git push origin main && git push public main`

---

**END — SESSION CHANGELOG 20260728 S1**
