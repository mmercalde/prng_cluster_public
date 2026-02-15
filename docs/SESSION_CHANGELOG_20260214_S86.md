# SESSION CHANGELOG -- S86
**Date:** 2026-02-14
**Session:** 86
**Focus:** Chapter 14 Tasks 8.5-8.7 Soak Harness + feature_names Fix

---

## Summary

Built `test_phase_8_soak.py` v1.5.0 -- two-mode soak test harness for Ch14
Phase 8 Tasks 8.5-8.7. Five rounds of Team Beta review evolved Mode B from
synthetic Monte Carlo (v1.0) through to calling the REAL S85 observe-only
hook with verified signatures (v1.5). Also fixed `feature_names=None` gap in
`_load_best_model_if_available()`, identified 27 stale project files for
removal, and created progress tracker v3.9.

---

## Part 1: Soak Test Harness

### Version History

| Version | What Changed | TB Finding |
|---------|-------------|------------|
| v1.0.0 | Mode A 11/11, Mode B was synthetic random.uniform() | "Mode B is Monte Carlo, not real soak" |
| v1.1.0 | Mode B switched to disk-loaded files | "Still reimplements classification, doesn't call real hook" |
| v1.2.0 | Mode B calls real hook, draw_id staleness, GPU guardrails | "Still reimplements prediction loader, unsafe __new__, signature brittleness" |
| v1.3.0 | Real constructor, real loader, positional calls | "Double-count bug, diagnostics pollution, no proof-of-call, classification drift" |
| v1.4.0 | Honest counters, input validation, classification allowlist, hook evidence | "Gate never fires (no regression data), but harness proven sound" |
| **v1.5.0** | **VERIFIED S85 signatures from Zeus inspect.signature. Fixed loader(path,draw_id) and hook(draw_result,predictions,model,model_type,feature_names). Model loaded once via _load_best_model_if_available(). Archive call added.** | **TB APPROVED** |

### Signature Verification (v1.5.0)

Zeus introspection revealed v1.0-v1.4 had WRONG call signatures:

| Method | v1.4 assumed | Real S85 signature |
|--------|-------------|-------------------|
| load_predictions_from_disk | (draw_id) | **(predictions_path: str, expected_draw_id: Optional[str])** |
| post_draw_root_cause_analysis | (diagnostics, predictions, draw_id) | **(draw_result, predictions, model, model_type, feature_names)** |
| _load_best_model_if_available | not called | **() -> {model, model_type, feature_names}** |
| _archive_post_draw_analysis | not called | **(analysis: Dict) -> None** |

### TB Review #4 Findings (all addressed in v1.4.0)

| # | Finding | Fix |
|---|---------|-----|
| 1 | Double-count: stale counter increments even when hook handles empty | Renamed to empty_predictions_returned, separate loader_exception counter |
| 2 | draw_id extracted in harness, not by orchestrator | Renamed to _best_effort_extract_draw_id, logs when None |
| 3 | Constructor may load GPU libs during init | Post-init assert: CUDA_VISIBLE_DEVICES still empty |
| 4 | No positive proof real methods are called | hook_result_keys_union + first_hook_result_sample in stats |
| 5 | Unknown classifications silently create buckets | Allowlist maps to known set, unknown_raw_classifications list |
| 6 | history/ files may be hook outputs, not inputs | Input validation: diagnostics files must contain required keys |

### TB Review #3 Findings (all addressed in v1.3.0)

| # | Finding | Fix |
|---|---------|-----|
| 1 | Still reimplements prediction load + staleness path | Calls real load_predictions_from_disk() |
| 2 | __new__ bypasses __init__ -- instance state missing | Proper constructor: Chapter13Orchestrator(use_llm=False) |
| 3 | Keyword args may not match S85 param names | All calls positional |
| 4 | Stale counter based on harness logic, not real method | Counter based on what real method returns |
| 5 | setdefault() doesn't override existing env vars | Force override: os.environ["CUDA_VISIBLE_DEVICES"] = "" |
| 6 | Prediction file globbing redundant with real loader | Removed |
| 7 | Stats claim CPU-only without verifying from hook output | Stats record whatever real hook returns |

### TB Review #2 Findings (all addressed in v1.2.0)

| # | Finding | Fix |
|---|---------|-----|
| 1 | Mode B reimplements classification instead of calling real hook | Imports Chapter13Orchestrator, calls real methods |
| 2 | No draw_id staleness enforcement in Mode B | Per-cycle draw_id extraction, match-or-reject |
| 3 | "real draw cycles" label is dishonest | Renamed to "disk-driven replay soak" |
| 4 | survivors_with_scores.json wrong schema for predictions | Removed |
| 5 | No GPU env guardrail | Sets CUDA/HIP/ROCR_VISIBLE_DEVICES="" |
| 6 | Docs say "scaffold ready" but code is implemented | Fixed |
| 7 | Test name mismatch between changelog and code | Aligned |

### Mode A -- Contract Tests (11/11 PASSED)

| Task | Test Name |
|------|-----------|
| 8.5 | diagnostics_structure_contract |
| 8.5 | worst_severity_wins |
| 8.5 | history_monotonic_capped |
| 8.6 | declining_ratio_fires_warning |
| 8.6 | stable_ratio_no_warning |
| 8.6 | trend_observe_only |
| 8.7 | regression_gate_contract |
| 8.7 | classification_heuristic_contract |
| 8.7 | root_cause_observe_only |
| 8.7 | cpu_only_contract |
| 8.7 | staleness_check_contract |

### Mode B -- Zeus Results (30/30 clean, 5/5 passed)

```
Total cycles:        30
Gate true rate:      0.0%  (no regression data on disk)
Diag files:          3 (validated)
Model:               lightgbm, features=62
Predictions:         NOT FOUND (pipeline hasn't generated yet)
GPU isolation:       True
CPU-only confirmed:  True
Signatures verified: True
Real hook used:      True
Errors:              0
```

---

## Part 2: feature_names Fallback Fix

### Problem
`_load_best_model_if_available()` returned `feature_names=None` because
`best_model.meta.json` sidecar doesn't include them. The trained LightGBM
model itself has 62 names via `model.feature_name()`.

### Fix
Added 14-line fallback in `_load_best_model_if_available()` (line 594):
if `feature_names is None` after sidecar read, extract from model object.
Supports LightGBM (`feature_name()`), sklearn (`feature_names_in_`),
XGBoost (`feature_names`). Wrapped in try/except (non-fatal).

### Verified
Before: `feature_names: NoneType None`
After: `feature_names: list len=62, first 5: ['Column_0', ..., 'Column_4']`

### Long-term
The proper fix is emitting `feature_names` into `best_model.meta.json`
at training time. The fallback stays as defense-in-depth.

---

## Part 3: Documentation Cleanup

27 stale project files identified for removal.
See `PROJECT_FILE_CLEANUP_S86.md` for complete manifest.

---

## Commits

| Hash | Description |
|------|-------------|
| `c468d3f` | feat: Ch14 Tasks 8.5-8.7 soak harness v1.5.0 -- TB approved, verified S85 signatures (S86) |
| `c52a3dd` | docs: S86 changelog |
| `c75a5d3` | fix: feature_names fallback from model object when sidecar lacks them (S86) |

---

## Key Learnings (S86)

**New rule adopted:** For harness/adapter code:
1. `inspect.signature()` before writing any call
2. Smoke run with 1 cycle
3. If a path is gated, create a synthetic trigger to exercise it

This would have caught the signature mismatches at v1.0 instead of v1.5.

---

## Next Steps (S87)

1. Create controlled regression diagnostics (Option A) to force gate=True
   and exercise the full downstream path
2. Generate predictions/ranked_predictions.json (pipeline run needed)
3. Remove 27 stale project files from Claude project
4. Upload progress v3.9 + fresh orchestrator to Claude project
5. Phase 9: First diagnostic investigation on Zeus with real data

---

## Memory Updates

- STATUS: Soak v1.5.0 TB-approved+deployed. Mode A 11/11, Mode B 30/30.
  gate_true=0% (data gap). feature_names=None fixed.
- RULES: Added harness rule: inspect.signature(), smoke 1 cycle,
  synthetic trigger for gated paths.
- TODOs: Updated (feature_names done, regression diag next).

---

*Session 86 -- Team Alpha*
