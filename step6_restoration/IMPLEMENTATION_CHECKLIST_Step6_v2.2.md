# IMPLEMENTATION CHECKLIST: Step 6 Restoration v2.2
## Progress Tracker

**Created:** December 23, 2025  
**Status:** PENDING APPROVAL → IMPLEMENTATION  
**Estimated Time:** 2-3 hours

---

## Pre-Implementation

- [ ] **Team Beta Approval** - Awaiting sign-off on v2.2 proposal
- [ ] **Michael Approval** - Final go-ahead
- [ ] **Backup all affected files**
  ```bash
  mkdir -p backups/step6_restoration_$(date +%Y%m%d)
  cp survivor_scorer.py backups/step6_restoration_$(date +%Y%m%d)/
  cp prediction_generator.py backups/step6_restoration_$(date +%Y%m%d)/
  cp meta_prediction_optimizer_anti_overfit.py backups/step6_restoration_$(date +%Y%m%d)/
  cp train_single_trial.py backups/step6_restoration_$(date +%Y%m%d)/
  cp subprocess_trial_coordinator.py backups/step6_restoration_$(date +%Y%m%d)/
  cp reinforcement_engine.py backups/step6_restoration_$(date +%Y%m%d)/
  cp generate_ml_jobs.py backups/step6_restoration_$(date +%Y%m%d)/
  cp models/__init__.py backups/step6_restoration_$(date +%Y%m%d)/
  cp agent_manifests/prediction.json backups/step6_restoration_$(date +%Y%m%d)/
  ```

---

## Phase 1: Create GPU-Neutral GlobalStateTracker Module

- [ ] **1.1** Create `models/global_state_tracker.py`
  - [ ] Copy GlobalStateTracker class from `reinforcement_engine.py` (lines 277-450)
  - [ ] Add module docstring with usage example
  - [ ] Add `get_feature_names()` method
  - [ ] Add `get_feature_values()` method  
  - [ ] Add constants: `GLOBAL_FEATURE_COUNT = 14`, `GLOBAL_FEATURE_NAMES`
  - [ ] Verify NO torch/cuda imports

- [ ] **1.2** Update `models/__init__.py`
  - [ ] Add import: `from models.global_state_tracker import GlobalStateTracker, GLOBAL_FEATURE_COUNT, GLOBAL_FEATURE_NAMES`
  - [ ] Add to `__all__` list

- [ ] **1.3** Test Phase 1
  ```bash
  python3 -c "
  from models.global_state_tracker import GlobalStateTracker, GLOBAL_FEATURE_COUNT
  tracker = GlobalStateTracker([100, 200, 300] * 100, {'mod': 1000})
  state = tracker.get_global_state()
  print(f'Features: {len(state)} (expected {GLOBAL_FEATURE_COUNT})')
  assert len(state) == GLOBAL_FEATURE_COUNT
  print('✅ Phase 1 PASSED')
  "
  ```

---

## Phase 2: Update Existing GlobalStateTracker Imports

- [ ] **2.1** Update `reinforcement_engine.py`
  - [ ] Add import: `from models.global_state_tracker import GlobalStateTracker`
  - [ ] Remove class definition (lines 277-450) OR comment out with note
  - [ ] Verify backward compat: existing code still works

- [ ] **2.2** Update `generate_ml_jobs.py`
  - [ ] Add import: `from models.global_state_tracker import GlobalStateTracker`
  - [ ] Remove duplicate class definition (line 458+)

- [ ] **2.3** Test Phase 2
  ```bash
  python3 -c "
  from reinforcement_engine import ReinforcementEngine, ReinforcementConfig
  print('✅ reinforcement_engine import works')
  "
  python3 -c "
  from generate_ml_jobs import GlobalStateTracker
  print('✅ generate_ml_jobs import works')
  "
  ```

---

## Phase 3: Fix survivor_scorer.py

- [ ] **3.1** Remove hardcoded PRNG (line 116)
  - [ ] Change `self.generate_sequence = java_lcg_sequence` 
  - [ ] To `self._cpu_func = get_cpu_reference(self.prng_type)`

- [ ] **3.2** Add `_generate_sequence()` method
  ```python
  def _generate_sequence(self, seed: int, n: int, skip: int = 0) -> np.ndarray:
      raw = self._cpu_func(seed=int(seed), n=n, skip=skip)
      return np.array([v % self.mod for v in raw], dtype=np.int64)
  ```

- [ ] **3.3** Update `extract_ml_features()` (line 124)
  - [ ] Change `self.generate_sequence(seed, n, self.mod)`
  - [ ] To `self._generate_sequence(seed, n, skip=skip)`

- [ ] **3.4** Add `compute_dual_sieve_intersection()` method
  - [ ] Returns `Dict` with `intersection` (sorted), `jaccard`, `counts`
  - [ ] NEVER discard valid intersection

- [ ] **3.5** Test Phase 3
  ```bash
  python3 -c "
  from survivor_scorer import SurvivorScorer
  scorer = SurvivorScorer(prng_type='java_lcg', mod=1000)
  
  # Test PRNG
  seq = scorer._generate_sequence(12345, 10)
  print(f'Sequence: {seq[:5]}...')
  
  # Test intersection
  result = scorer.compute_dual_sieve_intersection([1,2,3,4], [3,4,5,6])
  print(f'Intersection: {result}')
  assert result['intersection'] == [3, 4]  # sorted
  assert result['jaccard'] == 2/6
  print('✅ Phase 3 PASSED')
  "
  ```

---

## Phase 4: Update Step 5 (meta_prediction_optimizer_anti_overfit.py)

- [ ] **4.1** Add GlobalStateTracker import
  ```python
  from models.global_state_tracker import GlobalStateTracker, GLOBAL_FEATURE_COUNT
  ```

- [ ] **4.2** Initialize GlobalStateTracker after loading lottery data (~line 975)
  ```python
  global_tracker = GlobalStateTracker(lottery_draws, {'mod': 1000})
  global_state = global_tracker.get_global_state()
  global_feature_names = sorted(global_state.keys())
  global_feature_values = np.array([global_state[k] for k in global_feature_names])
  print(f"  ✅ Global features: {len(global_feature_names)}")
  ```

- [ ] **4.3** Concatenate global features to X matrix (~line 965)
  ```python
  global_broadcast = np.tile(global_feature_values, (X_per_seed.shape[0], 1))
  X = np.hstack([X_per_seed, global_broadcast])
  ```

- [ ] **4.4** Update sidecar generation to v3.2.0 schema
  - [ ] Add `per_seed_feature_count`
  - [ ] Add `per_seed_feature_names`
  - [ ] Add `global_feature_count`
  - [ ] Add `global_feature_names`
  - [ ] Add `total_features`
  - [ ] Add `combined_hash`
  - [ ] Update `schema_version` to "3.2.0"

- [ ] **4.5** Test Phase 4
  ```bash
  python3 meta_prediction_optimizer_anti_overfit.py \
      --survivors survivors_with_scores.json \
      --lottery-data synthetic_lottery.json \
      --model-type xgboost \
      --trials 2
  
  # Verify sidecar
  python3 -c "
  import json
  with open('models/reinforcement/best_model.meta.json') as f:
      meta = json.load(f)
  fs = meta['feature_schema']
  print(f'Schema version: {meta.get(\"schema_version\")}')
  print(f'Per-seed: {fs.get(\"per_seed_feature_count\")}')
  print(f'Global: {fs.get(\"global_feature_count\")}')
  print(f'Total: {fs.get(\"total_features\")}')
  assert fs.get('global_feature_count') == 14
  print('✅ Phase 4 PASSED')
  "
  ```

---

## Phase 5: Update Subprocess Worker (train_single_trial.py)

- [ ] **5.1** Update data loading to accept 62 features
  - [ ] Verify `X_train.shape[1]` matches expected
  - [ ] Add validation for feature count

- [ ] **5.2** Test Phase 5
  ```bash
  # Will be tested as part of Phase 4 Step 5 run
  ```

---

## Phase 6: Update subprocess_trial_coordinator.py

- [ ] **6.1** Save global features with trial data
  ```python
  np.savez(data_path,
      X_train=X_train, y_train=y_train,
      X_val=X_val, y_val=y_val,
      total_features=X_train.shape[1],
      per_seed_features=per_seed_count,
      global_features=global_count
  )
  ```

- [ ] **6.2** Test Phase 6
  ```bash
  # Will be tested as part of Phase 4 Step 5 run
  ```

---

## Phase 7: Update prediction_generator.py (Step 6)

- [ ] **7.1** Add imports
  ```python
  from models.global_state_tracker import GlobalStateTracker, GLOBAL_FEATURE_COUNT
  from models.model_factory import load_model_from_sidecar
  ```

- [ ] **7.2** Add `models_dir` to PredictionConfig dataclass

- [ ] **7.3** Load model in `__init__`
  ```python
  self.model, self.model_meta = load_model_from_sidecar(
      models_dir=config.models_dir,
      device='cuda' if GPU_AVAILABLE else 'cpu',
      survivors_file=config.survivors_forward,
      strict=True
  )
  ```

- [ ] **7.4** Initialize GlobalStateTracker in `generate_predictions()`
  ```python
  self.global_tracker = GlobalStateTracker(lottery_history, {'mod': self.config.mod})
  global_values = self.global_tracker.get_feature_values()
  ```

- [ ] **7.5** Update intersection call (returns Dict now)
  ```python
  intersection_result = self.scorer.compute_dual_sieve_intersection(...)
  intersection = intersection_result["intersection"]
  jaccard = intersection_result["jaccard"]
  ```

- [ ] **7.6** Add `_build_prediction_pool()` method
  - [ ] Handle both int and dict survivor formats
  - [ ] Use sidecar `feature_names` for ordering
  - [ ] Append global features
  - [ ] Validate `X.shape[1] == total_features`
  - [ ] Use `model.predict(X)` for scoring
  - [ ] Backward compat for old sidecar (no globals)

- [ ] **7.7** Add `_empty_pool_result()` helper

- [ ] **7.8** Remove call to `scorer.build_prediction_pool()` (now internal)

- [ ] **7.9** Test Phase 7
  ```bash
  python3 prediction_generator.py \
      --models-dir models/reinforcement \
      --survivors-forward survivors_with_scores.json \
      --lottery-history synthetic_lottery.json \
      --k 10
  ```

---

## Phase 8: Update Manifests

- [ ] **8.1** Fix `agent_manifests/prediction.json`
  - [ ] Change `"script": "generate_predictions.py"` to `"script": "prediction_generator.py"`
  - [ ] Verify `args_map` includes `models-dir`

- [ ] **8.2** Test Phase 8
  ```bash
  python3 -c "
  import json
  with open('agent_manifests/prediction.json') as f:
      manifest = json.load(f)
  assert 'prediction_generator.py' in str(manifest)
  print('✅ Phase 8 PASSED')
  "
  ```

---

## Phase 9: Integration Testing

- [ ] **9.1** Full Step 5 → Step 6 Pipeline Test
  ```bash
  # Step 5: Train with global features
  python3 meta_prediction_optimizer_anti_overfit.py \
      --survivors survivors_with_scores.json \
      --lottery-data synthetic_lottery.json \
      --compare-models \
      --trials 8

  # Step 6: Generate predictions
  python3 prediction_generator.py \
      --models-dir models/reinforcement \
      --survivors-forward survivors_with_scores.json \
      --lottery-history synthetic_lottery.json \
      --k 20
  ```

- [ ] **9.2** Test with integer survivors (backward compat)
  ```bash
  python3 prediction_generator.py \
      --models-dir models/reinforcement \
      --survivors-forward bidirectional_survivors.json \
      --lottery-history synthetic_lottery.json \
      --k 10
  ```

- [ ] **9.3** Test with old sidecar (no global features)
  ```bash
  # If old sidecar exists, test backward compat
  ```

- [ ] **9.4** Verify no GPU conflicts
  ```bash
  # Run Step 5 with --compare-models (all 4 model types)
  # Verify LightGBM (OpenCL) still works alongside CUDA models
  ```

---

## Phase 10: Documentation & Commit

- [ ] **10.1** Update `IMPLEMENTATION_CHECKLIST_v3_1_2.md`
  - [ ] Add Step 6 restoration section
  - [ ] Mark v3.2.0 sidecar schema

- [ ] **10.2** Update `SYSTEM_ARCHITECTURE_REFERENCE.md`
  - [ ] Add GlobalStateTracker module
  - [ ] Update feature count (48 → 62)

- [ ] **10.3** Git commit
  ```bash
  git add -A
  git commit -m "Step 6 Restoration v2.2: GlobalStateTracker extraction, global features integration, survivor format handling"
  ```

- [ ] **10.4** Push to remote
  ```bash
  git push origin main
  ```

---

## Rollback Plan

If issues arise:
```bash
# Restore from backups
cp backups/step6_restoration_*/survivor_scorer.py .
cp backups/step6_restoration_*/prediction_generator.py .
# ... etc for all files

# Or git revert
git revert HEAD
```

---

## Sign-Off

| Phase | Status | Completed By | Date |
|-------|--------|--------------|------|
| Pre-Implementation | ☐ | | |
| Phase 1: GlobalStateTracker module | ☐ | | |
| Phase 2: Update imports | ☐ | | |
| Phase 3: survivor_scorer.py | ☐ | | |
| Phase 4: Step 5 (meta optimizer) | ☐ | | |
| Phase 5: train_single_trial.py | ☐ | | |
| Phase 6: subprocess_coordinator | ☐ | | |
| Phase 7: prediction_generator.py | ☐ | | |
| Phase 8: Manifests | ☐ | | |
| Phase 9: Integration testing | ☐ | | |
| Phase 10: Documentation & commit | ☐ | | |

---

**IMPLEMENTATION READY UPON APPROVAL**
