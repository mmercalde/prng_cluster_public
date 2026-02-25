# S111 IMPLEMENTATION PLAN — Holdout Validation Redesign
**Version:** FINAL (verified against live Zeus code)  
**Date:** 2026-02-25  
**Author:** Claude (Team Alpha Lead Dev)  
**Status:** READY FOR DEPLOYMENT  
**Prerequisite:** v1.1 proposal APPROVED by Team Beta  
**Depends On:** S110 Phase 0 Baseline (COMPLETE)

---

## 0. Executive Summary

Replace the broken `holdout_hits` y-label (Poisson λ≈1 noise, R²=0.000155) with
`holdout_quality` — a composite score computed using the same SurvivorScorer
methodology applied to holdout draws. Three files modified, one new module added.

**Expected outcome:** R² improvement from 0.000155 to 0.01–0.10+ range (100×+ lift).
Even modest improvement validates the consistent scoring rule principle.

---

## 1. Files Inventory

| File | Action | Purpose |
|------|--------|---------|
| `holdout_quality.py` | **NEW** | Standalone module: `compute_holdout_quality()`, `get_survivor_skip()`, `compute_autocorrelation_diagnostics()` |
| `full_scoring_worker.py` | **MODIFY** | Step 3: Add holdout feature extraction + quality computation per survivor |
| `meta_prediction_optimizer_anti_overfit.py` | **MODIFY** | Step 5: Switch default target from `holdout_hits` → `holdout_quality` |
| `selfplay_orchestrator.py` | **MODIFY** (future) | Selfplay: Update target preference chain (deferred to separate session) |

**What does NOT change:**
- `train_single_trial.py` — receives pre-built NPZ arrays, no target logic
- `subprocess_trial_coordinator.py` — passes through X/y arrays, no target logic  
- `prediction_generator.py` (Step 6) — unchanged
- `chapter_13_orchestrator.py` — unchanged (external validation anchor)
- `run_step3_full_scoring.sh` — `--holdout-history` already passed
- NPZ format — no changes
- WATCHER agent — pipeline contracts unchanged

---

## 2. File 1: holdout_quality.py (NEW MODULE)

**Destination:** `~/distributed_prng_analysis/holdout_quality.py`  
**Status:** Already delivered in previous S111 session, verified correct.

### Contents:
- `get_survivor_skip(metadata)` → int: `skip_best` → `skip_min` → 0
- `compute_holdout_quality(holdout_features)` → float [0,1]: composite 50/30/20
- `compute_autocorrelation_diagnostics(survivors)` → dict: per-feature Pearson r

### Deployment:
```bash
# From ser8 Downloads:
scp holdout_quality.py rzeus:~/distributed_prng_analysis/

# Verify on Zeus:
cd ~/distributed_prng_analysis
source ~/venvs/torch/bin/activate
python3 -c "from holdout_quality import compute_holdout_quality, get_survivor_skip, compute_autocorrelation_diagnostics; print('OK')"
```

---

## 3. File 2: full_scoring_worker.py (STEP 3 EDITS)

**Verified line numbers from live Zeus code (2026-02-25).**

### Code Structure Summary:
```
Line 1-30:    Docstring + header
Line 212:     def compute_holdout_hits(...)
Line 255:     def compute_holdout_hits_batch(...)
Line 303:     holdout_history parameter in score_all_survivors()
Line 341-352: holdout_hits_map computation (EXISTING, stays vestigial)
Line 357:     try: all_features = scorer.extract_ml_features_batch(...)
Line 387-388: result["holdout_hits"] = holdout_hits_map.get(seed, 0.0)  ← PRIMARY INSERTION POINT
Line 420-424: Fallback sequential path with holdout_hits  ← SECONDARY INSERTION POINT
Line 439:     def main()
Line 480:     --holdout-history CLI arg
Line 530:     Load holdout history
```

### EDIT 1: Add import near top

Find the imports section (before first function definition). Add after existing imports:

```python
# S111: Holdout quality computation
from holdout_quality import compute_holdout_quality, get_survivor_skip
```

**Location:** Insert after the last `import` or `from ... import` statement in the header
block, before `HOST = socket.gethostname()` or equivalent constant definitions.

### EDIT 2: Add holdout feature extraction in GPU batch path (after line 388)

**Current code at lines 385-390:**
```python
            # v2.0: Add holdout_hits (y-label for Step 5)
            result["holdout_hits"] = holdout_hits_map.get(seed, 0.0)
            results.append(result)
```

**Insert AFTER `result["holdout_hits"] = ...` and BEFORE `results.append(result)`:**

```python
            # S111: Compute holdout_quality (consistent scoring rule)
            if holdout_history is not None and len(holdout_history) > 0:
                try:
                    meta = survivor_metadata.get(seed, {}) if survivor_metadata else {}
                    skip_val = get_survivor_skip(meta)
                    holdout_feats = scorer.extract_ml_features(
                        seed=seed,
                        lottery_history=holdout_history,
                        skip=skip_val
                    )
                    result["holdout_features"] = {
                        k: float(v) if isinstance(v, (int, float)) else v
                        for k, v in holdout_feats.items()
                    }
                    result["holdout_quality"] = compute_holdout_quality(holdout_feats)
                except Exception as e:
                    logger.warning(f"[S111] holdout_quality failed for seed {seed}: {e}")
                    result["holdout_features"] = {}
                    result["holdout_quality"] = 0.0
            else:
                result["holdout_features"] = {}
                result["holdout_quality"] = 0.0
```

**Result: the block becomes:**
```python
            # v2.0: Add holdout_hits (y-label for Step 5)
            result["holdout_hits"] = holdout_hits_map.get(seed, 0.0)
            # S111: Compute holdout_quality (consistent scoring rule)
            if holdout_history is not None and len(holdout_history) > 0:
                try:
                    meta = survivor_metadata.get(seed, {}) if survivor_metadata else {}
                    skip_val = get_survivor_skip(meta)
                    holdout_feats = scorer.extract_ml_features(
                        seed=seed,
                        lottery_history=holdout_history,
                        skip=skip_val
                    )
                    result["holdout_features"] = {
                        k: float(v) if isinstance(v, (int, float)) else v
                        for k, v in holdout_feats.items()
                    }
                    result["holdout_quality"] = compute_holdout_quality(holdout_feats)
                except Exception as e:
                    logger.warning(f"[S111] holdout_quality failed for seed {seed}: {e}")
                    result["holdout_features"] = {}
                    result["holdout_quality"] = 0.0
            else:
                result["holdout_features"] = {}
                result["holdout_quality"] = 0.0
            results.append(result)
```

### EDIT 3: Add holdout quality in fallback sequential path (after line 424)

**Current code at lines 420-426:**
```python
                results.append({
                    'seed': seed,
                    'score': float(score),
                    'features': features,
                    'metadata': {'prng_type': prng_type, 'mod': mod, 
                                'worker_hostname': HOST, 'timestamp': time.time()},
                    'holdout_hits': holdout_hits_map.get(seed, 0.0),
                })
```

**Replace with:**
```python
                fallback_result = {
                    'seed': seed,
                    'score': float(score),
                    'features': features,
                    'metadata': {'prng_type': prng_type, 'mod': mod, 
                                'worker_hostname': HOST, 'timestamp': time.time()},
                    'holdout_hits': holdout_hits_map.get(seed, 0.0),
                }
                # S111: Compute holdout_quality in fallback path
                if holdout_history is not None and len(holdout_history) > 0:
                    try:
                        meta = survivor_metadata.get(seed, {}) if survivor_metadata else {}
                        skip_val = get_survivor_skip(meta)
                        holdout_feats = scorer.extract_ml_features(
                            seed=seed,
                            lottery_history=holdout_history,
                            skip=skip_val
                        )
                        fallback_result["holdout_features"] = {
                            k: float(v) if isinstance(v, (int, float)) else v
                            for k, v in holdout_feats.items()
                        }
                        fallback_result["holdout_quality"] = compute_holdout_quality(holdout_feats)
                    except Exception as e2_hq:
                        logger.warning(f"[S111] holdout_quality fallback failed for seed {seed}: {e2_hq}")
                        fallback_result["holdout_features"] = {}
                        fallback_result["holdout_quality"] = 0.0
                else:
                    fallback_result["holdout_features"] = {}
                    fallback_result["holdout_quality"] = 0.0
                results.append(fallback_result)
```

### IMPORTANT NOTE ON SKIP PARAMETER

The `scorer.extract_ml_features()` call may or may not accept a `skip` parameter
depending on the current SurvivorScorer implementation. If it does not have a `skip`
parameter, the call should be:

```python
holdout_feats = scorer.extract_ml_features(
    seed=seed,
    lottery_history=holdout_history
)
```

**Verify before applying:**
```bash
grep -n "def extract_ml_features" survivor_scorer.py | head -5
python3 -c "from survivor_scorer import SurvivorScorer; import inspect; print(inspect.signature(SurvivorScorer.extract_ml_features))"
```

If `skip` is not a parameter, remove `skip=skip_val` from both insertion points.
The `get_survivor_skip()` call and variable are still useful for future use and
should be kept as comments.

---

## 4. File 3: meta_prediction_optimizer_anti_overfit.py (STEP 5 EDITS)

**Verified line numbers from live Zeus code (2026-02-25).**

### EDIT 1: Line 359 — Update signal quality default

```python
# OLD (line 359):
def compute_signal_quality(y: np.ndarray, target_name: str = "holdout_hits") -> Dict:

# NEW:
def compute_signal_quality(y: np.ndarray, target_name: str = "holdout_quality") -> Dict:
```

### EDIT 2: Line 591 — Update default target field

```python
# OLD (line 591):
    target_field: str = "holdout_hits",

# NEW:
    target_field: str = "holdout_quality",
```

### EDIT 3: Line 608 — Add holdout_quality to exclude list

```python
# OLD (line 608):
    exclude_features = exclude_features or ['score', 'confidence', 'holdout_hits']

# NEW:
    exclude_features = exclude_features or ['score', 'confidence', 'holdout_hits', 'holdout_quality']
```

### sed commands for Step 5:

```bash
cd ~/distributed_prng_analysis
source ~/venvs/torch/bin/activate

# Backup
cp meta_prediction_optimizer_anti_overfit.py meta_prediction_optimizer_anti_overfit.py.bak_S111

# Edit 1: Line 359
sed -i '359s/target_name: str = "holdout_hits"/target_name: str = "holdout_quality"/' meta_prediction_optimizer_anti_overfit.py

# Edit 2: Line 591
sed -i '591s/target_field: str = "holdout_hits"/target_field: str = "holdout_quality"/' meta_prediction_optimizer_anti_overfit.py

# Edit 3: Line 608
sed -i "608s/\['score', 'confidence', 'holdout_hits'\]/['score', 'confidence', 'holdout_hits', 'holdout_quality']/" meta_prediction_optimizer_anti_overfit.py

# Verify all 3 edits took:
echo "=== Verify edits ==="
sed -n '359p' meta_prediction_optimizer_anti_overfit.py
sed -n '591p' meta_prediction_optimizer_anti_overfit.py
sed -n '608p' meta_prediction_optimizer_anti_overfit.py

# Should show:
#   def compute_signal_quality(y: np.ndarray, target_name: str = "holdout_quality") -> Dict:
#       target_field: str = "holdout_quality",
#       exclude_features = exclude_features or ['score', 'confidence', 'holdout_hits', 'holdout_quality']

# Syntax check:
python3 -c "import ast; ast.parse(open('meta_prediction_optimizer_anti_overfit.py').read()); print('SYNTAX OK')"
```

---

## 5. Verification Suite

### Phase 1: Module Import Test
```bash
cd ~/distributed_prng_analysis
source ~/venvs/torch/bin/activate

# Test holdout_quality module
python3 -c "
from holdout_quality import compute_holdout_quality, get_survivor_skip, compute_autocorrelation_diagnostics

# Test with sample features
test_feats = {
    'residue_1000_match_rate': 0.15,
    'lane_agreement_8': 0.20,
    'lane_agreement_125': 0.18,
    'lane_consistency': 0.22,
    'residue_1000_coherence': 0.30,
    'residue_8_coherence': 0.28,
    'residue_125_coherence': 0.25,
    'temporal_stability_mean': 0.40,
}
q = compute_holdout_quality(test_feats)
print(f'holdout_quality = {q:.6f}')
assert 0.0 <= q <= 1.0, f'Out of range: {q}'

# Test skip extraction
assert get_survivor_skip({'skip_best': 5}) == 5
assert get_survivor_skip({'skip_min': 3}) == 3
assert get_survivor_skip({}) == 0
assert get_survivor_skip(None) == 0
print('All module tests PASSED')
"
```

### Phase 2: Syntax Validation
```bash
python3 -c "import ast; ast.parse(open('full_scoring_worker.py').read()); print('full_scoring_worker.py: SYNTAX OK')"
python3 -c "import ast; ast.parse(open('meta_prediction_optimizer_anti_overfit.py').read()); print('meta_prediction_optimizer: SYNTAX OK')"
python3 -c "import ast; ast.parse(open('holdout_quality.py').read()); print('holdout_quality.py: SYNTAX OK')"
```

### Phase 3: Import Chain Test
```bash
# Test that full_scoring_worker can import holdout_quality
python3 -c "
import sys
sys.path.insert(0, '.')
from holdout_quality import compute_holdout_quality, get_survivor_skip
print('Import chain OK')
"
```

### Phase 4: Step 5 Target Verification
```bash
# Verify the target field change
python3 -c "
import inspect, importlib.util
spec = importlib.util.spec_from_file_location('m', 'meta_prediction_optimizer_anti_overfit.py')
m = importlib.util.module_from_spec(spec)
# Don't exec — just check the source
source = open('meta_prediction_optimizer_anti_overfit.py').read()
assert 'target_field: str = \"holdout_quality\"' in source, 'target_field not updated!'
assert 'target_name: str = \"holdout_quality\"' in source, 'target_name not updated!'
assert \"'holdout_quality'\" in source.split('exclude_features')[2], 'exclude_features not updated!'
print('Step 5 target verification PASSED')
"
```

### Phase 5: S97 Symlink Check
```bash
# Check if survivors_with_scores.json is a symlink to an old run
ls -la survivors_with_scores.json
# If it points to old data without holdout_quality field, Step 5 will
# fall back to 0.0 for all targets. Must re-run Step 3 first.
```

---

## 6. Deployment Sequence

### Step A: Transfer files from ser8 to Zeus

```bash
# On ser8 (files in ~/Downloads from Claude delivery):
cd ~/Downloads
scp holdout_quality.py rzeus:~/distributed_prng_analysis/
scp S111_IMPLEMENTATION_PLAN_FINAL.md rzeus:~/distributed_prng_analysis/docs/
scp SESSION_CHANGELOG_20260225_S111.md rzeus:~/distributed_prng_analysis/docs/
```

### Step B: Backup on Zeus

```bash
cd ~/distributed_prng_analysis
source ~/venvs/torch/bin/activate
cp full_scoring_worker.py full_scoring_worker.py.bak_S111
cp meta_prediction_optimizer_anti_overfit.py meta_prediction_optimizer_anti_overfit.py.bak_S111
```

### Step C: Apply Step 5 edits (3 sed commands)

```bash
# These are safe one-line substitutions with verified line numbers
sed -i '359s/target_name: str = "holdout_hits"/target_name: str = "holdout_quality"/' meta_prediction_optimizer_anti_overfit.py
sed -i '591s/target_field: str = "holdout_hits"/target_field: str = "holdout_quality"/' meta_prediction_optimizer_anti_overfit.py
sed -i "608s/\['score', 'confidence', 'holdout_hits'\]/['score', 'confidence', 'holdout_hits', 'holdout_quality']/" meta_prediction_optimizer_anti_overfit.py
```

### Step D: Apply Step 3 edits (manual — open in editor)

The Step 3 edits require multi-line insertions. Use `nano` or `vim`:

```bash
nano full_scoring_worker.py
```

1. **Add import** near the top (after other imports):
   ```python
   from holdout_quality import compute_holdout_quality, get_survivor_skip
   ```

2. **Find line 388** (`result["holdout_hits"] = holdout_hits_map.get(seed, 0.0)`).
   Insert the S111 block from §3 EDIT 2 between that line and `results.append(result)`.

3. **Find the fallback path** (~line 420-426, the `results.append({...})` dict with
   `'holdout_hits'`). Replace with the expanded version from §3 EDIT 3.

4. Save and exit.

### Step E: Run verification suite (§5 above)

### Step F: Git commit

```bash
git add holdout_quality.py full_scoring_worker.py meta_prediction_optimizer_anti_overfit.py docs/
git commit -m "S111: Holdout validation redesign - holdout_quality replaces holdout_hits as ML target

New module: holdout_quality.py (compute_holdout_quality, get_survivor_skip, autocorrelation diagnostics)
Step 3: full_scoring_worker.py - add holdout feature extraction + quality computation per survivor
Step 5: meta_prediction_optimizer_anti_overfit.py - switch target_field to holdout_quality
Proposal: v1.1 approved by Team Beta, addresses Poisson noise floor + TFM inconsistency"
git push public main
```

---

## 7. What Happens Next (Post-Deployment)

### Immediate: Re-run Steps 3→6
```bash
# Step 3 must run first to generate survivors_with_scores.json containing holdout_quality
# Then Steps 4→5→6 will consume the new target
./run_step3_full_scoring.sh
# Then run Steps 4-6 per COMPLETE_OPERATING_GUIDE
```

### Validate: Check Phase 1 results
```bash
# After Step 5 completes, check:
python3 -c "
import json
with open('models/reinforcement/best_model.meta.json') as f:
    meta = json.load(f)
print(f'Training target: {meta.get(\"provenance\", {}).get(\"training_target\", \"UNKNOWN\")}')
# Should show: Training target: holdout_quality
"
```

### Expected metrics comparison:

| Metric | Phase 0 (holdout_hits) | Phase 1 (holdout_quality) |
|--------|----------------------|--------------------------|
| Target values | 8 discrete (Poisson) | Continuous [0,1] |
| Target variance | 1.0e-06 | Significantly higher |
| R² | 0.000155 | 0.01-0.10+ expected |
| Feature alignment | 0/6 dimensions | 6/6 dimensions |
| Skip semantics | Uniform 4000 | Per-survivor from NPZ |

### Deferred items:
- `selfplay_orchestrator.py` line 852: target preference chain update (separate session)
- Autocorrelation diagnostics analysis (after first Phase 1 run)
- Chapter 13 external validation (when live draw ingestion operational)

---

## 8. Rollback Plan

All changes are additive and backward-compatible:

1. **Step 5 rollback:** Change `holdout_quality` back to `holdout_hits` in the 3 lines
2. **Step 3 rollback:** The `holdout_quality` and `holdout_features` fields are additive — 
   old consumers ignore unknown fields. To fully revert, restore from `.bak_S111` backup.
3. **holdout_quality.py:** Has no consumers until Step 3 imports it. Can be removed safely.

---

## 9. Chain of Evidence

| What | How Verified | When |
|------|-------------|------|
| `full_scoring_worker.py` line numbers | Michael pasted Zeus output: `grep -n`, `sed -n '330,560p'` | This session |
| `train_single_trial.py` receives pre-built NPZ | Lines 682-687 in project files: `data = np.load(args.data_path)` | This session |
| `subprocess_trial_coordinator.py` passes through arrays | Lines 113-158: receives X/y, saves to NPZ | This session |
| Target extraction in `meta_prediction_optimizer_anti_overfit.py` | Lines 591, 608, 629-640: `target_field="holdout_hits"` | This session |
| `selfplay_orchestrator.py` line 852 | Project files: `survivor.get("holdout_hits", ...)` | This session |
| `holdout_quality.py` module correct | Verified in `/mnt/user-data/outputs/` from previous session | This session |

---

**END OF PLAN**
