# HOLDOUT_HITS Implementation Summary
**Date:** December 31, 2025
**Author:** Claude + Michael
**Status:** In Progress - Position Offset Bug Identified

---

## 1. WHAT IS HOLDOUT_HITS?

### The Problem It Solves
When sieving for PRNG seeds, we find "survivors" - seeds that match the lottery data. But some survivors are **TRUE** (actual generator seed) and some are **LUCKY** (coincidental matches).

### The Solution
Split lottery data into two parts:
- **Training Data:** Used by sieve to FIND candidates
- **Holdout Data:** Used to VALIDATE which candidates are real

### Why It Works
- **TRUE seed:** Matches training 100% AND holdout 100% (it generated ALL the data)
- **LUCKY seed:** Matches training well (coincidence) but holdout ~0.1% (random chance)

### Visual Example
```
Seed 12345 generates 5000 sequential draws:

Position:  0    1    2   ...  3999  4000  4001  ... 4999
Draw:     875  331  694  ...   xxx   yyy   zzz  ...  www
          |_______ TRAINING ______|  |____ HOLDOUT ____|
          
Sieve uses TRAINING to find candidates
holdout_hits uses HOLDOUT to validate them
```

---

## 2. CHANGES MADE TODAY

### A. Added --holdout-history CLI argument

**Files Modified:**
1. `full_scoring_worker.py` - Added CLI arg and computation
2. `generate_step3_scoring_jobs.py` - Added to job generation
3. `run_step3_full_scoring.sh` - Added variable, CLI parsing, scp

**Chain:**
```
run_step3_full_scoring.sh
  └─→ defines HOLDOUT_HISTORY variable
  └─→ passes --holdout-history to generator
  └─→ SCPs holdout file to remote nodes

generate_step3_scoring_jobs.py  
  └─→ accepts --holdout-history CLI arg
  └─→ adds --holdout-history to worker args

full_scoring_worker.py
  └─→ accepts --holdout-history CLI arg
  └─→ computes holdout_hits per seed
  └─→ outputs in scored results
```

### B. Fixed PRNG Function API

**Bug:** `prng_func(seed, position)` returns a LIST, not single value
**Fix:** Generate all predictions at once:
```python
# OLD (broken):
for position, actual in enumerate(holdout):
    predicted = prng_func(seed, position) % mod  # WRONG - returns list

# NEW (fixed):
predictions = prng_func(seed, num_draws)  # Get all at once
for position, actual in enumerate(holdout):
    predicted = predictions[position] % mod  # Index into list
```

### C. Regenerated Synthetic Test Data

**Bug:** Old `synthetic_lottery.json` used different PRNG implementation than `prng_registry.py`
- Old generator: output BEFORE state advance
- prng_registry: output AFTER state advance

**Fix:** Regenerated using exact same method as prng_registry:
```python
def java_lcg_step(state):
    a = 25214903917
    c = 11
    m = 0xFFFFFFFFFFFF
    state = (a * state + c) & m
    output = (state >> 16) & 0xFFFFFFFF  # Output AFTER advance
    return state, output
```

**New files:**
- `synthetic_lottery_v2.json` - 5000 draws (lottery format)
- `synthetic_train_v2.json` - 4000 draws (positions 0-3999)
- `synthetic_holdout_v2.json` - 1000 draws (positions 4000-4999)
- TRUE_SEED = 12345

---

## 3. CURRENT BUG - POSITION OFFSET

### Symptom
Seed 12345 (known true seed) gets `holdout_hits: 0.001` instead of `1.0`

### Root Cause
Worker computes predictions from position 0, but holdout contains positions 4000-4999.
```python
# Current (WRONG):
predictions = prng_func(seed, 1000)  # Generates positions 0-999

# Correct:
predictions = prng_func(seed, 1000, skip=4000)  # Generates positions 4000-4999
```

### Proposed Fix
Add `offset` parameter to `compute_holdout_hits()`:
```python
def compute_holdout_hits(
    seed: int,
    holdout_history: List[int],
    prng_type: str = 'java_lcg',
    mod: int = 1000,
    offset: int = 0  # NEW: skip this many positions first
) -> float:
    ...
    predictions = prng_func(seed, num_draws, skip=offset)
```

The offset = len(training_data) = 4000

### Question for Team
How should we pass the offset to the worker?
1. Explicit `--holdout-offset` CLI argument?
2. Derive from `--train-history` file length?
3. Include in chunk metadata?

---

## 4. TEST RESULTS

### Before Fix (real daily3.json data):
```
10,000 survivors scored
Mean holdout_hits:  0.0010 (random chance = 1/1000)
Max holdout_hits:   0.0060
Anomalies (>1%):    0
```
This is expected - real lottery likely has no detectable PRNG pattern.

### After Synthetic Data Fix:
```
Seed 12345 local test: holdout_hits = 1.0000 (100%) ✅
Seed 12345 distributed: holdout_hits = 0.001 (wrong) ❌
```
Local test works because we manually computed with correct offset.
Distributed fails due to missing offset in worker.

---

## 5. FILES MODIFIED

| File | Changes |
|------|---------|
| `full_scoring_worker.py` | Added --holdout-history CLI, compute_holdout_hits(), fixed prng API |
| `generate_step3_scoring_jobs.py` | Added holdout_history_file parameter throughout |
| `run_step3_full_scoring.sh` | Added HOLDOUT_HISTORY variable, CLI parsing, scp |
| `synthetic_lottery_v2.json` | NEW - regenerated with correct PRNG |
| `synthetic_train_v2.json` | NEW - training portion |
| `synthetic_holdout_v2.json` | NEW - holdout portion |

---

## 6. NEXT STEPS

1. **Fix offset bug** - Add offset parameter to holdout computation
2. **Redeploy** - Push fixed worker to remote nodes
3. **Retest** - Run synthetic test, verify seed 12345 gets 1.0
4. **Production run** - Score all 100k real survivors with holdout_hits

---

## 7. VERIFICATION COMMANDS
```bash
# Local verification (works):
python3 -c "
from prng_registry import get_cpu_reference
import json

with open('synthetic_holdout_v2.json') as f:
    holdout = json.load(f)

prng = get_cpu_reference('java_lcg')
outputs = prng(12345, 5000)  # Generate all 5000
predicted = [v % 1000 for v in outputs[4000:]]  # Take positions 4000-4999

hits = sum(1 for p, a in zip(predicted, holdout) if p == a)
print(f'Seed 12345: {hits}/{len(holdout)} = {hits/len(holdout):.4f}')
"
# Output: Seed 12345: 1000/1000 = 1.0000
```
