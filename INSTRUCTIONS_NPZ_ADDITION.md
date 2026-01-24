# INSTRUCTIONS.TXT ADDITION
## Binary Data Format (NPZ) - Updated January 23, 2026

---

**Add this section to instructions.txt:**

---

## Binary Survivor Data (NPZ Format v3.0)

### Overview

Steps 2 and 3 use NPZ binary format for survivor data loading.
This provides ~80x faster loading and ~80x smaller files compared to JSON.

**Version History:**
| Version | Date | Arrays | Notes |
|---------|------|--------|-------|
| v1.0 | Jan 3, 2026 | 3 | Initial: seeds, forward_matches, reverse_matches |
| v2.0 | Jan 19, 2026 | 3 | Added --output flag |
| v3.0 | Jan 23, 2026 | 22 | **CRITICAL FIX:** Full metadata preservation |

### Performance Comparison

| Format | File Size | Load Time | Arrays |
|--------|-----------|-----------|--------|
| JSON | 57.9 MB | 4.2s | 22 fields |
| NPZ v2.0 | 0.6 MB | 0.05s | 3 arrays ❌ (data loss) |
| NPZ v3.0 | 733 KB | 0.05s | 22 arrays ✅ |

**Impact:** Prevents CPU thrashing on mining rigs + preserves all ML metadata.

### Conversion Script

```bash
# One-time conversion (run on Zeus)
python3 convert_survivors_to_binary.py bidirectional_survivors.json

# Output files:
#   bidirectional_survivors_binary.npz (~733 KB)
#   bidirectional_survivors_binary.meta.json (conversion metadata)
```

### NPZ v3.0 Format Structure

```python
import numpy as np
data = np.load('bidirectional_survivors_binary.npz')

# === Core arrays (v1.0 compatible) ===
data['seeds']              # uint32 - seed values
data['forward_matches']    # float32 - forward match rates
data['reverse_matches']    # float32 - reverse match rates

# === Metadata arrays (v3.0 addition) ===
# Integer fields
data['window_size']        # int32
data['offset']             # int32
data['trial_number']       # int32
data['skip_min']           # int32
data['skip_max']           # int32
data['skip_range']         # int32

# Float fields (sieve counts & metrics)
data['forward_count']      # float32
data['reverse_count']      # float32
data['bidirectional_count']       # float32
data['intersection_count']        # float32
data['intersection_ratio']        # float32
data['intersection_weight']       # float32
data['bidirectional_selectivity'] # float32
data['forward_only_count']        # float32
data['reverse_only_count']        # float32
data['survivor_overlap_ratio']    # float32
data['score']              # float32

# Categorical fields (encoded as integers)
data['skip_mode']          # uint8 (0=constant, 1=variable)
data['prng_type']          # uint8 (0=java_lcg, 1=java_lcg_reverse, ...)
```

### Categorical Encodings

**skip_mode:**
| Value | Meaning |
|-------|---------|
| 0 | constant |
| 1 | variable |

**prng_type:**
| Value | Meaning |
|-------|---------|
| 0 | java_lcg |
| 1 | java_lcg_reverse |
| 2 | mt19937 |
| 3 | mt19937_reverse |
| 4 | xorshift128 |
| 5 | xorshift128_reverse |
| ... | ... |

### Modular Loader (utils/survivor_loader.py)

The loader auto-detects NPZ version and format:

```python
from utils.survivor_loader import load_survivors

# Auto-detect format (NPZ or JSON)
result = load_survivors('bidirectional_survivors_binary.npz')

print(f"Format: {result.format}")        # 'npz'
print(f"NPZ Version: {result.npz_version}")  # 3
print(f"Count: {result.count}")          # 98172
print(f"Fallback: {result.fallback_used}")  # False

# Access data
survivors = result.data  # Dict of numpy arrays

# Force conversion to list of dicts (for Step 3 metadata merge)
result = load_survivors(path, return_format="dict")
for survivor in result.data:
    print(f"Seed {survivor['seed']}: skip_min={survivor['skip_min']}")
```

### Version Detection

The loader automatically detects NPZ version:

```python
from utils.survivor_loader import detect_npz_version

data = np.load('bidirectional_survivors_binary.npz')
version = detect_npz_version(data)  # Returns 1 or 3

# v1: Only has seeds, forward_matches, reverse_matches
# v3: Has all 22 metadata fields
```

### Convenience Functions

```python
from utils.survivor_loader import get_survivor_count, get_survivor_metadata

# Quick count without loading full data
count = get_survivor_count('bidirectional_survivors_binary.npz')

# Get metadata indexed by seed (useful for Step 3)
metadata = get_survivor_metadata('bidirectional_survivors_binary.npz')
# Returns: {seed: {'skip_min': ..., 'forward_count': ..., ...}, ...}
```

### Deployment

NPZ files are copied to remote nodes during job setup:

```bash
# In run_scorer_meta_optimizer.sh / run_step3_full_scoring.sh
scp bidirectional_survivors_binary.npz $REMOTE:~/distributed_prng_analysis/
scp utils/survivor_loader.py $REMOTE:~/distributed_prng_analysis/utils/
```

### Regeneration

If survivor data changes (new Step 1 run), regenerate NPZ:

```bash
python3 convert_survivors_to_binary.py bidirectional_survivors.json
# Then re-copy to remotes:
scp bidirectional_survivors_binary.npz 192.168.3.120:~/distributed_prng_analysis/
scp bidirectional_survivors_binary.npz 192.168.3.154:~/distributed_prng_analysis/
```

### Critical Note: v3.0 Requirement for Step 3

**Step 3 (Full Scoring) REQUIRES NPZ v3.0** to compute all 47 ML features.

NPZ v2.0 or earlier will cause **14 features to be silently zeroed**:
- skip_min, skip_max, skip_range
- forward_count, reverse_count, bidirectional_count
- intersection_count, intersection_ratio, intersection_weight
- forward_only_count, reverse_only_count
- survivor_overlap_ratio, bidirectional_selectivity

Always verify NPZ version before Step 3:

```bash
python3 -c "
import numpy as np
data = np.load('bidirectional_survivors_binary.npz')
print(f'Arrays: {len(data.keys())}')
print('✅ v3.0' if len(data.keys()) >= 20 else '❌ v1/v2 - REGENERATE!')
"
```

---

## End of instructions.txt Addition

---

## Scripts Using Modular Loader

The following scripts import `utils.survivor_loader` for consistent NPZ/JSON handling:

| Script | Step | Import |
|--------|------|--------|
| `generate_full_scoring_jobs.py` | Step 3 | `from utils.survivor_loader import load_survivors` |
| `generate_step3_scoring_jobs.py` | Step 3 | `from utils.survivor_loader import load_survivors` |
| `scorer_trial_worker.py` | Step 2.5 | Auto-detects format via internal loader |
| `full_scoring_worker.py` | Step 3 | Uses chunk data with metadata reconstruction |

### Integration Pattern

All scripts follow this pattern for format-agnostic loading:

```python
from utils.survivor_loader import load_survivors

# Auto-detects NPZ vs JSON
result = load_survivors(args.survivors)

# Access data
if result.format == "npz":
    seeds = result.data['seeds']
    # For Step 3, request dict format for metadata:
    result = load_survivors(args.survivors, return_format="dict")
    for survivor in result.data:
        print(f"Seed {survivor['seed']}: skip_min={survivor['skip_min']}")
```

### Adding the Loader to New Scripts

```python
# 1. Add import at top of file
from utils.survivor_loader import load_survivors

# 2. Replace direct json.load() calls:
# OLD (breaks on NPZ):
with open(survivors_file) as f:
    survivors = json.load(f)

# NEW (format-agnostic):
result = load_survivors(survivors_file)
survivors = result.data
```

### WATCHER Visibility

The loader returns metadata for WATCHER agent visibility:

```python
result = load_survivors(path)
print(f"Format: {result.format}")        # 'npz' or 'json'
print(f"NPZ Version: {result.npz_version}")  # 1 or 3 (None for JSON)
print(f"Count: {result.count}")          # Number of survivors
print(f"Fallback: {result.fallback_used}")  # True if fell back to alternate format
```

This enables WATCHER to detect format mismatches or fallback situations during pipeline execution.
