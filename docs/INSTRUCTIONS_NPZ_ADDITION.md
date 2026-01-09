# INSTRUCTIONS.TXT ADDITION
## Binary Data Format (NPZ) - January 2026

---

**Add this section to instructions.txt:**

---

## Binary Survivor Data (NPZ Format)

### Overview

Step 2.5 (Scorer Meta-Optimizer) uses NPZ binary format for survivor data loading.
This provides 88x faster loading compared to JSON, critical for distributed GPU jobs.

### Performance Comparison

| Format | File Size | Load Time | Per-Job Overhead |
|--------|-----------|-----------|------------------|
| JSON | 258 MB | 4.2s | 4.2s × 26 GPUs = 109s |
| NPZ | 0.6 MB | 0.05s | 0.05s × 26 GPUs = 1.3s |

**Impact:** Prevents i3 CPU thrashing on mining rigs when 12 concurrent jobs parse JSON.

### Conversion Script

```bash
# One-time conversion (run on Zeus)
python3 convert_survivors_to_binary.py bidirectional_survivors.json

# Output files:
#   bidirectional_survivors_binary.npz (0.6 MB)
#   bidirectional_survivors_binary.meta.json (metadata)
```

### Format Structure

```python
# NPZ contains three arrays:
data = np.load('bidirectional_survivors_binary.npz')
seeds = data['seeds']              # uint32 array of seed values
forward_matches = data['forward_matches']   # float32 match rates
reverse_matches = data['reverse_matches']   # float32 match rates
```

### Worker Auto-Detection

`scorer_trial_worker.py` automatically detects format:

```python
def load_survivors(path):
    if path.endswith('.npz'):
        # Fast binary loading
        data = np.load(path)
        return data['seeds'], data['forward_matches'], data['reverse_matches']
    else:
        # JSON fallback for compatibility
        with open(path) as f:
            survivors = json.load(f)
        return extract_arrays_from_json(survivors)
```

### Deployment

NPZ files are copied to remote nodes during job setup:

```bash
# In run_scorer_meta_optimizer.sh
scp bidirectional_survivors_binary.npz $REMOTE:~/distributed_prng_analysis/
```

### Regeneration

If survivor data changes (new Step 1 run), regenerate NPZ:

```bash
python3 convert_survivors_to_binary.py bidirectional_survivors.json
# Then re-copy to remotes
```

---

## End of instructions.txt Addition
