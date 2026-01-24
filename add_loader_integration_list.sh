#!/bin/bash
# add_loader_integration_list.sh
# Adds "Scripts Using Modular Loader" section to INSTRUCTIONS_NPZ_ADDITION.md
#
# USAGE:
#   cd ~/distributed_prng_analysis
#   bash add_loader_integration_list.sh

set -e
cd ~/distributed_prng_analysis

echo "Adding modular loader integration list to INSTRUCTIONS_NPZ_ADDITION.md..."

# Append section to INSTRUCTIONS_NPZ_ADDITION.md
cat >> INSTRUCTIONS_NPZ_ADDITION.md << 'APPEND_SECTION'

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
APPEND_SECTION

echo "✓ Section appended to INSTRUCTIONS_NPZ_ADDITION.md"

# Verify
echo ""
echo "Verification (last 20 lines):"
tail -20 INSTRUCTIONS_NPZ_ADDITION.md

# Git
echo ""
echo "Next: git add and commit"
