#!/bin/bash
# patch_step3_chunk_metadata_v2.sh
# Team Beta Approved: January 23, 2026
#
# ISSUE: Step 3 chunks contain only seed integers, metadata discarded
# FIX: Write full survivor objects to chunks
# SCOPE: generate_step3_scoring_jobs.py only
#
# USAGE:
#   cd ~/distributed_prng_analysis
#   bash patch_step3_chunk_metadata_v2.sh

set -e
cd ~/distributed_prng_analysis

echo "=============================================="
echo "STEP 3 CHUNK METADATA FIX"
echo "Team Beta Approved: January 23, 2026"
echo "=============================================="

# Backup
cp generate_step3_scoring_jobs.py generate_step3_scoring_jobs.py.bak
echo "✓ Backed up to generate_step3_scoring_jobs.py.bak"

# Apply patch using Python for reliability
python3 << 'PATCH_SCRIPT'
import re

with open('generate_step3_scoring_jobs.py', 'r') as f:
    content = f.read()

# ============================================================
# PATCH 1: Add extract_survivors_full() after extract_seeds()
# ============================================================

old_extract_seeds = '''def extract_seeds(survivors_data: Any) -> List[int]:
    """
    Extract seed values from various input formats.
    
    Supports:
    - Flat list: [12345, 67890, ...]
    - Object list: [{"seed": 12345}, {"candidate_seed": 67890}, ...]
    """
    # Handle NPZ dict format with 'seeds' key
    if isinstance(survivors_data, dict) and 'seeds' in survivors_data:
        return [int(s) for s in survivors_data['seeds']]
    
    if not survivors_data:
        return []
    
    if isinstance(survivors_data[0], dict):
        seeds = []
        for s in survivors_data:
            seed = s.get('seed', s.get('candidate_seed', s.get('survivor_seed')))
            if seed is not None:
                seeds.append(int(seed))
        return seeds
    else:
        return [int(s) for s in survivors_data]'''

new_extract_functions = '''def extract_seeds(survivors_data: Any) -> List[int]:
    """
    Extract seed values only (for backward compatibility).
    For Step 3, use extract_survivors_full() instead.
    """
    survivors = extract_survivors_full(survivors_data)
    return [s['seed'] for s in survivors]


def extract_survivors_full(survivors_data: Any) -> List[Dict]:
    """
    Extract full survivor objects with all metadata.
    
    CRITICAL (Jan 23, 2026 - Team Beta):
    Previous version discarded metadata, causing 14/47 ML features = 0.
    Now preserves all fields from NPZ/JSON for chunk files.
    
    Returns:
        List of dicts, each with 'seed' + all metadata fields
    """
    result = []
    
    # Handle NPZ dict format with arrays
    if isinstance(survivors_data, dict) and 'seeds' in survivors_data:
        seeds = survivors_data['seeds']
        n = len(seeds)
        keys = list(survivors_data.keys())
        
        for i in range(n):
            obj = {}
            for k in keys:
                arr = survivors_data[k]
                if hasattr(arr, '__len__') and len(arr) == n:
                    val = arr[i]
                    # Convert numpy types to Python native
                    if hasattr(val, 'item'):
                        val = val.item()
                    # Rename 'seeds' to 'seed' for consistency
                    key = 'seed' if k == 'seeds' else k
                    obj[key] = val
            result.append(obj)
        
        # GUARDRAIL: Fail if metadata was dropped
        if result and len(result[0]) < 3:
            raise ValueError(
                f"METADATA LOSS DETECTED: Survivors have only {len(result[0])} fields. "
                f"Expected 20+. Check NPZ version or loader."
            )
        
        return result
    
    # Handle list of dicts (JSON format)
    if isinstance(survivors_data, list) and survivors_data:
        if isinstance(survivors_data[0], dict):
            for s in survivors_data:
                obj = dict(s)
                if 'candidate_seed' in obj and 'seed' not in obj:
                    obj['seed'] = obj['candidate_seed']
                result.append(obj)
            return result
        else:
            # Plain integers - convert to minimal dicts
            return [{'seed': int(s)} for s in survivors_data]
    
    return result'''

content = content.replace(old_extract_seeds, new_extract_functions)

# ============================================================
# PATCH 2: Use extract_survivors_full() for chunk creation
# ============================================================

old_chunk_logic = '''    seeds = extract_seeds(survivors_data)
    print(f"  Loaded {len(seeds)} survivor seeds")
    
    if not seeds:
        raise ValueError("No survivor seeds found in input file")'''

new_chunk_logic = '''    # Extract full survivor objects (with metadata) - NOT just seeds
    survivors_full = extract_survivors_full(survivors_data)
    seeds = [s['seed'] for s in survivors_full]
    print(f"  Loaded {len(survivors_full)} survivors with metadata")
    
    # Verify metadata is present (guardrail)
    if survivors_full:
        field_count = len(survivors_full[0])
        sample_keys = list(survivors_full[0].keys())[:6]
        print(f"  Fields per survivor: {field_count}")
        print(f"  Sample keys: {sample_keys}")
        if field_count < 5:
            print(f"  ⚠️  WARNING: Low field count - metadata may be missing!")
    
    if not survivors_full:
        raise ValueError("No survivor seeds found in input file")'''

content = content.replace(old_chunk_logic, new_chunk_logic)

# ============================================================
# PATCH 3: Split survivors_full into chunks (not seeds)
# ============================================================

old_split = '''    # Split seeds into chunks
    seed_chunks = chunk_list(seeds, chunk_size)
    num_chunks = len(seed_chunks)
    print(f"Split {len(seeds)} seeds into {num_chunks} chunks of ~{chunk_size} each")'''

new_split = '''    # Split full survivor objects into chunks (preserves metadata)
    survivor_chunks = chunk_list(survivors_full, chunk_size)
    num_chunks = len(survivor_chunks)
    print(f"Split {len(survivors_full)} survivors into {num_chunks} chunks of ~{chunk_size} each")'''

content = content.replace(old_split, new_split)

# ============================================================
# PATCH 4: Update chunk iteration variable
# ============================================================

old_loop = '''    for i, chunk in enumerate(seed_chunks):'''
new_loop = '''    for i, chunk in enumerate(survivor_chunks):'''
content = content.replace(old_loop, new_loop)

# ============================================================
# PATCH 5: Update seed_count in job spec
# ============================================================

old_seed_count = '''        "seed_count": len(chunk),'''
new_seed_count = '''        "seed_count": len(chunk),  # Now counting full survivor objects'''
content = content.replace(old_seed_count, new_seed_count)

# ============================================================
# PATCH 6: Add Dict to type imports if missing
# ============================================================

if 'from typing import' in content and 'Dict' not in content.split('from typing import')[1].split('\n')[0]:
    content = content.replace('from typing import Any, List', 'from typing import Any, List, Dict')

# Write patched file
with open('generate_step3_scoring_jobs.py', 'w') as f:
    f.write(content)

print("✓ Patched generate_step3_scoring_jobs.py")
PATCH_SCRIPT

echo ""
echo "Verifying patch..."

# Verification checks
grep -q "def extract_survivors_full" generate_step3_scoring_jobs.py && echo "✓ extract_survivors_full() added"
grep -q "survivor_chunks = chunk_list" generate_step3_scoring_jobs.py && echo "✓ survivor_chunks used for chunking"
grep -q "METADATA LOSS DETECTED" generate_step3_scoring_jobs.py && echo "✓ Guardrail assertion added"
grep -q "Fields per survivor" generate_step3_scoring_jobs.py && echo "✓ Field count logging added"

echo ""
echo "=============================================="
echo "PATCH COMPLETE"
echo "=============================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Regenerate chunks:"
echo "   python3 generate_step3_scoring_jobs.py \\"
echo "       --survivors bidirectional_survivors_binary.npz \\"
echo "       --train-history train_history.json \\"
echo "       --holdout-history holdout_history.json \\"
echo "       --config optimal_scorer_config.json"
echo ""
echo "2. Verify chunk has metadata:"
echo "   python3 -c \"import json; d=json.load(open('scoring_chunks/chunk_0000.json')); print(f'Type: {type(d[0])}, Fields: {len(d[0]) if isinstance(d[0],dict) else 0}, skip_min: {d[0].get(\\\"skip_min\\\", \\\"MISSING\\\") if isinstance(d[0],dict) else \\\"N/A\\\"}' )\""
echo ""
echo "3. Re-run Step 3:"
echo "   PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 3 --end-step 3"
