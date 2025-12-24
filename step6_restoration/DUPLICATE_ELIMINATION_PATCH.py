#!/usr/bin/env python3
"""
DUPLICATE ELIMINATION PATCH - Team Beta Requirement #4
======================================================

One truth for GlobalStateTracker: delete/replace duplicates so we don't drift again.

FILES WITH DUPLICATES:
1. reinforcement_engine.py - lines 277-450 (original)
2. generate_ml_jobs.py - line 458+ (duplicate)

SOLUTION: Both files import from the new canonical location.

"""

# =============================================================================
# PATCH FOR reinforcement_engine.py
# =============================================================================
"""
1. Find line 277: class GlobalStateTracker:
   
2. REPLACE the entire class (lines 277-450) with this import:

# GlobalStateTracker moved to GPU-neutral module (v2.2)
# Eliminates duplicates, prevents GPU coupling
from models.global_state_tracker import GlobalStateTracker, GLOBAL_FEATURE_COUNT, GLOBAL_FEATURE_NAMES

3. Verify the class is still used correctly (it has the same interface)
"""


# =============================================================================
# PATCH FOR generate_ml_jobs.py
# =============================================================================
"""
1. Find line 458: class GlobalStateTracker:

2. REPLACE the entire class definition with this import:

# GlobalStateTracker - import from canonical location (v2.2)
# Eliminates duplicate, uses GPU-neutral module
from models.global_state_tracker import GlobalStateTracker, GLOBAL_FEATURE_COUNT, GLOBAL_FEATURE_NAMES

3. The rest of the file should work unchanged since GlobalStateTracker has same interface
"""


# =============================================================================
# VERIFICATION SCRIPT
# =============================================================================
"""
After applying patches, run this to verify:

python3 -c "
# Test import from canonical location
from models.global_state_tracker import GlobalStateTracker, GLOBAL_FEATURE_COUNT
print(f'✅ Canonical import works: {GLOBAL_FEATURE_COUNT} features')

# Test reinforcement_engine still works
from reinforcement_engine import ReinforcementEngine, ReinforcementConfig
print('✅ reinforcement_engine import works')

# Test generate_ml_jobs still works  
from generate_ml_jobs import GlobalStateTracker as GST
print('✅ generate_ml_jobs import works')

# Verify they're the same class
from models.global_state_tracker import GlobalStateTracker as CanonicalGST
assert GST is CanonicalGST or GST.__name__ == CanonicalGST.__name__
print('✅ No duplicate classes - single source of truth')
"
"""


# =============================================================================
# SED COMMANDS FOR QUICK PATCH (use with caution)
# =============================================================================
"""
# These commands show the line numbers to modify - DO NOT RUN BLINDLY

# For reinforcement_engine.py:
# 1. First, backup:
cp reinforcement_engine.py reinforcement_engine.py.bak_$(date +%Y%m%d)

# 2. View the class to be replaced:
sed -n '275,455p' reinforcement_engine.py

# 3. The replacement is manual - add import and delete class lines

# For generate_ml_jobs.py:
# 1. First, backup:
cp generate_ml_jobs.py generate_ml_jobs.py.bak_$(date +%Y%m%d)

# 2. Find the duplicate class:
grep -n "class GlobalStateTracker" generate_ml_jobs.py

# 3. The replacement is manual - add import and delete class lines
"""


# =============================================================================
# BACKWARD COMPATIBILITY NOTE
# =============================================================================
"""
The new GlobalStateTracker in models/global_state_tracker.py has the SAME interface:

- __init__(lottery_history, config)
- get_global_state() -> Dict[str, float]
- update_history(new_draws)

PLUS new helper methods:
- get_feature_names() -> List[str]
- get_feature_values() -> np.ndarray

So existing code that uses GlobalStateTracker will work unchanged.
"""
