#!/bin/bash
# apply_watcher_validation_patch.sh
# Applies the file validation fix to watcher_agent.py
# Version: 1.1.0 (Team Beta Approved)
# Date: 2026-01-18

set -e  # Exit on error

WATCHER_FILE="agents/watcher_agent.py"
BACKUP_FILE="agents/watcher_agent.py.backup_$(date +%Y%m%d_%H%M%S)"

echo "════════════════════════════════════════════════════════════════"
echo "  WATCHER AGENT FILE VALIDATION PATCH v1.1.0"
echo "════════════════════════════════════════════════════════════════"

# Check we're in the right directory
if [ ! -f "$WATCHER_FILE" ]; then
    echo "ERROR: $WATCHER_FILE not found!"
    echo "Run this script from ~/distributed_prng_analysis/"
    exit 1
fi

# Create backup
echo "📦 Creating backup: $BACKUP_FILE"
cp "$WATCHER_FILE" "$BACKUP_FILE"

# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 1: Add fnmatch import
# ══════════════════════════════════════════════════════════════════════════════
echo "🔧 Change 1: Adding fnmatch import..."

# Check if already patched
if grep -q "import fnmatch" "$WATCHER_FILE"; then
    echo "   ⏭️  fnmatch import already present, skipping"
else
    sed -i 's/^import signal$/import signal\nimport fnmatch/' "$WATCHER_FILE"
    echo "   ✅ Added fnmatch import"
fi

# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 2: Add FILE_VALIDATION_CONFIG and helper functions
# ══════════════════════════════════════════════════════════════════════════════
echo "🔧 Change 2: Adding validation config and functions..."

# Check if already patched
if grep -q "FILE_VALIDATION_CONFIG" "$WATCHER_FILE"; then
    echo "   ⏭️  FILE_VALIDATION_CONFIG already present, skipping"
else
    # Create the insert block in a temp file
    cat > /tmp/validation_insert.txt << 'VALIDATION_BLOCK'


# ════════════════════════════════════════════════════════════════════════════════
# FILE VALIDATION CONFIG (Team Beta Approved v1.1.0)
# ════════════════════════════════════════════════════════════════════════════════

FILE_VALIDATION_CONFIG = {
    "min_sizes": {
        ".json": 50,
        ".npz": 100,
        ".pth": 1000,
        ".pt": 1000,
        ".xgb": 1000,
        ".lgb": 1000,
        ".cbm": 1000,
        "default": 10
    },
    "json_array_minimums": {
        "bidirectional_survivors*.json": 100,
        "forward_survivors*.json": 100,
        "reverse_survivors*.json": 100,
        "survivors_with_scores*.json": 100,
        "train_history*.json": 10,
        "holdout_history*.json": 5,
    },
    "json_required_keys": {
        "optimal_window_config*.json": ["window_size", "offset"],
        "optimal_scorer_config*.json": ["best_trial"],
        "best_model*.meta.json": ["model_type", "feature_schema"],
        "reinforcement_engine_config*.json": ["survivor_count"],
        "predictions*.json": ["predictions"],
        "prediction_pool*.json": ["predictions"],
    }
}


def _get_min_file_size(filepath: str) -> int:
    """Get minimum expected file size based on extension."""
    ext = Path(filepath).suffix.lower()
    return FILE_VALIDATION_CONFIG["min_sizes"].get(
        ext, FILE_VALIDATION_CONFIG["min_sizes"]["default"]
    )


def _match_config_by_pattern(filename: str, table: dict):
    """Match filename against pattern-based config table using fnmatch."""
    for pattern, value in table.items():
        if fnmatch.fnmatch(filename, pattern):
            return value
    return None


def _validate_json_structure(filepath: str) -> tuple:
    """Validate JSON file has meaningful content. Returns (is_valid, reason)."""
    filename = Path(filepath).name
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Read error: {e}"

    if data is None:
        return False, "JSON contains null"

    if isinstance(data, list):
        if len(data) == 0:
            return False, "JSON array is empty"
        min_count = _match_config_by_pattern(
            filename, FILE_VALIDATION_CONFIG["json_array_minimums"]
        )
        if min_count and len(data) < min_count:
            return False, f"JSON array has {len(data)} items, expected >= {min_count}"

    elif isinstance(data, dict):
        if len(data) == 0:
            return False, "JSON object is empty"
        required_keys = _match_config_by_pattern(
            filename, FILE_VALIDATION_CONFIG["json_required_keys"]
        )
        if required_keys:
            missing_keys = [k for k in required_keys if k not in data]
            if missing_keys:
                return False, f"Missing required keys: {missing_keys}"

    return True, "Valid JSON structure"


def _validate_file_content(filepath: str) -> tuple:
    """Validate file content based on type. Returns (is_valid, reason)."""
    ext = Path(filepath).suffix.lower()

    if ext == ".json":
        return _validate_json_structure(filepath)

    elif ext == ".npz":
        try:
            import numpy as np
            with np.load(filepath, mmap_mode="r") as data:
                if len(data.files) == 0:
                    return False, "NPZ contains no arrays"
                for key in data.files:
                    if data[key].size > 0:
                        return True, f"NPZ valid with {len(data.files)} arrays"
                return False, "All NPZ arrays are empty"
        except Exception as e:
            return False, f"NPZ read error: {e}"

    elif ext in [".pth", ".pt"]:
        try:
            with open(filepath, "rb") as f:
                magic = f.read(16)
            if len(magic) < 16:
                return False, "PyTorch file too small"
            return True, "PyTorch checkpoint present (not deserialized)"
        except Exception as e:
            return False, f"PyTorch file read error: {e}"

    elif ext in [".xgb", ".lgb", ".cbm"]:
        try:
            with open(filepath, "rb") as f:
                magic = f.read(16)
            if len(magic) < 16:
                return False, f"{ext} model file too small"
            return True, f"{ext} model present (not deserialized)"
        except Exception as e:
            return False, f"{ext} file read error: {e}"

    return True, "Content validation skipped (unknown type)"


def evaluate_file_exists(filepath: str, validate_content: bool = True) -> tuple:
    """
    Evaluate if file exists AND has meaningful content.
    Returns (success, reason) tuple.
    """
    if not os.path.exists(filepath):
        return False, f"File does not exist: {filepath}"

    if not os.path.isfile(filepath):
        return False, f"Path is not a file: {filepath}"

    file_size = os.path.getsize(filepath)
    min_size = _get_min_file_size(filepath)

    if file_size < min_size:
        return False, f"File too small: {file_size} bytes (min: {min_size})"

    if validate_content:
        is_valid, content_reason = _validate_file_content(filepath)
        if not is_valid:
            return False, f"Content validation failed: {content_reason}"
        return True, f"Valid file: {file_size} bytes - {content_reason}"

    return True, f"Valid file: {file_size} bytes"

VALIDATION_BLOCK

    # Insert after the doctrine import line
    # Using Python for reliable multi-line insert
    python3 << 'PYINSERT'
import re

with open("agents/watcher_agent.py", "r") as f:
    content = f.read()

# Find the doctrine import line and insert after it
insert_marker = "from agents.doctrine import validate_decision_against_doctrine"
with open("/tmp/validation_insert.txt", "r") as f:
    insert_block = f.read()

if insert_marker in content:
    content = content.replace(insert_marker, insert_marker + insert_block)
    with open("agents/watcher_agent.py", "w") as f:
        f.write(content)
    print("   ✅ Added validation config and functions")
else:
    print("   ❌ Could not find insert marker!")
    exit(1)
PYINSERT
fi

# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 3: Replace weak file check with robust validation
# ══════════════════════════════════════════════════════════════════════════════
echo "🔧 Change 3: Replacing file existence check with validation..."

# Check if already patched
if grep -q "evaluate_file_exists(p)" "$WATCHER_FILE"; then
    echo "   ⏭️  File validation already integrated, skipping"
else
    python3 << 'PYREPLACE'
with open("agents/watcher_agent.py", "r") as f:
    content = f.read()

# Old pattern to find
old_block = '''                missing = [p for p in required_files if not Path(p).exists()]
                success = not missing

                logger.info(f"File-based evaluation: {required_files} -> {'all exist' if success else f'missing: {missing}'}")'''

# New replacement
new_block = '''                # Validate files exist AND have meaningful content (v1.1.0 fix)
                validation_results = []
                for p in required_files:
                    is_valid, reason = evaluate_file_exists(p)
                    validation_results.append({"file": p, "valid": is_valid, "reason": reason})
                
                failed = [r for r in validation_results if not r["valid"]]
                success = len(failed) == 0

                if success:
                    logger.info(f"File-based evaluation: all {len(required_files)} files valid")
                else:
                    for f in failed:
                        logger.warning(f"File validation failed: {f['file']} - {f['reason']}")'''

if old_block in content:
    content = content.replace(old_block, new_block)
    with open("agents/watcher_agent.py", "w") as f:
        f.write(content)
    print("   ✅ Replaced file existence check")
else:
    print("   ⚠️  Could not find exact old pattern - may need manual review")
PYREPLACE
fi

# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 4: Update reasoning message
# ══════════════════════════════════════════════════════════════════════════════
echo "🔧 Change 4: Updating reasoning message..."

python3 << 'PYREASON'
with open("agents/watcher_agent.py", "r") as f:
    content = f.read()

old_reason = 'reasoning="All required files exist" if success else f"Missing required files: {missing}"'
new_reason = 'reasoning="All required files valid" if success else f"File validation failed: {[r[\\'file\\'] + \\': \\' + r[\\'reason\\'] for r in failed]}"'

if old_reason in content:
    content = content.replace(old_reason, new_reason)
    with open("agents/watcher_agent.py", "w") as f:
        f.write(content)
    print("   ✅ Updated reasoning message")
elif "All required files valid" in content:
    print("   ⏭️  Reasoning already updated, skipping")
else:
    print("   ⚠️  Could not find reasoning pattern - may need manual review")
PYREASON

# ══════════════════════════════════════════════════════════════════════════════
# VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  VERIFICATION"
echo "════════════════════════════════════════════════════════════════"

echo "Checking for fnmatch import:"
grep -n "import fnmatch" "$WATCHER_FILE" || echo "   ❌ NOT FOUND"

echo ""
echo "Checking for FILE_VALIDATION_CONFIG:"
grep -n "FILE_VALIDATION_CONFIG = {" "$WATCHER_FILE" | head -1 || echo "   ❌ NOT FOUND"

echo ""
echo "Checking for evaluate_file_exists function:"
grep -n "def evaluate_file_exists" "$WATCHER_FILE" || echo "   ❌ NOT FOUND"

echo ""
echo "Checking for validation integration:"
grep -n "evaluate_file_exists(p)" "$WATCHER_FILE" || echo "   ❌ NOT FOUND"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  PATCH COMPLETE"
echo "════════════════════════════════════════════════════════════════"
echo "Backup saved to: $BACKUP_FILE"
echo ""
echo "To verify syntax:"
echo "  python3 -m py_compile agents/watcher_agent.py"
echo ""
echo "To revert if needed:"
echo "  cp $BACKUP_FILE agents/watcher_agent.py"
echo "════════════════════════════════════════════════════════════════"
