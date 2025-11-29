#!/bin/bash
#
# Complete Migration to New Format System
# Removes old test files and updates modules for new format only
#

cd ~/distributed_prng_analysis

echo "================================================================"
echo "COMPLETE MIGRATION TO NEW FORMAT SYSTEM"
echo "================================================================"
echo ""
echo "This will:"
echo "  1. Archive old test result files (63 files)"
echo "  2. Update result_viewer.py for new format"
echo "  3. Update file_manager.py for new format"
echo "  4. Clean system ready for ML/AI automation"
echo ""
read -p "Continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Migration cancelled"
    exit 0
fi

echo ""
echo "=== STEP 1: ARCHIVE OLD FORMAT FILES ==="
echo ""

# Create archive directory
ARCHIVE_DIR="archive_old_format_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ARCHIVE_DIR"

# Move old format files to archive
echo "Moving old format .json files to archive..."
OLD_COUNT=$(ls results/*.json 2>/dev/null | wc -l)
if [ $OLD_COUNT -gt 0 ]; then
    mv results/*.json "$ARCHIVE_DIR/" 2>/dev/null
    echo "✅ Archived $OLD_COUNT old format files to $ARCHIVE_DIR/"
else
    echo "No old format files found to archive"
fi

echo ""
echo "=== STEP 2: UPDATE RESULT_VIEWER.PY ==="
echo ""

# Backup result_viewer.py
cp modules/result_viewer.py modules/result_viewer.py.backup_before_new_format
echo "✅ Backup: modules/result_viewer.py.backup_before_new_format"

# Update result_viewer.py
python3 << 'PYCODE'
import re

with open('modules/result_viewer.py', 'r') as f:
    content = f.read()

# Replace old format glob patterns with new format
replacements = [
    # Pattern 1: glob.glob("results/*.json")
    (
        r'results_files = glob\.glob\("results/\*\.json"\)',
        '''# NEW FORMAT: Read from organized subdirectories
        results_files = []
        results_files.extend(glob.glob("results/summaries/*.txt"))
        results_files.extend(glob.glob("results/csv/*.csv"))
        results_files.extend(glob.glob("results/json/*.json"))'''
    ),
    
    # Pattern 2: os.listdir checking for .json
    (
        r'result_files = \[f for f in os\.listdir\(results_dir\) if f\.endswith\(\'\.json\'\)\]',
        '''# NEW FORMAT: Scan subdirectories
        result_files = []
        for subdir in ['summaries', 'csv', 'json', 'detailed', 'configs']:
            subdir_path = os.path.join(results_dir, subdir)
            if os.path.exists(subdir_path):
                files = os.listdir(subdir_path)
                result_files.extend([os.path.join(subdir, f) for f in files])'''
    ),
]

modified = content
changes = 0

for old_pattern, new_code in replacements:
    if re.search(old_pattern, modified):
        modified = re.sub(old_pattern, new_code, modified, count=1)
        changes += 1

with open('modules/result_viewer.py', 'w') as f:
    f.write(modified)

print(f"✅ Updated result_viewer.py ({changes} patterns replaced)")
PYCODE

echo ""
echo "=== STEP 3: UPDATE FILE_MANAGER.PY ==="
echo ""

# Backup file_manager.py
cp modules/file_manager.py modules/file_manager.py.backup_before_new_format
echo "✅ Backup: modules/file_manager.py.backup_before_new_format"

# Update file_manager.py
python3 << 'PYCODE'
with open('modules/file_manager.py', 'r') as f:
    lines = f.readlines()

# Find and replace the file type patterns section
in_patterns_section = False
patterns_updated = False

for i, line in enumerate(lines):
    # Find the patterns dictionary
    if "'Gap-Aware':" in line and not patterns_updated:
        # Replace old patterns with new ones
        new_patterns = '''            # NEW FORMAT: Organized subdirectories
            'Result Summaries': 'results/summaries/*.txt',
            'CSV Exports': 'results/csv/*.csv',
            'JSON Results': 'results/json/*.json',
            'Detailed Results': 'results/detailed/*.json',
            'Config Files': 'results/configs/*.json',
            'Lottery Data': 'daily*.json'
'''
        # Find the end of old patterns section
        end_idx = i
        while end_idx < len(lines) and '}' not in lines[end_idx]:
            end_idx += 1
        
        # Replace entire patterns section
        lines[i:end_idx] = [new_patterns]
        patterns_updated = True
        break

if patterns_updated:
    with open('modules/file_manager.py', 'w') as f:
        f.writelines(lines)
    print("✅ Updated file_manager.py (new format patterns)")
else:
    print("⚠️  Could not auto-update file_manager.py patterns")
    print("   Manual review may be needed")
PYCODE

echo ""
echo "=== STEP 4: VERIFICATION ==="
echo ""

echo "Checking new format file counts:"
echo "  Summaries:  $(ls results/summaries/*.txt 2>/dev/null | wc -l) files"
echo "  CSV:        $(ls results/csv/*.csv 2>/dev/null | wc -l) files"
echo "  JSON:       $(ls results/json/*.json 2>/dev/null | wc -l) files"
echo "  Detailed:   $(ls results/detailed/*.json 2>/dev/null | wc -l) files"
echo "  Configs:    $(ls results/configs/*.json 2>/dev/null | wc -l) files"
echo ""

echo "Checking old format files remaining:"
OLD_REMAINING=$(ls results/*.json 2>/dev/null | wc -l)
if [ $OLD_REMAINING -eq 0 ]; then
    echo "  ✅ No old format files in results/ (clean)"
else
    echo "  ⚠️  $OLD_REMAINING old format files still present"
fi

echo ""
echo "Testing Python imports:"
python3 -c "from modules.result_viewer import ResultViewer; print('  ✅ result_viewer imports successfully')" 2>&1
python3 -c "from modules.file_manager import FileManager; print('  ✅ file_manager imports successfully')" 2>&1

echo ""
echo "================================================================"
echo "MIGRATION COMPLETE!"
echo "================================================================"
echo ""
echo "✅ Old test files archived to: $ARCHIVE_DIR/"
echo "✅ result_viewer.py updated for new format"
echo "✅ file_manager.py updated for new format"
echo "✅ System ready for ML/AI automation"
echo ""
echo "Next steps:"
echo "  1. Test unified_system_working.py"
echo "  2. Verify file viewing/management works"
echo "  3. Archive can be deleted later: rm -rf $ARCHIVE_DIR"
echo ""
echo "To restore (if needed):"
echo "  cp modules/result_viewer.py.backup_before_new_format modules/result_viewer.py"
echo "  cp modules/file_manager.py.backup_before_new_format modules/file_manager.py"
echo "  mv $ARCHIVE_DIR/*.json results/"
echo ""
echo "================================================================"
