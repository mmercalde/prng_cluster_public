#!/bin/bash
set -e
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_DIR="archives/cleanup_${TIMESTAMP}"
TRASH_DIR="trash_${TIMESTAMP}"

echo "=============================================="
echo "   PRNG Project Cleanup (CONSERVATIVE)"
echo "=============================================="

if [[ ! -f "coordinator.py" ]] || [[ ! -f "prng_registry.py" ]]; then
    echo "ERROR: Must run from ~/distributed_prng_analysis/"
    exit 1
fi

echo "Archive: $ARCHIVE_DIR"
echo "Trash:   $TRASH_DIR"
echo ""
echo "Press ENTER to continue or Ctrl+C to abort..."
read

mkdir -p "$ARCHIVE_DIR"/{backups,golden,logs,old_dirs}
mkdir -p "$TRASH_DIR"/{fix_scripts,debug_scripts,check_scripts,chunks,temp_jobs,txt_snippets,misc}

BEFORE_COUNT=$(ls -1 2>/dev/null | wc -l)
echo "Files before: $BEFORE_COUNT"

echo "[1/12] Archiving .backup* files..."
find . -maxdepth 1 -name "*.backup*" -type f -exec mv {} "$ARCHIVE_DIR/backups/" \; 2>/dev/null || true

echo "[2/12] Archiving .bak* files..."
find . -maxdepth 1 -name "*.bak*" -type f -exec mv {} "$ARCHIVE_DIR/backups/" \; 2>/dev/null || true

echo "[3/12] Archiving .back files..."
find . -maxdepth 1 -name "*.back" -type f -exec mv {} "$ARCHIVE_DIR/backups/" \; 2>/dev/null || true

echo "[4/12] Archiving parenthetical versions..."
find . -maxdepth 1 -name "*\(*\)*" -type f -exec mv {} "$ARCHIVE_DIR/backups/" \; 2>/dev/null || true

echo "[5/12] Archiving golden/working/broken snapshots..."
find . -maxdepth 1 -name "*.golden_*" -type f -exec mv {} "$ARCHIVE_DIR/golden/" \; 2>/dev/null || true
find . -maxdepth 1 -name "*.working_*" -type f -exec mv {} "$ARCHIVE_DIR/golden/" \; 2>/dev/null || true
find . -maxdepth 1 -name "*.broken_*" -type f -exec mv {} "$ARCHIVE_DIR/golden/" \; 2>/dev/null || true
find . -maxdepth 1 -name "*WORKING*" -type f ! -name "unified_system_working.py" -exec mv {} "$ARCHIVE_DIR/golden/" \; 2>/dev/null || true

echo "[6/12] Archiving log files..."
find . -maxdepth 1 -name "*.log" -type f -exec mv {} "$ARCHIVE_DIR/logs/" \; 2>/dev/null || true

echo "[7/12] Archiving old backup directories..."
for dir in backup_20* backups_progress old_*_backup_* results_backup_* archive_old_format_*; do
    if [[ -d "$dir" ]]; then
        mv "$dir" "$ARCHIVE_DIR/old_dirs/" 2>/dev/null || true
    fi
done

echo "[8/12] Moving fix/add/insert/patch scripts to trash..."
find . -maxdepth 1 -name "fix_*.py" -type f -exec mv {} "$TRASH_DIR/fix_scripts/" \; 2>/dev/null || true
find . -maxdepth 1 -name "add_*.py" -type f -exec mv {} "$TRASH_DIR/fix_scripts/" \; 2>/dev/null || true
find . -maxdepth 1 -name "insert_*.py" -type f -exec mv {} "$TRASH_DIR/fix_scripts/" \; 2>/dev/null || true
find . -maxdepth 1 -name "patch_*.py" -type f -exec mv {} "$TRASH_DIR/fix_scripts/" \; 2>/dev/null || true
find . -maxdepth 1 -name "update_*.py" -type f -exec mv {} "$TRASH_DIR/fix_scripts/" \; 2>/dev/null || true

echo "[9/12] Moving debug_*.py to trash..."
find . -maxdepth 1 -name "debug_*.py" -type f -exec mv {} "$TRASH_DIR/debug_scripts/" \; 2>/dev/null || true

echo "[10/12] Moving check/trace/diagnose scripts to trash..."
find . -maxdepth 1 -name "check_*.py" -type f -exec mv {} "$TRASH_DIR/check_scripts/" \; 2>/dev/null || true
find . -maxdepth 1 -name "trace_*.py" -type f -exec mv {} "$TRASH_DIR/check_scripts/" \; 2>/dev/null || true
find . -maxdepth 1 -name "diagnose_*.py" -type f -exec mv {} "$TRASH_DIR/check_scripts/" \; 2>/dev/null || true

echo "[11/12] Moving chunk/temp files to trash..."
find . -maxdepth 1 -name "chunk_scoring_seeds_*.json" -type f -exec mv {} "$TRASH_DIR/chunks/" \; 2>/dev/null || true
find . -maxdepth 1 -name "job_debug_*.json" -type f -exec mv {} "$TRASH_DIR/temp_jobs/" \; 2>/dev/null || true
find . -maxdepth 1 -name "job_seq_*.json" -type f -exec mv {} "$TRASH_DIR/temp_jobs/" \; 2>/dev/null || true
find . -maxdepth 1 -name "job_stagger_*.json" -type f -exec mv {} "$TRASH_DIR/temp_jobs/" \; 2>/dev/null || true
find . -maxdepth 1 -name "job_test_*.json" -type f -exec mv {} "$TRASH_DIR/temp_jobs/" \; 2>/dev/null || true

echo "[12/12] Moving text snippets to trash..."
find . -maxdepth 1 -name "*_registry_entries.txt" -type f -exec mv {} "$TRASH_DIR/txt_snippets/" \; 2>/dev/null || true
find . -maxdepth 1 -name "*_reverse_kernels.txt" -type f -exec mv {} "$TRASH_DIR/txt_snippets/" \; 2>/dev/null || true
find . -maxdepth 1 -name "*_insert.txt" -type f -exec mv {} "$TRASH_DIR/txt_snippets/" \; 2>/dev/null || true

AFTER_COUNT=$(ls -1 2>/dev/null | wc -l)
ARCHIVED=$(find "$ARCHIVE_DIR" -type f 2>/dev/null | wc -l)
TRASHED=$(find "$TRASH_DIR" -type f 2>/dev/null | wc -l)

echo ""
echo "=============================================="
echo "   DONE!"
echo "=============================================="
echo "Before: $BEFORE_COUNT | After: $AFTER_COUNT"
echo "Archived: $ARCHIVED | Trashed: $TRASHED"
echo ""
echo "To delete trash: rm -rf $TRASH_DIR"
