#!/bin/bash

# Backup Script: Creates a timestamped directory and copies all files/dirs from current location into it.
# Usage: ./backup_script.sh [optional_backup_name]
# Example: ./backup_script.sh my_project  # Creates backup_YYYYMMDD_HHMMSS_my_project

# Get current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Optional custom name (default empty)
BACKUP_NAME="${1:-}"

# Construct backup dir name
if [ -n "$BACKUP_NAME" ]; then
    BACKUP_DIR="backup_${TIMESTAMP}_${BACKUP_NAME}"
else
    BACKUP_DIR="backup_${TIMESTAMP}"
fi

# Create the timestamped directory
mkdir -p "$BACKUP_DIR"

# Check if directory was created
if [ $? -ne 0 ]; then
    echo "Error: Failed to create directory $BACKUP_DIR"
    exit 1
fi

# Copy all files and directories recursively (excluding the backup dir itself)
# Use rsync for efficiency and to preserve permissions/timestamps
rsync -av --exclude="$BACKUP_DIR" . "$BACKUP_DIR/" 2>/dev/null

# Alternative: Use cp -r if rsync not available
# cp -r . "$BACKUP_DIR/" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ Backup completed successfully!"
    echo "📁 Backup location: $(pwd)/$BACKUP_DIR"
    echo "📊 Contents copied: $(find . -maxdepth 1 -type f | wc -l) files and $(find . -maxdepth 1 -type d | wc -l - | wc -l) directories"
    ls -la "$BACKUP_DIR" | head -10  # Show first 10 items
else
    echo "❌ Backup failed!"
    exit 1
fi
