#!/bin/bash
# Clean up all previous ML training runs

echo "=========================================="
echo "CLEANING UP PREVIOUS ML TRAINING RUNS"
echo "=========================================="
echo ""

# Backup timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 1. Clean up model checkpoints
echo "1. Cleaning model checkpoints..."
if [ -d "./models/reinforcement" ]; then
    MODEL_COUNT=$(ls -1 ./models/reinforcement/best_model_epoch_*.pth 2>/dev/null | wc -l)
    echo "   Found $MODEL_COUNT model checkpoint files"
    
    # Option A: Delete them all
    rm -f ./models/reinforcement/best_model_epoch_*.pth
    echo "   ✅ Deleted all epoch checkpoints"
    
    # Option B: Or move to backup (uncomment if you want to keep them)
    # mkdir -p ./models/reinforcement/backup_${TIMESTAMP}
    # mv ./models/reinforcement/best_model_epoch_*.pth ./models/reinforcement/backup_${TIMESTAMP}/ 2>/dev/null
    # echo "   ✅ Moved to backup_${TIMESTAMP}"
fi

# 2. Clean up Optuna database
echo ""
echo "2. Cleaning Optuna databases..."
if [ -f "optuna_studies.db" ]; then
    SIZE=$(du -h optuna_studies.db | cut -f1)
    echo "   Found optuna_studies.db ($SIZE)"
    rm -f optuna_studies.db
    echo "   ✅ Deleted local Optuna database"
fi

if [ -d "/shared/ml/optuna" ]; then
    DB_COUNT=$(ls -1 /shared/ml/optuna/*.db 2>/dev/null | wc -l)
    if [ $DB_COUNT -gt 0 ]; then
        echo "   Found $DB_COUNT distributed Optuna databases"
        rm -f /shared/ml/optuna/*.db
        echo "   ✅ Deleted distributed Optuna databases"
    fi
fi

# 3. Clean up result files
echo ""
echo "3. Cleaning result files..."
if [ -d "/shared/ml/results" ]; then
    RESULT_COUNT=$(ls -1 /shared/ml/results/*.json 2>/dev/null | wc -l)
    if [ $RESULT_COUNT -gt 0 ]; then
        echo "   Found $RESULT_COUNT result files"
        rm -f /shared/ml/results/*.json
        echo "   ✅ Deleted distributed result files"
    fi
fi

# 4. Clean up model files
echo ""
echo "4. Cleaning saved models..."
if [ -d "/shared/ml/models" ]; then
    MODEL_COUNT=$(ls -1 /shared/ml/models/*.pth 2>/dev/null | wc -l)
    if [ $MODEL_COUNT -gt 0 ]; then
        echo "   Found $MODEL_COUNT distributed model files"
        rm -f /shared/ml/models/*.pth
        echo "   ✅ Deleted distributed model files"
    fi
fi

# Also clean local models
if [ -f "universal_emulator.pth" ]; then
    echo "   Found local universal_emulator.pth"
    rm -f universal_emulator.pth
    echo "   ✅ Deleted local final model"
fi

# 5. Clean up training summary files
echo ""
echo "5. Cleaning summary files..."
if [ -f "ml_training_summary.json" ]; then
    rm -f ml_training_summary.json
    echo "   ✅ Deleted ml_training_summary.json"
fi

# 6. Clean up job specs (optional)
echo ""
echo "6. Cleaning job specifications..."
if [ -f "ml_jobs.json" ]; then
    rm -f ml_jobs.json
    echo "   ✅ Deleted ml_jobs.json"
fi

# 7. Verify cleanup
echo ""
echo "=========================================="
echo "CLEANUP COMPLETE"
echo "=========================================="
echo ""
echo "Remaining files:"
echo ""

# Check what's left
echo "Local model checkpoints:"
ls -lh ./models/reinforcement/*.pth 2>/dev/null | wc -l | xargs echo "  "

echo "Optuna databases:"
ls -lh optuna_studies.db /shared/ml/optuna/*.db 2>/dev/null | wc -l | xargs echo "  "

echo "Result files:"
ls -lh /shared/ml/results/*.json 2>/dev/null | wc -l | xargs echo "  "

echo "Model files:"
ls -lh /shared/ml/models/*.pth universal_emulator.pth 2>/dev/null | wc -l | xargs echo "  "

echo ""
echo "✅ Ready for fresh training run!"
echo ""
