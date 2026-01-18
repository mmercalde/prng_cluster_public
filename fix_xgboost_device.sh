#!/bin/bash
# XGBoost Device Mismatch Fix
# ============================
# Fixes: "Falling back to prediction using DMatrix due to mismatched devices"
# 
# This adds a predict() method to XGBoostWrapper that converts numpy arrays
# to DMatrix before prediction, ensuring proper GPU device handling.
#
# The warning occurs because:
# - Feature matrix is built as NumPy array (CPU memory)
# - XGBoost model is loaded on cuda:0
# - XGBoost internally copies CPU→GPU, which is slower
#
# This fix pre-converts to DMatrix, eliminating the copy overhead.

set -e
cd ~/distributed_prng_analysis

WRAPPER_FILE="models/wrappers/xgboost_wrapper.py"

# Check if file exists
if [[ ! -f "$WRAPPER_FILE" ]]; then
    echo "❌ File not found: $WRAPPER_FILE"
    exit 1
fi

# Check if fix already applied
if grep -q "def predict.*self.*X" "$WRAPPER_FILE"; then
    echo "✅ Fix already applied to $WRAPPER_FILE"
    exit 0
fi

# Backup
cp "$WRAPPER_FILE" "${WRAPPER_FILE}.bak"
echo "📁 Backup created: ${WRAPPER_FILE}.bak"

# Add numpy import if not present
if ! grep -q "^import numpy" "$WRAPPER_FILE"; then
    sed -i '/^import xgboost/a import numpy as np' "$WRAPPER_FILE"
    echo "📦 Added numpy import"
fi

# Find the save() method and insert predict() before it
python3 << 'PYEOF'
import re

with open("models/wrappers/xgboost_wrapper.py", "r") as f:
    content = f.read()

# The predict method to add
predict_method = '''
    def predict(self, X, **kwargs):
        """
        Predict with automatic device handling.
        
        Converts numpy arrays to DMatrix to avoid the device mismatch warning:
        "Falling back to prediction using DMatrix due to mismatched devices.
         This might lead to higher memory usage and slower performance."
        
        By pre-converting to DMatrix, XGBoost handles device placement correctly
        and avoids the internal CPU→GPU copy overhead.
        
        Args:
            X: Feature matrix (numpy array or DMatrix)
            **kwargs: Additional arguments passed to predict
            
        Returns:
            Predictions array
        """
        if isinstance(X, np.ndarray):
            # Convert to DMatrix for proper GPU handling
            dmatrix = xgb.DMatrix(X)
            return self.model.predict(dmatrix, **kwargs)
        return self.model.predict(X, **kwargs)

'''

# Insert before the save() method
if "def save(" in content and "def predict(" not in content:
    content = content.replace(
        "    def save(",
        predict_method + "    def save("
    )
    
    with open("models/wrappers/xgboost_wrapper.py", "w") as f:
        f.write(content)
    
    print("✅ Added predict() method to XGBoostWrapper")
elif "def predict(" in content:
    print("⚠️  predict() method already exists - skipping")
else:
    print("❌ Could not find save() method to insert before")
    exit(1)
PYEOF

# Verify
echo ""
echo "Verification:"
grep -n "def predict" "$WRAPPER_FILE" | head -3
echo ""
echo "Test import:"
python3 -c "from models.wrappers.xgboost_wrapper import XGBoostWrapper; print('✅ Import OK')"
