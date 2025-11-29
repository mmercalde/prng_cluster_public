#!/usr/bin/env python3
"""
Final fix for normalization - properly handle zero-variance features
"""

with open('reinforcement_engine.py', 'r') as f:
    content = f.read()

# Find and replace the normalization application block
import re

# Pattern to match the current normalization block
pattern = r'(\s+)# Apply normalization if scaler is fitted\s+if self\.normalization_enabled and self\.scaler_fitted:\s+# Scaler needs 2D input, returns 2D output\s+normalized = self\.feature_scaler\.transform\(combined\.reshape\(1, -1\)\)\[0\]\s+# Handle features with zero variance.*?\s+combined = normalized\.astype\(np\.float32\)'

replacement = r'''\1# Apply normalization if scaler is fitted
\1if self.normalization_enabled and self.scaler_fitted:
\1    # Scaler needs 2D input, returns 2D output  
\1    normalized = self.feature_scaler.transform(combined.reshape(1, -1))[0]
\1    
\1    # For zero-variance features (scale=1.0), manually center them
\1    # These are constant across all survivors, so subtract mean to center at 0
\1    zero_var_mask = self.feature_scaler.scale_ == 1.0
\1    if np.any(zero_var_mask):
\1        # Manually apply centering: (x - mean) / 1.0 = x - mean
\1        normalized[zero_var_mask] = (combined[zero_var_mask] - self.feature_scaler.mean_[zero_var_mask])
\1    
\1    combined = normalized.astype(np.float32)'''

content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open('reinforcement_engine.py', 'w') as f:
    f.write(content)

print("✅ Applied final normalization fix")
