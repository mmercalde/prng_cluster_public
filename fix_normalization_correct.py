#!/usr/bin/env python3
"""
Correct fix: Save original values before transform
"""

with open('reinforcement_engine.py', 'r') as f:
    content = f.read()

import re

# Find the normalization block
old_block = '''        # Apply normalization if scaler is fitted
        if self.normalization_enabled and self.scaler_fitted:
            # Scaler needs 2D input, returns 2D output
            normalized = self.feature_scaler.transform(combined.reshape(1, -1))[0]
            
            # For zero-variance features (scale=1.0), manually center them
            # These are constant across all survivors, so subtract mean to center at 0
            zero_var_mask = self.feature_scaler.scale_ == 1.0
            if np.any(zero_var_mask):
                # Manually apply centering: (x - mean) / 1.0 = x - mean
                normalized[zero_var_mask] = (combined[zero_var_mask] - self.feature_scaler.mean_[zero_var_mask])
            
            combined = normalized.astype(np.float32)'''

new_block = '''        # Apply normalization if scaler is fitted
        if self.normalization_enabled and self.scaler_fitted:
            # Save original values BEFORE transform (needed for zero-variance features)
            original_combined = combined.copy()
            
            # Scaler needs 2D input, returns 2D output
            normalized = self.feature_scaler.transform(combined.reshape(1, -1))[0]
            
            # For zero-variance features (scale=1.0), manually center them
            # StandardScaler leaves them unchanged, so we need to manually center
            zero_var_mask = self.feature_scaler.scale_ == 1.0
            if np.any(zero_var_mask):
                # Manually apply centering: (x - mean) / 1.0 = x - mean
                normalized[zero_var_mask] = (original_combined[zero_var_mask] - self.feature_scaler.mean_[zero_var_mask])
            
            combined = normalized.astype(np.float32)'''

content = content.replace(old_block, new_block)

with open('reinforcement_engine.py', 'w') as f:
    f.write(content)

print("✅ Fixed: Now using original_combined for zero-variance centering")
