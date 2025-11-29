#!/usr/bin/env python3
"""
Properly fix normalization by handling constant features
"""

with open('reinforcement_engine.py', 'r') as f:
    content = f.read()

# Find the extract_combined_features method where normalization is applied
# Replace the normalization block

old_block = '''        # Apply normalization if scaler is fitted
        if self.normalization_enabled and self.scaler_fitted:
            # Scaler needs 2D input, returns 2D output
            combined = self.feature_scaler.transform(combined.reshape(1, -1))[0]
            # Ensure float32 after transform
            combined = combined.astype(np.float32)'''

new_block = '''        # Apply normalization if scaler is fitted
        if self.normalization_enabled and self.scaler_fitted:
            # Scaler needs 2D input, returns 2D output
            normalized = self.feature_scaler.transform(combined.reshape(1, -1))[0]
            
            # Handle features with zero variance (where scale was set to 1.0)
            # These features are constant, so just center them at 0
            zero_var_mask = self.feature_scaler.scale_ == 1.0
            if np.any(zero_var_mask):
                normalized[zero_var_mask] = 0.0
            
            combined = normalized.astype(np.float32)'''

content = content.replace(old_block, new_block)

with open('reinforcement_engine.py', 'w') as f:
    f.write(content)

print("✅ Fixed normalization to handle constant features")
