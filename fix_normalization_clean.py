#!/usr/bin/env python3
"""
Clean fix for normalization - handle zero-variance features properly
"""

with open('reinforcement_engine.py.backup_20251107_085520', 'r') as f:
    content = f.read()

# Find and replace the normalization block (lines 690-694)
old_code = """        # Apply normalization if scaler is fitted
        if self.normalization_enabled and self.scaler_fitted:
            combined = self.feature_scaler.transform([combined])[0]

        return combined"""

new_code = """        # Apply normalization if scaler is fitted
        if self.normalization_enabled and self.scaler_fitted:
            # Save original values before transform
            original_combined = combined.copy()
            
            # Transform with StandardScaler
            normalized = self.feature_scaler.transform([combined])[0]
            
            # Fix zero-variance features: StandardScaler leaves them unchanged (scale=1.0)
            # We need to manually center them by subtracting their mean
            zero_var_mask = self.feature_scaler.scale_ == 1.0
            if np.any(zero_var_mask):
                # For zero-variance: normalized_value = (original_value - mean) / 1.0
                normalized[zero_var_mask] = (original_combined[zero_var_mask] - 
                                            self.feature_scaler.mean_[zero_var_mask])
            
            combined = normalized.astype(np.float32)

        return combined"""

content = content.replace(old_code, new_code)

with open('reinforcement_engine.py', 'w') as f:
    f.write(content)

print("✅ Applied clean normalization fix")
print("   - Saves original values before transform")
print("   - Detects zero-variance features (scale == 1.0)")
print("   - Manually centers them by subtracting mean")
print("\nTest with: python3 test_normalization_properly.py")
