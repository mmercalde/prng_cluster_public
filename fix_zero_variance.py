import re

with open('reinforcement_engine.py', 'r') as f:
    lines = f.readlines()

# Find the line with self.feature_scaler.fit
for i, line in enumerate(lines):
    if 'self.feature_scaler.fit(features_array)' in line:
        # Get indentation
        indent = len(line) - len(line.lstrip())
        spaces = ' ' * indent
        
        # Replace with improved version
        lines[i] = f'''{spaces}# Fit scaler (handle zero variance features)
{spaces}self.feature_scaler.fit(features_array)
{spaces}
{spaces}# Fix any features with zero/tiny variance
{spaces}zero_var_mask = self.feature_scaler.scale_ < 1e-8
{spaces}if np.any(zero_var_mask):
{spaces}    logger.warning(f"   {{np.sum(zero_var_mask)}} features have zero variance, setting scale to 1.0")
{spaces}    self.feature_scaler.scale_[zero_var_mask] = 1.0
'''
        break

with open('reinforcement_engine.py', 'w') as f:
    f.writelines(lines)

print("✅ Added zero-variance handling to _fit_normalizer")
