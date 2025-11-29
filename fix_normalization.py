import re

with open('reinforcement_engine.py', 'r') as f:
    content = f.read()

# Find and replace the normalization section
old_pattern = r'combined = np\.array\(per_seed_values \+ global_values, dtype=np\.float32\)\s+# Apply normalization if scaler is fitted\s+if self\.normalization_enabled and self\.scaler_fitted:\s+combined = self\.feature_scaler\.transform\(\[combined\]\)\[0\]'

new_code = '''combined = np.array(per_seed_values + global_values, dtype=np.float32)

        # Apply normalization if scaler is fitted
        if self.normalization_enabled and self.scaler_fitted:
            # Scaler needs 2D input, returns 2D output
            combined = self.feature_scaler.transform(combined.reshape(1, -1))[0]
            # Ensure float32 after transform
            combined = combined.astype(np.float32)'''

content = re.sub(old_pattern, new_code, content)

with open('reinforcement_engine.py', 'w') as f:
    f.write(content)

print("✅ Fixed normalization in reinforcement_engine.py")
