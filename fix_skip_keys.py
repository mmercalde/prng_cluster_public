with open('coordinator.py', 'r') as f:
    lines = f.readlines()

# Find the sieve_config section and add skip_min/skip_max
for i, line in enumerate(lines):
    if "'skip_range': [" in line and i < len(lines) - 3:
        # Check if we haven't already added these
        if "'skip_min':" not in ''.join(lines[i:i+10]):
            # Add after skip_range
            indent = ' ' * 12
            lines.insert(i + 3, f"{indent}'skip_min': getattr(args, 'skip_min', 0),\n")
            lines.insert(i + 4, f"{indent}'skip_max': getattr(args, 'skip_max', 20),\n")
            break

with open('coordinator.py', 'w') as f:
    f.writelines(lines)

print("✅ Added skip_min and skip_max as separate keys")
