import sys

# Read coordinator.py
with open('coordinator.py', 'r') as f:
    lines = f.readlines()

# Find and update _create_sieve_jobs
in_function = False
updated_lines = []

for i, line in enumerate(lines):
    if 'def _create_sieve_jobs(self, args):' in line:
        in_function = True
        updated_lines.append(line)
        continue
    
    # Add seed_start after total_seeds calculation
    if in_function and 'seeds_per_worker = total_seeds // len(workers)' in line:
        updated_lines.append(line)
        updated_lines.append('\n')
        updated_lines.append('        # Handle timestamp searches - seed_start shifts the entire range\n')
        updated_lines.append('        base_seed_start = getattr(args, "seed_start", 0)\n')
        continue
    
    # Update seed_start calculation
    if in_function and 'seed_start = i * seeds_per_worker' in line:
        updated_lines.append('            seed_start = base_seed_start + (i * seeds_per_worker)\n')
        continue
    
    updated_lines.append(line)
    
    if in_function and line.strip().startswith('return jobs'):
        in_function = False

# Write back
with open('coordinator.py', 'w') as f:
    f.writelines(updated_lines)

print("✓ Updated coordinator.py")
print("  Added --seed-start parameter")
print("  Updated _create_sieve_jobs to use base_seed_start")
