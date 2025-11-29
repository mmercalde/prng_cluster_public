#!/usr/bin/env python3
"""Fix the order - patch before coordinator creation"""

with open('coordinator.py', 'r') as f:
    lines = f.readlines()

# Find the problematic section
coordinator_creation_line = None
window_opt_start = None

for i, line in enumerate(lines):
    if 'coordinator = MultiGPUCoordinator(' in line:
        coordinator_creation_line = i
    if '# Window Optimization Mode' in line and coordinator_creation_line and i > coordinator_creation_line:
        window_opt_start = i
        break

if not (coordinator_creation_line and window_opt_start):
    print("❌ Could not find target lines")
    exit(1)

print(f"Found coordinator creation at line {coordinator_creation_line + 1}")
print(f"Found window opt at line {window_opt_start + 1}")

# Extract the window optimization block (lines from window_opt_start to before if args.test_only)
window_opt_block = []
i = window_opt_start
while i < len(lines) and 'if args.test_only:' not in lines[i]:
    window_opt_block.append(lines[i])
    i += 1

# Find where coordinator creation ends
coord_end = coordinator_creation_line
bracket_count = 1
while bracket_count > 0 and coord_end < len(lines) - 1:
    coord_end += 1
    bracket_count += lines[coord_end].count('(') - lines[coord_end].count(')')

print(f"Coordinator creation ends at line {coord_end + 1}")

# Remove old window opt block
lines_without_window_opt = lines[:window_opt_start] + lines[i:]

# Now re-insert: patch code before coordinator, execution after
# Find coordinator creation again in new list
new_coord_line = None
for i, line in enumerate(lines_without_window_opt):
    if 'coordinator = MultiGPUCoordinator(' in line:
        new_coord_line = i
        break

# Extract just the patch part
patch_code = """        # Patch coordinator class for window optimization
        if args.optimize_window:
            print("\\n🔍 Window Optimization Mode Enabled")
            from window_optimizer_integration_final import add_window_optimizer_to_coordinator
            add_window_optimizer_to_coordinator()
        
"""

# Insert before coordinator creation
lines_with_patch = (lines_without_window_opt[:new_coord_line] + 
                   [patch_code] + 
                   lines_without_window_opt[new_coord_line:])

# Now find coordinator creation end in patched version
new_coord_line2 = None
for i, line in enumerate(lines_with_patch):
    if 'coordinator = MultiGPUCoordinator(' in line:
        new_coord_line2 = i
        break

# Find end of coordinator creation
coord_end2 = new_coord_line2
bracket_count = 1
while bracket_count > 0 and coord_end2 < len(lines_with_patch) - 1:
    coord_end2 += 1
    bracket_count += lines_with_patch[coord_end2].count('(') - lines_with_patch[coord_end2].count(')')

# Insert execution code after coordinator creation
execution_code = """        
        # Execute window optimization if requested
        if args.optimize_window:
            results = coordinator.optimize_window(
                dataset_path=args.target_file,
                seed_start=args.seed_start,
                seed_count=args.opt_seed_count,
                prng_base=args.prng_type if hasattr(args, 'prng_type') else 'java_lcg',
                strategy_name=args.opt_strategy,
                max_iterations=args.opt_iterations,
                output_file='window_optimization_results.json'
            )
            return 0
        
"""

final_lines = (lines_with_patch[:coord_end2+1] + 
              [execution_code] + 
              lines_with_patch[coord_end2+1:])

# Write back
with open('coordinator.py', 'w') as f:
    f.writelines(final_lines)

print("✅ Fixed the order!")
print("   Patch code now runs BEFORE coordinator creation")
print("   Execution code runs AFTER coordinator creation")

