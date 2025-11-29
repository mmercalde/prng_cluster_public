#!/usr/bin/env python3
"""Insert window optimization code into coordinator.py"""

with open('coordinator.py', 'r') as f:
    lines = f.readlines()

# Find the line with "if args.test_only:"
insertion_line = None
for i, line in enumerate(lines):
    if 'if args.test_only:' in line and 'coordinator' in ''.join(lines[max(0,i-20):i]):
        insertion_line = i
        break

if insertion_line is None:
    print("❌ Could not find insertion point!")
    exit(1)

print(f"✅ Found insertion point at line {insertion_line + 1}")

# The code to insert
window_opt_code = '''        
        # Window Optimization Mode
        if args.optimize_window:
            print("\\n🔍 Window Optimization Mode Enabled")
            from window_optimizer_integration_final import add_window_optimizer_to_coordinator
            add_window_optimizer_to_coordinator()
            
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
        
'''

# Insert the code
new_lines = lines[:insertion_line] + [window_opt_code] + lines[insertion_line:]

# Write back
with open('coordinator.py', 'w') as f:
    f.writelines(new_lines)

print("✅ Inserted window optimization code successfully!")
print(f"   Location: before line {insertion_line + 1} (if args.test_only:)")

