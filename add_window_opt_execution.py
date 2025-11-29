#!/usr/bin/env python3
"""
Add window optimizer execution logic to coordinator.py main()
"""

with open('coordinator.py', 'r') as f:
    content = f.read()

# Find where to insert - after coordinator creation and before normal execution
# Look for the line that creates the coordinator

insert_marker = "coordinator = GPUCoordinator("

if insert_marker not in content:
    print("❌ Could not find coordinator creation!")
    exit(1)

# Find the section after coordinator creation
# We need to insert after the coordinator is fully initialized

execution_code = """
    
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
        return
"""

# Find where to insert - look for the execute_parallel_dynamic_sieve_analysis call
lines = content.split('\n')
new_lines = []
inserted = False

for i, line in enumerate(lines):
    new_lines.append(line)
    
    # Insert after coordinator creation and before execution
    if 'coordinator = GPUCoordinator(' in line and not inserted:
        # Find the closing parenthesis
        bracket_count = line.count('(') - line.count(')')
        j = i + 1
        while bracket_count > 0 and j < len(lines):
            bracket_count += lines[j].count('(') - lines[j].count(')')
            new_lines.append(lines[j])
            j += 1
        
        # Now insert our code
        new_lines.append(execution_code)
        inserted = True
        
        # Skip ahead
        i = j - 1

if not inserted:
    print("⚠️  Could not auto-insert execution code.")
    print("Please manually add this code after coordinator creation:")
    print(execution_code)
else:
    # Write back
    with open('coordinator.py', 'w') as f:
        f.write('\n'.join(new_lines))
    print("✅ Added window optimizer execution logic to coordinator.py")

