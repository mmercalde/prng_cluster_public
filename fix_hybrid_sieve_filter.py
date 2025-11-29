#!/usr/bin/env python3
"""Add hybrid mode support to sieve_filter.py - precise whitespace matching"""

with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

# Find the line with "# Initialize sieve" and insert hybrid logic after the for loop starts
insert_after = None
for i, line in enumerate(lines):
    if 'for family_spec in prng_families:' in line:
        insert_after = i
        break

if insert_after is None:
    print("⚠️ Could not find insertion point")
    exit(1)

print(f"✅ Found insertion point at line {insert_after + 1}")

# The new hybrid mode logic to insert right after the for loop starts
hybrid_code = """
            # Check if hybrid mode is enabled
            use_hybrid = job.get('hybrid', False)
            
            # Hybrid mode only works with mt19937
            if use_hybrid and family_name == 'mt19937':
                print(f"Testing {family_name} in HYBRID mode...", file=sys.stderr)
                
                # Get strategies from job
                strategies_data = job.get('strategies')
                if not strategies_data:
                    # Import default strategies
                    try:
                        from hybrid_strategy import get_all_strategies
                        strategies = get_all_strategies()
                    except ImportError:
                        print("WARNING: Hybrid mode requested but strategies not available", file=sys.stderr)
                        print("         Falling back to standard mode", file=sys.stderr)
                        use_hybrid = False
                        strategies = None
                else:
                    # Reconstruct strategy objects from dict data
                    from hybrid_strategy import StrategyConfig
                    strategies = [StrategyConfig(**s) for s in strategies_data]
                
                if use_hybrid and strategies:
                    # Use hybrid sieve
                    phase1_threshold = job.get('phase1_threshold', 0.20)
                    phase2_threshold = job.get('phase2_threshold', 0.75)
                    
                    print(f"  Phase 1 threshold: {phase1_threshold:.1%}", file=sys.stderr)
                    print(f"  Phase 2 threshold: {phase2_threshold:.1%}", file=sys.stderr)
                    print(f"  Testing {len(strategies)} strategies", file=sys.stderr)
                    
                    result = sieve.run_hybrid_sieve(
                        prng_family='mt19937_hybrid',
                        seed_start=seed_start,
                        seed_end=seed_end,
                        residues=draws,
                        strategies=strategies,
                        min_match_threshold=phase2_threshold,
                        offset=offset
                    )
                    per_family_results.append(result)
                    all_survivors.extend(result['survivors'])
                    continue
            
"""

# Find where to insert (after the "family_name = ..." assignment)
insert_at = None
for i in range(insert_after, min(insert_after + 20, len(lines))):
    if 'custom_params = None' in lines[i]:
        insert_at = i + 1
        break

if insert_at is None:
    print("⚠️ Could not find exact insertion point")
    exit(1)

print(f"✅ Inserting hybrid code at line {insert_at + 1}")

# Insert the hybrid code
lines.insert(insert_at, hybrid_code)

# Write back
with open('sieve_filter.py', 'w') as f:
    f.writelines(lines)

print("✅ Hybrid mode added to sieve_filter.py")

# Test syntax
try:
    with open('sieve_filter.py', 'r') as f:
        compile(f.read(), 'sieve_filter.py', 'exec')
    print("✅ File syntax is valid!")
except SyntaxError as e:
    print(f"❌ Syntax error: {e}")
    import traceback
    traceback.print_exc()
