#!/usr/bin/env python3
"""Add hybrid mode support to sieve_filter.py execute_sieve_job function"""

with open('sieve_filter.py', 'r') as f:
    content = f.read()

# Find the section where sieve is run and add hybrid mode check
old_sieve_run = """        # Initialize sieve
        sieve = GPUSieve(gpu_id=gpu_id)
        
        # Run sieve for each family
        per_family_results = []
        all_survivors = []
        for family_spec in prng_families:
            # Handle both string names and dict with custom params
            if isinstance(family_spec, dict):
                family_name = family_spec['type']
                custom_params = family_spec.get('params', {})
            else:
                family_name = family_spec
                custom_params = None
            
            print(f"Testing {family_name}...", file=sys.stderr)
            result = sieve.run_sieve(
                prng_family=family_name,
                seed_start=seed_start,
                seed_end=seed_end,
                residues=draws,
                skip_range=skip_range,
                offset=offset,
                min_match_threshold=min_match_threshold,
                custom_params=custom_params
            )
            per_family_results.append(result)
            all_survivors.extend(result['survivors'])"""

new_sieve_run = """        # Initialize sieve
        sieve = GPUSieve(gpu_id=gpu_id)
        
        # Check if hybrid mode is enabled
        use_hybrid = job.get('hybrid', False)
        
        # Run sieve for each family
        per_family_results = []
        all_survivors = []
        for family_spec in prng_families:
            # Handle both string names and dict with custom params
            if isinstance(family_spec, dict):
                family_name = family_spec['type']
                custom_params = family_spec.get('params', {})
            else:
                family_name = family_spec
                custom_params = None
            
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
            
            # Standard fixed-skip mode
            print(f"Testing {family_name}...", file=sys.stderr)
            result = sieve.run_sieve(
                prng_family=family_name,
                seed_start=seed_start,
                seed_end=seed_end,
                residues=draws,
                skip_range=skip_range,
                offset=offset,
                min_match_threshold=min_match_threshold,
                custom_params=custom_params
            )
            per_family_results.append(result)
            all_survivors.extend(result['survivors'])"""

if old_sieve_run in content:
    content = content.replace(old_sieve_run, new_sieve_run)
    print("✅ Added hybrid mode support to execute_sieve_job")
else:
    print("⚠️ Could not find sieve run section")

# Write back
with open('sieve_filter.py', 'w') as f:
    f.write(content)

print("✅ Hybrid mode added to sieve_filter.py")

# Test syntax
try:
    compile(content, 'sieve_filter.py', 'exec')
    print("✅ File syntax is valid!")
except SyntaxError as e:
    print(f"❌ Syntax error: {e}")
