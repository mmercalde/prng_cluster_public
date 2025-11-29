#!/usr/bin/env python3
"""Fix sieve_filter.py to implement proper two-phase hybrid"""

with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

# Find the hybrid section (starts around line 407)
start_idx = None
end_idx = None

for i, line in enumerate(lines):
    if 'if use_hybrid and family_name == ' in line:
        start_idx = i
    if start_idx and '                    continue' in line:
        end_idx = i + 1
        break

if not start_idx or not end_idx:
    print("❌ Could not find hybrid section")
    exit(1)

print(f"✅ Found hybrid section: lines {start_idx+1} to {end_idx}")

# New two-phase implementation
new_hybrid_code = '''            # Hybrid mode only works with mt19937
            if use_hybrid and family_name == 'mt19937':
                print(f"Testing {family_name} in HYBRID TWO-PHASE mode...", file=sys.stderr)
                
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
                    # Get thresholds (modular - easy to tweak via job config)
                    phase1_threshold = job.get('phase1_threshold', 0.20)
                    phase2_threshold = job.get('phase2_threshold', 0.75)
                    
                    print(f"  Phase 1: FULL MT19937 fixed-skip wide search", file=sys.stderr)
                    print(f"    Threshold: {phase1_threshold:.1%} (finds partial matches)", file=sys.stderr)
                    
                    # PHASE 1: Wide search with FULL MT19937 and fixed skip
                    phase1_result = sieve.run_sieve(
                        prng_family='mt19937',  # FULL MT19937 (same as test data)
                        seed_start=seed_start,
                        seed_end=seed_end,
                        residues=draws,
                        skip_range=skip_range,
                        min_match_threshold=phase1_threshold,
                        offset=offset
                    )
                    
                    phase1_survivors = phase1_result.get('survivors', [])
                    print(f"  Phase 1 complete: {len(phase1_survivors)} survivors found", file=sys.stderr)
                    
                    # PHASE 2: Deep dive on survivors only with variable skip
                    if phase1_survivors:
                        print(f"  Phase 2: Variable skip analysis on {len(phase1_survivors)} survivors", file=sys.stderr)
                        print(f"    Threshold: {phase2_threshold:.1%} (confirms exact pattern)", file=sys.stderr)
                        print(f"    Testing {len(strategies)} strategies", file=sys.stderr)
                        
                        # Extract survivor seeds
                        survivor_seeds = [s['seed'] for s in phase1_survivors]
                        
                        # Run hybrid on survivors only
                        phase2_result = sieve.run_hybrid_sieve(
                            prng_family='mt19937_hybrid',  # SAME FULL MT19937, variable skip
                            seed_start=min(survivor_seeds),
                            seed_end=max(survivor_seeds) + 1,
                            residues=draws,
                            strategies=strategies,
                            min_match_threshold=phase2_threshold,
                            offset=offset
                        )
                        
                        final_survivors = phase2_result.get('survivors', [])
                        print(f"  Phase 2 complete: {len(final_survivors)} final matches", file=sys.stderr)
                        
                        # Build combined result
                        result = {
                            'family': 'mt19937_hybrid',
                            'seed_range': {'start': seed_start, 'end': seed_end},
                            'phase1': {
                                'survivors': len(phase1_survivors),
                                'threshold': phase1_threshold
                            },
                            'phase2': {
                                'survivors': len(final_survivors),
                                'threshold': phase2_threshold,
                                'strategies': len(strategies)
                            },
                            'survivors': final_survivors,
                            'stats': phase2_result.get('stats', {})
                        }
                    else:
                        print(f"  Phase 1 found no survivors - skipping Phase 2", file=sys.stderr)
                        result = {
                            'family': 'mt19937_hybrid',
                            'seed_range': {'start': seed_start, 'end': seed_end},
                            'phase1': {
                                'survivors': 0,
                                'threshold': phase1_threshold
                            },
                            'phase2': {'skipped': True},
                            'survivors': [],
                            'stats': phase1_result.get('stats', {})
                        }
                    
                    per_family_results.append(result)
                    all_survivors.extend(result['survivors'])
'''

# Replace the section
new_lines = lines[:start_idx] + [new_hybrid_code] + lines[end_idx:]

with open('sieve_filter.py', 'w') as f:
    f.writelines(new_lines)

print("✅ Added proper two-phase hybrid implementation")
print("✅ Phase 1: FULL MT19937 fixed-skip (wide search)")
print("✅ Phase 2: FULL MT19937 variable-skip (deep dive on survivors)")
print("✅ Thresholds are modular - configure via job file")

# Verify syntax
try:
    compile(''.join(new_lines), 'sieve_filter.py', 'exec')
    print("✅ File compiles!")
except SyntaxError as e:
    print(f"❌ Syntax error: {e}")
