#!/usr/bin/env python3
"""Add hybrid parameters to dynamic job_spec"""

with open('coordinator.py', 'r') as f:
    content = f.read()

# Find the sieve job_spec creation
old = """            if hasattr(args, 'method') and args.method == 'residue_sieve':
                # SIEVE job
                job_spec = {
                    'job_id': f"sieve_{job_id:03d}",
                    'seeds': seed_list,
                    'dataset_path': args.target_file,
                    'window_size': getattr(args, 'window_size', 768),
                    'min_match_threshold': getattr(args, 'threshold', 0.01),
                    'skip_range': [getattr(args, 'skip_min', 0), getattr(args, 'skip_max', 20)],
                    'offset': getattr(args, 'offset', 0),
                    'sessions': ['midday', 'evening'],
                    'search_type': 'residue_sieve'
                }
                if hasattr(args, 'prng_type'):
                    job_spec['prng_families'] = [args.prng_type]"""

new = """            if hasattr(args, 'method') and args.method == 'residue_sieve':
                # SIEVE job - include hybrid parameters
                use_hybrid = getattr(args, 'hybrid', False)
                
                job_spec = {
                    'job_id': f"sieve_{job_id:03d}",
                    'seeds': seed_list,
                    'dataset_path': args.target_file,
                    'window_size': getattr(args, 'window_size', 768),
                    'min_match_threshold': getattr(args, 'threshold', 0.01),
                    'skip_range': [getattr(args, 'skip_min', 0), getattr(args, 'skip_max', 20)],
                    'offset': getattr(args, 'offset', 0),
                    'sessions': ['midday', 'evening'],
                    'search_type': 'residue_sieve',
                    'hybrid': use_hybrid
                }
                if hasattr(args, 'prng_type'):
                    job_spec['prng_families'] = [args.prng_type]
                
                # Add hybrid-specific parameters if needed
                if use_hybrid:
                    try:
                        from hybrid_strategy import get_all_strategies
                        strategies = get_all_strategies()
                        job_spec['strategies'] = [
                            {
                                'name': s.name,
                                'max_consecutive_misses': s.max_consecutive_misses,
                                'skip_tolerance': s.skip_tolerance,
                                'enable_reseed_search': s.enable_reseed_search,
                                'skip_learning_rate': s.skip_learning_rate,
                                'breakpoint_threshold': s.breakpoint_threshold
                            }
                            for s in strategies
                        ]
                    except ImportError:
                        job_spec['strategies'] = []"""

if old in content:
    content = content.replace(old, new)
    with open('coordinator.py', 'w') as f:
        f.write(content)
    print("✅ Added hybrid parameters to dynamic job_spec!")
else:
    print("❌ Pattern not found")
    exit(1)
