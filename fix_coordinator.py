with open('coordinator.py', 'r') as f:
    content = f.read()

# Find and replace the sieve_config dict
old_config = """        sieve_config = {
            'dataset_path': args.target_file,
            'window_size': getattr(args, 'window_size', 512),
            'min_match_threshold': getattr(args, 'threshold', 0.01),
            'skip_range': [
                getattr(args, 'skip_min', 0),
                getattr(args, 'skip_max', 20)
            ],
            'offset': getattr(args, 'offset', 0),
            'prng_families': [args.prng_type] if hasattr(args, 'prng_type') and args.prng_type else ['mt19937'],
            'sessions': ['midday', 'evening'],
            'hybrid': use_hybrid,
            'phase2_threshold': p2 if use_hybrid else None,
        }"""

new_config = """        sieve_config = {
            'dataset_path': args.target_file,
            'window_size': getattr(args, 'window_size', 512),
            'min_match_threshold': getattr(args, 'threshold', 0.01),
            'skip_range': [
                getattr(args, 'skip_min', 0),
                getattr(args, 'skip_max', 20)
            ],
            'skip_min': getattr(args, 'skip_min', 0),
            'skip_max': getattr(args, 'skip_max', 20),
            'offset': getattr(args, 'offset', 0),
            'prng_families': [args.prng_type] if hasattr(args, 'prng_type') and args.prng_type else ['mt19937'],
            'sessions': ['midday', 'evening'],
            'hybrid': use_hybrid,
            'phase2_threshold': p2 if use_hybrid else None,
        }"""

content = content.replace(old_config, new_config)

with open('coordinator.py', 'w') as f:
    f.write(content)

print("✅ Added skip_min and skip_max keys to sieve_config")
