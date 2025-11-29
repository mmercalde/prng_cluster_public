#!/usr/bin/env python3
"""Add hybrid mode support to coordinator.py"""

with open('coordinator.py', 'r') as f:
    content = f.read()

# 1. Add hybrid imports at the top (after existing imports)
import_addition = """
# Hybrid strategy support (optional)
try:
    from hybrid_strategy import get_all_strategies, STRATEGY_PRESETS
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False
    print("Note: hybrid_strategy module not available. Hybrid mode disabled.")
"""

# Find the imports section (after "from collections import defaultdict, deque")
import_marker = "from collections import defaultdict, deque"
if import_marker in content:
    content = content.replace(import_marker, import_marker + "\n" + import_addition)
    print("✅ Added hybrid imports")
else:
    print("⚠️ Could not find import marker")

# 2. Update _create_sieve_jobs to support hybrid mode
# Find and replace the _sieve_config creation
old_sieve_config = """        self._sieve_config = {
            'dataset_path': args.target_file,
            'window_size': getattr(args, 'window_size', 10),
            'min_match_threshold': getattr(args, 'threshold', 0.6),
            'skip_range': [getattr(args, 'skip_min', 0), getattr(args, 'skip_max', 20)],
            'offset': args.offset if hasattr(args, 'offset') else 0,
            'prng_families': [args.prng_type] if hasattr(args, 'prng_type') else ['xorshift32'],
            'sessions': ['midday', 'evening']
        }"""

new_sieve_config = """        # Check if hybrid mode requested
        use_hybrid = getattr(args, 'hybrid', False)
        
        self._sieve_config = {
            'dataset_path': args.target_file,
            'window_size': getattr(args, 'window_size', 10),
            'min_match_threshold': getattr(args, 'threshold', 0.6),
            'skip_range': [getattr(args, 'skip_min', 0), getattr(args, 'skip_max', 20)],
            'offset': args.offset if hasattr(args, 'offset') else 0,
            'prng_families': [args.prng_type] if hasattr(args, 'prng_type') else ['xorshift32'],
            'sessions': ['midday', 'evening'],
            'hybrid': use_hybrid,
            'phase1_threshold': getattr(args, 'phase1_threshold', 0.20) if use_hybrid else None,
            'phase2_threshold': getattr(args, 'phase2_threshold', 0.75) if use_hybrid else None,
            'strategies': get_all_strategies() if (use_hybrid and HYBRID_AVAILABLE) else None
        }
        
        if use_hybrid:
            if HYBRID_AVAILABLE:
                print(f"🔬 Hybrid mode enabled with {len(self._sieve_config['strategies'])} strategies")
            else:
                print("⚠️ Hybrid mode requested but hybrid_strategy module not available")
                print("   Falling back to standard fixed-skip mode")
                self._sieve_config['hybrid'] = False"""

content = content.replace(old_sieve_config, new_sieve_config)
print("✅ Updated _sieve_config for hybrid support")

# 3. Update job_data creation in execute_local_job
old_local_job = """                job_data = {
                    'job_id': job.job_id,
                    'dataset_path': self.current_target_file or 'test_known.json',
                    'seed_start': seed_start,
                    'seed_end': seed_end,
                    'window_size': self._sieve_config.get('window_size', 10) if hasattr(self, '_sieve_config') else 10,
                    'min_match_threshold': self._sieve_config.get('min_match_threshold', 0.6) if hasattr(self, '_sieve_config') else 0.6,
                    'skip_range': self._sieve_config.get('skip_range', [0, 20]) if hasattr(self, '_sieve_config') else [0, 20],
                    'offset': self._sieve_config.get('offset', 0) if hasattr(self, '_sieve_config') else 0,
                    'prng_families': self._sieve_config.get('prng_families', ['xorshift32']) if hasattr(self, '_sieve_config') else ['xorshift32'],
                    'sessions': self._sieve_config.get('sessions', ['midday', 'evening']) if hasattr(self, '_sieve_config') else ['midday', 'evening']
                }"""

new_local_job = """                job_data = {
                    'job_id': job.job_id,
                    'dataset_path': self.current_target_file or 'test_known.json',
                    'seed_start': seed_start,
                    'seed_end': seed_end,
                    'window_size': self._sieve_config.get('window_size', 10) if hasattr(self, '_sieve_config') else 10,
                    'min_match_threshold': self._sieve_config.get('min_match_threshold', 0.6) if hasattr(self, '_sieve_config') else 0.6,
                    'skip_range': self._sieve_config.get('skip_range', [0, 20]) if hasattr(self, '_sieve_config') else [0, 20],
                    'offset': self._sieve_config.get('offset', 0) if hasattr(self, '_sieve_config') else 0,
                    'prng_families': self._sieve_config.get('prng_families', ['xorshift32']) if hasattr(self, '_sieve_config') else ['xorshift32'],
                    'sessions': self._sieve_config.get('sessions', ['midday', 'evening']) if hasattr(self, '_sieve_config') else ['midday', 'evening'],
                    'hybrid': self._sieve_config.get('hybrid', False) if hasattr(self, '_sieve_config') else False,
                    'phase1_threshold': self._sieve_config.get('phase1_threshold') if hasattr(self, '_sieve_config') else None,
                    'phase2_threshold': self._sieve_config.get('phase2_threshold') if hasattr(self, '_sieve_config') else None,
                    'strategies': [s.to_dict() for s in self._sieve_config['strategies']] if (hasattr(self, '_sieve_config') and self._sieve_config.get('strategies')) else None
                }"""

content = content.replace(old_local_job, new_local_job)
print("✅ Updated execute_local_job for hybrid parameters")

# 4. Update job_data creation in execute_remote_job (same changes)
# The remote job has the exact same structure, so we can replace again
content = content.replace(old_local_job, new_local_job)
print("✅ Updated execute_remote_job for hybrid parameters")

# Write back
with open('coordinator.py', 'w') as f:
    f.write(content)

print("✅ Hybrid support added to coordinator.py")

# Test syntax
try:
    compile(content, 'coordinator.py', 'exec')
    print("✅ File syntax is valid!")
except SyntaxError as e:
    print(f"❌ Syntax error: {e}")
