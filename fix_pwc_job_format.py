#!/usr/bin/env python3
"""
fix_pwc_job_format.py
=====================
Fix persistent_worker_coordinator.py job dict to match coordinator.py format.

Root cause: PWC sends wrong field names to sieve_gpu_worker.py:
  - 'threshold'  → should be 'min_match_threshold'
  - 'residues'   → should NOT be sent (worker loads draws from dataset_path)
  - missing 'skip_range', 'offset', 'sessions', 'prng_families'

The coordinator.py job format (the reference) sends:
  dataset_path, seed_start, seed_end, window_size, min_match_threshold,
  skip_range, offset, sessions, prng_families, search_type, job_id

This patch:
1. Adds offset, sessions, skip_range to run_sieve_pass() signature
2. Fixes job dict field names to match coordinator.py exactly
3. Adds run_trial_persistent() pass-through of offset, sessions, skip_range
"""
import shutil
from pathlib import Path

TARGET = Path('/home/michael/distributed_prng_analysis/persistent_worker_coordinator.py')
content = TARGET.read_text(encoding='utf-8')
shutil.copy2(TARGET, str(TARGET) + '.job_format_backup')
print(f'Backup: {TARGET.name}.job_format_backup')

original = content
fixes = 0

# ── Fix 1: run_sieve_pass signature — add offset, sessions, skip_range ──────
old = '''    def run_sieve_pass(self,
                       prng_type: str,
                       residues: List[int],
                       total_seeds: int,
                       threshold: float,
                       window_size: int,
                       output_file: str,
                       dataset_path: str = "",
                       strategies: Optional[List[Dict]] = None,
                       phase2_threshold: float = 0.5,
                       target_file: str = "") -> Dict[str, Any]:'''

new = '''    def run_sieve_pass(self,
                       prng_type: str,
                       residues: List[int],
                       total_seeds: int,
                       threshold: float,
                       window_size: int,
                       output_file: str,
                       dataset_path: str = "",
                       strategies: Optional[List[Dict]] = None,
                       phase2_threshold: float = 0.5,
                       target_file: str = "",
                       offset: int = 0,
                       sessions: Optional[List[str]] = None,
                       skip_range: Optional[List[int]] = None) -> Dict[str, Any]:'''

if old in content:
    content = content.replace(old, new, 1)
    print('✅ Fix 1: run_sieve_pass signature — added offset, sessions, skip_range')
    fixes += 1
else:
    print('⚠️  Fix 1: signature not found')

# ── Fix 2: Job dict — fix field names to match coordinator.py format ─────────
old = '''            job = {
                "job_id":            f"sieve_{idx:03d}",
                "prng_type":         prng_type,
                "search_type":       "residue_sieve",
                "seed_start":        seed_start,
                "seed_end":          seed_end,
                "residues":          residues,
                "window_size":       window_size,
                "threshold":         threshold,
                "phase2_threshold":  phase2_threshold,
                "strategies":        strategies if is_hybrid else None,
                "hybrid":            is_hybrid,
                "target_file":       target_file,
                "dataset_path":      dataset_path,
            }'''

new = '''            # Job dict matches coordinator.py residue_sieve format exactly
            # (field names verified against sieve_gpu_worker.py job.get() calls)
            _sessions   = sessions   if sessions   is not None else ["midday", "evening"]
            _skip_range = skip_range if skip_range is not None else [0, 147]
            # prng_families: strip _reverse/_hybrid suffixes — worker handles variants
            _base_prng  = prng_type.replace("_reverse", "").replace("_hybrid", "")
            job = {
                "job_id":               f"sieve_{idx:03d}",
                "search_type":          "residue_sieve",
                "dataset_path":         dataset_path or target_file,
                "seed_start":           seed_start,
                "seed_end":             seed_end,
                "window_size":          window_size,
                "min_match_threshold":  threshold,
                "skip_range":           _skip_range,
                "offset":               offset,
                "sessions":             _sessions,
                "prng_families":        [prng_type],
                "strategies":           strategies if is_hybrid else None,
                "hybrid":               is_hybrid,
                "phase2_threshold":     phase2_threshold,
            }'''

if old in content:
    content = content.replace(old, new, 1)
    print('✅ Fix 2: job dict — fixed field names to match coordinator.py format')
    fixes += 1
else:
    print('⚠️  Fix 2: job dict not found')

# ── Fix 3: run_trial_persistent — pass offset, sessions, skip_range ──────────
# Forward pass 1
old = '''        fwd_result = pwc.run_sieve_pass(
            prng_type    = prng_base,
            residues     = residues,
            total_seeds  = total_seeds,
            threshold    = forward_threshold,
            window_size  = ws,
            dataset_path = dataset_path,
            output_file  = f"results/window_opt_forward_{ws}_{off}_t{trial_number}.json",
        )'''

new = '''        fwd_result = pwc.run_sieve_pass(
            prng_type    = prng_base,
            residues     = residues,
            total_seeds  = total_seeds,
            threshold    = forward_threshold,
            window_size  = ws,
            dataset_path = dataset_path,
            output_file  = f"results/window_opt_forward_{ws}_{off}_t{trial_number}.json",
            offset       = config.offset,
            sessions     = list(config.sessions) if hasattr(config, 'sessions') else ["midday", "evening"],
            skip_range   = [config.skip_min, config.skip_max] if hasattr(config, 'skip_min') else [0, 147],
        )'''

if old in content:
    content = content.replace(old, new, 1)
    print('✅ Fix 3a: forward pass — added offset, sessions, skip_range')
    fixes += 1
else:
    print('⚠️  Fix 3a: forward pass not found')

# Reverse pass 2
old = '''        rev_result = pwc.run_sieve_pass(
            prng_type    = prng_reverse,
            residues     = residues,
            total_seeds  = total_seeds,
            threshold    = reverse_threshold,
            window_size  = ws,
            dataset_path = dataset_path,
            output_file  = f"results/window_opt_reverse_{ws}_{off}_t{trial_number}.json",
        )'''

new = '''        rev_result = pwc.run_sieve_pass(
            prng_type    = prng_reverse,
            residues     = residues,
            total_seeds  = total_seeds,
            threshold    = reverse_threshold,
            window_size  = ws,
            dataset_path = dataset_path,
            output_file  = f"results/window_opt_reverse_{ws}_{off}_t{trial_number}.json",
            offset       = config.offset,
            sessions     = list(config.sessions) if hasattr(config, 'sessions') else ["midday", "evening"],
            skip_range   = [config.skip_min, config.skip_max] if hasattr(config, 'skip_min') else [0, 147],
        )'''

if old in content:
    content = content.replace(old, new, 1)
    print('✅ Fix 3b: reverse pass — added offset, sessions, skip_range')
    fixes += 1
else:
    print('⚠️  Fix 3b: reverse pass not found')

# Forward hybrid pass 3
old = '''            fwd_h_result = pwc.run_sieve_pass(
                prng_type    = prng_hybrid,
                residues     = residues,
                total_seeds  = total_seeds,
                threshold    = forward_threshold,
                window_size  = ws,
                dataset_path = dataset_path,
                output_file  = f"results/window_opt_forward_hybrid_{ws}_{off}_t{trial_number}.json",
            )'''

new = '''            fwd_h_result = pwc.run_sieve_pass(
                prng_type    = prng_hybrid,
                residues     = residues,
                total_seeds  = total_seeds,
                threshold    = forward_threshold,
                window_size  = ws,
                dataset_path = dataset_path,
                output_file  = f"results/window_opt_forward_hybrid_{ws}_{off}_t{trial_number}.json",
                offset       = config.offset,
                sessions     = list(config.sessions) if hasattr(config, 'sessions') else ["midday", "evening"],
                skip_range   = [config.skip_min, config.skip_max] if hasattr(config, 'skip_min') else [0, 147],
            )'''

if old in content:
    content = content.replace(old, new, 1)
    print('✅ Fix 3c: forward hybrid — added offset, sessions, skip_range')
    fixes += 1
else:
    print('⚠️  Fix 3c: forward hybrid not found')

# Reverse hybrid pass 4
old = '''            rev_h_result = pwc.run_sieve_pass(
                prng_type    = prng_hybrid_rev,
                residues     = residues,
                total_seeds  = total_seeds,
                threshold    = reverse_threshold,
                window_size  = ws,
                dataset_path = dataset_path,
                output_file  = f"results/window_opt_reverse_hybrid_{ws}_{off}_t{trial_number}.json",
            )'''

new = '''            rev_h_result = pwc.run_sieve_pass(
                prng_type    = prng_hybrid_rev,
                residues     = residues,
                total_seeds  = total_seeds,
                threshold    = reverse_threshold,
                window_size  = ws,
                dataset_path = dataset_path,
                output_file  = f"results/window_opt_reverse_hybrid_{ws}_{off}_t{trial_number}.json",
                offset       = config.offset,
                sessions     = list(config.sessions) if hasattr(config, 'sessions') else ["midday", "evening"],
                skip_range   = [config.skip_min, config.skip_max] if hasattr(config, 'skip_min') else [0, 147],
            )'''

if old in content:
    content = content.replace(old, new, 1)
    print('✅ Fix 3d: reverse hybrid — added offset, sessions, skip_range')
    fixes += 1
else:
    print('⚠️  Fix 3d: reverse hybrid not found')

TARGET.write_text(content, encoding='utf-8')
print(f'\nTotal fixes applied: {fixes}/6')
if fixes == 6:
    print('✅ ALL FIXES APPLIED')
else:
    print('⚠️  Some fixes not applied — check anchors')
