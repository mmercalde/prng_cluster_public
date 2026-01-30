#!/usr/bin/env python3
"""
GPU Residue Sieve - Main Engine
Flexible, modular, standalone sieve for PRNG seed discovery
Usage: python3 sieve_filter.py --job-file job.json --gpu-id 0
Compatible with coordinator.py and unified_system.py
VERSION HISTORY:
================================================================================
Version 2.3.1 - October 29, 2025
- CRITICAL FIX: Fixed broken control flow in execute_sieve_job
  * Removed misplaced 'continue' statement that was breaking hybrid mode
  * Fixed indentation in two-phase hybrid execution
  * Reason: Phase 1 code was being skipped, causing NameError
Version 2.3 - October 29, 2025
- CRITICAL FIX: Fixed hardcoded 512 buffer in run_hybrid_sieve allocation
- CRITICAL FIX: Fixed skip_sequences reshape to use survivor count
- FIX: Added missing config variable in run_hybrid_sieve
- FIX: Corrected ArgumentParser syntax error
- ENHANCEMENT: Window size now fully dynamic from job config
================================================================================
"""

# --- ROCm env prelude: set BEFORE any CuPy/HIP import ---
import os, socket
HOST = socket.gethostname()
# Force ROCm compatibility on RX 6600/XT rigs
if HOST in ("rig-6600", "rig-6600b"):
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    os.environ.setdefault("HSA_ENABLE_SDMA", "0")
# Common ROCm include/lib search (harmless if already set)
os.environ.setdefault("ROCM_PATH", "/opt/rocm")
os.environ.setdefault("HIP_PATH", "/opt/rocm")
os.environ["PATH"] = "/opt/rocm/bin:" + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = "/opt/rocm/lib:/opt/rocm/hip/lib:" + os.environ.get("LD_LIBRARY_PATH","")
os.environ["CPATH"] = "/opt/rocm/include:/opt/rocm/hip/include:" + os.environ.get("CPATH","")

# Now safe to import everything else
from adaptive_thresholds import estimate_background_thresholds, coerce_threshold
import argparse
import json
import time
import sys
from typing import List, Dict, Any, Optional, Tuple
# Import PRNG registry
try:
    from prng_registry import KERNEL_REGISTRY, get_kernel_info, list_available_prngs
except ImportError:
    print("ERROR: prng_registry.py not found - must be in same directory", file=sys.stderr)
    sys.exit(1)
# GPU backend
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    print("ERROR: CuPy not available - GPU required for sieve", file=sys.stderr)
    sys.exit(1)
import numpy as np

def _best_effort_gpu_cleanup():
    """Clean GPU memory after job completion - safe, best-effort"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


# ============================================================================
# DATASET LOADING
# ============================================================================
def load_draws_from_daily3(path: str, window_size: int = 30, sessions=None, offset: int = 0):
    """Load draws and return exactly `window_size` values starting at `offset`."""
    import json
    with open(path, 'r') as f:
        data = json.load(f)
    if sessions:
        data = [e for e in data if e.get('session') in sessions]
    n = len(data)
    if n < window_size:
        raise ValueError(f"Dataset has only {n} entries, need at least {window_size}")
    start = max(0, min(int(offset), n - window_size))
    end = start + window_size
    window = data[start:end]
    draws = [int(entry.get("full_state", entry["draw"])) for entry in window]
    return draws
# ============================================================================
# GPU SIEVE ENGINE
# ============================================================================
class GPUSieve:
    """GPU-accelerated flexible residue sieve"""
    def __init__(self, gpu_id: int = 0):
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy not available")
        self.gpu_id = gpu_id
        # Always use device 0 - CUDA_VISIBLE_DEVICES has already isolated the correct GPU
        self.device = cp.cuda.Device(0)
        self.compiled_kernels = {}
    def _get_kernel(self, prng_family: str, custom_params: Optional[Dict] = None):
        """Get or compile kernel for PRNG family"""
        cache_key = f"{prng_family}_{hash(frozenset(custom_params.items()) if custom_params else 0)}"
        if cache_key in self.compiled_kernels:
            return self.compiled_kernels[cache_key]
        config = get_kernel_info(prng_family)
        kernel = cp.RawKernel(config['kernel_source'], config['kernel_name'])
        self.compiled_kernels[cache_key] = (kernel, config)
        return kernel, config
    def run_sieve(
        self,
        prng_family: str,
        seed_start: int,
        seed_end: int,
        residues: List[int],
        skip_range: Tuple[int, int] = (0, 16),
        min_match_threshold: float = 0.25,
        custom_params: Optional[Dict] = None,
        chunk_size: int = 1_000_000,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Run flexible residue sieve for specified PRNG family"""
        with self.device:
            kernel, config = self._get_kernel(prng_family, custom_params)
            # Prepare inputs
            k = len(residues)
            # Determine dtype based on PRNG seed_type (uint64 for java_lcg, etc.)
            seed_type = config.get("seed_type", "uint32")
            residue_dtype = cp.uint32  # Residues are always 32-bit output values
            # TEMPORAL REVERSAL: Reverse residues for _reverse kernels
            if '_reverse' in prng_family:
                residues_reversed = residues[::-1]
                residues_gpu = cp.array(residues_reversed, dtype=residue_dtype)
            else:
                residues_gpu = cp.array(residues, dtype=residue_dtype)
            skip_min, skip_max = skip_range
            # Result containers
            all_survivors = []
            all_match_rates = []
            all_best_skips = []
            total_tested = 0
            start_time = time.time()
            # Process in chunks
            for chunk_start in range(seed_start, seed_end, chunk_size):
                chunk_end = min(chunk_start + chunk_size, seed_end)
                n_seeds = chunk_end - chunk_start
                # Allocate arrays
                seed_type = config.get("seed_type", "uint32")
                dtype = cp.uint64 if seed_type == "uint64" else cp.uint32
                seeds_gpu = cp.arange(chunk_start, chunk_end, dtype=dtype)
                survivors_gpu = cp.zeros(n_seeds, dtype=dtype)
                match_rates_gpu = cp.zeros(n_seeds, dtype=cp.float32)
                best_skips_gpu = cp.zeros(n_seeds, dtype=cp.uint8)
                survivor_count_gpu = cp.zeros(1, dtype=cp.uint32)
                # Launch kernel
                threads_per_block = 256
                blocks = (n_seeds + threads_per_block - 1) // threads_per_block
                # Build kernel arguments
                kernel_args = [
                    seeds_gpu, residues_gpu, survivors_gpu,
                    match_rates_gpu, best_skips_gpu, survivor_count_gpu,
                    n_seeds, k, skip_min, skip_max, cp.float32(min_match_threshold)
                ]
                # Add PRNG-specific parameters
                default_params = config.get("default_params", {})
                if prng_family == 'xorshift32':
                    kernel_args.append(cp.int32(default_params.get("shift_a", 13)))
                    kernel_args.append(cp.int32(default_params.get("shift_b", 17)))
                    kernel_args.append(cp.int32(default_params.get("shift_c", 5)))
                elif prng_family == 'pcg32':
                    kernel_args.append(cp.uint64(default_params.get("increment", 1442695040888963407)))
                elif prng_family == 'lcg32':
                    kernel_args.append(cp.uint32(default_params.get("a", 1664525)))
                    kernel_args.append(cp.uint32(default_params.get("c", 1013904223)))
                    kernel_args.append(cp.uint32(default_params.get("m", 0xFFFFFFFF)))
                elif prng_family == 'java_lcg':
                    kernel_args.append(cp.uint64(default_params.get("a", 25214903917)))
                    kernel_args.append(cp.uint64(default_params.get("c", 11)))
                elif prng_family == 'minstd':
                    kernel_args.append(cp.uint32(default_params.get("a", 48271)))
                    kernel_args.append(cp.uint32(default_params.get("m", 2147483647)))
                elif prng_family == 'xorshift128':
                    kernel_args.append(cp.int32(0))
                    kernel_args.append(cp.int32(0))
                    kernel_args.append(cp.int32(0))
                elif 'hybrid' in prng_family:
                    strategies = custom_params.get('strategies', [
                        {'max_misses': 3, 'tolerance': 5},
                        {'max_misses': 5, 'tolerance': 10},
                        {'max_misses': 8, 'tolerance': 15}
                    ]) if custom_params else [
                        {'max_misses': 3, 'tolerance': 5},
                        {'max_misses': 5, 'tolerance': 10},
                        {'max_misses': 8, 'tolerance': 15}
                    ]
                    n_strategies = len(strategies)
                    max_misses = cp.array([s['max_misses'] for s in strategies], dtype=cp.int32)
                    tolerances = cp.array([s['tolerance'] for s in strategies], dtype=cp.int32)
                    kernel_args.append(max_misses)
                    kernel_args.append(tolerances)
                    kernel_args.append(cp.int32(n_strategies))
                # Add offset parameter LAST
                kernel_args.append(cp.int32(offset))
                # Execute kernel
                kernel((blocks,), (threads_per_block,), tuple(kernel_args))
                # Collect survivors
                count = int(survivor_count_gpu[0].get())
                if count > 0:
                    survivors = survivors_gpu[:count].get().tolist()
                    rates = match_rates_gpu[:count].get().tolist()
                    skips = best_skips_gpu[:count].get().tolist()
                    for i, rate in enumerate(rates):
                        if rate >= min_match_threshold:
                            all_survivors.append(survivors[i])
                            all_match_rates.append(rate)
                            all_best_skips.append(skips[i])
                total_tested += n_seeds
            duration_ms = (time.time() - start_time) * 1000
            # Build detailed survivor records
            survivor_records = []
            for seed, rate, skip in zip(all_survivors, all_match_rates, all_best_skips):
                matches = int(rate * k)
                survivor_records.append({
                    'seed': int(seed),
                    'family': prng_family,
                    'match_rate': float(rate),
                    'matches': matches,
                    'total': k,
                    'best_skip': int(skip)
                })
            return {
                'family': prng_family,
                'seed_range': {'start': seed_start, 'end': seed_end},
                'survivors': survivor_records,
                'stats': {
                    'seeds_tested': total_tested,
                    'survivors_found': len(survivor_records),
                    'duration_ms': duration_ms,
                    'seeds_per_sec': total_tested / (duration_ms / 1000) if duration_ms > 0 else 0
                }
            }
    def run_hybrid_sieve(
        self,
        prng_family: str,
        seed_start: int,
        seed_end: int,
        residues: List[int],
        strategies: List[Dict[str, Any]],
        min_match_threshold: float = 0.25,
        chunk_size: int = 100_000,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Run hybrid multi-strategy variable skip sieve"""
        # Import strategy helper
        try:
            from hybrid_strategy import analyze_skip_pattern
        except ImportError:
            print("WARNING: hybrid_strategy module not found, using basic analysis", file=sys.stderr)
            def analyze_skip_pattern(pattern):
                import statistics
                return {
                    'min': min(pattern) if pattern else 0,
                    'max': max(pattern) if pattern else 0,
                    'avg': statistics.mean(pattern) if pattern else 0,
                    'variance': statistics.variance(pattern) if len(pattern) > 1 else 0,
                    'std_dev': statistics.stdev(pattern) if len(pattern) > 1 else 0,
                }
        # Check if this PRNG supports hybrid mode
        from prng_registry import get_kernel_info
        prng_config = get_kernel_info(prng_family)
        if not prng_config.get('variable_skip', False):
            raise ValueError(f"Hybrid sieve requires a PRNG with variable_skip support. "
                             f"{prng_family} does not support variable skip patterns. "
                             f"Try using {prng_family.replace('_hybrid', '')} for constant skip.")
        with self.device:
            # Get kernel and config
            kernel, config = self._get_kernel(prng_family, None)
            k = len(residues)
            # Determine dtype based on PRNG seed_type (uint64 for java_lcg, etc.)
            seed_type = config.get("seed_type", "uint32")
            residue_dtype = cp.uint32  # Residues are always 32-bit output values
            # TEMPORAL REVERSAL: Reverse residues for _reverse kernels
            if '_reverse' in prng_family:
                residues_reversed = residues[::-1]
                residues_gpu = cp.array(residues_reversed, dtype=residue_dtype)
            else:
                residues_gpu = cp.array(residues, dtype=residue_dtype)
            # Prepare strategy parameters
            n_strategies = len(strategies)
            strategy_max_misses = cp.array([
                s['max_consecutive_misses'] if isinstance(s, dict) else s.max_consecutive_misses
                for s in strategies
            ], dtype=cp.int32)
            strategy_tolerances = cp.array([
                s['skip_tolerance'] if isinstance(s, dict) else s.skip_tolerance
                for s in strategies
            ], dtype=cp.int32)
            # Result containers
            all_survivors = []
            all_match_rates = []
            all_strategy_ids = []
            all_skip_sequences = []
            total_tested = 0
            start_time = time.time()
            # Process in smaller chunks for hybrid
            for chunk_start in range(seed_start, seed_end, chunk_size):
                chunk_end = min(chunk_start + chunk_size, seed_end)
                n_seeds = chunk_end - chunk_start
                # Allocate arrays
                seed_type = config.get("seed_type", "uint32")
                dtype = cp.uint64 if seed_type == "uint64" else cp.uint32
                seeds_gpu = cp.arange(chunk_start, chunk_end, dtype=dtype)
                survivors_gpu = cp.zeros(n_seeds, dtype=dtype)
                match_rates_gpu = cp.zeros(n_seeds, dtype=cp.float32)
                strategy_ids_gpu = cp.zeros(n_seeds, dtype=cp.uint32)
                skip_sequences_gpu = cp.zeros(n_seeds * k, dtype=cp.uint32)
                survivor_count_gpu = cp.zeros(1, dtype=cp.uint32)
                # Launch kernel
                threads_per_block = 256
                blocks = (n_seeds + threads_per_block - 1) // threads_per_block
                kernel_args = [
                    seeds_gpu, residues_gpu, survivors_gpu,
                    match_rates_gpu, skip_sequences_gpu, strategy_ids_gpu,
                    survivor_count_gpu, cp.int32(n_seeds), cp.int32(k),
                    strategy_max_misses, strategy_tolerances, cp.int32(n_strategies),
                    cp.float32(min_match_threshold),
                    # Add PRNG-specific params
                    *(
                        [] if '_reverse' in prng_family else
                        [cp.int32(config["default_params"]["shift_a"]),
                         cp.int32(config["default_params"]["shift_b"]),
                         cp.int32(config["default_params"]["shift_c"])]
                        if "xorshift32" in prng_family else
                        [cp.uint64(config["default_params"]["increment"])]
                        if "pcg32" in prng_family else
                        [cp.uint32(config["default_params"]["a"]),
                         cp.uint32(config["default_params"]["c"]),
                         cp.uint32(config["default_params"]["m"])]
                        if "lcg32" in prng_family else
                        [cp.uint64(config["default_params"]["a"]),
                         cp.uint64(config["default_params"]["c"])]
                        if "java_lcg" in prng_family else
                        [cp.uint32(config["default_params"]["a"]),
                         cp.uint32(config["default_params"]["m"])]
                        if "minstd" in prng_family else
                        [cp.int32(0), cp.int32(0), cp.int32(0)]
                        if "xorshift128" in prng_family else
                        []
                    ),
                    cp.int32(offset)
                ]
                kernel((blocks,), (threads_per_block,), tuple(kernel_args))
                cp.cuda.Device().synchronize()
                # Collect survivors
                count = int(survivor_count_gpu[0].get())
                if count > 0:
                    survivors = survivors_gpu[:count].get().tolist()
                    rates = match_rates_gpu[:count].get().tolist()
                    strat_ids = strategy_ids_gpu[:count].get().tolist()
                    skip_seqs_raw = skip_sequences_gpu[:count * k].get()
                    skip_seqs = skip_seqs_raw.reshape(count, k)
                    for i in range(count):
                        if rates[i] >= min_match_threshold:
                            all_survivors.append(survivors[i])
                            all_match_rates.append(rates[i])
                            all_strategy_ids.append(strat_ids[i])
                            skip_seq = skip_seqs[i, :k].tolist()
                            all_skip_sequences.append(skip_seq)
                total_tested += n_seeds
            duration_ms = (time.time() - start_time) * 1000
            # Build detailed survivor records
            survivor_records = []
            for seed, rate, strat_id, skip_seq in zip(
                all_survivors, all_match_rates, all_strategy_ids, all_skip_sequences
            ):
                matches = int(rate * k)
                skip_stats = analyze_skip_pattern(skip_seq)
                survivor_records.append({
                    'seed': int(seed),
                    'family': prng_family,
                    'match_rate': float(rate),
                    'matches': matches,
                    'total': k,
                    'strategy_id': int(strat_id),
                    'strategy_name': strategies[strat_id]['name'] if isinstance(strategies[strat_id], dict) else strategies[strat_id].name if strat_id < len(strategies) else 'unknown',
                    'skip_pattern': skip_seq,
                    'skip_stats': skip_stats
                })
            return {
                'family': prng_family,
                'seed_range': {'start': seed_start, 'end': seed_end},
                'survivors': survivor_records,
                'strategies_tested': n_strategies,
                'stats': {
                    'seeds_tested': total_tested,
                    'survivors_found': len(survivor_records),
                    'duration_ms': duration_ms,
                    'seeds_per_sec': total_tested / (duration_ms / 1000) if duration_ms > 0 else 0
                }
            }
# ============================================================================
# JOB EXECUTION
# ============================================================================
def execute_sieve_job(job: Dict[str, Any], gpu_id: int) -> Dict[str, Any]:
    """Execute a sieve job from job specification"""
    job_id = job.get('job_id', 'unknown')
    try:
        # Extract parameters
        dataset_path = job.get('dataset_path') or job.get('target_file')
        window_size = job.get('window_size', 10)
        print(f'🔍 SIEVE READ: window_size={window_size} from job file', file=sys.stderr)
        if 'seeds' in job:
            seed_start = job['seeds'][0]
            seed_end = job['seeds'][-1] + 1
        else:
            seed_start = job.get('seed_start', 0)
            seed_end = job.get('seed_end', 100000)
        skip_range = tuple(job.get('skip_range', [0, 16]))
        print(f'🔍 SIEVE READ: skip_range={skip_range} from job file', file=sys.stderr)
        min_match_threshold = job.get('min_match_threshold', 0.25)
        offset = job.get('offset', 0)
        sessions = job.get('sessions', ['midday', 'evening'])
        # Load draws
        draws = load_draws_from_daily3(dataset_path, window_size, sessions, offset)
        print(f'🔍 LOADED DRAWS: len(draws)={len(draws)}, requested window_size={window_size}', file=sys.stderr)
        if not draws:
            raise ValueError("No draws loaded from dataset")
        # Get PRNG families to test
        prng_families = job.get('prng_families')
        if not prng_families:
            prng_families = ['xorshift32', 'pcg32', 'mt19937']
        # Initialize sieve
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
            # Check if hybrid mode is enabled
            use_hybrid = job.get('hybrid', False)
            # Check if this family supports hybrid mode
            from prng_registry import get_kernel_info
            family_config = get_kernel_info(family_name)
            supports_hybrid = family_config.get('variable_skip', False)
            if use_hybrid and supports_hybrid:
                print(f"Testing {family_name} in HYBRID mode...", file=sys.stderr)
                # Get strategies from job
                strategies_data = job.get('strategies')
                if not strategies_data:
                    try:
                        from hybrid_strategy import get_all_strategies
                        strategies = get_all_strategies()
                    except ImportError:
                        print("WARNING: Hybrid mode requested but strategies not available", file=sys.stderr)
                        print(" Falling back to standard mode", file=sys.stderr)
                        use_hybrid = False
                        strategies = None
                else:
                    from hybrid_strategy import StrategyConfig
                    strategies = [StrategyConfig(**s) for s in strategies_data]
                if use_hybrid and strategies:
                    # Check if single-phase or two-phase hybrid
                    is_single_phase = ('_hybrid' in family_name)
                    if is_single_phase:
                        # Single-phase hybrid
                        print(f" Running SINGLE-PHASE HYBRID for {family_name}...", file=sys.stderr)
                        phase2_threshold = coerce_threshold(job.get('phase2_threshold', 'auto'), 0.50)
                        result = sieve.run_hybrid_sieve(
                            prng_family=family_name,
                            seed_start=seed_start,
                            seed_end=seed_end,
                            residues=draws,
                            strategies=strategies,
                            min_match_threshold=phase2_threshold,
                            offset=offset
                        )
                        result['single_phase'] = {
                            'threshold': round(phase2_threshold, 4),
                            'strategies_tested': len(strategies)
                        }
                    else:
                        # Two-phase hybrid (Phase 1: fixed skip, Phase 2: hybrid)
                        print(f" Running TWO-PHASE HYBRID for {family_name}...", file=sys.stderr)
                        phase1_in = job.get('phase1_threshold', 'auto')
                        phase2_in = job.get('phase2_threshold', 'auto')
                        phase1_threshold = coerce_threshold(phase1_in, 0.20)
                        phase2_threshold = coerce_threshold(phase2_in, 0.75)
                        # PHASE 1: Wide search with fixed skip
                        print(f" Phase 1: Fixed-skip search (threshold={phase1_threshold:.3f})...", file=sys.stderr)
                        phase1_start = time.time()
                        phase1_result = sieve.run_sieve(
                            prng_family=family_name,
                            seed_start=seed_start,
                            seed_end=seed_end,
                            residues=draws,
                            skip_range=skip_range,
                            min_match_threshold=phase1_threshold,
                            offset=offset
                        )
                        phase1_duration = (time.time() - phase1_start) * 1000
                        phase1_survivors = phase1_result.get('survivors', [])
                        print(f" Phase 1 complete: {len(phase1_survivors)} candidates ({phase1_duration:.1f}ms)", file=sys.stderr)
                        # PHASE 2: Refine with hybrid on candidates
                        if phase1_survivors:
                            print(f" Phase 2: Hybrid refinement (threshold={phase2_threshold:.3f})...", file=sys.stderr)
                            phase2_start = time.time()
                            # Extract seed list from phase1 survivors
                            phase1_seeds = [s['seed'] for s in phase1_survivors]
                            hybrid_family = family_name + '_hybrid'
                            # Run hybrid on candidate seeds
                            phase2_result = sieve.run_hybrid_sieve(
                                prng_family=hybrid_family,
                                seed_start=min(phase1_seeds),
                                seed_end=max(phase1_seeds) + 1,
                                residues=draws,
                                strategies=strategies,
                                min_match_threshold=phase2_threshold,
                                offset=offset
                            )
                            phase2_duration = (time.time() - phase2_start) * 1000
                            phase2_survivors = phase2_result.get('survivors', [])
                            print(f" Phase 2 complete: {len(phase2_survivors)} survivors ({phase2_duration:.1f}ms)", file=sys.stderr)
                            result = {
                                'family': family_name,
                                'seed_range': {'start': seed_start, 'end': seed_end},
                                'survivors': phase2_survivors,
                                'two_phase': {
                                    'phase1_threshold': round(phase1_threshold, 4),
                                    'phase1_candidates': len(phase1_survivors),
                                    'phase1_duration_ms': round(phase1_duration, 2),
                                    'phase2_threshold': round(phase2_threshold, 4),
                                    'phase2_survivors': len(phase2_survivors),
                                    'phase2_duration_ms': round(phase2_duration, 2),
                                    'total_duration_ms': round(phase1_duration + phase2_duration, 2)
                                },
                                'stats': {
                                    'seeds_tested': phase1_result['stats']['seeds_tested'],
                                    'survivors_found': len(phase2_survivors),
                                    'duration_ms': phase1_duration + phase2_duration,
                                    'seeds_per_sec': phase1_result['stats']['seeds_per_sec']
                                }
                            }
                        else:
                            print(f" Phase 1 found no candidates, skipping Phase 2", file=sys.stderr)
                            result = phase1_result
                    per_family_results.append(result)
                    all_survivors.extend(result.get('survivors', []))
                    continue # Move to next PRNG family
            # Standard (non-hybrid) sieve
            print(f"Testing {family_name} (standard mode)...", file=sys.stderr)
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
            all_survivors.extend(result['survivors'])
        # Compile final result
        total_tested = sum(r.get('stats', {}).get('seeds_tested', 0) for r in per_family_results)
        total_duration = sum(r.get('stats', {}).get('duration_ms', 0) for r in per_family_results)
        final_result = {
            'job_id': job_id,
            'success': True,
            'prng_families': [r.get('family', 'unknown') for r in per_family_results],
            'seed_range': {'start': seed_start, 'end': seed_end},
            'k': len(draws),
            'skip_range': list(skip_range),
            'min_match_threshold': min_match_threshold,
            'survivors': all_survivors,
            'stats': {
                'total_seeds_tested': total_tested,
                'total_survivors': len(all_survivors),
                'duration_ms': total_duration,
                'avg_seeds_per_sec': total_tested / (total_duration / 1000) if total_duration > 0 else 0
            },
            'per_family': {
                r['family']: {
                    **{k: v for k, v in r.items() if k not in ['family', 'seed_range', 'survivors', 'stats']},
                    'survivors': r.get('survivors', []),
                    'tested': r.get('stats', {}).get('seeds_tested', 0),
                    'found': r.get('stats', {}).get('survivors_found', 0),
                    'duration_ms': r.get('stats', {}).get('duration_ms', 0)
                }
                for r in per_family_results
            },
            'seeds_analyzed': total_tested
        }
        return final_result
    except Exception as e:
        import traceback
        return {
            'job_id': job_id,
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'seeds_analyzed': 0
        }
# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='GPU Residue Sieve - Flexible PRNG seed discovery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available PRNG families:
{chr(10).join(' - ' + name for name in list_available_prngs())}
Job file format (JSON):
{{
  "job_id": "sieve_001",
  "dataset_path": "daily3.json",
  "seed_start": 0,
  "seed_end": 1000000,
  "window_size": 10,
  "min_match_threshold": 0.25,
  "skip_range": [0, 16],
  "prng_families": ["xorshift32", "pcg32"],
  "sessions": ["midday", "evening"]
}}
        """
    )
    parser.add_argument('--job-file', required=True, help='Job specification JSON file')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU device ID')
    parser.add_argument('--list-prngs', action='store_true', help='List available PRNG families')
    args = parser.parse_args()
    if args.list_prngs:
        print("Available PRNG families:")
        for name in list_available_prngs():
            config = get_kernel_info(name)
            print(f" {name:25} - {config['description']}")
        return 0
    try:
        with open(args.job_file, 'r') as f:
            job = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load job file: {e}", file=sys.stderr)
        return 1
    job_id = job.get('job_id', 'unknown')
    result = execute_sieve_job(job, args.gpu_id)
    # Old result_{job_id}.json writing removed - now using new results_manager system
    # === NEW: Save results in new format ===
    try:
        from integration.sieve_integration import save_forward_sieve_results
        save_forward_sieve_results(
            survivors=result.get('survivors', []),
            config={
                'prng_type': result.get('prng_families', ['unknown'])[0] if result.get('prng_families') else 'unknown',
                'seed_start': job.get('seed_start', 0),
                'seed_end': job.get('seed_end', 0),
                'total_seeds': job.get('seed_end', 0) - job.get('seed_start', 0),  # FIX: Calculate from job range
                'window_size': job.get('window_size', 0),  # FIX: Read from job
                'offset': job.get('offset', 0),  # FIX: Read from job
                'skip_min': result.get('skip_range', [0, 0])[0],
                'skip_max': result.get('skip_range', [0, 0])[1],
                'threshold': job.get('min_match_threshold', 0),
                'dataset': job.get('dataset_path', 'unknown'),
                'sessions': job.get('sessions', [])
            },
            execution_time=result.get('stats', {}).get('duration_ms', 0) / 1000.0
        )
    except Exception as e:
        print(f"Note: New results format unavailable: {e}")
    _best_effort_gpu_cleanup()
    print(json.dumps(result))
    return 0 if result['success'] else 1
if __name__ == '__main__':
    sys.exit(main())
