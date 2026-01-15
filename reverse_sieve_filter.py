#!/usr/bin/env python3
"""
================================================================================
REVERSE SIEVE FILTER - GPU WORKER SCRIPT
================================================================================

⚠️  CRITICAL USAGE WARNING ⚠️

This is a WORKER SCRIPT designed to be called internally by the coordinator
system. It should NOT be run directly by users.

INCORRECT USAGE (causes segfault):
  ❌ python3 reverse_sieve_filter.py --job-file test.json --gpu-id 0

CORRECT USAGE:
  ✅ Use run_bidirectional_test() from window_optimizer_integration_final.py

Example:
  from coordinator import MultiGPUCoordinator
  from window_optimizer_integration_final import run_bidirectional_test
  from window_optimizer import WindowConfig
  
  coordinator = MultiGPUCoordinator('distributed_config.json')
  config = WindowConfig(window_size=244, offset=139, sessions=['evening'],
                        skip_min=3, skip_max=29)
  
  result = run_bidirectional_test(
      coordinator=coordinator,
      config=config,
      dataset_path='daily3.json',
      seed_start=0,
      seed_count=100_000,
      prng_base='java_lcg',
      threshold=0.012
  )

See test_100k_bidirectional_CORRECT.py for a complete working example.

Revision History:
  2025-11-03: Added usage documentation after discovering correct pattern
  
================================================================================
"""

import argparse
import json
import time
import os
import sys
from typing import List, Dict, Any, Optional, Tuple
import socket

# ROCm environment setup
HOST = socket.gethostname()
if HOST in ["rig-6600", "rig-6600b"]:
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    os.environ.setdefault("HSA_ENABLE_SDMA", "0")
os.environ.setdefault("ROCM_PATH", "/opt/rocm")
os.environ.setdefault("HIP_PATH", "/opt/rocm")

# GPU backend
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    print("ERROR: CuPy not available - GPU required", file=sys.stderr)
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



# Load kernels from prng_registry
try:
    from prng_registry import KERNEL_REGISTRY, get_kernel_info
except ImportError:
    print("ERROR: prng_registry.py not found", file=sys.stderr)
    sys.exit(1)

# Load strategies for hybrid mode
try:
    from hybrid_strategy import get_all_strategies, StrategyConfig
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False
    print("WARNING: hybrid_strategy.py not available - hybrid mode disabled", file=sys.stderr)

# ============================================================================
# DATASET LOADING
# ============================================================================
def load_draws_from_daily3(path: str, window_size: int = 30, sessions=None, offset: int = 0):
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
# GPU REVERSE SIEVE ENGINE
# ============================================================================
class GPUReverseSieve:
    def __init__(self, gpu_id: int = 0):
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy not available")
        self.gpu_id = gpu_id
        self.device = cp.cuda.Device(gpu_id)
        self.compiled_kernels = {}
        print(f"GPUReverseSieve initialized on GPU {gpu_id}", file=sys.stderr)

    def _get_kernel(self, prng_family: str):
        cache_key = f"{prng_family}_reverse"
        if cache_key in self.compiled_kernels:
            return self.compiled_kernels[cache_key]
        kernel_name = f"{prng_family}_reverse"
        if kernel_name not in KERNEL_REGISTRY:
            raise ValueError(f"Reverse kernel not available for {prng_family}")
        config = get_kernel_info(kernel_name)
        kernel = cp.RawKernel(config['kernel_source'], config['kernel_name'])
        self.compiled_kernels[cache_key] = (kernel, config)
        return kernel, config

    def run_reverse_sieve(self,
                          candidate_seeds: List[Dict],
                          prng_family: str,
                          draws: List[int],
                          skip_range: Tuple[int, int] = (0, 20),
                          min_match_threshold: float = 0.01,
                          offset: int = 0) -> Dict[str, Any]:
        start_time = time.time()
        with self.device:
            seeds_array = [c["seed"] if isinstance(c, dict) else c for c in candidate_seeds]
            skips_array = [c.get("skip", 0) if isinstance(c, dict) else 0 for c in candidate_seeds]
            n_candidates = len(candidate_seeds)
            k = len(draws)

            candidate_seeds_gpu = cp.array(seeds_array, dtype=cp.uint32)
            candidate_skips_gpu = cp.array(skips_array, dtype=cp.uint8)
            residues = cp.array(draws, dtype=cp.uint32)
            survivors = cp.zeros(n_candidates, dtype=cp.uint32)
            match_rates = cp.zeros(n_candidates, dtype=cp.float32)
            used_skips = cp.zeros(n_candidates, dtype=cp.uint8)
            survivor_count = cp.zeros(1, dtype=cp.uint32)

            kernel, config = self._get_kernel(prng_family)
            threads_per_block = 256
            blocks = (n_candidates + threads_per_block - 1) // threads_per_block
            kernel(
                (blocks,), (threads_per_block,),
                (candidate_seeds_gpu, candidate_skips_gpu, residues, survivors,
                 match_rates, used_skips, survivor_count, cp.int32(n_candidates), cp.int32(k),
                 cp.float32(min_match_threshold), cp.int32(offset))
            )

            count = int(survivor_count[0].get())
            survivor_records = []
            if count > 0:
                survivors_cpu = survivors[:count].get()
                rates_cpu = match_rates[:count].get()
                skips_cpu = used_skips[:count].get()
                for i in range(count):
                    survivor_records.append({
                        'seed': int(survivors_cpu[i]),
                        'skip': int(skips_cpu[i]),
                        'match_rate': float(rates_cpu[i])
                    })
            duration_ms = (time.time() - start_time) * 1000
            return {
                'family': f"{prng_family}_reverse",
                'survivors': survivor_records,
                'stats': {
                    'candidates_tested': n_candidates,
                    'survivors_found': count,
                    'duration_ms': duration_ms,
                    'device': f'GPU_{self.gpu_id}'
                }
            }

    def run_hybrid_reverse_sieve(self, *args, **kwargs):
        # Keep as-is — not used
        pass

# ============================================================================
# JOB EXECUTION
# ============================================================================
def execute_reverse_job(job: Dict[str, Any], gpu_id: int) -> Dict[str, Any]:
    job_id = job.get('job_id', 'unknown')
    try:
        dataset_path = job.get('dataset_path') or job.get('target_file')
        window_size = job.get('window_size', 30)
        min_match_threshold = job.get('min_match_threshold', 0.01)
        offset = job.get('offset', 0)
        sessions = job.get('sessions', ['midday', 'evening'])
        prng_families = job.get('prng_families', ['mt19937'])

        candidate_seeds = job.get('candidate_seeds')
        if not candidate_seeds:
            raise ValueError("candidate_seeds required for reverse sieve")

        draws = load_draws_from_daily3(dataset_path, window_size, sessions, offset)
        if not draws:
            raise ValueError("No draws loaded from dataset")

        sieve = GPUReverseSieve(gpu_id=gpu_id)
        per_family_results = []
        all_survivors = []

        for family in prng_families:
            result = sieve.run_reverse_sieve(
                candidate_seeds=candidate_seeds,  # PASS FULL DICTS
                prng_family=family,
                draws=draws,
                skip_range=(0, 0),                # IGNORED
                min_match_threshold=min_match_threshold,
                offset=offset
            )
            per_family_results.append(result)
            all_survivors.extend(result['survivors'])

        total_duration = sum(r['stats']['duration_ms'] for r in per_family_results)
        final_result = {
            'job_id': job_id,
            'success': True,
            'mode': 'reverse_sieve',
            'prng_families': [r['family'] for r in per_family_results],
            'candidates_tested': len(candidate_seeds),
            'window_size': window_size,
            'skip_range': [0, 0],
            'min_match_threshold': min_match_threshold,
            'survivors': all_survivors,
            'stats': {
                'total_candidates': len(candidate_seeds),
                'total_survivors': len(all_survivors),
                'duration_ms': total_duration
            }
        }
        return final_result
    except Exception as e:
        import traceback
        return {
            'job_id': job_id,
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }

# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-file', required=True)
    parser.add_argument('--gpu-id', type=int, default=0)
    args = parser.parse_args()

    with open(args.job_file, 'r') as f:
        job = json.load(f)

    result = execute_reverse_job(job, args.gpu_id)
    result_file = f"result_{job.get('job_id', 'unknown')}.json"

    # === NEW: Save results in new format ===
    try:
        from integration.sieve_integration import save_reverse_sieve_results
        save_reverse_sieve_results(
            survivors=result.get('survivors', []),
            config={
                'prng_type': result.get('prng_families', ['unknown'])[0] if result.get('prng_families') else 'unknown',
                'seed_start': 0,
                'seed_end': result.get('stats', {}).get('total_candidates', 0),
                'total_seeds': result.get('candidates_tested', 0),
                'window_size': result.get('window_size', 0),
                'offset': 0,
                'skip_min': 0,
                'skip_max': 0,
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
