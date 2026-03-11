#!/usr/bin/env python3
"""
Persistent GPU Sieve Worker - S129B
Boots ROCm/CUDA once, processes multiple jobs via stdin/stdout JSON-lines IPC.
Eliminates per-job SSH launch + ROCm init overhead (~80% of job wall time).

Architecture:
  Coordinator sends job JSON to worker stdin (one line per job)
  Worker processes job, writes result JSON to stdout (one line per result)
  Worker stays alive for entire run duration

IPC Protocol:
  Input:  {"command": "sieve", "job": {...job_spec...}}
          {"command": "shutdown"}
  Output: {"status": "ready", "gpu_id": N, "device": "..."}   (on startup)
          {"status": "ok", "job_id": "...", "result": {...}}   (job complete)
          {"status": "error", "job_id": "...", "error": "..."}  (job failed)

Usage:
  ROCR_VISIBLE_DEVICES=0 python3 sieve_gpu_worker.py --gpu-id 0
"""

import os, sys, socket, json, time, traceback

HOST = socket.gethostname()

# ROCm env prelude - must be before any CuPy/HIP import
if HOST in ("rig-6600", "rig-6600b", "rig-6600c"):
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    os.environ.setdefault("HSA_ENABLE_SDMA", "0")
os.environ.setdefault("ROCM_PATH", "/opt/rocm")
os.environ.setdefault("HIP_PATH", "/opt/rocm")
os.environ["PATH"] = "/opt/rocm/bin:" + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = "/opt/rocm/lib:/opt/rocm/hip/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["CPATH"] = "/opt/rocm/include:/opt/rocm/hip/include:" + os.environ.get("CPATH", "")

import argparse
import signal

# ── stdout must be line-buffered for reliable IPC ──
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)


def _emit(obj: dict):
    """Write JSON line to stdout and flush."""
    print(json.dumps(obj), flush=True)


def _log(msg: str):
    """Write log line to stderr (never pollutes stdout IPC channel)."""
    print(f"[sieve_worker] {msg}", file=sys.stderr, flush=True)


# ── Late imports (after env is set) ──
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    from prng_registry import KERNEL_REGISTRY, get_kernel_info, list_available_prngs
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False

try:
    from adaptive_thresholds import estimate_background_thresholds, coerce_threshold
except ImportError:
    def coerce_threshold(v, default):
        return float(v) if isinstance(v, (int, float)) else default


# ============================================================================
# GPU CLEANUP
# ============================================================================
def _best_effort_gpu_cleanup():
    try:
        import gc; gc.collect()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


# ============================================================================
# DATASET LOADER (cached - load once per unique path+window+offset)
# ============================================================================
_draw_cache = {}

def load_draws_cached(path: str, window_size: int, sessions, offset: int):
    key = (path, window_size, tuple(sessions or []), offset)
    if key in _draw_cache:
        return _draw_cache[key]
    with open(path, 'r') as f:
        data = json.load(f)
    if sessions:
        data = [e for e in data if e.get('session') in sessions]
    n = len(data)
    if n < window_size:
        raise ValueError(f"Dataset has only {n} entries, need {window_size}")
    start = max(0, min(int(offset), n - window_size))
    window = data[start:start + window_size]
    draws = [int(entry.get("full_state", entry["draw"])) for entry in window]
    _draw_cache[key] = draws
    _log(f"Loaded {len(draws)} draws (cached key={key[:3]})")
    return draws


# ============================================================================
# KERNEL CACHE (compiled once, reused across jobs)
# ============================================================================
_kernel_cache = {}

def _get_kernel(prng_family: str):
    if prng_family in _kernel_cache:
        return _kernel_cache[prng_family]
    config = get_kernel_info(prng_family)
    kernel = cp.RawKernel(config['kernel_source'], config['kernel_name'])
    _kernel_cache[prng_family] = (kernel, config)
    _log(f"Compiled kernel: {prng_family}")
    return kernel, config


# ============================================================================
# SIEVE EXECUTION (extracted from sieve_filter.py, worker-adapted)
# ============================================================================
def run_sieve_job(job: dict) -> dict:
    """
    Execute one sieve job. Equivalent to GPUSieve.run_sieve() but:
    - Uses cached kernels (compiled once at worker startup)
    - Uses cached draw data
    - Always uses device 0 (ROCR_VISIBLE_DEVICES has isolated the GPU)
    """
    job_id = job.get('job_id', 'unknown')

    # Extract parameters
    dataset_path = job.get('dataset_path') or job.get('target_file')
    window_size  = job.get('window_size', 10)
    seed_start   = job.get('seed_start', 0)
    seed_end     = job.get('seed_end', seed_start + 100_000)
    skip_range   = tuple(job.get('skip_range', [0, 16]))
    threshold    = job.get('min_match_threshold', 0.25)
    offset       = job.get('offset', 0)
    sessions     = job.get('sessions', ['midday', 'evening'])
    prng_families= job.get('prng_families', ['java_lcg'])

    draws = load_draws_cached(dataset_path, window_size, sessions, offset)
    k = len(draws)

    device = cp.cuda.Device(0)
    all_survivors = []
    per_family = []

    with device:
        for family_spec in prng_families:
            family_name   = family_spec['type'] if isinstance(family_spec, dict) else family_spec
            custom_params = family_spec.get('params', {}) if isinstance(family_spec, dict) else None

            kernel, config = _get_kernel(family_name)
            seed_type = config.get("seed_type", "uint32")
            dtype = cp.uint64 if seed_type == "uint64" else cp.uint32
            residue_dtype = cp.uint32

            if '_reverse' in family_name:
                residues_gpu = cp.array(draws[::-1], dtype=residue_dtype)
            else:
                residues_gpu = cp.array(draws, dtype=residue_dtype)

            skip_min, skip_max = skip_range
            survivors_out = []
            t0 = time.time()
            n_seeds = seed_end - seed_start

            # Single chunk per job (coordinator already sized chunks correctly)
            seeds_gpu          = cp.arange(seed_start, seed_end, dtype=dtype)
            survivors_gpu      = cp.zeros(n_seeds, dtype=dtype)
            match_rates_gpu    = cp.zeros(n_seeds, dtype=cp.float32)
            best_skips_gpu     = cp.zeros(n_seeds, dtype=cp.uint8)
            survivor_count_gpu = cp.zeros(1, dtype=cp.uint32)

            threads = 256
            blocks  = (n_seeds + threads - 1) // threads

            # Build kernel args (mirrors sieve_filter.py exactly)
            default_params = config.get("default_params", {})
            kernel_args = [
                seeds_gpu, residues_gpu, survivors_gpu,
                match_rates_gpu, best_skips_gpu, survivor_count_gpu,
                n_seeds, k, skip_min, skip_max, cp.float32(threshold)
            ]
            if family_name == 'xorshift32':
                kernel_args += [cp.int32(default_params.get("shift_a", 13)),
                                cp.int32(default_params.get("shift_b", 17)),
                                cp.int32(default_params.get("shift_c", 5))]
            elif family_name == 'pcg32':
                kernel_args.append(cp.uint64(default_params.get("increment", 1442695040888963407)))
            elif family_name == 'lcg32':
                kernel_args += [cp.uint32(default_params.get("a", 1664525)),
                                cp.uint32(default_params.get("c", 1013904223)),
                                cp.uint32(default_params.get("m", 0xFFFFFFFF))]
            elif family_name in ('java_lcg', 'java_lcg_reverse'):
                # NOTE: hybrid variants handled separately below — do NOT add them here.
                # S133-B: hybrid kernel has completely different signature/buffers.
                kernel_args += [cp.uint64(default_params.get("a", 25214903917)),
                                cp.uint64(default_params.get("c", 11))]
            elif family_name in ('java_lcg_hybrid', 'java_lcg_hybrid_reverse'):
                # S134: Hybrid kernel — different buffer layout from constant-skip.
                # Ported from sieve_filter.py run_hybrid_sieve().
                strategies_data = job.get('strategies') or []
                if not strategies_data:
                    try:
                        from hybrid_strategy import get_all_strategies
                        strategies_data = [
                            {"max_consecutive_misses": s.max_consecutive_misses,
                             "skip_tolerance": s.skip_tolerance}
                            if not isinstance(s, dict) else s
                            for s in get_all_strategies()
                        ]
                    except ImportError:
                        strategies_data = [{"max_consecutive_misses": 3, "skip_tolerance": 5}]
                n_strategies        = len(strategies_data)
                strategy_max_misses = cp.array([s["max_consecutive_misses"] for s in strategies_data], dtype=cp.int32)
                strategy_tolerances = cp.array([s["skip_tolerance"]         for s in strategies_data], dtype=cp.int32)
                strategy_ids_gpu    = cp.zeros(n_seeds,     dtype=cp.uint32)
                skip_sequences_gpu  = cp.zeros(n_seeds * k, dtype=cp.uint32)
                # Rebuild kernel_args from scratch — hybrid signature is different
                kernel_args = [
                    seeds_gpu, residues_gpu, survivors_gpu,
                    match_rates_gpu, skip_sequences_gpu, strategy_ids_gpu,
                    survivor_count_gpu, cp.int32(n_seeds), cp.int32(k),
                    strategy_max_misses, strategy_tolerances, cp.int32(n_strategies),
                    cp.float32(threshold),
                    cp.uint64(default_params.get("a", 25214903917)),
                    cp.uint64(default_params.get("c", 11)),
                ]
                kernel((blocks,), (threads,), tuple(kernel_args))
                count = int(survivor_count_gpu[0].get())
                if count > 0:
                    s_arr   = survivors_gpu[:count].get().tolist()
                    r_arr   = match_rates_gpu[:count].get().tolist()
                    sid_arr = strategy_ids_gpu[:count].get().tolist()
                    ss_raw  = skip_sequences_gpu[:count * k].get().reshape(count, k).tolist()
                    for seed, rate, sid, ss in zip(s_arr, r_arr, sid_arr, ss_raw):
                        if rate >= threshold:
                            survivors_out.append({
                                'seed': int(seed), 'family': family_name,
                                'match_rate': float(rate),
                                'matches': int(rate * k), 'total': k,
                                'strategy_id': int(sid), 'skip_sequence': ss,
                            })
                duration_ms = (time.time() - t0) * 1000
                per_family.append({
                    'family': family_name, 'tested': n_seeds,
                    'found': len(survivors_out), 'duration_ms': round(duration_ms, 2),
                    'seeds_per_sec': int(n_seeds / (duration_ms/1000)) if duration_ms > 0 else 0
                })
                all_survivors.extend(survivors_out)
                continue  # skip generic kernel launch + append below
            elif family_name == 'minstd':
                kernel_args += [cp.uint32(default_params.get("a", 48271)),
                                cp.uint32(default_params.get("m", 2147483647))]
            elif family_name == 'xorshift128':
                kernel_args += [cp.int32(0), cp.int32(0), cp.int32(0)]
            kernel_args.append(cp.int32(offset))

            kernel((blocks,), (threads,), tuple(kernel_args))

            count = int(survivor_count_gpu[0].get())
            if count > 0:
                s_arr = survivors_gpu[:count].get().tolist()
                r_arr = match_rates_gpu[:count].get().tolist()
                k_arr = best_skips_gpu[:count].get().tolist()
                for seed, rate, skip in zip(s_arr, r_arr, k_arr):
                    if rate >= threshold:
                        survivors_out.append({
                            'seed': int(seed), 'family': family_name,
                            'match_rate': float(rate),
                            'matches': int(rate * k), 'total': k,
                            'best_skip': int(skip)
                        })

            duration_ms = (time.time() - t0) * 1000
            per_family.append({
                'family': family_name,
                'tested': n_seeds,
                'found': len(survivors_out),
                'duration_ms': round(duration_ms, 2),
                'seeds_per_sec': int(n_seeds / (duration_ms / 1000)) if duration_ms > 0 else 0
            })
            all_survivors.extend(survivors_out)

    _best_effort_gpu_cleanup()

    total_tested   = sum(f['tested'] for f in per_family)
    total_duration = sum(f['duration_ms'] for f in per_family)
    return {
        'job_id': job_id,
        'success': True,
        'survivors': all_survivors,
        'seed_range': {'start': seed_start, 'end': seed_end},
        'stats': {
            'total_seeds_tested': total_tested,
            'total_survivors': len(all_survivors),
            'duration_ms': round(total_duration, 2),
            'avg_seeds_per_sec': int(total_tested / (total_duration / 1000)) if total_duration > 0 else 0
        },
        'per_family': {f['family']: f for f in per_family}
    }


# ============================================================================
# WORKER MAIN LOOP
# ============================================================================
def run_worker(gpu_id: int):
    if not GPU_AVAILABLE:
        _emit({"status": "error", "error": "CuPy not available"})
        sys.exit(1)
    if not REGISTRY_AVAILABLE:
        _emit({"status": "error", "error": "prng_registry not found"})
        sys.exit(1)

    # Warm up GPU - touch device to trigger ROCm init NOW (not at first job)
    _log(f"Warming up GPU {gpu_id}...")
    with cp.cuda.Device(0):
        _ = cp.zeros(1, dtype=cp.float32)
        cp.cuda.Device(0).synchronize()
    _log(f"GPU ready")

    device_name = "unknown"
    try:
        device_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    except Exception:
        pass

    # Signal ready
    _emit({"status": "ready", "gpu_id": gpu_id, "device": device_name})

    # Job loop
    jobs_processed = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            _emit({"status": "error", "job_id": "parse_error", "error": str(e)})
            continue

        command = msg.get("command", "sieve")

        if command == "shutdown":
            _log(f"Shutdown received. Jobs processed: {jobs_processed}")
            _emit({"status": "shutdown", "jobs_processed": jobs_processed})
            break

        elif command == "sieve":
            job = msg.get("job", msg)  # support bare job or wrapped
            job_id = job.get("job_id", "unknown")
            try:
                t0 = time.time()
                result = run_sieve_job(job)
                elapsed = time.time() - t0
                jobs_processed += 1
                _emit({"status": "ok", "job_id": job_id,
                       "elapsed_s": round(elapsed, 3), "result": result})
            except Exception as e:
                _emit({"status": "error", "job_id": job_id,
                       "error": str(e), "traceback": traceback.format_exc()})
        else:
            _emit({"status": "error", "job_id": "unknown",
                   "error": f"Unknown command: {command}"})


def main():
    parser = argparse.ArgumentParser(description='Persistent GPU Sieve Worker (S129B)')
    parser.add_argument('--gpu-id', type=int, default=0, help='Logical GPU id (for logging)')
    parser.add_argument('--persistent', action='store_true', default=False,
                        help='[S134] Persistent worker mode — stay alive for multiple jobs via stdin/stdout IPC')
    args = parser.parse_args()

    # Graceful SIGTERM handler
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))

    run_worker(args.gpu_id)


if __name__ == '__main__':
    main()
