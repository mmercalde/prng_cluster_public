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
    # [S150-slim_v1] compact separators — zero whitespace overhead
    print(json.dumps(obj, separators=(',', ':')), flush=True)


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

# S163 — sampling-gated memory instrumentation (active when S163_MEM_DEBUG=1)
_S163_CHUNK_COUNTER = 0
_S163_RSS_BASELINE  = None
_S163_SAMPLE_EVERY  = 25    # log every 25 chunks
_S163_RSS_WARN_MB   = 200   # alert if RSS grows >200MB from baseline
_S163_POOL_WARN_MB  = 256   # alert if pool exceeds cap

def _s163_read_proc(pid):
    """Read VmRSS and VmSize from /proc/PID/status. Returns (rss_mb, vmsize_mb)."""
    try:
        status = open(f'/proc/{pid}/status').read()
        rss_kb    = int(status.split('VmRSS:')[1].split()[0])
        vmsize_kb = int(status.split('VmSize:')[1].split()[0])
        return rss_kb // 1024, vmsize_kb // 1024
    except Exception:
        return 0, 0

def _best_effort_gpu_cleanup():
    global _S163_CHUNK_COUNTER, _S163_RSS_BASELINE
    import os, gc
    _S163_CHUNK_COUNTER += 1
    n = _S163_CHUNK_COUNTER
    should_log = False

    # ── S163 instrumentation — gated, sampled every N chunks ─────────────
    if os.environ.get('S163_MEM_DEBUG', '0') == '1':
        try:
            pool         = cp.get_default_memory_pool()
            used_before  = pool.used_bytes()  / 1024**2
            total_before = pool.total_bytes() / 1024**2
            free_before  = pool.n_free_blocks()
            rss, vmsize  = _s163_read_proc(os.getpid())

            if _S163_RSS_BASELINE is None:
                _S163_RSS_BASELINE = rss

            rss_delta        = rss - _S163_RSS_BASELINE
            threshold_breach = (total_before > _S163_POOL_WARN_MB or
                                rss_delta    > _S163_RSS_WARN_MB)
            should_log       = (n <= 3 or n % _S163_SAMPLE_EVERY == 0 or threshold_breach)

            if should_log:
                tag = "WARN" if threshold_breach else "INFO"
                print(
                    f"[S163-MEM/{tag}] chunk={n} "
                    f"pool_used_before={used_before:.1f}MB "
                    f"pool_total_before={total_before:.1f}MB "
                    f"free_blocks={free_before} "
                    f"VmRSS={rss}MB VmSize={vmsize}MB rss_delta={rss_delta:+d}MB",
                    flush=True
                )
        except Exception:
            pass

    # ── actual cleanup ────────────────────────────────────────────────────
    try:
        gc.collect()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass

    # [S163] free_all_blocks() REMOVED — redundant since S155 pool cap (256MB).
    # S155 cap prevents VM bloat without requiring explicit pool release.
    # free_all_blocks() under concurrent multi-process load causes race condition
    # (CuPy issue #4866 — cudaErrorIllegalAddress). Removed as both redundant
    # and dangerous. Pool cap is the correct mechanism for memory control.

    # ── S163 post-cleanup instrumentation ────────────────────────────────
    if os.environ.get('S163_MEM_DEBUG', '0') == '1' and should_log:
        try:
            pool       = cp.get_default_memory_pool()
            used_after  = pool.used_bytes()  / 1024**2
            total_after = pool.total_bytes() / 1024**2
            free_after  = pool.n_free_blocks()
            print(
                f"[S163-MEM/AFTER] chunk={n} "
                f"pool_used_after={used_after:.1f}MB "
                f"pool_total_after={total_after:.1f}MB "
                f"free_blocks_after={free_after}",
                flush=True
            )
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
def run_sieve_job(job: dict, gpu_id: int = 0) -> dict:
    """
    Execute one sieve job. Equivalent to GPUSieve.run_sieve() but:
    - Uses cached kernels (compiled once at worker startup)
    - Uses cached draw data
    - Uses cp.cuda.Device(gpu_id) for direct GPU selection [S149-B]
    - Workers see all GPUs — no HIP/CUDA/ROCR visibility masking in spawner
    - ROCR_VISIBLE_DEVICES not viable on this CuPy/ROCm stack
    """
    job_id = job.get('job_id', 'unknown')

    # Extract parameters
    dataset_path = job.get('dataset_path') or job.get('target_file')
    window_size  = job.get('window_size', 10)
    seed_start   = job.get('seed_start', 0)
    seed_end     = job.get('seed_end', seed_start + 100_000)
    skip_range   = tuple(job.get('skip_range', [0, 16]))
    threshold    = coerce_threshold(job.get('min_match_threshold', None), 0.25)
    offset       = job.get('offset', 0)
    sessions     = job.get('sessions', ['midday', 'evening'])
    prng_families= job.get('prng_families', ['java_lcg'])

    draws = load_draws_cached(dataset_path, window_size, sessions, offset)
    k = len(draws)

    # [S155-ROCR] ROCR_VISIBLE_DEVICES={gpu_id} in spawner remaps assigned GPU
    # to device index 0 in this worker's HIP context. Use Device(0).
    # gpu_id is retained for logging and mismatch detection only.
    _job_gpu_id = job.get('gpu_id', None)
    if _job_gpu_id is not None and int(_job_gpu_id) != gpu_id:
        raise ValueError(f"gpu_id mismatch: worker={gpu_id}, job={_job_gpu_id}")
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
            # Apply per-family custom_params overrides (fidelity — mirrors original path)
            default_params = dict(config.get("default_params", {}))
            if custom_params:
                default_params.update(custom_params)

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
            # Note: default_params already set above with custom_params applied
            kernel_args = [
                seeds_gpu, residues_gpu, survivors_gpu,
                match_rates_gpu, best_skips_gpu, survivor_count_gpu,
                cp.int32(n_seeds), cp.int32(k),
                cp.int32(skip_min), cp.int32(skip_max),
                cp.float32(threshold)
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
                # Kernel signatures differ between forward and reverse:
                #   Forward (java_lcg_hybrid_multi_strategy_sieve):
                #     ...threshold, unsigned long long a, unsigned long long c
                #   Reverse (java_lcg_hybrid_reverse_sieve):
                #     ...threshold, int offset  (a,c hardcoded inside kernel)
                strategies_data = job.get('strategies') or []
                if not strategies_data:
                    try:
                        from hybrid_strategy import get_all_strategies
                        strategies_data = [
                            # Full dict needed — sieve_filter.py StrategyConfig(**s) requires all fields
                            s.to_dict() if hasattr(s, 'to_dict') else s
                            for s in get_all_strategies()
                        ]
                    except ImportError:
                        strategies_data = [{"max_consecutive_misses": 3, "skip_tolerance": 5}]
                n_strategies        = len(strategies_data)
                strategy_max_misses = cp.array([s["max_consecutive_misses"] for s in strategies_data], dtype=cp.int32)
                strategy_tolerances = cp.array([s["skip_tolerance"]         for s in strategies_data], dtype=cp.int32)
                strategy_ids_gpu    = cp.zeros(n_seeds,     dtype=cp.uint32)
                skip_sequences_gpu  = cp.zeros(n_seeds * k, dtype=cp.uint32)
                # Use phase2_threshold for hybrid (mirrors sieve_filter.py single-phase hybrid)
                # Use coerce_threshold for safe handling — avoids 0.0 / string edge cases
                phase2_raw = job.get('phase2_threshold', None)
                hybrid_threshold = coerce_threshold(phase2_raw, threshold) if phase2_raw is not None else threshold
                if family_name == 'java_lcg_hybrid':
                    # Forward hybrid: kernel expects ...threshold, a, c
                    kernel_args = [
                        seeds_gpu, residues_gpu, survivors_gpu,
                        match_rates_gpu, skip_sequences_gpu, strategy_ids_gpu,
                        survivor_count_gpu, cp.int32(n_seeds), cp.int32(k),
                        strategy_max_misses, strategy_tolerances, cp.int32(n_strategies),
                        cp.float32(hybrid_threshold),
                        cp.uint64(default_params.get("a", 25214903917)),
                        cp.uint64(default_params.get("c", 11)),
                    ]
                else:
                    # Reverse hybrid: kernel expects ...threshold, offset (a,c hardcoded)
                    kernel_args = [
                        seeds_gpu, residues_gpu, survivors_gpu,
                        match_rates_gpu, skip_sequences_gpu, strategy_ids_gpu,
                        survivor_count_gpu, cp.int32(n_seeds), cp.int32(k),
                        strategy_max_misses, strategy_tolerances, cp.int32(n_strategies),
                        cp.float32(hybrid_threshold),
                        cp.int32(offset),
                    ]
                kernel((blocks,), (threads,), tuple(kernel_args))
                count = min(int(survivor_count_gpu[0].get()), n_seeds)  # defensive clamp
                if count > 0:
                    s_arr   = survivors_gpu[:count].get().tolist()
                    r_arr   = match_rates_gpu[:count].get().tolist()
                    sid_arr = strategy_ids_gpu[:count].get().tolist()
                    ss_raw  = skip_sequences_gpu[:count * k].get().reshape(count, k).tolist()
                    for seed, rate, sid, ss in zip(s_arr, r_arr, sid_arr, ss_raw):
                        if rate >= hybrid_threshold:  # use hybrid_threshold not threshold
                            # [S150-slim_v1] tuple: (seed, match_rate, strategy_id, skip_seq)
                            survivors_out.append((int(seed), float(rate), int(sid), list(ss)))
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

            count = min(int(survivor_count_gpu[0].get()), n_seeds)  # defensive clamp
            if count > 0:
                s_arr = survivors_gpu[:count].get().tolist()
                r_arr = match_rates_gpu[:count].get().tolist()
                k_arr = best_skips_gpu[:count].get().tolist()
                for seed, rate, skip in zip(s_arr, r_arr, k_arr):
                    if rate >= threshold:
                        # [S150-slim_v1] tuple: (seed, match_rate, None=no_strategy, [best_skip])
                        survivors_out.append((int(seed), float(rate), None, [int(skip)]))

            duration_ms = (time.time() - t0) * 1000
            per_family.append({
                'family': family_name,
                'tested': n_seeds,
                'found': len(survivors_out),
                'duration_ms': round(duration_ms, 2),
                'seeds_per_sec': int(n_seeds / (duration_ms / 1000)) if duration_ms > 0 else 0
            })
            all_survivors.extend(survivors_out)

    # [S154] Explicit GPU array deletion — prevents VM bloat OOM.
    # Root cause: CUPY_CUDA_MEMORY_POOL_TYPE=none + non-deterministic GC
    # causes unfreed CuPy arrays to accumulate → 41GB VM → OOM killer.
    try: del seeds_gpu
    except NameError: pass
    try: del survivors_gpu
    except NameError: pass
    try: del match_rates_gpu
    except NameError: pass
    try: del best_skips_gpu
    except NameError: pass
    try: del survivor_count_gpu
    except NameError: pass
    try: del residues_gpu
    except NameError: pass
    try: del strategy_ids_gpu
    except NameError: pass
    try: del skip_sequences_gpu
    except NameError: pass
    import gc; gc.collect()
    _best_effort_gpu_cleanup()

    total_tested   = sum(f['tested'] for f in per_family)
    total_duration = sum(f['duration_ms'] for f in per_family)
    # [S150-slim_v1] Build flat parallel-array result dict
    # IMPORTANT: run_worker() wraps this as {"status":"ok","result":<this>}
    # Coordinator reads inner=result.get("result",{}) then inner.get("format")
    # So "format" and array fields MUST be at top level here (not nested under "result")
    _seeds     = [t[0] for t in all_survivors]
    _rates     = [t[1] for t in all_survivors]
    # TB ruling: drive hybrid from job context, not survivor content
    # Zero-survivor hybrid chunks must still emit strategy_ids/skip_sequences
    # [S155] Fix: job payload sends "hybrid": bool — not "prng_type" or "skip_mode".
    # Both those keys are absent from PWC job dicts. job.get("hybrid", False) is the
    # correct key — set at sieve_range() dispatch from "is_hybrid = '_hybrid' in prng_type".
    _is_hybrid = bool(job.get("hybrid", False))
    _ret = {
        'job_id':      job_id,
        'success':     True,
        'format':      'slim_v1',
        'seeds':       _seeds,
        'match_rates': _rates,
        'seed_range':  {'start': seed_start, 'end': seed_end},
        'stats': {
            'total_seeds_tested': total_tested,
            'total_survivors':    len(_seeds),
            'duration_ms':        round(total_duration, 2),
            'avg_seeds_per_sec':  int(total_tested / (total_duration / 1000)) if total_duration > 0 else 0
        },
        'per_family': {f['family']: f for f in per_family}
    }
    if _is_hybrid:
        _ret['strategy_ids']   = [t[2] for t in all_survivors]
        _ret['skip_sequences'] = [t[3] for t in all_survivors]
    return _ret


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

    # [S155-v2] Cap CuPy memory pool — prevents OOM on AMD rigs.
    # Root cause: CUPY_CUDA_MEMORY_POOL_TYPE=none mmap'd full 8GB VRAM per worker
    # at device init. 8 workers × 8GB = ~64GB VA on a 7.7GB RAM machine → OOM.
    # Fix: re-enable pool (remove that env var in coordinator) + cap via set_limit().
    #
    # Sequencing requirement (TB review S155):
    #   set_limit() requires a live device context — must be called inside
    #   `with cp.cuda.Device()`. It must also precede the warmup cp.zeros(1)
    #   allocation so the very first pool allocation is already bounded.
    #
    # Pinned pool (TB review S155):
    #   PinnedMemoryPool has no set_limit() in CuPy 13.x/14.x (confirmed live).
    #   Methods: free, free_all_blocks, malloc, n_free_blocks — no limit API.
    #   Do NOT call set_limit() on the pinned pool.
    #
    # Configurability: PRNG_CUPY_POOL_LIMIT_MB injected via ROCM_ENV_VARS in
    # persistent_worker_coordinator.py — propagates through remote `env` command.
    # Default 256MB = 5-6x per-job working set (~42-50MB for 2M seeds).
    # 8 workers × 256MB = 2GB total pool — leaves ~5.7GB free on 7.7GB rigs.
    import os as _os
    _pool_mb = int(_os.environ.get("PRNG_CUPY_POOL_LIMIT_MB", "256"))
    _pool_bytes = _pool_mb * 1024 * 1024

    # Warm up GPU - touch device to trigger ROCm init NOW (not at first job)
    # set_limit() is inside the Device context and before cp.zeros(1) so the
    # warmup allocation itself is already bounded by the cap.
    # [S155-ROCR] ROCR_VISIBLE_DEVICES remaps assigned GPU to device 0.
    _log(f"Warming up GPU {gpu_id} (physical, appears as device 0 via ROCR)...")
    with cp.cuda.Device(0):
        cp.get_default_memory_pool().set_limit(_pool_bytes)   # cap BEFORE first alloc
        _ = cp.zeros(1, dtype=cp.float32)                     # warmup, now bounded
        cp.cuda.Device(0).synchronize()

    # Startup diagnostics — proves the cap is live and VM is sane (TB S155)
    # VmSize logged because the crash signature was total-vm:41452828kB (virtual
    # address space bloat from mmap). VmRSS alone would not have caught that.
    _mp = cp.get_default_memory_pool()
    _vm_size_kb = "unknown"
    _vm_rss_kb = "unknown"
    try:
        for _line in open("/proc/self/status").readlines():
            if _line.startswith("VmSize:"):
                _vm_size_kb = _line.split()[1]
            elif _line.startswith("VmRSS:"):
                _vm_rss_kb = _line.split()[1]
    except Exception:
        pass
    _log(
        f"GPU ready | pool_limit={_mp.get_limit() // (1024*1024)}MB "
        f"used={_mp.used_bytes() // 1024}KB "
        f"total={_mp.total_bytes() // 1024}KB "
        f"VmSize={_vm_size_kb}kB VmRSS={_vm_rss_kb}kB"
    )

    device_name = "unknown"
    try:
        device_name = cp.cuda.runtime.getDeviceProperties(gpu_id)['name'].decode()
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
                result = run_sieve_job(job, gpu_id)
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
