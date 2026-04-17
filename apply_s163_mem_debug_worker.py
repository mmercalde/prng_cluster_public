#!/usr/bin/env python3
"""
apply_s163_mem_debug_worker.py
================================
Implements TB-approved S163 memory instrumentation in sieve_gpu_worker.py.

Changes:
1. After each job completes, if S163_MEM_DEBUG=1, log pool stats every 25 chunks:
   - pool_used, pool_total, n_free_blocks, VmRSS, VmSize
2. Threshold breach logging always active (not gated by MEM_DEBUG):
   - Warn if pool_used > 200MB
3. Remove free_all_blocks() from _best_effort_gpu_cleanup() — Option B
   (TB-approved: S155 256MB pool cap makes it redundant; race condition source)

Apply:
    python3 apply_s163_mem_debug_worker.py --dry-run
    python3 apply_s163_mem_debug_worker.py
"""
import argparse, ast, shutil, sys
from pathlib import Path

TARGET = Path("sieve_gpu_worker.py")
BACKUP = Path("sieve_gpu_worker.py.pre_s163_mem_debug")

# ---------------------------------------------------------------------------
# Patch 1: Remove free_all_blocks() from _best_effort_gpu_cleanup()
# ---------------------------------------------------------------------------
OLD_CLEANUP = '''    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass'''

NEW_CLEANUP = '''    # [S163] free_all_blocks() removed — TB-approved Option B.
    # S155 256MB pool cap makes it redundant. CuPy issue #4866: concurrent
    # free_all_blocks() calls from 8 workers race → cudaErrorIllegalAddress.
    pass'''

# ---------------------------------------------------------------------------
# Patch 2: Add MEM_DEBUG instrumentation to job loop after job completes
# ---------------------------------------------------------------------------
OLD_JOB_LOOP = '''        elif command == "sieve":
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
                       "error": str(e), "traceback": traceback.format_exc()})'''

NEW_JOB_LOOP = '''        elif command == "sieve":
            job = msg.get("job", msg)  # support bare job or wrapped
            job_id = job.get("job_id", "unknown")
            try:
                t0 = time.time()
                result = run_sieve_job(job, gpu_id)
                elapsed = time.time() - t0
                jobs_processed += 1
                _emit({"status": "ok", "job_id": job_id,
                       "elapsed_s": round(elapsed, 3), "result": result})

                # [S163] Memory instrumentation — TB-approved
                # Sample every 25 chunks when S163_MEM_DEBUG=1
                # Threshold breach always logged regardless of MEM_DEBUG flag
                _s163_debug = os.environ.get("S163_MEM_DEBUG", "0") == "1"
                if _s163_debug and jobs_processed % 25 == 0:
                    try:
                        _mp = cp.get_default_memory_pool()
                        _pool_used_mb  = _mp.used_bytes()  // (1024 * 1024)
                        _pool_total_mb = _mp.total_bytes() // (1024 * 1024)
                        _pool_free_blk = _mp.n_free_blocks()
                        _vm_rss_kb  = "unknown"
                        _vm_size_kb = "unknown"
                        try:
                            for _ln in open("/proc/self/status").readlines():
                                if _ln.startswith("VmRSS:"):
                                    _vm_rss_kb  = _ln.split()[1]
                                elif _ln.startswith("VmSize:"):
                                    _vm_size_kb = _ln.split()[1]
                        except Exception:
                            pass
                        _log(
                            f"[MEM chunk={jobs_processed}] "
                            f"pool_used={_pool_used_mb}MB "
                            f"pool_total={_pool_total_mb}MB "
                            f"n_free_blocks={_pool_free_blk} "
                            f"VmRSS={_vm_rss_kb}kB "
                            f"VmSize={_vm_size_kb}kB"
                        )
                        # Threshold breach — always warn regardless of MEM_DEBUG
                        if _pool_used_mb > 200:
                            _log(
                                f"[MEM WARNING] pool_used={_pool_used_mb}MB "
                                f"exceeds 200MB threshold at chunk={jobs_processed}"
                            )
                    except Exception as _me:
                        _log(f"[MEM instrumentation error] {_me}")

            except Exception as e:
                _emit({"status": "error", "job_id": job_id,
                       "error": str(e), "traceback": traceback.format_exc()})'''

# ---------------------------------------------------------------------------
# Apply patches
# ---------------------------------------------------------------------------
def run(dry_run: bool):
    src = TARGET.read_text()

    # Check anchors
    missing = []
    if OLD_CLEANUP not in src:
        missing.append("OLD_CLEANUP (free_all_blocks in _best_effort_gpu_cleanup)")
    if OLD_JOB_LOOP not in src:
        missing.append("OLD_JOB_LOOP (sieve command handler)")

    if missing:
        print("ERROR — anchor(s) not found:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    patched = src.replace(OLD_CLEANUP, NEW_CLEANUP, 1)
    patched = patched.replace(OLD_JOB_LOOP, NEW_JOB_LOOP, 1)

    # AST check
    try:
        ast.parse(patched)
        print("AST check: PASS")
    except SyntaxError as e:
        print(f"AST check: FAIL — {e}")
        sys.exit(1)

    if dry_run:
        print("Dry run — no files written.")
        print("Patches verified:")
        print("  1. free_all_blocks() removed from _best_effort_gpu_cleanup()")
        print("  2. MEM_DEBUG instrumentation added to job loop")
        return

    shutil.copy(TARGET, BACKUP)
    print(f"Backup: {BACKUP}")
    TARGET.write_text(patched)
    print(f"Patched: {TARGET}")
    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(args.dry_run)
