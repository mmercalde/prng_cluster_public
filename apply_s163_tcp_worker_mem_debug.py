#!/usr/bin/env python3
"""
apply_s163_tcp_worker_mem_debug.py

Patches persistent/pwc_worker_service.py to add S163 MEM_DEBUG instrumentation
to the TCP-PWC worker path. This mirrors the instrumentation in sieve_gpu_worker.py
(lines 488-523) which only covers the SSH-PWC path.

Root cause of missing MEM logs:
  The S163 instrumentation at commit 3c20d2d only added sampling to
  sieve_gpu_worker.py. But with `--pwc-transport tcp` (the current default),
  workers launch persistent/pwc_worker_service.py which has NO memory sampling
  code — so S163_MEM_DEBUG=1 is exported but ignored.

Fix:
  Add the same sampling pattern (every 25 chunks, gated by S163_MEM_DEBUG=1,
  threshold breach always logged) to pwc_worker_service.py after the cleanup
  call that runs after each job.

Dry-run supported:
    python3 apply_s163_tcp_worker_mem_debug.py --dry-run
"""
import argparse
import shutil
import ast
import sys
from pathlib import Path

TARGET = "persistent/pwc_worker_service.py"

# Anchor: the GPU cleanup block after each job completes
OLD = """            try:
                from sieve_filter import _best_effort_gpu_cleanup
                _best_effort_gpu_cleanup()
            except Exception as _cleanup_exc:
                log.debug(f\"[{self.worker_id}] GPU cleanup skipped: {_cleanup_exc}\")"""

# New: cleanup + MEM_DEBUG instrumentation matching sieve_gpu_worker.py pattern
NEW = """            try:
                from sieve_filter import _best_effort_gpu_cleanup
                _best_effort_gpu_cleanup()
            except Exception as _cleanup_exc:
                log.debug(f\"[{self.worker_id}] GPU cleanup skipped: {_cleanup_exc}\")

            # [S163] Memory instrumentation — TB-approved
            # Mirrors sieve_gpu_worker.py lines 488-523 for the TCP-PWC path.
            # Sample every 25 jobs when S163_MEM_DEBUG=1.
            # Threshold breach (>200MB) always logged regardless of MEM_DEBUG flag.
            _s163_debug = os.environ.get(\"S163_MEM_DEBUG\", \"0\") == \"1\"
            _total_jobs = self.jobs_done + self.jobs_error
            if _total_jobs % 25 == 0 and _total_jobs > 0:
                try:
                    import cupy as _cp_s163
                    _mp = _cp_s163.get_default_memory_pool()
                    _pool_used_mb  = _mp.used_bytes()  // (1024 * 1024)
                    _pool_total_mb = _mp.total_bytes() // (1024 * 1024)
                    _pool_free_blk = _mp.n_free_blocks()
                    _vm_rss_kb  = \"unknown\"
                    _vm_size_kb = \"unknown\"
                    try:
                        for _ln in open(\"/proc/self/status\").readlines():
                            if _ln.startswith(\"VmRSS:\"):
                                _vm_rss_kb  = _ln.split()[1]
                            elif _ln.startswith(\"VmSize:\"):
                                _vm_size_kb = _ln.split()[1]
                    except Exception:
                        pass
                    if _s163_debug:
                        log.info(
                            f\"[MEM chunk={_total_jobs}] \"
                            f\"worker={self.worker_id} \"
                            f\"pool_used={_pool_used_mb}MB \"
                            f\"pool_total={_pool_total_mb}MB \"
                            f\"n_free_blocks={_pool_free_blk} \"
                            f\"VmRSS={_vm_rss_kb}kB \"
                            f\"VmSize={_vm_size_kb}kB\"
                        )
                    # Threshold breach — always warn regardless of MEM_DEBUG
                    if _pool_used_mb > 200:
                        log.warning(
                            f\"[MEM WARNING] worker={self.worker_id} \"
                            f\"pool_used={_pool_used_mb}MB \"
                            f\"exceeds 200MB threshold at chunk={_total_jobs}\"
                        )
                except Exception as _me:
                    log.debug(f\"[MEM instrumentation error] {_me}\")"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Verify match without writing")
    args = ap.parse_args()

    path = Path(TARGET)
    if not path.exists():
        sys.exit(f"❌ {TARGET} not found. Run from ~/distributed_prng_analysis/")

    src = path.read_text()

    count = src.count(OLD)
    if count == 0:
        sys.exit(
            "❌ OLD block not found. File may already be patched or has diverged.\n"
            "   Check persistent/pwc_worker_service.py around line 303-307"
        )
    if count > 1:
        sys.exit(f"❌ OLD block matches {count} times — ambiguous, aborting.")

    print(f"✅ Anchor found (1 match) at byte offset {src.index(OLD)}")

    new_src = src.replace(OLD, NEW)
    try:
        ast.parse(new_src)
        print("✅ AST check passed for proposed new contents")
    except SyntaxError as e:
        sys.exit(f"❌ AST check FAILED: {e}")

    if args.dry_run:
        print("DRY RUN — no changes written")
        print(f"Would replace {len(OLD)} bytes with {len(NEW)} bytes")
        return

    backup = path.with_suffix(path.suffix + ".pre_s163_mem_debug")
    shutil.copy(path, backup)
    print(f"✅ Backup written: {backup}")

    path.write_text(new_src)
    print(f"✅ Patched {TARGET}")
    print(f"   Added MEM_DEBUG sampling (every 25 jobs) + threshold breach logging")


if __name__ == "__main__":
    main()
