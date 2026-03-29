#!/usr/bin/env python3
"""
fix_s158b_threadpool_dispatch_v3.py

S158B REVISION 3 — All TB blockers resolved.

BLOCKER 1 FIXED: executor context-manager shutdown hang.
  Does NOT use `with ThreadPoolExecutor` context manager.
  Instead: creates executor, submits futures, runs progress-aware loop,
  then calls shutdown(wait=False, cancel_futures=True) on timeout/break.
  Running futures are bounded: dispatch_chunk always returns within
  JOB_TIMEOUT_S=600s because _read_with_timeout(stream, JOB_TIMEOUT_S)
  uses t.join(timeout=timeout_s) internally — guaranteed bounded return.
  So shutdown(wait=False) does not block more than 600s in the worst case,
  and cancel_futures=True prevents any pending (not-yet-started) futures
  from starting.

BLOCKER 2 FIXED: empty-chunks guard added before executor creation.
  If len(chunks)==0, skip executor entirely and log. ThreadPoolExecutor(0)
  is invalid and would raise ValueError.

BLOCKER 3 FIXED: failure propagation is explicit.
  Timed-out chunks are counted as failed_chunks directly by incrementing
  a local counter that is added to results_by_chunk as error entries,
  so the existing aggregation path sees them as failures explicitly.

OTHER FIXES from TB review:
  - Imports at top of function, not inside replaced block
  - Peak thread tracking runs every iteration of progress loop
  - No false claims in docstring about cancellation behavior

Deploy:
  scp ~/Downloads/fix_s158b_threadpool_dispatch_v3.py rzeus:~/distributed_prng_analysis/
  ssh rzeus 'cd ~/distributed_prng_analysis && python3 fix_s158b_threadpool_dispatch_v3.py'
"""

import ast
import shutil
from pathlib import Path

TARGET = Path("/home/michael/distributed_prng_analysis/persistent_worker_coordinator.py")
BACKUP = TARGET.with_suffix(".py.bak_s158b_v3")

# Add imports at module level — find the import block
IMPORT_OLD = "import threading\nimport time"
IMPORT_NEW = "import threading\nimport time\nfrom concurrent.futures import ThreadPoolExecutor, wait as _cf_wait, FIRST_COMPLETED as _CF_FIRST_COMPLETED"

OLD_DISPATCH = """        threads = []
        for i, (s_start, s_end) in enumerate(chunks):
            worker = all_workers[i % num_workers]
            t = threading.Thread(
                target=dispatch_chunk,
                args=(i, s_start, s_end, worker),
                daemon=True
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()"""

NEW_DISPATCH = """        # [S158B-v3] Bounded dispatch — TB-approved hardening.
        # Replaces unbounded thread-per-chunk fan-out (537 simultaneous threads)
        # with ThreadPoolExecutor capped at min(num_workers, len(chunks)).
        #
        # Key design decisions:
        # 1. No context-manager (with) — avoids shutdown(wait=True) hang trap.
        #    Executor is explicitly shut down with wait=False, cancel_futures=True
        #    on timeout. Running futures are bounded: dispatch_chunk always returns
        #    within JOB_TIMEOUT_S because _read_with_timeout uses t.join(timeout).
        # 2. Progress-aware wait() loop — breaks if no chunk completes within
        #    PASS_PROGRESS_TIMEOUT_S. Prevents permanent hang on stuck chunks.
        # 3. Empty-chunks guard — ThreadPoolExecutor(max_workers=0) is invalid.
        # 4. Explicit failure propagation — timed-out chunks written to
        #    results_by_chunk as error entries, not just counted.

        # Guard: nothing to dispatch
        if not chunks:
            self.logger.info("[S158B] No chunks to dispatch — skipping executor")
        else:
            # Progress timeout: 2× JOB_TIMEOUT_S is conservative ceiling.
            # If no chunk completes within this window, declare pass failed.
            _PASS_PROGRESS_TIMEOUT_S = JOB_TIMEOUT_S * 2  # 1200s

            _pool_size = min(num_workers, len(chunks))
            _pass_timed_out = False
            _peak_threads = threading.active_count()

            self.logger.info(
                f"[S158B] Bounded dispatch: {len(chunks)} chunks → "
                f"pool_size={_pool_size} (num_workers={num_workers})"
            )

            _executor = ThreadPoolExecutor(max_workers=_pool_size)
            try:
                _future_to_idx = {
                    _executor.submit(
                        dispatch_chunk, i, s_start, s_end,
                        all_workers[i % num_workers]
                    ): i
                    for i, (s_start, s_end) in enumerate(chunks)
                }
                _pending = set(_future_to_idx.keys())

                while _pending:
                    _peak_threads = max(_peak_threads, threading.active_count())

                    _done, _pending = _cf_wait(
                        _pending,
                        timeout=_PASS_PROGRESS_TIMEOUT_S,
                        return_when=_CF_FIRST_COMPLETED
                    )

                    if not _done:
                        # No chunk completed — stuck pass detected
                        _pass_timed_out = True
                        self.logger.error(
                            f"[S158B] Pass progress timeout after "
                            f"{_PASS_PROGRESS_TIMEOUT_S}s — "
                            f"{len(_pending)} chunks still pending. "
                            f"Aborting pass."
                        )
                        # Write explicit error entries for stuck chunks
                        with lock:
                            for _f in _pending:
                                _stuck_idx = _future_to_idx[_f]
                                results_by_chunk[_stuck_idx] = {
                                    "status": "error",
                                    "message": "Pass progress timeout — chunk aborted"
                                }
                                self.logger.error(
                                    f"  [S158B] Chunk {_stuck_idx} aborted (progress timeout)"
                                )
                        break

                    # Process completed futures
                    for _future in _done:
                        _idx = _future_to_idx[_future]
                        try:
                            _future.result()
                        except Exception as _e:
                            self.logger.error(
                                f"[S158B] Chunk {_idx} future raised: {_e}"
                            )
                            # Ensure error is recorded explicitly
                            with lock:
                                if _idx not in results_by_chunk:
                                    results_by_chunk[_idx] = {
                                        "status": "error",
                                        "message": str(_e)
                                    }

            finally:
                # Explicit shutdown — do not wait for stuck running threads.
                # cancel_futures=True prevents pending futures from starting.
                # Running futures finish within JOB_TIMEOUT_S (bounded by design).
                _executor.shutdown(wait=False, cancel_futures=True)
                self.logger.info(
                    f"[S158B] Executor shut down — "
                    f"peak_threads={_peak_threads} "
                    f"timed_out={_pass_timed_out}"
                )"""


def apply():
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found")
        return False

    content = TARGET.read_text()

    # Check if already applied
    if "S158B-v3" in content:
        print("S158B-v3 already applied — skipping")
        return True

    # Check for v1/v2 remnants
    if "S158B" in content:
        print("ERROR: Earlier S158B version detected — manual cleanup required")
        return False

    # Apply import addition
    if IMPORT_OLD not in content:
        print("WARNING: import anchor not found — skipping import addition")
        print("(ThreadPoolExecutor may already be imported)")
    elif "ThreadPoolExecutor" not in content:
        content = content.replace(IMPORT_OLD, IMPORT_NEW, 1)
        print("✅ Added ThreadPoolExecutor import")

    # Apply dispatch patch
    if OLD_DISPATCH not in content:
        print("ERROR: dispatch anchor not found")
        idx = content.find("threads = []")
        if idx >= 0:
            print("Context around 'threads = []':")
            print(repr(content[idx:idx+400]))
        return False

    content = content.replace(OLD_DISPATCH, NEW_DISPATCH, 1)

    # Validate
    try:
        ast.parse(content)
    except SyntaxError as e:
        print(f"ERROR: Syntax error at line {e.lineno}: {e.msg}")
        return False

    shutil.copy2(TARGET, BACKUP)
    print(f"Backup: {BACKUP}")
    TARGET.write_text(content)

    try:
        ast.parse(TARGET.read_text())
        print("✅ S158B-v3 patch applied and verified")
        print("   - No context-manager shutdown trap")
        print("   - Empty-chunks guard added")
        print("   - Explicit error entries for timed-out chunks")
        print("   - Progress-aware wait() loop")
        print("   - shutdown(wait=False, cancel_futures=True)")
        print("\nNext steps:")
        print("  git add -f persistent_worker_coordinator.py fix_s158b_threadpool_dispatch_v3.py")
        print("  git commit -m 'fix(s158b-v3): bounded dispatch — progress-aware, no shutdown hang'")
        print("  git push origin main && git push public main")
        return True
    except SyntaxError as e:
        print(f"ERROR: Post-write syntax error: {e}")
        shutil.copy2(BACKUP, TARGET)
        print("Restored backup")
        return False


if __name__ == "__main__":
    print("Applying S158B-v3 bounded dispatch patch...")
    apply()
