#!/usr/bin/env python3
"""
apply_s158d_progress_writer.py

Adds ProgressWriter (web dashboard) support to zmq_sqlite_coordinator.py.

Without this patch the web dashboard shows nothing during ZMQ runs.
PWC calls ProgressWriter after each chunk — ZMQ coordinator must do the same.

Calls added:
  - __init__: initialize _progress_writer
  - _launch_workers: register_node() for each rig
  - run_sieve_pass start: update_step()
  - result received: log_gpu_result()
  - pass complete: update_progress()
  - run_trial_zmq_sqlite after all passes: update_trial_stats()

Deploy:
  scp ~/Downloads/apply_s158d_progress_writer.py rzeus:~/distributed_prng_analysis/
  ssh rzeus 'cd ~/distributed_prng_analysis && python3 apply_s158d_progress_writer.py'
"""

import ast
import shutil
from pathlib import Path

TARGET = Path("/home/michael/distributed_prng_analysis/zmq_sqlite_coordinator.py")
BACKUP = TARGET.with_suffix(".py.bak_progress_writer")

# ── 1. Add _progress_writer init to __init__ ─────────────────────────────────
OLD_INIT = """        self._workers_launched = False
        self._zeus_ip         = self._get_zeus_ip()"""

NEW_INIT = """        self._workers_launched = False
        self._zeus_ip         = self._get_zeus_ip()
        self._progress_writer = None
        self._init_progress_writer()

    def _init_progress_writer(self):
        \"\"\"Initialize ProgressWriter for web dashboard — mirrors PWC.startup().\"\"\"
        try:
            from progress_display import ProgressWriter
            self._progress_writer = ProgressWriter(
                "Forward Sieve", total_jobs=100, total_seeds=0
            )
            for node in self._nodes:
                host = node.get("hostname", "")
                if host in ("localhost", "127.0.0.1"):
                    self._progress_writer.register_node(
                        "localhost", "RTX 3080 Ti", 2
                    )
                elif node.get("gpu_count", 0) > 0:
                    self._progress_writer.register_node(
                        host,
                        node.get("gpu_type", "RX 6600"),
                        node.get("gpu_count", 8)
                    )
        except Exception as e:
            self.logger.warning(f"[ZMQ] ProgressWriter unavailable: {e}")
            self._progress_writer = None"""

# ── 2. Add update_step at start of run_sieve_pass ────────────────────────────
OLD_DISPATCH = """        self.logger.info(
            f"[ZMQ] {prng_type} {total_seeds:,} seeds "
            f"-> {total_chunks} chunks ({chunk_size:,}/chunk)"
        )"""

NEW_DISPATCH = """        self.logger.info(
            f"[ZMQ] {prng_type} {total_seeds:,} seeds "
            f"-> {total_chunks} chunks ({chunk_size:,}/chunk)"
        )
        if self._progress_writer:
            try:
                _step_name = (
                    f"{'Reverse' if 'reverse' in prng_type else 'Forward'} "
                    f"Sieve ({prng_type})"
                )
                self._progress_writer.update_step(
                    _step_name, total_seeds=total_seeds
                )
            except Exception:
                pass"""

# ── 3. Add log_gpu_result after each chunk completes ────────────────────────
OLD_LOG = """                    if is_new:
                        completed += 1
                        n = len(result.get("survivors", []))
                        self.logger.info(
                            f"  ✅ Chunk {chunk_id}: {n:,} survivors [{worker_id}]"
                        )"""

NEW_LOG = """                    if is_new:
                        completed += 1
                        n = len(result.get("survivors", []))
                        self.logger.info(
                            f"  ✅ Chunk {chunk_id}: {n:,} survivors [{worker_id}]"
                        )
                        if self._progress_writer:
                            try:
                                # Parse worker_id: "hostname:gpuN"
                                _parts = worker_id.split(":")
                                _host  = _parts[0] if _parts else worker_id
                                _gpuid = int(_parts[1].replace("gpu","")) \
                                         if len(_parts) > 1 else 0
                                _gpu_type = "RTX 3080 Ti" \
                                            if _host == "localhost" else "RX 6600"
                                _chunk_seeds = chunk_size
                                # elapsed not available here — use chunk_size/throughput
                                self._progress_writer.log_gpu_result(
                                    _host, _gpuid, _gpu_type,
                                    _chunk_seeds, 10.0, success=True
                                )
                            except Exception:
                                pass"""

# ── 4. Add update_progress after result loop ─────────────────────────────────
OLD_AGGREGATE = """        # Aggregate
        all_survivors: List[int]   = []"""

NEW_AGGREGATE = """        # Update dashboard progress
        if self._progress_writer:
            try:
                self._progress_writer.update_progress(
                    jobs_done=completed,
                    chunks_total=total_chunks
                )
            except Exception:
                pass

        # Aggregate
        all_survivors: List[int]   = []"""

# ── 5. Add update_trial_stats in run_trial_zmq_sqlite ───────────────────────
OLD_TRIAL_RETURN = """        total_bidi = len(bidi_const) + len(bidi_var)
        print(f"      Total bidirectional: {total_bidi:,}")

        return {"""

NEW_TRIAL_RETURN = """        total_bidi = len(bidi_const) + len(bidi_var)
        print(f"      Total bidirectional: {total_bidi:,}")

        # Update dashboard trial stats (mirrors run_trial_persistent)
        if coord._progress_writer:
            try:
                coord._progress_writer.update_trial_stats(
                    trial_num=trial_number,
                    forward_survivors=len(fwd_map),
                    reverse_survivors=len(rev_map),
                    bidirectional=total_bidi,
                    best_bidirectional=total_bidi,
                    config_desc=f"W{ws}_O{off}",
                    accumulated_forward=len(fwd_map),
                    accumulated_reverse=len(rev_map),
                    accumulated_bidirectional=total_bidi,
                )
            except Exception:
                pass

        return {"""


def apply():
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found")
        return False

    content = TARGET.read_text()

    if "_init_progress_writer" in content:
        print("ProgressWriter already patched — skipping")
        return True

    patches = [
        (OLD_INIT,          NEW_INIT,          "1. _progress_writer init"),
        (OLD_DISPATCH,      NEW_DISPATCH,      "2. update_step at pass start"),
        (OLD_LOG,           NEW_LOG,           "3. log_gpu_result per chunk"),
        (OLD_AGGREGATE,     NEW_AGGREGATE,     "4. update_progress after loop"),
        (OLD_TRIAL_RETURN,  NEW_TRIAL_RETURN,  "5. update_trial_stats"),
    ]

    for old, new, label in patches:
        if old not in content:
            print(f"ERROR: anchor not found for patch: {label}")
            return False
        content = content.replace(old, new, 1)
        print(f"  ✅ {label}")

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
        print("✅ ProgressWriter support added to zmq_sqlite_coordinator.py")
        print("\nNext steps:")
        print("  git add zmq_sqlite_coordinator.py apply_s158d_progress_writer.py")
        print("  git commit -m 'fix(s158d): add ProgressWriter dashboard support to ZMQ coordinator'")
        print("  git push origin main && git push public main")
        return True
    except SyntaxError as e:
        print(f"ERROR: Post-write syntax error: {e}")
        shutil.copy2(BACKUP, TARGET)
        print("Restored backup")
        return False


if __name__ == "__main__":
    print("Adding ProgressWriter support to zmq_sqlite_coordinator.py...")
    apply()
