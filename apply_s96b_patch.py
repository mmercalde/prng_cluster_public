#!/usr/bin/env python3
"""
apply_s96b_patch.py - S96B Persistent GPU Worker Patch
=======================================================

Surgically patches meta_prediction_optimizer_anti_overfit.py to add:
  1. S96B worker management methods (_spawn, _shutdown, _read_line, _dispatch)
  2. study.optimize() wrapped with worker spawn/shutdown in try/finally
  3. _run_nn_optuna_trial routes through worker when _s96b_workers is populated
  4. CLI flags: --persistent-workers / --no-persistent-workers (default OFF)
  5. will_use_subprocess detection extended for persistent-workers mode

All patches tagged [S96B] for auditability.
Backup created before any writes. Syntax verified after patch.

Usage:
  python3 apply_s96b_patch.py             # dry-run (shows what will change)
  python3 apply_s96b_patch.py --apply     # apply patch
  python3 apply_s96b_patch.py --verify    # verify existing file has S96B markers

Author: Team Alpha (Claude) - Session S96B 2026-02-18
"""

import argparse
import ast
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

TARGET = Path("meta_prediction_optimizer_anti_overfit.py")

# =============================================================================
# Patch 1: S96B worker management methods
# Inserted just before _sample_hyperparameters (line ~1966 in production)
# Anchor: "    def _sample_hyperparameters(self, trial)"
# =============================================================================

S96B_WORKER_METHODS = '''\
    # =========================================================================
    # [S96B] Persistent GPU Worker management
    # =========================================================================

    def _spawn_persistent_workers(self, gpu_ids: list) -> dict:
        """
        [S96B] Spawn one persistent nn_gpu_worker.py subprocess per GPU.

        Each worker boots torch/CUDA once and processes all Optuna NN trials
        via stdin/stdout JSON IPC, eliminating ~85% of per-trial overhead.

        Returns dict: {gpu_id: {"proc": Popen, "alive": bool, "lock": Lock, "device": str}}
        """
        import subprocess as _sp
        import threading as _th
        import os as _os
        from pathlib import Path as _Path  # [S96B Fix 2] local alias: no bare Path/sys in method
        import sys as _sys                 # [S96B Fix 2] local alias: avoids NameError if globals differ

        workers = {}
        worker_script = _Path(__file__).parent / "nn_gpu_worker.py"
        if not worker_script.exists():
            self.logger.error("[S96B] nn_gpu_worker.py not found - cannot spawn workers")
            return workers

        for gpu_id in gpu_ids:
            env = dict(_os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            try:
                proc = _sp.Popen(
                    [_sys.executable, str(worker_script)],
                    stdin=_sp.PIPE,
                    stdout=_sp.PIPE,
                    stderr=None,    # inherit stderr so worker logs reach console
                    env=env,
                    text=True,
                    bufsize=1,      # line-buffered
                )
                # Wait for ready signal (up to 30s)
                ready = self._s96b_read_worker_line(proc, timeout=30)
                if ready and ready.get("status") == "ready":
                    self.logger.info(
                        f"[S96B] GPU-{gpu_id} worker ready: {ready.get('device', '?')}"
                    )
                    workers[gpu_id] = {
                        "proc":   proc,
                        "alive":  True,
                        "lock":   _th.Lock(),
                        "device": ready.get("device", f"cuda:{gpu_id}"),
                    }
                else:
                    self.logger.warning(
                        f"[S96B] GPU-{gpu_id} worker did not send ready signal - discarding"
                    )
                    try:
                        proc.terminate()
                    except Exception:
                        pass
            except Exception as exc:
                self.logger.error(f"[S96B] Failed to spawn worker GPU-{gpu_id}: {exc}")

        return workers

    def _shutdown_persistent_workers(self, workers: dict) -> None:
        """[S96B] Send shutdown to all alive workers and wait for exit."""
        import json as _json
        import subprocess as _sp
        for gpu_id, w in workers.items():
            if not w.get("alive"):
                continue
            try:
                proc = w["proc"]
                if proc.poll() is None:
                    proc.stdin.write(_json.dumps({"command": "shutdown"}) + "\\n")
                    proc.stdin.flush()
                    proc.stdin.close()   # [S96B TB-rec-B] force EOF so worker exits even if readline blocks
                    proc.wait(timeout=10)
                    self.logger.info(f"[S96B] GPU-{gpu_id} worker shut down cleanly")
            except Exception as exc:
                self.logger.warning(f"[S96B] GPU-{gpu_id} shutdown error ({exc}) - killing")
                try:
                    w["proc"].kill()
                except Exception:
                    pass

    def _s96b_read_worker_line(self, proc, timeout: float = 60):
        """
        [S96B] Read one JSON line from worker stdout within timeout seconds.
        Uses threading.Queue to implement non-blocking readline with timeout.
        Returns parsed dict or None on timeout / broken pipe.
        """
        import json as _json
        import queue as _q
        import threading as _th

        result_q = _q.Queue()

        def _reader():
            try:
                line = proc.stdout.readline()
                result_q.put(line)
            except Exception:
                result_q.put("")

        t = _th.Thread(target=_reader, daemon=True)
        t.start()
        try:
            raw = result_q.get(timeout=timeout)
        except _q.Empty:
            return None

        if not raw:
            return None
        try:
            return _json.loads(raw.strip())
        except Exception:
            self.logger.warning(f"[S96B] Malformed worker stdout: {raw[:200]!r}")
            return None

    def _s96b_dispatch(self, workers: dict, gpu_id: int, job: dict,
                       fallback_fn, timeout: float = 60) -> dict:
        """
        [S96B] Send job to worker for gpu_id, return result dict.

        Fallback strategy (Team Beta spec):
          1. On any failure, log error and attempt single restart
          2. If restart succeeds, retry the job once on fresh worker
          3. If restart fails, mark worker dead and fall back to fallback_fn()
             (= S96A train_single_trial.py subprocess)
          Result: S96B is never worse than S96A.
        """
        import json as _json

        w = workers.get(gpu_id)
        if not w or not w.get("alive"):
            self.logger.warning(
                f"[S96B] GPU-{gpu_id} worker not alive - subprocess fallback"
            )
            return fallback_fn()

        with w["lock"]:
            try:
                proc = w["proc"]
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"Worker GPU-{gpu_id} already exited (rc={proc.poll()})"
                    )
                proc.stdin.write(_json.dumps(job) + "\\n")
                proc.stdin.flush()

                result = self._s96b_read_worker_line(proc, timeout=timeout)
                if result is None:
                    raise RuntimeError(
                        f"Worker GPU-{gpu_id} timeout ({timeout}s) or broken pipe "
                        f"[trial={job.get('trial_number','?')} fold={job.get('fold_idx','?')}]"
                    )
                if result.get("status") == "error":
                    raise RuntimeError(
                        f"Worker GPU-{gpu_id} error: {result.get('error', '?')[:300]}"
                    )
                return result

            except Exception as exc:
                self.logger.error(
                    f"[S96B] Worker GPU-{gpu_id} failed ({exc}) - restart once then fallback"
                )
                w["alive"] = False
                try:
                    w["proc"].kill()
                except Exception:
                    pass

                # Single restart attempt
                restarted = self._spawn_persistent_workers([gpu_id])
                if restarted:
                    workers[gpu_id] = restarted[gpu_id]
                    self.logger.info(f"[S96B] GPU-{gpu_id} worker restarted successfully")
                    try:
                        w2 = workers[gpu_id]
                        with w2["lock"]:
                            proc2 = w2["proc"]
                            proc2.stdin.write(_json.dumps(job) + "\\n")
                            proc2.stdin.flush()
                            result2 = self._s96b_read_worker_line(proc2, timeout=timeout)
                            if result2 and result2.get("status") != "error":
                                return result2
                    except Exception as exc2:
                        self.logger.error(
                            f"[S96B] GPU-{gpu_id} restart also failed ({exc2}) - subprocess fallback"
                        )
                        workers[gpu_id]["alive"] = False
                else:
                    self.logger.warning(
                        f"[S96B] GPU-{gpu_id} restart failed - subprocess fallback for remaining trials"
                    )
                return fallback_fn()

'''

# =============================================================================
# Patch 2: Wrap study.optimize() with S96B worker spawn/shutdown
# Anchor line (exact): "        study.optimize(self._optuna_objective, n_trials=n_trials, n_jobs=n_jobs)"
# =============================================================================

STUDY_OPTIMIZE_OLD = "        study.optimize(self._optuna_objective, n_trials=n_trials, n_jobs=n_jobs)"

STUDY_OPTIMIZE_NEW = """\
        # [S96B] Spawn persistent workers if --persistent-workers active
        _s96b_workers = {}
        if getattr(self, '_s96b_use_persistent_workers', False):
            _s96b_gpu_count = self._s95_detect_cuda_gpus_no_torch()
            _s96b_gpu_ids = list(range(_s96b_gpu_count))
            if _s96b_gpu_ids:
                self.logger.info(
                    f"[S96B] Spawning {len(_s96b_gpu_ids)} persistent GPU workers: {_s96b_gpu_ids}"
                )
                _s96b_workers = self._spawn_persistent_workers(_s96b_gpu_ids)
                self.logger.info(f"[S96B] {len(_s96b_workers)} workers ready")
            else:
                self.logger.warning("[S96B] No CUDA GPUs found - persistent-workers disabled")
        self._s96b_workers = _s96b_workers   # make available to _optuna_objective
        try:
            study.optimize(self._optuna_objective, n_trials=n_trials, n_jobs=n_jobs)
        finally:
            if _s96b_workers:
                self.logger.info("[S96B] Shutting down persistent workers")
                self._shutdown_persistent_workers(_s96b_workers)
                self._s96b_workers = {}"""

# =============================================================================
# Patch 3: Route _run_nn_optuna_trial through worker when available
# Anchor: "            proc = subprocess.run("  inside _run_nn_optuna_trial
# We replace the subprocess.run block with a worker dispatch that falls back
# to the existing subprocess.run path.
# =============================================================================

# The exact block to replace in _run_nn_optuna_trial (lines ~1862-1884 approx)
# We look for the distinctive subprocess.run call inside this method.
# Strategy: replace the `proc = subprocess.run(` assignment + parse block
# with a worker dispatch + fallback. We find the boundary by searching for
# the specific comment + proc = subprocess.run pattern inside the method.

NN_TRIAL_OLD_ANCHOR = """            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=600,
                cwd=str(Path(__file__).parent),
                env=sub_env
            )
            
            # Parse JSON from last stdout line
            r2 = -999.0
            train_mse = 0.0
            val_mse = float('inf')
            
            if proc.returncode == 0:
                try:
                    for line in (proc.stdout or "").strip().split("\\n"):
                        line = line.strip()
                        if line.startswith("{") and line.endswith("}"):
                            output = json.loads(line)
                            r2 = float(output.get("r2", -999.0))
                            train_mse = float(output.get("train_mse", 0.0))
                            val_mse = float(output.get("val_mse", float('inf')))
                            break
                except Exception as parse_err:
                    self.logger.warning(
                        f"[Phase 2.2] Could not parse subprocess output "
                        f"(trial {trial_number} fold {fold_idx}): {parse_err}"
                    )
            else:
                stderr_tail = (proc.stderr or "")[-300:]
                self.logger.warning(
                    f"[Phase 2.2] Subprocess failed (trial {trial_number} fold {fold_idx}, "
                    f"rc={proc.returncode}): {stderr_tail}"
                )"""

NN_TRIAL_NEW = """            # [S96B] Route through persistent worker if available, else S96A subprocess
            _s96b_workers = getattr(self, '_s96b_workers', {})
            _gpu_id_for_dispatch = gpu_id if gpu_id is not None else 0

            if _s96b_workers and _gpu_id_for_dispatch in _s96b_workers:
                # Build worker job from already-constructed cmd parameters
                _worker_job = {
                    "command":           "train",
                    "X_train_path":      npz_path,
                    "params":            config,
                    "trial_number":      trial_number,
                    "fold_idx":          fold_idx,
                    "normalize_features": True,    # Category B always ON
                    "use_leaky_relu":     True,    # Category B always ON
                    "batch_mode":         "auto",  # [S96A]
                }

                def _subprocess_fallback():
                    from pathlib import Path as _Path  # [S96B polish] local alias for consistency
                    proc_fb = subprocess.run(
                        cmd, capture_output=True, text=True,
                        timeout=600,
                        cwd=str(_Path(__file__).parent),
                        env=sub_env
                    )
                    r2_fb = -999.0; train_mse_fb = 0.0; val_mse_fb = float('inf')
                    if proc_fb.returncode == 0:
                        try:
                            for _line in (proc_fb.stdout or "").strip().split("\\n"):
                                _line = _line.strip()
                                if _line.startswith("{") and _line.endswith("}"):
                                    _out = json.loads(_line)
                                    r2_fb = float(_out.get("r2", -999.0))
                                    train_mse_fb = float(_out.get("train_mse", 0.0))
                                    val_mse_fb   = float(_out.get("val_mse", float('inf')))
                                    break
                        except Exception:
                            pass
                    return {"r2": r2_fb, "train_mse": train_mse_fb, "val_mse": val_mse_fb}

                _result = self._s96b_dispatch(
                    _s96b_workers, _gpu_id_for_dispatch, _worker_job,
                    fallback_fn=_subprocess_fallback, timeout=60
                )
                r2        = float(_result.get("r2", -999.0))
                train_mse = float(_result.get("train_mse", 0.0))
                val_mse   = float(_result.get("val_mse", float('inf')))

            else:
                # [S96A] Original subprocess path (unchanged)
                proc = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=600,
                    cwd=str(Path(__file__).parent),
                    env=sub_env
                )

                # Parse JSON from last stdout line
                r2 = -999.0
                train_mse = 0.0
                val_mse = float('inf')

                if proc.returncode == 0:
                    try:
                        for line in (proc.stdout or "").strip().split("\\n"):
                            line = line.strip()
                            if line.startswith("{") and line.endswith("}"):
                                output = json.loads(line)
                                r2 = float(output.get("r2", -999.0))
                                train_mse = float(output.get("train_mse", 0.0))
                                val_mse = float(output.get("val_mse", float('inf')))
                                break
                    except Exception as parse_err:
                        self.logger.warning(
                            f"[Phase 2.2] Could not parse subprocess output "
                            f"(trial {trial_number} fold {fold_idx}): {parse_err}"
                        )
                else:
                    stderr_tail = (proc.stderr or "")[-300:]
                    self.logger.warning(
                        f"[Phase 2.2] Subprocess failed (trial {trial_number} fold {fold_idx}, "
                        f"rc={proc.returncode}): {stderr_tail}"
                    )"""

# =============================================================================
# Patch 4: CLI flags - add after --allow-inline-nn-fallback argument
# =============================================================================

CLI_ANCHOR = "    parser.add_argument('--allow-inline-nn-fallback', action='store_true',"

CLI_FLAGS_NEW = """\
    # [S96B] Persistent GPU workers
    parser.add_argument('--persistent-workers', action='store_true',
                       default=False, dest='persistent_workers',
                       help='[S96B] Persistent GPU workers for NN trials (default OFF)')
    parser.add_argument('--no-persistent-workers', action='store_false',
                       dest='persistent_workers',
                       help='[S96B] Disable persistent GPU workers (default)')
"""

# =============================================================================
# Patch 5: Thread persistent_workers into optimizer instance
# Anchor: "    optimizer._cli_enable_diagnostics = getattr(args, 'enable_diagnostics', False)"
# =============================================================================

THREAD_ANCHOR = "    optimizer._cli_enable_diagnostics = getattr(args, 'enable_diagnostics', False)"

THREAD_NEW = """\
    optimizer._cli_enable_diagnostics = getattr(args, 'enable_diagnostics', False)
    # [S96B] Thread persistent-workers flag into optimizer
    optimizer._s96b_use_persistent_workers = getattr(args, 'persistent_workers', False)"""


# =============================================================================
# Patch application engine
# =============================================================================

def check_already_patched(source: str) -> bool:
    return "_s96b_use_persistent_workers" in source or "_s96b_workers" in source


def apply_all_patches(source: str, dry_run: bool) -> str:
    changes = []

    # ── Patch 1: Insert S96B worker methods ──────────────────────────────────
    anchor1 = "    def _sample_hyperparameters(self, trial) -> Dict:"
    if anchor1 in source:
        if not dry_run:
            source = source.replace(anchor1, S96B_WORKER_METHODS + anchor1, 1)
        changes.append("✅ Patch 1: S96B worker methods inserted before _sample_hyperparameters")
    else:
        changes.append("❌ Patch 1: Anchor '_sample_hyperparameters' NOT FOUND")

    # ── Patch 2: Wrap study.optimize() ───────────────────────────────────────
    if STUDY_OPTIMIZE_OLD in source:
        if not dry_run:
            source = source.replace(STUDY_OPTIMIZE_OLD, STUDY_OPTIMIZE_NEW, 1)
        changes.append("✅ Patch 2: study.optimize() wrapped with S96B worker spawn/shutdown")
    else:
        changes.append("❌ Patch 2: study.optimize() anchor NOT FOUND")

    # ── Patch 3: Route _run_nn_optuna_trial through worker ───────────────────
    if NN_TRIAL_OLD_ANCHOR in source:
        if not dry_run:
            source = source.replace(NN_TRIAL_OLD_ANCHOR, NN_TRIAL_NEW, 1)
        changes.append("✅ Patch 3: _run_nn_optuna_trial routes through S96B worker with S96A fallback")
    else:
        changes.append("❌ Patch 3: _run_nn_optuna_trial subprocess.run anchor NOT FOUND")

    # ── Patch 4: CLI flags ────────────────────────────────────────────────────
    if CLI_ANCHOR in source:
        if not dry_run:
            # Insert BEFORE the allow-inline-nn-fallback line
            source = source.replace(CLI_ANCHOR, CLI_FLAGS_NEW + CLI_ANCHOR, 1)
        changes.append("✅ Patch 4: CLI flags --persistent-workers / --no-persistent-workers added")
    else:
        changes.append("❌ Patch 4: CLI anchor '--allow-inline-nn-fallback' NOT FOUND")

    # ── Patch 5: Thread flag into optimizer ───────────────────────────────────
    if THREAD_ANCHOR in source:
        if not dry_run:
            source = source.replace(THREAD_ANCHOR, THREAD_NEW, 1)
        changes.append("✅ Patch 5: _s96b_use_persistent_workers threaded into optimizer instance")
    else:
        changes.append("❌ Patch 5: Thread anchor NOT FOUND")

    return source, changes


def main():
    parser = argparse.ArgumentParser(description="Apply S96B persistent GPU worker patch")
    parser.add_argument("--apply",  action="store_true", help="Apply patch (default: dry-run)")
    parser.add_argument("--verify", action="store_true", help="Verify S96B markers present")
    parser.add_argument("--target", default=str(TARGET), help=f"Target file (default: {TARGET})")
    args = parser.parse_args()

    target = Path(args.target)
    if not target.exists():
        print(f"❌ Target not found: {target}")
        sys.exit(1)

    source = target.read_text(encoding="utf-8")
    print(f"{'='*70}")
    print(f"S96B Persistent GPU Worker Patch")
    print(f"Target : {target}  ({len(source.splitlines())} lines)")
    print(f"Mode   : {'VERIFY' if args.verify else 'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"{'='*70}\n")

    if args.verify:
        if check_already_patched(source):
            print("✅ S96B markers detected - patch has been applied")
        else:
            print("❌ S96B markers NOT found - patch not yet applied")
        return

    if check_already_patched(source):
        print("⚠️  S96B markers already present - aborting to prevent double-patch")
        sys.exit(1)

    _, changes = apply_all_patches(source, dry_run=True)
    print("Planned changes:")
    for c in changes:
        print(f"  {c}")

    failed = [c for c in changes if c.startswith("❌")]
    if failed:
        print(f"\n❌ {len(failed)} patch(es) would fail. Fix anchors before applying.")
        sys.exit(1)

    if not args.apply:
        print("\n[DRY-RUN] All anchors found. Re-run with --apply to patch.")
        return

    # Apply
    backup = target.with_suffix(
        f".py.s96b_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copy2(target, backup)
    print(f"\n✅ Backup: {backup}")

    patched, _ = apply_all_patches(source, dry_run=False)
    target.write_text(patched, encoding="utf-8")
    print(f"✅ Written: {target} ({len(patched.splitlines())} lines)")

    try:
        ast.parse(patched)
        print("✅ Syntax verification passed")
    except SyntaxError as exc:
        print(f"❌ SYNTAX ERROR: {exc}")
        print(f"   Restoring backup...")
        shutil.copy2(backup, target)
        print(f"   Backup restored.")
        sys.exit(1)

    print(f"\n{'='*70}")
    print("Acceptance tests:")
    print("  # Test 0: Syntax")
    print("  python3 -c \"import ast; ast.parse(open('meta_prediction_optimizer_anti_overfit.py').read()); print('OK')\"")
    print()
    print("  # Test 1: Verify markers")
    print("  python3 apply_s96b_patch.py --verify")
    print()
    print("  # Test 2: Worker smoke test")
    print("  echo '{\"command\":\"shutdown\"}' | CUDA_VISIBLE_DEVICES=0 python3 nn_gpu_worker.py")
    print()
    print("  # Test 3: WATCHER compare-models default (S96A path, workers OFF)")
    print("  PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline \\")
    print("    --start-step 5 --end-step 5 --params '{\"compare_models\": true, \"trials\": 20}'")
    print()
    print("  # Test 4: WATCHER compare-models with persistent workers ENABLED")
    print("  PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline \\")
    print("    --start-step 5 --end-step 5 \\")
    print("    --params '{\"compare_models\": true, \"trials\": 20, \"persistent_workers\": true}'")
    print()
    print("  # Test 5: Timing (target: <2 min vs ~10 min S96A)")
    print("  time python3 meta_prediction_optimizer_anti_overfit.py \\")
    print("    --survivors survivors_with_scores.json \\")
    print("    --lottery-data train_history.json \\")
    print("    --model-type neural_net --trials 20 --persistent-workers")


if __name__ == "__main__":
    main()
