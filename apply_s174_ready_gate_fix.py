#!/usr/bin/env python3
"""
S174 Ready-Gate Hard Fix
========================

Implements TB-approved coordinator-side hard ready gate to block dispatch
when ready_count < min_workers. Fixes the wiring bug that caused S174
baseline to dispatch at ready=2 with --min-workers=24.

Three patches, all idempotent:

  Patch 1 — persistent_worker_coordinator.py
    a) Add `min_workers` parameter to run_trial_persistent() signature
    b) Pass `min_workers` to PersistentWorkerCoordinator() constructor
       inside run_trial_persistent()
    c) Replace _tcp_wait_ready() timeout-then-return with timeout-then-RAISE
    d) Replace dispatch-site `_ready == 0` guard with `_ready < min_workers`
       guard that RAISES

  Patch 2 — window_optimizer_integration_final.py
    Add min_workers=getattr(coordinator,'pwc_min_workers',1) to the
    run_trial_persistent() call site

Acceptance criteria post-patch:
  - Log emits "READY GATE PASSED: N/M ready — dispatch allowed" on success
  - Log emits "READY GATE FAILED: N/M ready < min_workers=K — aborting"
    on insufficient ready count
  - RuntimeError raised before any job_assign fires when ready < min_workers

Usage on Zeus:
  cd ~/distributed_prng_analysis
  python3 apply_s174_ready_gate_fix.py            # apply
  python3 apply_s174_ready_gate_fix.py --dry-run  # show diffs only
  python3 apply_s174_ready_gate_fix.py --verify   # AST verify post-apply
"""

import argparse
import ast
import os
import shutil
import sys
import time
from pathlib import Path

PWC_FILE = "persistent_worker_coordinator.py"
WOI_FILE = "window_optimizer_integration_final.py"

# ============================================================================
# Patch 1a + 1b: run_trial_persistent signature + PWC ctor
# ============================================================================

PWC_OLD_SIG = """def run_trial_persistent(coordinator_cfg: str,
                         config,           # WindowConfig from window_optimizer
                         trial_number: int,
                         prng_base: str,
                         residues: List[int],
                         total_seeds: int,
                         forward_threshold: float,
                         reverse_threshold: float,
                         test_both_modes: bool,
                         dataset_path: str = "",
                         worker_pool_size: int = 8,
                         seed_cap_nvidia: int = 5_000_000,
                         seed_cap_amd:   int  = 2_000_000,
                         pwc_transport: str = "ssh",
                         pwc_host: str = "0.0.0.0",
                         pwc_port: int = 5600,
                         node_allowlist=None) -> Dict[str, Any]:  # [S163-KARG-PWC]"""

PWC_NEW_SIG = """def run_trial_persistent(coordinator_cfg: str,
                         config,           # WindowConfig from window_optimizer
                         trial_number: int,
                         prng_base: str,
                         residues: List[int],
                         total_seeds: int,
                         forward_threshold: float,
                         reverse_threshold: float,
                         test_both_modes: bool,
                         dataset_path: str = "",
                         worker_pool_size: int = 8,
                         seed_cap_nvidia: int = 5_000_000,
                         seed_cap_amd:   int  = 2_000_000,
                         pwc_transport: str = "ssh",
                         pwc_host: str = "0.0.0.0",
                         pwc_port: int = 5600,
                         node_allowlist=None,
                         min_workers: int = 1) -> Dict[str, Any]:  # [S163-KARG-PWC] + [S174 ready gate]"""

PWC_OLD_CTOR = """    pwc = PersistentWorkerCoordinator(
        config_file      = coordinator_cfg,
        worker_pool_size = worker_pool_size,
        seed_cap_nvidia  = seed_cap_nvidia,
        seed_cap_amd     = seed_cap_amd,
        pwc_transport    = pwc_transport,
        pwc_host         = pwc_host,
        pwc_port         = pwc_port,
        node_allowlist   = node_allowlist,  # [S163-KARG-PWC] partition node filter
    )"""

PWC_NEW_CTOR = """    pwc = PersistentWorkerCoordinator(
        config_file      = coordinator_cfg,
        worker_pool_size = worker_pool_size,
        seed_cap_nvidia  = seed_cap_nvidia,
        seed_cap_amd     = seed_cap_amd,
        pwc_transport    = pwc_transport,
        pwc_host         = pwc_host,
        pwc_port         = pwc_port,
        node_allowlist   = node_allowlist,  # [S163-KARG-PWC] partition node filter
        min_workers      = min_workers,     # [S174] ready gate
    )"""

# ============================================================================
# Patch 1c: _tcp_wait_ready hard-fail
# ============================================================================

PWC_OLD_WAIT_READY = """    def _tcp_wait_ready(self, expected: int, timeout_s: float = 180.0) -> int:
        \"\"\"
        S161 v2: Wait for workers to report ready (compute-ready after ROCm init).
        Ready = dispatch-eligible. Timeout covers parallel ROCm warmup (~90s).
        Returns count of ready workers when min_workers met or deadline reached.
        \"\"\"
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            count = self._tcp_transport.ready_count()
            if count >= self.min_workers:
                self.logger.info(
                    f"[PWC-TCP] {count}/{expected} workers ready — dispatching"
                )
                return count
            time.sleep(0.5)
        count = self._tcp_transport.ready_count()
        self.logger.warning(
            f"[PWC-TCP] ready timeout: {count}/{expected} workers ready after {timeout_s:.0f}s"
        )
        return count"""

PWC_NEW_WAIT_READY = """    def _tcp_wait_ready(self, expected: int, timeout_s: float = 180.0) -> int:
        \"\"\"
        S161 v2 + S174: Wait for workers to report ready (compute-ready after ROCm init).
        Ready = dispatch-eligible. Timeout covers parallel ROCm warmup (~90s).

        S174 hard gate: on success, emits READY GATE PASSED and returns count.
        On timeout with count < min_workers, shuts down workers, emits
        READY GATE FAILED, and RAISES RuntimeError BEFORE any job dispatch
        can occur.
        \"\"\"
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            count = self._tcp_transport.ready_count()
            if count >= self.min_workers:
                self.logger.info(
                    f"[PWC-TCP] READY GATE PASSED: {count}/{expected} ready "
                    f"(min_workers={self.min_workers}) — dispatch allowed"
                )
                return count
            time.sleep(0.5)
        # Timeout path: hard-fail per S174 spec
        count = self._tcp_transport.ready_count()
        self.logger.error(
            f"[PWC-TCP] READY GATE FAILED: {count}/{expected} ready "
            f"< min_workers={self.min_workers} — aborting before dispatch"
        )
        # S174 TB hardening: actively clean up workers before raising,
        # so failed gate does not leave TCP workers waiting/reconnecting.
        try:
            self.shutdown()
        except Exception as _shutdown_exc:
            self.logger.warning(
                f"[PWC-TCP] READY GATE FAILED cleanup warning: {_shutdown_exc}"
            )
        raise RuntimeError(
            f"PWC TCP ready gate failed: {count}/{expected} ready "
            f"< min_workers={self.min_workers} after {timeout_s:.0f}s timeout"
        )"""

# ============================================================================
# Patch 1d: dispatch-site guard
# ============================================================================

PWC_OLD_DISPATCH_GUARD = """        # Just confirm ready count before dispatch — no waiting needed here
        if self._tcp_transport is not None:
            _ready = self._tcp_transport.ready_count()
            if _ready == 0:
                self.logger.error("[PWC-TCP] no ready workers — aborting dispatch")
                return {"status": "error", "survivor_count": 0,
                        "survivors": [], "failed_chunks": 1, "total_chunks": 1}
            self.logger.info(f"[PWC-TCP] {_ready} ready worker(s) — dispatching")"""

PWC_NEW_DISPATCH_GUARD = """        # S174 defense-in-depth: ready gate already enforced in _tcp_wait_ready,
        # but verify once more at dispatch site to block any race condition.
        if self._tcp_transport is not None:
            _ready = self._tcp_transport.ready_count()
            if _ready < self.min_workers:
                self.logger.error(
                    f"[PWC-TCP] DISPATCH BLOCKED: {_ready} ready "
                    f"< min_workers={self.min_workers} — refusing job_assign"
                )
                raise RuntimeError(
                    f"PWC dispatch blocked: ready={_ready} "
                    f"< min_workers={self.min_workers}"
                )
            self.logger.info(
                f"[PWC-TCP] dispatch confirmed: {_ready} ready workers "
                f"(min_workers={self.min_workers})"
            )"""

# ============================================================================
# Patch 2: window_optimizer_integration_final.py caller
# ============================================================================

WOI_OLD_CALL_TAIL = """            pwc_port          = getattr(coordinator, 'pwc_port', 5600),       # [S163-KARG-FIX1] hop 5
            node_allowlist    = getattr(coordinator, 'node_allowlist', None), # [S163-KARG-PWC] hop 6
        )"""

WOI_NEW_CALL_TAIL = """            pwc_port          = getattr(coordinator, 'pwc_port', 5600),       # [S163-KARG-FIX1] hop 5
            node_allowlist    = getattr(coordinator, 'node_allowlist', None), # [S163-KARG-PWC] hop 6
            min_workers       = getattr(coordinator, 'pwc_min_workers', 1),   # [S174] ready gate wiring
        )"""

# ============================================================================
# Patch engine
# ============================================================================

def replace_once(content: str, old: str, new: str, label: str) -> tuple:
    """Returns (new_content, changed_bool, status_msg)."""
    if new in content and old not in content:
        return content, False, f"  [{label}] already applied — skipping"
    occurrences = content.count(old)
    if occurrences == 0:
        return content, False, f"  [{label}] OLD text not found — UNABLE TO PATCH"
    if occurrences > 1:
        return content, False, f"  [{label}] OLD text found {occurrences}× — ambiguous, refusing"
    return content.replace(old, new), True, f"  [{label}] APPLIED"


def patch_file(path: str, patches: list, dry_run: bool):
    """patches: list of (old, new, label) tuples."""
    p = Path(path)
    if not p.exists():
        print(f"  ERROR: {path} not found")
        return False

    original = p.read_text()
    content = original
    all_ok = True
    any_changed = False

    for old, new, label in patches:
        content, changed, msg = replace_once(content, old, new, label)
        print(msg)
        if "UNABLE" in msg or "ambiguous" in msg:
            all_ok = False
        if changed:
            any_changed = True

    if not all_ok:
        print(f"  ABORT — {path} not modified")
        return False

    if not any_changed:
        print(f"  {path} — no changes needed")
        return True

    if dry_run:
        print(f"  [DRY-RUN] would write {path} ({len(content)} bytes)")
        return True

    backup = f"{path}.bak.s174_{int(time.time())}"
    shutil.copy2(path, backup)
    p.write_text(content)

    # AST verify
    try:
        ast.parse(content)
        print(f"  AST verify: OK  (backup: {backup})")
        return True
    except SyntaxError as e:
        print(f"  AST verify FAILED: {e} — REVERTING")
        shutil.copy2(backup, path)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="show what would change without writing")
    ap.add_argument("--verify",  action="store_true", help="verify expected strings present, no changes")
    args = ap.parse_args()

    if not Path(PWC_FILE).exists():
        print(f"ERROR: must run from repo root containing {PWC_FILE}")
        sys.exit(2)

    print("=" * 70)
    print("S174 Ready-Gate Hard Fix")
    print("=" * 70)

    if args.verify:
        print("\n[verify mode]")
        for f, needles in [
            (PWC_FILE, [
                "READY GATE PASSED",
                "READY GATE FAILED",
                "min_workers      = min_workers,     # [S174]",
                "min_workers: int = 1) -> Dict[str, Any]:  # [S163-KARG-PWC] + [S174 ready gate]",
                "DISPATCH BLOCKED",
            ]),
            (WOI_FILE, [
                "min_workers       = getattr(coordinator, 'pwc_min_workers', 1)",
            ]),
        ]:
            content = Path(f).read_text()
            print(f"\n  {f}:")
            for n in needles:
                ok = n in content
                print(f"    [{'X' if ok else ' '}] {n[:70]}")
        return

    print(f"\n--- Patching {PWC_FILE} ---")
    pwc_ok = patch_file(PWC_FILE, [
        (PWC_OLD_SIG,            PWC_NEW_SIG,            "1a sig+min_workers param"),
        (PWC_OLD_CTOR,           PWC_NEW_CTOR,           "1b ctor min_workers"),
        (PWC_OLD_WAIT_READY,     PWC_NEW_WAIT_READY,     "1c _tcp_wait_ready hard-fail"),
        (PWC_OLD_DISPATCH_GUARD, PWC_NEW_DISPATCH_GUARD, "1d dispatch-site guard"),
    ], dry_run=args.dry_run)

    print(f"\n--- Patching {WOI_FILE} ---")
    woi_ok = patch_file(WOI_FILE, [
        (WOI_OLD_CALL_TAIL, WOI_NEW_CALL_TAIL, "2 caller min_workers"),
    ], dry_run=args.dry_run)

    print()
    print("=" * 70)
    if pwc_ok and woi_ok:
        print("DONE." + (" (dry-run — no files modified)" if args.dry_run else ""))
        if not args.dry_run:
            print("Run:  python3 apply_s174_ready_gate_fix.py --verify")
    else:
        print("PATCH FAILED — see errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
