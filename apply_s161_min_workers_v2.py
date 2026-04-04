#!/usr/bin/env python3
"""
apply_s161_min_workers.py
==========================
TB-approved Gate 2 fix: --min-workers N readiness gate for PWC TCP dispatch.

Changes:
1. Add min_workers param to __init__ (default=1 preserves Gate 1 behavior)
2. Add --min-workers CLI flag
3. Reduce TCP launch stagger 4s -> 1s, track launch_complete_time
4. Replace simple worker_count()==0 wait with full readiness gate:
   - wait until connected >= min_workers OR launch_complete + ready_timeout
   - log expected, connected, deadline, dispatch reason

Apply:
    python3 apply_s161_min_workers.py --dry-run
    python3 apply_s161_min_workers.py
"""
import argparse, ast, shutil, sys
from pathlib import Path

TARGET = Path("persistent_worker_coordinator.py")
BACKUP = Path("persistent_worker_coordinator.py.pre_s161_min_workers")

# TCP stagger constant for TCP-only path (1s vs 4s for SSH-PWC)
TCP_SPAWN_STAGGER_S = 1.0
# ROCm startup budget after last launch
ROCM_READY_TIMEOUT_S = 110.0

# ---------------------------------------------------------------------------
# Patch 0: Add TCP-specific constants to coordinator module level
# ---------------------------------------------------------------------------
OLD_CONSTANTS = "ROCM_SPAWN_STAGGER_S   = 4.0   # seconds between worker spawns per gpu_id"

NEW_CONSTANTS = (
    "ROCM_SPAWN_STAGGER_S   = 4.0   # seconds between worker spawns per gpu_id\n"
    "TCP_SPAWN_STAGGER_S    = 1.0   # S161: TCP-only stagger (SSH is bootstrap only)\n"
    "ROCM_READY_TIMEOUT_S   = 110.0 # S161: ROCm startup budget after last TCP worker launched"
)

# ---------------------------------------------------------------------------
# Patch 1: Add min_workers to __init__
# ---------------------------------------------------------------------------
OLD_INIT = '''                 pwc_transport: str = "ssh",
                 pwc_host: str = "0.0.0.0",
                 pwc_port: int = 5600):
        self.config_file     = config_file
        self.worker_pool_size = worker_pool_size
        self.seed_cap_nvidia = seed_cap_nvidia
        self.seed_cap_amd    = seed_cap_amd
        self.max_per_node    = max_per_node
        # [TCP transport] additive — default ssh preserves legacy behavior
        self.pwc_transport   = pwc_transport
        self.pwc_host        = pwc_host
        self.pwc_port        = pwc_port
        self._tcp_transport  = None  # lazily started if pwc_transport == "tcp"'''

NEW_INIT = '''                 pwc_transport: str = "ssh",
                 pwc_host: str = "0.0.0.0",
                 pwc_port: int = 5600,
                 min_workers: int = 1):
        self.config_file     = config_file
        self.worker_pool_size = worker_pool_size
        self.seed_cap_nvidia = seed_cap_nvidia
        self.seed_cap_amd    = seed_cap_amd
        self.max_per_node    = max_per_node
        # [TCP transport] additive — default ssh preserves legacy behavior
        self.pwc_transport   = pwc_transport
        self.pwc_host        = pwc_host
        self.pwc_port        = pwc_port
        self.min_workers     = min_workers  # S161: readiness gate
        self._tcp_transport  = None  # lazily started if pwc_transport == "tcp"
        self._tcp_launch_complete_time: float = 0.0  # set by _tcp_launch_workers()
        self._tcp_expected_workers: int = 0           # set by _tcp_launch_workers()'''

# ---------------------------------------------------------------------------
# Patch 2: Reduce TCP stagger + track launch_complete_time
# ---------------------------------------------------------------------------
OLD_STAGGER = '''                # Stagger to prevent simultaneous HIP init
                if gpu_id < pool - 1:
                    _t.sleep(ROCM_SPAWN_STAGGER_S)

        self.logger.info(
            f"[PWC-TCP] {total_launched} workers launched across all rigs — "
            f"waiting up to 180s for connections"
        )'''

NEW_STAGGER = '''                # Stagger — TCP path uses 1s (SSH is bootstrap only, not compute)
                if gpu_id < pool - 1:
                    _t.sleep(TCP_SPAWN_STAGGER_S)

        import time as _t2
        self._tcp_launch_complete_time = _t2.time()
        self._tcp_expected_workers = total_launched  # TB: use actual launched count
        self.logger.info(
            "[PWC-TCP] " + str(total_launched) + " workers launched across all rigs"
            " — launch_complete_time recorded, readiness gate active"
        )'''

# ---------------------------------------------------------------------------
# Patch 3: Replace simple worker_count()==0 wait with full readiness gate
# ---------------------------------------------------------------------------
OLD_WAIT = '''        # Build chunk list — divide total seeds across all available workers
        # S161: in TCP mode, wait for at least 1 worker to connect before dispatch
        if self._tcp_transport is not None:
            _wait_start = time.time()
            while self._tcp_transport.worker_count() == 0:
                if time.time() - _wait_start > 180.0:
                    self.logger.error("[PWC-TCP] No workers connected after 180s — aborting")
                    return {"status": "error", "survivor_count": 0,
                            "survivors": [], "failed_chunks": 1, "total_chunks": 1}
                time.sleep(0.5)
            self.logger.info(f"[PWC-TCP] {self._tcp_transport.worker_count()} worker(s) connected — dispatching")'''

NEW_WAIT = '''        # Build chunk list — divide total seeds across all available workers
        # S161: TB-approved readiness gate — wait for min_workers OR launch_complete + timeout
        if self._tcp_transport is not None:
            _expected    = getattr(self, "_tcp_expected_workers", 0)  # actual launched count
            _min_needed  = self.min_workers
            _launch_done = self._tcp_launch_complete_time
            _ready_deadline = _launch_done + ROCM_READY_TIMEOUT_S
            _fallback_deadline = time.time() + 180.0  # absolute fallback

            if _min_needed > _expected > 0:
                self.logger.warning(
                    f"[PWC-TCP] min_workers={_min_needed} exceeds expected={_expected} "
                    f"(actual launched) — will wait until ready_deadline then dispatch"
                )
            self.logger.info(
                f"[PWC-TCP] readiness gate: min_workers={_min_needed} "
                f"expected={_expected} ready_deadline=+{max(0, _ready_deadline - time.time()):.0f}s"
            )

            while True:
                _connected = self._tcp_transport.worker_count()
                _now = time.time()
                if _connected >= _min_needed:
                    self.logger.info(
                        f"[PWC-TCP] {_connected}/{_expected} worker(s) connected "
                        f"— min_workers threshold reached, dispatching"
                    )
                    break
                if _launch_done > 0 and _now > _ready_deadline:
                    if _connected == 0:
                        self.logger.error(
                            f"[PWC-TCP] No workers connected after ready_deadline — aborting"
                        )
                        return {"status": "error", "survivor_count": 0,
                                "survivors": [], "failed_chunks": 1, "total_chunks": 1}
                    self.logger.info(
                        f"[PWC-TCP] {_connected}/{_expected} worker(s) connected "
                        f"— ready_deadline expired, dispatching with available workers"
                    )
                    break
                if _now > _fallback_deadline:
                    self.logger.error("[PWC-TCP] absolute timeout — aborting")
                    return {"status": "error", "survivor_count": 0,
                            "survivors": [], "failed_chunks": 1, "total_chunks": 1}
                time.sleep(0.5)'''

# ---------------------------------------------------------------------------
# Patch 4: Add --min-workers CLI flag
# ---------------------------------------------------------------------------
OLD_CLI = '''    p.add_argument("--pwc-port",      type=int, default=5600,
                   help="TCP server port (tcp mode only)")'''

NEW_CLI = '''    p.add_argument("--pwc-port",      type=int, default=5600,
                   help="TCP server port (tcp mode only)")
    p.add_argument("--min-workers",   type=int, default=1,
                   help="[TCP] minimum connected workers before dispatch (default=1)")'''

# ---------------------------------------------------------------------------
# Patch 5: Pass min_workers to PWC in smoke test
# ---------------------------------------------------------------------------
OLD_SMOKE = '''    pwc = PersistentWorkerCoordinator(
        config_file      = args.config,
        worker_pool_size = args.pool_size,
        pwc_transport    = args.pwc_transport,
        pwc_host         = args.pwc_host,
        pwc_port         = args.pwc_port,'''

NEW_SMOKE = '''    pwc = PersistentWorkerCoordinator(
        config_file      = args.config,
        worker_pool_size = args.pool_size,
        pwc_transport    = args.pwc_transport,
        pwc_host         = args.pwc_host,
        pwc_port         = args.pwc_port,
        min_workers      = args.min_workers,'''

PATCHES = [
    ("module constants",            OLD_CONSTANTS, NEW_CONSTANTS),
    ("__init__ min_workers param",    OLD_INIT,    NEW_INIT),
    ("TCP stagger 1s + launch time",  OLD_STAGGER, NEW_STAGGER),
    ("readiness gate",                OLD_WAIT,    NEW_WAIT),
    ("--min-workers CLI flag",        OLD_CLI,     NEW_CLI),
    ("smoke test min_workers",        OLD_SMOKE,   NEW_SMOKE),
]

def apply(dry_run=False):
    content = TARGET.read_text(encoding="utf-8")
    for name, old, new in PATCHES:
        count = content.count(old)
        if count == 0:
            print(f"ERROR: anchor not found for [{name}]")
            sys.exit(1)
        if count > 1:
            print(f"ERROR: {count} matches for [{name}]")
            sys.exit(1)
        print(f"OK anchor: [{name}]")

    if dry_run:
        print("DRY RUN — no files modified")
        return

    shutil.copy(TARGET, BACKUP)
    print(f"Backup: {BACKUP}")

    for name, old, new in PATCHES:
        content = content.replace(old, new, 1)
        print(f"Applied: [{name}]")

    try:
        ast.parse(content)
    except SyntaxError as e:
        print(f"AST FAIL line {e.lineno}: {e.msg}")
        lines = content.splitlines()
        for i in range(max(0, e.lineno-3), min(len(lines), e.lineno+3)):
            print(f"{i+1}: {lines[i]}")
        sys.exit(1)
    print("AST OK")
    TARGET.write_text(content, encoding="utf-8")
    print(f"Written: {TARGET}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    apply(args.dry_run)
