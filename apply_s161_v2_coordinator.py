#!/usr/bin/env python3
"""
apply_s161_v2_coordinator.py
==============================
Coordinator-side changes for pwc_worker_service.py v2 two-phase startup.

Changes to persistent_worker_coordinator.py:
1. _tcp_launch_workers: remove per-worker connect wait (workers connect fast now)
2. New _tcp_wait_online(): wait for all workers to report "online"
3. New _tcp_broadcast_init(): send init command to all online workers
4. New _tcp_wait_ready(): wait for all workers to report "ready"
5. startup(): call these three methods after launching workers
6. run_sieve_pass(): replace worker-wait gate with ready_count check

Changes to persistent/pwc_transport_tcp.py:
1. _handle_client: handle "online" message → add to _online_workers
2. _handle_client: handle "ready" message → add to _ready_workers
3. New online_count() and ready_count() methods
4. New broadcast_init() method
5. Late joiner handling: if init already sent, send init immediately on connect

Apply:
    python3 apply_s161_v2_coordinator.py --dry-run
    python3 apply_s161_v2_coordinator.py
"""
import argparse, ast, shutil, sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Patch pwc_transport_tcp.py
# ─────────────────────────────────────────────────────────────────────────────

TCP_TARGET = Path("persistent/pwc_transport_tcp.py")
TCP_BACKUP = Path("persistent/pwc_transport_tcp.py.pre_s161_v2")

# Patch T1: Add online/ready tracking to __init__
TCP_OLD_INIT = '''        self._workers: Dict[str, FramedSocket] = {}  # worker_id → conn
        self._workers_lock = threading.Lock()'''

TCP_NEW_INIT = '''        self._workers: Dict[str, FramedSocket] = {}  # worker_id → conn
        self._workers_lock = threading.Lock()

        # S161 v2: two-phase state tracking
        self._online_workers: set = set()   # TCP connected, not yet compute-ready
        self._ready_workers:  set = set()   # compute-ready after init
        self._state_lock = threading.Lock()
        self._init_sent: bool = False       # True after broadcast_init()'''

# Patch T2: Add online_count(), ready_count(), broadcast_init() methods
TCP_OLD_WORKER_COUNT = '''    def worker_count(self) -> int:
        with self._workers_lock:
            return len(self._workers)'''

TCP_NEW_WORKER_COUNT = '''    def worker_count(self) -> int:
        with self._workers_lock:
            return len(self._workers)

    def online_count(self) -> int:
        """Workers that have reported online (TCP connected, not compute-ready)."""
        with self._state_lock:
            return len(self._online_workers)

    def ready_count(self) -> int:
        """Workers that have reported ready (compute-ready after init)."""
        with self._state_lock:
            return len(self._ready_workers)

    def broadcast_init(self) -> int:
        """
        Send init command to all currently online workers.
        Sets _init_sent so late joiners get init immediately on connect.
        Returns count of workers init was sent to.
        """
        with self._state_lock:
            self._init_sent = True
        with self._workers_lock:
            targets = list(self._workers.items())
        sent = 0
        for worker_id, conn in targets:
            try:
                conn.send_obj({
                    "message_type": "command",
                    "command":      "init",
                    "worker_id":    "coordinator",
                    "timestamp":    time.time(),
                })
                sent += 1
            except Exception:
                pass
        return sent'''

# Patch T3: Handle "online" and "ready" in _handle_client, late-joiner init
TCP_OLD_HANDLE = '''            with self._workers_lock:
                self._workers[worker_id] = conn
            with self._last_seen_lock:
                self._last_seen[worker_id] = time.time()

            # Main dispatch loop
            while not self._stop.is_set():
                msg = conn.recv_obj()
                mtype = msg.get("message_type")

                # Update last-seen on every message — TB blocker E fix
                with self._last_seen_lock:
                    self._last_seen[worker_id] = time.time()

                if mtype == "request_job":'''

TCP_NEW_HANDLE = '''            with self._workers_lock:
                self._workers[worker_id] = conn
            with self._last_seen_lock:
                self._last_seen[worker_id] = time.time()

            # S161 v2: if init already broadcast, send it immediately (late joiner)
            with self._state_lock:
                _init_already_sent = self._init_sent
            if _init_already_sent:
                try:
                    conn.send_obj({
                        "message_type": "command",
                        "command":      "init",
                        "worker_id":    "coordinator",
                        "timestamp":    time.time(),
                    })
                except Exception:
                    pass

            # Main dispatch loop
            while not self._stop.is_set():
                msg = conn.recv_obj()
                mtype = msg.get("message_type")

                # Update last-seen on every message — TB blocker E fix
                with self._last_seen_lock:
                    self._last_seen[worker_id] = time.time()

                if mtype == "online":
                    # S161 v2: worker is TCP-connected, not yet compute-ready
                    with self._state_lock:
                        self._online_workers.add(worker_id)
                    continue

                elif mtype == "ready":
                    # S161 v2: worker completed ROCm warmup — compute-ready
                    with self._state_lock:
                        self._ready_workers.add(worker_id)
                    continue

                elif mtype == "request_job":'''

# Patch T4: Clean up online/ready on disconnect
TCP_OLD_FINALLY = '''        finally:
            # Reclaim any inflight jobs for this worker
            self._reclaim_worker_jobs(worker_id)
            with self._workers_lock:
                self._workers.pop(worker_id, None)
            with self._last_seen_lock:
                self._last_seen.pop(worker_id, None)
            conn.close()'''

TCP_NEW_FINALLY = '''        finally:
            # Reclaim any inflight jobs for this worker
            self._reclaim_worker_jobs(worker_id)
            with self._workers_lock:
                self._workers.pop(worker_id, None)
            with self._last_seen_lock:
                self._last_seen.pop(worker_id, None)
            with self._state_lock:
                self._online_workers.discard(worker_id)
                self._ready_workers.discard(worker_id)
            conn.close()'''

TCP_PATCHES = [
    ("init state tracking",     TCP_OLD_INIT,          TCP_NEW_INIT),
    ("online/ready/broadcast",  TCP_OLD_WORKER_COUNT,  TCP_NEW_WORKER_COUNT),
    ("handle online/ready",     TCP_OLD_HANDLE,        TCP_NEW_HANDLE),
    ("cleanup on disconnect",   TCP_OLD_FINALLY,       TCP_NEW_FINALLY),
]

# ─────────────────────────────────────────────────────────────────────────────
# Patch persistent_worker_coordinator.py
# ─────────────────────────────────────────────────────────────────────────────

COORD_TARGET = Path("persistent_worker_coordinator.py")
COORD_BACKUP = Path("persistent_worker_coordinator.py.pre_s161_v2_coord")

# Patch C1: Remove per-worker connect wait from _tcp_launch_workers
COORD_OLD_WAIT = '''                # S161: Mirror SSH-PWC — wait for this worker to connect before
                # launching next GPU. Prevents simultaneous ROCm context competition.
                # CRITICAL: deadline starts from _launch_time not from previous connect,
                # because ROCm init on this GPU starts immediately after launch script runs.
                # Each GPU takes ~WORKER_HEARTBEAT_TIMEOUT_S to init independently.
                _prev_count = self._tcp_transport.worker_count()
                _deadline = _launch_time + WORKER_HEARTBEAT_TIMEOUT_S
                while _t.time() < _deadline:
                    if self._tcp_transport.worker_count() > _prev_count:
                        _connected = True
                        break
                    _t.sleep(0.5)
                if _connected:
                    _launch_time = _t.time()  # reset for next GPU
                    self.logger.info(
                        "[PWC-TCP] " + host + ":GPU" + str(gpu_id) +
                        " ready (" + str(self._tcp_transport.worker_count()) + " total connected)"
                    )
                else:
                    self.logger.warning(
                        "[PWC-TCP] " + host + ":GPU" + str(gpu_id) +
                        " did not connect within " + str(WORKER_HEARTBEAT_TIMEOUT_S) + "s"
                    )'''

COORD_NEW_WAIT = '''                # S161 v2: no per-worker wait — workers connect fast (no ROCm at startup)
                # All workers launched with 1s stagger, then _tcp_wait_online() handles
                # the online barrier before broadcasting init.
                self.logger.info(
                    "[PWC-TCP] " + host + ":GPU" + str(gpu_id) + " launched"
                )'''

# Patch C2: Remove _connected tracking from total_launched
COORD_OLD_TOTAL = '''                    if _connected:
                        total_launched += 1
                except Exception as e:
                    self.logger.error("[PWC-TCP] " + host + ":GPU" + str(gpu_id) + " launch failed: " + str(e))'''

COORD_NEW_TOTAL = '''                    total_launched += 1
                except Exception as e:
                    self.logger.error("[PWC-TCP] " + host + ":GPU" + str(gpu_id) + " launch failed: " + str(e))'''

# Patch C3: Remove _connected init before try
COORD_OLD_CONNECTED = '''                _connected = False  # S161: init before try so except block can reference it
                try:'''

COORD_NEW_CONNECTED = '''                try:'''

# Patch C4: Add _tcp_wait_online, _tcp_broadcast_init, _tcp_wait_ready methods
# Insert before _ensure_worker_alive
COORD_OLD_ENSURE = '''    def _ensure_worker_alive(self, handle: WorkerHandle) -> bool:
        """Check worker still alive; respawn if dead."""'''

COORD_NEW_ENSURE = '''    def _tcp_wait_online(self, expected: int, timeout_s: float = 30.0) -> int:
        """
        S161 v2: Wait for expected workers to report online (TCP connected).
        Online = fast TCP connect, no ROCm. Timeout is short (30s).
        Returns count of online workers when deadline reached or expected met.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            count = self._tcp_transport.online_count()
            if count >= expected:
                self.logger.info(
                    f"[PWC-TCP] all {count}/{expected} workers online — proceeding to init"
                )
                return count
            time.sleep(0.5)
        count = self._tcp_transport.online_count()
        self.logger.warning(
            f"[PWC-TCP] online timeout: {count}/{expected} workers online after {timeout_s:.0f}s"
        )
        return count

    def _tcp_broadcast_init(self) -> int:
        """
        S161 v2: Broadcast init command to all online workers.
        Workers will import sieve_filter (ROCm warmup) in parallel.
        Returns count of workers init was sent to.
        """
        sent = self._tcp_transport.broadcast_init()
        self.logger.info(
            f"[PWC-TCP] init broadcast to {sent} workers — parallel ROCm warmup starting"
        )
        return sent

    def _tcp_wait_ready(self, expected: int, timeout_s: float = 180.0) -> int:
        """
        S161 v2: Wait for workers to report ready (compute-ready after ROCm init).
        Ready = dispatch-eligible. Timeout covers parallel ROCm warmup (~90s).
        Returns count of ready workers when min_workers met or deadline reached.
        """
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
        return count

    def _ensure_worker_alive(self, handle: WorkerHandle) -> bool:
        """Check worker still alive; respawn if dead."""'''

# Patch C5: Update startup() to call online/init/ready after launch
COORD_OLD_STARTUP_END = '''        import time as _t2
        self._tcp_launch_complete_time = _t2.time()
        self._tcp_expected_workers = total_launched  # TB: use actual launched count
        self.logger.info(
            "[PWC-TCP] " + str(total_launched) + " workers launched across all rigs"
            " — launch_complete_time recorded, readiness gate active"
        )'''

COORD_NEW_STARTUP_END = '''        self.logger.info(
            "[PWC-TCP] " + str(total_launched) + " workers launched across all rigs"
            " — waiting for online, then init, then ready"
        )
        # S161 v2: three-phase startup
        # Phase 1: wait for all workers to come online (fast TCP connect)
        _online = self._tcp_wait_online(expected=total_launched, timeout_s=30.0)
        # Phase 2: broadcast init to all online workers (parallel ROCm warmup)
        self._tcp_broadcast_init()
        # Phase 3: wait for workers to become compute-ready
        _ready = self._tcp_wait_ready(expected=total_launched, timeout_s=180.0)
        self.logger.info(
            f"[PWC-TCP] startup complete: {_online} online, {_ready} ready"
        )'''

# Patch C6: Replace worker-wait gate in run_sieve_pass with ready_count check
COORD_OLD_GATE = '''        # S161: TB-approved readiness gate — wait for min_workers OR launch_complete + timeout
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

COORD_NEW_GATE = '''        # S161 v2: workers already online+ready from startup() three-phase init
        # Just confirm ready count before dispatch — no waiting needed here
        if self._tcp_transport is not None:
            _ready = self._tcp_transport.ready_count()
            if _ready == 0:
                self.logger.error("[PWC-TCP] no ready workers — aborting dispatch")
                return {"status": "error", "survivor_count": 0,
                        "survivors": [], "failed_chunks": 1, "total_chunks": 1}
            self.logger.info(f"[PWC-TCP] {_ready} ready worker(s) — dispatching")'''

# Also update _get_available_workers to use ready_count
COORD_OLD_GET_WORKERS = '''            tcp_count = self._tcp_transport.worker_count()'''
COORD_NEW_GET_WORKERS = '''            tcp_count = self._tcp_transport.ready_count()  # S161 v2: only ready workers'''

COORD_PATCHES = [
    ("remove per-worker wait",       COORD_OLD_WAIT,          COORD_NEW_WAIT),
    ("remove _connected from total", COORD_OLD_TOTAL,         COORD_NEW_TOTAL),
    ("remove _connected init",       COORD_OLD_CONNECTED,     COORD_NEW_CONNECTED),
    ("add wait/init/ready methods",  COORD_OLD_ENSURE,        COORD_NEW_ENSURE),
    ("startup three-phase call",     COORD_OLD_STARTUP_END,   COORD_NEW_STARTUP_END),
    ("run_sieve_pass gate v2",       COORD_OLD_GATE,          COORD_NEW_GATE),
    ("get_workers uses ready_count", COORD_OLD_GET_WORKERS,   COORD_NEW_GET_WORKERS),
]

# ─────────────────────────────────────────────────────────────────────────────
# Apply
# ─────────────────────────────────────────────────────────────────────────────

def apply_patches(target, backup, patches, dry_run):
    content = target.read_text(encoding="utf-8")
    for name, old, new in patches:
        count = content.count(old)
        if count == 0:
            print(f"ERROR: anchor not found for [{name}] in {target}")
            sys.exit(1)
        if count > 1:
            print(f"ERROR: {count} matches for [{name}] in {target}")
            sys.exit(1)
        print(f"OK anchor: [{name}]")

    if dry_run:
        return

    shutil.copy(target, backup)
    print(f"Backup: {backup}")
    for name, old, new in patches:
        content = content.replace(old, new, 1)
        print(f"Applied: [{name}]")

    try:
        ast.parse(content)
    except SyntaxError as e:
        print(f"AST FAIL line {e.lineno}: {e.msg}")
        sys.exit(1)
    print("AST OK")
    target.write_text(content, encoding="utf-8")
    print(f"Written: {target}")

def apply(dry_run=False):
    print(f"\n=== {'DRY RUN' if dry_run else 'APPLYING'}: pwc_transport_tcp.py ===")
    apply_patches(TCP_TARGET, TCP_BACKUP, TCP_PATCHES, dry_run)
    print(f"\n=== {'DRY RUN' if dry_run else 'APPLYING'}: persistent_worker_coordinator.py ===")
    apply_patches(COORD_TARGET, COORD_BACKUP, COORD_PATCHES, dry_run)
    if dry_run:
        print("\nDRY RUN — no files modified")
    else:
        print("\nAll patches applied successfully")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    apply(args.dry_run)
