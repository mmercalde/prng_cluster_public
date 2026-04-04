"""
apply_pwc_tcp_transport_patch.py
==================================
Additive patch to persistent_worker_coordinator.py to wire in
--pwc-transport flag and TCP transport backend.

TB-approved S159G proposal. Team Alpha implementation.

What this patch does (additive only):
1. Adds pwc_transport / pwc_host / pwc_port parameters to
   PersistentWorkerCoordinator.__init__() with defaults that preserve
   existing SSH behavior exactly.
2. Adds tcp_transport property that starts/stops TCPWorkerTransport.
3. Adds pwc_transport parameter to run_trial_persistent() shim.
4. Adds --pwc-transport / --pwc-host / --pwc-port CLI flags to smoke-test.

What this patch does NOT do:
- Does not modify _spawn_worker() — SSH path unchanged
- Does not modify _dispatch_job() — SSH path unchanged
- Does not modify run_sieve_pass() core logic
- Does not change any Step 1 output artifacts
- Default remains --pwc-transport ssh (legacy behavior)

Apply:
    python3 apply_pwc_tcp_transport_patch.py --dry-run
    python3 apply_pwc_tcp_transport_patch.py
"""
from __future__ import annotations

import argparse
import ast
import shutil
import sys
from pathlib import Path


TARGET = Path("persistent_worker_coordinator.py")
BACKUP = Path("persistent_worker_coordinator.py.pre_tcp_transport")


# ---------------------------------------------------------------------------
# Patch 1: Add pwc_transport params to __init__
# ---------------------------------------------------------------------------

OLD_INIT = '''    def __init__(self,
                 config_file: str = "distributed_config.json",
                 worker_pool_size: int = 8,
                 seed_cap_nvidia: int = 5_000_000,
                 seed_cap_amd: int = 2_000_000,
                 max_per_node: int = 8):
        self.config_file     = config_file
        self.worker_pool_size = worker_pool_size
        self.seed_cap_nvidia = seed_cap_nvidia
        self.seed_cap_amd    = seed_cap_amd
        self.max_per_node    = max_per_node'''

NEW_INIT = '''    def __init__(self,
                 config_file: str = "distributed_config.json",
                 worker_pool_size: int = 8,
                 seed_cap_nvidia: int = 5_000_000,
                 seed_cap_amd: int = 2_000_000,
                 max_per_node: int = 8,
                 pwc_transport: str = "ssh",
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
        self._tcp_transport  = None  # lazily started if pwc_transport == "tcp"
        self._progress_writer = None  # guard for TCP early-return startup path'''


# ---------------------------------------------------------------------------
# Patch 2: Add TCP transport startup/shutdown to startup() and shutdown()
# ---------------------------------------------------------------------------

OLD_STARTUP_END = '''        self._started = True
        alive = sum(1 for w in self.workers if w.alive)
        self.logger.info(f"Worker pool ready: {alive}/{len(self.workers)} alive")'''

NEW_STARTUP_END = '''        self._started = True
        alive = sum(1 for w in self.workers if w.alive)
        self.logger.info(f"Worker pool ready: {alive}/{len(self.workers)} alive")

        # [TCP transport] start TCP server if requested — additive, SSH unchanged
        if self.pwc_transport == "tcp":
            from persistent.pwc_transport_base import build_transport
            self._tcp_transport = build_transport(
                pwc_transport=self.pwc_transport,
                pwc_host=self.pwc_host,
                pwc_port=self.pwc_port,
            )
            if self._tcp_transport is not None:
                self._tcp_transport.start()
                self.logger.info(
                    f"[PWC-TCP] transport started on {self.pwc_host}:{self.pwc_port}"
                )'''


OLD_SHUTDOWN_END = '''        self.logger.info("All workers shut down")'''

NEW_SHUTDOWN_END = '''        # [TCP transport] stop TCP server if running — additive
        if self._tcp_transport is not None:
            try:
                self._tcp_transport.stop()
                self.logger.info("[PWC-TCP] transport stopped")
            except Exception as _e:
                self.logger.warning(f"[PWC-TCP] transport stop error: {_e}")
            self._tcp_transport = None

        self.logger.info("All workers shut down")'''


# ---------------------------------------------------------------------------
# Patch 3: Add pwc_transport param to run_trial_persistent()
# ---------------------------------------------------------------------------

OLD_RUN_TRIAL_SIG = '''def run_trial_persistent(coordinator_cfg: str,
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
                         seed_cap_amd:   int  = 2_000_000) -> Dict[str, Any]:'''

NEW_RUN_TRIAL_SIG = '''def run_trial_persistent(coordinator_cfg: str,
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
                         pwc_port: int = 5600) -> Dict[str, Any]:'''


OLD_PWC_INIT_IN_TRIAL = '''    pwc = PersistentWorkerCoordinator(
        config_file      = coordinator_cfg,
        worker_pool_size = worker_pool_size,
        seed_cap_nvidia  = seed_cap_nvidia,
        seed_cap_amd     = seed_cap_amd,
    )'''

NEW_PWC_INIT_IN_TRIAL = '''    pwc = PersistentWorkerCoordinator(
        config_file      = coordinator_cfg,
        worker_pool_size = worker_pool_size,
        seed_cap_nvidia  = seed_cap_nvidia,
        seed_cap_amd     = seed_cap_amd,
        pwc_transport    = pwc_transport,
        pwc_host         = pwc_host,
        pwc_port         = pwc_port,
    )'''


# ---------------------------------------------------------------------------
# Patch 4: Add CLI flags to smoke-test argparse
# ---------------------------------------------------------------------------

OLD_CLI = '''    p.add_argument("--startup-only", action="store_true",'''

NEW_CLI = '''    p.add_argument("--pwc-transport", default="ssh",
                   choices=["ssh", "tcp"],
                   help="Transport backend: ssh (default/legacy) or tcp (new)")
    p.add_argument("--pwc-host",      default="0.0.0.0",
                   help="TCP server bind host (tcp mode only)")
    p.add_argument("--pwc-port",      type=int, default=5600,
                   help="TCP server port (tcp mode only)")
    p.add_argument("--startup-only", action="store_true",'''


# ---------------------------------------------------------------------------
# Patch 5: Pass transport args to PWC in smoke-test
# ---------------------------------------------------------------------------

OLD_SMOKE_PWC = '''    pwc = PersistentWorkerCoordinator(
        config_file      = args.config,
        worker_pool_size = args.pool_size,
    )'''

NEW_SMOKE_PWC = '''    pwc = PersistentWorkerCoordinator(
        config_file      = args.config,
        worker_pool_size = args.pool_size,
        pwc_transport    = args.pwc_transport,
        pwc_host         = args.pwc_host,
        pwc_port         = args.pwc_port,
    )'''


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Patch 6: Dispatch bridge — wire TCP into _run_once (THE decisive integration)
# ---------------------------------------------------------------------------

OLD_RUN_ONCE = '''            def _run_once(wh):
                if isinstance(wh, WorkerHandle):
                    job["gpu_id"] = wh.gpu_id
                    return self._dispatch_to_worker(wh, job)
                else:
                    # Mirror coordinator.py: semaphore limits Zeus to 2 concurrent local jobs
                    job["gpu_id"] = idx % wh.gpu_count
                    self._localhost_semaphore.acquire()
                    try:
                        return self._dispatch_local_sieve(job, wh)
                    finally:
                        self._localhost_semaphore.release()'''

NEW_RUN_ONCE = '''            def _run_once(wh):
                # [TCP transport] dispatch bridge — additive, SSH path unchanged
                if self._tcp_transport is not None:
                    return self._dispatch_to_tcp(job)
                if isinstance(wh, WorkerHandle):
                    job["gpu_id"] = wh.gpu_id
                    return self._dispatch_to_worker(wh, job)
                else:
                    # Mirror coordinator.py: semaphore limits Zeus to 2 concurrent local jobs
                    job["gpu_id"] = idx % wh.gpu_count
                    self._localhost_semaphore.acquire()
                    try:
                        return self._dispatch_local_sieve(job, wh)
                    finally:
                        self._localhost_semaphore.release()'''


# ---------------------------------------------------------------------------
# Patch 7: Add _dispatch_to_tcp() method — submits job, receives normalized result
# ---------------------------------------------------------------------------

OLD_DISPATCH_LOCAL = '''    def _dispatch_local_sieve(self, job: Dict[str, Any], node: WorkerNode) -> Dict[str, Any]:'''

NEW_DISPATCH_LOCAL = '''    def _dispatch_to_tcp(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """
        [TCP transport] Submit job to TCPWorkerTransport and block for result.
        Converts normalized transport result back into the same dict contract
        as _dispatch_to_worker() so all downstream code is transport-blind.
        TB ruling: result normalizer is the contract wall.
        """
        import uuid as _uuid
        if "job_id" not in job:
            job = dict(job)
            job["job_id"] = str(_uuid.uuid4())

        self._tcp_transport.submit_job(job)

        # Block for result correlated by job_id — TB fix 4
        result = self._tcp_transport.recv_result(
            timeout_s=JOB_TIMEOUT_S,
            job_id=job["job_id"],
        )
        if result is None:
            return {"status": "error", "message": "TCP worker timeout"}

        # Normalize to PWC result contract
        if result.get("status") == "error":
            return result

        inner = result.get("result", {})
        payload = inner.get("payload", inner)

        # Accept slim_v1 parallel arrays or legacy dict-list — same as _dispatch_to_worker
        if isinstance(payload, dict) and payload.get("format") == "slim_v1":
            survivors   = [int(s) for s in payload.get("seeds", [])]
            match_rates = list(payload.get("match_rates", []))
            n = len(survivors)
            _is_hybrid_job = bool(job.get("hybrid", False))
            if _is_hybrid_job:
                strat_ids = list(payload.get("strategy_ids", [0]*n))
                skip_seqs = list(payload.get("skip_sequences", [[]]*n))
            else:
                strat_ids = [0] * n
                skip_seqs = [[] for _ in survivors]
        else:
            raw_surv    = payload.get("survivors", [])
            survivors   = [s["seed"]                  if isinstance(s, dict) else int(s) for s in raw_surv]
            match_rates = [s.get("match_rate", 0.5)   if isinstance(s, dict) else 0.5    for s in raw_surv]
            skip_seqs   = [s.get("skip_sequence", []) if isinstance(s, dict) else []      for s in raw_surv]
            strat_ids   = [s.get("strategy_id", 0)    if isinstance(s, dict) else 0       for s in raw_surv]

        return {
            "status":         "ok",
            "job_id":         job["job_id"],
            "survivors":      survivors,
            "match_rates":    match_rates,
            "skip_sequences": skip_seqs,
            "strategy_ids":   strat_ids,
        }

    def _dispatch_local_sieve(self, job: Dict[str, Any], node: WorkerNode) -> Dict[str, Any]:'''


# ---------------------------------------------------------------------------
# Patch 8: startup() — skip SSH worker spawning in TCP mode
# ---------------------------------------------------------------------------

OLD_SPAWN_GUARD = '''        self.logger.info("Starting persistent worker pool...")'''

NEW_SPAWN_GUARD = '''        # [TCP transport] in TCP mode, skip SSH worker spawning entirely
        if self.pwc_transport == "tcp":
            self.logger.info("[PWC-TCP] TCP transport mode — skipping SSH worker spawn")
            self._started = True
            self.logger.info("Worker pool ready: 0/0 alive (TCP mode — workers connect via TCP)")
            from persistent.pwc_transport_base import build_transport
            self._tcp_transport = build_transport(
                pwc_transport=self.pwc_transport,
                pwc_host=self.pwc_host,
                pwc_port=self.pwc_port,
            )
            if self._tcp_transport is not None:
                self._tcp_transport.start()
                self.logger.info(
                    f"[PWC-TCP] transport started on {self.pwc_host}:{self.pwc_port}"
                )
            return

        self.logger.info("Starting persistent worker pool...")'''


OLD_GET_WORKERS = '''    def _get_available_workers(self) -> List:
        """
        Returns list of available dispatch targets — mix of WorkerHandle (AMD)
        and WorkerNode (Zeus local). Only alive, non-quarantined workers included.
        """
        available = []
        # AMD persistent workers
        for w in self.workers:
            if not w.quarantined and w.alive:
                available.append(w)
        # Zeus local nodes
        for node in self.nodes:
            if self._is_localhost(node.hostname):
                available.append(node)
        return available'''

NEW_GET_WORKERS = '''    def _get_available_workers(self) -> List:
        """
        Returns list of available dispatch targets — mix of WorkerHandle (AMD)
        and WorkerNode (Zeus local). Only alive, non-quarantined workers included.
        In TCP mode, derives count from connected TCP workers — TB fix 3.
        """
        # [TCP transport] use TCP worker count as pool size — TB fix 3
        if self._tcp_transport is not None:
            tcp_count = max(1, self._tcp_transport.worker_count())
            # Return synthetic placeholder list sized to connected TCP workers
            # Actual dispatch goes through _dispatch_to_tcp(), not these handles
            return [None] * tcp_count

        available = []
        # AMD persistent workers
        for w in self.workers:
            if not w.quarantined and w.alive:
                available.append(w)
        # Zeus local nodes
        for node in self.nodes:
            if self._is_localhost(node.hostname):
                available.append(node)
        return available'''

PATCHES = [
    ("__init__ params",          OLD_INIT,              NEW_INIT),
    ("_get_available_workers",   OLD_GET_WORKERS,       NEW_GET_WORKERS),
    ("startup SSH skip",         OLD_SPAWN_GUARD,       NEW_SPAWN_GUARD),
    ("shutdown TCP stop",        OLD_SHUTDOWN_END,      NEW_SHUTDOWN_END),
    ("run_trial_persistent sig", OLD_RUN_TRIAL_SIG,     NEW_RUN_TRIAL_SIG),
    ("run_trial_persistent init",OLD_PWC_INIT_IN_TRIAL, NEW_PWC_INIT_IN_TRIAL),
    ("CLI flags",                OLD_CLI,               NEW_CLI),
    ("smoke-test PWC init",      OLD_SMOKE_PWC,         NEW_SMOKE_PWC),
    ("dispatch bridge",          OLD_RUN_ONCE,          NEW_RUN_ONCE),
    ("_dispatch_to_tcp method",  OLD_DISPATCH_LOCAL,    NEW_DISPATCH_LOCAL),
]


def apply(dry_run: bool = False) -> None:
    content = TARGET.read_text(encoding="utf-8")

    for name, old, new in PATCHES:
        count = content.count(old)
        if count == 0:
            print(f"ERROR: anchor not found for patch [{name}]")
            sys.exit(1)
        if count > 1:
            print(f"ERROR: anchor matches {count} times for patch [{name}] — not unique")
            sys.exit(1)
        print(f"OK anchor: [{name}]")

    print()

    if dry_run:
        print("DRY RUN — no files modified")
        return

    shutil.copy(TARGET, BACKUP)
    print(f"Backup: {BACKUP}")

    for name, old, new in PATCHES:
        content = content.replace(old, new, 1)
        print(f"Applied: [{name}]")

    # AST check
    try:
        ast.parse(content)
        print("AST OK")
    except SyntaxError as e:
        print(f"AST FAILED: {e}")
        shutil.copy(BACKUP, TARGET)
        print("Restored from backup")
        sys.exit(1)

    TARGET.write_text(content, encoding="utf-8")
    print(f"Written: {TARGET}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    apply(dry_run=args.dry_run)
