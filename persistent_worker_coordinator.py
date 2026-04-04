#!/usr/bin/env python3
"""
persistent_worker_coordinator.py
=================================
Standalone persistent-worker engine for Step 1 (Window Optimizer) sieve passes.

DROP-IN PARALLEL PATH — activated by --use-persistent-workers flag on window_optimizer.py.
Zero changes to coordinator.py, window_optimizer.py, or window_optimizer_integration_final.py.

WATCHER COMPATIBILITY: Fully transparent. WATCHER passes --use-persistent-workers via
agent_manifests/window_optimizer.json → window_optimizer.py → here. WATCHER never
knows or cares which path ran — output files are identical.

Architecture
------------
- One persistent sieve_gpu_worker.py process per AMD GPU (8 per rig × 3 rigs = 24)
- Zeus GPUs use execute_local_sieve_job() — no persistent worker needed (already fast)
- Workers stay alive across ALL 4 sieve passes (forward, reverse, forward_hybrid, reverse_hybrid)
- Workers receive jobs via stdin (JSON), return results via stdout (JSON)
- Hybrid jobs (steps 3+4) send strategy objects in payload — worker allocates correct buffers

ROCm Stability Envelope (from S130/S133 learnings)
----------------------------------------------------
- Spawn stagger: 4.0s per gpu_id to prevent simultaneous HIP init
- worker_pool_size cap: respects configured limit (default 8 per rig)
- HSA_ENABLE_SDMA=0, HSA_OVERRIDE_GFX_VERSION=10.3.0 always set
- GFXOFF disabled via kernel params (cluster-level, not our concern here)
- Per-rig fault isolation: single rig death quarantines that rig, run continues
- Semaphore throttle: ssh_pool semaphore gates all dispatch (max_per_node limit)
- Heartbeat check before each job dispatch — respawn dead worker

Hybrid Kernel Support (from S133-B root cause analysis)
--------------------------------------------------------
- Hybrid kernels require: skip_sequences_gpu (uint32, n×k), strategy_ids_gpu (uint32),
  strategy_max_misses (int32[]), strategy_tolerances (int32[]), n_strategies (int32)
- strategies loaded from hybrid_strategy.get_all_strategies() if not in payload
- sieve_gpu_worker.py receives strategies in job payload and allocates correct buffers
- Same logic as sieve_filter.py run_hybrid_sieve() — ported to IPC worker path

Job Payload Protocol (stdin → sieve_gpu_worker.py)
----------------------------------------------------
{
  "job_id": "sieve_000",
  "prng_type": "java_lcg",           # or java_lcg_reverse, java_lcg_hybrid, etc.
  "seed_start": 0,
  "seed_end": 192307,
  "residues": [...],                  # window draws
  "window_size": 8,
  "threshold": 0.25,
  "gpu_id": 0,
  "strategies": null,                 # null for constant skip; list of dicts for hybrid
  "phase2_threshold": 0.5             # hybrid only
}

Result Protocol (stdout → coordinator)
---------------------------------------
{"status": "ok", "survivors": [...], "match_rates": [...], "job_id": "..."}
{"status": "error", "message": "..."}

Usage (from window_optimizer_integration_final.py)
---------------------------------------------------
from persistent_worker_coordinator import PersistentWorkerCoordinator

pwc = PersistentWorkerCoordinator(config_file="distributed_config.json",
                                   worker_pool_size=8)
pwc.startup()   # spawns workers, staggers HIP init

# Drop-in replacement for coordinator.execute_distributed_analysis():
result = pwc.run_sieve_pass(
    prng_type="java_lcg",
    residues=residues,
    total_seeds=5_000_000,
    threshold=0.30,
    window_size=8,
    output_file="results/window_opt_forward_8_43_t1.json"
)

pwc.shutdown()  # clean worker teardown
"""

import json
import os
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait as _cf_wait, FIRST_COMPLETED as _CF_FIRST_COMPLETED
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# ROCm stability constants (from S130/S133 learnings)
# ─────────────────────────────────────────────────────────────────────────────
ROCM_SPAWN_STAGGER_S   = 4.0   # seconds between worker spawns per gpu_id
TCP_SPAWN_STAGGER_S    = 1.0   # S161: TCP-only stagger (SSH is bootstrap only)
ROCM_READY_TIMEOUT_S   = 110.0 # S161: ROCm startup budget after last TCP worker launched
ROCM_ENV_VARS = [
    # [S155] Removed CUPY_CUDA_MEMORY_POOL_TYPE=none — caused 41GB VM mmap OOM.
    # Each worker mmaps full 8GB VRAM at device init with pool disabled.
    # 8 workers × 8GB = 64GB VA on a 7.7GB RAM machine → OOM killer.
    # Pool race condition now handled via set_limit() in sieve_gpu_worker.py.
    # Cache race (original S151 concern) addressed separately via per-worker
    # CUPY_CACHE_DIR (S152). These are independent issues. TB ruling: approved.
    #
    # [S155-v2] Propagate pool limit into remote worker env.
    # _spawn_worker() runs: env {rocm_env} python sieve_gpu_worker.py
    # ROCM_ENV_VARS is the only injection path — must include this var or
    # the worker falls back to its hardcoded default silently.
    # Operator can override on any rig: PRNG_CUPY_POOL_LIMIT_MB=512 bash sweep_run1.sh
    "PRNG_CUPY_POOL_LIMIT_MB=256",
    # [S155-v3] Defense-in-depth: CUPY_GPU_MEMORY_LIMIT is CuPy's own env var
    # (confirmed in memory.pyx _parse_limit_string — called at pool construction).
    # Read at SingleDeviceMemoryPool.__init__() before any worker code runs.
    # set_limit() inside Device context later supersedes this per-device value.
    # Layering: env var caps ALL devices at init; set_limit() reinforces for our device.
    # Format: plain integer bytes. 268435456 = 256MB.
    "CUPY_GPU_MEMORY_LIMIT=268435456",
    "HSA_OVERRIDE_GFX_VERSION=10.3.0",
    "HSA_ENABLE_SDMA=0",
    "ROCM_PATH=/opt/rocm",
    "HIP_PATH=/opt/rocm/hip",
    "LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/hip/lib:${LD_LIBRARY_PATH}",
    "PATH=/opt/rocm/bin:${PATH}",
    "CUPY_CACHE_DIR=${HOME}/.cache/cupy",
]
WORKER_SCRIPT = "sieve_gpu_worker.py"
WORKER_HEARTBEAT_TIMEOUT_S = 120  # [S161] 120s allows ROCm sequential init on 8-GPU rigs (90s was edge case)
JOB_TIMEOUT_S = 600               # seconds max per sieve job


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class WorkerNode:
    hostname: str
    gpu_type: str
    gpu_count: int
    python_env: str
    script_path: str
    username: Optional[str] = None

@dataclass
class WorkerHandle:
    node: WorkerNode
    gpu_id: int
    proc: Optional[subprocess.Popen] = None
    alive: bool = False
    quarantined: bool = False
    jobs_completed: int = 0
    jobs_failed: int = 0
    dispatch_lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def key(self) -> Tuple[str, int]:
        return (self.node.hostname, self.gpu_id)


# ─────────────────────────────────────────────────────────────────────────────
# PersistentWorkerCoordinator
# ─────────────────────────────────────────────────────────────────────────────
class PersistentWorkerCoordinator:
    """
    Manages a pool of persistent sieve_gpu_worker.py processes across the cluster.
    Provides run_sieve_pass() as a drop-in replacement for
    coordinator.execute_distributed_analysis() for Step 1 sieve jobs.
    """

    def __init__(self,
                 config_file: str = "distributed_config.json",
                 worker_pool_size: int = 8,
                 seed_cap_nvidia: int = 5_000_000,
                 seed_cap_amd: int = 2_000_000,
                 max_per_node: int = 8,
                 pwc_transport: str = "ssh",
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
        self._tcp_expected_workers: int = 0           # set by _tcp_launch_workers()
        self._progress_writer = None  # guard for TCP early-return startup path

        self.nodes: List[WorkerNode] = []
        self.workers: List[WorkerHandle] = []   # AMD rig workers (persistent)
        self._lock = threading.Lock()
        self._started = False
        # Mirror coordinator.py: semaphore limits Zeus concurrent local jobs (2 GPUs)
        self._localhost_semaphore = threading.Semaphore(2)
        # Per-node semaphores — throttle concurrent dispatch to max_per_node (S133-A lesson)
        self._node_semaphores: Dict[str, threading.Semaphore] = {}
        # [S151] Per-node respawn locks — serialize respawns with stagger to prevent SSH hammer
        self._node_respawn_locks: Dict[str, threading.Lock] = {}

        self.logger = logging.getLogger("PersistentWorkerCoordinator")
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("[PWC] %(levelname)s %(message)s"))
            self.logger.addHandler(h)
        self.logger.setLevel(logging.INFO)

        self._load_config()

    # ─────────────────────────────────────────────────────────────────────────
    # Config
    # ─────────────────────────────────────────────────────────────────────────
    def _load_config(self):
        try:
            with open(self.config_file) as f:
                cfg = json.load(f)
        except Exception as e:
            self.logger.error(f"Cannot load {self.config_file}: {e}")
            return

        for nc in cfg.get("nodes", []):
            node = WorkerNode(
                hostname   = nc["hostname"],
                gpu_type   = nc.get("gpu_type", "unknown"),
                gpu_count  = nc.get("gpu_count", 1),
                python_env = nc["python_env"],
                script_path= nc["script_path"],
                username   = nc.get("username"),
            )
            self.nodes.append(node)
            # Create per-node semaphore — limits concurrent in-flight jobs to max_per_node
            self._node_semaphores[node.hostname] = threading.Semaphore(self.max_per_node)
            # [S151] Per-node respawn lock — serializes respawns with stagger
            self._node_respawn_locks[node.hostname] = threading.Lock()
            self.logger.info(f"Node loaded: {node.hostname} ({node.gpu_count}× {node.gpu_type})")

    def _is_rocm(self, node: WorkerNode) -> bool:
        gt = (node.gpu_type or "").lower()
        return ("rx" in gt) or ("amd" in gt) or ("rocm" in gt)

    def _is_localhost(self, hostname: str) -> bool:
        return hostname in ("localhost", "127.0.0.1", socket.gethostname())

    def _seed_cap(self, node: WorkerNode) -> int:
        if "RTX 3080" in node.gpu_type or "RTX 3090" in node.gpu_type:
            return self.seed_cap_nvidia
        return self.seed_cap_amd

    # ─────────────────────────────────────────────────────────────────────────
    # Worker lifecycle
    # ─────────────────────────────────────────────────────────────────────────
    def startup(self):
        """Spawn persistent workers on all AMD rigs with ROCm stagger."""
        if self._started:
            return
        # ── [S156-BANDAID v2] Pre-spawn targeted cleanup ──────────────────────
        # TEMPORARY SAFETY NET — not the root fix.
        # Root fix: session-scoped PWC (Phase B, S157).
        # TB: targeted --persistent match, SIGTERM first, log exact procs.
        import subprocess as _s156_sp
        import time as _s156_time
        for _s156_node in self.nodes:
            if self._is_localhost(_s156_node.hostname):
                continue
            if not self._is_rocm(_s156_node):
                continue
            try:
                _s156_host = _s156_node.hostname
                _s156_user = _s156_node.username
                _s156_find = (
                    "ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "
                    "-o BatchMode=yes "
                    + _s156_user + "@" + _s156_host + " "
                    + "'pgrep -af \"sieve_gpu_worker.*--persistent\" 2>/dev/null "
                    "|| echo none'"
                )
                _s156_r = _s156_sp.run(
                    _s156_find, shell=True, capture_output=True, text=True, timeout=10
                )
                _s156_found = _s156_r.stdout.strip()
                if _s156_found and _s156_found != "none":
                    self.logger.info(
                        "  [S156] " + _s156_host + ": found stale workers: "
                        + _s156_found[:200]
                    )
                    _s156_reap = (
                        "ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "
                        "-o BatchMode=yes "
                        + _s156_user + "@" + _s156_host + " "
                        + "'pkill -15 -f \"sieve_gpu_worker.*--persistent\" 2>/dev/null; "
                        "sleep 2; "
                        "pkill -9 -f \"sieve_gpu_worker.*--persistent\" 2>/dev/null; "
                        "sleep 1; "
                        "remaining=$(pgrep -c -f \"sieve_gpu_worker.*--persistent\" "
                        "2>/dev/null || echo 0); "
                        "echo \"reaped:$remaining\"'"
                    )
                    _s156_r2 = _s156_sp.run(
                        _s156_reap, shell=True, capture_output=True, text=True, timeout=15
                    )
                    self.logger.info(
                        "  [S156] " + _s156_host + ": reap result: "
                        + _s156_r2.stdout.strip()
                    )
                else:
                    self.logger.info(
                        "  [S156] " + _s156_host + ": no stale persistent workers"
                    )
            except Exception as _s156_e:
                self.logger.warning(
                    "  [S156] " + _s156_host
                    + ": pre-spawn cleanup failed: " + str(_s156_e)
                )
        _s156_time.sleep(2)
        # ── end [S156-BANDAID v2] ───────────────────────────────────────────
        # [TCP transport] in TCP mode, skip SSH worker spawning entirely
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
            # S161: SSH-launch workers on all AMD rigs now that TCP server is bound
            self._tcp_launch_workers()

            # Initialize ProgressWriter for web dashboard — must be done in TCP path
            # SSH-PWC does this after spawn loop but TCP returns early above
            try:
                from progress_display import ProgressWriter
                self._progress_writer = ProgressWriter("Forward Sieve", total_jobs=100, total_seeds=0)
                for node in self.nodes:
                    if self._is_localhost(node.hostname):
                        self._progress_writer.register_node("localhost", "RTX 3080 Ti", 2)
                    else:
                        self._progress_writer.register_node(node.hostname, node.gpu_type, node.gpu_count)
                self.logger.info("[PWC-TCP] ProgressWriter initialized for web dashboard")
            except Exception as e:
                self.logger.warning(f"ProgressWriter unavailable: {e}")
                self._progress_writer = None

            return

        self.logger.info("Starting persistent worker pool...")
        for node in self.nodes:
            if self._is_localhost(node.hostname):
                self.logger.info(f"  Zeus ({node.hostname}) — uses local sieve path, no persistent worker")
                continue
            if not self._is_rocm(node):
                self.logger.info(f"  {node.hostname} — non-ROCm, skipping persistent workers")
                continue
            pool = min(self.worker_pool_size, node.gpu_count)
            self.logger.info(f"  {node.hostname}: spawning {pool} workers (stagger {ROCM_SPAWN_STAGGER_S}s)")
            for gpu_id in range(pool):
                handle = WorkerHandle(node=node, gpu_id=gpu_id)
                success = self._spawn_worker(handle)
                if success:
                    self.workers.append(handle)
                    self.logger.info(f"    ✅ {node.hostname}:GPU{gpu_id} — worker alive")
                else:
                    handle.quarantined = True
                    self.workers.append(handle)
                    self.logger.warning(f"    ⚠️  {node.hostname}:GPU{gpu_id} — spawn failed, quarantined")
                # Stagger to prevent simultaneous HIP init (S130/S133 lesson)
                if gpu_id < pool - 1:
                    time.sleep(ROCM_SPAWN_STAGGER_S)
        self._started = True
        alive = sum(1 for w in self.workers if w.alive)
        self.logger.info(f"Worker pool ready: {alive}/{len(self.workers)} alive")
        # Initialize ProgressWriter for web dashboard (mirrors coordinator.py)
        try:
            from progress_display import ProgressWriter
            self._progress_writer = ProgressWriter("Forward Sieve", total_jobs=100, total_seeds=0)
            for node in self.nodes:
                if self._is_localhost(node.hostname):
                    self._progress_writer.register_node("localhost", "RTX 3080 Ti", 2)
                else:
                    self._progress_writer.register_node(node.hostname, node.gpu_type, node.gpu_count)
        except Exception as e:
            self.logger.warning(f"ProgressWriter unavailable: {e}")
            self._progress_writer = None

        # S161 v2: no hostname map needed — extract IP from worker_id directly
        # worker_id format: "192_168_3_120_gpu0" → IP "192.168.3.120"
        # DNS resolution not used — Zeus cannot resolve rig-* hostnames

    def _spawn_worker(self, handle: WorkerHandle) -> bool:
        """SSH + launch sieve_gpu_worker.py on remote GPU, confirm heartbeat."""
        node   = handle.node
        gpu_id = handle.gpu_id

        # [S155-ROCR] Restore ROCR_VISIBLE_DEVICES per-worker GPU isolation.
        # Root cause of rrig6600c OOM: without ROCR isolation, each worker's
        # HIP runtime enumerates ALL 8 GPUs and maps ALL their VRAM into the
        # process VA space: 8 GPUs × ~5GB = ~41GB VA per worker.
        # Under production load the kernel tries to back these pages → OOM.
        # With ROCR_VISIBLE_DEVICES={gpu_id}: each worker only sees 1 GPU,
        # VA per worker = ~5GB. 8 workers × 5GB = 40GB total — safe.
        # Worker uses Device(0) because ROCR remaps assigned GPU → device 0.
        rocm_env = " ".join(ROCM_ENV_VARS + [
            f"ROCR_VISIBLE_DEVICES={gpu_id}",
            f"CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_{gpu_id}",  # [S157] per-worker isolated cache
        ])

        activate = f"source {os.path.join(os.path.dirname(node.python_env), 'activate')}"
        cmd_body = (
            f"cd {node.script_path} && "
            f"{activate} && "
            f"env {rocm_env} {node.python_env} -u {WORKER_SCRIPT} --gpu-id {gpu_id} --persistent"
        )
        ssh_cmd = [
            "ssh",
            "-q",                              # suppress SSH banners/warnings
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=10",    # [S152] probe 10× before giving up = 300s tolerance
            "-o", "ConnectTimeout=10",         # [S152] fail fast on dead hosts
            f"{node.username}@{node.hostname}" if node.username else node.hostname,
            cmd_body
        ]

        try:
            proc = subprocess.Popen(
                ssh_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,  # unbuffered binary — readline handles framing
            )
            # Drain lines until we see the worker's {"status": "ready"} heartbeat.
            # Shell activation (conda/venv banners) may emit non-JSON lines first.
            deadline = time.time() + WORKER_HEARTBEAT_TIMEOUT_S
            ready = False
            while time.time() < deadline:
                line = _read_with_timeout(proc.stdout, 5.0)
                if line is None:
                    break
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                if '"status"' in line and '"ready"' in line:
                    ready = True
                    break
            if not ready:
                proc.kill()
                self.logger.error(f"Spawn failed {node.hostname}:GPU{gpu_id}: no heartbeat within {WORKER_HEARTBEAT_TIMEOUT_S}s")
                return False
            handle.proc  = proc
            handle.alive = True
            return True
        except Exception as e:
            self.logger.error(f"Spawn failed {node.hostname}:GPU{gpu_id}: {e}")
            return False

    def _tcp_launch_workers(self) -> None:
        """
        S161: SSH-launch pwc_worker_service on each AMD rig GPU via nohup.
        TB Gate 1 requirements:
          - TCP server already bound before this is called
          - Kill stale pwc_worker_service processes first
          - nohup launch per GPU with per-GPU log file
          - ROCm env vars same as _spawn_worker()
          - 180s connect deadline starts AFTER last launch issued
        """
        import subprocess as _sp
        import time as _t

        total_launched = 0

        for node in self.nodes:
            if self._is_localhost(node.hostname):
                continue
            if not self._is_rocm(node):
                continue

            host = node.hostname
            user = node.username
            pool = min(self.worker_pool_size, node.gpu_count)
            ssh_base = ["ssh", "-q",
                        "-o", "StrictHostKeyChecking=no",
                        "-o", "BatchMode=yes",
                        "-o", "ConnectTimeout=10",
                        user + "@" + host]
            activate_path = node.python_env.replace("/bin/python", "/bin/activate")

            # Step 1: Kill stale pwc_worker_service processes
            try:
                _sp.run(
                    ssh_base + ["pkill -9 -f pwc_worker_service 2>/dev/null; echo killed"],
                    capture_output=True, timeout=10
                )
                self.logger.info("[PWC-TCP] " + host + ": stale workers killed")
            except Exception as e:
                self.logger.warning("[PWC-TCP] " + host + ": kill failed: " + str(e))

            # Step 2: Launch one worker per GPU via temp bash script

            for gpu_id in range(pool):
                rocm_env = " ".join(ROCM_ENV_VARS + [
                    f"ROCR_VISIBLE_DEVICES={gpu_id}",
                    f"CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_{gpu_id}",
                ])
                log_file = f"/tmp/pwc_tcp_worker_{host.replace('.', '_')}_gpu{gpu_id}.log"
                worker_id = f"{host.replace('.', '_')}_gpu{gpu_id}"

                script    = f"/tmp/pwc_tcp_launch_gpu{gpu_id}.sh"

                # Build script content — no quoting issues since it is a file
                script_lines = [
                    "#!/bin/bash",
                    "source " + activate_path,
                    "cd " + node.script_path,
                ]
                for var in ROCM_ENV_VARS:
                    script_lines.append("export " + var)
                script_lines += [
                    "export ROCR_VISIBLE_DEVICES=" + str(gpu_id),
                    "export CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_" + str(gpu_id),
                    "export PYTHONPATH=.",
                    (
                        "nohup " + node.python_env + " -m persistent.pwc_worker_service "
                        "--host 192.168.3.127 --port " + str(self.pwc_port) + " "
                        "--gpu-id " + str(gpu_id) + " --worker-id " + worker_id + " "
                        ">> " + log_file + " 2>&1 &"
                    ),
                    "echo PID=$!",
                ]
                script_content = chr(10).join(script_lines)

                try:
                    # Write script to rig via stdin pipe
                    write_r = _sp.run(
                        ssh_base + ["cat > " + script + " && chmod +x " + script],
                        input=script_content.encode(),
                        capture_output=True, timeout=10
                    )
                    if write_r.returncode != 0:
                        self.logger.error("[PWC-TCP] " + host + ":GPU" + str(gpu_id) + " script write failed")
                        continue

                    # Execute the script
                    exec_r = _sp.run(
                        ssh_base + ["bash " + script],
                        capture_output=True, text=True, timeout=10
                    )
                    _launch_time = _t.time()  # S161: start deadline from actual launch
                    pid_line = exec_r.stdout.strip()
                    self.logger.info(
                        "[PWC-TCP] " + host + ":GPU" + str(gpu_id) +
                        " launched — " + pid_line + " log=" + log_file
                    )
                    total_launched += 1
                except Exception as e:
                    self.logger.error("[PWC-TCP] " + host + ":GPU" + str(gpu_id) + " launch failed: " + str(e))

                # S161 v2: no per-worker wait — workers connect fast (no ROCm at startup)
                # All workers launched with 1s stagger, then _tcp_wait_online() handles
                # the online barrier before broadcasting init.
                self.logger.info(
                    "[PWC-TCP] " + host + ":GPU" + str(gpu_id) + " launched"
                )

        self.logger.info(
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
        )

    def _tcp_wait_online(self, expected: int, timeout_s: float = 30.0) -> int:
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
        """Check worker still alive; respawn if dead."""
        if handle.quarantined:
            return False
        if handle.proc is None or handle.proc.poll() is not None:
            self.logger.warning(f"Worker {handle.node.hostname}:GPU{handle.gpu_id} dead — respawning")
            handle.alive = False
            # [S151] Per-node respawn lock + stagger — prevents SSH hammer when multiple
            # workers die simultaneously on the same rig (PCIe 1x crypto-miner constraint)
            respawn_lock = self._node_respawn_locks.get(handle.node.hostname)
            if respawn_lock:
                with respawn_lock:
                    time.sleep(ROCM_SPAWN_STAGGER_S)
                    success = self._spawn_worker(handle)
            else:
                success = self._spawn_worker(handle)
            if not success:
                handle.quarantined = True
                self.logger.error(f"Respawn failed — {handle.node.hostname}:GPU{handle.gpu_id} quarantined")
            return success
        return True

    def shutdown(self):
        """Send shutdown to all workers and reap processes."""
        self.logger.info("Shutting down persistent workers...")
        if hasattr(self, '_progress_writer') and self._progress_writer:
            try:
                self._progress_writer.finish()
            except Exception:
                pass
        for handle in self.workers:
            if handle.proc and handle.proc.poll() is None:
                try:
                    handle.proc.stdin.write((json.dumps({"cmd": "shutdown"}) + "\n").encode())
                    handle.proc.stdin.flush()
                    handle.proc.wait(timeout=10)
                except Exception:
                    handle.proc.kill()
            handle.alive = False
        # [TCP transport] stop TCP server if running — additive
        if self._tcp_transport is not None:
            try:
                self._tcp_transport.stop()
                self.logger.info("[PWC-TCP] transport stopped")
            except Exception as _e:
                self.logger.warning(f"[PWC-TCP] transport stop error: {_e}")
            self._tcp_transport = None

        self.logger.info("All workers shut down")

    # ─────────────────────────────────────────────────────────────────────────
    # Job dispatch
    # ─────────────────────────────────────────────────────────────────────────
    def _dispatch_to_worker(self, handle: WorkerHandle, job: Dict[str, Any]) -> Dict[str, Any]:
        """Send job JSON to worker stdin, read result from stdout."""
        if not self._ensure_worker_alive(handle):
            return {"status": "error", "message": f"Worker {handle.key} unavailable"}
        # Per-worker lock — each worker is a single process; only one job at a time
        with handle.dispatch_lock:
            sem = self._node_semaphores.get(handle.node.hostname)
            if sem:
                sem.acquire()
            try:
                line = (json.dumps(job) + "\n").encode()
                handle.proc.stdin.write(line)
                handle.proc.stdin.flush()
                result_line = _read_with_timeout(handle.proc.stdout, JOB_TIMEOUT_S)
                if result_line is None:
                    handle.alive = False
                    return {"status": "error", "message": "Worker timeout"}
                if isinstance(result_line, bytes):
                    result_line = result_line.decode("utf-8", errors="replace")
                if not result_line.strip():
                    handle.alive = False
                    return {"status": "error", "message": "Worker returned empty response (pipe closed)"}
                result = json.loads(result_line.strip())
                if result.get("status") == "ok":
                    handle.jobs_completed += 1
                    inner = result.get("result", {})
                    # [S150-slim_v1] Accept both slim parallel-array and legacy dict-list
                    if inner.get("format") == "slim_v1":
                        # Fast path — parallel arrays (TB approved Option A)
                        survivors   = [int(s) for s in inner.get("seeds", [])]
                        match_rates = list(inner.get("match_rates", []))
                        n = len(survivors)
                        # [S152-IPC] Confirm slim_v1 fast path active
                        _gpu_tag = f"{job.get('hostname','?')}:GPU{job.get('gpu_id','?')}"
                        self.logger.debug(f"[slim_v1] {_gpu_tag} chunk → {n} survivors")
                        # TB ruling: strategy_ids+skip_sequences required for hybrid
                        # [S155] Fix: job payload sends "hybrid": bool — not "prng_type"/"skip_mode".
                        # Both those keys are absent from PWC job dicts. "hybrid" is the correct key,
                        # set at dispatch: "hybrid": is_hybrid (from "'_hybrid' in prng_type").
                        _is_hybrid_job = bool(job.get("hybrid", False))
                        if _is_hybrid_job:
                            if "strategy_ids" not in inner or "skip_sequences" not in inner:
                                handle.alive = False
                                return {"status": "error", "message": "slim_v1 hybrid payload missing strategy_ids/skip_sequences"}
                            strat_ids = list(inner["strategy_ids"])
                            skip_seqs = list(inner["skip_sequences"])
                        else:
                            strat_ids = [0] * n
                            skip_seqs = [[] for _ in survivors]
                        # TB guardrail: all arrays must match len(seeds)
                        if len(match_rates) != n or len(strat_ids) != n or len(skip_seqs) != n:
                            handle.alive = False
                            return {"status": "error", "message": f"slim_v1 length mismatch: seeds={n} match_rates={len(match_rates)} strat_ids={len(strat_ids)} skip_seqs={len(skip_seqs)}"}
                    else:
                        # Legacy path — list of dicts (kept for rollout safety)
                        # [S152-IPC] WARN: expected slim_v1 from updated workers
                        _gpu_tag = f"{job.get('hostname','?')}:GPU{job.get('gpu_id','?')}"
                        self.logger.warning(f"[legacy-ipc] {_gpu_tag} — worker sent legacy dict-list format (expected slim_v1)")
                        raw_survivors = inner.get("survivors", [])
                        survivors   = [s["seed"]       if isinstance(s, dict) else int(s) for s in raw_survivors]
                        match_rates = [s["match_rate"] if isinstance(s, dict) else 0.5     for s in raw_survivors]
                        skip_seqs   = [s.get("skip_sequence", []) if isinstance(s, dict) else [] for s in raw_survivors]
                        strat_ids   = [s.get("strategy_id",    0) if isinstance(s, dict) else 0  for s in raw_survivors]
                    return {
                        "status":         "ok",
                        "job_id":         result.get("job_id", job.get("job_id")),
                        "survivors":      survivors,
                        "match_rates":    match_rates,
                        "skip_sequences": skip_seqs,
                        "strategy_ids":   strat_ids,
                    }
                else:
                    handle.jobs_failed += 1
                return result
            except Exception as e:
                handle.alive = False
                return {"status": "error", "message": str(e)}
            finally:
                if sem:
                    sem.release()

    def _dispatch_to_tcp(self, job: Dict[str, Any]) -> Dict[str, Any]:
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

        # S161: wait for at least one TCP worker to connect before dispatching
        _wait_start = time.time()
        while self._tcp_transport.worker_count() == 0:
            if time.time() - _wait_start > 60.0:
                return {"status": "error", "message": "TCP worker connect timeout (60s)"}
            time.sleep(0.5)
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
            # S161 dashboard: preserve worker identity for ProgressWriter
            "worker_id":      result.get("worker_id", ""),
            "hostname":       result.get("hostname", ""),
            "gpu_id":         result.get("gpu_id", 0),
        }

    def _dispatch_local_sieve(self, job: Dict[str, Any], node: WorkerNode) -> Dict[str, Any]:
        """
        Zeus local path — run sieve_filter.py as subprocess (already fast, no persistent
        worker needed). Mirrors coordinator.execute_local_job() behavior.
        """
        import tempfile
        payload_file = tempfile.mktemp(suffix=".json", dir=node.script_path)
        result_file  = payload_file.replace(".json", "_result.json")
        try:
            with open(payload_file, "w") as f:
                json.dump(job, f)
            cmd = [
                node.python_env, "-u", "sieve_filter.py",
                "--job-file", os.path.basename(payload_file),
                "--gpu-id", str(job.get("gpu_id", 0)),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(job.get("gpu_id", 0))
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=JOB_TIMEOUT_S,
                cwd=node.script_path, env=env
            )
            # Parse JSON from stdout
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.startswith("{"):
                    try:
                        raw = json.loads(line)
                        # sieve_filter.py emits {"success": true, "survivors": [...]}
                        # Normalize to flat format (same as _dispatch_to_worker)
                        is_ok = (raw.get("status") == "ok") or (raw.get("success") is True)
                        if is_ok:
                            inner    = raw.get("result", raw)
                            raw_surv = inner.get("survivors", [])
                            if raw_surv and isinstance(raw_surv[0], dict):
                                survivors   = [s["seed"]                  for s in raw_surv]
                                match_rates = [s.get("match_rate", 0.5)   for s in raw_surv]
                                skip_seqs   = [s.get("skip_sequence", []) for s in raw_surv]
                                strat_ids   = [s.get("strategy_id", 0)    for s in raw_surv]
                            else:
                                survivors   = [int(s) for s in raw_surv]
                                match_rates = inner.get("match_rates", [0.5]*len(survivors))
                                skip_seqs   = inner.get("skip_sequences", [])
                                strat_ids   = inner.get("strategy_ids",   [])
                            return {
                                "status":         "ok",
                                "job_id":         raw.get("job_id", "local"),
                                "survivors":      survivors,
                                "match_rates":    match_rates,
                                "skip_sequences": skip_seqs,
                                "strategy_ids":   strat_ids,
                            }
                        return raw
                    except Exception:
                        pass
            return {"status": "error", "message": result.stderr[:500]}
        except subprocess.TimeoutExpired:
            return {"status": "error", "message": "Local sieve timeout"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
        finally:
            for f in (payload_file, result_file):
                try:
                    os.unlink(f)
                except Exception:
                    pass

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry point — drop-in for execute_distributed_analysis()
    # ─────────────────────────────────────────────────────────────────────────
    def run_sieve_pass(self,
                       prng_type: str,
                       residues: List[int],
                       total_seeds: int,
                       threshold: float,
                       window_size: int,
                       output_file: str,
                       dataset_path: str = "",
                       strategies: Optional[List[Dict]] = None,
                       phase2_threshold: float = 0.5,
                       target_file: str = "",
                       offset: int = 0,
                       sessions: Optional[List[str]] = None,
                       skip_range: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Run a full distributed sieve pass (forward OR reverse, constant OR hybrid).
        Returns a result dict compatible with extract_survivor_records() in
        window_optimizer_integration_final.py.

        Parameters match coordinator.execute_distributed_analysis() semantics:
          prng_type        — e.g. "java_lcg", "java_lcg_reverse", "java_lcg_hybrid"
          residues         — list of draw values for the window
          total_seeds      — total seed space to search
          threshold        — match threshold for survivor filtering
          window_size      — number of draws in window
          output_file      — path to write result JSON (mirrors existing paths)
          strategies       — None for constant skip; list of strategy dicts for hybrid
                             (auto-loaded from hybrid_strategy if None and prng is hybrid)
          phase2_threshold — hybrid second-phase threshold
        """
        is_hybrid = "_hybrid" in prng_type

        # Auto-load strategies for hybrid if not provided
        if is_hybrid and strategies is None:
            try:
                from hybrid_strategy import get_all_strategies
                raw = get_all_strategies()
                strategies = [
                    # Send full strategy dict — sieve_filter.py needs all StrategyConfig fields
                    s.to_dict() if hasattr(s, 'to_dict') else s
                    for s in raw
                ]
                self.logger.info(f"Loaded {len(strategies)} strategies for hybrid sieve")
            except ImportError:
                self.logger.warning("hybrid_strategy not available — hybrid will use default")
                strategies = []

        # Build chunk list — divide total seeds across all available workers
        # S161 v2: workers already online+ready from startup() three-phase init
        # Just confirm ready count before dispatch — no waiting needed here
        if self._tcp_transport is not None:
            _ready = self._tcp_transport.ready_count()
            if _ready == 0:
                self.logger.error("[PWC-TCP] no ready workers — aborting dispatch")
                return {"status": "error", "survivor_count": 0,
                        "survivors": [], "failed_chunks": 1, "total_chunks": 1}
            self.logger.info(f"[PWC-TCP] {_ready} ready worker(s) — dispatching")
        all_workers = self._get_available_workers()
        num_workers = max(1, len(all_workers))
        ideal_chunk = max(1, total_seeds // num_workers)

        # Cap chunk size at per-GPU OOM ceiling
        # Use AMD cap as conservative ceiling (all remote workers are AMD)
        chunk_cap   = self.seed_cap_amd
        chunk_size  = min(ideal_chunk, chunk_cap)

        chunks = []
        seed   = 0
        while seed < total_seeds:
            end = min(seed + chunk_size, total_seeds)
            chunks.append((seed, end))
            seed = end

        self.logger.info(
            f"[{prng_type}] {total_seeds:,} seeds → {len(chunks)} chunks "
            f"({chunk_size:,}/chunk) across {num_workers} workers"
        )
        if self._progress_writer:
            try:
                step_name = f"{'Reverse' if 'reverse' in prng_type else 'Forward'} Sieve ({prng_type})"
                self._progress_writer.update_step(step_name, total_seeds=total_seeds)
            except Exception:
                pass

        # Dispatch all chunks in parallel threads
        results_by_chunk: Dict[int, Dict] = {}
        lock = threading.Lock()

        def dispatch_chunk(idx: int, seed_start: int, seed_end: int,
                           worker_handle_or_node):
            # Job dict matches coordinator.py residue_sieve format exactly
            # (field names verified against sieve_gpu_worker.py job.get() calls)
            _sessions   = sessions   if sessions   is not None else ["midday", "evening"]
            _skip_range = skip_range if skip_range is not None else [0, 147]
            # prng_families: strip _reverse/_hybrid suffixes — worker handles variants
            _base_prng  = prng_type.replace("_reverse", "").replace("_hybrid", "")
            job = {
                "job_id":               f"sieve_{idx:03d}",
                "search_type":          "residue_sieve",
                "dataset_path":         dataset_path or target_file,
                "seed_start":           seed_start,
                "seed_end":             seed_end,
                "window_size":          window_size,
                "min_match_threshold":  threshold,
                "skip_range":           _skip_range,
                "offset":               offset,
                "sessions":             _sessions,
                "prng_families":        [prng_type],
                "strategies":           strategies if is_hybrid else None,
                "hybrid":               is_hybrid,
                "phase2_threshold":     phase2_threshold,
            }

            def _run_once(wh):
                # [TCP transport] dispatch bridge — additive, SSH path unchanged
                if self._tcp_transport is not None and not isinstance(wh, WorkerNode):
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
                        self._localhost_semaphore.release()

            t0 = time.time()
            res = _run_once(worker_handle_or_node)
            elapsed = time.time() - t0

            # One retry on transient pipe/empty-response failures
            if res.get("status") != "ok":
                err = res.get("message", "")
                if "empty response" in err or "pipe" in err.lower() or "timeout" in err.lower():
                    self.logger.warning(f"  Chunk {idx} transient failure ({err}) — retrying once")
                    time.sleep(1)
                    t0 = time.time()
                    res = _run_once(worker_handle_or_node)
                    elapsed = time.time() - t0

            with lock:
                results_by_chunk[idx] = res
            status = "✅" if res.get("status") == "ok" else "❌"
            survivors = len(res.get("survivors", []))
            seeds_in_chunk = seed_end - seed_start
            self.logger.info(f"  {status} Chunk {idx}: {seeds_in_chunk:,} seeds → {survivors:,} survivors")

            # Log to ProgressWriter for web dashboard (mirrors coordinator.py line 1611)
            if self._progress_writer and res.get("status") == "ok" and elapsed > 0:
                try:
                    if worker_handle_or_node is None:
                        # S161 v2: TCP worker — extract from result payload
                        _payload = res.get("result", {}).get("payload", res.get("result", {}))
                        gpu_id   = _payload.get("gpu_id", 0)
                        gpu_type = "RX 6600"
                        # S161 dashboard fix: extract IP from worker_id
                        # worker_id format: "192_168_3_120_gpu0" → "192.168.3.120"
                        # worker_id is in res["worker_id"] from normalizer
                        _worker_id = res.get("worker_id", "")
                        try:
                            # Strip _gpuN suffix, replace _ with . to get IP
                            _ip_part = _worker_id.rsplit("_gpu", 1)[0].replace("_", ".")
                            hostname = _ip_part if _ip_part else "tcp-worker"
                        except Exception:
                            hostname = "tcp-worker"
                        # Get correct gpu_type from matching node
                        for _n in self.nodes:
                            if _n.hostname == hostname:
                                gpu_type = _n.gpu_type
                                break
                    elif isinstance(worker_handle_or_node, WorkerHandle):
                        hostname = worker_handle_or_node.node.hostname
                        gpu_type = worker_handle_or_node.node.gpu_type
                        gpu_id   = worker_handle_or_node.gpu_id
                    else:
                        hostname = worker_handle_or_node.hostname
                        gpu_type = worker_handle_or_node.gpu_type
                        gpu_id   = 0
                    self._progress_writer.log_gpu_result(hostname, gpu_id, gpu_type, seeds_in_chunk, elapsed, success=True)
                except Exception:
                    pass

        # [S158B-v3] Bounded dispatch — TB-approved hardening.
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
                )

        # Aggregate results
        all_survivors   = []
        all_match_rates = []
        all_skip_seqs   = []
        all_strat_ids   = []
        failed_chunks   = 0

        for i in range(len(chunks)):
            res = results_by_chunk.get(i, {"status": "error"})
            if res.get("status") == "ok":
                all_survivors   .extend(res.get("survivors",    []))
                all_match_rates .extend(res.get("match_rates",  []))
                all_skip_seqs   .extend(res.get("skip_sequences", []))
                all_strat_ids   .extend(res.get("strategy_ids",  []))
            else:
                failed_chunks += 1
                err_msg = res.get('message') or res.get('error', 'unknown')
                tb      = res.get('traceback', '')
                self.logger.warning(f"Chunk {i} failed: {err_msg}")
                if tb:
                    self.logger.warning(f"  Worker traceback:\n{tb}")

        if failed_chunks:
            self.logger.warning(f"{failed_chunks}/{len(chunks)} chunks failed for {prng_type}")

        if self._progress_writer:
            try:
                self._progress_writer.update_progress(
                    jobs_done=len(chunks) - failed_chunks,
                    chunks_total=len(chunks)
                )
            except Exception:
                pass

        # Build result dict compatible with extract_survivor_records()
        result = {
            "survivors":         all_survivors,
            "match_rates":       all_match_rates,
            "skip_sequences":    all_skip_seqs,
            "strategy_ids":      all_strat_ids,
            "total_tested":      total_seeds,
            "survivor_count":    len(all_survivors),
            "prng_type":         prng_type,
            "threshold":         threshold,
            "failed_chunks":     failed_chunks,
            "total_chunks":      len(chunks),
        }

        # Save to output_file (mirrors coordinator behavior)
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        try:
            with open(output_file, "w") as f:
                json.dump(result, f)
            self.logger.info(f"Results saved: {output_file} ({len(all_survivors):,} survivors)")
        except Exception as e:
            self.logger.error(f"Failed to save {output_file}: {e}")

        return result

    def _get_available_workers(self) -> List:
        """
        Returns list of available dispatch targets — mix of WorkerHandle (AMD)
        and WorkerNode (Zeus local). Only alive, non-quarantined workers included.
        In TCP mode, derives count from connected TCP workers — TB fix 3.
        """
        # [TCP transport] use TCP worker count as pool size — TB fix 3
        if self._tcp_transport is not None:
            tcp_count = self._tcp_transport.ready_count()  # S161 v2: only ready workers
            # TCP workers as synthetic placeholders + Zeus local nodes
            available = [None] * tcp_count
            for node in self.nodes:
                if self._is_localhost(node.hostname):
                    available.append(node)
            return available

        available = []
        # AMD persistent workers
        for w in self.workers:
            if not w.quarantined and w.alive:
                available.append(w)
        # Zeus local nodes
        for node in self.nodes:
            if self._is_localhost(node.hostname):
                available.append(node)
        return available


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────
def _read_with_timeout(stream, timeout_s: float) -> Optional[str]:
    """Read one line from stream with timeout. Returns None on timeout."""
    result = [None]
    def _reader():
        try:
            result[0] = stream.readline()
        except Exception:
            pass
    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    return result[0] if result[0] else None


# ─────────────────────────────────────────────────────────────────────────────
# Integration shim for window_optimizer_integration_final.py
# ─────────────────────────────────────────────────────────────────────────────
def run_trial_persistent(coordinator_cfg: str,
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
                         pwc_port: int = 5600) -> Dict[str, Any]:
    """
    Shim called by window_optimizer_integration_final.py when use_persistent_workers=True.

    Runs all 4 sieve passes (forward, reverse, forward_hybrid, reverse_hybrid)
    using persistent workers and returns the same dict structure as the original
    run_trial() — bidirectional_count, bidirectional_constant, bidirectional_variable, etc.

    This function manages PersistentWorkerCoordinator lifecycle internally so that
    the caller (run_trial) doesn't need to know about workers at all.
    """
    pwc = PersistentWorkerCoordinator(
        config_file      = coordinator_cfg,
        worker_pool_size = worker_pool_size,
        seed_cap_nvidia  = seed_cap_nvidia,
        seed_cap_amd     = seed_cap_amd,
        pwc_transport    = pwc_transport,
        pwc_host         = pwc_host,
        pwc_port         = pwc_port,
    )
    pwc.startup()

    try:
        ws  = config.window_size
        off = config.offset

        # ── Pass 1: Forward constant skip ────────────────────────────────────
        print(f"\n    Running FORWARD sieve ({prng_base}) [CONSTANT SKIP] [PERSISTENT]...")
        fwd_result = pwc.run_sieve_pass(
            prng_type    = prng_base,
            residues     = residues,
            total_seeds  = total_seeds,
            threshold    = forward_threshold,
            window_size  = ws,
            dataset_path = dataset_path,
            output_file  = f"results/window_opt_forward_{ws}_{off}_t{trial_number}.json",
            offset       = config.offset,
            sessions     = list(config.sessions) if hasattr(config, 'sessions') else ["midday", "evening"],
            skip_range   = [config.skip_min, config.skip_max] if hasattr(config, 'skip_min') else [0, 147],
        )
        fwd_survivors   = fwd_result.get("survivors", [])
        fwd_match_rates = fwd_result.get("match_rates", [])
        fwd_map = dict(zip(fwd_survivors, fwd_match_rates))
        print(f"      Forward: {len(fwd_survivors):,} survivors")

        if not fwd_survivors:
            pwc.shutdown()
            return {
                "pruned": True,
                "reason": "forward_zero",
                "bidirectional_count": 0,
                "bidirectional_constant": set(),
                "bidirectional_variable": set(),
                "forward_records": [],
                "reverse_records": [],
            }

        # ── Pass 2: Reverse constant skip ────────────────────────────────────
        prng_reverse = prng_base + "_reverse"
        print(f"    Running REVERSE sieve ({prng_reverse}) [CONSTANT SKIP] [PERSISTENT]...")
        rev_result = pwc.run_sieve_pass(
            prng_type    = prng_reverse,
            residues     = residues,
            total_seeds  = total_seeds,
            threshold    = reverse_threshold,
            window_size  = ws,
            dataset_path = dataset_path,
            output_file  = f"results/window_opt_reverse_{ws}_{off}_t{trial_number}.json",
            offset       = config.offset,
            sessions     = list(config.sessions) if hasattr(config, 'sessions') else ["midday", "evening"],
            skip_range   = [config.skip_min, config.skip_max] if hasattr(config, 'skip_min') else [0, 147],
        )
        rev_survivors   = rev_result.get("survivors", [])
        rev_match_rates = rev_result.get("match_rates", [])
        rev_map = dict(zip(rev_survivors, rev_match_rates))
        print(f"      Reverse: {len(rev_survivors):,} survivors")

        bidirectional_constant = set(fwd_map.keys()) & set(rev_map.keys())
        print(f"      ✨ Bidirectional (constant): {len(bidirectional_constant):,} survivors")

        # ── Passes 3+4: Variable skip (hybrid) ───────────────────────────────
        bidirectional_variable = set()
        fwd_records_hybrid = []
        rev_records_hybrid = []

        if test_both_modes and not prng_base.endswith("_hybrid"):
            prng_hybrid = f"{prng_base}_hybrid"
            prng_hybrid_rev = f"{prng_hybrid}_reverse"

            # [S147 Q2] Single strategy for full-range scan — 5x work reduction
            # TB ruling: balanced_hybrid for discovery, all 5 for refinement only
            # Uses pwc.logger — run_trial_persistent is a standalone function, not a method
            try:
                from hybrid_strategy import get_strategy as _get_strategy
                _s = _get_strategy("balanced_hybrid")
                _hybrid_strategies = [_s.to_dict() if hasattr(_s, "to_dict") else vars(_s)]
            except Exception as _e:
                pwc.logger.warning(f"Q2: could not load balanced_hybrid ({_e}) — using all strategies")
                _hybrid_strategies = None  # fallback: auto-load all in run_sieve_pass

            print(f"    Running FORWARD sieve ({prng_hybrid}) [VARIABLE SKIP] [PERSISTENT]...")
            fwd_h_result = pwc.run_sieve_pass(
                prng_type    = prng_hybrid,
                residues     = residues,
                total_seeds  = total_seeds,
                threshold    = forward_threshold,
                window_size  = ws,
                dataset_path = dataset_path,
                output_file  = f"results/window_opt_forward_hybrid_{ws}_{off}_t{trial_number}.json",
                offset       = config.offset,
                sessions     = list(config.sessions) if hasattr(config, 'sessions') else ["midday", "evening"],
                skip_range   = [config.skip_min, config.skip_max] if hasattr(config, 'skip_min') else [0, 147],
                strategies   = _hybrid_strategies,  # [S147 Q2] single strategy
            )
            fwd_h_survivors   = fwd_h_result.get("survivors", [])
            fwd_h_match_rates = fwd_h_result.get("match_rates", [])
            fwd_h_map = dict(zip(fwd_h_survivors, fwd_h_match_rates))
            print(f"      Forward (variable): {len(fwd_h_survivors):,} survivors")

            # [S147 Q0] Gate: skip hybrid reverse if hybrid forward = 0
            # Mirrors constant-skip B1 gate. SKIP not prune — constant results preserved.
            if not fwd_h_survivors:
                print(f"      Hybrid forward zero survivors — skipping hybrid reverse (Q0 gate)")
                rev_h_survivors   = []
                rev_h_match_rates = []
                rev_h_map         = {}
            else:
                print(f"    Running REVERSE sieve ({prng_hybrid_rev}) [VARIABLE SKIP] [PERSISTENT]...")
                rev_h_result = pwc.run_sieve_pass(
                    prng_type    = prng_hybrid_rev,
                    residues     = residues,
                    total_seeds  = total_seeds,
                    threshold    = reverse_threshold,
                    window_size  = ws,
                    dataset_path = dataset_path,
                    output_file  = f"results/window_opt_reverse_hybrid_{ws}_{off}_t{trial_number}.json",
                    offset       = config.offset,
                    sessions     = list(config.sessions) if hasattr(config, 'sessions') else ["midday", "evening"],
                    skip_range   = [config.skip_min, config.skip_max] if hasattr(config, 'skip_min') else [0, 147],
                    strategies   = _hybrid_strategies,  # [S147 Q2] single strategy
                )
                rev_h_survivors   = rev_h_result.get("survivors", [])
                rev_h_match_rates = rev_h_result.get("match_rates", [])
                rev_h_map = dict(zip(rev_h_survivors, rev_h_match_rates))
                print(f"      Reverse (variable): {len(rev_h_survivors):,} survivors")

            bidirectional_variable = set(fwd_h_map.keys()) & set(rev_h_map.keys())
            print(f"      ✨ Bidirectional (variable): {len(bidirectional_variable):,} survivors")

            fwd_records_hybrid = [{"seed": s, "match_rate": fwd_h_map[s]} for s in fwd_h_survivors]
            rev_records_hybrid = [{"seed": s, "match_rate": rev_h_map[s]} for s in rev_h_survivors]

        total_bidi = len(bidirectional_constant) + len(bidirectional_variable)
        print(f"      📊 Total bidirectional: {total_bidi:,}")

        # Update dashboard with trial survivor stats (mirrors coordinator._progress_writer call
        # in window_optimizer_integration_final.py line 373)
        if pwc._progress_writer:
            try:
                pwc._progress_writer.update_trial_stats(
                    trial_num=trial_number,
                    forward_survivors=len(fwd_survivors),
                    reverse_survivors=len(rev_map),
                    bidirectional=len(bidirectional_constant),
                    best_bidirectional=len(bidirectional_constant),
                    config_desc=f"W{config.window_size}_O{config.offset}",
                    accumulated_forward=len(fwd_survivors),
                    accumulated_reverse=len(rev_map),
                    accumulated_bidirectional=total_bidi,
                )
            except Exception:
                pass

        return {
            "pruned":                 False,
            "bidirectional_count":    total_bidi,
            "bidirectional_constant": bidirectional_constant,
            "bidirectional_variable": bidirectional_variable,
            "forward_map":            fwd_map,
            "reverse_map":            rev_map,
            "forward_records":        [{"seed": s, "match_rate": fwd_map[s]} for s in fwd_survivors],
            "reverse_records":        [{"seed": s, "match_rate": rev_map[s]} for s in rev_survivors],
            "forward_records_hybrid": fwd_records_hybrid,
            "reverse_records_hybrid": rev_records_hybrid,
        }

    finally:
        pwc.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# CLI — smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Persistent Worker Coordinator — smoke test")
    p.add_argument("--config",       default="distributed_config.json")
    p.add_argument("--pool-size",    type=int, default=2)
    p.add_argument("--total-seeds",  type=int, default=500_000)
    p.add_argument("--prng-type",    default="java_lcg")
    p.add_argument("--pwc-transport", default="ssh",
                   choices=["ssh", "tcp"],
                   help="Transport backend: ssh (default/legacy) or tcp (new)")
    p.add_argument("--pwc-host",      default="0.0.0.0",
                   help="TCP server bind host (tcp mode only)")
    p.add_argument("--pwc-port",      type=int, default=5600,
                   help="TCP server port (tcp mode only)")
    p.add_argument("--min-workers",   type=int, default=1,
                   help="[TCP] minimum connected workers before dispatch (default=1)")
    p.add_argument("--startup-only", action="store_true",
                   help="Just spawn workers and report alive count, then shutdown")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    pwc = PersistentWorkerCoordinator(
        config_file      = args.config,
        worker_pool_size = args.pool_size,
        pwc_transport    = args.pwc_transport,
        pwc_host         = args.pwc_host,
        pwc_port         = args.pwc_port,
        min_workers      = args.min_workers,
    )
    pwc.startup()

    if args.startup_only:
        alive = sum(1 for w in pwc.workers if w.alive)
        print(f"\nAlive workers: {alive}/{len(pwc.workers)}")
        pwc.shutdown()
        sys.exit(0)

    # Minimal smoke test sieve pass
    residues = [0, 1, 2, 3, 4, 5, 6, 7]   # placeholder
    result = pwc.run_sieve_pass(
        prng_type    = args.prng_type,
        residues     = residues,
        total_seeds  = args.total_seeds,
        threshold    = 0.30,
        window_size  = 8,
        dataset_path = "daily3.json",
        output_file  = "/tmp/pwc_smoke_test.json",
    )
    print(f"\nSmoke test result: {result.get('survivor_count', 0)} survivors")
    pwc.shutdown()
