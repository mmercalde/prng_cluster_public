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
    threshold=0.25,
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
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# ROCm stability constants (from S130/S133 learnings)
# ─────────────────────────────────────────────────────────────────────────────
ROCM_SPAWN_STAGGER_S   = 4.0   # seconds between worker spawns per gpu_id
ROCM_ENV_VARS = [
    "HSA_OVERRIDE_GFX_VERSION=10.3.0",
    "HSA_ENABLE_SDMA=0",
    "ROCM_PATH=/opt/rocm",
    "HIP_PATH=/opt/rocm/hip",
    "LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/hip/lib:${LD_LIBRARY_PATH}",
    "PATH=/opt/rocm/bin:${PATH}",
    "CUPY_CACHE_DIR=${HOME}/.cache/cupy",
]
WORKER_SCRIPT = "sieve_gpu_worker.py"
WORKER_HEARTBEAT_TIMEOUT_S = 30   # seconds to wait for worker heartbeat on startup
JOB_TIMEOUT_S = 300               # seconds max per sieve job


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
                 max_per_node: int = 8):
        self.config_file     = config_file
        self.worker_pool_size = worker_pool_size
        self.seed_cap_nvidia = seed_cap_nvidia
        self.seed_cap_amd    = seed_cap_amd
        self.max_per_node    = max_per_node

        self.nodes: List[WorkerNode] = []
        self.workers: List[WorkerHandle] = []   # AMD rig workers (persistent)
        self._lock = threading.Lock()
        self._started = False
        # Per-node semaphores — throttle concurrent dispatch to max_per_node (S133-A lesson)
        self._node_semaphores: Dict[str, threading.Semaphore] = {}

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

    def _spawn_worker(self, handle: WorkerHandle) -> bool:
        """SSH + launch sieve_gpu_worker.py on remote GPU, confirm heartbeat."""
        node   = handle.node
        gpu_id = handle.gpu_id

        rocm_env = " ".join(ROCM_ENV_VARS + [
            f"CUDA_VISIBLE_DEVICES={gpu_id}",
            f"HIP_VISIBLE_DEVICES={gpu_id}",
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

    def _ensure_worker_alive(self, handle: WorkerHandle) -> bool:
        """Check worker still alive; respawn if dead."""
        if handle.quarantined:
            return False
        if handle.proc is None or handle.proc.poll() is not None:
            self.logger.warning(f"Worker {handle.node.hostname}:GPU{handle.gpu_id} dead — respawning")
            handle.alive = False
            success = self._spawn_worker(handle)
            if not success:
                handle.quarantined = True
                self.logger.error(f"Respawn failed — {handle.node.hostname}:GPU{handle.gpu_id} quarantined")
            return success
        return True

    def shutdown(self):
        """Send shutdown to all workers and reap processes."""
        self.logger.info("Shutting down persistent workers...")
        for handle in self.workers:
            if handle.proc and handle.proc.poll() is None:
                try:
                    handle.proc.stdin.write((json.dumps({"cmd": "shutdown"}) + "\n").encode())
                    handle.proc.stdin.flush()
                    handle.proc.wait(timeout=10)
                except Exception:
                    handle.proc.kill()
            handle.alive = False
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
                       target_file: str = "") -> Dict[str, Any]:
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
                    {"max_consecutive_misses": s.max_consecutive_misses,
                     "skip_tolerance": s.skip_tolerance}
                    if not isinstance(s, dict) else s
                    for s in raw
                ]
                self.logger.info(f"Loaded {len(strategies)} strategies for hybrid sieve")
            except ImportError:
                self.logger.warning("hybrid_strategy not available — hybrid will use default")
                strategies = []

        # Build chunk list — divide total seeds across all available workers
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

        # Dispatch all chunks in parallel threads
        results_by_chunk: Dict[int, Dict] = {}
        lock = threading.Lock()

        def dispatch_chunk(idx: int, seed_start: int, seed_end: int,
                           worker_handle_or_node):
            job = {
                "job_id":            f"sieve_{idx:03d}",
                "prng_type":         prng_type,
                "search_type":       "residue_sieve",
                "seed_start":        seed_start,
                "seed_end":          seed_end,
                "residues":          residues,
                "window_size":       window_size,
                "threshold":         threshold,
                "phase2_threshold":  phase2_threshold,
                "strategies":        strategies if is_hybrid else None,
                "hybrid":            is_hybrid,
                "target_file":       target_file,
                "dataset_path":      dataset_path,
            }

            def _run_once(wh):
                if isinstance(wh, WorkerHandle):
                    job["gpu_id"] = wh.gpu_id
                    return self._dispatch_to_worker(wh, job)
                else:
                    job["gpu_id"] = idx % wh.gpu_count
                    return self._dispatch_local_sieve(job, wh)

            res = _run_once(worker_handle_or_node)

            # One retry on transient pipe/empty-response failures
            if res.get("status") != "ok":
                err = res.get("message", "")
                if "empty response" in err or "pipe" in err.lower() or "timeout" in err.lower():
                    self.logger.warning(f"  Chunk {idx} transient failure ({err}) — retrying once")
                    import time; time.sleep(1)
                    res = _run_once(worker_handle_or_node)

            with lock:
                results_by_chunk[idx] = res
            status = "✅" if res.get("status") == "ok" else "❌"
            survivors = len(res.get("survivors", []))
            self.logger.info(f"  {status} Chunk {idx}: {seed_end - seed_start:,} seeds → {survivors:,} survivors")

        threads = []
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
            t.join()

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
                         seed_cap_amd:   int  = 2_000_000) -> Dict[str, Any]:
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

            print(f"    Running FORWARD sieve ({prng_hybrid}) [VARIABLE SKIP] [PERSISTENT]...")
            fwd_h_result = pwc.run_sieve_pass(
                prng_type    = prng_hybrid,
                residues     = residues,
                total_seeds  = total_seeds,
                threshold    = forward_threshold,
                window_size  = ws,
                dataset_path = dataset_path,
                output_file  = f"results/window_opt_forward_hybrid_{ws}_{off}_t{trial_number}.json",
            )
            fwd_h_survivors   = fwd_h_result.get("survivors", [])
            fwd_h_match_rates = fwd_h_result.get("match_rates", [])
            fwd_h_map = dict(zip(fwd_h_survivors, fwd_h_match_rates))
            print(f"      Forward (variable): {len(fwd_h_survivors):,} survivors")

            print(f"    Running REVERSE sieve ({prng_hybrid_rev}) [VARIABLE SKIP] [PERSISTENT]...")
            rev_h_result = pwc.run_sieve_pass(
                prng_type    = prng_hybrid_rev,
                residues     = residues,
                total_seeds  = total_seeds,
                threshold    = reverse_threshold,
                window_size  = ws,
                dataset_path = dataset_path,
                output_file  = f"results/window_opt_reverse_hybrid_{ws}_{off}_t{trial_number}.json",
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
    p.add_argument("--startup-only", action="store_true",
                   help="Just spawn workers and report alive count, then shutdown")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    pwc = PersistentWorkerCoordinator(
        config_file      = args.config,
        worker_pool_size = args.pool_size,
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
        prng_type   = args.prng_type,
        residues    = residues,
        total_seeds = args.total_seeds,
        threshold   = 0.25,
        window_size = 8,
        output_file = "/tmp/pwc_smoke_test.json",
    )
    print(f"\nSmoke test result: {result.get('survivor_count', 0)} survivors")
    pwc.shutdown()
