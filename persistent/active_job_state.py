"""
S173 Fault Attribution Instrumentation — Active Job State

Standalone module providing the active-worker JSON file that captures the
state of each worker during chunk execution. Read by crash_forensic_daemon
on ser8 after a fault to attribute the failure to a specific worker, GPU,
chunk, and configuration.

Design notes (per TB review of S173 instrumentation plan):
  - Atomic write via tmp + os.replace (never partial reads)
  - Both monotonic_time_ns and wall_time captured (clock-skew resistant)
  - prior_completed_chunks counter (tests state-drift hypothesis)
  - recent_elapsed_ms rolling window of 10 (catches latency drift before fault)
  - last_successful_chunk snapshot (boundary between valid and divergent state)
  - GPU-to-PCI-bus mapping (PCI bus from netconsole → gpu_id correlation)
  - First-fault-only design: one file per GPU, overwritten each chunk

Files written:
  /tmp/prng_active_worker_gpu{N}.json  - latest state per GPU
  /tmp/prng_gpu_bus_map.json           - one-time at worker startup, stable

Both files MUST be readable by other users on the rig (worker logs).
"""

import json
import os
import socket
import subprocess
import sys
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple


# Output paths — both /tmp because they reset on reboot which is desired.
_ACTIVE_FILE_FMT = "/tmp/prng_active_worker_gpu{gpu_id}.json"
_GPU_BUS_MAP_FILE = "/tmp/prng_gpu_bus_map.json"

# Rolling latency window size (TB spec: 10)
_RECENT_LATENCY_MAXLEN = 10


def _atomic_write_json(path: str, payload: dict) -> None:
    """
    Write JSON atomically: write to <path>.tmp, fsync, os.replace().
    Mirrors the pattern in pwc_worker_service.py._atomic_write_json so the
    daemon can rely on never seeing a partial file. NEVER raises.
    """
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        # Best-effort world-readable so daemon-pulled logs work
        try:
            os.chmod(path, 0o644)
        except Exception:
            pass
    except Exception:
        # Instrumentation must NEVER break the worker
        pass


def _query_pci_bus(gpu_id: int) -> Optional[str]:
    """
    Resolve PCI bus address for a given GPU index via rocm-smi.

    Returns canonical 'XXXX:XX:XX.X' format on success, None on any failure.

    rocm-smi --showbus -d <gpu_id> output looks like:
      GPU[0]    : PCI Bus: 0000:03:00.0
    """
    try:
        r = subprocess.run(
            ["rocm-smi", "--showbus", "-d", str(gpu_id)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode != 0:
            return None
        for line in r.stdout.splitlines():
            if "PCI Bus:" in line:
                # last whitespace-separated token is the address
                bus = line.strip().split()[-1].lower()
                # Sanity: should contain colons and dot
                if ":" in bus and "." in bus:
                    return bus
        return None
    except Exception:
        return None


def write_gpu_bus_map(gpu_id: int, worker_pid: int) -> None:
    """
    Write the PCI-bus map file for THIS worker (one entry per worker process).
    Called once at worker startup, after _setup_env, before the main loop.

    Each worker writes its own file path key under a JSON object. To avoid
    races between concurrent workers writing the same shared file, each
    worker writes a SEPARATE file:
      /tmp/prng_gpu_bus_map_gpu{gpu_id}.json

    The daemon globs /tmp/prng_gpu_bus_map_gpu*.json to assemble the map.
    """
    pci_bus = _query_pci_bus(gpu_id)
    payload = {
        "host": socket.gethostname(),
        "worker_pid": worker_pid,
        "gpu_id": gpu_id,
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES", ""),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "pci_bus": pci_bus,
        "captured_at_monotonic_ns": time.monotonic_ns(),
        "captured_at_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
    }
    path = "/tmp/prng_gpu_bus_map_gpu{}.json".format(gpu_id)
    _atomic_write_json(path, payload)


# ---------------------------------------------------------------------------
# ActiveJobState: per-worker rolling state holder
# ---------------------------------------------------------------------------

class ActiveJobState:
    """
    Per-worker rolling state for S173 fault attribution.

    Held as an instance member of PWCWorkerService. Lifecycle:
      1. __init__ at worker startup — captures worker_start_monotonic/wall
      2. on_chunk_start(job) — writes active-worker JSON with current chunk info
      3. on_chunk_end(elapsed_ms, success) — updates rolling stats + last_successful

    All file writes are atomic and never raise.
    """

    def __init__(self, worker_id: str, gpu_id: int, host: str) -> None:
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.host = host
        self.pid = os.getpid()
        self.active_path = _ACTIVE_FILE_FMT.format(gpu_id=gpu_id)

        # Worker-startup baseline (TB enhancement #2 — clock skew fallback)
        self._start_monotonic_ns: int = time.monotonic_ns()
        self._start_wall: str = time.strftime(
            "%Y-%m-%dT%H:%M:%S%z", time.localtime()
        )

        # Counters
        self._chunks_completed: int = 0

        # Rolling latency window (TB enhancement #5)
        self._recent_elapsed_ms: Deque[int] = deque(maxlen=_RECENT_LATENCY_MAXLEN)

        # Continuous-runtime tracker (TB enhancement #6)
        self._last_idle_monotonic_ns: int = self._start_monotonic_ns

        # Last successful chunk snapshot (TB enhancement #4)
        self._last_successful_chunk: Optional[Dict[str, Any]] = None

        # [TB Hardening] Cache the full job dict from on_chunk_start so
        # on_chunk_end can merge over a sparse follow-up dict.
        # Without this, a sparse on_chunk_end() call would overwrite the
        # rich pre-kernel state with mostly-null fields — which would make
        # the crash manifest useless if the fault hits during cleanup.
        self._cached_start_job: Dict[str, Any] = {}

        # Current chunk start time (set in on_chunk_start)
        self._current_chunk_started_monotonic_ns: Optional[int] = None
        self._current_chunk_started_wall: Optional[str] = None

    # -------------------------------------------------------------------
    # Public API — called from PWCWorkerService._execute_job
    # -------------------------------------------------------------------

    def on_chunk_start(self, job: Dict[str, Any]) -> None:
        """
        Called immediately before kernel launch for chunk dispatch.
        Writes the active-worker JSON capturing all S173 attribution fields.

        Caches the job dict so on_chunk_end can merge over any sparse
        follow-up dict it might receive.
        """
        now_monotonic = time.monotonic_ns()
        now_wall = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())

        self._current_chunk_started_monotonic_ns = now_monotonic
        self._current_chunk_started_wall = now_wall

        # [TB Hardening] Cache the full job dict for merge-fallback in on_chunk_end.
        # We copy explicitly — never store a reference, in case the caller
        # mutates the dict between calls.
        try:
            self._cached_start_job = dict(job) if job else {}
        except Exception:
            self._cached_start_job = {}

        payload = self._build_payload(
            job=job,
            now_monotonic=now_monotonic,
            now_wall=now_wall,
            phase="chunk_active",
        )
        _atomic_write_json(self.active_path, payload)

    def on_chunk_end(
        self,
        job: Dict[str, Any],
        elapsed_ms: int,
        success: bool,
    ) -> None:
        """
        Called after kernel return. Updates rolling stats and writes the
        active-worker JSON in 'idle_after_chunk' phase. If success, also
        snapshots last_successful_chunk.

        [TB Hardening] If the passed job dict is sparse (e.g. only job_id),
        merge it over the cached on_chunk_start dict. Passed values take
        precedence ONLY where they are not None — never overwrite rich
        cached state with null. This protects the forensic artifact.
        """
        now_monotonic = time.monotonic_ns()
        now_wall = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())

        # Merge: start with cached, overlay passed-non-null values
        merged_job: Dict[str, Any] = dict(self._cached_start_job)
        if job:
            for k, v in job.items():
                if v is not None:
                    merged_job[k] = v

        if success:
            self._chunks_completed += 1
            self._recent_elapsed_ms.append(int(elapsed_ms))
            self._last_successful_chunk = {
                "chunk_id": merged_job.get("job_id", "?"),
                "elapsed_ms": int(elapsed_ms),
                "wall": now_wall,
                "monotonic_ns": now_monotonic,
                "prior_chunks": self._chunks_completed - 1,
            }

        # Update idle marker on every end (success or fail)
        self._last_idle_monotonic_ns = now_monotonic

        payload = self._build_payload(
            job=merged_job,
            now_monotonic=now_monotonic,
            now_wall=now_wall,
            phase="idle_after_chunk",
            extras={
                "last_chunk_elapsed_ms": int(elapsed_ms),
                "last_chunk_success": bool(success),
            },
        )
        _atomic_write_json(self.active_path, payload)

    # -------------------------------------------------------------------
    # Internal payload builder
    # -------------------------------------------------------------------

    def _build_payload(
        self,
        job: Dict[str, Any],
        now_monotonic: int,
        now_wall: str,
        phase: str,
        extras: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Job-derived fields
        skip = job.get("skip_range", [job.get("skip_min"), job.get("skip_max")])
        skip_min = skip[0] if isinstance(skip, (list, tuple)) and len(skip) > 0 else None
        skip_max = skip[1] if isinstance(skip, (list, tuple)) and len(skip) > 1 else None

        # Build a config-string for human inspection
        w  = job.get("window_size", "?")
        o  = job.get("offset", job.get("chunk_offset", "?"))
        ft = job.get("min_match_threshold", job.get("threshold", "?"))
        rt = job.get("phase2_threshold", "?")
        config_str = "W{}_O{}_S{}-{}_FT{}_RT{}".format(w, o, skip_min, skip_max, ft, rt)

        time_since_idle_ms = max(
            0, (now_monotonic - self._last_idle_monotonic_ns) // 1_000_000
        )
        continuous_runtime_ms = max(
            0, (now_monotonic - self._start_monotonic_ns) // 1_000_000
        )

        payload: Dict[str, Any] = {
            "schema": "s173_active_worker.v1",
            "phase": phase,

            # Identity
            "host": self.host,
            "pid": self.pid,
            "worker_id": self.worker_id,
            "gpu_id": self.gpu_id,

            # Timing baselines (TB enhancement #2)
            "ts": now_wall,
            "monotonic_time_ns": now_monotonic,
            "worker_start_monotonic_ns": self._start_monotonic_ns,
            "worker_start_wall": self._start_wall,

            # Environment for daemon correlation
            "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES", ""),
            "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", ""),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),

            # Current job
            "job_id": job.get("job_id"),
            "chunk_id": job.get("job_id"),  # alias for daemon convenience
            "seed_start": job.get("seed_start"),
            "seed_end": job.get("seed_end"),
            "seed_count": (
                int(job.get("seed_end", 0)) - int(job.get("seed_start", 0))
                if job.get("seed_start") is not None and job.get("seed_end") is not None
                else None
            ),
            "window_size": job.get("window_size"),
            "offset": job.get("offset", job.get("chunk_offset")),
            "skip_min": skip_min,
            "skip_max": skip_max,
            "forward_threshold": job.get("min_match_threshold", job.get("threshold")),
            "reverse_threshold": job.get("phase2_threshold"),
            "hybrid": bool(job.get("hybrid", job.get("is_hybrid", False))),
            "prng_family": (
                (job.get("prng_families", [None]) or [None])[0]
                or job.get("prng_type")
            ),
            "config": config_str,

            # Drift-detection fields (TB enhancement #3 + #5 + #6)
            "prior_completed_chunks_on_worker": self._chunks_completed,
            "recent_elapsed_ms": list(self._recent_elapsed_ms),
            "time_since_last_idle_ms": time_since_idle_ms,
            "continuous_runtime_ms": continuous_runtime_ms,

            # Pre-fault boundary marker (TB enhancement #4)
            "last_successful_chunk": self._last_successful_chunk,
        }
        if extras:
            payload.update(extras)
        return payload

    # -------------------------------------------------------------------
    # Accessors (for testing / dashboard)
    # -------------------------------------------------------------------

    @property
    def chunks_completed(self) -> int:
        return self._chunks_completed

    @property
    def recent_elapsed_ms(self) -> List[int]:
        return list(self._recent_elapsed_ms)
