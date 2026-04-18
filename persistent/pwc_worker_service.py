"""
persistent/pwc_worker_service.py
==================================
PWC Transport Adapter v2 — TCP worker daemon.
S161 two-phase startup: online → init → ready (lazy ROCm import).

Protocol:
  LAUNCH → CONNECT → send online → wait for init → import sieve → send ready → job loop

S163-KARG-HB: Added TB-spec worker heartbeat instrumentation.
  - Emits latest JSON + event JSONL at every lifecycle transition.
  - Files written to ~/worker_log_snapshots/worker_heartbeats/ on the rig.
  - Crash forensic daemon pulls these on every capture.
  - Instrumentation is fully defensive: any failure is logged and ignored.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import sys
import time
import traceback as tb
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from persistent.pwc_transport_tcp import FramedSocket


log = logging.getLogger("PWCWorker")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)

RECONNECT_DELAY_S      = 2.0
JOB_REQUEST_INTERVAL_S = 0.5
HEARTBEAT_INTERVAL_S   = 15.0
INLINE_MAX_BYTES       = 32 * 1024 * 1024
ROCM_READY_TIMEOUT_S   = 120.0  # max time to wait for sieve_filter import

# ── TB-spec heartbeat constants ───────────────────────────────────────────────
HEARTBEAT_SCHEMA_VERSION = "1.0.0"
HEARTBEAT_DIR = os.path.expanduser("~/worker_log_snapshots/worker_heartbeats")

# Canonical worker states — do NOT rename once deployed
_STATES = frozenset([
    "connected", "init_start", "init_done", "idle",
    "job_start", "pre_kernel", "post_kernel", "result_sent",
    "exception", "shutdown",
])


# ── TB-spec module-level helpers ──────────────────────────────────────────────

def _utc_now_iso() -> str:
    """UTC ISO-8601 timestamp with milliseconds."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _safe_cupy_pool_stats() -> dict:
    """
    Sample CuPy default + pinned memory pool stats.
    Returns dict with used/total bytes. All values None if CuPy not ready.
    Never raises.
    """
    try:
        import cupy as cp
        mp = cp.get_default_memory_pool()
        return {
            "used_bytes":         mp.used_bytes(),
            "total_bytes":        mp.total_bytes(),
            "pinned_used_bytes":  None,   # CuPy pinned pool has no used_bytes API
            "pinned_total_bytes": None,
        }
    except Exception:
        return {
            "used_bytes":         None,
            "total_bytes":        None,
            "pinned_used_bytes":  None,
            "pinned_total_bytes": None,
        }


def _atomic_write_json(path: str, payload: dict) -> None:
    """
    Write JSON atomically via tmp file + os.replace.
    Ensures the reader never sees a partial write.
    """
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _append_jsonl(path: str, payload: dict) -> None:
    """Append one compact JSON line to a JSONL file."""
    with open(path, "a") as f:
        f.write(json.dumps(payload, separators=(",", ":")) + "\n")


def _ensure_heartbeat_dir() -> bool:
    """Create heartbeat directory if it doesn't exist. Returns success."""
    try:
        os.makedirs(HEARTBEAT_DIR, exist_ok=True)
        return True
    except Exception as exc:
        log.debug(f"[HB] could not create heartbeat dir: {exc}")
        return False


# ── Main worker class ─────────────────────────────────────────────────────────

class PWCWorkerService:
    """
    TCP worker daemon. One instance per GPU.
    S161 v2: two-phase startup — connect fast, import ROCm lazily after init.
    S163-KARG-HB: emits TB-spec heartbeat JSON/JSONL at every lifecycle point.
    """

    def __init__(
        self,
        worker_id: str,
        host: str,
        port: int,
        gpu_id: int,
        use_rocm: bool = True,
    ) -> None:
        self.worker_id = worker_id
        self.host      = host
        self.port      = port
        self.gpu_id    = gpu_id
        self.use_rocm  = use_rocm

        self.jobs_done  = 0
        self.jobs_error = 0

        self._conn: Optional[FramedSocket] = None
        self._sock: Optional[socket.socket] = None
        self._execute_sieve_job = None
        self._current_job_id: Optional[str] = None

        # Heartbeat internal state
        self._connected_at: Optional[str] = None
        self._last_hb_ts: float = 0.0
        self._current_job_payload: Optional[dict] = None

        # Persistent "last_*" timestamps — survive across later state transitions
        self._last_job_start_ts:      Optional[str] = None
        self._last_kernel_launch_ts:  Optional[str] = None
        self._last_kernel_return_ts:  Optional[str] = None
        self._last_result_send_ts:    Optional[str] = None
        self._last_done_job_id:       Optional[str] = None

        # Derived file paths (set in _setup_env after worker_id is known)
        self._hb_json_path:  Optional[str] = None
        self._hb_jsonl_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run_forever(self) -> None:
        self._setup_env()

        _had_session  = False
        _refused_count = 0
        _MAX_REFUSED  = 5

        while True:
            try:
                self._connect()
                log.info(f"[{self.worker_id}] connected to {self.host}:{self.port}")
                _refused_count = 0

                self._emit_heartbeat("connected")

                # S161 v2: two-phase startup
                self._send_online()
                self._wait_for_init()
                self._import_sieve()
                self._send_ready()

                self._main_loop()
                _had_session = True
            except KeyboardInterrupt:
                log.info(f"[{self.worker_id}] interrupted")
                self._emit_heartbeat("shutdown")
                break
            except (ConnectionError, OSError) as exc:
                err_str = str(exc)
                if _had_session and "Connection refused" in err_str:
                    _refused_count += 1
                    if _refused_count >= _MAX_REFUSED:
                        log.info(
                            f"[{self.worker_id}] coordinator gone after session "
                            f"({_refused_count} refused) — exiting cleanly"
                        )
                        self._emit_heartbeat("shutdown")
                        break
                log.warning(
                    f"[{self.worker_id}] transport error: {exc} "
                    f"— reconnecting in {RECONNECT_DELAY_S}s"
                )
                self._close()
                time.sleep(RECONNECT_DELAY_S)
            except Exception as exc:
                log.error(
                    f"[{self.worker_id}] unexpected error: {exc}\n{tb.format_exc()}"
                )
                self._emit_heartbeat("exception", error=str(exc)[:500])
                self._close()
                time.sleep(RECONNECT_DELAY_S)

    # ------------------------------------------------------------------
    # Two-phase startup
    # ------------------------------------------------------------------

    def _send_online(self) -> None:
        """S161 v2: notify coordinator we are TCP-connected (not yet compute-ready)."""
        assert self._conn is not None
        self._conn.send_obj({
            "message_type": "online",
            "worker_id":    self.worker_id,
            "timestamp":    time.time(),
        })
        log.info(f"[{self.worker_id}] sent online")

    def _wait_for_init(self) -> None:
        """S161 v2: wait for coordinator to broadcast init command."""
        assert self._conn is not None
        deadline = time.time() + ROCM_READY_TIMEOUT_S
        log.info(f"[{self.worker_id}] waiting for init command...")
        while time.time() < deadline:
            msg = self._conn.recv_obj()
            mtype = msg.get("message_type")
            if mtype == "command" and msg.get("command") == "init":
                log.info(f"[{self.worker_id}] received init — importing sieve_filter")
                self._emit_heartbeat("init_start")
                return
            elif mtype == "shutdown":
                raise ConnectionError("shutdown before init")
        raise TimeoutError(f"init not received within {ROCM_READY_TIMEOUT_S}s")

    def _send_ready(self) -> None:
        """S161 v2: notify coordinator we are compute-ready."""
        assert self._conn is not None
        self._conn.send_obj({
            "message_type": "ready",
            "worker_id":    self.worker_id,
            "timestamp":    time.time(),
        })
        log.info(f"[{self.worker_id}] sent ready — entering job loop")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_env(self) -> None:
        if self.use_rocm:
            os.environ.setdefault("ROCR_VISIBLE_DEVICES",     str(self.gpu_id))
            os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
            os.environ.setdefault("HSA_ENABLE_SDMA",          "0")
            os.environ.setdefault("ROCM_PATH",                "/opt/rocm")
            os.environ.setdefault("HIP_PATH",                 "/opt/rocm")
            os.environ.setdefault(
                "CUPY_CACHE_DIR", f"/tmp/cupy_cache_gpu_{self.gpu_id}"
            )
        else:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(self.gpu_id))
            os.environ.setdefault(
                "CUPY_CACHE_DIR", f"/tmp/cupy_cache_zeus_gpu{self.gpu_id}"
            )

        # Set up heartbeat file paths and ensure directory exists
        _ensure_heartbeat_dir()
        safe_name = self.worker_id.replace(":", "_").replace("/", "_")
        self._hb_json_path  = os.path.join(HEARTBEAT_DIR, f"{safe_name}.json")
        self._hb_jsonl_path = os.path.join(HEARTBEAT_DIR, f"{safe_name}.events.jsonl")

    def _import_sieve(self) -> None:
        sys.path.insert(
            0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        try:
            from sieve_filter import execute_sieve_job
            self._execute_sieve_job = execute_sieve_job
            log.info(
                f"[{self.worker_id}] sieve_filter.execute_sieve_job imported "
                f"— gpu={self.gpu_id} rocm={self.use_rocm}"
            )
            self._emit_heartbeat("init_done")
        except Exception as exc:
            log.error(f"[{self.worker_id}] sieve_filter import FAILED: {exc}")
            self._emit_heartbeat("exception", error=f"sieve_filter import failed: {exc}")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        sock = socket.create_connection((self.host, self.port), timeout=10)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock
        self._conn = FramedSocket(sock)

        self._conn.send_obj({
            "message_type":     "hello",
            "protocol_version": 1,
            "worker_id":        self.worker_id,
            "timestamp":        time.time(),
            "gpu_id":           self.gpu_id,
            "hostname":         socket.gethostname(),
            "transport":        "tcp",
            "capabilities":     {"result_formats": ["legacy_json", "slim_v1"]},
        })

        ack = self._conn.recv_obj()
        if not ack.get("accepted", True):
            raise ConnectionError(
                f"coordinator rejected hello: {ack.get('reason')}"
            )
        # S163: remove connect timeout — recv_obj must block for init (up to 120s)
        sock.settimeout(None)
        log.info(f"[{self.worker_id}] handshake complete")
        self._connected_at = _utc_now_iso()

    def _close(self) -> None:
        if self._conn is not None:
            self._conn.close()
        self._conn  = None
        self._sock  = None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _main_loop(self) -> None:
        assert self._conn is not None

        while True:
            # Emit idle heartbeat — but throttle to avoid excessive writes
            now = time.time()
            if now - self._last_hb_ts >= 2.0:
                self._emit_heartbeat("idle")

            self._conn.send_obj({
                "message_type":     "request_job",
                "protocol_version": 1,
                "worker_id":        self.worker_id,
                "timestamp":        time.time(),
                "idle":             True,
            })

            msg = self._conn.recv_obj()
            mtype = msg.get("message_type")

            if mtype == "shutdown":
                log.info(f"[{self.worker_id}] shutdown received")
                self._emit_heartbeat("shutdown")
                break

            if mtype != "job_assign":
                log.warning(f"[{self.worker_id}] unexpected: {mtype}")
                time.sleep(0.1)
                continue

            payload = msg.get("payload")
            if payload is None:
                time.sleep(JOB_REQUEST_INTERVAL_S)
                continue

            job_id   = msg.get("job_id", "?")
            lease_id = msg.get("lease_id", job_id)
            attempt  = msg.get("attempt", 0)

            self._current_job_id      = job_id
            self._current_job_payload = payload

            log.info(
                f"[{self.worker_id}] START job={job_id} attempt={attempt} "
                f"seeds={payload.get('seed_start', 0):,}→"
                f"{payload.get('seed_end', 0):,} "
                f"prng={payload.get('prng_type', '?')}"
            )

            self._emit_heartbeat("job_start", job=payload)

            hb_stop = threading.Event()
            hb_thread = threading.Thread(
                target=self._heartbeat_loop,
                args=(hb_stop,),
                daemon=True,
            )
            hb_thread.start()

            t0     = time.time()
            result = self._execute_job(payload)
            elapsed = time.time() - t0

            hb_stop.set()
            hb_thread.join(timeout=2.0)

            if result.get("success"):
                log.info(f"[{self.worker_id}] DONE job={job_id} {elapsed:.1f}s")
                self.jobs_done += 1
            else:
                log.error(
                    f"[{self.worker_id}] ERROR job={job_id} {elapsed:.1f}s "
                    f"— {result.get('error')}"
                )
                self.jobs_error += 1

            self._send_result(payload, result, job_id, lease_id, attempt)
            self._emit_heartbeat("result_sent", job=payload)

            # Clear job tracking AFTER result_sent heartbeat so job_id is
            # preserved in the most important transition. TB fix — S163-KARG-HB.
            self._current_job_id      = None
            self._current_job_payload = None

            try:
                from sieve_filter import _best_effort_gpu_cleanup
                _best_effort_gpu_cleanup()
            except Exception as _cleanup_exc:
                log.debug(f"[{self.worker_id}] GPU cleanup skipped: {_cleanup_exc}")

            # [S163] Memory instrumentation — TB-approved
            _s163_debug = os.environ.get("S163_MEM_DEBUG", "0") == "1"
            _total_jobs = self.jobs_done + self.jobs_error
            if _total_jobs % 25 == 0 and _total_jobs > 0:
                try:
                    import cupy as _cp_s163
                    _mp = _cp_s163.get_default_memory_pool()
                    _pool_used_mb  = _mp.used_bytes()  // (1024 * 1024)
                    _pool_total_mb = _mp.total_bytes() // (1024 * 1024)
                    _pool_free_blk = _mp.n_free_blocks()
                    _vm_rss_kb  = "unknown"
                    _vm_size_kb = "unknown"
                    try:
                        for _ln in open("/proc/self/status").readlines():
                            if _ln.startswith("VmRSS:"):
                                _vm_rss_kb  = _ln.split()[1]
                            elif _ln.startswith("VmSize:"):
                                _vm_size_kb = _ln.split()[1]
                    except Exception:
                        pass
                    if _s163_debug:
                        log.info(
                            f"[MEM chunk={_total_jobs}] "
                            f"worker={self.worker_id} "
                            f"pool_used={_pool_used_mb}MB "
                            f"pool_total={_pool_total_mb}MB "
                            f"n_free_blocks={_pool_free_blk} "
                            f"VmRSS={_vm_rss_kb}kB "
                            f"VmSize={_vm_size_kb}kB"
                        )
                    if _pool_used_mb > 200:
                        log.warning(
                            f"[MEM WARNING] worker={self.worker_id} "
                            f"pool_used={_pool_used_mb}MB "
                            f"exceeds 200MB threshold at chunk={_total_jobs}"
                        )
                except Exception as _me:
                    log.debug(f"[MEM instrumentation error] {_me}")

    # ------------------------------------------------------------------
    # Coordinator heartbeat (TCP protocol — unchanged)
    # ------------------------------------------------------------------

    def _heartbeat_loop(self, stop_event: threading.Event) -> None:
        while not stop_event.wait(HEARTBEAT_INTERVAL_S):
            if self._conn is None:
                break
            try:
                self._conn.send_obj({
                    "message_type":     "heartbeat",
                    "protocol_version": 1,
                    "worker_id":        self.worker_id,
                    "timestamp":        time.time(),
                    "jobs_done":        self.jobs_done,
                    "jobs_error":       self.jobs_error,
                })
            except Exception:
                break

    # ------------------------------------------------------------------
    # Job execution
    # ------------------------------------------------------------------

    def _execute_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        assert self._execute_sieve_job is not None
        chunk_id = job.get("job_id", "?")

        try:
            sieve_job = {
                "job_id":              chunk_id,
                "search_type":         job.get("search_type", "residue_sieve"),
                "dataset_path":        job["dataset_path"],
                "seed_start":          job["seed_start"],
                "seed_end":            job["seed_end"],
                "window_size":         job["window_size"],
                "min_match_threshold": job.get("min_match_threshold",
                                               job.get("threshold", 0.3)),
                "skip_range":          job.get("skip_range",
                                               [job.get("skip_min", 0),
                                                job.get("skip_max", 147)]),
                "offset":              job.get("offset", job.get("chunk_offset", 0)),
                "sessions":            job.get("sessions", ["midday", "evening"]),
                "prng_families":       job.get("prng_families",
                                               [job.get("prng_type", "java_lcg")]),
                "strategies":          job.get("strategies"),
                "hybrid":              bool(job.get("hybrid",
                                               job.get("is_hybrid", False))),
                "phase2_threshold":    job.get("phase2_threshold", 0.5),
            }

            self._emit_heartbeat("pre_kernel", job=job)

            result = self._execute_sieve_job(sieve_job, 0)

            self._emit_heartbeat("post_kernel", job=job)

            return {
                "hostname":      socket.gethostname(),
                "gpu_id":        self.gpu_id,
                "success":       result.get("success", False),
                "payload":       result,
                "error":         result.get("error") if not result.get("success") else None,
                "result_format": "legacy_json",
            }

        except Exception as exc:
            self._emit_heartbeat(
                "exception",
                job=job,
                error=(str(exc) + "\n" + tb.format_exc())[:500],
            )
            return {
                "hostname":      socket.gethostname(),
                "gpu_id":        self.gpu_id,
                "success":       False,
                "error":         str(exc),
                "traceback":     tb.format_exc(),
                "payload":       {},
                "result_format": "legacy_json",
            }

    # ------------------------------------------------------------------
    # Result sending
    # ------------------------------------------------------------------

    def _send_result(
        self,
        job: Dict[str, Any],
        result: Dict[str, Any],
        job_id: str,
        lease_id: str,
        attempt: int,
    ) -> None:
        assert self._conn is not None
        self._conn.send_obj({
            "message_type":     "result_inline",
            "protocol_version": 1,
            "worker_id":        self.worker_id,
            "timestamp":        time.time(),
            "job_id":           job_id,
            "lease_id":         lease_id,
            "attempt":          attempt,
            "result":           result,
        })

    # ------------------------------------------------------------------
    # TB-spec worker heartbeat emission
    # ------------------------------------------------------------------

    def _emit_heartbeat(
        self,
        state: str,
        job: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Write latest heartbeat JSON (atomic) and append event JSONL.
        NEVER raises — all failures are silently logged at DEBUG level.
        Called at every canonical state transition.
        """
        if not self._hb_json_path:
            return  # _setup_env not yet called

        # Fix B: warn on non-canonical state (never blocks)
        if state not in _STATES:
            try:
                log.debug(f"[HB] non-canonical state emitted: {state!r}")
            except Exception:
                pass

        try:
            now_iso = _utc_now_iso()
            pool    = _safe_cupy_pool_stats()

            # Normalise job payload fields
            j = job or {}
            skip = j.get("skip_range", [j.get("skip_min"), j.get("skip_max")])
            skip_min = skip[0] if isinstance(skip, (list, tuple)) and len(skip) > 0 else None
            skip_max = skip[1] if isinstance(skip, (list, tuple)) and len(skip) > 1 else None

            pid  = os.getpid()
            ppid = os.getppid()
            try:
                pgid = os.getpgid(pid)
                sid  = os.getsid(pid)
            except Exception:
                pgid = None
                sid  = None

            # Fix 1: Update persistent timestamps on the transitions that own them
            if state == "job_start":
                self._last_job_start_ts     = now_iso
                self._last_kernel_launch_ts = None   # reset for new job
                self._last_kernel_return_ts = None
                self._last_result_send_ts   = None
            elif state == "pre_kernel":
                self._last_kernel_launch_ts = now_iso
            elif state == "post_kernel":
                self._last_kernel_return_ts = now_iso
            elif state == "result_sent":
                self._last_result_send_ts = now_iso
                self._last_done_job_id    = j.get("job_id") or self._current_job_id

            # Fix 2: phase must be an explicit payload field — do not substitute search_type
            phase = j.get("phase", "unknown")

            hb = {
                "schema_version":       HEARTBEAT_SCHEMA_VERSION,
                "ts":                   now_iso,
                "host":                 socket.gethostname(),
                "worker_name":          self.worker_id,
                "gpu_id":               self.gpu_id,

                "pid":                  pid,
                "ppid":                 ppid,
                "pgid":                 pgid,
                "sid":                  sid,

                "state":                state,
                "connected_at":         self._connected_at,
                "last_transition_at":   now_iso,

                "job_id":               j.get("job_id") or self._current_job_id,
                "job_type":             j.get("search_type", "unknown"),
                "phase":                phase,
                "trial_hint":           j.get("trial_number"),  # null if not in payload

                "window_size":          j.get("window_size"),
                "skip_min":             skip_min,
                "skip_max":             skip_max,
                "hybrid":               bool(j.get("hybrid", j.get("is_hybrid", False))),
                "threshold":            j.get("min_match_threshold", j.get("threshold")),
                "prng_family":          (j.get("prng_families", [None]) or [None])[0]
                                        or j.get("prng_type"),

                # Fix 1: always serialize the stored persistent values
                "last_job_start_ts":      self._last_job_start_ts,
                "last_kernel_launch_ts":  self._last_kernel_launch_ts,
                "last_kernel_return_ts":  self._last_kernel_return_ts,
                "last_result_send_ts":    self._last_result_send_ts,
                "last_done_job_id":       self._last_done_job_id,

                "env": {
                    "CUDA_VISIBLE_DEVICES":  os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "HIP_VISIBLE_DEVICES":   os.environ.get("HIP_VISIBLE_DEVICES", ""),
                    "ROCR_VISIBLE_DEVICES":  os.environ.get("ROCR_VISIBLE_DEVICES", ""),
                    "HSA_OVERRIDE_GFX_VERSION":
                        os.environ.get("HSA_OVERRIDE_GFX_VERSION", ""),
                },

                "cupy_pool":            pool,
                "last_error":           error,
            }

            # Atomic write of latest state
            _atomic_write_json(self._hb_json_path, hb)

            # Append compact transition record
            event = {
                "ts":                    now_iso,
                "worker_name":           self.worker_id,
                "host":                  socket.gethostname(),
                "gpu_id":                self.gpu_id,
                "pid":                   pid,
                "state":                 state,
                "job_id":                hb["job_id"],
                "phase":                 phase,
                "window_size":           hb["window_size"],
                "skip_min":              skip_min,
                "skip_max":              skip_max,
                "cupy_pool_used_bytes":  pool.get("used_bytes"),
                "cupy_pool_total_bytes": pool.get("total_bytes"),
                "error":                 error,
            }
            _append_jsonl(self._hb_jsonl_path, event)

            self._last_hb_ts = time.time()

        except Exception as exc:
            # NEVER let heartbeat errors affect worker operation
            try:
                log.debug(f"[HB] emit failed (state={state}): {exc}")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    gpu_id    = int(os.environ.get("PWC_GPU_ID",    "0"))
    host      = os.environ.get("PWC_HOST",      "192.168.3.127")
    port      = int(os.environ.get("PWC_PORT",      "5600"))
    worker_id = os.environ.get(
        "PWC_WORKER_ID", f"{socket.gethostname()}:gpu{gpu_id}"
    )
    use_rocm  = os.environ.get("PWC_USE_ROCM", "1") != "0"

    svc = PWCWorkerService(
        worker_id=worker_id,
        host=host,
        port=port,
        gpu_id=gpu_id,
        use_rocm=use_rocm,
    )
    svc.run_forever()
