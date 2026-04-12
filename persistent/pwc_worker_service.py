"""
persistent/pwc_worker_service.py
==================================
PWC Transport Adapter v2 — TCP worker daemon.
S161 two-phase startup: online → init → ready (lazy ROCm import).

Protocol:
  LAUNCH → CONNECT → send online → wait for init → import sieve → send ready → job loop
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
from typing import Any, Dict, Optional

from persistent.pwc_transport_tcp import FramedSocket


log = logging.getLogger("PWCWorker")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)

RECONNECT_DELAY_S     = 2.0
JOB_REQUEST_INTERVAL_S = 0.5
HEARTBEAT_INTERVAL_S  = 15.0
INLINE_MAX_BYTES = 32 * 1024 * 1024

ROCM_READY_TIMEOUT_S = 120.0  # max time to wait for sieve_filter import


class PWCWorkerService:
    """
    TCP worker daemon. One instance per GPU.
    S161 v2: two-phase startup — connect fast, import ROCm lazily after init.
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

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run_forever(self) -> None:
        self._setup_env()
        # S161 v2: NO sieve import here — lazy import after coordinator sends init

        _had_session  = False
        _refused_count = 0
        _MAX_REFUSED  = 5

        while True:
            try:
                self._connect()
                log.info(f"[{self.worker_id}] connected to {self.host}:{self.port}")
                _refused_count = 0

                # S161 v2: two-phase startup
                self._send_online()
                self._wait_for_init()
                self._import_sieve()
                self._send_ready()

                self._main_loop()
                _had_session = True
            except KeyboardInterrupt:
                log.info(f"[{self.worker_id}] interrupted")
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
        except Exception as exc:
            log.error(f"[{self.worker_id}] sieve_filter import FAILED: {exc}")
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

            self._current_job_id = job_id
            log.info(
                f"[{self.worker_id}] START job={job_id} attempt={attempt} "
                f"seeds={payload.get('seed_start', 0):,}→"
                f"{payload.get('seed_end', 0):,} "
                f"prng={payload.get('prng_type', '?')}"
            )

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
            self._current_job_id = None

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

            try:
                from sieve_filter import _best_effort_gpu_cleanup
                _best_effort_gpu_cleanup()
            except Exception as _cleanup_exc:
                log.debug(f"[{self.worker_id}] GPU cleanup skipped: {_cleanup_exc}")

    # ------------------------------------------------------------------
    # Heartbeat
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
            result = self._execute_sieve_job(sieve_job, 0)

            return {
                "hostname":      socket.gethostname(),
                "gpu_id":        self.gpu_id,
                "success":       result.get("success", False),
                "payload":       result,
                "error":         result.get("error") if not result.get("success") else None,
                "result_format": "legacy_json",
            }

        except Exception as exc:
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
