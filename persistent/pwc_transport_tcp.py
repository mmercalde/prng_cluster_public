"""
persistent/pwc_transport_tcp.py
================================
PWC Transport Adapter v1 — Framed TCP helpers + coordinator-side transport.
TB-approved S159G proposal. Team Alpha implementation.

Wire format:
  4-byte big-endian length + UTF-8 JSON body

Fixes vs prototype:
  - PWCTransportBase imported from pwc_transport_base (correct layering)
  - Inflight lease table with reclaim on worker disconnect
  - Heartbeat timeout — reclaim if worker silent too long
  - No broker. No proxy. Direct TCP connection per worker.
"""
from __future__ import annotations

import json
import queue
import socket
import struct
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from persistent.pwc_transport_base import PWCTransportBase
from persistent.pwc_result_normalizer import normalize_transport_result


# Reclaim inflight jobs if worker heartbeat silent for this long
HEARTBEAT_TIMEOUT_S = 60.0
# How often coordinator checks for expired leases
LEASE_CHECK_INTERVAL_S = 10.0
# Max job attempts before abandoning
MAX_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# Framed socket — shared by coordinator and worker
# ---------------------------------------------------------------------------

class FramedSocket:
    """
    Wraps a raw TCP socket with length-prefixed JSON framing.
    Thread-safe for concurrent send/recv via internal locks.
    """

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self._send_lock = threading.Lock()
        self._recv_lock = threading.Lock()

    def send_obj(self, obj: Dict[str, Any]) -> None:
        body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        header = struct.pack(">I", len(body))
        with self._send_lock:
            self._sendall(header + body)

    def recv_obj(self) -> Dict[str, Any]:
        with self._recv_lock:
            header = self._recvall(4)
            if not header:
                raise ConnectionError("socket closed while reading header")
            size = struct.unpack(">I", header)[0]
            if size > 64 * 1024 * 1024:  # 64 MB sanity cap
                raise ValueError(f"oversized message: {size} bytes")
            body = self._recvall(size)
            if len(body) != size:
                raise ConnectionError("socket closed while reading body")
            return json.loads(body.decode("utf-8"))

    def _sendall(self, data: bytes) -> None:
        view = memoryview(data)
        while view:
            sent = self.sock.send(view)
            if sent <= 0:
                raise ConnectionError("socket send failed")
            view = view[sent:]

    def _recvall(self, n: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < n:
            chunk = self.sock.recv(n - len(chunks))
            if not chunk:
                break
            chunks.extend(chunk)
        return bytes(chunks)

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Coordinator-side TCP transport
# ---------------------------------------------------------------------------

class TCPWorkerTransport(PWCTransportBase):
    """
    Coordinator-side TCP transport.

    Zeus binds a TCP server. Workers connect, pull jobs, push results.
    No SSH pipes. No broker. Direct framed TCP.

    Key reliability features (TB blockers addressed):
      - Inflight lease table: jobs not lost on worker disconnect
      - Reclaim on disconnect: inflight jobs requeued automatically
      - Heartbeat timeout: reclaim if worker goes silent
      - Attempt counter: abandon job after MAX_ATTEMPTS
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5600,
    ) -> None:
        self.host = host
        self.port = port

        self._server: Optional[socket.socket] = None
        self._stop = threading.Event()
        self._results: queue.Queue[Dict[str, Any]] = queue.Queue()
        self._pending_jobs: queue.Queue[Dict[str, Any]] = queue.Queue()

        # Per-job result queues for correlation — TB fix 4
        # Prevents one thread consuming another job's result in parallel dispatch
        self._job_queues: Dict[str, queue.Queue] = {}  # job_id → Queue
        self._job_queues_lock = threading.Lock()

        # Worker registry
        self._workers: Dict[str, FramedSocket] = {}  # worker_id → conn
        self._workers_lock = threading.Lock()

        # S161 v2: two-phase state tracking
        self._online_workers: set = set()   # TCP connected, not yet compute-ready
        self._ready_workers:  set = set()   # compute-ready after init
        self._state_lock = threading.Lock()
        self._init_sent: bool = False       # True after broadcast_init()

        # Inflight lease table — TB blocker C fix
        # {job_id: {worker_id, lease_id, assigned_at, attempt, job}}
        self._inflight: Dict[str, Dict[str, Any]] = {}
        self._inflight_lock = threading.Lock()

        # Worker last-seen for heartbeat timeout — TB blocker E fix
        self._last_seen: Dict[str, float] = {}  # worker_id → timestamp
        self._last_seen_lock = threading.Lock()

        self._accept_thread: Optional[threading.Thread] = None
        self._lease_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self.host, self.port))
        self._server.listen(32)
        self._server.settimeout(1.0)
        self._stop.clear()

        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True, name="pwc-tcp-accept"
        )
        self._accept_thread.start()

        self._lease_thread = threading.Thread(
            target=self._lease_reaper_loop, daemon=True, name="pwc-tcp-lease-reaper"
        )
        self._lease_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._server is not None:
            try:
                self._server.close()
            except Exception:
                pass
        with self._workers_lock:
            for worker_id, conn in list(self._workers.items()):
                try:
                    conn.send_obj({
                        "message_type":     "shutdown",
                        "protocol_version": 1,
                        "worker_id":        "coordinator",
                        "timestamp":        time.time(),
                        "reason":           "coordinator_request",
                    })
                except Exception:
                    pass
                conn.close()
            self._workers.clear()

    def submit_job(self, job: Dict[str, Any]) -> None:
        if "job_id" not in job:
            job["job_id"] = str(uuid.uuid4())
        if "attempt" not in job:
            job["attempt"] = 0
        # Create per-job result queue before submitting — TB fix 4
        job_id = job["job_id"]
        with self._job_queues_lock:
            self._job_queues[job_id] = queue.Queue()
        self._pending_jobs.put(job)

    def recv_result(self, timeout_s: Optional[float] = None,
                    job_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Receive result. If job_id provided, returns result for that specific job.
        Prevents one thread consuming another job's result — TB fix 4.
        """
        if job_id is not None:
            with self._job_queues_lock:
                job_q = self._job_queues.get(job_id)
            if job_q is None:
                return None
            try:
                result = job_q.get(timeout=timeout_s)
                # Clean up per-job queue after consumption
                with self._job_queues_lock:
                    self._job_queues.pop(job_id, None)
                return result
            except queue.Empty:
                return None
        # Fallback: global queue (used by non-correlated callers)
        try:
            return self._results.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def worker_count(self) -> int:
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
        return sent

    # ------------------------------------------------------------------
    # Inflight lease management — TB blocker C fix
    # ------------------------------------------------------------------

    def _lease_job(self, job: Dict[str, Any], worker_id: str) -> str:
        """Register job as inflight. Returns lease_id."""
        lease_id = str(uuid.uuid4())
        with self._inflight_lock:
            self._inflight[job["job_id"]] = {
                "worker_id":   worker_id,
                "lease_id":    lease_id,
                "assigned_at": time.time(),
                "attempt":     job.get("attempt", 0),
                "job":         job,
            }
        return lease_id

    def _complete_lease(self, job_id: str) -> None:
        """Remove completed job from inflight table."""
        with self._inflight_lock:
            self._inflight.pop(job_id, None)

    def _reclaim_worker_jobs(self, worker_id: str) -> List[Dict[str, Any]]:
        """
        On worker disconnect — find all inflight jobs for this worker,
        remove from inflight table, requeue if attempts remain.
        Returns list of reclaimed job_ids.
        """
        reclaimed = []
        with self._inflight_lock:
            for job_id, info in list(self._inflight.items()):
                if info["worker_id"] == worker_id:
                    job = info["job"]
                    attempt = info["attempt"] + 1
                    del self._inflight[job_id]
                    if attempt < MAX_ATTEMPTS:
                        job["attempt"] = attempt
                        self._pending_jobs.put(job)
                        reclaimed.append(job_id)
                    else:
                        # Push error result so coordinator doesn't hang
                        err_result = {
                            "job_id":    job_id,
                            "status":    "error",
                            "error":     f"max attempts ({MAX_ATTEMPTS}) exceeded after worker disconnect",
                            "worker_id": worker_id,
                            "result":    {},
                        }
                        with self._job_queues_lock:
                            job_q = self._job_queues.get(job_id)
                        if job_q is not None:
                            job_q.put(err_result)
                        else:
                            self._results.put(err_result)
        return reclaimed

    def _lease_reaper_loop(self) -> None:
        """
        Background thread — reclaim inflight jobs whose worker
        has gone silent beyond HEARTBEAT_TIMEOUT_S.
        TB blocker E fix.
        """
        while not self._stop.is_set():
            time.sleep(LEASE_CHECK_INTERVAL_S)
            now = time.time()
            with self._last_seen_lock:
                timed_out = [
                    wid for wid, ts in self._last_seen.items()
                    if now - ts > HEARTBEAT_TIMEOUT_S
                ]
            for worker_id in timed_out:
                reclaimed = self._reclaim_worker_jobs(worker_id)
                if reclaimed:
                    pass  # coordinator will see error results or requeued jobs
                with self._last_seen_lock:
                    self._last_seen.pop(worker_id, None)
                with self._workers_lock:
                    conn = self._workers.pop(worker_id, None)
                    if conn:
                        conn.close()

    # ------------------------------------------------------------------
    # Accept loop
    # ------------------------------------------------------------------

    def _accept_loop(self) -> None:
        assert self._server is not None
        while not self._stop.is_set():
            try:
                client, addr = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            t = threading.Thread(
                target=self._handle_client,
                args=(client, addr),
                daemon=True,
                name=f"pwc-tcp-worker-{addr}",
            )
            t.start()

    # ------------------------------------------------------------------
    # Per-worker handler
    # ------------------------------------------------------------------

    def _handle_client(self, client: socket.socket, addr: tuple) -> None:
        conn = FramedSocket(client)
        worker_id = f"{addr[0]}:{addr[1]}"

        try:
            # Handshake
            hello = conn.recv_obj()
            worker_id = hello.get("worker_id", worker_id)
            conn.send_obj({
                "message_type":     "hello_ack",
                "protocol_version": 1,
                "worker_id":        "coordinator",
                "timestamp":        time.time(),
                "accepted":         True,
            })

            with self._workers_lock:
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

                elif mtype == "request_job":
                    try:
                        job = self._pending_jobs.get_nowait()
                    except queue.Empty:
                        conn.send_obj({
                            "message_type":     "job_assign",
                            "protocol_version": 1,
                            "worker_id":        "coordinator",
                            "timestamp":        time.time(),
                            "job_id":           "",
                            "lease_id":         "",
                            "attempt":          0,
                            "payload":          None,
                        })
                        continue

                    lease_id = self._lease_job(job, worker_id)
                    conn.send_obj({
                        "message_type":     "job_assign",
                        "protocol_version": 1,
                        "worker_id":        "coordinator",
                        "timestamp":        time.time(),
                        "job_id":           job["job_id"],
                        "lease_id":         lease_id,
                        "attempt":          job.get("attempt", 0),
                        "payload":          job,
                    })

                elif mtype in ("result_inline", "result_spooled"):
                    # Complete lease before normalizing
                    job_id = msg.get("job_id", "")
                    self._complete_lease(job_id)
                    normalized = normalize_transport_result(msg)
                    # Route to per-job queue if registered — TB fix 4
                    with self._job_queues_lock:
                        job_q = self._job_queues.get(job_id)
                    if job_q is not None:
                        job_q.put(normalized)
                    else:
                        # Fallback to global queue
                        self._results.put(normalized)

                elif mtype == "heartbeat":
                    pass  # last_seen already updated above

                elif mtype == "shutdown":
                    break

        except (ConnectionError, OSError, json.JSONDecodeError):
            pass  # worker disconnected
        finally:
            # Reclaim any inflight jobs for this worker
            self._reclaim_worker_jobs(worker_id)
            with self._workers_lock:
                self._workers.pop(worker_id, None)
            with self._last_seen_lock:
                self._last_seen.pop(worker_id, None)
            with self._state_lock:
                self._online_workers.discard(worker_id)
                self._ready_workers.discard(worker_id)
            conn.close()
