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
HEARTBEAT_TIMEOUT_S = 300.0
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

        # S162: Dispatch semaphore — limits concurrent job payload sends to 8.
        # Prevents 24 simultaneous conn.send_obj() calls from flooding Zeus's
        # 1GbE NIC (Intel I210) when all workers request jobs at the same time.
        # Value=8 = one rig's worth of workers — allows full rig parallelism
        # while preventing cross-rig NIC saturation.
        self._dispatch_semaphore = threading.Semaphore(8)

        # [S168-FIRST-ASSIGN-JITTER]
        self._first_assign_done = set()
        self._first_assign_lock = threading.Lock()

        # [S169-PER-WORKER-PACING]
        self._last_assign_time = {}
        self._last_assign_lock = threading.Lock()

        # [PWC-STAGE1 A3 level 1] launch-set admission. None = no list (every
        # hello accepted, the behaviour before this change, kept for any caller
        # that never sets one). A set = only those worker ids are accepted; any
        # other hello gets accepted=False and is never registered.
        self._allowed_worker_ids: Optional[set] = None
        self._allowed_lock = threading.Lock()
        self.refused_hellos: List[str] = []   # worker ids refused, in order (bounded)
        # [PWC-STAGE2 R2] (worker_id, job_id) of results dropped as not for an active lease
        self.stale_results: List[tuple] = []



    # [PWC-STAGE1 A3 level 1] launch-set admission
    def set_allowed_worker_ids(self, ids) -> None:
        with self._allowed_lock:
            self._allowed_worker_ids = None if ids is None else set(str(i) for i in ids)

    def allow_worker_id(self, worker_id: str) -> None:
        with self._allowed_lock:
            if self._allowed_worker_ids is None:
                raise RuntimeError("allow_worker_id() before set_allowed_worker_ids(): "
                                   "the launch set was never opened")
            self._allowed_worker_ids.add(str(worker_id))

    def revoke_worker_id(self, worker_id: str) -> None:
        """[PWC-STAGE2 BETA-3] Revocation now also applies to a connection the id
        already holds (Beta X-PWC-STAGE1 R1 finding 3: a worker that connected while
        its SSH launch was still running kept its accepted socket, reported ready and
        took a job after revocation). The id leaves the online and ready sets at
        once, so it is never counted or dispatched to, and its socket is closed; its
        handler then ends and reclaims anything it had leased."""
        wid = str(worker_id)
        with self._allowed_lock:
            if self._allowed_worker_ids is not None:
                self._allowed_worker_ids.discard(wid)
        with self._state_lock:
            self._online_workers.discard(wid)
            self._ready_workers.discard(wid)
        with self._workers_lock:
            conn = self._workers.get(wid)
        if conn is not None:
            # shutdown, not just close: close() does not wake the handler thread
            # blocked in recv on this socket; shutdown makes that recv return EOF,
            # so the handler ends and cleans up as its connection's owner.
            try:
                conn.sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass

    def _admits(self, worker_id: str) -> bool:
        with self._allowed_lock:
            return self._allowed_worker_ids is None or str(worker_id) in self._allowed_worker_ids

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

    def _lease_job(self, job: Dict[str, Any], worker_id: str, conn: Any = None) -> str:
        """Register job as inflight. Returns lease_id. [PWC-STAGE2] `conn` records
        which connection carried it, so a replaced connection reclaims only its own."""
        lease_id = str(uuid.uuid4())
        with self._inflight_lock:
            self._inflight[job["job_id"]] = {
                "conn":        conn,
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

    def _take_active_lease(self, job_id: str, lease_id: Any, conn: Any) -> bool:
        """[PWC-STAGE2 R2, Beta finding 2] Retire the lease for `job_id` ONLY if it is
        the ACTIVE lease, the result names that lease, and it arrived on the
        connection that holds it. Anything else is a late or foreign result (a worker
        still finishing a job that timed out and was handed to another worker, or a
        result for a lease that was already reclaimed): it must neither retire the
        current lease nor be delivered as the current job's result."""
        with self._inflight_lock:
            info = self._inflight.get(job_id)
            if info is None or info.get("lease_id") != lease_id or info.get("conn") is not conn:
                return False
            del self._inflight[job_id]
            return True

    def _reclaim_worker_jobs(self, worker_id: str, conn: Any = None) -> List[Dict[str, Any]]:
        """
        On worker disconnect — find all inflight jobs for this worker,
        remove from inflight table, requeue if attempts remain.
        Returns list of reclaimed job_ids.
        """
        reclaimed = []
        with self._inflight_lock:
            for job_id, info in list(self._inflight.items()):
                if info["worker_id"] == worker_id and (conn is None or info.get("conn") is conn):
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
            self._reap_silent_workers(time.time())

    def _reap_silent_workers(self, now: float) -> None:
        """[PWC-STAGE2 R2, Beta X-PWC-STAGE2 R1 Q6] Timeout eviction under the same
        ownership rule as a disconnect: the registered connection is evicted, and
        its id's registry, last-seen and online/ready state are cleared, in ONE
        critical section under `_workers_lock`, re-checking the silence there (a
        reconnect may have refreshed it). Only the leases the evicted connection
        carried are reclaimed. The socket is shut down so its handler wakes and ends
        (it is then no longer the owner and clears nothing else)."""
        with self._last_seen_lock:
            timed_out = [
                wid for wid, ts in self._last_seen.items()
                if now - ts > HEARTBEAT_TIMEOUT_S
            ]
        for worker_id in timed_out:
            with self._workers_lock:
                with self._last_seen_lock:
                    ts = self._last_seen.get(worker_id)
                    if ts is None or now - ts <= HEARTBEAT_TIMEOUT_S:
                        continue
                    self._last_seen.pop(worker_id, None)
                conn = self._workers.pop(worker_id, None)
                with self._state_lock:
                    self._online_workers.discard(worker_id)
                    self._ready_workers.discard(worker_id)
            if conn is not None:
                self._reclaim_worker_jobs(worker_id, conn)
                try:
                    conn.sock.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass

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
        refused = False

        try:
            # Handshake
            hello = conn.recv_obj()
            worker_id = hello.get("worker_id", worker_id)
            if not self._admits(worker_id):
                # [PWC-STAGE1 A3 level 1] not launched by this coordinator: refuse,
                # register nothing, and leave every table untouched (the finally
                # block below must not reclaim or pop a same-named entry).
                refused = True
                if len(self.refused_hellos) < 10_000:
                    self.refused_hellos.append(str(worker_id))
                try:
                    conn.send_obj({
                        "message_type":     "hello_ack",
                        "protocol_version": 1,
                        "worker_id":        "coordinator",
                        "timestamp":        time.time(),
                        "accepted":         False,
                        "reason":           "worker_id not in this coordinator's launch set",
                    })
                except (ConnectionError, OSError):
                    pass
                return
            conn.send_obj({
                "message_type":     "hello_ack",
                "protocol_version": 1,
                "worker_id":        "coordinator",
                "timestamp":        time.time(),
                "accepted":         True,
            })

            # [PWC-STAGE2 R3, Beta X-PWC-STAGE2 R2 Q1] Register AND refresh last-seen in
            # ONE critical section under `_workers_lock`, the section the timeout
            # reaper uses (`_reap_silent_workers`). The reaper therefore never sees
            # this fresh connection paired with the previous connection's stale
            # timestamp.
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
                # [PWC-STAGE2 R3] only while THIS connection is the registered one for
                # the id: a replaced connection's messages never refresh (or create)
                # the timestamp the reaper judges the current connection by.
                with self._workers_lock:
                    if self._workers.get(worker_id) is not conn:
                        break
                    with self._last_seen_lock:
                        self._last_seen[worker_id] = time.time()

                if mtype == "online":
                    # S161 v2: worker is TCP-connected, not yet compute-ready
                    # [PWC-STAGE2 R3] set only by the registered connection, inside
                    # the registry lock, so an evicted/replaced one cannot set it
                    with self._workers_lock:
                        if self._workers.get(worker_id) is not conn:
                            break
                        with self._state_lock:
                            self._online_workers.add(worker_id)
                    continue

                elif mtype == "ready":
                    # S161 v2: worker completed ROCm warmup — compute-ready
                    if not self._admits(worker_id):     # [PWC-STAGE2 BETA-3]
                        break
                    # [PWC-STAGE2 R3] as for online: registered connection only
                    with self._workers_lock:
                        if self._workers.get(worker_id) is not conn:
                            break
                        with self._state_lock:
                            self._ready_workers.add(worker_id)
                    continue

                elif mtype == "request_job":
                    # [PWC-STAGE2 BETA-3] an id revoked after its hello gets no job
                    if not self._admits(worker_id):
                        break
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

                    
                    # [S168-FIRST-ASSIGN-JITTER]
                    try:
                        import os as _s168_os
                        import time as _s168_time
                        import zlib as _s168_zlib

                        jitter = float(_s168_os.environ.get("PRNG_PWC_FIRST_ASSIGN_JITTER_SEC", "0") or 0)

                        if jitter > 0:
                            with self._first_assign_lock:
                                first = worker_id not in self._first_assign_done
                                if first:
                                    self._first_assign_done.add(worker_id)

                            if first:
                                slots = max(1, int(jitter * 1000))
                                delay = (_s168_zlib.crc32(worker_id.encode("utf-8")) % (slots + 1)) / 1000.0

                                if delay > 0:
                                    _s168_time.sleep(delay)

                    except Exception:
                        pass

                    
                    # [S169-PER-WORKER-PACING]
                    # Optional steady-state smoothing. Default off.
                    try:
                        import os as _s169_os
                        import time as _s169_time

                        min_gap = float(_s169_os.environ.get("PRNG_PWC_PER_WORKER_MIN_GAP_SEC", "0") or 0)

                        if min_gap > 0:
                            with self._last_assign_lock:
                                now = _s169_time.time()
                                last = self._last_assign_time.get(worker_id, 0.0)
                                wait = min_gap - (now - last)

                                if wait > 0:
                                    _s169_time.sleep(wait)
                                    now = _s169_time.time()

                                self._last_assign_time[worker_id] = now

                    except Exception:
                        pass

                    lease_id = self._lease_job(job, worker_id, conn)

                    # [S168A-DIAG] job assignment telemetry (PASSIVE, env-gated)
                    try:
                        import os as _s168_os
                        import json as _s168_json
                        import time as _s168_time
                        from pathlib import Path as _S168Path
                        if _s168_os.environ.get("PRNG_PWC_STARTUP_DIAG", "0") == "1":
                            _S168Path("logs").mkdir(exist_ok=True)
                            with open("logs/pwc_startup_diag_simple.jsonl", "a") as _f:
                                _f.write(_s168_json.dumps({
                                    "event": "job_assign",
                                    "t": _s168_time.time(),
                                    "worker_id": worker_id,
                                    "job_id": job.get("job_id"),
                                    "seed_start": job.get("seed_start"),
                                    "seed_end": job.get("seed_end"),
                                    "prng_type": job.get("prng_type"),
                                    "attempt": job.get("attempt", 0)
                                }) + "\n")
                    except Exception:
                        pass
                    # S162: Semaphore limits concurrent job sends to 8 at a time.
                    # Prevents NIC flood on Zeus's 1GbE (Intel I210) when all
                    # 24 workers request jobs simultaneously.
                    with self._dispatch_semaphore:
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
                    # [PWC-STAGE2 R2, Beta finding 2] Only a result for the ACTIVE
                    # lease, from the connection holding it, retires the lease and is
                    # delivered. A late result (its job timed out and was handed to
                    # another worker, or its lease was reclaimed) is dropped and
                    # counted; it never completes or answers the current job.
                    job_id = msg.get("job_id", "")
                    if not self._take_active_lease(job_id, msg.get("lease_id"), conn):
                        if len(self.stale_results) < 10_000:
                            self.stale_results.append((str(worker_id), str(job_id)))
                        continue
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
            # [PWC-STAGE1 A3 level 1] a refused hello registered nothing, so it
            # must not reclaim or pop anything filed under the same id.
            if not refused:
                # [PWC-STAGE2 BETA-3] Only the connection that still owns this id
                # cleans up under it. A later connection with the same id (a
                # reconnect) replaces the registry entry; when the older handler
                # ends, it must not pop, reclaim or un-ready the newer one (Beta
                # X-PWC-STAGE1 R1 Q3).
                # [PWC-STAGE2 R2, Beta X-PWC-STAGE2 R1 Q6] The ownership check and
                # every piece of per-id state it governs (registry, last-seen,
                # online/ready) change in ONE critical section under
                # `_workers_lock`, the lock a replacement takes to register. A
                # replacement therefore registers either before (this handler is not
                # the owner and clears nothing) or after (it reports online/ready
                # only after registering, so nothing of it exists yet to clear).
                with self._workers_lock:
                    owner = self._workers.get(worker_id) is conn
                    if owner:
                        self._workers.pop(worker_id, None)
                        with self._last_seen_lock:
                            self._last_seen.pop(worker_id, None)
                        with self._state_lock:
                            self._online_workers.discard(worker_id)
                            self._ready_workers.discard(worker_id)
                # Reclaim ONLY the leases this connection carried, owner or not: a
                # replacement's lease is never requeued or failed by this handler.
                self._reclaim_worker_jobs(worker_id, conn)
            conn.close()
