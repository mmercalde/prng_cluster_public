"""
zmq_sqlite_coordinator.py  —  S158D v2: ZMQ + SQLite Distributed Sieve Coordinator
====================================================================================
DROP-IN REPLACEMENT for PersistentWorkerCoordinator.
Activated by --use-zmq-sqlite flag on window_optimizer.py.

TB-APPROVED GUARDRAILS (v2):
  1. SQLite schema has lease_expires_at, attempt_count, claimed_by (worker identity)
  2. Zeus is the SOLE SQLite writer — workers never touch the DB
  3. Result ingestion is idempotent — duplicate chunk_id results silently ignored
  4. Worker identity is explicit: "hostname:gpuN" bound to SQLite claims
  5. JSON only — no pickle anywhere
  6. Install via venv, not --break-system-packages

Architecture:
  Zeus runs ZMQ PUSH (job dispatch) + PULL (result collection) sockets.
  SQLite tracks chunk state with lease expiry. Zeus-only writer.
  Workers launched ONCE via SSH, then run independently via ZMQ TCP.

ZMQ Pattern:
  Zeus PUSH (port 5557) -> Workers PULL  (job dispatch)
  Workers PUSH          -> Zeus PULL (port 5558) (result collection)
"""

import json
import logging
import os
import socket
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ZMQSQLiteCoordinator")

ZMQ_JOB_PORT     = 5557
ZMQ_RESULT_PORT  = 5558
DB_PATH          = "zmq_job_queue.db"
LEASE_DURATION_S = 900
MAX_ATTEMPTS     = 3
WORKER_SETTLE_S  = 5
RESULT_POLL_MS   = 500


class JobQueue:
    """
    SQLite-backed job queue. Zeus is the SOLE writer.
    WAL mode. Single writer = no write-write conflicts.
    Workers never touch this DB — ZMQ only.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._write_lock = threading.Lock()
        self._init_db()

    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._write_lock:
            with self._conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS job_queue (
                        chunk_id         TEXT PRIMARY KEY,
                        run_id           TEXT NOT NULL,
                        prng_type        TEXT NOT NULL,
                        seed_start       INTEGER NOT NULL,
                        seed_end         INTEGER NOT NULL,
                        window_size      INTEGER NOT NULL,
                        chunk_offset     INTEGER NOT NULL,
                        threshold        REAL NOT NULL,
                        skip_min         INTEGER NOT NULL,
                        skip_max         INTEGER NOT NULL,
                        sessions_json    TEXT NOT NULL,
                        dataset_path     TEXT NOT NULL,
                        is_hybrid        INTEGER NOT NULL DEFAULT 0,
                        strategies_json  TEXT,
                        status           TEXT NOT NULL DEFAULT 'pending',
                        claimed_by       TEXT,
                        claimed_at       REAL,
                        lease_expires_at REAL,
                        attempt_count    INTEGER NOT NULL DEFAULT 0,
                        completed_at     REAL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS job_results (
                        chunk_id         TEXT PRIMARY KEY,
                        run_id           TEXT NOT NULL,
                        survivors_json   TEXT NOT NULL,
                        match_rates_json TEXT NOT NULL,
                        skip_seqs_json   TEXT NOT NULL,
                        strat_ids_json   TEXT NOT NULL,
                        worker_id        TEXT NOT NULL,
                        created_at       REAL NOT NULL
                    )
                """)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_status_run "
                    "ON job_queue (status, run_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_lease "
                    "ON job_queue (lease_expires_at, status)"
                )
                conn.commit()

    def enqueue_chunks(self, run_id, prng_type, chunks, window_size, offset,
                       threshold, skip_min, skip_max, sessions, dataset_path,
                       is_hybrid=False, strategies=None):
        strat_json = json.dumps(strategies) if strategies else None
        sess_json  = json.dumps(sessions)
        rows = [
            (f"{run_id}_{i}", run_id, prng_type, s, e, window_size, offset,
             threshold, skip_min, skip_max, sess_json, dataset_path,
             1 if is_hybrid else 0, strat_json,
             "pending", None, None, None, 0, None)
            for i, (s, e) in enumerate(chunks)
        ]
        with self._write_lock:
            with self._conn() as conn:
                conn.executemany("""
                    INSERT OR REPLACE INTO job_queue
                    (chunk_id,run_id,prng_type,seed_start,seed_end,
                     window_size,chunk_offset,threshold,skip_min,skip_max,
                     sessions_json,dataset_path,is_hybrid,strategies_json,
                     status,claimed_by,claimed_at,lease_expires_at,
                     attempt_count,completed_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, rows)
                conn.commit()
        logger.info(f"[ZMQ-DB] Enqueued {len(rows)} chunks run={run_id}")

    def claim_chunk(self, chunk_id, worker_id):
        """Zeus records chunk dispatched to worker with lease."""
        now = time.time()
        with self._write_lock:
            with self._conn() as conn:
                conn.execute("""
                    UPDATE job_queue
                    SET status='claimed', claimed_by=?, claimed_at=?,
                        lease_expires_at=?, attempt_count=attempt_count+1
                    WHERE chunk_id=? AND status='pending'
                """, (worker_id, now, now + LEASE_DURATION_S, chunk_id))
                conn.commit()

    def complete_chunk(self, chunk_id, worker_id, survivors, match_rates,
                       skip_seqs, strat_ids):
        """
        Zeus records result. Idempotent — duplicate chunk_id silently ignored.
        Returns True if first completion, False if duplicate.
        """
        with self._write_lock:
            with self._conn() as conn:
                existing = conn.execute(
                    "SELECT chunk_id FROM job_results WHERE chunk_id=?",
                    (chunk_id,)
                ).fetchone()
                if existing:
                    logger.debug(
                        f"[ZMQ-DB] Duplicate result {chunk_id} "
                        f"from {worker_id} — ignored"
                    )
                    return False
                conn.execute("""
                    INSERT INTO job_results
                    (chunk_id,run_id,survivors_json,match_rates_json,
                     skip_seqs_json,strat_ids_json,worker_id,created_at)
                    SELECT chunk_id,run_id,?,?,?,?,?,?
                    FROM job_queue WHERE chunk_id=?
                """, (json.dumps(survivors), json.dumps(match_rates),
                      json.dumps(skip_seqs), json.dumps(strat_ids),
                      worker_id, time.time(), chunk_id))
                conn.execute(
                    "UPDATE job_queue SET status='done', completed_at=? "
                    "WHERE chunk_id=?", (time.time(), chunk_id)
                )
                conn.commit()
                return True

    def reclaim_expired_leases(self, run_id):
        """Re-queue expired leases. Zeus-only write."""
        now = time.time()
        reclaimed = []
        with self._write_lock:
            with self._conn() as conn:
                expired = conn.execute("""
                    SELECT chunk_id, claimed_by, attempt_count
                    FROM job_queue
                    WHERE run_id=? AND status='claimed' AND lease_expires_at<?
                """, (run_id, now)).fetchall()
                for row in expired:
                    if row["attempt_count"] < MAX_ATTEMPTS:
                        conn.execute("""
                            UPDATE job_queue
                            SET status='pending', claimed_by=NULL,
                                claimed_at=NULL, lease_expires_at=NULL
                            WHERE chunk_id=?
                        """, (row["chunk_id"],))
                        reclaimed.append(dict(row))
                        logger.warning(
                            f"[ZMQ-DB] Lease expired {row['chunk_id']} "
                            f"worker={row['claimed_by']} — reclaimed "
                            f"(attempt {row['attempt_count']}/{MAX_ATTEMPTS})"
                        )
                    else:
                        conn.execute(
                            "UPDATE job_queue SET status='failed' "
                            "WHERE chunk_id=?", (row["chunk_id"],)
                        )
                        logger.error(
                            f"[ZMQ-DB] {row['chunk_id']} exceeded "
                            f"{MAX_ATTEMPTS} attempts — failed"
                        )
                conn.commit()
        return reclaimed

    def get_pending_jobs(self, run_id):
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM job_queue WHERE run_id=? AND status='pending' "
                "ORDER BY chunk_id", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_results(self, run_id):
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM job_results WHERE run_id=?", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def count_by_status(self, run_id):
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as n FROM job_queue "
                "WHERE run_id=? GROUP BY status", (run_id,)
            ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    def cleanup(self, run_id):
        with self._write_lock:
            with self._conn() as conn:
                conn.execute("DELETE FROM job_queue WHERE run_id=?", (run_id,))
                conn.execute("DELETE FROM job_results WHERE run_id=?", (run_id,))
                conn.commit()


class ZMQSQLiteCoordinator:
    """
    Zeus-side coordinator. Sole SQLite writer. ZMQ PUSH/PULL server.
    Drop-in replacement for PersistentWorkerCoordinator.
    """

    def __init__(self, config_file="distributed_config.json",
                 seed_cap_amd=2_000_000, seed_cap_nvidia=5_000_000,
                 worker_pool_size=8,
                 zmq_job_port=ZMQ_JOB_PORT, zmq_result_port=ZMQ_RESULT_PORT):
        self.config_file      = config_file
        self.seed_cap_amd     = seed_cap_amd
        self.seed_cap_nvidia  = seed_cap_nvidia
        self.worker_pool_size = worker_pool_size
        self.zmq_job_port     = zmq_job_port
        self.zmq_result_port  = zmq_result_port
        self.db               = JobQueue()
        self.logger           = logging.getLogger("ZMQSQLiteCoordinator")
        self._nodes           = self._load_nodes()
        self._workers_launched = False
        self._zeus_ip         = self._get_zeus_ip()
        self._progress_writer = None
        self._init_progress_writer()

    def _init_progress_writer(self):
        """Initialize ProgressWriter for web dashboard — mirrors PWC.startup()."""
        try:
            from progress_display import ProgressWriter
            self._progress_writer = ProgressWriter(
                "Forward Sieve", total_jobs=100, total_seeds=0
            )
            for node in self._nodes:
                host = node.get("hostname", "")
                if host in ("localhost", "127.0.0.1"):
                    self._progress_writer.register_node(
                        "localhost", "RTX 3080 Ti", 2
                    )
                elif node.get("gpu_count", 0) > 0:
                    self._progress_writer.register_node(
                        host,
                        node.get("gpu_type", "RX 6600"),
                        node.get("gpu_count", 8)
                    )
        except Exception as e:
            self.logger.warning(f"[ZMQ] ProgressWriter unavailable: {e}")
            self._progress_writer = None

    def _get_zeus_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("192.168.3.120", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "localhost"

    def _load_nodes(self):
        try:
            with open(self.config_file) as f:
                return json.load(f).get("nodes", [])
        except Exception as e:
            self.logger.warning(f"Could not load {self.config_file}: {e}")
            return []

    def _launch_workers(self):
        """SSH to each rig ONCE. Fire and forget. SSH closes after launch."""
        if self._workers_launched:
            return
        import subprocess

        for node in self._nodes:
            host      = node.get("hostname", "")
            username  = node.get("username", "michael")
            gpu_count = node.get("gpu_count", 0)
            if not host or host in ("localhost", "127.0.0.1") or gpu_count == 0:
                continue
            py_env      = node.get("python_env", "~/rocm_env/bin/python3")
            script_path = node.get("script_path", "~/distributed_prng_analysis")
            activate    = f"source {os.path.join(os.path.dirname(py_env), 'activate')}"
            kill_cmd    = "pkill -9 -f zmq_sqlite_worker.py 2>/dev/null; sleep 1"

            launches = []
            for gpu_id in range(gpu_count):
                worker_id = f"{host}:gpu{gpu_id}"
                env_vars  = (
                    f"ROCR_VISIBLE_DEVICES={gpu_id} "
                    f"CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_{gpu_id} "
                    f"HSA_OVERRIDE_GFX_VERSION=10.3.0"
                )
                launches.append(
                    f"nohup {env_vars} {py_env} -u "
                    f"{script_path}/zmq_sqlite_worker.py "
                    f"--zeus-host {self._zeus_ip} "
                    f"--job-port {self.zmq_job_port} "
                    f"--result-port {self.zmq_result_port} "
                    f"--worker-id {worker_id} "
                    f"--gpu-id {gpu_id} "
                    f"> /tmp/zmq_worker_gpu{gpu_id}.log 2>&1 &"
                )

            full_cmd = (
                f"{activate} && cd {script_path} && "
                f"{kill_cmd} && " + "\n".join(launches)
            )
            try:
                # Fire and forget — nohup workers run independently after SSH exits.
                # Do NOT wait() — SSH session stays open while background processes
                # run on rig, causing timeout and killing workers before they start.
                subprocess.Popen(
                    ["ssh", "-q",
                     "-o", "StrictHostKeyChecking=no",
                     "-o", "BatchMode=yes",
                     "-o", "ConnectTimeout=10",
                     "-o", "ServerAliveInterval=0",
                     f"{username}@{host}", full_cmd],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self.logger.info(
                    f"[ZMQ] Launched {gpu_count} workers on {host}"
                )
            except Exception as e:
                self.logger.error(f"[ZMQ] Failed to launch on {host}: {e}")

        # Zeus local CUDA workers
        import subprocess as sp
        for gpu_id in range(2):
            worker_id = f"localhost:gpu{gpu_id}"
            try:
                sp.Popen(
                    ["python3", "zmq_sqlite_worker.py",
                     "--zeus-host", "localhost",
                     "--job-port",    str(self.zmq_job_port),
                     "--result-port", str(self.zmq_result_port),
                     "--worker-id",   worker_id,
                     "--gpu-id",      str(gpu_id),
                     "--cuda"],
                    stdout=open(f"/tmp/zmq_zeus_gpu{gpu_id}.log", "w"),
                    stderr=sp.STDOUT
                )
                self.logger.info(f"[ZMQ] Zeus CUDA worker launched ({worker_id})")
            except Exception as e:
                self.logger.error(f"[ZMQ] Zeus GPU{gpu_id} launch failed: {e}")

        time.sleep(WORKER_SETTLE_S)
        self._workers_launched = True
        self.logger.info("[ZMQ] All workers launched and settled")

    def run_sieve_pass(self, prng_type, residues, total_seeds, threshold,
                       window_size, output_file, dataset_path="",
                       strategies=None, phase2_threshold=0.5,
                       target_file="", offset=0, sessions=None,
                       skip_range=None):
        try:
            import zmq
        except ImportError:
            raise ImportError(
                "pyzmq required. Install in venv: pip install pyzmq"
            )

        is_hybrid  = "_hybrid" in prng_type
        sessions   = sessions   or ["midday", "evening"]
        skip_range = skip_range or [0, 147]
        skip_min, skip_max = skip_range[0], skip_range[1]
        run_id     = f"{prng_type}_{uuid.uuid4().hex[:8]}"

        if is_hybrid and strategies is None:
            try:
                from hybrid_strategy import get_strategy as _gs
                s = _gs("balanced_hybrid")
                strategies = [s.to_dict() if hasattr(s, "to_dict") else vars(s)]
            except Exception:
                strategies = []

        # Build chunks
        total_workers = sum(
            n.get("gpu_count", 0) for n in self._nodes
            if n.get("hostname") not in ("localhost", "127.0.0.1")
        ) + 2
        num_workers = max(1, total_workers)
        chunk_size  = min(
            max(1, total_seeds // num_workers), self.seed_cap_amd
        )
        chunks = []
        seed = 0
        while seed < total_seeds:
            chunks.append((seed, min(seed + chunk_size, total_seeds)))
            seed += chunk_size
        total_chunks = len(chunks)

        self.logger.info(
            f"[ZMQ] {prng_type} {total_seeds:,} seeds "
            f"-> {total_chunks} chunks ({chunk_size:,}/chunk)"
        )
        if self._progress_writer:
            try:
                _step_name = (
                    f"{'Reverse' if 'reverse' in prng_type else 'Forward'} "
                    f"Sieve ({prng_type})"
                )
                self._progress_writer.update_step(
                    _step_name, total_seeds=total_seeds
                )
            except Exception:
                pass

        self.db.enqueue_chunks(
            run_id=run_id, prng_type=prng_type, chunks=chunks,
            window_size=window_size, offset=offset, threshold=threshold,
            skip_min=skip_min, skip_max=skip_max, sessions=sessions,
            dataset_path=dataset_path or target_file,
            is_hybrid=is_hybrid, strategies=strategies
        )

        # Bind sockets BEFORE launching workers so workers can connect immediately
        ctx         = zmq.Context()
        job_sock    = ctx.socket(zmq.PUSH)
        result_sock = ctx.socket(zmq.PULL)
        job_sock.setsockopt(zmq.SNDHWM, total_chunks + 100)
        result_sock.setsockopt(zmq.RCVTIMEO, RESULT_POLL_MS)
        job_sock.bind(f"tcp://*:{self.zmq_job_port}")
        result_sock.bind(f"tcp://*:{self.zmq_result_port}")

        # Launch workers AFTER sockets are bound — workers connect immediately
        self._launch_workers()

        try:
            # Dispatch all pending chunks
            for job in self.db.get_pending_jobs(run_id):
                self.db.claim_chunk(job["chunk_id"], f"dispatched:{job['chunk_id']}")
                job_sock.send(json.dumps(job).encode())
            self.logger.info(f"[ZMQ] Dispatched {total_chunks} chunks")

            completed    = 0
            last_reclaim = time.time()
            last_log     = time.time()

            while completed < total_chunks:
                if time.time() - last_reclaim > 60:
                    reclaimed = self.db.reclaim_expired_leases(run_id)
                    for _ in reclaimed:
                        for retry_job in self.db.get_pending_jobs(run_id):
                            self.db.claim_chunk(
                                retry_job["chunk_id"],
                                f"retry:{retry_job['chunk_id']}"
                            )
                            job_sock.send(json.dumps(retry_job).encode())
                    last_reclaim = time.time()

                if time.time() - last_log > 30:
                    c = self.db.count_by_status(run_id)
                    self.logger.info(
                        f"[ZMQ] {prng_type}: done={c.get('done',0)} "
                        f"claimed={c.get('claimed',0)} "
                        f"pending={c.get('pending',0)} "
                        f"failed={c.get('failed',0)}"
                    )
                    last_log = time.time()

                try:
                    msg    = result_sock.recv()
                    result = json.loads(msg.decode())
                except zmq.Again:
                    c = self.db.count_by_status(run_id)
                    if c.get("done", 0) + c.get("failed", 0) >= total_chunks:
                        break
                    continue

                chunk_id  = result.get("chunk_id", "")
                worker_id = result.get("worker_id", "unknown")

                if result.get("status") == "ok":
                    is_new = self.db.complete_chunk(
                        chunk_id, worker_id,
                        result.get("survivors",       []),
                        result.get("match_rates",     []),
                        result.get("skip_sequences",  []),
                        result.get("strategy_ids",    []),
                    )
                    if is_new:
                        completed += 1
                        n = len(result.get("survivors", []))
                        self.logger.info(
                            f"  ✅ Chunk {chunk_id}: {n:,} survivors [{worker_id}]"
                        )
                        if self._progress_writer:
                            try:
                                # Parse worker_id: "hostname:gpuN"
                                _parts = worker_id.split(":")
                                _host  = _parts[0] if _parts else worker_id
                                _gpuid = int(_parts[1].replace("gpu",""))                                          if len(_parts) > 1 else 0
                                _gpu_type = "RTX 3080 Ti"                                             if _host == "localhost" else "RX 6600"
                                _chunk_seeds = chunk_size
                                # elapsed not available here — use chunk_size/throughput
                                self._progress_writer.log_gpu_result(
                                    _host, _gpuid, _gpu_type,
                                    _chunk_seeds, 10.0, success=True
                                )
                            except Exception:
                                pass
                else:
                    self.logger.error(
                        f"  ❌ Chunk {chunk_id} error: "
                        f"{result.get('message','?')} [{worker_id}]"
                    )
                    c = self.db.count_by_status(run_id)
                    if c.get("done", 0) + c.get("failed", 0) >= total_chunks:
                        break

        finally:
            job_sock.close()
            result_sock.close()
            ctx.term()

        # Update dashboard progress
        if self._progress_writer:
            try:
                self._progress_writer.update_progress(
                    jobs_done=completed,
                    chunks_total=total_chunks
                )
            except Exception:
                pass

        # Aggregate
        all_survivors: List[int]   = []
        all_match_rates: List[float] = []
        all_skip_seqs: List        = []
        all_strat_ids: List[int]   = []

        for r in self.db.get_results(run_id):
            all_survivors.extend(  json.loads(r["survivors_json"]))
            all_match_rates.extend(json.loads(r["match_rates_json"]))
            all_skip_seqs.extend(  json.loads(r["skip_seqs_json"]))
            all_strat_ids.extend(  json.loads(r["strat_ids_json"]))

        counts        = self.db.count_by_status(run_id)
        failed_chunks = counts.get("failed", 0)
        if failed_chunks:
            self.logger.warning(
                f"[ZMQ] {prng_type}: {failed_chunks}/{total_chunks} failed"
            )

        try:
            os.makedirs(
                os.path.dirname(output_file) if os.path.dirname(output_file)
                else ".", exist_ok=True
            )
            with open(output_file, "w") as f:
                json.dump({
                    "survivors":      all_survivors,
                    "match_rates":    all_match_rates,
                    "skip_sequences": all_skip_seqs,
                    "strategy_ids":   all_strat_ids,
                    "total_tested":   total_seeds,
                    "survivor_count": len(all_survivors),
                    "prng_type":      prng_type,
                    "threshold":      threshold,
                    "failed_chunks":  failed_chunks,
                    "total_chunks":   total_chunks,
                }, f)
        except Exception as e:
            self.logger.warning(f"[ZMQ] Output file write failed: {e}")

        self.db.cleanup(run_id)

        return {
            "survivors":      all_survivors,
            "match_rates":    all_match_rates,
            "skip_sequences": all_skip_seqs,
            "strategy_ids":   all_strat_ids,
            "total_tested":   total_seeds,
            "survivor_count": len(all_survivors),
            "prng_type":      prng_type,
            "threshold":      threshold,
            "failed_chunks":  failed_chunks,
            "total_chunks":   total_chunks,
        }

    def shutdown(self):
        try:
            import zmq
            ctx  = zmq.Context()
            sock = ctx.socket(zmq.PUSH)
            sock.connect(f"tcp://localhost:{self.zmq_job_port}")
            total_workers = sum(n.get("gpu_count", 0) for n in self._nodes) + 2
            for _ in range(total_workers * 2):
                sock.send(json.dumps({"cmd": "shutdown"}).encode())
            time.sleep(1)
            sock.close()
            ctx.term()
        except Exception:
            pass


def run_trial_zmq_sqlite(
        coordinator_cfg, config, trial_number, prng_base, residues,
        total_seeds, forward_threshold, reverse_threshold, test_both_modes,
        dataset_path="", worker_pool_size=8,
        seed_cap_nvidia=5_000_000, seed_cap_amd=2_000_000,
        zmq_job_port=ZMQ_JOB_PORT, zmq_result_port=ZMQ_RESULT_PORT,
) -> Dict[str, Any]:
    """
    Identical signature and return contract to run_trial_persistent().
    _build_test_result_from_pw() works unchanged on the returned dict.
    """
    coord = ZMQSQLiteCoordinator(
        config_file=coordinator_cfg,
        seed_cap_amd=seed_cap_amd, seed_cap_nvidia=seed_cap_nvidia,
        worker_pool_size=worker_pool_size,
        zmq_job_port=zmq_job_port, zmq_result_port=zmq_result_port,
    )

    ws         = config.window_size
    off        = config.offset
    sessions   = (list(config.sessions) if hasattr(config, "sessions")
                  else ["midday", "evening"])
    skip_range = ([config.skip_min, config.skip_max]
                  if hasattr(config, "skip_min") else [0, 147])

    try:
        print(f"\n    Running FORWARD sieve ({prng_base}) [ZMQ-SQLITE]...")
        fwd = coord.run_sieve_pass(
            prng_type=prng_base, residues=residues, total_seeds=total_seeds,
            threshold=forward_threshold, window_size=ws, dataset_path=dataset_path,
            output_file=f"results/zmq_fwd_{ws}_{off}_t{trial_number}.json",
            offset=off, sessions=sessions, skip_range=skip_range,
        )
        fwd_map = dict(zip(fwd.get("survivors",[]), fwd.get("match_rates",[])))
        print(f"      Forward: {len(fwd_map):,} survivors")

        if not fwd_map:
            return {
                "pruned": True, "reason": "forward_zero",
                "bidirectional_count": 0,
                "bidirectional_constant": set(), "bidirectional_variable": set(),
                "forward_records": [], "reverse_records": [],
                "forward_records_hybrid": [], "reverse_records_hybrid": [],
                "forward_map": {}, "reverse_map": {},
            }

        print(f"    Running REVERSE sieve ({prng_base}_reverse) [ZMQ-SQLITE]...")
        rev = coord.run_sieve_pass(
            prng_type=prng_base+"_reverse", residues=residues,
            total_seeds=total_seeds, threshold=reverse_threshold,
            window_size=ws, dataset_path=dataset_path,
            output_file=f"results/zmq_rev_{ws}_{off}_t{trial_number}.json",
            offset=off, sessions=sessions, skip_range=skip_range,
        )
        rev_map = dict(zip(rev.get("survivors",[]), rev.get("match_rates",[])))
        print(f"      Reverse: {len(rev_map):,} survivors")

        bidi_const = set(fwd_map) & set(rev_map)
        print(f"      Bidirectional (constant): {len(bidi_const):,}")

        bidi_var = set()
        fwd_h_rec, rev_h_rec = [], []

        if test_both_modes and not prng_base.endswith("_hybrid"):
            try:
                from hybrid_strategy import get_strategy as _gs
                _s = _gs("balanced_hybrid")
                _strats = [_s.to_dict() if hasattr(_s,"to_dict") else vars(_s)]
            except Exception:
                _strats = None

            prng_h = f"{prng_base}_hybrid"
            print(f"    Running FORWARD sieve ({prng_h}) [ZMQ-SQLITE]...")
            fwd_h = coord.run_sieve_pass(
                prng_type=prng_h, residues=residues, total_seeds=total_seeds,
                threshold=forward_threshold, window_size=ws,
                dataset_path=dataset_path,
                output_file=f"results/zmq_fwdh_{ws}_{off}_t{trial_number}.json",
                offset=off, sessions=sessions, skip_range=skip_range,
                strategies=_strats,
            )
            fwd_h_map = dict(zip(
                fwd_h.get("survivors",[]), fwd_h.get("match_rates",[])
            ))
            print(f"      Forward (variable): {len(fwd_h_map):,} survivors")

            if fwd_h_map:
                print(f"    Running REVERSE sieve ({prng_h}_reverse) [ZMQ-SQLITE]...")
                rev_h = coord.run_sieve_pass(
                    prng_type=f"{prng_h}_reverse", residues=residues,
                    total_seeds=total_seeds, threshold=reverse_threshold,
                    window_size=ws, dataset_path=dataset_path,
                    output_file=f"results/zmq_revh_{ws}_{off}_t{trial_number}.json",
                    offset=off, sessions=sessions, skip_range=skip_range,
                    strategies=_strats,
                )
                rev_h_map = dict(zip(
                    rev_h.get("survivors",[]), rev_h.get("match_rates",[])
                ))
                print(f"      Reverse (variable): {len(rev_h_map):,} survivors")
                bidi_var  = set(fwd_h_map) & set(rev_h_map)
                fwd_h_rec = [{"seed":s,"match_rate":fwd_h_map[s]} for s in fwd_h_map]
                rev_h_rec = [{"seed":s,"match_rate":rev_h_map[s]} for s in rev_h_map]

            print(f"      Bidirectional (variable): {len(bidi_var):,}")

        total_bidi = len(bidi_const) + len(bidi_var)
        print(f"      Total bidirectional: {total_bidi:,}")

        # Update dashboard trial stats (mirrors run_trial_persistent)
        if coord._progress_writer:
            try:
                coord._progress_writer.update_trial_stats(
                    trial_num=trial_number,
                    forward_survivors=len(fwd_map),
                    reverse_survivors=len(rev_map),
                    bidirectional=total_bidi,
                    best_bidirectional=total_bidi,
                    config_desc=f"W{ws}_O{off}",
                    accumulated_forward=len(fwd_map),
                    accumulated_reverse=len(rev_map),
                    accumulated_bidirectional=total_bidi,
                )
            except Exception:
                pass

        return {
            "pruned":                 False,
            "bidirectional_count":    total_bidi,
            "bidirectional_constant": bidi_const,
            "bidirectional_variable": bidi_var,
            "forward_map":            fwd_map,
            "reverse_map":            rev_map,
            "forward_records":  [{"seed":s,"match_rate":fwd_map[s]} for s in fwd_map],
            "reverse_records":  [{"seed":s,"match_rate":rev_map[s]}  for s in rev_map],
            "forward_records_hybrid":  fwd_h_rec,
            "reverse_records_hybrid":  rev_h_rec,
        }

    finally:
        coord.shutdown()
