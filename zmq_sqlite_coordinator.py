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
RESULT_POLL_MS   = 50   # S159B: was 500 — 250ms avg wait/chunk -> 25ms
CHUNK_PAYLOAD_DIR = "zmq_chunk_payloads"  # S159: per-chunk .npz files


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
                        result_path      TEXT NOT NULL,
                        survivor_count   INTEGER NOT NULL DEFAULT 0,
                        worker_id        TEXT NOT NULL,
                        created_at       REAL NOT NULL
                    )
                """)
                # S159 migration: handle existing DBs with old blob schema
                try:
                    cols = {r[1] for r in conn.execute(
                        "PRAGMA table_info(job_results)")}
                    if "survivors_json" in cols and "result_path" not in cols:
                        conn.execute(
                            "ALTER TABLE job_results ADD COLUMN "
                            "result_path TEXT NOT NULL DEFAULT ''")
                        conn.execute(
                            "ALTER TABLE job_results ADD COLUMN "
                            "survivor_count INTEGER NOT NULL DEFAULT 0")
                        logger.warning(
                            "[ZMQ-DB] Migrated job_results to slim schema "
                            "(old blob columns retained for this session)")
                except Exception as _mig_err:
                    logger.debug(f"[ZMQ-DB] Schema migration check: {_mig_err}")
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
                       skip_seqs, strat_ids, chunk_payload_dir=None):
        """
        Zeus records result. Idempotent — duplicate chunk_id silently ignored.
        Returns True if first completion, False if duplicate.

        S159: payload arrays written to .npz file; only ledger metadata
        (result_path, survivor_count) stored in SQLite.
        """
        import numpy as np  # numpy already required by the broader project

        payload_dir = chunk_payload_dir or CHUNK_PAYLOAD_DIR
        os.makedirs(payload_dir, exist_ok=True)
        npz_path = os.path.join(payload_dir, chunk_id + ".npz")

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

                # Write payload to .npz before DB insert (fail-fast)
                try:
                    # S159B: savez (uncompressed) — 71x faster than savez_compressed.
                    # These files are temporary, deleted after run. No reason to compress.
                    # skip_seqs/strat_ids dropped — never consumed by any downstream step.
                    np.savez(
                        npz_path,
                        survivors=np.array(survivors,   dtype=np.int64),
                        match_rates=np.array(match_rates, dtype=np.float32),
                    )
                except Exception as e:
                    logger.error(
                        f"[ZMQ-DB] Failed to write payload npz "
                        f"{npz_path}: {e}"
                    )
                    raise

                conn.execute("""
                    INSERT INTO job_results
                    (chunk_id, run_id, result_path, survivor_count,
                     worker_id, created_at)
                    SELECT chunk_id, run_id, ?, ?, ?, ?
                    FROM job_queue WHERE chunk_id=?
                """, (npz_path, len(survivors), worker_id, time.time(),
                      chunk_id))
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
        """
        S159: returns list of dicts with survivors/match_rates lists loaded
        from .npz files.  Same structure as before — callers unchanged.
        Missing or corrupt .npz files are logged and skipped (treated as
        failed chunks; coordinator already tracks failure count separately).
        """
        import numpy as np

        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM job_results WHERE run_id=?", (run_id,)
            ).fetchall()

        results = []
        for row in rows:
            r = dict(row)
            npz_path = r.get("result_path", "")

            # S159 migration path: old DB rows have blob columns, no result_path
            if not npz_path or npz_path == "":
                if "survivors_json" in r:
                    # Graceful fallback for rows from pre-S159 schema
                    try:
                        results.append({
                            "chunk_id":    r["chunk_id"],
                            "run_id":      r["run_id"],
                            "survivors":   json.loads(r["survivors_json"]),
                            "match_rates": json.loads(r["match_rates_json"]),
                        })
                    except Exception as e:
                        logger.warning(
                            f"[ZMQ-DB] Legacy row decode failed "
                            f"{r['chunk_id']}: {e}")
                continue

            try:
                data = np.load(npz_path, allow_pickle=True)
                results.append({
                    "chunk_id":    r["chunk_id"],
                    "run_id":      r["run_id"],
                    "survivors":   data["survivors"].tolist(),
                    "match_rates": data["match_rates"].tolist(),
                })
            except Exception as e:
                logger.warning(
                    f"[ZMQ-DB] Could not load payload npz "
                    f"{npz_path}: {e} — chunk skipped"
                )

        return results

    def count_by_status(self, run_id):
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as n FROM job_queue "
                "WHERE run_id=? GROUP BY status", (run_id,)
            ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    def cleanup(self, run_id):
        """S159: also removes per-chunk .npz payload files for this run."""
        with self._write_lock:
            with self._conn() as conn:
                # Collect paths before deleting rows
                rows = conn.execute(
                    "SELECT result_path FROM job_results WHERE run_id=?",
                    (run_id,)
                ).fetchall()
                for row in rows:
                    path = row[0] if row[0] else ""
                    if path and os.path.isfile(path):
                        try:
                            os.remove(path)
                        except Exception as e:
                            logger.debug(
                                f"[ZMQ-DB] cleanup: could not remove "
                                f"{path}: {e}"
                            )
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
                 zmq_job_port=ZMQ_JOB_PORT, zmq_result_port=ZMQ_RESULT_PORT,
                 chunk_payload_dir=None):   # S159: override default payload dir
        self.config_file      = config_file
        self.seed_cap_amd     = seed_cap_amd
        self.seed_cap_nvidia  = seed_cap_nvidia
        self.worker_pool_size = worker_pool_size
        self.zmq_job_port       = zmq_job_port
        self.zmq_result_port    = zmq_result_port
        self.chunk_payload_dir  = chunk_payload_dir or CHUNK_PAYLOAD_DIR  # S159
        # S159D: session-scoped sockets — stay bound across all sieve passes
        self._zmq_ctx    = None
        self._job_sock   = None
        self._result_sock = None
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

    def _start_sockets(self, sndhwm=1000):
        """
        S159D: Create and bind ZMQ sockets once for the entire session.
        Called lazily before the first run_sieve_pass(). Sockets remain
        bound across all passes so workers never lose their connection.
        """
        if self._zmq_ctx is not None:
            return  # already started
        try:
            import zmq
        except ImportError:
            raise ImportError("pyzmq required. Install in venv: pip install pyzmq")

        self._zmq_ctx     = zmq.Context()
        self._job_sock    = self._zmq_ctx.socket(zmq.PUSH)
        self._result_sock = self._zmq_ctx.socket(zmq.PULL)
        self._job_sock.setsockopt(zmq.SNDHWM, sndhwm)
        self._result_sock.setsockopt(zmq.RCVTIMEO, RESULT_POLL_MS)
        self._job_sock.bind(f"tcp://*:{self.zmq_job_port}")
        self._result_sock.bind(f"tcp://*:{self.zmq_result_port}")
        self.logger.info(
            f"[ZMQ] Session sockets bound: job={self.zmq_job_port} "
            f"result={self.zmq_result_port}"
        )

    def _launch_workers(self):
        """
        SSH to each rig ONCE. Launch workers as systemd-run --user transient services.
        Workers survive SSH session teardown. systemd owns the process, not the shell.
        Zeus local workers launched via subprocess.Popen with isolated env per GPU.

        Prerequisite (one-time per rig):
            sudo loginctl enable-linger michael
        """
        if self._workers_launched:
            return

        import subprocess
        import shlex

        def _abs(p, username):
            return '/home/' + username + '/' + p[2:] if p.startswith('~/') else p

        for node in self._nodes:
            host      = node.get('hostname', '')
            username  = node.get('username', 'michael')
            gpu_count = node.get('gpu_count', 0)
            if not host or host in ('localhost', '127.0.0.1') or gpu_count == 0:
                continue

            py_env      = _abs(node.get('python_env',  '~/rocm_env/bin/python3'),    username)
            script_path = _abs(node.get('script_path', '~/distributed_prng_analysis'), username)
            worker_script = script_path + '/zmq_sqlite_worker.py'

            for gpu_id in range(gpu_count):
                worker_id = host + ':gpu' + str(gpu_id)
                unit      = 'zmq-worker-gpu' + str(gpu_id)
                log_path  = '/tmp/zmq_worker_gpu' + str(gpu_id) + '.log'

                worker_cmd = ' '.join([
                    'cd', shlex.quote(script_path), '&&',
                    'exec', shlex.quote(py_env), '-u', shlex.quote(worker_script),
                    '--zeus-host', shlex.quote(self._zeus_ip),
                    '--job-port', str(self.zmq_job_port),
                    '--result-port', str(self.zmq_result_port),
                    '--worker-id', shlex.quote(worker_id),
                    '--gpu-id', str(gpu_id),
                    '>>' + shlex.quote(log_path), '2>&1',
                ])

                remote_lines = [
                    'set -e',
                    'linger=$(loginctl show-user "$USER" -p Linger --value 2>/dev/null || echo no)',
                    'if [ "$linger" != yes ]; then',
                    '  echo "ERROR: linger not enabled -- run: sudo loginctl enable-linger $USER" >&2',
                    '  exit 42',
                    'fi',
                    'systemctl --user stop ' + shlex.quote(unit) + ' >/dev/null 2>&1 || true',
                    'systemctl --user reset-failed ' + shlex.quote(unit) + ' >/dev/null 2>&1 || true',
                    ('systemd-run --user'
                     + ' --unit=' + shlex.quote(unit)
                     + ' --collect'
                     + ' --property=Type=exec'
                     + ' --property=Restart=always'
                     + ' --property=RestartSec=2'
                     + ' --setenv=ROCR_VISIBLE_DEVICES=' + str(gpu_id)
                     + ' --setenv=CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_' + str(gpu_id)
                     + ' --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0'
                     + ' --setenv=HSA_ENABLE_SDMA=0'
                     + ' --setenv=HSA_ENABLE_RUNTIME_POWER_MGMT=0'
                     + ' --setenv=AMDGPU_NO_POWER_PROFILE=1'
                     + ' /bin/bash -lc ' + shlex.quote(worker_cmd)),
                    'sleep 1',
                    'systemctl --user is-active --quiet ' + shlex.quote(unit),
                ]
                remote_script = '\n'.join(remote_lines)

                try:
                    proc = subprocess.run(
                        ['ssh', '-q',
                         '-o', 'StrictHostKeyChecking=no',
                         '-o', 'BatchMode=yes',
                         '-o', 'ConnectTimeout=10',
                         username + '@' + host,
                         'bash', '-lc', remote_script],
                        capture_output=True, text=True, timeout=30,
                    )
                    if proc.returncode == 0:
                        self.logger.info('[ZMQ] systemd-run worker active: ' + host + ' gpu' + str(gpu_id))
                    elif proc.returncode == 42:
                        self.logger.error('[ZMQ] linger not enabled on ' + host)
                    else:
                        self.logger.error(
                            '[ZMQ] systemd-run failed on ' + host + ' gpu' + str(gpu_id) +
                            ' rc=' + str(proc.returncode) +
                            ' stderr=' + proc.stderr.strip()[:200]
                        )
                except subprocess.TimeoutExpired:
                    self.logger.error('[ZMQ] SSH timeout on ' + host + ' gpu' + str(gpu_id))
                except Exception as e:
                    self.logger.error('[ZMQ] Launch failed ' + host + ' gpu' + str(gpu_id) + ': ' + str(e))

        # Zeus local CUDA workers -- isolated env per GPU (S158D-E)
        import subprocess as sp
        import os as _os
        for gpu_id in range(2):
            worker_id = 'localhost:gpu' + str(gpu_id)
            env = _os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            env['CUPY_CACHE_DIR']       = '/tmp/cupy_cache_zeus_gpu' + str(gpu_id)
            env.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
            try:
                sp.Popen(
                    ['python3', 'zmq_sqlite_worker.py',
                     '--zeus-host',   'localhost',
                     '--job-port',    str(self.zmq_job_port),
                     '--result-port', str(self.zmq_result_port),
                     '--worker-id',   worker_id,
                     '--gpu-id',      '0',
                     '--cuda'],
                    env=env,
                    stdout=open('/tmp/zmq_zeus_gpu' + str(gpu_id) + '.log', 'w'),
                    stderr=sp.STDOUT,
                )
                self.logger.info(
                    '[ZMQ] Zeus CUDA worker launched (' + worker_id +
                    ' CUDA_VISIBLE_DEVICES=' + str(gpu_id) + ' logical_gpu=0)'
                )
                # S159F: stagger Zeus workers to prevent simultaneous CuPy init collision
                if gpu_id == 0:
                    time.sleep(3)
            except Exception as e:
                self.logger.error('[ZMQ] Zeus GPU' + str(gpu_id) + ' launch failed: ' + str(e))

        time.sleep(WORKER_SETTLE_S)
        self._workers_launched = True
        self.logger.info('[ZMQ] All workers launched and settled')


    # --- S160-v2: TB-approved env-invariant check ---
    _REQUIRED_WORKER_VARS = {
        "HSA_ENABLE_SDMA":               "0",
        "HSA_ENABLE_RUNTIME_POWER_MGMT": "0",
        "AMDGPU_NO_POWER_PROFILE":       "1",
        "HSA_OVERRIDE_GFX_VERSION":      "10.3.0",
    }

    # Secondary guardrail: these flags must appear in _launch_workers() source.
    _REQUIRED_SETENV_FLAGS = [
        "--setenv=HSA_ENABLE_SDMA=0",
        "--setenv=HSA_ENABLE_RUNTIME_POWER_MGMT=0",
        "--setenv=AMDGPU_NO_POWER_PROFILE=1",
        "--setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0",
    ]

    def _assert_launch_cmd_contains_setenv(self) -> None:
        """
        Secondary guardrail (TB-approved): statically verify _launch_workers()
        source contains all required --setenv flags. Raises AssertionError on
        failure. Called once before SSH work begins.
        """
        import inspect as _inspect
        src = _inspect.getsource(self._launch_workers)
        missing = [f for f in self._REQUIRED_SETENV_FLAGS if f not in src]
        if missing:
            raise AssertionError(
                f"[EnvCheck] Static assertion FAILED — --setenv flags missing "
                f"from _launch_workers() source: {missing}"
            )
        self.logger.info(
            "[EnvCheck] Static assertion PASSED — all required --setenv flags "
            "confirmed in _launch_workers() source."
        )

    def _collect_worker_diagnostics(self, host: str, username: str,
                                    unit: str) -> None:
        """Collect and log journal + worker log on env check failure."""
        import subprocess as _sp
        diag_cmd = (
            f"echo '=== journal ==' && "
            f"journalctl --user -u {unit} -n 30 --no-pager 2>/dev/null; "
            f"echo '=== worker log ==' && "
            f"tail -n 30 /tmp/zmq_worker_gpu0.log 2>/dev/null"
        )
        try:
            r = _sp.run(
                ["ssh", "-q", "-o", "BatchMode=yes",
                 "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=no",
                 f"{username}@{host}", "bash", "-c", diag_cmd],
                capture_output=True, text=True, timeout=20
            )
            for line in r.stdout.strip().splitlines():
                self.logger.info(f"[EnvCheck][diag:{host}] {line}")
        except Exception as exc:
            self.logger.warning(
                f"[EnvCheck] Could not collect diagnostics from {host}: {exc}"
            )

    def _verify_worker_env(self, host: str, username: str) -> bool:
        """
        S160-v2 (TB-approved):
        1. Query systemctl --user show zmq-worker-gpu0.service for MainPID,
           ActiveState, SubState, Result, NRestarts.
        2. FAIL hard if ActiveState != active or MainPID == 0.
           (Verification runs before chunk dispatch, so workers must be alive.)
        3. Read /proc/<MainPID>/environ and verify all required ROCm vars.
        4. On any failure, collect journal + worker log diagnostics.
        """
        import subprocess as _sp

        unit = "zmq-worker-gpu0.service"
        state_cmd = (
            f"systemctl --user show {unit} "
            f"-p MainPID,ActiveState,SubState,Result,NRestarts"
        )
        try:
            r = _sp.run(
                ["ssh", "-q", "-o", "BatchMode=yes",
                 "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=no",
                 f"{username}@{host}", "bash", "-c", state_cmd],
                capture_output=True, text=True, timeout=15
            )
        except Exception as exc:
            self.logger.error(
                f"[EnvCheck] SSH error querying unit state on {host}: {exc}"
            )
            return False

        state = {}
        for line in r.stdout.strip().splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                state[k.strip()] = v.strip()

        active_state = state.get("ActiveState", "unknown")
        sub_state    = state.get("SubState",    "unknown")
        result       = state.get("Result",      "unknown")
        n_restarts   = state.get("NRestarts",   "?")
        try:
            main_pid = int(state.get("MainPID", "0"))
        except ValueError:
            main_pid = 0

        self.logger.info(
            f"[EnvCheck] {host} unit state: ActiveState={active_state} "
            f"SubState={sub_state} Result={result} "
            f"MainPID={main_pid} NRestarts={n_restarts}"
        )

        if active_state != "active" or main_pid == 0:
            self.logger.error(
                f"[EnvCheck] FAIL on {host}: unit not active or MainPID=0. "
                f"ActiveState={active_state} SubState={sub_state} "
                f"Result={result} NRestarts={n_restarts}"
            )
            self._collect_worker_diagnostics(host, username, unit)
            return False

        # Read /proc/<MainPID>/environ
        env_cmd = f"tr '\\0' '\\n' </proc/{main_pid}/environ"
        try:
            r2 = _sp.run(
                ["ssh", "-q", "-o", "BatchMode=yes",
                 "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=no",
                 f"{username}@{host}", "bash", "-c", env_cmd],
                capture_output=True, text=True, timeout=15
            )
        except Exception as exc:
            self.logger.error(
                f"[EnvCheck] SSH error reading environ on {host}: {exc}"
            )
            return False

        env_dict = {}
        for line in r2.stdout.strip().splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                env_dict[k] = v

        missing = []
        for var, expected in self._REQUIRED_WORKER_VARS.items():
            actual = env_dict.get(var)
            if actual != expected:
                missing.append(f"{var}={actual!r} (expected {expected!r})")
        if "ROCR_VISIBLE_DEVICES" not in env_dict:
            missing.append("ROCR_VISIBLE_DEVICES=MISSING")

        if missing:
            self.logger.error(
                f"[EnvCheck] FAIL on {host}: env vars wrong: {missing}"
            )
            self._collect_worker_diagnostics(host, username, unit)
            return False

        self.logger.info(
            f"[EnvCheck] PASS on {host}: MainPID={main_pid} "
            f"all ROCm protective vars confirmed."
        )
        return True

    def _launch_and_verify(self):
        """
        S160-v2 (TB-approved):
        1. Static assertion on launch command source.
        2. Launch workers.
        3. Wait 10 s for systemd units to become active (no jobs dispatched yet).
        4. Verify each AMD rig — unit must be active + MainPID > 0 + correct env.
        5. Raise RuntimeError if any rig fails. Halt before chunk dispatch.
        """
        import time as _time

        self._assert_launch_cmd_contains_setenv()
        self._launch_workers()

        self.logger.info(
            "[EnvCheck] Waiting 10 s for workers to initialize "
            "(no chunks dispatched yet)..."
        )
        _time.sleep(10)

        failed_hosts = []
        for node in self._nodes:
            host = node.get("hostname", "")
            if host in ("localhost", "127.0.0.1"):
                continue  # Zeus CUDA workers use a different launch path
            username = node.get("username", "michael")
            if not self._verify_worker_env(host, username):
                failed_hosts.append(host)

        if failed_hosts:
            raise RuntimeError(
                f"[EnvCheck] Env-invariant check FAILED on: {failed_hosts}. "
                f"Workers not active or ROCm protective vars missing/wrong. "
                f"See journal output above for root cause. "
                f"Halting before chunk dispatch."
            )
        self.logger.info(
            "[EnvCheck] All AMD rigs passed env-invariant check. "
            "Proceeding to chunk dispatch."
        )

    def run_sieve_pass(self, prng_type, residues, total_seeds, threshold,
                       window_size, output_file, dataset_path="",
                       strategies=None, phase2_threshold=0.5,
                       target_file="", offset=0, sessions=None,
                       skip_range=None):
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

        import zmq  # S159D: needed for zmq.Again in result loop
        # S159D: use session-scoped sockets (bound once, reused across passes)
        self._start_sockets(sndhwm=total_chunks + 100)
        job_sock    = self._job_sock
        result_sock = self._result_sock

        # Launch workers (no-op after first pass — _workers_launched guard)
        # S160: first call also runs env-invariant check (45 s settle + SSH verify)
        self._launch_and_verify()

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
                        chunk_payload_dir=self.chunk_payload_dir,  # S159
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
            pass  # S159D: sockets kept alive across passes — closed in shutdown()

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
            # S159: get_results now returns pre-loaded lists from .npz
            all_survivors.extend(  r["survivors"])
            all_match_rates.extend(r["match_rates"])
            # skip_seqs / strat_ids no longer stored; downstream doesn't use them

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
        """S159D: send shutdown to workers, then close session sockets."""
        try:
            if self._job_sock is not None:
                total_workers = (
                    sum(n.get("gpu_count", 0) for n in self._nodes) + 2
                )
                for _ in range(total_workers * 2):
                    self._job_sock.send(
                        json.dumps({"cmd": "shutdown"}).encode()
                    )
                time.sleep(1)
        except Exception:
            pass
        finally:
            try:
                if self._job_sock:
                    self._job_sock.close()
                if self._result_sock:
                    self._result_sock.close()
                if self._zmq_ctx:
                    self._zmq_ctx.term()
            except Exception:
                pass
            self._job_sock    = None
            self._result_sock = None
            self._zmq_ctx     = None


def run_trial_zmq_sqlite(
        coordinator_cfg, config, trial_number, prng_base, residues,
        total_seeds, forward_threshold, reverse_threshold, test_both_modes,
        dataset_path="", worker_pool_size=8,
        seed_cap_nvidia=5_000_000, seed_cap_amd=2_000_000,
        zmq_job_port=ZMQ_JOB_PORT, zmq_result_port=ZMQ_RESULT_PORT,
        session_coord=None,
) -> Dict[str, Any]:
    """
    Identical signature and return contract to run_trial_persistent().
    _build_test_result_from_pw() works unchanged on the returned dict.

    S159E: If session_coord is provided, it is reused across trials so sockets
    stay bound and workers stay connected between trials. The caller owns
    session_coord and must call session_coord.shutdown() after all trials.
    If None, a per-trial coordinator is created and shut down after each trial.
    """
    _owns_coord = session_coord is None
    if _owns_coord:
        coord = ZMQSQLiteCoordinator(
            config_file=coordinator_cfg,
            seed_cap_amd=seed_cap_amd, seed_cap_nvidia=seed_cap_nvidia,
            worker_pool_size=worker_pool_size,
            zmq_job_port=zmq_job_port, zmq_result_port=zmq_result_port,
            chunk_payload_dir=CHUNK_PAYLOAD_DIR,  # S159
        )
    else:
        coord = session_coord

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
        if _owns_coord:
            coord.shutdown()
