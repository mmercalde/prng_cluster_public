"""
zmq_sqlite_worker.py  —  S158D v2: ZMQ GPU Sieve Worker (Rig Side)
====================================================================
Launched ONCE per GPU via SSH. Runs independently via ZMQ TCP.
No persistent SSH pipes. No heartbeats. Zeus is sole SQLite writer.

Worker identity (--worker-id): "hostname:gpuN"
  - Explicit, stable, logged with every result
  - Zeus uses this for SQLite claimed_by tracking

Launch (done automatically by ZMQSQLiteCoordinator._launch_workers()):
  ROCR_VISIBLE_DEVICES=0 python3 zmq_sqlite_worker.py \
    --zeus-host 192.168.3.1 --job-port 5557 --result-port 5558 \
    --worker-id 192.168.3.162:gpu0 --gpu-id 0

Install pyzmq in existing venv:
  source ~/rocm_env/bin/activate && pip install pyzmq
"""

import argparse
import json
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("ZMQWorker")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zeus-host",   default="localhost")
    p.add_argument("--job-port",    type=int, default=5557)
    p.add_argument("--result-port", type=int, default=5558)
    p.add_argument("--worker-id",   required=True,
                   help="Stable identity e.g. '192.168.3.162:gpu0'")
    p.add_argument("--gpu-id",      type=int, default=0)
    p.add_argument("--cuda",        action="store_true")
    return p.parse_args()


def run_sieve_job(job: dict, gpu_id: int, use_cuda: bool, worker_id: str) -> dict:
    """
    Execute one sieve chunk using existing sieve_filter.py infrastructure.
    Same GPU kernel code as PWC — only IPC changes.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    if not use_cuda:
        os.environ.setdefault("ROCR_VISIBLE_DEVICES",     str(gpu_id))
        os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        from sieve_filter import execute_sieve_job

        sieve_job = {
            "job_id":              job["chunk_id"],
            "search_type":         "residue_sieve",
            "dataset_path":        job["dataset_path"],
            "seed_start":          job["seed_start"],
            "seed_end":            job["seed_end"],
            "window_size":         job["window_size"],
            "min_match_threshold": job["threshold"],
            "skip_range":          [job["skip_min"], job["skip_max"]],
            "offset":              job["chunk_offset"],
            "sessions":            json.loads(job["sessions_json"]),
            "prng_families":       [job["prng_type"]],
            "strategies":          (json.loads(job["strategies_json"])
                                    if job.get("strategies_json") else None),
            "hybrid":              bool(job["is_hybrid"]),
            "gpu_id":              0,  # remapped by ROCR_VISIBLE_DEVICES
        }

        result = execute_sieve_job(sieve_job)

        if result.get("status") == "ok":
            inner = result.get("result", {})
            if inner.get("format") == "slim_v1":
                survivors   = [int(s) for s in inner.get("seeds", [])]
                match_rates = list(inner.get("match_rates", []))
                skip_seqs   = list(inner.get("skip_sequences",
                                             [[]] * len(survivors)))
                strat_ids   = list(inner.get("strategy_ids",
                                             [0] * len(survivors)))
            else:
                raw         = inner.get("survivors", [])
                survivors   = [s["seed"]          if isinstance(s, dict)
                               else int(s) for s in raw]
                match_rates = [s.get("match_rate", 0.5) if isinstance(s, dict)
                               else 0.5 for s in raw]
                skip_seqs   = [s.get("skip_sequence", []) if isinstance(s, dict)
                               else [] for s in raw]
                strat_ids   = [s.get("strategy_id",    0) if isinstance(s, dict)
                               else 0 for s in raw]

            return {
                "chunk_id":       job["chunk_id"],
                "worker_id":      worker_id,
                "status":         "ok",
                "survivors":      survivors,
                "match_rates":    match_rates,
                "skip_sequences": skip_seqs,
                "strategy_ids":   strat_ids,
            }
        else:
            return {
                "chunk_id":       job["chunk_id"],
                "worker_id":      worker_id,
                "status":         "error",
                "message":        result.get("message", "unknown"),
                "survivors":      [], "match_rates":    [],
                "skip_sequences": [], "strategy_ids":   [],
            }

    except Exception as e:
        import traceback
        return {
            "chunk_id":       job["chunk_id"],
            "worker_id":      worker_id,
            "status":         "error",
            "message":        str(e),
            "traceback":      traceback.format_exc(),
            "survivors":      [], "match_rates":    [],
            "skip_sequences": [], "strategy_ids":   [],
        }


def main():
    args      = parse_args()
    worker_id = args.worker_id
    gpu_id    = args.gpu_id
    use_cuda  = args.cuda

    try:
        import zmq
    except ImportError:
        log.error(
            "pyzmq not installed. "
            "Run: source ~/rocm_env/bin/activate && pip install pyzmq"
        )
        sys.exit(1)

    ctx         = zmq.Context()
    job_sock    = ctx.socket(zmq.PULL)
    result_sock = ctx.socket(zmq.PUSH)
    job_sock.connect(f"tcp://{args.zeus_host}:{args.job_port}")
    result_sock.connect(f"tcp://{args.zeus_host}:{args.result_port}")
    job_sock.setsockopt(zmq.RCVTIMEO, 5000)

    log.info(
        f"Worker ready — id={worker_id} gpu={gpu_id} cuda={use_cuda} "
        f"zeus={args.zeus_host}:{args.job_port}"
    )

    jobs_done = 0
    try:
        while True:
            try:
                msg = job_sock.recv()
            except zmq.Again:
                continue

            job = json.loads(msg.decode())

            if job.get("cmd") == "shutdown":
                log.info(f"{worker_id}: shutdown received")
                break

            chunk_id = job.get("chunk_id", "?")
            log.info(
                f"{worker_id}: chunk {chunk_id} "
                f"seeds {job.get('seed_start',0):,}"
                f"→{job.get('seed_end',0):,} "
                f"prng={job.get('prng_type','?')}"
            )

            t0     = time.time()
            result = run_sieve_job(job, gpu_id, use_cuda, worker_id)
            elapsed = time.time() - t0

            n = len(result.get("survivors", []))
            log.info(
                f"{worker_id}: chunk {chunk_id} done "
                f"{n:,} survivors {elapsed:.1f}s"
            )

            # Send result back to Zeus via ZMQ (JSON only — no pickle)
            result_sock.send(json.dumps(result).encode())
            jobs_done += 1

    except KeyboardInterrupt:
        log.info(f"{worker_id}: interrupted")
    finally:
        job_sock.close()
        result_sock.close()
        ctx.term()
        log.info(f"{worker_id}: exiting after {jobs_done} jobs")


if __name__ == "__main__":
    main()
