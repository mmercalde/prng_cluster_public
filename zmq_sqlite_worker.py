"""
zmq_sqlite_worker.py  —  S158D v3: ZMQ GPU Sieve Worker (Rig Side)
====================================================================
Launched ONCE per GPU via SSH. Runs independently via ZMQ TCP.
No persistent SSH pipes. Zeus is sole SQLite writer.

Worker identity (--worker-id): "hostname:gpuN"

Launch (done automatically by ZMQSQLiteCoordinator._launch_workers()):
  ROCR_VISIBLE_DEVICES=0 python3 zmq_sqlite_worker.py \
    --zeus-host 192.168.3.1 --job-port 5557 --result-port 5558 \
    --worker-id 192.168.3.162:gpu0 --gpu-id 0

Install pyzmq in existing venv:
  source ~/rocm_env/bin/activate && pip install pyzmq

v3 fixes vs v2:
  1. execute_sieve_job(sieve_job, gpu_id) — was missing gpu_id positional arg
  2. result.get("success") not result.get("status")=="ok" — wrong key in v2
  3. Survivor dicts: "best_skip" (standard) or "skip_sequence" (hybrid)
     v2 was looking for nested "result"/"slim_v1" format that does not exist
  4. Defensive logging — full traceback in worker log for every failure path
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback as tb

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
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


def run_sieve_job(job: dict, gpu_id: int, use_cuda: bool, worker_id: str, execute_sieve_job=None) -> dict:
    """
    Execute one sieve chunk using sieve_filter.execute_sieve_job().

    execute_sieve_job(job, gpu_id) returns:
      {"success": True,  "survivors": [{"seed":int, "match_rate":float,
                                        "best_skip":int (standard) or
                                        "skip_sequence":list, "strategy_id":int (hybrid)},
                                        ...], ...}
      {"success": False, "error": str, "traceback": str, ...}

    We log the full traceback locally to /tmp/zmq_worker_gpuN.log on every
    failure path so Zeus errors are never opaque.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # S159F: env vars now set at worker startup before sieve_filter import
    chunk_id = job.get("chunk_id", "unknown_chunk")




    # S159F: sieve_filter imported once at worker startup — eliminates per-chunk CUDA context collision
    if execute_sieve_job is None:
        return _error_result(job.get("chunk_id","?"), worker_id,
                             "execute_sieve_job not provided to run_sieve_job", "")





    # ── Stage 2: build job dict ───────────────────────────────────────────────
    try:
        sieve_job = {
            "job_id":              chunk_id,
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
        }
    except Exception as e:
        full_tb = tb.format_exc()
        log.error(f"{worker_id}: JOB BUILD FAILED chunk={chunk_id} — {e}\n{full_tb}")
        return _error_result(chunk_id, worker_id,
                             f"job build failed: {e}", full_tb)

    # ── Stage 3: execute sieve ────────────────────────────────────────────────
    try:
        # gpu_id=0 because ROCR_VISIBLE_DEVICES remaps the device index
        result = execute_sieve_job(sieve_job, 0)
    except Exception as e:
        full_tb = tb.format_exc()
        log.error(f"{worker_id}: EXECUTE FAILED chunk={chunk_id} — {e}\n{full_tb}")
        return _error_result(chunk_id, worker_id,
                             f"execute_sieve_job raised: {e}", full_tb)

    # ── Stage 4: check success flag ───────────────────────────────────────────
    if not result.get("success", False):
        err_msg  = result.get("error",     "no error field")
        err_tb   = result.get("traceback", "no traceback field")
        full_msg = f"execute_sieve_job returned success=False: {err_msg}"
        log.error(
            f"{worker_id}: SIEVE FAILED chunk={chunk_id}\n"
            f"  error:     {err_msg}\n"
            f"  traceback: {err_tb}"
        )
        return _error_result(chunk_id, worker_id, full_msg, err_tb)

    # ── Stage 5: parse survivors ──────────────────────────────────────────────
    try:
        raw = result.get("survivors", [])
        log.debug(
            f"{worker_id}: chunk={chunk_id} raw survivors={len(raw)} "
            f"first={raw[0] if raw else 'none'}"
        )

        survivors   = [int(s["seed"])                                  for s in raw]
        match_rates = [float(s.get("match_rate", 0.5))                 for s in raw]
        # Standard mode: best_skip (int) → wrap in list for consistency
        # Hybrid mode:   skip_sequence (list of ints)
        skip_seqs   = [s.get("skip_sequence", [s.get("best_skip", 0)]) for s in raw]
        strat_ids   = [int(s.get("strategy_id", 0))                    for s in raw]

        return {
            "chunk_id":       chunk_id,
            "worker_id":      worker_id,
            "status":         "ok",
            "survivors":      survivors,
            "match_rates":    match_rates,
            "skip_sequences": skip_seqs,
            "strategy_ids":   strat_ids,
        }
    except Exception as e:
        full_tb = tb.format_exc()
        log.error(
            f"{worker_id}: PARSE FAILED chunk={chunk_id} — {e}\n"
            f"  raw result keys: {list(result.keys())}\n"
            f"  first survivor:  {result.get('survivors', [None])[0]}\n"
            f"{full_tb}"
        )
        return _error_result(chunk_id, worker_id,
                             f"result parse failed: {e}", full_tb)


def _error_result(chunk_id: str, worker_id: str,
                  message: str, full_tb: str = "") -> dict:
    """Consistent error result format with full traceback."""
    return {
        "chunk_id":       chunk_id,
        "worker_id":      worker_id,
        "status":         "error",
        "message":        message[:500],
        "traceback":      full_tb[:2000],
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
    job_sock.setsockopt(zmq.RCVTIMEO, 500)  # S159B: was 5000ms

    # S159F: set env vars BEFORE sieve_filter import (ROCm requires this)
    if not use_cuda:
        os.environ.setdefault("ROCR_VISIBLE_DEVICES",     str(gpu_id))
        os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
        os.environ.setdefault("HSA_ENABLE_SDMA",          "0")
        os.environ.setdefault("ROCM_PATH",                "/opt/rocm")
        os.environ.setdefault("HIP_PATH",                 "/opt/rocm")
    else:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES",     str(gpu_id))

    # S159F: import sieve_filter ONCE at worker startup — not per chunk
    # Eliminates simultaneous CUDA context init collision between Zeus workers
    try:
        from sieve_filter import execute_sieve_job as _execute_sieve_job
        log.info(f"[S159F] sieve_filter imported at startup — gpu={gpu_id} cuda={use_cuda}")
    except Exception as _e:
        log.error(f"[S159F] sieve_filter import FAILED at startup: {_e}")
        sys.exit(1)


    log.info(
        f"Worker ready — id={worker_id} gpu={gpu_id} cuda={use_cuda} "
        f"zeus={args.zeus_host}:{args.job_port} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','unset')} "
        f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES','unset')}"
    )

    jobs_done  = 0
    jobs_error = 0

    try:
        while True:
            try:
                msg = job_sock.recv()
            except zmq.Again:
                continue

            try:
                job = json.loads(msg.decode())
            except Exception as e:
                log.error(f"{worker_id}: JSON decode failed — {e}")
                continue

            if job.get("cmd") == "shutdown":
                log.info(f"{worker_id}: shutdown received")
                break

            chunk_id = job.get("chunk_id", "?")
            log.info(
                f"{worker_id}: START chunk={chunk_id} "
                f"seeds={job.get('seed_start',0):,}→{job.get('seed_end',0):,} "
                f"prng={job.get('prng_type','?')} "
                f"threshold={job.get('threshold','?')}"
            )

            t0      = time.time()
            result  = run_sieve_job(job, gpu_id, use_cuda, worker_id, _execute_sieve_job)
            elapsed = time.time() - t0

            n = len(result.get("survivors", []))
            if result["status"] == "ok":
                log.info(
                    f"{worker_id}: DONE chunk={chunk_id} "
                    f"{n:,} survivors {elapsed:.1f}s"
                )
                jobs_done += 1
            else:
                log.error(
                    f"{worker_id}: ERROR chunk={chunk_id} "
                    f"after {elapsed:.1f}s — {result.get('message','?')}"
                )
                jobs_error += 1

            # Always send result back — Zeus decides what to do with errors
            try:
                result_sock.send(json.dumps(result).encode())
            except Exception as e:
                log.error(f"{worker_id}: ZMQ send failed chunk={chunk_id} — {e}")

    except KeyboardInterrupt:
        log.info(f"{worker_id}: interrupted")
    except Exception as e:
        log.error(f"{worker_id}: FATAL — {e}\n{tb.format_exc()}")
    finally:
        job_sock.close()
        result_sock.close()
        ctx.term()
        log.info(
            f"{worker_id}: exiting — "
            f"done={jobs_done} errors={jobs_error}"
        )


if __name__ == "__main__":
    main()
