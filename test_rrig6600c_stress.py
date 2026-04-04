#!/usr/bin/env python3
"""
test_rrig6600c_stress.py
=========================
Stress test harness for rrig6600c that replicates the exact PWC Step 1
spawn + dispatch pattern to reproduce and diagnose the GPU4 quarantine crash.

What it does:
  1. Spawns N workers on rrig6600c (default: 6, GPUs 0-5) with 4s stagger
  2. Sends real sieve jobs to all workers simultaneously (same as PWC)
  3. Runs multiple rounds of job dispatch to accumulate stress
  4. Monitors VRAM usage on rrig6600c via rocm-smi after each round
  5. Logs any spawn failures, timeouts, or errors per GPU
  6. Reports which GPU fails first and under what conditions

Usage:
  # Run on Zeus:
  python3 test_rrig6600c_stress.py --gpu-count 6 --rounds 20
  python3 test_rrig6600c_stress.py --gpu-count 4 --rounds 20   # test stable baseline
  python3 test_rrig6600c_stress.py --gpu-count 6 --rounds 5 --dry-run
"""

import argparse
import json
import subprocess
import threading
import time
import os
import sys

# ── Config ────────────────────────────────────────────────────────────────────
RIG_HOST        = "192.168.3.162"
RIG_USER        = "michael"
RIG_SCRIPT_PATH = "/home/michael/distributed_prng_analysis"
RIG_PYTHON      = "/home/michael/rocm_env/bin/python"
WORKER_SCRIPT   = "sieve_gpu_worker.py"

ROCM_ENV_VARS = [
    "HSA_OVERRIDE_GFX_VERSION=10.3.0",
    "HSA_ENABLE_SDMA=0",
]

SPAWN_STAGGER_S       = 4.0
WORKER_HEARTBEAT_S    = 30.0
JOB_TIMEOUT_S         = 120.0

# Minimal real sieve job — same format as PWC dispatches
STRESS_JOB_TEMPLATE = {
    "job_id":              "stress_{gpu_id}_{round}",
    "dataset_path":        "daily3.json",
    "window_size":         5,
    "seed_start":          0,
    "seed_end":            2_000_000,
    "skip_range":          [5, 20],
    "min_match_threshold": 0.30,
    "offset":              5,
    "sessions":            ["midday"],
    "prng_families":       [{"type": "java_lcg"}],
    "gpu_id":              None,  # filled per worker
}


def log(msg, level="INFO"):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def ssh_rocm_smi(host, user):
    """Get VRAM usage per GPU on remote host."""
    try:
        result = subprocess.run(
            ["ssh", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
             f"{user}@{host}", "rocm-smi --showmemuse 2>/dev/null"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout
    except Exception as e:
        return f"rocm-smi failed: {e}"


def spawn_worker(gpu_id, results):
    """Spawn one persistent worker on rrig6600c. Records result in results dict."""
    activate = f"source {os.path.join(os.path.dirname(RIG_PYTHON), 'activate')}"
    rocm_env = " ".join(ROCM_ENV_VARS)
    cmd_body = (
        f"cd {RIG_SCRIPT_PATH} && "
        f"{activate} && "
        f"env {rocm_env} {RIG_PYTHON} -u {WORKER_SCRIPT} --gpu-id {gpu_id} --persistent"
    )
    ssh_cmd = [
        "ssh", "-q",
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        "-o", "ServerAliveInterval=30",
        f"{RIG_USER}@{RIG_HOST}",
        cmd_body
    ]

    try:
        proc = subprocess.Popen(
            ssh_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        # Wait for ready heartbeat
        deadline = time.time() + WORKER_HEARTBEAT_S
        ready = False
        while time.time() < deadline:
            try:
                line = proc.stdout.readline()
                if not line:
                    break
                line = line.decode("utf-8", errors="replace").strip()
                if '"status"' in line and '"ready"' in line:
                    ready = True
                    break
            except Exception:
                break

        if not ready:
            proc.kill()
            log(f"GPU{gpu_id}: SPAWN FAILED — no heartbeat", "ERROR")
            results[gpu_id] = {"status": "spawn_failed", "proc": None}
            return

        log(f"GPU{gpu_id}: spawned and ready")
        results[gpu_id] = {"status": "alive", "proc": proc}

    except Exception as e:
        log(f"GPU{gpu_id}: SPAWN EXCEPTION — {e}", "ERROR")
        results[gpu_id] = {"status": "exception", "proc": None, "error": str(e)}


def dispatch_job(gpu_id, proc, round_num, results):
    """Send one sieve job to a worker and wait for result."""
    job = dict(STRESS_JOB_TEMPLATE)
    job["job_id"] = f"stress_{gpu_id}_{round_num}"
    job["gpu_id"] = gpu_id

    try:
        line = (json.dumps(job, separators=(',', ':')) + "\n").encode()
        proc.stdin.write(line)
        proc.stdin.flush()

        deadline = time.time() + JOB_TIMEOUT_S
        while time.time() < deadline:
            result_line = proc.stdout.readline()
            if not result_line:
                log(f"GPU{gpu_id} round {round_num}: PIPE CLOSED", "ERROR")
                results[gpu_id] = "pipe_closed"
                return
            result_line = result_line.decode("utf-8", errors="replace").strip()
            if result_line:
                try:
                    result = json.loads(result_line)
                    status = result.get("status", "unknown")
                    if status == "ok":
                        inner = result.get("result", {})
                        n_seeds = inner.get("stats", {}).get("total_seeds_tested", 0)
                        n_surv = len(inner.get("seeds", []))
                        log(f"GPU{gpu_id} round {round_num}: OK — {n_seeds:,} seeds, {n_surv} survivors")
                        results[gpu_id] = "ok"
                    else:
                        log(f"GPU{gpu_id} round {round_num}: ERROR — {result}", "ERROR")
                        results[gpu_id] = f"error:{status}"
                except json.JSONDecodeError:
                    log(f"GPU{gpu_id} round {round_num}: BAD JSON — {result_line[:100]}", "WARN")
                    results[gpu_id] = "bad_json"
                return

        log(f"GPU{gpu_id} round {round_num}: TIMEOUT after {JOB_TIMEOUT_S}s", "ERROR")
        results[gpu_id] = "timeout"

    except Exception as e:
        log(f"GPU{gpu_id} round {round_num}: DISPATCH EXCEPTION — {e}", "ERROR")
        results[gpu_id] = f"exception:{e}"


def kill_workers(worker_procs):
    log("Killing all workers...")
    for gpu_id, info in worker_procs.items():
        proc = info.get("proc")
        if proc:
            try:
                proc.kill()
            except Exception:
                pass
    # Also kill any stragglers on the rig
    subprocess.run(
        ["ssh", "-q", f"{RIG_USER}@{RIG_HOST}", "pkill -f sieve_gpu_worker 2>/dev/null"],
        capture_output=True
    )


def main():
    parser = argparse.ArgumentParser(description="rrig6600c stress test")
    parser.add_argument("--gpu-count", type=int, default=6,
                        help="Number of GPUs to spawn workers on (default: 6)")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Number of job rounds to run (default: 10)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config and exit without running")
    args = parser.parse_args()

    print("=" * 60)
    print(f"rrig6600c Stress Test — PWC Step 1 Replication")
    print(f"Host:       {RIG_HOST}")
    print(f"GPU count:  {args.gpu_count} (GPUs 0-{args.gpu_count-1})")
    print(f"Rounds:     {args.rounds}")
    print(f"Stagger:    {SPAWN_STAGGER_S}s between spawns")
    print(f"Job size:   2M seeds per chunk")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY-RUN] Config looks good. Exiting.")
        return

    # ── Phase 1: Spawn workers with stagger ──────────────────────────────────
    log("Phase 1: Spawning workers with stagger...")
    worker_procs = {}
    spawn_threads = []

    for gpu_id in range(args.gpu_count):
        t = threading.Thread(
            target=spawn_worker,
            args=(gpu_id, worker_procs),
            daemon=True
        )
        spawn_threads.append(t)
        t.start()
        if gpu_id < args.gpu_count - 1:
            log(f"Staggering {SPAWN_STAGGER_S}s before GPU{gpu_id+1}...")
            time.sleep(SPAWN_STAGGER_S)

    for t in spawn_threads:
        t.join(timeout=WORKER_HEARTBEAT_S + 5)

    alive = [gpu_id for gpu_id, info in worker_procs.items()
             if info.get("status") == "alive"]
    failed = [gpu_id for gpu_id, info in worker_procs.items()
              if info.get("status") != "alive"]

    log(f"Worker pool ready: {len(alive)}/{args.gpu_count} alive")
    if failed:
        log(f"FAILED GPUs: {failed}", "ERROR")
        kill_workers(worker_procs)
        return

    # ── Phase 2: VRAM baseline ────────────────────────────────────────────────
    log("\nPhase 2: VRAM baseline (before jobs):")
    print(ssh_rocm_smi(RIG_HOST, RIG_USER))

    # ── Phase 3: Dispatch rounds ──────────────────────────────────────────────
    log(f"Phase 3: Running {args.rounds} job rounds...")
    round_failures = []

    for round_num in range(args.rounds):
        log(f"\n--- Round {round_num + 1}/{args.rounds} ---")
        round_results = {}
        threads = []

        for gpu_id in alive:
            proc = worker_procs[gpu_id]["proc"]
            t = threading.Thread(
                target=dispatch_job,
                args=(gpu_id, proc, round_num, round_results),
                daemon=True
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=JOB_TIMEOUT_S + 10)

        # Check for failures
        failures = {gpu_id: result for gpu_id, result in round_results.items()
                    if result != "ok"}
        if failures:
            log(f"Round {round_num + 1} FAILURES: {failures}", "ERROR")
            round_failures.append((round_num, failures))

            # Check VRAM at failure point
            log("VRAM at failure point:")
            print(ssh_rocm_smi(RIG_HOST, RIG_USER))
            break

        # Every 5 rounds check VRAM
        if (round_num + 1) % 5 == 0:
            log(f"VRAM after round {round_num + 1}:")
            print(ssh_rocm_smi(RIG_HOST, RIG_USER))

    # ── Phase 4: Results ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if round_failures:
        print(f"✗ STRESS TEST FAILED")
        print(f"  Failed at round: {round_failures[0][0] + 1}")
        print(f"  Failed GPUs: {round_failures[0][1]}")
    else:
        print(f"✓ STRESS TEST PASSED — {args.rounds} rounds, {len(alive)} GPUs, no failures")

    print("=" * 60)

    kill_workers(worker_procs)


if __name__ == "__main__":
    main()
