#!/usr/bin/env python3
"""
ca_d3_threshold_calibration.py  (v2 — S148)
============================================
Threshold calibration harness for CA Daily 3 using the REAL sieve GPU
worker (sieve_gpu_worker.py) with the actual java_lcg kernel.

KEY DESIGN PRINCIPLE
--------------------
We don't know CA's exact skip rate, dual-RNG stride, or session overhead.
We DO know our own PRNG completely. So we:

  1. Pick a plausible CA-like draw_skip (approximating pre-test overhead,
     session setup, etc.) from a configurable set of scenarios.
  2. Generate synthetic draws from a KNOWN Java LCG seed using exactly
     the same step model the GPU kernel uses:
         burn draw_skip steps → advance one step → take output → repeat
  3. Run the real sieve kernel over a seed range containing the known
     seed, sweeping skip_range [0, skip_sweep_max] so the kernel finds
     whatever skip fits.
  4. Record whether the known seed survives at each threshold and what
     skip the kernel identified as best match.

This gives empirical threshold baselines for draw patterns that
approximate what CA Daily 3 would look like if it were Java LCG.

SCENARIOS
---------
Each scenario is a draw_skip value representing a plausible CA machine
state advancement between live draw outputs:
  skip=0   : consecutive outputs, no inter-draw overhead
  skip=3   : ≈ one pre-test cycle for a 3-digit draw
  skip=5   : ≈ pre-test + small session overhead
  skip=10  : ≈ pre-test + larger overhead or dual-RNG stride
  skip=20  : ≈ heavier session overhead scenario

Usage:
    python3 ca_d3_threshold_calibration.py [options]

    --seed SEED            Known LCG seed (default: 3141592)
    --radius RADIUS        Seed search radius (default: 100000)
    --window WINDOW        Sieve window size in draws (default: 8)
    --num-draws N          Total synthetic draws to generate (default: 30)
    --scenarios S [S ...]  draw_skip values to test (default: 0 3 5 10 20)
    --skip-sweep-max N     Max skip in kernel sweep range (default: 50)
    --thresh-min F         Threshold sweep start (default: 0.10)
    --thresh-max F         Threshold sweep end (default: 0.75)
    --thresh-step F        Threshold sweep step (default: 0.05)
    --worker-path PATH     Path to sieve_gpu_worker.py
                           (default: ./sieve_gpu_worker.py)
    --data-dir PATH        Where to write temp draw JSON (default: /tmp)
    --gpu-id N             GPU id for worker (default: 0)
    --keep-data            Don't delete temp draw files after run

Deploy and run on Zeus:
    scp ~/Downloads/ca_d3_threshold_calibration.py rzeus:~/distributed_prng_analysis/
    ssh rzeus "cd ~/distributed_prng_analysis && \\
        source ~/venvs/torch/bin/activate && \\
        python3 ca_d3_threshold_calibration.py"

Author: Team Alpha S148
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

# ── Java LCG constants (match GPU kernel exactly) ─────────────────────────────
_A    = 25214903917   # 0x5DEECE66D
_C    = 11            # 0xB
_MASK = (1 << 48) - 1

def _step(state: int) -> int:
    return (_A * state + _C) & _MASK

def _output(state: int) -> int:
    """Match GPU kernel: (state >> 16) & 0xFFFFFFFF then % 1000."""
    return ((state >> 16) & 0xFFFFFFFF) % 1000


# ── Draw generation — kernel-aligned model ────────────────────────────────────

def generate_draws(seed: int, n: int, draw_skip: int) -> List[Dict]:
    """
    Generate n draw records using EXACTLY the same step model as the
    GPU kernel:
        for each draw:
            burn draw_skip steps
            advance one step
            output = ((state >> 16) & 0xFFFFFFFF) % 1000

    This guarantees the kernel will find match_rate=1.0 at draw_skip
    from the known seed. draw_skip represents whatever combination of
    CA machine overhead we are modelling.

    Sessions alternate midday/evening matching CA D3 structure.
    """
    state = seed & _MASK

    BASE_UNIX  = 1704067200 + 8 * 3600  # 2024-01-01 00:00 PT
    MIDDAY_TS  = 13 * 3600 + 10         # 1:00:10 PM
    EVENING_TS = 18 * 3600 + 30 * 60 + 5  # 6:30:05 PM

    draws = []
    day   = 0

    for i in range(n):
        session = 'midday' if i % 2 == 0 else 'evening'
        if i > 0 and i % 2 == 0:
            day += 1

        for _ in range(draw_skip):
            state = _step(state)
        state = _step(state)
        val   = _output(state)

        ts = BASE_UNIX + day * 86400 + (MIDDAY_TS if session == 'midday'
                                        else EVENING_TS)
        draws.append({"draw": val, "session": session, "timestamp": ts})

    return draws


# ── Sieve worker IPC ───────────────────────────────────────────────────────────

def spawn_worker(worker_path: str, gpu_id: int) -> subprocess.Popen:
    cmd = [sys.executable, worker_path, '--persistent', '--gpu-id', str(gpu_id)]
    return subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, bufsize=1,
    )


def worker_readline(proc: subprocess.Popen,
                    timeout: float = 60.0) -> Optional[dict]:
    import select
    start = time.time()
    while time.time() - start < timeout:
        ready, _, _ = select.select([proc.stdout], [], [], 1.0)
        if ready:
            line = proc.stdout.readline()
            if line:
                try:
                    return json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
        if proc.poll() is not None:
            return None
    return None


def wait_for_ready(proc: subprocess.Popen,
                   timeout: float = 120.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        msg = worker_readline(proc, timeout=5.0)
        if msg is None:
            continue
        if msg.get('status') == 'ready':
            print(f"  Worker ready: GPU={msg.get('gpu_id')} "
                  f"device={msg.get('device','?')}")
            return True
        if msg.get('status') == 'error':
            print(f"  Worker error: {msg.get('error')}")
            return False
    return False


def send_job(proc: subprocess.Popen, job: dict,
             timeout: float = 300.0) -> Optional[dict]:
    proc.stdin.write(json.dumps({"command": "sieve", "job": job}) + "\n")
    proc.stdin.flush()
    job_id = job.get('job_id', 'unknown')
    start  = time.time()
    while time.time() - start < timeout:
        resp = worker_readline(proc, timeout=10.0)
        if resp is None:
            continue
        if resp.get('status') in ('ok', 'error') and \
                resp.get('job_id') == job_id:
            return resp
    return None


def shutdown_worker(proc: subprocess.Popen) -> None:
    try:
        proc.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
        proc.stdin.flush()
        proc.wait(timeout=10)
    except Exception:
        proc.kill()


# ── Per-scenario threshold sweep ──────────────────────────────────────────────

def run_scenario(proc, draw_skip: int, args,
                 draw_file: Path) -> List[Dict]:
    seed_start = args.seed - args.radius
    seed_end   = args.seed + args.radius + 1

    thresholds = []
    t = args.thresh_min
    while t <= args.thresh_max + 1e-9:
        thresholds.append(round(t, 4))
        t += args.thresh_step

    rows = []
    for thresh in thresholds:
        job = {
            "job_id":              f"skip{draw_skip}_t{thresh:.4f}",
            "dataset_path":        str(draw_file),
            "window_size":         args.window,
            "seed_start":          seed_start,
            "seed_end":            seed_end,
            "skip_range":          [0, args.skip_sweep_max],
            "min_match_threshold": thresh,
            "offset":              0,
            "sessions":            ["midday", "evening"],
            "prng_families":       ["java_lcg"],
        }

        resp = send_job(proc, job)
        if resp is None or resp.get('status') != 'ok':
            err = resp.get('error', 'timeout') if resp else 'timeout'
            print(f"    {thresh:>7.4f}  ERROR: {err}")
            rows.append({"threshold": thresh, "error": err})
            continue

        survivors  = resp['result'].get('survivors', [])
        surv_set   = {s['seed'] for s in survivors}
        known_surv = next((s for s in survivors if s['seed'] == args.seed),
                          None)
        known_ok   = known_surv is not None
        noise      = len(surv_set) - (1 if known_ok else 0)
        match_rate = known_surv['match_rate'] if known_surv else 0.0
        best_skip  = known_surv.get('best_skip', -1) if known_surv else -1

        rows.append({
            "threshold":       thresh,
            "total_survivors": len(surv_set),
            "known_survived":  known_ok,
            "match_rate":      match_rate,
            "best_skip":       best_skip,
            "noise":           noise,
        })

        print(f"    {thresh:>7.4f}  {len(surv_set):>10}  "
              f"{'YES' if known_ok else 'no':>9}  "
              f"{f'{match_rate:.4f}' if known_ok else 'n/a':>10}  "
              f"{str(best_skip) if known_ok else 'n/a':>9}  "
              f"{str(noise) if known_ok else '-':>6}  "
              f"{'✅' if known_ok else '❌'}")

    return rows


def summarise_scenario(draw_skip: int,
                       rows: List[Dict]) -> Optional[float]:
    surviving  = [r for r in rows if r.get('known_survived')]
    zero_noise = [r for r in surviving if r.get('noise', 999) == 0]

    if not surviving:
        print(f"  ⚠  skip={draw_skip}: known seed never survived — "
              f"widen --skip-sweep-max or increase --window")
        return None

    lo = min(r['threshold'] for r in surviving)
    hi = max(r['threshold'] for r in surviving)
    best_skip_found = surviving[0]['best_skip']

    if zero_noise:
        rec = min(r['threshold'] for r in zero_noise)
        print(f"  ✅ skip={draw_skip}: survival {lo:.2f}→{hi:.2f}  "
              f"zero-noise at {rec:.4f}  "
              f"kernel best_skip={best_skip_found}")
        return rec
    else:
        print(f"  ⚠  skip={draw_skip}: survival {lo:.2f}→{hi:.2f}  "
              f"no zero-noise found  "
              f"kernel best_skip={best_skip_found}")
        return None


# ── Main ───────────────────────────────────────────────────────────────────────

def run_calibration(args) -> None:
    print("=" * 70)
    print("  CA Daily 3 Threshold Calibration — Real GPU Sieve  (S148 v2)")
    print("=" * 70)
    print(f"  Known seed       : {args.seed}")
    print(f"  Search radius    : ±{args.radius:,}  ({2*args.radius+1:,} seeds)")
    print(f"  Window size      : {args.window}")
    print(f"  Scenarios (skip) : {args.scenarios}")
    print(f"  Kernel skip sweep: [0, {args.skip_sweep_max}]")
    print(f"  Threshold sweep  : {args.thresh_min:.2f} → {args.thresh_max:.2f} "
          f"step {args.thresh_step:.2f}")
    print()

    worker_path = Path(args.worker_path).resolve()
    if not worker_path.exists():
        print(f"  ERROR: Worker not found: {worker_path}")
        sys.exit(1)

    print(f"  Spawning worker: {worker_path}")
    proc = spawn_worker(str(worker_path), args.gpu_id)

    if not wait_for_ready(proc, timeout=120):
        print("  ERROR: Worker failed to become ready")
        proc.kill()
        sys.exit(1)
    print()

    recommendations = {}
    draw_files      = []

    for draw_skip in args.scenarios:
        print(f"── Scenario: draw_skip={draw_skip} " + "─" * 38)
        draws = generate_draws(args.seed, args.num_draws, draw_skip)
        print(f"  Generated {len(draws)} draws  "
              f"first 5: {[d['draw'] for d in draws[:5]]}")

        draw_file = Path(args.data_dir) / \
            f"ca_d3_calib_seed{args.seed}_skip{draw_skip}.json"
        with open(draw_file, 'w') as f:
            json.dump(draws, f)
        draw_files.append(draw_file)

        print(f"  {'Thresh':>7}  {'Survivors':>10}  {'KnownSeed':>9}  "
              f"{'MatchRate':>10}  {'BestSkip':>9}  {'Noise':>6}")
        print("  " + "-" * 62)

        rows = run_scenario(proc, draw_skip, args, draw_file)
        rec  = summarise_scenario(draw_skip, rows)
        if rec is not None:
            recommendations[draw_skip] = rec
        print()

    shutdown_worker(proc)

    if not args.keep_data:
        for f in draw_files:
            try:
                f.unlink()
            except Exception:
                pass

    # Final cross-scenario summary
    print("=" * 70)
    print("  CROSS-SCENARIO SUMMARY")
    print("=" * 70)
    print()
    print("  Lowest threshold giving zero false positives per scenario:")
    print()
    print(f"  {'Skip':>6}  {'Recommended':>12}  Note")
    print("  " + "-" * 55)

    notes = {
        0:  "consecutive outputs — no overhead",
        3:  "≈ 1 pre-test cycle (3 digits)",
        5:  "≈ pre-test + small overhead",
        10: "≈ pre-test + larger overhead / dual-RNG",
        20: "≈ heavier session overhead",
    }

    for skip in args.scenarios:
        if skip in recommendations:
            note = notes.get(skip, "")
            print(f"  {skip:>6}  {recommendations[skip]:>12.4f}  {note}")
        else:
            print(f"  {skip:>6}  {'— not recovered':>12}  "
                  f"(try --skip-sweep-max or --window)")

    if recommendations:
        vals         = list(recommendations.values())
        conservative = max(vals)
        permissive   = min(vals)
        print()
        print(f"  Range across scenarios : {permissive:.4f} → {conservative:.4f}")
        print(f"  Conservative baseline  : {conservative:.4f}  "
              f"(safe regardless of CA skip scenario)")
        print(f"  Permissive baseline    : {permissive:.4f}  "
              f"(tightest scenario — highest specificity)")
        print()
        print("  Update production defaults if these differ from current 0.25:")
        print("    persistent_worker_coordinator.py  →  threshold")
        print("    window_optimizer.py               →  default_forward_threshold")
        print("    WindowSearchBounds                →  min_forward_threshold")

    print("=" * 70)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="CA Daily 3 threshold calibration using real GPU sieve worker"
    )
    p.add_argument("--seed",           type=int,   default=3_141_592)
    p.add_argument("--radius",         type=int,   default=100_000)
    p.add_argument("--window",         type=int,   default=8)
    p.add_argument("--num-draws",      type=int,   default=30)
    p.add_argument("--scenarios",      type=int,   nargs='+',
                   default=[0, 3, 5, 10, 20])
    p.add_argument("--skip-sweep-max", type=int,   default=50)
    p.add_argument("--thresh-min",     type=float, default=0.10)
    p.add_argument("--thresh-max",     type=float, default=0.75)
    p.add_argument("--thresh-step",    type=float, default=0.05)
    p.add_argument("--worker-path",    type=str,
                   default="./sieve_gpu_worker.py")
    p.add_argument("--data-dir",       type=str,   default="/tmp")
    p.add_argument("--gpu-id",         type=int,   default=0)
    p.add_argument("--keep-data",      action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_calibration(args)
