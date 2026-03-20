#!/usr/bin/env python3
"""
ca_d3_window_calibration.py  (S148)
=====================================
Window size sensitivity experiment.
Runs the real GPU sieve across window sizes [8, 10, 12, 16] at a fixed
representative skip (5), sweeping thresholds to find the lowest zero-noise
threshold for each window size.

Hypothesis: larger windows require lower thresholds to achieve zero noise
because more draw constraints mean fewer accidental false positives.

Deploy and run on Zeus:
    scp ~/Downloads/ca_d3_window_calibration.py rzeus:~/distributed_prng_analysis/
    ssh rzeus "cd ~/distributed_prng_analysis && \\
        source ~/venvs/torch/bin/activate && \\
        python3 ca_d3_window_calibration.py"

Author: Team Alpha S148
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

_A    = 25214903917
_C    = 11
_MASK = (1 << 48) - 1

def _step(s): return (_A * s + _C) & _MASK
def _output(s): return ((s >> 16) & 0xFFFFFFFF) % 1000

def generate_draws(seed: int, n: int, draw_skip: int) -> List[Dict]:
    state = seed & _MASK
    BASE  = 1704067200 + 8 * 3600
    MID   = 13 * 3600 + 10
    EVE   = 18 * 3600 + 30 * 60 + 5
    draws = []
    day   = 0
    for i in range(n):
        session = 'midday' if i % 2 == 0 else 'evening'
        if i > 0 and i % 2 == 0:
            day += 1
        for _ in range(draw_skip):
            state = _step(state)
        state = _step(state)
        ts = BASE + day * 86400 + (MID if session == 'midday' else EVE)
        draws.append({"draw": _output(state), "session": session, "timestamp": ts})
    return draws


def spawn_worker(worker_path: str, gpu_id: int) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, worker_path, '--persistent', '--gpu-id', str(gpu_id)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, bufsize=1,
    )

def worker_readline(proc, timeout=60.0):
    import select
    start = time.time()
    while time.time() - start < timeout:
        ready, _, _ = select.select([proc.stdout], [], [], 1.0)
        if ready:
            line = proc.stdout.readline()
            if line:
                try: return json.loads(line.strip())
                except: continue
        if proc.poll() is not None: return None
    return None

def wait_for_ready(proc, timeout=120.0):
    start = time.time()
    while time.time() - start < timeout:
        msg = worker_readline(proc, 5.0)
        if msg is None: continue
        if msg.get('status') == 'ready':
            print(f"  Worker ready: GPU={msg.get('gpu_id')} device={msg.get('device','?')}")
            return True
        if msg.get('status') == 'error':
            print(f"  Worker error: {msg.get('error')}"); return False
    return False

def send_job(proc, job, timeout=300.0):
    proc.stdin.write(json.dumps({"command": "sieve", "job": job}) + "\n")
    proc.stdin.flush()
    job_id = job.get('job_id', 'unknown')
    start  = time.time()
    while time.time() - start < timeout:
        resp = worker_readline(proc, 10.0)
        if resp is None: continue
        if resp.get('status') in ('ok','error') and resp.get('job_id') == job_id:
            return resp
    return None

def shutdown_worker(proc):
    try:
        proc.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
        proc.stdin.flush()
        proc.wait(timeout=10)
    except: proc.kill()


def run_window_experiment(args):
    print("=" * 70)
    print("  CA Daily 3 — Window Size Sensitivity Experiment  (S148)")
    print("=" * 70)
    print(f"  Known seed        : {args.seed}")
    print(f"  Search radius     : ±{args.radius:,}  ({2*args.radius+1:,} seeds)")
    print(f"  Fixed draw_skip   : {args.draw_skip}")
    print(f"  Window sizes      : {args.windows}")
    print(f"  Kernel skip sweep : [0, {args.skip_sweep_max}]")
    print(f"  Threshold sweep   : {args.thresh_min:.2f} → {args.thresh_max:.2f} "
          f"step {args.thresh_step:.2f}")
    print()

    worker_path = Path(args.worker_path).resolve()
    if not worker_path.exists():
        print(f"  ERROR: Worker not found: {worker_path}"); sys.exit(1)

    print(f"  Spawning worker: {worker_path}")
    proc = spawn_worker(str(worker_path), args.gpu_id)
    if not wait_for_ready(proc, 120):
        print("  ERROR: Worker failed to become ready"); proc.kill(); sys.exit(1)
    print()

    # Generate draws once at max window size — all smaller windows use a subset
    max_window = max(args.windows)
    num_draws  = max(max_window + 5, args.num_draws)
    all_draws  = generate_draws(args.seed, num_draws, args.draw_skip)

    seed_start = args.seed - args.radius
    seed_end   = args.seed + args.radius + 1

    thresholds = []
    t = args.thresh_min
    while t <= args.thresh_max + 1e-9:
        thresholds.append(round(t, 4))
        t += args.thresh_step

    results = {}   # window -> {threshold -> row}
    draw_files = []

    for window in args.windows:
        print(f"── Window size: {window} " + "─" * 50)

        # Write draw file for this window (use first `window` draws)
        draw_file = Path(args.data_dir) / \
            f"ca_d3_win{window}_seed{args.seed}_skip{args.draw_skip}.json"
        with open(draw_file, 'w') as f:
            json.dump(all_draws[:num_draws], f)
        draw_files.append(draw_file)

        print(f"  Draws (first 5): {[d['draw'] for d in all_draws[:5]]}")
        print(f"  {'Thresh':>7}  {'Survivors':>10}  {'KnownSeed':>9}  "
              f"{'MatchRate':>10}  {'BestSkip':>9}  {'Noise':>6}")
        print("  " + "-" * 62)

        window_rows = {}
        first_zero_noise = None
        last_survival    = None

        for thresh in thresholds:
            job = {
                "job_id":              f"win{window}_t{thresh:.4f}",
                "dataset_path":        str(draw_file),
                "window_size":         window,
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
                err = resp.get('error','timeout') if resp else 'timeout'
                print(f"  {thresh:>7.4f}  ERROR: {err}")
                continue

            survivors  = resp['result'].get('survivors', [])
            surv_set   = {s['seed'] for s in survivors}
            known_surv = next((s for s in survivors if s['seed'] == args.seed), None)
            known_ok   = known_surv is not None
            noise      = len(surv_set) - (1 if known_ok else 0)
            match_rate = known_surv['match_rate'] if known_surv else 0.0
            best_skip  = known_surv.get('best_skip', -1) if known_surv else -1

            if known_ok:
                last_survival = thresh
                if noise == 0 and first_zero_noise is None:
                    first_zero_noise = thresh

            window_rows[thresh] = {
                "threshold":       thresh,
                "total_survivors": len(surv_set),
                "known_survived":  known_ok,
                "match_rate":      match_rate,
                "best_skip":       best_skip,
                "noise":           noise,
            }

            print(f"  {thresh:>7.4f}  {len(surv_set):>10}  "
                  f"{'YES' if known_ok else 'no':>9}  "
                  f"{f'{match_rate:.4f}' if known_ok else 'n/a':>10}  "
                  f"{str(best_skip) if known_ok else 'n/a':>9}  "
                  f"{str(noise) if known_ok else '-':>6}  "
                  f"{'✅' if known_ok else '❌'}")

        results[window] = {
            "rows":             window_rows,
            "first_zero_noise": first_zero_noise,
            "last_survival":    last_survival,
        }

        if first_zero_noise is not None:
            print(f"  ✅ window={window}: zero-noise at {first_zero_noise:.4f}  "
                  f"survival up to {last_survival:.4f}")
        else:
            print(f"  ⚠  window={window}: no zero-noise threshold found in sweep")
        print()

    shutdown_worker(proc)

    if not args.keep_data:
        for f in draw_files:
            try: f.unlink()
            except: pass

    # Summary table
    print("=" * 70)
    print("  WINDOW SIZE SENSITIVITY RESULTS")
    print("=" * 70)
    print()
    print(f"  Fixed draw_skip={args.draw_skip}, radius=±{args.radius:,}, "
          f"kernel sweep [0,{args.skip_sweep_max}]")
    print()
    print(f"  {'Window':>8}  {'Zero-noise at':>14}  {'Survives to':>12}  "
          f"{'vs window=8':>12}")
    print("  " + "-" * 55)

    baseline = results.get(8, {}).get('first_zero_noise')

    for window in args.windows:
        r   = results[window]
        znf = r['first_zero_noise']
        ls  = r['last_survival']
        if znf is not None:
            delta = f"{znf - baseline:+.4f}" if baseline is not None and window != 8 else "baseline"
            print(f"  {window:>8}  {znf:>14.4f}  {ls:>12.4f}  {delta:>12}")
        else:
            print(f"  {window:>8}  {'not found':>14}  "
                  f"{str(ls) if ls else 'never':>12}  {'—':>12}")

    print()
    print("  INTERPRETATION:")
    print("  If zero-noise threshold DECREASES as window grows → larger windows")
    print("  provide more discriminating power, meaning the sieve is more")
    print("  selective and a lower (more permissive) threshold is safe.")
    print()
    print("  If threshold stays FLAT → window size above 8 gives no extra")
    print("  selectivity benefit with this PRNG — window=8 is sufficient.")
    print()

    # Recommendation
    min_znf = min((r['first_zero_noise'] for r in results.values()
                   if r['first_zero_noise'] is not None), default=None)
    if min_znf is not None:
        print(f"  Best zero-noise threshold across all windows: {min_znf:.4f}")
        print(f"  Current production default                  : 0.2500")
        if min_znf > 0.25:
            print(f"  ⬆  Recommend RAISING threshold to {min_znf:.4f}")
        elif min_znf < 0.25:
            print(f"  ⬇  Larger windows allow LOWERING threshold to {min_znf:.4f}")
        else:
            print(f"  ✅ Current default 0.25 is well calibrated")

    print("=" * 70)


def parse_args():
    p = argparse.ArgumentParser(
        description="Window size sensitivity experiment for CA Daily 3 sieve calibration"
    )
    p.add_argument("--seed",           type=int,   default=3_141_592)
    p.add_argument("--radius",         type=int,   default=100_000)
    p.add_argument("--windows",        type=int,   nargs='+',
                   default=[8, 10, 12, 16])
    p.add_argument("--draw-skip",      type=int,   default=5,
                   help="Fixed draw_skip representing CA machine overhead (default: 5)")
    p.add_argument("--num-draws",      type=int,   default=50)
    p.add_argument("--skip-sweep-max", type=int,   default=50)
    p.add_argument("--thresh-min",     type=float, default=0.10)
    p.add_argument("--thresh-max",     type=float, default=0.75)
    p.add_argument("--thresh-step",    type=float, default=0.05)
    p.add_argument("--worker-path",    type=str,   default="./sieve_gpu_worker.py")
    p.add_argument("--data-dir",       type=str,   default="/tmp")
    p.add_argument("--gpu-id",         type=int,   default=0)
    p.add_argument("--keep-data",      action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    run_window_experiment(parse_args())
