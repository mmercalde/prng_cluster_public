#!/usr/bin/env python3
"""
Backtest prediction pools across historical data.

For each date in range:
  1. Train on data before date (run sieve)
  2. Build pools from survivors
  3. Evaluate against actual draw on that date
  4. Track Hit@K metrics over time

Usage:
  python3 backtest_pools.py --dataset daily3.json --start 2025-01-01 --end 2025-09-30 --out backtest_results.json
"""

import argparse
import json
import subprocess
import os
import time
import shutil
from datetime import datetime, timedelta
from typing import List, Dict, Any

def iso_now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

def load_dataset(dataset_path: str) -> List[Dict]:
    """Load and sort dataset chronologically with stable session ordering"""
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    # Stable sort by date then session
    session_order = {"midday": 0, "evening": 1}
    data.sort(key=lambda x: (x.get('date', ''), session_order.get(x.get('session', ''), 99)))
    return data

def filter_data_before_date(data: List[Dict], cutoff_date: str, session: str) -> List[Dict]:
    """Return only draws before cutoff date, filtered by session"""
    # Filter by session first
    data = [entry for entry in data if entry.get('session') == session]
    # Then filter by date
    return [entry for entry in data if entry.get('date', '') < cutoff_date]

def get_draw_for_date(data: List[Dict], target_date: str, session: str) -> int:
    """Get the actual draw for a specific date/session"""
    for entry in data:
        if entry.get('date') == target_date and entry.get('session') == session:
            return int(entry.get('draw'))
    return None

def create_temp_dataset(data: List[Dict], temp_path: str):
    """Write filtered data to temporary JSON file"""
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=2)

def run_sieve(dataset_path: str, seed_start: int, seed_end: int,
              prng_type: str, workdir: str, run_id: str, offset: int = 0) -> str:
    """Run sieve and return path to results file"""
    cmd = [
        'python3', 'coordinator.py',
        dataset_path,
        '--method', 'residue_sieve',
        '--prng-type', prng_type,
        '--seeds', str(seed_end - seed_start),
        '--offset', str(offset)
    ]

    print(f"    Running sieve: {seed_start:,} to {seed_end:,} seeds...", end=' ', flush=True)
    start = time.time()

    result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')

    elapsed = time.time() - start
    print(f"{elapsed:.1f}s")

    if result.returncode != 0:
        print(f"      Sieve failed: {result.stderr[:200]}")
        return None

    # Find most recent results file
    import glob
    results_files = glob.glob('results/multi_gpu_analysis_*.json')
    if not results_files:
        return None

    latest = max(results_files, key=os.path.getmtime)

    # Copy to workdir with run_id
    dest = os.path.join(workdir, f'sieve_{run_id}.json')
    shutil.copy(latest, dest)

    return dest

def run_pool_builder(survivors_file: str, pools_output: str, prng_type: str) -> bool:
    """Build prediction pools from sieve survivors"""

    cmd = [
        'python3', 'build_pools.py',
        '--survivors', survivors_file,
        '--prng-type', prng_type,
        '--pools', '20,100,300',
        '--out', pools_output
    ]

    print(f"    Building pools...", end=' ', flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"FAILED")
        return False

    print(f"OK")
    return True

def run_evaluator(pools_file: str, truth_file: str, target_date: str,
                  session: str, metrics_output: str) -> Dict:
    """Evaluate pools against actual draw"""

    cmd = [
        'python3', 'evaluate_pools.py',
        '--pools', pools_file,
        '--truth', truth_file,
        '--date', target_date,
        '--session', session,
        '--out', metrics_output
    ]

    print(f"    Evaluating...", end=' ', flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"FAILED")
        return None

    # Load metrics with safe parsing
    try:
        with open(metrics_output, 'r') as f:
            metrics = json.load(f)

        hit_20 = metrics.get('summary', {}).get('hit_20', False)
        hit_100 = metrics.get('summary', {}).get('hit_100', False)
        hit_300 = metrics.get('summary', {}).get('hit_300', False)

        print(f"Hit@20: {hit_20}  Hit@100: {hit_100}  Hit@300: {hit_300}")

        return metrics
    except Exception as e:
        print(f"FAILED to parse metrics: {e}")
        return None

def generate_date_range(start_date: str, end_date: str) -> List[str]:
    """Generate list of dates to backtest"""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)

    return dates

def main():
    ap = argparse.ArgumentParser(description="Backtest prediction pools over historical data")
    ap.add_argument("--dataset", required=True, help="daily3.json or similar")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    ap.add_argument("--session", default="midday", help="Session: midday or evening")
    ap.add_argument("--prng-type", default="xorshift32", help="PRNG type")
    ap.add_argument("--seed-start", type=int, default=0, help="Seed range start")
    ap.add_argument("--seed-end", type=int, default=100000, help="Seed range end")
    ap.add_argument("--window", type=int, default=30, help="Training window size")
    ap.add_argument("--out", default="backtest_results.json", help="Output results file")
    ap.add_argument("--out-csv", default="backtest_summary.csv", help="Output CSV summary")
    args = ap.parse_args()

    print(f"\nBacktest Framework")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Date range: {args.start} to {args.end}")
    print(f"Session: {args.session}")
    print(f"PRNG: {args.prng_type}")
    print(f"Seeds: {args.seed_start:,} to {args.seed_end:,}")
    print(f"Training window: {args.window} draws")
    print("=" * 70)

    # Load full dataset
    full_data = load_dataset(args.dataset)
    print(f"\nLoaded {len(full_data)} total draws")

    # Generate test dates
    test_dates = generate_date_range(args.start, args.end)
    print(f"Testing {len(test_dates)} dates")

    # Results tracking
    all_results = []
    hit_20_count = 0
    hit_100_count = 0
    hit_300_count = 0
    total_time = 0

    # Backtest each date
    for i, test_date in enumerate(test_dates, 1):
        date_start = time.time()

        print(f"\n[{i}/{len(test_dates)}] Testing {test_date} {args.session}")

        # Check if draw exists for this date
        actual_draw = get_draw_for_date(full_data, test_date, args.session)
        if actual_draw is None:
            print(f"  ⊘ No draw found, skipping")
            continue

        print(f"  Actual draw: {actual_draw}")

        # Create isolated workdir for this test
        run_id = f"{test_date}_{args.session}".replace('-', '')
        workdir = f'backtest_temp/{run_id}'
        os.makedirs(workdir, exist_ok=True)

        # Create training dataset (all data before test_date, filtered by session)
        training_data = filter_data_before_date(full_data, test_date, args.session)

        # Use only recent window
        training_data = training_data[-args.window:]

        if len(training_data) < 10:
            print(f"  ⊘ Insufficient training data ({len(training_data)} draws), skipping")
            continue

        print(f"  Training on {len(training_data)} draws (session: {args.session})")

        # Create temp dataset file
        temp_dataset = os.path.join(workdir, 'train.json')
        create_temp_dataset(training_data, temp_dataset)

        # Calculate offset: how many draws came before the training window
        all_session_draws = [entry for entry in full_data 
                             if entry.get('session') == args.session 
                             and entry.get('date', '') < test_date]
        offset = len(all_session_draws) - args.window
        if offset < 0:
            offset = 0

        # Run sieve
        survivors_file = run_sieve(temp_dataset, args.seed_start, args.seed_end,
                                   args.prng_type, workdir, run_id, offset)

        if not survivors_file:
            print(f"  ✗ Sieve failed")
            continue

        # Build pools
        pools_file = os.path.join(workdir, 'pools.json')
        if not run_pool_builder(survivors_file, pools_file, args.prng_type):
            print(f"  ✗ Pool building failed")
            continue

        # Evaluate
        metrics_file = os.path.join(workdir, 'metrics.json')
        metrics = run_evaluator(pools_file, args.dataset, test_date,
                               args.session, metrics_file)

        if not metrics:
            print(f"  ✗ Evaluation failed")
            continue

        # Track results
        hit_20 = metrics.get('summary', {}).get('hit_20', False)
        hit_100 = metrics.get('summary', {}).get('hit_100', False)
        hit_300 = metrics.get('summary', {}).get('hit_300', False)

        if hit_20:
            hit_20_count += 1
        if hit_100:
            hit_100_count += 1
        if hit_300:
            hit_300_count += 1

        survivors = metrics.get('prediction_metadata', {}).get('survivor_count', 0)

        all_results.append({
            'date': test_date,
            'session': args.session,
            'actual_draw': actual_draw,
            'hit_20': hit_20,
            'hit_100': hit_100,
            'hit_300': hit_300,
            'survivors': survivors
        })

        date_elapsed = time.time() - date_start
        total_time += date_elapsed

        # Progress update
        completed = len(all_results)
        if completed > 0:
            avg_time = total_time / completed
            remaining = len(test_dates) - i
            eta_min = (remaining * avg_time) / 60

            rate_20 = hit_20_count / completed * 100
            rate_100 = hit_100_count / completed * 100
            rate_300 = hit_300_count / completed * 100

            print(f"  Progress: {completed}/{len(test_dates)} | "
                  f"Hit rates: {rate_20:.0f}%/{rate_100:.0f}%/{rate_300:.0f}% | "
                  f"ETA: {eta_min:.1f}min")

    # Calculate overall metrics
    total_tests = len(all_results)

    if total_tests == 0:
        print("\n✗ No successful backtests completed")
        return 1

    hit_20_rate = hit_20_count / total_tests
    hit_100_rate = hit_100_count / total_tests
    hit_300_rate = hit_300_count / total_tests

    # Summary
    print(f"\n{'=' * 70}")
    print(f"BACKTEST SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total tests: {total_tests}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Hit@20:  {hit_20_count:3d} / {total_tests} = {hit_20_rate*100:5.1f}%  (Lift: {hit_20_rate/0.02:5.1f}x)")
    print(f"Hit@100: {hit_100_count:3d} / {total_tests} = {hit_100_rate*100:5.1f}%  (Lift: {hit_100_rate/0.10:5.1f}x)")
    print(f"Hit@300: {hit_300_count:3d} / {total_tests} = {hit_300_rate*100:5.1f}%  (Lift: {hit_300_rate/0.30:5.1f}x)")

    # Save detailed results
    output = {
        'version': 1,
        'run_id': iso_now(),
        'config': {
            'dataset': args.dataset,
            'date_range': {'start': args.start, 'end': args.end},
            'session': args.session,
            'prng_type': args.prng_type,
            'seed_range': {'start': args.seed_start, 'end': args.seed_end},
            'window': args.window
        },
        'summary': {
            'total_tests': total_tests,
            'total_time_sec': round(total_time, 2),
            'hit_20': {'count': hit_20_count, 'rate': round(hit_20_rate, 4), 'lift': round(hit_20_rate/0.02, 2)},
            'hit_100': {'count': hit_100_count, 'rate': round(hit_100_rate, 4), 'lift': round(hit_100_rate/0.10, 2)},
            'hit_300': {'count': hit_300_count, 'rate': round(hit_300_rate, 4), 'lift': round(hit_300_rate/0.30, 2)}
        },
        'results': all_results
    }

    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed results: {args.out}")

    # Save CSV summary (Windows-safe)
    with open(args.out_csv, 'w', newline='') as f:
        f.write("date,session,actual_draw,hit_20,hit_100,hit_300,survivors\n")
        for r in all_results:
            f.write(f"{r['date']},{r['session']},{r['actual_draw']},{r['hit_20']},{r['hit_100']},{r['hit_300']},{r['survivors']}\n")

    print(f"CSV summary: {args.out_csv}")

    return 0

if __name__ == "__main__":
    exit(main())
