#!/usr/bin/env python3
"""
Timestamp-Based PRNG Seed Search - FIXED VERSION
Handles reverse chronological data correctly
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pytz


def load_dataset_info(dataset_path: str, sessions: list, window_size: int) -> dict:
    """Extract date range - handles reverse chronological data"""
    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        if sessions:
            filtered = [e for e in data if e.get('session') in sessions]
        else:
            filtered = data
        
        if len(filtered) < window_size:
            raise ValueError(f"Dataset has only {len(filtered)} entries, need at least {window_size}")
        
        # Take last N draws (most recent in reverse-chrono data)
        window_data = filtered[:window_size]
        
        # Get date range (handle reverse chronological)
        dates = [entry['date'] for entry in window_data]
        first_date = min(dates)  # Oldest date
        last_date = max(dates)   # Newest date
        
        return {
            'total_entries': len(filtered),
            'window_size': window_size,
            'first_date': first_date,
            'last_date': last_date,
            'sample_draws': [e['draw'] for e in window_data[:5]]
        }
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        sys.exit(1)


def calculate_timestamp_range(first_date: str, last_date: str, 
                              mode: str = 'millisecond',
                              timezone: str = 'America/Los_Angeles',
                              time_buffer: int = None) -> tuple:
    """Calculate timestamp range"""
    tz = pytz.timezone(timezone)
    
    start_date = datetime.strptime(first_date, '%Y-%m-%d')
    end_date = datetime.strptime(last_date, '%Y-%m-%d')
    
    if time_buffer:
        start_date -= timedelta(seconds=time_buffer)
        end_date += timedelta(seconds=time_buffer)
    
    start_dt = tz.localize(datetime(start_date.year, start_date.month, start_date.day, 0, 0, 0))
    end_dt = tz.localize(datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59, 999000))
    
    if mode == 'microsecond':
        start_ts = int(start_dt.timestamp() * 1_000_000)
        end_ts = int(end_dt.timestamp() * 1_000_000)
        unit = 'microseconds'
    elif mode == 'millisecond':
        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)
        unit = 'milliseconds'
    elif mode == 'second':
        start_ts = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())
        unit = 'seconds'
    elif mode == 'minute':
        start_ts = int(start_dt.timestamp() // 60)
        end_ts = int(end_dt.timestamp() // 60)
        unit = 'minutes'
    elif mode == 'decisecond':
        start_ts = int(start_dt.timestamp() * 10)
        end_ts = int(end_dt.timestamp() * 10)
        unit = 'deciseconds'
    elif mode == 'centisecond':
        start_ts = int(start_dt.timestamp() * 100)
        end_ts = int(end_dt.timestamp() * 100)
        unit = 'centiseconds'
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return start_ts, end_ts, unit


def submit_to_coordinator(dataset_path: str, start_ts: int, end_ts: int,
                         output_file: str, args) -> int:
    """Submit to coordinator with proper seed range"""
    
    total_seeds = end_ts - start_ts
    
    cmd = [
        'python3', 'coordinator.py',
        dataset_path,
        '--method', 'residue_sieve',
        '--seeds', str(total_seeds),
        '--seed-start', str(start_ts),
        '--window-size', str(args.window),
        '--offset', str(args.offset),
        '--prng-type', args.prngs[0] if args.prngs else 'xorshift32',
        '--skip-min', str(args.skip_min),
        '--skip-max', str(args.skip_max),
        '--threshold', str(args.threshold),
        '--output', output_file,
        '--resume-policy', args.resume_policy
    ]
    
    if args.max_concurrent:
        cmd.extend(['--max-concurrent', str(args.max_concurrent)])
    if args.timeout:
        cmd.extend(['--job-timeout', str(args.timeout)])
    
    # Add hybrid mode arguments if enabled
    if hasattr(args, 'hybrid') and args.hybrid:
        cmd.append('--hybrid')
        cmd.extend(['--phase1-threshold', str(args.phase1_threshold)])
        cmd.extend(['--phase2-threshold', str(args.phase2_threshold)])
    
    print(f"\n{'='*70}")
    print("SUBMITTING TO DISTRIBUTED COORDINATOR")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(description='Timestamp Search - Fixed for reverse chrono data')
    
    parser.add_argument('dataset', help='Dataset file')
    parser.add_argument('--window', type=int, default=30)
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--offset', type=int, default=0,
                       help='Window start offset in dataset (0 = first draw)')
    parser.add_argument('--mode', 
                       choices=['microsecond', 'millisecond', 'centisecond', 'decisecond', 'second', 'minute'],
                       default='millisecond')
    parser.add_argument('--time-buffer', type=int)
    parser.add_argument('--skip-min', type=int, default=0)
    parser.add_argument('--skip-max', type=int, default=100)
    
    # Hybrid variable skip mode arguments
    parser.add_argument('--hybrid', action='store_true',
                       help='Enable hybrid variable skip detection (multi-strategy)')
    parser.add_argument('--phase1-threshold', type=float, default=0.20,
                       help='Phase 1 threshold for initial filtering (default: 0.20)')
    parser.add_argument('--phase2-threshold', type=float, default=0.75,
                       help='Phase 2 threshold for variable skip analysis (default: 0.75)')
    
    parser.add_argument('--prngs', nargs='+',
                       default=['xorshift32', 'lcg32', 'pcg32', 'mt19937', 'xorshift64'])
    parser.add_argument('--sessions', nargs='+', default=['midday', 'evening'])
    parser.add_argument('--output', '-o')
    parser.add_argument('--resume-policy', choices=['prompt', 'resume', 'restart', 'auto'], default='auto')
    parser.add_argument('--max-concurrent', type=int)
    parser.add_argument('--timeout', type=int)
    parser.add_argument('--dry-run', action='store_true')
    
    args = parser.parse_args()
    
    if not Path(args.dataset).exists():
        print(f"ERROR: Dataset not found: {args.dataset}")
        return 1
    
    print(f"{'='*70}")
    print("TIMESTAMP-BASED PRNG SEED SEARCH (FIXED)")
    print(f"{'='*70}")
    
    # Load dataset
    print(f"\n[1/5] Loading dataset: {args.dataset}")
    dataset_info = load_dataset_info(args.dataset, args.sessions, args.window)
    print(f"  ✓ Total entries: {dataset_info['total_entries']:,}")
    print(f"  ✓ Window size: {dataset_info['window_size']}")
    print(f"  ✓ Date range: {dataset_info['first_date']} to {dataset_info['last_date']} (CORRECTED)")
    print(f"  ✓ Sample draws: {dataset_info['sample_draws']}")
    
    # Calculate timestamps
    print(f"\n[2/5] Calculating {args.mode} timestamp range...")
    start_ts, end_ts, unit = calculate_timestamp_range(
        dataset_info['first_date'],
        dataset_info['last_date'],
        mode=args.mode,
        time_buffer=args.time_buffer
    )
    total_timestamps = end_ts - start_ts
    
    print(f"  ✓ Resolution: {args.mode}")
    print(f"  ✓ Start: {start_ts:,} ({datetime.fromtimestamp(start_ts if args.mode == 'second' else start_ts/1000)})")
    print(f"  ✓ End: {end_ts:,} ({datetime.fromtimestamp(end_ts if args.mode == 'second' else end_ts/1000)})")
    print(f"  ✓ Total {unit}: {total_timestamps:,} (POSITIVE)")
    
    # Performance estimate
    print(f"\n[3/5] Estimating performance...")
    gpus = 26
    throughput = gpus * 60_000_000
    estimated_time = total_timestamps / throughput
    print(f"  ✓ GPUs: {gpus}")
    print(f"  ✓ Throughput: {throughput:,} seeds/sec")
    print(f"  ✓ Time per PRNG: {estimated_time:.1f}s")
    print(f"  ✓ Total time ({len(args.prngs)} PRNGs): {estimated_time * len(args.prngs):.1f}s")
    
    # Config
    print(f"\n[4/5] Search configuration:")
    print(f"  ✓ PRNGs: {', '.join(args.prngs)}")
    print(f"  ✓ Threshold: {args.threshold:.1%} ({int(args.threshold*args.window)}/{args.window})")
    if args.hybrid:
        print(f"  ✓ Hybrid mode: ENABLED")
        print(f"    - Phase 1 threshold: {args.phase1_threshold:.1%}")
        print(f"    - Phase 2 threshold: {args.phase2_threshold:.1%}")
    else:
        print(f"  ✓ Skip mode: Fixed (standard)")
    print(f"  ✓ Skip: {args.skip_min}-{args.skip_max}")
    
    if not args.output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f"results/ts_search_{args.mode}_{timestamp}.json"
    print(f"  ✓ Output: {args.output}")
    
    if args.dry_run:
        print(f"\n{'='*70}")
        print("DRY RUN - No jobs submitted")
        print(f"{'='*70}")
        return 0
    
    # Submit
    print(f"\n[5/5] Submitting to coordinator...")
    return submit_to_coordinator(args.dataset, start_ts, end_ts, args.output, args)


if __name__ == '__main__':
    sys.exit(main())
