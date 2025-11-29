#!/usr/bin/env python3
"""
Evaluate prediction pools against actual draws.

Usage:
  python3 evaluate_pools.py --pools prediction_pools.json --truth daily3.json --date 2025-09-07 --session midday --out metrics.json
"""

import argparse
import json
from datetime import datetime, timezone
from typing import Dict, Optional

def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def find_draw(truth_file: str, date: str, session: str) -> Optional[int]:
    """Find the draw for a specific date and session."""
    with open(truth_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for entry in data:
        if entry.get('date') == date and entry.get('session') == session:
            return int(entry.get('draw'))
    
    return None

def evaluate_pools(pools_data: Dict, actual_draw: int) -> Dict:
    """Check if actual draw appears in each pool."""
    results = {}
    
    for pool_size, pool_info in pools_data['pools'].items():
        numbers = pool_info['numbers']
        hit = actual_draw in numbers
        
        # Calculate lift vs random
        # Random baseline: k/1000 probability
        k = int(pool_size)
        random_prob = k / 1000.0
        actual_prob = 1.0 if hit else 0.0
        lift = actual_prob / random_prob if random_prob > 0 else 0.0
        
        results[pool_size] = {
            'hit': hit,
            'size': k,
            'coverage': pool_info['coverage'],
            'random_baseline': round(random_prob, 4),
            'lift': round(lift, 2)
        }
    
    return results

def main():
    ap = argparse.ArgumentParser(description="Evaluate prediction pools against actual draws.")
    ap.add_argument("--pools", required=True, help="prediction_pools.json")
    ap.add_argument("--truth", required=True, help="daily3.json with actual draws")
    ap.add_argument("--date", required=True, help="Draw date (YYYY-MM-DD)")
    ap.add_argument("--session", required=True, help="Session: midday or evening")
    ap.add_argument("--out", default="metrics.json", help="Output metrics file")
    args = ap.parse_args()
    
    # Load pools
    with open(args.pools, 'r', encoding='utf-8') as f:
        pools_data = json.load(f)
    
    # Find actual draw
    actual_draw = find_draw(args.truth, args.date, args.session)
    
    if actual_draw is None:
        print(f"ERROR: No draw found for {args.date} {args.session}")
        return 1
    
    # Evaluate
    pool_results = evaluate_pools(pools_data, actual_draw)
    
    # Calculate overall metrics
    hit_any = any(r['hit'] for r in pool_results.values())
    hit_20 = pool_results.get('20', {}).get('hit', False)
    hit_100 = pool_results.get('100', {}).get('hit', False)
    hit_300 = pool_results.get('300', {}).get('hit', False)
    
    # Compile output
    metrics = {
        "version": 1,
        "run_id": iso_now(),
        "evaluation": {
            "date": args.date,
            "session": args.session,
            "actual_draw": actual_draw
        },
        "prediction_metadata": {
            "run_id": pools_data.get('run_id'),
            "survivor_count": pools_data.get('survivor_count'),
            "mapping": pools_data.get('mapping'),
            "prng_type": pools_data.get('prng_type')
        },
        "results": pool_results,
        "summary": {
            "hit_any_pool": hit_any,
            "hit_20": hit_20,
            "hit_100": hit_100,
            "hit_300": hit_300,
            "overall_lift": round(pool_results.get('100', {}).get('lift', 0.0), 2)
        }
    }
    
    # Save
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    # Console output
    print(f"\nEvaluation Results: {args.date} {args.session}")
    print(f"Actual draw: {actual_draw}")
    print(f"Survivors used: {pools_data.get('survivor_count')}")
    print(f"\nPool Performance:")
    for pool_size in ['20', '100', '300']:
        if pool_size in pool_results:
            r = pool_results[pool_size]
            status = "✓ HIT" if r['hit'] else "✗ MISS"
            print(f"  Top {pool_size:>3}: {status} | Lift: {r['lift']:>5.1f}x | Coverage: {r['coverage']*100:>5.1f}%")
    
    print(f"\nMetrics saved: {args.out}")
    
    return 0

if __name__ == "__main__":
    exit(main())
