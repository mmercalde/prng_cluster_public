#!/usr/bin/env python3
"""
Per-Draw Timestamp Sieve
Each draw has its own timestamp-based seed
"""
import cupy as cp
import json
from datetime import datetime, timedelta
from prng_registry import get_kernel_info

def estimate_draw_time(date_str, session):
    """Estimate when draw occurred"""
    # Typical lottery draw times (adjust for your lottery)
    if session == 'midday':
        time_str = '12:30:00'
    else:  # evening
        time_str = '19:30:00'
    
    dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    return int(dt.timestamp())

def test_single_draw_timestamps(draw_value, timestamp_center, 
                                window_seconds=300, prng_family='xorshift32'):
    """
    Test if any timestamp within ±window_seconds can generate draw_value
    
    Returns: List of (seed, output) pairs that match
    """
    kernel_info = get_kernel_info(prng_family)
    
    # Simple single-output kernel
    kernel_code = r'''
    extern "C" __global__
    void test_single_output(
        unsigned int* seeds,
        unsigned int target,
        unsigned int* matches,
        unsigned int* match_count,
        int n_seeds
    ) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= n_seeds) return;
        
        unsigned int seed = seeds[idx];
        unsigned int state = seed;
        
        // Generate ONE output with xorshift32
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        
        // Check if matches target
        if ((state % 1000) == (target % 1000)) {
            unsigned int pos = atomicAdd(match_count, 1);
            if (pos < 10000) {  // Safety limit
                matches[pos] = seed;
            }
        }
    }
    '''
    
    kernel = cp.RawKernel(kernel_code, 'test_single_output')
    
    # Test range
    seed_start = timestamp_center - window_seconds
    seed_end = timestamp_center + window_seconds
    n_seeds = seed_end - seed_start
    
    seeds_gpu = cp.arange(seed_start, seed_end, dtype=cp.uint32)
    matches_gpu = cp.zeros(10000, dtype=cp.uint32)
    match_count_gpu = cp.zeros(1, dtype=cp.uint32)
    
    threads = 256
    blocks = (n_seeds + threads - 1) // threads
    
    kernel((blocks,), (threads,), (
        seeds_gpu, draw_value, matches_gpu, match_count_gpu, n_seeds
    ))
    
    cp.cuda.Stream.null.synchronize()
    
    count = int(match_count_gpu[0].get())
    if count > 0:
        return matches_gpu[:count].get().tolist()
    return []

def analyze_lottery_timestamps(dataset_path, sessions=['midday', 'evening']):
    """
    Find timestamp seeds for each draw independently
    """
    # Load data
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    filtered = [e for e in data if e.get('session') in sessions]
    
    # Test last 30 draws
    test_draws = filtered[-30:]
    
    print(f"=== Per-Draw Timestamp Analysis ===")
    print(f"Testing {len(test_draws)} draws independently\n")
    
    results = []
    
    for i, entry in enumerate(test_draws):
        date = entry['date']
        session = entry['session']
        draw = entry['draw']
        
        # Estimate timestamp
        est_timestamp = estimate_draw_time(date, session)
        
        print(f"Draw {i}: {date} {session} = {draw}")
        print(f"  Estimated timestamp: {est_timestamp} ({datetime.fromtimestamp(est_timestamp)})")
        
        # Find matching timestamps
        matches = test_single_draw_timestamps(draw, est_timestamp, window_seconds=300)
        
        print(f"  Found {len(matches)} timestamp candidates")
        if matches:
            print(f"  Candidates: {matches[:5]}{'...' if len(matches) > 5 else ''}")
            results.append({
                'draw_index': i,
                'date': date,
                'session': session,
                'draw_value': draw,
                'estimated_timestamp': est_timestamp,
                'candidate_seeds': matches
            })
        print()
    
    # Analyze patterns
    if len(results) >= 2:
        print("\n=== Pattern Analysis ===")
        for i in range(len(results) - 1):
            if results[i]['candidate_seeds'] and results[i+1]['candidate_seeds']:
                seed_1 = results[i]['candidate_seeds'][0]
                seed_2 = results[i+1]['candidate_seeds'][0]
                diff = seed_2 - seed_1
                print(f"Draw {i} to {i+1}: Δ = {diff} seconds")
    
    return results

if __name__ == "__main__":
    results = analyze_lottery_timestamps('daily3.json')
    
    # Save results
    with open('/tmp/per_draw_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to /tmp/per_draw_results.json")
