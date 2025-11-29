#!/usr/bin/env python3
"""
Generate 10K draw dataset with VARIABLE skip rates
Simulates real-world PRNG behavior with changing patterns
"""

import json
import random
from datetime import datetime, timedelta

def xorshift32_step(state):
    """Reference xorshift32 implementation"""
    x = state & 0xFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= (x >> 17) & 0xFFFFFFFF
    x ^= (x << 5) & 0xFFFFFFFF
    return x & 0xFFFFFFFF

def generate_variable_skip_dataset(base_seed=42, total_draws=10000, skip_pattern="periodic"):
    """
    Generate dataset with variable skip rates
    
    skip_pattern options:
    - "periodic": Skip changes every N draws (simulates reseeding)
    - "random": Random skip per draw (chaotic)
    - "drift": Gradually increasing skip (drift over time)
    - "burst": Periods of constant skip with sudden changes
    - "mixed": Combination of patterns
    """
    
    state = base_seed
    draws = []
    start_date = datetime(2020, 1, 1)
    
    # Track what skip was used for each draw
    skip_log = []
    
    print(f"Generating {total_draws} draws with '{skip_pattern}' skip pattern...")
    print(f"Base seed: {base_seed}")
    
    for draw_idx in range(total_draws):
        # Determine skip for this draw based on pattern
        if skip_pattern == "periodic":
            # Change skip every 1000 draws (simulates periodic reseeding)
            period = draw_idx // 1000
            skip = period % 8  # Cycles through 0-7
            
        elif skip_pattern == "random":
            # Completely random skip each time (0-10)
            skip = random.randint(0, 10)
            
        elif skip_pattern == "drift":
            # Gradually increasing skip (drift)
            skip = min((draw_idx // 500), 15)  # Increases every 500 draws, max 15
            
        elif skip_pattern == "burst":
            # Stable periods with sudden jumps
            if draw_idx % 2000 < 1000:
                skip = 2  # First half: skip=2
            else:
                skip = 7  # Second half: skip=7
                
        elif skip_pattern == "mixed":
            # Realistic mix: mostly stable with occasional changes
            if draw_idx < 3000:
                skip = 0  # First 3000: no skip
            elif draw_idx < 5000:
                skip = 3  # Next 2000: skip=3
            elif draw_idx < 7000:
                skip = 1  # Next 2000: skip=1
            elif draw_idx < 9000:
                skip = 5  # Next 2000: skip=5
            else:
                skip = 2  # Last 1000: skip=2
        
        else:
            raise ValueError(f"Unknown skip_pattern: {skip_pattern}")
        
        # Advance PRNG state (skip+1) times
        for _ in range(skip + 1):
            state = xorshift32_step(state)
        
        # Calculate date (2 draws per day: midday/evening)
        days_offset = draw_idx // 2
        session = "midday" if draw_idx % 2 == 0 else "evening"
        date = start_date + timedelta(days=days_offset)
        
        draws.append({
            "date": date.strftime("%Y-%m-%d"),
            "session": session,
            "draw": state % 1000,
            "full_state": int(state),
            "skip_used": skip  # Metadata for verification
        })
        
        skip_log.append(skip)
    
    # Statistics
    unique_skips = sorted(set(skip_log))
    skip_changes = sum(1 for i in range(1, len(skip_log)) if skip_log[i] != skip_log[i-1])
    
    print(f"\n✓ Generated {len(draws)} draws")
    print(f"  Unique skip values used: {unique_skips}")
    print(f"  Skip value changes: {skip_changes}")
    print(f"  Date range: {draws[0]['date']} to {draws[-1]['date']}")
    print(f"  First 10 skips: {skip_log[:10]}")
    print(f"  Last 10 skips: {skip_log[-10:]}")
    
    return draws, skip_log

def analyze_windows(draws, skip_log, window_size=30):
    """
    Analyze different windows to see what skip values dominate
    """
    print(f"\n{'='*60}")
    print(f"Window Analysis (window_size={window_size})")
    print(f"{'='*60}")
    
    # Check windows at different positions
    positions = [0, 1000, 3000, 5000, 7000, 9000, 9970]  # Last one is last 30
    
    for pos in positions:
        if pos + window_size > len(draws):
            continue
            
        window_skips = skip_log[pos:pos+window_size]
        unique = set(window_skips)
        dominant = max(set(window_skips), key=window_skips.count)
        dominant_pct = (window_skips.count(dominant) / len(window_skips)) * 100
        
        print(f"\nDraws {pos}-{pos+window_size-1}:")
        print(f"  Unique skips in window: {sorted(unique)}")
        print(f"  Dominant skip: {dominant} ({dominant_pct:.1f}% of window)")
        print(f"  Date range: {draws[pos]['date']} to {draws[pos+window_size-1]['date']}")

def save_dataset(draws, filename, include_skip_metadata=False):
    """Save dataset to file"""
    
    # Remove skip metadata if not requested (simulate real data)
    if not include_skip_metadata:
        clean_draws = []
        for d in draws:
            clean = {k: v for k, v in d.items() if k != 'skip_used'}
            clean_draws.append(clean)
        draws = clean_draws
    
    with open(filename, 'w') as f:
        json.dump(draws, f, indent=2)
    
    print(f"\n✓ Saved to {filename}")
    print(f"  File size: {len(json.dumps(draws)) / 1024:.1f} KB")

def main():
    """Generate multiple test datasets with different skip patterns"""
    
    patterns = {
        "periodic": "Skip changes every 1000 draws (0→1→2→...→7)",
        "random": "Random skip 0-10 for each draw",
        "drift": "Gradually increasing skip (simulates degradation)",
        "burst": "Stable skip=2 and skip=7 periods",
        "mixed": "Realistic: long stable periods with occasional changes"
    }
    
    print("="*60)
    print("VARIABLE SKIP RATE DATASET GENERATOR")
    print("="*60)
    print("\nAvailable patterns:")
    for pattern, desc in patterns.items():
        print(f"  {pattern}: {desc}")
    
    # Generate all patterns
    for pattern in patterns.keys():
        print(f"\n{'#'*60}")
        print(f"# Generating: {pattern}")
        print(f"{'#'*60}")
        
        draws, skip_log = generate_variable_skip_dataset(
            base_seed=42,
            total_draws=10000,
            skip_pattern=pattern
        )
        
        # Analyze windows
        analyze_windows(draws, skip_log, window_size=30)
        
        # Save both versions
        save_dataset(draws, f"dataset_10k_{pattern}_with_metadata.json", include_skip_metadata=True)
        save_dataset(draws, f"dataset_10k_{pattern}.json", include_skip_metadata=False)
        
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print("\nGenerated files:")
    print("  dataset_10k_periodic.json - Period-based skip changes")
    print("  dataset_10k_random.json - Chaotic random skips")
    print("  dataset_10k_drift.json - Gradually increasing skip")
    print("  dataset_10k_burst.json - Sudden skip transitions")
    print("  dataset_10k_mixed.json - Realistic mixed pattern (RECOMMENDED)")
    print("\nAll files also saved with _with_metadata.json for verification")
    
    print("\n" + "="*60)
    print("RECOMMENDED TEST STRATEGY")
    print("="*60)
    print("""
1. Test on 'mixed' pattern (most realistic):
   - Sieve should find seed 42 in windows 0-30, 3000-3030, etc.
   - Different windows will have different optimal skip values
   
2. Test window selection matters:
   - Window at draws 0-30: skip should be 0
   - Window at draws 3000-3030: skip should be 3
   - Window at draws 5000-5030: skip should be 1
   
3. Verify sieve adapts:
   - Run sieve on different window positions
   - Each should find seed 42 but with correct skip for that window
   
4. Test full scan:
   - Sieve entire 10K draws with skip_range=[0,10]
   - Should identify multiple survivor windows with different skips
""")

if __name__ == "__main__":
    main()
