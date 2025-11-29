#!/usr/bin/env python3
"""
Generate dataset with LARGE skip gaps (0-50) to test skip discovery
"""
import json
from datetime import datetime, timedelta

def xorshift32_step(state):
    x = state & 0xFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= (x >> 17) & 0xFFFFFFFF
    x ^= (x << 5) & 0xFFFFFFFF
    return x & 0xFFFFFFFF

def generate_large_skip_dataset(base_seed=42, total_draws=5000):
    """
    Generate dataset with various skip patterns from 0-50
    Tests if sieve can discover large gaps
    """
    state = base_seed
    draws = []
    start_date = datetime(2020, 1, 1)
    
    print(f"Generating {total_draws} draws with LARGE skip gaps (0-50)...")
    print(f"Base seed: {base_seed}")
    
    # Pattern: Gradual increase then reset
    for draw_idx in range(total_draws):
        # Skip pattern: increases every 500 draws, then resets
        period = draw_idx // 500
        skip = min(period * 5, 50)  # 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
        
        # Advance (skip+1) times
        for _ in range(skip + 1):
            state = xorshift32_step(state)
        
        # Calculate date
        days_offset = draw_idx // 2
        session = "midday" if draw_idx % 2 == 0 else "evening"
        date = start_date + timedelta(days=days_offset)
        
        draws.append({
            "date": date.strftime("%Y-%m-%d"),
            "session": session,
            "draw": state % 1000,
            "full_state": int(state),
            "skip_used": skip
        })
    
    # Save with and without metadata
    output_file = "dataset_large_skip.json"
    clean_draws = [{k: v for k, v in d.items() if k != 'skip_used'} for d in draws]
    
    with open(output_file, 'w') as f:
        json.dump(clean_draws, f, indent=2)
    
    with open("dataset_large_skip_with_metadata.json", 'w') as f:
        json.dump(draws, f, indent=2)
    
    # Statistics
    print(f"\n✓ Generated {len(draws)} draws")
    print(f"  Skip progression: 0→5→10→15→20→25→30→35→40→45→50")
    print(f"  File saved: {output_file}")
    print(f"  Metadata file: dataset_large_skip_with_metadata.json")
    
    # Show skip distribution
    skip_counts = {}
    for d in draws:
        skip = d['skip_used']
        skip_counts[skip] = skip_counts.get(skip, 0) + 1
    
    print(f"\n  Skip distribution:")
    for skip in sorted(skip_counts.keys()):
        print(f"    skip={skip:2d}: {skip_counts[skip]:4d} draws")
    
    return draws

if __name__ == "__main__":
    draws = generate_large_skip_dataset()
    
    print("\n" + "="*70)
    print("TESTING STRATEGY:")
    print("="*70)
    print("Test recent windows with wide skip_range to see if sieve discovers:")
    print("  Window 4500-4529 (last 30): skip=45 expected")
    print("  Window 4000-4029: skip=40 expected")
    print("  Window 2500-2529: skip=25 expected")
    print("  Window 0-29: skip=0 expected")
    print("\nUse: --skip-range 0 50 to test all possible skips")
