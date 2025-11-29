#!/usr/bin/env python3
"""
Generate dataset using the SAME algorithm as the mt19937 kernel
This lets us verify the sieve works correctly
"""
import json
from datetime import datetime, timedelta

def fake_mt19937_step(state):
    """Exactly what the kernel does - NOT real MT19937!"""
    # Update state
    state = (1812433253 * (state ^ (state >> 30)) + 1) & 0xFFFFFFFF
    
    # Temper for output
    y = state
    y ^= y >> 11
    y ^= (y << 7) & 0x9D2C5680
    y ^= (y << 15) & 0xEFC60000
    y ^= y >> 18
    
    return state, y & 0xFFFFFFFF

def generate_fake_mt_dataset(base_seed=12345, total_draws=20000, skip=0):
    """
    Generate dataset using fake-MT algorithm
    skip: number of outputs to skip between recorded draws
    """
    state = base_seed
    draws = []
    start_date = datetime(2020, 1, 1)
    
    print(f"Generating {total_draws} draws with fake-MT algorithm")
    print(f"  Seed: {base_seed}")
    print(f"  Skip: {skip}")
    
    for draw_idx in range(total_draws):
        # Advance (skip+1) times
        for _ in range(skip + 1):
            state, output = fake_mt19937_step(state)
        
        # Calculate date (2 draws per day)
        days_offset = draw_idx // 2
        session = "midday" if draw_idx % 2 == 0 else "evening"
        date = start_date + timedelta(days=days_offset)
        
        draws.append({
            "date": date.strftime("%Y-%m-%d"),
            "session": session,
            "draw": int(output % 1000),
            "full_state": int(output),
            "skip_used": skip
        })
    
    # Save files
    output_file = f"dataset_fake_mt_{total_draws}_skip{skip}.json"
    clean_draws = [{k: v for k, v in d.items() if k != 'skip_used'} for d in draws]
    
    with open(output_file, 'w') as f:
        json.dump(clean_draws, f, indent=2)
    
    with open(f"dataset_fake_mt_{total_draws}_skip{skip}_with_metadata.json", 'w') as f:
        json.dump(draws, f, indent=2)
    
    print(f"\n✓ Generated {len(draws)} draws")
    print(f"  Date range: {draws[0]['date']} to {draws[-1]['date']}")
    print(f"  First 10 draws: {[d['draw'] for d in draws[:10]]}")
    print(f"  File saved: {output_file}")
    
    return draws

if __name__ == "__main__":
    # Create dataset with skip=0
    draws = generate_fake_mt_dataset(base_seed=12345, total_draws=20000, skip=0)
    
    print("\n" + "="*70)
    print("TEST THIS WITH:")
    print("="*70)
    print("Seed: 12345")
    print("Skip: 0")
    print("Window: Last 30 draws (or any window)")
    print("Expected result: Should find seed 12345 with 100% match rate")
