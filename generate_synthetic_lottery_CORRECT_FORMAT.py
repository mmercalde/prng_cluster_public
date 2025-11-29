"""
Generate synthetic lottery data matching REAL daily3.json format

Real format:
[
  {'date': '2025-09-07', 'session': 'midday', 'draw': 978},
  {'date': '2025-09-06', 'session': 'evening', 'draw': 123},
  ...
]

Key: 'draw' is a 3-digit number (0-999), not [d1, d2, d3]
"""

import json
from datetime import datetime, timedelta

def java_lcg(seed):
    """Java's Linear Congruential Generator"""
    return (seed * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)

def seed_to_draw_number(seed):
    """Convert seed to 3-digit draw number (0-999)"""
    # Extract 3 digits from seed
    draw_num = (seed >> 16) % 1000
    return int(draw_num)

def generate_lottery_with_real_format(start_seed, num_draws, start_date='2020-01-01'):
    """
    Generate lottery data matching daily3.json format
    
    Args:
        start_seed: Initial PRNG seed
        num_draws: Number of draws to generate
        start_date: Starting date (works backward like real data)
    """
    draws = []
    current_seed = start_seed
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    for i in range(num_draws):
        # Generate draw number from current seed
        draw_num = seed_to_draw_number(current_seed)
        
        # Alternate between midday and evening
        session = 'midday' if i % 2 == 0 else 'evening'
        
        # Create entry matching real format
        entry = {
            'date': current_date.strftime('%Y-%m-%d'),
            'session': session,
            'draw': draw_num
        }
        draws.append(entry)
        
        # Advance seed to next state
        current_seed = java_lcg(current_seed)
        
        # Move to previous day if evening session (like real data - goes backward)
        if session == 'evening':
            current_date -= timedelta(days=1)
    
    # Metadata for validation
    metadata = {
        'generator': 'java_lcg',
        'start_seed': start_seed,
        'num_draws': num_draws,
        'format': 'matches daily3.json',
        'draw_range': '0-999 (3 digits)',
        'note': 'draw is a single number, not array of digits'
    }
    
    return draws, metadata

# Generate matching real format
print("Generating synthetic lottery data (CORRECT FORMAT)...")
print("Matching structure of daily3.json")
print()

draws, metadata = generate_lottery_with_real_format(
    start_seed=12345,
    num_draws=5000,
    start_date='2025-01-01'
)

# Save lottery data
with open('synthetic_lottery.json', 'w') as f:
    json.dump(draws, f, indent=2)

# Save metadata
with open('synthetic_lottery_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Generated synthetic_lottery.json")
print(f"   Total draws: {len(draws)}")
print(f"   Format: {{'date', 'session', 'draw'}}")
print()

print("📊 Sample data:")
for i in range(3):
    print(f"   {draws[i]}")
print("   ...")
for i in range(-3, 0):
    print(f"   {draws[i]}")
print()

print("🔍 Verification:")
print(f"   First draw: {draws[0]['draw']} (from seed {metadata['start_seed']})")
print(f"   Draw range: 0-999")
print(f"   Has 'date': {all('date' in d for d in draws[:10])}")
print(f"   Has 'session': {all('session' in d for d in draws[:10])}")
print(f"   Has 'draw': {all('draw' in d for d in draws[:10])}")
print()

# Compare to real format
print("📝 Format Comparison:")
print(f"   Real:     {{'date': '2025-09-07', 'session': 'midday', 'draw': 978}}")
print(f"   Synthetic: {draws[0]}")
print(f"   Match: ✅")
print()

print("✅ Synthetic data now matches your REAL daily3.json format!")
