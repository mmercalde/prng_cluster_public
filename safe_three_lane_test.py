from coordinator import MultiGPUCoordinator
from window_optimizer_integration_final import add_window_optimizer_to_coordinator, run_bidirectional_test
from window_optimizer import WindowConfig

add_window_optimizer_to_coordinator()
coordinator = MultiGPUCoordinator('distributed_config.json')

# Your W244_O139_S3-29 config with THREE-LANE FILTERS enabled
config = WindowConfig(
    window_size=244,
    offset=139,
    sessions=['evening'],
    skip_min=3,
    skip_max=29
)

print("="*70)
print("SAFE GPU-ACCELERATED THREE-LANE FILTERING")
print("Testing 10 MILLION seeds with W244_O139_S3-29")
print("="*70)

result = run_bidirectional_test(
    coordinator=coordinator,
    config=config,
    dataset_path='daily3.json',
    seed_start=0,
    seed_count=10_000_000,  # Only 10M seeds for safety
    prng_base='java_lcg',
    threshold=0.01  # 1% threshold
)

print(f"\n📊 RESULTS:")
print(f"  Forward survivors:      {result.forward_count:,}")
print(f"  Reverse survivors:      {result.reverse_count:,}")
print(f"  Bidirectional:          {result.bidirectional_count:,}")
print(f"  Precision:              {result.precision*100:.2f}%")
print(f"  Recall:                 {result.recall*100:.2f}%")

# Calculate survival rate
survival_rate = (result.bidirectional_count / 10_000_000) * 100
print(f"  Survival rate:          {survival_rate:.6f}%")

# Project to 1B seeds
projected_1B = result.bidirectional_count * 100
print(f"  Projected 1B seeds:     {projected_1B:,} survivors")

print("="*70)

if survival_rate > 0.1:
    print("🚨 EXTREMELY HIGH SURVIVAL RATE!")
    print("   This PRNG has very strong mathematical patterns")
elif survival_rate > 0.01:
    print("⚠️  High survival rate - PRNG has clear patterns")
else:
    print("✅ Reasonable survival rate")
