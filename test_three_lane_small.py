from coordinator import MultiGPUCoordinator

def test_small_scale():
    print("=== TESTING THREE-LANE FILTERS AT SMALL SCALE ===")
    
    coordinator = MultiGPUCoordinator('distributed_config.json')
    
    # Test with only 1 million seeds to see the survival rate
    result = coordinator.optimize_window(
        dataset_path='daily3.json',
        seed_start=0,
        seed_count=1_000_000,  # Only 1M seeds
        prng_base='java_lcg',
        strategy_name='grid',
        max_iterations=1,
        output_file='small_test_1M.json'
    )
    
    if result and 'best_result' in result:
        survivors = result['best_result'].get('bidirectional_count', 0)
        survival_rate = (survivors / 1_000_000) * 100
        
        print(f"📊 RESULTS from 1M seeds:")
        print(f"   Survivors found: {survivors:,}")
        print(f"   Survival rate: {survival_rate:.4f}%")
        
        # Project to 1B seeds
        projected_1B = survivors * 1000
        print(f"   Projected for 1B seeds: {projected_1B:,} survivors")
        
        if survival_rate > 0.1:  # More than 0.1% survival rate
            print("🚨 WARNING: Extremely high survival rate!")
            print("   This PRNG has very strong mathematical patterns")
        else:
            print("✅ Reasonable survival rate")
    
    return result

if __name__ == "__main__":
    test_small_scale()
