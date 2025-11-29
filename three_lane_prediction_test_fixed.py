import json
from collections import Counter

def load_three_lane_survivors():
    print("=== LOADING THREE-LANE FILTER SURVIVORS ===")
    print("Target: 27,902 bidirectional survivors from W244_O139_S3-29 test")
    
    forward_file = 'results/window_opt_forward_244_139.json'
    reverse_file = 'results/window_opt_reverse_244_139.json'
    
    def extract_survivor_seeds(data):
        """Extract seed values from survivor dictionaries"""
        seeds = set()
        if 'results' in data:
            for result in data['results']:
                if 'survivors' in result and isinstance(result['survivors'], list):
                    for survivor in result['survivors']:
                        if isinstance(survivor, dict) and 'seed' in survivor:
                            seeds.add(survivor['seed'])
                        elif isinstance(survivor, (int, float)):
                            seeds.add(int(survivor))
                        else:
                            print(f"Unexpected survivor type: {type(survivor)} - {survivor}")
        return seeds
    
    # Load forward survivors
    print(f"\nLoading forward survivors from: {forward_file}")
    forward_seeds = set()
    try:
        with open(forward_file, 'r') as f:
            data = json.load(f)
            print(f"Forward file structure: {list(data.keys())}")
            
            # Debug: show sample of survivors structure
            if 'results' in data and len(data['results']) > 0:
                sample_result = data['results'][0]
                print(f"Sample result keys: {list(sample_result.keys())}")
                if 'survivors' in sample_result and sample_result['survivors']:
                    sample_survivor = sample_result['survivors'][0]
                    print(f"Sample survivor type: {type(sample_survivor)}")
                    print(f"Sample survivor: {sample_survivor}")
            
            forward_seeds = extract_survivor_seeds(data)
            print(f"✅ Loaded {len(forward_seeds):,} forward survivor seeds")
            
    except Exception as e:
        print(f"❌ Error loading forward survivors: {e}")
        return None
    
    # Load reverse survivors  
    print(f"\nLoading reverse survivors from: {reverse_file}")
    reverse_seeds = set()
    try:
        with open(reverse_file, 'r') as f:
            data = json.load(f)
            print(f"Reverse file structure: {list(data.keys())}")
            
            reverse_seeds = extract_survivor_seeds(data)
            print(f"✅ Loaded {len(reverse_seeds):,} reverse survivor seeds")
            
    except Exception as e:
        print(f"❌ Error loading reverse survivors: {e}")
        return None
    
    # Calculate bidirectional survivors (intersection)
    bidirectional_seeds = forward_seeds & reverse_seeds
    print(f"\n🎯 BIDIRECTIONAL SURVIVOR SEEDS: {len(bidirectional_seeds):,}")
    print(f"   (Expected: 27,902 from your test results)")
    
    if bidirectional_seeds:
        print(f"Sample seeds: {list(bidirectional_seeds)[:10]}")
    
    return list(bidirectional_seeds)

def test_prediction_with_three_lane_survivors(seeds):
    print("\n" + "="*60)
    print("PREDICTION TEST WITH THREE-LANE FILTERED SURVIVORS")
    print("="*60)
    
    # Load lottery data
    try:
        with open('daily3.json', 'r') as f:
            lottery_data = json.load(f)
        print(f"✅ Loaded {len(lottery_data)} lottery draws")
    except Exception as e:
        print(f"❌ Failed to load lottery data: {e}")
        return
    
    # Show recent draws for context
    recent_draws = [d['draw'] for d in lottery_data[-10:]]
    print(f"Recent actual draws (last 10): {recent_draws}")
    
    # PREDICTION TEST
    print(f"\n🎯 Testing {len(seeds):,} three-lane filtered seeds...")
    
    matches = 0
    tested_seeds = 0
    detailed_matches = []
    
    # Test a subset for speed
    test_seeds = seeds[:500]  # Test first 500 seeds
    
    for seed in test_seeds:
        # Generate test sequence from this seed using simple LCG
        test_sequence = []
        current = seed
        for _ in range(100):  # Generate 100 values
            # Simple LCG simulation (you would use actual PRNG here)
            current = (current * 1103515245 + 12345) & 0x7fffffff
            test_sequence.append(current % 1000)  # Daily3 range
        
        # Check for pattern matches with recent draws
        found_match = False
        for i in range(len(test_sequence) - 3):
            # Check for 3-number sequence matches
            test_window = test_sequence[i:i+3]
            for j in range(len(recent_draws) - 3):
                actual_window = recent_draws[j:j+3]
                if test_window == actual_window:
                    matches += 1
                    found_match = True
                    detailed_matches.append({
                        'seed': seed,
                        'test_window': test_window,
                        'actual_window': actual_window,
                        'position': j
                    })
                    break
            if found_match:
                break
                
        tested_seeds += 1
    
    # Calculate accuracy
    accuracy = (matches / tested_seeds * 100) if tested_seeds > 0 else 0
    
    print(f"\n📊 PREDICTION TEST RESULTS:")
    print(f"   Seeds tested: {tested_seeds:,}")
    print(f"   Seeds that matched recent patterns: {matches:,}")
    print(f"   Pattern match rate: {accuracy:.2f}%")
    
    # Show detailed matches
    if detailed_matches:
        print(f"\n🔍 DETAILED MATCHES (first 5):")
        for match in detailed_matches[:5]:
            print(f"   Seed {match['seed']}: {match['test_window']} → Actual {match['actual_window']}")
    
    # Interpretation
    print(f"\n🎯 INTERPRETATION:")
    if accuracy > 5.0:
        print("   🚨 EXCELLENT PREDICTIVE POWER!")
        print("   Three-lane survivors can strongly predict lottery patterns")
    elif accuracy > 1.0:
        print("   ⚠️  GOOD PREDICTIVE POWER")
        print("   Survivors show meaningful predictive capability")  
    elif accuracy > 0.1:
        print("   📈 SOME PREDICTIVE POWER")
        print("   Potential for prediction with refinement")
    else:
        print("   🔍 LIMITED PREDICTIVE POWER")
        print("   May need different prediction approach")
    
    return accuracy, detailed_matches

if __name__ == "__main__":
    # Load the three-lane filtered seeds
    three_lane_seeds = load_three_lane_survivors()
    
    if three_lane_seeds:
        print(f"\n✅ SUCCESS: Loaded {len(three_lane_seeds):,} three-lane filtered seeds")
        
        # Test prediction capability
        accuracy, matches = test_prediction_with_three_lane_survivors(three_lane_seeds)
        
        if matches:
            print(f"\n🎯 NEXT STEPS:")
            print(f"   1. Use the {len(matches)} matching seeds for focused prediction")
            print(f"   2. Test with actual PRNG instead of simple LCG")
            print(f"   3. Build prediction engine with best-performing seeds")
    else:
        print(f"\n❌ Need to find where the 27,902 bidirectional survivors are stored")
