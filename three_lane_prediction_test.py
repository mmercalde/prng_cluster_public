import json
from collections import Counter

def load_three_lane_survivors():
    print("=== LOADING THREE-LANE FILTER SURVIVORS ===")
    print("Target: 27,902 bidirectional survivors from W244_O139_S3-29 test")
    
    # The bidirectional survivors should be in the intersection of forward and reverse files
    forward_file = 'results/window_opt_forward_244_139.json'
    reverse_file = 'results/window_opt_reverse_244_139.json'
    
    # Load forward survivors
    print(f"\nLoading forward survivors from: {forward_file}")
    forward_survivors = set()
    try:
        with open(forward_file, 'r') as f:
            data = json.load(f)
            print(f"Forward file structure: {list(data.keys())}")
            
            # Extract survivors from forward results
            if 'results' in data:
                for result in data['results']:
                    if 'survivors' in result and isinstance(result['survivors'], list):
                        forward_survivors.update(result['survivors'])
            print(f"✅ Loaded {len(forward_survivors):,} forward survivors")
    except Exception as e:
        print(f"❌ Error loading forward survivors: {e}")
        return None
    
    # Load reverse survivors  
    print(f"\nLoading reverse survivors from: {reverse_file}")
    reverse_survivors = set()
    try:
        with open(reverse_file, 'r') as f:
            data = json.load(f)
            print(f"Reverse file structure: {list(data.keys())}")
            
            # Extract survivors from reverse results
            if 'results' in data:
                for result in data['results']:
                    if 'survivors' in result and isinstance(result['survivors'], list):
                        reverse_survivors.update(result['survivors'])
            print(f"✅ Loaded {len(reverse_survivors):,} reverse survivors")
    except Exception as e:
        print(f"❌ Error loading reverse survivors: {e}")
        return None
    
    # Calculate bidirectional survivors (intersection)
    bidirectional_survivors = forward_survivors & reverse_survivors
    print(f"\n🎯 BIDIRECTIONAL SURVIVORS: {len(bidirectional_survivors):,}")
    print(f"   (Expected: 27,902 from your test results)")
    
    if len(bidirectional_survivors) == 0:
        print("❌ No bidirectional survivors found. Let me check the data structure...")
        # Debug: show sample of what's in the files
        with open(forward_file, 'r') as f:
            data = json.load(f)
            if 'results' in data and len(data['results']) > 0:
                sample_result = data['results'][0]
                print(f"Sample result keys: {list(sample_result.keys())}")
                if 'survivors' in sample_result:
                    survivors_data = sample_result['survivors']
                    print(f"Survivors type: {type(survivors_data)}")
                    if isinstance(survivors_data, list) and survivors_data:
                        print(f"Sample survivors: {survivors_data[:5]}")
        return None
    
    return list(bidirectional_survivors)

def test_prediction_with_three_lane_survivors(survivors):
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
    recent_draws = [d['draw'] for d in lottery_data[-20:]]  # Last 20 draws
    print(f"Recent draws (last 20): {recent_draws}")
    
    # SIMPLE PREDICTION TEST
    print(f"\n🎯 Testing {len(survivors):,} three-lane filtered survivors...")
    
    # Test 1: Check if survivors can reproduce recent patterns
    matches = 0
    tested_survivors = 0
    
    # Use a subset for quick testing
    test_survivors = survivors[:1000]  # Test first 1000
    
    for survivor in test_survivors:
        # For each survivor seed, generate a test sequence
        # This simulates what the PRNG would generate from this seed
        test_sequence = []
        current = survivor
        for _ in range(50):  # Generate 50 values
            # Simple LCG-like generation for testing
            current = (current * 1103515245 + 12345) & 0x7fffffff
            test_sequence.append(current % 1000)  # Daily3 range 0-999
        
        # Check if this sequence contains patterns from recent draws
        for i in range(len(test_sequence) - 3):
            window = test_sequence[i:i+3]
            # Check if any recent 3-draw pattern matches
            for j in range(len(recent_draws) - 3):
                recent_window = recent_draws[j:j+3]
                if window == recent_window:
                    matches += 1
                    break
            if matches > 0:  # Found at least one match
                break
                
        tested_survivors += 1
    
    accuracy = (matches / tested_survivors * 100) if tested_survivors > 0 else 0
    print(f"📊 PREDICTION TEST RESULTS:")
    print(f"   Survivors tested: {tested_survivors:,}")
    print(f"   Survivors that matched recent patterns: {matches:,}")
    print(f"   Pattern match rate: {accuracy:.1f}%")
    
    if accuracy > 1.0:
        print("🚨 HIGH MATCH RATE - Strong predictive potential!")
    elif accuracy > 0.1:
        print("⚠️  Moderate match rate - Some predictive value")
    else:
        print("✅ Low match rate - May need different approach")
    
    return accuracy

if __name__ == "__main__":
    # Step 1: Load the three-lane filtered survivors
    three_lane_survivors = load_three_lane_survivors()
    
    if three_lane_survivors:
        print(f"\n✅ SUCCESS: Loaded {len(three_lane_survivors):,} three-lane filtered survivors")
        
        # Step 2: Test prediction capability
        accuracy = test_prediction_with_three_lane_survivors(three_lane_survivors)
        
        # Step 3: Next steps based on results
        if accuracy > 0:
            print(f"\n🎯 NEXT: We can now build actual prediction with these {len(three_lane_survivors):,} high-quality seeds")
        else:
            print(f"\n🔍 NEXT: May need to adjust the prediction approach or test different patterns")
    else:
        print(f"\n❌ Could not load the three-lane survivors. Need to check data format.")
