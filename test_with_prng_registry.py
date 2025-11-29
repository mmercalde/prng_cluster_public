import json
import sys
import os

# Add current directory to path to import your PRNG registry
sys.path.insert(0, os.getcwd())

def test_with_actual_prng_registry():
    print("=== TESTING WITH YOUR ACTUAL PRNG REGISTRY ===")
    
    try:
        # Import your PRNG registry
        from prng_registry import KERNEL_REGISTRY, get_cpu_reference
        print("✅ Successfully imported PRNG registry")
        print(f"Available PRNGs: {len(KERNEL_REGISTRY)}")
    except ImportError as e:
        print(f"❌ Failed to import PRNG registry: {e}")
        return None
    
    # Load bidirectional seeds
    seeds = load_bidirectional_seeds()
    if not seeds:
        print("❌ No seeds loaded")
        return None
    
    # Load lottery data
    with open('daily3.json', 'r') as f:
        lottery_data = json.load(f)
    recent_draws = [d['draw'] for d in lottery_data[-10:]]
    print(f"Recent draws: {recent_draws}")
    print(f"Testing {len(seeds):,} bidirectional seeds")
    
    # Test each PRNG in your registry
    results = {}
    
    # Test first 10 PRNGs for speed
    test_prngs = list(KERNEL_REGISTRY.keys())[:10]
    
    for prng_name in test_prngs:
        print(f"\n🔍 Testing {prng_name}...")
        
        try:
            # Get the CPU reference implementation
            cpu_func = get_cpu_reference(prng_name)
            matches = 0
            tested_seeds = 0
            
            # Test first 50 seeds with this PRNG
            for seed in seeds[:50]:
                try:
                    # Generate sequence using the actual PRNG CPU implementation
                    sequence = cpu_func(seed, n=50, skip=0)
                    
                    # Convert to Daily3 format (0-999)
                    daily3_sequence = [x % 1000 for x in sequence]
                    
                    # Check for pattern matches
                    if check_sequence_matches(daily3_sequence, recent_draws):
                        matches += 1
                    
                    tested_seeds += 1
                    
                except Exception as e:
                    # Some PRNGs might not work with certain seeds
                    continue
            
            accuracy = (matches / tested_seeds * 100) if tested_seeds > 0 else 0
            results[prng_name] = {
                'accuracy': accuracy,
                'matches': matches,
                'tested': tested_seeds
            }
            
            print(f"  {prng_name}: {accuracy:.1f}% ({matches}/{tested_seeds})")
            
        except Exception as e:
            print(f"  ❌ {prng_name} failed: {e}")
            results[prng_name] = {'accuracy': 0, 'matches': 0, 'tested': 0, 'error': str(e)}
    
    return results

def load_bidirectional_seeds():
    """Load the 27,902 bidirectional seeds"""
    forward_file = 'results/window_opt_forward_244_139.json'
    reverse_file = 'results/window_opt_reverse_244_139.json'
    
    def extract_seeds(data):
        seeds = set()
        if 'results' in data:
            for result in data['results']:
                if 'survivors' in result and isinstance(result['survivors'], list):
                    for survivor in result['survivors']:
                        if isinstance(survivor, dict) and 'seed' in survivor:
                            seeds.add(survivor['seed'])
        return seeds
    
    try:
        with open(forward_file, 'r') as f:
            forward_data = json.load(f)
            forward_seeds = extract_seeds(forward_data)
        
        with open(reverse_file, 'r') as f:
            reverse_data = json.load(f)
            reverse_seeds = extract_seeds(reverse_data)
        
        bidirectional_seeds = list(forward_seeds & reverse_seeds)
        print(f"✅ Loaded {len(bidirectional_seeds):,} bidirectional seeds")
        return bidirectional_seeds
        
    except Exception as e:
        print(f"❌ Error loading seeds: {e}")
        return []

def check_sequence_matches(sequence, recent_draws):
    """Check if sequence contains any 3-number pattern from recent draws"""
    if len(sequence) < 3:
        return False
        
    for i in range(len(sequence) - 2):
        test_window = sequence[i:i+3]
        for j in range(len(recent_draws) - 2):
            if test_window == recent_draws[j:j+3]:
                return True
    return False

def main():
    print("🎯 PREDICTION TEST WITH YOUR 44 PRNG IMPLEMENTATIONS")
    print("=" * 60)
    
    results = test_with_actual_prng_registry()
    
    if results:
        print(f"\n📊 FINAL RESULTS:")
        print("=" * 40)
        
        # Sort by accuracy
        sorted_results = sorted(
            [(name, data) for name, data in results.items() if data['tested'] > 0],
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )
        
        for prng_name, data in sorted_results:
            accuracy = data['accuracy']
            matches = data['matches']
            tested = data['tested']
            
            if accuracy > 0:
                print(f"  {prng_name:20} {accuracy:5.1f}% ({matches:2d}/{tested:2d})")
        
        # Find best performer
        if sorted_results:
            best_name, best_data = sorted_results[0]
            best_accuracy = best_data['accuracy']
            
            if best_accuracy > 0:
                print(f"\n🎯 BEST PRNG: {best_name} ({best_accuracy:.1f}% accuracy)")
                print("🚀 NEXT: Use this PRNG with your matching seeds for prediction!")
            else:
                print(f"\n🔍 No strong matches found")
                print("The lottery PRNG might be:")
                print("  1. A custom implementation not in your registry")
                print("  2. Using different parameters than standard")
                print("  3. More complex than expected")
        else:
            print("❌ No PRNGs produced valid results")
    else:
        print("❌ No results generated")

if __name__ == "__main__":
    main()
