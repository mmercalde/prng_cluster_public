import json
import importlib
import sys
from collections import Counter

def discover_your_prng_modules():
    """Find your actual PRNG implementation files"""
    print("=== DISCOVERING YOUR 44 PRNG IMPLEMENTATIONS ===")
    
    prng_files = []
    # Common patterns for PRNG files
    patterns = ['*prng*.py', '*PRNG*.py', '*random*.py', '*generator*.py', '*.py']
    
    import glob
    for pattern in patterns:
        files = glob.glob(pattern)
        for file in files:
            # Exclude common non-PRNG files
            if (('prng' in file.lower() or 'random' in file.lower() or 'generator' in file.lower()) 
                and not file.startswith('test_') 
                and file != '__init__.py'):
                prng_files.append(file)
    
    print(f"Found {len(prng_files)} potential PRNG files:")
    for file in sorted(prng_files)[:20]:  # Show first 20
        print(f"  {file}")
    
    return prng_files

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
    
    with open(forward_file, 'r') as f:
        forward_data = json.load(f)
        forward_seeds = extract_seeds(forward_data)
    
    with open(reverse_file, 'r') as f:
        reverse_data = json.load(f)
        reverse_seeds = extract_seeds(reverse_data)
    
    return list(forward_seeds & reverse_seeds)

def test_with_prng_class(prng_class, seeds, recent_draws, class_name):
    """Test a specific PRNG class with the seeds"""
    matches = 0
    matching_seeds = []
    
    for seed in seeds[:50]:  # Test first 50 seeds
        try:
            # Try to initialize the PRNG with the seed
            prng = prng_class(seed)
            
            # Generate sequence
            sequence = []
            for _ in range(50):
                try:
                    # Try different common method names
                    if hasattr(prng, 'next_int'):
                        value = prng.next_int() % 1000
                    elif hasattr(prng, 'next'):
                        value = prng.next() % 1000
                    elif hasattr(prng, 'random'):
                        value = int(prng.random() * 1000)
                    elif hasattr(prng, 'generate'):
                        value = prng.generate() % 1000
                    else:
                        # Try calling directly
                        value = prng() % 1000
                    sequence.append(value)
                except:
                    break
            
            # Check for matches
            if check_sequence_matches(sequence, recent_draws):
                matches += 1
                matching_seeds.append(seed)
                
        except Exception as e:
            # This PRNG class might not work with simple initialization
            continue
    
    return matches, matching_seeds

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
    # Load data
    seeds = load_bidirectional_seeds()
    print(f"Loaded {len(seeds):,} bidirectional seeds")
    
    with open('daily3.json', 'r') as f:
        lottery_data = json.load(f)
    recent_draws = [d['draw'] for d in lottery_data[-10:]]
    print(f"Recent draws: {recent_draws}")
    
    # Discover PRNG modules
    prng_files = discover_your_prng_modules()
    
    if not prng_files:
        print("\n❌ No PRNG files found automatically.")
        print("Please tell me the names of your PRNG implementation files.")
        return
    
    print(f"\n🎯 TESTING WITH YOUR PRNG IMPLEMENTATIONS")
    print("="*50)
    
    results = {}
    
    for prng_file in prng_files[:10]:  # Test first 10 files
        try:
            # Import the module
            module_name = prng_file.replace('.py', '')
            print(f"\nTesting module: {module_name}")
            
            # Try to import
            spec = importlib.util.spec_from_file_location(module_name, prng_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find PRNG classes in the module
            prng_classes = []
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    attr_name not in ['__class__', '__base__'] and
                    not attr_name.startswith('_')):
                    prng_classes.append((attr_name, attr))
            
            print(f"  Found {len(prng_classes)} classes: {[name for name, _ in prng_classes]}")
            
            # Test each class
            for class_name, prng_class in prng_classes:
                try:
                    matches, matching_seeds = test_with_prng_class(
                        prng_class, seeds, recent_draws, class_name
                    )
                    
                    if matches > 0:
                        key = f"{module_name}.{class_name}"
                        results[key] = {
                            'matches': matches,
                            'tested': 50,
                            'matching_seeds': matching_seeds,
                            'accuracy': (matches / 50 * 100)
                        }
                        print(f"  ✅ {class_name}: {matches}/50 matches ({results[key]['accuracy']:.1f}%)")
                    else:
                        print(f"  ❌ {class_name}: 0/50 matches")
                        
                except Exception as e:
                    print(f"  ⚠️  {class_name}: Error - {e}")
                    
        except Exception as e:
            print(f"❌ Failed to import {prng_file}: {e}")
    
    # Show results
    if results:
        print(f"\n🎉 PREDICTION SUCCESS!")
        print("="*50)
        for prng_name, result in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            print(f"{prng_name}: {result['accuracy']:.1f}% accuracy")
            print(f"  Matching seeds: {result['matching_seeds'][:3]}")
    else:
        print(f"\n🔍 No matches found with discovered PRNGs")
        print("\nNext steps:")
        print("1. Tell me the exact names of your PRNG implementation files")
        print("2. We can test them directly by name")
        print("3. The PRNG might be in a different format or location")

if __name__ == "__main__":
    main()
