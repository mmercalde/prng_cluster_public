import json
import numpy as np
from collections import Counter

def load_and_test_survivors():
    print("=== CONCRETE PREDICTION DEMONSTRATION ===")
    print("Testing if survivors can predict WITHIN historical data")
    
    # Try to load your forward survivors
    try:
        with open('results/window_opt_forward_244_139.json', 'r') as f:
            print("Loading forward survivors...")
            # Read first few lines to understand structure
            first_lines = []
            for i, line in enumerate(f):
                if i >= 5:  # Just check first 5 lines
                    break
                first_lines.append(line.strip())
            
            print("First few lines structure:")
            for i, line in enumerate(first_lines):
                print(f"  Line {i+1}: {line[:100]}...")
                
    except Exception as e:
        print(f"❌ Cannot load survivor file: {e}")
        print("Let me check what result files exist:")
        import glob
        files = glob.glob('results/*.json')
        for f in sorted(files)[:10]:
            print(f"  {f}")
        return

    # Load your lottery data to test prediction
    try:
        with open('daily3.json', 'r') as f:
            lottery_data = json.load(f)
            print(f"✅ Loaded {len(lottery_data)} lottery draws")
            
            # Show recent draws for context
            print("Recent draws (last 10):")
            for i, draw in enumerate(lottery_data[-10:]):
                print(f"  Draw {-9+i}: {draw}")
                
    except Exception as e:
        print(f"❌ Cannot load lottery data: {e}")
        return

    print("\n🎯 PREDICTION TEST PLAN:")
    print("1. Load survivors from your results")
    print("2. Use them to 'predict' already-known draws") 
    print("3. Measure accuracy on historical data")
    print("4. Only then consider future prediction")

def analyze_survivor_files():
    """Check what data we actually have to work with"""
    print("\n" + "="*50)
    print("ANALYZING AVAILABLE DATA FILES")
    print("="*50)
    
    import os
    import glob
    
    result_files = glob.glob('results/*.json')
    print(f"Found {len(result_files)} result files")
    
    for file in sorted(result_files)[:10]:  # Show first 10
        size_mb = os.path.getsize(file) / (1024*1024)
        print(f"  {file}: {size_mb:.1f} MB")
        
        # Try to peek at structure
        try:
            with open(file, 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    # Try to parse as JSON to understand structure
                    import json
                    data = json.loads(first_line)
                    if 'survivors' in data:
                        survivors = data['survivors']
                        count = len(survivors) if isinstance(survivors, list) else 1
                        print(f"    → Contains {count} survivors")
                    else:
                        print(f"    → Keys: {list(data.keys())}")
        except:
            print(f"    → Cannot parse structure")

if __name__ == "__main__":
    load_and_test_survivors()
    analyze_survivor_files()
