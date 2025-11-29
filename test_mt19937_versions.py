#!/usr/bin/env python3
"""
Compare outputs of mt19937_cpu vs mt19937_cpu_simple
"""
import sys
sys.path.insert(0, '.')

# Import both functions
from prng_registry import mt19937_cpu

# We need to test mt19937_cpu_simple but it's broken, so let's fix it temporarily
# Or test with mt19937_cpu first to see what it produces

seed = 12345
n = 20

print("Testing mt19937_cpu (WORKING)...")
try:
    outputs_cpu = mt19937_cpu(seed, n, skip=0)
    print(f"✅ mt19937_cpu works!")
    print(f"   First 10 outputs: {outputs_cpu[:10]}")
except Exception as e:
    print(f"❌ mt19937_cpu failed: {e}")

print("\nTesting mt19937_cpu_simple (BROKEN)...")
try:
    # This will fail with NameError
    from prng_registry import mt19937_cpu_simple
    outputs_simple = mt19937_cpu_simple(seed, n, skip=0)
    print(f"✅ mt19937_cpu_simple works!")
    print(f"   First 10 outputs: {outputs_simple[:10]}")
    
    # Compare
    if outputs_cpu == outputs_simple:
        print("\n✅ IDENTICAL OUTPUTS - They're the same algorithm!")
    else:
        print("\n⚠️ DIFFERENT OUTPUTS - They implement different variants!")
        print(f"   Differences in first 10:")
        for i in range(10):
            if outputs_cpu[i] != outputs_simple[i]:
                print(f"     Position {i}: cpu={outputs_cpu[i]}, simple={outputs_simple[i]}")
                
except Exception as e:
    print(f"❌ mt19937_cpu_simple failed: {e}")
    print("\n📝 mt19937_cpu_simple is broken (NameError)")
    print("   We need to fix it first before comparing outputs")

