#!/usr/bin/env python3
"""
Analyze mt19937 hybrid kernel to create a template
"""

print("="*70)
print("ANALYZING MT19937 HYBRID KERNEL STRUCTURE")
print("="*70)

with open('prng_registry.py', 'r') as f:
    content = f.read()

# Extract the hybrid kernel
start = content.find('MT19937_HYBRID_KERNEL = r"""')
end = content.find('"""', start + 30)
hybrid_kernel = content[start:end]

print("\n1. KERNEL SIGNATURE:")
sig_start = hybrid_kernel.find('void mt19937_hybrid')
sig_end = hybrid_kernel.find('{', sig_start)
print(hybrid_kernel[sig_start:sig_end])

print("\n2. KEY PARAMETERS:")
print("   - strategy_max_misses[] - array of max misses per strategy")
print("   - strategy_tolerances[] - tolerance levels")
print("   - n_strategies - number of patterns to test")
print("   - skip_sequences[] - OUTPUT: full pattern array")

print("\n3. STRATEGY LOOP STRUCTURE:")
lines = hybrid_kernel.split('\n')
for i, line in enumerate(lines):
    if 'for (int strat' in line:
        print(f"   Line {i}: {line.strip()}")
        # Show next 5 lines
        for j in range(i+1, min(i+6, len(lines))):
            print(f"   Line {j}: {lines[j].strip()}")
        break

print("\n4. STATE ADVANCEMENT (PRNG-SPECIFIC PART):")
for i, line in enumerate(lines):
    if 'mt19937_extract' in line or 'state ^=' in line:
        print(f"   Line {i}: {line.strip()}")
        if i > 500:  # Only show first occurrence
            break

print("\n" + "="*70)
print("TEMPLATE STRUCTURE:")
print("="*70)
print("""
1. Same signature for all PRNGs (just change name)
2. Keep strategy testing logic (same for all)
3. Keep 3-lane matching (mod 1000, mod 8, mod 125)
4. ONLY CHANGE: State advancement (PRNG-specific)
5. Same output: skip_sequences[] array

NEXT STEP: Create xorshift32_hybrid using this template
""")

