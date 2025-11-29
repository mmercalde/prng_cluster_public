#!/usr/bin/env python3
"""
Verify EXACTLY what parameters are needed vs what we're passing
"""

print("="*70)
print("PARAMETER VERIFICATION")
print("="*70)

# 1. What does xorshift32_hybrid kernel expect?
print("\n1. XORSHIFT32_HYBRID KERNEL SIGNATURE:")
print("   From prng_registry.py line ~969:")

with open('prng_registry.py', 'r') as f:
    content = f.read()
    start = content.find('void xorshift32_hybrid_multi_strategy_sieve(')
    end = content.find(') {', start)
    signature = content[start:end+2]
    
    # Parse parameters
    params = signature.split('\n')
    print("\n   Parameters in order:")
    for i, p in enumerate(params):
        if p.strip() and 'void' not in p:
            print(f"   {i}. {p.strip()}")

# 2. What does mt19937_hybrid kernel expect?
print("\n\n2. MT19937_HYBRID KERNEL SIGNATURE (for comparison):")

start = content.find('void mt19937_hybrid_multi_strategy_sieve(')
end = content.find(') {', start)
signature = content[start:end+2]

params = signature.split('\n')
print("\n   Parameters in order:")
for i, p in enumerate(params):
    if p.strip() and 'void' not in p:
        print(f"   {i}. {p.strip()}")

# 3. What is sieve_filter.py currently passing?
print("\n\n3. WHAT SIEVE_FILTER.PY IS CURRENTLY PASSING:")

with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'kernel_args = [' in line:
        context = ''.join(lines[max(0,i-5):i+1])
        if 'strategy_max_misses' in ''.join(lines[i-10:i+20]):
            print(f"\n   Found at line {i+1}:")
            # Print the kernel_args array
            j = i
            depth = 0
            arg_num = 1
            while j < len(lines):
                line_content = lines[j].strip()
                print(f"   {arg_num}. {line_content}")
                if '[' in lines[j]:
                    depth += 1
                if ']' in lines[j]:
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
                if line_content and line_content != '[':
                    arg_num += 1
            break

print("\n\n4. THE DIFFERENCE:")
print("   xorshift32_hybrid needs: ..., threshold, shift_a, shift_b, shift_c, offset")
print("   Currently passing:       ..., threshold, offset")
print("   ❌ MISSING: shift_a, shift_b, shift_c")

print("\n5. DEFAULT PARAMS:")
from prng_registry import get_kernel_info
config = get_kernel_info('xorshift32_hybrid')
print(f"   xorshift32_hybrid default_params: {config.get('default_params')}")

print("\n✅ CONFIRMED: We need to add shift_a, shift_b, shift_c before offset")

