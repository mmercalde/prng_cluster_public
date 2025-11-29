#!/usr/bin/env python3
"""
Automated GPU Kernel Bug Finder for xoshiro256pp_reverse
Run this in ~/distributed_prng_analysis directory
"""

import re
import sys

def check_kernel_code():
    """Extract and analyze the GPU kernel code"""
    
    try:
        with open('prng_registry.py', 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print("❌ Error: prng_registry.py not found")
        print("   Run this script from ~/distributed_prng_analysis directory")
        sys.exit(1)
    
    # Extract xoshiro256pp_reverse kernel
    pattern = r'void xoshiro256pp_reverse_sieve.*?(?=\n\s{0,4}def |\n\s{0,4}void |\Z)'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        print("❌ Could not find xoshiro256pp_reverse_sieve kernel")
        sys.exit(1)
    
    kernel_code = match.group(0)
    
    print("="*70)
    print("GPU KERNEL BUG ANALYSIS")
    print("="*70)
    print()
    
    bugs_found = []
    warnings = []
    
    # Check 1: Output variable type
    print("1. Checking output variable type...")
    if re.search(r'unsigned int\s+output\s*=', kernel_code):
        bugs_found.append({
            'severity': 'CRITICAL',
            'issue': 'Output variable declared as unsigned int (32-bit)',
            'line': 'unsigned int output = ...',
            'fix': 'Change to: unsigned long long output = ...'
        })
        print("   ❌ CRITICAL BUG: Output is 32-bit! Should be 64-bit")
    elif re.search(r'unsigned long long\s+output\s*=', kernel_code):
        print("   ✅ Output variable is 64-bit")
    else:
        warnings.append("   ⚠️  Could not find output variable declaration")
    
    # Check 2: rotl return type
    print("\n2. Checking rotl function...")
    rotl_pattern = r'auto rotl\s*=\s*\[.*?\]\s*\((.*?)\)\s*(->.*?)?\s*\{'
    rotl_match = re.search(rotl_pattern, kernel_code, re.DOTALL)
    
    if rotl_match:
        has_return_type = rotl_match.group(2) is not None
        if not has_return_type or '-> unsigned long long' not in rotl_match.group(0):
            bugs_found.append({
                'severity': 'HIGH',
                'issue': 'rotl lambda missing explicit return type',
                'line': rotl_match.group(0)[:80],
                'fix': 'Add: -> unsigned long long before opening brace'
            })
            print("   ❌ HIGH: rotl missing explicit 64-bit return type")
        else:
            print("   ✅ rotl has explicit 64-bit return type")
    else:
        warnings.append("   ⚠️  Could not find rotl function")
    
    # Check 3: State initialization
    print("\n3. Checking state initialization...")
    expected_constants = {
        's1': '0x9E3779B97F4A7C15',
        's2': '0x6A09E667F3BCC908',
        's3': '0xBB67AE8584CAA73B'
    }
    
    for var, expected in expected_constants.items():
        pattern = rf'{var}\s*=\s*(0x[0-9A-Fa-f]+)'
        match = re.search(pattern, kernel_code)
        if match:
            actual = match.group(1).upper()
            expected_normalized = expected.upper()
            # Check with and without ULL suffix
            if actual.replace('ULL', '') == expected_normalized.replace('ULL', ''):
                print(f"   ✅ {var} = {expected}")
            else:
                bugs_found.append({
                    'severity': 'CRITICAL',
                    'issue': f'{var} initialization incorrect',
                    'line': f'{var} = {actual}',
                    'fix': f'Should be: {var} = {expected}ULL;'
                })
                print(f"   ❌ CRITICAL: {var} = {actual} (expected {expected})")
    
    # Check 4: Algorithm order
    print("\n4. Checking algorithm implementation...")
    
    # Look for the output generation line
    if 'rotl(s0 + s3, 23) + s0' in kernel_code:
        print("   ✅ Output formula: rotl(s0 + s3, 23) + s0")
    else:
        warnings.append("   ⚠️  Output formula might be incorrect")
    
    # Check 5: Comparison logic
    print("\n5. Checking comparison logic...")
    comparison_pattern = r'if\s*\(\s*\(\(.*?%\s*1000.*?\)\s*==.*?\)'
    if re.search(comparison_pattern, kernel_code):
        print("   ✅ Found 3-lane comparison with mod 1000")
    else:
        warnings.append("   ⚠️  Could not verify comparison logic")
    
    # Check 6: Type casting in arithmetic
    print("\n6. Checking for potential overflow issues...")
    # Look for additions without explicit casting
    add_pattern = r'(?<!unsigned long long\s)s[0-3]\s*\+\s*s[0-3]'
    if re.findall(add_pattern, kernel_code):
        warnings.append("   ⚠️  Found additions that might need explicit 64-bit casts")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if bugs_found:
        print(f"\n🚨 Found {len(bugs_found)} CRITICAL BUGS:\n")
        for i, bug in enumerate(bugs_found, 1):
            print(f"{i}. [{bug['severity']}] {bug['issue']}")
            print(f"   Current: {bug['line'][:80]}")
            print(f"   Fix: {bug['fix']}\n")
    else:
        print("\n✅ No critical bugs found in automated checks")
    
    if warnings:
        print(f"\n⚠️  {len(warnings)} Warnings:\n")
        for warning in warnings:
            print(warning)
    
    # Print relevant kernel section
    print("\n" + "="*70)
    print("RELEVANT KERNEL CODE (first 60 lines)")
    print("="*70)
    lines = kernel_code.split('\n')[:60]
    for i, line in enumerate(lines, 1):
        print(f"{i:3d}: {line}")
    
    return len(bugs_found) > 0

if __name__ == '__main__':
    print("Xoshiro256++ GPU Kernel Bug Finder")
    print()
    
    has_bugs = check_kernel_code()
    
    if has_bugs:
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("\n1. Fix the bugs listed above in prng_registry.py")
        print("2. Test with: python3 -c 'from prng_registry import KERNEL_REGISTRY'")
        print("3. Run the test again:")
        print("   python3 sieve_filter.py --job-file test_xoshiro_direct.json --gpu-id 0")
        sys.exit(1)
    else:
        print("\nNo obvious bugs found. The issue might be more subtle.")
        print("Consider running the manual kernel test from diagnose_gpu_kernel.md")
        sys.exit(0)
