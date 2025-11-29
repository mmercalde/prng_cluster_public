#!/usr/bin/env python3
"""
Add xorshift32_hybrid kernel in 2 steps:
1. Add the kernel definition (before KERNEL_REGISTRY)
2. Add the registry entry (inside KERNEL_REGISTRY, after mt19937_hybrid)
"""

# Read the generated kernel
with open('xorshift32_hybrid_kernel.txt', 'r') as f:
    kernel_code = f.read()

# Read current prng_registry.py
with open('prng_registry.py', 'r') as f:
    lines = f.readlines()

# STEP 1: Find where to add kernel definition (before KERNEL_REGISTRY line)
kernel_registry_line = None
for i, line in enumerate(lines):
    if line.strip() == 'KERNEL_REGISTRY = {':
        kernel_registry_line = i
        break

if kernel_registry_line is None:
    print("❌ Could not find KERNEL_REGISTRY = {")
    exit(1)

print(f"✅ Found KERNEL_REGISTRY at line {kernel_registry_line + 1}")

# Insert kernel definition before KERNEL_REGISTRY
kernel_definition = f"\n# Xorshift32 Hybrid Kernel\nXORSHIFT32_HYBRID_KERNEL = r'''{kernel_code}'''\n\n"
lines.insert(kernel_registry_line, kernel_definition)

# Adjust line numbers after insertion
kernel_registry_line += 1

# STEP 2: Find mt19937_hybrid entry and add after it
# We need to find the closing }, of mt19937_hybrid
in_mt19937_hybrid = False
mt19937_hybrid_end = None

for i in range(kernel_registry_line, len(lines)):
    line = lines[i]
    
    if "'mt19937_hybrid':" in line:
        in_mt19937_hybrid = True
        print(f"✅ Found mt19937_hybrid entry at line {i + 1}")
    
    if in_mt19937_hybrid and line.strip() == '},':
        # Check if next line is 'xorshift64' (that's where we insert)
        if i + 1 < len(lines) and "'xorshift64':" in lines[i + 1]:
            mt19937_hybrid_end = i + 1
            print(f"✅ Found insertion point at line {i + 2} (before xorshift64)")
            break

if mt19937_hybrid_end is None:
    print("❌ Could not find insertion point after mt19937_hybrid")
    exit(1)

# Create the registry entry
registry_entry = """    'xorshift32_hybrid': {
        'kernel_source': XORSHIFT32_HYBRID_KERNEL,
        'kernel_name': 'xorshift32_hybrid_multi_strategy_sieve',
        'cpu_reference': xorshift32_cpu,
        'default_params': {
            'shift_a': 13,
            'shift_b': 17,
            'shift_c': 5,
        },
        'description': 'Xorshift32 Hybrid - Variable skip pattern detection',
        'seed_type': 'uint32',
        'state_size': 4,
    },
"""

# Insert the entry
lines.insert(mt19937_hybrid_end, registry_entry)

# Write back
with open('prng_registry.py', 'w') as f:
    f.writelines(lines)

print("✅ Added XORSHIFT32_HYBRID_KERNEL definition")
print("✅ Added xorshift32_hybrid to KERNEL_REGISTRY")
print("\nVerifying...")

