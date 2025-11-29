#!/usr/bin/env python3
"""
Add xorshift32_hybrid kernel to prng_registry.py
"""

# Read the generated kernel
with open('xorshift32_hybrid_kernel.txt', 'r') as f:
    xorshift32_hybrid_kernel = f.read()

# Read prng_registry.py
with open('prng_registry.py', 'r') as f:
    content = f.read()

# Find where to insert (after MT19937_HYBRID_KERNEL)
insertion_point = content.find("# === KERNEL REGISTRY ===")
if insertion_point == -1:
    # Try alternative location
    insertion_point = content.find("KERNEL_REGISTRY = {")

if insertion_point == -1:
    print("❌ Could not find insertion point!")
    exit(1)

# Create the kernel definition
kernel_def = f"\n# Xorshift32 Hybrid Kernel\nXORSHIFT32_HYBRID_KERNEL = r'''{xorshift32_hybrid_kernel}'''\n\n"

# Insert before KERNEL_REGISTRY
content = content[:insertion_point] + kernel_def + content[insertion_point:]

# Now add to KERNEL_REGISTRY
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

# Find mt19937_hybrid in registry and add after it
mt_hybrid_pos = content.find("'mt19937_hybrid':")
if mt_hybrid_pos > 0:
    # Find the end of mt19937_hybrid entry (next },)
    end_pos = content.find("},", mt_hybrid_pos)
    if end_pos > 0:
        insert_pos = end_pos + 3  # After },\n
        content = content[:insert_pos] + registry_entry + content[insert_pos:]
        print("✅ Added xorshift32_hybrid to KERNEL_REGISTRY")
    else:
        print("❌ Could not find end of mt19937_hybrid entry")
        exit(1)
else:
    print("❌ Could not find mt19937_hybrid in registry")
    exit(1)

# Write back
with open('prng_registry.py', 'w') as f:
    f.write(content)

print("✅ Successfully added xorshift32_hybrid kernel to prng_registry.py")
print("\nNext steps:")
print("  1. Test locally: python3 test_xorshift32_hybrid_local.py")
print("  2. Test distributed: python3 test_xorshift32_hybrid.py")

