with open('prng_registry.py', 'r') as f:
    content = f.read()

# Add seed_type to philox4x32_reverse
content = content.replace(
    "    'philox4x32_reverse': {\n        'kernel_source': PHILOX4X32_REVERSE_KERNEL,\n        'kernel_name': 'philox4x32_reverse_sieve',\n        'description': 'philox4x32 reverse sieve - fixed skip backward validation'\n    },",
    "    'philox4x32_reverse': {\n        'kernel_source': PHILOX4X32_REVERSE_KERNEL,\n        'kernel_name': 'philox4x32_reverse_sieve',\n        'description': 'philox4x32 reverse sieve - fixed skip backward validation',\n        'seed_type': 'uint64'\n    },"
)

# Add seed_type to philox4x32_hybrid_reverse
content = content.replace(
    "    'philox4x32_hybrid_reverse': {\n        'kernel_source': PHILOX4X32_HYBRID_REVERSE_KERNEL,\n        'kernel_name': 'philox4x32_hybrid_reverse_sieve',\n        'description': 'philox4x32 hybrid reverse - variable skip backward validation',\n        'variable_skip': True\n    },",
    "    'philox4x32_hybrid_reverse': {\n        'kernel_source': PHILOX4X32_HYBRID_REVERSE_KERNEL,\n        'kernel_name': 'philox4x32_hybrid_reverse_sieve',\n        'description': 'philox4x32 hybrid reverse - variable skip backward validation',\n        'variable_skip': True,\n        'seed_type': 'uint64'\n    },"
)

with open('prng_registry.py', 'w') as f:
    f.write(content)

print("✅ Added seed_type to philox4x32_reverse entries")
