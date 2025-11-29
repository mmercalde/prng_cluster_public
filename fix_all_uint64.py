with open('prng_registry.py', 'r') as f:
    content = f.read()

# Fix java_lcg_reverse
content = content.replace(
    "'java_lcg_reverse': {\n        'kernel_source': JAVA_LCG_REVERSE_KERNEL,\n        'kernel_name': 'java_lcg_reverse_sieve',\n        'description': 'java_lcg reverse sieve - fixed skip backward validation'\n    },",
    "'java_lcg_reverse': {\n        'kernel_source': JAVA_LCG_REVERSE_KERNEL,\n        'kernel_name': 'java_lcg_reverse_sieve',\n        'description': 'java_lcg reverse sieve - fixed skip backward validation',\n        'seed_type': 'uint64'\n    },"
)

# Fix java_lcg_hybrid_reverse
content = content.replace(
    "'java_lcg_hybrid_reverse': {\n        'kernel_source': JAVA_LCG_HYBRID_REVERSE_KERNEL,\n        'kernel_name': 'java_lcg_hybrid_reverse_sieve',\n        'description': 'java_lcg hybrid reverse - variable skip backward validation',\n        'variable_skip': True\n    },",
    "'java_lcg_hybrid_reverse': {\n        'kernel_source': JAVA_LCG_HYBRID_REVERSE_KERNEL,\n        'kernel_name': 'java_lcg_hybrid_reverse_sieve',\n        'description': 'java_lcg hybrid reverse - variable skip backward validation',\n        'variable_skip': True,\n        'seed_type': 'uint64'\n    },"
)

# Fix xorshift64_reverse
content = content.replace(
    "'xorshift64_reverse': {\n        'kernel_source': XORSHIFT64_REVERSE_KERNEL,\n        'kernel_name': 'xorshift64_reverse_sieve',\n        'description': 'xorshift64 reverse sieve - fixed skip backward validation'\n    },",
    "'xorshift64_reverse': {\n        'kernel_source': XORSHIFT64_REVERSE_KERNEL,\n        'kernel_name': 'xorshift64_reverse_sieve',\n        'description': 'xorshift64 reverse sieve - fixed skip backward validation',\n        'seed_type': 'uint64'\n    },"
)

# Fix xorshift64_hybrid_reverse
content = content.replace(
    "'xorshift64_hybrid_reverse': {\n        'kernel_source': XORSHIFT64_HYBRID_REVERSE_KERNEL,\n        'kernel_name': 'xorshift64_hybrid_reverse_sieve',\n        'description': 'xorshift64 hybrid reverse - variable skip backward validation',\n        'variable_skip': True\n    },",
    "'xorshift64_hybrid_reverse': {\n        'kernel_source': XORSHIFT64_HYBRID_REVERSE_KERNEL,\n        'kernel_name': 'xorshift64_hybrid_reverse_sieve',\n        'description': 'xorshift64 hybrid reverse - variable skip backward validation',\n        'variable_skip': True,\n        'seed_type': 'uint64'\n    },"
)

with open('prng_registry.py', 'w') as f:
    f.write(content)

print("✅ Added seed_type: 'uint64' to 4 reverse PRNGs")
