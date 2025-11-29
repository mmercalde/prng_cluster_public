#!/usr/bin/env python3
import re

with open('prng_registry.py', 'r') as f:
    content = f.read()

# Fix LCG32_REVERSE_KERNEL signature
content = re.sub(
    r'(void lcg32_reverse_sieve\([^)]*?)unsigned int a, unsigned int c, unsigned int m, (int offset\))',
    r'\1\2',
    content
)

# Add constants after opening brace in lcg32_reverse_sieve
content = re.sub(
    r'(void lcg32_reverse_sieve\([^)]*?\) \{\n)',
    r'\1    const unsigned int a = 1103515245;\n    const unsigned int c = 12345;\n    const unsigned int m = 0x7FFFFFFF;\n',
    content
)

# Fix LCG32_HYBRID_REVERSE_KERNEL signature  
content = re.sub(
    r'(void lcg32_hybrid_reverse_sieve\([^)]*?)unsigned int a, unsigned int c, unsigned int m, (int offset\))',
    r'\1\2',
    content
)

# Add constants after opening brace in lcg32_hybrid_reverse_sieve
content = re.sub(
    r'(void lcg32_hybrid_reverse_sieve\([^)]*?\) \{\n)',
    r'\1    const unsigned int a = 1103515245;\n    const unsigned int c = 12345;\n    const unsigned int m = 0x7FFFFFFF;\n',
    content
)

with open('prng_registry.py', 'w') as f:
    f.write(content)

print("✅ Fixed!")
