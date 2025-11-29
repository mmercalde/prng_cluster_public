#!/usr/bin/env python3
"""Fix missing config in run_hybrid_sieve"""

import re

with open('sieve_filter.py', 'r') as f:
    content = f.read()

# Fix 1: Add config parameter to function signature
content = content.replace(
    'def run_hybrid_sieve(self,',
    'def run_hybrid_sieve(self, config,'
)

# Fix 2: Add config to function call
content = re.sub(
    r'result = sieve\.run_hybrid_sieve\(',
    'result = sieve.run_hybrid_sieve(config, ',
    content
)

with open('sieve_filter.py', 'w') as f:
    f.write(content)

print("✅ Fixed run_hybrid_sieve to accept config parameter")
