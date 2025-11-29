#!/usr/bin/env python3
"""
Fix xorshift64_cpu to match GPU kernel
Remove the incorrect mask after left shift operation
"""

with open('prng_registry.py', 'r') as f:
    content = f.read()

# Fix the left shift masking
old_line = "        state ^= (state << 25) & 0xFFFFFFFFFFFFFFFF"
new_line = "        state ^= state << 25"

# Replace both occurrences (in skip loop and output loop)
content = content.replace(old_line, new_line)

with open('prng_registry.py', 'w') as f:
    f.write(content)

print("✅ Fixed xorshift64_cpu!")
print("   Removed incorrect mask after left shift")
print("   CPU now matches GPU implementation")

