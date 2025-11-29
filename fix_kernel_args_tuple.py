#!/usr/bin/env python3
"""
Convert kernel_args to tuple before passing to kernel
"""

with open('sieve_filter.py', 'r') as f:
    content = f.read()

# Find and replace the kernel call
old_call = "kernel((blocks,), (threads_per_block,), kernel_args)"
new_call = "kernel((blocks,), (threads_per_block,), tuple(kernel_args))"

if old_call in content:
    content = content.replace(old_call, new_call)
    print(f"✅ Changed kernel call to use tuple(kernel_args)")
else:
    print(f"❌ Could not find the kernel call")

with open('sieve_filter.py', 'w') as f:
    f.write(content)

print("✅ Fixed sieve_filter.py")

