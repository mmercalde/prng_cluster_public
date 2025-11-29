#!/usr/bin/env python3
"""
Properly fix xorshift64: GPU needs to mask left shift to match standard algorithm
"""

with open('prng_registry.py', 'r') as f:
    content = f.read()

# REVERT the CPU "fix" (put the mask back)
old_cpu = "        state ^= state << 25"
new_cpu = "        state ^= (state << 25) & 0xFFFFFFFFFFFFFFFF"
content = content.replace(old_cpu, new_cpu)

# FIX the GPU kernel (this won't work in CUDA - left shift is already masked!)
# Actually, in CUDA the left shift on uint64_t IS automatically masked
# So the issue must be something else...

print("Wait, let me reconsider...")
print("In CUDA, unsigned long long is 64-bit, so << 25 is automatically masked")
print("The GPU output of 255297705 suggests it IS using the masked version")
print("")
print("But we calculated that WITHOUT mask gives 1001498695")
print("And WITH mask gives 255297705")
print("")
print("So the GPU is using the MASKED version (which is correct)")
print("And we INCORRECTLY removed the mask from CPU")
print("")
print("Solution: REVERT the CPU change - put the mask back!")

with open('prng_registry.py', 'w') as f:
    f.write(content)

print("\n✅ Reverted CPU to use mask (correct xorshift64 algorithm)")

