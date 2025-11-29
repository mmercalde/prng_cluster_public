import sys

with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

# Find the line to insert before
insert_line = None
for i, line in enumerate(lines):
    if 'seeds_gpu = cp.arange(chunk_start, chunk_end, dtype=cp.uint32)' in line and i > 250:
        insert_line = i
        break

if insert_line:
    # Insert dtype detection code
    indent = ' ' * 16  # Match indentation
    dtype_code = [
        f'{indent}# Determine dtype based on seed_type\n',
        f'{indent}seed_type = config.get("seed_type", "uint32")\n',
        f'{indent}dtype = cp.uint64 if seed_type == "uint64" else cp.uint32\n',
    ]
    
    # Replace hardcoded uint32 with dtype variable
    lines[insert_line] = lines[insert_line].replace('dtype=cp.uint32', 'dtype=dtype')
    lines[insert_line + 1] = lines[insert_line + 1].replace('dtype=cp.uint32', 'dtype=dtype')
    
    # Insert dtype detection
    lines = lines[:insert_line] + dtype_code + lines[insert_line:]
    
    with open('sieve_filter.py', 'w') as f:
        f.writelines(lines)
    
    print("✅ Fixed hybrid section dtype handling")
else:
    print("❌ Could not find insertion point")
