#!/usr/bin/env python3
"""Add hybrid arguments to command builder in timestamp_search.py"""

with open('timestamp_search.py', 'r') as f:
    content = f.read()

# Find and update the command builder
old_builder = """    if args.max_concurrent:
        cmd.extend(['--max-concurrent', str(args.max_concurrent)])
    if args.timeout:
        cmd.extend(['--job-timeout', str(args.timeout)])"""

new_builder = """    if args.max_concurrent:
        cmd.extend(['--max-concurrent', str(args.max_concurrent)])
    if args.timeout:
        cmd.extend(['--job-timeout', str(args.timeout)])
    
    # Add hybrid mode arguments if enabled
    if hasattr(args, 'hybrid') and args.hybrid:
        cmd.append('--hybrid')
        cmd.extend(['--phase1-threshold', str(args.phase1_threshold)])
        cmd.extend(['--phase2-threshold', str(args.phase2_threshold)])"""

if old_builder in content:
    content = content.replace(old_builder, new_builder)
    print("✅ Added hybrid arguments to command builder")
else:
    print("⚠️ Could not find command builder section")

# Write back
with open('timestamp_search.py', 'w') as f:
    f.write(content)

print("✅ Hybrid command builder updated")

# Test syntax
try:
    compile(content, 'timestamp_search.py', 'exec')
    print("✅ File syntax is valid!")
except SyntaxError as e:
    print(f"❌ Syntax error: {e}")
