#!/usr/bin/env python3
"""
Fix distributed_worker.py JSON parsing to scan backwards for valid JSON
(matching coordinator's approach) instead of just taking the last line.
"""

import re

# Read the file
with open('/home/michael/distributed_prng_analysis/distributed_worker.py', 'r') as f:
    content = f.read()

# Old code block (script job path - around line 371-380)
old_block_1 = '''                        # Parse JSON result from last non-empty line of stdout
                        output_lines = [l for l in stdout.splitlines() if l.strip()]
                        if output_lines:
                            return json.loads(output_lines[-1])
                        else:
                            return {
                                'job_id': job_data.get('job_id', 'unknown'),
                                'success': False,
                                'error': f"No output from script. stderr: {stderr[-500:] if stderr else 'none'}"
                            }'''

# New code block - scan backwards like coordinator
new_block_1 = '''                        # Parse JSON result - scan backwards for valid JSON (like coordinator)
                        for line in reversed(stdout.splitlines()):
                            line = line.strip()
                            if line.startswith('{') and line.endswith('}'):
                                try:
                                    return json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                        # No valid JSON found
                        return {
                            'job_id': job_data.get('job_id', 'unknown'),
                            'success': False,
                            'error': f"No valid JSON in output. stderr: {stderr[-500:] if stderr else 'none'}"
                        }'''

# Also need to remove the json.JSONDecodeError except block since we handle it inline now
old_except_block = '''                    except json.JSONDecodeError as e:
                        return {
                            'job_id': job_data.get('job_id', 'unknown'),
                            'success': False,
                            'error': f"Invalid JSON output: {str(e)}. Output was: {output_lines[-1] if output_lines else 'empty'}"
                        }'''

new_except_block = '''                    except json.JSONDecodeError as e:
                        return {
                            'job_id': job_data.get('job_id', 'unknown'),
                            'success': False,
                            'error': f"Invalid JSON output: {str(e)}"
                        }'''

# Apply fixes
if old_block_1 in content:
    content = content.replace(old_block_1, new_block_1)
    print("✅ Fixed script job JSON parsing (block 1)")
else:
    print("❌ Could not find block 1 to replace")

if old_except_block in content:
    content = content.replace(old_except_block, new_except_block)
    print("✅ Fixed JSONDecodeError exception handler")
else:
    print("⚠️  JSONDecodeError block not found (may already be fixed or different)")

# Write back
with open('/home/michael/distributed_prng_analysis/distributed_worker.py', 'w') as f:
    f.write(content)

print("\n✅ distributed_worker.py updated!")
print("Now deploy to remote nodes:")
print("  scp ~/distributed_prng_analysis/distributed_worker.py michael@192.168.3.120:~/distributed_prng_analysis/")
print("  scp ~/distributed_prng_analysis/distributed_worker.py michael@192.168.3.154:~/distributed_prng_analysis/")
