#!/usr/bin/env python3
"""Remove local JobSpec redefinition and use imported one"""

with open('coordinator.py', 'r') as f:
    lines = f.readlines()

output = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # Skip the local JobSpec definition (lines ~1113-1130)
    if i >= 1112 and 'from dataclasses import dataclass, field' in line:
        # Skip until we find "job = JobSpec("
        while i < len(lines) and 'job = JobSpec(' not in lines[i]:
            i += 1
        # Now we're at "job = JobSpec(" line - keep it
        output.append(lines[i])
        i += 1
        continue
    
    output.append(line)
    i += 1

# Now fix the JobSpec instantiation to include required fields
for i, line in enumerate(output):
    if 'job = JobSpec(' in line and i > 1100 and i < 1150:
        # This is the sieve job creation - need to add mining_mode
        # Find the closing parenthesis
        j = i
        while ')' not in output[j] or 'search_type=' in output[j]:
            j += 1
        # Insert mining_mode before the closing paren
        output[j] = output[j].replace(
            "search_type='residue_sieve'",
            "mining_mode=False,\n                            search_type='residue_sieve'"
        )

with open('coordinator.py', 'w') as f:
    f.writelines(output)

print("✅ Fixed coordinator.py")
print("   - Removed local JobSpec class definition")
print("   - Using imported JobSpec from models.py")
print("   - Added required mining_mode field")
