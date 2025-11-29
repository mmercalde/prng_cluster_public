import sys

with open('coordinator.py', 'r') as f:
    content = f.read()

# Add uuid import at top
if 'import uuid' not in content:
    content = content.replace('import json', 'import json\nimport uuid')

# Fix the job ID generation
old_line = '                "job_id": f"reverse_{i:03d}",'
new_line = '                "job_id": f"reverse_{uuid.uuid4().hex[:8]}",'
content = content.replace(old_line, new_line)

old_line2 = '                job_id=f"reverse_{i:03d}",'
new_line2 = '                job_id=f"reverse_{uuid.uuid4().hex[:8]}",'
content = content.replace(old_line2, new_line2)

with open('coordinator.py', 'w') as f:
    f.write(content)

print("✅ Fixed job ID collision issue")
