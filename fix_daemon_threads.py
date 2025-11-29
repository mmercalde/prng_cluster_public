#!/usr/bin/env python3
"""Remove daemon=True so threads can finish putting results"""

with open('coordinator.py', 'r') as f:
    content = f.read()

old = """            thread.daemon = True # Allow main thread to exit
            thread.start()"""

new = """            thread.daemon = False # Wait for all threads to complete properly
            thread.start()"""

if old in content:
    content = content.replace(old, new)
    with open('coordinator.py', 'w') as f:
        f.write(content)
    print("✅ Fixed daemon threads!")
else:
    print("❌ Pattern not found")
    exit(1)
