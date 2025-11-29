#!/usr/bin/env python3
"""
Manually fix the indentation
"""

with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

# Find line 496 (0-indexed: 495)
# Lines 496-608 need 4 more spaces

start_line = 495  # Line 496 (0-indexed)
end_line = 608    # Approximate end of two-phase block

print(f"Adding 4 spaces to lines {start_line+1} to {end_line}")

for i in range(start_line, min(end_line, len(lines))):
    line = lines[i]
    
    # Only indent non-empty lines
    if line.strip():
        # Check current indentation
        stripped = line.lstrip()
        current_indent = len(line) - len(stripped)
        
        # If it's at ~20-22 spaces, add 4
        if 18 <= current_indent <= 22 and not line.strip().startswith('elif'):
            lines[i] = '    ' + line
        elif current_indent > 22:
            # Already indented properly (nested content)
            pass

with open('sieve_filter.py', 'w') as f:
    f.writelines(lines)

print("✅ Fixed indentation manually")

