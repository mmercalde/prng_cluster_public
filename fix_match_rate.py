#!/usr/bin/env python3
"""
Fix match_rate -> score in continuous_learning_loop
"""

with open('reinforcement_engine.py', 'r') as f:
    content = f.read()

# Replace match_rate with score
old_line = "            hit_rate = predicted['match_rate']"
new_line = "            hit_rate = predicted['score']"

if old_line in content:
    content = content.replace(old_line, new_line)
    with open('reinforcement_engine.py', 'w') as f:
        f.write(content)
    print("✅ Fixed: match_rate -> score")
else:
    print("❌ Pattern not found - may already be fixed")
