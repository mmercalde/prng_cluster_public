#!/usr/bin/env python3
"""Add hybrid arguments to coordinator.py argument parser"""

with open('coordinator.py', 'r') as f:
    content = f.read()

# Find and update the argument section
old_args = """    parser.add_argument('--threshold', type=float, default=0.6, help='Match threshold for sieve (0.0-1.0)')
    parser.add_argument('--session-filter', choices=['midday', 'evening', 'both'],"""

new_args = """    parser.add_argument('--threshold', type=float, default=0.6, help='Match threshold for sieve (0.0-1.0)')
    
    # Hybrid variable skip mode arguments
    parser.add_argument('--hybrid', action='store_true',
                       help='Enable hybrid variable skip detection (multi-strategy)')
    parser.add_argument('--phase1-threshold', type=float, default=0.20,
                       help='Phase 1 threshold for initial filtering (default: 0.20)')
    parser.add_argument('--phase2-threshold', type=float, default=0.75,
                       help='Phase 2 threshold for variable skip analysis (default: 0.75)')
    
    parser.add_argument('--session-filter', choices=['midday', 'evening', 'both'],"""

if old_args in content:
    content = content.replace(old_args, new_args)
    print("✅ Added hybrid arguments to coordinator.py parser")
else:
    print("⚠️ Could not find argument insertion point")

# Write back
with open('coordinator.py', 'w') as f:
    f.write(content)

print("✅ Hybrid arguments added to coordinator.py")

# Test syntax
try:
    compile(content, 'coordinator.py', 'exec')
    print("✅ File syntax is valid!")
except SyntaxError as e:
    print(f"❌ Syntax error: {e}")
