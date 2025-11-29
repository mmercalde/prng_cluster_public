#!/usr/bin/env python3
"""
Add window optimizer arguments to coordinator.py
"""

# First, let's find where to insert the arguments
with open('coordinator.py', 'r') as f:
    content = f.read()

# Find the argparse section - look for the last argument before the parse
# We'll insert after the --draw-match argument

insert_marker = "parser.add_argument('--draw-match', type=int, help='Target draw number for matching analysis')"

if insert_marker not in content:
    print("❌ Could not find insertion point!")
    print("Looking for alternative marker...")
    
    # Try alternative marker
    insert_marker = "parser.add_argument('--session-filter'"
    
    if insert_marker not in content:
        print("❌ Could not find session-filter either!")
        exit(1)

# The new arguments to add
new_arguments = """
    
    # Window Optimization arguments
    parser.add_argument('--optimize-window', action='store_true',
                       help='Run window optimization to find best configuration')
    parser.add_argument('--opt-strategy', 
                       choices=['random', 'grid', 'bayesian', 'evolutionary'],
                       default='bayesian',
                       help='Optimization search strategy (default: bayesian)')
    parser.add_argument('--opt-iterations', type=int, default=50,
                       help='Maximum optimization iterations (default: 50)')
    parser.add_argument('--opt-seed-count', type=int, default=10_000_000,
                       help='Number of seeds to test per configuration (default: 10M)')
"""

# Find the position and insert
lines = content.split('\n')
new_lines = []
inserted = False

for i, line in enumerate(lines):
    new_lines.append(line)
    if insert_marker in line and not inserted:
        # Insert after this line (and any continuation lines)
        new_lines.append(new_arguments)
        inserted = True

if not inserted:
    print("❌ Could not insert arguments!")
    exit(1)

# Write back
with open('coordinator.py', 'w') as f:
    f.write('\n'.join(new_lines))

print("✅ Added window optimizer arguments to coordinator.py")

