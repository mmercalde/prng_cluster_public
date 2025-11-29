#!/usr/bin/env python3
"""Add hybrid mode arguments to timestamp_search.py"""

with open('timestamp_search.py', 'r') as f:
    content = f.read()

# Find the argument parser section and add hybrid arguments after --skip-max
old_args = """    parser.add_argument('--skip-max', type=int, default=100)
    parser.add_argument('--prngs', nargs='+',"""

new_args = """    parser.add_argument('--skip-max', type=int, default=100)
    
    # Hybrid variable skip mode arguments
    parser.add_argument('--hybrid', action='store_true',
                       help='Enable hybrid variable skip detection (multi-strategy)')
    parser.add_argument('--phase1-threshold', type=float, default=0.20,
                       help='Phase 1 threshold for initial filtering (default: 0.20)')
    parser.add_argument('--phase2-threshold', type=float, default=0.75,
                       help='Phase 2 threshold for variable skip analysis (default: 0.75)')
    
    parser.add_argument('--prngs', nargs='+',"""

if old_args in content:
    content = content.replace(old_args, new_args)
    print("✅ Added hybrid arguments to parser")
else:
    print("⚠️ Could not find argument insertion point")

# Update the submit_to_coordinator function to pass hybrid args
old_cmd_build = """    cmd = [
        'python3', 'coordinator.py',
        args.dataset,
        '--method', 'residue_sieve',
        '--prng-type', prng,
        '--window-size', str(args.window),
        '--threshold', str(args.threshold),
        '--skip-min', str(args.skip_min),
        '--skip-max', str(args.skip_max),
        '--seeds', str(total_seeds),
        '--seed-start', str(seed_start),
        '--session-filter', 'both',
        '--resume-policy', args.resume_policy
    ]"""

new_cmd_build = """    cmd = [
        'python3', 'coordinator.py',
        args.dataset,
        '--method', 'residue_sieve',
        '--prng-type', prng,
        '--window-size', str(args.window),
        '--threshold', str(args.threshold),
        '--skip-min', str(args.skip_min),
        '--skip-max', str(args.skip_max),
        '--seeds', str(total_seeds),
        '--seed-start', str(seed_start),
        '--session-filter', 'both',
        '--resume-policy', args.resume_policy
    ]
    
    # Add hybrid mode arguments if enabled
    if args.hybrid:
        cmd.extend(['--hybrid'])
        cmd.extend(['--phase1-threshold', str(args.phase1_threshold)])
        cmd.extend(['--phase2-threshold', str(args.phase2_threshold)])"""

if old_cmd_build in content:
    content = content.replace(old_cmd_build, new_cmd_build)
    print("✅ Updated command builder for hybrid args")
else:
    print("⚠️ Could not find command builder")

# Update the configuration display to show hybrid status
old_display = """    print(f"  ✓ PRNGs: {', '.join(args.prngs)}")
    print(f"  ✓ Threshold: {args.threshold:.1%} ({int(args.threshold*args.window)}/{args.window})")"""

new_display = """    print(f"  ✓ PRNGs: {', '.join(args.prngs)}")
    print(f"  ✓ Threshold: {args.threshold:.1%} ({int(args.threshold*args.window)}/{args.window})")
    if args.hybrid:
        print(f"  ✓ Hybrid mode: ENABLED")
        print(f"    - Phase 1 threshold: {args.phase1_threshold:.1%}")
        print(f"    - Phase 2 threshold: {args.phase2_threshold:.1%}")
    else:
        print(f"  ✓ Skip mode: Fixed (standard)")"""

if old_display in content:
    content = content.replace(old_display, new_display)
    print("✅ Updated display for hybrid status")
else:
    print("⚠️ Could not find display section")

# Write back
with open('timestamp_search.py', 'w') as f:
    f.write(content)

print("✅ Hybrid arguments added to timestamp_search.py")

# Test syntax
try:
    compile(content, 'timestamp_search.py', 'exec')
    print("✅ File syntax is valid!")
except SyntaxError as e:
    print(f"❌ Syntax error: {e}")
