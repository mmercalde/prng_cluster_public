#!/usr/bin/env python3
"""
fix_dashboard_state_none.py
============================
Fix TypeError: unsupported format string passed to Undefined.__format__
when /tmp/cluster_progress.json doesn't exist (e.g. after reboot).

The dashboard passes state=None to the template when no progress file exists.
Jinja2's "{:,}".format(state.seeds_completed) crashes when state is None.

Fix: add |default(0) to all format calls on state fields.
"""

import sys
import shutil
from pathlib import Path

DRY_RUN = '--dry-run' in sys.argv
TARGET = Path('/home/michael/distributed_prng_analysis/web_dashboard.py')

def read(p): return Path(p).read_text(encoding='utf-8')
def write(p, c):
    if DRY_RUN:
        print(f"  [DRY-RUN] would write {p.name}")
        return
    Path(p).write_text(c, encoding='utf-8')

def backup(path):
    bak = Path(str(path) + '.dashboard_none_backup')
    if DRY_RUN:
        print(f"  [DRY-RUN] would backup → {bak.name}")
        return
    shutil.copy2(path, bak)
    print(f"  ✅ Backup: {bak.name}")

print("fix_dashboard_state_none.py")
print("=" * 55)
if DRY_RUN:
    print("MODE: DRY RUN")

content = read(TARGET)
original_lines = len(content.splitlines())
backup(TARGET)

replacements = [
    # Line 663 - seeds_completed
    (
        '{{ "{:,}".format(state.seeds_completed) }}',
        '{{ "{:,}".format(state.seeds_completed|default(0)) }}'
    ),
    # Line 761 - forward_survivors
    (
        '{{ "{:,}".format(state.trial_stats.forward_survivors) }}',
        '{{ "{:,}".format(state.trial_stats.forward_survivors|default(0)) }}'
    ),
    # Line 765 - reverse_survivors
    (
        '{{ "{:,}".format(state.trial_stats.reverse_survivors) }}',
        '{{ "{:,}".format(state.trial_stats.reverse_survivors|default(0)) }}'
    ),
    # Line 769 - bidirectional
    (
        '{{ "{:,}".format(state.trial_stats.bidirectional) }}',
        '{{ "{:,}".format(state.trial_stats.bidirectional|default(0)) }}'
    ),
    # Line 773 - best_bidirectional
    (
        '{{ "{:,}".format(state.trial_stats.best_bidirectional) }}',
        '{{ "{:,}".format(state.trial_stats.best_bidirectional|default(0)) }}'
    ),
    # Line 962 - seeds_completed in table
    (
        '<td>{{ "{:,}".format(state.seeds_completed) }}</td>',
        '<td>{{ "{:,}".format(state.seeds_completed|default(0)) }}</td>'
    ),
]

all_ok = True
for old, new in replacements:
    if old in content:
        content = content.replace(old, new, 1)
        print(f"  ✅ Fixed: {old[:60]}...")
    else:
        print(f"  ⚠️  Not found: {old[:60]}...")

if not DRY_RUN:
    write(TARGET, content)
    new_lines = len(content.splitlines())
    print(f"\nLines: {original_lines} → {new_lines}")

print("\n" + "=" * 55)
print("✅ DONE")
print()
print("Restart dashboard:")
print("  pkill -f web_dashboard.py 2>/dev/null; sleep 1 && \\")
print("  cd ~/distributed_prng_analysis && \\")
print("  source ~/venvs/torch/bin/activate && \\")
print("  nohup python3 web_dashboard.py > logs/dashboard.log 2>&1 &")
