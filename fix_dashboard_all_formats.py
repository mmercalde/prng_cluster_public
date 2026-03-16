#!/usr/bin/env python3
"""
fix_dashboard_all_formats.py
Fix ALL unguarded format strings in web_dashboard.py template.
"""
import shutil
from pathlib import Path

TARGET = Path('/home/michael/distributed_prng_analysis/web_dashboard.py')
content = TARGET.read_text(encoding='utf-8')
shutil.copy2(TARGET, str(TARGET) + '.allformats_backup')

original = content

# Fix all unguarded format strings
fixes = [
    # total_sps - appears 3 times
    (
        '"{:,.0f}".format(total_sps)',
        '"{:,.0f}".format(total_sps|default(0))'
    ),
    # node.current_seeds_per_sec
    (
        '"{:,.0f}".format(node.current_seeds_per_sec)',
        '"{:,.0f}".format(node.current_seeds_per_sec|default(0))'
    ),
    # node division - trickier, use safe division
    (
        '"{:,.0f}".format(node.current_seeds_per_sec / node.total_gpus)',
        '"{:,.0f}".format((node.current_seeds_per_sec|default(0)) / ((node.total_gpus|default(1)) or 1))'
    ),
]

count = 0
for old, new in fixes:
    occurrences = content.count(old)
    if occurrences > 0:
        content = content.replace(old, new)
        print(f'✅ Fixed {occurrences}x: {old[:60]}')
        count += occurrences
    else:
        print(f'⚠️  Not found: {old[:60]}')

TARGET.write_text(content, encoding='utf-8')
print(f'\nTotal fixes: {count}')
print('Done — restart dashboard to apply')
