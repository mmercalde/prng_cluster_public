#!/usr/bin/env python3
"""
apply_s147_gpu_display_fix.py
Replaces dead GPU clock (0 MHz) display in web_dashboard.py with
seeds/s per GPU — no ROCm-smi polling, no overhead, more useful info.

Two locations patched:
1. Overview worker table — AVG CLOCK column → AVG SEEDS/GPU
2. Workers detail page — Clock stat per GPU card → Seeds/s
"""
import os, shutil, sys, argparse

TARGET = os.path.expanduser("~/distributed_prng_analysis/web_dashboard.py")

# ── Patch 1: Overview table header — rename AVG CLOCK → AVG SEEDS/GPU ──
OLD_HEADER = """                <th>Worker</th>
                <th>Avg Clock</th>
                <th>Status</th>"""

NEW_HEADER = """                <th>Worker</th>
                <th>Avg Seeds/GPU</th>
                <th>Status</th>"""

# ── Patch 2: Overview table cell — replace clock lookup with seeds/s per GPU ──
OLD_CLOCK_CELL = """                <td>
                    {% set gpu_data = gpu_stats.get(hostname, []) %}
                    {% set avg_clock = 0 %}
                    {% if gpu_data %}
                        {% set clock_sum = namespace(val=0) %}
                        {% for g in gpu_data %}
                            {% set clock_sum.val = clock_sum.val + g.get('clock', 0) %}
                        {% endfor %}
                        {% set avg_clock = (clock_sum.val / gpu_data|length)|int %}
                    {% endif %}
                    <span style="color: {% if avg_clock > 1000 %}var(--accent-green){% elif avg_clock > 100 %}var(--accent-orange){% else %}var(--text-muted){% endif %}; font-weight: 600;">
                        {{ avg_clock }} MHz
                    </span>
                </td>"""

NEW_CLOCK_CELL = """                <td>
                    {% set per_gpu = (node.current_seeds_per_sec / node.total_gpus)|int if node.total_gpus > 0 else 0 %}
                    <span style="color: {% if per_gpu > 1000 %}var(--accent-green){% elif per_gpu > 100 %}var(--accent-orange){% else %}var(--text-muted){% endif %}; font-weight: 600;">
                        {{ "{:,}".format(per_gpu) }} s/s
                    </span>
                </td>"""

# ── Patch 3: Overview mini-chart bars — use throughput % instead of clock % ──
OLD_MINI = """                    <div class="mini-chart">
                        {% set gpu_data = gpu_stats.get(hostname, []) %}
                        {% for i in range(node.total_gpus) %}
                            {% if gpu_data and i < gpu_data|length %}
                                {% set clock_pct = (gpu_data[i].get('clock', 0) / 2000 * 100)|int %}
                                <div class="mini-bar" style="height: {{ clock_pct if clock_pct > 5 else 5 }}%; background: {% if clock_pct > 50 %}var(--accent-green){% elif clock_pct > 10 %}var(--accent-orange){% else %}var(--text-muted){% endif %};"></div>
                            {% else %}
                                <div class="mini-bar" style="height: 5%;"></div>
                            {% endif %}
                        {% endfor %}
                    </div>"""

NEW_MINI = """                    <div class="mini-chart">
                        {% set active = node.current_seeds_per_sec > 0 %}
                        {% for i in range(node.total_gpus) %}
                            <div class="mini-bar" style="height: {% if active %}85{% else %}5{% endif %}%; background: {% if active %}var(--accent-green){% else %}var(--text-muted){% endif %};"></div>
                        {% endfor %}
                    </div>"""

# ── Patch 4: Workers detail GPU card — replace clock stat with seeds/s ──
OLD_GPU_CARD_CLOCK = """                    <div>
                        {% set gpu_list = gpu_stats.get(hostname, []) %}
                        {% set gpu_info = gpu_list[i] if i < gpu_list|length else {} %}
                        {% set gpu_clock = gpu_info.get('clock', 0) if gpu_info else 0 %}
                        <div class="gpu-stat-value" style="color: {% if gpu_clock > 1000 %}var(--accent-green){% elif gpu_clock > 0 %}var(--accent-orange){% else %}var(--text-secondary){% endif %};">{{ gpu_clock }} MHz</div>
                        <div class="gpu-stat-label">Clock</div>
                    </div>"""

NEW_GPU_CARD_CLOCK = """                    <div>
                        {% set per_gpu = (node.current_seeds_per_sec / node.total_gpus)|int if node.total_gpus > 0 else 0 %}
                        <div class="gpu-stat-value" style="color: {% if per_gpu > 500 %}var(--accent-green){% elif per_gpu > 0 %}var(--accent-orange){% else %}var(--text-secondary){% endif %};">{{ "{:,}".format(per_gpu) }}</div>
                        <div class="gpu-stat-label">Seeds/s</div>
                    </div>"""


PATCHES = [
    ("Overview table header", OLD_HEADER, NEW_HEADER),
    ("Overview clock cell", OLD_CLOCK_CELL, NEW_CLOCK_CELL),
    ("Overview mini-chart bars", OLD_MINI, NEW_MINI),
    ("Workers GPU card clock", OLD_GPU_CARD_CLOCK, NEW_GPU_CARD_CLOCK),
]


def apply_patches(dry_run):
    if not os.path.exists(TARGET):
        print(f"ERROR: {TARGET} not found")
        sys.exit(1)

    with open(TARGET, 'r') as f:
        content = f.read()

    results = []
    new_content = content

    for label, old, new in PATCHES:
        count = new_content.count(old)
        if count == 0:
            print(f"  SKIP {label}: anchor not found")
            results.append(False)
        elif count > 1:
            print(f"  WARN {label}: {count} matches — ambiguous, skipping")
            results.append(False)
        else:
            before = len(new_content.splitlines())
            new_content = new_content.replace(old, new, 1)
            after = len(new_content.splitlines())
            if dry_run:
                print(f"  DRY  {label}: {before} → {after} lines")
            else:
                print(f"  OK   {label}: {before} → {after} lines")
            results.append(True)

    ok = sum(results)
    skip = len(results) - ok
    print(f"\n{'DRY RUN ' if dry_run else ''}Summary: {ok} applied, {skip} skipped")

    if dry_run or ok == 0:
        return

    bak = TARGET + ".bak_s147_gpu"
    if not os.path.exists(bak):
        shutil.copy(TARGET, bak)
        print(f"BAK: {bak}")

    with open(TARGET, 'w') as f:
        f.write(new_content)

    print("\nRestart dashboard to pick up changes:")
    print("  pkill -f web_dashboard")
    print("  cd ~/distributed_prng_analysis && source ~/venvs/torch/bin/activate")
    print("  nohup python3 web_dashboard.py > logs/dashboard.log 2>&1 &")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    apply_patches(args.dry_run)
