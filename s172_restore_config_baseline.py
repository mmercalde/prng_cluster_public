#!/usr/bin/env python3
"""
S172 Pre-Deployment — distributed_config.json restoration.

WHY THIS EXISTS:
    Pre-S172 diagnostics found two unintended modifications to live
    distributed_config.json on Zeus that are outside the approved S172 scope:

    1. rrig6600b (192.168.3.154) was removed from the nodes list during S170
       diagnostic isolation. Per TB ruling 2026-04-30, this MUST be restored
       before S172 validation — full 26-GPU cluster is required to avoid
       false-positive stability under reduced load.

    2. forward_threshold and reverse_threshold bounds were raised from
       0.30 to 0.40 (min) and from 0.30 to 0.50 (default) by a previous
       (rejected) S172 patch attempt on Apr 29. Per TB refined ruling,
       these MUST be restored to the pre-attempt values so Optuna can
       freely explore the full search space.

WHAT THIS SCRIPT DOES (SURGICAL):
    Targets ONLY the specific drift it knows about:
    1. Restores any missing expected node (rrig6600b in particular).
    2. Resets forward_threshold/reverse_threshold (min/max/default) to the
       values committed at HEAD.

    Preserves:
    - Existing per-node entries that ARE on disk (in case of local edits).
    - window_size bounds (owned by s172_config_patch.py).
    - All other config blocks not touched here.

    This means it is safe to run BEFORE or AFTER s172_config_patch.py —
    re-running won't wipe the window_size patch. (However, running it BEFORE
    is the recommended order for clarity.)

    Does NOT:
    - Apply any other patches.
    - Push or commit anything.

EXECUTION ORDER:
    1. python3 s172_restore_config_baseline.py    ← THIS SCRIPT
    2. python3 s172_threshold_patch.py            ← code patch
    3. python3 s172_config_patch.py               ← window_size.min: 2 → 6
    4. (verify, commit, push)

PROPERTIES:
    - Idempotent (no-op if disk already matches HEAD)
    - Creates timestamped backup of pre-restoration state
    - Confirms node count = 4 after write
    - Confirms threshold values match HEAD after write
"""
from __future__ import annotations
import argparse
import datetime as _dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent
CFG_PATH   = REPO_ROOT / "distributed_config.json"
TIMESTAMP  = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def run_git(args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["git"] + args,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def main():
    parser = argparse.ArgumentParser(description="Restore distributed_config.json to git HEAD baseline")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing")
    args = parser.parse_args()

    print("=" * 76)
    print(f"S172 CONFIG RESTORATION  ({'DRY RUN' if args.dry_run else 'APPLY'})")
    print(f"Repo: {REPO_ROOT}")
    print("=" * 76)
    print()

    # 1. Verify we're in a git repo and read HEAD content
    rc, head_sha, err = run_git(["rev-parse", "HEAD"])
    if rc != 0:
        print(f"❌ git rev-parse HEAD failed: {err}")
        return 2
    head_sha = head_sha.strip()
    print(f"Git HEAD: {head_sha}")

    rc, head_content, err = run_git(["show", "HEAD:distributed_config.json"])
    if rc != 0:
        print(f"❌ git show HEAD:distributed_config.json failed: {err}")
        return 2

    # Verify HEAD content is parseable
    try:
        head_cfg = json.loads(head_content)
    except json.JSONDecodeError as e:
        print(f"❌ HEAD distributed_config.json is not valid JSON: {e}")
        return 2

    head_nodes = head_cfg.get("nodes", [])
    print(f"HEAD has {len(head_nodes)} nodes:")
    for n in head_nodes:
        print(f"  - {n.get('hostname')} ({n.get('gpu_count')} × {n.get('gpu_type')})")
    print()

    head_ft = head_cfg["search_bounds"]["forward_threshold"]
    head_rt = head_cfg["search_bounds"]["reverse_threshold"]
    head_ws = head_cfg["search_bounds"]["window_size"]
    print(f"HEAD threshold bounds:")
    print(f"  forward_threshold: min={head_ft.get('min')}, max={head_ft.get('max')}, default={head_ft.get('default')}")
    print(f"  reverse_threshold: min={head_rt.get('min')}, max={head_rt.get('max')}, default={head_rt.get('default')}")
    print(f"  window_size:       min={head_ws.get('min')}, max={head_ws.get('max')}, default={head_ws.get('default')}")
    print()

    # 2. Read current disk content
    if not CFG_PATH.exists():
        print(f"❌ {CFG_PATH} not found")
        return 2

    current_content = CFG_PATH.read_text()
    try:
        current_cfg = json.loads(current_content)
    except json.JSONDecodeError as e:
        print(f"❌ Current distributed_config.json is not valid JSON: {e}")
        return 2

    current_nodes = current_cfg.get("nodes", [])
    print(f"Current disk has {len(current_nodes)} nodes:")
    for n in current_nodes:
        print(f"  - {n.get('hostname')} ({n.get('gpu_count')} × {n.get('gpu_type')})")
    print()

    current_ft = current_cfg["search_bounds"]["forward_threshold"]
    current_rt = current_cfg["search_bounds"]["reverse_threshold"]
    current_ws = current_cfg["search_bounds"]["window_size"]
    print(f"Current disk threshold bounds:")
    print(f"  forward_threshold: min={current_ft.get('min')}, max={current_ft.get('max')}, default={current_ft.get('default')}")
    print(f"  reverse_threshold: min={current_rt.get('min')}, max={current_rt.get('max')}, default={current_rt.get('default')}")
    print(f"  window_size:       min={current_ws.get('min')}, max={current_ws.get('max')}, default={current_ws.get('default')}")
    print()

    # 3. Idempotency check — based on the SPECIFIC drifts this script targets.
    # We only care about:
    #   (a) all 4 expected nodes present
    #   (b) threshold bounds at HEAD baseline values
    # Other modifications (e.g. window_size.min raised by s172_config_patch.py
    # in a subsequent step) are NOT reasons to re-restore.
    expected_hosts = {"localhost", "192.168.3.120", "192.168.3.154", "192.168.3.162"}
    actual_hosts   = {n.get("hostname") for n in current_nodes}

    nodes_ok      = (expected_hosts == actual_hosts)
    ft_ok         = (current_ft.get("min") == head_ft.get("min")
                     and current_ft.get("default") == head_ft.get("default"))
    rt_ok         = (current_rt.get("min") == head_rt.get("min")
                     and current_rt.get("default") == head_rt.get("default"))

    if nodes_ok and ft_ok and rt_ok:
        print("✅ Targeted invariants already satisfied:")
        print(f"     all 4 nodes present, threshold bounds at HEAD baseline")
        print(f"  No restoration needed.")
        return 0

    # 4. Show summary of what will change
    print("─── Differences (current → HEAD) ───────────────────────────────────────")
    current_hostnames = {n.get("hostname") for n in current_nodes}
    head_hostnames    = {n.get("hostname") for n in head_nodes}
    nodes_added   = head_hostnames - current_hostnames
    nodes_removed = current_hostnames - head_hostnames
    if nodes_added:
        print(f"  Nodes to be RESTORED:  {sorted(nodes_added)}")
    if nodes_removed:
        print(f"  Nodes to be REMOVED:   {sorted(nodes_removed)}")

    if current_ft != head_ft:
        print(f"  forward_threshold: {current_ft} → {head_ft}")
    if current_rt != head_rt:
        print(f"  reverse_threshold: {current_rt} → {head_rt}")
    if current_ws != head_ws:
        print(f"  window_size: {current_ws} → {head_ws}")

    # Other diff hints (we don't enumerate every key change but flag if there are
    # other top-level keys differing)
    other_diffs = []
    for k in set(current_cfg.keys()) | set(head_cfg.keys()):
        if k in ("nodes", "search_bounds"):
            continue
        if current_cfg.get(k) != head_cfg.get(k):
            other_diffs.append(k)
    if other_diffs:
        print(f"  Other top-level keys differing: {other_diffs}")
    print()

    if args.dry_run:
        print("DRY RUN — no file written")
        return 0

    # 5. Surgical restoration — modify ONLY the targeted fields in the current
    # file. We do NOT blow away the entire file with HEAD content because that
    # would also revert any changes from s172_config_patch.py if it has already
    # run (window_size.min: 2 → 6).
    bak = CFG_PATH.with_suffix(CFG_PATH.suffix + f".pre_s172_restore_{TIMESTAMP}")
    shutil.copy2(CFG_PATH, bak)
    print(f"📦 Backup: {bak.name}")

    # Re-read current as OrderedDict so we preserve key order and formatting style
    from collections import OrderedDict
    cur = json.loads(CFG_PATH.read_text(), object_pairs_hook=OrderedDict)

    # Restore missing nodes (preserving order from HEAD)
    cur_hostnames = {n.get("hostname") for n in cur.get("nodes", [])}
    head_node_by_host = {n.get("hostname"): n for n in head_nodes}

    # Rebuild nodes list in HEAD order, using HEAD entries for any host not on disk
    new_nodes = []
    for h_node in head_nodes:
        host = h_node.get("hostname")
        if host in cur_hostnames:
            # Keep current disk entry (preserves any intentional local edits)
            for c_node in cur["nodes"]:
                if c_node.get("hostname") == host:
                    new_nodes.append(c_node)
                    break
        else:
            # Restore from HEAD
            new_nodes.append(OrderedDict(h_node))
    cur["nodes"] = new_nodes

    # Restore threshold bounds — surgical
    cur_sb = cur["search_bounds"]
    cur_sb["forward_threshold"]["min"]     = head_ft.get("min", 0.3)
    cur_sb["forward_threshold"]["max"]     = head_ft.get("max", 0.75)
    cur_sb["forward_threshold"]["default"] = head_ft.get("default", 0.3)
    cur_sb["reverse_threshold"]["min"]     = head_rt.get("min", 0.3)
    cur_sb["reverse_threshold"]["max"]     = head_rt.get("max", 0.75)
    cur_sb["reverse_threshold"]["default"] = head_rt.get("default", 0.3)

    # NOTE: We deliberately do NOT touch window_size — the companion
    # s172_config_patch.py owns that field.

    new_text = json.dumps(cur, indent=2) + "\n"
    CFG_PATH.write_text(new_text)
    print(f"✅ Wrote {CFG_PATH.name} (surgical restore)")

    # 6. Verify the result
    print()
    print("─── Post-write verification ────────────────────────────────────────────")
    written = json.loads(CFG_PATH.read_text())
    written_nodes = written.get("nodes", [])
    print(f"  Nodes after restore: {len(written_nodes)}")
    for n in written_nodes:
        print(f"    - {n.get('hostname')} ({n.get('gpu_count')} × {n.get('gpu_type')})")

    failures = []
    if len(written_nodes) != 4:
        failures.append(f"expected 4 nodes, got {len(written_nodes)}")

    expected_hosts = {"localhost", "192.168.3.120", "192.168.3.154", "192.168.3.162"}
    actual_hosts   = {n.get("hostname") for n in written_nodes}
    if expected_hosts != actual_hosts:
        failures.append(f"hostnames mismatch: missing {expected_hosts - actual_hosts}, extra {actual_hosts - expected_hosts}")

    written_ft = written["search_bounds"]["forward_threshold"]
    written_rt = written["search_bounds"]["reverse_threshold"]
    if written_ft.get("min") != 0.3 or written_ft.get("default") != 0.3:
        failures.append(f"forward_threshold not restored: {written_ft}")
    if written_rt.get("min") != 0.3 or written_rt.get("default") != 0.3:
        failures.append(f"reverse_threshold not restored: {written_rt}")

    if failures:
        print()
        print("❌ POST-WRITE CHECK FAILED:")
        for f in failures:
            print(f"  {f}")
        return 1

    print()
    print("✅ Restoration complete — all 4 nodes present, thresholds at HEAD baseline")
    print()
    print("Next steps:")
    print("  1. python3 s172_threshold_patch.py    # FIX 1 + FIX 2 code patches")
    print("  2. python3 s172_config_patch.py       # window_size.min: 2 → 6")
    return 0


if __name__ == "__main__":
    sys.exit(main())
