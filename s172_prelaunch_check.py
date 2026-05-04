#!/usr/bin/env python3
"""S172 pre-launch verification.

Run from: ~/distributed_prng_analysis on Zeus

Validates that:
  1. Git state is at the expected S172 commit
  2. distributed_config.json has all S172 invariants
  3. FIX 2 markers are present in window_optimizer_integration_final.py
  4. SearchBounds dataclass defaults are at safe floors

Exits 0 if all clear, 1 otherwise.
"""
from __future__ import annotations
import ast, json, subprocess, sys
from pathlib import Path

REPO = Path.cwd()


def check(label, cond, detail=""):
    s = "PASS" if cond else "FAIL"
    print(f"  [{s}] {label}" + (f" — {detail}" if detail else ""))
    return cond


def main():
    all_ok = True

    # 1. Git state
    print("=== 1. Git state ===")
    head = subprocess.run(["git", "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    msg = subprocess.run(["git", "log", "-1", "--pretty=%s"],
                         capture_output=True, text=True).stdout.strip()
    print(f"  HEAD: {head[:12]}")
    print(f"  msg:  {msg}")
    all_ok &= check("HEAD is S172 commit (3fdf434)", head.startswith("3fdf434"))

    # 2. Cluster config
    print()
    print("=== 2. distributed_config.json invariants ===")
    cfg = json.loads((REPO / "distributed_config.json").read_text())
    nodes = cfg["nodes"]
    total_gpus = sum(n["gpu_count"] for n in nodes)
    expected_hosts = {"localhost", "192.168.3.120", "192.168.3.154", "192.168.3.162"}
    actual_hosts = {n["hostname"] for n in nodes}

    print(f"  nodes ({len(nodes)}):")
    for n in nodes:
        print(f"    - {n['hostname']:20s} {n['gpu_count']} x {n['gpu_type']}")
    print(f"  total GPUs: {total_gpus}")

    all_ok &= check("4 expected nodes", expected_hosts == actual_hosts,
                    f"missing {expected_hosts - actual_hosts}" if expected_hosts != actual_hosts else "")
    all_ok &= check("26 GPUs total", total_gpus == 26, f"got {total_gpus}")

    sb = cfg["search_bounds"]
    ws, ft, rt = sb["window_size"], sb["forward_threshold"], sb["reverse_threshold"]
    print(f"  window_size:       min={ws['min']}, max={ws['max']}, default={ws['default']}")
    print(f"  forward_threshold: min={ft['min']}, max={ft['max']}, default={ft['default']}")
    print(f"  reverse_threshold: min={rt['min']}, max={rt['max']}, default={rt['default']}")
    all_ok &= check("window_size.min >= 6", ws["min"] >= 6)
    all_ok &= check("FT bounds preserved [0.30, 0.75] / 0.30",
                    ft["min"] == 0.3 and ft["max"] == 0.75 and ft["default"] == 0.3)
    all_ok &= check("RT bounds preserved [0.30, 0.75] / 0.30",
                    rt["min"] == 0.3 and rt["max"] == 0.75 and rt["default"] == 0.3)

    # 3. FIX 2 markers
    print()
    print("=== 3. FIX 2 — Optuna threshold drop fix ===")
    int_src = (REPO / "window_optimizer_integration_final.py").read_text()
    fix2_ft = "ft = getattr(config, 'forward_threshold'" in int_src
    fix2_rt = "rt = getattr(config, 'reverse_threshold'" in int_src
    all_ok &= check("forward marker in test_config", fix2_ft)
    all_ok &= check("reverse marker in test_config", fix2_rt)

    # 4. Dataclass defaults
    print()
    print("=== 4. SearchBounds dataclass defaults (defensive) ===")
    src = (REPO / "window_optimizer.py").read_text()
    tree = ast.parse(src)
    defs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "SearchBounds":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    if isinstance(item.value, ast.Constant):
                        defs[item.target.id] = item.value.value
    for k in ["min_forward_threshold", "min_reverse_threshold",
              "default_forward_threshold", "default_reverse_threshold"]:
        print(f"  {k} = {defs.get(k)}")
    all_ok &= check("min_*_threshold >= 0.40",
                    defs.get("min_forward_threshold", 0) >= 0.4
                    and defs.get("min_reverse_threshold", 0) >= 0.4)
    all_ok &= check("default_*_threshold >= 0.50",
                    defs.get("default_forward_threshold", 0) >= 0.5
                    and defs.get("default_reverse_threshold", 0) >= 0.5)

    print()
    print("=" * 60)
    if all_ok:
        print("ALL S172 INVARIANTS SATISFIED — cluster ready for launch")
        return 0
    else:
        print("VERIFICATION FAILED — investigate before launch")
        return 1


if __name__ == "__main__":
    sys.exit(main())
