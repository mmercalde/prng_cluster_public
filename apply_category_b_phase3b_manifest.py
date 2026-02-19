#!/usr/bin/env python3
"""
Category B Phase 3B — Patch agent_manifests/reinforcement.json
==============================================================

Adds Category B parameter bounds to the WATCHER manifest so that
the agent can safely emit normalize_features, use_leaky_relu, and
dropout override in retry payloads.

Fix 3B-1: Both 'dropout_override' (preferred for retry semantics) and
'dropout' (alias, matches CLI --dropout) are declared. Phase 2's
_s88_run_compare_models reads both:
  args_dict.get("dropout_override") or args_dict.get("dropout")

Verified against live code:
  github.com/mmercalde/prng_cluster_public/main/agent_manifests/reinforcement.json
  Commit: 3c3f9ae

Author: Team Alpha (S92)
Date: 2026-02-15
"""

import sys
import json
import shutil
from pathlib import Path

TARGET = Path("agent_manifests/reinforcement.json")
BACKUP = TARGET.with_suffix(".pre_category_b_phase3")


def verify_preconditions():
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found. Run from ~/distributed_prng_analysis/")
        return False

    data = json.loads(TARGET.read_text())

    if 'normalize_features' in data.get('parameter_bounds', {}):
        print("ERROR: normalize_features already in parameter_bounds. Already patched?")
        return False

    if 'parameter_bounds' not in data:
        print("ERROR: No parameter_bounds key in manifest")
        return False

    print("All preconditions PASSED")
    return True


def apply_patch():
    content = TARGET.read_text()
    data = json.loads(content)

    pb = data['parameter_bounds']

    pb['normalize_features'] = {
        "type": "bool",
        "default": True,
        "description": "Category B: StandardScaler input normalization for NN (Option A: default ON)"
    }

    pb['use_leaky_relu'] = {
        "type": "bool",
        "default": True,
        "description": "Category B: LeakyReLU(0.01) activation for NN (default ON)"
    }

    # Fix 3B-1: Both canonical name and alias for end-to-end consistency
    pb['dropout_override'] = {
        "type": "float",
        "min": 0.0,
        "max": 0.9,
        "default": None,
        "description": "Category B: NN dropout override (preferred key for WATCHER retry payloads)"
    }

    pb['dropout'] = {
        "type": "float",
        "min": 0.0,
        "max": 0.9,
        "default": None,
        "description": "Category B: Alias for dropout_override (matches CLI --dropout). Phase 2 reads both."
    }

    # Bump version
    data['version'] = '1.6.0'

    # Add to version_history
    vh = data.get('version_history', [])
    vh.append({
        "version": "1.6.0",
        "date": "2026-02-15",
        "changes": [
            "Category B: Added normalize_features, use_leaky_relu parameter bounds (default ON)",
            "Category B: Added dropout_override + dropout alias for NN training override",
            "Fix 3B-1: Both dropout_override (preferred) and dropout (alias) declared for consistency"
        ]
    })
    data['version_history'] = vh

    TARGET.write_text(json.dumps(data, indent=2) + "\n")
    print("[1/1] Added Category B parameter bounds (with dropout alias) + bumped to v1.6.0")
    return True


def main():
    print("=" * 60)
    print("Category B Phase 3B: agent_manifests/reinforcement.json")
    print("  (v2 — with Fix 3B-1: dropout_override + dropout alias)")
    print("=" * 60)

    if not verify_preconditions():
        sys.exit(1)

    shutil.copy2(TARGET, BACKUP)
    print(f"Backup: {BACKUP}")

    if not apply_patch():
        shutil.copy2(BACKUP, TARGET)
        print("REVERTED to backup due to patch failure")
        sys.exit(1)

    try:
        json.loads(TARGET.read_text())
        print("JSON validation PASSED")
    except json.JSONDecodeError as e:
        print(f"JSON ERROR: {e}")
        shutil.copy2(BACKUP, TARGET)
        print("REVERTED to backup due to JSON error")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Phase 3B COMPLETE")
    print("=" * 60)
    print(f"  File: {TARGET}")
    print(f"  Version: 1.5.0 -> 1.6.0")
    print(f"  Backup: {BACKUP}")
    print(f"  Added parameter_bounds:")
    print(f"    normalize_features: bool, default=True")
    print(f"    use_leaky_relu: bool, default=True")
    print(f"    dropout_override: float [0.0, 0.9], default=None (preferred)")
    print(f"    dropout: float [0.0, 0.9], default=None (alias, matches CLI)")


if __name__ == "__main__":
    main()
