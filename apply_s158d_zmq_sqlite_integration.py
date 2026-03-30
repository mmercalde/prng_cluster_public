#!/usr/bin/env python3
"""
apply_s158d_zmq_sqlite_integration.py

Adds --use-zmq-sqlite flag support to:
1. window_optimizer_integration_final.py  — additive gate (mirrors PWC gate)
2. window_optimizer.py                    — flag + coordinator attribute
3. agent_manifests/window_optimizer.json  — args_map + default_params

Zero changes to any other file. Fully backwards compatible.
PWC path untouched.

Deploy:
  scp ~/Downloads/apply_s158d_zmq_sqlite_integration.py rzeus:~/distributed_prng_analysis/
  scp ~/Downloads/zmq_sqlite_coordinator.py rzeus:~/distributed_prng_analysis/
  scp ~/Downloads/zmq_sqlite_worker.py rzeus:~/distributed_prng_analysis/
  ssh rzeus 'cd ~/distributed_prng_analysis && python3 apply_s158d_zmq_sqlite_integration.py'
"""

import ast
import json
import shutil
from pathlib import Path

BASE = Path("/home/michael/distributed_prng_analysis")


def backup(path: Path):
    bak = path.with_suffix(path.suffix + ".bak_s158d")
    shutil.copy2(path, bak)
    print(f"  Backup: {bak}")


def ast_check(path: Path) -> bool:
    try:
        ast.parse(path.read_text())
        return True
    except SyntaxError as e:
        print(f"  SYNTAX ERROR in {path}: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 1. window_optimizer_integration_final.py
# ─────────────────────────────────────────────────────────────────────────────
def patch_integration():
    path = BASE / "window_optimizer_integration_final.py"
    content = path.read_text()

    if "use_zmq_sqlite" in content:
        print("  [integration] ZMQ-SQLite gate already present — skipping")
        return True

    # Add import at top (after existing persistent_worker import)
    OLD_IMPORT = """try:
    from persistent_worker_coordinator import run_trial_persistent
except ImportError:
    run_trial_persistent = None  # only needed when --use-persistent-workers is set"""

    NEW_IMPORT = """try:
    from persistent_worker_coordinator import run_trial_persistent
except ImportError:
    run_trial_persistent = None  # only needed when --use-persistent-workers is set

try:
    from zmq_sqlite_coordinator import run_trial_zmq_sqlite
except ImportError:
    run_trial_zmq_sqlite = None  # only needed when --use-zmq-sqlite is set"""

    if OLD_IMPORT not in content:
        print("  [integration] WARNING: import anchor not found — skipping import patch")
    else:
        content = content.replace(OLD_IMPORT, NEW_IMPORT, 1)

    # Add ZMQ-SQLite gate after PWC gate (same pattern)
    OLD_GATE = """    # ========================================================================
    # END PERSISTENT WORKER PATH — original path continues unchanged below
    # ========================================================================"""

    NEW_GATE = """    # ========================================================================
    # END PERSISTENT WORKER PATH — original path continues unchanged below
    # ========================================================================

    # ========================================================================
    # [S158D] ZMQ-SQLITE PATH — activated by use_zmq_sqlite=True
    # Zero changes to original path. Purely additive gate.
    # ========================================================================
    _use_zmq = getattr(coordinator, 'use_zmq_sqlite', False)
    if _use_zmq:
        if run_trial_zmq_sqlite is None:
            raise ImportError(
                "zmq_sqlite_coordinator.py not found — cannot use --use-zmq-sqlite"
            )
        _zmq_result = run_trial_zmq_sqlite(
            coordinator_cfg   = getattr(coordinator, 'config_file', 'distributed_config.json'),
            config            = config,
            trial_number      = trial_number,
            prng_base         = prng_base,
            residues          = _get_residues_for_config(config, dataset_path),
            total_seeds       = seed_count,
            forward_threshold = forward_threshold,
            reverse_threshold = reverse_threshold,
            test_both_modes   = test_both_modes,
            dataset_path      = dataset_path,
            worker_pool_size  = getattr(coordinator, 'worker_pool_size', 8),
            seed_cap_nvidia   = getattr(coordinator, 'seed_cap_nvidia', 5_000_000),
            seed_cap_amd      = getattr(coordinator, 'seed_cap_amd',    2_000_000),
        )
        if _zmq_result.get("pruned"):
            return TestResult(
                config              = config,
                forward_count       = 0,
                reverse_count       = 0,
                bidirectional_count = 0,
                iteration           = trial_number,
            )
        return _build_test_result_from_pw(
            _zmq_result, accumulator, config, prng_base, trial_number, optuna_trial
        )
    # ========================================================================
    # END ZMQ-SQLITE PATH
    # ========================================================================"""

    if OLD_GATE not in content:
        print("  [integration] ERROR: gate anchor not found")
        return False

    content = content.replace(OLD_GATE, NEW_GATE, 1)

    backup(path)
    path.write_text(content)

    if not ast_check(path):
        shutil.copy2(path.with_suffix(".py.bak_s158d"), path)
        return False

    print("  [integration] ✅ ZMQ-SQLite gate added")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 2. window_optimizer.py
# ─────────────────────────────────────────────────────────────────────────────
def patch_window_optimizer():
    path = BASE / "window_optimizer.py"
    content = path.read_text()

    if "use_zmq_sqlite" in content:
        print("  [window_optimizer] ZMQ-SQLite flag already present — skipping")
        return True

    # Add parameter to optimize_window() signature
    OLD_SIG = "    use_persistent_workers: bool = False,   # S134"
    NEW_SIG = """    use_persistent_workers: bool = False,   # S134
    use_zmq_sqlite: bool = False,            # S158D"""

    if OLD_SIG not in content:
        print("  [window_optimizer] WARNING: signature anchor not found")
    else:
        content = content.replace(OLD_SIG, NEW_SIG, 1)

    # Set coordinator attribute
    OLD_ATTR = "    coordinator.use_persistent_workers = use_persistent_workers"
    NEW_ATTR = """    coordinator.use_persistent_workers = use_persistent_workers
    coordinator.use_zmq_sqlite = use_zmq_sqlite
    if use_zmq_sqlite:
        print(f"   [S158D] ZMQ-SQLite coordinator ENABLED")"""

    if OLD_ATTR not in content:
        print("  [window_optimizer] WARNING: attribute anchor not found")
    else:
        content = content.replace(OLD_ATTR, NEW_ATTR, 1)

    # Add to argparse
    OLD_ARGPARSE = "    parser.add_argument('--use-persistent-workers', action='store_true', default=False,"
    NEW_ARGPARSE = """    parser.add_argument('--use-persistent-workers', action='store_true', default=False,
                        help='Use persistent SSH worker coordinator (S134)')
    parser.add_argument('--use-zmq-sqlite', action='store_true', default=False,"""

    # Find and patch argparse block
    if OLD_ARGPARSE in content:
        # Find the end of the use-persistent-workers argparse block
        idx = content.find(OLD_ARGPARSE)
        end = content.find(")\n", idx) + 2
        old_block = content[idx:end]
        new_block = old_block + """    parser.add_argument('--use-zmq-sqlite', action='store_true', default=False,
                        help='Use ZMQ+SQLite coordinator (S158D — no persistent SSH pipes)')
"""
        content = content[:idx] + new_block + content[end:]

    # Add to optimize_window call
    OLD_CALL = "            use_persistent_workers=getattr(args, 'use_persistent_workers', False),  # S134"
    NEW_CALL = """            use_persistent_workers=getattr(args, 'use_persistent_workers', False),  # S134
            use_zmq_sqlite=getattr(args, 'use_zmq_sqlite', False),                      # S158D"""

    if OLD_CALL in content:
        content = content.replace(OLD_CALL, NEW_CALL, 1)

    backup(path)
    path.write_text(content)

    if not ast_check(path):
        shutil.copy2(path.with_suffix(".py.bak_s158d"), path)
        return False

    print("  [window_optimizer] ✅ --use-zmq-sqlite flag added")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 3. agent_manifests/window_optimizer.json
# ─────────────────────────────────────────────────────────────────────────────
def patch_manifest():
    path = BASE / "agent_manifests" / "window_optimizer.json"
    with open(path) as f:
        manifest = json.load(f)

    changed = False

    # Add to actions args_map
    for action in manifest.get("actions", []):
        if action.get("script") == "window_optimizer.py":
            if "use-zmq-sqlite" not in action.get("args_map", {}):
                action["args_map"]["use-zmq-sqlite"] = "use_zmq_sqlite"
                changed = True

    # Add to default_params (disabled by default)
    if "use_zmq_sqlite" not in manifest.get("default_params", {}):
        manifest["default_params"]["use_zmq_sqlite"] = False
        changed = True

    # Add param doc
    if "use_zmq_sqlite" not in manifest.get("param_docs", {}):
        manifest.setdefault("param_docs", {})["use_zmq_sqlite"] = {
            "type": "bool",
            "default": False,
            "description": "S158D: Use ZMQ+SQLite coordinator instead of PWC. "
                          "Eliminates persistent SSH pipes. "
                          "Workers launched once per run, pull jobs via ZMQ TCP."
        }
        changed = True

    if changed:
        backup(path)
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
        print("  [manifest] ✅ use_zmq_sqlite added to args_map + default_params")
    else:
        print("  [manifest] Already up to date — skipping")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("Applying S158D ZMQ-SQLite integration patches...")
    print()

    results = []

    print("1. window_optimizer_integration_final.py")
    results.append(patch_integration())

    print("2. window_optimizer.py")
    results.append(patch_window_optimizer())

    print("3. agent_manifests/window_optimizer.json")
    results.append(patch_manifest())

    print()
    if all(results):
        print("✅ All patches applied successfully")
        print()
        print("Next steps:")
        print("  pip install pyzmq --break-system-packages  # on Zeus")
        print("  # On each rig:")
        print("  ssh rrig6600  'pip install pyzmq --break-system-packages'")
        print("  ssh rrig6600b 'pip install pyzmq --break-system-packages'")
        print("  ssh rrig6600c 'pip install pyzmq --break-system-packages'")
        print()
        print("  git add -f zmq_sqlite_coordinator.py zmq_sqlite_worker.py \\")
        print("    apply_s158d_zmq_sqlite_integration.py \\")
        print("    window_optimizer_integration_final.py window_optimizer.py \\")
        print("    agent_manifests/window_optimizer.json")
        print("  git commit -m 'feat(s158d): ZMQ+SQLite coordinator — no persistent SSH pipes'")
        print("  git push origin main && git push public main")
        print()
        print("  # Test with:")
        print("  python3 window_optimizer.py ... --use-zmq-sqlite")
    else:
        print("❌ Some patches failed — check output above")


if __name__ == "__main__":
    main()
