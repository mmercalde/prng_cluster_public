#!/usr/bin/env python3
"""
S140b Warm-Start Fix
====================
Fixes two issues:
1. Remove warm_start_* from window_optimizer.json default_params
   (they are transient WATCHER-internal params, not CLI args)
2. Add warm_start_* to WATCHER's internal strip list before CLI build

Usage (from ~/distributed_prng_analysis/):
  python3 apply_s140b_warmstart_fix.py [--dry-run]
"""

import sys, os, json, shutil, subprocess
from pathlib import Path

DRY_RUN = '--dry-run' in sys.argv

def backup(path):
    bak = path + '.bak_s140b_fix'
    if not DRY_RUN:
        shutil.copy2(path, bak)
    print(f"  BAK  {bak}")

def write(path, content):
    if not DRY_RUN:
        with open(path, 'w') as f:
            f.write(content)
    print(f"  {'DRY' if DRY_RUN else 'WRT'} {path}")

def check(condition, msg):
    if not condition:
        print(f"  ABORT: {msg}")
        sys.exit(1)


def fix_manifest():
    print("\n[1/2] agent_manifests/window_optimizer.json — remove warm_start_* from default_params")
    path = 'agent_manifests/window_optimizer.json'
    backup(path)
    with open(path) as f:
        manifest = json.load(f)

    warm_keys = [k for k in manifest.get('default_params', {}) if k.startswith('warm_start_')]
    if not warm_keys:
        print("  SKIP — no warm_start_* in default_params")
        return

    for k in warm_keys:
        del manifest['default_params'][k]
        print(f"  REMOVE default_params.{k}")
    if 'parameter_bounds' in manifest:
        for k in warm_keys:
            if k in manifest['parameter_bounds']:
                del manifest['parameter_bounds'][k]
                print(f"  REMOVE parameter_bounds.{k}")

    write(path, json.dumps(manifest, indent=2))
    with open(path) as f: json.load(f)
    print("  JSON valid — OK")


def fix_watcher():
    print("\n[2/2] agents/watcher_agent.py — strip warm_start_* before CLI build")
    path = 'agents/watcher_agent.py'
    backup(path)
    with open(path) as f:
        content = f.read()

    if 'warm_start_' in content and 'INTERNAL_ONLY_PARAMS' in content:
        print("  SKIP — already patched")
        return

    # Strip warm_start_* from final_params before CLI command is built
    # Anchor: right before the CLI build loop
    old_anchor = ('            for key, value in final_params.items():\n'
                  '                # Use manifest-declared CLI arg name if available,\n'
                  '                # otherwise fall back to underscore->hyphen conversion.\n'
                  '                cli_key = _param_to_cli.get(key, key.replace("_", "-"))')

    new_anchor = ('            # [S140b FIX] Strip internal-only params injected by WATCHER preflight\n'
                  '            # These are consumed by the bayesian optimizer via trial_history_context\n'
                  '            # dict — they are never CLI args and must not be passed to the script.\n'
                  '            _INTERNAL_ONLY_PARAMS = {\n'
                  "                'warm_start_window', 'warm_start_offset',\n"
                  "                'warm_start_skip_min', 'warm_start_skip_max',\n"
                  "                'warm_start_session', 'warm_start_fwd_thresh',\n"
                  "                'warm_start_rev_thresh',\n"
                  '            }\n'
                  '            _cli_params = {k: v for k, v in final_params.items()\n'
                  '                           if k not in _INTERNAL_ONLY_PARAMS}\n'
                  '\n'
                  '            for key, value in _cli_params.items():\n'
                  '                # Use manifest-declared CLI arg name if available,\n'
                  '                # otherwise fall back to underscore->hyphen conversion.\n'
                  '                cli_key = _param_to_cli.get(key, key.replace("_", "-"))')

    check(old_anchor in content, "CLI build anchor not found in watcher_agent.py")
    content = content.replace(old_anchor, new_anchor)
    write(path, content)

    r = subprocess.run(['python3', '-m', 'py_compile', path], capture_output=True)
    print(f"  {'✅' if r.returncode == 0 else '❌'} syntax: {path}")
    if r.returncode != 0:
        print(f"  {r.stderr.decode()}")
        sys.exit(1)
    print("  OK")


if __name__ == '__main__':
    if DRY_RUN:
        print("DRY RUN — no files modified\n")

    if not Path('window_optimizer.py').exists():
        print("ERROR: Run from ~/distributed_prng_analysis/")
        sys.exit(1)

    fix_manifest()
    fix_watcher()

    print("\n" + "="*50)
    print("✅ S140b warm-start fix applied")
    print("\nNext:")
    print("  git add agents/watcher_agent.py agent_manifests/window_optimizer.json")
    print("  git commit -m 'S140b fix: strip warm_start_* from CLI args — internal params only'")
    print("  git push origin main && git push public main")
    print("\nThen clear outputs and relaunch:")
    print("  rm -f optimal_window_config.json bidirectional_survivors.json")
    print("  [launch command]")
