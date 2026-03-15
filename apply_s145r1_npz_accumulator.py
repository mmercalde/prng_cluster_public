#!/usr/bin/env python3
"""
apply_s145r1_npz_accumulator.py
================================
S145-R1 v2 — Replace JSON accumulator with NPZ→NPZ merge

Replaces the JSON-based accumulator (installed by apply_s145r1_progressive_sweep.py)
with a direct NPZ→NPZ merge. Eliminates the 700MB+ JSON intermediary file entirely.

WHY:
  - JSON with indent=2 at 1M+ survivors = 700MB-1GB per run
  - NPZ compressed = ~6 bytes/survivor = ~6MB for 1M survivors
  - Steps 2-6 consume bidirectional_survivors_binary.npz exclusively
  - bidirectional_survivors.json is never read by any downstream step
  - convert_survivors_to_binary.py remains untouched — still works for manual use

BACKWARD COMPATIBILITY:
  - bidirectional_survivors.json: still written per-run (compact JSON, no indent)
  - bidirectional_survivors_binary.npz: same path, same schema, same 22 fields
  - Steps 2-6: zero changes — they read NPZ as always
  - convert_survivors_to_binary.py: untouched standalone utility
  - Any run of any size/trial-count works identically

NEW FILES:
  - bidirectional_survivors_all.npz: persistent NPZ accumulator (replaces _all.json)

CHANGES:
  1. window_optimizer_integration_final.py
     - Remove indent=2 from bidirectional_survivors.json write (compact JSON)
     - Replace JSON accumulator block with NPZ merge block
     - NPZ merge: load prior NPZ arrays, concat with new, dedup by seed (best score wins)
     - Write final merged arrays directly as bidirectional_survivors_binary.npz
     - No subprocess call to convert_survivors_to_binary.py needed

  2. .gitignore
     - Remove !bidirectional_survivors_all.json exception (no longer needed)
     - Add !bidirectional_survivors_all.npz exception

Usage:
    python3 apply_s145r1_npz_accumulator.py [--dry-run]
"""

import sys
import shutil
import json
from pathlib import Path

DRY_RUN = '--dry-run' in sys.argv
PROJECT_ROOT = Path('/home/michael/distributed_prng_analysis')

FILES = {
    'integration': PROJECT_ROOT / 'window_optimizer_integration_final.py',
    'gitignore':   PROJECT_ROOT / '.gitignore',
}

def read(p): return Path(p).read_text(encoding='utf-8')
def write(p, c):
    if DRY_RUN:
        print(f"  [DRY-RUN] would write {p.name}")
        return
    Path(p).write_text(c, encoding='utf-8')

def backup(path):
    bak = Path(str(path) + '.s145r1v2_backup')
    if DRY_RUN:
        print(f"  [DRY-RUN] would backup → {bak.name}")
        return
    shutil.copy2(path, bak)
    print(f"  ✅ Backup: {bak.name}")

def apply_patch(content, old, new, description):
    if old not in content:
        print(f"  ❌ ANCHOR NOT FOUND: {description}")
        return content, False
    result = content.replace(old, new, 1)
    print(f"  ✅ Patched: {description}")
    return result, True

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 1 — window_optimizer_integration_final.py
# Two sub-patches:
#   A) Remove indent=2 from bidirectional_survivors.json write
#   B) Replace JSON accumulator + subprocess NPZ conversion with NPZ merge
# ─────────────────────────────────────────────────────────────────────────────

def patch_integration(path):
    print("\n[1/2] window_optimizer_integration_final.py")
    content = read(path)
    original_lines = len(content.splitlines())
    backup(path)
    all_ok = True

    # ── Sub-patch A: Remove indent=2 from per-run JSON write ─────────────────
    OLD_A = (
        "            with open('bidirectional_survivors.json', 'w') as f:\n"
        "                json.dump(sorted(bidirectional_deduped, key=lambda x: x['seed']), f, indent=2)\n"
        "            print(f\"✅ Saved bidirectional_survivors.json: {len(bidirectional_deduped)} unique seeds\")"
    )
    NEW_A = (
        "            with open('bidirectional_survivors.json', 'w') as f:\n"
        "                json.dump(sorted(bidirectional_deduped, key=lambda x: x['seed']), f)\n"
        "            print(f\"✅ Saved bidirectional_survivors.json: {len(bidirectional_deduped)} unique seeds\")"
    )
    content, ok = apply_patch(content, OLD_A, NEW_A,
        "remove indent=2 from bidirectional_survivors.json write")
    all_ok = all_ok and ok

    # ── Sub-patch B: Replace JSON accumulator + subprocess with NPZ merge ────
    # Anchor: the S145-R1 accumulator block installed by apply_s145r1_progressive_sweep.py
    # Anchor built from exact live Zeus text (retrieved via ssh sed)
    OLD_B = (
        "            # [S145-R1] SURVIVOR ACCUMULATOR — merge into persistent cross-run store\n"
        "            # Merge policy: best per-seed score wins on conflict (TB ruling S145-R1)\n"
        "            # bidirectional_survivors.json still written above — no change to existing output\n"
        "            import os as _os_s145\n"
        "            _accum_path = 'bidirectional_survivors_all.json'\n"
        "            try:\n"
        "                if _os_s145.path.exists(_accum_path):\n"
        "                    with open(_accum_path) as _af:\n"
        "                        _prior_survivors = json.load(_af)\n"
        "                else:\n"
        "                    _prior_survivors = []\n"
        "                _prior_count = len(_prior_survivors)\n"
        "                # Merge — best per-seed score wins on conflict\n"
        "                _merged = {s['seed']: s for s in _prior_survivors}\n"
        "                for s in bidirectional_deduped:\n"
        "                    if s['seed'] not in _merged or                        float(s.get('score', 0)) > float(_merged[s['seed']].get('score', 0)):\n"
        "                        _merged[s['seed']] = s\n"
        "                _merged_list = sorted(_merged.values(), key=lambda x: x['seed'])\n"
        "                with open(_accum_path, 'w') as _af:\n"
        "                    json.dump(_merged_list, _af)\n"
        "                _net_new = len(_merged_list) - _prior_count\n"
        "                print(f\"\\n[S145-R1][ACCUMULATOR] {len(_merged_list):,} total survivors across all runs\")\n"
        "                print(f\"   This run: +{len(bidirectional_deduped):,} candidates | Net new: +{_net_new:,}\")\n"
        "                print(f\"   Accumulator: {_accum_path}\")\n"
        "            except Exception as _accum_err:\n"
        "                print(f\"\\n⚠️  [S145-R1][ACCUMULATOR] Failed (non-fatal): {_accum_err}\")\n"
        "                print(f\"   Falling back to per-run NPZ conversion\")\n"
        "                _accum_path = 'bidirectional_survivors.json'\n"
        "\n"
        "            # Convert accumulated set to NPZ binary format (required by Step 2)\n"
        "            # Uses accumulator if available, falls back to per-run file on accumulator error\n"
        "            from subprocess import run as subprocess_run, CalledProcessError\n"
        "            try:\n"
        "                subprocess_run(\n"
        "                    [\"python3\", \"convert_survivors_to_binary.py\", _accum_path],\n"
        "                    check=True\n"
        "                )\n"
        "                print(f\"✅ Converted {_accum_path} to bidirectional_survivors_binary.npz\")\n"
        "            except CalledProcessError as e:\n"
        "                print(f\"❌ NPZ conversion failed: {e}\")\n"
        "                raise RuntimeError(\"Step 1 incomplete - NPZ conversion required for Step 2\")"
    )

    NEW_B = '''            # [S145-R1 v2] NPZ ACCUMULATOR — direct NPZ→NPZ merge
            # Replaces JSON accumulator (v1) — eliminates 700MB+ JSON intermediary
            # Merge policy: best per-seed score wins on conflict (TB ruling S145-R1)
            # Backward compatible: bidirectional_survivors_binary.npz same path/schema/22 fields
            # Steps 2-6 unaffected — they consume bidirectional_survivors_binary.npz exclusively
            import os as _os_s145
            import numpy as _np_s145
            _SKIP_ENC = {'constant': 0, 'variable': 1}
            _PRNG_ENC = {
                'java_lcg': 0, 'java_lcg_reverse': 1,
                'mt19937': 2, 'mt19937_reverse': 3,
                'xorshift128': 4, 'xorshift128_reverse': 5,
                'lcg32': 6, 'lcg32_reverse': 7,
                'minstd': 8, 'minstd_reverse': 9,
                'randu': 10, 'randu_reverse': 11,
            }

            def _survivors_to_arrays(survivors):
                """Convert list of survivor dicts to NPZ field arrays."""
                def _parse_skip_range(val):
                    if isinstance(val, int): return val
                    if isinstance(val, (list, tuple)) and len(val) == 2:
                        return int(val[1]) - int(val[0])
                    if isinstance(val, str) and '-' in val:
                        try: return int(val.split('-')[1]) - int(val.split('-')[0])
                        except: return 0
                    try: return int(val)
                    except: return 0
                n = len(survivors)
                return {
                    'seeds':                  _np_s145.array([s['seed'] for s in survivors], dtype=_np_s145.uint32),
                    'forward_matches':        _np_s145.array([s.get('forward_match_rate', s.get('forward_matches', 0.0)) for s in survivors], dtype=_np_s145.float32),
                    'reverse_matches':        _np_s145.array([s.get('reverse_match_rate', s.get('reverse_matches', 0.0)) for s in survivors], dtype=_np_s145.float32),
                    'window_size':            _np_s145.array([s.get('window_size', 0) for s in survivors], dtype=_np_s145.int32),
                    'offset':                 _np_s145.array([s.get('offset', 0) for s in survivors], dtype=_np_s145.int32),
                    'trial_number':           _np_s145.array([s.get('trial_number', 0) for s in survivors], dtype=_np_s145.int32),
                    'skip_min':               _np_s145.array([s.get('skip_min', 0) for s in survivors], dtype=_np_s145.int32),
                    'skip_max':               _np_s145.array([s.get('skip_max', 0) for s in survivors], dtype=_np_s145.int32),
                    'skip_range':             _np_s145.array([_parse_skip_range(s.get('skip_range', 0)) for s in survivors], dtype=_np_s145.int32),
                    'forward_count':          _np_s145.array([s.get('forward_count', 0.0) for s in survivors], dtype=_np_s145.float32),
                    'reverse_count':          _np_s145.array([s.get('reverse_count', 0.0) for s in survivors], dtype=_np_s145.float32),
                    'bidirectional_count':    _np_s145.array([s.get('bidirectional_count', 0.0) for s in survivors], dtype=_np_s145.float32),
                    'intersection_count':     _np_s145.array([s.get('intersection_count', 0.0) for s in survivors], dtype=_np_s145.float32),
                    'intersection_ratio':     _np_s145.array([s.get('intersection_ratio', 0.0) for s in survivors], dtype=_np_s145.float32),
                    'intersection_weight':    _np_s145.array([s.get('intersection_weight', 0.0) for s in survivors], dtype=_np_s145.float32),
                    'bidirectional_selectivity': _np_s145.array([s.get('bidirectional_selectivity', 0.0) for s in survivors], dtype=_np_s145.float32),
                    'forward_only_count':     _np_s145.array([s.get('forward_only_count', 0.0) for s in survivors], dtype=_np_s145.float32),
                    'reverse_only_count':     _np_s145.array([s.get('reverse_only_count', 0.0) for s in survivors], dtype=_np_s145.float32),
                    'survivor_overlap_ratio': _np_s145.array([s.get('survivor_overlap_ratio', 0.0) for s in survivors], dtype=_np_s145.float32),
                    'score':                  _np_s145.array([s.get('score', 0.0) for s in survivors], dtype=_np_s145.float32),
                    'skip_mode':              _np_s145.array([_SKIP_ENC.get(s.get('skip_mode', 'constant'), 0) for s in survivors], dtype=_np_s145.uint8),
                    'prng_type':              _np_s145.array([_PRNG_ENC.get(s.get('prng_type', s.get('prng_base', 'java_lcg')), 0) for s in survivors], dtype=_np_s145.uint8),
                }

            _accum_npz = 'bidirectional_survivors_all.npz'
            try:
                # Load prior accumulated NPZ if exists
                if _os_s145.path.exists(_accum_npz):
                    _prior_npz = _np_s145.load(_accum_npz)
                    _prior_seeds = _prior_npz['seeds'].astype(_np_s145.int64)
                    _prior_scores = _prior_npz['score'].astype(_np_s145.float32)
                    _prior_count = len(_prior_seeds)
                    # Build seed→index map for prior
                    _prior_idx = {int(_prior_seeds[i]): i for i in range(_prior_count)}
                else:
                    _prior_npz = None
                    _prior_seeds = _np_s145.array([], dtype=_np_s145.int64)
                    _prior_scores = _np_s145.array([], dtype=_np_s145.float32)
                    _prior_idx = {}
                    _prior_count = 0

                # Convert current run survivors to arrays
                _new_arrays = _survivors_to_arrays(bidirectional_deduped)
                _new_seeds = _new_arrays['seeds'].astype(_np_s145.int64)
                _new_scores = _new_arrays['score']

                # Determine indices: keep from prior (not beaten), add from new (new or better)
                _keep_prior = []  # indices into prior arrays
                _keep_new = []    # indices into new arrays

                # Track which prior seeds are superseded by new
                _superseded = set()
                for _ni in range(len(_new_seeds)):
                    _seed = int(_new_seeds[_ni])
                    if _seed not in _prior_idx:
                        _keep_new.append(_ni)  # Genuinely new seed
                    else:
                        _pi = _prior_idx[_seed]
                        if float(_new_scores[_ni]) > float(_prior_scores[_pi]):
                            _keep_new.append(_ni)   # New run has better score
                            _superseded.add(_pi)
                        # else: prior has equal or better score — keep prior

                # Keep all prior seeds not superseded
                _keep_prior = [i for i in range(_prior_count) if i not in _superseded]

                # Build merged field arrays
                _FIELDS_INT32  = ['window_size','offset','trial_number','skip_min','skip_max','skip_range']
                _FIELDS_FLOAT32 = ['forward_matches','reverse_matches','forward_count','reverse_count',
                                   'bidirectional_count','intersection_count','intersection_ratio',
                                   'intersection_weight','bidirectional_selectivity','forward_only_count',
                                   'reverse_only_count','survivor_overlap_ratio','score']
                _FIELDS_UINT8  = ['skip_mode','prng_type']
                _FIELDS_UINT32 = ['seeds']

                _merged_arrays = {}
                for _fname in _FIELDS_UINT32 + _FIELDS_INT32 + _FIELDS_FLOAT32 + _FIELDS_UINT8:
                    _dtype = (_np_s145.uint32 if _fname in _FIELDS_UINT32 else
                              _np_s145.int32  if _fname in _FIELDS_INT32  else
                              _np_s145.uint8  if _fname in _FIELDS_UINT8  else
                              _np_s145.float32)
                    _parts = []
                    if _keep_prior and _prior_npz is not None and _fname in _prior_npz:
                        _parts.append(_prior_npz[_fname][_keep_prior].astype(_dtype))
                    if _keep_new and _fname in _new_arrays:
                        _parts.append(_new_arrays[_fname][_keep_new].astype(_dtype))
                    if _parts:
                        _merged_arrays[_fname] = _np_s145.concatenate(_parts)
                    else:
                        _merged_arrays[_fname] = _np_s145.array([], dtype=_dtype)

                # Sort merged arrays by seed value
                _sort_idx = _np_s145.argsort(_merged_arrays['seeds'])
                for _fname in _merged_arrays:
                    _merged_arrays[_fname] = _merged_arrays[_fname][_sort_idx]

                _total = len(_merged_arrays['seeds'])
                _net_new = len(_keep_new)
                _superseded_count = len(_superseded)

                # Save accumulator NPZ
                _np_s145.savez_compressed(_accum_npz, **_merged_arrays)

                # Save as canonical bidirectional_survivors_binary.npz (Steps 2-6 input)
                _np_s145.savez_compressed('bidirectional_survivors_binary.npz', **_merged_arrays)

                print(f"\\n[S145-R1 v2][NPZ ACCUMULATOR] {_total:,} total survivors across all runs")
                print(f"   Prior kept:   {len(_keep_prior):,}")
                print(f"   Net new:      +{_net_new:,}")
                print(f"   Superseded:   {_superseded_count:,} (prior seeds beaten by new score)")
                print(f"   Accumulator:  {_accum_npz}")
                print(f"✅ bidirectional_survivors_binary.npz written ({_total:,} seeds, 22 fields)")

            except Exception as _accum_err:
                print(f"\\n⚠️  [S145-R1 v2][NPZ ACCUMULATOR] Failed: {_accum_err}")
                print(f"   Falling back to per-run convert_survivors_to_binary.py")
                import traceback as _tb_s145
                _tb_s145.print_exc()
                # Fallback: use original conversion path
                from subprocess import run as subprocess_run, CalledProcessError
                try:
                    subprocess_run(
                        ["python3", "convert_survivors_to_binary.py",
                         "bidirectional_survivors.json"],
                        check=True
                    )
                    print(f"✅ Fallback: converted bidirectional_survivors.json to NPZ")
                except CalledProcessError as _e:
                    print(f"❌ NPZ conversion failed: {_e}")
                    raise RuntimeError("Step 1 incomplete - NPZ conversion required for Step 2")'''

    content, ok = apply_patch(content, OLD_B, NEW_B,
        "replace JSON accumulator with NPZ→NPZ merge")
    all_ok = all_ok and ok

    if all_ok:
        write(path, content)
        new_lines = len(content.splitlines())
        print(f"  Lines: {original_lines} → {new_lines} (+{new_lines - original_lines})")
    return all_ok

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 2 — .gitignore
# Swap JSON accumulator exception for NPZ accumulator exception
# ─────────────────────────────────────────────────────────────────────────────

def patch_gitignore(path):
    print("\n[2/2] .gitignore — swap accumulator exception JSON→NPZ")
    content = read(path)
    original_lines = len(content.splitlines())
    backup(path)

    OLD_G = ('!bidirectional_survivors_all.json   '
             '# [S145-R1] persistent cross-run survivor accumulator')
    NEW_G = ('!bidirectional_survivors_all.npz    '
             '# [S145-R1 v2] persistent NPZ accumulator (replaces JSON)')

    if OLD_G not in content:
        # Try to add it if not present at all
        if 'bidirectional_survivors_all' not in content:
            OLD_G2 = '!schema_*.json'
            NEW_G2 = ('!schema_*.json\n'
                      '!bidirectional_survivors_all.npz    '
                      '# [S145-R1 v2] persistent NPZ accumulator')
            content, ok = apply_patch(content, OLD_G2, NEW_G2,
                "add NPZ accumulator exception (fresh)")
        else:
            print("  ⚠️  Unexpected gitignore state — manual review needed")
            return False
    else:
        content, ok = apply_patch(content, OLD_G, NEW_G,
            "swap accumulator exception JSON → NPZ")

    if ok:
        # Validate no *.npz ignore rule would block this
        if '*.npz' in content:
            print("  ⚠️  WARNING: *.npz wildcard found in gitignore — verify exception takes precedence")
        write(path, content)
        new_lines = len(content.splitlines())
        print(f"  Lines: {original_lines} → {new_lines}")
    return ok

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("apply_s145r1_npz_accumulator.py")
    print("S145-R1 v2 — NPZ→NPZ accumulator (replaces JSON accumulator)")
    if DRY_RUN:
        print("MODE: DRY RUN")
    print("=" * 65)

    print("\nVerifying target files...")
    for name, path in FILES.items():
        if path.exists():
            print(f"  ✅ {path.name} ({len(read(path).splitlines())} lines)")
        else:
            print(f"  ❌ NOT FOUND: {path}")
            sys.exit(1)

    results = []
    results.append(patch_integration(FILES['integration']))
    results.append(patch_gitignore(FILES['gitignore']))

    print("\n" + "=" * 65)
    passed = sum(results)
    print(f"RESULT: {passed}/2 patches applied")

    if passed == 2:
        print("\n✅ ALL PATCHES APPLIED")
        print("\nNext steps:")
        print("  1. Smoke test (clean run):")
        print("     bash s145r1_smoke_tests.sh")
        print("  2. Verify in log:")
        print("     [S145-R1 v2][NPZ ACCUMULATOR]")
        print("     bidirectional_survivors_all.npz exists")
        print("  3. Commit:")
        print("     git add -f window_optimizer_integration_final.py .gitignore")
        print("     git add -f apply_s145r1_npz_accumulator.py")
        print("     git add -f bidirectional_survivors_all.npz bidirectional_survivors_binary.npz")
        print("     git commit -m 'feat(s145-r1v2): NPZ accumulator replaces JSON — eliminates 700MB intermediary'")
        print("     git push origin main && git push public main")
    else:
        print("\n⚠️  SOME PATCHES FAILED — check anchors above")
        sys.exit(1)

if __name__ == '__main__':
    main()
