#!/usr/bin/env python3
"""
S119 Patch: Add digit-wise agreement features to survivor_scorer.py

Adds 4 new features alongside existing CRT lanes (additive, zero removals):
  - hundreds_digit_agreement    (float 0.0-1.0)
  - tens_digit_agreement        (float 0.0-1.0)
  - ones_digit_agreement        (float 0.0-1.0)
  - expected_digit_match_count  (float 0.0-3.0)

Rationale: CA Lottery Daily 3 spec 03:00-09r = three independent Z10 draws.
Digit-level features are semantically correct. CRT lanes kept for ablation.

Approved: Team Alpha + Team Beta (S119)
Touches:  survivor_scorer.py only — 4 surgical insertions, zero deletions.

Deploy:
  scp ~/Downloads/apply_s119_digit_features.py rzeus:~/distributed_prng_analysis/
  ssh rzeus "cd ~/distributed_prng_analysis && \
    source ~/venvs/torch/bin/activate && \
    python3 apply_s119_digit_features.py"
"""

import sys
import shutil
from pathlib import Path

TARGET = Path.home() / "distributed_prng_analysis" / "survivor_scorer.py"

if not TARGET.exists():
    print(f"ERROR: {TARGET} not found. Run from home dir or check path.", file=sys.stderr)
    sys.exit(1)

# ── Backup ────────────────────────────────────────────────────────────────────
backup = TARGET.with_suffix(".py.s119_backup")
shutil.copy2(TARGET, backup)
print(f"Backup: {backup}")

content = TARGET.read_text()
original = content

# ═════════════════════════════════════════════════════════════════════════════
# PATCH 1 — Single-seed feature block (~line 424)
# Insert digit features immediately after lane_consistency line
# ═════════════════════════════════════════════════════════════════════════════
OLD_P1 = (
    "        features['lane_consistency'] = "
    "(features['lane_agreement_8'] + features['lane_agreement_125']) / 2\n"
    "\n"
    "        # Compute pred_min/max and residual features (FIX: these were never computed!)"
)

NEW_P1 = (
    "        features['lane_consistency'] = "
    "(features['lane_agreement_8'] + features['lane_agreement_125']) / 2\n"
    "\n"
    "        # Digit-wise agreement features (S119) — CA Lottery spec 03:00-09r\n"
    "        # Daily 3 = three independent Z10 draws; score each digit position directly.\n"
    "        # Additive alongside CRT lanes — do not remove CRT until ablation confirms redundancy.\n"
    "        _h = float(((pred // 100) % 10 == (act // 100) % 10).float().mean().item())\n"
    "        _t = float(((pred // 10)  % 10 == (act // 10)  % 10).float().mean().item())\n"
    "        _o = float(((pred)        % 10 == (act)        % 10).float().mean().item())\n"
    "        features['hundreds_digit_agreement']   = _h\n"
    "        features['tens_digit_agreement']       = _t\n"
    "        features['ones_digit_agreement']       = _o\n"
    "        # expected_digit_match_count: mean matched digit positions per draw (0.0-3.0)\n"
    "        features['expected_digit_match_count'] = _h + _t + _o\n"
    "\n"
    "        # Compute pred_min/max and residual features (FIX: these were never computed!)"
)

if OLD_P1 not in content:
    print("ERROR: Patch 1 anchor not found. Aborting — file unchanged.", file=sys.stderr)
    sys.exit(1)

content = content.replace(OLD_P1, NEW_P1, 1)
print("Patch 1 applied: single-seed digit features")

# ═════════════════════════════════════════════════════════════════════════════
# PATCH 2 — Batch GPU lane block (~line 599)
# Insert digit tensors immediately after lane_consistency tensor
# ═════════════════════════════════════════════════════════════════════════════
OLD_P2 = (
    "        lane_consistency = (lane_8 + lane_125) / 2\n"
    "        \n"
    "        # Residue features - batch compute for each mod"
)

NEW_P2 = (
    "        lane_consistency = (lane_8 + lane_125) / 2\n"
    "\n"
    "        # Digit-wise agreement features (S119) — CA Lottery spec 03:00-09r\n"
    "        # Vectorized: operates on (batch_size, n) tensors, result is (batch_size,)\n"
    "        _hd = ((predictions // 100) % 10 == (hist_expanded // 100) % 10).float().mean(dim=1)\n"
    "        _td = ((predictions // 10)  % 10 == (hist_expanded // 10)  % 10).float().mean(dim=1)\n"
    "        _od = ((predictions)        % 10 == (hist_expanded)        % 10).float().mean(dim=1)\n"
    "        _edc = _hd + _td + _od  # expected_digit_match_count: 0.0-3.0\n"
    "        \n"
    "        # Residue features - batch compute for each mod"
)

if OLD_P2 not in content:
    print("ERROR: Patch 2 anchor not found. Aborting — restoring backup.", file=sys.stderr)
    TARGET.write_text(original)
    sys.exit(1)

content = content.replace(OLD_P2, NEW_P2, 1)
print("Patch 2 applied: batch GPU digit tensors")

# ═════════════════════════════════════════════════════════════════════════════
# PATCH 3 — Batch results_gpu dict (~line 698)
# Add 4 digit tensor entries between lane_consistency and temporal_stability_mean
# ═════════════════════════════════════════════════════════════════════════════
OLD_P3 = (
    "            'lane_agreement_8': lane_8,\n"
    "            'lane_agreement_125': lane_125,\n"
    "            'lane_consistency': lane_consistency,\n"
    "            'temporal_stability_mean': temporal_mean,"
)

NEW_P3 = (
    "            'lane_agreement_8': lane_8,\n"
    "            'lane_agreement_125': lane_125,\n"
    "            'lane_consistency': lane_consistency,\n"
    "            # Digit-wise features (S119)\n"
    "            'hundreds_digit_agreement':   _hd,\n"
    "            'tens_digit_agreement':       _td,\n"
    "            'ones_digit_agreement':       _od,\n"
    "            'expected_digit_match_count': _edc,\n"
    "            'temporal_stability_mean': temporal_mean,"
)

if OLD_P3 not in content:
    print("ERROR: Patch 3 anchor not found. Aborting — restoring backup.", file=sys.stderr)
    TARGET.write_text(original)
    sys.exit(1)

content = content.replace(OLD_P3, NEW_P3, 1)
print("Patch 3 applied: results_gpu dict entries")

# ═════════════════════════════════════════════════════════════════════════════
# PATCH 4 — _empty_ml_features() key list (~line 464)
# Extend the key list so fallback/error records have consistent schema
# ═════════════════════════════════════════════════════════════════════════════
OLD_P4 = (
    "                 'lane_agreement_8','lane_agreement_125','lane_consistency']\n"
    "        # Battery Tier 1A (23 columns) — S113"
)

NEW_P4 = (
    "                 'lane_agreement_8','lane_agreement_125','lane_consistency',\n"
    "                 # Digit-wise features (S119)\n"
    "                 'hundreds_digit_agreement','tens_digit_agreement',\n"
    "                 'ones_digit_agreement','expected_digit_match_count']\n"
    "        # Battery Tier 1A (23 columns) — S113"
)

if OLD_P4 not in content:
    print("ERROR: Patch 4 anchor not found. Aborting — restoring backup.", file=sys.stderr)
    TARGET.write_text(original)
    sys.exit(1)

content = content.replace(OLD_P4, NEW_P4, 1)
print("Patch 4 applied: _empty_ml_features() key list")

# ── Write ─────────────────────────────────────────────────────────────────────
TARGET.write_text(content)

# ── Verify ────────────────────────────────────────────────────────────────────
final = TARGET.read_text()
checks = {
    'hundreds_digit_agreement':   3,   # touch 1, touch 3, touch 4
    'tens_digit_agreement':       3,
    'ones_digit_agreement':       3,
    'expected_digit_match_count': 3,
    'lane_agreement_8':           3,   # must still be present (regression check)
    'lane_agreement_125':         3,
    'lane_consistency':           3,
}

print("\n── Verification ─────────────────────────────────────────────────────")
all_ok = True
for key, min_count in checks.items():
    count = final.count(key)
    ok = count >= min_count
    status = "OK" if ok else "FAIL"
    print(f"  {status}  '{key}' appears {count}x (expected >={min_count})")
    if not ok:
        all_ok = False

print()
if all_ok:
    print("ALL PATCHES APPLIED SUCCESSFULLY.")
    print(f"File   : {TARGET}")
    print(f"Backup : {backup}")
    print()
    print("Next steps:")
    print("  1. Run smoke test:")
    print("     python3 -c \"")
    print("       import json, survivor_scorer")
    print("       s = survivor_scorer.SurvivorScorer()")
    print("       h = [r['draw'] for r in json.load(open('daily3.json'))[:62]]")
    print("       f = s.extract_ml_features(seed=12345, lottery_history=h)")
    print("       keys = ['hundreds_digit_agreement','tens_digit_agreement',")
    print("               'ones_digit_agreement','expected_digit_match_count',")
    print("               'lane_agreement_8','lane_agreement_125','lane_consistency']")
    print("       [print(k, f.get(k,'MISSING')) for k in keys]")
    print("     \"")
    print()
    print("  2. Run pipeline from Step 3:")
    print("     PYTHONPATH=. python3 agents/watcher_agent.py \\")
    print("       --run-pipeline --start-step 3 --end-step 6")
else:
    print("PATCH FAILED — one or more checks failed.")
    print("Restoring backup...")
    TARGET.write_text(original)
    print(f"Restored from {backup}")
    sys.exit(1)
