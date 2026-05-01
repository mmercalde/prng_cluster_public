#!/usr/bin/env python3
"""
S172 Threshold Patch — Two-fix surgical patch (refined per TB ruling 2026-04-30).

Run from: ~/distributed_prng_analysis/ on Zeus

Addresses two distinct issues found during S172 investigation:

  FIX 1 (defensive coding — DEAD-PATH HARDENING):
    Raise unsafe threshold defaults across fallback paths so that any
    misconfigured / partially-configured load path yields a SAFE threshold
    rather than a discovery-grade 0.01 or 0.25 value.

    These are dead-path defaults — the production Optuna flow does not
    currently use them. They exist to prevent future drift.

  FIX 2 (Optuna threshold drop — REAL BUG):
    The integration layer's `test_config(config, ft=bounds.default_forward_threshold,
    rt=bounds.default_reverse_threshold)` never reads `config.forward_threshold`,
    silently dropping Optuna's sampled threshold. Every trial uses the
    DEFAULT regardless of what Optuna sampled. Fix: read the threshold from
    `config.forward_threshold` and `config.reverse_threshold`, falling back
    to `bounds.default_*` only if missing.

    THIS IS THE PRIMARY FIX. With this in place, Optuna's TPE sampler can
    actually optimize FT/RT across the JSON-configured search space.

What this patch does NOT do:
    Does NOT modify distributed_config.json. Per TB's refined ruling, the
    JSON-configured search bounds are intentionally preserved so Optuna
    can continue exploring the full operator-set range. (The companion
    s172_config_patch.py raises window_size.min only — window=2 is
    mathematically broken regardless of what threshold Optuna picks.)

Usage on Zeus:
    cd ~/distributed_prng_analysis
    python3 s172_threshold_patch.py --dry-run     # preview only
    python3 s172_threshold_patch.py               # apply

Properties:
    - Deterministic anchor-based replacements (not regex)
    - AST-validated after every file edit
    - Idempotent (safe to re-run; reports "already applied" for each fix)
    - Creates timestamped backups (.s172_bak_<timestamp>)
    - Post-condition check verifies code defaults are safe and FIX 2 took effect
"""
from __future__ import annotations
import argparse
import ast
import datetime as _dt
import shutil
import sys
from pathlib import Path

# ─── Constants ────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent

SAFE_FT_MIN     = 0.40
SAFE_RT_MIN     = 0.40
SAFE_FT_DEFAULT = 0.50
SAFE_RT_DEFAULT = 0.50

WINDOW_OPTIMIZER_PATH = REPO_ROOT / "window_optimizer.py"
INTEGRATION_PATH      = REPO_ROOT / "window_optimizer_integration_final.py"

TIMESTAMP = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


# ─── Substitutions ────────────────────────────────────────────────────────────

# Each substitution is an (old_str, new_str, description) triple. Anchor-based:
# old_str must appear exactly ONCE in the file. AST-validated after each apply.
# Order is significant within a file — earlier substitutions must not invalidate
# later anchors.

WINDOW_OPTIMIZER_SUBS = [
    # FIX 1.A — config-loader defaults block
    (
        '        "forward_threshold": {"min": 0.001, "max": 0.10, "default": 0.01},\n'
        '        "reverse_threshold": {"min": 0.001, "max": 0.10, "default": 0.01}',
        '        "forward_threshold": {"min": 0.40, "max": 0.75, "default": 0.50},\n'
        '        "reverse_threshold": {"min": 0.40, "max": 0.75, "default": 0.50}',
        "FIX 1.A — load_search_bounds_from_config defaults",
    ),

    # FIX 1.B — SearchBounds dataclass min defaults
    (
        "    min_forward_threshold: float = 0.30\n"
        "    max_forward_threshold: float = 0.75\n"
        "    min_reverse_threshold: float = 0.30\n"
        "    max_reverse_threshold: float = 0.75",
        "    min_forward_threshold: float = 0.40\n"
        "    max_forward_threshold: float = 0.75\n"
        "    min_reverse_threshold: float = 0.40\n"
        "    max_reverse_threshold: float = 0.75",
        "FIX 1.B — SearchBounds min_*_threshold dataclass defaults",
    ),

    # FIX 1.C — SearchBounds dataclass default_*_threshold
    (
        "    default_forward_threshold: float = 0.30\n"
        "    default_reverse_threshold: float = 0.30",
        "    default_forward_threshold: float = 0.50\n"
        "    default_reverse_threshold: float = 0.50",
        "FIX 1.C — SearchBounds default_*_threshold dataclass defaults",
    ),

    # FIX 1.D — from_config fallback for missing 'default' key
    (
        '            default_forward_threshold=cfg["forward_threshold"].get("default", 0.01),\n'
        '            default_reverse_threshold=cfg["reverse_threshold"].get("default", 0.01)',
        '            default_forward_threshold=cfg["forward_threshold"].get("default", 0.50),\n'
        '            default_reverse_threshold=cfg["reverse_threshold"].get("default", 0.50)',
        "FIX 1.D — from_config 'default' key fallback raised 0.01 → 0.50",
    ),

    # FIX 1.E — baseline validation fallback
    (
        "        fwd = baseline.get('forward_threshold', 0.25)\n"
        "        rev = baseline.get('reverse_threshold', 0.25)",
        "        fwd = baseline.get('forward_threshold', 0.50)\n"
        "        rev = baseline.get('reverse_threshold', 0.50)",
        "FIX 1.E — baseline.get('*_threshold', 0.25) → 0.50",
    ),
]


# Integration file — the BIG fix is the threshold drop bug
INTEGRATION_SUBS = [
    # FIX 1.F — function default for run_bidirectional_test (defensive)
    (
        "                           forward_threshold: float = 0.01,\n"
        "                           reverse_threshold: float = 0.01,",
        "                           forward_threshold: float = 0.50,\n"
        "                           reverse_threshold: float = 0.50,",
        "FIX 1.F — run_bidirectional_test() function default 0.01 → 0.50",
    ),

    # FIX 2 — Optuna threshold drop bug.
    #
    # The current code uses bounds.default_forward_threshold as the keyword
    # default for ft/rt in test_config. This means EVERY trial uses the
    # default, ignoring config.forward_threshold (the Optuna-sampled value).
    #
    # Fix: read from config.forward_threshold / config.reverse_threshold
    # at CALL time, falling back to bounds.default_* only if missing.
    #
    # This is the more invasive change — it modifies test_config's body,
    # not just defaults.
    (
        "        def test_config(config,\n"
        "                        ss=seed_start, sc=seed_count,\n"
        "                        ft=bounds.default_forward_threshold,\n"
        "                        rt=bounds.default_reverse_threshold,\n"
        "                        optuna_trial=None):  # S115 M2\n"
        "            trial_counter['count'] += 1\n",
        "        def test_config(config,\n"
        "                        ss=seed_start, sc=seed_count,\n"
        "                        ft=None,\n"
        "                        rt=None,\n"
        "                        optuna_trial=None):  # S115 M2 + S172 threshold-drop fix\n"
        "            trial_counter['count'] += 1\n"
        "            # [S172] Read threshold from config (Optuna-sampled value).\n"
        "            # Previously ft/rt defaulted to bounds.default_*, dropping\n"
        "            # Optuna's per-trial threshold. Now we honor the WindowConfig\n"
        "            # value with bounds.default_* as the safety fallback.\n"
        "            if ft is None:\n"
        "                ft = getattr(config, 'forward_threshold', None) or bounds.default_forward_threshold\n"
        "            if rt is None:\n"
        "                rt = getattr(config, 'reverse_threshold', None) or bounds.default_reverse_threshold\n",
        "FIX 2 — Optuna threshold drop bug (test_config now honors config.forward_threshold)",
    ),
]


# ─── Patch logic ──────────────────────────────────────────────────────────────

def apply_subs(path: Path, subs: list, dry_run: bool) -> tuple[bool, list[str], list[str]]:
    """
    Apply a list of substitutions to a file.

    Returns (any_changes, applied_descriptions, skipped_descriptions).
    """
    if not path.exists():
        print(f"  ❌ File not found: {path}")
        return False, [], []

    original = path.read_text()
    text = original
    applied = []
    skipped = []

    for old, new, desc in subs:
        n_old = text.count(old)
        n_new = text.count(new)

        if n_old == 1:
            text = text.replace(old, new)
            applied.append(desc)
        elif n_old == 0 and n_new >= 1:
            skipped.append(f"{desc} (already applied)")
        elif n_old > 1:
            print(f"  ⚠️ Anchor '{desc}' matches {n_old} times — refusing to edit (ambiguous)")
            return False, [], []
        else:
            print(f"  ⚠️ Anchor '{desc}' not found — file may have drifted from expected version")
            print(f"     Old anchor (first 100 chars): {old[:100]!r}")
            return False, [], []

    if text == original:
        return False, [], skipped

    if dry_run:
        return True, applied, skipped

    # Validate AST before writing
    try:
        ast.parse(text)
    except SyntaxError as e:
        print(f"  ❌ AST parse FAILED after edits — refusing to write: {e}")
        return False, [], []

    # Backup original
    bak_path = path.with_suffix(path.suffix + f".s172_bak_{TIMESTAMP}")
    shutil.copy2(path, bak_path)
    print(f"  📦 Backup created: {bak_path.name}")

    path.write_text(text)
    return True, applied, skipped


def post_condition_check() -> bool:
    """Verify the patches took effect at code level.

    Per TB refined ruling 2026-04-30, the JSON-configured threshold bounds
    are intentionally preserved (Optuna explores them). So this check verifies:

      1. The patched DATACLASS DEFAULTS in window_optimizer.py meet safe floors.
         (These only fire if SearchBounds() is called with no args — defensive.)

      2. FIX 2 (test_config honors config.forward_threshold) is structurally
         present in the integration file.
    """
    print()
    print("─── Post-condition check ──────────────────────────────────────────────")

    # 1. Verify dataclass defaults via AST inspection (no module import required)
    src = WINDOW_OPTIMIZER_PATH.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"  ❌ window_optimizer.py AST parse failed: {e}")
        return False

    dataclass_defaults = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "SearchBounds":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    name = item.target.id
                    if name in (
                        "min_forward_threshold", "min_reverse_threshold",
                        "default_forward_threshold", "default_reverse_threshold",
                    ):
                        if isinstance(item.value, ast.Constant):
                            dataclass_defaults[name] = item.value.value

    print("  Dataclass defaults (in window_optimizer.py SearchBounds):")
    for k, v in dataclass_defaults.items():
        print(f"    {k} = {v}")

    failures = []
    if dataclass_defaults.get("min_forward_threshold", 0) < SAFE_FT_MIN:
        failures.append(f"dataclass min_forward_threshold {dataclass_defaults.get('min_forward_threshold')} < {SAFE_FT_MIN}")
    if dataclass_defaults.get("min_reverse_threshold", 0) < SAFE_RT_MIN:
        failures.append(f"dataclass min_reverse_threshold {dataclass_defaults.get('min_reverse_threshold')} < {SAFE_RT_MIN}")
    if dataclass_defaults.get("default_forward_threshold", 0) < SAFE_FT_DEFAULT:
        failures.append(f"dataclass default_forward_threshold {dataclass_defaults.get('default_forward_threshold')} < {SAFE_FT_DEFAULT}")
    if dataclass_defaults.get("default_reverse_threshold", 0) < SAFE_RT_DEFAULT:
        failures.append(f"dataclass default_reverse_threshold {dataclass_defaults.get('default_reverse_threshold')} < {SAFE_RT_DEFAULT}")

    # 2. Verify FIX 2 — test_config reads config.forward_threshold
    int_src = INTEGRATION_PATH.read_text()
    fix2_marker_a = "ft = getattr(config, 'forward_threshold', None) or bounds.default_forward_threshold"
    fix2_marker_b = "rt = getattr(config, 'reverse_threshold', None) or bounds.default_reverse_threshold"

    fix2_ft_present = fix2_marker_a in int_src
    fix2_rt_present = fix2_marker_b in int_src

    print()
    print("  FIX 2 (Optuna threshold drop) markers in integration file:")
    print(f"    forward read from config: {'✅ present' if fix2_ft_present else '❌ MISSING'}")
    print(f"    reverse read from config: {'✅ present' if fix2_rt_present else '❌ MISSING'}")

    if not fix2_ft_present:
        failures.append("FIX 2 forward marker missing — test_config did not get patched")
    if not fix2_rt_present:
        failures.append("FIX 2 reverse marker missing — test_config did not get patched")

    if failures:
        print()
        print("  ❌ POST-CONDITION FAILURES:")
        for f in failures:
            print(f"     {f}")
        return False

    print()
    print("  ✅ All code-level post-conditions satisfied")
    return True


def distributed_config_check() -> bool:
    """
    Read distributed_config.json and verify ONLY window_size.min (the
    mathematical invariant). Per TB refined ruling 2026-04-30, threshold
    bounds in the JSON are intentionally preserved for Optuna to explore.

    This check exists because the threshold-drop bug in test_config means
    Optuna's threshold sample was previously being dropped — so the JSON
    threshold bounds didn't matter. After FIX 2, they DO matter (Optuna
    will actually explore them), and that's the desired behavior.

    The only invariant this script enforces in the JSON is:
        window_size.min >= 6

    Because W=2 / W=3 produces ~39%/53% survivor rate by random chance
    alone, regardless of what threshold Optuna picks. This is a math
    constraint, not a tuning preference.
    """
    import json
    cfg_path = REPO_ROOT / "distributed_config.json"
    if not cfg_path.exists():
        print(f"  ⚠️ distributed_config.json not found at {cfg_path}")
        return False
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception as e:
        print(f"  ❌ distributed_config.json parse failed: {e}")
        return False

    sb = cfg.get("search_bounds", {})
    ws = sb.get("window_size", {})
    ft = sb.get("forward_threshold", {})
    rt = sb.get("reverse_threshold", {})

    print()
    print("─── distributed_config.json bounds ────────────────────────────────────")
    print(f"  window_size:        min={ws.get('min')}, max={ws.get('max')}, default={ws.get('default')}")
    print(f"  forward_threshold:  min={ft.get('min')}, max={ft.get('max')}, default={ft.get('default')}")
    print(f"  reverse_threshold:  min={rt.get('min')}, max={rt.get('max')}, default={rt.get('default')}")
    print()
    print("  (Threshold bounds intentionally preserved — Optuna will explore them.)")
    print()

    # Only check window_size.min — the math invariant
    SAFE_WINDOW_MIN = 6
    cur_window_min = ws.get("min", 0)
    if cur_window_min < SAFE_WINDOW_MIN:
        print(f"  ⚠️ window_size.min ({cur_window_min}) < {SAFE_WINDOW_MIN}")
        print(f"     W=2/W=3 produces ~39%/53% survivor rate by chance, regardless of threshold.")
        print(f"     Run the companion script: python3 s172_config_patch.py")
        return False

    print(f"  ✅ window_size.min = {cur_window_min} (>= {SAFE_WINDOW_MIN})")
    return True


def main():
    parser = argparse.ArgumentParser(description="S172 threshold patch")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    print("=" * 76)
    print(f"S172 THRESHOLD PATCH  ({'DRY RUN' if args.dry_run else 'APPLY'})")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Timestamp: {TIMESTAMP}")
    print("=" * 76)
    print()

    print(f"=== Patching {WINDOW_OPTIMIZER_PATH.name} ===")
    changed_a, applied_a, skipped_a = apply_subs(
        WINDOW_OPTIMIZER_PATH, WINDOW_OPTIMIZER_SUBS, args.dry_run
    )
    for d in applied_a: print(f"  ✅ Applied:   {d}")
    for d in skipped_a: print(f"  ⏭  Skipped:  {d}")
    if not changed_a and not skipped_a:
        print("  ❌ No substitutions could be applied — aborting")
        return 2

    print()
    print(f"=== Patching {INTEGRATION_PATH.name} ===")
    changed_b, applied_b, skipped_b = apply_subs(
        INTEGRATION_PATH, INTEGRATION_SUBS, args.dry_run
    )
    for d in applied_b: print(f"  ✅ Applied:   {d}")
    for d in skipped_b: print(f"  ⏭  Skipped:  {d}")
    if not changed_b and not skipped_b:
        print("  ❌ No substitutions could be applied — aborting")
        return 2

    if args.dry_run:
        print()
        print("─── DRY RUN — no files modified ──────────────────────────────────────")
        return 0

    # Post-condition checks
    runtime_ok = post_condition_check()
    config_ok  = distributed_config_check()

    print()
    print("=" * 76)
    if runtime_ok and config_ok:
        print("✅ PATCH APPLIED — code post-conditions pass, window_size.min >= 6")
    elif runtime_ok and not config_ok:
        print("⚠️  PATCH APPLIED — code-level fixes good, BUT distributed_config.json")
        print("    has window_size.min < 6 (math invariant). Run companion script:")
        print("        python3 s172_config_patch.py")
    else:
        print("❌ PATCH APPLIED but code-level post-condition check FAILED")
        print("    Manual investigation required.")
    print("=" * 76)
    return 0 if (runtime_ok and config_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
