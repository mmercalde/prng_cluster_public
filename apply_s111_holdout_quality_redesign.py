#!/usr/bin/env python3
"""
apply_s111_holdout_quality_redesign.py  (v2 — fixed)
======================================

Fully automatic patcher for S111 (v1.1 approved).

Edits:
- full_scoring_worker.py (Step 3):
    * imports holdout_quality helpers
    * computes holdout_features + holdout_quality per survivor (both batch + fallback paths)
    * retains holdout_hits unchanged (vestigial)
- meta_prediction_optimizer_anti_overfit.py (Step 5):
    * defaults to holdout_quality target
    * excludes holdout_quality from X feature set
    * writes autocorr diagnostics JSON when holdout_features exist

Constraints:
- No line-number assumptions.
- Idempotent: safe to re-run.
- Creates .bak_S111 backups if not already present.

v2 fixes:
- Sentinel placed as comment on next line (not inline breaking colon)
- Fallback path uses line-by-line replacement (not regex on multiline dict)
"""

from __future__ import annotations

import os
import re
import sys
import shutil


S111_SENTINEL_STEP3 = "# --- S111_HOLDOUT_QUALITY_INTEGRATION ---"
S111_SENTINEL_STEP5 = "# --- S111_TARGET_HOLDOUT_QUALITY ---"


def read_text(p: str) -> str:
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


def write_text(p: str, s: str) -> None:
    with open(p, "w", encoding="utf-8") as f:
        f.write(s)


def backup_once(p: str) -> None:
    bak = f"{p}.bak_S111"
    if not os.path.exists(bak):
        shutil.copy2(p, bak)


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def insert_after_import_block(src: str, insertion: str) -> str:
    lines = src.splitlines(True)
    last_imp = -1
    for i, ln in enumerate(lines[:300]):
        if ln.startswith("import ") or ln.startswith("from "):
            last_imp = i
        elif last_imp >= 0 and ln.strip() == "":
            continue
        elif last_imp >= 0:
            break

    if last_imp >= 0:
        lines.insert(last_imp + 1, "\n" + insertion + "\n")
        return "".join(lines)

    return insertion + "\n\n" + src


def patch_full_scoring_worker(path: str) -> None:
    src = read_text(path)
    if S111_SENTINEL_STEP3 in src:
        print(f"[S111] Step 3 already patched: {path}")
        return

    # 1) Add imports + helper function
    import_block = (
        f"{S111_SENTINEL_STEP3}\n"
        "# S111: Holdout quality computation (TFM-consistent)\n"
        "from holdout_quality import compute_holdout_quality, get_survivor_skip\n"
        "import inspect as _s111_inspect\n"
        "\n"
        "def _s111_extract_features_with_optional_skip(scorer, *, seed, lottery_history, skip_val):\n"
        "    \"\"\"Call scorer.extract_ml_features with skip if supported by signature.\"\"\"\n"
        "    fn = getattr(scorer, 'extract_ml_features', None)\n"
        "    if fn is None:\n"
        "        raise AttributeError('SurvivorScorer missing extract_ml_features')\n"
        "    try:\n"
        "        sig = _s111_inspect.signature(fn)\n"
        "        if 'skip' in sig.parameters:\n"
        "            return fn(seed=seed, lottery_history=lottery_history, skip=skip_val)\n"
        "    except Exception:\n"
        "        pass\n"
        "    return fn(seed=seed, lottery_history=lottery_history)\n"
    )
    src2 = insert_after_import_block(src, import_block)

    # 2) Insert holdout_quality block in batch path
    anchor_pat = r'^\s*result\[\s*["\']holdout_hits["\']\s*\]\s*=\s*holdout_hits_map\.get\(\s*seed\s*,\s*0\.0\s*\)\s*$'
    m = re.search(anchor_pat, src2, flags=re.MULTILINE)
    if not m:
        fail("S111 Step3: Could not find anchor line setting result['holdout_hits'] from holdout_hits_map.")

    insert_block = (
        "\n"
        "            # S111: Compute holdout_quality + holdout_features (holdout-only, TFM-consistent)\n"
        "            if holdout_history is not None and len(holdout_history) > 0:\n"
        "                try:\n"
        "                    meta = survivor_metadata.get(seed, {}) if survivor_metadata else {}\n"
        "                    skip_val = get_survivor_skip(meta)\n"
        "                    holdout_feats = _s111_extract_features_with_optional_skip(\n"
        "                        scorer,\n"
        "                        seed=seed,\n"
        "                        lottery_history=holdout_history,\n"
        "                        skip_val=skip_val,\n"
        "                    )\n"
        "                    result['holdout_features'] = {\n"
        "                        k: (float(v) if isinstance(v, (int, float)) else v)\n"
        "                        for k, v in (holdout_feats or {}).items()\n"
        "                    }\n"
        "                    result['holdout_quality'] = compute_holdout_quality(holdout_feats)\n"
        "                except Exception as e:\n"
        "                    logger.warning(f\"[S111] holdout_quality failed for seed {seed}: {e}\")\n"
        "                    result['holdout_features'] = {}\n"
        "                    result['holdout_quality'] = 0.0\n"
        "            else:\n"
        "                result['holdout_features'] = {}\n"
        "                result['holdout_quality'] = 0.0\n"
    )

    insert_pos = m.end()
    src3 = src2[:insert_pos] + insert_block + src2[insert_pos:]

    # 3) Patch fallback path using line-by-line approach
    lines = src3.split('\n')
    fallback_start = -1
    fallback_end = -1

    for i, ln in enumerate(lines):
        stripped = ln.strip()
        if stripped.startswith('results.append({') and fallback_start == -1:
            lookahead = '\n'.join(lines[i:i+12])
            if 'holdout_hits_map' in lookahead:
                fallback_start = i

        if fallback_start >= 0 and fallback_end == -1 and i >= fallback_start:
            if stripped.endswith('})'):
                fallback_end = i
                break

    if fallback_start >= 0 and fallback_end >= 0:
        orig_line = lines[fallback_start]
        ind = orig_line[:len(orig_line) - len(orig_line.lstrip())]

        replacement_lines = [
            ind + "fallback_result = {",
            ind + "    'seed': seed,",
            ind + "    'score': float(score),",
            ind + "    'features': features,",
            ind + "    'metadata': {'prng_type': prng_type, 'mod': mod,",
            ind + "                'worker_hostname': HOST, 'timestamp': time.time()},",
            ind + "    'holdout_hits': holdout_hits_map.get(seed, 0.0),",
            ind + "}",
            ind + "# S111: Compute holdout_quality in fallback path",
            ind + "if holdout_history is not None and len(holdout_history) > 0:",
            ind + "    try:",
            ind + "        meta = survivor_metadata.get(seed, {}) if survivor_metadata else {}",
            ind + "        skip_val = get_survivor_skip(meta)",
            ind + "        holdout_feats = _s111_extract_features_with_optional_skip(",
            ind + "            scorer,",
            ind + "            seed=seed,",
            ind + "            lottery_history=holdout_history,",
            ind + "            skip_val=skip_val,",
            ind + "        )",
            ind + "        fallback_result['holdout_features'] = {",
            ind + "            k: (float(v) if isinstance(v, (int, float)) else v)",
            ind + "            for k, v in (holdout_feats or {}).items()",
            ind + "        }",
            ind + "        fallback_result['holdout_quality'] = compute_holdout_quality(holdout_feats)",
            ind + "    except Exception as e2_hq:",
            ind + "        logger.warning(f'[S111] holdout_quality fallback failed for seed {seed}: {e2_hq}')",
            ind + "        fallback_result['holdout_features'] = {}",
            ind + "        fallback_result['holdout_quality'] = 0.0",
            ind + "else:",
            ind + "    fallback_result['holdout_features'] = {}",
            ind + "    fallback_result['holdout_quality'] = 0.0",
            ind + "results.append(fallback_result)",
        ]

        new_lines = lines[:fallback_start] + replacement_lines + lines[fallback_end + 1:]
        src4 = '\n'.join(new_lines)
    else:
        print("[S111][WARN] Step3 fallback results.append({...holdout_hits...}) block not found; leaving fallback path unchanged.")
        src4 = src3

    backup_once(path)
    write_text(path, src4)
    print(f"[S111] Patched Step 3: {path}")


def patch_meta_optimizer(path: str) -> None:
    src = read_text(path)
    if S111_SENTINEL_STEP5 in src:
        print(f"[S111] Step 5 already patched: {path}")
        return

    # 1) Replace holdout_hits with holdout_quality in compute_signal_quality default
    # Put sentinel as a comment on the line AFTER the def
    lines = src.split('\n')
    n1 = 0
    for i, ln in enumerate(lines):
        if 'def compute_signal_quality' in ln and 'holdout_hits' in ln:
            lines[i] = ln.replace('holdout_hits', 'holdout_quality')
            indent = ln[:len(ln) - len(ln.lstrip())]
            lines.insert(i + 1, indent + "    " + S111_SENTINEL_STEP5)
            n1 = 1
            break
    src2 = '\n'.join(lines)

    # 2) default target_field in training config
    src3, n2 = re.subn(
        r'(target_field\s*:\s*str\s*=\s*["\'])holdout_hits(["\'])',
        r'\1holdout_quality\2',
        src2,
        count=1
    )

    # 3) exclude_features: add holdout_quality wherever holdout_hits appears in exclude lists
    # Use simple string replacement - works regardless of line structure
    src4 = src3.replace(
        "['score', 'confidence', 'holdout_hits']",
        "['score', 'confidence', 'holdout_hits', 'holdout_quality']"
    ).replace(
        '["score", "confidence", "holdout_hits"]',
        '["score", "confidence", "holdout_hits", "holdout_quality"]'
    )
    # Count how many replacements happened
    n3 = src3.count("['score', 'confidence', 'holdout_hits']") + src3.count('["score", "confidence", "holdout_hits"]')

    # 4) Autocorr diagnostics helper
    if "compute_autocorrelation_diagnostics" not in src4:
        helper = (
            "\n# --- S111_AUTOCORR_DIAGNOSTICS ---\n"
            "def _s111_write_autocorr_if_available(survivors, out_path='diagnostics_outputs/holdout_feature_autocorr.json'):\n"
            "    try:\n"
            "        import os, json\n"
            "        from holdout_quality import compute_autocorrelation_diagnostics\n"
            "        if not survivors or not isinstance(survivors, list) or not isinstance(survivors[0], dict):\n"
            "            return None\n"
            "        if survivors[0].get('holdout_features') is None:\n"
            "            return None\n"
            "        out = compute_autocorrelation_diagnostics(survivors)\n"
            "        os.makedirs(os.path.dirname(out_path), exist_ok=True)\n"
            "        with open(out_path, 'w') as f:\n"
            "            json.dump(out, f, indent=2)\n"
            "        return out\n"
            "    except Exception:\n"
            "        return None\n"
            "# --- S111_AUTOCORR_DIAGNOSTICS_END ---\n"
        )
        src5 = insert_after_import_block(src4, helper)
    else:
        src5 = src4

    # Insert call after survivors JSON load
    call = (
        "\n    # --- S111_AUTOCORR_DIAGNOSTICS_CALL ---\n"
        "    try:\n"
        "        _ac = _s111_write_autocorr_if_available(survivors)\n"
        "        if _ac is not None:\n"
        "            print('[S111] Wrote diagnostics_outputs/holdout_feature_autocorr.json')\n"
        "    except Exception:\n"
        "        pass\n"
        "    # --- S111_AUTOCORR_DIAGNOSTICS_CALL_END ---\n"
    )

    sm = re.search(r"^\s*survivors\s*=\s*json\.load\(.+\)\s*$", src5, flags=re.MULTILINE)
    if sm:
        src6 = src5[:sm.end()] + call + src5[sm.end():]
    else:
        src6 = src5

    if n1 == 0:
        print("[S111][WARN] compute_signal_quality default target_name anchor not found.")
    if n2 == 0:
        print("[S111][WARN] target_field default anchor not found.")
    if n3 == 0:
        print("[S111][WARN] exclude_features anchor not found.")

    backup_once(path)
    write_text(path, src6)
    print(f"[S111] Patched Step 5: {path}")


def main() -> int:
    step3 = "full_scoring_worker.py"
    step5 = "meta_prediction_optimizer_anti_overfit.py"

    if not os.path.isfile(step3):
        fail(f"Missing {step3} in current directory.")
    if not os.path.isfile(step5):
        fail(f"Missing {step5} in current directory.")

    patch_full_scoring_worker(step3)
    patch_meta_optimizer(step5)

    print("[S111] Done. Backups created with .bak_S111 suffix.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[S111][FATAL] {e}")
        sys.exit(1)
