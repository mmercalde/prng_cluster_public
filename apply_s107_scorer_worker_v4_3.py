#!/usr/bin/env python3
"""
apply_s107_scorer_worker_v4_3.py
=================================
Patches scorer_trial_worker.py to v4.3 — enrichment objective (TB ruling S107).

CHANGES:
  1. Add npz_skip_mode to globals
  2. Load skip_mode in load_data()
  3. Replace bc_score/ir_score objective with skip_mode enrichment
  4. Update _log_trial_metrics signature and call sites
  5. Update docstring version marker

OBJECTIVE (TB ruling):
    is_rare  = (skip_mode == 1)
    p_global = mean(is_rare)
    p_sub    = mean(is_rare[mask])
    enrich   = log((p_sub + eps) / (p_global + eps))

    objective = 0.70 * tanh(enrich)
              + 0.20 * coverage
              - 0.10 * size_penalty
    clipped to [-1, 1]
"""

import re
import sys
import ast
from pathlib import Path

TARGET = Path("scorer_trial_worker.py")
BACKUP = Path("scorer_trial_worker.py.bak_v4_2")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(path):
    return path.read_text(encoding="utf-8")

def save(path, text):
    path.write_text(text, encoding="utf-8")

def check(label, condition, detail=""):
    status = "OK " if condition else "FAIL"
    print(f"  [{status}] {label}" + (f": {detail}" if detail else ""))
    return condition

def patch(text, old, new, label):
    if old not in text:
        print(f"  [FAIL] Anchor not found: {label}")
        return text, False
    count = text.count(old)
    if count > 1:
        print(f"  [WARN] Anchor appears {count}x — replacing first: {label}")
    result = text.replace(old, new, 1)
    print(f"  [OK  ] Patched: {label}")
    return result, True

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("scorer_trial_worker.py v4.3 patcher")
    print("Enrichment objective (TB ruling S107)")
    print("=" * 60)

    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found. Run from distributed_prng_analysis/")
        sys.exit(1)

    # Backup
    BACKUP.write_text(TARGET.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Backup: {BACKUP}")

    text = load(TARGET)
    results = []

    # ------------------------------------------------------------------
    # PRE-FLIGHT CHECKS
    # ------------------------------------------------------------------
    print("\n--- Pre-flight checks ---")
    results.append(check("v4.2 bc_score present", "bc_score" in text))
    results.append(check("v4.2 ir_score present", "ir_score" in text))
    results.append(check("npz_bidirectional_count global", "npz_bidirectional_count = None" in text))
    results.append(check("load_data present", "def load_data(" in text))
    results.append(check("run_trial present", "def run_trial(" in text))
    results.append(check("_log_trial_metrics present", "def _log_trial_metrics(" in text))
    results.append(check("skip_mode NOT yet loaded", "npz_skip_mode" not in text))

    failed = sum(1 for r in results if not r)
    if failed > 0:
        print(f"\n{failed} pre-flight check(s) failed. Aborting.")
        sys.exit(1)
    print("All pre-flight checks passed.")

    # ------------------------------------------------------------------
    # PATCH 1: Add npz_skip_mode to globals block
    # ------------------------------------------------------------------
    print("\n--- Patch 1: Add npz_skip_mode to globals ---")
    OLD_GLOBALS = (
        "npz_bidirectional_count = None   # float32 ndarray -- survival frequency (v4.2)\n"
        "npz_intersection_ratio  = None   # float32 ndarray -- bidirectional tightness (v4.2)\n"
        "npz_trial_number        = None   # int32   ndarray -- trial_number per seed (v4.1)"
    )
    NEW_GLOBALS = (
        "npz_bidirectional_count = None   # float32 ndarray -- survival frequency (v4.2)\n"
        "npz_intersection_ratio  = None   # float32 ndarray -- bidirectional tightness (v4.2)\n"
        "npz_trial_number        = None   # int32   ndarray -- trial_number per seed (v4.1)\n"
        "npz_skip_mode           = None   # int32   ndarray -- skip_mode per seed (v4.3)"
    )
    text, ok = patch(text, OLD_GLOBALS, NEW_GLOBALS, "globals: add npz_skip_mode")
    if not ok: sys.exit(1)

    # ------------------------------------------------------------------
    # PATCH 2: Load skip_mode in load_data (after trial_number block)
    # ------------------------------------------------------------------
    print("\n--- Patch 2: Load skip_mode in load_data ---")
    OLD_LOAD = (
        "            if 'trial_number' in survivors:\n"
        "                npz_trial_number = survivors['trial_number'].astype(_np.int32)\n"
        "                logger.info(\n"
        "                    f'NPZ trial_number: min={npz_trial_number.min()}  '\n"
        "                    f'max={npz_trial_number.max()}  '\n"
        "                    f'unique={len(_np.unique(npz_trial_number))}'\n"
        "                )\n"
        "            else:\n"
        "                logger.warning('NPZ missing trial_number -- coverage bonus disabled')\n"
        "                npz_trial_number = _np.zeros(len(seeds_to_score), dtype=_np.int32)"
    )
    NEW_LOAD = (
        "            if 'trial_number' in survivors:\n"
        "                npz_trial_number = survivors['trial_number'].astype(_np.int32)\n"
        "                logger.info(\n"
        "                    f'NPZ trial_number: min={npz_trial_number.min()}  '\n"
        "                    f'max={npz_trial_number.max()}  '\n"
        "                    f'unique={len(_np.unique(npz_trial_number))}'\n"
        "                )\n"
        "            else:\n"
        "                logger.warning('NPZ missing trial_number -- coverage bonus disabled')\n"
        "                npz_trial_number = _np.zeros(len(seeds_to_score), dtype=_np.int32)\n"
        "\n"
        "            # v4.3: load skip_mode for enrichment objective (TB ruling S107)\n"
        "            global npz_skip_mode\n"
        "            if 'skip_mode' in survivors:\n"
        "                npz_skip_mode = survivors['skip_mode'].astype(_np.int32)\n"
        "                rare_frac = float((_np.array(npz_skip_mode) == 1).mean())\n"
        "                logger.info(\n"
        "                    f'NPZ skip_mode: unique={len(_np.unique(npz_skip_mode))}  '\n"
        "                    f'skip_mode==1 pct={rare_frac:.4f}'\n"
        "                )\n"
        "            else:\n"
        "                logger.warning('NPZ missing skip_mode -- enrichment objective disabled, using zeros')\n"
        "                npz_skip_mode = _np.zeros(len(seeds_to_score), dtype=_np.int32)"
    )
    text, ok = patch(text, OLD_LOAD, NEW_LOAD, "load_data: add skip_mode loading")
    if not ok: sys.exit(1)

    # ------------------------------------------------------------------
    # PATCH 3: Add npz_skip_mode sanity guard after existing guards
    # ------------------------------------------------------------------
    print("\n--- Patch 3: Add npz_skip_mode sanity guard ---")
    OLD_GUARD = (
        "    if npz_bidirectional_count is None or npz_trial_number is None:\n"
        "        raise RuntimeError(\n"
        "            'NPZ bidirectional_count/trial_number are None after load -- v4.2 cannot run.'\n"
        "        )"
    )
    NEW_GUARD = (
        "    if npz_bidirectional_count is None or npz_trial_number is None:\n"
        "        raise RuntimeError(\n"
        "            'NPZ bidirectional_count/trial_number are None after load -- v4.2 cannot run.'\n"
        "        )\n"
        "    if npz_skip_mode is None:\n"
        "        raise RuntimeError(\n"
        "            'NPZ skip_mode is None after load -- v4.3 enrichment objective cannot run.'\n"
        "        )"
    )
    text, ok = patch(text, OLD_GUARD, NEW_GUARD, "sanity guard: npz_skip_mode")
    if not ok: sys.exit(1)

    # ------------------------------------------------------------------
    # PATCH 4: Replace run_trial local array setup to add sm_arr
    # ------------------------------------------------------------------
    print("\n--- Patch 4: Add sm_arr local reference in run_trial ---")
    OLD_ARRAYS = (
        "        bc_arr    = npz_bidirectional_count[sample_idx]\n"
        "        ir_arr    = npz_intersection_ratio[sample_idx]\n"
        "        fwd_arr   = npz_forward_matches[sample_idx]\n"
        "        rev_arr   = npz_reverse_matches[sample_idx]\n"
        "        tn_arr    = npz_trial_number[sample_idx]"
    )
    NEW_ARRAYS = (
        "        bc_arr    = npz_bidirectional_count[sample_idx]\n"
        "        ir_arr    = npz_intersection_ratio[sample_idx]\n"
        "        fwd_arr   = npz_forward_matches[sample_idx]\n"
        "        rev_arr   = npz_reverse_matches[sample_idx]\n"
        "        tn_arr    = npz_trial_number[sample_idx]\n"
        "        sm_arr    = npz_skip_mode[sample_idx]"
    )
    text, ok = patch(text, OLD_ARRAYS, NEW_ARRAYS, "run_trial: add sm_arr (sampled case)")
    if not ok: sys.exit(1)

    # Also patch the full (no sample) case
    print("\n--- Patch 4b: Add sm_arr full case ---")
    OLD_ARRAYS2 = (
        "        bc_arr     = npz_bidirectional_count\n"
        "        ir_arr     = npz_intersection_ratio\n"
        "        fwd_arr    = npz_forward_matches\n"
        "        rev_arr    = npz_reverse_matches\n"
        "        tn_arr     = npz_trial_number"
    )
    NEW_ARRAYS2 = (
        "        bc_arr     = npz_bidirectional_count\n"
        "        ir_arr     = npz_intersection_ratio\n"
        "        fwd_arr    = npz_forward_matches\n"
        "        rev_arr    = npz_reverse_matches\n"
        "        tn_arr     = npz_trial_number\n"
        "        sm_arr     = npz_skip_mode"
    )
    text, ok = patch(text, OLD_ARRAYS, NEW_ARRAYS, "run_trial: add sm_arr")
    if not ok: sys.exit(1)

    # ------------------------------------------------------------------
    # PATCH 5: Replace objective block (bc_score → enrichment)
    # ------------------------------------------------------------------
    print("\n--- Patch 5: Replace objective block with enrichment ---")
    OLD_OBJ = (
        "    # Primary: bidirectional_count -- median (robust vs heavy tail, TB Q2)\n"
        "    bc_subset = bc_arr[mask]\n"
        "    bc_stat   = float(np.median(bc_subset))\n"
        "    bc_score  = float(np.mean(npz_bidirectional_count < bc_stat))  # global percentile\n"
        "\n"
        "    # Secondary: intersection_ratio (TB Q3, optional bonus weight=0.10)\n"
        "    ir_disabled = bool(np.all(npz_intersection_ratio == 0))\n"
        "    if ir_disabled:\n"
        "        logger.warning('intersection_ratio all zeros -- ir_score=0.0 for this trial')\n"
        "        ir_stat  = 0.0\n"
        "        ir_score = 0.0\n"
        "    else:\n"
        "        ir_subset = ir_arr[mask]\n"
        "        ir_stat   = float(np.median(ir_subset))\n"
        "        ir_score  = float(np.mean(npz_intersection_ratio < ir_stat))  # global percentile\n"
        "\n"
        "    # Balance bonus\n"
        "    fwd_mean = float(fwd_arr[mask].mean())\n"
        "    rev_mean = float(rev_arr[mask].mean())\n"
        "    bal      = float(np.clip(1.0 - abs(fwd_mean - rev_mean), 0.0, 1.0))\n"
        "\n"
        "    # Temporal coverage via trial_number\n"
        "    uniq_total = max(len(np.unique(tn_arr)), 1)\n"
        "    uniq_sel   = len(np.unique(tn_arr[mask]))\n"
        "    coverage   = uniq_sel / uniq_total\n"
        "    tw_weight  = float(np.clip(tw_size / 1000.0, 0.05, 0.20))\n"
        "\n"
        "    # Size penalty, capped\n"
        "    size_penalty = min(\n"
        "        abs(math.log((keep + EPS) / TARGET_KEEP)),\n"
        "        SIZE_PEN_CAP\n"
        "    )\n"
        "\n"
        "    # TB v4.2 composite objective\n"
        "    objective = (\n"
        "        bc_score * (0.75 + 0.25 * bal)\n"
        "        + tw_weight * coverage\n"
        "        + IR_WEIGHT * ir_score\n"
        "        - LAMBDA_SIZE * size_penalty\n"
        "    )\n"
        "    objective = float(np.clip(objective, -1.0, 1.0))\n"
        "\n"
        "    logger.info(\n"
        "        f'Objective={objective:.6f}  bc_stat={bc_stat:.0f}  bc_score={bc_score:.4f}  '\n"
        "        f'ir_stat={ir_stat:.4f}  ir_score={ir_score:.4f}  '\n"
        "        f'bal={bal:.4f}  coverage={coverage:.4f}  tw_weight={tw_weight:.3f}  '\n"
        "        f'size_pen={size_penalty:.4f}'\n"
        "    )\n"
        "\n"
        "    _log_trial_metrics(\n"
        "        trial_num, params, subset_n, keep,\n"
        "        bc_stat=bc_stat, bc_score=bc_score,\n"
        "        ir_stat=ir_stat, ir_score=ir_score,\n"
        "        fwd_mean=fwd_mean, rev_mean=rev_mean, bal=bal,\n"
        "        coverage=coverage, tw_weight=tw_weight,\n"
        "        size_penalty=size_penalty, objective=objective, reason='ok'\n"
        "    )"
    )
    NEW_OBJ = (
        "    # v4.3: Enrichment objective (TB ruling S107)\n"
        "    # bc_score (median percentile-rank) is structurally dead:\n"
        "    # 79.2% of pool at bc>=11300 => any large subset has constant median.\n"
        "    # Residue arithmetic has no structural correlation with bc tier.\n"
        "    # Enrichment measures whether the mask preferentially selects\n"
        "    # the skip_mode==1 minority island (8.1% of pool, structurally distinct).\n"
        "\n"
        "    EPS_ENRICH = 1e-6\n"
        "    is_rare_global = (sm_arr == 1).astype(np.float32)\n"
        "    is_rare_subset = (sm_arr[mask] == 1).astype(np.float32)\n"
        "    p_global = float(is_rare_global.mean()) + EPS_ENRICH\n"
        "    p_sub    = float(is_rare_subset.mean()) + EPS_ENRICH\n"
        "    enrich   = float(np.log(p_sub / p_global))  # positive=enriched, negative=depleted\n"
        "\n"
        "    # Temporal coverage via trial_number\n"
        "    uniq_total = max(len(np.unique(tn_arr)), 1)\n"
        "    uniq_sel   = len(np.unique(tn_arr[mask]))\n"
        "    coverage   = uniq_sel / uniq_total\n"
        "\n"
        "    # Size penalty, capped\n"
        "    size_penalty = min(\n"
        "        abs(math.log((keep + EPS) / TARGET_KEEP)),\n"
        "        SIZE_PEN_CAP\n"
        "    )\n"
        "\n"
        "    # TB v4.3 enrichment objective\n"
        "    objective = (\n"
        "        0.70 * float(np.tanh(enrich))\n"
        "        + 0.20 * coverage\n"
        "        - 0.10 * size_penalty\n"
        "    )\n"
        "    objective = float(np.clip(objective, -1.0, 1.0))\n"
        "\n"
        "    logger.info(\n"
        "        f'Objective={objective:.6f}  p_global={p_global:.4f}  p_sub={p_sub:.4f}  '\n"
        "        f'enrich={enrich:.4f}  coverage={coverage:.4f}  '\n"
        "        f'size_pen={size_penalty:.4f}'\n"
        "    )\n"
        "\n"
        "    _log_trial_metrics(\n"
        "        trial_num, params, subset_n, keep,\n"
        "        enrich=enrich, p_global=p_global, p_sub=p_sub,\n"
        "        coverage=coverage,\n"
        "        size_penalty=size_penalty, objective=objective, reason='ok'\n"
        "    )"
    )
    text, ok = patch(text, OLD_OBJ, NEW_OBJ, "run_trial: enrichment objective")
    if not ok: sys.exit(1)

    # ------------------------------------------------------------------
    # PATCH 6: Update _log_trial_metrics signature
    # ------------------------------------------------------------------
    print("\n--- Patch 6: Update _log_trial_metrics signature ---")
    OLD_LOG_SIG = (
        "def _log_trial_metrics(trial_num, params, subset_n, keep,\n"
        "                       bc_stat=None, bc_score=None,\n"
        "                       ir_stat=None, ir_score=None,\n"
        "                       fwd_mean=None, rev_mean=None, bal=None,\n"
        "                       coverage=None, tw_weight=None,\n"
        "                       size_penalty=None, objective=None, reason='ok'):"
    )
    NEW_LOG_SIG = (
        "def _log_trial_metrics(trial_num, params, subset_n, keep,\n"
        "                       enrich=None, p_global=None, p_sub=None,\n"
        "                       coverage=None,\n"
        "                       size_penalty=None, objective=None, reason='ok'):"
    )
    text, ok = patch(text, OLD_LOG_SIG, NEW_LOG_SIG, "_log_trial_metrics: signature")
    if not ok: sys.exit(1)

    # ------------------------------------------------------------------
    # PATCH 7: Update _log_trial_metrics docstring / body
    # ------------------------------------------------------------------
    print("\n--- Patch 7: Update _log_trial_metrics body ---")
    OLD_LOG_BODY = (
        "    Key signal: bc_score must vary across trials for real landscape.\n"
    )
    NEW_LOG_BODY = (
        "    Key signal: enrich must vary across trials for real landscape (v4.3).\n"
    )
    text, ok = patch(text, OLD_LOG_BODY, NEW_LOG_BODY, "_log_trial_metrics: docstring")
    if not ok: sys.exit(1)

    OLD_LOG_DICT = (
        "        'bc_score'    : round(bc_score, 6) if bc_score is not None else None,\n"
        "        'ir_stat'     : round(ir_stat, 6) if ir_stat is not None else None,\n"
        "        'ir_score'    : round(ir_score, 6) if ir_score is not None else None,\n"
    )
    NEW_LOG_DICT = (
        "        'enrich'      : round(enrich, 6) if enrich is not None else None,\n"
        "        'p_global'    : round(p_global, 6) if p_global is not None else None,\n"
        "        'p_sub'       : round(p_sub, 6) if p_sub is not None else None,\n"
    )
    text, ok = patch(text, OLD_LOG_DICT, NEW_LOG_DICT, "_log_trial_metrics: dict fields")
    if not ok: sys.exit(1)

    # Also remove bc_stat, fwd_mean, rev_mean, bal, tw_weight from dict
    OLD_LOG_EXTRA = (
        "        'bc_stat'     : round(bc_stat, 6) if bc_stat is not None else None,\n"
    )
    if OLD_LOG_EXTRA in text:
        text = text.replace(OLD_LOG_EXTRA, "", 1)
        print("  [OK  ] Removed bc_stat from log dict")

    for field in ["'fwd_mean'", "'rev_mean'", "'bal'", "'tw_weight'"]:
        # Remove lines containing these fields from the metrics dict
        lines = text.split('\n')
        new_lines = [l for l in lines if not (field in l and 'round(' in l)]
        if len(new_lines) < len(lines):
            text = '\n'.join(new_lines)
            print(f"  [OK  ] Removed {field} from log dict")

    # ------------------------------------------------------------------
    # PATCH 8: Update _reject call sites (already use keyword args, should be fine)
    # ------------------------------------------------------------------
    # _reject calls _log_trial_metrics with objective=-1.0, reason=reason only
    # That's compatible with new signature (all new fields have defaults)
    print("\n--- Patch 8: _reject call sites (compatibility check) ---")
    reject_ok = "_log_trial_metrics(trial_num, params, subset_n, keep,\n                           objective=-1.0, reason=reason)" in text
    check("_reject call site compatible with new signature", reject_ok)

    # ------------------------------------------------------------------
    # WRITE OUTPUT
    # ------------------------------------------------------------------
    print("\n--- Writing patched file ---")
    save(TARGET, text)

    # ------------------------------------------------------------------
    # POST-FLIGHT CHECKS
    # ------------------------------------------------------------------
    print("\n--- Post-flight checks ---")
    final = load(TARGET)
    post = []
    post.append(check("bc_score REMOVED from run_trial", "bc_score  = float" not in final))
    post.append(check("ir_score REMOVED from run_trial", "ir_score  = float" not in final))
    post.append(check("enrich present in run_trial", "enrich   = float(np.log(" in final))
    post.append(check("tanh(enrich) in objective", "np.tanh(enrich)" in final))
    post.append(check("npz_skip_mode global declared", "npz_skip_mode           = None" in final))
    post.append(check("skip_mode loaded in load_data", "npz_skip_mode = survivors['skip_mode']" in final))
    post.append(check("coverage still present", "coverage   = uniq_sel / uniq_total" in final))
    post.append(check("size_penalty still present", "size_penalty = min(" in final))

    # AST check
    try:
        ast.parse(final)
        post.append(check("AST parse", True))
    except SyntaxError as e:
        post.append(check("AST parse", False, str(e)))

    failed_post = sum(1 for r in post if not r)
    print(f"\n{'='*60}")
    if failed_post == 0:
        print(f"v4.3 patcher COMPLETE — {len(post)}/{len(post)} checks passed")
        print(f"scorer_trial_worker.py updated in place")
        print(f"Backup: {BACKUP}")
    else:
        print(f"FAILED: {failed_post}/{len(post)} post-flight checks failed")
        print(f"Restore with: cp {BACKUP} {TARGET}")
        sys.exit(1)

if __name__ == "__main__":
    main()
