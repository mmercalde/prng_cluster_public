#!/usr/bin/env python3
"""
Mutation runner for scripts/diagnostics/e2e_grader.py (TASK_COMMON rule 8).

For each mutant it proves three things and fails if any is missing:
  APPLIED  the original text occurs exactly once and the mutant file differs from the original;
  RAN      the mutated code executed (the mutant calls _MUT_HIT(id), which appends the id to a
           marker file named by E2E_MUTANT_MARKER — also from forked control workers);
  CREDITED the credited gate failed with the credited assertion tag "[A-...]" (an unrelated
           crash or a different assertion does not count as a kill).
Also runs the credited gate against the UNMUTATED grader first, which must pass.

Output: data/e2e_grader_selftest/mutants_<stamp>/ (mutant sources, markers, logs, report.json).
Run:  python3 tests/mutate_e2e_grader.py [--workers N]   (mutants run in parallel)
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GRADER = REPO / "scripts" / "diagnostics" / "e2e_grader.py"
TEST = REPO / "tests" / "test_e2e_grader.py"
OUT = REPO / "data" / "e2e_grader_selftest" / time.strftime("mutants_%Y%m%dT%H%M%S")

PRELUDE = '''import os as _mut_os
def _MUT_HIT(tag):
    with open(_mut_os.environ["E2E_MUTANT_MARKER"], "a") as _f:
        _f.write(tag + "\\n")
    return True
'''

# (id, original text, mutated text, credited gate, credited tag, what it breaks)
MUTANTS = [
    ("M1", "    distinct = tuple(sorted(set(raw_pool)))",
     "    distinct = (_MUT_HIT(\"M1\") and tuple(sorted(raw_pool)))",
     "test_dup_values", "A-K-DISTINCT", "k counts repeats instead of distinct values (R4)"),
    ("M3", "    return isinstance(x, int) and not isinstance(x, bool)",
     "    return _MUT_HIT(\"M3\") and isinstance(x, int)",
     "test_wrong_type", "A-TYPE-BOOL", "bool accepted as an integer (R6)"),
    ("M4", "        if v < 0 or v > draw_space - 1:",
     "        if _MUT_HIT(\"M4\") and (v < 0 or v > draw_space):",
     "test_out_of_range", "A-RANGE-HI", "off-by-one upper bound accepts 1000 (R6)"),
    ("M5", "    return 1 if outcome in distinct_values else 0",
     "    return 1 if (_MUT_HIT(\"M5\") and outcome and outcome in distinct_values) else 0",
     "test_draw_000", "A-ZERO-HIT", "draw 000 treated as falsy, graded a miss (E.3)"),
    ("M6", "            per_arm[aid][\"pending\"].append(chosen)",
     "            _MUT_HIT(\"M6\")",
     "test_pending", "A-PENDING", "pending target silently dropped (R16)"),
    ("M7", "            if rule == \"error\":",
     "            if _MUT_HIT(\"M7\") and rule == \"never\":",
     "test_duplicate_record", "A-DUP-RECORD", "duplicate records not rejected under rule 'error' (R15)"),
    ("M8", "        if missing:",
     "        if _MUT_HIT(\"M8\") and False:",
     "test_missing", "A-MISSING", "missing record not an error (R14)"),
    ("M9", "        if t < arm[\"upstream_unseen_from_index\"]:",
     "        if _MUT_HIT(\"M9\") and False:",
     "test_leak", "A-LEAK", "upstream-seen target graded (R2)"),
    ("M10", "        return [None if p is None else min(1.0, m * p) for p in pvals]",
     "        return [None if p is None else min(1.0, (_MUT_HIT(\"M10\") and 1) * p) for p in pvals]",
     "test_bonferroni_and_family", "A-BONF", "no correction for declared arms (R10)"),
    ("M11", "    return float(min(1.0, pmf[h:].sum()))",
     "    return float(min(1.0, (_MUT_HIT(\"M11\") and pmf[h + 1:].sum())))",
     "test_pb_exact", "A-PTAIL", "p-value uses P(X > H) instead of P(X >= H) (R8)"),
    ("M12", "    if actual != expected_sha256.lower():",
     "    if _MUT_HIT(\"M12\") and False:",
     "test_dataset_sha", "A-DSHA", "dataset sha256 binding skipped (R14)"),
    ("M13", "        pool = rng.permutation(ds)[:k].tolist()",
     "        pool = rng.permutation(ds)[:(_MUT_HIT(\"M13\") and max(1, k - 1))].tolist()",
     "test_uniform_control", "A-RCTRL-PASS", "random control pools one value short (R18)"),
    ("M14", "            if tgt.get(\"date\") != d_date or tgt.get(\"session\") != d_sess:",
     "            if _MUT_HIT(\"M14\") and False:",
     "test_stale_target", "A-STALE", "stale target identity accepted (R14)"),
    ("M15", "        if cutoff > t:",
     "        if _MUT_HIT(\"M15\") and False:",
     "test_cutoff", "A-CUTOFF", "forecast using the target or later accepted (R3)"),
    ("M16", "    ps = [r[\"k\"] / draw_space for r in graded_rows]",
     "    ps = [(_MUT_HIT(\"M16\") and 20) / draw_space for r in graded_rows]",
     "test_known_count", "A-EXPECTED", "expected hits from a fixed k, not k_t (R7)"),
    ("M17", "    if graded_rows and fcb[\"end_index\"] > min(r[\"combined_index\"] for r in graded_rows):",
     "    if _MUT_HIT(\"M17\") and False:",
     "test_frequency_control", "A-FREQ-CAUSAL", "frequency control may overlap graded draws"),
    ("M19", "    arm_results.sort(key=lambda r: (r[\"above_chance\"], r[\"arm_id\"]))",
     "    arm_results.sort(key=lambda r: (_MUT_HIT(\"M19\") and not r[\"above_chance\"], r[\"arm_id\"]))",
     "test_null_first", "A-NULL-FIRST", "positive results listed before nulls (R20)"),
    ("M20", "    if p.exists():",
     "    if _MUT_HIT(\"M20\") and False:",
     "test_outdir", "A-OUTDIR", "existing out-dir not refused (R21)"),
    ("M21", "            if not abst[\"allow_abstention\"]:",
     "            if _MUT_HIT(\"M21\") and False:",
     "test_abstention", "A-ABST-DENY", "abstention accepted when the rule forbids it (R17)"),
    ("M23", "    m = m_arms + (len(arms_decl[\"comparisons\"])",
     "    m = (_MUT_HIT(\"M23\") and m_arms) + (0 * len(arms_decl[\"comparisons\"])",
     "test_bonferroni_and_family", "A-M-ALL", "declared comparisons left out of the family (R10)"),
    ("M24", "    equal_budget = all(a[t][\"k\"] == b[t][\"k\"] for t in common)",
     "    equal_budget = _MUT_HIT(\"M24\") or True",
     "test_comparison", "A-UNEQUAL", "McNemar run on unequal budgets (R12)"),
    ("M25", "            if _overlaps(gb[\"start_index\"], gb[\"end_index\"], b0, b1):",
     "            if _MUT_HIT(\"M25\") and False:",
     "test_previously_inspected", "A-R13", "re-grading a previously inspected period allowed (R13)"),
    # ---- TASK_LOOP_GRADE (R22 development mode, R23 per-draw detail), 2026-11-23 ----
    ("M26", "                if development:             # R22: recorded, not refused; counted mode raises",
     "                if _MUT_HIT(\"M26\") or development:",
     "test_dev_mode_explicit_r13", "A-DEV-DEFAULT-R13", "R13 waived in counted mode too (R22)"),
    ("M27", "                    inspected[aid].append([b0, b1])",
     "                    _MUT_HIT(\"M27\")",
     "test_dev_mode_explicit_r13", "A-DEV-R13-RECORDED", "development overlap with an inspected block not recorded"),
    ("M28", "        if development:                     # R22: stamped into every per_draw row\n            row[\"mode\"] = \"DEVELOPMENT\"",
     "        if development:\n            _MUT_HIT(\"M28\")",
     "test_dev_mode_stamps", "A-DEV-STAMP-ROWS", "per_draw rows not stamped DEVELOPMENT"),
    ("M29", "    if mode == \"development\":\n        row[\"mode\"] = \"DEVELOPMENT\"",
     "    if mode == \"development\":\n        _MUT_HIT(\"M29\")",
     "test_dev_mode_stamps", "A-DEV-STAMP-DETAIL", "detail rows not stamped DEVELOPMENT"),
    ("M30", "        extra.append(\"DEVELOPMENT\")",
     "        _MUT_HIT(\"M30\")",
     "test_dev_mode_stamps", "A-DEV-MARKER", "DEVELOPMENT marker not listed in SHA256SUMS"),
    ("M31", "            r[\"above_chance\"] = None",
     "            _MUT_HIT(\"M31\")",
     "test_dev_never_counted_verdict", "A-DEV-NO-VERDICT-FIELD", "development result keeps a counted above_chance verdict"),
    ("M32", "            r[\"headline\"] = _dev_headline(r, len(eligible))",
     "            r[\"headline\"] = (_MUT_HIT(\"M32\") and r[\"headline\"])",
     "test_dev_never_counted_verdict", "A-DEV-NO-VERDICT-TEXT", "development headline uses counted verdict wording"),
    ("M33", "    if gb.get(\"mode\") == \"development\" and not development:",
     "    if _MUT_HIT(\"M33\") and False:",
     "test_counted_refuses_dev_block", "A-DEV-BLOCK", "a development block graded in counted mode"),
    ("M34", "    rank = (raw.index(y) + 1) if (graded and y in raw) else None",
     "    rank = (raw.index(y) + (0 if _MUT_HIT(\"M34\") else 1)) if (graded and y in raw) else None",
     "test_per_draw_detail", "A-DETAIL-RANK", "vote rank 0-based"),
    ("M35", "    rank = (raw.index(y) + 1) if (graded and y in raw) else None",
     "    rank = (_MUT_HIT(\"M35\") and (len(raw) - raw[::-1].index(y))) if (graded and y in raw) else None",
     "test_per_draw_detail", "A-DETAIL-RANK", "vote rank of the last occurrence, not the first"),
    ("M36", "\"vote_count\": raw.count(y) if graded else None",
     "\"vote_count\": (_MUT_HIT(\"M36\") and 1) if graded else None",
     "test_per_draw_detail", "A-DETAIL-COUNT", "vote count ignores repeats"),
    ("M37", "    \"per_draw_detail\": \"off\",        # R23",
     "    \"per_draw_detail\": \"on\" if _MUT_HIT(\"M37\") else \"off\",",
     "test_per_draw_detail", "A-DETAIL-OFF", "detail file written by default (default must be today's)"),
    ("M38", "    \"mode\": \"counted\",               # R22; development must be chosen explicitly",
     "    \"mode\": \"development\" if _MUT_HIT(\"M38\") else \"counted\",",
     "test_dev_mode_explicit_r13", "A-DEV-DEFAULT-R13", "development is the default (must be explicit)"),
    ("M39", "        if development:                     # R22: stamped into every per_draw row",
     "        if _MUT_HIT(\"M39\") or development:",
     "test_counted_default_equivalence", "A-EQUIV-PERDRAW", "counted per_draw rows changed (counted path weakened)"),
    # ---- TASK_LOOP_ADDENDUM_1 (R24 rank, R25 sessions), 2026-11-23 ----
    ("M40", "    return rankdata(-scores, method=\"average\"), source",
     "    return rankdata(-scores, method=(\"min\" if _MUT_HIT(\"M40\") else \"average\")), source",
     "test_rank_detail", "A-RANK-TIES", "ties take the lowest rank, not the mid-rank"),
    ("M41", "    return rankdata(-scores, method=\"average\"), source",
     "    return rankdata((scores if _MUT_HIT(\"M41\") else -scores), method=\"average\"), source",
     "test_rank_summary", "A-RANK-PLANT", "rank direction flipped (1 = bottom of the vote)"),
    ("M42", "    scores = np.zeros(draw_space, dtype=float)\n    if use_record and vr is not None:",
     "    scores = (np.arange(draw_space, dtype=float) * -1e-6 if _MUT_HIT(\"M42\") else np.zeros(draw_space, dtype=float))\n    if use_record and vr is not None:",
     "test_rank_detail", "A-RANK-POOL", "values not in the pool not tied (pool-derived ordering invents an order)"),
    ("M43", "        le += int((sims <= S + 1e-6).sum())",
     "        le += int((sims < S - 1e-6).sum()) if _MUT_HIT(\"M43\") else 0",
     "test_rank_summary", "A-RANK-P-TIES", "Monte Carlo p counts sim < obs (ties dropped)"),
    ("M44", "    var = (M * M).mean(axis=1) - mu * mu",
     "    var = (M * M).mean(axis=1) - (0 if _MUT_HIT(\"M44\") else mu * mu)",
     "test_rank_summary", "A-RANK-SD", "exact null variance of the rank wrong"),
    ("M45", "    if not use_record and rank_source == \"record_only\":",
     "    if _MUT_HIT(\"M45\") and False:",
     "test_rank_inputs", "A-RANK-RECONLY", "record_only accepts a record without a full ordering"),
    ("M46", "                or len(set(vr)) != draw_space or min(vr) < 0 or max(vr) > draw_space - 1):",
     "                or (_MUT_HIT(\"M46\") and False) or min(vr) < 0 or max(vr) > draw_space - 1):",
     "test_rank_inputs", "A-RANK-MALFORMED", "a value_ranking with repeated values accepted"),
    ("M47", "        sub = [e for e in graded if e[\"session\"] == s]",
     "        sub = [e for e in graded if _MUT_HIT(\"M47\")]",
     "test_per_session_report", "A-SESS-SPLIT", "per-session report not split by session"),
    ("M48", "    \"rank_summary\": \"off\",           # R24",
     "    \"rank_summary\": \"on\" if _MUT_HIT(\"M48\") else \"off\",",
     "test_counted_default_equivalence", "A-EQUIV-SUMMARY", "rank summary on by default (default must be today's)"),
    ("M49", "            \"p_lower_monte_carlo\": (1 + le) / (R + 1),",
     "            \"p_lower_monte_carlo\": (1 + (ge if _MUT_HIT(\"M49\") else le)) / (R + 1),",
     "test_rank_summary", "A-RANK-P", "one-sided Monte Carlo p in the wrong tail"),
    ("M50", "                e[\"rank_tie_size\"] = int((rk == rk[y]).sum())",
     "                e[\"rank_tie_size\"] = (_MUT_HIT(\"M50\") and 1)",
     "test_rank_detail", "A-RANK-TIESIZE", "tie-group size of the drawn number not reported"),
    ("M51", "        if key in seen:                                                       # R26",
     "        if _MUT_HIT(\"M51\") and False:",
     "test_repeated_draw", "A-R26-REFUSE", "repeated (date, session) accepted: one draw graded twice (R26)"),
    ("M52", "        key = (str(d[\"date\"]), str(d[\"session\"]))",
     "        key = (_MUT_HIT(\"M52\") and str(d[\"date\"]))",
     "test_repeated_draw", "A-R26-SESSION", "the two sessions of one date refused as a repeat (R26)"),
]

# ---- compare_settings.py (LL-4) mutants: (id, original, mutated, gate, tag, what) ----
CS_MUTANTS = [
    ("CM1", "        if hidden:", "        if _MUT_HIT(\"CM1\") and False:",
     "test_cs_hidden_diff", "A-CS-HIDDEN-DIFF", "undeclared run-record difference accepted (C3)"),
    ("CM2", "        if not setting_diffs:", "        if _MUT_HIT(\"CM2\") and False:",
     "test_cs_no_setting_diff", "A-CS-NO-DIFF", "variants with no setting difference accepted (C4)"),
    ("CM3", "            if bad:", "            if _MUT_HIT(\"CM3\") and False:",
     "test_cs_equal_k", "A-CS-EQUAL-K", "unequal k accepted under require_equal_k (C7)"),
    ("CM4", "            if r.get(\"arm_id\") != aid:", "            if _MUT_HIT(\"CM4\") and False:",
     "test_cs_third_arm", "A-CS-THIRD-ARM", "an undeclared third arm passed to the grader (C6)"),
    ("CM5", "\"previously_inspected_blocks\": decl[\"previously_inspected_blocks\"],",
     "\"previously_inspected_blocks\": (_MUT_HIT(\"CM5\") and []),",
     "test_cs_r13_and_development", "A-CS-R13", "inspected blocks not passed to the grader (C8)"),
    ("CM6", "    if got != expected_sha256.lower():           # C1", "    if _MUT_HIT(\"CM6\") and False:",
     "test_cs_third_arm", "A-CS-DECL-SHA", "declaration edited after declaring accepted (C1)"),
    ("CM7", "    if got != expected_sha256.lower():\n        raise CompareError(f\"grader sha256 mismatch",
     "    if _MUT_HIT(\"CM7\") and False:\n        raise CompareError(f\"grader sha256 mismatch",
     "test_cs_bindings", "A-CS-GRADER-SHA", "unpinned grader accepted (C2)"),
    ("CM8", "            if got != want:", "            if _MUT_HIT(\"CM8\") and False:",
     "test_cs_bindings", "A-CS-IDENTITY", "run-record identity mismatch accepted (C5)"),
    ("CM9", "                    if _covers(x, y) or _covers(y, x):", "                    if _MUT_HIT(\"CM9\") and False:",
     "test_cs_overlap_groups", "A-CS-OVERLAP", "setting and bookkeeping paths may overlap (C4)"),
    ("CM10", "    return path == declared or path.startswith(declared + \".\") or path.startswith(declared + \"[\")",
     "    return _MUT_HIT(\"CM10\") and path.startswith(declared)",
     "test_cs_hidden_diff", "A-CS-COVER-SEP", "a declared path covers a sibling sharing its prefix (C3)"),
    ("CM11", "        if got != want.lower():", "        if _MUT_HIT(\"CM11\") and False:",
     "test_cs_bindings", "A-CS-INPUT-SHA", "a run record changed after declaring accepted (C2)"),
]
# ---- switch_check.py (TASK_SWITCH_REGISTER + ADDENDUM_1) mutants ----
SC_MUTANTS = [
    ("SM1", "                if k in internal:\n                    continue",
     "                if _MUT_HIT(\"SM1\") and False:\n                    continue",
     "test_sc_reach", "A-SC-REACH", "_INTERNAL_ONLY_PARAMS not stripped (R2)"),
    ("SM2", "            m[norm(info[\"param_to_cli\"].get(k, k.replace(\"_\", \"-\")))] = k",
     "            m[norm(k.replace(\"_\", \"-\") if _MUT_HIT(\"SM2\") else k)] = k",
     "test_sc_reach", "A-SC-REACH", "args_map reverse lookup ignored (R2)"),
    ("SM3", "        reg = preg.get(key) if (script and script == key) else None",
     "        reg = (_MUT_HIT(\"SM3\") and preg.get(key))",
     "test_sc_registry_class", "A-SC-CLASS", "registry matched by basename, not by the registered path (R6)"),
    ("SM4", "            elif cf not in flags and _clean(sp.get(\"name\", \"\")) not in flags:",
     "            elif _MUT_HIT(\"SM4\") and False:",
     "test_sc_registry_class", "A-SC-STALE", "stale ParameterSpecs not reported (R6)"),
    ("SM5", "    blobs = read_blobs(main, commit, paths)",
     "    blobs = (_MUT_HIT(\"SM5\") and {p: (main / p).read_bytes() for p in paths if (main / p).is_file()})",
     "test_sc_commit_not_working_copy", "A-SC-COMMIT", "main tree read from the working copy, not the commit (W1)"),
    ("SM6", "                if b is not None and _same(b, r):\n                    continue",
     "                if _MUT_HIT(\"SM6\") and False:\n                    continue",
     "test_sc_worktrees", "A-SC-WT-DEDUP", "unchanged switches attributed to worktrees (W2)"),
    ("SM7", "                and not any(fnmatch.fnmatch(r[\"file\"], g) for g in rules[\"patch_exclude_globs\"])]",
     "                and (_MUT_HIT(\"SM7\") or True)]",
     "test_sc_patch", "A-SC-PATCH-EXCL", "test-harness flags put into the registry patch (R8)"),
    ("SM8", "                if prev[\"spec\"] != spec:",
     "                if _MUT_HIT(\"SM8\") and False:",
     "test_sc_patch", "A-SC-CONFLICT", "two worktrees' different specs for one name not reported (R8)"),
    ("SM9", "        if names & frozen:",
     "        if _MUT_HIT(\"SM9\") and False:",
     "test_sc_who_and_record", "A-SC-WHO", "FROZEN_PARAMETERS ignored (R4)"),
    ("SM10", "                        if x.startswith(\"--\"):\n                            argv.setdefault",
     "                        if _MUT_HIT(\"SM10\") and False:\n                            argv.setdefault",
     "test_sc_who_and_record", "A-SC-RUNREC", "argv tokens in run records not counted (R3)"),
    ("SM11", "            if cli not in have:",
     "            if _MUT_HIT(\"SM11\") and False:",
     "test_sc_reach", "A-SC-NOTACCEPTED", "manifest keys the script does not accept not flagged (R5)"),
    ("SM12", "    elif ch is not None and not (isinstance(ch, str) and ch.startswith(\"expr:\")):\n        pt = \"CHOICE\"",
     "    elif _MUT_HIT(\"SM12\") and False:\n        pt = \"CHOICE\"",
     "test_sc_would_register", "A-SC-WOULD", "choices lost in the generated ParameterSpec (R7)"),
    ("SM13", "            idx = preg[script][\"params_end_line\"] - 1",
     "            idx = preg[script][\"params_end_line\"] - (2 if _MUT_HIT(\"SM13\") else 1)",
     "test_sc_argparse", "A-SC-RC", "new specs inserted inside an existing spec: the checker must refuse (R8)"),
    ("SM14", "    via = script if script and script.endswith(\".sh\") else None",
     "    via = (_MUT_HIT(\"SM14\") and None)",
     "test_sc_reach", "A-SC-WRAPPER", "wrapper-called script never reachable (R2)"),
]
SC_MUTANTS += [
    ("SM15", "        if info.get(\"arg_style\") == \"positional\":",
     "        if _MUT_HIT(\"SM15\") and False:",
     "test_sc_positional_and_popped", "A-SC-POSITIONAL", "arg_style positional treated as named (R2)"),
    ("SM16", "    internal = set(ctx[\"internal_only\"]) | set(ctx.get(\"popped\", []))",
     "    internal = set(ctx[\"internal_only\"]) if _MUT_HIT(\"SM16\") else set()",
     "test_sc_positional_and_popped", "A-SC-POPPED", "keys WATCHER pops treated as passed (R2)"),
]
SC_MUTANTS += [
    ("SM17", "    if set(got) != want:", "    if _MUT_HIT(\"SM17\") and False:",
     "test_sc_repair_refusals", "A-SC-REPAIR-COVER", "decisions that do not cover every stale spec accepted (R9)"),
    ("SM18", "        if e.get(\"action\") != \"DROPPED\":", "        if _MUT_HIT(\"SM18\") and False:",
     "test_sc_repair_refusals", "A-SC-REPAIR-ACTION", "an unimplemented action accepted (R9)"),
    ("SM19", "        if not isinstance(e.get(\"why\"), str) or not e[\"why\"].strip():", "        if _MUT_HIT(\"SM19\") and False:",
     "test_sc_repair_refusals", "A-SC-REPAIR-WHY", "a drop without a documented why accepted (R9)"),
    ("SM20", "    if missing_flag:", "    if _MUT_HIT(\"SM20\") and False:",
     "test_sc_repair_refusals", "A-SC-REPAIR-FLAGCHECK", "a kept spec whose flag the script lacks accepted (R9)"),
    ("SM21", "        mine = [sp for sp in reg[\"specs\"] if (script, sp.get(\"name\")) in got]",
     "        mine = list(reg[\"specs\"]) if _MUT_HIT(\"SM21\") else []",
     "test_sc_repair", "A-SC-REPAIR", "good specs dropped with the stale ones (R9)"),
    ("SM22", "            drop[sp[\"line\"] - 1] = (sp[\"end_line\"] - 1,",
     "            drop[sp[\"line\"] - 1] = (sp[\"end_line\"] - (2 if _MUT_HIT(\"SM22\") else 1),",
     "test_sc_repair", "A-SC-REPAIR", "a dropped spec's span cut short (R9)"),
    ("SM23", "        if mine and len(mine) == len(reg[\"specs\"]) and decisions.get(\"drop_empty_registries\", True):",
     "        if _MUT_HIT(\"SM23\") and False:",
     "test_sc_repair", "A-SC-REPAIR-EMPTY", "an emptied registry left behind (R9)"),
]
# ---- the REAL repaired registry draft (TASK_REGISTRY_REPAIR_20261124) mutants ----
RR_MUTANTS = [
    ("RR1", '            # DROPPED 2026-11-24 (owner ruling #7): confidence_threshold --confidence-threshold (row 19 of docs/drafts/loop_v1/REGISTRY_REPAIR_20261124.md)',
     "            ParameterSpec(name=\"confidence_threshold\", cli_flag=\"--confidence-threshold\", description=str(_MUT_HIT(\"RR1\"))),",
     "test_rr_dropped", "A-RR-DROPPED", "a dropped spec re-added"),
    ("RR2", "default=200", "default=(_MUT_HIT(\"RR2\") and 20)",
     "test_rr_kept_unchanged", "A-RR-KEPT", "a kept spec's default changed"),
    ("RR3", "cli_flag=\"--max-seeds\",", "cli_flag=(\"--max-seed\" if _MUT_HIT(\"RR3\") else \"--max-seeds\"),",
     "test_rr_flags_exist", "A-RR-FLAG", "a kept spec names a flag its script does not define"),
    ("RR4", "description=\"Whether this parameter is required\"",
     "description=(\"changed\" if _MUT_HIT(\"RR4\") else \"Whether this parameter is required\")",
     "test_rr_outside_unchanged", "A-RR-OUTSIDE", "a line outside PARAMETER_REGISTRIES changed"),
    ("RR5", '            # DROPPED 2026-11-24 (owner ruling #7): epochs --epochs (row 15 of docs/drafts/loop_v1/REGISTRY_REPAIR_20261124.md)', "            *([] if _MUT_HIT(\"RR5\") else []),",
     "test_rr_dropped", "A-RR-DOC", "a drop left undocumented in the file"),
    ("RR6", '    # DROPPED 2026-11-24 (owner ruling #7): registry "scorer_meta_optimizer.py" (n_trials, threshold_min, threshold_max, k_folds, distributed): every spec stale; see docs/drafts/loop_v1/REGISTRY_REPAIR_20261124.md',
     "    \"scorer_meta_optimizer.py\": ScriptParameterRegistry(script_name=\"scorer_meta_optimizer.py\", parameters=[], description=str(_MUT_HIT(\"RR6\"))),",
     "test_rr_dropped", "A-RR-DROPPED", "the registry of a script that never existed re-added"),
]
RR_SRC = REPO / "docs" / "drafts" / "loop_v1" / "registry_repair_20261124" / "parameter_registry.py.repaired"
RR_TEST = REPO / "tests" / "test_registry_repair.py"
SC_SRC = REPO / "scripts" / "diagnostics" / "switch_check.py"
SC_TEST = REPO / "tests" / "test_switch_check.py"
CS_SRC = REPO / "scripts" / "diagnostics" / "compare_settings.py"
CS_TEST = REPO / "tests" / "test_compare_settings.py"
# target -> (source file, test file, env var naming the mutated copy)
TARGETS = {"grader": (GRADER, TEST, "E2E_GRADER_PATH"), "cs": (CS_SRC, CS_TEST, "E2E_COMPARE_PATH"),
           "sc": (SC_SRC, SC_TEST, "E2E_SWITCH_CHECK_PATH"), "rr": (RR_SRC, RR_TEST, "E2E_REPAIRED_REGISTRY")}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def run_gate(mutated_path, gate, marker, work, log, target="grader"):
    _, test, envvar = TARGETS[target]
    env = dict(os.environ, E2E_MUTANT_MARKER=str(marker), E2E_SELFTEST_DIR=str(work))
    env[envvar] = str(mutated_path)
    p = subprocess.run([sys.executable, "-u", str(test), "--only", gate], env=env,
                       capture_output=True, text=True)
    log.write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    return p.returncode, p.stdout


def one(m, src, target="grader"):
    mid, old, new, gate, tag, what = m
    d = OUT / mid
    d.mkdir(parents=True)
    rep = {"id": mid, "target": target, "gate": gate, "tag": tag, "breaks": what}
    count = src.count(old)
    rep["original_occurrences"] = count
    if count != 1:
        rep["verdict"] = "NOT APPLIED"
        return rep
    mutated = PRELUDE + src.replace(old, new)
    mpath = d / TARGETS[target][0].name
    mpath.write_text(mutated)
    rep["applied"] = sha(mutated.encode()) != sha(src.encode())
    marker = d / "marker.txt"
    rc, out = run_gate(mpath, gate, marker, d / "work", d / "gate.log", target)
    hits = marker.read_text().split() if marker.exists() else []
    rep["ran"] = mid in hits
    rep["gate_rc"] = rc
    fail_line = next((l for l in out.splitlines() if l.startswith(f"FAIL {gate}:")), "")
    rep["fail_line"] = fail_line[:300]
    rep["credited"] = f"AssertionError: [{tag}]" in fail_line
    rep["verdict"] = ("KILLED" if rep["applied"] and rep["ran"] and rep["credited"] and rc != 0
                      else "SURVIVED_OR_UNCREDITED")
    return rep


def main():
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    only_target = sys.argv[sys.argv.index("--target") + 1] if "--target" in sys.argv else None
    OUT.mkdir(parents=True)
    src = GRADER.read_text()
    cs_src = CS_SRC.read_text()
    sc_src = SC_SRC.read_text()
    rr_src = RR_SRC.read_text()
    t0 = time.time()
    baseline = {}
    for target, mlist in (("grader", MUTANTS), ("cs", CS_MUTANTS), ("sc", SC_MUTANTS), ("rr", RR_MUTANTS)):
        if only_target and target != only_target:
            continue
        for gate in sorted({m[3] for m in mlist}):
            rc, out = run_gate(TARGETS[target][0], gate, OUT / f"baseline_{gate}.marker",
                               OUT / "baseline_work", OUT / f"baseline_{gate}.log", target)
            baseline[gate] = rc
    jobs = ([(m, src, "grader") for m in MUTANTS] + [(m, cs_src, "cs") for m in CS_MUTANTS]
            + [(m, sc_src, "sc") for m in SC_MUTANTS] + [(m, rr_src, "rr") for m in RR_MUTANTS])
    jobs = [j for j in jobs if not only_target or j[2] == only_target]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        reps = list(ex.map(lambda j: one(*j), jobs))
    ok = all(v == 0 for v in baseline.values()) and all(r["verdict"] == "KILLED" for r in reps)
    report = {"grader_sha256": sha(src.encode()), "compare_settings_sha256": sha(cs_src.encode()),
              "switch_check_sha256": sha(sc_src.encode()), "repaired_registry_sha256": sha(rr_src.encode()),
              "baseline_gate_rc": baseline,
              "mutants": reps, "all_killed_and_credited": ok, "seconds": time.time() - t0,
              "workers": workers}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    for g, rc in baseline.items():
        print(f"baseline {g}: {'PASS' if rc == 0 else 'FAIL'}", flush=True)
    for r in reps:
        print(f"{r['id']:4s} {r['verdict']:22s} applied={r.get('applied')} ran={r.get('ran')} "
              f"credited={r.get('credited')} gate={r['gate']} [{r['tag']}] — {r['breaks']}", flush=True)
    print(f"{sum(r['verdict'] == 'KILLED' for r in reps)}/{len(reps)} killed and credited; "
          f"baseline gates pass: {all(v == 0 for v in baseline.values())}; "
          f"{report['seconds']:.1f}s; report {OUT / 'report.json'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
