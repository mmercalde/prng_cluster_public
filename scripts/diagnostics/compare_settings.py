#!/usr/bin/env python3
"""
compare_settings.py — setting-comparison harness, LL-4 of the Chapter 14 learning-loop plan
(~/tfm_worktrees/ch14-learn/docs/drafts/ch14_v1/PLAN_CH14_LEARNING_LOOP.md:134, contract §4
:182-210). TASK_LOOP_GRADE item 3 (logs/e2e_session/TASK_LOOP_GRADE.txt, main tree).

Status: DRAFT, uncommitted, not Beta-reviewed. LL-4 is in Beta's before-commit class (the
verifier path for counted results, plan :134). It launches nothing but the grader, on CPU.

What it does. Given two or more run records that are meant to differ ONLY in declared
setting fields, it proves that they do, checks their pools, writes one arm list with the
declared comparisons, and calls the hit-rate grader (scripts/diagnostics/e2e_grader.py)
EXACTLY ONCE. It does not re-implement any grading or statistic; the McNemar test, the
corrections, the controls and every count come from the grader's own summary.

==========================================================================================
RULES (each enforced at the function named)
==========================================================================================
 C1 Declared before grading, bound by sha256. The declaration file's sha256 is given on
    the command line; a mismatch (an edit after declaring) is refused. (check_bindings)
    [plan §4 item 4 :199-201]
 C2 The grader is pinned: its sha256 is given on the command line; a mismatch is refused.
    Every input named in the declaration (run records, pools, grade block, abstention) is
    sha256-bound; a mismatch is refused. (check_bindings)
 C3 Everything else identical, and checked. The run records are flattened to dotted paths
    (lists as [i]). Every path whose value differs between a variant and the first variant
    (or that exists in only one) must be covered by a declared `setting_fields`,
    `bookkeeping_fields` or `identity_fields` path (a path covers itself and its children).
    Any other difference is refused and listed. (diff_run_records) [plan §4 item 2 :191-196]
 C4 Each non-first variant differs from the first in at least one declared setting field;
    otherwise there is nothing to compare and the run is refused. Setting, bookkeeping and
    identity paths may not overlap. (diff_run_records, parse_declaration)
 C5 Identity fields: for each declared identity field (arm_id, config_id, pools_sha256),
    the run record's value at the declared path must equal the variant's declared value.
    (check_identity)
 C6 Only declared arms. Every pool record in a variant's pools file must carry that
    variant's arm_id and config_id; a third arm (or a foreign record) is refused here,
    before the grader is called. (check_pools)
 C7 Equal k (default `require_equal_k: true`): on every grade-block target, every variant's
    distinct k must be equal, so the grader's exact paired test runs (grader R12). A
    mismatch is refused. The check reads pools only; no outcome is read. (check_pools)
 C8 Re-using an inspected block: `previously_inspected_blocks` is written into every arm,
    so the grader refuses an overlapping grade block in counted mode (grader R13). In
    `--mode development` the grader records it instead (grader R22). (build_arms)
 C9 One grader call, one new out-dir. Inputs are staged under <out-dir>/inputs/, the
    grader writes <out-dir>/grade/, and compare_record.json binds every sha256, the full
    run-record diff, the grader argv, its exit code and its comparison results. The exit
    code is the grader's (0 ok, 2 input error, 3 control failed). A refusal by this harness
    exits 2 and writes nothing. (main)

DECLARATION (schema tfm.e2e.compare_settings.v1), paths absolute or relative to the file:
  {"schema": "tfm.e2e.compare_settings.v1", "name": str,
   "setting_fields": ["effective_config.k", ...],          # non-empty
   "bookkeeping_fields": ["timing", "memory", ...]?,        # may differ, not a setting
   "identity_fields": {"arm_id": "arm_id", "config_id": "config_id",
                       "pools_sha256": "pools_jsonl.sha256"}?,
   "variants": [{"arm_id", "config_id", "run_record", "run_record_sha256", "pools",
                 "pools_sha256", "upstream_unseen_from_index": int, "min_k"?, "max_k"?,
                 "description"?}, ...],                     # two or more
   "comparisons": [{"arm": "<arm_id>", "baseline": "<arm_id>"}],   # non-empty
   "grade_block": path, "grade_block_sha256": hex,
   "abstention": path, "abstention_sha256": hex,
   "previously_inspected_blocks": [[s, e], ...]?,
   "require_equal_k": true?, "family_alpha": 0.05?, "correction": "bonferroni"|"holm"?,
   "grader_options": {"out_of_block_records"?, "multiplicity_family"?, "control_reps"?,
                      "control_seed"?, "control_tolerance_z"?, "workers"?,
                      "descriptive_ci_level"?, "duplicate_rule"?, "rank_summary"?,
                      "rank_source"?, "rank_test_reps"?, "rank_test_seed"?,
                      "per_session_report"?}?}
"""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_DECL = "tfm.e2e.compare_settings.v1"
SCHEMA_RECORD = "tfm.e2e.compare_settings_record.v1"
VERSION = "compare_settings 0.1.0-draft (LL-4 first form)"
GRADER_OPTIONS = {"out_of_block_records": "--out-of-block-records",
                  "multiplicity_family": "--multiplicity-family",
                  "control_reps": "--control-reps", "control_seed": "--control-seed",
                  "control_tolerance_z": "--control-tolerance-z", "workers": "--workers",
                  "descriptive_ci_level": "--descriptive-ci-level",
                  "duplicate_rule": "--duplicate-rule",
                  # grader R24/R25 (TASK_LOOP_ADDENDUM_1): descriptive, never in the verdict
                  "rank_summary": "--rank-summary", "rank_source": "--rank-source",
                  "rank_test_reps": "--rank-test-reps", "rank_test_seed": "--rank-test-seed",
                  "per_session_report": "--per-session-report"}
IDENTITY_KEYS = ("arm_id", "config_id", "pools_sha256")


class CompareError(Exception):
    """Fail-closed refusal. Nothing is written."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_grader(path, expected_sha256):
    """C2: the grader is pinned by sha256, then loaded for its helpers (not re-implemented)."""
    p = Path(path).resolve()
    if not p.is_file():
        raise CompareError(f"grader not found: {p}")
    got = sha256_file(p)
    if got != expected_sha256.lower():
        raise CompareError(f"grader sha256 mismatch: pinned {expected_sha256}, got {got}")
    spec = importlib.util.spec_from_file_location("e2e_grader_pinned", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod, p, got


# --------------------------------------------------------------------------- declaration --
def _resolve(base, p):
    q = Path(p)
    return q if q.is_absolute() else (base / q)


def _covers(declared, path):
    return path == declared or path.startswith(declared + ".") or path.startswith(declared + "[")


def parse_declaration(path, expected_sha256):
    p = Path(path)
    if not p.is_file():
        raise CompareError(f"declaration not found: {p}")
    got = sha256_file(p)
    if got != expected_sha256.lower():           # C1
        raise CompareError(f"declaration sha256 mismatch: declared {expected_sha256}, got {got} "
                           f"(edited after declaring?)")
    try:
        d = json.loads(p.read_bytes())
    except json.JSONDecodeError as e:
        raise CompareError(f"declaration is not valid JSON: {e}")
    if not isinstance(d, dict) or d.get("schema") != SCHEMA_DECL:
        raise CompareError(f"declaration must be an object with schema '{SCHEMA_DECL}'")
    base = p.resolve().parent

    def strlist(key, required):
        v = d.get(key, [] if not required else None)
        if (not isinstance(v, list) or (required and not v)
                or not all(isinstance(x, str) and x for x in v)):
            raise CompareError(f"declaration: '{key}' must be a {'non-empty ' if required else ''}"
                               f"list of non-empty strings")
        return v
    setting = strlist("setting_fields", True)
    bookkeeping = strlist("bookkeeping_fields", False)
    ident = d.get("identity_fields", {})
    if (not isinstance(ident, dict) or not set(ident) <= set(IDENTITY_KEYS)
            or not all(isinstance(v, str) and v for v in ident.values())):
        raise CompareError(f"declaration: 'identity_fields' maps a subset of {list(IDENTITY_KEYS)} "
                           f"to non-empty paths")
    groups = [("setting", setting), ("bookkeeping", bookkeeping),
              ("identity", list(ident.values()))]
    for i, (na, a) in enumerate(groups):           # C4: no overlap between the three groups
        for nb, b in groups[i + 1:]:
            for x in a:
                for y in b:
                    if _covers(x, y) or _covers(y, x):
                        raise CompareError(f"declaration: {na} path {x!r} overlaps {nb} path {y!r}")
    variants = d.get("variants")
    if not isinstance(variants, list) or len(variants) < 2:
        raise CompareError("declaration: 'variants' must list two or more run records")
    seen = set()
    vs = []
    for v in variants:
        if not isinstance(v, dict):
            raise CompareError("declaration: each variant must be an object")
        for key in ("arm_id", "config_id", "run_record", "run_record_sha256", "pools", "pools_sha256"):
            if not isinstance(v.get(key), str) or not v[key]:
                raise CompareError(f"declaration: variant needs a non-empty string '{key}'")
        if v["arm_id"] in seen:
            raise CompareError(f"declaration: duplicate variant arm_id {v['arm_id']}")
        seen.add(v["arm_id"])
        u = v.get("upstream_unseen_from_index")
        if not isinstance(u, int) or isinstance(u, bool):
            raise CompareError(f"declaration: variant {v['arm_id']} needs an integer "
                               f"upstream_unseen_from_index")
        vs.append(dict(v, run_record=_resolve(base, v["run_record"]), pools=_resolve(base, v["pools"])))
    comps = d.get("comparisons")
    if not isinstance(comps, list) or not comps:
        raise CompareError("declaration: 'comparisons' must be a non-empty list")
    for c in comps:
        if (not isinstance(c, dict) or c.get("arm") not in seen or c.get("baseline") not in seen
                or c["arm"] == c["baseline"]):
            raise CompareError(f"declaration: bad comparison {c!r} (both must be declared variants)")
    for key in ("grade_block", "grade_block_sha256", "abstention", "abstention_sha256"):
        if not isinstance(d.get(key), str) or not d[key]:
            raise CompareError(f"declaration: needs a non-empty string '{key}'")
    pib = d.get("previously_inspected_blocks", [])
    if not isinstance(pib, list):
        raise CompareError("declaration: 'previously_inspected_blocks' must be a list")
    opts = d.get("grader_options", {})
    if not isinstance(opts, dict) or not set(opts) <= set(GRADER_OPTIONS):
        raise CompareError(f"declaration: 'grader_options' keys must be among {sorted(GRADER_OPTIONS)}")
    req_k = d.get("require_equal_k", True)
    if not isinstance(req_k, bool):
        raise CompareError("declaration: 'require_equal_k' must be true or false")
    return {"path": str(p.resolve()), "sha256": got, "name": str(d.get("name", "")),
            "setting_fields": setting, "bookkeeping_fields": bookkeeping, "identity_fields": ident,
            "variants": vs, "comparisons": [{"arm": c["arm"], "baseline": c["baseline"]} for c in comps],
            "grade_block": _resolve(base, d["grade_block"]), "grade_block_sha256": d["grade_block_sha256"],
            "abstention": _resolve(base, d["abstention"]), "abstention_sha256": d["abstention_sha256"],
            "previously_inspected_blocks": pib, "require_equal_k": req_k,
            "family_alpha": d.get("family_alpha", 0.05), "correction": d.get("correction", "bonferroni"),
            "grader_options": opts}


def check_bindings(decl):
    """C2: every named input is sha256-bound."""
    def bound(path, want, what):
        if not Path(path).is_file():
            raise CompareError(f"{what} not found: {path}")
        got = sha256_file(path)
        if got != want.lower():
            raise CompareError(f"{what} sha256 mismatch: declared {want}, got {got}")
    for v in decl["variants"]:
        bound(v["run_record"], v["run_record_sha256"], f"variant {v['arm_id']} run record")
        bound(v["pools"], v["pools_sha256"], f"variant {v['arm_id']} pools")
    bound(decl["grade_block"], decl["grade_block_sha256"], "grade block")
    bound(decl["abstention"], decl["abstention_sha256"], "abstention rule")


# ---------------------------------------------------------------------------- run records --
def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        if not obj and prefix:
            out[prefix] = {}
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        if not obj and prefix:
            out[prefix] = []
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


_ABSENT = "<absent>"


def _get(flat, path):
    return flat.get(path, _ABSENT)


def diff_run_records(decl, records):
    """C3, C4. records: {arm_id: parsed run record}. Returns the full diff per variant."""
    first = decl["variants"][0]["arm_id"]
    base = flatten(records[first])
    allowed_setting = decl["setting_fields"]
    allowed_other = decl["bookkeeping_fields"] + list(decl["identity_fields"].values())
    report = {}
    for v in decl["variants"][1:]:
        aid = v["arm_id"]
        other = flatten(records[aid])
        diffs = sorted(p for p in set(base) | set(other) if _get(base, p) != _get(other, p))
        hidden = [p for p in diffs if not any(_covers(f, p) for f in allowed_setting + allowed_other)]
        if hidden:
            shown = "; ".join(f"{p}: {_get(base, p)!r} vs {_get(other, p)!r}" for p in hidden[:10])
            raise CompareError(f"run records {first} and {aid} differ in {len(hidden)} undeclared "
                               f"field(s) (C3): {shown}")
        setting_diffs = [p for p in diffs if any(_covers(f, p) for f in allowed_setting)]
        if not setting_diffs:
            raise CompareError(f"run records {first} and {aid} differ in no declared setting "
                               f"field (C4): nothing to compare")
        report[aid] = {"against": first, "n_paths_compared": len(set(base) | set(other)),
                       "setting_differences": [{"path": p, first: _get(base, p), aid: _get(other, p)}
                                               for p in setting_diffs],
                       "other_declared_differences": [p for p in diffs if p not in setting_diffs],
                       "declared_setting_fields_equal": [f for f in allowed_setting
                                                         if not any(_covers(f, p) for p in setting_diffs)]}
    return report


def check_identity(decl, records):
    """C5: declared identity fields in each run record equal the variant's declared values."""
    for v in decl["variants"]:
        flat = flatten(records[v["arm_id"]])
        for key, path in decl["identity_fields"].items():
            got = _get(flat, path)
            want = v[key]
            if got != want:
                raise CompareError(f"variant {v['arm_id']}: run record {path} = {got!r}, declared "
                                   f"{key} {want!r} (C5)")


# ---------------------------------------------------------------------------------- pools --
def check_pools(G, decl, gb):
    """C6, C7. Pools only; no outcome is read."""
    ks = {}
    for v in decl["variants"]:
        aid = v["arm_id"]
        try:
            recs = G.read_pool_records(v["pools"])
        except G.GraderError as e:
            raise CompareError(f"variant {aid}: {e}")
        kv = {}
        for lineno, r in recs:
            if r.get("arm_id") != aid:
                raise CompareError(f"variant {aid} pools line {lineno}: arm_id {r.get('arm_id')!r} "
                                   f"is not this variant's declared arm (C6: undeclared arm)")
            if r.get("config_id") != v["config_id"]:
                raise CompareError(f"variant {aid} pools line {lineno}: config_id differs from the "
                                   f"declared {v['config_id']} (C6)")
            tgt = r.get("target") or {}
            t = tgt.get("combined_index")
            if (isinstance(t, int) and gb["start_index"] <= t < gb["end_index"]
                    and tgt.get("session") in gb["sessions"] and not r.get("abstain", False)):
                try:
                    distinct, _, _ = G.validate_pool_values(r.get("pool", []),
                                                            G.DEFAULTS["draw_space"],
                                                            f"variant {aid} pools line {lineno}")
                except G.GraderError as e:
                    raise CompareError(str(e))
                kv[t] = len(distinct)
        ks[aid] = kv
    if decl["require_equal_k"]:
        first = decl["variants"][0]["arm_id"]
        for v in decl["variants"][1:]:
            aid = v["arm_id"]
            common = sorted(set(ks[first]) & set(ks[aid]))
            bad = [t for t in common if ks[first][t] != ks[aid][t]]
            if bad:
                t = bad[0]
                raise CompareError(f"distinct k differs between {first} and {aid} on {len(bad)} of "
                                   f"{len(common)} grade-block targets (first: target {t}, k "
                                   f"{ks[first][t]} vs {ks[aid][t]}); require_equal_k (C7)")
    return {aid: {"targets_in_block": len(kv), "k_min": min(kv.values()) if kv else None,
                  "k_max": max(kv.values()) if kv else None} for aid, kv in ks.items()}


def build_arms(G, decl):
    """C8: previously_inspected_blocks go into every arm so the grader enforces R13."""
    arms = []
    for v in decl["variants"]:
        a = {"arm_id": v["arm_id"], "config_id": v["config_id"],
             "upstream_unseen_from_index": v["upstream_unseen_from_index"],
             "previously_inspected_blocks": decl["previously_inspected_blocks"],
             "description": v.get("description", "")}
        for key in ("min_k", "max_k"):
            if key in v:
                a[key] = v[key]
        arms.append(a)
    return {"schema": G.SCHEMA_ARMS, "arms": arms, "comparisons": decl["comparisons"],
            "family_alpha": decl["family_alpha"], "correction": decl["correction"]}


# ----------------------------------------------------------------------------------- main --
def run(args):
    G, gpath, gsha = load_grader(args.grader, args.grader_sha256)
    try:
        out = G._check_out_dir(args.out_dir)        # same rule as the grader (R21)
    except G.GraderError as e:
        raise CompareError(str(e))
    decl = parse_declaration(args.declaration, args.declaration_sha256)
    check_bindings(decl)
    records = {}
    for v in decl["variants"]:
        try:
            records[v["arm_id"]] = json.loads(Path(v["run_record"]).read_bytes())
        except json.JSONDecodeError as e:
            raise CompareError(f"variant {v['arm_id']} run record is not valid JSON: {e}")
        if not isinstance(records[v["arm_id"]], dict):
            raise CompareError(f"variant {v['arm_id']} run record must be a JSON object")
    check_identity(decl, records)
    diff = diff_run_records(decl, records)
    try:
        gb = G.parse_grade_block(json.loads(Path(decl["grade_block"]).read_bytes()))
    except (G.GraderError, json.JSONDecodeError) as e:
        raise CompareError(f"grade block: {e}")
    k_summary = check_pools(G, decl, gb)
    arms = build_arms(G, decl)

    # ---- every refusal is above this line; from here on files are written (C9) ----
    out.mkdir(parents=False, exist_ok=False)
    inputs = out / "inputs"
    inputs.mkdir()
    arms_path = inputs / "arms.json"
    arms_path.write_text(json.dumps(arms, indent=2) + "\n", encoding="ascii")
    pools_path = inputs / "pools_concat.jsonl"
    with open(pools_path, "wb") as f:
        for v in decl["variants"]:
            b = Path(v["pools"]).read_bytes()
            f.write(b if b.endswith(b"\n") else b + b"\n")
    argv = [sys.executable, "-u", str(gpath), "--pools", str(pools_path),
            "--pools-sha256", sha256_file(pools_path), "--dataset", str(args.dataset),
            "--dataset-sha256", args.dataset_sha256, "--grade-block", str(decl["grade_block"]),
            "--arms", str(arms_path), "--abstention", str(decl["abstention"]),
            "--out-dir", str(out / "grade"), "--mode", args.mode,
            "--per-draw-detail", args.per_draw_detail,
            "--run-label", f"compare_settings {decl['name']} ({args.mode})"]
    for key, flag in GRADER_OPTIONS.items():
        if key in decl["grader_options"]:
            argv += [flag, str(decl["grader_options"][key])]
    if args.no_provenance:
        argv.append("--no-provenance")
    else:
        argv += ["--repo-root", str(args.repo_root)]
    p = subprocess.run(argv, capture_output=True, text=True)      # the one grader call (C9)
    grade = None
    sp = out / "grade" / "summary.json"
    if sp.is_file():
        s = json.loads(sp.read_text())
        grade = {"summary_sha256": sha256_file(sp), "mode": s.get("mode"),
                 "controls_ok": s.get("controls_ok"), "denominators": s.get("denominators"),
                 "null_results_first": s.get("null_results_first"),
                 "above_chance": s.get("above_chance"),
                 "development_above_background": s.get("development_above_background"),
                 "comparisons": s.get("comparisons")}
    record = {"schema": SCHEMA_RECORD, "version": VERSION,
              "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "mode": args.mode.upper(), "name": decl["name"],
              "declaration": {"path": decl["path"], "sha256": decl["sha256"]},
              "grader": {"path": str(gpath), "sha256": gsha},
              "compare_settings_sha256": sha256_file(Path(__file__).resolve()),
              "variants": [{"arm_id": v["arm_id"], "config_id": v["config_id"],
                            "run_record": str(v["run_record"]), "run_record_sha256": v["run_record_sha256"],
                            "pools": str(v["pools"]), "pools_sha256": v["pools_sha256"]}
                           for v in decl["variants"]],
              "setting_fields": decl["setting_fields"], "bookkeeping_fields": decl["bookkeeping_fields"],
              "identity_fields": decl["identity_fields"], "run_record_diff": diff,
              "pool_k_check": {"require_equal_k": decl["require_equal_k"], "per_variant": k_summary},
              "inputs": {"arms.json": sha256_file(arms_path),
                         "pools_concat.jsonl": sha256_file(pools_path)},
              "grader_calls": 1, "grader_argv": argv, "grader_exit": p.returncode,
              "grader_stdout": p.stdout, "grader_stderr": p.stderr, "grade": grade,
              "claims": "None about TFM. Hit counts and comparisons exactly as the grader "
                        "reported them for the declared variants on the declared block."}
    (out / "compare_record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="ascii")
    sys.stdout.write(p.stdout)
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
    print(f"compare_settings: grader exit {p.returncode}; record {out / 'compare_record.json'}",
          flush=True)
    return p.returncode


def build_parser():
    ap = argparse.ArgumentParser(description="LL-4 setting-comparison harness (draft).")
    ap.add_argument("--declaration", required=True)
    ap.add_argument("--declaration-sha256", required=True, help="C1: sha256 of the declaration")
    ap.add_argument("--grader", required=True, help="path to e2e_grader.py")
    ap.add_argument("--grader-sha256", required=True, help="C2: pinned grader sha256")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dataset-sha256", required=True)
    ap.add_argument("--out-dir", required=True, help="NEW directory (must not exist)")
    ap.add_argument("--mode", choices=["counted", "development"], default="counted",
                    help="passed to the grader (R22); development must be chosen explicitly")
    ap.add_argument("--per-draw-detail", choices=["off", "on"], default="off")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--no-provenance", action="store_true")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except CompareError as e:
        print(f"COMPARE_SETTINGS REFUSED (nothing written): {e}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
