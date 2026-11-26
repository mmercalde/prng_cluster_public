#!/usr/bin/env python3
"""
scripts/diagnostics/e2e_pipeline_caps.py -- NON-CERTIFYING DIAGNOSTIC helper (e2e thin path).

Pipeline-switch capability check for the walk-forward driver and the Step 3 harness
(owner requirement, TASK_LAND_WF_VOTE_20261125_ADDENDUM_1.txt, 2026-11-25 06:35):
any requested setting that needs a pipeline switch the pipeline files do not have must REFUSE
before any work, naming the switch and the file that lacks it.

How it decides: it reads the SOURCE of the pipeline files the caller is about to use (Python
ast; nothing is imported or executed) and looks for the switch itself:
  LL-1  prediction_generator.py   PredictionConfig has a `schema_check` field AND from_json
                                  reads 'schema_check'; models/feature_manifest.py defines
                                  check_rows_against_schema.   Requested by a vote config whose
                                  prediction section sets schema_check to anything but 'off'.
  LL-2  full_scoring_worker.py    score_survivors takes per_seed_sieve_features and
                                  fallback_metadata_merge; survivor_scorer.py defines
                                  metadata_merge_fields.        Requested by Step 3 args
                                  --per-seed-sieve-features != off or
                                  --fallback-metadata-merge != legacy6.
  LL-3  meta_prediction_optimizer_anti_overfit.py  declares --exclude-features,
                                  --include-feature-groups and --exclude-feature-groups;
                                  models/feature_manifest.py defines resolve_step5_exclusions.
                                  Requested by any of those three flags in the Step 5 args.
No version string, no git state, no worktree name is consulted.

A requested value equal to the default ('off', 'legacy6', no flag) is NOT a switch: it is
today's behaviour, and on a pipeline without the switch it is applied by not passing it.
"""
import ast
import json
import shlex
from pathlib import Path

LL2_PARAMS = ("per_seed_sieve_features", "fallback_metadata_merge")
LL2_DEFAULTS = {"--per-seed-sieve-features": "off", "--fallback-metadata-merge": "legacy6"}
LL3_FLAGS = ("--exclude-features", "--include-feature-groups", "--exclude-feature-groups")
LL3_BOOL = "--allow-target-columns-as-features"     # store_true (lead TASK_LL_SWITCHES_ADDENDUM_2)
LL3_SIDECAR_KEYS = {"--exclude-features": "exclude_features", "--include-feature-groups": "include_groups",
                    "--exclude-feature-groups": "exclude_groups"}


class MissingSwitch(Exception):
    """A requested setting needs a pipeline switch the pipeline file does not have."""


class BadSwitchRequest(Exception):
    """A switch request that the pipeline would refuse or misread (an empty value)."""


def _tree(path: Path):
    try:
        return ast.parse(Path(path).read_text(), filename=str(path))
    except FileNotFoundError:
        return None


def _defs(tree):
    return {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))} if tree else {}


def _str_consts(node):
    return {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def capabilities(root) -> dict:
    """Per switch: present (bool), and the file(s) checked with what each lacks."""
    root = Path(root)
    fm = _defs(_tree(root / "models" / "feature_manifest.py"))
    out = {}

    # LL-1
    pg_path = root / "prediction_generator.py"
    pg = _tree(pg_path)
    d = _defs(pg)
    lack = []
    pc = d.get("PredictionConfig")
    field = pc is not None and any(isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
                                   and s.target.id == "schema_check" for s in pc.body)
    fj = None
    if pc is not None:
        fj = next((s for s in pc.body if isinstance(s, ast.FunctionDef) and s.name == "from_json"), None)
    reads = fj is not None and "schema_check" in _str_consts(fj)
    if pg is None:
        lack.append(f"{pg_path}: file missing")
    else:
        if not field:
            lack.append(f"{pg_path}: PredictionConfig has no schema_check field")
        if not reads:
            lack.append(f"{pg_path}: PredictionConfig.from_json does not read 'schema_check'")
    if "check_rows_against_schema" not in fm:
        lack.append(f"{root / 'models/feature_manifest.py'}: no check_rows_against_schema")
    out["LL-1"] = {"present": not lack, "lacks": lack, "files": [str(pg_path), str(root / "models/feature_manifest.py")]}

    # LL-2
    fsw_path = root / "full_scoring_worker.py"
    ss_path = root / "survivor_scorer.py"
    d = _defs(_tree(fsw_path))
    lack = []
    f = d.get("score_survivors")
    if f is None:
        lack.append(f"{fsw_path}: no score_survivors")
    else:
        names = [x.arg for x in f.args.args + f.args.kwonlyargs]
        for p in LL2_PARAMS:
            if p not in names:
                lack.append(f"{fsw_path}: score_survivors has no parameter {p}")
    if "metadata_merge_fields" not in _defs(_tree(ss_path)):
        lack.append(f"{ss_path}: no metadata_merge_fields")
    out["LL-2"] = {"present": not lack, "lacks": lack, "files": [str(fsw_path), str(ss_path)]}

    # LL-3
    m5_path = root / "meta_prediction_optimizer_anti_overfit.py"
    m5 = _tree(m5_path)
    lack = []
    if m5 is None:
        lack.append(f"{m5_path}: file missing")
    else:
        declared = set()
        for n in ast.walk(m5):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument"):
                declared |= {a.value for a in n.args if isinstance(a, ast.Constant) and isinstance(a.value, str)}
        for fl in LL3_FLAGS + (LL3_BOOL,):
            if fl not in declared:
                lack.append(f"{m5_path}: no {fl} option")
    if "resolve_step5_exclusions" not in fm:
        lack.append(f"{root / 'models/feature_manifest.py'}: no resolve_step5_exclusions")
    out["LL-3"] = {"present": not lack, "lacks": lack, "files": [str(m5_path), str(root / "models/feature_manifest.py")]}
    return out


def _opt_values(tokens, name):
    vals = []
    for i, t in enumerate(tokens):
        if t == name:
            vals.append(tokens[i + 1] if i + 1 < len(tokens) else "")
        elif t.startswith(name + "="):
            vals.append(t.split("=", 1)[1])
    return vals


def requested_ll2(step3_args) -> dict:
    """Non-default LL-2 values asked for in Step 3 args ({} = none)."""
    toks = shlex.split(step3_args) if isinstance(step3_args, str) else list(step3_args)
    req = {}
    for flag, dflt in LL2_DEFAULTS.items():
        vals = _opt_values(toks, flag)
        if vals and vals[-1] != dflt:
            req[flag] = vals[-1]
    return req


def _ll3_csv(v):
    """The optimizer's own split (meta_prediction_optimizer_anti_overfit._ll3_csv)."""
    return [x.strip() for x in v.split(",") if x.strip()]


def requested_ll3(step5_args) -> dict:
    """LL-3 values asked for in one argument string; the last occurrence wins, as in argparse.
    A supplied empty value ('' or ',') is refused here, never reported as applied (Beta
    X-LL-SWITCHES R1 Q2; lead TASK_LL_SWITCHES_ADDENDUM_2 Fix 2)."""
    toks = shlex.split(step5_args) if isinstance(step5_args, str) else list(step5_args)
    req = {fl: _opt_values(toks, fl)[-1] for fl in LL3_FLAGS if _opt_values(toks, fl)}
    for fl, v in req.items():
        if not _ll3_csv(v):
            raise BadSwitchRequest(f"LL-3 {fl} was given an empty value {v!r}; omit it for the default; "
                                   f"refused, nothing run")
    if LL3_BOOL in toks:
        req[LL3_BOOL] = True
    return req


def requested_ll3_by_family(step5_args, nn_args, families) -> dict:
    """[LL-3] per family: --step5-extra-args go to every family, --nn-args are appended after them
    for neural_net only (e2e_walk_forward.run_origin), so for neural_net a flag in both is the
    --nn-args value. {} when nothing is requested."""
    every, nn = requested_ll3(step5_args), requested_ll3(nn_args)
    if not every and not nn:
        return {}
    per = {f: ({**every, **nn} if f == "neural_net" else dict(every)) for f in families}
    return {"step5_extra_args (every family)": every, "nn_args (neural_net only)": nn, "per_family": per}


def expected_ll3_selection(fam_req: dict):
    """What the Step 5 sidecar's feature_schema.feature_selection.requested must hold for one
    family's request; None = no LL-3 setting, so the sidecar must carry no feature_selection."""
    if not fam_req:
        return None
    out = {k: (_ll3_csv(fam_req[fl]) if fl in fam_req else None) for fl, k in LL3_SIDECAR_KEYS.items()}
    out["allow_target_columns_as_features"] = bool(fam_req.get(LL3_BOOL, False))
    return out


def check_ll3_sidecar(sidecar_path, fam_req: dict) -> dict:
    """Compare one trained family's sidecar with the recorded request (after training)."""
    try:
        side = json.load(open(sidecar_path))
    except (OSError, ValueError) as e:
        return {"ok": False, "expected": expected_ll3_selection(fam_req), "found": f"unreadable: {e}"}
    sel = (side.get("feature_schema") or {}).get("feature_selection")
    exp = expected_ll3_selection(fam_req)
    found = None if sel is None else sel.get("requested", "missing 'requested'")
    return {"ok": found == exp, "expected": exp, "found": found,
            "trained_feature_count": (side.get("feature_schema") or {}).get("feature_count")}


def requested_ll1(vote_config_path) -> dict:
    try:
        data = json.load(open(vote_config_path))
    except (OSError, ValueError):
        return {}
    cd = data.get("prediction", data) if isinstance(data, dict) else {}
    v = cd.get("schema_check", "off") if isinstance(cd, dict) else "off"
    return {} if v in (None, "off") else {"schema_check": v}


class PassthroughRefused(Exception):
    """An extra-args token is unsafe: an option the driver/shim sets itself, an abbreviation, or
    an option the receiving parser does not declare (lead ADDENDUM_4 Fixes 2-3, Beta
    X-E2E-WF-VOTE R1 Q1/Q2)."""


def parser_options(path) -> set:
    """Every option string declared with add_argument(...) in a file, read as source (ast)."""
    t = _tree(path)
    if t is None:
        raise FileNotFoundError(path)
    opts = set()
    for n in ast.walk(t):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument":
            opts |= {a.value for a in n.args if isinstance(a, ast.Constant) and isinstance(a.value, str)
                     and a.value.startswith("-")}
    return opts


def check_passthrough(tokens, known: set, driver_set: set, where: str, mutant=None) -> None:
    """Refuse, naming the token, any option in `tokens` that
      (1) the driver or shim sets itself, in '--opt VALUE' or '--opt=VALUE' form;
      (2) is not declared by the receiving parser but is a prefix of one (an abbreviation,
          which argparse would accept by default and resolve silently);
      (3) is not declared at all (the receiving parser would fail late, after writes).
    Values (tokens not starting with '-', or negative numbers) are skipped. `mutant` exists for the
    walk-forward selftest only."""
    toks = shlex.split(tokens) if isinstance(tokens, str) else list(tokens)
    muts = set((mutant or "").split("+")) - {""}
    for t in toks:
        if not t.startswith("-") or t == "-" or t[1:2].isdigit() or t[1:2] == ".":
            continue
        name = t.split("=", 1)[0]
        if "=" in t and "passthrough_eq_form_ignored" in muts:
            continue
        if name in driver_set and "passthrough_driver_set_ignored" not in muts:
            raise PassthroughRefused(f"{where}: {t!r} sets {name}, which the driver/shim sets itself; "
                                     f"refused, nothing run")
        if name in known:
            continue
        longer = sorted(k for k in known | driver_set if k.startswith(name) and k != name)
        if longer:
            if "passthrough_abbrev_allowed" in muts:
                continue
            raise PassthroughRefused(f"{where}: {t!r} is an abbreviation of {longer}; give the full "
                                     f"option name; refused, nothing run")
        raise PassthroughRefused(f"{where}: {t!r} is not an option of the receiving parser; "
                                 f"refused, nothing run")


def check(requests: dict, roots: dict) -> dict:
    """requests: {switch: {setting: value}}; roots: {switch: repo root whose files are called}.
    Returns the applied-switch record; raises MissingSwitch naming the switch and the file."""
    rec, bad = {}, []
    for sw in ("LL-1", "LL-2", "LL-3"):
        caps = capabilities(roots[sw])[sw]
        req = requests.get(sw) or {}
        rec[sw] = {"requested": req, "pipeline_has_switch": caps["present"], "files_checked": caps["files"],
                   "applied": req if req else "default (switch not used)"}
        if req and not caps["present"]:
            bad.append(f"{sw} switch needed by {req} is missing: " + "; ".join(caps["lacks"]))
    if bad:
        raise MissingSwitch(" | ".join(bad))
    return rec
