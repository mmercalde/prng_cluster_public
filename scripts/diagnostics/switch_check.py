#!/usr/bin/env python3
"""
switch_check.py — a CHECKER of every switch in TFM against the system that is meant to know and
reach it (TASK_SWITCH_REGISTER, lead 2026-11-23 18:50, as changed by TASK_SWITCH_REGISTER_ADDENDUM_1,
19:00: "do NOT create a new, parallel register as the source of truth"; owner 18:46 and 18:50).

The source of truth stays where the repo already puts it:
  KNOW   agents/registry/parameter_registry.py  PARAMETER_REGISTRIES -> ScriptParameterRegistry ->
         ParameterSpec (name, cli_flag, param_type, min/max, choices, default) — "Enables LLM
         introspection of legal parameter bounds" (:1-9); shown to the LLM via
         agents/parameters/parameter_context.py ParameterContext.build (:177-195), which walks the
         manifest's PRIMARY action args_map and looks each flag up in the registry.
  REACH  agent_manifests/<step>.json default_params (+ args_map names): WATCHER merges only keys
         declared there (watcher_agent.py step-scoped filter; ruled "a deliberate executable-
         interface boundary", CHAPTER_14_AUDIT.md:248) and builds argv from them; "all new CLI
         params MUST appear in manifest default_params" (CHAPTER_12_WATCHER_AGENT.md:1232).
This tool only READS those and reports, per switch, which of the three it is:
  registered and reachable | registered but unreachable | not registered
and, for a switch that is not registered, the exact ParameterSpec and manifest entry that would
register it (generated from the code's own argparse call; defaults unchanged). With --patch-dir it
also writes a DRAFT patch set (never applied): a unified diff for parameter_registry.py and JSON
fragments for agent_manifests/ (HAND_LANDED_DIRS: the owner lands those by hand).

It imports nothing from the project and nothing heavy: Python sources are parsed with `ast`, shell
wrappers with one regular expression, JSON with `json`. Git is called read-only.

==========================================================================================
WHAT IS WALKED (each source is recorded with its sha256 in the output)
==========================================================================================
 W1 Main tree AT A COMMIT (default 41ad509), read with `git cat-file`, never the working copy:
    every tracked .py (argparse `add_argument` calls), the WATCHER step shell wrappers
    (`case` options), agent_manifests/*.json (default_params, args_map, parameter_bounds,
    param_docs), distributed_config.json (search_bounds, sieve_defaults,
    reverse_sieve_defaults, evaluation_thresholds), watcher_policies.json (every leaf not
    starting with "_"), config_manifests/parameter_registry.json (compared, R5).
 W2 Every worktree under --worktrees-root: only files it changes against its own HEAD
    (tracked working changes + untracked, gitignore respected) and, for a branch that is not
    an ancestor of the commit, the files its own commits change since the merge base. A switch
    is attributed to the worktree only if it is absent from (added) or different in (changed)
    the same file at the commit. Directories data/ and logs/ are skipped.
 W3 Run records on disk (not at the commit): each manifest's declared outputs under the main
    tree's working directory, plus run_record*.json under every worktree's data/ (paths with
    any --exclude-run-record-part skipped), each at most --max-record-mb.

==========================================================================================
RULES (generated columns; each names its source)
==========================================================================================
 R1 Pipeline step: STEP_SCRIPTS / STEP_MANIFESTS parsed from agents/watcher_agent.py; a .py
    named inside a step's shell wrapper gets that step; a manifest action's script gets the
    manifest's pipeline_step. Otherwise "-" (not a WATCHER step).
 R2 WATCHER can pass it today: the step's manifest default_params has a key k whose CLI name
    (actions[].args_map / top-level args_map reverse lookup, else k with "_" -> "-") equals the
    switch's long flag, and k is not in _INTERNAL_ONLY_PARAMS (watcher_agent.py). This is the
    command builder's own rule (watcher_agent.py, "Build reverse lookup from actions[0].args_map").
    For a .py called by a step's shell wrapper: reachable only if the wrapper accepts the same
    flag and the manifest passes it ("via wrapper"; the wrapper's forwarding is NOT traced).
 R3 Run record stores it: its normalised name appears as a key, or its long flag as an argv
    token, in at least one W3 file. Evidence files are listed. A record format not on disk is
    not seen.
 R4 Who may change it, first match wins:
      frozen                 name in chapter_13_acceptance.py FROZEN_PARAMETERS;
      owner only             defined in a file in GOVERNED_FILES (scripts/gate12_parity_gate.py)
                             or HAND_LANDED_FILES / HAND_LANDED_DIRS (scripts/owner_land.py) —
                             this includes every agent_manifests/ default;
      learner within bounds  the name has bounds in a bounds source (distributed_config.json
                             search_bounds, a manifest parameter_bounds, watcher_policies.json
                             parameter_bounds); proposals still need human approval (Chapter 13
                             S1; PLAN_CH14_LEARNING_LOOP LL-16 default apply_mode human_approval);
      lead                   everything else (a code setting chosen per run).
 R6 Registry status (TASK_SWITCH_REGISTER_ADDENDUM_1 item 2). "Registered" = PARAMETER_REGISTRIES
    has a ScriptParameterRegistry keyed by the defining file's basename (get_registry,
    parameter_registry.py) with a ParameterSpec whose name, normalised as get_param does
    (lstrip "-", "-" -> "_"), equals the switch's dest or long flag. "Reachable" = R2. "LLM-visible"
    = registered AND the flag is in the manifest's PRIMARY action args_map (ParameterContext.build
    reads only that one). STALE = a ParameterSpec whose cli_flag the named script does not define,
    or whose script does not exist (ALREADY RECORDED: CHAPTER_10_AUDIT.md B1).
 R7 The generated ParameterSpec: param_type BOOL for store_true/store_false, CHOICE when choices
    are given, INT/FLOAT for type=int/float, LIST for nargs, else STRING; default = the argparse
    default when it is a literal (otherwise omitted, with the expression in a comment);
    min_value/max_value only when a bounds source (R4) gives them for that name; description = help
    text. The manifest entry: default_params {key: default} and, when the key's hyphen form is not
    the flag, an args_map {flag: key} entry in the primary action. Only for WATCHER step scripts;
    any other script has no manifest, so registering it makes it KNOWN, not reachable.
 R8 Draft patch set: the switches a worktree ADDS (W2) in a file modified on or after --new-since.
    The registry diff inserts each new ParameterSpec at the end of its script's parameters list, or
    a new ScriptParameterRegistry before the closing brace of PARAMETER_REGISTRIES. Nothing
    existing is changed or removed.
 R9 Registry repair (TASK_REGISTRY_REPAIR_20261124; owner ruling #7, 2026-11-24 10:10): with
    --repair-decisions/--repair-dir, one declared decision per stale ParameterSpec (R6) is applied
    and documented; see repair_registry. Draft only; nothing is applied to the main tree.
 R5 Flags: NO_RUN_RECORD (R3 false); WATCHER_UNREACHABLE (a step switch R2 cannot pass);
    MANIFEST_KEY_NOT_ACCEPTED (a default_params key the step script has no flag for);
    MANIFEST_KEY_NOT_PASSED (a declared key WATCHER never puts on argv: arg_style "positional"
    passes only positional_args and flag_args; _INTERNAL_ONLY_PARAMS and final_params.pop keys);
    TWO_NAMES (certain: an args_map entry whose CLI name and param key differ; candidate: equal
    token sets after the declared synonym map, or identical help text, under different names);
    REGISTRY_DIFFERS (parameter_registry.json gives another default, allowed values, range or
    CLI flag than the code); REGISTRY_ONLY (the registry lists a switch its target file does
    not define); BOUNDS_DIFFER (two bounds sources disagree); FROZEN_BUT_BOUNDED (a frozen name
    that a bounds source lets a search move).
"""
import argparse
import ast
import difflib
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "tfm.switch_check.v1"
VERSION = "switch_check 0.1.0-draft"
DEFAULT_RULES = {
    "watcher_agent": "agents/watcher_agent.py",
    "owner_land": "scripts/owner_land.py",
    "frozen_source": "chapter_13_acceptance.py",
    "frozen_name": "FROZEN_PARAMETERS",
    "manifest_glob": "agent_manifests/*.json",
    "distributed_config": "distributed_config.json",
    "distributed_config_sections": ["search_bounds", "sieve_defaults", "reverse_sieve_defaults",
                                    "evaluation_thresholds"],
    "watcher_policies": "watcher_policies.json",
    "parameter_registry": "config_manifests/parameter_registry.json",   # gitignored; read from disk
    "param_registry_py": "agents/registry/parameter_registry.py",
    "skip_dirs": ["data", "logs", ".git", "__pycache__"],
    "exclude_run_record_parts": ["selftest", "cs_run_", "mutants_", "fixture"],
    # R8: files whose flags are test-harness flags, not pipeline switches; never put in the patch
    "patch_exclude_globs": ["tests/*", "*/test_*.py", "test_*.py", "*selftest*.py", "*mutant*.py", "backups/*"],
    # Abbreviations treated as the same word for the TWO_NAMES candidate check (declared here,
    # overridable with --rules; nothing else in the register is typed).
    "synonyms": {"fwd": "forward", "rev": "reverse", "thresh": "threshold", "thr": "threshold",
                 "cfg": "config", "num": "n", "dir": "directory", "lr": "learning_rate",
                 "max": "max", "min": "min", "seeds": "seed", "trials": "trial"},
}


class RegisterError(Exception):
    pass


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def norm(name):
    return str(name).strip().lstrip("-").lower().replace("-", "_")


# ---------------------------------------------------------------------------------- git --
def git(repo, *args, check=True):
    p = subprocess.run(["git", "-C", str(repo)] + list(args), capture_output=True)
    if check and p.returncode != 0:
        raise RegisterError(f"git {' '.join(args)} in {repo}: {p.stderr.decode(errors='replace').strip()}")
    return p


def ls_tree(repo, commit):
    return git(repo, "ls-tree", "-r", "--name-only", commit).stdout.decode().splitlines()


def read_blobs(repo, commit, paths):
    """`git cat-file --batch` for commit:path; returns {path: bytes}."""
    if not paths:
        return {}
    inp = "".join(f"{commit}:{p}\n" for p in paths).encode()
    p = subprocess.run(["git", "-C", str(repo), "cat-file", "--batch"], input=inp, capture_output=True)
    if p.returncode != 0:
        raise RegisterError(f"git cat-file in {repo}: {p.stderr.decode(errors='replace')}")
    out, data, i = {}, p.stdout, 0
    for path in paths:
        nl = data.index(b"\n", i)
        header = data[i:nl].split()
        i = nl + 1
        if len(header) == 2 and header[1] == b"missing":
            continue
        size = int(header[2])
        out[path] = data[i:i + size]
        i += size + 1
    return out


# ------------------------------------------------------------------------------ ast eval --
class Consts:
    """Module-level names bound to literal-ish nodes, resolved lazily and recursively."""

    def __init__(self, tree):
        self.nodes = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                self.nodes[node.targets[0].id] = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
                self.nodes[node.target.id] = node.value

    def resolve(self, node, depth=0):
        """Returns (ok, value)."""
        if node is None or depth > 6:
            return False, None
        try:
            return True, ast.literal_eval(node)
        except Exception:
            pass
        if isinstance(node, ast.Name) and node.id in self.nodes:
            return self.resolve(self.nodes[node.id], depth + 1)
        if isinstance(node, ast.Dict):
            out = {}
            for k, v in zip(node.keys, node.values):
                okk, kk = self.resolve(k, depth + 1)
                okv, vv = self.resolve(v, depth + 1)
                if not okk:
                    return False, None
                out[kk] = vv if okv else f"expr:{ast.unparse(v)}"
            return True, out
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            vals = []
            for e in node.elts:
                ok, v = self.resolve(e, depth + 1)
                if not ok:
                    return False, None
                vals.append(v)
            return True, vals
        if isinstance(node, ast.Subscript):
            ok, base = self.resolve(node.value, depth + 1)
            okk, key = self.resolve(node.slice, depth + 1)
            if ok and okk and isinstance(base, dict) and key in base:
                return True, base[key]
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in ("list", "tuple", "sorted", "set", "frozenset") and len(node.args) == 1:
                ok, v = self.resolve(node.args[0], depth + 1)
                if ok and isinstance(v, (list, tuple, set, dict)):
                    return True, sorted(v) if f.id == "sorted" else list(v)
            if (isinstance(f, ast.Attribute) and f.attr == "get" and node.args):
                ok, base = self.resolve(f.value, depth + 1)
                okk, key = self.resolve(node.args[0], depth + 1)
                if ok and okk and isinstance(base, dict) and key in base:
                    return True, base[key]
        return False, None

    def value(self, node):
        ok, v = self.resolve(node)
        if ok:
            return v
        return f"expr:{ast.unparse(node)}" if node is not None else None


def parse_argparse(src, relpath):
    """Every `<x>.add_argument(...)` call. Returns rows (without the generated columns)."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [], f"{relpath}: SyntaxError {e.msg} line {e.lineno}"
    consts = Consts(tree)
    rows = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        flags, positional = [], None
        for a in node.args:
            ok, v = consts.resolve(a)
            if ok and isinstance(v, str):
                if v.startswith("-"):
                    flags.append(v)
                elif positional is None:
                    positional = v
            else:
                flags.append(f"expr:{ast.unparse(a)}")
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        action = consts.value(kw.get("action"))
        longs = [f for f in flags if f.startswith("--")]
        if "dest" in kw:
            dest = consts.value(kw["dest"])
        elif longs:
            dest = norm(longs[0])
        elif flags and not flags[0].startswith("expr:"):
            dest = norm(flags[0])
        else:
            dest = positional
        if "default" in kw:
            default = consts.value(kw["default"])
        elif action == "store_true":
            default = False
        elif action == "store_false":
            default = True
        else:
            default = None
        choices = consts.value(kw["choices"]) if "choices" in kw else None
        if action in ("store_true", "store_false") and choices is None:
            choices = [True, False]
        typ = ast.unparse(kw["type"]) if "type" in kw else None
        hlp = consts.value(kw.get("help")) if "help" in kw else None
        rows.append({"kind": "argparse", "name": longs[0] if longs else (flags[0] if flags else positional),
                     "flags": flags, "dest": dest, "positional": positional is not None and not flags,
                     "file": relpath, "line": node.lineno, "default": default, "choices": choices,
                     "type": typ, "action": action,
                     "required": consts.value(kw["required"]) if "required" in kw else False,
                     "plain_words": hlp if isinstance(hlp, str) else None})
    return rows, None


SH_CASE = re.compile(r"^\s*((?:--?[A-Za-z0-9][\w-]*\s*\|\s*)*--[A-Za-z0-9][\w-]*)\s*\)")
SH_PY = re.compile(r"([\w./-]+\.py)\b")


def parse_shell(src, relpath):
    rows, called = [], []
    for i, line in enumerate(src.splitlines(), 1):
        m = SH_CASE.match(line)
        if m:
            flags = [f.strip() for f in m.group(1).split("|")]
            longs = [f for f in flags if f.startswith("--")]
            rows.append({"kind": "shell_case", "name": longs[0], "flags": flags, "dest": norm(longs[0]),
                         "file": relpath, "line": i, "default": None, "choices": None, "type": None,
                         "action": None, "required": False,
                         "plain_words": None})
        if not line.lstrip().startswith("#"):
            for py in SH_PY.findall(line):
                called.append(os.path.basename(py))
    return rows, sorted(set(called))


def json_line(text, key, after=0):
    """1-based line of the first `"key"` at or after character `after` (text search)."""
    i = text.find(f'"{key}"', after)
    return text.count("\n", 0, i) + 1 if i >= 0 else None


def literal_names(src, names):
    """Module-level literal values for the given names (ast)."""
    tree = ast.parse(src)
    consts = Consts(tree)
    return {n: (consts.resolve(consts.nodes[n])[1] if n in consts.nodes and consts.resolve(consts.nodes[n])[0]
                else None) for n in names}


def find_assigned_set(src, name):
    """A set/list/tuple literal assigned to `name` anywhere (e.g. inside a function)."""
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            try:
                v = ast.literal_eval(node.value)
                return sorted(v), node.lineno
            except Exception:
                continue
    return None, None


def frozen_from(src, name):
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            v = node.value
            if isinstance(v, ast.Call) and v.args:
                v = v.args[0]
            try:
                return sorted(ast.literal_eval(v)), node.lineno
            except Exception:
                return None, node.lineno
    return None, None


# --------------------------------------------------------------------------- JSON rows ---
def manifest_rows(text, relpath):
    m = json.loads(text)
    if not isinstance(m, dict):
        return [], None
    step = m.get("pipeline_step")
    rows = []
    dp_at = text.find('"default_params"')
    docs = m.get("param_docs") or {}
    pb = m.get("parameter_bounds") or {}
    for k, v in (m.get("default_params") or {}).items():
        d = docs.get(k) if isinstance(docs, dict) else None
        if isinstance(d, dict):
            d = d.get("description") or json.dumps(d)[:200]
        b = pb.get(k) if isinstance(pb, dict) else None
        rows.append({"kind": "manifest_default", "name": k, "flags": [], "dest": norm(k), "file": relpath,
                     "line": json_line(text, k, max(dp_at, 0)), "default": v, "choices": None, "type": None,
                     "action": None, "required": False,
                     "plain_words": d if isinstance(d, str) else (b.get("description") if isinstance(b, dict) else None),
                     "manifest_step": step})
    p2c = {}
    for action in m.get("actions", []) or []:
        if isinstance(action, dict):
            for cli, key in (action.get("args_map") or {}).items():
                p2c[key] = cli
    for cli, key in (m.get("args_map") or {}).items():
        p2c[key] = cli
    scripts = [a.get("script") for a in (m.get("actions") or []) if isinstance(a, dict) and a.get("script")]
    acts = [a for a in (m.get("actions") or []) if isinstance(a, dict)]
    primary_args_map = dict(acts[0].get("args_map") or {}) if acts else {}
    info = {"step": step, "default_params": list((m.get("default_params") or {}).keys()), "param_to_cli": p2c,
            "action_scripts": scripts, "parameter_bounds": pb if isinstance(pb, dict) else {},
            "primary_args_map": primary_args_map,
            "arg_style": m.get("arg_style", "named"), "positional_args": list(m.get("positional_args") or []),
            "flag_args": list(m.get("flag_args") or []),
            "outputs": [o if isinstance(o, str) else o.get("file_pattern") for o in (m.get("outputs") or [])
                        if isinstance(o, (str, dict))] + ([m["primary_output"]] if m.get("primary_output") else [])}
    return rows, info


def flatten(obj, prefix=""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).startswith("_"):
                continue
            p = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict) and v and not _is_bound(v):
                out += flatten(v, p)
            else:
                out.append((p, v))
    else:
        out.append((prefix, obj))
    return out


def _is_bound(v):
    return isinstance(v, dict) and ({"min", "max"} <= set(v) or "allowed" in v or "choices" in v)


def config_rows(text, relpath, sections, kind):
    obj = json.loads(text)
    rows, bounds = [], {}
    secs = sections if sections is not None else [None]
    for sec in secs:
        base = obj.get(sec) if sec else obj
        if base is None:
            continue
        for path, v in flatten(base):
            full = f"{sec}.{path}" if sec else path
            leaf = path.split(".")[-1]
            b = v if _is_bound(v) else None
            if b:
                bounds[norm(leaf)] = {"min": b.get("min"), "max": b.get("max"),
                                      "allowed": b.get("allowed", b.get("choices"))}
            note = None
            if isinstance(v, dict):
                note = next((v[k] for k in v if str(k).startswith("_") and isinstance(v[k], str)), None) \
                    or v.get("description")
            rows.append({"kind": kind, "name": full, "flags": [], "dest": norm(leaf), "file": relpath,
                         "line": json_line(text, leaf), "default": (v.get("default") if isinstance(v, dict) else v),
                         "choices": None, "bounds_here": b, "type": None, "action": None, "required": False,
                         "plain_words": note})
    return rows, bounds


def policy_bounds(obj):
    out = {}
    pb = obj.get("parameter_bounds") if isinstance(obj, dict) else None
    if isinstance(pb, dict):
        for path, v in flatten(pb):
            if _is_bound(v):
                out[norm(path.split(".")[-1])] = {"min": v.get("min"), "max": v.get("max"),
                                                  "allowed": v.get("allowed", v.get("choices"))}
    return out


def registry_entries(obj):
    """parameter_registry.json -> [(section, dotted name, spec, section_description)]."""
    out = []
    for sec, body in obj.items():
        if sec.startswith("_") or not isinstance(body, dict):
            continue
        desc = " ".join(str(body.get(k, "")) for k in ("_description", "_config_file"))
        for name, spec in body.items():
            if name.startswith("_") or not isinstance(spec, dict):
                continue
            if isinstance(spec.get("fields"), dict):
                for fn, fs in spec["fields"].items():
                    if isinstance(fs, dict):
                        out.append((sec, f"{name}.{fn}", fs, desc))
            else:
                out.append((sec, name, spec, desc))
    return out


# --------------------------------------------------------------------------- the walker --
def is_code_path(p, rules):
    parts = Path(p).parts
    if any(s in parts for s in rules["skip_dirs"]):
        return False
    return (p.endswith(".py") or p.endswith(".sh") or fnmatch.fnmatch(p, rules["manifest_glob"])
            or p in (rules["distributed_config"], rules["watcher_policies"], rules["parameter_registry"]))


def rows_for_file(path, data, ctx, rules):
    """Switch rows defined in one file (any tree)."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return [], f"{path}: not UTF-8"
    try:
        if path.endswith(".py"):
            return parse_argparse(text, path)
        if path.endswith(".sh"):
            if path in ctx["step_wrappers"]:
                return parse_shell(text, path)[0], None
            return [], None
        if fnmatch.fnmatch(path, rules["manifest_glob"]):
            return manifest_rows(text, path)[0], None
        if path == rules["distributed_config"]:
            return config_rows(text, path, rules["distributed_config_sections"], "config_default")[0], None
        if path == rules["watcher_policies"]:
            return config_rows(text, path, None, "policy")[0], None
    except (json.JSONDecodeError, ValueError) as e:
        return [], f"{path}: {e}"
    return [], None


def build_context(blobs, rules):
    """Governance, step maps and bounds from the main tree at the commit."""
    txt = lambda p: blobs[p].decode("utf-8") if p in blobs else None
    wa = txt(rules["watcher_agent"])
    if wa is None:
        raise RegisterError(f"{rules['watcher_agent']} not found at the commit")
    consts = literal_names(wa, ["STEP_SCRIPTS", "STEP_MANIFESTS"])
    internal, internal_line = find_assigned_set(wa, "_INTERNAL_ONLY_PARAMS")
    popped = sorted(set(re.findall(r'final_params\.pop\(\s*["\']([^"\']+)["\']', wa)))   # keys WATCHER removes
    ol = txt(rules["owner_land"])
    gov_files, hl_files, hl_dirs, gate_rel = [], [], [], None
    if ol:
        v = literal_names(ol, ["PARITY_GATE_REL", "HAND_LANDED_DIRS"])
        gate_rel, hl_dirs = v["PARITY_GATE_REL"], list(v["HAND_LANDED_DIRS"] or [])
        tree = ast.parse(ol)
        c = Consts(tree)
        ok, hf = c.resolve(c.nodes.get("HAND_LANDED_FILES"))
        hl_files = list(hf) if ok else []
        if gate_rel and txt(gate_rel):
            g = literal_names(txt(gate_rel), ["GOVERNED_FILES"])["GOVERNED_FILES"]
            gov_files = sorted(g) if isinstance(g, (list, tuple, set, dict)) else []
    fz = txt(rules["frozen_source"])
    frozen, frozen_line = frozen_from(fz, rules["frozen_name"]) if fz else (None, None)
    steps = {int(k): v for k, v in (consts["STEP_SCRIPTS"] or {}).items()}
    smans = {int(k): v for k, v in (consts["STEP_MANIFESTS"] or {}).items()}
    manifests, bounds = {}, {}
    for p in blobs:
        if fnmatch.fnmatch(p, rules["manifest_glob"]):
            try:
                _, info = manifest_rows(txt(p), p)
            except (json.JSONDecodeError, ValueError):
                continue
            if info:
                manifests[os.path.basename(p)] = info
                for k, b in info["parameter_bounds"].items():
                    if _is_bound(b):
                        bounds.setdefault(norm(k), {})[p] = {"min": b.get("min"), "max": b.get("max"),
                                                              "allowed": b.get("allowed", b.get("choices"))}
    if rules["distributed_config"] in blobs:
        _, b = config_rows(txt(rules["distributed_config"]), rules["distributed_config"], ["search_bounds"], "x")
        for k, v in b.items():
            bounds.setdefault(k, {})[rules["distributed_config"] + " search_bounds"] = v
    if rules["watcher_policies"] in blobs:
        for k, v in policy_bounds(json.loads(txt(rules["watcher_policies"]))).items():
            bounds.setdefault(k, {})[rules["watcher_policies"] + " parameter_bounds"] = v
    step_of_file, wrappers = {}, {}
    for s, script in steps.items():
        step_of_file[script] = s
        if script.endswith(".sh") and script in blobs:
            wrappers[script] = s
            for py in parse_shell(txt(script), script)[1]:
                for p in blobs:
                    if os.path.basename(p) == py and p.endswith(".py"):
                        step_of_file.setdefault(p, s)
    for mname, info in manifests.items():
        for sc in info["action_scripts"]:
            if info["step"] is not None:
                step_of_file.setdefault(sc, info["step"])
    return {"step_scripts": steps, "step_manifests": smans, "internal_only": internal or [], "popped": popped,
            "internal_only_line": internal_line, "governed_files": gov_files, "hand_landed_files": hl_files,
            "hand_landed_dirs": hl_dirs, "frozen": frozen or [], "frozen_line": frozen_line,
            "manifests": manifests, "bounds": bounds, "step_of_file": step_of_file, "step_wrappers": wrappers}


def scan_worktree(wt, commit, main_repo, rules):
    """Paths a worktree changes against its own HEAD (+ its branch commits not in `commit`)."""
    info = {"path": str(wt), "name": wt.name}
    head = git(wt, "rev-parse", "HEAD", check=False)
    if head.returncode != 0:
        info["status"] = "not a git worktree; skipped"
        return info, []
    info["head"] = head.stdout.decode().strip()
    anc = git(wt, "merge-base", "--is-ancestor", "HEAD", commit, check=False).returncode == 0
    desc = git(wt, "merge-base", "--is-ancestor", commit, "HEAD", check=False).returncode == 0
    info["relation"] = ("at the commit" if anc and desc else "behind the commit (HEAD is an ancestor)" if anc
                        else "ahead of the commit" if desc else "diverged")
    paths = set(git(wt, "diff", "--name-only", "HEAD", "--").stdout.decode().splitlines())
    paths |= set(git(wt, "ls-files", "-o", "--exclude-standard").stdout.decode().splitlines())
    if not anc:
        mb = git(wt, "merge-base", "HEAD", commit, check=False).stdout.decode().strip()
        if mb:
            paths |= set(git(wt, "diff", "--name-only", mb, "HEAD", "--").stdout.decode().splitlines())
    paths = sorted(p for p in paths if is_code_path(p, rules) and (wt / p).is_file())
    info["changed_code_paths"] = len(paths)
    return info, paths


def _key(r):
    return (r["kind"], r["name"])


def _same(a, b):
    return all(a.get(k) == b.get(k) for k in ("default", "choices", "flags", "dest"))


def collect_run_records(main_repo, wt_root, ctx, rules, max_mb):
    files = []
    for info in ctx["manifests"].values():
        for o in info["outputs"]:
            if not o:
                continue
            for p in sorted(Path(main_repo).glob(o)):
                if p.suffix == ".json":
                    files.append(p)
    if wt_root and Path(wt_root).is_dir():
        for wt in sorted(Path(wt_root).iterdir()):
            d = wt / "data"
            if not d.is_dir():
                continue
            for root, dirs, fs in os.walk(d):
                dirs[:] = [x for x in dirs if not any(s in x for s in rules["exclude_run_record_parts"])]
                for f in fs:
                    if fnmatch.fnmatch(f, "run_record*.json"):
                        p = Path(root) / f
                        if not any(s in str(p) for s in rules["exclude_run_record_parts"]):
                            files.append(p)
    keys, argv, used, skipped = {}, {}, [], []
    seen = set()
    for p in files:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        try:
            if p.stat().st_size > max_mb * 1e6:
                skipped.append({"path": str(p), "reason": f"larger than {max_mb} MB"})
                continue
            obj = json.loads(p.read_bytes())
        except (OSError, json.JSONDecodeError) as e:
            skipped.append({"path": str(p), "reason": str(e)[:120]})
            continue
        used.append(str(p))

        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    keys.setdefault(norm(k), set()).add(str(p))
                    walk(v)
            elif isinstance(o, list):
                if o and all(isinstance(x, str) for x in o) and any(x.startswith("--") for x in o):
                    for x in o:
                        if x.startswith("--"):
                            argv.setdefault(norm(x.split("=")[0]), set()).add(str(p))
                for x in o:
                    walk(x)
        walk(obj)
    return keys, argv, used, skipped


def classify(rows, ctx, rr_keys, rr_argv, rules):
    frozen = {norm(x) for x in ctx["frozen"]}
    gov = set(ctx["governed_files"]) | set(ctx["hand_landed_files"])
    syn = rules["synonyms"]
    for r in rows:
        f = r["file"]
        r["step"] = ctx["step_of_file"].get(f, r.get("manifest_step", "-") if r["kind"] == "manifest_default" else "-")
        if r["kind"] == "manifest_default":
            r["step"] = r.get("manifest_step", "-")
        longs = [x for x in r["flags"] if x.startswith("--")]
        names = {norm(r["dest"])} if r.get("dest") else set()
        names |= {norm(x) for x in longs}
        if r["kind"] != "argparse":
            names.add(norm(str(r["name"]).split(".")[-1]))
        # R4
        if names & frozen:
            r["who"], r["who_basis"] = "frozen", f"{rules['frozen_source']}:{ctx['frozen_line']} {rules['frozen_name']}"
        elif f in gov or any(f.startswith(d) for d in ctx["hand_landed_dirs"]):
            src = ("GOVERNED_FILES" if f in ctx["governed_files"] else
                   "HAND_LANDED_FILES" if f in ctx["hand_landed_files"] else "HAND_LANDED_DIRS")
            r["who"], r["who_basis"] = "owner only", f"{src} ({rules['owner_land']})"
        elif any(n in ctx["bounds"] for n in names):
            n = next(n for n in names if n in ctx["bounds"])
            r["who"] = "learner within bounds"
            r["who_basis"] = (f"bounds in {', '.join(sorted(ctx['bounds'][n]))}; proposals still need human "
                              f"approval (Ch13 S1)")
        else:
            r["who"], r["who_basis"] = "lead", "no frozen, governed or bounds entry"
        r["bounds"] = {src: b for n in names if n in ctx["bounds"] for src, b in ctx["bounds"][n].items()}
        # R3
        ev = set()
        for n in names:
            ev |= rr_keys.get(n, set()) | rr_argv.get(n, set())
        r["run_record"] = {"stored": bool(ev), "evidence": sorted(ev)[:3], "n_files": len(ev)}
        r["_names"] = names
        r["_tokens"] = tuple(sorted({syn.get(t, t) for n in names for t in n.split("_") if t}))
        r["flags_raised"] = []
        if not ev:
            r["flags_raised"].append("NO_RUN_RECORD")
        if names & frozen and r["bounds"]:
            r["flags_raised"].append("FROZEN_BUT_BOUNDED")
        bvals = {json.dumps([b.get("min"), b.get("max")]) for b in r["bounds"].values()
                 if b.get("min") is not None or b.get("max") is not None}
        if len(bvals) > 1:
            r["flags_raised"].append("BOUNDS_DIFFER")


def watcher_reach(rows, ctx):
    """R2, plus MANIFEST_KEY_NOT_ACCEPTED rows."""
    by_file = {}
    for r in rows:
        by_file.setdefault(r["file"], []).append(r)
    internal = set(ctx["internal_only"]) | set(ctx.get("popped", []))
    step_cli = {}       # step -> {cli flag: manifest key}
    not_passed = {}     # step -> keys declared but never put on argv (positional style / popped)
    for s, mname in ctx["step_manifests"].items():
        info = ctx["manifests"].get(mname)
        if not info:
            continue
        m = {}
        if info.get("arg_style") == "positional":
            # watcher_agent.py command builder, arg_style "positional": values of positional_args in
            # order, then "--<flag>" for each truthy flag_args key; nothing else is passed.
            for k in info["flag_args"]:
                if k in info["default_params"]:
                    m[norm(k.replace("_", "-"))] = k
            not_passed[s] = {k: "positional style passes only positional_args and flag_args"
                             for k in info["default_params"] if k not in info["positional_args"] and k not in info["flag_args"]}
            for k in info["positional_args"]:
                not_passed.setdefault(s, {})
        else:
            for k in info["default_params"]:
                if k in internal:
                    continue
                m[norm(info["param_to_cli"].get(k, k.replace("_", "-")))] = k
            not_passed[s] = {k: ("WATCHER removes it before building argv" if k in ctx.get("popped", [])
                                 else "internal-only (_INTERNAL_ONLY_PARAMS)") for k in info["default_params"] if k in internal}
        step_cli[s] = m
    wrapper_flags = {s: {norm(x) for r in by_file.get(sh, []) for x in r["flags"] if x.startswith("--")}
                     for sh, s in ctx["step_wrappers"].items()}
    for r in rows:
        s = r["step"]
        if r["kind"] not in ("argparse", "shell_case") or not isinstance(s, int):
            r["watcher"] = {"reachable": None, "note": "not a WATCHER step script" if r["kind"] in ("argparse", "shell_case")
                            else ("manifest default (this IS what WATCHER passes)" if r["kind"] == "manifest_default"
                                  else "config/policy file read by code, not a CLI switch")}
            continue
        script = ctx["step_scripts"].get(s)
        longs = [norm(x) for x in r["flags"] if x.startswith("--")]
        hit = next((step_cli.get(s, {}).get(x) for x in longs if x in step_cli.get(s, {})), None)
        if r["file"] == script:
            r["watcher"] = {"reachable": bool(hit), "manifest_key": hit,
                            "note": "manifest default_params key -> CLI flag (watcher_agent.py command builder)"}
        else:
            via = script if script and script.endswith(".sh") else None
            ok = bool(hit and via and any(x in wrapper_flags.get(s, set()) for x in longs))
            r["watcher"] = {"reachable": ok, "manifest_key": hit if ok else None, "via": via,
                            "note": "called by the step wrapper; reachable only if the wrapper accepts the "
                                    "flag and the manifest passes it; the wrapper's forwarding is not traced"}
        if not r["watcher"]["reachable"]:
            r["flags_raised"].append("WATCHER_UNREACHABLE")
    for s, keys in not_passed.items():
        for r in rows:
            if r["kind"] == "manifest_default" and r["step"] == s and r["name"] in keys:
                r["flags_raised"].append("MANIFEST_KEY_NOT_PASSED")
                r["watcher"]["not_passed_because"] = keys[r["name"]]
    for s, m in step_cli.items():
        script = ctx["step_scripts"].get(s)
        have = {norm(x) for r in by_file.get(script, []) for x in r["flags"] if x.startswith("--")}
        for cli, key in m.items():
            if cli not in have:
                for r in rows:
                    if r["kind"] == "manifest_default" and r["step"] == s and r["name"] == key:
                        r["flags_raised"].append("MANIFEST_KEY_NOT_ACCEPTED")
                        r["watcher"]["not_accepted_by"] = script


def two_names(rows, ctx):
    groups = []
    for mname, info in ctx["manifests"].items():
        for key, cli in info["param_to_cli"].items():
            if norm(key) != norm(cli):
                groups.append({"certainty": "certain", "basis": f"agent_manifests/{mname} args_map",
                               "names": sorted({key, cli})})
    by_tok, by_help = {}, {}
    for r in rows:
        if r["kind"] == "argparse" and r["_tokens"]:
            by_tok.setdefault(r["_tokens"], set()).add(norm(r["dest"] or r["name"]))
        h = (r.get("plain_words") or "").strip().lower()
        if r["kind"] == "argparse" and len(h) > 20:
            by_help.setdefault(h, set()).add(norm(r["dest"] or r["name"]))
    for tok, names in by_tok.items():
        if len(names) > 1:
            groups.append({"certainty": "candidate", "basis": f"same words after the synonym map {list(tok)}",
                           "names": sorted(names)})
    for h, names in by_help.items():
        if len(names) > 1:
            groups.append({"certainty": "candidate", "basis": f"identical help text: {h[:80]!r}",
                           "names": sorted(names)})
    idx = {}
    for g in groups:
        for n in g["names"]:
            idx.setdefault(norm(n), []).append(g)
    for r in rows:
        if any(n in idx for n in r["_names"]):
            r["flags_raised"].append("TWO_NAMES")
            r["two_names"] = [g for n in r["_names"] for g in idx.get(n, [])][:3]
    return groups


def registry_compare(rows, reg_obj, ctx, rules):
    diffs, only = [], []
    steps = ctx["step_scripts"]
    main_rows = [r for r in rows if r["tree"] == "main"]
    for sec, name, spec, desc in registry_entries(reg_obj):
        target = None
        m = re.search(r"([\w./-]+\.(?:py|json))", desc)
        if m:
            target = m.group(1)
        else:
            s = re.search(r"\(Step (\d+)", desc)
            if s and int(s.group(1)) in steps:
                target = steps[int(s.group(1))]
        leaf = norm(name.split(".")[-1])
        cli = spec.get("cli_flag")
        cli_long = None
        if isinstance(cli, str):
            cl = [c for c in cli.split("/") if c.startswith("--")]
            cli_long = norm(cl[0]) if cl else None
        cands = [r for r in main_rows if target and (r["file"] == target or os.path.basename(r["file"]) == target)
                 and (leaf in r["_names"] or (cli_long and cli_long in r["_names"]))]
        if not cands:
            cands = [r for r in main_rows if leaf in r["_names"] and r["kind"] == "argparse"] if not target else []
        if not cands:
            only.append({"section": sec, "name": name, "target": target, "cli_flag": cli,
                         "default": spec.get("default"),
                         "why": "no switch with this name or flag in the target file at the commit"
                                if target else "registry section names no file and no step; nothing to compare"})
            continue
        for r in cands:
            d = {}
            if "default" in spec and not (isinstance(r["default"], str) and str(r["default"]).startswith("expr:")) \
                    and spec["default"] != r["default"]:
                d["default"] = {"registry": spec["default"], "code": r["default"]}
            av = spec.get("allowed_values")
            if av is not None and r.get("choices") not in (None, [True, False]) and sorted(map(str, av)) != sorted(map(str, r["choices"])):
                d["allowed_values"] = {"registry": av, "code": r["choices"]}
            rg = spec.get("range")
            if rg and r["bounds"]:
                for src, b in r["bounds"].items():
                    if [b.get("min"), b.get("max")] != list(rg):
                        d.setdefault("range", {"registry": rg})[src] = [b.get("min"), b.get("max")]
            if cli_long and r["kind"] == "argparse" and cli_long not in {norm(x) for x in r["flags"]}:
                d["cli_flag"] = {"registry": cli, "code": r["flags"]}
            if d:
                r["flags_raised"].append("REGISTRY_DIFFERS")
                r["registry_differs"] = d
                diffs.append({"section": sec, "name": name, "file": r["file"], "line": r["line"], "differences": d})
    return diffs, only


# ----------------------------------------------------------- the existing registry (R6) --
SPEC_FIELDS = ("name", "cli_flag", "param_type", "description", "min_value", "max_value", "choices",
               "default", "required", "affects_runtime")


def _literal_or_expr(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return f"expr:{ast.unparse(node)}"


def parse_param_registry(src):
    """PARAMETER_REGISTRIES in agents/registry/parameter_registry.py, by ast.
    Returns ({script: {"line", "params_end_line", "specs": [spec dict]}}, dict_close_line)."""
    tree = ast.parse(src)
    for node in tree.body:
        tgt = node.target if isinstance(node, ast.AnnAssign) else (node.targets[0] if isinstance(node, ast.Assign) else None)
        if not (isinstance(tgt, ast.Name) and tgt.id == "PARAMETER_REGISTRIES" and isinstance(node.value, ast.Dict)):
            continue
        out = {}
        for k, v in zip(node.value.keys, node.value.values):
            if not (isinstance(k, ast.Constant) and isinstance(v, ast.Call)):
                continue
            kw = {x.arg: x.value for x in v.keywords}
            params = kw.get("parameters")
            specs = []
            if isinstance(params, ast.List):
                for c in params.elts:
                    if isinstance(c, ast.Call):
                        d = {"line": c.lineno, "end_line": c.end_lineno}
                        for x in c.keywords:
                            if x.arg == "param_type" and isinstance(x.value, ast.Attribute):
                                d["param_type"] = x.value.attr
                            else:
                                try:
                                    d[x.arg] = ast.literal_eval(x.value)
                                except Exception:
                                    d[x.arg] = f"expr:{ast.unparse(x.value)}"
                        specs.append(d)
            out[k.value] = {"line": k.lineno, "end_line": v.end_lineno,
                            "params_end_line": params.end_lineno if params is not None else v.end_lineno,
                            "description": _literal_or_expr(kw["description"]) if "description" in kw else "",
                            "specs": specs}
        return out, node.value.end_lineno
    raise RegisterError("PARAMETER_REGISTRIES not found in the registry file")


def _clean(n):
    return str(n).lstrip("-").replace("-", "_")


def registry_check(rows, preg, ctx, script_flags):
    """R6 classes per switch row; stale registry entries."""
    for r in rows:
        if r["kind"] not in ("argparse", "shell_case", "manifest_default"):
            r["registry"] = {"class": "n/a (config/policy key, not a script switch)"}
            continue
        if r["kind"] == "manifest_default":
            script = ctx["step_scripts"].get(r["step"]) if isinstance(r["step"], int) else None
            info = ctx["manifests"].get(os.path.basename(r["file"]), {})
            names = {_clean(info.get("param_to_cli", {}).get(r["name"], r["name"].replace("_", "-"))), _clean(r["name"])}
        else:
            script = r["file"]
            names = {_clean(r["dest"] or "")} | {_clean(x) for x in r["flags"] if x.startswith("--")}
        # get_registry keys are bare script names meaning the pipeline script at the repo root; a
        # file elsewhere with the same basename (e.g. backups/) is not that script.
        key = os.path.basename(script) if script else None
        reg = preg.get(key) if (script and script == key) else None
        spec = None
        if reg:
            spec = next((sp for sp in reg["specs"] if _clean(sp.get("name", "")) in names), None)
        reach = r.get("watcher", {}).get("reachable")
        if r["kind"] == "manifest_default":
            reach = not ({"MANIFEST_KEY_NOT_ACCEPTED", "MANIFEST_KEY_NOT_PASSED"} & set(r["flags_raised"]))
        cls = ("registered and reachable" if spec and reach else "registered but unreachable" if spec
               else "not registered")
        primary = set()
        if isinstance(r["step"], int):
            m = ctx["manifests"].get(ctx["step_manifests"].get(r["step"], ""), {})
            primary = {_clean(c) for c in m.get("primary_args_map", {})}
        if not reg and script and script != key and key in preg:
            cls = "not registered (same basename as registered script, different path)"
        r["registry"] = {"class": cls, "script_key": os.path.basename(script) if script else None,
                         "spec": spec, "reachable": bool(reach) if reach is not None else False,
                         "llm_visible": bool(spec and (names & primary))}
    stale = []
    for script, reg in preg.items():
        flags = script_flags.get(script)
        for sp in reg["specs"]:
            cf = _clean(sp.get("cli_flag") or sp.get("name", ""))
            if flags is None:
                stale.append({"script": script, "spec": sp.get("name"), "line": sp["line"],
                              "why": "no tracked file with this name at the commit"})
            elif cf not in flags and _clean(sp.get("name", "")) not in flags:
                stale.append({"script": script, "spec": sp.get("name"), "line": sp["line"],
                              "why": f"{sp.get('cli_flag')} is not defined by {script}"})
    return stale


def _pytxt(v):
    return repr(v)


def spec_for(r, ctx):
    """R7: the ParameterSpec (as a dict and as source text) and manifest entry for one row."""
    longs = [x for x in r["flags"] if x.startswith("--")]
    flag = longs[0] if longs else (r["flags"][0] if r["flags"] else None)
    name = _clean(r["dest"] or flag or r["name"])
    act, ch, typ = r.get("action"), r.get("choices"), (r.get("type") or "")
    if act in ("store_true", "store_false"):
        pt = "BOOL"
    elif ch is not None and not (isinstance(ch, str) and ch.startswith("expr:")):
        pt = "CHOICE"
    elif typ == "int":
        pt = "INT"
    elif typ == "float":
        pt = "FLOAT"
    else:
        pt = "STRING"
    spec = {"name": name, "cli_flag": flag or "", "param_type": pt,
            "description": (r.get("plain_words") or "").strip()}
    b = next(iter(r.get("bounds", {}).values()), None) if r.get("bounds") else None
    if b and pt in ("INT", "FLOAT"):
        if b.get("min") is not None:
            spec["min_value"] = b["min"]
        if b.get("max") is not None:
            spec["max_value"] = b["max"]
    if pt == "CHOICE":
        spec["choices"] = list(ch)
    dflt = r.get("default")
    expr = isinstance(dflt, str) and dflt.startswith("expr:")
    if not expr and dflt is not None:
        spec["default"] = dflt
    if r.get("required") is True:
        spec["required"] = True
    lines = ["            ParameterSpec("]
    for k in ("name", "cli_flag", "param_type", "description", "min_value", "max_value", "choices", "default", "required"):
        if k in spec:
            v = f"ParamType.{spec[k]}" if k == "param_type" else _pytxt(spec[k])
            lines.append(f"                {k}={v},")
    if expr:
        lines.append(f"                # default not a literal in the code: {dflt[5:]}")
    lines.append(f"                # generated by switch_check.py from {r['file']}:{r['line']} ({r['tree']})")
    lines.append("            ),")
    manifest = None
    if isinstance(r["step"], int) and r["file"] == ctx["step_scripts"].get(r["step"]):
        key = name
        manifest = {"manifest": f"agent_manifests/{ctx['step_manifests'].get(r['step'])}",
                    "default_params": {key: (None if expr else dflt)},
                    "args_map_primary_action": ({flag.lstrip('-'): key} if flag and flag.lstrip("-") != key.replace("_", "-") else {}),
                    "note": ("default is an expression in the code; a literal must be chosen before it can be declared"
                             if expr else "declares the existing default; WATCHER would then pass it explicitly")}
    return spec, "\n".join(lines), manifest


def build_patch(reg_src, preg, close_line, new_rows, ctx):
    """R8: insert generated specs; returns (patched source, list of insertions)."""
    lines = reg_src.splitlines(keepends=True)
    inserts = {}            # 0-based line index -> [text]
    done, conflicts = [], []
    by_script = {}
    for r in new_rows:
        by_script.setdefault(os.path.basename(r["file"]), []).append(r)
    for script, rs in sorted(by_script.items()):
        texts, seen = [], {}
        for r in sorted(rs, key=lambda x: (x["tree"], x["line"] or 0)):
            spec, txt, man = spec_for(r, ctx)
            if spec["name"] in seen:
                prev = seen[spec["name"]]
                if prev["spec"] != spec:
                    conflicts.append({"script": script, "name": spec["name"], "kept": prev["from"],
                                      "kept_tree": prev["tree"], "other": f"{r['file']}:{r['line']}",
                                      "other_tree": r["tree"], "kept_spec": prev["spec"], "other_spec": spec})
                continue
            seen[spec["name"]] = {"spec": spec, "from": f"{r['file']}:{r['line']}", "tree": r["tree"]}
            texts.append(txt + "\n")
            done.append({"script": script, "name": spec["name"], "from": f"{r['file']}:{r['line']}", "tree": r["tree"],
                         "spec": spec, "manifest_entry": man})
        if script in preg:
            idx = preg[script]["params_end_line"] - 1      # the line holding the list's closing "]"
            inserts.setdefault(idx, []).extend(texts)
        else:
            block = [f'    "{script}": ScriptParameterRegistry(\n', f'        script_name="{script}",\n',
                     f'        description="(generated by switch_check.py; description to be written)",\n',
                     "        parameters=[\n"] + texts + ["        ]\n", "    ),\n"]
            inserts.setdefault(close_line - 1, []).extend(block)
    out = []
    for i, ln in enumerate(lines):
        if i in inserts:
            out.extend(inserts[i])
        out.append(ln)
    return "".join(out), done, conflicts


# ----------------------------------------------------- registry repair (R9, 2026-11-24) --
def repair_registry(preg_src, preg, stale, decisions, script_flags):
    """R9 (TASK_REGISTRY_REPAIR_20261124; owner ruling #7 2026-11-24 10:10 "fix those but
    document everything"). Applies one declared decision per stale ParameterSpec. Only DROPPED is
    implemented: a spec is removed by its exact ast span and a dated comment takes its place; a
    ScriptParameterRegistry left with no spec is removed the same way (declared
    `drop_empty_registries`). Refuses unless the decisions cover exactly the stale set, every
    other spec is unchanged, and every remaining spec's flag is defined by its script."""
    entries = decisions.get("entries")
    if not isinstance(entries, list):
        raise RegisterError("repair decisions: 'entries' must be a list")
    want = {(d["script"], d["spec"]) for d in stale}
    got = {}
    for e in entries:
        k = (e.get("script"), e.get("name"))
        if k in got:
            raise RegisterError(f"repair decisions: duplicate entry {k}")
        if e.get("action") != "DROPPED":
            raise RegisterError(f"repair decisions: {k}: only action DROPPED is implemented (got {e.get('action')!r})")
        if not isinstance(e.get("why"), str) or not e["why"].strip():
            raise RegisterError(f"repair decisions: {k}: a 'why' is required (document everything)")
        got[k] = e
    if set(got) != want:
        raise RegisterError(f"repair decisions must cover exactly the stale specs: missing "
                            f"{sorted(want - set(got))}, not stale {sorted(set(got) - want)}")
    tag = decisions.get("comment_tag", "DROPPED")
    lines = preg_src.splitlines(keepends=True)
    drop = {}                              # 0-based first line -> (last line incl., comment)
    for script, reg in preg.items():
        mine = [sp for sp in reg["specs"] if (script, sp.get("name")) in got]
        if mine and len(mine) == len(reg["specs"]) and decisions.get("drop_empty_registries", True):
            names = ", ".join(sp.get("name") for sp in mine)
            drop[reg["line"] - 1] = (reg["end_line"] - 1,
                                     f'    # {tag}: registry "{script}" ({names}): every spec stale; '
                                     f'see {decisions.get("record", "the repair record")}\n')
            continue
        for sp in mine:
            e = got[(script, sp.get("name"))]
            drop[sp["line"] - 1] = (sp["end_line"] - 1,
                                    f'            # {tag}: {sp.get("name")} {sp.get("cli_flag")} '
                                    f'(row {e.get("row", "?")} of {decisions.get("record", "the repair record")})\n')
    out, i = [], 0
    while i < len(lines):
        if i in drop:
            last, comment = drop[i]
            out.append(comment)
            i = last + 1
            continue
        out.append(lines[i])
        i += 1
    repaired = "".join(out)
    try:
        rpreg, _ = parse_param_registry(repaired)
    except SyntaxError as e:
        raise RegisterError(f"repaired registry does not parse: {e.msg} line {e.lineno}")
    key = lambda sp: {f: sp.get(f) for f in SPEC_FIELDS}
    before = {(k, sp.get("name")): key(sp) for k, v in preg.items() for sp in v["specs"] if (k, sp.get("name")) not in got}
    after = {(k, sp.get("name")): key(sp) for k, v in rpreg.items() for sp in v["specs"]}
    if before != after:
        raise RegisterError(f"repair changed a spec it was not told to drop: "
                            f"{sorted(set(before) ^ set(after)) or [k for k in before if before[k] != after.get(k)]}")
    missing_flag = [(k, n) for (k, n), sp in after.items()
                    if _clean(sp.get("cli_flag") or n) not in (script_flags.get(k) or set())]
    if missing_flag:
        raise RegisterError(f"after the repair these specs name a flag their script does not define: {missing_flag}")
    return repaired, {"dropped": [dict(got[k], registry_line=next(sp["line"] for sp in preg[k[0]]["specs"]
                                                                   if sp.get("name") == k[1]))
                                  for k in sorted(got)],
                      "kept": sorted(f"{k}:{n}" for k, n in after),
                      "registries_removed": sorted(script for script, reg in preg.items()
                                                   if script not in rpreg),
                      "n_specs_before": sum(len(v["specs"]) for v in preg.values()),
                      "n_specs_after": len(after)}


def walk(args, rules):
    main = Path(args.main_repo).resolve()
    commit = git(main, "rev-parse", args.commit).stdout.decode().strip()
    paths = [p for p in ls_tree(main, commit) if is_code_path(p, rules)]
    blobs = read_blobs(main, commit, paths)
    ctx = build_context(blobs, rules)
    rows, errors = [], []
    main_by_file = {}
    for p in sorted(blobs):
        rs, err = rows_for_file(p, blobs[p], ctx, rules)
        if err:
            errors.append(err)
        for r in rs:
            r["tree"], r["worktree_status"] = "main", f"at {commit[:7]}"
        rows += rs
        main_by_file[p] = {_key(r): r for r in rs}
    wts = []
    roots = sorted(Path(args.worktrees_root).iterdir()) if args.worktrees_root and Path(args.worktrees_root).is_dir() else []
    seen = set()
    for wt in roots:
        if not wt.is_dir() or wt.resolve() in seen or wt.resolve() == main:
            continue
        seen.add(wt.resolve())
        info, wpaths = scan_worktree(wt, commit, main, rules)
        added = changed = 0
        for p in wpaths:
            rs, err = rows_for_file(p, (wt / p).read_bytes(), ctx, rules)
            if err:
                errors.append(f"{wt.name}: {err}")
            base = main_by_file.get(p, {})
            mdate = datetime.fromtimestamp((wt / p).stat().st_mtime).strftime("%Y-%m-%d")
            for r in rs:
                b = base.get(_key(r))
                if b is not None and _same(b, r):
                    continue
                r["file_mtime_date"] = mdate
                r["tree"] = wt.name
                r["worktree_status"] = "added" if b is None else "changed"
                if b is not None:
                    r["main_value"] = {"default": b["default"], "choices": b.get("choices"), "line": b["line"]}
                added += b is None
                changed += b is not None
                rows.append(r)
        info.update(added=added, changed=changed)
        wts.append(info)
    rr_keys, rr_argv, rr_used, rr_skipped = collect_run_records(main, args.worktrees_root, ctx, rules, args.max_record_mb)
    classify(rows, ctx, rr_keys, rr_argv, rules)
    watcher_reach(rows, ctx)
    groups = two_names(rows, ctx)
    # parameter_registry.json is gitignored (.gitignore "*.json"), so it is read from the main
    # tree's working directory, not from the commit, and labelled so.
    reg_json_path = main / rules["parameter_registry"]
    reg_json = {"path": str(reg_json_path), "tracked_at_commit": rules["parameter_registry"] in blobs,
                "on_disk": reg_json_path.is_file()}
    if reg_json["on_disk"]:
        reg_bytes = reg_json_path.read_bytes()
        reg_json["sha256"] = sha256_bytes(reg_bytes)
        reg_diffs, reg_only = registry_compare(rows, json.loads(reg_bytes), ctx, rules)
    else:
        reg_diffs, reg_only = [], []
    # R6: the existing Pydantic registry, at the commit
    preg_src = blobs.get(rules["param_registry_py"])
    if preg_src is None:
        raise RegisterError(f"{rules['param_registry_py']} not found at the commit")
    preg_src = preg_src.decode("utf-8")
    preg, close_line = parse_param_registry(preg_src)
    script_flags = {}
    for script in preg:
        exact = [r for r in rows if r["tree"] == "main" and r["file"] == script and r["kind"] == "argparse"]
        anyb = [r for r in rows if r["tree"] == "main" and os.path.basename(r["file"]) == script and r["kind"] == "argparse"]
        use = exact or anyb
        if use or script in blobs:
            script_flags[script] = ({_clean(x) for r in use for x in r["flags"] if x.startswith("--")}
                                    | {_clean(r["dest"]) for r in use if r.get("dest")})
    stale = registry_check(rows, preg, ctx, script_flags)
    new_rows = [r for r in rows if r["tree"] != "main" and r.get("worktree_status") == "added"
                and r["kind"] == "argparse" and r.get("file_mtime_date", "") >= args.new_since
                and not r.get("positional") and r["registry"]["class"] == "not registered"
                and not any(fnmatch.fnmatch(r["file"], g) for g in rules["patch_exclude_globs"])]
    for r in rows:
        if r["kind"] in ("argparse", "shell_case") and r["registry"]["class"] == "not registered":
            spec, txt, man = spec_for(r, ctx)
            r["registry"]["would_register"] = {"parameter_spec": spec, "source": txt, "manifest_entry": man}
    repair = None
    if getattr(args, "repair_decisions", None):
        dec = json.loads(Path(args.repair_decisions).read_text())
        repaired, rep_log = repair_registry(preg_src, preg, stale, dec, script_flags)
        repair = {"dir": str(args.repair_dir), "decisions": {"path": str(Path(args.repair_decisions).resolve()),
                                                             "sha256": sha256_bytes(Path(args.repair_decisions).read_bytes())},
                  "repaired_source": repaired, **rep_log}
    patch = None
    if args.patch_dir:
        patched, inserted, conflicts = build_patch(preg_src, preg, close_line, new_rows, ctx)
        try:
            check_preg, _ = parse_param_registry(patched)      # the patched file must still parse
        except SyntaxError as e:
            raise RegisterError(f"patched registry does not parse: {e.msg} line {e.lineno}")
        old_specs = {(k, sp.get("name")): {f: sp.get(f) for f in SPEC_FIELDS} for k, v in preg.items() for sp in v["specs"]}
        new_specs = {(k, sp.get("name")): {f: sp.get(f) for f in SPEC_FIELDS} for k, v in check_preg.items() for sp in v["specs"]}
        unchanged = all(new_specs.get(k) == v for k, v in old_specs.items())
        added_ok = all((d["script"], d["name"]) in new_specs for d in inserted)
        if not (unchanged and added_ok):
            raise RegisterError("patched registry check failed (existing specs changed or new ones missing)")
        patch = {"dir": str(args.patch_dir), "patched_source": patched, "inserted": inserted,
                 "conflicts": conflicts,
                 "existing_specs_unchanged": unchanged, "n_existing_specs": len(old_specs)}
    for r in rows:
        r.pop("_tokens", None)
        r["_names"] = sorted(r["_names"])
        r["names_normalised"] = r.pop("_names")
    rows.sort(key=lambda r: (str(r["step"]) if r["step"] != "-" else "z", r["file"], r["line"] or 0, r["tree"]))
    for i, r in enumerate(rows, 1):
        r["id"] = f"S{i:05d}"
    counts = {"rows": len(rows), "by_kind": {}, "by_who": {}, "by_tree": {}, "flags": {}}
    for r in rows:
        for key, val in (("by_kind", r["kind"]), ("by_who", r["who"]), ("by_tree", r["tree"])):
            counts[key][val] = counts[key].get(val, 0) + 1
        for f in set(r["flags_raised"]):
            counts["flags"][f] = counts["flags"].get(f, 0) + 1
    counts["registry_only"] = len(reg_only)
    counts["registry_class"] = {}
    for r in rows:
        c = r["registry"]["class"]
        counts["registry_class"][c] = counts["registry_class"].get(c, 0) + 1
    counts["registry_class_step_scripts"] = {}
    for r in rows:
        if r["kind"] in ("argparse", "shell_case") and isinstance(r["step"], int):
            c = r["registry"]["class"]
            counts["registry_class_step_scripts"][c] = counts["registry_class_step_scripts"].get(c, 0) + 1
    counts["stale_registry_specs"] = len(stale)
    counts["new_switches_for_patch"] = len(new_rows)
    sources = {p: sha256_bytes(blobs[p]) for p in (rules["watcher_agent"], rules["owner_land"], rules["frozen_source"],
                                                   rules["distributed_config"], rules["watcher_policies"],
                                                   rules["parameter_registry"]) if p in blobs}
    return {"schema": SCHEMA, "version": VERSION,
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "main": {"repo": str(main), "commit": commit, "files_walked": len(blobs), "source_sha256": sources},
            "context": {k: ctx[k] for k in ("step_scripts", "step_manifests", "internal_only", "internal_only_line",
                                             "governed_files", "hand_landed_files", "hand_landed_dirs",
                                             "frozen", "frozen_line")},
            "rules": rules, "worktrees": wts,
            "run_records": {"files_read": rr_used, "skipped": rr_skipped,
                            "note": "read from disk at generation time, not from the commit"},
            "counts": counts, "two_name_groups": groups, "registry_differences": reg_diffs,
            "registry_only": reg_only, "parse_errors": errors,
            "parameter_registry_json": reg_json,
            "pydantic_registry": {"path": rules["param_registry_py"], "sha256": sha256_bytes(preg_src.encode()),
                                  "scripts": {k: {"line": v["line"], "n_specs": len(v["specs"])} for k, v in preg.items()},
                                  "stale_specs": stale},
            "new_since": args.new_since, "patch": patch, "repair": repair,
            "tool_sha256": sha256_bytes(Path(__file__).read_bytes()), "rows": rows}


# ------------------------------------------------------------------------------- output --
def _cell(v, n=60):
    s = "" if v is None else (v if isinstance(v, str) else json.dumps(v))
    s = s.replace("|", "\\|").replace("\n", " ")
    return s if len(s) <= n else s[:n - 1] + "…"


def render_md(reg):
    c = reg["counts"]
    pr = reg["pydantic_registry"]
    L = [f"# Switch check against the existing registry and manifests — generated {reg['generated_utc']}", "",
         "This is a CHECK, not a register. The source of truth stays `agents/registry/parameter_registry.py` "
         "(`PARAMETER_REGISTRIES`, Pydantic `ParameterSpec`) and `agent_manifests/*.json` (`default_params`, "
         "`args_map`). Regenerate; do not edit by hand.", "",
         f"Generated by `scripts/diagnostics/switch_check.py` (sha256 `{reg['tool_sha256']}`). Main tree "
         f"`{reg['main']['repo']}` at `{reg['main']['commit']}` ({reg['main']['files_walked']} files walked); "
         f"{len(reg['worktrees'])} worktrees. Rules R1–R8 are in the tool's header. JSON twin: same name, `.json`.", "",
         "## Registry classes (R6)", "",
         f"- All switch rows: {json.dumps(c['registry_class'])}.",
         f"- **WATCHER step scripts only** (argparse and step shell wrappers): {json.dumps(c['registry_class_step_scripts'])}.",
         f"- The Pydantic registry `{pr['path']}` at the commit (sha256 `{pr['sha256']}`): "
         f"{sum(v['n_specs'] for v in pr['scripts'].values())} `ParameterSpec`s in {len(pr['scripts'])} scripts; "
         f"**{c['stale_registry_specs']} stale** (cli_flag not in the script, or script absent; ALREADY RECORDED "
         f"CHAPTER_10_AUDIT.md B1).",
         f"- `config_manifests/parameter_registry.json`: tracked at the commit = "
         f"{reg['parameter_registry_json']['tracked_at_commit']}; on disk = {reg['parameter_registry_json']['on_disk']} "
         f"(gitignored by `*.json`); compared from disk below.",
         f"- Draft patch set (R8): {c['new_switches_for_patch']} switches added by worktrees in files modified on/after "
         f"{reg['new_since']}" + (f"; written to `{reg['patch']['dir']}`; existing specs unchanged = "
                                   f"{reg['patch']['existing_specs_unchanged']}." if reg.get("patch") else "; no --patch-dir given."),
         "", "### WATCHER step-script switches, by class", "",
         "| Step | Switch | File:line | Tree | Class | LLM-visible | Spec line or what would register it |",
         "|---|---|---|---|---|---|---|"]
    for r in reg["rows"]:
        if r["kind"] in ("argparse", "shell_case") and isinstance(r["step"], int):
            g = r["registry"]
            if g.get("spec"):
                what = f"`{pr['path']}:{g['spec']['line']}` {g['spec'].get('name')}"
            else:
                wr = g.get("would_register") or {}
                sp = wr.get("parameter_spec", {})
                me = wr.get("manifest_entry") or {}
                what = (f"ParameterSpec(name={sp.get('name')!r}, cli_flag={sp.get('cli_flag')!r}, "
                        f"param_type=ParamType.{sp.get('param_type')}, default={sp.get('default')!r}) in "
                        f"`{g.get('script_key')}`" + (f"; {me.get('manifest')} default_params "
                                                      f"{json.dumps(me.get('default_params'))}" if me else ""))
            tree = r["tree"] if r["tree"] == "main" else f"{r['tree']} ({r['worktree_status']})"
            L.append(f"| {r['step']} | `{_cell(r['name'], 40)}` | `{r['file']}:{r['line']}` | {tree} | {g['class']} | "
                     f"{'yes' if g.get('llm_visible') else 'no'} | {_cell(what, 170)} |")
    L += ["", "### Stale ParameterSpecs (the LLM is shown these as legal bounds)", "",
          "| Script key | Spec | Registry line | Why |", "|---|---|---|---|"]
    for d in pr["stale_specs"]:
        L.append(f"| {d['script']} | {d['spec']} | `{pr['path']}:{d['line']}` | {d['why']} |")
    if reg.get("patch"):
        L += ["", "### Draft patch set: what it registers", "",
              "| Script key | Spec name | From | Tree | Manifest fragment |", "|---|---|---|---|---|"]
        for d in reg["patch"]["inserted"]:
            m = d.get("manifest_entry")
            L.append(f"| {d['script']} | {d['name']} | `{d['from']}` | {d['tree']} | "
                     f"{_cell(m['manifest'] + ' ' + json.dumps(m['default_params']) if m else 'none: not a WATCHER step script', 90)} |")
        L += ["", f"### Draft patch set: same switch name added by two worktrees with different specs "
                  f"({len(reg['patch']['conflicts'])}; the first by worktree name is kept, the other listed)", "",
              "| Script key | Name | Kept | Other |", "|---|---|---|---|"]
        for d in reg["patch"]["conflicts"]:
            L.append(f"| {d['script']} | {d['name']} | {d['kept_tree']} `{d['kept']}` | {d['other_tree']} `{d['other']}` |")
    L += ["", "## Other checks (secondary)", "",
         "## Counts", "",
         f"- Rows: **{c['rows']}**. By kind: {json.dumps(c['by_kind'])}.",
         f"- Who may change it: {json.dumps(c['by_who'])}.",
         f"- By tree: {json.dumps(c['by_tree'])}.",
         f"- Flags (rows carrying each): {json.dumps(c['flags'])}; registry entries with no switch: "
         f"{c['registry_only']}.", "",
         "## Flagged: parameter_registry.json lists it differently", "",
         "| Registry entry | File:line | Differences |", "|---|---|---|"]
    for d in reg["registry_differences"]:
        L.append(f"| {d['section']}.{d['name']} | `{d['file']}:{d['line']}` | {_cell(d['differences'], 160)} |")
    L += ["", "### Registry entries with no switch", "", "| Registry entry | Target | CLI flag | Why |", "|---|---|---|---|"]
    for d in reg["registry_only"]:
        L.append(f"| {d['section']}.{d['name']} | {_cell(d['target'])} | {_cell(d['cli_flag'])} | {d['why']} |")
    L += ["", "## Flagged: the same idea under two names", "", "| Certainty | Names | Basis |", "|---|---|---|"]
    for g in reg["two_name_groups"]:
        L.append(f"| {g['certainty']} | {', '.join(g['names'])} | {_cell(g['basis'], 120)} |")
    for flag, title in (("WATCHER_UNREACHABLE", "a step switch WATCHER cannot pass today"),
                        ("MANIFEST_KEY_NOT_ACCEPTED", "a manifest key the step script has no flag for"),
                        ("MANIFEST_KEY_NOT_PASSED", "a manifest key WATCHER never puts on argv"),
                        ("BOUNDS_DIFFER", "bounds sources disagree"),
                        ("FROZEN_BUT_BOUNDED", "a frozen name a bounds source lets a search move")):
        sel = [r for r in reg["rows"] if flag in r["flags_raised"]]
        L += ["", f"## Flagged: {title} ({len(sel)})", "", "| ID | Switch | File:line | Step | Note |", "|---|---|---|---|---|"]
        for r in sel:
            L.append(f"| {r['id']} | `{_cell(r['name'], 40)}` | `{r['file']}:{r['line']}` | {r['step']} | "
                     f"{_cell(r.get('watcher', {}).get('note') if flag != 'BOUNDS_DIFFER' else r['bounds'], 100)} |")
    n_nr = c["flags"].get("NO_RUN_RECORD", 0)
    L += ["", f"## Flagged: no run record stores it ({n_nr} rows)", "",
          "Every row with `NO_RUN_RECORD` is marked in the full register below; run records read: "
          f"{len(reg['run_records']['files_read'])} files (listed in the JSON).", "",
          "## Worktrees", "", "| Worktree | HEAD | Relation to the commit | Changed code paths | Switches added | Changed |",
          "|---|---|---|---|---|---|"]
    for w in reg["worktrees"]:
        L.append(f"| {w['name']} | {str(w.get('head', ''))[:7]} | {w.get('relation', w.get('status'))} | "
                 f"{w.get('changed_code_paths', '')} | {w.get('added', '')} | {w.get('changed', '')} |")
    L += ["", "## Full register", "",
          "| ID | Switch | Kind | File:line | Tree | Step | Default | Allowed / bounds | What it changes | WATCHER | Registry class | Run record | Who | Flags |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in reg["rows"]:
        allowed = r.get("choices") if r.get("choices") is not None else (r["bounds"] or None)
        w = r.get("watcher", {})
        wtxt = {True: "yes", False: "no", None: "n/a"}[w.get("reachable")]
        tree = r["tree"] if r["tree"] == "main" else f"{r['tree']} ({r['worktree_status']})"
        L.append(f"| {r['id']} | `{_cell(r['name'], 40)}` | {r['kind']} | `{r['file']}:{r['line']}` | {tree} | "
                 f"{r['step']} | {_cell(r['default'], 30)} | {_cell(allowed, 40)} | {_cell(r.get('plain_words'), 70)} | "
                 f"{wtxt} | {r['registry']['class']} | {'yes' if r['run_record']['stored'] else 'no'} | {r['who']} | "
                 f"{', '.join(sorted(set(r['flags_raised'])))} |")
    return "\n".join(L) + "\n"


def build_parser():
    ap = argparse.ArgumentParser(description="Check every TFM switch against the existing registry and manifests (read-only).")
    ap.add_argument("--main-repo", default=str(Path("~/distributed_prng_analysis").expanduser()),
                    help="main tree (read at --commit with git cat-file)")
    ap.add_argument("--commit", default="41ad509", help="commit of the main tree to read")
    ap.add_argument("--worktrees-root", default=str(Path("~/tfm_worktrees").expanduser()),
                    help="directory whose subdirectories are worktrees ('' to skip)")
    ap.add_argument("--out-json", required=True, help="NEW file")
    ap.add_argument("--out-md", required=True, help="NEW file")
    ap.add_argument("--patch-dir", default=None, help="NEW directory for the draft patch set (R8); omitted = none")
    ap.add_argument("--new-since", default="2026-11-23",
                    help="R8: a worktree's added switch enters the patch set if its file was modified on/after this date")
    ap.add_argument("--repair-decisions", default=None,
                    help="R9: JSON with one decision per stale ParameterSpec (requires --repair-dir)")
    ap.add_argument("--repair-dir", default=None, help="R9: NEW directory for the repaired registry draft")
    ap.add_argument("--rules", default=None, help="JSON overriding DEFAULT_RULES keys")
    ap.add_argument("--max-record-mb", type=float, default=20.0, help="largest run-record file read")
    return ap


def _check_new(p, is_dir=False):
    rp = Path(p).resolve()
    if rp.exists():
        raise RegisterError(f"{rp} exists; the checker writes only new files")
    if str(rp).startswith(("/tmp/", "/dev/shm/", "/var/tmp/")):
        raise RegisterError(f"{rp}: write under the project, not /tmp")
    if not rp.parent.is_dir():
        raise RegisterError(f"{rp.parent} does not exist")
    return rp


def write_patch(reg, rules, main_repo, commit):
    pd = Path(reg["patch"]["dir"])
    pd.mkdir()
    patched = reg["patch"].pop("patched_source")
    orig = git(main_repo, "show", f"{commit}:{rules['param_registry_py']}").stdout.decode("utf-8")
    diff = "".join(difflib.unified_diff(orig.splitlines(keepends=True), patched.splitlines(keepends=True),
                                        fromfile=f"a/{rules['param_registry_py']}", tofile=f"b/{rules['param_registry_py']}"))
    (pd / "parameter_registry.py.patch").write_text(diff)
    (pd / "parameter_registry.py.patched").write_text(patched)
    frags = {}
    for d in reg["patch"]["inserted"]:
        m = d.get("manifest_entry")
        if m:
            f = frags.setdefault(m["manifest"], {"default_params": {}, "args_map_primary_action": {}, "notes": []})
            f["default_params"].update(m["default_params"])
            f["args_map_primary_action"].update(m["args_map_primary_action"])
            f["notes"].append(f"{d['name']}: {m['note']}")
    (pd / "manifest_fragments.json").write_text(json.dumps(
        {"status": "DRAFT, not applied. agent_manifests/ is HAND_LANDED_DIRS (scripts/owner_land.py): the owner lands it by hand.",
         "fragments": frags}, indent=1) + "\n")
    sums = {n: sha256_bytes((pd / n).read_bytes()) for n in ("parameter_registry.py.patch", "parameter_registry.py.patched",
                                                             "manifest_fragments.json")}
    reg["patch"]["files_sha256"] = sums
    (pd / "SHA256SUMS").write_text("".join(f"{v}  {k}\n" for k, v in sums.items()))


def write_repair(reg, rules, main_repo, commit):
    rd = Path(reg["repair"]["dir"])
    rd.mkdir()
    repaired = reg["repair"].pop("repaired_source")
    orig = git(main_repo, "show", f"{commit}:{rules['param_registry_py']}").stdout.decode("utf-8")
    diff = "".join(difflib.unified_diff(orig.splitlines(keepends=True), repaired.splitlines(keepends=True),
                                        fromfile=f"a/{rules['param_registry_py']}", tofile=f"b/{rules['param_registry_py']}"))
    (rd / "parameter_registry.py.repair.patch").write_text(diff)
    (rd / "parameter_registry.py.repaired").write_text(repaired)
    rep = {k: v for k, v in reg["repair"].items() if k != "dir"}
    (rd / "repair_record.json").write_text(json.dumps(rep, indent=1) + "\n")
    sums = {n: sha256_bytes((rd / n).read_bytes()) for n in ("parameter_registry.py.repair.patch",
                                                             "parameter_registry.py.repaired", "repair_record.json")}
    reg["repair"]["files_sha256"] = sums
    (rd / "SHA256SUMS").write_text("".join(f"{v}  {k}\n" for k, v in sums.items()))


def main(argv=None):
    args = build_parser().parse_args(argv)
    rules = json.loads(json.dumps(DEFAULT_RULES))
    try:
        if args.rules:
            rules.update(json.loads(Path(args.rules).read_text()))
        for p in (args.out_json, args.out_md):
            _check_new(p)
        if args.patch_dir:
            _check_new(args.patch_dir)
        if bool(args.repair_decisions) != bool(args.repair_dir):
            raise RegisterError("--repair-decisions and --repair-dir go together")
        if args.repair_dir:
            _check_new(args.repair_dir)
        reg = walk(args, rules)
        if reg["patch"]:
            write_patch(reg, rules, Path(args.main_repo).resolve(), reg["main"]["commit"])
        if reg["repair"]:
            write_repair(reg, rules, Path(args.main_repo).resolve(), reg["main"]["commit"])
    except RegisterError as e:
        print(f"SWITCH_CHECK ERROR (nothing written): {e}", file=sys.stderr, flush=True)
        return 2
    Path(args.out_json).write_text(json.dumps(reg, indent=1, default=str) + "\n")
    Path(args.out_md).write_text(render_md(reg))
    c = reg["counts"]
    print(f"rows {c['rows']}; registry classes {c['registry_class']}; step scripts {c['registry_class_step_scripts']}; "
          f"stale specs {c['stale_registry_specs']}; new switches for the patch {c['new_switches_for_patch']}; "
          f"flags {c['flags']}; worktrees {len(reg['worktrees'])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
