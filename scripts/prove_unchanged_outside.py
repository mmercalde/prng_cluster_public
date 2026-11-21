#!/usr/bin/env python3
"""Prove that a Python file is unchanged outside a declared set of functions/classes.

Written for PWC-BACKUP Stage 3 (and reusable for any later stage). It compares one
file at a BASE commit with the working tree (or with another revision or file), and
reports every difference that falls OUTSIDE the names the caller allows to change.

What counts as "the same"
-------------------------
Equality is `ast.dump(node, include_attributes=False)`: the parsed program text with
line/column positions dropped. Comments and blank lines are not in the AST, so they
are reported by the byte-diff summary only (see --strict-bytes). Docstrings ARE in
the AST and count.

How allowed names are handled
-----------------------------
`--allow` takes qualified names: `func`, `Class`, `Class.method`, `outer.inner`
(a function nested in a function). An allowed name covers its whole definition,
including its decorators, signature, defaults and anything nested inside it.
`foo` does NOT cover `foobar`, and `Class.method` does NOT cover the rest of `Class`.

In both trees every allowed definition is replaced by a fixed placeholder that keeps
only its name and kind. Then:

  1. Top-level statements are compared in order. Any statement that differs, was
     added, was removed or moved is reported. So a moved import, a changed
     module-level constant, a changed `if`/`try` block at module level, or a
     reordered definition is caught.
  2. Every function and class definition at any depth that is not allowed (and is
     not inside an allowed one) is compared by qualified name. A change anywhere
     inside it — a default argument, a decorator, a nested function, a method's
     order — is reported, with the innermost changed name where it can be found.
  3. Every allowed name must exist exactly once at the same position in both
     trees' structure, so an allowed definition cannot be used to smuggle in a
     move. A missing allowed name is an error.

Byte-diff summary
-----------------
A line diff of the two files, with each changed line attributed to the innermost
definition that contains it (or `<module>`), and marked allowed / NOT allowed. A
changed line outside the allowed definitions that the AST check does not see (a
comment or formatting change) is listed. With `--strict-bytes` it also fails.

Exit status: 0 = no unexpected difference; 1 = at least one (printed);
             2 = usage or setup error (bad revision, unparsable file, ...).

Usage
-----
  python3 scripts/prove_unchanged_outside.py --base ee39404 --file sieve_filter.py \
          --allow execute_sieve_job
  python3 scripts/prove_unchanged_outside.py --base-file old.py --new-file new.py \
          --allow Foo.bar baz --json

Read-only. CPU only. It imports nothing from the file it checks.
"""
import argparse
import ast
import copy
import difflib
import hashlib
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


class SetupError(Exception):
    pass


# --------------------------------------------------------------------------- names

def _walk_defs(body, prefix=""):
    """Yield (qualname, node) for every def/class at any depth, in source order.
    Statements like `if`/`try`/`with`/`for` do not add to the qualified name, so a
    function defined inside a module-level `try:` is still `name`."""
    for stmt in body:
        yield from _walk_stmt(stmt, prefix)


def _walk_stmt(stmt, prefix):
    if isinstance(stmt, _DEFS):
        q = f"{prefix}{stmt.name}"
        yield q, stmt
        yield from _walk_defs(stmt.body, q + ".")
        return
    for field in ("body", "orelse", "finalbody", "handlers", "cases"):
        for child in getattr(stmt, field, None) or []:
            if isinstance(child, (ast.ExceptHandler,)) or type(child).__name__ == "match_case":
                yield from _walk_defs(child.body, prefix)
            else:
                yield from _walk_stmt(child, prefix)


def _is_allowed(qualname, allow):
    return any(qualname == a or qualname.startswith(a + ".") for a in allow)


def _placeholder(node, qualname):
    kind = type(node).__name__
    return ast.Expr(value=ast.Constant(value=f"<ALLOWED {kind} {qualname}>"))


class _Masker(ast.NodeTransformer):
    """Replace every allowed definition with a placeholder carrying name + kind."""

    def __init__(self, allow):
        self.allow, self.stack, self.masked = allow, [], []

    def _visit_def(self, node):
        q = ".".join(self.stack + [node.name])
        if q in self.allow:
            self.masked.append(q)
            return _placeholder(node, q)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()
        return node

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _visit_def


def _dump(node):
    return ast.dump(node, include_attributes=False)


def _parse(src, label):
    try:
        return ast.parse(src)
    except SyntaxError as exc:
        raise SetupError(f"{label} does not parse: {exc}") from exc


def _stmt_label(stmt):
    if isinstance(stmt, _DEFS):
        return f"{type(stmt).__name__} {stmt.name}"
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) \
            and isinstance(stmt.value.value, str) and stmt.value.value.startswith("<ALLOWED "):
        return stmt.value.value
    txt = ast.unparse(stmt).splitlines()[0] if hasattr(ast, "unparse") else type(stmt).__name__
    return (txt[:100] + "...") if len(txt) > 100 else txt


# --------------------------------------------------------------------------- core

def _top_names(stmt):
    """Names a top-level statement defines, for --allow-added: a def/class name,
    or every target name of a plain/annotated assignment. Anything else: None."""
    if isinstance(stmt, _DEFS):
        return {stmt.name}
    if isinstance(stmt, ast.Assign) and all(isinstance(t, ast.Name) for t in stmt.targets):
        return {t.id for t in stmt.targets}
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return {stmt.target.id}
    return None


def compare(base_src, new_src, allow, allow_added=()):
    """Return a report dict. report['ok'] is True only if nothing outside `allow`
    differs. `allow_added` names top-level definitions or assignments that are
    NEW by design: each must exist exactly once in the new file, at top level,
    and not at all in the base; they are then left out of the comparison.
    Pure function: no I/O."""
    allow, allow_added = set(allow), set(allow_added)
    bt, nt = _parse(base_src, "base"), _parse(new_src, "new")
    problems = []
    added_report = {}
    raw_new_statements = len(nt.body)
    if allow_added:
        b_top = [n for st in bt.body for n in (_top_names(st) or ())]
        b_all = set(b_top) | {q for q, _ in _walk_defs(bt.body)}
        n_hits = {}
        for st in nt.body:
            names = _top_names(st)
            if names and names <= allow_added:
                for nm in names:
                    n_hits[nm] = n_hits.get(nm, 0) + 1
        for nm in sorted(allow_added):
            added_report[nm] = {"in_base": nm in b_all, "new_count": n_hits.get(nm, 0)}
            if nm in b_all or n_hits.get(nm, 0) != 1:
                problems.append({"kind": "allowed-added-name-invalid", "name": nm,
                                 "base_count": int(nm in b_all),
                                 "new_count": n_hits.get(nm, 0)})
        nt = copy.deepcopy(nt)
        nt.body = [st for st in nt.body
                   if not ((_top_names(st) or None) and _top_names(st) <= allow_added)]

    # 3. allowed names: present exactly once on each side
    bq = [q for q, _ in _walk_defs(bt.body)]
    nq = [q for q, _ in _walk_defs(nt.body)]
    allowed_report = {}
    for a in sorted(allow):
        cb, cn = bq.count(a), nq.count(a)
        allowed_report[a] = {"base_count": cb, "new_count": cn}
        if cb != 1 or cn != 1:
            problems.append({"kind": "allowed-name-not-unique-or-missing", "name": a,
                             "base_count": cb, "new_count": cn})

    bm, nm = _Masker(allow), _Masker(allow)
    bmt, nmt = bm.visit(copy.deepcopy(bt)), nm.visit(copy.deepcopy(nt))
    for a in sorted(allow):
        bmasked = [q for q, n in _walk_defs(bt.body) if q == a]
        nmasked = [q for q, n in _walk_defs(nt.body) if q == a]
        allowed_report[a]["changed"] = (
            [_dump(n) for q, n in _walk_defs(bt.body) if q == a]
            != [_dump(n) for q, n in _walk_defs(nt.body) if q == a]) if bmasked and nmasked else None

    # 1. top-level statements, in order, allowed defs masked
    bd = [_dump(s) for s in bmt.body]
    nd = [_dump(s) for s in nmt.body]
    top = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, bd, nd, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        top.append({"op": op,
                    "base": [{"index": i, "stmt": _stmt_label(bmt.body[i])} for i in range(i1, i2)],
                    "new": [{"index": j, "stmt": _stmt_label(nmt.body[j])} for j in range(j1, j2)]})
    if top:
        problems.append({"kind": "top-level-statements-differ", "hunks": top})

    # 2. every non-allowed definition, by qualified name (masked, so an allowed
    #    method inside a checked class does not count against the class)
    def defmap(masked_tree):
        out = {}
        for q, n in _walk_defs(masked_tree.body):
            out.setdefault(q, []).append(_dump(n))
        return out

    bdef, ndef = defmap(bmt), defmap(nmt)
    changed_defs = []
    for q in sorted(set(bdef) | set(ndef), key=lambda s: (s.count("."), s)):
        if _is_allowed(q, allow):
            continue
        if q not in ndef:
            changed_defs.append({"name": q, "change": "removed"})
        elif q not in bdef:
            changed_defs.append({"name": q, "change": "added"})
        elif bdef[q] != ndef[q]:
            changed_defs.append({"name": q, "change": "changed"})
    if changed_defs:
        # innermost: a changed name none of whose changed children explains it alone
        names = {d["name"] for d in changed_defs}
        for d in changed_defs:
            d["has_changed_child"] = any(n.startswith(d["name"] + ".") for n in names)
        problems.append({"kind": "definitions-differ", "definitions": changed_defs})

    return {
        "allowed": allowed_report,
        "allowed_added": added_report,
        "base_top_level_statements": len(bt.body),
        "new_top_level_statements": raw_new_statements,
        "new_top_level_statements_compared": len(nt.body),
        "base_definitions": len(bq),
        "new_definitions": len(nq),
        "checked_definitions": len([q for q in set(bdef) | set(ndef) if not _is_allowed(q, allow)]),
        "problems": problems,
        "ok": not problems,
    }


# --------------------------------------------------------------------------- bytes

def _spans(tree):
    """[(start, end, qualname)] for every def, start including decorators."""
    out = []
    for q, n in _walk_defs(tree.body):
        start = min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])])
        out.append((start, n.end_lineno, q))
    return out


def _owner(lineno, spans):
    best = None
    for s, e, q in spans:
        if s <= lineno <= e and (best is None or (e - s) < (best[1] - best[0])):
            best = (s, e, q)
    return best[2] if best else "<module>"


def byte_summary(base_src, new_src, allow, allow_added=()):
    allow = set(allow) | set(allow_added)
    bl, nl = base_src.splitlines(keepends=True), new_src.splitlines(keepends=True)
    bs, ns = _spans(ast.parse(base_src)), _spans(ast.parse(new_src))
    for st in ast.parse(new_src).body:
        names = _top_names(st)
        if names and not isinstance(st, _DEFS) and names <= set(allow_added):
            ns.append((st.lineno, st.end_lineno, sorted(names)[0]))
    regions, outside = {}, []
    hunks = 0
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, bl, nl, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        hunks += 1
        for side, lo, hi, spans, lines, key in (("base", i1, i2, bs, bl, "removed"),
                                                ("new", j1, j2, ns, nl, "added")):
            for k in range(lo, hi):
                owner = _owner(k + 1, spans)
                r = regions.setdefault(owner, {"added": 0, "removed": 0,
                                               "allowed": _is_allowed(owner, allow)})
                r[key] += 1
                if not r["allowed"]:
                    outside.append({"side": side, "line": k + 1, "owner": owner,
                                    "text": lines[k].rstrip("\n")[:160]})
    return {
        "base_bytes": len(base_src.encode()), "new_bytes": len(new_src.encode()),
        "base_sha256": hashlib.sha256(base_src.encode()).hexdigest(),
        "new_sha256": hashlib.sha256(new_src.encode()).hexdigest(),
        "identical": base_src == new_src,
        "hunks": hunks,
        "lines_added": sum(r["added"] for r in regions.values()),
        "lines_removed": sum(r["removed"] for r in regions.values()),
        "by_region": dict(sorted(regions.items())),
        "changed_lines_outside_allowed": outside,
    }


# --------------------------------------------------------------------------- CLI

def _git_show(repo, rev, path):
    p = subprocess.run(["git", "-C", repo, "show", f"{rev}:{path}"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise SetupError(f"git show {rev}:{path} failed: {p.stderr.strip()[:300]}")
    return p.stdout


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError as exc:
        raise SetupError(f"cannot read {path}: {exc}") from exc


def _text(report):
    a, b = report["ast"], report["bytes"]
    out = [f"file      {report['file']}",
           f"base      {report['base']}",
           f"new       {report['new']}",
           f"allowed   {', '.join(sorted(a['allowed'])) or '(none)'}",
           f"added     {', '.join(sorted(a.get('allowed_added', {}))) or '(none)'}",
           f"AST       top-level statements base={a['base_top_level_statements']} "
           f"new={a['new_top_level_statements']}; definitions compared base={a['base_definitions']} "
           f"new={a['new_definitions']}; checked (not allowed)={a['checked_definitions']}"]
    for name, info in sorted(a["allowed"].items()):
        out.append(f"  allowed {name}: base x{info['base_count']}, new x{info['new_count']}, "
                   f"changed={info['changed']}")
    for p in a["problems"]:
        if p["kind"] == "top-level-statements-differ":
            out.append("  UNEXPECTED top-level statements:")
            for h in p["hunks"]:
                for s in h["base"]:
                    out.append(f"    [{h['op']}] base #{s['index']}: {s['stmt']}")
                for s in h["new"]:
                    out.append(f"    [{h['op']}] new  #{s['index']}: {s['stmt']}")
        elif p["kind"] == "definitions-differ":
            out.append("  UNEXPECTED definitions:")
            for d in p["definitions"]:
                tag = " (a nested definition also changed)" if d["has_changed_child"] else ""
                out.append(f"    {d['change']}: {d['name']}{tag}")
        else:
            out.append(f"  ERROR {p['kind']}: {p['name']} base x{p['base_count']} "
                       f"new x{p['new_count']}")
    for name, info in sorted(a.get("allowed_added", {}).items()):
        out.append(f"  added (allowed) {name}: in base={info['in_base']}, new x{info['new_count']}")
    out.append(f"bytes     base {b['base_bytes']} B sha256 {b['base_sha256'][:16]}; "
               f"new {b['new_bytes']} B sha256 {b['new_sha256'][:16]}; "
               f"{'identical' if b['identical'] else str(b['hunks']) + ' hunk(s), +' + str(b['lines_added']) + '/-' + str(b['lines_removed']) + ' lines'}")
    for region, r in b["by_region"].items():
        out.append(f"  {region:<40} +{r['added']:<4} -{r['removed']:<4} "
                   f"{'allowed' if r['allowed'] else 'NOT ALLOWED'}")
    if b["changed_lines_outside_allowed"]:
        out.append("  changed lines outside allowed definitions:")
        for c in b["changed_lines_outside_allowed"][:40]:
            out.append(f"    {c['side']}:{c['line']} [{c['owner']}] {c['text']}")
        if len(b["changed_lines_outside_allowed"]) > 40:
            out.append(f"    ... {len(b['changed_lines_outside_allowed']) - 40} more")
    out.append(f"verdict   {report['verdict']}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--base", help="base git revision (read with git show)")
    g.add_argument("--base-file", help="base version as a file instead of a revision")
    ap.add_argument("--file", help="repository-relative path (required with --base)")
    ap.add_argument("--new-rev", help="compare against this revision instead of the working tree")
    ap.add_argument("--new-file", help="compare against this file instead of the working tree")
    ap.add_argument("--repo", default=REPO, help="repository root (default: this script's repo)")
    ap.add_argument("--allow", nargs="*", default=[], help="qualified names allowed to change")
    ap.add_argument("--allow-added", nargs="*", default=[],
                    help="top-level definitions/assignments that are new by design")
    ap.add_argument("--strict-bytes", action="store_true",
                    help="also fail on comment/format-only changes outside the allowed names")
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    a = ap.parse_args(argv)
    try:
        if a.base and not a.file:
            raise SetupError("--file is required with --base")
        if a.new_rev and a.new_file:
            raise SetupError("give at most one of --new-rev / --new-file")
        if a.base:
            base_src, base_label = _git_show(a.repo, a.base, a.file), f"{a.base}:{a.file}"
        else:
            base_src, base_label = _read(a.base_file), a.base_file
        if a.new_file:
            new_src, new_label = _read(a.new_file), a.new_file
        elif a.new_rev:
            if not a.file:
                raise SetupError("--file is required with --new-rev")
            new_src, new_label = _git_show(a.repo, a.new_rev, a.file), f"{a.new_rev}:{a.file}"
        else:
            if not a.file:
                raise SetupError("--file is required when comparing against the working tree")
            new_src, new_label = _read(os.path.join(a.repo, a.file)), f"working tree:{a.file}"
        report = {"file": a.file or a.new_file, "base": base_label, "new": new_label,
                  "ast": compare(base_src, new_src, a.allow, a.allow_added),
                  "bytes": byte_summary(base_src, new_src, a.allow, a.allow_added)}
    except SetupError as exc:
        print(f"prove_unchanged_outside: {exc}", file=sys.stderr)
        return 2
    ok = report["ast"]["ok"]
    if a.strict_bytes and report["bytes"]["changed_lines_outside_allowed"]:
        ok = False
    report["strict_bytes"] = a.strict_bytes
    report["verdict"] = "UNCHANGED OUTSIDE ALLOWED" if ok else "UNEXPECTED DIFFERENCE"
    print(json.dumps(report, indent=2, sort_keys=True) if a.json else _text(report))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
