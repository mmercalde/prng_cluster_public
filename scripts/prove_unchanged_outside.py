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


def _bound_names(body):
    """Every name bound by MODULE-LEVEL code, in order, with repeats: def/class
    names, assignment / augmented / annotated targets (incl. tuple unpacking),
    `import x` / `import x.y` (binds x) / `import x as y` / `from m import a as b`,
    `for` and `with ... as` targets and `except ... as` names, inside module-level
    if/try/with/for/while blocks too. Not inside defs or classes (those are local).
    [Stage 3a, Beta X-PWC-STAGE2B R1 Q8: an added `time = 0` after `import time`
    rebinds an import, so imports count as bindings.] Walrus targets are not read."""
    out = []

    def tgt(t):
        if isinstance(t, ast.Name):
            out.append(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                tgt(e)
        elif isinstance(t, ast.Starred):
            tgt(t.value)

    def walk(stmts):
        for st in stmts:
            if isinstance(st, _DEFS):
                out.append(st.name)
                continue
            if isinstance(st, ast.Import):
                out.extend(a.asname or a.name.split(".")[0] for a in st.names)
            elif isinstance(st, ast.ImportFrom):
                out.extend(a.asname or a.name for a in st.names)
            elif isinstance(st, ast.Assign):
                for t in st.targets:
                    tgt(t)
            elif isinstance(st, (ast.AnnAssign, ast.AugAssign)):
                tgt(st.target)
            elif isinstance(st, (ast.For, ast.AsyncFor)):
                tgt(st.target)
            elif isinstance(st, (ast.With, ast.AsyncWith)):
                for it in st.items:
                    if it.optional_vars is not None:
                        tgt(it.optional_vars)
            for field in ("body", "orelse", "finalbody"):
                walk(getattr(st, field, None) or [])
            for h in getattr(st, "handlers", None) or []:
                if h.name:
                    out.append(h.name)
                walk(h.body)
            for c in getattr(st, "cases", None) or []:
                walk(c.body)
    walk(body)
    return out


def _is_literal(node):
    try:
        ast.literal_eval(node)
        return True
    except (ValueError, TypeError, SyntaxError, RecursionError):
        return False


def _is_inert_annotation(node):
    """Names, dotted names, subscripts of those, constants, tuples/lists of those,
    and `A | B`. No calls, no other operators."""
    if node is None or isinstance(node, (ast.Name, ast.Constant)):
        return True
    if isinstance(node, ast.Attribute):
        return _is_inert_annotation(node.value)
    if isinstance(node, ast.Subscript):
        return _is_inert_annotation(node.value) and _is_inert_annotation(node.slice)
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_is_inert_annotation(e) for e in node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _is_inert_annotation(node.left) and _is_inert_annotation(node.right)
    if hasattr(ast, "Index") and isinstance(node, getattr(ast, "Index")):   # Python < 3.9
        return _is_inert_annotation(node.value)
    return False


def _header_parts(node):
    """What Python evaluates when it EXECUTES a def/class statement (i.e. at import
    for a module-level one): decorators, argument defaults and annotations, the
    return annotation; for a class, its bases and keywords."""
    if isinstance(node, ast.ClassDef):
        return {"decorators": node.decorator_list, "bases": node.bases,
                "keywords": [k.value for k in node.keywords]}
    a = node.args
    anns = [x.annotation for x in a.posonlyargs + a.args + a.kwonlyargs]
    anns += [x.annotation for x in (a.vararg, a.kwarg) if x is not None]
    return {"decorators": node.decorator_list,
            "defaults": list(a.defaults) + [d for d in a.kw_defaults if d is not None],
            "annotations": [x for x in anns if x is not None] + ([node.returns] if node.returns else [])}


def _header_issues(node, where):
    """Import-time evaluation in a def/class header that is not inert."""
    parts, issues = _header_parts(node), []
    if parts["decorators"]:
        issues.append(f"{where}: decorator(s) {[ast.unparse(d) for d in parts['decorators']]} run at import")
    for d in parts.get("defaults", []):
        if not _is_literal(d):
            issues.append(f"{where}: default `{ast.unparse(d)}` is evaluated at import")
    for an in parts.get("annotations", []):
        if not _is_inert_annotation(an):
            issues.append(f"{where}: annotation `{ast.unparse(an)}` is evaluated at import")
    for b in parts.get("bases", []) + parts.get("keywords", []):
        if not _is_inert_annotation(b):
            issues.append(f"{where}: class base/keyword `{ast.unparse(b)}` is evaluated at import")
    return issues


def _param_table(fn):
    """A def's parameters in signature order: (kind, name, annotation, default).
    Positional defaults are aligned to the LAST len(defaults) positional parameters."""
    a = fn.args
    pos = [("posonly", p) for p in a.posonlyargs] + [("arg", p) for p in a.args]
    first_def = len(pos) - len(a.defaults)
    rows = [(k, p.arg, p.annotation, a.defaults[i - first_def] if i >= first_def else None)
            for i, (k, p) in enumerate(pos)]
    if a.vararg is not None:
        rows.append(("vararg", a.vararg.arg, a.vararg.annotation, None))
    rows += [("kwonly", p.arg, p.annotation, d) for p, d in zip(a.kwonlyargs, a.kw_defaults)]
    if a.kwarg is not None:
        rows.append(("kwarg", a.kwarg.arg, a.kwarg.annotation, None))
    return rows


def _changed_header_issues(fb, fn, where):
    """[Stage 3a, Beta Q8, narrowed 2026-11-21] For an ALLOWED definition whose header
    changed. Returns (import_time_issues, order_issues, removed_parameters).

    - Decorators: if the decorator list differs from the base, every decorator in the
      new header is flagged (a decorator always runs code at import).
    - A parameter's default or annotation that is not inert is flagged only when the
      parameter is new, or its default/annotation AST differs from the SAME-NAMED
      parameter's in the base. An unchanged non-inert default (already evaluated at
      import in the base) is not flagged.
    - The return annotation: flagged when not inert and different from the base's.
    - Order: the parameters present on both sides must keep their relative order and
      their kind (positional-only / positional / *args / keyword-only / **kwargs).
      Any change is an order issue, whatever the defaults.
    Classes keep the full check of _header_issues (bases and keywords when changed)."""
    if isinstance(fn, ast.ClassDef) or isinstance(fb, ast.ClassDef):
        issues = _header_issues(fn, where)
        if [_dump(d) for d in fb.decorator_list] == [_dump(d) for d in fn.decorator_list]:
            issues = [i for i in issues if "decorator(s)" not in i]
        return issues, [], []
    issues, order = [], []
    if [_dump(d) for d in fb.decorator_list] != [_dump(d) for d in fn.decorator_list] and fn.decorator_list:
        issues.append(f"{where}: decorator(s) {[ast.unparse(d) for d in fn.decorator_list]} run at import")
    tb, tn = _param_table(fb), _param_table(fn)
    bmap, nmap = {r[1]: r for r in tb}, {r[1]: r for r in tn}
    common_b = [r[1] for r in tb if r[1] in nmap]
    common_n = [r[1] for r in tn if r[1] in bmap]
    if common_b != common_n:
        order.append(f"{where}: parameter order changed: base {common_b} -> new {common_n}")
    for name in common_n:
        if bmap[name][0] != nmap[name][0]:
            order.append(f"{where}: parameter `{name}` changed kind {bmap[name][0]} -> {nmap[name][0]}")
    for kind, name, ann, dflt in tn:
        b = bmap.get(name)
        for label, node, bnode, inert in (("default", dflt, b[3] if b else None, _is_literal),
                                          ("annotation", ann, b[2] if b else None, _is_inert_annotation)):
            if node is None or inert(node):
                continue
            if b is None:
                issues.append(f"{where}: new parameter `{name}` {label} `{ast.unparse(node)}` is evaluated at import")
            elif bnode is None or _dump(bnode) != _dump(node):
                issues.append(f"{where}: parameter `{name}` {label} changed to `{ast.unparse(node)}`, "
                              f"evaluated at import")
    if fn.returns is not None and not _is_inert_annotation(fn.returns) and \
            (fb.returns is None or _dump(fb.returns) != _dump(fn.returns)):
        issues.append(f"{where}: return annotation `{ast.unparse(fn.returns)}` is evaluated at import")
    removed = [r[1] for r in tb if r[1] not in nmap]
    return issues, order, removed


def _import_time_issues(stmt, where):
    """[Stage 3a, Beta Q8] What an ADDED top-level statement runs at import, if it is
    not inert: a non-literal assigned value; a def with decorators, non-literal
    defaults or non-inert annotations; a class with a non-inert header or a body
    statement other than defs, literal assignments, a docstring or `pass`."""
    if isinstance(stmt, ast.Assign):
        return [] if _is_literal(stmt.value) else \
            [f"{where}: value `{ast.unparse(stmt.value)}` is evaluated at import (not a literal)"]
    if isinstance(stmt, ast.AnnAssign):
        out = [] if _is_inert_annotation(stmt.annotation) else \
            [f"{where}: annotation `{ast.unparse(stmt.annotation)}` is evaluated at import"]
        if stmt.value is not None and not _is_literal(stmt.value):
            out.append(f"{where}: value `{ast.unparse(stmt.value)}` is evaluated at import (not a literal)")
        return out
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return _header_issues(stmt, where)
    if isinstance(stmt, ast.ClassDef):
        out = _header_issues(stmt, where)
        for i, b in enumerate(stmt.body):
            sub = f"{where}.{getattr(b, 'name', '<body %d>' % i)}"
            if isinstance(b, ast.Pass) or (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant)):
                continue
            if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)):
                out.extend(_import_time_issues(b, sub))
            else:
                out.append(f"{sub}: class-body statement `{_stmt_label(b)}` runs at import")
        return out
    return [f"{where}: statement kind {type(stmt).__name__} cannot be checked"]


def _find_def(tree, qual):
    hits = [n for q, n in _walk_defs(tree.body) if q == qual]
    return hits[0] if len(hits) == 1 else None


def _header_text(stmt):
    return ast.unparse(stmt).splitlines()[0].strip()


def _find_block(fn, header):
    """Within def `fn` (not entering nested defs/classes), the compound statements
    whose first source line, as ast.unparse prints it, equals `header`. Returns
    [(top_index, node)], where top_index is the index in fn.body of the top-level
    statement that contains the block."""
    found = []

    def walk(stmts, top):
        for i, st in enumerate(stmts):
            ti = i if top is None else top
            if isinstance(st, _DEFS):
                continue
            if any(getattr(st, f, None) for f in ("body", "handlers", "cases")) and \
                    _header_text(st) == header:
                found.append((ti, st))
            for f in ("body", "orelse", "finalbody"):
                walk(getattr(st, f, None) or [], ti)
            for h in getattr(st, "handlers", None) or []:
                walk(h.body, ti)
            for c in getattr(st, "cases", None) or []:
                walk(c.body, ti)
    walk(fn.body, None)
    return found


def keep_block_check(bt, nt, spec):
    """[Stage 3a, Beta X-PWC-STAGE2B R1 Q8] `QUALNAME::HEADER`. The block (e.g. the
    `if _use_miner:` branch of a function that is otherwise allowed to change) must
    be found exactly once in that function on each side, and both the block and its
    PREAMBLE (every statement of the function body before the top-level statement
    that holds the block) must be AST-identical. The signature/decorators of the
    function are part of the preamble check too. Returns a list of problems."""
    if "::" not in spec:
        raise SetupError(f"--keep-block needs QUALNAME::HEADER, got {spec!r}")
    qual, header = spec.split("::", 1)
    header = header.strip()
    fb, fn = _find_def(bt, qual), _find_def(nt, qual)
    if fb is None or fn is None:
        return [{"kind": "kept-block-function-not-unique", "name": qual, "block": header,
                 "base_count": int(fb is not None), "new_count": int(fn is not None)}]
    hb, hn = _find_block(fb, header), _find_block(fn, header)
    if len(hb) != 1 or len(hn) != 1:
        return [{"kind": "kept-block-not-unique", "name": qual, "block": header,
                 "base_count": len(hb), "new_count": len(hn)}]
    (ib, nb), (in_, nn) = hb[0], hn[0]
    out = []
    if _dump(nb) != _dump(nn):
        out.append({"kind": "kept-block-differs", "name": qual, "block": header})
    sig_b = _dump(ast.FunctionDef(name="_", args=fb.args, body=[], decorator_list=fb.decorator_list,
                                  returns=fb.returns)) if not isinstance(fb, ast.ClassDef) else _dump(fb)
    sig_n = _dump(ast.FunctionDef(name="_", args=fn.args, body=[], decorator_list=fn.decorator_list,
                                  returns=fn.returns)) if not isinstance(fn, ast.ClassDef) else _dump(fn)
    if ib != in_ or [_dump(x) for x in fb.body[:ib]] != [_dump(x) for x in fn.body[:in_]] or sig_b != sig_n:
        out.append({"kind": "kept-block-preamble-differs", "name": qual, "block": header,
                    "base_preamble_statements": ib, "new_preamble_statements": in_})
    return out


def compare(base_src, new_src, allow, allow_added=(), keep_blocks=()):
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
        b_all = set(_bound_names(bt.body)) | {q for q, _ in _walk_defs(bt.body)}
        n_bound = _bound_names(nt.body)
        n_hits = {}
        for st in nt.body:
            names = _top_names(st)
            if names and names <= allow_added:
                for nm in names:
                    n_hits[nm] = n_hits.get(nm, 0) + 1
                issues = _import_time_issues(st, "/".join(sorted(names)))
                if issues:
                    problems.append({"kind": "allowed-added-import-time-evaluation",
                                     "name": "/".join(sorted(names)), "issues": issues})
        for nm in sorted(allow_added):
            added_report[nm] = {"in_base": nm in b_all, "new_count": n_hits.get(nm, 0),
                                "new_bindings": n_bound.count(nm)}
            # bound anywhere in the base (incl. as an import), or bound more than
            # once in the new file (the added statement plus a rebinding): invalid
            if nm in b_all or n_hits.get(nm, 0) != 1 or n_bound.count(nm) != 1:
                problems.append({"kind": "allowed-added-name-invalid", "name": nm,
                                 "base_count": int(nm in b_all),
                                 "new_count": n_hits.get(nm, 0),
                                 "new_bindings": n_bound.count(nm)})
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

    # [Stage 3a, Beta Q8] an allowed definition's HEADER runs when its def/class
    # statement runs (module-level: at import). When the header changed, every part
    # of the new header must be inert -- narrowed 2026-11-21 (owner): a non-inert part
    # is flagged only where it differs from the same parameter's in the base, or the
    # parameter is new; any reorder or kind change of existing parameters is caught.
    header_notes = []
    for a in sorted(allow):
        fb_, fn_ = _find_def(bt, a), _find_def(nt, a)
        if fb_ is None or fn_ is None:
            continue
        hb_ = {k: [_dump(x) for x in v] for k, v in _header_parts(fb_).items()}
        hn_ = {k: [_dump(x) for x in v] for k, v in _header_parts(fn_).items()}
        if not isinstance(fb_, ast.ClassDef) and not isinstance(fn_, ast.ClassDef):
            # parameter names and kinds too: a pure reorder of default-less parameters
            # leaves decorators, defaults and annotations unchanged
            hb_["parameters"] = [r[:2] for r in _param_table(fb_)]
            hn_["parameters"] = [r[:2] for r in _param_table(fn_)]
        if hb_ != hn_:
            changed_parts = sorted(k for k in set(hb_) | set(hn_) if hb_.get(k) != hn_.get(k))
            issues, order, removed = _changed_header_issues(fb_, fn_, a)
            header_notes.append({"name": a, "changed": changed_parts, "issues": issues,
                                 "order_issues": order, "removed_parameters": removed})
            if issues:
                problems.append({"kind": "allowed-header-import-time-evaluation", "name": a,
                                 "issues": issues})
            if order:
                problems.append({"kind": "allowed-header-parameter-order-changed", "name": a,
                                 "issues": order})
    kept = []
    for spec in keep_blocks:
        kp = keep_block_check(bt, nt, spec)
        kept.append({"spec": spec, "ok": not kp})
        problems.extend(kp)

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
        "allowed_header_changes": header_notes,
        "kept_blocks": kept,
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
        elif "issues" in p:
            out.append(f"  ERROR {p['kind']}: {p['name']}")
            for i in p["issues"]:
                out.append(f"    {i}")
        elif p["kind"].startswith("kept-block"):
            out.append(f"  ERROR {p['kind']}: {p['name']} :: {p['block']}"
                       + (f" (base x{p['base_count']}, new x{p['new_count']})" if "base_count" in p else ""))
        else:
            out.append(f"  ERROR {p['kind']}: {p['name']} base x{p['base_count']} "
                       f"new x{p['new_count']}" + (f" bindings in new x{p['new_bindings']}"
                                                   if "new_bindings" in p else ""))
    for h in a.get("allowed_header_changes", []):
        out.append(f"  allowed {h['name']}: header changed ({', '.join(h['changed'])}); "
                   f"import-time issues: {len(h['issues'])}; order/kind issues: "
                   f"{len(h.get('order_issues', []))}"
                   + (f"; removed parameters: {h['removed_parameters']}" if h.get("removed_parameters") else ""))
        for i in h["issues"] + h.get("order_issues", []):
            out.append(f"    - {i}")
    for k in a.get("kept_blocks", []):
        out.append(f"  kept block {k['spec']}: {'identical, with preamble' if k['ok'] else 'DIFFERS'}")
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
    ap.add_argument("--keep-block", nargs="*", default=[], metavar="QUALNAME::HEADER",
                    help="a block inside an allowed function that must stay AST-identical, "
                         "with everything before it in that function (e.g. "
                         "'run_bidirectional_test::if _use_miner:')")
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
                  "ast": compare(base_src, new_src, a.allow, a.allow_added, a.keep_block),
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
