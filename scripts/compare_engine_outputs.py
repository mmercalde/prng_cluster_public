#!/usr/bin/env python3
"""compare_engine_outputs.py — the same Step 1 job, run by RANGE-MINER and by PWC:
are the outputs the same?

Owner requirement, 2026-11-20 15:55 PST: "And once finished your testing should run each
and verify the results are the same" (docs/PWC_BACKUP_OWNER_RULINGS_20261119.md).
Design: docs/drafts/pwc_backup_stage3_v1/PLAN.md §F (F2 inputs, F3 outputs).

It reads two ENGINE OUTPUT BUNDLES (below) and reports, field by field, EQUAL or
DIFFERENT. It exits non-zero on any difference outside the fields F3 allows to differ.
Read-only. CPU only. It imports no engine code: only numpy and the frozen 22-array
contract (`utils.canonical_arrays.CANONICAL_ARRAY_CONTRACT`).

Why `tests/phase6/wall_ab_gate.py` `compare_bundles` is NOT imported
-------------------------------------------------------------------
- Importing that module pulls in `window_optimizer`, `utils.run_finalizer` and
  `utils.survivor_loader` at module load (its top-level imports) for one function.
- Its verdict requires equal `artifact_sha256` (`compare_bundles`, `:709-713`). Between
  two ENGINES the artifact's sidecar/provenance may legitimately differ (F3), so that
  rule is wrong here.
Its method is kept: every array compared exactly, and a difference is localized (array,
first differing index, both values), never only summarized to a boolean.

Bundle format (`engine_output_bundle_v1`) — one JSON file, plus the NPZ it names
-------------------------------------------------------------------------------
{
  "schema":  "engine_output_bundle_v1",
  "inputs":  {...},                        # MUST be equal: the same job (F2)
  "phases":  {"1": {"seeds": [...], "match_rates": [...]}, "2": ..., "3": ..., "4": ...},
  "bidirectional": {"constant": [...], "variable": [...]},
  "phase_counts":  {"1": {...}, ...},       # per-phase counters, compared key by key
  "credit":  {"granted": bool, "swept_intervals": {"1": [s, e], ...}},
  "npz":     "path/to/survivors.npz",      # relative to the bundle file
  # allowed to differ (F3), never compared:
  "engine", "run_id", "search_id", "study_id", "created_at", "finished_at",
  "duration_s", "provenance", "logs", "worker_ids", "job_ids", "geometry",
  "artifact_sha256"
}
Any other top-level key is UNCLASSIFIED and counts as a difference, so a new field can
never be ignored by accident. Writing a bundle from each engine's real output is Stage 4
work (an extractor per engine); this tool only compares.

Exit status: 0 = the same (outside the allowed fields); 1 = outputs differ;
             2 = not comparable (inputs differ) or a usage/format error.
"""
import argparse
import json
import os
import sys
import zipfile
from collections import Counter

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from utils.canonical_arrays import CANONICAL_ARRAY_CONTRACT   # noqa: E402

SCHEMA = "engine_output_bundle_v1"
PHASES = ("1", "2", "3", "4")
COMPARED_KEYS = ("schema", "inputs", "phases", "bidirectional", "phase_counts", "credit", "npz")
ALLOWED_TO_DIFFER = frozenset({
    "engine", "run_id", "search_id", "study_id", "created_at", "finished_at",
    "duration_s", "provenance", "logs", "worker_ids", "job_ids", "geometry",
    "artifact_sha256",
})
ARRAY_NAMES = tuple(n for n, _ in CANONICAL_ARRAY_CONTRACT)
ARRAY_DTYPES = {n: d for n, d in CANONICAL_ARRAY_CONTRACT}
_SHOW = 5          # differing items listed per field


class BundleError(Exception):
    pass


# --------------------------------------------------------------------------- loading
def load_bundle(path):
    try:
        with open(path, encoding="utf-8") as f:
            b = json.load(f)
    except (OSError, ValueError) as exc:
        raise BundleError(f"cannot read bundle {path}: {exc}") from exc
    if not isinstance(b, dict) or b.get("schema") != SCHEMA:
        raise BundleError(f"{path}: schema is not {SCHEMA!r}")
    missing = [k for k in COMPARED_KEYS if k not in b]
    if missing:
        raise BundleError(f"{path}: missing required section(s) {missing}")
    b["_dir"] = os.path.dirname(os.path.abspath(path))
    return b


def load_npz(bundle):
    """(ordered names, {name: array}) — order is the order of the arrays in the file."""
    path = os.path.join(bundle["_dir"], bundle["npz"])
    try:
        with zipfile.ZipFile(path) as z:
            order = [n[:-4] for n in z.namelist() if n.endswith(".npy")]
        with np.load(path, allow_pickle=False) as d:
            arrays = {n: d[n] for n in d.files}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise BundleError(f"cannot read NPZ {path}: {exc}") from exc
    return order, arrays


# --------------------------------------------------------------------------- compare
def _row(field, equal, detail=""):
    return {"field": field, "equal": bool(equal), "detail": detail}


def compare_inputs(a, b):
    rows = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k, "<absent>"), b.get(k, "<absent>")
        rows.append(_row(f"inputs.{k}", va == vb, "" if va == vb else f"A={va!r} B={vb!r}"))
    return rows


def _phase_map(phase, field):
    seeds = phase.get("seeds", [])
    rates = phase.get("match_rates", [])
    if len(seeds) != len(rates):
        raise BundleError(f"{field}: {len(seeds)} seeds but {len(rates)} match rates")
    return seeds, rates


def compare_phases(a, b):
    rows = []
    for ph in PHASES:
        fa, fb = a.get(ph), b.get(ph)
        base = f"phases.{ph}"
        if fa is None or fb is None:
            rows.append(_row(f"{base}", fa is None and fb is None,
                             "" if fa is None and fb is None else
                             f"present in {'A' if fa is not None else 'B'} only"))
            continue
        sa, ra = _phase_map(fa, f"A {base}")
        sb, rb = _phase_map(fb, f"B {base}")
        ca, cb = Counter(sa), Counter(sb)
        only_a = sorted((ca - cb).elements())
        only_b = sorted((cb - ca).elements())
        dup = sorted({s for s, n in ca.items() if n > 1} | {s for s, n in cb.items() if n > 1})
        same = not only_a and not only_b
        detail = []
        if only_a:
            detail.append(f"{len(only_a)} only in A, e.g. {only_a[:_SHOW]}")
        if only_b:
            detail.append(f"{len(only_b)} only in B, e.g. {only_b[:_SHOW]}")
        if dup:
            detail.append(f"repeated seeds {dup[:_SHOW]}")
        rows.append(_row(f"{base}.survivors", same, "; ".join(detail) or f"{len(sa)} seeds"))
        # match rates, per seed, exact (first occurrence; a repeated seed is shown above)
        ma, mb = {}, {}
        for s, r in zip(sa, ra):
            ma.setdefault(s, r)
        for s, r in zip(sb, rb):
            mb.setdefault(s, r)
        diff = [(s, ma[s], mb[s]) for s in sorted(set(ma) & set(mb)) if ma[s] != mb[s]]
        rows.append(_row(f"{base}.match_rates", not diff,
                         "" if not diff else f"{len(diff)} differ, e.g. " +
                         ", ".join(f"seed {s}: A={x!r} B={y!r}" for s, x, y in diff[:_SHOW])))
    extra = sorted((set(a) | set(b)) - set(PHASES))
    if extra:
        rows.append(_row("phases.<unknown>", False, f"unexpected phase keys {extra}"))
    return rows


def compare_sets(a, b, prefix):
    rows = []
    for k in sorted(set(a) | set(b)):
        sa, sb = set(a.get(k, [])), set(b.get(k, []))
        oa, ob = sorted(sa - sb), sorted(sb - sa)
        rows.append(_row(f"{prefix}.{k}", not oa and not ob,
                         f"{len(sa)} seeds" if not oa and not ob else
                         f"only A {oa[:_SHOW]} ({len(oa)}); only B {ob[:_SHOW]} ({len(ob)})"))
    return rows


def compare_mapping(a, b, prefix):
    rows = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k, "<absent>"), b.get(k, "<absent>")
        if isinstance(va, dict) and isinstance(vb, dict):
            rows.extend(compare_mapping(va, vb, f"{prefix}.{k}"))
        else:
            rows.append(_row(f"{prefix}.{k}", va == vb, "" if va == vb else f"A={va!r} B={vb!r}"))
    return rows


def _first_diff(x, y):
    xf, yf = x.reshape(-1), y.reshape(-1)
    xb = xf.view(np.uint8).reshape(xf.size, -1) if xf.size else xf
    yb = yf.view(np.uint8).reshape(yf.size, -1) if yf.size else yf
    neq = np.flatnonzero(np.any(xb != yb, axis=1)) if xf.size else np.array([], int)
    return neq


def compare_npz(order_a, arr_a, order_b, arr_b):
    rows = []
    want = list(ARRAY_NAMES)
    for label, order in (("A", order_a), ("B", order_b)):
        rows.append(_row(f"npz.order[{label}]", order == want,
                         "" if order == want else
                         f"file order {order} is not the frozen contract order"))
    for name in ARRAY_NAMES:
        x, y = arr_a.get(name), arr_b.get(name)
        f = f"npz.{name}"
        if x is None or y is None:
            rows.append(_row(f, False, f"MISSING in {'A' if x is None else ''}{'B' if y is None else ''}"))
            continue
        problems = []
        for label, arr in (("A", x), ("B", y)):
            if arr.dtype != ARRAY_DTYPES[name]:
                problems.append(f"{label} dtype {arr.dtype}, contract {ARRAY_DTYPES[name]}")
        if x.dtype != y.dtype or x.shape != y.shape:
            problems.append(f"A {x.dtype}{x.shape} vs B {y.dtype}{y.shape}")
        else:
            neq = _first_diff(x, y)        # bit-exact (NaN-safe, -0.0 vs 0.0 differ)
            if neq.size:
                j = int(neq[0])
                problems.append(f"first differing index {j}: A={x.reshape(-1)[j]!r} "
                                f"B={y.reshape(-1)[j]!r} ({neq.size} of {x.size} differ)")
        rows.append(_row(f, not problems, "; ".join(problems) or f"{x.dtype} {x.shape}"))
    extra = sorted((set(arr_a) | set(arr_b)) - set(ARRAY_NAMES))
    if extra:
        rows.append(_row("npz.<extra arrays>", False, f"not in the frozen contract: {extra}"))
    return rows


def compare(bundle_a, bundle_b):
    """Pure comparison of two loaded bundles. Returns the report dict."""
    report = {"inputs": compare_inputs(bundle_a["inputs"], bundle_b["inputs"])}
    report["comparable"] = all(r["equal"] for r in report["inputs"])
    rows = []
    keys = (set(bundle_a) | set(bundle_b)) - {"_dir"}
    unclassified = sorted(keys - set(COMPARED_KEYS) - ALLOWED_TO_DIFFER)
    for k in unclassified:
        rows.append(_row(f"<unclassified>.{k}", False,
                         "top-level key neither compared nor allowed to differ"))
    rows += compare_phases(bundle_a["phases"], bundle_b["phases"])
    rows += compare_sets(bundle_a["bidirectional"], bundle_b["bidirectional"], "bidirectional")
    rows += compare_mapping(bundle_a["phase_counts"], bundle_b["phase_counts"], "phase_counts")
    rows += compare_mapping(bundle_a["credit"], bundle_b["credit"], "credit")
    oa, aa = load_npz(bundle_a)
    ob, ab = load_npz(bundle_b)
    rows += compare_npz(oa, aa, ob, ab)
    report["outputs"] = rows
    report["ignored"] = sorted(keys & ALLOWED_TO_DIFFER)
    report["same"] = report["comparable"] and all(r["equal"] for r in rows)
    return report


def _print(report, a_label, b_label):
    print(f"A = {a_label}")
    print(f"B = {b_label}")
    print("INPUTS (must be the same job):")
    for r in report["inputs"]:
        print(f"  {'EQUAL    ' if r['equal'] else 'DIFFERENT'}  {r['field']}  {r['detail']}")
    print("OUTPUTS:")
    for r in report["outputs"]:
        print(f"  {'EQUAL    ' if r['equal'] else 'DIFFERENT'}  {r['field']}  {r['detail']}")
    print(f"IGNORED (allowed to differ): {', '.join(report['ignored']) or '(none present)'}")
    n_diff = sum(not r["equal"] for r in report["outputs"])
    if not report["comparable"]:
        print("VERDICT: NOT COMPARABLE — the inputs differ, so this is not the same job")
    else:
        print(f"VERDICT: {'SAME' if report['same'] else 'DIFFERENT'} "
              f"({n_diff} differing output field(s) of {len(report['outputs'])})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bundle_a", help="engine output bundle JSON (e.g. RANGE-MINER)")
    ap.add_argument("bundle_b", help="engine output bundle JSON (e.g. PWC)")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    a = ap.parse_args(argv)
    try:
        ba, bb = load_bundle(a.bundle_a), load_bundle(a.bundle_b)
        report = compare(ba, bb)
    except BundleError as exc:
        print(f"compare_engine_outputs: {exc}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        _print(report, a.bundle_a, a.bundle_b)
    if not report["comparable"]:
        return 2
    return 0 if report["same"] else 1


if __name__ == "__main__":
    sys.exit(main())
