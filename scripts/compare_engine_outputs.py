#!/usr/bin/env python3
"""compare_engine_outputs.py — the same Step 1 job, run by RANGE-MINER and by PWC:
are the outputs the same?

Owner requirement, 2026-11-20 15:55 PST: "And once finished your testing should run each
and verify the results are the same" (docs/PWC_BACKUP_OWNER_RULINGS_20261119.md).
Design: docs/drafts/pwc_backup_stage3_v1/PLAN.md §F (F2 inputs, F3 outputs).

It reads two ENGINE OUTPUT BUNDLES (below) and reports, field by field, one of three
statuses:
  EQUAL         both bundles carry the field and the values are identical;
  DIFFERENT     the values differ, or only one bundle carries the field;
  NOT COMPARED  the field is absent or null in both bundles, or null in one of them (that
                engine's outputs do not carry it). Never EQUAL and never DIFFERENT:
                nothing was compared, so nothing is claimed (Stage 4, 2026-11-21).
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
  "credit":  {"granted": bool, "swept_intervals": {"1": [s, e], ...}, ...},
  "npz":     "path/to/survivors.npz",      # relative to the bundle file
  # allowed to differ (F3), never compared:
  "engine", "run_id", "search_id", "study_id", "created_at", "finished_at",
  "duration_s", "provenance", "logs", "worker_ids", "job_ids", "geometry",
  "artifact_sha256"
}
Any other top-level key is UNCLASSIFIED and counts as a difference, so a new field can
never be ignored by accident. Every compared top-level key must be PRESENT.
`null` means "this engine's outputs do not carry it" and is how an extractor says so:
a null on EITHER side makes that field NOT COMPARED (one engine persisting a list the
other does not is not a measured difference). A key ABSENT from one bundle while the
other carries a value is DIFFERENT (e.g. a phase one engine ran and the other did not
report); absent from both is NOT COMPARED. An EMPTY list is data, not absence: a phase
with no survivors is `{"seeds": [], "match_rates": []}` and compares EQUAL to another
empty phase.

What each extractor writes (the agreed field sources, Stage 4, 2026-11-21)
-------------------------------------------------------------------------
Both extractors (`scripts/extract_pwc_bundle.py`, `scripts/extract_miner_bundle.py`)
fill the sections the same way, so a difference is the engines', not the extractors':
- `inputs`: `build_inputs()` below, from (a) the WATCHER `--params` JSON the run used,
  with the engine-selection keys of `ENGINE_PARAM_KEYS` moved to `geometry`, (b) the
  published generation's `provenance.json` (`prng_base`, `skip_modes_executed`,
  `seed_start`, `seed_end_exclusive`, `repository_commit`) and (c) the run's
  `certified_coverage` row (`dataset_sha256`).
- `phases.N`: the seeds the engine's phase N returned, with each seed's match rate as
  the exact float32 value (`float(np.float32(r))` must equal `r`; an extractor refuses
  a rate that is not exactly representable rather than round it). Seeds sorted
  ascending, stably, so a repeated seed keeps its first rate. `null` when the engine
  persisted no seed list for that phase.
- `bidirectional.constant` / `.variable`: the engine's forward∩reverse seed sets for that
  skip mode, sorted. `null` when not available. (NOT the NPZ split by `skip_mode`: the
  finalizer keeps one row per seed, `utils/run_finalizer.py:714-744`.)
- `phase_counts.N`: `{"survivor_rows": <rows the phase returned, repeats included>,
  "distinct_survivors": <distinct seeds, or null>}`.
- `credit`: `{"granted": <a certified_coverage row exists whose generation_id AND
  artifact_sha256 are this generation's, with a step1_chunk_link row naming that
  coverage_id and the same interval>,
  "certified_interval": [s, e] or null, "linked_interval": [s, e] or null (the
  step1_chunk_link row), "swept_intervals": {"1": [s, e], ...} or null (the engine's own
  sweep evidence)}`.
- `npz`: the generation's `bidirectional_survivors_binary.npz`, COPIED next to the bundle,
  its sha256 checked against `provenance.json` `artifact_sha256`. The generation must have
  no parent (`prior_row_count == 0`): a merged generation carries another run's rows.

Exit status: 0 = SAME (every field compared and equal, outside the allowed fields);
             1 = outputs differ;
             2 = not comparable: an F2 job-identity input (F2_REQUIRED_INPUTS) is
                 missing, null or different in either bundle, no input was compared,
                 or (Stage 4 default) a bundle lacks granted credit with certified,
                 linked and swept intervals equal to the run interval; or a
                 usage/format error;
             3 = no difference found, but at least one field was NOT COMPARED
                 ("SAME WHERE COMPARED" — not a claim that the outputs are the same).
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
BIDI_MODES = ("constant", "variable")
CREDIT_KEYS = ("granted", "certified_interval", "linked_interval", "swept_intervals")

# [Beta X-PWC-STAGE4 R1 Q2] The F2 job identity (docs/drafts/pwc_backup_stage3_v1/PLAN.md
# :299-312), as `build_inputs` names it. EVERY key must be present, non-null and EQUAL in
# both bundles before any engine-difference verdict; one matching parameter no longer
# suffices. (The constant skip range is in it via the two skip pins; the strategies and
# generator phase are fixed by the code at the recorded commit, `run.repository_commit`.)
F2_REQUIRED_INPUTS = (
    "run.prng_base", "run.skip_modes_executed", "run.seed_start", "run.seed_end_exclusive",
    "run.repository_commit", "run.dataset_sha256",
    "param.prng_type", "param.test_both_modes",
    "param.warm_start_window", "param.warm_start_offset", "param.warm_start_session_idx",
    "param.warm_start_fwd_thresh", "param.warm_start_rev_thresh",
    "param.warm_start_skip_min", "param.warm_start_skip_max",
)
MODE_PHASES = {"constant": ("1", "2"), "variable": ("3", "4")}
COMPARED_KEYS = ("schema", "inputs", "phases", "bidirectional", "phase_counts", "credit", "npz")
ALLOWED_TO_DIFFER = frozenset({
    "engine", "run_id", "search_id", "study_id", "created_at", "finished_at",
    "duration_s", "provenance", "logs", "worker_ids", "job_ids", "geometry",
    "artifact_sha256",
})
ARRAY_NAMES = tuple(n for n, _ in CANONICAL_ARRAY_CONTRACT)
ARRAY_DTYPES = {n: d for n, d in CANONICAL_ARRAY_CONTRACT}
_SHOW = 5          # differing items listed per field

EQUAL, DIFFERENT, NOT_COMPARED = "EQUAL", "DIFFERENT", "NOT COMPARED"

# WATCHER --params keys that select or shape an ENGINE, not the job. They legitimately
# differ between a RANGE-MINER run and a PWC run, so they go to `geometry` (F3).
# `seed_start` / `max_seeds` are replaced by WATCHER from the open search
# (agents/watcher_agent.py:2306-2308), so the job's interval comes from provenance.json.
ENGINE_PARAM_KEYS = frozenset({
    "use_range_miner", "use_persistent_workers", "use_zmq_sqlite", "pwc_transport",
    "pwc_coordinator_address", "worker_pool_size", "min_workers", "seed_cap_nvidia",
    "seed_cap_amd", "miner_stripe_size", "miner_substripes", "miner_output_dir",
    "staging_dir", "staging_workers", "staging_queue_depth", "staging_deferred_max",
    "staging_capacity_timeout", "staging_high_water_files", "staging_high_water_bytes",
    "rig_profile", "execution_set_nodes", "seed_start", "max_seeds",
})
PROVENANCE_INPUT_KEYS = ("prng_base", "skip_modes_executed", "seed_start",
                         "seed_end_exclusive", "repository_commit")


class BundleError(Exception):
    pass


# --------------------------------------------------------------------------- shared
def build_inputs(params, provenance, dataset_sha256):
    """(inputs, geometry) — the `inputs` section both extractors write, and the engine
    keys moved aside. `params` is the WATCHER --params dict the run used; `provenance`
    the generation's provenance.json; `dataset_sha256` from the certified_coverage row
    (None when there is none: that input is then NOT COMPARED, never assumed)."""
    if not isinstance(params, dict) or not isinstance(provenance, dict):
        raise BundleError("build_inputs: params and provenance must be JSON objects")
    inputs, geometry = {}, {}
    for k in sorted(params):
        (geometry if k in ENGINE_PARAM_KEYS else inputs)[k if k in ENGINE_PARAM_KEYS
                                                         else f"param.{k}"] = params[k]
    for k in PROVENANCE_INPUT_KEYS:
        inputs[f"run.{k}"] = provenance.get(k)
    inputs["run.dataset_sha256"] = dataset_sha256
    return inputs, geometry


def exact_float32(values, where):
    """Each value as the exact float32 it is; refuse (BundleError) any value that is not
    exactly representable, rather than round it."""
    out = []
    for i, v in enumerate(values):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise BundleError(f"{where}[{i}]: match rate {v!r} is not a number")
        f = float(np.float32(v))
        if f != float(v):
            raise BundleError(f"{where}[{i}]: match rate {v!r} is not an exact float32 "
                              f"(nearest is {f!r}); refusing to round it")
        out.append(f)
    return out


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
        raise BundleError(f"{path}: missing required section(s) {missing} "
                          f"(a section the engine does not carry is present as null)")
    for k in ("inputs", "phases", "bidirectional", "phase_counts", "credit"):
        if b[k] is not None and not isinstance(b[k], dict):
            raise BundleError(f"{path}: section {k!r} must be an object or null, "
                              f"got {type(b[k]).__name__}")
    if b["npz"] is not None and not isinstance(b["npz"], str):
        raise BundleError(f"{path}: 'npz' must be a path or null")
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
def _row(field, status, detail=""):
    assert status in (EQUAL, DIFFERENT, NOT_COMPARED), status
    return {"field": field, "status": status, "equal": status == EQUAL,
            "compared": status != NOT_COMPARED, "detail": detail}


def _eq(field, same, detail=""):
    return _row(field, EQUAL if same else DIFFERENT, detail)


def _nc(field, why="absent or null in both bundles"):
    """The one place a NOT COMPARED row is made (the code-mutant gate patches it)."""
    return _row(field, NOT_COMPARED, why)


_ABSENT = object()        # key not in the bundle at all (distinct from null)


def _get(d, k):
    return d.get(k, _ABSENT) if isinstance(d, dict) else _ABSENT


def _presence(field, va, vb):
    """None when both bundles carry a value; else the row to report instead.
    absent+absent, null+anything -> NOT COMPARED; absent+value -> DIFFERENT."""
    na, nb = va is _ABSENT or va is None, vb is _ABSENT or vb is None
    if not na and not nb:
        return None
    if na and nb:
        return _nc(field, "absent in both bundles" if va is _ABSENT and vb is _ABSENT
                   else "absent or null in both bundles")
    present, other, ov = ("A", "B", vb) if not na else ("B", "A", va)
    if ov is None:
        return _nc(field, f"{other}'s engine outputs do not carry it (null); {present} does")
    return _eq(field, False, f"present in {present} only")


def compare_inputs(a, b):
    r = _presence("inputs", a or None, b or None)       # {} carries no input either
    if r is not None:
        return [r]
    rows = []
    for k in sorted(set(a) | set(b) | set(F2_REQUIRED_INPUTS)):
        va, vb = _get(a, k), _get(b, k)
        p = _presence(f"inputs.{k}", va, vb)
        if p is not None:
            rows.append(p)
            continue
        rows.append(_eq(f"inputs.{k}", va == vb, "" if va == vb else f"A={va!r} B={vb!r}"))
    return rows


def _phase_map(phase, field):
    if not isinstance(phase, dict):
        raise BundleError(f"{field}: a phase must be an object or null")
    seeds = phase.get("seeds")
    rates = phase.get("match_rates")
    if not isinstance(seeds, list) or not isinstance(rates, list):
        raise BundleError(f"{field}: 'seeds' and 'match_rates' must both be lists "
                          f"(null the whole phase when the engine has no seed list)")
    if len(seeds) != len(rates):
        raise BundleError(f"{field}: {len(seeds)} seeds but {len(rates)} match rates")
    return seeds, rates


def compare_phases(a, b):
    r = _presence("phases", a, b)
    if r is not None:
        return [r]
    rows = []
    for ph in PHASES:
        fa, fb = _get(a, ph), _get(b, ph)
        base = f"phases.{ph}"
        p = _presence(base, fa, fb)
        if p is not None:
            rows.append(p)
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
        rows.append(_eq(f"{base}.survivors", same, "; ".join(detail) or f"{len(sa)} seeds"))
        # match rates, per seed, exact (first occurrence; a repeated seed is shown above)
        ma, mb = {}, {}
        for s, r_ in zip(sa, ra):
            ma.setdefault(s, r_)
        for s, r_ in zip(sb, rb):
            mb.setdefault(s, r_)
        diff = [(s, ma[s], mb[s]) for s in sorted(set(ma) & set(mb)) if ma[s] != mb[s]]
        rows.append(_eq(f"{base}.match_rates", not diff,
                        "" if not diff else f"{len(diff)} differ, e.g. " +
                        ", ".join(f"seed {s}: A={x!r} B={y!r}" for s, x, y in diff[:_SHOW])))
    extra = sorted((set(a) | set(b)) - set(PHASES))
    if extra:
        rows.append(_eq("phases.<unknown>", False, f"unexpected phase keys {extra}"))
    return rows


def compare_sets(a, b, prefix, expected=()):
    r = _presence(prefix, a, b)
    if r is not None:
        return [r]
    rows = []
    for k in sorted(set(a) | set(b) | set(expected)):
        va, vb = _get(a, k), _get(b, k)
        p = _presence(f"{prefix}.{k}", va, vb)
        if p is not None:
            rows.append(p)
            continue
        sa, sb = set(va), set(vb)
        oa, ob = sorted(sa - sb), sorted(sb - sa)
        rows.append(_eq(f"{prefix}.{k}", not oa and not ob,
                        f"{len(sa)} seeds" if not oa and not ob else
                        f"only A {oa[:_SHOW]} ({len(oa)}); only B {ob[:_SHOW]} ({len(ob)})"))
    return rows


def compare_mapping(a, b, prefix, expected=()):
    """Key by key, recursing into objects. A key absent or null in both, or an object
    empty in both, is NOT COMPARED; `expected` keys are reported even when neither
    bundle has them."""
    if isinstance(a, dict) and isinstance(b, dict) and not a and not b and not expected:
        return [_nc(prefix, "empty in both bundles")]
    r = _presence(prefix, a, b)
    if r is not None:
        return [r]
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return [_eq(prefix, a == b, "" if a == b else f"A={a!r} B={b!r}")]
    rows = []
    for k in sorted(set(a) | set(b) | set(expected)):
        rows.extend(compare_mapping(_get(a, k), _get(b, k), f"{prefix}.{k}"))
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
        rows.append(_eq(f"npz.order[{label}]", order == want,
                        "" if order == want else
                        f"file order {order} is not the frozen contract order"))
    for name in ARRAY_NAMES:
        x, y = arr_a.get(name), arr_b.get(name)
        f = f"npz.{name}"
        if x is None or y is None:
            rows.append(_eq(f, False, f"MISSING in {'A' if x is None else ''}{'B' if y is None else ''}"))
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
        rows.append(_eq(f, not problems, "; ".join(problems) or f"{x.dtype} {x.shape}"))
    extra = sorted((set(arr_a) | set(arr_b)) - set(ARRAY_NAMES))
    if extra:
        rows.append(_eq("npz.<extra arrays>", False, f"not in the frozen contract: {extra}"))
    return rows


def credit_evidence_problems(bundle, label):
    """[Beta X-PWC-STAGE4 R1 Q2] The Stage 4 session is a searched, credited run: each bundle
    must carry granted credit whose certified, linked and every required phase's swept
    interval all equal the run's interval (`run.seed_start`, `run.seed_end_exclusive`). A
    shared `granted=false` does not meet F3 (PLAN.md:318-324). Returns the problems found."""
    inp, cr = bundle.get("inputs") or {}, bundle.get("credit")
    if not isinstance(cr, dict):
        return [f"{label}: no credit section"]
    s, e = inp.get("run.seed_start"), inp.get("run.seed_end_exclusive")
    if not (isinstance(s, int) and isinstance(e, int)):
        return [f"{label}: the run interval (run.seed_start/run.seed_end_exclusive) is not recorded"]
    want, out = [s, e], []
    if cr.get("granted") is not True:
        out.append(f"{label}: credit.granted is {cr.get('granted')!r}, not true")
    for k in ("certified_interval", "linked_interval"):
        if cr.get(k) != want:
            out.append(f"{label}: credit.{k} {cr.get(k)!r} != run interval {want}")
    modes = inp.get("run.skip_modes_executed")
    if not isinstance(modes, list) or not modes or any(m not in MODE_PHASES for m in modes):
        out.append(f"{label}: run.skip_modes_executed {modes!r} is not usable")
        return out
    sw = cr.get("swept_intervals")
    for ph in [ph for m in modes for ph in MODE_PHASES[m]]:
        iv = sw.get(ph) if isinstance(sw, dict) else None
        if iv != want:
            out.append(f"{label}: credit.swept_intervals.{ph} {iv!r} != run interval {want}")
    return out


def compare(bundle_a, bundle_b, require_credit=True):
    """Pure comparison of two loaded bundles. Returns the report dict.

    NOT COMPARABLE (exit 2) unless every F2 job-identity input is EQUAL and, with
    `require_credit` (the Stage 4 default), both bundles carry the credit evidence of
    `credit_evidence_problems`. Precedence is unchanged: 2, then 1, then 3, then 0."""
    report = {"inputs": compare_inputs(bundle_a["inputs"], bundle_b["inputs"])}
    by_field = {r["field"]: r for r in report["inputs"]}
    report["identity_missing"] = [
        k for k in F2_REQUIRED_INPUTS
        if by_field.get(f"inputs.{k}", {}).get("status") != EQUAL]
    report["credit_problems"] = (credit_evidence_problems(bundle_a, "A")
                                 + credit_evidence_problems(bundle_b, "B")
                                 if require_credit else [])
    report["comparable"] = (not any(r["status"] == DIFFERENT for r in report["inputs"])
                            and any(r["status"] == EQUAL for r in report["inputs"])
                            and not report["identity_missing"]
                            and not report["credit_problems"])
    rows = []
    keys = (set(bundle_a) | set(bundle_b)) - {"_dir"}
    unclassified = sorted(keys - set(COMPARED_KEYS) - ALLOWED_TO_DIFFER)
    for k in unclassified:
        rows.append(_eq(f"<unclassified>.{k}", False,
                        "top-level key neither compared nor allowed to differ"))
    rows += compare_phases(bundle_a["phases"], bundle_b["phases"])
    rows += compare_sets(bundle_a["bidirectional"], bundle_b["bidirectional"],
                         "bidirectional", BIDI_MODES)
    rows += compare_mapping(bundle_a["phase_counts"], bundle_b["phase_counts"],
                            "phase_counts", PHASES)
    rows += compare_mapping(bundle_a["credit"], bundle_b["credit"], "credit", CREDIT_KEYS)
    p = _presence("npz", bundle_a["npz"], bundle_b["npz"])
    if p is not None:
        rows.append(p)
    else:
        oa, aa = load_npz(bundle_a)
        ob, ab = load_npz(bundle_b)
        rows += compare_npz(oa, aa, ob, ab)
    report["outputs"] = rows
    report["ignored"] = sorted(keys & ALLOWED_TO_DIFFER)
    allrows = report["inputs"] + rows
    report["not_compared"] = [r["field"] for r in allrows if r["status"] == NOT_COMPARED]
    report["n_different"] = sum(r["status"] == DIFFERENT for r in rows)
    report["same"] = (report["comparable"] and report["n_different"] == 0
                      and not report["not_compared"])
    if not report["comparable"]:
        report["verdict"] = "NOT COMPARABLE"
    elif report["n_different"]:
        report["verdict"] = "DIFFERENT"
    elif report["not_compared"]:
        report["verdict"] = "SAME WHERE COMPARED"
    else:
        report["verdict"] = "SAME"
    return report


def exit_code(report):
    return {"NOT COMPARABLE": 2, "DIFFERENT": 1, "SAME WHERE COMPARED": 3,
            "SAME": 0}[report["verdict"]]


def _print(report, a_label, b_label):
    print(f"A = {a_label}")
    print(f"B = {b_label}")
    print("INPUTS (must be the same job):")
    for r in report["inputs"]:
        print(f"  {r['status']:<12}  {r['field']}  {r['detail']}")
    print("OUTPUTS:")
    for r in report["outputs"]:
        print(f"  {r['status']:<12}  {r['field']}  {r['detail']}")
    print(f"IGNORED (allowed to differ): {', '.join(report['ignored']) or '(none present)'}")
    n_out, n_nc = len(report["outputs"]), len(report["not_compared"])
    v = report["verdict"]
    if v == "NOT COMPARABLE":
        for k in report.get("identity_missing", []):
            print(f"  JOB IDENTITY not established: inputs.{k}")
        for c in report.get("credit_problems", []):
            print(f"  CREDIT EVIDENCE: {c}")
        print("VERDICT: NOT COMPARABLE — the F2 job identity is not established in both "
              "bundles, the inputs differ, or the credit evidence is missing, so no "
              "engine-difference verdict is given")
    elif v == "SAME WHERE COMPARED":
        print(f"VERDICT: SAME WHERE COMPARED — no difference in the compared fields, but "
              f"{n_nc} field(s) NOT COMPARED: {', '.join(report['not_compared'])}. "
              f"This is not a claim that the outputs are the same.")
    else:
        print(f"VERDICT: {v} ({report['n_different']} differing output field(s) of {n_out}"
              f"{f'; {n_nc} NOT COMPARED' if n_nc else ''})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bundle_a", help="engine output bundle JSON (e.g. RANGE-MINER)")
    ap.add_argument("bundle_b", help="engine output bundle JSON (e.g. PWC)")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    ap.add_argument("--no-require-credit", action="store_true",
                    help="do not require granted credit with matching certified, linked and "
                         "swept intervals (the Stage 4 session requires it; default)")
    a = ap.parse_args(argv)
    try:
        ba, bb = load_bundle(a.bundle_a), load_bundle(a.bundle_b)
        report = compare(ba, bb, require_credit=not a.no_require_credit)
    except BundleError as exc:
        print(f"compare_engine_outputs: {exc}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        _print(report, a.bundle_a, a.bundle_b)
    return exit_code(report)


if __name__ == "__main__":
    sys.exit(main())
