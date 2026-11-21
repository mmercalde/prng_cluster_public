#!/usr/bin/env python3
"""extract_pwc_bundle.py — turn ONE finished PWC Step 1 run into an
`engine_output_bundle_v1` for scripts/compare_engine_outputs.py (Stage 4).

CPU only. Read-only on the run: it opens the run tree's ledger with
`?mode=ro&immutable=1` (no -wal/-shm sidecars), never writes into the run tree, and
writes only into a NEW --out-dir. No GPU, no SSH, no engine run.

What a finished PWC run leaves on disk, and what this reads
----------------------------------------------------------
(a) The four per-pass result files PWC writes from `run_sieve_pass`
    (`persistent_worker_coordinator.py:2504-2535`), named by `run_trial_persistent`
    (`:2715`, `:2742`, `:2787`, `:2812`):
        <run-dir>/results/window_opt_{forward,reverse,forward_hybrid,reverse_hybrid}_<W>_<O>_t<T>.json
    Each holds `pwc_pass_accounting` (`:2431-2449`). At or below 100,000 survivors it
    also holds the full `survivors` / `match_rates` lists (the survivor map); above
    that, only a summary (`:2517-2532`) and that phase is then null (NOT COMPARED).
(b) The trial completion record `pwc_completion` is NOT written to disk by the run; it
    lives in the result dict only (`:2654-2665`). This tool REBUILDS it from (a) with
    the run's own builder, `build_pwc_trial_completion` (`:725-771`), imported, not
    re-implemented. Its `swept_intervals` is the credit evidence the run used
    (`window_optimizer_integration_final.py:1465-1483`).
(c) The published generation: `<run-dir>/.s172_accumulator/current/` with
    `provenance.json` and `bidirectional_survivors_binary.npz`
    (`utils/run_finalizer.py:124-136`). Its sha256 must equal `artifact_sha256`. It must
    have NO parent: the finalizer merges the prior generation into each new one
    (`utils/run_finalizer.py:1620-1640`), so a merged NPZ carries other runs' rows and
    is refused here.
(d) The run tree's `prng_analysis.db`: the `certified_coverage` row for this generation id
    AND artifact hash (`utils/seed_coverage_ledger.py:645-665`) and the `step1_chunk_link`
    row naming that coverage id (`utils/step1_search_ledger.py:306-321`). A non-empty
    `-wal` is read from a stable byte copy outside the run, as `extract_miner_bundle.py`
    does; otherwise an immutable read.
(e) The WATCHER `--params` JSON the run used (a file the runbook saves before launch).

Field sources are the ones agreed in `scripts/compare_engine_outputs.py`'s docstring;
`build_inputs` and `exact_float32` are imported from there so both extractors share them.

Refusals (exit 2, nothing written): a live path (the main tree or ~/miner_staging, on the
given roots and on every resolved file opened); an --out-dir inside the run tree, the
generation, the results dir or the ledger's directory; a params file that is not a PWC
run; a missing or stale pass file; a pass not on TCP or not over the generation's interval;
pass accounting that fails the producer's completion conditions; a threshold record not
agreed AND verified, or not at the trial's requested threshold; an incomplete trial; a
generation with a parent; an artifact hash mismatch; a match rate that is not an exact
float32; NPZ rows outside an available mode's bidirectional set, or a set seed with no
row, or (all modes available) NPZ seeds that are not the union of the sets; a run ledger
that is UNAVAILABLE (non-empty -journal, or changing under a -wal copy); no
certified_coverage row bound to BOTH this generation id and this artifact hash, any row
sharing only one of them, a certified interval that is not the generation's; no
step1_chunk_link row, or a link naming another coverage id or interval (Beta
X-PWC-STAGE4 R1 Q3/Q5).

Usage:
  python3 -u scripts/extract_pwc_bundle.py --run-dir /home/michael/stage4_pwc_tree \\
      --params-json /home/michael/stage4_pwc_tree/data/stage4/pwc_params.json \\
      --out-dir /home/michael/distributed_prng_analysis/data/stage4/bundle_pwc
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import zipfile

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import importlib.util                                                        # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "compare_engine_outputs", os.path.join(_ROOT, "scripts", "compare_engine_outputs.py"))
CEO = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CEO)
from persistent_worker_coordinator import (                                 # noqa: E402
    PWC_PASS_ACCOUNTING_SCHEMA, PWC_PHASES, PWC_TRIAL_COMPLETION_SCHEMA,
    build_pwc_trial_completion,
)
from utils.prng_encoding import SKIP_MODE_ENCODING                         # noqa: E402

ENGINE = "pwc"
ACCUMULATOR_CURRENT = os.path.join(".s172_accumulator", "current")
BINARY_NPZ = "bidirectional_survivors_binary.npz"
SIDECAR = "provenance.json"
# phase -> (file stem, prng_type suffix); the names run_trial_persistent uses
PASS_FILES = {"1": ("forward", ""), "2": ("reverse", "_reverse"),
              "3": ("forward_hybrid", "_hybrid"), "4": ("reverse_hybrid", "_hybrid_reverse")}
MODE_PHASES = {"constant": ("1", "2"), "variable": ("3", "4")}
LIVE_PATHS = (os.path.expanduser("~/distributed_prng_analysis"),
              os.path.expanduser("~/miner_staging"))


class ExtractRefused(Exception):
    pass


def _real(p):
    return os.path.realpath(os.path.abspath(p))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def refuse_live(path, what, live_paths=LIVE_PATHS):
    rp = _real(path)
    for lp in live_paths:
        lr = _real(lp)
        if rp == lr or rp.startswith(lr + os.sep):
            raise ExtractRefused(f"{what} {path} is under the live path {lp}; Stage 4 runs "
                                 f"and reads only a separate run tree")


def _load_json(path, what):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        raise ExtractRefused(f"cannot read {what} {path}: {exc}") from exc


# --------------------------------------------------------------------------- generation
def read_generation(gen_dir):
    prov_path = os.path.join(gen_dir, SIDECAR)
    npz_path = os.path.join(gen_dir, BINARY_NPZ)
    prov = _load_json(prov_path, "provenance")
    if prov.get("parent_generation_id") is not None or prov.get("prior_row_count") != 0:
        raise ExtractRefused(
            f"generation {gen_dir} has a parent ({prov.get('parent_generation_id')!r}, "
            f"prior_row_count={prov.get('prior_row_count')!r}): the finalizer merged an "
            f"earlier generation into it, so its NPZ is not this run's alone")
    if not os.path.isfile(npz_path):
        raise ExtractRefused(f"no {BINARY_NPZ} in {gen_dir}")
    sha = _sha256(npz_path)
    if sha != prov.get("artifact_sha256"):
        raise ExtractRefused(f"{npz_path} sha256 {sha} != provenance artifact_sha256 "
                             f"{prov.get('artifact_sha256')}")
    try:
        with zipfile.ZipFile(npz_path) as z:
            names = [n[:-4] for n in z.namelist() if n.endswith(".npy")]
        with np.load(npz_path, allow_pickle=False) as d:
            arrays = {n: d[n] for n in d.files}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ExtractRefused(f"cannot read {npz_path}: {exc}") from exc
    if names != list(CEO.ARRAY_NAMES):
        raise ExtractRefused(f"{npz_path}: arrays {names} are not the frozen 22-array order")
    return prov, prov_path, npz_path, arrays


# --------------------------------------------------------------------------- passes
def pass_file(results_dir, ph, ws, off, trial):
    stem, _ = PASS_FILES[ph]
    return os.path.join(results_dir, f"window_opt_{stem}_{ws}_{off}_t{trial}.json")


_OUTCOMES = ("completed", "failed", "timed_out", "not_dispatched", "missing")


def validate_accounting(ph, d, acc, req):
    """[Beta X-PWC-STAGE4 R1 Q5] The producer's own completion conditions, not the saved
    flag alone: every planned chunk completed, the completed chunks tile exactly the
    requested interval, the outcome counts add up, and nothing failed, timed out, went
    undispatched or missing (persistent_worker_coordinator.py:2424-2449)."""
    planned = acc.get("planned")
    counts = {k: acc.get(k) for k in _OUTCOMES}
    probs = []
    if not isinstance(planned, int) or isinstance(planned, bool) or planned < 0:
        probs.append(f"planned={planned!r}")
    elif any(not isinstance(v, int) or isinstance(v, bool) for v in counts.values()):
        probs.append(f"outcome counts {counts}")
    else:
        if sum(counts.values()) != planned:
            probs.append(f"outcomes {counts} do not add up to planned={planned}")
        if counts["completed"] != planned:
            probs.append(f"completed={counts['completed']} of planned={planned}")
        bad = {k: v for k, v in counts.items() if k != "completed" and v}
        if bad:
            probs.append(f"non-completed chunks {bad}")
        if acc.get("dispatched") != planned:
            probs.append(f"dispatched={acc.get('dispatched')!r} != planned={planned}")
    empty = req[0] == req[1]
    if not empty and acc.get("completed_intervals") != [list(req)]:
        probs.append(f"completed_intervals {acc.get('completed_intervals')!r} != [{list(req)}]")
    if acc.get("complete") is not True:
        probs.append(f"complete={acc.get('complete')!r}")
    if acc.get("incomplete_chunks"):
        probs.append(f"{len(acc['incomplete_chunks'])} incomplete chunk row(s)")
    if acc.get("pass_progress_timeout") is not False:
        probs.append(f"pass_progress_timeout={acc.get('pass_progress_timeout')!r}")
    if d.get("pass_complete") is not True or d.get("failed_chunks") not in (0, None):
        probs.append(f"pass_complete={d.get('pass_complete')!r} failed_chunks={d.get('failed_chunks')!r}")
    if probs:
        raise ExtractRefused(f"phase {ph}: pass accounting does not meet the producer's "
                             f"completion conditions: " + "; ".join(probs))


def validate_threshold(ph, rec, want_type, want):
    """[Beta X-PWC-STAGE4 R1 Q5] The credit path's threshold wall
    (window_optimizer_integration_final.py:1456-1463: agree AND verified), plus the record
    naming this pass and the trial's requested directional threshold exactly."""
    if not isinstance(rec, dict):
        raise ExtractRefused(f"phase {ph}: no threshold record")
    hybrid = ph in ("3", "4")
    probs = []
    if rec.get("agree") is not True or rec.get("verified") is not True:
        probs.append(f"agree={rec.get('agree')!r} verified={rec.get('verified')!r}")
    if rec.get("prng_type") != want_type:
        probs.append(f"prng_type {rec.get('prng_type')!r} != {want_type!r}")
    if hybrid and rec.get("kind") != "hybrid":
        probs.append(f"kind {rec.get('kind')!r} != 'hybrid'")
    try:
        if float(rec.get("requested")) != float(want):
            probs.append(f"requested {rec.get('requested')!r} != the trial's {want!r}")
    except (TypeError, ValueError):
        probs.append(f"requested {rec.get('requested')!r} is not a number")
    if rec.get("effective_unreported_chunks") != 0 or not rec.get("effective"):
        probs.append(f"effective={rec.get('effective')!r} unreported="
                     f"{rec.get('effective_unreported_chunks')!r}")
    if probs:
        raise ExtractRefused(f"phase {ph}: threshold record not verified: " + "; ".join(probs))


def read_passes(results_dir, prov, prov_path, ws, off, trial, params, live_paths=LIVE_PATHS):
    """{phase: pass dict} for every phase the run executed, each checked."""
    fwd, rev = params.get("warm_start_fwd_thresh"), params.get("warm_start_rev_thresh")
    thresholds = {"1": fwd, "2": rev, "3": fwd, "4": rev}
    base = prov["prng_base"]
    req = [int(prov["seed_start"]), int(prov["seed_end_exclusive"])]
    modes = list(prov.get("skip_modes_executed") or [])
    if not modes or any(m not in MODE_PHASES for m in modes):
        raise ExtractRefused(f"provenance skip_modes_executed {modes!r} is not usable")
    required = [ph for m in modes for ph in MODE_PHASES[m]]
    prov_mtime = os.path.getmtime(prov_path)
    out, files = {}, {}
    for ph in required:
        path = pass_file(results_dir, ph, ws, off, trial)
        refuse_live(path, f"phase {ph} pass file", live_paths)
        if not os.path.isfile(path):
            raise ExtractRefused(f"phase {ph}: pass file {path} is missing")
        if os.path.getmtime(path) > prov_mtime:
            raise ExtractRefused(f"phase {ph}: {path} is newer than the generation's "
                                 f"provenance.json; it is not this run's output")
        d = _load_json(path, f"phase {ph} pass file")
        acc = d.get("pwc_pass_accounting")
        if not isinstance(acc, dict) or acc.get("schema") != PWC_PASS_ACCOUNTING_SCHEMA:
            raise ExtractRefused(f"phase {ph}: {path} has no {PWC_PASS_ACCOUNTING_SCHEMA}")
        want_type = base + PASS_FILES[ph][1]
        if acc.get("prng_type") != want_type or d.get("prng_type") != want_type:
            raise ExtractRefused(f"phase {ph}: prng_type {acc.get('prng_type')!r} / "
                                 f"{d.get('prng_type')!r}, expected {want_type!r}")
        if list(acc.get("requested_interval") or []) != req:
            raise ExtractRefused(f"phase {ph}: requested_interval "
                                 f"{acc.get('requested_interval')} != the generation's {req}")
        if acc.get("transport") != "tcp":
            raise ExtractRefused(f"phase {ph}: ran on {acc.get('transport')!r}, not tcp")
        validate_accounting(ph, d, acc, req)
        validate_threshold(ph, acc.get("threshold"), want_type, thresholds[ph])
        out[ph] = d
        files[ph] = {"path": path, "sha256": _sha256(path)}
    return out, files, required, req


def phase_section(d, ph):
    """(phase entry or None, counts) from one pass file."""
    rows = d.get("survivor_count")
    if "survivors" not in d:
        return None, {"survivor_rows": rows, "distinct_survivors": None}
    seeds, rates = d["survivors"], d.get("match_rates")
    if not isinstance(seeds, list) or not isinstance(rates, list) or len(seeds) != len(rates):
        raise ExtractRefused(f"phase {ph}: survivors/match_rates are not equal-length lists")
    if any(isinstance(s, bool) or not isinstance(s, int) for s in seeds):
        raise ExtractRefused(f"phase {ph}: a survivor seed is not an integer")
    if rows != len(seeds):
        raise ExtractRefused(f"phase {ph}: survivor_count {rows} != {len(seeds)} listed")
    try:
        rates = CEO.exact_float32(rates, f"phase {ph} match_rates")
    except CEO.BundleError as exc:
        raise ExtractRefused(str(exc)) from exc
    order = sorted(range(len(seeds)), key=lambda i: seeds[i])        # stable
    entry = {"seeds": [seeds[i] for i in order], "match_rates": [rates[i] for i in order]}
    return entry, {"survivor_rows": rows, "distinct_survivors": len(set(seeds))}


# --------------------------------------------------------------------------- ledger
def _stat_key(path):
    st = os.stat(path)
    return st.st_size, st.st_mtime_ns


def read_sqlite(path, queries, scratch_parent, optional=(), live_paths=LIVE_PATHS):
    """[Beta X-PWC-STAGE4 R1 Q5] Read-only queries that see committed WAL content and change
    nothing beside `path`; the rule of the Step 1 credit read
    (window_optimizer_integration_final.py:1576-1610) and of extract_miner_bundle.read_sqlite:
      * a non-empty -journal: UNAVAILABLE (a rollback transaction is in flight), refused;
      * no -wal, or an empty one: `?mode=ro&immutable=1` on the database itself;
      * a non-empty -wal: the database and its -wal are copied byte for byte into a scratch
        directory OUTSIDE the run, the sources must be unchanged across the copy (else
        UNAVAILABLE: not a stopped run), and the copy is read.
    A query named in `optional` whose table does not exist returns None."""
    refuse_live(path, "ledger", live_paths)
    if not os.path.isfile(path):
        raise ExtractRefused(f"no run ledger at {path}")
    jp, wp = path + "-journal", path + "-wal"
    if os.path.exists(jp) and os.path.getsize(jp) > 0:
        raise ExtractRefused(f"run ledger UNAVAILABLE: {jp} is non-empty (a rollback "
                             f"transaction is in flight)")

    def run(con):
        out = {}
        for k, (sql, args) in queries.items():
            try:
                out[k] = con.execute(sql, args).fetchall()
            except sqlite3.OperationalError as exc:
                if k in optional and "no such table" in str(exc):
                    out[k] = None
                else:
                    raise ExtractRefused(f"cannot read the run ledger {path}: {k}: {exc}") from exc
        return out

    try:
        if not (os.path.exists(wp) and os.path.getsize(wp) > 0):
            con = sqlite3.connect(f"file:{_real(path)}?mode=ro&immutable=1", uri=True)
            try:
                return run(con), "immutable"
            finally:
                con.close()
        tmp = tempfile.mkdtemp(prefix=".extract_sqlite_", dir=scratch_parent)
        try:
            before = (_stat_key(path), _stat_key(wp))
            dst = os.path.join(tmp, "copy.db")
            shutil.copyfile(path, dst)
            shutil.copyfile(wp, dst + "-wal")
            if (_stat_key(path), _stat_key(wp)) != before:
                raise ExtractRefused(f"run ledger UNAVAILABLE: {path} changed while it was "
                                     f"being copied; the run is not stopped")
            con = sqlite3.connect(dst)
            try:
                return run(con), "wal_copy"
            finally:
                con.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    except sqlite3.Error as exc:
        raise ExtractRefused(f"cannot read the run ledger {path}: {exc}") from exc


def read_ledger(db_path, prov, scratch_parent, live_paths=LIVE_PATHS):
    """[Beta X-PWC-STAGE4 R1 Q3] The coverage row and the link, each bound to this
    generation by its cross-identifiers, not only by the key it was looked up with:
      coverage: artifact_sha256 AND generation_id are this generation's, and its interval
                is the generation's;
      link:     generation_id is this generation's, coverage_id is that coverage row's, and
                its interval equals the coverage interval.
    A Stage 4 run is a searched, credited run, so both rows are REQUIRED."""
    rows, how = read_sqlite(db_path, {
        "cov": ("SELECT seed_start, seed_end_exclusive, dataset_sha256, coverage_id, "
                "generation_id, artifact_sha256 FROM certified_coverage "
                "WHERE artifact_sha256=? OR generation_id=?",
                (prov["artifact_sha256"], prov["generation_id"])),
        "link": ("SELECT seed_start, seed_end_exclusive, search_id, coverage_id, generation_id "
                 "FROM step1_chunk_link WHERE generation_id=?", (prov["generation_id"],)),
    }, scratch_parent, optional=("link",), live_paths=live_paths)
    covs, links = rows["cov"] or [], rows["link"]
    req = [int(prov["seed_start"]), int(prov["seed_end_exclusive"])]
    mine = [c for c in covs if c[4] == prov["generation_id"] and c[5] == prov["artifact_sha256"]]
    other = [c for c in covs if c not in mine]
    if other:
        raise ExtractRefused(
            f"certified_coverage holds {len(other)} row(s) sharing this artifact hash or "
            f"generation id but not both (e.g. generation {other[0][4]!r}, artifact "
            f"{str(other[0][5])[:16]}): another generation's credit cannot be attached")
    if len(mine) != 1:
        raise ExtractRefused(f"{len(mine)} certified_coverage rows for generation "
                             f"{prov['generation_id']} and its artifact; exactly one is "
                             f"required (a Stage 4 run is credited)")
    cov = mine[0]
    if [int(cov[0]), int(cov[1])] != req:
        raise ExtractRefused(f"certified interval {[cov[0], cov[1]]} != the generation's {req}")
    if links is None:
        raise ExtractRefused("the run ledger has no step1_chunk_link table: the run was not "
                             "a search chunk, and a Stage 4 run is")
    if len(links) != 1:
        raise ExtractRefused(f"{len(links)} step1_chunk_link rows for generation "
                             f"{prov['generation_id']}; exactly one is required")
    link = links[0]
    if link[3] != cov[3]:
        raise ExtractRefused(f"the link names coverage {link[3]!r}, not this generation's "
                             f"coverage {cov[3]!r}")
    if [int(link[0]), int(link[1])] != [int(cov[0]), int(cov[1])]:
        raise ExtractRefused(f"linked interval {[link[0], link[1]]} != certified interval "
                             f"{[cov[0], cov[1]]}")
    return cov, link, how


# --------------------------------------------------------------------------- main
def _inside(path, root):
    rp, rr = _real(path), _real(root)
    return rp == rr or rp.startswith(rr + os.sep)


def extract(run_dir, params_path, out_dir, trial=0, generation_dir=None, results_dir=None,
            ledger_db=None, live_paths=LIVE_PATHS):
    refuse_live(run_dir, "--run-dir", live_paths)
    gen_dir = _real(generation_dir or os.path.join(run_dir, ACCUMULATOR_CURRENT))
    results_dir = results_dir or os.path.join(run_dir, "results")
    ledger_db = ledger_db or os.path.join(run_dir, "prng_analysis.db")
    for p, what in ((gen_dir, "generation"), (results_dir, "results dir"),
                    (ledger_db, "ledger")):
        refuse_live(p, what, live_paths)
    # [Beta X-PWC-STAGE4 R1 Q5] the output never lands inside anything the tool reads
    for root, what in ((run_dir, "the run tree"), (gen_dir, "the generation"),
                       (results_dir, "the results dir"),
                       (os.path.dirname(_real(ledger_db)), "the ledger's directory")):
        if _inside(out_dir, root):
            raise ExtractRefused(f"--out-dir {out_dir} is inside {what} ({root}); the "
                                 f"extractor never writes into the run")
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise ExtractRefused(f"--out-dir {out_dir} exists and is not empty; a bundle is "
                             f"never overwritten")

    params = _load_json(params_path, "params")
    if not isinstance(params, dict):
        raise ExtractRefused("params JSON is not an object")
    if params.get("use_persistent_workers") is not True or params.get("use_range_miner") is not False:
        raise ExtractRefused(f"params are not a PWC run (use_persistent_workers="
                             f"{params.get('use_persistent_workers')!r}, use_range_miner="
                             f"{params.get('use_range_miner')!r})")
    ws, off = params.get("warm_start_window"), params.get("warm_start_offset")
    if not isinstance(ws, int) or not isinstance(off, int):
        raise ExtractRefused("params lack integer warm_start_window / warm_start_offset; "
                             "the pass files are named by them")

    prov, prov_path, npz_path, arrays = read_generation(gen_dir)
    for p, what in ((prov_path, "provenance"), (npz_path, "NPZ")):
        refuse_live(p, what, live_paths)          # the resolved files actually opened
    passes, pass_files, required, req = read_passes(results_dir, prov, prov_path, ws, off,
                                                    trial, params, live_paths)

    comp = build_pwc_trial_completion(tuple(req), {ph: passes.get(ph) for ph, *_ in PWC_PHASES
                                                   if ph in passes})
    bad = [ph for ph in required if comp["passes"][ph]["status"] != "complete"]
    if comp["schema"] != PWC_TRIAL_COMPLETION_SCHEMA or bad or not comp["trial_complete"]:
        raise ExtractRefused(f"trial not complete: required passes {bad} not complete "
                             f"(incomplete_passes={comp['incomplete_passes']})")
    swept = {ph: comp["swept_intervals"][ph] for ph in required}

    phases, counts = {}, {}
    for ph, *_ in PWC_PHASES:
        if ph in passes:
            phases[ph], counts[ph] = phase_section(passes[ph], ph)
        else:
            phases[ph], counts[ph] = None, None

    def _bidi(f, r):
        if phases.get(f) is None or phases.get(r) is None:
            return None
        return sorted(set(phases[f]["seeds"]) & set(phases[r]["seeds"]))
    bidi = {m: (_bidi(*MODE_PHASES[m]) if m in prov["skip_modes_executed"] else None)
            for m in CEO.BIDI_MODES}

    # [Beta X-PWC-STAGE4 R1 Q5] membership is checked for EVERY mode whose pair of lists is
    # available; the union equality only when all executed modes are available.
    checks = []
    npz_seeds = set(int(s) for s in arrays["seeds"].tolist())
    for m in prov["skip_modes_executed"]:
        if bidi.get(m) is None:
            checks.append(f"{m}: membership check NOT RUN (a phase of this mode has no seed list)")
            continue
        code = SKIP_MODE_ENCODING[m]
        rows_m = set(int(s) for s in arrays["seeds"][arrays["skip_mode"] == code].tolist())
        if not rows_m <= set(bidi[m]):
            raise ExtractRefused(f"NPZ rows with skip_mode {m} are not all in the {m} "
                                 f"bidirectional set: {sorted(rows_m - set(bidi[m]))[:5]}")
        if not set(bidi[m]) <= npz_seeds:
            raise ExtractRefused(f"{m} bidirectional seeds missing from the NPZ: "
                                 f"{sorted(set(bidi[m]) - npz_seeds)[:5]}")
        checks.append(f"{m}: NPZ rows of this skip_mode lie inside its bidirectional set, "
                      f"and every seed of the set has an NPZ row")
    if all(bidi[m] is not None for m in prov["skip_modes_executed"]):
        union = set().union(*(set(bidi[m]) for m in prov["skip_modes_executed"]))
        if npz_seeds != union:
            raise ExtractRefused(
                f"NPZ seeds ({len(npz_seeds)}) are not the union of the bidirectional sets "
                f"({len(union)}): only NPZ {sorted(npz_seeds - union)[:5]}, only sets "
                f"{sorted(union - npz_seeds)[:5]}")
        checks.append(f"NPZ seeds == union of bidirectional sets ({len(union)})")

    os.makedirs(out_dir, exist_ok=True)
    try:
        cov, link, how = read_ledger(ledger_db, prov, out_dir, live_paths)
        inputs, geometry = CEO.build_inputs(params, prov, cov[2])
        credit = {
            "granted": True,
            "certified_interval": [int(cov[0]), int(cov[1])],
            "linked_interval": [int(link[0]), int(link[1])],
            "swept_intervals": swept,
        }
        checks.append(f"run ledger read: {how}; coverage {cov[3][:16]} and link bound to "
                      f"generation {prov['generation_id']}")
        out_npz = os.path.join(out_dir, "survivors.npz")
        shutil.copy2(npz_path, out_npz)
        if _sha256(out_npz) != prov["artifact_sha256"]:
            raise ExtractRefused("the copied NPZ does not hash to artifact_sha256")
        bundle = {
            "schema": CEO.SCHEMA,
            "engine": ENGINE,
            "run_id": prov.get("run_id"),
            "search_id": link[2],
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "artifact_sha256": prov["artifact_sha256"],
            "geometry": geometry,
            "provenance": {
                "extractor": "scripts/extract_pwc_bundle.py",
                "run_dir": _real(run_dir), "generation_dir": gen_dir,
                "generation_id": prov.get("generation_id"),
                "coverage_id": cov[3], "ledger_read": how,
                "pass_files": pass_files, "trial": trial,
                "pwc_completion_rebuilt": comp, "self_checks": checks,
            },
            "inputs": inputs,
            "phases": phases,
            "bidirectional": bidi,
            "phase_counts": counts,
            "credit": credit,
            "npz": "survivors.npz",
        }
        path = os.path.join(out_dir, "bundle.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=1, sort_keys=True)
    except BaseException:
        for fn in ("survivors.npz", "bundle.json"):
            if os.path.exists(os.path.join(out_dir, fn)):
                os.remove(os.path.join(out_dir, fn))
        raise
    return path, bundle


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True, help="the separate tree the PWC run ran in")
    ap.add_argument("--params-json", required=True, help="the WATCHER --params JSON the run used")
    ap.add_argument("--out-dir", required=True, help="new, empty directory for the bundle")
    ap.add_argument("--trial", type=int, default=0, help="trial number (default 0)")
    ap.add_argument("--generation-dir", help="default <run-dir>/.s172_accumulator/current")
    ap.add_argument("--results-dir", help="default <run-dir>/results")
    ap.add_argument("--ledger-db", help="default <run-dir>/prng_analysis.db")
    a = ap.parse_args(argv)
    try:
        path, b = extract(a.run_dir, a.params_json, a.out_dir, a.trial, a.generation_dir,
                          a.results_dir, a.ledger_db)
    except ExtractRefused as exc:
        print(f"extract_pwc_bundle: REFUSED: {exc}", file=sys.stderr)
        return 2
    nulls = [f"phases.{p}" for p, v in b["phases"].items() if v is None]
    print(f"BUNDLE_WRITTEN {path}")
    print(f"  generation {b['provenance']['generation_id']}  artifact {b['artifact_sha256'][:16]}")
    print(f"  phase rows {[(p, (c or {}).get('survivor_rows')) for p, c in b['phase_counts'].items()]}")
    print(f"  credit granted={b['credit']['granted']}  swept={b['credit']['swept_intervals']}")
    print(f"  null (will be NOT COMPARED unless the other bundle has them): {nulls or 'none'}")
    for c in b["provenance"]["self_checks"]:
        print(f"  check: {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
