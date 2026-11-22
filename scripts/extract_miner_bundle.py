#!/usr/bin/env python3
"""extract_miner_bundle.py: turn ONE finished RANGE-MINER Step 1 trial into an
`engine_output_bundle_v1` for scripts/compare_engine_outputs.py (Stage 4).

CPU only. Read-only on the run: the miner ledger and the run ledger are opened without
writing beside them (an immutable read, or a byte copy when a `-wal` is non-empty), the
certified NPZ is copied byte for byte, and only a NEW --out-dir (or a new receipt file)
is written. No GPU, no SSH, no engine run. Anchors for every source:
docs/drafts/pwc_backup_stage4_v1/EXTRACT_MINER_README.md.

Field sources are the ones agreed in `scripts/compare_engine_outputs.py`'s docstring;
`build_inputs` is imported from there, as `scripts/extract_pwc_bundle.py` imports it.

TWO STEPS (Beta X-PWC-STAGE4 R1 Q6: bind the miner trial, the publication and the params)
  1. --write-receipt PATH, right after the miner Step 1 exits and BEFORE the PWC arm
     starts. It runs every binding check below and records the identities it bound.
  2. --out-dir DIR --receipt PATH. It re-runs every binding check, refuses if any bound
     identity differs from the receipt, then writes the bundle.
  Nothing in the run records which miner trial produced which generation (the Step 1
  sweep evidence that names the miner run_id is held in memory only), so the receipt is
  the retained evidence. The binding checks below are what make a mixed-source pair fail
  even when the receipt itself is written for the wrong pair.

BINDING CHECKS (each is a refusal)
  B1 params: `use_range_miner` true; `staging_dir` present and equal to --staging-dir.
  B2 params pins (warm_start_window / _offset / _skip_min / _skip_max / _fwd_thresh /
     _rev_thresh) and `prng_type` equal the trial's recorded `trial_context`;
     `test_both_modes` matches the generation's executed skip modes. (The session pin
     `warm_start_session_idx` is an index into WATCHER's session options, which the run
     does not record beside it, so it is carried but not bound.)
  B3 every NPZ row carries this trial's trial_number, window_size, offset (= window
     anchor), skip_min and skip_max.
  B4 every NPZ row's forward_count / reverse_count equals the miner's own per-phase count
     for that row's skip mode (phases 1/2 constant, 3/4 variable).
  B5 every phase's swept interval equals the generation's [seed_start, seed_end_exclusive).
  B6 publication follows the trial: 0 <= sidecar created_at - trial finalized_at <= the
     trial's own duration (finalized_at - created_at). A generation published by a later
     run of the same job lags by at least that run's execution.
  B7 the certified_coverage row for this artifact carries THIS generation_id, the
     generation's interval and the trial's dataset_sha256.
  B8 the step1_chunk_link row for this generation carries that coverage_id and interval.

What a finished miner run does not keep is written as null (the comparator's null rule:
NOT COMPARED, never EQUAL): every `phases.N`, both `bidirectional` sets and
`phase_counts.N.distinct_survivors`. An empty list is never written for a missing piece.

Exit: 0 written; 2 refused (nothing written), with the reason on stderr.
"""
import argparse
import ast
import datetime
import hashlib
import importlib.util
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_spec = importlib.util.spec_from_file_location(
    "compare_engine_outputs", os.path.join(_ROOT, "scripts", "compare_engine_outputs.py"))
CEO = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CEO)
from utils.prng_encoding import SKIP_MODE_ENCODING                         # noqa: E402

ENGINE = "range_miner"
RECEIPT_SCHEMA = "stage4_miner_run_receipt_v1"
LEDGER_NAME = "miner_ledger.db"
RUN_LEDGER_NAME = "prng_analysis.db"
ACCUMULATOR_CURRENT = os.path.join(".s172_accumulator", "current")
SIDECAR = "provenance.json"
BINARY_NPZ = "bidirectional_survivors_binary.npz"
BUNDLE_NPZ = "survivors.npz"
MODE_PHASES = {"constant": (1, 2), "variable": (3, 4)}
LIVE_PATHS = (os.path.expanduser("~/distributed_prng_analysis"),
              os.path.expanduser("~/miner_staging"))
# params pin -> trial_context column (window_optimizer_bayesian.py warm start; B2)
PIN_TO_CONTEXT = {"warm_start_window": "window_size", "warm_start_offset": "window_anchor",
                  "warm_start_skip_min": "skip_min", "warm_start_skip_max": "skip_max",
                  "warm_start_fwd_thresh": "forward_threshold",
                  "warm_start_rev_thresh": "reverse_threshold"}
# NPZ array -> trial_context column (utils/canonical_records.py:215-228; B3)
NPZ_TO_CONTEXT = {"window_size": "window_size", "offset": "window_anchor",
                  "skip_min": "skip_min", "skip_max": "skip_max"}
# the receipt fields extraction must find unchanged
RECEIPT_BOUND = ("staging_dir", "output_root", "run_ledger", "params_json", "params_sha256",
                 "miner_run_id", "trial_number", "trial_created_at", "trial_finalized_at",
                 "miner_evidence_sha256", "generation_id", "generation_dir", "artifact_sha256",
                 "sidecar_sha256", "coverage_id", "link", "job_identity")


class ExtractRefused(Exception):
    """The run cannot be extracted honestly; nothing is written."""


def _real(p):
    return os.path.realpath(os.path.abspath(p))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _canon_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     default=list).encode()).hexdigest()


def _stat_key(path):
    st = os.stat(path)
    return st.st_size, st.st_mtime_ns


def refuse_live(path, what, live_paths):
    rp = _real(path)
    for lp in live_paths:
        lr = _real(lp)
        if rp == lr or rp.startswith(lr + os.sep):
            raise ExtractRefused(f"{what} {path} is under the live path {lp}; Stage 4 runs "
                                 f"and reads only a separate run tree")


def _inside(path, roots):
    return any(path == r or path.startswith(r + os.sep) for r in roots)


def _load_json(path, what):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        raise ExtractRefused(f"cannot read {what} {path}: {exc}") from exc


# --------------------------------------------------------------------------- sqlite
def read_sqlite(path, queries, scratch_parent, optional=()):
    """Run read-only queries without creating or changing anything beside `path`.

    Same rule as the Step 1 credit read (window_optimizer_integration_final.py:1576-1610):
    a non-empty -journal is refused; an empty or absent -wal allows an immutable read; a
    non-empty -wal means the database and its -wal are copied byte for byte (source
    unchanged across the copy) and the COPY is read. A query named in `optional` whose
    table does not exist returns None instead of refusing."""
    if not os.path.isfile(path):
        raise ExtractRefused(f"{path}: not found")
    jp, wp = path + "-journal", path + "-wal"
    if os.path.exists(jp) and os.path.getsize(jp) > 0:
        raise ExtractRefused(f"{jp} is non-empty: a rollback transaction is in flight")

    def run(con):
        out = {}
        for k, (sql, args) in queries.items():
            try:
                out[k] = con.execute(sql, args).fetchall()
            except sqlite3.OperationalError as exc:
                if k in optional and "no such table" in str(exc):
                    out[k] = None
                else:
                    raise ExtractRefused(f"{path}: {k}: {exc}") from exc
        return out

    if not (os.path.exists(wp) and os.path.getsize(wp) > 0):
        con = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
        try:
            return run(con)
        finally:
            con.close()
    tmp = tempfile.mkdtemp(prefix=".extract_sqlite_", dir=scratch_parent)
    try:
        before = (_stat_key(path), _stat_key(wp))
        dst = os.path.join(tmp, "copy.db")
        shutil.copyfile(path, dst)
        shutil.copyfile(wp, dst + "-wal")
        if (_stat_key(path), _stat_key(wp)) != before:
            raise ExtractRefused(f"{path} changed while it was being copied; the run is not finished")
        con = sqlite3.connect(dst)
        try:
            return run(con)
        finally:
            con.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def read_miner_ledger(staging_dir, run_id, scratch):
    db = os.path.join(staging_dir, LEDGER_NAME)
    trials = read_sqlite(db, {"t": ("SELECT run_id, trial_number, state, commit_delivery_status, "
                                    "created_at, finalized_at FROM trials ORDER BY run_id", ())},
                         scratch)["t"]
    if run_id is None:
        if len(trials) != 1:
            raise ExtractRefused(f"{db} holds {len(trials)} trial(s) {[t[0] for t in trials]}; "
                                 f"name one with --run-id")
        run_id = trials[0][0]
    trial = [t for t in trials if t[0] == run_id]
    if not trial:
        raise ExtractRefused(f"{db}: no trial with run_id {run_id!r}")
    t = trial[0]
    if t[2] != "committed" or t[3] != "done":
        raise ExtractRefused(f"trial {run_id} is state={t[2]!r} commit_delivery={t[3]!r}; "
                             f"only a committed, delivered trial is a finished run")
    rows = read_sqlite(db, {
        "ctx": ("SELECT trial_number, window_size, window_anchor_val, generator_phase, "
                "sessions_json, skip_min, skip_max, prng_base, forward_threshold, "
                "reverse_threshold, dataset_sha256, residue_sha256 FROM trial_context "
                "WHERE run_id=?", (run_id,)),
        "stripes": ("SELECT stripe_id, phase, seed_start, seed_count, state, current_attempt, "
                    "survivors_total FROM stripes WHERE run_id=? ORDER BY phase, seed_start",
                    (run_id,)),
        "shards": ("SELECT stripe_id, attempt, sub_index, worker_id, survivor_count, "
                   "local_cleanup_status FROM shards WHERE run_id=? "
                   "ORDER BY stripe_id, attempt, sub_index", (run_id,)),
        "plan": ("SELECT miner_stripe_size, macro_stripe_count FROM preflight_plans "
                 "WHERE run_id=?", (run_id,)),
    }, scratch)
    if not rows["ctx"]:
        raise ExtractRefused(f"{db}: no trial_context row for {run_id}; the trial's inputs "
                             f"are not recorded")
    c = rows["ctx"][0]
    ctx = {"trial_number": c[0], "window_size": c[1], "window_anchor": c[2],
           "generator_phase": c[3], "sessions": json.loads(c[4]) if c[4] is not None else None,
           "skip_min": c[5], "skip_max": c[6], "prng_base": c[7], "forward_threshold": c[8],
           "reverse_threshold": c[9], "dataset_sha256": c[10], "residue_sha256": c[11]}
    return {"db": db, "run_id": run_id, "trial": t, "ctx": ctx, "stripes": rows["stripes"],
            "shards": rows["shards"], "plan": rows["plan"]}


# --------------------------------------------------------------------------- generation
def read_generation(gen_dir):
    prov_path, npz_path = os.path.join(gen_dir, SIDECAR), os.path.join(gen_dir, BINARY_NPZ)
    if not os.path.isdir(gen_dir):
        raise ExtractRefused(f"{gen_dir}: no published generation (the finalizer never committed)")
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
                             f"{prov.get('artifact_sha256')} (truncated or modified)")
    try:
        with zipfile.ZipFile(npz_path) as z:
            order = [n[:-4] for n in z.namelist() if n.endswith(".npy")]
        with np.load(npz_path, allow_pickle=False) as d:
            arrays = {n: d[n] for n in d.files}
    except (OSError, ValueError, zipfile.BadZipFile, EOFError) as exc:
        raise ExtractRefused(f"cannot read {npz_path}: {exc}") from exc
    return prov, prov_path, npz_path, sha, order, arrays


def _parse_utc(text, what):
    try:
        return datetime.datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=datetime.timezone.utc).timestamp()
    except (TypeError, ValueError) as exc:
        raise ExtractRefused(f"{what} {text!r} is not a finalizer UTC timestamp") from exc


# --------------------------------------------------------------------------- derive
def swept_intervals(stripes, required):
    """Same rule as `_step1_miner_sweep_evidence` (window_optimizer_integration_final.py
    :1614-1628): a phase counts only if every stripe is `done` and the stripes tile one
    contiguous interval. Here a phase that fails is a refusal (the trial is not complete),
    as `extract_pwc_bundle.py` refuses an incomplete pass."""
    by_phase = {}
    for s in stripes:
        by_phase.setdefault(int(s[1]), []).append((int(s[2]), int(s[3]), s[4]))
    if sorted(by_phase) != sorted(required):
        raise ExtractRefused(f"stripe phases {sorted(by_phase)} != the phases of the "
                             f"generation's skip modes {sorted(required)}")
    swept = {}
    for ph, items in sorted(by_phase.items()):
        items.sort()
        if any(st != "done" for _, _, st in items):
            raise ExtractRefused(f"phase {ph}: a stripe is not done; the trial is not complete")
        cur = items[0][0]
        for s, n, _ in items:
            if s != cur:
                raise ExtractRefused(f"phase {ph}: stripes leave a gap or overlap at seed {cur}")
            cur = s + n
        swept[str(ph)] = [items[0][0], cur]
    return by_phase, swept


def phase_counts(stripes, shards, phases, na):
    """`{"survivor_rows": n, "distinct_survivors": None}` per phase. `survivor_rows` is the
    sum of the stripe rows' `survivors_total` (the reconciled count of each stripe's
    current attempt, range_miner_coordinator.py:1128-1170), cross-checked against that
    attempt's shard rows. A NULL or disagreeing count is null, never 0."""
    out = {}
    for ph in phases:
        st = [s for s in stripes if int(s[1]) == ph]
        bad = [s[0] for s in st if s[6] is None]
        mism = [s[0] for s in st if s[6] is not None and
                sum(int(sh[4]) for sh in shards if sh[0] == s[0] and sh[1] == s[5]) != int(s[6])]
        rows = None
        if bad:
            na(f"phase_counts.{ph}.survivor_rows", f"stripes {bad} have survivors_total NULL",
               ["miner_ledger.stripes"])
        elif mism:
            na(f"phase_counts.{ph}.survivor_rows",
               f"stripes {mism}: survivors_total != current-attempt shard sum",
               ["miner_ledger.stripes", "miner_ledger.shards"])
        else:
            rows = sum(int(s[6]) for s in st)
        na(f"phase_counts.{ph}.distinct_survivors",
           "the seeds are not kept, so distinct seeds cannot be counted",
           ["miner_ledger.shards", "NPZ"])
        out[str(ph)] = {"survivor_rows": rows, "distinct_survivors": None}
    return out


def read_skip_range_code(code_root, durable_switch=None):
    """Provenance only: which constant skip range this tree's code makes phases 1-2 run
    (F-1). The worker's `payload.get("skip_range", <default>)` literal, and whether the
    coordinator's payload builder names `skip_range` at all."""
    wpath = os.path.join(code_root, "miner", "range_miner_worker.py")
    cpath = os.path.join(code_root, "miner", "range_miner_coordinator.py")
    note = {"sha256": {}}
    default = delivered = None
    try:
        for node in ast.walk(ast.parse(Path(wpath).read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get" and len(node.args) == 2
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "skip_range"):
                default = ast.literal_eval(node.args[1])
                note["worker_default_line"] = f"miner/range_miner_worker.py:{node.lineno}"
        for node in ast.walk(ast.parse(Path(cpath).read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef) and node.name == "build_stripe_assign_payload":
                delivered = any(isinstance(n, ast.Constant) and n.value == "skip_range"
                                for n in ast.walk(node))
                # [SKIP-DELIVERY SWITCH] a keyword-only `skip_range` defaulting to None is
                # the switch (docs/drafts/miner_skip_switch_v1/PLAN.md), not an
                # unconditional delivery
                switch = any(a.arg == "skip_range" and isinstance(d, ast.Constant)
                             and d.value is None
                             for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults))
                note["payload_builder_line"] = f"miner/range_miner_coordinator.py:{node.lineno}"
        note["sha256"] = {"miner/range_miner_worker.py": _sha256(wpath),
                          "miner/range_miner_coordinator.py": _sha256(cpath)}
    except (OSError, SyntaxError, ValueError) as exc:
        note["error"] = str(exc)
    if default is None or delivered is None:
        note["effective"], note["why"] = None, "worker default or payload builder not found"
    elif delivered and switch:
        # the switch's state comes from the run's durable search record (B2b)
        if durable_switch is None:
            note["switch"] = "unverified"
            note["effective"], note["why"] = None, (
                "the builder carries the delivery switch and no durable record of its state "
                "was found (no search link for this generation); left unverified, never "
                "inferred from params")
        elif durable_switch:
            note["switch"] = "on"
            note["effective"], note["why"] = "requested", (
                "the run's search record carries the switch on: phases 1-2 run the trial's "
                "own range")
        else:
            note["switch"] = "off"
            note["effective"] = list(default)
            note["why"] = ("the run's search record carries the switch off: phases 1-2 run "
                           "the worker default")
    elif delivered:
        note["effective"], note["why"] = None, ("the payload builder names skip_range; which "
                                                "range this run delivered is not recorded")
    else:
        note["effective"] = list(default)
        note["why"] = "the payload carries no skip_range, so phases 1-2 run the worker default"
    return note


# --------------------------------------------------------------------------- evidence
def collect(staging_dir, output_root, params_json, run_id, ledger_db, live_paths, scratch,
            code_root):
    """Every read and every binding check (B1-B8). Returns the evidence both steps use."""
    staging_dir, output_root = _real(staging_dir), _real(output_root)
    gen_dir = _real(os.path.join(output_root, ACCUMULATOR_CURRENT))
    ledger_db = _real(ledger_db or os.path.join(output_root, RUN_LEDGER_NAME))
    params_json = _real(params_json)
    for p, what in ((staging_dir, "--staging-dir"), (output_root, "--output-root"),
                    (gen_dir, "generation"), (ledger_db, "run ledger")):
        refuse_live(p, what, live_paths)
    checks = []

    # ---- B1 params: a RANGE-MINER run whose staging_dir is this one
    params = _load_json(params_json, "params")
    if not isinstance(params, dict):
        raise ExtractRefused("params JSON is not an object")
    if params.get("use_range_miner") is not True or params.get("use_persistent_workers") is True:
        raise ExtractRefused(f"params are not a RANGE-MINER run (use_range_miner="
                             f"{params.get('use_range_miner')!r}, use_persistent_workers="
                             f"{params.get('use_persistent_workers')!r})")
    p_stage = params.get("staging_dir")
    if not isinstance(p_stage, str) or _real(p_stage) != staging_dir:
        raise ExtractRefused(f"B1: params staging_dir {p_stage!r} is not --staging-dir "
                             f"{staging_dir}: these params did not drive this miner ledger")

    led = read_miner_ledger(staging_dir, run_id, scratch)
    ctx, t = led["ctx"], led["trial"]

    # ---- B2 params pins == the trial's recorded inputs
    bad = {k: (params[k], ctx[c]) for k, c in PIN_TO_CONTEXT.items()
           if k in params and params[k] != ctx[c]}
    if "prng_type" in params and params["prng_type"] != ctx["prng_base"]:
        bad["prng_type"] = (params["prng_type"], ctx["prng_base"])
    if bad:
        raise ExtractRefused(f"B2: params pins differ from the trial's trial_context "
                             f"(pin: (params, trial)): {bad}")

    prov, prov_path, npz_path, sha, order, arrays = read_generation(gen_dir)
    modes = list(prov.get("skip_modes_executed") or [])
    if not modes or any(m not in MODE_PHASES for m in modes):
        raise ExtractRefused(f"provenance skip_modes_executed {modes!r} is not usable")
    required = [ph for m in modes for ph in MODE_PHASES[m]]
    req = [int(prov["seed_start"]), int(prov["seed_end_exclusive"])]

    # ---- B3 every NPZ row belongs to this trial
    tns = sorted({int(x) for x in arrays.get("trial_number", np.array([])).tolist()})
    if tns and tns != [int(t[1])]:
        raise ExtractRefused(f"B3: the NPZ carries trial_number(s) {tns}, not only this "
                             f"trial's {t[1]}: it is not this trial's output alone")
    for arr, col in NPZ_TO_CONTEXT.items():
        vals = sorted({int(x) for x in arrays.get(arr, np.array([])).tolist()})
        if vals and vals != [int(ctx[col])]:
            raise ExtractRefused(f"B3: NPZ {arr} values {vals} are not the trial's {col} "
                                 f"{ctx[col]}")

    if "test_both_modes" in params and bool(params["test_both_modes"]) != ("variable" in modes):
        raise ExtractRefused(f"B2: params test_both_modes {params['test_both_modes']!r} but the "
                             f"generation executed {modes}")
    by_phase, swept = swept_intervals(led["stripes"], required)
    na_list = []

    def na(piece, reason, sources):
        na_list.append({"piece": piece, "value": None, "reason": reason,
                        "sources_checked": list(sources)})

    counts = phase_counts(led["stripes"], led["shards"], required, na)

    # ---- B4 the NPZ's per-trial counts are the miner's own per-phase counts
    n_rows = int(arrays["seeds"].shape[0]) if "seeds" in arrays else 0
    for m in modes:
        f, r = MODE_PHASES[m]
        sel = arrays["skip_mode"] == SKIP_MODE_ENCODING[m] if n_rows else np.zeros(0, bool)
        for arr, ph in (("forward_count", f), ("reverse_count", r)):
            vals = sorted({int(x) for x in arrays[arr][sel].tolist()}) if n_rows else []
            mine = counts[str(ph)]["survivor_rows"]
            if vals and mine is not None and vals != [mine]:
                raise ExtractRefused(f"B4: NPZ {m} rows carry {arr} {vals}, but this miner "
                                     f"trial's phase {ph} returned {mine} rows")
    if not n_rows:
        checks.append("B3/B4 not applicable: the NPZ has no rows (zero bidirectional survivors)")

    # ---- B5 every phase swept exactly the generation's interval
    off = {ph: iv for ph, iv in swept.items() if iv != req}
    if off:
        raise ExtractRefused(f"B5: swept intervals {off} != the generation's {req}")

    # ---- B6 publication follows this trial's commit by less than the trial took
    pub = _parse_utc(prov.get("created_at"), "provenance created_at")
    created, finalized = t[4], t[5]
    if created is None or finalized is None:
        raise ExtractRefused(f"B6: trial {led['run_id']} has no created_at/finalized_at")
    lag, duration = pub - float(finalized), float(finalized) - float(created)
    if not 0 <= lag <= duration:
        raise ExtractRefused(
            f"B6: the generation was published {lag:.1f} s after this trial committed, outside "
            f"[0, {duration:.1f}] s (the trial's own duration): it is not this trial's "
            f"publication")

    # ---- B7/B8 credit rows bound to this generation
    lr = read_sqlite(ledger_db, {
        "cov": ("SELECT seed_start, seed_end_exclusive, dataset_sha256, coverage_id, "
                "generation_id FROM certified_coverage WHERE artifact_sha256=?", (sha,)),
        "link": ("SELECT seed_start, seed_end_exclusive, search_id, coverage_id FROM "
                 "step1_chunk_link WHERE generation_id=?", (prov.get("generation_id"),)),
        # [SKIP-DELIVERY SWITCH] the durable record of the switch: the linked search's
        # search space (utils/step1_search_ledger.presented_configuration)
        "space": ("SELECT s.search_space_json FROM step1_chunk_link l JOIN step1_search s "
                  "ON s.search_id = l.search_id WHERE l.generation_id=?",
                  (prov.get("generation_id"),)),
    }, scratch, optional=("link", "space"))    # no search tables: a run outside a search
    cov_rows, link_rows = lr["cov"] or [], lr["link"] or []
    if len(cov_rows) > 1 or len(link_rows) > 1:
        raise ExtractRefused(f"{len(cov_rows)} coverage rows / {len(link_rows)} link rows for "
                             f"one artifact; expected at most one each")
    cov = cov_rows[0] if cov_rows else None
    link = link_rows[0] if link_rows else None
    if cov is not None:
        if cov[4] != prov.get("generation_id"):
            raise ExtractRefused(f"B7: the certified_coverage row for this artifact names "
                                 f"generation {cov[4]!r}, not {prov.get('generation_id')!r} "
                                 f"(identical bytes from another generation)")
        if [int(cov[0]), int(cov[1])] != req:
            raise ExtractRefused(f"B7: certified interval {[cov[0], cov[1]]} != generation {req}")
        if cov[2] != ctx["dataset_sha256"]:
            raise ExtractRefused(f"B7: certified dataset_sha256 {cov[2]} != the trial's "
                                 f"{ctx['dataset_sha256']}")
    if link is not None:
        if cov is None:
            raise ExtractRefused("B8: a step1_chunk_link row names this generation but no "
                                 "certified_coverage row binds it")
        if link[3] != cov[3] or [int(link[0]), int(link[1])] != [int(cov[0]), int(cov[1])]:
            raise ExtractRefused(f"B8: link (coverage {link[3]!r}, {[link[0], link[1]]}) != "
                                 f"coverage ({cov[3]!r}, {[cov[0], cov[1]]})")

    # B2b [SKIP-DELIVERY SWITCH, Beta R1 finding 3] the switch comes from DURABLE run
    # evidence — the linked search record — never from the params file alone. Params that
    # disagree with the record are refused; with no record the effective range is left
    # unverified, never inferred from params.
    durable_switch = None
    space_rows = lr.get("space") or []
    if space_rows:
        durable_switch = json.loads(space_rows[0][0]).get("constant_skip_delivery") == "trial_range"
        if params.get("deliver_skip_range", False) is not durable_switch:
            raise ExtractRefused(
                f"B2b: params deliver_skip_range={params.get('deliver_skip_range', False)!r} but "
                f"the run's search record says the switch was "
                f"{'on' if durable_switch else 'off'}")
    skip_code = read_skip_range_code(code_root, durable_switch)
    if skip_code.get("effective") == "requested":
        skip_code["effective"] = [ctx["skip_min"], ctx["skip_max"]]
    if skip_code.get("effective") is not None and skip_code["effective"] != [ctx["skip_min"],
                                                                               ctx["skip_max"]]:
        checks.append(f"WARNING F-1: phases 1-2 ran the worker default {skip_code['effective']} "
                      f"(by this tree's code) while the trial requested [{ctx['skip_min']}, "
                      f"{ctx['skip_max']}]; a PWC run sends the requested range, so their "
                      f"outputs differ by design")
    elif skip_code.get("effective") is None:
        checks.append(f"WARNING F-1: the constant skip range phases 1-2 ran is not known "
                      f"({skip_code.get('why')})")
    if order != list(CEO.ARRAY_NAMES):
        checks.append("WARNING the NPZ's array order is not the frozen contract order; "
                      "copied as is, so the comparator reports it")

    # F2 job identity (PLAN.md §F2), from the miner's own records
    job = {
        "seed_range": req,
        "window_size": ctx["window_size"], "window_anchor": ctx["window_anchor"],
        "sessions": ctx["sessions"], "dataset_sha256": ctx["dataset_sha256"],
        "generator_phase": ctx["generator_phase"],
        "forward_threshold": ctx["forward_threshold"],
        "reverse_threshold": ctx["reverse_threshold"],
        "constant_skip_range_requested": [ctx["skip_min"], ctx["skip_max"]],
        "constant_skip_range_effective": skip_code.get("effective"),
        "strategies": None,                       # resolved on the worker, not recorded
        "phases": sorted(by_phase),
        "prng_family": ctx["prng_base"],
        "transport": None,                        # not applicable to the miner
    }
    evidence_rows = {"trial": list(t), "ctx": ctx, "stripes": [list(s) for s in led["stripes"]],
                     "shards": [list(s) for s in led["shards"]]}
    return {
        "staging_dir": staging_dir, "output_root": output_root, "run_ledger": ledger_db,
        "params_json": params_json, "params_sha256": _sha256(params_json), "params": params,
        "miner_run_id": led["run_id"], "trial_number": int(t[1]),
        "trial_created_at": float(created), "trial_finalized_at": float(finalized),
        "miner_evidence_sha256": _canon_sha(evidence_rows),
        "generation_id": prov.get("generation_id"), "generation_dir": gen_dir,
        "artifact_sha256": sha, "sidecar_sha256": _sha256(prov_path),
        "coverage_id": cov[3] if cov else None,
        "link": ({"search_id": link[2], "coverage_id": link[3],
                  "interval": [int(link[0]), int(link[1])]} if link else None),
        "job_identity": job, "publication_lag_s": lag,
        # not bound by the receipt
        "_led": led, "_prov": prov, "_npz_path": npz_path, "_order": order,
        "_by_phase": by_phase, "_swept": swept, "_counts": counts, "_cov": cov,
        "_na": na_list, "_checks": checks, "_skip_code": skip_code, "_modes": modes,
        "_required": required,
    }


# --------------------------------------------------------------------------- steps
def write_receipt(staging_dir, output_root, params_json, receipt_path, run_id=None,
                  ledger_db=None, live_paths=LIVE_PATHS, code_root=_ROOT):
    receipt_path = _real(receipt_path)
    if _inside(receipt_path, (_real(staging_dir), _real(output_root))):
        raise ExtractRefused(f"--write-receipt {receipt_path} is inside the run; the "
                             f"extractor never writes into the run")
    if os.path.exists(receipt_path):
        raise ExtractRefused(f"{receipt_path} already exists; a receipt is never overwritten")
    parent = os.path.dirname(receipt_path)
    os.makedirs(parent, exist_ok=True)
    ev = collect(staging_dir, output_root, params_json, run_id, ledger_db, live_paths, parent,
                 code_root)
    receipt = {"schema": RECEIPT_SCHEMA,
               "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               **{k: ev[k] for k in RECEIPT_BOUND}, "publication_lag_s": ev["publication_lag_s"],
               "binding_checks": "B1-B8 passed at capture"}
    with open(receipt_path, "x", encoding="utf-8") as f:
        json.dump(receipt, f, indent=1, sort_keys=True)
    return receipt_path, receipt


def build_bundle(staging_dir, output_root, params_json, out_dir, receipt_path, run_id=None,
                 ledger_db=None, live_paths=LIVE_PATHS, code_root=_ROOT):
    if not hasattr(CEO, "build_inputs"):
        raise ExtractRefused("scripts/compare_engine_outputs.py has no build_inputs(): it "
                             "predates the agreed field sources (Stage 4, 2026-11-21)")
    out_dir = _real(out_dir)
    if _inside(out_dir, (_real(staging_dir), _real(output_root))):
        raise ExtractRefused(f"--out-dir {out_dir} is inside the run; the extractor never "
                             f"writes into the run")
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise ExtractRefused(f"--out-dir {out_dir} exists and is not empty; a bundle is "
                             f"never overwritten")
    if not receipt_path:
        raise ExtractRefused("--receipt is required: nothing in the run binds the miner trial "
                             "to the generation, so the receipt written with --write-receipt "
                             "is the evidence (Beta X-PWC-STAGE4 R1 Q6)")
    receipt = _load_json(receipt_path, "receipt")
    if not isinstance(receipt, dict) or receipt.get("schema") != RECEIPT_SCHEMA:
        raise ExtractRefused(f"{receipt_path}: not a {RECEIPT_SCHEMA}")
    os.makedirs(out_dir, exist_ok=True)
    try:
        ev = collect(staging_dir, output_root, params_json, run_id, ledger_db, live_paths,
                     out_dir, code_root)
        diff = {k: (receipt.get(k), ev[k]) for k in RECEIPT_BOUND
                if json.loads(json.dumps(ev[k])) != receipt.get(k)}
        if diff:
            raise ExtractRefused(f"the run does not match its receipt {receipt_path} "
                                 f"(field: (receipt, now)): {diff}")
        return _write_bundle(ev, out_dir, _real(receipt_path))
    except BaseException:
        for fn in (BUNDLE_NPZ, "bundle.json"):
            if os.path.exists(os.path.join(out_dir, fn)):
                os.remove(os.path.join(out_dir, fn))
        raise


def _write_bundle(ev, out_dir, receipt_path):
    led, prov, cov = ev["_led"], ev["_prov"], ev["_cov"]
    t = led["trial"]
    na = list(ev["_na"])
    cleanup = sorted({sh[5] for sh in led["shards"]})
    why_phase = ("a finished run does not keep per-phase survivor sets: the four seed->rate "
                 "maps live only in the in-memory assembly, and the staged spools are deleted "
                 f"at commit (shards.local_cleanup_status seen: {cleanup})")
    phases = {}
    for ph in CEO.PHASES:
        phases[ph] = None
        if int(ph) in ev["_required"]:
            na.append({"piece": f"phases.{ph}", "value": None, "reason": why_phase,
                       "sources_checked": ["miner_ledger.shards", "staging_dir spools", "NPZ",
                                           "provenance.json"]})
    bidi = {}
    for m in CEO.BIDI_MODES:
        bidi[m] = None
        if m in ev["_modes"]:
            na.append({"piece": f"bidirectional.{m}", "value": None,
                       "reason": "the NPZ keeps one row per seed (utils/run_finalizer.py "
                                 "_select_l2_winners), so a seed in both sets keeps one "
                                 "skip_mode; the per-mode sets are not recoverable",
                       "sources_checked": ["NPZ"]})
    counts = {ph: None for ph in CEO.PHASES}
    counts.update(ev["_counts"])
    inputs, geometry = CEO.build_inputs(ev["params"], prov, cov[2] if cov else None)
    missing = [k for k in getattr(CEO, "F2_REQUIRED_INPUTS", ()) if inputs.get(k) is None]
    if missing:
        ev["_checks"].append(f"WARNING F2 job-identity inputs missing or null {missing}: the "
                             f"comparator reports NOT COMPARABLE (exit 2)")
    # granted: a coverage row bound to this generation AND artifact (B7), with a link row
    # bound to that coverage and interval (B8) -- the comparator's agreed definition
    credit = {"granted": cov is not None and ev["link"] is not None,
              "certified_interval": [int(cov[0]), int(cov[1])] if cov else None,
              "linked_interval": ev["link"]["interval"] if ev["link"] else None,
              "swept_intervals": ev["_swept"]}

    out_npz = os.path.join(out_dir, BUNDLE_NPZ)
    shutil.copyfile(ev["_npz_path"], out_npz)      # byte for byte; a re-save would hide an order
    if _sha256(out_npz) != ev["artifact_sha256"]:
        raise ExtractRefused("the copied NPZ does not hash to artifact_sha256")
    geometry = dict(geometry)
    geometry["miner_stripes_per_phase"] = {str(p): len(v) for p, v in sorted(ev["_by_phase"].items())}
    geometry["miner_preflight_plan"] = [list(r) for r in led["plan"]]
    logs_path = os.path.join(ev["staging_dir"], "threshold_provenance.json")
    bundle = {
        "schema": CEO.SCHEMA,
        "engine": ENGINE,
        "run_id": ev["miner_run_id"],
        "search_id": ev["link"]["search_id"] if ev["link"] else None,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "finished_at": t[5],
        "duration_s": ev["trial_finalized_at"] - ev["trial_created_at"],
        "worker_ids": sorted({sh[3] for sh in led["shards"]}),
        "job_ids": sorted(s[0] for s in led["stripes"]),
        "artifact_sha256": ev["artifact_sha256"],
        "geometry": geometry,
        "logs": [logs_path] if os.path.isfile(logs_path) else [],
        "provenance": {
            "extractor": "scripts/extract_miner_bundle.py",
            "receipt": receipt_path, "receipt_sha256": _sha256(receipt_path),
            "staging_dir": ev["staging_dir"], "miner_ledger": led["db"],
            "output_root": ev["output_root"], "generation_dir": ev["generation_dir"],
            "generation_id": ev["generation_id"], "sidecar_sha256": ev["sidecar_sha256"],
            "run_ledger": ev["run_ledger"], "coverage_id": ev["coverage_id"],
            "link": ev["link"], "params_json": ev["params_json"],
            "params_sha256": ev["params_sha256"], "trial_number": ev["trial_number"],
            "publication_lag_s": ev["publication_lag_s"],
            "miner_evidence_sha256": ev["miner_evidence_sha256"],
            "residue_sha256": led["ctx"]["residue_sha256"],
            "job_identity": ev["job_identity"],
            "constant_skip_range_code_reading": ev["_skip_code"],
            "npz_array_order_matches_contract": ev["_order"] == list(CEO.ARRAY_NAMES),
            "npz_array_order": ev["_order"],
            "binding_checks": "B1-B8 passed; receipt fields unchanged",
            "not_available": na, "self_checks": ev["_checks"],
        },
        "inputs": inputs,
        "phases": phases,
        "bidirectional": bidi,
        "phase_counts": counts,
        "credit": credit,
        "npz": BUNDLE_NPZ,
    }
    path = os.path.join(out_dir, "bundle.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=1, sort_keys=True)
    return path, bundle


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--staging-dir", required=True,
                    help="the run's coordinator staging_dir (holds miner_ledger.db)")
    ap.add_argument("--output-root", required=True,
                    help="the dir finalize_run published into (the Step 1 CWD)")
    ap.add_argument("--params-json", required=True, help="the WATCHER --params JSON the run used")
    step = ap.add_mutually_exclusive_group(required=True)
    step.add_argument("--write-receipt", metavar="PATH",
                      help="step 1: bind and record the run (right after the miner run)")
    step.add_argument("--out-dir", help="step 2: new, empty directory for the bundle")
    ap.add_argument("--receipt", help="step 2: the receipt written in step 1 (required)")
    ap.add_argument("--run-id", help="miner trial run_id (required if the ledger holds several)")
    ap.add_argument("--ledger-db", help="default <output-root>/prng_analysis.db")
    a = ap.parse_args(argv)
    try:
        if a.write_receipt:
            path, r = write_receipt(a.staging_dir, a.output_root, a.params_json,
                                    a.write_receipt, run_id=a.run_id, ledger_db=a.ledger_db)
            print(f"RECEIPT_WRITTEN {path}")
            print(f"  miner run {r['miner_run_id']}  generation {r['generation_id']}  "
                  f"artifact {r['artifact_sha256'][:16]}  lag {r['publication_lag_s']:.1f} s")
            return 0
        path, b = build_bundle(a.staging_dir, a.output_root, a.params_json, a.out_dir,
                               a.receipt, run_id=a.run_id, ledger_db=a.ledger_db)
    except ExtractRefused as exc:
        print(f"extract_miner_bundle: REFUSED: {exc}", file=sys.stderr)
        return 2
    print(f"BUNDLE_WRITTEN {path}")
    print(f"  generation {b['provenance']['generation_id']}  artifact {b['artifact_sha256'][:16]}")
    print(f"  phase rows {[(p, (c or {}).get('survivor_rows')) for p, c in b['phase_counts'].items()]}")
    print(f"  credit granted={b['credit']['granted']}  swept={b['credit']['swept_intervals']}")
    for e in b["provenance"]["not_available"]:
        print(f"  null (NOT COMPARED): {e['piece']}: {e['reason']}")
    for c in b["provenance"]["self_checks"]:
        print(f"  check: {c}", file=sys.stderr if c.startswith("WARNING") else sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
