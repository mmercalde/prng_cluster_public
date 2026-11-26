#!/usr/bin/env python3
"""
tests/test_ll_feature_settings.py - self-test for Ch14 learning loop wave 1 lane W1-a:
  LL-1 feature manifest + Step 6 schema check     (models/feature_manifest.py, prediction_generator.py)
  LL-2 per-seed sieve features + shared merge list (survivor_scorer.py, full_scoring_worker.py)
  LL-3 Step 5 feature list as a setting            (meta_prediction_optimizer_anti_overfit.py)

CPU only, small synthetic fixtures, plus small run H fixtures under tests/fixtures/ (the
91 Step 3 feature names, the winner sidecar's feature_schema block, the leak-gate settings;
each records the sha256 of the run H file it was cut from; no draw data). --runh-root DIR
re-derives them from the live run H files and fails on any difference (optional). The run H
dataset itself is read by path (--runh-dataset), checked against the fixture's sha256.
The baseline is the pre-LL code at a FIXED revision (--baseline-rev, default 50cd683, the
main commit before the LL landing), not HEAD: after the landing HEAD has the switches and the
default-identity gates would compare the new code with itself (lead TASK_LL_SWITCHES_ADDENDUM_1). Every gate is a
function returning (ok, detail). Every mutant is applied, proven to have run
(call counter > 0), and must make its target gate FAIL.

Writes only under --out (must be inside <worktree>/data/). Multi-core: the
Step 3 cases run in a process pool (--workers).
"""
import argparse
import copy
import hashlib
import json
import os
import random
import resource
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

FIXTURES = REPO / "tests" / "fixtures"
FX_NAMES = FIXTURES / "runH_step3_feature_names.txt"        # 91 names, sorted
FX_SCHEMA = FIXTURES / "runH_winner_feature_schema.json.txt"  # {"source_sha256", "feature_schema"}
FX_LEAK = FIXTURES / "runH_leak_inputs.json.txt"             # window settings, train length, source sha256s
# The pre-LL baseline: main at the commit before the LL landing (full sha). Override: --baseline-rev.
BASELINE_REV = "50cd6837abda2821277fc1cf6cc5f2d0e8f35a9a"
DEFAULT_RUNH_DATASET = Path.home() / "skipsw_run_thr3030/daily3.json"


def _sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fixture_names():
    return [l for l in FX_NAMES.read_text().splitlines() if l and not l.startswith("#")]


def fixture_schema():
    return json.loads(FX_SCHEMA.read_text())["feature_schema"]


def fixture_leak():
    return json.loads(FX_LEAK.read_text())

RESULTS = []


def gate(name, ok, detail=""):
    RESULTS.append({"gate": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    return ok


def sha(b):
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def make_fixture(d: Path, seed=7, n=40, drop_sieve_for_one=False, holdout_seed=None):
    d.mkdir(parents=True)
    r = random.Random(seed)
    chunk = []
    for i in range(n):
        row = {"seed": r.randrange(2 ** 32),
               "forward_matches": r.randrange(4, 9) / 12, "reverse_matches": r.randrange(4, 8) / 12,
               "forward_count": float(r.randrange(50, 150)), "reverse_count": float(r.randrange(50, 150)),
               "bidirectional_count": float(r.randrange(5, 20)),
               "intersection_count": float(r.randrange(5, 20)), "intersection_ratio": r.random(),
               "intersection_weight": r.random(), "bidirectional_selectivity": r.random(),
               "forward_only_count": float(r.randrange(40, 90)), "reverse_only_count": float(r.randrange(40, 90)),
               "survivor_overlap_ratio": r.random(), "skip_min": 0, "skip_max": 16, "skip_range": 16,
               "score": 0.4, "window_size": 12, "offset": 25, "trial_number": 1,
               "skip_mode": "constant", "prng_type": "java_lcg"}
        chunk.append(row)
    if drop_sieve_for_one:
        del chunk[3]["forward_matches"]
    json.dump(chunk, open(d / "chunk.json", "w"))
    json.dump([r.randrange(1000) for _ in range(300)], open(d / "train.json", "w"))
    hr = random.Random(holdout_seed if holdout_seed is not None else seed + 1)
    json.dump([hr.randrange(1000) for _ in range(60)], open(d / "holdout.json", "w"))
    return chunk


def run_case(out: Path, name, code_dir, fixture, settings=None, force_fallback=False, mutant=None):
    o = out / "step3" / f"{name}.json"
    cmd = [sys.executable, str(REPO / "tests/ll_step3_runner.py"), "--code-dir", str(code_dir),
           "--repo-root", str(REPO), "--fixture", str(fixture),
           "--settings", json.dumps(settings or {}), "--out", str(o)]
    if force_fallback:
        cmd.append("--force-fallback")
    if mutant:
        cmd += ["--mutant", mutant]
    p = subprocess.run(cmd, capture_output=True, text=True)
    (out / "step3" / f"{name}.log").write_text(p.stdout + p.stderr)
    if p.returncode != 0:
        return {"name": name, "rc": p.returncode, "raised": "runner exit " + str(p.returncode),
                "results": None, "calls": {}}
    d = json.load(open(o))
    d["name"] = name
    d["rc"] = 0
    return d


def rows_bytes(d):
    return json.dumps(d["results"], sort_keys=True).encode()


# ---------------------------------------------------------------------------
# LL-2 gates (on runner outputs)
# ---------------------------------------------------------------------------
def g_identical(a, b):
    if a["results"] is None or b["results"] is None:
        return False, f"raised a={a.get('raised')} b={b.get('raised')}"
    ok = rows_bytes(a) == rows_bytes(b)
    return ok, f"{sha(rows_bytes(a))[:12]} vs {sha(rows_bytes(b))[:12]}"


def g_on_adds_exactly_two(off, on, chunk):
    if on["results"] is None:
        return False, f"raised {on.get('raised')}"
    meta = {c["seed"]: c for c in chunk}
    for r0, r1 in zip(off["results"], on["results"]):
        f0, f1 = r0["features"], r1["features"]
        if set(f1) - set(f0) != {"forward_matches", "reverse_matches"} or set(f0) - set(f1):
            return False, f"name sets differ wrongly for seed {r0['seed']}"
        for k in f0:
            if f0[k] != f1[k]:
                return False, f"{k} changed for seed {r0['seed']}"
        m = meta[r1["seed"]]
        if f1["forward_matches"] != float(m["forward_matches"]) or \
           f1["reverse_matches"] != float(m["reverse_matches"]):
            return False, f"sieve values != NPZ metadata for seed {r1['seed']}"
        for k in ("holdout_hits", "holdout_quality"):
            if r0.get(k) != r1.get(k):
                return False, f"label {k} changed"
        st = r1.get("feature_settings", {})
        if st.get("feature_schema_version") != "v2" or "feature_settings" in r0:
            return False, f"stamp wrong: {st}"
    return True, f"{len(on['results'])} rows, 2 added names equal NPZ values, 91 others unchanged"


def g_fparity(batch, fb, fields):
    if batch["results"] is None or fb["results"] is None:
        return False, f"raised b={batch.get('raised')} f={fb.get('raised')}"
    if fb["calls"].get("batch_forced_raise", 0) < 1:
        return False, "fallback case did not take the fallback path"
    if any("worker_gpu" not in r["metadata"] for r in batch["results"]):
        return False, "batch case took the fallback path"
    worst = 0.0
    ndiff = 0
    for rb, rf in zip(batch["results"], fb["results"]):
        fb_, ff = rb["features"], rf["features"]
        if set(fb_) != set(ff):
            return False, f"name sets differ: only batch {sorted(set(fb_)-set(ff))[:5]}, only fallback {sorted(set(ff)-set(fb_))[:5]}"
        for k in fields:
            if fb_.get(k) != ff.get(k):
                return False, f"metadata-backed {k} differs: {fb_.get(k)} vs {ff.get(k)}"
        for k in fb_:
            if k in fields:
                continue
            a, b = fb_[k], ff[k]
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                dlt = abs(float(a) - float(b))
                if dlt > 1e-6:
                    ndiff += 1
                worst = max(worst, dlt)
    return True, (f"names equal; {len(fields)} metadata-backed fields exactly equal; "
                  f"computed features: {ndiff} values differ by >1e-6, max |diff| {worst:.3g} (measured, see note)")


def g_legacy_zeroes_seven(batch, fb_legacy):
    expect = {"intersection_count", "intersection_ratio", "intersection_weight",
              "bidirectional_selectivity", "forward_only_count", "reverse_only_count",
              "survivor_overlap_ratio"}
    seen = set()
    for rb, rf in zip(batch["results"], fb_legacy["results"]):
        for k in expect | {"forward_count", "reverse_count", "bidirectional_count",
                           "skip_min", "skip_max", "skip_range"}:
            if rb["features"].get(k) != rf["features"].get(k):
                seen.add(k)
    zero = all(rf["features"].get(k, 0.0) == 0.0 for rf in fb_legacy["results"] for k in expect)
    return seen == expect and zero, f"differing metadata fields {sorted(seen)}; all zero in fallback={zero}"


def g_fail_closed(case):
    r = case.get("raised") or ""
    ok = case["results"] is None and "per_seed_sieve_features='on'" in r and "ValueError" in r
    return ok, r[:120]


# ---------------------------------------------------------------------------
# LL-1 gates
# ---------------------------------------------------------------------------
def g_manifest_rebuild(fm, manifest_path):
    names = fixture_names()
    built = fm.build_manifest(names, {})
    have = json.load(open(manifest_path))
    have = {k: v for k, v in have.items() if k != "source"}
    built = {k: v for k, v in built.items() if k != "source"}
    return built == have, f"rebuilt from {len(names)} run H names (fixture) vs {manifest_path.name}"


def g_manifest_vs_sidecar(fm, manifest_path):
    m = fm.load_manifest(manifest_path)
    side = fixture_schema()
    v1 = m["versions"]["v1"]
    ok = (v1["trained_by_default"] == side["feature_names"]
          and v1["trained_by_default_hash"] == side["feature_schema_hash"]
          and v1["extracted_count"] == 91 and v1["trained_by_default_count"] == 89
          and m["versions"]["v2"]["extracted_count"] == 93)
    return ok, (f"v1 {v1['extracted_count']}/{v1['trained_by_default_count']} hash "
                f"{v1['trained_by_default_hash']} vs sidecar {side['feature_schema_hash']}")


def g_manifest_vs_code(fm, manifest_path):
    import survivor_scorer as ss
    from models.global_state_tracker import GLOBAL_FEATURE_NAMES
    m = fm.load_manifest(manifest_path)
    g = m["groups"]
    ok = (set(g["npz_metadata"]) | set(g["dead"]) == set(ss.METADATA_MERGE_FIELDS)
          and len(ss.METADATA_MERGE_FIELDS) == 18
          and set(g["dead"]) == set(ss.DEAD_PLACEHOLDER_FIELDS)
          and set(g["per_seed_sieve"]) == set(ss.PER_SEED_SIEVE_FIELDS)
          and set(g["global"]) == {"global_" + n for n in GLOBAL_FEATURE_NAMES}
          and set(ss.LEGACY_FALLBACK_MERGE_FIELDS) < set(ss.METADATA_MERGE_FIELDS))
    return ok, {k: len(v) for k, v in g.items()}


def g_schema_check(fm):
    names = ["a", "b", "c"]
    h = fm.schema_hash(names)

    def boom():
        raise AssertionError("rows iterated in off mode")
        yield

    try:
        r = fm.check_rows_against_schema(names, h, boom(), mode="off")
        if r != {"mode": "off"}:
            return False, "off returned a report"
    except AssertionError as e:
        return False, str(e)
    good = [{"features": {"a": 1, "b": 2, "c": 3, "extra": 0}}] * 3
    bad = good[:2] + [{"features": {"a": 1, "b": 2}}]
    renamed = [{"features": {"a": 1, "b": 2, "c_renamed": 3}}]
    if fm.check_rows_against_schema(names, h, iter(good), mode="fail")["problems"]:
        return False, "clean rows reported a problem"
    w = fm.check_rows_against_schema(names, h, iter(bad), mode="warn")
    if w["rows_missing"] != 1 or w["missing_names"] != {"c": 1}:
        return False, f"warn report wrong {w}"
    for label, rows, hh in (("missing", bad, h), ("renamed", renamed, h), ("bad hash", good, "0" * 16)):
        try:
            fm.check_rows_against_schema(names, hh, iter(rows), mode="fail")
            return False, f"fail mode accepted {label}"
        except fm.ManifestError:
            pass
    return True, "off: rows untouched; warn: reports; fail: refuses missing, renamed, bad hash"


def g_step6_wiring(fm):
    """Drive PredictionGenerator._build_prediction_pool on a stub: 'fail' must refuse a
    row missing a sidecar name before X is built; 'off' must not run the check."""
    import prediction_generator as pg

    class Stub:
        pass

    def mk(mode):
        s = Stub()
        s.config = pg.PredictionConfig()
        s.config.schema_check = mode
        s.feature_names = ["a", "b"]
        s.model_meta = {"feature_schema": {"feature_names": ["a", "b"],
                                           "feature_schema_hash": fm.schema_hash(["a", "b"])}}
        import logging
        s.logger = logging.getLogger("stub")
        s.model = None
        s._get_survivor_skip = lambda sv, fb: 0
        s._empty_pool_result = lambda: {"empty": True}
        return s

    rows = [{"seed": 1, "features": {"a": 1.0}}]
    s = mk("fail")
    try:
        pg.PredictionGenerator._build_prediction_pool(s, rows, [], 5)
        return False, "fail mode did not refuse"
    except fm.ManifestError:
        pass
    except Exception as e:  # went past the check into the stubbed generator
        return False, f"fail mode did not refuse (reached {type(e).__name__})"
    s = mk("off")
    try:
        pg.PredictionGenerator._build_prediction_pool(s, rows, [], 5)
    except fm.ManifestError:
        return False, "off mode ran the check"
    except Exception:
        pass  # the stub is not a full generator; only the check is under test
    if hasattr(s, "schema_check_report"):
        return False, "off mode produced a report"
    d = pg.PredictionConfig().to_dict()
    if "schema_check" in d or pg.PredictionConfig().schema_check != "off":
        return False, "default config changed"
    return True, "fail refuses before X; off skips; default config dict unchanged"


# ---------------------------------------------------------------------------
# leak check: the per-seed sieve values come from a window inside train history
# ---------------------------------------------------------------------------
def sieve_window_indices(dataset, sessions, window_size, anchor):
    idx = [i for i, e in enumerate(dataset) if not sessions or e.get("session") in sessions]
    return idx[anchor:anchor + window_size]


def g_leak_runH(dataset_path):
    """The per-seed sieve window lies inside run H's train history. The window settings and the
    train length come from the fixture; the dataset is read by path and must have the fixture's
    sha256. The train file's own prefix check is part of --runh-root (live), since the fixture
    holds no draws."""
    from miner.range_miner_worker import load_residue_window
    L = fixture_leak()
    cfg, n_train = L["window_config"], L["train_len"]
    if not Path(dataset_path).is_file():
        return False, f"run H dataset missing: {dataset_path} (--runh-dataset)"
    if _sha_file(dataset_path) != L["dataset_sha256"]:
        return False, f"run H dataset {dataset_path} sha256 != fixture {L['dataset_sha256'][:12]}"
    data = json.load(open(dataset_path))
    idx = sieve_window_indices(data, cfg["sessions"], cfg["window_size"], cfg["offset"])
    canon = load_residue_window(str(dataset_path), cfg["window_size"], cfg["sessions"], cfg["offset"])
    mine = [int(data[i].get("full_state", data[i]["draw"])) for i in idx]
    ok = mine == canon and max(idx) < n_train
    return ok, (f"window W{cfg['window_size']} anchor {cfg['offset']} {cfg['sessions']} -> combined "
                f"indices {idx[0]}..{idx[-1]}; train = first {n_train} draws (fixture); "
                f"index map == load_residue_window: {mine == canon}")


def g_runh_live(root: Path, dataset_path):
    """--runh-root: re-derive every fixture from the live run H files; any difference fails."""
    root = Path(root)
    L = fixture_leak()
    s3 = root / "step3_full/survivors_with_scores.json"
    side = root / "step45/winner/best_model.meta.json"
    tr = root / "copy/train_history.json"
    probs = []
    for p in (s3, side, tr):
        if not p.is_file():
            probs.append(f"missing {p}")
    if probs:
        return False, "; ".join(probs)
    head = FX_NAMES.read_text().splitlines()[1]
    if _sha_file(s3) not in head:
        probs.append("step3 file sha256 != the one the names were cut from")
    if sorted(json.load(open(s3))[0]["features"]) != fixture_names():
        probs.append("names differ")
    if _sha_file(side) != json.loads(FX_SCHEMA.read_text())["source_sha256"] or \
            json.load(open(side))["feature_schema"] != fixture_schema():
        probs.append("sidecar feature_schema differs")
    train = json.load(open(tr))
    if _sha_file(tr) != L["train_history_sha256"] or len(train) != L["train_len"]:
        probs.append("train history differs")
    data = json.load(open(dataset_path))
    if [int(e["draw"]) for e in data[:len(train)]] != [int(x) for x in train]:
        probs.append("train history is not the dataset's prefix")
    # Beta X-LL-SWITCHES R1 Q3 (lead ADDENDUM_2 Fix 3): the window settings the leak gate takes
    # from the fixture must be run H's own window config, by sha256 and by projected settings
    wc = root / "copy/configs/optimal_window_config.json"
    if not wc.is_file():
        probs.append(f"missing {wc}")
    else:
        if _sha_file(wc) != L["window_config_source_sha256"]:
            probs.append("window config sha256 != the fixture's source")
        cfg = json.load(open(wc))
        proj = {k: cfg.get(k) for k in L["window_config"]}
        if proj != L["window_config"]:
            probs.append(f"window config settings {proj} != fixture {L['window_config']}")
    return not probs, ("live run H files equal the fixtures (names, sidecar schema, train history, window "
                       "config sha256 and settings); train is the dataset prefix"
                       if not probs else "; ".join(probs))


def g_runh_live_catches_window_change(root: Path, dataset_path, work: Path):
    """Negative check of the window-config comparison: a run H tree whose only change is the
    window config's offset must fail g_runh_live. The other run H files are symlinked, read only."""
    root = Path(root)
    t = work / "runH_window_changed"
    for rel in ("step3_full/survivors_with_scores.json", "step45/winner/best_model.meta.json",
                "copy/train_history.json"):
        (t / rel).parent.mkdir(parents=True, exist_ok=True)
        os.symlink(root / rel, t / rel)
    cfg = json.load(open(root / "copy/configs/optimal_window_config.json"))
    cfg["offset"] = cfg["offset"] + 1
    (t / "copy/configs").mkdir(parents=True)
    (t / "copy/configs/optimal_window_config.json").write_text(json.dumps(cfg))
    ok, det = g_runh_live(t, dataset_path)
    return (not ok and "window config" in det), f"changed offset -> live gate {'fails' if not ok else 'PASSES'}: {det[:160]}"


def g_leak_detects_crossing():
    data = [{"session": "midday" if i % 2 else "evening", "draw": i} for i in range(100)]
    idx = sieve_window_indices(data, ["midday"], 10, 40)
    return max(idx) >= 60, f"a window at midday anchor 40 reaches combined index {max(idx)} (>= 60: caught)"


# ---------------------------------------------------------------------------
# LL-3 gates
# ---------------------------------------------------------------------------
def make_step5_file(path, case):
    rows = []
    for r in case["results"]:
        rows.append({"seed": r["seed"], "features": r["features"], "holdout_hits": r["holdout_hits"],
                     "holdout_quality": r["holdout_quality"], "score": r["score"]})
    json.dump(rows, open(path, "w"))


def load_data_with(mod, survivors_file, settings):
    import logging
    cls = mod.AntiOverfitMetaOptimizer
    o = object.__new__(cls)
    o.survivors_file = str(survivors_file)
    o.logger = logging.getLogger("ll3")
    o.lottery_history = list(range(300))
    o.holdout_draws = 60
    o.prng_type = "java_lcg"
    o.mod = 1000
    if settings is not None:
        o._ll3_settings = settings
    o._load_data()
    return o


def g_step5_default_identical(new_mod, base_mod, f):
    a = load_data_with(new_mod, f, None)
    a2 = load_data_with(new_mod, f, (None, None, None))
    b = load_data_with(base_mod, f, None)
    ok = (a.X.tobytes() == b.X.tobytes() == a2.X.tobytes()
          and json.dumps(a.feature_schema, sort_keys=True) == json.dumps(b.feature_schema, sort_keys=True)
          == json.dumps(a2.feature_schema, sort_keys=True)
          and a.y.tobytes() == b.y.tobytes())
    return ok, f"X {a.X.shape}, schema hash {a.feature_schema['feature_schema_hash']} vs base {b.feature_schema['feature_schema_hash']}"


def g_step5_group(new_mod, f, settings, expect_dropped, expect_kept):
    o = load_data_with(new_mod, f, settings)
    names = o.feature_schema["feature_names"]
    sel = o.feature_schema.get("feature_selection") or {}
    ok = (len(names) == expect_kept and o.X.shape[1] == expect_kept
          and set(sel.get("dropped_by_groups", [])) == set(expect_dropped)
          and not (set(names) & set(expect_dropped)))
    return ok, f"kept {len(names)} (want {expect_kept}), dropped {len(sel.get('dropped_by_groups', []))} (want {len(expect_dropped)})"


def g_step5_refusals(fm, names):
    bad = []
    for label, kw in (("unknown group", {"exclude_groups": ["nope"]}),
                      ("unknown column", {"exclude_groups": ["global"], "_extra": True}),
                      ("empty exclude", {"exclude_features": []}),
                      ("inc and exc overlap", {"include_groups": ["global"], "exclude_groups": ["global"]})):
        kw = dict(kw)
        nm = list(names) + (["mystery_col"] if kw.pop("_extra", False) else [])
        try:
            fm.resolve_step5_exclusions(nm, **kw)
            bad.append(label)
        except fm.ManifestError:
            pass
    return not bad, f"accepted: {bad}" if bad else "all four refused"


# ---- LL-3 refusals through the REAL optimizer CLI (lead TASK_LL_SWITCHES_ADDENDUM_2 Fix 1;
# Beta X-LL-SWITCHES R1 Q2). Each case: argv, text the refusal must contain. <ALL91> = every
# feature name of the fixture; <ALL95> = those plus the four default exclusions.
CLI_REFUSALS = [
    ("include_groups_blank", ["--include-feature-groups", ""], "--include-feature-groups was given an empty value"),
    ("exclude_groups_comma", ["--exclude-feature-groups", ","], "--exclude-feature-groups was given an empty value"),
    ("exclude_features_blank", ["--exclude-features", ""], "--exclude-features was given an empty value"),
    ("exclude_confidnce", ["--exclude-features", "confidnce"], "not among the survivors file's features: ['confidnce']"),
    ("exclude_4_plus_confidnce", ["--exclude-features", "score,confidence,holdout_hits,holdout_quality,confidnce"],
     "not among the survivors file's features: ['confidnce']"),
    ("exclude_all_91", ["--exclude-features", "<ALL91>"], "omits the default exclusion(s) ['holdout_hits', 'holdout_quality']"),
    ("exclude_three_of_four", ["--exclude-features", "score,confidence,holdout_hits"], "omits the default exclusion(s) ['holdout_quality']"),
    ("exclude_all_91_and_4", ["--exclude-features", "<ALL95>"], "leaves no feature to train on"),
    ("allow_flag_alone", ["--allow-target-columns-as-features"], "has no effect without --exclude-features"),
]
# One mutant per check, applied to a copy of models/feature_manifest.py in a scratch tree; each must
# make the named CLI case stop refusing (killed = that case's gate fails under the mutant).
CLI_MUTANTS = [
    ("empty_value_accepted (c)", "if v is not None and not [x for x in v if str(x).strip()]:", "if False:", "include_groups_blank"),
    ("unknown_name_accepted (a)", "unknown = [n for n in exclude_features if n not in known]", "unknown = []", "exclude_4_plus_confidnce"),
    ("default_exclusion_dropped (b)", "if missing and not allow_target_columns:", "if False:", "exclude_three_of_four"),
    ("nothing_left_accepted (d)", "if not [n for n in data_feature_names if n not in base]:", "if False:", "exclude_all_91_and_4"),
    ("allow_alone_accepted", "if allow_target_columns and exclude_features is None:", "if False:", "allow_flag_alone"),
]


def mutant_tree(dst: Path, old: str, new: str) -> Path:
    """REPO entries symlinked, except the optimizer (copied, so sys.path[0] is this tree) and
    models/feature_manifest.py (mutated copy)."""
    dst.mkdir(parents=True)
    for e in REPO.iterdir():
        if e.name in ("models", "data", ".git", "__pycache__", "meta_prediction_optimizer_anti_overfit.py"):
            continue
        os.symlink(e, dst / e.name)
    shutil.copyfile(REPO / "meta_prediction_optimizer_anti_overfit.py", dst / "meta_prediction_optimizer_anti_overfit.py")
    (dst / "models").mkdir()
    for e in (REPO / "models").iterdir():
        if e.name in ("feature_manifest.py", "__pycache__"):
            continue
        os.symlink(e, dst / "models" / e.name)
    src = (REPO / "models" / "feature_manifest.py").read_text()
    if src.count(old) != 1:
        raise SystemExit(f"mutant text not found exactly once: {old!r}")
    (dst / "models" / "feature_manifest.py").write_text(src.replace(old, new))
    return dst


def run_step5_cli(tree: Path, d: Path, s5: Path, lottery: Path, extra):
    """The real optimizer CLI on the Step 5 fixture (CPU, lightgbm, 1 trial, 2 folds). Its study
    storage, model dir and log are inside d."""
    d.mkdir(parents=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", S88_FORCE_LGBM_CPU="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=str(tree))
    argv = [sys.executable, "-B", str(tree / "meta_prediction_optimizer_anti_overfit.py"),
            "--survivors", str(s5), "--lottery-data", str(lottery), "--model-type", "lightgbm",
            "--trials", "1", "--k-folds", "2", "--holdout-draws", "60",
            "--output-dir", str(d / "models"), "--study-name", "ll3_cli",
            "--storage", f"sqlite:///{d / 'optuna.db'}"] + list(extra)
    p = subprocess.run(argv, cwd=d, env=env, capture_output=True, text=True)
    (d / "cli.log").write_text(f"argv {argv}\nrc {p.returncode}\n--- stdout\n{p.stdout}\n--- stderr\n{p.stderr}")
    return p


def cli_extra(extra, names91):
    rep = {"<ALL91>": ",".join(names91), "<ALL95>": ",".join(sorted(set(names91) | {
        "score", "confidence", "holdout_hits", "holdout_quality"}))}
    return [rep.get(x, x) for x in extra]


def g_cli_refused(p, d: Path, want: str):
    ok = (p.returncode == 2 and "[LL-3] REFUSED" in p.stderr and want in p.stderr
          and not (d / "models").exists() and not (d / "optuna.db").exists())
    msg = next((l for l in p.stderr.splitlines() if "[LL-3] REFUSED" in l), p.stderr.strip()[-200:])
    return ok, f"rc {p.returncode}; model dir {'absent' if not (d / 'models').exists() else 'CREATED'}; " \
               f"study db {'absent' if not (d / 'optuna.db').exists() else 'CREATED'}; {msg[:180]}"


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--baseline-rev", default=BASELINE_REV,
                    help="revision whose Step 3/5 files are the pre-LL baseline (default: main before the LL landing)")
    ap.add_argument("--runh-root", default=None,
                    help="optional: run H tree (e.g. ~/tfm_worktrees/e2e-map/data/e2e_runH) to cross-check the fixtures live")
    ap.add_argument("--runh-dataset", default=str(DEFAULT_RUNH_DATASET),
                    help="run H dataset, read by path (checked against the fixture's sha256)")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if not str(out).startswith(str(REPO / "data") + "/"):
        raise SystemExit("--out must lie under <worktree>/data/")
    out.mkdir(parents=True)
    (out / "step3").mkdir()
    t0 = time.time()

    # baseline = the Step 3/5 files at the fixed pre-LL revision (not HEAD), nothing else
    base = out / "baseline_head"
    base.mkdir()
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", a.baseline_rev], capture_output=True,
                          text=True).stdout.strip()
    for f in ("survivor_scorer.py", "full_scoring_worker.py", "meta_prediction_optimizer_anti_overfit.py"):
        g = subprocess.run(["git", "-C", str(REPO), "show", f"{a.baseline_rev}:{f}"], capture_output=True)
        if g.returncode != 0:
            raise SystemExit(f"--baseline-rev {a.baseline_rev}: cannot read {f}: {g.stderr.decode()[:200]}")
        (base / f).write_bytes(g.stdout)
    _b = {f: (base / f).read_text() for f in ("survivor_scorer.py", "full_scoring_worker.py",
                                              "meta_prediction_optimizer_anti_overfit.py")}
    gate("LL.baseline_precondition",
         "def metadata_merge_fields" not in _b["survivor_scorer.py"]
         and "per_seed_sieve_features" not in _b["full_scoring_worker.py"]
         and "--exclude-feature-groups" not in _b["meta_prediction_optimizer_anti_overfit.py"],
         f"baseline {a.baseline_rev[:12]} must be pre-LL (no LL-2/LL-3 switch); if this fails the revision "
         f"has the switches and every *_identical_to_HEAD gate would compare the new code with itself")
    fx = out / "fx"
    chunk = make_fixture(fx)
    fx_miss = out / "fx_missing"
    make_fixture(fx_miss, drop_sieve_for_one=True)
    fx_hold = out / "fx_other_holdout"
    make_fixture(fx_hold, holdout_seed=99)

    ON = {"per_seed_sieve_features": "on"}
    cases = {
        "A_base_default": (base, fx, None, False, None),
        "B_new_default": (REPO, fx, None, False, None),
        "C_new_explicit_defaults": (REPO, fx, {"per_seed_sieve_features": "off",
                                               "fallback_metadata_merge": "legacy6"}, False, None),
        "D_new_on": (REPO, fx, ON, False, None),
        "D2_new_on_other_holdout": (REPO, fx_hold, ON, False, None),
        "E_base_fallback": (base, fx, None, True, None),
        "F_new_fallback_default": (REPO, fx, None, True, None),
        "G_new_fallback_shared_on": (REPO, fx, {"per_seed_sieve_features": "on",
                                                "fallback_metadata_merge": "shared"}, True, None),
        "H_new_on_missing": (REPO, fx_miss, ON, False, None),
        "H2_new_off_missing": (REPO, fx_miss, None, False, None),
        "MUT_fallback_omits_sieve": (REPO, fx, {"per_seed_sieve_features": "on",
                                                "fallback_metadata_merge": "shared"}, True,
                                     "fallback_omits_sieve"),
        "MUT_fwd_rev_swapped": (REPO, fx, ON, False, "fwd_rev_swapped"),
        "MUT_off_leaks_sieve": (REPO, fx, None, False, "off_leaks_sieve"),
        "MUT_no_fail_closed": (REPO, fx_miss, ON, False, "no_fail_closed"),
        "MUT_batch_list_drops_one": (REPO, fx, None, False, "batch_list_drops_one"),
    }
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {k: ex.submit(run_case, out, k, *v) for k, v in cases.items()}
        R = {k: f.result() for k, f in futs.items()}
    t_step3 = time.time() - t0
    for k, v in R.items():
        if v["rc"] != 0:
            print(f"[WARN] case {k} runner exit {v['rc']}")
    import survivor_scorer as ss

    # ---- LL-2 ----
    gate("LL2.loaded_paths", R["A_base_default"]["loaded"]["survivor_scorer"].startswith(str(base))
         and R["B_new_default"]["loaded"]["survivor_scorer"] == str(REPO / "survivor_scorer.py"),
         "baseline ran HEAD copies, new ran the worktree")
    gate("LL2.default_bit_identical_to_HEAD", *g_identical(R["A_base_default"], R["B_new_default"]))
    gate("LL2.explicit_defaults_identical", *g_identical(R["B_new_default"], R["C_new_explicit_defaults"]))
    gate("LL2.on_adds_two_npz_values", *g_on_adds_exactly_two(R["B_new_default"], R["D_new_on"], chunk))
    hold_ok = all(r0["features"]["forward_matches"] == r1["features"]["forward_matches"]
                  and r0["features"]["reverse_matches"] == r1["features"]["reverse_matches"]
                  for r0, r1 in zip(R["D_new_on"]["results"], R["D2_new_on_other_holdout"]["results"]))
    lab_moved = any(r0["holdout_hits"] != r1["holdout_hits"] or r0["holdout_quality"] != r1["holdout_quality"]
                    for r0, r1 in zip(R["D_new_on"]["results"], R["D2_new_on_other_holdout"]["results"]))
    gate("LL2.leak.sieve_features_ignore_holdout", hold_ok and lab_moved,
         f"holdout changed (labels moved={lab_moved}); sieve features unchanged={hold_ok}")
    gate("LL2.leak.runH_window_inside_train", *g_leak_runH(a.runh_dataset))
    if a.runh_root:
        gate("LL.runH_fixtures_equal_live", *g_runh_live(Path(a.runh_root).expanduser(), a.runh_dataset))
        gate("LL.runH_live_catches_window_config_change",
             *g_runh_live_catches_window_change(Path(a.runh_root).expanduser(), a.runh_dataset, out))
    gate("LL2.leak.check_catches_crossing", *g_leak_detects_crossing())
    gate("LL2.fallback_default_bit_identical_to_HEAD", *g_identical(R["E_base_fallback"], R["F_new_fallback_default"]))
    gate("LL2.legacy6_fallback_zeroes_the_seven (today's defect, kept as default)",
         *g_legacy_zeroes_seven(R["B_new_default"], R["F_new_fallback_default"]))
    fp_fields = tuple(ss.METADATA_MERGE_FIELDS) + tuple(ss.PER_SEED_SIEVE_FIELDS)
    gate("LL2.F-PARITY shared+on: batch == fallback", *g_fparity(R["D_new_on"], R["G_new_fallback_shared_on"], fp_fields))
    gate("LL2.on_fails_closed_on_missing_value", *g_fail_closed(R["H_new_on_missing"]))
    gate("LL2.off_unaffected_by_missing_value", R["H2_new_off_missing"]["results"] is not None,
         str(R["H2_new_off_missing"].get("raised")))

    # ---- LL-2 mutants: applied, ran, and the target gate fails ----
    def mut(name, gate_fn):
        c = R[name]
        ran = c.get("calls", {}).get("mutant_calls", 0) > 0
        try:
            ok, det = gate_fn(c)
        except Exception as e:
            ok, det = False, f"gate raised {type(e).__name__}"
        gate(f"MUTANT {name} killed", ran and not ok, f"calls={c.get('calls')}, target gate -> {det[:90]}")

    mut("MUT_fallback_omits_sieve", lambda c: g_fparity(R["D_new_on"], c, fp_fields))
    mut("MUT_fwd_rev_swapped", lambda c: g_on_adds_exactly_two(R["B_new_default"], c, chunk))
    mut("MUT_off_leaks_sieve", lambda c: g_identical(R["A_base_default"], c))
    mut("MUT_no_fail_closed", lambda c: g_fail_closed(c))
    mut("MUT_batch_list_drops_one", lambda c: g_identical(R["A_base_default"], c))

    # ---- LL-1 ----
    import models.feature_manifest as fm
    MP = fm.MANIFEST_PATH
    gate("LL1.manifest_loads_groups_disjoint", bool(fm.load_manifest(MP)), str(MP.name))
    gate("LL1.manifest_rebuilds_from_live_names", *g_manifest_rebuild(fm, MP))
    gate("LL1.manifest_v1_equals_runH_sidecar", *g_manifest_vs_sidecar(fm, MP))
    gate("LL1.manifest_agrees_with_code_constants", *g_manifest_vs_code(fm, MP))
    gate("LL1.schema_check_modes", *g_schema_check(fm))
    gate("LL1.step6_wiring", *g_step6_wiring(fm))
    v2_on = sorted(R["D_new_on"]["results"][0]["features"])
    gate("LL1.step3_on_output_is_manifest_v2", fm.identify_version(fm.load_manifest(MP), v2_on) == "v2"
         and fm.identify_version(fm.load_manifest(MP), sorted(R["B_new_default"]["results"][0]["features"])) == "v1",
         f"on -> {fm.identify_version(fm.load_manifest(MP), v2_on)}")

    # LL-1 mutants (in process, on temp copies; never on the real manifest)
    mdir = out / "manifest_mutants"
    mdir.mkdir()
    m0 = json.load(open(MP))
    m_drop = copy.deepcopy(m0)
    m_drop["groups"]["battery"].remove(m_drop["groups"]["battery"][0])
    (mdir / "drop.json").write_text(json.dumps(m_drop))
    try:
        ok, det = g_manifest_rebuild(fm, mdir / "drop.json")
        ok2 = True
        try:
            fm.load_manifest(mdir / "drop.json")
        except fm.ManifestError:
            ok2 = False
        gate("MUTANT LL1.drop_name killed", not ok and not ok2, "rebuild differs and load refuses")
    except Exception as e:
        gate("MUTANT LL1.drop_name killed", True, f"refused: {type(e).__name__}")
    m_swap = copy.deepcopy(m0)
    m_swap["groups"]["global"].remove("global_regime_age")
    m_swap["groups"]["seed_history"].append("global_regime_age")
    (mdir / "swap.json").write_text(json.dumps(m_swap))
    ok_a, _ = g_manifest_rebuild(fm, mdir / "swap.json")
    ok_b, _ = g_manifest_vs_code(fm, mdir / "swap.json")
    gate("MUTANT LL1.swap_namespace_of_a_global killed", not ok_a and not ok_b, "rebuild and code gates fail")
    orig_check = fm.check_rows_against_schema
    calls = {"n": 0}

    def noop(*x, **k):
        calls["n"] += 1
        m = k.get("mode", "off")
        return {"mode": "off"} if m == "off" else {"mode": m, "rows": 0, "rows_missing": 0,
                                                    "missing_names": {}, "problems": []}
    fm.check_rows_against_schema = noop
    try:
        ok, det = g_schema_check(fm)
        ok_w, _ = g_step6_wiring(fm)
    finally:
        fm.check_rows_against_schema = orig_check
    gate("MUTANT LL1.schema_check_noop killed", calls["n"] > 0 and not ok and not ok_w, f"calls={calls['n']}: {det}")

    # ---- LL-3 ----
    import importlib.util
    sys.path.insert(0, str(REPO))
    import meta_prediction_optimizer_anti_overfit as new_mod
    spec = importlib.util.spec_from_file_location("base_meta_opt", base / "meta_prediction_optimizer_anti_overfit.py")
    base_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base_mod)
    s5 = out / "step5_fixture_survivors.json"
    make_step5_file(s5, R["B_new_default"])
    names91 = sorted(R["B_new_default"]["results"][0]["features"])
    man = fm.load_manifest(MP)
    G = man["groups"]
    res = fm.resolve_step5_exclusions(names91)
    gate("LL3.resolve_default_is_today", res == {"exclude": list(fm.STEP5_DEFAULT_EXCLUDE), "selection": None},
         str(res["exclude"]))
    gate("LL3.step5_default_identical_to_HEAD", *g_step5_default_identical(new_mod, base_mod, s5))
    gate("LL3.exclude_group_global", *g_step5_group(new_mod, s5, (None, None, ["global"]), G["global"], 89 - 14))
    gate("LL3.include_alias_survivor_local", *g_step5_group(
        new_mod, s5, (None, ["survivor_local"], None), G["global"] + G["dead"], 89 - 14 - 5))
    gate("LL3.refusals", *g_step5_refusals(fm, names91))
    s5v2 = out / "step5_fixture_survivors_v2.json"
    make_step5_file(s5v2, R["D_new_on"])
    o = load_data_with(new_mod, s5v2, (None, None, ["per_seed_sieve"]))
    o_on = load_data_with(new_mod, s5v2, None)
    gate("LL3.v2_data: default trains 91, dropping per_seed_sieve gives the v1 89",
         o_on.X.shape[1] == 91 and o.X.shape[1] == 89
         and o.feature_schema["feature_names"] == man["versions"]["v1"]["trained_by_default"]
         and o_on.feature_schema["feature_schema_hash"] == man["versions"]["v2"]["trained_by_default_hash"],
         f"{o_on.X.shape[1]} / {o.X.shape[1]}")

    # LL-3 mutants
    orig_def = fm.STEP5_DEFAULT_EXCLUDE
    fm.STEP5_DEFAULT_EXCLUDE = ("score", "confidence")
    try:
        res_m = fm.resolve_step5_exclusions(names91)
        ok = res_m == {"exclude": ["score", "confidence", "holdout_hits", "holdout_quality"], "selection": None}
    finally:
        fm.STEP5_DEFAULT_EXCLUDE = orig_def
    gate("MUTANT LL3.default_drifts killed", not ok, "resolve-default gate fails")
    calls = {"n": 0}

    def wrong_load(path=MP):
        calls["n"] += 1
        m = json.load(open(path))
        m["aliases"]["survivor_local"] = [g for g in m["aliases"]["survivor_local"] if g != "battery"]
        return m
    orig_load = fm.load_manifest
    fm.load_manifest = wrong_load
    try:
        ok, det = g_step5_group(new_mod, s5, (None, ["survivor_local"], None), G["global"] + G["dead"], 70)
    finally:
        fm.load_manifest = orig_load
    gate("MUTANT LL3.group_resolves_wrong killed", calls["n"] > 0 and not ok, f"calls={calls['n']}: {det}")

    # LL-3 refusals through the real CLI, before any fitting (lead ADDENDUM_2 Fix 1, Beta R1 Q2)
    lot = out / "step5_cli_lottery.json"
    lot.write_text(json.dumps(list(range(300))))
    cli = out / "step5_cli"
    by_case = {}
    for case, extra, want in CLI_REFUSALS:
        d = cli / case
        p = run_step5_cli(REPO, d, s5, lot, cli_extra(extra, names91))
        by_case[case] = (extra, want)
        gate(f"LL3.cli_refuses {case}", *g_cli_refused(p, d, want))
    # the explicit switch: a list without score and confidence trains on them, recorded in the sidecar
    d = cli / "allow_targets_on"
    p = run_step5_cli(REPO, d, s5, lot, ["--exclude-features", "holdout_hits,holdout_quality",
                                         "--allow-target-columns-as-features"])
    side = d / "models" / "best_model.meta.json"
    fs = json.load(open(side)).get("feature_schema", {}) if side.is_file() else {}
    sel = fs.get("feature_selection") or {}
    gate("LL3.cli_allow_target_columns: trains on score and confidence, flag and request in the sidecar",
         p.returncode == 0 and fs.get("feature_count") == 91 and {"score", "confidence"} <= set(fs.get("feature_names", []))
         and sel.get("allow_target_columns_as_features") is True
         and sel.get("requested") == {"exclude_features": ["holdout_hits", "holdout_quality"], "include_groups": None,
                                      "exclude_groups": None, "allow_target_columns_as_features": True},
         f"rc {p.returncode}; features {fs.get('feature_count')}; selection.requested {sel.get('requested')}")
    for mname, old, new, case in CLI_MUTANTS:
        t = mutant_tree(cli / f"tree_{case}", old, new)
        extra, want = by_case[case]
        d = cli / f"mutant_{case}"
        p = run_step5_cli(t, d, s5, lot, cli_extra(extra, names91))
        ok, det = g_cli_refused(p, d, want)
        gate(f"MUTANT LL3.cli {mname} killed", not ok, f"under the mutant {case}: {det}")

    wall = time.time() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    child_rss = max((v.get("peak_rss_mb", 0) for v in R.values()), default=0)
    summ = {"head": head, "gates": len(RESULTS), "passed": sum(r["ok"] for r in RESULTS),
            "failed": [r["gate"] for r in RESULTS if not r["ok"]],
            "mutants": sum(1 for r in RESULTS if r["gate"].startswith("MUTANT")),
            "mutants_killed": sum(1 for r in RESULTS if r["gate"].startswith("MUTANT") and r["ok"]),
            "wall_seconds": round(wall, 1), "step3_cases_seconds": round(t_step3, 1),
            "workers": a.workers, "parent_peak_rss_mb": round(rss, 1),
            "child_peak_rss_mb_max": child_rss, "results": RESULTS}
    (out / "selftest_summary.json").write_text(json.dumps(summ, indent=1))
    print(f"\n{summ['passed']}/{summ['gates']} gates pass; mutants killed "
          f"{summ['mutants_killed']}/{summ['mutants']}; wall {summ['wall_seconds']} s; "
          f"parent RSS {summ['parent_peak_rss_mb']} MB, child max {child_rss} MB", flush=True)
    return 0 if not summ["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
