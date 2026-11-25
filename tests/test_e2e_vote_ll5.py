#!/usr/bin/env python3
"""
Self-test for LL-5 in scripts/diagnostics/e2e_vote_harness.py (--emit-ranked-predictions,
TODO-E2E-V11). CPU only, the H6 synthetic fixture only (tests/test_e2e_vote_harness.build_fixture).

Run:  python3 -u tests/test_e2e_vote_ll5.py [--out-root DIR] | tee <log>

Gates:
  G15a  off is today's behaviour. Harness rev 5 with the flag off gives the same pools.jsonl
        (apart from config_id, which carries the harness sha256) and the same diagnostics as the
        archived rev 4 copy (tests/fixtures/e2e_vote_harness_rev4.py.txt, sha256
        99791061...). Its identity inputs differ only in harness_sha256. With the flag on, pools.jsonl,
        diagnostics and config_id are byte-identical to off.
  G15b  on, learned (LightGBM sidecar) and unlearned (field_sum) arms. For each target:
        - one line, in order;
        - the real Chapter 13 reader (Chapter13Orchestrator.load_predictions_from_disk) accepts the
          line with expected_draw_id = draw_<t+1> and refuses it as stale with draw_<t>;
        - features is a name-keyed map whose names equal the sidecar schema (learned) or the row
          features sorted (unlearned);
        - values finite, ranks 1..depth unique, prediction = pool slot;
        - independent oracles: the seed's own sequence at t (SurvivorScorer) equals the
          prediction; the feature values equal the seed's Step 3 row; for field_sum, the seeds
          are the key order.
  G15c  fail-closed: depth > pool size; depth or draw-id rule without on; on with a blocked
        signal gate.
Mutants (killed = applied, path ran, the credited gate fired):
  M19 features written as a positional list instead of a map (G15b; the reader alone accepts it)
  M20 stale draw_id = draw_<t> (G15b)
  M21 slot-to-survivor map shifted by one (G15b, through the harness's own score check)
"""

import argparse
import importlib.machinery
import importlib.util
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))
import test_e2e_vote_harness as T   # noqa: E402  (fixture, runner, GateFail, run_mutant)

H = T.H
GateFail, check = T.GateFail, T.check
REV4 = REPO / "tests" / "fixtures" / "e2e_vote_harness_rev4.py.txt"   # tracked copy of data/e2e_loop_miss/provenance/ (same sha256)
ADDED = {"pipeline_switches"}   # the only record key added since rev 4/5 (LL-1 check, ADDENDUM_2)
REV4_SHA = "99791061cde4c05fcf7e1dd831c05a04bda7f389e4000ce524f58bac55774544"


def load_rev4():
    check("G15a", H.sha256_file(REV4) == REV4_SHA, "archived rev 4 copy sha256 changed")
    sys.dont_write_bytecode = True     # no __pycache__ under tests/fixtures/ (ADDENDUM_2 item 3)
    loader = importlib.machinery.SourceFileLoader("e2e_vote_harness_rev4", str(REV4))
    spec = importlib.util.spec_from_loader("e2e_vote_harness_rev4", loader)
    m = importlib.util.module_from_spec(spec)
    sys.modules["e2e_vote_harness_rev4"] = m
    loader.exec_module(m)
    m.REPO_ROOT = REPO     # the copy lives under tests/fixtures/; point it at the worktree as rev 4 was
    return m


def c13_reader():
    sys.dont_write_bytecode = True
    import chapter_13_orchestrator as c13
    return c13.Chapter13Orchestrator.load_predictions_from_disk


def run_with(mod, F, out_root, extra, tag):
    """Run a harness module's main in the fixture cwd. Returns (record, out_dir)."""
    T.RUN_N["n"] += 1
    out = out_root / f"ll5_{tag}_{T.RUN_N['n']:03d}"
    fx = F["fx"]
    argv = ["--survivors", str(fx / "survivors_with_scores.json"), "--dataset",
            str(fx / "dataset.json"), "--history", str(fx / "train_history.json"),
            str(fx / "holdout_history.json"), "--config", str(fx / "prediction_config.json"),
            "--out-dir", str(out), "--data-cutoff-index", "t"] + extra
    cwd = os.getcwd()
    os.chdir(fx)
    try:
        rc = mod.main(argv)
    finally:
        os.chdir(cwd)
    assert rc == 0
    return json.loads((out / "run_record.json").read_text()), out


def lines_of(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines()]


def strip_cid(recs):
    return [{k: v for k, v in r.items() if k != "config_id"} for r in recs]


def g15a_off(F, out_root):
    R4 = load_rev4()
    lg = ["--models-dir", str(F["models"]["lightgbm"]), "--block-start", "250", "--block-end",
          "300", "--workers", "1"]
    rec4, o4 = run_with(R4, F, out_root, lg, "rev4")
    rec5, o5 = run_with(H, F, out_root, lg, "off")
    p4, p5 = lines_of(o4 / "pools.jsonl"), lines_of(o5 / "pools.jsonl")
    check("G15a", strip_cid(p4) == strip_cid(p5), "off pools differ from rev 4 (beyond config_id)")
    check("G15a", (o4 / "vote_diagnostics.jsonl").read_bytes() ==
          (o5 / "vote_diagnostics.jsonl").read_bytes(), "off diagnostics differ from rev 4")
    i4 = dict(rec4["config_identity_inputs"]); i5 = dict(rec5["config_identity_inputs"])
    check("G15a", i4.pop("harness_sha256") == REV4_SHA, "rev 4 run did not record its own sha")
    i5.pop("harness_sha256")
    check("G15a", i4 == i5, "identity inputs differ beyond harness_sha256")
    check("G15a", "ranked_predictions" not in rec5 and not (o5 / "ranked_predictions.jsonl").exists(),
          "off wrote ranked output")
    check("G15a", set(rec5) - set(rec4) == ADDED and set(rec4) <= set(rec5),
          f"record keys differ beyond {ADDED}: {set(rec4) ^ set(rec5)}")
    ll1 = rec5["pipeline_switches"]["LL-1"]
    check("G15a", ll1["requested"] == {} and ll1["applied_schema_check"] == "off",
          f"off run records an LL-1 setting: {ll1}")
    check("G15a", rec4["declared_parameters"] == rec5["declared_parameters"],
          "declared parameters differ under off")
    w4 = lg[:-1] + ["4"]
    rec_off, o_off = run_with(H, F, out_root, w4, "off4")
    rec_on, o_on = run_with(H, F, out_root, w4 + ["--emit-ranked-predictions", "on"], "on4")
    for f in ("pools.jsonl", "vote_diagnostics.jsonl"):
        check("G15a", (o_off / f).read_bytes() == (o_on / f).read_bytes(), f"on changed {f}")
    check("G15a", rec_off["config_id"] == rec_on["config_id"], "on changed config_id")
    check("G15a", (o_off / "pools.jsonl").read_bytes() == (o5 / "pools.jsonl").read_bytes(),
          "workers=4 off differs from workers=1 off")
    return (f"rev4 vs rev5-off: 50 pools equal except config_id, diagnostics byte-equal, identity "
            f"differs only in harness sha; on vs off (4 workers): pools/diagnostics/config_id "
            f"byte-equal ({rec_on['config_id'][:12]})")


def check_ranked(F, out_root, out, rec, names_expected, reader, oracle_key=None):
    from survivor_scorer import SurvivorScorer
    sc = SurvivorScorer("java_lcg", 1000)
    rows = json.loads((F["fx"] / "survivors_with_scores.json").read_text())
    pools = lines_of(out / "pools.jsonl")
    ranked = lines_of(out / "ranked_predictions.jsonl")
    check("G15b", rec["ranked_predictions"]["sha256"] == H.sha256_file(out / "ranked_predictions.jsonl"),
          "ranked file sha256 not in record")
    check("G15b", [r["combined_index"] for r in ranked] == [p["target"]["combined_index"] for p in pools],
          "ranked lines do not match the pools one for one")
    depth = rec["ranked_predictions"]["depth"]
    tmpd = out / "reader_check"; tmpd.mkdir()
    n_entries = 0
    for obj, pr in zip(ranked, pools):
        t = obj["combined_index"]
        check("G15b", obj["schema"] == "tfm.e2e.ranked_predictions.v1", "schema id")
        fp = tmpd / f"t{t}.json"
        fp.write_text(json.dumps(obj))
        got = reader(str(fp), expected_draw_id=f"draw_{t + 1}")
        check("G15b", got is not None and len(got) == depth,
              f"t={t}: Chapter 13 reader refused the line for draw_{t + 1}")
        check("G15b", reader(str(fp), expected_draw_id=f"draw_{t}") is None,
              f"t={t}: reader accepted it for draw_{t}: draw_id is stale or missing")
        ents = obj["predictions"]
        check("G15b", [e["rank"] for e in ents] == list(range(1, depth + 1)), "ranks not 1..depth")
        check("G15b", len({e["row_index"] for e in ents}) == depth, "row indexes repeat")
        for i, e in enumerate(ents):
            check("G15b", isinstance(e["features"], dict), f"t={t}: features is not a name-keyed map")
            check("G15b", list(e["features"]) == names_expected, f"t={t}: feature names differ")
            check("G15b", all(math.isfinite(v) for v in e["features"].values()), "non-finite value")
            check("G15b", e["prediction"] == pr["pool"][i], f"t={t} slot {i}: prediction != pool")
            row = rows[e["row_index"]]
            check("G15b", int(row["seed"]) == e["seed"], "seed != row seed")
            check("G15b", all(e["features"][n] == float(row["features"].get(n, 0.0))
                              for n in names_expected), f"t={t}: features != the seed's Step 3 row")
            seq = sc._generate_sequence(e["seed"], t + 1, skip=0)
            check("G15b", int(seq[t]) == e["prediction"],
                  f"t={t} slot {i}: seed {e['seed']} votes {int(seq[t])}, entry says {e['prediction']}")
            n_entries += 1
        if oracle_key is not None:
            check("G15b", [e["seed"] for e in ents] == oracle_key[:depth], "field_sum order differs")
    return n_entries


def g15b_on(F, out_root, light=False):
    reader = c13_reader()
    sidecar = json.loads((F["models"]["lightgbm"] / "best_model.meta.json").read_text())
    names = sidecar["feature_schema"]["feature_names"]
    extra = ["--models-dir", str(F["models"]["lightgbm"]), "--block-start", "270",
             "--block-end", "280" if light else "300", "--workers", "1" if light else "3",
             "--emit-ranked-predictions", "on"]
    try:
        rec, out = run_with(H, F, out_root, extra, "on")
    except H.HarnessError as e:
        raise GateFail("G15b", f"harness refused: {e}")
    n1 = check_ranked(F, out_root, out, rec, names, reader)
    if light:
        return n1
    # unlearned arm, field_sum over the NPZ fields (distinct sums in the fixture)
    import numpy as np
    z = np.load(F["fx"] / "step1_survivors.npz")
    key = {int(s): float(a) + float(b) for s, a, b in zip(z["seeds"], z["forward_matches"],
                                                         z["reverse_matches"])}
    rows = json.loads((F["fx"] / "survivors_with_scores.json").read_text())
    # rows sorted by key desc, ties (the duplicated row) by row order, as seed_asc would not split them
    order = sorted(range(len(rows)), key=lambda r: (-key[int(rows[r]["seed"])], int(rows[r]["seed"])))
    oracle = [int(rows[r]["seed"]) for r in order]
    a0 = ["--ranking", "field_sum", "--ranking-fields", "forward_matches,reverse_matches",
          "--ranking-npz", str(F["fx"] / "step1_survivors.npz"), "--tie-break", "seed_asc",
          "--block-start", "270", "--block-end", "300", "--workers", "3",
          "--emit-ranked-predictions", "on", "--ranked-predictions-depth", "10"]
    try:
        rec0, out0 = run_with(H, F, out_root, a0, "on_a0")
    except H.HarnessError as e:
        raise GateFail("G15b", f"harness refused A0: {e}")
    names0 = sorted({k for r in rows for k in r["features"]
                     if k not in ("score", "confidence", "seed", "skip")})
    n0 = check_ranked(F, out_root, out0, rec0, names0, reader, oracle_key=oracle)
    return (f"learned: 30 targets x 20 entries = {n1} checked; unlearned field_sum: 30 x 10 = {n0}; "
            f"reader accepts draw_<t+1>, refuses draw_<t>; {len(names)} sidecar names, "
            f"{len(names0)} row names")


def g15c_fail_closed(F, out_root):
    base = ["--models-dir", str(F["models"]["lightgbm"]), "--block-start", "270", "--block-end",
            "272"]
    cases = [(base + ["--emit-ranked-predictions", "on", "--ranked-predictions-depth", "21"],
              "must be in"),
             (base + ["--ranked-predictions-depth", "5"], "need --emit-ranked-predictions on"),
             (base + ["--ranked-draw-id-rule", "c13_fallback", "--ranked-predictions-depth", "19"],
              "need --emit-ranked-predictions on"),
             (["--models-dir", str(F["models"]["gate_blocked"]), "--block-start", "270",
               "--block-end", "272", "--emit-ranked-predictions", "on"], "signal gate")]
    out = []
    for argv, want in cases:
        try:
            run_with(H, F, out_root, argv, "fc")
        except H.HarnessError as e:
            check("G15c", want in str(e), f"wrong refusal: {e}")
            out.append(want)
            continue
        raise GateFail("G15c", f"accepted: {argv}")
    return f"{len(out)} refusals"


def mutants(F, out_root):
    import numpy as np
    res = []
    gate = lambda: g15b_on(F, out_root, light=True)

    def m19():
        hits = {"n": 0}; orig = H.ranked_line
        def mut(*a, **k):
            o = orig(*a, **k)
            for p in o["predictions"]:
                hits["n"] += 1
                p["features"] = list(p["features"].values())
            return o
        H.ranked_line = mut
        return (lambda: setattr(H, "ranked_line", orig)), hits
    res.append(T.run_mutant("M19 positional feature list", m19, gate, "G15b"))

    def m20():
        hits = {"n": 0}; orig = H.draw_id_for
        def mut(t, rule):
            hits["n"] += 1
            return f"draw_{t}"
        H.draw_id_for = mut
        return (lambda: setattr(H, "draw_id_for", orig)), hits
    res.append(T.run_mutant("M20 stale draw_id", m20, gate, "G15b"))

    def m21():
        hits = {"n": 0}; orig = H.ranked_order
        def mut(scores):
            hits["n"] += 1
            return np.roll(orig(scores), 1)
        H.ranked_order = mut
        return (lambda: setattr(H, "ranked_order", orig)), hits
    res.append(T.run_mutant("M21 slot map shifted by one", m21, gate, "G15b"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=None)
    a = ap.parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(a.out_root) if a.out_root else REPO / "data" / "e2e_vote_selftest" / f"ll5_{stamp}"
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=False)
    t_all = time.time()
    print(f"[ll5] out_root={out_root}", flush=True)
    F = T.build_fixture(out_root)
    summary = {"out_root": str(out_root), "harness_sha256": H.sha256_file(T.HARNESS_PATH),
               "gates": {}, "mutants": []}
    results = []
    for name, fn in (("G15a off = rev 4; on leaves pools and config_id unchanged", lambda: g15a_off(F, out_root)),
                     ("G15b on: Chapter 13 reader shape, name-keyed map, oracles", lambda: g15b_on(F, out_root)),
                     ("G15c fail-closed", lambda: g15c_fail_closed(F, out_root))):
        t = time.time()
        try:
            r = fn()
            results.append(True); summary["gates"][name] = {"pass": True, "detail": r}
            print(f"PASS {name} ({time.time() - t:.1f}s): {r}", flush=True)
        except Exception as e:
            results.append(False); summary["gates"][name] = {"pass": False, "error": str(e)}
            print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)
    mres = mutants(F, out_root)
    for name, killed, msg in mres:
        print(f"{'KILLED' if killed else 'SURVIVED'} {name}: {msg}", flush=True)
        summary["mutants"].append({"name": name, "killed": killed, "detail": msg})
    summary["total_seconds"] = round(time.time() - t_all, 1)
    summary["peak_rss_mib"] = {
        "self": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "largest_child": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1)}
    ng = sum(results); nk = sum(1 for _, k, _ in mres if k)
    summary["verdict"] = {"gates": f"{ng}/{len(results)} pass", "mutants": f"{nk}/{len(mres)} killed"}
    (out_root / "ll5_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[ll5] gates {summary['verdict']['gates']}, mutants {summary['verdict']['mutants']}, "
          f"{summary['total_seconds']}s, peak RSS {summary['peak_rss_mib']} MiB", flush=True)
    return 0 if (ng == len(results) and nk == len(mres)) else 1


if __name__ == "__main__":
    sys.exit(main())
