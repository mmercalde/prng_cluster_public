#!/usr/bin/env python3
"""
Self-test for TODO-E2E-V12 (--emit-value-ordering full_vote) and TODO-E2E-V13
(--ranking-key-round-decimals) in scripts/diagnostics/e2e_vote_harness.py. CPU, the H6
synthetic fixture only (tests/test_e2e_vote_harness.build_fixture). No run H data.

Run:  python3 -u tests/test_e2e_vote_ordering.py [--out-root DIR] [--cost-probe] | tee <log>

Oracle for the ordering: the module itself. Step 6 run with --pool-size = --k = all survivors
emits the vote of every survivor in ranking order (prediction_generator.py:841, :653). The
value_scores the harness writes must equal -(first position) in that full module pool, value for
value. The grader's own draw_ranks (e2e_grader.py 3ab8ce2a, R24; imported read-only) must rank
each pool's distinct values 1..d in pool order ("top-k of the ordering equals the pool").

Gates:
  G16a off = rev 5 (archived copy): pools and diagnostics equal apart from config_id; identity
       differs only in harness_sha256; no new fields; record keys gain only pipeline_switches (default).
  G16b on, learned (LightGBM), block 250..301 (pending included), 3 workers: oracle equality on
       every record; grader draw_ranks top-k = pool; config_id differs from off (declared).
  G16c on, per_skip arm (skip 3): oracle equality (the matrix uses the arm's skip).
  G16d on, A0 field_sum + seed_asc: oracle equality.
  G17  rounding: a key with float noise. Off: order by the raw float sums. On (6 decimals): the
       noisy pair ties and seed_asc orders it; distinct key values before > after; enters
       config_id; off is byte-equal to rev 5.
  G18  fail-closed: rounding without field_sum; negative decimals; ordering on a blocked gate.
Mutants: M22 first occurrence replaced by last (G16b); M23 matrix ignores the per-survivor skip
  (G16c); M24 rounding dropped (G17).
"""
import argparse
import importlib.machinery
import importlib.util
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))
import test_e2e_vote_harness as T   # noqa: E402
import test_e2e_vote_ll5 as L       # noqa: E402  (run_with, lines_of, strip_cid)

H = T.H
GateFail, check = T.GateFail, T.check
REV5 = REPO / "tests" / "fixtures" / "e2e_vote_harness_rev5.py.txt"   # tracked copy of data/e2e_loop_miss/provenance/ (same sha256)
ADDED = {"pipeline_switches"}   # the only record key added since rev 4/5 (LL-1 check, ADDENDUM_2)
REV5_SHA = "a26e39c1ab4925e997b87d8795db77d2d9c651960c1ee46f67e08e2faffd2fb7"
GRADER = REPO / "scripts" / "diagnostics" / "e2e_grader.py"   # this repo's copy (was the e2e-grader worktree)
GRADER_SHA = "3ab8ce2a5330fe36ae638e8bf2915e23bdce08bcefc0221eb47e21177fc8f341"   # landed, main fe5ac3b (R26)


def load_copy(path, sha, name):
    check("G16a", H.sha256_file(path) == sha, f"{path.name} sha256 changed")
    sys.dont_write_bytecode = True     # no __pycache__ under tests/fixtures/ (ADDENDUM_2 item 3)
    ld = importlib.machinery.SourceFileLoader(name, str(path))
    sp = importlib.util.spec_from_loader(name, ld)
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    ld.exec_module(m)
    m.REPO_ROOT = REPO
    return m


def load_grader():
    sys.dont_write_bytecode = True
    check("G16b", H.sha256_file(GRADER) == GRADER_SHA, "grader sha256 is not 3ab8ce2a")
    sp = importlib.util.spec_from_file_location("e2e_grader_ro", GRADER)
    g = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(g)
    return g


def run(F, out_root, extra, tag, expect_fail=False):
    try:
        return L.run_with(H, F, out_root, extra, tag)
    except H.HarnessError as e:
        if expect_fail:
            return None, str(e)
        raise GateFail(TAG["g"], f"harness refused: {e}")


TAG = {"g": "G16b"}


def oracle_check(F, out_root, extra, tag, n):
    """Harness with ordering on vs the module's own full pool (pool size = k = n)."""
    rec, out = run(F, out_root, extra + ["--emit-value-ordering", "full_vote"], f"{tag}_on")
    full, outf = run(F, out_root, extra + ["--pool-size", str(n), "--k", str(n)], f"{tag}_full")
    pools = L.lines_of(out / "pools.jsonl")
    fulls = {p["target"]["combined_index"]: p for p in L.lines_of(outf / "pools.jsonl")}
    checked = 0
    for p in pools:
        t = p["target"]["combined_index"]
        if p["abstain"]:
            continue
        fp = fulls[t]["pool"]
        check(TAG["g"], len(fp) == n, f"t={t}: module full pool has {len(fp)} votes, not {n}")
        want = [-(n + 1)] * 1000
        for i, v in enumerate(fp):
            if want[v] == -(n + 1):
                want[v] = -(i + 1)
        check(TAG["g"], p.get("value_scores") == want, f"t={t}: value_scores != module full vote")
        check(TAG["g"], fp[:len(p["pool"])] == p["pool"], f"t={t}: pool is not the full pool's prefix")
        check(TAG["g"], p.get("rank_basis") == H.FULL_VOTE_RANK_BASIS, "rank_basis missing")
        checked += 1
    return rec, out, pools, checked


def g16a_off(F, out_root):
    R5 = load_copy(REV5, REV5_SHA, "e2e_vote_harness_rev5")
    out = {}
    for name, extra in (("learned", ["--models-dir", str(F["models"]["lightgbm"])]),
                        ("A0", ["--ranking", "field_sum", "--ranking-fields",
                                "forward_matches,reverse_matches", "--ranking-npz",
                                str(F["fx"] / "step1_survivors.npz"), "--tie-break", "seed_asc"])):
        ex = extra + ["--block-start", "250", "--block-end", "300", "--workers", "1"]
        r5, o5 = L.run_with(R5, F, out_root, ex, "rev5")
        r6, o6 = L.run_with(H, F, out_root, ex, "off")
        check("G16a", L.strip_cid(L.lines_of(o5 / "pools.jsonl")) ==
              L.strip_cid(L.lines_of(o6 / "pools.jsonl")), f"{name}: off pools differ from rev 5")
        check("G16a", (o5 / "vote_diagnostics.jsonl").read_bytes() ==
              (o6 / "vote_diagnostics.jsonl").read_bytes(), f"{name}: diagnostics differ")
        i5, i6 = dict(r5["config_identity_inputs"]), dict(r6["config_identity_inputs"])
        check("G16a", i5.pop("harness_sha256") == REV5_SHA, "rev 5 sha not recorded")
        i6.pop("harness_sha256")
        check("G16a", i5 == i6, f"{name}: identity differs beyond the harness sha")
        check("G16a", set(r6) - set(r5) == ADDED and set(r5) <= set(r6)
              and r5["declared_parameters"] == r6["declared_parameters"],
              f"{name}: record keys (beyond {ADDED}) or declared parameters differ")
        ll1 = r6["pipeline_switches"]["LL-1"]
        check("G16a", ll1["requested"] == {} and ll1["applied_schema_check"] == "off",
              f"{name}: off run records an LL-1 setting: {ll1}")
        check("G16a", not any("value_scores" in p for p in L.lines_of(o6 / "pools.jsonl")),
              "off wrote value_scores")
        out[name] = "equal"
    return out


def g16b_learned(F, out_root, light=False):
    TAG["g"] = "G16b"
    G = load_grader()
    n = F["n_rows"]
    end = "260" if light else "301"
    extra = ["--models-dir", str(F["models"]["lightgbm"]), "--block-start", "250",
             "--block-end", end, "--workers", "1" if light else "3"]
    if not light:   # t = 300 is the pending draw; the harness requires its declared identity
        extra += ["--pending-date", T.PENDING_DATE, "--pending-session", T.PENDING_SESSION]
    rec, out, pools, checked = oracle_check(F, out_root, extra, "g16b", n)
    if light:
        return checked
    off, _ = run(F, out_root, extra, "g16b_off")
    check("G16b", off["config_id"] != rec["config_id"], "ordering on did not enter config_id")
    gb = 0
    for i, p in enumerate(pools, 1):
        if p["abstain"]:
            continue
        e = {"source_line": i, "value_scores": p["value_scores"], "value_ranking": None,
             "raw_pool": p["pool"]}
        ranks, src = G.draw_ranks(e, 1000, "record_only")
        check("G16b", src == "record_value_scores", f"grader used {src}")
        distinct = list(dict.fromkeys(p["pool"]))
        check("G16b", [float(ranks[v]) for v in distinct] == [float(j + 1) for j in range(len(distinct))],
              f"t={p['target']['combined_index']}: grader ranks of the pool's values are not 1..d")
        check("G16b", all(ranks[v] > len(distinct) for v in range(1000) if v not in set(distinct)),
              "a value outside the pool ranks inside the top d")
        gb += 1
    check("G16b", H.sha256_file(GRADER) == GRADER_SHA, "grader changed during the test")
    vo = rec["value_ordering"]
    return (f"{checked} records = module full vote ({n} survivors); grader draw_ranks top-k = pool on "
            f"{gb}; pending t=300 included; matrix {vo['matrix_shape']} in {vo['matrix_seconds']}s")


def g16c_per_skip(F, out_root):
    TAG["g"] = "G16c"
    extra = ["--models-dir", str(F["models"]["lightgbm"]), "--vote-arm", "per_skip",
             "--arm-skip", "3", "--block-start", "270", "--block-end", "280", "--workers", "1"]
    _, _, _, checked = oracle_check(F, out_root, extra, "g16c", F["n_rows"])
    return f"{checked} records = module full vote at skip 3"


def g16d_a0(F, out_root):
    TAG["g"] = "G16d"
    extra = ["--ranking", "field_sum", "--ranking-fields", "forward_matches,reverse_matches",
             "--ranking-npz", str(F["fx"] / "step1_survivors.npz"), "--tie-break", "seed_asc",
             "--block-start", "270", "--block-end", "300", "--workers", "2"]
    _, _, _, checked = oracle_check(F, out_root, extra, "g16d", F["n_rows"])
    return f"{checked} records = module full vote (field_sum, seed_asc)"


def noisy_npz(F):
    """forward+reverse sums in classes of 0.05; inside each class, members differ by float noise
    of 1e-10 (in either direction), so only rounding makes them tie."""
    import numpy as np
    p = F["fx"] / "step1_noisy.npz"
    if p.exists():
        return p
    z = np.load(F["fx"] / "step1_survivors.npz")
    seeds = z["seeds"]
    n = len(seeds)
    rs = np.random.RandomState(5)
    cls = rs.randint(0, 4, n) * 0.05 + 0.6
    noise = (rs.randint(0, 3, n) - 1) * 1e-10
    fm = (cls / 2 + noise).astype(np.float64)
    rm = (cls / 2).astype(np.float64)
    np.savez(p, seeds=seeds, forward_matches=fm, reverse_matches=rm)
    return p


def g17_rounding(F, out_root):
    TAG["g"] = "G17"
    import numpy as np
    npz = noisy_npz(F)
    base = ["--ranking", "field_sum", "--ranking-fields", "forward_matches,reverse_matches",
            "--ranking-npz", str(npz), "--tie-break", "seed_asc", "--block-start", "270",
            "--block-end", "275", "--workers", "1", "--emit-ranked-predictions", "on",
            "--ranked-predictions-depth", "20"]
    roff, ooff = run(F, out_root, base, "g17_off")
    ron, oon = run(F, out_root, base + ["--ranking-key-round-decimals", "6"], "g17_on")
    rows = json.loads((F["fx"] / "survivors_with_scores.json").read_text())
    z = np.load(npz)
    key = {int(s): float(a) + float(b) for s, a, b in zip(z["seeds"], z["forward_matches"], z["reverse_matches"])}
    def oracle(rnd):
        k = (lambda s: round(key[s], 6)) if rnd else (lambda s: key[s])
        order = sorted(range(len(rows)), key=lambda r: (-k(int(rows[r]["seed"])), int(rows[r]["seed"])))
        return [int(rows[r]["seed"]) for r in order][:20]
    got_off = [e["seed"] for e in L.lines_of(ooff / "ranked_predictions.jsonl")[0]["predictions"]]
    got_on = [e["seed"] for e in L.lines_of(oon / "ranked_predictions.jsonl")[0]["predictions"]]
    check("G17", got_off == oracle(False), "off order != raw float key order")
    check("G17", got_on == oracle(True), "rounded order != (rounded key desc, seed asc)")
    check("G17", got_on != got_off, "fixture noise did not change the order (test too weak)")
    ki = ron["ranking_info"]
    check("G17", ki.get("distinct_before_round", 0) > ki.get("distinct_after_round", 1e9),
          f"rounding did not merge classes: {ki}")
    check("G17", ki["distinct_after_round"] == 4, f"expected 4 classes, got {ki['distinct_after_round']}")
    check("G17", ron["config_id"] != roff["config_id"], "rounding did not enter config_id")
    check("G17", "ranking_key_round_decimals" not in roff["config_identity_inputs"], "off has the key")
    return (f"off: raw float order; on: {ki['distinct_before_round']} -> {ki['distinct_after_round']} "
            f"key values, seed_asc inside classes; config_id differs")


def g18_fail_closed(F, out_root):
    TAG["g"] = "G18"
    lg = ["--models-dir", str(F["models"]["lightgbm"]), "--block-start", "270", "--block-end", "272"]
    cases = [(lg + ["--ranking-key-round-decimals", "6"], "needs --ranking field_sum"),
             (["--ranking", "field_sum", "--ranking-fields", "forward_matches,reverse_matches",
               "--ranking-npz", str(F["fx"] / "step1_survivors.npz"), "--block-start", "270",
               "--block-end", "272", "--ranking-key-round-decimals", "-1"], "needs --ranking field_sum"),
             (["--models-dir", str(F["models"]["gate_blocked"]), "--block-start", "270",
               "--block-end", "272", "--emit-value-ordering", "full_vote"], "signal gate")]
    for argv, want in cases:
        _, msg = run(F, out_root, argv, "g18", expect_fail=True)
        check("G18", msg is not None and isinstance(msg, str) and want in msg, f"not refused as expected: {msg}")
    return f"{len(cases)} refusals"


def mutants(F, out_root):
    import numpy as np
    res = []

    def m22():
        hits = {"n": 0}; orig = H.value_scores_for
        def mut(col, n_values):
            hits["n"] += 1
            rev = col[::-1]
            vals, last = np.unique(rev, return_index=True)
            s = np.full(n_values, -(len(col) + 1), dtype=np.int64)
            s[vals] = -(len(col) - last)
            return s
        H.value_scores_for = mut
        return (lambda: setattr(H, "value_scores_for", orig)), hits
    res.append(T.run_mutant("M22 last occurrence instead of first", m22,
                            lambda: g16b_learned(F, out_root, light=True), "G16b"))

    def m23():
        hits = {"n": 0}; orig = H.full_vote_matrix
        def mut(gen, survivors, default_skip, vis, workers):
            hits["n"] += 1
            return orig(gen, [{k: v for k, v in r.items() if k != "skip"} for r in survivors],
                        default_skip, vis, workers)
        H.full_vote_matrix = mut
        return (lambda: setattr(H, "full_vote_matrix", orig)), hits
    res.append(T.run_mutant("M23 matrix ignores per-survivor skip", m23,
                            lambda: g16c_per_skip(F, out_root), "G16c"))

    def m24():
        hits = {"n": 0}; orig = H.build_key
        def mut(rows, ranking, fields, npz_map, round_decimals=None):
            if round_decimals is not None:
                hits["n"] += 1
            return orig(rows, ranking, fields, npz_map, None)
        H.build_key = mut
        return (lambda: setattr(H, "build_key", orig)), hits
    res.append(T.run_mutant("M24 rounding dropped", m24, lambda: g17_rounding(F, out_root), "G17"))
    return res


def cost_probe(out_root, n_rows=12156, block=600, end=18068, workers=6):
    """Synthetic run-H shape: 12,156 survivors, a 600-draw block ending at 18068, learned
    ranking (LightGBM on random features), ordering off and on. No run H data."""
    import numpy as np
    from models.wrappers.lightgbm_wrapper import LightGBMWrapper
    d = out_root / "cost"; d.mkdir()
    rs = np.random.RandomState(11)
    names = [f"f{i:02d}" for i in range(89)]
    Xf = rs.rand(n_rows, 89)
    rows = [{"seed": int(s), "score": 0.0, "features": {nm: float(v) for nm, v in zip(names, xr)},
             "metadata": {}} for s, xr in zip(rs.randint(0, 2 ** 31, n_rows), Xf)]
    (d / "survivors.json").write_text(json.dumps(rows))
    hist = [int(x) for x in rs.randint(0, 1000, end)]
    (d / "dataset.json").write_text(json.dumps(
        [{"date": f"d{i:05d}", "session": "midday" if i % 2 == 0 else "evening", "draw": v}
         for i, v in enumerate(hist)]))
    (d / "cfg.json").write_text(json.dumps({"prediction": {"pool_size": 20, "k": 20}}))
    m = d / "models"; m.mkdir()
    lg = LightGBMWrapper(config={"n_estimators": 100, "force_cpu": True}, device="cpu")
    lg.fit(Xf.astype(np.float32), rs.rand(n_rows).astype(np.float32))
    lg.save(str(m / "best_model.txt"))
    (m / "best_model.meta.json").write_text(json.dumps({
        "model_type": "lightgbm", "checkpoint_path": str(m / "best_model.txt"),
        "feature_schema": {"feature_names": names},
        "signal_quality": {"prediction_allowed": True, "signal_status": "healthy"}}))
    out = {}
    for mode in ("off", "full_vote"):
        o = d / f"out_{mode}"
        cmd = [sys.executable, "-u", str(T.HARNESS_PATH), "--survivors", str(d / "survivors.json"),
               "--dataset", str(d / "dataset.json"), "--data-cutoff-index", "t",
               "--config", str(d / "cfg.json"), "--models-dir", str(m),
               "--block-start", str(end - block), "--block-end", str(end), "--workers", str(workers),
               "--out-dir", str(o), "--emit-value-ordering", mode]
        t0 = time.time()
        p = subprocess.run(["/usr/bin/time", "-v"] + cmd, cwd=str(d), capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(p.stderr[-2000:])
        rss = [l.strip() for l in p.stderr.splitlines() if "Maximum resident" in l]
        rec = json.loads((o / "run_record.json").read_text())
        out[mode] = {"wall_s": round(time.time() - t0, 1), "vote_s": rec["timing"]["vote_seconds"],
                     "value_ordering": rec.get("value_ordering", {}).get("seconds"),
                     "matrix_s": rec.get("value_ordering", {}).get("matrix_seconds"),
                     "pools_jsonl_bytes": (o / "pools.jsonl").stat().st_size,
                     "max_rss": rss[0] if rss else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--cost-probe", action="store_true")
    a = ap.parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(a.out_root) if a.out_root else REPO / "data" / "e2e_vote_selftest" / f"ord_{stamp}"
    out_root = out_root.resolve(); out_root.mkdir(parents=True, exist_ok=False)
    t_all = time.time()
    print(f"[ord] out_root={out_root}", flush=True)
    F = T.build_fixture(out_root)
    summary = {"harness_sha256": H.sha256_file(T.HARNESS_PATH), "gates": {}, "mutants": []}
    results = []
    for name, fn in (("G16a off = rev 5", lambda: g16a_off(F, out_root)),
                     ("G16b learned: ordering = module full vote; grader top-k = pool", lambda: g16b_learned(F, out_root)),
                     ("G16c per_skip arm", lambda: g16c_per_skip(F, out_root)),
                     ("G16d A0 field_sum", lambda: g16d_a0(F, out_root)),
                     ("G17 key rounding", lambda: g17_rounding(F, out_root)),
                     ("G18 fail-closed", lambda: g18_fail_closed(F, out_root))):
        t = time.time()
        try:
            r = fn(); results.append(True); summary["gates"][name] = {"pass": True, "detail": r}
            print(f"PASS {name} ({time.time() - t:.1f}s): {r}", flush=True)
        except Exception as e:
            results.append(False); summary["gates"][name] = {"pass": False, "error": str(e)}
            print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)
    mres = mutants(F, out_root)
    for name, killed, msg in mres:
        print(f"{'KILLED' if killed else 'SURVIVED'} {name}: {msg}", flush=True)
        summary["mutants"].append({"name": name, "killed": killed, "detail": msg})
    if a.cost_probe:
        try:
            summary["cost_probe"] = cost_probe(out_root)
            print(f"[cost] {json.dumps(summary['cost_probe'])}", flush=True)
        except Exception as e:
            summary["cost_probe"] = {"error": str(e)}; print(f"[cost] ERROR {e}", flush=True)
    summary["total_seconds"] = round(time.time() - t_all, 1)
    summary["peak_rss_mib"] = {
        "self": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "largest_child": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1)}
    ng, nk = sum(results), sum(1 for _, k, _ in mres if k)
    summary["verdict"] = {"gates": f"{ng}/{len(results)} pass", "mutants": f"{nk}/{len(mres)} killed"}
    (out_root / "ord_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[ord] gates {summary['verdict']['gates']}, mutants {summary['verdict']['mutants']}, "
          f"{summary['total_seconds']}s, peak RSS {summary['peak_rss_mib']} MiB", flush=True)
    return 0 if (ng == len(results) and nk == len(mres)) else 1


if __name__ == "__main__":
    sys.exit(main())
