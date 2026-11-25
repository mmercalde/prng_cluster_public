#!/usr/bin/env python3
"""
Self-test for scripts/diagnostics/e2e_miss_report.py, the pure core. CPU, synthetic votes only.
No run H data is read.

Run:  python3 -u tests/test_e2e_miss_report.py | tee <log>

Fixture: 400 survivors x 600 draws of votes on 0..999, with 40 duplicated vote rows (so pools
repeat values) and features:
  planted (per-seed; outcome voters are enriched in it on 35% of draws), noise1, noise2 (per-seed),
  global_varying (prefix global_, varies and tracks planted: it must never be seed evidence),
  global_const, run_const (one value in the run).
A null fixture has the same votes with uniform outcomes.

Gates: MG1 ranks = brute force; MG2 disjoint bands, counts sum to the draws, expected = brute-force
distinct prefixes / 1000; MG3 planted ranks first in C2 with an interval excluding 0, per-run
columns only as context; MG4 abstention on small cohorts; MG5 PIT uniform under the null, low
under the plant; MG6 concentration hit counts = brute force, Monte Carlo near N*draws/1000;
MG7 C1 = selection reference + remainder, the remainder near 0 when outcomes are uniform.
Mutants: MM1 nested bands, MM2 global counted as seed evidence, MM3 expectation N/1000 (repeats
ignored), MM4 abstention ignored, MM5 rank off by one.
"""
import importlib.util
import json
import math
import resource
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
P = REPO / "scripts" / "diagnostics" / "e2e_miss_report.py"
spec = importlib.util.spec_from_file_location("e2e_miss_report", P)
M = importlib.util.module_from_spec(spec)
sys.modules["e2e_miss_report"] = M
spec.loader.exec_module(M)

import numpy as np   # noqa: E402


class GateFail(AssertionError):
    def __init__(self, tag, msg):
        super().__init__(f"[{tag}] {msg}")
        self.tag = tag


def check(tag, cond, msg):
    if not cond:
        raise GateFail(tag, msg)


RULES = {"min_pairs": 20, "min_draws": 10, "min_survivors": 5}
NAMES = ["global_const", "global_varying", "noise1", "noise2", "planted", "run_const"]


def fixture(planted=True, seed=7):
    rng = np.random.default_rng(seed)
    n, d = 400, 600
    V = rng.integers(0, 1000, size=(n, d)).astype(np.int16)
    V[360:400] = V[0:40]                       # duplicated rows: repeats inside pools
    pl = rng.normal(size=n)
    F = np.column_stack([np.full(n, 3.0), pl + 0.1 * rng.normal(size=n), rng.normal(size=n),
                         rng.normal(size=n), pl, np.full(n, -1.0)])
    o = rng.integers(0, 1000, size=d)
    if planted:
        w = np.exp(2.0 * (pl - pl.mean()) / pl.std()); w /= w.sum()
        for t in range(d):
            if rng.random() < 0.35:
                o[t] = V[rng.choice(n, p=w), t]
    order = np.argsort(-rng.random(n))          # an arbitrary ranking
    return V, o, F, order


def brute_rank(col, o):
    for i, v in enumerate(col):
        if v == o:
            return i + 1
    return 0


def mg1(V, o, order):
    Vr = V[order]
    R = M.first_ranks(Vr, o)
    want = [brute_rank(Vr[:, t], o[t]) for t in range(V.shape[1])]
    check("MG1", R.tolist() == want, "first_ranks != brute force")
    Dr = M.distinct_rank(Vr, o, R)
    for t in range(0, V.shape[1], 7):
        if want[t]:
            seen = []
            for v in Vr[:, t]:
                if v not in seen:
                    seen.append(v)
            check("MG1", Dr[t] == seen.index(o[t]) + 1, f"distinct rank t={t}")
    return f"{V.shape[1]} ranks = brute force, {int(np.sum(R == 0))} never voted"


def mg2(V, o, order):
    Vr = V[order]
    n = V.shape[0]
    pool_sizes = [20, 100, 300]
    edges = M.band_edges(pool_sizes, n)
    cuts = sorted({0, n} | set(pool_sizes))
    D = M.distinct_counts(Vr, cuts)
    D_at = {c: D[:, j] for j, c in enumerate(cuts)}
    R = M.first_ranks(Vr, o)
    tab = M.band_table(R, D_at, edges)
    los = [r["lo"] for r in tab[:-1]]; his = [r["hi"] for r in tab[:-1]]
    check("MG2", all(his[i] == los[i + 1] for i in range(len(los) - 1)) and los[0] == 0,
          f"bands not disjoint and contiguous: {list(zip(los, his))}")
    check("MG2", sum(r["observed"] for r in tab) == V.shape[1], "band counts do not sum to draws")
    check("MG2", abs(sum(r["expected"] for r in tab) - V.shape[1]) < 1e-6, "expected does not sum to draws")
    for r in tab[:-1]:
        brute = sum((len(set(Vr[:r["hi"], t].tolist())) - len(set(Vr[:r["lo"], t].tolist()))) / 1000
                    for t in range(V.shape[1]))
        check("MG2", abs(brute - r["expected"]) < 1e-9, f"band {r['band']}: expected {r['expected']} != {brute}")
    naive = 300 * V.shape[1] / 1000
    exp300 = sum(r["expected"] for r in tab[:3])
    check("MG2", naive - exp300 > 1.0, f"fixture has too few repeats to test dedup ({naive} vs {exp300})")
    return {r["band"]: (r["observed"], round(r["expected"], 2)) for r in tab}


def analyse(F, names, prefixes):
    per_seed, per_run = M.classify_columns(names, F, prefixes)
    Z = M.standardise(F[:, [names.index(x) for x in per_seed]])
    return per_seed, per_run, Z


def mg3(V, o, F, order):
    per_seed, per_run, Z = analyse(F, NAMES, ["global_"])
    run_names = {r["name"] for r in per_run}
    check("MG3", run_names == {"global_const", "global_varying", "run_const"},
          f"per-run set {sorted(run_names)}")
    check("MG3", not (run_names & set(per_seed)), "a per-run column is in the seed set")
    EQ = V == o[None, :]
    c2 = M.contrast_c2(EQ, Z, per_seed, RULES, 400, np.random.default_rng(1))
    feats = c2["features"]
    check("MG3", not any(f["feature"] in run_names for f in feats), "per-run column reported as seed evidence")
    check("MG3", feats[0]["feature"] == "planted" and feats[0]["ci_excludes_0"],
          f"planted not first with an interval excluding 0: {feats[:2]}")
    cc = M.contrasts(EQ, Z, order[:100], per_seed, RULES, 400, np.random.default_rng(2))
    for key in ("C1", "C1b"):
        if not cc[key]["abstained"]:
            check("MG3", not any(f["feature"] in run_names for f in cc[key]["features"]),
                  f"{key}: per-run column reported")
    return {"c2_top": feats[0], "per_run": sorted(run_names)}


def mg4(V, o, F, order):
    per_seed, _, Z = analyse(F, NAMES, ["global_"])
    EQ = V == o[None, :]
    big = dict(RULES, min_pairs=10 ** 6)
    cc = M.contrasts(EQ, Z, order[:20], per_seed, big, 100, np.random.default_rng(3))
    check("MG4", cc["C1"]["abstained"] and "features" not in cc["C1"], "C1 did not abstain")
    check("MG4", cc["C1b"]["abstained"] and "features" not in cc["C1b"], "C1b did not abstain")
    c2 = M.contrast_c2(EQ[:, :3], Z, per_seed, RULES, 100, np.random.default_rng(4))
    check("MG4", c2["abstained"], "C2 on 3 draws did not abstain")
    ok = M.contrasts(EQ, Z, order[:300], per_seed, RULES, 100, np.random.default_rng(5))
    check("MG4", not ok["C1"]["abstained"], f"C1 abstained on a large cohort: {ok['C1'].get('abstain_reasons')}")
    return {"small": "abstained", "C1_large_H_pairs": ok["C1"]["H"]["pairs"]}


def mg5(V, o_null, o_pl, order):
    from scipy.stats import kstest
    Vr = V[order]; n = V.shape[0]
    res = {}
    for name, o in (("null", o_null), ("planted", o_pl)):
        R = M.first_ranks(Vr, o)
        Dr = M.distinct_rank(Vr, o, R)
        Dn = M.distinct_counts(Vr, [n])[:, 0]
        u = M.pit_values(Dr, Dn, np.random.default_rng(6))
        res[name] = (float(kstest(u, "uniform").pvalue), float(u.mean()))
    check("MG5", res["null"][0] > 0.01, f"null PIT not uniform: {res['null']}")
    return res


def mg6(V, o, order):
    N = 100
    Vt = V[order[:N]]
    s = np.sort(np.random.default_rng(8).normal(size=V.shape[0]))[::-1]
    c = M.concentration(Vt, o, s, N, 300, np.random.default_rng(9), 0.01)
    brute = sum(int(Vt[i, t] == o[t]) for i in range(N) for t in range(V.shape[1]))
    check("MG6", c["hit_spread"]["observed"]["hit_pairs"] == brute, "hit pairs != brute force")
    mc = c["hit_spread"]["monte_carlo_uniform_draws"]["hit_pairs"]["mean"]
    check("MG6", abs(mc - N * V.shape[1] / 1000) < 3, f"MC hit pairs mean {mc} far from {N * V.shape[1] / 1000}")
    return {"hit_pairs": brute, "mc_mean": mc}


def mg7(V, o_null, F):
    """Ranking selects on noise1; outcomes uniform. C1 for noise1 is large (selection), the
    selection reference carries it, and the remainder mean(H) - mean(top N) does not."""
    per_seed, _, Z = analyse(F, NAMES, ["global_"])
    j = per_seed.index("noise1")
    order = np.argsort(-F[:, NAMES.index("noise1")])
    EQ = V == o_null[None, :]
    cc = M.contrasts(EQ, Z, order[:100], per_seed, RULES, 400, np.random.default_rng(11))["C1"]
    get = lambda rows: next(r for r in rows if r["feature"] == "noise1")
    c1, sel, rem = get(cc["features"]), get(cc["selection_reference"]), get(cc["minus_selection"])
    check("MG7", c1["ci_excludes_0"] and c1["point"] > 1, f"C1 noise1 not large: {c1}")
    check("MG7", abs(c1["point"] - (sel["point"] + rem["point"])) < 1e-9, "C1 != selection + remainder")
    check("MG7", not rem["ci_excludes_0"], f"remainder excludes 0 under uniform outcomes: {rem}")
    return {"C1": round(c1["point"], 3), "selection": round(sel["point"], 3), "remainder": rem}


def run_mutant(name, apply, gate, tag):
    restore, hits = apply()
    try:
        try:
            gate()
            return name, False, "gate passed under mutant (SURVIVED)"
        except GateFail as e:
            return name, e.tag == tag and hits["n"] > 0, f"tag={e.tag} hits={hits['n']} :: {str(e)[:110]}"
        except Exception as e:
            return name, False, f"uncredited {type(e).__name__}: {str(e)[:110]}"
    finally:
        restore()


def patch(attr, make):
    orig = getattr(M, attr)
    hits = {"n": 0}

    def wrapped(*a, **k):
        hits["n"] += 1
        return make(orig, *a, **k)
    setattr(M, attr, wrapped)
    return (lambda: setattr(M, attr, orig)), hits


def main():
    t0 = time.time()
    V, o, F, order = fixture(True)
    _, o_null, _, _ = fixture(False)
    gates = [("MG1 ranks", lambda: mg1(V, o, order)),
             ("MG2 disjoint bands, exact expectation", lambda: mg2(V, o, order)),
             ("MG3 planted per-seed signal first; per-run as context", lambda: mg3(V, o, F, order)),
             ("MG4 abstention", lambda: mg4(V, o, F, order)),
             ("MG5 PIT", lambda: mg5(V, o_null, o, order)),
             ("MG6 concentration", lambda: mg6(V, o, order)),
             ("MG7 C1 split into selection and remainder", lambda: mg7(V, o_null, F))]
    ok_all = True
    summary = {"gates": {}, "mutants": []}
    for name, fn in gates:
        try:
            r = fn()
            print(f"PASS {name}: {r}", flush=True); summary["gates"][name] = {"pass": True, "detail": r}
        except Exception as e:
            ok_all = False
            print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)
            summary["gates"][name] = {"pass": False, "error": str(e)}
    n = V.shape[0]
    muts = [
        ("MM1 nested bands", lambda: patch("band_edges", lambda f, ps, nn: [(0, p) for p in sorted(ps)] + [(0, nn)]),
         lambda: mg2(V, o, order), "MG2"),
        ("MM2 global counted as seed evidence", lambda: patch("classify_columns", lambda f, names, Fm, pre: f(names, Fm, [])),
         lambda: mg3(V, o, F, order), "MG3"),
        ("MM3 expectation N/1000 (repeats ignored)",
         lambda: patch("distinct_counts", lambda f, Vr, cuts: np.tile(np.minimum(np.asarray(cuts), Vr.shape[0]), (Vr.shape[1], 1))),
         lambda: mg2(V, o, order), "MG2"),
        ("MM4 abstention ignored", lambda: patch("cohort_ok", lambda f, *a: (True, [])),
         lambda: mg4(V, o, F, order), "MG4"),
        ("MM5 rank off by one", lambda: patch("first_ranks", lambda f, Vr, oo: np.where(f(Vr, oo) > 0, f(Vr, oo) - 1, 0)),
         lambda: mg1(V, o, order), "MG1"),
    ]
    nk = 0
    for name, ap, g, tag in muts:
        nm, killed, msg = run_mutant(name, ap, g, tag)
        nk += killed
        print(f"{'KILLED' if killed else 'SURVIVED'} {nm}: {msg}", flush=True)
        summary["mutants"].append({"name": nm, "killed": killed, "detail": msg})
    rss = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    ng = sum(1 for v in summary["gates"].values() if v["pass"])
    print(f"[miss-selftest] gates {ng}/{len(gates)} pass, mutants {nk}/{len(muts)} killed, "
          f"{time.time() - t0:.1f}s, peak RSS {rss} MiB", flush=True)
    return 0 if (ok_all and nk == len(muts)) else 1


if __name__ == "__main__":
    sys.exit(main())
