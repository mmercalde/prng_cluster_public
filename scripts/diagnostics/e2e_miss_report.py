#!/usr/bin/env python3
"""
e2e_miss_report.py - read-only miss report for run H (TASK_LOOP_MISS, window h6).

Status: DRAFT, uncommitted, Team Alpha (Claude), worktree e2e-vote, 2026-11-23.
Owner question: when a pool missed, what made it miss.

OBSERVE-ONLY. It writes nothing outside --out-dir, trains nothing, changes no setting and
grades nothing. Every number is DEVELOPMENT / DESCRIPTIVE evidence on blocks that have been
looked at already; none is a counted test. Nothing here closes or downgrades a lead (L-9).

INPUTS (all declared in one --spec JSON, sha256 in the run record):
  - the Step 3 rows (survivors_with_scores.json);
  - the H6 pool files and run records for each arm and pool size;
  - the grader's per_draw.jsonl files (the outcomes come ONLY from these: the grader already
    opened the sealed draws; this script never opens the dataset).

WHAT IS RECONSTRUCTED, AND HOW IT IS PROVEN
  The full vote: every one of the 12,156 survivors' vote on every draw of both blocks.
  - The ranking of each arm is rebuilt with the harness's own build_generator (module code)
    from the arm's recorded argv, scored once through _build_prediction_pool(pool_size=0)
    (prediction_generator.py:752-836; no vote is cast). The X sha256 and score sha256 must
    equal the run record's ranking_proof, or the run stops.
  - The order is the module's argsort (:836), via the harness's ranked_order.
  - Each survivor's vote at t is seq[t] of the module's own SurvivorScorer._generate_sequence
    (:846; survivor_scorer.py:119-125) with the module's per-survivor skip (:795).
  - Proof: for EVERY record of EVERY declared pool file, the first N votes of the rebuilt
    vote equal the pool, value for value. For every graded per_draw row, hit = outcome in
    the pool and the grader's k = distinct values. Any difference stops the run.

DEFINITIONS (declared; see --help of the spec fields)
  rank R_t   position (1-based, in ranked-survivor order) of the first survivor whose vote
             equals the drawn number; 0 = no survivor voted it. A pool of N hits iff
             1 <= R_t <= N, so pool 20/100/300 hits are R_t bands.
  expected   under the modelled background (drawn number uniform on 0..999, independent of
             the votes): P(R_t <= r) = D_t(r)/1000, D_t(r) = distinct values among the first
             r votes on draw t. This uses the actual votes, repeats included; it is exact for
             each draw under that assumption. Variance: sum of p(1-p) over draws.
  PIT u_t    (distinct rank of the drawn number - 1 + U)/1000 if voted, else
             (D_t(n) + U*(1000 - D_t(n)))/1000, U ~ U(0,1) seeded. Uniform on (0,1) under the
             background; low u = drawn number voted early.
  bands      DISJOINT survivor-rank bands (1-20, 21-100, 101-300, 301-n, never).
  cohorts    per arm, pool N, block, session:
             H  = (survivor, draw) pairs: survivor in the top N voted the drawn number (hit draws);
             M  = (survivor, draw) pairs: any survivor voted the drawn number on a miss draw
                  (all such survivors rank > N);
             C1 = mean(H) - mean(M), with its selection reference mean(top N) - mean(M) and
                  the remainder mean(H) - mean(top N); C1b = among the top N, voters of the drawn number minus
                  non-voters over all draws; C2 (rank-free, arm-free) = mean over all
                  outcome-voting pairs, background value 0.
             Features are standardised over all survivors (mean 0, sd 1, ddof 0).
             Bootstrap: resample draws with replacement (B, seed declared); percentile 95%.
             Abstain when a cohort has fewer than min_pairs pairs, min_draws draws or
             min_survivors distinct survivors.
  namespaces per-run = name starts with a declared prefix (default global_, set at
             full_scoring_worker.py:405, :460) OR one value across all rows of this run
             (measured). Per-run columns are context only: never reported as seed evidence.
  concentration  score margins and tie classes at N; vote agreement among the top N;
             how hits spread over the top-N survivors, against a Monte Carlo of uniform draws.

Shortcuts (TODO-E2E-n): TODO-E2E-M1 the ranking line np.argsort(scores)[::-1] is the module's
:836 reached through H.ranked_order (shared with LL-5), not called inside the module.
TODO-E2E-M2 the full vote is seq[t] of the module's generator for every survivor (the module
votes only ranked_idx[:pool_size], :841); identity with the module is proven on the top 300.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CPU only; set before any project import

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_PATH = REPO_ROOT / "scripts" / "diagnostics" / "e2e_vote_harness.py"
REPORT_FORMAT = "tfm.e2e.miss_report.v1"
MOD = 1000


class MissReportError(RuntimeError):
    pass


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_harness():
    spec = importlib.util.spec_from_file_location("e2e_vote_harness", HARNESS_PATH)
    m = importlib.util.module_from_spec(spec)
    sys.modules["e2e_vote_harness"] = m
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------
# pure core (tested by tests/test_e2e_miss_report.py)
# ---------------------------------------------------------------------------
def first_ranks(Vr, o):
    """Vr: (n_ranked, n_draws) votes in ranked order. o: (n_draws,). Returns R (1-based, 0 =
    never voted)."""
    import numpy as np
    eq = Vr == o[None, :]
    anyv = eq.any(axis=0)
    R = np.where(anyv, eq.argmax(axis=0) + 1, 0)
    return R.astype(np.int64)


def first_occurrence_positions(col):
    """Sorted 0-based positions where a value appears for the first time in col."""
    import numpy as np
    _, idx = np.unique(col, return_index=True)
    return np.sort(idx)


def distinct_counts(Vr, cuts):
    """D[t, j] = distinct values among the first cuts[j] votes of draw t."""
    import numpy as np
    n_draws = Vr.shape[1]
    D = np.zeros((n_draws, len(cuts)), dtype=np.int64)
    cuts = np.asarray(cuts)
    for t in range(n_draws):
        fo = first_occurrence_positions(Vr[:, t])
        D[t] = np.searchsorted(fo, cuts, side="left")
    return D


def distinct_rank(Vr, o, R):
    """1-based position of the drawn number among the distinct values in first-appearance
    order; 0 if never voted."""
    import numpy as np
    out = np.zeros(len(o), dtype=np.int64)
    for t in range(len(o)):
        if R[t] > 0:
            fo = first_occurrence_positions(Vr[:, t])
            out[t] = int(np.searchsorted(fo, R[t] - 1, side="left")) + 1
    return out


def band_edges(pool_sizes, n):
    edges = [0] + sorted(pool_sizes) + [n]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def band_table(R, D_at, edges):
    """R: ranks; D_at: dict cut -> (n_draws,) distinct counts at that cut (cut 0 = zeros).
    Disjoint bands (lo, hi] plus 'never'. Expected from D/1000 per draw."""
    import numpy as np
    rows = []
    for lo, hi in edges:
        p = (D_at[hi] - D_at[lo]) / MOD
        obs = int(np.sum((R > lo) & (R <= hi)))
        rows.append({"band": f"{lo + 1}-{hi}", "lo": lo, "hi": hi, "observed": obs,
                     "expected": float(p.sum()), "null_sd": float(math.sqrt(np.sum(p * (1 - p))))})
    n_hi = edges[-1][1]
    p = 1 - D_at[n_hi] / MOD
    rows.append({"band": "never voted", "lo": n_hi, "hi": None, "observed": int(np.sum(R == 0)),
                 "expected": float(p.sum()), "null_sd": float(math.sqrt(np.sum(p * (1 - p))))})
    for r in rows:
        r["z"] = (r["observed"] - r["expected"]) / r["null_sd"] if r["null_sd"] > 0 else None
    return rows


def pit_values(Dr, Dn, rng):
    import numpy as np
    U = rng.random(len(Dr))
    return np.where(Dr > 0, (Dr - 1 + U) / MOD, (Dn + U * (MOD - Dn)) / MOD)


def classify_columns(names, F, prefixes):
    """per-run: declared prefix or one value across all rows (measured); else per-seed."""
    import numpy as np
    per_seed, per_run = [], []
    for j, n in enumerate(names):
        col = F[:, j]
        constant = bool(np.all(col == col[0]))
        by_prefix = any(n.startswith(p) for p in prefixes)
        if by_prefix or constant:
            per_run.append({"name": n, "rule": ("prefix" if by_prefix else "") +
                            ("+" if by_prefix and constant else "") +
                            ("constant_in_run" if constant else ""),
                            "distinct_values": int(len(np.unique(col)))})
        else:
            per_seed.append(n)
    return per_seed, per_run


def standardise(F):
    import numpy as np
    mu = F.mean(axis=0)
    sd = F.std(axis=0)
    if np.any(sd == 0):
        raise MissReportError("standardise: a constant column reached the per-seed matrix")
    return (F - mu) / sd


def boot_weights(n, B, rng):
    import numpy as np
    return rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.float64)


def ratio_stat(S, c, W):
    """Point and bootstrap of sum(S)/sum(c); S (n_draws, F), c (n_draws,), W (B, n_draws)."""
    import numpy as np
    pt = S.sum(axis=0) / c.sum() if c.sum() > 0 else np.full(S.shape[1], np.nan)
    num = W @ S
    den = W @ c
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = num / den[:, None]
    return pt, bs


def interval(bs, level=0.95):
    import numpy as np
    a = (1 - level) / 2
    lo = np.nanpercentile(bs, 100 * a, axis=0)
    hi = np.nanpercentile(bs, 100 * (1 - a), axis=0)
    return lo, hi


def cohort_ok(n_pairs, n_draws, n_surv, rules):
    reasons = []
    if n_pairs < rules["min_pairs"]:
        reasons.append(f"pairs {n_pairs} < {rules['min_pairs']}")
    if n_draws < rules["min_draws"]:
        reasons.append(f"draws {n_draws} < {rules['min_draws']}")
    if n_surv < rules["min_survivors"]:
        reasons.append(f"distinct survivors {n_surv} < {rules['min_survivors']}")
    return (not reasons), reasons


def feature_rows(names, pt, lo, hi):
    rows = []
    for j, n in enumerate(names):
        rows.append({"feature": n, "point": float(pt[j]), "ci95": [float(lo[j]), float(hi[j])],
                     "ci_excludes_0": bool(lo[j] > 0 or hi[j] < 0)})
    rows.sort(key=lambda r: -abs(r["point"]))
    return rows


def contrasts(EQ, Z, top, names, rules, B, rng):
    """EQ: (n_surv, n_draws) bool, survivor voted the drawn number (survivor row order).
    Z: (n_surv, F) standardised per-seed features. top: row indices of the top N (ranked).
    Returns C1 and C1b dicts."""
    import numpy as np
    n_draws = EQ.shape[1]
    W = boot_weights(n_draws, B, rng)
    EQt = EQ[top]                                   # (N, n_draws)
    hit = EQt.any(axis=0)
    EQf = EQ.astype(np.float64)
    S_all = EQf.T @ Z                               # (n_draws, F): sum over outcome voters
    c_all = EQf.sum(axis=0)
    Zt = Z[top]
    S_top = EQt.astype(np.float64).T @ Zt           # voters within the top N
    c_top = EQt.sum(axis=0).astype(np.float64)
    out = {}
    # C1: H (top-N voters on hit draws) - M (all voters on miss draws)
    SH = S_top * hit[:, None]; cH = c_top * hit
    SM = S_all * (~hit)[:, None]; cM = c_all * (~hit)
    h_surv = int(np.unique(np.nonzero(EQt[:, hit])[0]).size)
    m_surv = int(np.unique(np.nonzero(EQ[:, ~hit])[0]).size)
    okH, rH = cohort_ok(int(cH.sum()), int(hit.sum()), h_surv, rules)
    okM, rM = cohort_ok(int(cM.sum()), int((~hit).sum()), m_surv, rules)
    c1 = {"H": {"pairs": int(cH.sum()), "draws": int(hit.sum()), "distinct_survivors": h_surv},
          "M": {"pairs": int(cM.sum()), "draws": int((~hit).sum()), "distinct_survivors": m_surv}}
    if okH and okM:
        pH, bH = ratio_stat(SH, cH, W)
        pM, bM = ratio_stat(SM, cM, W)
        lo, hi = interval(bH - bM)
        c1["features"] = feature_rows(names, pH - pM, lo, hi)
        c1["abstained"] = False
        # selection reference: the top-N survivors' own mean (what the ranking picked) - M.
        # C1 - selection = mean(H) - mean(top N): what separates the top-N survivors that
        # voted the drawn number from the top N as a whole.
        sel = Zt.mean(axis=0)
        lo_s, hi_s = interval(sel[None, :] - bM)
        c1["selection_reference"] = feature_rows(names, sel - pM, lo_s, hi_s)
        lo_r, hi_r = interval(bH - sel[None, :])
        c1["minus_selection"] = feature_rows(names, pH - sel, lo_r, hi_r)
    else:
        c1["abstained"] = True
        c1["abstain_reasons"] = {"H": rH, "M": rM}
    out["C1"] = c1
    # C1b: among the top N, voters minus non-voters (all draws)
    N = len(top)
    Ssum = Zt.sum(axis=0)[None, :]
    SV, cV = S_top, c_top
    SNV, cNV = Ssum - S_top, N - c_top
    v_surv = int(np.unique(np.nonzero(EQt)[0]).size)
    okV, rV = cohort_ok(int(cV.sum()), int((cV > 0).sum()), v_surv, rules)
    c1b = {"voters": {"pairs": int(cV.sum()), "draws": int((cV > 0).sum()),
                      "distinct_survivors": v_surv},
           "non_voters": {"pairs": int(cNV.sum())}}
    if okV:
        pV, bV = ratio_stat(SV, cV, W)
        pN, bN = ratio_stat(SNV, cNV, W)
        lo, hi = interval(bV - bN)
        c1b["features"] = feature_rows(names, pV - pN, lo, hi)
        c1b["abstained"] = False
    else:
        c1b["abstained"] = True
        c1b["abstain_reasons"] = rV
    out["C1b"] = c1b
    return out


def contrast_c2(EQ, Z, names, rules, B, rng):
    import numpy as np
    n_draws = EQ.shape[1]
    W = boot_weights(n_draws, B, rng)
    EQf = EQ.astype(np.float64)
    S, c = EQf.T @ Z, EQf.sum(axis=0)
    n_surv = int(np.unique(np.nonzero(EQ)[0]).size)
    ok, r = cohort_ok(int(c.sum()), int((c > 0).sum()), n_surv, rules)
    res = {"pairs": int(c.sum()), "draws_with_a_voter": int((c > 0).sum()),
           "distinct_survivors": n_surv,
           "background_value": 0.0,
           "background_basis": "drawn number uniform and independent of the votes: the expected "
                               "sum per draw is n*mean(z)/1000 = 0"}
    if ok:
        pt, bs = ratio_stat(S, c, W)
        lo, hi = interval(bs)
        res["features"] = feature_rows(names, pt, lo, hi)
        res["abstained"] = False
    else:
        res["abstained"] = True
        res["abstain_reasons"] = r
    return res


def multiplicity(rows, level=0.05):
    if rows is None:
        return None
    n = len(rows)
    k = sum(1 for r in rows if r["ci_excludes_0"])
    return {"features": n, "ci95_excluding_0": k,
            "modelled_count_if_no_feature_differed": round(level * n, 2),
            "basis": "5% of features, if intervals were independent; the features are correlated, "
                     "so this is a rough reference, not a test"}


def concentration(Vt, o, raw_scores_sorted, N, mc_reps, rng, delta):
    """Vt: (N, n_draws) votes of the top N in ranked order. raw_scores_sorted: all raw scores
    descending."""
    import numpy as np
    n_draws = Vt.shape[1]
    s = raw_scores_sorted
    sd = float(np.std(s))
    sN = float(s[N - 1]); sN1 = float(s[N]) if N < len(s) else None
    score = {"score_at_N": sN, "score_at_N_plus_1": sN1,
             "gap_in_sd": (sN - sN1) / sd if (sN1 is not None and sd > 0) else None,
             "tie_class_at_N": int(np.sum(s == sN)),
             "tie_class_inside_pool": int(np.sum(s[:N] == sN)),
             f"survivors_within_{delta}_sd_of_score_at_N": int(np.sum(np.abs(s - sN) <= delta * sd)),
             "distinct_scores_all": int(len(np.unique(s)))}
    pairs = 0
    for t in range(n_draws):
        _, cnt = np.unique(Vt[:, t], return_counts=True)
        pairs += int(np.sum(cnt * (cnt - 1) // 2))
    tot_pairs = n_draws * N * (N - 1) // 2
    agree = {"equal_vote_pairs": pairs, "pairs_total": tot_pairs,
             "rate": pairs / tot_pairs if tot_pairs else None,
             "background_rate": 1 / MOD,
             "distinct_vote_sequences_in_top_N": int(len(np.unique(Vt, axis=0)))}
    eq = Vt == o[None, :]
    k_s = eq.sum(axis=1)

    def summ(k):
        ks = np.sort(k)[::-1]
        tot = int(ks.sum())
        return {"hit_pairs": tot, "survivors_with_a_hit": int(np.sum(k > 0)),
                "max_hits_one_survivor": int(ks[0]) if len(ks) else 0,
                "share_top5_survivors": float(ks[:5].sum() / tot) if tot else None}
    obs = summ(k_s)
    sims = {"survivors_with_a_hit": [], "max_hits_one_survivor": [], "hit_pairs": []}
    for _ in range(mc_reps):
        oo = rng.integers(0, MOD, size=n_draws)
        ss = summ((Vt == oo[None, :]).sum(axis=1))
        for k in sims:
            sims[k].append(ss[k])
    mc = {k: {"mean": float(np.mean(v)), "p2_5": float(np.percentile(v, 2.5)),
              "p97_5": float(np.percentile(v, 97.5))} for k, v in sims.items()}
    hits = {"observed": obs, "monte_carlo_uniform_draws": mc, "reps": mc_reps,
            "hitting_survivors_rank_ordered": [int(i) + 1 for i in np.nonzero(k_s)[0]],
            "hits_by_rank": {str(int(i) + 1): int(k_s[i]) for i in np.nonzero(k_s)[0]}}
    return {"score": score, "vote_agreement": agree, "hit_spread": hits}


# ---------------------------------------------------------------------------
# reconstruction (module code)
# ---------------------------------------------------------------------------
def rebuild_ranking(H, pg, rec):
    """The arm's ranking, from its run record argv, through the harness's build_generator."""
    import logging
    import numpy as np
    args = H.parse_args(rec["argv"][1:])
    cwd = os.getcwd()
    os.chdir(rec["cwd"])
    try:
        rows = H.load_survivor_rows(args.survivors)
        npz_map = None
        if args.ranking_npz:
            npz_map, _ = H._load_npz_fields(args.ranking_npz, args.ranking_fields or [])
        config = H.build_config(pg, args)
        logger = logging.getLogger("e2e_miss_report.pg")
        logger.setLevel(logging.WARNING)
        gen, rank, info = H.build_generator(pg, config, logger, args, rows, npz_map)
        # compared after build_generator: it sets require_feature_schema=False for unlearned
        # arms, and the harness records the config after that (miss run1 failed closed here)
        if config.to_dict() != rec["effective_config"]:
            raise MissReportError("rebuilt effective config differs from the run record")
        for flag, key in (("expect_sidecar_sha256", "sidecar_sha256"),
                          ("expect_checkpoint_sha256", "checkpoint_sha256")):
            want = getattr(args, flag)
            if want is not None and info.get(key) != want:
                raise MissReportError(f"{key} {info.get(key)} != declared {want}")
        gen._build_prediction_pool(rows, [], 0)    # scores once, casts no vote (:841 [:0])
    finally:
        os.chdir(cwd)
    rp = rec["ranking_proof"]
    if rank.x_sha256 != rp["x_sha256"] or rank.scores_sha256() != rp["scores_sha256"]:
        raise MissReportError("rebuilt ranking X/scores sha256 differ from the run record")
    order = np.asarray(H.ranked_order(rank.scores))
    skips = [gen._get_survivor_skip(r, config.default_skip) for r in rows]
    return {"rows": rows, "order": order, "raw_scores": np.asarray(rank.raw_scores, dtype=np.float64),
            "scores_sha256": rp["scores_sha256"], "x_sha256": rp["x_sha256"],
            "rank_info": info, "gen": gen, "skips": skips, "config": config}


_VW = {}


def _vote_rows(chunk):
    import numpy as np
    sc, rows, skips, n_len, t0 = _VW["scorer"], _VW["rows"], _VW["skips"], _VW["n_len"], _VW["t0"]
    out = np.empty((len(chunk), n_len - t0), dtype=np.int16)
    for j, r in enumerate(chunk):
        seq = sc._generate_sequence(int(rows[r]["seed"]), n_len, skip=skips[r])
        if len(seq) < n_len:
            raise MissReportError(f"sequence for row {r} shorter than {n_len}")
        out[j] = seq[t0:n_len]
    return chunk[0], out


def full_vote(gen, rows, skips, t0, t1, workers):
    """V[row, t - t0] = the module's seq[t] for every survivor row, t in [t0, t1)."""
    import multiprocessing as mp
    import numpy as np
    n = len(rows)
    V = np.empty((n, t1 - t0), dtype=np.int16)
    _VW.update(scorer=gen.scorer, rows=rows, skips=skips, n_len=t1, t0=t0)
    idx = list(range(n))
    size = max(1, -(-n // (workers * 8)))
    chunks = [idx[i:i + size] for i in range(0, n, size)]
    if workers <= 1:
        res = map(_vote_rows, chunks)
        for start, block in res:
            V[start:start + len(block)] = block
    else:
        with mp.get_context("fork").Pool(workers) as p:
            for start, block in p.imap_unordered(_vote_rows, chunks):
                V[start:start + len(block)] = block
    return V


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Read-only miss report (TASK_LOOP_MISS).")
    p.add_argument("--spec", required=True, help="Declared spec JSON (inputs, rules, seeds).")
    p.add_argument("--out-dir", required=True, help="New directory; must not exist.")
    p.add_argument("--workers", type=int, default=6, help="Forked workers (default 6).")
    return p.parse_args(argv)


def main(argv=None):
    import numpy as np
    t_start = time.time()
    a = parse_args(argv)
    out = Path(a.out_dir)
    if out.exists():
        raise MissReportError(f"--out-dir {out} exists")
    spec = json.loads(Path(a.spec).read_text())
    rules = spec["abstain"]
    B = int(spec["bootstrap"]["B"])
    rng_master = np.random.default_rng(int(spec["seed"]))
    sys.path.insert(0, str(REPO_ROOT))
    H = load_harness()
    import logging
    logging.basicConfig(level=logging.WARNING)
    import prediction_generator as pg
    import utils.survivor_loader as sl
    signatures = H.check_signatures(pg, sl)       # inspect.signature, CLAUDE.md 1.7

    # ---- survivors and features
    surv_path = spec["survivors"]["path"]
    if sha256_file(surv_path) != spec["survivors"]["sha256"]:
        raise MissReportError("survivors sha256 differs from the spec")
    # ---- arms: rebuild rankings, check every pool record
    arms = {}
    t0, t1 = spec["vote_range"]
    V, base_rows, base_gen, base_skips = None, None, None, None
    for arm_id, arm in spec["arms"].items():
        files = []
        R0 = None
        for f in arm["pool_files"]:
            rec = json.loads((Path(f["run_dir"]) / "run_record.json").read_text())
            pp = Path(f["run_dir"]) / "pools.jsonl"
            if sha256_file(pp) != f["pools_sha256"] or rec["pools_jsonl"]["sha256"] != f["pools_sha256"]:
                raise MissReportError(f"{pp}: sha256 differs from spec/run record")
            if rec["effective_config"]["pool_size"] != f["pool_size"]:
                raise MissReportError(f"{pp}: pool size differs")
            if R0 is None:
                R0 = rebuild_ranking(H, pg, rec)
            elif rec["ranking_proof"]["scores_sha256"] != R0["scores_sha256"]:
                raise MissReportError(f"{pp}: ranking differs within arm {arm_id}")
            files.append((f, rec, pp))
        if base_rows is None:
            base_rows, base_gen, base_skips = R0["rows"], R0["gen"], R0["skips"]
            ts_v = time.time()
            V = full_vote(base_gen, base_rows, base_skips, t0, t1, a.workers)
            vote_seconds = time.time() - ts_v
        else:
            if [int(r["seed"]) for r in R0["rows"]] != [int(r["seed"]) for r in base_rows]:
                raise MissReportError("arms use different survivor rows")
            if R0["skips"] != base_skips:
                raise MissReportError("arms use different skips")
        order = R0["order"]
        checked = 0
        for f, rec, pp in files:
            N = f["pool_size"]
            for line in pp.read_text().splitlines():
                r = json.loads(line)
                t = r["target"]["combined_index"]
                if r["abstain"]:
                    raise MissReportError(f"{pp} t={t}: abstention; not handled")
                want = V[order[:N], t - t0].astype(int).tolist()
                if r["pool"] != want:
                    raise MissReportError(f"{pp} t={t}: rebuilt vote != pool")
                checked += 1
        arms[arm_id] = {"R": R0, "files": files, "pool_records_checked": checked}

    # ---- outcomes from the grader only, cross-checked
    pool_by_cid = {}
    for arm_id, A in arms.items():
        for f, rec, pp in A["files"]:
            pool_by_cid[rec["config_id"]] = (arm_id, f["pool_size"],
                                             {json.loads(x)["target"]["combined_index"]: json.loads(x)
                                              for x in pp.read_text().splitlines()})
    outcome, session_of, date_of = {}, {}, {}
    grade_files, recon = [], {"graded_rows": 0, "hit_agree": 0}
    for gdir in spec["grade_dirs"]:
        g = Path(gdir)
        pd = g / "per_draw.jsonl"
        sums = dict(reversed(l.split()) for l in (g / "SHA256SUMS").read_text().splitlines() if l.strip())
        sh = sha256_file(pd)
        if sums.get("per_draw.jsonl") != sh:
            raise MissReportError(f"{pd}: sha256 not as in SHA256SUMS")
        n_g = 0
        for line in pd.read_text().splitlines():
            d = json.loads(line)
            t = d["combined_index"]
            session_of.setdefault(t, d["session"]); date_of.setdefault(t, d["date"])
            if session_of[t] != d["session"]:
                raise MissReportError(f"t={t}: session differs between grader files")
            if d["status"] != "graded":
                continue
            o = int(d["outcome"])
            if outcome.setdefault(t, o) != o:
                raise MissReportError(f"t={t}: outcome differs between grader files")
            if d["config_id"] not in pool_by_cid:
                raise MissReportError(f"{pd}: config_id {d['config_id'][:12]} not a declared pool file")
            arm_id, N, prec = pool_by_cid[d["config_id"]]
            pr = prec[t]
            if pr["pool_sha256"] != d["pool_sha256"]:
                raise MissReportError(f"{pd} t={t}: pool sha differs")
            if int(d["hit"]) != int(o in set(pr["pool"])) or int(d["k"]) != len(set(pr["pool"])):
                raise MissReportError(f"{pd} t={t}: grader hit/k differ from the pool")
            recon["graded_rows"] += 1; recon["hit_agree"] += 1; n_g += 1
        grade_files.append({"path": str(pd), "sha256": sh, "graded_rows": n_g})

    names = sorted({k for r in base_rows for k in r["features"]})
    Fm = np.array([[float(r["features"].get(n, np.nan)) for n in names] for r in base_rows])
    if not np.all(np.isfinite(Fm)):
        raise MissReportError("non-finite or missing Step 3 feature value")
    per_seed, per_run = classify_columns(names, Fm, spec["per_run_prefixes"])
    Z = standardise(Fm[:, [names.index(n) for n in per_seed]])

    pool_sizes = spec["pool_sizes"]
    n_surv = len(base_rows)
    edges = band_edges(pool_sizes, n_surv)
    cuts = sorted({0, n_surv} | set(pool_sizes) | set(spec["rank_grid"]))
    report = {"format": REPORT_FORMAT, "label": spec["label"], "cells": {}, "c2": {},
              "namespaces": {"per_seed": per_seed, "per_run": per_run,
                             "rule": f"per-run = prefix {spec['per_run_prefixes']} "
                                     "(full_scoring_worker.py:405, :460) or one value across all "
                                     "rows (measured)"},
              "reconstruction": {a_: {"pool_records_checked": A["pool_records_checked"],
                                      "scores_sha256": A["R"]["scores_sha256"],
                                      "x_sha256": A["R"]["x_sha256"],
                                      "model_type": A["R"]["rank_info"].get("model_type")}
                                 for a_, A in arms.items()},
              "grader_reconciliation": recon, "grade_files": grade_files}
    per_draw_lines = []

    # ---- C2 per block/session (arm-free)
    for bname, (b0, b1) in spec["blocks"].items():
        for sess in spec["sessions"]:
            ts = [t for t in range(b0, b1) if session_of.get(t) == sess]
            miss_o = [t for t in ts if t not in outcome]
            if miss_o:
                raise MissReportError(f"{bname}/{sess}: {len(miss_o)} draws without an outcome")
            cols = np.array([t - t0 for t in ts])
            o = np.array([outcome[t] for t in ts])
            EQ = V[:, cols] == o[None, :]
            rng = np.random.default_rng(rng_master.integers(1 << 62))
            report["c2"][f"{bname}/{sess}"] = dict(
                contrast_c2(EQ, Z, per_seed, rules, B, rng), draws=len(ts))
            report["c2"][f"{bname}/{sess}"]["multiplicity"] = multiplicity(
                report["c2"][f"{bname}/{sess}"].get("features"))

    # ---- per arm cells
    for arm_id, arm in spec["arms"].items():
        A = arms[arm_id]["R"]
        order = A["order"]
        raw_sorted = np.sort(A["raw_scores"])[::-1]
        for bname in arm["blocks"]:
            b0, b1 = spec["blocks"][bname]
            for sess in spec["sessions"]:
                ts = [t for t in range(b0, b1) if session_of.get(t) == sess]
                cols = np.array([t - t0 for t in ts])
                o = np.array([outcome[t] for t in ts])
                Vr = V[order][:, cols]
                R = first_ranks(Vr, o)
                D = distinct_counts(Vr, cuts)
                D_at = {c: D[:, j] for j, c in enumerate(cuts)}
                Dr = distinct_rank(Vr, o, R)
                rng = np.random.default_rng(rng_master.integers(1 << 62))
                u = pit_values(Dr, D_at[n_surv], rng)
                from scipy.stats import kstest
                ks = kstest(u, "uniform")
                hist = np.histogram(u, bins=10, range=(0, 1))[0]
                grid = [{"r": r, "observed_frac_R_le_r": float(np.mean((R > 0) & (R <= r))),
                         "expected_frac": float(np.mean(D_at[r] / MOD)),
                         "observed": int(np.sum((R > 0) & (R <= r))),
                         "expected": float(np.sum(D_at[r] / MOD)),
                         "null_sd": float(math.sqrt(np.sum((D_at[r] / MOD) * (1 - D_at[r] / MOD))))}
                        for r in spec["rank_grid"]]
                cell = {"arm": arm_id, "block": bname, "session": sess, "draws": len(ts),
                        "bands": band_table(R, D_at, edges),
                        "rank_grid": grid,
                        "rank_quantiles_observed_voted": (
                            {q: float(np.percentile(R[R > 0], q)) for q in (10, 25, 50, 75, 90)}
                            if np.any(R > 0) else None),
                        "pit": {"hist10": hist.tolist(), "expected_per_bin": len(ts) / 10,
                                "ks_D": float(ks.statistic), "ks_p": float(ks.pvalue),
                                "mean_u": float(np.mean(u)),
                                "basis": "u uniform on (0,1) if the drawn number is uniform and "
                                         "independent of the votes (assumed)"},
                        "hits_at": {str(N): int(np.sum((R > 0) & (R <= N))) for N in pool_sizes},
                        "expected_hits_at": {str(N): float(np.sum(D_at[N] / MOD)) for N in pool_sizes},
                        "cohorts": {}, "concentration": {}}
                EQ = V[:, cols] == o[None, :]
                for N in pool_sizes:
                    top = order[:N]
                    rng = np.random.default_rng(rng_master.integers(1 << 62))
                    cc = contrasts(EQ, Z, top, per_seed, rules, B, rng)
                    for key in ("C1", "C1b"):
                        cc[key]["multiplicity"] = multiplicity(cc[key].get("features"))
                    # which survivors voted the drawn number on hits (top N) and on misses
                    EQt = EQ[top]
                    hit = EQt.any(axis=0)
                    h_rows = np.nonzero(EQt[:, hit].any(axis=1))[0]
                    cc["H_survivors"] = [{"rank": int(i) + 1, "seed": int(base_rows[top[i]]["seed"]),
                                          "hit_draws": int(EQt[i, hit].sum())} for i in h_rows]
                    rank_pos = np.empty(n_surv, dtype=np.int64); rank_pos[order] = np.arange(1, n_surv + 1)
                    mr = rank_pos[np.nonzero(EQ[:, ~hit])[0]]
                    cc["M_voter_rank_quantiles"] = ({q: float(np.percentile(mr, q))
                                                     for q in (1, 10, 50, 90)} if mr.size else None)
                    cell["cohorts"][str(N)] = cc
                    rng = np.random.default_rng(rng_master.integers(1 << 62))
                    cell["concentration"][str(N)] = concentration(
                        V[top][:, cols], o, raw_sorted, N, int(spec["mc_reps"]), rng,
                        float(spec["score_delta_sd"]))
                report["cells"][f"{arm_id}/{bname}/{sess}"] = cell
                for j, t in enumerate(ts):
                    per_draw_lines.append({"arm": arm_id, "block": bname, "combined_index": t,
                                           "date": date_of[t], "session": sess,
                                           "outcome": int(o[j]), "rank": int(R[j]),
                                           "distinct_rank": int(Dr[j]), "pit_u": float(u[j]),
                                           **{f"hit_{N}": int(0 < R[j] <= N) for N in pool_sizes}})

    out.mkdir(parents=True)
    rp = out / "miss_report.json"
    rp.write_text(json.dumps(report, indent=1, default=float))
    pdp = out / "per_draw_ranks.jsonl"
    pdp.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in per_draw_lines))
    rec = {"format": "tfm.e2e.miss_report_run_record.v1", "label": spec["label"],
           "argv": sys.argv if argv is None else ["e2e_miss_report.py"] + list(argv),
           "spec": {"path": str(Path(a.spec).resolve()), "sha256": sha256_file(a.spec)},
           "code": {"miss_report_sha256": sha256_file(__file__),
                    "harness_sha256": sha256_file(HARNESS_PATH),
                    "prediction_generator_sha256": sha256_file(REPO_ROOT / "prediction_generator.py"),
                    "survivor_scorer_sha256": sha256_file(REPO_ROOT / "survivor_scorer.py")},
           "outputs": {"miss_report.json": sha256_file(rp), "per_draw_ranks.jsonl": sha256_file(pdp)},
           "vote_matrix": {"rows": int(V.shape[0]), "draws": int(V.shape[1]), "range": [t0, t1],
                           "seconds": round(vote_seconds, 1)},
           "workers": a.workers, "signatures": signatures,
           "timing_seconds": round(time.time() - t_start, 1),
           "peak_rss_mib": {"self": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                            "largest_child": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1)}}
    (out / "run_record.json").write_text(json.dumps(rec, indent=2))
    print(f"[miss] {len(report['cells'])} cells, {recon['graded_rows']} graded rows reconciled, "
          f"pool records checked {sum(A['pool_records_checked'] for A in arms.values())}; "
          f"{rec['timing_seconds']}s, peak {rec['peak_rss_mib']}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except MissReportError as e:
        print(f"[miss] FAIL-CLOSED: {e}", file=sys.stderr, flush=True)
        sys.exit(2)
