#!/usr/bin/env python3
"""
scripts/diagnostics/e2e_step5_ranker.py -- DEVELOPMENT / NON-CERTIFYING. A Step 5 ARM that trains a
RANKER instead of a regressor (TASK_LOOP_ADDENDUM_2, owner yes 2026-11-23). Default pipeline is
unchanged: this script runs only when the walk-forward driver is given --ranker-arms.

OBJECTIVES (libraries already installed; no new dependency):
  xgb_rank_ndcg      XGBoost  objective rank:ndcg       -> best_model.json  (model_type xgboost)
  xgb_rank_pairwise  XGBoost  objective rank:pairwise   -> best_model.json  (model_type xgboost)
  lgbm_lambdarank    LightGBM objective lambdarank      -> best_model.txt   (model_type lightgbm)
  catboost_yetirank  CatBoost loss YetiRank             -> best_model.cbm   (model_type catboost)
  The checkpoint is written in the format Step 6's own loaders read (models/model_factory.py:30-32;
  xgboost_wrapper.load = Booster().load_model, lightgbm_wrapper.load = Booster(model_file),
  catboost_wrapper.load = CatBoostRegressor().load_model), so the vote harness votes with it unchanged.

GROUP CONSTRUCTION (exact):
  label draws  y_g for g = 0..L-1 = the run copy's label_block_history.json, combined index T + g
               (T = --label-start = len(train history)). Nothing at or after T + L is read.
  items        every survivor row r of the Step 3 survivors file (N rows), in file order.
  vote         v[r, g] = seq_r[T + g], seq_r = SurvivorScorer(prng_type, mod)._generate_sequence(
               seed_r, T + L, skip=PredictionGenerator._get_survivor_skip(row_r, default_skip)):
               the same calls Step 6 makes for its vote at next_idx = t (prediction_generator.py:
               814, :865-868; scorer as built at :267-270). Each sequence is generated ONCE up to T+L
               (the MISS agent's matrix method, e2e_miss_report.py:443-476), not once per draw.
  relevance    rel[r, g] = 1 if v[r, g] == y_g else 0  (that survivor voted for the drawn number).
  group g      the rows {(r, g)}: features X[r] (the Step 5 sidecar's feature names, in its order),
               label rel[r, g]. Features are per survivor, so a survivor carries the same X in
               every group; only the relevance changes.
  size         all groups and all items = N * L rows (run H: 12,156 x 1,807 = 21,965,892 rows,
               x 89 float32 = 7.8 GB). Bounded by two settings: --groups G (label draws sampled,
               seeded, 0 = all) and --negatives-per-group M (every positive kept, M negatives
               sampled per group, seeded, 0 = all). Rows = sum_g (pos_g + min(M, N - pos_g)).
               --drop-empty-groups (default on) drops groups with no positive (no gradient).
  score        NDCG@--ndcg-k on --valid-fraction of the sampled groups (seeded split, all items
               of those groups kept), reported; a TRAINING score, not a hit rate.

OUTPUTS (--out-dir, must not exist): best_model.<ext>, best_model.meta.json (the template Step 5
sidecar of the same origin with model_type, checkpoint_path/format replaced, `optuna` removed and a
`ranking_objective` block added), ranker_record.json (inputs with sha256, sizes, timings).
"""
from __future__ import annotations

import sys as _sys
_sys.dont_write_bytecode = True

import argparse
import copy
import datetime as _dt
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]
LABEL = "DEVELOPMENT"
OBJECTIVES = {"xgb_rank_ndcg": ("xgboost", ".json"), "xgb_rank_pairwise": ("xgboost", ".json"),
              "lgbm_lambdarank": ("lightgbm", ".txt"), "catboost_yetirank": ("catboost", ".cbm")}
MUTANT_ENV = "E2E_RANKER_MUTANT"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


_W = {}


def _votes(chunk):
    sc, seeds, skips, T, L = _W["sc"], _W["seeds"], _W["skips"], _W["T"], _W["L"]
    out = np.empty((len(chunk), L), dtype=np.int16)
    for j, r in enumerate(chunk):
        seq = sc._generate_sequence(int(seeds[r]), T + L, skip=int(skips[r]))
        if len(seq) < T + L:
            raise RuntimeError(f"sequence for row {r} shorter than {T + L}")
        out[j] = seq[T:T + L]
    return chunk[0], out


def vote_matrix(sc, seeds, skips, T, L, workers):
    import multiprocessing as mp
    n = len(seeds)
    V = np.empty((n, L), dtype=np.int16)
    _W.update(sc=sc, seeds=seeds, skips=skips, T=T, L=L)
    idx = list(range(n))
    size = max(1, -(-n // (max(1, workers) * 8)))
    chunks = [idx[i:i + size] for i in range(0, n, size)]
    if workers <= 1:
        for start, blk in map(_votes, chunks):
            V[start:start + len(blk)] = blk
    else:
        with mp.get_context("fork").Pool(workers) as p:
            for start, blk in p.imap_unordered(_votes, chunks):
                V[start:start + len(blk)] = blk
    return V


def ndcg_at(scores, rel, k):
    o = np.argsort(-scores, kind="stable")[:k]
    gains = rel[o] / np.log2(np.arange(2, len(o) + 2))
    ideal = np.sort(rel)[::-1][:k] / np.log2(np.arange(2, min(k, len(rel)) + 2))
    return float(gains.sum() / ideal.sum()) if ideal.sum() > 0 else None


def main(argv=None):
    ap = argparse.ArgumentParser(description="DEVELOPMENT Step 5 ranker arm (TASK_LOOP_ADDENDUM_2)")
    ap.add_argument("--survivors", required=True)
    ap.add_argument("--label-history", required=True, help="the run copy's label_block_history.json")
    ap.add_argument("--label-start", type=int, required=True, help="T = combined index of the first label draw")
    ap.add_argument("--expect-label-sha256", required=True)
    ap.add_argument("--sidecar-template", required=True, help="a Step 5 best_model.meta.json of this origin")
    ap.add_argument("--pg-config", required=True, help="prediction_generator_config.json (prng_type, mod, default_skip)")
    ap.add_argument("--objective", required=True, choices=sorted(OBJECTIVES))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--groups", type=int, default=0, help="label draws sampled as groups (0 = all)")
    ap.add_argument("--negatives-per-group", type=int, default=0, help="0 = all items")
    ap.add_argument("--drop-empty-groups", choices=["on", "off"], default="on")
    ap.add_argument("--valid-fraction", type=float, default=0.2)
    ap.add_argument("--ndcg-k", type=int, default=20)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=0.1)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-rows", type=int, default=5_000_000, help="refuse above this many training rows")
    ap.add_argument("--save-votes", choices=["off", "on"], default="off",
                    help="on: write votes.npy (N x L int16) for the self-test's equality gate")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    out = Path(a.out_dir)
    if out.exists():
        sys.exit(f"[{LABEL}] refuse: {out} exists")
    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(WT))
    mut = os.environ.get(MUTANT_ENV)

    lab_sha = sha256_file(a.label_history)
    if lab_sha != a.expect_label_sha256:
        sys.exit(f"[{LABEL}] refuse: label history sha256 {lab_sha} != expected")
    y = np.asarray(json.load(open(a.label_history)), dtype=np.int64)
    L, T = len(y), a.label_start
    tpl = json.load(open(a.sidecar_template))
    names = tpl["feature_schema"]["feature_names"]
    rows = json.load(open(a.survivors))
    N = len(rows)

    # features: the sidecar's names in its order, through Step 5's own loader
    import meta_prediction_optimizer_anti_overfit as mpo
    X_all, y_unused, fs, _ = mpo.load_survivors_with_features(
        a.survivors, target_field="holdout_quality",
        exclude_features=tpl["feature_schema"].get("excluded_features") or
        ["score", "confidence", "holdout_hits", "holdout_quality"])
    miss = [n for n in names if n not in fs["feature_names"]]
    if miss:
        sys.exit(f"[{LABEL}] refuse: template features missing from the loader: {miss[:5]}")
    X = X_all[:, [fs["feature_names"].index(n) for n in names]].astype(np.float32)
    if X.shape[0] != N:
        sys.exit(f"[{LABEL}] refuse: loader rows {X.shape[0]} != file rows {N}")

    # votes: Step 6's own skip rule and sequence generator
    import prediction_generator as pg
    from survivor_scorer import SurvivorScorer
    cfg = json.load(open(a.pg_config))
    pcfg = pg.PredictionConfig.from_json(a.pg_config) if hasattr(pg.PredictionConfig, "from_json") else None
    prng_type = getattr(pcfg, "prng_type", cfg.get("prng_type", "java_lcg"))
    mod = int(getattr(pcfg, "mod", cfg.get("mod", 1000)))
    default_skip = int(getattr(pcfg, "default_skip", cfg.get("default_skip", 0)))
    sc = SurvivorScorer(prng_type=prng_type, mod=mod)
    seeds = [int(r["seed"]) for r in rows]
    skips = [pg.PredictionGenerator._get_survivor_skip(None, r, default_skip) for r in rows]
    tv = time.time()
    V = vote_matrix(sc, seeds, skips, T, L, a.workers)
    vote_s = round(time.time() - tv, 2)
    if mut == "relevance_shift":              # self-test mutant: votes one position late
        V = np.roll(V, -1, axis=1)
    rel = (V == y[None, :]).astype(np.int32)  # (N, L)
    pos_per_group = rel.sum(axis=0)

    # groups and rows
    rng = np.random.default_rng(a.seed)
    gidx = np.arange(L)
    if a.drop_empty_groups == "on":
        gidx = gidx[pos_per_group[gidx] > 0]
    n_empty = int((pos_per_group == 0).sum())
    if a.groups and a.groups < len(gidx):
        gidx = np.sort(rng.choice(gidx, size=a.groups, replace=False))
    if len(gidx) < 2:
        sys.exit(f"[{LABEL}] refuse: {len(gidx)} usable groups (positives per group {pos_per_group.tolist()[:10]}...)")
    perm = rng.permutation(len(gidx))
    n_val = max(1, int(round(a.valid_fraction * len(gidx)))) if a.valid_fraction > 0 else 0
    val_g = np.sort(gidx[perm[:n_val]])
    tr_g = np.sort(gidx[perm[n_val:]])
    Xr, yr, sizes, item_rows = [], [], [], 0
    for g in tr_g:
        r_pos = np.flatnonzero(rel[:, g])
        r_neg = np.flatnonzero(rel[:, g] == 0)
        if a.negatives_per_group and a.negatives_per_group < len(r_neg):
            r_neg = np.sort(rng.choice(r_neg, size=a.negatives_per_group, replace=False))
        items = np.concatenate([r_pos, r_neg])
        item_rows += len(items)
        if item_rows > a.max_rows:
            sys.exit(f"[{LABEL}] refuse: more than --max-rows {a.max_rows} training rows; "
                     f"lower --groups or --negatives-per-group")
        Xr.append(items)
        yr.append(rel[items, g])
        sizes.append(len(items))
    rows_idx = np.concatenate(Xr)
    Xt, yt = X[rows_idx], np.concatenate(yr)
    gid = np.repeat(np.arange(len(sizes)), sizes)

    family, ext = OBJECTIVES[a.objective]
    out.mkdir(parents=True)
    ck = out / f"best_model{ext}"
    tt = time.time()
    if family == "xgboost":
        import xgboost as xgb
        d = xgb.DMatrix(Xt, label=yt)
        d.set_group(sizes)
        params = {"objective": "rank:ndcg" if a.objective == "xgb_rank_ndcg" else "rank:pairwise",
                  "eta": a.learning_rate, "max_depth": a.max_depth, "seed": a.seed,
                  "tree_method": "hist", "device": "cuda" if a.device == "cuda" else "cpu",
                  "eval_metric": f"ndcg@{a.ndcg_k}"}
        booster = xgb.train(params, d, num_boost_round=a.n_estimators)
        booster.save_model(str(ck))
        predict = lambda Z: booster.predict(xgb.DMatrix(Z))
    elif family == "lightgbm":
        import lightgbm as lgb
        params = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [a.ndcg_k],
                  "learning_rate": a.learning_rate, "max_depth": a.max_depth,
                  "num_leaves": min(2 ** a.max_depth, 131072) if a.max_depth > 0 else 31,
                  "seed": a.seed, "deterministic": True, "force_row_wise": True, "verbose": -1,
                  "device_type": "gpu" if a.device == "cuda" else "cpu"}
        booster = lgb.train(params, lgb.Dataset(Xt, label=yt, group=sizes), num_boost_round=a.n_estimators)
        booster.save_model(str(ck))
        predict = booster.predict
    else:
        from catboost import CatBoost, Pool
        params = {"loss_function": "YetiRank", "iterations": a.n_estimators, "depth": a.max_depth,
                  "learning_rate": a.learning_rate, "random_seed": a.seed, "verbose": False,
                  "task_type": "GPU" if a.device == "cuda" else "CPU", "allow_writing_files": False}
        if a.device == "cuda":
            params["devices"] = "0"
        m = CatBoost(params)
        m.fit(Pool(Xt, label=yt, group_id=gid))
        m.save_model(str(ck))
        predict = m.predict
    train_s = round(time.time() - tt, 2)

    s_all = np.asarray(predict(X), dtype=np.float64)       # one score per survivor
    nd_val = [ndcg_at(s_all, rel[:, g].astype(float), a.ndcg_k) for g in val_g]
    nd_tr = [ndcg_at(s_all, rel[:, g].astype(float), a.ndcg_k) for g in tr_g[:200]]
    nd_val = [v for v in nd_val if v is not None]
    nd_tr = [v for v in nd_tr if v is not None]

    side = copy.deepcopy(tpl)
    side.pop("optuna", None)
    side["model_type"] = family
    side["checkpoint_path"] = str(ck.resolve())
    side["checkpoint_format"] = ext.lstrip(".")
    side["training_metrics"] = {"valid_ndcg_at_k": float(np.mean(nd_val)) if nd_val else None,
                                "train_ndcg_at_k_first200": float(np.mean(nd_tr)) if nd_tr else None,
                                "k": a.ndcg_k, "note": "training scores, not hit rates"}
    side["hyperparameters"] = {"objective": a.objective, "n_estimators": a.n_estimators,
                               "max_depth": a.max_depth, "learning_rate": a.learning_rate, "seed": a.seed}
    side["ranking_objective"] = {
        "label": LABEL, "objective": a.objective, "family": family,
        "relevance": "rel[r,g] = 1 if survivor r's Step 6 vote at combined index T+g equals the drawn number",
        "label_start_T": T, "label_draws_L": L, "label_sha256": lab_sha,
        "groups_total": L, "groups_empty": n_empty, "groups_used_train": int(len(tr_g)),
        "groups_used_valid": int(len(val_g)), "negatives_per_group": a.negatives_per_group,
        "items_per_group_N": N, "train_rows": int(len(yt)), "train_positives": int(yt.sum()),
        "positives_per_group_mean": float(pos_per_group.mean()),
        "sidecar_template": {"path": str(Path(a.sidecar_template).resolve()),
                             "sha256": sha256_file(a.sidecar_template)},
        "signal_quality_note": "signal_quality is the template's (the regressor's target on the same label block)"}
    (out / "best_model.meta.json").write_text(json.dumps(side, indent=2))
    if a.save_votes == "on":
        np.save(out / "votes.npy", V)
    rec = {"label": LABEL, "kind": "e2e_step5_ranker", "argv": sys.argv, "args": vars(a),
           "inputs": {"survivors": {"path": str(Path(a.survivors).resolve()), "sha256": sha256_file(a.survivors)},
                      "label_history": {"path": str(Path(a.label_history).resolve()), "sha256": lab_sha},
                      "pg_config": {"path": str(Path(a.pg_config).resolve()), "sha256": sha256_file(a.pg_config)}},
           "vote": {"prng_type": prng_type, "mod": mod, "default_skip": default_skip,
                    "skips_distinct": sorted(set(skips)), "seconds": vote_s},
           "positives_per_group": pos_per_group.tolist(),
           "ranking_objective": side["ranking_objective"], "training_metrics": side["training_metrics"],
           "train_seconds": train_s, "wall_seconds": round(time.time() - t0, 2),
           "checkpoint": {"path": str(ck), "sha256": sha256_file(ck)},
           "code": {"self": sha256_file(__file__), "prediction_generator": sha256_file(WT / "prediction_generator.py"),
                    "survivor_scorer": sha256_file(WT / "survivor_scorer.py")},
           "mutant": mut, "finished": _dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    (out / "ranker_record.json").write_text(json.dumps(rec, indent=1))
    print(f"[{LABEL}] ranker {a.objective}: {len(tr_g)} train groups, {len(yt)} rows, "
          f"{int(yt.sum())} positives; valid NDCG@{a.ndcg_k} {side['training_metrics']['valid_ndcg_at_k']}; "
          f"vote {vote_s} s, train {train_s} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
