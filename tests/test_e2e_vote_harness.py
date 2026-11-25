#!/usr/bin/env python3
"""
Self-test for scripts/diagnostics/e2e_vote_harness.py (H6). CPU only, small fixtures only.

Run:  python3 -u tests/test_e2e_vote_harness.py [--out-root DIR] [--scale-probe] | tee <log>

The fixture is built inside the test, under --out-root (default data/e2e_vote_selftest/<stamp>):
  - Step 3 rows from the real worker: full_scoring_worker.score_survivors on CPU over 40
    seeds, with a synthetic 240-draw train history and a 60-draw label holdout. One row is
    duplicated, as the aggregator can emit. The rows are sorted by score, as the .sh does
    (run_step3_full_scoring.sh:398).
  - X and y from Step 5's loader (meta_prediction_optimizer_anti_overfit.load_survivors_with_features)
    and signal_quality from Step 5's compute_signal_quality.
  - LightGBM and XGBoost checkpoints fitted on CPU through the project wrappers, with a
    sidecar in Step 5's 3.4.0 shape (meta_prediction_optimizer_anti_overfit.py:1288-1320).
    FIXTURE-MADE, not Step-5-made.
No run H data is read. The oracle for every pool is the real Step 6 CLI
(prediction_generator.py main), run as a subprocess with CUDA hidden.

G19 (lead TASK_LAND_WF_VOTE_20261125_ADDENDUM_2): the LL-1 capability check. A config with
schema_check set refuses (exit-2 HarnessError, no out-dir, switch and file named) on
prediction_generator files that lack LL-1 (this repo's HEAD copy, and the harness's own repo when
it lacks it); non-vacuity: the same request passes on files that have it (env
E2E_VOTE_CAPS_WITH_ROOT, default ~/tfm_worktrees/e2e-map). Mutant M25 disables the check.

Gates (G*) must pass. Mutants (M*) must each be killed: the mutant was applied, the mutated
path ran (hit counter > 0), and the credited gate's assertion fired, not some other error.
"""

import argparse
import copy
import importlib.util
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

GPU_MODE = "--gpu" in sys.argv   # owner go 2026-11-23 08:29: synthetic fixtures only
if not GPU_MODE:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU mode: no GPU anywhere in this test

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
HARNESS_PATH = REPO / "scripts" / "diagnostics" / "e2e_vote_harness.py"

spec = importlib.util.spec_from_file_location("e2e_vote_harness", HARNESS_PATH)
H = importlib.util.module_from_spec(spec)
sys.modules["e2e_vote_harness"] = H  # forked Pool workers pickle _worker_chunk by module name
spec.loader.exec_module(H)


LABEL_LEN = 30          # fixture label block: combined [240, 270)
UNSEEN_FROM = 270       # first combined index nothing upstream saw
PENDING_DATE, PENDING_SESSION = "2020-05-30", "midday"   # declared for t = 300 (pending)
POOL_FIELDS = {"schema", "arm_id", "config_id", "dataset_sha256", "data_cutoff_index",
               "target", "pool", "pool_sha256", "abstain", "abstain_reason"}
GRADER_PATH = REPO / "scripts" / "diagnostics" / "e2e_grader.py"   # this repo's copy (was the e2e-grader worktree)


class GateFail(AssertionError):
    def __init__(self, tag, msg):
        super().__init__(f"[{tag}] {msg}")
        self.tag = tag


def check(tag, cond, msg):
    if not cond:
        raise GateFail(tag, msg)


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------
def build_fixture(root):
    import numpy as np
    import full_scoring_worker as fw
    from survivor_scorer import SurvivorScorer
    import meta_prediction_optimizer_anti_overfit as mpo
    from models.wrappers.lightgbm_wrapper import LightGBMWrapper
    from models.wrappers.xgboost_wrapper import XGBoostWrapper

    fx = root / "fixture"
    fx.mkdir(parents=True)
    rs = np.random.RandomState(20261123)
    train = [int(x) for x in rs.randint(0, 1000, 240)]
    hold = [int(x) for x in rs.randint(0, 1000, 60)]
    hold[5] = 0  # a draw of 000 inside the history
    combined = train + hold
    seeds = sorted(set(int(x) for x in rs.randint(0, 2 ** 31, 60)))[:40]
    rs.shuffle(seeds)
    # Step 3 labels from a label block only (map Q1 route): combined [240, 270).
    # Combined [270, 300) is seen by nothing upstream; G12 grades it.
    rows = fw.score_survivors(seeds, train, prng_type="java_lcg", mod=1000,
                              scorer_class=SurvivorScorer, holdout_history=hold[:LABEL_LEN])
    rows.append(copy.deepcopy(rows[3]))  # duplicate row: same seed, same features
    rows.sort(key=lambda x: x.get("score", 0), reverse=True)  # .sh aggregation order
    sv = fx / "survivors_with_scores.json"
    sv.write_text(json.dumps(rows, indent=2))

    (fx / "train_history.json").write_text(json.dumps(train))
    (fx / "holdout_history.json").write_text(json.dumps(hold))
    d0 = date(2020, 1, 1)
    ident = []
    for i, v in enumerate(combined):
        ident.append({"date": (d0 + timedelta(days=i // 2)).isoformat(),
                      "session": "midday" if i % 2 == 0 else "evening", "draw": v})
    (fx / "dataset.json").write_text(json.dumps(ident))
    (fx / "combined_history.json").write_text(json.dumps(combined))
    # the upstream config the CLI and the harness both inherit prng_type from (:149-160)
    (fx / "optimal_window_config.json").write_text(json.dumps({"prng_type": "java_lcg"}))
    cfg = {"prediction": {"pool_size": 20, "k": 20, "use_dual_sieve": True,
                          "prng": {"prng_type": "java_lcg", "mod": 1000, "skip": 0},
                          "output": {"save_predictions": True, "log_level": "WARNING"}}}
    (fx / "prediction_config.json").write_text(json.dumps(cfg, indent=2))

    X, y, schema, ymeta = mpo.load_survivors_with_features(str(sv))
    sq = mpo.compute_signal_quality(y, target_name="holdout_quality")
    check("FIX", sq["prediction_allowed"], f"fixture target degenerate: {sq}")

    def write_sidecar(mdir, model_type, ckpt):
        sc = {"schema_version": "3.4.0", "model_type": model_type,
              "checkpoint_path": str(ckpt), "checkpoint_format": ckpt.suffix.lstrip("."),
              "feature_schema": schema, "signal_quality": sq,
              "data_context": {"fingerprint_hash": "fixture"},
              "training_metrics": {}, "hyperparameters": {},
              "agent_metadata": {"pipeline_step": 5, "run_id": f"fixture_{model_type}"},
              "provenance": {"note": "FIXTURE-MADE by tests/test_e2e_vote_harness.py"}}
        (mdir / "best_model.meta.json").write_text(json.dumps(sc, indent=2))

    models = {}
    lg_dir = fx / "models_lightgbm"; lg_dir.mkdir()
    lg = LightGBMWrapper(config={"n_estimators": 30, "num_leaves": 7, "force_cpu": True}, device="cpu")
    lg.fit(X, y)
    lg.save(str(lg_dir / "best_model.txt"))
    write_sidecar(lg_dir, "lightgbm", lg_dir / "best_model.txt")
    models["lightgbm"] = lg_dir

    xg_dir = fx / "models_xgboost"; xg_dir.mkdir()
    xg = XGBoostWrapper(config={"n_estimators": 30, "max_depth": 3}, device="cpu")
    xg.fit(X, y)
    xg.save(str(xg_dir / "best_model.json"))
    write_sidecar(xg_dir, "xgboost", xg_dir / "best_model.json")
    models["xgboost"] = xg_dir

    blk = fx / "models_gate_blocked"; blk.mkdir()
    sc = json.loads((lg_dir / "best_model.meta.json").read_text())
    sc["checkpoint_path"] = None
    sc["signal_quality"] = dict(sq, prediction_allowed=False, signal_status="degenerate")
    (blk / "best_model.meta.json").write_text(json.dumps(sc, indent=2))
    models["gate_blocked"] = blk

    miss = fx / "models_missing_ckpt"; miss.mkdir()
    sc = json.loads((lg_dir / "best_model.meta.json").read_text())
    sc["checkpoint_path"] = str(miss / "best_model.txt")  # does not exist
    (miss / "best_model.meta.json").write_text(json.dumps(sc, indent=2))
    models["missing_ckpt"] = miss

    # neural_net sidecars from Step 5's own NN trainer (train_single_trial.train_neural_net)
    import torch
    import train_single_trial as tst
    torch.manual_seed(20261123)
    nn_params = {"hidden_layers": [16, 8], "dropout": 0.1, "epochs": 20,
                 "learning_rate": 0.01, "batch_size": 16}
    Xtr, ytr, Xva, yva = X[:30], y[:30], X[30:], y[30:]
    for name, norm in (("nn", True), ("nn_noscaler", False)):
        d = fx / f"models_{name}"; d.mkdir()
        tst.train_neural_net(Xtr, ytr, Xva, yva, dict(nn_params), save_path=str(d / "best_model.pth"),
                             normalize_features=norm)
        write_sidecar(d, "neural_net", d / "best_model.pth")
        models[name] = d
    d = fx / "models_nn_noy"; d.mkdir()
    ck = torch.load(models["nn"] / "best_model.pth", map_location="cpu", weights_only=False)
    ck.pop("y_mean"); ck.pop("y_std")
    torch.save(ck, d / "best_model.pth")
    write_sidecar(d, "neural_net", d / "best_model.pth")
    models["nn_noy"] = d

    # Step 1 style NPZ (v1 arrays) for the field_sum key: distinct sums, no ties
    fm = np.array([0.30 + 0.01 * i for i in range(len(seeds))], dtype=np.float32)
    rm = np.array([0.30 + 0.001 * ((i * 7) % len(seeds)) for i in range(len(seeds))], dtype=np.float32)
    np.savez(fx / "step1_survivors.npz", seeds=np.array(seeds, dtype=np.uint32),
             forward_matches=fm, reverse_matches=rm)

    # survivors with a top-level skip (the CLI oracle for the per_skip arm)
    rows_skip3 = [dict(r, skip=3) for r in rows]
    (fx / "survivors_skip3.json").write_text(json.dumps(rows_skip3))
    return {"fx": fx, "rows": rows, "combined": combined, "train": train, "hold": hold,
            "models": models, "n_rows": len(rows), "feature_count": len(schema["feature_names"])}


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------
CLI_RUNS = {"n": 0, "seconds": 0.0}


def run_cli(F, survivors, history_file, models_dir, k=None, pool_size=None, tag="cli",
            no_require_schema=False):
    fx = F["fx"]
    pred = fx / "predictions" / "next_draw_prediction.json"
    if pred.exists():
        pred.unlink()
    cmd = [sys.executable, str(REPO / "prediction_generator.py"),
           "--config", str(fx / "prediction_config.json"),
           "--survivors-forward", str(survivors), "--lottery-history", str(history_file),
           "--models-dir", str(models_dir), "--log-level", "WARNING"]
    if k is not None:
        cmd += ["--k", str(k)]
    if pool_size is not None:
        cmd += ["--pool-size", str(pool_size)]
    if no_require_schema:
        cmd += ["--no-require-schema"]
    env = dict(os.environ) if GPU_MODE else dict(os.environ, CUDA_VISIBLE_DEVICES="")
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(fx), env=env, capture_output=True, text=True, timeout=300)
    CLI_RUNS["n"] += 1
    CLI_RUNS["seconds"] += time.time() - t0
    if p.returncode != 0:
        raise RuntimeError(f"{tag}: CLI exit {p.returncode}\n{p.stderr[-2000:]}")
    return json.loads(pred.read_text())


RUN_N = {"n": 0}


def run_harness(F, out_root, extra, history=None, survivors=None, expect_fail=None,
                dataset=None, auto_pending=True):
    """Run H.main in-process with cwd = fixture dir (the CLI's cwd). Returns (record, lines).
    `lines` merges each tfm.e2e.pool.v1 record ("rec") with its diagnostics line ("diag")."""
    RUN_N["n"] += 1
    out = out_root / f"h{RUN_N['n']:03d}"
    fx = F["fx"]
    argv = ["--survivors", str(survivors or fx / "survivors_with_scores.json"),
            "--dataset", str(dataset or fx / "dataset.json"),
            "--history"] + [str(h) for h in (history or [fx / "train_history.json",
                                                          fx / "holdout_history.json"])] + \
           ["--config", str(fx / "prediction_config.json"), "--out-dir", str(out)] + extra
    if GPU_MODE:
        argv += ["--allow-gpu"]
    if "--data-cutoff-index" not in extra:
        argv += ["--data-cutoff-index", "t"]
    if auto_pending and "--block-end" in extra and "--pending-date" not in extra \
            and int(extra[extra.index("--block-end") + 1]) > len(F["combined"]):
        argv += ["--pending-date", PENDING_DATE, "--pending-session", PENDING_SESSION]
    cwd = os.getcwd()
    os.chdir(fx)
    try:
        if expect_fail is not None:
            try:
                H.main(argv)
            except expect_fail as e:
                return None, str(e)
            raise GateFail("FAILCLOSED", f"expected {expect_fail.__name__}, run succeeded: {argv}")
        rc = H.main(argv)
    finally:
        os.chdir(cwd)
    assert rc == 0
    rec = json.loads((out / "run_record.json").read_text())
    pools = [json.loads(x) for x in (out / "pools.jsonl").read_text().splitlines()]
    diags = [json.loads(x) for x in (out / "vote_diagnostics.jsonl").read_text().splitlines()]
    check("FMT", len(pools) == len(diags), "pools and diagnostics differ in length")
    lines = []
    for r, d in zip(pools, diags):
        check("FMT", set(r) <= POOL_FIELDS and POOL_FIELDS - {"abstain_reason"} <= set(r),
              f"pool record fields {sorted(r)} are not the grader schema")
        check("FMT", r["schema"] == "tfm.e2e.pool.v1", "schema id")
        check("FMT", r["target"]["combined_index"] == d["combined_index"], "pool/diag misaligned")
        lines.append({"t": d["combined_index"], "pool": r["pool"], "pool_sha256": r["pool_sha256"],
                      "raw_scores": d["raw_scores"], "pending": d["pending"],
                      "status": d["vote_status"], "abstain_reason": r.get("abstain_reason"),
                      "draw_date": r["target"]["date"], "draw_session": r["target"]["session"],
                      "vote_index": d["vote_index"], "rec": r, "diag": d})
    return rec, lines


def jbytes(x):
    return json.dumps(x, separators=(",", ":")).encode()


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def g1_byte_identity(F, out_root):
    """TODO-E2E-8: at t = len(history) the harness pool equals the CLI pool byte for byte."""
    fx, n = F["fx"], len(F["combined"])
    cases = [("lightgbm", None, None), ("lightgbm", 10, None), ("lightgbm", 25, 30),
             ("xgboost", None, None)]
    detail = []
    for model, k, ps in cases:
        cli = run_cli(F, fx / "survivors_with_scores.json", fx / "dataset.json",
                      F["models"][model], k=k, pool_size=ps, tag=f"G1 {model}")
        extra = ["--models-dir", str(F["models"][model]), "--block-start", str(n),
                 "--block-end", str(n + 1)]
        if k is not None:
            extra += ["--k", str(k)]
        if ps is not None:
            extra += ["--pool-size", str(ps)]
        rec, lines = run_harness(F, out_root, extra)
        check("G1", len(lines) == 1, "one line expected")
        ln = lines[0]
        check("G1", ln["t"] == n and ln["vote_index"] == n and ln["pending"], f"line identity {ln}")
        check("G1", jbytes(ln["pool"]) == jbytes(cli["predictions"]),
              f"{model} k={k} ps={ps}: pool {ln['pool']} != CLI {cli['predictions']}")
        check("G1", jbytes(ln["raw_scores"]) == jbytes(cli["raw_scores"]),
              f"{model}: raw_scores differ")
        check("G1", ln["pool_sha256"] == H.pool_sha256(cli["predictions"]), "pool sha differs")
        check("G1", (ln["draw_date"], ln["draw_session"]) == (PENDING_DATE, PENDING_SESSION),
              "pending target must carry the declared date/session")
        detail.append(f"{model} k={k or 'cfg'} ps={ps or 'cfg'}: {len(ln['pool'])} ints, "
                      f"{len(set(ln['pool']))} distinct, sha {ln['pool_sha256'][:12]}")
    return detail


def g2_walk_forward(F, out_root):
    """For t inside the history, the harness line at t equals the CLI run on history[:t]."""
    fx = F["fx"]
    rec, lines = run_harness(F, out_root, [
        "--models-dir", str(F["models"]["lightgbm"]), "--block-start", "250", "--block-end", "300"])
    check("G2", [l["t"] for l in lines] == list(range(250, 300)), "one line per t, in order")
    by_t = {l["t"]: l for l in lines}
    for t in (250, 275, 299):
        hf = fx / f"hist_upto_{t}.json"
        hf.write_text(json.dumps(F["combined"][:t]))
        cli = run_cli(F, fx / "survivors_with_scores.json", hf, F["models"]["lightgbm"], tag=f"G2 t={t}")
        check("G2", jbytes(by_t[t]["pool"]) == jbytes(cli["predictions"]),
              f"t={t}: harness {by_t[t]['pool']} != CLI {cli['predictions']}")
    ds = json.loads((fx / "dataset.json").read_text())
    check("G2", by_t[255]["draw_date"] == ds[255]["date"] and
          by_t[255]["draw_session"] == ds[255]["session"], "draw identity misaligned")
    check("G2", not any(l["pending"] for l in lines), "no pending lines inside the history")
    check("G2", all(not ({"draw", "hit", "outcome"} & (set(l["rec"]) | set(l["diag"])))
                    for l in lines), "records must not carry the outcome")
    check("G2", all(l["rec"]["data_cutoff_index"] == l["t"] for l in lines), "cutoff 't' = target")
    return rec, lines


def g3_rank_once(rec, lines):
    rp = rec["ranking_proof"]
    check("G3", rp["model_calls"] == 1, f"model called {rp['model_calls']} times")
    check("G3", rp["draws_reusing_ranking"] == len(lines) - 1, f"reuse count {rp}")
    check("G3", rp["x_shape"][0] == rec["inputs"]["survivors"]["rows"], "X rows != survivor rows")
    return rp


def g4_workers(F, out_root, rec1, lines1):
    rec4, _ = run_harness(F, out_root, [
        "--models-dir", str(F["models"]["lightgbm"]), "--block-start", "250", "--block-end", "300",
        "--workers", "4"])
    check("G4", rec4["pools_jsonl"]["sha256"] == rec1["pools_jsonl"]["sha256"],
          "workers=4 output differs from workers=1")
    check("G4", rec4["config_id"] == rec1["config_id"], "config_id depends on workers")
    check("G4", rec4["vote_diagnostics_jsonl"]["sha256"] == rec1["vote_diagnostics_jsonl"]["sha256"],
          "workers=4 diagnostics differ")
    check("G4", rec4["ranking_proof"]["draws_reusing_ranking"] == 49, "fork reuse count")
    return rec4


def g5_abstain(F, out_root):
    rec, lines = run_harness(F, out_root, [
        "--models-dir", str(F["models"]["gate_blocked"]), "--block-start", "280", "--block-end", "301"])
    check("G5", len(lines) == 21, "every draw in the block gets a line")
    check("G5", all(l["status"] == "abstain" and l["pool"] == [] and l["rec"]["abstain"] is True
                    for l in lines), "all abstain")
    check("G5", all(l["abstain_reason"] == "signal_gate:degenerate" for l in lines), "reason")
    check("G5", rec["counts"]["abstentions_total"] == 21 and rec["counts"]["pools"] == 0, "counts")
    check("G5", rec["counts"]["pending_lines"] == 1, "t=300 is pending")
    return rec["counts"]


def g6_fail_closed(F, out_root):
    fx = F["fx"]
    lg = ["--models-dir", str(F["models"]["lightgbm"])]
    out = []
    # a) out-dir exists
    exist = out_root / "exists"; exist.mkdir()
    cwd = os.getcwd(); os.chdir(fx)
    try:
        try:
            H.main(["--survivors", str(fx / "survivors_with_scores.json"), "--dataset",
                    str(fx / "dataset.json"), "--config", str(fx / "prediction_config.json"),
                    "--data-cutoff-index", "t",
                    "--out-dir", str(exist), "--block-start", "0", "--block-end", "1"] + lg)
            raise GateFail("G6", "existing out-dir accepted")
        except H.HarnessError as e:
            check("G6", "exists" in str(e), str(e)); out.append("out-dir exists: refused")
    finally:
        os.chdir(cwd)
    # b) a bare-int survivor row
    bad = fx / "survivors_bad.json"
    bad.write_text(json.dumps(F["rows"][:3] + [12345]))
    _, msg = run_harness(F, out_root, lg + ["--block-start", "299", "--block-end", "300"],
                         survivors=bad, expect_fail=H.HarnessError)
    check("G6", "expected dict" in msg, msg); out.append("int survivor row: refused")
    # c) missing checkpoint (Ch7 B2): harness refuses, with and without --no-require-schema
    for extra in ([], ["--no-require-schema"]):
        _, msg = run_harness(F, out_root, ["--models-dir", str(F["models"]["missing_ckpt"]),
                                           "--block-start", "299", "--block-end", "300"] + extra,
                             expect_fail=H.HarnessError)
        check("G6", "no model loaded" in msg, msg)
        out.append(f"missing checkpoint {extra or '(schema required)'}: refused")
    # d) block outside history+1
    _, msg = run_harness(F, out_root, lg + ["--block-start", "299", "--block-end", "302"],
                         expect_fail=H.HarnessError)
    check("G6", "outside" in msg, msg); out.append("block past len+1: refused")
    # e) midday hook arm
    _, msg = run_harness(F, out_root, lg + ["--block-start", "299", "--block-end", "300",
                                            "--vote-arm", "midday_stream_index"],
                         expect_fail=NotImplementedError)
    check("G6", "hook only" in msg, msg); out.append("midday_stream_index: NotImplementedError")
    # f) misaligned identity file
    ds = json.loads((fx / "dataset.json").read_text())
    ds[10]["draw"] = (ds[10]["draw"] + 1) % 1000
    badid = fx / "dataset_misaligned.json"; badid.write_text(json.dumps(ds))
    _, msg = run_harness(F, out_root, lg + ["--block-start", "299", "--block-end", "300"],
                         dataset=badid, expect_fail=H.HarnessError)
    check("G6", "not aligned" in msg, msg); out.append("dataset misaligned with history: refused")
    # f2) pending draw in the block without declared date/session
    _, msg = run_harness(F, out_root, lg + ["--block-start", "299", "--block-end", "301"],
                         auto_pending=False, expect_fail=H.HarnessError)
    check("G6", "--pending-date" in msg, msg); out.append("pending target undeclared: refused")
    # f3) declared cutoff after a target in the block
    _, msg = run_harness(F, out_root, lg + ["--block-start", "299", "--block-end", "300",
                                            "--data-cutoff-index", "300"],
                         expect_fail=H.HarnessError)
    check("G6", "cutoff <= every target" in msg, msg); out.append("cutoff > target: refused")
    # g) missing config
    cwd = os.getcwd(); os.chdir(fx)
    try:
        try:
            H.main(["--survivors", str(fx / "survivors_with_scores.json"), "--dataset",
                    str(fx / "dataset.json"), "--config", str(fx / "nope.json"),
                    "--data-cutoff-index", "t",
                    "--out-dir", str(out_root / "nocfg"), "--block-start", "0", "--block-end", "1"] + lg)
            raise GateFail("G6", "missing config accepted")
        except H.HarnessError as e:
            check("G6", "not found" in str(e), str(e)); out.append("missing config: refused")
    finally:
        os.chdir(cwd)
    # h) observation, not a gate: the CLI itself on the missing checkpoint
    for nrs in (False, True):
        try:
            cli = run_cli(F, fx / "survivors_with_scores.json", fx / "combined_history.json",
                          F["models"]["missing_ckpt"], tag="G6 obs", no_require_schema=nrs)
            out.append(f"OBSERVED: CLI, missing checkpoint, no_require_schema={nrs}: exit 0, "
                       f"status={cli.get('status')}, model_type="
                       f"{cli.get('metadata', {}).get('model_type')}, "
                       f"{len(cli.get('predictions', []))} predictions")
        except RuntimeError as e:
            last = [l for l in str(e).splitlines() if l.strip()][-1:]
            out.append(f"OBSERVED: CLI, missing checkpoint, no_require_schema={nrs}: "
                       f"{str(e).splitlines()[0]} :: {last}")
    return out


def g7_per_skip(F, out_root, base_lines):
    fx = F["fx"]
    lg = ["--models-dir", str(F["models"]["lightgbm"])]
    rec0, l0 = run_harness(F, out_root, lg + ["--block-start", "250", "--block-end", "300",
                                              "--vote-arm", "per_skip", "--arm-skip", "0"])
    check("G7", [l["pool"] for l in l0] == [l["pool"] for l in base_lines],
          "per_skip 0 must equal the baseline (Step 3 rows carry no skip, fallback 0)")
    n = len(F["combined"])
    rec3, l3 = run_harness(F, out_root, lg + ["--block-start", str(n), "--block-end", str(n + 1),
                                              "--vote-arm", "per_skip", "--arm-skip", "3"])
    cli = run_cli(F, fx / "survivors_skip3.json", fx / "combined_history.json",
                  F["models"]["lightgbm"], tag="G7 skip3")
    check("G7", jbytes(l3[0]["pool"]) == jbytes(cli["predictions"]),
          f"per_skip 3 {l3[0]['pool']} != CLI on rows with skip=3 {cli['predictions']}")
    base_n = json.loads((out_root / "h001" / "run_record.json").read_text())
    check("G7", rec3["ranking_proof"]["x_sha256"] == base_n["ranking_proof"]["x_sha256"],
          "skip override changed X")
    check("G7", jbytes(l3[0]["pool"]) != jbytes(
        json.loads((out_root / "h001" / "pools.jsonl").read_text().splitlines()[0])["pool"]),
          "skip 3 pool identical to skip 0 pool (the override had no effect)")
    return rec3["arm_info"]


def g8_unlearned(F, out_root):
    """Unlearned keys go through the module's own argsort and vote. Oracle: the module's
    SurvivorScorer on the expected order (distinct keys, so the order is defined)."""
    from survivor_scorer import SurvivorScorer
    sc = SurvivorScorer(prng_type="java_lcg", mod=1000)
    rows, fx = F["rows"], F["fx"]
    t0, t1 = 290, 300
    res = {}
    import numpy as np
    npz = np.load(fx / "step1_survivors.npz")
    fsum = {int(s): float(a) + float(b) for s, a, b in
            zip(npz["seeds"], npz["forward_matches"], npz["reverse_matches"])}
    orders = {
        "file_order": list(range(len(rows))),
        "seed_asc": sorted(range(len(rows)), key=lambda i: (int(rows[i]["seed"]), i)),
        "field_sum": sorted(range(len(rows)), key=lambda i: -fsum[int(rows[i]["seed"])]),
    }
    for ranking, order in orders.items():
        extra = ["--ranking", ranking, "--block-start", str(t0), "--block-end", str(t1)]
        if ranking == "field_sum":
            extra += ["--ranking-fields", "forward_matches,reverse_matches",
                      "--ranking-npz", str(fx / "step1_survivors.npz")]
        rec, lines = run_harness(F, out_root, extra)
        for ln in lines:
            t = ln["t"]
            exp = [int(sc._generate_sequence(int(rows[i]["seed"]), t + 1, skip=0)[t])
                   for i in order[:20]]
            if ranking in ("seed_asc", "field_sum"):
                # duplicated seed rows tie; both vote the same value, so compare values
                pass
            check("G8", ln["pool"] == exp, f"{ranking} t={t}: {ln['pool']} != {exp}")
        check("G8", rec["ranking_proof"]["model_calls"] == 1, "unlearned key computed once")
        res[ranking] = rec["ranking_info"].get("rule")
    # G8b: the field_sum fields are not in Step 3 rows; without the NPZ the run stops
    _, msg = run_harness(F, out_root, ["--ranking", "field_sum", "--ranking-fields",
                                       "forward_matches,reverse_matches", "--block-start", "299",
                                       "--block-end", "300"], expect_fail=H.HarnessError)
    check("G8", "missing for seed" in msg, msg)
    res["field_sum_without_npz"] = "refused: " + msg[:80]
    return res


def g9_repeats(F, out_root):
    """The raw pool keeps repeats as written. A file_order arm whose top rows include the
    duplicated seed must show that seed's vote twice."""
    rows = F["rows"]
    seeds = [int(r["seed"]) for r in rows]
    dup = [s for s in set(seeds) if seeds.count(s) > 1]
    check("G9", len(dup) == 1, f"fixture should hold exactly one duplicated seed: {dup}")
    first = seeds.index(dup[0])
    ps = max(20, seeds.index(dup[0], first + 1) + 1)
    rec, lines = run_harness(F, out_root, ["--ranking", "file_order", "--pool-size", str(ps),
                                           "--k", str(ps), "--block-start", "299", "--block-end", "300"])
    pool = lines[0]["pool"]
    check("G9", len(pool) == ps, f"pool len {len(pool)} != {ps}")
    check("G9", pool[first] == pool[seeds.index(dup[0], first + 1)], "duplicate vote not repeated")
    check("G9", len(set(pool)) < len(pool), "pool has no repeats; repeats were collapsed")
    return {"pool_len": len(pool), "distinct": len(set(pool))}


def g10_tie_break(F, out_root):
    """A heavily tied key: the declared secondary key orders the ties; the tie report sees them."""
    import numpy as np
    from survivor_scorer import SurvivorScorer
    sc = SurvivorScorer(prng_type="java_lcg", mod=1000)
    rows, fx = F["rows"], F["fx"]
    seeds = sorted(set(int(r["seed"]) for r in rows))
    fm = np.array([0.25 if (s % 3) else 0.5 for s in seeds], dtype=np.float32)
    rm = np.array([0.25 if (s % 5) else 0.5 for s in seeds], dtype=np.float32)
    np.savez(fx / "step1_ties.npz", seeds=np.array(seeds, dtype=np.uint32),
             forward_matches=fm, reverse_matches=rm)
    fsum = {s: float(a) + float(b) for s, a, b in zip(seeds, fm, rm)}
    base = ["--ranking", "field_sum", "--ranking-fields", "forward_matches,reverse_matches",
            "--ranking-npz", str(fx / "step1_ties.npz"), "--block-start", "295", "--block-end", "300"]
    rec, lines = run_harness(F, out_root, base + ["--tie-break", "seed_asc"])
    order = sorted(range(len(rows)), key=lambda i: (-fsum[int(rows[i]["seed"])], int(rows[i]["seed"])))
    for ln in lines:
        t = ln["t"]
        exp = [int(sc._generate_sequence(int(rows[i]["seed"]), t + 1, skip=0)[t]) for i in order[:20]]
        check("G10", ln["pool"] == exp, f"tie-break seed_asc t={t}: {ln['pool']} != {exp}")
    ties = rec["ranking_proof"]["ties"]
    check("G10", ties["boundary_tie"] and ties["pool_slots_set_by_tie_order"] > 0, f"ties {ties}")
    recm, linesm = run_harness(F, out_root, base)
    recm2, _ = run_harness(F, out_root, base)
    check("G10", recm["pools_jsonl"]["sha256"] == recm2["pools_jsonl"]["sha256"],
          "module tie order not reproducible run to run")
    check("G10", recm["config_id"] != rec["config_id"], "tie-break not in config identity")
    return {"ties": ties, "module_vs_seed_asc_same_pool":
            [a["pool"] for a in linesm] == [b["pool"] for b in lines]}


def g11_missing_feature(F, out_root):
    fx, n = F["fx"], len(F["combined"])
    rows = copy.deepcopy(F["rows"])
    sidecar = json.loads((F["models"]["lightgbm"] / "best_model.meta.json").read_text())
    victim = sidecar["feature_schema"]["feature_names"][0]
    del rows[2]["features"][victim]
    sv = fx / "survivors_missing_feature.json"
    sv.write_text(json.dumps(rows))
    lg = ["--models-dir", str(F["models"]["lightgbm"]), "--block-start", str(n),
          "--block-end", str(n + 1)]
    _, msg = run_harness(F, out_root, lg, survivors=sv, expect_fail=H.HarnessError)
    check("G11", "lack schema features" in msg, msg)
    rec, lines = run_harness(F, out_root, lg + ["--on-missing-feature", "record"], survivors=sv)
    check("G11", rec["ranking_info"]["missing_schema_features"]["rows_with_missing"] == 1, "count")
    cli = run_cli(F, sv, fx / "combined_history.json", F["models"]["lightgbm"], tag="G11")
    check("G11", jbytes(lines[0]["pool"]) == jbytes(cli["predictions"]), "record mode != CLI")
    return {"victim_feature": victim, "default": "refused", "record": "== CLI pool"}


def g13_cpu_nn(F, out_root):
    """CPU mode: a neural_net sidecar cannot be loaded with CUDA hidden. Both the harness
    and the CLI must stop; no pool may be written. Records the exact failure (V6)."""
    fx, n = F["fx"], len(F["combined"])
    out = {}
    RUN_N["n"] += 0
    try:
        run_harness(F, out_root, ["--models-dir", str(F["models"]["nn"]), "--block-start", str(n),
                                  "--block-end", str(n + 1)])
        raise GateFail("G13cpu", "NN sidecar produced a pool on CPU")
    except GateFail:
        raise
    except Exception as e:
        out["harness"] = f"{type(e).__name__}: {str(e)[:200]}"
    try:
        run_cli(F, fx / "survivors_with_scores.json", fx / "combined_history.json",
                F["models"]["nn"], tag="G13cpu")
        raise GateFail("G13cpu", "CLI produced a pool with an NN sidecar on CPU")
    except RuntimeError as e:
        lines = [l for l in str(e).splitlines() if l.strip()]
        out["cli"] = lines[0][:120] + " :: " + (lines[-1][:200] if lines else "")
    return out


# ------------------------------- GPU mode (owner go 08:29; synthetic fixtures only) -------
class GpuMonitor:
    """Polls nvidia-smi every 0.2 s for total card memory used; records the peak."""
    def __init__(self):
        import threading
        self.peak = 0
        self.samples = 0
        self.baseline = self._read()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _read():
        p = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                            "-i", "0"], capture_output=True, text=True, timeout=10)
        return int(p.stdout.strip().splitlines()[0])

    def _run(self):
        while not self._stop.is_set():
            try:
                v = self._read()
                self.peak = max(self.peak, v)
                self.samples += 1
            except Exception:
                pass
            self._stop.wait(0.2)

    def start(self):
        self._t.start(); return self

    def stop(self):
        self._stop.set(); self._t.join(timeout=5)
        return {"baseline_mib": self.baseline, "peak_mib": self.peak,
                "peak_above_baseline_mib": self.peak - self.baseline, "samples": self.samples}


def card_busy():
    """Other compute processes on the card, excluding the allowlisted `sunshine` and us."""
    p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                        "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
    busy = []
    for line in p.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and parts[1].split("/")[-1] != "sunshine" and parts[0] != str(os.getpid()):
            busy.append(line)
    return busy


def gg1_gpu_identity(F, out_root):
    """GPU visible: H6 pool = CLI pool at t = len(history) for LightGBM, XGBoost, NN."""
    fx, n = F["fx"], len(F["combined"])
    out = []
    for model in ("lightgbm", "xgboost", "nn"):
        cli = run_cli(F, fx / "survivors_with_scores.json", fx / "dataset.json", F["models"][model],
                      tag=f"GG1 {model}")
        rec, lines = run_harness(F, out_root, ["--models-dir", str(F["models"][model]),
                                               "--block-start", str(n), "--block-end", str(n + 1)])
        check("GG1", jbytes(lines[0]["pool"]) == jbytes(cli["predictions"]),
              f"{model}: {lines[0]['pool']} != CLI {cli['predictions']}")
        check("GG1", jbytes(lines[0]["raw_scores"]) == jbytes(cli["raw_scores"]), f"{model}: raw_scores")
        out.append(f"{model}: = CLI ({len(lines[0]['pool'])} ints), "
                   f"device={rec['ranking_info'].get('nn_normalisation', {}).get('device', 'n/a')}")
    return out


def gg2_gpu_walk(F, out_root):
    """NN on GPU: walk-forward lines equal the CLI on history[:t]; 2 forked workers = 1."""
    fx = F["fx"]
    nn = ["--models-dir", str(F["models"]["nn"]), "--block-start", "290", "--block-end", "300"]
    rec1, l1 = run_harness(F, out_root, nn)
    rec2, _ = run_harness(F, out_root, nn + ["--workers", "2"])
    check("GG2", rec1["pools_jsonl"]["sha256"] == rec2["pools_jsonl"]["sha256"], "workers=2 != 1 on GPU")
    check("GG2", rec1["ranking_proof"]["model_calls"] == 1, "ranking once")
    by_t = {l["t"]: l for l in l1}
    for t in (290, 299):
        hf = fx / f"hist_upto_{t}.json"
        hf.write_text(json.dumps(F["combined"][:t]))
        cli = run_cli(F, fx / "survivors_with_scores.json", hf, F["models"]["nn"], tag=f"GG2 t={t}")
        check("GG2", jbytes(by_t[t]["pool"]) == jbytes(cli["predictions"]), f"t={t} != CLI")
    return {"lines": len(l1), "workers2_equal": True, "nn_normalisation": rec1["ranking_info"]["nn_normalisation"]}


def g13_nn_check(F, out_root):
    """Lead rule: a neural_net sidecar without the input scaler or without y_mean/y_std is
    refused. The CLI runs best-effort on the same sidecars (recorded as observations)."""
    fx, n = F["fx"], len(F["combined"])
    blk = ["--block-start", str(n), "--block-end", str(n + 1)]
    out = {}
    for name, needle in (("nn_noscaler", "input scaler not loaded"), ("nn_noy", "y_mean/y_std not loaded")):
        try:
            run_harness(F, out_root, ["--models-dir", str(F["models"][name])] + blk)
            raise GateFail("G13", f"{name}: accepted (no refusal)")
        except H.HarnessError as e:
            check("G13", needle in str(e), f"{name}: wrong refusal {e}")
            out[name] = "refused: " + str(e)[:110]
        cli = run_cli(F, fx / "survivors_with_scores.json", fx / "combined_history.json",
                      F["models"][name], tag=f"G13 {name}")
        out[name + "_cli"] = (f"OBSERVED: CLI exit 0, status={cli.get('status')}, "
                              f"{len(cli.get('predictions', []))} predictions, raw_scores "
                              f"[{min(cli['raw_scores']):.4g}, {max(cli['raw_scores']):.4g}]")
    # the complete sidecar is accepted and its report is complete (checked after the refusals)
    rec, _ = run_harness(F, out_root, ["--models-dir", str(F["models"]["nn"])] + blk)
    rep_ = rec["ranking_info"].get("nn_normalisation") or {}
    check("G13", rep_.get("normalize_features") is True
          and rep_.get("scaler_len") == rep_.get("schema_len") and rep_.get("y_std") is not None,
          f"complete NN sidecar report {rep_}")
    out["complete"] = rep_
    return out


def gpu_mutants(F, out_root):
    res = []
    def m15():
        hits = {"n": 0}; orig = H.check_nn_normalisation
        def mut(gen):
            hits["n"] += 1
            return {"disabled": True}
        H.check_nn_normalisation = mut
        return (lambda: setattr(H, "check_nn_normalisation", orig)), hits
    res.append(run_mutant("M15 NN normalisation check removed", m15, lambda: g13_nn_check(F, out_root), "G13"))

    def m16():
        hits = {"n": 0}; orig = H.check_nn_normalisation
        def mut(gen):  # y-part blinded: pretend y_mean/y_std exist
            hits["n"] += 1
            saved = (getattr(gen.model, "_y_mean", None), getattr(gen.model, "_y_std", None))
            if saved[0] is None:
                gen.model._y_mean, gen.model._y_std = 0.0, 1.0
            try:
                return orig(gen)
            finally:
                gen.model._y_mean, gen.model._y_std = saved
        H.check_nn_normalisation = mut
        return (lambda: setattr(H, "check_nn_normalisation", orig)), hits
    res.append(run_mutant("M16 y_mean/y_std part blinded", m16, lambda: g13_nn_check(F, out_root), "G13"))
    return res


def main_gpu(out_root):
    import resource as _r
    waited = 0
    while card_busy() and waited < 300:
        print(f"[gpu] card busy, waiting: {card_busy()}", flush=True)
        time.sleep(15); waited += 15
    if card_busy():
        print("[gpu] card still busy after 300 s; not running", flush=True)
        return 1
    mon = GpuMonitor().start()
    t_all = time.time()
    summary = {"mode": "gpu", "out_root": str(out_root), "gates": {}, "mutants": []}
    F = build_fixture(out_root)
    results = []
    for name, fn in (("GG1 GPU byte identity (lgb, xgb, nn) vs CLI", lambda: gg1_gpu_identity(F, out_root)),
                     ("GG2 NN walk-forward on GPU + workers=2", lambda: gg2_gpu_walk(F, out_root)),
                     ("G13 NN scaler + y_mean/y_std check", lambda: g13_nn_check(F, out_root))):
        t = time.time()
        try:
            r = fn(); results.append(True)
            summary["gates"][name] = {"pass": True, "detail": r, "s": round(time.time() - t, 1)}
            print(f"PASS {name} ({time.time() - t:.1f}s): {str(r)[:700]}", flush=True)
        except Exception as e:
            results.append(False)
            summary["gates"][name] = {"pass": False, "error": f"{type(e).__name__}: {e}"}
            print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)
    mres = gpu_mutants(F, out_root)
    for name, killed, msg in mres:
        print(f"{'KILLED' if killed else 'SURVIVED'} {name}: {msg}", flush=True)
        summary["mutants"].append({"name": name, "killed": killed, "detail": msg})
    summary["card_memory"] = mon.stop()
    summary["wall_seconds"] = round(time.time() - t_all, 1)
    summary["peak_rss_mib"] = {"self": round(_r.getrusage(_r.RUSAGE_SELF).ru_maxrss / 1024, 1),
                               "largest_child": round(_r.getrusage(_r.RUSAGE_CHILDREN).ru_maxrss / 1024, 1)}
    summary["cli_runs"] = CLI_RUNS
    ok = all(results) and all(k for _, k, _ in mres) and summary["card_memory"]["peak_above_baseline_mib"] < 3000
    summary["verdict"] = {"gates": f"{sum(results)}/{len(results)} pass",
                          "mutants": f"{sum(1 for _, k, _ in mres if k)}/{len(mres)} killed",
                          "card_under_3GB": summary["card_memory"]["peak_above_baseline_mib"] < 3000}
    (out_root / "selftest_gpu_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[gpu] {summary['verdict']} wall {summary['wall_seconds']}s card {summary['card_memory']} "
          f"rss {summary['peak_rss_mib']}", flush=True)
    return 0 if ok else 1


def load_grader_readonly():
    """Import the grader from its own worktree without writing there (no __pycache__)."""
    if not GRADER_PATH.is_file():
        raise GateFail("G12", f"grader not found at {GRADER_PATH}")
    sys.dont_write_bytecode = True
    gspec = importlib.util.spec_from_file_location("e2e_grader_readonly", GRADER_PATH)
    G = importlib.util.module_from_spec(gspec)
    sys.modules["e2e_grader_readonly"] = G
    gspec.loader.exec_module(G)
    return G


def grader_state():
    pyc = GRADER_PATH.parent / "__pycache__"
    return {"grader_sha256": H.sha256_file(GRADER_PATH),
            "pycache_exists": pyc.exists(),
            "pycache_entries": sorted(x.name for x in pyc.iterdir()) if pyc.exists() else []}


def g12_grader_reads(F, out_root, run_cli_grader=False):
    """H6 fixture output for three arms goes through the grader's own reader functions
    (read_pool_records, load_dataset, parse_*, resolve_records), unmodified and read-only."""
    fx, n = F["fx"], len(F["combined"])
    before = grader_state()
    G = load_grader_readonly()
    blk = ["--block-start", str(UNSEEN_FROM), "--block-end", str(n + 1)]
    recA1, lA1 = run_harness(F, out_root, ["--models-dir", str(F["models"]["lightgbm"]),
                                           "--arm-id", "A1fx", "--data-cutoff-index",
                                           str(UNSEEN_FROM)] + blk)
    recA0, lA0 = run_harness(F, out_root, ["--ranking", "file_order", "--arm-id", "A0fx"] + blk)
    recAb, lAb = run_harness(F, out_root, ["--models-dir", str(F["models"]["gate_blocked"]),
                                           "--arm-id", "Ablk", "--data-cutoff-index",
                                           str(UNSEEN_FROM)] + blk)
    gdir = out_root / f"grader_inputs_{RUN_N['n']:03d}"
    gdir.mkdir()
    pools = gdir / "pools_all_arms.jsonl"
    with open(pools, "wb") as f:
        for r in (recA1, recA0, recAb):
            f.write(Path(r["pools_jsonl"]["path"]).read_bytes())
    ds_sha = H.sha256_file(fx / "dataset.json")
    gb = {"schema": G.SCHEMA_GRADE_BLOCK, "name": "fixture_unseen",
          "start_index": UNSEEN_FROM, "end_index": n + 1, "sessions": ["evening", "midday"],
          "frequency_control_block": {"start_index": 0, "end_index": 240}}
    arms = {"schema": G.SCHEMA_ARMS, "arms": [
        {"arm_id": "A1fx", "config_id": recA1["config_id"], "upstream_unseen_from_index": UNSEEN_FROM},
        {"arm_id": "A0fx", "config_id": recA0["config_id"], "upstream_unseen_from_index": UNSEEN_FROM},
        {"arm_id": "Ablk", "config_id": recAb["config_id"], "upstream_unseen_from_index": UNSEEN_FROM}]}
    ab = {"schema": G.SCHEMA_ABSTENTION, "allow_abstention": True, "allowed_reasons": None,
          "empty_pool": "error"}
    for name, obj in (("grade_block.json", gb), ("arms.json", arms), ("abstention.json", ab)):
        (gdir / name).write_text(json.dumps(obj, indent=2))
    try:
        records = G.read_pool_records(pools)
        dataset, dsha = G.load_dataset(fx / "dataset.json", ds_sha, 1000)
        gbo, _ = G.load_declaration(gdir / "grade_block.json", G.SCHEMA_GRADE_BLOCK, "grade block")
        aro, _ = G.load_declaration(gdir / "arms.json", G.SCHEMA_ARMS, "arm list")
        abo, _ = G.load_declaration(gdir / "abstention.json", G.SCHEMA_ABSTENTION, "abstention")
        per_arm, eligible = G.resolve_records(records, G.parse_arms(aro), G.parse_grade_block(gbo),
                                              G.parse_abstention(abo), dataset, dsha,
                                              dict(G.DEFAULTS))
    except G.GraderError as e:
        raise GateFail("G12", f"grader reader rejected H6 output: {e}")
    m = n - UNSEEN_FROM
    check("G12", len(eligible) == m, f"eligible {len(eligible)} != {m}")
    for aid, lines in (("A1fx", lA1), ("A0fx", lA0)):
        pa = per_arm[aid]
        check("G12", len(pa["graded"]) == m and len(pa["pending"]) == 1 and not pa["abstained"],
              f"{aid}: graded {len(pa['graded'])} pending {len(pa['pending'])}")
        by_t = {l["t"]: l for l in lines}
        for e in pa["graded"] + pa["pending"]:
            l = by_t[e["combined_index"]]
            check("G12", e["pool_sha256"] == l["pool_sha256"] == G.pool_sha256(l["pool"]),
                  f"{aid} t={e['combined_index']}: pool sha differs")
            check("G12", e["k_raw"] == len(l["pool"]) and e["k"] == len(set(l["pool"])), "k")
    check("G12", len(per_arm["Ablk"]["abstained"]) == m and len(per_arm["Ablk"]["pending"]) == 1,
          "gate-blocked arm: abstentions not read as abstentions")
    after = grader_state()
    check("G12", after == before, f"grader worktree changed: {before} -> {after}")
    detail = {"eligible": m, "A1fx": "graded %d, pending 1" % m, "A0fx": "graded %d, pending 1" % m,
              "Ablk": "abstained %d, pending 1" % m, "grader_sha256": before["grader_sha256"],
              "grader_pycache_unchanged": True}
    if run_cli_grader:
        # observation, not the credited assertion: the grader's full CLI on two arms
        pools2 = gdir / "pools_A1_A0.jsonl"
        pools2.write_bytes(Path(recA1["pools_jsonl"]["path"]).read_bytes()
                           + Path(recA0["pools_jsonl"]["path"]).read_bytes())
        arms2 = dict(arms, arms=arms["arms"][:2])
        (gdir / "arms_A1_A0.json").write_text(json.dumps(arms2, indent=2))
        gout = gdir / "grader_out"
        p = subprocess.run([sys.executable, "-B", str(GRADER_PATH), "--pools", str(pools2),
                            "--dataset", str(fx / "dataset.json"), "--dataset-sha256", ds_sha,
                            "--grade-block", str(gdir / "grade_block.json"),
                            "--arms", str(gdir / "arms_A1_A0.json"),
                            "--abstention", str(gdir / "abstention.json"),
                            "--out-dir", str(gout), "--no-provenance", "--workers", "1",
                            "--control-reps", "50", "--run-label", "H6 fixture format check"],
                           capture_output=True, text=True, timeout=600, cwd=str(gdir))
        detail["grader_cli_exit"] = p.returncode
        detail["grader_cli_tail"] = (p.stdout + p.stderr).strip().splitlines()[-3:]
        detail["grader_cli_outputs"] = sorted(x.name for x in gout.iterdir()) if gout.exists() else []
        after2 = grader_state()
        check("G12", after2 == before, f"grader worktree changed by its CLI run: {after2}")
    return detail


SEAL_AT = 270


def g14_sealed(F, out_root):
    """--seal-from (TODO-E2E-V10): the dataset's draw values at index >= seal are never held.
    Proof by a dataset whose sealed draws are not even integers: the sealed run must succeed
    and give the same pools and diagnostics as the unsealed run on the real dataset."""
    fx = F["fx"]
    h270 = fx / "hold_upto_270.json"
    h270.write_text(json.dumps(F["hold"][:SEAL_AT - len(F["train"])]))
    hist = [fx / "train_history.json", h270]
    ds = json.loads((fx / "dataset.json").read_text())
    garbage = fx / "dataset_sealed_garbage.json"
    garbage.write_text(json.dumps([dict(r, draw="SEALED-NOT-AN-INT") if i >= SEAL_AT else r
                                   for i, r in enumerate(ds)]))
    n = len(F["combined"])
    blk = ["--models-dir", str(F["models"]["lightgbm"]), "--block-start", "265",
           "--block-end", str(n + 1)]
    _, base = run_harness(F, out_root, blk)
    try:
        _, real = run_harness(F, out_root, blk + ["--seal-from", str(SEAL_AT)], history=hist)
        rec_g, garb = run_harness(F, out_root, blk + ["--seal-from", str(SEAL_AT)], history=hist,
                                  dataset=garbage)
    except H.HarnessError as e:
        raise GateFail("G14", f"sealed run touched a sealed draw value: {e}")
    strip = lambda ln, ks: {k: v for k, v in ln["rec"].items() if k not in ks}
    for name, other, ks in (("real", real, {"config_id"}),
                            ("garbage", garb, {"config_id", "dataset_sha256"})):
        check("G14", len(other) == len(base) == n + 1 - 265, f"{name}: line count")
        for a, b in zip(base, other):
            check("G14", strip(a, ks) == strip(b, ks),
                  f"{name} t={a['t']}: sealed pool {b['pool']} != unsealed {a['pool']}")
            check("G14", a["diag"] == b["diag"], f"{name} t={a['t']}: diagnostics differ")
    seal = rec_g["inputs"]["dataset"]["seal"]
    check("G14", seal["draw_values_held_rows"] == SEAL_AT and
          seal["draw_values_discarded_at_parse_rows"] == n - SEAL_AT, f"seal record {seal}")
    _, draws = H.load_dataset(garbage, SEAL_AT)
    check("G14", len(draws) == SEAL_AT and all(isinstance(v, int) for v in draws),
          "load_dataset returned a sealed value")
    # refusals
    _, e1 = run_harness(F, out_root, blk, dataset=garbage, expect_fail=H.HarnessError)
    check("G14", "not an int" in e1, f"unsealed run on the garbage dataset: {e1}")
    _, e2 = run_harness(F, out_root, blk + ["--seal-from", str(SEAL_AT)], expect_fail=H.HarnessError)
    check("G14", "not aligned" in e2, f"history longer than the seal: {e2}")
    _, e3 = run_harness(F, out_root, blk + ["--seal-from", str(SEAL_AT), "--data-cutoff-index", "260"],
                        history=hist, expect_fail=H.HarnessError)
    check("G14", "below the history" in e3, f"cutoff below the handed history: {e3}")
    sc = H.sha256_file(F["models"]["lightgbm"] / "best_model.meta.json")
    _, e4 = run_harness(F, out_root, blk + ["--expect-checkpoint-sha256", "0" * 64],
                        expect_fail=H.HarnessError)
    check("G14", "expect-checkpoint-sha256" in e4, f"checkpoint sha mismatch not refused: {e4}")
    _, ok = run_harness(F, out_root, ["--models-dir", str(F["models"]["lightgbm"]),
                                      "--block-start", "299", "--block-end", "300",
                                      "--expect-sidecar-sha256", sc])
    return {"lines": len(base), "pad_draws": n + 1 - SEAL_AT, "seal": seal,
            "refusals": 4, "sidecar_sha_accepted": bool(ok)}


# ---------------------------------------------------------------------------
# mutants
# ---------------------------------------------------------------------------
CAPS_WITH_ROOT = Path(os.environ.get("E2E_VOTE_CAPS_WITH_ROOT",
                                    str(Path.home() / "tfm_worktrees" / "e2e-map")))


def g19_ll1_switch(F, out_root):
    """LL-1 capability check (ADDENDUM_1/2). Refusal on files without the switch; non-vacuity on
    files with it; end to end on this harness's own repo; default config records 'off'."""
    fx = F["fx"]
    cfg = json.loads((fx / "prediction_config.json").read_text())
    (cfg.get("prediction", cfg))["schema_check"] = "fail"
    cfg_ll1 = fx / "prediction_config_ll1.json"
    cfg_ll1.write_text(json.dumps(cfg, indent=2))
    check("G19", H.CAPS_PATH.is_file(), f"capability module missing: {H.CAPS_PATH}")
    # without: this repo's committed prediction_generator (+ manifest if committed) at HEAD
    RUN_N["g19"] = RUN_N.get("g19", 0) + 1            # one fresh dir per call (gate, then mutant)
    wo = out_root / f"g19_without_{RUN_N['g19']}"
    (wo / "models").mkdir(parents=True)
    for f in ("prediction_generator.py", "models/feature_manifest.py"):
        g = subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{f}"], capture_output=True)
        if g.returncode == 0:
            (wo / f).write_bytes(g.stdout)
    out = []
    try:
        r = H.ll1_check(cfg_ll1, root=wo)
        raise GateFail("G19", f"no refusal on HEAD's files (they must lack LL-1): {r}")
    except H.HarnessError as e:
        msg = str(e)
        check("G19", "LL-1" in msg and str(wo / "prediction_generator.py") in msg,
              f"refusal does not name the switch and the file: {msg}")
        out.append("HEAD files: refused, switch and file named")
    # non-vacuity: same request on files that have the switch
    check("G19", (CAPS_WITH_ROOT / "prediction_generator.py").is_file(),
          f"non-vacuity side missing: {CAPS_WITH_ROOT}")
    r = H.ll1_check(cfg_ll1, root=CAPS_WITH_ROOT)
    check("G19", r["pipeline_has_switch"] is True and r["requested"] == {"schema_check": "fail"},
          f"non-vacuity: same request not accepted on {CAPS_WITH_ROOT}: {r}")
    out.append(f"{CAPS_WITH_ROOT.name} files: accepted, applied {r['applied']}")
    # end to end on the harness's own repo
    n = len(F["combined"])
    ext = ["--models-dir", str(F["models"]["lightgbm"]), "--block-start", str(n), "--block-end", str(n + 1),
           "--config", str(cfg_ll1)]
    here_has = H.ll1_check(fx / "prediction_config.json")["pipeline_has_switch"]
    if not here_has:
        before = RUN_N["n"] + 1
        _, msg = run_harness(F, out_root, ext, expect_fail=H.HarnessError)
        check("G19", "LL-1" in msg and "prediction_generator.py" in msg and
              not (out_root / f"h{before:03d}").exists(),
              f"own repo lacks LL-1 but the harness did not refuse before writing: {msg}")
        out.append("own repo lacks LL-1: refused, no out-dir")
    else:
        rec, _ = run_harness(F, out_root, ext)
        check("G19", rec["pipeline_switches"]["LL-1"]["applied_schema_check"] == "fail",
              f"own repo has LL-1 but the record says {rec['pipeline_switches']}")
        out.append("own repo has LL-1: ran, record says schema_check fail")
    rec, _ = run_harness(F, out_root, ext[:-2])
    check("G19", rec["pipeline_switches"]["LL-1"]["requested"] == {} and
          rec["pipeline_switches"]["LL-1"]["applied_schema_check"] == "off",
          f"default config: record says {rec['pipeline_switches']}")
    out.append("default config: record says requested {}, applied off")
    return "; ".join(out)


def run_mutant(name, apply, gate, expect_tag):
    """apply() -> (restore, hits). Killed iff applied, path ran, and GateFail(expect_tag)."""
    restore, hits = apply()
    try:
        try:
            gate()
            return name, False, "gate passed under mutant (SURVIVED)"
        except GateFail as e:
            ok = e.tag == expect_tag and hits["n"] > 0
            return name, ok, f"tag={e.tag} hits={hits['n']} :: {str(e)[:110]}"
        except Exception as e:  # not credited
            return name, False, f"uncredited {type(e).__name__}: {str(e)[:110]} hits={hits['n']}"
    finally:
        restore()


def mutants(F, out_root):
    res = []
    n = len(F["combined"])
    fx = F["fx"]

    def g1_light():
        cli = run_cli(F, fx / "survivors_with_scores.json", fx / "combined_history.json",
                      F["models"]["lightgbm"], tag="M")
        _, lines = run_harness(F, out_root, ["--models-dir", str(F["models"]["lightgbm"]),
                                             "--block-start", str(n), "--block-end", str(n + 1)])
        check("G1", jbytes(lines[0]["pool"]) == jbytes(cli["predictions"]), "pool != CLI")

    def g2_light():
        hf = fx / "hist_upto_275.json"
        cli = run_cli(F, fx / "survivors_with_scores.json", hf, F["models"]["lightgbm"], tag="M")
        _, lines = run_harness(F, out_root, ["--models-dir", str(F["models"]["lightgbm"]),
                                             "--block-start", "275", "--block-end", "276"])
        check("G2", jbytes(lines[0]["pool"]) == jbytes(cli["predictions"]), "pool != CLI at t=275")

    # M1 vote index off by one (next_idx = t - 1)
    def m1():
        hits = {"n": 0}; orig = H.vote_index_for
        def mut(arm, t, ctx):
            hits["n"] += 1
            return orig(arm, t, ctx) - 1
        H.vote_index_for = mut
        return (lambda: setattr(H, "vote_index_for", orig)), hits
    res.append(run_mutant("M1 vote index t-1", m1, g2_light, "G2"))

    # M2 ranking reversed inside RankOnce
    def m2():
        hits = {"n": 0}; orig = H.RankOnce.predict
        def mut(self, X):
            hits["n"] += 1
            return -orig(self, X)
        H.RankOnce.predict = mut
        return (lambda: setattr(H.RankOnce, "predict", orig)), hits
    res.append(run_mutant("M2 ranking reversed", m2, g1_light, "G1"))

    # M3 features drift after the first draw: RankOnce must stop the run
    def m3():
        hits = {"n": 0}; orig = H.vote_one
        def mut(gen, survivors, combined, t, arm, k):
            if t > 280:
                hits["n"] += 1
                f0 = survivors[0]["features"]
                for key in list(f0):  # every numeric feature, so a schema column changes
                    if isinstance(f0[key], float):
                        f0[key] += 1.0
            return orig(gen, survivors, combined, t, arm, k)
        H.vote_one = mut
        return (lambda: setattr(H, "vote_one", orig)), hits
    def g3_drift():
        try:
            run_harness(F, out_root, ["--models-dir", str(F["models"]["lightgbm"]),
                                      "--block-start", "280", "--block-end", "284"])
        except H.HarnessError as e:
            check("G3", "changed between draws" not in str(e), "RankOnce caught X drift: " + str(e)[:60])
            raise
    res.append(run_mutant("M3 X drift between draws", m3, g3_drift, "G3"))

    # M4 a line is dropped: the block integrity check must stop the run
    def m4():
        hits = {"n": 0}; orig = H.make_line
        def mut(t, i, *a):
            ln = orig(t, i, *a)
            if t == 282:
                hits["n"] += 1
                ln["pool"]["target"]["combined_index"] = 283
            return ln
        H.make_line = mut
        return (lambda: setattr(H, "make_line", orig)), hits
    def g_integrity():
        try:
            run_harness(F, out_root, ["--models-dir", str(F["models"]["lightgbm"]),
                                      "--block-start", "280", "--block-end", "285"])
        except H.HarnessError as e:
            check("GI", "one-for-one" not in str(e), "integrity check fired: " + str(e)[:60])
            raise
    res.append(run_mutant("M4 line dropped/shifted", m4, g_integrity, "GI"))

    # M5 repeats collapsed in the pool
    def m5():
        hits = {"n": 0}; orig = H.make_line
        def mut(t, i, vi, status, reason, pool, *a):
            hits["n"] += 1
            return orig(t, i, vi, status, reason, list(dict.fromkeys(pool)), *a)
        H.make_line = mut
        return (lambda: setattr(H, "make_line", orig)), hits
    res.append(run_mutant("M5 repeats collapsed", m5, lambda: g9_repeats(F, out_root), "G9"))

    # M6 per_skip override ignored
    def m6():
        hits = {"n": 0}; orig = H.survivors_for_arm
        def mut(arm, rows, arm_skip):
            if arm == "per_skip":
                hits["n"] += 1
                return rows, {}
            return orig(arm, rows, arm_skip)
        H.survivors_for_arm = mut
        return (lambda: setattr(H, "survivors_for_arm", orig)), hits
    def g7_light():
        _, l3 = run_harness(F, out_root, ["--models-dir", str(F["models"]["lightgbm"]),
                                          "--block-start", str(n), "--block-end", str(n + 1),
                                          "--vote-arm", "per_skip", "--arm-skip", "3"])
        cli = run_cli(F, fx / "survivors_skip3.json", fx / "combined_history.json",
                      F["models"]["lightgbm"], tag="M6")
        check("G7", jbytes(l3[0]["pool"]) == jbytes(cli["predictions"]), "per_skip 3 != CLI")
    res.append(run_mutant("M6 per_skip ignored", m6, g7_light, "G7"))

    # M7 the no-model refusal removed: the mean fallback would be credited as a pool
    def m7():
        hits = {"n": 0}; orig = H.build_generator
        def mut(pg, config, logger, args, rows, npz_map):
            gen = pg.PredictionGenerator(config, logger)
            if gen.model is None:
                hits["n"] += 1
                return gen, None, {"ranking": "sidecar", "signal_gate_blocked": False}
            return orig(pg, config, logger, args, rows, npz_map)
        H.build_generator = mut
        return (lambda: setattr(H, "build_generator", orig)), hits
    def g6_light():
        try:
            run_harness(F, out_root, ["--models-dir", str(F["models"]["missing_ckpt"]),
                                      "--block-start", "299", "--block-end", "300",
                                      "--no-require-schema"])
        except H.HarnessError as e:
            check("G6", "no model loaded" in str(e), str(e))
            return
        raise GateFail("G6", "missing checkpoint produced a pool (mean fallback credited)")
    res.append(run_mutant("M7 no-model refusal removed", m7, g6_light, "G6"))

    # M8 the CLI's dual-sieve setting left on: with a reverse file it would change the population
    def m8():
        hits = {"n": 0}; orig = H.vote_one
        def mut(gen, survivors, combined, t, arm, k):
            hits["n"] += 1
            gen.config.use_dual_sieve = True
            result = gen.generate_predictions(survivors, combined[:t], survivors[:5], k=k)
            return t, "pool", None, [int(p) for p in result["predictions"]], \
                [float(x) for x in result.get("raw_scores", [])], None
        H.vote_one = mut
        return (lambda: setattr(H, "vote_one", orig)), hits
    res.append(run_mutant("M8 reverse survivors passed", m8, g1_light, "G1"))

    # M9 the declared tie-break is ignored
    def m9():
        hits = {"n": 0}; orig = H.tie_break_key
        def mut(rule, rows):
            hits["n"] += 1
            return None
        H.tie_break_key = mut
        return (lambda: setattr(H, "tie_break_key", orig)), hits
    res.append(run_mutant("M9 tie-break ignored", m9, lambda: g10_tie_break(F, out_root), "G10"))

    # M10 the missing-feature check reports nothing
    def m10():
        hits = {"n": 0}; orig = H.missing_schema_features
        def mut(rows, names):
            hits["n"] += 1
            return {"rows_with_missing": 0, "per_feature": {}}
        H.missing_schema_features = mut
        return (lambda: setattr(H, "missing_schema_features", orig)), hits
    def g11_light():
        try:
            g11_missing_feature(F, out_root)
        except GateFail as e:
            if e.tag == "FAILCLOSED":
                raise GateFail("G11", "missing feature not refused: " + str(e)[:80])
            raise
    res.append(run_mutant("M10 missing-feature check blind", m10, g11_light, "G11"))

    g12 = lambda: g12_grader_reads(F, out_root)

    # M11 data_cutoff_index written after the target (forecast would use the target)
    def m11():
        hits = {"n": 0}; orig = H.cutoff_for
        def mut(t, ctx):
            hits["n"] += 1
            return t + 1
        H.cutoff_for = mut
        return (lambda: setattr(H, "cutoff_for", orig)), hits
    res.append(run_mutant("M11 cutoff after target", m11, g12, "G12"))

    # M12 target date taken from the neighbouring row (stale target)
    def m12():
        hits = {"n": 0}; orig = H.target_for
        def mut(t, ctx):
            hits["n"] += 1
            tg = orig(t, ctx)
            if t + 1 < len(ctx["identity"]):
                tg["date"], tg["session"] = ctx["identity"][t + 1]
            return tg
        H.target_for = mut
        return (lambda: setattr(H, "target_for", orig)), hits
    res.append(run_mutant("M12 stale target date/session", m12, g12, "G12"))

    # M13 pool_sha256 by a different serialisation
    def m13():
        hits = {"n": 0}; orig = H.pool_sha256
        def mut(pool):
            hits["n"] += 1
            return H.sha256_text(json.dumps(pool))
        H.pool_sha256 = mut
        return (lambda: setattr(H, "pool_sha256", orig)), hits
    res.append(run_mutant("M13 pool_sha256 redefined", m13, g12, "G12"))

    # M14 dataset_sha256 of the wrong file
    def m14():
        hits = {"n": 0}; orig = H.make_line
        def mut(*a):
            hits["n"] += 1
            ln = orig(*a)
            ln["pool"]["dataset_sha256"] = H.sha256_text("not the dataset")
            return ln
        H.make_line = mut
        return (lambda: setattr(H, "make_line", orig)), hits
    res.append(run_mutant("M14 dataset_sha256 wrong", m14, g12, "G12"))

    g14 = lambda: g14_sealed(F, out_root)

    # M17 the seal is ignored at parse time: sealed draw values are held and validated
    def m17():
        hits = {"n": 0}; orig = H.load_dataset
        def mut(path, seal_from=None):
            if seal_from is not None:
                hits["n"] += 1
            return orig(path, None)
        H.load_dataset = mut
        return (lambda: setattr(H, "load_dataset", orig)), hits
    res.append(run_mutant("M17 seal ignored (sealed draws read)", m17, g14, "G14"))

    # M18 the padded history is one short: the vote index past the seal shifts
    def m18():
        hits = {"n": 0}; orig = H.history_for
        def mut(combined, vi):
            h = orig(combined, vi)
            if vi > len(combined):
                hits["n"] += 1
                return h[:-1]
            return h
        H.history_for = mut
        return (lambda: setattr(H, "history_for", orig)), hits
    res.append(run_mutant("M18 padded history one short", m18, g14, "G14"))

    # M25 the LL-1 capability check is skipped: a schema_check config is accepted anywhere
    def m25():
        hits = {"n": 0}; orig = H.ll1_check
        def mut(config_path, root=None):
            hits["n"] += 1
            return {"requested": {}, "pipeline_has_switch": None, "files_checked": [],
                    "applied": "default (switch not used)"}
        H.ll1_check = mut
        return (lambda: setattr(H, "ll1_check", orig)), hits
    res.append(run_mutant("M25 LL-1 capability check skipped", m25,
                          lambda: g19_ll1_switch(F, out_root), "G19"))
    return res


# ---------------------------------------------------------------------------
# scale probe (synthetic, CPU): run-H-shaped size, to estimate time and memory
# ---------------------------------------------------------------------------
def scale_probe(out_root, n_rows=12156, n_feat=89, hist_len=18068, n_draws=24, workers=6):
    import numpy as np
    from models.wrappers.lightgbm_wrapper import LightGBMWrapper
    d = out_root / "scale"; d.mkdir()
    rs = np.random.RandomState(7)
    names = [f"f{i:02d}" for i in range(n_feat)]
    Xf = rs.rand(n_rows, n_feat)
    rows = [{"seed": int(s), "score": 0.0,
             "features": {nm: float(v) for nm, v in zip(names, xr)}, "metadata": {}}
            for s, xr in zip(rs.randint(0, 2 ** 31, n_rows), Xf)]
    (d / "survivors_with_scores.json").write_text(json.dumps(rows))
    hist = [int(x) for x in rs.randint(0, 1000, hist_len)]
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
    for w, nd in ((1, 6), (workers, n_draws)):
        o = d / f"out_w{w}"
        t0 = time.time()
        cmd = [sys.executable, "-u", str(HARNESS_PATH), "--survivors", str(d / "survivors_with_scores.json"),
               "--dataset", str(d / "dataset.json"), "--data-cutoff-index", "t",
               "--pending-date", "dNEXT", "--pending-session", "midday",
               "--config", str(d / "cfg.json"),
               "--models-dir", str(m), "--block-start", str(hist_len - nd + 1),
               "--block-end", str(hist_len + 1), "--workers", str(w), "--out-dir", str(o)]
        p = subprocess.run(["/usr/bin/time", "-v"] + cmd, cwd=str(d), capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(p.stderr[-2000:])
        rss = [l for l in p.stderr.splitlines() if "Maximum resident" in l]
        rec = json.loads((o / "run_record.json").read_text())
        out[f"workers={w}"] = {"draws": nd, "wall_s": round(time.time() - t0, 1),
                               "vote_s": rec["timing"]["vote_seconds"],
                               "vote_s_per_draw": round(rec["timing"]["vote_seconds"] / nd, 3),
                               "time_v_max_rss": rss[0].strip() if rss else None,
                               "record_memory": rec["memory"]}
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--scale-probe", action="store_true")
    ap.add_argument("--gpu", action="store_true",
                    help="GPU-path checks only (owner go 2026-11-23 08:29; synthetic fixtures)")
    a = ap.parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(a.out_root) if a.out_root else REPO / "data" / "e2e_vote_selftest" / stamp
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=False)
    if a.gpu:
        print(f"[gpu] out_root={out_root}", flush=True)
        return main_gpu(out_root)
    t_all = time.time()
    print(f"[selftest] out_root={out_root}", flush=True)
    summary = {"out_root": str(out_root), "gates": {}, "mutants": []}

    t0 = time.time()
    F = build_fixture(out_root)
    summary["fixture"] = {"rows": F["n_rows"], "features_in_schema": F["feature_count"],
                          "history": len(F["combined"]), "seconds": round(time.time() - t0, 1)}
    print(f"[selftest] fixture {summary['fixture']}", flush=True)

    results = []
    def gate(name, fn):
        t = time.time()
        try:
            r = fn()
            results.append((name, True)); summary["gates"][name] = {"pass": True, "detail": r,
                                                                    "s": round(time.time() - t, 1)}
            print(f"PASS {name} ({time.time() - t:.1f}s): {str(r)[:600]}", flush=True)
            return r
        except Exception as e:
            results.append((name, False)); summary["gates"][name] = {"pass": False, "error": str(e)}
            print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)
            return None

    gate("G1 byte identity at t=len(history) vs CLI", lambda: g1_byte_identity(F, out_root))
    g2 = gate("G2 walk-forward t=250..299 vs CLI on history[:t]", lambda: g2_walk_forward(F, out_root))
    if g2:
        rec1, lines1 = g2
        summary["gates"]["G2 walk-forward t=250..299 vs CLI on history[:t]"]["detail"] = \
            f"{len(lines1)} lines, sha {rec1['pools_jsonl']['sha256'][:16]}"
        gate("G3 ranking computed once, proven per draw", lambda: g3_rank_once(rec1, lines1))
        gate("G4 workers=4 byte-equal to workers=1", lambda: g4_workers(F, out_root, rec1, lines1))
        gate("G7 per_skip arm (hook implemented, CLI oracle)", lambda: g7_per_skip(F, out_root, lines1))
    gate("G5 abstentions written, one line per draw", lambda: g5_abstain(F, out_root))
    gate("G6 fail-closed cases", lambda: g6_fail_closed(F, out_root))
    gate("G8 unlearned ranking keys (A0)", lambda: g8_unlearned(F, out_root))
    gate("G9 raw pool keeps repeats", lambda: g9_repeats(F, out_root))
    gate("G10 declared tie-break and tie report", lambda: g10_tie_break(F, out_root))
    gate("G11 missing schema feature: fail by default, CLI-identical in record mode",
         lambda: g11_missing_feature(F, out_root))
    gate("G12 H6 output read by the grader's own loader (read-only)",
         lambda: g12_grader_reads(F, out_root, run_cli_grader=True))
    gate("G13cpu neural_net sidecar with CUDA hidden: no pool, exact failure recorded",
         lambda: g13_cpu_nn(F, out_root))
    gate("G14 sealed dataset: sealed draws never held, pools unchanged",
         lambda: g14_sealed(F, out_root))
    gate("G19 LL-1 switch: refusal without it, non-vacuity with it, record states schema_check",
         lambda: g19_ll1_switch(F, out_root))

    t = time.time()
    mres = mutants(F, out_root)
    for name, killed, msg in mres:
        print(f"{'KILLED' if killed else 'SURVIVED'} {name}: {msg}", flush=True)
        summary["mutants"].append({"name": name, "killed": killed, "detail": msg})
    summary["mutants_seconds"] = round(time.time() - t, 1)

    if a.scale_probe:
        t = time.time()
        try:
            summary["scale_probe"] = scale_probe(out_root)
            print(f"[scale] {json.dumps(summary['scale_probe'])}", flush=True)
        except Exception as e:
            summary["scale_probe"] = {"error": str(e)}
            print(f"[scale] ERROR {e}", flush=True)

    summary["cli_runs"] = CLI_RUNS
    summary["total_seconds"] = round(time.time() - t_all, 1)
    summary["peak_rss_mib"] = {
        "self": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "largest_child": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1)}
    n_gate_fail = sum(1 for _, ok in results if not ok)
    n_surv = sum(1 for _, k, _ in mres if not k)
    summary["verdict"] = {"gates": f"{len(results) - n_gate_fail}/{len(results)} pass",
                          "mutants": f"{len(mres) - n_surv}/{len(mres)} killed"}
    (out_root / "selftest_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[selftest] gates {summary['verdict']['gates']}, mutants {summary['verdict']['mutants']}, "
          f"{summary['total_seconds']}s, peak RSS {summary['peak_rss_mib']} MiB", flush=True)
    return 0 if (n_gate_fail == 0 and n_surv == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
