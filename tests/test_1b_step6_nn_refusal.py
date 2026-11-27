#!/usr/bin/env python3
"""
tests/test_1b_step6_nn_refusal.py - [item 1b, owner 2026-11-26 17:16]

Step 6 refuses a neural_net it cannot normalise the way the net was trained.
CPU only, synthetic fixtures only, writes only under --out (default: a new temp dir).

Run:  python3 -u tests/test_1b_step6_nn_refusal.py [--out DIR] [--only GATE ...] | tee <log>

Fixtures: tests/test_e2e_vote_harness.build_fixture (40 seeds through the real Step 3 worker;
LightGBM and XGBoost through the project wrappers; neural nets through Step 5's own
train_single_trial.train_neural_net). The broken neural nets are copies of the complete one's
checkpoint with one thing removed or damaged.

Gates (each must pass):
  S6NN.complete_nn_scaled        a complete NN is accepted; model.predict receives exactly (X - mean) / scale
  S6NN.refuses_at_load           ten broken NNs: the constructor raises NNNormalisationError naming the
                                 cause and the checkpoint path; model.predict is never called
  S6NN.relocated_nn_scored       a complete NN whose folder was moved (written path gone, file found in
                                 models_dir) is scored WITH its scaler, exactly as the original
                                 (Beta X-1B-STEP6-NN R1 Q2)
  S6NN.refuses_at_scoring        a loaded NN whose scaler is removed or the wrong width afterwards is
                                 refused when the pool is built; model.predict is never called
  S6NN.loader_reasons            models.inference_bundle.load_nn_scaler still returns (does not raise) and
                                 names the reason in every no-scaler exit; a complete checkpoint: reason None
  S6NN.every_reason_refuses      each loader reason, and an unknown one, refuses with its own words
  S6NN.trees_untouched           a LightGBM and an XGBoost sidecar: the NN check is never called and
                                 model.predict receives the raw matrix
  S6NN.cli_exit_codes            Step 6's CLI: broken NN -> exit 4, "STEP 6 REFUSED", no prediction file, and
                                 an EARLIER run's prediction file is moved to predictions/history/ unchanged
                                 (Beta R1 Q5); complete NN and LightGBM -> exit 0 with a prediction file
Mutants (each must be KILLED: the text change was applied exactly once, the changed line ran where it can
be counted, and the credited gate's own assertion failed):
  M1 load-time check does nothing          -> S6NN.refuses_at_load
  M2 refusal is a FileNotFoundError        -> S6NN.refuses_at_load   (the constructor would swallow it)
  M3 scoring-time width check off          -> S6NN.refuses_at_scoring
  M4 scoring-time no-scaler check off      -> S6NN.refuses_at_scoring
  M5 y_mean/y_std presence check off       -> S6NN.refuses_at_load
  M6 CLI returns 0 on a refusal            -> S6NN.cli_exit_codes
  M7 earlier prediction file left in place -> S6NN.cli_exit_codes
  M8 scaler read from the written path     -> S6NN.relocated_nn_scored
  M9 damaged-scale check off               -> S6NN.refuses_at_load
"""
import argparse
import importlib.util
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CPU only, set before anything imports torch
sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))


class GateFail(AssertionError):
    pass


def check(cond, msg):
    if not cond:
        raise GateFail(msg)


def load_pg(name, path):
    """Load a prediction_generator.py file under its own module name, CPU only."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    mod.GPU_AVAILABLE = False     # cupy can import with no visible device; force the CPU model load
    # the module puts its own directory first on sys.path; a mutant copy's directory must not stay there
    parent = str(Path(path).resolve().parent)
    if parent != str(REPO):
        while parent in sys.path:
            sys.path.remove(parent)
    return mod


def quiet_logger(tag, level=logging.DEBUG):
    buf = io.StringIO()
    lg = logging.getLogger(f"s6nn_{tag}_{time.time_ns()}")
    lg.handlers = [logging.StreamHandler(buf)]
    lg.setLevel(level)
    lg.propagate = False
    return lg, buf


def make_gen(pg, models_dir):
    cfg = pg.PredictionConfig()
    cfg.models_dir = str(models_dir)
    cfg.save_predictions = False
    lg, buf = quiet_logger(Path(models_dir).name)
    return pg.PredictionGenerator(cfg, lg), buf


def spy_predict(gen):
    cap = {"n": 0, "X": None}
    orig = gen.model.predict

    def spy(X):
        import numpy as np
        cap["n"] += 1
        cap["X"] = np.array(X, copy=True)
        return orig(X)
    gen.model.predict = spy
    return cap


def raw_matrix(gen, rows):
    import numpy as np
    return np.array([[float(r["features"].get(n, 0.0)) for n in gen.feature_names] for r in rows],
                    dtype=np.float32)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def build(out):
    import numpy as np
    import torch
    import test_e2e_vote_harness as E2E
    F = E2E.build_fixture(out / "e2e")
    models = dict(F["models"])
    base = models["nn"]
    ck0 = torch.load(base / "best_model.pth", map_location="cpu", weights_only=False)
    sidecar = json.loads((base / "best_model.meta.json").read_text())

    def variant(name, edit=None, written_path=None):
        d = F["fx"] / f"models_{name}"
        d.mkdir()
        ck = dict(ck0)
        if edit:
            edit(ck)
        torch.save(ck, d / "best_model.pth")
        sc = dict(sidecar, checkpoint_path=str(written_path or d / "best_model.pth"))
        (d / "best_model.meta.json").write_text(json.dumps(sc, indent=2))
        models[name] = d
        return d

    def no_arrays(ck):
        ck.pop("scaler_mean"); ck.pop("scaler_scale")

    def short(ck):
        ck["scaler_mean"] = np.asarray(ck["scaler_mean"])[:-1].copy()
        ck["scaler_scale"] = np.asarray(ck["scaler_scale"])[:-1].copy()

    def nan(ck):
        m = np.asarray(ck["scaler_mean"]).copy(); m[0] = np.nan
        ck["scaler_mean"] = m

    def ybad(ck):
        ck["y_std"] = float("nan")

    def no_ymean(ck):
        ck.pop("y_mean")

    def no_ystd(ck):
        ck.pop("y_std")

    def neg_scale(ck):
        s = np.asarray(ck["scaler_scale"]).copy(); s[0] = -np.inf
        ck["scaler_scale"] = s

    def tiny_scale(ck):
        s = np.asarray(ck["scaler_scale"]).copy(); s[1] = 1e-13
        ck["scaler_scale"] = s

    variant("nn_arrays_missing", no_arrays)
    variant("nn_width", short)
    variant("nn_nonfinite", nan)
    variant("nn_ybad", ybad)
    variant("nn_no_ymean", no_ymean)
    variant("nn_no_ystd", no_ystd)
    variant("nn_neg_scale", neg_scale)
    variant("nn_tiny_scale", tiny_scale)
    # relocated: the sidecar's written path does not exist; the model file is found in models_dir
    variant("nn_relocated", None, written_path=F["fx"] / "moved_away" / "best_model.pth")
    F["models"] = models
    F["ck0"] = ck0
    return F


# (fixture name, words the refusal must contain)
BROKEN = [
    ("nn_noscaler", "trained WITHOUT input normalisation"),
    ("nn_arrays_missing", "holds no scaler_mean/scaler_scale"),
    ("nn_width", "but the sidecar names"),
    ("nn_nonfinite", "not finite"),
    ("nn_noy", "no y_mean/y_std"),
    ("nn_ybad", "y_mean/y_std are not usable"),
    ("nn_no_ymean", "no y_mean/y_std pair (missing: y_mean)"),
    ("nn_no_ystd", "no y_mean/y_std pair (missing: y_std)"),
    ("nn_neg_scale", "value(s) below 1e-12"),
    ("nn_tiny_scale", "value(s) below 1e-12"),
]


# ---------------------------------------------------------------------------
# gates (each takes the prediction_generator module under test)
# ---------------------------------------------------------------------------
def g_complete(pg, F, out):
    import numpy as np
    gen, buf = make_gen(pg, F["models"]["nn"])
    cap = spy_predict(gen)
    rows = F["rows"]
    res = gen._build_prediction_pool(rows, F["combined"], 10)
    raw = raw_matrix(gen, rows)
    want = (raw - gen._scaler_mean) / gen._scaler_scale
    check(cap["n"] == 1, f"model.predict called {cap['n']} times")
    check(np.array_equal(cap["X"], want), "model.predict did not receive (X - mean) / scale")
    check(not np.array_equal(cap["X"], raw), "vacuous: the scaled matrix equals the raw one")
    check("[1b] NN normalisation present" in buf.getvalue(), "no [1b] acceptance line in the log")
    check(len(res["predictions"]) > 0, "no predictions from the complete NN")
    return f"{raw.shape[0]}x{raw.shape[1]} rows scaled; {len(res['predictions'])} predictions"


def g_refuse_load(pg, F, out):
    seen = {}
    for name, needle in BROKEN:
        d = F["models"][name]
        calls = {"n": 0}
        import models.wrappers.neural_net_wrapper as nnw
        orig = nnw.NeuralNetWrapper.predict

        def counted(self, X, _o=orig, _c=calls):
            _c["n"] += 1
            return _o(self, X)
        nnw.NeuralNetWrapper.predict = counted
        try:
            try:
                gen, _ = make_gen(pg, d)
            except pg.NNNormalisationError as e:
                msg = str(e)
                check(not isinstance(e, FileNotFoundError), f"{name}: the refusal is a FileNotFoundError")
                check(needle in msg, f"{name}: refusal lacks {needle!r}: {msg[:200]}")
                written = json.loads((d / "best_model.meta.json").read_text())["checkpoint_path"]
                check(written in msg, f"{name}: refusal does not name the checkpoint path")
                check(calls["n"] == 0, f"{name}: model.predict ran {calls['n']} times before the refusal")
                seen[name] = needle
                continue
            except Exception as e:
                raise GateFail(f"{name}: wrong error {type(e).__name__}: {str(e)[:160]}")
            raise GateFail(f"{name}: accepted (no refusal); model loaded: {gen.model is not None}")
        finally:
            nnw.NeuralNetWrapper.predict = orig
    return f"{len(seen)}/{len(BROKEN)} refused at load: {sorted(seen)}"


def g_relocated(pg, F, out):
    import numpy as np
    rows = F["rows"]
    res = {}
    for name in ("nn", "nn_relocated"):
        try:
            gen, _ = make_gen(pg, F["models"][name])
        except pg.NNNormalisationError as e:
            raise GateFail(f"{name}: refused: {str(e).splitlines()[0]}")
        cap = spy_predict(gen)
        pool = gen._build_prediction_pool(rows, F["combined"], 10)
        res[name] = (gen, cap, pool)
    g0, c0, p0 = res["nn"]
    g1, c1, p1 = res["nn_relocated"]
    written = json.loads((F["models"]["nn_relocated"] / "best_model.meta.json").read_text())["checkpoint_path"]
    check(not Path(written).exists(), "vacuous: the relocated sidecar's written path exists")
    check(g1.model_checkpoint_path == written
          and g1.model_checkpoint_resolved == str(F["models"]["nn_relocated"] / "best_model.pth"),
          f"relocated: written {g1.model_checkpoint_path}, read {g1.model_checkpoint_resolved}")
    check(g1._normalize_features is True and np.array_equal(g1._scaler_mean, g0._scaler_mean)
          and np.array_equal(g1._scaler_scale, g0._scaler_scale), "relocated: scaler differs from the original's")
    check(c1["n"] == 1 and np.array_equal(c1["X"], c0["X"]), "relocated: model.predict input differs from the original's")
    check(not np.array_equal(c1["X"], raw_matrix(g1, rows)), "vacuous: the relocated input is the raw matrix")
    check(p1["predictions"] == p0["predictions"] and p1["confidence_scores"] == p0["confidence_scores"],
          "relocated: pool or scores differ from the original's")
    return f"relocated NN scored with its scaler; pool identical to the original ({len(p1['predictions'])} predictions)"


def g_refuse_scoring(pg, F, out):
    done = []
    for label, needle in (("width", "columns but the input scaler has"),
                          ("no_scaler", "not loaded at scoring time")):
        gen, _ = make_gen(pg, F["models"]["nn"])
        cap = spy_predict(gen)
        if label == "width":
            gen._scaler_mean = gen._scaler_mean[:-1]
            gen._scaler_scale = gen._scaler_scale[:-1]
        else:
            gen._normalize_features = False
        try:
            gen._build_prediction_pool(F["rows"], F["combined"], 10)
        except pg.NNNormalisationError as e:
            check(needle in str(e), f"{label}: refusal lacks {needle!r}: {str(e)[:200]}")
            check(cap["n"] == 0, f"{label}: model.predict ran before the refusal")
            done.append(label)
            continue
        except Exception as e:
            raise GateFail(f"{label}: wrong error {type(e).__name__}: {str(e)[:160]}")
        raise GateFail(f"{label}: scored without refusal (model.predict calls: {cap['n']})")
    return f"refused at scoring: {done}"


def g_loader_reasons(pg, F, out):
    import numpy as np
    import torch
    import models.inference_bundle as ib
    lg, _ = quiet_logger("loader")
    d = out / "loader"
    d.mkdir(exist_ok=True)
    (d / "garbage.pth").write_bytes(b"this is not a checkpoint")
    torch.save([1, 2, 3], d / "notdict.pth")
    cases = {"checkpoint_missing": d / "absent.pth", "checkpoint_unreadable": d / "garbage.pth",
             "checkpoint_not_dict": d / "notdict.pth",
             "normalize_flag_off": F["models"]["nn_noscaler"] / "best_model.pth",
             "scaler_arrays_missing": F["models"]["nn_arrays_missing"] / "best_model.pth"}
    for reason, path in cases.items():
        sc = ib.load_nn_scaler(str(path), lg)     # must return, not raise
        check(sc["reason"] == reason, f"{path.name}: reason {sc['reason']!r}, expected {reason!r}")
        check(sc["normalize_features"] is False and sc["scaler_mean"] is None and sc["scaler_scale"] is None,
              f"{reason}: a no-scaler exit returned scaler state")
        check(reason in ib.SCALER_ABSENT_REASONS, f"{reason} has no message")
    sc = ib.load_nn_scaler(str(F["models"]["nn"] / "best_model.pth"), lg)
    check(sc["reason"] is None and sc["normalize_features"] is True, f"complete checkpoint: {sc['reason']!r}")
    check(np.array_equal(sc["scaler_mean"], np.asarray(F["ck0"]["scaler_mean"], dtype=np.float32)),
          "complete checkpoint: scaler_mean differs from the checkpoint's")
    return f"{len(cases)} no-scaler reasons named; complete checkpoint reason None"


def g_every_reason(pg, F, out):
    import models.inference_bundle as ib
    lg, _ = quiet_logger("reasons")
    texts = set()
    for reason in list(ib.SCALER_ABSENT_REASONS) + ["something_new"]:
        class Stub:
            pass
        s = Stub()
        s.logger, s.model_meta, s.model_checkpoint_path = lg, {"model_type": "neural_net"}, "/x/best_model.pth"
        s.config = pg.PredictionConfig()
        s._scaler_mean = s._scaler_scale = None
        s._normalize_features = False
        s._scaler_absent_reason = reason
        s.feature_names, s.model = ["a"], None
        s._nn_refusal = lambda what, _s=s: pg.PredictionGenerator._nn_refusal(_s, what)
        try:
            pg.PredictionGenerator._require_nn_normalisation(s)
        except pg.NNNormalisationError as e:
            first = str(e).splitlines()[0]
            want = ib.SCALER_ABSENT_REASONS.get(reason, reason)
            check(want in first, f"{reason}: message lacks its own words: {first}")
            texts.add(first)
            continue
        raise GateFail(f"{reason}: not refused")
    check(len(texts) == len(ib.SCALER_ABSENT_REASONS) + 1, "two reasons share one message")
    return f"{len(texts)} reasons, {len(texts)} distinct refusals"


def g_trees(pg, F, out):
    import numpy as np
    calls = {"n": 0}
    orig = pg.PredictionGenerator._require_nn_normalisation

    def counted(self):
        calls["n"] += 1
        return orig(self)
    pg.PredictionGenerator._require_nn_normalisation = counted
    try:
        outp = {}
        for fam in ("lightgbm", "xgboost"):
            gen, _ = make_gen(pg, F["models"][fam])
            cap = spy_predict(gen)
            res = gen._build_prediction_pool(F["rows"], F["combined"], 10)
            check(cap["n"] == 1 and np.array_equal(cap["X"], raw_matrix(gen, F["rows"])),
                  f"{fam}: model.predict did not receive the raw matrix")
            check(len(res["predictions"]) > 0, f"{fam}: no predictions")
            outp[fam] = len(res["predictions"])
        check(calls["n"] == 0, f"the NN check ran {calls['n']} times for tree models")
    finally:
        pg.PredictionGenerator._require_nn_normalisation = orig
    return f"NN check calls 0; predictions {outp}"


STALE = b'{"predictions": [111, 222, 333], "note": "an EARLIER run, written by the 1b self-test"}\n'


def run_cli(pg, F, out, tag, models_dir, stale=False):
    cwd = out / "cli" / f"{tag}_{time.time_ns()}"
    cwd.mkdir(parents=True)
    if stale:
        (cwd / "predictions").mkdir()
        (cwd / "predictions" / "next_draw_prediction.json").write_bytes(STALE)
    res = cwd / "result.json"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-B", str(REPO / "tests/ll6_step6_runner.py"),
                        "--pg-file", str(pg.__file__), "--repo-root", str(REPO),
                        "--models-dir", str(models_dir),
                        "--survivors", str(F["fx"] / "survivors_with_scores.json"),
                        "--history", str(F["fx"] / "combined_history.json"),
                        "--cwd", str(cwd), "--out", str(res)],
                       env=env, capture_output=True, text=True, timeout=600)
    check(res.exists(), f"{tag}: the CLI runner wrote no result (rc {p.returncode}): {p.stderr[-400:]}")
    r = json.loads(res.read_text())
    r["stderr"] = p.stderr
    r["pred_file"] = (cwd / "predictions" / "next_draw_prediction.json").exists()
    hist = cwd / "predictions" / "history"
    r["retired"] = sorted(hist.glob("retired_after_refusal_*")) if hist.exists() else []
    return r


def g_cli(pg, F, out):
    seen = {}
    for name, needle in (("nn_noscaler", "trained WITHOUT input normalisation"), ("nn_noy", "no y_mean/y_std")):
        r = run_cli(pg, F, out, name, F["models"][name])
        log = "\n".join(r["log_lines"]) + r["stderr"]
        check(r["rc"] == 4, f"{name}: CLI exit {r['rc']}, expected 4")
        check(not r["pred_file"] and r["result"] is None, f"{name}: a prediction file was written")
        check("STEP 6 REFUSED" in log and needle in log, f"{name}: the refusal is not in the CLI output")
        check(r["retired"] == [], f"{name}: something was retired though no earlier file existed")
        seen[name] = "exit 4, no file"
    # an earlier run's canonical file must not survive a refusal as the canonical output (Beta R1 Q5:
    # WATCHER's freshness check goes by file times, agents/watcher_agent.py check_output_freshness)
    r = run_cli(pg, F, out, "nn_noscaler_stale", F["models"]["nn_noscaler"], stale=True)
    check(r["rc"] == 4, f"stale: CLI exit {r['rc']}, expected 4")
    check(not r["pred_file"], "stale: the earlier prediction file is still the canonical output after a refusal")
    check(len(r["retired"]) == 1 and r["retired"][0].read_bytes() == STALE,
          f"stale: the earlier file was not kept unchanged in predictions/history/: {r['retired']}")
    seen["earlier_file"] = "moved to history unchanged; canonical path empty"
    for name in ("nn", "lightgbm"):
        r = run_cli(pg, F, out, name, F["models"][name], stale=True)
        check(r["rc"] == 0 and r["pred_file"] and r["result"] and len(r["result"].get("predictions", [])) > 0,
              f"{name}: CLI exit {r['rc']}, prediction file {r['pred_file']}")
        check(r["retired"] == [] and r["result"].get("predictions") != [111, 222, 333],
              f"{name}: a successful run retired a file or left the earlier one")
        seen[name] = f"exit 0, {len(r['result']['predictions'])} predictions"
    return str(seen)


GATES = [
    ("S6NN.complete_nn_scaled", g_complete),
    ("S6NN.refuses_at_load", g_refuse_load),
    ("S6NN.relocated_nn_scored", g_relocated),
    ("S6NN.refuses_at_scoring", g_refuse_scoring),
    ("S6NN.loader_reasons", g_loader_reasons),
    ("S6NN.every_reason_refuses", g_every_reason),
    ("S6NN.trees_untouched", g_trees),
    ("S6NN.cli_exit_codes", g_cli),
]

HIT = "_s6nn_mut_hit()"
# name, [(old, new, count)], counted (the changed line can be counted), credited gate
MUTANTS = [
    ("M1 load-time check does nothing",
     [("        import math\n        from models.inference_bundle import SCALER_ABSENT_REASONS\n",
       f"        {HIT}\n        return\n", 1)], True, "S6NN.refuses_at_load"),
    ("M2 refusal is a FileNotFoundError",
     [("class NNNormalisationError(RuntimeError):", "class NNNormalisationError(FileNotFoundError):", 1),
      ('                self.logger.warning(f"  Model not found: {e}")',
       f'                {HIT}; self.logger.warning(f"  Model not found: {{e}}")', 1)], True, "S6NN.refuses_at_load"),
    ("M3 scoring-time width check off",
     [("            if X.shape[1] != len(self._scaler_mean):", f"            if {HIT} and False:", 1)],
     True, "S6NN.refuses_at_scoring"),
    ("M4 scoring-time no-scaler check off",
     [("            if (not getattr(self, '_normalize_features', False)\n"
       "                    or self._scaler_mean is None or self._scaler_scale is None):",
       f"            if {HIT} and False:", 1)], True, "S6NN.refuses_at_scoring"),
    ("M5 y_mean/y_std presence check off",
     [("        if ym is None or ys is None:", f"        if {HIT} and False:", 1)], True, "S6NN.refuses_at_load"),
    ("M6 CLI returns 0 on a refusal",
     [("        return EXIT_NN_NORMALISATION_REFUSED\n", "        return 0\n", 1)], False, "S6NN.cli_exit_codes"),
    ("M7 earlier prediction file left in place",
     [("        _retire_canonical_prediction(logger)\n", "        pass\n", 1)], False, "S6NN.cli_exit_codes"),
    ("M8 scaler read from the written path",
     [("                        self._load_nn_scaler(self.model_checkpoint_resolved)",
       f"                        {HIT}; self._load_nn_scaler(self.model_checkpoint_path)", 1)],
     True, "S6NN.relocated_nn_scored"),
    ("M9 damaged-scale check off",
     [("        if n_clamped:\n", f"        if {HIT} and False:\n", 1)], True, "S6NN.refuses_at_load"),
]
HIT_DEF = "\n\n_S6NN_MUT = {'n': 0}\n\n\ndef _s6nn_mut_hit():\n    _S6NN_MUT['n'] += 1\n    return True\n"


def run_mutant(i, m, F, out):
    name, edits, counted, credited = m
    src = (REPO / "prediction_generator.py").read_text()
    for old, new, count in edits:
        if src.count(old) != count:
            return name, False, f"NOT APPLIED: {src.count(old)} matches for {old[:50]!r}"
        src = src.replace(old, new)
    # the hit counter is defined before `class PredictionGenerator` so it exists when the class body runs
    marker = "class NNNormalisationError("
    src = src.replace(marker, HIT_DEF.strip("\n") + "\n\n\n" + marker, 1)
    d = out / f"mutant_{i}"
    d.mkdir()
    (d / "prediction_generator.py").write_text(src)
    mod = load_pg(f"s6nn_mutant_{i}", d / "prediction_generator.py")
    gate = dict(GATES)[credited]
    try:
        gate(mod, F, out)
    except GateFail as e:
        hits = mod._S6NN_MUT["n"]
        if counted and hits == 0:
            return name, False, f"gate failed but the changed line never ran: {str(e)[:120]}"
        return name, True, f"hits={hits if counted else 'n/a (subprocess)'} :: {str(e)[:120]}"
    except Exception as e:
        return name, False, f"uncredited {type(e).__name__}: {str(e)[:140]}"
    return name, False, "gate passed under the mutant (SURVIVED)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--no-mutants", action="store_true")
    a = ap.parse_args()
    out = Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix="s6nn_1b_"))
    if out.exists() and any(out.iterdir()):
        print(f"REFUSED: --out {out} is not empty")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[1b] out: {out}   repo: {REPO}   CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}",
          flush=True)
    F = build(out)
    print(f"[1b] fixtures built in {time.time() - t0:.1f}s: {sorted(F['models'])}", flush=True)
    pg = load_pg("s6nn_pg", REPO / "prediction_generator.py")
    results = []
    for name, fn in GATES:
        if a.only and name not in a.only:
            continue
        t = time.time()
        try:
            detail = fn(pg, F, out)
            results.append((name, True))
            print(f"PASS {name} ({time.time() - t:.1f}s): {detail}", flush=True)
        except GateFail as e:
            results.append((name, False))
            print(f"FAIL {name}: {e}", flush=True)
        except Exception as e:
            results.append((name, False))
            print(f"FAIL {name}: uncaught {type(e).__name__}: {e}", flush=True)
    mres = []
    if not a.no_mutants and not a.only:
        for i, m in enumerate(MUTANTS, 1):
            r = run_mutant(i, m, F, out)
            mres.append(r)
            print(f"{'KILLED' if r[1] else 'SURVIVED'} {r[0]}: {r[2]}", flush=True)
    gp, gt = sum(1 for _, ok in results if ok), len(results)
    mk, mt = sum(1 for _, k, _ in mres if k), len(mres)
    full = not a.only and not a.no_mutants
    ok = gp == gt and mk == mt and gt > 0
    print(f"[1b] gates {gp}/{gt}   mutants killed {mk}/{mt}   {time.time() - t0:.1f}s", flush=True)
    if ok and full:
        print(f"COMPLETION SENTINEL: PASS ({gp}/{gt} gates, {mk}/{mt} mutants killed)", flush=True)
        return 0
    print("COMPLETION SENTINEL: " + ("INCOMPLETE (partial run)" if ok else "FAIL"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
