#!/usr/bin/env python3
"""
attribution_cycle.py - [LL-8, Ch14 learning loop, 2026-11-23; landing changes 2026-11-26] OBSERVE-ONLY.

The first "what to extract" report from real predictions. It attributes EVERY
ranked survivor with the Step 5 model (signed, by name, LL-7, through the same
loader as Step 6, LL-6), then asks two questions:

  1. What does the ranking lean on?  Per DISJOINT rank band (1-20, 21-100,
     101-300, 301-n), the share of |attribution| carried by each feature and by
     each manifest group (models/feature_manifest.json), split by namespace.
     Per-run columns can only move the whole run, never one survivor against
     another, so they are reported as CONTEXT, never as seed evidence.
  2. Did what it leaned on separate the survivors that voted the drawn number
     from those that did not?  The same cohorts, bands, selection split,
     bootstrap and abstention rules as the miss report
     (scripts/diagnostics/e2e_miss_report.py of this tree, pinned by
     sha256, its functions CALLED, not copied), applied to the standardised
     signed-attribution matrix instead of the raw feature matrix.

OUTPUT: attribution_cycle.json (everything), extraction_priorities.json
(advisory: ranked groups with intervals, recurring contrasts, and the declared
LL-3 Step 5 variants that would test them), run_record.json.

It writes only under --out-dir, trains nothing, changes no setting, grades
nothing, and never opens the dataset: outcomes come only from grader
per_draw.jsonl files (sha256 checked against their SHA256SUMS). Every number
is DEVELOPMENT evidence on looked-at blocks. It is a POINTER: the measured
answer to "does dropping or adding group X change pool hits?" is a graded
setting comparison (LL-3 variants on the walk-forward, LL-4), not this report.

Inputs, all in one declared --spec JSON (sha256 recorded):
  survivors{path,sha256}  Step 3 rows (survivors_with_scores.json)
  models_dir, expect_checkpoint_sha256 (REQUIRED, owner decision D3 2026-11-26)
  scaler_source           optional: "resolved" (default) | "as_written" (see G-Q4-SCALER)
  votes{path,sha256}      .npz: V (n_surv, n_draws) int, seeds (n_surv,), combined_index (n_draws,)
                          V[i, j] = survivor i's vote on draw combined_index[j]; the miss
                          report's full_vote() produces exactly this matrix.
  grade_dirs              grader output dirs holding per_draw.jsonl + SHA256SUMS
  blocks {name: [b0, b1]}, sessions, pool_sizes, per_run_prefixes,
  abstain {min_pairs, min_draws, min_survivors}, band_min_survivors,
  bootstrap {B}, seed, attribution {method, ig_steps}

LANDING CHANGES (plan PLAN_LAND_LL7_LL8_20261126 §5.2; branch ch14-learn blob f4a8adbd):
  - the miss-report default is this tree's copy (the branch pointed into another worktree);
  - the checkpoint pin is required; run_record.json records the bundle's checkpoint path and
    feature_schema_hash and the sidecar's WRITTEN checkpoint path;
  - G-Q4-SCALER (Beta X-LL6-BUNDLE R1 Q4) and the Step 6 evidence rule (Beta X-LL7-LL8 R1 Q3).
    Whether Step 6 scaled an NN's input depends on whether the sidecar's WRITTEN checkpoint_path
    existed WHEN STEP 6 SCORED (models/inference_bundle.py load_nn_scaler, best effort). Its state
    at report time does not show that: a model can be moved after scoring. So:
      * the spec may name step6_evidence: saved Step 6 scores {path, sha256[, atol, rtol]} (.npy, or
        .npz with key "scores" and optionally "seeds" in survivor order) and/or scores_sha256
        (the vote harness's RankOnce.scores_sha256: sha256 of dtype name + raw score bytes, only
        meaningful with no tie-break), and optionally scaler_applied (the vote harness's
        ranking_info.nn_scaler_applied). When given, this run's own forward is checked against
        every item and any mismatch is REFUSED, whichever scaler_source is set;
      * scaler_source "as_written" without step6_evidence is REFUSED: nothing shows that Step 6
        scored without its scaler;
      * without step6_evidence, a neural_net whose written checkpoint_path does not exist is
        REFUSED (G-Q4-SCALER); when it exists the run goes ahead and the report says the Step 6
        match was NOT checked (step6_match.checked false).
    A-NORM here checks the attribution forward against this script's own forward (bundle.predict);
    the comparison with what Step 6 actually scored is step6_match.
  - a function entry point for a later runner (unit W, not built here):
        run_attribution_cycle(spec, out_dir, model_family=..., flag_id=...) -> result dict
    and the CLI flags --model-family, --flag-id, --result-file. Both write a machine-readable
    result file (RESULT_FORMAT): status ok | refused | error, exit_code 0 | 2 | 3, the error
    class and message, the output paths and sha256s. flag_id is the why-report flag id
    "<model_type>@<anchor>" (training_health_check.why_flag_id); its family must equal the
    loaded model's family. The flag file itself is not read or written here.
    The result file is a new path distinct from every cycle output (never one of OUTPUT_NAMES in
    --out-dir, never inside <out-dir>/_env, never --out-dir or one of its parents); it is created
    exclusively and never overwrites a file. A failure resolving paths or publishing the result
    returns status error, exit 3, result_written null, with the reason on stderr (Beta X-LL7-LL8
    R1 Q5).
  - caller state (Beta X-LL7-LL8 R1 Q7): importing this module changes nothing. While a cycle runs,
    CUDA_VISIBLE_DEVICES is "" (CPU only; the pinned miss report also clears it at its own module
    top) and the temp and cache roots point under <out>/_env, whose directories are all created
    BEFORE any variable changes. On success, refusal and error the caller's exact values (absent
    included), tempfile.tempdir, sys.path, the root logger's handlers and level, and (when torch
    was already imported) torch's CPU RNG state are restored. A later runner should still call it
    in a child process (plan §§6.3, 6.7): library state initialised during inference is not undone.
  - score-scaled tolerances (LL-8 fix2, 2026-11-26). A-NORM and the Step 6 saved-scores check
    compare two float32 forwards of the same model on the same input. Their gap is rounding, and
    rounding grows with the size of the numbers: an NN fed unscaled inputs (as_written, inputs up
    to ~2.8e4) scored up to ~38.5 and the two forwards differed by 1.144e-5, which a fixed
    absolute 1e-5 refused. Measured (data/ll8_anorm_measure, not tracked): the gap is the same in
    every process and at 1, 2 and 5 threads (90 runs, zero spread); it differs between trained
    models (1.07e-6 .. 1.54e-6 for models scoring 0.44 .. 2.25; 1.144e-5 for the one scoring
    38.5), because a batch of 1..N rows and the full batch take different float32 summation
    orders. So the allowed gap is
        allowed = atol + rtol * max|reference scores|
    (reference: this script's own forward for A-NORM, the saved Step 6 scores for step6_evidence),
    one scale for all rows because the rounding follows the network's internal magnitudes, not a
    row's own score. Spec keys a_norm_atol / a_norm_rtol and step6_evidence.scores.atol / .rtol;
    defaults below. The report records the measured gap, atol, rtol, the scale and the allowed gap.
    A real mismatch (wrong scaler, wrong model) differs by the size of the scores themselves
    (38.4 in the as_written fixture), orders of magnitude above rtol = 1e-5 of them.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
FORMAT = "tfm.ll8.attribution_cycle.v1"
MISS_REPORT_DEFAULT = REPO_ROOT / "scripts/diagnostics/e2e_miss_report.py"
RESULT_FORMAT = "tfm.ll8.attribution_cycle.result.v1"
SCALER_SOURCES = ("resolved", "as_written")
OUTPUT_NAMES = ("attribution_cycle.json", "extraction_priorities.json", "signed_attribution.npz", "run_record.json")
Q3_REF = "Beta X-LL7-LL8 R1 Q3"
EXIT_OK, EXIT_REFUSED, EXIT_ERROR = 0, 2, 3
Q4_REF = "Beta X-LL6-BUNDLE R1 Q4 (docs/exchange/from_beta/X-LL6-BUNDLE_R1_BETA_REVIEW.md)"
MISS_REPORT_SHA256 = "ec7c7ad16b25362e1800258c7a3667e1f8150b73739e08b43b5522f028116d76"
ADVISORY = ("ADVISORY / DEVELOPMENT. A pointer, not a measurement of hit rate. The measured "
            "answer is a graded comparison of declared Step 5 variants (LL-3 flags) on the "
            "walk-forward, at equal k, against k/1000 and the unlearned arm.")
# Score-scaled tolerances (see the docstring). rtol 1e-5 is ~84 float32 epsilons: 34x the worst
# measured gap (1.144e-5 at scale 38.5 -> allowed 3.95e-4) and >= 9x every other measured gap,
# while a real mismatch differs by O(scale).
A_NORM_ATOL_DEFAULT = 1e-5
A_NORM_RTOL_DEFAULT = 1e-5
STEP6_ATOL_DEFAULT = 1e-5
STEP6_RTOL_DEFAULT = 1e-5


class CycleError(RuntimeError):
    pass


def allowed_gap(atol, rtol, reference, what):
    """atol + rtol * max|reference|, the largest gap two forwards of one model may show. Returns
    (allowed, scale). Refuses a negative or non-finite atol/rtol and non-finite reference scores."""
    import numpy as np
    try:
        atol, rtol = float(atol), float(rtol)
    except (TypeError, ValueError):
        raise CycleError(f"{what}: atol {atol!r} and rtol {rtol!r} must be numbers")
    for k, v in (("atol", atol), ("rtol", rtol)):
        if not (math.isfinite(v) and v >= 0):
            raise CycleError(f"{what}: {k} must be a finite number >= 0, got {v}")
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    scale = float(np.max(np.abs(ref))) if ref.size else 0.0
    if not math.isfinite(scale):
        raise CycleError(f"{what}: the reference scores are not all finite")
    return atol + rtol * scale, scale


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_miss_report(path, want_sha):
    got = sha256_file(path)
    if got != want_sha:
        raise CycleError(f"miss report {path} sha256 {got} != pinned {want_sha}")
    spec = importlib.util.spec_from_file_location("e2e_miss_report_pinned", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def scores_sha256(scores):
    """The vote harness's RankOnce.scores_sha256 (e2e_vote_harness.py:391-397): dtype name + bytes."""
    import numpy as np
    s = np.ascontiguousarray(np.asarray(scores))
    h = hashlib.sha256()
    h.update(str(s.dtype).encode())
    h.update(s.tobytes())
    return h.hexdigest()


def check_step6_evidence(ev, scores, seeds, scaler_applied):
    """Check this run's forward against saved Step 6 evidence (Beta X-LL7-LL8 R1 Q3). Returns the
    step6_match record; raises CycleError on any mismatch or malformed evidence."""
    import numpy as np
    if not isinstance(ev, dict) or not ("scores" in ev or "scores_sha256" in ev):
        raise CycleError("step6_evidence must name saved Step 6 scores {path, sha256} and/or scores_sha256")
    rec = {"checked": True, "source": ev.get("source"), "items": []}
    if "scaler_applied" in ev:
        want = ev["scaler_applied"]
        if not isinstance(want, bool):
            raise CycleError(f"step6_evidence.scaler_applied must be true or false, got {want!r}")
        if want != scaler_applied:
            raise CycleError(f"Step 6 evidence: Step 6 scaler_applied={want} but this run's transform applies "
                             f"the scaler: {scaler_applied}; the report would explain a ranking Step 6 did not "
                             f"make ({Q3_REF})")
        rec["items"].append({"scaler_applied": want})
    if "scores_sha256" in ev:
        got = scores_sha256(scores)
        if got != ev["scores_sha256"]:
            raise CycleError(f"Step 6 evidence: scores_sha256 of this run's forward {got} != saved "
                             f"{ev['scores_sha256']} ({Q3_REF})")
        rec["items"].append({"scores_sha256": got})
    if "scores" in ev:
        sv = ev["scores"]
        sp = sv["path"]
        if sha256_file(sp) != sv["sha256"]:
            raise CycleError(f"step6_evidence scores {sp}: sha256 differs from the spec")
        saved_seeds = None
        if str(sp).endswith(".npz"):
            with np.load(sp, allow_pickle=False) as z:
                saved = np.asarray(z["scores"], dtype=np.float64).reshape(-1)
                if "seeds" in z.files:
                    saved_seeds = z["seeds"].astype(np.int64).tolist()
        else:
            saved = np.asarray(np.load(sp, allow_pickle=False), dtype=np.float64).reshape(-1)
        if saved_seeds is not None and saved_seeds != list(seeds):
            raise CycleError("Step 6 evidence: saved scores' seeds are not the survivors' rows in order")
        if saved.shape != (len(scores),):
            raise CycleError(f"Step 6 evidence: {saved.shape[0]} saved scores for {len(scores)} rows")
        atol, rtol = sv.get("atol", STEP6_ATOL_DEFAULT), sv.get("rtol", STEP6_RTOL_DEFAULT)
        allowed, scale = allowed_gap(atol, rtol, saved, "step6_evidence.scores")
        gap = float(np.max(np.abs(saved - np.asarray(scores, dtype=np.float64)))) if len(scores) else 0.0
        if not gap <= allowed:
            raise CycleError(f"Step 6 evidence: this run's forward differs from the saved Step 6 scores by {gap} "
                             f"> allowed {allowed} (atol {float(atol)} + rtol {float(rtol)} x max|Step 6 score| "
                             f"{scale}); the report would explain a ranking Step 6 did not make ({Q3_REF})")
        rec["items"].append({"scores_path": str(sp), "scores_sha256_file": sv["sha256"], "max_abs_diff": gap,
                             "atol": float(atol), "rtol": float(rtol), "score_scale": scale,
                             "allowed_gap": allowed, "seeds_checked": saved_seeds is not None})
    return rec


# ---------------------------------------------------------------------------
# pure core (tests/test_ll_wave2.py)
# ---------------------------------------------------------------------------
def band_members(order, edges):
    """Disjoint bands of ranked row indices: band (lo, hi] = order[lo:hi]."""
    return [(f"{lo + 1}-{hi}", order[lo:hi]) for lo, hi in edges]


def band_shares(A, members, names, group_of, per_run, B, rng, min_survivors):
    """Per band: mean signed and mean |A| per feature; share of |A| per manifest group and
    per namespace; bootstrap over the band's survivors for the group shares."""
    import numpy as np
    per_run_set = set(per_run)
    groups = sorted({group_of.get(n, "unmanifested") for n in names})
    gidx = {g: [j for j, n in enumerate(names) if group_of.get(n, "unmanifested") == g] for g in groups}
    ns_idx = {"per_seed": [j for j, n in enumerate(names) if n not in per_run_set],
              "per_run_context": [j for j, n in enumerate(names) if n in per_run_set]}
    out = []
    for label, idx in members:
        idx = np.asarray(idx)
        row = {"band": label, "survivors": int(idx.size)}
        if idx.size < min_survivors:
            row["abstained"] = True
            row["abstain_reason"] = f"survivors {idx.size} < {min_survivors}"
            out.append(row)
            continue
        Ab = np.abs(A[idx])
        tot_row = Ab.sum(axis=1)
        tot_row[tot_row == 0] = 1.0
        share = Ab / tot_row[:, None]                        # per survivor, sums to 1
        boots = rng.integers(0, idx.size, size=(B, idx.size))

        def summarise(cols):
            v = share[:, cols].sum(axis=1)
            bs = v[boots].mean(axis=1)
            return {"share": float(v.mean()), "ci95": [float(np.percentile(bs, 2.5)),
                                                       float(np.percentile(bs, 97.5))]}
        row["abstained"] = False
        row["groups"] = {g: dict(summarise(c), namespace=("per_run_context"
                                 if all(names[j] in per_run_set for j in c) else
                                 "per_seed" if not any(names[j] in per_run_set for j in c)
                                 else "mixed")) for g, c in gidx.items()}
        row["namespaces"] = {k: summarise(c) if c else None for k, c in ns_idx.items()}
        mean_signed = A[idx].mean(axis=0)
        mean_abs = Ab.mean(axis=0)
        top = np.argsort(-mean_abs)[:15]
        row["top_features"] = [{"feature": names[j], "mean_abs": float(mean_abs[j]),
                                "mean_signed": float(mean_signed[j]),
                                "namespace": "per_run_context" if names[j] in per_run_set else "per_seed",
                                "group": group_of.get(names[j], "unmanifested")} for j in top]
        out.append(row)
    return out


def recurring(contrast_sets, key="C1b"):
    """Features whose interval excludes 0 with the SAME sign in >= 2 SEPARATE draw sets.
    Keys are '<block>/<session>/N=<n>'; pool sizes on the same draws are nested and
    dependent, so a draw set is '<block>/<session>' and counts once (any sign conflict
    across N or across draw sets disqualifies the feature)."""
    seen = {}
    for set_name, cc in contrast_sets.items():
        c = cc.get(key) or {}
        if c.get("abstained", True):
            continue
        draw_set = set_name.split("/N=")[0]
        for r in c.get("features", []):
            if r["ci_excludes_0"]:
                seen.setdefault(r["feature"], []).append((draw_set, set_name,
                                                          1 if r["point"] > 0 else -1))
    out = []
    for f, lst in seen.items():
        draw_sets = {d for d, _, _ in lst}
        signs = {sg for _, _, sg in lst}
        if len(draw_sets) >= 2 and len(signs) == 1:
            out.append({"feature": f, "sign": signs.pop(), "draw_sets": sorted(draw_sets),
                        "cells": sorted(n for _, n, _ in lst)})
    return sorted(out, key=lambda r: (-len(r["draw_sets"]), r["feature"]))


def priorities(bands, rec, group_of, per_run, band_label):
    """Advisory ranking: per-seed manifest groups by |attribution| share in the tightest band,
    recurring contrasts, and the declared Step 5 variants that would test each."""
    b = next((x for x in bands if x["band"] == band_label), None)
    ranked, context = [], []
    if b and not b.get("abstained"):
        for g, v in sorted(b["groups"].items(), key=lambda kv: -kv[1]["share"]):
            item = {"group": g, "share": v["share"], "ci95": v["ci95"], "namespace": v["namespace"],
                    "test_with": {"step5_variant": f"--exclude-feature-groups {g}",
                                  "compare_against": "the same arm without the flag, equal k, "
                                                     "graded on walk-forward blocks (LL-4)"}}
            (context if v["namespace"] == "per_run_context" else ranked).append(item)
    rec_groups = {}
    for r in rec:
        rec_groups.setdefault(group_of.get(r["feature"], "unmanifested"), []).append(r)
    return {"advisory": ADVISORY, "band": band_label,
            "abstained": (b is None or b.get("abstained", True)),
            "per_seed_groups_ranked": ranked,
            "per_run_context_groups": context,
            "per_run_rule": "per-run columns cannot separate survivors within one run; listed as "
                            "context only (system map v1.2 §8.2)",
            "recurring_contrasts": rec, "recurring_by_group": rec_groups}


# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="LL-8 observe-only attribution cycle")
    p.add_argument("--spec", required=True)
    p.add_argument("--out-dir", required=True, help="new directory under <worktree>/data/")
    p.add_argument("--miss-report", default=str(MISS_REPORT_DEFAULT))
    p.add_argument("--miss-report-sha256", default=MISS_REPORT_SHA256)
    p.add_argument("--model-family", default=None,
                   help="the model family this run is for (neural_net, xgboost, lightgbm, catboost); "
                        "must equal the loaded model's family")
    p.add_argument("--flag-id", default=None,
                   help="the why-report flag id this run is for, '<model_type>@<anchor>' "
                        "(training_health_check.why_flag_id); recorded, family checked")
    p.add_argument("--result-file", default=None,
                   help="where to write the result JSON (default <out-dir>/result.json); must lie under "
                        "<worktree>/data/ and not exist")
    return p.parse_args(argv)


def _under_data(p):
    return str(Path(p).resolve()).startswith(str(REPO_ROOT / "data") + "/")


def _result_target_problem(rf, out):
    """None when rf is a safe, distinct place for the result file; else why not (Beta X-LL7-LL8 R1 Q5)."""
    if not _under_data(rf):
        return "must lie under <worktree>/data/"
    if rf.exists() or rf.is_symlink():
        return "exists (the result never overwrites a file)"
    if rf == out or rf in out.parents:
        return "is --out-dir or one of its parents"
    if rf.parent == out and rf.name in OUTPUT_NAMES:
        return f"is a cycle output ({rf.name})"
    env = out / "_env"
    if rf == env or env in rf.parents:
        return "lies inside <out-dir>/_env"
    return None


def _publish(target, res):
    """Create the result file exclusively (never overwrites); a partial file is removed."""
    target.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(res, indent=2, default=str)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
    except BaseException:
        try:
            os.unlink(target)
        except OSError:
            pass
        raise


def check_flag(model_family, flag_id):
    """The family named for this run: from --model-family, from the flag id, or both (then equal)."""
    fam_flag = None
    if flag_id is not None:
        fid = str(flag_id)
        fam_flag, sep, anchor = fid.partition("@")
        if not sep or not fam_flag or not anchor or any(c.isspace() for c in fid):
            raise CycleError(f"flag_id {fid!r} is not '<model_type>@<anchor>' (training_health_check.why_flag_id)")
    if model_family is not None and fam_flag is not None and model_family != fam_flag:
        raise CycleError(f"--model-family {model_family!r} differs from the flag's family {fam_flag!r}")
    return model_family or fam_flag


class _CallerScope:
    """What the cycle changes in the calling process, undone on the way out (Beta X-LL7-LL8 R1 Q7).
    TMPDIR, XDG_CACHE_HOME, MPLCONFIGDIR and CUDA_CACHE_PATH under <out>/_env (Beta X-LL6-BUNDLE R1
    Q5: a LightGBM import made /tmp/matplotlib-*); CUDA_VISIBLE_DEVICES "" (CPU only). Every
    directory is created before any variable changes; restored: the env values (absent included),
    tempfile.tempdir, sys.path, the root logger's handlers and level, torch's CPU RNG state when
    torch was imported before the cycle."""
    KEYS = {"TMPDIR": "tmp", "XDG_CACHE_HOME": "cache", "MPLCONFIGDIR": "mpl", "CUDA_CACHE_PATH": "cuda_cache"}
    HIDE = {"CUDA_VISIBLE_DEVICES": ""}

    def __init__(self, out):
        self.out = out
        self.saved = None

    def __enter__(self):
        import logging
        import tempfile
        dirs = {k: self.out / "_env" / d for k, d in self.KEYS.items()}
        for d in dirs.values():                 # all directories first: a failure here has changed nothing
            d.mkdir(parents=True, exist_ok=True)
        root = logging.getLogger()
        torch = sys.modules.get("torch")
        self.saved = {"env": {k: os.environ.get(k) for k in list(dirs) + list(self.HIDE)},
                      "tempdir": tempfile.tempdir, "sys_path": list(sys.path),
                      "log_handlers": list(root.handlers), "log_level": root.level,
                      "torch_rng": torch.get_rng_state() if torch is not None else None}
        try:
            for k, d in dirs.items():
                os.environ[k] = str(d)
            for k, v in self.HIDE.items():
                os.environ[k] = v
            tempfile.tempdir = None
            sys.path.insert(0, str(REPO_ROOT))
        except BaseException:
            self._restore()
            raise
        return self

    def _restore(self):
        import logging
        import tempfile
        s = self.saved
        for k, v in s["env"].items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        tempfile.tempdir = s["tempdir"]
        sys.path[:] = s["sys_path"]
        root = logging.getLogger()
        root.handlers[:] = s["log_handlers"]
        root.setLevel(s["log_level"])
        if s["torch_rng"] is not None and "torch" in sys.modules:
            sys.modules["torch"].set_rng_state(s["torch_rng"])

    def __exit__(self, *exc):
        self._restore()
        return False


def _cycle(spec_path, out, miss_report, miss_report_sha256, model_family, flag_id, argv, t_start):
    """The cycle itself. Raises CycleError on any refused input; writes only under `out`."""
    import numpy as np
    spec = json.loads(Path(spec_path).read_text())
    miss = load_miss_report(miss_report, miss_report_sha256)
    from models.inference_bundle import load_inference_bundle
    from models.feature_manifest import load_manifest, group_of as _group_of, manifest_sha256
    import per_survivor_attribution as psa
    import logging
    logging.basicConfig(level=logging.WARNING)

    # ---- inputs, sha-checked
    want_ck = spec.get("expect_checkpoint_sha256")
    if not (isinstance(want_ck, str) and len(want_ck) == 64 and all(c in "0123456789abcdef" for c in want_ck)):
        raise CycleError("expect_checkpoint_sha256 is required (64 lower-case hex; owner decision D3, "
                         f"2026-11-26); got {want_ck!r}")
    scaler_source = spec.get("scaler_source", "resolved")
    if scaler_source not in SCALER_SOURCES:
        raise CycleError(f"scaler_source {scaler_source!r} not in {SCALER_SOURCES}")
    ev = spec.get("step6_evidence")
    if scaler_source == "as_written" and ev is None:
        raise CycleError("scaler_source \"as_written\" needs step6_evidence (saved Step 6 scores or score hash, "
                         "and the scaler state the vote harness recorded): nothing shows that Step 6 scored "
                         f"without its scaler ({Q3_REF})")
    sp = spec["survivors"]["path"]
    if sha256_file(sp) != spec["survivors"]["sha256"]:
        raise CycleError("survivors sha256 differs from the spec")
    rows = json.loads(Path(sp).read_text())
    bundle = load_inference_bundle(spec["models_dir"], device="cpu", scaler_source=scaler_source)
    if bundle.checkpoint_hash != want_ck:
        raise CycleError(f"checkpoint sha256 {bundle.checkpoint_hash} != declared {want_ck}")
    if model_family is not None and bundle.model_type != model_family:
        raise CycleError(f"model family {bundle.model_type!r} loaded from {spec['models_dir']} != the family "
                         f"this run is for, {model_family!r}")
    written_ck = bundle.meta.get("checkpoint_path")
    written_exists = bool(written_ck) and Path(written_ck).exists()
    if bundle.model_type == "neural_net" and not written_exists and ev is None:
        # G-Q4-SCALER: Step 6 reads the scaler from the WRITTEN path, which is missing now; without saved
        # Step 6 evidence nothing shows whether it was missing when Step 6 scored
        raise CycleError(f"G-Q4-SCALER: neural_net sidecar's written checkpoint_path {written_ck} does not "
                         f"exist, so whether Step 6 scored with its scaler is unknown ({Q4_REF}; {Q3_REF}). "
                         f"Name step6_evidence (saved Step 6 scores or score hash, and scaler_applied).")
    vp = spec["votes"]["path"]
    if sha256_file(vp) != spec["votes"]["sha256"]:
        raise CycleError("votes sha256 differs from the spec")
    with np.load(vp, allow_pickle=False) as z:
        V, v_seeds, v_cidx = z["V"], z["seeds"].astype(np.int64), z["combined_index"].astype(np.int64)
    if [int(r["seed"]) for r in rows] != v_seeds.tolist():
        raise CycleError("vote matrix rows are not the survivors' rows in order")
    if V.shape != (len(rows), len(v_cidx)):
        raise CycleError(f"V shape {V.shape} != ({len(rows)}, {len(v_cidx)})")
    col_of = {int(t): j for j, t in enumerate(v_cidx)}

    outcome, session_of, grade_files = {}, {}, []
    for gdir in spec["grade_dirs"]:
        g = Path(gdir)
        pd = g / "per_draw.jsonl"
        sums = dict(reversed(l.split()) for l in (g / "SHA256SUMS").read_text().splitlines() if l.strip())
        sh = sha256_file(pd)
        if sums.get("per_draw.jsonl") != sh:
            raise CycleError(f"{pd}: sha256 not as in SHA256SUMS")
        n = 0
        for line in pd.read_text().splitlines():
            d = json.loads(line)
            t = int(d["combined_index"])
            if session_of.setdefault(t, d["session"]) != d["session"]:
                raise CycleError(f"t={t}: session differs between grader files")
            if d.get("status", "graded") != "graded":
                continue
            o = int(d["outcome"])
            if outcome.setdefault(t, o) != o:
                raise CycleError(f"t={t}: outcome differs between grader files")
            n += 1
        grade_files.append({"path": str(pd), "sha256": sh, "graded_rows": n})

    # ---- ranking: Step 6's prediction through the shared loader; the module's order rule
    X = bundle.matrix([r["features"] for r in rows])          # by name; refuses a missing name
    scores = bundle.predict(X)
    scaler_applied = bool(bundle.model_type == "neural_net" and bundle.normalize_features
                          and bundle.scaler_mean is not None)
    if ev is not None:
        step6_match = check_step6_evidence(ev, scores, v_seeds.tolist(), scaler_applied)
        scaler_note = (f"Step 6 match checked against saved evidence ({len(step6_match['items'])} items); "
                       f"scaler applied: {scaler_applied}; scaler_source {scaler_source}")
    else:
        step6_match = {"checked": False,
                       "reason": "no step6_evidence in the spec" + (
                           "; the written checkpoint_path exists now, but whether it existed when Step 6 "
                           "scored is not known" if bundle.model_type == "neural_net" else "")}
        scaler_note = "Step 6 match NOT checked: " + step6_match["reason"]
    step6_match["scaler_applied_this_run"] = scaler_applied
    order = np.argsort(scores)[::-1]                          # prediction_generator.py:807 argsort rule
    att = spec.get("attribution", {})
    Z = bundle.transform(X)
    M = psa.per_survivor_attribution_signed_matrix(
        bundle.attribution_model, bundle.model_type, Z, bundle.feature_names,
        method=att.get("method"), expected_feature_count=bundle.feature_count,
        y_mean=bundle.y_mean, y_std=bundle.y_std, ig_steps=int(att.get("ig_steps", 64)))
    A = M["signed"]
    anorm = float(np.max(np.abs(np.asarray(M["prediction"], dtype=np.float64) - scores))) if len(scores) else 0.0
    a_atol, a_rtol = spec.get("a_norm_atol", A_NORM_ATOL_DEFAULT), spec.get("a_norm_rtol", A_NORM_RTOL_DEFAULT)
    a_allowed, a_scale = allowed_gap(a_atol, a_rtol, scores, "A-NORM")
    if not anorm <= a_allowed:
        raise CycleError(f"A-NORM: attribution forward differs from Step 6 prediction by {anorm} > allowed "
                         f"{a_allowed} (atol {float(a_atol)} + rtol {float(a_rtol)} x max|score| {a_scale})")

    names = list(bundle.feature_names)
    per_seed, per_run_rows = miss.classify_columns(names, X.astype(np.float64), spec["per_run_prefixes"])
    per_run = [r["name"] for r in per_run_rows]
    man = load_manifest()
    gof = _group_of(man)
    n_surv = len(rows)
    pool_sizes = spec["pool_sizes"]
    edges = miss.band_edges(pool_sizes, n_surv)
    rng_master = np.random.default_rng(int(spec["seed"]))
    B = int(spec["bootstrap"]["B"])
    rules = spec["abstain"]

    bands = band_shares(A, band_members(order, edges), names, gof, per_run, B,
                        np.random.default_rng(rng_master.integers(1 << 62)),
                        int(spec["band_min_survivors"]))

    # ---- graded contrasts on standardised attributions (per-seed only, non-constant)
    ps_idx = [names.index(n) for n in per_seed]
    var = A[:, ps_idx].std(axis=0)
    used = [j for j, s in zip(ps_idx, var) if s > 0]
    unused = [names[j] for j, s in zip(ps_idx, var) if s == 0]
    att_names = [names[j] for j in used]
    ZA = miss.standardise(A[:, used]) if used else None
    cells, c2 = {}, {}
    for bname, (b0, b1) in spec["blocks"].items():
        for sess in spec["sessions"]:
            ts = [t for t in range(b0, b1) if session_of.get(t) == sess and t in outcome]
            missing = [t for t in ts if t not in col_of]
            if missing:
                raise CycleError(f"{bname}/{sess}: {len(missing)} graded draws not in the vote matrix")
            key = f"{bname}/{sess}"
            if not ts or ZA is None:
                cells[key] = {"draws": len(ts), "abstained": True,
                              "abstain_reason": "no draws" if not ts else "no varying attribution"}
                continue
            cols = np.array([col_of[t] for t in ts])
            o = np.array([outcome[t] for t in ts])
            EQ = V[:, cols] == o[None, :]
            c2[key] = miss.contrast_c2(EQ, ZA, att_names, rules, B,
                                       np.random.default_rng(rng_master.integers(1 << 62)))
            c2[key]["multiplicity"] = miss.multiplicity(c2[key].get("features"))
            cell = {"draws": len(ts), "cohorts": {}}
            for N in pool_sizes:
                cc = miss.contrasts(EQ, ZA, order[:N], att_names, rules, B,
                                    np.random.default_rng(rng_master.integers(1 << 62)))
                for k in ("C1", "C1b"):
                    cc[k]["multiplicity"] = miss.multiplicity(cc[k].get("features"))
                cell["cohorts"][str(N)] = cc
            cells[key] = cell
    rec_sets = {f"{k}/N={N}": v["cohorts"][str(N)] for k, v in cells.items() if "cohorts" in v
                for N in pool_sizes}
    rec = recurring(rec_sets, "C1b")
    prio = priorities(bands, rec, gof, per_run, f"1-{min(pool_sizes)}")
    run_for = {"model_family": model_family, "flag_id": flag_id}
    prio["run_for"] = run_for
    prio["scaler_note"] = scaler_note
    prio["step6_match"] = step6_match

    report = {"format": FORMAT, "label": spec.get("label", "DEVELOPMENT"), "advisory": ADVISORY,
              "scaler_note": scaler_note, "step6_match": step6_match, "run_for": run_for,
              "model": bundle.describe(),
              "attribution": {"method": M["method"], "ig_steps": M["ig_steps"],
                              "feature_names_sha256": M["feature_names_sha256"],
                              "model_feature_names": M["model_feature_names"],
                              "completeness_residual_max_abs": float(np.max(np.abs(M["completeness_residual"]))),
                              "completeness_exact_method": M["completeness_exact_method"],
                              "a_norm_max_abs_diff": anorm, "a_norm_atol": float(a_atol),
                              "a_norm_rtol": float(a_rtol), "a_norm_score_scale": a_scale,
                              "a_norm_allowed": a_allowed,
                              "a_norm_scope": "attribution forward vs this script's own forward (bundle.predict); "
                                              "not checked against Step 6 scores written to disk"},
              "namespaces": {"per_seed": per_seed, "per_run": per_run_rows,
                             "rule": f"miss report classify_columns, prefixes {spec['per_run_prefixes']}"},
              "per_seed_features_with_zero_attribution_variance": unused,
              "bands": bands, "contrasts": cells, "c2": c2, "recurring_C1b": rec,
              "grade_files": grade_files}
    p1 = out / "attribution_cycle.json"
    p1.write_text(json.dumps(report, indent=1, default=float))
    p2 = out / "extraction_priorities.json"
    p2.write_text(json.dumps(prio, indent=1, default=float))
    np.savez_compressed(out / "signed_attribution.npz", A=A.astype(np.float32),
                        prediction=M["prediction"], baseline=M["baseline"],
                        seeds=v_seeds, feature_names=np.array(names))
    rr = {"format": FORMAT + ".run_record", "argv": list(argv),
          "run_for": run_for,
          "spec": {"path": str(Path(spec_path).resolve()), "sha256": sha256_file(spec_path)},
          "model_binding": {"checkpoint_path": bundle.checkpoint_path, "checkpoint_sha256": bundle.checkpoint_hash,
                            "expect_checkpoint_sha256": want_ck,
                            "sidecar_written_checkpoint_path": written_ck,
                            "sidecar_written_checkpoint_exists": written_exists,
                            "feature_schema_hash": bundle.feature_schema_hash,
                            "scaler_source": scaler_source, "scaler_note": scaler_note,
                            "step6_match": step6_match,
                            "normalize_features": bundle.normalize_features},
          "code": {"attribution_cycle_sha256": sha256_file(__file__),
                   "miss_report": {"path": str(miss_report), "sha256": sha256_file(miss_report)},
                   "per_survivor_attribution_sha256": sha256_file(REPO_ROOT / "per_survivor_attribution.py"),
                   "inference_bundle_sha256": sha256_file(REPO_ROOT / "models/inference_bundle.py"),
                   "feature_manifest_sha256": manifest_sha256()},
          "outputs": {p.name: sha256_file(p) for p in (p1, p2, out / "signed_attribution.npz")},
          "seconds": round(time.time() - t_start, 2),
          "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
    p3 = out / "run_record.json"
    p3.write_text(json.dumps(rr, indent=2))
    print(f"[LL-8] {n_surv} survivors attributed ({bundle.model_type}), {len(cells)} draw sets, "
          f"{len(rec)} recurring contrasts; {rr['seconds']} s, {rr['peak_rss_mib']} MiB", flush=True)
    return {"report_path": str(p1), "priorities_path": str(p2), "run_record_path": str(p3),
            "signed_attribution_path": str(out / "signed_attribution.npz"),
            "outputs": dict(rr["outputs"], **{p3.name: sha256_file(p3)}),
            "checkpoint_sha256": bundle.checkpoint_hash, "model_type": bundle.model_type,
            "scaler_source": scaler_source, "scaler_note": scaler_note, "step6_match": step6_match,
            "seconds": rr["seconds"], "peak_rss_mib": rr["peak_rss_mib"]}


def run_attribution_cycle(spec_path, out_dir, model_family=None, flag_id=None, miss_report=None,
                          miss_report_sha256=MISS_REPORT_SHA256, result_file=None, argv=None):
    """Function entry point (for a later runner; unit W is not built here). Never raises: returns the
    result dict, which is also written as JSON to `result_file` (default <out_dir>/result.json):
      status      "ok" | "refused" (a CycleError: an input failed a check, nothing was reported) |
                  "error" (anything else, including a failure resolving paths or publishing the result;
                  message and traceback tail recorded)
      exit_code   0 | 2 | 3 (what the CLI exits with)
      result_written  the path the result was written to, or None when no safe place existed
                  (out_dir already exists or lies outside <worktree>/data/, and no valid result_file)
                  or publication failed (then status error, exit 3, the run's own status, class and
                  message kept under before_publication, the reason on stderr)
    Writes only under out_dir (and result_file). Changes no setting; reads no flag file; leaves the
    caller's process state as it found it (_CallerScope)."""
    import traceback
    t_start = time.time()
    res = {"format": RESULT_FORMAT, "status": None, "exit_code": None, "error_class": None, "message": None,
           "model_family": model_family, "flag_id": flag_id, "spec": str(spec_path), "out_dir": str(out_dir),
           "advisory": ADVISORY, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    out = rf = None
    writable_out = False
    try:
        miss_report = str(miss_report or MISS_REPORT_DEFAULT)
        out = Path(out_dir).resolve()
        res["out_dir"] = str(out)
        rf = Path(result_file).resolve() if result_file else None
        if rf is not None:
            why = _result_target_problem(rf, out)
            if why:
                rf = None
                raise CycleError(f"--result-file {result_file} {why}")
        if out.exists():
            raise CycleError(f"--out-dir {out} exists")
        if not _under_data(out):
            raise CycleError("--out-dir must lie under <worktree>/data/")
        out.mkdir(parents=True)
        writable_out = True
        res["model_family"] = check_flag(model_family, flag_id)
        with _CallerScope(out):
            res.update(_cycle(spec_path, out, miss_report, miss_report_sha256, res["model_family"], flag_id,
                              argv if argv is not None else sys.argv, t_start))
        res["status"], res["exit_code"] = "ok", EXIT_OK
    except CycleError as e:
        res.update(status="refused", exit_code=EXIT_REFUSED, error_class=type(e).__name__, message=str(e))
        print(f"[LL-8] FAIL-CLOSED: {e}", file=sys.stderr, flush=True)
    except Exception as e:   # surfaced, never swallowed: status error, class, message, traceback tail
        res.update(status="error", exit_code=EXIT_ERROR, error_class=type(e).__name__, message=str(e),
                   traceback_tail=traceback.format_exc()[-2000:])
        print(f"[LL-8] ERROR: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    res["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    target = rf or (out / "result.json" if writable_out else None)
    res["result_written"] = str(target) if target else None
    if target is not None:
        try:
            _publish(target, res)
        except Exception as e:
            before = {k: res.get(k) for k in ("status", "exit_code", "error_class", "message")}
            res.update(status="error", exit_code=EXIT_ERROR, error_class=type(e).__name__,
                       message=f"result not published to {target}: {type(e).__name__}: {e}",
                       result_written=None, before_publication=before)
            print(f"[LL-8] ERROR: result not published to {target}: {type(e).__name__}: {e}; the run itself was "
                  f"{before['status']}" + (f": {before['message']}" if before["message"] else ""),
                  file=sys.stderr, flush=True)
    return res


def main(argv=None):
    a = parse_args(argv)
    res = run_attribution_cycle(a.spec, a.out_dir, model_family=a.model_family, flag_id=a.flag_id,
                                miss_report=a.miss_report, miss_report_sha256=a.miss_report_sha256,
                                result_file=a.result_file,
                                argv=(sys.argv if argv is None else list(argv)))
    return res["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
