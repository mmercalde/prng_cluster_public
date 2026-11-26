#!/usr/bin/env python3
"""
tests/ll_step3_runner.py - [LL-2 self-test helper] run full_scoring_worker.score_survivors
once on a CPU fixture, in a fresh process, from a chosen code directory.

  --code-dir DIR   put DIR first on sys.path (the 41ad509 baseline copy, or the worktree)
  --fixture DIR    holds chunk.json, train.json, holdout.json
  --settings JSON  extra keyword arguments for score_survivors ({} = call exactly as before)
  --force-fallback make extract_ml_features_batch raise, so the sequential fallback runs
  --mutant NAME    apply one named self-test mutant (see MUTANTS); counted and recorded
  --out FILE       results JSON (metadata.timestamp removed: wall clock, not data)

CPU only: refuses to run if torch sees a CUDA device.
"""
import argparse
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True

MUTANTS = ("fallback_omits_sieve", "fwd_rev_swapped", "off_leaks_sieve",
           "no_fail_closed", "batch_list_drops_one")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-dir", required=True)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--settings", default="{}")
    ap.add_argument("--force-fallback", action="store_true")
    ap.add_argument("--mutant", default=None, choices=MUTANTS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.repo_root)
    sys.path.insert(0, a.code_dir)
    import torch
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise SystemExit("runner is CPU-only but torch sees CUDA")
    import full_scoring_worker as fsw
    import survivor_scorer as ss
    from survivor_scorer import SurvivorScorer
    loaded = {"full_scoring_worker": os.path.realpath(fsw.__file__),
              "survivor_scorer": os.path.realpath(ss.__file__)}
    calls = {"mutant_calls": 0, "batch_forced_raise": 0}

    if a.force_fallback:
        def _boom(*x, **k):
            calls["batch_forced_raise"] += 1
            raise RuntimeError("self-test: batch path forced to fail")
        SurvivorScorer.extract_ml_features_batch = _boom

    if a.mutant:
        m = a.mutant
        if m == "fallback_omits_sieve":
            orig = ss.metadata_merge_fields
            def mm(path, *x, **k):
                f = orig(path, *x, **k)
                if path == "fallback":
                    calls["mutant_calls"] += 1
                    f = tuple(n for n in f if n not in ss.PER_SEED_SIEVE_FIELDS)
                return f
            ss.metadata_merge_fields = mm
        elif m == "fwd_rev_swapped":
            orig = ss.merge_survivor_metadata
            def mg(features, meta, fields):
                calls["mutant_calls"] += 1
                meta = dict(meta)
                if "forward_matches" in meta and "reverse_matches" in meta:
                    meta["forward_matches"], meta["reverse_matches"] = (
                        meta["reverse_matches"], meta["forward_matches"])
                return orig(features, meta, fields)
            ss.merge_survivor_metadata = mg
        elif m == "off_leaks_sieve":
            orig = ss.metadata_merge_fields
            def mm(path, per_seed_sieve_features="off", *x, **k):
                calls["mutant_calls"] += 1
                return orig(path, "on", *x, **k)
            ss.metadata_merge_fields = mm
        elif m == "no_fail_closed":
            def nf(*x, **k):
                calls["mutant_calls"] += 1
            ss.check_per_seed_sieve_metadata = nf
        elif m == "batch_list_drops_one":
            calls["mutant_calls"] += 1
            ss.METADATA_MERGE_FIELDS = ss.METADATA_MERGE_FIELDS[:-1]

    chunk = json.load(open(os.path.join(a.fixture, "chunk.json")))
    train = json.load(open(os.path.join(a.fixture, "train.json")))
    hold = json.load(open(os.path.join(a.fixture, "holdout.json")))
    seeds = [s["seed"] for s in chunk]
    meta = {s["seed"]: s for s in chunk}
    settings = json.loads(a.settings)
    t0 = time.time()
    out = {"loaded": loaded, "settings": settings, "force_fallback": a.force_fallback,
           "mutant": a.mutant}
    try:
        res = fsw.score_survivors(seeds=seeds, train_history=train, prng_type="java_lcg",
                                  mod=1000, batch_size=100, forward_survivors=None,
                                  reverse_survivors=None, scorer_class=SurvivorScorer,
                                  survivor_metadata=meta, holdout_history=hold, **settings)
        for r in res:
            r.get("metadata", {}).pop("timestamp", None)
        out["results"] = res
        out["raised"] = None
    except Exception as e:
        out["results"] = None
        out["raised"] = f"{type(e).__name__}: {e}"
    out["calls"] = calls
    out["seconds"] = round(time.time() - t0, 3)
    import resource
    out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    with open(a.out, "w") as f:
        json.dump(out, f, sort_keys=True)


if __name__ == "__main__":
    main()
