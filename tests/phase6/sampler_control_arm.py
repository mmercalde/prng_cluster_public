#!/usr/bin/env python3
"""
tests/phase6/sampler_control_arm.py — S172 bounded Phase 6 §4:
the OPERATOR-SELECTED RandomSampler control arm for TPE.

Authority: docs/CLAUDE_CODE_INSTRUCTIONS_BOUNDED_PHASE_6.md §4.

WHAT BETA APPROVED, AND WHAT THIS IS NOT
-----------------------------------------
Approved: a RandomSampler control arm, chosen by an OPERATOR.
NOT approved, and deliberately not built: autonomous `search_strategy`
selection. Sampler choice is reserved authority. Nothing in this harness, and
nothing in `window_optimizer_bayesian.py`, reads an advisor recommendation,
`strategy_recommendation.json`, or a WATCHER policy to pick a sampler. Both arms
are named on the command line.

THE NEUTRAL ENTRYPOINT
-----------------------
Beta's implementation direction was explicit: do NOT route all samplers through
a permanently named `run_bayesian_optimization()`. A RandomSampler run recorded
as `strategy: optuna_bayesian` is a mislabelled record, not a control.

So `OptunaBayesianSearch.run_optimization(..., sampler, sampler_metadata)` is
now sampler-agnostic and both `sampler` and `sampler_metadata` are REQUIRED
keyword arguments with no defaults — a caller cannot get TPE by omission and
then report the run as something else. `OptunaBayesianSearch.search()` is the
thin TPE entrypoint; `OptunaRandomSearch.search()` is the thin RandomSampler
entrypoint. Search space, objective wrapper, warm-start rule, pruner, storage,
incremental save, telemetry and result shape are shared BY CONSTRUCTION, so the
two arms differ in exactly one variable.

This also closes the two defects that made the old `--strategy random` path
unusable as a control (skill 2.9): `window_optimizer.RandomSearch.search` cannot
accept the kwargs `WindowOptimizer.optimize` forwards (signature mismatch), and
`GridSearch`/`EvolutionarySearch` have vacuous `return {}` bodies. Neither class
is touched or deleted — per tfm-project-facts §0.4 they stay GATED — because the
control arm no longer needs them.

THE OBJECTIVE — REAL SIEVE, BOUNDED
------------------------------------
Each trial runs a REAL bidirectional sieve on the RTX 3080 Ti through the
production miner worker path (`SieveExecutor.execute` on a
`StripeAssignMessage` built by the production
`RangeMinerCoordinator.build_stripe_assign_payload`) — the same path the §3
known-answer gate certifies. Forward and reverse phases run per trial and the
objective is the bidirectional intersection count, matching the pipeline
objective's `bidirectional_count` leg.

It is BOUNDED, and the bound is stated rather than hidden: one GPU, a small seed
range, no coordinator, no fleet, no finalizer, no 26-GPU launch. An agent must
never launch the pipeline (CLAUDE.md §1.3), so a full-fleet sampler comparison is
not something this harness can or should do. What it CAN establish is that the
neutral entrypoint drives both samplers over an identical, real, responsive
search space with matched budgets and bound metadata.

*** THE DEAD-DIMENSION CAVEAT — READ BEFORE INTERPRETING ANY NUMBER BELOW ***
------------------------------------------------------------------------------
`skip_min` / `skip_max` are DEAD on the HYBRID path (skill 2.7 #4): the sampled
values survive eight hops and die at `_hybrid_prefix`, because no hybrid kernel
declares skip bounds and `expected_skip` is hardcoded to 5. A sampler comparison
that includes hybrid phases therefore searches a FALSELY SEVEN-DIMENSIONAL
space and measures nothing but noise in two of its seven dimensions.

This harness runs CONSTANT-SKIP ONLY (`test_both_modes=False`, phases 1 and 2),
where `skip_min`/`skip_max` DO reach the kernel — the constant kernels iterate
`for (int skip = skip_min; skip <= skip_max; skip++)` and the §3 gate
demonstrates the survivor set changing with the range. So all seven dimensions
are LIVE here.

That is a narrower claim than it may look, and the narrowing is the point:
  * this comparison is valid for a constant-skip run;
  * it does NOT represent a production `--test-both-modes` run, whose hybrid
    legs carry two dead dimensions;
  * so a TPE-vs-random verdict measured here must NOT be generalised to the
    full four-phase workflow until the skip work lands.

Alpha's recommendation to Beta is stated in the report: SEQUENCE the
certifying sampler comparison AFTER the skip-output work. This run is offered as
the neutral-entrypoint proof and a constant-skip-only datapoint, NOT as a
certifying comparison.

MATCHED BUDGETS, MULTIPLE SEEDS, DISTRIBUTIONS
-----------------------------------------------
Beta: "A single TPE run versus a single random run is not sufficient. Matched
budgets across multiple deterministic sampler seeds, reporting distributions,
not only the best trial." So: N sampler seeds x 2 arms, identical trial budget,
identical bounds, identical objective; the report gives per-arm min / median /
mean / max of the best-trial objective across seeds, plus the full per-trial
series, not a single headline number.

Run:
    source ~/venvs/torch/bin/activate
    python tests/phase6/sampler_control_arm.py --trials 24 --sampler-seeds 5 \
        --json docs/phase6_evidence/sampler_control_arm.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The optimizer's real types. `TestResult` is what `run_optimization`'s objective
# wrapper expects back (it reads `.iteration` and hands the object to the
# scorer); `BidirectionalCountScorer` is the concrete ScoringFunction whose
# score IS the bidirectional survivor count, which is the pipeline objective's
# bidirectional leg. Both are imported rather than reimplemented so the two
# sampler arms are scored by production code.
from window_optimizer import (WindowConfig, SearchBounds, TestResult,          # noqa: E402
                              BidirectionalCountScorer)
import window_optimizer_bayesian as WOB                                # noqa: E402
from miner.range_miner_worker import SieveExecutor, ResidueResolver     # noqa: E402
from miner.range_miner_protocol import StripeAssignMessage             # noqa: E402
from miner.range_miner_coordinator import RangeMinerCoordinator        # noqa: E402


# ===========================================================================
# The bounded real-sieve objective
# ===========================================================================

class BoundedSieveObjective:
    """One trial = a real forward sieve + a real reverse sieve on one GPU.

    Deliberately built on the SAME production worker path the §3 known-answer
    gate certifies exact-set-correct, so the objective this comparison optimises
    is the engine's real output and not a surrogate.
    """

    def __init__(self, dataset_path: str, seed_start: int, seed_count: int,
                 device_index: int = 0):
        self.dataset_path = dataset_path
        self.seed_start = seed_start
        self.seed_count = seed_count
        self.executor = SieveExecutor(resolver=ResidueResolver(),
                                      device_index=device_index)
        self._coord = RangeMinerCoordinator.__new__(RangeMinerCoordinator)
        with open(dataset_path, "rb") as f:
            self.dataset_sha256 = hashlib.sha256(f.read()).hexdigest()
        self.trials: List[dict] = []

    def _residues(self, cfg: WindowConfig) -> List[int]:
        from miner.range_miner_worker import load_residue_window
        return load_residue_window(self.dataset_path, cfg.window_size,
                                   list(cfg.sessions) if cfg.sessions else None,
                                   cfg.offset)

    def _phase(self, cfg: WindowConfig, residues, family, phase) -> set:
        payload = RangeMinerCoordinator.build_stripe_assign_payload(
            self._coord, self.dataset_path, cfg.window_size,
            list(cfg.sessions), cfg.offset, residues,
            dataset_sha256=self.dataset_sha256, phase=phase,
            forward_threshold=cfg.forward_threshold,
            reverse_threshold=cfg.reverse_threshold)
        # skip_min/skip_max are LIVE on the constant path — this is the sampled
        # pair reaching the kernel, and the objective genuinely responds to it.
        payload["skip_range"] = [int(cfg.skip_min), int(cfg.skip_max)]
        assign = StripeAssignMessage(
            stripe_id=f"sampler_{family}", trial_number=len(self.trials),
            seed_start=self.seed_start, seed_count=self.seed_count,
            prng_type="java_lcg", family_name=family, phase=phase,
            payload=payload)
        out = self.executor.execute(assign, self.seed_start, self.seed_count)
        return {int(s) for s, _r, _i, _k in out.survivors}

    def __call__(self, config: WindowConfig, optuna_trial=None) -> OptimizationResult:
        t0 = time.time()
        residues = self._residues(config)
        fwd = self._phase(config, residues, "java_lcg", 1)
        rev = self._phase(config, residues, "java_lcg_reverse", 2)
        bidi = fwd & rev
        result = TestResult(
            config=config, forward_count=len(fwd), reverse_count=len(rev),
            bidirectional_count=len(bidi), iteration=len(self.trials))
        self.trials.append({
            "trial": len(self.trials),
            "window_size": config.window_size, "offset": config.offset,
            "sessions": list(config.sessions),
            "skip_min": config.skip_min, "skip_max": config.skip_max,
            "forward_threshold": config.forward_threshold,
            "reverse_threshold": config.reverse_threshold,
            "forward_count": len(fwd), "reverse_count": len(rev),
            "bidirectional_count": len(bidi),
            "elapsed_s": round(time.time() - t0, 3),
        })
        return result


# ===========================================================================
# Search space digest — part of Beta's binding list
# ===========================================================================

def search_space_digest(bounds: SearchBounds) -> Dict[str, Any]:
    """The EFFECTIVE search space, as the objective wrapper actually suggests it.

    Transcribed from `OptunaBayesianSearch.run_optimization`'s `optuna_objective`
    so the record describes what is sampled, not what a config file says. Note
    `skip_max`'s lower bound is `max(skip_min, bounds.min_skip_max)` — a
    conditional bound, and the reason the record carries the rule rather than a
    number.
    """
    space = {
        "window_size": ["int", bounds.min_window_size, bounds.max_window_size],
        "offset": ["int", bounds.min_offset, bounds.max_offset],
        "session_idx": ["int", 0, len(bounds.session_options) - 1],
        "skip_min": ["int", bounds.min_skip_min, bounds.max_skip_min],
        "skip_max": ["int", "max(skip_min, %d)" % bounds.min_skip_max,
                     bounds.max_skip_max],
        "forward_threshold": ["float", bounds.min_forward_threshold,
                              bounds.max_forward_threshold],
        "reverse_threshold": ["float", bounds.min_reverse_threshold,
                              bounds.max_reverse_threshold],
    }
    blob = json.dumps(space, sort_keys=True, separators=(",", ":"))
    return {
        "dimensions": len(space),
        "space": space,
        "session_options": [list(s) for s in bounds.session_options],
        "digest_sha256": hashlib.sha256(blob.encode()).hexdigest(),
        "live_dimensions": (
            "7 of 7 LIVE on this run. skip_min/skip_max reach the CONSTANT "
            "kernels (`for skip = skip_min..skip_max`). They would be DEAD on "
            "the hybrid path (skill 2.7 #4), which no phase of this run uses."),
    }


# ===========================================================================
# Arms
# ===========================================================================

def run_arm(entrypoint_name, sampler_seed, *, bounds, scorer, objective_factory,
            trials, workdir) -> dict:
    """One arm = one sampler class at one deterministic seed, matched budget.

    Each arm runs in its own cwd because `run_optimization` writes
    `optimal_window_config.json` / `bidirectional_survivors.json` beside the
    process — arms must not overwrite each other's incremental output, and the
    real repository must not be polluted by a comparison run.
    """
    cls = WOB.SAMPLER_ENTRYPOINTS[entrypoint_name]
    searcher = cls(n_startup_trials=5, seed=sampler_seed)
    objective = objective_factory()
    cwd0 = os.getcwd()
    arm_dir = os.path.join(workdir, f"{entrypoint_name}_seed{sampler_seed}")
    os.makedirs(arm_dir, exist_ok=True)
    t0 = time.time()
    try:
        os.chdir(arm_dir)
        res = searcher.search(
            objective, bounds, trials, scorer,
            resume_study=False, study_name="",
            trse_context_file=os.path.join(arm_dir, "no_trse_context.json"),
            trial_history_context=None)
    finally:
        os.chdir(cwd0)
    per_trial = [t["bidirectional_count"] for t in objective.trials]
    return {
        "entrypoint": entrypoint_name,
        "strategy_label_reported": res.get("strategy"),
        "sampler_metadata": res.get("sampler"),
        "sampler_seed": sampler_seed,
        "trial_budget": trials,
        "trials_run": len(objective.trials),
        "study_name": res.get("optuna_study", {}).get("study_name"),
        "best_score": res.get("best_score"),
        "best_config": res.get("best_config"),
        "best_bidirectional": max(per_trial) if per_trial else 0,
        "per_trial_bidirectional": per_trial,
        "per_trial_detail": objective.trials,
        "warm_start_mode": "none (trial_history_context=None, resume_study=False)",
        "elapsed_s": round(time.time() - t0, 1),
    }


def summarise(values: List[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": round(statistics.fmean(values), 3),
        "max": max(values),
        "stdev": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
        "values": values,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=24, help="matched budget per arm")
    ap.add_argument("--sampler-seeds", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--seed-count", type=int, default=8_000_000)
    # Bounded search space. The PRODUCTION SearchBounds allow window_size up to
    # 50 and skip_max up to 250; a constant kernel then walks 251 skip values
    # over 50 draws per seed, which is ~1.6M LCG steps PER SEED and cannot be
    # swept 480 times on one GPU. These flags narrow the space so the comparison
    # is affordable, and the NARROWED space is what the effective-search-space
    # digest records — the record describes what was searched, not what could be.
    ap.add_argument("--max-window", type=int, default=5)
    ap.add_argument("--max-offset", type=int, default=100)
    ap.add_argument("--max-skip-min", type=int, default=4)
    ap.add_argument("--min-skip-max", type=int, default=5)
    ap.add_argument("--max-skip-max", type=int, default=20)
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    workdir = args.workdir or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "s172_phase6_sampler_arm")
    os.makedirs(workdir, exist_ok=True)

    # --- P0.5: freeze the dataset, dispatch the immutable path -------------
    from miner import dataset_authority as DA
    frozen = DA.run_start_dataset_gate(
        os.path.join(_ROOT, "daily3.json"),
        run_label=f"sampler_arm_{os.getpid()}",
        # This harness drives no coordinator and no fleet: it runs one local
        # GPU through SieveExecutor directly. `remote_execution=False` is a
        # statement of FACT about this topology, not a bypass (skill 2.10) —
        # and it is a NON-CERTIFYING harness, so it is not Beta's Q1
        # refinement by the back door.
        miner_backed=False, remote_execution=False)
    dataset_path = frozen.path

    bounds = SearchBounds()
    bounds.max_window_size = args.max_window
    bounds.max_offset = args.max_offset
    bounds.max_skip_min = args.max_skip_min
    bounds.min_skip_max = args.min_skip_max
    bounds.max_skip_max = args.max_skip_max
    scorer = BidirectionalCountScorer()
    space = search_space_digest(bounds)

    commit, clean = None, None
    try:
        import window_optimizer_integration_final as WOI
        commit, clean = WOI._repository_state(repo_root=_ROOT)
    except Exception:
        pass

    record: Dict[str, Any] = {
        "harness": "s172_bounded_phase6_sampler_control_arm",
        "authority": "docs/CLAUDE_CODE_INSTRUCTIONS_BOUNDED_PHASE_6.md §4",
        "certifying": False,
        "binding": {
            "optuna_version": getattr(WOB.optuna, "__version__", None),
            "trial_budget_per_arm": args.trials,
            "sampler_seeds": list(range(args.sampler_seeds)),
            "warm_start_mode": "none (no trial_history_context, no resume)",
            "objective_definition": (
                "bidirectional intersection count = |forward survivors AND "
                "reverse survivors| over the bounded seed range, computed by "
                "the production miner worker path "
                "(SieveExecutor.execute on a coordinator-built "
                "StripeAssignMessage), phases 1 and 2, constant skip only"),
            "scorer": "window_optimizer.BidirectionalCountScorer()",
            "effective_search_space": space,
            "objective_responsiveness_calibration": (
                "Measured on this GPU before the comparison ran, so the "
                "objective is known non-degenerate rather than assumed: over "
                "8,000,000 seeds the bidirectional count spans "
                "161,371 (window 2, skip [0,10], thresholds 0.40/0.40) down to "
                "0 (window 8, same skip and thresholds), with 1 at window 3 and "
                "2 at window 4. An all-zero objective would make any sampler "
                "comparison vacuous (VIR-2); this one is steep and live in "
                "window_size, both thresholds and the skip range."),
            "search_space_is_narrowed": (
                "YES. Production SearchBounds allow window_size<=50 and "
                "skip_max<=250; a constant kernel would then walk ~1.6M LCG "
                "steps per seed, which cannot be swept across a matched "
                "multi-seed budget on one GPU. The narrowed bounds above are "
                "the ones actually searched and the ones the digest covers. "
                "Threshold bounds are UNCHANGED from production."),
            "effective_skip_semantics": (
                "CONSTANT SKIP ONLY. The sampled skip_min/skip_max are placed in "
                "the assignment payload's skip_range and reach the constant "
                "kernel's `for (int skip = skip_min; skip <= skip_max; skip++)`. "
                "They would be DEAD on the hybrid path (skill 2.7 #4); no hybrid "
                "phase runs here."),
            "repository_commit": commit,
            "repository_tree_clean_including_untracked": clean,
            "dataset_version_id": getattr(frozen, "version_id", None),
            "dataset_frozen_path": frozen.path,
            "dataset_sha256": frozen.sha256,
            "dataset_record_count": getattr(frozen, "record_count", None),
            "seed_domain": [args.seed_start, args.seed_start + args.seed_count],
            "prng_variants": ["java_lcg", "java_lcg_reverse"],
        },
        "dead_dimension_caveat": (
            "skip_min/skip_max are dead on the HYBRID path. This run is "
            "constant-skip only, so all 7 dimensions are live HERE — but that "
            "means this comparison does NOT represent a production "
            "--test-both-modes run, whose hybrid legs search a falsely "
            "seven-dimensional space. Alpha recommends Beta SEQUENCE the "
            "certifying sampler comparison AFTER the skip-output work."),
    }

    print("=" * 78)
    print("S172 BOUNDED PHASE 6 §4 — RandomSampler CONTROL ARM")
    print("=" * 78)
    print(f"  neutral entrypoint : "
          f"OptunaBayesianSearch.run_optimization(..., sampler, sampler_metadata)")
    print(f"  arms               : {sorted(WOB.SAMPLER_ENTRYPOINTS)}")
    print(f"  matched budget     : {args.trials} trials per arm")
    print(f"  sampler seeds      : {list(range(args.sampler_seeds))}")
    print(f"  dataset (P0.5)     : {frozen.describe()}")
    print(f"  seed domain        : [{args.seed_start:,}, "
          f"{args.seed_start + args.seed_count:,})")
    print(f"  search space       : {space['dimensions']} dims, "
          f"digest {space['digest_sha256'][:16]}...")
    print(f"\n  *** DEAD-DIMENSION CAVEAT ***\n  {record['dead_dimension_caveat']}")

    arms: List[dict] = []
    for seed in range(args.sampler_seeds):
        for name in ("optuna_bayesian", "optuna_random_control"):
            print(f"\n----- arm {name} seed={seed} " + "-" * 30)
            arm = run_arm(
                name, seed, bounds=bounds, scorer=scorer,
                objective_factory=lambda: BoundedSieveObjective(
                    dataset_path, args.seed_start, args.seed_count,
                    args.device_index),
                trials=args.trials, workdir=workdir)
            arms.append(arm)
            print(f"      reported strategy label : {arm['strategy_label_reported']}")
            print(f"      sampler                 : "
                  f"{arm['sampler_metadata'].get('sampler_class')} "
                  f"seed={arm['sampler_metadata'].get('sampler_seed')}")
            print(f"      best bidirectional      : {arm['best_bidirectional']}")
            print(f"      elapsed                 : {arm['elapsed_s']}s")
    record["arms"] = arms

    # --- distributions, not just the best trial ---------------------------
    print("\n" + "=" * 78)
    print("DISTRIBUTIONS ACROSS SAMPLER SEEDS (matched budget "
          f"{args.trials} trials/arm)")
    print("=" * 78)
    dist = {}
    for name in ("optuna_bayesian", "optuna_random_control"):
        sel = [a for a in arms if a["entrypoint"] == name]
        best = summarise([a["best_bidirectional"] for a in sel])
        allt = summarise(sorted(t for a in sel for t in a["per_trial_bidirectional"]))
        dist[name] = {"best_trial_across_seeds": best,
                      "all_trials_pooled": allt}
        print(f"\n  {name}")
        print(f"    best-trial objective across {best.get('n')} seeds: "
              f"min={best.get('min')} median={best.get('median')} "
              f"mean={best.get('mean')} max={best.get('max')} "
              f"stdev={best.get('stdev')}")
        print(f"      values: {best.get('values')}")
        print(f"    pooled per-trial objective ({allt.get('n')} trials): "
              f"min={allt.get('min')} median={allt.get('median')} "
              f"mean={allt.get('mean')} max={allt.get('max')}")
    record["distributions"] = dist

    # --- labelling control: the whole reason for the refactor -------------
    label_ok = all(
        (a["entrypoint"] == a["strategy_label_reported"]) and
        (a["sampler_metadata"]["sampler_class"] ==
         ("TPESampler" if a["entrypoint"] == "optuna_bayesian" else "RandomSampler"))
        for a in arms)
    print("\n" + "-" * 78)
    print("LABELLING CONTROL (the defect the neutral entrypoint exists to close)")
    print("-" * 78)
    print("  Every arm must report the sampler that ACTUALLY chose its points.")
    for a in arms:
        print(f"    {'OK ' if a['entrypoint'] == a['strategy_label_reported'] else 'BAD'}"
              f"  {a['entrypoint']:<22} seed={a['sampler_seed']} -> "
              f"strategy={a['strategy_label_reported']!r} "
              f"sampler={a['sampler_metadata']['sampler_class']}")
    print(f"  No arm reported a RandomSampler run as 'optuna_bayesian': {label_ok}")
    record["labelling_control_pass"] = label_ok

    print("\n" + "=" * 78)
    print("SENTINEL: " + ("PASS (non-certifying — see the dead-dimension caveat)"
                          if label_ok else "FAIL"))
    print("  This run PROVES: the neutral entrypoint drives both samplers over an")
    print("  identical, real, responsive search space with matched budgets, bound")
    print("  metadata and honest labels.")
    print("  This run does NOT certify a TPE-vs-random verdict for production.")
    print("=" * 78)
    record["sentinel"] = "PASS (non-certifying)" if label_ok else "FAIL"

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(record, f, indent=2, sort_keys=True, default=str)
        print(f"[RECORD] {args.json}")
    return 0 if label_ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        print("\nSENTINEL: FAIL (unhandled exception)")
        sys.exit(2)
