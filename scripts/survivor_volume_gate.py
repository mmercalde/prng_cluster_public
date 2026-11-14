#!/usr/bin/env python3
"""SURVIVOR-VOLUME GATE — refuse a launch whose predicted survivor volume (LOAD)
exceeds the owner's cap in any phase. It predicts and decides. It launches nothing.

WHAT THIS IS
------------
Plan item A2 / finished condition F3 (`docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md`
§2, §3.A; `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md` §7.3). The estimate is
`utils.survivor_volume_estimator.estimate_volume` — this script is one caller of it
and adds only the cap, the ownership check, the GPU check and the verdict.

A SURVIVOR COUNT IS LOAD, NOT QUALITY. The verdict is a LOAD verdict: it says
whether the predicted volume fits inside the largest volume a complete run has
carried. It says nothing about whether the run will hit (CLAUDE.md, "The yardstick").

THE CAP FILE — `config_manifests/survivor_volume.cap`, JSON text, owner-committed
    {"cap_survivors_per_phase": int > 0,
     "compare": "point" | "upper95",
     "min_sample_seeds": int > 0,
     "gpu_process_allowlist": [process name, ...],
     "evidence": str}
Exactly these five fields. Ownership is the run-authorization rule, by the same
function (`env_freeze_gate.tracked_unmodified`): a file tracked and byte-identical
to HEAD is the owner's. The `.cap` extension is deliberate — `.gitignore:41`
ignores `*.json`. `gpu_process_allowlist` is the owner's ruling of 2026-11-14:
compute processes named there (VM101's `sunshine` streaming service) do not make
the GPU busy; any other compute process does.

PARAMETERS (`--params`, a JSON object or @file)
    seed_start, seed_end_exclusive          (required; `max_seeds`, if present, must
                                             equal their difference)
    prng_type, window_size, window_anchor, generator_phase, sessions,
    forward_threshold, reverse_threshold, skip_min, skip_max, dataset,
    test_both_modes                         (required, EXACT values; a range is refused)

THE RULE
    cap file absent / unparseable / field missing or wrong      -> REFUSE
    cap file untracked or differs from HEAD; git cannot run     -> REFUSE
    a parameter missing, unknown, or a range                    -> REFUSE
    --sample-seeds below min_sample_seeds                       -> REFUSE
    no GPU, or a non-allowlisted compute process on it          -> UNAVAILABLE (= REFUSE)
    the sampler raises / times out / returns a malformed result -> ERROR (= REFUSE)
    a sampled block outside the range; sample below the minimum -> REFUSE
    sampled residue window != the payload builder's            -> REFUSE
    any phase's compared figure > cap                           -> REFUSE (names it)
    every phase's compared figure <= cap                        -> PROCEED
The compared figure is the projection (`point`) or the upper end of its 95% interval
(`upper95`). Both are always in the record, whichever rule decided.

Exit status: 0 = PROCEED, 1 = anything else.
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(REPO_ROOT, "scripts")
for _p in (REPO_ROOT, _SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import env_freeze_gate as EF                                        # noqa: E402
import utils.survivor_volume_estimator as SVE                       # noqa: E402

EXIT_PROCEED = EF.EXIT_PROCEED
EXIT_REFUSE = EF.EXIT_REFUSE

GATE = "survivor_volume_gate"
CAP_REL = "config_manifests/survivor_volume.cap"
CAP_FIELDS = ("cap_survivors_per_phase", "compare", "min_sample_seeds",
              "gpu_process_allowlist", "evidence")
COMPARE_RULES = ("point", "upper95")
SEED_KEYS = ("seed_start", "seed_end_exclusive")
DEFAULT_PHASE4_FACTOR = 4      # brief §1.3: phase 4 sampled at 4x (2^28 vs 2^26)

V_PROCEED, V_REFUSE, V_UNAVAILABLE, V_ERROR = "PROCEED", "REFUSE", "UNAVAILABLE", "ERROR"


class CapError(ValueError):
    pass


def _reject_constant(name):
    raise ValueError(f"non-finite number {name} is not permitted")


def _loads(text):
    return json.loads(text, parse_constant=_reject_constant)


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


# ---------------------------------------------------------------------------
# the cap file
# ---------------------------------------------------------------------------

def validate_cap(doc):
    if not isinstance(doc, dict):
        raise CapError("cap file is not a JSON object")
    missing = [f for f in CAP_FIELDS if f not in doc]
    if missing:
        raise CapError(f"missing field(s): {missing}")
    extra = sorted(set(doc) - set(CAP_FIELDS))
    if extra:
        raise CapError(f"unknown field(s): {extra}")
    if not _is_int(doc["cap_survivors_per_phase"]) or doc["cap_survivors_per_phase"] < 1:
        raise CapError("cap_survivors_per_phase is not a positive integer")
    if doc["compare"] not in COMPARE_RULES:
        raise CapError(f"compare {doc['compare']!r} is not one of {COMPARE_RULES}")
    if not _is_int(doc["min_sample_seeds"]) or doc["min_sample_seeds"] < 1:
        raise CapError("min_sample_seeds is not a positive integer")
    al = doc["gpu_process_allowlist"]
    if not (isinstance(al, list) and all(isinstance(x, str) and x for x in al)):
        raise CapError("gpu_process_allowlist is not a list of process names")
    if not isinstance(doc["evidence"], str) or not doc["evidence"]:
        raise CapError("evidence is not a non-empty string")


def load_cap(repo_root, runner=None):
    """(doc, sha256, text, None) or (None, None, None, refusal)."""
    status, reason, data = EF.tracked_unmodified(repo_root, CAP_REL, runner)
    if status != "OK":
        return None, None, None, f"cap file {status}: {reason}"
    try:
        text = data.decode("utf-8")
        doc = _loads(text)
        validate_cap(doc)
    except Exception as exc:                                        # noqa: BLE001
        return None, None, None, f"cap file {CAP_REL} unparseable or malformed: {exc}"
    return doc, hashlib.sha256(data).hexdigest(), text, None


# ---------------------------------------------------------------------------
# the GPU
# ---------------------------------------------------------------------------

def nvidia_probe():
    """{"status": OK|BUSY|ABSENT|UNAVAILABLE, "gpus": [...], "processes": [...]}.
    `processes` are COMPUTE processes as nvidia-smi reports them."""
    try:
        g = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True,
                           timeout=30)
        p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                            "--format=csv,noheader"], capture_output=True, text=True,
                           timeout=30)
    except FileNotFoundError:
        return {"status": "ABSENT", "detail": "nvidia-smi not found", "gpus": [],
                "processes": []}
    except Exception as exc:                                        # noqa: BLE001
        return {"status": "UNAVAILABLE", "detail": f"{type(exc).__name__}: {exc}",
                "gpus": [], "processes": []}
    if g.returncode != 0 or p.returncode != 0:
        return {"status": "UNAVAILABLE",
                "detail": f"nvidia-smi exited {g.returncode}/{p.returncode}: "
                          f"{(g.stderr or p.stderr).strip()}", "gpus": [], "processes": []}
    gpus = [ln.strip() for ln in g.stdout.splitlines() if ln.strip()]
    procs = []
    for ln in p.stdout.splitlines():
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) >= 2 and parts[0]:
            procs.append({"pid": parts[0], "name": parts[1],
                          "used_memory": parts[2] if len(parts) > 2 else None})
    if not gpus:
        return {"status": "ABSENT", "detail": "no GPU listed", "gpus": [], "processes": procs}
    return {"status": "OK", "gpus": gpus, "processes": procs}


def gpu_verdict(probe, allowlist, self_pid=None):
    """(ok, reason). Busy = any compute process whose basename is not allowlisted,
    other than this process itself (`self_pid`): a caller that has already sampled
    once holds its own CUDA context, and that is not another workload."""
    if not isinstance(probe, dict) or probe.get("status") not in ("OK", "ABSENT",
                                                                  "UNAVAILABLE"):
        return False, f"GPU probe returned an unrecognised result: {probe!r}"
    if probe["status"] != "OK":
        return False, f"GPU {probe['status']}: {probe.get('detail', '')}"
    allowed = set(allowlist)
    busy = [pr for pr in probe.get("processes", [])
            if os.path.basename(str(pr.get("name", ""))) not in allowed
            and not (self_pid is not None and str(pr.get("pid", "")) == str(self_pid))]
    if busy:
        return False, f"GPU busy: compute process(es) not in the owner's allowlist " \
                      f"{sorted(allowed)}: {busy}"
    return True, None


# ---------------------------------------------------------------------------
# the decision
# ---------------------------------------------------------------------------

def decide(phases, cap, compare):
    """Pure. (verdict, reasons). REFUSE iff a compared figure is strictly above cap."""
    key = "projection" if compare == "point" else "upper95"
    reasons = []
    for p in phases:
        fig = p[key]
        if fig > cap:
            reasons.append(
                f"phase {p['phase']} ({p['family_name']}): {compare} figure {fig:.1f} > "
                f"cap {cap} — projection {p['projection']:.1f}, 95% interval "
                f"[{p['lower95']:.1f}, {p['upper95']:.1f}], {p['hits']} hits in "
                f"{p['sample_seeds']} sampled seeds")
    return (V_REFUSE if reasons else V_PROCEED), reasons


def _git_head(repo_root):
    try:
        return subprocess.run(["git", "-C", repo_root, "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30).stdout.strip() \
            or None
    except Exception:                                               # noqa: BLE001
        return None


def _parse_params(arg):
    if arg.startswith("@"):
        with open(arg[1:], encoding="utf-8") as fh:
            arg = fh.read()
    return _loads(arg)


def split_params(params):
    """(geometry, seed_start, seed_end_exclusive). Raises GeometryError."""
    if not isinstance(params, dict):
        raise SVE.GeometryError("--params is not a JSON object")
    for k in SEED_KEYS:
        if k not in params:
            raise SVE.GeometryError(f"{k} missing")
    s, e = params["seed_start"], params["seed_end_exclusive"]
    SVE.validate_range(s, e)
    if "max_seeds" in params and not (_is_int(params["max_seeds"])
                                      and e == s + params["max_seeds"]):
        raise SVE.GeometryError(f"seed_end_exclusive {e} != seed_start {s} + max_seeds "
                                f"{params['max_seeds']!r}")
    geometry = {k: v for k, v in params.items() if k not in SEED_KEYS + ("max_seeds",)}
    return geometry, s, e


def sample_plan(geometry, sample_seeds, phase4_seeds):
    from miner.range_miner_coordinator import workflow_stages_for
    stages = workflow_stages_for(geometry["prng_type"], geometry["test_both_modes"])
    return {ph: (phase4_seeds if ph == 4 else sample_seeds) for _f, ph in stages}


def evaluate(params_arg, *, repo_root=REPO_ROOT, runner=None, sampler=None,
             gpu_probe=None, sample_seeds=None, phase4_seeds=None, timeout_s=1800.0,
             clock=None):
    """Returns the evidence record (a dict) whose `verdict` decides the exit code."""
    now = (clock or (lambda: datetime.datetime.now(datetime.timezone.utc)))()
    rec = {"gate": GATE, "verdict_kind": "LOAD (survivor volume) — not a quality score",
           "utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "repository_commit": _git_head(repo_root),
           "params": None, "cap_file": CAP_REL, "cap_sha256": None, "cap_contents": None,
           "gpu": None, "estimate": None, "verdict": None, "reasons": []}

    def done(verdict, reasons):
        rec["verdict"] = verdict
        rec["reasons"] = list(reasons)
        return rec

    doc, cap_sha, cap_text, why = load_cap(repo_root, runner)
    if why:
        return done(V_REFUSE, [why])
    rec["cap_sha256"], rec["cap_contents"] = cap_sha, cap_text
    try:
        params = _parse_params(params_arg)
    except Exception as exc:                                        # noqa: BLE001
        return done(V_REFUSE, [f"--params unparseable: {exc}"])
    rec["params"] = params
    try:
        geometry, s, e = split_params(params)
        SVE.validate_geometry(geometry)
    except SVE.EstimatorError as exc:
        return done(V_REFUSE, [f"parameters: {exc}"])

    n = doc["min_sample_seeds"] if sample_seeds is None else sample_seeds
    n4 = n * DEFAULT_PHASE4_FACTOR if phase4_seeds is None else phase4_seeds
    if not (_is_int(n) and _is_int(n4)):
        return done(V_REFUSE, ["sample sizes are not integers"])
    if min(n, n4) < doc["min_sample_seeds"]:
        return done(V_REFUSE, [f"sample size {min(n, n4)} < min_sample_seeds "
                               f"{doc['min_sample_seeds']}"])
    try:
        plan = sample_plan(geometry, n, n4)
    except Exception as exc:                                        # noqa: BLE001
        return done(V_REFUSE, [f"parameters: workflow table refused: {exc}"])

    probe = (gpu_probe or nvidia_probe)()
    rec["gpu"] = probe
    rec["gpu_self_pid_excluded"] = os.getpid()
    ok, why = gpu_verdict(probe, doc["gpu_process_allowlist"], self_pid=os.getpid())
    if not ok:
        return done(V_UNAVAILABLE, [why])

    try:
        est = SVE.estimate_volume(geometry, s, e, sample_seeds=plan, sampler=sampler,
                                  timeout_s=timeout_s)
    except SVE.SamplerUnavailable as exc:
        return done(V_UNAVAILABLE, [f"sampler: {exc}"])
    except (SVE.SamplerFailed, SVE.SamplerTimeout, SVE.SamplerMalformed) as exc:
        return done(V_ERROR, [f"sampler: {type(exc).__name__}: {exc}"])
    except (SVE.GeometryError, SVE.SampleError, SVE.ResidueMismatch) as exc:
        return done(V_REFUSE, [f"{type(exc).__name__}: {exc}"])
    except Exception as exc:                                        # noqa: BLE001
        return done(V_ERROR, [f"estimator raised {type(exc).__name__}: {exc}"])
    rec["estimate"] = est
    for p in est["phases"]:
        if p["sample_seeds"] < doc["min_sample_seeds"]:
            return done(V_REFUSE, [f"phase {p['phase']}: sample {p['sample_seeds']} < "
                                   f"min_sample_seeds {doc['min_sample_seeds']}"])
    verdict, reasons = decide(est["phases"], doc["cap_survivors_per_phase"],
                              doc["compare"])
    rec["compare"] = doc["compare"]
    rec["cap_survivors_per_phase"] = doc["cap_survivors_per_phase"]
    return done(verdict, reasons)


def _report(rec):
    v = rec["verdict"]
    est = rec.get("estimate")
    if est:
        print(f"  range [{est['seed_start']}, {est['seed_end_exclusive']})  residue "
              f"{est['residue_sha256'][:16]}…  dataset {est['dataset_sha256'][:16]}…",
              flush=True)
        for p in est["phases"]:
            print(f"  phase {p['phase']} {p['family_name']:<24} hits {p['hits']:>8} / "
                  f"{p['sample_seeds']:>11} sampled  (kernel {p['kernel_hits']})  "
                  f"projection {p['projection']:>14.1f}  95% "
                  f"[{p['lower95']:.1f}, {p['upper95']:.1f}]  {p['seconds']:.1f}s",
                  flush=True)
    if v == V_PROCEED:
        print(f"SURVIVOR-VOLUME GATE : PROCEED (load verdict) — every phase at or below "
              f"the cap {rec.get('cap_survivors_per_phase')} under '{rec.get('compare')}'",
              flush=True)
        return EXIT_PROCEED
    print(f"SURVIVOR-VOLUME GATE : {v} (load verdict) — {len(rec['reasons'])} reason(s):",
          flush=True)
    for r in rec["reasons"]:
        print(f"  * {r}", flush=True)
    return EXIT_REFUSE


def _write_record(path, rec):
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")


def main(argv=None, *, repo_root=REPO_ROOT, runner=None, sampler=None, gpu_probe=None,
         clock=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--params", required=True, help="JSON object, or @file")
    ap.add_argument("--record-out", default=None, help="evidence record (JSON)")
    ap.add_argument("--sample-seeds", type=int, default=None,
                    help="seeds sampled per phase (default: the cap file's minimum)")
    ap.add_argument("--phase4-sample-seeds", type=int, default=None,
                    help=f"seeds sampled in phase 4 (default {DEFAULT_PHASE4_FACTOR}x)")
    ap.add_argument("--sampler-timeout-s", type=float, default=1800.0)
    args = ap.parse_args(argv)
    try:
        rec = evaluate(args.params, repo_root=repo_root, runner=runner, sampler=sampler,
                       gpu_probe=gpu_probe, sample_seeds=args.sample_seeds,
                       phase4_seeds=args.phase4_sample_seeds,
                       timeout_s=args.sampler_timeout_s, clock=clock)
    except Exception as exc:                                        # noqa: BLE001
        rec = {"gate": GATE, "verdict": V_ERROR,
               "reasons": [f"gate raised {type(exc).__name__}: {exc}"]}
    try:
        _write_record(args.record_out, rec)
    except Exception as exc:                                        # noqa: BLE001
        print(f"SURVIVOR-VOLUME GATE : ERROR — record not written: {exc}", flush=True)
        return EXIT_REFUSE
    return _report(rec)


if __name__ == "__main__":
    sys.exit(main())
