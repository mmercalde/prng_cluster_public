#!/usr/bin/env python3
"""
why_report_runner.py - unit W1 of PLAN_LAND_LL7_LL8_20261126 §6 (TODO-WHY-1, trigger half), 2026-11-26.
ADVISORY. Stand-alone; callable by hand, later by WATCHER (W2, owner lands by hand).

For ONE open why-report flag (why_report_flags.py; raised by training_health_check._raise_why_flag):
  1. resolve the inputs
       checkpoint  the checkpoint the flagged family's model loads NOW: the sidecar in --models-dir
                   (default: agent_manifests/prediction.json default_params.models_dir under --root),
                   models/inference_bundle.resolve_checkpoint_path, sha256. Sidecar missing, of another
                   family, signal gate blocked, or checkpoint missing => no_checkpoint (lead D-2).
       block       the LATEST graded walk-forward block (highest block end, then start, then origin)
                   under the --wf-root dirs whose origin_summary.json records, for the flagged family,
                   frozen_families[s<seed>][family].checkpoint_sha256 == that sha256 (lead D-1), with a
                   vote arm of that family (role "family" first, else "learned") and its grade dirs.
                   None => no_graded_block. A --wf-root is a walk-forward run dir (origin_*/ inside) or
                   one origin dir.
       survivors   <origin>/step3/survivors_with_scores.json, sha256 == labels.survivors_with_scores_sha256
                   (missing => missing_survivors; differs => input_sha_mismatch). The frozen model files,
                   the arm's pools.jsonl and each grade dir's per_draw.jsonl (against its SHA256SUMS) are
                   checked the same way (input_sha_mismatch; a recorded file that is gone counts as a
                   mismatch).
       votes       the block's full vote matrix, rebuilt by scripts/diagnostics/e2e_vote_npz.py (child
                   process) from an arm's pools.jsonl and run_record.json; every pool record is checked
                   against the rebuilt matrix by that script. A refusal => spec_build_error.
                   Which arm (second pass, one rule for every family, votes_arm_for): the block's
                   UNLEARNED arm of the same pool size when the origin has one (V does not depend on the
                   ranking, so no model is re-scored; a card-voted neural_net arm rebuilt on CPU did not
                   match its recorded pools); else the flagged family's own arm. Recorded as votes_source
                   in the spec (votes.source), the result and the attach record. The grade dirs stay the
                   family arm's.
  2. write the LL-8 spec (<out>/spec.json; analysis settings: ANALYSIS_DEFAULTS, overridable)
  3. run LL-8 (scripts/diagnostics/attribution_cycle.py) as a CHILD process, CPU only, inside the time
     limit (one deadline for the vote rebuild and LL-8 together)
  4. on success attach the report (attribution_cycle.json) with why_report_flags.attach(), passing its
     sha256 (attach re-checks it), so the flag closes
  5. on ANY failure record a failed attempt (why_report_flags.record_attempt): the flag shows as
     why_report_failed with the reason. A flag is NEVER closed on a failure.
Every attempt, ok or failed, is one line in diagnostics_outputs/why_report_attempts.jsonl.

Failure reasons (lead D-5, fixed set): REASONS below.
One attempt per flag per run_open_flags() call; no retry inside a call (lead D-3).
Writes only under its out dir (default <this tree>/data/why_reports/<flag id safe name>/<UTC stamp>/;
--out-root replaces <this tree>/data/why_reports) and the append-only records under
<root>/diagnostics_outputs/. Changes no setting (lead D-4). An --out-root outside <this tree>/data/ is a bad
argument (CLI exit 2; run_for_flag: spec_build_error), refused before anything is created. No bytecode is
written (sys.dont_write_bytecode before the first project import; the children run with -B).
Third pass (Beta X-W1-RUNNER R1): every grade dir is bound to the origin-pinned grader summary (arm, pool size,
dataset, block, sessions; check_grade_binding); a pooled-only grade is analysed over the dataset sessions; a
report with no graded draw in any cell fails no_graded_draws; analysis may set only ANALYSIS_KEYS; a child's
process group is killed whenever the child ends; the FAILED log line can fail without losing the attempt.

CLI:
  python3 scripts/diagnostics/why_report_runner.py (--flag-id ID | --all-open) --wf-root DIR [--wf-root DIR]
          [--root DIR] [--out-root DIR] [--models-dir DIR] [--timeout-s N] [--json]
  exit 0: every attempted flag got a report; 1: any attempt failed, an attempt was not recorded, or the flags
  could not be read; 2: bad arguments (including --analysis-json keys outside ANALYSIS_KEYS and an --out-root
  outside <this tree>/data/).
"""
import argparse
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# [third pass, Beta X-W1-RUNNER R1 Q4] the documented plain CLI writes no bytecode: set before the first
# project import (why_report_flags below, models.inference_bundle later)
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import why_report_flags as wrf  # noqa: E402

LL8 = REPO / "scripts" / "diagnostics" / "attribution_cycle.py"
VOTE_NPZ = REPO / "scripts" / "diagnostics" / "e2e_vote_npz.py"
DEFAULT_OUT_BASE = REPO / "data" / "why_reports"
MANIFEST_REL = "agent_manifests/prediction.json"     # default_params.models_dir (Step 6's models dir)
BY = "why_report_runner (W1)"
FORMAT = "tfm.w1.why_report_runner.result.v1"
REASONS = ("no_graded_block", "no_checkpoint", "missing_survivors", "input_sha_mismatch", "cycle_refused",
           "cycle_error", "timeout", "killed", "spec_build_error", "attach_error", "internal_error",
           "no_graded_draws")
# LL-8 analysis settings. Source: the lead's hand LL-8 run on walk-forward block 0
# (ch14-learn data/ch14_learn/why_block0/spec_<family>.json, 2026-11-24; untracked) - the only real
# LL-8 spec on record. Overridable per call (analysis=..., --analysis-json).
ANALYSIS_DEFAULTS = {"per_run_prefixes": ["global_"],
                     "abstain": {"min_pairs": 20, "min_draws": 10, "min_survivors": 5},
                     "band_min_survivors": 10, "bootstrap": {"B": 2000}, "seed": 20261124,
                     "attribution": {"ig_steps": 64}, "a_norm_atol": 1e-05}
ANALYSIS_KEYS = tuple(ANALYSIS_DEFAULTS)   # the ONLY keys analysis / --analysis-json may set (Beta R1 Q3)
POLL_S = 0.1

logger = logging.getLogger("why_report_runner")


class RunnerFailure(Exception):
    def __init__(self, reason, message, error_class=None):
        assert reason in REASONS, reason
        super().__init__(message)
        self.reason, self.error_class = reason, error_class or "RunnerFailure"


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def safe_name(flag_id):
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(flag_id))[:120]


def _log(log_path, msg):
    with open(log_path, "a") as f:
        f.write(f"{_utc()} {msg}\n")


def check_analysis(analysis):
    """analysis may override ONLY the declared analysis keys (ANALYSIS_KEYS). The block, grade dirs,
    sessions, pool sizes, survivors, votes and model bindings always come from the archive (Beta R1 Q3)."""
    if analysis is None:
        return {}
    if not isinstance(analysis, dict):
        raise RunnerFailure("spec_build_error", f"analysis must be a JSON object, got {type(analysis).__name__}",
                            "BadAnalysis")
    bad = sorted(k for k in analysis if k not in ANALYSIS_KEYS)
    if bad:
        raise RunnerFailure("spec_build_error", f"analysis keys {bad} are not analysis settings; only "
                                                f"{list(ANALYSIS_KEYS)} may be overridden", "BadAnalysis")
    return dict(analysis)


def check_out_base(out_root):
    """The resolved out base must lie inside <this tree>/data/ (plan's data/ boundary). Checked BEFORE
    anything is created (Beta R1 Q4)."""
    base = (Path(out_root) if out_root else DEFAULT_OUT_BASE).resolve()
    data = (REPO / "data").resolve()
    if base != data and data not in base.parents:
        raise RunnerFailure("spec_build_error", f"out root {base} is not inside {data}/", "OutRootOutsideData")
    return base


# ── 1. inputs ───────────────────────────────────────────────────────────────

def default_models_dir(root):
    p = Path(root or ".") / MANIFEST_REL
    try:
        return json.loads(p.read_text())["default_params"]["models_dir"]
    except Exception as e:
        raise RunnerFailure("no_checkpoint", f"models dir not given and {p} has no default_params.models_dir: "
                                             f"{type(e).__name__}: {e}")


def current_checkpoint(family, root=None, models_dir=None):
    """The checkpoint the flagged family's model loads now (lead D-2: anything unusable => no_checkpoint)."""
    from models.inference_bundle import resolve_checkpoint_path
    md = Path(models_dir if models_dir is not None else default_models_dir(root))
    if not md.is_absolute():
        md = Path(root or ".") / md
    meta_p = md / "best_model.meta.json"
    if not meta_p.is_file():
        raise RunnerFailure("no_checkpoint", f"sidecar missing: {meta_p}")
    try:
        meta = json.loads(meta_p.read_text())
    except Exception as e:
        raise RunnerFailure("no_checkpoint", f"sidecar unreadable: {meta_p}: {e}", type(e).__name__)
    if meta.get("model_type") != family:
        raise RunnerFailure("no_checkpoint", f"{meta_p} holds model_type {meta.get('model_type')!r}, not the "
                                             f"flagged family {family!r}")
    sq = meta.get("signal_quality") or {}
    if not sq.get("prediction_allowed", True):
        raise RunnerFailure("no_checkpoint", f"sidecar signal gate blocks prediction: {sq.get('signal_status')}")
    if not meta.get("checkpoint_path"):
        raise RunnerFailure("no_checkpoint", f"{meta_p}: checkpoint_path is null")
    try:
        ck = resolve_checkpoint_path(str(md), meta)
    except Exception as e:
        raise RunnerFailure("no_checkpoint", str(e), type(e).__name__)
    return {"models_dir": str(md), "checkpoint_path": str(ck), "checkpoint_sha256": _sha(ck)}


def _summaries(wf_roots):
    for r in wf_roots or []:
        r = Path(r)
        if (r / "origin_summary.json").is_file():
            yield r / "origin_summary.json"
        else:
            yield from sorted(r.glob("origin_*/origin_summary.json"))


def find_block(family, ck_sha, wf_roots):
    """The latest graded walk-forward block for `family` whose frozen checkpoint sha256 is ck_sha (D-1)."""
    best, looked, unreadable = None, 0, []
    for sp in _summaries(wf_roots):
        looked += 1
        try:
            s = json.loads(sp.read_text())
        except Exception as e:   # surfaced: logged at ERROR, and named in a no_graded_block message
            unreadable.append(f"{sp}: {type(e).__name__}")
            logger.error("why-report runner: walk-forward summary %s unreadable, not considered: %r", sp, e)
            continue
        if not s.get("finished") or not s.get("grades"):
            continue
        for seed_key, fams in (s.get("frozen_families") or {}).items():
            f = (fams or {}).get(family)
            if not f or f.get("checkpoint_sha256") != ck_sha:
                continue
            arms = [v for v in s.get("votes") or [] if v.get("family") == family
                    and v.get("role") in ("family", "learned") and f"s{v.get('split_seed')}" == seed_key]
            arms.sort(key=lambda v: (v["role"] != "family", -int(v.get("pool_size") or 0)))
            for v in arms:
                gr = [g for g in s["grades"] if g.get("arm_id") == v["arm_id"]]
                sel = [g for g in gr if g.get("session") != "pooled"] or gr
                gdirs = [g["dir"] for g in sel]
                if not gdirs:
                    continue
                b = s["block"]
                key = (int(b["end_index"]), int(b["start_index"]), int(s.get("origin", -1)), str(sp))
                if best is None or key > best["key"]:
                    best = {"key": key, "summary_path": str(sp), "origin_dir": str(sp.parent), "summary": s,
                            "seed_key": seed_key, "frozen": f, "arm": v, "grade_dirs": gdirs, "grades": sel,
                            "votes_arm": votes_arm_for(s, v)}
                break
    if best is None:
        raise RunnerFailure("no_graded_block",
                            f"no graded walk-forward block for {family} with checkpoint sha256 {ck_sha} "
                            f"({looked} origin summaries under {list(map(str, wf_roots or []))}"
                            + (f"; unreadable: {unreadable}" if unreadable else "") + ")")
    return best


def votes_arm_for(summary, arm):
    """The arm the votes matrix V is rebuilt from (lead decision, second pass; one rule for every family).
    V does not depend on the ranking (e2e_vote_npz.py docstring: row i of V is survivor row i of the
    Step 3 file), so the block's UNLEARNED arm of the same pool size is used when the origin has one: it
    needs no model, so a card-voted neural_net arm is never re-scored on CPU. Only when no unlearned arm
    of that pool size exists is V rebuilt from the flagged family's own arm. The grade dirs stay the
    family arm's (find_block)."""
    un = [u for u in summary.get("votes") or [] if u.get("role") == "unlearned"
          and int(u.get("pool_size") or 0) == int(arm.get("pool_size") or 0) and u.get("dir")]
    if un:
        return dict(un[0], votes_source="unlearned")
    return dict(arm, votes_source="family_arm_fallback")


def check_inputs(cand):
    """sha256 checks of every recorded input (missing survivors => missing_survivors; else input_sha_mismatch)."""
    s, od = cand["summary"], Path(cand["origin_dir"])
    surv = od / "step3" / "survivors_with_scores.json"
    if not surv.is_file():
        raise RunnerFailure("missing_survivors", f"Step 3 survivors file missing: {surv}")
    want = (s.get("labels") or {}).get("survivors_with_scores_sha256")
    got = _sha(surv)
    if got != want:
        raise RunnerFailure("input_sha_mismatch", f"{surv}: sha256 {got} != recorded {want}")
    fz = cand["frozen"]
    for name, w in (fz.get("sha256") or {}).items():
        p = Path(fz["dir"]) / name
        if not p.is_file() or _sha(p) != w:
            raise RunnerFailure("input_sha_mismatch", f"frozen model file {p} missing or sha256 != recorded {w}")
    arms = [cand["arm"]] + ([cand["votes_arm"]] if cand.get("votes_arm", {}).get("arm_id")
                            not in (None, cand["arm"]["arm_id"]) else [])
    for a in arms:   # the family arm (its grades are read) and the arm V is rebuilt from
        pools = Path(a["dir"]) / "pools.jsonl"
        if not pools.is_file() or _sha(pools) != a.get("pools_sha256"):
            raise RunnerFailure("input_sha_mismatch", f"{pools} missing or sha256 != recorded "
                                                      f"{a.get('pools_sha256')}")
    for gd in cand["grade_dirs"]:
        pd, sums = Path(gd) / "per_draw.jsonl", Path(gd) / "SHA256SUMS"
        if not pd.is_file() or not sums.is_file():
            raise RunnerFailure("input_sha_mismatch", f"{gd}: per_draw.jsonl or SHA256SUMS missing")
        listed = dict(reversed(l.split()) for l in sums.read_text().splitlines() if l.strip())
        if listed.get("per_draw.jsonl") != _sha(pd):
            raise RunnerFailure("input_sha_mismatch", f"{pd}: sha256 not as in SHA256SUMS")
    check_grade_binding(cand)
    return {"path": str(surv), "sha256": got}


def check_grade_binding(cand):
    """[third pass, Beta R1 Q3] Each grade dir belongs to the selected arm, block and dataset: its grader
    summary.json has the sha256 the origin summary pinned (e2e_walk_forward.py grades[].summary_sha256), and
    that summary records (e2e_grader.py write_outputs / summary "inputs", "declared", "grade_block") the
    per_draw.jsonl sha256 on disk, the arm's pools.jsonl sha256, exactly this arm with max_k == its pool size,
    the walk-forward run's dataset sha256 (run_record inputs.dataset), and this block's range and sessions
    (pooled: the dataset sessions). Any miss, or a missing summary => input_sha_mismatch."""
    arm, b, fr = cand["arm"], cand["summary"]["block"], cand["frame"]
    for g in cand["grades"]:
        gd = Path(g["dir"])
        sp = gd / "summary.json"
        if not sp.is_file():
            raise RunnerFailure("input_sha_mismatch", f"{gd}: grader summary.json missing")
        want = g.get("summary_sha256")
        if not want or _sha(sp) != want:
            raise RunnerFailure("input_sha_mismatch", f"{sp}: sha256 != the origin summary's pin {want}")
        try:
            gs = json.loads(sp.read_text())
            inp, gb = gs.get("inputs") or {}, gs.get("grade_block") or {}
            got = {"per_draw": ((gs.get("outputs") or {}).get("per_draw.jsonl") or {}).get("sha256"),
                   "pools": (inp.get("pools") or {}).get("sha256"),
                   "dataset": (inp.get("dataset") or {}).get("sha256"),
                   "arms": [(a.get("arm_id"), a.get("max_k")) for a in (gs.get("declared") or {}).get("arms") or []],
                   "range": [gb.get("start_index"), gb.get("end_index")],
                   "sessions": sorted(gb.get("sessions") or [])}
        except Exception as e:
            raise RunnerFailure("input_sha_mismatch", f"{sp}: unreadable bindings: {type(e).__name__}: {e}",
                                type(e).__name__)
        sess = fr.get("dataset_sessions") if g.get("session") == "pooled" else [g.get("session")]
        need = {"per_draw": _sha(gd / "per_draw.jsonl"), "pools": arm.get("pools_sha256"),
                "dataset": fr.get("dataset_sha256"),
                "arms": [(arm["arm_id"], int(arm.get("pool_size") or 0))],
                "range": [int(b["start_index"]), int(b["end_index"])], "sessions": sorted(sess or [])}
        off = [k for k in need if need[k] is None or not need[k] or got[k] != need[k]]
        if g.get("arm_id") != arm["arm_id"] or int(g.get("pool_size") or 0) != int(arm.get("pool_size") or 0):
            off.append("origin grade entry arm/pool")
        if off:
            raise RunnerFailure("input_sha_mismatch", f"{gd}: grade not bound to arm {arm['arm_id']} / block "
                                                      f"{need['range']} / dataset: {off} differ "
                                                      f"(got {[(k, got.get(k)) for k in off if k in got]})")


def wf_frame(cand):
    """pool_sizes and graded sessions from the walk-forward run_record.json (origin dir's parent)."""
    rp = Path(cand["origin_dir"]).parent / "run_record.json"
    try:
        rr = json.loads(rp.read_text())
        fr = rr["frame"]
        sessions = [x for x in fr["grade_sessions"] if x != "pooled"]
        ds = fr.get("dataset_sessions")
        return {"pool_sizes": list(fr["pool_sizes"]), "sessions": sessions, "run_record": str(rp),
                "dataset_sessions": list(ds) if ds else None,
                "dataset_sha256": ((rr.get("inputs") or {}).get("dataset") or {}).get("sha256")}
    except Exception as e:
        raise RunnerFailure("spec_build_error", f"walk-forward frame unreadable from {rp}: {type(e).__name__}: {e}",
                            type(e).__name__)


# ── child processes ─────────────────────────────────────────────────────────

def _kill_group(pgid):
    """SIGKILL whatever is left of the child's process group. Called while the leader is still unreaped,
    so its pid (== the group id) cannot have been reused."""
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_child(cmd, log_path, deadline, cwd=None):
    """Run cmd as a child (own session), stdout+stderr appended to log_path, killed at `deadline`
    (time.monotonic()). Returns dict(rc, signal, timed_out, wall_s, maxrss_mib).
    [third pass, Beta R1 Q1] However the child ends (normal exit, failure, signal, time limit, or this wait
    being interrupted) what is left of its process group is killed, then the direct child is reaped: the
    exit is seen with WNOWAIT (the leader stays a zombie, so its group id cannot be reused) before the kill."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    t0 = time.monotonic()
    with open(log_path, "a") as lf:
        lf.write(f"{_utc()} $ {' '.join(map(str, cmd))}\n")
        lf.flush()
        p = subprocess.Popen([str(c) for c in cmd], stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT,
                             cwd=str(cwd or REPO), env=env, start_new_session=True)
        timed_out = False
        try:
            while True:
                if os.waitid(os.P_PID, p.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
                    break                                  # exited; not reaped yet
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                time.sleep(POLL_S)
        finally:
            _kill_group(p.pid)                             # every ending: the group's leftovers die
            pid, status, ru = os.wait4(p.pid, 0)           # then the direct child is reaped
        p.returncode = os.waitstatus_to_exitcode(status)
    rc = p.returncode
    return {"rc": rc, "signal": -rc if rc < 0 else None, "timed_out": timed_out,
            "wall_s": round(time.monotonic() - t0, 2), "maxrss_mib": round(ru.ru_maxrss / 1024, 1)}


def _child_failure(what, r, log_path):
    if r["timed_out"]:
        return RunnerFailure("timeout", f"{what} killed at the time limit after {r['wall_s']} s; log {log_path}",
                             "TimeoutExpired")
    if r["signal"] is not None:
        return RunnerFailure("killed", f"{what} killed by signal {r['signal']} (not the runner's time limit; "
                                       f"out of memory?); log {log_path}", f"Signal{r['signal']}")
    return None


def build_votes_npz(cand, out, log_path, deadline):
    """Default votes builder: e2e_vote_npz.py as a child on the arm's own pool file (it refuses on any
    mismatch between the rebuilt matrix and a recorded pool)."""
    b, v = cand["summary"]["block"], cand.get("votes_arm") or cand["arm"]
    vs = {"survivors": cand["survivors"], "vote_range": [int(b["start_index"]), int(b["end_index"])],
          "arms": {v["arm_id"]: {"pool_files": [{"run_dir": v["dir"], "pools_sha256": v["pools_sha256"],
                                                 "pool_size": int(v["pool_size"])}]}}}
    sp = out / "votes_spec.json"
    sp.write_text(json.dumps(vs, indent=1))
    npz = out / "votes.npz"
    r = run_child([sys.executable, "-u", "-B", VOTE_NPZ, "--spec", sp, "--out", npz], log_path, deadline)
    cand["children"]["vote_npz"] = r
    f = _child_failure("e2e_vote_npz", r, log_path)
    if f:
        raise f
    if r["rc"] != 0 or not npz.is_file():
        raise RunnerFailure("spec_build_error", f"e2e_vote_npz exited {r['rc']} (votes matrix not rebuilt); "
                                                f"log {log_path}", "VoteNpzFailed")
    return npz


def votes_source(cand):
    """Which arm V was rebuilt from (recorded in the spec, the result and the attach record)."""
    va = cand.get("votes_arm") or dict(cand["arm"], votes_source="family_arm_fallback")
    return {"arm_id": va.get("arm_id"), "role": va.get("role"), "pool_size": va.get("pool_size"),
            "source": va.get("votes_source"), "grade_arm_id": cand["arm"]["arm_id"]}


def build_spec(flag_id, family, ck, cand, out, analysis):
    fr = cand.get("frame") or wf_frame(cand)
    b = cand["summary"]["block"]
    # [third pass, Beta R1 Q3] a pooled-only grade: the sessions are the dataset's own, as the walk-forward
    # recorded them (run_record frame.dataset_sessions; e2e_walk_forward grades "pooled" over them)
    if any(g.get("session") != "pooled" for g in cand.get("grades") or [{}]):
        sessions = fr["sessions"]
    else:
        sessions = fr.get("dataset_sessions")
        if not sessions:
            raise RunnerFailure("spec_build_error", f"pooled-only grade but {fr['run_record']} records no "
                                                    f"frame.dataset_sessions", "NoDatasetSessions")
    o = cand["summary"].get("origin")
    spec = {"label": f"DEVELOPMENT, walk-forward origin {o} block [{b['start_index']},{b['end_index']}); "
                     f"automatic why-report for flag {flag_id} (W1); advisory, no verdict",
            "written_by": BY, "written_at": _utc(), "flag_id": flag_id,
            "walk_forward": {"origin_summary": cand["summary_path"], "run_record": fr["run_record"],
                             "arm_id": cand["arm"]["arm_id"], "split_seed": cand["seed_key"]},
            "survivors": cand["survivors"],
            "models_dir": cand["frozen"]["dir"],
            "expect_checkpoint_sha256": ck["checkpoint_sha256"],
            "current_model": ck,
            "votes": {"path": str(cand["votes_npz"]), "sha256": _sha(cand["votes_npz"]),
                      "source": votes_source(cand)},
            "grade_dirs": list(cand["grade_dirs"]),
            "blocks": {f"block{o}": [int(b["start_index"]), int(b["end_index"])]},
            "sessions": list(sessions), "pool_sizes": fr["pool_sizes"]}
    spec.update(ANALYSIS_DEFAULTS)
    spec.update(check_analysis(analysis))       # only ANALYSIS_KEYS; checked again here (Beta R1 Q3)
    p = out / "spec.json"
    p.write_text(json.dumps(spec, indent=1))
    return p


# ── one flag ────────────────────────────────────────────────────────────────

def run_for_flag(flag_id, wf_roots, root=None, out_root=None, timeout_s=1800, models_dir=None, analysis=None,
                 votes_builder=None, ll8_cmd=None, by=BY):
    """One attempt for one open flag. Never raises. Returns the result dict (status ok | failed | skipped)."""
    started = _utc()
    t0 = time.monotonic()
    deadline = t0 + float(timeout_s)
    res = {"format": FORMAT, "flag_id": flag_id, "status": None, "reason": None, "error_class": None,
           "message": None, "started_at": started, "ended_at": None, "out_dir": None, "log_path": None,
           "report_path": None, "report_sha256": None, "attempt": None, "children": {}}
    out = log_path = None
    try:
        rfl = wrf.read_flags(root)
        g = rfl["groups"].get(flag_id)
        if g is None:
            raise ValueError(f"unknown flag_id {flag_id!r}")
        if not g["open"]:
            res.update(status="skipped", message="flag already has a why-report attached")
            return res
        if rfl.get("attempts_error"):    # no attempt could be numbered or recorded: nothing is run
            raise RunnerFailure("internal_error", f"attempts file unreadable, nothing run: {rfl['attempts_error']}",
                                "AttemptsUnreadable")
        family = g["model_type"]
        analysis = check_analysis(analysis)           # before anything is created or run (Beta R1 Q3)
        out = check_out_base(out_root)                # before anything is created (Beta R1 Q4)
        out = out / safe_name(flag_id) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        out.mkdir(parents=True, exist_ok=False)
        log_path = out / "runner.log"
        res.update(out_dir=str(out), log_path=str(log_path))
        _log(log_path, f"flag {flag_id} family {family}; wf roots {list(map(str, wf_roots or []))}; "
                       f"timeout {timeout_s}s")
        ck = current_checkpoint(family, root, models_dir)
        res["checkpoint"] = ck
        cand = find_block(family, ck["checkpoint_sha256"], wf_roots)
        cand["children"] = res["children"]
        res["block"] = {"origin_summary": cand["summary_path"], "arm_id": cand["arm"]["arm_id"],
                        "block": cand["summary"]["block"].get("block"),
                        "range": [cand["summary"]["block"]["start_index"], cand["summary"]["block"]["end_index"]]}
        res["votes_source"] = votes_source(cand)
        _log(log_path, f"block {res['block']}; votes from {res['votes_source']}")
        cand["frame"] = wf_frame(cand)
        cand["survivors"] = check_inputs(cand)
        try:
            cand["votes_npz"] = (votes_builder or build_votes_npz)(cand, out, log_path, deadline)
        except RunnerFailure:
            raise
        except Exception as e:
            raise RunnerFailure("spec_build_error", f"votes matrix: {type(e).__name__}: {e}", type(e).__name__)
        try:
            spec_p = build_spec(flag_id, family, ck, cand, out, analysis)
        except RunnerFailure:
            raise
        except Exception as e:
            raise RunnerFailure("spec_build_error", f"{type(e).__name__}: {e}", type(e).__name__)
        res["spec_path"] = str(spec_p)
        rf = out / "cycle_result.json"
        cmd = (ll8_cmd or [sys.executable, "-u", "-B", LL8]) + [
            "--spec", spec_p, "--out-dir", out / "cycle", "--model-family", family, "--flag-id", flag_id,
            "--result-file", rf]
        r = run_child(cmd, log_path, deadline)
        res["children"]["ll8"] = r
        f = _child_failure("LL-8 attribution cycle", r, log_path)
        if f:
            raise f
        cres = json.loads(rf.read_text()) if rf.is_file() else {}
        res["cycle_result_path"] = str(rf) if rf.is_file() else None
        if r["rc"] == 2:
            raise RunnerFailure("cycle_refused", cres.get("message") or f"LL-8 exit 2; log {log_path}",
                                cres.get("error_class") or "CycleError")
        if r["rc"] != 0:
            raise RunnerFailure("cycle_error", cres.get("message") or f"LL-8 exit {r['rc']}; log {log_path}",
                                cres.get("error_class") or f"Exit{r['rc']}")
        if cres.get("status") != "ok" or not cres.get("report_path") or not Path(cres["report_path"]).is_file():
            raise RunnerFailure("cycle_error", f"LL-8 exit 0 but result {rf} is not ok with a report: "
                                               f"{cres.get('status')!r}", "ResultNotOk")
        if cres.get("checkpoint_sha256") != ck["checkpoint_sha256"]:
            raise RunnerFailure("input_sha_mismatch", f"report checkpoint {cres.get('checkpoint_sha256')} != the "
                                                      f"model's {ck['checkpoint_sha256']}")
        rp = cres["report_path"]
        res["graded_draws_in_cells"] = n_cells = report_graded_draws(rp)
        if n_cells == 0:      # [third pass, Beta R1 Q3] a report with no graded draws closes nothing
            raise RunnerFailure("no_graded_draws", f"LL-8 ok but no graded draw entered any report cell "
                                                   f"(sessions {json.loads(spec_p.read_text()).get('sessions')}); "
                                                   f"report {rp}", "NoGradedDraws")
        rsha = _sha(rp)
        try:
            att = wrf.attach(flag_id, rp, by, root, note="automatic why-report (W1, LL-8); advisory",
                             report_sha256=rsha,
                             extra={"run_record_path": cres.get("run_record_path"),
                                    "run_record_sha256": _sha(cres["run_record_path"]),
                                    "priorities_path": cres.get("priorities_path"), "spec_path": str(spec_p),
                                    "checkpoint_sha256": ck["checkpoint_sha256"], "block": res["block"],
                                    "votes_source": res["votes_source"],
                                    "runner_log": str(log_path)})
        except Exception as e:
            raise RunnerFailure("attach_error", f"{type(e).__name__}: {e}", type(e).__name__)
        res.update(status="ok", report_path=rp, report_sha256=rsha, attachment=att)
        res["ended_at"] = _utc()
        try:
            res["attempt"] = wrf.record_attempt(flag_id, "ok", root, log_path=str(log_path), started_at=started,
                                                ended_at=res["ended_at"], report_path=rp, report_sha256=rsha,
                                                out_dir=str(out))["attempt"]
        except Exception as e:   # the report is attached; only the attempt line is missing: surfaced
            logger.error("why-report attached to %s but its attempt line was not written: %r", flag_id, e,
                         exc_info=True)
            res["attempt_record_error"] = repr(e)
    except Exception as e:
        if isinstance(e, RunnerFailure):
            reason, ecls = e.reason, e.error_class
        else:
            reason, ecls = "internal_error", type(e).__name__
            logger.error("why-report runner internal error on %s: %r", flag_id, e, exc_info=True)
            res["traceback_tail"] = traceback.format_exc()[-2000:]
        res.update(status="failed", reason=reason, error_class=ecls, message=str(e)[:500], ended_at=_utc())
        if log_path is not None:
            try:      # [third pass, Beta R1 Q1] a failed log append never costs the attempt or its reason
                _log(log_path, f"FAILED {reason} {ecls}: {str(e)[:500]}")
            except Exception as le:
                res["log_error"] = repr(le)
                logger.error("why-report runner: FAILED line not written to %s (%r); reason %s kept",
                             log_path, le, reason, exc_info=True)
                print(f"ERROR why-report runner: {flag_id}: FAILED line not written to {log_path}: {le!r} "
                      f"(attempt reason {reason} kept)", file=sys.stderr, flush=True)
        try:
            res["attempt"] = wrf.record_attempt(flag_id, "failed", root, reason=reason, error_class=ecls,
                                                message=str(e), log_path=str(log_path) if log_path else None,
                                                started_at=started, ended_at=res["ended_at"],
                                                out_dir=str(out) if out else None)["attempt"]
        except Exception as e2:
            logger.error("why-report attempt for %s (%s) could not be recorded: %r", flag_id, reason, e2,
                         exc_info=True)
            res["attempt_record_error"] = repr(e2)
            print(f"ERROR why-report runner: {flag_id}: attempt ({reason}) not recorded: {e2!r}", file=sys.stderr,
                  flush=True)
    finally:
        res["wall_s"] = round(time.monotonic() - t0, 2)
        if out is not None and out.is_dir():
            try:
                (out / "runner_result.json").write_text(json.dumps(res, indent=2, default=str))
            except Exception as e:
                logger.error("runner_result.json not written in %s: %r", out, e, exc_info=True)
    return res


def report_graded_draws(report_path):
    """Graded draws that entered the report's cells (LL-8 attribution_cycle.json "contrasts"[*].draws)."""
    try:
        cells = json.loads(Path(report_path).read_text()).get("contrasts") or {}
        return sum(int(c.get("draws") or 0) for c in cells.values() if isinstance(c, dict))
    except Exception as e:
        raise RunnerFailure("cycle_error", f"report {report_path} unreadable: {type(e).__name__}: {e}",
                            type(e).__name__)


def run_open_flags(wf_roots, root=None, **kw):
    """One attempt for every open flag (why_report_failed included), once each, oldest first (D-3).
    Never raises: an exception becomes internal_error on that flag, logged at ERROR with its traceback.
    [third pass, Beta R1 Q2] If the open flags cannot be read, the result is ONE failed entry (flag_id None,
    reason internal_error), so the CLI exits 1; it is not an empty list."""
    try:
        fids = list(wrf.open_flags(root))
    except Exception as e:
        logger.error("why-report runner could not read the open flags: %r", e, exc_info=True)
        print(f"ERROR why-report runner: open flags not readable: {e!r}", file=sys.stderr, flush=True)
        return [{"format": FORMAT, "flag_id": None, "status": "failed", "reason": "internal_error",
                 "error_class": type(e).__name__, "message": f"open flags not readable: {e}"[:500],
                 "attempt_record_error": "no flag to record against"}]
    results = []
    for fid in fids:
        try:
            results.append(run_for_flag(fid, wf_roots, root=root, **kw))
        except Exception as e:
            logger.error("why-report runner raised on %s: %r", fid, e, exc_info=True)
            r = {"format": FORMAT, "flag_id": fid, "status": "failed", "reason": "internal_error",
                 "error_class": type(e).__name__, "message": str(e)[:500]}
            try:
                r["attempt"] = wrf.record_attempt(fid, "failed", root, reason="internal_error",
                                                  error_class=type(e).__name__, message=str(e))["attempt"]
            except Exception as e2:
                logger.error("attempt for %s not recorded: %r", fid, e2, exc_info=True)
                r["attempt_record_error"] = repr(e2)
            results.append(r)
    return results


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description="W1: run LL-8 for open why-report flags and attach the report")
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--flag-id")
    sel.add_argument("--all-open", action="store_true")
    ap.add_argument("--wf-root", action="append", required=True, help="walk-forward run dir (repeatable)")
    ap.add_argument("--root", default=None, help="repo root holding diagnostics_outputs/ (default: cwd)")
    ap.add_argument("--out-root", default=None, help=f"base for the out dirs (default {DEFAULT_OUT_BASE})")
    ap.add_argument("--models-dir", default=None, help=f"default: {MANIFEST_REL} default_params.models_dir")
    ap.add_argument("--timeout-s", type=float, default=1800)
    ap.add_argument("--analysis-json", default=None, help="JSON object overriding ANALYSIS_DEFAULTS keys")
    ap.add_argument("--json", action="store_true")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    if a.timeout_s <= 0:
        print("ERROR --timeout-s must be > 0", file=sys.stderr)
        return 2
    analysis = None
    if a.analysis_json:
        try:
            analysis = json.loads(a.analysis_json)
            assert isinstance(analysis, dict)
        except Exception:
            print("ERROR --analysis-json must be a JSON object", file=sys.stderr)
            return 2
    try:   # bad arguments, refused before anything is created (Beta R1 Q3, Q4)
        check_analysis(analysis)
        check_out_base(a.out_root)
    except RunnerFailure as e:
        print(f"ERROR {e}", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.WARNING)
    kw = dict(root=a.root, out_root=a.out_root, timeout_s=a.timeout_s, models_dir=a.models_dir, analysis=analysis)
    if a.flag_id:
        try:
            known = a.flag_id in wrf.read_flags(a.root)["groups"]
        except Exception as e:      # cannot enumerate: non-zero, surfaced
            logger.error("why-report runner could not read the flags: %r", e, exc_info=True)
            print(f"ERROR flags not readable: {e!r}", file=sys.stderr)
            return 1
        if not known:
            print(f"ERROR unknown flag_id {a.flag_id!r}", file=sys.stderr)
            return 2
        results = [run_for_flag(a.flag_id, a.wf_root, **kw)]
    else:
        results = run_open_flags(a.wf_root, **kw)
    if a.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        for r in results:
            print(f"{r['flag_id']}: {r['status']}" + (f" {r['reason']}: {r['message']}" if r.get("reason") else "")
                  + (f" report {r['report_path']}" if r.get("report_path") else ""))
    attempted = [r for r in results if r.get("status") in ("ok", "failed")]
    return 1 if any(r["status"] == "failed" or r.get("attempt_record_error") for r in attempted) else 0


if __name__ == "__main__":
    sys.exit(main())
