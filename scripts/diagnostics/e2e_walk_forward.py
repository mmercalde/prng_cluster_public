#!/usr/bin/env python3
"""
scripts/diagnostics/e2e_walk_forward.py -- DEVELOPMENT walk-forward driver (TASK_LOOP_WF,
2026-11-23). NON-CERTIFYING. Every number it writes is DEVELOPMENT evidence on blocks that
have been looked at before; it can never produce a counted verdict.

FRAME (lead decision in logs/e2e_session/TASK_LOOP_COMMON.txt, revised by TASK_LOOP_ADDENDUM_1;
every element is a setting, defaults shown):
  Range [start, end): default start = the earliest workable origin (train >= --min-train-draws
  2000 and, with --labels-after-sieve on, a label window that starts at or after the sieve
  window end), default end = the dataset end. The range is cut into blocks of about
  --block-length 600 draws; with --block-cut date each cut moves forward to a new date.
  For origin block i = [b_i, b_{i+1}):
    train    Step 3 features use [0, T_i) only; T_i = b_i - L (--label-window fixed, L =
             --label-draws 1807, run H's label size), so the train end moves with the origin
    labels   Step 3 holdout labels use [T_i, b_i) only (positional, never session-filtered)
    training Step 5 trains on those labels (shim, one run per family and split seed)
    choice   --model-choice cv (default, the run H winner rule on CV scores), or
             prev_block_hits / prev_block_rank: every family is voted and graded on every
             block, and origin i takes the family that did best on the previous graded
             origin's block (always before b_i). Graded results go back in and change the
             next round. The first origin falls back to cv.
    vote     pools for every draw of block i, learned arm (winner), unlearned arm (Step-1-only
             key), and with prev_block_* one arm per family; sealed at b_i
    grade    block i through the grader in DEVELOPMENT mode, per session and pooled
             (--grade-sessions midday,evening,pooled), against k/1000; rank of the drawn number
             in the vote (RANK_RULE) beside the hits.
  Then block i is inside the train/label history of every later origin.

WHAT IT CALLS (each as a subprocess, each fail-closed on a non-zero exit; nothing retried):
  scripts/diagnostics/e2e_build_run_copy.py   this worktree; per-origin run copy
  scripts/diagnostics/e2e_step3_harness.py    this worktree; Step 3 on that copy
  scripts/diagnostics/e2e_step5_shim.py       this worktree; Step 5 (module unchanged)
  scripts/diagnostics/e2e_step5_check.py      this worktree; per-family checks
  <vote worktree>/scripts/diagnostics/e2e_vote_harness.py   READ ONLY, pinned by sha256
  <grader worktree>/scripts/diagnostics/e2e_grader.py       READ ONLY, pinned by sha256,
                                                            --mode development required
LEAK RULE (checked at run time, before the step that would use the data, and by the
self-test's perturbation gate): no draw at or after the origin block start b_i reaches
labels, training or model choice for origin i. Runtime checks (fail closed, exit 4):
  L1 the run copy's label block is exactly dataset draws [T_i, b_i) and its train history is
     exactly [0, T_i), by length and value
  L2 Step 3's recorded holdout and train inputs are those files (sha256)
  L3 Step 5's --survivors is that origin's Step 3 output (or the shim's derived target file
     made from it) and --lottery-data is the train history [0, T_i)
  L4 the choice uses Step 5 check reports and, for prev_block_*, only grades of blocks that
     end at or before b_i; the winner and every family are frozen (sha256, read-only) before
     the first vote of the origin
  L5 every vote is sealed at b_i (--seal-from b_i) with history = train + labels = [0, b_i)
  L6 every grade is of block i only, with upstream_unseen_from_index = b_i and
     frequency_control_block [0, b_i)
Exit codes: 0 done; 2 refused before anything ran; 3 a step failed (stop, no retry);
            4 a leak check failed; 5 a guard stopped a step (memory, card).
"""
from __future__ import annotations

import sys as _sys
_sys.dont_write_bytecode = True

import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import math
import os
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]
LABEL = "DEVELOPMENT"
LABEL_TEXT = ("DEVELOPMENT: walk-forward over blocks that have been looked at before; "
              "never a counted test, never a counted verdict")
VERSION = "e2e_walk_forward 0.2.0-draft-20261123 (ADDENDUM_1)"
BUILDER = HERE / "e2e_build_run_copy.py"
STEP3 = HERE / "e2e_step3_harness.py"
SHIM = HERE / "e2e_step5_shim.py"
CHECK5 = HERE / "e2e_step5_check.py"
RANKER = HERE / "e2e_step5_ranker.py"
CAPS = HERE / "e2e_pipeline_caps.py"   # pipeline-switch capability check (ADDENDUM_1 of TASK_LAND_WF_VOTE_20261125)
# Options the driver (and the Step 5 shim) set themselves. A passthrough (--step3-extra-args,
# --step5-extra-args, --nn-args) may not carry any of them, in either form (lead ADDENDUM_4 Fix 3,
# Beta X-E2E-WF-VOTE R1 Q2). Each list is checked against the argv the driver actually builds
# (own_options_check below), so it cannot drift from the construction.
STEP3_DRIVER_SET = {"--run-copy", "--out-dir", "--chunk-size", "--workers", "--device", "--config", "--gpu-ids"}
STEP5_DRIVER_SET = {"--survivors", "--lottery-data", "--model-type", "--output-dir", "--trials", "--k-folds"}
SHIM_SET = {"--split-seed", "--capture-dir", "--script", "--target-field"}
RANKER_OBJECTIVES = ["xgb_rank_ndcg", "xgb_rank_pairwise", "lgbm_lambdarank", "catboost_yetirank"]
MODULE5 = WT / "meta_prediction_optimizer_anti_overfit.py"
TST = WT / "train_single_trial.py"
FAMILY_TIE_ORDER = ["neural_net", "lightgbm", "xgboost", "catboost"]   # run H winner rule
WINNER_RULE = ("among families whose run exits 0 with a fresh sidecar and a loadable matching "
               "checkpoint, the highest finite Optuna cross-validation score on the same survivor "
               "folds and the same trial budget; tie order neural_net, lightgbm, xgboost, catboost; "
               "the winner and its artifacts are frozen before any grade.")
MUTANT_ENV = "E2E_WF_MUTANT"        # self-test only; recorded in every record when set
FORBIDDEN = [Path("~/distributed_prng_analysis").expanduser(), Path("/dev/shm"), Path("/tmp"),
             Path("~/miner_staging").expanduser()]


class Refused(Exception):
    pass


class StepFailed(Exception):
    pass


class LeakCheck(Exception):
    pass


class GuardStop(Exception):
    pass


def mutant():
    return os.environ.get(MUTANT_ENV) or None


def mut(name):
    m = mutant()
    return bool(m) and name in m.split("+")


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, obj):
    tmp = Path(str(path) + ".partial")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=False, default=str) + "\n")
    os.replace(tmp, path)


def now():
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def is_under(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------- dataset ----
def load_dataset_index(path: Path):
    """(date, session) for every row, and the draw values. The driver itself reads draw
    values only through held_draws(limit) for check L1."""
    rows = json.load(open(path))
    if not isinstance(rows, list) or not rows:
        raise Refused("dataset is not a non-empty list")
    idx, draws = [], []
    for i, r in enumerate(rows):
        if not (isinstance(r, dict) and {"date", "session", "draw"} <= set(r)):
            raise Refused(f"dataset row {i} is not {{date, session, draw}}")
        idx.append((str(r["date"]), str(r["session"])))
        draws.append(r["draw"])
    return idx, draws


def cut_blocks(idx, h0, h1, b, mode):
    """Block starts b_0 = h0 < b_1 < ... < b_B = h1."""
    n = h1 - h0
    if b < 2 or n < 2 * b:
        raise Refused(f"--blocks {b} on {n} holdout rows: need B >= 2 and >= 2 rows per block")
    cuts = [h0]
    for j in range(1, b):
        c = h0 + round(j * n / b)
        if mode == "date":
            while c < h1 and idx[c][0] == idx[c - 1][0]:
                c += 1
        if c <= cuts[-1] or c >= h1:
            raise Refused(f"block cut {j} at {c} is empty or past the holdout end")
        cuts.append(c)
    cuts.append(h1)
    return cuts


# --------------------------------------------------------------------------- guards -----
class Watch:
    """Samples MemAvailable, the child's process-group RSS and card memory once a second.
    Kills the group (recorded PGID, never a pattern) if MemAvailable < floor or the card
    passes the ceiling."""

    def __init__(self, pgid, mem_floor_mib, card_ceiling_mib, log_path):
        self.pgid, self.floor, self.ceiling = pgid, mem_floor_mib, card_ceiling_mib
        self.log = open(log_path, "a")
        self.min_avail = None
        self.peak_rss = 0
        self.peak_card = None
        self.fired = None
        self._stop = threading.Event()
        self.t = threading.Thread(target=self.loop, daemon=True)

    @staticmethod
    def mem_available_mib():
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
        return None

    def group_rss_mib(self):
        tot = 0
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                st = open(f"/proc/{d}/stat").read()
                fields = st[st.rindex(")") + 2:].split()
                if int(fields[2]) != self.pgid:
                    continue
                tot += int(open(f"/proc/{d}/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            except (OSError, ValueError, IndexError):
                continue
        return tot // (1024 * 1024)

    @staticmethod
    def card_used_mib():
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout.split()
            return int(out[0]) if out else None
        except (OSError, subprocess.SubprocessError, ValueError):
            return None

    def loop(self):
        while not self._stop.is_set():
            av, rss, card = self.mem_available_mib(), self.group_rss_mib(), self.card_used_mib()
            self.min_avail = av if self.min_avail is None else min(self.min_avail, av)
            self.peak_rss = max(self.peak_rss, rss)
            if card is not None:
                self.peak_card = card if self.peak_card is None else max(self.peak_card, card)
            self.log.write(f"{now()} avail_mib={av} group_rss_mib={rss} card_mib={card}\n")
            self.log.flush()
            why = None
            if av is not None and av < self.floor:
                why = f"MemAvailable {av} MiB < floor {self.floor} MiB"
            elif self.ceiling and card is not None and card > self.ceiling:
                why = f"card memory {card} MiB > ceiling {self.ceiling} MiB"
            if why:
                self.fired = why
                self.log.write(f"{now()} GUARD: {why}; killing PGID {self.pgid}\n")
                self.log.flush()
                try:
                    os.killpg(self.pgid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                return
            self._stop.wait(1.0)

    def start(self):
        self.t.start()

    def stop(self):
        self._stop.set()
        self.t.join(timeout=15)
        self.log.close()


def gpu_precheck(allowed_names, min_free_mib):
    """Owner ruling 2026-11-14: only `sunshine` may be on the card. Reads the process table
    of plain nvidia-smi (graphics and compute) and the free memory."""
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20).stdout
        free = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=20).stdout.split()
    except (OSError, subprocess.SubprocessError) as e:
        raise GuardStop(f"nvidia-smi failed: {e}")
    procs, in_tab = [], False
    for line in out.splitlines():
        if "Processes:" in line:
            in_tab = True
            continue
        if in_tab and line.startswith("|") and "PID" not in line and "No running" not in line:
            parts = line.strip("|").split()
            if len(parts) >= 6 and parts[0].isdigit():
                name = " ".join(parts[5:-1]) if parts[-1].endswith("MiB") else " ".join(parts[5:])
                procs.append(os.path.basename(name))
    others = [p for p in procs if p not in allowed_names]
    free_mib = int(free[0]) if free else 0
    if others:
        raise GuardStop(f"card busy: processes {others} (allowed: {allowed_names})")
    if free_mib < min_free_mib:
        raise GuardStop(f"card free {free_mib} MiB < {min_free_mib} MiB")
    return {"processes": procs, "free_mib": free_mib, "checked": now()}


# --------------------------------------------------------------------------- runner -----
class Runner:
    def __init__(self, args, out: Path, record_sha: str):
        self.a, self.out, self.record_sha = args, out, record_sha
        self.log = open(out / "driver.log", "a")
        self.seq = 0

    def say(self, msg):
        line = f"{now()} [{LABEL}] {msg}"
        print(line, flush=True)
        self.log.write(line + "\n")
        self.log.flush()

    def step(self, name, argv, cwd: Path, env: dict, log_path: Path, done_path: Path,
             outputs, gpu=False, prepare=None):
        """Run one subprocess step. `outputs` is a callable returning {relname: path} after
        success; their sha256s go in the DONE record. With --resume a step whose DONE record
        has the same argv and whose outputs still verify is skipped."""
        self.seq += 1
        if done_path.exists():
            if not self.a.resume:
                raise Refused(f"{done_path} exists and --resume not given")
            d = json.load(open(done_path))
            if d.get("argv") != argv or d.get("run_record_sha256") != self.record_sha:
                raise Refused(f"resume: {done_path} was made by a different argv or run record")
            for rel, s in d["outputs"].items():
                p = Path(rel) if os.path.isabs(rel) else cwd / rel
                if not p.is_file() or sha256_file(p) != s:
                    raise Refused(f"resume: output {p} missing or changed since {done_path}")
            self.say(f"skip (resume, verified) {name}")
            return d
        if self.a.resume and log_path.exists():
            raise Refused(f"resume: {name} has a log but no DONE record (partial); "
                          f"remove it by hand after inspection")
        gcheck = gpu_precheck(self.a.gpu_allowed_procs.split(","), self.a.gpu_min_free_mib) if gpu else None
        if prepare:
            prepare()
        self.say(f"start {name}: {shlex.join(argv)} (cwd {cwd})")
        t0 = time.time()
        with open(log_path, "w") as lf:
            p = subprocess.Popen(argv, cwd=cwd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            w = Watch(p.pid, self.a.mem_floor_mib, self.a.card_ceiling_mib, str(log_path) + ".watch")
            w.start()
            rc = p.wait()
            w.stop()
        wall = round(time.time() - t0, 2)
        if w.fired:
            raise GuardStop(f"{name}: {w.fired} (rc {rc}); see {log_path}.watch")
        if rc != 0:
            raise StepFailed(f"{name}: exit {rc}; see {log_path} (no retry)")
        outs = {}
        for rel, pth in outputs().items():
            if not Path(pth).is_file():
                raise StepFailed(f"{name}: expected output {pth} missing")
            outs[str(pth)] = sha256_file(pth)
        d = {"label": LABEL, "step": name, "seq": self.seq, "argv": argv, "cwd": str(cwd),
             "env": {k: env.get(k) for k in sorted(env) if k in ENV_KEYS},
             "rc": rc, "wall_seconds": wall, "started": _dt.datetime.fromtimestamp(t0).isoformat(),
             "finished": now(), "min_mem_available_mib": w.min_avail,
             "peak_group_rss_mib": w.peak_rss, "peak_card_used_mib": w.peak_card,
             "gpu_precheck": gcheck, "outputs": outs, "run_record_sha256": self.record_sha,
             "mutant": mutant()}
        write_json(done_path, d)
        self.say(f"done {name}: {wall} s, peak group RSS {w.peak_rss} MiB, card peak {w.peak_card} MiB")
        return d


ENV_KEYS = {"PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "TMPDIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH",
            "MPLCONFIGDIR", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "S88_FORCE_LGBM_CPU",
            "OMP_NUM_THREADS", MUTANT_ENV, "E2E_SHIM_MUTANT"}


def base_env(scratch: Path, cuda_visible: str):
    for d in ("tmp", "cache", "cuda_cache", "mpl"):
        (scratch / d).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.pop("OMP_NUM_THREADS", None)
    env.update(PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(scratch / "tmp"),
               XDG_CACHE_HOME=str(scratch / "cache"), CUDA_CACHE_PATH=str(scratch / "cuda_cache"),
               MPLCONFIGDIR=str(scratch / "mpl"), CUDA_VISIBLE_DEVICES=cuda_visible)
    return env


# --------------------------------------------------------------------------- plan -------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=f"{LABEL_TEXT}. {VERSION}")
    g = ap.add_argument_group("inputs (all sha256-bound)")
    g.add_argument("--dataset", required=True, help="frozen [{date, session, draw}]; the only draw source")
    g.add_argument("--dataset-sha256", required=True)
    g.add_argument("--npz", required=True)
    g.add_argument("--generation-digest", required=True)
    g.add_argument("--scorer-config", action="append", default=[],
                   help="passed to the run-copy builder as --config (repeatable); Step 3 reads "
                        "configs/<basename of the first>")
    g.add_argument("--sieve-window-end-index", type=int, required=True,
                   help="combined index one past the last draw the Step 1 sieve saw (a property of the "
                        "generation; run H: 1112 = 2002-12-10 midday + 1)")
    g = ap.add_argument_group("frame (TASK_LOOP_ADDENDUM_1 item 1)")
    g.add_argument("--range-start", default="auto",
                   help="first origin block start: combined index, YYYY-MM-DD (first row on/after), "
                        "or auto = the earliest workable origin (see --min-train-draws, --label-draws)")
    g.add_argument("--range-end", default="end", help="exclusive end: index, date, or end = dataset end")
    g.add_argument("--block-length", type=int, default=600, help="about this many draws per block")
    g.add_argument("--blocks", type=int, default=0, help="override: exact block count (0 = from length)")
    g.add_argument("--block-cut", choices=["date", "index"], default="date")
    g.add_argument("--origins", default="all", help="'all' or comma list of block numbers (0-based)")
    g.add_argument("--origin-stride", type=int, default=1, help="every n-th block of the selection")
    g.add_argument("--label-window", choices=["fixed", "expanding"], default="fixed",
                   help="fixed: labels [b-L, b), train [0, b-L) (train end moves with the origin); "
                        "expanding: labels [--label-start-index, b), train [0, label start)")
    g.add_argument("--label-draws", type=int, default=1807, help="L for fixed (run H label block: 1807)")
    g.add_argument("--label-start-index", default="sieve_end",
                   help="expanding: first label index (default the sieve window end)")
    g.add_argument("--min-train-draws", type=int, default=2000,
                   help="refuse an origin whose train history is shorter (2000 = every global_* "
                        "feature computed: global_state_tracker.py:156 (<100 -> defaults), :264 "
                        "(regime features need 2 x 1000))")
    g.add_argument("--labels-after-sieve", choices=["on", "off"], default="on",
                   help="on: refuse a label window that overlaps the sieve window")
    g.add_argument("--grade-sessions", default="midday,evening,pooled",
                   help="sessions graded and reported, each alone; 'pooled' = all of them together "
                        "(ADDENDUM_1 item 3, default both and pooled)")
    g.add_argument("--pool-sizes", default="20", help="comma list; each is --k N --pool-size N")
    g.add_argument("--previously-inspected", action="append", default=[],
                   help="START:END (half-open) already looked at; recorded per arm by the grader")
    g = ap.add_argument_group("learned arm")
    g.add_argument("--target-field", default="holdout_quality",
                   help="Step 5 target via the shim (default holdout_quality = today; holdout_hits; "
                        "holdout_features.<name>)")
    g.add_argument("--families", default=",".join(FAMILY_TIE_ORDER))
    g.add_argument("--split-seeds", default="42", help="comma list; one learned arm per seed")
    g.add_argument("--trials", type=int, default=20)
    g.add_argument("--k-folds", type=int, default=20)
    g.add_argument("--nn-args", default="--normalize-features --use-leaky-relu",
                   help="extra module args for neural_net (compare-models Option A, :240-245)")
    g.add_argument("--step5-extra-args", default="--enable-diagnostics")
    g.add_argument("--step5-cuda", choices=["visible", "hidden"], default="visible")
    g.add_argument("--on-family-failure", choices=["stop", "exclude"], default="stop")
    g.add_argument("--model-choice", choices=["cv", "prev_block_hits", "prev_block_rank"], default="cv",
                   help="cv = the declared run H winner rule (default). prev_block_* = every family "
                        "is voted and graded on each block; origin i picks the family with the most "
                        "hits / the lowest mean rank on the previous graded origin's block (always "
                        "before b_i); the first origin falls back to cv")
    g.add_argument("--choice-pool-size", type=int, default=0, help="0 = the largest pool size")
    g.add_argument("--choice-session", default="pooled")
    g.add_argument("--ranker-arms", default="",
                   help="ADDENDUM_2: comma list of ranker Step 5 arms (xgb_rank_ndcg, xgb_rank_pairwise, "
                        "lgbm_lambdarank, catboost_yetirank); each is its own learned arm, voted and graded at "
                        "every pool size; default none = today")
    g.add_argument("--ranker-groups", type=int, default=0, help="label draws sampled as groups (0 = all)")
    g.add_argument("--ranker-negatives", type=int, default=0, help="negatives kept per group (0 = all)")
    g.add_argument("--ranker-n-estimators", type=int, default=200)
    g.add_argument("--ranker-max-depth", type=int, default=6)
    g.add_argument("--ranker-learning-rate", type=float, default=0.1)
    g.add_argument("--ranker-device", choices=["cpu", "cuda"], default="cpu")
    g.add_argument("--ranker-max-rows", type=int, default=5_000_000)
    g.add_argument("--vote-all-families", choices=["auto", "on", "off"], default="auto",
                   help="auto = on when --model-choice is prev_block_*; family arms vote at the "
                        "choice pool size")
    g = ap.add_argument_group("Step 3")
    g.add_argument("--step3-device", choices=["cpu", "cuda"], default="cpu")
    g.add_argument("--step3-workers", type=int, default=4)
    g.add_argument("--step3-chunk-size", type=int, default=1000)
    g.add_argument("--step3-extra-args", default="")
    g = ap.add_argument_group("unlearned arm")
    g.add_argument("--unlearned-fields", default="forward_matches,reverse_matches")
    g.add_argument("--unlearned-tie-break", default="seed_asc")
    g = ap.add_argument_group("vote and grader (read only, pinned)")
    g.add_argument("--vote-harness", default=str(HERE / "e2e_vote_harness.py"))   # this repo's copy
    g.add_argument("--vote-harness-sha256", required=True)
    g.add_argument("--vote-config", default=str(WT / "prediction_generator_config.json"))
    g.add_argument("--vote-config-sha256", required=True)
    g.add_argument("--vote-workers", type=int, default=6)
    g.add_argument("--vote-allow-gpu", action="store_true",
                   help="needed for a neural_net arm (vote READY E2E-V6)")
    g.add_argument("--grader", default=str(HERE / "e2e_grader.py"))   # this repo's copy (landed 76234eb)
    g.add_argument("--grader-sha256", required=True)
    g.add_argument("--grader-workers", type=int, default=4)
    g.add_argument("--control-reps", type=int, default=200)
    g.add_argument("--rank-test-reps", type=int, default=10000, help="grader R24 Monte Carlo reps")
    g.add_argument("--vote-value-ordering", choices=["off", "full_vote"], default="off",
                   help="full_vote: the vote harness emits every value's score from the module's own vote "
                        "over all survivors (MISS agent, TODO-E2E-V12) on the rank-depth pool, and the "
                        "grader ranks the drawn number in it (R24); off = rank censored at the largest pool")
    g = ap.add_argument_group("guards and run")
    g.add_argument("--out-dir", required=True, help="new, under <worktree>/data/")
    g.add_argument("--gpu-id", default="0")
    g.add_argument("--gpu-allowed-procs", default="sunshine")
    g.add_argument("--gpu-min-free-mib", type=int, default=8192)
    g.add_argument("--mem-floor-mib", type=int, default=2048)
    g.add_argument("--card-ceiling-mib", type=int, default=0, help="0 = off")
    g.add_argument("--resume", action="store_true")
    g.add_argument("--check-only", action="store_true",
                   help="preflight only: every refusal the driver has (inputs, pins, pipeline switches, "
                        "passthroughs, out-dir), then exit 0 having written nothing (ADDENDUM_4 Fix 1)")
    g.add_argument("--plan-only", action="store_true",
                   help="write run_record.json and PLAN.md and stop (reads dates/sessions only)")
    return ap.parse_args(argv)


def csv_list(s, typ=str):
    return [typ(x.strip()) for x in s.split(",") if x.strip()]


def resolve_index(spec, idx, what):
    if isinstance(spec, str) and len(spec) == 10 and spec[4] == "-":
        for t, (d, _) in enumerate(idx):
            if d >= spec:
                return t
        raise Refused(f"{what} {spec}: no row on or after that date")
    try:
        return int(spec)
    except ValueError:
        raise Refused(f"{what} {spec!r}: integer index or YYYY-MM-DD")


def label_window(fr, b):
    """(T, b): train [0, T), labels [T, b)."""
    if fr["label_window"] == "fixed":
        return b - fr["label_draws"], b
    return fr["label_start_index"], b


def earliest_origin(fr):
    L = fr["label_draws"] if fr["label_window"] == "fixed" else None
    if fr["label_window"] == "fixed":
        lo = fr["min_train_draws"] + L
        if fr["labels_after_sieve"] == "on":
            lo = max(lo, fr["sieve_window_end_index"] + L)
        return max(lo, fr["sieve_window_end_index"])
    ls = fr["label_start_index"]
    return max(ls + 1, fr["sieve_window_end_index"])


def _caps():
    spec = importlib.util.spec_from_file_location("e2e_pipeline_caps", CAPS)
    caps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(caps)
    return caps


def check_passthroughs(a) -> dict:
    """Before any write (ADDENDUM_4 Fixes 2-3): every extra-args token must be a full option name the
    receiving parser declares (read as source), and none may be one the driver or shim sets."""
    caps = _caps()
    m = os.environ.get(MUTANT_ENV)
    s3_known = caps.parser_options(STEP3)
    m5_known = caps.parser_options(MODULE5)
    try:
        caps.check_passthrough(a.step3_extra_args, s3_known, STEP3_DRIVER_SET, "--step3-extra-args", m)
        caps.check_passthrough(a.step5_extra_args, m5_known, STEP5_DRIVER_SET | SHIM_SET, "--step5-extra-args", m)
        caps.check_passthrough(a.nn_args, m5_known, STEP5_DRIVER_SET | SHIM_SET, "--nn-args", m)
    except caps.PassthroughRefused as e:
        raise Refused(f"passthrough refused: {e}")
    return {"step3_receiver": str(STEP3), "step5_receiver": str(MODULE5),
            "driver_set": {"step3": sorted(STEP3_DRIVER_SET), "step5": sorted(STEP5_DRIVER_SET),
                           "shim": sorted(SHIM_SET)},
            "rule": "full declared option names only; no option the driver or shim sets"}


def own_options_check(argv, allowed: set, where: str) -> None:
    """The options the driver itself put in argv must be exactly within the declared set."""
    own = {x.split("=", 1)[0] for x in argv if isinstance(x, str) and x.startswith("--")}
    if not own <= allowed:
        raise StepFailed(f"internal: {where} builds {sorted(own - allowed)} not in its declared "
                         f"driver-set list; the passthrough guard would miss them")


def pipeline_switches(a) -> dict:
    spec = importlib.util.spec_from_file_location("e2e_pipeline_caps", CAPS)
    caps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(caps)
    requests = {"LL-1": caps.requested_ll1(Path(a.vote_config).expanduser()),
                "LL-2": caps.requested_ll2(a.step3_extra_args),
                "LL-3": caps.requested_ll3(a.step5_extra_args + " " + a.nn_args)}
    roots = {"LL-1": Path(a.vote_harness).expanduser().resolve().parents[2],   # the vote harness imports its repo's prediction_generator
             "LL-2": WT,                                                       # the Step 3 harness imports this repo's full_scoring_worker
             "LL-3": MODULE5.parent}                                           # Step 5 runs this repo's module
    try:
        return caps.check(requests, roots)
    except caps.MissingSwitch as e:
        raise Refused(f"pipeline switch missing, nothing run: {e}")


def build_declaration(a, plan_only=False):
    ds = Path(a.dataset).expanduser().resolve()
    if not ds.is_file() or sha256_file(ds) != a.dataset_sha256.lower():
        raise Refused("--dataset missing or its sha256 != --dataset-sha256")
    for p, s, what in [(a.vote_harness, a.vote_harness_sha256, "vote harness"),
                       (a.grader, a.grader_sha256, "grader"),
                       (a.vote_config, a.vote_config_sha256, "vote config")]:
        if not Path(p).is_file() or sha256_file(p) != s.lower():
            raise Refused(f"{what} {p} missing or sha256 != pinned {s}")
    # Pipeline switches (owner, TASK_LAND_WF_VOTE_20261125_ADDENDUM_1): a setting that needs an
    # LL-1/2/3 switch the pipeline files to be called do not have refuses here, before anything
    # is written. The files are read as source (e2e_pipeline_caps.py); the record below says
    # which switches were requested and applied, so no output carries a label it did not get.
    switches = pipeline_switches(a)        # first, so a missing LL switch is named as such
    passthrough = check_passthroughs(a)
    genv = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    gh = subprocess.run([sys.executable, "-B", a.grader, "--help"], capture_output=True, text=True, env=genv)
    if "--mode" not in gh.stdout or "development" not in gh.stdout or "--per-draw-detail" not in gh.stdout:
        raise Refused("pinned grader has no --mode development / --per-draw-detail (TASK_LOOP_GRADE "
                      "items 1-2); the walk-forward grades only in development mode")
    vh = subprocess.run([sys.executable, "-B", a.vote_harness, "--help"], capture_output=True, text=True, env=genv)
    if "--seal-from" not in vh.stdout:
        raise Refused("pinned vote harness has no --seal-from")
    if "--rank-summary" not in gh.stdout:
        raise Refused("pinned grader has no --rank-summary (R24)")
    if a.vote_value_ordering != "off" and "--emit-value-ordering" not in vh.stdout:
        raise Refused("pinned vote harness has no --emit-value-ordering (TODO-E2E-V12)")
    idx, draws = load_dataset_index(ds)
    N = len(idx)
    fams = csv_list(a.families)
    if not fams or any(f not in FAMILY_TIE_ORDER for f in fams) or len(set(fams)) != len(fams):
        raise Refused(f"--families {a.families}: subset of {FAMILY_TIE_ORDER}")
    seeds = csv_list(a.split_seeds, int)
    pools = csv_list(a.pool_sizes, int)
    if not seeds or not pools or any(k < 1 for k in pools) or len(set(pools)) != len(pools):
        raise Refused("--split-seeds and --pool-sizes must be non-empty, distinct, pool sizes >= 1")
    sess_all = sorted({s for _, s in idx})
    gsess = csv_list(a.grade_sessions)
    for s in gsess:
        if s != "pooled" and s not in sess_all:
            raise Refused(f"--grade-sessions {s}: not a dataset session {sess_all}")
    if not gsess:
        raise Refused("--grade-sessions is empty")
    rankers = csv_list(a.ranker_arms)
    if any(r not in RANKER_OBJECTIVES for r in rankers) or len(set(rankers)) != len(rankers):
        raise Refused(f"--ranker-arms {a.ranker_arms}: subset of {RANKER_OBJECTIVES}")
    choice_k = a.choice_pool_size or max(pools)
    vote_fams = a.vote_all_families == "on" or (a.vote_all_families == "auto" and a.model_choice != "cv")
    if a.model_choice != "cv":
        if not vote_fams:
            raise Refused("--model-choice prev_block_* needs --vote-all-families on/auto")
        if choice_k not in pools or a.choice_session not in gsess:
            raise Refused("--choice-pool-size must be one of --pool-sizes and --choice-session graded")
    if "neural_net" in fams and vote_fams and not a.vote_allow_gpu:
        raise Refused("neural_net family arms vote only on the GPU (vote READY E2E-V6): add --vote-allow-gpu")
    swe = a.sieve_window_end_index
    if not 0 < swe < N:
        raise Refused(f"--sieve-window-end-index {swe} outside the dataset")
    fr = {"sieve_window_end_index": swe, "label_window": a.label_window, "label_draws": a.label_draws,
          "label_start_index": (swe if a.label_start_index == "sieve_end"
                                else resolve_index(a.label_start_index, idx, "--label-start-index")),
          "min_train_draws": a.min_train_draws, "labels_after_sieve": a.labels_after_sieve}
    if a.label_window == "fixed" and a.label_draws < 1:
        raise Refused("--label-draws must be >= 1")
    e0 = earliest_origin(fr)
    start = e0 if a.range_start == "auto" else resolve_index(a.range_start, idx, "--range-start")
    end = N if a.range_end == "end" else resolve_index(a.range_end, idx, "--range-end")
    if start < e0:
        raise Refused(f"--range-start {start} is before the earliest workable origin {e0} "
                      f"(train >= {a.min_train_draws}, labels {a.label_window}"
                      f"{' after the sieve window' if a.labels_after_sieve == 'on' else ''})")
    if a.block_cut == "date":
        while 0 < start < N and idx[start][0] == idx[start - 1][0]:
            start += 1
    if not start < end <= N:
        raise Refused(f"range [{start}, {end}) is empty or past the dataset ({N} rows)")
    B = a.blocks or max(1, round((end - start) / a.block_length))
    cuts = cut_blocks(idx, start, end, B, a.block_cut) if B > 1 else [start, end]
    prev = []
    for s in a.previously_inspected:
        x, y = (int(v) for v in s.split(":"))
        if not 0 <= x < y:
            raise Refused(f"--previously-inspected {s}")
        prev.append([x, y])
    blocks = []
    for j in range(B):
        s, e = cuts[j], cuts[j + 1]
        per = {}
        for t in range(s, e):
            per[idx[t][1]] = per.get(idx[t][1], 0) + 1
        T, _ = label_window(fr, s)
        blocks.append({"block": j, "start_index": s, "end_index": e, "rows": e - s,
                       "first_date": idx[s][0], "last_date": idx[e - 1][0], "sessions": per,
                       "train": [0, T], "labels": [T, s],
                       "train_dates": [idx[0][0], idx[T - 1][0]] if T > 0 else None,
                       "label_dates": [idx[T][0], idx[s - 1][0]] if s > T else None,
                       "overlaps_previously_inspected": [p for p in prev if p[0] < e and s < p[1]]})
    sel = list(range(B)) if a.origins == "all" else csv_list(a.origins, int)
    if not sel or any(o < 0 or o >= B for o in sel) or sel != sorted(set(sel)):
        raise Refused(f"--origins {a.origins}: need ascending block numbers in 0..{B - 1}")
    origins = sel[::max(1, a.origin_stride)]
    for o in origins:
        T, b = blocks[o]["train"][1], blocks[o]["start_index"]
        if T < a.min_train_draws:
            raise Refused(f"origin {o}: train [0, {T}) shorter than --min-train-draws {a.min_train_draws}")
        if b - T < 1:
            raise Refused(f"origin {o}: empty label window")
        if a.labels_after_sieve == "on" and T < swe:
            raise Refused(f"origin {o}: label window [{T}, {b}) overlaps the sieve window (ends {swe})")
    code = {str(p.relative_to(WT)): sha256_file(p) for p in (Path(__file__).resolve(), BUILDER, STEP3,
                                                              SHIM, CHECK5, RANKER, MODULE5, TST, CAPS)}
    head = subprocess.run(["git", "-C", str(WT), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    fr.update({"range": [start, end], "earliest_workable_origin": e0, "blocks": B,
               "block_length_setting": a.block_length, "block_cut": a.block_cut,
               "block_table": blocks, "origins": origins, "grade_sessions": gsess,
               "dataset_sessions": sess_all, "pool_sizes": pools, "rank_depth": max(pools),
               "previously_inspected": prev,
               "label_block_rule": "fixed: [b_i - L, b_i); expanding: [label_start, b_i)",
               "train_rule": "[0, label start): the train-history end moves with the origin",
               "grade_block_rule": "[b_i, b_{i+1})",
               "seal_rule": "vote --seal-from b_i; history = train + label window = [0, b_i)",
               "unseen_rule": "grader upstream_unseen_from_index = b_i; frequency block [0, b_i)"})
    decl = {
        "kind": "e2e_walk_forward_run_record", "label": LABEL, "label_text": LABEL_TEXT,
        "version": VERSION,
        "authority": {"task": "logs/e2e_session/TASK_LOOP_WF.txt + TASK_LOOP_ADDENDUM_1.txt (main tree)",
                      "frame": "TASK_LOOP_COMMON.txt (lead decision, a setting)"},
        "inputs": {"dataset": {"path": str(ds), "sha256": a.dataset_sha256.lower(), "rows": N},
                   "npz": {"path": str(Path(a.npz).expanduser().resolve()),
                           "generation_digest": a.generation_digest.lower()},
                   "scorer_configs": [{"path": str(Path(c).expanduser().resolve()),
                                       "sha256": sha256_file(Path(c).expanduser())}
                                      for c in a.scorer_config]},
        "frame": fr,
        "learned": {"target_field": a.target_field, "families": fams, "split_seeds": seeds,
                    "trials": a.trials, "k_folds": a.k_folds, "nn_args": shlex.split(a.nn_args),
                    "step5_extra_args": shlex.split(a.step5_extra_args), "step5_cuda": a.step5_cuda,
                    "on_family_failure": a.on_family_failure, "winner_rule": WINNER_RULE,
                    "tie_order": FAMILY_TIE_ORDER, "model_choice": a.model_choice,
                    "choice_pool_size": choice_k, "choice_session": a.choice_session,
                    "vote_all_families": vote_fams,
                    "ranker": {"arms": rankers, "groups": a.ranker_groups, "negatives_per_group": a.ranker_negatives,
                               "n_estimators": a.ranker_n_estimators, "max_depth": a.ranker_max_depth,
                               "learning_rate": a.ranker_learning_rate, "device": a.ranker_device,
                               "max_rows": a.ranker_max_rows,
                               "role": "separate learned arms (not in the family choice); relevance = the "
                                       "survivor's Step 6 vote equals the drawn number, label draws only"},
                    "choice_rule": {
                        "cv": "the declared winner rule (WINNER_RULE)",
                        "prev_block_hits": "most hits at the choice pool size and session on the previous "
                                           "graded origin's block; ties: higher cv, then tie order",
                        "prev_block_rank": "lowest mean rank (RANK_RULE) on the previous graded origin's "
                                           "block; ties: higher cv, then tie order",
                        "first_origin": "no previous block in this run: falls back to cv"}},
        "rank": {"rule": RANK_RULE, "depth": max(pools), "vote_value_ordering": a.vote_value_ordering,
                 "grader": "--rank-summary on, --rank-source record_or_pool, --rank-test-reps "
                           f"{a.rank_test_reps}; with ordering off the grader's pool-derived rank must equal "
                           "the driver's (checked per draw, fail closed)"},
        "unlearned": {"ranking": "field_sum", "fields": csv_list(a.unlearned_fields),
                      "tie_break": a.unlearned_tie_break, "data_cutoff_index": "b_i (origin)"},
        "pipeline_switches": switches,
        "passthrough_guard": passthrough,
        "step3": {"device": a.step3_device, "workers": a.step3_workers,
                  "chunk_size": a.step3_chunk_size, "extra_args": shlex.split(a.step3_extra_args)},
        "vote": {"harness": str(Path(a.vote_harness).resolve()), "harness_sha256": a.vote_harness_sha256,
                 "config": str(Path(a.vote_config).resolve()), "config_sha256": a.vote_config_sha256,
                 "workers": a.vote_workers, "allow_gpu": a.vote_allow_gpu},
        "grader": {"path": str(Path(a.grader).resolve()), "sha256": a.grader_sha256,
                   "mode": "development", "per_draw_detail": "on",
                   "provenance": "off (--no-provenance; the driver records the grader sha256)",
                   "out_of_block_records": "exclude_and_count", "control_reps": a.control_reps,
                   "workers": a.grader_workers,
                   "abstention": {"allow_abstention": False, "empty_pool": "error"}},
        "guards": {"gpu_id": a.gpu_id, "gpu_allowed_procs": a.gpu_allowed_procs,
                   "gpu_min_free_mib": a.gpu_min_free_mib, "mem_floor_mib": a.mem_floor_mib,
                   "card_ceiling_mib": a.card_ceiling_mib},
        "code": {"head": head, "sha256": code, "python": sys.executable},
        "mutant": mutant(),
    }
    return decl, idx, draws


RANK_RULE = ("rank of the drawn number in the arm's vote at depth R (the largest pool size): the "
             "pool's distinct values in first-occurrence (vote) order get ranks 1..d; a drawn number "
             "not among them gets the mid-rank of the unvoted values, (d + 1 + 1000) / 2. Under the "
             "uniform background every draw has expected rank 500.5 exactly; its variance is computed "
             "exactly per draw from d. Descriptive (normal approximation), development only. "
             "Censored at R because Step 6 votes cost O(R x t) per draw (prediction_generator.py:841-846).")


# --------------------------------------------------------------- selfplay interface -----
SELFPLAY_INTERFACE = "e2e_walk_forward.selfplay_window v1"


def selfplay_window(frame, i):
    """STABLE INTERFACE for selfplay's shadow validator (TASK_LOOP_ADDENDUM_3, window nn; lead
    answer to selfplay Q3/Q4). `frame` is run_record.json["frame"] of a walk-forward run (or the
    dict build_declaration returns). For origin block i >= 1:
      validate_block  [b_{i-1}, b_i): candidates are judged on it (pool hits / rank via the grader
                      in development mode)
      validate_labels [T', b_{i-1}) and validate_train [0, T'), T' = label start for b_{i-1}
                      under the run's label window: what the candidate may learn from
      use_block       [b_i, b_{i+1}): where the promoted policy is used; nothing in the
                      validation pass reads it
      seal            b_i: no draw at or after it may be read by the validation pass
    Raises ValueError for i < 1 or i >= number of blocks. Fields are only ever added."""
    bt = frame["block_table"]
    if not 1 <= i < len(bt):
        raise ValueError(f"selfplay_window: origin block {i} not in 1..{len(bt) - 1}")
    v, u = bt[i - 1], bt[i]
    T, b = label_window(frame, v["start_index"])
    return {"interface": SELFPLAY_INTERFACE, "origin_block": i,
            "validate_block": [v["start_index"], v["end_index"]], "validate_dates": [v["first_date"], v["last_date"]],
            "validate_labels": [T, b], "validate_train": [0, T],
            "use_block": [u["start_index"], u["end_index"]], "use_dates": [u["first_date"], u["last_date"]],
            "seal": u["start_index"], "label_window": frame["label_window"], "label_draws": frame["label_draws"],
            "sieve_window_end_index": frame["sieve_window_end_index"]}


# --------------------------------------------------------------------------- rank -------
def rank_of(pool, y, space=1000):
    seen, order = set(), []
    for v in pool:
        if v not in seen:
            seen.add(v)
            order.append(v)
    d = len(order)
    if y in seen:
        r = order.index(y) + 1
        cens = False
    else:
        r = (d + 1 + space) / 2
        cens = True
    mid = (d + 1 + space) / 2
    var = (sum((q - 500.5) ** 2 for q in range(1, d + 1)) + (space - d) * (mid - 500.5) ** 2) / space
    return r, cens, d, var


def rank_summary(rows):
    n = len(rows)
    if not n:
        return None
    s = sum(r["rank"] for r in rows)
    v = sum(r["rank_null_var"] for r in rows)
    mean = s / n
    z = (s - 500.5 * n) / math.sqrt(v) if v > 0 else None
    return {"n": n, "mean_rank": mean, "expected_mean_rank": 500.5,
            "sd_of_mean_under_uniform": math.sqrt(v) / n, "z_lower_is_better": z,
            "censored": sum(1 for r in rows if r["rank_censored"]),
            "note": "descriptive; normal approximation; ties = mid-rank of unvoted values"}


# --------------------------------------------------------------------------- origin -----
def py():
    return [sys.executable, "-u", "-B"]


def origin_paths(out: Path, i: int):
    o = out / f"origin_{i}"
    return {"o": o, "split": o / "split", "copy": o / "copy", "step3": o / "step3", "step5": o / "step5",
            "frozen": o / "frozen", "winner": o / "winner", "vote": o / "vote", "grade": o / "grade",
            "scratch": o / "_env"}


def freeze(src_dir: Path, ck: Path, dst: Path):
    dst.mkdir(parents=True)
    shutil.copyfile(src_dir / "best_model.meta.json", dst / "best_model.meta.json")
    shutil.copyfile(ck, dst / ck.name)
    shas = {n: sha256_file(dst / n) for n in ("best_model.meta.json", ck.name)}
    return shas


def make_read_only(d: Path):
    for f in d.iterdir():
        f.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    d.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)


def run_origin(R: Runner, a, decl, i, idx, draws, prev_rec):
    fr = decl["frame"]
    bt = fr["block_table"][i]
    b_i, b_next = bt["start_index"], bt["end_index"]
    T, _ = label_window(fr, b_i)
    P = origin_paths(R.out, i)
    P["o"].mkdir(exist_ok=a.resume)
    P["scratch"].mkdir(exist_ok=True)
    rec = {"label": LABEL, "origin": i, "block": bt, "train": [0, T], "label_block": [T, b_i],
           "leak_checks": {}, "mutant": mutant()}
    label_end = b_next if mut("label_includes_origin") else b_i
    cpu_env = base_env(P["scratch"], "")

    # ---- 0. split files: train [0, T), holdout [T, N) as the builder's contract needs --------
    P["split"].mkdir(exist_ok=True)
    trp, hop = P["split"] / "train_history.json", P["split"] / "holdout_history.json"
    if not trp.exists():
        write_json(trp, draws[:T])
        write_json(hop, draws[T:])      # the builder checks dataset == train + holdout; it lists
                                        # holdout[label_count:] only as grade records nobody upstream reads
    train_sha = sha256_file(trp)
    rec["train_history"] = {"path": str(trp), "sha256": train_sha, "draws": T}

    # ---- 1. run copy (labels = [T, b_i)) ----------------------------------------------
    argv = py() + [str(BUILDER), "--npz", decl["inputs"]["npz"]["path"],
                   "--generation-digest", decl["inputs"]["npz"]["generation_digest"],
                   "--train-history", str(trp), "--holdout-history", str(hop),
                   "--dataset", decl["inputs"]["dataset"]["path"],
                   "--dataset-digest", decl["inputs"]["dataset"]["sha256"]]
    for c in decl["inputs"]["scorer_configs"]:
        argv += ["--config", c["path"]]
    argv += ["--label-start", "0", "--label-count", str(label_end - T),
             "--grade-sessions", "all", "--out-dir", str(P["copy"])]
    R.step(f"o{i}/copy", argv, WT, cpu_env, P["o"] / "copy.log", P["o"] / "copy.DONE.json",
           lambda: {"label": P["copy"] / "label_block_history.json",
                    "train": P["copy"] / "train_history.json",
                    "record": P["copy"] / "run_record.json"})
    lab = json.load(open(P["copy"] / "label_block_history.json"))
    ok_l1 = (len(lab) == b_i - T and lab == draws[T:b_i]
             and sha256_file(P["copy"] / "train_history.json") == train_sha)
    rec["leak_checks"]["L1_label_block"] = {"expected": [T, b_i], "len": len(lab), "ok": ok_l1,
                                            "sha256": sha256_file(P["copy"] / "label_block_history.json")}
    if not ok_l1 and not mut("nocheck"):
        raise LeakCheck(f"L1 origin {i}: label block is not dataset draws [{T}, {b_i}) or train != [0, {T})")

    # ---- 2. Step 3 -------------------------------------------------------------------
    cfg = Path(decl["inputs"]["scorer_configs"][0]["path"]).name if decl["inputs"]["scorer_configs"] else None
    argv = py() + [str(STEP3), "--run-copy", str(P["copy"]), "--out-dir", str(P["step3"]),
                   "--chunk-size", str(a.step3_chunk_size), "--workers", str(a.step3_workers),
                   "--device", a.step3_device]
    if cfg:
        argv += ["--config", f"configs/{cfg}"]
    if a.step3_device == "cuda":
        argv += ["--gpu-ids", a.gpu_id]
    own_options_check(argv[len(py()) + 1:], STEP3_DRIVER_SET, "Step 3 argv")
    argv += shlex.split(a.step3_extra_args)
    s3env = base_env(P["scratch"], a.gpu_id if a.step3_device == "cuda" else "")
    R.step(f"o{i}/step3", argv, WT, s3env, P["o"] / "step3.log", P["o"] / "step3.DONE.json",
           lambda: {"survivors": P["step3"] / "survivors_with_scores.json",
                    "record": P["step3"] / "run_record.json"}, gpu=(a.step3_device == "cuda"))
    s3 = json.load(open(P["step3"] / "run_record.json"))
    s3_in = s3.get("inputs", {})
    hold_sha = (s3_in.get("holdout") or {}).get("sha256")
    s3_train = (s3_in.get("train") or {}).get("sha256")
    ok_l2 = (s3.get("status") == "ok" and not s3.get("partial") and
             hold_sha == rec["leak_checks"]["L1_label_block"]["sha256"] and s3_train == train_sha)
    surv = P["step3"] / "survivors_with_scores.json"
    rec["leak_checks"]["L2_step3_inputs"] = {"holdout_sha256": hold_sha, "train_sha256": s3_train,
                                             "status": s3.get("status"), "ok": ok_l2}
    if not ok_l2 and not mut("nocheck"):
        raise LeakCheck(f"L2 origin {i}: Step 3 inputs are not the label block and train history")
    rec["labels"] = {"survivors_with_scores_sha256": sha256_file(surv),
                     "rows": (s3.get("checks") or {}).get("rows")}

    # ---- 3. Step 5 per seed and family ----------------------------------------------
    lottery = str(trp)
    if mut("step5_lottery_future"):
        fut = P["o"] / "lottery_with_future.json"
        json.dump(draws[:b_next], open(fut, "w"))     # self-test mutant: block i reaches Step 5
        lottery = str(fut)
    fam_by_seed = {}
    for seed in decl["learned"]["split_seeds"]:
        fam_res = {}
        for fam in decl["learned"]["families"]:
            rd = P["step5"] / f"s{seed}" / fam
            rd.parent.mkdir(parents=True, exist_ok=True)
            if rd.exists() and not a.resume:
                raise Refused(f"{rd} exists and --resume not given")
            cuda = a.gpu_id if decl["learned"]["step5_cuda"] == "visible" else ""
            env = base_env(rd / "_env", cuda)
            env["PYTHONPATH"] = str(WT)
            if fam == "lightgbm":
                env["S88_FORCE_LGBM_CPU"] = "1"     # as the compare-models parent (:264-265)

            def prepare(rd=rd, fam=fam, seed=seed, env=env):
                # The final NN call runs train_single_trial.py relative to cwd (:2563, :2602);
                # fold calls use the module's directory (:1949-1953). Both are worktree files.
                shutil.copyfile(TST, rd / "train_single_trial.py")
                lines = [f"date={now()}", f"family={fam}", f"split_seed={seed}", f"cwd={rd}",
                         f"train_single_trial_sha256={sha256_file(rd / 'train_single_trial.py')}",
                         f"module_sha256={sha256_file(MODULE5)}"]
                lines += sorted(f"{k}={env[k]}" for k in env if k in ENV_KEYS)
                (rd / "run_env.txt").write_text("\n".join(lines) + "\n")

            margs = ["--survivors", str(surv), "--lottery-data", lottery, "--model-type", fam,
                     "--output-dir", str(rd / "models" / "reinforcement"),
                     "--trials", str(decl["learned"]["trials"]), "--k-folds", str(decl["learned"]["k_folds"])]
            own_options_check(margs, STEP5_DRIVER_SET, "Step 5 module argv")
            margs += decl["learned"]["step5_extra_args"]
            if fam == "neural_net":
                margs += decl["learned"]["nn_args"]
            shim_own = ["--split-seed", str(seed), "--capture-dir", str(rd / "shim_capture"),
                        "--script", str(MODULE5), "--target-field", decl["learned"]["target_field"]]
            own_options_check(shim_own, SHIM_SET, "Step 5 shim argv")
            argv = py() + [str(SHIM)] + shim_own + ["--"] + margs
            gpu = bool(cuda) and fam in ("neural_net", "xgboost", "catboost")
            try:
                R.step(f"o{i}/step5/s{seed}/{fam}", argv, rd, env, Path(str(rd) + ".log"),
                       Path(str(rd) + ".DONE.json"),
                       lambda rd=rd: {"sidecar": rd / "models" / "reinforcement" / "best_model.meta.json",
                                      "shim": rd / "shim_capture" / "shim_summary.json"},
                       gpu=gpu, prepare=prepare)
                ck_out = Path(str(rd) + ".check.json")
                R.step(f"o{i}/step5/s{seed}/{fam}/check",
                       py() + [str(CHECK5), "--run-dir", str(rd), "--family", fam,
                               "--expected-trials", str(decl["learned"]["trials"]), "--out", str(ck_out)],
                       WT, base_env(rd / "_env", ""), Path(str(rd) + ".check.log"),
                       Path(str(rd) + ".check.DONE.json"), lambda ck_out=ck_out: {"check": ck_out})
                ck = json.load(open(ck_out))
            except StepFailed as e:
                if decl["learned"]["on_family_failure"] == "stop":
                    raise
                fam_res[fam] = {"eligible": False, "reason": str(e)}
                continue
            shim = json.load(open(rd / "shim_capture" / "shim_summary.json"))
            sargv = shim.get("argv", [])
            used_surv = sargv[sargv.index("--survivors") + 1] if "--survivors" in sargv else None
            used_lot = sargv[sargv.index("--lottery-data") + 1] if "--lottery-data" in sargv else None
            tgt = shim.get("target") or {}
            surv_ok = (used_surv == str(surv)) if not tgt.get("rewritten") else \
                (tgt.get("original_sha256") == rec["labels"]["survivors_with_scores_sha256"]
                 and used_surv == tgt.get("derived_survivors"))
            lot_ok = used_lot is not None and sha256_file(used_lot) == train_sha
            rec["leak_checks"].setdefault("L3_step5_inputs", {})[f"s{seed}/{fam}"] = {
                "survivors": used_surv, "lottery_data": used_lot, "target": tgt.get("field"),
                "ok": surv_ok and lot_ok}
            if not (surv_ok and lot_ok) and not mut("nocheck"):
                raise LeakCheck(f"L3 origin {i} s{seed}/{fam}: Step 5 inputs are not this origin's "
                                f"labels and train history")
            fam_res[fam] = {"eligible": ck.get("status") == "ok" and ck.get("cv_score") is not None
                            and math.isfinite(ck["cv_score"]),
                            "cv_score": ck.get("cv_score"), "check_status": ck.get("status"),
                            "problems": ck.get("problems"),
                            "first_permutation_sha256": (ck.get("shim", {}).get("first_permutation") or {}).get("sha256"),
                            "sidecar_sha256": ck.get("sidecar", {}).get("sha256"),
                            "checkpoint_path": ck.get("checkpoint", {}).get("path"),
                            "checkpoint_sha256": ck.get("checkpoint", {}).get("sha256"),
                            "model_dir": str(rd / "models" / "reinforcement")}
            if ck.get("status") != "ok" and decl["learned"]["on_family_failure"] == "stop":
                raise StepFailed(f"o{i} s{seed}/{fam}: check problems {ck.get('problems')}")
        fam_by_seed[seed] = fam_res

    # ---- 4. freeze every eligible family, then choose (L4) ----------------------------
    winners, frozen_all = {}, {}
    for seed, fam_res in fam_by_seed.items():
        frozen_all[seed] = {}
        for fam, r in fam_res.items():
            if not r.get("eligible"):
                continue
            fd = P["frozen"] / f"s{seed}" / fam
            if fd.exists():
                if not a.resume:
                    raise Refused(f"{fd} exists")
                shas = {n: sha256_file(fd / n) for n in ("best_model.meta.json", Path(r["checkpoint_path"]).name)}
            else:
                shas = freeze(Path(r["model_dir"]), Path(r["checkpoint_path"]), fd)
                make_read_only(fd)
            if shas["best_model.meta.json"] != r["sidecar_sha256"] or \
                    shas[Path(r["checkpoint_path"]).name] != r["checkpoint_sha256"]:
                raise StepFailed(f"o{i} s{seed}/{fam}: frozen copy differs from the checked artifacts")
            frozen_all[seed][fam] = {"dir": str(fd), "sha256": shas, **{k: r[k] for k in (
                "cv_score", "checkpoint_sha256", "sidecar_sha256", "first_permutation_sha256")}}
        winners[seed] = choose(R, a, decl, P, i, seed, fam_res, frozen_all[seed], prev_rec, b_i)
    rec["step5"] = {f"s{s}": w for s, w in winners.items()}
    rec["frozen_families"] = {f"s{s}": v for s, v in frozen_all.items()}
    rec["leak_checks"]["L4_choice"] = {
        f"s{s}": {"model_type": w["model_type"], "frozen_at": w["frozen_at"], "rule": w["rule_used"],
                  "evidence_blocks": w["evidence_blocks"], "winner_json_sha256": w["winner_json_sha256"],
                  "ok": all(e[1] <= b_i for e in w["evidence_blocks"])} for s, w in winners.items()}
    if not all(v["ok"] for v in rec["leak_checks"]["L4_choice"].values()) and not mut("nocheck"):
        raise LeakCheck(f"L4 origin {i}: model choice used a block at or after b_i")

    # ---- 4b. ranker arms (ADDENDUM_2), label draws [T, b_i) only (L7) -------------------
    rankers = {}
    for seed in decl["learned"]["split_seeds"]:
        for obj in decl["learned"]["ranker"]["arms"]:
            if not frozen_all[seed]:
                raise StepFailed(f"o{i} s{seed}: no frozen Step 5 sidecar to template the ranker on")
            tpl = Path(next(iter(frozen_all[seed].values()))["dir"]) / "best_model.meta.json"
            rd = P["step5"] / f"s{seed}" / f"R_{obj}"
            rk = decl["learned"]["ranker"]
            lab_p = P["copy"] / "label_block_history.json"
            argv = py() + [str(RANKER), "--survivors", str(surv), "--label-history", str(lab_p),
                           "--label-start", str(T + 1 if mut("ranker_label_shift") else T), "--expect-label-sha256", rec["leak_checks"]["L1_label_block"]["sha256"],
                           "--sidecar-template", str(tpl), "--pg-config", decl["vote"]["config"],
                           "--objective", obj, "--seed", str(seed), "--groups", str(rk["groups"]),
                           "--negatives-per-group", str(rk["negatives_per_group"]),
                           "--n-estimators", str(rk["n_estimators"]), "--max-depth", str(rk["max_depth"]),
                           "--learning-rate", str(rk["learning_rate"]), "--device", rk["device"],
                           "--workers", str(a.step3_workers), "--max-rows", str(rk["max_rows"]),
                           "--out-dir", str(rd)]
            renv = base_env(P["scratch"], a.gpu_id if rk["device"] == "cuda" else "")
            renv["PYTHONPATH"] = str(WT)
            R.step(f"o{i}/step5/s{seed}/R_{obj}", argv, WT, renv, Path(str(rd) + ".log"),
                   Path(str(rd) + ".DONE.json"),
                   lambda rd=rd: {"sidecar": rd / "best_model.meta.json", "record": rd / "ranker_record.json"},
                   gpu=(rk["device"] == "cuda"))
            rr = json.load(open(rd / "ranker_record.json"))
            ok7 = (rr["inputs"]["label_history"]["sha256"] == rec["leak_checks"]["L1_label_block"]["sha256"]
                   and rr["ranking_objective"]["label_start_T"] == T
                   and rr["ranking_objective"]["label_draws_L"] == b_i - T
                   and rr["inputs"]["survivors"]["sha256"] == rec["labels"]["survivors_with_scores_sha256"])
            rec["leak_checks"].setdefault("L7_ranker_inputs", {})[f"s{seed}/{obj}"] = {
                "label_sha256": rr["inputs"]["label_history"]["sha256"], "T": rr["ranking_objective"]["label_start_T"],
                "L": rr["ranking_objective"]["label_draws_L"], "ok": ok7}
            if not ok7 and not mut("nocheck"):
                raise LeakCheck(f"L7 origin {i} s{seed}/{obj}: ranker inputs are not this origin's labels [{T}, {b_i})")
            ck = Path(rr["checkpoint"]["path"])
            fd = P["frozen"] / f"s{seed}" / f"R_{obj}"
            if fd.exists():
                if not a.resume:
                    raise Refused(f"{fd} exists")
                shas = {n: sha256_file(fd / n) for n in ("best_model.meta.json", ck.name)}
            else:
                shas = freeze(rd, ck, fd)
                make_read_only(fd)
            if shas[ck.name] != rr["checkpoint"]["sha256"]:
                raise StepFailed(f"o{i} s{seed}/{obj}: frozen ranker differs from its record")
            rankers.setdefault(seed, {})[obj] = {"dir": str(fd), "sha256": shas,
                                                 "checkpoint_sha256": rr["checkpoint"]["sha256"],
                                                 "training_metrics": rr["training_metrics"],
                                                 "ranking_objective": rr["ranking_objective"]}
    rec["rankers"] = {f"s{s}": v for s, v in rankers.items()}

    # ---- 5. votes (sealed at b_i) ----------------------------------------------------
    if mut("grade_before_freeze"):          # self-test: the vote would run with no frozen winner
        for s in winners:
            (P["winner"] / f"s{s}").chmod(0o755)
            (P["winner"] / f"s{s}" / "WINNER.json").unlink()
    seal = b_next if mut("seal_after_origin") else b_i
    hist = [str(trp), str(P["copy"] / "label_block_history.json")]
    if mut("seal_after_origin"):
        extra = P["o"] / "block_i_history.json"
        json.dump(draws[b_i:b_next], open(extra, "w"))
        hist.append(str(extra))
    rec["leak_checks"]["L5_vote_seal"] = {"seal_from": seal, "history": hist,
                                          "history_sha256": [sha256_file(h) for h in hist], "ok": seal == b_i}
    if seal != b_i and not mut("nocheck"):
        raise LeakCheck(f"L5 origin {i}: vote seal {seal} != b_i {b_i}")
    P["vote"].mkdir(exist_ok=True)
    plan = []          # (arm_id, role, seed, family, k, models_dir, sidecar_sha, ck_sha)
    for k in fr["pool_sizes"]:
        for s, w in winners.items():
            plan.append((f"A1s{s}_o{i}_p{k}", "learned", s, w["model_type"], k, w["frozen_dir"],
                         w["frozen_sha256"]["best_model.meta.json"], w["checkpoint_sha256"]))
        plan.append((f"A0_o{i}_p{k}", "unlearned", None, None, k, None, None, None))
        for s, objs in rankers.items():
            for obj, rinfo in objs.items():
                plan.append((f"R{obj}s{s}_o{i}_p{k}", "ranker", s, obj, k, rinfo["dir"],
                             rinfo["sha256"]["best_model.meta.json"], rinfo["checkpoint_sha256"]))
    if decl["learned"]["vote_all_families"]:
        ck_k = decl["learned"]["choice_pool_size"]
        for s, fams in frozen_all.items():
            for fam, f in fams.items():
                plan.append((f"F{fam}s{s}_o{i}_p{ck_k}", "family", s, fam, ck_k, f["dir"],
                             f["sha256"]["best_model.meta.json"], f["checkpoint_sha256"]))
    votes = []
    for arm_id, role, s, fam, k, mdir, ssha, csha in plan:
        vd = P["vote"] / arm_id
        common = ["--survivors", str(surv), "--dataset", decl["inputs"]["dataset"]["path"],
                  "--seal-from", str(seal), "--history", *hist,
                  "--config", decl["vote"]["config"], "--k", str(k), "--pool-size", str(k),
                  "--block-start", str(b_i), "--block-end", str(b_next),
                  "--workers", str(a.vote_workers), "--out-dir", str(vd), "--arm-id", arm_id,
                  "--data-cutoff-index", str(b_i)]
        if decl["rank"]["vote_value_ordering"] != "off" and k == fr["rank_depth"]:
            common += ["--emit-value-ordering", decl["rank"]["vote_value_ordering"]]
        if role in ("learned", "family", "ranker"):
            if role == "learned":
                wj = P["winner"] / f"s{s}" / "WINNER.json"
                if not wj.is_file() and not mut("nocheck"):
                    raise LeakCheck(f"L4 origin {i}: winner s{s} not frozen before the vote")
            extra = ["--models-dir", mdir, "--expect-sidecar-sha256", ssha, "--expect-checkpoint-sha256", csha]
            gpu = decl["vote"]["allow_gpu"]
            if gpu:
                extra.append("--allow-gpu")
        else:
            extra = ["--ranking", "field_sum", "--ranking-fields", ",".join(decl["unlearned"]["fields"]),
                     "--ranking-npz", str(P["copy"] / "bidirectional_survivors_binary.npz"),
                     "--tie-break", decl["unlearned"]["tie_break"]]
            gpu = False
        venv = base_env(P["scratch"], a.gpu_id if gpu else "")
        R.step(f"o{i}/vote/{arm_id}", py() + [decl["vote"]["harness"]] + common + extra, P["o"], venv,
               P["vote"] / f"{arm_id}.log", P["vote"] / f"{arm_id}.DONE.json",
               lambda vd=vd: {"pools": vd / "pools.jsonl", "record": vd / "run_record.json"}, gpu=gpu)
        vr = json.load(open(vd / "run_record.json"))
        votes.append({"arm_id": arm_id, "role": role, "split_seed": s, "family": fam, "pool_size": k,
                      "dir": str(vd), "config_id": vr["config_id"],
                      "pools_sha256": sha256_file(vd / "pools.jsonl"),
                      "model_type": (vr.get("ranking_info") or {}).get("model_type")})
    rec["votes"] = votes

    # ---- 6. grades (DEVELOPMENT, block i only) ---------------------------------------
    P["grade"].mkdir(exist_ok=True)
    prev_bt = fr["block_table"][i - 1] if i > 0 else None
    gb_start, gb_end = ((prev_bt["start_index"], prev_bt["end_index"]) if mut("grade_previous_block") and prev_bt
                        else (b_i, b_next))
    unseen = b_i
    rec["leak_checks"]["L6_grade_block"] = {"grade_block": [gb_start, gb_end], "upstream_unseen": unseen,
                                            "ok": [gb_start, gb_end] == [b_i, b_next]}
    if [gb_start, gb_end] != [b_i, b_next] and not mut("nocheck"):
        raise LeakCheck(f"L6 origin {i}: grade block {[gb_start, gb_end]} != block i {[b_i, b_next]}")
    for s in winners:
        wj = P["winner"] / f"s{s}" / "WINNER.json"
        if not wj.is_file() or sha256_file(wj) != winners[s]["winner_json_sha256"]:
            raise LeakCheck(f"L4 origin {i}: WINNER.json s{s} missing or changed before grading")
    abst = P["grade"] / "abstention.json"
    write_json(abst, {"schema": "tfm.e2e.abstention.v1", "allow_abstention": False,
                      "allowed_reasons": None, "empty_pool": "error"})
    grades = []
    for sess in fr["grade_sessions"]:
        ss = fr["dataset_sessions"] if sess == "pooled" else [sess]
        gbp = P["grade"] / f"grade_block_{sess}.json"
        write_json(gbp, {"schema": "tfm.e2e.grade_block.v1",
                         "name": f"{LABEL} walk-forward origin {i} block [{gb_start},{gb_end}) {sess}",
                         "start_index": gb_start, "end_index": gb_end, "sessions": ss,
                         "frequency_control_block": {"start_index": 0, "end_index": b_i, "sessions": ss},
                         "mode": "development"})
        for v in votes:
            gid = f"{v['arm_id']}_{sess}"
            armp = P["grade"] / f"arms_{gid}.json"
            write_json(armp, {"schema": "tfm.e2e.arms.v1", "arms": [{
                "arm_id": v["arm_id"], "config_id": v["config_id"], "upstream_unseen_from_index": unseen,
                "min_k": 1, "max_k": v["pool_size"], "previously_inspected_blocks": fr["previously_inspected"],
                "description": f"{LABEL} {v['role']} arm, origin {i}, pool {v['pool_size']}"}],
                "family_alpha": 0.05, "correction": "bonferroni"})
            gd = P["grade"] / gid
            mode = "counted" if mut("counted_mode") else "development"
            argv = py() + [decl["grader"]["path"], "--pools", str(Path(v["dir"]) / "pools.jsonl"),
                           "--pools-sha256", v["pools_sha256"],
                           "--dataset", decl["inputs"]["dataset"]["path"],
                           "--dataset-sha256", decl["inputs"]["dataset"]["sha256"],
                           "--grade-block", str(gbp), "--arms", str(armp), "--abstention", str(abst),
                           "--out-dir", str(gd), "--run-label", f"{LABEL} walk-forward o{i} {gid}",
                           "--out-of-block-records", ("error" if sess == "pooled" and
                                                      set(ss) >= set(fr["dataset_sessions"]) else
                                                      "exclude_and_count"),
                           "--control-reps", str(a.control_reps), "--workers", str(a.grader_workers),
                           "--no-provenance", "--mode", mode, "--per-draw-detail", "on",
                           "--rank-summary", "on", "--rank-test-reps", str(a.rank_test_reps)]
            R.step(f"o{i}/grade/{gid}", argv, P["o"], cpu_env, P["grade"] / f"{gid}.log",
                   P["grade"] / f"{gid}.DONE.json",
                   lambda gd=gd: {"summary": gd / "summary.json", "per_draw": gd / "per_draw.jsonl",
                                  "detail": gd / "per_draw_detail.jsonl", "marker": gd / "DEVELOPMENT"})
            sm = json.load(open(gd / "summary.json"))
            if str(sm.get("mode", "")).upper() != "DEVELOPMENT" or not (gd / "DEVELOPMENT").is_file():
                raise LeakCheck(f"o{i} {gid}: grader output is not stamped DEVELOPMENT")
            if not sm.get("controls_ok"):
                raise StepFailed(f"o{i} {gid}: grader controls failed; results not read")
            r0 = sm["results"][0]
            st = r0["stats"]
            # rank (RANK_RULE), from the pool as written and the grader's drawn number
            pools = {json.loads(l)["target"]["combined_index"]: json.loads(l)["pool"]
                     for l in open(Path(v["dir"]) / "pools.jsonl")}
            rrows, mism = [], 0
            ordered = decl["rank"]["vote_value_ordering"] != "off" and v["pool_size"] == fr["rank_depth"]
            for line in open(gd / "per_draw_detail.jsonl"):
                row = json.loads(line)
                if row.get("status") != "graded":
                    continue
                rk, cens, d, var = rank_of(pools[row["combined_index"]], row["drawn"])
                if mut("rank_off_by_one"):
                    rk += 1
                if not ordered and row.get("rank") is not None and abs(float(row["rank"]) - rk) > 1e-9:
                    mism += 1
                rrows.append({"combined_index": row["combined_index"], "rank": rk, "rank_censored": cens,
                              "rank_distinct_d": d, "rank_null_var": var, "rank_grader": row.get("rank")})
            if mism:
                raise StepFailed(f"o{i} {gid}: driver and grader ranks differ on {mism} draws (ordering off)")
            grades.append({"arm_id": v["arm_id"], "role": v["role"], "split_seed": v["split_seed"],
                           "family": v["family"], "pool_size": v["pool_size"], "session": sess,
                           "n_graded": st["n_graded"], "hits": st["hits"],
                           "expected_hits": st["expected_hits"], "null_sd": st["null_sd"],
                           "sum_k": st["sum_k"], "p_one_sided": st["p_one_sided"],
                           "lift_point": st["lift_point"], "lift_ci": r0.get("lift_ci_descriptive"),
                           "development_above_background": r0.get("development_above_background"),
                           "n_abstained": r0.get("n_abstained"), "n_pending": r0.get("n_pending"),
                           "previously_inspected_overlaps": r0.get("previously_inspected_overlaps"),
                           "rank": rank_summary(rrows) if v["pool_size"] == fr["rank_depth"] or
                           v["role"] == "family" else None,
                           "_rank_rows": rrows,
                           "rank_grader": r0.get("rank_summary"),
                           "rank_check": ("full ordering: grader rank is full-depth, driver rank is censored"
                                          if ordered else "driver == grader on every graded draw"),
                           "summary_sha256": sha256_file(gd / "summary.json"),
                           "detail_sha256": sha256_file(gd / "per_draw_detail.jsonl"),
                           "headline": r0.get("headline"), "dir": str(gd)})

    # ---- 7. per-draw hit file --------------------------------------------------------
    pdh = P["o"] / "per_draw_hits.jsonl"
    with open(str(pdh) + ".partial", "w") as f:
        for g in grades:
            rk = {r["combined_index"]: r for r in g["_rank_rows"]}
            for line in open(Path(g["dir"]) / "per_draw_detail.jsonl"):
                row = json.loads(line)
                if row.get("status") == "excluded_out_of_block":
                    continue
                rr = rk.get(row.get("combined_index"), {})
                row.update(label=LABEL, origin=i, role=g["role"], split_seed=g["split_seed"],
                           family=g["family"], pool_size_declared=g["pool_size"], session_view=g["session"],
                           rank_driver=rr.get("rank"), rank_censored=rr.get("rank_censored"),
                           rank_distinct_d=rr.get("rank_distinct_d"))
                f.write(canon(row) + "\n")
    os.replace(str(pdh) + ".partial", pdh)
    for g in grades:
        g.pop("_rank_rows")
    rec["grades"] = grades
    rec["per_draw_hits"] = {"path": str(pdh), "sha256": sha256_file(pdh)}
    rec["finished"] = now()
    write_json(P["o"] / "origin_summary.json", rec)
    return rec


def choose(R, a, decl, P, i, seed, fam_res, frozen, prev_rec, b_i):
    """Model choice (L4). cv: the declared winner rule on Step 5 check reports. prev_block_*:
    the previous graded origin's per-family grades, whose block ends at or before b_i."""
    elig = {f: r for f, r in fam_res.items() if r.get("eligible") and f in frozen}
    if not elig:
        raise StepFailed(f"o{i} s{seed}: no eligible family ({ {f: r.get('reason') or r.get('problems') for f, r in fam_res.items()} })")
    perms = {r["first_permutation_sha256"] for r in elig.values()}
    if len(perms) != 1:
        raise StepFailed(f"o{i} s{seed}: families saw different splits {perms} (same-folds rule)")
    mc = decl["learned"]["model_choice"]
    rule, evidence, metric = "cv", [], {}
    if mc != "cv" and prev_rec is not None:
        ck, cs = decl["learned"]["choice_pool_size"], decl["learned"]["choice_session"]
        for g in prev_rec["grades"]:
            if g["role"] == "family" and g["split_seed"] == seed and g["pool_size"] == ck and g["session"] == cs:
                if g["family"] in elig:
                    metric[g["family"]] = (g["hits"] if mc == "prev_block_hits" else -g["rank"]["mean_rank"])
                    evidence.append([prev_rec["block"]["start_index"], prev_rec["block"]["end_index"],
                                     g["summary_sha256"]])
        if mut("choice_uses_current_block"):
            evidence.append([b_i, b_i + 1, "mutant"])
        if metric:
            rule = mc
    if rule == "cv":
        best = max(r["cv_score"] for r in elig.values())
        ties = [f for f in FAMILY_TIE_ORDER if f in elig and elig[f]["cv_score"] == best]
    else:
        best = max(metric.values())
        cand = [f for f in metric if metric[f] == best]
        top_cv = max(elig[f]["cv_score"] for f in cand)
        ties = [f for f in FAMILY_TIE_ORDER if f in cand and elig[f]["cv_score"] == top_cv]
    fam = ties[0]
    w = elig[fam]
    wd = P["winner"] / f"s{seed}"
    if (wd / "WINNER.json").exists():
        if not a.resume:
            raise Refused(f"{wd} exists")
        prev = json.load(open(wd / "WINNER.json"))
        if prev["model_type"] != fam or prev["checkpoint_sha256"] != w["checkpoint_sha256"]:
            raise Refused(f"resume: frozen winner {wd} differs from the rule's choice now")
        prev["winner_json_sha256"] = sha256_file(wd / "WINNER.json")
        return prev
    wd.mkdir(parents=True)
    fz = Path(frozen[fam]["dir"])
    for n in frozen[fam]["sha256"]:
        shutil.copyfile(fz / n, wd / n)
    shas = {n: sha256_file(wd / n) for n in frozen[fam]["sha256"]}
    if shas != frozen[fam]["sha256"]:
        raise StepFailed(f"o{i} s{seed}: winner copy differs from the frozen family")
    out = {"label": LABEL, "origin": i, "split_seed": seed, "model_type": fam,
           "cv_score": w["cv_score"], "rule_declared": mc, "rule_used": rule,
           "rule_text": WINNER_RULE if rule == "cv" else decl["learned"]["choice_rule"][rule],
           "choice_metric": metric or None, "evidence_blocks": evidence,
           "cv_ranking": sorted(([f, r["cv_score"]] for f, r in elig.items()), key=lambda x: -x[1]),
           "ranking": sorted(([f, r["cv_score"]] for f, r in elig.items()), key=lambda x: -x[1]),
           "ties_at_best": ties, "ineligible": {f: r for f, r in fam_res.items() if not r.get("eligible")},
           "first_permutation_sha256": next(iter(perms)), "frozen_dir": str(wd), "frozen_sha256": shas,
           "checkpoint_path_in_sidecar": w["checkpoint_path"], "checkpoint_sha256": w["checkpoint_sha256"],
           "frozen_at": now(),
           "inputs_used": ("Step 5 check reports (cv_score) only" if rule == "cv" else
                           "previous origin's per-family DEVELOPMENT grades (blocks listed) + cv for ties")}
    write_json(wd / "WINNER.json", out)
    make_read_only(wd)
    out["winner_json_sha256"] = sha256_file(wd / "WINNER.json")
    R.say(f"o{i} s{seed}: winner {fam} by {rule} (cv {w['cv_score']:.6g}; metric {metric or '-'}) frozen in {wd}")
    return out


# --------------------------------------------------------------------------- table ------
def write_table(out: Path, decl, recs):
    rows = []
    for r in recs:
        for g in r["grades"]:
            rows.append({"origin": r["origin"], "block": [r["block"]["start_index"], r["block"]["end_index"]],
                         "dates": [r["block"]["first_date"], r["block"]["last_date"]],
                         "train_draws": r["train"][1], "label_block": r["label_block"],
                         **{k: g[k] for k in ("arm_id", "role", "split_seed", "family", "pool_size", "session",
                                              "n_graded", "hits", "expected_hits", "null_sd",
                                              "p_one_sided", "lift_point", "rank", "rank_grader")},
                         "winner_rule": ((r["step5"].get(f"s{g['split_seed']}") or {}).get("rule_used")
                                         if g["role"] == "learned" else None)})
    if mut("table_drop_origin") and rows:
        rows = [x for x in rows if x["origin"] != rows[-1]["origin"]]
    want = {(r["origin"], g["arm_id"], g["session"]) for r in recs for g in r["grades"]}
    have = {(x["origin"], x["arm_id"], x["session"]) for x in rows}
    if want != have:
        raise StepFailed(f"table incomplete: missing {sorted(want - have)}")
    tab = {"label": LABEL, "label_text": LABEL_TEXT, "kind": "e2e_walk_forward_table",
           "run_record_sha256": sha256_file(out / "run_record.json"), "rows": rows, "rank_rule": RANK_RULE,
           "note": ("Rows are not independent: the same survivors vote in every origin, pools are "
                    "nested across pool sizes, arms share draws, and 'pooled' contains the session rows. "
                    "No combined p-value is given. Background: each draw uniform on 0-999, independent "
                    "(assumed).")}
    write_json(out / "table.json", tab)
    L = [f"# Walk-forward table — {LABEL}", "", LABEL_TEXT + ".", "",
         f"Run record sha256 `{tab['run_record_sha256']}`. {tab['note']}", "",
         "## Hits against k/1000", "",
         "| origin | block | dates | train | session | pool | arm | family | graded | hits | expected | null sd | p (one-sided, unadj.) | lift |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = {s: n for n, s in enumerate(decl["frame"]["grade_sessions"])}
    for x in sorted(rows, key=lambda x: (order.get(x["session"], 9), x["pool_size"], x["origin"],
                                         ["learned", "unlearned", "family", "ranker"].index(x["role"]), x["arm_id"])):
        L.append(f"| {x['origin']} | [{x['block'][0]},{x['block'][1]}) | {x['dates'][0]}..{x['dates'][1]} | "
                 f"{x['train_draws']} | {x['session']} | {x['pool_size']} | {x['arm_id']} | {x['family'] or '-'} | "
                 f"{x['n_graded']} | {x['hits']} | {x['expected_hits']:.2f} | {x['null_sd']:.2f} | "
                 f"{x['p_one_sided']:.4g} | {x['lift_point'] if x['lift_point'] is None else round(x['lift_point'], 3)} |")
    L += ["", f"## Rank of the drawn number (depth {decl['frame']['rank_depth']}; lower is better; uniform expectation 500.5)",
          "", RANK_RULE, "",
          "Grader columns: R24 rank summary (mid-ranks, exact null, Monte Carlo p toward small ranks); "
          "driver columns: RANK_RULE (equal to the grader's when the full ordering is off).", "",
          "| origin | session | arm | family | n | mean rank (grader) | expected | p lower (MC) | mean rank (driver, censored) | censored draws |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for x in sorted(rows, key=lambda x: (order.get(x["session"], 9), x["origin"], x["arm_id"])):
        if x["rank"]:
            q, gq = x["rank"], x.get("rank_grader") or {}
            fmt = lambda v, d=1: "-" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))
            L.append(f"| {x['origin']} | {x['session']} | {x['arm_id']} | {x['family'] or '-'} | {q['n']} | "
                     f"{fmt(gq.get('mean_rank'))} | {fmt(gq.get('expected_mean_rank'))} | "
                     f"{fmt(gq.get('p_lower_monte_carlo'), 4)} | {q['mean_rank']:.1f} | {q['censored']} |")
    L += ["", "## Sums over origins (plain arithmetic; rows are not independent)", "",
          "| session | pool | arm role | graded | hits | expected |", "|---|---|---|---|---|---|"]
    agg = {}
    for x in rows:
        key = (x["session"], x["pool_size"], x["role"] + (f" {x['family']}" if x["role"] == "family" else "")
               + (f" s{x['split_seed']}" if x["split_seed"] is not None else ""))
        a = agg.setdefault(key, [0, 0, 0.0])
        a[0] += x["n_graded"]
        a[1] += x["hits"]
        a[2] += x["expected_hits"]
    for key in sorted(agg, key=lambda k: (order.get(k[0], 9), k[1], k[2])):
        L.append(f"| {key[0]} | {key[1]} | {key[2]} | {agg[key][0]} | {agg[key][1]} | {agg[key][2]:.2f} |")
    (out / "TABLE.md").write_text("\n".join(L) + "\n")
    return tab


# --------------------------------------------------------------------------- plan text --
def write_plan(out: Path, decl):
    fr = decl["frame"]
    L = [f"# Walk-forward plan — {LABEL}", "", LABEL_TEXT + ".", "",
         f"Range [{fr['range'][0]}, {fr['range'][1]}); earliest workable origin {fr['earliest_workable_origin']}; "
         f"sieve window ends at {fr['sieve_window_end_index']}; label window {fr['label_window']} "
         f"(L = {fr['label_draws']}); {fr['blocks']} blocks; origins {fr['origins']}.", "",
         "| block | grade [start, end) | dates | rows | sessions | train [0, T) | labels [T, b) | previously inspected |",
         "|---|---|---|---|---|---|---|---|"]
    for b in fr["block_table"]:
        L.append(f"| {b['block']} | [{b['start_index']}, {b['end_index']}) | {b['first_date']}..{b['last_date']} | "
                 f"{b['rows']} | {b['sessions']} | {b['train'][1]} | [{b['labels'][0]}, {b['labels'][1]}) | "
                 f"{b['overlaps_previously_inspected'] or '-'} |")
    L += ["", "Per origin: split files -> run copy -> Step 3 -> Step 5 per seed x family (shim) -> check -> "
          "freeze every family -> choose (rule) -> votes sealed at b_i (pool sizes x arms; family arms at the "
          "choice pool size when voted) -> grader (development) per vote x session view."]
    (out / "PLAN.md").write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------- main -------
def main(argv=None) -> int:
    a = parse_args(argv)
    try:
        out = Path(os.path.abspath(Path(a.out_dir).expanduser()))
        if not is_under(out.parent if not out.exists() else out, WT / "data"):
            raise Refused(f"--out-dir {out} must lie under {WT / 'data'}")
        if any(is_under(out.parent if not out.exists() else out, f) for f in FORBIDDEN):
            raise Refused(f"--out-dir {out} lies under a forbidden root")
        if out.exists() and not a.resume:
            raise Refused(f"--out-dir {out} exists (use --resume to continue a run)")
        if a.check_only and (a.plan_only or a.resume):
            raise Refused("--check-only cannot be combined with --plan-only or --resume")
        decl, idx, draws = build_declaration(a)
        if a.check_only:
            print(f"[{LABEL}] PREFLIGHT OK: {out} would be written; pipeline switches "
                  f"{ {k: v['applied'] for k, v in decl['pipeline_switches'].items()} }; nothing written",
                  flush=True)
            return 0
        if a.plan_only:
            del draws
        if out.exists():
            prev = json.load(open(out / "run_record.json"))
            strip = lambda d: {k: v for k, v in d.items() if k not in ("written",)}
            if canon(strip(prev)) != canon(strip(decl)):
                raise Refused("resume: the declaration differs from the existing run_record.json")
        else:
            out.mkdir(parents=False)
            decl["written"] = now()
            write_json(out / "run_record.json", decl)
            write_plan(out, decl)
    except Refused as e:
        print(f"[{LABEL}] REFUSED: {e}", file=sys.stderr, flush=True)
        return 2
    rsha = sha256_file(out / "run_record.json")
    if a.plan_only:
        print(f"[{LABEL}] plan written: {out / 'run_record.json'} (sha256 {rsha}), {out / 'PLAN.md'}")
        return 0
    R = Runner(a, out, rsha)
    R.say(f"run record {rsha}; origins {decl['frame']['origins']}; mutant {mutant()}")
    recs = []
    pins = [(decl["vote"]["harness"], decl["vote"]["harness_sha256"]),
            (decl["grader"]["path"], decl["grader"]["sha256"])]
    try:
        for i in decl["frame"]["origins"]:
            for p, s in pins:
                if sha256_file(p) != s:
                    raise Refused(f"pinned file changed during the run: {p}")
            recs.append(run_origin(R, a, decl, i, idx, draws, recs[-1] if recs else None))
        for p, s in pins:
            if sha256_file(p) != s:
                raise Refused(f"pinned file changed during the run: {p}")
        tab = write_table(out, decl, recs)
    except Refused as e:
        R.say(f"REFUSED: {e}")
        return 2
    except StepFailed as e:
        R.say(f"STEP FAILED (stopped, no retry): {e}")
        return 3
    except LeakCheck as e:
        R.say(f"LEAK CHECK FAILED: {e}")
        return 4
    except GuardStop as e:
        R.say(f"GUARD STOP: {e}")
        return 5
    R.say(f"done: {len(recs)} origins, {len(tab['rows'])} table rows; {out / 'TABLE.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
