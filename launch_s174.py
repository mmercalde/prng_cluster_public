#!/usr/bin/env python3
"""
S174 Python Launcher — final

TB-approved orchestration wrapper for crash-reproduction experiments.
Replaces bash launchers that silently truncated post-run artifacts.

==============================================================================
SLICE 1 (approved): argparse, mode→config, RUN_ID, paths, provenance,
                    preflight, optimizer argv/env construction
SLICE 2 (approved): subprocess.Popen + stream-to-log + SIGINT/SIGTERM
                    handling + try/finally + exit-code capture + sentinel
                    classification + summary.{json,txt} guarantees
SLICE 3 (approved): post-child observation window with per-rig snapshots
                    + bundle assembly with all S173 v2 forensic
                    instrumentation + final-summary mirror into bundle/local/
SLICE 4 (final):    real semantic classification (READY_GATE_FAILED /
                    VALID_CLEAN / VALID_ACTIVE_FAULT /
                    VALID_POST_COMPLETION_FAULT / INTERRUPTED_BUNDLED /
                    PYTHON_EXIT_NONZERO_NO_FAULT / INVALID_MISSING_EVIDENCE)
                    derived from log content + bundle evidence

==============================================================================
TB DESIGN CONSTRAINTS (from S174 launcher rewrite ruling):
  Allowed:    launch window_optimizer.py, monitor process state, collect
              logs/snapshots, enforce observation window, write bundle and
              summary, handle SIGTERM/SIGINT
  Not allowed: change coordinator behavior, change worker dispatch behavior,
              change PWC TCP protocol, change S173 instrumentation semantics

This launcher is orchestration only. Coordinator hard-gate is the source of
truth for readiness; this launcher does not re-implement gating logic.

==============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any


# =============================================================================
# Constants — known cluster topology (matches existing bash launchers / cluster
# operating manual). Hardcoded because the experiment depends on a fixed cluster.
# =============================================================================

RIG_HOSTS: List[str] = [
    "192.168.3.120",   # rrig6600
    "192.168.3.154",   # rrig6600b
    "192.168.3.162",   # rrig6600c
]

REPO_ROOT_DEFAULT = Path.home() / "distributed_prng_analysis"
VENV_ACTIVATE = Path.home() / "venvs" / "torch" / "bin" / "activate"
# TB Fix 1: explicit interpreter path — must NOT fall back to sys.executable
# (system Python). The old bash launchers sourced ~/venvs/torch/bin/activate;
# Python equivalent is to invoke the venv's interpreter directly.
OPTIMIZER_PYTHON_DEFAULT = Path.home() / "venvs" / "torch" / "bin" / "python3"

# Required CLI flags that must be present in window_optimizer.py --help output
# Mirrors the bash launchers' preflight check.
REQUIRED_OPTIMIZER_FLAGS: List[str] = [
    "--pwc-transport",
    "--min-workers",
    "--worker-pool-size",
    "--seed-cap-amd",
    "--seed-cap-nvidia",
    "--max-seeds",
]

# Additional flags required only for forced-config (warm-start) runs like D1
REQUIRED_OPTIMIZER_FLAGS_FORCED: List[str] = [
    "--warm-start-window",
    "--warm-start-offset",
    "--warm-start-skip-min",
    "--warm-start-skip-max",
    "--warm-start-fwd-thresh",
    "--warm-start-rev-thresh",
    "--warm-start-session-idx",
]


# =============================================================================
# Config model
# =============================================================================

@dataclass
class ForcedConfig:
    """Forced warm-start parameters for crash reproduction (e.g. D1 FT=0.73)."""
    window: int
    offset: int
    skip_min: int
    skip_max: int
    fwd_thresh: float
    rev_thresh: float
    session_idx: int = 0

    def to_cli_args(self) -> List[str]:
        return [
            "--warm-start-window",       str(self.window),
            "--warm-start-offset",       str(self.offset),
            "--warm-start-skip-min",     str(self.skip_min),
            "--warm-start-skip-max",     str(self.skip_max),
            "--warm-start-fwd-thresh",   str(self.fwd_thresh),
            "--warm-start-rev-thresh",   str(self.rev_thresh),
            "--warm-start-session-idx",  str(self.session_idx),
        ]

    def label(self) -> str:
        return (
            f"W{self.window}_O{self.offset}_S{self.skip_min}-{self.skip_max}"
            f"_FT{self.fwd_thresh}_RT{self.rev_thresh}"
        )


@dataclass
class LauncherConfig:
    """All inputs needed to construct a deterministic run."""
    run_id_prefix: str          # e.g. "S174_D1_FT073_50K_425M"
    purpose: str                # human-readable banner text
    pool: int                   # --worker-pool-size
    chunk: int                  # --seed-cap-amd / --seed-cap-nvidia
    max_seeds: int              # --max-seeds
    min_workers: int            # --min-workers (passed to coordinator gate)
    trials: int                 # --trials
    lottery_file: str           # --lottery-file (default daily3.json)
    prng_type: str              # --prng-type (default java_lcg)
    strategy: str               # --strategy (default bayesian)
    seed_start: int             # --seed-start
    forced_config: Optional[ForcedConfig]   # None = open Optuna
    repo_root: Path
    optimizer_python: Path           # TB Fix 1: explicit interpreter for window_optimizer.py
    observation_window_minutes: int = 10
    observation_snapshot_interval_sec: int = 60

    def expected_chunks_total(self) -> int:
        return self.max_seeds // self.chunk

    def expected_chunks_per_amd_worker(self) -> int:
        # 24 AMD workers (3 rigs × 8 GPUs); Zeus contributes 2 NVIDIA workers
        # but those have separate seed cap and aren't the bottleneck.
        return self.max_seeds // self.chunk // 24


# =============================================================================
# Provenance
# =============================================================================

@dataclass
class Provenance:
    """
    Captured at launcher start; embedded in summary.json + summary.txt.

    TB Fix 3: records BOTH the launcher's argv AND the optimizer's argv +
    env actually passed, so D1 forensic reproducibility is exact (not
    derived-from-mode).
    """
    run_id: str
    started_at_iso: str
    git_sha: str
    git_branch: str
    git_dirty: bool
    launcher_argv: List[str]              # argv given to this launcher
    optimizer_argv: List[str]             # argv that will be Popen'd
    optimizer_env_subset: Dict[str, str]  # env vars set/added for the child
    launcher_env_subset: Dict[str, str]   # ambient env at launcher start
    cluster_rigs: List[str]
    repo_root: str
    optimizer_python: str
    launcher_path: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Environment variables the launcher records (not all of os.environ — would
# leak secrets and bloat summary). These mirror what the bash launchers set.
ENV_VARS_RECORDED: List[str] = [
    "PRNG_PWC_STARTUP_DIAG",
    "PRNG_PWC_FIRST_ASSIGN_JITTER_SEC",
    "PRNG_PWC_PER_WORKER_MIN_GAP_SEC",
    "S163_MEM_DEBUG",
    "PYTHONPATH",
    "PYTHONUNBUFFERED",
    "ROCM_PATH",
    "HSA_OVERRIDE_GFX_VERSION",
    "VIRTUAL_ENV",
]


def capture_provenance(
    run_id: str,
    cfg: LauncherConfig,
    launcher_argv: List[str],
    optimizer_argv: List[str],
    optimizer_env: Dict[str, str],
) -> Provenance:
    """
    Snapshot run-time provenance. Called BEFORE subprocess starts.

    TB Fix 3: takes the actual optimizer_argv + env that will be Popen'd, so
    summary records what was *really* sent to the child, not what mode would
    derive in the abstract.
    """
    git_sha = "unknown"
    git_branch = "unknown"
    git_dirty = False
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cfg.repo_root), text=True
        ).strip()
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(cfg.repo_root), text=True
        ).strip()
        # Dirty = any uncommitted change in tracked files
        diff_out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(cfg.repo_root), text=True
        ).strip()
        git_dirty = bool(diff_out)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # git not available or not a repo — record sentinel and continue
        pass

    # Launcher's ambient env (vars present at launcher start that we record)
    launcher_env_subset = {k: os.environ[k] for k in ENV_VARS_RECORDED if k in os.environ}

    # Child's env subset = what we explicitly set/forwarded for the optimizer
    optimizer_env_subset = {k: optimizer_env[k] for k in ENV_VARS_RECORDED if k in optimizer_env}

    return Provenance(
        run_id=run_id,
        started_at_iso=datetime.now(timezone.utc).astimezone().isoformat(),
        git_sha=git_sha,
        git_branch=git_branch,
        git_dirty=git_dirty,
        launcher_argv=list(launcher_argv),
        optimizer_argv=list(optimizer_argv),
        optimizer_env_subset=optimizer_env_subset,
        launcher_env_subset=launcher_env_subset,
        cluster_rigs=list(RIG_HOSTS),
        repo_root=str(cfg.repo_root),
        optimizer_python=str(cfg.optimizer_python),
        launcher_path=str(Path(__file__).resolve()),
    )


# =============================================================================
# RUN_ID generation
# =============================================================================

def generate_run_id(prefix: str) -> str:
    """
    Generate timestamped RUN_ID. Format: <prefix>_YYYYMMDD_HHMMSS

    Examples:
      S174_D1_FT073_50K_425M_20260508_124530
      S174_GATE_VALIDATION_20260508_124530
      S174_NEGATIVE_TEST_20260508_124530
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


# =============================================================================
# Path layout
# =============================================================================

@dataclass
class RunPaths:
    """All filesystem paths for a single run. Computed once from RUN_ID."""
    run_id: str
    repo_root: Path
    logs_dir: Path
    run_log: Path                       # window_optimizer.py stdout/stderr
    launcher_log: Path                  # this launcher's own log
    summary_txt: Path
    summary_json: Path
    bundle_dir: Path
    observation_dir: Path
    classification_file: Path           # one-liner: VALID_CLEAN | etc.

    @classmethod
    def for_run(cls, run_id: str, repo_root: Path) -> "RunPaths":
        logs = repo_root / "logs"
        bundle = logs / f"{run_id}_bundle"
        return cls(
            run_id=run_id,
            repo_root=repo_root,
            logs_dir=logs,
            run_log=logs / f"{run_id}.log",
            launcher_log=logs / f"{run_id}_launcher.log",
            summary_txt=logs / f"{run_id}_summary.txt",
            summary_json=logs / f"{run_id}_summary.json",
            bundle_dir=bundle,
            observation_dir=bundle / "observation_window",
            classification_file=bundle / "classification.txt",
        )

    def ensure_dirs(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        self.observation_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# window_optimizer.py command construction
# =============================================================================

def build_optimizer_argv(cfg: LauncherConfig, run_log_path: Path) -> List[str]:
    """
    Construct the argv to launch window_optimizer.py.
    Returns a list (no shell). Output redirection handled by Popen.

    TB Fix 1: uses cfg.optimizer_python (explicit venv interpreter), NOT
    sys.executable. Prevents accidentally invoking window_optimizer with
    system Python if the launcher itself was started outside the venv.

    TB Fix 2 (Slice 2 follow-up): -u disables Python output buffering so
    the child's stdout reaches our stream-to-log loop in near-real time.
    Combined with PYTHONUNBUFFERED=1 in env, this prevents block-buffering
    when stdout is a pipe (the default Python behavior). Critical for D1
    forensics where READY GATE PASSED / dispatch confirmed / chunk lines
    must appear promptly to correlate with cluster-side faults.
    """
    argv = [
        str(cfg.optimizer_python), "-u", "window_optimizer.py",
        "--strategy",         cfg.strategy,
        "--lottery-file",     cfg.lottery_file,
        "--trials",           str(cfg.trials),
        "--output",           "optimal_window_config.json",
        "--prng-type",        cfg.prng_type,
        "--use-persistent-workers",
        "--pwc-transport",    "tcp",
        "--min-workers",      str(cfg.min_workers),
        "--worker-pool-size", str(cfg.pool),
        "--seed-cap-amd",     str(cfg.chunk),
        "--seed-cap-nvidia",  str(cfg.chunk),
        "--max-seeds",        str(cfg.max_seeds),
        "--seed-start",       str(cfg.seed_start),
    ]
    if cfg.forced_config is not None:
        argv.extend(cfg.forced_config.to_cli_args())
    return argv


def build_optimizer_env(cfg: LauncherConfig) -> Dict[str, str]:
    """
    Environment vars to set for window_optimizer.py subprocess.
    Inherits parent env, adds S174 instrumentation toggles.

    TB Fix 2: PYTHONPATH derives from cfg.repo_root (NOT REPO_ROOT_DEFAULT)
    so --repo-root is honored.

    TB Slice 2 Fix 2: PYTHONUNBUFFERED=1 prevents block-buffering of stdout
    when the launcher pipes the child. Belt-and-suspenders with -u flag in
    argv — set both because either can be defeated by certain library
    init paths in window_optimizer.py.
    """
    env = os.environ.copy()
    env["PRNG_PWC_STARTUP_DIAG"] = "1"
    env["PRNG_PWC_FIRST_ASSIGN_JITTER_SEC"] = "3"
    env["PRNG_PWC_PER_WORKER_MIN_GAP_SEC"] = "0.02"
    env["S163_MEM_DEBUG"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    # PYTHONPATH must include selected repo root so local imports resolve
    repo_root_str = str(cfg.repo_root)
    existing_pp = env.get("PYTHONPATH", "")
    if repo_root_str not in existing_pp.split(":"):
        env["PYTHONPATH"] = f"{repo_root_str}:{existing_pp}" if existing_pp else repo_root_str
    return env


# =============================================================================
# Preflight (preserved from bash launchers — verifies CLI surface before launch)
# =============================================================================

def preflight_check(cfg: LauncherConfig) -> List[str]:
    """
    Run window_optimizer.py --help and verify all required flags are present.
    Returns list of missing flags (empty list = preflight pass).

    NOT a coordinator-side check — purely sanity that the binary we're about
    to invoke supports the flags we plan to pass.

    TB Fix 1: also validates the optimizer_python interpreter exists and is
    executable, since we rely on the venv's Python for ROCm/torch/optuna deps.
    """
    required = list(REQUIRED_OPTIMIZER_FLAGS)
    if cfg.forced_config is not None:
        required.extend(REQUIRED_OPTIMIZER_FLAGS_FORCED)

    # Validate the interpreter exists before trying to use it
    if not cfg.optimizer_python.exists():
        return [f"<INTERPRETER_MISSING: {cfg.optimizer_python}>"] + required
    if not os.access(cfg.optimizer_python, os.X_OK):
        return [f"<INTERPRETER_NOT_EXECUTABLE: {cfg.optimizer_python}>"] + required

    try:
        help_text = subprocess.check_output(
            [str(cfg.optimizer_python), "window_optimizer.py", "--help"],
            cwd=str(cfg.repo_root), text=True, stderr=subprocess.STDOUT,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        # If --help itself fails, treat all flags as missing (forces operator
        # to fix the binary before launching)
        return required

    return [flag for flag in required if flag not in help_text]


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="S174 Python launcher — orchestration wrapper for "
                    "window_optimizer.py crash-reproduction experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Gate validation (positive case, 5M seeds)\n"
            "  python3 launch_s174.py --mode gate-validation\n\n"
            "  # Negative test (should fail gate)\n"
            "  python3 launch_s174.py --mode negative-test --min-workers 27\n\n"
            "  # D1 forced FT=0.73 reproduction\n"
            "  python3 launch_s174.py --mode d1-forced\n\n"
            "  # Custom config (advanced)\n"
            "  python3 launch_s174.py --mode custom --pool 8 --chunk 50000 \\\n"
            "      --max-seeds 425000000 --min-workers 24 \\\n"
            "      --forced W=21,O=66,SMIN=10,SMAX=209,FT=0.73,RT=0.31\n"
        ),
    )
    p.add_argument(
        "--mode", required=True,
        choices=["gate-validation", "negative-test", "d1-forced", "baseline", "custom"],
        help="Preset run mode. 'custom' requires explicit --pool/--chunk/etc.",
    )
    # Override knobs (each preset sets defaults; CLI overrides if provided)
    p.add_argument("--pool", type=int, default=None,
                   help="Worker pool size per AMD rig (default depends on mode)")
    p.add_argument("--chunk", type=int, default=None,
                   help="Seed-cap per chunk (default depends on mode)")
    p.add_argument("--max-seeds", type=int, default=None,
                   help="Total seed budget (default depends on mode)")
    p.add_argument("--min-workers", type=int, default=None,
                   help="Coordinator hard-gate threshold (default 24)")
    p.add_argument("--trials", type=int, default=None,
                   help="Number of Optuna trials (default 1)")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--lottery-file", default="daily3.json")
    p.add_argument("--prng-type", default="java_lcg")
    p.add_argument("--strategy", default="bayesian")
    p.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT,
                   help=f"Path to distributed_prng_analysis (default {REPO_ROOT_DEFAULT})")
    p.add_argument("--optimizer-python", type=Path, default=OPTIMIZER_PYTHON_DEFAULT,
                   help=f"Python interpreter for window_optimizer.py "
                        f"(default {OPTIMIZER_PYTHON_DEFAULT}). MUST be the "
                        f"venv with ROCm/torch/optuna deps.")
    p.add_argument("--forced", default=None,
                   help="Forced config for custom mode: 'W=N,O=N,SMIN=N,SMAX=N,FT=F,RT=F'")
    p.add_argument("--observation-min", type=int, default=10,
                   help="Post-completion observation window minutes (default 10)")
    p.add_argument("--observation-interval", type=int, default=60,
                   help="Observation snapshot interval seconds (default 60)")
    p.add_argument("--dry-run", action="store_true",
                   help="Build config + show argv, do NOT execute window_optimizer")
    return p.parse_args(argv)


def parse_forced_string(s: str) -> ForcedConfig:
    """Parse 'W=21,O=66,SMIN=10,SMAX=209,FT=0.73,RT=0.31' into ForcedConfig."""
    parts = dict(kv.split("=") for kv in s.split(","))
    return ForcedConfig(
        window=int(parts["W"]),
        offset=int(parts["O"]),
        skip_min=int(parts["SMIN"]),
        skip_max=int(parts["SMAX"]),
        fwd_thresh=float(parts["FT"]),
        rev_thresh=float(parts["RT"]),
        session_idx=int(parts.get("SIDX", 0)),
    )


def _val(passed: Optional[int], default: int) -> int:
    """TB Fix 4: explicit None check (not 'or') so 0 stays 0 — caught by validate()."""
    return passed if passed is not None else default


def validate_config(cfg: LauncherConfig) -> List[str]:
    """TB Fix 4: validate numeric fields. Returns list of errors (empty = OK)."""
    errors: List[str] = []
    if cfg.pool <= 0:
        errors.append(f"pool must be > 0 (got {cfg.pool})")
    if cfg.chunk <= 0:
        errors.append(f"chunk must be > 0 (got {cfg.chunk})")
    if cfg.max_seeds <= 0:
        errors.append(f"max_seeds must be > 0 (got {cfg.max_seeds})")
    if cfg.min_workers <= 0:
        errors.append(f"min_workers must be > 0 (got {cfg.min_workers})")
    if cfg.trials <= 0:
        errors.append(f"trials must be > 0 (got {cfg.trials})")
    if cfg.observation_window_minutes < 0:
        errors.append(f"observation_window_minutes must be >= 0 (got {cfg.observation_window_minutes})")
    if cfg.observation_snapshot_interval_sec <= 0:
        errors.append(f"observation_snapshot_interval_sec must be > 0 (got {cfg.observation_snapshot_interval_sec})")
    if cfg.max_seeds % cfg.chunk != 0:
        # Not a hard error — coordinator handles the remainder. Warn only.
        pass
    return errors


# =============================================================================
# Mode → config translation
# =============================================================================

# Preset configs. Each mode is a partial LauncherConfig; CLI args fill gaps
# and override.

def config_for_mode(args: argparse.Namespace) -> LauncherConfig:
    """Translate argparse output into a LauncherConfig per the chosen mode."""
    repo_root = args.repo_root.resolve()
    optimizer_python = args.optimizer_python.resolve()

    if args.mode == "gate-validation":
        return LauncherConfig(
            run_id_prefix="S174_GATE_VALIDATION",
            purpose=("S174 gate validation — POSITIVE CASE. NOT a crash repro. "
                     "Verifies coordinator hard-gate dispatches with ready>=24."),
            pool=_val(args.pool, 8),
            chunk=_val(args.chunk, 25_000),
            max_seeds=_val(args.max_seeds, 5_000_000),
            min_workers=_val(args.min_workers, 24),
            trials=_val(args.trials, 1),
            lottery_file=args.lottery_file,
            prng_type=args.prng_type,
            strategy=args.strategy,
            seed_start=args.seed_start,
            forced_config=None,
            repo_root=repo_root,
            optimizer_python=optimizer_python,
            observation_window_minutes=args.observation_min,
            observation_snapshot_interval_sec=args.observation_interval,
        )

    if args.mode == "negative-test":
        return LauncherConfig(
            run_id_prefix="S174_NEGATIVE_TEST",
            purpose=("S174 gate negative test — expects READY GATE FAILED. "
                     "min_workers=27 > 26 GPUs forces abort path."),
            pool=_val(args.pool, 8),
            chunk=_val(args.chunk, 25_000),
            max_seeds=_val(args.max_seeds, 5_000_000),
            min_workers=_val(args.min_workers, 27),
            trials=_val(args.trials, 1),
            lottery_file=args.lottery_file,
            prng_type=args.prng_type,
            strategy=args.strategy,
            seed_start=args.seed_start,
            forced_config=None,
            repo_root=repo_root,
            optimizer_python=optimizer_python,
            observation_window_minutes=args.observation_min,
            observation_snapshot_interval_sec=args.observation_interval,
        )

    if args.mode == "d1-forced":
        return LauncherConfig(
            run_id_prefix="S174_D1_FT073_50K_425M",
            purpose=("S174 D1 — FORCED CRASH REPRODUCTION. "
                     "W21_O66_FT0.73_RT0.31 / pool=8 / chunk=50k / 425M seeds."),
            pool=_val(args.pool, 8),
            chunk=_val(args.chunk, 50_000),
            max_seeds=_val(args.max_seeds, 425_000_000),
            min_workers=_val(args.min_workers, 24),
            trials=_val(args.trials, 1),
            lottery_file=args.lottery_file,
            prng_type=args.prng_type,
            strategy=args.strategy,
            seed_start=args.seed_start,
            forced_config=ForcedConfig(
                window=21, offset=66, skip_min=10, skip_max=209,
                fwd_thresh=0.73, rev_thresh=0.31, session_idx=0,
            ),
            repo_root=repo_root,
            optimizer_python=optimizer_python,
            observation_window_minutes=args.observation_min,
            observation_snapshot_interval_sec=args.observation_interval,
        )

    if args.mode == "baseline":
        return LauncherConfig(
            run_id_prefix="S174_BASELINE_POOL8_25K",
            purpose=("S174 baseline (open Optuna, healthy-cluster control). "
                     "NOT a crash repro."),
            pool=_val(args.pool, 8),
            chunk=_val(args.chunk, 25_000),
            max_seeds=_val(args.max_seeds, 213_000_000),
            min_workers=_val(args.min_workers, 24),
            trials=_val(args.trials, 1),
            lottery_file=args.lottery_file,
            prng_type=args.prng_type,
            strategy=args.strategy,
            seed_start=args.seed_start,
            forced_config=None,
            repo_root=repo_root,
            optimizer_python=optimizer_python,
            observation_window_minutes=args.observation_min,
            observation_snapshot_interval_sec=args.observation_interval,
        )

    if args.mode == "custom":
        for required_arg in ("pool", "chunk", "max_seeds", "min_workers"):
            if getattr(args, required_arg) is None:
                raise SystemExit(f"--mode custom requires --{required_arg.replace('_','-')}")
        forced = parse_forced_string(args.forced) if args.forced else None
        return LauncherConfig(
            run_id_prefix="S174_CUSTOM",
            purpose="S174 custom run (operator-specified config).",
            pool=args.pool,
            chunk=args.chunk,
            max_seeds=args.max_seeds,
            min_workers=args.min_workers,
            trials=_val(args.trials, 1),
            lottery_file=args.lottery_file,
            prng_type=args.prng_type,
            strategy=args.strategy,
            seed_start=args.seed_start,
            forced_config=forced,
            repo_root=repo_root,
            optimizer_python=optimizer_python,
            observation_window_minutes=args.observation_min,
            observation_snapshot_interval_sec=args.observation_interval,
        )

    raise SystemExit(f"unknown mode: {args.mode}")


# =============================================================================
# Slice 2: Sentinel classifications (placeholder values — final classification
# logic in Slice 4)
# =============================================================================

SENTINEL_PRECHECK_FAILED              = "PRECHECK_FAILED"
SENTINEL_CHILD_EXIT_0                 = "CHILD_EXIT_0_UNCLASSIFIED"
SENTINEL_CHILD_EXIT_NONZERO           = "CHILD_EXIT_NONZERO_UNCLASSIFIED"
SENTINEL_INTERRUPTED                  = "INTERRUPTED_UNCLASSIFIED"
SENTINEL_LAUNCHER_EXCEPTION           = "LAUNCHER_EXCEPTION_UNCLASSIFIED"
SENTINEL_DRY_RUN                      = "DRY_RUN_NO_CHILD"

VALID_SENTINELS = {
    SENTINEL_PRECHECK_FAILED,
    SENTINEL_CHILD_EXIT_0,
    SENTINEL_CHILD_EXIT_NONZERO,
    SENTINEL_INTERRUPTED,
    SENTINEL_LAUNCHER_EXCEPTION,
    SENTINEL_DRY_RUN,
}


# =============================================================================
# Slice 4: Final semantic classifications (replace _UNCLASSIFIED sentinels
# after evidence inspection)
# =============================================================================

CLASS_VALID_CLEAN                    = "VALID_CLEAN"
CLASS_VALID_ACTIVE_FAULT             = "VALID_ACTIVE_FAULT"
CLASS_VALID_POST_COMPLETION_FAULT    = "VALID_POST_COMPLETION_FAULT"
CLASS_READY_GATE_FAILED              = "READY_GATE_FAILED"
CLASS_PYTHON_EXIT_NONZERO_NO_FAULT   = "PYTHON_EXIT_NONZERO_NO_FAULT"
CLASS_INTERRUPTED_BUNDLED            = "INTERRUPTED_BUNDLED"
CLASS_INVALID_MISSING_EVIDENCE       = "INVALID_MISSING_EVIDENCE"

VALID_CLASSIFICATIONS = {
    CLASS_VALID_CLEAN,
    CLASS_VALID_ACTIVE_FAULT,
    CLASS_VALID_POST_COMPLETION_FAULT,
    CLASS_READY_GATE_FAILED,
    CLASS_PYTHON_EXIT_NONZERO_NO_FAULT,
    CLASS_INTERRUPTED_BUNDLED,
    CLASS_INVALID_MISSING_EVIDENCE,
}

# Fault keywords that indicate a real GPU/coordinator-level fault.
# Matched case-insensitively against run_log + netconsole + observation snapshots.
# Drawn from S163KARG forensic analysis + 2026-05-03/05 manifest signatures.
FAULT_KEYWORDS: List[str] = [
    "GCVM_L2_PROTECTION_FAULT",
    "PROTECTION_FAULT",
    "amdgpu: GPU reset",
    "amdgpu: GPU recovery",
    "ring sdma",
    "ring gfx_0",
    "ring comp",
    "soft recovery failed",
    "Failed to send message",
    "SMU Failed",
    "smu_resp",
    "KIQ fence wait",
    "KIQ fence timeout",
    "WARNING: CPU:",
    "BUG:",
    "Kernel panic",
    "fence fallback timer",
    "[drm:amdgpu",
    "ring timeout",
    "process information:",  # Often appears alongside fault dump
    "amdgpu_job_timedout",
    # === Patch 1 (2026-05-10, TB-approved) — RDNA2 SMU breakdown ===
    # These are the canonical signatures of post-completion SMU
    # communication failure observed during S174 D1 (rrig6600). Without
    # these, a real VALID_POST_COMPLETION_FAULT will be misclassified
    # as VALID_CLEAN. See SESSION_CHANGELOG_20260509_S174_D1.md for the
    # forensic analysis that drove this addition.
    "response:0xFFFFFFFF",
    "Failed to retrieve enabled ppfeatures",
    "TransferTableSmu2Dram",
    "GetEnabledSmuFeaturesHigh",
    "GetEnabledSmuFeaturesLow",
]



# =============================================================================
# Slice 2: Mutable run state — set as launcher progresses, embedded in summary
# =============================================================================

@dataclass
class RunState:
    """
    Mutable state collected during the run. Constructed at launcher start,
    updated as events happen, serialized into summary.{json,txt} on exit.

    Slice 2 fills: started_at, ended_at, child_pid, child_exit_code,
    interrupted, exception_text, classification (sentinel only).

    Slice 3 fills: observation_started_at_iso, observation_ended_at_iso,
    observation_snapshot_count, bundle_started_at_iso, bundle_ended_at_iso,
    bundle_files_collected, bundle_files_missing.

    Slice 4 will replace: classification (real semantic value).
    """
    run_id: str
    started_at_iso: str
    ended_at_iso: Optional[str] = None
    child_pid: Optional[int] = None
    child_exit_code: Optional[int] = None
    child_started_at_iso: Optional[str] = None
    child_ended_at_iso: Optional[str] = None
    preflight_missing: List[str] = field(default_factory=list)
    interrupted: bool = False
    interrupt_signal: Optional[str] = None
    launcher_exception_text: Optional[str] = None
    classification: str = SENTINEL_LAUNCHER_EXCEPTION   # safest default
    config_validation_errors: List[str] = field(default_factory=list)
    # --- Slice 3 fields ---
    observation_started_at_iso: Optional[str] = None
    observation_ended_at_iso: Optional[str] = None
    observation_snapshot_count: int = 0
    observation_skipped_reason: Optional[str] = None  # if observation didn't run
    bundle_started_at_iso: Optional[str] = None
    bundle_ended_at_iso: Optional[str] = None
    bundle_files_collected: List[str] = field(default_factory=list)
    bundle_files_missing: List[str] = field(default_factory=list)
    bundle_remote_unreachable: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Slice 2: Signal handling
# =============================================================================
# We use a module-level mutable container (not a global flag in the function)
# so signal handlers can reach it without a closure.
# =============================================================================

class _SignalState:
    """Holds 'should we abort' flag so signal handlers can communicate
    with the main loop without globals."""
    def __init__(self) -> None:
        self.received_signal: Optional[str] = None

    def trip(self, signum: int) -> None:
        try:
            name = signal.Signals(signum).name
        except (ValueError, AttributeError):
            name = f"signal_{signum}"
        # Only record the FIRST signal — operator slamming Ctrl-C twice
        # shouldn't lose context of the original cause.
        if self.received_signal is None:
            self.received_signal = name


def _install_signal_handlers(sigstate: _SignalState) -> None:
    """Install handlers for SIGINT (Ctrl-C) and SIGTERM."""
    def handler(signum: int, _frame) -> None:
        sigstate.trip(signum)
    signal.signal(signal.SIGINT,  handler)
    signal.signal(signal.SIGTERM, handler)


# =============================================================================
# Slice 2: Subprocess execution with stream-to-log
# =============================================================================

def _terminate_child(proc: subprocess.Popen, log_handle, hard_kill_after_sec: float = 10.0) -> None:
    """
    Try graceful SIGTERM; escalate to SIGKILL if the child won't exit.
    Logs each step to the run log so the failure path is traceable.

    TB Fix 1: signals the child's PROCESS GROUP, not just the direct child.
    Because we Popen with start_new_session=True, the child is a process
    group leader; without os.killpg the child's own subprocesses (eventual
    SSH'd workers spawned via PWC) would survive. For D1 crash forensics
    the operator must be able to tear down the entire tree.
    """
    if proc.poll() is not None:
        return  # already exited

    # Resolve process group ID up front; if the child has already gone away
    # between poll() and getpgid(), pgid will be None and we fall back to
    # signaling the direct child.
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        pgid = None

    try:
        log_handle.write(f"\n[launcher] terminating child PID {proc.pid} via SIGTERM\n")
        log_handle.flush()
        if pgid is not None:
            log_handle.write(f"[launcher] sending SIGTERM to process group {pgid}\n")
            log_handle.flush()
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                # Group vanished between getpgid and killpg — race, harmless
                log_handle.write(f"[launcher] process group {pgid} already gone\n")
        else:
            proc.terminate()

        try:
            proc.wait(timeout=hard_kill_after_sec)
            log_handle.write(f"[launcher] child PID {proc.pid} exited after SIGTERM "
                             f"(rc={proc.returncode})\n")
        except subprocess.TimeoutExpired:
            log_handle.write(f"[launcher] child PID {proc.pid} did not exit within "
                             f"{hard_kill_after_sec}s — sending SIGKILL\n")
            log_handle.flush()
            if pgid is not None:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # already gone
            else:
                proc.kill()
            try:
                proc.wait(timeout=5.0)
                log_handle.write(f"[launcher] child PID {proc.pid} killed (rc={proc.returncode})\n")
            except subprocess.TimeoutExpired:
                log_handle.write(f"[launcher] child PID {proc.pid} could not be killed — abandoned\n")
    except Exception as exc:
        log_handle.write(f"[launcher] terminate raised: {exc!r}\n")
    finally:
        log_handle.flush()


def run_optimizer_subprocess(
    cfg: LauncherConfig,
    optimizer_argv: List[str],
    optimizer_env: Dict[str, str],
    paths: RunPaths,
    state: RunState,
    sigstate: _SignalState,
    poll_interval_sec: float = 0.5,
) -> None:
    """
    Launch window_optimizer.py and stream its stdout+stderr to paths.run_log
    line-by-line. Block until the child exits (clean or via signal).

    Updates state in-place:
      - state.child_pid, child_started_at_iso, child_exit_code, child_ended_at_iso
      - state.interrupted + state.interrupt_signal if SIGINT/SIGTERM received

    On signal received: SIGTERM the child, escalate to SIGKILL after grace period,
    record interruption, return (do not raise — caller's finally still runs).
    """
    # Open run log for streaming. Line-buffered so partial output is visible
    # even if the child is killed mid-line.
    state.child_started_at_iso = datetime.now(timezone.utc).astimezone().isoformat()

    with open(paths.run_log, "w", buffering=1) as run_log:
        run_log.write(f"# S174 Python launcher — run log\n")
        # TB Fix 3: timestamped run_id, not bare prefix
        run_log.write(f"# RUN_ID: {state.run_id}\n")
        run_log.write(f"# child_started_at: {state.child_started_at_iso}\n")
        run_log.write(f"# argv: {' '.join(shlex.quote(a) for a in optimizer_argv)}\n")
        run_log.write(f"# cwd: {cfg.repo_root}\n")
        run_log.write("# " + "=" * 70 + "\n\n")
        run_log.flush()

        try:
            proc = subprocess.Popen(
                optimizer_argv,
                cwd=str(cfg.repo_root),
                env=optimizer_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,   # merge stderr into stdout for chronological log
                text=True,
                bufsize=1,                   # line-buffered child stdout
                # Place child in its own process group so we can signal the
                # local process tree (window_optimizer + any local descendants
                # it spawns) cleanly via os.killpg.
                #
                # Scope note: this signals LOCAL processes only. Remote PWC
                # TCP workers spawned over SSH (with their own nohup wrappers
                # on rigs) are NOT killed by killpg on this Zeus-side group.
                # Their cleanup is handled separately by:
                #   - PWC's own shutdown path is NOT guaranteed to reach
                #     remote workers in a faulted/terminated D1
                #   - PWC startup-cleanup on next run (kills stale workers)
                #   - operator's monitor_all.sh / manual SSH if needed
                # Slice 3's bundle workflow captures remote state for forensics.
                start_new_session=True,
            )
        except (OSError, FileNotFoundError) as exc:
            run_log.write(f"\n[launcher] failed to spawn child: {exc!r}\n")
            state.child_exit_code = -1
            state.child_ended_at_iso = datetime.now(timezone.utc).astimezone().isoformat()
            state.launcher_exception_text = f"Popen failed: {exc!r}"
            state.classification = SENTINEL_LAUNCHER_EXCEPTION
            return

        state.child_pid = proc.pid

        # TB Fix 4: selectors-based polling with explicit timeout. The previous
        # implementation called proc.stdout.readline() in a loop and relied on
        # SIGINT/SIGTERM interrupting the blocking read — which is system-
        # behavior dependent and not safe enough for a crash-forensics
        # launcher. Selector + timeout guarantees the loop returns to check
        # sigstate.received_signal and proc.poll() at most poll_interval_sec
        # after either condition becomes true.
        sel = selectors.DefaultSelector()
        try:
            assert proc.stdout is not None
            os.set_blocking(proc.stdout.fileno(), False)
            sel.register(proc.stdout, selectors.EVENT_READ)

            partial_buffer = ""

            while True:
                # Did a signal arrive?
                if sigstate.received_signal is not None and not state.interrupted:
                    state.interrupted = True
                    state.interrupt_signal = sigstate.received_signal
                    run_log.write(
                        f"\n[launcher] received {sigstate.received_signal} — "
                        f"terminating child\n"
                    )
                    run_log.flush()
                    _terminate_child(proc, run_log)
                    break

                # Has the child exited on its own?
                if proc.poll() is not None:
                    break

                # Wait briefly for stdout activity (or timeout to re-check
                # signals/exit). select() does NOT block forever — bounded
                # by poll_interval_sec.
                events = sel.select(timeout=poll_interval_sec)
                if not events:
                    # No data ready; loop continues to re-check signals/exit.
                    continue

                for key, _mask in events:
                    try:
                        data = key.fileobj.read(65536)
                    except (BlockingIOError, InterruptedError):
                        continue
                    except Exception as exc:
                        run_log.write(f"\n[launcher] read error: {exc!r}\n")
                        data = ""

                    if data == "" or data is None:
                        # Child closed stdout (likely about to exit). Drop
                        # out of inner loop; outer loop will detect via poll.
                        continue

                    # Combine with any leftover partial line for clean writes
                    partial_buffer += data
                    if "\n" in partial_buffer:
                        complete, _, partial_buffer = partial_buffer.rpartition("\n")
                        run_log.write(complete + "\n")
                    # Else: keep accumulating, will flush on next iteration

            # Flush any final partial line that didn't end in newline
            if partial_buffer:
                run_log.write(partial_buffer + "\n")
        except Exception as exc:
            run_log.write(f"\n[launcher] streaming loop raised: {exc!r}\n")
            run_log.write(traceback.format_exc())
            state.launcher_exception_text = f"streaming: {exc!r}"
        finally:
            try:
                sel.close()
            except Exception:
                pass

        # Drain any final output after child exit (rare race window).
        # Stdout is now non-blocking; read until empty or EOF.
        try:
            assert proc.stdout is not None
            while True:
                try:
                    chunk = proc.stdout.read(65536)
                except (BlockingIOError, InterruptedError):
                    break
                except Exception:
                    break
                if not chunk:
                    break
                run_log.write(chunk)
        except Exception:
            pass

        # Make sure child is fully reaped
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            run_log.write("[launcher] child did not reap within 10s after stream end — killing\n")
            _terminate_child(proc, run_log)

        state.child_exit_code = proc.returncode
        state.child_ended_at_iso = datetime.now(timezone.utc).astimezone().isoformat()
        run_log.write(f"\n# child_exit_code: {state.child_exit_code}\n")
        run_log.write(f"# child_ended_at: {state.child_ended_at_iso}\n")


# =============================================================================
# Slice 2: Sentinel classification (placeholder — Slice 4 replaces this)
# =============================================================================

def assign_sentinel_classification(state: RunState, is_dry_run: bool = False) -> None:
    """
    Set state.classification to a sentinel value based on what happened.
    These are PLACEHOLDERS — Slice 4 will replace with real classification
    derived from log content.

    Precedence (highest first):
      0. DRY_RUN — explicit no-op flag, no child was supposed to start
      1. INTERRUPTED — a signal arrived (operator killed run)
      2. PRECHECK_FAILED — preflight rejected before child started
      3. LAUNCHER_EXCEPTION — launcher itself failed (Popen failure,
         streaming loop crash, etc.) — supersedes any child exit code
         because we can't trust it represents the child's real state
      4. CHILD_EXIT_0 — child exited cleanly
      5. CHILD_EXIT_NONZERO — child exited with error

    TB Slice 2 follow-up fix: launcher_exception_text now ALWAYS leads to
    SENTINEL_LAUNCHER_EXCEPTION regardless of whether child_exit_code was
    set. Previously a Popen failure (which set child_exit_code=-1) was
    misclassified as CHILD_EXIT_NONZERO. Slice 4 may refine mixed cases
    if needed.
    """
    if is_dry_run and not state.interrupted and not state.launcher_exception_text:
        state.classification = SENTINEL_DRY_RUN
        return
    if state.interrupted:
        state.classification = SENTINEL_INTERRUPTED
        return
    if state.preflight_missing:
        state.classification = SENTINEL_PRECHECK_FAILED
        return
    if state.launcher_exception_text:
        # Any launcher-side exception means we can't trust child_exit_code
        # as authoritative — classify as LAUNCHER_EXCEPTION.
        state.classification = SENTINEL_LAUNCHER_EXCEPTION
        return
    if state.child_exit_code is None:
        # Child never ran AND no exception recorded — defensive default
        state.classification = SENTINEL_LAUNCHER_EXCEPTION
        return
    if state.child_exit_code == 0:
        state.classification = SENTINEL_CHILD_EXIT_0
        return
    state.classification = SENTINEL_CHILD_EXIT_NONZERO


# =============================================================================
# Slice 2: Summary writers — guaranteed to run via finally block
# =============================================================================

def write_summary(
    cfg: LauncherConfig,
    paths: RunPaths,
    provenance: Provenance,
    state: RunState,
    findings: Optional["EvidenceFindings"] = None,
) -> None:
    """
    Always-writes-something summary emitter. Called from main()'s finally
    block so even crashes/kills produce a summary file.

    Writes both summary.json (machine-readable) and summary.txt (human-readable).
    Also writes classification.txt with the final classification value.

    Slice 4: optional `findings` (EvidenceFindings) is embedded in JSON +
    TXT so operators and downstream tooling can see what evidence drove
    the classification.
    """
    # JSON: full structured dump
    summary_doc: Dict[str, Any] = {
        # TB Fix 3: timestamped run_id, not bare prefix
        "run_id": state.run_id,
        "purpose": cfg.purpose,
        "config": {
            "pool": cfg.pool,
            "chunk_cap": cfg.chunk,
            "max_seeds": cfg.max_seeds,
            "min_workers": cfg.min_workers,
            "trials": cfg.trials,
            "lottery_file": cfg.lottery_file,
            "prng_type": cfg.prng_type,
            "strategy": cfg.strategy,
            "seed_start": cfg.seed_start,
            "forced_config": (
                cfg.forced_config.label() if cfg.forced_config else None
            ),
            "expected_chunks_total": cfg.expected_chunks_total(),
            "expected_chunks_per_amd_worker": cfg.expected_chunks_per_amd_worker(),
            "observation_window_minutes": cfg.observation_window_minutes,
            "observation_snapshot_interval_sec": cfg.observation_snapshot_interval_sec,
        },
        "provenance": provenance.to_dict(),
        "state": state.to_dict(),
        "paths": {
            "run_log": str(paths.run_log),
            "launcher_log": str(paths.launcher_log),
            "summary_json": str(paths.summary_json),
            "summary_txt": str(paths.summary_txt),
            "bundle_dir": str(paths.bundle_dir),
            "observation_dir": str(paths.observation_dir),
            "classification_file": str(paths.classification_file),
        },
        "slice_version": "slice_4_final_classification",
        "evidence_findings": findings.to_dict() if findings else None,
    }
    try:
        paths.summary_json.write_text(json.dumps(summary_doc, indent=2))
    except Exception as exc:
        # Last-resort: write to stderr so operator at least sees something
        sys.stderr.write(f"[launcher] CRITICAL: summary.json write failed: {exc!r}\n")

    # TXT: human-readable mirror of the most important fields
    txt_lines = [
        # TB Fix 3: timestamped run_id, not bare prefix
        f"=== {state.run_id} ===",
        f"purpose: {cfg.purpose}",
        "",
        f"=== Config ===",
        f"pool: {cfg.pool}",
        f"chunk_cap: {cfg.chunk}",
        f"max_seeds: {cfg.max_seeds}",
        f"min_workers: {cfg.min_workers}",
        f"trials: {cfg.trials}",
        f"forced_config: {cfg.forced_config.label() if cfg.forced_config else '(none — open Optuna)'}",
        f"expected_chunks_total: {cfg.expected_chunks_total()}",
        f"expected_chunks_per_amd_worker: {cfg.expected_chunks_per_amd_worker()}",
        "",
        f"=== Provenance ===",
        f"git_sha: {provenance.git_sha}",
        f"git_branch: {provenance.git_branch}",
        f"git_dirty: {provenance.git_dirty}",
        f"started_at: {state.started_at_iso}",
        f"ended_at: {state.ended_at_iso or '(launcher still running)'}",
        f"optimizer_python: {provenance.optimizer_python}",
        f"repo_root: {provenance.repo_root}",
        "",
        f"=== Run state ===",
        f"child_pid: {state.child_pid}",
        f"child_started_at: {state.child_started_at_iso or '(child never started)'}",
        f"child_ended_at: {state.child_ended_at_iso or '(child not yet ended)'}",
        f"child_exit_code: {state.child_exit_code if state.child_exit_code is not None else '(unknown)'}",
        f"interrupted: {state.interrupted}" + (
            f" (signal: {state.interrupt_signal})" if state.interrupt_signal else ""
        ),
        f"preflight_missing: {state.preflight_missing or '(none)'}",
        f"config_validation_errors: {state.config_validation_errors or '(none)'}",
        f"launcher_exception: {state.launcher_exception_text or '(none)'}",
        "",
        f"=== Observation window (Slice 3) ===",
        f"observation_started_at: {state.observation_started_at_iso or '(not run)'}",
        f"observation_ended_at: {state.observation_ended_at_iso or '(not run)'}",
        f"observation_snapshot_count: {state.observation_snapshot_count}",
        f"observation_skipped_reason: {state.observation_skipped_reason or '(not skipped)'}",
        "",
        f"=== Bundle assembly (Slice 3) ===",
        f"bundle_started_at: {state.bundle_started_at_iso or '(not run)'}",
        f"bundle_ended_at: {state.bundle_ended_at_iso or '(not run)'}",
        f"bundle_files_collected: {len(state.bundle_files_collected)}",
        f"bundle_files_missing: {len(state.bundle_files_missing)}",
        f"bundle_remote_unreachable: {state.bundle_remote_unreachable or '(none)'}",
        "",
        f"=== Evidence findings (Slice 4) ===",
        f"ready_gate_passed: {findings.ready_gate_passed if findings else '(no findings)'}"
            + (f" (count={findings.ready_gate_passed_count})"
               if findings and findings.ready_gate_passed_count is not None else ""),
        f"ready_gate_failed: {findings.ready_gate_failed if findings else '(no findings)'}",
        f"dispatch_confirmed: {findings.dispatch_confirmed if findings else '(no findings)'}",
        f"legacy_dispatch_seen: {findings.legacy_dispatch_seen if findings else '(no findings)'}",
        f"optimizer_completed_clean: {findings.optimizer_completed_clean if findings else '(no findings)'}",
        f"chunks_observed_in_log: {findings.chunks_observed if findings else '(no findings)'}",
        f"faults_in_run_log: {findings.faults_in_run_log if findings else '(no findings)'}",
        f"faults_in_netconsole: {findings.faults_in_netconsole if findings else '(no findings)'}",
        f"faults_in_observation: {findings.faults_in_observation if findings else '(no findings)'}",
        f"fault_seen_pre_child_exit: {findings.fault_seen_pre_child_exit if findings else '(no findings)'}",
        f"fault_seen_post_child_exit: {findings.fault_seen_post_child_exit if findings else '(no findings)'}",
        "",
        f"=== Classification (Slice 4 final) ===",
        f"classification: {state.classification}",
        "",
        f"=== Paths ===",
        f"run_log: {paths.run_log}",
        f"launcher_log: {paths.launcher_log}",
        f"summary_json: {paths.summary_json}",
        f"bundle_dir: {paths.bundle_dir}",
    ]
    try:
        paths.summary_txt.write_text("\n".join(txt_lines) + "\n")
    except Exception as exc:
        sys.stderr.write(f"[launcher] CRITICAL: summary.txt write failed: {exc!r}\n")

    # Classification file: one-liner so external tools can grep it cheaply
    try:
        paths.classification_file.write_text(state.classification + "\n")
    except Exception as exc:
        sys.stderr.write(f"[launcher] CRITICAL: classification file write failed: {exc!r}\n")


# =============================================================================
# Slice 2: Launcher log writer (separate from run_log which is the child's)
# =============================================================================

def _open_launcher_log(path: Path):
    """Returns a line-buffered append-mode file handle for the launcher's
    own log (separate from window_optimizer's run log)."""
    return open(path, "a", buffering=1)


# =============================================================================
# Slice 3: Per-rig snapshot collection
# =============================================================================
# Snapshots run via SSH against each rig. Failures are non-fatal — we record
# the failure in the snapshot itself so forensic analysis can see "rig X was
# unreachable at T+3min" as positive evidence. Each rig snapshot is written
# as one JSON file per rig per snapshot tick.
#
# TB scope: reachable, hostname, rocm-smi summary, ps grep for relevant
# processes, /tmp/prng_active_worker_gpu*.json contents, /tmp/prng_gpu_bus_map_gpu*.json
# contents.
# =============================================================================

# Commands to run on each remote rig per snapshot tick. Tuples of (key, cmd).
# Each command must be idempotent + read-only. Exit codes are captured.
RIG_SNAPSHOT_COMMANDS: List[tuple] = [
    ("hostname",
     "hostname"),
    ("uptime",
     "uptime"),
    ("rocm_smi",
     "rocm-smi --showid --showtemp --showuse --showmemuse 2>&1 | head -200"),
    ("ps_workers",
     "ps -eo pid,ppid,etime,stat,comm,args | "
     "grep -E 'pwc_worker_service|sieve_gpu_worker|window_optimizer|persistent_worker' | "
     "grep -v grep || true"),
    ("active_worker_jsons",
     "for f in /tmp/prng_active_worker_gpu*.json; do "
     "  if [ -f \"$f\" ]; then "
     "    echo \"=== $f ===\"; cat \"$f\"; echo; "
     "  fi; "
     "done"),
    ("gpu_bus_maps",
     "for f in /tmp/prng_gpu_bus_map_gpu*.json; do "
     "  if [ -f \"$f\" ]; then "
     "    echo \"=== $f ===\"; cat \"$f\"; echo; "
     "  fi; "
     "done"),
    ("dmesg_tail",
     "dmesg -T 2>/dev/null | tail -80 || journalctl -k --since '10 minutes ago' 2>/dev/null | tail -80 || echo 'NO_KERNEL_LOG_ACCESS'"),
]

SSH_OPTIONS = [
    "-o", "ConnectTimeout=8",
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ServerAliveInterval=5",
    "-o", "ServerAliveCountMax=2",
]


def _ssh_run(rig: str, cmd: str, timeout: float = 30.0) -> Dict[str, Any]:
    """
    Run one shell command on a remote rig via SSH. Returns a dict with:
      ok (bool), exit_code (int), stdout (str), stderr (str), error (str|None).

    Never raises — captures all failure modes into the dict.
    """
    full_argv = ["ssh"] + SSH_OPTIONS + [rig, cmd]
    try:
        result = subprocess.run(
            full_argv,
            capture_output=True, text=True, timeout=timeout,
        )
        return {
            "ok": result.returncode == 0,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False, "exit_code": -1, "stdout": "", "stderr": "",
            "error": f"timeout after {timeout}s",
        }
    except FileNotFoundError:
        return {
            "ok": False, "exit_code": -1, "stdout": "", "stderr": "",
            "error": "ssh binary not found",
        }
    except Exception as exc:
        return {
            "ok": False, "exit_code": -1, "stdout": "", "stderr": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def collect_rig_snapshot(rig: str, ts_iso: str, ticks: int) -> Dict[str, Any]:
    """
    Collect one snapshot from one rig. Returns a dict ready for JSON dump.
    Reachability test runs first; if it fails, command outputs are skipped.
    """
    snap: Dict[str, Any] = {
        "rig": rig,
        "tick": ticks,
        "ts": ts_iso,
        "reachable": False,
        "commands": {},
    }
    # Cheap reachability probe — single-shot true/false
    probe = _ssh_run(rig, "true", timeout=10.0)
    snap["reachable"] = probe["ok"]
    if not probe["ok"]:
        snap["unreachable_reason"] = probe.get("error") or f"exit={probe['exit_code']}"
        return snap

    for key, cmd in RIG_SNAPSHOT_COMMANDS:
        snap["commands"][key] = _ssh_run(rig, cmd, timeout=30.0)
    return snap


def collect_observation_snapshot(
    rigs: List[str], tick: int, observation_dir: Path,
    interval_sec: int,
) -> Dict[str, int]:
    """
    Collect a per-rig snapshot at the given tick (T+0, T+interval, ...).
    Writes one JSON per tick: observation_window/T+NNNs.json (filename uses
    elapsed SECONDS, not minutes, since interval is configurable — TB Fix 4).
    The payload also includes a "label" field so test runs with non-60s
    intervals are unambiguous.
    Returns {"reachable": N, "unreachable": M} for caller logging.
    """
    ts_iso = datetime.now(timezone.utc).astimezone().isoformat()
    rigs_data: List[Dict[str, Any]] = []
    reach = 0
    unreach = 0
    for rig in rigs:
        snap = collect_rig_snapshot(rig, ts_iso, tick)
        rigs_data.append(snap)
        if snap["reachable"]:
            reach += 1
        else:
            unreach += 1

    elapsed_s = tick * interval_sec
    label = f"T+{elapsed_s:04d}s"
    out_path = observation_dir / f"{label}.json"
    payload = {
        "tick": tick,
        "elapsed_seconds": elapsed_s,
        "interval_seconds": interval_sec,
        "label": label,
        "ts": ts_iso,
        "rigs": rigs_data,
        "summary": {
            "reachable": reach,
            "unreachable": unreach,
            "total": len(rigs),
        },
    }
    try:
        out_path.write_text(json.dumps(payload, indent=2))
    except Exception as exc:
        sys.stderr.write(f"[launcher] snapshot write failed: {exc!r}\n")
    return {"reachable": reach, "unreachable": unreach}


def run_observation_window(
    cfg: LauncherConfig,
    paths: RunPaths,
    state: RunState,
    sigstate: _SignalState,
    launcher_log,
) -> None:
    """
    Run the post-completion observation window.
      - Snapshots at T+0s, T+interval, T+2*interval, ..., T+window_seconds
        inclusive (filename label uses elapsed seconds, e.g. T+0060s.json)
      - Each tick collects per-rig data via SSH (collect_observation_snapshot)
      - Skipped entirely if observation_window_minutes <= 0
      - Aborts early if SIGTERM/SIGINT received (records partial completion)

    State updates:
      observation_started_at_iso, observation_ended_at_iso,
      observation_snapshot_count, observation_skipped_reason
    """
    if cfg.observation_window_minutes <= 0:
        state.observation_skipped_reason = "observation_window_minutes <= 0"
        launcher_log.write(
            f"# observation: skipped ({state.observation_skipped_reason})\n"
        )
        return

    state.observation_started_at_iso = datetime.now(timezone.utc).astimezone().isoformat()
    launcher_log.write(
        f"# observation_window: started at {state.observation_started_at_iso} "
        f"({cfg.observation_window_minutes} min, every {cfg.observation_snapshot_interval_sec}s)\n"
    )
    launcher_log.flush()

    interval_sec = cfg.observation_snapshot_interval_sec
    window_seconds = cfg.observation_window_minutes * 60
    # Snapshot at T+0, T+interval, T+2*interval, ..., T+window_seconds inclusive.
    # ticks_total counts how many snapshots we want. TB Fix 4 corollary:
    # works correctly with any interval_sec, not just 60s.
    ticks_total = (window_seconds // interval_sec) + 1 if interval_sec > 0 else 1

    for tick in range(ticks_total):
        # If interrupted during observation, record partial progress and exit.
        # Snapshot already collected for this tick is written first, then bail.
        if sigstate.received_signal is not None and not state.interrupted:
            state.interrupted = True
            state.interrupt_signal = sigstate.received_signal
            launcher_log.write(
                f"# observation: interrupted by {sigstate.received_signal} at tick {tick}\n"
            )
            break

        try:
            elapsed_s_label = f"T+{tick * interval_sec:04d}s"
            counts = collect_observation_snapshot(
                RIG_HOSTS, tick, paths.observation_dir, interval_sec,
            )
            state.observation_snapshot_count += 1
            launcher_log.write(
                f"# observation tick {elapsed_s_label} (tick={tick}): "
                f"reachable={counts['reachable']}/{counts['reachable']+counts['unreachable']}\n"
            )
            launcher_log.flush()
        except Exception as exc:
            elapsed_s_label = f"T+{tick * interval_sec:04d}s"
            launcher_log.write(
                f"# observation tick {elapsed_s_label} (tick={tick}): "
                f"snapshot raised {exc!r}\n"
            )
            # Continue to next tick — don't let one bad snapshot kill observation

        # Sleep until next tick — but split into small chunks so SIGTERM is responsive.
        # Don't sleep after the final tick.
        if tick < ticks_total - 1:
            sleep_remaining = float(interval_sec)
            while sleep_remaining > 0:
                if sigstate.received_signal is not None:
                    break
                step = min(0.5, sleep_remaining)
                time.sleep(step)
                sleep_remaining -= step

    state.observation_ended_at_iso = datetime.now(timezone.utc).astimezone().isoformat()
    launcher_log.write(
        f"# observation_window: ended at {state.observation_ended_at_iso}, "
        f"snapshots={state.observation_snapshot_count}/{ticks_total}\n"
    )
    launcher_log.flush()


# =============================================================================
# Slice 3: Bundle assembly
# =============================================================================
# Bundle is a directory containing forensic artifacts copied/captured for one
# run. Lives at logs/<RUN_ID>_bundle/. Layout:
#
#   logs/<RUN_ID>_bundle/
#     observation_window/             (already populated by run_observation_window)
#       T+00min.json
#       T+01min.json
#       ...
#     local/                          (Zeus-side artifacts)
#       run_log copy
#       launcher_log copy
#       netconsole_all_rigs.log copy
#       s173_job_assignment_ledger.jsonl copy
#       pwc_startup_diag*.jsonl copies
#       optimal_window_config.json copy
#     remote/                         (one subdir per rig, populated via SSH)
#       <rig>/
#         pwc_tcp_worker_*.log copies
#         active_worker JSON copies
#         gpu_bus_map JSON copies
#         rocm-smi snapshot
#         ps snapshot
#         dmesg/journalctl tail
#         reachable.txt              (single line: yes|no + reason)
#     summary.json (already at logs/<RUN_ID>_summary.json)
#     summary.txt  (already at logs/<RUN_ID>_summary.txt)
#     classification.txt (already at bundle_dir/classification.txt)
#
# All copy operations are wrapped in try/except — missing files become entries
# in state.bundle_files_missing. Unreachable rigs are listed in
# state.bundle_remote_unreachable. Slice 4 will use these to refine
# classification (e.g. INVALID_MISSING_EVIDENCE if too many key files absent).
# =============================================================================

# Local Zeus-side files to copy into bundle/local/, with relative paths from cfg.repo_root.
# Each entry is (description, repo_relative_path).
LOCAL_BUNDLE_FILES: List[tuple] = [
    ("netconsole",        "logs/netconsole_all_rigs.log"),
    ("pwc_ledger",        "logs/s173_job_assignment_ledger.jsonl"),
    ("pwc_startup_diag",  "logs/pwc_startup_diag_simple.jsonl"),
    ("optimal_config",    "optimal_window_config.json"),
    ("optimization_results", "window_optimization_results.json"),
    ("survivors_npz",     "bidirectional_survivors_binary.npz"),
]

# TB Slice 4 Fix 2: First-fault manifests and crash daemon logs may live
# at absolute paths (often /tmp or ~/crash_dumps), not under the repo root.
# These globs are expanded at bundle time; matched files are copied into
# bundle/local/forensic/. Missing matches are recorded but not fatal.
#
# Note: if the crash_forensic_daemon.py runs on SER8 (per session memory)
# rather than Zeus, these Zeus-side paths will produce no matches. Operator
# should pull SER8 crash_dumps separately. We capture what's reachable from
# Zeus and record the rest as missing for visibility.
LOCAL_ABSOLUTE_BUNDLE_GLOBS: List[tuple] = [
    ("first_fault_manifests_home_zeus",
     str(Path.home() / "crash_dumps")),                  # daemon's standard output dir
    ("crash_daemon_logs_zeus",
     "/tmp/crash_daemon_*.log"),                          # daemon's run logs
    ("first_fault_manifests_repo",
     "logs/first_fault*.txt"),                            # in-repo manifests if any
    ("first_fault_raw_repo",
     "logs/first_fault*.json"),                           # in-repo raw manifests
]

# Remote files/commands to capture per rig. Tuples of (subpath_in_rig_dir, mode, source).
# mode: "scp"  → fetch a file via scp
#       "cmd"  → run a command via ssh and save stdout
REMOTE_BUNDLE_ITEMS: List[tuple] = [
    # Worker logs — capture all of them. Use shell glob via cmd mode + tar to preserve set.
    ("pwc_tcp_worker_logs.tar.gz", "tar",
     "/tmp/pwc_tcp_worker_*.log"),
    ("active_worker_jsons.tar.gz", "tar",
     "/tmp/prng_active_worker_gpu*.json"),
    ("gpu_bus_maps.tar.gz",        "tar",
     "/tmp/prng_gpu_bus_map_gpu*.json"),
    # Snapshots taken at bundle-time (final state)
    ("final_rocm_smi.txt",   "cmd",
     "rocm-smi --showid --showtemp --showuse --showmemuse 2>&1 | head -300"),
    ("final_ps.txt",         "cmd",
     "ps -eo pid,ppid,etime,stat,comm,args"),
    ("final_dmesg_tail.txt", "cmd",
     "dmesg -T 2>/dev/null | tail -200 || journalctl -k --since '20 minutes ago' 2>/dev/null | tail -200 || echo 'NO_KERNEL_LOG_ACCESS'"),
    ("final_reachable.txt",  "cmd",
     "echo OK; hostname; date -Iseconds"),
]


def _copy_local_file_into_bundle(
    src_relative: str, repo_root: Path, dest_dir: Path,
    description: str, state: RunState, launcher_log,
) -> None:
    """Copy a local file into bundle/local/. Records missing files in state."""
    src = repo_root / src_relative
    if not src.exists():
        state.bundle_files_missing.append(f"local: {description} ({src_relative})")
        launcher_log.write(f"# bundle local: MISSING {description} ({src_relative})\n")
        return
    dest = dest_dir / src.name
    try:
        shutil.copy2(src, dest)
        state.bundle_files_collected.append(f"local: {description} → {dest.name}")
        launcher_log.write(f"# bundle local: copied {description} ({src_relative}, "
                           f"{dest.stat().st_size} bytes)\n")
    except Exception as exc:
        state.bundle_files_missing.append(
            f"local: {description} ({src_relative}) — copy failed: {exc!r}"
        )
        launcher_log.write(
            f"# bundle local: copy FAILED {description} ({src_relative}): {exc!r}\n"
        )


def _copy_absolute_glob_into_bundle(
    description: str, glob_or_dir: str, dest_dir: Path,
    state: RunState, launcher_log,
) -> None:
    """
    TB Slice 4 Fix 2: copy files matching an absolute path or glob into
    bundle/local/forensic/<description>/. Used for first-fault manifests
    and crash daemon logs whose paths aren't relative to repo_root.

    glob_or_dir may be:
      - a directory path: contents copied recursively
      - a glob pattern: matching files copied flat
      - a single file path: copied
    """
    forensic_dir = dest_dir / "forensic" / description
    src_path = Path(glob_or_dir)

    matched: List[Path] = []
    try:
        if src_path.is_dir():
            # Directory: recursive copy
            try:
                forensic_dir.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src_path, forensic_dir, dirs_exist_ok=True)
                # Count copied items for state
                copied_count = sum(1 for _ in forensic_dir.rglob("*") if _.is_file())
                state.bundle_files_collected.append(
                    f"local-abs: {description} (dir, {copied_count} files)"
                )
                launcher_log.write(
                    f"# bundle local-abs: copied dir {description} "
                    f"({src_path}, {copied_count} files)\n"
                )
                return
            except Exception as exc:
                state.bundle_files_missing.append(
                    f"local-abs: {description} ({glob_or_dir}) — dir copy failed: {exc!r}"
                )
                launcher_log.write(
                    f"# bundle local-abs: dir copy FAILED {description}: {exc!r}\n"
                )
                return
        elif src_path.exists():
            # Single file
            matched = [src_path]
        else:
            # Treat as glob — split into parent + pattern
            parent = src_path.parent
            pattern = src_path.name
            if parent.exists():
                matched = list(parent.glob(pattern))
    except Exception as exc:
        state.bundle_files_missing.append(
            f"local-abs: {description} ({glob_or_dir}) — match failed: {exc!r}"
        )
        launcher_log.write(
            f"# bundle local-abs: match FAILED {description}: {exc!r}\n"
        )
        return

    if not matched:
        state.bundle_files_missing.append(
            f"local-abs: {description} ({glob_or_dir}) — no matches"
        )
        launcher_log.write(
            f"# bundle local-abs: NO MATCH {description} ({glob_or_dir})\n"
        )
        return

    try:
        forensic_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        launcher_log.write(f"# bundle local-abs: mkdir failed for {description}: {exc!r}\n")
        return

    copied = 0
    for src in matched:
        try:
            shutil.copy2(src, forensic_dir / src.name)
            copied += 1
        except Exception as exc:
            state.bundle_files_missing.append(
                f"local-abs: {description} {src.name} — copy failed: {exc!r}"
            )
    if copied > 0:
        state.bundle_files_collected.append(
            f"local-abs: {description} ({copied} files)"
        )
        launcher_log.write(
            f"# bundle local-abs: copied {description} ({copied} files from {glob_or_dir})\n"
        )


def _scp_remote(rig: str, remote_path: str, local_dest: Path, timeout: float = 30.0) -> Dict[str, Any]:
    """Single-file scp from a rig. Returns same dict shape as _ssh_run."""
    cmd = ["scp"] + SSH_OPTIONS + [f"{rig}:{remote_path}", str(local_dest)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {
            "ok": result.returncode == 0,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "exit_code": -1, "stdout": "", "stderr": "",
                "error": f"timeout after {timeout}s"}
    except Exception as exc:
        return {"ok": False, "exit_code": -1, "stdout": "", "stderr": "",
                "error": f"{type(exc).__name__}: {exc}"}


def _capture_remote_tar(
    rig: str, glob_pattern: str, dest_tarball: Path, timeout: float = 60.0,
) -> Dict[str, Any]:
    """
    Capture a glob of remote files into a tarball. Done remote-side via
    `tar czf - <glob>` piped over SSH so we don't need to scp each file
    individually. If the glob matches nothing, the tarball will be empty
    but the call still returns ok.

    TB Slice 3 portability fix: explicitly invoke remote bash -lc and pass
    the glob as a positional argument. Previously used `set -o pipefail`
    and `shopt -s nullglob` which are bash-specific; if the remote shell
    is /bin/sh (often dash on Ubuntu), those features aren't portable.
    Forcing bash -lc + compgen -G handles glob expansion robustly across
    shells without quoting headaches.
    """
    remote_script = r'''set -u
pattern="$1"
FILES=$(compgen -G "$pattern" || true)
if [ -z "$FILES" ]; then
  echo NO_MATCH >&2
  exit 0
fi
tar czf - $FILES
'''
    # The "bash" before the glob_pattern is $0 (script name) for bash -lc
    cmd = (
        ["ssh"] + SSH_OPTIONS
        + [rig, "bash", "-lc", remote_script, "bash", glob_pattern]
    )
    try:
        with open(dest_tarball, "wb") as out:
            result = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, timeout=timeout)
        return {
            "ok": result.returncode == 0,
            "exit_code": result.returncode,
            "stderr": result.stderr.decode("utf-8", errors="replace") if result.stderr else "",
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "exit_code": -1, "stderr": "",
                "error": f"timeout after {timeout}s"}
    except Exception as exc:
        return {"ok": False, "exit_code": -1, "stderr": "",
                "error": f"{type(exc).__name__}: {exc}"}


def _capture_remote_cmd(
    rig: str, command: str, dest_path: Path, timeout: float = 30.0,
) -> Dict[str, Any]:
    """Run a remote command via SSH and save stdout to dest_path."""
    result = _ssh_run(rig, command, timeout=timeout)
    try:
        body = result.get("stdout", "") or ""
        if result.get("error"):
            body = f"# ERROR: {result['error']}\n# stderr: {result.get('stderr','')}\n" + body
        elif not result.get("ok"):
            body = (f"# command exited nonzero ({result.get('exit_code')}); "
                    f"stderr:\n{result.get('stderr','')}\n# stdout follows\n" + body)
        dest_path.write_text(body)
    except Exception as exc:
        try:
            dest_path.write_text(f"# write failed: {exc!r}\n")
        except Exception:
            pass
    return result


def _bundle_remote_for_rig(
    rig: str, rig_dir: Path, state: RunState, launcher_log,
) -> bool:
    """
    Capture all REMOTE_BUNDLE_ITEMS for one rig into rig_dir.
    Returns True if rig was reachable (regardless of per-item success),
    False if rig was unreachable and the rig_dir was annotated with that.
    """
    rig_dir.mkdir(parents=True, exist_ok=True)

    # Reachability gate
    probe = _ssh_run(rig, "true", timeout=10.0)
    if not probe["ok"]:
        try:
            (rig_dir / "UNREACHABLE.txt").write_text(
                f"rig: {rig}\nprobed_at: {datetime.now(timezone.utc).astimezone().isoformat()}\n"
                f"reason: {probe.get('error') or probe.get('exit_code')}\n"
                f"stderr: {probe.get('stderr','')}\n"
            )
        except Exception:
            pass
        state.bundle_remote_unreachable.append(rig)
        launcher_log.write(f"# bundle remote: {rig} UNREACHABLE\n")
        return False

    for subname, mode, source in REMOTE_BUNDLE_ITEMS:
        dest = rig_dir / subname
        try:
            if mode == "tar":
                res = _capture_remote_tar(rig, source, dest)
            elif mode == "cmd":
                res = _capture_remote_cmd(rig, source, dest)
            elif mode == "scp":
                res = _scp_remote(rig, source, dest)
            else:
                res = {"ok": False, "error": f"unknown mode {mode}"}

            if res.get("ok"):
                state.bundle_files_collected.append(f"remote {rig}: {subname}")
                launcher_log.write(f"# bundle remote {rig}: collected {subname}\n")
            else:
                err = res.get("error") or f"exit={res.get('exit_code')}"
                state.bundle_files_missing.append(f"remote {rig}: {subname} — {err}")
                launcher_log.write(f"# bundle remote {rig}: FAILED {subname} ({err})\n")
        except Exception as exc:
            state.bundle_files_missing.append(
                f"remote {rig}: {subname} — exception {exc!r}"
            )
            launcher_log.write(f"# bundle remote {rig}: EXCEPTION {subname}: {exc!r}\n")

    launcher_log.flush()
    return True


def assemble_bundle(
    cfg: LauncherConfig,
    paths: RunPaths,
    state: RunState,
    launcher_log,
) -> None:
    """
    Build the forensic bundle. Called from main()'s finally so it runs even
    on launcher exception (best-effort: collects whatever it can, records
    the gaps in state.bundle_files_missing).

    Layout:
      <bundle_dir>/local/<files>     (Zeus-side copies)
      <bundle_dir>/remote/<rig>/<files>  (per-rig captures via SSH/SCP)
      <bundle_dir>/observation_window/T+NNNNs.json  (already populated)
      <bundle_dir>/classification.txt  (already populated)

    state.bundle_started_at_iso / bundle_ended_at_iso bracket this work.
    """
    state.bundle_started_at_iso = datetime.now(timezone.utc).astimezone().isoformat()
    launcher_log.write(f"# bundle: assembly started at {state.bundle_started_at_iso}\n")
    launcher_log.flush()

    local_dir = paths.bundle_dir / "local"
    remote_dir = paths.bundle_dir / "remote"
    try:
        local_dir.mkdir(parents=True, exist_ok=True)
        remote_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        launcher_log.write(f"# bundle: mkdir failed: {exc!r}\n")
        # continue anyway — copies will fail visibly

    # 1. Local Zeus-side files
    # Always include the launcher's own logs even though they live in logs/
    # (not via LOCAL_BUNDLE_FILES list because their path uses RUN_ID).
    for desc, path_obj in [
        ("run_log", paths.run_log),
        ("launcher_log", paths.launcher_log),
    ]:
        if path_obj.exists():
            try:
                shutil.copy2(path_obj, local_dir / path_obj.name)
                state.bundle_files_collected.append(f"local: {desc} → {path_obj.name}")
                launcher_log.write(f"# bundle local: copied {desc}\n")
            except Exception as exc:
                state.bundle_files_missing.append(f"local: {desc} — copy failed: {exc!r}")
                launcher_log.write(f"# bundle local: copy FAILED {desc}: {exc!r}\n")
        else:
            state.bundle_files_missing.append(f"local: {desc} — file does not exist")
            launcher_log.write(f"# bundle local: MISSING {desc}\n")

    # The configured local files (netconsole, ledger, startup diag, etc.)
    for desc, rel_path in LOCAL_BUNDLE_FILES:
        _copy_local_file_into_bundle(
            rel_path, cfg.repo_root, local_dir, desc, state, launcher_log,
        )

    # TB Slice 4 Fix 2: absolute-path forensic captures (first-fault
    # manifests, crash daemon logs). These are NOT under repo_root.
    # Land in bundle/local/forensic/<description>/.
    for desc, glob_path in LOCAL_ABSOLUTE_BUNDLE_GLOBS:
        _copy_absolute_glob_into_bundle(
            desc, glob_path, local_dir, state, launcher_log,
        )

    # 2. Remote rigs
    for rig in RIG_HOSTS:
        rig_dir = remote_dir / rig
        try:
            _bundle_remote_for_rig(rig, rig_dir, state, launcher_log)
        except Exception as exc:
            state.bundle_files_missing.append(f"remote {rig}: assembly raised {exc!r}")
            launcher_log.write(f"# bundle remote {rig}: assembly EXCEPTION {exc!r}\n")

    state.bundle_ended_at_iso = datetime.now(timezone.utc).astimezone().isoformat()
    launcher_log.write(
        f"# bundle: assembly ended at {state.bundle_ended_at_iso}, "
        f"collected={len(state.bundle_files_collected)} "
        f"missing={len(state.bundle_files_missing)} "
        f"unreachable_rigs={len(state.bundle_remote_unreachable)}\n"
    )
    launcher_log.flush()


# =============================================================================
# Slice 4: Evidence inspection + semantic classification
# =============================================================================
# After bundle assembly, scan log content + observation snapshots to upgrade
# the sentinel classification to a real semantic value. This is read-only
# inspection of artifacts already on disk.
#
# TB precedence (1 = highest priority):
#   1. PRECHECK_FAILED / launcher exception / dry-run remain non-D1 sentinels.
#   2. READY_GATE_FAILED if run_log contains "READY GATE FAILED".
#   3. INVALID_MISSING_EVIDENCE if "READY GATE PASSED"/"dispatch confirmed"
#      markers absent for a run that started.
#   4. VALID_ACTIVE_FAULT if fault keywords occur before child exit / during
#      active chunk phase.
#   5. VALID_POST_COMPLETION_FAULT if child exited / optimizer completed,
#      then fault appears during observation window.
#   6. PYTHON_EXIT_NONZERO_NO_FAULT if child nonzero and no fault evidence.
#   7. INTERRUPTED_BUNDLED if interrupted and bundle exists.
#   8. VALID_CLEAN if child exit 0, gate markers present, observation
#      completed, and no fault evidence.
# =============================================================================


@dataclass
class EvidenceFindings:
    """Aggregated read-only inspection results from bundle artifacts."""
    # Gate evidence
    ready_gate_passed: bool = False
    ready_gate_passed_count: Optional[int] = None     # parsed N from "N/M ready"
    ready_gate_failed: bool = False
    dispatch_confirmed: bool = False
    legacy_dispatch_seen: bool = False                # old "N ready worker(s)" line

    # Optimizer completion
    optimizer_completed_clean: bool = False           # "Bayesian optimization complete"
    chunks_observed: int = 0                          # rough count from log

    # Fault evidence (from any source we inspect)
    faults_in_run_log: List[str] = field(default_factory=list)
    faults_in_netconsole: List[str] = field(default_factory=list)
    faults_in_observation: List[str] = field(default_factory=list)

    # Phase localization
    fault_seen_pre_child_exit: bool = False           # before child_ended_at_iso
    fault_seen_post_child_exit: bool = False          # in observation snapshots

    # Bundle integrity
    bundle_has_run_log: bool = False
    # TB Slice 4 Fix 3: bundle_has_classification removed — classification.txt
    # is written by write_summary() AFTER inspect_evidence() runs, so this
    # field would always be False here, which is misleading.

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _scan_text_for_keywords(text: str, keywords: List[str]) -> List[str]:
    """Returns list of keywords found in text (case-insensitive substring)."""
    if not text:
        return []
    lowered = text.lower()
    found = []
    seen_set = set()
    for kw in keywords:
        if kw.lower() in lowered and kw not in seen_set:
            found.append(kw)
            seen_set.add(kw)
    return found


def _read_text_safely(path: Path, max_bytes: int = 50_000_000) -> str:
    """Read a text file, capped to max_bytes. Returns '' if file missing/unreadable."""
    if not path.exists():
        return ""
    try:
        size = path.stat().st_size
        if size > max_bytes:
            # Read first half + last half so we catch headers AND tail faults
            half = max_bytes // 2
            with open(path, "rb") as f:
                head = f.read(half)
                f.seek(-half, 2)
                tail = f.read(half)
            return (head + b"\n...(truncated)...\n" + tail).decode("utf-8", errors="replace")
        return path.read_text(errors="replace")
    except Exception:
        return ""


def inspect_evidence(
    cfg: LauncherConfig, paths: RunPaths, state: RunState,
) -> EvidenceFindings:
    """
    Read bundle artifacts and populate EvidenceFindings.
    All file reads are tolerant of missing/unreadable inputs — missing
    artifacts become absent findings rather than exceptions.
    """
    f = EvidenceFindings()

    # === Run log inspection ===
    run_log_text = _read_text_safely(paths.run_log)
    if run_log_text:
        # Gate markers (Slice S174 hard-gate adds these)
        if "READY GATE PASSED" in run_log_text:
            f.ready_gate_passed = True
            # Parse "READY GATE PASSED: N/M ready" — best-effort
            for line in run_log_text.split("\n"):
                if "READY GATE PASSED" in line and "ready" in line:
                    try:
                        # e.g. "READY GATE PASSED: 26/26 ready (min_workers=24)"
                        after = line.split("READY GATE PASSED:")[1]
                        frac = after.split("ready")[0].strip()
                        n_str = frac.split("/")[0].strip()
                        f.ready_gate_passed_count = int(n_str)
                    except (IndexError, ValueError):
                        pass
                    break
        if "READY GATE FAILED" in run_log_text:
            f.ready_gate_failed = True
        if "dispatch confirmed:" in run_log_text:
            f.dispatch_confirmed = True
        # Legacy line that the patch removed — useful as a regression sentinel.
        if "ready worker(s) — dispatching" in run_log_text:
            f.legacy_dispatch_seen = True
        # Optimizer completion marker
        if "Bayesian optimization complete" in run_log_text:
            f.optimizer_completed_clean = True
        # Rough chunk count — count distinct "Chunk N:" log lines.
        # Best effort; doesn't have to be exact, just non-zero if work happened.
        for line in run_log_text.split("\n"):
            if "Chunk " in line and "seeds" in line and "→" in line:
                f.chunks_observed += 1
        # Fault keywords
        f.faults_in_run_log = _scan_text_for_keywords(run_log_text, FAULT_KEYWORDS)

    # === Netconsole log inspection ===
    netconsole_path = cfg.repo_root / "logs" / "netconsole_all_rigs.log"
    netconsole_text = _read_text_safely(netconsole_path)
    if netconsole_text:
        f.faults_in_netconsole = _scan_text_for_keywords(netconsole_text, FAULT_KEYWORDS)

    # === Observation snapshot inspection ===
    if paths.observation_dir.exists():
        for snap_path in sorted(paths.observation_dir.glob("T+*.json")):
            try:
                snap_data = json.loads(snap_path.read_text())
            except Exception:
                continue
            # Each snapshot has rigs[].commands.dmesg_tail.stdout (and similar)
            for rig in snap_data.get("rigs", []):
                if not rig.get("reachable", False):
                    continue
                cmds = rig.get("commands", {})
                for cmd_key in ("dmesg_tail", "rocm_smi", "ps_workers"):
                    cmd_result = cmds.get(cmd_key, {})
                    if isinstance(cmd_result, dict):
                        out = cmd_result.get("stdout", "") or ""
                        kws = _scan_text_for_keywords(out, FAULT_KEYWORDS)
                        for kw in kws:
                            entry = f"{snap_path.name}/{rig.get('rig')}/{cmd_key}: {kw}"
                            if entry not in f.faults_in_observation:
                                f.faults_in_observation.append(entry)

    # === Phase localization: when did faults appear? ===
    # TB Slice 4 Fix 1: netconsole-only faults are NOT defaulted to
    # post-completion. The kernel ring buffer commonly captures GPU faults
    # that never reach Python's run_log (especially during in-flight ROCm
    # divergence — D1's primary target). Defaulting netconsole-only to
    # post-completion would mislabel an active-kernel crash.
    #
    # Conservative rule:
    #   - run_log fault → pre-child-exit
    #   - observation snapshot fault → post-child-exit (observation only
    #     starts after child exit, so this is unambiguous)
    #   - netconsole fault NOT seen in observation → treat as ACTIVE
    #     unless observation snapshots independently corroborate post-phase
    if f.faults_in_run_log:
        f.fault_seen_pre_child_exit = True
    if f.faults_in_observation:
        f.fault_seen_post_child_exit = True
    if f.faults_in_netconsole:
        if f.faults_in_observation:
            # Observation independently picked up the same fault →
            # post-completion is corroborated, mark both phases.
            f.fault_seen_post_child_exit = True
        else:
            # Netconsole-only without observation corroboration → ACTIVE.
            # D1's main target is sudden in-kernel ROCm divergence; calling
            # this post-completion without timestamp evidence is unsafe.
            f.fault_seen_pre_child_exit = True

    # === Bundle integrity ===
    f.bundle_has_run_log = (paths.bundle_dir / "local").exists() and any(
        p.name.endswith(".log") and "launcher" not in p.name
        for p in (paths.bundle_dir / "local").glob("*")
    )

    return f


def derive_final_classification(
    state: RunState, findings: EvidenceFindings, is_dry_run: bool,
) -> str:
    """
    Apply the Slice 4 precedence ladder to produce a final semantic
    classification. Returns the classification string. Does NOT mutate state.

    Precedence (TB Slice 4 ruling):
      1. Non-D1 sentinels stay as-is (PRECHECK_FAILED, LAUNCHER_EXCEPTION,
         DRY_RUN, CHILD_EXIT_0/NONZERO_UNCLASSIFIED never apply when we
         have real evidence — replaced below).
      2. READY_GATE_FAILED trumps fault detection — a gate-rejected run
         never dispatched, so any "fault" claim is suspect.
      3. INVALID_MISSING_EVIDENCE when child started but gate markers absent
         (couldn't have safely run; bundle is incomplete forensically).
      4. VALID_ACTIVE_FAULT — fault during active chunk phase.
      5. VALID_POST_COMPLETION_FAULT — fault after optimizer completed.
      6. PYTHON_EXIT_NONZERO_NO_FAULT — child exited nonzero with no
         fault evidence (probably a Python-level error, not a hardware fault).
      7. INTERRUPTED_BUNDLED — operator killed run, bundle still exists.
      8. VALID_CLEAN — clean exit, gate passed, dispatch confirmed,
         observation completed, no fault evidence.
    """
    # === Non-D1 sentinels carried forward unchanged ===
    if is_dry_run:
        return SENTINEL_DRY_RUN
    if state.classification == SENTINEL_PRECHECK_FAILED:
        return SENTINEL_PRECHECK_FAILED
    if state.classification == SENTINEL_LAUNCHER_EXCEPTION:
        return SENTINEL_LAUNCHER_EXCEPTION

    # === READY GATE FAILED — highest D1 precedence ===
    if findings.ready_gate_failed:
        return CLASS_READY_GATE_FAILED

    # === INTERRUPTED ladder ===
    # If we were interrupted and we DO have gate markers + bundle, mark as
    # INTERRUPTED_BUNDLED (operator kill, evidence preserved). If gate
    # markers are missing AND we were interrupted very early, that's still
    # INTERRUPTED_BUNDLED — evidence quality is "what we got".
    if state.interrupted:
        return CLASS_INTERRUPTED_BUNDLED

    # === Did the run get far enough to dispatch? ===
    # If child started but we never saw "READY GATE PASSED" AND never saw
    # "dispatch confirmed", we don't have enough evidence to claim either
    # success or fault — classify as INVALID_MISSING_EVIDENCE.
    child_actually_ran = state.child_started_at_iso is not None
    has_gate_evidence = findings.ready_gate_passed or findings.dispatch_confirmed
    if child_actually_ran and not has_gate_evidence:
        return CLASS_INVALID_MISSING_EVIDENCE

    # === Fault evidence dominates classification of dispatched runs ===
    # ACTIVE fault: appears before child exited (i.e. in run_log during
    # the chunk phase). VALID_ACTIVE_FAULT means D1 successfully reproduced
    # an in-flight crash.
    if findings.fault_seen_pre_child_exit:
        return CLASS_VALID_ACTIVE_FAULT

    # POST-COMPLETION fault: appears after child exited, in observation
    # snapshots or netconsole tail. VALID_POST_COMPLETION_FAULT is the
    # 2026-05-03/05 signature — child ran clean, then GCVM_L2 cascade after.
    if findings.fault_seen_post_child_exit:
        return CLASS_VALID_POST_COMPLETION_FAULT

    # === No fault evidence ===
    if state.child_exit_code is not None and state.child_exit_code != 0:
        return CLASS_PYTHON_EXIT_NONZERO_NO_FAULT

    # === Clean run ===
    # Valid clean requires: child exit 0, gate markers present.
    if state.child_exit_code == 0 and has_gate_evidence:
        return CLASS_VALID_CLEAN

    # === Catch-all: child completed but missing some clean markers ===
    # E.g. child exit 0 but no "Bayesian optimization complete" line.
    # Conservative: INVALID_MISSING_EVIDENCE rather than VALID_CLEAN.
    return CLASS_INVALID_MISSING_EVIDENCE


def apply_final_classification(
    cfg: LauncherConfig, paths: RunPaths, state: RunState,
    is_dry_run: bool, launcher_log,
) -> EvidenceFindings:
    """
    Read evidence, derive final classification, update state in-place.
    Returns the EvidenceFindings for inclusion in summary.
    Wrapped in try/except — failure preserves the sentinel classification
    rather than overwriting it with garbage.
    """
    try:
        findings = inspect_evidence(cfg, paths, state)
    except Exception as exc:
        launcher_log.write(f"# classification: inspect_evidence raised {exc!r}\n"
                           f"{traceback.format_exc()}\n")
        return EvidenceFindings()  # empty — sentinel stays

    try:
        final = derive_final_classification(state, findings, is_dry_run)
        prev = state.classification
        state.classification = final
        launcher_log.write(
            f"# classification: sentinel={prev} → final={final}\n"
        )
    except Exception as exc:
        launcher_log.write(f"# classification: derive_final raised {exc!r}\n"
                           f"{traceback.format_exc()}\n")
    launcher_log.flush()
    return findings


def main(argv: List[str]) -> int:
    args = parse_args(argv[1:])

    # Build config — may raise SystemExit for missing custom args
    cfg = config_for_mode(args)

    # Numeric validation (Fix 4)
    config_errors = validate_config(cfg)
    if config_errors:
        print("CONFIG VALIDATION FAILED:", file=sys.stderr)
        for err in config_errors:
            print(f"  - {err}", file=sys.stderr)
        return 2

    # RUN_ID + paths now stable for the rest of the run
    run_id = generate_run_id(cfg.run_id_prefix)
    paths = RunPaths.for_run(run_id, cfg.repo_root)
    paths.ensure_dirs()

    # Build optimizer argv + env (Fixes 1, 2)
    optimizer_env = build_optimizer_env(cfg)
    optimizer_argv = build_optimizer_argv(cfg, paths.run_log)

    # Capture provenance (Fix 3) — done BEFORE child starts
    provenance = capture_provenance(run_id, cfg, argv, optimizer_argv, optimizer_env)

    # State + signal plumbing for the run
    state = RunState(
        run_id=run_id,
        started_at_iso=provenance.started_at_iso,
        config_validation_errors=list(config_errors),
    )
    sigstate = _SignalState()

    # Launcher's own log file (separate from window_optimizer's run_log).
    # Opened early so even early failures get a paper trail.
    launcher_log = _open_launcher_log(paths.launcher_log)
    launcher_log.write(
        f"\n# === S174 launcher start: {provenance.started_at_iso} ===\n"
    )
    launcher_log.write(f"# RUN_ID: {run_id}\n")
    launcher_log.write(f"# launcher_argv: {' '.join(shlex.quote(a) for a in argv)}\n")
    launcher_log.write(f"# optimizer_argv: {' '.join(shlex.quote(a) for a in optimizer_argv)}\n")
    launcher_log.flush()

    # Print plan summary (same info as Slice 1, briefer)
    print("=" * 76)
    print(f"S174 launcher — RUN_ID: {run_id}")
    print(f"mode: {args.mode}")
    print(f"git_sha: {provenance.git_sha} (dirty={provenance.git_dirty})")
    print(f"optimizer_python: {cfg.optimizer_python}")
    print(f"forced: {cfg.forced_config.label() if cfg.forced_config else '(none)'}")
    print(f"pool={cfg.pool} chunk={cfg.chunk} max_seeds={cfg.max_seeds} "
          f"min_workers={cfg.min_workers} trials={cfg.trials}")
    print(f"run_log: {paths.run_log}")
    print(f"summary: {paths.summary_txt}")
    print("=" * 76)

    # Install signal handlers AFTER paths are ready, so we can write a summary
    # if SIGINT arrives during preflight
    _install_signal_handlers(sigstate)

    # Dry-run flag captured into state so assign_sentinel_classification
    # respects it and doesn't overwrite with LAUNCHER_EXCEPTION
    state_is_dry_run = bool(args.dry_run)

    try:
        # Preflight (CLI flags + interpreter) — failure means do NOT start child
        if args.dry_run:
            print("[launcher] --dry-run set: skipping subprocess execution")
            launcher_log.write("# --dry-run set: skipping subprocess execution\n")
            # Classification handled in finally via assign_sentinel_classification
            # (it checks state_is_dry_run flag — see below)
        else:
            print("[launcher] preflight: checking CLI flags + interpreter")
            missing = preflight_check(cfg)
            state.preflight_missing = missing
            if missing:
                print(f"[launcher] PRECHECK_FAILED: {missing}", file=sys.stderr)
                launcher_log.write(f"# PRECHECK_FAILED: {missing}\n")
                # state.classification gets set in finally via assign_sentinel_classification
            else:
                print("[launcher] preflight passed — starting child")
                launcher_log.write("# preflight passed — starting child\n")
                launcher_log.flush()

                # Check if a signal already arrived during preflight
                if sigstate.received_signal is not None:
                    state.interrupted = True
                    state.interrupt_signal = sigstate.received_signal
                    print(f"[launcher] signal received during preflight: "
                          f"{sigstate.received_signal} — aborting", file=sys.stderr)
                    launcher_log.write(
                        f"# signal during preflight: {sigstate.received_signal} — aborting\n"
                    )
                else:
                    # Run the child. Updates state in-place.
                    run_optimizer_subprocess(
                        cfg, optimizer_argv, optimizer_env,
                        paths, state, sigstate,
                    )
                    launcher_log.write(
                        f"# child exit_code={state.child_exit_code} "
                        f"interrupted={state.interrupted}\n"
                    )

                    # Slice 3: post-completion observation window.
                    # Runs only if child actually ran (no preflight failure).
                    # Skipped if launcher itself crashed during streaming.
                    if state.launcher_exception_text:
                        state.observation_skipped_reason = (
                            "launcher exception during child execution"
                        )
                        launcher_log.write(
                            f"# observation: skipped ({state.observation_skipped_reason})\n"
                        )
                    else:
                        try:
                            run_observation_window(cfg, paths, state, sigstate, launcher_log)
                        except BaseException as exc:
                            # Don't let observation crash kill the launcher —
                            # bundle still needs to run via finally.
                            launcher_log.write(
                                f"# observation: raised {type(exc).__name__}: {exc}\n"
                                f"{traceback.format_exc()}\n"
                            )
                            state.observation_skipped_reason = (
                                f"observation raised: {type(exc).__name__}: {exc}"
                            )

    except SystemExit:
        # Let SystemExit propagate (don't classify it as a launcher exception)
        raise
    except BaseException as exc:
        # Catch ANYTHING so we always write a summary
        state.launcher_exception_text = (
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        )
        launcher_log.write(f"\n# LAUNCHER EXCEPTION: {state.launcher_exception_text}\n")
        # state.classification → SENTINEL_LAUNCHER_EXCEPTION via assign_sentinel
    finally:
        # === Always-runs cleanup ===
        # Slice 3: bundle assembly runs HERE so even crashes during observation
        # or streaming still produce whatever evidence has been collected so far.
        # Bundle is wrapped in its own try/except so a bundle failure can't
        # block summary write below.
        try:
            # Skip bundle if we never even constructed paths/state (defensive).
            # Skip bundle for dry-run since there's nothing to forensically
            # capture (no child ran, no remote artifacts to gather).
            if not state_is_dry_run:
                assemble_bundle(cfg, paths, state, launcher_log)
            else:
                launcher_log.write("# bundle: skipped (--dry-run)\n")
        except BaseException as exc:
            sys.stderr.write(
                f"[launcher] WARNING: bundle assembly raised: {exc!r}\n"
            )
            try:
                launcher_log.write(
                    f"# bundle: assembly EXCEPTION {type(exc).__name__}: {exc}\n"
                    f"{traceback.format_exc()}\n"
                )
            except Exception:
                pass

        state.ended_at_iso = datetime.now(timezone.utc).astimezone().isoformat()
        assign_sentinel_classification(state, is_dry_run=state_is_dry_run)

        # Slice 4: upgrade sentinel → semantic classification by reading
        # evidence from the bundle artifacts. This MUST run after bundle
        # assembly (so observation snapshots are present to inspect) and
        # before write_summary (so summary captures the final value, not
        # the sentinel). Wrapped in try/except — any failure preserves
        # the sentinel rather than corrupting state.classification.
        try:
            findings = apply_final_classification(
                cfg, paths, state, state_is_dry_run, launcher_log,
            )
        except BaseException as exc:
            sys.stderr.write(
                f"[launcher] WARNING: classification upgrade raised {exc!r}\n"
            )
            findings = None

        try:
            write_summary(cfg, paths, provenance, state, findings=findings)
        except BaseException as exc:
            sys.stderr.write(
                f"[launcher] CRITICAL: write_summary itself raised: {exc!r}\n"
            )

        try:
            launcher_log.write(f"# === launcher end: classification={state.classification} ===\n")
            launcher_log.close()
        except Exception:
            pass

        # Slice 3 finalization (TB Fix 1 + Fix 2): mirror the FINAL versions
        # of run_log, launcher_log, summary.json, summary.txt, and
        # classification.txt into bundle/local/. This guarantees the bundle
        # is self-contained AND captures the closed launcher_log with all
        # final lines flushed (the earlier copy in assemble_bundle was made
        # while launcher_log was still open).
        # Skipped for --dry-run since bundle was skipped entirely there.
        if not state_is_dry_run:
            try:
                final_dest = paths.bundle_dir / "local"
                final_dest.mkdir(parents=True, exist_ok=True)
                for src in [paths.run_log, paths.launcher_log,
                            paths.summary_json, paths.summary_txt,
                            paths.classification_file]:
                    try:
                        if src.exists():
                            shutil.copy2(src, final_dest / src.name)
                    except Exception as copy_exc:
                        sys.stderr.write(
                            f"[launcher] WARNING: final mirror of {src.name} "
                            f"failed: {copy_exc!r}\n"
                        )
            except Exception as exc:
                sys.stderr.write(
                    f"[launcher] WARNING: final summary mirror into bundle "
                    f"failed: {exc!r}\n"
                )

        print()
        print(f"[launcher] classification: {state.classification}")
        print(f"[launcher] summary: {paths.summary_txt}")
        print(f"[launcher] summary_json: {paths.summary_json}")
        print(f"[launcher] bundle:  {paths.bundle_dir}")

    # === Exit code mapping ===
    # Slice 2 sentinels:
    if state.classification == SENTINEL_PRECHECK_FAILED:
        return 3
    if state.classification == SENTINEL_LAUNCHER_EXCEPTION:
        return 4
    if state.classification == SENTINEL_DRY_RUN:
        return 0
    # Slice 4 classifications:
    if state.classification == CLASS_READY_GATE_FAILED:
        return 5    # distinguishable from PRECHECK_FAILED (3)
    if state.classification == CLASS_INTERRUPTED_BUNDLED:
        return 130  # conventional SIGINT
    if state.classification == CLASS_INVALID_MISSING_EVIDENCE:
        return 6    # something happened but we can't tell what
    if state.classification == CLASS_VALID_ACTIVE_FAULT:
        return 0    # exit 0 — we WANTED a fault for D1, this is success
    if state.classification == CLASS_VALID_POST_COMPLETION_FAULT:
        return 0    # same — fault during teardown is the 2026-05-03/05 signature
    if state.classification == CLASS_PYTHON_EXIT_NONZERO_NO_FAULT:
        return state.child_exit_code if state.child_exit_code else 1
    if state.classification == CLASS_VALID_CLEAN:
        return 0
    # Sentinel CHILD_EXIT_* fallbacks (shouldn't happen post-Slice-4 but
    # defensive in case classification couldn't run):
    if state.classification == SENTINEL_CHILD_EXIT_NONZERO:
        return state.child_exit_code if state.child_exit_code else 1
    if state.classification == SENTINEL_CHILD_EXIT_0:
        return 0
    # Unknown / interrupted sentinel fallback
    if state.classification == SENTINEL_INTERRUPTED:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
