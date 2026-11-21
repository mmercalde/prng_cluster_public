#!/usr/bin/env python3
"""
persistent_worker_coordinator.py
=================================
Standalone persistent-worker engine for Step 1 (Window Optimizer) sieve passes.

DROP-IN PARALLEL PATH — activated by --use-persistent-workers flag on window_optimizer.py.
Zero changes to coordinator.py, window_optimizer.py, or window_optimizer_integration_final.py.

WATCHER COMPATIBILITY: Fully transparent. WATCHER passes --use-persistent-workers via
agent_manifests/window_optimizer.json → window_optimizer.py → here. WATCHER never
knows or cares which path ran — output files are identical.

Architecture
------------
- One persistent sieve_gpu_worker.py process per AMD GPU (8 per rig × 3 rigs = 24)
- Zeus GPUs use execute_local_sieve_job() — no persistent worker needed (already fast)
- Workers stay alive across ALL 4 sieve passes (forward, reverse, forward_hybrid, reverse_hybrid)
- Workers receive jobs via stdin (JSON), return results via stdout (JSON)
- Hybrid jobs (steps 3+4) send strategy objects in payload — worker allocates correct buffers

ROCm Stability Envelope (from S130/S133 learnings)
----------------------------------------------------
- Spawn stagger: 4.0s per gpu_id to prevent simultaneous HIP init
- worker_pool_size cap: respects configured limit (default 8 per rig)
- HSA_ENABLE_SDMA=0, HSA_OVERRIDE_GFX_VERSION=10.3.0 always set
- GFXOFF disabled via kernel params (cluster-level, not our concern here)
- Per-rig fault isolation: single rig death quarantines that rig, run continues
- Semaphore throttle: ssh_pool semaphore gates all dispatch (max_per_node limit)
- Heartbeat check before each job dispatch — respawn dead worker

Hybrid Kernel Support (from S133-B root cause analysis)
--------------------------------------------------------
- Hybrid kernels require: skip_sequences_gpu (uint32, n×k), strategy_ids_gpu (uint32),
  strategy_max_misses (int32[]), strategy_tolerances (int32[]), n_strategies (int32)
- strategies loaded from hybrid_strategy.get_all_strategies() if not in payload
- sieve_gpu_worker.py receives strategies in job payload and allocates correct buffers
- Same logic as sieve_filter.py run_hybrid_sieve() — ported to IPC worker path

Job Payload Protocol (stdin → sieve_gpu_worker.py)
----------------------------------------------------
{
  "job_id": "sieve_000",
  "prng_type": "java_lcg",           # or java_lcg_reverse, java_lcg_hybrid, etc.
  "seed_start": 0,
  "seed_end": 192307,
  "residues": [...],                  # window draws
  "window_size": 8,
  "threshold": 0.25,
  "gpu_id": 0,
  "strategies": null,                 # null for constant skip; list of dicts for hybrid
  "phase2_threshold": 0.25            # [PWC-STAGE2] = threshold, every job (no 0.5 default)
}

Result Protocol (stdout → coordinator)
---------------------------------------
{"status": "ok", "survivors": [...], "match_rates": [...], "job_id": "..."}
{"status": "error", "message": "..."}

Usage (from window_optimizer_integration_final.py)
---------------------------------------------------
from persistent_worker_coordinator import PersistentWorkerCoordinator

pwc = PersistentWorkerCoordinator(config_file="distributed_config.json",
                                   worker_pool_size=8)
pwc.startup()   # spawns workers, staggers HIP init

# Drop-in replacement for coordinator.execute_distributed_analysis():
result = pwc.run_sieve_pass(
    prng_type="java_lcg",
    residues=residues,
    total_seeds=5_000_000,
    threshold=0.30,
    window_size=8,
    output_file="results/window_opt_forward_8_43_t1.json"
)

pwc.shutdown()  # clean worker teardown
"""

import json
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait as _cf_wait, FIRST_COMPLETED as _CF_FIRST_COMPLETED
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

# S172 Phase-5 D3.25-A: the versioned `step1_trial_populations_v2` producer
# contract. `build_trial_populations` assembles the return block AND validates
# it at producer egress, so the invariant
# `bidirectional_M == set(forward_map_M) & set(reverse_map_M)` is asserted on
# EVERY return path out of run_trial_persistent — the full return and the
# pruned early return alike.
from utils.canonical_records import build_trial_populations

# ─────────────────────────────────────────────────────────────────────────────
# [RESOLVED EXECUTION SET] consumer seam — see coordinator.py for the rationale.
# Lazy, defensive, and a no-op when no set is frozen.
# ─────────────────────────────────────────────────────────────────────────────
def _execution_set_nodes(node_dicts, *, consumer: str):
    try:
        from execution_set import filter_config_nodes
    except ImportError:
        return list(node_dicts)
    return filter_config_nodes(node_dicts, consumer=consumer)


# ─────────────────────────────────────────────────────────────────────────────
# ROCm stability constants (from S130/S133 learnings)
# ─────────────────────────────────────────────────────────────────────────────
ROCM_SPAWN_STAGGER_S   = 4.0   # seconds between worker spawns per gpu_id
TCP_SPAWN_STAGGER_S    = 1.0   # S161: TCP-only stagger (SSH is bootstrap only)
ROCM_READY_TIMEOUT_S   = 110.0 # S161: ROCm startup budget after last TCP worker launched
ROCM_ENV_VARS = [
    # [S155] Removed CUPY_CUDA_MEMORY_POOL_TYPE=none — caused 41GB VM mmap OOM.
    # Each worker mmaps full 8GB VRAM at device init with pool disabled.
    # 8 workers × 8GB = 64GB VA on a 7.7GB RAM machine → OOM killer.
    # Pool race condition now handled via set_limit() in sieve_gpu_worker.py.
    # Cache race (original S151 concern) addressed separately via per-worker
    # CUPY_CACHE_DIR (S152). These are independent issues. TB ruling: approved.
    #
    # [S155-v2] Propagate pool limit into remote worker env.
    # _spawn_worker() runs: env {rocm_env} python sieve_gpu_worker.py
    # ROCM_ENV_VARS is the only injection path — must include this var or
    # the worker falls back to its hardcoded default silently.
    # Operator can override on any rig: PRNG_CUPY_POOL_LIMIT_MB=512 bash sweep_run1.sh
    "PRNG_CUPY_POOL_LIMIT_MB=256",
    # [S155-v3] Defense-in-depth: CUPY_GPU_MEMORY_LIMIT is CuPy's own env var
    # (confirmed in memory.pyx _parse_limit_string — called at pool construction).
    # Read at SingleDeviceMemoryPool.__init__() before any worker code runs.
    # set_limit() inside Device context later supersedes this per-device value.
    # Layering: env var caps ALL devices at init; set_limit() reinforces for our device.
    # Format: plain integer bytes. 268435456 = 256MB.
    "CUPY_GPU_MEMORY_LIMIT=268435456",
    "HSA_OVERRIDE_GFX_VERSION=10.3.0",
    "HSA_ENABLE_SDMA=0",
    # [S165] Disable KFD debug trap — prevents NULL pointer dereference
    # in kfd_dbg_trap_activate during concurrent GPU reset + worker init.
    # Root cause: kq_acquire_packet_buffer crashes when called mid-reset.
    "HSA_ENABLE_DEBUG_TRAP=0",
    "ROCM_PATH=/opt/rocm",
    "HIP_PATH=/opt/rocm/hip",
    "LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/hip/lib:${LD_LIBRARY_PATH}",
    "PATH=/opt/rocm/bin:${PATH}",
    "CUPY_CACHE_DIR=${HOME}/.cache/cupy",
]

# [PWC-CACHE-RESET] Fresh CuPy kernel cache per run on each rig (LEADS L-7: the
# phase-4 fault follows a warm cache; cold runs were clean). Both rig start paths
# give worker N CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_<N> (the later assignment wins
# over the ${HOME} default above). Before any worker starts, each rig empties
# exactly those folders and proves them empty, or the rig is refused.
PWC_CACHE_RESET_OK = "PWC_CACHE_RESET_OK"


def pwc_cache_reset_command(pool: int) -> str:
    """Remote shell: empty /tmp/cupy_cache_gpu_0..pool-1; print the OK marker only if all are empty.
    The GPU list is literal (no seq), and every command's exit status is checked."""
    gpus = " ".join(str(g) for g in range(int(pool)))
    return (
        "for g in %s; do d=/tmp/cupy_cache_gpu_$g; "
        "rm -rf -- \"$d\" || exit 3; mkdir -- \"$d\" || exit 4; "
        "n=$(ls -A -- \"$d\") || exit 5; [ -z \"$n\" ] || exit 5; "
        "done; echo %s" % (gpus, PWC_CACHE_RESET_OK)
    )


# [PWC-STAGE1 B5] Stale-worker kill on a rig. The old remote string was
# `pkill -9 -f pwc_worker_service ...; pkill -9 -f sieve_gpu_worker ...; echo killed`.
# The remote shell running it carries that text in its own command line, so the
# first pkill matched and SIGKILLed the shell itself: the second pkill and the
# echo never ran, and the result was not read (CLAUDE.md hard rule 8; executed
# in tests/test_pwc_stage1.py, G-B5-OLD-SELF-KILL). A bracket pattern such as
# `[p]wc_worker_service` matches a worker's command line but not the shell's,
# which holds the brackets literally. The marker prints only if no match is left.
PWC_STALE_KILL_OK = "PWC_STALE_KILL_OK"
PWC_STALE_KILL_PATTERNS = ("[p]wc_worker_service", "[s]ieve_gpu_worker")
# The S156 reap (SSH route) pattern, bracketed for the same reason.
PWC_S156_PERSISTENT_PATTERN = "[s]ieve_gpu_worker.*--persistent"

# [PWC-STAGE2 BETA-2] A process check must prove it could see processes before
# "none found" counts as clean (docs/VERIFICATION_INTEGRITY_STANDARD.md VIR-1 and
# VIR-5, :30-35, :93-117; the S182 lesson, docs/SESSION_CHANGELOG_20260814_S182.md
# :122-126). Beta X-PWC-STAGE1 R1 finding 2 ran the Stage 1 strings with pgrep
# stubbed to exit 2: the TCP kill still printed the OK marker, and the S156 find
# and reap printed `none` and `reaped:0`.
#
# The positive control: the remote shell that runs one of these strings carries
# the whole string in its own command line (`bash -c '<string>'`, the very fact
# that made the old pkill kill its own shell). This token is in that string
# without brackets, so `pgrep -f` of it MUST find that shell. If it does not,
# pgrep cannot see this host's processes, and every later "no match" is
# UNAVAILABLE, not clean. pgrep's own exit status is then read exactly:
# 1 = no process matched; 0 = one matched; anything else (2 syntax, 3 fatal,
# 127 missing) = the check did not run.
PWC_PROBE_CONTROL_TOKEN = "PWC_PROBE_CONTROL_7f3a"
PWC_KILL_EXIT_SURVIVOR = 6      # a matching process is still alive after the kill
PWC_KILL_EXIT_UNAVAILABLE = 7   # the check could not observe processes
PWC_S156_UNAVAILABLE = "probe_unavailable"


def _pwc_probe_control() -> str:
    """Shell fragment: exit PWC_KILL_EXIT_UNAVAILABLE unless pgrep finds this shell."""
    return ("pgrep -f '%s' >/dev/null 2>&1 || exit %d; "
            % (PWC_PROBE_CONTROL_TOKEN, PWC_KILL_EXIT_UNAVAILABLE))


def pwc_stale_kill_command(patterns=PWC_STALE_KILL_PATTERNS) -> str:
    """Shell: SIGKILL stale PWC and sieve workers; print PWC_STALE_KILL_OK only when
    the probe is proven to work AND reports none left. Exit 6 when one survives,
    exit 7 when the check could not observe processes."""
    kills = "".join("pkill -9 -f '%s' 2>/dev/null; " % p for p in patterns)
    return (kills + "sleep 2; " + _pwc_probe_control()
            + "pgrep -f '%s' >/dev/null 2>&1; r=$?; "
            "if [ $r -eq 1 ]; then echo %s; elif [ $r -eq 0 ]; then exit %d; "
            "else exit %d; fi"
            % ("|".join(patterns), PWC_STALE_KILL_OK,
               PWC_KILL_EXIT_SURVIVOR, PWC_KILL_EXIT_UNAVAILABLE))


# The VM101-local kill before the local TCP workers start ([S165-ZEUS-TCP]). It
# keeps the single pattern the local kill always had: only PWC workers, never
# anything else on the dev box.
PWC_LOCAL_STALE_KILL_PATTERNS = ("[p]wc_worker_service",)


def pwc_s156_find_command() -> str:
    """Remote shell for the S156 find: prints the matching processes, `none` when
    a working probe found none, or PWC_S156_UNAVAILABLE when it could not look."""
    return ("pgrep -f '%s' >/dev/null 2>&1 || { echo %s; exit 0; }; "
            "out=$(pgrep -af \"%s\" 2>/dev/null); r=$?; "
            "if [ $r -eq 1 ]; then echo none; elif [ $r -eq 0 ]; then printf '%%s\\n' \"$out\"; "
            "else echo %s; fi"
            % (PWC_PROBE_CONTROL_TOKEN, PWC_S156_UNAVAILABLE,
               PWC_S156_PERSISTENT_PATTERN, PWC_S156_UNAVAILABLE))


def pwc_s156_reap_command() -> str:
    """Remote shell for the S156 reap: prints `reaped:<count left>` from a working
    probe, or `reaped:unavailable`. `pgrep -c` prints 0 AND exits 1 when nothing
    matches, so exit 0 and 1 are both a real count; the old `|| echo 0` printed a
    second zero there and a fake zero on a probe error (S182)."""
    p = PWC_S156_PERSISTENT_PATTERN
    return ("pkill -15 -f \"%s\" 2>/dev/null; sleep 2; "
            "pkill -9 -f \"%s\" 2>/dev/null; sleep 1; "
            "pgrep -f '%s' >/dev/null 2>&1 || { echo reaped:unavailable; exit 0; }; "
            "remaining=$(pgrep -c -f \"%s\" 2>/dev/null); r=$?; "
            "if [ $r -le 1 ]; then echo \"reaped:$remaining\"; else echo reaped:unavailable; fi"
            % (p, p, PWC_PROBE_CONTROL_TOKEN, p))


# [PWC-STAGE4 SHUTDOWN, Beta X-PWC-SHUTDOWN R1 Q3] BOUND TO THIS COORDINATOR. Every rig worker
# this coordinator launches carries `--launch-token <hex>` on its command line (forensic argv,
# like the --worker-id beside it; the worker reads only its environment,
# persistent/pwc_worker_service.py `__main__`). The token is fresh per coordinator instance
# (uuid4, never reused), so the shutdown reap matches ONLY this coordinator's launches: a
# second coordinator's workers, an earlier run's leftovers and a reused PID cannot match.
# The STARTUP stale kill stays host-wide on purpose (it clears a crashed predecessor).
PWC_LAUNCH_TOKEN_FLAG = "--launch-token"
_PWC_TOKEN_RE = __import__("re").compile(r"^[0-9a-f]{32}$")


def pwc_token_pattern(token: str) -> str:
    """The bracketed (no self-match) pgrep/pkill pattern for one coordinator's workers."""
    if not isinstance(token, str) or not _PWC_TOKEN_RE.match(token):
        raise ValueError(f"launch token {token!r} is not 32 lowercase hex characters")
    return f"[p]wc_worker_service.*{PWC_LAUNCH_TOKEN_FLAG} {token}"


# [PWC-STAGE4 SHUTDOWN] The shutdown reap on the TCP route. The S156 lifecycle fix
# (TB-approved, docs/PROPOSAL_PWC_LIFECYCLE_FIX_S156_v2_0.md:271-293, "shutdown must not
# rely solely on stdin pipe close") asked for an explicit remote reap AFTER shutdown;
# the code ran the S156 reap at startup only. On TCP, stop() sends `shutdown` to every
# connected worker; a worker that was busy, or whose socket closed first, may never read
# it. So after the transport stops, each rig that got workers is checked over SSH:
# wait up to PWC_SHUTDOWN_GRACE_S for every `[p]wc_worker_service` to exit by itself,
# then SIGKILL any left and verify none remain (the probe-checked stale-kill tail).
# Only PWC workers: never sieve_gpu_worker, never anything of RANGE-MINER's.
PWC_SHUTDOWN_GRACE_S = 15
PWC_SHUTDOWN_PATTERNS = ("[p]wc_worker_service",)
PWC_SHUTDOWN_WAIT_TAG = "PWC_SHUTDOWN_WAIT"


def pwc_shutdown_reap_command(grace_s: int = PWC_SHUTDOWN_GRACE_S,
                              patterns=PWC_SHUTDOWN_PATTERNS, token: Optional[str] = None) -> str:
    """Remote shell: first line `PWC_SHUTDOWN_WAIT:<seconds waited>:<pgrep status>`
    (status 1 = all had exited by themselves; 0 = some still alive at the deadline);
    then the stale-kill tail, which prints PWC_STALE_KILL_OK only when a working probe
    finds none left, exits 6 when one survives and 7 when it cannot observe.
    With `token`, the patterns are replaced by this coordinator's token pattern only."""
    if token is not None:
        patterns = (pwc_token_pattern(token),)
    pat = "|".join(patterns)
    return ("n=0; r=0; while [ $n -lt %d ]; do pgrep -f '%s' >/dev/null 2>&1; r=$?; "
            "[ $r -eq 0 ] || break; n=$((n+1)); sleep 1; done; echo %s:$n:$r; "
            % (int(grace_s), pat, PWC_SHUTDOWN_WAIT_TAG)) + pwc_stale_kill_command(patterns)


def classify_shutdown_reap(returncode: int, stdout: str) -> str:
    """exited | killed | survivor | unavailable | error, from one reap's result."""
    lines = (stdout or "").strip().splitlines()
    wait = [ln for ln in lines if ln.startswith(PWC_SHUTDOWN_WAIT_TAG + ":")]
    if returncode == PWC_KILL_EXIT_SURVIVOR:
        return "survivor"
    if returncode == PWC_KILL_EXIT_UNAVAILABLE:
        return "unavailable"
    if returncode != 0 or lines[-1:] != [PWC_STALE_KILL_OK] or not wait:
        return "error"
    status = wait[0].split(":")[-1]
    return "exited" if status == "1" else "killed" if status == "0" else "unavailable"


WORKER_SCRIPT = "sieve_gpu_worker.py"
WORKER_HEARTBEAT_TIMEOUT_S = 120  # [S161] 120s allows ROCm sequential init on 8-GPU rigs (90s was edge case)
JOB_TIMEOUT_S = 600               # seconds max per sieve job


# ─────────────────────────────────────────────────────────────────────────────
# [PWC-STAGE2 RULING 1] The PWC variable-skip (hybrid) QUARANTINE is LIFTED.
#
# Owner ruling 1 (docs/PWC_BACKUP_OWNER_RULINGS_20261119.md, 2026-11-19 20:57):
# "yes- once the threshold is fixed(same as range miner) we lift the block."
# The fix is the threshold contract, done here as RANGE-MINER does it
# (miner/range_miner_coordinator.py build_stripe_assign_payload: one resolved
# directional threshold written as BOTH min_match_threshold and phase2_threshold,
# the pair asserted equal):
#   * run_sieve_pass has no phase2_threshold default; a hybrid pass without one,
#     or with one different from `threshold`, refuses before dispatch;
#   * both hybrid calls in run_trial_persistent pass the trial's directional
#     threshold; the job carries it as both keys;
#   * the hybrid pass is checked against what the worker reports it used
#     (pwc_hybrid_threshold_record below) and fails on a mismatch or a missing
#     report;
#   * the SSH transport (sieve_gpu_worker.py) reports no threshold, so hybrid
#     passes refuse on SSH.
# The other half of the old quarantine text, the hybrid kernels' hardcoded
# expected_skip, is the shared, still-open skip-bound question
# (docs/HYBRID_SKIP_BOUND_AUDIT.md §0); it is the same in RANGE-MINER, so it no
# longer blocks PWC under ruling 1. Removed with the block:
# PWC_HYBRID_QUARANTINE_CODE, PwcHybridThresholdContractUncertified and
# assert_pwc_hybrid_not_quarantined.


# ─────────────────────────────────────────────────────────────────────────────
# [PWC-BACKUP STEP 1] Refusals at the PWC layer
#
# Brief: docs/CLAUDE_CODE_INSTRUCTIONS_PWC_BACKUP_STEP1_20261114.md (S1-S6).
# Gap numbers are those of docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STAGE_A.md P3.
#
# Every authority below is ONE function, refuses by raising, and names the
# setting it refused. None of them has an override flag. [PWC-STAGE2] The hybrid
# quarantine that used to sit above is lifted (ruling 1). None of them makes a
# PWC result count: the
# Step-1 credit gate still refuses PWC (window_optimizer_integration_final.py,
# `_step1_unaccounted_route_evidence`), and that is a step-2 / owner decision.
# ─────────────────────────────────────────────────────────────────────────────
class PwcSettingRefused(RuntimeError):
    """A PWC run refused at the PWC layer, before any worker is spawned or any
    chunk is dispatched, because a setting is missing, unrecognised or outside
    what the record supports. The message names the setting."""


class PwcSeedIntervalRefused(PwcSettingRefused):
    """[S2] The requested seed interval is outside [0, PWC_SEED_SPACE_END)."""


class PwcCoordinatorAddressUnset(PwcSettingRefused):
    """[S3] The address the rig workers dial is unset or not dialable."""


class PwcStableEnvelopeRefused(PwcSettingRefused):
    """[S5] Workers-per-rig x chunk size lies outside PWC_STABLE_ENVELOPE."""


class PwcThresholdContractMismatch(RuntimeError):
    """[S6] For a constant pass, the requested, payload and worker-reported
    (effective) thresholds disagree. [PWC-STAGE2] For a hybrid pass, the same,
    or a completed chunk did not report the threshold it used."""


class PwcWindowAnchorRefused(PwcSettingRefused):
    """[PWC-STAGE2] A window offset other than 0. See validate_pwc_window_offset."""


# ── [PWC-STAGE2] window offset ──────────────────────────────────────────────
# RANGE-MINER keeps two numbers apart (miner/range_miner_worker.py:22-29,
# :601-605): `window_anchor` picks which draws form the residue window, on the
# host, and `generator_phase`, the kernel's trailing int32, is pinned to 0.
# PWC sends one `offset`, and sieve_filter.py uses it for BOTH: it slices the
# draws at it (execute_sieve_job -> load_draws_from_daily3) and passes it as the
# kernel's trailing int32 (GPUSieve.run_sieve and run_hybrid_sieve). At offset 0
# the two engines sieve the same thing; at offset k != 0 PWC also advances the
# generator by k on every kernel that takes the argument (every constant pass
# and the reverse variable-skip pass), which RANGE-MINER does not.
# Separating them needs sieve_filter.py AND the TCP worker, which copies a fixed
# list of job keys (persistent/pwc_worker_service.py _execute_job). The owner
# chose (2026-11-20) to hold that change and REFUSE offset != 0 until it lands,
# so PWC never silently sieves something RANGE-MINER would not.
def validate_pwc_window_offset(offset) -> int:
    if not _is_plain_int(offset):
        raise PwcWindowAnchorRefused(
            f"window offset must be an integer, got {offset!r} ({type(offset).__name__})")
    if int(offset) != 0:
        raise PwcWindowAnchorRefused(
            f"window offset {int(offset)} != 0 refused: PWC would use it both as the "
            f"draw-window start and as the kernel's generator phase, where RANGE-MINER "
            f"pins the phase to 0. Held until the window-anchor / generator-phase split "
            f"lands in sieve_filter.py and the TCP worker (PWC Stage 2 README).")
    return 0


# ── [PWC-BACKUP STAGE 2b] window anchor / generator phase ───────────────────
# Owner ruling 2026-11-20 08:30 PST: the trial's `offset` picks the draw window
# only, the generator phase is 0, and PWC moves to RANGE-MINER's method. The
# split now exists on the TCP worker and in sieve_filter.execute_sieve_job, so
# on the TCP and local routes a job carries `window_anchor` (the trial's offset)
# and `generator_phase` (pinned to 0) and NO `offset`, exactly as the miner's
# coordinator builds its context (miner/range_miner_coordinator.py:2731-2732);
# the worker checks the anchor against the post-session-filter domain and never
# clamps (miner/range_miner_worker.py:913-925). The SSH route ends in
# sieve_gpu_worker.py, a parity-pinned file that reads one `offset` for both
# jobs, so it cannot carry the split: there Stage 2's refusal of offset != 0
# stays (owner, 2026-11-20). The pin is restated, not imported from miner/,
# so PWC does not import the miner package.
PWC_GENERATOR_PHASE_V1_PIN = 0


def validate_pwc_window_anchor(anchor, *, ssh_route: bool) -> int:
    """The trial's window anchor (`config.offset`), checked before dispatch.
    A non-integer or negative anchor is refused here; the upper bound needs the
    dataset and is checked by the worker. On the SSH route anything but 0 is
    refused (validate_pwc_window_offset)."""
    if ssh_route:
        return validate_pwc_window_offset(anchor)
    if not _is_plain_int(anchor):
        raise PwcWindowAnchorRefused(
            f"window anchor must be an integer, got {anchor!r} ({type(anchor).__name__})")
    if int(anchor) < 0:
        raise PwcWindowAnchorRefused(f"window anchor {int(anchor)} < 0 refused")
    return int(anchor)


def pwc_new_key_job(job: Dict[str, Any], window_anchor: int) -> Dict[str, Any]:
    """A copy of an assembled PWC job for the TCP and local routes: `offset` is
    replaced, in its place in the key order, by `window_anchor` and
    `generator_phase` (the v1 pin). Nothing else changes."""
    out: Dict[str, Any] = {}
    for k, v in job.items():
        if k == "offset":
            out["window_anchor"] = int(window_anchor)
            out["generator_phase"] = PWC_GENERATOR_PHASE_V1_PIN
        else:
            out[k] = v
    return out


def pwc_anchor_echo_error(payload: Any, job: Dict[str, Any]) -> Optional[str]:
    """None when the worker's result echoes exactly the window_anchor and
    generator_phase the job carried; otherwise why not. A worker from before
    Stage 2b drops both keys and sieves the window at 0 without saying so, and
    echoes neither: this is where that run is caught instead of counted."""
    if "window_anchor" not in job:
        return None
    if not isinstance(payload, dict):
        return "worker result has no payload to check window_anchor/generator_phase against"
    for k in ("window_anchor", "generator_phase"):
        if k not in payload:
            return (f"worker did not echo {k} (sent {job[k]!r}); a worker without the "
                    f"window-anchor split would have sieved a different window")
        if not _is_plain_int(payload[k]) or int(payload[k]) != int(job[k]):
            return f"worker echoed {k}={payload[k]!r}, job sent {job[k]!r}"
    return None


# ── [S2] seed interval ──────────────────────────────────────────────────────
# End bound (exclusive) of the seed space a PWC pass may partition. Source: the
# owner-decided interval check for Step 1, `0 <= seed_start` and
# `seed_start + total_seeds <= 2**32` (docs/CLAUDE_CODE_INSTRUCTIONS_MINER_SEED_
# START_20261114.md R2), restated for PWC in the step-1 brief item S2. No code
# module commits this constant today; it is not imported from miner/ so that PWC
# keeps loading nothing RANGE-MINER owns.
PWC_SEED_SPACE_END = 1 << 32


def _is_plain_int(v) -> bool:
    import numbers
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


def validate_pwc_seed_interval(seed_start, total_seeds) -> Tuple[int, int]:
    """Return (start, end) for [seed_start, seed_start + total_seeds), or raise
    PwcSeedIntervalRefused. ONE authority, called before the fleet is built
    (run_trial_persistent) and again before chunking (run_sieve_pass)."""
    if not _is_plain_int(seed_start):
        raise PwcSeedIntervalRefused(
            f"seed_start must be an integer, got {seed_start!r} "
            f"({type(seed_start).__name__})")
    if not _is_plain_int(total_seeds):
        raise PwcSeedIntervalRefused(
            f"total_seeds must be an integer, got {total_seeds!r} "
            f"({type(total_seeds).__name__})")
    start, n = int(seed_start), int(total_seeds)
    if start < 0:
        raise PwcSeedIntervalRefused(f"seed_start={start} < 0")
    if n < 0:
        raise PwcSeedIntervalRefused(f"total_seeds={n} < 0")
    end = start + n
    if end > PWC_SEED_SPACE_END:
        raise PwcSeedIntervalRefused(
            f"seed_start + total_seeds = {start} + {n} = {end} > "
            f"PWC_SEED_SPACE_END = {PWC_SEED_SPACE_END} (2**32)")
    return start, end


# ── [S3] the address the rig workers dial ──────────────────────────────────
# The ONE setting is `pwc_coordinator_address` (a constructor / run_trial_persistent
# keyword). There is no default: the literal 192.168.3.127 that used to sit in the
# launch script (and as the worker's own fallback) is the FROZEN bare-metal Zeus,
# not the canonical VM101 (.177) — CLAUDE.md §0, §3. It is distinct from
# `pwc_host`, which is the address the coordinator's TCP server BINDS (0.0.0.0 is a
# valid bind address and a meaningless dial address).
PWC_COORDINATOR_ADDRESS_SETTING = "pwc_coordinator_address"
_UNDIALABLE_ADDRESSES = frozenset({"0.0.0.0", "::", "*", "localhost"})


def validate_pwc_coordinator_address(address) -> str:
    """Return the address rig workers must dial, or raise PwcCoordinatorAddressUnset.

    Refuses: unset/empty, a wildcard bind address, and any loopback address (a
    rig dialling its own loopback reaches itself, not the coordinator)."""
    if address is None or not isinstance(address, str) or not address.strip():
        raise PwcCoordinatorAddressUnset(
            f"{PWC_COORDINATOR_ADDRESS_SETTING} is not set (got {address!r}). It is "
            f"the address the PWC-TCP rig workers dial back to, and it has NO default: "
            f"the former literal 192.168.3.127 is the frozen bare-metal Zeus "
            f"(CLAUDE.md §0). Pass it explicitly, e.g. the canonical dev box's address.")
    a = address.strip()
    if a in _UNDIALABLE_ADDRESSES:
        raise PwcCoordinatorAddressUnset(
            f"{PWC_COORDINATOR_ADDRESS_SETTING}={a!r} is a wildcard/loopback name, "
            f"not an address a rig can dial")
    try:
        import ipaddress
        ip = ipaddress.ip_address(a)
    except ValueError:
        ip = None                         # a hostname: accepted as given
    if ip is not None and (ip.is_loopback or ip.is_unspecified):
        raise PwcCoordinatorAddressUnset(
            f"{PWC_COORDINATOR_ADDRESS_SETTING}={a!r} is a loopback/unspecified "
            f"address; a rig dialling it reaches itself, not the coordinator")
    return a


# ── [S4] transport ──────────────────────────────────────────────────────────
PWC_TRANSPORTS = ("ssh", "tcp")


def validate_pwc_transport(pwc_transport) -> str:
    """Refuse an unset or unrecognised transport instead of falling through to the
    legacy SSH path — the S170-PARITY-1 class (docs/SESSION_CHANGELOG_20260425_
    S170.md:215-221), where a dropped setting silently became the legacy path."""
    if pwc_transport not in PWC_TRANSPORTS:
        raise PwcSettingRefused(
            f"pwc_transport={pwc_transport!r} is not one of {PWC_TRANSPORTS}. PWC "
            f"refuses rather than defaulting to the legacy SSH transport (the "
            f"S170-PARITY-1 silent-downgrade class).")
    return pwc_transport


# The values a PersistentWorkerCoordinator used to take silently when the caller
# omitted them. They are still applied (callers and committed tests construct
# the class without them), but now with a WARNING naming the setting, and the
# stable envelope below refuses any run in which they would size work outside the
# record (the 2,000,000 cap is outside it for every real-size run).
_LEGACY_DEFAULT_WORKER_POOL_SIZE = 8
_LEGACY_DEFAULT_SEED_CAP_NVIDIA  = 5_000_000
_LEGACY_DEFAULT_SEED_CAP_AMD     = 2_000_000


# ── [S5] the recorded stable envelope ──────────────────────────────────────
# Source: docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STAGE_A.md P4 ONLY, which reads
# docs/ROCm_Saturation_Report_S172.md:29-35 (full 26-GPU fleet, three rigs) as its
# repeatable envelope. Each row is a RECORDED point with a cited run:
#     pool 6 per rig, chunk 50,000  — 5 trials clean, 29:30            (:31)
#     pool 8 per rig, chunk 25,000  — 5 trials, 1 transient eviction,
#                                      recovered, 1:30:43              (:33)
# Deliberately NOT encoded (P4 says why):
#     pool 6 / 100,000 — "STABLE — Per operator history": no run cited  (:32)
#     pool 8 / 100,000 — S162 VICTORY clean, but S172 crashed at ~12 min (:35);
#                        P4 adjudicates neither, so the conservative side wins
#     pool 8 / 50,000  — crashed at ~23 min                             (:34)
# A run is INSIDE when some row has workers_per_rig <= row.workers AND
# chunk_seeds <= row.chunk. That dominance reading is an inference from the
# report's own finding that "the empirical boundary is concurrency-driven,
# modulated by chunk size" (:19); it admits, e.g., pool 4 at 50,000 or pool 7 at
# 25,000, which no run measured directly. There is no override flag.
PWC_STABLE_ENVELOPE = (
    # (max workers per rig, max chunk seeds, record)
    (6, 50_000, "ROCm_Saturation_Report_S172.md:31 (pool 6, 50K: 5 trials clean)"),
    (8, 25_000, "ROCm_Saturation_Report_S172.md:33 (pool 8, 25K: 5 trials, 1 recovered eviction)"),
)


def pwc_envelope_violation(workers_per_rig: int, chunk_seeds: int) -> Optional[str]:
    """None if (workers_per_rig, chunk_seeds) is inside PWC_STABLE_ENVELOPE, else
    the reason. Zero workers on every rig is not a configuration to judge."""
    w, c = int(workers_per_rig), int(chunk_seeds)
    if w <= 0:
        return None
    for max_w, max_c, _src in PWC_STABLE_ENVELOPE:
        if w <= max_w and c <= max_c:
            return None
    rows = "; ".join(f"<= {mw} workers/rig at <= {mc:,} seeds/chunk"
                     for mw, mc, _ in PWC_STABLE_ENVELOPE)
    return (f"workers_per_rig={w}, chunk_size={c:,} is outside PWC_STABLE_ENVELOPE "
            f"({rows}; source docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STAGE_A.md P4)")


def assert_pwc_within_stable_envelope(workers_per_rig: int, chunk_seeds: int,
                                      where: str) -> None:
    why = pwc_envelope_violation(workers_per_rig, chunk_seeds)
    if why is not None:
        raise PwcStableEnvelopeRefused(f"[{where}] {why}. No override exists.")


# ── [S6] directional threshold contract, constant passes ───────────────────
def _thr_equal(a, b) -> bool:
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


def pwc_threshold_record(prng_type: str, requested, payload_values, effective_values,
                         effective_unreported: int) -> Dict[str, Any]:
    """The requested / payload / effective record for one constant pass.

    `payload_values` are the `min_match_threshold` values actually written into
    the job dicts; `effective_values` are the values the worker reports it passed
    to the sieve (TCP: sieve_filter's echoed `min_match_threshold`). A worker that
    does not report one is counted, not guessed."""
    payload = sorted({float(v) for v in payload_values})
    effective = sorted({float(v) for v in effective_values})
    disagree = [v for v in payload if not _thr_equal(v, requested)] + \
               [v for v in effective if not _thr_equal(v, requested)]
    return {
        "prng_type": prng_type,
        "requested": float(requested),
        "payload": payload,
        "effective": effective,
        "effective_unreported_chunks": int(effective_unreported),
        "agree": not disagree,
        # verified: every completed chunk reported an effective value, and every
        # value seen equals the request. Unreported is recorded, never assumed.
        "verified": (not disagree) and effective_unreported == 0 and bool(effective),
    }


# [PWC-STAGE2 RULING 1] the hybrid (variable-skip) contract.
# What the TCP worker reports for a hybrid pass is the threshold sieve_filter
# passed to run_hybrid_sieve, ROUNDED TO 4 PLACES (sieve_filter.py:597-599,
# result['single_phase']['threshold']). sieve_filter.py is not edited in Stage 2,
# so the effective value is compared at that resolution; requested and payload
# are compared exactly. RANGE-MINER compares exactly; the gap is recorded in the
# record itself ("effective_resolution": 4).
PWC_HYBRID_ECHO_PLACES = 4


def pwc_hybrid_threshold_record(prng_type: str, requested, payload_min_values,
                                payload_phase2_values, effective_values,
                                effective_unreported: int) -> Dict[str, Any]:
    """Requested / payload / effective record for one hybrid pass.

    `payload_*` are the values written into the job dicts as min_match_threshold
    and phase2_threshold (RANGE-MINER writes the one resolved value as both);
    `effective_values` are the worker's single_phase thresholds. A completed chunk
    that reports none is counted, and for a hybrid pass that fails the contract."""
    req = float(requested)
    req_r = round(req, PWC_HYBRID_ECHO_PLACES)
    p_min = sorted({float(v) for v in payload_min_values})
    p_p2 = sorted({float(v) for v in payload_phase2_values})
    eff = sorted({float(v) for v in effective_values})
    disagree = ([v for v in p_min + p_p2 if not _thr_equal(v, req)]
                + [v for v in eff if not _thr_equal(v, req_r)])
    return {
        "prng_type": prng_type,
        "kind": "hybrid",
        "requested": req,
        "payload_min_match_threshold": p_min,
        "payload_phase2_threshold": p_p2,
        "effective": eff,
        "effective_resolution": PWC_HYBRID_ECHO_PLACES,
        "effective_unreported_chunks": int(effective_unreported),
        "agree": not disagree and int(effective_unreported) == 0,
        "verified": (not disagree) and int(effective_unreported) == 0 and bool(eff),
    }


def assert_pwc_hybrid_threshold_contract(record: Dict[str, Any]) -> None:
    if not record["agree"]:
        raise PwcThresholdContractMismatch(
            f"[{record['prng_type']}] hybrid threshold contract violated: requested="
            f"{record['requested']!r} payload min={record['payload_min_match_threshold']!r} "
            f"phase2={record['payload_phase2_threshold']!r} effective={record['effective']!r} "
            f"(to {record['effective_resolution']} places), completed chunks with no "
            f"reported threshold={record['effective_unreported_chunks']}. A variable-skip "
            f"pass filters at the trial's directional threshold, reported by the worker, "
            f"or it does not return.")


def assert_pwc_threshold_contract(record: Dict[str, Any]) -> None:
    if not record["agree"]:
        raise PwcThresholdContractMismatch(
            f"[{record['prng_type']}] threshold contract violated: requested="
            f"{record['requested']!r} payload={record['payload']!r} "
            f"effective={record['effective']!r}. A constant pass filters at the "
            f"trial's directional threshold or it does not return.")


# ── [S1] completion accounting ──────────────────────────────────────────────
PWC_PASS_ACCOUNTING_SCHEMA  = "pwc_pass_accounting_v1"
PWC_TRIAL_COMPLETION_SCHEMA = "pwc_trial_completion_v1"
_INCOMPLETE_DETAIL_CAP = 100   # chunk rows listed per pass; counts are never capped

# Step-1 phase numbering used by the credit gate (`credit_decision`,
# utils/step1_search_ledger.py) and the legacy evidence builder
# (window_optimizer_integration_final.py, `_legacy_phase_results`).
PWC_PHASES = (
    ("1", "forward", "constant"),
    ("2", "reverse", "constant"),
    ("3", "forward", "variable"),
    ("4", "reverse", "variable"),
)


def _merge_intervals(ivs: List[Tuple[int, int]]) -> List[List[int]]:
    out: List[List[int]] = []
    for s, e in sorted(ivs):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _chunk_outcome(res: Optional[Dict[str, Any]], interval: Tuple[int, int],
                   cancelled: bool) -> Tuple[str, Optional[str]]:
    """(outcome, reason) for one planned chunk.

    outcomes: completed | failed | timed_out | not_dispatched | missing.
    A chunk is `completed` only on status ok AND, when the worker reports the
    seed range it swept, that range equals the chunk. A mismatched range is a
    failure: its survivors belong to some other interval."""
    if cancelled:
        return "not_dispatched", "cancelled before dispatch (pass aborted)"
    if res is None:
        return "missing", "no result recorded"
    if res.get("status") == "ok":
        wr = res.get("worker_seed_range")
        if wr is not None:
            try:
                ws, we = int(wr["start"]), int(wr["end"])
            except Exception:
                return "failed", f"unreadable worker_seed_range {wr!r}"
            if (ws, we) != (int(interval[0]), int(interval[1])):
                return "failed", (f"worker reported seed_range [{ws}, {we}) for "
                                  f"chunk [{interval[0]}, {interval[1]})")
        return "completed", None
    msg = str(res.get("message") or res.get("error") or "unknown")
    if "timeout" in msg.lower():
        return "timed_out", msg[:200]
    return "failed", msg[:200]


def pwc_pass_complete(accounting: Optional[Dict[str, Any]]) -> bool:
    return bool(accounting) and accounting.get("schema") == PWC_PASS_ACCOUNTING_SCHEMA \
        and bool(accounting.get("complete"))


def build_pwc_trial_completion(requested_interval: Tuple[int, int],
                               pass_results: Dict[str, Optional[Dict[str, Any]]]
                               ) -> Dict[str, Any]:
    """Carry each pass's accounting to the trial, per phase and direction.

    `pass_results` maps phase ("1".."4") -> the run_sieve_pass result, or None
    when the pass did not run. A pass that ran but carries no accounting (a
    result from something other than this module's run_sieve_pass) is
    `unaccounted`, which is incomplete — never assumed complete.

    `swept_intervals` holds ONLY complete passes, in the shape the Step-1 credit
    gate reads ({phase: [start, end]}); step 2 would copy it into the PWC sweep
    evidence. Nothing here grants credit."""
    req = [int(requested_interval[0]), int(requested_interval[1])]
    passes: Dict[str, Any] = {}
    swept: Dict[str, List[int]] = {}
    incomplete: List[str] = []
    for ph, direction, mode in PWC_PHASES:
        res = pass_results.get(ph)
        if res is None:
            passes[ph] = {"direction": direction, "mode": mode, "status": "not_run",
                          "accounting": None}
            continue
        acc = res.get("pwc_pass_accounting") if isinstance(res, dict) else None
        if not acc or acc.get("schema") != PWC_PASS_ACCOUNTING_SCHEMA:
            status = "unaccounted"
        elif acc.get("complete") and list(acc.get("requested_interval", [])) == req:
            status = "complete"
            swept[ph] = list(req)
        else:
            status = "incomplete"
        if status != "complete":
            incomplete.append(ph)
        passes[ph] = {"direction": direction, "mode": mode, "status": status,
                      "accounting": acc}
    ran = [ph for ph, *_ in PWC_PHASES if passes[ph]["status"] != "not_run"]
    return {
        "schema": PWC_TRIAL_COMPLETION_SCHEMA,
        "requested_interval": req,
        "passes": passes,
        "passes_run": ran,
        "incomplete_passes": incomplete,
        "swept_intervals": swept,
        # Complete = at least one pass ran and every pass that ran is complete.
        # Which phases are REQUIRED for credit is the credit gate's decision.
        "trial_complete": bool(ran) and not incomplete,
    }


def _pwc_hybrid_echo(payload: Any, job: Dict[str, Any]) -> Optional[float]:
    """[PWC-STAGE2] sieve_filter.execute_sieve_job's report of the single-phase
    hybrid threshold for the job's one family: final_result['per_family'][family]
    ['single_phase']['threshold']. None for a constant job or when not reported."""
    if not job.get("hybrid") or not isinstance(payload, dict):
        return None
    fams = job.get("prng_families") or []
    if len(fams) != 1:
        return None
    try:
        v = payload["per_family"][fams[0]]["single_phase"]["threshold"]
    except (KeyError, TypeError):
        return None
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class WorkerNode:
    hostname: str
    gpu_type: str
    gpu_count: int
    python_env: str
    script_path: str
    username: Optional[str] = None

@dataclass
class WorkerHandle:
    node: WorkerNode
    gpu_id: int
    proc: Optional[subprocess.Popen] = None
    alive: bool = False
    quarantined: bool = False
    jobs_completed: int = 0
    jobs_failed: int = 0
    dispatch_lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def key(self) -> Tuple[str, int]:
        return (self.node.hostname, self.gpu_id)


# ─────────────────────────────────────────────────────────────────────────────
# PersistentWorkerCoordinator
# ─────────────────────────────────────────────────────────────────────────────
class PersistentWorkerCoordinator:
    """
    Manages a pool of persistent sieve_gpu_worker.py processes across the cluster.
    Provides run_sieve_pass() as a drop-in replacement for
    coordinator.execute_distributed_analysis() for Step 1 sieve jobs.
    """

    def __init__(self,
                 config_file: str = "distributed_config.json",
                 worker_pool_size: Optional[int] = None,
                 seed_cap_nvidia: Optional[int] = None,
                 seed_cap_amd: Optional[int] = None,
                 max_per_node: int = 8,
                 pwc_transport: Optional[str] = None,
                 pwc_host: str = "0.0.0.0",
                 pwc_port: int = 5600,
                 min_workers: int = 1,
                 node_allowlist=None,    # [S163-KARG-PWC] partition node filter
                 pwc_coordinator_address: Optional[str] = None,  # [PWC-BACKUP S3]
                 planned_total_seeds: Optional[int] = None):     # [PWC-BACKUP S5]
        self.logger = logging.getLogger("PersistentWorkerCoordinator")
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("[PWC] %(levelname)s %(message)s"))
            self.logger.addHandler(h)
        self.logger.setLevel(logging.INFO)

        # [PWC-BACKUP S4] a sizing setting the caller omitted is no longer taken
        # silently: it is named in a WARNING, and the stable envelope (S5) refuses
        # the run at startup() if the value would size work outside the record.
        def _sized(name, value, legacy):
            if value is None:
                self.logger.warning(
                    f"[PWC-BACKUP S4] {name} not set by the caller — using the legacy "
                    f"default {legacy:,}. The stable-envelope check still applies.")
                return legacy
            return value
        self.config_file     = config_file
        self.node_allowlist  = node_allowlist  # [S163-KARG-PWC] applied in _load_config
        self.worker_pool_size = _sized("worker_pool_size", worker_pool_size,
                                       _LEGACY_DEFAULT_WORKER_POOL_SIZE)
        self.seed_cap_nvidia = _sized("seed_cap_nvidia", seed_cap_nvidia,
                                      _LEGACY_DEFAULT_SEED_CAP_NVIDIA)
        self.seed_cap_amd    = _sized("seed_cap_amd", seed_cap_amd,
                                      _LEGACY_DEFAULT_SEED_CAP_AMD)
        self.max_per_node    = max_per_node
        # [TCP transport] [PWC-BACKUP S4] no default transport: an unset or
        # unrecognised value refuses in startup() (validate_pwc_transport), never
        # becomes the legacy SSH path.
        self.pwc_transport   = pwc_transport
        self.pwc_host        = pwc_host     # the TCP server's BIND address
        self.pwc_port        = pwc_port
        # [PWC-BACKUP S3] the address rig workers DIAL; validated in startup()
        self.pwc_coordinator_address = pwc_coordinator_address
        # [PWC-BACKUP S5] seeds the run will partition, when the caller knows it;
        # bounds the chunk size the startup envelope check judges
        self.planned_total_seeds = planned_total_seeds
        # [PWC-BACKUP S5] effective workers launched per remote rig, filled in by
        # startup(); the dispatch-site envelope check reads it
        self._effective_pool_by_host: Dict[str, int] = {}
        # [PWC-STAGE4 SHUTDOWN] what shutdown() must reap on the TCP route
        self._tcp_reap_targets: Dict[str, List[str]] = {}   # host -> ssh argv prefix
        import uuid as _uuid_tok
        self._launch_token: str = _uuid_tok.uuid4().hex       # [Beta X-PWC-SHUTDOWN R1 Q3]
        self._tcp_local_procs: List[Any] = []                # VM101 local Popen objects
        self.shutdown_reap: Dict[str, str] = {}              # host -> classify_shutdown_reap
        self.min_workers     = min_workers  # S161: readiness gate
        self._tcp_transport  = None  # lazily started if pwc_transport == "tcp"
        self._tcp_launch_complete_time: float = 0.0  # set by _tcp_launch_workers()
        self._tcp_expected_workers: int = 0           # set by _tcp_launch_workers()
        self._progress_writer = None  # guard for TCP early-return startup path

        self.nodes: List[WorkerNode] = []
        self.workers: List[WorkerHandle] = []   # AMD rig workers (persistent)
        self._lock = threading.Lock()
        self._started = False
        # [S163] Zeus local GPU slot pool — bind GPU by acquired slot, not chunk index.
        # Chunk index (idx % gpu_count) is NOT a safe proxy for live GPU ownership under
        # concurrent local dispatch — two simultaneous jobs can map to the same gpu_id.
        # Use a Queue of available GPU IDs so each acquired slot carries its own GPU binding.
        import queue as _queue
        self._localhost_gpu_slots: "_queue.Queue[int]" = _queue.Queue()
        for _gpu_id in range(2):  # Zeus has 2 GPUs
            self._localhost_gpu_slots.put(_gpu_id)
        # Per-node semaphores — throttle concurrent dispatch to max_per_node (S133-A lesson)
        self._node_semaphores: Dict[str, threading.Semaphore] = {}
        # [S151] Per-node respawn locks — serialize respawns with stagger to prevent SSH hammer
        self._node_respawn_locks: Dict[str, threading.Lock] = {}

        self._load_config()

    # ─────────────────────────────────────────────────────────────────────────
    # Config
    # ─────────────────────────────────────────────────────────────────────────
    def _load_config(self):
        try:
            with open(self.config_file) as f:
                cfg = json.load(f)
        except Exception as e:
            # [PWC-BACKUP S4] This used to log and return with ZERO nodes, which
            # then ran on as an empty fleet (SSH: IndexError at dispatch; TCP: a
            # ready-gate timeout minutes later). Refuse here, naming the file.
            raise PwcSettingRefused(
                f"cannot load the PWC node config {self.config_file!r}: {e}") from e

        # [RESOLVED EXECUTION SET] The PWC ready gate (:864, min_workers) is the
        # real per-GPU, full-fleet, blocking check — and it was pointed at the
        # bare-metal addresses, so it structurally cannot pass while the rigs are
        # booted into Proxmox. The gate itself is UNCHANGED: `min_workers` keeps
        # its value and its meaning, the live GPU probe still caps the spawn, and
        # a short pool is still a `RuntimeError` before dispatch. Only the node
        # list and the addresses now come from the run's frozen set.
        for nc in _execution_set_nodes(cfg.get("nodes", []),
                                       consumer="PersistentWorkerCoordinator"):
            hostname = nc["hostname"]
            # [S163-KARG-PWC] Apply node_allowlist filter BEFORE building any per-node
            # state (semaphores, respawn_locks, S156 cleanup targets, worker launches).
            # TB ruling: filter at load time so each partition only owns its nodes.
            if self.node_allowlist is not None and hostname not in self.node_allowlist:
                self.logger.info(
                    f"Node skipped (not in partition allowlist): {hostname}"
                )
                continue
            node = WorkerNode(
                hostname   = hostname,
                gpu_type   = nc.get("gpu_type", "unknown"),
                gpu_count  = nc.get("gpu_count", 1),
                python_env = nc["python_env"],
                script_path= nc["script_path"],
                username   = nc.get("username"),
            )
            self.nodes.append(node)
            # Create per-node semaphore — limits concurrent in-flight jobs to max_per_node
            self._node_semaphores[node.hostname] = threading.Semaphore(self.max_per_node)
            # [S151] Per-node respawn lock — serializes respawns with stagger
            self._node_respawn_locks[node.hostname] = threading.Lock()
            self.logger.info(f"Node loaded: {node.hostname} ({node.gpu_count}× {node.gpu_type})")

    def _is_rocm(self, node: WorkerNode) -> bool:
        gt = (node.gpu_type or "").lower()
        return ("rx" in gt) or ("amd" in gt) or ("rocm" in gt)

    def _probe_live_rocm_gpu_count(self, node: WorkerNode) -> Optional[int]:
        """
        [S170-LIVE-GPU-PROBE]
        Return live CuPy-visible ROCm GPU count for this rig.

        Fail-closed policy:
          - valid integer >= 0: use it
          - probe error/timeout: return 0, skip this node for this run

        Rationale: distributed_config.json is desired topology, not live truth.
        """
        if self._is_localhost(node.hostname) or not self._is_rocm(node):
            return node.gpu_count

        import subprocess as _probe_sp

        host = node.hostname
        user = node.username
        activate_path = node.python_env.replace("/bin/python", "/bin/activate")

        probe_py = "import cupy as cp\nprint(cp.cuda.runtime.getDeviceCount())\n"

        cmd = (
            "source " + activate_path + " && "
            "cd " + node.script_path + " && "
            "export HSA_OVERRIDE_GFX_VERSION=10.3.0 && "
            "export HSA_ENABLE_SDMA=0 && "
            "export HSA_ENABLE_DEBUG_TRAP=0 && "
            "export ROCM_PATH=/opt/rocm && "
            "export HIP_PATH=/opt/rocm/hip && "
            "export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/hip/lib:${LD_LIBRARY_PATH} && "
            "export PATH=/opt/rocm/bin:${PATH} && "
            "unset ROCR_VISIBLE_DEVICES && unset HIP_VISIBLE_DEVICES && unset CUDA_VISIBLE_DEVICES && "
            "python3 - <<'PY'\n" + probe_py + "PY"
        )

        ssh_cmd = [
            "ssh", "-q",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=10",
            (user + "@" + host) if user else host,
            cmd,
        ]

        try:
            r = _probe_sp.run(ssh_cmd, capture_output=True, text=True, timeout=45)
            if r.returncode != 0:
                self.logger.error(
                    f"[S170-LIVE-GPU-PROBE] {host}: probe failed rc={r.returncode}; "
                    f"FAIL-CLOSED skipping node. stderr={r.stderr.strip()[:300]}"
                )
                return 0

            lines = [x.strip() for x in r.stdout.splitlines() if x.strip()]
            live = int(lines[-1])
            configured = int(node.gpu_count)

            if live != configured:
                self.logger.warning(
                    f"[S170-LIVE-GPU-PROBE] {host}: live ROCm GPU count "
                    f"{live}/{configured}; launching only live GPUs this run"
                )
            else:
                self.logger.info(
                    f"[S170-LIVE-GPU-PROBE] {host}: live ROCm GPU count OK {live}/{configured}"
                )

            return max(0, live)

        except Exception as e:
            self.logger.error(
                f"[S170-LIVE-GPU-PROBE] {host}: probe exception {e}; "
                f"FAIL-CLOSED skipping node"
            )
            return 0

    def _is_localhost(self, hostname: str) -> bool:
        return hostname in ("localhost", "127.0.0.1", socket.gethostname())

    def _seed_cap(self, node: WorkerNode) -> int:
        if "RTX 3080" in node.gpu_type or "RTX 3090" in node.gpu_type:
            return self.seed_cap_nvidia
        return self.seed_cap_amd

    # ─────────────────────────────────────────────────────────────────────────
    # Worker lifecycle
    # ─────────────────────────────────────────────────────────────────────────
    def _remote_rocm_nodes(self) -> List[WorkerNode]:
        return [n for n in self.nodes
                if not self._is_localhost(n.hostname) and self._is_rocm(n)]

    def _startup_refusals(self) -> None:
        """[PWC-BACKUP S3/S4/S5] Everything that can refuse a run, checked BEFORE
        the first SSH (the S156 reap below), the first spawn or the TCP bind.

        * transport: unset/unknown refuses (S4);
        * TCP with any remote rig: the dial address must be set (S3);
        * envelope (S5): judged on the largest per-rig pool this run can launch,
          min(worker_pool_size, configured gpu_count) — the live probe can only
          lower it — against the largest chunk it can cut, seed_cap_amd, bounded
          by planned_total_seeds when the caller supplied it. Both are upper
          bounds, so a run judged inside here cannot be outside at dispatch;
          run_sieve_pass re-checks the exact values anyway."""
        validate_pwc_transport(self.pwc_transport)
        # [PWC-STAGE3 R1 3b-Q3] a nonpositive chunk cap cuts no seeds (cap 0) or
        # walks the cursor backwards (cap < 0), so run_sieve_pass's chunk loop
        # never ends. Refuse it here, before any worker is touched, as RANGE-MINER
        # refuses a nonpositive size (miner/range_miner_coordinator.py:769-770,
        # :789-790; miner/range_miner_worker.py:681-682). The envelope below
        # judges upper bounds only.
        for _cap_name in ("seed_cap_amd", "seed_cap_nvidia"):
            _cap = getattr(self, _cap_name)
            if int(_cap) <= 0:
                raise PwcSettingRefused(
                    f"[startup, before any worker is spawned] {_cap_name} must be "
                    f"positive, got {_cap!r}. No override exists.")
        rigs = self._remote_rocm_nodes()
        if self.pwc_transport == "tcp" and rigs:
            self.pwc_coordinator_address = validate_pwc_coordinator_address(
                self.pwc_coordinator_address)
        if rigs:
            pool_bound = max(min(int(self.worker_pool_size), int(n.gpu_count))
                             for n in rigs)
            chunk_bound = int(self.seed_cap_amd)
            if self.planned_total_seeds is not None:
                chunk_bound = min(chunk_bound, max(1, int(self.planned_total_seeds)))
            assert_pwc_within_stable_envelope(
                pool_bound, chunk_bound,
                where=(f"startup, before any worker is spawned: worker_pool_size="
                       f"{self.worker_pool_size}, seed_cap_amd={self.seed_cap_amd}, "
                       f"planned_total_seeds={self.planned_total_seeds}"))

    def startup(self):
        """Spawn persistent workers on all AMD rigs with ROCm stagger."""
        if self._started:
            return
        # [PWC-BACKUP] refuse before anything touches a rig
        self._startup_refusals()
        # ── [S156-BANDAID v2] Pre-spawn targeted cleanup ──────────────────────
        # TEMPORARY SAFETY NET — not the root fix.
        # Root fix: session-scoped PWC (Phase B, S157).
        # TB: targeted --persistent match, SIGTERM first, log exact procs.
        import subprocess as _s156_sp
        import time as _s156_time
        for _s156_node in self.nodes:
            if self._is_localhost(_s156_node.hostname):
                continue
            if not self._is_rocm(_s156_node):
                continue
            try:
                _s156_host = _s156_node.hostname
                _s156_user = _s156_node.username
                _s156_ssh = ("ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "
                             "-o BatchMode=yes " + _s156_user + "@" + _s156_host + " ")
                # [PWC-STAGE2 BETA-2] the find and reap prove their probe works first
                # (pwc_s156_find_command / pwc_s156_reap_command); an unavailable
                # probe is reported as such, never as "no stale workers".
                _s156_find = _s156_ssh + shlex.quote(pwc_s156_find_command())
                _s156_r = _s156_sp.run(
                    _s156_find, shell=True, capture_output=True, text=True, timeout=10
                )
                _s156_found = _s156_r.stdout.strip()
                if _s156_r.returncode != 0 or not _s156_found \
                        or _s156_found == PWC_S156_UNAVAILABLE:
                    self.logger.error(
                        "  [S156] " + _s156_host + ": stale-worker probe UNAVAILABLE "
                        "(rc=" + str(_s156_r.returncode) + ", out=" + repr(_s156_found[:80])
                        + ") — not known to be clean")
                elif _s156_found != "none":
                    self.logger.info(
                        "  [S156] " + _s156_host + ": found stale workers: "
                        + _s156_found[:200]
                    )
                    _s156_reap = _s156_ssh + shlex.quote(pwc_s156_reap_command())
                    _s156_r2 = _s156_sp.run(
                        _s156_reap, shell=True, capture_output=True, text=True, timeout=15
                    )
                    self.logger.info(
                        "  [S156] " + _s156_host + ": reap result: "
                        + _s156_r2.stdout.strip()
                    )
                    _s156_tokens = _s156_r2.stdout.split()
                    if "reaped:unavailable" in _s156_tokens:
                        self.logger.error(
                            "  [S156] " + _s156_host + ": reap probe UNAVAILABLE — "
                            "remaining workers not known")
                    elif "reaped:0" not in _s156_tokens:   # [PWC-STAGE1 B5]
                        self.logger.warning(
                            "  [S156] " + _s156_host + ": reap did not report "
                            "'reaped:0' (rc=" + str(_s156_r2.returncode) + ")")
                else:
                    self.logger.info(
                        "  [S156] " + _s156_host + ": no stale persistent workers"
                    )
            except Exception as _s156_e:
                self.logger.warning(
                    "  [S156] " + _s156_host
                    + ": pre-spawn cleanup failed: " + str(_s156_e)
                )
        _s156_time.sleep(2)
        # ── end [S156-BANDAID v2] ───────────────────────────────────────────
        # [TCP transport] in TCP mode, skip SSH worker spawning entirely
        if self.pwc_transport == "tcp":
            self.logger.info("[PWC-TCP] TCP transport mode — skipping SSH worker spawn")
            self._started = True
            self.logger.info("Worker pool ready: 0/0 alive (TCP mode — workers connect via TCP)")
            from persistent.pwc_transport_base import build_transport
            self._tcp_transport = build_transport(
                pwc_transport=self.pwc_transport,
                pwc_host=self.pwc_host,
                pwc_port=self.pwc_port,
            )
            if self._tcp_transport is not None:
                # [PWC-STAGE1 A3 level 1] Only worker ids this startup() launched
                # are served. The set starts EMPTY, before the listener exists, so
                # a stale worker left by an earlier coordinator cannot register in
                # the gap before the rig kill (it is refused and retries). Ids are
                # added in _tcp_launch_workers as each launch is issued, and
                # removed again when a launch returns no PID.
                self._tcp_transport.set_allowed_worker_ids(set())
                self._tcp_transport.start()
                self.logger.info(
                    f"[PWC-TCP] transport started on {self.pwc_host}:{self.pwc_port}"
                )
            # S161: SSH-launch workers on all AMD rigs now that TCP server is bound
            self._tcp_launch_workers()

            # Initialize ProgressWriter for web dashboard — must be done in TCP path
            # SSH-PWC does this after spawn loop but TCP returns early above
            try:
                from progress_display import ProgressWriter
                self._progress_writer = ProgressWriter("Forward Sieve", total_jobs=100, total_seeds=0)
                for node in self.nodes:
                    if self._is_localhost(node.hostname):
                        self._progress_writer.register_node("localhost", "RTX 3080 Ti", 2)
                    else:
                        self._progress_writer.register_node(node.hostname, node.gpu_type, node.gpu_count)
                self.logger.info("[PWC-TCP] ProgressWriter initialized for web dashboard")
            except Exception as e:
                self.logger.warning(f"ProgressWriter unavailable: {e}")
                self._progress_writer = None

            return

        self.logger.info("Starting persistent worker pool...")
        for node in self.nodes:
            if self._is_localhost(node.hostname):
                self.logger.info(f"  Zeus ({node.hostname}) — uses local sieve path, no persistent worker")
                continue
            if not self._is_rocm(node):
                self.logger.info(f"  {node.hostname} — non-ROCm, skipping persistent workers")
                continue
            pool = min(self.worker_pool_size, node.gpu_count)
            _target = f"{node.username}@{node.hostname}" if node.username else node.hostname
            if pool > 0 and not self._reset_rig_cupy_cache(
                    ["ssh", "-q", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
                     "-o", "ConnectTimeout=10", _target], node.hostname, pool):
                for gpu_id in range(pool):   # same state as a failed spawn
                    handle = WorkerHandle(node=node, gpu_id=gpu_id)
                    handle.quarantined = True
                    self.workers.append(handle)
                continue
            self.logger.info(f"  {node.hostname}: spawning {pool} workers (stagger {ROCM_SPAWN_STAGGER_S}s)")
            for gpu_id in range(pool):
                handle = WorkerHandle(node=node, gpu_id=gpu_id)
                success = self._spawn_worker(handle)
                if success:
                    self.workers.append(handle)
                    self.logger.info(f"    ✅ {node.hostname}:GPU{gpu_id} — worker alive")
                else:
                    handle.quarantined = True
                    self.workers.append(handle)
                    self.logger.warning(f"    ⚠️  {node.hostname}:GPU{gpu_id} — spawn failed, quarantined")
                # Stagger to prevent simultaneous HIP init (S130/S133 lesson)
                if gpu_id < pool - 1:
                    time.sleep(ROCM_SPAWN_STAGGER_S)
            # [PWC-BACKUP S5] effective workers on this rig, for the dispatch check
            self._effective_pool_by_host[node.hostname] = sum(
                1 for w in self.workers
                if w.node.hostname == node.hostname and not w.quarantined)
        self._started = True
        alive = sum(1 for w in self.workers if w.alive)
        self.logger.info(f"Worker pool ready: {alive}/{len(self.workers)} alive")
        # Initialize ProgressWriter for web dashboard (mirrors coordinator.py)
        try:
            from progress_display import ProgressWriter
            self._progress_writer = ProgressWriter("Forward Sieve", total_jobs=100, total_seeds=0)
            for node in self.nodes:
                if self._is_localhost(node.hostname):
                    self._progress_writer.register_node("localhost", "RTX 3080 Ti", 2)
                else:
                    self._progress_writer.register_node(node.hostname, node.gpu_type, node.gpu_count)
        except Exception as e:
            self.logger.warning(f"ProgressWriter unavailable: {e}")
            self._progress_writer = None

        # S161 v2: no hostname map needed — extract IP from worker_id directly
        # worker_id format: "192_168_3_120_gpu0" → IP "192.168.3.120"
        # DNS resolution not used — Zeus cannot resolve rig-* hostnames

    def _reset_rig_cupy_cache(self, ssh_base: List[str], host: str, pool: int) -> bool:
        """[PWC-CACHE-RESET] Empty this rig's per-GPU CuPy caches. False = refuse the rig."""
        try:
            r = subprocess.run(ssh_base + [pwc_cache_reset_command(pool)],
                               capture_output=True, text=True, timeout=30)
        except Exception as e:
            self.logger.error(f"[PWC-CACHE-RESET] {host}: REFUSED, reset not run: {e}")
            return False
        if r.returncode != 0 or r.stdout.strip().splitlines()[-1:] != [PWC_CACHE_RESET_OK]:
            self.logger.error(f"[PWC-CACHE-RESET] {host}: REFUSED, rc={r.returncode} "
                              f"stderr={r.stderr.strip()[:200]!r}")
            return False
        self.logger.info(f"[PWC-CACHE-RESET] {host}: /tmp/cupy_cache_gpu_0..{pool - 1} emptied and verified")
        return True

    def _spawn_worker(self, handle: WorkerHandle) -> bool:
        """SSH + launch sieve_gpu_worker.py on remote GPU, confirm heartbeat."""
        node   = handle.node
        gpu_id = handle.gpu_id

        # [S155-ROCR] Restore ROCR_VISIBLE_DEVICES per-worker GPU isolation.
        # Root cause of rrig6600c OOM: without ROCR isolation, each worker's
        # HIP runtime enumerates ALL 8 GPUs and maps ALL their VRAM into the
        # process VA space: 8 GPUs × ~5GB = ~41GB VA per worker.
        # Under production load the kernel tries to back these pages → OOM.
        # With ROCR_VISIBLE_DEVICES={gpu_id}: each worker only sees 1 GPU,
        # VA per worker = ~5GB. 8 workers × 5GB = 40GB total — safe.
        # Worker uses Device(0) because ROCR remaps assigned GPU → device 0.
        rocm_env = " ".join(ROCM_ENV_VARS + [
            f"ROCR_VISIBLE_DEVICES={gpu_id}",
            f"CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_{gpu_id}",  # [S157] per-worker isolated cache
        ])

        activate = f"source {os.path.join(os.path.dirname(node.python_env), 'activate')}"
        cmd_body = (
            f"cd {node.script_path} && "
            f"{activate} && "
            f"env {rocm_env} {node.python_env} -u {WORKER_SCRIPT} --gpu-id {gpu_id} --persistent"
        )
        ssh_cmd = [
            "ssh",
            "-q",                              # suppress SSH banners/warnings
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=10",    # [S152] probe 10× before giving up = 300s tolerance
            "-o", "ConnectTimeout=10",         # [S152] fail fast on dead hosts
            f"{node.username}@{node.hostname}" if node.username else node.hostname,
            cmd_body
        ]

        try:
            proc = subprocess.Popen(
                ssh_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,  # unbuffered binary — readline handles framing
            )
            # Drain lines until we see the worker's {"status": "ready"} heartbeat.
            # Shell activation (conda/venv banners) may emit non-JSON lines first.
            deadline = time.time() + WORKER_HEARTBEAT_TIMEOUT_S
            ready = False
            while time.time() < deadline:
                line = _read_with_timeout(proc.stdout, 5.0)
                if line is None:
                    break
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                if '"status"' in line and '"ready"' in line:
                    ready = True
                    break
            if not ready:
                proc.kill()
                self.logger.error(f"Spawn failed {node.hostname}:GPU{gpu_id}: no heartbeat within {WORKER_HEARTBEAT_TIMEOUT_S}s")
                return False
            handle.proc  = proc
            handle.alive = True
            return True
        except Exception as e:
            self.logger.error(f"Spawn failed {node.hostname}:GPU{gpu_id}: {e}")
            return False

    def _tcp_launch_workers(self) -> None:
        """
        S161: SSH-launch pwc_worker_service on each AMD rig GPU via nohup.
        TB Gate 1 requirements:
          - TCP server already bound before this is called
          - Kill stale pwc_worker_service processes first
          - nohup launch per GPU with per-GPU log file
          - ROCm env vars same as _spawn_worker()
          - 180s connect deadline starts AFTER last launch issued
        """
        import subprocess as _sp
        import time as _t

        total_launched = 0
        self.logger.info(f"[PWC-TCP] launch token {self._launch_token} (the shutdown reap "
                         f"matches only workers carrying it)")

        # [S165-ZEUS-TCP] Kill stale Zeus local pwc workers before launching fresh ones
        # [PWC-STAGE2 BETA-2] The result is READ now (Beta X-PWC-STAGE1 R1 Q3: the
        # old local pkill ignored its result and always logged success). The same
        # probe-checked shell as the rig kill runs locally under `bash -c`; unless
        # it confirms none are left, the local workers are NOT launched this run.
        _zeus_kill_ok = False
        try:
            import subprocess as _sp_zeus_kill
            _zk = _sp_zeus_kill.run(
                ["bash", "-c", pwc_stale_kill_command(PWC_LOCAL_STALE_KILL_PATTERNS)],
                capture_output=True, text=True, timeout=15
            )
            _zeus_kill_ok = (_zk.returncode == 0 and
                             (_zk.stdout or "").strip().splitlines()[-1:] == [PWC_STALE_KILL_OK])
            if _zeus_kill_ok:
                self.logger.info("[PWC-TCP] Zeus: stale local workers killed and none left")
            else:
                self.logger.error(
                    "[PWC-TCP] Zeus: local stale-worker kill not confirmed (rc="
                    + str(_zk.returncode) + ("; probe UNAVAILABLE" if _zk.returncode ==
                                             PWC_KILL_EXIT_UNAVAILABLE else "")
                    + ") — local workers REFUSED this run")
        except Exception as _zke:
            self.logger.error("[PWC-TCP] Zeus: local stale-worker kill not run: " + str(_zke)
                              + " — local workers REFUSED this run")

        for node in self.nodes:
            if self._is_localhost(node.hostname):
                continue
            if not self._is_rocm(node):
                continue

            host = node.hostname
            user = node.username
            # [S170-LIVE-GPU-PROBE] Use live ROCm enumeration, not static config only.
            _live_gpu_count = self._probe_live_rocm_gpu_count(node)
            node.gpu_count = min(int(node.gpu_count), int(_live_gpu_count))
            pool = min(self.worker_pool_size, node.gpu_count)

            if pool <= 0:
                self.logger.error(
                    "[S170-LIVE-GPU-PROBE] " + host +
                    ": no live ROCm GPUs available; skipping node for this run"
                )
                continue

            ssh_base = ["ssh", "-q",
                        "-o", "StrictHostKeyChecking=no",
                        "-o", "BatchMode=yes",
                        "-o", "ConnectTimeout=10",
                        user + "@" + host]
            activate_path = node.python_env.replace("/bin/python", "/bin/activate")

            # Step 1: Kill stale pwc_worker_service AND sieve_gpu_worker processes
            # S162: sieve_gpu_worker reap added — same fix as S156 SSH-PWC bandaid.
            # Stale sieve_gpu_worker processes accumulate PageTables → crash.
            # [PWC-STAGE1 B5] bracket patterns (no self-match) and the result is
            # READ: no marker, or a non-zero exit, refuses the rig for this run,
            # the same way a failed cache reset does (Step 1b).
            try:
                _kill_r = _sp.run(
                    ssh_base + [pwc_stale_kill_command()],
                    capture_output=True, text=True, timeout=15
                )
            except Exception as e:
                self.logger.error("[PWC-TCP] " + host + ": REFUSED, stale-worker kill not run: " + str(e))
                continue
            if (_kill_r.returncode != 0
                    or (_kill_r.stdout or "").strip().splitlines()[-1:] != [PWC_STALE_KILL_OK]):
                self.logger.error(
                    "[PWC-TCP] " + host + ": REFUSED, stale-worker kill not confirmed "
                    "(rc=" + str(_kill_r.returncode)
                    + {PWC_KILL_EXIT_SURVIVOR: ", a worker survived the kill",
                       PWC_KILL_EXIT_UNAVAILABLE: ", process probe UNAVAILABLE"}.get(
                           _kill_r.returncode, "")
                    + ", stderr="
                    + repr((_kill_r.stderr or "").strip()[:200]) + ")")
                continue
            self.logger.info("[PWC-TCP] " + host + ": stale workers killed and none left (pwc + sieve_gpu)")

            # Step 1b: [PWC-CACHE-RESET] fresh CuPy cache, after the kill, before any launch
            if not self._reset_rig_cupy_cache(ssh_base, host, pool):
                continue

            # Step 2: Launch one worker per GPU via temp bash script
            # [PWC-BACKUP S3] The worker reads its dial target ONLY from the
            # environment (persistent/pwc_worker_service.py `__main__` has no
            # argparse), so PWC_HOST and PWC_PORT are exported here. Before this,
            # neither was exported: every rig worker dialled the worker's own
            # fallback 192.168.3.127 on port 5600 whatever this coordinator bound,
            # and the `--host/--port` arguments below were never read.
            _dial = shlex.quote(self.pwc_coordinator_address)
            _launched_here = 0
            # [Beta X-PWC-SHUTDOWN R1 Q3] recorded at the launch ATTEMPT, not on a confirmed
            # PID: if every acknowledgment from this rig is lost, a launch may still have run.
            self._tcp_reap_targets[host] = list(ssh_base)

            for gpu_id in range(pool):
                rocm_env = " ".join(ROCM_ENV_VARS + [
                    f"ROCR_VISIBLE_DEVICES={gpu_id}",
                    f"CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_{gpu_id}",
                ])
                log_file = f"/tmp/pwc_tcp_worker_{host.replace('.', '_')}_gpu{gpu_id}.log"
                worker_id = f"{host.replace('.', '_')}_gpu{gpu_id}"

                script    = f"/tmp/pwc_tcp_launch_gpu{gpu_id}.sh"

                # Build script content — no quoting issues since it is a file
                script_lines = [
                    "#!/bin/bash",
                    "source " + activate_path,
                    "cd " + node.script_path,
                ]
                for var in ROCM_ENV_VARS:
                    script_lines.append("export " + var)
                script_lines += [
                    "export ROCR_VISIBLE_DEVICES=" + str(gpu_id),
                    "export CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_" + str(gpu_id),
                    "export PWC_GPU_ID=" + str(gpu_id),
                    "export PWC_WORKER_ID=" + worker_id,
                    "export PWC_HOST=" + _dial,                # [PWC-BACKUP S3]
                    "export PWC_PORT=" + str(int(self.pwc_port)),  # [PWC-BACKUP S4]
                    "export PYTHONPATH=.",
                    (
                        # --host/--port kept for process-listing forensics only;
                        # the worker reads PWC_HOST/PWC_PORT above.
                        "nohup " + node.python_env + " -m persistent.pwc_worker_service "
                        "--host " + _dial + " --port " + str(int(self.pwc_port)) + " "
                        "--gpu-id " + str(gpu_id) + " --worker-id " + worker_id + " "
                        + PWC_LAUNCH_TOKEN_FLAG + " " + self._launch_token + " "   # [X-PWC-SHUTDOWN Q3]
                        ">> " + log_file + " 2>&1 &"
                    ),
                    "echo PID=$!",
                ]
                script_content = chr(10).join(script_lines)

                try:
                    # Write script to rig via stdin pipe
                    write_r = _sp.run(
                        ssh_base + ["cat > " + script + " && chmod +x " + script],
                        input=script_content.encode(),
                        capture_output=True, timeout=10
                    )
                    if write_r.returncode != 0:
                        self.logger.error("[PWC-TCP] " + host + ":GPU" + str(gpu_id) + " script write failed")
                        continue

                    # Execute the script
                    # [PWC-STAGE1 A3 level 1] admit this id before the worker can dial
                    self._tcp_transport.allow_worker_id(worker_id)
                    try:
                        exec_r = _sp.run(
                            ssh_base + ["bash " + script],
                            capture_output=True, text=True, timeout=10
                        )
                    except Exception:
                        self._tcp_transport.revoke_worker_id(worker_id)
                        raise
                    _launch_time = _t.time()  # S161: start deadline from actual launch
                    pid_line = (exec_r.stdout or "").strip()
                    if exec_r.returncode != 0 or not any(
                            ln.startswith("PID=") for ln in pid_line.splitlines()):
                        # [PWC-STAGE1 A3 level 1] no PID: not in the launch set
                        # [PWC-STAGE2 BETA-3] revoke also drops a connection this id
                        # may already hold (pwc_transport_tcp.revoke_worker_id), and
                        # an unconfirmed launch is not counted: it must not size the
                        # online wait or the rig's effective pool.
                        self._tcp_transport.revoke_worker_id(worker_id)
                        self.logger.error(
                            "[PWC-TCP] " + host + ":GPU" + str(gpu_id) + " launch NOT "
                            "confirmed (rc=" + str(exec_r.returncode) + ", no PID line) — "
                            "id revoked, not counted")
                        continue
                    self.logger.info(
                        "[PWC-TCP] " + host + ":GPU" + str(gpu_id) +
                        " launched — " + pid_line + " log=" + log_file
                    )
                    total_launched += 1
                    _launched_here += 1
                except Exception as e:
                    self.logger.error("[PWC-TCP] " + host + ":GPU" + str(gpu_id) + " launch failed: " + str(e))

                # S161 v2: no per-worker wait — workers connect fast (no ROCm at startup)
                # All workers launched with 1s stagger, then _tcp_wait_online() handles
                # the online barrier before broadcasting init.
                self.logger.info(
                    "[PWC-TCP] " + host + ":GPU" + str(gpu_id) + " launched"
                )
            # [PWC-BACKUP S5] effective workers on this rig, for the dispatch check
            self._effective_pool_by_host[host] = _launched_here

        # [S165-ZEUS-TCP] Launch persistent CUDA workers on Zeus (localhost)
        # Workers connect back to 127.0.0.1:port — same TCP path as AMD rigs.
        # PWC_USE_ROCM=0 → CUDA_VISIBLE_DEVICES path in pwc_worker_service._setup_env()
        import subprocess as _sp_zeus
        for _zeus_node in self.nodes:
            if not self._is_localhost(_zeus_node.hostname):
                continue
            if not _zeus_kill_ok:                # [PWC-STAGE2 BETA-2]
                continue
            _zeus_pool = min(2, _zeus_node.gpu_count)  # 2 RTX 3080Ti
            for _zeus_gpu in range(_zeus_pool):
                _zeus_worker_id = f"zeus_gpu{_zeus_gpu}"
                _zeus_log = f"/tmp/pwc_tcp_worker_zeus_gpu{_zeus_gpu}.log"
                _zeus_env = os.environ.copy()
                _zeus_env["PWC_GPU_ID"]           = str(_zeus_gpu)
                _zeus_env["PWC_WORKER_ID"]        = _zeus_worker_id
                _zeus_env["PWC_HOST"]             = "127.0.0.1"
                _zeus_env["PWC_PORT"]             = str(self.pwc_port)
                _zeus_env["PWC_USE_ROCM"]         = "0"
                _zeus_env["CUDA_VISIBLE_DEVICES"] = str(_zeus_gpu)
                _zeus_env["CUPY_CACHE_DIR"]       = f"/tmp/cupy_cache_zeus_gpu{_zeus_gpu}"
                _zeus_env["PYTHONPATH"]           = _zeus_node.script_path
                self._tcp_transport.allow_worker_id(_zeus_worker_id)   # [PWC-STAGE1 A3]
                try:
                    with open(_zeus_log, "a") as _lf:
                        _zeus_proc = _sp_zeus.Popen(
                            [_zeus_node.python_env, "-m", "persistent.pwc_worker_service"],
                            env=_zeus_env,
                            cwd=_zeus_node.script_path,
                            stdout=_lf, stderr=_lf,
                            start_new_session=True,
                        )
                    self._tcp_local_procs.append(_zeus_proc)    # [PWC-STAGE4 SHUTDOWN]
                    self.logger.info(
                        f"[PWC-TCP] Zeus GPU{_zeus_gpu} local worker launched "
                        f"— PID={_zeus_proc.pid} log={_zeus_log}"
                    )
                    total_launched += 1
                except Exception as _ze:
                    self._tcp_transport.revoke_worker_id(_zeus_worker_id)  # [PWC-STAGE1 A3]
                    self.logger.error(
                        f"[PWC-TCP] Zeus GPU{_zeus_gpu} local worker launch failed: {_ze}"
                    )

        self.logger.info(
            "[PWC-TCP] " + str(total_launched) + " workers launched across all rigs"
            " — waiting for online, then init, then ready"
        )
        # S161 v2: three-phase startup
        # Phase 1: wait for all workers to come online (fast TCP connect)
        _online = self._tcp_wait_online(expected=total_launched, timeout_s=60.0)
        # Phase 2: broadcast init to all online workers (parallel ROCm warmup)
        self._tcp_broadcast_init()
        # Phase 3: wait for workers to become compute-ready
        _ready = self._tcp_wait_ready(expected=total_launched, timeout_s=180.0)
        self.logger.info(
            f"[PWC-TCP] startup complete: {_online} online, {_ready} ready"
        )

    def _tcp_wait_online(self, expected: int, timeout_s: float = 30.0) -> int:
        """
        S161 v2: Wait for expected workers to report online (TCP connected).
        Online = fast TCP connect, no ROCm. Timeout is short (30s).
        Returns count of online workers when deadline reached or expected met.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            count = self._tcp_transport.worker_count()
            if count >= expected:
                self.logger.info(
                    f"[PWC-TCP] all {count}/{expected} workers online — proceeding to init"
                )
                return count
            time.sleep(0.5)
        count = self._tcp_transport.worker_count()
        self.logger.warning(
            f"[PWC-TCP] online timeout: {count}/{expected} workers online after {timeout_s:.0f}s"
        )
        return count

    def _tcp_broadcast_init(self) -> int:
        """
        S161 v2: Broadcast init command to all online workers.
        Workers will import sieve_filter (ROCm warmup) in parallel.
        Returns count of workers init was sent to.
        """
        sent = self._tcp_transport.broadcast_init()
        self.logger.info(
            f"[PWC-TCP] init broadcast to {sent} workers — parallel ROCm warmup starting"
        )
        return sent

    def _tcp_wait_ready(self, expected: int, timeout_s: float = 180.0) -> int:
        """
        S161 v2 + S174: Wait for workers to report ready (compute-ready after ROCm init).
        Ready = dispatch-eligible. Timeout covers parallel ROCm warmup (~90s).

        S174 hard gate: on success, emits READY GATE PASSED and returns count.
        On timeout with count < min_workers, shuts down workers, emits
        READY GATE FAILED, and RAISES RuntimeError BEFORE any job dispatch
        can occur.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            count = self._tcp_transport.ready_count()
            if count >= self.min_workers:
                self.logger.info(
                    f"[PWC-TCP] READY GATE PASSED: {count}/{expected} ready "
                    f"(min_workers={self.min_workers}) — dispatch allowed"
                )
                return count
            time.sleep(0.5)
        # Timeout path: hard-fail per S174 spec
        count = self._tcp_transport.ready_count()
        self.logger.error(
            f"[PWC-TCP] READY GATE FAILED: {count}/{expected} ready "
            f"< min_workers={self.min_workers} — aborting before dispatch"
        )
        # S174 TB hardening: actively clean up workers before raising,
        # so failed gate does not leave TCP workers waiting/reconnecting.
        try:
            self.shutdown()
        except Exception as _shutdown_exc:
            self.logger.warning(
                f"[PWC-TCP] READY GATE FAILED cleanup warning: {_shutdown_exc}"
            )
        raise RuntimeError(
            f"PWC TCP ready gate failed: {count}/{expected} ready "
            f"< min_workers={self.min_workers} after {timeout_s:.0f}s timeout"
        )

    def _ensure_worker_alive(self, handle: WorkerHandle) -> bool:
        """Check worker still alive; respawn if dead."""
        if handle.quarantined:
            return False
        if handle.proc is None or handle.proc.poll() is not None:
            self.logger.warning(f"Worker {handle.node.hostname}:GPU{handle.gpu_id} dead — respawning")
            handle.alive = False
            # [S151] Per-node respawn lock + stagger — prevents SSH hammer when multiple
            # workers die simultaneously on the same rig (PCIe 1x crypto-miner constraint)
            respawn_lock = self._node_respawn_locks.get(handle.node.hostname)
            if respawn_lock:
                with respawn_lock:
                    time.sleep(ROCM_SPAWN_STAGGER_S)
                    success = self._spawn_worker(handle)
            else:
                success = self._spawn_worker(handle)
            if not success:
                handle.quarantined = True
                self.logger.error(f"Respawn failed — {handle.node.hostname}:GPU{handle.gpu_id} quarantined")
            return success
        return True

    def shutdown(self):
        """Send shutdown to all workers and reap processes."""
        self.logger.info("Shutting down persistent workers...")
        if hasattr(self, '_progress_writer') and self._progress_writer:
            try:
                self._progress_writer.finish()
            except Exception:
                pass
        for handle in self.workers:
            if handle.proc and handle.proc.poll() is None:
                try:
                    handle.proc.stdin.write((json.dumps({"cmd": "shutdown"}) + "\n").encode())
                    handle.proc.stdin.flush()
                    handle.proc.wait(timeout=10)
                except Exception:
                    handle.proc.kill()
            handle.alive = False
        # [TCP transport] stop TCP server if running — additive
        if self._tcp_transport is not None:
            try:
                self._tcp_transport.stop()
                self.logger.info("[PWC-TCP] transport stopped")
            except Exception as _e:
                self.logger.warning(f"[PWC-TCP] transport stop error: {_e}")
            self._tcp_transport = None
            _reap = self._tcp_reap_after_shutdown()      # [PWC-STAGE4 SHUTDOWN]
            _bad = {h: st for h, st in _reap.items() if st not in ("exited", "killed")}
            if _bad:
                # [Beta X-PWC-SHUTDOWN R1 Q7] never an unconditional all-clear after an error
                self.logger.error(f"[PWC-TCP] shutdown INCOMPLETE: workers not confirmed gone "
                                  f"on {_bad}")
                return

        self.logger.info("All workers shut down")

    def _tcp_reap_after_shutdown(self) -> Dict[str, str]:
        """[PWC-STAGE4 SHUTDOWN] After a PLANNED transport stop: every worker this
        coordinator launched must be gone. Local VM101 workers are waited for (their
        Popen objects), then killed if still alive at PWC_SHUTDOWN_GRACE_S; each rig
        that got workers runs pwc_shutdown_reap_command over SSH, in parallel. Results
        go to `self.shutdown_reap` and the log (ERROR unless exited/killed). Never
        raises: shutdown() runs in run_trial_persistent's `finally`, and a reap failure
        must not hide the trial's own outcome."""
        import subprocess as _sp
        results: Dict[str, str] = {}
        for proc in list(self._tcp_local_procs):
            key = f"localhost:pid{getattr(proc, 'pid', '?')}"
            try:
                proc.wait(timeout=PWC_SHUTDOWN_GRACE_S)
                results[key] = "exited"
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=10)
                    results[key] = "killed"
                except Exception as _e:
                    results[key] = "error"
                    self.logger.error(f"[PWC-TCP] shutdown reap {key}: {_e}")
        self._tcp_local_procs = []

        def _reap(host, ssh_base):
            try:
                r = _sp.run(ssh_base + [pwc_shutdown_reap_command(token=self._launch_token)],
                            capture_output=True,
                            text=True, timeout=PWC_SHUTDOWN_GRACE_S + 30)
                return host, classify_shutdown_reap(r.returncode, r.stdout), r
            except Exception as _e:
                return host, "error", _e

        targets = dict(self._tcp_reap_targets)
        if targets:
            with ThreadPoolExecutor(max_workers=len(targets)) as ex:
                for host, status, detail in ex.map(lambda kv: _reap(*kv), targets.items()):
                    results[host] = status
                    msg = (f"[PWC-TCP] shutdown reap {host}: {status}"
                           + ("" if status in ("exited", "killed") else f" ({detail!r:.300})"))
                    (self.logger.info if status in ("exited", "killed") else self.logger.error)(msg)
        self._tcp_reap_targets = {}
        self.shutdown_reap = results
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Job dispatch
    # ─────────────────────────────────────────────────────────────────────────
    def _dispatch_to_worker(self, handle: WorkerHandle, job: Dict[str, Any]) -> Dict[str, Any]:
        """Send job JSON to worker stdin, read result from stdout."""
        if not self._ensure_worker_alive(handle):
            return {"status": "error", "message": f"Worker {handle.key} unavailable"}
        # Per-worker lock — each worker is a single process; only one job at a time
        with handle.dispatch_lock:
            sem = self._node_semaphores.get(handle.node.hostname)
            if sem:
                sem.acquire()
            try:
                line = (json.dumps(job) + "\n").encode()
                handle.proc.stdin.write(line)
                handle.proc.stdin.flush()
                result_line = _read_with_timeout(handle.proc.stdout, JOB_TIMEOUT_S)
                if result_line is None:
                    handle.alive = False
                    return {"status": "error", "message": "Worker timeout"}
                if isinstance(result_line, bytes):
                    result_line = result_line.decode("utf-8", errors="replace")
                if not result_line.strip():
                    handle.alive = False
                    return {"status": "error", "message": "Worker returned empty response (pipe closed)"}
                result = json.loads(result_line.strip())
                if result.get("status") == "ok":
                    handle.jobs_completed += 1
                    inner = result.get("result", {})
                    # [S150-slim_v1] Accept both slim parallel-array and legacy dict-list
                    if inner.get("format") == "slim_v1":
                        # Fast path — parallel arrays (TB approved Option A)
                        survivors   = [int(s) for s in inner.get("seeds", [])]
                        match_rates = list(inner.get("match_rates", []))
                        n = len(survivors)
                        # [S152-IPC] Confirm slim_v1 fast path active
                        _gpu_tag = f"{job.get('hostname','?')}:GPU{job.get('gpu_id','?')}"
                        self.logger.debug(f"[slim_v1] {_gpu_tag} chunk → {n} survivors")
                        # TB ruling: strategy_ids+skip_sequences required for hybrid
                        # [S155] Fix: job payload sends "hybrid": bool — not "prng_type"/"skip_mode".
                        # Both those keys are absent from PWC job dicts. "hybrid" is the correct key,
                        # set at dispatch: "hybrid": is_hybrid (from "'_hybrid' in prng_type").
                        _is_hybrid_job = bool(job.get("hybrid", False))
                        if _is_hybrid_job:
                            if "strategy_ids" not in inner or "skip_sequences" not in inner:
                                handle.alive = False
                                return {"status": "error", "message": "slim_v1 hybrid payload missing strategy_ids/skip_sequences"}
                            strat_ids = list(inner["strategy_ids"])
                            skip_seqs = list(inner["skip_sequences"])
                        else:
                            strat_ids = [0] * n
                            skip_seqs = [[] for _ in survivors]
                        # TB guardrail: all arrays must match len(seeds)
                        if len(match_rates) != n or len(strat_ids) != n or len(skip_seqs) != n:
                            handle.alive = False
                            return {"status": "error", "message": f"slim_v1 length mismatch: seeds={n} match_rates={len(match_rates)} strat_ids={len(strat_ids)} skip_seqs={len(skip_seqs)}"}
                    else:
                        # Legacy path — list of dicts (kept for rollout safety)
                        # [S152-IPC] WARN: expected slim_v1 from updated workers
                        _gpu_tag = f"{job.get('hostname','?')}:GPU{job.get('gpu_id','?')}"
                        self.logger.warning(f"[legacy-ipc] {_gpu_tag} — worker sent legacy dict-list format (expected slim_v1)")
                        raw_survivors = inner.get("survivors", [])
                        survivors   = [s["seed"]       if isinstance(s, dict) else int(s) for s in raw_survivors]
                        match_rates = [s["match_rate"] if isinstance(s, dict) else 0.5     for s in raw_survivors]
                        skip_seqs   = [s.get("skip_sequence", []) if isinstance(s, dict) else [] for s in raw_survivors]
                        strat_ids   = [s.get("strategy_id",    0) if isinstance(s, dict) else 0  for s in raw_survivors]
                    return {
                        "status":         "ok",
                        "job_id":         result.get("job_id", job.get("job_id")),
                        "survivors":      survivors,
                        "match_rates":    match_rates,
                        "skip_sequences": skip_seqs,
                        "strategy_ids":   strat_ids,
                        # [PWC-BACKUP S1/S6] what the worker says it swept / used;
                        # None when it does not say (sieve_gpu_worker.py reports
                        # seed_range but no threshold)
                        "worker_seed_range":          inner.get("seed_range"),
                        "worker_min_match_threshold": inner.get("min_match_threshold"),
                    }
                else:
                    handle.jobs_failed += 1
                return result
            except Exception as e:
                handle.alive = False
                return {"status": "error", "message": str(e)}
            finally:
                if sem:
                    sem.release()

    def _dispatch_to_tcp(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """
        [TCP transport] Submit job to TCPWorkerTransport and block for result.
        Converts normalized transport result back into the same dict contract
        as _dispatch_to_worker() so all downstream code is transport-blind.
        TB ruling: result normalizer is the contract wall.
        """
        import uuid as _uuid
        if "job_id" not in job:
            job = dict(job)
            job["job_id"] = str(_uuid.uuid4())

        # S161: wait for at least one TCP worker to connect before dispatching
        _wait_start = time.time()
        while self._tcp_transport.worker_count() == 0:
            if time.time() - _wait_start > 60.0:
                return {"status": "error", "message": "TCP worker connect timeout (60s)"}
            time.sleep(0.5)
        self._tcp_transport.submit_job(job)

        # Block for result correlated by job_id — TB fix 4
        result = self._tcp_transport.recv_result(
            timeout_s=JOB_TIMEOUT_S,
            job_id=job["job_id"],
        )
        if result is None:
            return {"status": "error", "message": "TCP worker timeout"}

        # Normalize to PWC result contract
        if result.get("status") == "error":
            return result

        inner = result.get("result", {})
        payload = inner.get("payload", inner)

        # Accept slim_v1 parallel arrays or legacy dict-list — same as _dispatch_to_worker
        if isinstance(payload, dict) and payload.get("format") == "slim_v1":
            survivors   = [int(s) for s in payload.get("seeds", [])]
            match_rates = list(payload.get("match_rates", []))
            n = len(survivors)
            _is_hybrid_job = bool(job.get("hybrid", False))
            if _is_hybrid_job:
                strat_ids = list(payload.get("strategy_ids", [0]*n))
                skip_seqs = list(payload.get("skip_sequences", [[]]*n))
            else:
                strat_ids = [0] * n
                skip_seqs = [[] for _ in survivors]
        else:
            raw_surv    = payload.get("survivors", [])
            survivors   = [s["seed"]                  if isinstance(s, dict) else int(s) for s in raw_surv]
            match_rates = [s.get("match_rate", 0.5)   if isinstance(s, dict) else 0.5    for s in raw_surv]
            # [PWC-STAGE1 B6] sieve_filter.py's legacy survivor record names this
            # field `skip_pattern` (sieve_filter.py run_hybrid_sieve); reading only
            # `skip_sequence` returned [] for every survivor. Both names are read.
            skip_seqs   = [(s.get("skip_pattern", s.get("skip_sequence", [])) if isinstance(s, dict) else [])
                           for s in raw_surv]
            strat_ids   = [s.get("strategy_id", 0)    if isinstance(s, dict) else 0       for s in raw_surv]

        # [PWC-BACKUP STAGE 2b] the worker must echo the anchor and phase it ran
        _echo_err = pwc_anchor_echo_error(payload, job)
        if _echo_err is not None:
            return {"status": "error", "message": f"[PWC-BACKUP STAGE 2b] {_echo_err}"}

        return {
            "status":         "ok",
            "job_id":         job["job_id"],
            "survivors":      survivors,
            "match_rates":    match_rates,
            "skip_sequences": skip_seqs,
            "strategy_ids":   strat_ids,
            # S161 dashboard: preserve worker identity for ProgressWriter
            "worker_id":      result.get("worker_id", ""),
            "hostname":       result.get("hostname", ""),
            "gpu_id":         result.get("gpu_id", 0),
            # [PWC-BACKUP S1/S6] sieve_filter.execute_sieve_job echoes the range it
            # swept and the min_match_threshold it passed to run_sieve
            # (sieve_filter.py final_result); None when absent, never guessed.
            "worker_seed_range":          payload.get("seed_range") if isinstance(payload, dict) else None,
            "worker_min_match_threshold": payload.get("min_match_threshold") if isinstance(payload, dict) else None,
            # [PWC-STAGE2 ruling 1] the threshold sieve_filter passed to
            # run_hybrid_sieve for this family, as it reports it (4 places,
            # sieve_filter.py:597-599); None when absent, never guessed.
            "worker_hybrid_threshold":    _pwc_hybrid_echo(payload, job),
        }

    def _dispatch_local_sieve(self, job: Dict[str, Any], node: WorkerNode) -> Dict[str, Any]:
        """
        Zeus local path — run sieve_filter.py as subprocess (already fast, no persistent
        worker needed). Mirrors coordinator.execute_local_job() behavior.
        """
        import tempfile
        payload_file = tempfile.mktemp(suffix=".json", dir=node.script_path)
        result_file  = payload_file.replace(".json", "_result.json")
        try:
            with open(payload_file, "w") as f:
                json.dump(job, f)
            cmd = [
                node.python_env, "-u", "sieve_filter.py",
                "--job-file", os.path.basename(payload_file),
                "--gpu-id", str(job.get("gpu_id", 0)),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(job.get("gpu_id", 0))
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=JOB_TIMEOUT_S,
                cwd=node.script_path, env=env
            )
            # Parse JSON from stdout
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.startswith("{"):
                    try:
                        raw = json.loads(line)
                        # sieve_filter.py emits {"success": true, "survivors": [...]}
                        # Normalize to flat format (same as _dispatch_to_worker)
                        is_ok = (raw.get("status") == "ok") or (raw.get("success") is True)
                        if is_ok:
                            inner    = raw.get("result", raw)
                            raw_surv = inner.get("survivors", [])
                            if raw_surv and isinstance(raw_surv[0], dict):
                                survivors   = [s["seed"]                  for s in raw_surv]
                                match_rates = [s.get("match_rate", 0.5)   for s in raw_surv]
                                skip_seqs   = [s.get("skip_sequence", []) for s in raw_surv]
                                strat_ids   = [s.get("strategy_id", 0)    for s in raw_surv]
                            else:
                                survivors   = [int(s) for s in raw_surv]
                                match_rates = inner.get("match_rates", [0.5]*len(survivors))
                                skip_seqs   = inner.get("skip_sequences", [])
                                strat_ids   = inner.get("strategy_ids",   [])
                            # [PWC-BACKUP STAGE 2b] the anchor and phase it ran
                            _echo_err = pwc_anchor_echo_error(inner, job)
                            if _echo_err is not None:
                                return {"status": "error",
                                        "message": f"[PWC-BACKUP STAGE 2b] {_echo_err}"}
                            return {
                                "status":         "ok",
                                "job_id":         raw.get("job_id", "local"),
                                "survivors":      survivors,
                                "match_rates":    match_rates,
                                "skip_sequences": skip_seqs,
                                "strategy_ids":   strat_ids,
                                # [PWC-BACKUP S1/S6]
                                "worker_seed_range":          inner.get("seed_range"),
                                "worker_min_match_threshold": inner.get("min_match_threshold"),
                            }
                        return raw
                    except Exception:
                        pass
            return {"status": "error", "message": result.stderr[:500]}
        except subprocess.TimeoutExpired:
            return {"status": "error", "message": "Local sieve timeout"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
        finally:
            for f in (payload_file, result_file):
                try:
                    os.unlink(f)
                except Exception:
                    pass

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry point — drop-in for execute_distributed_analysis()
    # ─────────────────────────────────────────────────────────────────────────
    def run_sieve_pass(self,
                       prng_type: str,
                       residues: List[int],
                       total_seeds: int,
                       threshold: float,
                       window_size: int,
                       output_file: str,
                       dataset_path: str = "",
                       strategies: Optional[List[Dict]] = None,
                       phase2_threshold: Optional[float] = None,   # [PWC-STAGE2] no 0.5 default
                       target_file: str = "",
                       offset: int = 0,
                       sessions: Optional[List[str]] = None,
                       skip_range: Optional[List[int]] = None,
                       seed_start: int = 0) -> Dict[str, Any]:
        """
        Run a full distributed sieve pass (forward OR reverse, constant OR hybrid).
        Returns a result dict compatible with extract_survivor_records() in
        window_optimizer_integration_final.py.

        Parameters match coordinator.execute_distributed_analysis() semantics:
          prng_type        — e.g. "java_lcg", "java_lcg_reverse", "java_lcg_hybrid"
          residues         — list of draw values for the window
          total_seeds      — total seed space to search
          threshold        — match threshold for survivor filtering
          window_size      — number of draws in window
          output_file      — path to write result JSON (mirrors existing paths)
          strategies       — None for constant skip; list of strategy dicts for hybrid
                             (auto-loaded from hybrid_strategy if None and prng is hybrid)
          phase2_threshold — [PWC-STAGE2 ruling 1] the threshold of a hybrid
                             (variable-skip) pass. REQUIRED for a hybrid pass and
                             must equal `threshold` (RANGE-MINER writes the one
                             directional value as both keys). For a constant pass
                             it may be omitted; the job then carries `threshold`.
          seed_start       — [PWC-BACKUP S2] first seed; the pass partitions
                             [seed_start, seed_start + total_seeds). Default 0
                             gives exactly the chunks this method cut before.
        """
        is_hybrid = "_hybrid" in prng_type

        # [PWC-STAGE2 ruling 1] The quarantine raise that stood here is replaced by
        # the threshold contract, checked before strategy loading, chunking or any
        # dispatch. Every caller crosses it, not just run_trial_persistent.
        if is_hybrid:
            if phase2_threshold is None:
                raise PwcSettingRefused(
                    f"[{prng_type}] a variable-skip pass needs phase2_threshold (the "
                    f"trial's directional threshold); there is no default (the old 0.5 "
                    f"default filtered at 0.5 whatever the trial asked for)")
            if not _thr_equal(phase2_threshold, threshold):
                raise PwcSettingRefused(
                    f"[{prng_type}] phase2_threshold={phase2_threshold!r} != threshold="
                    f"{threshold!r}; RANGE-MINER runs one kernel threshold and pins the "
                    f"two keys equal (build_stripe_assign_payload)")
            if self._tcp_transport is None:
                raise PwcSettingRefused(
                    f"[{prng_type}] variable-skip passes need the TCP transport: the SSH "
                    f"route (sieve_gpu_worker.py) reports no threshold, so the contract "
                    f"cannot be checked (pwc_transport={self.pwc_transport!r})")
        _job_phase2 = float(threshold) if phase2_threshold is None else phase2_threshold

        # [PWC-STAGE2] window offset other than 0 refused (validate_pwc_window_offset)
        # [PWC-BACKUP STAGE 2b] now only on the SSH route; the TCP and local routes
        # carry it as window_anchor (validate_pwc_window_anchor).
        _window_anchor = validate_pwc_window_anchor(
            offset, ssh_route=self._tcp_transport is None)

        # [PWC-BACKUP S2] the interval, checked before anything is chunked
        _req_start, _req_end = validate_pwc_seed_interval(seed_start, total_seeds)

        # [PWC-BACKUP S4] dispatch_chunk below falls back to both sessions and to
        # skip_range [0, 147] when these are None. Those are silent defaults of
        # the trial's own configuration, so a pass now refuses without them.
        if sessions is None or skip_range is None:
            raise PwcSettingRefused(
                f"[{prng_type}] run_sieve_pass needs explicit sessions and skip_range "
                f"(got sessions={sessions!r}, skip_range={skip_range!r}); PWC no "
                f"longer substitutes ['midday', 'evening'] / [0, 147]")

        # Auto-load strategies for hybrid if not provided
        if is_hybrid and strategies is None:
            try:
                from hybrid_strategy import get_all_strategies
                raw = get_all_strategies()
                strategies = [
                    # Send full strategy dict — sieve_filter.py needs all StrategyConfig fields
                    s.to_dict() if hasattr(s, 'to_dict') else s
                    for s in raw
                ]
                self.logger.info(f"Loaded {len(strategies)} strategies for hybrid sieve")
            except ImportError as _e:
                # [PWC-STAGE2] was: warn and send [], which sieve_filter then
                # replaced with its own choice. RANGE-MINER loads the same five.
                raise PwcSettingRefused(
                    f"[{prng_type}] hybrid_strategy.get_all_strategies unavailable "
                    f"({_e}); PWC does not substitute a strategy set") from _e

        # Build chunk list — divide total seeds across all available workers
        # S161 v2: workers already online+ready from startup() three-phase init
        # S174 defense-in-depth: ready gate already enforced in _tcp_wait_ready,
        # but verify once more at dispatch site to block any race condition.
        if self._tcp_transport is not None:
            _ready = self._tcp_transport.ready_count()
            if _ready < self.min_workers:
                self.logger.error(
                    f"[PWC-TCP] DISPATCH BLOCKED: {_ready} ready "
                    f"< min_workers={self.min_workers} — refusing job_assign"
                )
                raise RuntimeError(
                    f"PWC dispatch blocked: ready={_ready} "
                    f"< min_workers={self.min_workers}"
                )
            self.logger.info(
                f"[PWC-TCP] dispatch confirmed: {_ready} ready workers "
                f"(min_workers={self.min_workers})"
            )
        all_workers = self._get_available_workers()
        num_workers = max(1, len(all_workers))
        ideal_chunk = max(1, total_seeds // num_workers)

        # Cap chunk size at per-GPU OOM ceiling
        # Use AMD cap as conservative ceiling (all remote workers are AMD)
        chunk_cap   = self.seed_cap_amd
        chunk_size  = min(ideal_chunk, chunk_cap)

        # [PWC-BACKUP S5] exact values, at the last point before dispatch. The
        # startup check already judged upper bounds of both, so this cannot fire
        # for a run that started; it is here so no other caller of this method can
        # dispatch outside the envelope either.
        _wpr = max(self._effective_pool_by_host.values(), default=0)
        assert_pwc_within_stable_envelope(
            _wpr, chunk_size, where=f"run_sieve_pass[{prng_type}], before dispatch")

        # [PWC-BACKUP S2] partition [seed_start, seed_start + total_seeds).
        # chunk_size is unchanged (it depends on total_seeds only), so at
        # seed_start = 0 the chunk list is identical to the one cut before.
        chunks = []
        seed   = _req_start
        while seed < _req_end:
            end = min(seed + chunk_size, _req_end)
            chunks.append((seed, end))
            seed = end

        self.logger.info(
            f"[{prng_type}] {total_seeds:,} seeds → {len(chunks)} chunks "
            f"({chunk_size:,}/chunk) across {num_workers} workers"
        )
        if self._progress_writer:
            try:
                step_name = f"{'Reverse' if 'reverse' in prng_type else 'Forward'} Sieve ({prng_type})"
                self._progress_writer.update_step(step_name, total_seeds=total_seeds)
            except Exception:
                pass

        # [PWC-STAGE2 R2, Beta X-PWC-STAGE2 R1 finding 2] Job ids are unique to THIS
        # pass. They were `sieve_000`, `sieve_001`, ... in every pass, so a worker still
        # finishing a timed-out job of one pass could have its late result delivered
        # as the same-numbered job of the next pass (and pass its range and
        # threshold checks whenever the two passes' thresholds are equal). The
        # transport also accepts a result only for the active lease
        # (persistent/pwc_transport_tcp.py `_take_active_lease`).
        import uuid as _uuid_pass
        _pass_id = _uuid_pass.uuid4().hex[:12]

        # Dispatch all chunks in parallel threads
        results_by_chunk: Dict[int, Dict] = {}
        lock = threading.Lock()

        def dispatch_chunk(idx: int, seed_start: int, seed_end: int,
                           worker_handle_or_node):
            # Job dict matches coordinator.py residue_sieve format exactly
            # (field names verified against sieve_gpu_worker.py job.get() calls)
            _sessions   = sessions   if sessions   is not None else ["midday", "evening"]
            _skip_range = skip_range if skip_range is not None else [0, 147]
            # prng_families: strip _reverse/_hybrid suffixes — worker handles variants
            _base_prng  = prng_type.replace("_reverse", "").replace("_hybrid", "")
            job = {
                "job_id":               f"sieve_{_pass_id}_{idx:03d}",   # [PWC-STAGE2 R2]
                "search_type":          "residue_sieve",
                "dataset_path":         dataset_path or target_file,
                "seed_start":           seed_start,
                "seed_end":             seed_end,
                "window_size":          window_size,
                "min_match_threshold":  threshold,
                "skip_range":           _skip_range,
                "offset":               offset,
                "sessions":             _sessions,
                "prng_families":        [prng_type],
                "strategies":           strategies if is_hybrid else None,
                "hybrid":               is_hybrid,
                "phase2_threshold":     _job_phase2,   # [PWC-STAGE2] == threshold
            }

            def _run_once(wh):
                # [TCP transport] dispatch bridge — additive, SSH path unchanged
                if self._tcp_transport is not None:  # [S166] route ALL workers via TCP incl Zeus
                    return self._dispatch_to_tcp(pwc_new_key_job(job, _window_anchor))  # [STAGE 2b]
                if isinstance(wh, WorkerHandle):
                    job["gpu_id"] = wh.gpu_id
                    return self._dispatch_to_worker(wh, job)
                else:
                    # [S163] Slot-based GPU binding — acquire a specific GPU slot from the
                    # pool. GPU ID comes from the slot, not from chunk index. This ensures
                    # two concurrent Zeus local subprocesses always get distinct gpu_ids and
                    # thus distinct CUDA_VISIBLE_DEVICES. chunk index is not a safe proxy.
                    local_gpu_id = self._localhost_gpu_slots.get(block=True)
                    job["gpu_id"] = local_gpu_id
                    self.logger.info(
                        f"[S163-ZEUS] chunk={idx} gpu_slot={local_gpu_id} "
                        f"CUDA_VISIBLE_DEVICES={local_gpu_id}"
                    )
                    try:
                        return self._dispatch_local_sieve(
                            pwc_new_key_job(job, _window_anchor), wh)  # [PWC-BACKUP STAGE 2b]
                    finally:
                        self._localhost_gpu_slots.put(local_gpu_id)

            t0 = time.time()
            res = _run_once(worker_handle_or_node)
            elapsed = time.time() - t0

            # One retry on transient pipe/empty-response failures
            if res.get("status") != "ok":
                err = res.get("message", "")
                if "empty response" in err or "pipe" in err.lower() or "timeout" in err.lower():
                    self.logger.warning(f"  Chunk {idx} transient failure ({err}) — retrying once")
                    time.sleep(1)
                    t0 = time.time()
                    res = _run_once(worker_handle_or_node)
                    elapsed = time.time() - t0

            # [PWC-BACKUP S6] record the threshold this chunk's job dict CARRIED,
            # read off the job itself rather than restated from the argument
            res = dict(res)
            res["pwc_payload_min_match_threshold"] = job["min_match_threshold"]
            res["pwc_payload_phase2_threshold"] = job["phase2_threshold"]   # [PWC-STAGE2]

            with lock:
                results_by_chunk[idx] = res
            status = "✅" if res.get("status") == "ok" else "❌"
            survivors = len(res.get("survivors", []))
            seeds_in_chunk = seed_end - seed_start
            self.logger.info(f"  {status} Chunk {idx}: {seeds_in_chunk:,} seeds → {survivors:,} survivors")

            # Log to ProgressWriter for web dashboard (mirrors coordinator.py line 1611)
            if self._progress_writer and res.get("status") == "ok" and elapsed > 0:
                try:
                    if worker_handle_or_node is None:
                        # S161 v2: TCP worker — extract from result payload
                        _payload = res.get("result", {}).get("payload", res.get("result", {}))
                        gpu_id   = _payload.get("gpu_id", 0)
                        gpu_type = "RX 6600"
                        # S161 dashboard fix: extract IP from worker_id
                        # worker_id format: "192_168_3_120_gpu0" → "192.168.3.120"
                        # worker_id is in res["worker_id"] from normalizer
                        _worker_id = res.get("worker_id", "")
                        try:
                            # Strip _gpuN suffix, replace _ with . to get IP
                            _ip_part = _worker_id.rsplit("_gpu", 1)[0].replace("_", ".")
                            hostname = _ip_part if _ip_part else "tcp-worker"
                        except Exception:
                            hostname = "tcp-worker"
                        # Get correct gpu_type from matching node
                        for _n in self.nodes:
                            if _n.hostname == hostname:
                                gpu_type = _n.gpu_type
                                break
                    elif isinstance(worker_handle_or_node, WorkerHandle):
                        hostname = worker_handle_or_node.node.hostname
                        gpu_type = worker_handle_or_node.node.gpu_type
                        gpu_id   = worker_handle_or_node.gpu_id
                    else:
                        hostname = worker_handle_or_node.hostname
                        gpu_type = worker_handle_or_node.gpu_type
                        gpu_id   = 0
                    self._progress_writer.log_gpu_result(hostname, gpu_id, gpu_type, seeds_in_chunk, elapsed, success=True)
                except Exception:
                    pass

        # [S158B-v3] Bounded dispatch — TB-approved hardening.
        # Replaces unbounded thread-per-chunk fan-out (537 simultaneous threads)
        # with ThreadPoolExecutor capped at min(num_workers, len(chunks)).
        #
        # Key design decisions:
        # 1. No context-manager (with) — avoids shutdown(wait=True) hang trap.
        #    Executor is explicitly shut down with wait=False, cancel_futures=True
        #    on timeout. Running futures are bounded: dispatch_chunk always returns
        #    within JOB_TIMEOUT_S because _read_with_timeout uses t.join(timeout).
        # 2. Progress-aware wait() loop — breaks if no chunk completes within
        #    PASS_PROGRESS_TIMEOUT_S. Prevents permanent hang on stuck chunks.
        # 3. Empty-chunks guard — ThreadPoolExecutor(max_workers=0) is invalid.
        # 4. Explicit failure propagation — timed-out chunks written to
        #    results_by_chunk as error entries, not just counted.

        # [PWC-BACKUP S1] kept at pass scope so the accounting below can tell a
        # chunk that was dispatched from one cancelled before it ever started
        _future_to_idx: Dict[Any, int] = {}
        _pass_timed_out = False

        # Guard: nothing to dispatch
        if not chunks:
            self.logger.info("[S158B] No chunks to dispatch — skipping executor")
        else:
            # Progress timeout: 2× JOB_TIMEOUT_S is conservative ceiling.
            # If no chunk completes within this window, declare pass failed.
            _PASS_PROGRESS_TIMEOUT_S = JOB_TIMEOUT_S * 2  # 1200s

            _pool_size = min(num_workers, len(chunks))
            _pass_timed_out = False
            _peak_threads = threading.active_count()

            self.logger.info(
                f"[S158B] Bounded dispatch: {len(chunks)} chunks → "
                f"pool_size={_pool_size} (num_workers={num_workers})"
            )

            _executor = ThreadPoolExecutor(max_workers=_pool_size)
            try:
                _future_to_idx = {
                    _executor.submit(
                        dispatch_chunk, i, s_start, s_end,
                        all_workers[i % num_workers]
                    ): i
                    for i, (s_start, s_end) in enumerate(chunks)
                }
                _pending = set(_future_to_idx.keys())

                while _pending:
                    _peak_threads = max(_peak_threads, threading.active_count())

                    _done, _pending = _cf_wait(
                        _pending,
                        timeout=_PASS_PROGRESS_TIMEOUT_S,
                        return_when=_CF_FIRST_COMPLETED
                    )

                    if not _done:
                        # No chunk completed — stuck pass detected
                        _pass_timed_out = True
                        self.logger.error(
                            f"[S158B] Pass progress timeout after "
                            f"{_PASS_PROGRESS_TIMEOUT_S}s — "
                            f"{len(_pending)} chunks still pending. "
                            f"Aborting pass."
                        )
                        # Write explicit error entries for stuck chunks
                        with lock:
                            for _f in _pending:
                                _stuck_idx = _future_to_idx[_f]
                                results_by_chunk[_stuck_idx] = {
                                    "status": "error",
                                    "message": "Pass progress timeout — chunk aborted"
                                }
                                self.logger.error(
                                    f"  [S158B] Chunk {_stuck_idx} aborted (progress timeout)"
                                )
                        break

                    # Process completed futures
                    for _future in _done:
                        _idx = _future_to_idx[_future]
                        try:
                            _future.result()
                        except Exception as _e:
                            self.logger.error(
                                f"[S158B] Chunk {_idx} future raised: {_e}"
                            )
                            # Ensure error is recorded explicitly
                            with lock:
                                if _idx not in results_by_chunk:
                                    results_by_chunk[_idx] = {
                                        "status": "error",
                                        "message": str(_e)
                                    }

            finally:
                # Explicit shutdown — do not wait for stuck running threads.
                # cancel_futures=True prevents pending futures from starting.
                # Running futures finish within JOB_TIMEOUT_S (bounded by design).
                _executor.shutdown(wait=False, cancel_futures=True)
                self.logger.info(
                    f"[S158B] Executor shut down — "
                    f"peak_threads={_peak_threads} "
                    f"timed_out={_pass_timed_out}"
                )

        # [PWC-BACKUP S1] One snapshot, taken under the lock: a chunk thread still
        # running after a pass abort must not change the record between the
        # count and the aggregation.
        with lock:
            _snapshot = dict(results_by_chunk)
        _cancelled = {i for f, i in _future_to_idx.items() if f.cancelled()}

        # Aggregate results
        all_survivors   = []
        all_match_rates = []
        all_skip_seqs   = []
        all_strat_ids   = []
        failed_chunks   = 0
        _outcomes: Dict[str, int] = {"completed": 0, "failed": 0, "timed_out": 0,
                                     "not_dispatched": 0, "missing": 0}
        _done_ivs: List[Tuple[int, int]] = []
        _incomplete_rows: List[Dict[str, Any]] = []
        _range_unreported = 0
        _payload_thr: List[Any] = []
        _payload_p2: List[Any] = []          # [PWC-STAGE2]
        _effective_thr: List[Any] = []
        _effective_unreported = 0
        _hyb_effective: List[Any] = []       # [PWC-STAGE2]
        _hyb_unreported = 0

        for i in range(len(chunks)):
            res = _snapshot.get(i)
            outcome, why = _chunk_outcome(res, chunks[i], i in _cancelled)
            _outcomes[outcome] += 1
            if res is not None and "pwc_payload_min_match_threshold" in res:
                _payload_thr.append(res["pwc_payload_min_match_threshold"])
            if res is not None and "pwc_payload_phase2_threshold" in res:
                _payload_p2.append(res["pwc_payload_phase2_threshold"])
            if outcome == "completed":
                _done_ivs.append(chunks[i])
                if res.get("worker_seed_range") is None:
                    _range_unreported += 1
                if res.get("worker_min_match_threshold") is None:
                    _effective_unreported += 1
                else:
                    _effective_thr.append(res["worker_min_match_threshold"])
                if res.get("worker_hybrid_threshold") is None:     # [PWC-STAGE2]
                    _hyb_unreported += 1
                else:
                    _hyb_effective.append(res["worker_hybrid_threshold"])
                all_survivors   .extend(res.get("survivors",    []))
                all_match_rates .extend(res.get("match_rates",  []))
                all_skip_seqs   .extend(res.get("skip_sequences", []))
                all_strat_ids   .extend(res.get("strategy_ids",  []))
            else:
                failed_chunks += 1
                res = res or {}
                err_msg = why or res.get('message') or res.get('error', 'unknown')
                tb      = res.get('traceback', '')
                self.logger.warning(f"Chunk {i} {outcome}: {err_msg}")
                if tb:
                    self.logger.warning(f"  Worker traceback:\n{tb}")
                if len(_incomplete_rows) < _INCOMPLETE_DETAIL_CAP:
                    _incomplete_rows.append({"idx": i, "interval": list(chunks[i]),
                                             "outcome": outcome, "reason": err_msg})

        if failed_chunks:
            self.logger.warning(f"{failed_chunks}/{len(chunks)} chunks failed for {prng_type}")

        _merged = _merge_intervals(_done_ivs)
        _pass_complete = (
            _outcomes["completed"] == len(chunks)
            and (_merged == [[_req_start, _req_end]] or _req_start == _req_end))
        assert sum(_outcomes.values()) == len(chunks), (
            f"PWC accounting lost a chunk: {_outcomes} vs {len(chunks)} planned")
        _accounting = {
            "schema":                PWC_PASS_ACCOUNTING_SCHEMA,
            "prng_type":             prng_type,
            "requested_interval":    [_req_start, _req_end],
            "chunk_size":            chunk_size,
            "workers_per_rig":       _wpr,
            "planned":               len(chunks),
            "dispatched":            len(chunks) - _outcomes["not_dispatched"] - _outcomes["missing"],
            **_outcomes,
            "seed_range_unreported": _range_unreported,
            "pass_progress_timeout": bool(_pass_timed_out),
            "completed_intervals":   _merged,
            "incomplete_chunks":     _incomplete_rows,
            "incomplete_chunks_truncated": failed_chunks > len(_incomplete_rows),
            "complete":              bool(_pass_complete),
            # [PWC-STAGE2] which route ran the pass. Credit is Stage 3; this lets
            # the Stage 3 evidence builder refuse a pass that was not on TCP.
            "transport":             "tcp" if self._tcp_transport is not None else "ssh",
        }
        if not _pass_complete:
            self.logger.error(
                f"[PWC-BACKUP S1] PASS INCOMPLETE [{prng_type}] "
                f"{_outcomes['completed']}/{len(chunks)} chunks completed "
                f"(failed={_outcomes['failed']} timed_out={_outcomes['timed_out']} "
                f"not_dispatched={_outcomes['not_dispatched']} missing={_outcomes['missing']}) "
                f"over [{_req_start}, {_req_end}) — the result is marked incomplete")

        # [PWC-BACKUP S6] constant passes; [PWC-STAGE2 ruling 1] hybrid passes below
        if is_hybrid:
            _thr_rec = pwc_hybrid_threshold_record(
                prng_type, threshold, _payload_thr, _payload_p2, _hyb_effective,
                _hyb_unreported)
            _accounting["threshold"] = _thr_rec
            assert_pwc_hybrid_threshold_contract(_thr_rec)
        else:
            _thr_rec = pwc_threshold_record(prng_type, threshold, _payload_thr,
                                            _effective_thr, _effective_unreported)
            _accounting["threshold"] = _thr_rec
            assert_pwc_threshold_contract(_thr_rec)
            if not _thr_rec["verified"]:
                self.logger.warning(
                    f"[PWC-BACKUP S6] [{prng_type}] effective threshold not reported by "
                    f"{_effective_unreported} completed chunk(s); requested/payload agree "
                    f"({_thr_rec['requested']!r}) but the worker-side value is unverified")

        if self._progress_writer:
            try:
                self._progress_writer.update_progress(
                    jobs_done=len(chunks) - failed_chunks,
                    chunks_total=len(chunks)
                )
            except Exception:
                pass

        # Build result dict compatible with extract_survivor_records()
        result = {
            "survivors":         all_survivors,
            "match_rates":       all_match_rates,
            "skip_sequences":    all_skip_seqs,
            "strategy_ids":      all_strat_ids,
            "total_tested":      total_seeds,
            "survivor_count":    len(all_survivors),
            "prng_type":         prng_type,
            "threshold":         threshold,
            "failed_chunks":     failed_chunks,
            "total_chunks":      len(chunks),
            # [PWC-BACKUP S1/S2] added fields; none removed
            "seed_start":          _req_start,
            "seed_end":            _req_end,
            "pass_complete":       bool(_pass_complete),
            "pwc_pass_accounting": _accounting,
        }

        # Save to output_file (mirrors coordinator behavior)
        # [S163] Skip full JSON write when survivor count is large — json.dump of
        # millions of records causes multi-hour serialization hang. Write a lightweight
        # summary instead. The result dict in memory is used for intersection directly.
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        _JSON_WRITE_LIMIT = 100_000  # skip full write above this threshold
        try:
            if len(all_survivors) <= _JSON_WRITE_LIMIT:
                with open(output_file, "w") as f:
                    json.dump(result, f)
                self.logger.info(f"Results saved: {output_file} ({len(all_survivors):,} survivors)")
            else:
                # Write lightweight summary only — avoids multi-GB JSON serialization hang
                summary = {
                    "survivor_count": len(all_survivors),
                    "total_tested":   total_seeds,
                    "prng_type":      prng_type,
                    "threshold":      threshold,
                    "failed_chunks":  failed_chunks,
                    "total_chunks":   len(chunks),
                    "pass_complete":  bool(_pass_complete),          # [PWC-BACKUP S1]
                    "pwc_pass_accounting": _accounting,              # [PWC-BACKUP S1]
                    "note":           "Full survivors omitted (count > 100K) — see NPZ accumulator",
                }
                with open(output_file, "w") as f:
                    json.dump(summary, f)
                self.logger.info(
                    f"Results summary saved: {output_file} "
                    f"({len(all_survivors):,} survivors — full JSON skipped, count > {_JSON_WRITE_LIMIT:,})"
                )
        except Exception as e:
            self.logger.error(f"Failed to save {output_file}: {e}")

        return result

    def _get_available_workers(self) -> List:
        """
        Returns list of available dispatch targets — mix of WorkerHandle (AMD)
        and WorkerNode (Zeus local). Only alive, non-quarantined workers included.
        In TCP mode, derives count from connected TCP workers — TB fix 3.
        """
        # [TCP transport] use TCP worker count as pool size — TB fix 3
        if self._tcp_transport is not None:
            tcp_count = self._tcp_transport.ready_count()  # S161 v2: only ready workers
            # TCP workers as synthetic placeholders + Zeus local nodes
            available = [None] * tcp_count
            for node in self.nodes:
                if self._is_localhost(node.hostname):
                    available.append(node)
            return available

        available = []
        # AMD persistent workers
        for w in self.workers:
            if not w.quarantined and w.alive:
                available.append(w)
        # Zeus local nodes
        for node in self.nodes:
            if self._is_localhost(node.hostname):
                available.append(node)
        return available


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────
def _read_with_timeout(stream, timeout_s: float) -> Optional[str]:
    """Read one line from stream with timeout. Returns None on timeout."""
    result = [None]
    def _reader():
        try:
            result[0] = stream.readline()
        except Exception:
            pass
    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    return result[0] if result[0] else None


# ─────────────────────────────────────────────────────────────────────────────
# Integration shim for window_optimizer_integration_final.py
# ─────────────────────────────────────────────────────────────────────────────
def run_trial_persistent(coordinator_cfg: str,
                         config,           # WindowConfig from window_optimizer
                         trial_number: int,
                         prng_base: str,
                         residues: List[int],
                         total_seeds: int,
                         forward_threshold: float,
                         reverse_threshold: float,
                         test_both_modes: bool,
                         dataset_path: str = "",
                         worker_pool_size: Optional[int] = None,
                         seed_cap_nvidia: Optional[int] = None,
                         seed_cap_amd:   Optional[int] = None,
                         pwc_transport: Optional[str] = None,
                         pwc_host: str = "0.0.0.0",
                         pwc_port: int = 5600,
                         node_allowlist=None,
                         min_workers: int = 1,                   # [S163-KARG-PWC] + [S174 ready gate]
                         seed_start: int = 0,                    # [PWC-BACKUP S2]
                         pwc_coordinator_address: Optional[str] = None,  # [PWC-BACKUP S3]
                         ) -> Dict[str, Any]:
    """
    Shim called by window_optimizer_integration_final.py when use_persistent_workers=True.

    Runs all 4 sieve passes (forward, reverse, forward_hybrid, reverse_hybrid)
    using persistent workers and returns the same dict structure as the original
    run_trial() — bidirectional_count, bidirectional_constant, bidirectional_variable, etc.

    This function manages PersistentWorkerCoordinator lifecycle internally so that
    the caller (run_trial) doesn't need to know about workers at all.

    [PWC-BACKUP STEP 1] Added return fields (none removed, v2 shape unchanged):
      pwc_completion     — build_pwc_trial_completion(): per phase/direction
                           accounting, `swept_intervals` of complete passes only,
                           and `trial_complete`;
      pwc_trial_complete — the same boolean, at top level.
    A pass with any failed, timed-out, undispatched or missing chunk is marked
    incomplete there. The existing caller does not read these yet (step 2).
    """
    # [PWC-BACKUP S2] the interval, before the fleet is built
    _req = validate_pwc_seed_interval(seed_start, total_seeds)

    # [PWC-BACKUP S4] the hasattr() fallbacks that used to sit at each pass
    # (both sessions, skip [0, 147]) were silent defaults of the trial's own
    # configuration. Refuse before the fleet is built instead.
    _missing = [a for a in ("window_size", "offset", "sessions", "skip_min", "skip_max")
                if not hasattr(config, a)]
    if _missing:
        raise PwcSettingRefused(
            f"run_trial_persistent: config {type(config).__name__} lacks {_missing}; "
            f"PWC no longer substitutes defaults for them")
    _sessions   = list(config.sessions)
    _skip_range = [config.skip_min, config.skip_max]
    _pass_results: Dict[str, Optional[Dict[str, Any]]] = {}
    # [PWC-STAGE2] window offset other than 0 refused before the fleet is built
    # [PWC-BACKUP STAGE 2b] now only on the SSH route (validate_pwc_window_anchor);
    # on TCP the anchor's upper bound is the worker's to check.
    validate_pwc_window_anchor(config.offset, ssh_route=(pwc_transport != "tcp"))
    # [PWC-STAGE2 ruling 4] phases follow RANGE-MINER's workflow_stages_for: 1-2
    # always, 3-4 when test_both_modes (and the base is not itself a hybrid, the
    # same test _step1_required_phases uses). Never cut short by a survivor count.
    _run_variable = bool(test_both_modes) and not prng_base.endswith("_hybrid")
    # [PWC-STAGE2 ruling 1] variable-skip passes need TCP (threshold report); refuse
    # here rather than after two constant passes have run.
    if (_run_variable or prng_base.endswith("_hybrid")) and pwc_transport != "tcp":
        raise PwcSettingRefused(
            f"run_trial_persistent: variable-skip passes need pwc_transport='tcp' (got "
            f"{pwc_transport!r}); the SSH route reports no threshold to check")

    def _completion_extra() -> Dict[str, Any]:
        comp = build_pwc_trial_completion(_req, _pass_results)
        if not comp["trial_complete"]:
            # the named logger PersistentWorkerCoordinator itself logs to, not
            # `pwc.logger`, so the record does not depend on the instance
            logging.getLogger("PersistentWorkerCoordinator").error(
                f"[PWC-BACKUP S1] TRIAL {trial_number} INCOMPLETE: passes "
                f"{comp['incomplete_passes']} not complete over {list(_req)} — "
                f"marked in pwc_completion; the result must not be read as complete")
        return {"pwc_completion": comp, "pwc_trial_complete": comp["trial_complete"]}
    # [S163-KARG-PWC] Pass node_allowlist so each partition only sees its own nodes.
    # Without this, both P0 and P1 read all nodes from config and launch 24 workers each,
    # competing for the same rigs. P1 wins the race; P0 gets 0 ready workers.
    pwc = PersistentWorkerCoordinator(
        config_file      = coordinator_cfg,
        worker_pool_size = worker_pool_size,
        seed_cap_nvidia  = seed_cap_nvidia,
        seed_cap_amd     = seed_cap_amd,
        pwc_transport    = pwc_transport,
        pwc_host         = pwc_host,
        pwc_port         = pwc_port,
        node_allowlist   = node_allowlist,  # [S163-KARG-PWC] partition node filter
        min_workers      = min_workers,     # [S174] ready gate
        pwc_coordinator_address = pwc_coordinator_address,  # [PWC-BACKUP S3]
        planned_total_seeds     = total_seeds,              # [PWC-BACKUP S5]
    )
    pwc.startup()

    try:
        ws  = config.window_size
        off = config.offset

        # [S163] Announce trial start to dashboard immediately — always show
        # trial number regardless of survivor count
        if pwc._progress_writer:
            try:
                pwc._progress_writer.update_trial_stats(
                    trial_num=trial_number,
                    forward_survivors=0,
                    reverse_survivors=0,
                    bidirectional=0,
                    best_bidirectional=0,
                    config_desc=f"W{ws}_O{off}",
                    accumulated_forward=0,
                    accumulated_reverse=0,
                    accumulated_bidirectional=0,
                )
            except Exception:
                pass

        # ── Pass 1: Forward constant skip ────────────────────────────────────
        print(f"\n    Running FORWARD sieve ({prng_base}) [CONSTANT SKIP] [PERSISTENT]...")
        fwd_result = pwc.run_sieve_pass(
            prng_type    = prng_base,
            residues     = residues,
            total_seeds  = total_seeds,
            threshold    = forward_threshold,
            window_size  = ws,
            dataset_path = dataset_path,
            output_file  = f"results/window_opt_forward_{ws}_{off}_t{trial_number}.json",
            offset       = config.offset,
            sessions     = _sessions,      # [PWC-BACKUP S4] no fallback
            skip_range   = _skip_range,    # [PWC-BACKUP S4] no fallback
            seed_start   = _req[0],        # [PWC-BACKUP S2]
            phase2_threshold = forward_threshold,   # [PWC-STAGE2] one value, both keys
        )
        _pass_results["1"] = fwd_result                     # [PWC-BACKUP S1]
        fwd_survivors   = fwd_result.get("survivors", [])
        fwd_match_rates = fwd_result.get("match_rates", [])
        fwd_map = dict(zip(fwd_survivors, fwd_match_rates))
        print(f"      Forward: {len(fwd_survivors):,} survivors")

        # [PWC-STAGE2 ruling 4] The S147 B1 forward-zero prune that stood here is
        # removed: RANGE-MINER runs every phase of the trial whatever the survivor
        # counts (workflow_stages_for). A zero-survivor phase is a complete result.

        # ── Pass 2: Reverse constant skip ────────────────────────────────────
        prng_reverse = prng_base + "_reverse"
        print(f"    Running REVERSE sieve ({prng_reverse}) [CONSTANT SKIP] [PERSISTENT]...")
        rev_result = pwc.run_sieve_pass(
            prng_type    = prng_reverse,
            residues     = residues,
            total_seeds  = total_seeds,
            threshold    = reverse_threshold,
            window_size  = ws,
            dataset_path = dataset_path,
            output_file  = f"results/window_opt_reverse_{ws}_{off}_t{trial_number}.json",
            offset       = config.offset,
            sessions     = _sessions,      # [PWC-BACKUP S4] no fallback
            skip_range   = _skip_range,    # [PWC-BACKUP S4] no fallback
            seed_start   = _req[0],        # [PWC-BACKUP S2]
            phase2_threshold = reverse_threshold,   # [PWC-STAGE2] one value, both keys
        )
        _pass_results["2"] = rev_result                     # [PWC-BACKUP S1]
        rev_survivors   = rev_result.get("survivors", [])
        rev_match_rates = rev_result.get("match_rates", [])
        rev_map = dict(zip(rev_survivors, rev_match_rates))
        print(f"      Reverse: {len(rev_survivors):,} survivors")

        bidirectional_constant = set(fwd_map.keys()) & set(rev_map.keys())
        print(f"      ✨ Bidirectional (constant): {len(bidirectional_constant):,} survivors")

        # ── Passes 3+4: Variable skip (hybrid) ───────────────────────────────
        bidirectional_variable = set()
        fwd_records_hybrid = []
        rev_records_hybrid = []
        # D3.25-A: the two VARIABLE maps are initialized here, at the same scope
        # as the constant pair, so the v2 return shape never varies. Before
        # D3.25 they existed only inside the `test_both_modes` branch and were
        # discarded on return; a constant-only trial must still return them,
        # empty, rather than omit them.
        fwd_h_map = {}
        rev_h_map = {}

        if _run_variable:
            prng_hybrid = f"{prng_base}_hybrid"
            prng_hybrid_rev = f"{prng_hybrid}_reverse"

            # [PWC-STAGE2] strategies=None: run_sieve_pass loads all five from
            # hybrid_strategy.get_all_strategies(), the source RANGE-MINER's worker
            # uses (miner/range_miner_worker.py _load_strategies). The S147 Q2
            # single balanced_hybrid strategy is removed (owner default rule:
            # PWC does what RANGE-MINER does).
            print(f"    Running FORWARD sieve ({prng_hybrid}) [VARIABLE SKIP] [PERSISTENT]...")
            fwd_h_result = pwc.run_sieve_pass(
                prng_type    = prng_hybrid,
                residues     = residues,
                total_seeds  = total_seeds,
                threshold    = forward_threshold,
                window_size  = ws,
                dataset_path = dataset_path,
                output_file  = f"results/window_opt_forward_hybrid_{ws}_{off}_t{trial_number}.json",
                offset       = config.offset,
                sessions     = _sessions,      # [PWC-BACKUP S4] no fallback
                skip_range   = _skip_range,    # [PWC-BACKUP S4] no fallback
                seed_start   = _req[0],        # [PWC-BACKUP S2]
                strategies   = None,           # [PWC-STAGE2] all five
                phase2_threshold = forward_threshold,   # [PWC-STAGE2 ruling 1]
            )
            _pass_results["3"] = fwd_h_result               # [PWC-BACKUP S1]
            fwd_h_survivors   = fwd_h_result.get("survivors", [])
            fwd_h_match_rates = fwd_h_result.get("match_rates", [])
            fwd_h_map = dict(zip(fwd_h_survivors, fwd_h_match_rates))
            print(f"      Forward (variable): {len(fwd_h_survivors):,} survivors")

            # [PWC-STAGE2 ruling 4] The S147 Q0 gate (skip the reverse variable pass
            # when the forward variable pass found nothing) is removed: phase 4
            # always runs when phase 3 does, as in RANGE-MINER.
            print(f"    Running REVERSE sieve ({prng_hybrid_rev}) [VARIABLE SKIP] [PERSISTENT]...")
            rev_h_result = pwc.run_sieve_pass(
                prng_type    = prng_hybrid_rev,
                residues     = residues,
                total_seeds  = total_seeds,
                threshold    = reverse_threshold,
                window_size  = ws,
                dataset_path = dataset_path,
                output_file  = f"results/window_opt_reverse_hybrid_{ws}_{off}_t{trial_number}.json",
                offset       = config.offset,
                sessions     = _sessions,      # [PWC-BACKUP S4] no fallback
                skip_range   = _skip_range,    # [PWC-BACKUP S4] no fallback
                seed_start   = _req[0],        # [PWC-BACKUP S2]
                strategies   = None,           # [PWC-STAGE2] all five
                phase2_threshold = reverse_threshold,   # [PWC-STAGE2 ruling 1]
            )
            _pass_results["4"] = rev_h_result               # [PWC-BACKUP S1]
            rev_h_survivors   = rev_h_result.get("survivors", [])
            rev_h_match_rates = rev_h_result.get("match_rates", [])
            rev_h_map = dict(zip(rev_h_survivors, rev_h_match_rates))
            print(f"      Reverse (variable): {len(rev_h_survivors):,} survivors")

            bidirectional_variable = set(fwd_h_map.keys()) & set(rev_h_map.keys())
            print(f"      ✨ Bidirectional (variable): {len(bidirectional_variable):,} survivors")

            fwd_records_hybrid = [{"seed": s, "match_rate": fwd_h_map[s]} for s in fwd_h_survivors]
            rev_records_hybrid = [{"seed": s, "match_rate": rev_h_map[s]} for s in rev_h_survivors]

        total_bidi = len(bidirectional_constant) + len(bidirectional_variable)
        print(f"      📊 Total bidirectional: {total_bidi:,}")

        # Update dashboard with trial survivor stats (mirrors coordinator._progress_writer call
        # in window_optimizer_integration_final.py line 373)
        if pwc._progress_writer:
            try:
                pwc._progress_writer.update_trial_stats(
                    trial_num=trial_number,
                    forward_survivors=len(fwd_survivors),
                    reverse_survivors=len(rev_map),
                    bidirectional=len(bidirectional_constant),
                    best_bidirectional=len(bidirectional_constant),
                    config_desc=f"W{config.window_size}_O{config.offset}",
                    accumulated_forward=len(fwd_survivors),
                    accumulated_reverse=len(rev_map),
                    accumulated_bidirectional=total_bidi,
                )
            except Exception:
                pass

        # D3.25-A: the v2 four-map contract, egress-validated. The legacy
        # `forward_map` / `reverse_map` aliases (== the CONSTANT pair) and the
        # four record lists remain as telemetry/compatibility data ONLY; the v2
        # adapter must never read them (G7), and no consumer may reconstruct a
        # map from a record list — `forward_records_hybrid` is built from the
        # raw survivor SEQUENCE here and from map KEYS in ZMQ, so a repeated
        # raw seed makes list and map inequivalent (REV3 §0.3, binding).
        return build_trial_populations(
            forward_map_constant=fwd_map,
            reverse_map_constant=rev_map,
            forward_map_variable=fwd_h_map,
            reverse_map_variable=rev_h_map,
            bidirectional_constant=bidirectional_constant,
            bidirectional_variable=bidirectional_variable,
            pruned=False, reason=None,
            extra={
                "bidirectional_count":    total_bidi,
                "forward_map":            fwd_map,
                "reverse_map":            rev_map,
                "forward_records":        [{"seed": s, "match_rate": fwd_map[s]} for s in fwd_survivors],
                "reverse_records":        [{"seed": s, "match_rate": rev_map[s]} for s in rev_survivors],
                "forward_records_hybrid": fwd_records_hybrid,
                "reverse_records_hybrid": rev_records_hybrid,
                **_completion_extra(),                       # [PWC-BACKUP S1]
            },
            origin="pwc-egress",
        )

    finally:
        pwc.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# CLI — smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Persistent Worker Coordinator — smoke test")
    p.add_argument("--config",       default="distributed_config.json")
    p.add_argument("--pool-size",    type=int, default=2)
    p.add_argument("--total-seeds",  type=int, default=500_000)
    p.add_argument("--prng-type",    default="java_lcg")
    # [PWC-BACKUP S4] no default transport — the CLI no longer becomes SSH silently
    p.add_argument("--pwc-transport", required=True,
                   choices=list(PWC_TRANSPORTS),
                   help="Transport backend: ssh (legacy) or tcp. Required.")
    # [PWC-BACKUP S3] the address rig workers dial (tcp mode with rigs); no default
    p.add_argument("--pwc-coordinator-address", default=None,
                   help="Address the TCP rig workers dial back to. Required for tcp "
                        "with remote rigs; there is no default.")
    # [PWC-BACKUP S4/S5] chunk cap is explicit; the stable envelope judges it
    p.add_argument("--seed-cap-amd", type=int, required=True,
                   help="Chunk-size cap. Must lie inside PWC_STABLE_ENVELOPE.")
    p.add_argument("--sessions", nargs="+", default=None,
                   help="Sessions to sieve (required unless --startup-only)")
    p.add_argument("--skip-min", type=int, default=None)
    p.add_argument("--skip-max", type=int, default=None)
    p.add_argument("--pwc-host",      default="0.0.0.0",
                   help="TCP server bind host (tcp mode only)")
    p.add_argument("--pwc-port",      type=int, default=5600,
                   help="TCP server port (tcp mode only)")
    p.add_argument("--min-workers",   type=int, default=1,
                   help="[TCP] minimum connected workers before dispatch (default=1)")
    p.add_argument("--startup-only", action="store_true",
                   help="Just spawn workers and report alive count, then shutdown")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    pwc = PersistentWorkerCoordinator(
        config_file      = args.config,
        worker_pool_size = args.pool_size,
        pwc_transport    = args.pwc_transport,
        pwc_host         = args.pwc_host,
        pwc_port         = args.pwc_port,
        min_workers      = args.min_workers,
        seed_cap_amd     = args.seed_cap_amd,
        pwc_coordinator_address = args.pwc_coordinator_address,
        planned_total_seeds     = args.total_seeds,
    )
    if not args.startup_only and (args.sessions is None or args.skip_min is None
                                  or args.skip_max is None):
        p.error("--sessions, --skip-min and --skip-max are required for a sieve pass")
    pwc.startup()

    if args.startup_only:
        alive = sum(1 for w in pwc.workers if w.alive)
        print(f"\nAlive workers: {alive}/{len(pwc.workers)}")
        pwc.shutdown()
        sys.exit(0)

    # Minimal smoke test sieve pass
    residues = [0, 1, 2, 3, 4, 5, 6, 7]   # placeholder
    result = pwc.run_sieve_pass(
        prng_type    = args.prng_type,
        residues     = residues,
        total_seeds  = args.total_seeds,
        threshold    = 0.30,
        window_size  = 8,
        dataset_path = "daily3.json",
        output_file  = "/tmp/pwc_smoke_test.json",
        sessions     = args.sessions,
        skip_range   = [args.skip_min, args.skip_max],
    )
    print(f"\nSmoke test result: {result.get('survivor_count', 0)} survivors")
    pwc.shutdown()
