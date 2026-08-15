#!/usr/bin/env bash
# =============================================================================
# STOPGAP — MANUAL 25-DAEMON RANGE-MINER FLEET LAUNCH
# =============================================================================
#
# ⚠ THIS IS A STOPGAP, NOT THE DELIVERABLE.
#
# RANGE-MINER has no worker-launch mechanism of its own. Team Beta requires a
# BACKEND-OWNED launcher — worker startup solved inside the RANGE-MINER backend
# (it is a standalone Step-1 backend behind one flag, so startup must not become
# a new layer above Step 1) — as a precondition for final Phase-7 certification.
# That launcher is explicitly OUT OF SCOPE for the staging patch
# (docs/CLAUDE_CODE_INSTRUCTIONS_STAGING_DIR_PART_B.md §5/§7).
#
# This file exists only so a live fleet can be raised BY HAND to exercise
# G-PROD-SHAPE (tests/gate_s172_prod_shape.py) while that launcher does not yet
# exist. It is a one-off operator tool:
#   * it does not supervise, restart, health-check or reap anything;
#   * it is not invoked by WATCHER, the optimizer, or the coordinator;
#   * nothing in production imports or depends on it.
# DELETE IT once the backend owns worker startup.
#
# It is deliberately NOT the autonomous startup of Part B §7.
#
# -----------------------------------------------------------------------------
# Fleet definition is READ FROM COMMITTED SOURCE, never baked in here
# (CLAUDE.md §1 rule 4). Endpoints come from rig_profiles_config.json — the
# boot-selector join table — under its declared default_profile (override with
# RIG_PROFILE=baremetal|proxmox). GPU counts, interpreter paths and repo paths
# come from distributed_config.json. Today that resolves to the 25 identities of
# the frozen Resolved Execution Set (zeus-ubuntu-vm:gpu0 + 8 each on
# rrig6600/b/c). If the rigs are booted back to bare metal, flip the profile —
# do not edit addresses into this script.
#
# Per-worker environment:
#   ROCR_VISIBLE_DEVICES=N  exposes exactly ONE ROCm device to the daemon, which
#                           is why every rig worker binds --device-index 0 while
#                           keeping its true --gpu-id N (worker identity is
#                           f"{hostname}:gpu{gpu_id}" and must match the frozen
#                           execution set).
#   CUDA_VISIBLE_DEVICES=N  the same containment for the local NVIDIA box.
#   CUPY_CACHE_DIR          PER WORKER — a shared JIT cache is the S157 race
#                           when 8 daemons compile simultaneously.
#   PYTHONPATH=<repo>       REQUIRED. range_miner_worker.py does
#                           `from miner.range_miner_protocol import ...` and does
#                           not touch sys.path, so running it by file path puts
#                           <repo>/miner on sys.path — not <repo> — and the
#                           import fails. Verified 2026-08-05: without PYTHONPATH
#                           the daemon dies at once with
#                           `ModuleNotFoundError: No module named 'miner'`.
#
# Stagger defaults to 3 s. With the ATTEMPT-6 release barrier it paces SENTINEL
# EMISSION, not registration: registration is paced by the release token, so the
# 75 s of dispatch no longer competes with the 180 s admission window at all and
# the launch gets strictly more margin than attempts 2 and 5 had.
#
# ⚠ ORDER — CORRECTED BY THE ATTEMPT-6 REMEDIATION (§8.4.3). THE OLD COMMENT HERE
# SAID "start the coordinator FIRST, then this script", AND THAT IS NOW WRONG.
# An operator following it would launch into an UNRELEASED fleet: with
# RUN_NONCE set, every worker emits its startup sentinel and then PARKS on the
# per-host release file, so nothing connects out until the harness has verified
# 25/25 sentinel delivery and written the tokens. The order is:
#
#   1. launch the fleet          (this script; workers warm GPUs, emit the
#                                 sentinel and park — the coordinator need not
#                                 exist yet, because nothing connects out)
#   2. verify 25/25 sentinels    scripts/gate12_sentinel_gate.py --phase verify
#                                 ANY shortfall/UNAVAILABLE/ERROR -> REFUSAL
#   3. launch the pipeline       the coordinator binds and listens
#   4. wait for the listener
#   5. write the release tokens  scripts/gate12_sentinel_gate.py --phase release
#   6. workers connect + REGISTER
#
# Steps 1-2 sit OUTSIDE the run, so verification time is free and cannot spend
# the admission budget — which is how a slow probe sweep would otherwise
# manufacture attempt 2's `worker_admission_timeout` with the fix in place.
#
# ⚠ WHAT THIS SCRIPT'S COMPLETION MEANS — corrected by the D6 integration repair
# (Beta, 2026-08-14). THE OLD SHAPE ENDED THE DISPATCH LOOP WITH AN
# ARGUMENT-LESS `wait`, WHICH WAITED FOR THE LOCAL WORKER TOO:
#
#     launch_fleet_manual.sh completion
#         MEANS:         all worker DISPATCH operations have been dispositioned
#         DOES NOT MEAN: all launched worker PROCESSES have exited
#
# The local worker is a `nohup … &` background job OF THIS SHELL, so a bare
# `wait` blocked on it. With RUN_NONCE set that worker parks at the release
# barrier, and the release is written by a LATER step of the caller — which this
# script has not returned to. Measured in the D6 dry run of 2026-08-14: the final
# dispatch echo printed, the completion block never did, and the local worker
# logged `SESSION_RELEASE_ABORTED waited_s=595.299` when it was killed at ten
# minutes. A synchronous caller cannot advance to sentinel verification and
# release while waiting for the long-lived local worker whose continuation
# depends on that later release.
#
# The wait set is therefore the SHORT-LIVED REMOTE DISPATCH JOBS ONLY. The local
# worker's PID is recorded and deliberately EXCLUDED, and it must still be ALIVE
# AND PARKED when this script returns — which is asserted below rather than
# assumed. Release is NOT moved earlier: sentinel verification must still precede
# it, and that ordering is the whole point of the two-phase launch.
#
# Without RUN_NONCE the script behaves exactly as before (no sentinel, no
# barrier, coordinator-first), so an existing single-worker smoke run is
# unaffected.
#
# Usage:  bash scripts/launch_fleet_manual.sh <coordinator_host> [port] [logdir]
#   env:  RIG_PROFILE=baremetal|proxmox   STAGGER=<seconds>
#         RUN_NONCE=<token>               RELEASE_DEADLINE=<seconds>
#         REMOTE_RELEASE_DIR=<dir>
# =============================================================================
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COORD_HOST="${1:-}"
PORT="${2:-5700}"
LOGDIR="${3:-$REPO_ROOT/logs/miner_workers}"
STAGGER="${STAGGER:-3}"
# [ATTEMPT-6 §8.4.2] the startup sentinel + pre-REGISTER barrier. Empty RUN_NONCE
# = pre-amendment behaviour, so every existing use of this stopgap is unaffected.
RUN_NONCE="${RUN_NONCE:-}"
RELEASE_DEADLINE="${RELEASE_DEADLINE:-900}"
REMOTE_RELEASE_DIR="${REMOTE_RELEASE_DIR:-/tmp/minerlogs}"

if [ -z "$COORD_HOST" ]; then
    echo "usage: bash scripts/launch_fleet_manual.sh <coordinator_host> [port] [logdir]" >&2
    echo "  e.g. bash scripts/launch_fleet_manual.sh 192.168.3.177 5700" >&2
    exit 2
fi

# ---- resolve the fleet from committed source -------------------------------
FLEET="$(cd "$REPO_ROOT" && python3 - <<'PYEOF'
import json, os, sys

with open("rig_profiles_config.json") as f:
    rp = json.load(f)
with open("distributed_config.json") as f:
    dc_nodes = json.load(f)["nodes"]

profile = os.environ.get("RIG_PROFILE") or rp["default_profile"]
if profile not in rp["profiles"]:
    sys.exit("RIG_PROFILE=%r is not one of %s" % (profile, rp["profiles"]))

by_host = {n["hostname"]: n for n in dc_nodes}
for node in rp["nodes"]:
    cfg = by_host.get(node["config_hostname"])
    if cfg is None:
        sys.exit("no distributed_config.json node for %r" % node["config_hostname"])
    endpoint = node["endpoints"][profile]
    print("\t".join([
        node["worker_hostname"],
        endpoint,
        str(cfg["gpu_count"]),
        cfg["python_env"],
        cfg["script_path"],
        "local" if node.get("local") else "remote",
    ]))
PYEOF
)" || { echo "[launch] FATAL: could not resolve the fleet from committed config" >&2; exit 1; }

TOTAL=$(echo "$FLEET" | awk -F'\t' '{s+=$3} END {print s}')

mkdir -p "$LOGDIR"
echo "[launch] coordinator = $COORD_HOST:$PORT"
echo "[launch] profile     = ${RIG_PROFILE:-$(python3 -c "import json;print(json.load(open('$REPO_ROOT/rig_profiles_config.json'))['default_profile'])")}"
echo "[launch] stagger     = ${STAGGER}s   (total dispatch ~$((TOTAL * STAGGER))s)"
echo "[launch] local logs  = $LOGDIR"
echo "[launch] workers     = $TOTAL"
echo "$FLEET" | awk -F'\t' '{printf "[launch]   %-16s %-16s %s GPU(s)  [%s]\n", $1, $2, $3, $6}'
echo

# ---- dispatch ---------------------------------------------------------------
# Local first (it is the coordinator's own box), then the rigs. Every remote
# dispatch is backgrounded so the rigs come up in parallel; the stagger paces
# registration, not the ssh calls.
#
# TWO STDIN DEFENCES, both required — this loop truncated itself on 2026-08-05
# and dispatched 9 of 25 (see below):
#
#   `ssh -n`   ssh reads its own stdin and forwards it to the remote command.
#              A BACKGROUNDED ssh here inherited THIS LOOP'S stdin — the fleet
#              here-string — and drained the rrig6600b and rrig6600c records
#              into the first rrig6600 worker's remote shell. `read` then hit
#              EOF and the loop ended after one rig. -n binds ssh's stdin to
#              /dev/null. (Measured: the remote end received exactly
#              "C\t192.168.3.156..." and "D\t192.168.3.164..." on stdin.)
#              This is a RACE, not a certainty — an ssh whose remote command
#              exits immediately can finish before it drains anything, which is
#              why a short-lived probe command does NOT reproduce it. The real
#              remote command lives long enough. Never rely on ssh losing.
#
#   fd 3       The record stream is read from a dedicated descriptor, so no
#              child of this loop can consume it even if one is added later
#              without -n. Defence in depth: -n fixes today's bug, fd 3 fixes
#              the class.
#
# The dry-run that "verified 25 workers" stubbed ssh out with echo — and echo
# does not read stdin, so the stub deleted the exact behaviour that fails.
# Hence the DISPATCHED counter below: it counts real dispatches, so a truncated
# run is LOUD instead of printing a confident total it never reached.
#
# [D6-REPAIR §B] TWO PID SETS, AND THE DIFFERENCE BETWEEN THEM IS THE FIX.
# REMOTE_DISPATCH_PIDS holds the short-lived `ssh` jobs — those are dispatch
# operations and waiting for them is what "dispatched" means. LOCAL_WORKER_PIDS
# holds the long-lived local daemon, which is NOT a dispatch operation: it is the
# work itself, and it is excluded from the wait set. Keeping them in one
# undifferentiated job table is exactly what the bare `wait` did.
DISPATCHED=0
REMOTE_DISPATCH_PIDS=()
REMOTE_DISPATCH_LABELS=()
LOCAL_WORKER_PIDS=()
LOCAL_WORKER_LABELS=()
while IFS=$'\t' read -r WHOST ENDPOINT NGPU PYBIN SCRIPTPATH KIND <&3; do
    [ -n "${WHOST:-}" ] || continue
    N=0
    while [ "$N" -lt "$NGPU" ]; do
        if [ "$KIND" = "local" ]; then
            echo "[launch] $WHOST gpu$N (local)"
            LOCAL_LOG="$LOGDIR/${WHOST}_gpu$N.log"
            LOCAL_SENTINEL_ARGS=""
            if [ -n "$RUN_NONCE" ]; then
                LOCAL_SENTINEL_ARGS="--run-nonce $RUN_NONCE \
                    --session-release-file $LOGDIR/gate12_release_$RUN_NONCE \
                    --release-deadline $RELEASE_DEADLINE \
                    --sentinel-log-path $LOCAL_LOG"
            fi
            CUDA_VISIBLE_DEVICES="$N" \
            CUPY_CACHE_DIR="/tmp/cupy_cache_local_gpu$N" \
            PYTHONPATH="$SCRIPTPATH" \
            nohup "$PYBIN" "$SCRIPTPATH/miner/range_miner_worker.py" \
                --host "$COORD_HOST" --port "$PORT" \
                --gpu-id "$N" --device-index 0 \
                $LOCAL_SENTINEL_ARGS \
                > "$LOCAL_LOG" 2>&1 < /dev/null &
            LOCAL_PID=$!
            echo "[launch]   pid=$LOCAL_PID"
            LOCAL_WORKER_PIDS+=("$LOCAL_PID")
            LOCAL_WORKER_LABELS+=("$WHOST:gpu$N")
        else
            echo "[launch] $WHOST ($ENDPOINT) gpu$N"
            REMOTE_SENTINEL_ARGS=""
            if [ -n "$RUN_NONCE" ]; then
                REMOTE_SENTINEL_ARGS="--run-nonce $RUN_NONCE \
                    --session-release-file $REMOTE_RELEASE_DIR/gate12_release_$RUN_NONCE \
                    --release-deadline $RELEASE_DEADLINE \
                    --sentinel-log-path /tmp/minerlogs/gpu$N.log"
            fi
            ssh -n -o BatchMode=yes -o ConnectTimeout=10 "michael@$ENDPOINT" \
                "mkdir -p /tmp/minerlogs $REMOTE_RELEASE_DIR /tmp/cupy_cache_gpu$N && \
                 cd $SCRIPTPATH && \
                 ROCR_VISIBLE_DEVICES=$N \
                 CUPY_CACHE_DIR=/tmp/cupy_cache_gpu$N \
                 PYTHONPATH=$SCRIPTPATH \
                 nohup $PYBIN $SCRIPTPATH/miner/range_miner_worker.py \
                   --host $COORD_HOST --port $PORT \
                   --gpu-id $N --device-index 0 \
                   $REMOTE_SENTINEL_ARGS \
                   > /tmp/minerlogs/gpu$N.log 2>&1 & \
                 echo started" >/dev/null 2>&1 &
            REMOTE_DISPATCH_PIDS+=("$!")
            REMOTE_DISPATCH_LABELS+=("$WHOST($ENDPOINT):gpu$N")
        fi
        DISPATCHED=$((DISPATCHED + 1))
        N=$((N + 1))
        sleep "$STAGGER"
    done
done 3<<< "$FLEET"

# [D6-REPAIR §B] WAIT ONLY FOR THE REMOTE DISPATCH JOBS. An argument-less `wait`
# here is what blocked the launcher on its own parked local worker for the whole
# release deadline. Each PID is waited individually so its status is READ rather
# than discarded: `wait` with arguments yields the job's exit status, and an ssh
# that could not connect or authenticate is a dispatch that DID NOT LAND. The old
# shape counted loop iterations, so `DISPATCHED == TOTAL` could be printed over a
# fleet no ssh ever reached.
REMOTE_FAILURES=0
i=0
while [ "$i" -lt "${#REMOTE_DISPATCH_PIDS[@]}" ]; do
    wait "${REMOTE_DISPATCH_PIDS[$i]}"
    RC=$?
    if [ "$RC" -ne 0 ]; then
        echo "[launch] *** DISPATCH FAILED: ${REMOTE_DISPATCH_LABELS[$i]} — ssh exited $RC ***" >&2
        REMOTE_FAILURES=$((REMOTE_FAILURES + 1))
    fi
    i=$((i + 1))
done
echo "[launch] remote dispatch jobs dispositioned: ${#REMOTE_DISPATCH_PIDS[@]} (failures: $REMOTE_FAILURES)"

# The local worker must be ALIVE AND PARKED when this script returns. It is the
# one worker the caller cannot re-probe by ssh, and a dead local worker is the
# silent single-worker loss: the sentinel gate would still find its sentinel in
# the log it already wrote, the cohort would freeze at 25 and admit 24 — attempt
# 2's `worker_admission_timeout` shape, manufactured by the launch harness rather
# than by the fleet. So it is asserted, not assumed.
LOCAL_DEAD=0
i=0
while [ "$i" -lt "${#LOCAL_WORKER_PIDS[@]}" ]; do
    if kill -0 "${LOCAL_WORKER_PIDS[$i]}" 2>/dev/null; then
        echo "[launch] local worker ${LOCAL_WORKER_LABELS[$i]} pid=${LOCAL_WORKER_PIDS[$i]} ALIVE (excluded from the wait set)"
    else
        echo "[launch] *** LOCAL WORKER NOT ALIVE: ${LOCAL_WORKER_LABELS[$i]} pid=${LOCAL_WORKER_PIDS[$i]} ***" >&2
        LOCAL_DEAD=$((LOCAL_DEAD + 1))
    fi
    i=$((i + 1))
done

echo
if [ "$REMOTE_FAILURES" -ne 0 ]; then
    echo "[launch] *** $REMOTE_FAILURES of ${#REMOTE_DISPATCH_PIDS[@]} REMOTE DISPATCHES FAILED ***" >&2
    echo "[launch] The fleet is INCOMPLETE. Admission will fall short and the run" >&2
    echo "[launch] will fail at the worker_admission_timeout. Do not treat a" >&2
    echo "[launch] partial fleet as a production-shape trial." >&2
    exit 1
fi
if [ "$LOCAL_DEAD" -ne 0 ]; then
    echo "[launch] *** $LOCAL_DEAD LOCAL WORKER(S) DIED DURING DISPATCH ***" >&2
    echo "[launch] The local worker's log already carries its sentinel, so the" >&2
    echo "[launch] sentinel gate would pass on a process that no longer exists." >&2
    echo "[launch] Refusing here instead: see $LOGDIR for its log." >&2
    exit 1
fi
if [ "$DISPATCHED" -ne "$TOTAL" ]; then
    echo "[launch] *** TRUNCATED DISPATCH: $DISPATCHED of $TOTAL ***" >&2
    echo "[launch] The fleet is INCOMPLETE. Admission will fall short and the run" >&2
    echo "[launch] will fail at the worker_admission_timeout. Do not treat a" >&2
    echo "[launch] partial fleet as a production-shape trial." >&2
    exit 1
fi
echo "[launch] all $DISPATCHED of $TOTAL daemons dispatched"
echo "[launch] DISPATCH COMPLETE — this means every dispatch operation has been"
echo "[launch] dispositioned. It does NOT mean the worker processes have exited:"
echo "[launch] the local worker is deliberately still running and is excluded"
echo "[launch] from the wait set."
echo "[launch] local worker logs : $LOGDIR"
echo "[launch] rig worker logs   : /tmp/minerlogs/gpu<N>.log on each rig"
echo "[launch] verify admission in the coordinator log before trusting this line —"
echo "[launch] 'dispatched' means ssh returned, NOT that a daemon registered."
if [ -n "$RUN_NONCE" ]; then
  echo "[launch] run nonce         : $RUN_NONCE"
  echo "[launch] release token     : <logdir>/gate12_release_$RUN_NONCE (per host)"
  echo "[launch] THE FLEET IS PARKED AT THE RELEASE BARRIER and has sent nothing"
  echo "[launch] to any coordinator. Verify the sentinels, then write the tokens:"
  echo "[launch]   python3 scripts/gate12_sentinel_gate.py --phase verify \\"
  echo "[launch]       --run-nonce $RUN_NONCE --local-log-dir $LOGDIR"
fi
