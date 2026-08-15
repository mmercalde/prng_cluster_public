#!/usr/bin/env python3
"""GATE-12 SENTINEL-CORRELATED WORKER LIVENESS GATE — the last worker-state wall.

WHY THIS EXISTS, AND WHAT IT IS NOT
-----------------------------------
In D6 dry run #2 (2026-08-14) the worker-log sentinel gate returned
`PASS 25/25` over a fleet in which the local worker was ALREADY DEAD. Nothing
about that was a sentinel-gate defect: its contract is LOG DELIVERY, the dead
worker genuinely had delivered its current-nonce sentinel through the production
session-event path, and the record it delivered was true when it was written. The
defect was in the LAUNCH AUTHORITY — treating delivery as though it implied
liveness. That is a demonstration on hardware of the property Beta had required
be measured rather than argued, so:

    ⚠ DO NOT WIDEN THE SENTINEL GATE. It answers "did the channel work".
      This gate answers "is that worker still there, and still parked".

THE JOIN, WHICH IS THE WHOLE MECHANISM
--------------------------------------
    A log record does not itself prove liveness.
    A PID does not itself prove identity.

Either half alone is satisfiable by the wrong world. A sentinel proves a process
existed at T0 and said who it was; a live PID proves *something* is running now.
Only the JOIN — this run's sentinel names a PID, and that exact PID is now
running an argv that is this worker — proves the fact the launch authority
actually needs. Per each of the expected identities, all eleven of:

     1. this run's SESSION_SENTINEL is present in the expected worker log
     2. the PID emitted BY THAT SENTINEL is extracted
     3. that exact PID exists right now
     4. that PID's /proc cmdline is read
     5. it is range_miner_worker.py
     6. it carries THIS run's nonce
     7. it carries the expected --gpu-id
     8. it carries the nonced release-file / barrier arguments
     9. this run's SESSION_RELEASE_WAIT is present — PARSED, with its event,
        run_nonce, worker_id and release_path all matching this run and the
        live process's own barrier argument
    10. this run has NOT emitted SESSION_RELEASED
    11. this run has NOT emitted SESSION_RELEASE_ABORTED

PASS only at a FULL COUNT of validated live parked identities. REFUSE on any of:
a short count · a duplicate PID on one host · wrong gpu · wrong nonce · the PID
gone · the PID reused by an unrelated process · a worker that never reached the
barrier · a released worker · an aborted worker · SSH unavailable · log/PID
evidence that cannot be classified.

[R1] 9 IS WHAT ENTITLES THIS GATE TO PRINT THE WORD "PARKED", and it was missing
from the first implementation. 1-8 prove a process is CONFIGURED TO PARK:
`--session-release-file` is a property of how it was launched. But worker startup
is `emit_startup_sentinel() -> await_session_release() -> connect()`
(`miner/range_miner_worker.py:2165-2170`), so there is a real interval in which
the sentinel exists, the argv is perfect, and the barrier has not been reached.
The attempt-6 gate proves that SOURCE ORDERING; it does not convert "has the
arguments necessary to park" into "has arrived". `SESSION_RELEASE_WAIT` is the
first act inside `await_session_release` (`:1498-1500`) — the worker's own
statement that it got there — so it is observed rather than inferred.

10 and 11 are not bookkeeping either. A worker already RELEASED is no longer
parked, so it is not in the state this gate is asked about; one that already
ABORTED is on its way out. Both are "alive" to a naive process count and neither
is a worker this run may proceed on.

THE RESIDUAL, STATED AS BETA STATES IT
--------------------------------------
No prelaunch probe can guarantee a worker will not die a microsecond later, and
this gate probes the identities SEQUENTIALLY — there is no instant at which all
25 were seen together, and the last identity's observation does not retroactively
refresh the first identity's /proc read. So the property established is exactly:

    "During the final pre-coordinator liveness sweep, every expected
     sentinel-correlated identity was observed alive and parked; the sweep
     completed immediately before coordinator creation."

Stated that way because that is the measurement performed. The earlier wording
("at the last pre-coordinator worker-state observation, all expected processes
existed and remained parked") implied a single simultaneous observation that
never happened. Batching per host, or probing hosts concurrently to reduce sweep
skew, would narrow the window — it is an optimization and is deliberately NOT
done here. The 25-worker admission wall remains the runtime authority once
registration begins, and this gate does not touch it.

WHY `pgrep -c -f` IS NOT USED, AND MAY NOT BE (Beta ruling 3)
------------------------------------------------------------
`pgrep -c -f "range_miner_worker"` counted 16 for 8 workers on every rig in D6
dry run #2 — eight wrapper subshells whose argv contained the worker path, and
eight Python workers. That is a truthful count of matching COMMAND-LINE STRINGS
and an untruthful count of WORKERS, and no constant corrects it: it is a count of
the wrong thing. It is also blind to identity — it cannot say which gpu, which
nonce, or whether the process it counted is the one whose sentinel was verified.
Raw `ps`/`pgrep -af` output may travel in the evidence bundle as diagnostic
context; it may not decide PASS or FAIL, here or anywhere.

Exit status: 0 = proceed, 1 = REFUSE. Nothing is hardcoded: the identities, their
endpoints, per-node GPU counts, ssh user and log paths are DERIVED — and derived
by calling the sentinel gate's own resolver, so the two gates cannot drift into
asking about different workers or reading different files. That shared derivation
is what makes the join a join.
"""

import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The identity derivation, the log-path convention and the three-outcome
# vocabulary all come from the sentinel gate rather than being re-declared here.
# The gates must agree about WHICH worker and WHICH file by construction: a
# second copy of `log_path_for` would be a second convention, and this gate's
# entire claim is that it is talking about the same records the sentinel gate
# accepted.
import gate12_sentinel_gate as SG                                   # noqa: E402

PROBE_OK = SG.PROBE_OK
PROBE_UNAVAILABLE = SG.PROBE_UNAVAILABLE
PROBE_ERROR = SG.PROBE_ERROR

SSH_CONNECT_TIMEOUT = SG.SSH_CONNECT_TIMEOUT
PROBE_TIMEOUT = SG.PROBE_TIMEOUT
SSH_TRANSPORT_FAILURE_STATUS = SG.SSH_TRANSPORT_FAILURE_STATUS

EXIT_PROCEED = SG.EXIT_PROCEED
EXIT_REFUSE = SG.EXIT_REFUSE

# Truncation sentinels. A probe whose output stops early has NOT told us the
# fleet is dead — it has told us nothing, and the difference is the whole
# UNAVAILABLE-is-not-zero rule. Without the END marker a short read looks exactly
# like a worker with no sentinel records.
BEGIN_SENTINEL = "TFM-LIVENESS-BEGIN"
END_SENTINEL = "TFM-LIVENESS-END"

# argv is transported NUL->US (0x1f) rather than NUL->space, so an argument that
# contains a space cannot masquerade as two arguments. `--gpu-id 3` must be two
# adjacent argv elements; a substring test would accept `--gpu-id 30`.
ARGV_SEP = "\x1f"

WORKER_ENTRYPOINT_BASENAME = "range_miner_worker.py"


def worker_targets(rig_profile="proxmox", admission_count=25):
    """The same identities the sentinel gate probed, from the same resolver."""
    return SG.worker_targets(rig_profile, admission_count)


def build_probe_script(log_path, nonce):
    """ONE argv element. ssh joins trailing arguments with spaces WITHOUT
    re-quoting them, so a script passed as several words is re-parsed on the far
    side with its quoting gone — the defect the certified GPU probe was corrected
    for, and it is not repeated here.

    The script emits a delimited, line-oriented record and NOTHING ELSE:

        TFM-LIVENESS-BEGIN
        LOG_READABLE=0|1
        SENTINEL <the whole matching log line, JSON and all>   (0..n)
        WAITREC  <the whole matching log line, JSON and all>   (0..n)
        RELEASED=<n>
        ABORTED=<n>
        PROC <pid> <argv, US-separated>                        (0..n)
        PROC <pid> ABSENT                                      (0..n)
        TFM-LIVENESS-END

    [R1] `WAITREC` IS WHY THIS GATE MAY SAY "PARKED". Without it the probe
    gathered only the sentinel, the terminal records and `/proc`, which together
    prove a worker HAS THE ARGUMENTS NECESSARY TO PARK — not that it ever
    reached the barrier. Worker startup is
    `emit_startup_sentinel() -> await_session_release() -> connect()`
    (`miner/range_miner_worker.py:2165-2170`), so there is a real interval in
    which the sentinel exists, the argv is perfect, and the worker has not yet
    entered the wait. `SESSION_RELEASE_WAIT` is emitted as the first act inside
    `await_session_release` (`:1498-1500`), so its presence is the worker's own
    statement that it has arrived.

    The whole records are shipped, not a count, because the count alone cannot
    answer "whose wait was it" — and a wait record belonging to another identity
    proves nothing about this one.

    [R2] AND THE `grep '<nonce>'` HERE IS A PREFILTER, NOT AN AUTHORITY. Text
    containment is not semantic equality: the current nonce can appear in a line
    whose authoritative `run_nonce` names a different run. Every shipped record
    is reparsed and compared field by field in `classify()`; nothing is accepted
    on the strength of this grep.

    The PID loop is driven by the sentinel lines themselves — `sed` lifts the
    `"pid":` field out of each matching record — so the /proc read is about the
    process THIS RUN'S SENTINEL NAMED, never about whatever happens to be running
    under a matching name. That is the join, done at the only place it can be
    done in one round trip. The lifted value is not trusted: `classify()`
    re-parses the same JSON authoritatively and refuses if the two disagree.

    `grep -c` prints 0 and exits 1 on no match. Nothing here adds `|| echo 0` —
    those two constructs together manufactured attempt 1's `0/8`, and a count
    that arrives twice is worse than one that arrives once.
    """
    lp = log_path
    return (
        f"echo {BEGIN_SENTINEL}; "
        f"if [ -r '{lp}' ]; then "
        f"echo LOG_READABLE=1; "
        f"grep 'SESSION_SENTINEL' '{lp}' | grep '{nonce}' | sed 's/^/SENTINEL /'; "
        f"grep 'SESSION_RELEASE_WAIT' '{lp}' | grep '{nonce}' | sed 's/^/WAITREC /'; "
        f"echo RELEASED=$(grep 'SESSION_RELEASED' '{lp}' | grep -c '{nonce}'); "
        f"echo ABORTED=$(grep 'SESSION_RELEASE_ABORTED' '{lp}' | grep -c '{nonce}'); "
        f"for p in $(grep 'SESSION_SENTINEL' '{lp}' | grep '{nonce}' | "
        f"sed -n 's/.*\"pid\": *\\([0-9][0-9]*\\).*/\\1/p' | sort -u); do "
        f"if [ -r /proc/$p/cmdline ]; then "
        f"echo \"PROC $p $(tr '\\0' '\\037' < /proc/$p/cmdline)\"; "
        f"else echo \"PROC $p ABSENT\"; fi; "
        f"done; "
        f"else echo LOG_READABLE=0; fi; "
        f"echo {END_SENTINEL}"
    )


def _run(cmd, timeout):
    return subprocess.run(cmd, capture_output=True, timeout=timeout)


def probe_liveness(target, nonce, log_path):
    """One identity, one round trip, three possible transport outcomes.

    UNAVAILABLE means the probe did not run or did not finish: ssh transport
    failure, timeout, a truncated record, an unreadable log. ERROR means it ran
    and its output cannot be classified. Neither is ever rendered as a count, and
    both REFUSE — but they are kept apart because "we could not look" and "we
    looked and saw nothing" are different facts about the world, and collapsing
    them is what let `GPU_COUNT_MISMATCH: 0/8` through a 3/3 preflight.

    [R2 rule, inherited deliberately] ssh's own failure is status 255 and this
    gate RESERVES that value for transport, exactly as the sentinel gate does.
    The reservation is sound only because the script above cannot produce 255 —
    its branches end in `echo`/`fi` — and if that script changes this rule must
    be revisited. The check is gated to the remote branch: locally, 255 would be
    an ordinary command status carrying no transport meaning.
    """
    result = {"worker_id": target["worker_id"], "endpoint": target["endpoint"],
              "gpu": target["gpu"], "local": target["local"],
              "log_path": log_path, "status": None, "reason": None,
              "pid": None, "argv": None, "sentinels": 0, "release_waits": 0,
              "released": None, "aborted": None, "stderr": ""}
    script = build_probe_script(log_path, nonce)
    try:
        if target["local"]:
            proc = _run(["bash", "-c", script], PROBE_TIMEOUT)
        else:
            proc = _run(
                ["ssh", "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT}",
                 "-o", "BatchMode=yes",
                 f"{target['ssh_user']}@{target['endpoint']}", script],
                PROBE_TIMEOUT)
    except subprocess.TimeoutExpired:
        result.update(status=PROBE_UNAVAILABLE, reason="timeout")
        return result
    except Exception as e:                                          # noqa: BLE001
        result.update(status=PROBE_UNAVAILABLE,
                      reason=f"{type(e).__name__}:{e}")
        return result

    result["stderr"] = proc.stderr.decode(errors="replace").strip()
    if not target["local"] and proc.returncode == SSH_TRANSPORT_FAILURE_STATUS:
        result.update(status=PROBE_UNAVAILABLE,
                      reason=f"ssh_transport_failure (ssh_exit_{proc.returncode})")
        return result
    lines = proc.stdout.decode(errors="replace").splitlines()
    if BEGIN_SENTINEL not in lines:
        result.update(status=PROBE_UNAVAILABLE,
                      reason=f"probe_did_not_start:{lines[:3]!r}")
        return result
    if END_SENTINEL not in lines:
        # Truncation. NOT "no sentinels found" — the record is incomplete, so
        # every count in it is a lower bound and none of them may be read.
        result.update(status=PROBE_UNAVAILABLE,
                      reason="truncated_probe_output (no END sentinel)")
        return result
    body = lines[lines.index(BEGIN_SENTINEL) + 1:lines.index(END_SENTINEL)]
    return classify(result, body, target, nonce)


def _parse_sentinel(line):
    """The authoritative parse: the record's own JSON, not a regex over it."""
    brace = line.find("{")
    if brace < 0:
        raise ValueError("no JSON object in sentinel record")
    return json.loads(line[brace:])


def classify(result, body, target, nonce):
    """Turn one probe's record into OK-with-a-verdict, or UNAVAILABLE/ERROR.

    Checks form the eleven-part conjunction below. Diagnostic evaluation order is
    intentional and is not identical to the numbered presentation order — the
    terminal records (10, 11) are read before the wait proof (9) so that the
    refusal an operator sees is the most specific one available. The first
    unsatisfied check names the refusal, so an operator reads WHICH property
    failed rather than a generic short count.
    """
    readable = None
    sentinel_lines, wait_lines, procs = [], [], {}
    released = aborted = None
    for line in body:
        if line.startswith("LOG_READABLE="):
            readable = line.split("=", 1)[1].strip()
        elif line.startswith("SENTINEL "):
            sentinel_lines.append(line[len("SENTINEL "):])
        elif line.startswith("WAITREC "):
            wait_lines.append(line[len("WAITREC "):])
        elif line.startswith("RELEASED="):
            released = line.split("=", 1)[1].strip()
        elif line.startswith("ABORTED="):
            aborted = line.split("=", 1)[1].strip()
        elif line.startswith("PROC "):
            rest = line[len("PROC "):]
            pid, _, argv = rest.partition(" ")
            procs[pid] = argv

    if readable == "0":
        result.update(status=PROBE_UNAVAILABLE, reason="log_unreadable")
        return result
    if readable != "1":
        result.update(status=PROBE_ERROR,
                      reason=f"unparseable_probe_output:{body[:4]!r}")
        return result

    result["sentinels"] = len(sentinel_lines)
    result["release_waits"] = len(wait_lines)
    result["released"], result["aborted"] = released, aborted

    # (1) this run's sentinel is present at all
    if not sentinel_lines:
        result.update(status=PROBE_OK,
                      reason="no_sentinel_record_for_this_nonce")
        return result

    # (2) the PID emitted BY THAT SENTINEL. Two sentinels for one nonce naming
    # two different PIDs is not a tie to break — it is an ambiguous identity, and
    # a gate that picked one would be choosing which fact to believe.
    parsed = []
    for line in sentinel_lines:
        try:
            parsed.append(_parse_sentinel(line))
        except Exception as e:                                      # noqa: BLE001
            result.update(status=PROBE_ERROR,
                          reason=f"unparseable_sentinel_record:{type(e).__name__}:{e}")
            return result
    pids = {rec.get("pid") for rec in parsed}
    if len(pids) != 1 or None in pids:
        result.update(status=PROBE_OK,
                      reason=f"ambiguous_sentinel_pids:{sorted(map(str, pids))}")
        return result
    pid = str(next(iter(pids)))
    result["pid"] = pid

    # The remote `sed` lift is CHECKED, not trusted: the /proc block was gathered
    # for the pid the shell extracted, so if that disagrees with the JSON the
    # evidence describes a process nobody asked about.
    if pid not in procs:
        result.update(status=PROBE_ERROR,
                      reason=f"pid_evidence_missing_for_sentinel_pid:{pid} "
                             f"(probe collected {sorted(procs)})")
        return result

    rec = parsed[0]
    # the sentinel must be THIS identity's — a log file holding another worker's
    # record proves nothing about this one
    if rec.get("worker_id") != target["worker_id"]:
        result.update(status=PROBE_OK,
                      reason=f"sentinel_identity_mismatch:"
                             f"{rec.get('worker_id')!r}!={target['worker_id']!r}")
        return result
    if str(rec.get("gpu_id")) != str(target["gpu"]):
        result.update(status=PROBE_OK,
                      reason=f"sentinel_gpu_mismatch:"
                             f"{rec.get('gpu_id')!r}!={target['gpu']!r}")
        return result
    if str(rec.get("run_nonce")) != str(nonce):
        result.update(status=PROBE_OK,
                      reason=f"sentinel_nonce_mismatch:{rec.get('run_nonce')!r}")
        return result

    # (10)(11) released or aborted -> not parked, whatever a process count says
    for label, value in (("released", released), ("aborted", aborted)):
        if value is None or not value.isdigit():
            result.update(status=PROBE_ERROR,
                          reason=f"unparseable_{label}_count:{value!r}")
            return result
        if int(value):
            result.update(
                status=PROBE_OK,
                reason=f"worker_already_{label} ({label}_records={value}) — the "
                       f"process is no longer parked at the barrier")
            return result

    # (3) the PID exists right now
    argv_raw = procs[pid]
    if argv_raw == "ABSENT":
        result.update(status=PROBE_OK,
                      reason=f"pid_not_present:{pid} — the sentinel was "
                             f"delivered and the process is GONE")
        return result
    # (4) read its argv. A live process with an empty cmdline is a zombie or a
    # kernel thread: it exists and cannot be identified, which is a refusal, not
    # a pass by default.
    argv = [a for a in argv_raw.split(ARGV_SEP) if a != ""]
    result["argv"] = argv
    if not argv:
        result.update(status=PROBE_OK,
                      reason=f"pid_argv_unreadable:{pid} — process exists but "
                             f"its identity cannot be read (zombie/reaped)")
        return result

    # (5) it is the worker
    if not any(a.endswith(WORKER_ENTRYPOINT_BASENAME) for a in argv):
        result.update(status=PROBE_OK,
                      reason=f"pid_reused_by_unrelated_process:{pid} "
                             f"argv={' '.join(argv)[:120]!r}")
        return result
    # (6)(7)(8) nonce, gpu and the nonced barrier arguments, as ADJACENT argv
    # pairs — never substring tests, or `--gpu-id 3` is satisfied by `30`.
    pairs = {(argv[i], argv[i + 1]) for i in range(len(argv) - 1)}
    if ("--run-nonce", str(nonce)) not in pairs:
        result.update(status=PROBE_OK,
                      reason=f"wrong_or_missing_run_nonce_in_argv:{pid}")
        return result
    if ("--gpu-id", str(target["gpu"])) not in pairs:
        result.update(status=PROBE_OK,
                      reason=f"wrong_or_missing_gpu_id_in_argv:{pid} "
                             f"(expected {target['gpu']})")
        return result
    release_args = [v for k, v in pairs if k == "--session-release-file"]
    if not release_args:
        result.update(status=PROBE_OK,
                      reason=f"no_session_release_file_argument:{pid} — this "
                             f"process is not parked at any barrier")
        return result
    if not any(str(nonce) in v for v in release_args):
        result.update(status=PROBE_OK,
                      reason=f"barrier_file_is_not_this_run's:{release_args!r}")
        return result

    # [R1, BETA BINDING] EVERYTHING ABOVE PROVES THE WORKER IS CONFIGURED TO
    # PARK. This proves it ARRIVED.
    #
    # The distinction is the whole correction: `--session-release-file` in argv
    # is a property of how the process was launched, and the interval between
    # emitting the sentinel and entering `await_session_release` is real. A gate
    # that printed ALIVE+PARKED on the strength of the argument was displaying a
    # state stronger than the one it measured — the eighth instance in this arc
    # of a verdict exceeding its evidence, and the one Beta caught in review.
    #
    # DELIBERATELY LAST, so the refusal an operator reads is the most specific
    # one available: a worker that died before parking reports `pid_not_present`
    # (checked above) rather than the vaguer "never parked", and a worker that is
    # demonstrably alive and correct but has not yet arrived reports exactly
    # that.
    #
    # `>= 1`, NOT exactly-once. There is one emission site
    # (`range_miner_worker.py:1499`) and one production call site (`:2168`), so
    # in production it is emitted once per process — but that is an observation
    # about the current callers, not a guarantee the contract makes. Nothing in
    # source forbids a second `await_session_release`, so exactly-once is not
    # encoded here.
    if not wait_lines:
        result.update(
            status=PROBE_OK,
            reason=f"not_parked_no_release_wait_record:{pid} — the process is "
                   f"alive and correctly configured for the barrier, but has "
                   f"emitted no SESSION_RELEASE_WAIT for this run, so it has "
                   f"NOT been observed to reach it")
        return result
    # [R2, BETA BINDING] THE RECORD IS PARSED AND COMPARED, NOT GREPPED.
    #
    # The remote filter is `grep 'SESSION_RELEASE_WAIT' | grep '<nonce>'` — TEXT
    # CONTAINMENT, not semantic equality. A record whose authoritative
    # `run_nonce` belongs to ANOTHER run still passes that filter whenever the
    # current nonce appears anywhere in the line, and `release_path` is exactly
    # such a place:
    #
    #     {"event": "SESSION_RELEASE_WAIT", "worker_id": "rrig6600:gpu3",
    #      "run_nonce": "some-other-run",
    #      "release_path": "/tmp/gate12_release_<CURRENT_NONCE>"}
    #
    # Correct worker, current nonce present as text, and it is not this run's
    # wait. That would have satisfied R1's "current-run WAIT" property while
    # violating it — the ninth instance in this arc of a check passing on a fact
    # it does not verify, this time a text match standing in for equality. The
    # sentinel half already reparses and compares exactly; this gives the wait
    # half the same discipline, from evidence the probe already gathers.
    #
    # `release_path` is compared against the LIVE PROCESS'S OWN
    # `--session-release-file` argv value, which is what makes this a
    # correlation rather than a second self-consistent story: the log record and
    # the running process must name the same barrier file.
    valid, rejects = 0, []
    for line in wait_lines:
        try:
            rec = _parse_sentinel(line)
        except Exception as e:                                      # noqa: BLE001
            result.update(status=PROBE_ERROR,
                          reason=f"unparseable_release_wait_record:"
                                 f"{type(e).__name__}:{e}")
            return result
        if rec.get("event") != "SESSION_RELEASE_WAIT":
            rejects.append(f"release_wait_wrong_event:{rec.get('event')!r}")
        elif str(rec.get("run_nonce")) != str(nonce):
            rejects.append(f"release_wait_nonce_mismatch:"
                           f"{rec.get('run_nonce')!r}!={nonce!r} (the current "
                           f"nonce appears in the line as TEXT only)")
        elif rec.get("worker_id") != target["worker_id"]:
            rejects.append(f"release_wait_identity_mismatch:"
                           f"{rec.get('worker_id')!r}!={target['worker_id']!r}")
        elif rec.get("release_path") not in release_args:
            rejects.append(f"release_wait_path_mismatch:"
                           f"{rec.get('release_path')!r} is not the live "
                           f"process's --session-release-file {release_args!r}")
        else:
            valid += 1
    # `>= 1` valid, AND no invalid record present.
    #
    # DECLARED DEVIATION FROM THE LITERAL BRIEF: Beta wrote "then >= 1 such valid
    # record is sufficient", which read alone would let an anomalous record sit
    # beside a valid one and be ignored. Beta also wrote "EVERY accepted WAITREC
    # must satisfy all four", and this takes that reading, which is also R1's
    # all-quantifier ("all matching current-nonce WAIT records identify the same
    # worker") preserved rather than narrowed. Stale same-worker records from an
    # earlier run cannot reach here — they carry the old nonce throughout, so the
    # remote text filter drops them — which means a record that arrives here and
    # fails validation is genuinely anomalous, and this gate is fail-closed at a
    # point where refusing costs nothing. If Beta prefers the literal filter
    # reading, deleting the `rejects` branch below is the whole change.
    if rejects:
        result.update(status=PROBE_OK,
                      reason=f"{rejects[0]} (valid_wait_records={valid})")
        return result
    if not valid:
        result.update(status=PROBE_OK,
                      reason="no_valid_current_run_release_wait_record")
        return result
    result["release_waits"] = valid

    result.update(status=PROBE_OK, reason=None)
    return result


def evaluate(results):
    """Fail-close, and the default is refusal: a status this function does not
    recognise is not a pass. Returns (allowed, refusals).

    The cross-identity rule lives here because it cannot be seen from inside one
    probe: two identities whose sentinels name the SAME pid ON THE SAME HOST are
    one process claiming to be two workers, and the per-identity checks would
    both pass. It is scoped per host deliberately — PIDs are per kernel, so
    rrig6600 and rrig6600b legitimately share numbers, and a global uniqueness
    test would refuse a correct 25-worker fleet roughly always.
    """
    refusals = []
    seen = {}
    for r in results:
        if r["status"] == PROBE_OK and r["reason"] is None:
            key = (r["endpoint"], r["pid"])
            if key in seen:
                refusals.append(
                    f"{r['worker_id']}: duplicate_pid — pid {r['pid']} on "
                    f"{r['endpoint']} is also claimed by {seen[key]}; one "
                    f"process cannot be two parked workers")
                continue
            seen[key] = r["worker_id"]
            continue
        if r["status"] == PROBE_OK:
            refusals.append(f"{r['worker_id']} ({r['log_path']}): {r['reason']}")
        elif r["status"] == PROBE_UNAVAILABLE:
            refusals.append(
                f"{r['worker_id']}: {PROBE_UNAVAILABLE} (reason={r['reason']}) "
                f"— the probe did not run, so liveness is UNKNOWN, not absent")
        elif r["status"] == PROBE_ERROR:
            refusals.append(
                f"{r['worker_id']}: {PROBE_ERROR} (reason={r['reason']}) — the "
                f"probe ran but its output could not be classified")
        else:
            refusals.append(
                f"{r['worker_id']}: unrecognized probe status {r['status']!r} — "
                f"refusing by default")
    return (not refusals), refusals


def render(r):
    """An UNAVAILABLE or ERROR identity NEVER renders count-shaped."""
    if r["status"] == PROBE_OK and r["reason"] is None:
        # `wait=` is shown because it is the record that entitles this line to
        # say PARKED. A verdict should render the evidence it rests on.
        return (f"  {r['worker_id']:<22} {'ALIVE+PARKED':<14} "
                f"pid={r['pid']:<8} wait={r['release_waits']}  "
                f"sentinel->pid->argv->wait joined")
    if r["status"] == PROBE_OK:
        return (f"  {r['worker_id']:<22} {'NOT PARKED':<14} "
                f"pid={r['pid'] or '-':<8} {r['reason']}")
    return (f"  {r['worker_id']:<22} {r['status']:<14} "
            f"liveness={r['status']} — reason={r['reason']}"
            + (f" stderr={r['stderr']!r}" if r["stderr"] else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-nonce", required=True)
    ap.add_argument("--rig-profile", default="proxmox")
    ap.add_argument("--admission-count", type=int, default=25)
    ap.add_argument("--remote-log-dir", default=SG.DEFAULT_REMOTE_LOG_DIR)
    ap.add_argument("--local-log-dir", required=True)
    ap.add_argument("--evidence-json", default=None,
                    help="write the per-identity evidence rows here")
    args = ap.parse_args(argv)

    targets = worker_targets(args.rig_profile, args.admission_count)
    print("=" * 70)
    print("GATE-12 SENTINEL-CORRELATED WORKER LIVENESS")
    print("=" * 70)
    print(f"run nonce         : {args.run_nonce}")
    print(f"profile           : {args.rig_profile}")
    print(f"expected identities: {len(targets)} (derived from the execution set)")
    print("")

    results = [
        probe_liveness(t, args.run_nonce,
                       SG.log_path_for(t, args.remote_log_dir,
                                       args.local_log_dir))
        for t in targets
    ]
    for r in results:
        print(render(r))
    print("")

    if args.evidence_json:
        try:
            with open(args.evidence_json, "w", encoding="utf-8") as fh:
                json.dump({"run_nonce": args.run_nonce,
                           "rig_profile": args.rig_profile,
                           "expected": len(targets),
                           "results": results}, fh, indent=2, sort_keys=True)
            print(f"evidence written  : {args.evidence_json}")
        except OSError as e:
            # Evidence that cannot be written does not change the verdict, and
            # must not silently look like evidence that was.
            print(f"EVIDENCE WRITE FAILED: {e}")

    allowed, refusals = evaluate(results)
    if allowed:
        print(f"GATE-12 LIVENESS  : PASS — {len(results)}/{len(results)} "
              f"identities are sentinel-correlated, ALIVE and still PARKED.")
        print("During the final pre-coordinator liveness sweep, every expected")
        print("sentinel-correlated identity was observed alive and parked; the")
        print("sweep completed immediately before coordinator creation.")
        print("The identities are probed SEQUENTIALLY, so that is a sweep and")
        print("not a simultaneous snapshot, and no probe can promise the next")
        print("microsecond — the 25-worker admission wall remains the runtime")
        print("authority once registration begins.")
        return EXIT_PROCEED

    print(f"GATE-12 LIVENESS  : REFUSED — {len(refusals)} of {len(results)} "
          f"identity(ies) are not proven alive and parked:")
    for line in refusals:
        print(f"  * {line}")
    print("")
    print("ABORTING BEFORE THE COORDINATOR EXISTS. A sentinel PASS does not")
    print("authorize a launch: in D6 dry run #2 the sentinel gate passed 25/25")
    print("over a worker that was already dead, and the cohort would have frozen")
    print("at 25 and admitted 24 — attempt 2's worker_admission_timeout,")
    print("manufactured by the launch harness rather than by the fleet.")
    print("NO REDUCED COHORT AND NO AUTOMATIC DOWNSIZING.")
    return EXIT_REFUSE


if __name__ == "__main__":
    sys.exit(main())
