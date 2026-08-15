#!/usr/bin/env python3
"""GATE-12 WORKER-LOG SENTINEL GATE — a harness rule, and a real barrier.

WHAT THIS IS
------------
Across gate-12 attempts 4 and 5, **no rig-worker session event was observably
emitted on the production logging surface**: all 24 rig logs in the attempt-5
frozen bundle are byte-identical 138-byte files carrying only three
`[sieve_worker] Compiled kernel:` lines, while the identical code emitted
correctly on the Zeus-local worker in the same run. Buffering was REFUTED by a
positive control run with the exact production invocation shape.
**The cause of the silence is UNRESOLVED, and nothing here claims otherwise.**

A sentinel does not need to know why the channel was silent. It needs to prove,
BEFORE the run commits any GPU-seconds or freezes a cohort, that a record written
by the worker THROUGH THE SAME CALL PATH the session events use arrives at a
place the operator can read. Whatever broke it — an environment difference, a
stream replacement, a redirect that did not survive, a launch that produced a
different file than the one collected — a sentinel that fails closed catches all
of them without naming any of them.

WHY HARNESS-SIDE AND NOT COORDINATOR-SIDE
-----------------------------------------
A coordinator-side check can only verify the worker's CLAIM that it emitted (an
extra field on `RegisterMessage` saying "I logged the nonce"). The thing that
failed is DELIVERY, and only an out-of-band reader of the file can witness
delivery. The harness-side probe is therefore not the weaker-but-smaller option;
it is the one that measures the property in question. The REGISTER echo is the
weaker mechanism despite being the more invasive one, and it also risks
re-opening Defect A §13: `IDENTITY_FIELDS` is an enumerated allowlist precisely
because a whole-message compare carries a per-message timestamp, and a new field
that entered it would make the frozen-cohort comparison depend on a per-session
nonce.

THE TWO PHASES, AND WHY THE ORDER IS THE POINT
----------------------------------------------
    verify   every eligible worker's log must contain a SESSION_SENTINEL record
             CARRYING THIS run's nonce — one record satisfying both facts, never
             one record per fact (R1-A)
    release  write the per-host release token so the parked workers may REGISTER

A sentinel verified AFTER register proves the channel was alive for a run that
has already committed GPU-seconds — an intention, not an enforced ordering
property. `--phase verify` runs with the fleet parked at the barrier and the
coordinator not yet started; `--phase release` runs only after the coordinator is
listening. Verification therefore happens OUTSIDE the 180 s admission window, so
a slow probe sweep cannot spend the admission budget and manufacture attempt 2's
terminal with the fix.

THE RULE — fail-close on anything that is not a truthful full count. Every arm
below is decided by ONE number: the count of records carrying the sentinel event
AND this run's nonce TOGETHER.

    OK, >=1 such record on EVERY eligible worker log  -> PROCEED
    any log with no such record                       -> REFUSE
    any log whose sentinels carry a PREVIOUS nonce    -> REFUSE
    any log with this nonce but on some OTHER event   -> REFUSE  (R1-A)
    any log with no sentinel record at all            -> REFUSE  (R1-A)
    UNAVAILABLE — THE PROBE DID NOT RUN               -> REFUSE
        ssh transport failure (ssh exit 255)                    (R2)
        ssh/probe timeout
        the log file is unreadable
    ERROR — the probe RAN, its output is unclassifiable -> REFUSE
        e.g. a remotely executed probe returning malformed stdout

UNAVAILABLE is reported AS UNAVAILABLE and NEVER as a count. "The probe could not
run" and "the probe ran and saw zero sentinels" are different facts, and a gate
that renders the first as `0/8` has destroyed exactly the distinction that let
`GPU_COUNT_MISMATCH: 0/8` sail through a 3/3 preflight in attempt 1. There is no
advisory treatment here and no automatic downsizing: a refusal is a refusal, not
permission to launch with fewer workers.

Exit status: 0 = proceed, 1 = REFUSE. The caller aborts on non-zero.

Nothing is hardcoded: the fleet, its endpoints, its per-node GPU counts and the
ssh user are derived from the committed execution set (`rig_profiles_config.json`
joined with `distributed_config.json`) — the same authority the run itself
resolves its fleet from.
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The three-outcome vocabulary, imported from the certified GPU probe rather than
# re-declared: one vocabulary, one meaning, and a rename cannot leave two copies
# disagreeing about what UNAVAILABLE is.
import preflight_check as PF                                        # noqa: E402

PROBE_OK = PF.GPU_PROBE_OK
PROBE_UNAVAILABLE = PF.GPU_PROBE_UNAVAILABLE
PROBE_ERROR = PF.GPU_PROBE_ERROR

SSH_CONNECT_TIMEOUT = PF.SSH_TIMEOUT_SECONDS
PROBE_TIMEOUT = PF.GPU_CHECK_TIMEOUT_SECONDS

# [R2] ssh returns this for ITS OWN failure — connect timeout, no route,
# host-key mismatch, BatchMode auth refusal — and THIS GATE RESERVES THE VALUE as
# its remote-transport classification under the current probe script. Not a
# protocol guarantee: ssh passes a remote command's status through unchanged, so
# the reservation holds only because the script this gate sends cannot produce
# 255. A named constant rather than a literal at the comparison, so the gate
# asserts on the name and a reviewer sees the claim being made. See
# `probe_sentinel` for why this is one reserved value and not "any nonzero".
SSH_TRANSPORT_FAILURE_STATUS = 255

EXIT_PROCEED = 0
EXIT_REFUSE = 1

# Log-path conventions, matching `scripts/launch_fleet_manual.sh`. They are
# arguments with defaults rather than constants in the body, so a harness that
# redirects elsewhere passes the path instead of editing this file.
DEFAULT_REMOTE_LOG_DIR = "/tmp/minerlogs"
DEFAULT_REMOTE_RELEASE_DIR = "/tmp/minerlogs"


def worker_targets(rig_profile="proxmox", admission_count=25):
    """Every eligible worker identity, with the host it runs on.

    DERIVED, never transcribed. `resolve_execution_set` joins the rig profile map
    with `distributed_config.json`; the identities are exactly
    `<worker_hostname>:gpu<n>`, the same strings the miner's own `worker_id` is
    built from, so no translation layer can drift between this gate and the run.
    """
    import execution_set as XS
    s = XS.resolve_execution_set(
        backend="miner", invoked_by="gate12_sentinel_gate",
        rig_profile=rig_profile, admission_count=admission_count)
    out = []
    for n in s.nodes:
        for gpu in range(n.gpu_count):
            out.append({
                "node_id": n.node_id,
                "endpoint": n.endpoint,
                "ssh_user": n.ssh_user,
                "local": n.local,
                "gpu": gpu,
                "worker_hostname": n.worker_hostname,
                "worker_id": f"{n.worker_hostname}:gpu{gpu}",
            })
    return out


def log_path_for(target, remote_log_dir, local_log_dir):
    if target["local"]:
        # `<worker_hostname>_gpu<N>.log`, which is the name
        # `scripts/launch_fleet_manual.sh` writes for the local node — the worker
        # hostname, NOT the logical node_id. A probe reading the wrong filename
        # would report UNAVAILABLE for a worker that logged perfectly.
        return os.path.join(local_log_dir,
                            f"{target['worker_hostname']}_gpu{target['gpu']}.log")
    return f"{remote_log_dir}/gpu{target['gpu']}.log"


def release_path_for(target, nonce, remote_release_dir, local_log_dir):
    """PER HOST, and the nonce is in BOTH the name and the content.

    The four hosts share no filesystem, so the token is written once per host.
    The nonce in the content is what makes a stale token from an earlier run
    unable to release this one — the worker compares it — and the nonce in the
    name means a stale token is not even found."""
    base = f"gate12_release_{nonce}"
    if target["local"]:
        return os.path.join(local_log_dir, base)
    return f"{remote_release_dir}/{base}"


def _run(cmd, timeout):
    return subprocess.run(cmd, capture_output=True, timeout=timeout)


def probe_sentinel(target, nonce, log_path):
    """One worker, one probe, exactly three possible outcomes.

    [R1-A, BETA BINDING] ACCEPTANCE IS A SAME-RECORD CONJUNCTION. The count that
    decides this worker is

        lines( record contains SESSION_SENTINEL  AND  that same record
               contains THIS run's nonce )

    produced by ONE pipeline, `grep SESSION_SENTINEL | grep -c <nonce>`, so the
    two facts cannot be satisfied by two different records.

    WHY THAT MATTERS, AND WHY IT IS NOT A MALFORMED-LOG HYPOTHETICAL. The
    previous shape ran two INDEPENDENT counts and accepted on the nonce count
    alone, proving *"a SESSION_SENTINEL exists somewhere AND this nonce exists
    somewhere"* when the contract is *"a SESSION_SENTINEL CARRYING this nonce"*.
    The worker enters the release barrier immediately after emitting the
    sentinel, and the barrier's own `SESSION_RELEASE_WAIT` carries the same run
    nonce (`range_miner_worker.py:1498-1500`), so a log reading

        [old run] SESSION_SENTINEL     ... nonce=OLD
        [new run] SESSION_RELEASE_WAIT ... nonce=CURRENT

    satisfied both counts although THIS run's sentinel was never observed —
    defeating the one property this gate exists to establish. The stale-nonce arm
    did not catch it: that arm tests an old sentinel with NO current nonce, not
    the split-fact case.

    `sentinel_lines_any_nonce` is DIAGNOSTIC ONLY and is deliberately named so
    that its role cannot be misread. It exists to tell an operator which refusal
    they are looking at — a sentinel present but stale, versus no sentinel at
    all — and `evaluate()` never reads it. That separation is itself gated
    (RXP-3 arm 11), because "a count that is only for the message" is exactly the
    kind of claim that decays into an acceptance input.

    `grep -c` prints 0 AND exits 1 when there is no match, and a `|| echo 0`
    would then print a SECOND zero — the two constructs that between them
    manufactured attempt 1's `0/8`. So the count is read from stdout and the exit
    status of a no-match grep is NOT treated as a probe failure; only ssh
    itself failing, or output that cannot be parsed, produce UNAVAILABLE/ERROR.
    A pipeline reports the LAST command's status, so the conjunctive count
    behaves identically here.

    [R2, BETA BINDING] SSH TRANSPORT FAILURE IS `UNAVAILABLE`, NOT `ERROR`.
    An ordinary connectivity or authentication failure is NOT an exception from
    `subprocess.run`: it returns a COMPLETED process carrying a nonzero ssh
    status and diagnostic stderr. Before this correction nothing here read
    `proc.returncode`, so such a failure produced empty stdout, fell through to
    the two-line check and was reported `ERROR: unparseable_probe_output` — i.e.
    *"the probe ran and its output could not be classified"* about a probe that
    never ran at all. **Both outcomes refuse, so there was no safety
    consequence; the consequence was evidentiary**, and it is the same class as
    R1-A: a gate declaring A+B while exercising only B.

    THE RULE, AND WHY IT IS 255 AND NOT "ANY NONZERO":
    ssh returns **255** for its own failure — connect timeout, no route, host-key
    mismatch, `BatchMode` auth refusal — and **this gate reserves that value as
    its remote-transport classification under the current probe script.** That is
    a decision about this gate, not a protocol guarantee: ssh passes a remote
    command's status through unchanged, so a remote command CAN exit 255 and be
    reported as a transport failure here. The reservation is safe because the
    script this gate sends cannot produce it — a shell `if` whose taken branch
    ends in `head -1` or `echo` — and **if that script changes, this rule must be
    revisited**, which is why the two live next to each other in this function.

    The converse, classifying every nonzero as transport, is what Beta forbade:
    the script's taken branch could legitimately carry a nonzero status if it
    ever changed, and a "no match" would then be reported as "the fleet is
    unreachable".

    THE CERTIFIED GPU PROBE USES THE OPPOSITE RULE, AND THE DIFFERENCE IS
    DELIBERATE. `preflight_check.py:512` treats **any** nonzero as
    `ssh_exit_<rc>` → UNAVAILABLE. It can, because
    `_build_gpu_probe_script` designs the ambiguity out: every internal failure
    branch ends in an explicit `exit 0`, so a nonzero status there can only have
    come from ssh. This gate's script carries no such guarantee, so it reserves
    one value instead of every nonzero one. The reason string keeps the certified
    `ssh_exit_<rc>` token so one grep finds both.

    The check is gated to the REMOTE branch. The local branch runs `bash -c`,
    where 255 would be an ordinary command status and carries no transport
    meaning.
    """
    result = {"worker_id": target["worker_id"], "endpoint": target["endpoint"],
              "log_path": log_path, "status": None, "count": None,
              "sentinel_lines_any_nonce": None,
              "reason": None, "stderr": ""}
    # One argv element: ssh joins trailing arguments with spaces without
    # re-quoting them, so a pipeline passed as several words is re-parsed
    # remotely with the quoting gone.
    script = (f"if [ -r {log_path} ]; then "
              f"grep 'SESSION_SENTINEL' {log_path} | grep -c '{nonce}' "
              f"| head -1; "
              f"grep -c 'SESSION_SENTINEL' {log_path} | head -1; "
              f"else echo UNREADABLE; fi")
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
        # A hung probe OBSERVED NOTHING. It is emphatically not a zero.
        result.update(status=PROBE_UNAVAILABLE, reason="timeout")
        return result
    except Exception as e:                                          # noqa: BLE001
        result.update(status=PROBE_UNAVAILABLE,
                      reason=f"{type(e).__name__}:{e}")
        return result

    result["stderr"] = proc.stderr.decode(errors="replace").strip()
    # [R2] BEFORE anything is read from stdout: if ssh itself failed, whatever
    # arrived on stdout is not this probe's output and must not be classified as
    # though it were.
    if not target["local"] and proc.returncode == SSH_TRANSPORT_FAILURE_STATUS:
        result.update(
            status=PROBE_UNAVAILABLE,
            reason=f"ssh_transport_failure (ssh_exit_{proc.returncode})")
        return result
    out = proc.stdout.decode(errors="replace").strip().splitlines()
    if "UNREADABLE" in out:
        result.update(status=PROBE_UNAVAILABLE, reason="log_unreadable")
        return result
    if len(out) != 2:
        result.update(status=PROBE_ERROR,
                      reason=f"unparseable_probe_output:{out!r}")
        return result
    try:
        # out[0] IS the acceptance count and the only number `evaluate()` sees;
        # out[1] is the diagnostic.
        conjunctive, sentinel_lines = int(out[0]), int(out[1])
    except ValueError:
        result.update(status=PROBE_ERROR,
                      reason=f"non_numeric_counts:{out!r}")
        return result
    if conjunctive:
        reason = None
    elif sentinel_lines:
        # The split-fact case, named as itself: sentinels are being written to
        # this file, but none of them belongs to this run.
        reason = (f"sentinel_present_but_none_carries_this_nonce "
                  f"(sentinel_lines_any_nonce={sentinel_lines})")
    else:
        reason = "no_sentinel_record_at_all (sentinel_lines_any_nonce=0)"
    result.update(status=PROBE_OK, count=conjunctive,
                  sentinel_lines_any_nonce=sentinel_lines, reason=reason)
    return result


def render(r):
    """One line per worker. An UNAVAILABLE or ERROR worker NEVER renders
    count-shaped: `0/1` and `UNAVAILABLE` must not be confusable by the operator
    reading the evidence block."""
    if r["status"] == PROBE_OK:
        # A zero renders its DIAGNOSTIC reason, so the operator can tell the two
        # R1-A refusals apart at a glance: sentinels present but none for this
        # run, versus no sentinel record at all.
        verdict = ("OK" if r["count"]
                   else f"NO SENTINEL FOR THIS NONCE — {r['reason']}")
        return (f"  {r['worker_id']:<22} {r['status']:<12} "
                f"{r['count']}/1  {verdict}")
    return (f"  {r['worker_id']:<22} {r['status']:<12} "
            f"count={r['status']} — reason={r['reason']}"
            + (f" stderr={r['stderr']!r}" if r["stderr"] else ""))


def evaluate(results):
    """Fail-close. Returns (allowed, refusals). The default is refusal: a status
    this function does not recognise falls through to "not allowed"."""
    refusals = []
    for r in results:
        if r["status"] == PROBE_OK and r["count"]:
            continue
        if r["status"] == PROBE_OK:
            refusals.append(
                f"{r['worker_id']} ({r['log_path']}): no SESSION_SENTINEL line "
                f"carrying this run's nonce — {r['reason']}")
        elif r["status"] == PROBE_UNAVAILABLE:
            refusals.append(
                f"{r['worker_id']}: {PROBE_UNAVAILABLE} (reason={r['reason']}) "
                f"— the probe did not run, so delivery is UNKNOWN, not absent")
        elif r["status"] == PROBE_ERROR:
            refusals.append(
                f"{r['worker_id']}: {PROBE_ERROR} (reason={r['reason']}) — the "
                f"probe ran but its output could not be classified")
        else:
            refusals.append(
                f"{r['worker_id']}: unrecognized probe status {r['status']!r} — "
                f"refusing by default")
    return (not refusals), refusals


def write_release(targets, nonce, remote_release_dir, local_log_dir):
    """Write the per-host release token. ONE write per HOST, not per worker: the
    workers on a host share a filesystem and every one of them is parked on the
    same path."""
    by_host = {}
    for t in targets:
        by_host.setdefault((t["endpoint"], t["local"], t["ssh_user"]), t)
    written, failed = [], []
    for (endpoint, local, ssh_user), t in sorted(by_host.items()):
        path = release_path_for(t, nonce, remote_release_dir, local_log_dir)
        try:
            if local:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(nonce)
                ok = True
                err = ""
            else:
                proc = _run(
                    ["ssh", "-n", "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT}",
                     "-o", "BatchMode=yes", f"{ssh_user}@{endpoint}",
                     f"mkdir -p {remote_release_dir} && "
                     f"printf '%s' '{nonce}' > {path}"],
                    PROBE_TIMEOUT)
                ok = proc.returncode == 0
                err = proc.stderr.decode(errors="replace").strip()
        except Exception as e:                                      # noqa: BLE001
            ok, err = False, f"{type(e).__name__}:{e}"
        (written if ok else failed).append((endpoint, path, err))
    return written, failed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=("verify", "release"), required=True)
    ap.add_argument("--run-nonce", required=True)
    ap.add_argument("--rig-profile", default="proxmox")
    ap.add_argument("--admission-count", type=int, default=25)
    ap.add_argument("--remote-log-dir", default=DEFAULT_REMOTE_LOG_DIR)
    ap.add_argument("--remote-release-dir", default=DEFAULT_REMOTE_RELEASE_DIR)
    ap.add_argument("--local-log-dir", required=True)
    args = ap.parse_args(argv)

    targets = worker_targets(args.rig_profile, args.admission_count)
    print("=" * 70)
    print(f"GATE-12 WORKER-LOG SENTINEL — phase {args.phase.upper()}")
    print("=" * 70)
    print(f"run nonce         : {args.run_nonce}")
    print(f"profile           : {args.rig_profile}")
    print(f"eligible workers  : {len(targets)} (derived from the execution set)")
    print("")

    if args.phase == "release":
        written, failed = write_release(
            targets, args.run_nonce, args.remote_release_dir, args.local_log_dir)
        for endpoint, path, _ in written:
            print(f"  released  {endpoint:<16} {path}")
        for endpoint, path, err in failed:
            print(f"  FAILED    {endpoint:<16} {path}  {err}")
        if failed:
            print("")
            print(f"GATE-12 RELEASE   : FAILED on {len(failed)} host(s). The "
                  f"parked workers will fail closed at their release deadline "
                  f"rather than registering — which is the designed outcome, not "
                  f"a second failure.")
            return EXIT_REFUSE
        print("")
        print(f"GATE-12 RELEASE   : WRITTEN on {len(written)} host(s). The "
              f"fleet may now REGISTER.")
        return EXIT_PROCEED

    results = [probe_sentinel(t, args.run_nonce,
                              log_path_for(t, args.remote_log_dir,
                                           args.local_log_dir))
               for t in targets]
    for r in results:
        print(render(r))
    print("")
    allowed, refusals = evaluate(results)
    if allowed:
        print(f"GATE-12 SENTINEL  : PASS — {len(results)}/{len(results)} "
              f"identities proved log delivery. Launch may proceed.")
        return EXIT_PROCEED

    print(f"GATE-12 SENTINEL  : REFUSED — {len(refusals)} identity(ies) did not "
          f"prove session-log delivery:")
    for line in refusals:
        print(f"  * {line}")
    print("")
    print("ABORTING. NO REDUCED COHORT AND NO AUTOMATIC DOWNSIZING: a gate")
    print("refusal is a refusal, not permission to launch with fewer workers.")
    print("The fleet is parked at the release barrier and has sent nothing to a")
    print("coordinator, so nothing has been committed and the attempt is not")
    print("consumed — kill the fleet, fix the channel, and launch again.")
    return EXIT_REFUSE


if __name__ == "__main__":
    sys.exit(main())
