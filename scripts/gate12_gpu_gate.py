#!/usr/bin/env python3
"""GATE-12 GPU FAIL-CLOSE GATE — a harness rule, not a preflight policy change.

WHAT THIS IS
------------
Gate 12 is a saturation claim about 24 rig GPUs. A run that begins without them
cannot produce that evidence, and the attempt-1 forensics recorded a `0/8` GPU
reading that did not stop anything — because the generic `PreflightChecker`
reports GPU findings through `add_warning` and is NON-BLOCKING BY DESIGN
(`preflight_check.check_gpu_health`, and the WATCHER gate asserted in
`tests/test_s172_resolved_execution_set.py`). That advisory policy is correct
for WATCHER and is deliberately NOT changed here. This module adds a rule that
applies to the GATE-12 HARNESS ONLY: `gate12_launch.sh` refuses to launch unless
all three rigs truthfully report OK with the expected device count.

WHY IT REUSES THE CERTIFIED PROBE
---------------------------------
`_build_gpu_probe_script` and `_parse_gpu_probe` are imported from
`preflight_check`, not reimplemented. A second probe with its own parsing would
be a second place for the `|| echo 0` class of defect to live — the exact defect
that made a failed probe indistinguishable from an observed zero. There is one
probe and one classifier; this module only decides what to DO with the outcome.

THE RULE — fail-close on anything that is not a truthful full count:

    OK   and count == expected   on ALL three rigs -> PROCEED
    OK   and count != expected   on ANY rig        -> REFUSE  (incl. a real 0)
    UNAVAILABLE                  on ANY rig        -> REFUSE
    ERROR                        on ANY rig        -> REFUSE

UNAVAILABLE is reported AS UNAVAILABLE and never as a count. "The probe could
not run" and "the probe ran and saw zero devices" are different facts, and a
gate that renders the first as `0/8` has destroyed the distinction the
three-outcome probe exists to preserve.

Exit status: 0 = proceed, 1 = REFUSE. The caller aborts on non-zero BEFORE the
sampler is armed and before any coordinator process is created.

No value here is hardcoded: the rig endpoints and their expected device counts
are derived from the committed execution set (`rig_profiles_config.json` joined
with `distributed_config.json`), which is the same authority the run itself
resolves its fleet from.
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import preflight_check as PF                                        # noqa: E402

# The certified probe surface, imported rather than copied. Named here so the
# dependency is explicit and a rename cannot silently fall back to a local copy.
_build_gpu_probe_script = PF._build_gpu_probe_script
_parse_gpu_probe = PF._parse_gpu_probe
GPU_PROBE_OK = PF.GPU_PROBE_OK
GPU_PROBE_UNAVAILABLE = PF.GPU_PROBE_UNAVAILABLE
GPU_PROBE_ERROR = PF.GPU_PROBE_ERROR

SSH_CONNECT_TIMEOUT = PF.SSH_TIMEOUT_SECONDS
PROBE_TIMEOUT = PF.GPU_CHECK_TIMEOUT_SECONDS

EXIT_PROCEED = 0
EXIT_REFUSE = 1


def gate_targets(rig_profile="proxmox"):
    """The rigs to probe and the count each must report, from committed source.

    DERIVED, never transcribed. `resolve_execution_set` joins the rig profile
    map with `distributed_config.json`; `remote_nodes()` drops the local node,
    which owns no rig GPUs. If the profile map is re-pointed, this gate follows
    it — a gate probing addresses the run does not use would be worse than no
    gate, because it would pass while the fleet was somewhere else.
    """
    import execution_set as XS
    s = XS.resolve_execution_set(
        backend="miner", invoked_by="gate12_gpu_gate",
        rig_profile=rig_profile, admission_count=8)
    return [(n.node_id, n.endpoint, n.ssh_user, n.gpu_count)
            for n in s.remote_nodes()]


def probe_rig(endpoint, ssh_user, expected):
    """One rig, one probe, exactly three possible outcomes.

    Mirrors `check_gpu_health`'s invocation: the probe script is passed as ONE
    argv element, because ssh joins trailing arguments with spaces without
    re-quoting them.
    """
    script = _build_gpu_probe_script()
    result = {"endpoint": endpoint, "expected": expected,
              "status": None, "gpu_count": None, "reason": None,
              "binary": None, "stderr": ""}
    try:
        proc = subprocess.run(
            ["ssh",
             "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT}",
             "-o", "BatchMode=yes",
             f"{ssh_user}@{endpoint}",
             script],
            capture_output=True, timeout=PROBE_TIMEOUT)
    except subprocess.TimeoutExpired:
        # A hung probe observed nothing. It is emphatically not a zero.
        result.update(status=GPU_PROBE_UNAVAILABLE, reason="timeout")
        return result
    except Exception as e:                                          # noqa: BLE001
        result.update(status=GPU_PROBE_UNAVAILABLE,
                      reason=f"{type(e).__name__}:{e}")
        return result

    result["stderr"] = proc.stderr.decode(errors="replace").strip()
    if proc.returncode != 0:
        result.update(status=GPU_PROBE_UNAVAILABLE,
                      reason=f"ssh_exit_{proc.returncode}")
        return result

    parsed = _parse_gpu_probe(proc.stdout.decode(errors="replace"))
    result.update(status=parsed["status"], gpu_count=parsed["gpu_count"],
                  reason=parsed["reason"], binary=parsed["binary"])
    return result


def render_outcome(r):
    """One line per rig. An UNAVAILABLE or ERROR rig NEVER renders count-shaped.

    `0/8` and `UNAVAILABLE` must not be confusable by the operator reading the
    evidence block, so the un-observed outcomes carry no numerator at all.
    """
    if r["status"] == GPU_PROBE_OK:
        verdict = "OK" if r["gpu_count"] == r["expected"] else "COUNT MISMATCH"
        return (f"  {r['endpoint']:<16} {r['status']:<12} "
                f"{r['gpu_count']}/{r['expected']}  {verdict}")
    return (f"  {r['endpoint']:<16} {r['status']:<12} "
            f"count={r['status']} (expected {r['expected']}) — "
            f"reason={r['reason']}"
            + (f" stderr={r['stderr']!r}" if r["stderr"] else ""))


def evaluate(results):
    """Fail-close. Returns (allowed, refusals).

    A rig contributes a refusal unless it is OK AND at the expected count. The
    default is refusal: a status this function does not recognise falls through
    to "not allowed" rather than to "proceed".
    """
    refusals = []
    for r in results:
        if r["status"] == GPU_PROBE_OK and r["gpu_count"] == r["expected"]:
            continue
        if r["status"] == GPU_PROBE_OK:
            refusals.append(
                f"{r['endpoint']}: OK but reported {r['gpu_count']} of "
                f"{r['expected']} expected GPUs")
        elif r["status"] == GPU_PROBE_UNAVAILABLE:
            refusals.append(
                f"{r['endpoint']}: {GPU_PROBE_UNAVAILABLE} "
                f"(reason={r['reason']}) — the probe did not run, so the device "
                f"count is UNKNOWN, not zero")
        elif r["status"] == GPU_PROBE_ERROR:
            refusals.append(
                f"{r['endpoint']}: {GPU_PROBE_ERROR} (reason={r['reason']}) — "
                f"the probe ran but its output could not be classified")
        else:
            refusals.append(
                f"{r['endpoint']}: unrecognized probe status {r['status']!r} — "
                f"refusing by default")
    return (not refusals), refusals


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rig-profile", default="proxmox")
    args = ap.parse_args(argv)

    targets = gate_targets(args.rig_profile)
    print("=" * 70)
    print("GATE-12 GPU FAIL-CLOSE GATE — all rigs must report OK at full count")
    print("=" * 70)
    print(f"profile           : {args.rig_profile}")
    print(f"rigs (derived)    : {', '.join(e for _, e, _, _ in targets)}")
    print(f"expected per rig  : "
          f"{', '.join(f'{e}={c}' for _, e, _, c in targets)}")
    print("probe             : preflight_check._build_gpu_probe_script "
          "(certified; not reimplemented here)")
    print("")

    results = [probe_rig(endpoint, user, expected)
               for _nid, endpoint, user, expected in targets]
    for r in results:
        print(render_outcome(r))
    print("")

    allowed, refusals = evaluate(results)
    if allowed:
        print(f"GATE-12 GPU GATE  : PASS — {len(results)}/{len(results)} rigs "
              f"OK at full count. Launch may proceed.")
        return EXIT_PROCEED

    print(f"GATE-12 GPU GATE  : REFUSED — {len(refusals)} rig(s) did not "
          f"truthfully report a full device count:")
    for line in refusals:
        print(f"  * {line}")
    print("")
    print("ABORTING BEFORE THE SAMPLER IS ARMED AND BEFORE ANY COORDINATOR")
    print("PROCESS IS CREATED. Gate 12 is a saturation claim about the rig")
    print("GPUs; a run that starts without them cannot produce that evidence,")
    print("and a partial run would consume the attempt while proving nothing.")
    return EXIT_REFUSE


if __name__ == "__main__":
    sys.exit(main())
