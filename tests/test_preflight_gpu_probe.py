#!/usr/bin/env python3
"""
GATE SUITE — preflight GPU probe: three outcomes, never two
===========================================================
NOT a certified S172 suite. New file, standalone, no production behaviour is
imported from any S172 gate.

WHAT THIS PROVES
  probe returns a count      -> that count is reported
  binary missing             -> UNAVAILABLE, and specifically NOT 0
  rocm-smi non-zero exit     -> UNAVAILABLE
  ssh non-zero exit          -> UNAVAILABLE
  timeout                    -> UNAVAILABLE
  unparseable output         -> ERROR
  preflight still PASSES in every one of those cases (advisory unchanged)
  MUTATION: the pre-fix probe (`|| echo 0`) reports 0 on the missing-binary
            fixture -> the missing-binary gate must RED under it.

HOW IT MEASURES (behavioural, not variable-assignment)
  A real `ssh` shim on PATH executes the probe's command string through a real
  shell against a controlled fixture PATH. `check_gpu_health` runs unmodified
  and unaware: the subprocess call, the argv shape, the remote shell parse and
  the classification are all genuinely exercised. Nothing is stubbed at the
  Python seam the fix lives behind.

RUN:  source ~/venvs/torch/bin/activate && python3 -u tests/test_preflight_gpu_probe.py
"""

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import preflight_check as pf  # noqa: E402

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"

_RESULTS = []


def check(name, ok, detail=""):
    tag = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    _RESULTS.append((name, bool(ok)))
    print(f"  [{tag}] {name:<34} {detail}")


def unavailable(name, detail=""):
    """VIR-3: a check that could not run terminates UNAVAILABLE, never PASS."""
    _RESULTS.append((name, False))
    print(f"  [{YELLOW}UNAVAIL{RESET}] {name:<34} {detail}")


# ──────────────────────────────────────────────────────────────────────────────
# Fixture: a fake `ssh` that runs the probe string through a real shell, plus a
# controlled PATH holding whichever `rocm-smi` variant an arm needs.
# ──────────────────────────────────────────────────────────────────────────────

SSH_SHIM = r"""#!/usr/bin/env bash
# Fake ssh. Consumes -o KEY=VAL pairs and the hostname, then runs the remaining
# single argument as a shell command -- exactly what a real remote login shell
# does with the one argv element ssh hands it.
while [ $# -gt 0 ]; do
  case "$1" in
    -o) shift 2 ;;
    -n) shift ;;
    -*) shift ;;
    *)  break ;;
  esac
done
shift                      # drop the hostname
exec bash -c "$*"
"""

# Emits a device table shaped like rocm-smi's: N lines start with a device
# number followed by whitespace, which is what the probe counts.
ROCM_SMI_OK = r"""#!/usr/bin/env bash
echo "======================= ROCm System Management Interface ======================="
echo "GPU  Temp   AvgPwr  SCLK    MCLK    Fan   Perf  PwrCap  VRAM%  GPU%"
for i in 0 1 2 3 4 5 6 7; do
  printf '%d    45.0c  12.0W   500Mhz  96Mhz   0%%   auto  100.0W    0%%   0%%\n' "$i"
done
echo "================================================================================"
"""

ROCM_SMI_EXIT3 = r"""#!/usr/bin/env bash
echo "ERROR: unable to open kmfd device" >&2
exit 3
"""

ROCM_SMI_HANG = r"""#!/usr/bin/env bash
sleep 30
"""

# Ignores the probe script entirely and returns output the classifier cannot
# turn into a count: the probe DID run, so this is ERROR, not UNAVAILABLE.
SSH_SHIM_GARBLED = r"""#!/usr/bin/env bash
echo "TFM_PROBE_BIN=/opt/rocm/bin/rocm-smi"
echo "TFM_PROBE_STATUS=OK"
echo "TFM_PROBE_COUNT=eight"
"""

SSH_SHIM_FAILS = r"""#!/usr/bin/env bash
echo "ssh: connect to host port 22: No route to host" >&2
exit 255
"""


def _write_exec(path: Path, body: str):
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class Fixture:
    """One temp world: a bin/ dir prepended to PATH, and a config with one node."""

    def __init__(self, ssh_body=SSH_SHIM, rocm_body=None, expected=8):
        self.tmp = Path(tempfile.mkdtemp(prefix="tfm_gpuprobe_"))
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        _write_exec(self.bin / "ssh", ssh_body)
        if rocm_body is not None:
            _write_exec(self.bin / "rocm-smi", rocm_body)
        # A fallback path that deliberately does not exist, so the
        # missing-binary arm cannot be rescued by a real /opt/rocm.
        self.fallback = self.tmp / "nonexistent" / "rocm-smi"
        self.cfg = self.tmp / "distributed_config.json"
        self.cfg.write_text(json.dumps({"nodes": [
            {"hostname": "localhost", "gpu_count": 1},
            {"hostname": "rig-under-test", "gpu_count": expected,
             "ramdisk_path": "/dev/shm/prng"},
        ]}))

    def __enter__(self):
        self._path = os.environ.get("PATH", "")
        self._fallbacks = pf.ROCM_SMI_FALLBACK_PATHS
        os.environ["PATH"] = f"{self.bin}:{self._path}"
        pf.ROCM_SMI_FALLBACK_PATHS = (str(self.fallback),)
        return self

    def __exit__(self, *exc):
        os.environ["PATH"] = self._path
        pf.ROCM_SMI_FALLBACK_PATHS = self._fallbacks
        shutil.rmtree(self.tmp, ignore_errors=True)

    def checker(self):
        return pf.PreflightChecker(config_file=str(self.cfg))

    def probe(self):
        return self.checker().check_gpu_health()


def node_of(result):
    return result["nodes"]["rig-under-test"]


def issues_of(result):
    return [i for i in result["issues"] if i["node"] == "rig-under-test"]


# ──────────────────────────────────────────────────────────────────────────────
# G1-G6: the six outcomes
# ──────────────────────────────────────────────────────────────────────────────

def g1_count_reported():
    with Fixture(rocm_body=ROCM_SMI_OK, expected=8) as f:
        r = f.probe()
        n = node_of(r)
        check("G1-COUNT-OBSERVED",
              n["status"] == pf.GPU_PROBE_OK and n["gpu_count"] == 8
              and r["all_healthy"] is True and not issues_of(r),
              f"status={n['status']} count={n['gpu_count']} via {n['binary']}")


def g1b_absolute_fallback_used():
    """PATH-less resolution still works: this is the production condition."""
    with Fixture(expected=8) as f:
        # No rocm-smi on PATH; place one at the fallback location instead.
        f.fallback.parent.mkdir(parents=True, exist_ok=True)
        _write_exec(f.fallback, ROCM_SMI_OK)
        n = node_of(f.probe())
        check("G1B-ABSOLUTE-FALLBACK",
              n["status"] == pf.GPU_PROBE_OK and n["gpu_count"] == 8
              and n["binary"] == str(f.fallback),
              f"count={n['gpu_count']} via {n['binary']}")


def g2_missing_binary_is_unavailable():
    """THE defect. Missing binary must be UNAVAILABLE, and must not be 0."""
    with Fixture(rocm_body=None, expected=8) as f:
        r = f.probe()
        n = node_of(r)
        iss = issues_of(r)
        ok = (n["status"] == pf.GPU_PROBE_UNAVAILABLE
              and n["gpu_count"] is None
              and n["gpu_count"] != 0
              and n["reason"] == "binary_not_found"
              and len(iss) == 1
              and iss[0]["type"] == "GPU_PROBE_UNAVAILABLE"
              and iss[0]["observed"] is None
              and r["all_healthy"] is False)
        check("G2-MISSING-BINARY-UNAVAIL", ok,
              f"status={n['status']} count={n['gpu_count']!r} reason={n['reason']}")


def g3_nonzero_exit_is_unavailable():
    with Fixture(rocm_body=ROCM_SMI_EXIT3, expected=8) as f:
        n = node_of(f.probe())
        check("G3-NONZERO-EXIT-UNAVAIL",
              n["status"] == pf.GPU_PROBE_UNAVAILABLE
              and n["gpu_count"] is None
              and n["reason"] == "rocm_smi_exit_3",
              f"reason={n['reason']} stderr={n['stderr'][:48]!r}")


def g3b_ssh_failure_is_unavailable():
    with Fixture(ssh_body=SSH_SHIM_FAILS, expected=8) as f:
        n = node_of(f.probe())
        check("G3B-SSH-FAIL-UNAVAIL",
              n["status"] == pf.GPU_PROBE_UNAVAILABLE
              and n["gpu_count"] is None
              and n["reason"] == "ssh_exit_255",
              f"reason={n['reason']} stderr={n['stderr'][:48]!r}")


def g4_timeout_is_unavailable():
    with Fixture(rocm_body=ROCM_SMI_HANG, expected=8) as f:
        saved = pf.GPU_CHECK_TIMEOUT_SECONDS
        pf.GPU_CHECK_TIMEOUT_SECONDS = 2
        try:
            n = node_of(f.probe())
        finally:
            pf.GPU_CHECK_TIMEOUT_SECONDS = saved
        check("G4-TIMEOUT-UNAVAIL",
              n["status"] == pf.GPU_PROBE_UNAVAILABLE
              and n["gpu_count"] is None
              and n["reason"] == "timeout",
              f"reason={n['reason']}")


def g5_unparseable_is_error():
    with Fixture(ssh_body=SSH_SHIM_GARBLED, expected=8) as f:
        r = f.probe()
        n = node_of(r)
        iss = issues_of(r)
        check("G5-UNPARSEABLE-ERROR",
              n["status"] == pf.GPU_PROBE_ERROR
              and n["gpu_count"] is None
              and n["reason"] == "unparseable_device_count"
              and iss and iss[0]["type"] == "GPU_PROBE_ERROR",
              f"status={n['status']} reason={n['reason']}")


def g6_genuine_zero_is_still_zero():
    """The converse of G2: an observed zero must NOT be laundered into UNAVAILABLE."""
    empty = "#!/usr/bin/env bash\necho 'GPU  Temp'\n"
    with Fixture(rocm_body=empty, expected=8) as f:
        n = node_of(f.probe())
        iss = issues_of(f.probe())
        check("G6-OBSERVED-ZERO-IS-ZERO",
              n["status"] == pf.GPU_PROBE_OK and n["gpu_count"] == 0
              and iss and iss[0]["type"] == "GPU_COUNT_MISMATCH"
              and iss[0]["observed"] == 0,
              f"status={n['status']} count={n['gpu_count']}")


# ──────────────────────────────────────────────────────────────────────────────
# G7: gating unchanged — every outcome is advisory
# ──────────────────────────────────────────────────────────────────────────────

def g7_preflight_still_passes():
    """Step 6 needs no ramdisk and no input files, so for arms whose SSH
    transport works the GPU check is the ONLY thing that could fail the run.
    If GPU gating changed, this reds.

    The `ssh-fail` arm is scored differently ON PURPOSE: there, check #1 (SSH
    connectivity) legitimately fails and legitimately blocks. That is a
    pre-existing, separate, BLOCKING check and this item does not touch it. What
    must hold there is only that no failure is attributed to the GPU probe --
    asserting `passed is True` would be asserting the wrong thing.
    """
    arms = [
        ("count",     dict(rocm_body=ROCM_SMI_OK),     True),
        ("no-binary", dict(rocm_body=None),            True),
        ("exit-3",    dict(rocm_body=ROCM_SMI_EXIT3),  True),
        ("garbled",   dict(ssh_body=SSH_SHIM_GARBLED), True),
        ("ssh-fail",  dict(ssh_body=SSH_SHIM_FAILS),   False),
    ]
    bad = []
    for label, kw, ssh_ok in arms:
        with Fixture(expected=8, **kw) as f:
            res = f.checker().check_all(step=6, auto_remediate=False)
            gpu_failures = [x for x in res.failures if "GPU" in x]
            if gpu_failures:
                bad.append(f"{label}: GPU-attributed failure {gpu_failures}")
            # The GPU check must always be counted as passed. With SSH working,
            # every check passes; with SSH broken, exactly one (the SSH check)
            # does not -- so the GPU check is still inside checks_passed.
            expect_passed = res.checks_run if ssh_ok else res.checks_run - 1
            if res.checks_passed != expect_passed:
                bad.append(f"{label}: checks_passed={res.checks_passed} "
                           f"expected={expect_passed}")
            if ssh_ok and not res.passed:
                bad.append(f"{label}: passed={res.passed} failures={res.failures}")
    check("G7-GATING-UNCHANGED", not bad,
          "GPU advisory in all 5 arms" if not bad else "; ".join(bad))


def g8_warning_never_renders_a_count():
    """An UNAVAILABLE node must not render as `0/8` or `None/8`."""
    with Fixture(rocm_body=None, expected=8) as f:
        res = f.checker().check_all(step=6, auto_remediate=False)
        warns = [w for w in res.warnings if "rig-under-test" in w]
        ok = (len(warns) == 1
              and "UNAVAILABLE" in warns[0]
              and "0/8" not in warns[0]
              and "None/8" not in warns[0]
              and "binary_not_found" in warns[0])
        check("G8-WARNING-NAMES-UNAVAIL", ok, warns[0] if warns else "<no warning>")


# ──────────────────────────────────────────────────────────────────────────────
# MUTATION — restore `|| echo 0`; the missing-binary arm must red
# ──────────────────────────────────────────────────────────────────────────────

LEGACY_SHELL = "rocm-smi 2>/dev/null | grep -cE '^[0-9]+[[:space:]]' || echo 0"

# The baseline this authenticity check reads the PRE-FIX probe out of. It is a
# pinned commit, NOT `HEAD`, and that is the whole point:
#
#   * `HEAD` is the pre-fix source only while this repair is uncommitted. The
#     moment it lands, `git show HEAD:preflight_check.py` returns the POST-fix
#     file, LEGACY_SHELL is no longer in it, and M1A reports the authentic
#     transcription as unauthentic — the gate reds on its own success. That is
#     exactly what `G-MATRIX-DIFF-a` did when `4b1aad6` became HEAD, and the same
#     amendment (`tests/test_s172_staging_backpressure.py:1550-1560`) is why the
#     repo already pins baselines by hash.
#   * A commit hash is the only thing that anchors "what the file looked like
#     before the change", so it belongs here — it anchors a certified artifact,
#     not a value copied from memory.
_PRE_FIX_REV = "c4e003743893f489b85310aa8a2d36505185a2ec"  # probe repair's parent


def _legacy_check_gpu_health(checker):
    """The PRE-FIX probe, transcribed from c4e0037:preflight_check.py
    check_gpu_health.

    Authenticity of the transcription is asserted separately in
    `m1_mutation_missing_binary` by finding LEGACY_SHELL in the committed file.
    """
    results = {"all_healthy": True, "nodes": {}, "issues": []}
    for node in checker.nodes:
        hostname = node["hostname"]
        expected = node.get("gpu_count", 12)
        cmd = ["ssh", "-o", f"ConnectTimeout={pf.SSH_TIMEOUT_SECONDS}", hostname,
               "bash", "-lc", LEGACY_SHELL]
        proc = subprocess.run(cmd, capture_output=True,
                              timeout=pf.GPU_CHECK_TIMEOUT_SECONDS)
        if proc.returncode == 0:
            output = proc.stdout.decode().strip()
            lines = [l.strip() for l in output.splitlines() if l.strip()]
            gpu_count = 0
            for line in reversed(lines):
                if line.isdigit():
                    gpu_count = int(line)
                    break
            results["nodes"][hostname] = {"gpu_count": gpu_count, "expected": expected}
            if gpu_count < expected:
                results["issues"].append({"node": hostname,
                                          "type": "GPU_COUNT_MISMATCH",
                                          "observed": gpu_count,
                                          "expected": expected})
                results["all_healthy"] = False
    return results


def m1_mutation_missing_binary():
    # (a) the transcription is the real pre-fix construct, not an approximation
    try:
        committed = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{_PRE_FIX_REV}:preflight_check.py"],
            capture_output=True, timeout=30)
        if committed.returncode != 0:
            unavailable("M1A-MUTANT-AUTHENTIC", "git show failed")
            return
        # COMMENT-STRIPPED (Beta R2 §3). The repair DOCUMENTS the construct it
        # replaced — `preflight_check.py:62` quotes this exact string in a
        # comment — so a bare substring test can be satisfied by commentary
        # rather than by an executable probe, and would go vacuous rather than
        # red if the baseline ever moved. The match must be in live code.
        found = LEGACY_SHELL in "\n".join(
            l for l in committed.stdout.decode(errors="replace").splitlines()
            if not l.lstrip().startswith("#"))
        check("M1A-MUTANT-AUTHENTIC", found,
              f"`|| echo 0` construct located in EXECUTABLE code at "
              f"{_PRE_FIX_REV[:7]}:preflight_check.py")
        if not found:
            return
    except Exception as e:
        unavailable("M1A-MUTANT-AUTHENTIC", f"{type(e).__name__}: {e}")
        return

    # (b) on the SAME fixture, the mutant reports a definite 0 where the fix
    #     reports UNAVAILABLE. The missing-binary gate discriminates them.
    with Fixture(rocm_body=None, expected=8) as f:
        checker = f.checker()
        mutant = _legacy_check_gpu_health(checker)
        fixed = checker.check_gpu_health()
        m = mutant["nodes"]["rig-under-test"]
        x = fixed["nodes"]["rig-under-test"]
        mutant_says_zero = (m.get("gpu_count") == 0
                            and m.get("status") is None)
        fix_says_unavailable = (x["status"] == pf.GPU_PROBE_UNAVAILABLE
                                and x["gpu_count"] is None)
        check("M1B-MUTANT-REDS-G2", mutant_says_zero and fix_says_unavailable,
              f"mutant={m.get('gpu_count')!r} (0/8) vs fixed={x['status']}")


# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PREFLIGHT GPU PROBE — three outcomes, never two")
    print("=" * 70)
    print("\n-- outcomes --")
    g1_count_reported()
    g1b_absolute_fallback_used()
    g2_missing_binary_is_unavailable()
    g3_nonzero_exit_is_unavailable()
    g3b_ssh_failure_is_unavailable()
    g4_timeout_is_unavailable()
    g5_unparseable_is_error()
    g6_genuine_zero_is_still_zero()
    print("\n-- gating (must be unchanged) --")
    g7_preflight_still_passes()
    g8_warning_never_renders_a_count()
    print("\n-- mutation --")
    m1_mutation_missing_binary()

    passed = sum(1 for _, ok in _RESULTS if ok)
    total = len(_RESULTS)
    print("=" * 70)
    print(f"{passed}/{total} checks green")
    if passed != total:
        print("FAILURES: " + ", ".join(n for n, ok in _RESULTS if not ok))
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
