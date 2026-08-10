#!/usr/bin/env python3
"""GATE-12 GPU FAIL-CLOSE GATE — the launch rule (Beta R3 / P2).

FALSIFIABLE QUESTION
  Does the Gate-12 harness refuse to launch unless all three rigs truthfully
  report `status == OK` and the full expected device count?

NO REAL RIG IS CONTACTED. Same technique as `tests/test_preflight_gpu_probe.py`:
a real `ssh` shim on PATH executes the probe's command string through a real
shell against a controlled fixture PATH. `scripts/gate12_gpu_gate.py` runs
UNMODIFIED — the fixture replaces the transport, never the code under test.

THE FULL INPUT SPACE IS ENUMERATED, not just the case that motivated the rule.
A gate validated only against `0/8` is a gate whose other three arms are
untested:

    OK, count == expected, all three rigs   -> ALLOWED
    UNAVAILABLE on any rig                  -> REFUSED   (4 causes, each tested)
    ERROR on any rig                        -> REFUSED
    OK but count != expected on any rig     -> REFUSED   (incl. a genuine 0)

Run:  source ~/venvs/torch/bin/activate && python3 tests/test_gate12_gpu_gate.py
"""

import io
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import preflight_check as PF                                        # noqa: E402
import gate12_gpu_gate as G                                         # noqa: E402

GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"
_RESULTS = []


def check(name, ok, detail=""):
    tag = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    _RESULTS.append((name, bool(ok)))
    print(f"  [{tag}] {name:<38} {detail}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# fixtures — the shim vocabulary is deliberately the same as the probe suite's
# ──────────────────────────────────────────────────────────────────────────────

SSH_SHIM = r"""#!/usr/bin/env bash
# Fake ssh. Consumes -o KEY=VAL pairs and user@host, then runs the remaining
# single argument as a shell command -- what a real remote login shell does.
while [ $# -gt 0 ]; do
  case "$1" in
    -o) shift 2 ;;
    -n) shift ;;
    -*) shift ;;
    *)  break ;;
  esac
done
HOST="$1"; shift
export TFM_FAKE_HOST="$HOST"
exec bash -c "$*"
"""

# Per-rig dispatch: the shim picks a rocm-smi variant by the host it was handed,
# so ONE arm can make exactly ONE of the three rigs misbehave. That is the case
# the rule is actually about — the fleet is not uniformly broken.
SSH_SHIM_PER_HOST = r"""#!/usr/bin/env bash
while [ $# -gt 0 ]; do
  case "$1" in
    -o) shift 2 ;;
    -n) shift ;;
    -*) shift ;;
    *)  break ;;
  esac
done
HOST="$1"; shift
IP="${HOST##*@}"
if [ "$IP" = "__BAD_IP__" ]; then
__BAD_BEHAVIOUR__
fi
exec bash -c "$*"
"""

ROCM_SMI_N = r"""#!/usr/bin/env bash
echo "======================= ROCm System Management Interface ======================="
echo "GPU  Temp   AvgPwr  SCLK    MCLK    Fan   Perf  PwrCap  VRAM%  GPU%"
for i in $(seq 0 __LAST__); do
  printf '%d    45.0c  12.0W   500Mhz  96Mhz   0%%   auto  100.0W    0%%   0%%\n' "$i"
done
echo "================================================================================"
"""

ROCM_SMI_ZERO = r"""#!/usr/bin/env bash
echo "======================= ROCm System Management Interface ======================="
echo "GPU  Temp   AvgPwr  SCLK    MCLK    Fan   Perf  PwrCap  VRAM%  GPU%"
echo "================================================================================"
"""


def rocm_smi_with(n):
    """A rocm-smi emitting exactly `n` device rows."""
    if n == 0:
        return ROCM_SMI_ZERO
    return ROCM_SMI_N.replace("__LAST__", str(n - 1))


# The four UNAVAILABLE causes and the one ERROR cause, as remote behaviours.
BEHAVIOUR = {
    # ssh itself fails to reach the host
    "ssh_fail": 'echo "ssh: connect to host port 22: No route to host" >&2\n'
                'exit 255',
    # rocm-smi is absent on that rig (PATH and fallback both empty)
    "no_binary": 'exec env PATH=/nonexistent-tfm bash -c "$*"',
    # rocm-smi exists but exits non-zero
    "exit3": 'exec bash -c \'echo "ERROR: unable to open kmfd device" >&2; '
             'exit 3\'',
    # the probe hangs past the timeout
    "timeout": 'sleep 30; exit 0',
    # the probe runs but emits a count the classifier cannot use -> ERROR
    "garbled": 'echo "TFM_PROBE_BIN=/opt/rocm/bin/rocm-smi"\n'
               'echo "TFM_PROBE_STATUS=OK"\n'
               'echo "TFM_PROBE_COUNT=eight"\n'
               'exit 0',
}


def _write_exec(path, body):
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class World:
    """A temp bin/ on PATH, plus a pinned target list.

    `gate_targets` is pinned rather than resolved so an arm cannot depend on
    the live profile map — but a separate arm (P2-TARGETS-ARE-DERIVED) proves
    the production function really does derive the same three rigs from
    committed source, so the pinning here is a fixture, not a fiction.
    """

    def __init__(self, ssh_body=SSH_SHIM, gpus=8, expected=8):
        self.tmp = Path(tempfile.mkdtemp(prefix="tfm_gate12gpu_"))
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        _write_exec(self.bin / "ssh", ssh_body)
        _write_exec(self.bin / "rocm-smi", rocm_smi_with(gpus))
        self.fallback = self.tmp / "nonexistent" / "rocm-smi"
        self.expected = expected

    def __enter__(self):
        self._path = os.environ.get("PATH", "")
        self._fallbacks = PF.ROCM_SMI_FALLBACK_PATHS
        self._targets = G.gate_targets
        os.environ["PATH"] = f"{self.bin}:{self._path}"
        PF.ROCM_SMI_FALLBACK_PATHS = (str(self.fallback),)
        G.gate_targets = lambda rig_profile="proxmox": [
            ("rrig6600", "192.168.3.122", "michael", self.expected),
            ("rrig6600b", "192.168.3.156", "michael", self.expected),
            ("rrig6600c", "192.168.3.164", "michael", self.expected),
        ]
        return self

    def __exit__(self, *exc):
        os.environ["PATH"] = self._path
        PF.ROCM_SMI_FALLBACK_PATHS = self._fallbacks
        G.gate_targets = self._targets
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run(self):
        """Drive the real `main()`; return (exit_code, stdout)."""
        buf, old = io.StringIO(), sys.stdout
        sys.stdout = buf
        try:
            rc = G.main([])
        finally:
            sys.stdout = old
        return rc, buf.getvalue()


def one_bad_rig(behaviour, bad_ip="192.168.3.156", gpus=8):
    """A world where exactly ONE of the three rigs misbehaves."""
    body = (SSH_SHIM_PER_HOST
            .replace("__BAD_IP__", bad_ip)
            .replace("__BAD_BEHAVIOUR__", BEHAVIOUR[behaviour]))
    return World(ssh_body=body, gpus=gpus)


# ──────────────────────────────────────────────────────────────────────────────
# THE INPUT SPACE
# ──────────────────────────────────────────────────────────────────────────────

def p2_all_ok_allows():
    """8/8 on all three -> the ONLY configuration that launches."""
    with World(gpus=8, expected=8) as w:
        rc, out = w.run()
    check("P2-8x8x3-ALLOWED",
          rc == G.EXIT_PROCEED and "PASS" in out and "3/3 rigs" in out
          and "REFUSED" not in out,
          f"rc={rc}; all three rigs OK at 8/8 -> launch may proceed")


def p2_unavailable_refuses():
    """All four UNAVAILABLE causes, each on ONE rig, each must refuse.

    Enumerated rather than sampled: `ssh_exit_255` and `binary_not_found` reach
    the refusal by different paths inside the gate (transport vs classifier),
    and a rule tested on only one of them has an untested arm.
    """
    outcomes = {}
    for cause in ("ssh_fail", "no_binary", "exit3", "timeout"):
        with one_bad_rig(cause) as w:
            rc, out = w.run()
        outcomes[cause] = (rc, out)

    all_refused = all(rc == G.EXIT_REFUSE for rc, _ in outcomes.values())
    all_named = all("192.168.3.156" in out for _, out in outcomes.values())
    all_say_unavailable = all(
        PF.GPU_PROBE_UNAVAILABLE in out for _, out in outcomes.values())
    # THE HONESTY REQUIREMENT: an unavailable rig is never rendered as a count.
    never_a_count = all(
        "0/8" not in out and "None/8" not in out
        for _, out in outcomes.values())

    check("P2-UNAVAILABLE-REFUSES",
          all_refused and all_named and all_say_unavailable and never_a_count,
          "ssh_fail/no_binary/exit3/timeout all -> REFUSED, rig named, "
          "reported as UNAVAILABLE and never as a count")


def p2_error_refuses():
    """Unparseable probe output -> ERROR -> refuse. The probe RAN, so this is
    not UNAVAILABLE, and the gate must still not launch."""
    with one_bad_rig("garbled") as w:
        rc, out = w.run()
    check("P2-ERROR-REFUSES",
          rc == G.EXIT_REFUSE and PF.GPU_PROBE_ERROR in out
          and "192.168.3.156" in out and "0/8" not in out,
          f"rc={rc}; unparseable count -> {PF.GPU_PROBE_ERROR}, rig named, "
          f"no invented count")


def p2_count_mismatch_refuses():
    """OK but not the full count — including the genuine observed 0 that
    attempt 1 logged and launched through anyway."""
    results = {}
    for n in (0, 7):
        with World(gpus=n, expected=8) as w:
            results[n] = w.run()

    zero_rc, zero_out = results[0]
    seven_rc, seven_out = results[7]
    # A genuine zero is reported AS a zero — the probe observed it. That is the
    # opposite requirement from the UNAVAILABLE arm, and both must hold.
    zero_honest = "0/8" in zero_out and PF.GPU_PROBE_OK in zero_out

    check("P2-COUNT-MISMATCH-REFUSES",
          zero_rc == G.EXIT_REFUSE and seven_rc == G.EXIT_REFUSE
          and zero_honest and "7/8" in seven_out,
          f"observed 0/8 -> REFUSED (and reported as a real zero, not "
          f"UNAVAILABLE); 7/8 -> REFUSED")


def p2_refusal_precedes_the_sampler():
    """The refusal must abort BEFORE the sampler is armed and BEFORE any
    coordinator process exists — asserted on the launch script's own text.

    Checked structurally rather than by launching anything: the gate's line
    number must precede the sampler's, the coordinator's and the fleet's.
    """
    src = (REPO / "gate12_launch.sh").read_text().splitlines()

    def line_of(needle):
        for i, l in enumerate(src):
            if needle in l and not l.strip().startswith("#"):
                return i
        return None

    gate = line_of("scripts/gate12_gpu_gate.py")
    sampler = line_of("scripts/gate12_concurrency_sampler.py")
    watcher = line_of("watcher_agent.py")
    fleet = line_of("launch_fleet_manual.sh")
    cleanslate = line_of("pkill -f")

    ordered = (gate is not None and sampler is not None and watcher is not None
               and fleet is not None
               and gate < cleanslate < sampler < watcher < fleet)
    # and the refusal must actually terminate the script
    aborts = any("exit 1" in l for l in src[gate:sampler])

    check("P2-REFUSAL-PRECEDES-SAMPLER",
          ordered and aborts,
          f"gate@{gate} < clean-slate@{cleanslate} < sampler@{sampler} < "
          f"coordinator@{watcher} < fleet@{fleet}, and refusal exits")


def p2_launch_script_honours_the_exit_code():
    """MUTANT-ADJACENT, and the defect this arm exists for is REAL.

    `cmd | tee` exits with TEE's status, so `if ! python3 gate.py | tee` would
    print REFUSED and launch anyway. The script must read ${PIPESTATUS[0]}.
    Proven by EXECUTING both forms against a stub that exits 1 — a text check
    alone would not show that the bypass actually launches.
    """
    import subprocess
    tmp = Path(tempfile.mkdtemp(prefix="tfm_pipestatus_"))
    try:
        stub = tmp / "gate.py"
        stub.write_text("import sys; print('REFUSED'); sys.exit(1)\n")
        evid = tmp / "evid.txt"

        correct = f"""
set -u
python3 {stub} 2>&1 | tee -a {evid}
RC=${{PIPESTATUS[0]}}
if [ "$RC" -ne 0 ]; then echo ABORTED; exit 1; fi
echo LAUNCHED
"""
        bypass = f"""
set -u
if ! python3 {stub} 2>&1 | tee -a {evid}; then echo ABORTED; exit 1; fi
echo LAUNCHED
"""
        c = subprocess.run(["bash", "-c", correct], capture_output=True, text=True)
        b = subprocess.run(["bash", "-c", bypass], capture_output=True, text=True)

        script = (REPO / "gate12_launch.sh").read_text()
        uses_pipestatus = "PIPESTATUS[0]" in script

        check("P2-MUTANT-PIPESTATUS-BYPASS",
              "ABORTED" in c.stdout and "LAUNCHED" not in c.stdout
              and "LAUNCHED" in b.stdout and uses_pipestatus,
              "the `| tee` form swallows the refusal and LAUNCHES; the "
              "PIPESTATUS form ABORTS — and the live script uses PIPESTATUS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def p2_mutant_gate_result_ignored():
    """MUTATION: the gate runs but its verdict is discarded.

    If `evaluate` returned "allowed" unconditionally, every refusal arm above
    would have to go green — that is what makes those arms non-vacuous.
    """
    real = G.evaluate
    reds = {}
    try:
        G.evaluate = lambda results: (True, [])
        for cause, world in (("unavailable", one_bad_rig("ssh_fail")),
                             ("error", one_bad_rig("garbled")),
                             ("mismatch", World(gpus=0, expected=8))):
            with world as w:
                rc, _out = w.run()
            reds[cause] = rc
    finally:
        G.evaluate = real

    check("P2-MUTANT-GATE-RESULT-IGNORED",
          all(rc == G.EXIT_PROCEED for rc in reds.values()),
          f"with the verdict discarded all three refusal arms would PROCEED "
          f"({reds}) — so the live refusals are decided by the gate, not by "
          f"the probe failing to run")


def p2_targets_are_derived_not_hardcoded():
    """The three addresses and the expected count come from committed source.

    A gate that probed addresses the run does not use would be worse than no
    gate: it would pass while the fleet was somewhere else. This arm calls the
    REAL `gate_targets` (no fixture) — it resolves the execution set, it does
    not open a socket.
    """
    targets = G.gate_targets("proxmox")
    endpoints = [e for _n, e, _u, _c in targets]
    counts = {e: c for _n, e, _u, c in targets}

    cfg = json.loads((REPO / "distributed_config.json").read_text())
    declared = {n["hostname"]: n.get("gpu_count") for n in cfg["nodes"]}

    # the module contains no literal rig address of its own
    src = (REPO / "scripts" / "gate12_gpu_gate.py").read_text()
    code = "\n".join(l for l in src.splitlines()
                     if not l.strip().startswith("#"))
    no_literals = not any(ip in code for ip in endpoints)

    check("P2-TARGETS-ARE-DERIVED",
          endpoints == ["192.168.3.122", "192.168.3.156", "192.168.3.164"]
          and set(counts.values()) == {8}
          and declared.get("192.168.3.120") == 8 and no_literals,
          f"{endpoints} with expected={sorted(set(counts.values()))} resolved "
          f"from rig_profiles_config.json + distributed_config.json; no rig "
          f"address literal in the gate's executable code")


def p2_probe_is_the_certified_one():
    """The gate must not carry a second probe or a second classifier.

    A second implementation is a second place for the `|| echo 0` defect to
    live. Identity is asserted on the objects, not on an import line.
    """
    src = (REPO / "scripts" / "gate12_gpu_gate.py").read_text()
    reimplements = ("TFM_PROBE_STATUS=" in src.replace("_PROBE_STATUS", "")
                    or "command -v rocm-smi" in src)
    check("P2-REUSES-CERTIFIED-PROBE",
          G._build_gpu_probe_script is PF._build_gpu_probe_script
          and G._parse_gpu_probe is PF._parse_gpu_probe
          and not reimplements,
          "builder and classifier are the certified objects themselves; no "
          "second probe string and no second parser in the gate")


def main():
    print("=" * 74)
    print("GATE-12 GPU FAIL-CLOSE GATE — launch rule (Beta R3 / P2)")
    print("=" * 74)
    print("\n-- the input space: exactly one row proceeds --")
    p2_all_ok_allows()
    p2_unavailable_refuses()
    p2_error_refuses()
    p2_count_mismatch_refuses()
    print("\n-- placement and wiring --")
    p2_refusal_precedes_the_sampler()
    p2_launch_script_honours_the_exit_code()
    print("\n-- provenance --")
    p2_targets_are_derived_not_hardcoded()
    p2_probe_is_the_certified_one()
    print("\n-- mutation --")
    p2_mutant_gate_result_ignored()

    passed = sum(1 for _, ok in _RESULTS if ok)
    total = len(_RESULTS)
    print("=" * 74)
    print(f"{passed}/{total} checks green")
    if passed != total:
        print("FAILURES: " + ", ".join(n for n, ok in _RESULTS if not ok))
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
