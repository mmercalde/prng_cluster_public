#!/usr/bin/env python3
"""S172 D6 INTEGRATION REPAIR — the gate battery.

Beta authorized a NARROW repair of the fleet-deployment / prelaunch integration
layer after the D6 parked-fleet dry run of 2026-08-14. The attempt-6
coordinator/worker mechanics are CERTIFIED and FROZEN and are not touched here.

TWO FALSIFIABLE QUESTIONS, ONE SUITE:

  A  RIG CODE-PARITY GATE (`scripts/gate12_parity_gate.py`)
     Does a fail-closed source-parity wall refuse to launch unless every rig
     carries every governed file at the canonical full-SHA256 digest — and does
     it refuse on mismatch, on a missing file, on malformed output, on an
     unavailable ssh, on a rig answering under the wrong hostname, and on a
     worker-side project import that entered the statically reachable
     project-local import / deployment closure without parity coverage?

  B  LAUNCHER WAIT-SET (`scripts/launch_fleet_manual.sh`)
     With NO release token present, does the launcher RETURN while its own
     long-lived local worker remains ALIVE AND PARKED, with NO REGISTER having
     occurred and NO release token in existence — and does the pinned pre-fix
     shape fail that same proof?

WHAT WRONG INPUT MAKES EACH GATE RED is stated in the arm's own docstring, per
the brief. Every RED arm is pinned to a FULL SHA256 and verifies the anchor still
carries the defect surface; a drifted anchor reports UNAVAILABLE and never a
pass.

NO REAL RIG IS CONTACTED and NO REAL FLEET IS LAUNCHED. Part A replaces the ssh
transport with a real `ssh` shim on PATH executing the gate's own probe script
through a real shell against fixture trees — `scripts/gate12_parity_gate.py` runs
UNMODIFIED. Part B copies the launcher's live bytes into a temporary tree (digest
equality asserted) beside stub config, a stub worker and a stub ssh, and drives
it against a real loopback listener.

Run:  source ~/venvs/torch/bin/activate && \
      python3 -u tests/test_s172_d6_integration_repair.py
"""

import ast
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import gate12_parity_gate as PG                                     # noqa: E402

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"
_RESULTS = []


def _porcelain():
    return sorted(subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain"],
        capture_output=True, text=True).stdout.rstrip("\n").splitlines())


# Captured BEFORE any arm runs. HI-3 asserts the suite did not CHANGE the
# working tree, which is the actual claim — an allowlist of expected entries
# would go stale the moment an unrelated file is edited, and would then be
# "fixed" by widening it, which is the wrong reflex (see Gate 22).
_PORCELAIN_AT_START = _porcelain()


def check(name, ok, detail=""):
    tag = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    _RESULTS.append((name, bool(ok), False))
    print(f"  [{tag}] {name:<44} {detail}", flush=True)


def unavailable(name, detail=""):
    """VIR-3: UNAVAILABLE is a terminal state of its own and never a pass."""
    _RESULTS.append((name, False, True))
    print(f"  [{YELLOW}UNAV{RESET}] {name:<44} {detail}", flush=True)


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
# PART A FIXTURES — a real ssh shim, and fake rig trees whose bytes we control
# ═══════════════════════════════════════════════════════════════════════════

SHIM_REAL = r"""#!/usr/bin/env bash
# Fake ssh. Consumes -o KEY=VAL pairs, -n, and user@host, then runs the
# remaining single argument as a shell command -- what a real remote login shell
# does with what the gate sends it.
while [ $# -gt 0 ]; do
  case "$1" in
    -o) shift 2 ;;
    -n) shift ;;
    -*) shift ;;
    *)  break ;;
  esac
done
shift            # user@host
exec bash -c "$*"
"""

SHIM_TRANSPORT_FAILURE = r"""#!/usr/bin/env bash
# ssh's OWN failure: no route / auth refusal / host-key mismatch. ssh reports
# 255 and writes a diagnostic to stderr. It never runs the remote command.
echo "ssh: connect to host port 22: No route to host" >&2
exit 255
"""

SHIM_TRUNCATED = r"""#!/usr/bin/env bash
# The probe starts and its output is cut short -- a dropped connection or a
# killed remote shell. The END sentinel never arrives.
while [ $# -gt 0 ]; do
  case "$1" in
    -o) shift 2 ;;
    -n) shift ;;
    -*) shift ;;
    *)  break ;;
  esac
done
shift
bash -c "$*" | head -n 4
exit 0
"""

SHIM_MALFORMED = r"""#!/usr/bin/env bash
# The probe RAN and its output cannot be classified: a digest that is not
# 64 lowercase hex. This is ERROR, not UNAVAILABLE.
echo TFM-PARITY-BEGIN
printf 'HOST\t%s\n' "malformed-rig"
printf 'FILE\t%s\t%s\t%s\n' "prng_registry.py" "not-a-digest" "10"
echo TFM-PARITY-END
"""

SHIM_WRONG_HOSTNAME = r"""#!/usr/bin/env bash
# The rig answers -- but as some other machine. Three rigs must not be able to
# be one machine answering thrice (skill 2.17).
while [ $# -gt 0 ]; do
  case "$1" in
    -o) shift 2 ;;
    -n) shift ;;
    -*) shift ;;
    *)  break ;;
  esac
done
shift
export PATH="$TFM_FAKE_HOSTNAME_BIN:$PATH"
exec bash -c "$*"
"""

SHIM_NO_DIR = r"""#!/usr/bin/env bash
# The deployment directory does not exist on the rig. The probe's own exit 3.
exit 3
"""


class ShimPath:
    """A temp bin/ at the FRONT of PATH holding one `ssh`."""

    def __init__(self, shim_src, extra_env=None):
        self.shim_src = shim_src
        self.extra_env = extra_env or {}
        self.tmp = None
        self._path = None
        self._saved_env = {}

    def __enter__(self):
        self.tmp = tempfile.mkdtemp(prefix="tfm_parity_shim_")
        binp = Path(self.tmp) / "bin"
        binp.mkdir()
        ssh = binp / "ssh"
        ssh.write_text(self.shim_src)
        ssh.chmod(ssh.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        self._path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{binp}:{self._path}"
        for k, v in self.extra_env.items():
            self._saved_env[k] = os.environ.get(k)
            os.environ[k] = v
        return self

    def __exit__(self, *exc):
        os.environ["PATH"] = self._path
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.tmp, ignore_errors=True)
        return False


def make_rig_tree(root, governed=PG.GOVERNED_FILES):
    """A fake rig deployment: the real governed files, byte-for-byte."""
    for rel in governed:
        dst = Path(root) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dst)
    return root


def target(script_path, node_id="rrigX", endpoint="10.255.255.1",
           hostname=None):
    return {
        "node_id": node_id,
        "endpoint": endpoint,
        "ssh_user": "michael",
        "worker_hostname": hostname or socket.gethostname(),
        "script_path": str(script_path),
    }


# ═══════════════════════════════════════════════════════════════════════════
# PART A — THE RIG CODE-PARITY GATE
# ═══════════════════════════════════════════════════════════════════════════

def part_a():
    section("PART A — RIG CODE-PARITY GATE (scripts/gate12_parity_gate.py)")
    expected = PG.expected_digests()

    # ---- PAR-1 clean control -------------------------------------------
    # RED WHEN: a rig whose governed files are byte-identical to the canonical
    # tree is reported as anything other than MATCH. Without this arm every
    # refusal below could be produced by a gate that refuses unconditionally.
    with tempfile.TemporaryDirectory() as td:
        rig = make_rig_tree(Path(td) / "rig")
        with ShimPath(SHIM_REAL):
            r = PG.probe_rig(target(rig))
        ok, refusals = PG.evaluate([r], expected)
        check("PAR-1  clean control: identical fleet PASSES",
              r["status"] == PG.PROBE_OK and ok and not refusals,
              f"status={r['status']} refusals={len(refusals)}")
        rows = PG.evidence_rows([r], expected)
        check("PAR-1b clean control: every row MATCH",
              all(row["verdict"] == PG.MATCH for row in rows)
              and len(rows) == len(PG.GOVERNED_FILES),
              f"{len(rows)} rows")

    # ---- PAR-2 digest mismatch (the D6 defect itself) -------------------
    # RED WHEN: a rig carries ANY governed file at a different digest. This is
    # the condition the live fleet is in today.
    with tempfile.TemporaryDirectory() as td:
        rig = make_rig_tree(Path(td) / "rig")
        victim = Path(rig) / "miner/range_miner_worker.py"
        victim.write_bytes(victim.read_bytes() + b"\n# drift\n")
        with ShimPath(SHIM_REAL):
            r = PG.probe_rig(target(rig))
        ok, refusals = PG.evaluate([r], expected)
        named = any("miner/range_miner_worker.py" in x for x in refusals)
        check("PAR-2  digest mismatch REFUSES and names the file",
              (not ok) and len(refusals) == 1 and named,
              f"refusals={len(refusals)}")
        rows = {x["canonical_path"]: x for x in PG.evidence_rows([r], expected)}
        row = rows["miner/range_miner_worker.py"]
        check("PAR-2b mismatch row carries BOTH full digests",
              row["verdict"] == PG.MISMATCH
              and len(row["expected_sha256"]) == 64
              and len(row["observed_sha256"]) == 64
              and row["expected_sha256"] != row["observed_sha256"],
              row["reason"])

    # ---- PAR-3 missing file --------------------------------------------
    # RED WHEN: a governed file is absent from the deployed tree. It must be a
    # MISMATCH (measured absence), never UNAVAILABLE (could not measure) — the
    # distinction the three-outcome vocabulary exists to preserve.
    with tempfile.TemporaryDirectory() as td:
        rig = make_rig_tree(Path(td) / "rig")
        os.remove(Path(rig) / "execution_set.py")
        with ShimPath(SHIM_REAL):
            r = PG.probe_rig(target(rig))
        ok, refusals = PG.evaluate([r], expected)
        rows = {x["canonical_path"]: x for x in PG.evidence_rows([r], expected)}
        row = rows["execution_set.py"]
        check("PAR-3  missing file REFUSES",
              (not ok) and any("ABSENT" in x for x in refusals),
              f"refusals={len(refusals)}")
        check("PAR-3b missing renders MISMATCH/MISSING, not UNAVAILABLE",
              row["verdict"] == PG.MISMATCH
              and row["observed_sha256"] == "MISSING"
              and row["reason"] == "file_absent_on_rig",
              f"verdict={row['verdict']} observed={row['observed_sha256']}")

    # ---- PAR-4 ssh transport failure -----------------------------------
    # RED WHEN: ssh itself fails. That is UNAVAILABLE — the probe did not run —
    # and the evidence must not render a count or an empty digest for it.
    with tempfile.TemporaryDirectory() as td:
        rig = make_rig_tree(Path(td) / "rig")
        with ShimPath(SHIM_TRANSPORT_FAILURE):
            r = PG.probe_rig(target(rig))
        ok, refusals = PG.evaluate([r], expected)
        rows = PG.evidence_rows([r], expected)
        check("PAR-4  ssh exit 255 is UNAVAILABLE and REFUSES",
              r["status"] == PG.PROBE_UNAVAILABLE
              and r["reason"] == "ssh_transport_failure" and not ok,
              f"stderr={r['stderr'][:40]!r}")
        check("PAR-4b UNAVAILABLE rows never render a digest",
              all(row["verdict"] == PG.UNAVAILABLE
                  and row["observed_sha256"] == PG.UNAVAILABLE
                  and row["observed_size"] == PG.UNAVAILABLE for row in rows)
              and len(rows) == len(PG.GOVERNED_FILES),
              f"{len(rows)} rows")
        # UNAVAILABLE and ERROR both render the row verdict UNAVAILABLE, so the
        # frozen bundle must keep them distinguishable by probe_status.
        with tempfile.TemporaryDirectory() as td2:
            rig2 = make_rig_tree(Path(td2) / "rig")
            with ShimPath(SHIM_MALFORMED):
                r_err = PG.probe_rig(target(rig2, hostname="malformed-rig"))
        err_rows = PG.evidence_rows([r_err], expected)
        check("PAR-4c UNAVAILABLE vs ERROR stay distinguishable in the bundle",
              all(row["probe_status"] == PG.PROBE_UNAVAILABLE for row in rows)
              and all(row["probe_status"] == PG.PROBE_ERROR
                      for row in err_rows)
              and {row["verdict"] for row in err_rows} == {PG.UNAVAILABLE},
              f"{rows[0]['probe_status']} vs {err_rows[0]['probe_status']}")

    # ---- PAR-5 truncated output ----------------------------------------
    # RED WHEN: the probe's output is cut short. VIR-1: truncation is never a
    # pass, and a partial listing must not be mistaken for a complete one.
    with tempfile.TemporaryDirectory() as td:
        rig = make_rig_tree(Path(td) / "rig")
        with ShimPath(SHIM_TRUNCATED):
            r = PG.probe_rig(target(rig))
        ok, _ = PG.evaluate([r], expected)
        check("PAR-5  truncated probe output is UNAVAILABLE",
              r["status"] == PG.PROBE_UNAVAILABLE
              and r["reason"] == "truncated_probe_output" and not ok,
              f"reason={r['reason']}")

    # ---- PAR-6 malformed output ----------------------------------------
    # RED WHEN: the probe ran and produced output that cannot be classified.
    # That is ERROR, distinct from UNAVAILABLE, and it still refuses.
    with tempfile.TemporaryDirectory() as td:
        rig = make_rig_tree(Path(td) / "rig")
        with ShimPath(SHIM_MALFORMED):
            r = PG.probe_rig(target(rig, hostname="malformed-rig"))
        ok, _ = PG.evaluate([r], expected)
        check("PAR-6  malformed output is ERROR and REFUSES",
              r["status"] == PG.PROBE_ERROR
              and str(r["reason"]).startswith("malformed_digest_for:")
              and not ok,
              f"reason={r['reason']}")

    # ---- PAR-7 hostname mismatch ---------------------------------------
    # RED WHEN: the machine that answers is not the machine the execution set
    # names. One box answering for three is the failure this forbids.
    with tempfile.TemporaryDirectory() as td:
        rig = make_rig_tree(Path(td) / "rig")
        fake_bin = Path(td) / "fakebin"
        fake_bin.mkdir()
        hn = fake_bin / "hostname"
        hn.write_text("#!/usr/bin/env bash\necho someone-elses-rig\n")
        hn.chmod(hn.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        with ShimPath(SHIM_WRONG_HOSTNAME,
                      {"TFM_FAKE_HOSTNAME_BIN": str(fake_bin)}):
            r = PG.probe_rig(target(rig, hostname="rrig6600"))
        ok, refusals = PG.evaluate([r], expected)
        check("PAR-7  wrong hostname REFUSES even with all digests MATCHing",
              r["status"] == PG.PROBE_OK and r["hostname"] == "someone-elses-rig"
              and (not ok) and any("answered as hostname" in x for x in refusals),
              f"hostname={r['hostname']}")

    # ---- PAR-8 no such directory ---------------------------------------
    with tempfile.TemporaryDirectory() as td:
        rig = make_rig_tree(Path(td) / "rig")
        with ShimPath(SHIM_NO_DIR):
            r = PG.probe_rig(target(rig))
        ok, _ = PG.evaluate([r], expected)
        check("PAR-8  missing deployment dir is UNAVAILABLE",
              r["status"] == PG.PROBE_UNAVAILABLE
              and str(r["reason"]).startswith("no_such_directory:") and not ok,
              f"reason={r['reason']}")

    # ---- PAR-9 unrecognized status falls through to refusal ------------
    fabricated = {"node_id": "x", "endpoint": "e", "expected_hostname": "h",
                  "script_path": "/p", "status": "SOMETHING_NEW",
                  "hostname": None, "observed": {}, "reason": None,
                  "stderr": "", "collected_at": None}
    ok, refusals = PG.evaluate([fabricated], expected)
    check("PAR-9  unrecognized probe status refuses BY DEFAULT",
          (not ok) and any("unrecognized probe status" in x for x in refusals))

    # ---- PAR-10 the pin covers the derived closure (LIVE tree) ----------
    # RED WHEN: a worker-side project import enters the statically reachable
    # project-local import / deployment closure and GOVERNED_FILES was not
    # extended. This is the arm that keeps the pin from silently going stale.
    # "Statically reachable", never "executed": Beta governs the conservative
    # superset deliberately (2026-08-14), because whether a given run reaches a
    # file depends on arguments, branches, failure handling and deferred
    # imports and is far harder to prove than static reachability.
    covered, uncovered, missing_min = PG.closure_coverage()
    check("PAR-10 pin covers the LIVE derived closure",
          covered, f"uncovered={uncovered} missing_minimum={missing_min}")
    check("PAR-10b Beta's five-file minimum is inside the pin",
          all(f in PG.GOVERNED_FILES for f in PG.BETA_MINIMUM_FILES),
          f"minimum={len(PG.BETA_MINIMUM_FILES)} pin={len(PG.GOVERNED_FILES)}")

    # ---- PAR-11 derivation: positive + fault-injection controls ---------
    # VIR-2: a coverage check that cannot go red is vacuous. These arms drive
    # the derivation on a controlled tree and prove BOTH directions.
    with tempfile.TemporaryDirectory() as td:
        r = Path(td)
        (r / "pkg").mkdir()
        (r / "a.py").write_text(
            "import os, json\n"
            "import b\n"
            "from pkg import c\n"
            "import definitely_not_a_real_module_xyz\n"
            "def later():\n"
            "    import d\n")          # function-local: must still be found
        (r / "b.py").write_text("import sys\n")
        (r / "pkg" / "__init__.py").write_text("")
        (r / "pkg" / "c.py").write_text("from .. import e\n")
        (r / "d.py").write_text("")
        (r / "e.py").write_text("")
        derived = PG.derive_worker_import_closure(str(r), "a.py")
        check("PAR-11 derivation finds transitive project files",
              derived == {"a.py", "b.py", "pkg/__init__.py", "pkg/c.py",
                          "d.py", "e.py"},
              f"{sorted(derived)}")
        check("PAR-11b stdlib and unresolvable names are NEVER governed",
              PG._module_to_repo_path("os", str(r)) is None
              and PG._module_to_repo_path("json", str(r)) is None
              and PG._module_to_repo_path(
                  "definitely_not_a_real_module_xyz", str(r)) is None)
        cov, unc, _ = PG.closure_coverage(str(r), governed=("a.py",),
                                          root="a.py")
        check("PAR-11c an uncovered closure entry REFUSES coverage",
              (not cov) and set(unc) == {"b.py", "pkg/__init__.py", "pkg/c.py",
                                         "d.py", "e.py"},
              f"uncovered={len(unc)}")

    # ---- PAR-12 expectations are DERIVED, never transcribed -------------
    # RED WHEN: a digest (or a 12-hex display prefix) is baked into the gate as
    # a comparison value. Docstrings may discuss them; executable string
    # literals may not carry them.
    src = (REPO / "scripts" / "gate12_parity_gate.py").read_text()
    tree = ast.parse(src)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            ds = ast.get_docstring(node, clean=False)
            if ds:
                docstrings.add(ds)
    hexish = re.compile(r"[0-9a-f]{12,}")
    # The ONE permitted hex-shaped literal is the validation alphabet the parser
    # uses to reject a non-hex digest. It is allowlisted by exact value, not by
    # pattern, so a 16-character digest prefix cannot hide behind the exemption.
    HEX_ALPHABET = "0123456789abcdef"
    offenders = [n.value for n in ast.walk(tree)
                 if isinstance(n, ast.Constant) and isinstance(n.value, str)
                 and n.value not in docstrings and n.value != HEX_ALPHABET
                 and hexish.search(n.value)]
    check("PAR-12 no digest literal outside docstrings",
          not offenders, f"offenders={offenders[:2]}")
    live = PG.expected_digests()
    independent = {f: sha256_file(REPO / f) for f in PG.GOVERNED_FILES}
    check("PAR-12b expected digests are full 64-hex of the local tree",
          all(len(v["sha256"]) == 64 for v in live.values())
          and {k: v["sha256"] for k, v in live.items()} == independent,
          f"{len(live)} files")

    # ---- PAR-13 git identity is never an input to the verdict -----------
    ev_src = ast.get_source_segment(src, next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "evaluate"))
    check("PAR-13 evaluate() references no git identity",
          "local_head" not in ev_src and "rev-parse" not in ev_src
          and "commit" not in ev_src)
    with tempfile.TemporaryDirectory() as td:
        rig = make_rig_tree(Path(td) / "rig")
        with ShimPath(SHIM_REAL):
            r1 = PG.probe_rig(target(rig))
        saved = PG.local_head
        try:
            PG.local_head = lambda *a, **k: "deadbeef" * 5
            ok_a, _ = PG.evaluate([r1], expected)
        finally:
            PG.local_head = saved
        ok_b, _ = PG.evaluate([r1], expected)
        check("PAR-13b verdict is unchanged by a fabricated HEAD",
              ok_a is True and ok_b is True)

    # ---- PAR-14 the canonical-tree precondition -------------------------
    # RED WHEN: a governed file is dirty locally, so the expectation would not
    # come from the canonical clean tree. Driven on a real throwaway git repo.
    with tempfile.TemporaryDirectory() as td:
        r = Path(td)
        subprocess.run(["git", "init", "-q", str(r)], check=True)
        subprocess.run(["git", "-C", str(r), "config", "user.email", "t@t"],
                       check=True)
        subprocess.run(["git", "-C", str(r), "config", "user.name", "t"],
                       check=True)
        (r / "g.py").write_text("x = 1\n")
        subprocess.run(["git", "-C", str(r), "add", "g.py"], check=True)
        subprocess.run(["git", "-C", str(r), "commit", "-qm", "i"], check=True)
        clean_lines, unav = PG.local_worktree_dirty(str(r), ("g.py",))
        check("PAR-14 clean control: committed file reports clean",
              clean_lines == [] and unav is None, f"{clean_lines}")
        (r / "g.py").write_text("x = 2\n")
        dirty_lines, unav2 = PG.local_worktree_dirty(str(r), ("g.py",))
        check("PAR-14b a modified governed file reports dirty",
              dirty_lines and unav2 is None, f"{dirty_lines}")

        def _boom(cmd, timeout):
            raise OSError("git is not installed")
        _, unav3 = PG.local_worktree_dirty(str(r), ("g.py",), runner=_boom)
        check("PAR-14c an unrunnable git is UNAVAILABLE, never a silent clean",
              unav3 is not None and "OSError" in unav3, unav3)

    # ---- PAR-15 end-to-end main(): PASS, REFUSE, and the evidence bundle -
    with tempfile.TemporaryDirectory() as td:
        rig_ok = make_rig_tree(Path(td) / "ok")
        rig_bad = make_rig_tree(Path(td) / "bad")
        v = Path(rig_bad) / "miner/range_miner_protocol.py"
        v.write_bytes(v.read_bytes() + b"\n# drift\n")
        ev = Path(td) / "evidence.json"
        saved = PG.gate_targets
        try:
            with ShimPath(SHIM_REAL):
                # --no-verify-clean here ONLY: this arm is about the fleet
                # verdict, and the canonical-tree precondition is covered on its
                # own real git fixture at PAR-14/PAR-16. Leaving it on would make
                # these arms red for an unrelated edit elsewhere in the tree.
                PG.gate_targets = lambda *a, **k: [
                    target(rig_ok, node_id="rigA", endpoint="10.0.0.1"),
                    target(rig_ok, node_id="rigB", endpoint="10.0.0.2")]
                rc_pass = PG.main(["--no-verify-clean",
                                   "--evidence-json", str(ev)])
                bundle = json.loads(ev.read_text())

                PG.gate_targets = lambda *a, **k: [
                    target(rig_ok, node_id="rigA", endpoint="10.0.0.1"),
                    target(rig_bad, node_id="rigB", endpoint="10.0.0.2")]
                rc_refuse = PG.main(["--no-verify-clean"])
        finally:
            PG.gate_targets = saved
        check("PAR-15 main() PROCEEDS (0) on an identical fleet",
              rc_pass == PG.EXIT_PROCEED, f"rc={rc_pass}")
        check("PAR-15b main() REFUSES (1) on one drifted rig",
              rc_refuse == PG.EXIT_REFUSE, f"rc={rc_refuse}")
        cols = {"hostname", "canonical_path", "expected_sha256",
                "observed_sha256", "observed_size", "verdict", "collected_at"}
        check("PAR-15c evidence bundle carries Beta's §C columns",
              bundle["rows"] and cols.issubset(set(bundle["rows"][0]))
              and len(bundle["rows"]) == 2 * len(PG.GOVERNED_FILES)
              and bundle["local_head_context_only"],
              f"{len(bundle['rows'])} rows")
        check("PAR-15d bundle freezes the gate's own governed set + closure",
              bundle["governed_files"] == list(PG.GOVERNED_FILES)
              and bundle["derived_closure"]
              and all(len(x["sha256"]) == 64
                      for x in bundle["expected"].values()))

    # ---- PAR-16 main() refuses when the local expectation is not canonical
    saved = PG.local_worktree_dirty
    try:
        PG.local_worktree_dirty = lambda *a, **k: ([" M prng_registry.py"], None)
        rc = PG.main([])
    finally:
        PG.local_worktree_dirty = saved
    check("PAR-16 main() REFUSES on a dirty local governed file",
          rc == PG.EXIT_REFUSE, f"rc={rc}")

    saved = PG.local_worktree_dirty
    try:
        PG.local_worktree_dirty = lambda *a, **k: ([], "git exited 128")
        rc = PG.main([])
    finally:
        PG.local_worktree_dirty = saved
    check("PAR-16b main() REFUSES when the local tree state is UNAVAILABLE",
          rc == PG.EXIT_REFUSE, f"rc={rc}")


# ═══════════════════════════════════════════════════════════════════════════
# PART B FIXTURES — a launcher harness with a real listener and a stub fleet
# ═══════════════════════════════════════════════════════════════════════════

# The pinned pre-fix anchor for the RED arm. FULL SHA256, never a prefix.
PREFIX_ANCHOR_COMMIT = "69ff2228bb19913183e08aaa735b85aa4a20516c"
PREFIX_ANCHOR_SHA256 = (
    "793c97ea5904315c92b56973b5a9ba321b72c530723459e008a9bcf32e39afc4")

STUB_WORKER = r'''#!/usr/bin/env python3
"""Stub range-miner worker: emits its sentinel, PARKS on the release file, and
connects only after release. It never touches a GPU and never speaks the real
protocol -- what is under test is the LAUNCHER'S WAIT SET, not the worker."""
import os, socket, sys, time

argv = sys.argv[1:]
def opt(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default

nonce = opt("--run-nonce")
release = opt("--session-release-file")
deadline = float(opt("--release-deadline", "900"))
print("SESSION_SENTINEL nonce=%s" % nonce, flush=True)
if os.environ.get("TFM_STUB_DIE_NOW"):
    print("STUB_DIED_IMMEDIATELY", flush=True)
    sys.exit(9)
if release:
    print("SESSION_RELEASE_WAIT", flush=True)
    t0 = time.time()
    while time.time() - t0 < deadline:
        if os.path.exists(release):
            print("SESSION_RELEASED", flush=True)
            break
        time.sleep(0.05)
    else:
        print("SESSION_RELEASE_ABORTED waited_s=%.3f" % (time.time() - t0),
              flush=True)
        sys.exit(3)
s = socket.create_connection((opt("--host"), int(opt("--port"))), timeout=5)
s.sendall(b"REGISTER")
print("REGISTERED", flush=True)
time.sleep(3600)
'''

# [D6-I1] THE PINNED ANCHOR FOR THE REMOTE-CHANNEL DEFECT. Distinct from the
# bare-`wait` anchor above and pinned separately, because they are two defects
# and one commit carried the first repair while still carrying the second:
# 3e1327b is the first FLEET-PROVEN state, and it is the state D6 dry run #2 ran.
CHANNEL_ANCHOR_COMMIT = "3e1327bddb62f9e223ca0ad8d084e3f228007271"
CHANNEL_ANCHOR_SHA256 = (
    "cda96d2ee3694028c6b2b94b57d765c13f4b8a74c5df1ee0e735ee5e0fdd5cae")

# ---------------------------------------------------------------------------
# THE SSH FIXTURES, AND WHY THERE ARE NOW THREE
# ---------------------------------------------------------------------------
# `STUB_SSH_OK` (`sleep .2; exit 0`) ENCODES THE PREMISE. It cannot fail on the
# condition the launcher's promptness claim is about, because an ssh that always
# returns in 0.2 s makes "the launcher returned promptly" true no matter what
# shape the remote command has. It is the seventh recorded instance of a check
# that could not fail on the condition it claimed to cover — and it was written
# AFTER Beta named that pattern.
#
# It is RETAINED, but only for ATTRIBUTION arms: where a fast-returning ssh is
# what isolates a different defect (WS-3 and M4 prove the bare `wait` blocks on
# the LOCAL worker, and a slow ssh would muddy which thing did the blocking). It
# may never again be the fixture under a green arm claiming promptness.
STUB_SSH_OK = """#!/usr/bin/env bash
sleep 0.2
exit 0
"""

# THE REAL-DEFECT FIXTURE. It does not simulate an outcome; it EXECUTES the
# remote command string and models the SSH CHANNEL, so the outcome is produced by
# the command's own shape:
#
#   * `bash -c "$cmd" 2>&1 | cat` — the pipe IS the channel. `cat` returns at
#     EOF, which is when the LAST process holding the write end exits, so any
#     process the remote command leaves attached to stdout/stderr keeps this
#     "ssh" open exactly as sshd keeps a real channel open.
#   * `${PIPESTATUS[0]}` — the remote command's own status is returned, so the
#     dispatch-status disposition R1 certified is still under test.
#
# Under the pre-fix shape (`mkdir && cd && worker … & echo started`) the `&`
# binds to the whole list, the forked subshell runs the worker in ITS foreground
# holding the pipe, and this fixture BLOCKS — the D6 dry run #2 behaviour,
# reproduced rather than asserted. Under the repaired shape only the worker is
# backgrounded and its three streams are detached, so the remote shell exits, the
# pipe reaches EOF and this returns promptly with the worker still alive.
#
# The two `/tmp` rewrites relocate the rig-absolute log and cache paths into the
# fixture tree. They move WHERE bytes land; they change nothing about which
# process holds which descriptor, which is the entire property under test.
STUB_SSH_EXEC = r"""#!/usr/bin/env bash
# fake ssh: last argument is the remote command
cmd="${@: -1}"
cmd="${cmd//\/tmp\/minerlogs/$TFM_STUB_REMOTE_ROOT\/minerlogs}"
cmd="${cmd//\/tmp\/cupy_cache_gpu/$TFM_STUB_REMOTE_ROOT\/cupy_cache_gpu}"
TFM_STUB_DIE_NOW="${TFM_STUB_DIE_REMOTE:-}" bash -c "$cmd" 2>&1 | cat
exit "${PIPESTATUS[0]}"
"""

STUB_SSH_FAIL = """#!/usr/bin/env bash
sleep 0.1
echo "ssh: could not resolve hostname" >&2
exit 255
"""


class Listener:
    """A real loopback listener. Zero accepts == no REGISTER occurred."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(64)
        self.port = self.sock.getsockname()[1]
        self.accepts = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        self.sock.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, addr = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self.accepts.append((time.time(), addr))
            try:
                conn.close()
            except OSError:
                pass

    def close(self):
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass


def build_launcher_fixture(td, launcher_bytes, ssh_shim=STUB_SSH_OK,
                           remote_gpus=2):
    """A temp tree the live launcher can run inside, unmodified."""
    root = Path(td)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    lp = root / "scripts" / "launch_fleet_manual.sh"
    lp.write_bytes(launcher_bytes)
    lp.chmod(lp.stat().st_mode | stat.S_IEXEC)

    (root / "miner").mkdir(parents=True, exist_ok=True)
    w = root / "miner" / "range_miner_worker.py"
    w.write_text(STUB_WORKER)

    (root / "rig_profiles_config.json").write_text(json.dumps({
        "default_profile": "proxmox",
        "profiles": ["baremetal", "proxmox"],
        "nodes": [
            {"node_id": "localhost", "config_hostname": "localhost",
             "worker_hostname": "stubhost", "local": True, "ssh_user": "michael",
             "endpoints": {"baremetal": "localhost", "proxmox": "localhost"}},
            {"node_id": "stubrig", "config_hostname": "10.255.255.1",
             "worker_hostname": "stubrig", "local": False, "ssh_user": "michael",
             "endpoints": {"baremetal": "10.255.255.1",
                           "proxmox": "10.255.255.1"}},
        ]}))
    # The remote node names a REAL interpreter and a REAL script path. Under
    # STUB_SSH_EXEC the remote command is actually executed, so `cd $SCRIPTPATH`
    # and the worker invocation have to be satisfiable — otherwise the fixture
    # would exercise a `cd` failure and call it a channel test. The other two
    # shims never run the command, so the values are inert for them.
    (root / "distributed_config.json").write_text(json.dumps({
        "nodes": [
            {"hostname": "localhost", "gpu_count": 1,
             "python_env": sys.executable, "script_path": str(root)},
            {"hostname": "10.255.255.1", "gpu_count": remote_gpus,
             "python_env": sys.executable, "script_path": str(root)},
        ]}))

    binp = root / "bin"
    binp.mkdir(exist_ok=True)
    sh = binp / "ssh"
    sh.write_text(ssh_shim)
    sh.chmod(sh.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return root


def run_launcher(root, listener, nonce="waitset-test", deadline=60,
                 wait_timeout=45, die_now=False, die_remote=False,
                 settle=None):
    """Start the launcher; return (proc, logdir, released_path, elapsed|None).

    `die_now` kills the LOCAL worker, `die_remote` the REMOTE one — two knobs,
    because one variable would make the local-liveness refusal and the remote
    dispatch-status refusal indistinguishable, and each arm needs the other path
    to be the one that is quiet.
    """
    logdir = Path(root) / "logs" / "miner_workers"
    env = dict(os.environ)
    env["PATH"] = f"{root / 'bin'}:{env.get('PATH', '')}"
    env["STAGGER"] = "0"
    env["RUN_NONCE"] = nonce
    env["RELEASE_DEADLINE"] = str(deadline)
    env["REMOTE_RELEASE_DIR"] = str(Path(root) / "remote_release")
    env["TFM_STUB_REMOTE_ROOT"] = str(Path(root) / "remote")
    if settle is not None:
        env["REMOTE_LAUNCH_SETTLE"] = str(settle)
    if die_now:
        env["TFM_STUB_DIE_NOW"] = "1"
    if die_remote:
        env["TFM_STUB_DIE_REMOTE"] = "1"
    out = open(Path(root) / "launcher.log", "wb")
    t0 = time.time()
    proc = subprocess.Popen(
        ["bash", str(Path(root) / "scripts" / "launch_fleet_manual.sh"),
         "127.0.0.1", str(listener.port), str(logdir)],
        stdout=out, stderr=subprocess.STDOUT, env=env, cwd=str(root),
        start_new_session=True)
    try:
        rc = proc.wait(timeout=wait_timeout)
        elapsed = time.time() - t0
    except subprocess.TimeoutExpired:
        rc, elapsed = None, None
    return proc, rc, elapsed, logdir


def kill_tree(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def local_worker_pids(logdir_root):
    """PIDs the launcher reported for its local worker, read from its own log."""
    text = (Path(logdir_root) / "launcher.log").read_text(errors="replace")
    return [int(m) for m in re.findall(r"^\[launch\]   pid=(\d+)$", text,
                                       re.M)]


def alive(pid):
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def remote_worker_pids(logdir):
    """[D6-I1] The PIDs the REMOTE side acked, read from the per-dispatch logs.

    These come from the launch ACK itself — `printf 'started pid=%s'` executed by
    the remote shell — so the test learns the worker's identity the same way the
    launcher does. There is no other way to know it: the worker is detached by
    design and is nobody's child by the time ssh returns, which is the property
    being proved.
    """
    pids = []
    for p in sorted(Path(logdir).glob("dispatch_*.log")):
        m = re.search(r"started pid=(\d+)", p.read_text(errors="replace"))
        if m:
            pids.append(int(m.group(1)))
    return pids


def reap(pids):
    for p in pids:
        try:
            os.kill(p, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def teardown(proc, root=None, logdir=None):
    """Kill the launcher's group AND the workers it left behind.

    `kill_tree` alone is NOT sufficient, and the reason is worth keeping: on the
    arms where the launcher REFUSES (WS-4, WS-5, LCH-6) it has already exited and
    been reaped by `run_launcher`, so `os.getpgid(proc.pid)` raises and the group
    kill silently does nothing — while the local stub worker, orphaned to PID 1,
    keeps running until its release deadline. Found by the liveness suite's own
    leak check: a stub from THIS suite was still alive minutes later, which is
    exactly the cross-suite contamination that makes a later run's /proc reads
    answer about the wrong process.
    """
    if proc is not None:
        kill_tree(proc)
    if root is not None:
        reap(local_worker_pids(root))
    if logdir is not None:
        reap(remote_worker_pids(logdir))


# ═══════════════════════════════════════════════════════════════════════════
# PART B — THE LAUNCHER WAIT SET
# ═══════════════════════════════════════════════════════════════════════════

def part_b():
    section("PART B — LAUNCHER WAIT SET (scripts/launch_fleet_manual.sh)")
    live_path = REPO / "scripts" / "launch_fleet_manual.sh"
    live_bytes = live_path.read_bytes()

    # ---- WS-0 RED AUTHENTICITY, run FIRST ------------------------------
    # Every RED arm below depends on the pinned anchor genuinely carrying the
    # defect. VIR-2: an anchor that has drifted is UNAVAILABLE, never a pass.
    try:
        pinned = subprocess.run(
            ["git", "-C", str(REPO), "show",
             f"{PREFIX_ANCHOR_COMMIT}:scripts/launch_fleet_manual.sh"],
            capture_output=True, check=True).stdout
    except Exception as exc:                                        # noqa: BLE001
        pinned = None
        unavailable("WS-0  pinned anchor is readable", f"{exc}")
    anchor_ok = False
    if pinned is not None:
        got = sha256_bytes(pinned)
        if got != PREFIX_ANCHOR_SHA256:
            unavailable("WS-0  anchor sha256 matches the pin",
                        f"{got} != {PREFIX_ANCHOR_SHA256}")
        elif not re.search(r"(?m)^wait$", pinned.decode()):
            unavailable("WS-0  anchor still carries the defect surface",
                        "no bare `wait` in the pinned bytes")
        else:
            anchor_ok = True
            check("WS-0  RED anchor authentic: full SHA + bare `wait` present",
                  True, f"{PREFIX_ANCHOR_COMMIT[:7]} sha={got[:16]}…")

    # ---- WS-1 structural: the wait set excludes the local worker --------
    # RED WHEN: an argument-less `wait` reappears, or a local worker PID is
    # passed to `wait`. Asserted on the LIVE bytes.
    text = live_bytes.decode()
    no_bare_wait = re.search(r"(?m)^\s*wait\s*$", text) is None
    waits_remote = 'wait "${REMOTE_DISPATCH_PIDS[$i]}"' in text
    # [D6-I1] TWO SHELLS NOW WAIT, AND ONLY ONE OF THEM IS THIS ONE. The remote
    # dispatch script contains `wait \$worker_pid` — escaped, because it is
    # expanded and executed on the RIG, where waiting on the just-launched worker
    # is how its immediate death becomes a nonzero dispatch. Counting it here
    # would make this arm red for the repair it is supposed to admit, so the two
    # are separated by the escaping that already distinguishes them.
    wait_lines = [ln for ln in text.splitlines()
                  if re.match(r"^\s*wait\b", ln) and "\\$" not in ln]
    remote_wait_lines = [ln for ln in text.splitlines()
                         if re.match(r"^\s*wait\b", ln) and "\\$" in ln]
    no_local_in_wait = all("LOCAL_WORKER_PIDS" not in ln
                           for ln in wait_lines + remote_wait_lines)
    check("WS-1  live launcher has NO argument-less `wait`", no_bare_wait)
    check("WS-1b live launcher waits on the remote dispatch PIDs",
          waits_remote and len(wait_lines) == 1,
          f"{len(wait_lines)} local wait stmt(s), "
          f"{len(remote_wait_lines)} remote (escaped) wait stmt(s)")
    check("WS-1c no local worker PID is ever passed to `wait`",
          no_local_in_wait)

    # self-protection: the same three assertions must FAIL on the anchor
    if anchor_ok:
        atext = pinned.decode()
        a_bare = re.search(r"(?m)^\s*wait\s*$", atext) is not None
        a_remote = 'wait "${REMOTE_DISPATCH_PIDS[$i]}"' in atext
        check("WS-1d SELF-PROTECTION: WS-1/1b are FALSE on the pinned anchor",
              a_bare and not a_remote,
              "the assertions have teeth")
    else:
        unavailable("WS-1d SELF-PROTECTION on the pinned anchor",
                    "anchor unavailable")

    # ---- WS-2 THE REQUIRED FOUR-PART GATE -------------------------------
    # With NO release token present, prove all four:
    #   launcher returns AND local worker still alive/parked
    #   AND no REGISTER has occurred AND no release token exists.
    # RED WHEN: the launcher blocks on its own parked local worker (the D6
    # measurement), or the local worker is dead when it returns, or anything
    # connected to the coordinator before release, or a token appeared.
    # [D6-I1] NOW RUN UNDER THE REAL-DEFECT FIXTURE, not `STUB_SSH_OK`. Under a
    # fake ssh that always returns in 0.2 s, "the launcher returned" was true by
    # construction and WS-2a could not fail on a remote channel that never
    # closed — which is precisely how the launcher shipped with the second half
    # of the defect intact. STUB_SSH_EXEC executes the real remote command and
    # models the channel, so promptness here is now a property of the launcher.
    listener = Listener()
    proc = None
    try:
        with tempfile.TemporaryDirectory() as td:
            root = build_launcher_fixture(td, live_bytes,
                                          ssh_shim=STUB_SSH_EXEC)
            proc, rc, elapsed, logdir = run_launcher(root, listener,
                                                     deadline=60,
                                                     wait_timeout=45)
            returned = rc is not None
            check("WS-2a launcher RETURNS while the fleet is parked",
                  returned and rc == 0,
                  f"rc={rc} elapsed={elapsed if elapsed is None else round(elapsed, 2)}s "
                  f"(release deadline 60s)")
            pids = local_worker_pids(root)
            still_alive = bool(pids) and all(alive(p) for p in pids)
            check("WS-2b local worker is ALIVE AND PARKED at return",
                  still_alive, f"pids={pids}")
            log = (Path(logdir) / "stubhost_gpu0.log").read_text(errors="replace")
            check("WS-2b2 …and parked at the barrier, not merely running",
                  "SESSION_RELEASE_WAIT" in log
                  and "SESSION_RELEASED" not in log
                  and "SESSION_RELEASE_ABORTED" not in log,
                  log.strip().splitlines()[-1] if log.strip() else "(empty)")
            check("WS-2c NO REGISTER has occurred",
                  listener.accepts == [], f"accepts={len(listener.accepts)}")
            tokens = list((Path(root) / "logs" / "miner_workers")
                          .glob("gate12_release_*"))
            tokens += list((Path(root) / "remote_release").glob("gate12_release_*")
                           ) if (Path(root) / "remote_release").exists() else []
            check("WS-2d NO release token exists",
                  tokens == [], f"tokens={tokens}")
            out = (Path(root) / "launcher.log").read_text(errors="replace")
            check("WS-2e completion states what it does and does not mean",
                  "DISPATCH COMPLETE" in out
                  and "does NOT mean the worker processes have exited"
                      .lower() in out.lower()
                  and "excluded from the wait set" in out)
            teardown(proc, root, logdir)
    finally:
        listener.close()

    # ---- WS-3 THE RED ARM: the pinned pre-fix shape fails WS-2a ----------
    # RED WHEN (i.e. what this arm proves): the pre-fix launcher does NOT
    # return while the local worker is parked. Bounded observation — the claim
    # is "still blocked at T", never a prediction about what it does at 900 s.
    if not anchor_ok:
        unavailable("WS-3  RED arm on the pinned pre-fix launcher",
                    "anchor unavailable — no pass may be reported")
    else:
        listener = Listener()
        proc = None
        try:
            with tempfile.TemporaryDirectory() as td:
                root = build_launcher_fixture(td, pinned)
                proc, rc, elapsed, logdir = run_launcher(
                    root, listener, deadline=60, wait_timeout=15)
                blocked = rc is None
                check("WS-3  RED: pre-fix launcher is STILL BLOCKED at 15 s",
                      blocked,
                      "the local worker is parked and `wait` waits for it"
                      if blocked else f"returned rc={rc} in {elapsed}s")
                pids = local_worker_pids(root)
                check("WS-3b RED: it is blocked WITH the local worker alive",
                      blocked and bool(pids) and all(alive(p) for p in pids),
                      f"pids={pids}")
                check("WS-3c RED: and with no REGISTER and no token",
                      listener.accepts == []
                      and not list((Path(root) / "logs" / "miner_workers")
                                   .glob("gate12_release_*")),
                      "the block is not a side effect of release")
                teardown(proc, root, logdir)
        finally:
            listener.close()

    # ---- WS-4 a failed remote dispatch is dispositioned, not counted -----
    # RED WHEN: an ssh that never landed is still counted as a dispatch. The
    # pre-fix loop counted iterations, so `25 of 25 dispatched` could be printed
    # over a fleet no ssh reached.
    listener = Listener()
    proc = None
    try:
        with tempfile.TemporaryDirectory() as td:
            root = build_launcher_fixture(td, live_bytes,
                                          ssh_shim=STUB_SSH_FAIL)
            proc, rc, elapsed, logdir = run_launcher(root, listener,
                                                     deadline=30,
                                                     wait_timeout=45)
            out = (Path(root) / "launcher.log").read_text(errors="replace")
            check("WS-4  a failing ssh makes the launcher REFUSE",
                  rc == 1 and "DISPATCH FAILED" in out
                  and "REMOTE DISPATCHES FAILED" in out, f"rc={rc}")
            check("WS-4b the refusal names the worker that did not land",
                  "stubrig(10.255.255.1):gpu0" in out)
            teardown(proc, root, logdir)
    finally:
        listener.close()

    # ---- WS-5 a dead local worker is REFUSED, not silently lost ----------
    # RED WHEN: the local worker dies during dispatch and the launcher still
    # returns 0 — the silent single-worker loss that freezes a cohort of 25 and
    # admits 24.
    listener = Listener()
    proc = None
    try:
        with tempfile.TemporaryDirectory() as td:
            # Real-defect fixture here too, and `die_now` is the LOCAL knob only:
            # the remote dispatches must genuinely SUCCEED, so the refusal can
            # only be the local one. An arm where both paths fail proves neither.
            root = build_launcher_fixture(td, live_bytes,
                                          ssh_shim=STUB_SSH_EXEC)
            proc, rc, elapsed, logdir = run_launcher(root, listener,
                                                     deadline=30,
                                                     wait_timeout=45,
                                                     die_now=True)
            out = (Path(root) / "launcher.log").read_text(errors="replace")
            check("WS-5  a dead local worker makes the launcher REFUSE",
                  rc == 1 and "LOCAL WORKER NOT ALIVE" in out
                  and "REMOTE DISPATCHES FAILED" not in out, f"rc={rc}")
            check("WS-5b the refusal explains the sentinel-gate consequence",
                  "sentinel gate would pass on a process that no longer exists"
                  in out)
            teardown(proc, root, logdir)
    finally:
        listener.close()

    # ---- WS-6 release is NOT moved earlier ------------------------------
    # Beta, binding: do not "fix" the wait by releasing sooner. The launcher
    # must still write no token; sentinel verification precedes release.
    creators = [ln for ln in text.splitlines()
                if "gate12_release_" in ln
                and not re.match(r"^\s*echo\b", ln)
                and re.search(r">\s*\"?\$?\{?[A-Za-z_/{}$]*gate12_release_"
                              r"|touch .*gate12_release_", ln)]
    check("WS-6  the launcher creates NO release token",
          creators == [], f"{creators}")
    check("WS-6b it still directs the operator to verify sentinels first",
          "--phase verify" in text and "THE FLEET IS PARKED AT THE RELEASE BARRIER"
          in text)

    # ---- WS-7 the frozen surfaces the repair must not have moved ---------
    # The dispatch mechanics themselves are unchanged: `ssh -n`, the fd-3 record
    # stream, the stagger, the PYTHONPATH and per-worker cache rules.
    for name, needle in (
            ("ssh -n stdin defence", 'ssh -n -o BatchMode=yes'),
            ("fd 3 record stream", 'done 3<<< "$FLEET"'),
            ("read from fd 3", 'read -r WHOST ENDPOINT NGPU PYBIN SCRIPTPATH KIND <&3'),
            ("PYTHONPATH rule", 'PYTHONPATH=$SCRIPTPATH'),
            ("per-worker cupy cache", 'CUPY_CACHE_DIR=/tmp/cupy_cache_gpu$N'),
            ("stagger paces dispatch", 'sleep "$STAGGER"')):
        check(f"WS-7  frozen: {name}", needle in text)


# ═══════════════════════════════════════════════════════════════════════════
# PART E — [D6-I1] REMOTE DISPATCH DETACHMENT, under the real-defect fixture
# ═══════════════════════════════════════════════════════════════════════════

def part_e():
    section("PART E — REMOTE DISPATCH DETACHMENT (D6-I1, long-lived channel)")
    live_path = REPO / "scripts" / "launch_fleet_manual.sh"
    live_bytes = live_path.read_bytes()
    text = live_bytes.decode()

    # ---- LCH-0 the channel anchor is authentic --------------------------
    # VIR-2: every RED arm below is worthless if the pinned commit does not
    # actually carry the defect. An anchor that has drifted is UNAVAILABLE.
    try:
        pinned = subprocess.run(
            ["git", "-C", str(REPO), "show",
             f"{CHANNEL_ANCHOR_COMMIT}:scripts/launch_fleet_manual.sh"],
            capture_output=True, check=True).stdout
    except Exception as exc:                                        # noqa: BLE001
        pinned = None
        unavailable("LCH-0 channel anchor is readable", f"{exc}")
    channel_anchor_ok = False
    if pinned is not None:
        got = sha256_bytes(pinned)
        ptext = pinned.decode()
        # The defect surface, stated as the shape and not as a hash: the worker
        # redirection ends the `&&` list with `&`, and the very next statement is
        # the `echo started` that supplied the success status.
        trap = re.search(r"2>&1 &\s*\\\s*\n\s*echo started", ptext) is not None
        if got != CHANNEL_ANCHOR_SHA256:
            unavailable("LCH-0 anchor sha256 matches the pin",
                        f"{got} != {CHANNEL_ANCHOR_SHA256}")
        elif not ("echo started" in ptext and trap):
            unavailable("LCH-0 anchor still carries the defect surface",
                        "the `… & echo started` precedence trap is absent")
        else:
            channel_anchor_ok = True
            check("LCH-0 RED anchor authentic: full SHA + `& echo started` trap",
                  True, f"{CHANNEL_ANCHOR_COMMIT[:7]} sha={got[:16]}…")

    # ---- LCH-1 THE RED ARM: the pinned remote command holds the channel --
    # RED WHEN (what this proves): with an ssh that models the channel, the
    # PINNED launcher does NOT return while its remote worker is parked. Bounded
    # observation — "still blocked at T", never a prediction about 900 s.
    if not channel_anchor_ok:
        unavailable("LCH-1 pinned remote command BLOCKS the launcher",
                    "anchor unavailable — no pass may be reported")
    else:
        listener = Listener()
        proc = None
        try:
            with tempfile.TemporaryDirectory() as td:
                root = build_launcher_fixture(td, pinned,
                                              ssh_shim=STUB_SSH_EXEC,
                                              remote_gpus=1)
                proc, rc, elapsed, logdir = run_launcher(
                    root, listener, deadline=60, wait_timeout=20)
                rlog = Path(root) / "remote" / "minerlogs" / "gpu0.log"
                parked = (rlog.exists()
                          and "SESSION_RELEASE_WAIT" in rlog.read_text(errors="replace")
                          and "SESSION_RELEASED" not in rlog.read_text(errors="replace"))
                check("LCH-1 RED: pinned remote command leaves the launcher "
                      "BLOCKED while the remote worker is parked",
                      rc is None and parked,
                      "the forked subshell runs the worker in its foreground "
                      "and holds the channel"
                      if rc is None else f"returned rc={rc} in {elapsed}s")
                teardown(proc, root, logdir)
        finally:
            listener.close()

    # ---- LCH-2..5, LCH-7 one repaired run, five properties --------------
    # RED WHEN: the worker is not detached (the launcher blocks, LCH-2), or the
    # detachment killed the worker (LCH-3), or anything registered before
    # release (LCH-4), or a token appeared (LCH-5), or success was declared
    # before every dispatch was dispositioned (LCH-7).
    listener = Listener()
    proc = None
    try:
        with tempfile.TemporaryDirectory() as td:
            root = build_launcher_fixture(td, live_bytes,
                                          ssh_shim=STUB_SSH_EXEC,
                                          remote_gpus=3)
            proc, rc, elapsed, logdir = run_launcher(root, listener,
                                                     deadline=60,
                                                     wait_timeout=45)
            out = (Path(root) / "launcher.log").read_text(errors="replace")
            rpids = remote_worker_pids(logdir)
            check("LCH-2 repaired: ssh dispatch RETURNS PROMPTLY while the "
                  "remote worker stays alive",
                  rc == 0 and elapsed is not None and elapsed < 15,
                  f"rc={rc} elapsed="
                  f"{'n/a' if elapsed is None else round(elapsed, 2)}s "
                  f"(release deadline 60s)")
            check("LCH-3 remote worker is STILL PARKED after ssh returned",
                  len(rpids) == 3 and all(alive(p) for p in rpids)
                  and all(
                      "SESSION_RELEASE_WAIT" in (Path(root) / "remote" /
                                                 "minerlogs" / f"gpu{i}.log"
                                                 ).read_text(errors="replace")
                      and "SESSION_RELEASED" not in
                      (Path(root) / "remote" / "minerlogs" / f"gpu{i}.log"
                       ).read_text(errors="replace")
                      for i in range(3)),
                  f"acked pids={rpids}")
            check("LCH-4 zero REGISTER", listener.accepts == [],
                  f"accepts={len(listener.accepts)}")
            tokens = list((Path(root) / "logs" / "miner_workers")
                          .glob("gate12_release_*"))
            if (Path(root) / "remote_release").exists():
                tokens += list((Path(root) / "remote_release")
                               .glob("gate12_release_*"))
            check("LCH-5 zero release token", tokens == [], f"tokens={tokens}")
            # LCH-7: ORDER, not just presence. Success may not be announced
            # before every dispatch is dispositioned and the local worker is
            # asserted alive.
            i_disp = out.find("remote dispatch jobs dispositioned: 3 (failures: 0)")
            i_local = out.find("ALIVE (excluded from the wait set)")
            i_done = out.find("DISPATCH COMPLETE")
            acks = len(re.findall(r"started pid=\d+", out))
            check("LCH-7 every remote dispatch dispositioned AND the local "
                  "worker asserted alive BEFORE success is announced",
                  -1 < i_disp < i_done and -1 < i_local < i_done and acks == 3,
                  f"disposition@{i_disp} local@{i_local} complete@{i_done} "
                  f"acks={acks}")
            teardown(proc, root, logdir)
    finally:
        listener.close()

    # ---- LCH-6 immediate remote startup failure -> NONZERO -> REFUSE ----
    # The fault-injection control, and the one Candidate A would have failed:
    # backgrounding the group and ending with `echo started` returns 0 here,
    # because the echo succeeded.
    listener = Listener()
    proc = None
    try:
        with tempfile.TemporaryDirectory() as td:
            root = build_launcher_fixture(td, live_bytes,
                                          ssh_shim=STUB_SSH_EXEC,
                                          remote_gpus=2)
            proc, rc, elapsed, logdir = run_launcher(root, listener,
                                                     deadline=30,
                                                     wait_timeout=45,
                                                     die_remote=True)
            out = (Path(root) / "launcher.log").read_text(errors="replace")
            check("LCH-6 immediate remote worker startup failure -> ssh returns "
                  "NONZERO and the launcher REFUSES",
                  rc == 1 and "DISPATCH FAILED" in out
                  and "REMOTE DISPATCHES FAILED" in out
                  and "LOCAL WORKER NOT ALIVE" not in out,
                  f"rc={rc}")
            check("LCH-6b the worker's own exit status survives the dispatch",
                  "ssh exited 9" in out,
                  "the stub worker exits 9; ssh must not launder it to 0")
            teardown(proc, root, logdir)
    finally:
        listener.close()

    # ---- LCH-8 structural: the precedence trap is gone ------------------
    # RED WHEN: `… & echo started` reappears in any form, or the worker line
    # stops being the only backgrounded unit.
    # Comment lines are stripped first: the repair's own comment quotes the trap
    # verbatim so a future reader knows what not to restore, and an assertion
    # that cannot tell code from the warning about it is not measuring the code.
    code_only = "\n".join(ln for ln in text.splitlines()
                          if not ln.lstrip().startswith("#"))
    check("LCH-8 the `& echo started` precedence trap is ABSENT from live source",
          "echo started" not in code_only
          and re.search(r"2>&1 &\s*\\\s*\n\s*echo started", code_only) is None)
    check("LCH-8b mkdir and cd are SYNCHRONOUS with their own exits",
          "|| exit $RC_REMOTE_MKDIR" in text and "|| exit $RC_REMOTE_CD" in text)
    check("LCH-8c the worker's three streams are detached from the channel",
          "< /dev/null > /tmp/minerlogs/gpu$N.log 2>&1 &" in text)
    check("LCH-8d a worker that exits 0 during settle is NOT a success",
          "RC_REMOTE_WORKER_EXITED_ZERO" in text
          and r'if [ \"\$rc\" -eq 0 ]; then rc=$RC_REMOTE_WORKER_EXITED_ZERO; fi'
          in text)
    check("LCH-8e the dispatch status is JOINED to a positive launch ACK",
          "sent NO launch ACK" in text
          # at least ONE digit: `[0-9]*` would also match an ACK printed with an
          # unset pid, which is a positive check that can be positive about
          # nothing. M6 is what found this.
          and "grep -o 'started pid=[0-9][0-9]*'" in text)


# ═══════════════════════════════════════════════════════════════════════════
# PART C — the harness's own integrity
# ═══════════════════════════════════════════════════════════════════════════

def part_c():
    section("PART C — HARNESS INTEGRITY")
    # The launcher under test in Part B is the LIVE file, byte-for-byte.
    live = REPO / "scripts" / "launch_fleet_manual.sh"
    with tempfile.TemporaryDirectory() as td:
        root = build_launcher_fixture(td, live.read_bytes())
        copied = root / "scripts" / "launch_fleet_manual.sh"
        check("HI-1  Part B executes the LIVE launcher bytes",
              sha256_file(copied) == sha256_file(live),
              sha256_file(live)[:16] + "…")
    # The parity gate under test is the live module, loaded from scripts/.
    check("HI-2  Part A imports the live parity gate module",
          Path(PG.__file__).resolve()
          == (REPO / "scripts" / "gate12_parity_gate.py").resolve(),
          PG.__file__)
    # Nothing in this suite may have written into the repository. Measured as a
    # DELTA against the porcelain captured before the first arm ran, so the arm
    # states what it means: this suite changed nothing.
    # HI-2b — NO STUB WORKER MAY OUTLIVE THIS SUITE.
    # It did. On every arm where the launcher REFUSES, `run_launcher` reaps it
    # before `kill_tree` runs, so the group kill silently no-ops and the local
    # stub worker is orphaned to PID 1 for the rest of its release deadline. The
    # liveness suite's own leak check found one alive minutes later. A leaked
    # worker is not cosmetic here: the next suite's /proc reads would answer
    # about the wrong process, and these suites already have to run sequentially.
    leaked = subprocess.run(
        ["pgrep", "-af", "miner/range_miner_worker.py --host 127.0.0.1"],
        capture_output=True, text=True).stdout.strip().splitlines()
    check("HI-2b no stub worker outlived the suite", not leaked,
          f"leaked={[ln.split()[0] for ln in leaked]}")

    now = _porcelain()
    appeared = [ln for ln in now if ln not in _PORCELAIN_AT_START]
    vanished = [ln for ln in _PORCELAIN_AT_START if ln not in now]
    check("HI-3  the suite changed NOTHING in the working tree",
          not appeared and not vanished,
          f"appeared={appeared} vanished={vanished}")


# ═══════════════════════════════════════════════════════════════════════════
# PART D — MUTANTS. Each proven APPLIED, EXECUTED and DETECTED.
# ═══════════════════════════════════════════════════════════════════════════

def _mutant_module(src, old, new, name):
    """Compile a source-level mutant of the parity gate into a fresh module.

    APPLIED is proven by asserting the substitution actually changed the text —
    a mutant that failed to apply is the classic vacuous mutation, and it looks
    identical to one that was killed.
    """
    import types
    assert old in src, f"mutation site not found for {name}"
    mutated = src.replace(old, new, 1)
    assert mutated != src
    mod = types.ModuleType(name)
    mod.__file__ = str(REPO / "scripts" / "gate12_parity_gate.py")
    exec(compile(mutated, mod.__file__, "exec"), mod.__dict__)
    return mod


def part_d():
    section("PART D — MUTANTS (applied · executed · detected)")
    src = (REPO / "scripts" / "gate12_parity_gate.py").read_text()
    expected = PG.expected_digests()

    # ---- M1: the digest comparison is removed ---------------------------
    m1 = _mutant_module(
        src,
        '            if digest == expected[rel]["sha256"]:\n                continue',
        '            if True:\n                continue',
        "pg_m1")
    with tempfile.TemporaryDirectory() as td:
        rig = make_rig_tree(Path(td) / "rig")
        v = Path(rig) / "miner/range_miner_worker.py"
        v.write_bytes(v.read_bytes() + b"\n# drift\n")
        with ShimPath(SHIM_REAL):
            r = PG.probe_rig(target(rig))
        real_ok, _ = PG.evaluate([r], expected)
        mut_ok, _ = m1.evaluate([r], expected)
    check("M1 digest comparison removed -> DETECTED",
          real_ok is False and mut_ok is True,
          "real REFUSES, mutant ALLOWS a drifted rig")

    # ---- M2: the truncation sentinel check is removed -------------------
    m2 = _mutant_module(
        src,
        "    if END_SENTINEL not in lines:",
        "    if False:",
        "pg_m2")
    with tempfile.TemporaryDirectory() as td:
        rig = make_rig_tree(Path(td) / "rig")
        with ShimPath(SHIM_TRUNCATED):
            real = PG.probe_rig(target(rig))
            try:
                mut_status = m2.probe_rig(target(rig))["status"]
            except Exception as exc:                                # noqa: BLE001
                # A mutant that CRASHES on truncated output is detected just as
                # surely as one that misclassifies it — and it shows the guard
                # is what keeps the parser from indexing a sentinel that is not
                # there. Either way it is not UNAVAILABLE.
                mut_status = f"raised:{type(exc).__name__}"
    check("M2 END-sentinel check removed -> DETECTED",
          real["status"] == PG.PROBE_UNAVAILABLE
          and mut_status != PG.PROBE_UNAVAILABLE,
          f"real={real['status']} mutant={mut_status}")

    # ---- M3: one governed file dropped from the pin ---------------------
    narrowed = tuple(f for f in PG.GOVERNED_FILES
                     if f != "miner/range_miner_coordinator.py")
    cov, unc, _ = PG.closure_coverage(governed=narrowed)
    check("M3 a governed file dropped from the pin -> DETECTED",
          (not cov) and "miner/range_miner_coordinator.py" in unc,
          f"uncovered={unc}")

    # ---- M4: the launcher's remote-PID wait becomes a bare `wait` -------
    live_bytes = (REPO / "scripts" / "launch_fleet_manual.sh").read_bytes()
    mutated = live_bytes.replace(
        b'wait "${REMOTE_DISPATCH_PIDS[$i]}"', b'wait', 1)
    if mutated == live_bytes:
        unavailable("M4 bare-`wait` regression -> DETECTED",
                    "mutation site not found in the live launcher")
    else:
        listener = Listener()
        proc = None
        try:
            with tempfile.TemporaryDirectory() as td:
                root = build_launcher_fixture(td, mutated)
                proc, rc, elapsed, logdir = run_launcher(root, listener,
                                                         deadline=60,
                                                         wait_timeout=15)
                check("M4 bare-`wait` regression -> DETECTED",
                      rc is None,
                      "WS-2a's assertion fails on the mutant (still blocked "
                      "at 15 s)" if rc is None else f"mutant returned rc={rc}")
                teardown(proc, root, logdir)
        finally:
            listener.close()

    # ---- M5: the remote launch detachment is REMOVED --------------------
    # The mutant Beta required: drop the `&` that backgrounds the worker, so the
    # worker runs in the REMOTE SHELL'S foreground and holds the channel — the
    # same end state the precedence trap produced by a different route. It must
    # be caught BY THE LONG-LIVED-CHANNEL FIXTURE; under STUB_SSH_OK it would
    # survive, which is the whole reason that fixture exists.
    mutated = live_bytes.replace(
        b"  < /dev/null > /tmp/minerlogs/gpu$N.log 2>&1 &\n",
        b"  < /dev/null > /tmp/minerlogs/gpu$N.log 2>&1\n", 1)
    if mutated == live_bytes:
        unavailable("M5 remote launch detachment removed -> DETECTED",
                    "mutation site not found in the live launcher")
    else:
        listener = Listener()
        proc = None
        try:
            with tempfile.TemporaryDirectory() as td:
                root = build_launcher_fixture(td, mutated,
                                              ssh_shim=STUB_SSH_EXEC,
                                              remote_gpus=1)
                proc, rc, elapsed, logdir = run_launcher(root, listener,
                                                         deadline=60,
                                                         wait_timeout=20)
                check("M5 remote launch detachment removed -> DETECTED",
                      rc is None,
                      "LCH-2's assertion fails on the mutant (still blocked at "
                      "20 s)" if rc is None else f"mutant returned rc={rc}")
                teardown(proc, root, logdir)
        finally:
            listener.close()

    # ---- M6: the truthful status is replaced by Candidate A's `echo` -----
    # Beta REJECTED Candidate A because `echo started` supplies the success
    # status even when the worker died at once. Encoded as a mutant so the
    # rejection is enforced rather than remembered: with the settle check and the
    # ACK replaced by a bare `echo started`, a worker that dies immediately still
    # produces rc=0 — and LCH-6's refusal disappears.
    src_text = live_bytes.decode()
    start = src_text.find("worker_pid=\\$!")
    ack = src_text.find("printf '[dispatch] started pid=")
    mutated6 = None
    if start >= 0 and ack > start:
        # Candidate A exactly: the settle check is deleted and the ACK is
        # printed unconditionally from `$!`. It still carries a real pid, so the
        # ACK join alone does not kill it — only the settle-status check does,
        # which is the property Beta's rejection turns on.
        cand = src_text[:start] + src_text[ack:]
        cand = cand.replace(r'\"\$worker_pid\"', r'\"\$!\"', 1)
        if cand != src_text and r'\"\$!\"' in cand:
            mutated6 = cand.encode()
    if mutated6 is None:
        unavailable("M6 Candidate-A success-by-echo -> DETECTED",
                    "mutation site not found in the live launcher")
    else:
        listener = Listener()
        proc = None
        try:
            with tempfile.TemporaryDirectory() as td:
                root = build_launcher_fixture(td, mutated6,
                                              ssh_shim=STUB_SSH_EXEC,
                                              remote_gpus=1)
                proc, rc, elapsed, logdir = run_launcher(root, listener,
                                                         deadline=30,
                                                         wait_timeout=45,
                                                         die_remote=True)
                out = (Path(root) / "launcher.log").read_text(errors="replace")
                check("M6 Candidate-A success-by-echo -> DETECTED",
                      not (rc == 1 and "REMOTE DISPATCHES FAILED" in out),
                      f"the mutant reports rc={rc} for a worker that died at "
                      f"once — LCH-6 REFUSES on live source")
                teardown(proc, root, logdir)
        finally:
            listener.close()


def main():
    print("=" * 78)
    print("S172 D6 INTEGRATION REPAIR — CODE-PARITY GATE + LAUNCHER WAIT SET")
    print("=" * 78)
    part_a()
    part_b()
    part_e()
    part_d()
    part_c()

    total = len(_RESULTS)
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    unav = sum(1 for _, _, u in _RESULTS if u)
    section("RESULT")
    for name, ok, u in _RESULTS:
        if not ok:
            print(f"  {'UNAVAILABLE' if u else 'FAILED'}: {name}")
    print(f"\n  {passed}/{total} checks green"
          + (f"  ({unav} UNAVAILABLE)" if unav else ""))
    if passed == total:
        print("COMPLETION SENTINEL: PASS — S172 D6 integration repair, "
              "the parity gate and the launcher wait-set battery are green")
        return 0
    print("COMPLETION SENTINEL: FAIL — see the list above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
