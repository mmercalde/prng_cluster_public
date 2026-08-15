#!/usr/bin/env python3
"""GATE-12 CLEAN-TREE ADMISSION REPAIR — C1-C5 plus C5A (Beta 2026-08-11).

FALSIFIABLE QUESTION
  Does the Gate-12 harness refuse to launch from any repository state that D3.5
  publication would later reject for cleanliness alone — and does it still hold
  that state all the way to fleet dispatch, including across its own launch
  preparation?

WHAT ATTEMPT 3 DID (the state these gates are built against)
  `distributed_config_t1_d606edbe`, 2026-08-10. Four stages, 128/128 stripes,
  full [0,2^31), saturation SATISFIED, then:

      utils.run_finalizer.RunParameterError: repository_tree_clean is False

  The three untracked entries responsible are frozen verbatim in
  `/home/michael/gate12_attempt3_20260810_200824/git_status_porcelain.txt`:
  `miner_ledger.db-shm`, `miner_ledger.db-wal`,
  `optimal_window_config.json.stale_1786149572`. The SAME three appear in that
  run's launch-time evidence block under `--- TREE STATE ---`. The old harness
  printed the reason it was going to fail and launched anyway.

NO FLEET, NO GPU, NO COORDINATOR IS EVER CONTACTED.
  C1 executes the REAL `gate12_launch.sh` end-to-end inside a sandboxed HOME,
  behind recording shims for `pkill`, `ssh`, `setsid` and `ss`. The witness file
  those shims write IS the "zero dispatch" evidence: the repaired script leaves
  it empty, the committed pre-repair script does not. The script under test is
  never modified; only its transport and its HOME are.

THE RED ARMS ARE THE POINT (VIR-2)
  A gate that cannot fail on the real failing state proves nothing. Two of the
  arms below run the COMMITTED PRE-REPAIR SOURCE rather than a retyped copy:
    C1-RED   the old admission path, on the exact attempt-3 state, admits.
    C5A-RED  the old `pregate12` rename, on a clean tree, dirties it.

  [R1, Beta 2026-08-11] That source is pinned to an IMMUTABLE COMMIT, not to
  `HEAD`. `git show HEAD:gate12_launch.sh` was the pre-repair script only while
  HEAD sat at 3254a306; the instant the repair is committed, HEAD becomes the
  REPAIRED script and both RED arms would silently start proving nothing —
  C1-RED would run the repaired launch, C1-RED-NO-TREE-GUARD would inspect the
  new gate and find the guard it expects absent, and C5A-RED might extract no
  rotation at all. A permanent adversarial suite that goes vacuous on commit is
  worse than no suite. `_launch_source_at()` therefore resolves the pinned
  object and REFUSES to return it unless BOTH old defect surfaces are still
  present in its executable lines, so a wrong or drifted anchor reports
  UNAVAILABLE instead of crediting a RED arm. The RED tests are green both
  before and after the repair commit; the post-repair case is demonstrated by
  running this suite inside a scratch clone whose HEAD carries the repair.

THE C1 FIXTURE IS DERIVED FROM THE FROZEN BUNDLE, NOT TRANSCRIBED
  [R2, Beta 2026-08-11] A gate named "exact attempt-3 reproduction" may not
  hard-code the entries it claims to reproduce. `attempt3_entries()` reads
  `git_status_porcelain.txt` out of the frozen evidence bundle, verifies that
  file against the bundle's OWN `SHA256SUMS.txt` line for it, parses the
  porcelain, and asserts cardinality and status. Absent, unreadable, tampered
  or unexpected evidence reports UNAVAILABLE — it never falls back to
  constants. The bundle is READ-ONLY: this suite opens exactly two files in it
  for reading and writes nothing.

Run:  source ~/venvs/torch/bin/activate && \
      python3 -u tests/test_gate12_cleantree_admission.py
"""

import ast
import hashlib
import inspect
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import window_optimizer_integration_final as WOI                    # noqa: E402
import gate12_cleantree_gate as G                                   # noqa: E402
from utils import run_finalizer as RF                               # noqa: E402

GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"
_RESULTS = []

LAUNCH = REPO / "gate12_launch.sh"
GATE = REPO / "scripts" / "gate12_cleantree_gate.py"

# The frozen attempt-3 evidence bundle. READ-ONLY: exactly two files are opened
# for reading (`git_status_porcelain.txt` and `SHA256SUMS.txt`) and nothing is
# ever written into this directory.
ATTEMPT3_BUNDLE = Path("/home/michael/gate12_attempt3_20260810_200824")
ATTEMPT3_PORCELAIN = ATTEMPT3_BUNDLE / "git_status_porcelain.txt"
ATTEMPT3_SUMS = ATTEMPT3_BUNDLE / "SHA256SUMS.txt"
ATTEMPT3_EXPECTED_ENTRY_COUNT = 3

# [R1] The immutable pre-repair anchor. NOT `HEAD` — see the module docstring.
# This is the commit attempt 3 actually ran from: it is `HEAD.txt` in the frozen
# bundle and the SHA quoted in that run's own RunParameterError.
PRE_REPAIR_COMMIT = "3254a306ee0abdf02465b2cd3cd6793650911893"


def check(name, ok, detail=""):
    tag = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    _RESULTS.append((name, bool(ok)))
    print(f"  [{tag}] {name:<34} {detail}", flush=True)


def check_unavailable(name, reason):
    """VIR-3: UNAVAILABLE terminates the gate and does NOT accept.

    Rendered distinctly from FAIL so an operator can tell "the check ran and the
    property does not hold" from "the check could not run at all", while both
    stop the suite from reporting green.
    """
    _RESULTS.append((name, False))
    print(f"  [{RED}UNAV{RESET}] {name:<34} UNAVAILABLE: {reason}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# fixtures
# ──────────────────────────────────────────────────────────────────────────────

# `.gitignore` for the fixture repos. It carries ONLY the two rules this repair
# depends on, copied in meaning from the live file: the generated Step-1 output
# is ignored (live `.gitignore:115`) and `logs/` is ignored as a whole directory
# (live `.gitignore:62`). Nothing resembling a runtime-residue filename
# exception is present — a fixture that quietly ignored `*.stale_*` would make
# C1 vacuous.
FIXTURE_GITIGNORE = "optimal_window_config.json\nlogs/\n"


def _git(repo, *args, check_rc=True):
    p = subprocess.run(["git", "-C", str(repo), *args],
                       capture_output=True, text=True)
    if check_rc and p.returncode != 0:
        raise RuntimeError(f"git {args} -> {p.returncode}: {p.stderr}")
    return p


def make_repo(root, tracked=None):
    """A real git repository with one commit and a clean tree."""
    root = Path(root)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / ".gitignore").write_text(FIXTURE_GITIGNORE)
    (root / "production_module.py").write_text("VALUE = 1\n")
    (root / "tests" / "test_governance.py").write_text("GATE = 'green'\n")
    for rel, body in (tracked or {}).items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "gate12@fixture.local")
    _git(root, "config", "user.name", "gate12 fixture")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fixture baseline")
    return root


def porcelain(repo):
    return _git(repo, "status", "--porcelain").stdout


def run_gate(repo, phase="admission"):
    """Invoke the real gate script as the launch harness does."""
    env = dict(os.environ, PYTHONPATH=str(REPO))
    p = subprocess.run(
        [sys.executable, "-u", str(GATE), "--repo-root", str(repo),
         "--phase", phase],
        capture_output=True, text=True, env=env)
    return p.returncode, p.stdout + p.stderr


# ──────────────────────────────────────────────────────────────────────────────
# extraction of REAL source regions — never a paraphrase
# ──────────────────────────────────────────────────────────────────────────────

_ROT_BEGIN = "# --- GATE12-CONFIG-ROTATION BEGIN ---"
_ROT_END = "# --- GATE12-CONFIG-ROTATION END ---"


def extract_rotation_region(text):
    """The live clean-slate config rotation, verbatim, between its markers."""
    if _ROT_BEGIN not in text or _ROT_END not in text:
        return None
    return text.split(_ROT_BEGIN, 1)[1].split(_ROT_END, 1)[0]


# ── [R1] the pinned pre-repair anchor, and its integrity ──────────────────────

class AnchorUnavailable(Exception):
    """The pinned pre-repair source could not be obtained, or is not it."""


def _code_lines(text):
    """Executable lines only.

    The defect surfaces are STATEMENTS. The repaired script quotes both of them
    verbatim in its own header comments explaining what it fixed, so a probe run
    over raw text would match the repaired script and be blind to exactly the
    drift it exists to detect. Stripping comments is what makes the probes
    discriminate.
    """
    return "\n".join(ln for ln in text.splitlines()
                     if not ln.lstrip().startswith("#"))


# The two pre-repair defect surfaces Beta named. Both must be present in the
# pinned object's executable lines, or the anchor is not the pre-repair script.
_PRE_REPAIR_SURFACES = {
    "untested `--- TREE STATE ---` porcelain print":
        lambda code: re.search(
            r'echo "--- TREE STATE ---";\s*git status --porcelain', code)
        is not None,
    "self-dirtying `optimal_window_config.json.pregate12_${STAMP}` rename":
        lambda code: re.search(
            r"mv\s+optimal_window_config\.json\s+"
            r"optimal_window_config\.json\.pregate12_", code) is not None,
}


def _missing_surfaces(text):
    code = _code_lines(text)
    return [name for name, probe in _PRE_REPAIR_SURFACES.items()
            if not probe(code)]


def _launch_source_at(commit, repo=None):
    """`gate12_launch.sh` at `commit`, refusing anything that is not pre-repair.

    Two independent ways to be wrong are both closed here:
      * the object does not resolve (bad anchor, shallow clone, missing file)
      * the object resolves but has drifted off the pre-repair script
    Either raises. Nothing downstream can credit a RED arm from a source that
    does not carry the defect the RED arm exists to demonstrate.
    """
    repo = repo or REPO
    p = _git(repo, "show", f"{commit}:gate12_launch.sh", check_rc=False)
    if p.returncode != 0:
        raise AnchorUnavailable(
            f"pinned object {commit}:gate12_launch.sh does not resolve in "
            f"{repo} (git exit {p.returncode}: {p.stderr.strip()!r})")
    missing = _missing_surfaces(p.stdout)
    if missing:
        raise AnchorUnavailable(
            f"object {commit}:gate12_launch.sh resolved ({len(p.stdout)} bytes) "
            f"but is NOT the pre-repair script — missing defect surface(s): "
            f"{missing}")
    return p.stdout


def pre_repair_launch_source():
    """The COMMITTED pre-repair launch script, pinned and integrity-checked."""
    return _launch_source_at(PRE_REPAIR_COMMIT)


def extract_old_rotation(head_text):
    """The pre-repair rename, verbatim, from the committed script."""
    lines = head_text.splitlines()
    for i, ln in enumerate(lines):
        if "optimal_window_config.json.pregate12_" in ln:
            start = i - 1 if i and lines[i - 1].rstrip().endswith("\\") else i
            return "\n".join(lines[start:i + 1])
    return None


def extract_old_treestate(head_text):
    """The pre-repair admission 'check': an echo and an untested porcelain."""
    for ln in head_text.splitlines():
        if "TREE STATE" in ln and "git status --porcelain" in ln:
            return ln
    return None


# ── [R2] the C1 fixture, DERIVED from the frozen evidence ─────────────────────

class FrozenEvidenceUnavailable(Exception):
    """The frozen attempt-3 evidence could not be used to build the fixture."""


_ATTEMPT3_CACHE = {}


def _bundle_recorded_sha(path):
    """The bundle's OWN recorded digest for one of its files.

    Self-authenticating: the fixture is not merely read out of a directory whose
    name looks right, it is read out of a file that matches the digest the
    bundle itself recorded. A swapped or edited evidence file cannot silently
    become the fixture.
    """
    if not ATTEMPT3_SUMS.is_file():
        raise FrozenEvidenceUnavailable(f"{ATTEMPT3_SUMS} is absent")
    try:
        text = ATTEMPT3_SUMS.read_text()
    except OSError as e:
        raise FrozenEvidenceUnavailable(f"{ATTEMPT3_SUMS} unreadable: {e}")
    want = str(path)
    for ln in text.splitlines():
        parts = ln.split(None, 1)
        if len(parts) == 2 and parts[1].strip() == want:
            return parts[0].strip()
    raise FrozenEvidenceUnavailable(
        f"{ATTEMPT3_SUMS} records no digest for {want}")


def attempt3_entries(porcelain_path=None, sums_required=True):
    """The attempt-3 residue, parsed from the frozen bundle. Never transcribed.

    Returns a list of (status, path). Raises FrozenEvidenceUnavailable on an
    absent, unreadable, tampered, unparseable or unexpected-shape evidence file
    — it never substitutes constants.
    """
    path = Path(porcelain_path) if porcelain_path else ATTEMPT3_PORCELAIN
    key = (str(path), sums_required)
    if key in _ATTEMPT3_CACHE:
        cached = _ATTEMPT3_CACHE[key]
        if isinstance(cached, Exception):
            raise cached
        return cached
    try:
        if not path.is_file():
            raise FrozenEvidenceUnavailable(f"{path} is absent")
        try:
            raw = path.read_bytes()
        except OSError as e:
            raise FrozenEvidenceUnavailable(f"{path} unreadable: {e}")

        if sums_required:
            recorded = _bundle_recorded_sha(path)
            actual = hashlib.sha256(raw).hexdigest()
            if actual != recorded:
                raise FrozenEvidenceUnavailable(
                    f"{path} does not match the digest the bundle records for "
                    f"it: recorded {recorded[:16]}…, actual {actual[:16]}…")

        entries = []
        for ln in raw.decode().splitlines():
            if not ln.strip():
                continue
            m = re.match(r"^(..) (.+)$", ln)
            if not m:
                raise FrozenEvidenceUnavailable(
                    f"unparseable porcelain line in {path}: {ln!r}")
            status, rel = m.group(1), m.group(2).strip()
            if " -> " in rel:
                raise FrozenEvidenceUnavailable(
                    f"rename entry in {path} is not a shape this fixture "
                    f"builder handles: {ln!r}")
            if rel.startswith("/") or ".." in Path(rel).parts:
                raise FrozenEvidenceUnavailable(
                    f"unsafe fixture path in {path}: {rel!r}")
            entries.append((status, rel))

        if len(entries) != ATTEMPT3_EXPECTED_ENTRY_COUNT:
            raise FrozenEvidenceUnavailable(
                f"{path} holds {len(entries)} entries, expected "
                f"{ATTEMPT3_EXPECTED_ENTRY_COUNT}")
        untracked = [e for e in entries if e[0] == "??"]
        if len(untracked) != len(entries):
            raise FrozenEvidenceUnavailable(
                f"{path} holds non-untracked entries: "
                f"{[e for e in entries if e[0] != '??']}")
    except FrozenEvidenceUnavailable as e:
        _ATTEMPT3_CACHE[key] = e
        raise
    _ATTEMPT3_CACHE[key] = entries
    return entries


def attempt3_paths():
    return [rel for _status, rel in attempt3_entries()]


def place_residue(repo, rel_paths):
    """Materialize the derived fixture paths, whatever shape they turn out to be."""
    for rel in rel_paths:
        p = Path(repo) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("residue\n")


# ──────────────────────────────────────────────────────────────────────────────
# sandboxed end-to-end execution of the REAL launch script
# ──────────────────────────────────────────────────────────────────────────────

_SHIM = """#!/usr/bin/env bash
echo "{name} $*" >> "$GATE12_WITNESS"
exit 0
"""

# `ss` must report the coordinator port as unbound, so that even if a run ever
# got that far it would take the "COORDINATOR NEVER BOUND" branch rather than
# calling the fleet launcher. Belt and braces: the witness proves it never gets
# near either.
_SS_SHIM = """#!/usr/bin/env bash
echo "ss $*" >> "$GATE12_WITNESS"
exit 0
"""

_GPU_GATE_STUB = """#!/usr/bin/env python3
# Fixture stand-in for the certified GPU gate. It PASSES unconditionally so the
# old script is free to walk past §0.5 into the clean slate — which is exactly
# what C1-RED needs to observe. The real GPU gate is covered by
# tests/test_gate12_gpu_gate.py and is not under test here.
import os, sys
open(os.environ["GATE12_WITNESS"], "a").write("gpu_gate_stub\\n")
print("GATE-12 GPU GATE  : PASS (fixture stub)")
sys.exit(0)
"""


def build_sandbox(tmp, launch_text, dirty_entries=(), with_config=False):
    """A sandboxed HOME containing a fixture repo and the script under test.

    `gate12_launch.sh` opens with `cd ~/distributed_prng_analysis`, so HOME is
    redirected rather than the script edited. PYTHONPATH points at the real
    repository so the gate imports the REAL `_repository_state`; `--repo-root`
    still defaults to the gate script's own location, i.e. the fixture.
    """
    home = Path(tmp) / "home"
    repo = home / "distributed_prng_analysis"
    repo.mkdir(parents=True)
    make_repo(repo)

    (repo / "gate12_launch.sh").write_text(launch_text)
    shutil.copy2(GATE, repo / "scripts" / "gate12_cleantree_gate.py")
    (repo / "scripts" / "gate12_gpu_gate.py").write_text(_GPU_GATE_STUB)
    (repo / "scripts" / "gate12_concurrency_sampler.py").write_text(
        "#!/usr/bin/env python3\nimport time\ntime.sleep(600)\n")
    (repo / "scripts" / "launch_fleet_manual.sh").write_text(
        "#!/usr/bin/env bash\necho \"launch_fleet_manual $*\" "
        ">> \"$GATE12_WITNESS\"\n")
    (repo / "scripts" / "launch_fleet_manual.sh").chmod(0o755)
    # The evidence block imports this; a stub keeps the block's failure from
    # being about a missing module. It is inside `{ ... } | tee` either way.
    (repo / "database_system.py").write_text(
        "class DistributedPRNGDatabase:\n"
        "    def get_certified_cursor(self, *a, **k):\n"
        "        return 'fixture-cursor'\n")

    venv = home / "venvs" / "torch" / "bin"
    venv.mkdir(parents=True)
    (venv / "activate").write_text("# fixture venv, intentionally inert\n")

    shims = Path(tmp) / "shims"
    shims.mkdir()
    for name in ("pkill", "ssh", "setsid", "nohup"):
        p = shims / name
        p.write_text(_SHIM.format(name=name))
        p.chmod(0o755)
    (shims / "ss").write_text(_SS_SHIM)
    (shims / "ss").chmod(0o755)

    if with_config:
        (repo / "optimal_window_config.json").write_text('{"fixture": true}\n')
    place_residue(repo, dirty_entries)

    witness = Path(tmp) / "witness.txt"
    witness.write_text("")
    env = dict(os.environ)
    env.update(HOME=str(home),
               PATH=f"{shims}:{env['PATH']}",
               PYTHONPATH=str(REPO),
               GATE12_WITNESS=str(witness))
    return repo, witness, env


def run_launch(tmp, launch_text, dirty_entries=(), with_config=False):
    repo, witness, env = build_sandbox(tmp, launch_text, dirty_entries,
                                       with_config)
    # Prove the shims win the PATH race before anything destructive could run.
    probe = subprocess.run(["bash", "-c", "command -v pkill ssh setsid ss"],
                           capture_output=True, text=True, env=env)
    shimdir = str(Path(tmp) / "shims")
    if not all(ln.startswith(shimdir) for ln in probe.stdout.split()):
        raise RuntimeError(f"shims did not win PATH: {probe.stdout!r}")

    p = subprocess.run(["bash", str(repo / "gate12_launch.sh")],
                       capture_output=True, text=True, env=env,
                       cwd=str(repo), timeout=300)
    return {"rc": p.returncode,
            "out": p.stdout + p.stderr,
            "witness": witness.read_text(),
            "repo": repo,
            "porcelain": porcelain(repo)}


# ──────────────────────────────────────────────────────────────────────────────
# C1 — exact attempt-3 reproduction
# ──────────────────────────────────────────────────────────────────────────────

def r2_fixture_is_derived_from_frozen_evidence():
    """[R2] The fixture comes out of the bundle, digest-checked, or not at all."""
    try:
        entries = attempt3_entries()
    except FrozenEvidenceUnavailable as e:
        check_unavailable("R2-FIXTURE-DERIVED", str(e))
        return
    check("R2-FIXTURE-DERIVED", True,
          f"{len(entries)} entries parsed from {ATTEMPT3_PORCELAIN.name} "
          f"(digest matches the bundle's own SHA256SUMS line), all '??': "
          f"{', '.join(rel for _s, rel in entries)}")


def c1_attempt3_state_is_refused():
    try:
        paths = attempt3_paths()
    except FrozenEvidenceUnavailable as e:
        for n in ("C1-REFUSED", "C1-ZERO-DISPATCH", "C1-PUBLICATION-REASON"):
            check_unavailable(n, f"fixture not derivable: {e}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        r = run_launch(tmp, LAUNCH.read_text(), dirty_entries=paths)
        named = all(p in r["out"] for p in paths)
        check("C1-REFUSED",
              r["rc"] != 0 and "REFUSED" in r["out"] and named,
              f"rc={r['rc']}, all {len(paths)} derived attempt-3 entries named "
              f"in the refusal")
        check("C1-ZERO-DISPATCH",
              r["witness"].strip() == "",
              "no pkill, no ssh, no setsid, no sampler, no coordinator, no "
              "fleet launcher — witness file empty")
        check("C1-PUBLICATION-REASON",
              "run_finalizer.py:1589" in r["out"]
              and "repository_tree_clean" in r["out"],
              "refusal states that publication would reject this state, and "
              "names the enforcement site")


def r1_anchor_integrity():
    """[R1] The pinned object exists AND still carries both defect surfaces."""
    try:
        src = pre_repair_launch_source()
    except AnchorUnavailable as e:
        check_unavailable("R1-ANCHOR-INTEGRITY", str(e))
        return
    head = _git(REPO, "rev-parse", "HEAD").stdout.strip()
    posture = ("HEAD is still the pinned pre-repair commit"
               if head == PRE_REPAIR_COMMIT
               else f"HEAD has moved to {head[:12]}… — the pin is what keeps "
                    f"the RED arms alive")
    check("R1-ANCHOR-INTEGRITY", True,
          f"{PRE_REPAIR_COMMIT[:12]}…:gate12_launch.sh resolved ({len(src)} "
          f"bytes), both defect surfaces present; {posture}")


def c1_red_old_preflight_admits():
    """The pre-repair script, on the exact attempt-3 state, does NOT refuse."""
    try:
        old = pre_repair_launch_source()
    except AnchorUnavailable as e:
        check_unavailable("C1-RED-OLD-ADMITS", str(e))
        check_unavailable("C1-RED-NO-TREE-GUARD", str(e))
        return
    try:
        paths = attempt3_paths()
    except FrozenEvidenceUnavailable as e:
        check_unavailable("C1-RED-OLD-ADMITS", f"fixture not derivable: {e}")
        check_unavailable("C1-RED-NO-TREE-GUARD", f"fixture not derivable: {e}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        r = run_launch(tmp, old, dirty_entries=paths)
        printed = all(p in r["out"] for p in paths)
        got_past = "pkill" in r["witness"]
        check("C1-RED-OLD-ADMITS",
              printed and got_past,
              f"pinned pre-repair script PRINTED all {len(paths)} entries and "
              f"still reached the clean slate (witness: "
              f"{len(r['witness'].splitlines())} shim calls)")
        check("C1-RED-NO-TREE-GUARD",
              "cleantree" not in old
              and not re.search(r"if[^\n]*git status --porcelain", old),
              "no conditional anywhere in the pinned pre-repair script tests "
              "the porcelain output")


# ──────────────────────────────────────────────────────────────────────────────
# C2 / C3 / C4 — the rest of the input space
# ──────────────────────────────────────────────────────────────────────────────

def c2_clean_repo_passes():
    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo(Path(tmp) / "r")
        rc, out = run_gate(repo)
        check("C2-CLEAN-PASSES",
              rc == 0 and "PASS" in out and porcelain(repo) == "",
              "git status --porcelain empty -> rc=0")


def c3_modified_production_file_refused():
    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo(Path(tmp) / "r")
        (repo / "production_module.py").write_text("VALUE = 2\n")
        rc, out = run_gate(repo)
        check("C3-MODIFIED-PROD-REFUSED",
              rc != 0 and "production_module.py" in out,
              "a modified TRACKED production file refuses and is named")


def c4_modified_governance_file_refused():
    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo(Path(tmp) / "r")
        (repo / "tests" / "test_governance.py").write_text("GATE = 'red'\n")
        rc, out = run_gate(repo)
        check("C4-MODIFIED-TEST-REFUSED",
              rc != 0 and "tests/test_governance.py" in out,
              "a modified TRACKED test/governance file refuses and is named")


def c_extra_unobservable_refuses():
    """VIR-5: a predicate that could not be evaluated is not one that passed."""
    with tempfile.TemporaryDirectory() as tmp:
        not_a_repo = Path(tmp) / "plain"
        not_a_repo.mkdir()
        rc, out = run_gate(not_a_repo)
        check("C-VIR5-UNAVAILABLE-REFUSES",
              rc != 0 and "UNAVAILABLE" in out,
              "producer raised; reported UNAVAILABLE and refused, never clean")


def c_extra_decide_input_space():
    """Every value `decide` can receive, not only the one that motivated it."""
    rows = [(True, G.EXIT_PROCEED), (False, G.EXIT_REFUSE),
            (None, G.EXIT_REFUSE), (1, G.EXIT_REFUSE),
            ("", G.EXIT_REFUSE), ([], G.EXIT_REFUSE)]
    ok = all(G.decide(v) == want for v, want in rows)
    sig = inspect.signature(G.decide)
    check("C-DECIDE-INPUT-SPACE",
          ok and len(sig.parameters) == 1,
          f"{len(rows)} inputs enumerated; decide{sig} reads the boolean and "
          f"nothing else, so no listing can influence the verdict")


def c_extra_listing_is_not_the_predicate():
    """An empty or failed diagnostic listing must not rescue a dirty tree."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo(Path(tmp) / "r")
        (repo / "residue.txt").write_text("x\n")
        original = G.diagnostic_entries
        try:
            G.diagnostic_entries = lambda root: ([], None)
            empty = G.render_refusal(str(repo), False, "deadbeef", None,
                                     "admission")
            G.diagnostic_entries = lambda root: (None, "simulated git failure")
            failed = G.render_refusal(str(repo), False, "deadbeef", None,
                                      "admission")
        finally:
            G.diagnostic_entries = original
        check("C-LISTING-NOT-PREDICATE",
              "REFUSED" in empty and "EMPTY" in empty
              and "REFUSED" in failed and "UNAVAILABLE" in failed,
              "listing degraded to empty and to failed; refusal stood on the "
              "producer's boolean both times")


# ──────────────────────────────────────────────────────────────────────────────
# C5 — launch predicate == D3.5 predicate, for every fixture
# ──────────────────────────────────────────────────────────────────────────────

def _d3_5_verdict(clean, commit):
    """What `finalize_run` does with this boolean. Reaches run_finalizer:1589.

    Returns 'REJECTED_FOR_CLEANLINESS' or 'PASSED_CLEANLINESS_CHECK'. Anything
    downstream of the cleanliness check (candidate validation, publication) is
    NOT this gate's business and is classified as having passed it.
    """
    with tempfile.TemporaryDirectory() as out:
        try:
            RF.finalize_run(
                [], output_root=Path(out), run_id="c5_probe",
                prng_base="java_lcg", skip_modes_executed=("constant",),
                seed_start=0, seed_count=1,
                repository_commit=commit, repository_tree_clean=clean)
        except RF.RunParameterError as e:
            if "repository_tree_clean is False" in str(e):
                return "REJECTED_FOR_CLEANLINESS"
            return "PASSED_CLEANLINESS_CHECK"
        except Exception:                                           # noqa: BLE001
            return "PASSED_CLEANLINESS_CHECK"
    return "PASSED_CLEANLINESS_CHECK"


def c5_finalizer_agreement():
    try:
        a3_paths = attempt3_paths()
    except FrozenEvidenceUnavailable as e:
        check_unavailable("C5-FINALIZER-AGREEMENT",
                          f"attempt-3 fixture not derivable: {e}")
        check_unavailable("C5-ONE-PRODUCER",
                          f"attempt-3 fixture not derivable: {e}")
        return
    fixtures = []
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        clean = make_repo(base / "clean")
        fixtures.append(("clean", clean))

        a3 = make_repo(base / "attempt3")
        place_residue(a3, a3_paths)
        fixtures.append(("attempt3", a3))

        prod = make_repo(base / "prod")
        (prod / "production_module.py").write_text("VALUE = 2\n")
        fixtures.append(("modified-production", prod))

        gov = make_repo(base / "gov")
        (gov / "tests" / "test_governance.py").write_text("GATE = 'red'\n")
        fixtures.append(("modified-governance", gov))

        cfg = make_repo(base / "ignored-config")
        (cfg / "optimal_window_config.json").write_text("{}\n")
        fixtures.append(("ignored-config-present", cfg))

        rows, agree = [], True
        for label, repo in fixtures:
            launch_clean, commit, _err = G.evaluate(str(repo))
            producer_clean = WOI._repository_state(repo_root=str(repo))[1]
            rc = G.decide(launch_clean)
            verdict = _d3_5_verdict(bool(launch_clean), commit)
            admitted = (rc == G.EXIT_PROCEED)
            rejected = (verdict == "REJECTED_FOR_CLEANLINESS")
            row_ok = (launch_clean == producer_clean) and not (admitted
                                                              and rejected)
            agree &= row_ok
            rows.append(f"{label}={'ADMIT' if admitted else 'REFUSE'}/"
                        f"{'REJECT' if rejected else 'ACCEPT'}")

        check("C5-FINALIZER-AGREEMENT", agree, "; ".join(rows))
        check("C5-ONE-PRODUCER",
              G._repository_state is WOI._repository_state,
              "the gate holds the finalizer's producer object itself — "
              "equality of two implementations is not asserted because there "
              "is only one")


# ──────────────────────────────────────────────────────────────────────────────
# C5A — admission-to-dispatch stability
# ──────────────────────────────────────────────────────────────────────────────

def _run_rotation(repo, region, stamp="20260811_000000"):
    (repo / "logs").mkdir(exist_ok=True)          # mirrors gate12_launch.sh:48
    p = subprocess.run(["bash", "-c", f"set -u\nSTAMP={stamp}\n{region}"],
                       cwd=str(repo), capture_output=True, text=True)
    return p


def c5a_admission_to_dispatch_stability():
    region = extract_rotation_region(LAUNCH.read_text())
    if region is None:
        check("C5A-STABILITY", False, "rotation markers absent from the live "
                                      "script — extraction is vacuous")
        return
    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo(Path(tmp) / "r")
        (repo / "optimal_window_config.json").write_text('{"prior": true}\n')

        initial_clean, _c, _e = G.evaluate(str(repo))
        proc = _run_rotation(repo, region)
        post_clean, _c2, _e2 = G.evaluate(str(repo))
        rc_pre, _ = run_gate(repo, phase="pre-dispatch")

        moved = list((repo / "logs").glob("*pregate12*"))
        check("C5A-STABILITY",
              initial_clean is True and post_clean is True
              and rc_pre == 0 and proc.returncode == 0,
              f"initial=CLEAN, post-preparation=CLEAN, pre-dispatch gate rc=0; "
              f"coordinator dispatched=NO")
        check("C5A-ROLLBACK-PRESERVED",
              len(moved) == 1
              and not (repo / "optimal_window_config.json").exists(),
              f"clean-slate purpose kept and the rollback copy survives at "
              f"logs/{moved[0].name if moved else '<missing>'}")


def c5a_red_old_rotation_dirties():
    """The committed `pregate12` rename manufactures a Git-visible residue."""
    try:
        pinned = pre_repair_launch_source()
    except AnchorUnavailable as e:
        check_unavailable("C5A-RED-OLD-DIRTIES", str(e))
        return
    old = extract_old_rotation(pinned)
    if old is None:
        check("C5A-RED-OLD-DIRTIES", False,
              "no pregate12 rename found in the pinned pre-repair source — "
              "the RED arm would be vacuous")
        return
    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo(Path(tmp) / "r")
        (repo / "optimal_window_config.json").write_text('{"prior": true}\n')
        before, _c, _e = G.evaluate(str(repo))
        _run_rotation(repo, old)
        after, _c2, _e2 = G.evaluate(str(repo))
        rc_pre, out_pre = run_gate(repo, phase="pre-dispatch")
        check("C5A-RED-OLD-DIRTIES",
              before is True and after is False and rc_pre != 0
              and "pregate12" in out_pre,
              "pinned pre-repair rename executed verbatim: CLEAN -> DIRTY, and "
              "the pre-dispatch assertion catches it by name")


def c5a_pregate12_name_is_not_ignored():
    """The mechanism, measured against the LIVE ignore rules — not asserted."""
    probe = "optimal_window_config.json.pregate12_20260811_000000"
    src = subprocess.run(["git", "-C", str(REPO), "check-ignore", "-v",
                          "optimal_window_config.json"],
                         capture_output=True, text=True)
    old_dst = subprocess.run(["git", "-C", str(REPO), "check-ignore", "-v",
                              probe], capture_output=True, text=True)
    new_dst = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", "-v",
         "logs/gate12_20260811_000000_pregate12_optimal_window_config.json"],
        capture_output=True, text=True)
    check("C5A-IGNORE-MECHANISM",
          src.returncode == 0 and old_dst.returncode != 0
          and new_dst.returncode == 0,
          f"source IGNORED ({src.stdout.split(':')[1] if src.stdout else '?'}"
          f"), old destination NOT IGNORED, new destination IGNORED "
          f"({new_dst.stdout.split(':')[1] if new_dst.stdout else '?'})")


# ──────────────────────────────────────────────────────────────────────────────
# placement, wiring, and non-weakening
# ──────────────────────────────────────────────────────────────────────────────

def w_placement_precedes_everything():
    """The clean-tree wall's placement, and nothing beyond it.

    WHAT THIS ARM OWNS:

        admission < GPU gate < parity gate < clean slate < config rotation
                  < pre-dispatch clean-tree assertion

        pre-dispatch assertion < fleet dispatch
        pre-dispatch assertion < sampler creation
        pre-dispatch assertion < coordinator creation
        admission refusal exit  < clean slate

    That is the one predicate evaluated four times — admission, then preparation
    must preserve it, then the last pre-dispatch assertion, then compute, then
    D3.5 — plus the fact that nothing which commits or mutates happens before the
    pre-dispatch assertion clears.

    WHAT IT DELIBERATELY DOES NOT OWN:

        the relative order of fleet dispatch, sampler creation and coordinator
        creation. Those three are asserted as a FAN out of the pre-dispatch
        assertion and are never compared to one another. The two-phase attempt-6
        architecture — fleet first and parked, then sentinel verification, then
        the coordinator, then release — is owned by its own dedicated gates.

    WHY (Beta, 2026-08-14). The previous shape was a single chain ending
    `… < sampler < coordinator < fleet`, which asserted `coordinator < fleet`:
    the PRE-attempt-6 order, encoded incidentally by writing a fan as a chain. It
    went red when §8.4.3 inverted that order deliberately. Beta's diagnosis is
    that the mistake was allowing an older test to claim more ordering territory
    than the property it actually governed — a false constraint the moment the
    architecture moves, and green right up until then.

    RED WHEN: any link of the owned chain inverts, any named step disappears from
    the launch script, one of the three fanned steps is moved ahead of the
    pre-dispatch assertion, or the admission refusal stops exiting before the
    clean slate.
    """
    lines = LAUNCH.read_text().splitlines()

    def first(pred, code_only=True):
        # Comments mention these same filenames — the header documents the
        # sampler and the GPU gate by name. Ordering is a property of the
        # EXECUTABLE lines, so comment lines are skipped.
        for i, ln in enumerate(lines):
            if code_only and ln.lstrip().startswith("#"):
                continue
            if pred(ln):
                return i
        return 10 ** 6

    MISSING = 10 ** 6            # `first`'s not-found sentinel

    adm = first(lambda l: "gate12_cleantree_gate.py --phase admission" in l)
    adm_exit = first(lambda l: "CLEAN-TREE ADMISSION GATE (rc=" in l)
    pre = first(lambda l: "gate12_cleantree_gate.py --phase pre-dispatch" in l)
    gpu = first(lambda l: "gate12_gpu_gate.py" in l)
    parity = first(lambda l: "gate12_parity_gate.py" in l)
    slate = first(lambda l: l.startswith("pkill "))
    rot = first(lambda l: "GATE12-CONFIG-ROTATION BEGIN" in l, code_only=False)
    samp = first(lambda l: "gate12_concurrency_sampler.py" in l)
    watch = first(lambda l: "watcher_agent.py --clear-halt" in l)
    fleet = first(lambda l: "launch_fleet_manual.sh" in l)

    # THE OWNED CHAIN — the clean-tree wall and everything that must precede the
    # pre-dispatch assertion.
    chain = [("admission", adm), ("gpu", gpu), ("parity", parity),
             ("clean-slate", slate), ("rotation", rot), ("pre-dispatch", pre)]
    chain_ok = all(chain[i][1] < chain[i + 1][1] for i in range(len(chain) - 1))

    # THE FAN — each asserted against the pre-dispatch assertion ONLY. These
    # three are never compared to one another; their relative order belongs to
    # the two-phase architecture and its own gates.
    fan = {"fleet": fleet, "sampler": samp, "coordinator": watch}
    fan_ok = all(pre < v for v in fan.values())

    # A needle that no longer matches must be RED, not silently ordered by the
    # not-found sentinel.
    found_ok = all(v != MISSING for _n, v in chain) and \
        all(v != MISSING for v in fan.values()) and adm_exit != MISSING

    check("W-ADMISSION-FIRST",
          found_ok and chain_ok and fan_ok and adm_exit < slate,
          " < ".join(f"{n}({v})" for n, v in chain)
          + "; pre-dispatch precedes "
          + " · ".join(f"{n}({v})" for n, v in sorted(fan.items(),
                                                      key=lambda kv: kv[1]))
          + f"; admission-refusal-exit({adm_exit}) < clean-slate({slate})"
          + ("" if found_ok else "  [A NEEDLE DID NOT MATCH]"))


def w_exit_codes_are_honoured():
    """PIPESTATUS, not $?. `cmd | tee` exits with tee's status, always 0."""
    text = LAUNCH.read_text()
    both = len(re.findall(
        r"gate12_cleantree_gate\.py --phase \w+[^\n]*\| tee -a \"\$EVID\"\n"
        r"CLEANTREE\w*_RC=\$\{PIPESTATUS\[0\]\}", text)) == 2
    demo = subprocess.run(
        ["bash", "-c",
         "false | tee /dev/null; A=$?; false | tee /dev/null; "
         "B=${PIPESTATUS[0]}; echo $A $B"],
        capture_output=True, text=True)
    check("W-PIPESTATUS",
          both and demo.stdout.strip() == "0 1",
          "both invocations read ${PIPESTATUS[0]}; live shell confirms $? "
          "would have been 0 for a failing gate")


def w_no_weakening_of_d3_5():
    """Nothing certified moved: not the finalizer, not the producer, not
    `.gitignore`. The repair is additive and lives in the harness."""
    def head_sha(rel):
        return hashlib.sha256(
            _git(REPO, "show", f"HEAD:{rel}").stdout.encode()).hexdigest()

    def work_sha(rel):
        return hashlib.sha256((REPO / rel).read_text().encode()).hexdigest()

    frozen = ["utils/run_finalizer.py",
              "window_optimizer_integration_final.py",
              ".gitignore"]
    same = {rel: head_sha(rel) == work_sha(rel) for rel in frozen}
    residue_exception = any(
        pat in (REPO / ".gitignore").read_text()
        for pat in ("*.stale_*", "*.db-shm", "*.db-wal", "*.pregate12*"))
    check("W-NO-WEAKENING",
          all(same.values()) and not residue_exception,
          "run_finalizer, the producer and .gitignore are byte-identical to "
          "HEAD; no runtime-residue filename exception was added")


def w_gate_reuses_the_producer():
    """No second porcelain implementation feeds the decision.

    Structural, via AST: `diagnostic_entries` — the only other place git is
    run — is called from `render_refusal` alone, which by definition runs after
    the verdict.
    """
    tree = ast.parse(GATE.read_text())
    callers = set()
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == "diagnostic_entries":
                callers.add(fn.name)
    check("W-ONE-DECISION-PATH",
          callers == {"render_refusal"}
          and G._repository_state is WOI._repository_state,
          f"diagnostic_entries called only from {sorted(callers)}; the "
          f"verdict comes from the imported producer object")


# ──────────────────────────────────────────────────────────────────────────────
# mutation — proof the credited assertions are load-bearing
# ──────────────────────────────────────────────────────────────────────────────

def m1_mutant_gate_always_proceeds():
    """Mutate `decide` to admit unconditionally; C1's refusal must vanish."""
    mutated = GATE.read_text().replace(
        "    return EXIT_PROCEED if clean is True else EXIT_REFUSE",
        "    return EXIT_PROCEED  # MUTANT")
    applied = "MUTANT" in mutated and mutated != GATE.read_text()
    try:
        paths = attempt3_paths()
    except FrozenEvidenceUnavailable as e:
        check_unavailable("M1-MUTANT-KILLED", f"fixture not derivable: {e}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo(Path(tmp) / "r")
        place_residue(repo, paths)
        mpath = Path(tmp) / "mutant_gate.py"
        mpath.write_text(mutated)
        env = dict(os.environ, PYTHONPATH=str(REPO))
        p = subprocess.run(
            [sys.executable, "-u", str(mpath), "--repo-root", str(repo)],
            capture_output=True, text=True, env=env)
        clean_rc, _ = run_gate(repo)
        check("M1-MUTANT-KILLED",
              applied and p.returncode == 0 and clean_rc != 0,
              "mutant applied and EXECUTED (rc=0 on the attempt-3 state); "
              "unmutated gate refuses the identical fixture (rc=1)")


def m2_mutant_pipestatus_dropped():
    """`if ! cmd | tee` tests tee's status — the decorative-gate defect."""
    demo = subprocess.run(
        ["bash", "-c",
         "if ! false | tee /dev/null; then echo REFUSED; else echo "
         "LAUNCHED_ANYWAY; fi"],
        capture_output=True, text=True)
    check("M2-DECORATIVE-FORM-KILLED",
          demo.stdout.strip() == "LAUNCHED_ANYWAY"
          and "if ! python3 -u scripts/gate12_cleantree_gate.py"
              not in LAUNCH.read_text(),
          "the rejected form demonstrably launches on a failing gate; the "
          "live script does not use it")


def m3_mutant_rotation_target_moved_back():
    """Put the destination back in the repo root: C5A must red."""
    region = extract_rotation_region(LAUNCH.read_text())
    mutated = region.replace(
        '"logs/gate12_${STAMP}_pregate12_optimal_window_config.json"',
        'optimal_window_config.json.pregate12_${STAMP}')
    applied = mutated != region
    with tempfile.TemporaryDirectory() as tmp:
        repo = make_repo(Path(tmp) / "r")
        (repo / "optimal_window_config.json").write_text("{}\n")
        before, _c, _e = G.evaluate(str(repo))
        _run_rotation(repo, mutated)
        after, _c2, _e2 = G.evaluate(str(repo))
        check("M3-ROTATION-MUTANT-KILLED",
              applied and before is True and after is False,
              "destination moved back to the repo root; the same fixture that "
              "stays CLEAN under the live region goes DIRTY")


def m4_extraction_is_not_vacuous():
    """A region that does not contain the rename would make C5A prove nothing."""
    region = extract_rotation_region(LAUNCH.read_text()) or ""
    check("M4-EXTRACTION-NON-VACUOUS",
          "optimal_window_config.json" in region and "mv " in region
          and "logs/" in region
          and region in LAUNCH.read_text(),
          f"{len(region.strip().splitlines())} lines extracted verbatim from "
          f"the live script and they are the rename")


def m5_mutant_anchor_object_missing():
    """[R1] An anchor whose object does not resolve must RAISE, never credit.

    The wrong commit is DERIVED — the first parent of the commit that added
    `gate12_launch.sh`, where the file provably does not exist — rather than
    transcribed, so this mutant cannot go stale against a hardcoded hash.
    """
    adder = _git(REPO, "log", "--diff-filter=A", "--format=%H", "--",
                 "gate12_launch.sh").stdout.split()
    if not adder:
        check_unavailable("M5-ANCHOR-MISSING-KILLED",
                          "cannot derive a commit predating gate12_launch.sh")
        return
    before = f"{adder[-1]}^"
    raised, reason = False, ""
    try:
        _launch_source_at(before)
    except AnchorUnavailable as e:
        raised, reason = True, str(e)
    # And the honest control: the real anchor does resolve in the same repo.
    anchor_ok = True
    try:
        pre_repair_launch_source()
    except AnchorUnavailable:
        anchor_ok = False
    check("M5-ANCHOR-MISSING-KILLED",
          raised and "does not resolve" in reason and anchor_ok,
          f"anchor moved to {before[:12]}… (derived, predates the file) -> "
          f"AnchorUnavailable; the pinned anchor still resolves")


def m6_mutant_anchor_drifted_to_repaired():
    """[R1] An anchor pointing at the REPAIRED script must RAISE, never credit.

    This is the failure mode the whole R1 correction exists for: after Michael
    commits, `HEAD:gate12_launch.sh` IS the repaired script. Both defect
    surfaces must be reported missing so no RED arm can be credited from it.
    The probes run over executable lines only — the repaired script quotes both
    surfaces verbatim in its own header comments, so a raw-text probe would
    match and this mutant would survive.
    """
    repaired = LAUNCH.read_text()
    missing = _missing_surfaces(repaired)
    raised = False
    with tempfile.TemporaryDirectory() as tmp:
        # A real one-commit repo whose HEAD:gate12_launch.sh is the REPAIRED
        # script — the post-repair world, built without committing anything in
        # ~/distributed_prng_analysis.
        drifted = make_repo(Path(tmp) / "drift",
                            tracked={"gate12_launch.sh": repaired})
        try:
            _launch_source_at("HEAD", repo=drifted)
        except AnchorUnavailable as e:
            raised = "is NOT the pre-repair script" in str(e)
        # Control: the same helper accepts the genuine pre-repair source.
        pinned = make_repo(Path(tmp) / "pinned", tracked={})
        try:
            src = pre_repair_launch_source()
            (pinned / "gate12_launch.sh").write_text(src)
            _git(pinned, "add", "gate12_launch.sh")
            _git(pinned, "commit", "-q", "-m", "pre-repair")
            control = _launch_source_at("HEAD", repo=pinned) == src
        except AnchorUnavailable:
            control = False
    check("M6-ANCHOR-DRIFT-KILLED",
          len(missing) == len(_PRE_REPAIR_SURFACES) and raised and control,
          f"repaired script reports {len(missing)}/{len(_PRE_REPAIR_SURFACES)} "
          f"defect surfaces missing and is REFUSED as an anchor; the genuine "
          f"pre-repair source is accepted by the same helper")


def m7_mutant_fixture_evidence_tampered():
    """[R2] Tampered or wrong-shaped frozen evidence must NOT build a fixture.

    Three independent tampers, each against a COPY in a temp directory. The
    frozen bundle itself is never written to.
    """
    try:
        real = ATTEMPT3_PORCELAIN.read_text()
    except OSError as e:
        check_unavailable("M7-EVIDENCE-TAMPER-KILLED", str(e))
        return
    outcomes = {}
    with tempfile.TemporaryDirectory() as tmp:
        cases = {
            "absent": None,
            "truncated": "\n".join(real.splitlines()[:2]) + "\n",
            "digest-mismatch": real + "?? injected_extra_entry\n",
            "not-untracked": real.replace("??", " M", 1),
        }
        for label, body in cases.items():
            p = Path(tmp) / f"{label}.txt"
            if body is not None:
                p.write_text(body)
            try:
                # `sums_required` stays ON for digest-mismatch (that IS the
                # test); the shape cases run with it OFF so they exercise the
                # parser rather than tripping the digest first.
                attempt3_entries(porcelain_path=p,
                                 sums_required=(label == "digest-mismatch"))
                outcomes[label] = "ACCEPTED"
            except FrozenEvidenceUnavailable:
                outcomes[label] = "REFUSED"
    good = all(v == "REFUSED" for v in outcomes.values())
    still_real = False
    try:
        still_real = len(attempt3_paths()) == ATTEMPT3_EXPECTED_ENTRY_COUNT
    except FrozenEvidenceUnavailable:
        pass
    check("M7-EVIDENCE-TAMPER-KILLED",
          good and still_real,
          f"{outcomes}; the untampered frozen evidence still yields "
          f"{ATTEMPT3_EXPECTED_ENTRY_COUNT} entries")


def m8_bundle_still_verifies():
    """The frozen bundle is read-only, and this suite proves it left it alone."""
    if not ATTEMPT3_SUMS.is_file():
        check_unavailable("M8-BUNDLE-INTACT", f"{ATTEMPT3_SUMS} absent")
        return
    p = subprocess.run(["sha256sum", "-c", ATTEMPT3_SUMS.name],
                       cwd=str(ATTEMPT3_BUNDLE), capture_output=True, text=True)
    ok_lines = [ln for ln in p.stdout.splitlines() if ln.endswith(": OK")]
    total = len([ln for ln in ATTEMPT3_SUMS.read_text().splitlines()
                 if ln.strip()])
    check("M8-BUNDLE-INTACT",
          p.returncode == 0 and len(ok_lines) == total,
          f"sha256sum -c -> {len(ok_lines)}/{total} OK after this suite ran")


def main():
    print("=" * 74, flush=True)
    print("GATE-12 CLEAN-TREE ADMISSION REPAIR — C1-C5 + C5A", flush=True)
    print("=" * 74, flush=True)
    print(f"launch script : {LAUNCH}", flush=True)
    print(f"gate          : {GATE}", flush=True)
    print(f"attempt-3 bundle (read-only) : {ATTEMPT3_BUNDLE}", flush=True)
    print(f"pinned pre-repair anchor     : {PRE_REPAIR_COMMIT}", flush=True)
    print(f"current HEAD                 : "
          f"{_git(REPO, 'rev-parse', 'HEAD').stdout.strip()}", flush=True)

    print("\n-- R1/R2: provenance of the adversarial inputs --", flush=True)
    r1_anchor_integrity()
    r2_fixture_is_derived_from_frozen_evidence()
    print("\n-- C1: exact attempt-3 reproduction --", flush=True)
    c1_attempt3_state_is_refused()
    print("\n-- C1-RED: the pinned pre-repair script on the same state --",
          flush=True)
    c1_red_old_preflight_admits()
    print("\n-- C2/C3/C4: the rest of the input space --", flush=True)
    c2_clean_repo_passes()
    c3_modified_production_file_refused()
    c4_modified_governance_file_refused()
    c_extra_unobservable_refuses()
    c_extra_decide_input_space()
    c_extra_listing_is_not_the_predicate()
    print("\n-- C5: launch predicate == D3.5 predicate --", flush=True)
    c5_finalizer_agreement()
    print("\n-- C5A: admission-to-dispatch stability --", flush=True)
    c5a_admission_to_dispatch_stability()
    c5a_red_old_rotation_dirties()
    c5a_pregate12_name_is_not_ignored()
    print("\n-- placement, wiring, non-weakening --", flush=True)
    w_placement_precedes_everything()
    w_exit_codes_are_honoured()
    w_no_weakening_of_d3_5()
    w_gate_reuses_the_producer()
    print("\n-- mutation --", flush=True)
    m1_mutant_gate_always_proceeds()
    m2_mutant_pipestatus_dropped()
    m3_mutant_rotation_target_moved_back()
    m4_extraction_is_not_vacuous()
    print("\n-- mutation: provenance anchors (R1/R2) --", flush=True)
    m5_mutant_anchor_object_missing()
    m6_mutant_anchor_drifted_to_repaired()
    m7_mutant_fixture_evidence_tampered()
    m8_bundle_still_verifies()

    passed = sum(1 for _, ok in _RESULTS if ok)
    total = len(_RESULTS)
    print("=" * 74, flush=True)
    print(f"{passed}/{total} checks green", flush=True)
    if passed != total:
        print("FAILURES: " + ", ".join(n for n, ok in _RESULTS if not ok),
              flush=True)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
