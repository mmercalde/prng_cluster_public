#!/usr/bin/env python3
"""GATE-12 RIG CODE-PARITY GATE — a harness rule, and the wall the D6 dry run
proved was missing.

WHAT THIS IS
------------
The pre-launch battery proves the *cards* are present (`gate12_gpu_gate.py`) and
that the worker's *log channel* delivers (`gate12_sentinel_gate.py`). **Nothing
proved the code.** On 2026-08-14 the D6 parked-fleet dry run dispatched 25
workers and 24 died at argparse, because all three rigs were carrying a
`miner/range_miner_worker.py` last deployed 2026-08-02 — predating Defect A
(`acd6f13`), its §14 deadline revision (`2532803`) and the attempt-6 sentinel
work (`69ff222`). The fleet had been at that vintage through attempts 3, 4 and 5.

This gate closes that hole the same way the GPU gate closed `0/8`: it measures
the fact, it fails closed, and it refuses before anything is dispatched.

WHY CONTENT IDENTITY AND NEVER GIT IDENTITY
-------------------------------------------
`git rev-parse` on a rig is not evidence and must never be accepted as evidence
(§2.17). The rigs are DEPLOYMENT TARGETS, not working copies: deployment is a
`git clone` once and targeted `scp` thereafter, `rrig6600` carries a worktree at
`8e2f5bf` with 84 dirty entries, and the other two have no git repository at all.
The D6 measurement then proved the stronger point — the deployed tree is MIXED
VINTAGE, two files ten days apart from each other — so no single commit
identifier can describe it even in principle. **The only acceptable acceptance
evidence is a content digest of the deployed bytes.** The local commit is
recorded in the evidence block as coordinator CONTEXT and is never an input to
the verdict.

WHY THE EXPECTED DIGESTS ARE DERIVED, NEVER TRANSCRIBED
-------------------------------------------------------
Expected values are the **full 64-hex SHA256** of the local canonical tree, read
at run time. The 12-character forms that appear in the forensic report
(`992464ba611e…`, `7b7f8e197914…`) are DISPLAY PREFIXES, not identities, and are
deliberately not present in this file: a gate that compares a truncated
transcription is a gate whose expectation drifts the moment the tree moves, and
a 12-hex prefix is a weaker equality than the one the operator believes they are
reading. `--verify-clean` (default on) additionally refuses if any governed file
is dirty in the local worktree, so "the canonical clean local tree" is a checked
property of the run rather than an assumption about it.

THE GOVERNED FILE SET — PINNED, AND CROSS-CHECKED AGAINST A DERIVATION
----------------------------------------------------------------------
`GOVERNED_FILES` is an explicit pin. `derive_worker_import_closure()` independently
re-derives the worker's STATICALLY REACHABLE PROJECT-LOCAL IMPORT / DEPLOYMENT
CLOSURE from the AST of the live sources, and the gate REFUSES if the derivation
is not covered by the pin. So a future worker-side project import cannot enter
that closure silently: the day someone adds one, this gate goes red and names the
uncovered file until the pin is updated deliberately.

Only paths that resolve INSIDE the repository are ever considered. Python stdlib
and site-packages are never hashed — they are not what this fleet deploys by
`scp`, and hashing them would make every venv difference a fleet defect.

⚠ STATICALLY REACHABLE, NOT "EXECUTED" — the distinction is deliberate and Beta
ruled on it (2026-08-14). The governed set is the CONSERVATIVE SUPERSET: every
project file the worker's imports can reach, whether or not a given run's
arguments, branches, failure handling or deferred imports actually execute it.
`execution_set.py` is the live example — statically reachable, and NOT executed on
today's normal worker path — and it is governed and must be deployed anyway.
A call-graph exception was REJECTED: "does this execution happen to reach this
file?" depends on arguments, branch paths, future code, failure handling and
deferred imports, and is far harder to prove than static reachability.

⚠ The closure is larger than the five files the D6 forensic measured, and that
is a finding rather than a formatting choice: `miner/__init__.py:19` imports the
coordinator at MODULE scope, so `from miner.range_miner_protocol import …` in the
worker pulls `miner/range_miner_coordinator.py` (and, at its own module scope,
`miner/dataset_authority.py`) into that closure. The forensic's five-file
measurement did not reach them.

THE RULE — fail-close on anything that is not a proven byte-for-byte match:

    OK and every governed file MATCHES on ALL rigs      -> PROCEED
    any governed file MISMATCHES on any rig             -> REFUSE
    any governed file MISSING on any rig                -> REFUSE
    a rig's reported hostname is not the expected one   -> REFUSE
    UNAVAILABLE (ssh transport failure, timeout,
                 truncated output, no such directory)   -> REFUSE
    ERROR       (the probe ran, output unclassifiable)  -> REFUSE
    the derived closure is not covered by the pin       -> REFUSE
    a governed file is dirty in the local worktree      -> REFUSE

There is no advisory treatment, no partial fleet and no automatic downsizing. A
refusal is a refusal.

WHAT WRONG INPUT MAKES THIS GATE RED — stated, because a gate whose red
condition is not written down is a gate nobody can falsify:

    a rig carrying ANY governed file at a different digest (the D6 defect
    itself); a rig missing a governed file; a rig answering under another
    machine's hostname (three rigs cannot be one machine answering thrice); an
    ssh that cannot connect or authenticate; a probe whose output is truncated
    (the END sentinel is absent) or unparseable; a worker-side project import
    added without extending the pin; a local worktree whose governed files are
    modified, so the expectation would not be canonical.

MISSING is a MISMATCH, not an UNAVAILABLE. The probe ran and observed the file's
absence — that is a measurement, and collapsing it into "could not measure" would
destroy the distinction the three-outcome vocabulary exists to preserve.

Exit status: 0 = proceed, 1 = REFUSE. The caller aborts on non-zero BEFORE any
worker is dispatched.

Nothing is hardcoded: the fleet, its endpoints and its ssh user come from the
committed execution set (`rig_profiles_config.json` joined with
`distributed_config.json`) — the same authority the run itself resolves its fleet
from — and the remote repository path comes from `distributed_config.json`, which
is where `scripts/launch_fleet_manual.sh` reads it from too.
"""

import argparse
import ast
import datetime
import hashlib
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# The three-outcome vocabulary, imported from the certified GPU probe rather than
# re-declared: one vocabulary, one meaning, and a rename cannot leave two copies
# disagreeing about what UNAVAILABLE is.
import preflight_check as PF                                        # noqa: E402

PROBE_OK = PF.GPU_PROBE_OK
PROBE_UNAVAILABLE = PF.GPU_PROBE_UNAVAILABLE
PROBE_ERROR = PF.GPU_PROBE_ERROR

SSH_CONNECT_TIMEOUT = PF.SSH_TIMEOUT_SECONDS
PROBE_TIMEOUT = PF.GPU_CHECK_TIMEOUT_SECONDS

# ssh returns this for ITS OWN failure — connect timeout, no route, host-key
# mismatch, BatchMode auth refusal — and this gate reserves the value as its
# remote-transport classification under the probe script below, exactly as
# `gate12_sentinel_gate.py` does. It is safe here for the same reason: the script
# this gate sends cannot produce 255 (its own failure branches exit 3 and 4). If
# that script changes, this reservation must be revisited, which is why the two
# live in the same file.
SSH_TRANSPORT_FAILURE_STATUS = 255

# The probe's own exit codes, so "no such directory" is distinguishable from a
# transport failure and from a shell that could not start.
PROBE_EXIT_NO_DIR = 3

EXIT_PROCEED = 0
EXIT_REFUSE = 1

BEGIN_SENTINEL = "TFM-PARITY-BEGIN"
END_SENTINEL = "TFM-PARITY-END"

# The root of the statically reachable project-local import / deployment
# closure: the file the launcher actually runs.
WORKER_ENTRYPOINT = "miner/range_miner_worker.py"

# ---------------------------------------------------------------------------
# THE PIN
# ---------------------------------------------------------------------------
# The governed set. Every entry is repo-relative POSIX. This is the SET, not the
# EXPECTED VALUES — the digests are derived at run time from the local tree, so
# this pin never goes stale against a legitimate code change; it goes stale only
# against a change in WHICH FILES the worker executes, which is exactly the event
# it exists to make loud.
#
# It is a superset of the five files the D6 forensic measured
# (prng_registry.py, sieve_gpu_worker.py, miner/__init__.py,
# miner/range_miner_worker.py, miner/range_miner_protocol.py) — Beta's declared
# MINIMUM — and every additional entry is here because the derivation below
# reaches it, not because it seemed prudent.
GOVERNED_FILES = (
    "adaptive_thresholds.py",
    "execution_set.py",
    "hybrid_strategy.py",
    "miner/__init__.py",
    "miner/dataset_authority.py",
    "miner/range_miner_coordinator.py",
    "miner/range_miner_protocol.py",
    "miner/range_miner_worker.py",
    "prng_registry.py",
    "sieve_gpu_worker.py",
)

# Beta's declared minimum, asserted against the pin at run time so a future
# narrowing of GOVERNED_FILES cannot drop one of them unnoticed.
BETA_MINIMUM_FILES = (
    "prng_registry.py",
    "sieve_gpu_worker.py",
    "miner/__init__.py",
    "miner/range_miner_worker.py",
    "miner/range_miner_protocol.py",
)


# ---------------------------------------------------------------------------
# derivation — the independent cross-check on the pin
# ---------------------------------------------------------------------------

def _module_to_repo_path(module, repo_root):
    """Resolve a dotted module name to a repo-relative .py path, or None.

    None means "not a project file" — stdlib, site-packages, a C extension, or a
    name that does not exist. Those are never hashed and never governed: this
    gate is about what the fleet deploys by scp, not about what the venv
    provides.
    """
    if not module:
        return None
    parts = module.split(".")
    cand = os.path.join(repo_root, *parts) + ".py"
    if os.path.isfile(cand):
        return os.path.relpath(cand, repo_root).replace(os.sep, "/")
    cand = os.path.join(repo_root, *parts, "__init__.py")
    if os.path.isfile(cand):
        return os.path.relpath(cand, repo_root).replace(os.sep, "/")
    return None


def _imported_names(relpath, repo_root):
    """Every module name this file imports, at ANY scope.

    Deliberately `ast.walk`, not module-scope only. The worker's GPU imports are
    function-local by design (`prng_registry` at :448, `hybrid_strategy` at :806,
    `sieve_gpu_worker` at :838, the coordinator constant at :1265) and they
    execute during mining; a module-scope-only derivation would omit the two
    files the D6 forensic had to measure by hand. `from X import Y` contributes
    both `X` and `X.Y`, because `Y` may itself be a submodule.
    """
    with open(os.path.join(repo_root, relpath), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    package = os.path.dirname(relpath).replace("/", ".")
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package
                for _ in range(node.level - 1):
                    base = base.rpartition(".")[0]
                if node.module:
                    module = f"{base}.{node.module}" if base else node.module
                else:
                    # `from .. import e` — the names ARE the modules, relative to
                    # `base`. When base walks up to the repo root, base is "" and
                    # each name is a top-level module; dropping them here is what
                    # silently loses a whole subtree from the closure.
                    module = base
            else:
                module = node.module or ""
            if module:
                names.add(module)
            for alias in node.names:
                names.add(f"{module}.{alias.name}" if module else alias.name)
    return names


def derive_worker_import_closure(repo_root=REPO_ROOT, root=WORKER_ENTRYPOINT):
    """The worker's transitive project-local import closure, from live source.

    This is the cross-check on `GOVERNED_FILES`, and it is deliberately derived
    from the AST of the files rather than from `sys.modules` after an import:
    a runtime probe sees only what the deferred imports happened to execute, and
    the mining-path imports do not execute at start-up. An AST closure is a
    conservative superset, which is the correct direction for a fail-closed gate.
    """
    seen = set()
    stack = [root]
    while stack:
        rel = stack.pop()
        if rel in seen:
            continue
        seen.add(rel)
        for module in _imported_names(rel, repo_root):
            resolved = _module_to_repo_path(module, repo_root)
            if resolved and resolved not in seen:
                stack.append(resolved)
    return seen


def closure_coverage(repo_root=REPO_ROOT, governed=GOVERNED_FILES,
                     root=WORKER_ENTRYPOINT):
    """(covered, uncovered, missing_minimum) — the pin's own integrity check."""
    derived = derive_worker_import_closure(repo_root, root)
    uncovered = sorted(derived - set(governed))
    missing_minimum = [f for f in BETA_MINIMUM_FILES if f not in governed]
    return (not uncovered and not missing_minimum), uncovered, missing_minimum


# ---------------------------------------------------------------------------
# expected values — derived from the canonical clean local tree, at run time
# ---------------------------------------------------------------------------

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_digests(repo_root=REPO_ROOT, governed=GOVERNED_FILES):
    """{relpath: {"sha256": <full 64 hex>, "size": int}} from the local tree.

    A governed file absent LOCALLY is a hard error rather than a refusal row:
    the gate has no expectation to compare against, and continuing would compare
    the fleet to nothing.
    """
    out = {}
    for rel in governed:
        p = os.path.join(repo_root, rel)
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"governed file {rel!r} does not exist in the local tree at "
                f"{repo_root} — the gate has no expectation to compare against")
        out[rel] = {"sha256": sha256_of(p), "size": os.path.getsize(p)}
    return out


def local_worktree_dirty(repo_root=REPO_ROOT, governed=GOVERNED_FILES,
                         runner=None):
    """Governed files that are not clean in the local worktree.

    Returns (entries, unavailable_reason). `entries` is the porcelain lines
    restricted to the governed paths. A git that cannot run is UNAVAILABLE and
    therefore a refusal — never a silent clean.
    """
    runner = runner or _run
    try:
        proc = runner(["git", "-C", repo_root, "status", "--porcelain", "--"]
                      + list(governed), PROBE_TIMEOUT)
    except Exception as exc:                                        # noqa: BLE001
        return [], f"{type(exc).__name__}:{exc}"
    if proc.returncode != 0:
        return [], (f"git exited {proc.returncode}: "
                    f"{proc.stderr.decode(errors='replace').strip()!r}")
    lines = [ln for ln in proc.stdout.decode(errors="replace").splitlines()
             if ln.strip()]
    return lines, None


def local_head(repo_root=REPO_ROOT, runner=None):
    """The local commit, recorded as CONTEXT ONLY.

    Beta, binding: acceptance authority is content identity, not Git identity.
    This value appears in the evidence block and is never read by `evaluate`.
    """
    runner = runner or _run
    try:
        proc = runner(["git", "-C", repo_root, "rev-parse", "HEAD"], PROBE_TIMEOUT)
    except Exception:                                               # noqa: BLE001
        return "UNAVAILABLE"
    if proc.returncode != 0:
        return "UNAVAILABLE"
    return proc.stdout.decode(errors="replace").strip() or "UNAVAILABLE"


# ---------------------------------------------------------------------------
# the fleet, and the probe
# ---------------------------------------------------------------------------

def gate_targets(rig_profile=None, repo_root=REPO_ROOT, config_path=None,
                 profile_map_path=None, provisioning_manifest_path=None):
    """The rigs to probe, with the remote repository path each deploys into.

    DERIVED, never transcribed. `resolve_execution_set` joins the rig profile map
    with `distributed_config.json`; `remote_nodes()` drops the local node, which
    is the reference tree rather than a deployment target. `script_path` comes
    from `distributed_config.json` keyed by the node's `config_hostname` — the
    same join `scripts/launch_fleet_manual.sh` performs to decide where the
    worker is launched from, so the gate hashes the directory the worker is
    actually run out of.
    """
    import execution_set as XS
    kwargs = dict(backend="miner", invoked_by="gate12_parity_gate",
                  rig_profile=rig_profile, repo_root=repo_root)
    if config_path:
        kwargs["config_path"] = config_path
    if profile_map_path:
        kwargs["profile_map_path"] = profile_map_path
    if provisioning_manifest_path:
        kwargs["provisioning_manifest_path"] = provisioning_manifest_path
    resolved = XS.resolve_execution_set(**kwargs)

    with open(config_path or os.path.join(repo_root, "distributed_config.json")) as fh:
        by_host = {n["hostname"]: n for n in json.load(fh)["nodes"]}

    targets = []
    for node in resolved.remote_nodes():
        cfg = by_host.get(node.config_hostname)
        if cfg is None:
            raise KeyError(
                f"no distributed_config.json node for {node.config_hostname!r} — "
                f"the gate cannot know where {node.node_id} deploys to")
        targets.append({
            "node_id": node.node_id,
            "endpoint": node.endpoint,
            "ssh_user": node.ssh_user,
            "worker_hostname": node.worker_hostname,
            "script_path": cfg["script_path"],
        })
    return targets


def build_probe_script(script_path, governed=GOVERNED_FILES):
    """ONE ssh per rig hashes every governed path.

    The BEGIN/END sentinels are the truncation defence (VIR-1): a probe whose
    output was cut short — a dropped connection, a killed shell — is missing its
    END line and is classified UNAVAILABLE. Without them a partial listing would
    be indistinguishable from a complete one that happened to be short, and the
    gate would pass on the files it managed to read.

    `cd` failing exits 3, so "the deployment directory is not there" is a named
    outcome rather than an empty stdout. Absence of a FILE is printed as MISSING
    with no size, never as an empty digest field.
    """
    files = " ".join(f"'{f}'" for f in governed)
    return (
        f"cd '{script_path}' || exit {PROBE_EXIT_NO_DIR}\n"
        f"echo {BEGIN_SENTINEL}\n"
        "printf 'HOST\\t%s\\n' \"$(hostname)\"\n"
        f"for f in {files}; do\n"
        "  if [ -f \"$f\" ]; then\n"
        "    printf 'FILE\\t%s\\t%s\\t%s\\n' \"$f\" "
        "\"$(sha256sum \"$f\" | cut -d' ' -f1)\" \"$(wc -c < \"$f\" | tr -d ' ')\"\n"
        "  else\n"
        "    printf 'FILE\\t%s\\tMISSING\\t-\\n' \"$f\"\n"
        "  fi\n"
        "done\n"
        f"echo {END_SENTINEL}\n"
    )


def _run(cmd, timeout):
    return subprocess.run(cmd, capture_output=True, timeout=timeout)


def parse_probe_output(text, governed=GOVERNED_FILES):
    """(status, hostname, {relpath: (digest_or_MISSING, size_or_None)}, reason).

    Refuses to guess. Missing sentinels, a missing HOST line, a row with the
    wrong field count, a digest that is not 64 lowercase hex, or a governed path
    with no row at all are all classified rather than repaired.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if BEGIN_SENTINEL not in lines:
        return PROBE_ERROR, None, {}, "no_begin_sentinel"
    if END_SENTINEL not in lines:
        # Truncation: the probe started and did not finish. Not a measurement.
        return PROBE_UNAVAILABLE, None, {}, "truncated_probe_output"
    body = lines[lines.index(BEGIN_SENTINEL) + 1:lines.index(END_SENTINEL)]

    hostname = None
    observed = {}
    for line in body:
        parts = line.split("\t")
        if parts[0] == "HOST":
            if len(parts) != 2 or not parts[1].strip():
                return PROBE_ERROR, None, {}, "malformed_host_row"
            hostname = parts[1].strip()
        elif parts[0] == "FILE":
            if len(parts) != 4:
                return PROBE_ERROR, hostname, {}, f"malformed_file_row:{line!r}"
            _, rel, digest, size = parts
            if digest == "MISSING":
                observed[rel] = ("MISSING", None)
                continue
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                return (PROBE_ERROR, hostname, {},
                        f"malformed_digest_for:{rel}")
            try:
                observed[rel] = (digest, int(size))
            except ValueError:
                return PROBE_ERROR, hostname, {}, f"malformed_size_for:{rel}"
        else:
            return PROBE_ERROR, hostname, {}, f"unknown_row:{line!r}"

    if hostname is None:
        return PROBE_ERROR, None, {}, "no_host_row"
    absent = [f for f in governed if f not in observed]
    if absent:
        return (PROBE_ERROR, hostname, observed,
                "no_row_for:" + ",".join(absent))
    return PROBE_OK, hostname, observed, None


def probe_rig(target, governed=GOVERNED_FILES, runner=None):
    """One rig, one ssh, exactly three possible outcomes.

    Mirrors `gate12_gpu_gate.probe_rig`: the script is passed as ONE argv
    element, because ssh joins trailing arguments with spaces without re-quoting
    them, and a re-parsed pipeline is a quoting accident waiting to happen.
    """
    runner = runner or _run
    script = build_probe_script(target["script_path"], governed)
    result = {
        "node_id": target["node_id"],
        "endpoint": target["endpoint"],
        "expected_hostname": target["worker_hostname"],
        "script_path": target["script_path"],
        "status": None,
        "hostname": None,
        "observed": {},
        "reason": None,
        "stderr": "",
        "collected_at": None,
    }
    try:
        proc = runner(
            ["ssh",
             "-n",
             "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT}",
             "-o", "BatchMode=yes",
             f"{target['ssh_user']}@{target['endpoint']}",
             script],
            PROBE_TIMEOUT)
    except subprocess.TimeoutExpired:
        # A hung probe observed nothing. It is emphatically not a mismatch and
        # emphatically not a match.
        result.update(status=PROBE_UNAVAILABLE, reason="timeout",
                      collected_at=_utcnow())
        return result
    except Exception as exc:                                        # noqa: BLE001
        result.update(status=PROBE_UNAVAILABLE,
                      reason=f"{type(exc).__name__}:{exc}",
                      collected_at=_utcnow())
        return result

    result["collected_at"] = _utcnow()
    result["stderr"] = proc.stderr.decode(errors="replace").strip()
    if proc.returncode == SSH_TRANSPORT_FAILURE_STATUS:
        result.update(status=PROBE_UNAVAILABLE, reason="ssh_transport_failure")
        return result
    if proc.returncode == PROBE_EXIT_NO_DIR:
        result.update(status=PROBE_UNAVAILABLE,
                      reason=f"no_such_directory:{target['script_path']}")
        return result
    if proc.returncode != 0:
        result.update(status=PROBE_ERROR,
                      reason=f"probe_exit_{proc.returncode}")
        return result

    status, hostname, observed, reason = parse_probe_output(
        proc.stdout.decode(errors="replace"), governed)
    result.update(status=status, hostname=hostname, observed=observed,
                  reason=reason)
    return result


def _utcnow():
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# the verdict
# ---------------------------------------------------------------------------

MATCH = "MATCH"
MISMATCH = "MISMATCH"
UNAVAILABLE = "UNAVAILABLE"


def evidence_rows(results, expected, governed=GOVERNED_FILES):
    """§C: one row per rig per governed file, in Beta's declared column set.

    hostname · canonical path · expected full sha256 · observed full sha256 ·
    file size · MATCH|MISMATCH|UNAVAILABLE · collection timestamp

    A rig whose probe did not run contributes one UNAVAILABLE row per governed
    file, with the observed digest rendered as `UNAVAILABLE` and NEVER as an
    empty string or a zero — the evidence must not let "not measured" read as
    "measured and empty".

    Beta's row vocabulary has three values, and both PROBE_UNAVAILABLE and
    PROBE_ERROR map to the row verdict `UNAVAILABLE` — neither was measured. So
    every row also carries `probe_status`, which keeps "the probe did not run"
    and "the probe ran and could not be classified" distinguishable inside the
    frozen bundle rather than collapsing them at the moment of freezing.
    """
    rows = []
    for r in results:
        unavailable = r["status"] != PROBE_OK
        for rel in governed:
            exp = expected[rel]
            if unavailable:
                rows.append({
                    "node_id": r["node_id"],
                    "endpoint": r["endpoint"],
                    "hostname": r["hostname"] or UNAVAILABLE,
                    "path": f"{r['script_path']}/{rel}",
                    "canonical_path": rel,
                    "expected_sha256": exp["sha256"],
                    "observed_sha256": UNAVAILABLE,
                    "expected_size": exp["size"],
                    "observed_size": UNAVAILABLE,
                    "verdict": UNAVAILABLE,
                    "probe_status": r["status"],
                    "reason": r["reason"],
                    "collected_at": r["collected_at"],
                })
                continue
            digest, size = r["observed"][rel]
            ok = digest == exp["sha256"]
            rows.append({
                "node_id": r["node_id"],
                "endpoint": r["endpoint"],
                "hostname": r["hostname"],
                "path": f"{r['script_path']}/{rel}",
                "canonical_path": rel,
                "expected_sha256": exp["sha256"],
                "observed_sha256": digest,
                "expected_size": exp["size"],
                "observed_size": size if size is not None else "-",
                "verdict": MATCH if ok else MISMATCH,
                "probe_status": r["status"],
                "reason": None if ok else (
                    "file_absent_on_rig" if digest == "MISSING"
                    else "digest_differs"),
                "collected_at": r["collected_at"],
            })
    return rows


def evaluate(results, expected, governed=GOVERNED_FILES):
    """Fail-close. Returns (allowed, refusals).

    The default is refusal: a status this function does not recognise falls
    through to "not allowed" rather than to "proceed".
    """
    refusals = []
    for r in results:
        if r["status"] == PROBE_UNAVAILABLE:
            refusals.append(
                f"{r['endpoint']}: {PROBE_UNAVAILABLE} (reason={r['reason']}) — "
                f"the parity probe did not run, so the deployed bytes are "
                f"UNKNOWN, not current"
                + (f" stderr={r['stderr']!r}" if r["stderr"] else ""))
            continue
        if r["status"] == PROBE_ERROR:
            refusals.append(
                f"{r['endpoint']}: {PROBE_ERROR} (reason={r['reason']}) — the "
                f"probe ran but its output could not be classified"
                + (f" stderr={r['stderr']!r}" if r["stderr"] else ""))
            continue
        if r["status"] != PROBE_OK:
            refusals.append(
                f"{r['endpoint']}: unrecognized probe status {r['status']!r} — "
                f"refusing by default")
            continue
        if r["hostname"] != r["expected_hostname"]:
            refusals.append(
                f"{r['endpoint']}: answered as hostname {r['hostname']!r}, "
                f"expected {r['expected_hostname']!r} — the probe did not reach "
                f"the machine the execution set names")
        for rel in governed:
            digest, _size = r["observed"][rel]
            if digest == expected[rel]["sha256"]:
                continue
            if digest == "MISSING":
                refusals.append(
                    f"{r['endpoint']}: {rel} is ABSENT from the deployed tree "
                    f"at {r['script_path']}")
            else:
                refusals.append(
                    f"{r['endpoint']}: {rel} sha256 {digest} != expected "
                    f"{expected[rel]['sha256']}")
    return (not refusals), refusals


def render_rig(r, expected, governed=GOVERNED_FILES):
    if r["status"] != PROBE_OK:
        return [f"  {r['endpoint']:<16} {r['status']:<12} "
                f"reason={r['reason']}"
                + (f" stderr={r['stderr']!r}" if r["stderr"] else "")]
    head = (f"  {r['endpoint']:<16} {PROBE_OK:<12} hostname={r['hostname']} "
            f"(expected {r['expected_hostname']})  path={r['script_path']}")
    lines = [head]
    for rel in governed:
        digest, size = r["observed"][rel]
        exp = expected[rel]["sha256"]
        if digest == exp:
            lines.append(f"      MATCH     {rel:<34} {digest}")
        elif digest == "MISSING":
            lines.append(f"      MISMATCH  {rel:<34} MISSING (expected {exp})")
        else:
            lines.append(f"      MISMATCH  {rel:<34} {digest}")
            lines.append(f"                {'':<34} expected {exp} "
                         f"(size {size} vs {expected[rel]['size']})")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rig-profile", default=None,
                    help="baremetal|proxmox; default = the profile map's own "
                         "default_profile")
    ap.add_argument("--evidence-json", default=None,
                    help="write the §C source-digest evidence bundle here")
    ap.add_argument("--no-verify-clean", dest="verify_clean",
                    action="store_false", default=True,
                    help="do not require the governed files to be clean in the "
                         "local worktree (the expectation is then not provably "
                         "canonical; for harness testing only)")
    args = ap.parse_args(argv)

    print("=" * 78)
    print("GATE-12 RIG CODE-PARITY GATE — deployed bytes must equal the "
          "canonical tree")
    print("=" * 78)

    # ---- the pin's own integrity, before anything is probed ---------------
    covered, uncovered, missing_minimum = closure_coverage()
    print(f"governed files    : {len(GOVERNED_FILES)} "
          f"(pinned; Beta minimum {len(BETA_MINIMUM_FILES)} included)")
    print(f"closure derived   : {WORKER_ENTRYPOINT} -> "
          f"{len(derive_worker_import_closure())} project files (AST, "
          f"repo-local only; no stdlib, no site-packages)")
    if not covered:
        print("")
        print("GATE-12 PARITY GATE : REFUSED — the governed set does not cover "
              "the worker's derived import closure:")
        for f in uncovered:
            print(f"  * {f} is in the worker's statically reachable "
                  f"project-local import / deployment closure and is NOT "
                  f"governed")
        for f in missing_minimum:
            print(f"  * {f} is Beta's declared minimum and is NOT in the pin")
        print("")
        print("A worker-side project import was added without parity coverage. "
              "Extend GOVERNED_FILES deliberately; do not widen it by accident.")
        return EXIT_REFUSE

    # ---- the expectation, and its canonicality ----------------------------
    expected = expected_digests()
    head = local_head()
    print(f"local HEAD        : {head}   [CONTEXT ONLY — never an input to the "
          f"verdict]")

    if args.verify_clean:
        dirty, git_unavailable = local_worktree_dirty()
        if git_unavailable is not None:
            print("")
            print(f"GATE-12 PARITY GATE : REFUSED — the local worktree state is "
                  f"UNAVAILABLE ({git_unavailable}). The expected digests "
                  f"cannot be shown to come from a canonical clean tree.")
            return EXIT_REFUSE
        if dirty:
            print("")
            print("GATE-12 PARITY GATE : REFUSED — governed files are not clean "
                  "in the local tree, so the expectation is not canonical:")
            for line in dirty:
                print(f"  * {line}")
            return EXIT_REFUSE
        print("local tree        : governed files CLEAN (expectation is "
              "canonical)")

    try:
        targets = gate_targets(args.rig_profile)
    except Exception as exc:                                        # noqa: BLE001
        print("")
        print(f"GATE-12 PARITY GATE : REFUSED — could not resolve the fleet "
              f"from committed source: {type(exc).__name__}: {exc}")
        return EXIT_REFUSE

    print(f"rigs (derived)    : {', '.join(t['endpoint'] for t in targets)}")
    print("")

    results = [probe_rig(t) for t in targets]
    for r in results:
        for line in render_rig(r, expected):
            print(line)
    print("")

    rows = evidence_rows(results, expected)
    if args.evidence_json:
        bundle = {
            "gate": "gate12_parity_gate",
            "generated_at": _utcnow(),
            "local_head_context_only": head,
            "governed_files": list(GOVERNED_FILES),
            "beta_minimum_files": list(BETA_MINIMUM_FILES),
            "worker_entrypoint": WORKER_ENTRYPOINT,
            "derived_closure": sorted(derive_worker_import_closure()),
            "expected": expected,
            "rows": rows,
        }
        with open(args.evidence_json, "w", encoding="utf-8") as fh:
            json.dump(bundle, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"source-digest evidence bundle : {args.evidence_json}")

    allowed, refusals = evaluate(results, expected)
    counts = {MATCH: 0, MISMATCH: 0, UNAVAILABLE: 0}
    for row in rows:
        counts[row["verdict"]] += 1
    print(f"rows              : {counts[MATCH]} MATCH · "
          f"{counts[MISMATCH]} MISMATCH · {counts[UNAVAILABLE]} UNAVAILABLE")
    print("")

    if allowed:
        print(f"GATE-12 PARITY GATE : PASS — {len(results)}/{len(results)} rigs "
              f"carry every governed file at the canonical digest. Launch may "
              f"proceed.")
        return EXIT_PROCEED

    print(f"GATE-12 PARITY GATE : REFUSED — {len(refusals)} parity failure(s):")
    for line in refusals:
        print(f"  * {line}")
    print("")
    print("ABORTING BEFORE ANY WORKER IS DISPATCHED. The D6 dry run of "
          "2026-08-14 dispatched into this exact condition and 24 of 25 workers")
    print("died at argparse; the rigs had been carrying a pre-Defect-A worker "
          "since 2026-08-02, through attempts 3, 4 and 5. A digest mismatch is")
    print("a refusal, not a warning: deploy the canonical files with the "
          "workers stopped, re-run this gate, and launch only on a PASS.")
    return EXIT_REFUSE


if __name__ == "__main__":
    sys.exit(main())
