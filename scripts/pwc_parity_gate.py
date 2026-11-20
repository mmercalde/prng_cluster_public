#!/usr/bin/env python3
"""PWC RIG CODE-PARITY GATE — the deployed PWC worker bytes must equal the local tree.

WHY THIS EXISTS
---------------
`scripts/gate12_parity_gate.py` pins the files the RANGE-MINER worker executes. The
PWC route executes different files on the rigs, and none of the TCP route's files is
in that pin (docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STAGE_A.md P3 gap 6b). The stale-rig
discovery behind gate12 (a rig carrying a worker ten days older than the tree) shows
rig bytes cannot be assumed. This gate measures them for the PWC route, separately.

It is built in the manner of gate12 and deliberately imports NOTHING from it, so no
change made to gate12 for the miner can change what this gate decides, and nothing
here can change gate12. It reads `preflight_check` for the three-outcome probe
vocabulary and `execution_set` for the fleet, exactly as gate12 does.

WHAT IT GOVERNS
---------------
Per transport, the project-local files the rig executes on that PWC route:

  tcp  `python -m persistent.pwc_worker_service` (persistent_worker_coordinator.py,
       _tcp_launch_workers), which imports persistent.pwc_transport_tcp at module
       scope and sieve_filter at init — and everything those reach.
  ssh  `sieve_gpu_worker.py --persistent` (persistent_worker_coordinator.py,
       _spawn_worker), and everything it reaches.

Each pin is cross-checked against an AST derivation of the entrypoint's statically
reachable project-local import closure, PLUS the package `__init__.py` files that
running those modules executes (gate12's derivation does not add parent packages;
`python -m persistent.pwc_worker_service` executes persistent/__init__.py whether or
not anything imports the package by name). A derived file the pin does not cover is
a REFUSE until the pin is extended deliberately.

Expected digests are the full sha256 of the local tree, derived at run time, never
transcribed. Git identity is recorded as context only and is never an input.

THE RULE — every failure ends in REFUSE:

    every governed file MATCHES on every resolved rig      -> PROCEED (exit 0)
    any governed file MISMATCHES or is MISSING on a rig    -> REFUSE
    a rig answers under another hostname                   -> REFUSE
    UNAVAILABLE (ssh failure, timeout, truncated output,
                 no such directory)                        -> REFUSE
    ERROR (probe ran, output unclassifiable)               -> REFUSE
    an unrecognised probe status                           -> REFUSE
    the derived closure is not covered by the pin          -> REFUSE
    a governed file is dirty in the local worktree         -> REFUSE
    git cannot report the local worktree state             -> REFUSE
    the fleet cannot be resolved from committed source     -> REFUSE
    the fleet resolves to ZERO rigs                        -> REFUSE (gate12 would
                                                              pass 0/0; a parity gate
                                                              that measured nothing
                                                              has proven nothing)
    any exception inside the gate                          -> REFUSE

Exit status: 0 = proceed, 1 = REFUSE.

The runner (subprocess) and the fleet resolver are injectable, so every branch is
testable on CPU without touching a rig (tests/test_pwc_backup_step1.py).
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
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import preflight_check as PF                                        # noqa: E402

PROBE_OK = PF.GPU_PROBE_OK
PROBE_UNAVAILABLE = PF.GPU_PROBE_UNAVAILABLE
PROBE_ERROR = PF.GPU_PROBE_ERROR

SSH_CONNECT_TIMEOUT = PF.SSH_TIMEOUT_SECONDS
PROBE_TIMEOUT = PF.GPU_CHECK_TIMEOUT_SECONDS

# ssh's own failure status; the probe script below never exits 255 (its only
# non-zero exit is PROBE_EXIT_NO_DIR).
SSH_TRANSPORT_FAILURE_STATUS = 255
PROBE_EXIT_NO_DIR = 3

EXIT_PROCEED = 0
EXIT_REFUSE = 1

BEGIN_SENTINEL = "TFM-PWC-PARITY-BEGIN"
END_SENTINEL = "TFM-PWC-PARITY-END"

MATCH = "MATCH"
MISMATCH = "MISMATCH"
UNAVAILABLE = "UNAVAILABLE"

# ---------------------------------------------------------------------------
# THE PINS
# ---------------------------------------------------------------------------
ENTRYPOINTS = {
    "tcp": "persistent/pwc_worker_service.py",
    "ssh": "sieve_gpu_worker.py",
}

GOVERNED_FILES = {
    "tcp": (
        "adaptive_thresholds.py",
        "core/results_manager.py",
        "hybrid_strategy.py",
        "integration/__init__.py",
        "integration/metadata_writer.py",
        "integration/sieve_integration.py",
        "persistent/__init__.py",
        "persistent/pwc_result_normalizer.py",
        "persistent/pwc_transport_base.py",
        "persistent/pwc_transport_tcp.py",
        "persistent/pwc_worker_service.py",
        "prng_registry.py",
        "sieve_filter.py",
    ),
    "ssh": (
        "adaptive_thresholds.py",
        "hybrid_strategy.py",
        "prng_registry.py",
        "sieve_gpu_worker.py",
    ),
}

# The files the step-1 brief names (S8), asserted against the pin so a later
# narrowing cannot drop one silently.
MINIMUM_FILES = {
    "tcp": ("persistent/pwc_worker_service.py", "persistent/pwc_transport_tcp.py",
            "sieve_filter.py"),
    "ssh": ("sieve_gpu_worker.py",),
}


# ---------------------------------------------------------------------------
# derivation
# ---------------------------------------------------------------------------
def _module_to_repo_path(module, repo_root):
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
    """Every module name the file imports, at ANY scope (deferred imports execute)."""
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
                module = (f"{base}.{node.module}" if base else node.module) \
                    if node.module else base
            else:
                module = node.module or ""
            if module:
                names.add(module)
            for alias in node.names:
                names.add(f"{module}.{alias.name}" if module else alias.name)
    return names


def _parent_package_inits(relpath, repo_root):
    """Package __init__.py files executed when `relpath` is imported."""
    out = []
    d = os.path.dirname(relpath)
    while d:
        init = f"{d}/__init__.py"
        if os.path.isfile(os.path.join(repo_root, init)):
            out.append(init)
        d = os.path.dirname(d)
    return out


def derive_closure(transport, repo_root=REPO_ROOT):
    root = ENTRYPOINTS[transport]
    seen = set()
    stack = [root]
    while stack:
        rel = stack.pop()
        if rel in seen:
            continue
        seen.add(rel)
        for init in _parent_package_inits(rel, repo_root):
            if init not in seen:
                stack.append(init)
        for module in _imported_names(rel, repo_root):
            resolved = _module_to_repo_path(module, repo_root)
            if resolved and resolved not in seen:
                stack.append(resolved)
    return seen


def closure_coverage(transport, repo_root=REPO_ROOT, governed=None):
    governed = GOVERNED_FILES[transport] if governed is None else governed
    derived = derive_closure(transport, repo_root)
    uncovered = sorted(derived - set(governed))
    missing_minimum = [f for f in MINIMUM_FILES[transport] if f not in governed]
    return (not uncovered and not missing_minimum), uncovered, missing_minimum


# ---------------------------------------------------------------------------
# expectation
# ---------------------------------------------------------------------------
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_digests(governed, repo_root=REPO_ROOT):
    out = {}
    for rel in governed:
        p = os.path.join(repo_root, rel)
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"governed file {rel!r} does not exist in the local tree at {repo_root}")
        out[rel] = {"sha256": sha256_of(p), "size": os.path.getsize(p)}
    return out


def _run(cmd, timeout):
    return subprocess.run(cmd, capture_output=True, timeout=timeout)


def local_worktree_dirty(governed, repo_root=REPO_ROOT, runner=None):
    """(porcelain lines for governed paths, unavailable_reason)."""
    runner = runner or _run
    try:
        proc = runner(["git", "-C", repo_root, "status", "--porcelain", "--"]
                      + list(governed), PROBE_TIMEOUT)
    except Exception as exc:                                        # noqa: BLE001
        return [], f"{type(exc).__name__}:{exc}"
    if proc.returncode != 0:
        return [], (f"git exited {proc.returncode}: "
                    f"{proc.stderr.decode(errors='replace').strip()!r}")
    return [ln for ln in proc.stdout.decode(errors="replace").splitlines()
            if ln.strip()], None


def local_head(repo_root=REPO_ROOT, runner=None):
    """CONTEXT ONLY — never an input to the verdict."""
    runner = runner or _run
    try:
        proc = runner(["git", "-C", repo_root, "rev-parse", "HEAD"], PROBE_TIMEOUT)
    except Exception:                                               # noqa: BLE001
        return "UNAVAILABLE"
    if proc.returncode != 0:
        return "UNAVAILABLE"
    return proc.stdout.decode(errors="replace").strip() or "UNAVAILABLE"


# ---------------------------------------------------------------------------
# the fleet and the probe
# ---------------------------------------------------------------------------
def gate_targets(rig_profile=None, repo_root=REPO_ROOT, config_path=None,
                 profile_map_path=None, declared_nodes=None):
    """The rigs, from the committed execution set for backend `pwc`, joined with
    distributed_config.json for the directory each rig runs the worker from
    (`script_path`, which _tcp_launch_workers `cd`s into).

    `declared_nodes` is the ONLY way to gate a partial fleet, exactly as in
    execution_set.resolve_execution_set: an explicit list of node ids, never a
    subset inferred from which rigs answered. An unknown name is a hard error."""
    import execution_set as XS
    kwargs = dict(backend=XS.BACKEND_PWC, invoked_by="pwc_parity_gate",
                  rig_profile=rig_profile, repo_root=repo_root)
    if declared_nodes:
        kwargs["declared_nodes"] = list(declared_nodes)
    if config_path:
        kwargs["config_path"] = config_path
    if profile_map_path:
        kwargs["profile_map_path"] = profile_map_path
    resolved = XS.resolve_execution_set(**kwargs)
    with open(config_path or os.path.join(repo_root, "distributed_config.json")) as fh:
        by_host = {n["hostname"]: n for n in json.load(fh)["nodes"]}
    targets = []
    for node in resolved.remote_nodes():
        cfg = by_host.get(node.config_hostname)
        if cfg is None:
            raise KeyError(f"no distributed_config.json node for {node.config_hostname!r}")
        targets.append({
            "node_id": node.node_id,
            "endpoint": node.endpoint,
            "ssh_user": node.ssh_user,
            "worker_hostname": node.worker_hostname,
            "script_path": cfg["script_path"],
        })
    return targets


def build_probe_script(script_path, governed):
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


def parse_probe_output(text, governed):
    """(status, hostname, {relpath: (digest|MISSING, size|None)}, reason)."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if BEGIN_SENTINEL not in lines:
        return PROBE_ERROR, None, {}, "no_begin_sentinel"
    if END_SENTINEL not in lines:
        return PROBE_UNAVAILABLE, None, {}, "truncated_probe_output"
    body = lines[lines.index(BEGIN_SENTINEL) + 1:lines.index(END_SENTINEL)]
    hostname, observed = None, {}
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
                return PROBE_ERROR, hostname, {}, f"malformed_digest_for:{rel}"
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
        return PROBE_ERROR, hostname, observed, "no_row_for:" + ",".join(absent)
    return PROBE_OK, hostname, observed, None


def _utcnow():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def probe_rig(target, governed, runner=None):
    runner = runner or _run
    result = {"node_id": target["node_id"], "endpoint": target["endpoint"],
              "expected_hostname": target["worker_hostname"],
              "script_path": target["script_path"], "status": None,
              "hostname": None, "observed": {}, "reason": None, "stderr": "",
              "collected_at": None}
    try:
        proc = runner(["ssh", "-n", "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT}",
                       "-o", "BatchMode=yes",
                       f"{target['ssh_user']}@{target['endpoint']}",
                       build_probe_script(target["script_path"], governed)],
                      PROBE_TIMEOUT)
    except subprocess.TimeoutExpired:
        result.update(status=PROBE_UNAVAILABLE, reason="timeout", collected_at=_utcnow())
        return result
    except Exception as exc:                                        # noqa: BLE001
        result.update(status=PROBE_UNAVAILABLE, reason=f"{type(exc).__name__}:{exc}",
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
        result.update(status=PROBE_ERROR, reason=f"probe_exit_{proc.returncode}")
        return result
    status, hostname, observed, reason = parse_probe_output(
        proc.stdout.decode(errors="replace"), governed)
    result.update(status=status, hostname=hostname, observed=observed, reason=reason)
    return result


# ---------------------------------------------------------------------------
# the verdict
# ---------------------------------------------------------------------------
def evaluate(results, expected, governed):
    """Fail-close: (allowed, refusals). Refusal is the default for anything not
    recognised, and an empty fleet is a refusal."""
    refusals = []
    if not results:
        refusals.append("the fleet resolved to ZERO rigs — nothing was measured, so "
                        "nothing is proven")
    for r in results:
        if r["status"] == PROBE_UNAVAILABLE:
            refusals.append(f"{r['endpoint']}: {PROBE_UNAVAILABLE} (reason={r['reason']}) "
                            f"— deployed bytes UNKNOWN"
                            + (f" stderr={r['stderr']!r}" if r["stderr"] else ""))
            continue
        if r["status"] == PROBE_ERROR:
            refusals.append(f"{r['endpoint']}: {PROBE_ERROR} (reason={r['reason']})"
                            + (f" stderr={r['stderr']!r}" if r["stderr"] else ""))
            continue
        if r["status"] != PROBE_OK:
            refusals.append(f"{r['endpoint']}: unrecognised probe status "
                            f"{r['status']!r} — refusing by default")
            continue
        if r["hostname"] != r["expected_hostname"]:
            refusals.append(f"{r['endpoint']}: answered as {r['hostname']!r}, expected "
                            f"{r['expected_hostname']!r}")
        for rel in governed:
            digest, _ = r["observed"].get(rel, (None, None))
            if digest == expected[rel]["sha256"]:
                continue
            if digest == "MISSING":
                refusals.append(f"{r['endpoint']}: {rel} is ABSENT at {r['script_path']}")
            else:
                refusals.append(f"{r['endpoint']}: {rel} sha256 {digest} != expected "
                                f"{expected[rel]['sha256']}")
    return (not refusals), refusals


def evidence_rows(results, expected, governed):
    rows = []
    for r in results:
        for rel in governed:
            exp = expected[rel]
            if r["status"] != PROBE_OK:
                verdict, obs, size = UNAVAILABLE, UNAVAILABLE, UNAVAILABLE
            else:
                obs, size = r["observed"].get(rel, (UNAVAILABLE, None))
                verdict = MATCH if obs == exp["sha256"] else MISMATCH
            rows.append({"node_id": r["node_id"], "endpoint": r["endpoint"],
                         "hostname": r["hostname"] or UNAVAILABLE,
                         "canonical_path": rel,
                         "path": f"{r['script_path']}/{rel}",
                         "expected_sha256": exp["sha256"], "observed_sha256": obs,
                         "expected_size": exp["size"],
                         "observed_size": size if size is not None else "-",
                         "verdict": verdict, "probe_status": r["status"],
                         "reason": r["reason"], "collected_at": r["collected_at"]})
    return rows


def run_gate(transport, *, rig_profile=None, verify_clean=True, evidence_json=None,
             repo_root=REPO_ROOT, runner=None, resolve_targets=None, out=print,
             declared_nodes=None):
    """The whole gate. Returns EXIT_PROCEED or EXIT_REFUSE; never raises."""
    if resolve_targets is None:
        def resolve_targets(prof):
            return gate_targets(prof, repo_root=repo_root, declared_nodes=declared_nodes)
    try:
        return _run_gate(transport, rig_profile=rig_profile, verify_clean=verify_clean,
                         evidence_json=evidence_json, repo_root=repo_root,
                         runner=runner, resolve_targets=resolve_targets, out=out)
    except Exception as exc:                                        # noqa: BLE001
        out(f"PWC PARITY GATE : REFUSED — the gate itself failed: "
            f"{type(exc).__name__}: {exc}")
        return EXIT_REFUSE


def _run_gate(transport, *, rig_profile, verify_clean, evidence_json, repo_root,
              runner, resolve_targets, out):
    if transport not in GOVERNED_FILES:
        out(f"PWC PARITY GATE : REFUSED — unknown transport {transport!r}; expected "
            f"one of {sorted(GOVERNED_FILES)}")
        return EXIT_REFUSE
    governed = GOVERNED_FILES[transport]
    out("=" * 78)
    out(f"PWC RIG CODE-PARITY GATE [{transport}] — deployed bytes must equal the tree")
    out("=" * 78)

    covered, uncovered, missing_minimum = closure_coverage(transport, repo_root)
    out(f"governed files    : {len(governed)} (pinned for {ENTRYPOINTS[transport]})")
    if not covered:
        out("PWC PARITY GATE : REFUSED — the pin does not cover the derived closure:")
        for f in uncovered:
            out(f"  * {f} is reachable from {ENTRYPOINTS[transport]} and NOT governed")
        for f in missing_minimum:
            out(f"  * {f} is a named minimum file and NOT in the pin")
        return EXIT_REFUSE

    expected = expected_digests(governed, repo_root)
    head = local_head(repo_root, runner)
    out(f"local HEAD        : {head}   [CONTEXT ONLY — never an input to the verdict]")

    if verify_clean:
        dirty, unavailable = local_worktree_dirty(governed, repo_root, runner)
        if unavailable is not None:
            out(f"PWC PARITY GATE : REFUSED — local worktree state UNAVAILABLE "
                f"({unavailable})")
            return EXIT_REFUSE
        if dirty:
            out("PWC PARITY GATE : REFUSED — governed files are not clean locally:")
            for line in dirty:
                out(f"  * {line}")
            return EXIT_REFUSE
        out("local tree        : governed files CLEAN")

    try:
        targets = resolve_targets(rig_profile)
    except Exception as exc:                                        # noqa: BLE001
        out(f"PWC PARITY GATE : REFUSED — could not resolve the fleet from committed "
            f"source: {type(exc).__name__}: {exc}")
        return EXIT_REFUSE
    out(f"rigs (derived)    : {', '.join(t['endpoint'] for t in targets) or 'NONE'}")

    results = [probe_rig(t, governed, runner) for t in targets]
    rows = evidence_rows(results, expected, governed)
    if evidence_json:
        with open(evidence_json, "w", encoding="utf-8") as fh:
            json.dump({"gate": "pwc_parity_gate", "transport": transport,
                       "generated_at": _utcnow(), "local_head_context_only": head,
                       "entrypoint": ENTRYPOINTS[transport],
                       "governed_files": list(governed),
                       "derived_closure": sorted(derive_closure(transport, repo_root)),
                       "expected": expected, "rows": rows},
                      fh, indent=2, sort_keys=True)
            fh.write("\n")
        out(f"evidence bundle   : {evidence_json}")

    allowed, refusals = evaluate(results, expected, governed)
    counts = {MATCH: 0, MISMATCH: 0, UNAVAILABLE: 0}
    for row in rows:
        counts[row["verdict"]] += 1
    out(f"rows              : {counts[MATCH]} MATCH · {counts[MISMATCH]} MISMATCH · "
        f"{counts[UNAVAILABLE]} UNAVAILABLE")
    if allowed:
        out(f"PWC PARITY GATE : PASS — {len(results)}/{len(results)} rigs carry every "
            f"governed {transport} file at the local digest.")
        return EXIT_PROCEED
    out(f"PWC PARITY GATE : REFUSED — {len(refusals)} failure(s):")
    for line in refusals:
        out(f"  * {line}")
    out("Do not launch PWC. Deploy the local files with the workers stopped and re-run.")
    return EXIT_REFUSE


def main(argv=None, *, runner=None, resolve_targets=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--transport", required=True, choices=sorted(GOVERNED_FILES),
                    help="the PWC route whose rig files are checked (no default)")
    ap.add_argument("--rig-profile", default=None,
                    help="baremetal|proxmox; default = the profile map's default_profile")
    ap.add_argument("--nodes", nargs="+", default=None,
                    help="explicit node ids for a partial fleet (e.g. rrig6600 rrig6600b); "
                         "default = every remote node of the execution set")
    ap.add_argument("--evidence-json", default=None)
    ap.add_argument("--no-verify-clean", dest="verify_clean", action="store_false",
                    default=True,
                    help="do not require the governed files to be clean locally "
                         "(harness testing only)")
    args = ap.parse_args(argv)
    return run_gate(args.transport, rig_profile=args.rig_profile,
                    verify_clean=args.verify_clean, evidence_json=args.evidence_json,
                    runner=runner, resolve_targets=resolve_targets,
                    declared_nodes=args.nodes)


if __name__ == "__main__":
    sys.exit(main())
