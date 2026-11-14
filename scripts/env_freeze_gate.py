#!/usr/bin/env python3
"""G2 ENVIRONMENT-FREEZE GATE — refuse a launch if any host differs from the owner's baseline.

WHAT THIS IS
------------
The enforcement of "agents never update the system"
(`docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md` §4 items 1-2, §7.2). The owner
captures a baseline of every host in the resolved execution set once; this gate
compares live state to it before every delegated launch (`--phase pre`) and again
after the run (`--phase post`). Modelled on `scripts/gate12_parity_gate.py`, whose
vocabulary, exit codes, transport classification and ssh discipline it imports.

WHAT IS COMPARED (per host, the "record")
-----------------------------------------
    kernel_release        uname -r            (a CT100 sees its Proxmox HOST kernel)
    kernel_cmdline        /proc/cmdline
    python_version        <python_env> platform.python_version()
    pip_freeze            <python_env> -m pip freeze --all   (sha256 + sorted list)
    os_packages           dpkg-query -W  status/package/version (sha256 + sorted list)
    gpu_driver_nvidia     /proc/driver/nvidia/version  (version or ABSENT)
    gpu_driver_amdgpu     /sys/module/amdgpu/version   (version or ABSENT)
    cuda_version          /usr/local/cuda/version.json  cuda.version (or ABSENT)
    rocm_version          /opt/rocm/.info/version        (or ABSENT)

`python_env` comes from `distributed_config.json` per node (the same field the
launcher runs the worker with); nothing here names an interpreter path.

WHAT IS NEVER COMPARED
----------------------
Timestamps, endpoints, `local HEAD`, the capture's own provenance — CONTEXT ONLY.
`compare_host_records` takes two records and nothing else, so nothing printed for
context can reach a verdict. The one exception is the hostname check the parity
gate already makes: a host answering under another machine's name is REFUSE.

THE RULE — fail closed:
    every fact on every host equals the baseline                 -> PROCEED (exit 0)
    any fact differs on any host                                 -> REFUSE
    a host in the execution set is absent from the baseline,
      or a baseline host is absent from the execution set        -> REFUSE
    UNAVAILABLE (transport failure, timeout, truncated output)   -> REFUSE
    ERROR (probe ran, output unparseable)                        -> REFUSE
    baseline absent / untracked / differs from HEAD              -> REFUSE
    git cannot run                                               -> UNAVAILABLE -> REFUSE
`--phase post` applies the same rules and prints VIOLATION in place of REFUSE.

Exit status: 0 = proceed, 1 = REFUSE / VIOLATION.
"""

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
_SCRIPTS = os.path.join(REPO_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

# One vocabulary, one transport classification, one pair of exit codes — imported
# from the parity gate rather than re-declared.
import gate12_parity_gate as PG                                     # noqa: E402

PROBE_OK = PG.PROBE_OK
PROBE_UNAVAILABLE = PG.PROBE_UNAVAILABLE
PROBE_ERROR = PG.PROBE_ERROR
EXIT_PROCEED = PG.EXIT_PROCEED
EXIT_REFUSE = PG.EXIT_REFUSE
SSH_CONNECT_TIMEOUT = PG.SSH_CONNECT_TIMEOUT
SSH_TRANSPORT_FAILURE_STATUS = PG.SSH_TRANSPORT_FAILURE_STATUS
_run = PG._run
_utcnow = PG._utcnow

# pip freeze + dpkg-query over ~2.4k packages is slower than a sha256 of ten files;
# the parity gate's 15 s budget is not enough headroom on a loaded rig.
PROBE_TIMEOUT = 4 * PG.PROBE_TIMEOUT

BEGIN_SENTINEL = "TFM-ENVFREEZE-BEGIN"
END_SENTINEL = "TFM-ENVFREEZE-END"
PIP_BEGIN, PIP_END = "PIP-BEGIN", "PIP-END"
DPKG_BEGIN, DPKG_END = "DPKG-BEGIN", "DPKG-END"
ABSENT = "ABSENT"

BASELINE_REL = "config_manifests/env_freeze_baseline.cfg"
BASELINE_SCHEMA = "tfm-env-freeze-baseline/1"

SCALAR_FACTS = ("kernel_release", "kernel_cmdline", "python_version",
                "gpu_driver_nvidia", "gpu_driver_amdgpu",
                "cuda_version", "rocm_version")
LIST_FACTS = ("pip_freeze", "os_packages")
ALL_FACTS = SCALAR_FACTS + LIST_FACTS


# ---------------------------------------------------------------------------
# tracked-and-unmodified: only the owner commits, so a tracked file equal to
# HEAD is the owner's file
# ---------------------------------------------------------------------------

def tracked_unmodified(repo_root, rel, runner=None):
    """(status, reason, content_bytes).

    status: "OK" (file exists and is byte-identical to HEAD:<rel>), "ABSENT",
    "UNTRACKED", "MODIFIED", or UNAVAILABLE (git could not run). The bytes that
    were compared are returned, so the caller parses exactly what was checked.
    """
    runner = runner or _run
    path = os.path.join(repo_root, rel)
    if not os.path.isfile(path):
        return "ABSENT", f"{rel} does not exist", None
    with open(path, "rb") as fh:
        data = fh.read()
    try:
        proc = runner(["git", "-C", repo_root, "cat-file", "-e", f"HEAD:{rel}"],
                      PG.PROBE_TIMEOUT)
    except Exception as exc:                                        # noqa: BLE001
        return PROBE_UNAVAILABLE, f"git could not run: {type(exc).__name__}:{exc}", None
    if proc.returncode == 128 and b"not a git repository" in proc.stderr.lower():
        return PROBE_UNAVAILABLE, "not a git repository", None
    if proc.returncode != 0:
        return "UNTRACKED", f"{rel} is not in HEAD (untracked or uncommitted)", None
    try:
        proc = runner(["git", "-C", repo_root, "cat-file", "blob", f"HEAD:{rel}"],
                      PG.PROBE_TIMEOUT)
    except Exception as exc:                                        # noqa: BLE001
        return PROBE_UNAVAILABLE, f"git could not run: {type(exc).__name__}:{exc}", None
    if proc.returncode != 0:
        return PROBE_UNAVAILABLE, f"git cat-file exited {proc.returncode}", None
    if proc.stdout != data:
        return "MODIFIED", f"{rel} differs from HEAD:{rel}", None
    return "OK", None, data


# ---------------------------------------------------------------------------
# the hosts
# ---------------------------------------------------------------------------

def env_targets(rig_profile=None, repo_root=REPO_ROOT):
    """Every node of the resolved execution set, local node INCLUDED.

    `PG.gate_targets` is not reused: it drops the local node (`remote_nodes()`)
    and does not carry `python_env`. The join is the same one it performs —
    `resolve_execution_set` for the fleet, `distributed_config.json` keyed by
    `config_hostname` for per-node facts.
    """
    import execution_set as XS
    resolved = XS.resolve_execution_set(backend="miner",
                                        invoked_by="env_freeze_gate",
                                        rig_profile=rig_profile,
                                        repo_root=repo_root)
    with open(os.path.join(repo_root, XS.DISTRIBUTED_CONFIG_NAME)) as fh:
        by_host = {n["hostname"]: n for n in json.load(fh)["nodes"]}
    targets = []
    for node in resolved.nodes:
        cfg = by_host.get(node.config_hostname)
        if cfg is None or not cfg.get("python_env"):
            raise KeyError(f"no distributed_config.json python_env for "
                           f"{node.config_hostname!r} ({node.node_id})")
        targets.append({
            "node_id": node.node_id,
            "endpoint": node.endpoint,
            "ssh_user": node.ssh_user,
            "worker_hostname": node.worker_hostname,
            "local": bool(node.local),
            "python_env": cfg["python_env"],
        })
    return targets, resolved.rig_profile


def build_probe_script(python_env):
    """ONE script per host, framed BEGIN/END. Read-only; never exits 255."""
    py = python_env.replace("'", "'\\''")
    return (
        f"PY='{py}'\n"
        "export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1 LC_ALL=C\n"
        f"echo {BEGIN_SENTINEL}\n"
        "printf 'HOST\\t%s\\n' \"$(hostname)\"\n"
        "printf 'KERNEL_RELEASE\\t%s\\n' \"$(uname -r)\"\n"
        "printf 'KERNEL_CMDLINE\\t%s\\n' \"$(cat /proc/cmdline 2>/dev/null || echo ABSENT)\"\n"
        "if [ -x \"$PY\" ]; then\n"
        "  printf 'PYTHON_VERSION\\t%s\\n' \"$(\"$PY\" -c 'import platform;print(platform.python_version())' 2>/dev/null || echo ERROR)\"\n"
        f"  echo {PIP_BEGIN}\n"
        "  \"$PY\" -m pip freeze --all 2>/dev/null; rc=$?\n"
        f"  echo {PIP_END}\n"
        "  printf 'PIP_STATUS\\t%s\\n' \"$rc\"\n"
        "else\n"
        "  printf 'PYTHON_VERSION\\t%s\\n' ABSENT\n"
        f"  echo {PIP_BEGIN}; echo {PIP_END}\n"
        "  printf 'PIP_STATUS\\t%s\\n' ABSENT\n"
        "fi\n"
        f"echo {DPKG_BEGIN}\n"
        "dpkg-query -W -f='${binary:Package}\\t${db:Status-Abbrev}${Version}\\n' 2>/dev/null; rc=$?\n"
        f"echo {DPKG_END}\n"
        "printf 'DPKG_STATUS\\t%s\\n' \"$rc\"\n"
        "if [ -r /proc/driver/nvidia/version ]; then\n"
        "  printf 'GPU_DRIVER_NVIDIA\\t%s\\n' \"$(sed -n 's/.*Kernel Module *\\([0-9][0-9.]*\\).*/\\1/p' /proc/driver/nvidia/version | head -1)\"\n"
        "else printf 'GPU_DRIVER_NVIDIA\\tABSENT\\n'; fi\n"
        "if [ -r /sys/module/amdgpu/version ]; then\n"
        "  printf 'GPU_DRIVER_AMDGPU\\t%s\\n' \"$(cat /sys/module/amdgpu/version)\"\n"
        "elif [ -d /sys/module/amdgpu ]; then\n"
        "  printf 'GPU_DRIVER_AMDGPU\\tsrcversion:%s\\n' \"$(cat /sys/module/amdgpu/srcversion 2>/dev/null)\"\n"
        "else printf 'GPU_DRIVER_AMDGPU\\tABSENT\\n'; fi\n"
        "if [ -r /usr/local/cuda/version.json ]; then\n"
        "  printf 'CUDA_JSON_B64\\t%s\\n' \"$(base64 -w0 /usr/local/cuda/version.json)\"\n"
        "else printf 'CUDA_JSON_B64\\tABSENT\\n'; fi\n"
        "if [ -r /opt/rocm/.info/version ]; then\n"
        "  printf 'ROCM_VERSION\\t%s\\n' \"$(head -1 /opt/rocm/.info/version)\"\n"
        "else printf 'ROCM_VERSION\\tABSENT\\n'; fi\n"
        f"echo {END_SENTINEL}\n"
    )


def _list_fact(lines):
    entries = sorted(ln for ln in lines if ln.strip())
    digest = hashlib.sha256(("\n".join(entries) + "\n").encode()).hexdigest()
    return {"sha256": digest, "entries": entries}


def parse_probe_output(text):
    """(status, hostname, record, reason). Classifies; never repairs."""
    lines = text.splitlines()
    stripped = [ln.strip() for ln in lines]
    if BEGIN_SENTINEL not in stripped:
        return PROBE_ERROR, None, None, "no_begin_sentinel"
    if END_SENTINEL not in stripped:
        return PROBE_UNAVAILABLE, None, None, "truncated_probe_output"
    body = lines[stripped.index(BEGIN_SENTINEL) + 1:stripped.index(END_SENTINEL)]

    kv, blocks, cur = {}, {}, None
    for ln in body:
        s = ln.strip()
        if s in (PIP_BEGIN, DPKG_BEGIN):
            if cur is not None or s in blocks:
                return PROBE_ERROR, None, None, f"nested_or_repeated_block:{s}"
            cur, blocks[s] = s, []
            continue
        if s in (PIP_END, DPKG_END):
            if cur is None or {PIP_END: PIP_BEGIN, DPKG_END: DPKG_BEGIN}[s] != cur:
                return PROBE_ERROR, None, None, f"unbalanced_block:{s}"
            cur = None
            continue
        if cur is not None:
            blocks[cur].append(ln.rstrip("\n"))
            continue
        if not s:
            continue
        key, sep, val = ln.partition("\t")
        if not sep or key in kv:
            return PROBE_ERROR, None, None, f"malformed_or_repeated_row:{ln!r}"
        kv[key] = val
    if cur is not None:
        return PROBE_ERROR, None, None, f"unterminated_block:{cur}"

    need = ("HOST", "KERNEL_RELEASE", "KERNEL_CMDLINE", "PYTHON_VERSION",
            "PIP_STATUS", "DPKG_STATUS", "GPU_DRIVER_NVIDIA",
            "GPU_DRIVER_AMDGPU", "CUDA_JSON_B64", "ROCM_VERSION")
    missing = [k for k in need if k not in kv]
    if missing or PIP_BEGIN not in blocks or DPKG_BEGIN not in blocks:
        return PROBE_ERROR, None, None, "missing_rows:" + ",".join(
            missing + [b for b in (PIP_BEGIN, DPKG_BEGIN) if b not in blocks])
    hostname = kv["HOST"].strip()
    if not hostname:
        return PROBE_ERROR, None, None, "empty_host_row"
    for k in ("KERNEL_RELEASE", "PYTHON_VERSION", "GPU_DRIVER_NVIDIA",
              "GPU_DRIVER_AMDGPU", "ROCM_VERSION"):
        if not kv[k].strip() or kv[k].strip() == "ERROR":
            return PROBE_ERROR, hostname, None, f"unreadable:{k}"
    if kv["PIP_STATUS"] not in ("0", ABSENT):
        return PROBE_ERROR, hostname, None, f"pip_freeze_exit_{kv['PIP_STATUS']}"
    if kv["DPKG_STATUS"] != "0" or not blocks[DPKG_BEGIN]:
        return PROBE_ERROR, hostname, None, f"dpkg_query_exit_{kv['DPKG_STATUS']}"

    if kv["CUDA_JSON_B64"] == ABSENT:
        cuda = ABSENT
    else:
        try:
            cuda = json.loads(base64.b64decode(kv["CUDA_JSON_B64"], validate=True)
                              )["cuda"]["version"]
        except Exception:                                           # noqa: BLE001
            return PROBE_ERROR, hostname, None, "unparseable_cuda_version_json"
        if not isinstance(cuda, str) or not cuda:
            return PROBE_ERROR, hostname, None, "unparseable_cuda_version_json"

    record = {
        "kernel_release": kv["KERNEL_RELEASE"].strip(),
        "kernel_cmdline": kv["KERNEL_CMDLINE"].strip(),
        "python_version": kv["PYTHON_VERSION"].strip(),
        "pip_freeze": _list_fact(blocks[PIP_BEGIN]),
        "os_packages": _list_fact(blocks[DPKG_BEGIN]),
        "gpu_driver_nvidia": kv["GPU_DRIVER_NVIDIA"].strip(),
        "gpu_driver_amdgpu": kv["GPU_DRIVER_AMDGPU"].strip(),
        "cuda_version": cuda,
        "rocm_version": kv["ROCM_VERSION"].strip(),
    }
    return PROBE_OK, hostname, record, None


def probe_host(target, runner=None):
    """One host, one script, three outcomes. The local node runs under bash, no ssh."""
    runner = runner or _run
    script = build_probe_script(target["python_env"])
    if target["local"]:
        cmd = ["bash", "-c", script]
    else:
        cmd = ["ssh", "-n", "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT}",
               "-o", "BatchMode=yes",
               f"{target['ssh_user']}@{target['endpoint']}", script]
    result = {"node_id": target["node_id"], "endpoint": target["endpoint"],
              "expected_hostname": target["worker_hostname"],
              "local": target["local"], "status": None, "hostname": None,
              "record": None, "reason": None, "stderr": "", "collected_at": None}
    try:
        proc = runner(cmd, PROBE_TIMEOUT)
    except subprocess.TimeoutExpired:
        result.update(status=PROBE_UNAVAILABLE, reason="timeout",
                      collected_at=_utcnow())
        return result
    except Exception as exc:                                        # noqa: BLE001
        result.update(status=PROBE_UNAVAILABLE,
                      reason=f"{type(exc).__name__}:{exc}", collected_at=_utcnow())
        return result
    result["collected_at"] = _utcnow()
    result["stderr"] = proc.stderr.decode(errors="replace").strip()[-2000:]
    if not target["local"] and proc.returncode == SSH_TRANSPORT_FAILURE_STATUS:
        result.update(status=PROBE_UNAVAILABLE, reason="ssh_transport_failure")
        return result
    status, hostname, record, reason = parse_probe_output(
        proc.stdout.decode(errors="replace"))
    if status == PROBE_OK and proc.returncode != 0:
        status, reason = PROBE_ERROR, f"probe_exit_{proc.returncode}"
    result.update(status=status, hostname=hostname, record=record, reason=reason)
    return result


# ---------------------------------------------------------------------------
# the comparison — two records in, differences out, nothing else consulted
# ---------------------------------------------------------------------------

def _keyed(entries):
    """name -> rest. pip: `name==ver` / `name @ url`; dpkg: `pkg\\tstatusver`."""
    out = {}
    for e in entries:
        if "\t" in e:
            k, _, v = e.partition("\t")
        elif "==" in e:
            k, _, v = e.partition("==")
            k = k.strip().lower()
        elif " @ " in e:
            k, _, v = e.partition(" @ ")
            k = k.strip().lower()
        else:
            k, v = e, ""
        if k in out:                       # duplicate key: keep both, never merge
            k = f"{k}#dup{len(out)}"
        out[k] = v
    return out


def compare_host_records(baseline, live):
    """Every difference between two host records. Empty list == identical."""
    diffs = []
    for fact in sorted(set(baseline) | set(live)):
        if fact not in baseline or fact not in live:
            diffs.append({"fact": fact, "kind": "fact_missing",
                          "baseline": baseline.get(fact, ABSENT),
                          "live": live.get(fact, ABSENT)})
            continue
        b, lv = baseline[fact], live[fact]
        if fact in LIST_FACTS:
            if not (isinstance(b, dict) and isinstance(lv, dict)):
                diffs.append({"fact": fact, "kind": "malformed",
                              "baseline": b, "live": lv})
                continue
            bk, lk = _keyed(b.get("entries", [])), _keyed(lv.get("entries", []))
            added = sorted(f"{k} {lk[k]}".strip() for k in set(lk) - set(bk))
            removed = sorted(f"{k} {bk[k]}".strip() for k in set(bk) - set(lk))
            changed = sorted(f"{k}: {bk[k]} -> {lk[k]}"
                             for k in set(bk) & set(lk) if bk[k] != lk[k])
            if added or removed or changed or b.get("sha256") != lv.get("sha256"):
                diffs.append({"fact": fact, "kind": "list",
                              "baseline_sha256": b.get("sha256"),
                              "live_sha256": lv.get("sha256"),
                              "added": added, "removed": removed,
                              "changed": changed})
        elif b != lv:
            diffs.append({"fact": fact, "kind": "value",
                          "baseline": b, "live": lv})
    return diffs


def render_diff(node_id, d):
    if d["kind"] == "list":
        parts = [f"{node_id}: {d['fact']} differs "
                 f"(sha256 {d['baseline_sha256']} -> {d['live_sha256']})"]
        parts += [f"    added   {x}" for x in d["added"]]
        parts += [f"    removed {x}" for x in d["removed"]]
        parts += [f"    changed {x}" for x in d["changed"]]
        return parts
    return [f"{node_id}: {d['fact']} baseline={d['baseline']!r} live={d['live']!r}"]


# ---------------------------------------------------------------------------
# baseline I/O
# ---------------------------------------------------------------------------

def load_baseline(data):
    """Parse and structurally validate. Raises ValueError on anything malformed."""
    doc = json.loads(data.decode("utf-8"))
    if not isinstance(doc, dict) or doc.get("schema") != BASELINE_SCHEMA:
        raise ValueError(f"schema is not {BASELINE_SCHEMA!r}")
    hosts = doc.get("hosts")
    if not isinstance(hosts, dict) or not hosts:
        raise ValueError("no hosts")
    for nid, h in hosts.items():
        rec = h.get("record") if isinstance(h, dict) else None
        if not isinstance(rec, dict) or set(rec) != set(ALL_FACTS):
            raise ValueError(f"host {nid!r}: record does not hold exactly {ALL_FACTS}")
        for f in LIST_FACTS:
            lf = rec[f]
            if (not isinstance(lf, dict) or not isinstance(lf.get("entries"), list)
                    or _list_fact(lf["entries"])["sha256"] != lf.get("sha256")):
                raise ValueError(f"host {nid!r}: {f} sha256 does not match its entries")
    return doc


def evaluate(results, baseline_hosts):
    """Fail-close. (allowed, refusals). Unknown status falls through to refusal."""
    refusals = []
    live_ids = {r["node_id"] for r in results}
    for nid in sorted(set(baseline_hosts) - live_ids):
        refusals.append(f"{nid}: in the baseline but not in the execution set")
    for r in results:
        nid = r["node_id"]
        if nid not in baseline_hosts:
            refusals.append(f"{nid}: in the execution set but not in the baseline")
            continue
        if r["status"] == PROBE_UNAVAILABLE:
            refusals.append(f"{nid} ({r['endpoint']}): UNAVAILABLE "
                            f"(reason={r['reason']}) — state UNKNOWN, not clean")
            continue
        if r["status"] == PROBE_ERROR:
            refusals.append(f"{nid} ({r['endpoint']}): ERROR (reason={r['reason']})")
            continue
        if r["status"] != PROBE_OK:
            refusals.append(f"{nid}: unrecognized probe status {r['status']!r}")
            continue
        if r["hostname"] != r["expected_hostname"]:
            refusals.append(f"{nid} ({r['endpoint']}): answered as hostname "
                            f"{r['hostname']!r}, expected {r['expected_hostname']!r}")
        for d in compare_host_records(baseline_hosts[nid]["record"], r["record"]):
            refusals.extend(render_diff(nid, d))
    return (not refusals), refusals


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_results(results):
    for r in results:
        line = (f"  {r['node_id']:<11} {r['endpoint']:<16} {r['status']:<12}"
                f" hostname={r['hostname']} (expected {r['expected_hostname']})")
        if r["status"] != PROBE_OK:
            line += f" reason={r['reason']}"
        print(line)


def cmd_capture(args, repo_root, runner, targets_fn):
    print("ENV-FREEZE GATE — CAPTURE (owner only)")
    out = args.out
    if os.path.exists(out):
        print(f"ENV-FREEZE CAPTURE : REFUSED — {out} exists; a baseline is never "
              f"overwritten. Nothing written.")
        return EXIT_REFUSE
    try:
        targets, profile = targets_fn(args.rig_profile, repo_root)
    except Exception as exc:                                        # noqa: BLE001
        print(f"ENV-FREEZE CAPTURE : REFUSED — execution set unresolvable: "
              f"{type(exc).__name__}: {exc}. Nothing written.")
        return EXIT_REFUSE
    results = [probe_host(t, runner) for t in targets]
    _print_results(results)
    bad = [r for r in results if r["status"] != PROBE_OK
           or r["hostname"] != r["expected_hostname"]]
    if bad:
        print(f"ENV-FREEZE CAPTURE : REFUSED — {len(bad)}/{len(results)} host(s) "
              f"not measured cleanly ({', '.join(r['node_id'] for r in bad)}). "
              f"Nothing written.")
        return EXIT_REFUSE
    doc = {
        "schema": BASELINE_SCHEMA,
        "context": {"captured_at": _utcnow(), "rig_profile": profile,
                    "local_head": PG.local_head(repo_root, runner),
                    "note": "CONTEXT ONLY — never an input to the verdict"},
        "hosts": {r["node_id"]: {
            "context": {"endpoint": r["endpoint"], "hostname": r["hostname"],
                        "collected_at": r["collected_at"]},
            "record": r["record"]} for r in results},
    }
    try:
        with open(out, "x", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, sort_keys=True)
            fh.write("\n")
    except FileExistsError:
        print(f"ENV-FREEZE CAPTURE : REFUSED — {out} appeared during capture. "
              f"Nothing written.")
        return EXIT_REFUSE
    print(f"ENV-FREEZE CAPTURE : WROTE {out} ({len(results)} hosts). Commit it to "
          f"{BASELINE_REL} for verify to accept it.")
    return EXIT_PROCEED


def cmd_verify(args, repo_root, runner, targets_fn):
    word = "REFUSE" if args.phase == "pre" else "VIOLATION"
    print(f"ENV-FREEZE GATE — VERIFY --phase {args.phase}")
    print(f"local HEAD : {PG.local_head(repo_root, runner)}   [CONTEXT ONLY]")
    evidence = {"gate": "env_freeze_gate", "phase": args.phase,
                "generated_at": _utcnow(), "baseline": BASELINE_REL}

    def finish(allowed, reasons, results=None):
        evidence.update(verdict="PROCEED" if allowed else word, reasons=reasons,
                        results=[{k: v for k, v in r.items() if k != "record"}
                                 for r in (results or [])])
        if args.evidence_json:
            with open(args.evidence_json, "w", encoding="utf-8") as fh:
                json.dump(evidence, fh, indent=2, sort_keys=True)
                fh.write("\n")
        if allowed:
            print(f"ENV-FREEZE GATE : PROCEED — {len(results)}/{len(results)} hosts "
                  f"equal the baseline on every fact.")
            return EXIT_PROCEED
        print(f"ENV-FREEZE GATE : {word} — {len(reasons)} reason(s):")
        for line in reasons:
            print(f"  * {line}")
        return EXIT_REFUSE

    status, reason, data = tracked_unmodified(repo_root, BASELINE_REL, runner)
    if status != "OK":
        return finish(False, [f"baseline {status}: {reason}"])
    try:
        baseline = load_baseline(data)
    except Exception as exc:                                        # noqa: BLE001
        return finish(False, [f"baseline ERROR: unparseable or malformed: {exc}"])
    try:
        targets, _profile = targets_fn(args.rig_profile, repo_root)
    except Exception as exc:                                        # noqa: BLE001
        return finish(False, [f"execution set unresolvable: "
                              f"{type(exc).__name__}: {exc}"])
    results = [probe_host(t, runner) for t in targets]
    _print_results(results)
    allowed, refusals = evaluate(results, baseline["hosts"])
    return finish(allowed, refusals, results)


def main(argv=None, *, repo_root=REPO_ROOT, runner=None, targets_fn=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    owner_only = ("OWNER ONLY — run by Michael, never by an agent. Probes every host "
                  "in the execution set plus the local node and writes a NEW "
                  "baseline; refuses if any host is UNAVAILABLE or the file exists. "
                  f"Commit the result to {BASELINE_REL}.")
    c = sub.add_parser("capture", help=owner_only, description=owner_only)
    c.add_argument("--out", required=True)
    c.add_argument("--rig-profile", default=None)
    v = sub.add_parser("verify", help="compare live hosts to the committed baseline")
    v.add_argument("--phase", required=True, choices=("pre", "post"))
    v.add_argument("--evidence-json", default=None)
    v.add_argument("--rig-profile", default=None)
    args = ap.parse_args(argv)
    targets_fn = targets_fn or env_targets
    if args.cmd == "capture":
        return cmd_capture(args, repo_root, runner, targets_fn)
    return cmd_verify(args, repo_root, runner, targets_fn)


if __name__ == "__main__":
    sys.exit(main())
