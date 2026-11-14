#!/usr/bin/env python3
"""Gates for scripts/env_freeze_gate.py (G2) — E-A..E-N and three mutants.

Everything is built in a throwaway directory under data/delegated_run_gates/
(removed at the end), including a throwaway git repository for the
tracked-and-unmodified checks. No test contacts a rig: probe output comes through
the injectable runner, which passes only `git` through to the real binary.

Prints COMPLETION SENTINEL: PASS only when every gate ran and passed.
"""

import base64
import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import env_freeze_gate as LIVE                                      # noqa: E402

SRC = os.path.join(REPO, "scripts", "env_freeze_gate.py")
WORK_PARENT = os.path.join(REPO, "data", "delegated_run_gates")

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

RIGS = [("rrig6600", "192.168.3.122"), ("rrig6600b", "192.168.3.156"),
        ("rrig6600c", "192.168.3.164")]


def base_targets(extra_rig=False):
    t = [{"node_id": "localhost", "endpoint": "localhost", "ssh_user": "michael",
          "worker_hostname": "zeus-ubuntu-vm", "local": True,
          "python_env": "/home/michael/venvs/torch/bin/python"}]
    rigs = RIGS + ([("rrig6600d", "192.168.3.172")] if extra_rig else [])
    for nid, ep in rigs:
        t.append({"node_id": nid, "endpoint": ep, "ssh_user": "michael",
                  "worker_hostname": nid, "local": False,
                  "python_env": "/home/michael/rocm_env/bin/python"})
    return t


def facts_for(nid):
    if nid == "localhost":
        return {"KERNEL_RELEASE": "6.8.0-149-generic",
                "KERNEL_CMDLINE": "BOOT_IMAGE=/boot/vmlinuz-6.8.0-149-generic ro quiet",
                "PYTHON_VERSION": "3.10.12",
                "GPU_DRIVER_NVIDIA": "550.163.01", "GPU_DRIVER_AMDGPU": "ABSENT",
                "CUDA": "12.4.1", "ROCM_VERSION": "ABSENT",
                "pip": ["numpy==1.26.4", "torch==2.5.1+cu121", "cupy-cuda12x==13.5.1"],
                "dpkg": ["acl\tii 2.3.1-1", "bash\tii 5.1-6ubuntu1.1"]}
    return {"KERNEL_RELEASE": "6.8.12-9-pve",
            "KERNEL_CMDLINE": "BOOT_IMAGE=/boot/vmlinuz-6.8.12-9-pve ro quiet",
            "PYTHON_VERSION": "3.10.12" if nid == "rrig6600c" else "3.10.4",
            "GPU_DRIVER_NVIDIA": "ABSENT", "GPU_DRIVER_AMDGPU": "6.12.12",
            "CUDA": "ABSENT", "ROCM_VERSION": "6.4.3-128",
            "pip": ["numpy==1.26.4", "torch==2.3.0+rocm6.0", "cupy-rocm-6-0==13.0.0"],
            "dpkg": ["acl\tii 2.3.1-1", "bash\tii 5.1-6ubuntu1.1"]}


def all_facts():
    return {t["node_id"]: facts_for(t["node_id"]) for t in base_targets(True)}


def render(host, f, cut_before_end=False):
    cuda = ("ABSENT" if f["CUDA"] == "ABSENT" else
            base64.b64encode(json.dumps({"cuda": {"version": f["CUDA"]}}).encode()
                             ).decode())
    lines = [LIVE.BEGIN_SENTINEL, f"HOST\t{host}",
             f"KERNEL_RELEASE\t{f['KERNEL_RELEASE']}",
             f"KERNEL_CMDLINE\t{f['KERNEL_CMDLINE']}",
             f"PYTHON_VERSION\t{f['PYTHON_VERSION']}", LIVE.PIP_BEGIN, *f["pip"],
             LIVE.PIP_END, "PIP_STATUS\t0", LIVE.DPKG_BEGIN, *f["dpkg"],
             LIVE.DPKG_END, "DPKG_STATUS\t0",
             f"GPU_DRIVER_NVIDIA\t{f['GPU_DRIVER_NVIDIA']}",
             f"GPU_DRIVER_AMDGPU\t{f['GPU_DRIVER_AMDGPU']}",
             f"CUDA_JSON_B64\t{cuda}", f"ROCM_VERSION\t{f['ROCM_VERSION']}",
             LIVE.END_SENTINEL]
    if cut_before_end:
        lines = lines[:7]
    return "\n".join(lines) + "\n"


def make_runner(facts, timeout_nodes=(), cut_nodes=(), ssh_fail_nodes=()):
    by_ep = {t["endpoint"]: t["node_id"] for t in base_targets(True)}

    def runner(cmd, timeout):
        if cmd[0] == "git":
            return subprocess.run(cmd, capture_output=True, timeout=timeout)
        ep = "localhost" if cmd[0] == "bash" else cmd[-2].split("@", 1)[1]
        nid = by_ep[ep]
        if nid in timeout_nodes:
            raise subprocess.TimeoutExpired(cmd, timeout)
        if nid in ssh_fail_nodes:
            return subprocess.CompletedProcess(cmd, 255, b"", b"ssh: connect: No route")
        out = render(nid if nid != "localhost" else "zeus-ubuntu-vm", facts[nid],
                     cut_before_end=nid in cut_nodes)
        return subprocess.CompletedProcess(cmd, 0, out.encode(), b"")
    return runner


def git(repo, *args):
    p = subprocess.run(["git", "-C", repo, "-c", "user.name=test",
                        "-c", "user.email=test@example.invalid", *args],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"git {args} failed: {p.stderr}")
    return p.stdout


class Env:
    """A throwaway repo with an optional committed baseline captured from `facts`."""

    def __init__(self, G, root, commit=True, facts=None):
        self.G = G
        self.repo = tempfile.mkdtemp(prefix="efg_", dir=root)
        git(self.repo, "init", "-q")
        os.makedirs(os.path.join(self.repo, "config_manifests"))
        with open(os.path.join(self.repo, "README"), "w") as fh:
            fh.write("throwaway\n")
        git(self.repo, "add", "README")
        git(self.repo, "commit", "-q", "-m", "init")
        self.baseline = os.path.join(self.repo, LIVE.BASELINE_REL)
        self.facts = facts or all_facts()
        if commit is not None:
            rc, out = self.run(["capture", "--out", self.baseline])
            if rc != 0:
                raise RuntimeError(f"fixture capture failed:\n{out}")
            if commit:
                git(self.repo, "add", LIVE.BASELINE_REL)
                git(self.repo, "commit", "-q", "-m", "baseline")

    def run(self, argv, facts=None, extra_rig=False, **rk):
        buf = io.StringIO()
        runner = make_runner(facts or self.facts, **rk)
        with contextlib.redirect_stdout(buf):
            rc = self.G.main(argv, repo_root=self.repo, runner=runner,
                             targets_fn=lambda p, r: (base_targets(extra_rig), "proxmox"))
        return rc, buf.getvalue()

    def verify(self, **kw):
        return self.run(["verify", "--phase", "pre"], **kw)


# ---------------------------------------------------------------------------
# gates: each returns (ok, detail); `G` is the module under test
# ---------------------------------------------------------------------------

def changed(**per_node):
    f = all_facts()
    for nid, fn in per_node.items():
        fn(f[nid])
    return f


def g_EA(G, root):
    rc, out = Env(G, root).verify()
    return rc == 0 and "PROCEED" in out, out


def g_EB(G, root):
    e = Env(G, root)
    f = changed(rrig6600b=lambda x: x.update(KERNEL_RELEASE="6.8.12-10-pve"))
    rc, out = e.verify(facts=f)
    ok = (rc == 1 and "REFUSE" in out and "rrig6600b: kernel_release" in out
          and "6.8.12-9-pve" in out and "6.8.12-10-pve" in out)
    return ok, out


def g_EC(G, root):
    e = Env(G, root)
    f = changed(rrig6600=lambda x: x.update(
        pip=["numpy==2.0.0" if p.startswith("numpy") else p for p in x["pip"]]))
    rc, out = e.verify(facts=f)
    ok = rc == 1 and "rrig6600: pip_freeze differs" in out and \
        "changed numpy: 1.26.4 -> 2.0.0" in out
    return ok, out


def g_ED(G, root):
    e = Env(G, root)
    f = changed(rrig6600=lambda x: x["pip"].append("requests==2.32.3"),
                rrig6600c=lambda x: x["pip"].remove("cupy-rocm-6-0==13.0.0"))
    rc, out = e.verify(facts=f)
    ok = (rc == 1 and "added   requests 2.32.3" in out
          and "removed cupy-rocm-6-0 13.0.0" in out
          and "rrig6600: pip_freeze" in out and "rrig6600c: pip_freeze" in out)
    return ok, out


def g_EE(G, root):
    e = Env(G, root)
    f = changed(localhost=lambda x: x.update(GPU_DRIVER_NVIDIA="560.35.03"))
    rc, out = e.verify(facts=f)
    ok = rc == 1 and "localhost: gpu_driver_nvidia" in out and "560.35.03" in out
    return ok, out


def g_EF(G, root):
    rc, out = Env(G, root).verify(timeout_nodes=("rrig6600b",))
    ok = rc == 1 and "UNAVAILABLE (reason=timeout)" in out and "PROCEED" not in out
    return ok, out


def g_EG(G, root):
    rc, out = Env(G, root).verify(cut_nodes=("rrig6600c",))
    ok = rc == 1 and "UNAVAILABLE (reason=truncated_probe_output)" in out
    return ok, out


def g_EH(G, root):
    rc, out = Env(G, root, commit=None).verify()
    return rc == 1 and "baseline ABSENT" in out, out


def g_EI(G, root):
    rc, out = Env(G, root, commit=False).verify()
    return rc == 1 and "baseline UNTRACKED" in out, out


def g_EJ(G, root):
    e = Env(G, root)
    # Modify ONLY context, so the file still parses and still matches live: the
    # tracked-and-unmodified check is the only thing that can refuse it.
    doc = json.load(open(e.baseline))
    doc["context"]["captured_at"] = "1999-01-01T00:00:00Z"
    with open(e.baseline, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    rc, out = e.verify()
    return rc == 1 and "baseline MODIFIED" in out, out


def g_EK(G, root):
    e = Env(G, root)
    rc, out = e.verify(extra_rig=True)
    ok = rc == 1 and "rrig6600d: in the execution set but not in the baseline" in out
    return ok, out


def g_EL(G, root):
    e = Env(G, root, commit=None)
    rc, out = e.run(["capture", "--out", e.baseline], ssh_fail_nodes=("rrig6600",))
    ok = rc == 1 and not os.path.exists(e.baseline) and "Nothing written" in out
    return ok, out


def g_EM(G, root):
    e = Env(G, root)
    before = open(e.baseline, "rb").read()
    rc, out = e.run(["capture", "--out", e.baseline])
    ok = rc == 1 and open(e.baseline, "rb").read() == before and "exists" in out
    return ok, out


def g_EN(G, root):
    e = Env(G, root)
    doc = json.load(open(e.baseline))
    doc["context"]["captured_at"] = "1999-01-01T00:00:00Z"
    doc["context"]["local_head"] = "0" * 40
    for h in doc["hosts"].values():
        h["context"]["collected_at"] = "1999-01-01T00:00:00Z"
        h["context"]["endpoint"] = "10.0.0.1"
    with open(e.baseline, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    git(e.repo, "commit", "-q", "-am", "context-only change")
    rc, out = e.verify()
    return rc == 0 and "PROCEED" in out, out


# ---------------------------------------------------------------------------
# mutants — applied to a copy of live source; a mutant that fails to apply is an error
# ---------------------------------------------------------------------------

MUTANTS = {
    "E-MUT-1": ("comparison ignores pip_freeze",
                "    for fact in sorted(set(baseline) | set(live)):\n",
                "    for fact in sorted((set(baseline) | set(live)) - {'pip_freeze'}):\n",
                ["E-C"]),
    "E-MUT-2": ("UNAVAILABLE mapped to PROCEED",
                "        if r[\"status\"] == PROBE_UNAVAILABLE:\n            refusals.append(",
                "        if r[\"status\"] == PROBE_UNAVAILABLE:\n            continue\n"
                "            refusals.append(",
                ["E-F"]),
    "E-MUT-3": ("tracked-and-unmodified check removed",
                "    status, reason, data = tracked_unmodified(repo_root, BASELINE_REL, runner)\n",
                "    _p = os.path.join(repo_root, BASELINE_REL)\n"
                "    status, reason, data = (('OK', None, open(_p, 'rb').read())\n"
                "                            if os.path.isfile(_p) else ('ABSENT', 'absent', None))\n",
                ["E-I", "E-J"]),
}

GATES = [("E-A", g_EA), ("E-B", g_EB), ("E-C", g_EC), ("E-D", g_ED), ("E-E", g_EE),
         ("E-F", g_EF), ("E-G", g_EG), ("E-H", g_EH), ("E-I", g_EI), ("E-J", g_EJ),
         ("E-K", g_EK), ("E-L", g_EL), ("E-M", g_EM), ("E-N", g_EN)]


def load_mutant(name, old, new, root):
    src = open(SRC, encoding="utf-8").read()
    n = src.count(old)
    if n != 1:
        raise RuntimeError(f"{name}: mutation anchor occurs {n} times (need 1) — "
                           f"mutant failed to apply")
    path = os.path.join(root, f"env_freeze_gate_{name.replace('-', '_')}.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src.replace(old, new))
    spec = importlib.util.spec_from_file_location(f"efg_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    os.makedirs(WORK_PARENT, exist_ok=True)
    root = tempfile.mkdtemp(prefix="test_env_freeze_", dir=WORK_PARENT)
    results = []
    try:
        for name, fn in GATES:
            try:
                ok, out = fn(LIVE, root)
            except Exception as exc:                                # noqa: BLE001
                ok, out = False, f"EXCEPTION {type(exc).__name__}: {exc}"
            results.append((name, ok))
            print(f"[{'PASS' if ok else 'FAIL'}] {name}")
            if not ok:
                print("    " + out.replace("\n", "\n    "))
        gate_fns = dict(GATES)
        for mname, (desc, old, new, targets) in MUTANTS.items():
            try:
                mod = load_mutant(mname, old, new, root)
                reds = {}
                for t in targets:
                    ok, _out = gate_fns[t](mod, root)
                    reds[t] = not ok
                killed = all(reds.values())
                detail = ", ".join(f"{t} {'red' if r else 'STILL GREEN'}"
                                   for t, r in reds.items())
            except Exception as exc:                                # noqa: BLE001
                killed, detail = False, f"ERROR {type(exc).__name__}: {exc}"
            results.append((mname, killed))
            print(f"[{'PASS' if killed else 'FAIL'}] {mname} ({desc}): {detail}")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    expected = len(GATES) + len(MUTANTS)
    passed = sum(1 for _, ok in results if ok)
    print(f"\n{passed}/{expected} gates passed")
    if len(results) == expected and passed == expected:
        print("COMPLETION SENTINEL: PASS")
        return 0
    print("COMPLETION SENTINEL: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
