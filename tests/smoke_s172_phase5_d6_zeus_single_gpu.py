#!/usr/bin/env python3
"""
smoke_s172_phase5_d6_zeus_single_gpu.py — S172 Phase 5, Deliverable D6, gate
3.B: the Zeus single-GPU certified-generation smoke.

Spec: docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D6.md (REV1) §3.B.

    CUDA sieve (real 3080 Ti, java_lcg)
      -> sub-stripe result / spool
      -> coordinator publication
      -> Phase-5 assembly (serial_reference)
      -> 22-array validation
      -> certified generation (finalize_run)
      -> Step-2 loader successfully reads it back

ACCEPTANCE: a certified generation directory is produced, its 22-array bundle
passes validation, and the Step-2 loader reads it back — the first certified
accumulator generation on real silicon.

WHAT IS REAL HERE, AND WHAT IS HARNESS
  REAL: the GPU. `miner/range_miner_worker.py` runs as a SEPARATE PROCESS
        against the passed-through RTX 3080 Ti through cupy and the production
        `sieve_gpu_worker` kernels — no mock executor, no synthetic survivors.
  REAL: the coordinator. `run_bidirectional_test(use_range_miner=True)` is
        called exactly as Step 1 calls it, so the production `_use_miner` gate
        builds the Phase-5 sink, drives `serve_trial` over framed TCP, stages
        and verifies every sub-stripe spool, commits the trial, and D6 ingests
        the stored assembly.
  REAL: the finalizer. `finalize_run` is called with the SAME argument shape as
        the run-level Step-1 finalization (window_optimizer_integration_final.py
        :1812-1822).
  HARNESS: three things, each stated so no claim overreaches —
    1. the optimizer loop. This drives ONE trial directly instead of running
       `optimize_window`'s search; the trial itself is the production call.
    2. the bind address. Production binds 0.0.0.0:5700 for remote rigs; a
       single-GPU Zeus smoke binds 127.0.0.1 on an ephemeral port so it needs
       no fixed port and no second host.
    3. the repository identity — HARNESS ONLY IN THE DEFAULT (SCRATCH) MODE.
       `finalize_run` REFUSES a dirty tree (§7.3), and an agent sandbox may not
       commit. So by default the harness snapshots the exact source under test
       into a throwaway git repo (HEAD's tracked files with the D6 working-tree
       files overlaid), commits it there, and passes THAT root to the same
       `_repository_state` helper. The recorded SHA therefore identifies a tree
       byte-identical to the source that ran; it is not the project's own commit.

       `--release-grade` removes this harness leg entirely: no snapshot is taken
       and the provenance comes from the REAL repository at `_ROOT`. That is the
       mode Michael runs after committing, and the only mode whose generation is
       certified against the project's own commit.

REPOSITORY MODES (mutually exclusive; the banner names the active one)
  SCRATCH (default)      throwaway snapshot repo; scratch SHA; cleanliness of the
                         snapshot only. Pre-commit development mode.
  RELEASE-GRADE          real repository at `_ROOT`. Cleanliness policy is
                         TRACKED-CLEAN ONLY: `git status --porcelain
                         --untracked-files=no` must be empty or the run aborts.
                         Untracked files are PERMITTED — they are not part of the
                         committed source that produced the artifact — but every
                         untracked path is listed in the run output and written to
                         the evidence record `release_grade_repository_state.json`.
                         Note this is deliberately LOOSER than
                         `WOI._repository_state`, whose plain `--porcelain` counts
                         untracked; that helper's verdict is still reported, as
                         information, alongside the tracked-only one that governs.

Run:  python tests/smoke_s172_phase5_d6_zeus_single_gpu.py
      python tests/smoke_s172_phase5_d6_zeus_single_gpu.py --release-grade
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils import run_finalizer as RF                      # noqa: E402
from utils.canonical_arrays import validate_array_bundle   # noqa: E402
from utils.survivor_loader import load_survivors           # noqa: E402
from window_optimizer import WindowConfig                  # noqa: E402
import window_optimizer_integration_final as WOI           # noqa: E402

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"

# The frozen 22 array names, hand-transcribed (D3 / D3.5 / D4 / D5 all carry the
# same list) — never imported from the module that produces them.
ORACLE_ARRAY_NAMES = (
    "seeds", "forward_matches", "reverse_matches", "window_size", "offset",
    "trial_number", "skip_min", "skip_max", "skip_range", "forward_count",
    "reverse_count", "bidirectional_count", "intersection_count",
    "intersection_ratio", "intersection_weight", "bidirectional_selectivity",
    "forward_only_count", "reverse_only_count", "survivor_overlap_ratio",
    "score", "skip_mode", "prng_type",
)

DATASET = os.path.join(_ROOT, "daily3.json")
PRNG_BASE = "java_lcg"


class _Coordinator:
    """The attribute surface `run_bidirectional_test` reads off the coordinator.

    Every value is set EXPLICITLY — the gate must not silently inherit a
    getattr default and then report it as production configuration.
    """

    def __init__(self, *, port, staging_dir, seed_caps, stripe_size, substripes,
                 backend=None, backend_options=None, miner_host="127.0.0.1"):
        self.use_range_miner = True
        self.use_persistent_workers = False
        self.use_zmq_sqlite = False
        self.config_file = os.path.join(_ROOT, "distributed_config.json")
        self.worker_pool_size = 1
        self.seed_cap_nvidia = seed_caps
        self.seed_cap_amd = seed_caps
        self.seed_cap_nvidia_hybrid = seed_caps
        self.seed_cap_amd_hybrid = seed_caps
        self.miner_stripe_size = stripe_size
        self.miner_substripes = substripes
        self.miner_output_dir = staging_dir
        self.staging_dir = staging_dir
        self.staging_high_water_bytes = 16 * 1024 ** 3
        self.staging_high_water_files = 512
        self.compute_lease_timeout = 900.0
        self.staging_timeout = 900.0
        # [S172 Phase 6.0] Default "127.0.0.1" preserves the D6 3.B single-GPU
        # Zeus smoke exactly (see header). A REMOTE ROCm target passes 0.0.0.0 so
        # the rig's worker can dial in — the same value production uses
        # (window_optimizer_integration_final.py:1171).
        self.miner_host = miner_host
        self.miner_port = port
        self.node_allowlist = None
        self.serve_timeout = 1200.0
        # D6: no backend configured -> serial_reference
        self.assembly_backend = backend
        self.assembly_backend_options = backend_options


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _nvidia_smi(tag):
    print(f"\n----- nvidia-smi ({tag}) " + "-" * 40)
    out = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(out.stdout.strip())
    return out.stdout


# ===========================================================================
# [S172 Phase 6.0] REMOTE ROCm TARGET
#
# Phase 6.0 runs the SAME bounded trial twice: once on this VM's RTX 3080 Ti
# (the CUDA control, the unchanged D6 3.B path) and once on ONE RX 6600 XT in
# CT100 `rrig6600` (the ROCm subject). Acceptance is field-for-field equality of
# all 22 canonical arrays, so the two runs must differ in EXACTLY ONE variable:
# which silicon executed the sieve kernel.
#
# WHAT IS AND IS NOT REMOTE — this matters for what the comparison proves.
#   REMOTE: the GPU sieve only. `miner/range_miner_worker.py` runs on the rig
#           under ~/rocm_env against a real RX 6600 XT through cupy/HIP.
#   LOCAL : the coordinator, Phase-5 assembly, the finalizer and the NPZ writer
#           all still run HERE, on VM 101, for BOTH runs. So a difference in the
#           22 arrays can only come from the kernel — the writer is common mode
#           and cannot mask or manufacture a divergence.
#
# WHY NO TRANSFER ADAPTER IS NEEDED (verified, not assumed): the worker chooses
# inline-vs-spool by SIZE (range_miner_worker.py:1324, INLINE_BYTE_LIMIT =
# 48 MiB). WOI injects no TransferAdapter, so a spooled result would fail the
# stripe loudly at range_miner_coordinator.py:4014 rather than corrupt anything.
# The measured CUDA control shard maximum is ~1.83 MiB — 26x under the limit —
# so every sub-stripe result crosses the wire inline. `assert_no_spool_residue`
# re-checks the rig spool dir after the run instead of trusting that reasoning.
#
# WHY THE PARTITIONING MATCHES: the coordinator sizes sub-stripes with the cap
# the worker advertises (advertised_effective_cap -> select_seed_cap, which
# branches on backend: 'rocm' -> amd caps, 'cuda' -> nvidia caps,
# range_miner_worker.py:472-479). This harness advertises ALL FOUR caps equal,
# so the effective cap is identical on both platforms and the sub-stripe
# boundaries — hence canonical record ORDER — match by construction.
# ===========================================================================

_ROCM_FAULT_PATTERNS = (
    "GPU reset", "ring timeout", "L2 protection fault", "VM_L2", "VMC page fault",
    "vm fault", "amdgpu: ", "HSA_STATUS_ERROR", "hipError", "HIP error",
    "Memory access fault", "GPU hang", "soft recovery", "MES failed",
)


class _RemoteRocmTarget:
    """ONE RX 6600 XT in CT100 `rrig6600`, driven over SSH.

    Deliberately additive: the CUDA path never constructs one of these, and
    `run_smoke` keeps its original statements when `target is None`.
    """

    platform = "rocm"
    bind_host = "0.0.0.0"

    def __init__(self, *, host, user, python, repo, device, spool_dir,
                 coordinator_addr, work):
        self.host = host
        self.user = user
        self.python = python
        self.repo = repo
        self.device = int(device)
        self.spool_dir = spool_dir
        self.coordinator_addr = coordinator_addr
        self.work = Path(work)
        self.label = (f"ROCm / remote ({user}@{host} CT100 rrig6600, "
                      f"RX 6600 XT device {device})")
        self._proc = None

    # ----- ssh plumbing ---------------------------------------------------
    def _ssh_argv(self):
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                "-o", "ServerAliveInterval=15", f"{self.user}@{self.host}"]

    def sh(self, cmd, timeout=180):
        """Run a command on the rig; return stdout (stderr merged)."""
        r = subprocess.run(self._ssh_argv() + [cmd], capture_output=True,
                           text=True, timeout=timeout)
        return (r.stdout or "") + (r.stderr or "")

    # ----- identity + health (Beta-required, §4) --------------------------
    def identity(self):
        """Hardware/runtime identity, recorded verbatim into the evidence."""
        out = self.sh(
            'echo "## hostname"; hostname; '
            'echo "## kernel"; uname -r; '
            'echo "## amdgpu_module_version"; cat /sys/module/amdgpu/version; '
            'echo "## rocm_release"; cat /opt/rocm/.info/version; '
            'echo "## gpu_hw"; /opt/rocm/bin/rocm-smi -d %d --showhw; '
            'echo "## gpu_id"; /opt/rocm/bin/rocm-smi -d %d --showid; '
            'echo "## driver"; /opt/rocm/bin/rocm-smi --showdriverversion; '
            'echo "## env_overrides"; env | grep -iE "HSA|HIP|ROCR|GPU_|AMD_" '
            '|| echo "(none set)"'
            % (self.device, self.device))
        (self.work / "rocm_identity.txt").write_text(out)
        return out

    def backend_identity(self):
        """The worker's OWN backend determination, read from production code —
        NOT a harness assertion. range_miner_worker.py:1083 sets
        `backend = "rocm" if rt.is_hip else "cuda"`, and that value is what
        select_seed_cap() branches on."""
        out = self.sh(
            f'cd {self.repo} && {self.python} -c '
            '"import sys; sys.path.insert(0,\'.\'); '
            'import cupy; rt=cupy.cuda.runtime; '
            'print(\'is_hip\', rt.is_hip); '
            'print(\'backend_as_worker_computes_it\', '
            '\'rocm\' if getattr(rt,\'is_hip\',False) else \'cuda\'); '
            'print(\'runtimeGetVersion\', rt.runtimeGetVersion()); '
            'print(\'cupy\', cupy.__version__); '
            'p=rt.getDeviceProperties(%d); '
            'print(\'gcnArchName\', p.get(\'gcnArchName\')); '
            'print(\'device_name\', p.get(\'name\'))"' % self.device)
        (self.work / "rocm_backend_identity.txt").write_text(out)
        return out

    def health(self, tag):
        """Root-free hardware-fault surfaces, captured before AND after.

        NOTE (recorded as a limitation, not papered over): CT100 is an
        UNPRIVILEGED LXC. `dmesg` returns "read kernel buffer failed: Operation
        not permitted", /dev/kmsg is absent and `journalctl -k` is empty, so the
        in-container kernel ring buffer is NOT a usable signal — grepping it
        would return "no amdgpu errors" no matter what the GPU did, which is a
        vacuous pass. The amdgpu driver lives in the Proxmox HOST kernel. The
        surfaces below ARE readable without root and DO move when a 6600 XT
        resets, faults or falls off the bus."""
        out = self.sh(
            'echo "## dmesg_access"; (dmesg 2>&1 | tail -1) || true; '
            'echo "## kfd_topology_nodes"; ls /sys/class/kfd/kfd/topology/nodes/ | wc -l; '
            'echo "## dri_nodes"; ls /dev/dri/ | tr "\\n" " "; echo; '
            'echo "## pcie_replay"; /opt/rocm/bin/rocm-smi -d %d --showreplaycount; '
            'echo "## mem_use"; /opt/rocm/bin/rocm-smi -d %d --showmemuse; '
            'echo "## concise"; /opt/rocm/bin/rocm-smi -d %d --showhw'
            % (self.device, self.device, self.device))
        (self.work / f"rocm_health_{tag}.txt").write_text(out)
        return out

    def functional_probe(self):
        """A fresh kernel launch AFTER the run. If the GPU had reset or its
        context been lost, this fails — which is the specific failure class
        Phase 6.0 exists to rule out."""
        out = self.sh(
            f'cd {self.repo} && {self.python} -c '
            '"import sys; sys.path.insert(0,\'.\'); import cupy as cp; '
            'cp.cuda.Device(%d).use(); a=cp.arange(1000, dtype=cp.int64); '
            'print(\'post_run_kernel_sum\', int((a*2).sum())); '
            'print(\'expected\', 999*1000)"' % self.device)
        (self.work / "rocm_post_run_probe.txt").write_text(out)
        return out

    def probe(self, tag):
        print(f"\n----- rocm-smi ({tag}) " + "-" * 40)
        out = self.sh('/opt/rocm/bin/rocm-smi -d %d --showuse --showmemuse '
                      '--showtemp --showpower' % self.device)
        print(out.strip())
        return out

    # ----- lifecycle -------------------------------------------------------
    def prepare(self):
        self.sh(f'rm -rf {self.spool_dir} && mkdir -p {self.spool_dir}')

    def launch_worker(self, port, seed_caps, worker_log):
        """Same module, same flags, same cap set as the CUDA path — only the
        interpreter, the host and the coordinator address differ. `exec` lets
        SIGHUP on channel close reach python directly."""
        remote = (
            f"cd {self.repo} && exec {self.python} -m miner.range_miner_worker "
            f"--host {self.coordinator_addr} --port {port} "
            f"--gpu-id {self.device} --device-index {self.device} "
            f"--miner-output-dir {self.spool_dir} "
            f"--seed-cap-nvidia {seed_caps} --seed-cap-amd {seed_caps} "
            f"--seed-cap-nvidia-hybrid {seed_caps} "
            f"--seed-cap-amd-hybrid {seed_caps} "
            f"--heartbeat-interval 15")
        self._proc = subprocess.Popen(self._ssh_argv() + [remote],
                                      stdout=worker_log,
                                      stderr=subprocess.STDOUT)
        print(f"[WORKER] REMOTE ROCm worker via ssh pid={self._proc.pid} "
              f"-> {self.user}@{self.host} "
              f"({self.python} -m miner.range_miner_worker, "
              f"cupy/HIP on device {self.device})")
        print(f"[WORKER] coordinator address given to the rig: "
              f"{self.coordinator_addr}:{port}")
        return self._proc

    def cleanup(self, port):
        """Close the ssh channel, then PROVE no worker survives on the rig."""
        # `pkill -f` / `pgrep -f` match the FULL command line, and THIS cleanup
        # command necessarily mentions the worker module — so a naive pattern
        # makes pkill kill its own shell (producing NO cleanup evidence at all)
        # and makes pgrep count itself. Two defences, both needed:
        #   1. the bracket trick, so the regex text is not its own literal;
        #   2. drop the shell's own PID ($$) from the survivor list, because the
        #      surrounding message still contains the module name as plain text.
        left = self.sh(
            f'pkill -f "[r]ange_miner_worker.*--port {port}" >/dev/null 2>&1; '
            f'sleep 1; '
            f'survivors=$(pgrep -af "[r]ange_miner_worker" '
            f'| awk -v me=$$ "\\$1 != me"); '
            f'echo "surviving worker processes on the rig: '
            f'$(printf "%s" "$survivors" | grep -c . )"; '
            f'printf "%s" "$survivors"')
        print(f"[REMOTE CLEANUP] {left.strip()}")
        return left

    def assert_no_spool_residue(self):
        out = self.sh(f'find {self.spool_dir} -type f | wc -l; '
                      f'find {self.spool_dir} -type f | head -20')
        print(f"[REMOTE SPOOL] files left in {self.spool_dir}: {out.strip()}")
        return out


def _scan_worker_log_for_faults(log_path):
    """Scan the ROCm worker log for the failure class this rearchitecture
    exists to avoid. Absence must be EVIDENCED, so print what was searched."""
    text = Path(log_path).read_text(errors="replace") if Path(log_path).exists() else ""
    hits = [ln for ln in text.splitlines()
            if any(p.lower() in ln.lower() for p in _ROCM_FAULT_PATTERNS)]
    print("\n" + "-" * 78)
    print("ROCm WORKER LOG FAULT SCAN")
    print("-" * 78)
    print(f"  patterns searched : {len(_ROCM_FAULT_PATTERNS)} "
          f"({', '.join(_ROCM_FAULT_PATTERNS[:6])}, ...)")
    print(f"  log bytes         : {len(text):,}")
    print(f"  matching lines    : {len(hits)}")
    for ln in hits[:20]:
        print(f"        !! {ln}")
    if not hits:
        print("        (none)")
    return hits


def _source_snapshot_repo(dest: Path) -> str:
    """HEAD's tracked files with the working-tree D6 files overlaid, committed
    into a throwaway repo. Returns its commit SHA. See the header note 3."""
    dest.mkdir(parents=True, exist_ok=True)
    tar = subprocess.run(["git", "-C", _ROOT, "archive", "HEAD"],
                         check=True, capture_output=True)
    subprocess.run(["tar", "-x", "-C", str(dest)], input=tar.stdout, check=True)
    status = subprocess.run(["git", "-C", _ROOT, "status", "--porcelain"],
                            check=True, capture_output=True, text=True).stdout
    overlaid = []
    for line in status.splitlines():
        rel = line[3:].strip()
        if not rel.endswith(".py"):
            continue
        src = Path(_ROOT) / rel
        if not src.is_file():
            continue
        dst = dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        overlaid.append(rel)
    for cmd in (["git", "init", "-q"],
                ["git", "-c", "user.email=smoke@local", "-c", "user.name=smoke",
                 "add", "-A"],
                ["git", "-c", "user.email=smoke@local", "-c", "user.name=smoke",
                 "commit", "-q", "-m", "S172 D6 smoke source snapshot"]):
        subprocess.run(cmd, cwd=str(dest), check=True, capture_output=True)
    print(f"[SNAPSHOT] source snapshot repo: {dest}")
    print(f"[SNAPSHOT] working-tree .py files overlaid on HEAD: {overlaid}")
    return dest


def _git_lines(*args):
    out = subprocess.run(["git", "-C", _ROOT, *args],
                         check=True, capture_output=True, text=True).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def _release_grade_repository_state():
    """Provenance from the REAL repository, under the tracked-clean-only policy.

    Returns (commit, tracked_clean, tracked_dirty, untracked).

    The commit — and the untracked-inclusive verdict reported as information —
    come from the SAME production helper the run-level Step-1 finalization uses
    (`WOI._repository_state`, window_optimizer_integration_final.py:97), so the
    recorded SHA is the project's own commit and not a harness re-derivation.

    Pass/fail is TRACKED-ONLY (`--untracked-files=no`). Untracked files are
    permitted: they are, by definition, not part of the committed source that
    produced the artifact, so scratch material and unrelated briefs have no
    bearing on what this generation claims. They are not ignored either — every
    untracked path is listed here and in the evidence record. A dirty TRACKED
    tree aborts the run: it would mean the source that ran differs from the
    commit the artifact names.
    """
    commit, clean_including_untracked = WOI._repository_state(repo_root=_ROOT)
    tracked_dirty = _git_lines("status", "--porcelain", "--untracked-files=no")
    untracked = _git_lines("ls-files", "--others", "--exclude-standard")
    tracked_clean = not tracked_dirty

    print("\n" + "-" * 78)
    print("RELEASE-GRADE REPOSITORY PROVENANCE (real repository — no snapshot)")
    print("-" * 78)
    print(f"  repo_root                   : {_ROOT}")
    print(f"  repository_commit           : {commit}")
    print(f"  tracked_tree_clean (GOVERNS): {tracked_clean}   "
          f"[git status --porcelain --untracked-files=no]")
    print(f"  clean_including_untracked   : {clean_including_untracked}   "
          f"[WOI._repository_state, information only]")
    print(f"  POLICY: tracked-clean only. Untracked files are PERMITTED and "
          f"listed below;\n          a dirty TRACKED tree aborts the run.")
    # Tracked-dirty FIRST and under its own header: in an evidence record the two
    # lists must never be confusable — one is a hard failure, the other is waived.
    print(f"  TRACKED-DIRTY paths ({len(tracked_dirty)}) — these BLOCK the run:")
    for ln in tracked_dirty:
        print(f"          !! {ln}")
    if not tracked_dirty:
        print("          (none)")
    print(f"  UNTRACKED paths ({len(untracked)}) — permitted, recorded, waived:")
    for rel in untracked:
        print(f"          ?  {rel}")
    if not untracked:
        print("          (none)")

    if tracked_dirty:
        raise AssertionError(
            f"RELEASE-GRADE ABORT: the TRACKED working tree at {_ROOT} is dirty "
            f"({len(tracked_dirty)} path(s) listed above). A release-grade "
            f"generation must be certified against committed source, so it "
            f"cannot claim commit {commit} while tracked files differ from it. "
            f"Commit or revert them, then re-run. (Untracked files would NOT "
            f"have blocked this run.)")

    print(f"\n[RELEASE-GRADE] tracked tree clean at {commit} — certifying "
          f"against the project's own commit")
    return commit, tracked_clean, tracked_dirty, untracked


def _write_evidence_record(work, artifact, commit, tracked_clean, untracked):
    """The durable release-grade evidence record.

    `finalize_run`'s sidecar carries the commit and the clean flag, but its
    signature is frozen and cannot carry the untracked list — so the list that
    the tracked-only policy explicitly tolerates is recorded HERE, next to the
    run, where an auditor can see exactly what was present and waived.
    """
    record = {
        "smoke": "s172_phase5_d6_zeus_single_gpu",
        "repository_mode": "release-grade",
        "repo_root": _ROOT,
        "repository_commit": commit,
        "cleanliness_policy": "tracked-clean-only (--untracked-files=no)",
        "tracked_tree_clean": tracked_clean,
        "untracked_permitted": True,
        "untracked_count": len(untracked),
        "untracked_paths": list(untracked),
        "generation_id": artifact.generation_id,
        "generation_dir": str(artifact.generation_dir),
        "artifact_sha256": artifact.artifact_sha256,
        "sidecar_sha256": artifact.sidecar_sha256,
        "final_row_count": int(artifact.final_row_count),
    }
    path = Path(work) / "release_grade_repository_state.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"\n[EVIDENCE] release-grade repository state written: {path}")
    print(f"[EVIDENCE] untracked paths recorded: {len(untracked)}")
    return path


def run_smoke(seed_start, seed_count, stripe_size, seed_caps, window_size,
              forward_threshold, reverse_threshold, workdir, target=None):
    """`target=None` is the DEFAULT CUDA path — every statement it executes is
    the original D6 3.B code, unchanged. A `_RemoteRocmTarget` swaps only the
    bind address, the worker launch and the GPU probe."""
    work = Path(workdir)
    staging = work / "miner_output"
    staging.mkdir(parents=True, exist_ok=True)
    gen_root = work / "generation_root"
    gen_root.mkdir(parents=True, exist_ok=True)

    config = WindowConfig(
        window_size=window_size, offset=0,
        sessions=["midday", "evening"],
        # skip_min/skip_max are trial METADATA here: the constant-skip java_lcg
        # kernel takes its skip range from the assignment payload, whose
        # coordinator-built default is [0, 16]. Recording that same range keeps
        # the manifest honest about what actually ran.
        skip_min=0, skip_max=16,
        # [S172 D6 correction] ASYMMETRIC by default (0.31 / 0.47). Re-running
        # the pre-correction 0.25/0.25 would prove nothing: 0.25 is exactly the
        # value the broken path fell back to, so a symmetric run at 0.25 passes
        # whether or not the threshold ever reached the kernel.
        forward_threshold=forward_threshold,
        reverse_threshold=reverse_threshold)

    port = _free_port()
    coordinator = _Coordinator(port=port, staging_dir=str(staging),
                               seed_caps=seed_caps, stripe_size=stripe_size,
                               substripes=max(1, stripe_size // seed_caps),
                               miner_host=("127.0.0.1" if target is None
                                           else target.bind_host))

    accumulator = {"forward_count": 0, "reverse_count": 0, "bidirectional": []}
    holder = {}

    def _trial():
        try:
            t0 = time.time()
            holder["result"] = WOI.run_bidirectional_test(
                coordinator, config, DATASET, seed_start, seed_count,
                prng_base=PRNG_BASE, test_both_modes=False,
                forward_threshold=forward_threshold,
                reverse_threshold=reverse_threshold,
                trial_number=1, accumulator=accumulator)
            holder["elapsed"] = time.time() - t0
        except Exception:
            holder["err"] = traceback.format_exc()

    print(f"\n[TRIAL] coordinator binding {coordinator.miner_host}:{port}")
    print(f"[TRIAL] seeds [{seed_start:,}, {seed_start + seed_count:,}) "
          f"stripe={stripe_size:,} substripe_cap={seed_caps:,} "
          f"window={window_size} "
          f"forward_threshold={forward_threshold} "
          f"reverse_threshold={reverse_threshold}")
    t = threading.Thread(target=_trial, name="d6-smoke-trial", daemon=True)
    t.start()
    time.sleep(2.0)     # let the serve loop bind before the worker dials in

    worker_log = open(work / "worker.log", "w")
    if target is not None:
        # [S172 Phase 6.0] REMOTE ROCm subject. The CUDA branch below is
        # untouched; this branch is never taken on the default path.
        target.prepare()
        worker = target.launch_worker(port, seed_caps, worker_log)
    else:
      worker = subprocess.Popen(
        [sys.executable, "-m", "miner.range_miner_worker",
         "--host", "127.0.0.1", "--port", str(port),
         "--gpu-id", "0", "--device-index", "0",
         "--miner-output-dir", str(staging / "spool"),
         # ALL FOUR caps, not just the nvidia one: the coordinator quarantines a
         # worker whose ADVERTISED seed_caps differ from the central config in
         # ANY family ("registered but quarantined: seed_cap 'amd'=... != central
         # config ..."), and a quarantined sole worker leaves the trial with no
         # eligible worker, so it aborts. In production both sides come from the
         # same resolved §12.4 configuration; a hand-driven smoke must supply
         # them consistently.
         "--seed-cap-nvidia", str(seed_caps),
         "--seed-cap-amd", str(seed_caps),
         "--seed-cap-nvidia-hybrid", str(seed_caps),
         "--seed-cap-amd-hybrid", str(seed_caps),
         "--heartbeat-interval", "15"],
        cwd=_ROOT, stdout=worker_log, stderr=subprocess.STDOUT)
      print(f"[WORKER] real GPU worker pid={worker.pid} "
            f"(miner/range_miner_worker.py, cupy on device 0)")

    smi_during = None
    for _ in range(60):
        time.sleep(2.0)
        if smi_during is None and worker.poll() is None and t.is_alive():
            smi_during = (_nvidia_smi("during the CUDA sieve") if target is None
                          else target.probe("during the ROCm sieve"))
        if not t.is_alive():
            break
    t.join(timeout=1200)

    # [S172 Phase 6.0] §4 "clean worker exit" — capture the state that actually
    # carries the evidence, BEFORE we tear anything down. A worker that is still
    # alive here ran the whole trial without crashing; that, not the post-kill
    # code, is the meaningful signal. (After terminate() the remote worker's
    # code is the SSH CLIENT's — 255 on channel teardown — not the worker's own,
    # so reporting it alone would be misleading.)
    alive_before_terminate = (worker.poll() is None)

    try:
        worker.terminate()
        worker.wait(timeout=30)
    except Exception:
        worker.kill()
    worker_log.close()
    if target is not None:
        # ROCm evidence only, so the CUDA path's output stays identical to D6 3.B.
        print(f"[WORKER] alive for the whole trial, no premature exit: "
              f"{alive_before_terminate}")
        print(f"[WORKER] post-terminate code {worker.poll()} "
              f"(SSH CLIENT teardown code, not the worker's own exit status)")
        target.cleanup(port)

    if t.is_alive():
        raise AssertionError("the miner trial did not terminate")
    if "err" in holder:
        print(f"\n[WORKER LOG]\n{(work / 'worker.log').read_text()[-4000:]}")
        raise AssertionError(f"the miner trial failed:\n{holder['err']}")

    tr = holder["result"]
    print(f"\n[TRIAL] returned in {holder['elapsed']:.1f}s: "
          f"forward={tr.forward_count:,} reverse={tr.reverse_count:,} "
          f"bidirectional={tr.bidirectional_count:,}")
    print(f"[ACCUM] forward_count={accumulator['forward_count']:,} "
          f"reverse_count={accumulator['reverse_count']:,} "
          f"raw candidates={len(accumulator['bidirectional']):,}")

    # --- [S172 D6 correction] the three-leg threshold provenance ------------
    # requested / payload / effective, read off the audit record the trial
    # itself wrote next to its staged output. `effective` came back from the
    # real worker off the real executor — it is NOT recomputed here from the
    # config, which would only prove the config agrees with itself.
    prov = _report_threshold_provenance(staging, forward_threshold,
                                        reverse_threshold)

    return tr, accumulator, gen_root, smi_during, work, prov


def _report_threshold_provenance(staging, forward_threshold, reverse_threshold):
    """Print and VERIFY requested == payload == effective, per direction."""
    prov_path = Path(staging) / "threshold_provenance.json"
    print("\n" + "-" * 78)
    print("THRESHOLD PROVENANCE (S172 D6 correction — Beta §3)")
    print("-" * 78)
    assert prov_path.exists(), (
        f"no threshold provenance record at {prov_path} — the trial cannot "
        f"show what the kernel actually filtered at")
    prov = json.loads(prov_path.read_text())

    # [S172 D6, Beta commit ruling] The parent's fail-closed gate must have RUN
    # and PASSED for this trial. `validated` is set only by that gate, immediately
    # before commit_trial — so True here is evidence the four provenance
    # conditions were enforced on real silicon, not merely recorded.
    assert prov.get("validated") is True, (
        f"threshold provenance is not validated ({prov.get('validated')!r}) — the "
        f"parent-side fail-closed gate did not pass, so this generation must not "
        f"be treated as certified")

    requested = prov.get("requested", {})
    print(f"  requested : forward={requested.get('forward')} "
          f"reverse={requested.get('reverse')}")
    print(f"  payload   : {prov.get('payload')}")
    print(f"  effective : {prov.get('effective')}")
    print(f"  phase->dir: {prov.get('phase_direction')}")

    assert requested.get("forward") == forward_threshold, (
        f"requested forward leg {requested.get('forward')!r} != "
        f"{forward_threshold}")
    assert requested.get("reverse") == reverse_threshold, (
        f"requested reverse leg {requested.get('reverse')!r} != "
        f"{reverse_threshold}")
    assert forward_threshold != reverse_threshold, (
        "the smoke MUST run asymmetric thresholds — a symmetric run cannot "
        "distinguish a working threshold path from the 0.25 fallback")

    # test_both_modes=False -> phases 1 (forward/constant) and 2 (reverse).
    expected = {"1": forward_threshold, "2": reverse_threshold}
    for phase, want in expected.items():
        got_payload = prov.get("payload", {}).get(phase)
        got_effective = prov.get("effective", {}).get(phase)
        assert got_payload == [want], (
            f"phase {phase}: payload leg {got_payload!r}, expected [{want}]")
        assert got_effective == [want], (
            f"phase {phase}: EFFECTIVE leg {got_effective!r}, expected [{want}] "
            f"— the kernel did not filter at the transmitted value")
    print(f"\n[PROVENANCE] requested == payload == effective for BOTH "
          f"directions (forward={forward_threshold}, "
          f"reverse={reverse_threshold})")
    return prov


def finalize_and_verify(accumulator, gen_root, seed_start, seed_count, work,
                        release_grade=False):
    # --- repository identity (see header note 3 / REPOSITORY MODES) --------
    untracked = None
    if release_grade:
        # No snapshot at all: the provenance IS the real repository's.
        commit, clean, _dirty, untracked = _release_grade_repository_state()
    else:
        snap = _source_snapshot_repo(work / "source_snapshot")
        commit, clean = WOI._repository_state(repo_root=str(snap))
        print(f"[SNAPSHOT] commit={commit} tree_clean={clean}")
        assert clean, "the source snapshot repo is not clean"

    # --- the SAME call shape as the run-level Step-1 finalization ---------
    artifact = RF.finalize_run(
        accumulator["bidirectional"],
        output_root=gen_root,
        run_id=f"step1_{PRNG_BASE}_{int(seed_start)}",
        prng_base=PRNG_BASE,
        skip_modes_executed=("constant",),   # test_both_modes=False
        seed_start=int(seed_start),
        seed_count=int(seed_count),
        repository_commit=commit,
        repository_tree_clean=clean,
    )

    _mode_tag = ("RELEASE-GRADE (real repository commit)" if release_grade
                 else "SCRATCH SHA (throwaway snapshot repo)")
    print("\n" + "=" * 78)
    print(f"CERTIFIED GENERATION  [{_mode_tag}]")
    print("=" * 78)
    print(f"  repository_mode   : "
          f"{'release-grade' if release_grade else 'scratch'}")
    print(f"  generation_id     : {artifact.generation_id}")
    print(f"  generation_dir    : {artifact.generation_dir}")
    print(f"  binary_npz_path   : {artifact.binary_npz_path}")
    print(f"  all_npz_path      : {artifact.all_npz_path}")
    print(f"  sidecar_path      : {artifact.sidecar_path}")
    print(f"  artifact_sha256   : {artifact.artifact_sha256}")
    print(f"  sidecar_sha256    : {artifact.sidecar_sha256}")
    print(f"  raw_candidates    : {artifact.raw_candidate_count:,}")
    print(f"  l2_winners        : {artifact.l2_winner_count:,}")
    print(f"  prior_rows        : {artifact.prior_row_count:,}")
    print(f"  final_rows        : {artifact.final_row_count:,}")
    print(f"  repository_commit : {artifact.repository_commit}")
    print(f"  tree_clean        : {artifact.repository_tree_clean}")
    if release_grade:
        print(f"  untracked (waived): {len(untracked)} path(s) — "
              f"tracked-clean-only policy")
        for rel in untracked:
            print(f"                      ? {rel}")
        _write_evidence_record(work, artifact, commit, clean, untracked)

    # --- D6 fail-closed path check ---------------------------------------
    from miner.step1_ingress import certified_paths
    paths = certified_paths(artifact)
    for name, p in paths.items():
        assert os.path.exists(p), f"{name} missing on disk: {p}"
    print(f"\n[PATHS] all {len(paths)} certified paths exist on disk")

    # --- 22-array validation ---------------------------------------------
    with np.load(artifact.binary_npz_path) as npz:
        names = tuple(npz.files)
        bundle = {k: npz[k] for k in names}
    assert names == ORACLE_ARRAY_NAMES, f"array names/order drifted: {names}"
    validate_array_bundle(bundle)
    print(f"[22-ARRAY] {len(names)} arrays, order matches the frozen oracle, "
          f"validate_array_bundle() passed")
    print(f"[22-ARRAY] rows={len(bundle['seeds']):,}")

    # --- sidecar ----------------------------------------------------------
    sidecar = json.loads(Path(artifact.sidecar_path).read_text())
    print(f"[SIDECAR] keys={len(sidecar)} "
          f"schema={sidecar.get('sidecar_schema_version')} "
          f"encoding={sidecar.get('encoding_contract_version')}")
    for k in ("run_id", "prng_base", "seed_start", "seed_count",
              "final_row_count", "artifact_sha256", "repository_commit",
              "repository_tree_clean", "canonical_map_hash"):
        print(f"          {k}: {sidecar.get(k)}")

    # --- Step-2 loader reads it back --------------------------------------
    loaded = load_survivors(str(artifact.binary_npz_path))
    print(f"\n[STEP-2 LOADER] utils.survivor_loader.load_survivors "
          f"format={loaded.format} npz_version={loaded.npz_version} "
          f"count={loaded.count:,} fallback_used={loaded.fallback_used}")
    assert loaded.format == "npz", loaded.format
    assert loaded.fallback_used is False, "the Step-2 loader fell back to JSON"
    assert loaded.count == artifact.final_row_count, (
        loaded.count, artifact.final_row_count)
    return artifact, bundle, loaded


def compare_generations(cuda_npz, rocm_npz):
    """[S172 Phase 6.0] Beta's required addition: schema-valid ROCm output does
    NOT establish computational parity, because a platform-specific kernel
    defect can still produce a structurally valid generation. So compare
    field-for-field with np.array_equal across ALL 22 canonical arrays and
    report a 22-ROW MATRIX, never a summary boolean.

    A divergence is a finding to LOCALIZE: report which array, the first
    differing index, and both values."""
    with np.load(cuda_npz) as a:
        cuda = {k: a[k] for k in a.files}
        cuda_order = tuple(a.files)
    with np.load(rocm_npz) as b:
        rocm = {k: b[k] for k in b.files}
        rocm_order = tuple(b.files)

    print("=" * 78)
    print("S172 Phase 6.0 — CUDA vs ROCm 22-ARRAY EQUALITY MATRIX")
    print("=" * 78)
    print(f"  CUDA npz : {cuda_npz}")
    print(f"  ROCm npz : {rocm_npz}")
    print(f"\n  canonical ORDER identical to frozen oracle (CUDA): "
          f"{cuda_order == ORACLE_ARRAY_NAMES}")
    print(f"  canonical ORDER identical to frozen oracle (ROCm): "
          f"{rocm_order == ORACLE_ARRAY_NAMES}")
    print(f"  canonical ORDER identical CUDA vs ROCm           : "
          f"{cuda_order == rocm_order}")

    print("\n" + "-" * 78)
    print(f"  {'#':>2}  {'array':<28} {'dtype':<10} {'rows':>6}  "
          f"{'array_equal':<11} note")
    print("-" * 78)
    all_equal = True
    divergences = []
    for i, name in enumerate(ORACLE_ARRAY_NAMES, 1):
        if name not in cuda or name not in rocm:
            all_equal = False
            print(f"  {i:>2}  {name:<28} {'-':<10} {'-':>6}  "
                  f"{'MISSING':<11} present CUDA={name in cuda} "
                  f"ROCm={name in rocm}")
            divergences.append((name, "missing", None, None))
            continue
        ca, ra = cuda[name], rocm[name]
        eq = bool(np.array_equal(ca, ra))
        note = ""
        if not eq:
            all_equal = False
            if ca.shape != ra.shape:
                note = f"SHAPE {ca.shape} vs {ra.shape}"
                divergences.append((name, "shape", ca.shape, ra.shape))
            else:
                diff = np.flatnonzero(ca != ra)
                j = int(diff[0])
                note = f"first differing index {j}: CUDA={ca[j]!r} ROCm={ra[j]!r} " \
                       f"({diff.size} of {ca.size} differ)"
                divergences.append((name, j, ca[j], ra[j]))
        print(f"  {i:>2}  {name:<28} {str(ca.dtype):<10} {ca.shape[0]:>6}  "
              f"{str(eq):<11} {note}")
    print("-" * 78)

    for label, arrs in (("CUDA", cuda), ("ROCm", rocm)):
        fwd = int(arrs['forward_count'][0]) if arrs['forward_count'].size else 0
        rev = int(arrs['reverse_count'][0]) if arrs['reverse_count'].size else 0
        bi = int(arrs['bidirectional_count'][0]) if arrs['bidirectional_count'].size else 0
        print(f"  {label:<5} forward={fwd:,} reverse={rev:,} "
              f"bidirectional={bi:,} rows={arrs['seeds'].shape[0]:,}")

    print("\n" + "=" * 78)
    if all_equal and cuda_order == rocm_order == ORACLE_ARRAY_NAMES:
        print(f"[{_PASS}] ALL 22 CANONICAL ARRAYS ARE FIELD-FOR-FIELD EQUAL "
              f"across CUDA and ROCm,\n        and the canonical record ORDER "
              f"is identical on both platforms.")
        print("=" * 78)
        return 0
    print(f"[{_FAIL}] CROSS-PLATFORM DIVERGENCE in "
          f"{len(divergences)} array(s): "
          f"{', '.join(d[0] for d in divergences)}")
    print("=" * 78)
    return 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--seed-count", type=int, default=8_000_000)
    ap.add_argument("--stripe-size", type=int, default=4_000_000)
    ap.add_argument("--seed-cap", type=int, default=1_000_000)
    ap.add_argument("--window-size", type=int, default=3)
    # [S172 D6 correction] asymmetric by default — see run_smoke's note. The old
    # single `--threshold 0.25` could not tell a working path from the fallback.
    ap.add_argument("--forward-threshold", type=float, default=0.31)
    ap.add_argument("--reverse-threshold", type=float, default=0.47)
    ap.add_argument("--workdir", default=None)
    ap.add_argument(
        "--release-grade", action="store_true",
        help=(
            "Certify the generation against the REAL repository commit instead "
            "of a throwaway snapshot repo. Skips the source snapshot entirely "
            "and takes repository_commit / repository_tree_clean from "
            "WOI._repository_state(repo_root=<repo>). CLEANLINESS POLICY: "
            "TRACKED-CLEAN ONLY — the pass/fail check is `git status "
            "--porcelain --untracked-files=no`, and a dirty TRACKED tree aborts "
            "the run loudly. UNTRACKED files are PERMITTED (they are not part "
            "of the committed source that produced the artifact), but every "
            "untracked path is listed in the run output and written to "
            "release_grade_repository_state.json in the workdir. Default (off) "
            "is the pre-commit scratch-SHA mode, unchanged."))
    # --- [S172 Phase 6.0] remote ROCm target (default OFF = unchanged CUDA) ---
    ap.add_argument(
        "--rocm-remote", action="store_true",
        help=("Run the sieve on ONE RX 6600 XT in CT100 `rrig6600` instead of "
              "this VM's RTX 3080 Ti. The coordinator, Phase-5 assembly, "
              "finalizer and NPZ writer still run HERE, so the only variable "
              "is which silicon executed the kernel. Default OFF: the CUDA "
              "path is exactly the D6 3.B path."))
    ap.add_argument("--rocm-host", default="192.168.3.122")
    ap.add_argument("--rocm-user", default="michael")
    ap.add_argument("--rocm-python", default="/home/michael/rocm_env/bin/python")
    ap.add_argument("--rocm-repo", default="/home/michael/distributed_prng_analysis")
    ap.add_argument("--rocm-device", type=int, default=0)
    ap.add_argument("--rocm-spool-dir",
                    default="/home/michael/s172_phase60_spool")
    ap.add_argument("--coordinator-addr", default="192.168.3.177",
                    help="address the RIG dials back to (this VM on the LAN)")
    # --- [S172 Phase 6.0] 22-array cross-platform comparator -----------------
    ap.add_argument(
        "--compare", nargs=2, metavar=("CUDA_NPZ", "ROCM_NPZ"), default=None,
        help=("Compare two certified generations field-for-field with "
              "np.array_equal across all 22 canonical arrays and print the "
              "22-row matrix. Runs no GPU work."))
    args = ap.parse_args(argv)

    if args.compare:
        return compare_generations(args.compare[0], args.compare[1])

    mode = "RELEASE-GRADE" if args.release_grade else "SCRATCH"
    print("=" * 78)
    print("S172 Phase 5 D6 — 3.B Zeus single-GPU certified-generation smoke")
    print(f"REPOSITORY MODE: {mode}")
    if args.release_grade:
        print("  provenance : the REAL repository at "
              f"{_ROOT} (no snapshot taken)")
        print("  policy     : TRACKED-CLEAN ONLY — pass/fail is `git status "
              "--porcelain")
        print("               --untracked-files=no`; a dirty TRACKED tree "
              "aborts the run.")
        print("               UNTRACKED files are PERMITTED and every path is "
              "listed in this")
        print("               output and in the evidence record "
              "release_grade_repository_state.json.")
        print("  meaning    : this generation is certified against the "
              "project's own commit.")
    else:
        print("  provenance : a THROWAWAY SNAPSHOT REPO — the recorded SHA is "
              "a SCRATCH SHA,")
        print("               not this project's commit. NOT release-grade; "
              "re-run with")
        print("               --release-grade from a tracked-clean repository "
              "for that.")
    print("=" * 78)
    # CUDA path: this call keeps its original position, so the default path's
    # output ordering is identical to D6 3.B.
    if not args.rocm_remote:
        _nvidia_smi("before the run")

    workdir = args.workdir or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "d6_zeus_smoke")
    os.makedirs(workdir, exist_ok=True)
    print(f"\n[WORKDIR] {workdir}")

    # --- [S172 Phase 6.0] target selection -------------------------------
    target = None
    if args.rocm_remote:
        target = _RemoteRocmTarget(
            host=args.rocm_host, user=args.rocm_user, python=args.rocm_python,
            repo=args.rocm_repo, device=args.rocm_device,
            spool_dir=args.rocm_spool_dir,
            coordinator_addr=args.coordinator_addr, work=workdir)
        print("=" * 78)
        print("EXECUTION TARGET: ROCm / REMOTE  — S172 Phase 6.0")
        print("=" * 78)
        print(f"  {target.label}")
        print("  ARTIFACT CLASSIFICATION: ROCm platform-validation certified")
        print("                           generation — NON-AUTHORITATIVE.")
        print("                           The D6 CUDA generation at b08c2c5 "
              "remains the")
        print("                           authoritative release-grade artifact.")
        print("=" * 78)
        # CT-side source identity — proves identical source BY CONSTRUCTION.
        ct = target.sh(f'cd {args.rocm_repo} && '
                       f'echo "commit=$(git rev-parse HEAD)"; '
                       f'echo "tracked_dirty=[$(git status --porcelain '
                       f'--untracked-files=no)]"; '
                       f'echo "untracked=[$(git ls-files --others '
                       f'--exclude-standard | tr "\\n" " ")]"; '
                       f'echo "dataset_sha256=$(sha256sum daily3.json)"')
        print("\n----- CT100 DEPLOYED SOURCE IDENTITY " + "-" * 30)
        print(ct.strip())
        Path(workdir, "ct_source_identity.txt").write_text(ct)
        print("\n----- ROCm HARDWARE IDENTITY " + "-" * 38)
        print(target.identity().strip())
        print("\n----- ROCm BACKEND IDENTITY (from production code) " + "-" * 16)
        print(target.backend_identity().strip())
        print("\n----- ROCm HEALTH: BEFORE " + "-" * 41)
        print(target.health("before").strip())

    tr, acc, gen_root, _smi, work, prov = run_smoke(
        args.seed_start, args.seed_count, args.stripe_size, args.seed_cap,
        args.window_size, args.forward_threshold, args.reverse_threshold,
        workdir, target=target)

    print("\n" + "-" * 78)
    print("SURVIVOR COUNTS BY DIRECTION (asymmetric thresholds)")
    print("-" * 78)
    print(f"  forward  (phase 1, threshold {args.forward_threshold}) : "
          f"{acc['forward_count']:,}")
    print(f"  reverse  (phase 2, threshold {args.reverse_threshold}) : "
          f"{acc['reverse_count']:,}")
    print(f"  bidirectional (intersection)                : "
          f"{tr.bidirectional_count:,}")

    if not acc["bidirectional"]:
        print("\n[NOTE] the trial produced ZERO bidirectional candidates. The "
              "generation below is a real, certified EMPTY generation — "
              "honest, but a weaker smoke. Re-run with a smaller --window-size "
              "or a larger --seed-count for a populated one.")

    artifact, bundle, loaded = finalize_and_verify(
        acc, gen_root, args.seed_start, args.seed_count, work,
        release_grade=args.release_grade)

    if target is None:
        _nvidia_smi("after the run")
    else:
        # --- §4 ROCm health evidence: absence must be EVIDENCED -----------
        print("\n----- ROCm HEALTH: AFTER " + "-" * 42)
        after = target.health("after")
        print(after.strip())
        before = (Path(workdir) / "rocm_health_before.txt").read_text()
        print("\n" + "-" * 78)
        print("ROCm HEALTH BEFORE/AFTER DIFF (§4 — the failure class this")
        print("rearchitecture exists to avoid: GPU reset / L2 protection fault /")
        print("VM fault / new amdgpu error)")
        print("-" * 78)
        import difflib
        delta = list(difflib.unified_diff(
            before.splitlines(), after.splitlines(),
            fromfile="health_before", tofile="health_after", lineterm="", n=0))
        if delta:
            for ln in delta:
                print(f"  {ln}")
        else:
            print("  (no difference — identical before and after)")
        print("\n----- ROCm POST-RUN FUNCTIONAL PROBE " + "-" * 30)
        print(target.functional_probe().strip())
        _scan_worker_log_for_faults(Path(workdir) / "worker.log")
        target.assert_no_spool_residue()

    print("\n" + "=" * 78)
    print(f"REPOSITORY MODE: {mode}  (commit {artifact.repository_commit})")
    print(f"[{_PASS}] 3.B ACCEPTANCE — certified generation produced on real "
          f"silicon,\n        22-array bundle validated, Step-2 loader read it "
          f"back ({loaded.count:,} rows),\n        and the asymmetric "
          f"forward={args.forward_threshold} / reverse={args.reverse_threshold} "
          f"thresholds\n        reached the "
          f"{'ROCm' if target is not None else 'CUDA'} kernel unchanged "
          f"(requested == payload == effective).")
    print(f"\n[NPZ FOR PARITY COMPARISON] {artifact.binary_npz_path}")
    if target is not None:
        print("[CLASSIFICATION] ROCm platform-validation certified generation "
              "— NON-AUTHORITATIVE")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        print(f"\n[{_FAIL}] 3.B smoke did NOT reach acceptance")
        sys.exit(1)
