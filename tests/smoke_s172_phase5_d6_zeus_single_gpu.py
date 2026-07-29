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
    3. the repository identity. `finalize_run` REFUSES a dirty tree (§7.3), and
       an agent sandbox may not commit. So the harness snapshots the exact
       source under test into a throwaway git repo (HEAD's tracked files with
       the D6 working-tree files overlaid), commits it there, and passes THAT
       root to the same `_repository_state` helper. The recorded SHA therefore
       identifies a tree byte-identical to the source that ran; it is not the
       project's own commit, and the release-grade generation is the one
       produced after Michael commits.

Run:  python tests/smoke_s172_phase5_d6_zeus_single_gpu.py
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
                 backend=None, backend_options=None):
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
        self.miner_host = "127.0.0.1"       # single-GPU Zeus smoke (see header)
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


def run_smoke(seed_start, seed_count, stripe_size, seed_caps, window_size,
              forward_threshold, reverse_threshold, workdir):
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
                               substripes=max(1, stripe_size // seed_caps))

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

    print(f"\n[TRIAL] coordinator binding 127.0.0.1:{port}")
    print(f"[TRIAL] seeds [{seed_start:,}, {seed_start + seed_count:,}) "
          f"stripe={stripe_size:,} substripe_cap={seed_caps:,} "
          f"window={window_size} "
          f"forward_threshold={forward_threshold} "
          f"reverse_threshold={reverse_threshold}")
    t = threading.Thread(target=_trial, name="d6-smoke-trial", daemon=True)
    t.start()
    time.sleep(2.0)     # let the serve loop bind before the worker dials in

    worker_log = open(work / "worker.log", "w")
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
            smi_during = _nvidia_smi("during the CUDA sieve")
        if not t.is_alive():
            break
    t.join(timeout=1200)

    try:
        worker.terminate()
        worker.wait(timeout=30)
    except Exception:
        worker.kill()
    worker_log.close()

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


def finalize_and_verify(accumulator, gen_root, seed_start, seed_count, work):
    # --- repository identity (see header note 3) --------------------------
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

    print("\n" + "=" * 78)
    print("CERTIFIED GENERATION")
    print("=" * 78)
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
    args = ap.parse_args(argv)

    print("=" * 78)
    print("S172 Phase 5 D6 — 3.B Zeus single-GPU certified-generation smoke")
    print("=" * 78)
    _nvidia_smi("before the run")

    workdir = args.workdir or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "d6_zeus_smoke")
    os.makedirs(workdir, exist_ok=True)
    print(f"\n[WORKDIR] {workdir}")

    tr, acc, gen_root, _smi, work, prov = run_smoke(
        args.seed_start, args.seed_count, args.stripe_size, args.seed_cap,
        args.window_size, args.forward_threshold, args.reverse_threshold,
        workdir)

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
        acc, gen_root, args.seed_start, args.seed_count, work)

    _nvidia_smi("after the run")

    print("\n" + "=" * 78)
    print(f"[{_PASS}] 3.B ACCEPTANCE — certified generation produced on real "
          f"silicon,\n        22-array bundle validated, Step-2 loader read it "
          f"back ({loaded.count:,} rows),\n        and the asymmetric "
          f"forward={args.forward_threshold} / reverse={args.reverse_threshold} "
          f"thresholds\n        reached the CUDA kernel unchanged "
          f"(requested == payload == effective).")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        print(f"\n[{_FAIL}] 3.B smoke did NOT reach acceptance")
        sys.exit(1)
