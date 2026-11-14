"""Survivor-volume ESTIMATOR — a load estimate for one exact geometry and seed range.

WHAT THIS IS
------------
Plan item A2 / finished condition F3
(`docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md` §2, §3.A;
`docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md` §7.3). It turns the hand method of
2026-08-27 (`docs/S172_PHASE3_SURVIVOR_CAPACITY_CHARACTERIZATION.md` §3) into a
callable: sample a few blocks of the seed range through the PRODUCTION path, count
survivors per phase, and project to the whole range with an exact 95% interval.

    estimate_volume(geometry, seed_start, seed_end_exclusive,
                    sample_seeds={phase: n}, sampler=...) -> dict

It takes ONE exact geometry. It reads no cap, applies no cap and returns no verdict.
`scripts/survivor_volume_gate.py` is one caller and applies the cap; a later brief
will call the same function once per Optuna trial. Both get the same figures.

A SURVIVOR COUNT IS LOAD, NOT QUALITY. TFM is judged by prediction-pool hit rate
against k/1000 on unseen draws (CLAUDE.md, "The yardstick"). Nothing here is a
score, and nothing here may be read as one.

THE PRODUCTION PATH, NOT A MODEL OF IT
--------------------------------------
For each workflow phase (`workflow_stages_for`, the coordinator's own table):
  * residues      `load_residue_window` — the ONE derivation both the coordinator side
                  (`window_optimizer_integration_final._miner_residues_for_config`)
                  and the worker (`ResidueResolver`) use;
  * payload       `RangeMinerCoordinator.build_stripe_assign_payload` — the single
                  threshold chokepoint, direction-resolved per phase;
  * sub-stripes   `select_seed_cap` + `partition_stripe` — what `handle_stripe` does;
  * execution     `SieveExecutor.execute` with the production `ResidueResolver`, which
                  re-verifies dataset_sha256 and residue_sha256 against the payload,
                  compiles the production kernel, launches it and applies the host
                  post-filter;
  * count         `SubStripeOutcome.count` = survivors after the host filter — what a
                  run counts. The kernel-admitted count is read from the kernel's own
                  survivor counter at the one mockable launch seam (`_gpu_launch`),
                  observed after the production launch, never recomputed.
Nothing of that is re-implemented here.

THE SAMPLER CONTRACT (injectable, so tests run on CPU)
-----------------------------------------------------
    sampler(request) -> result
    request = {"phase", "family_name", "prng_base", "payload", "intervals"}
    result  = {"host_hits": int, "kernel_hits": int, "intervals": [[start, count], ...],
               "residue_sha256": str, "effective_threshold": float,
               "seconds": float, "gpu": str, "kernel": {...},
               "gpu_pool_peak_bytes": int | None}
A result that is not exactly this shape raises `SamplerMalformed`.

Memory: survivors are counted and discarded one sub-stripe at a time (at most one
production seed cap of seeds, 2.5M-5M), so host RSS does not grow with the sample.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

ESTIMATOR_VERSION = "survivor-volume-estimator/1"
CONFIDENCE = 0.95
N_BLOCKS = 8

# The geometry is these keys, each an EXACT value. `test_both_modes` decides which
# workflow phases a trial runs, so it is part of the geometry.
GEOMETRY_KEYS = (
    "prng_type", "window_size", "window_anchor", "generator_phase", "sessions",
    "forward_threshold", "reverse_threshold", "skip_min", "skip_max",
    "dataset", "test_both_modes",
)
_INT_KEYS = ("window_size", "window_anchor", "generator_phase", "skip_min", "skip_max")
_NUM_KEYS = ("forward_threshold", "reverse_threshold")


class EstimatorError(Exception):
    """Base. Each subclass is one failure class the caller maps to a verdict."""


class GeometryError(EstimatorError):
    """A parameter is missing, a range, of the wrong type, or refused by the
    production payload builder / residue loader."""


class SampleError(EstimatorError):
    """The sample requested or performed is not a valid sample of the range."""


class ResidueMismatch(EstimatorError):
    """The residue window sampled on is not the one the payload builder states."""


class SamplerUnavailable(EstimatorError):
    """The sampler could not obtain a GPU."""


class SamplerFailed(EstimatorError):
    """The sampler raised."""


class SamplerTimeout(EstimatorError):
    """The sampler did not return in time."""


class SamplerMalformed(EstimatorError):
    """The sampler returned something that is not a result."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def validate_geometry(geometry: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of the geometry with exact, typed values. Raises GeometryError
    on the first defect. A range (an object, or a list where a scalar belongs) is
    refused: sampled settings are judged one trial at a time."""
    if not isinstance(geometry, dict):
        raise GeometryError("geometry is not a JSON object")
    missing = [k for k in GEOMETRY_KEYS if k not in geometry]
    if missing:
        raise GeometryError(f"geometry parameter(s) missing: {missing}")
    extra = sorted(set(geometry) - set(GEOMETRY_KEYS))
    if extra:
        raise GeometryError(f"unknown geometry parameter(s) {extra}: an unknown "
                            f"parameter is never assumed harmless")
    g = dict(geometry)
    for k in GEOMETRY_KEYS:
        v = g[k]
        if isinstance(v, dict):
            raise GeometryError(f"{k} = {v!r} is a range/object, not an exact value")
    for k in _INT_KEYS:
        if not _is_int(g[k]):
            raise GeometryError(f"{k} = {g[k]!r} is not an exact integer")
    for k in _NUM_KEYS:
        if not _is_num(g[k]):
            raise GeometryError(f"{k} = {g[k]!r} is not an exact finite number")
        g[k] = float(g[k])
    if not isinstance(g["prng_type"], str) or not g["prng_type"]:
        raise GeometryError(f"prng_type = {g['prng_type']!r} is not a family name")
    if not isinstance(g["dataset"], str) or not g["dataset"]:
        raise GeometryError(f"dataset = {g['dataset']!r} is not a path")
    if not isinstance(g["test_both_modes"], bool):
        raise GeometryError(f"test_both_modes = {g['test_both_modes']!r} is not a boolean")
    s = g["sessions"]
    if not (isinstance(s, list) and s and all(isinstance(x, str) and x for x in s)):
        raise GeometryError(f"sessions = {s!r} is not a non-empty list of session names")
    if g["window_size"] < 1 or g["window_anchor"] < 0:
        raise GeometryError("window_size must be >= 1 and window_anchor >= 0")
    if g["skip_min"] > g["skip_max"]:
        raise GeometryError(f"skip_min {g['skip_min']} > skip_max {g['skip_max']}")
    return g


def validate_range(seed_start, seed_end_exclusive) -> int:
    if not (_is_int(seed_start) and _is_int(seed_end_exclusive)):
        raise GeometryError(f"seed range not integers: seed_start={seed_start!r} "
                            f"seed_end_exclusive={seed_end_exclusive!r}")
    if seed_start < 0 or seed_end_exclusive <= seed_start:
        raise GeometryError(f"seed range [{seed_start}, {seed_end_exclusive}) is "
                            f"empty or negative")
    return seed_end_exclusive - seed_start


def request_digest(geometry: Dict[str, Any], seed_start: int, seed_end_exclusive: int) -> str:
    return hashlib.sha256(canonical_json(
        {"geometry": geometry, "seed_start": seed_start,
         "seed_end_exclusive": seed_end_exclusive}).encode("utf-8")).hexdigest()


def sample_intervals(seed_start: int, seed_end_exclusive: int, sample_seeds: int,
                     digest: str, n_blocks: int = N_BLOCKS) -> List[List[int]]:
    """The fixed rule. Split the range into `n_blocks` near-equal segments and take
    one block from each, at an offset inside the segment chosen by sha256(digest,
    sample size, block index). Same request -> same sample. Every block lies inside
    its segment, so blocks never overlap and never leave the range."""
    range_size = validate_range(seed_start, seed_end_exclusive)
    if not _is_int(sample_seeds) or sample_seeds < 1:
        raise SampleError(f"sample size {sample_seeds!r} is not a positive integer")
    if sample_seeds > range_size:
        raise SampleError(f"sample size {sample_seeds} exceeds the range size {range_size}")
    n = max(1, min(n_blocks, sample_seeds))
    segs = [(seed_start + (range_size * i) // n, seed_start + (range_size * (i + 1)) // n)
            for i in range(n)]
    # Block sizes by segment capacity (Beta R1 finding 2): floor(S*seg/R) never exceeds
    # its segment because S <= R; the remainder goes one seed at a time to segments with
    # room, which always exists because total room >= remainder. Equal segments give
    # equal blocks, so power-of-two requests are unchanged.
    # Every segment first gets one seed (n <= S <= R, and every segment holds >= 1), so
    # no block is empty; the other S - n are shared by remaining capacity.
    if range_size == n:
        blks = [1] * n
    else:
        blks = [1 + ((sample_seeds - n) * (hi - lo - 1)) // (range_size - n)
                for lo, hi in segs]
    rem = sample_seeds - sum(blks)
    while rem:
        for i, (lo, hi) in enumerate(segs):
            if rem and blks[i] < hi - lo:
                blks[i] += 1
                rem -= 1
    out = []
    for i in range(n):
        seg_lo, seg_hi = segs[i]
        blk = blks[i]
        slack = (seg_hi - seg_lo) - blk
        if slack < 0:
            raise SampleError(f"block {i} of {blk} seeds does not fit its segment "
                              f"[{seg_lo}, {seg_hi})")
        h = hashlib.sha256(f"{digest}|{sample_seeds}|{i}".encode("utf-8")).digest()
        off = int.from_bytes(h, "big") % (slack + 1)
        out.append([seg_lo + off, blk])
    return out


def poisson_interval(k: int, confidence: float = CONFIDENCE) -> Tuple[float, float]:
    """Exact (Garwood) two-sided interval on a Poisson mean for k observed events.
    k = 0 gives (0, -ln(alpha/2)) — an upper bound that is never 0."""
    from scipy.stats import chi2
    alpha = 1.0 - confidence
    lo = 0.0 if k == 0 else float(chi2.ppf(alpha / 2.0, 2 * k)) / 2.0
    hi = float(chi2.ppf(1.0 - alpha / 2.0, 2 * k + 2)) / 2.0
    return lo, hi


def project(hits: int, sample_size: int, range_size: int,
            confidence: float = CONFIDENCE) -> Dict[str, Any]:
    """Projection = hits x (range size / sample size). The interval is the Poisson
    interval on the sample's count, scaled by the same factor. COMPUTED figures."""
    scale = range_size / sample_size
    lo, hi = poisson_interval(hits, confidence)
    return {"scale": scale, "projection": hits * scale,
            "lower95": lo * scale, "upper95": hi * scale,
            "interval_method": (
                f"Garwood {confidence:.0%} interval for a Poisson mean on the sample count, "
                f"scaled by range/sample. CONDITIONAL ON THE POISSON COUNT MODEL: the sample "
                f"is a few deterministic contiguous blocks of a finite PRNG population, and "
                f"neither block dependence nor coverage calibration has been measured. "
                f"Per-block hits are recorded for that calibration.")}


# ---------------------------------------------------------------------------
# the production sampler
# ---------------------------------------------------------------------------

def _counting_executor_class():
    """A SieveExecutor that additionally reads the kernel's own survivor counter
    after the production launch. `_gpu_launch` is the executor's single mockable
    kernel entry; the override calls it unchanged and only observes."""
    from miner.range_miner_worker import SieveExecutor

    class CountingSieveExecutor(SieveExecutor):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.last_kernel_count: Optional[int] = None
            self.last_n: Optional[int] = None
            self.pool_peak_bytes = 0

        def _gpu_launch(self, kernel, blocks, threads, kernel_args):
            super()._gpu_launch(kernel, blocks, threads, kernel_args)
            import cupy as cp
            counters = [a for a in kernel_args
                        if isinstance(a, cp.ndarray) and a.size == 1
                        and a.dtype == cp.uint32]
            if len(counters) != 1:
                raise RuntimeError(f"cannot identify the kernel survivor counter: "
                                   f"{len(counters)} candidate buffers")
            self.last_kernel_count = int(counters[0][0].get())
            self.pool_peak_bytes = max(self.pool_peak_bytes,
                                       int(cp.get_default_memory_pool().used_bytes()))

    return CountingSieveExecutor


def production_sample(device_index: int, backend: str, caps, request: Dict[str, Any]
                      ) -> Dict[str, Any]:
    """The GPU work of one phase's sample, through `SieveExecutor.execute`. Module-level
    so a spawned child process can run it (see `IsolatedProcessSampler`)."""
    from miner.range_miner_protocol import StripeAssignMessage
    from miner.range_miner_worker import (
        ResidueResolver, VramCaps, partition_stripe, select_seed_cap,
        sha256_residues)
    try:
        import cupy as cp
        ndev = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:                                        # noqa: BLE001
        raise SamplerUnavailable(f"no usable GPU runtime: {type(exc).__name__}: {exc}")
    if device_index >= ndev:
        raise SamplerUnavailable(f"device {device_index} absent ({ndev} present)")
    props = cp.cuda.runtime.getDeviceProperties(device_index)
    name = props["name"]
    gpu = name.decode() if isinstance(name, bytes) else str(name)
    caps = caps or VramCaps()
    family = request["family_name"]
    payload = request["payload"]
    cap = select_seed_cap(backend, family, caps)
    resolver = ResidueResolver()
    executor = _counting_executor_class()(resolver, device_index=device_index)
    host = kernel = 0
    effective = set()
    done: List[List[int]] = []
    block_hits: List[int] = []
    t0 = time.perf_counter()
    for start, count in request["intervals"]:
        assign = StripeAssignMessage(
            worker_id="survivor-volume-sampler",
            stripe_id=f"volume-sample-p{request['phase']}-{start}",
            trial_number=-1, prng_type=request["prng_base"],
            family_name=family, seed_start=start, seed_count=count,
            phase=request["phase"], attempt=0, payload=payload)
        in_block = 0
        for sub in partition_stripe(start, count, cap):
            out = executor.execute(assign, sub.seed_start, sub.seed_count)
            in_block += int(out.count)
            kernel += min(int(executor.last_kernel_count), sub.seed_count)
            effective.add(out.effective_threshold)
        host += in_block
        block_hits.append(in_block)
        done.append([start, count])
    seconds = time.perf_counter() - t0
    if len(effective) != 1:
        raise RuntimeError(f"effective threshold not unique across sub-stripes: "
                           f"{sorted(effective)}")
    from sieve_gpu_worker import _get_kernel
    _k, cfg = _get_kernel(family)
    return {
        "host_hits": host, "kernel_hits": kernel, "intervals": done,
        "block_hits": block_hits,
        "residue_sha256": sha256_residues(resolver.resolve(payload)),
        "effective_threshold": effective.pop(),
        "seconds": seconds, "gpu": f"{backend}:{device_index}:{gpu}",
        "kernel": {"family_name": family, "kernel_name": cfg.get("kernel_name"),
                   "kernel_source_sha256": hashlib.sha256(
                       cfg["kernel_source"].encode("utf-8")).hexdigest(),
                   "seed_cap": cap},
        "gpu_pool_peak_bytes": executor.pool_peak_bytes,
    }


# The child's whole program. It reads (sys.path, (fn, args, request)) from one private
# pipe and writes ("ok", result) | ("err", class, message) to another. It never imports
# the CALLER's main module, so a caller without an `if __name__ == "__main__"` guard
# cannot be re-executed in the child (the multiprocessing-spawn hazard).
_CHILD_PROGRAM = r"""
import json, os, pickle, sys
fd_in, fd_out = int(sys.argv[1]), int(sys.argv[2])
with os.fdopen(fd_in, "rb") as fh:
    path = json.loads(fh.readline())
    sys.path[:0] = [p for p in path if p not in sys.path]
    fn, args, request = pickle.load(fh)
try:
    out = ("ok", fn(*args, request))
except BaseException as exc:
    out = ("err", type(exc).__name__, str(exc))
with os.fdopen(fd_out, "wb") as fh:
    pickle.dump(out, fh)
"""


class IsolatedProcessSampler:
    """Runs `fn(*args, request)` in a separate child process (`python -c`), so the GPU
    context belongs to the child and a timeout can actually stop the work.

    [X-VOLUME-LIMIT R1, Beta Q6 LIVE] A thread cannot be cancelled: the R1 timeout
    returned ERROR while a non-daemon worker thread kept the process alive. Here the
    parent waits for the child; on timeout it terminates it (SIGTERM), then kills it
    (SIGKILL), and reaps it before raising. If the child still cannot be reaped, the
    raised SamplerTimeout says so and names the PID — it is never reported as cleaned
    up. A fresh interpreter, never `fork`: no initialised GPU context is inherited, and
    the parent never initialises one. `fn` must be importable by module name.

    `last_child` records the outcome of the most recent call, for the evidence record."""

    TERM_GRACE_S = 10.0

    def __init__(self, fn, *args):
        self.fn = fn
        self.args = args
        self.last_child: Optional[Dict[str, Any]] = None

    def __call__(self, request):                     # no timeout: wait indefinitely
        return self.run_isolated(request, None)

    def run_isolated(self, request, timeout_s):
        import pickle
        import subprocess
        import threading
        blob = (json.dumps(list(sys.path)) + "\n").encode("utf-8") + pickle.dumps(
            (self.fn, self.args, request))
        r_in, w_in = os.pipe()
        r_out, w_out = os.pipe()
        try:
            proc = subprocess.Popen(
                [sys.executable, "-c", _CHILD_PROGRAM, str(r_in), str(w_out)],
                pass_fds=(r_in, w_out), close_fds=True)
        finally:
            os.close(r_in)
            os.close(w_out)
        rec: Dict[str, Any] = {"pid": proc.pid}
        self.last_child = rec
        chunks: List[bytes] = []

        def drain():                    # daemon: never holds process exit
            with os.fdopen(r_out, "rb") as fh:
                while True:
                    b = fh.read(65536)
                    if not b:
                        return
                    chunks.append(b)

        reader = threading.Thread(target=drain, name="survivor-volume-child-reader",
                                  daemon=True)
        reader.start()
        try:
            with os.fdopen(w_in, "wb") as fh:
                fh.write(blob)
        except BrokenPipeError:
            pass                        # the child died early; reported below
        try:
            proc.wait(timeout_s)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(self.TERM_GRACE_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(self.TERM_GRACE_S)
                except subprocess.TimeoutExpired:
                    pass
            alive = proc.poll() is None
            rec.update(timed_out=True, exitcode=proc.returncode, alive=alive)
            if alive:
                raise SamplerTimeout(
                    f"sampler did not return within {timeout_s} s; child pid {proc.pid} "
                    f"could NOT be stopped (SIGTERM then SIGKILL) — UNRESOLVED")
            raise SamplerTimeout(
                f"sampler did not return within {timeout_s} s; child pid {proc.pid} "
                f"terminated (exitcode {proc.returncode})")
        reader.join(self.TERM_GRACE_S)
        rec.update(timed_out=False, exitcode=proc.returncode, alive=False)
        data = b"".join(chunks)
        if not data:
            raise SamplerFailed(f"sampler child pid {proc.pid} exited without a result "
                                f"(exitcode {proc.returncode})")
        msg = pickle.loads(data)
        if msg[0] == "ok":
            return msg[1]
        if msg[1] == "SamplerUnavailable":
            raise SamplerUnavailable(msg[2])
        raise SamplerFailed(f"sampler raised {msg[1]}: {msg[2]}")



class ProductionSampler(IsolatedProcessSampler):
    """Runs the sample on one local GPU through `SieveExecutor.execute`, in a spawned
    child process per call (`IsolatedProcessSampler`).

    backend: "cuda" or "rocm" — selects the production seed cap
    (`select_seed_cap`), exactly as `handle_stripe` does on that backend."""

    def __init__(self, device_index: int = 0, backend: str = "cuda", caps=None):
        super().__init__(production_sample, device_index, backend, caps)
        self.device_index = device_index
        self.backend = backend
        self.caps = caps


# ---------------------------------------------------------------------------
# the estimate
# ---------------------------------------------------------------------------

def _phase_payloads(g: Dict[str, Any]):
    """[(family, phase, payload, residues)] from the production table, builder and
    loader. Builder/loader refusals are GeometryError."""
    from miner.range_miner_coordinator import (
        CoordinatorConfig, MinerLedger, RangeMinerCoordinator, resolve_dataset_sha256,
        workflow_stages_for)
    from miner.range_miner_worker import load_residue_window
    try:
        stages = workflow_stages_for(g["prng_type"], g["test_both_modes"])
        residues = load_residue_window(g["dataset"], g["window_size"],
                                       list(g["sessions"]), g["window_anchor"])
        dataset_sha = resolve_dataset_sha256(g["dataset"])
        # The builder is an instance method; it reads nothing from the instance.
        # An in-memory ledger keeps construction free of any file write.
        coord = RangeMinerCoordinator(CoordinatorConfig(), MinerLedger(":memory:"))
        out = []
        for family, phase in stages:
            payload = coord.build_stripe_assign_payload(
                g["dataset"], g["window_size"], list(g["sessions"]),
                g["window_anchor"], residues, dataset_sha256=dataset_sha,
                phase=phase, forward_threshold=g["forward_threshold"],
                reverse_threshold=g["reverse_threshold"],
                generator_phase=g["generator_phase"])
            out.append((family, phase, payload))
    except EstimatorError:
        raise
    except Exception as exc:                                        # noqa: BLE001
        raise GeometryError(f"production builder/loader refused the geometry: "
                            f"{type(exc).__name__}: {exc}") from exc
    return out, residues, dataset_sha


def _call_with_timeout(fn, arg, timeout_s):
    """A sampler with `run_isolated` owns its own lifetime (a terminable child
    process). Any other sampler runs in a DAEMON thread: on timeout the call returns
    and the thread cannot hold the process open. A daemon thread is still not
    cancelled — production sampling therefore always goes through the process path."""
    if hasattr(fn, "run_isolated"):
        try:
            return fn.run_isolated(arg, timeout_s)
        except EstimatorError:
            raise
        except Exception as exc:                                    # noqa: BLE001
            raise SamplerFailed(f"sampler raised {type(exc).__name__}: {exc}") from exc
    import threading
    box: Dict[str, Any] = {}

    def run():
        try:
            box["result"] = fn(arg)
        except BaseException as exc:                                # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=run, name="survivor-volume-sampler", daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise SamplerTimeout(f"sampler did not return within {timeout_s} s "
                             f"(daemon thread abandoned; it does not hold process exit)")
    if "error" in box:
        exc = box["error"]
        if isinstance(exc, SamplerUnavailable):
            raise exc
        raise SamplerFailed(f"sampler raised {type(exc).__name__}: {exc}") from exc
    return box.get("result")


_RESULT_KEYS = {"host_hits", "kernel_hits", "intervals", "block_hits", "residue_sha256",
                "effective_threshold", "seconds", "gpu", "kernel", "gpu_pool_peak_bytes"}


def _check_result(res, request):
    if not isinstance(res, dict):
        raise SamplerMalformed(f"sampler returned {type(res).__name__}, not a dict")
    if set(res) != _RESULT_KEYS:
        raise SamplerMalformed(f"sampler result keys {sorted(res)} != {sorted(_RESULT_KEYS)}")
    for k in ("host_hits", "kernel_hits"):
        if not _is_int(res[k]) or res[k] < 0:
            raise SamplerMalformed(f"{k} = {res[k]!r} is not a non-negative integer")
    if res["host_hits"] > res["kernel_hits"]:
        raise SamplerMalformed(f"host_hits {res['host_hits']} > kernel_hits "
                               f"{res['kernel_hits']}: the host filter cannot add")
    iv = res["intervals"]
    if not (isinstance(iv, list) and all(
            isinstance(p, (list, tuple)) and len(p) == 2 and all(_is_int(x) for x in p)
            for p in iv)):
        raise SamplerMalformed(f"intervals {iv!r} are not [[start, count], ...]")
    bh = res["block_hits"]
    if not (isinstance(bh, list) and len(bh) == len(iv)
            and all(_is_int(x) and x >= 0 for x in bh) and sum(bh) == res["host_hits"]):
        raise SamplerMalformed(f"block_hits {bh!r} is not one non-negative count per "
                               f"block summing to host_hits {res['host_hits']}")
    if not isinstance(res["residue_sha256"], str):
        raise SamplerMalformed("residue_sha256 is not a string")
    if not _is_num(res["effective_threshold"]) or not _is_num(res["seconds"]):
        raise SamplerMalformed("effective_threshold/seconds are not finite numbers")
    if not isinstance(res["gpu"], str) or not isinstance(res["kernel"], dict):
        raise SamplerMalformed("gpu/kernel identity malformed")


def estimate_volume(geometry: Dict[str, Any], seed_start: int, seed_end_exclusive: int,
                    *, sample_seeds: Dict[int, int],
                    sampler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
                    timeout_s: float = 1800.0,
                    confidence: float = CONFIDENCE) -> Dict[str, Any]:
    """Estimate per-phase survivor volume (LOAD) for ONE exact geometry on
    [seed_start, seed_end_exclusive). No cap, no verdict.

    sample_seeds: {phase: seeds to sample}, one entry per workflow phase.
    Raises an EstimatorError subclass on any failure; never returns a partial."""
    g = validate_geometry(geometry)
    range_size = validate_range(seed_start, seed_end_exclusive)
    stages, residues, dataset_sha = _phase_payloads(g)
    phases_needed = sorted(p for _f, p, _pl in stages)
    if not isinstance(sample_seeds, dict) or sorted(sample_seeds) != phases_needed:
        raise SampleError(f"sample_seeds must name exactly phases {phases_needed}, "
                          f"got {sorted(sample_seeds) if isinstance(sample_seeds, dict) else sample_seeds!r}")
    sampler = sampler or ProductionSampler()
    digest = request_digest(g, seed_start, seed_end_exclusive)
    from miner.range_miner_worker import sha256_residues
    expected_residue = sha256_residues(residues)
    per_phase = []
    for family, phase, payload in stages:
        if payload["residue_sha256"] != expected_residue:  # pragma: no cover
            raise ResidueMismatch("payload builder digest != loader digest")
        intervals = sample_intervals(seed_start, seed_end_exclusive,
                                     sample_seeds[phase], digest)
        request = {"phase": phase, "family_name": family, "prng_base": g["prng_type"],
                   "payload": payload, "intervals": [list(p) for p in intervals]}
        res = _call_with_timeout(sampler, request, timeout_s)
        _check_result(res, request)
        for s, c in res["intervals"]:
            if c < 1 or s < seed_start or s + c > seed_end_exclusive:
                raise SampleError(f"phase {phase}: sample block [{s}, {s + c}) lies "
                                  f"outside the requested range "
                                  f"[{seed_start}, {seed_end_exclusive})")
        if [list(p) for p in res["intervals"]] != request["intervals"]:
            raise SampleError(f"phase {phase}: sampler sampled {res['intervals']} but "
                              f"was asked for {request['intervals']}")
        if res["residue_sha256"] != payload["residue_sha256"]:
            raise ResidueMismatch(
                f"phase {phase}: sampled on residue window {res['residue_sha256']} but "
                f"the payload builder gives {payload['residue_sha256']}")
        n = sum(c for _s, c in res["intervals"])
        proj = project(res["host_hits"], n, range_size, confidence)
        kproj = project(res["kernel_hits"], n, range_size, confidence)
        per_phase.append({
            "phase": phase, "family_name": family,
            "direction_threshold": payload["min_match_threshold"],
            "effective_threshold": res["effective_threshold"],
            "sample_seeds": n, "intervals": res["intervals"],
            "block_hits": res["block_hits"],
            "hits": res["host_hits"], "hits_label": "MEASURED (after host filter)",
            "kernel_hits": res["kernel_hits"], "kernel_hits_label": "MEASURED (kernel-admitted)",
            "projection": proj["projection"], "lower95": proj["lower95"],
            "upper95": proj["upper95"], "projection_label": "COMPUTED",
            "kernel_projection": kproj["projection"],
            "scale": proj["scale"], "interval_method": proj["interval_method"],
            "seconds": res["seconds"], "gpu": res["gpu"], "kernel": res["kernel"],
            "gpu_pool_peak_bytes": res["gpu_pool_peak_bytes"],
        })
    return {
        "estimator": ESTIMATOR_VERSION,
        "measure": "LOAD — survivor volume per phase; not a quality score",
        "geometry": g, "seed_start": seed_start,
        "seed_end_exclusive": seed_end_exclusive, "range_size": range_size,
        "request_digest": digest, "dataset_sha256": dataset_sha,
        "residue_sha256": expected_residue, "window_residues": list(residues),
        "skip_bounds_note": ("skip_min/skip_max are recorded geometry; the production "
                             "assign payload carries no skip_range "
                             "(build_stripe_assign_payload), so they do not reach the "
                             "sampled kernels — exactly as in a run"),
        "phases": per_phase,
    }
