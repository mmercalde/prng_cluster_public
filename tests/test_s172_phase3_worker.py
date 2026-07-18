#!/usr/bin/env python3
"""
test_s172_phase3_worker.py — S172 Phase 3 worker-daemon acceptance harness (rev-2)

Phase 3 rev-3 = miner/range_miner_worker.py after Team Beta's five blocker fixes.
This harness keeps the original 8 contract gates and adds the Beta-mandated
blocking gates for the dangerous paths.

Gates (all block-on-failure):
   1. Builder registry coverage (6 covered / 5 uncovered, no default).
   2. Per-family arg-shape: constant + forward-hybrid (audited lengths/dtypes) +
      reverse-hybrid (14) + dtype-preserving materialization (uint64 ABI).
   3. Per-family cap selection (rocm + cuda).
   4. Sub-stripe partitioning (no gaps/overlaps).
   5. Handshake + round-trip on a loopback socket (result + complete + error).
   6. Uncovered family launches no kernel.
   7. GPU smoke (skippable): one java_lcg sub-stripe end-to-end.
   8. Phase 0/1/2 non-regression (subprocess re-run).
   9. [B1] Per-assignment residue window: two assignments, different windows,
      correct residues each (fake loader records the key).
  10. [B2a] Spooled result: over-threshold result writes a spool file; path/size/
      sha256 set, inline=None, re-hash matches the file bytes.
  11. [B2b] Size-based (not count-based) selection: a FEW huge hybrid survivors
      spool; a LARGE count of tiny survivors stays inline.
  12. [B3] Cleanup after exception (skippable-GPU): forced launch failure still
      runs the full cleanup hook, and the daemon serves the next assignment.
  13. [B4] Exact capability advertisement: register advertises concrete variants
      incl. the hybrids now built; no uncovered variant claimed.
  14. [B4/§11.I] Non-Java full-mode: lcg32 + minstd test_both_modes run all four
      phase variants through the correct builders (CPU arg-shape).

Run:
    cd ~/distributed_prng_analysis
    source ~/venvs/torch/bin/activate      # cupy lives here (GPU gates run real)
    PYTHONPATH=. python3 tests/test_s172_phase3_worker.py

A CPU-only green validates the CONTRACT (arg shapes, protocol, spool, capability)
— it is NOT ROCm deploy-readiness; that is Phase 6 acceptance on real rigs.

Exit code 0 = all gates green. Exit code 1 = a gate failed (DO NOT COMMIT).
"""
import hashlib
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import traceback
from unittest import mock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"
_SKIP = "\033[93mSKIP\033[0m"

_results = []


def _check(name, fn):
    try:
        fn()
        _results.append((name, True, None))
        print(f"  [{_PASS}] {name}")
    except Exception as e:
        _results.append((name, False, traceback.format_exc()))
        print(f"  [{_FAIL}] {name}: {e}")


from miner.range_miner_protocol import (  # noqa: E402
    MinerShutdownMessage,
    MinerStatusMessage,
    StripeAssignMessage,
)
import miner.range_miner_worker as W  # noqa: E402
from miner.range_miner_worker import (  # noqa: E402
    BufferArg,
    BuildContext,
    COVERED_FAMILIES,
    GpuInfo,
    INLINE_BYTE_LIMIT,
    MinerFramedSocket,
    RangeMinerWorker,
    ResidueResolutionError,
    ResidueResolver,
    ScalarArg,
    SieveExecutor,
    SubStripe,
    SubStripeOutcome,
    VramCaps,
    _constant_prefix,
    _hybrid_prefix,
    base_family,
    build_substripe_payload_bytes,
    is_hybrid_family,
    is_reverse_family,
    kernel_args_builders,
    materialize_kernel_args,
    partition_stripe,
    resolve_builder,
    select_seed_cap,
    sha256_residues,
    supported_variants,
)

UNCOVERED = ["mt19937", "philox4x32", "sfc64", "xorshift64", "xoshiro256pp"]
COVERED = ["java_lcg", "lcg32", "minstd", "pcg32", "xorshift128", "xorshift32"]

# AUDITED forward-hybrid arg lengths (from live prng_registry.py signatures).
HYBRID_FWD_LEN = {
    "java_lcg": 15, "lcg32": 17, "minstd": 15,
    "pcg32": 15, "xorshift32": 16, "xorshift128": 16,
}
SEED_DTYPE = {b: ("uint64" if b == "java_lcg" else "uint32") for b in COVERED}


def _ctx(family, **over):
    base = base_family(family)
    d = dict(
        family_name=family,
        hybrid=is_hybrid_family(family),
        reverse=is_reverse_family(family),
        seed_dtype=SEED_DTYPE[base],
        n_seeds=1000, k=10, skip_min=0, skip_max=16,
        threshold=0.25, offset=3, params={}, n_strategies=4, hybrid_threshold=0.3,
    )
    d.update(over)
    return BuildContext(**d)


def _dummy_resolver(residues):
    return ResidueResolver(
        loader=lambda dataset, ws, sess, off: list(residues),
        file_hasher=lambda p: "deadbeef",
    )


# ---------------------------------------------------------------------------
# GATE 1 — builder registry coverage
# ---------------------------------------------------------------------------
def gate1_builder_registry_coverage():
    assert set(kernel_args_builders) == set(COVERED)
    assert set(COVERED) == set(COVERED_FAMILIES)
    assert kernel_args_builders.get("mt19937") is None
    for fam in COVERED:
        b = resolve_builder(fam)
        assert callable(b)
        assert resolve_builder(fam + "_reverse") is b
        assert resolve_builder(fam + "_hybrid") is b
        assert resolve_builder(fam + "_hybrid_reverse") is b
    for fam in UNCOVERED:
        for variant in (fam, fam + "_reverse", fam + "_hybrid", fam + "_hybrid_reverse"):
            try:
                resolve_builder(variant)
            except NotImplementedError as e:
                assert fam in str(e)
            else:
                raise AssertionError(f"uncovered {variant} did not raise")
    assert base_family("java_lcg_hybrid_reverse") == "java_lcg"
    assert base_family("xorshift128") == "xorshift128"


# ---------------------------------------------------------------------------
# GATE 2 — per-family arg-shapes (constant / forward-hybrid / reverse-hybrid)
# ---------------------------------------------------------------------------
def _buf_names(args):
    return [a.name for a in args if isinstance(a, BufferArg)]


def _last_scalar_dtypes(args, n):
    tail = args[-n:]
    assert all(isinstance(a, ScalarArg) for a in tail), tail
    return [a.dtype for a in tail]


def gate2_arg_shapes():
    # --- FORWARD constant families: 11-prefix + family tail + int32 offset ---
    for fam in COVERED:
        cp = _constant_prefix(_ctx(fam))
        args = resolve_builder(fam)(_ctx(fam))
        assert args[:11] == cp, fam
        assert isinstance(args[-1], ScalarArg) and args[-1].dtype == "int32", fam
        assert args[-1].value == 3, fam                       # offset
        assert "best_skips" in _buf_names(args), fam
        assert "skip_sequences" not in _buf_names(args), fam

    # --- REVERSE constant families: 12 args = _constant_prefix + int32(offset),
    #     NO family tail (generator params hardcoded in the reverse kernel). This
    #     is the rev-3 fix: forward and reverse constant layouts are NOT identical.
    for fam in COVERED:
        rc = resolve_builder(fam)(_ctx(fam + "_reverse"))
        assert len(rc) == 12, (fam + "_reverse", len(rc))
        assert rc == _constant_prefix(_ctx(fam + "_reverse")) \
            + [ScalarArg(3, "int32")], fam + "_reverse"
        assert isinstance(rc[-1], ScalarArg) and rc[-1].dtype == "int32", fam
        assert "best_skips" in _buf_names(rc), fam
        assert "skip_sequences" not in _buf_names(rc), fam

    # --- forward-hybrid AUDITED lengths + trailing dtypes ---
    assert len(_hybrid_prefix(_ctx("java_lcg_hybrid"))) == 13
    for fam in COVERED:
        fh = resolve_builder(fam)(_ctx(fam + "_hybrid"))
        assert len(fh) == HYBRID_FWD_LEN[fam], (fam, len(fh))
        bn = _buf_names(fh)
        for req in ("skip_sequences", "strategy_ids", "strategy_max_misses",
                    "strategy_tolerances"):
            assert req in bn, (fam, req)
        assert "best_skips" not in bn, fam
        # shares the AUDITED 13-element prefix (family-aware: seed dtype differs)
        assert fh[:13] == _hybrid_prefix(_ctx(fam + "_hybrid")), fam

    # exact trailing ABI per family (do not extrapolate)
    assert _last_scalar_dtypes(resolve_builder("java_lcg")(_ctx("java_lcg_hybrid")), 2) \
        == ["uint64", "uint64"]                                # a, c ; no offset
    assert _last_scalar_dtypes(resolve_builder("lcg32")(_ctx("lcg32_hybrid")), 4) \
        == ["uint32", "uint32", "uint32", "int32"]             # a, c, m, offset
    assert _last_scalar_dtypes(resolve_builder("minstd")(_ctx("minstd_hybrid")), 2) \
        == ["uint32", "uint32"]                                # a, m_val ; no offset
    assert _last_scalar_dtypes(resolve_builder("pcg32")(_ctx("pcg32_hybrid")), 2) \
        == ["uint64", "int32"]                                 # increment, offset
    assert _last_scalar_dtypes(resolve_builder("xorshift32")(_ctx("xorshift32_hybrid")), 3) \
        == ["int32", "int32", "int32"]                         # shifts ; no offset
    assert _last_scalar_dtypes(resolve_builder("xorshift128")(_ctx("xorshift128_hybrid")), 3) \
        == ["int32", "int32", "int32"]                         # dummies ; no offset

    # --- reverse-hybrid: ALL families 14 args ending int32 offset ---
    for fam in COVERED:
        rh = resolve_builder(fam)(_ctx(fam + "_hybrid_reverse"))
        assert len(rh) == 14, (fam, len(rh))
        assert isinstance(rh[-1], ScalarArg) and rh[-1].dtype == "int32", fam
        assert "skip_sequences" in _buf_names(rh), fam

    # --- materialization preserves dtypes (uint64 a,c survives) ---
    import numpy as np
    fh = resolve_builder("java_lcg")(_ctx("java_lcg_hybrid"))
    buffers = {n: np.zeros(2, dtype=np.uint32) for n in _buf_names(fh)}
    mat = materialize_kernel_args(fh, buffers, np)
    assert mat[-2].dtype == np.uint64 and mat[-1].dtype == np.uint64
    assert mat[7].dtype == np.int32      # n_seeds
    assert mat[12].dtype == np.float32   # hybrid_threshold


# ---------------------------------------------------------------------------
# GATE 3 — per-family cap selection
# ---------------------------------------------------------------------------
def gate3_cap_selection():
    caps = VramCaps(2_000_000, 5_000_000, 1_000_000, 2_500_000)
    assert select_seed_cap("rocm", "java_lcg", caps) == 2_000_000
    assert select_seed_cap("rocm", "java_lcg_hybrid", caps) == 1_000_000
    assert select_seed_cap("rocm", "lcg32_hybrid_reverse", caps) == 1_000_000
    assert select_seed_cap("cuda", "minstd", caps) == 5_000_000
    assert select_seed_cap("cuda", "minstd_hybrid", caps) == 2_500_000
    try:
        select_seed_cap("vulkan", "java_lcg", caps)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown backend must raise")


# ---------------------------------------------------------------------------
# GATE 4 — sub-stripe partitioning
# ---------------------------------------------------------------------------
def _assert_covers(subs, start, count):
    assert subs[0].seed_start == start
    cursor = start
    covered = 0
    for i, s in enumerate(subs):
        assert s.sub_index == i
        assert s.seed_start == cursor
        assert s.seed_count > 0
        cursor += s.seed_count
        covered += s.seed_count
    assert covered == count and cursor == start + count


def gate4_partitioning():
    STRIPE = 67_108_864
    for cap in (5_000_000, 2_000_000, 1_000_000, 2_500_000):
        subs = partition_stripe(0, STRIPE, cap)
        assert len(subs) == -(-STRIPE // cap)
        _assert_covers(subs, 0, STRIPE)
        assert all(s.seed_count <= cap for s in subs)
    assert [(s.seed_start, s.seed_count) for s in partition_stripe(1000, 30, 10)] \
        == [(1000, 10), (1010, 10), (1020, 10)]
    assert [(s.seed_start, s.seed_count) for s in partition_stripe(0, 25, 10)] \
        == [(0, 10), (10, 10), (20, 5)]
    _assert_covers(partition_stripe(1_000_000, STRIPE, 5_000_000), 1_000_000, STRIPE)
    assert partition_stripe(0, 0, 10) == []
    try:
        partition_stripe(0, 10, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("cap<=0 must raise")


# ---------------------------------------------------------------------------
# GATE 5 — handshake + round-trip on a loopback socket
# ---------------------------------------------------------------------------
def _spin_worker(worker):
    err_box = {}

    def run():
        try:
            worker.connect()
            worker.register()
            worker.serve_forever()
        except Exception:
            err_box["err"] = traceback.format_exc()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, err_box


def gate5_handshake_roundtrip():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    srv.settimeout(10)
    port = srv.getsockname()[1]

    def stub_exec(assign, seed_start, seed_count):
        if (assign.payload or {}).get("force_fail"):
            raise RuntimeError("synthetic sub-stripe failure")
        return SubStripeOutcome(survivors=[(seed_start, 0.9, None, [1])], count=1)

    with tempfile.TemporaryDirectory() as spool:
        worker = RangeMinerWorker(
            host="127.0.0.1", port=port, gpu_id=6, caps=VramCaps(),
            executor=stub_exec,
            gpu_info=GpuInfo("cuda", "stub", 12 * 1024**3),
            heartbeat_interval=999, miner_output_dir=spool,
        )
        t, err_box = _spin_worker(worker)
        conn, _ = srv.accept()
        fs = MinerFramedSocket(conn)
        try:
            reg = fs.recv_msg()
            assert reg.message_type == "register" and reg.worker_id.endswith(":gpu6")
            assert reg.backend == "cuda"

            fs.send_msg(StripeAssignMessage(
                stripe_id="s1", prng_type="java_lcg", family_name="java_lcg",
                seed_start=0, seed_count=100))
            r1 = fs.recv_msg()
            assert r1.message_type == "sub_stripe_result" and r1.survivor_count == 1
            assert r1.inline and r1.inline["survivors"][0][0] == 0
            r2 = fs.recv_msg()
            assert r2.message_type == "stripe_complete" and r2.substripes_done == 1

            fs.send_msg(StripeAssignMessage(
                stripe_id="s2", prng_type="java_lcg", family_name="java_lcg",
                seed_start=0, seed_count=100, payload={"force_fail": True}))
            e1 = fs.recv_msg()
            assert e1.message_type == "stripe_error" and e1.retryable is True
            assert e1.traceback and "synthetic sub-stripe failure" in e1.error

            fs.send_msg(MinerStatusMessage())
            st = fs.recv_msg()
            assert st.message_type == "status" and st.stats["stripes_error"] >= 1
            assert st.stats["stripes_done"] >= 1

            fs.send_msg(MinerShutdownMessage())
            t.join(timeout=5)
            assert not t.is_alive()
        finally:
            fs.close()
            srv.close()
    assert "err" not in err_box, err_box.get("err")


# ---------------------------------------------------------------------------
# GATE 6 — uncovered family launches no kernel
# ---------------------------------------------------------------------------
def gate6_uncovered_launches_no_kernel():
    ex = SieveExecutor(resolver=_dummy_resolver([1, 2, 3]), device_index=0)
    ex._gpu_launch = mock.Mock(name="_gpu_launch")
    assign = StripeAssignMessage(
        stripe_id="u1", prng_type="mt19937", family_name="mt19937",
        seed_start=0, seed_count=100)
    try:
        ex.execute(assign, 0, 100)
    except NotImplementedError as e:
        assert "mt19937" in str(e)
    else:
        raise AssertionError("uncovered family must raise")
    ex._gpu_launch.assert_not_called()


# ---------------------------------------------------------------------------
# GATE 7 — GPU smoke (skippable)
# ---------------------------------------------------------------------------
def _cupy_device_or_skip():
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() == 0:
            return None
        return cp
    except Exception:
        return None


def gate7_gpu_smoke():
    cp = _cupy_device_or_skip()
    if cp is None:
        print(f"    [{_SKIP}] no CUDA/ROCm device — GPU smoke skipped")
        return
    draws = [123456, 234567, 345678, 456789, 567890, 678901, 789012, 890123, 901234, 12345]
    ex = SieveExecutor(resolver=_dummy_resolver(draws), device_index=0)
    assign = StripeAssignMessage(
        stripe_id="smoke", prng_type="java_lcg", family_name="java_lcg",
        seed_start=0, seed_count=256,
        payload={"dataset": "x", "window_size": 10,
                 "skip_range": [0, 16], "min_match_threshold": 0.25})
    outcome = ex.execute(assign, 0, 256)
    assert isinstance(outcome, SubStripeOutcome)
    assert isinstance(outcome.count, int) and outcome.count == len(outcome.survivors)
    for s in outcome.survivors:
        assert len(s) == 4 and isinstance(s[0], int) and isinstance(s[1], float)
        assert s[2] is None and isinstance(s[3], list)
    print(f"    java_lcg smoke: {outcome.count} survivors over 256 seeds")

    # rev-3: exercise a REVERSE-CONSTANT variant on hardware so a 12-vs-14 arg
    # error would actually surface (CuPy raises on an arg-count/type mismatch),
    # not just in a CPU shape assertion.
    ex_r = SieveExecutor(resolver=_dummy_resolver(draws), device_index=0)
    assign_r = StripeAssignMessage(
        stripe_id="smoke_rev", prng_type="java_lcg", family_name="java_lcg_reverse",
        seed_start=0, seed_count=256,
        payload={"dataset": "x", "window_size": 10,
                 "skip_range": [0, 16], "min_match_threshold": 0.25})
    out_r = ex_r.execute(assign_r, 0, 256)
    assert isinstance(out_r, SubStripeOutcome)
    assert out_r.count == len(out_r.survivors)
    print(f"    java_lcg_reverse smoke: {out_r.count} survivors over 256 seeds")


# ---------------------------------------------------------------------------
# GATE 8 — Phase 0/1/2 non-regression
# ---------------------------------------------------------------------------
def _run_harness(rel_path):
    result = subprocess.run(
        [sys.executable, rel_path],
        cwd=_ROOT, env={**os.environ, "PYTHONPATH": _ROOT},
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=600,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"{rel_path} exited {result.returncode}\n"
            f"stdout: {result.stdout[-800:]}\nstderr: {result.stderr[-400:]}")


def gate8_phase012_still_green():
    _run_harness(os.path.join("tests", "test_prng_encoding.py"))
    _run_harness(os.path.join("tests", "test_s172_phase1_scaffolding.py"))
    _run_harness(os.path.join("tests", "test_s172_phase2_protocol.py"))


# ---------------------------------------------------------------------------
# GATE 9 — [B1] per-assignment residue window
# ---------------------------------------------------------------------------
def gate9_per_assignment_window():
    calls = []

    def fake_loader(dataset, window_size, sessions, offset):
        calls.append((dataset, window_size, tuple(sessions or ()), offset))
        # Distinct residues per window identity so we can prove the right set is used.
        return [offset * 1000 + i for i in range(window_size)]

    resolver = ResidueResolver(loader=fake_loader, file_hasher=lambda p: "sha-" + p)

    r1 = resolver.resolve({"dataset": "ds.json", "window_size": 5,
                           "sessions": ["evening", "midday"], "offset": 0})
    r2 = resolver.resolve({"dataset": "ds.json", "window_size": 7,
                           "sessions": ["evening", "midday"], "offset": 3})
    assert r1 == [0, 1, 2, 3, 4]
    assert r2 == [3000, 3001, 3002, 3003, 3004, 3005, 3006]
    assert r1 != r2, "two windows must yield different residues"
    # sessions canonicalized (sorted) before entering the loader/key
    assert calls[0][2] == ("evening", "midday")

    # same identity -> cached, loader NOT called again
    before = len(calls)
    r1b = resolver.resolve({"dataset": "ds.json", "window_size": 5,
                            "sessions": ["midday", "evening"], "offset": 0})
    assert r1b == r1 and len(calls) == before, "cache miss on identical window"

    # residue_sha256 verification: mismatch is non-retryable
    good = sha256_residues([10, 20, 30])
    ok = ResidueResolver(loader=lambda *a: [10, 20, 30], file_hasher=lambda p: "h")
    assert ok.resolve({"dataset": "d", "window_size": 3, "residue_sha256": good}) \
        == [10, 20, 30]
    bad = ResidueResolver(loader=lambda *a: [1, 2, 3], file_hasher=lambda p: "h")
    try:
        bad.resolve({"dataset": "d", "window_size": 3, "residue_sha256": good})
    except Exception as e:
        from miner.range_miner_worker import ResidueVerificationError
        assert isinstance(e, ResidueVerificationError)
    else:
        raise AssertionError("residue_sha256 mismatch must raise")

    # missing window fields -> non-retryable ResidueResolutionError
    try:
        resolver.resolve({"dataset": "d"})  # no window_size
    except ResidueResolutionError:
        pass
    else:
        raise AssertionError("missing window_size must raise ResidueResolutionError")

    # --- rev-3: drive TWO assignments through execute() (the assignment path) ---
    # Prove execute() requests a FRESH residue identity per assignment. Works on
    # both GPU (full run) and CPU-only (resolve records, then the cupy import
    # raises AFTER resolution — the identity was still requested per assignment).
    cp = _cupy_device_or_skip()
    recorded = []

    def rec_loader(dataset, window_size, sessions, offset):
        residues = [offset * 1000 + i for i in range(window_size)]
        recorded.append((dataset, window_size, tuple(sessions or ()), offset,
                         tuple(residues)))
        return residues

    ex = SieveExecutor(
        resolver=ResidueResolver(loader=rec_loader, file_hasher=lambda p: "sha-" + p),
        device_index=0)
    a1 = StripeAssignMessage(
        stripe_id="w1", prng_type="java_lcg", family_name="java_lcg",
        seed_start=0, seed_count=128,
        payload={"dataset": "ds", "window_size": 5, "offset": 0,
                 "sessions": ["evening", "midday"], "min_match_threshold": 0.25})
    a2 = StripeAssignMessage(
        stripe_id="w2", prng_type="java_lcg", family_name="java_lcg",
        seed_start=0, seed_count=128,
        payload={"dataset": "ds", "window_size": 7, "offset": 3,
                 "sessions": ["evening", "midday"], "min_match_threshold": 0.25})
    if cp is None:
        for a in (a1, a2):
            try:
                ex.execute(a, 0, 128)   # resolve records, then cupy import fails
            except Exception:
                pass
    else:
        o1 = ex.execute(a1, 0, 128)
        o2 = ex.execute(a2, 0, 128)
        assert isinstance(o1, SubStripeOutcome) and isinstance(o2, SubStripeOutcome)

    assert len(recorded) == 2, f"execute() must resolve once per assignment, got {recorded}"
    assert recorded[0][:4] != recorded[1][:4], "distinct residue identity per assignment"
    assert recorded[0][4] != recorded[1][4], "distinct residues per window"
    assert recorded[0][4] == (0, 1, 2, 3, 4) and recorded[1][4] == tuple(3000 + i for i in range(7))


# ---------------------------------------------------------------------------
# GATE 10 — [B2a] spooled result: file written, hash matches
# ---------------------------------------------------------------------------
def _worker_for_spool(spool_dir):
    return RangeMinerWorker(
        host="127.0.0.1", port=1, gpu_id=0, caps=VramCaps(),
        gpu_info=GpuInfo("cuda", "stub", 1), miner_output_dir=spool_dir,
        heartbeat_interval=999)


def gate10_spool_written():
    with tempfile.TemporaryDirectory() as spool:
        w = _worker_for_spool(spool)
        assign = StripeAssignMessage(stripe_id="sp1", family_name="java_lcg",
                                     seed_start=0, seed_count=100)
        sub = SubStripe(sub_index=2, seed_start=0, seed_count=100)
        outcome = SubStripeOutcome(survivors=[(i, 0.5, None, [1]) for i in range(20)],
                                   count=20)
        # Force spool by dropping the inline threshold below the message size.
        orig = W.INLINE_BYTE_LIMIT
        try:
            W.INLINE_BYTE_LIMIT = 10  # bytes — anything real spools
            msg = w._build_sub_result(assign, sub, outcome)
        finally:
            W.INLINE_BYTE_LIMIT = orig

        assert msg.inline is None, "spooled result must clear inline"
        assert msg.spool_path and os.path.isfile(msg.spool_path)
        raw = open(msg.spool_path, "rb").read()
        assert len(raw) == msg.size_bytes
        assert hashlib.sha256(raw).hexdigest() == msg.sha256
        # bytes are the canonical schema payload
        obj = json.loads(raw.decode("utf-8"))
        assert obj["schema_version"] == "s172_substripe_v1"
        assert obj["stripe_id"] == "sp1" and obj["sub_index"] == 2
        assert len(obj["survivors"]) == 20
        # size/sha are over the exact serialized bytes
        _, pb = build_substripe_payload_bytes("sp1", 2, 0, 100, outcome.survivors)
        assert msg.sha256 == hashlib.sha256(pb).hexdigest()


# ---------------------------------------------------------------------------
# GATE 11 — [B2b] size-based (not count-based) inline/spool selection
# ---------------------------------------------------------------------------
def gate11_size_based_selection():
    with tempfile.TemporaryDirectory() as spool:
        w = _worker_for_spool(spool)
        assign = StripeAssignMessage(stripe_id="sz", family_name="java_lcg_hybrid",
                                     seed_start=0, seed_count=100)

        # FEW survivors, each HUGE (long skip-sequence) -> big bytes, tiny count.
        big_seq = list(range(20000))
        few_huge = SubStripeOutcome(
            survivors=[(i, 0.9, 0, big_seq) for i in range(5)], count=5)
        # MANY survivors, each tiny -> small bytes, large count.
        many_tiny = SubStripeOutcome(
            survivors=[(i, 0.9, None, [1]) for i in range(5000)], count=5000)

        orig = W.INLINE_BYTE_LIMIT
        try:
            # threshold chosen between the two encoded sizes
            _, pb_big = build_substripe_payload_bytes("sz", 0, 0, 100, few_huge.survivors)
            _, pb_small = build_substripe_payload_bytes("sz", 0, 0, 100, many_tiny.survivors)
            assert len(pb_big) > len(pb_small), "test setup: big must encode larger"
            W.INLINE_BYTE_LIMIT = (len(pb_big) + len(pb_small)) // 2

            m_big = w._build_sub_result(
                assign, SubStripe(0, 0, 100), few_huge)
            m_small = w._build_sub_result(
                assign, SubStripe(1, 0, 100), many_tiny)
        finally:
            W.INLINE_BYTE_LIMIT = orig

        # Decision is by SIZE: the 5-survivor blob spools, the 5000-survivor blob inlines.
        assert m_big.inline is None and m_big.spool_path, "huge-few must spool (size)"
        assert m_big.survivor_count == 5
        assert m_small.inline is not None and not m_small.spool_path, \
            "tiny-many must inline despite the larger COUNT"
        assert m_small.survivor_count == 5000

        # --- rev-3: cross the REAL 64 MiB protocol cap, do NOT shrink the limit ---
        from miner.range_miner_protocol import MAX_FRAME_BYTES
        assert W.INLINE_BYTE_LIMIT == 48 * 1024 * 1024, "production limit unchanged"
        huge = list(range(200000))
        over_cap = SubStripeOutcome(
            survivors=[(i, 0.9, 0, huge) for i in range(60)], count=60)
        _, pb_over = build_substripe_payload_bytes("sz", 7, 0, 100, over_cap.survivors)
        assert len(pb_over) > MAX_FRAME_BYTES, (len(pb_over), MAX_FRAME_BYTES)

        # (B1) Production config: a genuinely >64 MiB inline candidate must SPOOL
        # and MUST NOT raise. Under the buggy rev-2 guard, message_to_bytes() here
        # raised ValueError and aborted stripe handling; the fix spools it instead.
        m_over = w._build_sub_result(assign, SubStripe(7, 0, 100), over_cap)
        assert m_over.inline is None and m_over.spool_path
        assert m_over.size_bytes == len(pb_over) and os.path.isfile(m_over.spool_path)

        # (B2) Exercise the ValueError-catch net specifically. RAISE the limit above
        # the frame cap (NOT lowering it) so the payload-size guard is bypassed and
        # framing hits encode_frame's 64 MiB ValueError, which must be treated as
        # "must spool" rather than propagating.
        orig2 = W.INLINE_BYTE_LIMIT
        try:
            W.INLINE_BYTE_LIMIT = 4 * MAX_FRAME_BYTES  # > payload, so guard is bypassed
            m_ve = w._build_sub_result(assign, SubStripe(8, 0, 100), over_cap)
        finally:
            W.INLINE_BYTE_LIMIT = orig2
        assert m_ve.inline is None and m_ve.spool_path, "ValueError-catch path must spool"
        assert m_ve.size_bytes == len(pb_over)


# ---------------------------------------------------------------------------
# GATE 12 — [B3] cleanup after exception (skippable-GPU)
# ---------------------------------------------------------------------------
def gate12_cleanup_after_exception():
    cp = _cupy_device_or_skip()
    if cp is None:
        print(f"    [{_SKIP}] no device — GPU cleanup-after-exception skipped")
        return
    draws = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ex = SieveExecutor(resolver=_dummy_resolver(draws), device_index=0)
    assign = StripeAssignMessage(
        stripe_id="x", prng_type="java_lcg", family_name="java_lcg",
        seed_start=0, seed_count=128,
        payload={"dataset": "x", "window_size": 10})

    spy = mock.Mock(wraps=W._best_effort_gpu_cleanup)
    with mock.patch.object(W, "_best_effort_gpu_cleanup", spy):
        # Force a launch failure AFTER allocation.
        ex._gpu_launch = mock.Mock(side_effect=RuntimeError("boom in launch"))
        try:
            ex.execute(assign, 0, 128)
        except RuntimeError as e:
            assert "boom in launch" in str(e)
        else:
            raise AssertionError("forced launch failure must propagate")
        assert spy.call_count >= 1, "cleanup must run on the exception path"

        # The SAME executor that survived the exception serves the next assignment
        # (rev-3: restore the real launch hook on `ex`, not a fresh instance).
        del ex._gpu_launch  # drop the instance override -> real class method
        out = ex.execute(assign, 0, 128)
        assert isinstance(out, SubStripeOutcome)
    print(f"    cleanup ran on exception; SAME executor's next assignment "
          f"produced {out.count} survivors")


# ---------------------------------------------------------------------------
# GATE 13 — [B4] exact capability advertisement
# ---------------------------------------------------------------------------
def gate13_capability_advertisement():
    variants = supported_variants()
    # concrete hybrid variants for every covered base are advertised
    for base in COVERED:
        for v in (base, base + "_reverse", base + "_hybrid", base + "_hybrid_reverse"):
            assert v in variants, f"missing advertised variant {v}"
    assert len(variants) == 24
    # no uncovered family (any variant) is claimed
    for fam in UNCOVERED:
        assert not any(v == fam or v.startswith(fam + "_") for v in variants), fam

    # the register message carries exactly this list
    w = RangeMinerWorker(host="h", port=1, gpu_id=3, caps=VramCaps(),
                         gpu_info=GpuInfo("rocm", "RX 6600", 8 * 1024**3),
                         miner_output_dir=tempfile.gettempdir())
    reg = w._build_register_message()
    assert reg.capabilities["supported_variants"] == variants
    assert "java_lcg_hybrid" in reg.capabilities["supported_variants"]
    assert "seed_caps" in reg.capabilities


# ---------------------------------------------------------------------------
# GATE 14 — [B4/§11.I] non-Java full-mode (test_both_modes) 4-phase dispatch
# ---------------------------------------------------------------------------
def gate14_non_java_full_mode():
    # §6.8 four-phase workflow for a base with test_both_modes=True.
    for base in ("lcg32", "minstd"):
        phases = [base, base + "_reverse", base + "_hybrid", base + "_hybrid_reverse"]
        expected_len = {
            phases[0]: 11 + (2 if base == "minstd" else 3) + 1,  # fwd constant + tail + offset
            phases[1]: 12,                                       # reverse constant = prefix+offset
            phases[2]: HYBRID_FWD_LEN[base],                     # forward hybrid
            phases[3]: 14,                                       # reverse hybrid
        }
        for fam in phases:
            builder = resolve_builder(fam)  # same base builder, variant-aware
            args = builder(_ctx(fam))
            assert len(args) == expected_len[fam], (fam, len(args), expected_len[fam])
            if is_hybrid_family(fam):
                assert "skip_sequences" in _buf_names(args), fam
                assert "best_skips" not in _buf_names(args), fam
            else:
                assert "best_skips" in _buf_names(args), fam
            # correct cap tier per phase
            cap = select_seed_cap("cuda", fam, VramCaps())
            assert cap == (2_500_000 if is_hybrid_family(fam) else 5_000_000), fam


def main():
    print("\nS172 Phase 3 worker-daemon acceptance harness (rev-3)")
    print("=" * 70)
    _check("Gate 1: builder registry coverage",                    gate1_builder_registry_coverage)
    _check("Gate 2: per-family arg-shapes + dtype materialization", gate2_arg_shapes)
    _check("Gate 3: per-family cap selection",                     gate3_cap_selection)
    _check("Gate 4: sub-stripe partitioning",                      gate4_partitioning)
    _check("Gate 5: handshake + round-trip on loopback socket",    gate5_handshake_roundtrip)
    _check("Gate 6: uncovered family launches no kernel",          gate6_uncovered_launches_no_kernel)
    _check("Gate 7: GPU smoke (skippable)",                        gate7_gpu_smoke)
    _check("Gate 8: Phase 0/1/2 harnesses still green",            gate8_phase012_still_green)
    _check("Gate 9: [B1] per-assignment residue window",           gate9_per_assignment_window)
    _check("Gate 10: [B2a] spooled result file + hash",            gate10_spool_written)
    _check("Gate 11: [B2b] size-based inline/spool selection",     gate11_size_based_selection)
    _check("Gate 12: [B3] cleanup after exception (skippable)",    gate12_cleanup_after_exception)
    _check("Gate 13: [B4] exact capability advertisement",         gate13_capability_advertisement)
    _check("Gate 14: [B4/§11.I] non-Java full-mode dispatch",      gate14_non_java_full_mode)
    print("=" * 70)

    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{passed}/{total} gates green")
    print("NOTE: a CPU-only green validates the CONTRACT (arg shapes, protocol, "
          "spool, capability),\n      NOT ROCm deploy-readiness — that is Phase 6 "
          "acceptance on real rigs.")
    if passed != total:
        print("\nFAILURES (DO NOT COMMIT):")
        for name, ok, tb in _results:
            if not ok:
                print(f"\n--- {name} ---\n{tb}")
        sys.exit(1)
    print("\nAll gates green — Phase 3 rev-3 worker daemon is contract-validated.")
    sys.exit(0)


if __name__ == "__main__":
    main()
