#!/usr/bin/env python3
"""
test_s172_phase2_protocol.py — S172 Phase 2 protocol acceptance harness

Phase 2 is miner/range_miner_protocol.py: 8 dataclass message types +
length-prefixed JSON framing wire-identical to persistent/pwc_transport_tcp.py.
No GPU work, no sockets — pure serialization contract.

Gates (all block-on-failure):
  1. All 8 message types construct with their required fields and expose
     the MinerBaseMessage envelope (message_type, worker_id, timestamp,
     protocol_version).
  2. Round-trip: msg -> wire bytes -> msg reproduces field-equal messages
     for all 8 types, including nested payload dicts.
  3. Unknown message_type in a decoded stream raises ValueError
     (Phase 0 hard-fail pattern).
  4. Truncated frames (partial header AND partial body) raise ValueError
     cleanly; oversized declared length raises ValueError.
  5. dataclasses.fields() filter accepts extra unknown kwargs safely
     (TB blocker-A pattern) — forward-compat with newer peers.
  6. Non-regression: Phase 0 (tests/test_prng_encoding.py) and Phase 1
     (tests/test_s172_phase1_scaffolding.py) harnesses still pass.

Run:
    cd ~/distributed_prng_analysis
    PYTHONPATH=. python3 tests/test_s172_phase2_protocol.py

Exit code 0 = all gates green (Phase 2 shippable).
Exit code 1 = a gate failed (DO NOT COMMIT).
"""
import os
import struct
import subprocess
import sys
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"

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
    MINER_PROTOCOL_VERSION,
    DEFAULT_MINER_PORT,
    HEADER_SIZE,
    MAX_FRAME_BYTES,
    MinerBaseMessage,
    RegisterMessage,
    StripeAssignMessage,
    SubStripeResultMessage,
    StripeCompleteMessage,
    StripeErrorMessage,
    MinerHeartbeatMessage,
    MinerShutdownMessage,
    MinerStatusMessage,
    _MINER_MSG_TYPES,
    from_dict,
    encode_frame,
    decode_frame,
    message_to_bytes,
    message_from_bytes,
)


def _sample_messages():
    """One populated instance of each of the 8 types."""
    wid = "rrig6600c:gpu6"
    return [
        RegisterMessage(worker_id=wid, hostname="rrig6600c", gpu_id=6,
                        gpu_name="AMD Radeon RX 6600", backend="rocm",
                        vram_bytes=8 * 1024**3,
                        capabilities={"seed_cap": 2_000_000,
                                      "seed_cap_hybrid": 1_000_000}),
        StripeAssignMessage(worker_id=wid, stripe_id="t0042-s0007",
                            trial_number=42, seed_start=469_762_048,
                            seed_count=67_108_864, substripes=8,
                            prng_type="java_lcg",
                            family_name="java_lcg_hybrid_reverse", phase=4,
                            attempt=0,
                            payload={"window_size": 512, "offset": 3,
                                     "skip_min": 0, "skip_max": 20,
                                     "forward_threshold": 0.35}),
        SubStripeResultMessage(worker_id=wid, stripe_id="t0042-s0007",
                               sub_index=3, seed_start=494_927_872,
                               seed_count=8_388_608, survivor_count=17,
                               inline={"seeds": [1, 2, 3],
                                       "forward_matches": [9, 9, 9]}),
        StripeCompleteMessage(worker_id=wid, stripe_id="t0042-s0007",
                              substripes_done=8, survivors_total=131,
                              elapsed_s=42.7),
        StripeErrorMessage(worker_id=wid, stripe_id="t0042-s0007",
                           sub_index=5, error="kernel launch failed",
                           traceback="Traceback ...", retryable=True),
        MinerHeartbeatMessage(worker_id=wid, stripes_done=12,
                              stripes_error=0,
                              current_stripe_id="t0042-s0008", busy=True),
        MinerShutdownMessage(worker_id=wid, reason="trial_complete"),
        MinerStatusMessage(worker_id=wid, state="mining",
                           current_stripe_id="t0042-s0008", sub_index=2,
                           progress=0.3125,
                           stats={"seeds_per_s": 1.9e7}),
    ]


# ---------------------------------------------------------------------------
# GATE 1 — all 8 types construct; envelope fields present; registry complete
# ---------------------------------------------------------------------------
def gate1_all_types_construct():
    msgs = _sample_messages()
    assert len(msgs) == 8, f"expected 8 sample messages, got {len(msgs)}"
    expected_types = {"register", "stripe_assign", "sub_stripe_result",
                      "stripe_complete", "stripe_error", "heartbeat",
                      "shutdown", "status"}
    seen = set()
    for m in msgs:
        assert isinstance(m, MinerBaseMessage)
        assert m.message_type in expected_types, m.message_type
        assert m.worker_id == "rrig6600c:gpu6"
        assert isinstance(m.timestamp, float) and m.timestamp > 0
        assert m.protocol_version == MINER_PROTOCOL_VERSION
        seen.add(m.message_type)
    assert seen == expected_types, f"missing types: {expected_types - seen}"
    # Factory registry covers exactly the 8 spec'd types
    assert set(_MINER_MSG_TYPES) == expected_types, set(_MINER_MSG_TYPES)
    # §8 coexistence constant
    assert DEFAULT_MINER_PORT == 5700, DEFAULT_MINER_PORT


# ---------------------------------------------------------------------------
# GATE 2 — round-trip msg -> wire bytes -> msg equality (all 8 types)
# ---------------------------------------------------------------------------
def gate2_round_trip():
    for m in _sample_messages():
        wire = message_to_bytes(m)
        # header sanity: declared length == body length
        (declared,) = struct.unpack(">I", wire[:HEADER_SIZE])
        assert declared == len(wire) - HEADER_SIZE
        back, consumed = message_from_bytes(wire)
        assert consumed == len(wire), (consumed, len(wire))
        assert type(back) is type(m), (type(back), type(m))
        assert back == m, f"round-trip inequality for {m.message_type}"
    # Multi-frame stream: all 8 concatenated, decoded in sequence
    originals = _sample_messages()
    stream = b"".join(message_to_bytes(m) for m in originals)
    offset, decoded = 0, []
    while offset < len(stream):
        msg, offset = message_from_bytes(stream, offset)
        decoded.append(msg)
    assert decoded == originals, "multi-frame stream round-trip inequality"


# ---------------------------------------------------------------------------
# GATE 3 — unknown message_type raises ValueError (Phase 0 hard-fail)
# ---------------------------------------------------------------------------
def gate3_unknown_type_hard_fails():
    bad = {"message_type": "job_assign",  # a PWC type — must NOT leak in
           "worker_id": "x", "timestamp": 0.0, "protocol_version": 1}
    for candidate in (bad,
                      {**bad, "message_type": "totally_bogus"},
                      {**bad, "message_type": ""},
                      {"worker_id": "x"}):  # message_type absent entirely
        wire = encode_frame(candidate)
        try:
            message_from_bytes(wire)
        except ValueError as e:
            assert "message_type" in str(e), str(e)
        else:
            raise AssertionError(
                f"unknown type {candidate.get('message_type')!r} did not raise")


# ---------------------------------------------------------------------------
# GATE 4 — truncated / oversized frames raise cleanly
# ---------------------------------------------------------------------------
def gate4_truncated_frame():
    wire = message_to_bytes(MinerHeartbeatMessage(worker_id="w"))
    # (a) partial header
    for cut in (0, 1, 2, 3):
        try:
            decode_frame(wire[:cut])
        except ValueError as e:
            assert "truncated" in str(e), str(e)
        else:
            raise AssertionError(f"partial header ({cut} bytes) did not raise")
    # (b) full header, partial body
    try:
        decode_frame(wire[:-5])
    except ValueError as e:
        assert "truncated" in str(e), str(e)
    else:
        raise AssertionError("partial body did not raise")
    # (c) oversized declared length (matches 64MB FramedSocket cap)
    evil = struct.pack(">I", MAX_FRAME_BYTES + 1) + b"{}"
    try:
        decode_frame(evil)
    except ValueError as e:
        assert "oversized" in str(e), str(e)
    else:
        raise AssertionError("oversized frame did not raise")
    # (d) intact frame still decodes after all that
    obj, consumed = decode_frame(wire)
    assert obj["message_type"] == "heartbeat" and consumed == len(wire)


# ---------------------------------------------------------------------------
# GATE 5 — fields() filter drops unknown kwargs safely (TB blocker-A)
# ---------------------------------------------------------------------------
def gate5_fields_filter_forward_compat():
    d = StripeAssignMessage(worker_id="w", stripe_id="s1",
                            seed_start=0, seed_count=100).to_dict()
    # A "newer" peer adds fields this version doesn't know about
    d["future_field"] = {"nested": True}
    d["another_new_thing"] = 123
    msg = from_dict(d)
    assert isinstance(msg, StripeAssignMessage)
    assert msg.stripe_id == "s1" and msg.seed_count == 100
    assert not hasattr(msg, "future_field")
    # Same via the wire path
    back, _ = message_from_bytes(encode_frame(d))
    assert back == msg


# ---------------------------------------------------------------------------
# GATE 6 — Phase 0 + Phase 1 non-regression
# ---------------------------------------------------------------------------
def _run_harness(rel_path):
    result = subprocess.run(
        [sys.executable, rel_path],
        cwd=_ROOT, env={**os.environ, "PYTHONPATH": _ROOT},
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"{rel_path} exited {result.returncode}\n"
            f"stdout: {result.stdout[-800:]}\nstderr: {result.stderr[-400:]}"
        )


def gate6_phase0_phase1_still_green():
    _run_harness(os.path.join("tests", "test_prng_encoding.py"))
    _run_harness(os.path.join("tests", "test_s172_phase1_scaffolding.py"))


def main():
    print("\nS172 Phase 2 protocol acceptance harness")
    print("=" * 66)
    _check("Gate 1: all 8 message types construct + envelope",   gate1_all_types_construct)
    _check("Gate 2: round-trip msg -> bytes -> msg equality",    gate2_round_trip)
    _check("Gate 3: unknown message_type raises ValueError",     gate3_unknown_type_hard_fails)
    _check("Gate 4: truncated/oversized frame raises cleanly",   gate4_truncated_frame)
    _check("Gate 5: fields() filter drops unknown kwargs",       gate5_fields_filter_forward_compat)
    _check("Gate 6: Phase 0 + Phase 1 harnesses still green",    gate6_phase0_phase1_still_green)
    print("=" * 66)

    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{passed}/{total} gates green")
    if passed != total:
        print("\nFAILURES (DO NOT COMMIT):")
        for name, ok, tb in _results:
            if not ok:
                print(f"\n--- {name} ---\n{tb}")
        sys.exit(1)
    print("\nAll gates green — Phase 2 protocol is deploy-ready.")
    sys.exit(0)


if __name__ == "__main__":
    main()
