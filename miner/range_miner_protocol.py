"""
miner/range_miner_protocol.py
=============================
S172 RANGE-MINER Phase 2 — Protocol definitions.
Spec: docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md (frozen at 1f6c0c5), §6, §8, §10.
Team Alpha implementation. Phase 0: 2389b61. Phase 1: 8d0183f.

Mirrors persistent/pwc_protocol.py structure:
  * dataclass messages with a BaseMessage envelope
  * factory from_dict() using dataclasses.fields() filter
    (TB blocker-A fix pattern — pwc_protocol.py:159-167)
  * unknown message_type hard-fails with ValueError
    (Phase 0 hard-fail pattern)

Wire format matches persistent/pwc_transport_tcp.py FramedSocket:
  4-byte big-endian length prefix + UTF-8 JSON body.
Framing helpers here are socket-free (pure bytes) so Phase 2 is testable
without a live TCP pair; Phase 3/4 wrap them around real sockets.

8 message types:
  register           worker -> coordinator   READY handshake (§6, Phase 3)
  stripe_assign      coordinator -> worker   60M-seed stripe lease
  sub_stripe_result  worker -> coordinator   per-sub-stripe survivors
  stripe_complete    worker -> coordinator   stripe finished
  stripe_error       worker -> coordinator   failure (one-retry, TB Q3)
  heartbeat          worker -> coordinator   liveness + counters
  shutdown           either direction        orderly stop
  status             either direction        state / progress query-report
"""
from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass, asdict, field, fields
from typing import Any, Dict, Optional, Tuple

MINER_PROTOCOL_VERSION = 1

# §8: miner default port 5700 (PWC default 5600) — OS-level coexistence.
DEFAULT_MINER_PORT = 5700

# Same 64 MB sanity cap as pwc_transport_tcp.FramedSocket.recv_obj.
MAX_FRAME_BYTES = 64 * 1024 * 1024

_HEADER = struct.Struct(">I")
HEADER_SIZE = _HEADER.size  # 4


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass
class MinerBaseMessage:
    # NOTE: all envelope fields carry defaults. Subclasses override
    # message_type with a per-type default; under dataclass inheritance a
    # non-default field (e.g. bare `worker_id: str`) following that override
    # is a TypeError at class-definition time on every CPython >= 3.7.
    # (The public-clone copy of persistent/pwc_protocol.py has that exact
    # bug and cannot import as published — do not copy its envelope verbatim.)
    message_type: str = ""
    worker_id: str = ""
    timestamp: float = field(default_factory=time.time)
    protocol_version: int = MINER_PROTOCOL_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_bytes(self) -> bytes:
        """Serialize to a length-prefixed JSON frame (wire format)."""
        return encode_frame(self.to_dict())


# ---------------------------------------------------------------------------
# Handshake
# ---------------------------------------------------------------------------

@dataclass
class RegisterMessage(MinerBaseMessage):
    """Worker READY handshake. worker_id = '{hostname}:gpu{gpu_id}'
    (infrastructure-neutral, hostname-based per S172_INFRASTRUCTURE_INTERFACE)."""
    message_type: str = "register"
    hostname: str = ""
    gpu_id: int = -1
    gpu_name: str = ""
    backend: str = ""                 # "rocm" | "cuda"
    vram_bytes: int = 0
    capabilities: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Stripe flow
# ---------------------------------------------------------------------------

@dataclass
class StripeAssignMessage(MinerBaseMessage):
    """Coordinator leases one stripe (default 67,108,864 seeds) to a worker.
    Sub-stripe sizing honors per-family VRAM caps (TB Q2, §6.3)."""
    message_type: str = "stripe_assign"
    stripe_id: str = ""
    trial_number: int = -1
    seed_start: int = 0
    seed_count: int = 0
    substripes: int = 8
    prng_type: str = ""               # base family, e.g. "java_lcg"
    family_name: str = ""             # resolved kernel variant for this phase
    phase: int = 0                    # 1-4 per §6.8 4-phase workflow
    attempt: int = 0                  # 0 = first try; 1 = the single retry (TB Q3)
    payload: Optional[Dict[str, Any]] = None  # window cfg, thresholds, residue ref


@dataclass
class SubStripeResultMessage(MinerBaseMessage):
    """One sub-stripe's survivors. Large payloads spool to disk (path+sha256);
    small ones ride inline — mirrors PWC result_inline/result_spooled split."""
    message_type: str = "sub_stripe_result"
    stripe_id: str = ""
    sub_index: int = -1
    seed_start: int = 0
    seed_count: int = 0
    survivor_count: int = 0
    inline: Optional[Dict[str, Any]] = None
    spool_path: str = ""
    size_bytes: int = 0
    sha256: str = ""
    # [S172 D6 correction] EFFECTIVE-threshold provenance. The value the kernel
    # ACTUALLY filtered at for this sub-stripe (constant kernels: the payload's
    # min_match_threshold; hybrid kernels: the payload's phase2_threshold, which
    # D6 pins equal to it). WindowConfig alone is not evidence the requested
    # value reached execution — this field is. Defaulted (like every envelope
    # field) so a pre-D6 peer that never sets it still decodes.
    effective_threshold: Optional[float] = None


@dataclass
class StripeCompleteMessage(MinerBaseMessage):
    message_type: str = "stripe_complete"
    stripe_id: str = ""
    substripes_done: int = 0
    survivors_total: int = 0
    # [S172 elapsed_s persistence, Beta R4] Worker-measured stripe SERVICE TIME in
    # seconds. `None` (not 0.0) is the "peer did not report it" value, matching the
    # `effective_threshold` idiom below: a pre-R4 peer omits the key, from_dict()
    # applies this default, and the ledger stores NULL — which a genuine 0.0 must
    # remain distinguishable from. Defaulted like every envelope field, so an older
    # peer still decodes.
    elapsed_s: Optional[float] = None
    # [S172 D6 correction] the effective threshold every sub-stripe of THIS
    # stripe filtered at. None when the stripe ran no sub-stripe; a stripe whose
    # sub-stripes disagreed is a defect and reports the disagreement explicitly
    # (the worker refuses to average or pick one).
    effective_threshold: Optional[float] = None


@dataclass
class StripeErrorMessage(MinerBaseMessage):
    """Sub-stripe/stripe failure. Coordinator applies one-retry-then-fail-trial
    (TB Q3, §12.3). retryable=False short-circuits straight to trial failure."""
    message_type: str = "stripe_error"
    stripe_id: str = ""
    sub_index: int = -1
    error: str = ""
    traceback: str = ""
    retryable: bool = True


# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------

@dataclass
class MinerHeartbeatMessage(MinerBaseMessage):
    message_type: str = "heartbeat"
    stripes_done: int = 0
    stripes_error: int = 0
    current_stripe_id: str = ""
    busy: bool = False


@dataclass
class MinerShutdownMessage(MinerBaseMessage):
    message_type: str = "shutdown"
    reason: str = "coordinator_request"


@dataclass
class MinerStatusMessage(MinerBaseMessage):
    """Bidirectional: empty-state instance = query; populated = report."""
    message_type: str = "status"
    state: str = ""                   # "idle" | "mining" | "draining" | ...
    current_stripe_id: str = ""
    sub_index: int = -1
    progress: float = 0.0             # 0.0-1.0 within current stripe
    stats: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Factory (TB blocker-A fields() filter pattern)
# ---------------------------------------------------------------------------

_MINER_MSG_TYPES = {
    "register":          RegisterMessage,
    "stripe_assign":     StripeAssignMessage,
    "sub_stripe_result": SubStripeResultMessage,
    "stripe_complete":   StripeCompleteMessage,
    "stripe_error":      StripeErrorMessage,
    "heartbeat":         MinerHeartbeatMessage,
    "shutdown":          MinerShutdownMessage,
    "status":            MinerStatusMessage,
}


def from_dict(d: Dict[str, Any]) -> MinerBaseMessage:
    """Reconstruct a typed message from a raw dict.

    Hard-fails (ValueError) on unknown message_type — Phase 0 pattern.
    Unknown keys are dropped via dataclasses.fields() filter so a newer
    peer with extra fields never crashes an older one (TB blocker-A fix).
    """
    mtype = d.get("message_type", "")
    cls = _MINER_MSG_TYPES.get(mtype)
    if cls is None:
        raise ValueError(f"Unknown miner message_type: {mtype!r}")
    known = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in d.items() if k in known}
    return cls(**filtered)


# ---------------------------------------------------------------------------
# Framing — wire-identical to persistent/pwc_transport_tcp.FramedSocket
# ---------------------------------------------------------------------------

def encode_frame(obj: Dict[str, Any]) -> bytes:
    """dict -> 4-byte big-endian length header + compact UTF-8 JSON body."""
    body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    if len(body) > MAX_FRAME_BYTES:
        raise ValueError(f"oversized message: {len(body)} bytes")
    return _HEADER.pack(len(body)) + body


def decode_frame(data: bytes, offset: int = 0) -> Tuple[Dict[str, Any], int]:
    """Decode one frame from a byte buffer starting at offset.

    Returns (obj, next_offset). Raises ValueError on truncated header,
    truncated body, or oversized frame — clean failure, never a partial dict.
    """
    if len(data) - offset < HEADER_SIZE:
        raise ValueError("truncated frame: incomplete 4-byte length header")
    (size,) = _HEADER.unpack_from(data, offset)
    if size > MAX_FRAME_BYTES:
        raise ValueError(f"oversized message: {size} bytes")
    body_start = offset + HEADER_SIZE
    body_end = body_start + size
    if len(data) < body_end:
        raise ValueError(
            f"truncated frame: header declares {size} bytes, "
            f"only {len(data) - body_start} available"
        )
    return json.loads(data[body_start:body_end].decode("utf-8")), body_end


def message_to_bytes(msg: MinerBaseMessage) -> bytes:
    """Typed message -> wire frame."""
    return encode_frame(msg.to_dict())


def message_from_bytes(data: bytes, offset: int = 0) -> Tuple[MinerBaseMessage, int]:
    """Wire frame -> typed message. ValueError on unknown type or bad frame."""
    obj, next_offset = decode_frame(data, offset)
    return from_dict(obj), next_offset
