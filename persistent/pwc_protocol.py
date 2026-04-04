"""
persistent/pwc_protocol.py
==========================
PWC Transport Adapter v1 — Protocol definitions.
TB-approved S159G proposal. Team Alpha implementation.

All TCP messages are length-prefixed:
  4-byte big-endian length + UTF-8 JSON body
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

PROTOCOL_VERSION = 1


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass
class BaseMessage:
    message_type: str
    worker_id: str
    timestamp: float = field(default_factory=time.time)
    protocol_version: int = PROTOCOL_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Handshake
# ---------------------------------------------------------------------------

@dataclass
class HelloMessage(BaseMessage):
    message_type: str = "hello"
    hostname: str = ""
    gpu_id: int = -1
    transport: str = "tcp"
    capabilities: Optional[Dict[str, Any]] = None


@dataclass
class HelloAckMessage(BaseMessage):
    message_type: str = "hello_ack"
    accepted: bool = True
    reason: str = ""


# ---------------------------------------------------------------------------
# Job flow
# ---------------------------------------------------------------------------

@dataclass
class RequestJobMessage(BaseMessage):
    message_type: str = "request_job"
    idle: bool = True


@dataclass
class JobAssignMessage(BaseMessage):
    message_type: str = "job_assign"
    job_id: str = ""
    lease_id: str = ""
    attempt: int = 0
    payload: Optional[Dict[str, Any]] = None


@dataclass
class JobAckMessage(BaseMessage):
    message_type: str = "job_ack"
    job_id: str = ""
    lease_id: str = ""


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class ResultInlineMessage(BaseMessage):
    message_type: str = "result_inline"
    job_id: str = ""
    lease_id: str = ""
    attempt: int = 0
    result: Optional[Dict[str, Any]] = None


@dataclass
class ResultSpooledMessage(BaseMessage):
    message_type: str = "result_spooled"
    job_id: str = ""
    lease_id: str = ""
    attempt: int = 0
    spool_path: str = ""
    size_bytes: int = 0
    sha256: str = ""
    summary: Optional[Dict[str, Any]] = None


@dataclass
class JobCompleteMessage(BaseMessage):
    message_type: str = "job_complete"
    job_id: str = ""
    lease_id: str = ""


@dataclass
class JobErrorMessage(BaseMessage):
    message_type: str = "job_error"
    job_id: str = ""
    lease_id: str = ""
    error: str = ""
    traceback: str = ""


# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------

@dataclass
class HeartbeatMessage(BaseMessage):
    message_type: str = "heartbeat"
    jobs_done: int = 0
    jobs_error: int = 0


@dataclass
class ShutdownMessage(BaseMessage):
    message_type: str = "shutdown"
    reason: str = "coordinator_request"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_MSG_TYPES = {
    "hello":          HelloMessage,
    "hello_ack":      HelloAckMessage,
    "request_job":    RequestJobMessage,
    "job_assign":     JobAssignMessage,
    "job_ack":        JobAckMessage,
    "result_inline":  ResultInlineMessage,
    "result_spooled": ResultSpooledMessage,
    "job_complete":   JobCompleteMessage,
    "job_error":      JobErrorMessage,
    "heartbeat":      HeartbeatMessage,
    "shutdown":       ShutdownMessage,
}


def from_dict(d: Dict[str, Any]) -> BaseMessage:
    """Reconstruct a typed message from a raw dict."""
    from dataclasses import fields
    mtype = d.get("message_type", "")
    cls = _MSG_TYPES.get(mtype)
    if cls is None:
        raise ValueError(f"Unknown message_type: {mtype!r}")
    # Only pass fields the dataclass knows about
    known = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in d.items() if k in known}
    return cls(**filtered)
