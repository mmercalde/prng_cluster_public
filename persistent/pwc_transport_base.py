"""
persistent/pwc_transport_base.py
==================================
PWC Transport Adapter v1 — Transport base class + factory.
TB-approved S159G proposal. Team Alpha implementation.

PWCTransportBase lives here. Both SSH and TCP backends import from here.
build_transport() is the factory called by the coordinator.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class PWCTransportBase:
    """
    Abstract base for transport backends.
    Both SSH and TCP implement this interface so coordinator code
    is fully transport-blind.
    """

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def submit_job(self, job: Dict[str, Any]) -> None:
        raise NotImplementedError

    def recv_result(self, timeout_s: Optional[float] = None) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def worker_count(self) -> int:
        raise NotImplementedError


def build_transport(pwc_transport: str, **kwargs) -> Optional[PWCTransportBase]:
    # [PWC-BACKUP S4, Beta X-PWC-BACKUP R1 Q3] no default: an omitted transport
    # used to return the SSH sentinel silently. It is now a TypeError.
    """
    Factory. Returns TCPWorkerTransport for tcp mode, None for ssh (legacy).

    Args:
        pwc_transport: "ssh" (legacy) or "tcp". Required; anything else raises.
        **kwargs: passed to transport constructor

    Returns:
        TCPWorkerTransport for tcp mode
        None for ssh mode — caller uses existing PWC SSH path unchanged
    """
    if pwc_transport == "tcp":
        from persistent.pwc_transport_tcp import TCPWorkerTransport
        host = kwargs.get("pwc_host", "0.0.0.0")
        port = int(kwargs.get("pwc_port", 5600))
        return TCPWorkerTransport(host=host, port=port)

    # SSH mode — legacy path unchanged
    if pwc_transport == "ssh":
        return None

    # [PWC-BACKUP S4] Any other value used to fall through to `return None`, i.e.
    # silently to the legacy SSH path (the S170-PARITY-1 class). Refuse instead.
    raise ValueError(
        f"pwc_transport={pwc_transport!r} is not 'ssh' or 'tcp'; refusing rather "
        f"than falling back to the legacy SSH transport")
