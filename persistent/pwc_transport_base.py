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


def build_transport(pwc_transport: str = "ssh", **kwargs) -> Optional[PWCTransportBase]:
    """
    Factory. Returns TCPWorkerTransport for tcp mode, None for ssh (legacy).

    Args:
        pwc_transport: "ssh" (default/legacy) or "tcp" (new)
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
    return None
