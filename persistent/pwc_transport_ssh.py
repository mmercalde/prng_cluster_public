"""
persistent/pwc_transport_ssh.py
================================
PWC Transport Adapter v1 — SSH pipe transport (legacy/default).
TB-approved S159G proposal. Team Alpha implementation.

Thin wrapper preserving existing PWC SSH behavior exactly.
Exists so coordinator code treats SSH and TCP identically via PWCTransportBase.
The actual SSH launch and stdin/stdout pipe logic remains in
persistent_worker_coordinator.py — this module just declares the interface.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from persistent.pwc_transport_base import PWCTransportBase


class SSHPipeTransport(PWCTransportBase):
    """
    Legacy SSH pipe transport.
    Delegates entirely to existing PWC SSH infrastructure.
    This is the default transport — behavior is completely unchanged.
    """

    def start(self) -> None:
        # SSH workers launched by PersistentWorkerCoordinator._spawn_worker()
        pass

    def stop(self) -> None:
        # SSH workers shut down by PersistentWorkerCoordinator.shutdown()
        pass

    def submit_job(self, job: Dict[str, Any]) -> None:
        # SSH transport uses direct stdin write in _dispatch_job()
        raise NotImplementedError("SSH transport uses direct dispatch, not submit_job()")

    def recv_result(self, timeout_s: Optional[float] = None) -> Optional[Dict[str, Any]]:
        # SSH transport reads directly from stdout in _dispatch_job()
        raise NotImplementedError("SSH transport uses direct dispatch, not recv_result()")

    def worker_count(self) -> int:
        # SSH workers tracked by PersistentWorkerCoordinator._worker_handles
        return -1  # unknown without coordinator reference
