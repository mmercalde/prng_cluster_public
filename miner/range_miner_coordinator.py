"""
range_miner_coordinator.py — S172 RANGE-MINER stripe coordinator

Phase 1 status: STUB. Implementation lands in Phase 4 per v1.4.4 §10.

This module exposes `run_trial_miner(...)` so the integration gate at
window_optimizer_integration_final.py:_use_miner can import it symmetrically
with `run_trial_persistent` (PWC) and `run_trial_zmq_sqlite` (ZMQ). Calling
it in Phase 1 raises NotImplementedError immediately — no silent behavior.

Phase 4 will implement:
  - Stripe assignment across READY workers (§6.3, §6.5)
  - Per-family VRAM caps (§6.4, TB Q2)
  - Fail-closed Phase 1/2 (§6.6)
  - One-retry-then-fail-trial Phase 3/4 (§12.3, TB Q3)
  - EXPECTED_NPZ_KEYS contract wall on emit (§12.1)

INFRASTRUCTURE-NEUTRAL DESIGN (S172_INFRASTRUCTURE_INTERFACE_v1_0):
  This module identifies workers by socket.gethostname() (Proxmox-container-
  transparent) and writes NPZs to a configurable output directory
  (--miner-output-dir), defaulting to /dev/shm/prng/miner/ when writable
  (LXC ramdisk-bind case) with fallback to ~/miner_output/ (VM/bare metal).
"""

from typing import Any, Dict


def run_trial_miner(
    coordinator_cfg: str,
    config,
    trial_number: int,
    prng_base: str,
    residues,
    total_seeds: int,
    forward_threshold: float,
    reverse_threshold: float,
    test_both_modes: bool,
    dataset_path: str,
    worker_pool_size: int = 8,
    seed_cap_nvidia: int = 5_000_000,
    seed_cap_amd: int = 2_000_000,
    miner_stripe_size: int = 67_108_864,
    miner_substripes: int = 8,
    miner_output_dir: str = None,
    node_allowlist=None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Stripe-based Step 1 backend (S172 RANGE-MINER).

    Phase 1: not implemented. Raises NotImplementedError.

    Signature mirrors run_trial_persistent (persistent_worker_coordinator.py:
    run_trial_persistent) for drop-in integration at
    window_optimizer_integration_final.py:_use_miner gate.
    """
    raise NotImplementedError(
        "S172 RANGE-MINER Phase 1 scaffolding is present but the coordinator, "
        "worker daemon, protocol, and NPZ contract wall are not yet implemented "
        "(Phases 2-5). See docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md §10 for the "
        "phase plan. Do not enable --use-range-miner in production until Phase "
        "7 acceptance completes."
    )
