"""
miner — S172 RANGE-MINER (Step 1 replacement, v1.4.4 spec)

Phase 1 scaffolding: package structure, argparse gate, integration hook.
No GPU code, no protocol, no worker daemon yet (Phases 2-4).

Module layout (per v1.4.4 §3, §12):
    miner/
        __init__.py                    ← this file
        range_miner_coordinator.py     ← stripe assignment (Phase 4)
        range_miner_worker.py          ← per-GPU daemon    (Phase 3)
        range_miner_protocol.py        ← 8 message types    (Phase 2)
        range_miner_npz_writer.py      ← contract wall     (Phase 5, §12.1)

Phase 1 exports `run_trial_miner` as a callable stub so the integration
gate at window_optimizer_integration_final.py can import it without
raising, matching the pattern of the existing PWC / ZMQ runners.
"""
from .range_miner_coordinator import (
    DEFAULT_WORKER_ADMISSION_TIMEOUT,
    run_trial_miner,
)

__all__ = ["run_trial_miner", "DEFAULT_WORKER_ADMISSION_TIMEOUT"]
