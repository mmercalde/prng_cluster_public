#!/usr/bin/env python3
"""
S170-PARITY-2: config-mode execution parameter parity.

Fixes window_optimizer.py --config-file mode silently ignoring:
  --seed-cap-amd
  --seed-cap-nvidia
  --worker-pool-size
  --min-workers

The first parity patch fixed transport. This patch fixes execution sizing.
"""

import ast
import sys
from pathlib import Path

MARKER_SIG = "# [S170-PARITY-2] execution sizing"
MARKER_BODY = "# [S170-PARITY-2] propagate execution sizing"
MARKER_CALL = "# [S170-PARITY-2] CLI execution sizing passthrough"

def fail(msg, code=1):
    print(f"ERROR: {msg}")
    sys.exit(code)

def replace_once(src, old, new, label):
    count = src.count(old)
    if count != 1:
        fail(f"{label} anchor count={count}, expected 1", 10)
    return src.replace(old, new, 1)

def main():
    if len(sys.argv) != 2:
        fail("Usage: python3 patch_config_mode_exec_parity.py window_optimizer.py")

    path = Path(sys.argv[1])
    if not path.exists():
        fail(f"target not found: {path}")

    src = path.read_text()

    if MARKER_SIG in src and MARKER_BODY in src and MARKER_CALL in src:
        print("[S170-PARITY-2] already applied — no changes")
        return

    # Requires S170-PARITY-1 to already be present.
    if "use_persistent_workers: bool = False" not in src or "pwc_transport: str = 'tcp'" not in src:
        fail("S170-PARITY transport patch not found; apply first parity patch before this one", 2)

    sig_old = """    output_holdout: str = 'holdout_history.json',
    use_persistent_workers: bool = False,   # [S170-PARITY] use_persistent_workers
    pwc_transport: str = 'tcp',             # [S170-PARITY] use_persistent_workers
) -> Dict[str, Any]:"""

    sig_new = """    output_holdout: str = 'holdout_history.json',
    use_persistent_workers: bool = False,   # [S170-PARITY] use_persistent_workers
    pwc_transport: str = 'tcp',             # [S170-PARITY] use_persistent_workers
    seed_cap_amd: int = 2_000_000,          # [S170-PARITY-2] execution sizing
    seed_cap_nvidia: int = 5_000_000,       # [S170-PARITY-2] execution sizing
    worker_pool_size: int = 8,              # [S170-PARITY-2] execution sizing
    min_workers: int = 1,                   # [S170-PARITY-2] execution sizing
) -> Dict[str, Any]:"""

    src = replace_once(src, sig_old, sig_new, "signature")

    body_old = """    coordinator.use_persistent_workers = use_persistent_workers
    coordinator.pwc_transport          = pwc_transport

    # Create WindowConfig object"""

    body_new = """    coordinator.use_persistent_workers = use_persistent_workers
    coordinator.pwc_transport          = pwc_transport

    # [S170-PARITY-2] propagate execution sizing — match Bayesian/PWC path
    # Without these, --config-file mode silently falls back to default chunk caps
    # such as seed_cap_amd=2_000_000 despite CLI --seed-cap-amd 100000.
    coordinator.seed_cap_amd           = seed_cap_amd
    coordinator.seed_cap_nvidia        = seed_cap_nvidia
    coordinator.worker_pool_size       = worker_pool_size
    coordinator.min_workers            = min_workers

    # Create WindowConfig object"""

    src = replace_once(src, body_old, body_new, "body")

    call_old = """            use_persistent_workers=getattr(args, 'use_persistent_workers', False),
            pwc_transport=getattr(args, 'pwc_transport', 'tcp'),
        )"""

    call_new = """            use_persistent_workers=getattr(args, 'use_persistent_workers', False),
            pwc_transport=getattr(args, 'pwc_transport', 'tcp'),
            # [S170-PARITY-2] CLI execution sizing passthrough
            seed_cap_amd=getattr(args, 'seed_cap_amd', 2_000_000),
            seed_cap_nvidia=getattr(args, 'seed_cap_nvidia', 5_000_000),
            worker_pool_size=getattr(args, 'worker_pool_size', 8),
            min_workers=getattr(args, 'min_workers', 1),
        )"""

    src = replace_once(src, call_old, call_new, "call site")

    ast.parse(src)
    path.write_text(src)

    print("[S170-PARITY-2] patched window_optimizer.py")
    print("  + run_with_config signature: seed caps / worker sizing")
    print("  + coordinator attributes: seed_cap_amd/nvidia, worker_pool_size, min_workers")
    print("  + config-file call site: CLI passthrough")

if __name__ == "__main__":
    main()
