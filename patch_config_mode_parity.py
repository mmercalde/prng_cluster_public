#!/usr/bin/env python3
"""
S170 Config-Mode Parity Patch
==============================

TB ruling (S170, 2026-04-25): APPROVE OPTION 1.

`window_optimizer.py --config-file` mode silently downgrades to legacy SSH
distribution because run_with_config() never propagates use_persistent_workers
and pwc_transport to the coordinator. This patch restores parity with the
Bayesian path.

Changes (3 surgical edits to window_optimizer.py):

  1. run_with_config() signature: add use_persistent_workers and pwc_transport
     kwargs with the same defaults the Bayesian search() uses.

  2. run_with_config() body: after MultiGPUCoordinator construction and
     add_window_optimizer_to_coordinator(), set the two attributes — same
     as Bayesian path lines 614-616.

  3. Call site (around line 1201): pass the CLI args through using getattr
     with the same defaults as the Bayesian call site.

No behavior change outside --config-file mode.
No new CLI flags (existing ones are reused).
No refactors.

Apply on Zeus:
    cd ~/distributed_prng_analysis
    cp window_optimizer.py window_optimizer.py.s170_config_parity_bak
    python3 patch_config_mode_parity.py window_optimizer.py
    python3 -c "import ast; ast.parse(open('window_optimizer.py').read()); print('AST OK')"
"""

import sys
import re
from pathlib import Path

MARKER_SIG  = "# [S170-PARITY] use_persistent_workers"
MARKER_BODY = "# [S170-PARITY] propagate persistent worker / transport"
MARKER_CALL = "# [S170-PARITY] CLI passthrough"


SIG_OLD = """def run_with_config(
    config_file: str,
    lottery_file: str,
    max_seeds: int,
    iterations: int,
    output_survivors: str = 'bidirectional_survivors.json',
    output_train: str = 'train_history.json',
    output_holdout: str = 'holdout_history.json'
) -> Dict[str, Any]:"""

SIG_NEW = """def run_with_config(
    config_file: str,
    lottery_file: str,
    max_seeds: int,
    iterations: int,
    output_survivors: str = 'bidirectional_survivors.json',
    output_train: str = 'train_history.json',
    output_holdout: str = 'holdout_history.json',
    use_persistent_workers: bool = False,   # [S170-PARITY] use_persistent_workers
    pwc_transport: str = 'tcp',             # [S170-PARITY] use_persistent_workers
) -> Dict[str, Any]:"""


# Body insertion anchor: exactly the line after add_window_optimizer_to_coordinator()
BODY_ANCHOR = """    # Add integration
    add_window_optimizer_to_coordinator()

    # Create WindowConfig object"""

BODY_REPLACEMENT = """    # Add integration
    add_window_optimizer_to_coordinator()

    # [S170-PARITY] propagate persistent worker / transport — match Bayesian path
    # (lines 614-616). Without these, --config-file mode silently downgrades to
    # legacy SSH distribution regardless of CLI flags.
    coordinator.use_persistent_workers = use_persistent_workers
    coordinator.pwc_transport          = pwc_transport

    # Create WindowConfig object"""


# Call site insertion: extend the kwargs list at line ~1201
CALL_ANCHOR = """        results = run_with_config(
            config_file=args.config_file,
            lottery_file=args.lottery_file,
            max_seeds=args.max_seeds,
            iterations=args.iterations,
            output_survivors=args.output_survivors,
            output_train=args.output_train,
            output_holdout=args.output_holdout
        )"""

CALL_REPLACEMENT = """        results = run_with_config(
            config_file=args.config_file,
            lottery_file=args.lottery_file,
            max_seeds=args.max_seeds,
            iterations=args.iterations,
            output_survivors=args.output_survivors,
            output_train=args.output_train,
            output_holdout=args.output_holdout,
            # [S170-PARITY] CLI passthrough — same defaults as Bayesian call site
            use_persistent_workers=getattr(args, 'use_persistent_workers', False),
            pwc_transport=getattr(args, 'pwc_transport', 'tcp'),
        )"""


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 patch_config_mode_parity.py <window_optimizer.py>")
        sys.exit(1)

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"ERROR: target not found: {target}")
        sys.exit(1)

    src = target.read_text()

    # Idempotency
    if MARKER_SIG in src and MARKER_BODY in src and MARKER_CALL in src:
        print(f"[S170-PARITY] markers already present in {target} — no changes")
        return

    # Edit 1: signature
    if SIG_OLD not in src:
        print(f"ERROR: SIG_OLD anchor not found in {target}")
        sys.exit(2)
    if src.count(SIG_OLD) != 1:
        print(f"ERROR: SIG_OLD anchor not unique (count={src.count(SIG_OLD)})")
        sys.exit(2)
    src = src.replace(SIG_OLD, SIG_NEW, 1)

    # Edit 2: body
    if BODY_ANCHOR not in src:
        print(f"ERROR: BODY_ANCHOR not found in {target}")
        sys.exit(3)
    if src.count(BODY_ANCHOR) != 1:
        print(f"ERROR: BODY_ANCHOR not unique (count={src.count(BODY_ANCHOR)})")
        sys.exit(3)
    src = src.replace(BODY_ANCHOR, BODY_REPLACEMENT, 1)

    # Edit 3: call site
    if CALL_ANCHOR not in src:
        print(f"ERROR: CALL_ANCHOR not found in {target}")
        sys.exit(4)
    if src.count(CALL_ANCHOR) != 1:
        print(f"ERROR: CALL_ANCHOR not unique (count={src.count(CALL_ANCHOR)})")
        sys.exit(4)
    src = src.replace(CALL_ANCHOR, CALL_REPLACEMENT, 1)

    # Validate with AST
    import ast
    try:
        ast.parse(src)
    except SyntaxError as e:
        print(f"ERROR: patched file has SyntaxError: {e}")
        print("--- not writing ---")
        sys.exit(5)

    target.write_text(src)
    print(f"[S170-PARITY] patched {target}")
    print("  3 edits:")
    print("    1) run_with_config() signature: +2 kwargs")
    print("    2) run_with_config() body: +2 coordinator attribute assignments")
    print("    3) call site: +2 getattr passthroughs")


if __name__ == "__main__":
    main()
