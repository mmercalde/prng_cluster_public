"""Pinned-build harness for the Run-4 routing gates.

`run4_routing_clean_control.py` is FROZEN: its sha256 is recorded in the STEP-0
control artifact and the capture script refuses to re-run against a modified
`agents/watcher_agent.py`, so it cannot be regenerated. This module therefore
adds the pinned-build capability by WRAPPING a builder -- reusing its entire stub
boundary with zero duplication, so pinned and unpinned argvs are produced by
byte-identical machinery. Any divergence between them is a property of the
patch, never of two different harnesses.

[MINER ADMISSION-STALL brief, Part 2, 2026-11-14] The builder wrapped is the
SUCCESSOR harness `run4_routing_search_control` (the frozen boundary plus the
stage-1 search collaborator, declared delta gated by G-SUCCESSOR-BOUNDARY). Since
stage 1 a Step 1 build cannot reach dispatch without an open search, so the frozen
module's own builder fails closed on today's tree; it is kept byte-identical for
its original 69ca910 comparison only.
"""

from . import run4_routing_search_control as FX
from . import run4_search_fixture as S


def build_argv_with_pin(agent_mod, params, pin_bundle):
    """Build the argv that run_step produces for `params` under `pin_bundle`."""
    real = agent_mod.WatcherAgent.run_step

    def _wrapper(self, step, p=None, *, _pin_bundle=None):
        return real(self, step, p, _pin_bundle=pin_bundle)

    agent_mod.WatcherAgent.run_step = _wrapper
    S.SearchDB.pin_bundle = pin_bundle      # the build's search follows its pin
    try:
        return FX.build_argv(agent_mod, params)
    finally:
        S.SearchDB.pin_bundle = None
        agent_mod.WatcherAgent.run_step = real


def warm_pairs(argv):
    """Extract {param: value} for every --warm-start-* pair present in argv."""
    out = {}
    for i, tok in enumerate(argv):
        if tok.startswith("--warm-start-"):
            key = tok[2:].replace("-", "_")
            out[key] = argv[i + 1] if i + 1 < len(argv) else None
    return out


def build_result(agent_mod, params, pin_bundle=None):
    """Return (argv, results) for ONE run_step dispatch.

    The builder (`run4_routing_search_control.build_argv`, which reproduces the
    frozen module's) discards run_step's return value, and the frozen module's
    sha256 is recorded in the STEP-0 control artifact, so the structured result
    is captured here instead — by wrapping,
    exactly as `build_argv_with_pin` does, so pinned and unpinned results come
    from byte-identical machinery. `pin_bundle=None` drives the genuine
    unpinned path.
    """
    real = agent_mod.WatcherAgent.run_step
    box = {}

    def _wrapper(self, step, p=None, *, _pin_bundle=None):
        out = real(self, step, p, _pin_bundle=pin_bundle)
        box["results"] = out
        return out

    agent_mod.WatcherAgent.run_step = _wrapper
    S.SearchDB.pin_bundle = pin_bundle      # the build's search follows its pin
    try:
        argv = FX.build_argv(agent_mod, params)
    finally:
        S.SearchDB.pin_bundle = None
        agent_mod.WatcherAgent.run_step = real
    return argv, box.get("results")
