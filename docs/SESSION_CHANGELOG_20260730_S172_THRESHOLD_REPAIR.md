# SESSION CHANGELOG — 2026-07-30 — S172 optimizer threshold propagation repair

**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_S172_THRESHOLD_REPAIR.md` (REV2).
**Report:** `docs/S172_THRESHOLD_PROPAGATION_REPAIR_REPORT.md`.
**Host:** VM 101 `192.168.3.177` as `michael`, venv `~/venvs/torch`.
**Base:** `a0442a0`. **Not committed, not pushed — stopped at the gate for Team Alpha.**

---

## What was wrong

Optuna's sampled `forward_threshold` / `reverse_threshold` were dropped **above the backend
split**, so every trial ran at the configured default (live `0.30 / 0.30`) while the study
recorded the suggested value. Route A was fixed on 2026-04-30 (`3fdf434`) and silently
reverted on 2026-07-07 (`2389b61`) by a **stale-copy overwrite** of
`window_optimizer_integration_final.py` during unrelated PRNG-encoding work. Route B was
never covered by that fix at all. Nothing detected either for over two months.

## What changed

| repair | file:line |
|---|---|
| **R1 Route A** (`--n-parallel 1`, default) — `test_config` no longer binds `bounds.default_*` in its signature; resolves from `config` at call time | `window_optimizer_integration_final.py:2348-2364` |
| **R2 Route B** (`--n-parallel > 1`) — the explicit `_local_bounds.default_*` override in `_local_test` is removed; the partition subprocess now imports the shared resolver | `window_optimizer_integration_final.py:1880-1885`, `:1780-1785` |
| **single authority** — NEW `resolve_directional_threshold()` (explicit > config > default, `is None` only, raises rather than inventing) used by both routes | `window_optimizer_integration_final.py:196-243` |
| **R3 PWC hybrid — Option B, quarantine** — variable-skip fails closed at the execution boundary with `PWC_HYBRID_THRESHOLD_CONTRACT_UNCERTIFIED`; constant-skip untouched | `persistent_worker_coordinator.py:145-204`, `:1210` |

The Route A fix landed at the **callee only** — the caller already hands over the config
that owns the value, and `WindowOptimizer.test_configuration` is an interface method whose
fallback body has no threshold parameters. Adding them at the caller would have created a
second authority for the same quantity.

Option B was chosen over A because Beta's Option A also requires a requested/payload/
effective gate, and because propagating the threshold alone would leave PWC hybrid still
running a configuration nobody requested (kernels hardcode `expected_skip = 5`) while
removing the one visible symptom. PWC no longer certifies anything.

**The quarantine's placement changed mid-session, caught by the non-regression run.** The
first version also guarded the top of `run_trial_persistent`. That turned **D3.25 G1 red**:
D3.25 drives the live `run_trial_persistent` both-mode against a *fake* sieve to assert the
v2 four-map return shape, and never executes a hybrid pass. Quarantining a return-shape
contract is not what Beta's ruling asks for. The guard now lives at the **execution
boundary only** (`run_sieve_pass`); a real both-mode trial still fails closed, on the first
hybrid pass, before any hybrid survivor exists. Rationale left in the code at
`persistent_worker_coordinator.py:1640-1652` so the absence reads as a decision.

## Verification

- `tests/test_s172_threshold_propagation.py` (NEW): **5/5 gates PASS, 3/3 mutants killed,
  COMPLETION SENTINEL: PASS.** Gates extract the **live source** of each call site by AST
  and execute it — a text-anchor check would not have caught `2389b61`, which replaced the
  whole block. G-KERNEL reads `float32(0.73)` / `float32(0.31)` off the **real cupy
  `RawKernel` launch arguments** on the RTX 3080 Ti, chained hop to hop.
- Non-regression, captured on a tree restored to HEAD before any edit and again after:
  **15/15 suites green both runs.** **D6-threshold 17/17 both runs.**
- `tests/test_s172_phase4_coordinator.py` gate 22 whitelist extended for the new harness
  (review-flagged, standing pattern).

Nothing under `miner/` was touched; `sieve_gpu_worker.py`, `prng_registry.py`,
`persistent/pwc_protocol.py`, `distributed_config.json` and every Optuna study database are
unmodified. `s172_threshold_patch.py` was not run.

## Reported, not fixed

- `docs/window_optimizer_integration_final.py` — tracked, pushed to both remotes,
  byte-identical to production at `e8a69f5` (2026-04-22), copied into `docs/` at `7313a43`
  (2026-05-03), so it carries the pre-`3fdf434` defect in both routes (`:1394`, `:928`).
  **It is a re-introduction vector, NOT the proven source of `2389b61`** — the file
  `2389b61` produced is not byte-identical to it. An earlier draft of this changelog said
  otherwise; corrected.
  `window_optimizer_integration_final_INTEGRATED.py` is a 234-line fragment from the initial
  commit (`0101306`), already ruled on by S103.
  **DISPOSITION — CLOSED 2026-07-31 (Michael): leave both alone, out of scope.** The new
  behavioural gates already defend the failure mode regardless of copy source; no module
  imports either file; deleting either reds Phase-4 gate 22. Deletion recommendation
  withdrawn.
- `sieve_gpu_worker.py:44` replaces `sys.stdout` at import time, discarding whatever the
  importing process had buffered. Observed live — it ate this harness's own first `PASS`
  lines. VIR-1 hazard for any in-process importer.
- `coordinator.py:744` two-hybrid-signal finding: **deferred, and benign today** —
  `job['hybrid']` drives result shape, not kernel selection, and `phase2_threshold=None`
  falls through to the directional threshold.

## Deferred (named, not built)

1. Hybrid skip-bound dead dimension (kernel-signature change; historical variable-skip
   trials independently suspect).
2. Study ↔ commit provenance binding (the gap that makes the 2026-05-11 study
   **indeterminate**, not poisoned).
3. Replacement-resistant standing regression gate over the whole propagation chain.

## Operational

Per the brief's P0 note (for Michael): **no new Optuna runs and no variable-skip
certification runs until this lands.** Existing study databases are regression evidence —
preserved, none read for mutation, deleted, moved or overwritten.

`fallback parity:` not re-checked this session (no phase boundary reached; `.127` not booted).
