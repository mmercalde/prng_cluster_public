# CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D6_CORRECTION.md — REV1

**S172 Phase 5 D6 — correction pass.** D6 is HELD by Team Beta on one correctness
blocker: RANGE-MINER silently ignores the configured forward/reverse sieve
thresholds and filters at a hardcoded `0.25`, so the optimizer certifies results
for a configuration other than the one it requested. This pass fixes that through
a single canonical threshold path, fixes the single-session residue asymmetry,
and documents the autonomy dependency — nothing else.

**Base:** current `main` tip (the D6 working tree; HEAD `bdd36b4` lineage, D6
changes uncommitted). Claude Code on VM 101 as `michael`, venv `~/venvs/torch`,
`--optimizer-python ~/distributed_prng_analysis/python3_with_venv.sh`. Implement
and iterate; do NOT commit/push/run WATCHER. STOP at the gate for Team Alpha
review.

**Authority:** Beta's D6 disposition + Beta's response to the correction plan
(both binding). This brief encodes them.

---

## 0. The blocker, confirmed at source

`build_stripe_assign_payload` (`miner/range_miner_coordinator.py:3082-3089`)
returns a payload with **no threshold key**. The worker
(`miner/range_miner_worker.py:734`) therefore does
`coerce_threshold(payload.get("min_match_threshold", None), 0.25)` → hardcoded
`0.25`, and filters at it (`:832`). The hybrid/variable `phase2_threshold`
(`:788`) is likewise absent. The 3.B smoke masked this by using `--threshold`
default `0.25` (`smoke:357`). Optuna sweeps `forward_threshold`/`reverse_threshold`
as real per-trial, per-direction hyperparameters — all silently dropped on the
miner path.

---

## 1. The single canonical threshold path (Beta §1)

The threshold is resolved **once, in the parent, before stripe assignment** —
never reinterpreted by the worker:

```
WindowConfig.forward_threshold / reverse_threshold
  → parent direction/phase resolution (§6.8 table)
  → canonical stripe payload
  → worker
  → executor
  → kernel
```

Per stripe, the parent resolves the directional value from the phase table
(`range_miner_coordinator.py:1482-1489`): **forward phases (1,3) →
`forward_threshold`; reverse phases (2,4) → `reverse_threshold`.** The worker MUST
NOT choose between forward and reverse or re-read the trial config.

`build_stripe_assign_payload` must emit the **resolved** value explicitly in every
D6 payload. The worker's `0.25` fallback stays ONLY for backward compatibility
with legacy (pre-D6) payloads. **A newly generated D6 payload that silently relies
on the fallback is a defect** — the gate must catch it (see G8 below).

This is the ONE path all threshold sources flow through — Optuna now, and the
(not-yet-built) WATCHER parameter-application path later. Do not create a second
threshold path anywhere.

## 2. `min_match_threshold` and `phase2_threshold` — one explicit contract (Beta §2)

Both fields are acceptable ONLY with defined, tested meanings:

- `min_match_threshold` = the resolved directional threshold for that stripe.
- `phase2_threshold` (variable/hybrid executor, `worker:788`) must be derived from
  the **same directional source** unless a separately governed config field
  already exists. Read the current hybrid executor to determine what it actually
  requires (`range_miner_worker.py` around `:767-803`). **Do not infer the
  first-stage/second-stage distinction from field names** — document whatever the
  algorithm actually does as an explicit contract in the payload builder's
  docstring.
- If the algorithm expects the two values identical, **assert that invariant** at
  payload construction or worker validation — do not let a contradictory pair pass
  silently.

## 3. Effective-threshold provenance (Beta §2, the teeth)

`WindowConfig` alone is NOT evidence the value reached execution. The worker
completion / result metadata must expose the **effective** threshold the kernel
used, so the parent and audit record can distinguish three values:

```
requested   (WindowConfig.forward/reverse_threshold)
payload      (what build_stripe_assign_payload emitted)
effective    (what the executor/kernel actually filtered at)
```

The gate (G9) asserts all three match for a non-default value. This is the
mechanism that makes a future "threshold adaptation applied" claim auditable
against physical reality — never assume, always record the effective value.

## 4. Required gates — `tests/test_s172_phase5_d6_threshold_path.py` (or extend the D6 adapter gate)

Beta's nine checks, asymmetric `forward=0.31 / reverse=0.47`:

1. Forward assignments carry exactly `0.31`.
2. Reverse assignments carry exactly `0.47`.
3. The worker receives each exact value.
4. The executor/kernel receives each exact value.
5. Values are not collapsed (forward ≠ reverse preserved).
6. Values are not swapped.
7. Legacy payload omission still resolves to `0.25` (backward compat).
8. **New D6 payloads always contain an explicit threshold** (no silent fallback).
9. **Effective-threshold provenance** matches requested and transmitted values.

**Three required mutants** (each under the four-part kill rule):
- **M-drop:** drop the payload field → worker falls back to `0.25` → killed by
  G8/G1-4 (a new payload must not rely on fallback).
- **M-collapse:** apply `forward_threshold` to both directions → killed by G2/G6.
- **M-swap:** forward↔reverse swapped → killed by G6. *This is its own mutant
  because two separately-wrong branches can look asymmetric while being
  consistently reversed — G5 (not-collapsed) passes, only G6 catches it.*

## 5. Residue asymmetry — shared authority (Beta §4)

Preferred: **one canonical residue-derivation function** whose inputs include the
session selection; both coordinator and worker consume it (or compare against its
canonical output). Do NOT duplicate session-filter logic. Cases: both sessions;
midday-only; evening-only; coordinator and worker produce **identical ordered
residues** in each; a mutant that ignores the session filter is killed.

Fallback (only if the shared-authority fix can't complete cleanly this pass): a
pre-dispatch eligibility guard that runs **before worker submission**, names the
unsupported single-session config clearly, creates **no** spool/process/GPU work,
and cannot be mistaken for a zero-survivor trial. State explicitly in the report
which route was taken.

## 6. Autonomy dependency — documentation only, NO policy field (Beta §5)

Three actions, and a **correction from the prior plan**:

- **Add** the ⚠️ BLOCKED-BY tripwire under Part B (near task B3) of
  `docs/TODO_SELFPLAY_AND_LLM_AUTONOMY.md` — sieve-threshold autonomy cannot be
  enabled until this pass's kernel-path gate is green; when built it must route
  through the single `build_stripe_assign_payload` chokepoint. (Text:
  `docs/D6_THRESHOLD_AUTONOMY_SIGNPOSTS.md` item 1.)
- **Add** an **invariant NOTE** (not a "guard" — a comment is documentation, not
  enforcement) at `build_stripe_assign_payload` stating every threshold source
  must pass through this chokepoint. (Text: signposts item 2.)
- **DO NOT** add `_parameter_application_note` or any field to
  `watcher_policies.json`. Beta rejected the ad-hoc field (unvalidated → breaks
  strict parsing or becomes ignored metadata; and a note doesn't make
  `"parameter_application": true` truthful). **Leave `watcher_policies.json`
  untouched in D6.** Record the discrepancy — that `parameter_application: true`
  is advisory-only in reality (`diagnostics_analysis_schema.py:76`) — in the
  **autonomy TODO and the session changelog only**, flagged for the dedicated
  Part-B implementation, which must audit `recommended` / `approved-applied` /
  `effective` and never record an adaptation as applied unless the effective
  execution value matches.

## 7. Do NOT touch in this pass

- `_flush_npz_incremental` — remains `D6.1`, a separate high-priority repair. Do
  not opportunistically fix it here (it would shift flush cadence). Confirm the
  existing G-FLUSH-CADENCE gate still pins current behavior.
- PWC/ZMQ ingress, the D3.25 four-map contract, `TestResult` shape — unchanged.
- `serial_reference` stays default; `process_sharded` unpromoted.

## 8. Record the D5 writer-freeze exception (Beta §7.7 / 4A)

The session changelog must carry, verbatim: *"D6 introduces one approved post-D5
extension to `AssemblingPhase5Sink`: an optional assembly-backend seam whose
`None` path is the exact pre-D6 behavior."* Stop describing the writer as
unconditionally frozen.

## 9. Rerun + real-silicon smoke (Beta §7)

Non-regression: **D1.1 18/18, D4 8/8, D5 24/24, D6** all green (the fix changes
worker payloads, so rerun D6 and the smoke). The real-silicon rerun **must use
non-default asymmetric thresholds** (`forward=0.31 / reverse=0.47`) — repeating
`0.25/0.25` proves nothing about the corrected path. A smaller instrumentation
smoke at `0.31/0.47` is acceptable before the full acceptance run if survivor
volume is impractical.

## 10. Return package (Beta §7)

Report: updated D6 gate count; updated mutant count; exact non-regression results;
smoke thresholds; **survivor counts by direction**; generation validation +
Step-2 load-back (`fallback_used=False`); whether residue was **fixed or guarded**;
and the three provenance values (requested/payload/effective) from the smoke.
Then STOP for Team Alpha review.

**Do not commit.** After Alpha + Beta pass: Michael commits D6, then runs the
release-grade smoke from the clean real repository and records the commit-linked
certified generation.
