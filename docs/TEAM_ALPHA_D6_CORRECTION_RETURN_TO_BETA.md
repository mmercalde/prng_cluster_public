# TEAM ALPHA → TEAM BETA — D6 correction package (return)

**Re:** S172 Phase 5 D6 — threshold propagation, effective-threshold provenance,
residue authority, autonomy dependency. Returning per Beta §7.

**Status:** correction pass complete. **Nothing committed, nothing pushed,
WATCHER not run.** D6 remains uncommitted in the VM-101 working tree pending
Beta's pass.

---

## 1. Modified-file set — stated up front

Alpha under-disclosed the writer seam last round; not repeating that. The full
set the correction pass leaves dirty:

| file | why |
|---|---|
| `miner/range_miner_coordinator.py` | threshold resolution + emission (the fix) + contract docstring |
| `miner/range_miner_worker.py` | threshold consumption, re-validation, effective-threshold reporting + contract comment |
| `miner/range_miner_protocol.py` | **+12/−0**, two optional `effective_threshold` fields (provenance carrier) |
| `miner/range_miner_npz_writer.py` | D6 `backend=None` seam (prior pass, Beta-approved 4A) |
| `window_optimizer_integration_final.py` | D6 adapter (prior pass) |
| `tests/test_s172_phase4_coordinator.py` | gate-22 coexistence allowlist, extended for `range_miner_protocol.py` + the new gate |
| `tests/test_s172_phase5_d3_25_candidate_ingress.py` | G13 (prior pass, Beta-approved) |
| `docs/TODO_SELFPLAY_AND_LLM_AUTONOMY.md` | autonomy tripwire (Beta §5) |

New: `tests/test_s172_phase5_d6_threshold_path.py`,
`tests/test_s172_phase5_d6_production_adapter.py`,
`tests/smoke_s172_phase5_d6_zeus_single_gpu.py`, `miner/step1_ingress.py`,
`docs/SESSION_CHANGELOG_20260728_PHASE5_D6_CORRECTION.md`.

**Protocol change reviewed by Alpha in full:** purely additive, two `Optional[float]
= None` fields on `SubStripeResultMessage` and `StripeCompleteMessage`. Defaulted
so a pre-D6 peer still decodes (wire compat preserved). A stripe whose sub-stripes
disagree **reports the disagreement explicitly — the worker refuses to average or
pick one.**

## 2. Gate + mutant counts

**D6 total: 21 checks / 23 mutants.**

- `test_s172_phase5_d6_production_adapter.py` (3.A, unchanged): **9/9, 16 mutants**
- `test_s172_phase5_d6_threshold_path.py` (new): **12/12, 7 mutant kills** — Beta's
  nine checks G1–G9 at asymmetric `0.31/0.47`, plus R1–R4 residue session cases

| mutant | killed by |
|---|---|
| M-drop (no threshold field) | G8; independently G1 |
| M-collapse (forward on both) | G2; independently G5 |
| **M-swap (forward↔reverse)** | **G6 only — G5 explicitly asserted to *survive*, proving the swap needed its own detector** |
| M-residue (session filter dropped parent-side) | R1–R3; independently R4 |

The M-swap result confirms Beta's reasoning verbatim: two consistently-reversed
branches still present as asymmetric, so the not-collapsed check passes and only a
dedicated swap detector reds.

## 3. Non-regression (exact)

Phase 4 **63/63** · D1.1 **18/18** · D4 **8/8** · D5 **24/24** · D6 3.A **9/9**.
`G-FLUSH-CADENCE` still pins pre-D6 behaviour. One test edit was required — the
gate-22 coexistence allowlist, extended for `miner/range_miner_protocol.py` and
the new gate, with rationale inline.

## 4. Real-silicon smoke — non-default asymmetric thresholds

`forward=0.31 / reverse=0.47` (now the defaults; the old single `--threshold 0.25`
was indistinguishable from the fallback and is gone). 8M seeds, window 3, real
worker process on the 3080 Ti, **23.3 s**.

- **Survivor counts by direction: forward 398,156 · reverse 383 · bidirectional 319.**
  The split is itself physical evidence the two directions filtered at different
  values — under the defect both sat at 0.25 and were near-identical
  (398,156 / 398,226). The magnitude of the reverse drop is the expected
  consequence of a stricter bar on the stricter direction; the reverse match-rate
  distribution has most of its mass below 0.47.
- **Provenance:** requested `fwd 0.31 / rev 0.47` = payload `{1:[0.31], 2:[0.47]}`
  = effective `{1:[0.31], 2:[0.47]}`. **The effective leg comes back off the real
  executor, not recomputed from config** — this is the evidence `WindowConfig`
  alone could not provide (Beta §2).
- **Generation:** certified, 22 arrays in frozen order, `validate_array_bundle()`
  passed, 319 rows. **Step-2 load-back:** `format=npz, npz_version=3, count=319,
  fallback_used=False`.

## 5. Residue — FIXED, not guarded (Beta §4 preferred route)

One canonical `load_residue_window(path, window_size, sessions, offset)`. The
worker consumes it via `ResidueResolver`'s default loader; the parent via the new
`_miner_residues_for_config`. **The session filter exists in exactly one place.**
The eligibility-guard fallback was not needed. `_get_residues_for_config` is
untouched, so PWC/ZMQ keep their derivation byte-for-byte. Cases R1–R4 cover both
sessions / midday-only / evening-only / coordinator-worker identical ordered
residues, with the session-filter-dropped mutant killed.

## 6. The `min_match_threshold` / `phase2_threshold` contract (Beta §2)

Established **by reading the executor, not the field names** (`_constant_prefix`,
`_hybrid_prefix`, `BuildContext`, the two host post-filters):

- Each stripe launches **one kernel with one threshold scalar**.
- **Constant kernels read `min_match_threshold`; hybrid kernels read
  `phase2_threshold`. Which key feeds the kernel is decided by skip mode, not
  stage.**
- `phase2_threshold` is a **legacy PWC/Step-1 job-schema name**
  (`phase1_threshold`/`phase2_threshold` in the old `job_sieve_*.json` shape),
  where "phase 2" meant the hybrid/variable-skip run — *a different kernel, not a
  later stage.* **There is no stage-2 filter in the miner.**
- `ctx.threshold` never reaches a hybrid kernel; `ctx.hybrid_threshold` never
  reaches a constant one.
- Therefore **D6 emits both keys equal, asserts the invariant at construction, and
  the worker re-validates on receipt (`ThresholdContractError`, non-retryable).**

This is recorded as a ⚠️ "THE `phase2_threshold` NAMING TRAP — READ BEFORE
'FIXING' THIS" section in the `build_stripe_assign_payload` docstring, with a
6-line pointer at the worker's hybrid-branch read. It carries the explicit
warning that relaxing the equality would **not** create a two-stage filter — it
would mean constant and variable stripes of the same trial silently filtering at
different thresholds; a genuine multi-stage sieve needs a new kernel ABI and its
own governed config field.

**Proven docs-only** (same method as the D5 wording fix): AST identity with
docstrings stripped, pre-edit vs post-edit — coordinator 118→118 docstrings,
worker 27→27, **AST identical: YES** on both; worker +6 lines all comments, zero
deletions; coordinator zero deletions, all additions inside the docstring. Gates
after the edit: threshold path **12/12**, 3.A **9/9** — both identical to before.

## 7. Autonomy dependency (Beta §5) — documentation only

- ⚠️ tripwire + Part-B status note added to `docs/TODO_SELFPLAY_AND_LLM_AUTONOMY.md`.
- Invariant **NOTE** (not "guard" — a comment is documentation, not enforcement)
  at the `build_stripe_assign_payload` chokepoint.
- **`watcher_policies.json` UNTOUCHED per Beta's ruling.** No `_parameter_application_note`,
  no field added. The `parameter_application: true` advisory-only discrepancy is
  recorded in the autonomy TODO and the changelog only, flagged for the dedicated
  Part-B implementation (which must audit recommended / approved-applied /
  effective and never record an adaptation as applied unless the effective
  execution value matches).

## 8. Scope held

Untouched as scoped: `_flush_npz_incremental` (**D6.1**, not opportunistically
repaired), PWC/ZMQ ingress, the D3.25 contract, `TestResult` shape,
`pwc_protocol.py`. `serial_reference` remains default; `process_sharded`
unpromoted.

## 9. One implementation choice to flag

`build_stripe_assign_payload`'s new `phase` / `forward_threshold` /
`reverse_threshold` parameters are **required keyword-only with no defaults** —
deliberately, so an omission is a loud `TypeError` rather than a silent fallback.
Any out-of-tree caller will break loudly; the two in-tree callers were updated.
Alpha considers this stronger than the brief required and flags it for Beta's
awareness rather than assuming approval.

## 10. Changelog

`docs/SESSION_CHANGELOG_20260728_PHASE5_D6_CORRECTION.md`, carrying the D5
writer-freeze exception sentence verbatim.

---

## Alpha disposition

The blocker is closed on the merits: the configured directional thresholds now
travel one canonical path to the kernel, and the effective value is measured off
the executor rather than assumed from config. Residue is fixed at the authority
rather than guarded. All three mutants — including the swap detector Beta required
— are killed under the four-part rule, and the real-silicon rerun proves the path
at non-default asymmetric values with the directional split as physical evidence.

Requesting Beta's pass. On approval the sequence is: Michael commits D6 → runs the
release-grade smoke from the clean real repository → records the commit-linked
certified generation (commit hash, generation ID, artifact SHA-256, tree-clean
result, array validation, Step-2 loader result, `fallback_used=False`) → `D6.1`
opens as the separate blocking prerequisite for the extended Phase 6/7 soak.
