# SESSION_CHANGELOG_20260725_PHASE5_D3_25.md

**S172 Phase 5, Deliverable D3.25** — mode-preserving backend result contract +
canonical candidate-ingress normalization.

Spec: `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3_25.md` (REV3).
Base: HEAD `c207e3a` (D3 committed). **Not committed, not pushed, WATCHER not run.**

---

## 1. Non-regression baseline — captured GREEN before any edit

| suite | baseline |
|---|---|
| Phase 3 worker | 17/17 |
| Phase 4 coordinator | 63/63 |
| D0 metadata seam | 12/12 |
| D1.0 workflow | 8/8 |
| D1.1 assembly engine | 18/18 |
| D2 directional uniqueness | 7/7 |
| D3.0 encoding contract | 10/10 |
| D3 columnizer | 10/10 |

## 2. Pre-fix failure capture (REV3 §1.2)

Before any edit, the unmodified adapter was driven with the G2/G3/G5/G6
fixtures: **14 RED / 2 GREEN**. Signatures:

- G2 — one record for the cross-mode seed 42, labelled `variable`, carrying the
  CONSTANT rates (`fwd 0.9 / rev 0.7`, score `0.8` where the variable score is
  `0.75`). The constant candidate was destroyed before the L2 boundary.
- G3 — every variable aggregate was the constant-mode value
  (`forward_count 4≠1`, `reverse_count 2≠3`, `bidirectional_count 3≠1`,
  `intersection_count 2≠1`, `intersection_ratio 0.5≠1/3`,
  `survivor_overlap_ratio 0.5≠1.0`, `intersection_weight 1/3≠0.25`).
- G5 — `skip_range` was the string `'5-56'`.
- G6 — a config lacking `sessions` returned normally with the fabricated scalar
  `'all'`; a scalar `'all'` reached the record unchanged.

## 3. What changed

**A — extraction (semantics-preserving).** `_mode_records`
(`miner/range_miner_npz_writer.py:432`) moved VERBATIM to
`utils/canonical_records.build_mode_records`, with `CANONICAL_RECORD_FIELDS`
relocated beside it [C3]. The writer imports and re-exports both; `_mode_records`
remains as a private alias so no D1 call site changed. `utils/canonical_records.py`
takes NO dependency on `WindowConfig` and imports nothing from `miner/` [C2].
D1's shared-`sessions` reference is preserved unchanged [C5].

**B — v2 producer contract.** PWC and ZMQ now return
`step1_trial_populations_v2`: all four directional maps, both bidirectional
sets, `pruned` and `reason`, on EVERY return path — including **both** pruned
early returns (PWC `:1621-1629`, ZMQ `:1091-1099`), which previously carried no
variable keys at all. Both variable maps are initialized at constant-pair scope
so the shape never varies. Both backends assemble through
`build_trial_populations`, which egress-validates the intersection invariant.
Legacy `forward_map`/`reverse_map` aliases and the four record lists remain as
telemetry only.

**C — adapter ingress.** The cross-mode set union at `:276` is replaced by
`normalize_trial_populations`, behind an independent ingress validation wall
that runs BEFORE any accumulator mutation. Ordering is trial-major, mode-minor.
Canonical forms: integer `skip_range`, `list[str]` `sessions` with a defensive
copy, per-mode `prng_type`. A missing v2 field FAILS CLOSED — it is never
defaulted to empty.

**Miner path detached.** `:426` now calls `_build_test_result_from_miner`
instead of the shared adapter: REV3 §4 forbids routing miner output through the
PWC/ZMQ contract, and D6 owns miner candidate ingress. Observable behavior is
unchanged from what the shared adapter actually did for this path (no candidates
appended, counts +0, same threshold-gated flush).

## 4. Gates

`tests/test_s172_phase5_d3_25_candidate_ingress.py` — **13/13 green**
(G1-G13), with all 16 G11 mutants killed. G1 drives the REAL
`run_trial_persistent` / `run_trial_zmq_sqlite` return paths against a fake
sieve backend (no GPU, no rig, no socket). Every oracle is hand-transcribed;
nothing is imported from the code under test.

### 4a. Correction round (Team Beta) — G13 miner-path isolation, TEST-ONLY

The original twelve gates never exercised `_build_test_result_from_miner`, the
helper D3.25 detached the range-miner call site into. **G13** closes that hole
with **no production change**. It drives a realistic `serve_trial`-shaped result
— exactly the seven keys `serve_trial` returns (`run_id`, `state`, `committed`,
`workers_registered`, `stripes`, `manifests`, `bound_addr`) and no population
key — through the helper with a PRE-POPULATED accumulator, and asserts:

- `bidirectional` / `forward_count` / `reverse_count` deltas are all **0**, and
  the pre-existing accumulator contents are undisturbed;
- `TestResult.forward_count == reverse_count == bidirectional_count == 0`;
- `_flush_npz_incremental` fires **exactly once** with label
  `chunk/trial-{trial_number}` and against the same accumulator object — the
  cadence and label hand-transcribed from the PRE-D3.25 shared path at `c207e3a`;
  the label tracks the trial number, and `accumulator=None` produces no flush;
- **no miner ingress**, proved two ways: behaviorally, by baiting the input with
  a fully-formed `MinerTrialAssembly` plus `canonical_records_constant` /
  `canonical_records_variable` and asserting none of it is consumed; and at
  source level, by asserting the function body references none of
  `MinerTrialAssembly`, the two canonical-record keys, `assemble_trial`, the v2
  normalizer/validator, `build_mode_records`, spool or manifest reads, and
  performs no append/extend on the candidate accumulator.

Two mutants prove G13 bites (red signatures):

```
killed by G13  miner path appends a candidate
   -> AssertionError: accumulator['bidirectional'] delta 1 != 0 — the miner
      path appended a candidate; D6 owns miner candidate ingress
killed by G13  miner path drops the flush call
   -> AssertionError: _flush_npz_incremental called 0 time(s); the pre-D3.25
      shared path called it EXACTLY ONCE per invocation — flush cadence must
      not shift
```

## 5. Decisions requiring Team Beta awareness

1. **Phase-4 gate 22 no longer asserts PWC/ZMQ are unmodified.** D3.25's
   approved scope (REV3 §6) lists both as may-modify, so the Phase-4-era
   coexistence claim is superseded by construction. Both files are now
   registered in the gate-22 whitelist; `persistent/pwc_protocol.py` remains
   asserted-unmodified. This is the deliverable that ends the "PWC/ZMQ are
   untouched" era.
2. **The miner call site was detached from the shared adapter** rather than fed
   a v2-shaped result (see §3). This is the reading of REV3 §4 that keeps the
   miner path working; the alternative — leaving `:426` on the v2 adapter —
   would fail closed at runtime for the wrong reason. **Now gate-covered by G13**
   (§4a), added in the Team Beta correction round.
3. **Producer egress is validated functionally, not by a live GPU run.** G1
   drives both real trial functions with a fake sieve; acceptance against real
   rigs remains Phase 6 (§11.A-E).

## 6. Pre-existing, NOT caused by D3.25

`test_persistent_worker_harness.py` (root, S134-era, not in the blocking NR
list) is **23/25 both before and after** — verified by running it in a detached
worktree at pristine `c207e3a`. T19 and T20 were already red at HEAD. T19's
failure mode changed (stale assertion → the new fail-closed contract error)
because it hand-builds a pre-v2 result dict; it encodes the superseded contract
and needs updating once D3.25 is accepted.

## 7. Fallback parity

`fallback parity: code=[not re-checked this session], env=[not re-checked this
session]` — no phase boundary crossed; D3.25 is unreviewed and uncommitted.

## 8. Status

STOP at the gate, per REV3 §8. Awaiting Team Alpha review, then Team Beta.

**Correction round (Team Beta, two conditions):** both discharged — (1) G13
miner-isolation gate added with two bite-proof mutants, TEST-ONLY, no production
change (§4a); (2) this changelog renamed from `SESSION_CHANGELOG_20260725_S179.md`
to `SESSION_CHANGELOG_20260725_PHASE5_D3_25.md`. Full D3.25 gate re-run at 13/13
and the complete blocking non-regression set re-run green afterwards.
