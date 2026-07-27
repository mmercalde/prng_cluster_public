# TB RULING REQUEST — D5 exception-precedence under read-all-then-merge

**From:** Team Alpha
**Re:** S172 Phase 5 D5. One equivalence-contract decision blocks advancing D5 to
Beta review. Everything else in the D5 report is Alpha-approved pending five
on-box verifications (listed at end, not part of this ruling).
**Base:** HEAD `3e8580a1e123243428e0d1b8d0ab043032ed11f7`.
**Decision needed:** A vs B below. Alpha recommends **B**.

---

## 1. The finding (verified at source, not extrapolated)

D5's Commit 1 was specified as a semantics-preserving extraction whose proof is
"D1.1 stays 18/18 with zero test edits." It is **not** a strict no-op. There is a
real, demonstrated exception-precedence divergence from the pre-D5
`assemble_trial`, and D1.1's suite does not cover it — which is the only reason
18/18 still passes.

**Root cause — the original interleaves read and merge.** In
`miner/range_miner_npz_writer.py`, the pre-D5 `assemble_trial` walks the
deterministic `order` and, for each spool, **reads then immediately merges** in
the same pass: `_read_and_validate_spool(...)` (may raise `SpoolIdentityError`)
is followed in the same loop body by the map-insert that may raise
`DirectionalDuplicateError`. So a duplicate in an earlier-order spool raises
**before** a later-order spool is ever read.

**What the brief mandated.** The D5 brief §3.2 specified read-**all**-then-merge:

```
projections = [read_and_validate_spool(run_id, manifests[i]) for i in order]
return merge_validated_spools(run_id, ...)
```

This is the correct shape for a parallel front end — you cannot merge what you
have not read, and reading in parallel is the entire point. But it reorders
observable exceptions.

**The divergence.** Trial with an earlier-order spool carrying a duplicate seed
**and** a later-order spool that is malformed:

| structure | error observed |
|---|---|
| pre-D5 interleaved (original) | `DirectionalDuplicateError` (earlier spool merges before later is read) |
| D5 read-all-then-merge | `SpoolIdentityError` (later read fails before any merge runs) |

Both are fail-closed producer-defect errors; the trial aborts either way. Only
*which diagnostic surfaces first* changes. No D1.1 test exercises earlier-dup +
later-malformed, so 18/18 is green over an uncovered behavioral change.

**Attribution.** This is a brief defect (Team Alpha authored §3.2 as
read-all-first without noticing the original interleaved). Claude Code
implemented the brief faithfully and flagged the divergence explicitly rather
than burying it. No implementer error.

---

## 2. Why this reaches you rather than being decided in the brief

It is an equivalence-contract question, and Beta owns the contract. The project's
standing principle — from the D5 ruling's own finding F4, *"exact replay is
preferable to reasoning that the order should not matter"* — is exactly the
principle in tension here, one layer down. Alpha will not weaken the equivalence
claim on its own authority.

---

## 3. The two options

### Option A — accept and pin
Declare exception precedence **among distinct fail-closed producer-defect errors**
(malformed-read vs directional-duplicate) as outside the preserved contract. Add
a test that **pins the new read-all-then-merge precedence** so it is intentional
and locked, not accidental and drifting. Document the carve-out in the spec.

- Cost: ~one test + a spec paragraph.
- Justification: divergence manifests only when a trial already carries two
  independent producer defects and is aborting regardless; no valid-input
  behavior changes; Phase 6's four-path oracle compares valid-input 22-arrays and
  never observes this path.
- Risk: spends the project's "exactness over 'shouldn't matter'" credibility.

### Option B — preserve exactly (Alpha recommendation)
Workers return **either** a `ValidatedSpoolProjection` **or** a captured
read-error descriptor — never a raised exception crossing the boundary. The
parent replays in deterministic `order` and, at each position, **either**
re-raises that spool's captured read-error **or** performs the merge-insert (which
may raise the duplicate). This reproduces the original interleaved precedence
**exactly**, in both serial and parallel paths.

- Verified to reproduce every corner: earlier-malformed + later-dup →
  malformed first (parent re-raises at earlier position before reaching later
  merge); earlier-dup + later-malformed → dup first (parent merges earlier
  position before re-raising later read-error); intra-spool-dup + later-malformed
  → intra-spool dup first (it is a merge error, fires at the earlier position).
- Cost: workers carry read-errors as data; parent gains an ordered re-raise
  branch. Reuses the consume-in-`order` machinery G-MALFORMED-DUAL already
  exercises. Bounded — roughly forty lines.
- Benefit: equivalence claim stays airtight; the parallel parent's error handling
  becomes **identical** to serial, which is easier to reason about and to test;
  Commit 1 becomes a true no-op (serial interleaving restored).

**Alpha recommends B.** F4 already established exact replay over
order-shouldn't-matter reasoning; this is the same call. B costs ~forty lines to
keep the equivalence contract from carrying a documented exception. Given how much
of this project rests on not hand-waving equivalence, Alpha would pay it.

---

## 4. If B is chosen — scope note

- Commit 1's serial `assemble_trial` must restore interleaving (lazy per-spool
  read in `order`, each followed by its merge) so it is byte-identical to the
  original including precedence. `merge_validated_spools` consumes an ordered
  source that yields per-spool "projection-or-read-error"; serial yields lazily,
  parallel yields eagerly from workers — same replay logic, different fill.
- Add gates: earlier-dup + later-malformed and earlier-malformed + later-dup,
  asserting the identical exception and attribution under serial_reference,
  process_sharded, **and** (regression) that both match the pre-D5 behavior.
- D1.1 remains 18/18 with zero edits; the two new corner tests live in the D5
  gate, not in D1.1.

## 5. Not part of this ruling (Alpha will verify on-box before advancing)

1. D4 live → 8/8 with all 9 mutants intact after Claude Code's fix.
2. gate-22 whitelist addition is the only D0–D4 change; the cited "standing
   whitelist rule" is real.
3. M1 now dies on the injected defect, not loader/type-identity.
4. Commit-1 diff: merge body character-identical modulo loop header;
   `prepare_trial_assembly` / `merge_validated_spools` are verbatim moves.
5. Logged for Phase 6 §17: process_sharded ~1.5× faster high-survivor but
   ~2–3× RAM (288→798 MiB pool sweep) and ~150× slower low-survivor — likely
   fails §17's ≤50% RAM clause; serial_reference likely stays low-survivor
   default. Measurement only; no D5 action.

---

**Decision requested:** A or B. On B, Alpha proceeds per §4. On A, Alpha adds the
pinning test and the spec carve-out and advances.
