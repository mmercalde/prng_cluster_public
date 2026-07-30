# CLAUDE_CODE_INSTRUCTIONS_S172_D6_2_CHECKPOINT_RECONCILIATION.md — REV1 (DRAFT — pending Beta approval)

**S172 — `D6.2: 24-field checkpoint, canonical reconciliation, and finalizer resume
path.`**

**This is the deferred half of D6.1.** D6.1 made the incremental checkpoint *write*
for the first time (relocated to `.s172_checkpoint/`, suffix bug fixed, failures
visible, fsync + temp cleanup) but left the **in-memory list-clear disabled**,
because the checkpoint persists 4 arrays while the D3.5 finalizer requires all 24
`CANONICAL_RECORD_FIELDS`. Consequently the **S166 OOM protection does not exist**:
the candidate list still grows unboundedly on long runs.

**Status:** DRAFT pending Beta's ruling on numbering and sequencing (Alpha proposed
`D6.2`, to run after Phase 6.0 and Phase 6, but it **must** land before Phase 7).

**Base:** the post-D6.1 commit (TBD). Claude Code on VM 101 as `michael`, venv
`~/venvs/torch`. Implement and iterate; do **NOT** commit, push, or run WATCHER.
STOP at the gate.

---

## 0. Why this exists — the three facts that define the work

1. **The checkpoint schema is inadequate for its stated purpose.** It persists 4
   arrays (`seeds`, `score`, `forward_match_rate`, `reverse_match_rate`). The
   finalizer at `window_optimizer_integration_final.py:1868` consumes the in-memory
   list and requires **24** canonical fields. 4 cannot reconstruct 24.
2. **The S166 comment's guarantee was doubly false.** "*data is safe in NPZ*"
   justified clearing the list — but the write always failed *and*, even had it
   succeeded, a 4-field file could never have backed the claim. D6.1 fixed the first
   half. D6.2 must make the claim true or leave the clear off forever.
3. **The current merge policy is not the canonical one.** The flush helper
   reconciles duplicate seeds inline with
   `s.get("score", 0.0) > seen[seed].get("score", 0.0)`. The canonical authority is
   `_l2_sort_key` / `_select_l2_winners` in `utils/run_finalizer.py:688-747`
   (frozen, Ruling D). They differ in ways that matter — see §2.

**Read all three at source before writing anything.**

## 1. The reconciliation authority — the load-bearing requirement

Team Beta ruled: *"merge by seed is valid only if it invokes the **existing
canonical accumulator reconciliation rule**. It must not introduce a new dictionary
policy such as arbitrary first-wins or last-wins… If differing records for the same
seed are a producer defect, recovery must raise rather than choose one."*

The canonical rule (`_l2_sort_key`, frozen under Ruling D) is **highest-wins on
every component**:
1. highest canonical **float32** score;
2. then **lowest `trial_number`**;
3. then constant-before-variable — **only** as a tiebreak within one trial.

Three properties of that rule the repair must inherit, not approximate:
- **The comparison domain is float32.** Two Python floats differing only beyond
  float32 precision are an **exact tie** and must fall through to the trial-number
  tiebreak. The docstring is explicit that comparing pre-rounding float64 while
  storing the rounded value *is the defect this converts away*.
- **The result is independent of input order** — because within one seed the key is
  a strict total order.
- **A same-trial/same-mode collision for one seed raises
  `AccumulatorConsistencyError`** — after D1/D2 it is impossible, so its presence
  means the accumulator received the same trial's population twice. **This is
  Beta's "recovery must raise rather than choose."**

**How the current flush helper violates each:** it compares float64; it has no
trial_number or mode tiebreak; `.get("score", 0.0)` silently defaults a missing
score to zero; it never raises on a same-trial/same-mode duplicate; and its
prior-NPZ merge path reconstructs bare `{"seed", "score"}` records
(`window_optimizer_integration_final.py:281-286`) — **discarding `trial_number`,
`skip_mode` and all provenance, so the information the canonical rule needs is
already gone.**

**Mandate: import and call `_select_l2_winners` / `_l2_sort_key`. Do not
reimplement, do not approximate, do not write a second policy.** If importing
creates a circular dependency, extract the authority to a shared module and have
*both* call sites use it — never fork it.

## 2. Schema — persist what the canonical rule and the finalizer require

The checkpoint must persist, per seed, the full set of `CANONICAL_RECORD_FIELDS`
required to (a) apply `_l2_sort_key` and (b) hand the finalizer a complete raw
candidate.

Constraints:
- **`allow_pickle=False`, no object arrays.** Some canonical fields are strings
  (`skip_mode`, mode/provenance identifiers) — encode as fixed-width or as an
  index-into-a-code-table with the table stored as its own array. State the choice.
- Preserve exact dtypes where the artifact contract fixes them; do **not** silently
  widen or narrow (see D5's int64/uint32 lessons — the assembly domain and the array
  domain differ deliberately).
- Compression: **keep `savez_compressed`** per Beta's D6.1 ruling (durable run
  checkpoint ≠ worker transport artifact; D5 §6.7.A stays untouched). Re-assert the
  separation in a gate.
- The checkpoint lives under `.s172_checkpoint/` (D6.1), never a finalizer-owned
  path. Do not reintroduce the D3.5 symlink collision.

**First deliverable step: read `CANONICAL_RECORD_FIELDS` and enumerate exactly which
24 fields must round-trip, with dtype and encoding for each. Report that table
before writing the writer.**

## 3. Transaction identity (Beta's D6.1 contract, now applied to a 24-field pair)

Both checkpoint members must carry matching transaction metadata:
`checkpoint_id`, `checkpoint_sequence`, `run_id`, logical candidate count,
canonical content digest.

Restart behaviour, exactly as Beta specified:
- matching IDs/digests → **pair accepted**;
- mismatched IDs → **interrupted sequential replacement detected**;
- malformed/unreadable member → **recover from the valid member where possible**;
- neither valid → **fail closed WITHOUT clearing in-memory state**.

The repaired pair must be **regenerated and revalidated before normal flushing
resumes.**

## 4. The finalizer resume path

The point of a 24-field checkpoint is that a run interrupted mid-flight can be
resumed and still produce a correct generation.

Required: on resume, read the checkpoint, rebuild the raw candidate records, and
produce a certified generation **field-for-field identical, and in identical
canonical order, to an uninterrupted reference run** on the same inputs.

**Beta's locked end-state claim — this is what D6.2 may assert:**
> Interrupted sequential replacement is detectable, and restart recovery
> reconstructs the same canonical cumulative checkpoint as an uninterrupted
> execution.

It may **not** claim the two-file checkpoint is jointly atomic. "Atomic checkpoint"
appears only with the qualification *each artifact replacement is atomic; the pair
is not jointly atomic.*

## 5. Enabling the list-clear (the actual OOM fix)

Only after §2–§4 hold may the clear be enabled. Required ordering is Beta's, verbatim:

```
construct cumulative canonical state
write both temporary artifacts
fsync/close as required
validate both temporary artifacts
replace destination A
replace destination B
validate the installed pair
only then clear the flushed in-memory entries
```

A mutant that clears **after the first replace but before the second** must fail.
D6.1 already gates the ordering property with the flag forced on; D6.2 turns the
flag on for real and must additionally prove the finalizer still receives complete
24-field input after a clear (i.e. via the resume path, not the truncated stump).

## 6. Gates — `tests/test_s172_d6_2_checkpoint_reconciliation.py`

Beta's required duplicate-seed matrix (§ from the D6.1 ruling) — for the same seed
appearing on both sides:

| case | required behaviour |
|---|---|
| identical records | reconciles to that record; idempotent |
| different match rates | canonical `_l2_sort_key` winner, float32 domain |
| float64-only difference (beyond float32) | **exact tie → falls through to trial-number tiebreak** |
| different trial_number, same score | **lower trial_number wins** |
| same trial + same mode | **raises `AccumulatorConsistencyError`** |
| different provenance/mode metadata | canonical rule decides; no ad-hoc choice |
| restart-replay duplicate | idempotent, no double-count |

Plus:
- **G-24-FIELD-ROUNDTRIP:** all 24 fields round-trip exactly; `allow_pickle=False`;
  no object arrays; dtypes preserved.
- **G-AUTHORITY:** reconciliation calls the canonical authority (AST + runtime — no
  second policy anywhere in the flush path).
- **G-IDENTITY:** the five transaction-identity fields present and matching on a
  healthy pair.
- **G-RESTART-{A,B,C,D}:** the four restart outcomes of §3, including **fail-closed
  without clearing** when neither member is valid.
- **G-RESUME-PARITY:** a run interrupted at each write/replace boundary, resumed,
  produces a generation **field-for-field and order-identical** to an uninterrupted
  reference run.
- **G-CLEAR-SAFE:** with the clear enabled, the finalizer still receives complete
  24-field input.
- **G-CADENCE:** D3.25's one-attempt-per-trial invariant unchanged; distinguish
  **one attempt per trial**, **one successful transaction per successful flush**, and
  **recovery retries** (recovery actions, not extra trial-triggered flushes).
- **G-COMPRESSION-CONTRACT:** D5 §6.7.A artifact ban still separate and intact.
- **G-NO-SYMLINK-COLLISION:** the checkpoint never writes a finalizer-owned path;
  publication still succeeds after arbitrarily many flushes.

**Mutants** (four-part kill rule; each must fail from its injected defect, and the
harness must swap the source every gate builds from — see the D6.1 vacuous-mutant
lesson): reintroduce the inline `score >` policy; compare float64 instead of
float32; drop the trial_number tiebreak; swallow the same-trial/same-mode collision
instead of raising; clear between the two replaces; drop a transaction-identity
field; write to a finalizer-owned path.

## 7. Non-regression

Capture green before any edit: D1.1 · D1.0 · D0 · D2 · D3.0 · D3 · D3.25 · D3.5 ·
D4 · D5 · D6 3.A · D6-threshold · D6.1 · Phase 3 · Phase 4. After: all green plus
D6.2. **D3.25 must stay 13/13** (cadence) and **D3.5 60/60** (the finalizer is
touched by the resume path — this is the suite that guards it).

## 8. Scope — do NOT touch

The D6 threshold/provenance/residue work; PWC/ZMQ ingress; the D3.25 four-map
contract; `TestResult` shape; D5's artifact contract; `serial_reference` as default.
Do not modify `_l2_sort_key` or `_select_l2_winners` — they are frozen under Ruling
D and are the authority being *reused*, not revised.

## 9. Report

The 24-field table (field, dtype, encoding) from §2; the reconciliation call path
proving the canonical authority is invoked and no second policy exists; the four
restart outcomes; the resume-parity evidence; gate/mutant counts; confirmation
D3.25 and D3.5 are unchanged; and the exact end-state claim language from §4. Then
STOP for Team Alpha review.
