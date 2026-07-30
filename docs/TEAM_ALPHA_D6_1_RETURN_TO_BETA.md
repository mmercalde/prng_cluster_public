# TEAM ALPHA → TEAM BETA — D6.1 return package + scope-change ruling request

**Re:** `D6.1 — incremental NPZ atomic flush and durability repair`.
Base `4c421fc`. **Nothing committed, nothing pushed, WATCHER not run.** D6.1 is
stopped at the review gate.

**Ruling requested (§6):** Beta must rule on a scope change — *repair the flush in
place* → *relocate the checkpoint to its own namespace and defer the list-clear.*
Alpha authorised Option 1 provisionally so the work could proceed; Beta owns the
final call.

---

## 1. Why the scope changed — a fifth defect made the briefed repair unsafe

All four briefed defects were confirmed at source, and D1 was **reproduced**
(numpy 1.22.0 wrote `...flush.tmp.npz`; `os.replace` raised `FileNotFoundError`).
Incremental durability has never existed.

But the briefed repair would have broken generation publication. Verified by
reproduction, not inference:

- The checkpoint targeted relative `bidirectional_survivors_all.npz` /
  `..._binary.npz`.
- **Since D3.5 those two paths are finalizer-owned compatibility symlinks**
  (`utils/run_finalizer.py:1400-1404`), and the finalizer **fails closed** if a
  regular file appears at either.
- Reproduction: repaired flush → symlink replaced by a regular file → next
  `finalize_run` raises `PublicationError`.

**Severity, stated precisely:** this is a persistent-until-fixed failure, not data
loss. Existing certified generations are unaffected; recovery is deleting the
regular file and letting the finalizer recreate its symlink. But because the
checkpoint rewrites that path on every flush, it would re-break publication after
every cleanup — a fix that would be reverted within a day.

**Second consequence — the list-clear would truncate the finalizer's input.**
`window_optimizer_integration_final.py:1868` feeds the **in-memory list** to the
D3.5 finalizer, which requires all **24** `CANONICAL_RECORD_FIELDS`. The checkpoint
persists **4** arrays. Enabling the clear (Alpha's brief requirement 3) would hand
the finalizer a 4-field stump. Upstream trial data still exists, so the cost is a
re-run rather than a loss — but the generation for that run would be built from a
truncated input.

**Root cause:** S152's checkpoint predates D3.5, which reassigned those paths under
Ruling F. The helper is a vestige whose targets were taken out from under it. The
collision stayed invisible **only because the write never succeeded** — and that
same fact makes relocation free, since nothing can depend on the checkpoint's
current path or content.

**Alpha owns this miss.** The D6.1 brief specified requirement 3 without checking
what consumed the in-memory list afterward. Claude Code caught it by reading the
consumer rather than trusting the brief. This is the third time in S172 that the
real defect lived at a component seam, and the second time the brief itself was the
defective artifact.

## 2. What shipped (Option 1 — own namespace, clear deferred)

Checkpoint relocated to `.s172_checkpoint/`, never the finalizer's aliases. All four
briefed fixes retained, plus `fsync` and stale-temp cleanup. The list-clear is
**deferred, not enabled** — its ordering property is nonetheless gated with the flag
forced on (M2), so it is provably correct if ever turned on.

**Suffix mechanism — open file handle.** Both candidates were validated empirically;
the handle won because it is the **only** option that also permits `fsync` on the
exact descriptor NumPy wrote. Naming the temp `...tmp.npz` would require a reopen,
leaving a window in which the "durable" temp is not durable. Alpha considers this
reasoning better than either option the brief offered. `G-SUFFIX` pins the
*property* (temp name is the D1-prone shape; the file NumPy creates equals the path
`os.replace` consumes), so it holds for any future naming. M1 restores the
un-suffixed form and reds.

**Exception contract — non-fatal, three visible tiers.** Expected/recoverable
(unreadable prior checkpoint) → stdout warning, flush still succeeds. Write failure
(`OSError`) → stderr ERROR + traceback + counter. Unexpected (anything else) →
stderr UNEXPECTED ERROR + traceback + counter. **All candidates retained on every
failure path.** `_flush_success_count` / `_flush_failure_count` /
`_flush_last_error` are observable by a soak. M5 re-broadens to the old single
stdout warning and reds.

**Three crash points characterised (Beta's identity/recovery contract).**
(a) *Before any replace* — both finals at complete prior content, no temps, all
candidates in memory. (b) *Between the replaces* — inconsistent pair (`_all`
advanced, `_binary` not); each file individually complete and `testzip()`-clean;
**detectable by seed-set comparison** and **self-repaired by the next flush**.
(c) *After both* — consistent pair; replay is idempotent via merge-by-seed dedup, so
no double-count. Described throughout as **sequential-atomic with self-repair**;
**joint atomicity is never claimed**, per Beta's locked language.

**D3.25 cadence unchanged at 13/13.** Its flush gate swaps in a spy and never
executes the body, so the repair cannot reach it. `G-CADENCE` therefore re-asserts
the invariant **independently by AST** — exactly one call site in each adapter —
plus a verbatim pin of the entry gate against `2a6e0f8`.

**D5's compression ban untouched.** D5 never scans the integration file; its ban is
scoped to `miner/assembly_shard_worker.py:308`, unmodified and still `ZIP_STORED`.
`G-COMPRESSION-CONTRACT` proves the two contracts stay **separate** rather than
harmonising them — matching Beta's rationale (worker transport artifacts on the hot
parallel validation path vs. a durable run checkpoint at one-flush-per-trial
cadence).

## 3. Results

**D6.1: 13/13 gates, 7/7 mutants killed.** Full non-regression green before and
after: D1.1 18/18 · D1.0 8/8 · D2 7/7 · D0 12/12 · D3.0 10/10 · D3 10/10 ·
D3.25 13/13 · D3.5 60/60 · D4 8/8 · D5 24/24 · D6 3.A 9/9 · D6-threshold 17/17 ·
Phase 3 17/17 · Phase 4 63/63.

## 4. Two self-caught process defects worth recording

1. **A vacuous mutant.** The first harness passed the mutated module as an argument
   while the gates built their own from production — **M2 survived without proving
   anything.** The survivor exposed it; the runner now swaps the source every gate
   builds from. Without that catch the mutation proof would have been worthless.
   This is the four-part rule catching a *harness* defect rather than a code defect.
2. **A stale allowlist assertion.** Registering the new gate in Phase-4 Gate 22
   surfaced an allowlist assertion that `_flush_npz_incremental` is byte-identical
   to `2a6e0f8` — now false. Corrected, on the explicit grounds that a load-bearing
   comment stating a guarantee that no longer holds **is precisely defect D4**.

## 5. What D6.1 does NOT deliver — the deferred item

**S166 in-memory OOM protection remains unimplemented.** The clear existed to stop
the candidate list growing unboundedly on long runs; it cannot be safely enabled
until the checkpoint persists full 24-field canonical records **and** the finalizer
has a read-back/resume path. That is recorded in
`docs/SESSION_CHANGELOG_20260729_PHASE5_D6_1.md` as its own tracked **Phase-7 soak
blocker**, not folded into D6.1.

Alpha is submitting a full brief for that work separately
(`docs/CLAUDE_CODE_INSTRUCTIONS_S172_D6_2_CHECKPOINT_RECONCILIATION.md`) rather
than leaving it as a changelog note, since it has real design content — schema,
reconciliation authority, and the resume contract — and it gates Phase 7.

Exploratory ROCm test: **not run.** Scope held; the repair is CPU-only and could not
satisfy any Phase 6.0 criterion. Still available before Phase 6.0.

## 6. Rulings requested

1. **Approve the scope change** (repair-in-place → relocate + defer the clear), or
   direct otherwise.
2. **Confirm `.s172_checkpoint/`** as the checkpoint namespace, or specify another.
3. **Numbering and sequencing of the deferred work.** Alpha proposes **`D6.2 —
   24-field checkpoint + finalizer resume path`**. It *must* land before Phase 7.
   Alpha's recommendation is to run it **after** Phase 6.0 and Phase 6, so the
   long-awaited ROCm and four-path validation are not delayed by a durability
   feature that only bites at soak scale — but this is a priority call Beta should
   make explicitly rather than inherit.
4. Note for the record: Alpha's brief was the defective artifact in this cycle
   (§1). Alpha will add a consumer-check step to future brief preparation — for any
   change to a shared buffer or on-disk path, identify every consumer before
   specifying the change.
