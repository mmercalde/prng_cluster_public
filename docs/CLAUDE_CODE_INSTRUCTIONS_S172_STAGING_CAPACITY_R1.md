# CLAUDE CODE INSTRUCTIONS — S172 STAGING-CAPACITY AMENDMENT, REVISION 1

**Host:** VM101, repo `~/distributed_prng_analysis`. The amendment is **uncommitted in the working
tree** at base `c7058d8`. `source ~/venvs/torch/bin/activate` before every test. Long suites:
`python3 -u <suite> | tee /tmp/<name>.log` — never `| tail`.

**Authority:** Team Beta ruling *"S172 STAGING-CAPACITY AMENDMENT + `elapsed_s`"* (2026-08-08).
**Disposition: RETURN FOR NARROW REVISION — architecture ACCEPTED, commit NOT authorized.**

**APPROVED AND CLOSED — do not revisit, do not re-argue, do not "improve":**
the Option-C architecture; `elapsed_s` in full (including `elapsed_s: Optional[float] = None`);
reuse of `ack_by_event_id` as the single release mechanism; `commit_cleanup_status` separate from
`abort_cleanup_status`; retention after sink failure; derive-by-default instead of a new magic
number; the full manifest→CLI→coordinator route for both high-waters; non-retryable classification
before dispatch; executor waits under the same lock/age domain as reader pauses; clearing executor
wait records in `finally`; the runtime-only byte ceiling; the `effective_high_water_files()`
no-plan fallback (accepted narrowly, §5 below).

**Beta §10 — make ONLY these five revisions.** *"No broader cleanup, telemetry expansion,
`.gitignore` work, seed-domain/cursor work, or Gate-12 execution belongs in this revision."*

**Hard constraints:** no commit, no push, **no pipeline launch, no fleet launch, no port 5700
bind.** Gate 12 and the Phase-7 soak are HELD. Do not touch the seed-domain/cursor surface. No
telemetry beyond `elapsed_s`. If a fix appears to need a file outside the amendment's existing
change set, STOP and report.

**Base verification:** the working tree must still carry the submitted amendment —
`git log --oneline -1` = `c7058d8`, `git diff --stat` showing the six amendment files, and
`python3 -u tests/test_s172_staging_backpressure.py` → **42/42**, `test_s172_elapsed_roundtrip.py`
→ **6/6**. Untracked runtime residue (WAL sidecars, `*.stale_*` rotations, delivered briefs) is
expected and is **not** a stop condition. Report the state; stop only on unexpected tracked drift.

---

## REVISION 1 — COMMIT CLEANUP MUST BE CRASH-RESUMABLE (Beta §2, BLOCKER A)

**Beta named this the first thing to fix.** The report claimed reuse of `ack_by_event_id` made
successful-commit cleanup safe across *"a duplicate commit, a crash between rows and a concurrent
abort."* **The crash-between-rows half is not satisfied.**

The submitted `commit_trial` treats `commit_delivery_status == done` as proof that cleanup already
happened, and on that branch marks the call duplicate and **returns before the reservation-discharge
sweep**. The stranding window:

```
sink commit returns successfully
  → commit_delivery_status = done
  → release reservation 1
  → PROCESS CRASHES
  → reservations 2..N remain held, commit_cleanup_status != done
  → restart, same commit event
  → delivery_status already done → duplicate branch RETURNS
  → remaining reservations are NEVER discharged
```

`ack_by_event_id` being idempotent is **necessary but not sufficient if the recovery path never
calls it.**

**Required rule — delivery and cleanup are two INDEPENDENT durable phases:**

```
IF delivery_status != done:
    call sink
    on success, persist delivery_status = done

IF delivery_status == done AND cleanup_status != done:
    DO NOT call the sink again
    resume the idempotent held-reservation sweep
    persist cleanup_status = done when complete

IF delivery_status == done AND cleanup_status == done:
    duplicate completed commit — release nothing, return
```

You already have the durable state needed; the defect is returning too early on the
`delivery == done` path. Preserve every property Beta approved: the sink is not called twice after
durable delivery, sink failure retains everything, `ack_by_event_id` stays the single release
mechanism, partial cleanup resumes safely, already-acked rows remain no-ops.

**Required gate — fault injection, all twelve steps (Beta §2):** N ≥ 3 held reservations; sink
commit succeeds; **fault after exactly one successful `ack_by_event_id`**; verify
`commit_delivery_status == done`; verify `commit_cleanup_status != done`; verify some reservations
remain held; **recreate/reopen the coordinator and ledger to model a process restart rather than
merely another method call** (do this if practical — if it is not, say so explicitly and state what
you did instead); call `commit_trial` for the same event; prove the sink is **not** called again;
prove all remaining reservations/files are discharged; prove the first reservation is **not**
discharged twice; prove `commit_cleanup_status == done`.

**This arm must be RED against the current submitted patch.** Prove it.

## REVISION 2 — STAGE-SPECIFIC ELIGIBILITY IN THE WHOLE-TRIAL PREFLIGHT (Beta §4, BLOCKER B)

The report describes the bound as *sum over stages, sum over stripes, **max over eligible
workers***. But the submitted coordinator computes the preflight when the **first stage's** eligible
set becomes available and passes that one collection across every planned stage.

That is safe only if the eligible set is identical for every planned concrete variant — and the
Phase-4 contract does not grant that. It **requires** the coordinator to verify a worker advertises
support for the **exact concrete variant** before assignment, so eligibility is family/phase
dependent by construction. Beta's example:

```
worker A supports java_lcg
worker B supports java_lcg_hybrid
```

If B has the tighter effective hybrid cap and is absent from stage 0's eligible set, sizing the
hybrid stage from stage 0's population **understates** that stage's file count — a
conservative-bound violation.

**Required correction — Beta's preference:**

```
eligible_by_stage[(family, phase)]
```

Resolve the eligible set for **every planned stage before the retention preflight**, using the same
exact-variant support and cap rules assignment will later use. Then:

```
for each planned stage:
    stage_requirement = conservative_bound(
        stripe_spans, eligible_by_stage[stage], stage.phase, stage.family)
```

**An alternative is acceptable ONLY if you prove and gate a stronger invariant** — that every
worker in the frozen execution set must support every planned variant, so all stage eligibility
sets are necessarily identical. Beta: *"Do not merely document the assumption."* If you take that
route, the invariant needs its own enforcing gate, not a comment.

**Required negative arm:** construct asymmetric variant support; demonstrate that the old
"reuse stage-0 eligibility everywhere" calculation **differs from / understates** the correctly
stage-resolved calculation; then prove the revised preflight uses the correct later-stage
population.

## REVISION 3 — CORRECT THE GATE-12 GEOMETRY EVIDENCE (Beta §3)

**Alpha's submitted §2.2 analysis was factually wrong and Beta corrected it. Delete the
conclusion.**

The real 2026-08-07 gate-12 production geometry is already recorded:

```
max_seeds         = 1,073,741,824
miner_stripe_size =    67,108,864
macro-stripes     = 16 per stage
stage 0 = 504 files · stage 1 = 524 files · total = 1,028
```

So **1,028 is simply stages 0 and 1 of a 16-stripe production run** hitting the 512 ceiling. It
does **not** imply ~five planned stripes. **Remove the "1,028 implies roughly five stripes"
conclusion wherever it appears** — the report, comments, docstrings.

The four-stripe figure (34+14+34+34 = 116 exact) is the **2026-08-05 staging-back-pressure
fixture**, built to prove the exact-vs-conservative burst-bound distinction. **Keep it** as a
compact mathematical/unit gate — but **label it with its true 2026-08-05 provenance**, not as
gate-12 geometry.

**Add the real gate-12 geometry regression** with `total_seeds = 1,073,741,824`,
`miner_stripe_size = 67,108,864`, `stripe_count = 16`, and let the production derivation compute
the full planned requirement. The gate establishes:

```
derived stripe_count == 16
derived requirement  > 512
explicit ceiling 512 ⇒ fail closed BEFORE StripeAssign
files ceiling None   ⇒ resolved ceiling == derived requirement
```

**Do not hardcode 1,028**, and **do not hardcode a newly hand-calculated full-workflow number
either** — Beta: *"The whole point of this amendment is that the answer comes from the execution
geometry."* Report the number the derivation produces; do not put it in an assertion as a literal.

## REVISION 4 — PERSIST THE PREFLIGHT GEOMETRY (Beta §5, now REQUIRED)

Alpha asked whether planned geometry should be recorded; **Beta ruled YES**, citing the 816/1,028
confusion as its own evidence. A derived safety decision that determines admissibility needs
durable provenance, and a post-mortem must not have to reconstruct what the coordinator believed
from surviving stripe rows.

**Persist the preflight plan before first dispatch.** Beta prescribes no SQL normalization, but the
record must be sufficient to **reproduce the decision**, at minimum:

`run_id`/trial identity · `total_seeds` · `miner_stripe_size` · macro-stripe count · actual stripe
spans (or a canonical representation/hash from which they are recoverable) · planned
`(family, phase)` stages · **stage-specific eligible execution set, or an immutable digest plus the
relevant backend/cap data** · per-stage derived file requirement · total derived requirement ·
high-water mode (`derived` | `operator`) · configured high-water when explicitly supplied ·
resolved high-water · timestamp/schema version.

**Write it from the SAME values the preflight consumes.** Beta: *"No parallel second derivation."*
Do not recompute for logging.

Note this interacts with Revision 2 — the stage-specific eligible sets are part of what must be
recorded.

## REVISION 5 — SUPERSEDE AND REPLACE GATE 37 (Beta §1, APPROVED)

Alpha's requested supersession is **approved**. The old *"file still exists after successful
commit"* assertion is **SUPERSEDED by the Option-C lifecycle**. Beta: *"Do not silently edit
history. Mark the old assertion as superseded… and replace it under that explicit authority."*

The replacement must prove **all seven**:

1. the manifest/staged object existed and was available to the sink before/during commit;
2. sink commit succeeded;
3. **only after** that success, the reservation was acknowledged;
4. the corresponding staged file is absent afterward;
5. the durable cleanup state says complete;
6. a duplicate completed commit neither re-delivers to the sink nor releases/deletes anything a
   second time;
7. the failed-commit path still retains the staged object.

**Gate 22 is not a regression** — the known dirty/untracked-`.py` sensitivity; it clears when the
test files are committed. Evaluate phase-4 in the clean/committed sense where you can and state
which method you used.

---

## VERIFICATION REQUIRED BEFORE RESUBMISSION (Beta §11)

- staging-backpressure suite: **all prior gates green plus the new crash-recovery and
  stage-eligibility arms**;
- `test_s172_elapsed_roundtrip.py` **6/6**;
- `test_s172_staging_partb.py` **24/24**;
- phase-4: Gate 37 under the superseding contract; Gate 22 evaluated clean/committed;
- **red-first evidence for BOTH new blocking arms** against the current submitted patch;
- pre-amendment gate assertions unchanged **except** the explicitly authorized Gate-37
  supersession — prove programmatically, the AST method already used;
- no gate-12 production run; no Phase-7 soak.

## REPORT

`docs/CLAUDE_CODE_REPORT_S172_STAGING_CAPACITY_R1.md`:

1. Per-revision implementation notes with `file:line`.
2. The two-phase commit rule as implemented, and every path that can now resume cleanup.
3. Whether you took Beta's `eligible_by_stage` route or the common-set-invariant alternative — and
   if the latter, the enforcing gate.
4. The number the 16-stripe derivation produces (stated in the report; **not** hardcoded in an
   assertion).
5. The persisted preflight record's schema, and proof it is written from the preflight's own
   values rather than recomputed.
6. Red-first and mutation evidence per new arm.
7. Full verification results, both hosts where applicable.
8. Files changed — expect the same six plus test additions. Anything else justified.
9. Any disagreement with this brief **reported, not worked around.**
10. Confirmation that the *"1,028 implies ~five stripes"* conclusion has been removed everywhere.
