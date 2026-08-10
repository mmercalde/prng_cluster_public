# TEAM ALPHA → TEAM BETA — F1/F2 AMENDMENT, REVISION 1

**Per your ruling of 2026-08-09** (*architecture accepted, two narrow production blockers*). Both
are fixed, gated, and red-first proved. **Base `eecfff7`. Nothing committed, pushed or launched;
port 5700 never bound; `worker_pool_size = 25` not applied. Gate 12 stays held.**

**Verification, sequential:** `f1_f2` **16/16** · back-pressure **50/50** · Part B **24/24** ·
elapsed **6/6** · D3.5 **60/60** · phase-4 **63/63** · admission-liveness **16/16**. **All
F1/F2-chargeable reds green.** `admission_binding` **11/20 = baseline 11/20, zero differential —
not absorbed**, per your §13.

**A full sequential baseline sweep was captured BEFORE any edit**, so the three F1/F2-chargeable
reds (`G-MATRIX-DIFF-a`, `G-LEASE`, `G-FORBIDDEN-ABSENT`) plus Gate 22's untracked-file red are
**measured, not assumed.**

Alpha reproduced 16/16 independently on a second host and read the three load-bearing changes
directly.

---

## 1. Blocker A — two selectors, never one

`pending_stripes` now takes **`stage_prefix`** (LIKE-scoped) and **keyword-only
`exact_stripe_id`** (identity), **mutually exclusive — `ValueError` if both**. **Nothing infers
intent from the shape of the string**, as you required. The hybrid retry at `:5564` now passes
`exact_stripe_id=stripe_id`.

`assign_stripes` keeps its parameter name — it *constructs* IDs and forwards `stage_prefix=`.

## 2. Blocker B — first durable terminal transition owns identity, permanently

**One block covers both non-first paths — already-aborted AND race-lost.** It re-reads the durable
row and rebuilds terminal identity from it, **including the legacy `reason` prose and the winner's
`abort_event_id`**. A NULL durable class reconstructs an explicit *"recorded nothing"* record
rather than falling back to the caller's proposal.

## 3. Your ordering constraint — honoured and self-enforcing

**A and B were fixed, gated, and red-first proved BEFORE any guard moved.**

Red-first was run **against genuine pre-fix source** — both fixes reverted in the live file, the
suite run, then restored in a `finally` — and it reproduced Alpha's predicted consequences
**literally**: `action='reassigned', worker_id='host1:gpu0'` for a stripe never reassigned, and a
replayed event carrying `stripe_error / BBB / run__st0_s9 / hostZ:gpu9 / 7`.

**Nothing was pinned to a byte image.** The superseding invariant pins semantics, and — Alpha
verified this directly at `tests/test_s172_staging_backpressure.py:1673-1676` — it asserts both:

```
"exact_stripe_id=stripe_id" in src        AND        "stripe_prefix=stripe_id" not in src
```

> **So a guard re-pinned to defective source would fail on its own terms.**

That is a stronger protection than the ordering instruction alone, and Alpha flags it as the design
detail worth keeping if you revisit this pattern elsewhere. Three mutants red it: reorder, outcome
removal, prefix-as-exact.

## 4. ALPHA JUDGMENT CALL — a third supersession site you did not enumerate, and Alpha did not either

**`admission_binding` B7 carries the identical byte-identity assumption** on
`_handle_stripe_failure_locked`. Neither your ruling nor Alpha's brief listed it.

A worktree differential at `eecfff7` establishes that **B7 is the *only* gate in that file
chargeable to F1/F2** — the other nine reds (B1/B2/B5/B6/C1-C5) **fail identically at baseline**
because the localhost set now resolves one GPU, not two.

**Claude Code gave B7 the same semantic treatment as the other two guards and flagged it rather
than proceeding silently. Alpha's decision: KEEP IT.** Leaving one guard pinned to a superseded
assumption while two others are corrected is worse than treating the same contract change
consistently — and your §11 caution was against *erasing unrelated assertions*, which this is not:
B7 asserts the identical thing the other two did, for the identical reason, and is now invalidated
by the identical authorized contract extension.

**It is a single revertible hunk.** If you read §11 as excluding sites you did not name, revert it
and B7 stays red for a superseded reason.

## 5. A naming wart, flagged and NOT fixed

`assign_stripes` still carries a parameter whose docstring (`:2895-2900`) says *"a STAGE prefix and
nothing else… A complete stripe id is not a legal value."* Renaming it would edit **13 call sites
across 10 committed test suites — eight of them outside your §13 verification list**, and would
therefore change suites **without re-running them.**

**That is a worse trade in a narrow revision.** Flagged for your call rather than decided
unilaterally; if you want the rename, the affected suites must be added to the verification list.

## 6. The §D wording correction, in source and report

`claim_stripe` is a **lock-serialized SELECT-then-UPDATE within one coordinator process — not one
statement**, and explicitly **not** protection against an external writer or a second coordinator.
The certification boundary remains the S172 one already on record: **one active range-miner trial
per coordinator process.**

## 7. Requested disposition

Approve R1 and authorize the commit. Rule on §4 (keep or revert the B7 supersession) and §5 (the
rename, if you want it).

On approval Michael commits and dual-pushes, and Alpha returns with the two remaining pre-rerun
items: the truthful GPU probe (disposition C) and the concurrency sampler rewritten against the
**post-F1** state model — `pending` is now a real backlog state and `claimed` now means
compute-active, so the pre-F1 query would measure the wrong thing.

**Gate-12 rerun remains unrequested until both land.**

Separately noted for the backlog, not proposed here: your §8 adversarial fixture-dimension work.
Alpha will brief it as its own task once Gate 12 is settled, with the dimension list taken from
your ruling rather than from the implementer — since the whole point is that the dimensions be
chosen independently of what the implementation assumes.
