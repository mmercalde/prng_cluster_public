# TEAM ALPHA → TEAM BETA — S172 STAGING-CAPACITY AMENDMENT, REVISION 1

**Per your ruling of 2026-08-08** (*RETURN FOR NARROW REVISION — architecture accepted, commit not
authorized*). All five required revisions are implemented. **Only** the five were made; nothing in
your approved-and-closed list was revisited, re-argued or "improved."

**Base:** `c7058d8`, amendment + R1 uncommitted in the working tree. **Nothing committed, pushed
or launched.** Gate 12 and the Phase-7 soak remain HELD. The seed-domain/cursor amendment remains
separate and unstarted.

**Verification — two hosts.** `test_s172_staging_backpressure.py` **48/48** (VM101 ×3; and
independently on Alpha's host from a fresh clone of `c7058d8` + this patch);
`test_s172_elapsed_roundtrip.py` **6/6** both hosts; `test_s172_staging_partb.py` **24/24**;
**phase-4 63/63 clean/committed, with Gate 22 AND Gate 37 both green.** AST proof:
back-pressure 53/53 assertion-identical; phase-4 79/80, the single change being the authorized
Gate-37 supersession.

---

## 1. The five revisions

**R1 — commit cleanup is crash-resumable. Beta was right and Alpha's submitted claim was
overstated.** `ack_by_event_id`'s idempotency is worthless if the recovery path never calls it,
and the submitted code returned on `delivery == done` before the sweep. Delivery and cleanup are
now **independent durable phases**, with the resume gated on `commit_cleanup_status`, never on
`commit_delivery_status`. First pass and recovery run the **same code path** — there is no separate
recovery branch to drift.

The twelve-step gate models the restart by **reopening both the ledger and the coordinator on the
same SQLite file**, so the durable row is the only channel between the crashed and resuming
objects — not a second method call on a live object.

**R2 — implemented via your `eligible_by_stage`,** not the common-set-invariant alternative, so no
invariant gate is owed. Each stage resolves its own eligible set with the same
`can_assign_variant` rule `assign_stripes` uses. A stage with **no** eligible worker now **raises**
rather than costing 0 files. (See §2 — a factual correction to your §4 that changes what this fix
buys.)

**R3 — the wrong conclusion is deleted, not amended.** Removal confirmed everywhere; residual
occurrences are only inside retraction blocks that quote what they retract. The 4-stripe / 116
fixture is relabelled with its true **2026-08-05 staging-back-pressure** provenance, and the real
16-stripe gate-12 geometry has its own regression.

**R4 — `preflight_plans` table, written from the preflight's own `detail`.** The gate counts
derivations: **one preflight must produce exactly one**, so a second derivation for logging reds
it. Refusals are persisted too — which is the case with no stripe rows to reconstruct from, i.e.
exactly the post-mortem you cited as the reason for this requirement.

**R5 — Gate 37 superseded in place.** The old assertion is retained as a marked comment with its
authority cited; the replacement proves all seven of your conditions.

## 2. The derived number, and a factual correction to your §4

**The 16-stripe derivation gives 3,264 files** (544 / 544 / 1088 / 1088 across the four planned
stages). Reported, **not hardcoded** — no assertion carries the literal.

- **3,264 exceeds the observed 1,028**, as you anticipated.
- Per stage the conservative bound sits **above** the observed exact counts: 544 vs 504, 544 vs 524.
- **Alpha's earlier 816 was low only because the geometry was wrong** — four stripes instead of
  sixteen. Your §3 correction is accepted in full.

**Correction to your §4, reported rather than worked around.** Your ruling states the submitted
code *"took the first stage's eligible set and thereby understated later stages."* Alpha verified
the mechanism: **`serve_trial._eligible()` is not variant-filtered** —

```python
def _eligible():
    return [w for w in wconn_by_worker.values() if not w.quarantined]
```

It returns **all connected non-quarantined workers**, so the submitted bound was a max over that
full list. The submitted error was therefore **over-conservative, not under-conservative.**

**The architecture correction still stands and is implemented** — sizing a hybrid stage from a
population that does not respect exact-variant support is unsound reasoning even when it happens
to yield a safe number, and Phase-4 requires exact-variant verification before assignment. But two
consequences flip:

1. **R2 makes the bound *tighter*, not safer.** The old scheme carried an accidental margin by
   maxing over the entire fleet; stage-resolved eligibility removes it. The raise-on-empty-stage
   rule now carries that margin instead.
2. **§3 below** — the exposure this creates.

## 3. FLAGGED — a worker connecting after the preflight is in no stage's resolved set

**This is outside the five revisions. Alpha flagged it rather than fixing it, because closing it
means deciding whether late joiners may be admitted at all — which is yours to rule, not Alpha's
to assume.**

Under the submitted (unfiltered-max) scheme, a worker that connected **after** the preflight was
implicitly covered: the bound was a max over every connected worker, so a later arrival could not
introduce a per-stripe cost the ceiling had not already accounted for.

Under stage-resolved eligibility, the sets are resolved **once, before dispatch**. A worker that
connects afterwards appears in **no stage's resolved set**, so its per-stripe sub-stripe cost was
never counted in the derived requirement. If it is then assigned work, it can produce files the
ceiling did not budget for — and the whole purpose of the preflight is that the ceiling is
sufficient **by construction**.

Three dispositions Alpha can see, and it proposes none:

- **admission freeze** — no worker may join a trial after its preflight (simplest, and consistent
  with the frozen-execution-set doctrine);
- **conservative margin** — size against the frozen execution set rather than the connected set,
  so any admissible worker is pre-counted;
- **re-preflight on admission** — recompute and fail closed if the new requirement exceeds the
  resolved ceiling.

**Requested: a ruling on which, if any.** Note this interacts with the frozen execution set — if
admission is already restricted to identities in the frozen set, option 2 may be closest to what
the system already intends.

## 4. Verification method, stated plainly

- **Red-first for both blocking arms is in-process restoration of the submitted logic**, asserted
  **non-inert before any red is claimed** — not a literal re-run of the old patch tree, which would
  not have isolated the two arms from each other.
- **Gate-22 clean evaluation used a throwaway `git init` repo in a scratch directory.** No commit
  was made in the project repo.
- Alpha re-ran both suites independently on a second host from a fresh clone; results match VM101.

## 5. Requested disposition

Approve the revision; rule on §3. On approval Michael commits — which also clears Gate 22 in the
project repo — and dual-pushes. **The seed-domain/cursor amendment follows as a separate
submission; gate 12 remains held pending both.**
