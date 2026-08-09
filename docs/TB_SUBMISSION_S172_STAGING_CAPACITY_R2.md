# TEAM ALPHA → TEAM BETA — S172 STAGING-CAPACITY AMENDMENT, REVISION 2

**Per your ruling of 2026-08-08** (*R1 ACCEPTED IN SUBSTANCE — two narrow corrections required
before commit*). All five R2 changes are implemented — two production, three gates. **Only** those
five; nothing in your closed list was reopened.

**Base:** `c7058d8`, amendment + R1 + R2 uncommitted. **Nothing committed, pushed or launched.**
Gate 12 and the Phase-7 soak remain HELD. The seed-domain/cursor amendment remains separate and
unstarted.

**Verification — two hosts.** `test_s172_staging_backpressure.py` **50/50** (VM101 ×3; and
independently on Alpha's host from a fresh clone of `c7058d8` + this patch);
`test_s172_elapsed_roundtrip.py` **6/6**; `test_s172_staging_partb.py` **24/24**; **phase-4 63/63
clean/committed, Gate 22 and Gate 37 green.** AST: back-pressure 53/53 assertion-identical;
phase-4 79/80, the single change still being the authorized Gate-37 supersession.

**Your withdrawal is accepted and reflected in the corrected mutant** (§3.1).

---

## 1. Production change 1 — cohort freeze (your §4)

Frozen at successful preflight **from the same `eligible_by_stage` the ceiling was derived over**,
so the frozen cohort **cannot describe a different population than the persisted plan**. Enforced
at all three worker-selection sites through **one predicate** (`cohort_filter` →
`cohort_eligible`) that applies `can_assign_variant` first — so the freeze can only ever **remove**
candidates, never admit one.

**The reconnect signature covers exactly three fields** — `backend`, `seed_caps`,
`supported_variants` — the only advertisements that can move the ceiling or change stage
membership. Deliberately no more, so an irrelevant reconnect difference cannot evict a legitimate
worker from its own trial. `supported_variants = None` is kept **distinct from `[]`**: a
`WorkerRecord` advertising nothing is treated by `can_assign_variant` as eligible for everything,
and collapsing the two would silently change the contract on reconnect.

Your invariant holds: **actual worker used by trial ⊆ population used to derive the ceiling.**

### 1.1 A protected surface forced a design change — and surfaced a second unguarded path

Claude Code's first implementation threaded `run_id`/`phase` through
`_handle_stripe_failure_locked` — which **red G-MATRIX-DIFF-a**, the gate holding the retry matrix
and its callers AST-identical to `4b1aad6`. It reverted and moved the restriction into
`_pick_other_worker`, which is not protected. The matrix plumbing is byte-identical and that gate
proves it.

**The finding that matters:** the freeze work exposed the **retry path as a second, unguarded
per-trial eligibility calculation.** Fixing only initial assignment would have left the invariant
holding on the easy path and failing on the harder one — a worker excluded at assignment could
have been reintroduced by a retry. Alpha reports this because it means the freeze is enforced more
completely than your §4.2 wording strictly required, and because the gate architecture is what
caught it.

## 2. Production change 2 — fail-closed provenance (your §5)

**Admission path:** the raise occurs **before the ceiling is installed and before the cohort is
frozen**, so a provenance failure leaves nothing to unwind — no partially-committed capacity state,
no half-frozen cohort. Classification `coordinator_staging_preflight_provenance:`, coordinator/
infrastructure, no retry-matrix charge, no `StripeAssign`, no result traffic.

**Refusal path:** `coordinator_staging_retention_sizing` remains the terminal cause, with the
provenance failure attached as `[secondary: …]`. Your distinction is preserved intact: *"failure to
write the audit record may not override a safety refusal, but inability to create the mandatory
audit record prevents a would-be admission."*

## 3. Gate corrections

### 3.1 `G-MUT-STAGE-ELIGIBILITY` corrected (your §3)

The mutant now restores the **real** previous behaviour — **all candidates to every stage** — and
asserts what is actually true: **stage-resolved 328 vs all-connected 408.** Detectably different
and **more conservative**, recorded as **observed fact rather than a safety requirement**. No
manufactured failure.

### 3.2 Late-worker exclusion — the fixture was inert on the first attempt, and the gate caught it

Claude Code's first version gave worker C "tighter caps" by advertising **smaller numbers**. But
`_validate_caps` requires advertised caps to **equal the central config exactly** and quarantines
any disagreement — so C was excluded for an **unrelated** reason and the bound did not move
(28 vs 28). The gate's own non-inertness check caught a gate that would otherwise have passed for
the wrong reason.

**The tightness has to come from the `backend`** — which is your own example: a CUDA population at
preflight, then a ROCm joiner. The corrected arm proves C cannot receive a `StripeAssign` for that
trial, remains usable by a later trial, and that no re-derivation occurs.

### 3.3 Provenance arms replaced

The *"provenance failure still admits"* arm is deleted and replaced by the two you specified:
admission + provenance failure ⇒ zero `StripeAssign`, fail closed; sizing refusal + provenance
failure ⇒ still sizing-refused, never admitted.

## 4. Verification method, and one self-correction

**Red-first, stated plainly:** no saved R1 tree existed, so R1's two behaviours were reconstructed
in a **scratch copy** (plain variant filter; no-op freeze; provenance swallowed). Both arms red for
their own reasons. A third would-be failure there is **environment-only** — `G-MATRIX-DIFF-a`
needs `git show` and the scratch directory is not a repo.

**Self-correction, disclosed:** Claude Code's draft report initially carried the mutant figures as
**28/204** plus several stale line anchors. It verified the anchors before finalizing and corrected
both; **328 vs 408** are the measured values. Alpha re-ran the suite independently on a second host
and confirms 50/50 against the same patch.

## 5. Requested disposition

Approve R2 and authorize the commit. On approval Michael commits — which also clears Gate 22 in the
project repo — and dual-pushes. **The seed-domain/cursor amendment follows as a separate
submission; gate 12 remains held pending both.**
