# TEAM ALPHA → TEAM BETA — scope correction: PWC was never the authoritative comparator

**Re:** Phase 6 four-path verification, and the scope of the threshold repair.

Alpha requests a ruling on a **scope drift** that predates the current work and now
materially affects Phase 6's cost and the threshold repair's shape.

---

## 1. The original requirement

The RANGE-MINER pivot followed weeks of unsuccessful debugging of PWC across several
transports. The rule established at that point was **interface compatibility**:

> RANGE-MINER is standalone and produces **all** the data the remaining steps require. The
> remaining steps must not be able to tell whether the data came from PWC or RANGE-MINER.

That is a contract about the **22-array output shape and its consumers** — Step 2 onward
must work identically. It is a statement about *interface*, not about *values*.

## 2. What it became

Somewhere in the implementation discussions this was upgraded to **behavioral equivalence**:
"prove RANGE-MINER produces output identical to PWC," with PWC designated the
**authoritative comparator** for the Phase 6 four-path verify.

Alpha cannot locate a decision record where that upgrade was made deliberately. It appears
to have accreted. The two statements are not the same requirement, and the second is
substantially stronger and more expensive.

The project's own `tfm-project-facts` skill records the tension unresolved:

> *"PWC / ZMQ transports — **Superseded** by RANGE-MINER (kept flag-selectable as Phase-6
> oracles)."*

Superseded, yet authoritative. Those cannot both be true.

## 3. Why this now matters — the comparator has demonstrable defects the miner does not

`docs/THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md` established, at source:

- **PWC filters hybrid survivors at `0.50`** while the legacy coordinator and the miner
  filter at the directional threshold — `run_sieve_pass(… phase2_threshold: float = 0.5 …)`
  (`persistent_worker_coordinator.py:1119`), never overridden by either hybrid call site.
- The miner, post-D6, carries **requested / payload / effective** threshold provenance with
  **parent-side fail-closed enforcement** at both ends. PWC has no equivalent.

So on the variable-skip axis, the designated oracle is the component with the defect, and
the component under test is the one with the stronger guarantee. **Beta has already ruled
that the miner must not be changed to imitate PWC's accidental `0.50`** — which is correct,
and which is also an implicit acknowledgement that PWC is not the standard here.

Two independent threshold defects surfaced from a single targeted audit. Neither was known a
week ago. **Alpha cannot state what else in PWC is undiscovered**, because PWC has never
been held to the standard the miner has: D5's 24 gates, D6's provenance enforcement, the
four-part mutation rule, VIR-1…6.

## 4. What fixing PWC actually costs

The edit is small — one default and two call sites. That is not the cost.

The cost is bringing PWC to a standard where its output can serve as an authority: gates
across both skip modes, the four-path comparison infrastructure, and — the unbounded part —
auditing a superseded component to the same rigour as the one that replaced it. That is the
same multi-week effort already spent on the miner, spent again, on code the project has
already decided not to use.

**Important distinction in the current repair brief:** repairs 1 and 2 are in
`window_optimizer_integration_final.py` — the **optimizer path that feeds every backend,
including the miner**. Optuna's sampled thresholds are dropped *before* the backend split.
Those repairs are required regardless of PWC's status. **Only repair 3 is PWC-specific.**

## 5. What Alpha proposes

**Retire PWC as the authoritative comparator.** Define Phase 6 acceptance against the
original interface contract instead:

- the miner produces certified generations satisfying the frozen **22-array contract**;
- the finalizer's validation passes (`validate_array_bundle()`, frozen array order);
- the **Step-2 loader** consumes them with `fallback_used=False`;
- multi-rig runs are **internally consistent** — identical inputs produce identical
  artifacts across nodes;
- provenance is complete: commit, dataset identity, threshold requested/payload/effective.

**Precedent: this is already how Phase 6.0 was accepted.** The CUDA/ROCm parity result —
byte-identical `artifact_sha256` across three artifacts on two vendors' silicon — involved
**no PWC**. Acceptance was self-consistency plus the D6 certified generation, and Beta
approved it on that basis.

Consequences if adopted:
- **repair 3 becomes optional** (PWC's hybrid `0.50` stops being a Phase-6 blocker);
- Phase 6 reduces from a four-path equivalence exercise to multi-rig certification of the
  miner;
- PWC/ZMQ remain in the tree, flag-selectable, for exploratory use — **not** as an
  authority;
- the D6 release-grade generation remains the reference artifact, as it already is.

## 6. What Alpha is *not* claiming

- Not that PWC is worthless — it produced the artifacts the project ran on for a long time.
- Not that comparison is valueless — a **one-off, non-certifying** PWC/miner comparison may
  still be informative, provided a divergence is investigated rather than automatically
  attributed to the miner.
- Not that the threshold repair should be abandoned. Repairs 1 and 2 are **required
  regardless**; the optimizer is dropping sampled thresholds for every backend.

## 7. Rulings requested

1. **Was PWC ever intended as the authoritative comparator, or is that drift?** If a
   deliberate decision record exists, Alpha will withdraw this and proceed as ruled.
2. If it is drift: **retire PWC as the Phase 6 authority** and re-scope Phase 6 acceptance
   to the interface contract in §5.
3. **Confirm repair 3's status** — required (PWC retained as authority) or optional (PWC
   retired). Repairs 1 and 2 proceed either way.
4. If PWC is retained as authority: **state what audit standard it must meet first.** Alpha
   will not certify a comparator that has not been held to the standard applied to the
   component it is judging.
