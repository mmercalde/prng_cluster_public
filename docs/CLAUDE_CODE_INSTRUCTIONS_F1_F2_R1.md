# CLAUDE CODE INSTRUCTIONS — F1/F2 AMENDMENT, REVISION 1 (NARROW)

**Host:** VM101, repo `~/distributed_prng_analysis`. The F1/F2 amendment is **uncommitted in the
working tree** at base `eecfff7`. `source ~/venvs/torch/bin/activate` before every test.
Long suites: `python3 -u <suite> | tee /tmp/<name>.log` — never `| tail`.

**Authority:** Team Beta ruling *"F1/F2 ACTIVE-LEASE AMENDMENT REVIEW"* (2026-08-09) —
**architecture ACCEPTED, commit NOT authorized, two narrow production defects.**

**APPROVED AND CLOSED — do not redesign, do not re-argue:** the F1 active-lease architecture ·
coordinator-owned backlog · one compute-active claim per serial worker · leases stamped at real
handoff · heartbeat + active-progress renewal · frozen-cohort scheduling · pending-backlog
cancellation at abort · the F2 durable terminal architecture and its atomicity boundary ·
**hybrid deferred placement (RATIFIED — Alpha's §4 position was correct)** · no worker protocol
change · dispatcher unchanged.

**Hard constraints:** no commit, no push, **no pipeline launch, no fleet launch, no port 5700
bind.** Gate 12 HELD. **Do not apply `worker_pool_size = 25`.** Nothing on the §19 do-not-touch
list. If a fix appears to need scope beyond the four items below, STOP and report.

**Base verification:** amendment intact at `eecfff7`; `tests/test_s172_f1_f2_active_lease.py`
**13/13**. Untracked runtime residue expected, not a stop condition.

---

## A. BLOCKER A — a PREFIX selector is being used as an EXACT stripe selector

**Alpha verified this against live source; it is exactly as Beta describes.**

`schedule_pending_stripes(..., stripe_prefix=...)` is **prefix-scoped by construction**: the
docstring at `:2073` says *"scopes the query to one stage"*, and the SQL at `:2077-2081` is
`stripe_id LIKE <prefix>%`. But the hybrid failure path at **`:5492`** passes
**`stripe_prefix=stripe_id`** — a *complete* stripe ID — intending to schedule only the failed
retry stripe.

**Those are not equivalent.** With gate-12 IDs:

```
failed stripe   run__st0_s1
prefix query    run__st0_s1%
also matches    run__st0_s10 … run__st0_s19
```

**The consequence is worse than a wrong query.** If every legitimate alternate is compute-busy and
the prior claimer A is idle:

- `s1` — prior claimer is A, only free worker is A ⇒ correctly skipped;
- `s10` — an ordinary pending stripe, no prior claimer ⇒ **claim succeeds**.

`placed` is then non-empty, and `:5493-5495` reports `action="reassigned", worker_id=A` — **while
the failed stripe was never reassigned at all.** An unrelated sibling was assigned instead. That
violates the retry result contract and the deferred-placement semantics Beta just ratified.

**Why 13/13 missed it:** `G-F1-HYBRID-MATRIX` uses **two** stripes, so no `s1`/`s10` lexical
sibling can exist. It fires at the 32-stripe production geometry gate 12 will run.

### Required correction (Beta §4)

**Separate the two selector concepts.** Beta's preferred shape — an explicit API distinction:

```
stage_prefix = …        (LIKE-scoped, for normal stage scheduling)
exact_stripe_id = …     (identity, for hybrid immediate placement)
```

The hybrid immediate-placement path **must use exact identity**. **Do not infer from the shape of
the string** whether the caller meant a stage or a stripe.

Beta notes an alternative — drop the immediate targeted placement entirely and let every retry
return `requeued` for the next normal scheduler pass — as *"architecturally valid, but it changes
the action contract more broadly."* **Beta prefers the narrower correction; take that unless you
find a concrete reason it cannot work, in which case STOP and report.**

### Required gate — lexical collision at production shape

**Preferably all 32 stripes**, at minimum enough to contain both `s1` and `s10`. Construct:
failed retry stripe `s1` · prior worker **A idle** · **every alternate compute-busy**. Invoke the
immediate hybrid retry placement and assert:

```
s1 remains pending · s1 attempt == 1
s10 … s19 unchanged — NO sibling becomes claimed as a side effect
returned action == "requeued"   NOT "reassigned"
```

Then free a legitimate alternate and prove `s1 → claimed by that alternate`, **fresh lease
stamped**, and **A is not selected**. **A mutant restoring prefix-as-exact behaviour must red.**

## B. BLOCKER B — F2 terminal replay parity: first durable transition must own terminal identity

The first abort atomically persists terminal record **A** and logs from **A**. Correct.

On a later idempotent call `abort_trial(..., terminal=B)` the code correctly refuses to overwrite
the durable row and correctly suppresses a second ERROR log — **but the local `terminal` object is
still B, and the outward event is constructed from it.** Reachable state:

```
durable ledger  = A
ERROR log       = A
replayed sink event = B          ← violates F2's canonical-record claim
```

### Required correction (Beta §7)

> **The first durable terminal transition wins terminal identity permanently.**

When an abort call discovers the trial is **already aborted** — **or loses the atomic transition
race** to another abort — it must **reconstruct/use the existing durable terminal fields** before
constructing any externally visible abort event. The later caller's proposed class/reason is **no
longer authoritative**.

### Required gate — `G-F2-IDEMPOTENT-PARITY`

Abort with **A**; capture durable row, ERROR log and sink event. Abort the same run again with a
**deliberately contradictory B**. Assert:

```
durable terminal remains A
no second terminal ERROR log
any returned abort event uses A
any sink/recovery delivery from the second call uses A, or is suppressed per the
    existing durable delivery contract
B appears NOWHERE as terminal authority
```

**Also exercise the race-shaped case** where `mark_trial_aborted()` returns False because another
terminal transition won after the caller's initial read.

## C. GUARD SUPERSESSION — granted, but ORDER MATTERS

Beta **granted** Alpha's §3 request in principle: *"failure matrix unchanged"* means **terminal
decision semantics unchanged**, not that the function can never change a byte. The guards' historical
purpose — preventing accidental modification during unrelated staging work — **must survive; the
byte identity itself need not.**

**⛔ BETA'S EXPLICIT ORDER CONSTRAINT: "Do not simply update the old byte hash/baseline to the
current submitted source. The current source contains Blockers A and B. That would certify the
defects."** Fix A and B **first**, then supersede.

**The superseding invariant** — preserve, **in order**, the four terminal decisions:

```
non-retryable                                   → fail_trial / non_retryable
constant phase                                  → fail_trial / constant_phase
hybrid first failure + no alternate eligible    → fail_trial / no_alternate_worker
hybrid second failure                           → fail_trial / hybrid_second_failure
```

And **separately certify the hybrid first-failure nonterminal branch**:

```
alternate exists → exactly one retry budget consumed → trial stays running
                 → immediate reassignment if an eligible alternate is truly idle
                 → OR pending/requeued if alternate capacity is temporarily busy
```

Apply to `tests/test_s172_admission_liveness.py:808-811` and `G-MATRIX-DIFF-a`
(`tests/test_s172_staging_backpressure.py:1583-1589`). **Update `G-LEASE` to the ratified
deferred-placement semantics** — its current expectation of direct `"reassigned"` behaviour is the
old model.

**Beta §11: "Do not use the supersession ruling to erase unrelated assertions from those gates.
Only the assumptions directly invalidated by F1/F2 may change."** Preserve every unaffected
call-site and matrix assertion; state in the report exactly which assertions changed and why.

## D. WORDING CORRECTION — the one-active claim (Alpha's overstatement)

Alpha's submission said the read and write are in *"the same SQL statement."* **They are not.**
The actual shape is:

```
coordinator-process _write_lock
    → SELECT for an existing active claim
    → if none: UPDATE the requested stripe   (terminal-state guard embedded here)
```

The correct claim, per Beta §10:

> **Within one coordinator process, the ledger write lock serializes the existing-active check and
> the subsequent claim update.**

**Do not claim protection against an independent external writer or a second coordinator process.**
This is a wording/certification-boundary correction, **not** a new production blocker — and it is
consistent with the S172 boundary already on record (one active trial per coordinator process).
Correct it in the source comment and in the report. Multi-coordinator writers would need
database-level enforcement; **that is outside F1 — do not implement it.**

---

## VERIFICATION (Beta §13) — run sequentially

```
F1/F2 active-lease suite   (incl. the exact-stripe lexical collision + G-F2-IDEMPOTENT-PARITY)
S172 staging backpressure · S172 Part B · S172 elapsed roundtrip
D3.5 finalizer · phase-4 coordinator · S172 admission liveness
```

**Expected before commit authorization: all F1/F2-chargeable reds GREEN.** Gate 22 may be evaluated
on a staged/commit-equivalent tree so the new test file is not merely untracked.
`admission_binding` C5 **fails at baseline and is not charged to this amendment** — rerun it as a
baseline differential if you keep it in the package, but do not absorb it. **No fleet execution.**

## REPORT

`docs/CLAUDE_CODE_REPORT_F1_F2_R1.md`:

1. The exact-vs-prefix API split as built, and every call site of each, with `file:line`.
2. The lexical-collision gate's stripe count and the mutant's observed behaviour.
3. How an already-aborted / race-lost abort reconstructs the durable terminal record, and proof B
   cannot reach any outward surface.
4. **Which assertions changed in the two superseded guards and in `G-LEASE`, and which were
   preserved** — enumerated, not summarised.
5. Confirmation that A and B were fixed **before** any guard was re-baselined.
6. The corrected one-active wording, in source and report.
7. Red-first and mutation evidence per new gate.
8. Full sequential verification results.
9. Files changed. **Any disagreement reported, not worked around.**
