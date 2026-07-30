# PHASE7_PREREQUISITES.md

**What stands between us and the Phase 7 soak?**

Durable answer to a question that keeps getting re-derived across sessions. Two
work items block Phase 7. This file names them, says what each one blocks and
why, and points at the full brief. It does not restate the briefs.

**Status:** current as of 2026-07-30, at D6.1 (implemented, gates green,
awaiting commit).
**Scope:** the *blockers*. The soak's own design lives in
`docs/SOAK_TEST_PLAN_PHASE7_v1_0.md`; WATCHER integration work lives in
`docs/TODO_PHASE7_WATCHER_INTEGRATION_REVISED_v3.md`. Neither is superseded by
this file.

---

## Approved sequence

```
D6.1  →  Phase 6.0  →  bounded Phase 6  →  D6.2  →  D6.3  →  Phase 7
 ↑          ↑              ↑                ↑        ↑         ↑
done     paired        see the         unblocks   makes    the soak
(gates   CUDA/ROCm     bounded-        the soak   the soak
 green)  smoke         Phase-6                    survivable
                       condition                  in duration
                       below
```

Phase 6 may begin before D6.2 **only** under the bounded condition in §3.

---

## 1. D6.2 — canonical 24-field checkpoint, reconciliation, finalizer resume

**Blocks Phase 7 because:** the S166 in-memory clear stays **disabled** until
D6.2 lands (`_FLUSH_CLEAR_IN_MEMORY = False`), so the candidate list grows
without bound for the whole lifetime of a run — and a soak is precisely a
long-lived, many-trial run. Unbounded candidate-list RAM growth is the failure
mode.

**Why the clear cannot simply be switched on:** D6.1's snapshot carries **four**
fields; the D3.5 finalizer consumes the in-memory list and requires all **24**
`CANONICAL_RECORD_FIELDS`. Clearing today would truncate the certified
generation's raw-candidate input, and 20 of the 24 fields are unrecoverable
from the snapshot. Enabling the clear therefore requires a 24-field checkpoint
**and** a finalizer read-back path — that is D6.2, not a flag flip.

**Reconciliation authority:** winner selection on resume must go through the
**frozen** `_l2_sort_key` (`utils/run_finalizer.py:690`) and
`_select_l2_winners` (`utils/run_finalizer.py:714`). D6.2 reconciles against
that existing authority; it does not introduce a second selection rule.
D6.1's merge-by-seed is provisional snapshot maintenance only and decides
nothing.

**Full brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_S172_D6_2_CHECKPOINT_RECONCILIATION.md`
(REV1, DRAFT — pending Beta approval).

---

## 2. D6.3 — checkpoint retention, lifecycle and garbage collection

**Blocks Phase 7 because:** `.s172_checkpoint/<run_id>/` is **never pruned**.
Each process creates one directory and nothing removes any of them, so the
directory count grows without bound across a soak or a WATCHER-driven series.
Run isolation is implemented and correct (it is what prevents collisions); what
is missing is anything that ever cleans up.

**Beta's binding constraint on the design:** retention must **never remove
active, unresolved, or audit-retained state merely for exceeding an age or
count threshold.** A naive "delete anything older than N days" or "keep the
last N runs" policy is explicitly ruled out — liveness and resolution status
govern, not age or count.

**Full brief:** not yet written. To be created as
`docs/CLAUDE_CODE_INSTRUCTIONS_S172_D6_3_CHECKPOINT_RETENTION.md`.

---

## 3. Beta's bounded-Phase-6 condition

Phase 6 may run **before** D6.2 only if **every** scenario satisfies all of the
following:

**Required:**
- a **fresh process** or an **explicitly reset accumulator** per scenario;
- a **declared maximum** seed/survivor volume;
- **host RSS monitored** during the run.

**Forbidden — no scenario may:**
- rely on list clearing in any way;
- make a **resume/restart acceptance claim**;
- use a **WATCHER long-lived loop**;
- exhibit **multi-trial soak behaviour**.

**Escalation:** if Phase 6 becomes long-lived with unbounded list growth,
**D6.2 moves in front of it** and Phase 6 stops until D6.2 lands.

---

## 4. What D6.1 did and did not settle

Recorded here so the boundary is not re-litigated. Detail in
`docs/SESSION_CHANGELOG_20260729_PHASE5_D6_1.md`.

**Settled by D6.1** — the incremental flush changed from an always-failing
attempt into a real provisional snapshot write, with per-file atomic
replacement, fsync durability, transaction identity (so an interrupted
replacement is detectable when seed sets match), tiered visible failures, and
isolation from the finalizer-owned aliases.

**NOT settled by D6.1, and not claimed:**
- full accumulator resume → **D6.2**
- finalizer reconstruction → **D6.2**
- S166 in-memory memory protection → **D6.2**
- checkpoint retention / GC → **D6.3**

The snapshot is **non-authoritative** until D6.2. The in-memory list remains
the finalizer's authoritative source.
