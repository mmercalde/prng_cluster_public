# CORRECTIONS — `docs/TEAM_ALPHA_FLEET_STATE_SUBMISSION.md`

**Required by Team Beta's fleet-state ruling, "Two audit corrections."** Both are **wording
corrections, not reversals** — Beta confirmed the findings and accepted the audit. Alpha
overstated two supporting statements and they must not stand in a committed document.

**Documentation only. Two edits, both in the submission.** The audit
(`FLEET_STATE_REQUIREMENTS_v1.md`) does not repeat either error — §5.1's table is correct as
written, and §4.1's heading is scoped to *"no stripes are lost"*, which remains true. Only the
submission's prose needs changing.

---

## EDIT 1 — the "five of six" claim was false by Alpha's own table

**Beta:**

> *"Five of six point at bare metal" is false by the audit's own table. **Three** mechanisms
> explicitly use `.120/.154/.162`: legacy connectivity, PWC, and WATCHER GPU health. P0.5 uses
> CT100; boot-notify is host-local; miner registration accepts whoever connects.*

Alpha counted "not-CT100" as "bare metal" and inflated the number. Two of the six do not point
at a fixed address set at all.

**In §3, replace:**

```
**Five of six point at bare metal; P0.5 points at the CT100s.** The rigs are currently booted
```

**With:**

```
**Three of the six explicitly point at bare metal** — legacy connectivity, PWC readiness and
WATCHER GPU health, all reading `.120/.154/.162`. **P0.5 points at the CT100s.** The remaining
two do not name a fixed address set at all: boot-notify is host-local, and miner registration
accepts **whoever connects** — which is its own problem, and the one Beta's Resolved Execution
Set closes (*"unknown miner workers must not become eligible merely because they connected"*).

The rigs are currently booted
```

*(Correction per Beta's fleet-state ruling. Alpha counted "not-CT100" as "bare metal"; two of
the six have no fixed address set.)*

---

## EDIT 2 — the capacity claim needs its precondition

**Beta:**

> *"A GPU that never registers costs capacity, not correctness" needs a condition. That is true
> only if the remaining eligible population satisfies the current admission threshold. **Below
> it, assignment never occurs and the present behavior is the pre-assignment hang.***

Alpha stated the good case without the condition that makes it good. As written it reads as an
unconditional reassurance, and it is not one — the same §4.3 hang applies.

**In §2, replace the opening sentence:**

```
A GPU that **never registers** costs capacity, not correctness. `assign_stripes`
```

**With:**

```
A GPU that **never registers** costs capacity, not correctness — **but only while the remaining
eligible population still satisfies the admission threshold.** Below `expected_workers`,
assignment never occurs at all and the behaviour is §4.3's pre-assignment hang, not a smaller
successful run. The claim below is conditional on the threshold being met.

`assign_stripes`
```

**And in the same section, replace the closing line:**

```
So the silent case is **capacity**, and the dangerous case is **a threshold crossing.**
```

**With:**

```
So **above the threshold** the silent case is capacity; **at or below it**, in either direction —
never reaching `expected_workers`, or falling under it mid-run — the case is §4.3's hang. The
threshold is the whole boundary between a degraded-but-correct run and a silent one.
```

---

## Not changed

- **No finding is reversed.** Beta accepted the audit and confirmed §4.3 as a Phase 7 blocker.
- `FLEET_STATE_REQUIREMENTS_v1.md` — §5.1's table is correct as written (it is the table Beta
  used to catch Alpha's miscount), and §4.1's *"no stripes are lost"* remains true within its
  stated scope. **Do not edit the audit.**
- No code, config, test or manifest.

## Why this is being done as a documented correction rather than a silent edit

Both statements are in a document Beta has already read and cited. Quietly changing them would
leave the record showing Beta correcting a claim the repository no longer contains. The edits
carry an inline note naming the correction — same treatment as the retracted seed-set claim in
D6.1, and consistent with `G-COMMENT-TRUTH`'s requirement that a retracted claim not simply
disappear.
