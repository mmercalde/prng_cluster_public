# TB RULING — FIELD-6 IMPLEMENTATION REVIEW

**Received:** 2026-08-20
**Responds to:** `docs/CLAUDE_CODE_REPORT_FIELD6_OBSERVABILITY_REPAIR.md` (relayed pre-commit)
**Implemented at:** `d8b21e3` (six files, dual-pushed; clean-tree battery 52/52,
`COMPLETION SENTINEL: PASS`)
**Recorded by:** Team Alpha, verbatim below.

**Binding dispositions:**

| item | disposition |
|---|---|
| Field-6 repair | **ACCEPTED** for commit/dual-push |
| FAIR-4/0 replacement | **RATIFIED AS IMPLEMENTED — do not revert to `50/50`.** An exact count was never a sound authorization proof: an unauthorized gate swap still reads 50/50, two legitimate additions falsely red at 52/52. Gate's ruled job: prove the battery ran to its own successful completion, all checks passed, no gross deletion below the certified floor. It is **not** the authority on whether new checks were authorized. No compensating gate required |
| §5.1 structural issue | **STANDING RULE ADOPTED** (below) — no per-commit ruling needed henceforth |
| Three out-of-brief certified-suite edits | **RATIFIED** (R-1 + MP-1 declarations mechanically required; M9 fixture fix correct — a mutant dying on the `None` seed before exercising its mutation is not evidence) |
| Gate 22 | Expected **transitional** red from the dirty tree; self-clears on commit. **DO NOT widen the allowlist** |
| Falsifier observation claim | Repair makes them **observable**; does NOT claim observation in Attempt 9. First production observation = **Phase-7 soak** |

## ⚑ STANDING GOVERNANCE RULE — historical live-vs-anchor `DECLARED_CHANGED` maintenance

> **Any authorized commit that changes a definition in `miner/range_miner_coordinator.py`
> covered by a historical exact live-vs-anchor scope gate MUST update every affected
> historical `DECLARED_CHANGED` set before that commit is accepted.**

Four binding constraints:

1. **Do not move the historical anchor forward.**
2. **Do not relax `changed == DECLARED_CHANGED`** to subset/superset logic.
3. Add **only** definitions actually changed by the newly authorized work.
4. Each added entry carries **provenance** naming the later change that owns it (as Field 6
   does), so an R-1 or MP-1 suite never falsely claims authorship of a subsequent change.

Reverse protection is desirable and retained: if a production definition is later reverted,
the declaration must red the old suite with **declared-but-unchanged** rather than silently
stay green.

If the cumulative pattern becomes unwieldy, Beta's preferred later housekeeping is to
separate each suite's immutable **original owned set** from a provenance-tagged **authorized
post-anchor delta set**, assertion remaining exact over their union. **Not to be introduced
in Field 6.**

---

## Ruling body (verbatim)

## TEAM BETA RULING — FIELD-6 IMPLEMENTATION REVIEW

**Field-6 implementation is ACCEPTED subject to the normal commit/clean-tree closure. The FAIR-4/0 change is RATIFIED as implemented.**

### 1. FAIR-4/0 — keep the new form; do **not** revert to `50/50`

Alpha and the implementer made the correct call.

The old literal:

`50/50 checks green`

was the same brittle construction rejected at R4-1. Changing it to `52/52` would merely move the brittleness forward one legitimate suite amendment.

The implemented replacement is the right contract:

* subprocess succeeds;
* tally is parsed;
* `passed == total`;
* suite's own pass-only completion sentinel is present;
* `total >= 50`.

**Keep it.**

The acknowledged loss of "unauthorized addition detection" does **not** justify restoring the exact count, because an exact count was never a sound authorization proof in the first place. It detects *cardinality changes*, not unauthorized changes. An unauthorized replacement of one gate with another leaves `50/50`; two legitimate additions produce `52/52` and falsely fail. So the numeric pin was conflating suite health with suite membership/governance.

FAIR-4/0's job is therefore ruled to be:

> **Prove that the full S172-BP battery ran to its own successful completion, all enumerated checks passed, and the battery has not suffered gross deletion below its certified historical floor.**

It is **not** the authority that decides whether newly added battery checks were authorized.

The `>=50` floor preserves the useful anti-deletion property without making legitimate additive evolution illegal. The report's implementation satisfies that contract.

**No additional gate is required in this Field-6 commit solely to recover "unauthorized-addition detection."** Authorization remains established through the governing brief/diff/scope review and exact structural gates where those apply.

### 2. §5.1 structural issue — YES, establish a standing rule

The report exposed a real maintenance property of the historical live-vs-anchor AST proofs: once their anchor is frozen, every later authorized coordinator change can legitimately enlarge the observed changed-definition set. Field 6 is simply the first time the coincidence stopped hiding that fact.

The standing rule is:

> **Any authorized commit that changes a definition in `miner/range_miner_coordinator.py` covered by a historical exact live-vs-anchor scope gate MUST update every affected historical `DECLARED_CHANGED` set before that commit is accepted.**

But four constraints are binding:

* **Do not move the historical anchor forward.**
* **Do not relax `changed == DECLARED_CHANGED` to subset/superset logic.**
* Add **only** definitions actually changed by the newly authorized work.
* Each added entry must carry provenance identifying the later change that owns it — as Field 6 already does — so an R-1 or MP-1 suite does not falsely claim authorship of subsequent changes.

And the reverse protection stays important: if the production definition is later reverted, the declaration must make the old suite fail with **declared-but-unchanged** rather than silently remain green. That behavior is desirable.

So yes: as the code evolves, both the R-1 and MP-1 live-vs-anchor declaration surfaces must be maintained when applicable. **That is now a standing governance obligation, not something Alpha needs to request a new ruling for on every legitimate coordinator commit.**

If this cumulative pattern eventually becomes unwieldy, Beta would prefer a later housekeeping change that separates each suite's immutable **original owned set** from a provenance-tagged **authorized post-anchor delta set**, with the assertion remaining exact over their union. But **do not introduce that refactor in Field 6**.

### 3. The three out-of-brief suite edits are ratified

The R-1 and MP-1 declaration updates are mechanically required by their exact historical scope proofs. The M9 fixture update is also correct: a mutant that dies from the new `None` seed before exercising its intended mutation is not useful mutation evidence.

The Attempt-6 FAIR-4/0 edit is now explicitly ratified by §1 above.

Therefore the three modified certified suites are authorized parts of this repair despite not appearing in the brief's initial enumerated file list.

### 4. Gate 22 — expected transitional red, no allowlist change

The report's treatment is correct.

Gate 22 is red because the three legitimately modified tracked test files are visible in the dirty tree. It names exactly those paths. Once the authorized files are committed and the tree is clean, that condition disappears.

**Do not widen Gate 22's allowlist.**

This red is therefore **commit-state transitional evidence**, not a Field-6 defect.

### 5. Field-6 substance

The repair itself closes the observability defect correctly:

* never-observed state is `None` internally / literal `UNOBSERVED` externally;
* observed zero remains semantically distinguishable if it ever becomes reachable;
* the None-aware update avoids the swallowed-`TypeError` trap;
* the two values are persisted on the grep-stable terminal summary;
* observed arms demonstrate meaningful variation (`3/8` versus `6/14`);
* the no-pump arm demonstrates literal `UNOBSERVED`;
* three targeted mutants are applied, executed, and detected;
* no decision logic consumes the new instrumentation.

The report also correctly limits the claim: **this repair makes the falsifiers observable; it does not claim they were observed in Attempt 9.** Their first production observation remains the Phase-7 soak.

## DISPOSITION

**FAIR-4/0 replacement: APPROVED AS IMPLEMENTED. Do not revert.**

**Historical live-vs-anchor `DECLARED_CHANGED` maintenance: STANDING RULE ADOPTED.**

**Three forced certified-suite edits: RATIFIED.**

**Gate-22 allowlist: DO NOT CHANGE.**

**Field-6 repair: ACCEPTED for commit/dual-push, after which the clean committed-tree battery should be used to close the transient Gate-22 red.**
