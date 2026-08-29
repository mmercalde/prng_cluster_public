# TB RULING — RUN-4 ROUTE-A PATCH REVIEW R2

**Received:** 2026-08-28 · **Recorded by:** Team Alpha (verbatim; delivered via ser8 → VM101) · **Status:** BINDING
**Reviews:** `agents/watcher_agent.py` sha256 `d14bcb3b3395877017e33e15fe00fb31b775945bdb1cf3d6bce8bd556a208e5f` (+374/−13), brief revision 3, R2/R2a recorded suite (20/20)

## Dispositions

| Item | Disposition |
|---|---|
| **Patch R2 (`d14bcb3b…`)** | **APPROVED FOR COMMIT** — R1 blockers closed; supersedes rejected `0398c0d1…` |
| Operator-origin authority (keyword-only `_operator_pin_params`, CLI sole populator, fail-loud on unauthorized keys) | **CLOSED** — G-ORIGIN + M4 adequate evidence, correct asymmetry present |
| MOVE vs DUPLICATE | **MOVE RATIFIED** — stronger construction; seven in exactly one authority-bearing location; retry copy can no longer become a second source |
| **NEW STANDING CONSTRAINT** | `_operator_pin_params` and internal `_pin_bundle` are **privileged internal seams**; no new production caller may populate either without explicit governance review. No code required now |
| Present-vs-usable validation | **CLOSED** — presence distinguished from routability; fail-loud, no silent collapse |
| True/False rejection | **RATIFIED** — not unauthorized expansion; same defect class via the builder's bool branch; all seven are numeric, so booleans would defeat the guard. 30-case G-VALUE-USABLE accepted |
| Invocation lifetime | **CLOSED** — two-invocation/same-agent proof is the important certificate (7/7 then 0/7, no provenance) |
| Unpinned behavior | **CLOSED** — pre-edit `69ca910` oracle list-equal at 47 tokens; EXEC-PIN-1 compliant; key evidence Route A opened a capability without creating a default |
| Provenance | **ACCEPTED** — authority evidence vs routing evidence separation correct; M1b is the proof nobody may treat the marker alone as proof the seven reached the optimizer. Sorted construction accepted |
| Mutation package (M1, M1b, M2a, M2b, M3, M4) | **ACCEPTED** — critical independent defect classes covered; no detection credited on an exception; three-point M1b digest proof sufficient |
| Regression state (20/20 · 26/26 · 14/1/0 · 13/13) | **ACCEPTED** — Pydantic warnings environmental, not chargeable |
| Recorded-but-unrepaired items (six-of-seven check, `_ws_source` mislabel, stale root `watcher_agent.py`, L-1, ingress byte bound, step_runner surface) | **UNCHANGED, none blocks commit** — WATCHER provenance remains authoritative over `_ws_source` |
| Minor housekeeping note (non-blocking) | Unusable-value rejection shares `Step1PinBundleError` with partial-pin failures; acceptable while durable error text stays specific; do NOT expand this change to rename classification fields |
| **Post-commit sequence** | (1) confirm clean tracked tree + committed production digest; (2) rerun commit-sensitive clean-tree/Gate-22 check WITHOUT widening allowlists; (3) verify Run-4 infrastructure hardening — actual GPU power caps, off-host power telemetry, persistent netconsole; (4) pre-launch Route-A provenance check: seven intended values enter through the CLI operator seam, generated Step-1 command contains exactly the approved pinned geometry; (5) Run 4 remains Michael-authorized separately |
| Claim boundary | **UNCHANGED** — this patch certifies routing and containment only; Run 4 is the production proof for volume/plumbing/publication/completion; neither may claim population equivalence |

---

## RULING BODY (VERBATIM)

## TEAM BETA PATCH REVIEW — RUN-4 ROUTE-A R2

**VERDICT: APPROVED FOR COMMIT. R1 BLOCKERS CLOSED.**

The reviewed production target is the R2/R2a `agents/watcher_agent.py` with SHA-256 **`d14bcb3b3395877017e33e15fe00fb31b775945bdb1cf3d6bce8bd556a208e5f`**. The previous rejected `0398c0d1…` surface is superseded. The recorded suite is **20/20 PASS**, with all six mutants applied/executed/detected, and the required regressions remain green.  

### 1. Operator-origin authority — CLOSED

The R1 defect is corrected properly.

Authority is no longer inferred from ordinary `run_pipeline(params=...)`. The patch gives it a separate keyword-only `_operator_pin_params` channel, default `None`, and the actual CLI `--run-pipeline --params` seam is the sole production populator. Ordinary programmatic parameters containing the seven keys fail loud before Step 1 and receive no operator provenance. 

**MOVE rather than DUPLICATE is ratified.**

That is the stronger construction because it leaves the seven in exactly one authority-bearing location. In particular, the normal retry copy of ordinary `params` can no longer become an accidental second source of the warm-start values. The frozen invocation-local bundle remains the sole source for legitimate replay inside that same authorized pipeline invocation. 

`G-ORIGIN` and M4 are adequate evidence. The important asymmetry is present: reverting origin back to ordinary params breaks G-ORIGIN while the normal pinned routing and unpinned control can remain green. 

I adopt one standing constraint from this implementation:

> `_operator_pin_params` and the internal `_pin_bundle` are privileged internal seams. No new production caller may populate either without an explicit governance review.

No additional code is required for that now.

### 2. Present-vs-usable validation — CLOSED

The second R1 blocker is corrected.

The code now distinguishes **presence** from **routability**, so a seven-key request cannot masquerade as complete while the command builder subsequently suppresses one or more values. The recorded gate covers the requested `None` and `''` cases and demonstrates fail-loud behavior rather than silent collapse to an unpinned run. 

The additional rejection of `True` and `False` is **RATIFIED**.

That is not unauthorized semantic expansion. Those values hit the command builder's bool-special branch and are the same defect class:

* `False` results in no flag;
* `True` produces a valueless numeric option and therefore a malformed command.

Since all seven warm-start arguments require numeric values, accepting Python/JSON booleans here would defeat the exact property this guard exists to enforce. 

The 30-case `G-VALUE-USABLE` battery is accepted.

### 3. Invocation lifetime — CLOSED

The authority bundle remains local to one `run_pipeline` invocation, is not stored on the `WatcherAgent`, and does not enter daemon state.

The two-invocation/same-agent proof remains the important certificate: first invocation routes 7/7; the second unpinned invocation routes 0/7 and carries no pin provenance. 

That closes the lifetime concern from the brief review.

### 4. Unpinned behavior — CLOSED

The pre-edit `69ca910` oracle remains the correct control.

The final R2 implementation still builds the unpinned command **list-equal at 47 tokens** to the artifact captured from clean `69ca910`. This is much stronger than executing historical source against current helpers and therefore remains compliant with EXEC-PIN-1. 

This is the key evidence that Route A has opened an explicit capability without converting it into a new default behavior.

### 5. Provenance — ACCEPTED

The separation between **authority evidence** and **routing evidence** remains correct:

* `step1_pin_source` says the invocation possessed legitimate explicit-operator authority.
* `step1_pin_argv` records what command was actually built.

M1b gives unusually good evidence for that distinction: under the surgical WALL-2 mutant, provenance remains green while routing goes red. So nobody may later treat the marker alone as proof that the seven arguments actually reached the optimizer. 

Sorted pin construction is also accepted. Stable argv ordering improves the evidentiary record without changing semantics.

### 6. Mutation package — ACCEPTED

The six-mutant roster now covers the critical independent defect classes:

* M1 — broad unconditional stripping;
* M1b — surgical WALL-2 regression;
* M2a — invocation-lifetime leakage;
* M2b — default contamination;
* M3 — over-broad manifest-derived authorization;
* M4 — authority inferred from ordinary params.

All are reported as **APPLIED / EXECUTED / DETECTED**, with the expected still-green controls, and no detection credited merely because an exception occurred. 

The three-point M1b source digest/restoration proof is also sufficient.

### 7. Regression state — ACCEPTED

The relevant regression evidence is clean:

* Route-A routing: **20/20**
* Brief-I: **26/26**
* Brief-I mutants: **14 detected / 1 previously accepted INVALID-by-scope / 0 survived**
* WATCHER LLM integration: **13/13**

The Pydantic warnings are environmental/pre-existing and not chargeable to this patch. 

### 8. Recorded-but-unrepaired items remain outside this patch

No change in disposition:

* downstream six-of-seven optimizer completeness check remains under WATCHER-MANIFEST-ROUTING-1;
* `_ws_source` still mislabels explicit operator pins as trial-history warm starts, so **WATCHER provenance remains authoritative**;
* stale root-level `watcher_agent.py` remains a wrong-file-edit hazard/backlog item;
* L-1 float32/float64 seam remains open;
* coordinator ingress byte-bound issue remains open;
* `step_runner`'s separate dispatch surface is not part of the certified Run-4 route.

None blocks this commit.

One minor observability note for later housekeeping: a full seven-key bundle rejected for unusable values currently shares the `Step1PinBundleError` family with partial-pin failures. As long as the durable error text remains specific, that does not block this patch; don't expand this change merely to rename classification fields.

## COMMIT DISPOSITION

**R2 is APPROVED.**

Michael may commit and dual-push the **exact reviewed R2/R2a patch and associated acceptance artifacts**.

After commit:

1. confirm clean tracked tree and committed production digest;
2. rerun any commit-sensitive clean-tree/Gate-22 check without widening allowlists;
3. verify the Run-4 infrastructure hardening state—actual GPU power caps, off-host power telemetry, and persistent netconsole;
4. perform a pre-launch Route-A provenance check showing the seven intended values enter through the CLI operator seam and the generated Step-1 command contains exactly the approved pinned geometry;
5. then Run 4 remains Michael-authorized separately.

**The approved Run-4 claim boundary is unchanged:** this patch certifies **routing and containment only**. Run 4 is the production proof for volume, plumbing, publication, and completion; neither the patch nor Run 4 can be used to claim population equivalence. 
