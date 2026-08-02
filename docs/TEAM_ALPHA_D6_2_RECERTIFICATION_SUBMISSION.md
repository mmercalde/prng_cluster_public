# TEAM ALPHA → TEAM BETA — D6.2 resubmitted for certification

**Bounded repair committed and dual-pushed at `18a2419`**, on top of `f7583bc`. **Nothing was
reverted. Nothing Beta ratified was touched in logic.** 6 files, 1,374 insertions, 250 deletions.
Nothing committed from the sandbox; WATCHER not run.

**31/31 gates · 377 assertions · 25/25 mutants · PASS.** Phase 4 restored to **63/63** at the
commit. Full evidence: `docs/S172_D6_2_BOUNDED_REPAIR_REPORT.md`.

---

## 0. How Alpha reviewed this, and why that changed

**Beta found blocker 1 by reading the live objective. Alpha had read the report's description of
it.** Alpha had verified at source that `trial_counter` and `trial.number` are different counters,
ratified the ordinal continuation that follows — **and never asked whether comparing the two still
made sense.** The finding and its consequence were one question apart.

**This submission was reviewed from the diff, not the report.** Every claim below was checked in
changed source. Where Alpha verified a property, it says so; where it is relying on the harness, it
says that instead.

---

## 1. Blocker 1 — the guard that rejected every normal resume

**Repaired.** `int(trial.number) <= int(_resume_trial_floor)` is **gone** from the objective, and
the pre-flight nonterminal scan with it.

- **Renamed to `resume_record_ordinal_floor` across 10 sites.** *The old name asserted a
  relationship to Optuna trial numbers that does not exist, and that false name is what made the
  comparison look reasonable — to Claude Code, to Beta's addendum, and to Alpha.*
- **Used only to initialize the persisted record counter**
  (`window_optimizer_integration_final.py:2623`, unchanged and already correct). The floor now has
  exactly one consumer. What crosses the attribute seam is a **boolean**, not an ordinal.
- **Both Optuna-number guards and the queued-trial mutant retired.**
- **An AST sweep of the whole tree asserts no executable `trial.number`-vs-floor comparison and no
  live `_resume_trial_floor` reference survives.** Alpha confirmed the removal in the diff
  independently of that sweep.

**The loaded-study wall (§1.3.5).** `create_study(..., load_if_exists=_resume)` creates a fresh
study when the named one is absent, so a checkpoint resume could silently become a fresh study and
restart the record ordinal against a recovered checkpoint. The wall keys on `_resume`, **which is
set only after `optuna.load_study(...)` returned a study carrying completed trials** — positive
evidence the study existed, **not an inference from the request.**

**`G-RESUME-INTEGRATED`** replaces the vacuous `G-TRIAL-NAMESPACE`. It derives **k=3 from a study
that actually ran**, and shows **Optuna trial 3 executing** — the exact case the old guard rejected
(`3 <= 3`) — with its record taking **ordinal 4**, and the collision arm proven non-vacuous.
**`G-MISSING-STUDY`** proves rejection before the first objective executes.
**M23** reinstates the exact comparison and must red.

## 2. Blocker 2 — NP2 executing before D6.2 validation

**Repaired by scope, as Beta directed.** `resume_checkpoint` + `n_parallel > 1` is rejected at
**`:1979`, the first executable statement of `optimize_window`** — above the NP2 block (2032), the
shared study creation (2309), the `[NP2-KILL]` SSH to every rig (2404) and the fork (2439/2463).
**Above the SSH, not merely above the fork.**

**`G-NP2-SCOPE` is CPU-only and proves zero starts three ways:** sentinels that trip the instant a
forbidden call is attempted (0 events) · **real child-PID counting from `/proc`** (0→0) · a clean
control confirming `n_parallel == 1` does not hit the rejection, and that the guard is single.

**The position arm derives all three hazard locations live rather than hardcoding line numbers**,
so a refactor that moves the NP2 block cannot leave a stale green gate. Alpha verified this in the
diff — it is the property that makes the gate durable rather than a snapshot.

**The scope statement, in both module headers and the error itself:** D6.2 checkpoint recovery and
the S166 clear are certified **only for `n_parallel == 1`** — and **that path still distributes each
sieve trial across the full GPU cluster.** The limit is on Optuna parallelism, not fleet use.
**No claim is made for `n_parallel > 1`:** not resume, not accumulator clearing. Concurrent
partition writers cannot safely share the present member pair; that needs a separate transaction
design.

**Alpha confirms the Phase-7 soak must pin `n_parallel=1`** until that separate work closes.

## 3. Textual correction

`utils/checkpoint_d6_2.py` — the stale `seeds` wording for member A corrected to `seed`, matching
`MEMBER_A_PAYLOAD_FIELDS` and the ratified record domain. **Alpha verified the module's changes are
text only: no logic in the ratified checkpoint core was touched.**

## 4. Non-regression

**22 suites run, 17 fully green.** Phase 4 **62/63 pre-commit — Gate 22 only**, naming exactly the
two then-uncommitted files; D1.0/D1.1/D2/D5 red **only** in their nested arms that shell out to
Phase 4. **Resolved by committing: Phase 4 is 63/63 at `18a2419`.** Gate 22 was not edited.

**No Wall A/B rerun** (Beta).

**Two disclosures Alpha judges correct and wants on the record:**
- The new gates drive the real study body, which stores studies under `optuna_studies/`. An early
  iteration **leaked three databases into the repo**; they were removed, and the gates now run in a
  scratch cwd and delete every database they create, **verified by directory diff.** Disclosed
  rather than quietly cleaned.
- **"Before" counts at `f7583bc` were not re-measured.** Exporting the tree to a non-git directory
  breaks `_repository_state()` and would contaminate four unrelated gates, producing four
  wrong-reason failures. `f7583bc`'s own recorded result is cited instead, and the limitation is
  stated. **Alpha prefers a declared gap to a manufactured number.**

## 5. Ruling requested

1. **Certify D6.2 at `18a2419`.** Both blockers closed; the four items Beta ratified at `f7583bc`
   are carried forward unchanged in logic.
2. **Confirm the `n_parallel == 1` scope statement** as worded — Alpha has tried to make it honest
   in both directions and would rather have it corrected than flattering.

## 6. VIR declaration

**Execution proof:** `G-RESUME-INTEGRATED` reports the derived `k`, the Optuna numbers used and the
ordinals produced — not a boolean; `G-NP2-SCOPE` reports sentinel events and child counts.
**Clean control:** a normal fresh run passes unchanged; **a normal `n_parallel == 1` resume now
succeeds, which it did not at `f7583bc`.** **Fault injection:** 25 mutants, four-part kill rule;
**M24 was credited behaviourally** and is declared as such. **Sentinel:** `PASS`.
**Unavailable-observer:** every gate here is CPU-only with no fleet dependency; **no arm is
`UNAVAILABLE`.** **Audit scope:** repo-scoped at `18a2419`; **Alpha's review was conducted against
the diff of all five changed production and test files**, not the report alone.
**Unavailable surfaces:** host state on VM101 and the rigs; deployed uncommitted files; the live
`KERNEL_REGISTRY` if changed.
