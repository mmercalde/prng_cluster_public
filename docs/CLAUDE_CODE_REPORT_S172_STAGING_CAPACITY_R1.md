# CLAUDE CODE REPORT — S172 STAGING-CAPACITY AMENDMENT, REVISION 1

**Host:** VM101 (`zeus-ubuntu-vm`, `192.168.3.177`) · repo `~/distributed_prng_analysis`
**Base:** `c7058d8`, amendment + R1 revisions **uncommitted in the working tree**
**Status:** all five revisions implemented. **Not committed, not pushed, no pipeline/fleet launch,
no port 5700 bind.** Gate 12 and the Phase-7 soak remain HELD.

**Authority:** Beta *"S172 STAGING-CAPACITY AMENDMENT + `elapsed_s`"* (2026-08-08) — RETURN FOR
NARROW REVISION. Only the five ruled revisions were made; nothing in the approved-and-closed list
was revisited, re-argued or "improved".

---

## 0. Base verification

| check | required | actual | |
|---|---|---|---|
| `git log --oneline -1` | `c7058d8` | `c7058d8` | ✅ |
| tracked drift | the six amendment files | exactly those six, diffstat identical to submission (1262 +/80 −) | ✅ |
| `test_s172_staging_backpressure.py` | 42/42 | **42/42**, exit 0, sentinel PASS | ✅ |
| `test_s172_elapsed_roundtrip.py` | 6/6 | **6/6**, exit 0, sentinel PASS | ✅ |

Untracked residue (WAL sidecars, the `*.stale_*` rotation, delivered briefs, the new test file) was
present as expected and is not a stop condition. **No unexpected tracked drift.**

---

## 1. REVISION 1 — commit cleanup is crash-resumable (Beta §2, BLOCKER A)

**Beta is right and my submitted claim was overstated.** I wrote that reuse of `ack_by_event_id`
made cleanup safe across "a crash between rows". Idempotency is necessary but not sufficient when
**the recovery path never calls it** — and the submitted `commit_trial` returned on the
`delivery == done` branch *before* the sweep, so reservations 2..N were stranded permanently.

### The two-phase rule as implemented

`miner/range_miner_coordinator.py:4764-4880`. Delivery and cleanup are now independent durable
phases, and **the gating question is `commit_cleanup_status`, never `commit_delivery_status`**:

```
PHASE 1 — DELIVERY            (:4765-4795)
  if delivery_status != done:
      call sink
      on success  -> persist delivery_status = done
      on failure  -> persist failed; RETAIN everything; return   (D1.1 retry contract)
  else:
      the sink is NOT called again

PHASE 2 — CLEANUP             (:4797-4880)
  re-read the trial row FROM THE LEDGER            (a restart has no in-memory history)
  if cleanup_status == done:
      duplicate -> release nothing, delete nothing, re-read no spool, return
  otherwise:
      run the idempotent held-reservation sweep    (first pass AND resume: same code path)
      persist cleanup_status = done
```

### Every path that can now resume cleanup

| entry state | behaviour |
|---|---|
| delivery `none`/`failed`, cleanup `none` | deliver, then sweep — `cleanup="done"` |
| delivery `done`, cleanup `none` (**the crash window**) | **no re-delivery**, sweep resumes — `cleanup="resumed"` |
| delivery `done`, cleanup `done` | duplicate — `cleanup="already_done"`, zero release |
| delivery `failed` | retained; a repaired retry re-enters phase 1 |
| trial `aborted` | `TrialAborted` (unchanged) |

Because `held_reservations` returns only rows still `held`, the resumed sweep naturally sees
exactly the remainder, and `ack_by_event_id` no-ops on any row an earlier partial sweep already
discharged. First pass and recovery are **the same code path** — recovery is a property of the
design, not a separate repair routine nobody exercises.

Every property Beta approved is preserved: the sink is not called twice after durable delivery;
sink failure retains everything; `ack_by_event_id` remains the single release mechanism; partial
cleanup resumes; already-acked rows are no-ops.

### The required gate — all twelve steps

`G-COMMIT-CRASH-RESUME` (`tests/test_s172_staging_backpressure.py`,
`gate_commit_cleanup_resumes_after_crash`). N = 3 held reservations with real files; sink commit
succeeds; **fault injected after exactly one successful `ack_by_event_id`**; then:

1. `commit_delivery_status == done` ✅
2. `commit_cleanup_status != done` ✅
3. `N-1` reservations still held, `N-1` staged files still on disk ✅
4. **process restart modelled by reopening BOTH the `MinerLedger` and the
   `RangeMinerCoordinator` against the same on-disk SQLite file** — the resuming objects share no
   in-memory state with the ones that died, so the durable row is the only channel between them.
   *(Beta asked me to do this "if practical". It was practical and it was done — this is a real
   reopen, not another method call on the same instance.)*
5. `commit_trial` called for the same event ✅
6. the sink is **not** called again (`sink2.commits == []`) ✅
7. all remaining reservations and files discharged ✅
8. the first reservation is **not** discharged twice (`released == N-1`, not `N`) ✅
9. `commit_cleanup_status == done` ✅
10. a further call is a genuine duplicate: `cleanup == "already_done"`, zero release ✅

---

## 2. REVISION 2 — stage-specific eligibility (Beta §4, BLOCKER B)

**Route taken: Beta's preferred `eligible_by_stage`.** I did not take the common-set-invariant
alternative, so no invariant gate is owed.

- `resolve_eligible_by_stage` (`:3288`) resolves `(family, phase) -> workers` using
  **`can_assign_variant`** — the same exact-variant rule `assign_stripes` applies when it builds
  its `compatible` pool (`:2451`), so each stage is sized by the population that stage will
  actually be assigned to.
- `trial_retention_files_required` (`:579`) now takes `eligible_by_stage` and returns
  `(total, per_stage)`. A planned stage with **no** eligible worker **raises** rather than
  contributing 0 files — a stage nobody can run must never be sized as free.
- `trial_retention_requirement` (`:3308`) resolves the sets for **every planned stage before the
  preflight** and carries per-stage detail through to the persisted record.

### Gate + negative arm

`G-STAGE-ELIGIBILITY` builds Beta's example concretely — worker A (cuda, constant variants only),
worker B (rocm, hybrid variants only, tighter hybrid cap) — and proves:

- the resolver partitions exactly as `assign_stripes` would (asserted per worker per stage);
- **the negative arm**: reusing stage 0's population everywhere ≠ the stage-resolved calculation,
  and specifically `hybrid sized from A alone < hybrid sized correctly` — the conservative-bound
  violation Beta named. The fixture is asserted **non-inert** first;
- the preflight uses the correct later-stage population (`per_stage[hybrid].eligible_worker_ids ==
  ["hostB:gpu0"]`);
- a planned stage with no eligible worker fails closed.

`G-MUT-STAGE-ELIGIBILITY` executes the restoration of the submitted "one collection, reused
everywhere" resolver and proves the derived requirement then differs and **understates**.

### ⚠ One factual correction to the ruling's stated mechanism — reported, not worked around

Beta's §4 says the submitted coordinator "computes the preflight when the **first stage's eligible
set** becomes available and passes that one collection across every planned stage", and that this
**understates** a later stage.

The architecture point is right and I implemented it. The stated mechanism is not quite what the
submitted code did, and the distinction matters for how the risk is understood:

- `serve_trial._eligible()` (`:5535`) returns `[w for w in wconn_by_worker.values() if not
  w.quarantined]` — **all connected non-quarantined workers. It is not variant-filtered.**
- `staging_burst_bound_conservative` does not variant-filter either; it takes the max over whatever
  list it is handed (`:567-570`).

So the submitted preflight sized every stage over a **superset** of each stage's true population.
Since the bound is a max over eligible workers, a superset can only **raise** it: the submitted
error was **over-conservative, not understating**. Beta's specific hazard would obtain if
`_eligible()` were variant-filtered; it is not.

**This does not weaken the ruling** — per-stage resolution is more correct, and it is what the
Phase-4 contract implies. But two consequences follow that Beta should see:

1. The corrected bound is **≤** the submitted one for asymmetric fleets. R2 makes sizing *tighter*,
   not safer, so it is the `raise`-on-empty-stage rule (above) that carries the safety margin here.
2. The residual hazard per-stage resolution does **not** address is **workers that connect after
   the preflight**. The preflight runs once, at first-stage setup; a later-connecting worker with a
   tighter cap is in no stage's resolved set. I have not changed that (it is outside the five
   revisions), and flag it as the open question in this area.

---

## 3. REVISION 3 — gate-12 geometry evidence corrected (Beta §3)

**My submitted §2.2 analysis was wrong. The conclusion is deleted, not amended.**

**Confirmation required by report item 10:** the *"1,028 implies roughly five planned
macro-stripes"* conclusion has been **removed everywhere it appeared**:

| location | disposition |
|---|---|
| `docs/CLAUDE_CODE_REPORT_S172_STAGING_CAPACITY_AMENDMENT.md` §2 | replaced by a RETRACTION block carrying Beta's correct geometry |
| `docs/TB_SUBMISSION_S172_STAGING_CAPACITY_AMENDMENT.md` §2.2 | struck through and replaced by a RETRACTION block |
| `miner/range_miner_coordinator.py:233-241` (`CoordinatorConfig` comment) | rewritten to state the real geometry; the "one assignment's exact count" phrasing removed |
| `trial_retention_files_required` docstring (`:603-606`) | rewritten to state 1,028 is stages 0+1 of a 16-stripe run |
| `gate_trial_retention_preflight` docstring | relabelled with its true 2026-08-05 provenance |

A full-tree sweep for `five planned` / `five stripes` / `roughly five` now returns hits **only
inside those retraction blocks** (which necessarily quote what they retract) and in Beta's own
brief. The retractions quote rather than erase, per Beta's own "do not silently edit history".

**The 4-stripe / 116-exact figure is kept, relabelled.** It is the **2026-08-05
staging-back-pressure fixture**, retained as a compact mathematical arm
(`gate_trial_retention_preflight`), no longer described as gate-12 geometry.

### The real gate-12 regression

`G-TRIAL-RETENTION-PREFLIGHT: REAL gate-12 geometry (16 macro-stripes)` —
`gate_trial_retention_preflight_gate12_geometry`, with `total_seeds = 1,073,741,824`,
`miner_stripe_size = 67,108,864`. It establishes exactly the four things Beta specified: derived
`stripe_count == 16`; derived requirement > 512; explicit ceiling 512 ⇒ fail closed before
StripeAssign; files ceiling `None` ⇒ resolved ceiling == derived requirement.

**No total is written as a literal in any assertion.** The gate recomputes the expected value from
the primitives per stage and compares; the only literals are the geometry *inputs* Beta supplied
and the stripe count.

### Report item 4 — the number the derivation produces

Stated here, **not hardcoded in an assertion**:

| stage | phase | eligible | files |
|---|---|---|---|
| `java_lcg` | 1 | 4 | 544 |
| `java_lcg_reverse` | 2 | 4 | 544 |
| `java_lcg_hybrid` | 3 | 4 | 1,088 |
| `java_lcg_hybrid_reverse` | 4 | 4 | 1,088 |
| | | **total** | **3,264** |

**Derived requirement = 3,264 files** for the real 16-stripe geometry against the recorded
heterogeneous worker set.

Two observations worth recording, both confirming the bound behaves as Beta said it should:

- **3,264 exceeds the observed 1,028**, which is what Beta anticipated ("the conservative number
  may exceed the observed"). My earlier 816 was below it only because the geometry was wrong.
- Per stage, the conservative bound sits **above** the observed exact counts — 544 vs the observed
  504 (stage 0) and 544 vs 524 (stage 1). That is the exact-vs-conservative relationship holding on
  real production numbers.

---

## 4. REVISION 4 — the preflight geometry is persisted (Beta §5)

New ledger table `preflight_plans` (`:917-947`), written by `MinerLedger.record_preflight_plan`
(`:1144`) via `_persist_preflight_plan` (`:3455`), **before first dispatch**.

### Schema

| column | content |
|---|---|
| `run_id` | trial identity (PK) |
| `schema_version` | `s172_preflight_plan_v1` |
| `created_at` | timestamp |
| `total_seeds`, `miner_stripe_size`, `macro_stripe_count` | the geometry inputs |
| `stripe_spans_json` + `stripe_spans_sha256` | the actual spans, and a digest over them |
| `stages_json` | planned `(family, phase)` stages |
| `per_stage_json` | per stage: family, phase, **eligible worker ids**, eligible count, derived files |
| `caps_json` | the resolved cap mapping |
| `execution_set_sha256` | immutable digest over `{per_stage, caps}` — binds the stage-specific eligible sets |
| `required_files` | total derived requirement |
| `high_water_mode` | `derived` \| `operator` |
| `configured_files` | the operator value when explicitly supplied, else NULL |
| `resolved_files` | the ceiling in force (NULL on a refusal) |
| `admitted` | whether the trial was admitted |

The stage-specific eligible sets are recorded, which is the R2 interlock Beta flagged.

### Proof it is written, not recomputed

`G-PREFLIGHT-PLAN-PERSISTED` asserts the stored row **equals the preflight's own `detail`** field
by field (`stripe_spans`, `stages`, `per_stage`, `required_files`, `resolved_files`, `caps`), and
then counts derivations: `trial_retention_requirement` is patched with a counter and one preflight
must produce **exactly one** call. A second derivation for logging would make it 2 and red the gate.

Also covered:

- **refusals are persisted** — `admitted = False`, `configured_files = 512`, `resolved_files =
  NULL`, `required_files > 512`. This is the case a post-mortem cannot otherwise reconstruct,
  because a refused trial creates no stripe rows at all;
- **a provenance-write failure must not change the decision** — `record_preflight_plan` is made to
  raise and the trial is still admitted with the same resolved ceiling. `_persist_preflight_plan`
  swallows-and-logs deliberately: a failed audit write must neither refuse an admissible trial nor
  mask a refusal.

---

## 5. REVISION 5 — Gate 37 superseded and replaced (Beta §1)

`tests/test_s172_phase4_coordinator.py`. The old assertion is **retained in place as a commented,
explicitly-marked SUPERSEDED line** with its authority cited — history is marked, not rewritten —
and the function docstring carries the supersession notice.

The replacement proves all seven required properties:

| # | property | how |
|---|---|---|
| 1 | staged object existed and was available to the sink before/during commit | `_CommitProbeSink` records `os.path.exists` for every published staged path **inside `Phase5Sink.commit_trial`** — the only instant at which "before/during" is observable |
| 2 | sink commit succeeded | `len(sink.commits) == 1`, `result["state"] == "committed"` |
| 3 | only after that success was the reservation acknowledged | the probe shows the files present *during* commit; they are absent after |
| 4 | staged file absent afterward | `not os.path.exists(p)` for every probed path |
| 5 | durable cleanup state complete | ledger reopened: `commit_delivery_status == done`, `commit_cleanup_status == done`, `held_reservations == []` |
| 6 | duplicate completed commit re-delivers nothing, releases/deletes nothing | second `commit_trial`: `duplicate is True`, `cleanup == "already_done"`, zero release, `sink.commits` still 1 |
| 7 | failed-commit path still retains | `_gate37_failed_commit_retains` — raising sink, then file retained, reservation held, `commit_cleanup_status != done` |

**Gate 22 — evaluated in the clean/committed sense. Method stated explicitly:** I copied the
working-tree state of every tracked file plus the new test file into a **scratch directory, ran
`git init` there and committed inside that throwaway repository**. That repo has no remotes and is
not the project repo — **no commit was made in `~/distributed_prng_analysis`**. Phase-4 was then
run against that clean tree.

---

## 6. Red-first and mutation evidence per new arm

Both new blocking arms carry executed evidence. **Method:** rather than reconstructing the whole
submitted tree, each arm **restores the submitted logic in-process and executes it**, which is this
suite's established `G-MUT-*` convention and is stronger than a file revert because the restoration
is asserted **non-inert** before the red is claimed. I state this plainly because it is not a
literal "run the old patch" reconstruction.

| arm | restored submitted logic | result |
|---|---|---|
| `G-MUT-COMMIT-CRASH-RESUME` | `MinerLedger.get_trial` patched so `commit_cleanup_status` reports `commit_delivery_status` — **the submitted conflation of the two durable statuses**, which is precisely the defect | non-inert asserted; then `released_reservations == 0`, `N-1` reservations still held, `N-1` staged files stranded. Restoring the real rule immediately recovers them (`cleanup == "resumed"`, `released == N-1`) |
| `G-MUT-STAGE-ELIGIBILITY` | `resolve_eligible_by_stage` patched to return one collection for every stage — the submitted behaviour | non-inert asserted; derived requirement differs and **understates** the stage-resolved value |

The other new R1 arm, `G-PREFLIGHT-PLAN-PERSISTED`, is red-first by construction against the
submitted patch: `preflight_plans`, `record_preflight_plan` and `get_preflight_plan` did not exist.

---

## 7. Full verification results

All runs on VM101 with `~/venvs/torch` active, **after the last edit** (final-state discipline).

| suite | result |
|---|---|
| `test_s172_staging_backpressure.py` ×3 | **48/48, 48/48, 48/48** — exit 0, sentinel PASS |
| `test_s172_staging_partb.py` | **24/24** — exit 0, sentinel PASS |
| `tests/test_s172_elapsed_roundtrip.py` | **6/6** — exit 0, sentinel PASS |
| `test_s172_phase4_coordinator.py` (clean/committed) | **63/63** — exit 0, Gate 22 **PASS**, Gate 37 **PASS** |

48 = the 42 previously green + the 6 R1 arms (real gate-12 geometry, crash-resume, its mutant,
stage-eligibility, its mutant, preflight-plan persistence).

### Assertion-unchanged proof (AST, vs `git show c7058d8:<path>`)

```
=== BACKPRESSURE ===                 === PHASE-4 ===
  pre-existing functions : 53          pre-existing functions : 80
  assertion-IDENTICAL    : 53          assertion-IDENTICAL    : 79
  assertion-CHANGED      : NONE        assertion-CHANGED      : ['gate37_serve_path_two_workers']
  removed                : NONE        removed                : NONE
  added                  : 14          added                  : 1
```

**Exactly one pre-existing gate's assertions changed, and it is the explicitly authorized Gate-37
supersession.** Everything else in both suites is assertion-identical to `c7058d8`.

**No gate-12 production run. No Phase-7 soak. No port 5700 bind.**

---

## 8. Files changed

| file | R1 revisions |
|---|---|
| `miner/range_miner_coordinator.py` | R1 two-phase commit · R2 per-stage eligibility · R3 comment/docstring corrections · R4 `preflight_plans` table + record/read |
| `tests/test_s172_staging_backpressure.py` | R3 relabel + real gate-12 regression · R1/R2/R4 gates and two mutants |
| `tests/test_s172_phase4_coordinator.py` | R5 Gate-37 supersession + replacement + point-7 helper |
| `docs/CLAUDE_CODE_REPORT_S172_STAGING_CAPACITY_AMENDMENT.md` | R3 retraction block |
| `docs/TB_SUBMISSION_S172_STAGING_CAPACITY_AMENDMENT.md` | R3 retraction block |
| `agent_manifests/window_optimizer.json`, `miner/range_miner_protocol.py`, `window_optimizer.py`, `window_optimizer_integration_final.py` | **unchanged in R1** (carried from the amendment) |

`tests/test_s172_phase4_coordinator.py` is the one file beyond the amendment's original six. It is
**required by Revision 5** and therefore in scope. Nothing else was touched: no `.gitignore` work,
no telemetry beyond `elapsed_s`, no seed-domain/cursor surface, no broader cleanup.

---

## 9. Disagreements — reported, not worked around

1. **§2's stated mechanism** (Beta §4). The `eligible_by_stage` correction is implemented as ruled;
   my disagreement is narrowly with the claim that the submitted code *understated* a later stage.
   It over-estimated, because `_eligible()` is not variant-filtered. Two consequences are flagged in
   §2 above: R2 makes the bound tighter rather than safer, and the post-preflight late-connecting
   worker remains unaddressed.
2. **Nothing else.** Revisions 1, 3, 4 and 5 I agree with without qualification — R1 in particular
   was a real stranding defect in my submission and Beta was right to block on it.

---

## 10. Verification-integrity controls (VIR-1…6)

- **execution proof:** completion sentinels + exit codes on all four suites; logs at
  `/tmp/r1_final_*.log`, `/tmp/r1_base_*.log`.
- **clean control:** 42/42 + 6/6 at base before any R1 edit; 79/80 and 53/53 assertion-identical.
- **fault-injection control:** two executed mutants, each asserted **non-inert** before its red is
  claimed; plus the crash-injection in `G-COMMIT-CRASH-RESUME` and the provenance-write failure arm.
- **completion sentinel:** present in all four suites.
- **unavailable-observer behavior:** `_persist_preflight_plan` degrades to a logged error rather
  than altering or masking the admissibility decision.
- **audit claim scope:** this repo tree on VM101 at `c7058d8` + working changes. **No claim about
  live fleet behaviour** — gate 12 and the soak were not run.
- **searched surfaces:** tracked repo; gitignored `agent_manifests/*.json` read live with
  `/bin/grep` (the shell `grep` wrapper honours `.gitignore` and skips `*.json`); `git show` of the
  committed baseline; live VM101 filesystem; live Python imports and execution; a scratch committed
  copy for the Gate-22 clean evaluation.
- **unavailable surfaces:** live rigs; any GPU path; the gate-12 production run itself.
- **governance trail searched:** the R1 brief, the prior amendment brief, my submitted report and
  TB submission cover; skill v19 §§2.7, 2.15, 2.19, §4, §7.

---

## 11. What is NOT done

- **Not committed, not pushed.** Michael commits and dual-pushes; build the `git add` list from §8.
- **Gate 12 / G-PROD-SHAPE and the Phase-7 soak remain HELD** — untouched and unrun.
- The post-preflight late-connecting-worker question (§2) is flagged, not addressed — it is outside
  the five revisions.
- The `.gitignore` sidecar gap reported in the previous round remains untouched, per Beta §10.
