# S172 D6.2 — BOUNDED REPAIR REPORT

**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_S172_D6_2_BOUNDED_REPAIR.md` (REV1).
**Base:** `f7583bc`, repaired on top — **not reverted**. Working tree at `ee6ce18` + this repair.
**Host:** VM101 (`192.168.3.177`) as `michael`, venv `~/venvs/torch`. Nothing committed, nothing
pushed, WATCHER not run.
**Completion sentinel: `PASS`.**

Beta ratified the architecture; only what the two blockers require was changed. No REV6 item was
reopened: the record-ordinal continuation remains the replay-key authority, the four ported NR
suites, `seed` as member A's field name, `member_role`, the 24-field schema, CSR encoding, digest
separation, the asymmetric A/B contract, nine-row recovery, sequence repair, pre-clear walls,
canonical reconciliation, array parity, run-ID confinement, context binding, durable resume
provenance and the clear-after-installed-pair ordering are all untouched.

---

## 1. BLOCKER 1 — the guard rejected every normal resume

### 1.1 The rename, and every site it touched

The concept is now **`resume_record_ordinal_floor`**: the maximum **persisted record ordinal**
recovered from the checkpoint. It is 1-based, it is a canonical *record* field (`trial_number`),
and it has no arithmetic relationship to Optuna's 0-based `trial.number`. The old name asserted
that relationship, and the false name is what produced the defect.

| # | file:line (post-repair) | site | before → after |
|---|---|---|---|
| 1 | `window_optimizer_integration_final.py:967` | `_prepare_checkpoint_run_context` docstring, return contract | `(context, recovered_max_trial_number_or_None)` → `(context, resume_record_ordinal_floor_or_None)`, with the "not an Optuna trial number" statement and the sole-consumer pointer |
| 2 | `window_optimizer_integration_final.py:986-992` | §4.4 matrix, row 3 wording | "enforced by the two checks in the Optuna study body" → the record-ordinal continuation + the loaded-study proof |
| 3 | `window_optimizer_integration_final.py:1069-1072` | recovery local | `_floor = max(...)` → `_record_ordinal_floor = max(...)`, with the naming rationale |
| 4 | `window_optimizer_integration_final.py:1078` | the `RESUMED` log line | `recovered_max_trial_number=` → `resume_record_ordinal_floor=` |
| 5 | `window_optimizer_integration_final.py:1088` | the return | `return _ctx, _floor` → `return _ctx, _record_ordinal_floor` |
| 6 | `window_optimizer_integration_final.py:2657-2658` | the call site in `optimize_window` | `_d6_2_context, _d6_2_resume_floor = ...` → `_d6_2_resume_record_ordinal_floor` |
| 7 | `window_optimizer_integration_final.py:2683-2696` | the record counter (**the floor's only consumer**) | `trial_counter = {'count': int(_d6_2_resume_floor or 0)}` → same statement on the new name, plus the §1.3.2 note that this is the only consumer and why the old name produced the off-by-one |
| 8 | `window_optimizer_integration_final.py:2765-2781` | the attribute seam | `strategy._resume_trial_floor = _d6_2_resume_floor` → `strategy._d6_2_require_loaded_study = bool(resume_checkpoint)` |
| 9 | `window_optimizer.py:451-461` | `BayesianOptimization.search` forwarding | forwarded the floor → forwards the boolean |
| 10 | `window_optimizer_bayesian.py:504-524` | the study-body read | `_resume_trial_floor = getattr(self, '_resume_trial_floor', None)` → `_require_loaded_study = bool(getattr(self, '_d6_2_require_loaded_study', False))` |

**Item 7 is the ratified repair and its behaviour is unchanged** — only the identifier moved.
`trial_counter` still initialises to the recovered floor so the next `+= 1` yields `floor + 1`.

### 1.2 Proof that no `trial.number`-vs-floor comparison survives

**AST**, whole tree, executable nodes only (`ast.Compare` mentioning `trial.number` alongside a
floor or the resume counter; and any live `ast.Name`/`ast.Attribute` still named
`_resume_trial_floor` / `_d6_2_resume_floor`):

```
AST executable hits: NONE
```

**grep**, all `.py` surfaces including comments:

```
tests/test_s172_d6_2_checkpoint_reconciliation.py:2193   (the gate that searches for the name)
tests/test_s172_d6_2_checkpoint_reconciliation.py:2198
tests/test_s172_d6_2_checkpoint_reconciliation.py:2201
window_optimizer_bayesian.py:506                          (historical note in a comment)
window_optimizer_integration_final.py:2691                (historical note in a comment)
```

Every remaining occurrence is a comment or the gate's own search string. The AST arm is the
binding one — a text grep alone would have gone green on `if False:` around a live comparison,
which is exactly the class of miss `2389b61` demonstrated.

The same AST sweep is now **inside** `G-RESUME-INTEGRATED`, so a reinstatement reds a gate rather
than requiring someone to re-run this command.

### 1.3 What was retired, named

* **`window_optimizer_bayesian.py`, the in-objective check** (was `:512-532`, the addendum §4
  "CHECK 2"): `if (_resume_trial_floor is not None and int(trial.number) <= int(_resume_trial_floor))`.
  **Deleted.** Replaced by the §1.3 comment block explaining the 0-based/1-based error, so the
  next reader cannot reintroduce it as an oversight.
* **`window_optimizer_bayesian.py`, the pre-flight nonterminal scan** (was `:762-792`, "CHECK 1"):
  the `WAITING`/`RUNNING` sweep rejecting enqueued trials at or below the floor. **Deleted**, with
  a note at the same location stating why an enqueued warm-start trial is not a replay-key hazard:
  the replay key carries the *persisted record ordinal*, which continues from the recovered
  maximum in the integration layer regardless of which number Optuna hands out.
* **`strategy._resume_trial_floor` forwarding** (at `f7583bc`,
  `window_optimizer_integration_final.py:2702`): **removed.** I checked before deleting — the only
  other consumer was `window_optimizer.py:454-455`, which existed solely to relay it into the study
  body; it now relays the boolean instead. Nothing else in the tree read it (see the AST sweep
  above). All line numbers in this bullet and the two above it are **pre-repair** positions.
* **Mutant `M23`** ("an enqueued trial at or below the recovered maximum executes") and its two
  helpers `_det_enqueued_guard` / `_assert_enqueued_mutant`: **removed**, together with the
  `_extract_block` helper that existed only to serve them.
* **Gate `G-TRIAL-NAMESPACE`** (`g_trial_namespace`, 15 assertions): **removed.**

### 1.4 The four-row combination policy, retained and strengthened

Rows 1/2/3/4 are unchanged and `G-COMBINATION-MATRIX` still passes 4/4. Checkpoint-without-study-
resume still rejects (row 4).

**The strengthening (§1.3.5).** `optuna.create_study(..., load_if_exists=_resume)` *creates* a
fresh study when the name does not exist, so a resume could silently become a fresh study —
restarting the record ordinal against a recovered checkpoint. Two walls now stand in
`run_optimization`, both **before the first objective can execute**:

* `window_optimizer_bayesian.py:708-736` — reject when a checkpoint resume is in force and
  `_resume` is false. `_resume` is set only after `optuna.load_study(...)` returned a study with
  completed trials, so it is positive evidence the study existed, not an inference from the
  request. This wall sits **above `create_study`**, so no study file is created either.
* `window_optimizer_bayesian.py:752-764` — reject when the constructed study object carries zero
  trials. The first wall is checked against the discovery bookkeeping; this one against the object
  that was actually built, so a future change to `load_if_exists` cannot make a resume silently
  fresh without one of the two reding.

---

## 2. BLOCKER 2 — NP2 executed before D6.2 validation existed

### 2.1 The rejection, and the exact line it sits above

`window_optimizer_integration_final.py:1979` — `if resume_checkpoint and int(n_parallel) > 1:`,
raising `CheckpointResumeError` at `:1980`. It is **the first executable statement of
`optimize_window`** (def at `:1926`; everything between is the hop-3 comment block).

Measured by AST in `G-NP2-SCOPE`, in the same run that produced this report:

| landmark | line |
|---|---|
| **the rejection guard** | **1979** |
| `if n_parallel > 1:` (the NP2 block) | 2032 |
| shared Optuna study creation (`_osetup.create_study`) | 2309 |
| `[NP2-KILL]` SSH to every AMD rig (`_pre_kill_sp.run`) | 2404 |
| the partition fork (`_mp.Process(...)` / `.start()`) | 2439 / 2463 |

The rejection precedes study creation, the SSH and the fork — not merely the fork, as the brief
requires.

`optimize_window` is wrapped once more at `:3059-3072` (the S159E ZMQ-shutdown `finally`); that
wrapper adds no statement before the delegation, so the rejection remains first for every real
caller.

### 2.2 What is claimed, and what is not

Stated in the module header (`window_optimizer_integration_final.py:362-376`), in
`utils/checkpoint_d6_2.py`'s docstring, and in the rejection message itself:

* D6.2 checkpoint recovery **and** the S166 in-memory clear are certified **only** for the default
  single-Optuna-trial path, `n_parallel == 1`.
* **That path still distributes every sieve trial across the full GPU cluster.** The scope limit is
  on Optuna parallelism, not on fleet use.
* **No NP2 claim is made** — not resume, not accumulator clearing. The forked partition workers
  carry no installed D6.2 run context, so their flush attempts cannot clear memory; and concurrent
  partition writers cannot safely share the present checkpoint member pair. That needs a separate
  transaction design and is not attempted here.

---

## 3. Textual correction (§3)

`utils/checkpoint_d6_2.py:22` stated member A carries **`seeds`**. Corrected to **`seed`** — the
singular canonical record field. The corrected text, in full:

```
  * **Member A is a marker / compatibility stub.** It carries `seed`, `score`
    and its complete identity block, and NOTHING more. The field name is
    `seed` — the singular canonical record field, not a plural array name; the
    ratified record domain and `MEMBER_A_PAYLOAD_FIELDS` both use `seed`, and
    the earlier `seeds` wording here was stale. It is never an
    accumulator backup, is never described as one, and no path here consumes it
    as one.
```

The implementation agrees: `MEMBER_A_PAYLOAD_FIELDS = ("seed", "score")`
(`utils/checkpoint_d6_2.py:190`), and the integration module's own header already said `seed`
(`window_optimizer_integration_final.py:340`).

---

## 4. The three new gates

### 4.1 `G-RESUME-INTEGRATED` — replaces `G-TRIAL-NAMESPACE`

**Why the old gate could not have caught this.** It exercised the guard with **fabricated numbers**
— trial 6 against floor 5, and 0/3/5 against floor 5 — instead of constructing the real
relationship between completed Optuna trials and persisted record ordinals. A gate built from
invented numbers cannot discover an off-by-one between two real counters. That is the VIR-2
vacuous-detector class, and it is why 29 green gates certified a feature that never worked.

**`k` is derived from an actual run — nothing is fabricated.** The gate drives the **live**
sampler-neutral study body (`OptunaBayesianSearch.run_optimization`, CPU-only, RandomSampler,
stub objective, no GPU and no coordinator) for a fresh run; then reads `k` off the study that
really ran (`len([t for t in study.trials if COMPLETE])`); writes a real checkpoint from the
records those trials produced; recovers it through the production
`_prepare_checkpoint_run_context`; and really resumes the same study.

The record ordinals are not restated by the gate either: the initialisation and the increment are
**extracted from the live `optimize_window` source by AST and executed**
(`_live_ordinal_statements` / `_live_record_ordinal_counter`), so the arithmetic under test is the
production arithmetic.

**Reported by the gate (not a boolean — VIR-1):**

```
ok  G-RESUME-INTEGRATED  [6 assertions]
    k=3 derived from a real run · optuna [0, 1, 2, 3] · ordinals [1, 2, 3, 4] · floor=3 ·
    trial 3 executed, its record took 4
```

* Phase 1 ran Optuna trials **0, 1, 2** and produced record ordinals **1, 2, 3** — `k = 3`.
* The recovered floor is **3**.
* Phase 2: **Optuna trial 3 EXECUTED.** At `f7583bc` this is exactly the trial the guard rejected
  (`3 <= 3`).
* Its record took ordinal **4 = k+1**.
* **No replay-key collision.** The new record deliberately reuses a recovered *seed*, so its
  ordinal is the only thing keeping it out of a recovered record's replay key: the union of
  `(seed, trial_number, skip_mode)` over the recovered records and the new one has 4 distinct keys
  for 4 records, `reconcile` accepts, and the new record enters the canonical state under ordinal
  4. Reconciliation returns 3 rows because it ends in the frozen per-seed `_select_l2_winners` —
  the row count is not the proof; the key set is.
* **The collision arm is not vacuous:** the same record with a *restarted* ordinal (1) raises
  `AccumulatorConsistencyError`, asserted in the same gate.

Also folded in: the AST sweep of §1.2, the repo-wide executable-name sweep for
`_resume_trial_floor`, and the retained "no assignment to `.number`" check.

### 4.2 `G-MISSING-STUDY`

```
ok  G-MISSING-STUDY  [3 assertions]
    absent study -> rejected with zero objectives and no DB created;
    present study -> the same wall lets the resume through
```

* **Fault case:** checkpoint resume in force, `resume_study=True`, named study absent → rejected.
  **Proof no objective ran:** the gate's objective appends its trial number to a list and the list
  is asserted empty. **Proof the rejection precedes study creation:** a before/after snapshot of
  `optuna_studies/` shows no new database. (The named-file check alone would have been unsound —
  the fresh path *renames* the study to `window_opt_<epoch>`, so only a directory diff catches it.)
* **Clean control:** the same wall with the study **present** — a real 2-trial run followed by a
  real resume, which proceeds and runs trial 2.

### 4.3 `G-NP2-SCOPE` — zero process starts, CPU-only

```
ok  G-NP2-SCOPE  [6 assertions]
    rejection line 1979 < NP2 block 2032 < study 2309 < [NP2-KILL] SSH 2404 ·
    0 sentinel events · children 0->0 · n_parallel==1 and no-checkpoint pass through
```

No rig, no GPU, no network. Two independent arms:

* **Execution.** The live `optimize_window` source is extracted by AST, dedented and actually
  **called** with `n_parallel=2, resume_checkpoint=<id>`, while every surface §2.2 names is
  sentinelled through `sys.modules`: `subprocess.run/Popen/call/check_call/check_output`,
  `multiprocessing.Process/Pool/Queue/SimpleQueue`, `optuna.create_study/load_study` and
  `optuna.storages.RDBStorage`, plus `os.fork`. Any touch records the event and raises.
  **Result: `CheckpointResumeError`, sentinel events `[]`.**
* **Process count.** Real children read from `/proc/self/task/*/children` before and after:
  **0 → 0**, no new child PIDs.
* **Position.** By AST (never text — `if False:` around a raise leaves the text in place): exactly
  one `resume_checkpoint`+`n_parallel` guard exists in `optimize_window`, and its line is above the
  NP2 block, the study creation and the SSH.
* **Clean controls:** the live guard node is compiled and executed with `n_parallel=1` (checkpoint
  present) and with `n_parallel=8` (no checkpoint) — neither raises.

---

## 5. Fault-injection controls (four-part kill rule on each)

| mutant | injected defect | detector must red | signature |
|---|---|---|---|
| **M23** | the `trial.number <= floor` comparison **reinstated** in `optuna_objective`. On a resume `_resumed_completed` **is** the recovered record-ordinal floor (k completed trials → k persisted ordinals), so this is the same arithmetic Beta found, not a lookalike. Gated on `_resume` so the fresh phase still runs and the kill is attributable to the resume. | `G-RESUME-INTEGRATED` | `AssertionError: the resume was REJECTED instead of continuing: RuntimeError: [M23] reinstated trial.number <= floor: trial.number 3 does not exceed the recovered maximum 3` |
| **M24** | the NP2 rejection **moved below the fork point** — disabled at the top, reinstated immediately before `_rq = _mp.Queue()` | `G-NP2-SCOPE` | `AssertionError: the NP2 path started or created ['optuna.storages.RDBStorage()'] BEFORE the rejection — §2.2 requires the refusal above study creation, worker launch, SSH and any other fleet action` |
| **M25** | the missing-study case **falls back to a fresh study** — both loaded-study walls bypassed | `G-MISSING-STUDY` | `AssertionError: a checkpoint resume against a NON-EXISTENT study was allowed to proceed — load_if_exists silently created a fresh study, restarting the record ordinal against a recovered checkpoint` |

All three carry the full rule: **applies-once** (each anchor asserted unique by `_patch`),
**mutated-path** (the mutated text is loaded as the module/source the detector actually
exercises), **detector-clean** (positive control against the unmutated module passes), and
**injected-defect** (the failure signature names the injected defect).

M24 and M25 each patch **two** sites because the single behaviour they remove is guarded in two
places; the concept injected is one, and the label says so. M24's kill is now **behavioural** (the
execution arm runs before the position arm, so a moved rejection is caught by what it *does*, not
only by a line number).

---

## 6. Gate and mutant counts

| | gates | assertions | mutants |
|---|---|---|---|
| **before** (`f7583bc`, as recorded in its commit) | 29/29 | 377 | 23/23 |
| **after** (this repair) | **31/31** | **377** | **25/25** |

The assertion total is unchanged by coincidence, and the arithmetic is exact: the removed
`G-TRIAL-NAMESPACE` contributed **15** assertions (measured), and the three replacements
contribute **6 + 3 + 6 = 15**. Gates +2 (one removed, three added); mutants +2 (M23 replaced,
M24 and M25 added).

I did not re-run the suite at `f7583bc` — the brief forbids reverting, and an export of the tree
to a non-git directory makes `_repository_state()` fail, which contaminates four unrelated gates.
The "before" row is `f7583bc`'s own recorded result; the 15-assertion figure for
`G-TRIAL-NAMESPACE` was measured from that exported tree, where the gate itself runs cleanly.

---

## 7. §4 non-regression

All on VM101 with `~/venvs/torch` active, each `python3 -u <suite> | tee`, never piped to `tail`.
Logs in `/tmp/nr_d6_2_repair/`.

| # | suite | file | result |
|---|---|---|---|
| 1 | **D6.2** (replaced + new gates) | `tests/test_s172_d6_2_checkpoint_reconciliation.py` | **31/31 green, 377 assertions, 25/25 mutants — `RESULT: PASS`** |
| 2 | D6.1 ported | `tests/test_s172_d6_1_flush_durability.py` | 15/15 green, 8 mutants |
| 3 | D6 3.A | `tests/test_s172_phase5_d6_production_adapter.py` | 9/9 green, 16 mutants |
| 4 | D6 threshold path | `tests/test_s172_phase5_d6_threshold_path.py` | 17/17 green, 11 mutants |
| 5 | D3.5 finalizer | `tests/test_s172_phase5_d3_5_finalizer.py` | 60/60 green |
| 6 | D3.25 candidate ingress | `tests/test_s172_phase5_d3_25_candidate_ingress.py` | 13/13 green |
| 7 | D3 columnizer | `tests/test_s172_phase5_d3_columnizer.py` | 10/10 green |
| 8 | D3.0 encoding contract | `tests/test_s172_phase5_d3_0_encoding_contract.py` | 10/10 green |
| 9 | D0 | `tests/test_s172_phase5_d0.py` | 12/12 green |
| 10 | D4 serial backend | `tests/test_s172_phase5_d4_serial_backend.py` | 8/8 green |
| 11 | import gate | `tests/test_s172_process_sharded_import_gate.py` | 7/7 green |
| 12 | Phase 1 | `tests/test_s172_phase1_scaffolding.py` | 6/6 green |
| 13 | Phase 2 | `tests/test_s172_phase2_protocol.py` | 6/6 green |
| 14 | Phase 3 | `tests/test_s172_phase3_worker.py` | 17/17 green |
| 15 | **Phase 4** | `tests/test_s172_phase4_coordinator.py` | **62/63 — Gate 22 only. EXPECTED, see below** |
| 16 | D1.0 | `tests/test_s172_phase5_d1_workflow.py` | 7/8 — the single red is its nested Phase-4 NR arm |
| 17 | D1.1 | `tests/test_s172_phase5_d1_engine.py` | 17/18 — same, nested Phase-4 NR arm |
| 18 | D2 directional uniqueness | `tests/test_s172_phase5_d2_directional_uniqueness.py` | 6/7 — same, via D1.1 |
| 19 | D5 process-sharded | `tests/test_s172_phase5_d5_process_sharded.py` | 24/25 — same, via D1.1 |
| 20 | 6-P0.5 dataset authority | `tests/test_s172_phase6_p05_dataset_authority.py` | 37/37 green — `RESULT: PASS` |
| 21 | Chapter 1 P0 | `tests/test_chapter1_p0_corrections.py` | 12/12 green (6 gates + 6 mutants) |
| 22 | PRNG encoding | `tests/test_prng_encoding.py` | 8/8 green |

**The Phase-4 red is the class the brief anticipated, and is not a regression.**
`gate22_coexistence` (`tests/test_s172_phase4_coordinator.py:1602-2363`) runs
`git status --porcelain` and asserts every changed `.py` is in a hardcoded allowlist. It names
exactly two files:

```
AssertionError: unexpected changed .py files:
  {'tests/test_s172_d6_2_checkpoint_reconciliation.py', 'utils/checkpoint_d6_2.py'}
```

Both are files this repair modified **and left uncommitted**. The other three files I touched —
`window_optimizer_integration_final.py`, `window_optimizer_bayesian.py`, `window_optimizer.py` —
are already in the allowlist, which is why they are not named. At `f7583bc` the tree was clean, so
`changed_py` was empty and the gate was green; it will be green again once Michael commits.
Rows 16–19 each fail **only** in a nested non-regression arm that shells out to Phase 4
(`tests/test_s172_phase4_coordinator.py exited 1`); every gate those suites own is green.

**Phase-4 Gate 22 was not edited**, per the brief and CLAUDE.md.

**No Wall A/B rerun** (Beta).

---

## 8. Scope — what was not touched

Nothing Beta ratified. `_l2_sort_key`, `_select_l2_winners`, `CANONICAL_ARRAY_CONTRACT`,
`utils/prng_encoding`, `canonical_map_hash`, the three finalizer validators,
`_RusageChildrenSampler`, **Phase-4 Gate 22**, the nine-row recovery matrix and the digest
preimages are all unchanged. `f7583bc` was not reverted.

Files changed: `window_optimizer_integration_final.py`, `window_optimizer_bayesian.py`,
`window_optimizer.py`, `utils/checkpoint_d6_2.py`,
`tests/test_s172_d6_2_checkpoint_reconciliation.py`. **No new file was created**, so Phase-4
Gate 22's untracked-`.py` sensitivity was never engaged and `git status --porcelain -uall` shows
no untracked entries.

**One side-effect note, disclosed rather than hidden.** `G-RESUME-INTEGRATED` and
`G-MISSING-STUDY` drive the real study body, and the live code resolves study storage under the
repository's `optuna_studies/` by absolute path. The gates therefore run in a scratch cwd carrying
a symlink to that directory (so `optimal_window_config.json` and `bidirectional_survivors.json`
land in the scratch dir, not the repo) and delete every database they create, verified by a
before/after directory diff — an early iteration of the gates leaked three, which were removed.
Every study is addressed by an **explicit name**, so the auto-discovery glob over
`window_opt_*.db` can never load or extend a production study.

---

## Verification-integrity controls (VIR-1…6)

* **execution proof:** `G-RESUME-INTEGRATED` reports the actual `k`, the Optuna numbers used and
  the record ordinals produced (`k=3 · optuna [0,1,2,3] · ordinals [1,2,3,4] · floor=3`), not a
  boolean. `G-NP2-SCOPE` reports the sentinel event list and the real child-process count
  (`0->0`). Every gate prints its assertion count; a gate that asserts nothing cannot pass.
* **clean control:** a normal fresh run (no checkpoint, no study resume) passes unchanged —
  `G-COMBINATION-MATRIX` rows 1 and 2, and phase 1 of `G-RESUME-INTEGRATED`. **A normal
  `n_parallel == 1` resume now succeeds, which it did not at `f7583bc`.** `G-MISSING-STUDY`'s
  present-study control proceeds through the same wall that rejects the absent one.
  `G-NP2-SCOPE` executes the live guard with `n_parallel=1` and with no checkpoint; neither raises.
* **fault-injection control:** M23, M24, M25 above, four-part kill rule on each, all three killed.
* **completion sentinel:** `PASS | FAIL | UNAVAILABLE | INCOMPLETE`; only `PASS` accepts.
  **D6.2: `PASS` (31/31, 377 assertions, 25/25 mutants).**
* **unavailable-observer behavior:** every gate here is CPU-only with no fleet dependency. No arm
  returned `UNAVAILABLE`.
* **audit claim scope:** repo-scoped at `f7583bc` plus this repair, on VM101's working tree.
* **searched surfaces:** the tracked repo (whole-tree AST walk for the retired seam and for
  `trial.number` comparisons; `/bin/grep` for the text, since the shell `grep` wrapper honours
  `.gitignore`); the live source of `optimize_window` and `run_optimization` by AST; the real
  `optuna_studies/` directory before and after each gate run.
* **unavailable surfaces:** host state on VM101 and the rigs; deployed uncommitted files; the
  `n_parallel > 1` execution path itself, which is now refused rather than exercised — that is the
  point of the repair, and no NP2 claim is made.

---

**STOP for Team Alpha review.** Nothing committed, nothing pushed, WATCHER not run.
