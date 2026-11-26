# READY · benching switch: a poor health grade never removes a model family (2026-11-26)

Worktree `~/.local/state/api-agents/worktrees/20261126-070805-build-benching-switch-933a4a0`, detached at main
`933a4a0` (full `933a4a08ac38fa1800f025905019793eca7c9022`). **Prepared, not landed.** Nothing was committed or pushed and
no pipeline was launched. CPU only, with no GPU, no rig and no ssh. Beta review comes first (draft request
`X-BENCHING-SWITCH_R1_ALPHA_REVIEW_REQUEST_benching_switch.md`, this folder, **not sent**).

**This changes a default.** Before this change, a model family with 3 consecutive critical health grades was
benched: WATCHER got `SKIP_MODEL`, and the next `--compare-models` left that family out. With this change the new switch
`model_skip_rules.benching` defaults to `off`. At that default no family is ever skipped. The count is still kept, and
once it reaches the limit a why-report flag is raised. This is the owner's ruling. It takes effect only when the owner
lands it.

## Prior record

**Searches run** (worktree at `933a4a0`, 2026-11-26). The list follows `CLAUDE.md` "Prior record first" and
`AGENTS.md:56-67`:

```
S = docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_* docs/TB_RULING_REQUEST_* docs/PROPOSAL_* docs/CHAPTER_*
    docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md docs/exchange/
/bin/grep -lEi 'bench|model_skip|max_consecutive_critical|SKIP_MODEL|why-report|why_report|WHY-1|skip registry|skip_registry' $S
/bin/grep -nEi 'model_skip|max_consecutive_critical|SKIP_MODEL|why-report|why_report|WHY-1|benching|benched|skip registry' $S
/bin/grep -rlnEi 'benching|TODO-WHY-1' docs/drafts docs/*.md
git ls-files diagnostics_outputs ; cat diagnostics_outputs/model_skip_registry.json
```
The broad pattern hits "benchmark" in many PROPOSAL and CHAPTER files. Those hits were read only far enough to rule
them out. `docs/BACKLOG.md`, `docs/TODO_CONSOLIDATED.md` and `docs/PROJECT_FILE_CATALOG.md` have no hit on the narrow
pattern.

**Documents found that decide, authorise, schedule or record the subject:**
- **The owner ruling, 2026-11-24, 10:47-10:50.** Source: `logs/e2e_session/OWNER_RULINGS_pending_list_20261124.txt:20-33`,
  quoted at `docs/drafts/loop_v1/READY_LAND_CH14_DIAG_20261125.md:167-184`. The ruling: "a poor health grade never removes a
  model family from the comparison; it triggers a why-report. Benching stays in the code as a switch, off. … Flips when
  landed." TODO-WHY-1 (`:27-28`) defers the automatic trigger and queue.
- **The owner confirmed it today, 2026-11-26, 07:05-07:07.** The record is this task's brief: "a model that keeps
  failing must be flagged so a human or AI can check, not skipped; build it and put it in front of Beta before landing".
- **`CLAUDE.md`, "The learning loop", item 3**, and **`AGENTS.md:54`**: "No model family is dropped for poor health. A
  poor result gets a why-report (Chapter 14)."
- **`docs/drafts/loop_v1/READY_LAND_CH14_DIAG_20261125.md:174-178`**: CH14-DIAG kept benching unchanged and built no off
  switch. Beta confirmed the count-three behaviour (X-CH14-DIAG R1 `:97`).
- **`docs/drafts/loop_v1/READY_LOOP_DIAG_2_20261123.md:229`**: "CD-09 `hours` mode does not reset the counter after
  expiry. One more critical after a skip expires re-skips at once." This is the defect item 3 of the brief fixes.
- **`docs/exchange/from_beta/X-CH14-DIAG_R3_BETA_REVIEW.md:109-117`**: CD-09 changes the compare's family selection.
  Beta asks to keep the explicit error for an invalid value, and leaves activation of benching to the owner.
- **`docs/CHAPTER_14_AUDIT.md:163`**, row 5: the counter "**Removes that model family from `--compare-models`** at ≥ 3.
  This is the one place diagnostics alter model selection."
- **`docs/CHAPTER_14_TRAINING_DIAGNOSTICS.md:2010-2016`**: the registry is a tracked file. Its history shows
  `neural_net` at or above 3 from S91 to S111, and each drop below 3 was a hand reset. Today the tracked file holds
  `neural_net: consecutive_critical 1` (`diagnostics_outputs/model_skip_registry.json`).
- **`docs/LEADS.md:891-897`**: TODO-WHY-1. Until the trigger and queue exist, the lead runs the attribution cycle by hand.
- **The switch registry convention:**
  - `scripts/diagnostics/switch_check.py:1-90` (R1-R10);
  - LL-1/2/3 registration `7e9c5ff`, made from argparse by R10;
  - the CH14-DIAG policy keys, proposed as `ParameterSpec`s with an empty `cli_flag` and never landed
    (`READY_LOOP_DIAG_2_20261123.md:111-131`).

**Labels:**

| Item | Label |
|---|---|
| The switch, default `off`, and the why-report flag | ALREADY RECORDED: the 11-24 ruling. Its default flips at landing, as ruled. |
| The hours-mode count restart after expiry | ALREADY RECORDED as a defect (`READY_LOOP_DIAG_2_20261123.md:229`). The fix is NEW. |
| The flag file `diagnostics_outputs/why_report_flags.jsonl`, its fields, and the action at the limit with benching off (`RETRY`) | NEW |
| The automatic why-report trigger and queue | ALREADY RECORDED, deferred (TODO-WHY-1). Not built. |
| CHANGES A RULING | none. The default change is the ruling itself. |

**Not searched:** the host outside this worktree, ser8 archives, the private repository, project files outside the
repository. `logs/e2e_session/` is not in this worktree, so the 11-24 ruling was read as quoted in the CH14-DIAG READY
note, not at its source.

## What changed (file:line, this worktree)

| File | ± | What |
|---|---|---|
| `training_health_check.py` | +111 −22 | the switch, the flag and the restart (below) |
| `meta_prediction_optimizer_anti_overfit.py` | +19 −2 | the compare reader (`:162-198`) |
| `watcher_policies.json` | +2 −1 | `"benching": "off"` declared at `:168`; description at `:169` |
| `tests/test_benching_switch.py` | new, 457 lines | the suite, 18 tests |
| `tests/bench_switch_mutants.py` | new, 107 lines | the mutants, 10 plus a clean run |
| `tests/test_ch14_diag_wave2.py` | +7 −3 | the CD-09 tests now set `benching: on` (`:307-311`, `:327`, `:350-352`) |
| `tests/ch14_wave2_mutants.py` | +3 −2 | the W13 anchor moved to the reader's new line |

**`training_health_check.py`:**
- `:114-126` constants: `BENCHING_SWITCH = ("off", "on")`, `DEFAULT_BENCHING = "off"`, `WHY_REPORT_FLAGS_PATH`, and the
  flag schema `ch14_why_report_flag/1`.
- `:149-154` `benching_mode()`: validates the value and raises `ValueError` on anything else. `:173` the value is
  validated in `_wave2_settings`, so every check fails closed, the same way `expiry_mode` does.
- `:935-943` `skip_active()`: with `off` a skip is never active. At or over the limit, `info['why_report_needed']` is set.
  Both readers go through this function: `is_model_skipped`, `get_skip_status`, and the compare.
- `:1074-1117` `_check_skip_registry()`:
  - **On, hours mode:** a critical that arrives after the timed skip has expired first restarts the count at 0, so the
    family gets its full retries again (`:1085-1089`). In until_reset mode `skip_active` is always true at or over the
    limit, so the count never restarts.
  - **Off:** the count goes up as before, and `critical_since` records the start of the streak (`:1091-1092`). At or
    over the limit the action stays `RETRY` and the flag is raised (`:1109-1110`).
- `:1120-1126` `_attach_skip_fields()`: sets `consecutive_critical` on `SKIP_MODEL`, as before. When a flag was raised,
  it also sets `consecutive_critical` and `why_report_needed`. It is used by all three result builders: legacy single,
  legacy multi, and fixed.
- `:1129-1161` `_raise_why_flag()`:
  - logs `[WHY-REPORT] …` at WARNING;
  - appends one JSON line to `diagnostics_outputs/why_report_flags.jsonl` with these fields: model family, count, limit,
    `critical_since`, `last_critical`, `flagged_at`, and a note naming TODO-WHY-1;
  - never raises.

  It reuses the CD-11 training health log's conventions: one writer, append-only JSONL beside the other
  `diagnostics_outputs` files, a `schema` field, and a write failure that never changes the outcome (`:979-1026`).
- `:1374-1376` `get_skip_status()` with `off`: adds `benching` and `why_report_needed`. `:1670-1671` the CLI `--status`
  line says `WHY-REPORT NEEDED`.

**`meta_prediction_optimizer_anti_overfit.py`:** the compare reader uses `skip_active` (`:184-187`). With `off` no
family is dropped. Families at or over the limit are named in `[S90][WHY-REPORT]` on stdout (`:195-198`). The reader
writes no flag file, because the health check is the one writer. The import-failure fallback reads `benching` itself and
raises on a bad value (`:172-182`, `:189-190`).

**Registration:**
- **Declared** in `watcher_policies.json:168`. WATCHER reaches it by reading that file (`_load_policies`,
  `training_health_check.py:1231`), the same as `max_consecutive_critical`.
- **Checker:** `switch_check.py` was run with `--main-repo` = the main checkout read at `933a4a0` and this worktree
  linked under `--worktrees-root`. It lists the key as row **S02119** `training_diagnostics.model_skip_rules.benching`,
  "bench-switch (added)", default `off`, class "n/a (config/policy key, not a script switch)". Evidence:
  `data/bench_switch/swc2.md` and `.json`, rc 0, `new switches for the patch 0`, `stale specs 0`.
- **Not in `parameter_registry.py`.**
  - That registry is argparse-keyed, and R10 builds only from argparse options.
  - A `ParameterSpec` for a policy key would be reported **STALE** by R6, because `training_health_check.py` defines no
    `--benching` flag (`switch_check.py:962-973`).
  - The CH14-DIAG policy keys were left unregistered for the same reason. Proposed form, held with those:
    `ParameterSpec(name="model_skip_rules.benching", param_type=ParamType.CHOICE, choices=["off","on"], default="off")`.
- **No manifest entry.** This is not a Step 5 argv switch: WATCHER never puts it on a command line.
- **Nothing governed or hand-landed changes.** None of the 7 files is in `GOVERNED_FILES`
  (`scripts/gate12_parity_gate.py:174-185`), `HAND_LANDED_FILES` or `HAND_LANDED_DIRS` (`scripts/owner_land.py:86-87`).

## What changes at defaults, in plain words

1. **A family is never benched.** After 3 consecutive critical grades the health check still answers `RETRY`, never
   `SKIP_MODEL`, and the next compare still trains that family.
2. **WATCHER is bounded as before.** RETRY re-runs Step 5 at most `max_retries` (2) times per pipeline run, then goes on
   to Step 6 with its "retries exhausted" message (`agents/watcher_agent.py:3192-3240`). The old path was "skip, go on to
   Step 6". So the next Step 6 starts at the same point. What differs is that the following compare includes the family.
3. **The flag.** From the third consecutive critical onward, each check:
   - carries `why_report_needed`;
   - logs one WARNING;
   - appends one line to `diagnostics_outputs/why_report_flags.jsonl`.

   A non-critical check resets the count (WATCHER's `reset_skip_registry`, unchanged), and that ends the flags.
4. **The registry gains `critical_since`** on each entry when benching is off.
5. **`get_skip_status()` adds two keys**: `benching` and `why_report_needed`.
6. **Nothing changes today on the tracked registry.** It holds `neural_net` at 1. A difference appears only when a family
   reaches 3.
7. **With `benching: on`**, behaviour equals `933a4a0` at `until_reset` (tested). In `hours` mode the count now restarts
   after expiry.

## What it predicts for hit rate (not measured)

- **What it changes.** The switch decides which model families stay eligible in the Step 5 compare. The compare picks
  its winner by score (`_s88_extract_score`, `mpo:94-138`; winner `mpo:344`). Step 6 builds pools from that winner.
- **Prediction.** With benching off, a family whose diagnostics grade critical keeps competing. Pools change only when
  that family would win the compare. On today's registry, nothing changes at the next run.
- **The other side.** Benching on can remove the family that would have won on score. That can only change pools, not
  improve the grading.
- **Measurement that would settle it:** walk-forward on back draws (`scripts/diagnostics/e2e_walk_forward.py`), over a
  sequence of WATCHER runs in which some family reaches 3 consecutive critical grades.
  - Run the same draws with `benching: off` and with `benching: on`.
  - Compare pool hits against `k/1000` on draws the sieve never saw.
  - [UNVERIFIED] whether the walk-forward driver can set a `watcher_policies.json` key per arm. Not checked.

## Tests (VM 101, CPU only, `CUDA_VISIBLE_DEVICES=""`, `PYTHONDONTWRITEBYTECODE=1`, suites one at a time)

| Suite | Result | Log (`data/bench_switch/`) |
|---|---|---|
| `tests/test_benching_switch.py` | **18 passed** | `final_benching_switch.log` |
| `tests/bench_switch_mutants.py` | **11/11**: the clean copy passes and 10/10 mutants are killed on their credited tag | `mutants2.log`, `mutants2/` |
| `tests/test_ch14_diag_wave1.py` | 37 passed | `final_ch14_diag_wave1.log` |
| `tests/test_ch14_diag_wave2.py` (CD-09 tests set `benching: on`) | 37 passed | `final_ch14_diag_wave2.log` |
| `tests/ch14_wave2_mutants.py` (all) | 36/36 killed | `w2mut_all.log` |
| `tests/ch14_wave1_mutants.py` (all) | 25/25 killed | `w1mut_all.log` |
| `tests/test_ch14_diag_land_defaults.py --part single` (base `7f4505a`) | 85/86 gates, 9/9 mutants (see below) | `land_single2.log` |
| `tests/test_ch14_diag_land_defaults.py --part compare` | 28/29 gates, 2/2 mutants (see below) | `land_compare.log` |
| `scripts/diagnostics/switch_check.py` | rc 0; `benching` seen as S02119 | `swc2.log`, `swc2.md` |

**The one red gate in each landing-defaults run** is `BOUNDARY.nothing_outside_out_changed`. In both runs its only
change is the run's own stdout log: `data/bench_switch/land_single2.log` or `land_compare.log`, which I redirected to a
place outside `--out`. Every other boundary gate is green.

A first single run (`land_single_run1_concurrent.log`) gave 84/86. I had written the new test files during that run, and
both boundary gates caught it. It was re-run with nothing else active.

**What each new test checks** (credited tags in brackets):
- **(a) benching off, absent key and explicit `off`.** Five criticals through `check_training_health()`:
  - every answer is `RETRY` [OFF_THC], and the count is 5 [OFF_COUNT];
  - `is_model_skipped` is False, and `get_skip_status` says `why_report_needed` [OFF_THC];
  - the flag is missing on checks 1-2 and present on 3-5, with family, count, limit and three timestamps
    [FLAG_RESULT];
  - the flag file holds counts 3, 4 and 5 under the test's temp dir [FLAG_FILE];
  - 3 WARNING records carry `[WHY-REPORT]` [FLAG_LOG].

  The multi-model path flags the winner. The compare reader, run with `subprocess.run` faked against a registry holding
  `neural_net` at 7, trains all four families and prints `[S90][WHY-REPORT]` [OFF_MPO, OFF_MPO_WHY]. Non-vacuity: with
  `on`, the same registry drops `neural_net` [ON_MPO].
- **(b) on + until_reset equals the code at `933a4a0`.** The base code is read with `git show`, never restated. One
  sequence is compared at every step: 4 single criticals, a reset, 1 critical, then 3 multi-model criticals. At each
  step the suite compares:
  - the result;
  - the registry (timestamps reduced to present/absent);
  - `is_model_skipped` for all four families;
  - `get_skip_status`;
  - the archived history;
  - the flag file.

  The sequence runs twice: with `{"benching":"on"}` and with every `model_skip_rules` key explicit [ON_IDENTITY]. The
  base must reach `SKIP_MODEL`, or the test fails as vacuous. The compare reader on a mixed registry gives the same family
  list and error as the base [ON_MPO_IDENTITY]. The base must drop two families.
- **(c) hours.** An entry at 3 whose skip expired 25 h ago, followed by three criticals:
  - new code: `RETRY, RETRY, SKIP_MODEL`, counts 1, 2, 3 [HOURS_RESTART];
  - base: `SKIP_MODEL` ×3, the recorded defect (non-vacuity).

  An entry still inside its 24 h is identical to the base [HOURS_ACTIVE_IDENTITY].
- **(d) bad values.** `"OFF"`, `"true"`, `True`, `0`, `None` and `"bench"` each raise `ValueError` naming `benching`
  from `benching_mode`, `skip_active` and `_wave2_settings` [BAD_VALUE]. `check_training_health()` turns the error into
  `PROCEED_WITH_NOTE`/`absent` with the module-error note naming benching, and writes no registry and no flag, the same
  way `expiry_mode` behaves. The compare reader raises before any training [BAD_VALUE_MPO].
- **(e) isolation.** At session start the suite snapshots the worktree root's top level, `data/` (minus `--basetemp`),
  `catboost_info/`, `diagnostics_outputs/`, `models/`, and `git status --porcelain --ignored`. It compares them at the
  end [ISOLATION]. The one declared exception is the file named by `BENCH_RUN_LOG`, the run's own stdout. A non-vacuity
  test shows that the snapshot sees a write in `data/`, in `catboost_info/` and at the top level. No git repository is
  created.

**Mutants** (`tests/bench_switch_mutants.py`). Each changes a copy under `--out` and must turn its named test red on
its tag:

| Mutant | Killed on |
|---|---|
| M01 off branch removed (registry) | OFF_THC |
| M02 off branch removed (`skip_active`) | OFF_MPO |
| M03 flag removed from result | FLAG_RESULT |
| M04 flag file write removed | FLAG_FILE |
| M05 warning demoted to info | FLAG_LOG |
| M06 hours restart removed | HOURS_RESTART |
| M07 restart made unconditional | ON_IDENTITY |
| M08 default flipped to on | OFF_THC |
| M09 validation removed | BAD_VALUE |
| M10 compare WHY line removed | OFF_MPO_WHY |

The first run had 10/11: M09 failed, but not on its tag, because `pytest.raises` carried no credit. The assertions were
credited and the second run gives 11/11.

## What I chose not to handle

1. **The automatic why-report trigger and queue (TODO-WHY-1)** are not built. Only the flag is raised.
2. **No change to `agents/watcher_agent.py`** (hand-landed). With benching off, WATCHER's existing RETRY branch logs the
   critical grade. The flag shows in the health check's own WARNING and in the flag file, not in WATCHER's log line or
   Telegram. **Optional owner item:** in the RETRY branch (`agents/watcher_agent.py:2598-2603`), add
   `if health.get("why_report_needed"): logger.warning("[WATCHER][HEALTH] why-report needed: %s", health["why_report_needed"])`.
   It is not required for the ruling.
3. **The flag file has no cap.** It gets one line per critical check at or over the limit, and a reset ends the lines.
4. **No `parameter_registry.py` entry** and no manifest entry. Reasons are under Registration.
5. **The compare reader's import-failure fallback** (`mpo:179-182`, `:188-190`) is not reached by any test. It is
   reached only when `training_health_check` cannot be imported.
6. **The fixed grading path** uses the same `_attach_skip_fields`, but no test drives it to the limit. That path is off
   by default (`grading_rules: legacy`).
7. **No GPU run, no real-data run, no hit-rate measurement.**
8. **Housekeeping.** I removed only the `__pycache__/` directories my own early runs created (newer than 07:08 today).
