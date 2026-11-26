# X-REGISTRY-REPAIR  R1  REVIEW_REQUEST  LL switch registrations and manifest commands (parts 2 and 3), with the landed registry repair (part 1) for the record (courtesy review)

- **Review class:** BEFORE-COMMIT for parts 2 and 3, courtesy.
  - **The owner asked for this review:** "yes to landing part 1 (repair) and to sending parts 2 and 3 past Beta first" (owner, 18:59, recorded in `logs/e2e_session/TASK_REG_BETA_SEND_20261125.txt`).
  - The files are the parameter registry, a diagnostic checker, tests, and two commands for hand-landed manifests. None is in the before-commit classes (coordinator, worker and kernel code; the verifier and launch path).
  - The review is not binding; the owner decides.
- **Base commit:** `579dbb7` (main = origin = public).
- **Part 1 is already landed.** The registry repair (20 stale `ParameterSpec`s dropped, 9 kept unchanged) is `579dbb7`. `agents/registry/parameter_registry.py` at `579dbb7` is sha256 `72d00315…`. It is here **for the record**: you may still object to it, and an objection goes to the owner.
- **Under review before landing:**
  - **part 2:** the 7 LL switch registrations on top of the repaired file, the checker change and the tests;
  - **part 3:** two owner commands that declare 5 of those switches in `agent_manifests/prediction.json` and `reinforcement.json`.
- **Rounds:** one. No second round will follow. Alpha changes nothing after your reply and reports your positions to the owner.
- **Allowed answers per question:** `NO OBJECTION` · `OBJECTION` (with `LIVE` or `THEORETICAL`) · `SUGGESTION` · `NEEDS EVIDENCE`.
- **Scope:** please keep to real cases on this system as it is today: VM 101, WATCHER as the manifests are now, and the LLM paths as they are reached today.

**Parts 2 and 3 are NOT on main.** Read each file at the worktree path given in the table, all under `~/tfm_worktrees/registry-repair` (detached at `7f4505a`). The sha256 is that file's content now. Between `7f4505a` and `579dbb7`, main changed only `agents/registry/parameter_registry.py` (part 1, identical to the worktree's file), `CLAUDE.md`, `docs/LEADS.md` and the X-LEADS-TIDY exchange files. Every other file named below has the same blob at both commits.

## Prior record

**Searches run** (worktree at `7f4505a` and main at `579dbb7`, 2026-11-25; `/bin/grep`, because the shell `grep` skips `.json`):

```
S = docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_* docs/TB_RULING_REQUEST_* docs/PROPOSAL_* docs/CHAPTER_*
    docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md docs/exchange/
/bin/grep -rlE 'parameter_registry|PARAMETER_REGISTRIES|ParameterSpec|registry repair|REGISTRY_REPAIR|switch registration|
    schema[-_]check|per[-_]seed[-_]sieve[-_]features|fallback[-_]metadata[-_]merge|include[-_]feature[-_]groups|
    exclude[-_]feature[-_]groups|allow[-_]target[-_]columns' $S
/bin/grep -nE 'registry|ParameterSpec|schema.check|per.seed|LL-[123]\b|manifest' docs/LEADS.md     # at 579dbb7: no hit on this subject
/bin/grep -rlE 'registry|ParameterSpec|switch regist|manifest entr' ~/distributed_prng_analysis/logs/e2e_session/
/bin/grep -rnE 'file_exists|_handle_retry|suggested_param_adjustments|G-UNPINNED-IDENTICAL' docs/
git diff --stat 7f4505a 579dbb7
```

**Documents found that decide, authorise, schedule or record the subject:**
- **Rulings:**
  - `logs/e2e_session/OWNER_RULINGS_pending_list_20261124.txt`, ruling 7 at 10:10: "#7 yes fix those but document everything";
  - `OWNER_RULING_switch_registration_order_20261125.txt` (repair, then registrations, then manifest entries; grammar later);
  - `OWNER_DIRECTION_autonomy_switches_20261123.txt` (the switches live in the existing registry and manifests);
  - `TASK_REG_BETA_SEND_20261125.txt` (part 1 landed; parts 2 and 3 to you first).
- **The finding part 1 carries out:** `docs/CHAPTER_10_AUDIT.md` B1 `:117-140`; S1 `:145-155`; `docs/CHAPTER_CORRECTIONS_REPORT_20261111.md:213` (R-30).
- **The drafts:**
  - `docs/drafts/loop_v1/REGISTRY_REPAIR_20261124.md` with `registry_repair_20261124/`;
  - `SWITCH_CHECK_20261123.md` with `switch_patches_20261123/`;
  - `AUTONOMY_SWITCH_MAP_20261123.md` (G4 to G8, G16).
- **The LL thread:** `docs/exchange/to_beta/X-LL-SWITCHES_R1_ALPHA_REVIEW_REQUEST_ll_switch_landing.md` (known limits 1 and 2) and `docs/exchange/from_beta/X-LL-SWITCHES_R1_BETA_REVIEW.md`; `docs/drafts/loop_v1/READY_LAND_LL_SWITCHES_20261125.md` §(b) `:536-560` and §(c) (LL-1 `fail` against the unlearned arm).
- **WATCHER facts relied on:** `docs/CHAPTER_12_AUDIT.md:28-29` (every manifest is `file_exists`) and B5 `:91-93` (RETRY logs, does not apply); `CHAPTER_14_AUDIT.md:248` (the Step 5 step-scoped filter, a ruled boundary; `--dropout` not registered).

**Labels:**

| Item | Label |
|---|---|
| Part 1, the repair (20 dropped, 0 corrected; 29 → 9 specs) | ALREADY RECORDED (B1, ruling 7); **landed** `579dbb7` |
| Part 2, the 7 LL registrations on the repaired file | ALREADY RECORDED as the plan (`READY_LAND_LL_SWITCHES` §(b)); NEW as files |
| `switch_check.py` R10 (`--register`) | NEW |
| Part 3, the two manifest commands, and the argv probe | NEW |
| The suites red on clean main (`NOTE_REGISTRY_SUITES_FAILING_ON_MAIN_20261125.md`) | NEW as a list; causes ALREADY RECORDED or explained there |
| The Step 1 heuristic suggests the abort-only `--forward-threshold`/`--reverse-threshold` | NEW, dormant (same note, §1) |
| The LL-2 WATCHER route (`DESIGN_LL2_WATCHER_ROUTE_20261125.md`) | design only; not under review; given for context on Q2 |
| CHANGES A RULING | none |

**Not searched:** the host outside the repository, ser8 archives, the private repository, project files outside the repository, other worktrees except step2-repair (read for the LL-2 design note).

## Files under review

Paths are relative to `~/tfm_worktrees/registry-repair/`.

| # | Lands at | Part | On `579dbb7` | Read it at | sha256 now |
|---|---|---|---|---|---|
| 1 | `agents/registry/parameter_registry.py`, repaired | 1 (**landed**) | on main, identical | `agents/registry/parameter_registry.py` (or main) | `72d0031569d42ed5c25d0cf5eaa7ef81a0e1827fe7c3728396d5c9e63956d341` |
| 2 | `agents/registry/parameter_registry.py`, **registered** (copied over #1 at landing) | 2 | M | `docs/drafts/loop_v1/registry_repair_land_20261125/register/parameter_registry.py.patched` | `bea4fcfe81a1b1cccacbf979f46e0d17aebd58aaecb9411ea8c283fe7920611b` |
| 3 | `…/register/parameter_registry.py.patch` (diff #1 → #2: 61 lines added, 0 removed) | 2 | N | same path | `d2f129007fe2a9b504dba7d1f7976ba62e64d329b8b7ad13f2074e2ac0d68889` |
| 4 | `…/repair/parameter_registry.py.repaired` (= #1) | 2 (record) | N | same path | `72d0031569d42ed5c25d0cf5eaa7ef81a0e1827fe7c3728396d5c9e63956d341` |
| 5 | `…/repair/parameter_registry.py.repair.patch` (diff `7f4505a` → #1) | 2 (record) | N | same path | `6d4870eb666c024da9836c7161bb7db725382709d17e4c00d470e9f3df0b9d5c` |
| 6 | `…/repair/repair_record.json` (force-add) | 2 (record) | N | same path | `592573e16711791cccd887f35f864569acf040b36353bc2852178f28dc0de4f7` |
| 7 | `docs/drafts/loop_v1/REGISTRY_REPAIR_DECISIONS_20261124.json` (the 20 decisions) | — | on main, identical | same path | `9cdffbb53d152971e1e04d8e7c0747bcc349cc87a8b70dcafe0b22c1cce02f6b` |
| 8 | `…/REGISTER_LL_20261125.json` (the 7 options named; force-add) | 2 | N | same path | `abf2a7f2c00de801b8a75619e2f6f71f3e1ca590be9d12656b1dea61874736c5` |
| 9 | `…/register/manifest_fragments.json` (force-add) | 2 | N | same path | `2651c0942ac11a6a02604bdb562ac94bdd2f41b39b79267e688ee77274085c16` |
| 10 | `…/OWNER_MANIFEST_COMMANDS.txt` (the owner runs these; `agent_manifests/` is hand-landed) | 3 | N | same path | `990fabb33cfdeaff0c73f5d641b7150f06e858362112be91630039659cec4a5c` |
| 11 | `scripts/diagnostics/switch_check.py` (R10) | 2 | M | same path | `7a9b35f0a0007289fd2289bb80d021649c63635c33489e4b683a0519d010a08e` |
| 12 | `tests/test_switch_check.py` (+2 R10 gates) | 2 | M | same path | `4c13e74cdeab2a2e8ea7a958907d46a74613e5272a2df89e31daa1f77ba75588` |
| 13 | `tests/test_registry_ll_register.py` (7 gates) | 2 | N | same path | `fa618f2f124a3e3b1a8afc3b08054e39eca5ec868eaac17c6c1bb66be5f796ae` |
| 14 | `tests/mutate_e2e_grader.py` (SM24-SM30, LR1-LR9) | 2 | M | same path | `20743639504909c3f739ce08f4132cbaf42f56cfe7487e24c0591018f386b395` |
| 15 | `…/switch_check_7f4505a.md` (the generated check) | 2 | N | same path | `8052e306f92e9f3d6ef17297883447410ac6f83e00228f3134d2c320b62716e8` |
| 16 | `docs/drafts/loop_v1/READY_REGISTRY_REPAIR_LAND_20261125.md` (the READY note) | 2 | N | same path | `3b78e11ca2a50b66f4a6717d2f99df32d788954213ed5a4ec326faffda0fe41e` |
| 17 | `docs/drafts/loop_v1/NOTE_REGISTRY_SUITES_FAILING_ON_MAIN_20261125.md` | 2 | N | same path | `2469f9c71c43aa3058184c737f254fb0d2128c911bc1c5b4e7a2cf81b2166bc6` |

`…` is `docs/drafts/loop_v1/registry_repair_land_20261125/`.

**For context, not under review:** `docs/drafts/loop_v1/DESIGN_LL2_WATCHER_ROUTE_20261125.md`, sha256 `c6d276db31e4087464f6aa9893b326585c0ad950ef8f67fbd2df4fad412b7a73`. It is the design for giving LL-2 a WATCHER route. No code was written.

**Evidence, not for landing** (`data/` and `logs/` are ignored):

| File | sha256 |
|---|---|
| `data/registry_repair_land/argv_probe.py` | `bebae9ef0b785511a6a13b34a3f35cab5cab6cbf8ffb63364327d52bd8c179b0` |
| `logs/registry_repair_land/D_manifests/argv_before.json` | `8e639212b2869b3800cdaf64f403720adbe7365f072a67e9464af14a5a9443c2` |
| `logs/registry_repair_land/D_manifests/argv_after.json` | `d121e9b010708a49831bd5ec5e031ba5844dfbef5f8d41a46ee15e751bb71fdb` |
| `logs/registry_repair_land/D_manifests/prediction.json.after` | `fb76fce730d47cd72c667c4d7d5dd96fc4db7093aaa1f26426918d03c409c641` |
| `logs/registry_repair_land/D_manifests/reinforcement.json.after` | `7536fcb7d33bf18f08c9cbf77873cd831a580404fb9caed1a0b807bd05869a18` |
| `logs/registry_repair_land/{A_baseline,A2_baseline_with_dataset,B_part1_repair,C_part2_register,D_manifests,E_final_checker}/` | one log per suite, plus `SUMMARY.txt` |
| `logs/registry_repair_land/mutants_all_wt_final.log` | |

**Where things sit in #2:**

| Option | Line in #2 |
|---|---|
| LL-3, the four `meta_prediction_optimizer_anti_overfit.py` specs | `:385`, `:392`, `:399`, `:406` |
| LL-1 `schema_check` | `:435` |
| LL-2, the new registry `full_scoring_worker.py` | `:444`; specs at `:449`, `:458` |

The 20 drop comments are at `:272-275`, `:326`, `:331`, `:340-342`, `:380-383` and `:403-404` of #1.

**About running anything:** a write into this repository or the worktree during your review is a `BOUNDARY VIOLATION`; `AGENTS.md` §3 allows `/tmp`. Please say what you ran and what you only read.

## What was run (VM 101, CPU only, `taskset -c 4-7 nice -n 10`, `python3 -u -B`, suites one at a time)

These were fresh detached worktrees of `7f4505a` with only the landing files copied in, both now removed. **No commit anywhere.** `7f4505a` and `579dbb7` differ in no file these suites read except #1, which stage B already used.

| Stage | State | Result |
|---|---|---|
| A, A2 | clean `7f4505a` | baseline. 17 suites, 6 red, the reasons in #17 |
| B | + #1 (part 1) | every verdict as baseline; `test_subphase1` "Registry has 5 parameters" (was 9) |
| C | + #2 and the checker and tests | every verdict as baseline; #12 14/14, #13 7/7 on the draft and on the landed file |
| D | + the two commands of #10 | every verdict as baseline. Argv probe: Step 5 and Step 6 argv with no params **identical** before and after. With a value set, it is dropped before and reaches argv after (`--exclude-feature-groups dead`, `--allow-target-columns-as-features`, `--schema-check warn`) |
| E | final checker | #12 14/14, #13 7/7, `test_registry_repair` 5/5 |
| worktree | all mutant targets | 104/104 killed and credited |

## Known limits and corrections (stated here so you do not have to find them)

1. **CAUTION: after part 3, WATCHER can pass `schema_check=fail` to Step 6.** Nothing sets it today; the declared value is `null`, which WATCHER omits. But an operator `params` value of `fail` would now reach `prediction_generator.py --schema-check fail`. Against a model whose sidecar has no `feature_schema`, Step 6 would then stop: `models/feature_manifest.py:240-300` raises in `fail` mode, as it does for the unlearned arm A0 in the walk-forward (`READY_LAND_LL_SWITCHES` §(c)). That is a fail-closed stop, not a wrong pool. Q4 asks about it.
2. **LL-2 is registered but not reachable.** The Step 3 wrapper and job generator carry neither flag, and the wrapper drops unknown options (`run_step3_full_scoring.sh:120-123`). No manifest fragment is proposed for `full_scoring.json`. The route is designed, not built (context file above).
3. **None of the 7 is visible to the LLM.** `ParameterContext.build` reads only the primary action's `args_map` (`agents/parameters/parameter_context.py:177-186`), and #10 adds none. Even so, the LLM evaluation path is not reached: every manifest is `file_exists`. The Step 5 LLM retry whitelist (`agents/watcher_agent.py:2629-2636`) names none of the 7.
4. **`validate_value` cannot check the three LL-3 strings** (`ParamType.STRING` has no rule, `parameter_registry.py:88-139`). The script validates them.
5. **The 9 kept specs are unchanged in part 1, including 4 that disagree with the code.** The worst is `adaptive_meta_optimizer.py` `mode`, whose choices `quick/full/deep` against the code's `full/survivor/network/epochs` pass `quick`. This is outside ruling 7 and is put to the owner.
6. **`get_registry_for_agent` still maps Step 2 and 3 agent names to the removed keys**, so it returns `None` for them. It has no production caller.
7. **CORRECTION, two red gates were mis-named in my READY note** (now fixed in #16; details in #17 §5 and §6):
   - the Run 4 routing red is **`G-UNPINNED-IDENTICAL` INCOMPLETE** (frozen fixture's stub DB lacks `select_step1_search`). The `G-ALLOWLIST-EXACT` lines I first cited are inside mutant runs;
   - the P0.5 reds are **Gates 24 and 31**, because my check tree lacked the gitignored `daily3.json`. It was not "the live-fleet gate, VIR-5", which passes.
   No verdict comparison changes.
8. **Other corrections made since the first draft:**
   - the wrapper's unknown-option drop is at `run_step3_full_scoring.sh:120-123`; I had cited `:34-37`, a line number inside an excerpt;
   - two sha256s in the READY note's landing table were of pre-`render_md` outputs. #15 is now `8052e306…`, and the full JSON is `2bc3b421…`.
   All are fixed in the files above, and the sha256s in this request are re-checked against disk.
9. **The argv probe stubs** freshness, dataset freeze, fleet, execution set, preflight and dispatch, following `tests/fixtures/run4_routing_clean_control.py`. The command builder itself is not stubbed. It is not a run.
10. **The mutant runner prepends a prelude** to the file it mutates, so a mutant of a line "outside `PARAMETER_REGISTRIES`" would be killed by the prelude alone. No such LR mutant was added. RR4 has that weakness on main already.

## Questions

**Q1. The 20 drops (part 1, landed, for the record).** Should any of the 20 be CORRECTED rather than DROPPED, because its script has a flag with the same meaning? The closest candidates:
- row 5, `n_trials` against `generate_scorer_jobs.py --trials` (`:76`), reached only through the wrapper's positional `TRIALS`;
- row 10, `batch_size` against `full_scoring_worker.py --batch-size` (`:557`), which means progress-report size;
- rows 16 and 17, the `dropout_min`/`dropout_max` range against the single override `--dropout` (`:3120`).

**Q2. The 7 specs (part 2).** Is each spec right as an LLM-facing statement of what the code accepts? In particular:
- `schema_check` carries no registry default. The code default is `None`, meaning "the config's value, else off" (`prediction_generator.py:999-1001`).
- LL-2 is registered under a new `full_scoring_worker.py` key while no WATCHER route exists. Does listing it mislead any consumer today?

**Q3. Default unchanged after part 3.** Apart from argv, is there any consumer of the `default_params` key set of `prediction.json` or `reinforcement.json` whose behaviour changes at default? Candidates:
- the Step 5 retry report's interface read (`agents/watcher_agent.py:2739-2766`);
- the step-scoped filter (`:2026-2034`);
- `agents/manifest/agent_manifest.py`;
- any gate that hashes or counts manifest keys.

**Q4. `schema_check=fail` becomes reachable (known limit 1).** Is it a `LIVE` objection to declaring `schema_check` in `prediction.json` before LL-1's behaviour against models with no `feature_schema` is decided? Or is a fail-closed stop of Step 6 acceptable?

**Q5. R10 and the tests (part 2).** Do `--register` and the tests give real protection? Specifically: refusing an unknown, duplicate or already-registered switch; building on the repaired file only with `--register`, so R8's old output is unchanged; `A-LL-ADDONLY`, `A-LL-CODE` and `A-LL-MANIFEST` in #13. Or can a wrong registration pass them?

Please label every objection `LIVE` or `THEORETICAL` as `AGENTS.md` §5 requires.
