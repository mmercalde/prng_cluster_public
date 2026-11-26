## Prior record

**X-REGISTRY-REPAIR · R1 · Team Beta · 2026-11-25 PST**
Courtesy review of parts 2 and 3, with part 1 reviewed as landed. Main base read was `579dbb7e558da1d90032967ca7b3e28054180e7a`. The submitted worktree was detached at `7f4505aaccebdd284393eb3565e75d09d1c0f3b4`. Michael decides the landing.

Main `git status --porcelain` at 19:07 PST

```text
?? docs/exchange/to_beta/X-REGISTRY-REPAIR_R1_ALPHA_REVIEW_REQUEST_registry_parts2_3.md
```

**Finding 1 — ALREADY RECORDED governing material missing from the request's prior record.** The lead's switch-register addendum explicitly requires the existing registry and manifests, distinguishes registration from reachability, and requires unchanged defaults for this draft. The previous LL review named it, and the READY note lists it among the files read. The Chain C hotfix instruction is the direct source for the Step 5 reporting boundary cited through Chapter 14. It bars adding the six diagnostic allowlist fields and requires truthful reporting at the interface. The four LL-3 declarations are outside that six-field list. B3 in the autonomy TODO schedules future manifest vocabulary extraction.
`logs/e2e_session/TASK_SWITCH_REGISTER_ADDENDUM_1.txt:1-4`, `docs/exchange/from_beta/X-LL-SWITCHES_R1_BETA_REVIEW.md:12`, `docs/drafts/loop_v1/READY_REGISTRY_REPAIR_LAND_20261125.md:31-36` (submitted worktree), `docs/CLAUDE_CODE_INSTRUCTIONS_CHAIN_C_TRUTHFULNESS_HOTFIX.md:33-45,54-57`, `docs/CHAPTER_14_AUDIT.md:247-251`, `docs/TODO_SELFPLAY_AND_LLM_AUTONOMY.md:133-139,158-163`, `agents/watcher_agent.py:2629-2639`.

- **Searches.** I used `rg -ni` and `git grep -ni` over the catalog, rulings and ruling requests, proposals, chapter audits, consolidated TODO, backlog, leads, exchange and LL drafts. Search terms included `ParameterSpec|registry repair|switch_check|R10|default_params|schema.check|manifest` and the LL flag names, plus manifest digest and key counts, duplicate specs and `nargs`. I used `rg --no-ignore` for the named owner and lead records, `git log --all -- scorer_meta_optimizer.py full_scoring.py` for the dropped registry targets, and `git diff --name-status 7f4505a 579dbb7` to verify the base delta.
- **ALREADY RECORDED.** Ruling 7 requires documenting a correction or drop for each stale entry. The 17:45 ruling orders the repair, registration and hand-landed declarations before the grammar work. Chapter 10 B1/S1 and R-30 record the stale specs and Step 1 anchor boundary. The prior LL thread and READY note record the Step 5 resolver fixes, the missing LL-2 WATCHER route and the A0 failure under strict LL-1. The Run 4 null-declaration rule is scoped to the seven Step 1 pins. The six baseline suite verdicts are reported in Alpha's note.
  `logs/e2e_session/OWNER_RULINGS_pending_list_20261124.txt:18`, `logs/e2e_session/OWNER_RULING_switch_registration_order_20261125.txt:2-6`, `docs/CHAPTER_10_AUDIT.md:117-152`, `docs/CHAPTER_CORRECTIONS_REPORT_20261111.md:213`, `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:52-69`, `docs/drafts/loop_v1/READY_LAND_LL_SWITCHES_20261125.md:18,536-574`, `docs/drafts/loop_v1/NOTE_REGISTRY_SUITES_FAILING_ON_MAIN_20261125.md:7-23` (submitted worktree).
- **NEW.** Q5 records a duplicate registration that passes all seven gates, one unsupported argument shape accepted by R10, and the measured R8 artifact differences. **CHANGES A RULING** none.
- **Not searched.** Ser8 archives, private remotes, rigs, live ledgers, or host files beyond the named worktree, evidence and test environment.

## Positions

| Question | Position | Severity |
|---|---|---|
| Q1 — The 20 drops | NO OBJECTION | — |
| Q2 — The seven specs | NO OBJECTION | — |
| Q3 — Defaults after the manifest declarations | NO OBJECTION | — |
| Q4 — Explicit `schema_check=fail` | NO OBJECTION | — |
| Q5 — R10 and the tests | SUGGESTION | — |

## Reasoning

### Q1 — The 20 drops

I found no correction to make under the original script keys. Row 5 has a functional successor for trial count. WATCHER already supplies `trials` in the wrapper's positional interface, and the wrapper forwards it as the generator's required `--trials`. A registration for the generator would use its own script key. The worker's `--batch-size` controls progress reporting. Step 3's job size is the wrapper's `--chunk-size`. Step 5's `--dropout` is a single NN override, while Optuna defines the trial range. Step 1's warm-start arguments pin trial 0. Step 4's actual parser and Step 6's config loader support the drop decisions for those scripts. The focused repair suite verified the 20 absences and field equality for the nine retained specs. Their recorded mismatches remain a status.
`agents/watcher_agent.py:404-423,2363-2376`, `agent_manifests/scorer_meta.json:118-126,136-137`, `run_scorer_meta_optimizer.sh:183-192,226-246`, `generate_scorer_jobs.py:52-83`, `full_scoring_worker.py:317-335,557-558`, `run_step3_full_scoring.sh:104-107,160-165`, `meta_prediction_optimizer_anti_overfit.py:2495,3120-3121,3235`, `window_optimizer.py:1688-1700,1735-1747`, `adaptive_meta_optimizer.py:1101-1116`, `prediction_generator.py:172,984-1007`, `tests/test_registry_repair.py:91-135`, `docs/drafts/loop_v1/REGISTRY_REPAIR_20261124.md:65-86,118-132`.

### Q2 — The seven specs

All seven match the parser's names, types, literal choices and defaults. LL-1's registry default of `None` keeps the config's value, with `off` as the config fallback. The LL-2 entries describe the worker CLI. WATCHER's Step 3 wrapper and job generator lack a route for both flags, and `ParameterContext` reads the primary action, which names the generator. The seven additions supply no new prompt fields on today's manifests. Normal file-based evaluation returns before the LLM evaluator, while the separate Step 5 health retry uses its own policy allowlist.
`docs/drafts/loop_v1/registry_repair_land_20261125/register/parameter_registry.py.patched:384-465` (submitted worktree), `prediction_generator.py:134,197,999-1001,1034-1053`, `full_scoring_worker.py:570-575,681-682`, `agent_manifests/full_scoring.json:23-45`, `run_step3_full_scoring.sh:88-123,159-178`, `generate_step3_scoring_jobs.py:236-267`, `agents/manifest/agent_manifest.py:361-383`, `agents/parameters/parameter_context.py:177-195`, `agents/watcher_agent.py:1138-1145,1195-1238,2629-2639,3182-3195`.

The LL-3 types describe the parser inputs. The resolver adds restrictions on feature names, groups, empty lists and the resulting feature set. The allow flag requires an explicit `exclude_features` list. Alpha's allow-only argv case proves routing, and the resolver would refuse that request before fitting. Use a JSON boolean for the WATCHER value. The registry's generic validator accepts the string `"false"`, but the probe showed WATCHER appends that string as an extra argument to a flag-only argparse option.
`agents/registry/parameter_registry.py:130-139`, `agents/watcher_agent.py:2412-2428`, `meta_prediction_optimizer_anti_overfit.py:1486-1505,3134-3145,3222-3231`, `models/feature_manifest.py:195-226,229-281`, `docs/drafts/loop_v1/READY_LAND_LL_SWITCHES_20261125.md:18,42-48`.

### Q3 — Defaults after the manifest declarations

I ran both owner commands on copies from `579dbb7`. The output bytes matched the packet, and both default argv arrays matched the originals in the WATCHER probe with dispatch stubbed. Pydantic preserved the added keys as extra manifest data, moving Step 5 from 10 to 14 keys and Step 6 from 7 to 8. The rest of each model dump compared equal. Outside argv, WATCHER's debug list of loaded keys grows, and the switch checker's manifest inventory sees new declarations. The step-scoped filter uses the larger set when an explicit override is supplied. Neither the default retry heuristics nor its diagnostic proposal allowlist offers any of the four new Step 5 keys, so the retry status is the same for proposals allowed today. I found no executable gate pinning either key count or either manifest's hash. The catalog contains a static count, Gate 12's governed list omits both manifests, and the Run 4 capture hashes `window_optimizer.json`.
`docs/drafts/loop_v1/registry_repair_land_20261125/OWNER_MANIFEST_COMMANDS.txt:7-11` (submitted worktree), `agents/watcher_agent.py:2011-2034,2412-2428,2629-2639,2661-2679,2751-2809`, `training_health_check.py:638-664`, `agents/manifest/agent_manifest.py:136-137,289-311,398-416`, `scripts/diagnostics/switch_check.py:373-407` (submitted worktree), `docs/PROJECT_FILE_CATALOG.md:853-868`, `scripts/gate12_parity_gate.py:174-185`, `tests/capture_run4_clean_control.py:93`.

### Q4 — Explicit strict checking

An explicit strict request is acceptable as a fail-closed option. I verified that the Step 6 args now carry `--schema-check fail` on request, and that the checker raises for an empty feature-name list and missing hash in strict mode. The same fixture returned a report in `warn` and returned immediately in `off`. Alpha's A0 result is already recorded. Step 6 also has an independent schema requirement, which can refuse before LL-1, depending on the model and config. When the LL-1 check raises, it does so before matrix construction, and WATCHER's explicit failure branch precedes its output-file grading. The declaration is sufficient for owner-selected strict runs. Any use of strict mode as a default or as a comparison requirement would need evidence from the intended arms, including A0.
`agents/watcher_agent.py:1100-1136,2025-2034,2412-2428,2465-2477`, `prediction_generator.py:770-798,999-1001,1052-1053`, `models/feature_manifest.py:297-353`, `docs/drafts/loop_v1/READY_LAND_LL_SWITCHES_20261125.md:562-574`.

### Q5 — R10 and the tests

R10 rejected an unknown flag, an exact duplicate file/flag pair and an already registered switch with exit 2 before writing patch output. Both generation paths reproduced the submitted source, unified diff and fragments byte for byte, using repair plus registration at `7f4505a` and registration alone at the repaired `579dbb7` base. With a temporary R8 worktree addition and both repair flags present, the old and new checker emitted identical registry source and patch without `--register`. The fragment's note for an omitted boolean default changed, and the report gains patch-base metadata. R8's registry patch behavior is preserved in that case.
`scripts/diagnostics/switch_check.py:1017-1029,1232-1292,1367-1372,1493-1515` (submitted worktree), `tests/test_switch_check.py:450-520` (submitted worktree).

- **A-LL-ADDONLY and A-LL-CODE.** I appended a second `schema_check` spec with only `off` allowed. All seven gates passed. The add-only gate collapses duplicate names in its dict, and code parity reads the first matching spec. Reversing the two entries made code parity fail while add-only still passed. Assert one spec per normalized name before comparing fields. The code parity gate checks parser shape and defaults, while LL-3's semantic restrictions live in the resolver.
  `tests/test_registry_ll_register.py:110-124,153-192` (submitted worktree), `agents/registry/parameter_registry.py:200-218`, `models/feature_manifest.py:195-226`.
- **A-LL-MANIFEST.** This tests the fragment and finds the skip rules as strings. It can pass with a different owner command or after a regression that leaves those strings present. The independent command and argv comparison supplied the missing evidence for this packet. Put that comparison in the gate to cover future changes to the actual handoff.
  `tests/test_registry_ll_register.py:228-253` (submitted worktree), `data/registry_repair_land/argv_probe.py:27-69,78-81` (submitted worktree), `agents/watcher_agent.py:2412-2428`.
- **R10 input scope.** With the real `chapter_13_triggers.py --steps` option, R10 returned success and generated an `INT` spec with `[3, 5, 6]` as its default. Its own validator rejected that default. The scanner discards argparse's `nargs` and then assigns a scalar type. Refuse unsupported shapes in R10 until the registry can represent them. The seven submitted flags are scalar or flag-only, and regeneration verified this packet.
  `chapter_13_triggers.py:885-886`, `scripts/diagnostics/switch_check.py:280-309,976-1007,1247-1262` (submitted worktree), `agents/registry/parameter_registry.py:104-109`.

## Verification

All 22 supplied digests matched, covering the 16 hashed landing-table files, the LL-2 design note and five evidence files. With `~/venvs/torch` active I ran the submitted LL suite (**7/7**) and the repair suite (**5/5**). I regenerated and compared the registration artifacts, exercised the three R10 refusals and compared an R8 worktree addition in `/tmp`. Those checker probes redirected the output-path check to the temporary directory and skipped run-record collection. The worktree addition used a temporary source file. I also ran the duplicate-spec and list-argument probes, both exact owner commands on temporary manifest copies, the submitted WATCHER argv probe with its dispatch and preflight stubs, and the Step 6 schema check on a synthetic row.

I read Alpha's baseline and mutation results. The full checker fixture suite was **UNAVAILABLE** under this review's boundaries because its fixture creates a Git commit and the checker CLI rejects `/tmp` outputs. The full mutation run and fleet or pipeline results were **UNAVAILABLE** independently. Source bytes and both worktree states were rechecked before filing.
`tests/test_switch_check.py:193-204,222-235` (submitted worktree), `scripts/diagnostics/switch_check.py:1482-1490` (submitted worktree), `logs/registry_repair_land/mutants_all_wt_final.log:145-166` (submitted worktree), `AGENTS.md:86-108`.
