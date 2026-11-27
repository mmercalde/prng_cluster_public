# X-LL7-LL8 · R1 · Team Beta review · 2026-11-26

## Prior record

- I ran `git grep -l -I -E 'LL-7|LL-8|attribution_cycle|A-NORM|G-Q4|scaler_source|compare_pool_tiers|TODO-WHY-1'` over the project catalog, rulings, proposals, chapters and audits, consolidated TODO, backlog, leads, exchange, system map, `CLAUDE.md` and `AGENTS.md` at `087bc36`. I used `rg -n -i` over the same record and the loop drafts for result-file collisions, scaler relocation, checkpoint and ranking provenance, and process state. I read the two READY notes and manifests, the external plan §§2–8, the named owner logs, the earlier LL-6 exchange, L-10 and the branch's block 0 record. I checked the local development archive used for block 0. The rigs, Proxmox, ser8, private remote and live run state were outside my search.
- **ALREADY RECORDED** — D1 assigns the tier fix to LL-7, and plan §3.2 assigns the positional refusal to the same unit. D2 defers ranked-record binding and real canonical Chapter 13 activation to LL-9. D3 requires the LL-8 checkpoint pin. The relocated scaler mismatch, LL-8's internal A-NORM scope and the import-time CUDA assignment are recorded. TODO-WHY-1's hand-run stage and the later child-process runner are recorded in L-10, the plan §§6.3 and 6.7, and the 07:46 owner ruling on the queue. `logs/e2e_session/OWNER_RULING_ll78_plan_20261126.txt:2-5`, `docs/exchange/from_beta/X-LL6-BUNDLE_R1_BETA_REVIEW.md:48-65`, `docs/CHAPTER_14_AUDIT.md:141`, `docs/LEADS.md:881-897`.
- **Finding 1 — ALREADY RECORDED governing material omitted from the request's prior record.** Cite the owner's learning-loop direction, the lead's LL-6 → LL-7 → LL-8 order and the 15:21 instruction that calls for this review and a fresh-copy proof before the owner decides. For Q1 and Q3, the system map's §4.2 A-NORM gate, §4.5 ranked-record contract and P0-C govern alongside its tier-comparison line. Chapter 13 audit B15 records the activation dependency. The earlier hand-run instruction explicitly called for A-NORM on the real models, and the branch READY records the archived CPU scores and the check against graded pools. That is the precedent for judging whether a report belongs to a graded block. `logs/e2e_session/OWNER_DIRECTION_learning_loop_team_20261123.txt:2-6`, `logs/e2e_session/TASK_LOOP_ADDENDUM_3.txt:14`, `logs/e2e_session/OWNER_RULING_next_work_20261126.txt:18`, `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:99-117` and `:255-256`, `docs/CHAPTER_13_AUDIT.md:149-156`, `logs/e2e_session/TASK_LEARN_WHY_BLOCK0_20261124.txt:3-5`, `ch14-learn:docs/drafts/loop_v1/READY_WHY_BLOCK0_20261124.md:97-129`.
- **NEW** — The post-scoring relocation reproduction in Q3, the result-file overwrite and publication error in Q5, the test gaps in Q6, and the additional caller-state effects in Q7. **CHANGES A RULING** — none.

I reviewed detached base `087bc366fb199dbbe26c1fe918a911158618d358` in `~/.local/state/api-agents/worktrees/20261126-104153-build-ll7-ll8-087bc36`. The five files matched both manifests and the saved test-chain hashes. Main was `95fa8fa85a1d6c62eaf9323504550334caf741fe`. The base-to-main diff contains exchange records and `tests/test_benching_switch.py`.

`git status --porcelain` in the review worktree on initial read

```text
 M chapter_13_orchestrator.py
 M per_survivor_attribution.py
?? data/
?? docs/drafts/loop_v1/MANIFEST_LL7_20261126.txt
?? docs/drafts/loop_v1/MANIFEST_LL8_20261126.txt
?? docs/drafts/loop_v1/READY_LL7_SIGNED_ATTRIBUTION_20261126.md
?? docs/drafts/loop_v1/READY_LL8_ATTRIBUTION_CYCLE_20261126.md
?? docs/drafts/loop_v1/X-LL7-LL8_R1_ALPHA_REVIEW_REQUEST_DRAFT.md
?? scripts/diagnostics/attribution_cycle.py
?? tests/test_ll7_signed_attribution.py
?? tests/test_ll8_attribution_cycle.py
```

Main reported only `?? docs/exchange/to_beta/X-LL7-LL8_R1_ALPHA_REVIEW_REQUEST_ll7_ll8.md` on initial read.

## Positions

| Question | Position | Severity |
|---|---|---|
| Q1 · LL-7 canonical tiers and positional refusal | NO OBJECTION | — |
| Q2 · LL-7 defaults | NO OBJECTION | — |
| Q3 · LL-8 scaler gate on archived blocks | OBJECTION | LIVE — Moving an NN checkpoint after scoring can make the hand-run report attribute a different ranking while claiming to reproduce the graded run. |
| Q4 · LL-8 checkpoint pin | NO OBJECTION | — |
| Q5 · LL-8 runner and result file | OBJECTION | LIVE — A valid CLI result path can overwrite the report while the command exits 0 and returns the report's old hash. |
| Q6 · Tests after landing | SUGGESTION | — |
| Q7 · Import and caller state | SUGGESTION | — |

## Q1 · Canonical tiers and positional refusal

The two root-cause attribution loops and the regime-shift tiers now share the by-name vector and scaler. A positional row fails at the same adapter even at matching width. In a CPU fixture with a forced regime-shift classification, the base did not write a tier file for the name-keyed rows and LL-7 did. The largest tier share difference between LL-7's canonical tier and the raw-input legacy tier was `0.3473`. A positional replay returned no canonical analysis or tier file. Alpha's Q3 fixture covers the inner handoff from `post_draw_root_cause_analysis` to tiers. The outer model-loader handoff is unchanged by this patch. `chapter_13_orchestrator.py:89-100`, `:344-367`, `:742-776`, `:840-859`, `:878-888`, `per_survivor_attribution.py:643-652`, `tests/test_ll7_signed_attribution.py:353-415` and `:457-469`.

This resolves LL-6 Q3 and Finding 2(a) within LL-7. A name-keyed file can still belong to a different checkpoint or schema, per Finding 2(b) and D2. The current reader checks draw ID when both sides give one, checks a sample for `features`, and hands the rows to the analysis. The LL-9 activation gate still applies. A real end-to-end canonical Chapter 13 draw replay is **UNAVAILABLE** as evidence for this review. `chapter_13_orchestrator.py:948-988`, `models/inference_bundle.py:186-195`, `docs/CHAPTER_13_AUDIT.md:149-156`.

## Q2 · Defaults

I found no missed default dispatch. The constructor and CLI both select `legacy`. With no bundle the tier call supplies no adapter, and the function still converts the original features to `float32`. My forced-regime NN fixture produced equal legacy diagnoses and tier values on the base and LL-7, with metadata omitted. Alpha's legacy gate also uses LightGBM and excludes tier metadata and the root-cause timestamp and path. Its Step 6 output gate compares stable fields for NN and XGBoost, and its separate import probe checks whether Step 6 loads either edited module. Those are functional comparisons. The optional LLM call is a recorded Chapter 13 B15 status. `chapter_13_orchestrator.py:128-139`, `:878-912`, `:1082-1086`, `per_survivor_attribution.py:643-652`, `tests/test_ll7_signed_attribution.py:310-321`, `:416-435` and `:470-517`, `docs/CHAPTER_13_AUDIT.md:156`.

## Q3 · G-Q4-SCALER on an archived block

The submitted gate covers LL-6 Q4 when the written path was missing **during Step 6 scoring** and is still missing during LL-8. The test runs Step 6 after relocation, so it proves that case. LL-8 checks the path only at report time, then the `as_written` escape takes the current unscaled route and states that it reproduces Step 6. Its internal A-NORM compares two forwards using the same replay input. `prediction_generator.py:349-370` and `:783-807`, `models/inference_bundle.py:75-78`, `:128-137` and `:251-264`, `scripts/diagnostics/attribution_cycle.py:301-317` and `:351-365`, `tests/test_ll8_attribution_cycle.py:473-503`.

I exercised the archive sequence in the reverse order. Step 6 scored 60 name-keyed fixture rows with the scaler before I moved the checkpoint. The archived bytes still matched the required pin. LL-8 refused `resolved`, then accepted `as_written` and labelled the result a reproduction. The maximum gap from the scores Step 6 had made was `8.0090`, the two top-20 sets shared 14 members, and LL-8's internal A-NORM was `0`. This can change which survivors enter the report's bands and contrasts. The measured effect is on the explanation attached to the graded block. A hit-rate effect from following its advice would require the usual equal-`k` walk-forward comparison against the active arm and `k/1000`. `scripts/diagnostics/attribution_cycle.py:310-317`, `:351-365`, `:373-381` and `:409-415`.

I also checked the actual block 0 NN archive. The written checkpoint path currently exists and its bytes match the frozen copy. With the pin set to the frozen SHA, this LL-8 completed on all 12,156 rows in `15.26 s`, with reported peak RSS `983.3 MiB`. The replay's row seeds and raw ranking scores matched the saved CPU Step 6 rebuild exactly, including the order of its top 300. The signed attribution forward differed from those saved scores by at most `1.49e-8`. This is a passing control for that saved block under its current path state. The earlier READY documents the two within-tie GPU/CPU slot swaps. `ch14-learn:docs/drafts/loop_v1/READY_WHY_BLOCK0_20261124.md:97-129`, `scripts/diagnostics/attribution_cycle.py:351-365` and `:447-464`.

For an archived report, require evidence of the original transform and ranking. When the original scored with the scaler, the pinned resolved copy can reproduce it after relocation. Use `as_written` as a historical replay only when saved provenance shows Step 6 scored without the scaler. In the walk-forward sidecar arm, the vote harness already records the scaler state and refuses an NN without its scaler. It saves score and input hashes that the miss-report rebuild checks. The saved CPU scores and ranked membership are another usable check where available. Add a fixture that scores first, relocates second and refuses the wrong historical transform despite a matching checkpoint pin and green internal A-NORM. `scripts/diagnostics/e2e_vote_harness.py:652-688`, `:691-718` and `:844-856`, `scripts/diagnostics/e2e_miss_report.py:402-437`, `scripts/diagnostics/e2e_vote_npz.py:53-66`.

## Q4 · Required pin and wrong-checkpoint refusal

The pin requires 64 lowercase hex characters and is compared against the actual resolved checkpoint's SHA before attribution. I independently got `refused` for a missing pin, a short pin and a different same-width NN checkpoint under the original pin. The right-pin control returned `ok`. For the hand run, take the expected pin from the graded run's record or frozen copy. This check establishes checkpoint identity for stable archival files. The original scorer transform and ranked order require the evidence in Q3. `scripts/diagnostics/attribution_cycle.py:290-306` and `:449-456`, `models/inference_bundle.py:244-264`, `tests/test_ll8_attribution_cycle.py:453-472`.

## Q5 · Runner interface and result publication

The normal refusal path works. With an existing output directory and no explicit result file, the CLI exited 2, printed the refusal on stderr and left the existing directory alone. A fresh separate result path returned exit 2 and a `refused` JSON record. The family and flag ID are sufficient for routing and recording the requested family. Flag lookup and attachment belong to W under plan §6.7. `scripts/diagnostics/attribution_cycle.py:213-245`, `:304-306`, `:497-533`, `tests/test_ll8_attribution_cycle.py:519-555`.

Two publication problems are reachable today.

- With `result_file=<new-out>/attribution_cycle.json`, the early existence check passes, the cycle writes and hashes the report, and the final write replaces it with the result JSON. My fixture returned `ok` and 0 with a report hash that failed verification. The same code path permits aliases of the other cycle artifacts. Constrain the result destination to a distinct path and publish it without overwriting an existing file. Verify the output hashes after publication in the success test. `scripts/diagnostics/attribution_cycle.py:440-474`, `:494-505` and `:518-533`.
- With a regular file where the result path expects a parent directory, the refusal was printed, then the function raised `FileExistsError` during publication and the CLI exited 1 with a traceback. Output and result path resolution also precede the protected section. Handle paths and result publication in the structured error path. On a publication failure, return `error`/3 and `result_written: null` with the publication error on stderr and retain the original reason if there was one. Plan §6.7's child-process caller can surface a failed attempt from the exit code and captured stderr even when a result file could not be published. `scripts/diagnostics/attribution_cycle.py:487-517` and `:518-537`.

## Q6 · Tests after landing

Both suites record the baseline SHA, and their source-mutant runners require a green unmutated target, a red mutated target and a marker where declared. I checked the manifests against the saved repeat logs and counted 20 green runs in each. Add happy-path and failure controls for Q3, Q5 and Q7 to the fresh-copy proof. Two narrower harness gaps are worth fixing there. `tests/test_ll7_signed_attribution.py:141-167`, `:190-230`, `tests/test_ll8_attribution_cycle.py:120-147` and `:254-283`.

- LL-8's absence precondition treats every failed `git show` as absence. A nonexistent 40-hex baseline returns the same `None` as a valid commit without LL-8. Verify that the SHA resolves to a commit before testing path absence. Use the actual LL-7 landing SHA for LL-8's fresh proof, as the plan §§1 and 7 specify. The LL-7 Step 6 comparator should require both baseline runs to succeed before using differences between them to identify volatile fields. `tests/test_ll6_inference_bundle.py:75-77`, `tests/test_ll8_attribution_cycle.py:254-283`, `tests/test_ll7_signed_attribution.py:500-511`.
- The in-process abstention mutant gets credit if it runs and the report is absent. Require a completed report that fails the sparse-cohort predicate after the hook ran. The boundary negative controls do catch a retained sibling write. Their snapshots can miss transient writes and edits that leave an already-dirty Git status unchanged. The LL-8 CLI test also takes both settings hashes after its children have run. Take the first hash before starting the children and compare it after they exit. This directly tests the byte-level settings claim. `tests/test_ll8_attribution_cycle.py:534-566` and `:589-623`, `tests/test_ll7_signed_attribution.py:524-542`, `tests/test_ll6_inference_bundle.py:103-141`, `docs/exchange/from_beta/X-LL6-BUNDLE_R1_BETA_REVIEW.md:59`.

The `test_ch14_diag_land_defaults.py` child-call discrepancy remains a reported harness discrepancy. The submitted record includes a green lead replay and a red build replay on clean code. A matched rerun is **UNAVAILABLE** as independent Beta evidence here. See `docs/drafts/loop_v1/READY_LL7_SIGNED_ATTRIBUTION_20261126.md:132` and the request's run evidence.

## Q7 · Import and caller state

Moving the CUDA assignment off module import fixes the reported import side effect if the scope restores the caller's exact original value, including absence, and covers the pinned miss-report import. That dependency also clears CUDA at its own module top. I tested Alpha's proposed move on a scratch copy with the real pinned dependency. The sentinel and `tempfile.tempdir` were restored after success. On submitted code, importing LL-8 cleared the sentinel immediately, and a refused call cleared a sentinel set after import. `scripts/diagnostics/attribution_cycle.py:63-81`, `:99-106`, `:248-287` and `:506-509`, `scripts/diagnostics/e2e_miss_report.py:67-82`.

The scratch variant still added a permanent `sys.path` entry and installed a root logging handler at WARNING in a process that started with no handlers. The NN load also advanced the process's Torch CPU RNG while constructing the module before loading its weights. `_EnvRoots.__enter__` creates directories and mutates the environment in the same loop, so an error creating a later directory can leave earlier keys changed before `__exit__` can run. Create the roots before changing environment state and test import, success, refusal and setup failure with a sentinel. Plan §§6.3 and 6.7 already specify a child process for W, which isolates that caller from these effects and from library state initialized during inference. `scripts/diagnostics/attribution_cycle.py:256-287` and `:506-509`, `models/wrappers/neural_net_wrapper.py:62-75` and `:422-433`.

## Checks run and evidence read

On VM 101 in the Torch venv I ran targeted CPU checks from an isolated copy at `/tmp/beta-ll78.qXfajz/`. They covered canonical and legacy NN tier behavior, relocation after Step 6 scoring, pin refusals and a right-pin control, result-file aliasing and publication failure, caller state with the submitted and scratch-scoped code, and the block 0 NN replay with saved scores. I read the source of both new suites and the READY notes and repeat logs. The full repeat chains and ancillary-suite reruns are Alpha's evidence. My block 0 replay read the existing development artifacts and wrote its report in scratch. It measured attribution parity on that block and supplies no hit-rate verdict on TFM's learning loop.
