## Prior record

**X-E2E-WF-VOTE · R1 · Team Beta · 2026-11-25 PST**  
Review class **BEFORE-COMMIT, courtesy**. Base read `fe5ac3bef0165e5017b25eecd428f4defa2e31dc`. Michael decides the landing.

`git status --porcelain` at initial read, 08:31 PST:

```text
?? docs/exchange/to_beta/X-E2E-WF-VOTE_R1_ALPHA_REVIEW_REQUEST_variant_b_landing.md
```

**Finding 1 — ALREADY RECORDED governing material omitted from the request's prior record.** The owner's Addendum 1 requires the launcher itself to return nonzero, name the switch and file, and write nothing. The lead's Addendum 2 applies the same requirement to direct voting and requires the effective schema setting in its record. The lead's walk-forward answers specify which settings the three variant commands represent. These sources decide the scope of Q1 and Q2. Addendum 3 sets the one-round courtesy review.  
`logs/e2e_session/TASK_LAND_WF_VOTE_20261125_ADDENDUM_1.txt:1-7`, `logs/e2e_session/TASK_LAND_WF_VOTE_20261125_ADDENDUM_2.txt:1-7`, `logs/e2e_session/LEAD_ANSWERS_WF_20261123.txt:1-9`, `logs/e2e_session/TASK_LAND_WF_VOTE_20261125_ADDENDUM_3.txt:1-10`.

- **Searches run.** I repeated `git grep -liE` at `fe5ac3b` across the catalog, `TB_RULING_*`, `TB_RULING_REQUEST_*`, proposals, chapters and audits, consolidated TODO, backlog, leads and exchange for the harness and driver names, walk-forward, LL-1/2/3, manifest, sieve feature and capability terms, thread IDs and grader pin. I also used `rg` and `git grep -n -iE` in `docs/` and `logs/e2e_session/` for the landing task and addenda, `01_sieve_on.sh`, refusal writes, argument overrides and run records.
- **ALREADY RECORDED.** The grader repin, the driver's forbidden-root rule, Run H's tree-local inputs, the external W10/G19 capability fixtures and the scope of Alpha's round-2 suites. The lead's frame treats these runs as development evidence. The owner's November 24 amendment governs eventual promotion using back draws that selfplay did not score itself on. `docs/drafts/e2e_v1/READY_GRADER_DUP_DRAW_20261124.md:115-126`, `e2e-map:docs/drafts/loop_v1/READY_LAND_WF_VOTE_20261125.md:88-111`, `:415-427`, `logs/e2e_session/TASK_LOOP_COMMON.txt:1-4`, `docs/CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md:10-26`, `CLAUDE.md:43-67`.
- **NEW.** Wrapper writes and exit status on refusal, driver handling of abbreviated LL-2 arguments, and Step 5 passthrough overriding inputs and folds. The snapshot vote-record check is new evidence for the recorded verification gap. **CHANGES A RULING** none.
- **Not searched.** Ser8 archives, private remotes, rigs, live ledgers or host material beyond the named worktrees, saved evidence and local test environment.

## Positions

| Question | Position | Severity |
|---|---|---|
| Q1 — Refusal | OBJECTION | LIVE — `01_sieve_on.sh` writes a log and timing file and reports success when its child refuses the missing LL-2 switch. |
| Q2 — Record | OBJECTION | LIVE — A normal Step 5 passthrough can make a target-hits run train on the original `holdout_quality` input while its record reports `holdout_hits` and its input check reports success. |
| Q3 — Additive | NO OBJECTION | — |

## Q1 — Refusal

**The shipped `01_sieve_on.sh` wrapper fails the refusal requirement.** Its shell redirects to the variant log and invokes `/usr/bin/time -o` before the Python check. It backgrounds that command and prints `started` without waiting for the child's status. I ran the exact wrapper from the scratch tree with `STAGE=smoke` and a `taskset` stub that kept the real `nice` and `/usr/bin/time` and substituted a child that printed a refusal and exited 2. The wrapper returned 0, printed `started`, and wrote both `logs/e2e_runH_wf_01_sieve_on_smoke.log` and `data/e2e_runH_wf_01_sieve_on_smoke.time`. The timing file recorded the child's exit 2. The driver's output directory was absent. This is an issue with the supported launcher even in a detached tree with valid inputs. The probe reported by Alpha passed the distinguishing switch straight to Python and had no wrapper around it.  
`e2e-map:docs/drafts/loop_v1/runH_commands/01_sieve_on.sh:10-24`, `:25-40`, `landing evidence/probe_ll.py:39-59`.

**The driver and Step 3 disagree about a CLI spelling the Step 3 parser accepts.** `--step3-extra-args='--per-seed-sieve-f on'` passes driver preflight because the helper matches complete names only. I observed exit 0, `PLAN.md` and `run_record.json` in plan mode. The record says LL-2 is unused and still carries the abbreviated argument in the Step 3 argv. The direct Step 3 CLI accepted that spelling and refused with exit 2, naming LL-2 and `full_scoring_worker.py`. In a full run, the driver has already written its record and runs the copy before forwarding this argument to Step 3. The late failure follows from that order. I exercised preflight and direct refusal, with no full run. A gate should cover the abbreviated form or explicitly reject it before any write.  
`e2e-map:scripts/diagnostics/e2e_pipeline_caps.py:125-143`, `e2e-map:scripts/diagnostics/e2e_step3_harness.py:571-604`, `:617-637`, `e2e-map:scripts/diagnostics/e2e_walk_forward.py:533-546`, `:709-711`, `:855-884`, `:1411-1451`.

**Verified for the fully spelled requests.** With valid pinned fixture inputs, the driver refused active LL-1 in the vote config, both LL-2 settings, and all three LL-3 flags through each Step 5 passthrough. It returned 2, named the relevant pipeline files, and left the proposed output directory absent. The direct vote CLI and the selftest's copied vote CLI each refused `schema_check=fail` with exit 2, named `prediction_generator.py`, and created no output directory. The direct Step 3 CLI also refused the fully spelled LL-2 request before creating its output directory. The paths I checked produced no completed base result under an LL switch label. The wrapper can falsely signal a start and the abbreviated option can leave a misleading plan.  
`e2e-map:scripts/diagnostics/e2e_walk_forward.py:533-562`, `:1411-1436`, `e2e-map:scripts/diagnostics/e2e_step3_harness.py:602-637`, `e2e-vote:scripts/diagnostics/e2e_vote_harness.py:558-592`, `:1133-1159`, `:1439-1444`.

**Recorded status for the main checkout.** Its `data/` is inside the forbidden root. The driver exits at the path check before inspecting the switches there, so it reports the forbidden root without naming the switch and file. The run-capable comparison uses main's bytes in a separate tree. A wrapper refusal gate should exercise the wrapper with a usable tree and artifact paths, assert a nonzero shell return, and check both the run directory and the sibling log and timing files.  
`e2e-map:scripts/diagnostics/e2e_walk_forward.py:77-99`, `:149-154`, `:1411-1436`, `e2e-map:docs/drafts/loop_v1/READY_LAND_WF_VOTE_20261125.md:420-424`.

## Q2 — Record

**A passthrough can change the Step 5 target file after the shim rewrites it.** The driver constructs `--survivors PATH`, then appends `--step5-extra-args` and, for neural net, `--nn-args`. The shim counts literal `--survivors` tokens, so a trailing `--survivors=PATH` survives the check and the rewrite. The optimizer parser accepts the equals form and takes the last value. The driver then validates the first literal `--survivors` and the shim's target declaration. In my CPU parser-only reproduction, the shim reported the derived `holdout_hits` file and L3's source predicates would pass. The unchanged optimizer parser selected the original Step 3 file, whose `holdout_quality` was `0.045`, while the derived file had `2`. The same passthrough ordering let a declared three-fold setting parse as two folds. The raw argv preserves the conflicting settings, while the reported target, fold count and L3 result can misrepresent what ran. Which way the hit rate moves is unmeasured. A regression gate should put distinct target values in the source and derived inputs and assert the file and settings the optimizer actually reads, or assert preflight refusal of conflicting passthrough options.  
`e2e-map:scripts/diagnostics/e2e_walk_forward.py:683-685`, `:929-948`, `:957-971`, `e2e-map:scripts/diagnostics/e2e_step5_shim.py:76-113`, `:124-149`, `:206-230`, `fe5ac3b:meta_prediction_optimizer_anti_overfit.py:610-660`, `:3037-3048`, `:3106`, `:3157-3166`, `:2025-2034`.

**Base, 02 and 03 use the unchanged pipeline in a usable tree.** The Step 3 harness accepts main's scoring signature and omits LL-2 keyword arguments on it. Script 02 adds the shim's `holdout_hits` target, and 03 adds separate ranker arms that use main's feature loader and vote implementation. Alpha's fixed-B probes completed all three on CPU on one origin. The 03 probe used one LightGBM ranker. Round-2 W8 exercised all four ranker objectives individually and two through the driver. A replay of the exact Run H GPU commands is **UNAVAILABLE** for this review.  
`e2e-map:scripts/diagnostics/e2e_step3_harness.py:188-207`, `:513-518`, `:777-785`, `e2e-map:docs/drafts/loop_v1/runH_commands/00_base.sh:24-38`, `e2e-map:docs/drafts/loop_v1/runH_commands/02_target_hits.sh:24-39`, `e2e-map:docs/drafts/loop_v1/runH_commands/03_rankers.sh:24-40`, `e2e-map:scripts/diagnostics/e2e_step5_ranker.py:148-177`, `landing evidence/probe_fixed_B.log:1-5`, `landing evidence/B2wf_e2e_walk_forward_selftest.log:53-101`.

**The vote snapshot records its own effective LL-1 setting.** The retained `kept_r2/wf_selftest_vote_run_record_sample.json` is a driver record by its `kind`, as Alpha reported. I used the submitted `snapshot_vote` function to create a fresh copy in the scratch tree. On a one-draw CPU vote, both the direct and copied harness records carried `requested: {}`, `pipeline_has_switch: false` and `applied_schema_check: off`. The copied harness's checked path pointed into the snapshot. This verifies the record path that Alpha left open.  
`e2e-map:scripts/diagnostics/e2e_walk_forward_selftest.py:113-128`, `e2e-vote:scripts/diagnostics/e2e_vote_harness.py:1211-1215`, `:1345`.

## Q3 — Additive

I checked all 33 file digests against the request and Alpha's round-2 copy record, and checked the READY note digest separately. Comparing the destinations with `fe5ac3b` gave exactly one existing path, `tests/test_step1_screen.py`. Its only differing line is 55, where the test uses the same-tree vote harness. The grader pin on line 54 is identical. The repointed `G-ORDERING` gate passed using the submitted harness digest. The destination list contains no replacement for a pipeline, manifest or governed file. `git check-ignore -v --no-index` found no ignored destination.  
`fe5ac3b:tests/test_step1_screen.py:48-55`, `e2e-vote:tests/test_step1_screen.py:48-55`, `:480-494`, `landing evidence/copied_B_r2.txt:1-33`.

## Verification scope

I used the `~/venvs/torch` environment with CUDA hidden and wrote the scratch fixtures under `/tmp`. I overlaid the requested bytes on a temporary `git archive fe5ac3b` tree. I first observed the driver's `/tmp` refusal, then removed that single entry from the in-memory forbidden list for `--plan-only` checks. The wrapper child was stubbed. The Step 5 shim ran the exact parser body extracted from main's optimizer and stopped before training. I ran the direct refusal and CPU vote probes described above, the repointed Step 1 screen ordering gate, and the miss-report suite (7/7 gates, 5/5 mutants killed). I rechecked all submitted digests before writing this review.

`landing evidence/` refers to `~/tfm_worktrees/e2e-map/data/landing_wfvote_20261125/`. I read the round-1 and round-2 summaries and relevant suite log excerpts, the probe and comparison records, and the runner and copy scripts. The round-2 log hashes match the READY note. The remaining suite results come from Alpha's records, including the seven that Alpha did not rerun in round 2. Fleet, GPU and full Run H replay are **UNAVAILABLE** under Beta's operating boundary. The evidence here concerns refusals, provenance and fixture execution. It supplies no hit-rate result on unseen draws.
