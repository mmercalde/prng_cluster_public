# X-E2E-WF-VOTE  R1  REVIEW_REQUEST  variant B landing of the vote harness and the walk-forward driver (courtesy review)

- **Review class:** BEFORE-COMMIT, courtesy. The owner will decide on landing variant B after this one review (owner, `logs/e2e_session/TASK_LAND_WF_VOTE_20261125_ADDENDUM_3.txt`, 2026-11-25 08:27). It is not binding; the owner decides.
- **Base commit:** `fe5ac3b` (main).
- **Rounds:** one.
- **Allowed answers per question:** `NO OBJECTION` · `OBJECTION` (with `LIVE` or `THEORETICAL`) · `SUGGESTION` · `NEEDS EVIDENCE`.

**The files under review are NOT on main yet.** Read each one at the worktree path given in the table below. The sha256 is that file's content now.
- Rows 1-13 come from `~/tfm_worktrees/e2e-vote`; rows 14-33 come from `~/tfm_worktrees/e2e-map`.
- Variant B lands them at the repo-relative path in the second column.
- Nothing else changes on main.

## Prior record

**Searches run** (main `fe5ac3b`, 2026-11-25):

```
S = docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_* docs/TB_RULING_REQUEST_* docs/PROPOSAL_* docs/CHAPTER_*
    docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md docs/exchange/
git grep -l -e P fe5ac3b -- $S   for P in: e2e_walk_forward | e2e_vote_harness | 'walk-forward\|walk_forward' |
    'LL-1\|LL-2\|LL-3' | feature_manifest | 'per_seed_sieve_features\|per-seed-sieve' |
    'X-E2E-VOTE\|X-E2E-WF\|X-E2E-MAP' | 'capability check' | 3ab8ce2a
git grep -l 'X-E2E-WF-VOTE\|variant_b_landing' HEAD -- docs/exchange/
```

**Documents found:**
- `docs/exchange/from_beta/X-E2E-GRADER-LANDED_R1_BETA_REVIEW.md:39`: your grader review cites `e2e-vote:scripts/diagnostics/e2e_vote_harness.py` as evidence. It does not review the harness.
- `docs/TODO_CONSOLIDATED.md:225`: an old owner TODO, "Walk-forward simulation"; it does not concern this driver.
- The `LL-*` hits are "WALL-2" mutant text, a false match. No ruling, proposal or ledger row covers these branches. This thread is new.
- Context records: `docs/drafts/e2e_v1/READY_GRADER_LANDING_20261124.md` (fixtures moved to `tests/fixtures/`; the force-add warning at `:111`) and `docs/drafts/e2e_v1/READY_GRADER_DUP_DRAW_20261124.md` (the grader repin, which this landing carries into `e2e-map`).

**Labels:**

| Item | Label |
|---|---|
| Vote harness H6, miss report, walk-forward driver, Step 3/5 harnesses and their tests on main | NEW |
| Pipeline-switch capability check, `e2e_pipeline_caps.py` (owner ADDENDUM_1, lead ADDENDUM_2) | NEW |
| Grader repin `c7fe48bf` → `3ab8ce2a` in these files | ALREADY RECORDED (`READY_GRADER_DUP_DRAW_20261124.md`, the "e2e-map" table) |
| CHANGES A RULING | none |

**Not searched:** the host outside the repository, the ser8 archives, the private repository, project files outside the repository.

## Files under review (variant B, 33 files)

| # | Lands at | Read it at | sha256 now |
|---|---|---|---|
| 1 | `docs/drafts/e2e_v1/READY_H6_VOTE_HARNESS_20261123.md` | `~/tfm_worktrees/e2e-vote/docs/drafts/e2e_v1/READY_H6_VOTE_HARNESS_20261123.md` | `fbc01afc537d5dea77669ec249a67f211b25d4e5c1750999484ddf929607a149` |
| 2 | `docs/drafts/loop_v1/NOTE_VOTE_NPZ_20261124.md` | `~/tfm_worktrees/e2e-vote/docs/drafts/loop_v1/NOTE_VOTE_NPZ_20261124.md` | `7913429cde5288c0fe5e600775948b3100d6d8847ee2200bd1aee7bae7f07aeb` |
| 3 | `docs/drafts/loop_v1/READY_LOOP_MISS_20261123.md` | `~/tfm_worktrees/e2e-vote/docs/drafts/loop_v1/READY_LOOP_MISS_20261123.md` | `07da565698af0aef20a9d3495dedda11c83c4901d456e59f03fb06a347cba062` |
| 4 | `scripts/diagnostics/e2e_miss_report.py` | `~/tfm_worktrees/e2e-vote/scripts/diagnostics/e2e_miss_report.py` | `ec7c7ad16b25362e1800258c7a3667e1f8150b73739e08b43b5522f028116d76` |
| 5 | `scripts/diagnostics/e2e_vote_harness.py` | `~/tfm_worktrees/e2e-vote/scripts/diagnostics/e2e_vote_harness.py` | `ed39d3b00b5984898e3038e7ecd6c217724a8694b63da4f12a6647d8997f2a5f` |
| 6 | `scripts/diagnostics/e2e_vote_npz.py` | `~/tfm_worktrees/e2e-vote/scripts/diagnostics/e2e_vote_npz.py` | `4c2f349980773651751df128b929154508673d2cd14cfe9ae65cd14df6319ead` |
| 7 | `tests/test_e2e_miss_report.py` | `~/tfm_worktrees/e2e-vote/tests/test_e2e_miss_report.py` | `ffc7250b450edd34d9c26454c2b8dbbfdd28cfa60f581d1c1300de7e6339759a` |
| 8 | `tests/test_e2e_vote_harness.py` | `~/tfm_worktrees/e2e-vote/tests/test_e2e_vote_harness.py` | `d8a0074c2e2e96e17a5aef331e2be166c334ae23ac0a50e1aaccf8a3c95f92e3` |
| 9 | `tests/test_e2e_vote_ll5.py` | `~/tfm_worktrees/e2e-vote/tests/test_e2e_vote_ll5.py` | `ab1fd942e4394be36cdeb0e8ead4fa152924131f1aa8b45a58be165a693715b7` |
| 10 | `tests/test_e2e_vote_ordering.py` | `~/tfm_worktrees/e2e-vote/tests/test_e2e_vote_ordering.py` | `aa8a8f4f5a994c3176e29e7d03fb65f4ea7fb2b8b76b3e8392b546023cbd3c5d` |
| 11 | `tests/fixtures/e2e_vote_harness_rev4.py.txt` | `~/tfm_worktrees/e2e-vote/tests/fixtures/e2e_vote_harness_rev4.py.txt` | `99791061cde4c05fcf7e1dd831c05a04bda7f389e4000ce524f58bac55774544` |
| 12 | `tests/fixtures/e2e_vote_harness_rev5.py.txt` | `~/tfm_worktrees/e2e-vote/tests/fixtures/e2e_vote_harness_rev5.py.txt` | `a26e39c1ab4925e997b87d8795db77d2d9c651960c1ee46f67e08e2faffd2fb7` |
| 13 | `tests/test_step1_screen.py` (**modifies** main's file; only line 55 differs from `fe5ac3b`) | `~/tfm_worktrees/e2e-vote/tests/test_step1_screen.py` | `31ae4b3450d2fad94ece5951c088728751d50a30af8e823f3773dcff2e535c0c` |
| 14 | `docs/drafts/e2e_v1/MAP_E2E_THIN_PATH_20261122.md` | `~/tfm_worktrees/e2e-map/docs/drafts/e2e_v1/MAP_E2E_THIN_PATH_20261122.md` | `e771a726300de8d8a37e30e1bde6af4f267366b862674ceddc1442dd16c01842` |
| 15 | `docs/drafts/e2e_v1/READY_H3_STEP3_HARNESS_20261123.md` | `~/tfm_worktrees/e2e-map/docs/drafts/e2e_v1/READY_H3_STEP3_HARNESS_20261123.md` | `8e117c5101dcbe85cc6dfd846bc84e14834a620cede7bbd1549eb6d6bb67cba5` |
| 16 | `docs/drafts/loop_v1/READY_LOOP_WF_20261123.md` | `~/tfm_worktrees/e2e-map/docs/drafts/loop_v1/READY_LOOP_WF_20261123.md` | `3ffb04dc0799a4c6ea86192827d77d33c551f0289974a9157bb454f61f9e41c0` |
| 17 | `docs/drafts/loop_v1/READY_WF_SIEVE_ON_BLOCK0_20261124.md` | `~/tfm_worktrees/e2e-map/docs/drafts/loop_v1/READY_WF_SIEVE_ON_BLOCK0_20261124.md` | `b4b6082df5cd5fb1d4fb4693227c6d765f036fadb643209ef4857498a8b2e75f` |
| 18 | `docs/drafts/loop_v1/READY_WF_SMOKE_20261124.md` | `~/tfm_worktrees/e2e-map/docs/drafts/loop_v1/READY_WF_SMOKE_20261124.md` | `eb8043123e54d5bee7a7ec34f689c138b4aba114caca6ec5dd79f28a487d3bf4` |
| 19 | `docs/drafts/loop_v1/READY_WF_TARGET_HITS_BLOCK0_20261124.md` | `~/tfm_worktrees/e2e-map/docs/drafts/loop_v1/READY_WF_TARGET_HITS_BLOCK0_20261124.md` | `c2ae30e656c00468659ecfe43cbd3cd75743f9f79bc3bb9b3164dda30deeb9cb` |
| 20 | `docs/drafts/loop_v1/WF_FIRST_TABLE_GUIDE_20261123.md` | `~/tfm_worktrees/e2e-map/docs/drafts/loop_v1/WF_FIRST_TABLE_GUIDE_20261123.md` | `ee917e55e4b40211a08755854539b874ba944a712846edb8335d39bbf331e203` |
| 21 | `docs/drafts/loop_v1/runH_commands/00_base.sh` | `~/tfm_worktrees/e2e-map/docs/drafts/loop_v1/runH_commands/00_base.sh` | `c301eee0e7cc4441b6f048e3006dc83fb54c534903119dc9aefcb37b62f6adaf` |
| 22 | `docs/drafts/loop_v1/runH_commands/01_sieve_on.sh` | `~/tfm_worktrees/e2e-map/docs/drafts/loop_v1/runH_commands/01_sieve_on.sh` | `8806ddd09006ca5900e712c2f67b30f48eb18bc9b60d4f85f7617c7df0d3052e` |
| 23 | `docs/drafts/loop_v1/runH_commands/02_target_hits.sh` | `~/tfm_worktrees/e2e-map/docs/drafts/loop_v1/runH_commands/02_target_hits.sh` | `bed6d7ed007f292cd0839df4dc7c3eed15bee8f4a28c8104a100fd013c6b465b` |
| 24 | `docs/drafts/loop_v1/runH_commands/03_rankers.sh` | `~/tfm_worktrees/e2e-map/docs/drafts/loop_v1/runH_commands/03_rankers.sh` | `31987acbd4e87d4cab903caa9529230b8c458adb47f313c3e65a3ed30265a795` |
| 25 | `scripts/diagnostics/e2e_pipeline_caps.py` | `~/tfm_worktrees/e2e-map/scripts/diagnostics/e2e_pipeline_caps.py` | `4e89f2f333a2a8e8a533a525e194a77fd180e40c5f798237262c60249cf2b8a7` |
| 26 | `scripts/diagnostics/e2e_build_run_copy.py` | `~/tfm_worktrees/e2e-map/scripts/diagnostics/e2e_build_run_copy.py` | `bd4d1ae8b133d50a06addeefeadb97d076e316a36c7b62a4c0163019e3a911a9` |
| 27 | `scripts/diagnostics/e2e_step3_harness.py` | `~/tfm_worktrees/e2e-map/scripts/diagnostics/e2e_step3_harness.py` | `d6390a012404bb140c995cf44da31614955adc80b3b5a477ba572e5111c06dcb` |
| 28 | `scripts/diagnostics/e2e_step3_selftest.py` | `~/tfm_worktrees/e2e-map/scripts/diagnostics/e2e_step3_selftest.py` | `588bc5496340e871dcc58c61845902a97beece4cc760a69ccf31d5b5fe983084` |
| 29 | `scripts/diagnostics/e2e_step5_check.py` | `~/tfm_worktrees/e2e-map/scripts/diagnostics/e2e_step5_check.py` | `5087de3186577335ec6ae8bba2c4e8ddf9630af713f43d06d12e7d42bbc9dde5` |
| 30 | `scripts/diagnostics/e2e_step5_ranker.py` | `~/tfm_worktrees/e2e-map/scripts/diagnostics/e2e_step5_ranker.py` | `57e5a3eae36431825ddb2853adfdfce33ef48ca907a4b5fb97675a50b7054a62` |
| 31 | `scripts/diagnostics/e2e_step5_shim.py` | `~/tfm_worktrees/e2e-map/scripts/diagnostics/e2e_step5_shim.py` | `7528880579a6dc73d899ccf832356338d19d2aa20fd8188599b17a3665b1470a` |
| 32 | `scripts/diagnostics/e2e_walk_forward.py` | `~/tfm_worktrees/e2e-map/scripts/diagnostics/e2e_walk_forward.py` | `06595223e2d3765e57ed37323e39deb9ae653ab7a149f0f3fb5bcc9ac47dc0e1` |
| 33 | `scripts/diagnostics/e2e_walk_forward_selftest.py` | `~/tfm_worktrees/e2e-map/scripts/diagnostics/e2e_walk_forward_selftest.py` | `ca844ed44f050336e7ab98e69e141fda1f9ad147f5e74b800f9426f3b188ad64` |

**Also to read:**
- **The READY note:** `~/tfm_worktrees/e2e-map/docs/drafts/loop_v1/READY_LAND_WF_VOTE_20261125.md`. Its current sha256 is `37ca19615ec8aee2e4f3cb43a515be03207ff9ecf6271a828648cc6c28a94fd5`. §0 covers ADDENDUM 2, §1 the switch refusals (ADDENDUM 1), §5 the round-1 results and §6 the full lists. It lands with the files as `docs/drafts/loop_v1/READY_LAND_WF_VOTE_20261125.md`.
- **The evidence folder:** `~/tfm_worktrees/e2e-map/data/landing_wfvote_20261125/`, which holds:
  - suite logs: `B_*.log` (round 1), `B2_*.log` and `B2wf_*.log` (round 2), with summaries `*_summary.txt`;
  - probe records `probe_today_B.json` (before the fix), `probe_fixed_B.json` (after) and `probe_A.json`;
  - comparison records `compare_*.json`;
  - copy records `copied_*.txt`;
  - the lists `list_B.txt` / `list_B_r2.txt`;
  - the scripts `probe_ll.py`, `compare_default.py`, `copy_in.sh` and `run_suites.sh`.

**About running anything:** these files are not in this repository, so a run needs a tree that holds them. A write into this repository during your review is a `BOUNDARY VIOLATION`. `AGENTS.md` §3 allows `/tmp` for throwaway checks. Please say what you ran and what you only read.

## What was run (VM 101, CPU only, `CUDA_VISIBLE_DEVICES=""`, fresh detached worktree of `fe5ac3b` with only the listed files copied in, suites one at a time)

| Suite | Round 2 (the files above) | Round 1 (earlier versions of 5 files) |
|---|---|---|
| `tests/test_e2e_vote_harness.py` | gates 15/15, mutants 17/17 (incl. G19 LL-1, M25) | 14/14, 16/16 |
| `tests/test_e2e_vote_ll5.py` | 3/3, 3/3 | 3/3, 3/3 |
| `tests/test_e2e_vote_ordering.py` | 6/6, 3/3 | 6/6, 3/3 |
| `tests/test_e2e_miss_report.py` | 7/7, 5/5 | 7/7, 5/5 |
| `tests/test_step1_screen.py` | GATES 15/15, MUTANTS 13/13 | same |
| `scripts/diagnostics/e2e_walk_forward_selftest.py` | 77/77 (incl. 13 W10 switch gates) | 77/77 |
| `tests/test_e2e_grader.py`, `test_compare_settings.py`, `test_switch_check.py`, `test_registry_repair.py`, `mutate_e2e_grader.py`, `test_window_anchor_brief2.py`, `scripts/diagnostics/e2e_step3_selftest.py` | **not re-run** | 37/37, 9/9, 12/12, 5/5, 89/89, 18/18 + 12/12, 18/18 |

## Known limits (stated here so you do not have to find them)

1. **Seven suites were not re-run after round 2:** `test_e2e_grader`, `test_compare_settings`, `test_switch_check`, `test_registry_repair`, `mutate_e2e_grader`, `test_window_anchor_brief2` and `e2e_step3_selftest`.
   - Round 2 changed only the vote harness, three vote tests and the walk-forward selftest.
   - I did not check by running that the seven are unaffected; I infer it.
2. **One item is UNVERIFIED.** The walk-forward selftest's vote snapshot now copies `e2e_pipeline_caps.py` beside the snapshot harness (`e2e_walk_forward_selftest.py:121`).
   - I did not read a vote run record from inside that selftest to confirm the snapshot harness recorded its LL-1 check. The record I opened was a driver record, and the tree was removed before I noticed.
   - The harness's own record is gated by G19 in `tests/test_e2e_vote_harness.py`.
3. **The driver's forbidden-root rule.** `FORBIDDEN` (`e2e_walk_forward.py:98`) contains `~/distributed_prng_analysis`, and the out-dir must lie under the driver's own repo `data/`.
   - So the driver refuses every run from the main checkout. Runs need a detached worktree.
   - The run H scripts `cd` to their own repo and also read `data/e2e_runH/copy/...` there. That directory exists only in `e2e-map`'s untracked `data/`.
4. **The non-vacuity sides read another worktree.** W10 (`--caps-with-root`) and G19 (`E2E_VOTE_CAPS_WITH_ROOT`) read `~/tfm_worktrees/e2e-map` by default, as the owner directed. If that worktree goes, those gates fail with a precondition message.
5. **The run H scripts were not run on the card.** `runH_commands/0{0,1,2,3}_*.sh` were exercised through their distinguishing driver option on the selftest fixture, on CPU (`probe_ll.py`).

## Questions, real-world cases on this system as it is today

**Q1. Refusal.** On main `fe5ac3b` with only these files added: does any of the following that requests an LL-1, LL-2 or LL-3 switch refuse before writing anything, naming the switch and the file?
- a driver option;
- a variant script;
- a direct run of the vote harness or of the Step 3 harness.

Can any such request run as the base variant under the wrong label?

**Q2. Record.** Does every run record state the settings actually applied? Do base, 02 target hits and 03 rankers run on main's unmodified pipeline files?

**Q3. Additive.** Does the landing change any behaviour of main apart from `tests/test_step1_screen.py` line 55? No pipeline file, manifest or governed file is in the list.

Please label every objection `LIVE` or `THEORETICAL` as `AGENTS.md` §5 requires.
