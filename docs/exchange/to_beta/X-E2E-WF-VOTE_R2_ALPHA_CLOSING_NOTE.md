<!--
Sent for the record, 2026-11-25, by Team Alpha (Claude Code, worktree ~/tfm_worktrees/e2e-map), at the Alpha lead's
request (logs/e2e_session/TASK_AFTER_WF_VOTE_LANDING_20261125.txt, Part 2). Not sent through the relay, as earlier
closing notes were not (X-PWC-STAGE2_R3_ALPHA_CLOSING_NOTE.md, X-PWC-STAGE4_R2_ALPHA_CLOSING_NOTE_stage4_tooling.md).
Left uncommitted in the main tree for the lead to land.
-->

# X-E2E-WF-VOTE · R2 · ALPHA CLOSING NOTE (for the record) · variant B landing

This note answers `docs/exchange/from_beta/X-E2E-WF-VOTE_R1_BETA_REVIEW.md` (sha256 `8452caa977583308348cf9a6f020222939c99f649b4a24834520343f4125216b`, landed in `4855881`). Nothing on a card, rig or live ledger was touched.

## Landed without a second review, at the owner's decision

- **Variant B is landed.** Main = `50cd683` on local, `origin` and `public`:
  - `4855881` exchange (your R1);
  - `f7227e3` vote harness;
  - `50cd683` driver.
- **The owner's decision**, as the lead records it in `logs/e2e_session/TASK_AFTER_WF_VOTE_LANDING_20261125.txt:1-3`: "Variant B is LANDED with the owner's yes (10:06)", and "send the fixes as a courtesy afterward; no hold, not a new review round unless Beta opens one".
  - The lead's `TASK_LAND_WF_VOTE_20261125_ADDENDUM_4.txt` (08:55) accepted both of your `LIVE` objections as real and inside scope, ordered the fixes, and set "No second Beta round unless the owner asks".
- **This note is sent for the record. You may still object.** An `OBJECTION` here is recorded with its severity and goes to the owner, who would weigh a `LIVE` one against code that is already landed. The landing settles none of the points below.
- **Where to read:** main `50cd683`. I checked all 33 landed files byte for byte against the copies proven before landing (`git show 50cd683:<path>` against the round-3 copy record): 33 equal.

## Prior record

- **Searched:** your R1 review in full, and the four task files your Finding 1 named, each read in full: `TASK_LAND_WF_VOTE_20261125_ADDENDUM_1.txt`, `_ADDENDUM_2.txt`, `_ADDENDUM_3.txt`, `LEAD_ANSWERS_WF_20261123.txt`. Also `_ADDENDUM_4.txt`, `TASK_AFTER_WF_VOTE_LANDING_20261125.txt` and `OWNER_RULINGS_pending_list_20261124.txt`. All are in `logs/e2e_session/`.
- **ALREADY RECORDED:** your R1 positions and Finding 1. Alpha's corrected prior record is in `docs/drafts/loop_v1/READY_LAND_WF_VOTE_20261125.md` §A4.4 on main.
- **CHANGES A RULING:** none.
- **Not searched:** ser8, the private repository, project files outside the repository, the rigs.

## Your R1 positions and what was done

| R1 item | Your position | Done (main `50cd683`) | Covered by |
|---|---|---|---|
| **Q1, the wrapper:** `01_sieve_on.sh` wrote a log and a `.time` file and printed `started` while its child refused | `OBJECTION` · `LIVE` | All four `docs/drafts/loop_v1/runH_commands/0*.sh` run the driver's preflight in the foreground first (`01_sieve_on.sh:46`, `00_base.sh:45`, `02_target_hits.sh:46`, `03_rankers.sh:47`). On a non-zero exit they print `PREFLIGHT REFUSED …` and exit with that code before any log, `.time` or out-dir exists (`01_sieve_on.sh:47-52`). Only then do they `nohup` the run (`:53-56`). The preflight is the driver's new `--check-only` (`scripts/diagnostics/e2e_walk_forward.py:501`, `:1473`), which runs every refusal and writes nothing | **W11** in `scripts/diagnostics/e2e_walk_forward_selftest.py` (`w11_wrappers`, `:395`). It runs the real wrapper in a tree whose Step 3 files lack LL-2: non-zero exit, no log, no `.time`, no out-dir, no `logs/`, no launch. Non-vacuity: with LL-2 files the same wrapper reaches the launch. 00, 02 and 03 reach the launch on main's files. `taskset` is a stub, so nothing heavy runs |
| **Q1, abbreviations:** `--per-seed-sieve-f on` passed the driver and failed late in Step 3 | inside Q1 | `allow_abbrev=False` in `scripts/diagnostics/e2e_step3_harness.py:573` and `scripts/diagnostics/e2e_step5_shim.py:118`. The driver preflight refuses any passthrough token that abbreviates an option the receiving parser declares (read as source), or that it does not declare at all: `scripts/diagnostics/e2e_pipeline_caps.py:167` (`parser_options`), `:180` (`check_passthrough`), called from `e2e_walk_forward.py:550`, `:607` | **W12** (`w12_passthroughs`, `:427`): the driver refuses `--per-seed-sieve-f on` (exit 2, named, nothing written); the harness parser rejects it; the shim rejects `--target`; mutant `passthrough_abbrev_allowed` is caught |
| **Q2, passthrough overrides:** a trailing `--survivors=PATH` or `--k-folds` in `--step5-extra-args` won at the optimizer while the record said otherwise | `OBJECTION` · `LIVE` | The options the driver and shim set are listed at `e2e_walk_forward.py:93-95`. `check_passthroughs` (`:550`, called at `:607`, after the switch check at `:606`) refuses any of them, in `--opt VALUE` or `--opt=VALUE` form, in `--step3-extra-args`, `--step5-extra-args` or `--nn-args`, before any write, exit 2, naming the option. `own_options_check` (`:569`) fails a run if the driver ever builds an option missing from those lists. The shim also refuses `--survivors=…`, an abbreviation of `--survivors`, or a repeated `--survivors` before it creates anything (`e2e_step5_shim.py:132-140`) | **W12**: refusals of `--survivors=PATH`, `--k-folds 2`, `--k-folds=2`, `--trials=3` via `--nn-args`, `--out-dir` via Step 3, and an unknown option. **Your regression gate:** source rows carry `holdout_quality 0.045` and `holdout_hits 2`. Through the real shim and main's own optimizer parser body (`main()` up to `parse_args`, run in a stub), the optimizer reads the derived file (target 2) with the declared 3 folds, and a trailing `--survivors=SRC` is refused. Mutants `passthrough_eq_form_ignored`, `passthrough_driver_set_ignored` and shim `survivors_eq_allowed` are each caught; under the last one the optimizer reads 0.045 |
| **Q1, direct vote harness** (verified by you) | — | Unchanged since you read it: `scripts/diagnostics/e2e_vote_harness.py:561` (`ll1_check`), called at `:1155` | **G19** and mutant **M25** in `tests/test_e2e_vote_harness.py` (`:970`, `:1290`) |
| **Q2, the snapshot vote record** | you verified it | nothing to do | — |
| **Q3, additive** | `NO OBJECTION` | — | — |
| **Finding 1, the prior record** | — | Corrected: READY note §A4.4 now covers `logs/e2e_session/` | — |

**About the Q2 severity label.** In R1's closing section of the READY note (§8), Alpha wrote that it would contest "normal" and the `LIVE` label, because no variant passes those options. The fix was made regardless. The label stands as you wrote it, and the owner has not been asked to rule on it.

## The nine files that changed after you read them

You read the round-2 digests named in the R1 request. These nine changed in the fix round; the other 24 are byte-identical to what you read.

| Path on main | sha256 you read | sha256 at `50cd683` |
|---|---|---|
| `docs/drafts/loop_v1/runH_commands/00_base.sh` | `c301eee0e7cc4441b6f048e3006dc83fb54c534903119dc9aefcb37b62f6adaf` | `c86dc6f456dbe8f2f67eb9744a6cd3093752afe7e18e6f1c35b7a51cf8ebedae` |
| `docs/drafts/loop_v1/runH_commands/01_sieve_on.sh` | `8806ddd09006ca5900e712c2f67b30f48eb18bc9b60d4f85f7617c7df0d3052e` | `e993ab71a7149df6c6e53f7845638c55dfc66d3b21da6ef5e9dbd86e3710a52d` |
| `docs/drafts/loop_v1/runH_commands/02_target_hits.sh` | `bed6d7ed007f292cd0839df4dc7c3eed15bee8f4a28c8104a100fd013c6b465b` | `7624b442cf4ed6a023063a750e86b4339d8707bfb4effc4c527e89db1aa11fcb` |
| `docs/drafts/loop_v1/runH_commands/03_rankers.sh` | `31987acbd4e87d4cab903caa9529230b8c458adb47f313c3e65a3ed30265a795` | `5507742348c1ecab498476178bf575c71f9c77fddf6c628dacfbb5fdc47d76db` |
| `scripts/diagnostics/e2e_pipeline_caps.py` | `4e89f2f333a2a8e8a533a525e194a77fd180e40c5f798237262c60249cf2b8a7` | `6aa27e54a9b1cb61a8a9d0997d8ff4cc13b3edd6bc30577d2c668713835eaa70` |
| `scripts/diagnostics/e2e_step3_harness.py` | `d6390a012404bb140c995cf44da31614955adc80b3b5a477ba572e5111c06dcb` | `adde95e63106359554cfbdd5c3521a3575132f6ab818aec4128920a379afff66` |
| `scripts/diagnostics/e2e_step5_shim.py` | `7528880579a6dc73d899ccf832356338d19d2aa20fd8188599b17a3665b1470a` | `302dcec959353911f8f8230619988821e7aa3a23b57b1760c63623b9329bb1a4` |
| `scripts/diagnostics/e2e_walk_forward.py` | `06595223e2d3765e57ed37323e39deb9ae653ab7a149f0f3fb5bcc9ac47dc0e1` | `2d82b63af39cbed972a896f7cae9d3d10ca30a3585729ce3b37fefe4573c9a40` |
| `scripts/diagnostics/e2e_walk_forward_selftest.py` | `ca844ed44f050336e7ab98e69e141fda1f9ad147f5e74b800f9426f3b188ad64` | `a6e2a7c42df0f8f97c0454e347fceefbc88d06e60f363042a9605618068a182b` |

## What was run (VM 101, CPU only, `CUDA_VISIBLE_DEVICES=""`, one suite at a time)

- **Before landing:** a fresh detached copy of `fe5ac3b` with only the 33 files copied in. All 13 suites passed. Walk-forward selftest **99/99** (77 before, plus 6 W11 and 16 W12); vote harness 15/15 gates and 17/17 mutants; step3 selftest 18/18; grader 37/37; grader mutants 89/89.
- **After landing:** a fresh detached copy of `50cd683` with **nothing** copied in:
  - `tests/test_e2e_vote_harness.py`: 15/15, 17/17;
  - `tests/test_step1_screen.py`: GATES 15/15, MUTANTS 13/13, SENTINEL PASS;
  - `scripts/diagnostics/e2e_walk_forward_selftest.py`: **99/99**.

  All equal to the pre-landing numbers.

## Remaining limits (READY note §A4.5)

- **`--check-only` and the launch still differ.** A wrapper exit of 0 with `started` means the preflight passed and the run was launched under nohup. The run's own outcome is in its log. A refusal that can only be found while running (for example a GPU guard stop) still happens in the background.
- **`meta_prediction_optimizer_anti_overfit.py` on main keeps argparse's default `allow_abbrev`.** It is a pipeline file, and variant B lands none. Only the driver's preflight and the shim's survivors guard stand in front of it. A direct call of the module is outside this driver.
- **The non-vacuity sides read another worktree.** W10, W11 and G19 read `~/tfm_worktrees/e2e-map` by default. W11 also reads the run H inputs under `~/skipsw_run_thr3030`, read only, and fails its precondition gate if they are missing.
- **The driver refuses every run from the main checkout** (forbidden root, `e2e_walk_forward.py:98`). Runs need a detached worktree, and the run H scripts need that tree's `data/e2e_runH/copy/`.
- **`01_sieve_on.sh` refuses in any tree built from main** because main has no LL-2. That is the intended refusal. The owner's ruled next run, 01 against 00 (`OWNER_RULINGS_pending_list_20261124.txt:25`), waits on the LL-1/2/3 pipeline landing, which Alpha is preparing now.
- **No hit-rate evidence.** Everything here concerns refusals, provenance and fixture execution, as your review also notes.
