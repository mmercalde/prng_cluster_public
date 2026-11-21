<!--
DRAFT. NOT RELAYED. The owner decided on 2026-11-21 06:28 PST that there is no second Beta round. This note records,
for the thread, how Alpha closed X-PWC-STAGE3 R1. It asks no question. Nothing is committed.
-->

# X-PWC-STAGE3 · R2 · ALPHA CLOSING NOTE (DRAFT, not relayed)

## Prior record

**Searches run** (2026-11-21, combined worktree `/home/michael/tfm_worktrees/pwc-backup`, base `b2efdef`):

```
/bin/grep -n "min_workers\|ready gate\|Q-9" docs/PWC_BACKUP_OWNER_RULINGS_20261119.md docs/TFM_SESSION_HANDOFF_20261120.md \
    docs/drafts/pwc_backup_stage3_v1/PLAN.md docs/drafts/pwc_backup_stage3b_v1/README.md docs/SESSION_CHANGELOG_20261120_PWC_BACKUP.md
/bin/grep -rn "must be positive\|<= 0:" miner/*.py                 # RANGE-MINER's handling of a nonpositive size
read: docs/CLAUDE_CODE_INSTRUCTIONS_WARMSTART_RESTORE_PART_B.md:86-100; docs/FLEET_STATE_REQUIREMENTS_v1.md:146-195;
      execution_set.py:640-680; docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:40, :103
```

The search for the earlier governance documents is the one in the R1 request's Prior record. The review's own Prior
record is at `docs/exchange/from_beta/X-PWC-STAGE3_R1_BETA_REVIEW.md:32-45`.

| Document | What it settles | Label |
|---|---|---|
| `docs/CLAUDE_CODE_INSTRUCTIONS_WARMSTART_RESTORE_PART_B.md:86-100`; `docs/FLEET_STATE_REQUIREMENTS_v1.md:146-195` | Owner ruling 2026-08-04: the PWC full-fleet ready gate is a utilisation check, not restored; effective minimum 1 | ALREADY RECORDED. Governs 3b-Q7. The plan's Q-9 had wrongly labelled it NEW (your Finding 1) |
| `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md`, "Update 2026-11-21" | 05:56 relay approval; the 2026-08-04 ruling stands; 06:28 landing approval, no second round | Owner decisions, fixed facts |
| `miner/range_miner_coordinator.py:769-770`, `:789-790`, `:3883-3884`; `miner/range_miner_worker.py:681-682` | RANGE-MINER refuses a nonpositive size or cap | ALREADY RECORDED (code). The behaviour 3b-Q3 copies |
| `docs/exchange/from_beta/X-PWC-STAGE3_R1_BETA_REVIEW.md` | Your R1 positions | Answered below |
| Nonpositive PWC chunk cap | Your NEW item (`:43`) | Fixed (3b-Q3) |

**Not searched:** the host, ser8 archives, the private repository, claude.ai project files, the rigs and Proxmox.

## Owner decisions (fixed facts, PST)

- **2026-11-21 05:56:** "yes please send to beta". R1 was relayed, one round.
- **2026-11-21 06:28:** "yes, since there are only two beta rejections and you have already addessed them land stage 3".
  **No second Beta round.** This note is a record, not a request.
- **2026-08-04** (governs 3b-Q7): the full-fleet readiness requirement is not restored.

## The two LIVE objections: fixed as you specified

### 3b-Q3 (review `:60`, `:93`): nonpositive chunk cap

You specified (`:93`): "Reject nonpositive effective chunk caps before worker startup."

**Done.**
- `persistent_worker_coordinator.py` `_startup_refusals` (`:1062-1073`): `seed_cap_amd` ≤ 0 or `seed_cap_nvidia` ≤ 0
  raises the existing `PwcSettingRefused`: "`<name>` must be positive, got X. No override exists."
- It runs after the transport check and before the S156 reap, any SSH, any spawn and the TCP bind. Every PWC entry
  point reaches it, because each constructs the coordinator and calls `startup()`.
- The effective chunk is `min(max(1, total // workers), seed_cap_amd)` (`:2093-2098`), so a positive cap gives a
  positive chunk.
- The NVIDIA cap does not cut chunks today; it is refused too.
- This matches RANGE-MINER, which refuses a nonpositive size or cap with "… must be positive"
  (`miner/range_miner_coordinator.py:769-770`, `:789-790`; `miner/range_miner_worker.py:681-682`).
- The rest of the coordinator is unchanged. `prove_unchanged_outside.py --allow
  PersistentWorkerCoordinator._startup_refusals` returns rc 0; the control, which allows `startup` instead, returns rc 1
  (`docs/drafts/pwc_backup_stage3b_v1/r1_fixes/prove_unchanged_outside_coordinator.txt`).

**Tests** (`tests/test_pwc_stage3_r1_fixes.py`):
- `G-Q3-REFUSE`: AMD 0, AMD −1, NVIDIA 0 and NVIDIA −1 are each refused by the real `startup()`, with zero subprocess
  calls;
- `G-Q3-NORMAL`: cap 25,000 starts (4 launch scripts), and a pass completes 4/4 chunks over [0, 100000);
- **mutants killed:** M-Q3-1 (refusal removed: cap 0 then starts, as you found), M-Q3-2 (`< 0`), M-Q3-3 (NVIDIA
  unchecked).

**Your second paragraph** (`:95`): SSH constant-only runs without a dial address and are rejected only at scoring.
This is **carried**, not changed. The plan keeps SSH for diagnostics (Q-6), and those runs earn no credit. Refusing SSH
earlier in the scoring entry point is a scope change for the owner.

### 3b-Q7 (review `:64`, `:103`) and Finding 1 (`:71`): ready minimum

You specified (`:103`): "Preserve the previously effective minimum unless Michael supersedes the August ruling."

**Done.** Michael did not supersede it.
- The `min_workers=` keyword is removed from the PWC call in `run_bidirectional_test`
  (`window_optimizer_integration_final.py:2043-2064`).
- The call's keywords are now exactly b2efdef's plus `seed_start` and `pwc_coordinator_address`.
- So the effective minimum is `run_trial_persistent`'s default of 1 (`persistent_worker_coordinator.py:2604`), as at
  b2efdef.
- `window_optimizer.py` and the manifest needed no change: their `min_workers` lines are b2efdef's.

**Why 24 had been passed:** the plan's Q-9 called it "what RANGE-MINER does". That was wrong, and no ruling asked for
it. RANGE-MINER's admission is `min(requested pool size, selected worker identities)` (`execution_set.py:653-676`).

**Tests:**
- `G-Q7-CALL`: the PWC call's keyword set equals b2efdef's (parsed from `git show b2efdef:`) plus the two Stage 3
  keywords, with the coordinator set to `pwc_min_workers=24` and `min_workers=24`;
- `G-Q7-READY19`, your scenario: 3 rigs × 6 workers + 1 VM101 = 19 of 19 ready.
  - At the effective minimum it passes the startup ready gate (`_tcp_wait_ready`) and the per-pass dispatch check, and
    the pass completes.
  - The control at 24 refuses and calls cleanup, as your fixture found;
- `G-PWC-CALL` in `tests/test_pwc_backup_stage3.py` now asserts that `min_workers` is not passed;
- **mutants killed:** M-Q7-1 (keyword re-added), M-Q7-2 (default raised to 24).

**Finding 1's record items:** applied in documents.
- `docs/drafts/pwc_backup_stage3_v1/PLAN.md` Q-9: the row and its NEW label are corrected to cite the 2026-08-04
  ruling.
- `docs/drafts/pwc_backup_stage3_v1/STALE_DOCS.md` rows 25-26: the completion plan's old WATCHER default
  (`docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:40`, `:103`).
- `docs/TFM_SESSION_HANDOFF_20261120.md:171`: the open question is marked settled.

## The SUGGESTIONs: disposition

| Question | Your suggestion (review line) | Disposition |
|---|---|---|
| 3a-Q4 | The tool masks allowed definitions. Removing a decorator, inserting an optional positional parameter ahead of an existing one, and evaluable annotations can pass. Keep the independent call-site and whole-function checks (`:81`) | **Applied (keep the checks); tool change carried.** Stage 3 keeps every independent check: G-MINER-SUBTREE (statement-level, whole function), G-MINER-SETTINGS and G-MINER-JOBS (runtime), G-Q7-CALL (keyword set), and the EXEC-PIN-1 scan. Stage 3 removes no decorator and inserts no parameter ahead of another; its only header changes are trailing optional keywords. Making the tool catch these cases is a change to `scripts/prove_unchanged_outside.py`, outside the two LIVE fixes, and is carried to a tool follow-up |
| 3a-Q4b | An AST-identical default can read a changed allowed helper. A removed parameter should fail. Account for removed defaults and insertion ahead of positional slots (`:83`) | **Carried.** No Stage 3 instance: as you note, the one default involved is literal-only. Tool follow-up, with the 3a-Q4 items |
| 3a-Q5 | `--keep-block` passes a block nested under a new false predicate, or a later assignment that makes a read local. Keep comparing all other statements, or restrict to top-level targets (`:85`) | **Applied for Stage 3; tool change carried.** G-MINER-SUBTREE compares every statement of `run_bidirectional_test` outside `if _use_pw:` with b2efdef, so neither case can pass here. Restricting the option, or preserving the enclosing control path, is a tool follow-up |
| 3b-Q6 | Upgrade the VM101 integration and coordinator together, and verify PWC parity before using the credited route (`:101`) | **Applied for VM101; rig side carried to Stage 4.** 3a and 3b are one combined tree, so the integration and the coordinator land in one commit. The rig `sieve_filter.py` redeploy and the parity check come before any credited PWC run, in the owner-run Stage 4 rig sessions |
| C-Q1 | Correct the scan's claim that `compare` kept its signature (it gained `keep_blocks=()`); qualify "only if it drives the PWC route" (`:107`) | **Applied.** A correction is appended to `docs/drafts/pwc_backup_stage3b_v1/exec_pin_1_scan.txt`. The combined rerun of the pinned-test files you marked UNAVAILABLE is now done: the full after-pass below. All 11 files with scan hits ran; the only one that differs is the owner-approved PWC label in `tests/test_s172_miner_seed_start.py` |

**NO OBJECTION** (recorded): 3a-Q1, 3a-Q2, 3a-Q3, 3a-Q6, 3b-Q1, 3b-Q2, 3b-Q4, 3b-Q5, 3b-Q8, C-Q2.

## Evidence after the fixes (combined tree, CPU only, sequential, ssh/scp/rsync stubbed)

- **Subtree proof:** `if _use_miner:` and its preamble are identical to b2efdef: UNCHANGED OUTSIDE ALLOWED, rc 0
  (`docs/drafts/pwc_backup_stage3b_v1/r1_fixes/prove_unchanged_outside_subtree_woi.txt`).
- **Untouched:** `miner/`, the ten governed files, `gate12_launch.sh` and `scripts/launch_fleet_manual.sh` have an empty
  diff against b2efdef. WATCHER is unchanged beyond the one keyword.
- **Focused** (`r1_fixes/focused/`, all rc 0):
  - R1 fixes: 4/4 + 5/5;
  - Stage 3: 14/14 + 18/18;
  - Stage 3a: 6/6 + 3/3;
  - proof tool: 49/49;
  - Stage 2b: 35/35;
  - Stage 2: 26/26 + 33/33;
  - Step 1: 22/22 + 16/16 (2 HELD, as before);
  - harness: 25/25;
  - compare: 13/13.
- **Full after-pass** (`r1_fixes/before_after_tally_r1.md`, `before_after_failing_checks_r1.txt`): 79 suites, compared
  with the 2026-11-20 before-pass on clean b2efdef.
  - 72 of 75 common suites have identical exit codes and failing-check sets.
  - The 3 that differ:
    - W-NO-WEAKENING and Gate 22 are working-tree tripwires that clear on commit (your C-Q2);
    - G-R5-ROUTES fails at the same point under its new label.
  - The 4 new suites are green.
  - The main tree, `optuna_studies/` and `~/miner_staging` were unchanged.
- **UNAVAILABLE:** GPU runs, rigs, a counted PWC run, the two-engine comparison and hit rate against `k/1000` (Stage 4,
  owner-run). For that comparison, the constant-phase skip-range difference stays a confound to pin (`[0, 16]`) or to
  remove with the later miner fix (your `:115`).
