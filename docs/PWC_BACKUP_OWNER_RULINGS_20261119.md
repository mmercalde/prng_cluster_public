# PWC as a full backup: owner rulings (2026-11-19)

Recorded by Claude (Team Alpha lead). Michael gave these in chat on 2026-11-19 at 20:57 PST, in answer to the four rulings asked for in `docs/drafts/SCOPE_PWC_FULL_BACKUP_20261119.md` (on main at `27eb991`). His words are quoted exactly.

## Governing intent (ALREADY RECORDED in the repo; restated by Michael, not new)

> "its a backup so should perform jsut like range miner- remember both range -miner and pwc were designed to be stand alone in that the rest of the code does not care where the data came from."

Prior record, checked against the public mirror at `27eb991` on 2026-11-19:

- **Later steps cannot tell the engines apart.** `docs/TEAM_ALPHA_PWC_COMPARATOR_SCOPE_CORRECTION.md:12-19`: the rule set at the pivot is interface compatibility; "The remaining steps must not be able to tell whether the data came from PWC or RANGE-MINER", a contract about the 22-array output and its consumers, not about values. Same rule at `docs/CHAPTER_1_WINDOW_OPTIMIZER.md:2020-2022`, `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:899`, `docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md:1470`, `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:327`, `docs/PIPELINE_BEHAVIOUR_MODEL.md:429` and `:1084`, and `tests/phase6/wall_ab_gate.py:8`.
- **PWC kept as a backup.** `docs/CLAUDE_CODE_INSTRUCTIONS_PWC_BACKUP_STAGE_A_20261114.md:33` records the owner on 2026-11-14: RANGE-MINER was designed as a separate callable program with PWC and ZMQ as fallbacks, and PWC should work and be kept as a backup.

Where a later question about PWC behaviour is not covered below, the default is: PWC does what RANGE-MINER does. Anything that would make PWC differ from RANGE-MINER goes back to Michael.

## The four rulings (new on 2026-11-19)

| # | Question | Ruling | His words |
|---|---|---|---|
| 1 | Lift PWC's variable-skip block once the threshold is fixed? | **Yes.** The fix is that PWC uses the trial thresholds the same way RANGE-MINER does. | "yes- once the threshold is fixed(same as range miner) we lift the block." |
| 2 | Should a PWC run earn search credit? | **Yes.** | "yes." |
| 3 | Drop PWC's refusal of a non-zero seed start? | **Yes.** | "yes, its a backup so should perform jsut like range miner" |
| 4 | Phase rule: always four phases, or stop early? | **Follow RANGE-MINER** (always four phases). | "im sure #3 answered this but if not PWC should follow range miner-." |

## What these do and do not authorise

- They settle design direction for Stages 2 and 3 of the scope. They are not a yes to land any code. Each landing still needs Michael's yes, and coordinator and launch-path code still goes to Beta for a before-commit review (repository contract).
- Beta is not asked to rule on these. Beta receives them as fixed facts in each review request and checks that the code does what was decided.
- "DO NOT break range miner" (Michael, 20:50 PST) stands over all of this work. Any change in a file RANGE-MINER also uses must be shown to leave RANGE-MINER's behaviour unchanged.
- The skip-range design question (whether the 6-99 range should steer the variable-skip phases at all) is separate, affects both engines, and is still Michael's to rule on. It does not block this work.

## Smaller decisions the scope also lists (not yet put to Michael)

The scope's Stage 0 row lists further items beyond the four: TCP-only for counted PWC runs, an HSA override, a PWC cache record, and the order of the constant skip-range question (before or after PWC). Under the governing intent above, the default is to match RANGE-MINER; any that cannot be settled that way will be brought to him one at a time when the stage that needs it is reached.

## Status

- Stage 1 (PWC-only, needs no rulings): coded by the PWC agent 2026-11-19/20, CPU only, rigs powered off. Beta review X-PWC-STAGE1 R1 (2026-11-20): no objections; three suggestions carried to Stages 2 and 3. Landed with this file on Michael's yes of 2026-11-20 06:24 PST.
- Stages 2 to 4 not started. Plan for Stages 2 and 3: `docs/drafts/BRIEF_PWC_BACKUP_STAGE2_STAGE3_20261119.md` (rig-gpu-fault worktree, not yet landed).

## Update 2026-11-20

- 08:28 PST, shared test files: owner ruled that only the named PWC checks in tests/test_s172_threshold_propagation.py and tests/test_s172_phase5_d3_25_candidate_ingress.py are edited; every RANGE-MINER check in those files stays identical, shown by a line-by-line diff.
- 08:30 PST, offset: owner agreed that offset picks the draw window only (window_anchor) and generator_phase stays 0, as RANGE-MINER does (docs/PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md:15-33). PWC refuses offset other than 0 until the worker-side split (stage 2b) lands.
- 12:34 PST, Beta rounds: owner decided that Beta's round 2 finding (X-PWC-STAGE2 R2-Q1, timeout eviction race) is fixed the way Beta specified and stage 2 lands with no third round.
- 15:53 PST, Step 1 default route: owner ruled, in his words, "Yes, range miner is the new default". Context: the Step 1 manifest default route is still PWC (`agent_manifests/window_optimizer.json:262`, `:270`); RANGE-MINER becomes the default and PWC is chosen only by its flag. The manifest change goes through Stage 3 review (`docs/drafts/pwc_backup_stage3_v1/PLAN.md` S3-12); nothing is edited yet.
- 15:55 PST, final test: owner required, in his words, "And once finished your testing should run each and verify the results are the same". Meaning: the final test runs the same job through RANGE-MINER and through PWC, each selected by its flag, and compares what comes out. Design: `docs/drafts/pwc_backup_stage3_v1/PLAN.md` §F (Stage 4); not run.
- 16:45 PST, constant skip range (F4): owner ruled, in his words, "Yes, PWC follows range miner. Unless you have a other way to verify they both return the same results". So on constant-skip phases 1-2 PWC sieves the same skip range RANGE-MINER does (`miner/range_miner_worker.py:1293`). Work item: `docs/drafts/pwc_backup_stage3_v1/PLAN.md` S3-13; not edited yet. Whether either engine should honour the trial's sampled skip range is a separate RANGE-MINER question, out of scope here.
- 20:24 PST, parallel trials: owner said "Yes" to the proposal to finish the PWC backup first and then take up re-enabling parallel trials (n_parallel > 1) as its own separate job. n_parallel > 1 is SWITCHED-OFF-BY-RULING on the miner path (`docs/CHAPTER_12_WATCHER_AGENT.md:1220`, `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_7_SOAK.md:20`, `docs/BACKLOG.md:298-300`). Re-enabling needs a checkpoint and memory design for partition workers plus Beta review; that job also fixes the two PWC items left alone in Stage 3 (S3-6, the self-killing pkill at `window_optimizer_integration_final.py` optimize_window; S3-7, the dial address not passed to partitions). Not designed now.
- 20:37 PST, parallel trials, scope: owner said, in his words, "Or better half space seed search- that IS tested". Meaning: the parallel-trials job targets half-range trials, the size already run one trial at a time on RANGE-MINER (Runs P and Q, `docs/TFM_SESSION_HANDOFF_20261119.md:19-20`, `:22`; also L and M, `docs/TFM_SESSION_HANDOFF_20261118_ADDENDUM.md`), not the full range. Half range is tested with n_parallel 1 only (the Run 4 launcher pins it, `gate12_launch.sh:472`); half range with parallel trials is still untested. Not designed now.
- 21:32-21:35 PST, S3-13 withdrawn: owner said, in his words, "Well range miner must be fixed none the less" and "Please do whatever method is the quickest and least recoding". This supersedes the 16:45 entry above for S3-13 only. F4 now stands as: PWC keeps the trial's skip range; the RANGE-MINER fix is scoped separately in `docs/drafts/pwc_backup_stage3a_v1/SCOPE_MINER_SKIP_RANGE_FIX.md` (agent 2, read-only scope; no miner file edited).
- Status: stage 2 landed with this update. Stage 2b (worker-side offset split), stage 3 (shared launch files) and stage 4 (rig sessions) are not landed. Beta's R1-Q3 (five strategies) is NEEDS EVIDENCE and is answered by the stage 4 rig sessions.

## Update 2026-11-21

- After 21:35 on 2026-11-20 (time not recorded), proof tool: of the three options put to him, the owner chose to narrow `scripts/prove_unchanged_outside.py`'s header check. A non-inert default is now flagged only when it is new or its AST differs from the base.
- 05:56 PST, Stage 3 review: owner said, in his words, "yes please send to beta". The joint request `docs/exchange/to_beta/X-PWC-STAGE3_R1_ALPHA_REVIEW_REQUEST_real_world_scope.md` was relayed, one round. Beta's review is `docs/exchange/from_beta/X-PWC-STAGE3_R1_BETA_REVIEW.md`. It has two LIVE objections, 3b-Q3 (`:60`, `:93`) and 3b-Q7 (`:64`, `:103`, with Finding 1 at `:71`).
- PWC ready minimum (3b-Q7 / Finding 1): the owner's ruling of 2026-08-04 stands. The full-fleet readiness requirement is not restored (`docs/CLAUDE_CODE_INSTRUCTIONS_WARMSTART_RESTORE_PART_B.md:86-100`; `docs/FLEET_STATE_REQUIREMENTS_v1.md:146-195`). Stage 3 no longer passes the CLI/manifest pool size (24) as PWC's `min_workers`. The call shape is b2efdef's, and the effective minimum is 1.
- 06:28 PST, Stage 3 lands: owner said, in his words, "yes, since there are only two beta rejections and you have already addessed them land stage 3". Both LIVE objections were fixed as Beta specified (closing note `docs/exchange/to_beta/X-PWC-STAGE3_R2_ALPHA_CLOSING_NOTE_DRAFT.md`, not relayed). **No second Beta round.**
