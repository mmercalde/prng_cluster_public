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
