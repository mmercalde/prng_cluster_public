# TFM SESSION HANDOFF — 2026-11-14

## Prior record

Written from this session's own record: the commits listed below (public mirror, read at
`80d2bb4`), the five agent reports and the Beta reviews landed today, and the owner's
messages in chat. Previous handoff: `docs/TFM_SESSION_HANDOFF_20261113.md`.
Not searched for this document: VM101's live state, ser8 archives, the private repository.
Every item below is ALREADY RECORDED in the cited file unless marked **owner, in chat**;
those are recorded here for the first time.

## 1. Where RANGE-MINER stands

**The code side is ready for Run 4.** Beta's objection to launching is withdrawn
(`docs/exchange/from_beta/X-MINER-STALL_R1_BETA_REVIEW.md` Q8). What remains is launch
preparation: `docs/CLAUDE_CODE_INSTRUCTIONS_RUN4_LAUNCH_PREP_20261114.md`, then the owner's
launch, planned for the morning of 2026-11-15.

| Item | State | Record |
|---|---|---|
| Step 1 can run again after the space is covered (search record, stage 1) | Landed `915d382`, `b1cd2fa` | `docs/CLAUDE_CODE_REPORT_STEP1_SEARCH_RECORD_STAGE1.md`; Beta X-STEP1-SEARCH-CODE R3 no objection |
| No credit without evidence of a completed sweep | Landed, same commits | same |
| Survivor-volume estimator and gate; cap 25,000 | Landed `35f5503` | `docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md`; Beta X-VOLUME-LIMIT R2 no objection |
| The seed start reaches the RANGE-MINER sieve; PWC and ZMQ refuse a non-zero start | Landed `c28555d`, `04aab30` | `docs/CLAUDE_CODE_REPORT_MINER_SEED_START.md`; Beta X-MINER-START R1 |
| Staging-admission stall (L-4) repaired; WATCHER log level (L-5); Run 4 routing successor control | Landed `89be3ee`, `80d2bb4` | `docs/CLAUDE_CODE_REPORT_MINER_ADMISSION_STALL.md`; Beta X-MINER-STALL R1 no objection |
| Rig copy of `miner/range_miner_coordinator.py` | **Owed at power-on.** sha256 `b8e259b3e12b95297fa38018a41215d5c52f4d9b3f4095e384829a3ec1f1517d` | parity set, `scripts/gate12_parity_gate.py:174` |
| Run 4 over `[2³¹, 2³²)` | Not run | exact counts expected 0 / 0 / 60 / 23,568 (volume report M-B) |

**Two defects found today that would have spoiled Run 4.** (1) The RANGE-MINER route swept
from seed 0 whatever start it was given, so Run 4 would have re-swept the lower half and
recorded the upper half as covered. (2) A stripe finishing by the StripeComplete path did
not release staging admission, so a run could hang with no timeout. Both were found by
agents running the code, after reading had missed them. Both are repaired and reviewed.

## 2. Owner decisions of 2026-11-14

1. **D1–D5 on the Step 1 search record** are in `docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md` §0.
2. **No second design round with Beta on that proposal; Beta reviews the code.** Same file, "Status".
3. **The landing script is used as it stands.** `docs/exchange/to_beta/X-OWNER-LAND_R1_ALPHA_NOTICE_owner_ruling_on_first_use.md`. The owner's routine: enter the command, check the plan, type `land`.
4. **Landing commands carry their manifest** (heredoc), never a separate download. **Owner, in chat.**
5. **Work is done by agents**: Claude Code sessions on Opus, each in its own worktree, each carrying its own Beta rounds. **Owner, in chat.**
6. **Survivor cap 25,000** in one phase. Beta's position was to keep 23,515 until a run completes higher. `config_manifests/survivor_volume.cap`; volume report.
7. **Feed the volume estimate back to the search**: skip a trial over the cap, record it as load and never as a score. Not built; due before the 50-trial soak. `docs/CLAUDE_CODE_INSTRUCTIONS_SURVIVOR_VOLUME_LIMIT_20261114.md` Prior record.
8. **`sunshine` on the VM101 GPU does not make it busy.** Allow-list in the cap file.
9. **The pre-dispatch refusal of a non-zero start covers PWC and ZMQ.** Built in `c28555d`.
10. **PWC is to become a working backup** that runs both sieves at the settings that were stable; ZMQ is not pursued. **RANGE-MINER takes top priority and is not to be put at risk.** `docs/CLAUDE_CODE_INSTRUCTIONS_PWC_BACKUP_STEP1_20261114.md` Prior record. Making a PWC result count changes Beta's 2026-07-31 ruling and is decided after Run 4.
11. **The lead developer handles the small things**; the owner talks only with the lead and is brought decisions that are his: typing `land`, launching fleet runs, rulings, changes of direction. **Owner, in chat.**
12. **After Step 1 (RANGE-MINER and PWC) is in the bag, the lead orchestrates a swarm of agents, with Codex, one per pipeline step.** **Owner, in chat.**

## 3. Open work, in order

**Before Run 4 (morning of 2026-11-15):**

1. Launch preparation session: `docs/CLAUDE_CODE_INSTRUCTIONS_RUN4_LAUNCH_PREP_20261114.md`.
   It produces `docs/RUN4_LAUNCH_CHECKLIST.md`. **Watch its dataset finding**: if the dataset
   pointer has moved since attempt 9, Run 4's search is refused until the owner decides.
2. The owner follows the checklist: rigs on, `boot-proxmox`, GPU gate, coordinator to the
   rigs, parity 30/30, netconsole, open the search, launch.
3. After the run: `G-PROD-SHAPE`, the closure report, Brief I acceptance (plan A8).

**Held until after Run 4:**

4. **PWC step 1 code.** Built and reviewed (Beta X-PWC-BACKUP R2 no objection), in
   `/home/michael/tfm_worktrees/pwc-step1`, **not landed**: Run 4's launch file imports the
   PWC file. Its review files are landed (`5bb854c`). Then PWC step 2.
5. **Groundwork for the swarm** (brief not yet written; the reading is listed in chat):
   batch landing and several exchange threads in one landing; worktree names with
   underscores; several Beta reviews at once; the `Bash(git:*)` ask rule in Claude Code's
   settings narrowed to commit and push; hard-wired paths that reach live state from a
   worktree; one coordinator session on VM101 that starts and supervises the agents; Beta's
   reviewer role as a saved Codex role file; a web terminal on SER8.
6. **The hit-rate instrument.** Needed before agents work on Steps 2 to 6.

**Findings recorded today and not yet decided:**

7. **The skip range never reaches phases 1 and 2 on the RANGE-MINER route.** The assign
   payload carries no `skip_range`; the worker falls back to `[0, 16]`
   (`miner/range_miner_worker.py:1164`). Volume report F-1. It contradicts
   `docs/HYBRID_SKIP_BOUND_AUDIT.md` §2.4 M6a and `docs/TEAM_ALPHA_BOUNDED_PHASE_6_SUBMISSION.md:382`.
   Skip is a cornerstone of the design. This is the owner's decision, after Run 4.
8. **Exact counting of the hybrid phases** in place of sampling in the volume gate (about
   75 s for phase 4). Not built.
9. **The frozen routing gate `G-UNPINNED-IDENTICAL`** cannot run on today's tree and reports
   INCOMPLETE. Retiring it is the owner's and Beta's call (stall report, Part 2).
10. **A search on a different dataset** from the accumulated generation is not yet checked
    against that generation (Beta X-STEP1-SEARCH-CODE R1 Q6, THEORETICAL, `docs/exchange/DEFERRED.md`).
    Due before the first search on newer draws, with the cross-dataset merge decision and
    session-scoped authority.
11. **Stage 2 of the search record**: per-chunk trial budget, the per-search trial table and
    evaluation records, the attempt 9 import, the named-dataset route, the ML origin index,
    the Chapter 13 opening route. In practice stage 1 runs one chunk per search.
12. **Doc corrections owed**: `docs/CLAUDE_CODE_REPORT_STEP1_PURPOSE_LINEAGE.md:213-215`; the
    two skip documents in item 7.
13. **Landing-script fixes** (low priority): refuse itself as a destination; verify relay
    hashes; the items in item 5.
14. Carried from the 2026-11-13 handoff and not touched today: proposal v1.2 with the WP-1
    recommendations; the X-RUN-GATES review request into the repository; Codex reviews of
    the gates and Stage A reports; the doc fix list.

## 4. How the work ran today

- Six Claude Code sessions on Opus, each in a worktree under `/home/michael/tfm_worktrees/`
  (`step1_search`, `volume_limit`, `miner-start`, `pwc-backup`, `pwc-step1`, `miner-stall`;
  the first two also have hyphen-named links). Beta reviewed five pieces of code through
  the relay, nine rounds in all. Agents stopped for the owner at each real stop gate.
- **Lessons for the next brief writer.** Name worktrees with hyphens. Tell sessions not to
  use scheduled wake-ups during a relay (it writes `.claude/scheduled_tasks.lock` into the
  main tree and the relay reports a boundary violation). `pgrep -f` matches its own command
  line; use `pgrep -f "[b]eta_relay.py"`. `gate_manifest_staging_dir_is_usable_here` writes
  a probe into the live staging folder. Before a brief tells a session to change test
  fixtures, read every Beta ruling on those fixtures in full.
- **Errors by Alpha today, on the record**: the manifest delivered as a separate download;
  underscores in worktree names; "no compute process" on a GPU that always carries one;
  a brief that contradicted X-RUN4-RANGE R1 Q4; "months" for the PWC backup, repeated from
  an agent without checking; the claim that the volume limit stood before Run 4.
- **Remote access.** Chrome with the Claude extension is installed on SER8 and on VM101.
  This chat can operate web pages in either; it has no terminal there. Not used today.

## 5. Next session: start here

1. Read this file and `docs/RUN4_LAUNCH_CHECKLIST.md` if it exists.
2. Clone the public mirror fresh; `git log --oneline -25`.
3. Read `docs/CLAUDE_CODE_REPORT_RUN4_LAUNCH_PREP.md`, its top section first.
4. One item at a time with the owner, in plain words. No commands until he agrees.
