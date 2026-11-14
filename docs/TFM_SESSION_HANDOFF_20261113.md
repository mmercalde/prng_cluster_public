# TFM SESSION HANDOFF — 2026-11-13

## Prior record

**What this was built from:** the full session of 2026-11-13 (07:14 to 20:55 PST), and the
public mirror at `87b3182`. Every commit listed in §3 was pulled and its files compared with
what was written in the session. **Not read:** VM101 directly, the rigs, ser8 archives, the
private repository.
**Two things were finished on VM101 but not yet committed when this was written:** the
landing script (worktree `owner_land`) and the Step 1 rerun Stage A report (worktree
`step1_rerun_a`). Run `git log --oneline -12` first and read anything after `87b3182`.

---

## 0. Read these first, in this order, before answering anything

1. `CLAUDE.md`: "The yardstick" and "Prior record first".
2. `AGENTS.md` (Beta's contract; Beta is Codex, run only through `scripts/beta_relay.py`).
3. `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md`.
4. `docs/exchange/RELAY_PROTOCOL.md`.
5. `docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md` (safe SQLite reads).
6. This file's §2 (owner rulings) and §6 (mistakes).

## 1. The two rules that were broken today, and are now enforced

**The yardstick.** TFM is functional mimicry of PRNG output, a black box. It is not seed
recovery and not state reversal. It is judged by prediction-pool hit rate against `k/1000`
on draws the pipeline never saw. The bidirectional sieve is Michael's method and is the
baseline. Alternatives are built as selectable configurations and compared by hit rate.
Alpha judged the sieve by whether a generating seed survives, twice (2026-08-08 and
2026-11-13). Do not use "in order", "reverse the state", "meet in the middle", or "the
sieve does not work". Foundational tests are not re-run.

**Prior record first.** Before writing any proposal, brief, report, review request or
recommendation, search the governance trail and show the search. Every such document opens
with a `## Prior record` section. Committed at `a41ac06`; also in the installed TFM skill
§1.0. Michael rejects anything without it. Codex's first finding on the first request under
the rule was four records Alpha still missed, so search wider than feels necessary:
rulings, proposals, audits, `CLAUDE_CODE_REPORT_*`, the fact file, the project PDFs.

## 2. Owner rulings and positions, 2026-11-13 (binding)

1. The yardstick, as above. The design governs; code written by many agents over more than
   a year is measured against it, not the reverse.
2. Prior record first, as above.
3. **Automation is a prime directive** of the code and of TFM in general. A step that needs
   a person is either a deliberate gate with a ruling behind it, or a gap.
4. **Step 1 requirements R1–R5:** pick up where an incomplete search left off; cover the
   space in manageable chunks across sessions (the four-session plan); track WHAT was run
   (dataset version, session, search settings); be able to run again when new draws come in
   and in like cases; automated. Retiring Step 1 permanently is against the learning design.
5. WP-1 (CPU/GPU generator contract): fix **authorised**, on the condition that it breaks
   nothing. Code waits for the contract and his sign-off.
6. WP-4a and WP-8 deferred until RANGE-MINER and the fixes are done.
7. **Pools:** build **both** the ranked list (S179) and the weighted vote (his original
   pools plan) and benchmark them against each other and against `k/1000`. First vote weight
   is the model's quality score. Gap stability waits for the survivor's own skip to be
   carried through. "Offset confidence" has no definition yet.
8. The X-FINDINGS-PLAN R2 boundary verdict: review accepted (`30d94bd`).
9. Build the landing script first. The inbox-branch idea (Alpha pushing to one branch) is
   **undecided**; "Alpha never pushes" still stands.
10. RANGE-MINER stays priority one. Agents on Opus. He does not use the word "lottery" for
    the project.

## 3. Commits made today (all on both remotes)

| Commit | What |
|---|---|
| `6267c7b` | Yardstick block in `CLAUDE.md` and `AGENTS.md`; pitfall record; doc fix list |
| `26d3147` | `.gitignore` for two evidence folders |
| `fbd8264` | Findings proposal v1.1; three read-only briefs; X-FINDINGS-PLAN request |
| `4b1e81e` | Beta R1 (two LIVE objections on SQLite reads); binding addendum; X-REVERSE-SIEVE R2 |
| `0066b6e` | The two run gates G2 and G3 (17/17, 18/18), not wired |
| `30d94bd` | Beta R2: both LIVE objections closed; one THEORETICAL deferred |
| `ea85436` | WP-2 Stage A report (Step 6 to Chapter 13 grading contract) |
| `2930a37` | RANGE-MINER VM101 reads report |
| `a41ac06` | Prior-record-first rule |
| `29a6cb9` | WP-1 Stage A report (CPU/GPU generator contract) |
| `8a314e5`, `ed1e669` | X-RUN4-RANGE request and Beta's R1 reply (one LIVE objection) |
| `5e100b3` | Brief for the landing script |
| `87b3182` | Stage A brief, Step 1 rerun policy (R1–R5) |

## 4. Where each thread stands

**RANGE-MINER / Run 4 (priority one).**
- The coverage cursor is at 2³¹. Run 4 as committed would sweep `[2³¹, 2³²)`; WATCHER
  overwrites the start (`agents/watcher_agent.py:2087-2089`). Attempt 9 published 0 rows.
- Once the second half is certified, `java_lcg` is COMPLETE and WATCHER generates no further
  Step 1 run at any window. That would refuse the Phase-7 soak. Codex: **LIVE** objection;
  settle the rule first, then Run 4 over the upper half, then Brief II and the soak.
- The Stage A report on this is written (`step1_rerun_a` worktree). Its summary: no route
  exists to run Step 1 again over covered seeds; R3 is not met; of 11 points in an
  unattended run, 3 are deliberate gates, 5 are gaps, 3 need no person; the four-session
  plan would run 0 trials from chunk 2 onward (read, not executed); a regime-shift rerun
  with no new draws skips Step 1. Options A to D; **B (a per-search cursor beside the full
  history) is the only one meeting R1–R4**, and it changes the 2026-08-07 ruling and the
  meaning of "finished" (F4), so it needs Michael's decision.
- **Next:** read that report in full; write the amendment (prior record first); Codex
  reviews it before any code.
- The Route-A pre-launch evidence is stale (WATCHER changed twice). Codex wants a scoped
  refresh: delta review, routing controls re-run, command captured with the real cursor.
- Still owed for RANGE-MINER: the survivor-volume limit brief (A2); a power-on date.

**The two run gates (G2, G3).** Committed, unused. Alpha found the launch count resets in a
second worktree (tested in a throwaway repo), plus three smaller points. The review request
`X-RUN-GATES_R1_ALPHA_REVIEW_REQUEST_delegated_run_gates.md` is rewritten with a prior
record and is **not yet in the repo**; a copy is in the claude.ai project under `claude/`.
It also carries five questions from the delegation proposal §13 that were never sent.

**The landing script.** Built in worktree `owner_land`, 33/33. Not yet landed, not yet
reviewed by Codex, **not to be used** until both. Use the narrower deny entry
`Bash(*scripts/owner_land.py*)`; adopt the longer no-touch list; replace the bodies of
`git_push.sh` and `push_to_claude.sh` with a refusal.

**WP-1 (CPU/GPU generator contract), report `29a6cb9`.** CPU routes read one consecutive
stream; the sieve uses a gap. They coincide only at skip 0. `holdout_quality`, the training
target, is the one route out of line with the rest. The survivor's own skip never reaches
Steps 3 or 6. The CPU batch fallback omits `% mod`. Options A, B, B+, C (deferred), C′, D.
Alpha has **not yet given recommendations**; they are owed with proposal v1.2.

**WP-2 (Step 6 to Chapter 13), report `ea85436`.** No Step 6 pool has been graded against
a real draw on VM101. Options A, B, C; Alpha leans C (standalone grader) then B. The target
draw is per-session under the July rulings. Chapter 7 §16 B1: Step 6 cannot currently read
its survivors.

**Proposal v1.2 is owed.** v1.1 cited none of the hit-rate governance. Map it onto the
records in §5.

## 5. The record that governs this work (found late; start here next time)

| Subject | Where |
|---|---|
| Original pool design and success criteria: Tight 20, Balanced 100, Wide 300 by weighted vote; Hit@100 above 70%, Hit@300 above 90% | project file `High_Probability_Draw_Pools_Plan.pdf` (not in the repo); `build_pools.py` |
| TFM has not completed a real-data cycle; walk-forward method | `docs/TB_RULING_S176_WATCHER_KPI.md` §3, §7 |
| Pool source, ranked unique list, Hit@K rules, evaluation identity; walk-forward waits for Phase 7 | `TB_RULING_S177` Blocker 5; `S178`; `S179` §2.5, §2.6, §6, §8; `PROPOSAL_WATCHER_KPI_GOVERNANCE_STATES_v1_1.md` §7 |
| Per-session rule; results not valid on the next dataset version | `docs/TFM_PROJECT_FACTS_SKILL.md` §2.10b |
| Step 6 known gaps | `docs/CHAPTER_7_PREDICTION_GENERATOR.md` §16–§17; `CHAPTER_7_AUDIT.md`; `CHAPTER_13_AUDIT.md` |
| Seed-domain cursor, COMPLETE, dataset hash is provenance only | `CLAUDE_CODE_INSTRUCTIONS_SEED_DOMAIN_CURSOR_AMENDMENT.md`; `..._R1.md:7-16` |
| Four-session sweep; "Step 1 retired permanently" rejected | `PROPOSAL_S145_R1_Progressive_Empirical_Sweep.md` |
| Step 1's purpose and the regime-shift rerun | `CLAUDE_CODE_REPORT_STEP1_PURPOSE_LINEAGE.md`; fact file `:1128-1136` |
| Run 4 design and claim boundary | `S172_RUN4_PROPOSAL_PINNED_GEOMETRY.md`; `TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md`; `TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md` |
| Plan, lanes, definition of finished | `PROPOSAL_S172_COMPLETION_PLAN_v1_1.md` |
| Agent-launched runs | `PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md` |
| Holdout alignment law; S111 skip choice; survivor skip not carried | `TB_RULING_WINDOW_ANCHOR_SEQUENCING.md:19`; `S111_IMPLEMENTATION_PLAN_FINAL.md`; `CHAPTER_6_AUDIT.md:71-78` |

## 6. Mistakes made today. Do not repeat them.

1. Judged the sieve by state recovery. See §1.
2. Wrote a proposal and three briefs without searching the trail. See §1.
3. Told sessions to open SQLite with `mode=ro`. That writes sidecar files. Use addendum §A.
   The repo already documented it (`tests/gate_s172_prod_shape.py:137-162`).
4. Ran other Claude Code sessions from the main folder during a relay. Claude Code saved
   `.claude/settings.local.json` there and the relay returned BOUNDARY VIOLATION.
   **Nothing else runs from the main folder while a relay runs.**
5. Proposed starting sessions from worktrees to avoid that. Unsafe as stated: no Claude
   Code settings file is tracked, so a worktree session would lose the permission rules.
6. Reused a file name for a revised download; the browser adds `(1)`. Ask Michael to delete
   the old copy first, and have the landing command check the byte size.
7. One session ran on ser8 over ssh. Sessions start on VM101.

## 7. How work moves (unchanged unless Michael says so)

- Michael commits and dual-pushes (`git push origin main && git push public main`). Alpha
  and agents never commit or push. Never `git add -A`.
- Every command block is tagged `# run on ser8`; VM101 is reached by
  `ssh michael@192.168.3.177`; absolute paths under `/home/michael/`. `nohup`, never `tmux`.
- Explain and recommend before commands; hold until he agrees.
- Sessions write only inside `/home/michael/tfm_worktrees/<name>` (detached), create no
  branch or tag, and stop at their report. Landings use a one-line command that commits only
  if the tree holds exactly the expected files.
- Claude Code asks approval for every `git` command; approve read-only ones.
- Agents never update PyTorch, drivers, kernel or OS; never reboot a Proxmox host. The
  fleet is off. Bare-metal Zeus `.127` is frozen. Dashboard port 5002 is fixed.

## 8. Owed

Proposal v1.2 · the Step 1 rerun amendment · A2 volume-limit brief · the hit-rate
instrument brief with Codex's wording (X-FINDINGS-PLAN R2 Q5) · WP-1 recommendations ·
Codex reviews of the gates, the landing script, and the three Stage A reports ·
`SESSION_CHANGELOG_20261113_YARDSTICK.md` · the doc fix list (`docs/DOC_FIX_LIST_20261113.md`)
plus: pitfall doc `:97-98` bound, addendum anchor `:213-223`, the four project PDFs and
X-REVERSE-SIEVE R1 into the repo, the tracked skill copy brought level with the installed
one, LEADS L-4 to L-9.
