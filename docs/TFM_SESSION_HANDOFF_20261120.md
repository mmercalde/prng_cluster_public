# TFM session handoff: 2026-11-20

**DRAFT** (written by the PWC agent, Claude Code, worktree `/home/michael/tfm_worktrees/pwc-backup` at main `0caeaf9`;
not committed). Lines marked **TODO-ALPHA** are facts only the lead or the owner has. Fill them in or delete them
before landing. All times PST. Anything not checked is marked UNVERIFIED or attributed.

## Bottom line

- **PWC full backup, Stages 1 and 2 landed** (`ee39404` at 06:25, `0caeaf9` at 14:35). They were PWC-only: no file
  under `miner/`, none of the ten rig parity-pinned files (`scripts/gate12_parity_gate.py:174-185`), and no launcher,
  manifest or WATCHER file was edited. The proof is the per-file sha tables in
  `docs/drafts/pwc_backup_stage1_v1/README.md` and `docs/drafts/pwc_backup_stage2_v1/README.md` §"The ten rig
  parity-pinned files".
- **PWC still earns no search credit and still refuses a nonzero seed start.** Both are done in the shared launch
  files, which is Stage 3, and Stage 3 is not started (plan only).
- **No runs today.** TODO-ALPHA: the rigs are powered off at the outlets, the GPU PSUs are on, and Zeus rebooted after
  a generator glitch (time, and what was checked afterwards).

## Landed

| Commit | When | What |
|---|---|---|
| `9b6b64c` | 2026-11-19 20:04 | Docs only: the 2026-11-19 handoff, the Run N and Run O notes, the PWC old-test note, the PWC full-backup scope, VM101 headless, selfplay NN status |
| `27eb991` | 2026-11-19 20:37 | Docs only: MCP tool register, a reference copy of the SER8 `cluster_status.py`, VM101 auto-start "marked done" (**that was wrong**, see VM101 below) |
| `ee39404` | 2026-11-20 06:25 | **PWC Stage 1** (19 files): the step-1 work S1–S8, the rig stale-worker kill fix (B5), the TCP launch set (A3 level 1), the `skip_pattern` parser (B6), the repaired old PWC harnesses (A4), the owner rulings file. Beta X-PWC-STAGE1 R1: no objection |
| `0caeaf9` | 2026-11-20 14:35 | **PWC Stage 2** (25 files): the rulings 1 and 4 code, Beta's Stage 1 items, Beta's Stage 2 R1/R2 fixes, the rulings update, the Stage 2/3 brief. Beta X-PWC-STAGE2 R1, R2 |

After `0caeaf9` the lead reported the RANGE-MINER coordinator suite at 63/63 on the committed tree (lead's report,
not re-run by this agent). Before commit, that suite's Gate 22 was red only because of uncommitted PWC `.py` files
(`docs/drafts/pwc_backup_stage2_v1/README.md`, before/after table).

## What PWC does now (main `0caeaf9`)

- **Variable-skip passes run** (ruling 1). There is no 0.5 default. Every job carries the trial's directional
  threshold as both `min_match_threshold` and `phase2_threshold`, as RANGE-MINER does. A variable-skip pass checks
  what the worker reports it used, at 4 places because `sieve_filter.py:597-599` rounds the report, and fails on a
  mismatch or a missing report. It refuses on the SSH route. `persistent_worker_coordinator.py` `run_sieve_pass`;
  record and assert functions near `:548-590`.
- **No early stop** (ruling 4): phases 1–2 always, 3–4 whenever `test_both_modes`.
- **All five hybrid strategies** (default rule; S147 Q2's single `balanced_hybrid` is removed). F5 still needs the
  owner's confirmation.
- **Refuses a window offset other than 0** (owner ruling 2026-11-20 08:30, `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:44`):
  PWC still sends one `offset` that `sieve_filter.py` uses both as the window start and as the kernel's generator
  phase. Most Optuna-sampled PWC trials, and Run 4's anchor 25, refuse until Stage 2b lands.
- **Process checks prove they can see processes** (VIR-1, VIR-5):
  - rig kill, S156 find/reap and the VM101 local kill all carry a positive control;
  - `pgrep`'s exit status is read exactly;
  - an unavailable check never reports clean.
- **TCP transport:**
  - only worker ids this run launched are served;
  - revoking an id drops its live connection;
  - a launch with no PID is not counted;
  - only the connection that owns an id cleans up under it, and timeout eviction follows the same rule;
  - job ids are unique per pass, and a result is accepted only for the active lease on its own connection;
  - a refused worker exits (code 43) instead of retrying forever.
- **Per-pass accounting:** completion is recorded per pass, along with the transport each pass ran on. A pass with a
  failed, missing or timed-out chunk is marked incomplete.
- **Still true:** PWC earns **no credit**: `window_optimizer_integration_final.py:2002-2006` stamps refusal evidence.
  The **seed-start refusal** stands at the integration (`:1960-1963`) and in `window_optimizer.py:678-712`. The
  pipeline call site passes no dial address and a chunk cap of 2,000,000, so **PWC with rigs refuses at startup from
  the pipeline** (`:1968-1986`).
- **Nothing was redeployed to the rigs** (none was on). Which PWC bytes they carry is UNVERIFIED (rigs off; the
  PWC parity gate `scripts/pwc_parity_gate.py --transport tcp` checks it). If a rig carries a pre-Stage-2 worker, a
  refused stale worker there still retries every 5 s. The redeploy steps are in `docs/drafts/pwc_backup_stage2_v1/README.md` §"What a later rig
  redeploy would involve".

Tests:
- `tests/test_pwc_backup_stage2.py`: 26 gates, 29 mutants killed.
- `tests/test_pwc_stage1.py`: 9 gates, 8 mutants.
- `tests/test_pwc_backup_step1.py`: 22 gates, 16 mutants.
- `test_persistent_worker_harness.py`: 25/25.

**Before/after:** 72 existing suites. 68 identical; the 4 that differ are explained in the Stage 2 README. All are CPU
only. `tests/test_s172_staging_partb.py` was not run, because it writes into the live `~/miner_staging`
(`tests/test_s172_staging_partb.py:599-609`).

**Not measured:** no PWC-produced pool has a hit rate against `k/1000`.

## Owner rulings (`docs/PWC_BACKUP_OWNER_RULINGS_20261119.md`)

- 2026-11-19 20:57:
  1. lift the variable-skip block once the threshold matches RANGE-MINER;
  2. PWC runs earn credit;
  3. drop the seed-start refusal;
  4. phases follow RANGE-MINER.

  Default rule: PWC does what RANGE-MINER does (`:14`).
- 2026-11-20 08:28: in the two shared suites, only the named PWC checks change (`:43`).
- 2026-11-20 08:30: the offset picks the draw window only, the generator phase is 0, PWC moves to RANGE-MINER's
  method, and PWC refuses offset ≠ 0 until then (`:44`).
- 2026-11-20 12:34: Beta's R2-Q1 is fixed as Beta specified; Stage 2 lands with no third round (`:45`).
- The skip-range question is still the owner's (`:30`).

## Beta exchanges

| Thread | Round | Relay verdict | Positions | Outcome |
|---|---|---|---|---|
| X-PWC-STAGE1 | R1 (05:42) | REVIEW RECEIVED | Q1, Q2, Q5, Q7 NO OBJECTION; Q3, Q4, Q6 SUGGESTION | Landed in `ee39404`. Q3/Q4 (fail-open probe, revoked connection, refused worker retrying) fixed in Stage 2 |
| X-PWC-STAGE2 | R1 (10:25) | REVIEW RECEIVED | Q1, Q2, Q4, Q5, Q8, Q9 NO OBJECTION; **Q3 NEEDS EVIDENCE** (five strategies on the rigs); **Q6 OBJECTION LIVE** (reconnect race); Q7 SUGGESTION; finding 2 LIVE (late result under the next phase's job id) | Q6, finding 2 and Q7 fixed in R2 |
| X-PWC-STAGE2 | R2 (11:59) | REVIEW RECEIVED | R2-Q2, Q3, Q4 NO OBJECTION; **R2-Q1 OBJECTION LIVE** (timeout eviction at the boundary); finding 1 SUGGESTION | Fixed as Beta specified; no third round (owner, 12:34). Closing note `docs/exchange/to_beta/X-PWC-STAGE2_R3_ALPHA_CLOSING_NOTE.md`, not sent |

The ledger rows are `docs/exchange/LEDGER.md:41-43`. `DEFERRED.md` is unchanged (no THEORETICAL objections).

**Label correction:** the five-strategies NEEDS EVIDENCE is **R1-Q3** of X-PWC-STAGE2. R2-Q3 was the refusal exit,
NO OBJECTION. The rulings file's update (`docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:46`) said "R2-Q3"; corrected in
the pwc-backup worktree on 2026-11-20 (uncommitted). It stays open for the owner's Stage 4 rig sessions.

## In flight (not landed)

- **Stage 2b**, `/home/michael/tfm_worktrees/pwc-offset` (another agent; read only here). It is uncommitted at base
  `ee39404`, and as seen at 2026-11-20 ~15:00 by `git status` it changes `persistent/pwc_worker_service.py` and
  `sieve_filter.py`. New files there:
  - `scripts/prove_unchanged_outside.py`;
  - `scripts/pwc_sieve_filter_import_proof.py`;
  - `tests/test_pwc_stage2b_anchor_phase.py`;
  - `tests/test_prove_unchanged_outside.py`;
  - a draft README and a draft Beta request.

  This is the worker side of the window/generator split. Status beyond that: UNVERIFIED.
- **Stage 3 plan**, `docs/drafts/pwc_backup_stage3_v1/PLAN.md` (this worktree, uncommitted). It lists the shared
  launch files and functions with `file:line`, which of them RANGE-MINER's launch reaches, the proof plan, and 16 owner
  questions.
  - **New finding:** a second copy of the self-killing `pkill` sits in the integration file's parallel path
    (`window_optimizer_integration_final.py:2897-2914`, any engine with `n_parallel > 1`).
  - **Brief error corrected:** the split needs a worker edit as well as `sieve_filter.py`.
- **Stale documentation list**, `docs/drafts/pwc_backup_stage3_v1/STALE_DOCS.md` (uncommitted).
- TODO-ALPHA: what agent 2's timing-sensitive full pass is, and its result.

## Open items recorded late on 2026-11-20

- **Parallel trials (n_parallel > 1): a separate job, after the PWC backup is finished** (owner "Yes", 2026-11-20
  20:24 PST; `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md`).
  - n_parallel > 1 is SWITCHED-OFF-BY-RULING on the miner path (`docs/CHAPTER_12_WATCHER_AGENT.md:1220`,
    `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_7_SOAK.md:20`, `docs/BACKLOG.md:298-300`).
  - Re-enabling it needs a checkpoint and memory design for partition workers, plus Beta review.
  - That job also fixes the two PWC items left alone in Stage 3: S3-6, the self-killing `pkill` with hardcoded
    bare-metal IPs in the integration file's parallel path; and S3-7, the PWC dial address not passed to partitions.
  - Scope (owner, 2026-11-20 20:37 PST, verbatim: "Or better half space seed search- that IS tested"): the job
    targets **half-range** trials, the size already run one trial at a time on RANGE-MINER (Runs P and Q,
    `docs/TFM_SESSION_HANDOFF_20261119.md:19-20`, `:22`; L and M in `docs/TFM_SESSION_HANDOFF_20261118_ADDENDUM.md`),
    not the full range. Half range is tested with n_parallel 1 only (`gate12_launch.sh:472`); parallel at half range is
    still **untested**.
  - **Priority: LOW** (owner, 2026-11-20 20:41 PST, verbatim: "I agree. Since we can search the complete seed space in
    two passes parallel runs doesn't make a lot of sense. Now.if it's finding multiple windows that's different").
    Before any design, Alpha pulls from the Run P and Q logs the fraction of trial time that is not GPU sieving.
  - Not designed.
- **New idea, UNVERIFIED (Alpha's): "several windows in one pass".**
  - The idea: one pass over the seeds tests several windows on the same GPU, sharing the per-seed stream generation.
  - Not checked against the kernels or the prior record. The first step is a record search and a kernel search.
  - It would touch pinned kernel files, so it needs Beta review and a rig redeploy.
  - It comes after the PWC backup. Not designed.

## Open owner questions

The full list is in `docs/drafts/pwc_backup_stage3_v1/PLAN.md` §D (Q-1 … Q-16). In short:
- who does the coordinator side of the split;
- the PWC dial address (a required flag, recommended);
- PWC chunk-size defaults outside the stable envelope;
- the parallel-path `pkill` fix;
- the credit rule (complete, all-TCP, thresholds verified);
- the SSH route (F1);
- the HSA variables (F2);
- cache records (F3);
- the skip range (F4): S3-13 withdrawn 21:32-21:35; PWC keeps the trial's range; the RANGE-MINER fix is scoped in `docs/drafts/pwc_backup_stage3a_v1/SCOPE_MINER_SKIP_RANGE_FIX.md`;
- five strategies (F5);
- the prune flag (F6);
- the exact threshold report;
- where `prove_unchanged_outside.py` lands;
- the manifest's default route being PWC (already recorded as owed after Phase 7, `docs/TODO_CONSOLIDATED.md:93`);
- the per-run launch token;
- the PWC ready gate on the pipeline route: **settled 2026-11-21.** The 2026-08-04 owner ruling stands (not restored).
  Stage 3 keeps `b2efdef`'s call shape, with effective minimum 1 (Beta X-PWC-STAGE3 R1 3b-Q7).

## VM101 and Zeus

- **Auto-start:** TODO-ALPHA. `qm set 101 --onboot 1` was actually set on 2026-11-20 at 12:57. The 2026-11-19 note
  said it was done; that was wrong. Corrected in `docs/VM101_HEADLESS_STARTUP_20261119.md`.
- **Discard:** TODO-ALPHA. Discard was enabled (which disk, and whether `fstrim` has run since).
- **Disk:** TODO-ALPHA. Four unused LLM model files were deleted (names and paths); the thin pool went from 85 to 58
  percent (where it was read).
- VM101 was restarted for disk maintenance during the day; this worktree was intact afterwards (lead's report).

## State at close

- TODO-ALPHA: rigs, Zeus, VM101 at close.
- Repo: main `0caeaf9`. This worktree: HEAD `0caeaf9`. Untracked here are `data/` (suite output, not to land) and this
  handoff, the VM101 correction and the two Stage 3 drafts.
- Live ledger at `/home/michael/miner_staging`: not touched by the PWC agent. In every suite run its mtime was
  unchanged (`logs/pwc_stage2_v1/after_r3/staging_mtime_per_suite.tsv`, ignored path).

## Not done

- **PWC credit, the seed-start refusal, the dial-address and sizing plumbing, and fleet fail-closed:** Stage 3
  (shared launch files). Not started.
- **The window/generator split:** Stage 2b in flight; the coordinator side is not started.
- **Any rig work:** no redeploy, no parity gate run on rigs, no five-strategy measurement (Beta R1-Q3).
- **The per-run launch token** (worker edit). **HSA variables. Cache records.**
- **A hit-rate measurement of PWC-produced pools.**
- **The session changelog for today's PWC work:** none written. Repo rule SR-2 (`docs/PROJECT_FILE_CATALOG.md:90-97`):
  name it `SESSION_CHANGELOG_20261120_<TOPIC>.md`, with no S-number. `CLAUDE.md:99` still says `..._SN.md`; that is
  listed in STALE_DOCS.
- **The stale PWC statements** in `CLAUDE.md`, the catalog, chapters, the fact file and registers are not edited;
  they are listed in STALE_DOCS for the owner.
