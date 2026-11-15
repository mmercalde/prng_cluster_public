# CLAUDE CODE REPORT — RUN 4 LAUNCH PREPARATION

**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_RUN4_LAUNCH_PREP_20261114.md` · **Session:** Claude Code
(Team Alpha), VM101, as `michael`, 2026-11-14 evening · **Base commit:**
`347b0369922076fa280efa46be3b1f03454d71ba` (`347b036`, the brief's own commit; parent `80d2bb4`).
**Worktree:** `/home/michael/tfm_worktrees/run4-prep` (detached, clean at HEAD). **Launched
nothing. No rig contacted. No GPU job. No live-state write. Nothing committed.**
**Review class:** document. No relay.

## For the owner, in plain words

**Is the committed tree green? Yes, to the extent its reports were.** 55 suites, one at a time, in a clean worktree at `347b036`. **No suite is red tonight that was green in its report.** Two that were red only because their code was uncommitted are now green: Gate 22 (phase 4, 63/63) and the cleantree gate's `W-NO-WEAKENING` (31/31). The pre-launch check is 10/10 and prints Run 4's command with the seven pins and `--seed-start 2147483648 --max-seeds 2147483648`. The routing suite is 21/22 with only the frozen gate INCOMPLETE, as recorded. R-1 44/44, MP-1 38/38. The other reds are the same pre-existing reds the reports list, gate for gate. Live state was unchanged by every suite.

**What is in the way of launching tomorrow:** one thing, and it is a decision, not a defect.

- **There is no committed command that launches Run 4.** The launch harness, `gate12_launch.sh`,
  has Gate 12's parameters written into it, with no pins. The pins enter only through a different
  `--params` value, and the committed pre-launch check prints that value. Nothing in the record
  joins the two. If the harness is run unchanged after Run 4's search is opened, WATCHER should
  refuse before any GPU work, because the pins differ from the search. **You decide how to launch
  (checklist D-1, step 13).** My recommendation: copy the harness outside the repository, change
  only its `--params` block to the checklist's step-11 value, check the diff, and run the copy.

**Nothing else found in the way.** Measured tonight, read-only:
- **The dataset has not moved.** The pointer still names attempt 9's file,
  `daily3-20260801T145551443433Z-513648160d35.json`, sha256 `513648160d35…68f6`. So the "different
  dataset" refusal (proposal §2.6) does not apply. The scraper unit is disabled and there is no
  cron.
- History: one certified row, `[0, 2³¹)`, attempt 9's. No search tables exist yet; no search is
  open. Halt file absent. Staging directory holds only the ledger and old copies; `-wal` is 0
  bytes. The Optuna folder (104 studies) is not resumed unless `resume_study` is set, and it is
  false by default.
- The rigs should differ from HEAD in exactly one governed file, the coordinator. That is from the
  record; the parity gate tomorrow is the measurement.

**What you must decide:**
1. **D-1, how to launch** (above).
2. Nothing else is yours tonight. Already decided and not reopened: the range `[2³¹, 2³²)`, the cap
   25,000, the claim boundary.

**Two things to know, both NEW, neither a blocker:**
- **The netconsole receiver is already running** on VM101 as `netconsole-listener.service`. Don't
  start a second one; the runbook doesn't mention the service.
- **A possible no-credit outcome, untested.** Step 1 refuses credit if the miner ledger's `-wal` file
  is non-empty when it reads the stripe rows. I checked on a throwaway file that a read-only reader
  (the concurrency sampler polls this way every 2 s) can leave `-wal` non-empty after the last writer
  closes. Whether that timing happens in Run 4 is unknown. If it does, the run computes, earns no
  credit, and loses nothing. Checklist step 14 says what to look for. Don't open the ledger with
  other tools during the run.

## Null results and limits first

- **The rigs were not reached.** Every statement about rig state is from the record, last measured
  2026-08-22. Parity, GPUs, the dataset on the rigs and netconsole senders are all unmeasured
  tonight.
- **No launcher for the pinned run exists in the record** (searched `*.sh`, `scripts/*.sh`,
  `~/dashboard_work/`, the governance trail). Step 13 of the checklist is `UNAVAILABLE` as a
  committed command.
- **No clean mid-run stop is recorded.** The halt file cannot stop a running Step 1
  (`agents/watcher_agent.py:3068`, `:3095` check it only before each step).
- **The ledger `-wal` risk** is shown on SQLite semantics only, not on the miner.
- **T1 needed three runs** because my worktree lacked gitignored files the earlier sessions had (the dataset files and `optuna_studies/`). Those reds were my setup, not the code; the T1 section says exactly which and how they were cleared.

## Prior record

### Searches run

All from `/home/michael/distributed_prng_analysis/docs` at `347b036` unless stated. Scope `S` =
`PROJECT_FILE_CATALOG.md TB_RULING_* PROPOSAL_* CHAPTER_* *_AUDIT.md TODO_CONSOLIDATED.md
BACKLOG.md LEADS.md exchange` (`TB_RULING_*` includes `TB_RULING_REQUEST_*`).

1. `/bin/grep -rlE "Run 4|RUN4|Run-4" $S` — 37 files; read: the four `TB_RULING_RUN4_*`, the
   completion plan, the delegation proposal, the search proposal v1.1, `TODO_CONSOLIDATED.md:79-87`,
   `BACKLOG.md:515-524`, `LEADS.md` L-3 to L-5 headers, X-RUN4-RANGE R1.
2. `/bin/grep -rlE "boot-proxmox" $S` — completion plan, delegation proposal, findings resolution
   v1.1. Plus `docs/RUNBOOK_NETCONSOLE_REARM.md`, `CLAUDE.md`, the fact file §6 and the memory file
   `rig-boot-selector-proxmox-recovery.md`.
3. `/bin/grep -rlE "netconsole" $S` — 10 files; read the runbook in full, `LEADS.md` L-3,
   `TODO_CONSOLIDATED.md:81`.
4. `/bin/grep -rlE "launcher|run4_launch|gate12_launch" $S` and `ls *.sh scripts/*.sh`,
   `/bin/grep -l "warm_start_window\|step1-search-open" *.sh scripts/*.sh` (no hit),
   `ls ~/dashboard_work` (host).
5. `/bin/grep -rlE "\-wal|immutable=1" $S` — 12 files; read every hit line in the Beta reviews
   (X-MINER-STALL `:132`, X-MINER-START `:100`, X-RUN4-RANGE `:106`, X-0001 `:165-183`). **None
   records a reader-left `-wal` at evidence time.**
6. `/bin/grep -rlE "redeploy|parity gate|30/30" $S`, `/bin/grep -rln "git archive" docs/`,
   `/bin/grep -rn "scp " docs/ --include=*.md`, `/bin/grep -n "parity\|30/30\|redeploy"
   docs/SESSION_CHANGELOG_202608{22..28}*.md`.
7. Code: `git log --since=2026-08-17` and `git diff --stat 48a8705 HEAD` over the ten governed files;
   `agents/watcher_agent.py` argparse and `step1_search_cli`; `scripts/survivor_volume_gate.py` main;
   `tests/gate_s172_prod_shape.py` usage; `window_optimizer_bayesian.py:655-760` (study selection).

### Read for this session — the documents Alpha had not read

| Document | Read | What it adds or corrects |
|---|---|---|
| `docs/PHASE6_PREREQS.md` | in full, 441 lines | REV5, 2026-08-02. Rigs are deployment targets; deploy = `git archive` export + `scp`, never `git pull`; parity is by digest. Its 17-file `miner/`+`utils/` parity is **superseded** by the ten-file parity gate (`scripts/gate12_parity_gate.py:174-185`). Its "host kernel-log observability is an accepted exception" stands |
| `docs/RUNBOOK_NETCONSOLE_REARM.md` | in full | Sender commands are `[RECONSTRUCTED]`; proof is three `NCPROOF` lines. **It does not know the receiver is a systemd service** (measured tonight) |
| `docs/CLAUDE_CODE_INSTRUCTIONS_MANUAL_FLEET_LAUNCH.md` | in full | **Its order (coordinator first, then workers) is superseded** by the attempt-6 parked-fleet order (`scripts/launch_fleet_manual.sh:56-72`; `gate12_launch.sh` §2-§5). Do not follow it for Run 4 |
| `docs/PHASE7_PREREQUISITES.md` | headings and status line | Status as of 2026-07-30 (D6.2, D6.3). Historical; nothing in it gates Run 4 |
| `docs/S172_PHASE_7_PREREQ_REPORT.md` | `:495-560` | The recorded deploy commands (`:520-526`) used for checklist step 6 |
| `docs/TEAM_ALPHA_PHASE7_LAUNCH_NOTICE.md` | `:60-126` | The soak's launch configuration and the binding `UNAVAILABLE` wording for GPU-fault absence. The same host-log surface is unavailable for Run 4 |
| `gate12_launch.sh` | in full, 434 lines | **Its `--params` block (`:377-383`) is Gate 12's, with no pins.** The rest is the certified launch order |
| `scripts/gate12_parity_gate.py` | `:150-200` | The ten-file pin |
| `scripts/gate12_gpu_gate.py` | verdict lines `:180-208` | Literal pass line |
| `scripts/gate12_concurrency_sampler.py` | `:73`, `:203` | Opens the ledger `mode=ro` per sample |
| `scripts/gate12_sentinel_gate.py`, `scripts/gate12_worker_liveness_gate.py` | **not read**; their invocations in `gate12_launch.sh` only | — |
| `scripts/launch_fleet_manual.sh` | `:1-80` | The parked-fleet order |
| `/home/michael/dashboard_work/R1_R4_FLEET_DEPLOY_RECORD.md` (host, not in the repo) | in full | The attempt-9 deploy procedure. Its archive/install scripts were in rig `/tmp` and are not recorded |

### Documents that already decide, authorise, schedule or record the subject

The brief's table stands; I re-read each at the lines it gives. Added:

| Document | What it records | Label |
|---|---|---|
| `docs/exchange/from_beta/X-RUN4-RANGE_R1_BETA_REVIEW.md:41`, `:49-51` | Beta: no objection to Run 4 on `[2³¹, 2³²)`; record both intervals; no phase-only comparison | ALREADY RECORDED |
| `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:148` ("full 2³¹ seed domain used by Attempt 9") | The original ruling held the seed domain constant | ALREADY RECORDED; settled by X-RUN4-RANGE Q1 and proposal §2.9. Not reopened |
| `docs/SESSION_CHANGELOG_20260822_BRIEF_I_PRODUCTION_SHAPE_REPAIR.md:16-18`; `docs/S172_BRIEF_I_PRODUCTION_SHAPE_CLASSIFICATION_REPORT.md:62`, `:341` | All ten governed files deployed at `48a8705`, 30/30 | ALREADY RECORDED; it is the basis for "only the coordinator differs" |
| `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:21` (owner ruling 3) and `TODO_CONSOLIDATED.md:80` | Power caps and off-host power telemetry: dropped | ALREADY RECORDED (`TODO_CONSOLIDATED.md:80` still lists it; the ruling drops it) |
| `docs/BACKLOG.md:521-522` | Don't change VM101's memory before Run 4 | ALREADY RECORDED |
| `docs/CLAUDE_CODE_REPORT_DELEGATED_RUN_GATES.md:322-360` | The environment-baseline capture and an example `RUN4.auth` for agent launches | ALREADY RECORDED; does not gate an owner launch |
| `docs/CLAUDE_CODE_REPORT_STEP1_SEARCH_RECORD_STAGE1.md:23-26`, `:410`, `:425-428` | One chunk per search in stage 1; close-then-open and `repeat_of`; the CLI smoke | ALREADY RECORDED |
| `docs/LEADS.md` L-3, L-4, L-5 | Netconsole follow-ups open; L-4 and L-5 need their first fleet run | ALREADY RECORDED. **Status text at `:283` and `:360` still says "NOT YET COMMITTED"**; the repairs landed in `89be3ee` and `80d2bb4` after `0a07ace` last edited the file. A doc correction is owed; not made (the brief forbids editing existing files) |

### Labels for what this session raises

| Item | Label |
|---|---|
| The checklist itself | ALREADY RECORDED steps, assembled |
| No committed Run 4 launcher with the pins (D-1) | NEW as a statement of the gap; the parts are ALREADY RECORDED (`gate12_launch.sh`, the pre-launch check) |
| Netconsole receiver already a systemd service | NEW (operator note) |
| Reader-left `-wal` can refuse credit (S7) or blank `G-PROD-SHAPE` §B | NEW (untested risk; SQLite behaviour measured on a throwaway file) |
| The halt file cannot stop a running Step 1 | NEW as an operator note; it is what the code does (`agents/watcher_agent.py:3068`, `:3095`) |
| `LEADS.md` L-4/L-5 status text stale | NEW (doc correction owed) |
| Anything that **changes a ruling** | none |

### Not searched

The rigs and the Proxmox hosts (off tonight); ser8 archives; the private repository; project files
not in the repository other than `~/dashboard_work/R1_R4_FLEET_DEPLOY_RECORD.md` and the other
sessions' worktrees named below; the Settings copy of the fact file.

## T1 — the committed tree, suite by suite

Clean worktree at `347b036`, CPU only (`CUDA_VISIBLE_DEVICES=""`), one suite at a time, each between
two live-state snapshots. Runner: `/home/michael/tfm_worktrees/run4-prep-tools/run_suites.sh`. Logs:
`/home/michael/tfm_worktrees/run4-prep/data/run4_prep/t2/` (comparison: `t2/compare.log`). The suite
list is the union of what the four reports name, plus the two delegated-gate suites the volume
report names. "Report" is the most recent report's figure; the comparison is of the **set of
non-passing gate names**, not only the count.

**Two runs, and why.** Run `t1` was stopped after 25 suites. My fresh worktree lacked three
gitignored files that the earlier sessions had copied into theirs (`daily3_current.json`,
`daily3.json`, `daily3-20260801T145551443433Z-513648160d35.json`), so `test_s172_miner_seed_start`
and `test_step1_search_record_stage1` failed with `PointerResolutionError: no pointer manifest at
…/run4-prep/daily3_current.json`. I copied the three files read-only from the main tree (same
sha256), stopped `t1` by its recorded PIDs, and ran all 55 again as `t2`. In `t2` one more suite
failed for the same kind of reason: `test_s172_d6_2_checkpoint_reconciliation` reported `RESULT:
INCOMPLETE` because the worktree had no `optuna_studies/` directory (gitignored;
`window_optimizer_bayesian.py:23` stores studies beside the module). With an empty
`optuna_studies/` created, its rerun (`t3`) gave `RESULT: PASS`. **All three were setup gaps in
my worktree, not code faults; the live tree has all of these files.** No other suite's failing
set changed between `t1` and `t2`.

**No suite is red tonight that was green in its report.**

| # | Suite | Tonight | Report | Difference explained |
|---|---|---|---|---|
| 1 | `test_s172_phase4_coordinator` | **63/63** | 62/63 (stall, seed-start, stage-1) | **Gate 22 green now.** It was red only on uncommitted files; the work is committed. No allowlist widened |
| 2 | `test_gate12_cleantree_admission` | **31/31** | 30/31 (stage-1, `W-NO-WEAKENING`) | Green now; it was red only while S7/S8 were uncommitted (stage-1 §8.1) |
| 3 | `tests/run4_prelaunch_provenance_check.py` | **10/10, `RUN4_PRELAUNCH_PROVENANCE_OK`** | 10/10 | Same. Built command below |
| 4 | `test_s172_run4_routing_patch` | **21/22**; only `G-UNPINNED-IDENTICAL` INCOMPLETE | 21/22 | Same, line for line (gate lines diffed against the stall log) |
| 5 | `test_s172_r1_drain_remedy` (R-1) | **44/44** | 44/44 | Same |
| 6 | `test_s172_mp1_drain_attribution` (MP-1) | **38/38, PASS** | PASS (Beta reran 38/38) | Same |
| 7 | `test_s172_admission_stall` | 11/11 gates, 4/4 mutants | 11/11, 4/4 | Same (2,267 s) |
| 8 | `test_s172_small_repairs_part3` | 3/3, 3/3 | 3/3, 3/3 | Same |
| 9 | `test_s172_miner_seed_start` | 10/10, 11/11 | 10/10, 11/11 | Same in `t2` (see above for `t1`) |
| 10 | `test_step1_search_record_stage1` | 39/39, 33/33 | 39/39, 33/33 | Same in `t2` |
| 11 | `test_step1_bayes_storage_path` | PASS | 3/3, 4/4 | Same |
| 12 | `test_survivor_volume_gate` | 21/21 | 21/21 | Same |
| 13 | `test_run_authorization_gate` | 18/18 | 18/18 | Same |
| 14 | `test_env_freeze_gate` | 17/17 | 17/17 | Same |
| 15 | `test_s172_admission_liveness` | 16/16 | 16/16 | Same |
| 16 | `test_s172_staging_backpressure` | 52/52 | PASS | Same |
| 17 | `test_s172_staging_partb` (probe case replaced) | 23/24, `G-ROUTE-2 SKIPPED-BY-BRIEF` | 23/24, same | Same; the live staging probe was not run, by the brief |
| 18 | `test_s172_f1_f2_active_lease` | 16/16 | 16/16 | Same |
| 19 | `test_s172_f1_lease_origin` | 18/18 | PASS | Same |
| 20 | `test_s172_admission_binding` | 11/20 | 11/20, same gates | Same set (only run ids differ) |
| 21 | `test_s172_attempt6_remediation` | 76/78 (RXP-3/6, RXP-3/7) | 76/78, same | Same |
| 22 | `test_s172_defect_a_transport_recovery` | 29/29 | 29/29 | Same |
| 23 | `test_s172_window_anchor_brief_i` | 25/26 (`G-SEP-1`) | 25/26 | Same |
| 24 | `test_s172_window_anchor_brief_i_mutants` | PASS | PASS | Same |
| 25 | `test_s172_phase5_d0` | 0/12 | 0/12 | Same |
| 26-28 | `phase5_d1_engine`, `d1_workflow`, `d2_directional_uniqueness` | FAIL: Phase 3 `expected '17/17 gates green'` (it prints 18/18) | FAIL | Same pre-existing red. The stall report (`:371`) says these stopped at Phase 4's transitional Gate 22; with Gate 22 green they now reach Phase 3's pre-existing check, exactly as that line predicts |
| 29-32 | `phase5_d3_0`, `d3_25`, `d3_5`, `d3_columnizer` | 10/10, 13/13, 60/60, 10/10 | PASS | Same |
| 33 | `test_s172_phase5_d4_serial_backend` | 4/8 | FAIL | Same set |
| 34 | `test_s172_phase5_d5_process_sharded` | FAIL, same Phase 3 check | FAIL | As 26-28 |
| 35 | `test_s172_phase5_d6_production_adapter` | 0/9, positive control fails | same | Same (fact file §2.51 item 5) |
| 36 | `test_s172_phase5_d6_threshold_path` | 1/17, positive control fails | same | Same |
| 37 | `test_s172_phase6_p05_dataset_authority` | 37/37 | PASS | Same |
| 38 | `test_s172_process_sharded_import_gate` | 7/7 | PASS | Same |
| 39 | `test_s172_threshold_propagation` | 3/5 · 1 FAIL · 1 UNAVAILABLE | same | Same |
| 40-42 | `h1h2_instrumentation`, `elapsed_roundtrip`, `d6_integration_repair` | 62/62, PASS, PASS | PASS | Same |
| 43 | `test_s172_d6_1_flush_durability` | 15/15 | PASS | Same |
| 44 | `test_s172_d6_2_checkpoint_reconciliation` | **PASS (`t3`)**; INCOMPLETE in `t2` | PASS | `t2`: no `optuna_studies/` in the worktree (above) |
| 45-47 | `d6_liveness_gate`, `resolved_execution_set`, `phase1_scaffolding` | PASS, 34/34, 6/6 | same | Same |
| 48 | `test_seed_domain_cursor_amendment` | 41/41 | 41/41 | Same |
| 49 | `test_gate_prod_shape_scoping` | 33/33 | PASS | Same |
| 50 | `test_gate12_gpu_gate` | 9/9 | PASS | Same |
| 51 | `test_chapter1_p0_corrections` | 11/12 (mutant M2) | 11/12 | Same (stage-1 §8.3) |
| 52-54 | `watcher_followup_repairs`, `watcher_safety_repairs`, `watcher_llm_integration` | 17/17, 21/21, 13 passed | same | Same |
| 55 | `test_owner_land` | 33/33 | 33/33 | Same |

**The command the pre-launch check builds, tonight, verbatim** (63 tokens; the stub dataset path
`daily3.json` and the stub search id are the fixture's, not Run 4's):

```
python3 window_optimizer.py --lottery-file daily3.json --strategy bayesian --max-seeds 2147483648 --prng-type java_lcg --output optimal_window_config.json --test-both-modes --trials 1 --trse-context trse_context.json --enable-pruning --n-parallel 1 --worker-pool-size 25 --seed-cap-nvidia 5000000 --seed-cap-amd 2000000 --seed-start 2147483648 --pwc-transport tcp --min-workers 24 --use-range-miner --miner-stripe-size 67108864 --miner-substripes 8 --staging-dir /home/michael/miner_staging --staging-workers 4 --staging-queue-depth 2 --staging-capacity-timeout 600.0 --staging-high-water-bytes 17179869184 --step1-search-id s1s-74c4b07f3d72110ae3aeac48577d91ec2f69368ee03ec99c633377e336cd4c70 --warm-start-fwd-thresh 0.71 --warm-start-offset 25 --warm-start-rev-thresh 0.47 --warm-start-session-idx 1 --warm-start-skip-max 99 --warm-start-skip-min 6 --warm-start-window 12
```

The CLI it models (its own printout):
`python3 agents/watcher_agent.py --run-pipeline --start-step 1 --end-step 1 --params '<the
step-11 JSON in the checklist>'`.

## T3 — live state, read-only

Every read was `stat`/`ls`/`cat`/`sha256sum` or SQLite `?mode=ro&immutable=1` (binding addendum §A,
`docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md:29-31`). I did not run
`--step1-search-status`: `DistributedPRNGDatabase()` opens the live database read-write
(`database_system.py:46-53`), and the search ledger would create its tables on first use.

**The history ledger** — `prng_analysis.db` table `certified_coverage`, opened immutable. 1 row:

```
coverage_id c6f28aedf7af12cde2a03595ac7647d8d7b30d6c98b3e7f6aae0841df424e030
run_id step1_java_lcg_0 · prng_base java_lcg · skip_modes_executed constant,variable
seed_start 0 · seed_end_exclusive 2147483648 · seed_domain_contract v1.1-stratum
dataset_sha256 513648160d356617c22a1e543ae1c9c65f4921ec21718989308b1f70c00768f6
repository_commit e9ca800ae65b44bc555f1402a9102932ba2e72ca · publication_status CERTIFIED
generation_id gen-20260818T013219550933Z-step1_java_lcg_0 · recorded_at 2026-08-18T01:32:19Z
```
**As expected: attempt 9, `[0, 2147483648)`.** The accumulator's `current` is the same generation,
`row_count 0`, `l2_winner_count 0`, `repository_tree_clean true`
(`.s172_accumulator/current/provenance.json`). Attempt 9 published zero bidirectional survivors
(its log: `Bidirectional survivors: 0`, `logs/gate12_20260817_181819.log:707`).

**The search ledger.** The live database has seven tables: `cache_results`, `certified_coverage`,
`exhaustive_progress`, `lottery_draws`, `search_jobs`, `sqlite_sequence`, `step1_trial_history`.
**No search table exists, so no search is open.** Opening Run 4's search (checklist step 11) will
be the first write of the search schema. `prng_analysis.db` was last modified 2026-08-21 21:29; no
`-wal`, `-shm` or `-journal` beside it.

**The dataset — unchanged since August.**

| Fact | Value | Command |
|---|---|---|
| pointer `daily3_current.json` | `version_id daily3-20260801T145551443433Z-513648160d35`, `sha256 513648160d356617c22a1e543ae1c9c65f4921ec21718989308b1f70c00768f6`, 18,068 records, last draw 2026-02-26 midday, lineage `daily3-combined-L001`, published 2026-08-01T14:55:51Z | `cat daily3_current.json` |
| the immutable file | sha256 equal to the pointer's | `sha256sum daily3-20260801T145551443433Z-513648160d35.json` |
| versions present | 1 (`ls daily3-*.json`) | |
| legacy alias `daily3.json` | same sha256 | `sha256sum daily3.json` |
| scraper | `daily3scraper.service` disabled, inactive; no crontab | `systemctl is-enabled/is-active`, `crontab -l` |

**It is attempt 9's dataset.** §2.6's different-dataset refusal does not arise.

**Halt file:** `/tmp/agent_halt` absent (`ls`). The path is `agents/watcher_agent.py:384`.

**Live staging directory** `/home/michael/miner_staging` (8.4 MB, `ls -la`): `miner_ledger.db`
(6,434,816 B, 2026-08-27), `miner_ledger.db-shm` (32,768 B, 2026-11-11), **`miner_ledger.db-wal`
0 bytes**, the three `*.pregate12_20260809` copies and `threshold_provenance.json` (2026-08-22). **No
staged shard files left from earlier runs.** A 0-byte `-wal` does not trigger the evidence
reader's refusal (`window_optimizer_integration_final.py:1449-1456` refuses only on a size above 0).

**Optuna** `optuna_studies/`: 104 study files, newest `window_opt_1787448179.db` (2026-08-22 18:42).
Step 1 resumes one only if `resume_study` is true (`window_optimizer_bayesian.py:676-716`); the
manifest default is `false` and `study_name` is empty (`agent_manifests/window_optimizer.json`,
read with `json.load`), and Run 4's `--params` sets neither. **Nothing will be picked up by
mistake**; Run 4 creates a fresh `window_opt_<time>.db` (`:673-674`, `:748-750`). By the search
record, the search then owns that study (stage-1 S9).

**Delegated-gate files.** `config_manifests/run_authorizations/` does not exist; no `*.auth` file.
No `config_manifests/env_freeze_baseline.cfg`. `config_manifests/survivor_volume.cap` is tracked
(`git ls-files`). None of `run_authorization_gate.py`, `env_freeze_gate.py` and
`survivor_volume_gate.py` is called by `gate12_launch.sh` or any other launch path (`grep` over
`*.py` and `*.sh` outside `tests/` and `docs/`), so an owner launch is not gated by them. The volume
gate is run by hand (checklist step 10).

**Also measured:** main tree `git status --porcelain` empty at 18:4x; netconsole receiver
listening on UDP 6667, PID 1102, `system.slice/netconsole-listener.service`, started 2026-11-14
08:57:30 (`ss -lunp`, `ps`, `/proc/1102/cgroup`); last `NCPROOF` lines 2026-08-22 19:26
(`logs/netconsole_all_rigs.log`); VM101 `free -m` available 12,300 MiB. Run outputs are ignored
(`git check-ignore -v`: `optuna_studies/` `.gitignore:137`, `optimal_window_config.json` `:138`,
`logs/` `:85`, `.s172_accumulator/` `:72`, the root NPZ `:73`).

## The ledger `-wal` note (NEW)

**Observed fact.** Step 1's credit reads the miner ledger's stripe rows immutable and refuses if
`-wal` or `-journal` is non-empty (`window_optimizer_integration_final.py:1443-1456`). `G-PROD-SHAPE`
makes its section B UNAVAILABLE on the same condition (`tests/gate_s172_prod_shape.py:39-41`). The
coordinator opens the ledger in WAL mode on a fresh connection per query
(`miner/range_miner_coordinator.py:1207-1213`). The concurrency sampler opens it `mode=ro` every 2 s
(`scripts/gate12_concurrency_sampler.py:203`; `gate12_launch.sh:364-366`) and keeps running until 10 s
after WATCHER exits (`:392-396`).

**Measured tonight, on a throwaway file** (`/home/michael/tfm_worktrees/run4-prep-tools/wal_probe/`):

| Step | `-wal` | `-shm` |
|---|---|---|
| A. writer alone, closed | absent | absent |
| B. writer writes 50 rows and closes while a `mode=ro` reader is open | 206,032 B | 32,768 B |
| C. the reader then closes (last) | 206,032 B | 32,768 B |
| D. a later writer opens and closes | absent | absent |

**What is not known:** whether, in Run 4, the coordinator's last write closes while a sampler read
is open, and whether some later write clears it before Step 1 reads the evidence. None of tonight's
suites runs the sampler against a real coordinator.

**As a prediction about the run:** if it happens, Run 4 computes, earns no credit, leaves the cursor
at 2³¹, and the S7 line names the `-wal` size. Nothing is lost. **The measurement that settles it** is
Run 4 itself: the S7 line, or `ls -la ~/miner_staging/miner_ledger.db-wal` right after the run.
No change is proposed. If it happens, any remedy (when the sampler stops, how the evidence is read)
is a decision for you and Beta's review.

## Every `UNAVAILABLE` in the checklist

| Step | What is missing |
|---|---|
| 13 — the launch command | A committed launcher that carries the seven pins. `gate12_launch.sh` carries Gate 12's `--params`; the pinned `--params` exists only as the pre-launch check's printout. D-1 |
| 15 — freezing the evidence bundle | No committed command or script; earlier bundles under `/home/michael/gate12_attempt*` were made by hand |

Also recorded only as a name or as reconstructed, not as an exact command: `boot-proxmox` (its
invocation, step 1); the ping sweep (step 1, `CLAUDE.md:143` gives no flags); the netconsole sender
(step 8, `[RECONSTRUCTED]` in the runbook); a clean mid-run stop (step 14: none exists; the hard stop
is `gate12_launch.sh:203-206`).

## Session conduct

- **Live state unchanged.** Every suite ran between two snapshots of the main tree (minus `.git`
  and `docs/exchange/`), `~/miner_staging` and `/tmp/agent_halt`; every diff was empty (T1 table,
  `live=` column). All 55 suites in `t2` and the `t3` rerun: `UNCHANGED`.
- **One slip, mine, corrected.** My snapshot helper also wrote a `*.exchange` side file, which no
  ignore rule covers, so for the first six suites of run `t1` the worktree held untracked files. Found when I
  checked `git status`; renamed to `*.exchange.log` and the helper fixed; the worktree was clean
  again from the seventh suite on. The six suites it overlapped (phase 4, cleantree admission,
  pre-launch check, routing, R-1, MP-1) match their reports exactly; Gate 22 counts only `.py`
  files, and the cleantree suite uses its own fixtures. The worktree data directory otherwise holds
  only `*.log` files.
- Tools outside the worktree: `/home/michael/tfm_worktrees/run4-prep-tools/` (runner, probe-skip
  wrapper, snapshot, comparison, the WAL probe). Logs:
  `/home/michael/tfm_worktrees/run4-prep/data/run4_prep/t1/`.
- `gate_manifest_staging_dir_is_usable_here` was not run: Part B ran through a wrapper that replaces
  only that case with a visible `SKIPPED-BY-BRIEF` failure, as the stall session did.
- **Files added to the worktree, all gitignored** (`git status --porcelain` empty at the end): the three dataset files, copied read-only (`cp -p`, `chmod 0444`) from the main tree, and an empty `optuna_studies/`. Run `t1` was stopped by its recorded PIDs (runner 158805 and its child), not by pattern.
- No scheduled wake-ups. Background shell waits (`until grep DONE …; sleep 30`) watched the runner's summary file; the one for the stopped `t1` was stopped. They wrote nothing.

## Outputs

| File | sha256 |
|---|---|
| `docs/RUN4_LAUNCH_CHECKLIST.md` | `3b0b0b38775f1ca9f08eac90bcad2e3024c39451904924ab0d7e692b9ca232d6` |
| `docs/CLAUDE_CODE_REPORT_RUN4_LAUNCH_PREP.md` | see note |

The report cannot contain its own digest. Its sha256 is in the session's final message and in
`data/run4_prep/t1/OUTPUT_SHA256SUMS.log` in the worktree.
