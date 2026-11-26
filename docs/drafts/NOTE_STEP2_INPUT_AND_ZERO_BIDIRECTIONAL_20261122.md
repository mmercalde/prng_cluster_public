# NOTE: what Step 1 hands to Steps 2–3, and what 0 bidirectional rows means for them (2026-11-22)

Read-only lookup, draft, uncommitted. Owner 2026-11-22 11:56: "yes- proceed". No rig, no GPU, no code change, no run.
Code was read at `00e2ee7` in the main tree (`/home/michael/distributed_prng_analysis`, HEAD `00e2ee7`).
**Bare "Step N" means the executable scheme** (`agents/watcher_agent.py` `STEP_SCRIPTS`: 1 window optimizer, 2
scorer meta-optimizer, 3 full scoring).

## Summary (plain language)

1. Steps 2 and 3 read one survivor file from Step 1: the 22-array NPZ `bidirectional_survivors_binary.npz`, which holds the bidirectional rows only.
2. Every RANGE-MINER run since August published **0 rows**: attempt 9, Runs L, M, P, Q, Stage 4 and today's A–D. The live `current` generation has 0 rows too.
3. The bidirectional set is forward ∩ reverse, per mode. With 0/0 constant survivors, and 13 (or 59) forward-hybrid seeds that are not among the 2995 (or 23,410) reverse-hybrid seeds, it is empty.
4. Earlier runs that did produce rows used other geometries. S162 (W6, evening, 0.68/0.70, constant only) produced 887; others are in §4.
5. **Reading the code, not running it:** Step 2 would run, score every trial −1.0, write a config and print SUCCESS. Step 3 would make 0 jobs and exit 1 ("No survivors aggregated!").
6. WATCHER checks only that the files exist and are newer than their inputs, not row counts. Its zero-survivor "critical failure" rule is never reached.
7. Nothing rules what Steps 2–3 should do with 0 rows. It is an open item (11-19 handoff), and related defects are already GOVERNED (§6).
8. **For hit rate:** 0 rows means no survivor-based pool, so these runs can say nothing about pools against k/1000. Settling that needs a geometry that yields rows, carried through Steps 2–6 and scored on unseen draws.

## Prior record

Searched read-only in the main tree at `00e2ee7`, with `/bin/grep`. Every hit listed was read in context.
`SC="docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_* docs/TB_RULING_REQUEST_* docs/PROPOSAL_* docs/CHAPTER_* docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md"`

**Governance trail:**

| # | Command | What it returned |
|---|---|---|
| G0 | `/bin/grep -n -i "bidirectional\|Step 2" docs/TFM_SESSION_HANDOFF_20261119.md` | `:26` "Both half runs ended with 0 seeds in the two-way (bidirectional) survivor file … What the record says this means for Step 2 has NOT been looked up yet." `:119` the open item. **This note answers it** |
| G1 | `/bin/grep -l -i -E "zero bidirectional\|0 bidirectional\|bidirectional[_ ]count *[=:]? *0\b\|empty npz\|no bidirectional survivors\|zero survivors" $SC docs/TFM_SESSION_HANDOFF_2026111*.md docs/RESULT_*.md docs/CLAUDE_CODE_REPORT_*.md` | **Read and relevant:** `CLAUDE_CODE_REPORT_RUN4_LAUNCH_PREP.md:245` (attempt 9 published 0); `CHAPTER_12_WATCHER_AGENT.md:726` (the Step 1 context's zero rule); `PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md:191` ("A completed sweep with zero survivors is credited"); `CLAUDE_CODE_REPORT_STEP1_SEARCH_RECORD_STAGE1.md:263`, `:403` (zero-survivor credit gates); `PROJECT_FILE_CATALOG.md:215` (the "rectangular empty NPZ" encoding seam, RULED). The rest are keyword-only |
| G2 | `/bin/grep -l -i bidirectional docs/TB_RULING_* docs/TB_RULING_REQUEST_*` | `TB_RULING_REQUEST_STEP2_v4_1_OBJECTIVE.md`, **`TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md`** (S107: 6,739 survivors, 98.8% of `bidirectional_selectivity` at floor), IPC S150, ROCR S149, window-anchor scope/sequencing. **No ruling or request addresses a 0-row Step 1 output** |
| G3 | `/bin/grep -l -i -E "bidirectional_survivors\|survivors_with_scores\|bidirectional_survivors_binary" docs/PROPOSAL_*` | 24 proposals. Read for the hand-off: `PROPOSAL_NPZ_Auto_Conversion_Step2.md` (the auto-conversion in `run_scorer_meta_optimizer.sh:4-155`) and `PROPOSAL_S172_RANGE_MINER_v1_4_4.md` (the 22-array contract). None specifies behaviour for 0 rows |
| G4 | `/bin/grep -n -i "bidirectional\|attempt 9\|zero" docs/LEADS.md`; `/bin/grep -n -i "file_exists\|failure-authority" docs/TODO_CONSOLIDATED.md` | **No LEADS entry** for zero bidirectional rows. `TODO_CONSOLIDATED.md:91`, `:260`: the WATCHER failure-authority defect, REVIEW (GOVERNED) |
| G5 | `sed -n 260,285p docs/CLAUDE_CODE_REPORT_RANGE_MINER_VM101_READS.md`; `/bin/grep -n "bidirectional\|predict" docs/S172_RUN4_PROPOSAL_PINNED_GEOMETRY.md` | attempt 9's phase rows and its "null observation, … not called chance and … not called a finding" (`:281-285`); Run 4's predicted envelope (`:38-50`) |

**Chapters:**

| # | Command | What it returned |
|---|---|---|
| H1 | `/bin/grep -n -i "manifold\|learning signal" docs/BIDIRECTIONAL_SIEVE_MATHEMATICAL_WHITEPAPER.md` | §7, `:116-130`, `:155-163` (quoted in §5) |
| H2 | `/bin/grep -n -i "Step 2\|NPZ\|22-array" docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md` | `:30` conceptual "Step 2 Bidirectional Sieve"; `:38` the carrier objects; `:142` "exactly the 22 arrays" |
| H3 | `/bin/grep -n -i "input\|bidirectional_survivors_binary" docs/CHAPTER_3_SCORER_META_OPTIMIZER.md` | `:688-691`, `:1403-1413`: Step 2's only survivor input is the NPZ; seven arrays are read by name |
| H4 | `/bin/grep -rn -i -E "Bidirectional survivors: *[0-9]\|bidirectional[_ ](count\|survivors)…[0-9]" docs/*.md` | the earlier counts in §4: Chapter 2 `:1104`; `AUDIT_STEP1_OFFSET_REACH.md:240`; `CHAPTER_1_PATCH_S114.md:71-72`; the S107 request |
| H5 | `/bin/grep -n -i "zero bidirectional\|0 rows\|Step 2.*bidirectional" docs/PIPELINE_BEHAVIOUR_MODEL.md` | only `:64-68`, the two numbering schemes (conceptual Step 2 = sieve). Nothing on 0 rows |

**Code** (each anchored in the body): `agent_manifests/scorer_meta.json`, `full_scoring.json`;
`agents/watcher_agent.py`; `preflight_check.py`; `run_scorer_meta_optimizer.sh`; `scorer_trial_worker.py`;
`generate_full_scoring_jobs.py`; `run_step3_full_scoring.sh`; `scripts_coordinator.py`; `utils/survivor_loader.py`;
`utils/canonical_records.py`; `window_optimizer_integration_final.py`; `agents/contexts/window_optimizer_context.py`.

**Labels:**
- **ALREADY RECORDED:** the hand-off contract, attempt 9's zero, the Run 4 predictions, the WATCHER failure-authority
  defect, and the Step 2 signal problem.
- **NEW:** the code-read answers to (2), and the DIVERGENT entries in §5.
- **CHANGES A RULING:** none.

**Not searched:** the host beyond the main tree's files, the ser8 archives, the private repository, the rigs, and the
claude.ai project file `TFM_SESSION_HANDOFF_20261111.md`, which is not in the repository.

## 1. What Steps 2 and 3 require from Step 1

| Requirement | Step 2 (scorer meta-optimizer) | Step 3 (full scoring) |
|---|---|---|
| Manifest `required_inputs` | `bidirectional_survivors_binary.npz`, `train_history.json`, `holdout_history.json` (`agent_manifests/scorer_meta.json:6`) | `bidirectional_survivors_binary.npz`, `optimal_scorer_config.json`, `train_history.json`, `holdout_history.json` (`full_scoring.json:6`) |
| Manifest `inputs` | `bidirectional_survivors.json`, `optimal_window_config.json`, `lottery_file` (`scorer_meta.json:13`) | the NPZ, `optimal_scorer_config.json`, `lottery_file` (`full_scoring.json:14`) |
| Survivors param default | `"survivors": "bidirectional_survivors.json"` (`scorer_meta.json:120`) | `"survivors": "bidirectional_survivors_binary.npz"` (`full_scoring.json:118`) |
| `preflight_check.py` | `2: ["bidirectional_survivors.json", "optimal_window_config.json"]` (`:284`) | `3: ["bidirectional_survivors_binary.npz", "optimal_scorer_config.json"]` (`:285`) |
| WATCHER | `check_output_freshness` (`agents/watcher_agent.py:960-991`): each required input exists, and the output is newer than every input. **Existence and mtime only; no row count, no schema** | the same |
| Script's survivor file | `SURVIVORS="bidirectional_survivors_binary.npz"` (`run_scorer_meta_optimizer.sh:31`); the JSON is only a conversion source (`:32`, `:139-150`) | `SURVIVORS_FILE="bidirectional_survivors_binary.npz"` (`run_step3_full_scoring.sh:66`); must exist (`:138-140`) |
| Arrays actually read | `seeds`, `forward_matches`, `reverse_matches` (hard-required, `scorer_trial_worker.py:179-196`), `bidirectional_count`, `intersection_ratio`, `trial_number`, `skip_mode` (`:201-246`); the CH3 table at `:1403-1413` | `seeds`, via `load_survivors` (`generate_full_scoring_jobs.py:48-53`); the 13 NPZ-backed metadata fields merged later (system map `:145`) |
| Row count | no minimum is declared anywhere. The worker rejects a trial whose masked subset is below `MIN_KEEP_COUNT = 10` (`scorer_trial_worker.py:336-338`, `:406-407`) | none declared; jobs = ⌈rows / 1000⌉ (`generate_full_scoring_jobs.py:61`) |

**In the main tree today:**
- `bidirectional_survivors_binary.npz` is a symlink to `.s172_accumulator/current/…` (`ls -la`).
- `current` is `gen-20261120T032801611651Z-step1_java_lcg_2147483648` (Run Q), with `final_row_count` 0
  (`.s172_accumulator/current/provenance.json`).
- `bidirectional_survivors.json` is a 1,229-byte **summary object**, not a survivor list: "The canonical Steps 2-6
  input is bidirectional_survivors_binary.npz; this file decides no winner".

## 2. What each does today with a 0-row NPZ (code reading; not executed)

### Step 2: runs to completion and reports success

1. **WATCHER freshness:** passes. The files exist (`agents/watcher_agent.py:972-976`).
2. **The script:** conversion is skipped when the NPZ is not older than the JSON (`run_scorer_meta_optimizer.sh:139-150`).
   - The finalizer writes both at publication. If the JSON were ever newer, `convert_survivors_to_binary.py` would be
     run on the summary object [UNVERIFIED outcome; see §5 DIVERGENT-2].
3. **The worker:**
   - `load_survivors` accepts an empty NPZ; it refuses only a missing `seeds` array (`utils/survivor_loader.py:273`).
   - `seeds_to_score` is `[]` (`scorer_trial_worker.py:179`).
   - `npz_forward_matches.mean()` on an empty array is NaN with a NumPy RuntimeWarning, not an exception (`:186-189`).
4. **Each trial:** `n_seeds = 0`, so `subset_n = 0 < MIN_KEEP_COUNT` and `_reject('too_small')` returns **−1.0**
   (`:351`, `:400-407`).
5. **Collection:**
   - `study.best_trial` is picked from trials that all scored −1.0;
   - `optimal_scorer_config.json` is written with that trial's params;
   - it prints `SUCCESS: Best parameters saved` (`run_scorer_meta_optimizer.sh:278-310`).
6. **WATCHER:** `evaluation_type: "file_exists"` (`scorer_meta.json:143`), `success_condition`
   `optimal_scorer_config.json` (`:53`), so **PROCEED** (Chapter 12 audit, `CHAPTER_12_WATCHER_AGENT.md:713-720`).
   The chosen scorer config carries no information.

### Step 3: refuses by failing

1. **WATCHER freshness:** passes.
2. **Job generation:** 0 seeds, so `num_chunks = 0` and `scoring_jobs.json` is an empty list
   (`generate_full_scoring_jobs.py:61-100`). The script continues: `NUM_JOBS` 0 (`run_step3_full_scoring.sh:184`).
3. **Fleet side effects still run:**
   - `ssh mkdir` and an `scp` of no chunk files to each node (`:198-211`);
   - `scripts_coordinator.py` with 0 jobs, which still creates the run-scoped output directory
     (`scripts_coordinator.py:587-609`, `_create_output_dirs`) and clears the HIP cache on the nodes (`:640`).
4. **Aggregation:** `LATEST_RUN` is that new empty directory (`run_step3_full_scoring.sh:238-242`). The coordinator
   names it `full_scoring_results/full_scoring_results_<timestamp>` (`scripts_coordinator.py:400`, `:413-415`), which
   matches the script's glob. It finds no
   survivors, prints `❌ ERROR: No survivors aggregated!` and **exits 1** (`:382-386`).
5. **Whether WATCHER then stops:** under the governed failure-authority defect (`TODO_CONSOLIDATED.md:91`), WATCHER
   judges by file existence. `survivors_with_scores.json` from an earlier run, if present, could read as PASS.
   [UNVERIFIED: whether a stale `survivors_with_scores.json` is present in the main tree was not checked.]

### WATCHER's own zero rule

`agents/contexts/window_optimizer_context.py:133-134`: "Zero survivors = critical failure", `return (False, 0.1)`.
**It is not reached:** all seven manifests are `file_exists`, and the heuristic runs only without one
(`CHAPTER_12_WATCHER_AGENT.md:712-727`).

## 3. How RANGE-MINER forms the bidirectional set, and why these counts give 0 rows

- **The contract:** `bidirectional_M == set(forward_map_M) & set(reverse_map_M)` for each mode M ∈ {constant,
  variable} (`utils/canonical_records.py:93-94`, `validate_trial_populations`). The adapter validates it at ingress
  (`window_optimizer_integration_final.py:1205`) and reads the two sets directly (`:1210-1215`). The PWC/legacy path
  computes the same: `forward_set & reverse_set` (`:2442-2444`) and `forward_set_hybrid & reverse_set_hybrid`
  (`:2571-2573`).
- **Which phases:**
  - constant = phase 1 (`java_lcg`, forward) ∩ phase 2 (`java_lcg_reverse`);
  - variable = phase 3 (`java_lcg_hybrid`) ∩ phase 4 (`java_lcg_hybrid_reverse`);
  - each phase sweeps the whole interval (`swept` 1–4 equal in every bundle).
- **What the NPZ holds:** only the records built from those sets, one row per seed after L2 selection
  (`utils/run_finalizer.py` `_select_l2_winners`, as quoted by the extractor). So the NPZ row count is
  |bidirectional_constant ∪ bidirectional_variable| after dedup.
- **0 / 0 / 13 / 2995 on `[2^31, 2^31 + 2^28)`** (Stage 4; today's A–D):
  - constant: 0 ∩ 0 = ∅;
  - variable: the 13 phase-3 seeds and the 2995 phase-4 seeds share no seed, so ∅;
  - **0 rows** (`extract_*.log:3`).
- **The half ranges, measured:**
  - attempt 9 `[0, 2^31)`: 0 / 0 / 59 / 23,515 (`CLAUDE_CODE_REPORT_RANGE_MINER_VM101_READS.md:261-264`);
  - Run M and Run P (lower half): phase 4 23,410, and P's phase 3 59 (`TFM_SESSION_HANDOFF_20261118_ADDENDUM.md:16`,
    `:22`; `TFM_SESSION_HANDOFF_20261119.md:19`);
  - Runs L and Q (upper half): 60 / 23,568 (`…ADDENDUM.md:14`; `…20261119.md:20`).
  - All published 0 bidirectional rows (`…20261119.md:26`; attempt 9 `logs/gate12_20260817_181819.log:707` via
    `CLAUDE_CODE_REPORT_RUN4_LAUNCH_PREP.md:245`).
- **A modelled expectation, not a decision rule.** If phase-3 and phase-4 membership were independent, the expected
  overlap would be the phase-3 count × the phase-4 survival rate:
  - 13 × 2995 / 2^28 ≈ **1.5 × 10⁻⁴** seeds on 2^28;
  - 59 × 23,410 / 2^31 ≈ **6.4 × 10⁻⁴** seeds on 2^31.
  - **Inputs:** both survival rates are measured on this data; independence is assumed.
  - **Status:** the measured 0 is consistent with that model. No follow-up has tested whether a non-empty
    intersection appears at these settings on other intervals. So this is a null result at one geometry, not a
    finding and not "chance".
- **The same at ruling time:** the Alpha simulation estimate of "about 0.01 bidirectional survivors at attempt-9
  settings" is not a repository fact (`CLAUDE_CODE_INSTRUCTIONS_RANGE_MINER_VM101_READS_20261113.md:46`; report
  `:282`).

## 4. Every recorded bidirectional count, with its geometry

| Run | Engine | Geometry (k, anchor, sessions, τ fwd/rev, skip, modes) | Interval | Bidirectional | Source |
|---|---|---|---|---|---|
| S114 trial 0 | PWC-era | W8 O43, skip 5–56 | [UNVERIFIED] | **43** | `CHAPTER_1_PATCH_S114.md:71` |
| S114 trial 3 | PWC-era | W3 O0, skip 10–28 | [UNVERIFIED] | **143,959** (window 3, since raised to a minimum of 6, `distributed_config.json:78`) | `:72` |
| S146 | PWC | not stated | [UNVERIFIED] | **313** (666 in the accumulator); "historical operating records of a path that no longer carries authority" | `CHAPTER_2_BIDIRECTIONAL_SIEVE.md:1103-1105` |
| S107 pool | PWC | many trials | [UNVERIFIED] | **6,739** survivors in the Step 2 pool | `TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md:43` |
| S162 "victory" | PWC | **W6 O64, evening, 0.68 / 0.70, skip 3–37, constant only** (`test_both_modes` false) | [UNVERIFIED] | **887** | `AUDIT_STEP1_OFFSET_REACH.md:240`; `optimal_window_config.s162_victory.json` |
| attempt 9 | RANGE-MINER | W12 A25, midday, 0.71 / 0.47, skip 6–99, both modes | [0, 2^31) | **0** (0/0/59/23,515) | VM101_READS report `:261-285` |
| Run 4 L, Q | RANGE-MINER | W12 A25, midday, 0.71 / 0.47, skip 6–99, generator_phase 0 | [2^31, 2^32) | **0** (0/0/60/23,568) | handoffs 11-18 add. `:14`, 11-19 `:20`, `:26` |
| Run 4 M, P | RANGE-MINER | the same | [0, 2^31) | **0** (P: 0/0/59/23,410) | 11-18 add. `:16`, 11-19 `:19` |
| Stage 4, both engines | RANGE-MINER, PWC | W12 A25, midday, 0.71 / 0.47, skip **0–16** | [2^31, 2^31+2^28) | **0** (0/0/13/2995) | `RESULT_PWC_BACKUP_STAGE4_20261121.md` §2, §4 |
| today A, B, C, D | RANGE-MINER | the same, skip [0,16] off, [0,16], [0,32], [0,250] on | the same | **0** (0/0/13/2995) | `RESULT_SKIP_SWITCH_RIG_RUNS_20261122.md` |

**What the Run 4 proposal predicted** (`S172_RUN4_PROPOSAL_PINNED_GEOMETRY.md:40-50`):
- phase 3 **~96** (attempt 9 actual 59);
- phase 4 **~22,656** (actual 23,515);
- computed from a measured rate.

**It predicted no bidirectional count.** The document has no bidirectional prediction (G5). It states what Run 4
could validate ("volume against prediction · four-stage plumbing end to end …", `:152`) and what it could not
("population equivalence", `:156`).

## 5. What the design documents say Step 1 hands to Step 2

- **Whitepaper §7** (`BIDIRECTIONAL_SIEVE_MATHEMATICAL_WHITEPAPER.md:116-130`): "Exact sieves eliminate *all
  variance*: Survivors = {s*} · No ranking · No gradients · No learning signal. Looser thresholds produce a
  *manifold* of near-consistent seeds." `:161`: "Loose thresholds are not a weakness — they are a mathematical
  necessity to expose a learnable structure."
- **System map** (`TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:38`): "Two carrier objects: the **22-array NPZ
  survivor contract** … and the **prediction pool + coverage/lift score**." `:142`: "The miner must emit **exactly the
  22 arrays and nothing more.**"
- **Chapter 3** (`CHAPTER_3_SCORER_META_OPTIMIZER.md:1403`): "Step 2's only survivor input is the **22-array NPZ**
  `bidirectional_survivors_binary.npz`."
- **Step 1 search proposal §2.3** (`PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md:191`): "A completed sweep with
  zero survivors is credited. Credit needs execution evidence, not a survivor count."

**DIVERGENT. Both sides are recorded, and neither is adjudicated here:**
- **DIVERGENT-1, intent against the configured geometry.** The whitepaper requires loose thresholds, so that Step 2
  receives a manifold. Every RANGE-MINER run uses the Run 4 pins, forward 0.71 (M = 9 of 12), and hands Step 2 an
  empty set. The pins are a recorded acceptance-run choice for capacity (`TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md`;
  Run 4 proposal `:59-60`), not a statement of what Step 2 should receive. This is a document-against-configuration
  tension, not a code defect.
- **DIVERGENT-2, the survivor file the manifest names against what exists:**
  - **The manifest and preflight name `bidirectional_survivors.json`** as Step 2's input and default `survivors`
    (`scorer_meta.json:13`, `:120`; `preflight_check.py:284`). The script treats it as the NPZ's conversion source
    (`run_scorer_meta_optimizer.sh:32`, `:139-150`).
  - **The finalizer writes it as a summary object** that "decides no winner"; the NPZ is canonical.
  - The Chapter 3 alignment audit records the JSON at 2 bytes at its time (`CHAPTER_3_ALIGNMENT_AUDIT.md:665`,
    `:893-894`).
  - [Whether the Step 2 path can ever reach the conversion with today's summary JSON was not tested.]
- **DIVERGENT-3, decision bands against live evaluation.** Chapter 10 v2 specifies PROCEED above 1000 bidirectional
  and ESCALATE below 10 (`CHAPTER_10_AUTONOMOUS_AGENT_FRAMEWORK_v2.md:948-951`). The live evaluation is
  `file_exists` (Chapter 12 audit `:712-727`). **ALREADY RECORDED** as part of the governed WATCHER defect.
- **The two numbering schemes:** "Step 2 = Bidirectional Sieve" is conceptual (system map `:30`); executable Step 2 is
  the scorer meta-optimizer. ALREADY RECORDED (`PIPELINE_BEHAVIOUR_MODEL.md:64-68`, §16 D1).

## 6. Is this already diagnosed or ruled? (GOVERNED)

- **GOVERNED, so these are statuses, not findings:**
  - **WATCHER passes on file existence and ignores a step's own failure.** `TODO_CONSOLIDATED.md:91`, `:260` (REVIEW);
    `CHAPTER_12_WATCHER_AGENT.md:727-735` ("BROKEN — B1, governed"). It is why Step 2 on 0 rows would read PASS.
  - **Step 2's objective has a dead signal on a populated pool.** `TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md` (S107), and
    the v4.1 → v4.3 evolution the worker carries (`scorer_trial_worker.py:413` onward).
  - **Attempt 9's 0** is recorded as a null observation, "not called chance and … not called a finding"
    (VM101_READS report `:281-285`).
- **Open, not ruled:**
  - what Steps 2–3 should do when Step 1 publishes 0 rows: run, refuse or skip;
  - which Step 1 geometry the first Step 1 → 6 pass should use.

  The 11-19 handoff (`:119`) is the only record of the question. No TB ruling, proposal or LEADS entry addresses it
  (G1, G2, G4).

## 7. What it predicts for pool hit rate, and the measurement that would settle it

- **What it predicts:** at the pinned geometry, Step 1 hands Steps 2–6 no survivors. Step 3 exits 1, and no
  survivor-based prediction pool can be formed. **These runs predict nothing about pool hit rate against `k/1000`,
  in either direction.** A survivor count is load, not quality (`PITFALL_STATE_RECOVERY_YARDSTICK.md` §1).
- **The measurement that would settle it:**
  1. Choose a Step 1 geometry and interval that yields a non-empty NPZ. The record has such geometries, for example
     S162's W6/evening/0.68/0.70 constant-only with 887 rows. Its interval is unstated, and it was PWC-era.
  2. Carry that NPZ through Steps 2–6, each step judged on its own result, not on file existence.
  3. Score the pools on draws the sieve never saw: `exact_hits / pool_size` per draw
     (`CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md:417`), against `k/1000`, with binomial intervals.
- **Size:** at k = 20, telling 2% from 3% needs about 1,400 unseen draws (normal approximation; independence
  assumed).

## 8. What I could not verify

- **Nothing was executed.** Step 2's −1.0-and-SUCCESS path and Step 3's exit 1 are code readings, not runs.
- **Whether a stale `survivors_with_scores.json`** sits in the main tree for WATCHER's `file_exists` to accept.
- **The S114, S146, S107 and S162 intervals and seed counts.** Not stated in what was read.
- **The finalizer's L2 dedup count against raw candidates for today's runs.** It is 0 = 0 trivially.
- **The claude.ai handoff of 2026-11-11** holding Alpha's 0.01 estimate. It is not in the repository.

## 9. Owner decisions (yes/no, with my recommendation)

| # | Decision | Recommendation |
|---|---|---|
| **D1** | Treat a 0-row Step 1 NPZ as a stop before Step 2, a guard to specify now and build later as a flag-selectable check, the current behaviour staying as the baseline? | **Yes.** Today Step 2 reports SUCCESS on nothing, and Step 3 fails only at aggregation, after fleet side effects |
| **D2** | Run Step 2 now on the live 0-row generation, to observe the code reading? | **No.** It produces a meaningless config, and WATCHER would pass it |
| **D3** | Pick a Step 1 geometry known to yield rows for the first Step 1 → 6 pass, recorded as a new choice beside Run 4's pins, which stay the baseline? | **Yes.** It is the only route to a hit-rate measurement. Which geometry is your call; the record's populated runs are all PWC-era (§4) |
| **D4** | Open a `docs/LEADS.md` entry for the zero intersection? | **No.** It is a null result consistent with the modelled background, with no surviving candidate. Track it as a status in `TODO_CONSOLIDATED.md` instead |
| **D5** | Have DIVERGENT-2 (`bidirectional_survivors.json` named as Step 2's input) looked at before any Step 2 run? | **Yes**, read-only first |
| **D6** | Fix the governed WATCHER failure-authority defect before any Step 2 → 6 run? | **Yes.** It is already recorded as required before autonomy (`TODO_CONSOLIDATED.md:91`). Without it, a failed step can read as PASS |
