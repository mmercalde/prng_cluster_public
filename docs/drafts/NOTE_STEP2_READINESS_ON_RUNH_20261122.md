# NOTE: what Step 2 (scorer meta-optimizer) needs to run on run H's 12,156-row generation, isolated (2026-11-22)

**Plain summary:**
1. **Input.** Run H (forward 0.30 / reverse 0.30, owner baseline 14:42, a moveable parameter and not a pin) published 12,156 variable-skip rows. Its tree `~/skipsw_run_thr3030` holds every file Step 2 names.
2. **Unmodified, the launcher stops before any trial.** `run_scorer_meta_optimizer.sh` reaches the legacy converter through its `-nt` test: the summary JSON is 59 ms newer than the NPZ. The converter cannot read the summary and the script exits 1. This is GOVERNED: Chapter 3 audit F1, switched off by ruling.
3. **Unmodified, it cannot be isolated.**
   - The localhost job runs with its working directory set to the **main tree**, and the worker writes its results into `~/distributed_prng_analysis/scorer_trial_results`.
   - Collection reads and deletes from that main-tree path.
   - The launcher copies three code files into each rig's live tree.
4. **The trials do no GPU work.** The worker's objective is numpy on the NPZ arrays. It reads no draws: the history arguments are accepted and ignored. GPUs serve only as job slots.
5. **A stale `optimal_scorer_config.json` (identical to the main tree's, dated 2026-03-11) already sits in run H's tree.** WATCHER's file-exists verdict would pass Step 2 on it whatever the run did (GOVERNED, Chapter 12 B1).
6. **The 0.30/0.30 thresholds do not reach Step 2 as tunables.** They are Step 1 search dimensions with bounds [0.30, 0.75] (`distributed_config.json`), so the baseline sits on the lower bound. Steps 2–6 see only their effect, as NPZ columns.
7. **Step 2's objective is not a hit-rate measurement.** It reads no unseen draws. Only pools scored on unseen draws against `k/1000` settle value.

**Open questions for the owner:**
- **Q1, route.** Either (a) a run-scoped harness that drives the existing worker functions in-process inside the isolated tree, a non-certifying diagnostic, or (b) a code change making Step 2's paths run-root-relative and the converter unreachable (Beta review before commit). Neither exists today.
- **Q2, hosts.** VM101 only, since the trials are CPU-bound, or the fleet? The fleet path writes into the rigs' live trees.
- **Q3, sample size.** The shell default is 5,000 of 12,156; WATCHER's 450 never reaches the worker (F13). Which one?
- **Q4, the baseline on a bound.** Step 1's search range starts at 0.30. Should the range be widened so a later mechanism can move the threshold down as well as up?

- **Author and scope:** Claude (Team Alpha), worktree `/home/michael/tfm_worktrees/pwc-offset`, uncommitted. Read-only research: no GPU, no rig, nothing run.
- **Code read:** main tree `41ad509`. The run H tree is at `00e2ee7`, and the Step 2 files have not changed between the two: their last commit is `48a8705` (`git log -1 -- scorer_trial_worker.py generate_scorer_jobs.py run_scorer_meta_optimizer.sh scripts_coordinator.py coordinator.py`).
- **Yardstick:** nothing here judges a seed. Step 2 output is a parameter set, not a result.

## Prior record

**Searches run** (main tree, `/bin/grep -rn -i`):

| # | Pattern | Scope | Read |
|---|---|---|---|
| P1 | `DIVERGENT` | `docs/PIPELINE_BEHAVIOUR_MODEL.md` | §16 register `:1186-1210` (D1, D8, D17 bear on Step 2) |
| P2 | `file-exists\|file exists` | `docs/CHAPTER_12_*.md`, `docs/CHAPTER_3_*.md`, `docs/BACKLOG.md`, `docs/TODO_CONSOLIDATED.md` | `CHAPTER_12_WATCHER_AGENT.md:725-736` (B1); `CHAPTER_12_AUDIT.md:70`, `:121` (D3) |
| P3 | `empty (input\|pool\|survivor)\|0 rows\|success.{0,40}empty` | same, plus `docs/drafts/NOTE_STEP2_INPUT_AND_ZERO_BIDIRECTIONAL_20261122.md` | that note §5–§6 |
| P4 | `survivor JSON\|bidirectional_survivors.json` | `docs/BACKLOG.md docs/TODO_CONSOLIDATED.md docs/CHAPTER_3_*.md docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` | `CHAPTER_3_ALIGNMENT_AUDIT.md` §1, A.0–A.7 (`:940-1118`); `CHAPTER_3_SCORER_META_OPTIMIZER.md:685`, `:1177` |
| P5 | `threshold.*tunable\|Optuna.*threshold` | `docs/CHAPTER_1_WINDOW_OPTIMIZER.md docs/TB_RULING_* docs/BACKLOG.md docs/TODO_CONSOLIDATED.md` | `CHAPTER_1:835` (`3fdf434`, Optuna threshold-drop fixed); `TODO_CONSOLIDATED.md:367` |
| P6 | `forward_threshold\|reverse_threshold\|fwd_thresh` | `*.py agents/ selfplay_orchestrator.py chapter_13_*.py` | `agents/watcher_agent.py:500-519`; `selfplay_orchestrator.py` (no hit); `agent_manifests/window_optimizer.json:116-133`; `distributed_config.json:92-101` |
| P7 | `Step 2\|scorer` | `docs/TODO_CONSOLIDATED.md` | `:91`, `:157` |

**Documents that already decide or record the subject:**
- **`docs/CHAPTER_3_ALIGNMENT_AUDIT.md`** (REV1 and the 2026-11-11 addendum): F1–F16, with N1–N3 BROKEN items.
- **`docs/TODO_CONSOLIDATED.md`:**
  - `:157` "Run Step 2 on current survivors … calls the prohibited legacy converter and `mv`s a regular file onto the finalizer-owned symlink", REVIEW;
  - `:91`, the WATCHER failure-authority defect, REVIEW.
- **`docs/TEAM_ALPHA_PHASE7_LAUNCH_NOTICE.md` §1** (D3.0-B, accepted as OPEN): Phase 6 certified Step 2/3 *consumption* (the loader) and never invoked the converter.
- **`docs/CHAPTER_12_WATCHER_AGENT.md:725-736`** (B1, governed).
- **My earlier note** `docs/drafts/NOTE_STEP2_INPUT_AND_ZERO_BIDIRECTIONAL_20261122.md` §5–§6 (DIVERGENT-1/2/3, GOVERNED list).
- **Owner ruling 14:42** (`logs/skipsw_session/OWNER_RULING_baseline_3030_20261122.txt`, sha256 `cdeaefeb…`): "yes- i prove the 0.30/0.30 baseline WITH the understanding that is a moveable parameter-by ml,self play or some other mechanism in the code".

**Labels:**

| Item | Label |
|---|---|
| Converter reached through `-nt` (§3.1) | **ALREADY RECORDED** (audit F1; TODO `:157`). New here: the run H tree shows the same condition (59 ms) |
| `mv` over the symlink | **ALREADY RECORDED** (F2) |
| Main-tree output path, localhost working directory, code push into rig trees (§3.3) | The push is **ALREADY RECORDED** (F9 `:257`). The **worker's and collector's hardcoded `~/distributed_prng_analysis/scorer_trial_results`** as an isolation blocker is **NEW** in this framing |
| Stale `optimal_scorer_config.json` in the run H tree | **NEW** (a box-state fact); the defect it exposes is **ALREADY RECORDED** (B1) |
| Ramdisk "already loaded" check is existence-only (§3.4) | **NEW** [UNVERIFIED as governed: not found in P2–P4] |
| No GPU compute in the worker; history ignored | **ALREADY RECORDED** in part (`scorer_trial_worker.py:524-526` comment; audit §6.4) |
| Thresholds not Step 2 tunables; baseline at the lower bound | **NEW** as a statement. The bounds are recorded (`distributed_config.json:92-101`, S148) |
| Anything that changes a ruling | **none** |

**Not searched:** the ser8 archives, the private repository, project files outside the repository, the changelog and `apply_s*.py` corpora. Rig trees were not read (no rig access).

## 1. Inputs and their paths (run H tree `~/skipsw_run_thr3030`, read now)

| Input | Path | State |
|---|---|---|
| Survivor NPZ (canonical) | `bidirectional_survivors_binary.npz` → symlink → `.s172_accumulator/current/…` | Generation `gen-20261122T223747461976Z-step1_java_lcg_2147483648`, artifact `3f6a884d…`, 12,156 rows. The bundle copy is at `logs/skipsw_session/bundle_thr3030/survivors.npz` (12,156 distinct seeds, all `skip_mode` 1) |
| Summary JSON | `bidirectional_survivors.json` (19-key dict, "decides no winner") | 14:37:50.387. The NPZ target is 14:37:50.328, so **the JSON is newer** |
| History | `train_history.json` (14,454), `holdout_history.json` (3,614) | 18,068 = the whole combined dataset at 80/20 (`window_optimizer.py:1086-1105`). Not session-separated. The worker ignores both (§2) |
| Step 1 config | `optimal_window_config.json` | `executed_thresholds` 0.30/0.30, `bidirectional_count` 12156 |
| Fleet config | `distributed_config.json` | localhost 1 × RTX 3080 Ti; `192.168.3.120/.154/.162` 8 GPUs each (the bare-metal addresses, per CLAUDE.md §3) |
| Collection config | `ml_coordinator_config.json` | localhost 2, rigs 8 each (26) |
| Stale output | `optimal_scorer_config.json` | **Present, 2026-11-22 14:31, byte-identical (`cmp`) to the main tree's 2026-03-11 file** |

**Manifest declarations disagree (F12):**
- `agent_manifests/scorer_meta.json` `required_inputs` names the NPZ and the two histories (`:6-10`).
- Its `inputs` and `default_params.survivors` name the JSON (`:13-17`, `:120`).

## 2. The command and what it runs

**Command:** `bash run_scorer_meta_optimizer.sh [TRIALS] [--sample-size N]`, from the tree root (default 100 trials, sample 5,000: `:171`, `:192`). WATCHER reaches it as `STEP_SCRIPTS[2]` (`agents/watcher_agent.py:405-408`) with positional args only (`:2089-2098`, F13).

**Sequence by reading:**
1. **Ramdisk preload** of the NPZ and both histories to `/dev/shm/prng/step2` on every node in `distributed_config.json` (`:14-24`; `ramdisk_preload.sh:104-135`). Remote failures are non-fatal.
2. **NPZ auto-conversion block** (`:140-150`). Here the run stops today (§3.1).
3. `generate_scorer_jobs.py` creates the Optuna study `sqlite:///./optuna_studies/scorer_meta_opt_<epoch>.db` in the tree (`:221-247`; `generate_scorer_jobs.py:101-103`). It pre-samples 100 parameter sets (`:44-69`: `residue_mod_1/2/3`, `max_offset`, temporal and NN parameters, `sample_size`). The job args point at `/dev/shm/prng/step2/…` (`:126-135`).
4. **Code push** of `survivor_scorer.py reinforcement_engine.py scorer_trial_worker.py` to the hardcoded `192.168.3.120 .154 .162` (`:257-261`).
5. `scripts_coordinator.py --jobs-file scorer_jobs.json --output-dir scorer_trial_results --preserve-paths` (`:266-269`). It caps batches at 6 jobs per node (`scripts_coordinator.py:105`).
6. **Collection** through `coordinator.MultiGPUCoordinator('ml_coordinator_config.json').collect_scorer_results(1)` (`:284-285`, F10), then `study.tell` and `optimal_scorer_config.json` (`:288-309`).

**What a trial computes** (`scorer_trial_worker.py`):
- `load_data` (`:151`) reads 7 of the 22 arrays (audit A.7). It runs with loader fallback enabled (F14).
- `run_trial` (`:286`) samples `sample_size` rows (`:351-366`) and scores with the v4.3 objective (`:413-444`). Only `residue_mod_1/2/3` and `max_offset` reach it (F6), and it steers toward `skip_mode == 1` (F8); **all 12,156 rows here are `skip_mode` 1.**
- **No GPU work:** `cupy` and `torch` appear only in the exit cleanup (`:125-148`).
- **No draws:** "args 2+3 accepted … but ignored" (`:524-526`).
- `--optuna-study-db` is parsed (`:549-552`) and never used by the worker.

## 3. Defects that bear on an isolated run (GOVERNED items are statuses, not findings)

### 3.1 The converter is reached on every certified run (GOVERNED: F1, switched off by ruling; TODO `:157`)
- **The trigger:** `[ bidirectional_survivors.json -nt bidirectional_survivors_binary.npz ]` follows the symlink. In run H's tree the JSON is 59 ms newer, so the test is true and `convert_to_npz` runs (`:144-147`).
- **What the converter does with the summary:** it loads the 19-key summary dict (`convert_survivors_to_binary.py:89-91`), then indexes `survivors[0]` on it (`:112`, KeyError). The script exits 1 (`:87-89`). [Read only; not executed.]
- **If any input the converter accepts got through,** `mv` replaces the finalizer-owned symlink with a regular file (F2, `:97`).
- **`distribute_npz` would then copy it** into each rig's `~/distributed_prng_analysis/` (`:104-135`).

### 3.2 Step 2 "SUCCESS" on empty or failed input
- **WATCHER's file-exists authority** (GOVERNED, B1; TODO `:91`): the verdict is the existence of `optimal_scorer_config.json` (`scorer_meta.json` `success_condition`) and ignores `run_step`'s `success: False` (`agents/watcher_agent.py:2177-2185`). **Run H's tree already holds that file** (§1), so WATCHER would PASS Step 2 even on the exit 1 of §3.1.
- **Empty input:** the converter treats `[]` as valid and writes a 22-array zero-length NPZ, "empty input is valid" (`convert_survivors_to_binary.py:94-108`). The worker's not-sampled branch never assigns `sm_arr`, so every trial raises `UnboundLocalError` whenever rows ≤ `sample_size` (F5 / N1; `scorer_trial_worker.py:368-375` against `:421`). **With 12,156 rows this bites at `--sample-size` ≥ 12,156**, not at the default 5,000.
- **All trials failing:** collection then gets 0 results and `study.best_trial` raises. The launcher exits non-zero (`set -e`, `:165`), but by §3.2's first point WATCHER does not see it.

### 3.3 DIVERGENT-2, the survivor JSON (my earlier note §5): resolved for this tree
The manifest and preflight name `bidirectional_survivors.json` (`scorer_meta.json:13`, `:120`; `preflight_check.py:282-284`). The finalizer writes it as a summary. That is what makes §3.1 fire. Earlier I wrote "[not tested]"; now it is answered by reading (audit A.4 F1), and the run H tree shows the condition.

### 3.4 Isolation blockers (main tree and rig trees)
- **The localhost job runs in the main tree:** `subprocess.run(..., cwd=node.script_path)` (`scripts_coordinator.py:893`), and run H's `distributed_config.json` localhost `script_path` is `/home/michael/distributed_prng_analysis`. So the main tree's `scorer_trial_worker.py` runs, not the isolated tree's (the code is identical, `48a8705`).
- **The worker writes into the main tree:** `Path.home() / "distributed_prng_analysis" / "scorer_trial_results"` (`scorer_trial_worker.py:498-516`).
- **The collector reads and deletes there:** `_read_local_scorer_results` uses the same hardcoded path and `os.remove` (`coordinator.py:2599-2614`). Remote results are read and removed over SFTP at `/home/<user>/distributed_prng_analysis/scorer_trial_results` (`:2635-2664`).
- **The code push overwrites three files in each rig's live tree** (`:257-261`, F9).
- **`/dev/shm/prng/step2` is shared by every tree on a node.** The "already loaded" check tests file existence only (`ramdisk_preload.sh:63-76`, used at `:110-113` and `:131-134`), so a stale `.ready` from another run would be served without a content check. [UNVERIFIED as governed.] `/dev/shm/prng` currently holds only `miner/` (read now).
- **The live ledger `~/miner_staging` is not touched by any Step 2 file:** grep for `miner_staging|miner_ledger` in the five files gives 0 hits.

### 3.5 Hosts, GPUs and the fleet surfaces (GOVERNED: F9, F16, D17)
- **Three fleet surfaces:**
  - dispatch reads `distributed_config.json` (25 slots: 1 + 8 + 8 + 8);
  - collection reads `ml_coordinator_config.json` (26);
  - the code push hardcodes three IPs.
  None of them reads the Resolved Execution Set.
- **Through WATCHER,** preflight hard-fails Step 2 when the rigs are booted into Proxmox, because the bare-metal addresses are unreachable (F16; `agents/watcher_agent.py:1660`, `:1753`; `preflight_check.py:354`). Rig boot state was not checked here (no rig access).
- **Because a trial uses no GPU** (§2), a VM101-only run needs only CPU slots (localhost `max_concurrent_script_jobs` 2). That is Q2.

### 3.6 Sample size cannot be set from WATCHER (GOVERNED: F13 / N3)
`scorer_meta.json:126` `sample_size: 450` has no path to the shell under `arg_style: positional`. The shell's 5,000 runs.

### 3.7 No dataset-provenance binding (GOVERNED: F15)
Step 2 records no dataset identity (`scorer_trial_worker.py:527-531`; `run_scorer_meta_optimizer.sh:213-218`). Run H's dataset sha256 (`51364816…`) is recorded only in Step 1's bundle and generation.

### 3.8 How thresholds reach Step 2
- **Step 1 thresholds are Step 1 search dimensions:** `search_bounds.forward_threshold` / `reverse_threshold` `{min 0.30, max 0.75, default 0.30}` (`distributed_config.json:92-101`). The manifest says its bounds are informational (`agent_manifests/window_optimizer.json:116-133`). An operator can pin them only as `warm_start_fwd_thresh` / `warm_start_rev_thresh` (`agents/watcher_agent.py:511-519`). Run H used that route: `optimal_window_config.json` `executed_thresholds`, `"source": "winning_trial_window_config"`.
- **Step 2's search space has no threshold dimension** (`generate_scorer_jobs.py:52-69`). The 0.30/0.30 choice reaches Step 2 only as data: which rows exist, and their `forward_matches` / `reverse_matches` columns.
- **No code path from Steps 2–6, selfplay or Chapter 13 back to the Step 1 thresholds was found** (P6: `selfplay_orchestrator.py` has no threshold hit; its `min_fitness_threshold` `:173` is a different quantity). [Chapter 13's step range was not re-traced this pass.]
- **So "moveable by ML, selfplay or another mechanism" (owner 14:42) is, today, Step 1's own Optuna search over [0.30, 0.75].** Run H's baseline sits on the lower bound. Q4.

### 3.9 Not defects, recorded for the run
- **Sessions:** the history files mix midday and evening (`BACKLOG.md` §5: per-session is the rule). The Step 2 worker does not read them; Steps 3–6 [not traced here] may.
- **Lead L-8** (`docs/LEADS.md`; `docs/drafts/NOTE_LEAD_BIDIRECTIONAL_EXCESS_20261122.md`): run H's two-way count is 2.31× the independence figure, an arithmetic property of the two passes. It does not gate this work.

## 4. Expected duration and peak memory

**[UNVERIFIED: no Step 2 has run since 2026-03-11 (audit A.0 #1), and nothing was run here.]**
- **What bounds the time:** per trial, a 5,000-row sample, numpy scoring, a Python start-up and the import of the worker module.
- **Wall time:** 100 trials at 2 concurrent local slots is dominated by process start-up. The fleet path adds SSH dispatch, batch cooldowns of 5 s (`scripts_coordinator.py:106`) and collection.
- **Peak memory:** a small multiple of the NPZ (12,156 rows × 22 arrays) per process. [UNVERIFIED]
- **The first harness run should measure and record both.**

## 5. What it writes

**In the launch tree:**
- `bidirectional_survivors_binary.npz` (only if the converter succeeds; F2);
- `optuna_studies/scorer_meta_opt_<epoch>.db`;
- `scorer_jobs.json`;
- `scorer_trial_results/` (the `--output-dir`; the worker does not write there);
- `optimal_scorer_config.json`.

**Outside it:**
- `~/distributed_prng_analysis/scorer_trial_results/trial_NNNN.json` on localhost (main tree) and on each rig;
- three code files in each rig's tree;
- `/dev/shm/prng/step2/` on every node.

## 6. What this predicts for pool hit rate

**Nothing on its own.** Step 2 picks scorer parameters by an objective computed from NPZ columns. It reads no draws, seen or unseen, so a Step 2 result is not a hit-rate result. Run H's population reaches a hit-rate measurement only through Steps 3–6, with pools scored on draws the sieve never saw against `k/1000`.
