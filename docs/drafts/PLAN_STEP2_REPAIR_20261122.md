# PLAN: repairing Step 2 so run H can go through a certified WATCHER Step 2 run (2026-11-22)

**Summary:**
1. **Scope.** Fourteen recorded items stand between run H's 12,156-row generation and a Step 2 run through WATCHER that could count. All of them are already recorded, so each is a status, not a finding.
2. **One is already repaired, and my readiness note was wrong about it.** WATCHER's "file exists means pass" defect (Chapter 12 B1) was fixed on 2026-11-11 at `b6cb789`: an explicit step failure now overrides output-file evaluation (`agents/watcher_agent.py:1099-1135`). What remains is narrower: a run that exits 0 with no usable trial still passes.
3. **The stale `optimal_scorer_config.json` is a tracked file** (`git ls-files`). Every checkout carries the 2026-03-11 config. That is why run H's tree has it.
4. **F6 is bigger than "dead in the objective."** None of the 11 tuned parameters reaches any later step:
   - Step 3 builds its scorer without the config (Chapter 4 B5).
   - Step 5 samples its own copies of the four NN parameters, in another encoding (Chapter 3 §17 B2).
   - Only 4 of the 11 reach Step 2's own objective.
   - The recorded intent (S108) is the opposite: one shared config serving Steps 3 and 5, "Do NOT remove."
5. **Most of F6 is an owner decision, not a code fix.** Wiring the parameters into Step 3 changes the 91/89 feature-name contract and `holdout_quality`. Wiring them into Step 5 needs an encoding decision. Changing the objective touches a TB-ruled design (S107 v4.3).
6. **Five landable changes cover the code items.** Two touch hand-landed files (WATCHER and the Step 2 manifest). Four need Beta. Nothing here touches `miner/` or the governed files list.
7. **Yardstick:** none of this produces a hit-rate result. A certified Step 2 yields a parameter set; only pools scored on unseen draws against `k/1000` settle value.

**Owner questions:**
- **O1, F6 intent.** Should Step 2's parameters drive Step 3 (residue mods, offset, temporal, confidence) and Step 5 (NN shape and rates), as S108 recorded? If yes, Step 3's feature names and `holdout_quality` must change, and Step 5 must stop sampling its own. If no, record them as intentionally unconsumed and remove them from Step 2's search.
- **O2, the objective.** v4.3 measures only `skip_mode` enrichment, which is identically 0 on run H's population (all variable-skip). Ask Beta for a v4.5 objective? (S107 already queued the `max_offset` scale flaw "Needs TB approval before v4.5".) Or run Step 2 as-is, as a pipeline exercise?
- **O3, sample size.** Manifest 450 against the shell's 5000. S107 retired 450, and the shell's 5000 is what runs. Which value becomes the single configured one?
- **O4, the tracked config.** Untrack `optimal_scorer_config.json` from git?
- **O5, fleet for Step 2.** The trials are CPU only. Run Step 2 on VM101 alone, which removes the rig, code-push and Proxmox-preflight items from the critical path? Or keep the 25-slot fleet?
- **O6, what "certified" means.** No Step 2 certification criterion exists in the record searched (P3 below). It needs defining, for example: input artifact sha256 bound to the generation, every trial accounted for, best value finite, provenance written.

- **Author and scope:** Claude (Team Alpha), worktree `/home/michael/tfm_worktrees/pwc-offset`, uncommitted.
- **Mode:** read-only planning; no code changed.
- **Code anchors:** main tree `41ad509`, unless marked "record", which means the anchor as written in the cited audit (`a15b524`).
- **Authority:** owner 16:08.

## Prior record

**Searches run** (main tree, `/bin/grep -n -i`):

| # | Pattern | Scope | Read |
|---|---|---|---|
| P1 | `F1`–`F16`, `B1`–`B5`, `N1`–`N3` | `docs/CHAPTER_3_ALIGNMENT_AUDIT.md` | §1 `:36-376`, addendum A.4–A.6 `:1037-1105` |
| P2 | `B2\|B5\|§17` | `docs/CHAPTER_3_SCORER_META_OPTIMIZER.md`, `docs/CHAPTER_4_AUDIT.md`, `docs/CHAPTER_4_FULL_SCORING.md` | Ch3 §6.1 `:544-597`, §17 `:1467-1520`, §18; Ch4 audit `:37`, `:111-135` |
| P3 | `temporal_window\|temporal_num\|min_confidence\|hidden_layers\|dropout\|learning_rate\|batch_size\|residue_mod\|max_offset\|search space\|certif` | `docs/PROPOSAL_STEP2_OBJECTIVE_FUNCTION_v1_3_0.md`, `docs/TB_RULING_REQUEST_STEP2_v4_1_OBJECTIVE.md`, `docs/TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md`, `docs/S107_session_log.md`, `docs/SESSION_CHANGELOG_20260222_S108.md` | S108 `:92-122`; v4.1 request `:40-70`, `:100-167`; S107 `:60-108` |
| P4 | `D3.0-B\|§12` | `docs/BACKLOG.md` | §12 `:250-285` |
| P5 | `R-1 WATCHER SAFETY`, `b6cb789` | `agents/watcher_agent.py`, `git log -S` | `:1099-1200`; commit `b6cb789` (2026-11-11, "failure authority … (Ch12 B1, B6, B3); 21/21 gates") |
| P6 | `HAND_LANDED`, `GOVERNED_FILES` | `scripts/owner_land.py:84-87`; `scripts/gate12_parity_gate.py:174` | the hand-landed set |
| P7 | earlier today | my readiness note's Prior record (P1–P7) | as written there |

**Documents that already decide, rule or record these items:**
- `CHAPTER_3_ALIGNMENT_AUDIT.md`: F1–F16, N1–N3 and the addendum.
- `CHAPTER_3_SCORER_META_OPTIMIZER.md` §17: B1–B5 BROKEN, S1 SWITCHED-OFF-BY-RULING, §18 open items.
- `CHAPTER_4_AUDIT.md` B5.
- `BACKLOG.md` §12: D3.0-B, Beta ruling 2026-08-02, "Do not invoke the legacy converter until D3.0-B closes".
- `TEAM_ALPHA_PHASE7_LAUNCH_NOTICE.md` §1.
- `TODO_CONSOLIDATED.md:91`, `:157`.
- `S107_session_log.md:96`, `:100`, `:104`.
- `SESSION_CHANGELOG_20260222_S108.md` §3.
- `PROPOSAL_NPZ_Auto_Conversion_Step2.md`: the conversion block's origin, "Team Beta Approved: January 19, 2026" (`run_scorer_meta_optimizer.sh:3-7`).
- `CLAUDE.md` §3: bare-metal addresses deliberate; the selectable rig profile "explicitly NOT done".
- `b6cb789`.

**Labels:** every item in §1 is **ALREADY RECORDED** and cited. The **NEW** statements are:
- the tracked status of `optimal_scorer_config.json`;
- B1's repair making my readiness note's §3.2 stale;
- the main-tree result paths as an isolation blocker (readiness note §3.4);
- the zero enrichment on run H (harness draft §4).

**Nothing here changes a ruling.** The items that would change one are marked **NEEDS OWNER** or **NEEDS BETA**.

**Correction to my readiness note** (`docs/drafts/NOTE_STEP2_READINESS_ON_RUNH_20261122.md`, summary item 5 and §3.2): B1 is **repaired at `b6cb789`**. Its text was taken from the Chapter 12 audit (written at `a15b524`, before the repair) and was not re-read live. Corrected here; that note is not edited.

**Not searched:** the ser8 archives, the private repository, project files outside the repository, the changelog and `apply_s*.py` corpora beyond the files named.

## 1. Item by item

Every item below carries the same five fields:
- **Record:** what the audit or chapter says, with anchors.
- **Ruled:** what was already ruled or deferred, and by whom.
- **Fix:** the smallest change that closes it.
- **Files:** what it touches, and whether those files are governed or hand-landed.
- **Test:** what would prove the fix.

**Files under `agents/` or `agent_manifests/` are owner-landed by hand.** No item touches `miner/` or the governed files list (`adaptive_thresholds.py`, `execution_set.py`, `hybrid_strategy.py`, `miner/*`, `prng_registry.py`, `sieve_gpu_worker.py`).

### F1: Step 2 calls the legacy converter
- **Record:**
  - The conversion call is `run_scorer_meta_optimizer.sh:87`; it is reached through `:144` (`-nt`) on every certified tree (audit F1 `:36-55`; A.4 `:1045`).
  - The converter fails on the summary dict (`convert_survivors_to_binary.py:89-112`).
  - Run H's tree: the JSON is 59 ms newer than the NPZ (readiness note §1).
- **Ruled:**
  - **SWITCHED-OFF-BY-RULING** (Ch3 §17.2 S1; `TEAM_ALPHA_PHASE7_LAUNCH_NOTICE.md` §1): do not invoke the converter until D3.0-B closes (`BACKLOG.md:275`, Beta 2026-08-02).
  - The block itself was Beta-approved on 2026-01-19 (`run_scorer_meta_optimizer.sh:3-7`).
- **Fix:** delete the NPZ auto-conversion block `:1-156` apart from `set -euo pipefail`. Replace it with a fail-closed check that `bidirectional_survivors_binary.npz` is a **symlink** whose target's sha256 equals the generation sidecar's `artifact_sha256`. Never convert in Step 2.
- **Files:** `run_scorer_meta_optimizer.sh` (not governed; owner_land can carry it).
- **Test:** a shell harness with stubbed `python3`/`scp`, plus a fixture tree holding a newer summary JSON and an NPZ symlink. Assert the converter is never executed, nothing is `scp`'d, and the symlink survives. A tampered NPZ (sha mismatch) exits non-zero.
- **NEEDS BETA:** it removes a Beta-approved block, and it touches the D3.0-B surface.

### F2: `mv` over the finalizer-owned symlink
- **Record:** `run_scorer_meta_optimizer.sh:97` replaces the alias, so the next `finalize_run` fails closed (`utils/run_finalizer.py:1400-1413`). Audit F2 `:57-85`.
- **Ruled:** the same S1 holding as F1.
- **Fix:** closed by the F1 fix, which deletes `:73-150`. No separate change.
- **Files:** as F1.
- **Test:** the F1 test's "symlink survives" assertion, plus running `finalize_run` again on the fixture afterwards without `PublicationError`.

### F5 / N1 / Ch3 B3: `UnboundLocalError` when rows ≤ `sample_size`
- **Record:**
  - `scorer_trial_worker.py:365-366` assigns `sm_arr` only in the sampling branch; the else branch `:368-375` does not; it is read at `:421`.
  - Execution-proven (audit F5 `:134-161`).
  - Reproduced today by harness self-test C.
- **Ruled:** nothing; it is listed for the repair list (Ch3 §17.1 B3, §18).
- **Fix:** add `sm_arr = npz_skip_mode` to the else branch, and drop the duplicate `:366`. This is a two-line change.
- **Files:** `scorer_trial_worker.py` (not governed).
- **Test:** `run_trial` on synthetic arrays with n ≤ sample_size returns a finite objective. Add a mutant that removes the line and expect red. The harness self-test C then flips from `raised` to `scored`.

### F6: sampled parameters that never reach the trial
See §2 for the full parameter table and the intent record.
- **Record:** only 4 of the 11 parameters reach the objective (audit F6 `:163-189`). None reaches a downstream step (Ch3 §6.1 note `:576-597`, §17 B2 `:1477`; Ch4 B5 `:111-135`).
- **Ruled:**
  - S108 closed "remove NN params" with "**Do NOT remove**" (`SESSION_CHANGELOG_20260222_S108.md:92-122`), on the premise that Steps 3 and 5 consume them. The audits later found that premise false.
  - The objective is TB-ruled v4.3 (S107; `PIPELINE_BEHAVIOUR_MODEL.md` §4 GOVERNED).
  - The `max_offset` scale flaw "Needs TB approval before v4.5" (`S107_session_log.md:96`).
- **Fix:** depends on O1 and O2 (§2). Not a code fix until the owner rules.

### F8: the objective steers toward `skip_mode == 1`
- **Record:** `scorer_trial_worker.py:414-426`; `utils/prng_encoding.py:37`. Audit F8 `:218-233`: "a contact point only… needs a governed decision, not a patch."
- **New measured fact:** on run H every row is `skip_mode` 1, so enrich ≡ 0 and the objective ranks only the keep fraction (harness draft §4).
- **Ruled:** v4.3 is TB S107.
- **Fix:** none without O2. **NEEDS OWNER + BETA.**

### F9: three fleet surfaces, the code push, and main-tree paths
- **Record:**
  - `scripts_coordinator.py:431-444` reads `distributed_config.json`.
  - The collector reads `ml_coordinator_config.json` (`run_scorer_meta_optimizer.sh:284`).
  - Three hardcoded IPs (`:257-261`); "a straight no-hardcoding violation", and failures are silent (audit F9 `:235-255`).
  - **From the readiness note §3.4** (anchors re-read today):
    - the localhost job runs with `cwd=node.script_path` (`scripts_coordinator.py:893`), which is the main tree;
    - the worker writes `~/distributed_prng_analysis/scorer_trial_results` (`scorer_trial_worker.py:498`);
    - the collector reads and deletes there (`coordinator.py:2603-2614`; remote `:2635-2664`).
- **Ruled:** nothing. S107 added the IPs deliberately ("added .162 to scp loop", `S107_session_log.md:68`).
- **Fix, three parts:**
  - (a) Delete the code push `:255-262`. Code reaches a rig only through a tree provisioned by the run's execution set, not an `scp` into its live tree.
  - (b) One run root: pass `--results-dir <run root>/scorer_trial_results` to the worker, and read the same path in collection, instead of `Path.home()/distributed_prng_analysis`.
  - (c) Build dispatch and collection from the frozen Resolved Execution Set (`execution_set.py`, read-only consumer), not from two JSON files.
- **Files:**
  - `run_scorer_meta_optimizer.sh` and `scorer_trial_worker.py`: not governed.
  - `coordinator.py` and `scripts_coordinator.py`: not governed, but shared by Steps 3 and 5. Change only the scorer-collection functions and the localhost `cwd` choice, or add a Step-2-only collector.
  - `execution_set.py` is governed; it is consumed, not edited.
- **Test:**
  - Run with `HOME` pointed at a scratch home: no file appears under `~/distributed_prng_analysis`, and results appear under the run root (the harness self-test's W checks carry over).
  - A no-IP-literal lint on the shell.
  - A mutant restoring `Path.home()` turns the test red.
- **NEEDS BETA** for (c), because it changes how a step resolves the fleet.

### F10 / Ch3 B1: `collect_scorer_results(1)`
- **Record:** `run_scorer_meta_optimizer.sh:285`; `coordinator.py:2591-2593`. Audit F10 `:257-275`.
- **Ruled:** nothing; on the repair list (§17 B1).
- **Fix:** pass `$TRIALS`, and **exit non-zero** when collected < `$TRIALS` or when no trial has a finite value. This also closes the residual of B1: a run that exits 0 with nothing usable.
- **Files:** `run_scorer_meta_optimizer.sh`.
- **Test:** the stubbed collector returns 3 of 100: the shell exits non-zero. All `-inf`: exits non-zero. All 100 finite: exits 0.

### F12: three disagreeing input declarations
- **Record:**
  - The manifest's `required_inputs` names the NPZ and histories (`agent_manifests/scorer_meta.json:6-10`).
  - The manifest's `inputs` and `default_params.survivors` name the summary JSON (`:13-17`, `:120`).
  - Preflight names the JSON (`preflight_check.py:284`).
  - Audit F12 `:298-310`.
- **Ruled:** nothing.
- **Fix:** all three name `bidirectional_survivors_binary.npz` plus `optimal_window_config.json`, with histories only where a consumer reads them. The Step 2 worker reads no history (`scorer_trial_worker.py:524-526`). Remove `description` "Step 2.5" from `:3`.
- **Files:**
  - `agent_manifests/scorer_meta.json`: **hand-landed**.
  - `preflight_check.py`: not governed.
- **Test:** a static test that the three declarations are equal and name no summary JSON. Preflight on a tree without the JSON passes; preflight on a tree without the NPZ fails.

### F13 / N3 / Ch3 §17.1 B5: sample size cannot come from WATCHER
- **Record:**
  - `scorer_meta.json:126` sets `sample_size: 450`, but `arg_style: positional` (`:136-140`).
  - WATCHER emits positional args only (`agents/watcher_agent.py:2360-2370`).
  - The shell uses 5000 (`run_scorer_meta_optimizer.sh:171`). Audit F13 `:312-327`.
- **Ruled:** S107 retired the 450 cap on purpose ("sample_size=450 obsolete cap" closed, `S107_session_log.md:104`). The manifest was never updated.
- **Fix:** one value, chosen by O3. Have WATCHER pass it as `--sample-size` (add to `flag_args` with a value form, or switch to named args), and have the shell refuse a missing value rather than default.
- **Files:**
  - `agent_manifests/scorer_meta.json`: hand-landed.
  - `agents/watcher_agent.py`, if the arg builder needs a valued flag form: hand-landed.
  - `run_scorer_meta_optimizer.sh`.
- **Test:** a WATCHER command-builder unit test: the manifest value appears in the built command and in `scorer_jobs.json`, for every job.
- **NEEDS OWNER** (O3).

### F14: loader fallback enabled
- **Record:** `scorer_trial_worker.py:166`; `utils/survivor_loader.py:106`, `:160-168`. Audit F14 `:329-340`.
- **Ruled:** nothing.
- **Fix:** `load_survivors(..., allow_fallback=False)`, and fail if `format != "npz"`.
- **Files:** `scorer_trial_worker.py`.
- **Test:** remove the NPZ while a `.json` sibling exists: `load_data` raises and does not load JSON (the harness already does this from outside).

### F15: no dataset identity in Step 2
- **Record:** the worker reads only the NPZ (`scorer_trial_worker.py:151-283`); the histories have no identity (`window_optimizer.py:1086-1105`); WATCHER P0.5 does nothing for Step 2. Audit F15 `:342-359`.
- **Ruled:** nothing.
- **Fix:**
  - Step 2 reads the generation sidecar beside the NPZ target. It checks that the sha256 of the NPZ equals the sidecar's `artifact_sha256`, and refuses on mismatch.
  - It writes `generation_id`, `artifact_sha256`, the dataset sha256 and `run_id` into a `_run_context` block of `optimal_scorer_config.json`. S107 already added a `_run_context` block once (`S107_session_log.md:59`, `:70`); the 2026-03 file has none (audit A.1).
- **Files:** `run_scorer_meta_optimizer.sh` (writer), `scorer_trial_worker.py` (check).
- **Test:**
  - a fixture generation with a correct sidecar passes, and `_run_context` matches;
  - a swapped NPZ is refused;
  - Step 3's config read (`generate_step3_scoring_jobs.py:201-209`) still works with the extra block.
- **NEEDS BETA** (provenance contract).

### F16: preflight blocks Step 2 when the rigs boot Proxmox
- **Record:**
  - `preflight_check.py:348-354` raises `SSH unreachable` for the config's bare-metal hosts.
  - WATCHER hard-fails on `ssh`/`unreachable` (`agents/watcher_agent.py:1866-1869`).
  - Audit F16 `:361-375`: "a boot-state consequence, not a defect."
- **Ruled:**
  - `CLAUDE.md` §3: the bare-metal addresses are deliberate and must not be "corrected".
  - The selectable rig profile is Phase 6 prep, "explicitly NOT done".
- **Fix:** none in code under the current ruling. The options are owner choices (O5):
  - (i) run Step 2 VM101-only, with an execution set of one local node, so there are no remote nodes to probe;
  - (ii) boot the rigs to bare metal;
  - (iii) authorize the rig-profile work.

  (i) also needs F9(c), so that preflight and dispatch read the same set.
- **Files:** none, for (i) beyond F9(c).
- **Test:** WATCHER Step 2 preflight with a one-node local execution set passes without contacting a rig. Socket guard as in the harness.

### B1 (Chapter 12): the file-exists authority
- **Record:** `CHAPTER_12_WATCHER_AGENT.md:725-736`; `TODO_CONSOLIDATED.md:91`.
- **Ruled and DONE:** **repaired at `b6cb789`** (2026-11-11, "Ch12 B1, B6, B3; 21/21 gates"; `tests/test_watcher_safety_repairs.py`). Live: `agents/watcher_agent.py:1099-1135` makes `success: False` override file evaluation. The TODO row and the Chapter 12 text predate it.
- **Residual:** a step that exits 0 with a degenerate result still passes on file existence. That is closed by the F10 fix (the shell exits non-zero), not by WATCHER.
- **Files:** none in WATCHER.
- **Test:** covered by F10's test, plus one WATCHER test: the Step 2 shell exits 1 with an old `optimal_scorer_config.json` present, and the decision is `step_failed` (an existing test class; extend it to Step 2).

### Stale `optimal_scorer_config.json`
- **Record:**
  - The file is **tracked** (`git ls-files optimal_scorer_config.json`), so every tree checks out the 2026-03-11 config (run H's copy is byte-identical; readiness note §1).
  - Recorded as a manual-cleanup chore: "Must manually `rm optimal_scorer_config.json scorer_jobs.json optuna_studies/*.db` before each re-run" (`S107_session_log.md:100`).
- **Why it matters now:**
  - **The freshness skip:** WATCHER skips a step whose output is newer than all required inputs (`agents/watcher_agent.py:962-994`, `:1980`; Step 1 alone is exempt, `:1990-1995`). In run H's tree the config (14:31) is older than the inputs (14:37), so Step 2 would run. Any tree whose checkout postdates its inputs would **skip Step 2 and hand Step 3 the March config.**
  - Step 3 reads `prng_type` and `mod` from it (`generate_step3_scoring_jobs.py:201-209`).
- **Ruled:** nothing.
- **Fix:**
  - `git rm --cached optimal_scorer_config.json` (and `scorer_jobs.json` if tracked), and add both to `.gitignore`. O4.
  - The F15 `_run_context` check gives Step 3 a way to refuse a config not bound to its generation.
- **Files:** git index and `.gitignore`; owner commits.
- **Test:** a fresh checkout has no `optimal_scorer_config.json`. A WATCHER freshness test: an old config whose `_run_context.artifact_sha256` does not match the current generation is treated as stale (needs a hand-landed WATCHER change, so put it with C4).

## 2. F6 in full: the eleven parameters

**Sampled at** `generate_scorer_jobs.py:52-66`, in `define_search_space`. `sample_size` (`:69`) and `optuna_trial_number` (`:119`) are added but not tuned.

**Where a parameter is dropped:** the "Step 2 objective" column names the worker line (`scorer_trial_worker.py`); the downstream columns name the step's line.

| Parameter (range) | Step 2 objective | Step 3 | Step 5 | Intent recorded (cite) | Owner decision? |
|---|---|---|---|---|---|
| `residue_mod_1` 5–20 | **used**: k-of-3 mask `:385-389` | dropped: `full_scoring_worker.py:344` builds `scorer_class(prng_type, mod)`; generator reads only `prng_type`, `mod` (`generate_step3_scoring_jobs.py:201-209`). The constructor would translate it (`survivor_scorer.py:100-107`) | — | Step 3 "literal residue filter" (S108 `:104`, `:118`); a residue FILTER selecting a subset (TB v4.1 request `:57-63`); Ch3 §6.1 "Small residue modulus → Step 3 SurvivorScorer" | **Yes:** wiring changes feature names `residue_{mod}_*` and `holdout_quality` (Ch4 B5 `:111-135`) |
| `residue_mod_2` 50–150 | used `:385-389` | dropped as above | — | as above | as above |
| `residue_mod_3` 500–1500 | used `:385-389` | dropped as above | — | as above | as above |
| `max_offset` 1–15 | used `:380-389` | dropped (constructor reads it `:108`, the call site does not) | — | "offset bounds" (S108 `:105`); "offset tolerance" (TB v4.1 `:62`). **Scale flaw**: "Needs TB approval before v4.5", proposal `keep_fraction` (S107 `:96`) | **Yes + Beta** (v4.5) |
| `temporal_window_size` 50–100 | **dropped**: read into `tw_size` `:347`, never used | dropped (constructor `:109`) | — | "Step 3 sliding window" (S108 `:106`, `:120`); "can control a temporal diversity bonus" (TB v4.1 `:110`, question 3 `:151`) | **Yes** |
| `temporal_num_windows` 3–10 | **dropped**: never read | dropped (constructor `:110`) | — | "Step 3 loop count" (S108 `:107`) | **Yes** |
| `min_confidence_threshold` 0.05–0.25 | **dropped**: never read | dropped (constructor `:111`) | — | "Step 3 confidence clamp" (S108 `:108`) | **Yes** |
| `hidden_layers` `128_64`/`256_128_64`/`512_256_128` | **dropped**: never read | — | dropped: Step 5 samples its own `'[128, 64]'` list-strings (`generate_anti_overfit_jobs.py:29-36`); `optimal_scorer_config` occurs 0 times in the four Step-5 files | "Step 5 NN architecture" (S108 `:109`, `:114-118`: "shared config serving multiple pipeline steps … All 11 Optuna dimensions are valid") | **Yes**: encoding and which study owns it |
| `dropout` 0.1–0.5 | dropped | — | dropped: own `0.1–0.5` (`:37`) | "Step 5 line 155" (S108 `:110`) | **Yes** |
| `learning_rate` 1e-4–1e-2 | dropped | — | dropped: own `1e-5–1e-2` (`:38`) | "Step 5 line 159" (S108 `:111`) | **Yes** |
| `batch_size` 32/64/128 | dropped | — | dropped: own `16/32/64/128` (`:39`) | "Step 5 line 160" (S108 `:112`) | **Yes** |

**What the record shows about intent:**
- **The only explicit statement is S108's:** Step 2 "finds a single config optimally serving all downstream consumers simultaneously … Do NOT remove". That is the design the owner's session closed on.
- **What the audits showed afterwards:** the consumers it names do not read the file (Ch4 B5; Ch3 §17 B2).
- **What is not recorded anywhere searched:** whether the owner, knowing that, wants the consumers wired or the parameters dropped. That is O1. **I do not infer it.**

**The code-only part of F6, separable from O1:**
- **Dead locals:** `bc_arr`, `ir_arr`, `fwd_arr`, `rev_arr`, `LAMBDA_SIZE`, `IR_WEIGHT`, `tw_size` (`scorer_trial_worker.py:339-374`).
- **The stale v4.2 docstrings:** `:3`, `:298`, `:311-322` (audit F3).
- **Fix:** remove the locals, or comment them as v4.2 remnants, and correct the docstrings. This changes no value: the objective at `:439-443` is unchanged.
- **Test:** objectives and masks are byte-identical before and after, on the run H NPZ, for 100 seeded trials. The harness draft is the instrument: run it at both commits and compare `trials.jsonl` and `trial_masks_packbits.npz`.

## 3. Grouped into the fewest landable changes

| Change | Items | Files | Lands via | Beta? |
|---|---|---|---|---|
| **C1, Step 2 launcher, NPZ only, fail-closed** | F1, F2, F9(a), F10, B1 residual; F11 (B4, the legacy branch's missing `\`) in passing | `run_scorer_meta_optimizer.sh`, a new shell-harness test | owner_land | **Yes**: removes a Beta-approved block on the D3.0-B surface |
| **C2, worker correctness** | F5/N1, F14, F6 code-only part, F3 docstrings | `scorer_trial_worker.py`, new unit tests; harness parity before/after | owner_land | No (no behaviour change except F5 and F14); Beta optional |
| **C3, one run root, one fleet** | F9(b)(c), enabling F16(i) | `scorer_trial_worker.py` (results dir), `coordinator.py` scorer collection, `scripts_coordinator.py` localhost cwd, the shell; consumes `execution_set.py` | owner_land | **Yes**: fleet resolution |
| **C4, WATCHER and manifest** | F12, F13, stale-config freshness binding, B1 Step-2 test | `agent_manifests/scorer_meta.json`, `agents/watcher_agent.py` (only if a valued flag is needed), `preflight_check.py`, tests | **owner by hand** (manifest, WATCHER) + owner_land (`preflight_check.py`, tests) | **Yes** (WATCHER) |
| **C5, Step 2 provenance** | F15, stale config `_run_context` | `run_scorer_meta_optimizer.sh`, `scorer_trial_worker.py`, `.gitignore`, `git rm --cached optimal_scorer_config.json` | owner_land + owner's git | **Yes** (provenance contract) |

**Not in any change until the owner rules:** F6 consumers (O1), F8 and the objective (O2; that needs Beta too), the F13 value (O3), F16 route (O5).

**Order:**
1. C2 first: no behaviour change, and it lets the harness prove parity.
2. C1 next.
3. C5.
4. C3.
5. C4 last, because it is hand-landed and depends on C1's argument form.

**First certified Step 2 on run H, once O1–O6 are answered:** C1, C2, C4 and C5, plus C3 if the fleet is kept (O5). With O5 = VM101-only, C3 shrinks to the results-dir part.

## 4. What this predicts for hit rate

**Nothing.** These are correctness and provenance repairs to a step that reads no draws.
- A repaired Step 2 yields a parameter set bound to its generation.
- The parameters reach pools only if O1 wires them into Step 3 and Step 5.
- The only settling measurement is unchanged: pools scored on unseen draws against `k/1000`, with the current pipeline as the baseline.
