# Chapter 4 audit — full scoring and feature extraction (WATCHER Step 3)

Repo: `/home/michael/distributed_prng_analysis` on VM 101, user `michael`. Code base = `a15b524`
(`a15b5247336255801de8cf67d52653701b1e8c80`). Current HEAD is `741822b`; `git diff --name-only a15b524 741822b`
touches only `docs/`, so the code at `741822b` is the code at `a15b524`. Every code anchor below is at that code.
**Working-tree caution:** during this pass another task left uncommitted edits in `agents/watcher_agent.py` (+88 lines)
and `agents/watcher_dispatch.py` (`git diff a15b524 --stat`). Every `agents/watcher_agent.py` anchor below was therefore
re-derived from `git show a15b524:agents/watcher_agent.py`, not from the working copy. No other code file read here differs
from `a15b524`. Those edits are not this pass's and were not touched.
Audited: `docs/CHAPTER_4_FULL_SCORING.md` (1037 lines at `a15b524`; byte-identical in the working tree before this pass).
Authority: `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md` REV1 (§0, §1, §3, §5) and the batch-2 common brief.
Method: read-only. Governance trail first, then chapters, then code, then host state. Every anchor below was opened in
this session. Nothing was executed except read-only inspection (`python3 -c` over JSON/NPZ files, `git log`, `ls`,
`systemctl list-units`, `crontab -l`). "Works" means the code path is complete on reading, not that it was run.
**No scoring, pipeline step or rig access was performed.**
Numbering: the EXECUTABLE scheme. Step 3 = `STEP_SCRIPTS[3] = "run_step3_full_scoring.sh"`
(`agents/watcher_agent.py:391`), manifest `full_scoring.json` (`:402`), name "Full Scoring" (`:413`). Chapter 4's
"Step 2.5" is executable Step 2 (scorer meta-optimizer); its "Step 5" is executable Step 5.
Framing held throughout: TFM is functional mimicry of surface output; no finding here assumes a "true seed".

---

## 1. Verdict

- Claims checked: **83**
- Accurate: **42**
- BROKEN: **7**
- SWITCHED-OFF-BY-RULING: **2**
- DESCRIPTION-ONLY: **15**
- UNVERIFIED: **2**
- Absorbed into an existing item (not counted separately): **15**

42 + 7 + 2 + 15 + 2 + 15 = **83**. The reconciliation table is §10.

Null results first:

1. **Step 2's optimized scorer parameters never reach Step 3 (B5).** The worker builds `SurvivorScorer(prng_type, mod)`
   with no `config_dict` (`full_scoring_worker.py:344`); the job generator reads only `prng_type`/`mod` from
   `optimal_scorer_config.json` (`generate_step3_scoring_jobs.py:201-209`); the worker has no `--config` argument
   (`:522-550`). Host evidence: the live config carries `residue_mod_1/2/3 = 17/150/1449`, `temporal_num_windows = 9`,
   `min_confidence_threshold = 0.2189`; the host `survivors_with_scores.json` (84 records, written 2026-03-11) carries
   `residue_8_*`, `residue_125_*`, `residue_1000_*` only. **Not found in the governance trail.** This is the one new finding
   of this audit with pipeline-wide reach.
2. **`best_offset` has never been computed (B2).** Hardcoded `0.0` since the initial commit `0101306`
   (`survivor_scorer.py:373`, `:707`); `self.max_offset` is assigned (`:108`) and read nowhere in tracked `*.py`.
3. **The feature contract is 91 extracted / 89 trained, not 47 (D1).** Re-derived from the producer, not taken from the
   trace: 77 per-seed keys from `SurvivorScorer` + 14 `global_*` = 91; Step 5 excludes `score`, `confidence`
   (`meta_prediction_optimizer_anti_overfit.py:1479-1482`) → 89. Agrees with committed `docs/CHAPTER_14_AUDIT.md` Q3.
   The 77 depends on the default three residue mods; see B5 for why that matters.

Everything else that disagrees is either governed (B6, B7, S1, S2) or a stale description.

---

## 2. Brief questions and the feature contract

| # | Question | Result | Evidence |
|---|---|---|---|
| Q1 | Numbering used by Chapter 4 | **Executable.** Title "(Step 3)" = `STEP_SCRIPTS[3]`. Chapter's "Step 2.5" = executable Step 2 | `agents/watcher_agent.py:387-395`, `:402`, `:413`; `docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.1 (`:42-60`), Step table `:101` |
| Q2 | Feature count, re-derived from live code | **91 extracted / 89 trained.** Per-seed (batch path): base 5 (`survivor_scorer.py:703-707`) · stats 10 (`:708-717`) · lane 3 (`:718-720`) · digit 4 (`:722-725`) · temporal 5 (`:726-730`) · residue 9 = 3 mods × 3 (`:624-667`, merged `:734`) · battery 23 (`:754-768`; names `:202-225`) · metadata 18 (`:774-779`, defaults `:784-791`) = **77**. Plus 14 `global_*` (`full_scoring_worker.py:347-349`, `:403-405`; names `models/global_state_tracker.py:366-381`) = **91**. Sequential path also yields 77 per-seed keys (`survivor_scorer.py:368-461`) when `n ≥ temporal_window_size/2`; below that the 5 temporal keys are absent (`:404`) | Host: `survivors_with_scores.json` first record has 91 `features`, 14 `global_*`, 77 `holdout_features` |
| Q3 | Composition | 13 NPZ-backed live metadata fields + 5 dead placeholders (B7) + 59 computed from the regenerated sequence + 14 run-global. Same as system map §7.1 (`docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:144-148`) and Ch14 Q3 | — |
| Q4 | Ch14 B5 (`f{i}` keys) | Not a Step-3 defect. Step 3 emits name-keyed dicts (`full_scoring_worker.py:390-405`); the names are lost downstream (committed Ch14 B5). Cross-referenced, not re-reported | `docs/CHAPTER_14_AUDIT.md:227` |
| Q5 | Ch13 B2 dataset authority at `full_scoring.json:120` | Confirmed at this code: `default_params.train_history = "train_history.json"`, `holdout_history` `:121`. WATCHER's P0.5 rewrite only touches `lottery_file`/`lottery_data`/`lottery_history` (`agents/watcher_agent.py:474`, `:1860`), so Step 3's two history files are not resolved through the dataset authority. Cross-referenced, not re-reported | `docs/CHAPTER_13_AUDIT.md:80-86` |
| Q6 | Ch9 O-6 | Confirmed for Step 3: `run_step3_full_scoring.sh:74-81` (REMOTE_NODES), `ramdisk_preload.sh:26-40`, `scripts_coordinator.py:430-448` read `distributed_config.json` hostnames (`.120/.154/.162`), not the resolved execution set. WATCHER does freeze a set at the top of every `run_step` (`agents/watcher_agent.py:1748`), but nothing on the Step-3 path reads it | `docs/CHAPTER_9_AUDIT.md:162`; `docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md:1742` |
| Q7 | `agent_manifests/definitions.json` (gitignored) | Read on the host: 220 lines, schema `1.1.0`, `updated_at 2026-01-02T03:00:00Z`; `pipeline_steps` has only `step5_anti_overfit` and `step6_prediction`. **No Step-3 entry.** No reader in tracked `*.py`/`*.sh` (`/bin/grep -rn definitions.json`, no hits). It neither constrains nor describes Step 3 | `git status --ignored` shows `!! agent_manifests/definitions.json` |

---

## 3. What actually works end to end today

Each item is a complete code path on reading. None was run. The last Step-3 output on the host is from 2026-03-11, before RANGE-MINER.

1. **WATCHER dispatch.** `run_step(3)` loads `full_scoring.json` `default_params` (`agents/watcher_agent.py:1799-1812`), applies step-scoped `--params` overrides (`:1813-1820`), builds `bash run_step3_full_scoring.sh --survivors … --chunk-size 1000 …` (`:2081-2154`). The script ignores the options it does not parse (`run_step3_full_scoring.sh:120-123`; manifest note `full_scoring.json:141`).
2. **Ramdisk preload to every configured node** (`run_step3_full_scoring.sh:33-40` → `ramdisk_preload.sh:79-150`): NPZ, scorer config, train and holdout history to `/dev/shm/prng/step3/` on localhost and each `distributed_config.json` node.
3. **Job generation.** NPZ rows become per-seed dicts carrying all 22 arrays (`generate_step3_scoring_jobs.py:76-93`), chunked (`:212-233`), each job pointing at the ramdisk histories (`:236-248`).
4. **Distribution and execution.** Chunks scp'd to remotes (`run_step3_full_scoring.sh:203-215`); `scripts_coordinator.py` rewrites outputs into a run-scoped directory (`:412-415`, `:465-471`), runs `distributed_worker.py` per GPU with HIP/CUDA visibility (`:964-997`), retries failures on localhost (`:1068-1112`).
5. **Feature extraction (java_lcg / java_lcg_hybrid).** GPU-batched path (`survivor_scorer.py:547-800`), metadata merge of 18 fields (`:770-781`), 14 globals (`full_scoring_worker.py:403-405`), `holdout_hits` with the offset law (`:274-310`, `:364-370`), `holdout_quality` + `holdout_features` (`:408-430`).
6. **Aggregation.** Latest run directory, global merge, sort by score, write `survivors_with_scores.json` (`run_step3_full_scoring.sh:299-408`); validation floor `< 46` (`:475-478`, governed BACKLOG §15).

---

## 4. BROKEN

**B1 — The ROCm prelude and the 80 % VRAM cap never fire on the live rigs.** *(Same class GOVERNED for `sieve_filter.py` as Chapter 2 F-5, §9.5, and for `distributed_worker.py` as Chapter 9 B-1; the four Step-3 instances here are not named there.)*
- Chapter: §1.2 BUG FIX 3/4, §1.3, §2.1, §2.2, §13.2 present them as active on "RX 6600 rigs".
- Code: `if HOST in ["rig-6600", "rig-6600b", "rig-6600c"]` at `survivor_scorer.py:22` and `:61`; `full_scoring_worker.py:90` and `:99`. Live worker hostnames are `rrig6600`, `rrig6600b`, `rrig6600c` (`rig_profiles_config.json:30`, `:42`, `:53`).
- Consequence on reading: `PYTORCH_HIP_ALLOC_CONF`, `set_per_process_memory_fraction(0.8)` and `cudnn.benchmark = False` are not applied on any rig. `setup_gpu_environment` takes the CUDA branch on rigs (`full_scoring_worker.py:103-105`); GPU isolation still holds because `scripts_coordinator.py:967-971` exports `HIP_VISIBLE_DEVICES`.
- Stops at: `survivor_scorer.py:22`, `:61`; `full_scoring_worker.py:90`, `:99`.
- Repair would have to: follow Chapter 2 §9.5 — decide whether the prelude is still supported; if retained, key it from the declared rig profile; never rename hosts or write another hostname tuple.

**B2 — `best_offset` is "best alignment offset 0–5"; no alignment search exists.** *(not found in the governance trail; the constant itself is recorded in Ch14 Q3)*
- Chapter: §3.1 `DEFAULT_MAX_OFFSET = 5` "Maximum offset for alignment", §4.3 `max_offset` "Alignment offset", §9.1 `best_offset` range 0–5.
- Code: `'best_offset': 0.0` (`survivor_scorer.py:373`), `torch.zeros(batch_size)` (`:707`). `self.max_offset` assigned at `:108`, read nowhere (`/bin/grep -rn '\.max_offset' --include=*.py`: only `survivor_scorer.py:108` among Step-3 code). Present as `0.0` since `0101306` (`git show 0101306:survivor_scorer.py`, `:142`). Step 2 still tunes `max_offset` (host `optimal_scorer_config.json`: 13).
- Stops at: `survivor_scorer.py:373`, `:707`.
- Repair would have to: either implement the offset search the chapter describes or classify `best_offset` as a dead column under P0-A (system map §13) — a schema decision, not a patch.

**B3 — `batch_score_vectorized`'s CPU fallback crashes and is not mod-reduced.** *(not found in the governance trail)*
- Chapter: §7.2 shows `else: # CPU fallback predictions = self._cpu_batch_generate(...)`.
- Code: `predictions` is assigned only inside `if has_pytorch_gpu(...)` (`survivor_scorer.py:496-504`); for a family without a PyTorch implementation `if predictions is None` (`:505`) raises `UnboundLocalError`. When reached via a GPU exception, the fallback stores raw 32-bit outputs (`:510-511`) — the registry CPU reference returns `(state >> 16) & 0xFFFFFFFF` (`prng_registry.py:181`) — and compares them with draws in `[0, 999]`.
- Reach: no pipeline caller of `batch_score_vectorized` at `a15b524` (only a docstring mention, `scorer_trial_worker.py:36`).
- Stops at: `survivor_scorer.py:505`.
- Repair would have to: initialise `predictions`, and reduce fallback outputs by `self.mod` as `_generate_sequence` does (`:125`).

**B4 — The batched feature path's CPU fallback is not mod-reduced, silently.** *(not found in the governance trail)*
- Chapter: §8.2 "PyTorch GPU PRNG or CPU fallback".
- Code: `_cpu_batch_generate` stores `seq[:n]` raw (`survivor_scorer.py:802-809`), used at `:578` (GPU exception) and `:581` (no PyTorch implementation). Only `java_lcg` and `java_lcg_hybrid` have `pytorch_gpu` (`prng_registry.py:3961`, `:3974`; live registry query: 2 of 44). Every other family, and any GPU failure, produces features from raw 32-bit values against 3-digit draws. The record keeps all 91 keys.
- Reach: TFM sieves `java_lcg` today (`CLAUDE.md` §7), so the default path is not hit; any other `prng_type` or GPU exception reaches it.
- Stops at: `survivor_scorer.py:808`.
- Repair would have to: apply `% self.mod` in the fallback, and include the fallback in F-PARITY (system map §7.3).

**B5 — Step 2's optimized scorer parameters do not reach Step 3.** *(not found in the governance trail)*
- Chapter: §4.2 and §12.5 present the Optuna translation as the route by which Step 2's `residue_mod_1/2/3` reach the scorer; the manifest describes Step 3 as "Applies optimized scoring parameters from Step 2.5 to ALL survivors" (`agent_manifests/full_scoring.json:3`).
- Code: the constructor does translate them (`survivor_scorer.py:97-111`), but Step 3 never calls it with a config: `scorer = scorer_class(prng_type=prng_type, mod=mod)` (`full_scoring_worker.py:344`); no `--config` argument (`:522-550`); the generator takes only `prng_type` and `mod` from the config file (`generate_step3_scoring_jobs.py:201-209`) and passes neither the file nor its other keys to the job (`:241-248`). `git log -S config_dict -- full_scoring_worker.py generate_step3_scoring_jobs.py`: no commit ever added it.
- Host evidence: `optimal_scorer_config.json` (mtime 2026-03-11) = `residue_mod_1 17, residue_mod_2 150, residue_mod_3 1449, max_offset 13, temporal_num_windows 9, min_confidence_threshold 0.2189`; host `survivors_with_scores.json` (2026-03-11 17:03 PDT, 84 records) has residue keys for 8/125/1000 only.
- Why the repair is not a one-liner: applying the config renames the residue features (`residue_{mod}_*`, `survivor_scorer.py:631`, `:666-667`), which changes the 91/89 name contract, and `compute_holdout_quality` reads `residue_1000_*`, `residue_8_*`, `residue_125_*` by name (`holdout_quality.py:61-82`), so those inputs would read as absent.
- Related committed text: `docs/CHAPTER_3_ALIGNMENT_AUDIT.md:211-214` says the seven scorer parameters "**are** consumed by Step 3's `SurvivorScorer`". That is true of the constructor and not of the Step-3 call site. Listed in §9 of the return report as an update needed elsewhere; Chapter 3 is not edited here.
- Stops at: `full_scoring_worker.py:344`.
- Repair would have to: a governed decision on whether Step 2's parameters should drive Step 3; if yes, carry them to the worker and version the feature schema (P0-A), and key `holdout_quality` from the configured mods.

**B6 — The sequential fallback merges 6 of the 18 metadata fields.** *(GOVERNED: system map §7.3 defect 2, `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:162-163`; P0-B "Authorized, P0" `:252-253`, `:270`; TODO_CONSOLIDATED P0-B)*
- Chapter: §8.5/§9.6 describe the metadata features as populated from survivor metadata.
- Code: batch merges 18 (`survivor_scorer.py:774-779`); the fallback after a batch exception (`full_scoring_worker.py:438-443`) merges `forward_count, reverse_count, bidirectional_count, skip_min, skip_max, skip_range` only (`:451-456`); the other seven NPZ-backed fields become `0.0` via `extract_ml_features`' defaults (`survivor_scorer.py:455-461`). No `F-PARITY` in any tracked `*.py` (`/bin/grep -rln 'F-PARITY\|F_PARITY'`: none).
- Stops at: `full_scoring_worker.py:453`.
- Repair would have to: what P0-B already authorizes — identical merges, exact required-field validation, F-PARITY, fail closed.

**B7 — Five "metadata" features have no producer.** *(GOVERNED as a fact: system map §7.1 "5 dead placeholders" `:147`; P0-A "classify … dead fields" `:250`; Ch14 Q3)*
- Chapter: §9.6 lists `skip_mean`, `skip_std`, `skip_entropy`, `survivor_velocity`, `velocity_acceleration` with source "Metadata".
- Code: in the merge list (`survivor_scorer.py:776-778`), absent from the 24 canonical record fields (`utils/canonical_records.py:115-123`) and from the 22 NPZ arrays (`convert_survivors_to_binary.py:51-74`; host NPZ `files` = the same 22), so always `0.0` (`survivor_scorer.py:784-791`).
- Stops at: `utils/canonical_records.py:115`.
- Repair would have to: P0-A — classify them as dead in the canonical manifest, or give them a producer under a new schema version.

---

## 5. SWITCHED-OFF-BY-RULING

**S1 — "Holdout data is the CONTINUATION of training data in the PRNG sequence" holds only for skip = 0.**
- Chapter: §1.4, closing sentence; §10.5 `compute_holdout_hits_batch` "offset is DERIVED".
- Code: the offset law is implemented exactly (`full_scoring_worker.py:299-305`, `:367`), but `java_lcg_cpu` applies `skip` once as a burn-in (`prng_registry.py:176-177`) where the sieve kernel applies it between every draw; they agree only at skip 0. `holdout_features` (the input to the Step-5 target) are generated with `skip = get_survivor_skip(meta)` (`skip_min`, `holdout_quality.py:26-56`) as a burn-in through the same CPU reference (`survivor_scorer.py:124`), with no train-length offset (`full_scoring_worker.py:411-418`). The per-survivor skip replacing the uniform offset is the S111 design (`docs/S111_IMPLEMENTATION_PLAN_FINAL.md:481`, approved `:6`).
- Held by: `docs/BACKLOG.md` §8.1 (`:157-170`), "Two read-only audits Beta wants before Phase 7. **No fix is authorised for either**", naming exactly `survivor_scorer.py:124` and `full_scoring_worker.py:305`; also `docs/KNOWN_ANSWER_VALIDATION_INVENTORY.md` §5 (`:297-337`), `docs/TODO_CONSOLIDATED.md:65`, `:164`. No standalone `TB_RULING_*` file for it was found (30 listed).
- Lifting it would enable: a repair that makes Step 3's CPU regeneration follow the kernel's per-draw skip, so that for skip > 0 survivors `holdout_hits` and `holdout_features` are computed at the PRNG positions the sieve actually tested.

**S2 — The per-seed sieve match rates are not Step-3 features.**
- Chapter: §1.1 "Bidirectional Features: Forward/reverse intersection analysis"; §9.6.
- Code: every sieve-derived feature that reaches the model is a trial-level count (`survivor_scorer.py:774-779`; `utils/canonical_records.py:214-239`). `forward_matches` / `reverse_matches` — per-seed (`utils/canonical_records.py:246-251`; `convert_survivors_to_binary.py:16-20`) — ride in every chunk (`generate_step3_scoring_jobs.py:81-93`) and are not merged.
- Held by: TB §8 ruling recorded in `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:165-171` ("do NOT silently add them"), authorization table `:276` ("Requires separate feature-schema decision + retraining plan"). Cross-referenced in `docs/CHAPTER_14_AUDIT.md` Q4 and `docs/PIPELINE_BEHAVIOUR_MODEL.md` §5.
- Lifting it would enable: Option A — the two per-seed sieve match rates entering the training vector under a new schema version, after leakage/redundancy analysis and full retraining; or Option B — recording why they must stay out.

---

## 6. DESCRIPTION-ONLY

| # | Chapter says | Code at `a15b524` | Anchor |
|---|---|---|---|
| D1 | 46 / 47 features (§1.1, §1.3, §5.1, §9.6 header "14", §9.8, §13.3, §14), categories omit digit, battery, global | 77 per-seed + 14 global = 91; 89 trained. §9.6 table has 18 rows | §2 Q2 above |
| D2 | `holdout_hits` is "the ML y-label for Step 5", "the ONLY non-circular training target", and a feature (§1.4, §9.7) | Step 5 target is `holdout_quality` (S111); `holdout_hits` is excluded from training and is a top-level record field, not in `features` | `meta_prediction_optimizer_anti_overfit.py:611`, `:1479-1482`; `full_scoring_worker.py:407`, `:423`; host record keys |
| D3 | WATCHER's preload is "Standalone mode" and "only populates the local node" (§2.5) | Preload copies to every node in `distributed_config.json`; "Standalone" (unset `WATCHER_MANAGED_RAMDISK`) only controls clearing | `ramdisk_preload.sh:26-40`, `:86-87`, `:104-150` |
| D4 | Verification/population commands to `.120` and `.154`, with "rig-6600 / rig-6600c" at `.120` (§2.5) | Config nodes are `.120`, `.154`, `.162`; under the Proxmox boot the endpoints are the CT100 addresses | `distributed_config.json` (parsed); `CLAUDE.md` §3 |
| D5 | Missing ramdisk file = "immediate argument parsing failure" (§2.5) | Argparse accepts the path; the load fails, is caught, prints a JSON error and exits 1 | `full_scoring_worker.py:579-614` |
| D6 | "12 concurrent workers", "12-worker launch" (§1.5, §2.2) | 8 GPUs per rig, `max_concurrent_script_jobs` 8; localhost 1 GPU, 2 jobs | `distributed_config.json` (parsed) |
| D7 | 8, 125, 1000 give "orthogonal validation" (§3.2) | By CRT (gcd(8,125)=1, 8·125=1000) a mod-1000 match is exactly a mod-8 and a mod-125 match; the third lane is not independent | `survivor_scorer.py:74`, `:377-390` |
| D8 | `java_lcg_sequence` is the "CPU-only fallback" (§3.3) | Defined, never called in live code; it applies Java's XOR scramble, which the registry reference deliberately does not | `survivor_scorer.py:81-87`; `prng_registry.py:120-122`, `:175`; grep of tracked `*.py` |
| D9 | `extract_ml_features(seed, history, forward_survivors, reverse_survivors, skip)` (§5.1) implies use of the survivor lists | Both list parameters are accepted and ignored; bidirectional keys are zero-filled | `survivor_scorer.py:350`, `:455-461` |
| D11 | "SINGLE CPU TRANSFER" (§8.2, §8.4) | A second transfer `predictions.cpu()` feeds a per-seed CPU battery computation | `survivor_scorer.py:746-757` |
| D12 | Used by `reinforcement_engine.py`, `meta_prediction_optimizer.py`, `coordinator.py` (§10.3) | `coordinator.py` does not import it; Step 3's caller `full_scoring_worker.py` and `prediction_generator.py` are not listed | `full_scoring_worker.py:565`; `prediction_generator.py:58`, `:263`; `meta_prediction_optimizer.py:34`; `reinforcement_engine.py:100` |
| D13 | `run_step3_full_scoring.sh` "scp to remotes" for holdout (§10.5) | Static files go by ramdisk preload; Phase 2 copies only chunk files | `run_step3_full_scoring.sh:36-40`, `:201`, `:212` |
| D14 | Example prints `[789, 101, 102]` and Jaccard 0.375 (§12.4) | Output is sorted `[101, 102, 789]`; 3 / 7 = 0.4286 | `survivor_scorer.py:165-173` |
| D15 | Header "Lines ~550"; summary "Chapter 6: Survivor Scorer"; "Next Chapter: Chapter 7: GPU Optimizer"; end string "End of Chapter 6" | `survivor_scorer.py` 809 + `full_scoring_worker.py` 735 lines; the live next chapter is Chapter 5 (ML architecture optimizer) | `wc -l`; `docs/PIPELINE_BEHAVIOUR_MODEL.md` §15.1 item 4 (already recorded) |
| D16 | "AMD RX 6600" (§1.3, §2.2) | The rigs are RX 6600 XT (gfx1032). Not re-checked on the rigs (no rig access) | `docs/PIPELINE_BEHAVIOUR_MODEL.md:66`; `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` §9.5 |

(D10 was merged into B3 during the audit and the number is not reused.)

---

## 7. RANGE-MINER, the 22-array NPZ contract and the window-anchor change — what they alter in Step 3

1. **The input bundle.** `bidirectional_survivors_binary.npz` is now RANGE-MINER output: on the host it is a symlink to `.s172_accumulator/current/bidirectional_survivors_binary.npz` with exactly the 22 frozen arrays (live `np.load(...).files`; contract `convert_survivors_to_binary.py:51-74`). **Point-in-time:** that bundle has 0 rows (`provenance.json` `final_row_count 0`, generation `gen-20260818T013219550933Z-step1_java_lcg_0`). Step 3 has not run on RANGE-MINER output: the newest `full_scoring_results_*` and `survivors_with_scores.json` are from 2026-03-11.
2. **What Step 3 reads from it.** All 22 arrays become per-seed chunk fields (`generate_step3_scoring_jobs.py:76-93`); 13 live fields + 5 dead names are merged (`survivor_scorer.py:774-779`); `skip_min` drives `holdout_features` (`holdout_quality.py:47`). `seeds` drives every computed feature. `forward_matches`, `reverse_matches`, `window_size`, `offset`, `trial_number`, `score`, `skip_mode`, `prng_type` are carried and not merged (S2; Ch14 Q4).
3. **Window anchor.** After Brief I, NPZ array 4 `offset` *is* the window anchor and never the generator phase (`utils/canonical_records.py:216-225`, `:368-374`). Step 3 does not merge it and does not use it in regeneration (sequences start at PRNG position 0 with `skip=0`, `survivor_scorer.py:574-575`). **Naming hazard:** Chapter 4's "offset is a law" (§1.4) is a different quantity — the holdout start `len(train_history)` — and is unrelated to NPZ `offset`.
4. **`prng_type`.** The chunk carries the per-row encoded `prng_type`, but the worker regenerates every seed with the single CLI `--prng-type` (`full_scoring_worker.py:530-531`, `:344`), which the generator takes from the config file or its default `java_lcg` (`generate_step3_scoring_jobs.py:206`, `:309`). The live config has no `prng_type` key, so `java_lcg` is used for every row, including rows the sieve labelled `java_lcg_hybrid`. Recorded as an observation and not labelled: the chapter makes no claim about per-row family, and Chapter 3 audit F8 already records the hybrid-population contact point.
5. **Unchanged by RANGE-MINER:** B6, S2 and the dead placeholders are pre-existing ("NOT caused by, and not fixable by, RANGE-MINER", system map §7.3 heading `:160`).

---

## 8. Surfaces searched, and surfaces unavailable

**Searched and read**
- Governance: `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md` (§7, §13-15), `docs/BACKLOG.md` (§8.1, §15, §16), `docs/TODO_CONSOLIDATED.md` (Step-3 rows), `docs/PIPELINE_BEHAVIOUR_MODEL.md` (§1.1, §5, §12, §15.1), `docs/S111_IMPLEMENTATION_PLAN_FINAL.md` (grep), `docs/KNOWN_ANSWER_VALIDATION_INVENTORY.md` (§5, grep), `docs/CHAPTER_3_ALIGNMENT_AUDIT.md` (F7-F8), list of all 30 `docs/TB_RULING_*`; greps across `docs/` for `optimal_scorer_config`, `config_dict`, `skip=0`, `non-zero skip`, `_cpu_batch_generate`, `_vectorized_scoring_kernel`, `F-PARITY`, `rrig6600` + Step-3 file names.
- Chapters: `CHAPTER_4_FULL_SCORING.md` (all), `CHAPTER_2_BIDIRECTIONAL_SIEVE.md` (§0, §9.5, outline; not edited), `CHAPTER_13_AUDIT.md` (layout, B2, §8-9, A.10), `CHAPTER_14_AUDIT.md` (Q3, Q4, B5), `CHAPTER_9_AUDIT.md` (B-1, O-6, accurate list), `CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md` (O-6 rows).
- Code, read in full: `survivor_scorer.py`, `full_scoring_worker.py`, `run_step3_full_scoring.sh`, `generate_step3_scoring_jobs.py`, `agent_manifests/full_scoring.json`. In part: `agents/watcher_agent.py`, `scripts_coordinator.py`, `ramdisk_preload.sh`, `utils/canonical_records.py`, `convert_survivors_to_binary.py`, `utils/survivor_loader.py`, `holdout_quality.py`, `prng_registry.py`, `models/global_state_tracker.py`, `meta_prediction_optimizer_anti_overfit.py`, `run_step3_full_scoring_v2.sh` (header), `agents/contexts/full_scoring_context.py` (grep).
- Gitignored / host: `agent_manifests/definitions.json`; `git status --ignored`; `git ls-files -o -i --exclude-standard` (Step-3 hits: `scoring_jobs*.json`, `step3_debug.log`, `step3_diagnostic.log`, `fix_scoring_worker_metadata.py`, `results_archive/*/survivors_with_scores.json`, `docs/CHAPTER_4_FULL_SCORING.md.bak` 995 lines); host `optimal_scorer_config.json`, `survivors_with_scores.json`, `scoring_jobs.json`, `bidirectional_survivors_binary.npz` + `provenance.json`, `distributed_config.json`, `rig_profiles_config.json`; `systemctl list-units --all` (no Step-3 unit; `nvidia-gpu-policy.service` is failed, unrelated); `crontab -l` (none).
- Git history: `git log -S config_dict`, `git log -S best_offset`, `git log -- full_scoring_worker.py / survivor_scorer.py`, `git show 0101306:survivor_scorer.py`.
- Absence claims were made with `/bin/grep -rn` over tracked and untracked `*.py`/`*.sh` (excluding `backups/`, `apply_s*`, `verify_s*`, `step6_restoration/`), then reading each hit.

**Not available, and where it matters**
- **Rigs** (no SSH by instruction): `/dev/shm/prng/step3/` contents on `.120/.154/.162` or the CTs, rig hostnames live (taken from `rig_profiles_config.json`), GPU SKU (D16). **UNAVAILABLE.**
- `/dev/shm/prng/` on VM 101 does not exist now (`ls`: no such directory) — consistent with no Step 3 since the last reboot; not a defect.
- **Not read:** `step3_debug.log`, `step3_diagnostic.log`, `fix_scoring_worker_metadata.py`, `results_archive/`, the S1xx changelogs other than those named, the `apply_s*` patch corpus (which the fact file warns carries governance in docstrings). A B-item marked "not found in the governance trail" was not checked against the patch corpus.

---

## 9. Accurate claims and unverified claims (for the count)

**Accurate (42)**
1. Five BUG FIX entries (§1.2) — `survivor_scorer.py:11-15`, implemented at `:22-28`, `:59-67`, `:97-105`, `:392-403`, `:359-363`.
2. `offset = len(train_history)` for `holdout_hits` (§1.4) — `full_scoring_worker.py:299-305`, `:367`.
3. `holdout_hits` = hits / holdout draws in [0, 1] (§1.4) — `:306-308`.
4. Manual default `CHUNK_SIZE=1000` at line 70 (§1.5) — `run_step3_full_scoring.sh:70`.
5. `--chunk-size` CLI override (§1.5) — `:104-107`, passed `:164`.
6. WATCHER passes manifest `default_params` explicitly, `chunk_size` 1000 (§1.5) — `agents/watcher_agent.py:1799-1812`, `:2135-2154`; `full_scoring.json:122`.
7. `--params` override is step-scoped (§1.5) — `agents/watcher_agent.py:1813-1820`.
8. The two places to change `chunk_size` (§1.5) — `run_step3_full_scoring.sh:70`; `full_scoring.json:122` (the `sed` pattern matches only that line; `parameter_bounds.chunk_size` is an object, `:66`).
9. ROCm environment block as printed (§2.1) — `survivor_scorer.py:18-31` (that it never fires is B1).
10. VRAM-limit block as printed (§2.2) — `:59-68`.
11. Safe `entropy` wrapper (§2.3) — `:37-42`.
12. Ramdisk path `/dev/shm/prng/step3/` with train/holdout history (§2.5) — `generate_step3_scoring_jobs.py:236-237`; `ramdisk_preload.sh:20-22`.
13. Default constants (§3.1) — `survivor_scorer.py:73-78`.
14. Constructor signature (§4.1) — `:90`.
15. Optuna `residue_mod_1/2/3` translation (§4.2) — `:97-105`.
16. Instance attributes and defaults (§4.3) — `:107-117`.
17. `_generate_sequence` via the registry, `% mod` (§4.4) — `:119-125`.
18. `_coerce_seed_list` (§4.5) — `:128-137`.
19. `extract_ml_features` signature (§5.1) — `:350`.
20. Two-step tensor transfer (§5.3) — `:361-363`.
21. Residue match rate / coherence / KL (§5.4) — `:377-390`.
22. Temporal stability, reusing `seq` (§5.5) — `:395-413`.
23. Dual-sieve intersection, return structure, Jaccard, "never discard" docstring (§6) — `:139-180`.
24. `batch_score_vectorized` signature (§7.1) — `:518-519`.
25. Legacy `batch_score` (§7.3) — `:537-545`.
26. `extract_ml_features_batch` signature and docstring (§8.1) — `:547-555`.
27. Batch step 1 shape `(batch_size, n)`, GPU-or-fallback routing (§8.2) — `:565-586` (fallback correctness is B4).
28. Vectorized `scatter_add_` histogram (§8.3) — `:636-656`.
29. Stack-and-transfer of the GPU dict (§8.4) — `:702-739` (the second transfer is D11).
30. Metadata merge on the batch path for the named fields (§8.5) — `:770-781`.
31. Base features 5 (§9.1, apart from `best_offset`, B2) — `:703-707`.
32. Residue features 9 for the default mods (§9.2) — `:623-667`.
33. Temporal features 5 (§9.3) — `:726-730`.
34. Statistical features 10 (§9.4) — `:708-717`.
35. Lane features 3 (§9.5) — `:718-720`.
36. The 18 metadata names in the §9.6 table — `:774-779` (header count is D1; five have no producer, B7).
37. Imports from `prng_registry` (§10.2) — `:45-51`.
38. Files added for holdout: worker `--holdout-history` + `compute_holdout_hits_batch`; generator `--holdout-history` (§10.5) — `full_scoring_worker.py:546-547`, `:274`; `generate_step3_scoring_jobs.py:244`, `:325`; `run_step3_full_scoring.sh:68`.
39. `compute_holdout_hits_batch` body (§10.5) — `full_scoring_worker.py:274-310`.
40. Method and module-function tables (§11.1, §11.2), as far as they go — `survivor_scorer.py:38`, `:81`, `:90-809`.
41. Optuna example yields `residue_mods [14, 137, 1136]` (§12.5) — `:100-107`.
42. `torch.cuda.empty_cache()` after scoring (§13.1) — `:464-465`, `:796-797`.

**UNVERIFIED (2)**
- **U1** OOM memory figures: ~500 MB per 1000-seed chunk, ~1.5 GB per 5000, OOM at 7+ concurrent (§1.5). Would verify: a measured peak-RSS/VRAM log of a Step-3 chunk on a rig.
- **U2** That the two-step transfer "stabilizes data transfer on the RX 6600 PCIe bus" (§5.3). Would verify: a recorded A/B on a rig.

---

## 10. Count reconciliation

| Bucket | Count | Items |
|---|---|---|
| Accurate | 42 | §9 items 1–42 |
| BROKEN | 7 | B1–B7 |
| SWITCHED-OFF-BY-RULING | 2 | S1, S2 |
| DESCRIPTION-ONLY | 15 | D1–D9, D11–D16 (D10 merged into B3, number not reused) |
| UNVERIFIED | 2 | U1, U2 |
| Absorbed into an existing item, not counted separately | 15 | into **D1** (8): §1.3 "47 ML Features", §5.1 docstring "46", §9.6 header "(14)", §9.8 total 47, §13.3 "All 46", §14 "46 ML features per seed", §5.2/§8.2 flow omits digit and battery steps, §11.1 omits `compute_battery_features`; into **D2** (1): §10.5 output format lacks `holdout_quality`/`holdout_features`/`metadata`; into **D3** (1): §2.5 "Required Files" lists 2 of the 4 preloaded; into **B1** (3): §2.2 "OOM Killer intervention prevented", §13.2 RX 6600 specifics, §1.3 "Memory Management — VRAM limits for 8GB GPUs"; into **B3** (1): §7.2 snippet omits the `try/except` and `**default_params` (was D10); into **D15** (1): "Next Chapter" pointer |
| **Total claims checked** | **83** | 42 + 7 + 2 + 15 + 2 + 15 = 83 |

No UNVERIFIED item was resolved in this pass, so none needs a later placement.

---

## 10a. Chapter edit record and clean control

- Chapter `docs/CHAPTER_4_FULL_SCORING.md`: 1037 → ~1566 lines, all changes made with the Edit tool; every pre-existing
  heading still present (heading diff against `git show a15b524:…`, new headings filtered: identical); `git diff --numstat`
  shows 3 lines changed (the Version/File/Lines header lines, original wording kept and annotated) and no deletions otherwise.
- This audit file was created once with Write; every later change used Edit.
- Clean control: `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` unmodified (`git diff --quiet`); `python3 tests/test_chapter2_content_gate.py`
  → "12/12 checks green (6 gates + 6 mutants) · SENTINEL : PASS".
- Completion sentinel for Chapter 4: **PASS**.

## 11. Observations outside the claim count (not chapter claims; recorded, not labelled)

- `full_scoring.json` declares outputs `scoring_statistics.json` (`:21`) and `feature_registry` `config_manifests/feature_registry.json` (`:13`); no tracked `*.py`/`*.sh` writes the first or reads the second. WATCHER's `_find_results` looks for `full_scoring_results.json` for step 3 (`agents/watcher_agent.py:2220`), which nothing writes. The actions/`STEP_SCRIPTS` divergence is governed (behaviour model D7; catalog §7 gap 2); `aggregate_scoring_results.py`, named by action 3 (`:49`), does not exist in the tree.
- `full_scoring.json` `parameter_bounds.chunk_size.default` is 5000 (`:70`) while `default_params.chunk_size` is 1000 (`:122`).
- `full_scoring_worker.py` docstring and argparse say "50 features" (`:5`, `:12`, `:508`); `generate_step3_scoring_jobs.py` says 46 (`:9`, `:284`). Already recorded (system map §7.2).
