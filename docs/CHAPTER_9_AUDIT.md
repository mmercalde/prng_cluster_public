# Chapter 9 audit — GPU cluster infrastructure

Repo: `/home/michael/distributed_prng_analysis` on VM 101 (`zeus-ubuntu-vm`) at `a15b524`
(`a15b5247336255801de8cf67d52653701b1e8c80`), live tree with full git history.
Audited: `docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md` (980 lines at `a15b524`) against the code.
Authority: `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md` REV1 (§0, §1, §3, §5).
Method: read-only. Governance trail first, then chapters, then code. Every anchor below was opened in this session.
Nothing was executed except read-only shell commands and one `python3` import of `utils/canonical_arrays.py` to count arrays. "Works" means the code path is complete on reading, not that it was run.
Step numbers are the EXECUTABLE scheme (Step 1 = `window_optimizer.py`, the sieve runs inside it; Step 2 = scorer meta-optimizer). The chapter itself was written in the conceptual scheme ("Step 2 = sieve, Step 2.5 = scorer meta"); each such place is noted.
Framing held throughout: TFM is functional mimicry of surface output; no finding here assumes a "true seed".

---

## 1. Verdict

- Accurate: **41** claims
- BROKEN: **2**
- SWITCHED-OFF-BY-RULING: **5** (S-5 arises from the new RANGE-MINER section, not from a pre-existing claim)
- DESCRIPTION-ONLY: **35**
- UNVERIFIED: **8**

(91 claims in total. Lists in sections 4, 5, 6 and 9 are numbered so the counts can be checked.)

Three things matter more than the rest:

1. **The chapter did not describe the engine that now runs the cluster's Step-1 work.** RANGE-MINER is absent from the 980 lines; the chapter calls PWC "the production Step 1 sieve execution backend", while PWC/ZMQ were retired from certifying authority on 2026-07-31 and PWC's hybrid passes are quarantined (`persistent_worker_coordinator.py:188-217`). A full RANGE-MINER section (§13) was added.
2. **WATCHER's Step-1 manifest still defaults to PWC** (`agent_manifests/window_optimizer.json:261` `use_persistent_workers: true`, `:269` `use_range_miner: false`) with `test_both_modes: true` (`:254`). On reading, a default WATCHER Step-1 run reaches the hybrid quarantine at `persistent_worker_coordinator.py:1230` and fails closed. This is the consequence of two rulings (S-2, S-4), not a defect; recorded, not changed.
3. **The fleet the chapter describes no longer exists in that shape.** 25 GPUs, not 26 (`distributed_config.json:6`, `f255912`; live `nvidia-smi -L` shows one RTX 3080 Ti); cards are RX 6600 XT (gfx1032), hostnames `rrig6600*`; the boot-selector model is now encoded in `rig_profiles_config.json` + `execution_set.py`, which makes `CLAUDE.md` §3's "rig_profile … explicitly NOT done" stale.

---

## 2. The caller's chapter-specific questions

| # | Question | Result | Evidence |
|---|---|---|---|
| 1 | Does Chapter 9 mention RANGE-MINER? | **No** (0 occurrences of `RANGE-MINER`, `miner/`, `range_miner` at `a15b524`). Section §13 added | `git show a15b524:docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md` |
| 2 | Phase status: Phases 0–6 done, Gate 12 passed? | **Confirmed.** Phase 0 `2389b61`, 1 `8d0183f`, 2 `e0c9d1c`, 3 `dbe3d0e`, 4 `f1e0c7c`, 5 D0 `4c697a8` … D6 `fc37fb5`/`2be51d5`, 6 bounded certification `d98298c`; Gate 12 PASSED attempt 9 (`docs/TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE.md:13-25`), tag `gate12-passed-attempt9` → `e9ca800` (`git rev-parse`). Phase 7 soak not run (`docs/TODO_CONSOLIDATED.md:7`) | `git log --oneline -1 <sha>` each |
| 3 | 22-array NPZ contract / `EXPECTED_NPZ_KEYS` | `CANONICAL_ARRAY_CONTRACT` has **22** arrays (`utils/canonical_arrays.py:98-122`, counted by import). `EXPECTED_NPZ_KEYS` is **not a symbol**: only a forbidden-token string at `tests/test_s172_phase4_coordinator.py:1608` and prose at `CLAUDE.md:168` | `git grep` at `a15b524` |
| 4 | Per-family VRAM caps (TB Q2) | `VramCaps` amd 2M / nvidia 5M / amd_hybrid 1M / nvidia_hybrid 2.5M (`miner/range_miner_worker.py:648-667`); coordinator sizes from the worker's advertised cap (`range_miner_coordinator.py:740-758`). Hybrid caps absent from the WATCHER manifest (`range_miner_worker.py:650-653`) | — |
| 5 | Retry policy (TB Q3) | One reassignment to a **different** worker, then fail — **hybrid phases 3/4 only**; constant phases 1/2 and any non-retryable failure fail the trial at once; no `MAX_ATTEMPTS` (`range_miner_coordinator.py:8417-8530`) | — |
| 6 | Window-anchor change where it touches the miner | `window_anchor` host-side only; `generator_phase` in the kernels' trailing `int32` (`range_miner_worker.py:22-29`, `:295-310`), pinned to 0 at both seams (`:605`, `:608-627`; coordinator `:9314-9325`); NPZ array `offset` now sourced from `window_anchor` (`utils/canonical_records.py:225`, `69ca910`). Brief I implemented, not accepted (production-shape run failed 2026-08-22; Run 4 not run) | `docs/S172_BRIEF_I_PRODUCTION_SHAPE_CLASSIFICATION_REPORT.md` verdict |
| 7 | Boot-selector / bare-metal addresses in `distributed_config.json` | Deliberate, not a bug. Both sets joined in `rig_profiles_config.json` (`default_profile: "proxmox"`) and resolved per run by `execution_set.py:537` | — |
| 8 | Rig GPU SKU | RX 6600 XT, gfx1032 (`docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md:46`, `:119-120`, `:135-136`). Config says `RX 6600` (left as is) | — |
| 9 | `agent_manifests/definitions.json` (gitignored) | Read on host: 6,635 bytes, schema 1.1.0, keys `pipeline_steps` (step5/step6 only), `watcher_protocol`, `sidecar_schema`. **No cluster or node configuration** | `python3 -c json.load(...)`, `/bin/grep` for gpu/node/rig/192.168 |
| 10 | Numbering scheme | Chapter now states EXECUTABLE numbering in its status header; conceptual "Step 2/2.5" corrected at each use | `agents/watcher_agent.py:387-395` |

**Host questions (brief §5, "all chapters" item, plus read-only cluster surfaces):**
- `hostname` → `zeus-ubuntu-vm`; `nvidia-smi -L` → `GPU 0: NVIDIA GeForce RTX 3080 Ti` only.
- `crontab -l` → "no crontab for michael".
- `systemctl list-units --all` (system and `--user`), filtered for prng/miner/watcher/dashboard/pwc/scraper/tfm/draw: no cluster service; one **failed** unit, `nvidia-gpu-policy.service` — `ExecStart=/usr/bin/nvidia-smi -i 1 -c DEFAULT` exits 6 (no GPU 1). `systemctl cat` shows it at `/etc/systemd/system/nvidia-gpu-policy.service`. Untracked; mentioned only in `docs/SESSION_CHANGELOG_20260419_S164.md`, `…20260420_S165.md`. Not touched.
- Rigs: **UNAVAILABLE** (no SSH by brief). Boot target, ROCm settings, GFXOFF, sshd config not observed.

---

## 3. What actually works end to end today (on reading)

1. **Legacy Step-1 chunk dispatch** through `coordinator.py`: profile-proportional allocation capped by seed caps (`coordinator.py:575-640`, `:560-568`), SSH pool, round-based retry with capped backoff (`recovery_manager.py:19-26`; `coordinator.py:1724-1756`), progress persistence under `.recovery/` with `--resume-policy` (`:2059-2111`, `:2719-2730`). Non-certifying.
2. **Step-2 dispatch and pull**: `scripts_coordinator.py` runs `scorer_trial_worker.py` jobs (`run_scorer_meta_optimizer.sh:266-269`), then `coordinator.collect_scorer_results` pulls over SFTP and deletes remote files (`coordinator.py:2564-2674`).
3. **Step-3 dispatch**: ramdisk preload (`run_step3_full_scoring.sh:30-40`), `scripts_coordinator.py` round-robin, per-node concurrency from config, step-aware batching, file-based success, one localhost retry (`scripts_coordinator.py:444`, `:91-113`, `:534-558`, `:1001`, `:1068-1110`).
4. **Worker wrapper** `distributed_worker.py`: `analysis_type` routing (`:249-430`), script jobs without parent GPU init (`:249-251`).
5. **RANGE-MINER Step 1** (certifying): framed protocol (`miner/range_miner_protocol.py`), REGISTER with advertised caps and variants, macro-/sub-stripe partitioning with proved completion, durable ledger, retry matrix, Resolved-Execution-Set admission, Phase-5 assembly into the 22-array bundle. Gate 12 passed on this path. Worker startup is by hand (`scripts/launch_fleet_manual.sh`).
6. **PWC constant-skip passes** (diagnostic only): persistent `sieve_gpu_worker.py` over SSH or TCP. Hybrid passes fail closed (S-2).

---

## 4. BROKEN

**B-1 — ROCm prelude hostname guards never fire on the live fleet.** *(Same class GOVERNED for `sieve_filter.py` as Chapter 2 F-5, §9.5; these two instances are not named there.)*
- Chapter §6.2 presents the guard as the critical first step on rigs.
- Code: `if HOST in ["rig-6600", "rig-6600b", "rig-6600c"]` (`distributed_worker.py:18`; corrected 2026-11-11 post-`741822b`, was `:17`, the comment line above the guard); script path `if hostname in ["rig-6600", "rig-6600b", "rig-6600c", "rig-6600xt"]` (`:354`). Live worker hostnames `rrig6600`, `rrig6600b`, `rrig6600c` (`rig_profiles_config.json` `nodes[*].worker_hostname`).
- Consequence on reading: the branch is dead. Harmless today because ROCm 6.4 needs no override (`docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_6_0_ROCM_PARITY.md:33`), and at `:354` the `else` branch still sets `CUDA_VISIBLE_DEVICES` while `scripts_coordinator.py:967-971` already exports `HIP_VISIBLE_DEVICES`.
- Stops at: `distributed_worker.py:18` (corrected 2026-11-11 post-`741822b`; was `:17`).
- Repair would have to: follow Chapter 2 §9.5 — decide whether the prelude is still supported; if retained, key it from the declared rig profile; never rename hosts or write another hostname tuple.

**B-2 — PWC's local GPU slot pool is hardcoded to two GPUs.** *(not found in the governance trail)*
- Chapter (PWC S146 table): `_localhost_semaphore` `Semaphore(2)`, "Zeus has 2 GPUs".
- Code: `for _gpu_id in range(2):  # Zeus has 2 GPUs` (`persistent_worker_coordinator.py:295-297`), against `distributed_config.json:6` `gpu_count: 1` (commit `f255912`, "the second 3080Ti is on VM100"). Commit `4643a11` records two other stale sites from the 2→1 correction; this one is not mentioned there, nor in `docs/PHASE6_PREREQS.md`, `docs/TEAM_ALPHA_D3_0_B_AND_ITEM1_NOTICE.md`, `docs/TEAM_ALPHA_PHASE7_LAUNCH_NOTICE.md` (searched `localhost_gpu_slots`, `localhost.gpu_count`).
- Consequence on reading: concurrent local PWC dispatch can bind `CUDA_VISIBLE_DEVICES=1` on a one-GPU VM. Non-certifying path; not run.
- Stops at: `persistent_worker_coordinator.py:296`.
- Repair would have to: take the local slot count from the resolved execution set / `distributed_config.json`.

---

## 5. SWITCHED-OFF-BY-RULING

Authority changed on 2026-11-11: the owner decides whether each stands. Facts only below.

**S-1 — Legacy fused reverse sieve dispatch closed.**
- Chapter §6.4, §9.1, §9.2 list `reverse_sieve` as a live job type.
- Code: `LEGACY_FUSED_ENGINE_CLOSED` raised at `distributed_worker.py:288-297`, `coordinator.py:832-840`, `coordinator_sieve_dynamic.py:563`, `run_complete_pipeline.py:76`, and `reverse_sieve_filter.py:123`.
- Ruling: Window-Anchor Brief I §2.4 (`docs/S172_WINDOW_ANCHOR_BRIEF_I.md:312`); "the legacy reverse engine is execution-closed" (`docs/TB_RULING_WINDOW_ANCHOR_BRIEF_I_CODE_REVIEW.md:70`).
- If lifted: `reverse_sieve_filter.py` would dispatch again, with one `offset` driving both window slice and generator advance (coherent only at skip 0, per its own message).

**S-2 — PWC hybrid passes quarantined.**
- Chapter: S146 hybrid passes validated; PWC hybrid invariants as live rules.
- Code: `assert_pwc_hybrid_not_quarantined` (`persistent_worker_coordinator.py:195-217`), called at `run_sieve_pass` (`:1230`); no override flag (`:183-185`).
- Ruling: commit `8a55a68` ("TB Ruling 24 items 1-2, comparator ruling 5"); `docs/S172_THRESHOLD_PROPAGATION_REPAIR_REPORT.md`. Standalone ruling text not located as a `TB_RULING_*` file.
- If lifted: PWC variable-skip would run filtering at `run_sieve_pass`'s 0.5 default rather than the directional threshold, with kernels that ignore `skip_min/skip_max`, unless comparator ruling §5 Option A is built first (message at `:209-216`).

**S-3 — PWC/ZMQ have no certifying authority.**
- Chapter: "PWC is the production Step 1 sieve execution backend".
- Ruling: Team Beta 2026-07-31, recorded in `docs/TFM_PROJECT_FACTS_SKILL.md:186-187` and Chapter 2 §10; request trail `docs/TEAM_ALPHA_PWC_COMPARATOR_SCOPE_CORRECTION.md`. No `TB_RULING_*` file contains "2026-07-31" (searched).
- If lifted: PWC/ZMQ output could serve as production Step-1 output or comparator.

**S-4 — RANGE-MINER is not WATCHER's default backend.**
- Code: `agent_manifests/window_optimizer.json:212-216` (description "Do not enable in production until Phase 7 acceptance completes"), `:269` `use_range_miner: false`, `:261` `use_persistent_workers: true`.
- Ruling: approved phase plan `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:503` (Phase 7 = WATCHER soak + production); open decision `docs/TODO_CONSOLIDATED.md:93`. Brief §5 asks this be recorded, not changed.
- If lifted: default WATCHER Step-1 runs would use the certifying engine instead of the PWC path that fails closed on hybrid passes.

**S-5 — `generator_phase` pinned to 0.** *(new-section item)*
- Code: `GENERATOR_PHASE_V1_PIN = 0` (`miner/range_miner_worker.py:605`), `assert_generator_phase_permitted` (`:608-627`); coordinator refusal (`miner/range_miner_coordinator.py:9314-9325`).
- Ruling: TB 2026-08-21, cited at `range_miner_worker.py:699`, `:1140`; DEP-ABI-V2 "recorded, NOT built".
- If lifted: nonzero device-side generator phase would become reachable through an assignment, on the variants whose ABI carries it.

---

## 6. DESCRIPTION-ONLY

All corrected in place in the chapter with a *Corrected 2026-11-11* note or a SUPERSEDED marker.

1. **D-1** Header: "26-GPU", file list omits the Step-1 engines and `execution_set.py`.
2. **D-2** "Two coordinators" (§1.1, §1.2, §3.1): four Step-1 backends (`window_optimizer_integration_final.py:1402`, `:1576`, `:1621`, fallthrough) plus `scripts_coordinator.py`.
3. **D-3** §2.1/§2.2 topology: 1 × RTX 3080 Ti on VM 101 (`distributed_config.json:6`; `nvidia-smi -L`), RX 6600 XT, `rrig6600*` hostnames, 25 GPUs.
4. **D-4** §3.2 `coordinator.py --seed-end`: no such flag (`coordinator.py:2687-2773`).
5. **D-5** §4.3 "Job Queue (priority)": `DynamicJobQueue` / job list (`coordinator.py:1208`, `:575-640`).
6. **D-6** §4.4 6× scaling, base 10,000: factor 2.80 (`gpu_optimizer.py:18`); allocation is capacity-proportional (`coordinator.py:586-611`).
7. **D-7** §5.3 `PER_NODE_CONCURRENCY` dict: absent; limits from `max_concurrent_script_jobs` (`scripts_coordinator.py:444`; `coordinator.py:373`).
8. **D-8** §5.4 job format: top-level list with `expected_output`, no `gpu_id` (`scripts_coordinator.py:450-457`, `:27-34`).
9. **D-9** §5.5 least-loaded node: round-robin (`:534-558`).
10. **D-10** §5.1 "Steps 3-6": WATCHER uses it for Steps 2 and 3.
11. **D-11** §6.4 `route_job` / `type`: `execute_analysis_job`, key `analysis_type` (`distributed_worker.py:233`, `:249`).
12. **D-12** §6.5 `subprocess.run(['python3', …])`: `Popen([sys.executable,'-u',…], start_new_session=True)` in a thread pool (`:339-411`).
13. **D-13** §6.2 details: `HIP_PATH=/opt/rocm/hip`, CuPy imported lazily (`:28`, `:44`).
14. **D-14** §7.2 profiles 29,000 / 5,000 / 6.0: now 2,210,000 / 787,950 / 2.80 (`gpu_optimizer.py:15-37`).
15. **D-15** §7.4 learning: recorded per worker, not fed into `get_gpu_profile` (`:82-89`, `:105-111`).
16. **D-16** §8.1/§8.2 override needed: not needed on ROCm 6.4 (`…PHASE_6_0_ROCM_PARITY.md:33`).
17. **D-17** §9.2 Step 4/5 rows: Step 4 local (`agent_manifests/ml_meta.json:30`); Step 5 local subprocess trials (`meta_prediction_optimizer_anti_overfit.py:336`).
18. **D-18** §10.1 retry code: round-based, `min(300, 5·2^(n−1))` (`recovery_manager.py:23-26`).
19. **D-19** §10.2 `progress.json`: `.recovery/` per analysis id (`recovery_manager.py:16-18`, `:41`).
20. **D-20** §10.3 `handle_node_failure` / `schedule_reconnect`: no such symbols in tracked `*.py`.
21. **D-21** §11.1 coordinator CLI: no `--seed-end`, `--resume`, `--output-dir`; has `--resume-policy`, `-o`, positional `target_file`.
22. **D-22** §11.2 scripts_coordinator CLI: no `--resume`; has `--max-retries`, `--dry-run`, `--config` (`:1142-1155`).
23. **D-23** §11.3 distributed_worker CLI: positional `job_file` (`:69`).
24. **D-24** §12 line counts: 2846 / 1194 / 607 / 136.
25. **D-25** §8.4 "12" concurrency: config 8 per rig (`distributed_config.json:19`, `:30`, `:41`).
26. **D-26** §8.5 manual ramdisk loop as the route: `run_step3_full_scoring.sh:30-40` preloads; verify script omits `.162`.
27. **D-27** PWC engine: 1957 lines not 855; shim `:1632` not `:669`; gate after the miner gate.
28. **D-28** PWC "Zeus uses local path": also TCP local workers since S165 (`:797-807`); TCP transport (`:265`, `:513-529`).
29. **D-29** `execute_remote_job`: no such symbol in `persistent_worker_coordinator.py`.
30. **D-30** `_localhost_semaphore Semaphore(2)` "mirrors coordinator.py line 269": PWC has `_localhost_gpu_slots` (`:295`); coordinator's is `Semaphore(self.max_local_concurrent)` at `:286`, CLI default 4 (`:2714`).
31. **D-31** `worker_pool_size` 4 "validated": manifest default 8 (`window_optimizer.json:262`).
32. **D-32** Dashboard `45.32.131.224:5002`: not in live code (`apply_s146_doc_updates.py:188`, `:380` only); `web_dashboard.py:2259` binds `0.0.0.0:5002`.
33. **D-33** Manifest "v1.5.0": now v1.8.0 (`window_optimizer.json:5`).
34. **D-34** rig-6600c "hardware limitation": S151 RCA found `MaxSessions 5`; CPU ruled out (`docs/ROOT_CAUSE_ANALYSIS_RRIG6600C_S151.md:11-15`, `:108`).
35. **D-35** PWC S146 section present twice verbatim.

---

## 7. New material written into the chapter, and what it rests on

- **§0** How the chapter was produced. **§13** RANGE-MINER: §13.1 purpose and selection (`window_optimizer_integration_final.py:1402-1440`; `miner/range_miner_coordinator.py:12279`, `:12336`); §13.2 package table (`wc -l`, module docstrings); §13.3 comparison table (`range_miner_protocol.py:41-46`, `:80-261`; `range_miner_coordinator.py:15-25`, `:552-558`, `:603`, `:761-786`, `:1114-1128`; `range_miner_worker.py:680-695`; `scripts/launch_fleet_manual.sh:8-33`); §13.4 coverage (`range_miner_worker.py:105-122`, `:474-485`, `:630-645`); §13.5 VRAM caps; §13.6 retry matrix; §13.7 Brief I; §13.8 22-array contract; §13.9 phase table with commits. **§14** topology and the Resolved Execution Set (`rig_profiles_config.json`; `execution_set.py:1-60`, `:496`, `:537`; consumers `coordinator.py:119-121`, `:310-313`, `persistent_worker_coordinator.py:105`, `:323`, `preflight_check.py:179-181`, `:317`, `window_optimizer.py:1530-1538`, `agents/watcher_agent.py:1623`) and host state. **§15** Known gaps (three lists). **§16** Open items O-1…O-11. **§17** VIR-1…6 and sentinel. **§18** Closure.
- One unlabelled observation is recorded as open item **O-6**: `scripts_coordinator.py` (`:78`, `:430-448`), `ramdisk_preload.sh:30-33`, `run_scorer_meta_optimizer.sh:257-261` and `ml_coordinator_config.json` read bare-metal addresses directly and are not consumers of the resolved set. The resolved-set brief (`docs/CLAUDE_CODE_INSTRUCTIONS_RESOLVED_EXECUTION_SET.md` §2) scoped six Step-1 mechanisms; I found no document saying Steps 2–3 should consume it, so this is not labelled BROKEN.

---

## 8. Surfaces searched, and surfaces unavailable

**Files opened this session**
- Brief and models: `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md`, the scratchpad common brief, `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` (header, §8–§11; not edited), `docs/CHAPTER_13_AUDIT.md`, `docs/TODO_CONSOLIDATED.md` (§1, `:7`, `:92-93`), `docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.1, `~/.claude/skills/lead-handling/SKILL.md` (head).
- Governance: `docs/TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE.md`, `docs/TB_RULING_WINDOW_ANCHOR_BRIEF_I_CODE_REVIEW.md` (`:70`), `docs/TB_RULING_WINDOW_ANCHOR_BRIEF_I_SCOPE_RULING.md` (grep), `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md` (grep; `:46`, `:465`, `:502-508`, `:588`), `docs/TEAM_ALPHA_PWC_COMPARATOR_SCOPE_CORRECTION.md`, `docs/TEAM_ALPHA_EXECUTION_SET_AND_CHAPTER2_SUBMISSION.md` (grep), `docs/CLAUDE_CODE_INSTRUCTIONS_RESOLVED_EXECUTION_SET.md` §1–§6, `docs/CLAUDE_CODE_INSTRUCTIONS_STAGING_DIR_PART_B.md` §7, `docs/S172_WINDOW_ANCHOR_BRIEF_I.md` (grep `:312`), `docs/S172_BRIEF_I_PRODUCTION_SHAPE_CLASSIFICATION_REPORT.md` (verdict, §1), `docs/PROJECT_FILE_CATALOG.md:36-60`, `docs/TFM_PROJECT_FACTS_SKILL.md` (`:180-195`, `:1630-1675`, grep), `docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md` (grep), `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_6_0_ROCM_PARITY.md:25-40`, `docs/CLAUDE_CODE_REPORT_PRERUN_PROBE_AND_SAMPLER.md:50-64`, `docs/ROOT_CAUSE_ANALYSIS_RRIG6600C_S151.md` (`:1-20`, `:100-115`).
- Code: `coordinator.py` (`:248-300`, `:560-640`, `:815-845`, `:2564-2674`, `:2687-2790`, grep), `recovery_manager.py:15-30`, `scripts_coordinator.py` (`:1-120`, `:165-240`, `:450-470`, `:534-560`, `:587-680`, `:826-1115`, grep), `distributed_worker.py` (`:1-90`, `:245-440`, grep), `gpu_optimizer.py` (all), `persistent_worker_coordinator.py` (`:1-80`, `:180-251`, `:288-300`, `:1120-1150`, `:1632-1660`, grep), `sieve_gpu_worker.py` (`:205-310`, `:478-492`, grep), `window_optimizer_integration_final.py` (`:1393-1475`, grep), `miner/range_miner_protocol.py` (grep), `miner/range_miner_worker.py` (`:1-80`, `:100-140`, `:474-500`, `:598-700`, `:2800-2830`, grep), `miner/range_miner_coordinator.py` (`:1-120`, `:590-612`, `:740-800`, `:8405-8530`, grep), `miner/assembly_backends.py` (grep), `utils/canonical_arrays.py` (`:90-125`, grep; imported), `utils/canonical_records.py` (grep), `execution_set.py` (`:1-60`, grep), `rig_profiles_config.json` (all), `distributed_config.json` (all), `ml_coordinator_config.json` (parsed), `agent_manifests/window_optimizer.json` (`:165-280`), `agent_manifests/{full_scoring,scorer_meta,ml_meta,prediction,reinforcement}.json` (grep/parsed), `agent_manifests/definitions.json` (gitignored; parsed), `agents/watcher_agent.py:386-416`, `run_scorer_meta_optimizer.sh:255-300`, `run_step3_full_scoring.sh` (`:30-45`, grep), `ramdisk_preload.sh` (grep), `generate_step3_scoring_jobs.py` (grep), `survivor_scorer.py` (grep), `web_dashboard.py` (grep), `scripts/launch_fleet_manual.sh:1-60`, `meta_prediction_optimizer_anti_overfit.py` (grep).
- Memory notes read as pointers only, not as anchors: `rig-gpu-sku-is-rx-6600-xt.md`, `rig-throughput-heterogeneity.md`.

**Absence searches** (for D-7, D-20, D-29, D-30, D-32, B-2, S-3): `git grep` over tracked `*.py`; `/bin/grep -rn` over `docs/` and over the working tree for `.json`; `git log -S PWC_HYBRID_QUARANTINE_CODE`; `git tag -l`.

**Gitignored surfaces read**: `agent_manifests/definitions.json`. `git check-ignore -v` confirms `.gitignore:41`. Note `distributed_config.json` and `rig_profiles_config.json` are tracked (`!*_config.json` un-ignore per `rig_profiles_config.json` `_naming`).

**Host surfaces read (VM 101, read-only)**: `hostname`, `nvidia-smi -L`, `systemctl list-units --all` (system + user), `systemctl status` and `systemctl cat nvidia-gpu-policy.service`, `crontab -l`.

**Not available, and where it matters**
- **Rigs (CT100 / bare metal)**: boot target, ROCm env, GFXOFF param, sshd settings, throughput. Decides U2, U8 and current reachability. UNAVAILABLE by brief (no SSH).
- **Private repo, ser8 archives**: not searched.
- **Gate-12 / production-shape artifacts under `logs/`**: not opened; status taken from the ruling and report documents.
- **Not read in full**: `miner/range_miner_coordinator.py` (12,555 lines; read by symbol), `miner/range_miner_npz_writer.py`, `miner/assembly_shard_worker.py`, `miner/dataset_authority.py`, `zmq_sqlite_coordinator.py`, `persistent/pwc_transport_*.py`, `docs/FLEET_STATE_REQUIREMENTS_v1.md`, `docs/S172_INFRASTRUCTURE_INTERFACE_v1_0.md`, the session changelogs, the `apply_s*.py` corpus.

**Chapters searched**: Chapter 2 (§8–§11, for RANGE-MINER, PWC retirement, ROCm prelude F-5); Chapter 13 audit (layout only).

---

## 9. Accurate claims and unverified claims (for the count)

**Accurate (41)**
1. SSH connection pool (§4.2) — `coordinator.py:259`; `connection_manager.py:12`.
2. Pull architecture, remote file deleted after pull (§4.2, §4.5) — `coordinator.py:2620-2674`.
3. Automatic retry with exponential backoff exists (§4.2) — `recovery_manager.py:19-26`.
4. Progress persistence and resume exist (§4.2) — `coordinator.py:2059-2111`.
5. Distribution by GPU capability through `gpu_optimizer` (§4.4 concept) — `coordinator.py:586-609`.
6. Script coordinator job type = script + args (§5.2) — `scripts_coordinator.py:230-239`.
7. Execution as Python subprocess (§5.2) — `:964-999`.
8. File-based success detection (§5.2) — `:1001`.
9. SSH to node with GPU binding env (§5.5 b–c) — `:967-973`, `:892-896`.
10. Script jobs skip GPU init, v1.8.0 (§6.5) — `distributed_worker.py:7-10`, `:249-251`.
11. Worker sees the bound GPU as device 0 (§6.3) — `:74-83`.
12. Environment set before GPU library imports (§6.2, ordering) — `:13-29`, lazy CuPy `:44`.
13. `sieve` → `sieve_filter.py` subprocess (§9.1) — `:257-277`.
14. `script` → direct Python subprocess (§9.1) — `:319-411`.
15. `correlation` → `analyze_correlation_gpu` (§9.1) — `:414-424`.
16. default → `run_statistical_analysis` (§9.1) — `:425-430`.
17. Chunk size min 500 / max 50× base (§7.3) — `gpu_optimizer.py:62-63`, `:90-97`.
18. EMA α = 0.3 (§7.4) — `:64`, `:120-123`.
19. `PYTORCH_HIP_ALLOC_CONF` and 0.8 memory fraction (§8.3) — `survivor_scorer.py:28`, `:65`.
20. Scorer meta via `scripts_coordinator.py --preserve-paths` (§3.2, §9.2) — `run_scorer_meta_optimizer.sh:266-269`.
21. Full scoring via `scripts_coordinator.py` (§9.2) — `run_step3_full_scoring.sh:226`.
22. Step 6 needs no coordinator (§9.2) — `agent_manifests/prediction.json` (`distributed: false`).
23. Ramdisk path `/dev/shm/prng/step3/` (§8.5) — `generate_step3_scoring_jobs.py:236-237`.
24. Ramdisk lost on reboot (§8.5) — `/dev/shm` is tmpfs; consistent with `distributed_config.json:20-21`.
25. `chunk_size=1000` in two places (§5.8) — `run_step3_full_scoring.sh:70`; `agent_manifests/full_scoring.json:122`.
26. Corrected throughput values (GPU Throughput) — `gpu_optimizer.py:17`, `:35`.
27. `--seed-cap-nvidia/amd` CLI 5M/2M (Seed Caps) — `coordinator.py:2703-2706`.
28. Constructor defaults 5M/2M — `coordinator.py:250`.
29. Integration passes caps explicitly — `window_optimizer_integration_final.py:1456-1457`, `:1594-1595`, `:1633-1634`, `:1652-1653`.
30. PWC IPC message table — `sieve_gpu_worker.py:13-17`, `:457`, `:473-490`.
31. Spawn stagger 4.0 s — `persistent_worker_coordinator.py:116`.
32. `ssh -q` and ready-drain loop — `:617`, `:635-648`.
33. Per-`WorkerHandle` dispatch lock — `:242`.
34. PWC exports `HSA_OVERRIDE_GFX_VERSION=10.3.0`, `HSA_ENABLE_SDMA=0` — `:140-141`, `:385-386`.
35. 24 AMD workers, 8 per rig — `:16`, `:556`.
36. `JOB_TIMEOUT_S = 600` — `:154`.
37. Hybrid kernel tail split forward/reverse — `sieve_gpu_worker.py:231-280`.
38. Hybrid threshold uses `phase2_threshold` via `coerce_threshold`; post-filter on `hybrid_threshold` — `:254-255`, `:285`.
39. Full `StrategyConfig.to_dict()` strategies — `:240-245`.
40. ProgressWriter `log_gpu_result` / `update_trial_stats` — `persistent_worker_coordinator.py:1397`, `:1697`, `:1860`.
41. Call chain shim → PWC, local path `sieve_filter.py` subprocess — `window_optimizer_integration_final.py:1576-1582`; `persistent_worker_coordinator.py:1124-1144`, `:1632`.

**UNVERIFIED (8)**
- **U1** TFLOPS (~285) and VRAM (216 GB) totals (§2.1). Would verify: a committed measurement; nominal spec gives 204 GB for the current fleet.
- **U2** §8.4 benchmark table and `amdgpu.ppfeaturemask=0xffff7fff` on the rigs. Would verify: rig kernel cmdline and the benchmark logs (rig access).
- **U3** "Manifest overrides script default" for WATCHER (§5.8). Would verify: the WATCHER argument builder for Step 3.
- **U4** PWC throughput 849,469 / 2,082,140 sps (S128/S130). Would verify: the S130 logs.
- **U5** S131 "Gate 1" three fallback paths and the 20 ms dead-pipe figure. Would verify: S131 changelog and test.
- **U6** S146 validation result (3 trials, zero errors). Would verify: S146 run logs.
- **U7** Dashboard at `45.32.131.224:5002`. Would verify: the host that serves it.
- **U8** rig-6600c "~50% throughput deficit". Would verify: a same-work per-rig measurement.
