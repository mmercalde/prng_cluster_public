# CLAUDE CODE REPORT — SURVIVOR-VOLUME LIMIT (plan item A2, finished condition F3)

## For the owner, in plain words

**Owner decision, 2026-11-14: the cap is 25,000 survivors in one phase.** The cap file now
carries `25000`, with `compare: point` and the `sunshine` allowlist unchanged.

**Under the new cap, the gate PROCEEDS on Run 4's declared range `[2³¹, 2³²)` under `point`
and REFUSES it under `upper95`.** Both results are from the GPU, re-run on 2026-11-14
(`data/survivor_volume_limit/live_case_cap25000.log`). The cap file says `point`, so as landed
the gate proceeds.

| phase | sample projection (95% interval) | exact count, every seed | `point` @ 25,000 | `upper95` @ 25,000 | (old cap 23,515: both rules) |
|---|---:|---:|---|---|---|
| 1 `java_lcg` | 0 (0 – 118) | 0 | pass | pass | pass |
| 2 `java_lcg_reverse` | 0 (0 – 118) | 0 | pass | pass | pass |
| 3 `java_lcg_hybrid` | 64 (7.8 – 231) | 60 | pass | pass | pass |
| 4 `java_lcg_hybrid_reverse` | **24,208 (23,353 – 25,086)** | **23,568** | **PROCEED** | **REFUSE** (25,086.2 > 25,000) | REFUSE |

- The sample projection is COMPUTED: 3,026 hits in 2²⁸ seeds × 8. The exact count is MEASURED:
  all 2³¹ seeds, through the same production path, in 75 s.
- **The exact count, 23,568, is 1,432 below the new cap.** The `upper95` refusal comes from the
  sample's interval, not the count. A census of phase 4 (§6, follow-on 1) would remove that
  difference.
- **23,568 is still 53 above the proven point of 23,515.** No completed run has carried it.
- **The lower half `[0, 2³¹)`** passes under both rules at 25,000. The exact count is 23,410, and
  the sample gave 23,136 with an upper bound of 23,995.

**Beta on the cap, as Beta stated it.** Beta has **not** reviewed the 25,000 decision; no relay
was run, by instruction. Its stated position, from X-VOLUME-LIMIT R2 Q7
(`docs/exchange/from_beta/X-VOLUME-LIMIT_R2_BETA_REVIEW.md:116-118`):

> Retain **23,515 as the owner's conservative starting policy**. The record demonstrates one
> completed operating point through publication, not a failure threshold or universal capacity
> guarantee … The proposal reserves increases to Michael on completed-run evidence. The local
> census establishes a recorded load of 23,568 in this geometry. It supplies no completed
> fleet-run evidence for raising the cap. A different number would need owner-authorized
> completed runs at the proposed operating point, resource observations and the required
> evidence review. Being 53 over the cap establishes numerical refusal, not a physical capacity
> cliff.

Beta R1 Q7 also said: "Any decision to permit Run 4 beyond the current cap belongs to Michael,
with the distinction between local load evidence and completed-run evidence explicit"
(`docs/exchange/from_beta/X-VOLUME-LIMIT_R1_BETA_REVIEW.md:141`).

**For the record, plainly:**
- `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:164-165` says the cap "is raised only by the
  owner, on evidence from completed runs".
- The owner raised it. The evidence behind 25,000 is the local census, not a completed run.
- The cap file's evidence text states both the proven point (23,515) and Run 4's exact count
  (23,568).
- **⚠ Which half Run 4 actually sweeps depends on a ruling you have pending.** `serve_trial`
  calls `assign_stripes` without `base_start`, which defaults to 0
  (`miner/range_miner_coordinator.py:10324-10326`, `:4000`; verified this session). A launch at
  `seed_start = 2³¹` would therefore sweep the lower half, at 23,410. The gate estimates the
  range it is given and cannot see this. The finding is another session's
  (`CLAUDE_CODE_REPORT_STEP1_SEARCH_RECORD_STAGE1.md` §6). I did not read that report; I only
  re-checked the two lines.
- **A survivor count is load, not quality.** None of this says anything about hit rate.

---

## Prior record

**Searches run in this session** (worktree at `7673924`):

- `grep -lIiE "volume[- ]limit|volume cap|survivor[- ]volume|pre-run sample|volume-aware|projected survivors"`
  over `docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_* docs/TB_RULING_REQUEST_* docs/PROPOSAL_*
  docs/CHAPTER_* docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md
  docs/exchange/*.md docs/exchange/*/*.md`. Ten hits: the brief's set plus
  `docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md:129`.
- `grep -rnIiE "survivor_volume|volume_gate|volume_cap|projected_survivors" --include=*.py --include=*.sh --include=*.json .`
  outside `docs/` and this work's own files: no hit.
- `grep -rnIl skip_range` over the governance set and over `miner/*.py`; then
  `git log -S'"skip_range"' -- miner/range_miner_coordinator.py`, which found no commit.
- Read: the brief in full; `docs/S172_PHASE3_SURVIVOR_CAPACITY_CHARACTERIZATION.md` in full;
  `docs/S172_RUN4_PROPOSAL_PINNED_GEOMETRY.md` §1–§2; `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md`
  §7.3; `docs/SESSION_CHANGELOG_20260828_RUN4_ROUTING_PATCH.md:190-240`;
  `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:150-185`; `docs/LEADS.md` head and headings;
  `docs/HYBRID_SKIP_BOUND_AUDIT.md` §2.4; `docs/exchange/RELAY_PROTOCOL.md`; `AGENTS.md` §2–§5;
  `scripts/run_authorization_gate.py`, `scripts/env_freeze_gate.py:103-138`,
  `tests/test_run_authorization_gate.py`.
- Ledger: `~/miner_ledger_archive/miner_ledger_pre_separation_9runs.db` (`mode=ro&immutable=1`),
  `trial_context` and `stripes` for `distributed_config_t1_554463d3`.

**Documents that already decide this subject:** the brief's table, plus:

| Document | What it settles | Label |
|---|---|---|
| `docs/TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE.md:15`, `:44`, `:145` | The primary acceptance of attempt 9, whose phase 4 is the cap's evidence; the Field-6 limitation | ALREADY RECORDED. **Added at Beta R1 finding 1 / Q9**; it was missing from R1's list. Gate 12 is not reopened |
| `docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md:129` | A2 is next, Beta before commit | ALREADY RECORDED |
| `docs/HYBRID_SKIP_BOUND_AUDIT.md` §2.4 M3→M4, M6a; `docs/TEAM_ALPHA_BOUNDED_PHASE_6_SUBMISSION.md:382` | Say sampled skip bounds reach the constant kernels via the payload's `skip_range` | Contradicted at source: finding F-1 |

**Items this report raises:**

| Item | Label |
|---|---|
| Run 4's exact phase-4 count on `[2³¹, 2³²)` is 23,568, above the 23,515 proven point | NEW (measurement) |
| F-1: the assign payload carries no `skip_range` | NEW to this record. Beta R1 Q1 treats it as an already-disclosed status. Not fixed; outside §2 |
| F-2: the GPU-busy rule excludes the gate's own PID | NEW design detail, inside §2 |
| F-4: Run 4 at a non-zero start sweeps `[0, seed_count)` | ALREADY RECORDED by another session; owner ruling pending. Cited here only for its effect on which number applies |
| `gpu_process_allowlist: ["sunshine"]` in the cap file | NEW. **Owner ruling 2026-11-14, this session** |
| The cap's number: **25,000, owner decision 2026-11-14** (was 23,515) | Owner decision, under the owner's reserved authority. `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:164-165` asks for completed-run evidence for a raise; this raise rests on the local census. Beta has not reviewed it |
| The comparison rule (`point`) | ALREADY RECORDED as the owner's. Not changed |

**Not searched:** ser8 archives, the private repository, the project PDFs, the
`/home/michael/tfm_worktrees/step1_search` worktree (excluded by the brief), the rigs.

---

## 1. First-step output

```
pgrep -af beta_relay.py   -> one line: this shell's own command (PID 13221). No relay running.
git rev-parse HEAD        -> 76739247ef46a0f921ed89d684a209800923b2d7
git status --porcelain    -> (empty; main tree clean)
git worktree add --detach /home/michael/tfm_worktrees/volume_limit HEAD  -> HEAD is now at 7673924
venv                      -> /home/michael/venvs/torch/bin/python3, Python 3.10.12
nvidia-smi                -> NVIDIA GeForce RTX 3080 Ti, 12288 MiB, driver 550.163.01, CUDA 12.4
compute processes         -> 2419, sunshine, 248 MiB   (and Xorg, type G)
```

Read literally, the brief's rule ("no compute process") made M-B `UNAVAILABLE`. `sunshine` is a
long-standing resident (`docs/CHAPTER_13_AUDIT.md:385`; `docs/D6_RELEASE_GRADE_SMOKE_20260729.log:33`).
**I asked the owner, who ruled: allow `sunshine` only, and any other compute process is busy.**

## 2. M-B — measurements

**Common to every run:**
- **Dataset:** `/home/michael/distributed_prng_analysis/daily3-20260801T145551443433Z-513648160d35.json`,
  sha256 `513648160d356617c22a1e543ae1c9c65f4921ec21718989308b1f70c00768f6`. This is the file
  `daily3_current.json` names, and `daily3.json` hashes identical.
- **Residue window:** `sha256_residues = c761272da958ba34699d612727f7a6a2aecf08eccd570dc521257964a145c008`,
  equal to attempt 9's `trial_context.residue_sha256`.
- **GPU:** `cuda:0` RTX 3080 Ti.
- **Commit:** `7673924` plus this work's uncommitted files.
- **Kernels:** `java_lcg_hybrid_multi_strategy_sieve` (source sha256 `b237b875…`) and
  `java_lcg_hybrid_reverse_sieve` (`6936e6c4…`).
- **Seed caps:** the production CUDA caps, 5,000,000 constant and 2,500,000 hybrid.

Host-filtered and kernel-admitted counts were identical in every cell. **MEASURED** means counted
on the GPU; **COMPUTED** means projected. Every figure was re-confirmed on the final code after
the R2 changes (`mb_control_final.log`, `live_case_final.log`).

### 2.1 Control — attempt 9's geometry on `[0, 2³¹)` (`data/survivor_volume_limit/mb_control.json`)

| phase | interval | recorded | **measured** | source of the recorded figure | s |
|---|---|---:|---:|---|---:|
| 3 | `[0, 2²⁶)` | 3 | **3** | hand sample 2026-08-27 (capacity record §3); = ledger `st2_s0` | 7.1 |
| 3 | `[2²⁶, 2·2²⁶)` | 3 | **3** | attempt 9 ledger `st2_s1` | 5.0 |
| 3 | `[2·2²⁶, 3·2²⁶)` | 4 | **4** | attempt 9 ledger `st2_s2` | 5.1 |
| 3 | `[3·2²⁶, 4·2²⁶)` | 3 | **3** | attempt 9 ledger `st2_s3` | 5.0 |
| 4 | `[0, 2²⁴)` | 177 | **177** | hand sample 2026-08-27 (capacity record §3.2; RUN4 §1.1) | 0.6 |

**The control reproduces 5/5, exactly.** It was re-run three times; the last run used the final
code.
- **Full-scale control:** `java_lcg_hybrid` takes no phase argument, so its census over
  `[0, 2³¹)` is generator-phase-independent. It is **59, equal to attempt 9's published phase-3
  total**.
- The hand sample's intervals were not recorded. `[0, 2²⁶)` and `[0, 2²⁴)` are the intervals that
  reproduce 3 and 177. [UNVERIFIED that the hand session used exactly these.]
- **Phase 4 cannot be checked against attempt 9's ledger, and that is expected.**
  `java_lcg_hybrid_reverse` takes a phase argument. Attempt 9 ran a fused offset of 25, and the
  current code pins `generator_phase = 0` (`miner/range_miner_coordinator.py:9322`). The ruling
  says the two populations are not regression-equivalent
  (`docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:154-179`).

### 2.2 Run 4, lower half `[0, 2³¹)` (`mb_run4_lower.json`, `mb_census_lower.json`)

The configuration is the built command's: window 12, anchor 25, `generator_phase 0`,
`["midday"]`, 0.71/0.47, skip 6–99, `test_both_modes`. **Its residue window and dataset equal the
control's** (`c761272d…`), and its counts are the control's population, since both run at
phase 0.

| phase | sample seeds (8 blocks) | hits | **projection** (COMPUTED) | 95% interval | exact count (MEASURED) | s |
|---|---:|---:|---:|---|---:|---:|
| 1 | 2²⁶ | 0 | 0 | 0 – 118.0 | — | 2.9 |
| 2 | 2²⁶ | 0 | 0 | 0 – 118.0 | — | 1.1 |
| 3 | 2²⁶ | 4 | 128 | 34.9 – 327.7 | **59** | 5.4 |
| 4 | 2²⁸ | 2,892 | 23,136 | 22,300.4 – 23,994.9 | **23,410** | 9.6 |

### 2.3 Run 4, upper half `[2³¹, 2³²)` — the declared range (`mb_run4_upper.json`, `mb_census_upper.json`)

| phase | sample seeds | hits | **projection** (COMPUTED) | 95% interval | **exact count** (MEASURED) | census s |
|---|---:|---:|---:|---|---:|---:|
| 1 | 2²⁶ | 0 | 0 | 0 – 118.0 | **0** | 31.4 |
| 2 | 2²⁶ | 0 | 0 | 0 – 118.0 | **0** | 28.0 |
| 3 | 2²⁶ | 2 | 64 | 7.8 – 231.2 | **60** | 170.9 |
| 4 | 2²⁸ | 3,026 | **24,208** | 23,353.1 – 25,086.2 | **23,568** | 75.2 |

- Phase-4 per-stripe counts run from 676 to 782 (lower half: 676 to 787).
- Phase-4 per-block sample hits (final run, `live_record_point.json`):
  `[398, 390, 381, 388, 363, 399, 352, 355]`. Mean 378.25, SD about 18, against the Poisson
  √378 ≈ 19.4. This is MEASURED on one sample of 8 blocks: a single observation, not a coverage
  calibration.
- **Why the census was run.** Per the lead-handling rule, the above-cap projection got a
  measurement, not a background argument. The sample is 2.7% above the exact count, inside its
  interval. The exact count is above the original cap of 23,515 on its own, and below the owner's
  25,000.

## 3. What the gate does

**The estimator** is `utils/survivor_volume_estimator.py`. Its entry point is `estimate_volume(geometry,
seed_start, seed_end_exclusive, *, sample_seeds, sampler, timeout_s)`. It takes one exact geometry
and a range, and returns hits, per-block hits, kernel hits, projection and interval per phase. It
has no CLI, reads no cap and returns no verdict (tested in V-M).

- **Geometry validation** (`validate_geometry`): eleven exact keys. A dict is a range and is
  refused, as are unknown keys and wrong types.
- **The production path, re-implemented nowhere** (`_phase_payloads`): the phase table
  `workflow_stages_for` (`miner/range_miner_coordinator.py:12003`); `load_residue_window`
  (`miner/range_miner_worker.py:867`), the coordinator side's own function
  (`window_optimizer_integration_final.py:273-296`); `resolve_dataset_sha256`; and
  `RangeMinerCoordinator.build_stripe_assign_payload` (`miner/range_miner_coordinator.py:9216`)
  on a coordinator built with `MinerLedger(":memory:")`, so no file is written.
- **The sampler** (`production_sample`) does what `handle_stripe` does
  (`miner/range_miner_worker.py:2239-2241`):
  - `select_seed_cap` + `partition_stripe`, then `SieveExecutor.execute` (`:1125`) with the
    production `ResidueResolver`, which re-checks `dataset_sha256` and `residue_sha256`;
  - the count is `SubStripeOutcome.count`, after the host filter (`:1281`, `:1290`);
  - the kernel-admitted count is read from the kernel's own counter, through a subclass of the
    mockable launch seam `_gpu_launch` (`:1122`) that calls the original and only observes.
- **Lifetime (revised in R2, Beta Q6).** `ProductionSampler` is an `IsolatedProcessSampler`.
  - Each phase runs in a **separate `python -c` child**. The pickled call goes over one private
    pipe and the result comes back over another.
  - On timeout the parent sends SIGTERM, waits 10 s, sends SIGKILL, waits 10 s, and only then
    raises.
  - If the child still cannot be reaped, the error says so and names the PID as **UNRESOLVED**.
    It is never reported as cleaned up.
  - The child is a fresh interpreter: no fork, and the parent never initialises CUDA.
  - The child never imports the caller's main module, so a caller without a
    `__name__ == "__main__"` guard cannot be re-executed.
  - A sampler without `run_isolated` (an injected one) runs in a **daemon** thread.
- **The sample rule** (`sample_intervals`), revised in R2 (Beta finding 2):
  - Eight near-equal segments, one block per segment.
  - Each block gets one seed, then `⌊(S−n)(seg−1)/(R−n)⌋` more, and the remainder goes to
    segments with room. Every request with `S ≤ R` fits, including a census of any size:
    0 violations over 33,676 cases.
  - Each block's offset is `sha256(request digest | size | index)` mod slack. The same request
    gives the same sample.
  - The M-B samples reproduce unchanged under the new rule (checked).
- **Projection** (`project`): hits × range ÷ sample. The interval is **Garwood 95% for a Poisson
  mean, conditional on the Poisson count model** (R2 wording, Beta Q3). Block dependence and
  coverage calibration are not measured, and per-block hits are recorded for that purpose. Zero
  hits gives 0 with an upper bound of 3.689 × scale.
- **Result checks** (`_check_result`, `estimate_volume`):
  - the result's shape is exact;
  - host hits ≤ kernel hits;
  - `block_hits` has one entry per block and sums to `host_hits`;
  - every block lies inside the range, and the blocks equal those requested;
  - the residue digest equals the payload builder's.

**The gate** is `scripts/survivor_volume_gate.py`, in `evaluate`.
- **The cap file:** `load_cap` → `env_freeze_gate.tracked_unmodified`, the same function the
  run-authorization loader uses (Beta Q5: no objection). `validate_cap` requires exactly five
  fields.
- **The GPU:** `nvidia_probe` + `gpu_verdict`. It is busy if a compute process is neither
  allowlisted nor the gate's own PID. A busy or absent GPU gives `UNAVAILABLE`.
- **Failure mapping:**
  - sampler failure, timeout or malformed result → `ERROR`;
  - geometry, sample or residue failure → `REFUSE`;
  - any other exception → `ERROR`.
  - Every non-PROCEED verdict exits 1, and PROCEED is reachable only from `decide` (`fig > cap`).
- **The record** holds the parameters, dataset sha256, residue digest, intervals, per-block hits,
  hits, projection and interval per phase, the cap file's sha256 and contents, the verdict and
  reasons, kernel identity, the GPU probe, the excluded self PID, the commit, the UTC time, and
  `verdict_kind: "LOAD …"`.

**Cost per call** (MEASURED, 3080 Ti, final code):
- About 24 s for the four phases at 2²⁶/2²⁶/2²⁶/2²⁸ seeds: phase 1 2.9–3.5 s, phase 2 2.8 s,
  phase 3 7.2 s, phase 4 11.1 s.
- That includes about 2 s per phase for each child's interpreter start, CUDA initialisation and
  kernel compile. The in-process R1 figure was 19.8 s.
- CuPy pool peak: 105 MB (constant phases) and 180 MB (hybrid phases), excluding the CUDA context
  (about 2 GB in nvidia-smi). Host RSS is about 560 MB.
- A full census of all four phases over 2³¹ seeds takes about 5 min.

**Fleet GPU:** `ProductionSampler(backend="rocm")` selects the AMD caps, as `handle_stripe` does,
but it was **not run on a rig [UNVERIFIED]**. `nvidia_probe` is CUDA-only, so on a rig the gate
reports `ABSENT` and refuses. A ROCm probe is not built.

**For a later per-trial caller:** the function takes one geometry and holds no state. The child
per phase costs about 2 s more. Production timeouts attempt termination and confirm child exit
when possible; a surviving child is reported UNRESOLVED with its PID. Injected daemon samplers are
abandoned, not cancelled. (Wording per Beta R2 Q8. Beta R1 Q8: "can serve a per-trial caller
without rewriting estimation".) Nothing ranks, scores or compares survivor sets.

## 4. Tests — `tests/test_survivor_volume_gate.py`

The suite runs on CPU only, with an injected sampler and probe, a synthetic dataset, and a
throwaway git repo per cap-file case. The production loader, builder and phase table run on the
synthetic file. Log: `data/survivor_volume_limit/test_run8.log`.

| Case | Result | What it checks |
|---|---|---|
| V-A | PASS | all under the cap: PROCEED, exit 0, record with cap sha, commit and `LOAD` label |
| V-B | PASS | range = sample = 2²⁶ (scale 1): 23,515 PROCEED; 23,516 REFUSE naming phase 3 |
| V-C | PASS | phase-4 projection 23,200 with upper bound about 24,060: `upper95` REFUSE, `point` PROCEED |
| V-D | PASS | zero hits everywhere: PROCEED; every upper bound > 0; phase 1 = 3.68888 × 32 |
| V-E | PASS | absent · unparseable · field missing · untracked · modified: each REFUSE |
| V-F | PASS | `window_anchor` missing · `forward_threshold` as `{min,max}`: each REFUSE |
| V-G | PASS | sample below the minimum · a sampled block outside the range: each REFUSE |
| V-H | PASS | residue digest ≠ the payload's: REFUSE |
| V-I | PASS | no GPU · non-allowlisted process: UNAVAILABLE exit 1; `sunshine` only, and own PID: PROCEED |
| V-J | PASS | sampler raises · times out · malformed: ERROR exit 1 each |
| **V-J2** | PASS | **R2, Beta Q6.** The gate runs in a *separate process* against a one-hour sampler with a 1 s timeout, on both paths. **Process path:** exits rc 1, child timed out and confirmed dead after `main`. **Daemon-thread path:** exits rc 1. The driver has no main guard, and its top level ran exactly once |
| V-K | PASS | same request, same intervals; changed geometry, different intervals |
| V-L | PASS | 96 at `[0,2³¹)` and `[2³¹,2³²)`; 5 × 3·10⁹/2²⁶ on `[1000, 1000+3·10⁹)`; blocks inside the range |
| V-M | PASS | function gives the same figures as the gate; no `.cap` opened; no verdict |
| **V-N** | PASS | **R2, Beta finding 2.** Census of `[1000, 1000+2²⁶+1)`: PROCEED, blocks tile the range exactly, per-block hits recorded |
| V-MUT-1 | **V-I red** | UNAVAILABLE mapped to PROCEED |
| V-MUT-2 | **V-B red** | `>` → `>=` at the cap |
| V-MUT-3 | **V-L red** | scale by `2³¹` instead of the range |
| V-MUT-4 | **V-E red** | ownership check replaced by a plain file read |
| **V-MUT-5** | **V-J2 red** | **the R1 defect restored** (non-daemon thread). `main` returned 1 but the process **did not exit within 60 s** (`test_run7.log:85`) |
| **V-MUT-6** | **V-J2 red** | terminate and kill removed. Child still alive after `main`, reported `could NOT be stopped … UNRESOLVED` (`test_run7.log:93-96`). The test then reaps it |

**21/21, `COMPLETION SENTINEL: PASS`.** Two notes on the history:
- **Fixture defects found while building this, each fixed in the test only:**
  - V-C's first fixture put the upper bound under the cap (`test_run1.log`).
  - V-J2's first driver did not pass `main`'s return code to its exit status
    (`test_run4.log`). V-MUT-5 and V-MUT-6 were green-for-the-wrong-reason in that run, and were
    re-shown red for the right reason afterwards.
  - The first V-MUT-6 anchor left the SIGKILL fallback in place.
  - The driver's output originally went to a pipe that an orphaned child kept open.
- **Two orphaned test children** from that last defect (PIDs 38575 and 38644, both the session's
  own V-MUT-6 children) were killed by PID. None remain.

## 5. Live case — the gate on Run 4's upper-half request, real sampler, real probe (final code)

Parameters: `data/survivor_volume_limit/run4_upper_params.json`.

- **From the worktree CLI:** `REFUSE — cap file UNTRACKED`, exit 1 (`live_worktree_final.log`).
  This is correct, because the file is not the owner's until he lands it.
- **In throwaway repos with the cap file committed** (`live_case.py`, `live_case_final.log`):

| cap | rule | verdict | reason |
|---|---|---|---|
| 23,515 (R2 final code) | `point` | **REFUSE**, exit 1 | phase 4 point figure 24,208.0 > cap 23,515 |
| 23,515 (R2 final code) | `upper95` | **REFUSE**, exit 1 | phase 4 upper95 figure 25,086.2 > cap 23,515 |
| **25,000 (owner, 2026-11-14)** | `point` | **PROCEED**, exit 0 | every phase ≤ 25,000; phase 4 projection 24,208.0 |
| **25,000 (owner, 2026-11-14)** | `upper95` | **REFUSE**, exit 1 | phase 4 upper95 figure 25,086.2 > cap 25,000 |

**The 25,000 rows** were re-run on the GPU after the cap change with no code change
(`live_case_cap25000.log`; records `live_record_point.json`, `live_record_upper95.json`).
- The sample is identical to before: the same 3,026 phase-4 hits and the same intervals.
- The worktree CLI still REFUSEs on the untracked cap file (`live_record_worktree_cap25000.json`).
- **Note:** `live_case.py` re-serialises the cap into each throwaway repo with that run's `compare`
  value. The `cap_sha256` in those two records is therefore the throwaway file's hash, not the
  landed file's.

Three earlier runs showed two of the defects fixed in R2.
- **F-2:** an in-process second call saw its own CUDA context as busy.
- **The multiprocessing re-import hazard:** `live_case.py` has no main guard, so each spawned
  child re-ran the gate. Multiprocessing refused the nested starts with "bootstrapping phase"
  errors, which failed closed. That led to the `python -c` child (`live_case_R2.log`).

## 6. Findings outside the brief's questions

- **F-1 — the assign payload carries no `skip_range`.**
  - `build_stripe_assign_payload` (`miner/range_miner_coordinator.py:9331-9358`) has no
    `skip_range` key, and `git log -S` finds no commit that ever added one.
  - The worker therefore takes `payload.get("skip_range", [0, 16])`
    (`miner/range_miner_worker.py:1164`): **phases 1–2 always scan skip 0–16, whatever was
    sampled.**
  - This contradicts `docs/HYBRID_SKIP_BOUND_AUDIT.md` §2.4 M6a and
    `docs/TEAM_ALPHA_BOUNDED_PHASE_6_SUBMISSION.md:382`.
  - The gate's sample is faithful to production (`skip_bounds_note`). Not fixed: outside §2.
  - **What would settle the hit-rate question** (Beta Q1): a flag-selectable delivered-skip
    configuration, compared with the baseline on unseen-draw pool hit rate.
- **F-2 — self-PID.** See §3.
- **F-3 — the cost of exactness.** A phase-4 census over 2³¹ seeds takes 75 s. Counting rather
  than sampling the hybrid phases would remove Q3's noise question for runs of this size. That is
  a design change, not built.
- **F-4 — the range the gate checks may not be the range the run sweeps.** See the owner summary.
  The gate cannot detect it. Once `base_start` is routed through the miner, the two agree.

### 6a. Follow-ons — NOT BUILT

1. **Exact counting of the hybrid phases in place of sampling.**
   - A census of phase 4 over 2³¹ seeds took 75.2 s, and of phase 3, 170.9 s (3080 Ti, MEASURED).
   - At Run 4's size, the 2²⁸ sample costs about 11 s and leaves a 95% interval of ±3.6%. Under
     the new cap that interval alone decides `upper95` (25,086.2 against an exact 23,568).
   - Counting phases 3–4 exactly would remove the sampling question for runs of this size. Phases
     1–2 can stay sampled.
   - This is a design change to the gate's decision path and needs its own brief and Beta review.
     Beta R1 finding 2 already noted it: "If full census becomes a supported decision path …".
     The allocator now accepts a census request of any size (V-N); nothing selects it.
2. **The skip range never reaches phases 1 and 2.**
   - `build_stripe_assign_payload` builds the payload at `miner/range_miner_coordinator.py:9331-9358`
     with no `skip_range` key. `git log -S'"skip_range"'` on that file finds no commit that ever
     added one.
   - The worker takes `payload.get("skip_range", [0, 16])` (`miner/range_miner_worker.py:1164`), so
     the constant kernels scan skip 0–16 whatever the trial sampled. Run 4's 6–99 is recorded in
     metadata only.
   - **Two documents say otherwise:**
     - `docs/HYBRID_SKIP_BOUND_AUDIT.md:113`, row M6a: "`ScalarArg(ctx.skip_min,"int32")`,
       `ScalarArg(ctx.skip_max,"int32")` … **CONSTANT** ABI | ✅ **reaches the kernel**". Its hop
       table goes from M3 (coordinator, `:110`) to M4 (worker unpack, `:111`) with no payload hop.
     - `docs/TEAM_ALPHA_BOUNDED_PHASE_6_SUBMISSION.md:382`: "Sampled `skip_min`/`skip_max` are
       placed in the payload's `skip_range` and reach `for (int skip = skip_min; skip <= skip_max;
       skip++)`."
   - Its effect on hit rate is unmeasured. The measurement that would settle it (Beta R1/R2 Q1) is
     a flag-selectable delivered-skip configuration compared with the current baseline on
     unseen-draw pool hit rate.
   - The fix lives in `miner/`, which this brief forbids editing.

## 7. Beta's positions

**R1** (`docs/exchange/from_beta/X-VOLUME-LIMIT_R1_BETA_REVIEW.md`; relay `REVIEW RECEIVED`).
Each position is Beta's; this table records what was done about it.

| Q | R1 position | What was done |
|---|---|---|
| 1 | NO OBJECTION | — |
| 2 | NO OBJECTION | — |
| 3 | SUGGESTION | **Adopted, within scope.** The interval is labelled conditional on the Poisson model, and per-block hits are recorded and checked. Cap and rule unchanged |
| 4 | NO OBJECTION | — |
| 5 | NO OBJECTION | — |
| 6 | **OBJECTION · LIVE** | **Agreed, and fixed within scope.** Process-isolated sampler with terminate/kill and an UNRESOLVED report; daemon thread for injected samplers; V-J2 subprocess test; V-MUT-5/6. The report's false "daemon thread" claim is corrected (§3, §8) |
| 7 | NO OBJECTION | — |
| 8 | NO OBJECTION | — |
| 9 | SUGGESTION | **Adopted:** `TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE.md` added to the prior record |
| finding 2 | SUGGESTION | **Adopted:** capacity-based block allocation; V-N |

### 7a. R2 — Beta's last positions

`docs/exchange/from_beta/X-VOLUME-LIMIT_R2_BETA_REVIEW.md` (relay `REVIEW RECEIVED`, exit 0).
Request: `docs/exchange/to_beta/X-VOLUME-LIMIT_R2_ALPHA_REPLY.md`.

| Q | R2 position (last) | What was done |
|---|---|---|
| 1 | NO OBJECTION | — |
| 2 | NO OBJECTION | Beta re-ran 33,672 allocator cases and regenerated every stored interval and projection |
| 3 | NO OBJECTION | — |
| 4 | NO OBJECTION | — |
| 5 | NO OBJECTION | — |
| 6 | NO OBJECTION | **R1's LIVE objection is closed by Beta.** Beta independently killed a SIGTERM-ignoring child (exit −9, reaped). A real GPU-driver hang remains UNAVAILABLE in both reviews |
| 7 | NO OBJECTION | Cap kept as the owner's conservative starting policy. The census is load evidence, not completed-run evidence |
| 8 | SUGGESTION (report wording only) | **Adopted**, §3 "For a later per-trial caller". No code change |
| 9 | NO OBJECTION | — |

**No LIVE objection remains.** Beta's additional findings are both ALREADY RECORDED statuses:
- range routing (F-4) matters to using this gate as launch evidence and is not settled here;
- G4 launch-path wiring is a separate item.

**Closing this review neither closes G4 nor authorises Run 4.** No third round was opened: nothing
was left to answer, and the adopted Q8 suggestion is wording in this report only. **Rounds used:
2 of 3.**

**Unresolved, with both positions:** none between Alpha and Beta. The open decisions are the
owner's: the cap and comparison rule against Run 4's 23,568, and the `base_start` routing ruling
(F-4).

## 8. Not verified

- The sampler on ROCm or on a rig: not run.
- That the hand session of 2026-08-27 used exactly `[0, 2²⁶)` and `[0, 2²⁴)`: inferred.
- A real GPU driver hang on timeout: exercised only with a CPU child that sleeps. **Correction of
  R1:** R1 said timed-out work "finishes in a daemon thread". That was false. It was a non-daemon
  pool thread that held process exit (Beta Q6). Now the child is terminated, then killed. A child
  stuck in an uninterruptible GPU call that survives SIGKILL is reported UNRESOLVED and named,
  not hidden.
- Beta's interval-coverage concern: one 8-block observation only (§2.3).
- The parallel `step1_search` session's files and report: not read. F-4 rests on two source
  lines re-checked here.

## 9. Landing list (new files; nothing existing was edited)

See §9a for sha256s of the final files. The report's own hash is in the review request, since the
file cannot contain it.

| file |
|---|
| `utils/survivor_volume_estimator.py` |
| `scripts/survivor_volume_gate.py` |
| `tests/test_survivor_volume_gate.py` |
| `config_manifests/survivor_volume.cap` (`.cap` is not ignored: `git check-ignore` returns nothing) |
| `docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md` |

Evidence under `data/survivor_volume_limit/` is the owner's choice to land. It holds `.py`
helpers (`mb_driver.py`, `live_case.py`) that would trip phase-4 Gate 22 if left untracked in the
main tree. It also holds the M-B, census and live records (`.json`, ignored by `.gitignore:41`)
and the `.log` files. Request files in the main tree:
`docs/exchange/to_beta/X-VOLUME-LIMIT_R1_ALPHA_REVIEW_REQUEST_survivor_volume_gate.md` and
`…_R2_ALPHA_REPLY.md`.

### 9a. sha256 (final)

| file | sha256 | note |
|---|---|---|
| `utils/survivor_volume_estimator.py` | `00b140db7d2bd7ff063286ef77f7dfe6a294bff5927d7a030f66e3fe44a2c130` | as reviewed in R2 |
| `scripts/survivor_volume_gate.py` | `48d558050accb21a380e4f6f33d3be0a45f243f5b2a27748af12ec347519bc39` | as reviewed in R1/R2 |
| `tests/test_survivor_volume_gate.py` | `9b668f72715834109bbf2ce5aabbf40a66dc7ba3b216fba1a6f3ed91af720d57` | as reviewed in R2 |
| `config_manifests/survivor_volume.cap` | `63528e660b209bc0bea480a4e03c974d8b60e4118aa8e34086fb4da9529643e5` | **changed after R2: owner cap 25,000 and new evidence text.** The R2-reviewed version was `4270b47d…` (cap 23,515). Beta has not reviewed this version |

**After the cap change** (no code changed):
- the suite re-ran at **21/21, `COMPLETION SENTINEL: PASS`, exit 0** (`data/survivor_volume_limit/test_run9_cap25000.log`);
- the suite's 23,515 is its own fixture (`tests/test_survivor_volume_gate.py:40`, used by V-B/V-C
  at `:187-201`), written into throwaway repos; it never reads the live cap file.
