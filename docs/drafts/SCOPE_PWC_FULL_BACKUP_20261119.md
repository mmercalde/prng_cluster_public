# SCOPE — PWC AS A FULL BACKUP FOR A RUN-4-STYLE STEP 1 (read-only scoping)

**Date:** 2026-11-19 · **Host:** VM101, as `michael` · **Model:** Opus (this session and three Opus sub-agents)
**Code read:** main `7444aea`, extracted with `git archive 7444aea` into
`/home/michael/tfm_worktrees/pwc-backup/logs/scope_main_7444aea/` (ignored path). The worktree itself is at `b11b6fd`
and was not used for code reads. **All `file:line` anchors below are main `7444aea` unless marked "candidate".**
**Candidate:** the unlanded PWC step-1 work in `/home/michael/tfm_worktrees/pwc-step1` (base `976a6db`, uncommitted;
read only).
**Written:** this file only. Nothing was committed, pushed, branched or tagged. Nothing in the main tree, on any rig or
in `~/miner_staging` was touched. No GPU, no SSH, no engine run.
**Executed (CPU only, in a disposable copy of the snapshot):**
- the three old PWC test scripts;
- one `git merge-file -p` of the candidate against `7444aea`, into scratch.

Logs are copied to `logs/scope_pwc_20261119_evidence/` (ignored).

**Yardstick.** TFM is judged by prediction-pool hit rate against `k/1000` on unseen draws. A backup engine is judged by
whether Steps 2–6 can consume its output the same way. Nothing here judges PWC by whether any seed is found. This
scope does not measure PWC's predictive value. **No hit-rate measurement of PWC-produced survivors exists.**

---

## Prior record

### Searches run (snapshot of `7444aea`; `/bin/grep`, because the shell `grep` skips `*.json`)

```
ls docs/CLAUDE_CODE_INSTRUCTIONS_PWC_BACKUP* docs/CLAUDE_CODE_REPORT_PWC_BACKUP* docs/exchange/*/X-PWC-BACKUP_* \
   docs/exchange/*/X-PWC-CACHE-RESET_* docs/TEAM_ALPHA_PWC_COMPARATOR_SCOPE_CORRECTION.md \
   docs/PROPOSAL_PWC_LIFECYCLE_FIX_S156_v2_0.md docs/drafts/pwc_cache_reset_v1/
/bin/grep -lIiE "PWC|persistent_worker_coordinator" docs/TB_RULING_*.md docs/TB_RULING_REQUEST_*.md \
   docs/PROPOSAL_*.md docs/CHAPTER_*.md docs/*_AUDIT*.md
for f in docs/PROJECT_FILE_CATALOG.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md; do
   /bin/grep -nIiE "PWC|persistent_worker|use_persistent" $f; done
/bin/grep -nI "PWC" docs/exchange/LEDGER.md docs/exchange/DEFERRED.md docs/TFM_SESSION_HANDOFF_20261118*.md \
   docs/drafts/NOTE_RUN_SCOPE_DECISION_20261118.md
/bin/grep -n "expected_skip\|skip-bound\|dead dimension" docs/TFM_PROJECT_FACTS_SKILL.md docs/LEADS.md \
   docs/BACKLOG.md docs/TODO_CONSOLIDATED.md
git log --oneline b11b6fd..7444aea ; git log --oneline 7444aea -- persistent_worker_coordinator.py persistent/
```

The sub-agents also read `docs/HYBRID_SKIP_BOUND_AUDIT.md`, `docs/RUN4_LAUNCH_CHECKLIST.md`,
`docs/TFM_SESSION_HANDOFF_20261114.md`, LEADS L-7 (`docs/LEADS.md:507-640`), the lifecycle proposal and
`docs/SESSION_CHANGELOG_20260323_S153.md`. They did **not** run a separate catalog or ruling search; this session ran it.

### Documents read in full, and what each settles

| Document | Settles | Label |
|---|---|---|
| `docs/CLAUDE_CODE_INSTRUCTIONS_PWC_BACKUP_STAGE_A_20261114.md` | Owner intent, 2026-11-14: PWC kept as a working backup, "underwhelmed", RANGE-MINER first | ALREADY RECORDED |
| `docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STAGE_A.md` (552 lines; the untracked copy here is byte-identical to main's) | Gaps 1–6g, operating history P4, rulings P1, risk P5, Options 0/A/B | ALREADY RECORDED. This scope carries its P3 table forward and does not repeat it as findings |
| `docs/CLAUDE_CODE_INSTRUCTIONS_PWC_BACKUP_STEP1_20261114.md` | Step 1 = PWC-only files (S1–S8); step 2 = everything shared, after Run 4 | ALREADY RECORDED |
| `pwc-step1/docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STEP1.md` (candidate, uncommitted) | S1–S8 built and tested (23/23 gates, 17/17 mutants); the step-2 list, items 1–7 | ALREADY RECORDED (exists, **not landed**) |
| `docs/exchange/{to,from}_beta/X-PWC-BACKUP_R1*`, `_R2*`; `docs/exchange/LEDGER.md:23`, `:25` | Beta: no LIVE objection to step 1. Owner: land after Run 4, no fleet test | ALREADY RECORDED |
| `docs/exchange/{to,from}_beta/X-PWC-CACHE-RESET_R1*`, `_R2*`; `docs/drafts/pwc_cache_reset_v1/README.md` | Fresh rig cache per run, landed in `7444aea`. TCP stale-worker reconnect gap accepted as a known limit, TO-DO if PWC is used | ALREADY RECORDED |
| `docs/TEAM_ALPHA_PWC_COMPARATOR_SCOPE_CORRECTION.md` §1, §5 | The contract is **interface compatibility**: later steps "must not be able to tell whether the data came from PWC or RANGE-MINER". It is not value identity | ALREADY RECORDED. It frames this scope's target |
| `docs/PROPOSAL_PWC_LIFECYCLE_FIX_S156_v2_0.md` (APPROVED WITH MODIFICATIONS, catalog `:606`) | Per-run fleet lifecycle. Only Phase A is in code (`persistent_worker_coordinator.py:484-531`); B and C1–C3 are absent (sub-agent read) | ALREADY RECORDED |
| `docs/HYBRID_SKIP_BOUND_AUDIT.md` §0; `docs/TFM_PROJECT_FACTS_SKILL.md:475` (§2.7 #4, OPEN); `docs/TODO_CONSOLIDATED.md:199` | Hybrid kernels take no skip bounds. **The miner drops them too** (`_hybrid_prefix`). It is an open decision, not a PWC defect | ALREADY RECORDED |
| `docs/TFM_SESSION_HANDOFF_20261114.md:76-80` (finding 7, volume report F-1) | The miner's constant phases get no `skip_range` and fall back to `[0,16]`. Owner decides after Run 4 | ALREADY RECORDED |
| `docs/TFM_SESSION_HANDOFF_20261118.md:56-60`; `docs/drafts/NOTE_RUN_SCOPE_DECISION_20261118.md:31-35` | The PWC warm-cache effect is UNVERIFIED; the fresh-per-run rule applies to PWC if PWC is used | ALREADY RECORDED |
| `docs/RUN4_LAUNCH_CHECKLIST.md:286-292`, `:425-434` | The Run 4 shape and its expected per-phase survivor counts (0, 0, 60, 23,568) | ALREADY RECORDED. Used here as the target |
| `docs/FLEET_STATE_REQUIREMENTS_v1.md:146-195` (via Beta X-PWC-CACHE-RESET R2 Finding 1) | `min_workers` reaches PWC as 1; owner ruling 2026-08-04, not restored | ALREADY RECORDED |
| `docs/SESSION_CHANGELOG_20260323_S153.md:20-48` | SSH-route constant reverse argument mismatch (`a, c, offset` into a 12-parameter kernel) | ALREADY RECORDED, still present (`sieve_gpu_worker.py:228-231`, `:299`) |

### Labels for what this scope raises

| Item | Label |
|---|---|
| The step-1 candidate is **not on main**. It must be merged over `7444aea` | NEW as a status. The merge is clean in code; two test suites need fixture edits (item 2) |
| **The forward-zero prune ends a PWC trial after phase 1.** Run 4's expected phase-1 count is 0, so a Run-4-style trial on PWC would never run phases 2–4 | The prune is ALREADY RECORDED (Stage A P2.1, `:1730-1749` there). **Its consequence for a Run-4-shaped trial is NEW** |
| The Q0 gate (hybrid reverse skipped when hybrid forward is empty) and the single `balanced_hybrid` strategy differ from the miner's four always-run phases and five strategies | Strategy set ALREADY RECORDED (Stage A `:281`). **Both are TB rulings of S147** (comment at `persistent_worker_coordinator.py:1840`, `:1866-1867`). Removing them as defaults **CHANGES A RULING**. Proposed here only as a flag |
| The skip-bound half of the hybrid quarantine's lifting condition is **shared with the miner and identical there** | ALREADY RECORDED (§2.7 #4). That the quarantine **could be lifted on the threshold half alone** **CHANGES A RULING** (comparator ruling §5, `8a55a68`) |
| Holding counted PWC to `offset = 0` (candidate step-2 item 5) **cannot run a Run-4-style trial**: Run 4's anchor is 25 | NEW as a consequence. The anchor/phase split is ALREADY RECORDED (Brief II) |
| Remote `pkill -f` commands match the shell that runs them (`persistent_worker_coordinator.py:761-765`, `:507-511`) | **NEW**, by reading, not executed |
| The TCP route's legacy result parser reads `skip_sequence`, while `sieve_filter` emits `skip_pattern` | NEW, by reading. It does not touch the 22 arrays |
| The miner and PWC send **different constant-phase skip ranges** (miner `[0,16]` by fallback, PWC the sampled range) | The miner half is ALREADY RECORDED (F-1). That it makes the two engines' constant phases differ is NEW as a juxtaposition |
| Credit, quarantine lift, seed-start refusal (R5) | **CHANGES A RULING** (cited per item below) |

### Not searched

The rigs (off; deployed bytes UNAVAILABLE), Proxmox, VM101 live state outside the snapshot, live ledgers and staging,
ser8 archives, the private repository, the project PDFs, the claude.ai project, private owner chat. Changelogs
missing from the repository (Stage A "Not searched") remain a gap.

---

## The answer, in plain words (for the owner)

1. **What "fully functional" means here.** One Run-4-style trial on PWC runs all four phases:
   - forward and reverse constant;
   - forward and reverse variable.

   It covers an arbitrary `[start, end)` inside `[0, 2³²)`, at window 12, anchor 25, thresholds 0.71/0.47 and skip
   6–99 (`docs/RUN4_LAUNCH_CHECKLIST.md:289`). The credit gate accepts it, and the same finalizer writes the same 22
   arrays (`utils/canonical_arrays.py:98-121`; writer `window_optimizer_integration_final.py:3503`).
2. **The output door is already shared.** PWC and the miner end in the same record builder and finalizer, so keys and
   dtypes are identical by construction. **Nothing downstream needs the miner's ledger or staging.** The work is all
   upstream of the door: running the four phases the way the miner does, and proving coverage.
3. **On main today PWC can run at most phase 1 and phase 2 of such a trial, and in the Run 4 shape only phase 1.**
   - Phase 1 is expected to find 0 survivors, and a zero forward-constant result ends the trial (`:1779`).
   - Phases 3–4 raise (quarantine, `:216-239`).
   - A non-zero start is refused (`window_optimizer_integration_final.py:1962`).
   - Its rig workers dial `192.168.3.127` (`:805`).
   - Nothing it produces can be credited (`:2002`).
4. **None of the ten rig-parity-pinned files has to change.** Use the TCP route only, and do the anchor/phase split in
   `sieve_filter.py`. That file is rig-deployed, but it is not imported by `miner/` and not in
   `scripts/gate12_parity_gate.py:174-185`. Then neither `prng_registry.py` nor `sieve_gpu_worker.py` is edited.
5. **What does need shared files:**
   - `window_optimizer_integration_final.py` and `window_optimizer.py`, which are Run 4's launch files;
   - the manifest and WATCHER pass-through.

   These cover seed start, dial address, PWC sizing keys, the credit evidence and the provenance wall. All of it can be
   new PWC-only functions or PWC-branch lines inside those files, with the miner branch proven byte-identical by AST
   comparison.
6. **Four owner rulings are needed:**
   - lift the hybrid quarantine on the threshold half;
   - give PWC credit;
   - drop R5 for PWC;
   - choose the default phase policy (the S147 prune, Q0 and strategy rulings against the miner's always-four-phases,
     five strategies).

   The constant skip-range question (F-1) is already yours to decide after Run 4. It affects which engine PWC should
   copy.
7. **Size:** about **5–7 sessions** of code and tests, plus Beta rounds, plus **2–3 owner-run rig sessions**. The first
   stage is PWC-only and can start whenever you choose. It does not touch anything Run 4 launches.

---

## Baseline: what is on main, and what is not

- **On main (`7444aea`):**
  - the fresh rig cache reset (`persistent_worker_coordinator.py:153-170`, `_reset_rig_cupy_cache`);
  - the hybrid quarantine (`:179-239`);
  - the R5 seed-start guard (`window_optimizer.py:678-712`, called at `:819`, `:1257`;
    `window_optimizer_integration_final.py:1960-1963`).
- **Not on main:** the step-1 candidate, S1–S8:
  - completion accounting;
  - `seed_start`;
  - required dial address;
  - no silent fallbacks;
  - stable envelope;
  - constant threshold echo gate;
  - protocol import;
  - PWC parity gate.

  It is 4 modified files (+696/−50) and 2 new files (588 + 841 lines), uncommitted at
  `/home/michael/tfm_worktrees/pwc-step1`. Beta: no LIVE objection (R1, R2). Owner: land after Run 4. **Run 4 attempt 3
  is credited** (CLAUDE.md §6 status note), so that condition appears met. [The owner confirms.]
- **Merge (executed in scratch):** `git merge-file -p <candidate> <976a6db> <7444aea>` returns rc 0 with no conflict
  markers, and the result parses. Order in the merged file: kill → cache reset → launch with
  `PWC_HOST`/`PWC_PORT`. Output: `logs/scope_pwc_20261119_evidence/merge/merged.py`. One small behaviour change: a rig
  refused by the reset is now *absent* from `_effective_pool_by_host` rather than 0. That is harmless for the
  candidate's `max()` at its `:1728` (sub-agent reading).

---

## The items

Columns, for every item:
- **Where:** what is wrong, with `file:line`.
- **Repair:** the smallest repair.
- **Files:** each file marked **[PWC-only]**, **[SHARED-launch]** (on Run 4's VM101 launch path:
  `window_optimizer.py`, `window_optimizer_integration_final.py`, `utils/*`, `agent_manifests/`,
  `agents/watcher_agent.py`) or **[SHARED-rig]** (rig-deployed and loaded by the miner or parity-pinned:
  `prng_registry.py`, `sieve_gpu_worker.py`, `hybrid_strategy.py`, `adaptive_thresholds.py`, `execution_set.py`,
  `miner/*`).
- **Risk to RANGE-MINER.**
- **Tests:** without rigs, and with rigs.
- **Size.**

`persistent_worker_coordinator.py` is imported at module scope by the integration file
(`window_optimizer_integration_final.py:57-60`; Beta X-PWC-BACKUP R1 Q5). It is PWC-only in execution, but an
import-time error in it would break the miner's launch process too. Every PWC-only change therefore keeps a
"miner launch imports cleanly" check.

### A. The six known gaps you named

**A1. Variable-skip quarantine** (`persistent_worker_coordinator.py:179-239`; raised in `run_sieve_pass` at `:1279`)
- **Where:**
  - **Threshold half: confirmed.** `run_sieve_pass` defaults `phase2_threshold=0.5` (`:1252`). The hybrid call sites
    `:1851-1863` and `:1878-1890` do not pass it. The job carries both thresholds (`:1361`, `:1368`).
    - TCP: `sieve_filter.py:587` uses `phase2_threshold`, so 0.5 reaches the kernel.
    - SSH: `sieve_gpu_worker.py:257-258` does the same.
    - The miner sends the directional threshold as both fields and checks they are equal
      (`miner/range_miner_coordinator.py:9303-9310`, `:9356-9361`; worker `range_miner_worker.py:1374-1380`).
  - **Skip-bound half: confirmed, and the same in both engines.**
    - The forward kernel `java_lcg_hybrid_multi_strategy_sieve` takes no skip bounds and seeds `expected_skip = 5`
      (`prng_registry.py:1006-1013`, `:1027`).
    - The reverse kernel has no `expected_skip`; it tries `0..skip_tolerance` (`:3172-3178`, `:3199`).
    - The miner's assign payload carries no `skip_range` (`range_miner_coordinator.py:9330-9358`), its hybrid prefix
      emits none (`range_miner_worker.py:276-291`), and it uses the same kernels (`_get_kernel`, `:1279`).
    - **The quarantine comment's anchors `:805`, `:885`, `:1159` are the mt19937/xorshift32/minstd forward hybrids**,
      not the java reverse kernel (doc nit, sub-agent).
- **Repair:**
  1. Pass `phase2_threshold=forward_threshold` / `reverse_threshold` at the two call sites.
  2. Add a requested/payload/effective gate for hybrid passes. Reuse the candidate's S6 `pwc_threshold_record` /
     `assert_pwc_threshold_contract`; the TCP worker already echoes.
  3. Replace the quarantine raise with that gate **after a ruling**.
  4. Make **no kernel change**. "Same as the miner" is met by the threshold alone, because the skip dimension is dead
     identically in both engines.
- **Files:** `persistent_worker_coordinator.py` [PWC-only]. No `sieve_filter.py` or `sieve_gpu_worker.py` edit is needed
  for the threshold.
- **Risk to miner:** none in execution.
- **Ruling:** lifting on the threshold half alone **CHANGES A RULING** (comparator ruling §5, `8a55a68`; lifting
  condition `:235-238`). The open §2.7 #4 decision then governs both engines together.
- **Tests:**
  - **No rigs:** CPU fake-worker gates for agree/disagree/unreported on phases 3–4, and a mutant that drops the kwarg.
    VM101 3080 Ti, one local TCP worker: a small interval through PWC phases 3–4, compared with the miner on VM101 over
    the same interval, same threshold. Survivor sets equal is the expected outcome. This is an interface/consistency
    check, not a yardstick.
  - **Rigs:** part of the four-phase smoke (stage 4).
- **Size:** about 60–100 lines and about 150 lines of tests; 0.5 session.

**A2. PWC refuses a non-zero seed start** (`c28555d`: `window_optimizer.py:678-712`, the refusal at `:705`;
integration call `:1960-1963`; PWC partitions from 0, `persistent_worker_coordinator.py` chunking in
`run_sieve_pass`)
- **Repair:**
  1. The candidate's S2 (`validate_pwc_seed_interval`, partition from `_req_start`). Already built, byte-equal at 0.
  2. Pass `seed_start=seed_start` at `window_optimizer_integration_final.py:1968-1986`.
  3. Narrow `:705` so it refuses ZMQ only.
- **Files:** coordinator [PWC-only]; integration call site and `window_optimizer.py` [SHARED-launch]. The edited lines
  are PWC/ZMQ-only: the guard is reached on the miner route only with the PWC/ZMQ flags false.
- **Risk to miner:** low, by AST identity of every function except `assert_step1_route_seed_start_supported`.
- **Ruling:** dropping R5 for PWC **CHANGES A RULING** (owner decision, `CLAUDE_CODE_INSTRUCTIONS_MINER_SEED_START_20261114.md:92`).
- **Tests:**
  - **No rigs:** the candidate's G-S2 gates, plus an integration-level gate (fake `run_trial_persistent`) that the
    start arrives, and the existing seed-start suite still green.
  - **Rigs:** a smoke at `seed_start = 2³¹`.
- **Size:** about 10 lines; 0.25 session (on top of the candidate).

**A3. TCP stale-worker reconnect** (Beta X-PWC-CACHE-RESET R1/R2; README "Known limit")
- **Where (sub-agent reading):**
  - `_handle_client` always replies `accepted: True` (`persistent/pwc_transport_tcp.py:393-411`).
  - It registers and inits any `worker_id` (`:413-426`).
  - The ready count is a set length (`:257-260`).
  - `request_job` has no identity check (`:514-550`).
  - The listener starts before the rig kill (coordinator `:534-546`).
  - `shutdown()` never tells workers to exit (`pwc_transport_tcp.py:994-1020`).
  - Workers retry every 5 s (`pwc_worker_service.py:39`), and a rejected hello retries forever (`:357-361`).
- **S185 does not close it.** PWC gets only `_execution_set_nodes` (`:105-111`). Registration refusal is miner-only
  (`execution_set.py:930-950`).
- **Repair:**
  - **Level 1:** the coordinator hands the transport the launch set (worker ids whose launch returned a PID, plus the
    local ones). Any other id gets `accepted: False`.
  - **Level 2:** a per-`startup()` `secrets.token_hex` exported as `PWC_LAUNCH_TOKEN`, sent in the hello and checked.
    The worker exits on explicit rejection.
- **Files:** `persistent/pwc_transport_tcp.py`, coordinator; `persistent/pwc_worker_service.py` for level 2 —
  [PWC-only]. They are rig-deployed (needs redeploy and the PWC parity pin), not in gate12's set, and not imported by
  `miner/` (comments only).
- **Risk to miner:** none.
- **Tests:**
  - **No rigs:** a real `TcpTransport` on loopback with an unlisted fake worker refused and a listed one served; a token
    mismatch refused; a mutant that always accepts.
  - **Rigs:** start a coordinator, kill it, leave the workers, start a second one, and require that no old id is served.
- **Size:** level 1 about 40 lines, level 2 about 80 lines, plus about 150 lines of tests; 0.5–1 session.

**A4. The ten stale checks in the three old test scripts** (executed by sub-agent, logs in evidence dir)
- `test_s149_rocr_isolation_harness.py` (repo root): **10/15, 5 FAIL.**
  - It asserts the S149 *bug* still exists: T2 ×3, T3, T7. The S155/S157 fix is in place (`:649-651`, `:780-781`).
  - Stale expectations, not defects. Repair: invert them to the S155 design, or archive the file (owner choice).
  - [PWC-only test]
- `test_persistent_worker_harness.py` (repo root): **21/25, 4 FAIL.**
  - T13/T14 hit the deliberate quarantine. Repair: assert the refusal.
  - T19 uses a legacy unversioned result rejected by the v2 wall (`canonical_records.py:431`). Repair: build the
    fixture with `build_trial_populations`.
  - T20: the `MagicMock` coordinator makes `use_range_miner` truthy, so the miner branch runs and
    `resolve_assembly_backend` raises (`miner/assembly_backends.py:368`). Repair: `use_range_miner=False`, plus a v2
    fixture.
  - [PWC-only test]
- `tests/test_s172_threshold_propagation.py`: **3/5 gates — G-MINER-UNCHANGED FAIL, G-KERNEL UNAVAILABLE.**
  - G-MINER-UNCHANGED fails through its D6 sub-harness `tests/test_s172_phase5_d6_threshold_path.py`. That harness is
    stale since `48a8705`: it passes `offset` and omits `generator_phase` to `build_stripe_assign_payload`
    (`:226`, `:420`, `:460`, `:732`, `:750` vs `range_miner_coordinator.py:9216-9221`).
  - G-KERNEL needs a GPU and `daily3.json`.
  - **This file is SHARED (a miner test).** Repairing the D6 sub-harness is miner-test work. Before-commit Beta class
    is not triggered (test only), but it belongs with the miner owner.
  - In the snapshot copy, G-MINER-UNCHANGED failed earlier at `git status` (no `.git`). That is an environment
    artefact; the cause above is from the git-clone run at
    `/home/michael/tfm_worktrees/rig-gpu-fault/logs/pwc_cache_v1/before/test_s172_threshold_propagation.log:9-32`.
- **Risk to miner:** none from the two root files. Editing the D6 sub-harness changes no production code.
- **Size:** about 80 lines of test edits; 0.5 session.

**A5. Search-record credit, volume gate, ledger/staging, output contract versus RANGE-MINER**
- **Output contract:** shared door, identical keys and dtypes. 22 arrays at `utils/canonical_arrays.py:98-121`; both
  routes end in `build_mode_records` (`utils/canonical_records.py:225`). **No gap in shape.**
  - Content gaps are items B1–B4 (what is run).
  - The asymmetric ingress wall is A5c.
- **A5a. Credit.**
  - `credit_decision` (`utils/step1_search_ledger.py:1044-1086`) needs a COMPLETE trial and `swept_intervals` equal
    to the request for phases (1,2,3,4) (`window_optimizer_integration_final.py:1386-1395`). Nothing else: no ledger,
    no run id, no stripe facts.
  - PWC is stamped `_step1_unaccounted_route_evidence` (`:2002-2006`), so `swept_intervals = {}`.
  - **The pruned return (`:1986-1994`) carries no evidence at all.**
  - **Repair:** a new top-level `_step1_pwc_sweep_evidence(_pw_result)` reading the candidate's `pwc_completion`
    (already in the `{phase: [s, e]}` shape). Stamp it on both the full and the pruned returns. Credit only when
    `pwc_trial_complete`.
  - **Files:** integration file [SHARED-launch], new PWC-only function plus 2 call lines.
  - **Ruling:** **CHANGES A RULING** (Beta 2026-07-31 retirement; X-STEP1-SEARCH-CODE R1 Q1 / R2).
  - **Tests (no rigs):** a failed-chunk trial gets no credit; a zero-survivor complete trial is credited (Beta's own
    paired control, Stage A `:448-449`); a mutant that drops accounting.
  - **Size:** about 60 lines and about 200 lines of tests; 0.5 session.
- **A5b. Survivor-volume gate (`35f5503`).**
  - A manual pre-launch script (`scripts/survivor_volume_gate.py`). Nothing on the launch path calls it. It samples
    through the **miner's** `SieveExecutor` (`utils/survivor_volume_estimator.py:270`, `:299-304`).
  - Once PWC matches the miner's threshold and phase policy (A1, B2), the miner-kernel estimate is the right estimate
    for PWC too. Before that, it does not describe PWC.
  - **Repair:** none in code. The PWC runbook runs the same manual step.
  - **Size:** 0.
- **A5c. Threshold provenance wall.**
  - Miner ingress must carry a validated `threshold_provenance` (`window_optimizer_integration_final.py:1351-1364`).
    PWC ingress (`_build_test_result_from_pw`, `:1162`) has none.
  - **Repair:** map the candidate's per-pass `accounting.threshold` records into the same provenance shape, PWC branch
    only.
  - **Files:** integration [SHARED-launch], PWC function.
  - **Size:** about 40 lines; 0.25 session.
- **A5d. Ledger and staging.**
  - **Not needed.** `finalize_run`, `record_certified_coverage` and the S8 link are route-neutral
    (`:3503`, `:3549-3566`, `:3578-3591`).
  - The miner ledger is read only by `_step1_miner_sweep_evidence` (`3579088` is miner-only).
  - **No repair.**
- **A5e. Fleet/dataset gates for a counted run.**
  - With a missing provisioning manifest, P0.5 is lenient for non-miner runs (`miner/dataset_authority.py:1109`).
  - **Repair:** pass `require_fleet=bool(use_persistent_workers)` at the two callers (`window_optimizer.py:1896`,
    `agents/watcher_agent.py:2122`). **Do not edit `dataset_authority.py`** (parity-pinned).
  - **Files:** [SHARED-launch], 2 lines.
  - Cache policy v7's cache record and join gate are miner-only (`9034e18`). Whether a counted PWC run needs an
    equivalent record is an **owner choice**. PWC's reset prints a marker but writes no record.
  - **Size:** 0.25 session, plus about 0.5 if a PWC cache record is wanted.

**A6. Never run on real hardware with the fresh-cache patch**
- The reset is tested only with fake ssh (21/21, `tests/test_pwc_cache_reset.py`).
- The warm-versus-fresh comparison is **not defined anywhere** (README `:38` names it in one line; L-7's 2×2 protocol
  is miner-only).
- It also needs a flag to *skip* the reset, which does not exist.
- **Repair:**
  - a PWC-only `pwc_cache_policy = per_run | keep` setting, default `per_run`, refusing `keep` on any counted run;
  - one owner-run test (stage 4).
- **Files:** coordinator [PWC-only].
- **Tests:**
  - **No rigs:** fake-ssh gates for both policies.
  - **Rigs:** one boot, two rigs, pool 6, chunk ≤ 50K, constant-only, three short runs (reset / keep / reset). Record
    worker ids (needs A3), reset markers, cache mtimes, the fault recorder and per-pass accounting. **Single runs
    cannot give a rate**; this shows only that the reset works on real ssh and whether PWC's warm path faults at all.
- **Size:** about 30 lines; 0.25 session, plus one rig session.

### B. Found in this scope (not in your list)

**B1. The forward-zero prune ends the trial after phase 1** (`persistent_worker_coordinator.py:1779-1793`; adapter
`window_optimizer_integration_final.py:1986-1994`)
- Run 4's expected phase-1 count is **0** (`RUN4_LAUNCH_CHECKLIST.md:429-434`), so a Run-4-shaped PWC trial stops
  before phases 2–4. The miner runs all four (`range_miner_coordinator.py:2754-2759`, `:12058-12062`).
- Also the **Q0 gate** (`:1866-1875`): hybrid reverse is skipped when hybrid forward is empty.
- Both are S147 design.
- **Repair:** a PWC-only `pwc_phase_policy = s147 | all_phases` setting:
  - `all_phases` runs 1–4 unconditionally;
  - `s147` stays the default and baseline, per the yardstick rule's point 4, until the owner rules;
  - a counted Run-4-style trial requires `all_phases`, because `credit_decision` needs all four intervals.
- **Files:** coordinator [PWC-only].
- **Ruling:** making `all_phases` the default for counted runs **CHANGES A RULING** (S147 Q0/B1).
- **Tests (no rigs):** a fake worker returning zero in phase 1 still runs phases 2–4 under `all_phases`; under `s147`
  it prunes, unchanged.
- **Size:** about 40 lines; 0.25 session.

**B2. One hybrid strategy versus the miner's five** (`:1840-1849` loads `balanced_hybrid`, "TB ruling: balanced_hybrid
for discovery"; miner loads all five, `range_miner_worker.py:1227-1238`, `hybrid_strategy.py:35-83`)
- **Repair:** part of `pwc_phase_policy=all_phases`: pass `strategies=None`, so `run_sieve_pass` loads all five
  (`:1282-1291`).
- **Files:** [PWC-only].
- **Ruling:** CHANGES A RULING (S147 Q2) as a default; fine as a flag.
- **Size:** in B1.

**B3. One `offset` is both window anchor and device generator phase** (`sieve_filter.py:535-538` slice; `:461`
forward-hybrid and `:311` constant kernel arg; SSH `sieve_gpu_worker.py:113`, `:278`, `:304`; miner pins phase 0,
`range_miner_coordinator.py:9316-9327`)
- With Run 4's anchor 25, PWC phases 1, 2 and 4 pre-advance the generator by 25 while the miner advances by 0. Both
  write `offset = 25` to the NPZ (`canonical_records.py:225`, `:374`). The rows are indistinguishable but encode
  different generator configurations. **This is exactly the "later steps cannot tell" property, broken in content.**
- Holding PWC to `offset = 0` (candidate step-2 item 5) avoids it but cannot run a Run-4-style trial.
- **Repair:**
  - TCP route only: split `offset` into `window_anchor` (slice) and `generator_phase` (kernel arg, pinned 0 like the
    miner) inside `sieve_filter.execute_sieve_job` / `run_sieve` / `run_hybrid_sieve`;
  - the coordinator sends both fields;
  - refuse the SSH transport for counted runs.
  - Also drop the stray trailing `offset` the forward java hybrid gets (16 args into a 15-parameter kernel,
    `sieve_filter.py:450-461` vs `prng_registry.py:1006-1013`; harmless or not is [UNVERIFIED]).
- **Files:** `sieve_filter.py` (rig-deployed, **not** miner-imported, **not** gate12-pinned, **is** in the candidate's
  PWC pin) and the coordinator — [PWC-only].
  - The alternative of fixing the SSH route needs `sieve_gpu_worker.py` [SHARED-rig] and is not recommended.
- **Risk to miner:** none.
- **Tests:**
  - **No rigs:** a CPU reference of the java kernels (exists, 2026-11-13, see the yardstick doc's §4 example) against
    `sieve_filter` arg construction. On the VM101 GPU: PWC TCP local worker against the miner on the same small
    interval at anchor 25, all four phases. The expected outcome is equal survivor sets.
  - **Rigs:** the four-phase smoke.
- **Size:** about 80 lines and about 200 lines of tests; 1 session. This is the largest PWC-only item.

**B4. Constant-phase skip range differs between the engines**
- PWC sends the sampled `[skip_min, skip_max]` (6–99 in Run 4) to phases 1–2 (`:1772`). The miner sends none and its
  worker uses `[0,16]` (F-1, `range_miner_worker.py:1164`/`:1293`).
- This is **not a PWC defect**. It is an open owner decision on the miner.
- **Repair:** PWC follows the sampled range (its current behaviour, the design intent "skip is a cornerstone").
- **Consequence:** until F-1 is decided, a PWC trial and a miner trial with identical parameters differ on phases 1–2.
  Engine-to-engine comparisons must then run at a skip range inside `[0,16]`, or wait for F-1. Optionally add a
  PWC-only `pwc_constant_skip = sampled | miner_fallback` flag for comparison runs only.
- **Files:** [PWC-only].
- **Size:** 0–20 lines.

**B5. Remote `pkill -f` matches its own shell** (TCP kill `persistent_worker_coordinator.py:761-765`; S156 reap
`:507-511`)
- The remote command string contains the pattern, so the remote `sh -c` that runs `pkill -9 -f pwc_worker_service`
  matches and is killed. The later `pkill … sieve_gpu_worker` and `echo` then likely never run. The return code and
  output are ignored (`:766-768`).
- The S156 regex `sieve_gpu_worker.*--persistent` also matches its own command line.
- By reading, not executed. It is the class CLAUDE.md hard rule 8 names.
- **Repair:** `pkill -f '[p]wc_worker_service'`-style bracket patterns (or PID files written by the launch script),
  then check the `killed` marker the way the cache reset checks its marker.
- **Files:** [PWC-only].
- **Tests (no rigs):** run the exact remote string under a local `sh -c` against a dummy process named
  `pwc_worker_service`, and require the marker.
- **Size:** about 20 lines; 0.25 session.

**B6. Small PWC-only items**
- **TCP legacy parser** reads `skip_sequence` (`:1157`) while `sieve_filter.py:497-500` emits `skip_pattern`, so skip
  sequences come back empty. They are not one of the 22 arrays. Fix for completeness, about 2 lines.
- **`HSA_OVERRIDE_GFX_VERSION=10.3.0`** injected into every PWC worker (`:140`). The miner sets none. Effect UNVERIFIED
  (Stage A 6g). Make it a PWC-only setting, default off, to match the miner, about 10 lines. **Owner decides**, since
  PWC's whole operating history ran with it.
- **Address resolution:** rigs resolve to the CT100 addresses only when the execution set is frozen (`:105-111`,
  `:351`). `node_allowlist` is compared after re-pointing (`:718-726`), so bare-metal names drop everything under
  Proxmox. Document it; refuse a counted PWC run without a frozen execution set (about 10 lines).
- **min_workers = 1** (owner ruling 2026-08-04): one VM101 local worker can satisfy the ready gate if every rig is
  skipped. For a counted run, completion accounting (S1) makes coverage honest even then. Status, no change.

**B7. Already-recorded items that stay as status**
- **Launch-per-trial** (`PROPOSAL_PWC_LIFECYCLE_FIX_S156_v2_0.md`, only Phase A built). Inside the envelope and on few
  trials it is tolerable. It is the launch-storm shape (facts §2.50). Optional stage 5.
- **SSH-route constant reverse arg mismatch** (S153) and the **missing SSH threshold echo** (candidate step-2 item 6):
  both need `sieve_gpu_worker.py` [SHARED-rig]. Avoided by refusing SSH for counted runs.
- **The manifest's pool 8 / 2M chunk** lies outside the recorded envelope (Stage A 6c). The candidate's S5 refuses it.
  The plumbing needs PWC-only sizing keys (candidate step-2 item 3), [SHARED-launch], about 40 lines across manifest,
  `window_optimizer.py` (both paths, S170 parity rule) and WATCHER pass-through. **The dial address** goes with them
  (`pwc_coordinator_address`, no default; VM101 `192.168.3.177` is still DHCP, so check it live).
- **The hybrid reverse kernel** writes survivors via `atomicAdd` with no bound (`prng_registry.py:3230-3231`). It is
  shared by both engines; not a PWC item. Recorded here only so that it is not lost [UNVERIFIED whether recorded
  elsewhere].

---

## What cannot be done without touching shared code

| Need | Shared file | Kind of edit | Avoidable? |
|---|---|---|---|
| Credit (A5a), provenance wall (A5c) | `window_optimizer_integration_final.py` [SHARED-launch] | New PWC-only functions + PWC-branch call lines | **No.** Proof: AST identity of `_build_test_result_from_miner`, `_step1_miner_sweep_evidence`, `_step1_ledger_rows_via_copy`, `_miner_residues_for_config`, `credit_decision`, `_step1_required_phases` and the `_use_miner` branch subtree |
| Seed start (A2) | integration call site; `window_optimizer.py:705` [SHARED-launch] | One kwarg; narrow one condition | **No** |
| Dial address, PWC sizing keys (B7) | manifest, `window_optimizer.py` (both paths), WATCHER pass-through [SHARED-launch] | New PWC-only keys | **No** for a WATCHER-launched run. **Yes** for a direct `run_trial_persistent` call (candidate §9), which is enough for the rig smokes |
| `require_fleet` for PWC (A5e) | `window_optimizer.py:1896`, `agents/watcher_agent.py:2122` [SHARED-launch] | One argument each | **No** for counted runs |
| SSH-route anchor/phase, threshold echo, reverse arg fix (B3, B7) | `sieve_gpu_worker.py` [SHARED-rig] | Kernel-argument changes | **Yes**: make counted PWC TCP-only |
| Hybrid skip bounds actually constraining the kernels | `prng_registry.py` [SHARED-rig] | Kernel signatures | **Not needed for a backup** (the miner has the same behaviour). Needed only if the owner decides §2.7 #4 for both engines |
| D6 sub-harness repair (A4) | `tests/test_s172_phase5_d6_threshold_path.py` (miner test) | Test fixture only | Can be left red as recorded, but it then keeps masking G-MINER-UNCHANGED |

**Nothing in `scripts/gate12_parity_gate.py:174-185` (the ten rig-pinned files) needs to change on this plan.**

---

## Staged plan (smallest safe stage first)

| Stage | What | Files | Rulings first | Tests | Size |
|---|---|---|---|---|---|
| **0** | Owner decisions only, no code: (a) TCP-only for counted PWC; (b) phase policy flag, default `s147`; (c) lift quarantine on threshold half; (d) PWC credit; (e) drop R5 for PWC; (f) HSA override; (g) PWC cache record; (h) the F-1 order (before or after PWC) | — | all listed | — | owner time |
| **1** | Land the step-1 candidate merged over `7444aea`. Repair `tests/test_pwc_backup_step1.py` (its fake ssh returns `"8\n"`, so the reset refuses every rig; return the marker for `cupy_cache_gpu_` commands) and `tests/test_pwc_cache_reset.py` (pass the address, `seed_cap_amd=25000`, the transport). Add B5 (pkill), A3 level 1, B6 parser, A4 root tests | [PWC-only] + the two root tests | none (Beta R1/R2 no LIVE objection; owner "after Run 4") | candidate suite + cache suite + six existing suites, sequentially; miner launch import check; gate 22 transitional red, clean on commit | ~1.5 sessions; Beta before-commit round |
| **2** | PWC semantics to match the miner, all PWC-only: A1 threshold + gate (quarantine still raising until ruling (c)); B1/B2 `pwc_phase_policy`; B3 anchor/phase split in `sieve_filter.py` and SSH refusal for counted runs; A6 cache-policy flag; A3 level 2 | [PWC-only] incl. `sieve_filter.py` (rig redeploy, PWC pin) | (c) to switch the raise off; (b) for the default | CPU gates and mutants; **VM101 3080 Ti local-worker four-phase comparison against the miner on a small interval at anchor 25 and skip inside `[0,16]`** (owner-initiated, it uses a GPU) | ~2 sessions; Beta round |
| **3** | Shared plumbing, PWC-only lines: A2 call site + guard; A5a evidence (full and pruned); A5c provenance; A5e `require_fleet`; B7 address + sizing keys in manifest, `window_optimizer.py` both paths, WATCHER | [SHARED-launch] | (d), (e) | AST-identity proof of the named miner functions and branch; Step-1 search, seed-start, credit-WAL, D3.25, resolved-execution-set, phase4 suites; miner gate12 sha check unchanged | ~1.5–2 sessions; Beta before-commit (launch path) |
| **4** | Rig, owner-run, nohup: (i) PWC parity gate against the rigs; (ii) A6 reset/keep/reset; (iii) A3 reconnect test; (iv) two-rig constant smoke inside the envelope; (v) one Run-4-shaped four-phase trial on a bounded interval, credited end to end, NPZ through the Step-2 loader with `fallback_used=False` | — | — | as named; byte or set comparison to a miner run of the same interval is an interface check, not the yardstick | 2–3 owner sessions |
| **5** (optional) | Per-run fleet lifecycle (S156 B, C1–C3); a measured envelope beyond 6/50K and 8/25K | [PWC-only] (+ integration session scope, shared) | owner | soak | 2+ sessions |

**After stage 4**, the yardstick question is open for PWC exactly as for the miner: the hit rate of pools built from
PWC-credited generations on unseen draws, measured the same way. Until that is measured nobody can say PWC output is
as good or worse.

---

## Not verified

- Deployed rig bytes; real SSH and TCP behaviour of the reset, the kill and the reconnect.
- Whether CuPy tolerates the 16th argument to the forward hybrid.
- How Run 4 overrides the manifest's PWC defaults.
- The full extra-key behaviour of `validate_trial_populations` (the candidate's tests say it accepts `pwc_completion`).
- Whether any ruling requires v7 cache records for PWC.
- B5, by reading only.
- Every sub-agent claim was spot-checked at its key anchors in this session:
  - the prune and Q0 gate, the strategy load, the Run 4 table, the miner phase table and `generator_phase` pin;
  - the 22-array count, the PWC branch and the R5 guard, the volume gate's callers;
  - the pkill commands, the forward-hybrid argument list, the test-log failures.

  The rest are the sub-agents' readings.
