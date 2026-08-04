# PIPELINE_BEHAVIOUR_MODEL.md — REV1

**A verified model of how the TFM pipeline works, and why it is built that way.**
**This is not an audit.** Nothing here claims anything is broken, and nothing was repaired.

**Produced:** 2026-08-03 on VM101 (`192.168.3.177`) as `michael`, venv `~/venvs/torch`,
from `/home/michael/distributed_prng_analysis` at HEAD **`49c13ad`** (`git pull` → *Already up to date*).
**Authority:** `docs/CLAUDE_CODE_INSTRUCTIONS_PIPELINE_BEHAVIOUR_MODEL.md` (REV1).
**Index used:** `docs/PROJECT_FILE_CATALOG.md` (read first, in full).

---

## 0. How to read this file

Every behaviour below carries **two anchors**:

| anchor | establishes | form |
|---|---|---|
| **WHY** | the intent | a chapter / whitepaper / proposal / TB ruling, cited `file:§` |
| **WHAT** | the implementation | a source location, `file:line`, **read on VM101 this session** |

Three markers are load-bearing:

- **`INCOMPLETE`** — a behaviour for which only a code anchor was found. That is a statement about
  *this search*, not about the repository. **The governing fact of this project is that every line
  has been documented; a missing WHY means the explanation has not been found yet.** §17 lists all
  of them so the next reader knows exactly where to look, not so anything gets deleted.
- **`DIVERGENT`** — documentation and code disagree. **Both readings are recorded and neither is
  adjudicated.** §16 is the full register.
- **`GOVERNED`** — a known condition already diagnosed, escalated or mid-remediation, with the
  ruling cited. **A `GOVERNED` item re-reported as a new finding is a governance error**, and
  preventing that is a large part of why this document exists.

**Line anchors expire.** Where a symbol name will survive an edit and a line number will not, the
symbol is cited and the line is offered as a convenience. Both Chapter 1 §17.2 and Chapter 2 §13
adopted this convention after their own anchors moved without changing.

---

# 1. THE PIPELINE ON ONE PAGE

## 1.1 Two numbering schemes — read this before anything else

There are **two** step numberings in this project, and confusing them is the single most common
error a new session makes.

| scheme | who uses it | where the sieve is |
|---|---|---|
| **conceptual** | the whitepaper, `TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md` §1, the chapter titles | *"Step 2 = Bidirectional Sieve"* |
| **executable** | `STEP_SCRIPTS` / `STEP_MANIFESTS` / `STEP_NAMES`, all seven `agent_manifests/*.json`, `preflight_check.py`, `README.md` | the sieve is **inside executable Step 1**; executable **Step 2 is the Scorer Meta-Optimizer** |

- **WHY:** `docs/PROJECT_FILE_CATALOG.md` §5.2 — *"Chapter numbers are not step numbers. Chapter 3
  documents Step 2.5 / WATCHER step 2; the bidirectional sieve documented in Chapter 2 runs inside
  Step 1."*
- **WHAT:** `agents/watcher_agent.py:386-416` — `STEP_SCRIPTS[1] = "window_optimizer.py"`,
  `STEP_SCRIPTS[2] = "run_scorer_meta_optimizer.sh"`. **There is no executable step whose script is
  the sieve.**
- **The mapping between the two schemes is written down nowhere.** `docs/CHAPTER_3_ALIGNMENT_AUDIT.md`
  §2 (Q1) searched for it and reports it as *folklore* — *"No file in the repo states 'conceptual
  Step 2 = the sieve, which executes inside executable Step 1'."* **DIVERGENT — D1** (§16).

Two further name hazards, both already recorded:

- **"Phase 7" names two unrelated milestones.** The Phase 7 marked COMPLETE in Chapters 10, 12 and 13
  is **WATCHER dispatch integration (Feb 2026)**. **S172 Phase 7 is the 25-GPU saturation + WATCHER
  soak** — **24 AMD RX 6600 XT + one VM101 RTX 3080 Ti** (the second 3080 Ti stays on VM100);
  owner-ruled, **Team Beta ratified the waiver**, frozen execution set `bea580e7…f67a8` (25
  identities, 25 requested, 25 admitted, unclamped, non-partial). (`PROJECT_FILE_CATALOG.md` §5.2.)
  **Older "26-GPU" wording predates this ruling** — but see §16 D17 and §17 I-5: the 26 in
  `ml_coordinator_config.json` is a live fact about a tracked file and is **not** a stale soak figure.
- **`PROPOSAL_S172_RANGE_MINER_v1_4_4.md` scopes RANGE-MINER as a "Step 1 replacement"** — it is
  written in the *executable* scheme, and is talking about the same engine the system map calls
  Step 2. Both are correct in their own scheme.

**Throughout this document, bare "Step N" means the EXECUTABLE scheme** (the one WATCHER runs).

## 1.2 The two carriers

Everything downstream of the sieve rides on exactly two objects:

1. **The frozen 22-array NPZ survivor bundle** — Step 1 → Step 2 → Step 3.
   - **WHY:** `TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md` §7.1; `PROPOSAL_S172_RANGE_MINER_v1_4_4.md` §4.2, §12.1.
   - **WHAT:** `utils/canonical_arrays.py:99-126` (`CANONICAL_ARRAY_CONTRACT`); verified live —
     **22 arrays**, and the 24-field record side at `utils/canonical_arrays.py:143`
     (`CANONICAL_RECORD_FIELDS`, **24 fields**).
2. **The prediction pool + coverage/lift score** — Step 6 → Chapter 13 feedback.
   - **WHY:** system map §1, §2.
   - **WHAT:** `evaluate_pools.py:28` (`evaluate_pools`, hit + lift-vs-random at `:36-40`);
     the S140b downstream score at `chapter_13_orchestrator.py:306-316`.

## 1.3 The executable map, as built

Parsed live this session from `agents/watcher_agent.py:386-416` and all seven
`agent_manifests/*.json`:

| step | name | script actually run | manifest | primary output |
|---|---|---|---|---|
| **0** | Regime Segmentation (TRSE) | `trse_step0.py` | `trse.json` v1.15.1 | `trse_context.json` |
| **1** | Window Optimizer | `window_optimizer.py` | `window_optimizer.json` v1.8.0 | `optimal_window_config.json` (+ the certified NPZ generation) |
| **2** | Scorer Meta-Optimizer | `run_scorer_meta_optimizer.sh` | `scorer_meta.json` v1.3.0 | `optimal_scorer_config.json` |
| **3** | Full Scoring | `run_step3_full_scoring.sh` | `full_scoring.json` v1.3.0 | `survivors_with_scores.json` |
| **4** | ML Meta-Optimizer | `adaptive_meta_optimizer.py` | `ml_meta.json` v2.0.0 | `reinforcement_engine_config.json` |
| **5** | Anti-Overfit Training | `meta_prediction_optimizer_anti_overfit.py` | `reinforcement.json` v1.10.0 | `models/reinforcement/best_model.meta.json` |
| **6** | Prediction Generator | `prediction_generator.py` | `prediction.json` v1.5.0 | `predictions/next_draw_prediction.json` |
| **fb** | Live Feedback Loop | `chapter_13_orchestrator.py` | — (not a WATCHER step) | diagnostics + retrain triggers |

Data flow, in the executable scheme:

```
daily3_current.json (pointer) ─► daily3-<UTC>Z-<sha12>.json (immutable)
        │
   Step 0  trse_step0.py ──────────────────► trse_context.json ─┐
        │                                                        │ (passive read)
   Step 1  window_optimizer.py ◄─────────────────────────────────┘
        │   └─ Optuna TPE over 7 dimensions
        │       └─ per trial: run_bidirectional_test()
        │            └─ backend cascade: RANGE-MINER │ PWC │ ZMQ │ legacy
        │                 └─ forward sieve ∩ reverse sieve  → survivors
        ├──► CERTIFIED NPZ GENERATION (22 arrays) ── the artifact that matters
        ├──► optimal_window_config.json
        └──► train_history.json (80%) + holdout_history.json (20%)
        │
   Step 2  run_scorer_meta_optimizer.sh ──► optimal_scorer_config.json  (11 scalars)
        │
   Step 3  run_step3_full_scoring.sh ────► survivors_with_scores.json   (91 features/seed)
        │
   Step 4  adaptive_meta_optimizer.py ───► reinforcement_engine_config.json (capacity only)
        │
   Step 5  meta_prediction_optimizer_anti_overfit.py ─► best_model.* + best_model.meta.json
        │
   Step 6  prediction_generator.py ──────► predictions/next_draw_prediction.json
        │
   Ch13   draw arrives → grade → (attribute) → decide → relearn (3→5→6)
```

---

# 2. STEP 0 — Regime Segmentation (TRSE)

**Purpose.** TRSE is a **classifier, not a search.** It reads the draw history, slides windows over
it at three scales (W200/W400/W800), extracts per-window entropy and digit-transition features, and
KMeans-clusters them to answer: *which temporal regime is the sequence in, how long has it been
there, and how stable is it?*

- **WHY:** `TRSE_v1_15_SPEC.md:59-61` — *"This is a classification based on signal shape, not a
  brute-force seed search."* Its entire sanctioned influence is **Step-1 SearchBounds narrowing
  only; Steps 2–6 unchanged** (`TRSE_INTEGRATION_PLAN_S121.md` §3).
- **WHAT:** `trse_step0.py` (`classify_regime_type:397`, `analyze_skip_entropy:483`,
  `detect_offset_periodicity:538`, `save_context:795-800`) — anchors from
  `docs/TRSE_STEP0_AUDIT_v1.md` §1, which obtained them from live source.

**Inputs.** `daily3.json` (manifest `required_inputs`, parsed live from `agent_manifests/trse.json`).
**Output.** `trse_context.json`.
**The seam.** *Passive*: Step 0 writes the file; Step 1 reads it if present and narrows its own
bounds. WATCHER parses nothing and injects nothing.

- **WHY:** `TRSE_INTEGRATION_PLAN_S121.md` §2C — *"Step 1 reads `trse_context.json` on its own if
  present… WATCHER doesn't need to parse or inject anything."*
- **WHAT:** consumer `window_optimizer_bayesian.py` `_load_trse_context` (chapter-cited `:25-47`,
  applied `:495-533`; anchors from `TRSE_STEP0_AUDIT_v1.md` §3–§4).

**Why it is built this way.** Three rules were designed; only one applies.

| rule | state | authority |
|---|---|---|
| **A** — narrow the `window_size` ceiling when the regime is `short_persistence` and confident | **APPLIED, wired end to end** | `TRSE_STEP0_AUDIT_v1.md` §4 traces it hop by hop into `trial.suggest_int('window_size', …)` |
| **B** — skip bounds | **ADVISORY BY DESIGN, deliberately not applied** | `SESSION_CHANGELOG_20260307_S122.md:56` — *"disabled per TB + S121 shuffle test"* |
| **C** — offset prior | **ADVISORY BY DESIGN**, and inert on live data (`confident: false`) | same ruling; `TRSE_v1_15_SPEC.md:326-328` predicted exactly this fallback |

**Rules B and C are not dropped wires.** `TEAM_ALPHA_TRSE_FIX_PROPOSAL.md` establishes this with
three independent citations. `TRSE_v1_15_SPEC.md:216-240`, which describes them as *applying*
bounds, is **SUPERSEDED**. **DIVERGENT — D13** (§16).

**Step 0's silent failure is architected, not accidental.** `agent_manifests/trse.json` sets
`skip_on_fail: true` with a stated reason, and §2C specifies the passive integration. A Step-0
failure is recorded as `action: proceed` and Step 1 runs with full default bounds.

- **WHY:** `PROJECT_FILE_CATALOG.md` §5.1 item 3; `TRSE_INTEGRATION_PLAN_S121.md` §2C.
- **WHAT:** `agent_manifests/trse.json` — parsed live; it declares **no `actions` key**, so the
  manifest and `STEP_SCRIPTS` cannot be compared for this step.

**GOVERNED issues (cite these, do not re-report):**

- **F1 — WATCHER's Step-0 command is malformed** (three `default_params` keys `trse_step0.py`'s
  argparse does not define; exit 2, masked by `skip_on_fail`). `TRSE_STEP0_AUDIT_v1.md` §8 F1, with
  execution proof. **AWAITING RULING** on `TEAM_ALPHA_TRSE_FIX_PROPOSAL.md`.
- **F2 — self-perpetuating freshness lock**: Step 1 writes back into `trse_context.json`, bumping
  the mtime that is Step 0's freshness sentinel. `TRSE_STEP0_AUDIT_v1.md` §8 F2.
- **F6 / `recommended_window_size`**: the manifest declares `8`, the code reads it into `_rec_ws`
  and never references it; Rule A uses a hardcoded `32`. **Root cause is known and is not "a field
  of unclear purpose"** — `TRSE_INTEGRATION_PLAN_S121.md` §2C specifies `min(rec_ws * 4, …)`, and
  `8 × 4 = 32`. **The value is correct; the wiring is missing.** **DIVERGENT — D14.**
- **Step 0 has no documenting chapter.** `PROJECT_FILE_CATALOG.md` §7 gap 1. Its design authority is
  `TRSE_v1_15_SPEC.md` + `TRSE_INTEGRATION_PLAN_S121.md`.

---

# 3. STEP 1 — Window Optimizer (and the sieve inside it)

**This is the largest and most thoroughly documented step. Chapter 1 is CLOSED and audited.**

**Purpose.** Two jobs in one module set: (a) Bayesian optimization of window parameters with Optuna
TPE, and (b) **survivor generation** — running real bidirectional sieves across the fleet and
accumulating the survivors into a certified artifact.

- **WHY:** `docs/CHAPTER_1_WINDOW_OPTIMIZER.md` §1.1.
- **WHAT:** three load-bearing modules, not two — `window_optimizer.py` (data structures, CLI, both
  entry points), `window_optimizer_integration_final.py` (backend cascade, `run_bidirectional_test`,
  finalizer hand-off), `window_optimizer_bayesian.py` (the entire Optuna search space, study storage,
  warm start). Chapter 1 header table.

## 3.1 Inputs

| input | source | anchor |
|---|---|---|
| the draw dataset | the **pointer manifest**, not a bare path (§13.2) | `agent_manifests/window_optimizer.json` `required_inputs: ["daily3.json"]` (legacy alias, resolved by WATCHER's P0.5 resolver) — parsed live |
| search bounds | `distributed_config.json → search_bounds`, merged **over** code defaults key-by-key | Chapter 1 §4.1; merge at `window_optimizer.py:85-87` (chapter-cited) |
| TRSE regime context | `trse_context.json`, optional | §2 above |
| 40 CLI flags | operator or WATCHER manifest | Chapter 1 §10.1 |

**The bounds authority is `distributed_config.json`, never the chapter.** Chapter 1 §4.1 carries a
machine-generated snapshot with a `repository_commit` and a `configuration_digest`, regenerated by
`scripts/extract_search_bounds_snapshot.py` and explicitly marked *"INFORMATIVE SNAPSHOT — NOT
AUTHORITATIVE"*. Every numeric bound in the pre-correction chapter was wrong; that is why the
snapshot mechanism exists.

## 3.2 The search space — seven sampled dimensions

- **WHY:** Chapter 1 §8.1.1.
- **WHAT:** all seven live inside the `optuna_objective` closure in
  `window_optimizer_bayesian.py` — `window_size`, `offset`, `session_idx`, `skip_min`, `skip_max`,
  `forward_threshold`, `reverse_threshold`. **Cite `optuna_objective` by name**: the block moved ~87
  lines when the sampler-neutral core was extracted and did not change.

The space is **not a rectangle** — `skip_max`'s floor is `max(skip_min, bounds.min_skip_max)`, so
the two skip dimensions are coupled.

## 3.3 The sampler-neutral core, and why `sampler` has no default

`OptunaBayesianSearch.run_optimization(..., *, sampler, sampler_metadata, ...)` — **both are
required and keyword-only, with no default.**

- **WHY:** Chapter 1 §8.1.2, quoting the in-source rationale: so *"a caller cannot get TPE by
  omission and then report the run as something else"*, and because *"an unlabelled run is not a
  control."*
- **WHAT:** `window_optimizer_bayesian.py` `run_optimization` (chapter-cited `:457-827`); the two
  thin wrappers are `OptunaBayesianSearch.search` (TPE) and `OptunaRandomSearch.search`
  (`RandomSampler`, label `optuna_random_control`).

`SAMPLER_ENTRYPOINTS` is **deliberately not wired** to any advisor, WATCHER policy or
`strategy_recommendation.json` — its only consumer is the comparison harness
`tests/phase6/sampler_control_arm.py`. **Autonomous sampler selection is reserved authority (Team
Beta).** A change that lets an advisor pick a sampler is a governance change, not a wiring change.

## 3.4 The threshold authority — one resolver, and why

Per-trial forward/reverse thresholds are resolved **once, in the parent**, by
`resolve_directional_threshold()`.

- **WHY:** Chapter 1 §7.2.1 — the invariant, and the regression history that produced it.
- **WHAT:** `window_optimizer_integration_final.py:214` — **read this session**; the
  `ThresholdResolutionError` it raises is declared at `:210-211`.

Three properties are load-bearing and are visible in the live source:

1. Precedence **explicit > config > default**, resolved once and never reinterpreted downstream.
2. **`is None` is the sole fallback trigger** — the docstring states *"0.0 is a legitimate threshold
   and must never be silently replaced"* and explicitly refuses the `getattr(...) or default` shape.
3. **Fail closed** — *"refusing to invent one"* when nothing resolves.

**GOVERNED history:** the fix landed `3fdf434` (2026-04-30), was **silently reverted** by a
stale-copy overwrite at `2389b61` (2026-07-07) whose commit message never mentions thresholds, and
was repaired for both routes at `8a55a68` (2026-07-31). Every threshold value recorded between those
commits is **non-executed**. Full trace: `docs/THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md`.
*Gate-design consequence, stated in Chapter 1 §7.2.1:* a regression gate on this invariant must
**execute the live call site (AST)**, because a whole-block replacement passes a text anchor.

## 3.5 The backend cascade — four backends, one certifying

`run_bidirectional_test` opens with `getattr`-gated branches, in this order:

| order | backend | gate variable | status |
|---|---|---|---|
| 1 | **RANGE-MINER** | `_use_miner` | **the certifying route** |
| 2 | PWC | `_use_pw` | non-certifying diagnostic; **hybrid quarantined** |
| 3 | ZMQ/SQLite | `_use_zmq` | non-certifying |
| 4 | legacy coordinator | fall-through | the path Chapter 1 §2.1's diagram draws |

- **WHY:** Chapter 1 §11.3; PWC/ZMQ retirement from certifying authority —
  `TEAM_ALPHA_PWC_COMPARATOR_SCOPE_CORRECTION.md` (**RULED**, 2026-07-31).
- **WHAT:** verified live this session — `window_optimizer_integration_final.py:1369`
  (`run_bidirectional_test`), gates at **`:1402`** (`_use_miner`), **`:1535`** (`_use_pw`),
  **`:1580`** (`_use_zmq`). Selection is mutually exclusive, enforced at argparse before the
  coordinator is constructed.

**Consequence for any reader:** *"what does Step 1 do here?"* cannot be answered without first
knowing which backend flag was set. Most production runs do **not** take the path the chapter
diagram draws.

## 3.6 Outputs, and which one actually matters

- **WHY:** Chapter 1 §12.1 — *"The canonical Step-1 → Steps-2–6 carrier is the certified NPZ
  generation."*
- **WHAT (verified this session):** `utils.run_finalizer.finalize_run` is called from
  `window_optimizer_integration_final.py:2966`; the demotion of the JSON is stated in-source at
  `:2993-2995` — *"NO LONGER the canonical Steps 2-6 input… Steps 2-6 consume the canonical NPZ."*

| file | what it really is |
|---|---|
| **certified NPZ generation** | **CANONICAL.** 22 arrays + sidecar, `artifact_sha256`, lineage. Generations **chain**. |
| `optimal_window_config.json` | best parameters + `agent_metadata` |
| `bidirectional_survivors.json` | **post-success SUMMARY — generation IDs and sha256s, no seeds** |
| `forward_survivors.json` / `reverse_survivors.json` | **count-only stubs** |
| `train_history.json` / `holdout_history.json` | 80/20 split of the draw data |

**Why forward/reverse are stubs, and why that is correct.** `[S166-ACCUM]` replaced survivor-object
retention with counters to stop a RAM bomb at 26-GPU scale; only `accumulator['bidirectional']` is
appended to. **Do not "restore" full retention** — the canonical NPZ carries what downstream needs
(Chapter 1 §12.1).

## 3.7 RANGE-MINER — the engine inside Step 1

**Purpose.** Replace the chunk-dispatch backend with **persistent per-GPU daemons** that pull
60M-seed stripes, execute them in sub-stripes sized to VRAM and the watchdog ceiling, and stream
survivors back through the unchanged 22-array contract.

- **WHY it exists:** PWC suffered silent hard resets and `GCVM_L2_PROTECTION_FAULT` on the RX 6600 XT
  rigs at full-fleet saturation, traced to launch-storm behaviour. `Chapter 2 §8.1`;
  `docs/ROCm_Saturation_Report_S172.md` is the measurement behind the pivot.
- **WHY the contract is what it is:** `PROPOSAL_S172_RANGE_MINER_v1_4_4.md` §1, quoting the binding
  TB invariant — *"Range-Miner is allowed to change how Step 1 computes, but not what Step 1
  emits."* **This is an INTERFACE contract, not "match PWC's values"**
  (`TEAM_ALPHA_PWC_COMPARATOR_SCOPE_CORRECTION.md`).
- **WHY the architecture is split the way it is:** `PROPOSAL_S172_RANGE_MINER_v1_4_5.md` §3.A/§3.B
  (absorbing the binding S175 ruling) — Phase 4 owns scheduling + asynchronous remote spool staging
  and **MUST NOT build arrays in the dispatch thread**; Phase 5 owns verification, columnization,
  dedup, ordering, assembly and contract validation.

**WHAT (module roles, verified present this session):**

| module | role |
|---|---|
| `miner/range_miner_protocol.py` | length-prefixed JSON framing, 8 message types |
| `miner/range_miner_coordinator.py` | stripe ledger, admission, staging, lease expiry, retry matrix, `serve_trial` |
| `miner/range_miner_worker.py` | READY handshake, sub-stripe loop, **per-family kernel ABI builders**, residue-window authority |
| `miner/range_miner_npz_writer.py` | Phase-5 assembly, canonical replay, trial assembly |
| `miner/assembly_backends.py` | frozen two-backend interface (`serial_reference` \| `process_sharded`) |
| `miner/dataset_authority.py` | pointer resolution, run-start freeze, per-node provisioning (`POINTER_MANIFEST_NAME` at `:68`, `FrozenDataset` at `:229`, `resolve_pointer` at `:381`, `freeze_run_dataset` at `:594`) |
| `miner/step1_ingress.py` | miner candidates → the Step-1 accumulator (`ingest_assembly:205`, `certified_paths:265`) |

**Seed-domain partitioning.** The **coordinator** partitions into contiguous macro-stripes with no
gap and no overlap; the **worker** partitions its one macro-stripe into GPU-safe sub-stripes at
runtime, with the cap branching on backend (`rocm` → AMD caps, `cuda` → NVIDIA caps). Completion is
**proved, not assumed**: sub-stripes done == expected == distinct sub-indices, seed counts sum,
survivor counts sum, **and the sub-stripe ranges tile the parent exactly**. (Chapter 2 §8.3.)

**Residue-window authority — one derivation, shared.** `load_residue_window()` is used by both parent
and worker, session-filtered, with identity by **content, not pathname**. Its docstring records the
D6 defect it closes and says explicitly: *"Do NOT reintroduce a second session-filter implementation
on either side."* (Chapter 2 §8.4.)

**Assembly.** `serial_reference` is the production default and the **correctness oracle**;
`process_sharded` is implemented, **available and UNPROMOTED**, and parallelises *only* spool-local
validation — the parent alone owns merge, dedup and intersection.

- **WHY unpromoted:** `PROPOSAL_S172_RANGE_MINER_v1_4_5.md` §17 — promotion requires **all four** of
  ≥20% median end-to-end improvement, identical final arrays, ≤50% host-RAM peak RSS, and no swap.
- **GOVERNED:** ~1.6× faster high-survivor at ~2–3× RAM; ~180× slower low-survivor.

**What is certified, and what is not.**

- **CERTIFIED and CLOSED** — bounded Phase 6, `d98298c`, TB ruling 2026-08-02. Wall A (the complete
  consumer chain with **value-by-value** metadata comparison), Wall B (repetition, backend
  equivalence, CUDA/ROCm equivalence, node-assignment independence across two ROCm rig pairs — all
  five arms reproduced `artifact_sha256 0e0092fe…c4b0`), and the Miner Known-Answer Transfer Gate
  (**8/8 populations exact-set equal**; F5–F7 prove reference independence by rejecting three wrong
  semantics). Chapter 2 §11.1.
- **Scope limit, stated explicitly** — Wall A/B used **constant-skip** generations; **hybrid worker
  semantics are covered by the transfer gate, not by a four-phase Wall-A consumer run.** The scratch
  generations are **not** release-grade. Chapter 2 §11.2.

**GOVERNED issue — hybrid skip bounds do not reach the kernel.** See §11.4; it is the single most
consequential open item on this step and it is **already ruled on**.

---

# 4. STEP 2 (executable) — Scorer Meta-Optimizer

**Chapter 3 is AUDITED and NOT CORRECTED. Read `docs/CHAPTER_3_ALIGNMENT_AUDIT.md` before trusting
any specific claim in the chapter — 55 claims, 17 accurate / 9 stale / 24 false / 5 unverifiable.**

**Purpose.** Optimize the **scoring parameters** Step 3 will use, using the sieve's own evidence as
ground truth rather than draw history — restoring a clean separation from Chapter 13's job.

- **WHY:** `docs/CHAPTER_3_SCORER_META_OPTIMIZER.md` §4.2 — *"Step 2: Find optimal SCORING
  PARAMETERS using SIEVE QUALITY as ground truth. Chapter 13: Compare predictions to real draws."*
- **WHAT:** `agents/watcher_agent.py:390` → `run_scorer_meta_optimizer.sh`; workers are
  `generate_scorer_jobs.py` → `scorer_trial_worker.py`, dispatched via `scripts_coordinator.py`.

**Why the pull architecture.** Workers write results **locally**; the coordinator pulls via SCP and
deletes after a verified pull. Direct Optuna workers would produce 26-way DB contention, and shared
storage a file-locking bottleneck. (Chapter 3 §2.2, §3.3 — the audit rates §3's pull architecture
**live and accurate**.)

**Inputs.** Seven arrays, all by name, from the NPZ only: `seeds`, `forward_matches`,
`reverse_matches`, `bidirectional_count`, `intersection_ratio`, `trial_number`, `skip_mode`
(`CHAPTER_3_ALIGNMENT_AUDIT.md` §4).
**Output.** `optimal_scorer_config.json` — **eleven scalar hyperparameters**, nothing per-seed.

**The seam is sound, and this is the strongest available evidence for the interface claim.** All
seven arrays Step 2 reads are in the frozen 22-array contract — verified in the audit against live
`utils.canonical_arrays.CANONICAL_ARRAY_CONTRACT`. **RANGE-MINER's certified artifact satisfies
every column this consumer needs, with nothing missing and nothing extra required.** This consumer
genuinely cannot tell which engine produced the bundle.

**GOVERNED issues — all diagnosed, none authorised for repair:**

| item | authority |
|---|---|
| the live objective is **v4.3** (`0.70·tanh(enrich) + 0.20·coverage − 0.10·size_penalty`), while the chapter, the module docstring and the "TB FORMULA (final v4.2)" block all still advertise v4.2 | `CHAPTER_3_ALIGNMENT_AUDIT.md` F3. **DIVERGENT — D6** |
| the rewrite landed inside a commit about **moving documentation** (`ca975f8`, subject *"chore(S109): move 58 stray docs from root to docs/"*) | F4 — the same mechanism as the threshold regression |
| the objective is **structurally blind to 7 of 11 sampled dimensions** | F6 — *reported as a discovery this would have told Beta about its own ruling* |
| the WSI v4.0 objective **measured itself** (`quality = fwd*rev` dominant at w3≈0.82; WSI = 0.9997 on trial 1); v4.1 then could not optimise because `bidirectional_selectivity` sat at floor with no variance | **`TB_RULING_REQUEST_STEP2_v4_1_OBJECTIVE.md`** → **`TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md`** — **RULED**, lineage v4.1→v4.2→v4.3→v4.4. **★ This is the document Alpha nearly re-reported to Beta as a new finding on 2026-08-02. Read it before reporting any Step-2 objective blindness.** |
| Step 2's fallback path invokes the **TB-prohibited legacy converter** and `mv`s a regular file onto the D3.5 finalizer-owned symlink | F1, F2 — the disposition is Beta's; `TEAM_ALPHA_D3_0_B_AND_ITEM1_NOTICE.md` records **D3.0-B accepted as OPEN** |

**Also:** `STEP_SCRIPTS[2]` (`run_scorer_meta_optimizer.sh`) appears in **no manifest action** —
`scorer_meta.json`'s actions are `generate_scorer_jobs.py` and `scorer_trial_worker.py` (parsed
live). **DIVERGENT — D8.** *"This divergence is how a soak hazard reached launch day"*
(`PROJECT_FILE_CATALOG.md` §5.1).

---

# 5. STEP 3 — Full Scoring

**Chapter 4 is UNAUDITED. Treat its specific numbers as unverified.**

**Purpose.** Regenerate each survivor's PRNG sequence from its seed and extract the per-survivor ML
feature vector, plus the holdout label.

- **WHY:** `docs/CHAPTER_4_FULL_SCORING.md` §1.1, §1.4.
- **WHAT (verified live):** `agents/watcher_agent.py:391` → `run_step3_full_scoring.sh`, which
  invokes `generate_step3_scoring_jobs.py` (`:160`), dispatches through `scripts_coordinator.py`
  (`:226`) and aggregates into `survivors_with_scores.json` (`:306`, `:455`).

**Inputs** (parsed live from `agent_manifests/full_scoring.json`): the 22-array NPZ,
`optimal_scorer_config.json`, `train_history.json`, `holdout_history.json`.
**Output:** `survivors_with_scores.json` + `scoring_statistics.json`.

**The feature contract — 91 extracted / 89 trained.**

- **WHY:** `TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md` §7.1, §8; primary evidence
  `docs/S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md`.
- Composition: **13** merged directly from NPZ arrays · **59** computed in Step 3 from the
  regenerated PRNG sequence vs `train_history` · **14** `global_*` run-context · **5** dead
  placeholders. `91 − score − confidence = 89` trained.
- **Three namespaces, and the distinction is architecturally load-bearing** (system map §8):
  **72 survivor-local** (the legitimate search space), **14 run-global** (identical for every
  survivor in a run — *filtering by one can only retain or remove the whole run*, and random row
  folds across runs **leak run identity**), **5 permanently zero**.

**Why `offset` for the holdout label is a law, not a parameter.**

- **WHY:** Chapter 4 §1.4 and Chapter 6 §3.2.1, quoting Team Beta — *"Remove offset as a choice.
  Keep offset as a law."* `offset = len(train_history)`, because a PRNG is a state machine and the
  holdout is contiguous future data; any manual offset defeats verification.

**The metadata merge — verified live this session.**

- **WHAT:** the batch path merges **18** metadata fields at `survivor_scorer.py:772-782`; the
  sequential fallback merges only **6** at `full_scoring_worker.py:451-455`.
- **GOVERNED:** system map §7.3 defect 2 — on GPU-batch failure seven NPZ-backed features silently
  become `0.0` while the record keeps all 91 keys and downstream schema checks pass. **Team Beta
  classifies this P0** and requires gate **F-PARITY** before fallback output may be used for
  training. **Not a defect of RANGE-MINER and not fixable by it.**

**The most consequential seam fact.** `forward_matches` and `reverse_matches` — **the only
independent per-seed sieve signal** — are transported through the NPZ and the chunk layer and are
**absent from the Step-3 merge list** (verified live at `survivor_scorer.py:772-782`).

- **WHY / ruling:** system map §7.3 defect 3 — *Team Beta: possibly the most consequential finding
  in the whole trace.* **Ruling: do NOT silently add them.** A separately governed feature-schema
  decision is required — **Option A** (add under a new schema version, leakage + redundancy
  analysis, regenerate Step-3 artifacts, retrain everything) or **Option B** (correct the converter
  documentation and record why they must not enter training). **The miner keeps emitting both
  regardless.**

**Also:** `STEP_SCRIPTS[3]` (`run_step3_full_scoring.sh`) appears in no manifest action;
`full_scoring.json`'s three actions name `generate_full_scoring_jobs.py` (dead code per the Jan-23
TB ruling), `full_scoring_worker.py` and `aggregate_scoring_results.py`. **DIVERGENT — D7**, and it
is the identical shape to Step 2's (`PROJECT_FILE_CATALOG.md` §5.1 item 2, §7 gap 2).

---

# 6. STEP 4 — Adaptive Meta-Optimizer

**Chapter 5 is UNAUDITED.**

**Purpose.** A **capacity and architecture planner**: how many survivors the model should train on,
how deep the network should be, how long training should run.

- **WHY:** `docs/CHAPTER_5_ML_ARCHITECTURE_OPTIMIZER_v2.md` §1.1.
- **WHAT:** `agents/watcher_agent.py:392` → `adaptive_meta_optimizer.py`; inputs
  `optimal_window_config.json` + `train_history.json`, output `reinforcement_engine_config.json`
  (parsed live from `agent_manifests/ml_meta.json`).

**Why it is built this way — the design principle is a prohibition.**

> **Step 4 is intentionally NOT data-aware.** (Chapter 5 §1.2)

It derives capacity from **sieve behaviour and history complexity only**. It does not read
`survivors_with_scores.json`, does not inspect `holdout_hits`, does not choose a model type, and does
not touch holdout data — because doing any of those would tune capacity on validation data and
compromise Step 5's evaluation (Chapter 5 §3.2, §3.3). `--survivor-data` and `--holdout-history`
exist for backward compatibility and are **intentionally ignored** (§7.3).

**Weighting:** window-optimizer results 0.60 · lottery-history complexity 0.35 · reinforcement
feedback 0.05 growing to a 0.25 cap on confidence (Chapter 5 §4.1).

**GOVERNED:** Step 4 reads `best_result.bidirectional_count`, `best_result.precision` and
`all_results[].bidirectional_count` from `optimal_window_config.json` — **none of which Step 1
writes** — and silently falls back to hardcoded defaults `{'min':100,'optimal':500,'max':2000}`.
**TB Q4 ruling: PRESERVE the drift. Do NOT add `best_result`/`all_results`/`mod` inside S172** —
that is a separate compatibility-changing patch, and the miner inherits the silent-fallback
behaviour unchanged (`PROPOSAL_S172_RANGE_MINER_v1_4_4.md` §4.4.3, §13 Q4).

---

# 7. STEP 5 — Anti-Overfit Training

**Chapter 6 is UNAUDITED, and its stated target is SUPERSEDED.**

**Purpose.** Train and select an ML model that ranks survivors by *predicted future* quality, with
K-fold cross-validation and explicit overfit detection, across four model families in **isolated
subprocesses**.

- **WHY:** `docs/CHAPTER_6_ANTI_OVERFIT_TRAINING.md` §5, §7.
- **WHAT:** `agents/watcher_agent.py:393` → `meta_prediction_optimizer_anti_overfit.py`; inputs
  `survivors_with_scores.json`, `train_history.json`, `reinforcement_engine_config.json`; outputs
  `models/reinforcement/best_model.*` + `best_model.meta.json` (parsed live from
  `agent_manifests/reinforcement.json`).

**Why subprocess isolation is mandatory, not stylistic.**

- **WHY:** `docs/DESIGN_INVARIANT_GPU_ISOLATION.md` — **MANDATORY, non-negotiable:** GPU-accelerated
  code must never run in the coordinating process when subprocess isolation is in use. Enforced
  since S72. Origin: `PROPOSAL_Multi_Model_Architecture_Addendum_F.md` (OpenCL/CUDA compatibility).

**Why the training target changed, and what it is now.**

- **The original bug:** `y = score`, where `score` *is* `exact_matches / total × 100` — mathematically
  the same quantity as `residue_1000_match_rate`. The model learned a tautology and ignored 60 of 62
  features (Chapter 6 §2.1–§2.3).
- **Chapter 6's fix:** `y = holdout_hits`.
- **SUPERSEDED:** the live target is **`holdout_quality`** (`1cb90aa`;
  `docs/S111_TEAM_BETA_BRIEFING.md`), and R² was abandoned as an objective at 0.000155 — zero
  signal. `PROJECT_FILE_CATALOG.md` §6.1. **DIVERGENT — D5.**
- **`holdout_hits`' governed classification:** **Classification A — authorized offline
  outcome-derived supervised label. Permitted as a training target; forbidden as a filter, weight,
  mask, window or production-time feature** (system map §10), with recorded non-overlapping
  intervals, train-history-only feature generation, and persisted history hashes.

**The sidecar is the seam.** `best_model.meta.json` carries `model_type`, the feature schema and its
hash, metrics, hyperparameters and provenance — and Step 6 validates the hash before predicting
(Chapter 6 §9).

---

# 8. STEP 6 — Prediction Generator

**Chapter 7 is UNAUDITED, and its I/O description does not match the live module.**

**Purpose.** Load the trained model **by sidecar only**, validate the feature schema, rank
survivors, and emit the next-draw prediction with confidence.

- **WHY:** `docs/CHAPTER_7_PREDICTION_GENERATOR.md` §1.1, §3.1.
- **WHAT:** `agents/watcher_agent.py:394` → `prediction_generator.py`.

**Why sidecar-only loading.** *"Model type is determined ONLY from `best_model.meta.json`. File
extensions are NEVER used"* (Chapter 7 §3.1) — and a feature-schema hash mismatch is a **FATAL**
error, because a reordered feature vector produces meaningless predictions from a model that will
not complain (§4.1).

**The abstention gate — verified live.**

- **WHY:** the in-source contract at `prediction_generator.py:470-476` — *"Learning steps declare
  signal quality; execution steps act only on declared usable signals; control agents decide
  recovery."* Step 6 **consumes** `signal_quality` from the sidecar and **does not recompute it**;
  WATCHER decides recovery.
- **WHAT:** `prediction_generator.py:485-510` — reads `signal_quality.prediction_allowed`; when
  false it logs `SIGNAL QUALITY GATE BLOCKED` and returns a skip result rather than predicting.

**Outputs, as built (verified live).** The module writes **`predictions/next_draw_prediction.json`**
(canonical path at `:899-907`, written at `:945-951`) plus a history copy, and builds **one** pool of
`pool_size` (default **20**, `:88-89`).

**DIVERGENT — D3.** Chapter 7 §1.3/§2.1/§6.1 describes outputs `ranked_predictions.json`,
`prediction_pools.json` and three pools named Tight(20)/Balanced(100)/Wide(300). Live:
`agent_manifests/prediction.json` declares `primary_output: predictions/next_draw_prediction.json`
and `outputs: ["predictions/next_draw_prediction.json"]` (parsed live); `ranked_predictions.json`
appears in the tree **only as a reader default** at `chapter_13_orchestrator.py:877`. **Both
readings recorded; neither adjudicated.** This is the same condition system map §4.2 records as
**Blocker 1** to attribution (*reader exists, no writer exists*) — so it is **GOVERNED**, not new.

**Where the 20/100/300 pools and the lift metric actually live.** In two standalone CLIs that are
**not** WATCHER steps:

- **WHAT:** `build_pools.py` — `--pools` default `"20,100,300"` (`:173`), `--out` default
  `prediction_pools.json` (`:174`), reads `results/multi_gpu_analysis_*.json`;
  `evaluate_pools.py:28` computes hit and **lift vs random** (`k/1000` baseline, `:36-40`). Their
  only in-repo caller is `backtest_pools.py:98,120`.
- **WHY they matter:** system map §12 item 3 records *"coverage-at-fixed-pool-size lift is the
  missing keystone"* as a **withdrawn** claim — *"Wrong — already in `evaluate_pools.py`."*
  **Do not propose building a coverage/lift metric. It exists.**

---

# 9. FEEDBACK — Chapter 13, the live loop

**Purpose.** Convert emitted predictions into learning: observe the real draw, measure error, refresh
labels through data accumulation, and re-run the *dynamic* steps.

- **WHY:** `docs/CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md` §1, §3.
- **WHAT:** `chapter_13_orchestrator.py` — `NEW_DRAW_FLAG = "new_draw.flag"` (`:73`), `run_cycle`
  (`:251`), flag check/clear (`:215`, `:235`).

**The core design principle — immutable structure, configurable intelligence.**

> **The LLM cannot** rewrite mathematical logic, invent features, bypass validation, mutate control
> flow or change step ordering. **The LLM can** interpret diagnostics, detect drift, propose
> parameter adjustments, recommend retraining. (Chapter 13 §2, §11.)

Enforced by Pydantic schemas, GBNF grammars, feature-hash validation, sidecar provenance and
manifest-scoped parameters.

**Static vs dynamic steps.** Steps **1, 2, 4** run once and re-run only on regime shift; steps
**3, 5, 6** are the learning loop. *"The system learns by weighting survivors, not by endlessly
searching new ones"* (§3.2). Labels evolve **through data accumulation, not mutation**: a new draw is
appended, Step 3 recomputes `holdout_hits` over expanded history, Step 5 retrains (§9.1–§9.2).

**Triggers — all defensive, verified live.**

- **WHAT:** `chapter_13_triggers.py:60-72` — `TriggerType` = `N_DRAWS`, `CONFIDENCE_DRIFT`,
  `CONSECUTIVE_MISSES`, `HIT_RATE_COLLAPSE`, `REGIME_SHIFT`, `LLM_PROPOSED`, `SELFPLAY_RETRAIN`;
  evaluated at `:251-315` with a fixed priority order at `:339-346`.
- **WHY the gap matters:** system map §6 — every trigger is degradation-based. **There is no
  opportunity trigger** (*"a heuristic shows durable edge — press it"*). The nuance, accepted from
  Michael's correction: the *pipeline* does find working configurations; the *feedback loop* is
  defensive, and discovery operates at config/window level rather than the per-survivor-heuristic
  level attribution would expose.

**The objective the loop writes back.**

- **WHAT (verified live):** `chapter_13_orchestrator.py:306-316` —
  `0.50·hit@20 + 0.30·hit@100 + 0.15·hit@300 + 0.05·pool_coverage`, derived from `best_rank`, written
  as **annotation only** and non-fatal on failure.
- **WHY:** S140b pipeline objective; `PROJECT_FILE_CATALOG.md` §6.1 (**R² is not the objective**).

**Attribution — the exact classification, and why the wording matters.**

> **Implemented, invoked, unreachable, unconsumed.**
> Team Beta: *"Algorithmically implemented, partially integrated, operationally disconnected and not
> yet behaviorally closed."* (system map §4.3)

**Never say "wired"; never say "not implemented". Both are wrong.** `per_survivor_attribution.py` is
real, test-exercised, and called by Chapter 13 with seed identity attached. It cannot execute because
of **four independent blockers** (system map §4.2): no producer for the ranked-prediction artifact;
a wrong sidecar key; Chapter 13 bypassing the canonical NN loader; and NN attribution omitting the
training scaler. **v1.1's "two cheap unblocks" is WITHDRAWN** — TB: *"One-line Chapter 13
feature-name patch alone — Not sufficient; do not land in isolation."* Activation is gated on
**A-NORM** prediction parity (§4.2).

**Selfplay is not a learning system.** It is a **policy-conditioned evaluation harness** and a
*discovery front-end* to an already-built grade→attribute→concentrate→reinforce loop.

- **WHY:** `TB_UPDATE_SELFPLAY_REFRAMING_2026-07-28.md` — **a correction of framing, not of
  architecture**, issued deliberately *before* REV2.1 was drafted. It **governs how selfplay may be
  described.** Authority boundaries: `CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md` (**RATIFIED**).
- **WHAT:** `chapter_13_acceptance.py:224` (`SelfplayCandidate`); `promote_candidate` at `:818`.
- **GOVERNED:** the promotion seam is broken (`SelfplayCandidate` lacks `transforms`) and
  `propose_transform_update` is a no-op.

---

# 10. CROSS-CUTTING: the bidirectional sieve

## 10.1 Why bidirectional — the mathematics

For an incorrect seed, forward and reverse survival are approximately independent, so

```
P(survive both) ≈ P(survive forward)²        e^(−cn)  →  e^(−2cn)
```

**Bidirectional squares the exponent — a catastrophic collapse of noise.** At exact match with
n = 50, false survival ≈ 10⁻³⁰⁰.

- **WHY:** `docs/BIDIRECTIONAL_SIEVE_MATHEMATICAL_WHITEPAPER.md` §3–§6 (read in full this session).
- **WHY it matters strategically:** *survivor validity rests on **sieve selectivity**, not on search
  extent* — this is the argument that makes the `high16 = 0` stratum an honest-labelling problem
  rather than a coverage problem (§15.3).

## 10.2 What "reverse" means — the most misread fact in the project

> **"Reverse" refers to the ORDER of the target data, NOT to inverting the PRNG.**

- **WHY:** Chapter 2 §3.2, confirmed against live source in that pass.
- **WHAT:** the host reverses the residues (`residues[::-1] if reverse else residues`,
  `miner/range_miner_worker.py:888`); the reverse kernel iterates the generator **forward**, step for
  step identical to the forward kernel (`prng_registry.py:3143` vs `:982`); direction is a **name
  test** (`family_name.endswith("_reverse")`, `miner/range_miner_worker.py:116`).
- **There is no modular inverse and no backward recurrence anywhere in the tree.** Most PRNGs are not
  invertible without full state. **Do not "fix" this** — it is on the looks-like-a-bug-isn't list.

## 10.3 What the kernel actually computes

Not a set intersection per draw. Each thread owns one seed, walks the window once per skip
hypothesis, and keeps a **match rate**:

- **WHY:** Chapter 2 §2.2, which explicitly corrects the recovered set-intersection pseudocode.
- Four properties that matter downstream: (1) **the output is a rate, not a boolean** — the sieve is
  a *scorer with a threshold*; (2) **skip is maximised over**, so a survivor is a **(seed,
  skip-hypothesis) pair**; (3) ties resolve to the **lowest** skip (`rate > best_rate`, strictly
  greater); (4) the rate is **float32** compared against a float32 threshold — doing it in float64
  puts boundary survivors on the wrong side of `>=`.

## 10.4 The intersection, and what a survivor means

`bidirectional_survivors = forward_survivors ∩ reverse_survivors`. **There is no joint gate, no
re-verification of the surviving pair, and no combined-rate threshold** — two independent scored runs
whose seed sets are intersected. `intersection_count` duplicating `bidirectional_count` is
**deliberate**.

> **A survivor is a scored candidate, not a verdict.** (Chapter 2 §4.3)

Survivors may be the true seed, one of several true seeds, a partial match valid before a reseed, or
a near-consistent neighbour **admitted on purpose**. **Deciding which is Step 3's and Step 5's job,
not Step 2's.** The recovered chapter's *"survivors are NOT false positives"* was **corrected** — it
is true only at τ = 1, the regime the system deliberately avoids.

## 10.5 Why thresholds are loose — and must stay tunable

> Exact sieves eliminate *all* variance. Survivors = {s\*}. No ranking, no gradients, **no learning
> signal.**

- **WHY:** whitepaper §7 (the counter-intuitive section that gets misread every time), restated at
  Chapter 1 §4.3 and Chapter 2 §1.3.
- Loose thresholds deliberately admit a **manifold** of near-consistent seeds sharing structured
  deviations that ML can learn to rank. **This is a mathematical necessity, not sloppiness** — which
  is why thresholds are Optuna-tuned per direction and why a threshold silently pinned to a constant
  is a serious defect (§3.4).

## 10.6 The three-lane CRT test

Every kernel but one evaluates `output % 1000 == residues[i] % 1000 && % 8 && % 125`.

- **WHY:** Chapter 2 §6 — the **only prose explanation in the project's history**, undocumented from
  `248e48c` until the restoration.
- **§6.3 proves the three lanes are exactly equivalent to the mod-1000 test alone.** Since
  1000 = 8 × 125 with gcd = 1, CRT makes the other two conjuncts **implied**. Verified two ways in
  that pass, including an exhaustive check with zero differing cases.
- **§6.4's "triple validation ≈ 10⁻⁷ per draw" is CORRECTED** — the per-draw false-positive rate is
  exactly 1/1000, and the test does **not** require a full 32-bit state match. Filtering power comes
  from sequential accumulation and bidirectional intersection, not the lane decomposition.
- **The lanes are NOT to be removed** (Chapter 2 §6.5): the transcription is deliberate on the
  known-answer reference side and would have to change in lockstep under a gate; and the redundancy
  may encode an intended lane-parallel CRT architecture that was never built (**F-3, open question**).
- **The count is 43, in 43 of 44 kernels**, with `mt19937_hybrid_multi_strategy_sieve` the single-lane
  exception. The earlier "39" is **withdrawn as unreproducible**; §6.2.1 publishes the counting
  method as executable code so the figure is re-derivable. Verified live this session:
  `KERNEL_REGISTRY` has **44** entries over **11** base families.

---

# 11. CROSS-CUTTING: skip

**§0.4 of the project-facts skill exists because of this parameter. Read §11.1 before forming any
opinion about whether `skip_min`/`skip_max` should exist.**

## 11.1 Why skip exists — the physical model

**The published draw sequence is not an uninterrupted PRNG output stream.**

Per the *California State Lottery Daily & SuperLotto Plus Draw Procedures* (eff. 2021-06-09) —
**citation `UNAVAILABLE`; the PDF is not in the repo and was not read this session**:

1. **One automatic pre-test session runs before an automatic Daily draw**; additional pre-test draws
   run **only when an anomaly requires them**. Pre-test outputs are generated, verified, certified —
   and **never published**.
2. **Draw equipment is selected per session** by an auditor-verified RNG program. Midday and evening
   are separate sessions with separate equipment selection.
3. The evening session draws **Daily 3, Daily 4, Fantasy 5 and Daily Derby together** — other games'
   outputs sit between the Daily 3 values an observer can see.

- **WHY:** `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` §5.1 (written there for the first time) and
  Chapter 1 §3.1.
- **Corrected 2026-08-01:** the earlier *"two pre-test draws before every live draw"* was an Alpha
  misreading — the "two test draws" language applies to **manual SuperLotto Plus equipment**. **Only
  the count was wrong.** The claim had propagated into two chapters, a source map, the skill and
  three Beta submissions.
- **What the procedures establish and what they do not** (Chapter 2 §5.1): they establish equipment
  selection, an unpublished pre-test, and co-drawn evening games — outputs **consumed and not
  published** between observable values. They do **not** establish that every omitted output belongs
  to one uninterrupted PRNG state stream. **These are physically motivated *candidate gaps*
  supporting skip as a detector — not proven state advances.**

**Therefore the observable sequence has real, structural gaps of unknown and varying size. Skip
models those gaps. It is a physical property of the data source, not a tuning convenience.**

## 11.2 Why this paragraph is in the chapter at all

In one session **Team Alpha, Team Beta and Claude Code independently recommended deleting
`skip_min`/`skip_max`** — a cornerstone of the design. All three inferred intent from the current
hybrid kernel signatures, which are themselves the defect. **They were wrong because the document
explaining why skip exists did not exist to be read.** Michael stopped it. (Chapter 2 §5.1.1.)

> **Standing rule: the fix is WIRE-IN, not removal. Absence of a working implementation is not
> evidence of absent intent.**

## 11.3 What the hybrid kernel actually does — correcting a common misreading

- **WHY:** Chapter 2 §5.3.
- **No pattern is supplied and none is generated.** `skip_sequences` is an **output**.
- It runs a **greedy per-draw adaptive search**: from a running estimate `expected_skip`, try every
  stride in `[expected_skip − tolerance, expected_skip + tolerance]`, and on a hit **re-centre the
  estimate on the stride that hit**.
- `expected_skip` is **hardcoded to 5**; the ancestor file still carries `// Initial guess` — **a
  guess, not a constant**.
- `strategy_tolerances` is the **half-width of the per-draw matching window**, not a generation
  parameter. **No coherence scoring exists** — the only score is `match_rate`.
- Forward hybrid scans all strategies and keeps the best; **reverse hybrid returns on the first
  strategy clearing the threshold** and does not maximise at all.
- The five documented strategies are **parameterisations of that one algorithm**, not distinct
  algorithms.

## 11.4 The defect — GOVERNED, OPEN, and already ruled on

**Constant kernels receive the sampled bounds; hybrid kernels do not.**

- **WHAT (verified live this session):** `miner/range_miner_worker.py` — `_constant_prefix` (`:160-174`)
  emits `ScalarArg(ctx.skip_min)` and `ScalarArg(ctx.skip_max)`; `_hybrid_prefix` (`:177-193`) returns
  **13 elements, none of which is a skip bound**. `_offset_tail` (`:196-197`) and
  `_reverse_hybrid_tail` (`:200-202`) add only `offset`. **The asymmetry is in the argument builder,
  not in the payload.** The legacy PWC route has the same shape —
  `sieve_gpu_worker.py:217-302` shows seven `family_name` branches, with the hybrid branch at `:232`
  rebuilding `kernel_args` from scratch.
- The sampled bounds survive **eight hops** — argparse, config, coordinator, ledger, manifest,
  payload, worker unpack, `BuildContext` — and **die one call before launch**.
- **WHY / governance:** Chapter 2 §5.4 and Chapter 1 §3.1.1 (dead dimensions **D-1/D-2**);
  `docs/HYBRID_SKIP_BOUND_AUDIT.md` for the wiring trace. **Consequence: hybrid optimization results
  are non-certifying.** Constant-skip Optuna exploration **may resume**; hybrid exploration is
  **non-certifying only**.
- **★ The "semantics are unspecified" premise is FALSE.** `HYBRID_SKIP_BOUND_AUDIT.md:318` recorded
  them as unspecified; `docs/SKIP_SEMANTICS_SEARCH_v1.md` **FOUND** them in two committed documents.
  This was the fourth falsified absence claim of that session, and the sharpest — the audit's own
  VIR-6 declared a full-tree grep that **reached the exact line and did not read it**. **Read
  `HYBRID_SKIP_BOUND_AUDIT.md` for the wiring trace, never for the semantics verdict.**

## 11.5 Two readings, two stages — not a contradiction

| stage | reading | source |
|---|---|---|
| **input** (Step 1 → 2) | *"Minimum/Maximum skip value **in pattern**"* — an element-wise bound on the discovered sequence; documented hybrid default `[0, 16]` | `docs/instructions.txt:1182-1183` |
| **output** (Step 2 → 3) | *"Minimum/Maximum gap that **worked**"* — an ML feature; *"Tight skip range = stronger hypothesis"* | `docs/PROPOSAL_ML_Architecture_Remediation_v2_0.md:150-158` |

**The two readings have very different costs, and conflating them is what makes the
sampler-comparison sequencing error possible** (Chapter 2 §5.7, §11.4):

- The **output** reading needs **no kernel change at all**. It is blocked only by the host discarding
  `skip_sequences` — **verified live this session** at
  `window_optimizer_integration_final.py:125` (`extract_survivor_records` reduces each survivor to
  `{'seed', 'match_rate'}`). That single discard is what kills the three dead features `skip_mean`,
  `skip_std`, `skip_entropy`, whose producer **exists on the GPU** and whose Oct-2025 ancestor spec is
  `pattern_stats` (`docs/instructions.txt:1230-1245`). **APPROVED work**
  (`TEAM_ALPHA_SKIP_SEMANTICS_SUBMISSION.md`).
- The **input** reading is §11.4 and needs the hybrid ABI wired. **Still open.**

**Two registries currently disagree** about which reading is authoritative:
`config_manifests/feature_registry.json` says "found during" (output);
`config_manifests/parameter_registry.json:160,166` says "for sieve search" (input). **DIVERGENT —
D12**; correcting it belongs to whichever change settles the semantics.

## 11.6 Design intent — the fingerprint framing

> **The goal was never to reverse state. It is to extract a fingerprint.**

Variable skip exists to **find the windows where coherent skip structure surfaces** — the fingerprint
glimpse — and to produce survivors with *varied* skip structure so tree and neural models have
something to rank on. **Variable skip is a detector, not a fitting procedure.**

- **WHY:** Chapter 2 §5.6, which is explicit about its own epistemic status: this records **Michael's
  governing design doctrine, accepted by Team Beta**, and is **not** a historically discovered
  repository statement. The corroboration table shows artifacts *consistent with* the doctrine; the
  NOT-FOUND row for the framing itself **stands and stays**.

## 11.7 The sequencing correction Beta issued against Alpha

The certifying four-phase **TPE-vs-random sampler comparison cannot be scheduled merely "after the
skip-output work."** The approved skip-output work restores the three dead features; **it does not
connect `skip_min`/`skip_max` to the hybrid kernels.** The comparison must wait until **either**
hybrid search-input bounds have defined effective semantics, **or** the comparison uses an explicitly
phase-aware search space that does not pretend dead hybrid dimensions are active. (Chapter 2 §11.4.)

---

# 12. CROSS-CUTTING: survivor manifold → ML → prediction

The chain, with the governed condition of each link:

```
sieve (loose thresholds)  →  a MANIFOLD of near-consistent seeds, by design
      ↓  22-array NPZ — only 4 columns carry per-seed information:
         seeds · forward_matches · reverse_matches · score
Step 3  →  91 features/seed  (72 survivor-local · 14 run-global · 5 dead)
      ↓  survivors_with_scores.json
Step 5  →  model trained on 89, target = holdout_quality
      ↓  best_model.* + sidecar (schema hash, signal_quality)
Step 6  →  ranked survivors → pool → next_draw_prediction.json, or ABSTAIN
      ↓
Ch13    →  grade against the real draw → downstream score → retrain trigger
```

- **WHY the manifold is the point:** whitepaper §7–§8 — ML operates in a **survival-conditioned
  high-signal posterior**, not raw PRNG space. *"ML does not guess. It refines a space already
  reduced from 2³² to 10⁴."*
- **WHY only 4 columns carry per-seed information:** Chapter 2 §8.6; system map §7.1 (`seeds` is
  identity **and** the PRNG-regeneration key that drives all 59 computed features; 13 are merged as
  features; 8 are carried but unconsumed).
- **GOVERNED breaks in the chain**, all with rulings, none to be re-reported:
  `forward_matches`/`reverse_matches` never reach the model (§5); the three skip-shape features are
  dead because the host discards `skip_sequences` (§11.5); attribution is unreachable (§9); the
  sequential-fallback zero-fill is P0 (§5).
- **Governance on the global namespace:** globals *"must not be searchable as ordinary per-survivor
  filter thresholds"*, and folds over multi-run datasets must be **grouped or temporally separated by
  run** (system map §8.2).
- **Static feature registries are no longer authoritative** — `feature_importance.py:95-119` omits 31
  live features and the same stale list is duplicated in `feature_drift_tracker.py`. Authority order
  is **model sidecar → versioned extraction manifest → runtime validation**, and **no diagnostic
  module may maintain a manually duplicated feature-name list** (system map §9).

---

# 13. CROSS-CUTTING: WATCHER's autonomy model

## 13.1 What WATCHER decides alone, and what it escalates

- **WHY:** `docs/CHAPTER_12_WATCHER_AGENT.md` §3.3, §8; `docs/WATCHER_POLICIES_REFERENCE.md` (**the
  canonical meaning of every flag in `watcher_policies.json`**).

| decision | authority |
|---|---|
| PROCEED (confidence ≥ 0.70) | WATCHER alone |
| RETRY (0.50–0.70) | WATCHER alone, bounded by `max_retries_per_step` = 3 and `max_total_retries` = 10 |
| ESCALATE (< 0.50, or retries exhausted) | **halts; human review required** |
| approve a Chapter-13 retrain request | **human, unless BOTH `test_mode` and `auto_approve_in_test_mode` are true** |
| apply an LLM parameter proposal | **filtered at the step boundary — deliberate** |
| select a sampler / a sieve strategy | **reserved authority — never WATCHER** |

**The five safety invariants** (`WATCHER_POLICIES_REFERENCE.md` §Safety Invariants) are worth
memorising:

1. **`test_mode=false` overrides everything.**
2. **Auto-approve requires BOTH flags** — either alone does nothing.
3. **`approval_route` is governance** — it decides *who executes*, not *whether to approve*.
4. **Invalid enum values fail safe** — an unknown `approval_route` reverts to `"orchestrator"`.
5. **WATCHER never mutates policies** — only humans change `watcher_policies.json`.

**The selfplay authorization invariants** (Chapter 12 §11.5): *WATCHER authorizes, does not learn* ·
*selfplay cannot self-dispatch* · *requests are append-only* · *cooldown enforced*.

## 13.2 The control chains — which knobs actually reach execution

This table exists so a wiring gap is found now rather than at Chapter 13. **Every row is governed.**

| chain | state |
|---|---|
| per-direction thresholds → kernel | **WORKS** (D6 + `8a55a68`) |
| dataset identity → all nodes | **WORKS** (P0.5 + Q2 closure `8600e75`) |
| fleet definition → the run | **WORKS** — Resolved Execution Set `63e627f`, admission bound `eff6616` |
| worker loss → failure matrix | **WORKS** (`ee0db06`) — was an unbounded hang |
| Advisor → selfplay `max_episodes`, `min_fitness_threshold` | **WORKS** |
| Optuna `skip_min`/`skip_max` → hybrid kernel | dies at `_hybrid_prefix` (§11.4) |
| Optuna `offset` → forward hybrid | dies in kernel args (§16 D11) |
| `skip_learning_rate` → kernel | kernel hard-adapts at 1.0 |
| Advisor → `strategy_recommendation.json` → WATCHER | **no code reads the file**; the working path is in-memory |
| diagnostics → Step-5 retry params | filtered at the step boundary — **deliberate**; reporting fixed `f8b751c` |
| Ch13 proposal → acceptance | `pending_approval` is a **valid authority boundary** |
| GPU `skip_sequences` → ML features | discarded at `…final.py:125`; kills 3 features |

**Reserved authority (human only):** feature engineering · survivor thresholds · sieve
strategy/mathematics · window-optimizer logic · PRNG-family authority · scoring logic ·
meta-optimizer search space · model families · policy authority.

**Two of these are NOT defects and must not be reported as such** — `TEAM_ALPHA_AUTONOMY_CONTROL_SURFACE_SUBMISSION.md`
opens with Alpha correcting its own reporting, and Beta upheld Chain D's `pending_approval` as a
valid authority boundary and the Step-5 `allowed_params` filter as a deliberate executable-interface
boundary.

## 13.3 Dataset authority — the run-start freeze

- **WHY:** `TEAM_ALPHA_DATASET_LIFECYCLE_FINDINGS.md` (**a fixed `expected_sha256` cannot work** — the
  invariant is **fleet consistency, not immutability**); `RUNTIME_DATASET_PROVISIONING_CONTRACT.md` +
  `PROVISIONING_CONTRACT_AMENDMENT.md`; `DATASET_PUBLICATION_SCHEMA_v1.md` (**FROZEN**; where a brief
  and the schema differ, **the schema wins**).
- **WHAT:** `miner/dataset_authority.py:68` (`POINTER_MANIFEST_NAME = "daily3_current.json"`), `:229`
  (`FrozenDataset`), `:381` (`resolve_pointer`), `:594` (`freeze_run_dataset`).
- **The pointer manifest is authoritative**; `daily3.json` is now a **legacy compatibility alias**.
  Identity is resolved **once at run start and frozen** — a pointer moving mid-run cannot alter a run
  in progress. Dispatch uses the **absolute immutable path**, never the alias, and fails **before the
  first worker dispatch**.
- **The status vocabulary is load-bearing:** **`UNAVAILABLE`** = a required verification was
  *attempted and could not complete* → fatal for a miner topology. **`NOT_APPLICABLE`** = this path
  never needed the check → proceed. *"We needed it and could not get it" is not "we did not need
  it."*
- **`remote_execution=False` is a topology statement, NOT a bypass.**

**Session separation is normative** (TB rulings 2026-07-30/31): midday and evening use
**independently selected equipment**, so there is **no evidentiary basis for advancing one PRNG state
through interleaved records**. Ordering is normative *within a session stream*; combined-container
order carries **no PRNG-advance meaning**. Production re-optimization is **per-session**;
combined-session sequential sieving is **non-certifying and prohibited by default**. The
chronological-reorder migration was **cancelled**.

## 13.4 Fleet authority — the Resolved Execution Set

- **WHY:** `docs/FLEET_STATE_REQUIREMENTS_v1.md` — six checks at three granularities over two address
  sets, and **none of them defines the fleet**. Beta's ruling: the sole future authority is a
  **frozen, run-scoped Resolved Execution Set**, created after backend and rig-profile selection but
  **before** dataset verification, GPU verification, coordinator construction and dispatch. All six
  existing mechanisms become **consumers**.
- **WHAT:** `execution_set.py:161` (`ResolvedExecutionSet`); the admission formula is stated in-source
  at `:176` — `admission_count = min(requested worker pool size, count of selected worker
  identities)` — with **both** numbers recorded (`requested_admission_count` at `:198`,
  `admission_count` at `:189`) and both in `set_id`. *A clamp that overwrites the request is a clamp
  nobody can audit.*
- **A partial set must be explicit and frozen before the run — never inferred from which workers
  happened to answer.** Declared at the CLI via `--execution-set-nodes`; the topology chosen by
  `--rig-profile`.
- **RETRACTION worth knowing:** Alpha claimed the freeze-after-read ordering could not be violated;
  **Beta's refutation was correct** — the read counter was incremented only inside
  `if _ACTIVE is not None`, so a consumer could read `None`, take the legacy path, and a freeze could
  still follow. *The empty read is not the harmless case; it is THE case that matters.* Counter now
  unconditional (`eff6616`).

## 13.5 Admission liveness — why `serve_timeout` stays `None`

- **WHY:** `TEAM_ALPHA_FLEET_STATE_SUBMISSION.md` → repaired `ee0db06`. `assign_stripes`,
  `_dispatch_pending`, `process_lease_expiry` **and** the stage advance were all behind one
  `len(eligible) >= expected_workers` guard while `serve_timeout` is `None` by design. A worker loss
  crossing the threshold stopped lease expiry being processed: **the trial neither completed nor
  failed.** *The Blocker-3 matrix was unreachable in exactly the situation it exists for.*
- **The repair separates admission from maintenance.** **ADMISSION is bounded**
  (`worker_admission_timeout`, default 180 s; the window re-arms only when `stage_idx` changes, so
  worker churn cannot extend it). **MAINTENANCE is unbounded** — once a stage is assigned, dispatch,
  lease expiry and completion evaluation run regardless of the current eligible count.
- **`serve_timeout` stays `None` deliberately** — a multi-billion-seed scan exceeds any wall clock,
  and the bounded clock belongs on *admission only*.

---

# 14. CROSS-CUTTING: the frozen authorities

**Importing these is mandatory; forking them is the defect they exist to prevent.**
All anchors below were **read on VM101 this session**.

| authority | anchor | what it guarantees, and why reimplementing it is forbidden |
|---|---|---|
| **`_l2_sort_key`** | `utils/run_finalizer.py:690` | The frozen L2 key (Ruling D): highest **float32** score → lowest `trial_number` → constant-before-variable **as a within-trial tiebreak only**. The in-source docstring states the reason: *"Comparing pre-rounding Python floats while storing the rounded value is the defect this converts away."* Two floats differing only beyond float32 precision are an **exact tie** and must fall through to the trial tiebreak. A fork that compares float64 silently reorders winners. |
| **`_select_l2_winners`** | `utils/run_finalizer.py:714` | Exactly one record per seed, **independent of input order** — because within one seed the key is a *strict total order*: the only possible three-way tie (same trial, same mode) is rejected outright as accumulator corruption (`AccumulatorConsistencyError`). Its appearance means the accumulator was fed twice. |
| **`CANONICAL_ARRAY_CONTRACT`** | `utils/canonical_arrays.py:99-126` | The frozen **22-array** NPZ schema — exact names, exact order, exact dtypes. This is the interface that makes *"downstream cannot tell which engine produced the survivors"* true. Expanding RANGE-MINER beyond 22 arrays is **not authorized** (system map §14). |
| **`CANONICAL_RECORD_FIELDS`** | `utils/canonical_arrays.py:143` **and** `utils/canonical_records.py:115` | The **24-field** canonical record. 24 − 2 = 22: `sessions` and `prng_base` do not become arrays (they are validated anyway); the two match-rate fields are **renamed**. **Defined in two modules — check which one your consumer imports.** |
| **`canonical_map_hash`** | `utils/run_finalizer.py:486` | The map-identity anchor carried through the generation chain; a link whose map identity differs breaks the chain. |
| **`utils/prng_encoding.py`** | `:37` (`SKIP_MODE_ENCODING`), `:43` (`PRNG_TYPE_ENCODING`), `:54`/`:76` (encode/decode) | The single registry-derived PRNG-type encoding (Phase 0, `2389b61`). **Verified live: 44 entries, matching the 44-entry `KERNEL_REGISTRY`.** Both encode and decode **raise** on unknown values — the pre-Phase-0 code silently collapsed every `*_hybrid` to `0` and decoded it back as `'java_lcg'`, destroying hybrid provenance between Step 1 and Step 3 (TB Q1, `PROPOSAL_S172_RANGE_MINER_v1_4_4.md` §5.4). |
| **The finalizer validators** | `utils/run_finalizer.py:522, 558, 585, 634, 665, 884, 1004, 1069, 1113, 1176` | Ten `_validate_*` functions including `_validate_current_pointer` (`:1113`) and `_validate_chain` (`:1176`). They enforce declared coverage, candidate coverage, run identity, candidate identity, raw candidates, sidecar payload, prior numeric domains, prior identity, the current pointer, and the chain. Generations **chain**; input identity is a lineage invariant, not annotation. |
| **D3.5 finalizer-owned root symlinks** | `utils/run_finalizer.py:1400-1418` | `bidirectional_survivors_all.npz` and `bidirectional_survivors_binary.npz` are **symlinks the finalizer owns**. A regular file appearing there raises `PublicationError` with the reason in-source: *"the historical root artifacts were removed under Ruling F, so something wrote outside the finalizer — failing closed rather than replacing it."* A wrong-target alias also fails closed. **This is what made the briefed D6.1 in-place flush repair unsafe.** |

**Two naming traps.**

1. **`EXPECTED_NPZ_KEYS` does not exist as a symbol.** `CLAUDE.md` §6 Phase 5 names it and
   `PROPOSAL_S172_RANGE_MINER_v1_4_4.md` §12.1 specifies it as pseudocode, but in the tree it appears
   only as a forbidden-token string in a Phase-4 test. The contract wall lives under the
   `utils/canonical_arrays.py` / `utils/canonical_records.py` names. **Do not write code or docs
   around a symbol that does not exist.** (Chapter 2 §8.6.)
2. **The D5 §6.7.A compressed-artifact ban is scoped to worker *transport* artifacts.** The D6.1
   **checkpoint may be compressed** — deliberately separate. **Do not harmonize them.**

---

# 15. WHAT THE PROJECT CLAIMS, PRECISELY

Recorded here because it is repeatedly restated imprecisely.

## 15.1 TFM is functional mimicry — black-box, not state recovery

**TFM = Triangulated Functional Mimicry: functional mimicry of PRNG surface output. It is NOT seed
recovery and NOT state reconstruction.**

- **WHY:** `CLAUDE.md` header; Chapter 2 §0.4; `PROPOSAL_Documentation_Paradigm_Correction_v1_2.md`
  (the origin of the naming rule); `SOAK_TEST_HANDOFF_PROMPT.md` — *"This is NOT specifically a
  lottery system"*, PRNG-agnostic by design, all generator behaviour abstracted via
  `prng_registry.py`.
- The word "lottery" is not used in project language; always "TFM".
- Chapter 2 §5.6 makes the distinction load-bearing rather than cosmetic: **variable skip is a
  detector looking for coherent windows, not a fitting procedure recovering generator state.**

**DIVERGENT — D16, recorded not fixed.** `docs/proposals/README.md:1` is titled *"Distributed PRNG
Analysis & **Seed Reconstruction** System"*, and `docs/README.md:11` / `README.md:4` open with
*"Reverse-engineer PRNG behavior"*. Verified live this session by `/bin/grep`. The catalog already
flags this (§6.4) as **noted for Alpha — not fixed here.**

## 15.2 The falsification criterion

**`holdout_hits` is the designated falsification criterion**, and its status is precisely defined:

- **Classification A — an authorized offline outcome-derived supervised label. Permitted as a
  training target; forbidden as a filter, weight, mask, window or production-time feature** (system
  map §10).
- **It is computed on data never seen during sieving**, at `offset = len(train_history)`, and is
  **not** one of the 91 model inputs (Chapter 4 §1.4, §9.7; Chapter 6 §3.2.1).

**What would falsify the thesis.** The thesis is that survivors carry learnable structure predictive
of *future* draws. It is falsified if, with the pipeline correctly wired:

1. `holdout_hits` shows **no measurable lift over the random baseline** — `k/1000` for a pool of size
   `k` (`evaluate_pools.py:36-40`). Chapter 13 §6.1 states the synthetic-mode targets: Hit@20 > 5%
   against 0.1% random, confidence correlation > 0.3, non-decreasing hit-rate trend, and calls this
   *"a quantitative pass/fail test for functional mimicry quality."*
2. The known-answer control fails — with `TRUE_SEED` planted, seed 12345 must reach
   `holdout_hits = 1.0` while all others sit at ≈ 0.001 (Chapter 6 §3.7).
3. Feature importance stays concentrated in the circular pair (`residue_1000_match_rate` +
   `exact_matches` ≈ 100%), which is the *tautology* signature rather than learning (Chapter 6 §2.2,
   §3.6).

**Caveat, stated rather than omitted:** the objective lineage is `score` → `holdout_hits` →
**`holdout_quality`**, and R² was abandoned at 0.000155 — zero signal, and **R² is not the
objective**. A falsification test must be run against the live target, not the chapter's.

## 15.3 Seed-Domain v1.1 — the honest-stratum resolution

- **WHY:** `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3_5_B.md` §0 (read this session).
- The `java_lcg` family has a **48-bit** internal state; the canonical artifact stores `seeds` as
  **uint32**. The sweep therefore covers the **`high16 = 0` stratum — 1 part in 65,536** of the state
  space. **The upper 16 bits are not invisible**: they are blind to the mod-8 lane but fully visible
  to mod-125, and at TFM's window all 65,536 high-state classes produce distinct draw sequences. **No
  reduction exists.**
- **Why honest labelling rather than a uint64 migration:**

  > *"This is a labelling problem, not a storage problem. TFM does functional mimicry, not state
  > reversal: the sweep exists to discover bidirectional survivors whose structure feeds the ML
  > ensemble, and **survivor validity rests on sieve selectivity rather than search extent**. So the
  > artifact stays `uint32` and declares honestly which stratum it is."*

- The resolution adds **nine frozen sidecar fields** — `seed_semantics = "internal_state"`,
  `seed_storage_dtype = "uint32"`, `seed_effective_bits = 32`, `seed_high16_prefix = 0`,
  `seed_domain_contract = "v1.1-stratum"`, `seed_domain_start = 0`,
  `seed_domain_end_exclusive = 4294967296`, `exhaustive_over = "high16=0 stratum only"`,
  `external_seed_transform = null` — separating three concepts the artifact previously conflated:
  **canonical PRNG coordinate (48-bit state) · stored artifact coordinate (uint32 low component) ·
  certified search stratum (high16 = 0)**. Every one is a fixed constant; any other value **fails
  closed**.

## 15.4 Sieve selectivity, not search extent

This is the load-bearing claim that ties §15.3 to §10.1. A survivor's validity comes from the
mathematics of bidirectional survival — `e^(−cn) → e^(−2cn)` — **not** from having searched a large
fraction of the domain. That is why a 1-in-65,536 stratum is a labelling obligation rather than a
validity problem, and why *"the sweep exists to discover bidirectional survivors whose structure
feeds the ML ensemble."*

---

# 16. DIVERGENT REGISTER

**Both readings recorded. Nothing adjudicated. Nothing changed.**

| # | documentation says | code / live state says | anchors |
|---|---|---|---|
| **D1** | conceptual scheme: sieve = "Step 2" (whitepaper, system map §1, chapter titles) | executable scheme: sieve runs inside Step 1; Step 2 = Scorer Meta-Optimizer | `agents/watcher_agent.py:386-416`; the mapping is documented nowhere — `CHAPTER_3_ALIGNMENT_AUDIT.md` §2 |
| **D2** | Chapter 12 §3.4.1: `STEP_SCRIPTS[3]="generate_full_scoring_jobs.py"`, `[6]="reinforcement_engine.py"`, no step 0 | live: `[3]="run_step3_full_scoring.sh"`, `[6]="prediction_generator.py"`, `[0]="trse_step0.py"` | `agents/watcher_agent.py:386-394` |
| **D3** | Chapter 7 §1.3/§2.1/§6.1: outputs `ranked_predictions.json` + `prediction_pools.json`, pools Tight/Balanced/Wide | live: one pool (`pool_size` default 20), output `predictions/next_draw_prediction.json`; no writer for `ranked_predictions.json` | `prediction_generator.py:88, 899-907, 945-951`; `agent_manifests/prediction.json`; `chapter_13_orchestrator.py:877`. **GOVERNED** as system-map Blocker 1 |
| **D4** | Chapter 4 §9.8: "47 features"; Chapter 6 §11.2: "62 features" | **91 extracted / 89 trained** | system map §7.2; `PROJECT_FILE_CATALOG.md` §6.1 |
| **D5** | Chapter 6 §3.3: `y = holdout_hits` | live target is **`holdout_quality`** (`1cb90aa`) | `PROJECT_FILE_CATALOG.md` §6.1; `S111_TEAM_BETA_BRIEFING.md` |
| **D6** | Chapter 3 §7.2 (and the module's own docstring): v4.2 `percentile(bidirectional_count) + 0.10·median(intersection_ratio)` | live v4.3 `0.70·tanh(enrich) + 0.20·coverage − 0.10·size_penalty` | `CHAPTER_3_ALIGNMENT_AUDIT.md` F3 — verdict **FALSE, not STALE** |
| **D7** | `full_scoring.json` actions: `generate_full_scoring_jobs.py`, `full_scoring_worker.py`, `aggregate_scoring_results.py` | `STEP_SCRIPTS[3]` = `run_step3_full_scoring.sh`, which invokes `generate_step3_scoring_jobs.py` and `scripts_coordinator.py` | parsed live; `run_step3_full_scoring.sh:160, 226`; `PROJECT_FILE_CATALOG.md` §5.1 item 2 |
| **D8** | `scorer_meta.json` actions: `generate_scorer_jobs.py`, `scorer_trial_worker.py` | `STEP_SCRIPTS[2]` = `run_scorer_meta_optimizer.sh`, in no manifest action | parsed live; `PROJECT_FILE_CATALOG.md` §5.1 item 1 |
| **D9** | `PROPOSAL_S172_RANGE_MINER_v1_4_5.md` §16.B: *"PWC remains the frozen authoritative comparator"* | PWC **retired from certifying authority** 2026-07-31; flag-selectable non-certifying diagnostic, hybrid additionally quarantined | `TEAM_ALPHA_PWC_COMPARATOR_SCOPE_CORRECTION.md` (**RULED**) |
| **D10** | whitepaper §4: `R(s) = 1[G(s,−i) = d_{n+1−i}]` — a **backward** generator step | implementation evaluates `G(s,i)` **forward** against a reversed residue array, so forward and reverse generate the identical sequence for one seed | Chapter 2 §3.5, F-7 — **named, not resolved; Beta's side of the boundary** |
| **D11** | `offset` has four definitions: "time offset from current draw" · head-relative array index · *"PRNG steps to skip before sequence"* (`instructions.txt:1181`) · advance by `offset*(skip+1)` (`parameter_registry.json:38-43`) | one payload scalar drives **both** the host data slice and the device pre-advance; coherent only at `skip = 0`; forward hybrids take **no** `offset` at all | Chapter 2 §7, F-4; Chapter 1 §3.1.2. **Beta: settles C-2 as an OBSERVED INCONSISTENCY, not the repair.** No single `(skip+1)` multiplier exists under variable skip; belongs in the future hybrid input-semantics design, **not a standalone arithmetic patch** |
| **D12** | `feature_registry.json`: `skip_min`/`skip_max` = "found during" (**output**) | `parameter_registry.json:160,166`: "for sieve search" (**input**) | Chapter 2 §5.7. One is wrong; correcting it belongs to whichever change settles the semantics |
| **D13** | `TRSE_v1_15_SPEC.md:216-240`: Rules B and C **apply** bounds | code logs only | `TRSE_STEP0_AUDIT_v1.md` §6 S1 — **spec SUPERSEDED** by the S121/S122 shuffle-test ruling |
| **D14** | `TRSE_INTEGRATION_PLAN_S121.md` §2C: ceiling = `min(rec_ws * 4, …)` | hardcoded `min(32, …)`; `recommended_window_size` read into `_rec_ws` and never used | `TRSE_STEP0_AUDIT_v1.md` §6 S2, §5 D2. **`8 × 4 = 32` — the value is correct, the wiring is missing** |
| **D15** | Chapter 1 §11.3/§12.1 anchors: `run_bidirectional_test` `:1138-1680`, gates `:1162/:1300/:1346`, `finalize_run` `:2603`, file 2703 lines | live at HEAD: `:1369`, gates `:1402/:1535/:1580`, `finalize_run` call `:2966`, file **3076 lines** | verified this session. Cause is known: Chapter 1 closed against base `81ef3f1`, and **D6.2 (`f7583bc` → `18a2419`) landed afterwards**, adding ~373 lines. Anchors *above* the insertion (`:125`, `:214`) are exact. **This is the drift the chapter itself predicted and told readers to expect — cite by symbol** |
| **D16** | project naming rule: functional mimicry, **not** seed recovery | `docs/proposals/README.md:1` "Seed Reconstruction System"; `docs/README.md:11` / `README.md:4` "reverse-engineer" | verified live by `/bin/grep`. Already noted in `PROJECT_FILE_CATALOG.md` §6.4 |
| **D17** | "26 GPUs" throughout the chapters | `distributed_config.json` totals **25** (localhost `gpu_count: 1` since `f255912`); `ml_coordinator_config.json` totals **26** | parsed live. Both files are used by Step 2 — `CHAPTER_3_ALIGNMENT_AUDIT.md` F9, Q5 claim 14 (**UNVERIFIABLE as written**) |
| **D18** | `TRIANGULATED_FUNCTIONAL_MIMICRY_VERIFIED_v1_0.md` Part I §2: "46 PRNG variants"; Chapter 8 header: `prng_registry.py` = 4,323 lines | live `KERNEL_REGISTRY` = **44** entries over **11** base families | verified live by import this session; `PROPOSAL_S172_RANGE_MINER_v1_4_4.md` §5.1 also states 44. The same document lists the rigs as "RX 6600" — they are **RX 6600 XT** (`PROJECT_FILE_CATALOG.md` §6.1) |

---

# 17. WHERE ONLY A CODE ANCHOR WAS FOUND (`INCOMPLETE`)

**Each of these is a statement about this search, not about the repository.** Per the governing fact,
the explanation very likely exists in a surface not opened here — the changelog corpus (168 files,
read as a group only), the eleven unaudited chapters, `instructions.txt` (152K, opened only at the
two skip anchors), `Cluster_operating_manual.txt` (96K, not opened), the two PDFs and the `.docx`
(binary, unread), or the pre-repository archives on ser8.

| # | behaviour with a code anchor and no WHY found here | anchor | where to look next |
|---|---|---|---|
| I-1 | **`build_pools.py` defaults `--prng-type` to `xorshift32`** and reads `results/multi_gpu_analysis_*.json`, not `survivors_with_scores.json` — a different lineage from Step 6 | `build_pools.py:169, 89` | it predates the java_lcg focus; check `PREDICTION_STRATEGIES_DOCUMENTED.md` and the S1xx changelogs |
| I-2 | **`backtest_pools.py` is the only in-repo caller** of `build_pools.py`/`evaluate_pools.py`; how the 20/100/300 pools are meant to be produced in production is not stated in any document opened here | `backtest_pools.py:98, 120` | Chapter 7's unread half (§7–§12); `NOTE_Step7_Not_Required_for_Autonomy.md` |
| I-3 | **`prediction_generator.py` writes a second history copy** beside the canonical output | `prediction_generator.py:945-951` | Chapter 7 §8 (Output Formats) — not read in this pass |
| I-4 | **`full_scoring_worker.py`'s sequential path merges 6 fields where the batch path merges 18** — the *defect* is governed (system map §7.3), but no document states why the sequential list was ever shorter | `full_scoring_worker.py:451-455` vs `survivor_scorer.py:772-782` | Chapter 4 is **unaudited**; the S1xx changelog for the sequential-fallback introduction |
| I-5 | **`ml_coordinator_config.json` is tracked and names a 26-GPU fleet** that no fleet mechanism in Beta's six-mechanism table references | `CHAPTER_3_ALIGNMENT_AUDIT.md` F9 (repo-verified) | `FLEET_STATE_REQUIREMENTS_v1.md` covers six mechanisms and does not include this one |
| I-6 | **`chapter_13_orchestrator.py` derives `run_id` as `f"step1_{prng}_{seed_start}"`** for the downstream write-back — the identity convention is not documented in any file opened here | `chapter_13_orchestrator.py:300-304` | Chapter 13 §14 (Outputs) and §18 — not read in this pass |

**None of these is a claim that something is undocumented.** Every one is a pointer for the next
session.

---

# 17.1 FOLLOWING THE §17 POINTERS — FOUR ITEMS, TRACKED SOURCES ONLY

**Added 2026-08-03.** Each of I-2, I-3, I-5 and I-6 carries a "where to look next" naming a
tracked file at HEAD. Each named source was opened and read. **No ser8 access, no new access
surface, nothing fixed.**

**Two verdicts are used, and the distinction is the point:**

- **FOUND** — the named source contains the WHY.
- **NOT-IN-NAMED-SOURCE** — the named source does **not** contain it. **This is never a claim
  that the behaviour is undocumented** (§17 preamble, and the skill's standing rule). Where the
  WHY was located in another *tracked* file during the same pass, it is given under
  **WHY-FOUND-ELSEWHERE** and attributed precisely.

| # | named source | verdict | WHY located? |
|---|---|---|---|
| I-2 | `CHAPTER_7_PREDICTION_GENERATOR.md` §7–§12 · `NOTE_Step7_Not_Required_for_Autonomy.md` | **FOUND** | yes — in the named source |
| I-3 | `CHAPTER_7_PREDICTION_GENERATOR.md` §8 | **NOT-IN-NAMED-SOURCE** | yes — elsewhere, at lower authority |
| I-5 | `FLEET_STATE_REQUIREMENTS_v1.md` | **NOT-IN-NAMED-SOURCE** | yes — elsewhere |
| I-6 | `CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md` §14, §18 | **NOT-IN-NAMED-SOURCE** | yes — elsewhere, decisively |

---

## I-2 — how the 20/100/300 pools are produced in production · **FOUND**

**`CHAPTER_7_PREDICTION_GENERATOR.md` §6.2**, corroborated at **§9.1**, **§10.1** and **§12.3**.

**Answer in one line:** the pools are built **inside Step 6 itself** — `prediction_generator.py`
slices the ranked survivor list at 20 / 100 / 300 into `tight` / `balanced` / `wide`
(`build_prediction_pools()`, §6.2), aggregates weighted votes per pool, and writes
`prediction_pools.json`; `--pool-sizes` defaults to `20,100,300` (§9.1) and the Step-6 agent
manifest declares that file as an output (§10.1).

**This also answers the second half of I-2.** `build_pools.py` / `evaluate_pools.py` are a
**separate backtest lineage**, not the production path — which is exactly why `backtest_pools.py`
is their only in-repo caller. Nothing in the production chain was ever meant to call them.

`NOTE_Step7_Not_Required_for_Autonomy.md` was read in full and does **not** bear on pool
production; it rules that Step 7 (post-pipeline export) is off the critical path because
Chapter 13 keeps its own rolling baseline. Read for completeness, not needed for the answer.

**DIVERGENT (recorded, not adjudicated).** The live module does not implement the three-pool
design. `prediction_generator.py:89` declares a **single** `pool_size: int = 20`; the builder is
`_build_prediction_pool` (**singular**, `:752`) feeding a Top-K selection; there is no
`build_prediction_pools`, no `prediction_pools.json`, no `ranked_predictions.json` and no
`--pool-sizes` anywhere in the module. `agent_manifests/prediction.json` (v1.6.0) **agrees with
the code**, not the chapter — one output, `predictions/next_draw_prediction.json`, with
`pool_size: 20, k: 20`. This is the same divergence already recorded at §18 observation 2 and
governed as system-map **Blocker 1**. Chapter 7 is dated Dec-2025 and predates the v6.0 module.

**Worth flagging for whoever adjudicates:** the S140b pipeline objective is
`0.50·hit@20 + 0.30·hit@100 + 0.15·hit@300 + 0.05·pool_coverage` — it still weights **three**
pool sizes, while the live generator emits **one**. **Not adjudicated here.**

---

## I-3 — `prediction_generator.py`'s second history copy · **NOT-IN-NAMED-SOURCE**

**What was read:** `CHAPTER_7_PREDICTION_GENERATOR.md` §8 in full — §8.1 (File Outputs), §8.2
(`ranked_predictions.json`), §8.3 (`prediction_pools.json`).

§8.1 tables **exactly three** outputs — `ranked_predictions.json`, `prediction_pools.json`,
`next_draw_prediction.json`. **There is no history copy, no `history/` directory and no archival
concept anywhere in §8.** As with I-2, §8 describes an output shape the live module does not
produce, so it could not explain a behaviour of the current code even in principle.

**WHY-FOUND-ELSEWHERE — and the authority is lower, which matters.** The rationale is asserted
by **the code's own comments**, not by a governing document:

```
prediction_generator.py:944   # Save canonical (WATCHER validates this)
prediction_generator.py:949   # Save to history archive (non-contractual)
```

with `history_path = canonical_dir / "history" / f"predictions_{timestamp}.json"` (`:911-913`).
So the second copy is a **timestamped audit archive deliberately placed outside the validated
contract** — the canonical file is what WATCHER validates, and the archive is explicitly
`non-contractual` so accumulating copies can never widen the contract.

**Under this document's two-anchor rule that is a WHAT-surface carrying an authorial WHY, not a
WHY-surface.** It is the author's stated intent, not a governed decision.

**Corroborating pattern in tracked docs, for a sibling module — not for this one.**
`NOTE_Step7_Not_Required_for_Autonomy.md` Finding 4 documents the identical arrangement for
`chapter_13_diagnostics.py` (`diagnostics_history/`, archived by timestamp "for audit trail and
historical analysis", `:851-878`), and Chapter 13 §14.1 assigns `diagnostics_history/` a **1-year
retention**. Same pattern, two modules — **but neither document names `prediction_generator.py`**,
so this is corroboration of a convention, not a citation for this behaviour.

**Try next:** the S1xx session changelog that introduced `prediction_generator.py` **v6.0** (the
module's own docstring carries the version, and §8's shape belongs to the pre-v6.0 design);
`docs/PROJECT_FILE_CATALOG.md` §4.8 for the patch that added the archive.

---

## I-5 — `ml_coordinator_config.json`'s 26-GPU fleet · **NOT-IN-NAMED-SOURCE**

**What was read:** `FLEET_STATE_REQUIREMENTS_v1.md` §0 (the falsifiable question and the
five-plus-one table), §5.1 (agreement/divergence matrix), §5.4 (failure modes covered by NO
mechanism), §8 (what was NOT done), plus a `/bin/grep` of the whole file.

**`ml_coordinator_config` appears zero times in the document.** §5.1 names each mechanism's
source of truth — `dataset_provisioning.json`, `distributed_config.json` (three times),
`/etc/cluster-boot-notify.conf`, `--worker-pool-size`. `ml_coordinator_config.json` is **not among
them**. §5.4 enumerates failure modes no mechanism covers and does not raise it; §8 records the
pass's scope limits and does not defer it. **This confirms §17's own prediction** rather than
overturning it.

**WHY-FOUND-ELSEWHERE — it is absent because it is not a fleet mechanism.** §0 scopes the
document to one question: *"what is the required fleet state for a run to proceed"* — i.e. **gates
a run passes through**. `ml_coordinator_config.json` is not a gate; it is the **node/GPU
allocation table for `MultiGPUCoordinator`**, consumed by the ML and scoring steps:

| anchor | what it shows |
|---|---|
| `REMOTE_NODE_SETUP_CHECKLIST.md:92` | *"`ml_coordinator_config.json` \| ML coordinator settings \| **Steps 2.5, 3, 5**"* |
| `steps/step2_execution_manager.py:64` | `def __init__(self, coordinator_config="ml_coordinator_config.json", max_concurrent=26)` |
| `run_full_scoring.sh:80`, `run_anti_overfit_optimizer.sh:115` | `MultiGPUCoordinator('ml_coordinator_config.json')` |

All six mechanisms in the matrix are **Step-1 / sieve / miner-side**. This file serves a
different subsystem, so its absence from the matrix is **scope, not omission** — and that is the
answer to "why does no fleet mechanism reference it."

**Live parse (this session):** 26 = localhost `2` + `.120` `8` + `.154` `8` + `.162` `8`,
bare-metal addresses, file **tracked**. **Caveat against over-reading the above:** its top-level
keys are `nodes`, `sieve_defaults`, `reverse_sieve_defaults` — it also carries *sieve* defaults,
so "purely an ML-steps config" is **not a clean reading**, and the boundary is **not adjudicated
here**.

**Try next:** **Chapter 4 is unaudited** and is where Step-3 coordinator construction is
documented; `CHAPTER_3_ALIGNMENT_AUDIT.md` F9 has already repo-verified the count. See also
**D17**, which records the 25-vs-26 split against `distributed_config.json`.

---

## I-6 — `chapter_13_orchestrator.py`'s `run_id` convention · **NOT-IN-NAMED-SOURCE**

**What was read:** `CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md` §14 (§14.1 Diagnostic Outputs, §14.2
Decision Outputs, §14.3 Audit Trail) and §18 (§18.1–§18.8, the configurable-parameter reference),
plus a `/bin/grep` of the whole file.

§14 names output **files** and the audit fields logged per decision; it never states a run
identity convention. §18 is a parameter reference and contains no `run_id`. **The entire document
contains exactly one `run_id` — `:351`, inside §8.2's diagnostics schema — and it is a *different
shape*: `"chapter13_20260111_172000"`.** So Chapter 13 documents its own Chapter-13-scoped
identity and is silent on the Step-1-scoped one the orchestrator builds.

**WHY-FOUND-ELSEWHERE — decisively, in the tracked one-shot patch corpus.**
`step1_{prng}_{seed_start}` is the **canonical row identity of the `step1_trial_history` database
table**, under a `UNIQUE(run_id, trial_number)` constraint:

- **`apply_s142_partition_runid.py:5-24`** — root cause, TB-confirmed: both NP2 partition workers
  wrote the *same* `run_id`, so `INSERT OR IGNORE` silently discarded whichever lost the race —
  *"~50% of COMPLETE trial rows missing, no exception, no print."* The fix appended
  `_p{partition_idx}`.
- **`apply_s142c_remove_worker_writes.py:6-22`** — TB Option A **superseded** that fix:
  `_worker_obj` is an incomplete execution path and must not write to the canonical table at all;
  backfill from the shared Optuna study becomes the only writer; and the `_backfill` suffix is
  **dropped** so the canonical `run_id` is exactly `step1_{prng}_{seed_start}`. Post-patch cleanup
  deletes every `_p0` / `_p1` / `_backfill` row.

**Answer in one line:** the orchestrator reconstructs that exact string from
`optimal_window_config.json` (`chapter_13_orchestrator.py:296-303`) because the downstream-score
write-back is an **annotation onto the Step-1 row that already exists** — the `run_id` is a
**join key, not a label**, which is precisely why it must match byte-for-byte and carry no suffix.
The code's own `# [S140b] DOWNSTREAM SCORE WRITE-BACK — annotation only` (`:292`) says as much.

**Corroboration that the shape is a shared contract, not local to Chapter 13:**
`tests/phase6/wall_ab_gate.py:393` and `tests/test_s172_phase5_d3_5_finalizer.py:248` build the
same string, and the authoritative generation is named `gen-…-step1_java_lcg_0`.

**Try next:** the **S140b / S142 session changelogs** for the TB ruling text itself — the patch
docstrings quote the ruling but are not the ruling.

---

## What this pass changes about where to look

**The tracked `apply_s*.py` one-shot patch corpus is a WHY surface, and §17 did not list it.**
§17's preamble named the changelog corpus, the unaudited chapters, the two large `.txt` files, the
binaries and ser8. **I-6's answer came from patch docstrings** that carry TB root-cause diagnoses
and rulings verbatim. `PROJECT_FILE_CATALOG.md` §4.8 indexes that corpus.

**Two of these four answers were not in the named source, and both were still in the repository.**
Neither required ser8. Per `SER8_ARCHIVE_INVENTORY.md`, ser8 was unreachable for credential
reasons in any case — **and was not needed.**

## Verification-integrity controls (VIR-1…6)

- **execution proof:** four named sources opened on VM101 at HEAD `cfb9f9c`, clean tree —
  `CHAPTER_7_PREDICTION_GENERATOR.md` (§6–§13 read, lines 386–933, plus a full heading map of all
  933) · `NOTE_Step7_Not_Required_for_Autonomy.md` (**full**) · `FLEET_STATE_REQUIREMENTS_v1.md`
  (§0, §5.1, §5.4, §8 read; full heading map; whole-file `/bin/grep`) ·
  `CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md` (§14 and §18 read; full heading map; whole-file
  `/bin/grep`). Source read for WHAT-anchors: `prediction_generator.py` (`:86-95`, `:900-980`,
  symbol map) · `chapter_13_orchestrator.py:288-315` · `apply_s142_partition_runid.py:1-30` ·
  `apply_s142c_remove_worker_writes.py:1-30`. Parsed live: `ml_coordinator_config.json` (26 GPUs)
  and `agent_manifests/prediction.json`.
- **clean control / fault-injection control:** `NOT_APPLICABLE` — this section resolves pointers;
  it is not a detector.
- **completion sentinel:** `PASS` for all four — each named source was read and returned a
  determinate verdict. **`PASS` means the pointer was followed, not that the behaviour is
  correct.** Nothing here was adjudicated and nothing was fixed.
- **unavailable-observer behaviour:** none — all four sources were tracked and readable. **No
  absence claim is made about any surface not opened.** Every NOT-IN-NAMED-SOURCE verdict is
  scoped to the named file alone.
- **audit claim scope:** the four §17 pointers only. **I-1 and I-4 were not investigated** — both
  point at the 168-file changelog corpus, outside this pass.
- **searched surfaces:** the four named tracked documents; the live source anchors above; the
  tracked `apply_s*.py` corpus by `/bin/grep`.
- **unavailable surfaces:** ser8 (no credential — see `SER8_ARCHIVE_INVENTORY.md`); the changelog
  corpus, `instructions.txt` and `Cluster_operating_manual.txt` (**not opened this pass**).

---

# 18. OBSERVATIONS

Short, labelled observations — **not findings**, and nothing was verified beyond the search named.

1. **Chapter 1's anchors into `window_optimizer_integration_final.py` have drifted since its closure.**
   *Search:* `git diff --stat 81ef3f1..HEAD`, `wc -l`, and `/bin/grep -n` for four symbols. The file
   is **3076** lines against the chapter's 2703; `run_bidirectional_test` is at `:1369` not `:1138`.
   Cause is benign and known — D6.2 landed after the closure pass. Anchors above the insertion point
   are exact. The chapter already instructs readers to cite by symbol (§17.2). **Observation, D15.**
2. **`prediction_generator.py`'s live output shape does not match Chapter 7's**, and the manifest
   agrees with the code. *Search:* `/bin/grep -n` for `json.dump|ranked_predictions|prediction_pools`
   across the module; repo-wide `/bin/grep -rn "ranked_predictions" --include=*.py`; live parse of
   `agent_manifests/prediction.json`. **Observation, D3** — and the absent writer is already governed
   as system-map Blocker 1.
3. **The Step-3 map divergence has the same shape as Step 2's**, confirmed live rather than inherited.
   *Search:* `/bin/grep -n` over `run_step3_full_scoring.sh`; live parse of `full_scoring.json`.
   `PROJECT_FILE_CATALOG.md` §7 gap 2 records it as *"not found in the documents indexed here"*, not
   as undocumented — Chapter 4 is unaudited and was not read line-by-line for it. **Observation, D7.**
4. **Chapter 4 ends with the string `*End of Chapter 6: Survivor Scorer*` and its "Next Chapter"
   points at "Chapter 7: GPU Optimizer"**, which is not the live Chapter 7. *Search:* read the file
   in full. Same self-labelling class the Chapter 3 audit records at its §2 (that chapter ends *"End
   of Chapter 13"*). **Observation.**
5. **`docs/proposals/README.md` is titled "Seed Reconstruction System".** *Search:*
   `/bin/grep -n -i "seed reconstruction|reverse-engineer"` over the three READMEs. Already noted in
   the catalog §6.4 as **not fixed here**; repeated only because §15.1 is where a new reader forms
   the project's self-description. **Observation, D16.**
6. **Two of the four `sieve_gpu_worker.py` hybrid observations are visible from one screen.**
   *Search:* `/bin/grep -n "family_name ==|family_name in" sieve_gpu_worker.py` — seven branches at
   `:217, 221, 223, 227, 232, 299, 302`, covering **6 base families**, exactly as `CLAUDE.md` §7 and
   `PROPOSAL_S172_RANGE_MINER_v1_4_4.md` §5.3 state. **Observation: the documented coverage fact is
   confirmed at HEAD.**
7. **The dead-placeholder count is confirmed from the merge site itself.** *Search:* read
   `survivor_scorer.py:765-790`. `skip_mean`, `skip_std`, `skip_entropy`, `survivor_velocity`,
   `velocity_acceleration` all appear in both the merge list and the zero-fill default list — i.e.
   they are *plumbed for*, and receive `0.0` because `survivor_metadata` never carries them.
   **Observation: consistent with system map §8.3 and Chapter 2 §8.6; the producer gap is upstream at
   `…final.py:125`, exactly as documented.**

---

# 19. WHAT A NEW SESSION MOST NEEDS TO KNOW

1. **Read `docs/PROJECT_FILE_CATALOG.md` §1 (the governance trail) before submitting anything to
   Beta.** That section is what was missed on 2026-08-02, twice.
2. **There are two step numberings and the mapping is written down nowhere.** *Executable* Step 1
   contains the sieve; *executable* Step 2 is the Scorer Meta-Optimizer; Chapter N ≠ Step N; "Phase
   7" names two unrelated milestones.
3. **The canonical Step-1 output is the certified NPZ generation, not any `*_survivors.json`.**
   `bidirectional_survivors.json` is a post-success summary with no seeds; the two directional files
   are count-only stubs, and that is deliberate (`[S166-ACCUM]`).
4. **Before proposing to remove, demote or simplify ANY component, find and cite the document
   explaining why it exists.** Skip is the canonical case: three parties independently recommended
   deleting `skip_min`/`skip_max` because the explanation had not been written yet. It has been now —
   Chapter 2 §5.1.
5. **"The code doesn't do X" usually means X broke, not that X was never wanted.** Six instances of
   *tuned parameter never reaches the kernel* are catalogued; the fix pattern is **one canonical path
   — resolve once in the parent, never reinterpret downstream, record requested/payload/effective**.
6. **Loose thresholds and forward-iterating reverse kernels are NOT defects.** Neither is
   `intersection_count` duplicating `bidirectional_count`, nor `serve_timeout = None`, nor
   `distributed_config.json`'s bare-metal addresses, nor Step 0's silent failure, nor Chain D's
   `pending_approval`, nor the Step-5 `allowed_params` filter.
7. **A keyword hit is not a finding until you read the surrounding text.** The sharpest recorded
   failure is an absence claim made after a full-tree grep that *reached the exact line and did not
   read it*.
8. **The repository is not the system (VIR-6).** systemd units, cron, host config and deployed
   uncommitted files are invisible to every repo-scoped gate. `.gitignore:41` is `*.json`, so
   gitignored configs are invisible to repo-scoped searches — and the shell `grep` here is a ugrep
   wrapper that honours `.gitignore`. **Use `/bin/grep` for complete `.json` searches.**
9. **A report is a snapshot and its findings expire.** This document included. Re-verify any anchor
   before acting on it.
10. **Certification scope is narrower than "Phase 6 passed".** Wall A/B used **constant-skip**
    generations. Hybrid worker semantics are covered by the transfer gate only, hybrid certification
    is **blocked** on §11.4, and combined-session sequential sieving is **prohibited by default**.

---

# 20. VERIFICATION-INTEGRITY CONTROLS (VIR-1…6)

**execution proof.** **21 documents plus the `tfm-project-facts` skill were opened and read this
session** — not summarised from filenames. Per subject:

| subject | documents read |
|---|---|
| orientation | `PROJECT_FILE_CATALOG.md` (803L, **full**) · `CLAUDE_CODE_INSTRUCTIONS_PIPELINE_BEHAVIOUR_MODEL.md` (141L, full) · `tfm-project-facts` skill (full) |
| foundations | `BIDIRECTIONAL_SIEVE_MATHEMATICAL_WHITEPAPER.md` (167L, full) · `TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md` (292L, full) · `TRIANGULATED_FUNCTIONAL_MIMICRY_VERIFIED_v1_0.md` (Part I) · `CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3_5_B.md` (§0 + sidecar field block) |
| **Step 0** | `TRSE_INTEGRATION_PLAN_S121.md` (192L, full) · `TRSE_STEP0_AUDIT_v1.md` (380 of 537L) — **2** |
| **Step 1 + sieve + RANGE-MINER** | `CHAPTER_1_WINDOW_OPTIMIZER.md` (2303L, **full**) · `CHAPTER_2_BIDIRECTIONAL_SIEVE.md` (1463L, **full**) · `PROPOSAL_S172_RANGE_MINER_v1_4_5.md` (344L, full) · `PROPOSAL_S172_RANGE_MINER_v1_4_4.md` (747L, full) — **4** |
| **Step 2** | `CHAPTER_3_SCORER_META_OPTIMIZER.md` (§1–§7) · `CHAPTER_3_ALIGNMENT_AUDIT.md` (629 of 925L) — **2** |
| **Step 3** | `CHAPTER_4_FULL_SCORING.md` (1037L, **full**) — **1** |
| **Step 4** | `CHAPTER_5_ML_ARCHITECTURE_OPTIMIZER_v2.md` (423L, **full**) — **1** |
| **Step 5** | `CHAPTER_6_ANTI_OVERFIT_TRAINING.md` (738L, **full**) — **1** |
| **Step 6** | `CHAPTER_7_PREDICTION_GENERATOR.md` (470 of 933L) — **1** |
| **WATCHER / feedback** | `CHAPTER_12_WATCHER_AGENT.md` (880L, **full**) · `CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md` (~560 of 1231L) · `WATCHER_POLICIES_REFERENCE.md` (211L, full) — **3** |

**Source verification.** Every `file:line` presented as a **WHAT** anchor was read on VM101 this
session by `Read` or `/bin/grep -n`, except where explicitly attributed to a cited chapter or audit
(marked *chapter-cited*). Files opened: `agents/watcher_agent.py` · `utils/canonical_arrays.py` ·
`utils/run_finalizer.py` · `utils/prng_encoding.py` · `window_optimizer_integration_final.py` ·
`miner/range_miner_worker.py` · `miner/dataset_authority.py` · `miner/step1_ingress.py` ·
`execution_set.py` · `survivor_scorer.py` · `full_scoring_worker.py` · `prediction_generator.py` ·
`build_pools.py` · `evaluate_pools.py` · `backtest_pools.py` · `chapter_13_orchestrator.py` ·
`chapter_13_triggers.py` · `chapter_13_acceptance.py` · `sieve_gpu_worker.py` ·
`run_step3_full_scoring.sh` · all seven `agent_manifests/*.json` (parsed) ·
`distributed_config.json` / `ml_coordinator_config.json` (parsed) · `prng_registry` (imported live:
**44 kernels / 11 base families**) · the three `README` files.

**clean control:** `NOT_APPLICABLE` — this deliverable produces an explanation, not a detector.
**fault-injection control:** `NOT_APPLICABLE` — same reason.

**completion sentinel, per step:**

```
Step 0  PASS       Step 3  PASS
Step 1  PASS       Step 4  PASS
Step 2  PASS       Step 5  PASS
RANGE-MINER PASS   Step 6  PASS
Feedback (Ch13)    PASS
OVERALL            PASS
```

`PASS` means **documented from sources with both anchors present, and every gap named** — not that
any step is verified, certified or complete. No step is `INCOMPLETE`: every one was documentable from
WHY + WHAT. Six *individual behaviours* are marked `INCOMPLETE` in §17 and are enumerated there.

**unavailable-observer behaviour.** Nothing here was established by execution. **The pipeline,
WATCHER and every scraper were not run; `convert_survivors_to_binary.py` was not invoked; `miner/`
was read but not modified; no CT100 worker or rig was contacted.** All work was on VM101.

**audit claim scope.** **Repo-scoped at HEAD `49c13ad`**, on the VM101 working tree.
`agent_manifests/definitions.json` — **not read**; it is untracked, gitignored (`.gitignore:41`) and
**absent from a fresh clone**, so any statement about it would be host-only. Host state (systemd,
cron, deployed uncommitted files) and the ser8 pre-repository archives are **out of scope and are not
implied by anything above.**

**searched surfaces.** `docs/` **and the governance trail** (VIR-6 addendum): `PROJECT_FILE_CATALOG.md`
§1 in full, plus every ruling/submission it pairs, cited by name above · the eight chapters listed ·
the two RANGE-MINER proposals · the whitepaper and system map · the TRSE spec/plan/audit · the
WATCHER policy reference · every code path anchored in §1–§15 · `agent_manifests/` (7 step manifests
parsed) · `git log 81ef3f1..HEAD` · live Python imports of `prng_registry` and `utils.*`.

**unavailable surfaces — declared, not assumed clean.**

1. **The CA draw-procedures PDF is not in the repo** and was not read. §11.1's citation is
   `UNAVAILABLE`; its statements come from Beta's ruling text, which is **not** verification at
   source. This absence is the root cause of both the two-pre-test misreading and the near-removal of
   `skip_min`/`skip_max`.
2. **Eleven chapters are unaudited** — 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, ~11,000 lines — against a
   measured base rate of 9/41 (Chapter 1) and 17/55 (Chapter 3). §§5–9 above rest partly on them and
   are qualified accordingly.
3. **168 session changelogs were not read individually.** They are the recorded home of governing
   decisions that never became their own document — the canonical example being
   `SESSION_CHANGELOG_20260307_S122.md:56`, the only place the TRSE Rules-B/C ruling exists.
4. **`instructions.txt` (152K) was opened only at its two skip anchors; `Cluster_operating_manual.txt`
   (96K) was not opened.** Both are load-bearing for skip semantics.
5. **The two PDFs and `TEAM_BETA_REVIEW_kfolds_S100.docx` are binary and unread.** If the k-folds
   review contains a binding ruling, this document does not carry it.
6. **Rig-deployed source was not compared against VM101.** Every code claim is a claim about the
   VM101 tree. **Repo ≠ system.**
7. **No runtime values.** Threshold, skip, partition, admission and dataset behaviour is traced by
   source, never measured.
8. **Team Beta's ruling texts exist outside the repo except where transcribed.**

---

⚠ **This file is untracked and dirties the working tree.** The Phase-7 soak's clean-tree preflight
rejects a dirty tree at finalization (`run_finalizer.py`), so **`docs/PIPELINE_BEHAVIOUR_MODEL.md`
must be committed before the soak launches, or before a running soak reaches publication.**

*Produced 2026-08-03 by Claude Code on VM101 under
`docs/CLAUDE_CODE_INSTRUCTIONS_PIPELINE_BEHAVIOUR_MODEL.md` REV1. Read-only except this file: no
code, config, manifest, chapter or gate was changed; nothing was committed or pushed; the pipeline,
WATCHER and the scrapers were not run; `convert_survivors_to_binary.py` was not invoked; no work was
dispatched to CT100 or the rigs; nothing was audited and no finding was fixed.*
