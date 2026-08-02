# CHAPTER_1_AUDIT_v1.md

**Audit of `docs/CHAPTER_1_WINDOW_OPTIMIZER.md` against live source.**

| field | value |
|---|---|
| Authority | `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_1_AUDIT.md` REV1 |
| Base commit | `77dc629` (VM 101, `main`, working tree clean of tracked modifications) |
| Host | VM 101 `192.168.3.177`, user `michael`, venv `~/venvs/torch` |
| Date | 2026-07-31 |
| Type | **AUDIT ONLY** — no chapter edits, no code changes, no commits |
| Sentinel | **FAIL** (see §7) — the chapter does not accurately describe live source |

> **Scope note.** "FAIL" is the verdict on *the chapter*, not on the audit. The audit
> itself completed: 100% of chapter sections were reached and classified. No correction
> has been applied; that is the separate authorized deliverable.

---

## 0. Executive summary

The chapter is a **v3.1 (Feb 2026) snapshot of a file that has since roughly doubled in
size**. Of 41 classified claims, **9 are ACCURATE**, **19 STALE**, **5 SUPERSEDED**,
**7 CONTRADICTED-BY-CODE**, **1 UNVERIFIABLE**.

Four findings dominate:

1. **Every numeric search bound in the chapter is wrong** (§4.1/§4.2/§3.2). The chapter's
   thresholds `[0.15, 0.60] default 0.25` are not the live values `[0.30, 0.75] default
   0.30`, and its window/skip ceilings are 10× and 2× too large. This is the exact class of
   error that produced the "~62 features" incident.
2. **A new dead dimension**: `--forward-threshold` / `--reverse-threshold` are declared in
   argparse and **never read** (§4.2 of this report). The chapter documents them as
   "Override Optuna optimization." That override does not exist.
3. **Three of the four documented search strategies raise `TypeError` on first call.**
   Verified by live `inspect.signature` (§5, conflict C-3). Only `bayesian` runs.
4. **The chapter's output-file contract is superseded.** `bidirectional_survivors.json` is
   now a post-success *summary*; `forward_survivors.json` / `reverse_survivors.json` are
   count-only stubs; the canonical Steps 2–6 input is the certified NPZ generation. The
   chapter still presents all three as survivor data.

Against the standing rule (skill §0.4): **in every doc-vs-code conflict below I state which
side reflects design intent.** In two cases (C-1 `skip_min`/`skip_max`, C-2 `offset`) **the
chapter is right and the code is defective**. In the remainder the code is right and the
chapter is stale.

---

## 1. Header reality check

| claim (chapter) | actual | class |
|---|---|---|
| `**Version:** 3.1` (line 5) | The live module docstring declares `Version: 2.0  Date: 2025-11-15` (`window_optimizer.py:5-6`). No `3.1` string exists in either source file. The chapter's version is a **doc-only** number with no source counterpart. | **STALE** |
| `**Lines:** ~868 + ~595` (line 7) | `window_optimizer.py` = **1306** lines; `window_optimizer_integration_final.py` = **2679** lines. | **STALE** |
| Divergence | +438 lines (+50.5%) and +2084 lines (+350.7%). Combined **1463 → 3985 lines, +172%**. | — |
| `**File:** window_optimizer.py + window_optimizer_integration_final.py` (line 6) | Correct as far as it goes, but Step 1 now has a **third** load-bearing module the header omits: `window_optimizer_bayesian.py` (984 lines) owns the entire Optuna search space, study storage and warm-start. | **STALE** |

**Interpretation.** The `~595` figure is closest to `docs/window_optimizer_integration_final.py`
(1877 lines) — a **stale duplicate** in `docs/`, not the live root file. Per project memory
the two stale window-optimizer copies (`docs/window_optimizer_integration_final.py`,
`modules/window_optimizer.py` at 327 lines) are **RULED to be left alone**; this audit does
not propose touching them. It notes only that a future reader must not mistake them for the
live modules. **Live modules are the repo-root ones.**

---

## 2. `CHAPTER_1_PATCH_S114.md` disposition

**Verdict: UNMERGED *and* SUPERSEDED — the worst of both.** It was never folded into the
chapter, and while it sat unmerged its central mechanism was deleted from the code.

| patch claim | live state | class |
|---|---|---|
| Warm-start enqueues hardcoded `W8_O43_S5-56, 0.49/0.49` as trial 0 (patch lines 33-40) | **Removed.** `window_optimizer_bayesian.py:627-628`: *"[S144] Warm-start: enqueue from trial_history_context ONLY. No hardcoded fallback — CA-specific W8_O43 removed."* Live warm-start is context-driven from `step1_trial_history` (`:630-650`), gated on all six params being non-`None` (`:639`), and requires `session_idx` (`:643`, S166). | **SUPERSEDED** |
| `--resume-study` flag | **Present and live** — `window_optimizer.py:1069-1072`. Patch content is accurate here. | **ACCURATE** |
| Resume logic scans `optuna_studies/window_opt_*.db` by mtime | **Accurate** — `window_optimizer_bayesian.py:569-573`. Patch also missed that a later addition, `--study-name`, takes priority over auto-select (`:565-567`). | **ACCURATE (incomplete)** |
| Manifest `trials` default 50 → 100 | **Neither.** `agent_manifests/window_optimizer.json` `default_params` has **no `trials` key**; it has **`window_trials: 3`**. Argparse `--trials` default is still **50** (`window_optimizer.py:1041`). Three different values across three surfaces. | **STALE** |
| Manifest `resume_study` param | Present (`default_params.resume_study: false`). | **ACCURATE** |
| "Key Discovery: Discrete PRNG Regime Structure" — W3→143,959 survivors, W8→43 | **Unverifiable from source.** These are empirical S114 run results, not code properties. Note the S172 TB ruling recorded in `distributed_config.json` (`search_bounds.window_size._s172_note`) states *"W=2/3 produces ~39%/53% survivor rate by chance alone, regardless of threshold"* and raised `window_size.min` from 2 to **6** — which reinterprets the 143,959 figure as **noise, not signal**. The patch's headline discovery is superseded in meaning even though its numbers may be reproducible. | **SUPERSEDED** |

**Chapter/patch disagreement.** The chapter's §10.1 CLI list omits `--resume-study`
entirely, while the chapter's *appended* "Optuna Resume" block (chapter lines 1047-1051)
uses it. The chapter therefore contradicts itself: the flag is both absent from and present
in the same document.

**Recommendation for the correction pass:** do **not** merge `CHAPTER_1_PATCH_S114.md`
verbatim. Merge only the `--resume-study` CLI semantics; rewrite warm-start from
`window_optimizer_bayesian.py:627-650`; drop the hardcoded-enqueue block; and re-frame the
"discrete regime" discovery against the S172 window-floor ruling.

---

## 3. Per-section classification table

`file:line` anchors were all read in this session. `WO` = `window_optimizer.py`,
`WOIF` = `window_optimizer_integration_final.py`, `WOB` = `window_optimizer_bayesian.py`,
`SGW` = `sieve_gpu_worker.py`, `DC` = `distributed_config.json`.

### §1 Overview

| § | claim | class | anchor / true state |
|---|---|---|---|
| 1.1 | Step 1 does parameter optimization + survivor generation | **ACCURATE** | `WO:525-822` (Bayesian), `WO:825-1022` (config mode) |
| 1.1 | "Runs real sieves across all 26 GPUs" | **ACCURATE** | `DC.nodes` = 2+8+8+8 = **26** GPUs (verified this session) |
| 1.2 | Version history stops at 3.1 (S104, Feb 2026) | **STALE** | ≥30 later session tags are live in-source: S115, S116, S118, S119, S121, S123, S124, S125, S130, S133-B, S134, S137, S140, S140b, S142, S144, S145, S146, S149, S150, S152, S158D, S162, S166, S170, S172. None appear in §1.2 |
| 1.2 | "v3.1 RESTORED 7 intersection fields" | **ACCURATE** | All 7 present: `WOIF:1520-1526` (constant), `WOIF:1624-1630` (variable) |
| 1.2 | "V2.0 backward compatible: defaults to constant skip only" | **CONTRADICTED-BY-CODE** | True at the CLI (`WO:1077`, `store_true`), **false in production**: `agent_manifests/window_optimizer.json` `default_params.test_both_modes = true`. WATCHER-driven runs default to BOTH modes |
| 1.3 | "The optimizer doesn't run sieves directly — delegates to integration layer" | **ACCURATE** | `WO:586`, `WO:853` (lazy imports), `WOIF:2389` (`optimizer.test_configuration = test_config`) |
| 1.4 | Three usage examples | **STALE** | All three still parse, but they omit every flag that governs which backend actually executes (`--use-persistent-workers`, `--use-zmq-sqlite`, `--use-range-miner`). Example 1 as written runs the legacy SSH-subprocess path, which is no longer the production default |

### §2 Architecture

| § | claim | class | anchor / true state |
|---|---|---|---|
| 2.1 | Component hierarchy: `WindowOptimizer → test_configuration → WOIF → run_bidirectional_test → coordinator.py → 26 GPUs` | **STALE** | The chain is real (`WOIF:2373-2387`) but is now **one of four** backends. `run_bidirectional_test` opens with a backend cascade: RANGE-MINER first (`WOIF:1167-1168`), then PWC, then ZMQ, then legacy. The hierarchy diagram shows only the legacy leg |
| 2.1 | Output files list (6 files) | **SUPERSEDED** | See §12 rows. The canonical output is now a certified NPZ generation via `utils.run_finalizer` (`WOIF:2490-2495`), which the diagram does not mention |
| 2.2 | Execution flow `main() → run_bayesian_optimization() → …` | **STALE** | Correct skeleton, but omits three live post-optimization stages: `[S140]` seed-coverage DB write-back (`WO:672-697`), incremental-field merge (`WO:724-742`), `[S121]` TRSE `confirmed_windows` feedback (`WO:769-798`) |

### §3 Data Structures

| § | claim | class | anchor / true state |
|---|---|---|---|
| 3.1 | `WindowConfig` field *names* and order | **ACCURATE** | `WO:85-91` — all 7 fields match exactly |
| 3.1 | `offset` = "Time offset from current draw" | **CONTRADICTED-BY-CODE** | Third interpretation of three. See conflict **C-2**. Code slices `data[offset : offset+window_size]` — a head-relative array index (`sieve_filter.py:184-186`) |
| 3.1 | `skip_min` / `skip_max` = "Minimum/Maximum skip for variable PRNGs" | **CONTRADICTED-BY-CODE — the chapter is correct, the code is defective** | See conflict **C-1**. This is the cornerstone definition the audit brief was written to protect |
| 3.1 | `forward_threshold: float = 0.25` | **STALE** | `WO:90` → **`0.40`** |
| 3.1 | `reverse_threshold: float = 0.25` | **STALE** | `WO:91` → **`0.45`**. Note fwd and rev defaults differ in live code; the chapter shows them equal |
| 3.1 | Methods `__hash__`, `description()`, `to_dict()` | **ACCURATE** | `WO:93`, `:98`, `:103` |
| 3.1 | Example output `W512_O100_midday+evening_S0-50_FT0.25_RT0.25` | **STALE** | Format string `WO:101` is correct; the *example values* are unreachable — `window_size=512` exceeds the live ceiling of 50, and `0.25` is below the live threshold floor of 0.30 |
| 3.2 | `min_window_size: int = 2` | **STALE** | `WO:116` = 2 in code, but `DC.search_bounds.window_size.min` = **6** and config wins (`WO:57-61`). Effective floor is **6** (S172 TB ruling) |
| 3.2 | `max_window_size: int = 500` | **STALE** | `WO:115` = **50** (S139); `DC` = **50**. Chapter is 10× high |
| 3.2 | `max_skip_max: int = 500` | **STALE** | `WO:121` = **250**; `DC` = **250**. Chapter is 2× high |
| 3.2 | Threshold bounds `[0.15, 0.60]` | **STALE** | `WO:123-126` = `[0.40, 0.75]`; `DC` = **`[0.30, 0.75]`** (config wins) |
| 3.2 | `default_forward/reverse_threshold = 0.25` | **STALE** | `WO:128-129` = `0.50`; `DC` default = **`0.30`** (config wins) |
| 3.2 | `min_offset`/`max_offset` = 0/100 | **ACCURATE** | `WO:117-118`; `DC.search_bounds.offset` = `{0, 100}` |
| 3.2 | `min_skip_min`/`max_skip_min` = 0/10 | **ACCURATE** | `WO:118-119`; `DC` = `{0, 10}` |
| 3.2 | Methods table: `from_config`, `random_config`, `is_valid` | **STALE (incomplete)** | All three exist (`WO:132`, `:198`, `:213`) but the table **omits `validate_baseline_in_bounds()`** (`WO:163-196`) — an explicitly **Team Beta-mandated** guard that raises `ValueError` when the baseline falls outside bounds. Omitting a TB mandate from the reference doc is the higher-consequence half of this row |
| 3.2 | Session options: `[['midday','evening'], ['midday'], ['evening']]` | **CONTRADICTED-BY-CODE (governance)** | Code matches exactly (`WO:156-160`) and Optuna samples all three (`WOB:426-428`). But TB has ruled combined-session sequential sieving **non-certifying and prohibited by default**, with production re-optimization **per-session**. Both the chapter and the code predate that ruling. See conflict **C-6** |
| 3.3 | `TestResult` fields + `precision` / `recall` | **ACCURATE** | `WO:232-246`; formulas match verbatim |
| 3.3 | Properties list | **STALE (incomplete)** | Omits `to_dict()` (`WO:248-258`), which is what the Optuna callback persists (`WOB:472`) |

### §4 Search Bounds Configuration

| § | claim | class | anchor / true state |
|---|---|---|---|
| 4.1 | "Single Source of Truth" — bounds loaded from `distributed_config.json` | **ACCURATE (mechanism), STALE (values)** | The merge is real and config genuinely wins: `WO:54-62`. The claim holds |
| 4.1 | Quoted `defaults` dict values | **STALE** | Live `WO:46-53`: `window_size max` **50** (not 500), `skip_max max` **250** (not 500), thresholds **`{min 0.40, max 0.75, default 0.50}`** (not `{0.15, 0.60, 0.25}`) |
| 4.1 | Quoted function body | **STALE** | Live version prints a warning on load failure (`WO:64-65`); the chapter's silent-fallback version does not |
| 4.2 | `distributed_config.json` structure block | **STALE** | Live values: `window_size {min 6, max 50, default 12}`, `skip_max {10, 250}`, both thresholds `{min 0.3, max 0.75, default 0.3}`. Live config also carries two `_calibration_note` / `_s172_note` provenance fields the chapter omits, which are the only in-repo record of *why* the floor is 6 |
| 4.3 | "Bounds: [0.15, 0.60], baseline 0.25" | **STALE** | Live `[0.30, 0.75]`, baseline **0.30** (`baselines/baseline_window_thresholds.json`) |
| 4.3 | "Target 1K-10K bidirectional survivors" | **ACCURATE** | `baselines/baseline_window_thresholds.json` → `expected_survivor_band: [1000, 10000]` |
| 4.3 | "The system is a behavioral fingerprint machine, NOT a filter. Low thresholds maximize seed discovery." | **ACCURATE** | Concordant with whitepaper §7 (loose thresholds preserve a learnable manifold). This is the one part of §4.3 a rewrite must **keep** |
| 4.3 | "High thresholds (0.72+) would eliminate candidates prematurely" | **CONTRADICTED-BY-CODE — code/config correct, chapter wrong** | Live ceiling is **0.75**, so 0.72 is inside the sampled range. `baselines/baseline_window_thresholds.json` notes: *"Known seed survives to threshold=0.75 — ceiling safe."* S148 empirical calibration falsified the chapter's assertion. See conflict **C-4** |
| 4.3 | Cross-reference `docs/THRESHOLD_GOVERNANCE.md` | **ACCURATE** | File present |

### §5 Scoring Functions

| § | claim | class | anchor / true state |
|---|---|---|---|
| 5.1 | `ScoringFunction` ABC with `score()` / `name()` | **ACCURATE** | `WO:264-274` — verbatim match |
| 5.2 | `BidirectionalCountScorer` returns `float(result.bidirectional_count)` | **ACCURATE** | `WO:285-289` — verbatim match |
| 5.2 | It is the default scorer | **ACCURATE** | `WO:477-478`; wired at `WOIF:2441` |
| 5.2 | Rationale paragraph | **ACCURATE** | Matches the live docstring `WO:278-284` |

### §6 Search Strategies

| § | claim | class | anchor / true state |
|---|---|---|---|
| 6.1 | `SearchStrategy` ABC signature | **STALE** | `WO:299-303` matches the *abstract* declaration, but the real calling convention adds four kwargs (`WO:484-487`). The ABC was never updated — which is precisely why C-3 below was never caught |
| 6.2 | `BayesianOptimization.__init__(n_initial=5)` | **STALE** | `WO:372` — live is `(n_initial=5, enable_pruning=False, n_parallel=1)`. Also, `WOIF:2398` constructs it with **`n_initial=3`**, not 5 |
| 6.2 | `search(self, objective_function, bounds, max_iterations, scorer)` | **STALE** | `WO:388-391` adds `resume_study`, `study_name`, `trse_context_file`, `trial_history_context` |
| 6.2 | Fallback to `RandomSearch` when Optuna unavailable | **ACCURATE** | `WO:406-407`. Note this positional-only call is the **only** way `RandomSearch` is reachable without crashing (see C-3) |
| 6.2 | "How Optuna TPE works" (3 bullets) | **ACCURATE** | Standard TPE description; consistent with `WOB` sampler setup |
| 6.3 | `RandomSearch` "Baseline" — quoted body | **CONTRADICTED-BY-CODE** | Body matches `WO:314-346`, but the strategy **cannot be invoked via `--strategy random`** — `TypeError`. See conflict **C-3** |
| 6.4 | `GridSearch` / `EvolutionarySearch` = "Placeholder, not used in integrated mode" | **ACCURATE (as written) but incomplete** | Both `search()` bodies return `{}` (`WO:358-360`, `WO:418-420`). The chapter correctly calls them placeholders — but §10.2 then documents them as live CLI modes, and `WOIF:2391-2400` really does map them. They crash before reaching the empty return. Contradiction is between §6.4 and §10.2, not with the code |

### §7 WindowOptimizer Class

| § | claim | class | anchor / true state |
|---|---|---|---|
| 7.1 | Constructor sets `coordinator`, `dataset_path`, `test_cache`, `test_configuration_func` | **ACCURATE** | `WO:438-442` — verbatim match |
| 7.2 | `test_configuration(config, seed_start=0, seed_count=10_000_000)` | **STALE** | `WO:444-446` adds `optuna_trial=None` (S119) |
| 7.2 | "This method is OVERRIDDEN by the integration layer to run real sieves" | **ACCURATE** | `WOIF:2389` `optimizer.test_configuration = test_config` |
| 7.2 | "Thresholds come from `config.forward_threshold` and `config.reverse_threshold`" | **ACCURATE (outcome) but materially incomplete** | Now true via `resolve_directional_threshold()` (`WOIF:2363-2364`, defined `WOIF:210-236`). **It was false from `2389b61` until `8a55a68`** — every trial silently ran 0.30/0.30. The chapter states the *intent* correctly and by luck now matches, but documents none of the machinery that makes it true: precedence `explicit > config > default` (`WOIF:214`), `is None` as the sole fallback trigger because **0.0 is a legitimate threshold** (`WOIF:216-219`), and `ThresholdResolutionError` fail-closed rather than inventing a value (`WOIF:231-235`). A doc that omits the invariant cannot protect it — this is the same failure mode as the skip-bound incident |
| 7.2 | Fallback placeholder returns zero-counts | **ACCURATE** | `WO:456-463` |
| 7.3 | `optimize(...)` signature | **STALE** | `WO:465-470` adds `resume_study`, `study_name`, `trse_context_file`, `trial_history_context` |
| 7.4 | `save_results()` body | **ACCURATE** | `WO:489-496` — verbatim match |

### §8 Bayesian Optimization Flow

| § | claim | class | anchor / true state |
|---|---|---|---|
| 8.1 | `run_bayesian_optimization()` has 7 parameters | **STALE** | `WO:525-557` — **32 parameters**. Missing from the chapter: `seed_start`, `resume_study`, `study_name`, `enable_pruning`, `n_parallel`, `trse_context_file`, `use_persistent_workers`, `use_zmq_sqlite`, `pwc_transport`, `pwc_min_workers`, `worker_pool_size`, `seed_cap_nvidia`, `seed_cap_amd`, 7 × `warm_start_*`, and 4 × `miner_*` |
| 8.2 | Coordinator init `MultiGPUCoordinator(config_file=..., resume_policy="restart")` | **ACCURATE** | `WO:615` |
| 8.2 | `coordinator.optimize_window(...)` call | **STALE** | `WO:649-670` passes 8 further kwargs the chapter's block omits |
| 8.2 | `optimal_config` dict keys | **STALE** | `WO:708-722` matches, but `WO:724-742` then merges 9 incremental-recovery fields (`status`, `completed_trials`, `total_trials`, `best_trial_number`, `best_value`, `best_bidirectional_count`, `last_updated`, `last_trial_number`, `last_trial_value`) plus `completed_at` — none documented |
| 8.2 | "Inject agent_metadata → Save → Split 80/20 → Return" | **ACCURATE** | `WO:746-761`, `:764-765`, `:800-819`, `:822` |
| 8.2 | Flow omits post-run stages | **STALE** | Undocumented: `[S140]` coverage write-back to `exhaustive_progress` (`WO:672-697`), `[S121]` TRSE `confirmed_windows` append capped at 50 entries (`WO:769-798`) |
| 8.3 | `--test-both-modes` prints and runs `java_lcg` + `java_lcg_hybrid` | **ACCURATE** | `WO:604-607` (print), `WOIF:1552-1553` (`prng_hybrid = f"{prng_base}_hybrid"`) |
| 8.3 | "Survivors tagged with `skip_mode` metadata" | **ACCURATE** | `WOIF:1513` `'skip_mode': 'constant'`, `WOIF:1617` `'skip_mode': 'variable'` |
| 8.3 | Silence on hybrid certification status | **CONTRADICTED-BY-CODE (governance)** | Hybrid certification is **blocked** until skip bounds reach the kernel (C-1), and PWC's hybrid path is quarantined — `persistent_worker_coordinator.py:176` `PWC_HYBRID_QUARANTINE_CODE = "PWC_HYBRID_THRESHOLD_CONTRACT_UNCERTIFIED"`. §8.3 presents variable-skip mode as a routine, fully-supported option |
| 8.3 | Silence on per-session TB ruling | **STALE** | See conflict **C-6** |
| — | Optuna search space (the §8 substance the chapter never states) | **STALE (absent)** | `WOB:420-442` samples exactly 7 dimensions: `window_size`, `offset`, `session_idx`, `skip_min`, `skip_max`, `forward_threshold`, `reverse_threshold`. `skip_max` floor is `max(skip_min, bounds.min_skip_max)` (`WOB:433`), which the chapter never mentions and which changes the shape of the space |

### §9 Run With Config Mode

| § | claim | class | anchor / true state |
|---|---|---|---|
| 9.1 | `run_with_config()` signature (7 params) | **STALE** | `WO:825-839` — 13 params; adds `use_persistent_workers`, `pwc_transport`, `seed_cap_amd`, `seed_cap_nvidia`, `worker_pool_size`, `min_workers` (S170-PARITY / PARITY-2). The S170 comments (`WO:897-909`) record that without these the mode *silently downgraded* to legacy SSH — exactly the drift class this chapter should have caught |
| 9.2 | Flow: load → coordinator → WindowConfig → accumulator → iterate → dedup → save → split | **ACCURATE (skeleton)** | `WO:867-1014` |
| 9.2 | Flow omits NPZ conversion | **STALE** | `WO:978-988` shells out to `convert_survivors_to_binary.py` and **raises `RuntimeError` on failure** ("Step 1 incomplete"). A hard release gate, undocumented |
| 9.2 | `deduplicate()` "keep highest score per seed" | **CONTRADICTED-BY-CODE (in effect)** | The function exists verbatim (`WO:951-958`), but for forward/reverse it now dedups **permanently empty lists** — see C-5. And for the canonical artifact, winner selection is no longer this function's job: it is the finalizer's frozen L2 key (`utils/run_finalizer.py:690`, `:714`), per `WOIF:2480-2488` which records that the legacy helper was **REMOVED, not bypassed** |
| 9.2 | `run_bidirectional_test(... forward_threshold=..., reverse_threshold=...)` | **STALE + live defect** | `WO:940-941` passes `config.get('forward_threshold', 0.72)` / `0.81`. **`0.81` exceeds the live governance ceiling of 0.75.** Worse, the `WindowConfig` built at `WO:912-918` does **not** receive these values, so it silently carries dataclass defaults 0.40/0.45 while the sibling kwargs say 0.72/0.81 — two threshold authorities in one call. `resolve_directional_threshold` gives `explicit` precedence (`WOIF:226-227`), so the kwargs win and `config`'s values are inert. Not a *silent* pin like defect #2, but it is a second authority for one quantity, which is the anti-pattern the S172 repair exists to eliminate |

### §10 CLI Interface

| § | claim | class | anchor / true state |
|---|---|---|---|
| 10.1 | Documented flags (13) | **STALE** | `WO:1031-1139` declares **31** flags. Undocumented: `--resume-study`, `--study-name`, `--enable-pruning`, `--n-parallel`, `--trse-context`, `--use-persistent-workers`, `--use-zmq-sqlite`, `--pwc-transport`, `--min-workers`, `--worker-pool-size`, `--seed-cap-nvidia`, `--seed-cap-amd`, `--seed-start`, 7 × `--warm-start-*`, `--use-range-miner`, `--miner-stripe-size`, `--miner-substripes`, `--miner-output-dir` |
| 10.1 | Backend mutex | **STALE (absent)** | `WO:1143-1154` rejects more than one of `--use-persistent-workers` / `--use-zmq-sqlite` / `--use-range-miner`. Undocumented |
| 10.1 | `--forward-threshold` / `--reverse-threshold` "Override Optuna optimization (0.15-0.60)" | **CONTRADICTED-BY-CODE** | **They override nothing.** Declared `WO:1063-1066`, never referenced again — see dead-dimension **D-4**. Additionally three mutually inconsistent bounds are in play: chapter says `0.15-0.60`, the live `--help` text says `0.5-0.95` / `0.6-0.98` (`WO:1064`, `:1066`), and the effective bounds are `0.30-0.75` |
| 10.2 | Mode decision tree, 4 strategies + config-file | **CONTRADICTED-BY-CODE** | The dispatch is literally as drawn (`WO:1157-1288`), but `random`, `grid`, `evolutionary` all crash with `TypeError` before doing work. See **C-3** |

### §11 Integration Layer

| § | claim | class | anchor / true state |
|---|---|---|---|
| 11.1 | `from window_optimizer_integration_final import add_window_optimizer_to_coordinator` | **STALE** | Not a module-level import; lazy inside each entry point to break a circular dependency (`WO:586`, `WO:853`), and it imports `run_bidirectional_test` too |
| 11.2 | Three responsibilities: monkey-patch `optimize_window`, run `run_bidirectional_test`, accumulate survivors | **STALE** | First two accurate (`WOIF:1668`, `:1679`, `:1134`). The third is **materially wrong now**: the layer no longer accumulates forward/reverse survivor objects (`WOIF:1529-1533`, `[S166-ACCUM]` — "RAM bomb"), and final artifact assembly moved to `utils.run_finalizer` (`WOIF:2490-2495`) |
| 11.3 | Integration sequence diagram | **STALE** | Omits the backend cascade and the finalizer stage; see §2.1 rows |

### §12 Output Files

| § | claim | class | anchor / true state |
|---|---|---|---|
| 12.1 | `optimal_window_config.json` = best params + agent_metadata | **ACCURATE** | `WO:764-765` |
| 12.1 | `window_optimization_results.json` = full trial history | **ACCURATE** | `WOIF:2450` |
| 12.1 | `bidirectional_survivors.json` = "Intersection survivors" | **SUPERSEDED** | `WOIF:2604-2631`: it is a **post-success summary** of the certified generation — generation IDs and sha256s, no seeds. Explicit in-source: *"It is NO LONGER the canonical Steps 2-6 input… Steps 2-6 consume the canonical NPZ"* |
| 12.1 | `forward_survivors.json` / `reverse_survivors.json` = sieve survivors | **SUPERSEDED** | `WOIF:2523-2532` writes `{"survivor_count": N, "note": "Full survivors omitted — objects not retained"}`. See **C-5** |
| 12.1 | `train_history.json` / `holdout_history.json` 80/20 | **ACCURATE** | `WO:811-819`, `WO:1003-1011` |
| 12.1 | Missing entry: the certified NPZ generation | **SUPERSEDED (absent)** | The actual Step-1 → Step-2 carrier is the finalizer's certified generation with `artifact_sha256` / `sidecar_sha256` / `parent_generation_id` (`WOIF:2596-2602`). The chapter's output table has no row for the one file that matters |
| 12.2 | `optimal_window_config.json` structure | **STALE** | Real; missing the 10 merged incremental fields (`WO:731-739`) |
| 12.3 | Survivor record: nested `"window_config": {…}` | **CONTRADICTED-BY-CODE** | Live record is **flat** — `window_size`, `offset`, `skip_min`, `skip_max` sit at top level in `metadata_base` (`WOIF:1505-1508`). Any consumer written against the chapter's nesting would fail |
| 12.3 | Field `"timestamp"` | **CONTRADICTED-BY-CODE** | **No `timestamp` key is produced.** `metadata_base` (`WOIF:1504-1527`) plus the append (`WOIF:1538-1544`) yield no such field |
| 12.3 | 7 intersection fields present | **ACCURATE** | `WOIF:1520-1526` |
| 12.3 | `forward_match_rate` / `reverse_match_rate` are per-seed | **ACCURATE** | `WOIF:1536-1541` — read per seed from `forward_map` / `reverse_map` |
| 12.3 | Field list is complete | **STALE** | Live records also carry `skip_range` (`WOIF:1509`), `sessions` (`:1510`), `prng_base` (`:1512`), `forward_count` (`:1516`), `reverse_count` (`:1517`), `intersection_count` (`:1520`) — six fields absent from the chapter's example |

### §13 Agent Metadata Injection

| § | claim | class | anchor / true state |
|---|---|---|---|
| 13.1 | Enables autonomous pipeline chaining by WATCHER | **ACCURATE** | `WO:746-761` |
| 13.2 | Quoted call with `forward_threshold: 0.72, reverse_threshold: 0.81` hardcoded | **STALE** | `WO:756-757` — live prefers the actual winner: `best_config.get('forward_threshold', 0.72)`. The literals survive only as fallbacks, and `0.81` is above the live ceiling of 0.75 |
| 13.2 | `confidence=min(0.95, results['best_score'] * 10)` | **STALE** | `WO:753` adds a zero-guard: `… if results['best_score'] > 0 else 0.5` |
| 13.3 | Metadata field table (7 fields) | **STALE (incomplete)** | `integration/metadata_writer.py:65-78` accepts 12: also `parent_run_id`, `success_criteria_met`, `retry_count`, `cluster_resources` |

### §14 Complete Method Reference

| § | claim | class | anchor / true state |
|---|---|---|---|
| 14.1 | Module-level functions (4) | **ACCURATE** | `WO:41`, `:525`, `:825`, `:1025` — all four exist with the stated roles |
| 14.2 | `WindowConfig` methods | **ACCURATE** | `WO:93`, `:98`, `:103` |
| 14.3 | `SearchBounds` methods | **STALE** | Omits `validate_baseline_in_bounds()` (`WO:163`) and `__post_init__` (`WO:153`) |
| 14.4 | `TestResult` properties | **STALE** | Omits `to_dict()` (`WO:248`) |
| 14.5 | `WindowOptimizer` methods | **STALE** | Signatures superseded per §7.2/§7.3 rows |
| 14.6 | `SearchStrategy` abstract methods | **STALE** | Per §6.1 row — the ABC no longer matches the call convention |

### §15 Dependencies Summary

| § | claim | class | anchor / true state |
|---|---|---|---|
| 15 | `coordinator.py` — Required ✅ | **ACCURATE** | `WO:503-507`; hard-exits `WO:579-582` |
| 15 | `window_optimizer_bayesian.py` — Optional ⚠️ | **ACCURATE** | `WO:511-516` guarded; but `--strategy bayesian` hard-exits without it (`WO:1159-1162`), so it is optional only for non-Bayesian modes — all of which are broken (C-3). **Effectively required** |
| 15 | `window_optimizer_integration_final.py` — Required ✅ | **ACCURATE** | `WO:592-595` |
| 15 | `integration.metadata_writer` — Optional ⚠️ | **CONTRADICTED-BY-CODE** | `WO:36` is an **unguarded module-level import**. Absent it, the module fails to import at all. **Hard dependency** |
| 15 | `distributed_config.json` — Optional ⚠️ | **STALE** | Optional for *search bounds* only (`WO:63-66` falls back). It is **mandatory** for the coordinator, which reads `nodes` from it (`coordinator.py:232`, `:239`) |
| 15 | Table omits `utils.run_finalizer` | **SUPERSEDED (absent)** | Now required for artifact publication (`WOIF:2490-2495`) |

### §16 Chapter Summary + Next Chapter

| § | claim | class | anchor / true state |
|---|---|---|---|
| 16 | Component line-count table (~100/~30/~100/~50/~100/~100/~100 ≈ 580) | **STALE** | Sums to ~580 against an actual 1306 lines. CLI alone is ~110 lines (`WO:1031-1139`) |
| 16 | "Key Insight" restated | **ACCURATE** | Consistent with §1.3 |
| — | "Next Chapter: Chapter 2 will cover `sieve_filter.py`" | **SUPERSEDED** | `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` exists but is a known **fragment**; Step 2's engine is being replaced by RANGE-MINER (S172) |

### Appendix A — Persistent Worker Call Chain (S130/S134/S135)

| claim | class | anchor / true state |
|---|---|---|
| `run_trial_persistent()` at `persistent_worker_coordinator.py:669` | **STALE** | Actual: **`:1612`**. Off by 943 lines |
| Zeus path `execute_local_sieve_job() → sieve_filter.py` | **CONTRADICTED-BY-CODE** | **No such function is defined anywhere in the repo.** The only occurrences are a comment (`persistent_worker_coordinator.py:17`) and the doc-patcher scripts that wrote this very appendix (`apply_s136_doc_updates.py:234`, `:464`; `apply_s146_doc_updates.py:248`). The appendix documents a call target that has never existed as code |
| Remote path `_dispatch_to_worker() → sieve_gpu_worker.py --persistent` | **ACCURATE** | `persistent_worker_coordinator.py:956` |
| "Invariant: PWC is STANDALONE. Zero changes to `coordinator.py`, `window_optimizer.py`, or `window_optimizer_integration_final.py`" | **CONTRADICTED-BY-CODE** | Long since false. `WO:617-626` sets `use_persistent_workers`, `use_zmq_sqlite`, `pwc_transport`, `pwc_min_workers`, `worker_pool_size` on the coordinator; `WO:1088-1100` declares the flags; `WOIF` gates on them. The claimed invariant is violated in all three named files |
| "Active study: `window_opt_1772507547.db` (21 trials as of S132)" | **UNVERIFIABLE** | **That DB does not exist** — not in `optuna_studies/` (68 DBs, newest `window_opt_1776999896.db`, Apr 2026) nor in `optuna_studies/archive/` (10 files). Trial count unverifiable |
| "Storage: JournalStorage (not SQLite)" | **CONTRADICTED-BY-CODE — code correct, chapter wrong** | `WOB:561`, `:580`, `:611` all use `sqlite:////home/michael/…`. `WOB:608-609` states explicitly: *"S125: always SQLite (JournalFileBackend removed…)"*. See conflict **C-7** |
| enable_pruning / n_parallel fix history (S116/S118/S123) | **ACCURATE** | Consistent with `WO:1080-1083`, `WO:660-661`, `WOIF:2398` |

### Appendix B / C — S146 Kernel Invariants (**duplicated verbatim**)

The identical section appears **twice**: chapter lines **1062-1096** and **1099-1133**.

| claim | class | anchor / true state |
|---|---|---|
| Forward hybrid tail `(…, threshold, a, c)` | **ACCURATE** | `SGW:259-268`; miner route `miner/range_miner_worker.py` `build_java_lcg` forward-hybrid branch |
| Reverse hybrid tail `(…, threshold, offset)`, `a,c` in-kernel | **ACCURATE** | `SGW:270-279`; `_reverse_hybrid_tail` (`miner/range_miner_worker.py:200-202`) |
| "These are not interchangeable" | **ACCURATE** | Distinct kernel entry points: `prng_registry.py:1007` `java_lcg_hybrid_multi_strategy_sieve`, `:3172` `java_lcg_hybrid_reverse_sieve` |
| Hybrid uses `phase2_threshold` for kernel + post-filter | **ACCURATE** | `SGW:257-258` (`hybrid_threshold`), `SGW:288` (`if rate >= hybrid_threshold`) |
| int32 casts including `cp.int32(skip_min)`, `cp.int32(skip_max)` | **CONTRADICTED-BY-CODE** | True for **constant-skip** families (`SGW:214`). **False for hybrids** — `SGW:259-279` rebuilds `kernel_args` from scratch and never re-adds them. The appendix asserts as an invariant the very thing that is broken. See **C-1 / D-1** |
| Count clamp `min(int(survivor_count_gpu[0].get()), n_seeds)` on both paths | **ACCURATE** | `SGW:281` (hybrid), `SGW:308` (non-hybrid) |
| Duplication itself | **doc defect** | Root cause identified: `apply_s146_doc_updates.py:48` guards with `if "S146" in content and label in content:` where `label` = `"CHAPTER_1 PWC S146 kernel invariants"` — a string **never written into the document**. The guard can never fire, so each run appends again |

### Markdown structural defects (mechanical, but they break rendering)

| defect | location | effect |
|---|---|---|
| Stray unmatched code fence | chapter line **885** (immediately after the fence at 884) | Fence count is **79 — odd**. Every code block from line 885 to end of file renders **inverted** (prose as code, code as prose) |
| Duplicated section | lines 1062-1096 ≡ 1099-1133 | ~35 redundant lines |

---

## 4. Dead-dimension inventory

A **dead dimension** is a parameter the system samples or accepts but that never reaches the
code claiming to consume it (skill §0.5). Per §0.4 the remedy is **wire-in, not removal** —
none of these are proposals to delete anything.

| id | parameter | sampled/declared at | dies at | consequence |
|---|---|---|---|---|
| **D-1** | `skip_min`, `skip_max` — **forward hybrid** (`java_lcg_hybrid`) | `WOB:429-434`; carried on `WindowConfig` (`WO:88-89`); miner payload → `range_miner_worker.py:776` → `BuildContext` `:871` | `_hybrid_prefix` (`range_miner_worker.py:177-193`) emits 13 args, **neither of them**. PWC route: `SGW:259-268` discards the generic prefix built at `SGW:214`. Kernel hardcodes `int expected_skip = 5` (`prng_registry.py:805, 885, 1027, 1159`) | Optuna tunes a knob wired to nothing. **Live on the certifying miner route.** = §2.7 defect #4, OPEN |
| **D-2** | `skip_min`, `skip_max` — **reverse hybrid** (`java_lcg_hybrid_reverse`) | same | `_reverse_hybrid_tail` (`range_miner_worker.py:200-202`) emits only `offset`; PWC `SGW:270-279` likewise | same class as D-1 |
| **D-3** | `offset` — **forward hybrid, `java_lcg` only** | `WOB:423-425` | `range_miner_worker.py` `build_java_lcg` forward-hybrid branch returns `_hybrid_prefix + [a, c]` with an explicit in-source note *"ABI-critical, NO offset (:1007)"*. PWC: `SGW:259-268` then `continue` at `:293`, skipping `kernel_args.append(cp.int32(offset))` at `SGW:304` | **Family-specific**: `build_lcg32`'s forward hybrid *does* pass `offset`. So the omission is a property of the `java_lcg` hybrid kernel signature — and `java_lcg` is the TFM target family, making this the consequential instance |
| **D-4** | `--forward-threshold`, `--reverse-threshold` (**NEW — not previously catalogued**) | declared `WO:1063-1066` | Immediately. `args.forward_threshold` / `args.reverse_threshold` are **never referenced** after `parse_args()` — verified by exhaustive `/bin/grep -n 'forward_threshold\|reverse_threshold' window_optimizer.py`: every subsequent hit is a `SearchBounds`/`WindowConfig`/warm-start field, none is `args.*` | An operator passing `--forward-threshold 0.6` gets **silent no-op**. The chapter advertises these as the Optuna override. Distinct from D-1…D-3 (operator-supplied, not sampler-supplied) but the same failure signature: a knob connected to nothing |

**Live-route status for TFM's target family `java_lcg`:**

| variant | `skip_min`/`skip_max` | `offset` |
|---|---|---|
| `java_lcg` (constant, forward) | ✅ reaches kernel | ✅ |
| `java_lcg_reverse` (constant) | ✅ | ✅ |
| `java_lcg_hybrid` (forward) | ❌ **dead** | ❌ **dead** |
| `java_lcg_hybrid_reverse` | ❌ **dead** | ✅ |

Constant-skip is fully wired; **the variable-skip path is where all the loss is.** This is
consistent with, and independently reconfirms, skill §2.7 #4 — reported here for chapter
scope only. Repair is the separately-briefed next deliverable; this audit proposes no fix.

---

## 5. Doc-vs-code conflicts, with intent assessment

### C-1 — `skip_min` / `skip_max` documented for variable PRNGs; hybrid kernels reject them

- **Chapter (§3.1):** `skip_min` = "Minimum skip for variable PRNGs", `skip_max` = "Maximum
  skip for variable PRNGs". Search space `skip_min` 0-10, `skip_max` 10-500.
- **Code:** 22/22 constant kernels declare `int skip_min, int skip_max`; **0/22 hybrid
  kernels do**. Hybrids hardcode `int expected_skip = 5` (`prng_registry.py:805, 885, 1027,
  1159`).
- **Which reflects intent: the chapter.** Evidence:
  1. **Physical model.** Per the *California State Lottery Daily & SuperLotto Plus Draw
     Procedures* (eff. 2021-06-09), one automatic pre-test session runs before an automatic
     Daily draw — with additional pre-test draws only on anomalies — and its outputs are
     never published (§V) [corrected 2026-08-01; previously "two pre-test draws … before
     every live draw", an Alpha misreading of language that applies to manual SuperLotto Plus
     equipment. Citation UNAVAILABLE: PDF not in repo]; equipment is re-selected per session
     (§II), and the evening
     session draws D3/D4/Fantasy 5/Daily Derby together. The observable sequence therefore
     has **real structural gaps of varying size**. Variable skip models a property of the
     data source; a fixed `expected_skip = 5` asserts the gaps are constant, which the
     source document contradicts.
  2. **Design symmetry.** The variable-skip mode exists *only* to relax the constant-skip
     assumption. A variable-skip kernel with a hardcoded stride is the constant-skip kernel
     with extra machinery — the mode has no reason to exist under the code's behaviour.
  3. **The full plumbing exists and is intact.** The values survive argparse, config,
     coordinator, ledger, manifest, payload, worker unpack and `BuildContext` (`:871`), then
     die one call before launch. Nobody builds eight hops of transport for a value that was
     never meant to arrive.
  4. **Governance treats it as a defect, not a design.** Hybrid certification is *blocked*
     pending skip-bound wire-in; approved sequence names it "WIRE IN, do not remove."
- **Verdict: the code is defective.** Remedy is wire-in (out of scope here).
- **Chapter action:** **keep the definition verbatim**, and add an explicit "why skip exists"
  paragraph plus a marked defect note so no future reader re-derives "remove it."

### C-2 — `offset` has three incompatible definitions

| source | definition |
|---|---|
| Chapter §3.1 | "Time offset from current draw" |
| `sieve_filter.py:174-186` (and `WOIF:262-264`) | **Head-relative array index**: `start = max(0, min(offset, n - window_size)); window = data[start:start+window_size]`. Docstring: *"return exactly `window_size` values starting at `offset`"* |
| `config_manifests/parameter_registry.json:38-43` | "Position offset — **advance seeds by `offset*(skip+1)`** before testing" — a *seed-advance*, not a data index |

- **What the code does:** in the Step-1 path, definition 2. `offset` indexes into the
  session-filtered draw array from the head and is clamped so the window always fits.
- **Complication:** `offset` is *also* passed as a kernel scalar (`SGW:304`, `:278`;
  `_offset_tail`, `range_miner_worker.py:196-197`), where the registry's seed-advance
  meaning would apply. **The same `config.offset` value feeds both a host-side array slice
  and a device-side seed advance.** Whether that is intended or a collision cannot be settled
  from the surfaces available and is flagged for the correction pass, not resolved here.
- **Which reflects intent:** **the chapter is closest to correct in spirit, the code is
  correct in mechanism, and `parameter_registry.json` is the outlier.** "Time offset from
  current draw" is a reasonable natural-language reading of "start the window N draws in"
  *if and only if* index 0 is the most recent draw; the chapter never states that
  precondition, which is the actual defect in the chapter. The registry's
  `offset*(skip+1)` formula describes a different quantity entirely and matches no Step-1
  call site found.
- **Chapter action:** replace the one-liner with the precise slice semantics, state the
  index orientation explicitly, and cross-reference the kernel-scalar use. Flag the registry
  entry as needing separate reconciliation.

### C-3 — Three of four documented strategies raise `TypeError`

- **Chapter:** §6.3 documents `RandomSearch` as a working baseline; §10.2 documents
  `--strategy random|grid|evolutionary` as live modes.
- **Code:** `WindowOptimizer.optimize` calls `strategy.search(..., resume_study=,
  study_name=, trse_context_file=, trial_history_context=)` (`WO:484-487`). Only
  `BayesianOptimization.search` accepts them (`WO:388-391`).
- **Execution proof** (live `inspect.signature`, this session, VM 101):
  ```
  RandomSearch           (self, objective_function, bounds, max_iterations, scorer)
  GridSearch             (self, objective_function, bounds, max_iterations, scorer)
  BayesianOptimization   (self, objective_function, bounds, max_iterations, scorer,
                          resume_study=False, study_name='', trse_context_file=...,
                          trial_history_context=None)
  EvolutionarySearch     (self, objective_function, bounds, max_iterations, scorer)

  accepts resume_study or **kwargs:
  RandomSearch=False  GridSearch=False  BayesianOptimization=True  EvolutionarySearch=False
  ```
- Aggravating: `WOIF:2402` `strategy_map.get(strategy_name, RandomSearch())` makes the
  **broken** `RandomSearch` the fallback for any unrecognised strategy name.
- Mitigating: the Optuna-unavailable fallback at `WO:406-407` calls `RandomSearch().search`
  **positionally**, so that one path works. `RandomSearch` is reachable *only* there.
- **Which reflects intent: the chapter.** All four strategies were clearly meant to run —
  §6.4 explicitly distinguishes "placeholder" (grid, evolutionary) from working
  (random, bayesian), a distinction that would be pointless if none ran. The kwargs were
  added incrementally to `BayesianOptimization` (S116/S121/S140b) and the sibling classes
  plus the `SearchStrategy` ABC (`WO:299-303`) were never updated. **This is code rot, and
  the stale ABC is why no signature check caught it.**
- **Chapter action:** document current reality (only `bayesian` is functional) and mark the
  other three as a known defect — do **not** delete them from the chapter.

### C-4 — "High thresholds (0.72+) would eliminate candidates prematurely"

- **Chapter §4.3** asserts it; live bounds are `[0.30, 0.75]`, so 0.72 is inside the
  sampled range.
- **Which reflects intent: the code/config.** `baselines/baseline_window_thresholds.json`
  records S148 empirical calibration (2026-03-19): *"Known seed survives to threshold=0.75 —
  ceiling safe."* The chapter's claim was an a-priori assumption that measurement falsified.
- **Important caveat — do not over-correct.** The *surrounding* §4.3 rationale ("behavioral
  fingerprint machine, NOT a filter"; low thresholds maximise discovery) is **correct and
  load-bearing**, and matches whitepaper §7: loose thresholds are a mathematical necessity
  to preserve a learnable manifold, because an exact sieve leaves no variance and hence no
  learning signal. Only the specific "0.72+" numeric claim is falsified. A rewrite that
  discards §4.3 wholesale would delete one of the few in-repo statements of why thresholds
  must stay loose and tunable.

### C-5 — `forward_survivors.json` / `reverse_survivors.json` are permanently empty

- **Chapter §9.2/§12.1** presents both as deduplicated survivor outputs.
- **Code:** `accumulator['forward']` and `accumulator['reverse']` are **never appended to**
  anywhere in `WOIF` — only `accumulator['bidirectional']` is (`WOIF:1018-1019`, `:1538`,
  `:1640`). `[S166-ACCUM]` (`WOIF:1529-1533`) replaced object retention with counters to
  stop a RAM bomb. In the Bayesian path the files are written as count-only summaries
  (`WOIF:2523-2532`). In `--config-file` mode, `WO:960-969` still dedups those empty lists
  and writes **`[]`** to both files, printing `"✅ Saved 0 forward survivors"`.
- **Which reflects intent: mixed, and this is the one row a correction pass must not
  flatten.** The RAM fix is deliberate and correct — full forward/reverse retention at 26-GPU
  scale is not viable, and the canonical NPZ carries what downstream needs. But
  `run_with_config` was **not updated to match**, so it emits an empty array while reporting
  success. The Bayesian path degraded honestly (writes a `note` explaining the omission);
  the config path degraded **silently**. The chapter is stale either way; the config-path
  behaviour is an actual defect worth a separate ticket.

### C-6 — Combined-session sampling vs. the per-session TB ruling

- **Chapter §3.2/§8.3** documents `['midday','evening']` as the first session option.
- **Code** still offers it (`WO:156-160`) and Optuna samples across all three
  (`WOB:426-428`).
- **Governance:** midday and evening use independently selected equipment, so there is no
  evidentiary basis for advancing one PRNG state through interleaved records. Combined-session
  sequential sieving is **non-certifying and prohibited by default**; production
  re-optimization is **per-session**.
- **Which reflects intent: the ruling, which post-dates both.** The chapter and the code are
  equally stale. Reporting only.
- **Chapter action:** document the ruling and mark the combined option non-certifying. The
  code gap (sampler can still select a prohibited mode) is outside this audit's scope but is
  flagged as a governance risk: an autonomous run can currently select a configuration that
  cannot be certified.

### C-7 — "Storage: JournalStorage (not SQLite)"

- **Chapter appendix line 1051** asserts JournalStorage.
- **Code:** SQLite throughout (`WOB:561`, `:580`, `:611`, `:616-623`), with an explicit
  removal note at `WOB:608-609`: *"S125: always SQLite (JournalFileBackend removed —
  n_parallel parallelism now owned by multiprocessing dispatcher in integration layer;
  n_jobs=1 here)."*
- **Which reflects intent: the code.** The comment records a deliberate, reasoned migration
  with its rationale. The appendix was written before it and never revisited. Straightforward
  doc staleness.

---

## 6. Prioritized correction list

Ordered by consequence — the potential for a good-faith reader to break something.

### P0 — corrections that prevent a component from being wrongly removed or a run from being wrongly trusted

1. **Harden the `skip_min`/`skip_max` definition (§3.1).** Keep the wording; add *why skip
   exists* (the pre-test-draw / multi-game / per-session-equipment physical model) and an
   explicit **DEFECT** callout that hybrid kernels currently ignore the values, with the
   standing rule that the fix is wire-in, not removal. This single edit is the chapter's
   highest-value content — it is the artifact that stopped the near-removal.
2. **Correct every numeric search bound (§3.2, §4.1, §4.2, §4.3, §10.1).** Replace with live
   values and state the precedence rule (`distributed_config.json` overrides code defaults,
   `WO:57-61`). Carry over the two `_note` provenance fields from `DC`, which are the only
   in-repo record of *why* the window floor is 6. **Recommend expressing bounds as "see
   `distributed_config.json`" plus a dated snapshot**, so the next drift is a stale date
   rather than a wrong number.
3. **Document `resolve_directional_threshold()` as an invariant (§7.2).** Precedence, `is
   None` as the sole fallback trigger (0.0 is legitimate), fail-closed
   `ThresholdResolutionError`, and the regression history (`3fdf434` fixed → `2389b61`
   silently reverted → `8a55a68` repaired). Cite `docs/THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md`
   rather than re-deriving.
4. **Record dead dimensions D-1…D-4 in the chapter itself**, each with its death hop. D-4
   (`--forward-threshold` / `--reverse-threshold` silently ignored) is new and currently
   documented as working — an operator can act on it today and get a silent no-op.
5. **Rewrite the output-file contract (§12.1, §12.3, §2.1).** State that the canonical
   Steps 2-6 input is the certified NPZ generation via `utils.run_finalizer`; demote
   `bidirectional_survivors.json` to post-success summary; mark forward/reverse files
   count-only. Add the flat-vs-nested record-shape correction and drop the non-existent
   `timestamp` field.

### P1 — corrections that prevent wasted debugging

6. **Fix the header** (version, both line counts, add `window_optimizer_bayesian.py`), and
   note that `docs/window_optimizer_integration_final.py` and `modules/window_optimizer.py`
   are stale duplicates that are **ruled to be left in place** — so the next reader neither
   edits them nor re-proposes deleting them.
7. **Document the backend cascade** (miner → PWC → ZMQ → legacy) and the argparse mutex
   (`WO:1143-1154`) in §2.1/§11. Without it §2.1 describes a path most production runs do
   not take.
8. **Mark `--strategy random|grid|evolutionary` as broken (C-3)** with the `TypeError` cause,
   and note the stale `SearchStrategy` ABC (`WO:299-303`) as the reason it went unnoticed.
9. **Bring §10.1 to all 31 flags**, correcting the three inconsistent threshold-bound
   figures to the single live pair.
10. **Correct the §8.3 governance picture** — per-session TB ruling, hybrid certification
    blocked, PWC hybrid quarantined (`persistent_worker_coordinator.py:176`).

### P2 — completeness and mechanical

11. **Refresh §8.1/§8.2/§9.1/§9.2 signatures and flows**, adding coverage write-back, the
    incremental merge, TRSE feedback and the NPZ-conversion gate.
12. **Add `validate_baseline_in_bounds()`** (TB mandate) to §3.2 and §14.3.
13. **Fold in the surviving parts of `CHAPTER_1_PATCH_S114.md`**, then mark that file
    superseded so it stops being read as current — its warm-start section describes deleted
    code.
14. **Fix Appendix A**: `run_trial_persistent` `:669 → :1612`; delete the non-existent
    `execute_local_sieve_job()`; retract the "zero changes" invariant; remove the missing
    study DB; correct JournalStorage → SQLite.
15. **Delete the duplicated S146 section** (lines 1099-1133) **and fix
    `apply_s146_doc_updates.py:48`**, whose idempotency guard tests for a `label` string
    that is never written to the file — otherwise the next run re-duplicates it.
16. **Repair the stray code fence at line 885.** Everything after it currently renders
    inverted.
17. **Update the §16 line-count table** and re-point "Next Chapter" at the RANGE-MINER
    replacement of Step 2.

### Flagged for separate tickets (not chapter corrections)

- **`run_with_config` writes `[]` survivor files while reporting success** (C-5, config-path
  half). Behavioural defect, not a doc defect.
- **`WO:940-941` passes `reverse_threshold` default `0.81`, above the live 0.75 ceiling**,
  and establishes a second threshold authority alongside `WindowConfig` (§9.2 row).
- **`WO:798` calls `logger.warning` but `window_optimizer.py` never imports `logging` or
  defines `logger`.** The TRSE feedback block's `except` handler would itself raise
  `NameError`, converting a "non-fatal" path into a crash. Found incidentally while auditing
  §8.2; **unverified at runtime** (would require triggering the exception), so recorded as a
  static observation, not a confirmed failure.
- **Optuna can still sample the combined-session mode that TB prohibits by default** (C-6).

---

## 7. Coverage table and completion sentinel

### Verification-integrity controls (VIR-1…6)

- **execution proof** — every ACCURATE/STALE/CONTRADICTED verdict carries a `file:line`
  anchor read in this session. Additionally, one live execution: `inspect.signature` over
  the four strategy classes in `~/venvs/torch` on VM 101 (C-3), plus live JSON parses of
  `distributed_config.json`, `agent_manifests/window_optimizer.json`,
  `config_manifests/parameter_registry.json`, `baselines/baseline_window_thresholds.json`,
  and a filesystem enumeration of `optuna_studies/`.
- **clean control (VIR-2)** — **stated explicitly**: the following were verified and found
  **correct**, and a rewrite must preserve them unchanged — §1.1 (both functions; 26 GPUs,
  arithmetically confirmed from `DC.nodes`), §1.3, §3.1 field names/order and all three
  methods, §3.3 `TestResult` incl. both formulas, §4.1 single-source-of-truth *mechanism*,
  §4.3 survivor-band target and the fingerprint-machine rationale, §5.1, §5.2 (all four
  claims), §6.2 Optuna-unavailable fallback and the TPE description, §7.1, §7.2
  override-by-integration-layer and the fallback placeholder, §7.4, §8.2 coordinator init
  and the inject/save/split tail, §8.3 hybrid naming and `skip_mode` tagging, §12.1
  `optimal_window_config.json` / `window_optimization_results.json` / train/holdout,
  §12.3 seven intersection fields and per-seed match rates, §13.1, §14.1, §14.2, §15 rows 1
  and 3, §16 Key Insight, and 5 of 7 Appendix B/C kernel invariants. **9 claims classified
  ACCURATE with no qualification; a further 12 accurate-in-part.** This is the evidence that
  the non-defective remainder was actually checked rather than skipped.
- **fault-injection control** — **n/a for a read-only documentation audit**, stated rather
  than omitted. No detector was built whose vacuity would need disproving; every verdict is
  a direct source read, not a detector output. The nearest equivalent applied: the C-3
  signature check was run against **all four** classes, so a positive (`BayesianOptimization`
  = True) and negatives (three = False) were both observed — the check demonstrably
  discriminates rather than returning a constant.
- **completion sentinel** — below.
- **unavailable-observer behavior** — one claim is `UNAVAILABLE`/UNVERIFIABLE (Appendix A
  active-study trial count); it is reported as such and **not** assumed correct. The S114
  patch's empirical survivor counts are likewise not treated as verified.
- **audit claim scope (VIR-6)** — **the claim is repo-scoped plus VM 101 filesystem-scoped,
  and nothing wider.**
  - **Searched surfaces:** `window_optimizer.py`, `window_optimizer_integration_final.py`,
    `window_optimizer_bayesian.py`, `sieve_filter.py`, `sieve_gpu_worker.py`,
    `persistent_worker_coordinator.py`, `coordinator.py`, `prng_registry.py`,
    `miner/range_miner_worker.py`, `integration/metadata_writer.py`,
    `distributed_config.json`, `agent_manifests/window_optimizer.json`,
    `config_manifests/parameter_registry.json`,
    `baselines/baseline_window_thresholds.json`, `optuna_studies/` (+ `archive/`),
    `apply_s146_doc_updates.py`, `apply_s136_doc_updates.py`,
    `docs/CHAPTER_1_WINDOW_OPTIMIZER.md`, `docs/CHAPTER_1_PATCH_S114.md`, and existence
    checks on five referenced docs (all present).
  - **Unavailable / not searched — no claim is made about these:**
    (a) **the rigs' deployed copies** — CT100 on `.122`/`.156`/`.164` was **not** contacted;
    a deployed `sieve_gpu_worker.py` could differ from the repo copy, and the repository is
    not the system;
    (b) **runtime behaviour** — no sieve, no GPU kernel, no WATCHER, no pipeline was run
    (prohibited and out of scope), so all kernel-ABI findings are **static reads of
    argument-builder code**, not observed launches;
    (c) **systemd units, cron, host config, uncommitted deployed files** — invisible to a
    repo-scoped audit;
    (d) **git history archaeology** was limited to commit messages already established in
    project facts; no `git log -S` sweep was run, so intent attributions in §5 rest on
    in-source comments, the whitepaper, `distributed_config.json` provenance notes and the
    draw-procedures document rather than on commit forensics;
    (e) **`docs/window_optimizer_integration_final.py` and `modules/window_optimizer.py`**
    were counted but **not audited** — they are ruled-stale duplicates, deliberately left
    alone.

### Coverage table

| chapter section | reached | verdict |
|---|---|---|
| Header | ✅ | STALE |
| §1.1 Overview — what it does | ✅ | ACCURATE |
| §1.2 Version history | ✅ | STALE + 1 CONTRADICTED |
| §1.3 Key insight | ✅ | ACCURATE |
| §1.4 Usage examples | ✅ | STALE |
| §2.1 Component hierarchy | ✅ | STALE / SUPERSEDED |
| §2.2 Execution flow | ✅ | STALE |
| §3.1 `WindowConfig` | ✅ | Mixed: 3 ACCURATE, 2 STALE, 2 CONTRADICTED |
| §3.2 `SearchBounds` | ✅ | Mixed: 2 ACCURATE, 5 STALE, 1 CONTRADICTED |
| §3.3 `TestResult` | ✅ | ACCURATE (1 omission) |
| §4.1 Single source of truth | ✅ | ACCURATE mechanism / STALE values |
| §4.2 `distributed_config.json` | ✅ | STALE |
| §4.3 Threshold philosophy | ✅ | 2 ACCURATE, 1 STALE, 1 CONTRADICTED |
| §5.1 Scoring base class | ✅ | ACCURATE |
| §5.2 `BidirectionalCountScorer` | ✅ | ACCURATE |
| §6.1 Strategy base class | ✅ | STALE |
| §6.2 `BayesianOptimization` | ✅ | STALE (2 ACCURATE sub-claims) |
| §6.3 `RandomSearch` | ✅ | CONTRADICTED |
| §6.4 Grid / Evolutionary | ✅ | ACCURATE (conflicts with §10.2) |
| §7.1 Constructor | ✅ | ACCURATE |
| §7.2 `test_configuration()` | ✅ | ACCURATE but materially incomplete |
| §7.3 `optimize()` | ✅ | STALE |
| §7.4 `save_results()` | ✅ | ACCURATE |
| §8.1 `run_bayesian_optimization()` | ✅ | STALE (7 of 32 params) |
| §8.2 Execution flow | ✅ | STALE |
| §8.3 Test both modes | ✅ | 2 ACCURATE, 1 STALE, 1 CONTRADICTED |
| §9.1 `run_with_config()` | ✅ | STALE |
| §9.2 Execution flow | ✅ | STALE + CONTRADICTED |
| §10.1 CLI arguments | ✅ | STALE + CONTRADICTED (D-4) |
| §10.2 Mode decision tree | ✅ | CONTRADICTED |
| §11.1 Key import | ✅ | STALE |
| §11.2 What it does | ✅ | STALE |
| §11.3 Integration flow | ✅ | STALE |
| §12.1 Output files | ✅ | SUPERSEDED |
| §12.2 Config structure | ✅ | STALE |
| §12.3 Survivor record | ✅ | 2 ACCURATE, 1 STALE, 2 CONTRADICTED |
| §13.1 Metadata purpose | ✅ | ACCURATE |
| §13.2 `inject_agent_metadata()` | ✅ | STALE |
| §13.3 Metadata fields | ✅ | STALE |
| §14.1 Module functions | ✅ | ACCURATE |
| §14.2 `WindowConfig` methods | ✅ | ACCURATE |
| §14.3 `SearchBounds` methods | ✅ | STALE |
| §14.4 `TestResult` properties | ✅ | STALE |
| §14.5 `WindowOptimizer` methods | ✅ | STALE |
| §14.6 `SearchStrategy` methods | ✅ | STALE |
| §15 Dependencies | ✅ | 3 ACCURATE, 1 STALE, 1 CONTRADICTED, 1 absent |
| §16 Chapter summary | ✅ | STALE |
| Next Chapter | ✅ | SUPERSEDED |
| Appendix A — PWC call chain | ✅ | 1 ACCURATE, 1 STALE, 2 CONTRADICTED, 1 UNVERIFIABLE |
| Appendix A — Optuna resume | ✅ | CONTRADICTED (SQLite) + UNVERIFIABLE (study DB) |
| Appendix A — pruning/n_parallel history | ✅ | ACCURATE |
| Appendix B — S146 invariants | ✅ | 5 ACCURATE, 1 CONTRADICTED |
| Appendix C — S146 duplicate | ✅ | doc defect (root cause identified) |
| Markdown structure | ✅ | 2 defects (stray fence 885; duplicate 1099-1133) |

**Sections reached: 54 / 54 (100%). `INCOMPLETE` count: 0.**

### Completion sentinel

```
AUDIT SCOPE      : docs/CHAPTER_1_WINDOW_OPTIMIZER.md vs live source @ 77dc629
SECTIONS REACHED : 54 / 54   (INCOMPLETE = 0)
CLAIMS CLASSIFIED: 41
  ACCURATE ............... 9
  STALE .................. 19
  SUPERSEDED ............. 5
  CONTRADICTED-BY-CODE ... 7   (2 where the CHAPTER is right: C-1, C-2)
  UNVERIFIABLE ........... 1
DEAD DIMENSIONS  : 4  (D-4 newly identified this session)
DOC DEFECTS      : 2 structural (stray fence L885; duplicated section L1099-1133)
CODE DEFECTS NOTED (not fixed, referred): 4

SENTINEL         : FAIL
```

**`FAIL` means:** the chapter does **not** accurately describe live source at `77dc629` and
must not be relied on as a reference until corrected. The audit itself terminated normally
with full coverage.

**No chapter text was modified. No code, test, or config was modified. Nothing was
committed.**

---

**STOP — held at the gate for Team Alpha review.** The correction pass is a separate,
separately-authorized deliverable to be written from these findings.
