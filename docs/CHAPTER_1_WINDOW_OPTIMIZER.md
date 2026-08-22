# Chapter 1: Window Optimizer (Step 1)

## PRNG Analysis Pipeline — Complete Operating Guide

**Chapter revision:** 3.1 — **a documentation-only number with no source counterpart.** The live
module docstring declares `Version: 2.0  Date: 2025-11-15` (`window_optimizer.py:5-6`); the string
`3.1` appears in no source file. Do not read it as a code version.

**Status:** **CLOSED at `81ef3f1`, 2026-08-02** — verified-and-bounded, not finished. See **§17**
for the closure statement, what remains open and where it is tracked, and the closure sentinel.
**Authority for the closure pass:** `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_1_AND_2_CLOSURE.md`
(REV1).

**Purpose:** Bayesian optimization of window parameters + survivor generation

**Live modules — Step 1 has THREE load-bearing files, not two:**

| module | role |
|---|---|
| `window_optimizer.py` | data structures, strategies, CLI, both entry points |
| `window_optimizer_integration_final.py` | backend cascade, `run_bidirectional_test`, survivor metadata, finalizer hand-off |
| `window_optimizer_bayesian.py` | **the module the old header omitted** — owns the entire Optuna search space, study storage and warm-start |

**Line counts are a moving target — re-derive, do not cite this table.** Measured on VM 101 at
commit `81ef3f1` (`wc -l`, 2026-08-02): `window_optimizer.py` **1753**,
`window_optimizer_integration_final.py` **2703**, `window_optimizer_bayesian.py` **1157**. The
pre-correction header claimed `~868 + ~595`. The correct action on any doubt is to run `wc -l`,
not to trust the figure above — the same discipline §4.1 applies to search bounds.

> **This table has already gone stale once inside this chapter's own lifetime, which is the point
> of the warning.** At the `40c3c83` correction pass the three files were **1592 / 2688 / 984**.
> Bounded Phase 6 (`d98298c`) added **+173** to `window_optimizer_bayesian.py` (the neutral
> `run_optimization` core, `describe_sampler`, `OptunaRandomSearch`, `SAMPLER_ENTRYPOINTS` —
> §8.1.2), and the Resolved Execution Set (`63e627f`) plus admission binding (`eff6616`) added
> **+161** to `window_optimizer.py`, all of it inside `main()`. **Every anchor in this chapter
> above `window_optimizer.py:1239` is unaffected by that growth and was re-verified unchanged;
> the anchors below it moved.** Where a citation could be made by function or symbol name instead
> of a line, it now is — that form survives edits and a line number does not.

> **Two stale duplicate copies exist and are RULED to be left in place** (2026-07-31):
> `docs/window_optimizer_integration_final.py` (1877 lines) and `modules/window_optimizer.py`
> (327 lines). They are **not** the live modules, they are not maintained, and they must be
> neither edited nor deleted — this note exists so the next reader does neither. The
> pre-correction `~595` figure most closely tracked the `docs/` duplicate, not the live file.
> **The live modules are the repo-root ones.**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Data Structures](#3-data-structures)
4. [Search Bounds Configuration](#4-search-bounds-configuration)
5. [Scoring Functions](#5-scoring-functions)
6. [Search Strategies](#6-search-strategies)
7. [WindowOptimizer Class](#7-windowoptimizer-class)
8. [Bayesian Optimization Flow](#8-bayesian-optimization-flow)
9. [Run With Config Mode](#9-run-with-config-mode)
10. [CLI Interface](#10-cli-interface)
11. [Integration Layer](#11-integration-layer)
12. [Output Files](#12-output-files)
13. [Agent Metadata Injection](#13-agent-metadata-injection)
14. [Complete Method Reference](#14-complete-method-reference)
15. [Dependencies Summary](#15-dependencies-summary)
16. [Chapter Summary](#16-chapter-summary)
17. [Closure statement](#17-closure-statement)

> Entries 15–17 were missing from this list; 15 and 16 predate the closure pass and 17 was added
> by it. **§17 is the last section in the file**, after the unnumbered appendices (Next Chapter,
> the Persistent Worker call chain and its Optuna resume notes, and the S146 kernel invariants) —
> the closure statement closes the chapter, appendices included.

---

## 1. Overview

### 1.1 What Window Optimizer Does

The Window Optimizer is **Step 1** of the 6-step pipeline. It performs two critical functions:

1. **Parameter Optimization:** Uses Bayesian optimization (Optuna TPE) to find optimal window parameters
2. **Survivor Generation:** Runs real sieves across all 26 GPUs and accumulates survivors

### 1.2 Version History

```
VERSION 3.1 (S104, Feb 2026):
- RESTORED: 7 intersection fields accidentally omitted in v3.0 rewrite
  (intersection_count, intersection_ratio, forward_only_count,
   reverse_only_count, survivor_overlap_ratio, bidirectional_selectivity,
   intersection_weight)
- These fields represent ~32% of ML feature importance (Chapter 6/11)

VERSION 3.0 (S103, Feb 2026):
- Per-seed match rates: forward_matches/reverse_matches now store per-seed
  values from GPU sieve kernel, not trial-level aggregates
- extract_survivor_records() preserves individual match_rate per seed
- Legacy alias extract_survivors_from_result() retained for compatibility
- convert_survivors_to_binary.py v3.1: maps to per-seed match rates
- NPZ percentage-based variance health check added

VERSION 2.0:
- --test-both-modes flag: Test constant AND variable skip patterns
- Survivors tagged with skip_mode metadata for ML feature engineering
- Backward compatible: defaults to constant skip only
```

### 1.3 Key Insight

**The optimizer doesn't run sieves directly.** It delegates to the integration layer (`window_optimizer_integration_final.py`) which coordinates with `coordinator.py` to run real sieves across all 26 GPUs.

### 1.4 Usage Examples

```bash
# Mode 1: Bayesian optimization (constant skip only)
python3 window_optimizer.py --strategy bayesian \
    --lottery-file lottery.json --trials 50

# Mode 2: Bayesian optimization (BOTH constant AND variable skip)
python3 window_optimizer.py --strategy bayesian \
    --lottery-file lottery.json --trials 50 --test-both-modes

# Mode 3: Run with existing optimal config
python3 window_optimizer.py --config-file optimal_window_config.json \
    --lottery-file lottery.json
```

---

## 2. Architecture

### 2.1 Component Hierarchy

```
window_optimizer.py
    │
    ├─→ WindowConfig, SearchBounds, TestResult (data structures)
    │
    ├─→ BayesianOptimization (strategy)
    │   └─→ window_optimizer_bayesian.py
    │       └─→ OptunaBayesianSearch (Optuna TPE)
    │
    ├─→ WindowOptimizer (main class)
    │   └─→ test_configuration()
    │       └─→ window_optimizer_integration_final.py
    │           └─→ run_bidirectional_test()
    │               └─→ coordinator.py
    │                   └─→ 26 GPUs execute sieves
    │
    └─→ Output files:
        ├─ CERTIFIED NPZ GENERATION  ← canonical Steps 2-6 input (utils.run_finalizer)
        ├─ optimal_window_config.json
        ├─ bidirectional_survivors.json   (post-success SUMMARY — no seeds)
        ├─ forward_survivors.json         (count-only stub)
        ├─ reverse_survivors.json         (count-only stub)
        ├─ train_history.json
        └─ holdout_history.json
```

Two corrections to the diagram above, both material:

1. **`coordinator.py` is one of FOUR backends, not the path.** `run_bidirectional_test` opens
   with a cascade — RANGE-MINER first (`window_optimizer_integration_final.py:1158-1168`),
   then PWC, then ZMQ, then the legacy coordinator leg drawn here. Most production runs do
   not take the drawn path. `window_optimizer.py:1429-1441` enforces a mutex: at most one of
   `--use-persistent-workers` / `--use-zmq-sqlite` / `--use-range-miner`.
2. **The artifact that matters is the certified NPZ generation**, assembled by
   `utils.run_finalizer` (`:2490-2495`). See §12.1 — the three `*_survivors.json` files are
   summaries and stubs, not survivor data.

### 2.2 Execution Flow

```
main()
    │
    ├─→ args.strategy == 'bayesian':
    │   └─→ run_bayesian_optimization()
    │       ├─→ MultiGPUCoordinator()
    │       ├─→ add_window_optimizer_to_coordinator()
    │       ├─→ coordinator.optimize_window()
    │       │   └─→ [N trials of real sieve execution]
    │       ├─→ inject_agent_metadata()
    │       └─→ Save outputs
    │
    └─→ args.config_file:
        └─→ run_with_config()
            ├─→ Load optimal config
            ├─→ MultiGPUCoordinator()
            ├─→ add_window_optimizer_to_coordinator()
            ├─→ [iterations × run_bidirectional_test()]
            ├─→ Deduplicate survivors
            └─→ Save outputs
```

---

## 3. Data Structures

### 3.1 WindowConfig

```python
@dataclass
class WindowConfig:
    """Complete window and skip configuration for sieve execution"""
    
    window_size: int           # Size of temporal window (number of draws)
    offset: int                # Time offset from current draw
    sessions: List[str]        # ['midday', 'evening'] or subset
    skip_min: int              # Minimum skip for variable PRNGs
    skip_max: int              # Maximum skip for variable PRNGs
    forward_threshold: float = 0.40   # Forward sieve threshold
    reverse_threshold: float = 0.45   # Reverse sieve threshold
```

Live source: `window_optimizer.py:100-131` (fields `:111-117`). The dataclass defaults shown above are the
*code* defaults; they are not the effective values a run uses. Effective thresholds are
resolved per trial by `resolve_directional_threshold()` (§7.2) and the sampled range comes
from `distributed_config.json` (§4.1). Do not read `0.40` / `0.45` as the operating point.

#### Why `skip_min` / `skip_max` exist — the physical model

> **This subsection is load-bearing.** In one session Team Alpha, Team Beta and Claude Code
> *independently* recommended removing `skip_min`/`skip_max` from variable-skip search,
> because no document any of them had read explained why skip exists. All three inferred
> intent from the current hybrid kernel signatures — which are the defect. The definitions
> above are kept **verbatim** for that reason.

The published draw sequence is **not** an uninterrupted PRNG output stream. Per the
*California State Lottery Daily & SuperLotto Plus Draw Procedures* (eff. 2021-06-09):

- **One automatic pre-test session runs before an automatic Daily draw** on the selected
  equipment (§V: Pre-Test via `[Start Draw Session]`). **Additional pre-test draws run only
  when an anomaly requires them.** Pre-test outputs are generated, verified and certified —
  and **never published**.
  *(Corrected 2026-08-01. This previously read "two pre-test draws run before every live
  draw" — an Alpha misreading; the "two test draws" language applies to **manual SuperLotto
  Plus equipment**. Only the count was wrong. Citation `UNAVAILABLE` — the PDF is not in the
  repo. See `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` §5.1.)*
- **Draw equipment is selected per session** by an RNG program, auditor-verified (§II).
  Midday and evening are separate sessions with separate equipment selection.
- The evening session draws **Daily 3, Daily 4, Fantasy 5 and Daily Derby together** — other
  games' outputs sit between the Daily 3 values that are observable.

**Therefore the observable sequence has real, structural gaps of unknown and varying size.**
Skip models those gaps. It is a **physical property of the data source**, not a tuning
convenience. A fixed stride asserts the gaps are constant, which the source document
contradicts.

| mode | assumption | kernels |
|---|---|---|
| constant skip | fixed stride *k* between observed outputs | 22 kernels, all declare `int skip_min, int skip_max` |
| variable skip (hybrid) | stride varies, e.g. `[5,5,3,7,5,5,8,4,5,5]` | 22 kernels, declare `skip_sequences` + `strategy_tolerances` |

#### DEFECT callout (Team Beta, verbatim)

```
DEFECT — current hybrid kernels do not execute the requested
skip_min/skip_max semantics and instead use a hard-coded stride.
Hybrid optimization results are non-certifying.
```

Anchors: hybrid kernels hardcode `int expected_skip = 5` (`prng_registry.py:805`, `:885`,
`:1027`, `:1159`) and neither hybrid signature declares `skip_min`/`skip_max`
(`prng_registry.py:1007-1013` forward, `:3172-3178` reverse). On the certifying miner route
the values survive argparse, config, coordinator, ledger, manifest, payload, worker unpack
and `BuildContext` (constructed at `miner/range_miner_worker.py:943`), then die one call before
launch in `_hybrid_prefix` (`:177-193`). Recorded as dead dimensions **D-1** / **D-2** (§3.1.1).

**Standing rule — the fix is WIRE-IN, not removal.** Absence of a working implementation is
not evidence of absent intent. The full transport chain exists and is intact; nobody builds
eight hops of plumbing for a value that was never meant to arrive. Before proposing to
remove, demote or simplify this parameter, cite this subsection and the draw-procedures
document. The purpose of this text is that no future reader re-derives "remove it."

**Methods:**

| Method | Returns | Purpose |
|--------|---------|---------|
| `__hash__()` | `int` | Make hashable for sets/dicts |
| `description()` | `str` | Human-readable, format string at `window_optimizer.py:127` |
| `to_dict()` | `Dict` | JSON serialization |

**Example** (values chosen inside the live bounds of §4.1 — the pre-correction example used
`window_size=512` and `0.25`, both of which are now **unreachable**: 512 exceeds the window
ceiling of 50 and 0.25 is below the threshold floor of 0.30):

```python
config = WindowConfig(
    window_size=12,
    offset=0,
    sessions=['midday'],
    skip_min=0,
    skip_max=16,
    forward_threshold=0.30,
    reverse_threshold=0.30
)
print(config.description())
# Output: W12_O0_midday_S0-16_FT0.3_RT0.3
```

#### 3.1.1 Dead dimensions D-1 … D-4

A **dead dimension** is a parameter the system samples or accepts but that never reaches the
code claiming to consume it. Each is a *defect to be wired in*, never a candidate for
removal.

| id | parameter | sampled / declared at | dies at | consequence |
|---|---|---|---|---|
| **D-1** | `skip_min`, `skip_max` — forward hybrid (`java_lcg_hybrid`) | `optuna_objective` (`window_optimizer_bayesian.py:516-521`); carried on `WindowConfig` (`window_optimizer.py:114-115`) | `_hybrid_prefix` (`miner/range_miner_worker.py:177-193`) emits 13 args, neither of them. PWC route `sieve_gpu_worker.py:259-268` discards the generic prefix. Kernel hardcodes `expected_skip = 5` | Optuna tunes a knob wired to nothing. **Live on the certifying miner route.** OPEN |
| **D-2** | `skip_min`, `skip_max` — reverse hybrid (`java_lcg_hybrid_reverse`) | same | `_reverse_hybrid_tail` (`miner/range_miner_worker.py:200-202`) emits only `offset`; `sieve_gpu_worker.py:270-279` likewise | same class as D-1. OPEN |
| **D-3** | `offset` — forward hybrid, `java_lcg` only | `optuna_objective` (`window_optimizer_bayesian.py:510-512`) | `build_java_lcg` forward-hybrid branch returns `_hybrid_prefix + [a, c]`, in-source note *"ABI-critical, NO offset"*; PWC skips `sieve_gpu_worker.py:304` via the `continue` at **`:298`** | Family-specific — `build_lcg32`'s forward hybrid *does* pass `offset`. `java_lcg` is the TFM target family, so this is the consequential instance. OPEN — and see §3.1.2, which settles what D-3 *is* and what it is not |
| **D-4** | `--forward-threshold`, `--reverse-threshold` | declared `window_optimizer.py:1285-1294` | immediately — `args.forward_threshold` / `args.reverse_threshold` were never referenced after `parse_args()` | Operator-facing. Was a **silent no-op** on a run reporting success. **CLOSED as a silent defect: the flags now fail closed** (§10.1) |

Constant-skip is fully wired on all four `java_lcg` variants; the variable-skip path is where
the loss is.

#### 3.1.2 `offset` — settles audit conflict C-2, and states what it does NOT settle

`docs/CHAPTER_1_AUDIT_v1.md` **C-2** found `offset` carrying **three incompatible definitions**,
could not settle the collision from Chapter 1's surfaces, and deferred it to Chapter 2. Chapter 2
§7 investigated it and recorded finding **F-4**. That finding is absorbed here.

| source | definition |
|---|---|
| this chapter's §3.1 field comment | *"time offset from current draw"* |
| host code | **head-relative array index** into the session-filtered draw list |
| `docs/instructions.txt:1181` | *"temporal alignment (**PRNG steps** to skip before sequence)"* |
| `config_manifests/parameter_registry.json:38-43` | advance seeds by **`offset*(skip+1)`** before testing |

**What the code did — both jobs, from one payload scalar.** AS AT THIS AUDIT'S ANCHOR, on the
certifying miner route the same `offset` was read twice out of `payload.get("offset", 0)`:

- **Host, as a data index** — `start = max(0, min(int(offset), n - window_size)); window =
  data[start:start + window_size]` in `load_residue_window` (`miner/range_miner_worker.py`).
- **Device, as a generator pre-advance** — `_offset_tail` emitted `ScalarArg(ctx.offset,
  "int32")`, consumed by the kernel as `for (o = 0; o < offset; o++) state = step(state)`
  (`prng_registry.py:974-976`).

> **SUBSEQUENT DISPOSITION — repair implemented by Window-Anchor Brief I; acceptance
> pending.** The scalar is split into `window_anchor` (host, record selection, validated
> against a derived domain and never clamped) and `generator_phase` (device, the existing
> kernel argument, pinned to 0 in v1). The kernel ABI is unchanged byte-for-byte and the
> retired key is hard-rejected, never mapped. Full description:
> `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` §7.2.1. **The finding above is the state at this
> audit's anchor and is not rewritten.**

**The coupling is self-consistent only at `skip = 0`.** When each observed draw consumes one PRNG
output, shifting the window by one record and pre-advancing by one step stay aligned. At
`skip = N` each observed draw consumes `N+1` outputs, so a one-record window shift *should*
correspond to a `(skip+1)`-step pre-advance — which is exactly the formula
`parameter_registry.json` states and the kernel does not implement. The kernel advances `offset`
steps flat.

> **Team Beta ruling — this settles C-2 as an OBSERVED INCONSISTENCY, not as the repair.**
> `parameter_registry.json` is **not** merely an outlier description, which is what the audit
> provisionally concluded; it describes the alignment the other two definitions would need in
> order to be jointly coherent at non-zero skip. But **`offset*(skip+1)` is not a general fix.**
> It is well defined only for **constant** skip. Under **variable** skip the per-record
> consumption varies by construction, so **no single `(skip+1)` multiplier exists** — the correct
> pre-advance for a window shift depends on the particular stride sequence, which is an *output*
> of the search, not an input to it (Chapter 2 §5.3).
>
> **F-4 therefore belongs inside the future hybrid input-semantics design, not a standalone
> arithmetic patch.** Applying a flat `offset*(skip+1)` in isolation would harden constant-skip
> semantics into a path whose hybrid half still has no defined input-bound meaning (D-1/D-2).
> The two must be decided together. **Described, not repaired.**

**Additionally — and this is D-3 above — forward hybrid kernels take no `offset` at all.** The
`java_lcg` forward-hybrid signature ends `float threshold, unsigned long long a, unsigned long
long c` with no offset parameter (`prng_registry.py:1007-1012`; builder comment
`miner/range_miner_worker.py:219`). On that path the window shifts and the generator does not
pre-advance whatsoever.

**The audit's own chapter action is discharged here.** The §3.1 field comment is a
natural-language reading of "start the window N draws in" that is correct *only if* index 0 is the
oldest retained draw — which is what `load_residue_window` implements: `offset` slices from the
**oldest** end (`docs/DAILY3_CONSUMER_CONTRACT_v1.md` §4.1–§4.4). The chapter never stated that
precondition; it now does. Full treatment: `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` §7.

### 3.2 SearchBounds

```python
@dataclass
class SearchBounds:
    """Search space boundaries for optimization"""
    
    # Window parameters
    min_window_size: int = 2
    max_window_size: int = 50     # S139: 500 -> 50
    min_offset: int = 0
    max_offset: int = 100
    
    # Skip parameters
    min_skip_min: int = 0
    max_skip_min: int = 10
    min_skip_max: int = 10
    max_skip_max: int = 250       # S139: 500 -> 250
    
    # Threshold bounds (LOW for discovery)
    min_forward_threshold: float = 0.40
    max_forward_threshold: float = 0.75
    min_reverse_threshold: float = 0.40
    max_reverse_threshold: float = 0.75
    
    # Defaults
    default_forward_threshold: float = 0.50
    default_reverse_threshold: float = 0.50
    
    # Session options
    session_options: List[List[str]] = None  # Auto-initialized
```

Live source: `window_optimizer.py:140-156`.

> **These are the CODE defaults, and they are NOT the effective search space.**
> `distributed_config.json` overrides them (§4.1), and it sets a *lower* threshold floor
> (`0.30`) and a *lower* threshold default (`0.30`) than the code does. Reading the numbers
> above as the operating bounds is the error this chapter previously made. **For effective
> values, see the extracted snapshot in §4.1.**

**Key Methods:**

| Method | Returns | Purpose |
|--------|---------|---------|
| `from_config(path)` | `SearchBounds` | Load from distributed_config.json (`window_optimizer.py:159-177`) |
| `__post_init__()` | — | Auto-initializes `session_options` when not supplied (`:179-186`) |
| `random_config()` | `WindowConfig` | Generate random config within bounds (`:224-237`) |
| `is_valid(config)` | `bool` | Validate config against bounds (`:239-247`) |
| `validate_baseline_in_bounds()` | `bool` | **Team Beta mandate** — the baseline must always be reachable inside the sampled bounds (`:189-222`) |

**`validate_baseline_in_bounds()` — the TB-mandated guard.** It reads
`baselines/baseline_window_thresholds.json` and checks three things against the *effective*
bounds: `forward_threshold` in range (`:208`), `reverse_threshold` in range (`:210`), and
`skip_max` not above the ceiling (`:212`). On any violation it prints each error and raises
`ValueError("Baseline thresholds outside search bounds - fix config before proceeding")` (`:219`).
The mandate it enforces is stated in the docstring: *"baseline must always be reachable"* — a
sampled space that cannot express the baseline makes the baseline unreproducible.

> **One fail-open condition, worth knowing before relying on it:** a **missing** baseline file is
> treated as *"no baseline = no constraint"* and returns `True` (`:195-197`). Absence of the file
> is therefore not the same as a passing check.

**Session Options (auto-initialized):**

```python
session_options = [
    ['midday', 'evening'],  # Both sessions
    ['midday'],             # Midday only
    ['evening']             # Evening only
]
```

### 3.3 TestResult

```python
@dataclass
class TestResult:
    """Result from testing a window configuration"""
    
    config: WindowConfig
    forward_count: int          # Survivors from forward sieve
    reverse_count: int          # Survivors from reverse sieve
    bidirectional_count: int    # Intersection (survived both)
    iteration: int              # Trial number
```

**Computed Properties:**

```python
@property
def precision(self) -> float:
    """Precision = bidirectional / forward"""
    return self.bidirectional_count / self.forward_count if self.forward_count > 0 else 0

@property
def recall(self) -> float:
    """Recall = bidirectional / reverse"""
    return self.bidirectional_count / self.reverse_count if self.reverse_count > 0 else 0
```

---

## 4. Search Bounds Configuration

### 4.1 Single Source of Truth — live authority, and a dated snapshot

**Precedence rule.** `distributed_config.json` → `search_bounds` **overrides** the code
defaults. The merge is a per-key `dict.update()` at `window_optimizer.py:85-87`, so config
wins key-by-key: a config block that supplies only `min` leaves `max` at the code default.

```python
def load_search_bounds_from_config(config_path: str = "distributed_config.json") -> dict:
    """Load search bounds from config file"""
    defaults = { ... }          # window_optimizer.py:72-79 — CODE defaults
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        bounds = config.get("search_bounds", {})
        for key in defaults:            # config values override defaults
            if key in bounds:
                defaults[key].update(bounds[key])
        return defaults
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"⚠️  Could not load search_bounds from {config_path}: {e}")
        print(f"   Using default bounds")     # window_optimizer.py:89-91
        return defaults
```

**Do not read numeric bounds out of this chapter.** Every numeric bound in the pre-correction
chapter was wrong — thresholds `[0.15, 0.60] default 0.25` against a live `[0.30, 0.75]
default 0.30`, the window ceiling 10× too large, the skip ceiling 2× too large. That is the
same class of error that produced the "~62 features" incident. Regenerate the snapshot below
instead of editing it by hand:

```bash
python3 scripts/extract_search_bounds_snapshot.py          # markdown block
python3 scripts/extract_search_bounds_snapshot.py --json   # machine-readable
```

**A date alone is insufficient provenance** — multiple code states share a date. The snapshot
therefore carries `repository_commit` (which tree) and `configuration_digest` (which
configuration bytes) as well.

<!-- BEGIN EXTRACTED BOUNDS SNAPSHOT — generated by scripts/extract_search_bounds_snapshot.py; do not hand-edit -->
```
Authority:
  distributed_config.json -> search_bounds (merged over code defaults by window_optimizer.load_search_bounds_from_config; config wins, window_optimizer.py:57-61)

Snapshot:
  generated_at         : 2026-08-02T07:37:03Z
  repository_commit    : 81ef3f11b0ca59f16cc85ee86776a4b3f976f150
  configuration_digest : sha256:6077bb1a6c7352bd21cbde736127394f464a082dbee6b098390e15dd1f2747cc
  status               : INFORMATIVE SNAPSHOT — NOT AUTHORITATIVE. Read the authority above for the binding values.

  extracted bounds:
    window_size        min=6, max=50, default=12
    offset             min=0, max=100
    skip_min           min=0, max=10
    skip_max           min=10, max=250
    forward_threshold  min=0.3, max=0.75, default=0.3
    reverse_threshold  min=0.3, max=0.75, default=0.3
```

**Provenance notes carried from `distributed_config.json`** (the only in-repo record of *why* these values are what they are):

- `window_size._calibration_note` — S148 Run-1 ruling: W12 is empirically preferred production baseline. Optuna still explores full [min,max] range. W12+T0.30 gives ~5 false fwd survivors/200k vs ~272 at W8/T0.25.
- `window_size._s172_note` — S172 (2026-04-30): min raised from 2 to 6 per TB ruling. W=2/3 produces ~39%/53% survivor rate by chance alone, regardless of threshold. Threshold bounds intentionally PRESERVED so Optuna can continue optimizing across [min, max].
<!-- END EXTRACTED BOUNDS SNAPSHOT -->

> **Regenerated at closure, and the values did not move.** The block above was re-emitted by
> `scripts/extract_search_bounds_snapshot.py` at `81ef3f1` and spliced in by script — it was not
> transcribed. `repository_commit` advanced `0c47fe3…` → `81ef3f1…`, and
> `configuration_digest` is **byte-identical** to the previous snapshot
> (`sha256:6077bb1a…2747cc`). **That identity is the finding, not a formality:** the six
> `search_bounds` entries have not changed since the correction pass, so every numeric bound this
> chapter defers to §4.1 for is still the operating value. A changed digest with unchanged
> printed numbers would have meant a `_note` or an unread key had moved.

> **Defect in the snapshot generator, flagged not fixed.** The `Authority:` line above is emitted
> by `scripts/extract_search_bounds_snapshot.py` with a **hardcoded, now-stale** anchor
> `window_optimizer.py:57-61` for the merge rule (generator `:21`, `:102`, `:124`). The live merge
> loop is at **`:85-87`** — the anchor drifted when `window_optimizer.py` grew. The block above is
> machine-generated and marked *do not hand-edit*, so it has deliberately been left alone: the
> repair belongs in the generator, and is **out of scope for this documentation tranche**
> (the only script edit authorized here is `apply_s146_doc_updates.py`). Values in the snapshot
> are unaffected — only the citation is stale.

### 4.2 distributed_config.json Structure

The `search_bounds` block is the authority. Its **shape** is stable; its **values** are the
snapshot in §4.1 and must be read from there or from the file, never from a transcription.

```json
{
    "search_bounds": {
        "window_size":       {"min": …, "max": …, "default": …,
                              "_calibration_note": "…", "_s172_note": "…"},
        "offset":            {"min": …, "max": …},
        "skip_min":          {"min": …, "max": …},
        "skip_max":          {"min": …, "max": …},
        "forward_threshold": {"min": …, "max": …, "default": …},
        "reverse_threshold": {"min": …, "max": …, "default": …}
    }
}
```

The two `_note` keys are not decoration — they are the only in-repo record of *why* the
window floor is 6, and they are carried into the §4.1 snapshot for that reason. Do not strip
them when editing the config.

### 4.3 Threshold Philosophy

**CRITICAL INSIGHT:** target **1K–10K** bidirectional survivors
(`baselines/baseline_window_thresholds.json` → `expected_survivor_band`). For the sampled
range and baseline, see the §4.1 snapshot — not a number quoted here. See
`docs/THRESHOLD_GOVERNANCE.md`.

```
The system is a behavioral fingerprint machine, NOT a filter.
Low thresholds maximize seed discovery.
Bidirectional intersection handles the actual filtering.
```

This rationale is **correct and load-bearing**, and matches whitepaper §7: an exact sieve
eliminates all variance, leaving `{s*}` with no ranking, no gradients and **no learning
signal**. Loose thresholds deliberately admit a *manifold* of near-consistent seeds sharing
structured deviations that ML can learn to rank. Loose thresholds are a mathematical
necessity, not sloppiness. **A rewrite must keep this paragraph.**

**Correction — one claim here was falsified by measurement.** The pre-correction chapter
added *"High thresholds (0.72+) would eliminate candidates prematurely."* That was an
a-priori assumption. The live ceiling is `0.75`, so `0.72` is inside the sampled range, and
`baselines/baseline_window_thresholds.json` records the S148 empirical calibration
(2026-03-19): *"Known seed survives to threshold=0.75 — ceiling safe."* The claim is removed;
the surrounding rationale is not.

---

## 5. Scoring Functions

### 5.1 Base Class

```python
class ScoringFunction(ABC):
    """Base class for scoring functions"""
    
    @abstractmethod
    def score(self, result: TestResult) -> float:
        """Score a test result. Higher is better."""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return name of scoring function"""
        pass
```

### 5.2 BidirectionalCountScorer (Default)

```python
class BidirectionalCountScorer(ScoringFunction):
    """Score = count of bidirectional survivors"""
    
    def score(self, result: TestResult) -> float:
        return float(result.bidirectional_count)
    
    def name(self) -> str:
        return "bidirectional_count"
```

**Rationale:** Seeds that survive BOTH forward AND reverse sieves are more likely to be temporally stable and produce good predictions.

---

## 6. Search Strategies

### 6.1 Strategy Base Class

```python
class SearchStrategy(ABC):
    """Base class for search strategies"""
    
    @abstractmethod
    def search(self,
               objective_function: Callable[[WindowConfig], TestResult],
               bounds: SearchBounds,
               max_iterations: int,
               scorer: ScoringFunction) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def name(self) -> str:
        pass
```

Live source: `window_optimizer.py:339-361`.

> **This ABC is STALE, and its staleness is load-bearing.** The abstract signature above is the
> pre-S116 four-positional convention. The real convention adds four forwarded kwargs
> (`:334`, `:622-625`). Because the ABC was never updated, no signature check had a correct
> reference, which is why three of the four strategies were uncallable for ~4½ months without
> anyone noticing (§10.1). The live docstring records this at `:343-348`, and
> `strategy_contract_gap()` (`:518`) now checks the **concrete classes** instead of the ABC.

### 6.2 BayesianOptimization (Recommended)

```python
class BayesianOptimization(SearchStrategy):
    """Bayesian optimization using Optuna TPE"""
    
    def __init__(self, n_initial=5):
        self.n_initial = n_initial  # Startup trials before TPE kicks in
        
        if BAYESIAN_AVAILABLE:
            from window_optimizer_bayesian import OptunaBayesianSearch
            self.optuna_search = OptunaBayesianSearch(
                n_startup_trials=n_initial, 
                seed=None
            )
    
    def search(self, objective_function, bounds, max_iterations, scorer):
        if self.optuna_search:
            return self.optuna_search.search(
                objective_function, bounds, max_iterations, scorer
            )
        else:
            # Fallback to random search
            return RandomSearch().search(
                objective_function, bounds, max_iterations, scorer
            )
```

**How Optuna TPE Works:**

1. First `n_initial` trials: Random sampling
2. Subsequent trials: TPE (Tree-structured Parzen Estimator) suggests parameters
3. TPE models P(x|y) for good and bad trials, suggests x that maximizes EI

### 6.3 RandomSearch (Baseline)

```python
class RandomSearch(SearchStrategy):
    """Random search - samples configs uniformly"""
    
    def search(self, objective_function, bounds, max_iterations, scorer):
        results = []
        best_result = None
        best_score = float('-inf')
        
        for i in range(max_iterations):
            config = bounds.random_config()  # Uniform random
            result = objective_function(config)
            result.iteration = i
            score = scorer.score(result)
            results.append(result)
            
            if score > best_score:
                best_score = score
                best_result = result
                print(f"✨ NEW BEST [{i+1}/{max_iterations}]")
        
        return {
            'strategy': 'random_search',
            'best_config': best_result.config.to_dict(),
            'best_result': best_result.to_dict(),
            'best_score': best_score,
            'all_results': [r.to_dict() for r in results],
            'iterations': len(results)
        }
```

### 6.4 Other Strategies (Placeholders)

| Strategy | Status | Notes |
|----------|--------|-------|
| `GridSearch` | Placeholder — `GridSearch.search` is `return {}` (`window_optimizer.py:410-412`) | Not used in integrated mode |
| `EvolutionarySearch` | Placeholder — `EvolutionarySearch.search` is `return {}` (`:484-486`) | Not used in integrated mode |

> **"Placeholder" records a deletion, not a design decision.** Both classes had complete working
> bodies — `GridSearch` a four-deep nested loop over windows × offsets × sessions × skip ranges —
> until they were cut to `# Placeholder … return {}` between 2025-11-14 and 2025-11-15. The
> documented design was **four Optuna samplers**. See `docs/STRATEGY_ORIGIN_AUDIT.md` and §10.1;
> the repair is pending a Team Beta ruling and is **not** specified here. Do not read this table
> as evidence that the two were never meant to work.

---

## 7. WindowOptimizer Class

### 7.1 Constructor

```python
class WindowOptimizer:
    """Main optimizer that coordinates the search process"""
    
    def __init__(self, coordinator, dataset_path: str):
        self.coordinator = coordinator
        self.dataset_path = dataset_path
        self.test_cache = {}
        self.test_configuration_func = None  # Set by integration layer
```

### 7.2 test_configuration()

```python
def test_configuration(self, config: WindowConfig, 
                       seed_start: int = 0,
                       seed_count: int = 10_000_000,
                       optuna_trial=None) -> TestResult:      # S119
    """
    Test a configuration.
    
    This method is OVERRIDDEN by the integration layer to run real sieves.
    Thresholds come from config.forward_threshold and config.reverse_threshold.
    """
    if self.test_configuration_func:
        return self.test_configuration_func(config, seed_start, seed_count,
                                            optuna_trial=optuna_trial)
    
    # Fallback placeholder (never called in integrated mode)
    return TestResult(config=config, forward_count=0, 
                     reverse_count=0, bidirectional_count=0, iteration=0)
```

Live source: `window_optimizer.py:582-601`. Override: the `test_config` closure inside
`optimize_window`, bound at `window_optimizer_integration_final.py:2405`
(`optimizer.test_configuration = test_config`; the closure itself is `:2364-2403`).

#### 7.2.1 INVARIANT — `resolve_directional_threshold()` is the single threshold authority

The sentence *"thresholds come from `config.forward_threshold` and `config.reverse_threshold`"*
is **true only because of a specific mechanism, and it has been false before.** A document
that states the outcome without the invariant cannot protect it — that is the same failure
mode as the skip-bound incident (§3.1).

**The authority** is `resolve_directional_threshold()`,
`window_optimizer_integration_final.py:214-240` (its `ThresholdResolutionError` is declared
immediately above at `:210-211`). It is the *only* place a directional threshold is resolved.
Do not add a second resolution path. **Cite it by name** — the function has already moved once
(`:210` → `:214`) without changing at all.

| rule | why |
|---|---|
| Precedence **explicit > config > default** | one resolution, in the parent, never reinterpreted downstream |
| **`is None` is the SOLE fallback trigger** | **`0.0` is a legitimate threshold.** A truthiness test (`getattr(...) or default`) silently replaces it — the shape `s172_threshold_patch.py` FIX 2 used, and deliberately not reused here |
| **Fail closed:** raises `ThresholdResolutionError` when nothing resolves | it refuses to invent a value. Never substitute a constant, never clamp into range |
| Record **requested / payload / effective** separately | the effective value is read back off the real executor (`miner/range_miner_worker.py:784`, `:858-863`, `:913`), so provenance is observed, not asserted |

**Regression history — read this before touching the file.** Full trace:
`docs/THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md` (cited, not re-derived here).

| commit | date | event |
|---|---|---|
| `3fdf434` | 2026-04-30 | Optuna threshold-drop bug **fixed** |
| `2389b61` | 2026-07-07 | **silently reverted** — a Phase-0 PRNG-encoding commit rewrote `window_optimizer_integration_final.py` from a pre-fix copy. The commit message never mentions thresholds. A stale-copy overwrite |
| `8a55a68` | 2026-07-31 | **repaired**, both routes, via the single resolver above |

Between `2389b61` and `8a55a68` **every trial ran at the configured default `0.30/0.30`**
while the study recorded the sampled suggestion. Treat every threshold value recorded in
that window — study DBs, `step1_trial_history`, `optimal_window_config.json` — as
**non-executed**.

> **Gate design consequence.** `2389b61` reverted the fix by replacing the whole block, so a
> text-anchor check would have gone green. Any regression gate on this invariant must
> **execute the live call site**, not match text against it. That is what
> `tests/test_s172_threshold_propagation.py` and `tests/test_s172_phase5_d6_threshold_path.py`
> do.

### 7.3 optimize()

```python
def optimize(self, strategy: SearchStrategy, bounds: SearchBounds,
             max_iterations: int = 50, scorer: ScoringFunction = None,
             seed_start: int = 0, seed_count: int = 10_000_000) -> Dict[str, Any]:
    """
    Run optimization using provided strategy.
    
    The strategy calls self.test_configuration() for each trial,
    which triggers real sieve execution via the integration layer.
    """
    if scorer is None:
        scorer = BidirectionalCountScorer()
    
    def objective(config: WindowConfig) -> TestResult:
        return self.test_configuration(config, seed_start, seed_count)
    
    return strategy.search(objective, bounds, max_iterations, scorer)
```

### 7.4 save_results()

```python
def save_results(self, results: Dict[str, Any], output_path: str):
    """Save optimization results to JSON"""
    output_dir = Path(output_path).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
```

---

## 8. Bayesian Optimization Flow

### 8.1 run_bayesian_optimization()

This is the main entry point for Bayesian optimization mode.

**The live signature takes 31 parameters** (`window_optimizer.py:663-695`, count verified by AST
this session). The seven below are the original set; the pre-correction chapter listed only these
and is not a usable reference for calling the function:

```python
def run_bayesian_optimization(
    lottery_file: str,
    trials: int,
    output_config: str,
    seed_count: int = 10_000_000,
    prng_type: str = 'java_lcg',
    test_both_modes: bool = False,    # NEW in V2.0
    strategy_name: str = 'bayesian',
    ...                               # + 24 more — see the table below
) -> Dict[str, Any]:
```

The other 24, grouped by the change that introduced them:

| group | parameters |
|---|---|
| coverage tracker (S140) | `seed_start` |
| study resume (S115/S116) | `resume_study`, `study_name` |
| pruning / parallelism (S115) | `enable_pruning`, `n_parallel` |
| TRSE (S121) | `trse_context_file` |
| backends (S134/S158D/S162) | `use_persistent_workers`, `use_zmq_sqlite`, `pwc_transport`, `pwc_min_workers`, `worker_pool_size` |
| seed caps (S137) | `seed_cap_nvidia`, `seed_cap_amd` |
| warm start (S166) | `warm_start_window`, `warm_start_offset`, `warm_start_skip_min`, `warm_start_skip_max`, `warm_start_fwd_thresh`, `warm_start_rev_thresh`, `warm_start_session_idx` |
| RANGE-MINER (S172) | `use_range_miner`, `miner_stripe_size`, `miner_substripes`, `miner_output_dir` |

#### 8.1.1 The Optuna search space — seven sampled dimensions

The chapter never stated the search space, which is the substance of §8. Optuna samples exactly
seven dimensions, all inside the **`optuna_objective` closure**
(`window_optimizer_bayesian.py:504-572`, sampling block `:507-529`):

| dimension | anchor | note |
|---|---|---|
| `window_size` | `:507-509` | bounds per §4.1 snapshot |
| `offset` | `:510-512` | **dead on the `java_lcg` forward hybrid** — D-3, §3.1.1; semantics settled in §3.1.2 |
| `session_idx` | `:513-515` | indexes `session_options`; applied at `:535`. See §8.3.1 — the combined option is prohibited by default |
| `skip_min` | `:516-518` | **dead on both hybrid directions** — D-1/D-2 |
| `skip_max` | `:519-521` | floor is `max(skip_min, bounds.min_skip_max)` (`:520`), so the space is **not** a plain rectangle — the two skip dimensions are coupled |
| `forward_threshold` | `:524-526` | reaches the kernel via `resolve_directional_threshold()` (§7.2.1) |
| `reverse_threshold` | `:527-529` | as above |

> **These anchors moved and the space did not.** At the previous correction pass the same seven
> dimensions were at `:420-441`. Bounded Phase 6 (`d98298c`) lifted the study body out of
> `search()` into the sampler-neutral `run_optimization` (§8.1.2), carrying `optuna_objective`
> ~87 lines down with it. **The sampled dimensions, their bounds, the `skip_max` coupling and the
> objective are unchanged** — verified line-by-line this session. `:420-441` now points at the
> thin TPE entrypoint, which is why a line-only citation to this block is a trap; **cite
> `optuna_objective` by name.**

#### 8.1.2 The sampler-neutral core — `run_optimization(..., sampler, sampler_metadata)`

**This section postdates the rest of the chapter.** Bounded Phase 6 (`d98298c`, certified and
closed 2026-08-02) extracted TPE-sampler construction *out* of the study body. The chapter as
corrected described a single Optuna path that always built a `TPESampler`; that is no longer the
shape of the code.

**The core** is `OptunaBayesianSearch.run_optimization` (`window_optimizer_bayesian.py:457-827`).
Live signature, read by `inspect.signature` this session:

```
run_optimization(self, objective_function, bounds, max_iterations, scorer, *,
                 sampler, sampler_metadata: Dict,
                 resume_study: bool = False, study_name: str = '',
                 trse_context_file: str = 'trse_context.json',
                 trial_history_context: dict = None) -> Dict
```

**`sampler` and `sampler_metadata` sit after the bare `*`, and neither has a default.** They are
**required and keyword-only**. That is deliberate, and the reason is recorded in-source at
`:478-483`: so *"a caller cannot get TPE by omission and then report the run as something else"*,
and because *"an unlabelled run is not a control."* Nothing in the body names, assumes or prefers
a sampler class (`:470-471`). **Do not add a default to either argument** — a default would
restore exactly the mislabelling the extraction exists to prevent.

**The two entrypoints are thin wrappers over that one body:**

| entrypoint | anchor | sampler it builds | strategy label |
|---|---|---|---|
| `OptunaBayesianSearch.search` | `:420-452` | `TPESampler(n_startup_trials=…, seed=…, multivariate=True)` (`:438-442`) | `optuna_bayesian` |
| `OptunaRandomSearch.search` | `:885-904` | `RandomSampler(seed=…)` (`:894-895`) | `optuna_random_control` |

Both call `describe_sampler()` (`:834-861`) to build the binding record — sampler class, sampler
module, Optuna version, seed, strategy label — which travels into the result dict. The `strategy`
key now reports **the sampler that actually chose the points**, instead of a hardcoded
`'optuna_bayesian'`.

`OptunaRandomSearch` (`:864-904`) is the **operator-selected RandomSampler control arm**: the same
search space, objective, warm-start rule, pruner, storage and result shape as the TPE arm,
differing in exactly one variable. Its docstring (`:870-872`) is explicit that the pre-existing
`window_optimizer.RandomSearch` (§6.3) is **not** a control — that class samples with Python's
`random` module outside Optuna entirely and shares none of the above. `n_startup_trials` is
inherited but meaningless for `RandomSampler` and is recorded as `None` rather than carried over
(`:876-878`), so the record cannot imply a warm-up that did not happen.

> **`SAMPLER_ENTRYPOINTS` is deliberately NOT wired to anything.** The registry
> (`window_optimizer_bayesian.py:909-912`) maps the two strategy labels to their classes, and the
> comment immediately above it (`:907-908`) states the constraint in source: *"Deliberately NOT
> wired to any advisor, WATCHER policy or `strategy_recommendation.json`."* Verified by search
> this session: the **only** consumer anywhere in the tree is the comparison harness
> `tests/phase6/sampler_control_arm.py` (`:251`, `:417`). No advisor, no WATCHER policy, no
> manifest and no `strategy_recommendation.json` reader touches it.
>
> **Autonomous sampler selection is reserved authority (Team Beta)**, alongside sieve strategy,
> feature engineering and the meta-optimizer search space. `OptunaRandomSearch` exists to be
> chosen *deliberately by an operator*; nothing in the codebase selects it on its own, and the
> class docstring says so (`:880-882`). **A future change that lets an advisor pick a sampler is
> a governance change, not a wiring change.**

> **Two limits on this section, stated rather than omitted.**
> 1. **Sampler provenance is unverified.** `run_optimization` trusts the caller-supplied
>    `sampler_class` / `sampler_module` / `optuna_version` in `sampler_metadata` and does not
>    check them against the actual object. The two existing wrappers are correctly labelled, so
>    nothing submitted is invalidated — but **a fail-before-study guard is required before direct
>    use of the neutral core or registration of a third sampler.** Open, tracked in the
>    project-facts skill §2.9.
> 2. **TPE remains the production default by status quo.** The RandomSampler arm is
>    **non-certifying**. The certifying four-phase TPE-vs-random comparison is blocked on the
>    dead-dimension question — see §8.3.1 and the sequencing correction in
>    `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` §11.4.

### 8.2 Execution Flow

```
run_bayesian_optimization()
    │
    ├─→ Check COORDINATOR_AVAILABLE
    │
    ├─→ Import window_optimizer_integration_final
    │
    ├─→ Print configuration:
    │   ├─ Lottery file
    │   ├─ Trials count
    │   ├─ Seed count
    │   ├─ PRNG type
    │   └─ Mode (constant vs both)
    │
    ├─→ Initialize coordinator:
    │   coordinator = MultiGPUCoordinator(
    │       config_file="distributed_config.json",
    │       resume_policy="restart"
    │   )
    │
    ├─→ Add integration:
    │   add_window_optimizer_to_coordinator()
    │
    ├─→ Run optimization:
    │   results = coordinator.optimize_window(
    │       dataset_path=lottery_file,
    │       seed_start=0,
    │       seed_count=seed_count,
    │       prng_base=prng_type,
    │       test_both_modes=test_both_modes,
    │       strategy_name=strategy_name,
    │       max_iterations=trials,
    │       output_file='window_optimization_results.json'
    │   )
    │
    ├─→ Build optimal_config dict:
    │   {
    │       'window_size': ...,
    │       'offset': ...,
    │       'skip_min': ...,
    │       'skip_max': ...,
    │       'sessions': [...],
    │       'prng_type': ...,
    │       'test_both_modes': ...,
    │       'seed_count': ...,
    │       'optimization_score': ...,
    │       'forward_count': ...,
    │       'reverse_count': ...,
    │       'bidirectional_count': ...
    │   }
    │
    ├─→ [S140] SEED COVERAGE WRITE-BACK          (:813-...)
    │
    ├─→ MERGE INCREMENTAL OUTPUT FIELDS          (:868-883)
    │
    ├─→ Inject agent_metadata (for pipeline chaining)
    │
    ├─→ Save optimal_window_config.json
    │
    ├─→ [S121] TRSE confirmed_windows feedback   (:953-982)
    │
    ├─→ Split lottery data:                      (:984-1003)
    │   ├─ train_history.json (80%)
    │   └─ holdout_history.json (20%)
    │
    └─→ Return results
```

#### 8.2.1 The three post-optimization stages the old flow omitted

| stage | what it does | anchor |
|---|---|---|
| **`[S140]` seed-coverage write-back** | logs this run's seed range to `exhaustive_progress` via `database_system.DistributedPRNGDatabase`, so WATCHER can advance `--seed-start` on the next run. Runs once per Step-1 completion. Failure is swallowed and printed, never raised | `window_optimizer.py:813-838` |
| **incremental-field merge** | if `optimal_window_config.json` already exists, nine crash-recovery fields are carried forward from it — `status`, `completed_trials`, `total_trials`, `best_trial_number`, `best_value`, `best_bidirectional_count`, `last_updated`, `last_trial_number`, `last_trial_value` — then `status` is set to `complete` and `completed_at` stamped. A corrupt file is swallowed and the new config used | `:868-883` (field list `:872-874`) |
| **`[S121]` TRSE feedback** | appends the winning window to `confirmed_windows` in the TRSE context, building a regime→window lookup across runs. Capped at the last **50** entries (`:976`). Passive by design: never raises, never blocks the pipeline if `trse_context.json` is absent | `:953-982` |

> **Separate ticket, not fixed here.** The TRSE block's own `except` handler calls
> `logger.warning` (`window_optimizer.py:982`) in a module that does not import `logging` or
> define `logger` — so the "non-fatal" path would itself raise `NameError`. **Unverified at
> runtime** (it requires triggering the exception), so this is recorded as a static observation,
> not a confirmed failure.

### 8.3 Test Both Modes (V2.0)

When `--test-both-modes` is enabled:

```python
if test_both_modes:
    print(f"Mode: TESTING BOTH CONSTANT AND VARIABLE SKIP")
    print(f"  - Will test {prng_type} (constant)")
    print(f"  - Will test {prng_type}_hybrid (variable)")
```

This runs sieves for both:
- **Constant skip:** `java_lcg` with fixed skip pattern
- **Variable skip:** `java_lcg_hybrid` with variable skip range

Survivors are tagged with `skip_mode` metadata for ML feature engineering.

#### 8.3.1 Governance — variable skip is NOT a routine option

The mechanism above is accurate, but an earlier revision of this chapter presented variable-skip
mode as a routine, fully-supported choice. It is not. Three governance constraints apply, none of
which are visible from the code:

| constraint | state | anchor / authority |
|---|---|---|
| **Hybrid certification** | **BLOCKED** until the sampled `skip_min`/`skip_max` actually reach the hybrid kernel. Hybrid exploration is permitted **non-certifying only**; constant-skip may resume normally | dead dimensions D-1/D-2 (§3.1.1); the DEFECT callout in §3.1 |
| **PWC hybrid path** | **QUARANTINED** — raises rather than running. Scope is variable-skip only; PWC constant-skip is untouched and still runs as a non-certifying diagnostic comparator. There is deliberately **no override flag** — an escape hatch would restore the "silently runnable" property Beta ruled out | `PWC_HYBRID_QUARANTINE_CODE = "PWC_HYBRID_THRESHOLD_CONTRACT_UNCERTIFIED"`, `persistent_worker_coordinator.py:188` (raised at `:208`) |
| **Session scope** | Production re-optimization is **per-session**. Combined-session *sequential* sieving is **non-certifying and prohibited by default** | Team Beta ruling, 2026-07-30/31 |

**Why per-session.** Midday and evening draws use **independently selected equipment**, so there
is no evidentiary basis for advancing one PRNG state through interleaved records. Ordering is
normative *within* a session stream; combined-container order carries no PRNG-advance meaning.

> **Known gap, reported not resolved — re-verified OPEN at `81ef3f1`.** The sampler can still
> select the prohibited mode: `session_options` still offers `['midday','evening']` as its first
> entry (`window_optimizer.py:182-186`, unchanged) and Optuna samples across all three
> (`window_optimizer_bayesian.py:513-515`, applied at `:535`). Both the chapter and the code
> predate the ruling. An autonomous run can therefore currently select a configuration that
> cannot be certified. The code remedy is outside this chapter's scope and is flagged as a
> governance risk.

---

### 8.4 Study resume and warm start

> **Provenance.** This section folds in the surviving content of `docs/CHAPTER_1_PATCH_S114.md`,
> which was **never merged** into this chapter and has since been **superseded in its central
> mechanism**. That file is now marked superseded; read this section instead. The warm-start
> behaviour below is rewritten from live source, not carried over from the patch.

#### 8.4.1 `--resume-study` / `--study-name` — accurate as patched

Study selection now lives in the sampler-neutral core (§8.1.2), `run_optimization`
(`window_optimizer_bayesian.py:643-703`). **Behaviour is unchanged from the previous pass; only
the line numbers moved (`:560-605` → `:643-703`), re-verified step-by-step this session:**

1. `--study-name`, when given, **takes priority over auto-select** — the candidate list becomes
   that one DB (`:645-647`). The patch missed this flag; it was added later.
2. Otherwise candidates are `optuna_studies/window_opt_*.db` sorted by **mtime, newest first**
   (`:649-653`).
3. A candidate is resumable when it has `COMPLETE` trials **and** either fewer than
   `max_iterations` of them **or** an explicit `--study-name` was given (`:670`). The explicit
   name therefore also lets you extend a study that already reached its trial count.
4. On resume, `load_if_exists=_resume` and the remaining trial count is
   `max_iterations - completed` (`:672-677`, `:702`). Otherwise a fresh study is created
   (`:687-691`, name and storage path built at `:640-641`).

Because the sampler is now supplied by the caller, `optuna.create_study(..., sampler=sampler, …)`
(`:696-703`) is the single place either arm's sampler is installed — the resume rules above apply
identically to the TPE and RandomSampler arms.

**When to use it** — the patch's rationale still holds and is worth keeping:

| scenario | resume? | reason |
|---|---|---|
| normal pipeline run | **no** | a fresh study is a clean audit artifact |
| session interrupted mid-run | **yes** | continue from the checkpoint |
| extending trial count on a good study | **yes** | leverages the existing TPE model |
| changed PRNG type, dataset, or thresholds | **no** | old trials are not comparable and would corrupt the TPE model |

Auto-resume was considered and **rejected**: it risks silently loading a stale study from weeks
earlier, and each run should be a clean audit artifact. Flag-based opt-in keeps resume a
deliberate, recorded decision.

#### 8.4.2 Warm start — the patch's version describes DELETED code

> **Do not implement the patch's warm-start block.** It enqueued a hardcoded California-specific
> `W8_O43_S5-56, 0.49/0.49` as trial 0. **S144 removed that hardcoded fallback outright**, with
> the in-source note *"Warm-start: enqueue from trial_history_context ONLY. No hardcoded fallback
> — CA-specific W8_O43 removed"* (`window_optimizer_bayesian.py:707-709`).

Live behaviour (`:710-732`) — **unchanged from the previous pass; anchors moved `:630-652` →
`:710-732` with the extraction into `run_optimization`, and every clause was re-read this
session:**

- Warm start applies **only on a fresh study** — on resume it is skipped, because the trial is
  already in the DB (`:731-732`).
- The source is **context-driven**: `trial_history_context`, populated from `step1_trial_history`.
  With no context, there is no warm start and Optuna explores freely (`:729-730`).
- It is gated on **all six** of `warm_start_window`, `warm_start_offset`, `warm_start_skip_min`,
  `warm_start_skip_max`, `warm_start_fwd_thresh`, `warm_start_rev_thresh` being non-`None`
  (`:719`). If any is missing the enqueue is skipped with an explicit message (`:727-728`) — it
  does **not** partially enqueue.
- `session_idx` is carried separately, defaulting to `0` (`:718`), and is included in the enqueued
  params because the objective function requires it (`:723`, S166).
- The enqueue itself is `study.enqueue_trial(_ws_params)` (`:725`).

**Consequence:** a new or different dataset gets **no** warm start by design. That is the correct
behaviour — the previous design seeded every study with a config tuned to one dataset.

#### 8.4.3 The patch's "discrete regime structure" discovery — superseded in meaning

The patch reported W3 → 143,959 survivors and W8 → 43-53, concluding the signal is **discrete**
and the PRNG reseeds at specific draw intervals.

**The S172 Team Beta ruling reinterprets the W3 figure as noise, not signal.** The ruling raised
`window_size.min` from 2 to **6** on the finding that *"W=2/3 produces ~39%/53% survivor rate by
chance alone, regardless of threshold"* — recorded in `distributed_config.json`
`search_bounds.window_size._s172_note` and carried into the §4.1 snapshot. A 143,959-survivor
count at W3 is what a window that short produces from chance alone. The numbers may well
reproduce; the *interpretation* does not survive, and **W3 is now below the floor and cannot be
sampled at all**.

The W8 observation is not addressed by that ruling either way. It sits above the floor and is
consistent with the `_calibration_note` preference for W12 as the production baseline; treat it
as an empirical run result, not a code property.

#### 8.4.4 The patch's manifest claim is wrong on both surfaces

The patch states `trials` default was updated 50 → 100. **Neither value is in effect.**
`agent_manifests/window_optimizer.json` `default_params` has **no `trials` key at all** — it has
**`window_trials: 3`** — while argparse `--trials` still defaults to **50**
(`window_optimizer.py:1262`). Three different values across three surfaces, verified live this
session. The manifest's `resume_study: false` entry *is* present and correct.

---

## 9. Run With Config Mode

### 9.1 run_with_config()

For running sieves with an existing optimal configuration.

**The live signature takes 13 parameters** (`window_optimizer.py:1009-1023`, count verified by AST
this session), not the 7 below:

```python
def run_with_config(
    config_file: str,
    lottery_file: str,
    max_seeds: int,
    iterations: int,
    output_survivors: str = 'bidirectional_survivors.json',
    output_train: str = 'train_history.json',
    output_holdout: str = 'holdout_history.json',
    ...                               # + 6 more, below
) -> Dict[str, Any]:
```

The six added by S170-PARITY / PARITY-2: `use_persistent_workers`, `pwc_transport`,
`seed_cap_amd`, `seed_cap_nvidia`, `worker_pool_size`, `min_workers`.

> **Why those six matter.** They are propagated onto the coordinator at
> `window_optimizer.py:1084-1096`, and the in-source comments record what happened without them:
> config mode *"silently downgrades to legacy SSH distribution regardless of CLI flags"*
> (`:1085-1086`) and *"silently falls back to default chunk caps … despite CLI --seed-cap-amd"*
> (`:1091-1092`). That is the same drift class this chapter exists to catch: a parameter accepted
> upstream that never reaches the thing it names.
>
> **Note a surviving inconsistency:** `min_workers` defaults to **1** in this signature (`:1022`)
> but **24** at the CLI (`:1323`). Two defaults for one quantity; reported, not resolved.

### 9.2 Execution Flow

```
run_with_config()
    │
    ├─→ Load config from file
    │
    ├─→ Check test_both_modes from config
    │
    ├─→ Initialize coordinator
    │
    ├─→ Add integration layer
    │
    ├─→ Create WindowConfig object
    │
    ├─→ Initialize accumulator:
    │   accumulator = {
    │       'forward': [],
    │       'reverse': [],
    │       'bidirectional': []
    │   }
    │
    ├─→ For each iteration:
    │   result = run_bidirectional_test(
    │       coordinator=coordinator,
    │       config=window_config,
    │       dataset_path=lottery_file,
    │       seed_start=iteration * max_seeds,
    │       seed_count=max_seeds,
    │       prng_base=prng_type,
    │       test_both_modes=test_both_modes,
    │       forward_threshold=...,
    │       reverse_threshold=...,
    │       trial_number=iteration,
    │       accumulator=accumulator
    │   )
    │
    ├─→ Deduplicate survivors:
    │   def deduplicate(survivor_list):
    │       """Keep survivor with highest score per seed"""
    │       seed_map = {}
    │       for survivor in survivor_list:
    │           seed = survivor['seed']
    │           if seed not in seed_map or survivor['score'] > seed_map[seed]['score']:
    │               seed_map[seed] = survivor
    │       return list(seed_map.values())
    │
    ├─→ Save survivors:
    │   ├─ forward_survivors.json      → writes []  (see below)
    │   ├─ reverse_survivors.json      → writes []  (see below)
    │   └─ bidirectional_survivors.json
    │
    ├─→ NPZ CONVERSION GATE — raises RuntimeError  (:1194-1202)
    │
    └─→ Split lottery data (80/20)                 (:1205-1228)
```

#### 9.2.1 Three corrections to this flow

**1. The NPZ conversion is a hard release gate, not a convenience step.** Config mode shells out
to `convert_survivors_to_binary.py` and raises
`RuntimeError("Step 1 incomplete - NPZ conversion required for Step 2")` on a non-zero exit
(`window_optimizer.py:1194-1202`). Nothing downstream runs if it fails.

**2. `deduplicate()` runs over permanently empty lists.** The function is real and works as
written (`:1165` ff.), but `accumulator['forward']` and `accumulator['reverse']` are never
appended to anywhere in the integration layer — only `accumulator['bidirectional']` is. So in
this mode the two files receive `[]` while the run prints `"✅ Saved 0 forward survivors"`. See
§12.1: the Bayesian path degraded *honestly* (it writes an explanatory `note`); this path
degrades **silently**. That asymmetry is a behavioural defect on a **separate ticket** and is not
fixed here.

**3. Winner selection for the canonical artifact is no longer this function's job.** For the
certified generation it is the finalizer's frozen L2 key (`utils/run_finalizer.py:690`, `:714`);
the legacy helper was **removed, not bypassed**. `deduplicate()` governs only the JSON summaries.

---

## 10. CLI Interface

### 10.1 Arguments

**All 40 flags**, as declared at `window_optimizer.py:1249-1383`. Defaults are the argparse
defaults read from that range; where a flag carries a *bound* rather than a default, the bound
lives in the §4.1 snapshot, never here.

> **Changed since the correction pass: 38 → 40.** The Resolved Execution Set (`63e627f`) added
> `--rig-profile` and `--execution-set-nodes` (both below), and pushed the four miner flags down
> by 19 lines. **Every flag anchor from `:1249` through `:1348` is unchanged and was re-verified
> individually this session**; only the four miner anchors moved.

**Mode selection and I/O**

| flag | default | purpose | anchor |
|---|---|---|---|
| `--strategy` | `bayesian` | `bayesian` \| `random` \| `grid` \| `evolutionary`. **Only `bayesian` works** — the other three fail closed (below) | `:1249` |
| `--config-file` | — | run with an existing config; skips optimization (§9) | `:1254` |
| `--lottery-file` | **required** | path to draw-data JSON | `:1258` |
| `--output` | `optimal_window_config.json` | optimal-config output path | `:1264` |
| `--output-survivors` | `bidirectional_survivors.json` | see §12.1 — this is a **summary**, not survivor data | `:1272` |
| `--output-train` | `train_history.json` | 80% split | `:1274` |
| `--output-holdout` | `holdout_history.json` | 20% split | `:1276` |
| `--prng-type` | `java_lcg` | base PRNG family from the registry | `:1280` |

**Search control**

| flag | default | purpose | anchor |
|---|---|---|---|
| `--trials` | `50` | Optuna trial count | `:1262` |
| `--max-seeds` | `10_000_000` | seeds per iteration (config mode) | `:1268` |
| `--iterations` | `1` | sieve iterations (config mode) | `:1270` |
| `--seed-start` | `0` | starting seed; **set automatically by the S140 coverage tracker** | `:1332` |
| `--test-both-modes` | off | test constant **and** variable skip (§8.3) | `:1304` |
| `--enable-pruning` | `False` | Optuna pruning (S115 R3) | `:1307` |
| `--n-parallel` | `1` | trial parallelism (S115 M1) | `:1309` |
| `--trse-context` | `trse_context.json` | Step-0 TRSE regime context (S121) | `:1311` |

**Threshold overrides — DECLARED BUT UNWIRED. These FAIL CLOSED.**

| flag | default | purpose | anchor |
|---|---|---|---|
| `--forward-threshold` | `None` | **passing this aborts the run** — see below | `:1285` |
| `--reverse-threshold` | `None` | **passing this aborts the run** — see below | `:1290` |

**Optuna study resume** (the surviving half of `CHAPTER_1_PATCH_S114.md`, §13 below)

| flag | default | purpose | anchor |
|---|---|---|---|
| `--resume-study` | off | resume the most recent **incomplete** study DB instead of starting fresh; skips warm-start enqueue if the study already has trials | `:1296` |
| `--study-name` | `''` | specific study to resume, e.g. `window_opt_1772507547`. **Takes priority over auto-select** (`window_optimizer_bayesian.py:563-566`); empty = auto-select most recent by mtime (`:568-573`). Only read when `--resume-study` is set | `:1300` |

**Warm start** (S166 — all seven must be non-`None` together, §13)

| flag | default | anchor |
|---|---|---|
| `--warm-start-window` | `None` | `:1336` |
| `--warm-start-offset` | `None` | `:1338` |
| `--warm-start-skip-min` | `None` | `:1340` |
| `--warm-start-skip-max` | `None` | `:1342` |
| `--warm-start-fwd-thresh` | `None` | `:1344` |
| `--warm-start-rev-thresh` | `None` | `:1346` |
| `--warm-start-session-idx` | `None` | `:1348` |

**Backend selection — mutually exclusive** (§2.1, §11)

| flag | default | purpose | anchor |
|---|---|---|---|
| `--use-persistent-workers` | `False` | PWC backend (S134). **Non-certifying**; hybrid path quarantined (§8.3) | `:1315` |
| `--use-zmq-sqlite` | `False` | ZMQ/SQLite backend (S158D). **Non-certifying** | `:1318` |
| `--use-range-miner` | `False` | RANGE-MINER stripe backend (S172) — **the certifying route** | `:1374` |
| `--pwc-transport` | `tcp` | `ssh` \| `tcp` (S162) | `:1320` |
| `--min-workers` | `24` | PWC readiness gate (S162) | `:1323` |
| `--worker-pool-size` | `8` | PWC worker pool (S134). **Under a frozen execution set this is now the REQUEST, not the answer** — see below | `:1326` |
| `--seed-cap-nvidia` | `5_000_000` | per-dispatch seed cap, NVIDIA (S137) | `:1328` |
| `--seed-cap-amd` | `2_000_000` | per-dispatch seed cap, AMD (S137) | `:1330` |
| `--miner-stripe-size` | `67_108_864` | seeds per stripe per GPU (S172 §6.2) | `:1376` |
| `--miner-substripes` | `8` | sub-stripes per stripe, sized to fit the watchdog | `:1378` |
| `--miner-output-dir` | `None` | auto-detect `/dev/shm/prng/miner/` if writable, else `~/miner_output/` | `:1383` |

**Fleet selection — NEW, added by the Resolved Execution Set (`63e627f`)**

| flag | default | purpose | anchor |
|---|---|---|---|
| `--rig-profile` | `None` → `default_profile` in `rig_profiles_config.json` | `baremetal` \| `proxmox`. Every machine is a boot-selector (`CLAUDE.md` §3); **both topologies are retained** and this picks which endpoints enter the resolved execution set. It is a topology *selector*, not a correction — the bare-metal addresses in `distributed_config.json` are deliberate | `:1352-1360` |
| `--execution-set-nodes` | `None` → the full declared fleet | comma-separated logical node ids (`localhost`, `rrig6600,rrig6600b`). **This is how a PARTIAL fleet is declared — explicitly, and frozen before the run.** A partial set is never inferred from which workers happen to answer, and a named node that does not exist is an error, never a silent drop | `:1361-1368` |

> **Why these two are in a chapter about the window optimizer.** They are resolved and **frozen at
> run start**, in `main()`, *after* backend selection and rig-profile selection but **before**
> dataset verification, GPU verification, `MultiGPUCoordinator` construction and any dispatch —
> the placement is the contract (`window_optimizer.py:1461-1471`, in-source). Step 1 is where the
> fleet becomes a fact for the whole run. Consequence for `--worker-pool-size`: with a set frozen,
> admission expectation is `min(requested pool size, count of selected worker identities)`, and
> **both numbers are recorded in the set identity** — a run that asked for 8 and was clamped to 2
> is distinguishable from one that asked for 2. With no set frozen the requested value is used
> unchanged. Full treatment is outside this chapter; see the project-facts skill §2.11–§2.12b.

> **Backend mutex.** At most one of `--use-persistent-workers` / `--use-zmq-sqlite` /
> `--use-range-miner` may be set; two or more is an argparse error naming the offending flags
> (`window_optimizer.py:1448-1459`). The check runs **before** `MultiGPUCoordinator` is
> constructed — and, since `63e627f`, before the execution set is resolved, because the mutex is
> where the backend becomes a fact.

> **Count provenance.** 40 is a live count (`/bin/grep -c add_argument window_optimizer.py`, VM 101
> at `81ef3f1`, 2026-08-02). It was **38** at `40c3c83` and at the audit base `77dc629`.
> `CHAPTER_1_AUDIT_v1.md` §3 §10.1 states 31; that figure is an undercount and should not be
> propagated. **Re-count rather than cite** — this number has now moved once inside this chapter's
> lifetime.

#### `--forward-threshold` / `--reverse-threshold` — dead dimension D-4, now fail-closed

The pre-correction chapter documented these as *"Override Optuna optimization (0.15-0.60)."*
**That override never existed.** They were declared in argparse and
`args.forward_threshold` / `args.reverse_threshold` were never referenced after
`parse_args()`. An operator passing `--forward-threshold 0.6` got a **silent no-op on a run
that reported success** — the first operator-facing dead dimension (D-4, §3.1.1). Three
mutually inconsistent bound figures were in play at once: this chapter said `0.15-0.60`, the
`--help` text said `0.5-0.95` / `0.6-0.98`, and the effective bounds were neither — for the
live pair see the §4.1 snapshot. The help text now states the unwired condition instead of a
bound (`window_optimizer.py:1285-1294`).

**Current behaviour:**

| invocation | result |
|---|---|
| flag absent | existing supported path, unchanged |
| flag present (any value, including `0.0`) | **explicit nonzero failure before coordinator construction**, diagnostic `WINDOW_OPTIMIZER_THRESHOLD_OVERRIDE_UNWIRED` |

The flags are **kept declared rather than deleted from argparse**, deliberately: the operator
intent they record is legitimate, and a named diagnostic tells the operator the capability is
*unwired* rather than misspelled. See `window_optimizer.py` for the recorded condition under
which they may return — they must feed the single `resolve_directional_threshold()` authority
(§7.2.1), preserve `0.0` via `is None`, and record requested/payload/effective. **They must
not create parallel threshold state.**

#### `--strategy random | grid | evolutionary` — fail closed (signature mismatch)

Only `--strategy bayesian` is functional. `WindowOptimizer.optimize` calls
`strategy.search(..., resume_study=, study_name=, trse_context_file=,
trial_history_context=)` (`window_optimizer.py:622-625`; the forwarded set is declared once at
`:334`, `OPTIMIZE_FORWARDED_KWARGS`); only `BayesianOptimization.search` accepts those kwargs
(`:440-443`). The other three raise `TypeError` on first call — verified by live
`inspect.signature`:

```
RandomSearch          (self, objective_function, bounds, max_iterations, scorer)
GridSearch            (self, objective_function, bounds, max_iterations, scorer)
EvolutionarySearch    (self, objective_function, bounds, max_iterations, scorer)
BayesianOptimization  (self, objective_function, bounds, max_iterations, scorer,
                       resume_study=False, study_name='', trse_context_file=...,
                       trial_history_context=None)
```

**Proximate cause: signature divergence.** The four kwargs were added to
`BayesianOptimization.search` **and to the shared call site** in three commits — `cd213e9`
(S116, 2026-03-04, `resume_study`/`study_name`), `2377228` (S123, `trse_context_file`),
`c6fde66` (S140b, `trial_history_context`) — and the three sibling classes were never touched.
`cd213e9` is the breaking commit: it is the one that changed the call site from positional-only
to kwarg-forwarding. From **2026-03-04** onward the three raise `TypeError` on first call — a
~4½-month unnoticed outage, explained by every recorded run having used `bayesian`. The
`SearchStrategy` ABC (`window_optimizer.py:339-361`) was never updated past the pre-S116
four-positional convention, so no signature check had a correct reference to compare against;
`:343-348` now records that staleness in-source.

> **Correction — an earlier revision of this chapter said "root cause is code rot, not design."
> That claim is refuted and has been withdrawn.** `docs/STRATEGY_ORIGIN_AUDIT.md` (2026-07-31,
> full git-history investigation) establishes:
>
> - **The documented design was four Optuna samplers**, not four hand-rolled strategies. Two
>   committed documents state it: `docs/SESSION_CHANGELOG_20260207_S63.md:9` describes
>   `search_strategy` as *"the Optuna sampler selection parameter"*, and
>   `docs/PROPOSAL_SEARCH_STRATEGY_VISIBILITY_FIX_v1_0.md:20` asserts *"All 4 Optuna samplers
>   (TPE, Random, Grid, CmaES) implemented ✅"*. Under §0.4 a stated intent outranks an inference
>   from current code shape.
> - **`GridSearch` and `EvolutionarySearch` were working code that was deleted.** Both had
>   complete bodies — Grid a 4-deep nested loop — until they were cut down to
>   `# Placeholder … return {}` between 2025-11-14 and 2025-11-15. The word "placeholder" in §6.4
>   describes *the result of that deletion*, not an original design decision.
> - **`RandomSearch` was always hand-rolled** and has never been Optuna-backed. No version of any
>   of the three, in any of the 727 commits or any recoverable pre-git snapshot, contains the
>   tokens `optuna`, `sampler`, `create_study` or `study`. Optuna entered `window_optimizer.py`
>   on 2025-11-01 and entered **only** `BayesianOptimization`.
>
> So "code rot" describes the *signature* half accurately and the *body* half not at all: for two
> of the three there is no working implementation for a signature repair to reach.

**The repair is not specified here — it is pending a Team Beta ruling**
(`TEAM_ALPHA_AUTONOMY_CONTROL_SURFACE_SUBMISSION.md` Q3). This chapter states what is broken and
why, and stops there. Per §0.4 these are **not** candidates for deletion.

> **One prescription is explicitly ruled out, and is recorded here so it is not re-proposed.**
> "Bring the signatures up to the calling convention" is **insufficient and misleading**.
> `GridSearch.search` (`window_optimizer.py:410-412`) and `EvolutionarySearch.search`
> (`:484-486`) are `return {}`. The contract gate is derived from **live signatures**
> (`strategy_contract_gap()`, `:518`), so adding the four kwargs would **clear the gate while
> `search()` still returns an empty dict** — `optimize()` would hand `{}` back to the integration
> layer with no `best_config` and no `all_results`, *after* the 26-GPU coordinator had been
> constructed, and the gate that exists to prevent exactly that would report **PASS**. That is a
> VIR-2 vacuous-pass: a signature-derived detector structurally cannot see a stub body. A
> signature repair is at best *necessary but not sufficient*, and for `RandomSearch` it would
> still produce a uniform-random run recorded under a strategy name the governance layer defines
> as Optuna-backed — the semantic-substitution concern that motivated the fail-closed below.

Requesting one of the three now aborts with `WINDOW_OPTIMIZER_STRATEGY_UNSUPPORTED` naming
the missing kwargs, rather than letting `TypeError` escape mid-run.

> **A second prescription is ruled out, and quantified here so it is not re-proposed either:
> "just restore Grid as an Optuna `GridSampler` arm." `GridSampler` is unconstructible against
> the live bounds.** Added at closure (`81ef3f1`, 2026-08-02); the arithmetic below was executed
> this session, not transcribed.
>
> `GridSampler.__init__` materialises the **entire** cartesian product eagerly —
> `self._all_grids = list(itertools.product(*self._search_space.values()))`, verified by
> `inspect.getsource` against the installed **optuna 4.4.0** — and then *shuffles that list*. It
> is not a lazy iterator, so the cost is paid at construction, before a single trial runs.
>
> Enumerating the seven sampled dimensions (§8.1.1) at their live §4.1 bounds:
>
> | dimension | live range | grid points |
> |---|---|---|
> | `window_size` | 6 … 50 | 45 |
> | `offset` | 0 … 100 | 101 |
> | `session_idx` | 0 … 2 | 3 |
> | `skip_min` | 0 … 10 | 11 |
> | `skip_max` | 10 … 250 | 241 |
> | `forward_threshold` | 0.30 … 0.75, rounded to 2 dp (`:538`) | 46 |
> | `reverse_threshold` | 0.30 … 0.75, rounded to 2 dp (`:539`) | 46 |
>
> ```
> 45 × 101 × 3 × 11 × 241 × 46 × 46  =  76,485,750,660  ≈  7.649 × 10¹⁰ grid points
> ```
>
> At `sys.getsizeof(7-tuple) = 96` bytes plus an 8-byte list slot — **measured** on this
> interpreter, not assumed — that is `76,485,750,660 × 104 = 7.95 × 10¹²` bytes ≈ **7.23 TiB of
> resident RAM at construction**, excluding the shuffle's own working set. VM 101 has nothing
> close to that.
>
> **What this does and does not mean.** It is not an argument for deleting `GridSearch` (§0.4
> forbids that, and the ruled-out prescription above still stands). It means the *documented*
> four-sampler design is **not implementable as stated for Grid at the current bounds** — a real
> Grid arm would require an explicitly coarsened per-dimension grid, which is a **design decision
> with governance consequences** (it changes what "grid search over the search space" means), not
> a restoration. That decision is Beta's, and this chapter does not make it. The two threshold
> dimensions alone contribute a 46 × 46 factor and are the obvious coarsening target, but naming
> the target is not the same as authorising it.

**Related, and equally fail-closed:** a Bayesian request when Optuna is unavailable **fails**.
It does not fall back to random search. Team Beta: that is *semantic substitution, not
graceful degradation* — the operator asked for TPE and would have received uniform sampling
under the same label, with the study recording it as Bayesian.

> **Surface not corrected in this tranche, re-verified at `81ef3f1`:**
> `agent_manifests/window_optimizer.json` still advertises all four strategies to WATCHER. Its
> `default_params.strategy` is `bayesian`, and a request for any of the other three now fails
> closed at the CLI rather than crashing, so the manifest is misleading but no longer dangerous.
> Flagged for the manifest owner.
>
> **Three further manifest observations, recorded this session and NOT repaired** (the manifest is
> out of scope for a documentation pass):
>
> 1. **`args_map` maps `forward-threshold` and `reverse-threshold`** — the two flags that now
>    **fail closed** (below). `default_params` supplies neither key, so the mapping is dormant and
>    no WATCHER run currently trips it. It would abort the run if either key were ever added.
> 2. **`parameter_bounds` is a live admission gate that is looser than §4.1's authority.**
>    `agents/step_runner/command_builder.py:151-170` validates supplied parameters against it and
>    rejects out-of-range values. Its numbers are **not** the §4.1 bounds: `window_size` min **2**
>    (the S172 TB ruling raised the search floor to **6**), `offset` max **2000** (live **100**),
>    `skip_min` max **50** (live **10**), `skip_max` **20–500** (live **10–250**). It cannot widen
>    the Optuna search space — that is read from `distributed_config.json` by
>    `SearchBounds.from_config()` — but it **would accept** a WATCHER-proposed `window_size=2`,
>    which is precisely the value the S172 ruling excluded as chance-driven. Two surfaces, one
>    quantity, different answers; recorded as open, not resolved here.
> 3. **`--rig-profile` and `--execution-set-nodes` are absent from `args_map`.** WATCHER therefore
>    cannot select a rig profile or declare a partial execution set; a WATCHER-launched run takes
>    the configured `default_profile` and the full declared fleet. Whether that is the intended
>    boundary is a governance question for the manifest owner, not a defect this chapter asserts.

### 10.2 Mode Decision Tree

```
main()
    │
    ├─→ args.strategy == 'bayesian':
    │   └─→ run_bayesian_optimization(strategy_name='bayesian')
    │
    ├─→ args.strategy == 'random':
    │   └─→ run_bayesian_optimization(strategy_name='random')
    │
    ├─→ args.strategy == 'grid':
    │   └─→ run_bayesian_optimization(strategy_name='grid')
    │
    ├─→ args.strategy == 'evolutionary':
    │   └─→ run_bayesian_optimization(strategy_name='evolutionary')
    │
    ├─→ args.config_file:
    │   └─→ run_with_config()
    │
    └─→ else:
        └─→ Print usage and exit(1)
```

---

## 11. Integration Layer

### 11.1 Key Import

```python
from window_optimizer_integration_final import add_window_optimizer_to_coordinator
```

**This is not a module-level import.** It is lazy, inside each entry point — `window_optimizer.py:724`
(Bayesian) and `:1037` (config mode) — to break a circular dependency, and each site also imports
`run_bidirectional_test`. Moving it to module scope will not work.

### 11.2 What It Does

The integration layer (`window_optimizer_integration_final.py`) provides:

1. **`add_window_optimizer_to_coordinator()`** (`:1683-2703`) — monkey-patches `optimize_window()` (`:1695-2684`) onto the coordinator
2. **`run_bidirectional_test()`** (`:1138-1680`) — selects a backend (§11.3), runs forward+reverse sieves, computes the intersection
3. **Artifact publication** — hands off to `utils.run_finalizer` (imported `:2515-2519`, called `:2603`) for the certified generation

> **The old third item — "collects survivors across all trials with metadata" — is materially
> wrong now.** The layer no longer accumulates forward/reverse survivor *objects*: `[S166-ACCUM]`
> replaced object retention with counters to stop a RAM bomb at 26-GPU scale (`:1025`, `:1544`,
> `:1648`, `:2056`, `:2545`). Bidirectional records are still accumulated with full metadata.
> Final artifact assembly moved out of this layer entirely — the in-source note at `:2498-2513`
> records that the local assembly was **replaced by** the shared finalizer, not wrapped by it,
> and that the legacy `deduplicate_survivors` helper was **removed, not merely bypassed**
> (`:2504-2512`).

### 11.3 Backend cascade — `run_bidirectional_test` has FOUR backends

§2.1's diagram draws only the legacy coordinator leg. `run_bidirectional_test` (`:1138-1680`)
opens with a cascade of `getattr`-gated branches, in this order:

| order | backend | gate | anchor | status |
|---|---|---|---|---|
| 1 | **RANGE-MINER** | `use_range_miner` | `:1162-1172` | **the certifying route** (S172 Phase 4) |
| 2 | **PWC** | `use_persistent_workers` | `:1300-1304` | non-certifying diagnostic; **hybrid quarantined** (§8.3.1) |
| 3 | **ZMQ/SQLite** | `use_zmq_sqlite` | `:1346-1349` | non-certifying (S158D) |
| 4 | **legacy coordinator** | none — fall-through | after the three gates | the path §2.1 draws |

Selection is mutually exclusive, enforced at argparse (`window_optimizer.py:1448-1459`), and the
miner gate is deliberately placed first *"so miner selection wins unambiguously"* (`:1164-1165`).
Each of the two additive gates records in-source that it makes **zero changes to the path below
it** (`:1301`, `:1347`). **Cite these by gate variable — `_use_miner` (`:1171`), `_use_pw`
(`:1304`), `_use_zmq` (`:1349`) — rather than by comment-block line; the whole cascade shifted
+15 lines between the correction pass and closure without changing at all.**

**Consequence for the reader:** most production runs do not take the path §2.1 draws, and a
question of the form "what does Step 1 do here?" cannot be answered without first knowing which
backend flag was set.

### 11.4 Integration Flow

```
window_optimizer.py                    window_optimizer_integration_final.py
       │                                              │
       │  add_window_optimizer_to_coordinator()      │
       │ ──────────────────────────────────────────→ │
       │                                              │
       │  coordinator.optimize_window(...)           │
       │ ──────────────────────────────────────────→ │
       │                                              │
       │                                              ├─→ Create WindowOptimizer
       │                                              │
       │                                              ├─→ Override test_configuration_func
       │                                              │
       │                                              ├─→ For each trial:
       │                                              │   └─→ run_bidirectional_test()
       │                                              │       ├─→ Forward sieve (coordinator)
       │                                              │       ├─→ Reverse sieve (coordinator)
       │                                              │       ├─→ Compute intersection
       │                                              │       └─→ Accumulate survivors
       │                                              │
       │  ←─────────────────────────────────────────  │
       │      Return results + accumulated survivors │
```

> **This diagram is a skeleton, and it omits two stages that matter.** (a) "Forward sieve /
> Reverse sieve (coordinator)" is only the **fourth** of four backends — see §11.3. (b) The
> diagram ends at "return survivors", but the run is not finished there: publication of the
> certified NPZ generation happens via `utils.run_finalizer.finalize_run` (`:2603`), which is the artifact
> Steps 2–6 actually consume (§12.1). "Accumulate survivors" also overstates the forward/reverse
> legs, which are counters (§11.2).

---

## 12. Output Files

### 12.1 Bayesian Mode Outputs

> **The canonical Step-1 → Steps-2–6 carrier is the certified NPZ generation**, produced by
> `utils.run_finalizer.finalize_run` (`window_optimizer_integration_final.py:2603`). It is the one
> output that matters, and the pre-correction table had no row for it. The three
> `*_survivors.json` files are **not** the survivor data they appear to be.

| File | Contents | Status |
|------|----------|--------|
| **certified NPZ generation** (`utils.run_finalizer`) | the **22-array NPZ contract** plus sidecar; carries `artifact_sha256`, `sidecar_sha256` and the generation lineage fields (`window_optimizer_integration_final.py:2639-2653`). Generations **chain** — the finalizer merges prior rows | **CANONICAL — this is what Steps 2–6 consume** |
| `optimal_window_config.json` | best parameters + `agent_metadata` (§12.2) | current |
| `window_optimization_results.json` | full trial history, written by `optimizer.save_results(results, output_file)` (`window_optimizer_integration_final.py:2474`; the `output_file` parameter defaults to `'window_optimization.json'` at `:1703` and is passed in by the caller) | current |
| `bidirectional_survivors.json` | **post-success SUMMARY of the certified generation** — generation IDs and sha256s, **no seeds** (`:2628-2655`). In-source: *"It is NO LONGER the canonical Steps 2-6 input… Steps 2-6 consume the canonical NPZ"* (`:2628-2632`) | **demoted — summary only** |
| `forward_survivors.json` | `{"survivor_count": N, "note": "Full survivors omitted — objects not retained; see <all-NPZ name>"}` (`:2545-2554`) | **count-only stub** |
| `reverse_survivors.json` | as above | **count-only stub** |
| `train_history.json` | 80% lottery data for training (`window_optimizer.py:984-1003`, `:1205-1228`) | current |
| `holdout_history.json` | 20% lottery data for validation | current |

**Why forward/reverse are count-only.** `accumulator['forward']` and `accumulator['reverse']`
are never appended to — only `accumulator['bidirectional']` is
(`window_optimizer_integration_final.py:1022-1023`, `:1553`, `:1655`); the forward/reverse legs
increment `accumulator['forward_count']` / `['reverse_count']` instead (`:1026-1027`, `:1547-1548`,
`:1649-1650`). `[S166-ACCUM]` (`:1544-1546`) replaced object retention with counters to stop a RAM
bomb at 26-GPU scale.
**That change is deliberate and correct** — the canonical NPZ carries what downstream needs.
Do not "restore" full retention.

> **Known defect, separate ticket (not fixed in this tranche).** In `--config-file` mode
> `window_optimizer.py:1165-1192` still dedups those permanently-empty lists and writes `[]` to
> both files while printing `"✅ Saved 0 forward survivors"`. The Bayesian path degraded
> *honestly* (it writes a `note` explaining the omission); the config path degrades
> **silently**.

**Undocumented hard gate.** `--config-file` mode shells out to
`convert_survivors_to_binary.py` and raises `RuntimeError("Step 1 incomplete - NPZ conversion
required for Step 2")` on failure (`window_optimizer.py:1194-1202`). This is a release gate, not
a convenience step.

### 12.2 optimal_window_config.json Structure

> **The values below are ILLUSTRATIVE SHAPE ONLY and two of them are unreachable.** Noted at
> closure rather than silently rewritten, because the shape is what this section documents:
> `window_size: 256` exceeds the live window ceiling of **50**, and `skip_max: 30` sits inside
> `[10, 250]` but the pair was never produced by a live run. This is the same defect §3.1's
> example already carries a correction for. **Read the keys, not the numbers**; for reachable
> values see the §4.1 snapshot.

```json
{
    "window_size": 256,
    "offset": 50,
    "skip_min": 0,
    "skip_max": 30,
    "sessions": ["midday", "evening"],
    "prng_type": "java_lcg",
    "test_both_modes": false,
    "seed_count": 10000000,
    "optimization_score": 847.0,
    "forward_count": 12543,
    "reverse_count": 9876,
    "bidirectional_count": 847,
    "run_id": "step1_20251215_143052_12345",
    "agent_metadata": {
        "inputs": [{"file": "lottery.json", "required": true}],
        "outputs": ["optimal_window_config.json", "bidirectional_survivors.json", ...],
        "pipeline_step": 1,
        "follow_up_agent": "scorer_meta_agent",
        "confidence": 0.847,
        "suggested_params": {...},
        "reasoning": "Optimization found 847 survivors with score 847.0000"
    }
}
```

### 12.3 Survivor Record Structure (v3.1)

**The record is FLAT.** The pre-correction chapter nested the window parameters under a
`"window_config"` key and listed a `"timestamp"` field. Neither is real: `window_size`,
`offset`, `skip_min` and `skip_max` sit at top level in `metadata_base`
(`window_optimizer_integration_final.py:1520-1523`), and **no `timestamp` key is produced**
anywhere — not by `metadata_base` (`:1519-1542`) nor by the append (`:1553-1559`). Any
consumer written against the old shape would fail.

```json
{
    "seed": 12345678,
    "score": 0.85,
    "forward_match_rate": 0.75,
    "reverse_match_rate": 0.50,
    "prng_type": "java_lcg",
    "prng_base": "java_lcg",
    "skip_mode": "constant",

    "window_size": 4,
    "offset": 26,
    "skip_min": 1,
    "skip_max": 108,
    "skip_range": [1, 108],
    "sessions": ["midday"],

    "trial_number": 6,
    "forward_count": 8987,
    "reverse_count": 8855,
    "intersection_count": 8929,

    "bidirectional_count": 8929,
    "intersection_ratio": 0.488,
    "forward_only_count": 8987,
    "reverse_only_count": 8855,
    "survivor_overlap_ratio": 0.498,
    "bidirectional_selectivity": 1.007,
    "intersection_weight": 0.244
}
```

- `forward_match_rate` / `reverse_match_rate` are **per-seed** values read from the GPU sieve
  kernel via `forward_map` / `reverse_map` (`:1550-1557` constant, `:1652-1659` variable), not
  trial-level aggregates.
- All 7 intersection fields are present (`:1535-1541` in `metadata_base`, constant; `:1639-1645`
  in `metadata_base_hybrid`, variable).
- `intersection_count` duplicating `bidirectional_count` is **deliberate** — not a defect.
- `skip_range`, `sessions`, `prng_base`, `forward_count`, `reverse_count` and
  `intersection_count` were absent from the pre-correction example.

---

## 13. Agent Metadata Injection

### 13.1 Purpose

Agent metadata enables autonomous pipeline chaining by the Watcher Agent.

### 13.2 inject_agent_metadata()

```python
from integration.metadata_writer import inject_agent_metadata

optimal_config = inject_agent_metadata(
    optimal_config,
    inputs=[{"file": lottery_file, "required": True}],
    outputs=["optimal_window_config.json", "bidirectional_survivors.json",
             "train_history.json", "holdout_history.json"],
    pipeline_step=1,
    follow_up_agent="scorer_meta_agent",
    confidence=min(0.95, results['best_score'] * 10),
    suggested_params={
        "window_size": best_config['window_size'],
        "forward_threshold": 0.72,
        "reverse_threshold": 0.81,
        "k_folds": 5
    },
    reasoning=f"Optimization found {survivors_count} survivors with score {score:.4f}"
)
```

### 13.3 Metadata Fields

| Field | Type | Purpose |
|-------|------|---------|
| `inputs` | List[Dict] | Required input files |
| `outputs` | List[str] | Generated output files |
| `pipeline_step` | int | Step number (1 for window optimizer) |
| `follow_up_agent` | str | Next agent in pipeline |
| `confidence` | float | 0.0-1.0 confidence score |
| `suggested_params` | Dict | Parameters for next step |
| `reasoning` | str | Human-readable explanation |

---

## 14. Complete Method Reference

### 14.1 Module-Level Functions

| Function | Parameters | Returns | Purpose |
|----------|------------|---------|---------|
| `load_search_bounds_from_config()` | `config_path` | `dict` | Load bounds from JSON |
| `run_bayesian_optimization()` | `lottery_file, trials, ...` | `Dict` | Main Bayesian entry point |
| `run_with_config()` | `config_file, lottery_file, ...` | `Dict` | Run with existing config |
| `main()` | — | — | CLI entry point |

### 14.2 WindowConfig Methods

| Method | Returns | Purpose |
|--------|---------|---------|
| `__hash__()` | `int` | Hashable for sets/dicts |
| `description()` | `str` | Human-readable description |
| `to_dict()` | `Dict` | JSON serialization |

### 14.3 SearchBounds Methods

| Method | Returns | Purpose | anchor |
|--------|---------|---------|---|
| `from_config(path)` | `SearchBounds` | Load from config file | `window_optimizer.py:159-177` |
| `__post_init__()` | — | Auto-initialize `session_options` | `:179-186` |
| `validate_baseline_in_bounds(path)` | `bool` | **Team Beta mandate** — raises `ValueError` if the baseline is outside the sampled bounds; see §3.2 | `:189-222` |
| `random_config()` | `WindowConfig` | Generate random config | `:224-237` |
| `is_valid(config)` | `bool` | Validate against bounds | `:239-247` |

### 14.4 TestResult Properties

| Property | Type | Formula |
|----------|------|---------|
| `precision` | `float` | `bidirectional / forward` |
| `recall` | `float` | `bidirectional / reverse` |

### 14.5 WindowOptimizer Methods

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `test_configuration()` | `config, seed_start, seed_count` | `TestResult` | Test single config |
| `optimize()` | `strategy, bounds, max_iterations, ...` | `Dict` | Run optimization |
| `save_results()` | `results, output_path` | — | Save to JSON |

### 14.6 SearchStrategy Methods (Abstract)

| Method | Parameters | Returns |
|--------|------------|---------|
| `search()` | `objective_function, bounds, max_iterations, scorer` | `Dict[str, Any]` |
| `name()` | — | `str` |

---

## 15. Dependencies Summary

| Dependency | Required | Purpose |
|------------|----------|---------|
| `coordinator.py` | ✅ Yes | Sieve execution |
| `window_optimizer_bayesian.py` | ⚠️ Optional | Optuna TPE |
| `window_optimizer_integration_final.py` | ✅ Yes | Integration layer |
| `integration.metadata_writer` | ⚠️ Optional | Agent metadata |
| `distributed_config.json` | ⚠️ Optional | Search bounds |

---

## 16. Chapter Summary

**Chapter 1: Window Optimizer** covers Step 1 of the pipeline:

Structure of `window_optimizer.py`, **re-extracted from the module AST on VM 101 at commit
`81ef3f1`** (2026-08-02). The pre-correction table summed to ~580 lines against a file that was
**1592**; every figure in it was low by roughly a factor of three. **Re-extract rather than cite
this table** — it is a snapshot, exactly like §4.1's bounds, and it has already moved once.

| Component | Live range | Lines | Purpose |
|-----------|-----------|-------|---------|
| Config-bounds loader | `:40-92` | 53 | `_MappingAttrView`, `load_search_bounds_from_config()` |
| Data structures | `:100-284` | 185 | `WindowConfig` (32), `SearchBounds` (114), `TestResult` (35) |
| Scoring functions | `:290-315` | 26 | `ScoringFunction` ABC, `BidirectionalCountScorer` |
| Strategy fail-closed guards | `:321-337`, `:518-560` | ~60 | `StrategyContractError`, `OPTIMIZE_FORWARDED_KWARGS`, `strategy_contract_gap()`, `require_supported_strategy()` |
| Search strategies | `:339-489` | 151 | `SearchStrategy` ABC (24), `RandomSearch` (38), `GridSearch` (13), `BayesianOptimization` (60), `EvolutionarySearch` (12) |
| `WindowOptimizer` class | `:567-634` | 68 | main coordinator |
| `run_bayesian_optimization()` | `:663-1006` | 344 | Bayesian entry point, 31 parameters |
| `run_with_config()` | `:1009-1236` | 228 | config-mode entry point, 13 parameters |
| CLI / `main()` | `:1239-1749` | **511** | 40 flags, backend mutex, execution-set freeze, P0.5 dataset gate, dispatch |

**Everything above `main()` is byte-for-byte where the correction pass left it.** The only row
that moved is the last: `main()` grew from **350 to 511 lines** (`:1239-1588` → `:1239-1749`) when
the Resolved Execution Set and admission binding landed. That is also why the flag count went
38 → 40 (§10.1).

Note the shape this reveals, now more pronounced than before: the two entry points and the CLI are
**1083 lines — 62% of the module** — while the data structures and strategies the chapter spends
most of its length on are **336**. The chapter's §§3–7 are not proportional to where the code is,
and the imbalance is growing.

**Key Insight:** The window optimizer doesn't run sieves directly — it delegates to the integration layer which coordinates real 26-GPU sieve execution.

---

## Next Chapter

**Chapter 2: Step 2 — the bidirectional sieve, and the engine that is replacing it.**

The pre-correction pointer promised a chapter on `sieve_filter.py` alone. That is no longer the
whole subject: **Step 2's engine is being replaced by RANGE-MINER (S172)**, a set of persistent
per-GPU daemons adopted after PWC suffered silent hard resets and `GCVM_L2_PROTECTION_FAULT` on
the RX 6600 XT rigs at full-fleet saturation. The replacement is an **interface** contract — the
22-array NPZ survivor contract — not a "match the old values" contract: the remaining steps must
not be able to tell which engine produced their input.

Chapter 2 must therefore cover both the legacy `sieve_filter.py` path and the miner that
supersedes it, including:

- forward/reverse sieve algorithms (**reverse kernels iterate the PRNG forward** — direction comes
  from reversing residues on the host; there is no inverse LCG, and this is not a defect)
- GPU memory management and residue-set computation
- the 22-array NPZ contract as the Step-1 → Step-2 → Step-3 carrier
- RANGE-MINER's stripe/sub-stripe model and where it diverges from the legacy path

> **Status — SUPERSEDED, and corrected at closure.** This paragraph previously read *"Chapter 2
> is a 128-line FRAGMENT, not a chapter … pending restore-and-audit … do not cite the fragment."*
> **That is no longer true.** `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` was **restored from
> `d14dcdd` and audited at `e1225a7`**, corrected at `e50e35f`, and **closed alongside this
> chapter at `81ef3f1`** — **1208 lines**, verified `wc -l` this session. It is now the reference
> for Step 2 and this chapter cites it for `offset` semantics (§3.1.2), the three-lane CRT test,
> and the hybrid skip-bound defect. The reconstruction map `docs/CHAPTER_2_SOURCE_MAP_v1.md`
> (651 lines) remains as the reconnaissance record.
>
> **The staleness itself is the lesson.** This paragraph was accurate when written and became
> false without anyone editing it, because it described *another document's state*. A chapter that
> asserts the condition of a sibling artifact acquires that artifact's maintenance burden. Prefer
> pointing at a document over grading it.

---

*End of Chapter 1: Window Optimizer*

---

## Persistent Worker Call Chain (S130/S134/S135)

When `--use-persistent-workers` is set, `window_optimizer_integration_final.py` routes through the
`run_trial_persistent()` shim in `persistent_worker_coordinator.py:1632` instead of the standard
coordinator path. **PWC is a non-certifying diagnostic backend** (§8.3.1, §11.3), and its hybrid
path is quarantined.

Call chain:
```
watcher_agent.py
  -> window_optimizer_integration_final.py  (use_persistent_workers=True, gate :1300-1304)
    -> run_trial_persistent()               (persistent_worker_coordinator.py:1632)
      -> PersistentWorkerCoordinator
            Remote:  _dispatch_to_worker()  (:976)
                     worker launched as `{WORKER_SCRIPT} --gpu-id N --persistent` (:613,
                     WORKER_SCRIPT = "sieve_gpu_worker.py" at :152)
```

**Two corrections to the pre-correction appendix, both verified this session:**

1. **`run_trial_persistent()` is at `:1632`, not `:669`** — the old anchor was off by 963 lines.
   *(It was `:1612` at the correction pass and has since moved by 20; cite it by name.)*
2. **The "Zeus: `execute_local_sieve_job()` → `sieve_filter.py`" leg has been removed from the
   diagram because no such function exists.** `/bin/grep -rn` across every `.py` in the tree
   returns **no definition** — only a prose comment (`persistent_worker_coordinator.py:17`) and
   the doc-patcher scripts that generated this appendix in the first place
   (`apply_s136_doc_updates.py:234`, `:464`; `apply_s146_doc_updates.py:248`). The appendix
   documented a call target that has never existed as code. Per §0.4 this is a **documentation**
   correction: nothing is being proposed for removal from the codebase, because there is nothing
   there to remove.

> **The "STANDALONE / zero changes" invariant is RETRACTED.** It read: *"Zero changes to
> `coordinator.py`, `window_optimizer.py`, or `window_optimizer_integration_final.py`."* That is
> false in all three named files and has been for a long time. `window_optimizer.py:759-761` and
> `:774` set `use_persistent_workers`, `use_zmq_sqlite`, `pwc_transport` and `use_range_miner` on
> the coordinator (and `:1087-1088` again in config mode); the flags are declared at `:1315-1330`;
> and `window_optimizer_integration_final.py` gates on them at `:1300-1304`. The *narrower*
> statement that remains true is that the PWC gate is **additive** — when the flag is off, the
> path below it is untouched (`:1286`). Do not rely on the original wording: a reader who
> believed it would expect PWC changes to be containable in one file.

### Optuna Resume

Flag: `--resume-study`, optionally with `--study-name <name>`. Full semantics — including the
`--study-name` priority rule the S114 patch predates — are in **§8.4.1**.

**Storage: SQLite.** The pre-correction appendix said *"JournalStorage (not SQLite)"*; that is
**backwards**. Storage is SQLite throughout (`window_optimizer_bayesian.py:641`, `:660`, `:691`),
and `:688-689` records the migration explicitly: *"S125: always SQLite (JournalFileBackend removed
— n_parallel parallelism now owned by multiprocessing dispatcher in integration layer;
n_jobs=1 here)."* Trial-unique output paths prevent cross-trial collisions.

> **The "active study" line has been removed as unverifiable, and re-checked at closure.** It named
> `window_opt_1772507547.db` (21 trials as of S132). **That database still does not exist** —
> confirmed absent from `optuna_studies/` again this session. Its trial count cannot be verified
> and no replacement figure is asserted.
>
> **Naming a "current" study in a chapter is a staleness generator, and the replacement figures
> proved it.** The correction pass recorded "61 `.db` files present, newest
> `window_opt_1778552567.db"`; at closure the directory holds **75** and the newest is
> `window_opt_1785633881.db`. Both counts were true when written and neither is worth carrying.
> **Do not restate a study inventory here** — use `--study-name` at run time and let auto-select
> (§8.4.1) handle the rest.

### enable_pruning / n_parallel Fix History (S116/S118/S123)

Both flags required fixes through the full call chain:
- CLI -> run_bayesian_optimization() signature (S116)
- -> optimize_window() signature -- enable_pruning was missing (S118)
- -> agent_manifests/window_optimizer.json args_map -- 4 keys missing (S123)

---

## Persistent Worker Mode — S146 Kernel Invariants

When `--use-persistent-workers` is active, Step 1 dispatches sieve jobs via
`persistent_worker_coordinator.py` → `sieve_gpu_worker.py`. The following invariants
were validated in S146 and must be preserved in any future modifications:

### Hybrid Kernel Arg Tails (CRITICAL)

```
Forward hybrid:  kernel_args = (..., threshold, a, c)
Reverse hybrid:  kernel_args = (..., threshold, offset)   # a,c hardcoded in kernel
```

These are **not interchangeable**. Passing `(threshold, a, c)` to a reverse hybrid kernel
causes an immediate crash.

### Threshold Invariant

Hybrid families use `phase2_threshold` for both kernel invocation and post-filter check.
Base threshold (`min_match_threshold`) is used only for constant-skip families.

### int32 Casts

All scalar kernel args must be explicitly cast: `cp.int32(n_seeds)`, `cp.int32(k)`,
`cp.int32(skip_min)`, `cp.int32(skip_max)`. ROCm/CuPy requires explicit types.

> **Correction — this invariant holds for constant-skip families ONLY.** It is true at
> `sieve_gpu_worker.py:214`, where the generic prefix is built. It is **false for hybrids**:
> `:259-268` (forward) and `:270-279` (reverse) rebuild `kernel_args` from scratch and never
> re-add `skip_min`/`skip_max`, then `continue` past the generic launch at `:298`. So the
> appendix asserts as a preserved invariant precisely the thing that is broken — this is dead
> dimensions **D-1/D-2** (§3.1.1) seen from the kernel side.
> Per §0.4 the remedy is **wire-in, not removal**, and it is the separately-briefed next
> deliverable. Until then, hybrid certification is blocked (§8.3.1).

### Count Clamp (defensive)

```python
count = min(int(survivor_count_gpu[0].get()), n_seeds)
```

Applied to both hybrid and non-hybrid extraction paths to prevent buffer overrun on
corrupt kernel writes.

---

## 17. Closure statement

### 17.1 Verified against

**Commit `81ef3f1`, 2026-08-02**, on VM 101 (`192.168.3.177`), working tree, venv
`~/venvs/torch`. The chapter's corrections pass ran against `40c3c83`; since then bounded Phase 6
(`d98298c`) changed `window_optimizer_bayesian.py` by **+233/−90**, and the Resolved Execution
Set (`63e627f`) and admission binding (`eff6616`) changed `window_optimizer.py` again. **This
pass re-verified rather than assumed.**

### 17.2 What is verified

**Every `file:line` anchor was checked against HEAD.** Live line counts at `81ef3f1`:
`window_optimizer.py` **1753**, `window_optimizer_bayesian.py` **1157**,
`window_optimizer_integration_final.py` **2703**.

The drift proved **systematic and explainable**, which is why it could be corrected rather than
re-derived: `window_optimizer.py` grew **only inside `main()`**, so every anchor above `:1239` is
unaffected and every anchor below it moved by a fixed amount; `window_optimizer_bayesian.py`
shifted by the `run_optimization` extraction; `window_optimizer_integration_final.py` shifted
+4 early / +15 late.

**Automated residual check.** After correction, all **289** parsed anchors across **11** cited
files resolve within their file's line count — zero anchors point past EOF, and zero cite a file
that does not exist. *This proves range validity, not content:* content was verified by reading
the cited lines section by section, and where a citation is stable by symbol the chapter now
**prefers the symbol** (§7.2.1 says so explicitly — `resolve_directional_threshold` *"has already
moved once without changing at all"*).

**The five items of the closure brief:**

| # | item | disposition |
|---|---|---|
| 1 | re-verify every anchor; prefer symbol citations | **done** — ~26 anchor groups corrected; symbol-first convention applied in §7.2.1, §8.1.1, §11.3, §12.1 |
| 2 | regenerate the §4.1 snapshot by script | **done, machine-generated, not hand-edited.** `scripts/extract_search_bounds_snapshot.py`; `repository_commit` `0c47fe3…` → `81ef3f1…`; **`configuration_digest` byte-identical** at `sha256:6077bb1a…2747cc` — the bounds themselves did not move |
| 3 | the sampler-neutral core | **done** — new **§8.1.2**: `run_optimization(..., *, sampler, sampler_metadata)`, both **required and keyword-only with no default**; TPE and Random are thin wrappers; `SAMPLER_ENTRYPOINTS` **deliberately unwired** from any advisor, WATCHER policy or `strategy_recommendation.json` — autonomous sampler selection is **reserved authority** |
| 4 | gated strategies still described as gated, not deleted | **verified, and completed.** §6.4 and §10.1 still describe them as CLI-gated and **not** deleted; the four-Optuna-sampler design intent is still recorded. **The `GridSampler` unconstructibility fact was missing and has been added** with the arithmetic executed this session |
| 5 | absorb Chapter 2's F-4 into C-2 | **done** — new **§3.1.2**. Settles C-2 as an **observed inconsistency, NOT the repair** |

**Clean control (VIR-2) — verified correct and unchanged, no edit required.** **62 of the
chapter's 95 sections** were re-checked and needed no change, including every section in §5
(scoring), §13 (agent metadata), §14 (method reference, all six subsections) and §15
(dependencies), plus §3.1/3.2/3.3, §4.2/4.3, §6.1–6.3, §7.1/7.3/7.4, §8.1/8.2/8.2.1/8.3/8.4.3/
8.4.4, §9.1/9.2/9.2.1, §10.2, §11.1, and the S146 kernel-invariant appendix. **A closure pass
that reported only its edits would give no evidence the rest was checked**, and "closed" would
then mean nothing.

**Fault-injection control (VIR-3) — the gate was run, and the fleet-state finding is recorded.**
`docs/CHAPTER_1_WINDOW_OPTIMIZER.md` is covered by `tests/test_chapter1_p0_corrections.py`, so
per the brief it was executed. **Final result: `SENTINEL : PASS`, 12/12** (6 gates + 6 mutants),
including the two mutants that specifically detect this pass's class of edit — **M5** (hand-edit
a bound in the chapter snapshot) and **M6** (remove the skip defect callout) — both correctly
red under mutation.

> **Recorded because it is worth knowing about the closure state: four of these twelve arms
> depend on a reachable GPU fleet.** Earlier in this same session the gate returned
> `SENTINEL : FAIL`, 8/12, with `G-FLAG-FAILCLOSED`, `G-STRATEGY-FAILCLOSED`, `M1` and `M2` all
> failing on one assertion — *"clean control: bayesian did not reach
> `run_bayesian_optimization`"* — because the **P0.5 dataset-provisioning preflight refused**:
> `No route to host` to `.122`/`.156`/`.164`. A ping sweep confirmed **all six rig addresses
> down**, both the Proxmox CT set and the bare-metal set.
>
> That was `UNAVAILABLE` under VIR-5 — *a required verification attempted and unable to
> complete* — **not** a regression, and the reading was **proven rather than asserted**, three
> ways:
> 1. **Empirically, single-variable.** The fleet came up; the *same edited chapter*, same gate,
>    same commit, went 8/12 → **12/12**. Only fleet reachability changed.
> 2. **Clean control.** The **pristine** chapter at `81ef3f1` (via `git stash`) under fleet-up
>    also returns **12/12** — so the edits neither introduce a failure nor mask one.
> 3. **Structurally.** Only `gate_snapshot_extracted`, `gate_skip_defect_note`, `M5` and `M6`
>    ever open the `CHAPTER` path
>    (`tests/test_chapter1_p0_corrections.py:64`, `:578-579`, `:646`, `:680-681`, `:707`). The
>    four arms that were red **never read the chapter file at all**, so a documentation edit
>    cannot reach them.
>
> **The standing fact this leaves:** these four arms are **not** a pure documentation gate. They
> shell out to the real CLI, and the fail-closed P0.5 dataset authority — correctly — refuses
> before dispatch when nodes are unreachable. **A green 12/12 therefore certifies the chapter
> *and* asserts a reachable fleet; a red one does not by itself indict the chapter.** Anyone
> re-running this gate must check fleet state before reading a failure as a documentation defect.
> Recorded as an observation for the gate owner, **not a proposal to change the gate.**

### 17.3 What remains open, and where it is tracked

**Nothing found this pass was repaired** — the brief is documentation-only, and no code, test,
config or manifest was touched.

| Open item | Where tracked | Status |
|---|---|---|
| **D-1 / D-2** — hybrid `skip_min`/`skip_max` sampled but never reaching the kernel | §3.1.1, §8.3.1, S146 appendix | **OPEN.** Remedy is **wire-in, not removal** (§0.4); separately briefed. Hybrid certification blocked until then |
| **D-3** — `offset` on the `java_lcg` forward hybrid | §3.1.1, §3.1.2 | **OPEN**, and see C-2 below |
| **D-4** — the two threshold CLI flags | §10.1 | **CLOSED as a silent no-op** — now **fail-closed**. The `args_map` mapping is dormant (§10.1) |
| **C-2 / Chapter 2 F-4** — `offset` drives host slice *and* device pre-advance from one scalar | §3.1.2; Chapter 2 §7.3 | **Settled as an OBSERVED INCONSISTENCY, not repaired.** Beta: no single `offset*(skip+1)` multiplier exists under variable skip; belongs in the future **hybrid input-semantics design**, not a standalone arithmetic patch |
| **Combined-session sampling can select a prohibited mode** | §8.3.1 | **re-verified OPEN at `81ef3f1`.** `session_options` still offers `['midday','evening']` first (`window_optimizer.py:182-186`) and Optuna still samples across all three (`window_optimizer_bayesian.py:513-515`, applied `:535`). An autonomous run can select a configuration that **cannot be certified**. Governance risk; code remedy out of scope |
| **`run_with_config` writes `[]` and prints success** | §9.3, §12.1 | **behavioural defect, separate ticket.** `accumulator['forward']`/`['reverse']` are never appended to; the run prints *"✅ Saved 0 forward survivors"*. The Bayesian path degrades **honestly** (writes a `note`); this path degrades **silently** |
| **TRSE `except` handler would itself raise `NameError`** | §8.4 note (`:1075-1077`) | **separate ticket. `UNAVAILABLE` — unverified at runtime.** The "non-fatal" path calls a `logger` the scope does not define |
| **`min_workers` has two defaults for one quantity** | §9.1 note | **OPEN** — **1** in the signature (`window_optimizer.py:1022`) vs **24** at the CLI (`:1323`). Reported, not resolved |
| **`args_map` maps two now-fail-closed flags** | §10.1 | dormant — `default_params` supplies neither key; would abort a run if either were added. Flagged for the manifest owner |
| **`parameter_bounds` is a live admission gate looser than §4.1** | §10.1 | **new this pass, OPEN.** `agents/step_runner/command_builder.py:151-170` validates against it: `window_size` min **2** where the S172 TB ruling raised the search floor to **6**; `offset` max 2000 vs 100; `skip_min` max 50 vs 10; `skip_max` 20–500 vs 10–250. It cannot widen the Optuna space, but **would accept a WATCHER-proposed `window_size=2`** — precisely the value S172 excluded as chance-driven |
| **`--rig-profile` / `--execution-set-nodes` absent from `args_map`** | §10.1 | **new this pass.** WATCHER cannot select a rig profile or declare a partial execution set; a WATCHER-launched run takes `default_profile` and the full declared fleet. Governance question for the manifest owner, **not a defect this chapter asserts** |
| **`GridSampler` unconstructible at live bounds** (7.649 × 10¹⁰ points ≈ 7.23 TiB) | §10.1 | **new this pass.** The documented four-sampler design is not implementable as stated for Grid without an explicitly coarsened grid — a **design decision for Beta**, not a restoration |
| **The strategy repair itself** | §10.1 | pending Beta ruling (`TEAM_ALPHA_AUTONOMY_CONTROL_SURFACE_SUBMISSION.md` Q3). Two prescriptions are explicitly **ruled out** and recorded so they are not re-proposed |
| **Sampler provenance is unverified**; TPE remains default by status quo | §8.1.2 | stated limits of the neutral core |
| **Four arms of the executable gate require a reachable fleet** | §17.2 | new this pass; observation for the gate owner |

### 17.4 What this chapter is NOT

- **Not an operator runbook for the gated strategies.** `random`, `grid` and `evolutionary` are
  documented **because they are broken and must not be silently re-enabled**. Nothing here is a
  procedure for running them; two proposed repairs are explicitly ruled out, and per §0.4 none
  of the three is a candidate for deletion.
- **Not a code version.** The "revision 3.1" on the header is documentation-only and appears in
  no source file (the live module docstring says `Version: 2.0`).
- **Not a certification of the hybrid path.** D-1/D-2 remain unwired; hybrid certification is
  blocked (§8.3.1), and §12.2's example config is **illustrative shape only** — two of its values
  are unreachable under live bounds.
- **Not an authority on the bounds.** §4.1 is a **dated, machine-generated snapshot**; the live
  authority is `distributed_config.json` read through `SearchBounds.from_config()`. Where the
  snapshot and the manifest disagree, that disagreement is a finding (§10.1), not a licence to
  pick one.
- **Not a claim that Step 1 is fully verified.** It documents Step 1 **as built**, including
  where as-built diverges from its own manifest and from the documented design.

### 17.5 Closure sentinel

```
CHAPTER 1 CLOSURE:  PASS
```

**`PASS` means verified-and-bounded, not finished.** It is claimed for exactly this scope: every
anchor re-verified against `81ef3f1` with the drift corrected, the §4.1 snapshot machine-
regenerated, the sampler-neutral core documented, the gated strategies confirmed gated and not
deleted (and the missing `GridSampler` fact supplied), Chapter 2's F-4 absorbed into C-2, the
executable gate run **green 12/12 including both chapter-reading mutants**, and every open item
enumerated in §17.3 with where it is tracked. It is **not** a claim that Step 1 is verified, that
the hybrid path is certified, or that any dead dimension, ticket or conflict was repaired.

**Files changed by this pass:** `docs/CHAPTER_1_WINDOW_OPTIMIZER.md` and
`docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` only. No code, tests, config or manifests.
