# SAMPLER_BEARING_v1 — cost and blast radius of four working Optuna samplers in Step 1

**Date:** 2026-07-31 · **Host:** VM101 (`michael@192.168.3.177`) · **Tree:** `/home/michael/distributed_prng_analysis`
**HEAD at bearing:** `179f7cd` · **venv:** `~/venvs/torch` · **Optuna:** **4.4.0**
(`/home/michael/venvs/torch/lib/python3.10/site-packages/optuna/__init__.py`)
**Authority:** `docs/CLAUDE_CODE_INSTRUCTIONS_SAMPLER_BEARING.md` (REV1)
**Mode:** READ-ONLY scoping. No code, config or documentation was modified; this file is the only
artifact created. No Optuna study was started. No pipeline, sieve, GPU kernel or WATCHER run.
**This is an estimate to inform a decision. It is not an implementation and not an authorisation.**

Established facts are cited from `docs/STRATEGY_ORIGIN_AUDIT.md` (HEAD `ddd2ac8`), not re-derived.

---

## 0. Headline

Two findings dominate and they point in opposite directions.

1. **The signature problem is moot under the natural implementation — yes, decisively (§3).** The
   calling machinery for all four strategies is *already built and already shared*. The CLI already
   routes `random`, `grid` and `evolutionary` through the same `run_bayesian_optimization`
   entrypoint that `bayesian` uses. If the three become thin delegates like `BayesianOptimization`,
   they inherit its signature, `strategy_contract_gap()` returns `()` for them, and the four
   forwarded kwargs stop being a problem — **and**, because the delegate has a real body, this
   simultaneously closes the vacuous-pass hole the origin audit raised at its §6(ii).

2. **`GridSampler` is not tractable here — not "expensive", mathematically unconstructible (§4.2).**
   The cartesian product over live bounds is **7.649 × 10¹⁰** points. Optuna 4.4.0 materialises the
   *entire* grid as a Python list in `GridSampler.__init__` (`_grid.py:120`) and shuffles it
   (`:124`). That is **≈ 7.2 TiB** on a **15 GiB** box. `GridSampler` cannot be constructed at all,
   independent of trial budget. Grid over the real space is off the table on arithmetic.

Everything else is bounded and ordinary. The effort is **small for the sampler mechanism itself and
medium for the shippable change**, and the honest answer to "what is gained" is **little, except for
one genuinely useful thing: a random control arm** (§8).

**Per `tfm-project-facts` §0.4 this report recommends removing nothing.** Two of these three were
already destroyed once by deletion; removal is a Beta ruling, never an audit conclusion. Where a
strategy is mathematically blocked, the finding is stated as a constraint on *how* it must be
specified, not as a case for retiring it.

---

## 1. How the working path builds its sampler — **hard-wired, single call site**

`BayesianOptimization` does not touch Optuna. It delegates (origin audit §7):

```
window_optimizer.py:433-436   from window_optimizer_bayesian import OptunaBayesianSearch
                              self.optuna_search = OptunaBayesianSearch(...)
              :452-455        return self.optuna_search.search(...)
```

All Optuna work is in `OptunaBayesianSearch` (`window_optimizer_bayesian.py:364`):

| What | Anchor | State |
|---|---|---|
| Sampler import | `:52` `from optuna.samplers import TPESampler` | **`TPESampler` is the only sampler imported anywhere** |
| Sampler construction | `:543-547` `sampler = TPESampler(n_startup_trials=…, seed=…, multivariate=True)` | **one** call site, **hard-wired** |
| `create_study` | `:616-623` `sampler=sampler` | **one** call site |
| Constructor | `:367-368` `__init__(self, n_startup_trials=5, seed=None, enable_pruning=False, n_parallel=1)` | **no sampler parameter** |

**Answer: hard-wired, not a parameter.** But it is hard-wired in exactly *one* place, feeding exactly
*one* `create_study`. That is the best possible shape for this change: making it a parameter is a
local edit, not a refactor.

**The rest of the call chain is already sampler-agnostic and already shared by all four strategies.**
This is the part that was not visible from the second-hand line reference, and it is what makes the
estimate come out small:

```
window_optimizer.py:1443           --strategy bayesian     ─┐
                   :1491-1507      --strategy random       ─┤
                   :1512-1529      --strategy grid         ─┼─► run_bayesian_optimization(
                   :1532-1548      --strategy evolutionary ─┘        …, strategy_name=<name>)
window_optimizer.py:663                                             (the shared entrypoint)
window_optimizer_integration_final.py:2392-2402                     strategy_map[name] → instance
                                     :2410-2411                     require_supported_strategy(name)
                                     :2446-2457                     optimizer.optimize(strategy=…, +4 kwargs)
window_optimizer.py:622                                             strategy.search(…, +4 kwargs)
```

All four CLI branches already call the **same** function with **all** the same arguments, differing
only in the `strategy_name` string. Nothing upstream of the strategy object needs to change.

---

## 2. Minimal change set

### 2.1 Sampler construction — `window_optimizer_bayesian.py`

| # | Anchor | Change | Size |
|---|---|---|---|
| C1 | `:367-368` `__init__` | add a sampler selector parameter (e.g. `sampler_name='tpe'`) | 1 line |
| C2 | `:543-547` | branch: `TPESampler(...)` / `RandomSampler(seed=…)` / `GridSampler(search_space, seed=…)` / `CmaEsSampler(...)` | ~15 lines |
| C3 | `:52` | import the additional samplers | 1 line |
| C4 | `:407-409` | banner hardcodes `"BAYESIAN OPTIMIZATION (Optuna TPE)"` and `"Startup trials"` — must reflect the actual sampler | 3 lines |
| C5 | **`:708` and `:728`** | `'strategy': 'optuna_bayesian'` is **hardcoded in both return paths** | **not optional — see below** |

**C5 is a correctness requirement, not cosmetics.** Routing a `RandomSampler` run through this class
without changing `:708`/`:728` would record it in `optimal_window_config.json` and downstream as
`optuna_bayesian`. That is the *same* semantic-substitution failure that the S178
`StrategyContractError` at `window_optimizer.py:465-473` was written to prevent, only inverted —
Beta's stated objection there was that a study "would record the run as Bayesian while a different
algorithm chose every point." Any four-sampler change that does not parameterise `:708`/`:728`
reintroduces exactly that defect.

`n_startup_trials` (`:544`) is TPE's parameter. `RandomSampler` and `GridSampler` do not accept it;
`CmaEsSampler` does. Verified live signatures:

```
RandomSampler (self, seed=None)
GridSampler   (self, search_space: Mapping[str, Sequence[GridValueType]], seed=None)
CmaEsSampler  (self, x0=None, sigma0=None, n_startup_trials=1, independent_sampler=None,
               warn_independent_sampling=True, seed=None, *, consider_pruned_trials=False,
               restart_strategy=None, popsize=None, inc_popsize=-1, use_separable_cma=False,
               with_margin=False, lr_adapt=False, source_trials=None)
```

### 2.2 The strategy classes — `window_optimizer.py`

| Class | Anchor | Today | Becomes |
|---|---|---|---|
| `RandomSearch` | `:364`, `search` `:366-401` | **real hand-rolled loop**, four-positional signature | thin delegate |
| `GridSearch` | `:403`, `search` `:410-412` | `# Placeholder` / **`return {}`** | thin delegate |
| `EvolutionarySearch` | `:478`, `search` `:484-486` | `# Placeholder` / **`return {}`** | thin delegate |
| `BayesianOptimization` | `:417`, `search` `:437-476` | delegate — **the template** | unchanged |

Each becomes a ~20-line copy of `BayesianOptimization` (`:417-476`) differing only in the
`sampler_name` it passes and its `name()` return. The `_survivor_accumulator` hand-off at `:444-446`
must be replicated verbatim in each — it is load-bearing (S152) and easy to miss.

`RandomSearch`'s existing hand-rolled body is **working code** (origin audit §3.4: it was never
degraded). Converting it to a delegate replaces a working uniform-random loop with an Optuna
`RandomSampler` run. That is a behaviour change even though both are "random": the delegate gains
study storage, resume, warm-start enqueue, trial history and pruning; the hand-rolled loop has none
of these. Whether the existing body is retained under another name is a design question for Beta,
not an audit conclusion — flagged, not decided.

### 2.3 The calling convention

No change required. See §3.

### 2.4 Supporting call sites

| Anchor | Note |
|---|---|
| `window_optimizer_integration_final.py:2392-2402` | `strategy_map` constructor args. `GridSearch(window_sizes=[512, 768, 1024], offsets=[0, 100], skip_ranges=[(0,20),(0,50)])` — these are **stale**: live `window_size` bounds are **6–50** (§4.2), so every one of `512/768/1024` is out of bounds. Harmless today because the body returns `{}`; becomes live input the moment the body is real. |
| `window_optimizer.py:511-515` `STRATEGY_CLASSES` | no change — names already map |
| `window_optimizer.py:1249` argparse `choices` | no change — already offers all four |
| `window_optimizer.py:518-529` `strategy_contract_gap` | no change — it reads live signatures and clears itself (§3) |

---

## 3. Does the natural implementation make the signature problem moot? — **YES**

**This is the most important finding in the report, and it is stronger than the brief anticipated.**

Live clean control run this session on `179f7cd`:

```
OPTIMIZE_FORWARDED_KWARGS = ('resume_study', 'study_name', 'trse_context_file', 'trial_history_context')
bayesian       gap=()
evolutionary   gap=('resume_study', 'study_name', 'trial_history_context', 'trse_context_file')
grid           gap=('resume_study', 'study_name', 'trial_history_context', 'trse_context_file')
random         gap=('resume_study', 'study_name', 'trial_history_context', 'trse_context_file')
```

`strategy_contract_gap` (`window_optimizer.py:518-529`) computes the gap from
`inspect.signature(strategy_cls.search)` against `OPTIMIZE_FORWARDED_KWARGS` (`:334-336`). A thin
delegate copies `BayesianOptimization.search`'s signature (`:437-440`), which is `gap=()` today.
**Therefore the gate clears itself the moment the delegate exists — the four forwarded kwargs are
never separately addressed, because the delegate accepts them by construction.**

Three consequences:

1. **The four-kwargs mismatch disappears as a side effect, not as a work item.** It costs zero
   additional effort under the Optuna-delegate implementation.

2. **It also closes the origin audit's §6(ii) objection.** That audit's concern was that following
   the `ddd2ac8` remedy comment (`window_optimizer.py:492-508`) would turn a signature-derived gate
   green on a function that still returns `{}` — a VIR-2 vacuous pass. Under the delegate
   implementation there is no `return {}` left to hide behind: the delegate's body is
   `OptunaBayesianSearch.search`. **The delegate route is the one repair that satisfies the gate and
   the audit's objection at the same time.** The "patch four kwargs onto three hand-rolled bodies"
   route satisfies only the gate.

3. **It matches the documented design.** `PROPOSAL_SEARCH_STRATEGY_VISIBILITY_FIX_v1_0.md:20` —
   *"All 4 Optuna samplers (TPE, Random, Grid, CmaES) implemented ✅"* — and origin audit §7.2: an
   Optuna-backed strategy in this codebase **is** a thin wrapper over a `create_study(sampler=…)`
   call. The natural implementation and the documented intent are the same implementation.

**The signature problem is not the cost of this work. The cost is elsewhere (§4, §5, §6).**

---

## 4. Per-sampler constraints

### 4.1 `RandomSampler` — no constraint. Confirmed.

Signature is `(seed=None)`. It imposes no requirement on the search space: no explicit enumeration,
no distribution-type restriction, no dimensionality floor, no external dependency. Constructed
successfully this session. **This is the straightforward one, exactly as the brief expected.**

### 4.2 `GridSampler` — **NOT TRACTABLE.** Mathematically blocked.

**Live bounds**, read via `SearchBounds.from_config('distributed_config.json')` on `179f7cd`
(`window_optimizer.py:134`, `:159-176`; config key `search_bounds`):

| dimension | live bounds | grid points |
|---|---|---|
| `window_size` | 6 – 50 | 45 |
| `offset` | 0 – 100 | 101 |
| `session_idx` | 0 – 2 (3 session options) | 3 |
| `skip_min` | 0 – 10 | 11 |
| `skip_max` | 10 – 250 | 241 |
| `forward_threshold` | 0.30 – 0.75, continuous; rounded to 2 dp at `window_optimizer_bayesian.py:451` | 46 @ 0.01 |
| `reverse_threshold` | 0.30 – 0.75, same | 46 @ 0.01 |

```
45 × 101 × 3 × 11 × 241 × 46 × 46  =  76,485,750,660  ≈  7.649 × 10¹⁰
```

**The decisive fact is not the trial count — it is that the sampler cannot be built.**
`optuna/samplers/_grid.py:120`:

```python
self._all_grids = list(itertools.product(*self._search_space.values()))
...
:122  self._n_min_trials = len(self._all_grids)
:124  self._rng.rng.shuffle(self._all_grids)
```

The **entire** grid is materialised as a Python list in `__init__` and then shuffled — both require
the whole thing resident. At 104 bytes per point (7-tuple + list slot, *excluding* the float objects):

```
7.649e10 × 104 B  =  7.95 × 10¹² B  =  7.23 TiB
```

VM101 has **15 GiB** total (`free -g`). **`GridSampler(search_space)` over the live space raises
`MemoryError` in the constructor.** No trial budget, no runtime, no wall-clock argument enters into
it. For scale if it *could* be built: at 1 min/trial the full grid is **1.45 × 10⁵ years**.

Coarsening does not rescue it:

| discretisation | points | memory |
|---|---|---|
| thresholds at 0.05 instead of 0.01 | 3.61 × 10⁹ | ~0.34 TiB — still impossible |
| aggressive hand-picked (5×5×3×3×5×4×4) | 18,000 | trivial — but this is a different object |

**The finding, stated precisely:** `GridSampler` cannot enumerate Step 1's real search space. It can
only run against a **hand-authored coarse discretisation** — a small explicit list of values per
dimension. That is not an implementation detail; it is an editorial claim about which parameter
values matter, and it would silently define the search space that a governed, autonomous surface
(`agent_manifests/window_optimizer.json:142-149`) offers as "grid". **Specifying that discretisation
is a governed decision that must precede any code, and it is Beta's, not Alpha's.**

**This is a constraint on how `grid` must be specified, not a case for removing it** (§0.4). The
strategy was working hand-rolled code deleted in November 2025 (origin audit §3.4); its documented
intent stands (`PROPOSAL_…_v1_0.md:20`). What this section retires is the assumption that `grid`
means "enumerate the search space" — here it cannot, and never could.

Two further mechanical notes for whoever specifies it:

- `_grid.py:186-189` raises `ValueError("All parameters must be specified when using GridSampler
  with enqueue_trial.")` if an enqueued trial leaves any parameter unsampled. The live warm-start
  (`window_optimizer_bayesian.py:639-645`) enqueues **all seven**, so it would not raise — but the
  enqueued point need not lie on the grid, so it consumes a trial without covering a grid cell.
- `_grid.py:196-200` warns when a grid value falls outside the live distribution. TRSE Rule A
  (§5.3) mutates `bounds.max_window_size` at runtime, so the `search_space` dict **must** be built
  *after* Rule A or this fires on every trial.

### 4.3 `CmaEsSampler` — mechanically compatible, semantically wrong on one axis, and **missing its dependency**

**(a) The categorical objection does not bite — but not for a reassuring reason.**
`_cmaes.py:72` and `:351` state `CategoricalDistribution` is unsupported;
`infer_relative_search_space` (`:339-355`) silently drops any distribution that is not
`FloatDistribution` or `IntDistribution`. Step 1's objective (`window_optimizer_bayesian.py:417-442`)
uses **`suggest_int` ×5 and `suggest_float` ×2 — and no `suggest_categorical` at all**. So CMA-ES
accepts all seven dimensions. The space is *semantically* mixed but *mechanically* all-numeric.

**(b) `session_idx` is a categorical wearing an integer's clothes, and CMA-ES would believe it.**
`:426-428` samples `session_idx` as an int in `[0, 2]`; `:448` uses it as
`bounds.session_options[session_idx]` → `[midday, evening]` / `[midday]` / `[evening]`. CMA-ES fits a
Gaussian with a full covariance matrix over its dimensions — it will treat this axis as ordered and
continuous and interpolate along it. There is no metric sense in which `[midday]` lies *between*
`[midday, evening]` and `[evening]`. CMA-ES would spend covariance-model capacity on a gradient that
does not exist, and TPE with `multivariate=True` (`:546`) does not have this problem in the same way
because it models a density over discrete values rather than a metric. The other six dimensions are
genuinely ordinal and are fine.

This is a *quality-of-inference* defect, not a crash. It would not be visible in any output.

**(c) `with_margin=True` is the option for integer dimensions** and would have to be set
deliberately (`_cmaes.py` doc: it "prevents samples in each discrete distribution … from being fixed
to a single point"). Default is `False`. Five of seven dimensions are integers, so leaving the
default is likely wrong.

**(d) Dynamic search space → silent degradation to random.** `_cmaes.py:390-400`: if
`optimizer.dim != len(trans.bounds)`, CmaEs logs a warning and **returns `{}`**, handing the trial to
`self._independent_sampler`, which defaults to `RandomSampler` (`:277`). CmaEs uses
`IntersectionSearchSpace` (`:281`), which drops parameters whose distribution differs across trials.

**Verified: under live bounds the space is STATIC, so this does not fire.** `skip_max`'s lower bound
is `max(skip_min, bounds.min_skip_max)` (`:433`); with live `skip_min ∈ [0,10]` and
`min_skip_max = 10`, that expression evaluates to `10` for every legal `skip_min`. Checked
exhaustively over `skip_min ∈ 0..10` → single value `{10}` → **STATIC**. This also independently
confirms the S119 comment at `:539` (*"Safe: search space is static (skip_max lower bound
always=10)"*) as **accurate** — a clean control on existing machinery.

**But it is bounds-conditional, not structural.** If `distributed_config.json` ever sets
`skip_min.max > skip_max.min`, the space becomes dynamic, `skip_max` drops out of the intersection,
and CmaEs degrades to `RandomSampler` — reported as a **log line**, not a fail-closed. A study would
still record itself as an evolutionary/CmaEs run. Under four-sampler support this becomes a live
semantic-substitution risk gated only on a config edit that nothing currently forbids.

**(e) The `cmaes` package is NOT INSTALLED in `~/venvs/torch`, and it fails LATE.** Verified this
session:

```
cmaes NOT INSTALLED: No module named 'cmaes'
CmaEsSampler constructed OK        ← constructor does not touch it
```

`_cmaes.py:37` binds `cmaes = _LazyImport("cmaes")`. First real use is `_init_optimizer`
(`:532`/`:537`) and `sample_relative` (`:411`), reached only **after** `n_startup_trials` completed
trials (`:369-371`). So a `--strategy evolutionary` run on the current environment would:
construct fine → build the 26-GPU coordinator → run `n_startup_trials` **real sieve trials** →
then raise `ModuleNotFoundError` mid-study.

**That is precisely the fail-late shape S178 P0-2 exists to eliminate** — the
`WINDOW_OPTIMIZER_STRATEGY_UNSUPPORTED` diagnostic (`window_optimizer.py:551-560`) is worded around
"it would raise TypeError on the first trial — after the coordinator was built." Any CmaEs support
must add an explicit availability pre-check at strategy-resolution time, not rely on the lazy import.

Installing `cmaes` is a **dependency change to VM101**, which under `CLAUDE.md` §4 obliges committing
it to the reproducible `requirements`/setup artifact. Small, but it is a real item and it touches the
fallback-parity surface (§5 of `CLAUDE.md`).

**(f) Budget.** CMA-ES needs a full population per generation before it learns anything;
`popsize` defaults to `None` → the `cmaes` library default, which scales with dimension. At 7
dimensions that is roughly a dozen trials per generation. Production trial counts are tens. CMA-ES
would complete only a handful of generations — see §8.

*Note on naming (origin audit §5.2): the one documented mapping names **CmaES** for `evolutionary`,
not NSGAII. `NSGAIISampler` appears nowhere in the repository or its history. Optuna 4.4.0 does ship
`NSGAIISampler`, `GPSampler`, `QMCSampler`, `BruteForceSampler` and others, but selecting any of them
would be a new design choice with no documented basis, and is out of scope here.*

---

## 5. Interactions with existing machinery

### 5.1 Study storage and resume — **a new provenance hole**

`create_study(..., sampler=sampler, load_if_exists=_resume)` (`:616-623`); resume logic `:554-611`.

**Optuna does not persist the sampler in storage.** A resumed study takes whatever sampler is
constructed *now*. Today that is invisible because there is only ever one sampler. Under four-sampler
support, `--resume-study --study-name window_opt_X` where `X` was created under `random` would
**silently continue under TPE** if the resuming invocation says `bayesian` — and nothing in the DB,
the config output or `step1_trial_history` would record the switch. The origin audit already noted
(§9, unavailable surfaces) that *"Optuna does not reliably persist sampler class in storage."*

This is a **new** semantic-substitution surface created by the change, in the same family as the ones
S178 P0-2 closed. It would need the sampler recorded in the study's `user_attrs` and checked on
resume. Small code, but it is a fail-closed obligation, not a nicety.

### 5.2 Warm-start enqueue — one sampler-specific caveat

`study.enqueue_trial(_ws_params)` at `:645`, params assembled `:630-644` from `trial_history_context`
(seven parameters, all present or the warm start is skipped at `:647-648`).

- `RandomSampler`, `CmaEsSampler`: no issue.
- `GridSampler`: no raise (all seven supplied, per `_grid.py:186-189`), but the enqueued point need
  not lie on the grid and consumes a trial without covering a cell. See §4.2.

### 5.3 TRSE Rule A bounds mutation — an **ordering constraint** for Grid only

`window_optimizer_bayesian.py:508-517` mutates `bounds.max_window_size` in place (`SearchBounds` is a
dataclass) when `regime_type == 'short_persistence'` and `confidence ≥ 0.70` and `regime_stable`.

This runs at `:508`, **before** the sampler is constructed at `:543` and before `create_study` at
`:616`. For `RandomSampler` and `CmaEsSampler` that ordering is already correct and nothing changes —
they read bounds through the objective's `suggest_*` calls, which happen after.

For `GridSampler` the `search_space` dict is a *constructor argument*, so it must be built after
`:517` and before `:543`. Get that wrong and every trial trips the out-of-range warning at
`_grid.py:196-200`. It is a real constraint but a cheap one, and the current code layout already puts
the two in the right order.

Rules B and C are logged-only and disabled per TB S121 (`:523-533`) — no interaction.

### 5.4 Trial-history callback — **sampler-agnostic, verified clean**

`create_incremental_save_callback` reads only generic Optuna study API: `study.trials`,
`study.best_trial`, `study.best_value`, `study.best_params`, `trial.number`, `trial.value`. It
contains **no TPE-specific assumption**. `_prune_telemetry` (`:673-679`) likewise. `study.optimize(…,
n_jobs=1)` (`:681-683`) is sampler-neutral. **No change required.** This is a clean control on the
claim that the surrounding machinery does not assume TPE.

### 5.5 `optimal_window_config.json` output shape — unchanged shape, **wrong label**

The callback writes `window_size`/`offset`/`skip_min`/`skip_max`/`forward_threshold`/
`reverse_threshold` from `study.best_params` — identical keys for any sampler. **Shape is unaffected.**

The `strategy` field is the problem: `'optuna_bayesian'` hardcoded at **`:708`** (all-pruned return)
and **`:728`** (normal return). See §2.1 C5 — this is a correctness requirement.

### 5.6 Pruning

`ThresholdPruner(lower=1.0)` when `enable_pruning` (`:614`). Orthogonal to sampler choice, but two
notes: CmaEs ignores pruned trials unless `consider_pruned_trials=True` (default `False`), and a
pruned trial under GridSampler still consumes its `grid_id`.

---

## 6. Blast radius

### 6.1 Surfaces that would change

| Surface | Anchor | Nature |
|---|---|---|
| `window_optimizer_bayesian.py` | `:52`, `:367-368`, `:407-409`, `:543-547`, `:708`, `:728` | sampler parameter, branch, labels |
| `window_optimizer.py` | `:364-401`, `:403-415`, `:478-489` | three class bodies → delegates |
| `window_optimizer_integration_final.py` | `:2392-2402` | `strategy_map` constructor args (stale `GridSearch` args, §2.4) |
| **`tests/test_chapter1_p0_corrections.py`** | **`:228-276`, `:278-292`** | **inverted by design — see 6.2** |
| `agent_manifests/window_optimizer.json` | `:142-149` | already declares all four; **no schema change**. `effect` text would need updating |
| `docs/CHAPTER_1_WINDOW_OPTIMIZER.md` | §6.4 (`:600-605`), and the `~:965-985` *"code rot, not design"* text | documentation, per origin audit §5.4 |
| `window_optimizer.py:492-508` | the `ddd2ac8` remedy comment | already known misleading (origin audit §6); being corrected separately |

### 6.2 The test gate is the largest single cost — and it inverts

`gate_strategy_failclosed` (`tests/test_chapter1_p0_corrections.py:228`) asserts, for each of
`random`, `grid`, `evolutionary`:

```
:239-241  assert gap, f"{broken}: expected a signature gap, live signature accepts all"
:250      assert rc != 0, f"--strategy {broken} exited 0"
:251-252  assert "WINDOW_OPTIMIZER_STRATEGY_UNSUPPORTED" in blob
:257-258  assert COORDINATOR_MARKER not in blob
```

**Every one of these goes RED the moment the strategies work.** That is correct behaviour — the gate
is doing its job — but it means the change cannot land without rewriting the gate's premise from
"these three must fail closed" to "these three must run and record their own sampler." The
fault-injection control `mutant_strategy_random_permitted` (`:278-292`) becomes meaningless and needs
a replacement mutant (the natural one: a mutant that mislabels a non-TPE run as `optuna_bayesian`,
which the new gate must catch).

Authoring a new gate **with a working fault-injection control** to VIR-2 standard is the bulk of the
honest effort in this change. The clean control at `:269-276` (bayesian still runs) survives as-is.

### 6.3 Gate 22 — **would go RED while uncommitted**

`gate22_coexistence` (`tests/test_s172_phase4_coordinator.py`, dispatched at `:3527`) reads
`git status --porcelain` and asserts every changed `.py` is in an explicit allowlist. The allowlist
contains `window_optimizer_integration_final.py` — but **not `window_optimizer.py` and not
`window_optimizer_bayesian.py`** (verified: 0 matches). Both are core to this change, so gate 22 reds
until they are either committed or added to the allowlist. Expected and mechanical, but it is a
required step, and it compounds the known sensitivity of this gate to working-tree state.

### 6.4 Does anything touch the miner, PWC, ZMQ, the kernels or the 22-array contract?

**Confirmed: NO.** Alpha's expectation holds.

The reasoning, traced across the boundary rather than assumed: the sampler determines only *which
point in the search space is proposed*. The proposed values are assembled into a `WindowConfig` at
`window_optimizer_bayesian.py:445-453`, which is **downstream of sampling and unchanged**. Everything
below that — `objective_function(config, optuna_trial=trial)` (`:456`) → `test_configuration`
(bound at `window_optimizer_integration_final.py:2390` from `test_config`, `:2349`) →
`run_bidirectional_test` → miner or PWC —
receives a `WindowConfig` with identical fields and identical types regardless of which sampler
produced them. No kernel signature, no NPZ array, no protocol message and no threshold-resolution
path is in the change set.

**One consequential caveat, and it is not structural but it is real.** Different samplers would drive
the *existing* machinery to combinations TPE has never proposed. Two known open defects live exactly
where that would land:

- **`tfm-project-facts` §2.7 #4 (OPEN):** hybrid kernels ignore sampled `skip_min`/`skip_max`;
  `expected_skip` is hardcoded to 5. `skip_min` × `skip_max` is **2,651** of the 7.6 × 10¹⁰ grid
  points' worth of axis — a large fraction of the space's *nominal* dimensionality. A sampler that
  explores skip more uniformly than TPE (`RandomSampler` certainly; `GridSampler` by construction)
  would spend proportionally **more** of the budget steering a knob connected to nothing on the
  hybrid route. Forward hybrids ignore `offset` too — same class.
- Under `test_both_modes`, that dead-dimension exposure applies to half the evaluated configurations.

This does not change the blast radius of the *code*, but it means **four-sampler support would make
the §2.7 #4 dead dimension worse in practice before it makes anything better**, and the approved
sequence already places the skip-bound wiring ahead of discretionary work.

---

## 7. Effort estimate

**Overall: SMALL for the sampler mechanism; MEDIUM for a shippable change.** Alpha's prior expectation
of "small" is **confirmed for the sampler parameter itself and refuted for the deliverable.**

| Component | Size | Why |
|---|---|---|
| Sampler parameter + branch (`:367`, `:543`) | **Small** | one construction site, one `create_study`, no refactor. This is the part Alpha's expectation was about, and it is correct. |
| Three thin delegate classes | **Small** | ~20 lines each, `BayesianOptimization:417-476` is a complete template. Watch the `_survivor_accumulator` hand-off. |
| Four forwarded kwargs | **Zero** | solved as a side effect (§3) |
| Label parameterisation (`:407`, `:708`, `:728`) | **Small but mandatory** | correctness, not cosmetics (§2.1 C5) |
| Sampler-on-resume provenance (§5.1) | **Small** | new fail-closed obligation |
| `RandomSampler` end to end | **Small** | no constraints, dependency present |
| `CmaEsSampler` end to end | **Small–Medium** | `pip install cmaes` + env capture, early-availability check, `with_margin` decision — plus an unresolved design question (§4.3b) that is not code |
| **`GridSampler` end to end** | **BLOCKED** | not an effort estimate. Requires a governed discretisation decision before any code exists (§4.2) |
| **Test gate rewrite (§6.2)** | **Medium — the bulk** | inverting `gate_strategy_failclosed` and authoring a new fault-injection control to VIR-2 |
| Documentation (Chapter 1 §6.4, remedy comment, manifest `effect`) | **Small** | already queued separately |

Sequenced honestly: **`random` alone is a small, self-contained, low-risk change.** `evolutionary`
via CmaEs is small code sitting on an open design question and a dependency addition. `grid` is not
schedulable until Beta specifies a discretisation.

---

## 8. What would actually be gained — **little, with one real exception**

The brief invites "possibly nothing." That is close to the truth, but not the whole of it.

**Baseline fact:** TPE has run every production trial. The only strategy value ever recorded in a live
artifact is `"strategy": "optuna_bayesian"` (origin audit §7); no artifact records a random, grid or
evolutionary run.

**`RandomSampler` — modest but genuine. This is the one with real value.**
There is currently **no way to answer "is TPE actually beating random?"** on this objective. That is a
live methodological gap, not a hypothetical: the Step 1 objective is a survivor count under loose
thresholds (`tfm-project-facts` §0.3) — a noisy, heavy-tailed signal on a 7-dimensional space with
tens of trials per study. That is precisely the regime where TPE's advantage over random is smallest
and least certain, and where the standard practice is to run a random control arm. A random arm would
also make the warm-start (`:630-646`) falsifiable: it is currently impossible to tell whether
warm-started TPE is finding good windows or rediscovering the enqueued point. **This is a real
experimental capability the project does not have today.**

**`GridSampler` — close to nothing, on two independent grounds.**
First, it cannot enumerate the space (§4.2); it can only run a hand-authored coarse lattice, which is
an assertion about the answer dressed as a search. Second, for a 7-dimensional space where only a
subset of dimensions matter, random sampling is the better-understood choice at equal budget — grid
spends its budget re-testing identical values along dimensions that turn out not to matter. Whatever
grid would tell us here, random tells us more cheaply.

**`CmaEsSampler` — likely nothing at current budgets, and a distortion risk.**
CMA-ES is designed for continuous, ill-conditioned problems with generous evaluation budgets
(hundreds to thousands). Step 1 has: tens of trials, five of seven dimensions integer, one dimension
(`session_idx`) a categorical that CMA-ES would wrongly treat as ordered (§4.3b), and a per-trial cost
measured in GPU-minutes across 26 GPUs. It would consume most of its budget in early generations
without the covariance estimate stabilising. It might matter at a far larger trial budget — which the
GPU cost makes unlikely to ever be authorised.

**The governance divergence is a separate question with more than one answer.**
`agent_manifests/window_optimizer.json:142-149` offers an autonomous governance surface four
selectable strategies, three of which currently fail closed — a `tfm-project-facts` §0.5 dead knob.
Closing that gap is real value. But *implementing three samplers* is only one of the ways to close
it, and choosing among them is Beta's ruling informed by the companion advisor audit
(`CLAUDE_CODE_INSTRUCTIONS_STRATEGY_ADVISOR_AUDIT.md`), not this brief's to make. **This report
recommends removing nothing** (§0.4).

**Honest summary for §2.7:** if the goal is better Step 1 optimisation, the expected gain is close to
zero and the §2.7 #4 dead-dimension interaction (§6.4) means the near-term effect could be mildly
negative. If the goal is a **control arm** — the ability to state, with evidence, that TPE earns its
place — then `RandomSampler` alone delivers most of the available value for the smallest fraction of
the cost.

---

## 9. Verification-integrity controls (VIR-1…6)

- **Execution proof.** Every claim carries a `file:line` anchor read on VM101 this session at HEAD
  `179f7cd`. Optuna behaviour was verified by reading the **installed** package source in
  `~/venvs/torch` (**optuna 4.4.0**) — `optuna/samplers/_grid.py` and `optuna/samplers/_cmaes.py` —
  not from recollection. Live executions performed: `git rev-parse`, `SearchBounds.from_config()`,
  `strategy_contract_gap()` over all four `STRATEGY_CLASSES`, `inspect.signature` on the three
  sampler constructors, `import cmaes`, `CmaEsSampler()` / `RandomSampler()` construction,
  `free -g`, the cartesian-product and memory arithmetic, and an exhaustive check of
  `max(skip_min, min_skip_max)` over `skip_min ∈ 0..10`. Outputs were captured, not summarised.

- **Clean control (VIR-2) — parts of the existing Optuna path verified as working as described:**
  1. `strategy_contract_gap(BayesianOptimization) == ()` while all three siblings return the same
     four-name gap — the gate is live-signature-derived and currently correct.
  2. `OPTIMIZE_FORWARDED_KWARGS` matches the four kwargs actually forwarded at
     `window_optimizer.py:622`.
  3. The S119 static-search-space comment at `window_optimizer_bayesian.py:539` is **accurate** —
     independently confirmed by exhaustive evaluation of the `skip_max` lower bound.
  4. `create_incremental_save_callback` uses only generic Optuna study API — verified sampler-agnostic
     by reading it, not assumed.
  5. TRSE Rule A executes at `:508-517`, strictly before sampler construction at `:543`.

- **Fault-injection (positive) control — n/a, explicitly.** This is read-only scoping; no code was
  changed, so no mutant could be run. Stated rather than omitted, per §4 of the brief. The one place
  where a fault-injection control *would* be required is the new test gate (§6.2), and its absence is
  named there as a cost of the work rather than glossed over.

- **Unavailable-observer behaviour (VIR-5).** The following are reported as **UNAVAILABLE**, not
  inferred:
  - **No sampler was ever executed.** No Optuna study was started (prohibited by the brief). All
    sampler behaviour is read from installed source. Claims about what TPE/Random/CmaEs would
    *produce* on this objective are therefore analytic, not measured — §8 is reasoned, not benchmarked.
  - **`GridSampler` was not constructed** over the live space. The `MemoryError` claim is derived from
    `_grid.py:120,124` plus arithmetic and `free -g`, not from triggering it. It is unambiguous, but
    it is a derivation.
  - **The exact `cmaes` default `popsize`** was not read — the package is not installed. §4.3f states
    the scaling qualitatively and does not assert a number.
  - **Runtime effect of a different sampler on the miner/PWC route** was not executed (no GPU, no
    pipeline). §6.4's "no" is a *call-boundary trace*, not a runtime observation.
  - **The nine `window_opt_*.db` storages were not opened** (consistent with origin audit §9).

- **Audit claim scope (VIR-6).** *Repo-scoped and venv-scoped, on VM101 at `179f7cd`.* The claim is:
  what it would cost, in files and risk, to route `random`/`grid`/`evolutionary` through Optuna
  samplers in Step 1. It does **not** cover what would select a sampler (companion brief), and it does
  **not** rule on whether the work should happen.

- **Searched surfaces:** `window_optimizer.py`, `window_optimizer_bayesian.py`,
  `window_optimizer_integration_final.py`, `distributed_config.json` (`search_bounds`),
  `agent_manifests/window_optimizer.json`, `tests/test_chapter1_p0_corrections.py`,
  `tests/test_s172_phase4_coordinator.py` (`gate22_coexistence` + dispatch),
  installed `optuna/samplers/_grid.py` and `_cmaes.py` in `~/venvs/torch`, `docs/STRATEGY_ORIGIN_AUDIT.md`,
  `docs/CLAUDE_CODE_INSTRUCTIONS_SAMPLER_BEARING.md`, and a repo-wide search for `strategy_map` /
  `require_supported_strategy` / `STRATEGY_CLASSES` consumers.

- **Unavailable surfaces:** live GPU/miner execution; any Optuna study execution; the `cmaes` package;
  the `public` remote's history; non-repo surfaces (host config, systemd, uncommitted deployed files —
  `tfm-project-facts` VIR-6); Team Beta correspondence not in `docs/`.

---

## 10. Coverage table

| Brief §5 requirement | Section | Status |
|---|---|---|
| 1. How the working path builds its sampler — parameter or hard-wired? | §1 | **Answered** — hard-wired, one call site, one `create_study` |
| 2. Minimal change set — files, functions, line anchors | §2 | **Answered** — anchored |
| 3. Does the natural implementation make the signature problem moot? | §3 | **Answered — YES**, with live evidence; also closes origin-audit §6(ii) |
| 4. Per-sampler constraints; is `GridSampler` tractable; what `CmaEs` cannot handle | §4 | **Answered** — Grid **NOT tractable** (7.649 × 10¹⁰ → 7.23 TiB in `__init__`); CmaEs: dependency absent + fails late, `session_idx` ordinality defect |
| 5. Interactions — resume, warm-start, TRSE Rule A, trial history, output shape | §5 | **Answered** — one new provenance hole (sampler not persisted on resume) |
| 6. Blast radius, incl. explicit confirmation on non-Step-1 surfaces | §6 | **Answered** — miner/PWC/ZMQ/kernels/22-array contract **confirmed NOT touched**; §2.7 #4 interaction flagged |
| 7. Effort estimate with reasoning | §7 | **Answered** — small mechanism, medium deliverable; grid **blocked**, not estimated |
| 8. What would be gained, honestly incl. "possibly nothing" | §8 | **Answered** — close to nothing except a random control arm |
| 9. Coverage table + completion sentinel | §10 | **This section** |
| VIR-1…6 controls | §9 | **Complete**, incl. explicit n/a for fault injection |
| Recommends removing no strategy (§0.4) | §0, §4.2, §8 | **Held** |
| Read-only: no code/config/doc changes, no commit, no study | header | **Held** |

---

## 11. Completion sentinel

**TERMINATION: PASS**

The falsifiable question of §1 of the brief — *"To make `random`, `grid` and `evolutionary` run as
genuine Optuna samplers, what exactly changes, in which files, and what does it risk?"* — is answered
with `file:line` anchors obtained this session. No surface required for the answer was inaccessible;
surfaces that were unavailable (§9) are declared and no claim rests on them.

The two questions flagged as potentially decisive both resolved decisively:
**(1) the signature problem is moot** under the natural implementation, and
**(2) `GridSampler` is mathematically unconstructible** over the live search space.

STOP — for Team Alpha review. No code, config or documentation was modified. No commit, no push, no
study, no pipeline. This report authorises nothing.
