# HYBRID SKIP-BOUND AUDIT — do the trial's sampled `skip_min` / `skip_max` reach the hybrid kernels?

**Date:** 2026-07-31 · **Host:** VM 101 `zeus-ubuntu` (`192.168.3.177`) · **Tree:** `/home/michael/distributed_prng_analysis` · **HEAD:** `8a55a68`
**Type:** read-only investigation. No code was changed.
**Falsifiable question:** do the trial's sampled `skip_min` / `skip_max` reach the hybrid (variable-skip) kernels; if not, what would it take to make them — or should they be removed from hybrid optimization entirely?

---

## 0. Verdict

**CONFIRMED.** The `tfm-project-facts` §2.7 claim — *"Hybrid kernels hardcode `expected_skip = 5`; `skip_min`/`skip_max` are not kernel parameters, so the configured skip range doesn't constrain variable-skip passes"* — is correct as written, and all four cited anchors are accurate at HEAD `8a55a68`:

| cited anchor | what is actually there | status |
|---|---|---|
| `prng_registry.py:1027` | `int expected_skip = 5;` in `java_lcg_hybrid_multi_strategy_sieve` | ✅ exact |
| `prng_registry.py:805` | `int expected_skip = 5;` in `mt19937_hybrid_multi_strategy_sieve` | ✅ exact |
| `prng_registry.py:885` | `int expected_skip = 5;` in `xorshift32_hybrid_multi_strategy_sieve` | ✅ exact |
| `prng_registry.py:1159` | `int expected_skip = 5;` in `minstd_hybrid_multi_strategy_sieve` | ✅ exact |

The claim is also **broader than stated**, in two ways worth recording:

1. The split is total and clean, not sampled. Of the **44** kernel entry points in `prng_registry.py`, **all 22 constant-skip kernels declare `int skip_min, int skip_max`** and **all 22 hybrid kernels declare neither** (they declare `unsigned int* skip_sequences` instead). There is no hybrid kernel anywhere in the registry that takes a skip interval. Machine-tabulated from the parsed signatures; full table in §3.2.
2. The `= 5` seed is only the **forward** hybrid's version of the defect. The **reverse** hybrids do not have an `expected_skip` at all — they search a *different* fixed window, `[0, skip_tolerance]`, non-adaptively (`prng_registry.py:3200`). So forward-variable and reverse-variable passes of the same trial do not even use the same skip model as each other. §3.3.

Two supporting facts also confirmed, both of which the skill does not currently record:

- **The defect is live on the certifying route.** It is not confined to the quarantined PWC path. The RANGE-MINER worker reads `skip_range` off the stripe payload (`miner/range_miner_worker.py:776`), carries it into the kernel-arg build context (`:871`), and then `_hybrid_prefix` (`:177-193`) simply never emits it. The values travel the entire distance and are dropped at the final hop.
- **11,870 historical variable-skip rows exist**, across 36 distinct recorded `(skip_min, skip_max)` pairs, none of which constrained the pass that produced them. §5.

---

## 1. VIR declarations

### VIR-6 — audit claim scope

**Claim scope:** the VM 101 working tree at HEAD `8a55a68` (tracked + untracked), plus locally-resident run artifacts on VM 101. **This is a repo-and-local-artifact-scoped audit. It is NOT system-scoped and must not be reported as one.**

**Searched surfaces:**
- Full-tree literal search for `skip_min` / `skip_max` / `skip_range` / `expected_skip` / `skip_tolerance` / `skip_sequence` / `best_skip` using `/bin/grep` (not the `grep` wrapper — that wrapper honours `.gitignore` and would have silently excluded `*.json`; see memory `grep-wrapper-ignores-json`).
- Whole-file reads of the decisive regions: `window_optimizer.py`, `window_optimizer_bayesian.py`, `window_optimizer_integration_final.py`, `sieve_gpu_worker.py`, `sieve_filter.py`, `coordinator.py`, `persistent_worker_coordinator.py`, `persistent/pwc_worker_service.py`, `miner/range_miner_worker.py`, `miner/range_miner_coordinator.py`, `utils/canonical_records.py`, `hybrid_strategy.py`, `prng_registry.py`.
- **Machine-parsed all 44 kernel signatures and bodies** in `prng_registry.py` (regex over `void <name>_sieve(` blocks) rather than eyeballing a sample — §3.2 / §3.3 are generated, not transcribed.
- `optuna_studies/*.db` — 61 SQLite study files opened read-only; 58 carry `skip_min`/`skip_max` trial params.
- `scoring_chunks/*.json` — 99 files, 97,749 rows parsed for `skip_mode` / `skip_min` / `skip_max`.
- All 17 `*.npz` artifacts in the tree (excluding `tmp/`), inspected for `skip_mode` population.
- `distributed_config.json`, `config_manifests/{feature,parameter}_registry.json`.

**Unavailable surfaces (VIR-5: unobservable ≠ clean):**
- **All three rigs are DOWN.** Live ping sweep from VM 101 at audit time: `.120/.121/.122/.154/.155/.156/.162/.163/.164` **all DOWN**; `ssh michael@192.168.3.122` → *No route to host*. **The kernel source actually deployed on the rigs was therefore NOT verified against this tree.** Local reference hashes for a later comparison: `prng_registry.py` md5 `bc15a5ad6f426ddc4c599c44b191f6a5`, `sieve_gpu_worker.py` md5 `4cd7cb8e34a38f7bddab17c7e5a34a14`, `grep -c 'int expected_skip = 5'` = **14**.
- **No kernel was executed.** No GPU run, no PTX/HIP disassembly, no runtime arg-count observation. Every kernel statement here is a source reading.
- **Git history was not mined.** Claims are about HEAD `8a55a68`, not about when each line was introduced.
- The public clone was not fetched; no comparison against it was made.

### VIR-1 — no unsearched absence is reported

Every "X does not contain Y" statement below is backed by a machine-parsed signature table (§3.2/§3.3) or an explicit anchored read. Where I did not search, I say so rather than reporting an absence — specifically: I did not search deployed rig source, kernel binaries, or git history.

---

## 2. Hop-by-hop trace

### 2.1 Producer chain — where the values come from and where they are recorded

| # | file:line | role | carries |
|---|---|---|---|
| P1 | `distributed_config.json` → `search_bounds.skip_min = {min:0, max:10}`, `search_bounds.skip_max = {min:10, max:250}` | bounds source of truth | the two intervals |
| P2 | `window_optimizer.py:41-63` (`load_search_bounds_from_config`), fallback defaults at `:49-50` | loader | same, or safe defaults |
| P3 | `window_optimizer.py:118-121` (`SearchBounds.min_skip_min/max_skip_min/min_skip_max/max_skip_max`), populated `:141-144` | typed bounds | same |
| P4 | `window_optimizer_bayesian.py:429-434` — `trial.suggest_int('skip_min', bounds.min_skip_min, bounds.max_skip_min)`, `trial.suggest_int('skip_max', max(skip_min, bounds.min_skip_max), bounds.max_skip_max)` | **primary Optuna sampler** | the sampled pair |
| P4b | `window_optimizer_integration_final.py:1909-1914` | second, in-file Optuna sampler (same two `suggest_int` calls) | the sampled pair |
| P4c | `window_optimizer.py:200-208` (`random_config`), `window_optimizer_bayesian.py:794-813` (vector decode), `:862-867` (GA mutate) | non-Optuna samplers of the same dimension | the sampled pair |
| P5 | `window_optimizer.py:88-89` — required `WindowConfig` fields, no defaults; hashed `:96`; stamped into `description()` `:101` as `S{skip_min}-{skip_max}` | the resolved config object | the pair |
| R1 | Optuna `trial_params` table, `optuna_studies/*.db` | study record | **58 of 61 studies carry both params** |
| R2 | `window_optimizer_bayesian.py:210-211` → best-config JSON | study record | `best_params.skip_min/skip_max` |
| R3 | `window_optimizer.py:711-712` → `optimal_window_config.json` | promoted config | the pair |
| R4 | `window_optimizer.py:780-781` → TRSE regime context entry | provenance | the pair |

Live sampled values, read from the study DBs: `skip_min` observed 1–10, `skip_max` observed 53–220. These are real sampled values, not defaults.

### 2.2 Consumer chain — Route legacy coordinator (`execute_distributed_analysis`)

| # | file:line | role | outcome |
|---|---|---|---|
| L1 | `window_optimizer_integration_final.py:1395-1396` — `Args.skip_min = config.skip_min`, `Args.skip_max = config.skip_max` | host arg object | carried |
| L2 | `coordinator.py:1344-1345`, `:2292`, `:2451-2452` — `sieve_config['skip_range'] = [args.skip_min, args.skip_max]`; `:1441` — `job_spec['skip_range']` | job dict | carried |
| L3 | `sieve_gpu_worker.py:155` — `skip_range = tuple(job.get('skip_range', [0, 16]))`; `:193` — `skip_min, skip_max = skip_range` | worker unpack | carried |
| L4a | `sieve_gpu_worker.py:210-216` — `cp.int32(skip_min), cp.int32(skip_max)` at `:214` | **CONSTANT** kernel args | ✅ **reaches the kernel** |
| L4b | `sieve_gpu_worker.py:232-279` — the `java_lcg_hybrid` / `java_lcg_hybrid_reverse` branch **reassigns** `kernel_args = [...]` at `:261` (forward) and `:272` (reverse), then launches at `:280` and `continue`s at `:298`, bypassing the shared tail | **HYBRID** kernel args | ❌ **DROPPED — `skip_min`/`skip_max` are absent from both rebuilt lists** |

The drop mechanism is worth naming precisely: the hybrid branch does not *filter out* the skip bounds, it **discards the entire prefix and rebuilds** — so nothing about the constant-path code makes the omission visible at the call site.

### 2.3 Consumer chain — Route PWC (persistent workers)

| # | file:line | role | outcome |
|---|---|---|---|
| W1 | `window_optimizer_integration_final.py:1295-1313` — the `run_trial_persistent(...)` call | trial entry | **`skip_min`/`skip_max` are not among the arguments passed** |
| W2 | `persistent_worker_coordinator.py:1703`, `:1744`, `:1792`, `:1819` — `skip_range = [config.skip_min, config.skip_max] if hasattr(config, 'skip_min') else [0, 147]` | recovered off the config object, with a fallback | carried |
| W3 | `persistent_worker_coordinator.py:1282`, `:1293` — `_skip_range` into the job payload | payload | carried |
| W4 | `persistent/pwc_worker_service.py:544-546` — `"skip_range": job.get("skip_range", [job.get("skip_min",0), job.get("skip_max",147)])` | worker service | carried into the same `sieve_gpu_worker` hops L3/L4 |
| W5 | `persistent_worker_coordinator.py:183-206` (`assert_pwc_hybrid_not_quarantined`) | **fail-closed gate, added in `8a55a68`** | PWC variable-skip **cannot execute**; the quarantine text at `:199-200` already names this exact defect |
| W6 | `persistent/active_job_state.py:276-285`, `:327-328`; `persistent/pwc_worker_service.py:643-645`, `:695-696`, `:735-736` | provenance / config-string surfaces stamping `S{min}-{max}` | carried as **annotation only** |

`8a55a68` documented the finding this audit was asked to verify. `persistent_worker_coordinator.py:160-169` states it as the *reason* for choosing quarantine over threshold propagation: *"the hybrid kernels also ignore the trial's sampled skip_min/skip_max and start from a hardcoded `int expected_skip = 5` (prng_registry.py:1027, :805, :885, :1159) — a second, independent divergence on the same axis, and a kernel-signature change to fix."* **That comment is accurate.**

### 2.4 Consumer chain — Route RANGE-MINER (the certifying route)

| # | file:line | role | outcome |
|---|---|---|---|
| M1 | `window_optimizer_integration_final.py:1255-1256` — `skip_min = config.skip_min, skip_max = config.skip_max` into `run_trial_miner` (the "D0 seam", added specifically because these were the one `WindowConfig` pair still being dropped here) | trial entry | carried |
| M2 | `miner/range_miner_coordinator.py:4175-4176` (params, fail-closed `None` default), `:4280-4281` (serve context) | coordinator | carried |
| M3 | `miner/range_miner_coordinator.py:527-528` (ledger schema), `:1368`/`:1382`/`:1399` (context field lists), `:1414-1415`, `:1438-1439`, `:1480-1481`, `:1547-1548` (manifest); `miner/range_miner_npz_writer.py:188` (`_CONTEXT_FIELDS`, the frozen 11-field canonical trial context) | **manifest identity** | carried — and **frozen**: the run's manifest identity depends on these two fields agreeing across every manifest |
| M4 | `miner/range_miner_worker.py:776` — `skip_min, skip_max = tuple(payload.get("skip_range", [0, 16]))` | worker unpack | carried |
| M5 | `miner/range_miner_worker.py:865-873` — into `BuildContext(skip_min=skip_min, skip_max=skip_max, ...)` at `:871`; fields declared `:147-148` | kernel-arg build context | carried |
| M6a | `miner/range_miner_worker.py:160-174` (`_constant_prefix`) — `ScalarArg(ctx.skip_min,"int32")`, `ScalarArg(ctx.skip_max,"int32")` at `:171-172` | **CONSTANT** ABI | ✅ **reaches the kernel** |
| M6b | `miner/range_miner_worker.py:177-193` (`_hybrid_prefix`, *"Forward/reverse hybrid common 13-element prefix (AUDITED, all families)"*) | **HYBRID** ABI | ❌ **DROPPED — the 13-element prefix contains no skip bound** |
| M6c | `miner/range_miner_worker.py:214-235` (`build_java_lcg`) — hybrid forward returns `_hybrid_prefix + a + c` (15 args), hybrid reverse returns `_hybrid_prefix + offset` (14 args) | per-family ABI | confirms the omission is ABI-faithful, i.e. the builder is correct *given* the kernel signature |

**This is the single most important hop in the audit.** On the certifying route the sampled skip bounds survive every hop — argparse, config, coordinator, ledger, manifest, network payload, worker unpack, kernel-arg build context — and are dropped **one function call before the kernel launch**, by a prefix builder that is faithfully mirroring a kernel signature that has no place to put them.

### 2.5 Consumer chain — Route `sieve_filter.GPUSieve` (in-process)

| # | file:line | role | outcome |
|---|---|---|---|
| F1 | `sieve_filter.py:216`, `:236`, `:269` — `run_sieve(..., skip_range=(0,16))` → `cp.int32(skip_min), cp.int32(skip_max)` | **CONSTANT** | ✅ reaches the kernel |
| F2 | `sieve_filter.py:353-363` — `run_hybrid_sieve(self, prng_family, seed_start, seed_end, residues, strategies, min_match_threshold=0.25, chunk_size=100_000, offset=0)` | **HYBRID** | ❌ **there is no `skip_range` parameter in the signature at all** — the concept does not exist on this API |
| F3 | `sieve_filter.py:432-461` — hybrid kernel-arg list | **HYBRID** | ❌ no skip bound emitted |

F2 is the cleanest statement of the semantic finding: on the one host API where constant and hybrid are sibling methods on the same class, the hybrid method **was never given the parameter to drop**.

### 2.6 Recording chain — where the values are written back onto results

| # | file:line | role | note |
|---|---|---|---|
| B1 | `window_optimizer_integration_final.py:1507-1509` — `metadata_base` for **constant** rows | truthful | describes a range the kernel actually swept |
| B2 | `window_optimizer_integration_final.py:1611-1613` — `metadata_base_hybrid` for **variable** rows: `'skip_min': config.skip_min, 'skip_max': config.skip_max, 'skip_range': config.skip_max - config.skip_min` | ⚠️ **the stamp** | describes a range **no kernel saw** |
| B3 | `utils/canonical_records.py:218-220` (`build_mode_records`), `:351-361` (`build_trial_context`), field list `:115-124` | same stamp, both modes, on the D3.25 path | `skip_min`/`skip_max`/`skip_range` are 3 of the 24 `CANONICAL_RECORD_FIELDS` |
| B4 | `window_optimizer_integration_final.py:1009-1010` — adapter passes `config.skip_min/skip_max` into `normalize_trial_populations` for all four populations | same stamp | |
| B5 | `tests/test_s172_phase5_d3_0_encoding_contract.py:98-100` — `("skip_min","int32"), ("skip_max","int32"), ("skip_range","int32")` | **3 of the 22 frozen NPZ arrays** | frozen contract; cannot be removed |
| B6 | `full_scoring_worker.py:453-456` — `for field in [... 'skip_min','skip_max','skip_range']: features[field] = float(meta[field])` | **ML feature ingestion** | the stamp becomes a model input |
| B7 | `config_manifests/feature_registry.json:329-355` — `skip_analysis.skip_min/skip_max/skip_range` | feature registry | described as *"found during sieve analysis"* |

### 2.7 The information that *is* real for hybrid, and where it dies

| # | file:line | role | outcome |
|---|---|---|---|
| S1 | `hybrid_strategy.py:19-20` — `skip_tolerance: int  # Search window around expected skip (±tolerance)` | **the hybrid's actual skip-range parameter** | |
| S2 | `hybrid_strategy.py:35-73` — five strategies, `skip_tolerance` ∈ {5, 20, 5, 10, 50}, `max_consecutive_misses` ∈ {3, 10, 5, 7, 20} | fixed sweep, not sampled | **`skip_tolerance` is not an Optuna dimension** — no `suggest_*` for it exists in `window_optimizer.py`, `window_optimizer_bayesian.py`, or `window_optimizer_integration_final.py` |
| S3 | `sieve_gpu_worker.py:239-252`; `miner/range_miner_worker.py:725-733`, `:833-841`; `coordinator.py:2307-2320`, `:2353-2354` | strategies → `strategy_tolerances` device array | ✅ reaches the kernel |
| S4 | kernel writes per-draw `skip_sequences[]` and `strategy_ids[]` — `prng_registry.py:1075-1077`, `:3236-3238` | **the actual skips used, per seed per draw** | produced |
| S5 | `sieve_gpu_worker.py:286-290`; `zmq_sqlite_worker.py:140-152`; `miner/range_miner_worker.py` hybrid collect | transported to the host | carried |
| S6 | `window_optimizer_integration_final.py:121-160` (`extract_survivor_records`) — returns `{'seed': seed, 'match_rate': rate}` and nothing else (`:147`, `:158`, `:160`) | **first host consumer** | ❌ **`skip_sequences` and `strategy_ids` are discarded here** |
| S7 | `CANONICAL_RECORD_FIELDS` (`utils/canonical_records.py:115-124`), the 22-array NPZ contract | persistence | ❌ neither `skip_sequences` nor `strategy_ids` nor `skip_tolerance` is a field |

**The system computes the real answer and throws it away, then persists a fabricated one in its place.** The genuinely-descriptive skip information for a variable-skip survivor (its per-draw skip sequence and which tolerance strategy won) is produced by the kernel, carried to the host, and dropped at `extract_survivor_records` — while `skip_min`/`skip_max`, which had no causal role, are persisted into the frozen NPZ contract and fed to the ML layer as features.

---

## 3. What the hybrid kernels actually do with skip

### 3.1 Is `expected_skip = 5` a constant, a default, or derived?

**A hardcoded constant in the CUDA source.** It is a literal initializer inside the per-strategy loop body, re-executed for every strategy of every seed:

```c
// prng_registry.py:1021-1034 — java_lcg_hybrid_multi_strategy_sieve
for (int strat_id = 0; strat_id < n_strategies; strat_id++) {
    int max_misses     = strategy_max_misses[strat_id];
    int skip_tolerance = strategy_tolerances[strat_id];
    unsigned long long state = seed & m;
    int matches = 0;
    int consecutive_misses = 0;
    int expected_skip = 5;                                                   // :1027  ← hardcoded
    ...
    for (int draw_idx = 0; draw_idx < k && draw_idx < 2048; draw_idx++) {
        int actual_skip = expected_skip;
        int search_min = (expected_skip > skip_tolerance) ? (expected_skip - skip_tolerance) : 0;  // :1033
        int search_max = expected_skip + skip_tolerance;                                            // :1034
        for (int test_skip = search_min; test_skip <= search_max; test_skip++) {
            ...
            if (match) { actual_skip = test_skip; expected_skip = test_skip; ... break; }           // :1048 adaptive
        }
    }
}
```

It is not a default (no parameter shadows it), not derived (no input feeds it), and not overridable (no host code can reach it). It is the seed value of an adaptive tracker.

### 3.2 Do the constant kernels take skip bounds? — signature difference

Yes. Machine-parsed over all 44 kernel entry points in `prng_registry.py`:

```
prng_registry.py:960   java_lcg_flexible_sieve(
    unsigned long long* seeds, unsigned int* residues, unsigned long long* survivors,
    float* match_rates, unsigned char* best_skips, unsigned int* survivor_count,
    int n_seeds, int k, int skip_min, int skip_max, float threshold,     ← :963  SKIP BOUNDS
    unsigned long long a, unsigned long long c, int offset)
                                                                          ← :972  for (int skip = skip_min; skip <= skip_max; skip++)

prng_registry.py:1007  java_lcg_hybrid_multi_strategy_sieve(
    unsigned long long* seeds, unsigned int* residues, unsigned long long* survivors,
    float* match_rates, unsigned int* skip_sequences, unsigned int* strategy_ids,   ← skip_sequences replaces best_skips
    unsigned int* survivor_count, int n_seeds, int k,
    int* strategy_max_misses, int* strategy_tolerances, int n_strategies,           ← tolerance replaces bounds
    float threshold, unsigned long long a, unsigned long long c)                    ← NO skip_min / skip_max
```

Across the whole registry, with no exceptions:

| kernel class | count | declares `skip_min`/`skip_max` | declares `skip_sequences` |
|---|---|---|---|
| `*_flexible_sieve` / `*_reverse_sieve` (constant) | 22 | **22 / 22 — yes** | 0 |
| `*_hybrid_multi_strategy_sieve` / `*_hybrid_reverse_sieve` | 22 | **0 / 22 — none** | **22 / 22 — yes** |

The two ABIs are not variants of one shape; they are two different parameterizations of "skip", and the host builders (`_constant_prefix` vs `_hybrid_prefix`, `sieve_gpu_worker` generic vs hybrid branch) mirror that split correctly.

### 3.3 Is there a skip-range concept in the hybrid algorithm at all?

**Yes — but it is a different quantity, and it is already parameterized.** Machine-parsed over all 22 hybrid kernels:

| kernel | line | `int offset` param | `expected_skip = 5` | adaptive (`expected_skip = test_skip`) | `try_skip 0..tolerance` |
|---|---|---|---|---|---|
| `mt19937_hybrid_multi_strategy_sieve` | 774 | Y | Y | Y | n |
| `xorshift32_hybrid_multi_strategy_sieve` | 864 | **n** | Y | Y | n |
| **`java_lcg_hybrid_multi_strategy_sieve`** | **1007** | **n** | **Y** | **Y** | **n** |
| `minstd_hybrid_multi_strategy_sieve` | 1138 | **n** | Y | Y | n |
| `xorshift128_hybrid_multi_strategy_sieve` | 1276 | **n** | Y | Y | n |
| `xoshiro256pp_hybrid_multi_strategy_sieve` | 1418 | Y | Y | n | n |
| `philox4x32_hybrid_multi_strategy_sieve` | 1560 | **n** | Y | n | n |
| `xoshiro256pp_hybrid_multi_strategy_sieve` *(2nd definition)* | 1729 | Y | Y | n | n |
| `sfc64_hybrid_multi_strategy_sieve` | 1869 | Y | Y | n | n |
| `sfc64_hybrid_multi_strategy_sieve` *(2nd definition)* | 2013 | **n** | Y | n | n |
| `pcg32_hybrid_multi_strategy_sieve` | 2095 | Y | Y | Y | n |
| `lcg32_hybrid_multi_strategy_sieve` | 2191 | Y | Y | Y | n |
| `xorshift64_hybrid_multi_strategy_sieve` | 2282 | Y | Y | Y | n |
| `lcg32_hybrid_reverse_sieve` | 2447 | Y | **n** | n | **Y** |
| `xorshift32_hybrid_reverse_sieve` | 2591 | Y | Y | Y | n |
| `xorshift64_hybrid_reverse_sieve` | 2757 | Y | **n** | n | **Y** |
| `xorshift128_hybrid_reverse_sieve` | 2898 | Y | **n** | n | **Y** |
| `pcg32_hybrid_reverse_sieve` | 3040 | Y | **n** | n | **Y** |
| **`java_lcg_hybrid_reverse_sieve`** | **3172** | **Y** | **n** | **n** | **Y** |
| `minstd_hybrid_reverse_sieve` | 3305 | Y | **n** | n | **Y** |
| `philox4x32_hybrid_reverse_sieve` | 3447 | Y | **n** | n | **Y** |
| `mt19937_hybrid_reverse_sieve` | 3648 | Y | **n** | n | **Y** |

So there are **three distinct hybrid skip models** in the registry:

- **(a) adaptive tracker** (9 kernels, incl. `java_lcg_hybrid` forward): per-draw window `[expected_skip − tol, expected_skip + tol]`, seeded at 5, re-centred on every match. **Relative and drifting.**
- **(b) non-adaptive centred window** (5 kernels): same `expected_skip ± tol` window, but `expected_skip` is never updated, so the window is fixed at `[5−tol, 5+tol]` for all draws.
- **(c) absolute low window** (8 reverse hybrids, incl. `java_lcg_hybrid_reverse`): `for (try_skip = 0; try_skip <= skip_tolerance; try_skip++)` — `prng_registry.py:3200`. **No `expected_skip` at all**; the window is `[0, tol]` and never moves.

In every one of the three, the width of the skip window is governed by **`skip_tolerance`**, supplied per strategy from `hybrid_strategy.py:35-73` and reaching the kernel as the `strategy_tolerances` device array. **The hybrid already has a skip-range parameter, it is already wired end-to-end, and it is already swept — over the five fixed strategies, with the winning strategy reported per seed as `strategy_ids`.**

**The TFM-relevant pair is internally inconsistent.** `java_lcg_hybrid` forward is model (a): drifting window seeded at 5. `java_lcg_hybrid_reverse` is model (c): fixed `[0, tol]`. The forward and reverse halves of the same variable-skip trial search structurally different skip spaces.

---

## 4. Is the hybrid algorithm compatible with a skip range? — the decisive question

**Mechanically: a clamp is possible. Semantically: `skip_min`/`skip_max` is not the hybrid's parameter, and making it one would be inventing semantics, not restoring them.**

The reasoning, in order of weight:

**(1) The axis is already occupied.** `skip_tolerance` *is* the hybrid's skip-range parameter — it is exactly "how wide a skip window to search". Adding `skip_min`/`skip_max` does not fill an empty slot; it adds a **second, absolute** bound governing the same window that a **relative** bound already governs. Two knobs on one window, in different coordinate systems, with no specified precedence. Whichever is tighter silently wins, and `strategy_ids` — the field that currently reports which tolerance won — becomes uninterpretable, because the reported strategy would no longer determine the window that was actually searched.

**(2) The two parameters quantify different unknowns.** In the constant kernel, `[skip_min, skip_max]` is the **search space of a single global unknown**: `for (int skip = skip_min; skip <= skip_max; skip++)` (`prng_registry.py:972`) tries each candidate constant skip against *all* draws and keeps the best, reporting the winner in `best_skips`. In the hybrid, skip is **not one unknown but k unknowns** — one per draw — and the algorithm's entire premise is that consecutive skips are *near each other*, which is why the window is relative to the last match and why the output is a *sequence*, not a scalar. An absolute global interval is a well-formed constraint on a constant skip. On a drifting per-draw skip it constrains a different object.

**(3) The historical values prove the mismatch is not hypothetical.** Every one of the 36 distinct `(skip_min, skip_max)` pairs recorded on real variable-skip rows has **`skip_min` ∈ [7, 10]** — that is, `skip_min > 5` in **100%** of cases (§5). The kernel's seed value, `expected_skip = 5`, lies **outside** every range that was ever recorded alongside a variable row. So a faithful clamp cannot just be added: it would have to *also* redefine the seed (to `skip_min`? to the midpoint?), and that choice is unspecified anywhere in the tree. A change that must invent an initial condition to be implementable is a redesign, not a repair — and it would produce a "fix" whose behaviour nobody can check against a prior specification.

**(4) The clamp would bind in the wrong direction.** With `skip_min` ∈ [7,10], clamping `search_min = max(search_min, skip_min)` **forbids small skips** — including skip 0 and skip 1, the most common skips in a dense variable-skip stream, and the *entire* search space of the reverse hybrids (model (c) searches `[0, tol]`, which for `skip_min=8` would clamp to `[8, tol]` and be **empty** for the three strategies with `tol ∈ {5,5}`... i.e. two of the five strategies would search nothing). Meanwhile `skip_max` ∈ [160,235] against `tol ≤ 50` means the upper clamp essentially never binds. The net effect of "honouring the configured range" would be to **delete** most of the hybrid's search space at the bottom while changing nothing at the top.

**Conclusion: the hybrid is not compatible with `skip_min`/`skip_max` in any sense that would make wiring them a correctness repair.** The parameters are meaningless for the hybrid as the hybrid is defined. Per the brief's own decision rule, **the fix is removal from hybrid semantics, not plumbing.**

---

## 5. What has been affected

**Historical variable-skip trials are confirmed suspect, and the value they actually ran at is not recoverable from any persisted artifact.**

Evidence, from `scoring_chunks/*.json` (99 files, 97,749 rows parsed this session):

| | count |
|---|---|
| rows with `skip_mode = 0` (constant) | 85,879 |
| **rows with `skip_mode = 1` (variable)** | **11,870** |
| distinct `(skip_min, skip_max)` pairs on variable rows | **36** |
| variable-row pairs with `skip_min > 5` | **36 / 36 (100%)** |

Most frequent recorded pairs on variable rows: `(8,185)` ×2289, `(8,209)` ×1575, `(9,198)` ×1289, `(8,218)` ×1066, `(8,201)` ×894, `(9,196)` ×685, `(7,170)` ×505, `(8,160)` ×477, `(7,212)` ×454, `(7,191)` ×409.

**Did the recorded `skip_min`/`skip_max` differ from what was run?** Yes, categorically — not by a margin but by kind. Every one of those 11,870 rows records an interval like `[8, 218]`. The kernel that produced them searched, per draw, `[max(0, e−tol), e+tol]` with `e` seeded at 5 and `tol` ∈ {5,20,5,10,50} — i.e. a first-draw union across strategies of `[0, 55]`, drifting thereafter. The recorded lower bound was never respected (5 < 7 in all cases) and the recorded upper bound was never approached (218 vs a reachable 55 at the first draw).

**Can the actual skip value be determined?** **No, not from any persisted artifact.** The per-draw skips *were* computed and returned (`skip_sequences`, `prng_registry.py:1075-1077` / `:3236-3238`), but `extract_survivor_records` (`window_optimizer_integration_final.py:121-160`) reduces every survivor to `{'seed', 'match_rate'}` at `:147`/`:158`, and neither `skip_sequences`, `strategy_ids`, nor `skip_tolerance` appears in `CANONICAL_RECORD_FIELDS` (`utils/canonical_records.py:115-124`) or the 22-array NPZ contract. The most that can be said retrospectively is the bound derived from the source: **every historical variable-skip row ran with a per-draw skip window seeded at 5 and half-width ∈ {5,10,20,50}, regardless of its recorded `skip_min`/`skip_max`.**

**Current exposure:**

- **No live NPZ artifact is contaminated.** All 17 `*.npz` files in the tree were inspected: **189,741 constant rows, 0 variable rows.** The D6 authoritative generation and the Phase 6.0 platform-validation runs are constant-mode only, so this defect does **not** touch the `artifact_sha256` byte-identity result.
- **`--test-both-modes` is opt-in** (`window_optimizer.py:1077`, `action='store_true'`), and the current probe configs set it false (`scripts/probe_phase_A_amd.sh:46`, `scripts/probe_phase_A_rtx.sh:46`).
- **The Optuna objective is nevertheless coupled.** When both modes do run, `_variable_bidi_count` is added into the trial score (`window_optimizer_integration_final.py:1551`, `:1601`, `:1658` — `_total_bidi = len(bidirectional_constant) + _variable_bidi_count`). So the sampler tunes `skip_min`/`skip_max` against an objective that is **part constant (genuinely responsive) and part variable (structurally unresponsive)** — the dimension is not merely dead in the variable half, it injects noise into the credit assignment for the constant half.
- **The stamp is an ML feature.** `full_scoring_worker.py:453-456` promotes `skip_min`/`skip_max`/`skip_range` from record metadata into the model feature dict. On variable rows those three features are a constant-per-trial value with no causal relationship to the row — a spurious trial-identity feature of exactly the kind §2.2 already flags for the 14 `global_*` features.
- **PWC variable-skip is already fail-closed** (`persistent_worker_coordinator.py:183-206`), so no *new* contaminated rows can come from that route.
- **The miner (certifying) route is not gated.** Nothing stops a `--use-range-miner --test-both-modes` run from producing new variable rows carrying the same fabricated stamp.

---

## 6. Blast radius

### 6.1 Option A — wire them in

| surface | file:line | change |
|---|---|---|
| kernel signatures | `prng_registry.py` — **22 hybrid kernels** (9 adaptive, 5 fixed-centred, 8 absolute-low; see §3.3); minimum TFM scope is 2 (`:1007` forward, `:3172` reverse) | add `int skip_min, int skip_max`; **and** decide + implement a new initial condition for `expected_skip` in the 14 kernels that have one, and a new lower bound for the 8 that search `[0, tol]` |
| legacy/PWC worker | `sieve_gpu_worker.py:261-269` (fwd), `:272-279` (rev) | add two args to both rebuilt lists |
| in-process sieve | `sieve_filter.py:353-363` (signature — currently has **no** `skip_range` param), `:432-461` (arg list) | add a parameter and thread it |
| **miner ABI** | `miner/range_miner_worker.py:177-193` (`_hybrid_prefix`) + every per-family builder's hybrid branch (`:214-235` `build_java_lcg`, `:238-262` `build_lcg32`, `:265+` `build_minstd`, …) | prefix grows 13 → 15 elements; **every** per-family hybrid arity shifts (java_lcg fwd 15→17, all reverse hybrids 14→16) |
| ABI test oracle | `tests/test_s172_phase3_worker.py` (documents *"reverse-hybrid (14) + dtype-preserving materialization"* at `:13`) | every hybrid arity assertion |
| certification | Phase 6.0 / D6 | **Yes, this changes the ABI the miner also uses.** The miner's hybrid ABI is derived from the same kernel signatures as the two oracle routes; all three builders must change in lockstep or the four-path comparison fails on arity, not on results. |

Cost: a coordinated 22-kernel + 3-builder + 1-oracle change, landing on the certifying route immediately before Phase 6-P0/6-P1, **whose semantics are unspecified** (§4.3) and whose most likely effect is to delete the bottom of the hybrid search space (§4.4).

### 6.2 Option B — remove from hybrid

**What cannot be removed, and why:**

- **Not from the Optuna search space** (`window_optimizer_bayesian.py:429-434`, `window_optimizer_integration_final.py:1909-1914`) — the constant kernels genuinely consume the pair (`prng_registry.py:963`, `:972`) and constant mode is the certified path.
- **Not from the 22-array NPZ contract** (`tests/test_s172_phase5_d3_0_encoding_contract.py:98-100`) — frozen (skill §2.3, §4).
- **Not from `CANONICAL_RECORD_FIELDS`** (`utils/canonical_records.py:115-124`) — 24-field contract, gate-checked, duplicated on purpose in `utils/canonical_arrays.py`.
- **Not from the miner's 11-field trial context** (`miner/range_miner_npz_writer.py:188`, `miner/range_miner_coordinator.py:1368`) — manifest identity; removing a field changes what "one run agrees on".

So Option B is **not** field deletion. It is a **semantic demotion**, and its surface is:

| surface | file:line | change |
|---|---|---|
| declared meaning | `window_optimizer.py:82-83` — docstring currently reads *"skip_min: Minimum skip value for **variable skip PRNGs**"*, which is the exact inverse of as-built | correct to constant-skip-only |
| variable-row stamp | `window_optimizer_integration_final.py:1611-1613`; `utils/canonical_records.py:218-220` via `:1009-1010` | keep the values (frozen contract) but re-declare them as **trial configuration**, not as a description of the pass; the honest option is to carry an explicit effective-vs-requested provenance record per the D6 pattern |
| feature layer | `full_scoring_worker.py:453-456`; `config_manifests/feature_registry.json:329-355` (description *"found during sieve analysis"* is false for variable rows) | stop treating the triple as a run descriptor on `skip_mode = variable` rows |
| the real parameter | `hybrid_strategy.py:35-73`; no `suggest_*` exists for `skip_tolerance` anywhere | if a tunable hybrid skip dimension is wanted, `skip_tolerance` is its homologue and is currently a fixed 5-point sweep |
| discarded truth | `window_optimizer_integration_final.py:147`, `:158` | `skip_sequences`/`strategy_ids` are dropped here; persisting a summary would make the hybrid's actual skip behaviour auditable for the first time |
| kernels / ABI / certification | — | **none** |

---

## 7. Recommendation

**Remove `skip_min` / `skip_max` from hybrid semantics (Option B). Do not wire them into the hybrid kernels.**

The reasoning that decides it, in one line each:

1. **The hybrid's skip-range parameter already exists, is already wired, and is already swept.** `skip_tolerance` governs the per-draw window in all 22 hybrid kernels and reaches every one of them as `strategy_tolerances`. Wiring `skip_min`/`skip_max` adds a second, absolute bound on a window a relative bound already controls — two coordinate systems, no specified precedence, and `strategy_ids` stops meaning what it means today.
2. **A faithful wire-in is impossible; only a redesign is.** `expected_skip = 5` sits outside every `skip_min` ever recorded on a variable row (36/36 have `skip_min ≥ 7`), so any clamp must also invent a new initial condition — a choice no spec in the tree authorizes. Per the brief's own rule: hybrid semantics do not admit this range, so the fix is removal.
3. **The clamp would damage the search.** With observed `skip_min` ∈ [7,10] and `tol ≤ 50`, the lower clamp deletes the small-skip region that a dense variable-skip stream lives in — and would empty the search entirely for the two `tol=5` strategies on the 8 reverse hybrids that search `[0, tol]`. The upper clamp (160–235 vs a reachable 55) never binds.
4. **Cost asymmetry, at the worst possible moment.** Option A is a 22-kernel ABI change propagating through three independent arg builders and a frozen Phase-3 test oracle, landing on the certifying miner route immediately before Phase 6-P0/6-P1. Option B touches no kernel, no ABI, no frozen contract, and cannot perturb the `artifact_sha256` byte-identity result.
5. **Option B is the D6 fix pattern applied honestly.** Beta's rule is *one canonical path — resolve once in the parent, record requested/payload/effective.* For the hybrid the **effective** skip parameterization is `(seed = 5, half-width = strategy tolerance)`, and it is knowable. The repair is to stop asserting an effective value the kernel never received, and start recording the one it did — not to bend the kernel to match the assertion.

**One consequence Option B does not resolve, flagged for Team Beta rather than assumed:** while `--test-both-modes` is on, the Optuna objective sums constant and variable survivor counts (`window_optimizer_integration_final.py:1658`), so the sampler will keep tuning `skip_min`/`skip_max` against a score that is half-unresponsive to them. Demoting the parameter's *meaning* does not decouple the *objective*. Either the objective is decomposed per mode, or variable-skip trials are excluded from the runs that tune skip bounds. That is a scope decision, not an implementation detail, and it is the item that actually bears on Phase 6's four-path comparison.

---

## 8. Adjacent findings (same code, not the asked question — reported, not acted on)

1. **The forward hybrid kernels also ignore `offset`, which is a sampled Optuna dimension.** `java_lcg_hybrid_multi_strategy_sieve` (`prng_registry.py:1007-1013`) declares no `int offset` and its body performs no offset pre-advance (`:1014-1024`), while its reverse counterpart does (`:3177`, `:3191-3193`). Five forward hybrids lack the parameter entirely (§3.3 table). `offset` is sampled at `window_optimizer_bayesian.py:423` and stamped onto variable rows at `window_optimizer_integration_final.py:1610`. **This is the same defect class on the same kernels, one column over** — and it is not currently recorded in skill §2.7.
2. **`sieve_filter.py:461` appends `cp.int32(offset)` to every hybrid launch**, including forward `java_lcg_hybrid`, whose kernel takes 15 args and no offset — an apparent extra-argument mismatch on that route. The other two builders (`sieve_gpu_worker.py:261-269`, `miner/range_miner_worker.py:214-223`) both correctly omit it. **Not executed this session — flagged for separate verification, not asserted as a live fault.**
3. **Two families have duplicate hybrid kernel definitions** in `prng_registry.py`: `xoshiro256pp_hybrid_multi_strategy_sieve` at `:1418` and `:1729`, `sfc64_hybrid_multi_strategy_sieve` at `:1869` and `:2013`. The pairs differ in whether they declare `int offset` (§3.3), so which one is live depends on which module string is compiled. Both families are among the 5 uncovered (`NotImplementedError`) families for TFM, so there is no current impact — recorded for completeness.
4. **`config_manifests/parameter_registry.json:156-167`** documents `skip_min`/`skip_max` as *"Minimum/Maximum skip value for sieve search"* with a `--skip-min`/`--skip-max` CLI flag, and **`feature_registry.json:329-355`** describes them as *"found during sieve analysis (from Step 2)"* — i.e. as an *observed output*, not a *configured input*. Both descriptions are wrong for the variable-skip case and the feature-registry one is arguably wrong for both.

---

## 9. Verification-integrity controls (VIR-1…6)

- **execution proof:** every table in §3.2/§3.3 and every count in §5 was generated by a script whose output is reproduced above; no signature or count was transcribed by hand. Anchors in §2 were each opened and read.
- **clean control:** the constant-skip path is the built-in negative control — it runs through the *same* files, the *same* job dict key (`skip_range`), the *same* worker unpack (`sieve_gpu_worker.py:193`, `miner/range_miner_worker.py:776`) and the *same* `BuildContext`, and **does** deliver the bounds to the kernel (`sieve_gpu_worker.py:214`, `miner/range_miner_worker.py:171-172`, consumed at `prng_registry.py:972`). The detector therefore distinguishes wired from unwired rather than reporting everything unwired.
- **fault-injection control:** not applicable — this is a read-only source audit with no detector to inject against. **Declared absent, not claimed.**
- **completion sentinel:** this document; all six brief questions answered in §2–§6, verdict in §0.
- **unavailable-observer behaviour:** rig-deployed source is **UNAVAILABLE**, not clean — all nine rig endpoints DOWN at audit time, `ssh` → *No route to host*. Local md5s recorded in §1 for a later comparison. No kernel was executed; no runtime observation is claimed.
- **audit claim scope:** repo + local artifacts on VM 101 at `8a55a68`. **Not system-scoped.**
- **searched surfaces / unavailable surfaces:** enumerated in §1.
- **terminal state (VIR-3):** **PASS** — the falsifiable question is answered with a CONFIRMED verdict and a decided recommendation.
