# Step 2 Bidirectional Sieve — Descriptive Trace

**Status:** read-only descriptive survey. No files were modified, nothing was executed,
no pipeline was launched.
**Date:** 2026-07-28
**Box:** VM 101 (`zeus-ubuntu`, `192.168.3.177`), tree `/home/michael/distributed_prng_analysis`
**Tree state at time of survey:** `git log -1` = `42a7229`, working tree dirty (see
`git status` in the session; the dirty files are S172 D6 work and are **out of scope** here).

**Scope note (explicit):** this document describes only the legacy/current bidirectional
sieve — the coordinator + `sieve_filter.py` path, the persistent-worker (`PWC`) +
`sieve_gpu_worker.py` path, the kernels in `prng_registry.py`, and the survivor record /
NPZ emission chain. The S172 RANGE-MINER implementation is deliberately **not** analysed,
compared against, or commented on. Where the miner appears in a code path it is noted only
as a branch that exists and is skipped.

**Claim discipline:** every factual claim below carries a `file:line` anchor. Statements that
are inference rather than direct reading are prefixed **[inferred]**.

**No recommendations.** Where behaviour looks anomalous it is described as behaviour and
placed in §10. Nothing here proposes a change.

---

## 1. Naming: what "Step 2" refers to in this tree

The repository uses the phrase inconsistently, and this matters for reading the code.

- `run_complete_pipeline.py:55-81` labels **Step 1** = `window_optimizer.py`
  ("runs sieves internally"), **Step 2** = `sieve_filter.py` ("Forward Sieve"),
  **Step 3** = `reverse_sieve_filter.py` ("Reverse Sieve"). This master runner passes
  flags (`--lottery-file`, `--lottery-data`, `--seed-start`) that `sieve_filter.py`'s
  argparse does not define (`sieve_filter.py:747-749` accepts only `--job-file`,
  `--gpu-id`, `--list-prngs`). **[inferred]** `run_complete_pipeline.py` is therefore not
  the live driver of the sieve.
- `convert_survivors_to_binary.py:85` calls the window optimizer "Step 1", and
  `window_optimizer_integration_final.py:251` calls the NPZ the "Steps 2-6 format" —
  i.e. Step 2 onward *consume* what the sieve produced.
- `preflight_check.py:131-132` treats `bidirectional_survivors.json` +
  `optimal_window_config.json` as the artifacts of stage 2 and
  `bidirectional_survivors_binary.npz` as the input of stage 3.

The **bidirectional sieve itself** — forward pass, reverse pass, intersection, survivor
emission — lives entirely inside the window-optimizer trial loop, in
`window_optimizer_integration_final.run_bidirectional_test()`
(`window_optimizer_integration_final.py:512-1043`). That function, and everything it calls,
is what this document traces.

---

## 2. Control flow: entry point to survivor emission

### 2.1 Top of the chain

```
window_optimizer.py  (CLI / watcher entry)
  └─ coordinator.optimize_window()                    ← monkey-patched in
       window_optimizer_integration_final.py:1046-2026 (add_window_optimizer_to_coordinator)
        └─ WindowOptimizer.optimize()                 window_optimizer.py:465-487
             └─ strategy.search(objective, ...)       window_optimizer.py:485
                  └─ objective(config, optuna_trial)  window_optimizer.py:480-482
                       └─ self.test_configuration(...)  window_optimizer.py:444-463
                            └─ test_config(...)       window_optimizer_integration_final.py:1707-1734
                                 └─ run_bidirectional_test(...)  :1720-1734
```

`optimizer.test_configuration = test_config` at
`window_optimizer_integration_final.py:1736` replaces the method on the instance, so the
`WindowOptimizer.test_configuration` body at `window_optimizer.py:444-463` never runs in
integrated mode.

The Bayesian strategy delegates to `window_optimizer_bayesian.OptunaBayesianSearch`
(`window_optimizer.py:380-383`, `:398-403`), whose per-trial objective is at
`window_optimizer_bayesian.py:437-456`.

### 2.2 Backend selection inside `run_bidirectional_test`

`run_bidirectional_test` is a four-way cascade, in this order:

| Order | Gate | Line | Backend |
|---|---|---|---|
| 1 | `coordinator.use_range_miner` | `:545-546` | RANGE-MINER (**out of scope**) |
| 2 | `coordinator.use_persistent_workers` | `:667-668` | PWC → `sieve_gpu_worker.py` / `sieve_filter.py` |
| 3 | `coordinator.use_zmq_sqlite` | `:712-713` | ZMQ-SQLite coordinator |
| 4 | (fallthrough) | `:762` onward | Legacy: `coordinator.execute_distributed_analysis` → `sieve_filter.py` |

Branches 2 and 3 return early via `_build_test_result_from_pw`
(`:702-703`, `:755-757`). Branch 4 is the original in-line implementation.

### 2.3 Legacy path (branch 4) in detail

`window_optimizer_integration_final.py:765-786` defines a local `Args` shim carrying
`method='residue_sieve'`, `window_size`, `offset`, `skip_min`, `skip_max`,
`threshold`, `max_concurrent=26`, and a `session_filter` string derived from
`config.sessions`. Note `Args` never sets a `hybrid` attribute.

Four sieve passes are possible per trial:

| Pass | `prng_type` set at | Threshold assigned at | Result extracted at |
|---|---|---|---|
| Forward constant | `:797` (`prng_base`) | `Args.threshold = forward_threshold` `:775` | `:810` |
| Reverse constant | `:826` (`prng_base + "_reverse"`) | `:827` | `:840` |
| Forward variable | `:937` (`prng_base + "_hybrid"`) | inherits `:775` | `:948` |
| Reverse variable | `:961` (`prng_base + "_hybrid_reverse"`) | `:959` | `:971` |

Each pass calls `coordinator.execute_distributed_analysis(target_file, results/…json,
args, seeds, 1000, 8, 50)` (`:799-807`, `:830-838`, `:940-946`, `:963-969`). The three
trailing positional literals `1000, 8, 50` are `samples`, `lmax`, `grid_size` — parameters
of the non-sieve correlation path; they are carried through
`execute_truly_parallel_dynamic` but are only consumed in the `else` branch at
`coordinator.py:1450-1461`, not in the sieve branch at `:1435-1447`.

### 2.4 Coordinator dispatch

`coordinator.execute_distributed_analysis` (`coordinator.py:1984-2007`) always sets
`use_parallel_dynamic = True` (`:1997-2001`) and delegates to
`execute_truly_parallel_dynamic` (`:2005-2007`). The static path below `:2009` is therefore
unreachable from this entry point.

Inside `execute_truly_parallel_dynamic`:

1. `self._sieve_config` is rebuilt from `args` at `coordinator.py:1339-1351`.
2. Seed space is chunked at `:1417-1430`: `base_chunk_size = min(total_seeds //
   num_workers, seed_cap)`, chunk records built at `:1435-1447` carrying
   `search_type='residue_sieve'` and `prng_type`.
3. `worker_loop` (`:1468-1639`) pulls chunks, converts each to a `JobSpec`
   (`:1567-1580`), and calls `execute_gpu_job` (`:1598`).
4. `execute_local_job` (`:717`) / `execute_remote_job` (`:914`) build the on-disk job JSON
   at `:732-748` and `:930-946` respectively. Both derive
   `'hybrid': '_hybrid' in job.prng_type` (`:744`, `:942`) and
   `'prng_families': [job.prng_type]` (`:742`, `:940`).
5. The worker command is `python -u sieve_filter.py --job-file <job> --gpu-id 0`
   (`:795-799` local, `:389-391` remote via `_build_sh_safe_cmd`).

Results are collected at `:1670-1682`, retried at `:1684-1784`, and compiled into
`final_results` at `:1811-1832`, where `"results": [r.results for r in successful_results
if r.results]` (`:1825`) is the list the integration layer parses.

### 2.5 GPU work dispatch — `sieve_filter.py`

`execute_sieve_job` (`sieve_filter.py:518-722`):

- loads the draw window via `load_draws_from_daily3` (`:174-188`, called at `:538`);
- iterates `prng_families` (`:551`);
- reads `use_hybrid = job.get('hybrid', False)` (`:560`) and
  `supports_hybrid = family_config.get('variable_skip', False)` (`:564`);
- hybrid + `'_hybrid' in family_name` → **single-phase** `run_hybrid_sieve`
  (`:584-600`);
- hybrid + not `_hybrid` in name → **two-phase** (constant sieve then hybrid refinement),
  `:601-665`;
- otherwise → `run_sieve` (`:669-680`).

`GPUSieve.run_sieve` (`:210-352`) and `GPUSieve.run_hybrid_sieve` (`:353-514`) are the two
kernel launchers. Both compile via `cp.RawKernel(config['kernel_source'],
config['kernel_name'])` (`:207`) with a per-family cache (`:201-209`).

Survivor records are built at `:331-341` (constant) and `:486-502` (hybrid) and returned
under `'survivors'` plus a `'per_family'` map assembled at `:701-710`.

### 2.6 GPU work dispatch — `sieve_gpu_worker.py` (PWC remote path)

`run_sieve_job` (`sieve_gpu_worker.py:139-382`) is the persistent-worker equivalent. It is
a single-chunk executor ("coordinator already sized chunks correctly", `:198`), with a
cached kernel table (`:124-133`) and a cached draw loader (`:100-118`). It emits a flat
parallel-array result (`slim_v1`, `:364-382`), not the record-list shape that
`sieve_filter.py` emits.

The PWC trial driver is `run_trial_persistent`
(`persistent_worker_coordinator.py:1542-…`), which runs the same four passes at
`:1608-1621` (forward constant), `:1648-1662` (reverse constant), `:1698-1711`
(forward variable), `:1725-1738` (reverse variable). Zeus-local chunks still go through
`sieve_filter.py` as a subprocess (`persistent_worker_coordinator.py:1040-1061`).

### 2.7 Result collection and survivor emission

- `extract_survivor_records` (`window_optimizer_integration_final.py:121-160`) walks
  `result['results']`, reading `job_result['survivors']` (`:141-147`) and
  `job_result['per_family'][family]['survivors']` (`:150-158`), dedup by seed keeping the
  **highest** `match_rate`. It returns only `{'seed', 'match_rate'}` — every other survivor
  field emitted by the kernel host code (`best_skip`, `matches`, `total`, `strategy_id`,
  `strategy_name`, `skip_pattern`, `skip_stats`) is dropped here.
- Bidirectional intersection and record construction: `:844-922` (constant),
  `:974-1024` (variable). Detail in §5.
- Threshold-gated incremental NPZ flush: `_flush_npz_incremental`
  (`:243-318`), called at `:424` (PWC/ZMQ adapter) and `:501` (miner adapter). Note it is
  **not** called from the legacy in-line accumulator block.
- Final canonical publication: `utils.run_finalizer.finalize_run` invoked at
  `:1926-1936`, which performs L2 winner selection
  (`utils/run_finalizer.py:690-746`), columnization via
  `utils/canonical_arrays.records_to_arrays` (`utils/canonical_arrays.py:480-540`),
  L3 merge against the certified prior (`utils/run_finalizer.py:752-808`), and global
  seed-ascending sort (`:811-827`).
- `bidirectional_survivors.json` is written at `:1956-1977` as a **post-success summary
  only** — it carries generation metadata, not survivor rows.
- `forward_survivors.json` / `reverse_survivors.json` are written at `:1870-1879` as
  count-only stubs; the full forward/reverse populations are no longer retained
  (`[S166-ACCUM]`, `:907-911`).

---

## 3. The reverse sieve: what "reverse" computes

### 3.1 The kernel

`java_lcg_reverse_sieve` — `prng_registry.py:3115-3169`:

```c
void java_lcg_reverse_sieve(
    unsigned long long* candidate_seeds, unsigned int* residues, unsigned long long* survivors,
    float* match_rates, unsigned char* best_skips, unsigned int* survivor_count,
    int n_candidates, int k, int skip_min, int skip_max, float threshold, int offset
) {
    ...
    const unsigned long long a = 25214903917ULL;   // :3125
    const unsigned long long c = 11ULL;            // :3126
    const unsigned long long m = 0xFFFFFFFFFFFFULL;// :3127
    for (int skip = skip_min; skip <= skip_max; skip++) {
        unsigned long long state = seed & m;
        for (int o = 0; o < offset; o++)  state = (a * state + c) & m;   // :3134-3136
        for (int s = 0; s < skip;   s++)  state = (a * state + c) & m;   // :3138-3140
        int matches = 0;
        for (int i = 0; i < k; i++) {
            state = (a * state + c) & m;                                 // :3143
            unsigned int output = (state >> 16) & 0xFFFFFFFF;            // :3144
            if (((output % 1000) == (residues[i] % 1000)) &&
                ((output %    8) == (residues[i] %    8)) &&
                ((output %  125) == (residues[i] %  125)))  matches++;   // :3146-3150
            for (int s = 0; s < skip; s++) state = (a * state + c) & m;  // :3152-3154
        }
        float rate = ((float)matches) / ((float)k);                      // :3156
        if (rate > best_rate) { best_rate = rate; best_skip_val = skip; }// :3157-3160
    }
    if (best_rate >= threshold) { ... }                                  // :3162-3167
}
```

Compare `java_lcg_flexible_sieve` (the **forward** kernel), `prng_registry.py:958-1004`.
The two bodies are the same algorithm, step for step:

| Aspect | Forward `java_lcg_flexible_sieve` | Reverse `java_lcg_reverse_sieve` |
|---|---|---|
| State recurrence | `state = (a*state + c) & m` (`:975`, `:978`, `:982`, `:988`) | `state = (a*state + c) & m` (`:3135`, `:3139`, `:3143`, `:3153`) |
| Direction of iteration | forward | **forward** |
| `a`, `c` | kernel arguments (`:964`) | hardcoded in the kernel body (`:3125-3126`) |
| Offset pre-advance | yes (`:974-976`) | yes (`:3134-3136`) |
| Skip burn before first draw | yes (`:977-979`) | yes (`:3138-3140`) |
| Output extraction | `(state >> 16) & 0xFFFFFFFF` (`:983`) | identical (`:3144`) |
| Match test | triple modulo 1000 / 8 / 125 (`:984-986`) | identical (`:3146-3148`) |
| Skip search | best over `[skip_min, skip_max]` (`:972`, `:992-995`) | identical (`:3131`, `:3157-3160`) |
| Rate | `matches / k` (`:991`) | `matches / k` (`:3156`) |

A byte-level comparison of the two source strings shows they differ (1775 vs 2075 chars),
but the difference is accounted for by the hardcoded `a`/`c`, the parameter list, the
`candidate_seeds`/`n_candidates` parameter names, and the added comments.

**There is no inverse-LCG step anywhere in the reverse kernel** — no modular inverse of
`a`, no backward recurrence. Reading the file confirms the same for every other
`*_reverse_sieve` in the registry that was inspected: `xoshiro256pp_reverse_sieve`
(`prng_registry.py:1649-1656`) and `sfc64_reverse_sieve` (`:1805-1812`) share the same
signature shape, and their registry descriptions state it outright —
`'Xoshiro256++ Reverse - Fixed skip **forward** validation'` (`:4099`) and
`'SFC64 Reverse - Fixed skip **forward** validation'` (`:4118`).

### 3.2 Where "reverse" actually comes from

The directionality is produced **entirely on the host, by reversing the draw array before
upload**:

- `sieve_filter.py:230-235` (constant path):
  ```python
  # TEMPORAL REVERSAL: Reverse residues for _reverse kernels
  if '_reverse' in prng_family:
      residues_reversed = residues[::-1]
      residues_gpu = cp.array(residues_reversed, dtype=residue_dtype)
  else:
      residues_gpu = cp.array(residues, dtype=residue_dtype)
  ```
- `sieve_filter.py:393-398` (hybrid path) — identical construct.
- `sieve_gpu_worker.py:188-191` — identical construct.

So, in plain terms:

> The **forward** sieve asks: starting from seed *s*, advancing the Java LCG with a constant
> skip *k*, do the successive outputs match the observed draws **in the order they occurred**?
>
> The **reverse** sieve asks exactly the same question about the same forward-iterated
> sequence, but scores it against the **draw list read back to front**. It is a
> *time-reversed target*, not a time-reversed generator.

Consequence, read directly from the code: for a given seed and skip, the forward and reverse
kernels generate the **identical** output sequence `G(s,1..k)`; only the residue vector they
are compared to differs (`d[0..k-1]` vs `d[k-1..0]`).

### 3.3 Comparison with the documentation

**`docs/BIDIRECTIONAL_SIEVE_MATHEMATICAL_WHITEPAPER.md`**, §3-§5:

- §3 forward predicate (`:26-28`):
  `F(s) = (1/n) Σ 1[G(s,i) = d_i] ≥ τ_f` — this matches the forward kernel.
- §4 reverse predicate (`:57-59`):
  `R(s) = (1/n) Σ 1[G(s,−i) = d_{n+1−i}] ≥ τ_r`

  The whitepaper's `G(s,−i)` is the generator evaluated at **negative index** — a backward
  step. The kernel evaluates `G(s, i)` (positive index, forward recurrence,
  `prng_registry.py:3143`) against `d_{n+1−i}` (the reversed residue array,
  `sieve_filter.py:232`). **The generator term diverges; the draw term matches.**
- §4 (`:61-62`) asserts forward and reverse matches are "approximately independent for
  incorrect seeds", and §5 (`:79`) derives `P(B(s)=1) ≈ P(F(s)=1)²` from that independence.
  The code's forward and reverse passes over one seed share the same generated sequence
  (§3.2), so the independence premise is stated about a construction that differs from the
  implemented one. Described here as a doc/code divergence only; no assessment of its
  statistical consequence is offered.
- §6 (`:100-112`) and §7 (`:116-131`) describe threshold regimes; the live default
  thresholds are `0.30` (`distributed_config.json` → `search_bounds.forward_threshold.default`
  and `.reverse_threshold.default`), consistent with §7's "loose thresholds" argument.

**`docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md`** — the file on disk is 128 lines and contains
**only** "§14 Inter-Chunk GPU Cleanup" (`:4-36`) and "§15 Persistent Worker Execution Path
(S146)", with §15 present **twice**, verbatim (`:38-83` and `:85-128`). Sections 1-13 are
absent from the file. What is present is accurate:

- `:22` says inter-chunk cleanup was added "in `sieve_filter.py` (lines 230, 385)". At
  present HEAD the calls are at `sieve_filter.py:326-327` and `:481-482`; the guard text
  `if chunk_start + chunk_size < seed_end:` matches.
- `:65-72` describes four pass types and their kernel arg tails. Verified against
  `sieve_gpu_worker.py:217-304`: constant → standard tail; hybrid forward →
  `threshold, a, c` (`:266-268`); hybrid reverse → `threshold, offset` (`:277-278`).
- `:74-75` states the hybrid forward/reverse branches "are implemented as separate elif
  blocks — they must not share kernel_args construction". Verified: `sieve_gpu_worker.py:259`
  and `:270` are the two arms, and the block `continue`s at `:298` before the shared
  launch at `:306`.

Chapter 2 does not describe what "reverse" computes at the algorithm level, so there is
nothing there to agree or disagree with the kernel.

### 3.4 The reverse hybrid kernel

`java_lcg_hybrid_reverse_sieve` — `prng_registry.py:3170-3244`. Also forward-iterating
(`:3192`, `:3204`, `:3207`). Behavioural differences from the forward hybrid:

- Offset pre-advance is applied (`:3191-3193`); the **forward** hybrid kernel has no
  `offset` parameter at all (`prng_registry.py:1007-1012`).
- Skip search per draw is `try_skip` from `0` upward to `skip_tolerance`
  (`:3200`), always starting from 0, with no adaptation of an expected skip.
- On failure it restores `state = state_save` (`:3218`); on a run of misses exceeding
  `max_consecutive_misses` it sets `failed` and abandons the strategy (`:3221-3227`).
- It **returns on the first strategy that clears the threshold** (`:3229-3240`, `return` at
  `:3239`) — it does not search for the best strategy.
- `skip_seq` is a `[2048]` thread-local array (`:3196`, comment `[S170-K512] was [512]`),
  written unconditionally for `i < k` (`:3216`, `:3226`, `:3237`).

The forward hybrid, `java_lcg_hybrid_multi_strategy_sieve`
(`prng_registry.py:1005-1081`), differs:

- no `offset` parameter (`:1012`);
- `expected_skip` initialised to a hardcoded `5` (`:1027`) and adapted after each hit
  (`:1048`);
- search window is `[expected_skip − skip_tolerance, expected_skip + skip_tolerance]`
  (`:1033-1034`), so it can search backwards from the running estimate;
- it scans **all** strategies and keeps the best `match_rate` (`:1061-1067`);
- loops are bounded `draw_idx < k && draw_idx < 2048` (`:1029`).

---

## 4. Forward/reverse intersection

### 4.1 Legacy path

`window_optimizer_integration_final.py:844-849`:

```python
forward_map = {r['seed']: r['match_rate'] for r in forward_records}
reverse_map = {r['seed']: r['match_rate'] for r in reverse_records}

forward_set = set(forward_map.keys())
reverse_set = set(reverse_map.keys())
bidirectional_constant = forward_set & reverse_set
```

Variable-skip equivalent at `:974-978`:

```python
forward_map_hybrid = {r['seed']: r['match_rate'] for r in forward_records_hybrid}
reverse_map_hybrid = {r['seed']: r['match_rate'] for r in reverse_records_hybrid}
forward_set_hybrid = set(forward_map_hybrid.keys())
reverse_set_hybrid = set(reverse_map_hybrid.keys())
bidirectional_variable = forward_set_hybrid & reverse_set_hybrid
```

A bidirectional survivor is therefore a **plain set intersection of seed IDs** — a seed that
independently cleared its own direction's threshold in both passes. There is no joint
score threshold, no combined-rate gate, and no re-verification of the pair.

Record emission, constant mode, `:913-922`:

```python
for seed in bidirectional_constant:
    fwd_rate = forward_map[seed]
    rev_rate = reverse_map[seed]
    accumulator['bidirectional'].append({
        'seed': seed,
        'forward_match_rate': fwd_rate,             # v3.0: per-seed
        'reverse_match_rate': rev_rate,             # v3.0: per-seed
        'score': (fwd_rate + rev_rate) / 2.0,       # v3.0: per-seed avg
        **metadata_base
    })
```

Variable mode is the same shape at `:1015-1024` with `metadata_base_hybrid`.

Iteration is over a `set`, so the legacy path's within-trial row order is
non-deterministic; the final artifact order is imposed later by
`utils/run_finalizer._sort_by_seed` (`utils/run_finalizer.py:811-827`).

### 4.2 PWC / ZMQ path

`persistent_worker_coordinator.py:1668`:

```python
bidirectional_constant = set(fwd_map.keys()) & set(rev_map.keys())
```

and `:1744`:

```python
bidirectional_variable = set(fwd_h_map.keys()) & set(rev_h_map.keys())
```

The four maps plus the two sets are returned under the versioned
`step1_trial_populations_v2` contract (`:1778-…`), validated at producer egress and again at
adapter ingress (`utils/canonical_records.py:389-440`, called from
`window_optimizer_integration_final.py:363`). Record construction moves to
`utils/canonical_records.build_mode_records` (`:191-250`), which recomputes the same
intersection at `:210-211` and emits records in **ascending seed order** (`:238`).

### 4.3 Constant vs hybrid ("variable skip") — what actually differs

| | Constant (`java_lcg` / `java_lcg_reverse`) | Variable / hybrid (`java_lcg_hybrid` / `java_lcg_hybrid_reverse`) |
|---|---|---|
| Skip model | one fixed skip per seed, best over `[skip_min, skip_max]` (`prng_registry.py:972`, `:3131`) | per-draw skip chosen greedily inside a strategy's tolerance (`:1035`, `:3200`) |
| Strategies | none | list of `{max_consecutive_misses, skip_tolerance}` |
| Early abort | none — always scans all `k` draws | yes, on consecutive misses (`:1057`, `:3223-3225`) |
| Extra outputs | `best_skips[]` (uint8) | `skip_sequences[]` (n_seeds × k uint32) + `strategy_ids[]` |
| Kernel buffers | `sieve_filter.py:250-254`, `sieve_gpu_worker.py:199-203` | `sieve_filter.py:423-428`, `sieve_gpu_worker.py:253-254` |
| Threshold used | trial forward/reverse threshold | `phase2_threshold` (see §9.3) |
| Gate before reverse pass | none | forward-hybrid-zero skips hybrid reverse: `window_optimizer_integration_final.py:953-955`; PWC `persistent_worker_coordinator.py:1719-1723` |
| Record `skip_mode` | `'constant'` (`:891`) | `'variable'` (`:995`) |
| Record `prng_type` | `prng_base` (`:892`) | `prng_base + '_hybrid'` (`:996`) |
| Strategy set used | n/a | legacy: all strategies from `hybrid_strategy.get_all_strategies()` (`sieve_filter.py:571-572`); PWC: **only** `balanced_hybrid` (`persistent_worker_coordinator.py:1690-1696`, `[S147 Q2]`) |

Both modes produce records with the identical 24-key shape. A seed that survives in **both**
modes yields **two** records (documented as intentional at
`window_optimizer_integration_final.py:413-415`), which then compete at L2
(`utils/run_finalizer.py:690-711`).

Neither `strategy_id` nor `skip_sequence` nor `best_skip` survives into the record: they are
dropped at `extract_survivor_records` (`window_optimizer_integration_final.py:147`,
`:158`), which keeps only `{'seed','match_rate'}`.

---

## 5. The 22 NPZ contract fields — provenance, field by field

Contract: `convert_survivors_to_binary._EMPTY_NPZ_DTYPES`
(`convert_survivors_to_binary.py:50-73`), mirrored in
`utils/canonical_arrays.CANONICAL_ARRAY_CONTRACT` (`utils/canonical_arrays.py:98-123`)
and in the `savez_compressed` call order (`convert_survivors_to_binary.py:201-225`).

The live production writer is the finalizer path
(`window_optimizer_integration_final.py:1926-1936` → `utils/run_finalizer.finalize_run`
→ `utils/canonical_arrays.records_to_arrays`). `convert_survivors_to_binary.py` remains a
standalone converter and is no longer called from `optimize_window` — the comment at
`window_optimizer_integration_final.py:1854-1856` records that the fallback subprocess call
was removed. Both writers agree on the 22 names, order and dtypes.

Legend for the **Origin** column:

- **PER-SEED** — a value that differs from seed to seed within one trial.
- **TRIAL-AGG** — a trial-level scalar stamped identically onto every record of that
  trial+mode.
- **CONFIG** — copied from the trial's `WindowConfig` / run parameters, constant per trial.
- **CATEGORICAL** — a mode/identity label encoded to uint8.

| # | Array | dtype | Origin | Computed at | Notes |
|---|---|---|---|---|---|
| 1 | `seeds` | uint32 | **PER-SEED** | legacy `:916` `'seed': seed`; canonical `utils/canonical_records.py:241` | The seed integer from the intersection set. Note the record's seed is emitted from a `uint64` kernel buffer for java_lcg (`prng_registry.py:961`) but the NPZ column is uint32; `utils/canonical_arrays.py:204` range-checks against uint32 and raises rather than wrapping. |
| 2 | `forward_matches` | float32 | **PER-SEED** | see §6 | Renamed from record field `forward_match_rate` (`utils/canonical_arrays.py:156-160`). |
| 3 | `reverse_matches` | float32 | **PER-SEED** | see §6 | Renamed from `reverse_match_rate`. |
| 4 | `window_size` | int32 | **CONFIG** | `:883` / `:987`; canonical `utils/canonical_records.py:216` | `config.window_size`. |
| 5 | `offset` | int32 | **CONFIG** | `:884` / `:988`; `utils/canonical_records.py:217` | `config.offset`. |
| 6 | `trial_number` | int32 | **CONFIG** | `:889` / `:993`; `utils/canonical_records.py:222` | Trial counter from `trial_counter['count']` (`window_optimizer_integration_final.py:1712`), **not** the Optuna trial number. |
| 7 | `skip_min` | int32 | **CONFIG** | `:885` / `:989` | `config.skip_min`. |
| 8 | `skip_max` | int32 | **CONFIG** | `:886` / `:990` | `config.skip_max`. |
| 9 | `skip_range` | int32 | **CONFIG (derived)** | `:887` / `:991`: `config.skip_max - config.skip_min`; canonical `utils/canonical_records.py:220` | A width, not a pair. `convert_survivors_to_binary._parse_skip_range` (`:139-154`) additionally tolerates list and `"min-max"` string forms for historical JSON. |
| 10 | `forward_count` | float32 | **TRIAL-AGG** | `:894` `len(forward_records)`; `:998` hybrid; canonical `utils/canonical_records.py:226` `len(fwd_map)` | Count of forward survivors for that trial+mode. Identical for every row of the trial+mode. |
| 11 | `reverse_count` | float32 | **TRIAL-AGG** | `:895` / `:999`; `utils/canonical_records.py:227` | Same, reverse pass. |
| 12 | `bidirectional_count` | float32 | **TRIAL-AGG** | `:896` / `:1000`; `utils/canonical_records.py:228` | `len(bidirectional_<mode>)`. |
| 13 | `intersection_count` | float32 | **TRIAL-AGG** | `:898` / `:1002`; `utils/canonical_records.py:229` | **Exact duplicate of `bidirectional_count`** — both are `len(both)`. The duplication is explicitly preserved as intentional at `utils/canonical_records.py:203-204`. |
| 14 | `intersection_ratio` | float32 | **TRIAL-AGG** | `:899` / `:1003`; `utils/canonical_records.py:230` | `len(both) / max(len(fwd ∪ rev), 1)` — a Jaccard index. |
| 15 | `intersection_weight` | float32 | **TRIAL-AGG** | `:904` / `:1008`; `utils/canonical_records.py:235` | `len(both) / max(len(fwd) + len(rev), 1)`. |
| 16 | `bidirectional_selectivity` | float32 | **TRIAL-AGG** | `:903` / `:1007`; `utils/canonical_records.py:234` | `len(fwd) / max(len(rev), 1)` — a forward:reverse population ratio; contains no intersection term. May exceed 1.0 legitimately (`utils/canonical_arrays.py:227-236`). |
| 17 | `forward_only_count` | float32 | **TRIAL-AGG** | `:900` / `:1004`; `utils/canonical_records.py:231` | `len(fwd − rev)`. |
| 18 | `reverse_only_count` | float32 | **TRIAL-AGG** | `:901` / `:1005`; `utils/canonical_records.py:232` | `len(rev − fwd)`. |
| 19 | `survivor_overlap_ratio` | float32 | **TRIAL-AGG** | `:902` / `:1006`; `utils/canonical_records.py:233` | `len(both) / max(len(fwd), 1)`. |
| 20 | `score` | float32 | **PER-SEED (derived)** | `:920` / `:1022`; `utils/canonical_records.py:244` | `(forward_match_rate + reverse_match_rate) / 2.0`. This is the only quantity L2/L3 compare on (`utils/run_finalizer.py:708`, `:794`). |
| 21 | `skip_mode` | uint8 | **CATEGORICAL** | `:891` / `:995`; encoded `utils/prng_encoding.py:95-103` | `{'constant': 0, 'variable': 1}` (`utils/prng_encoding.py:37`). |
| 22 | `prng_type` | uint8 | **CATEGORICAL** | `:892` / `:996`; encoded `utils/prng_encoding.py:54-73` | Registry-derived, alphabetical over all 44 `KERNEL_REGISTRY` keys (`utils/prng_encoding.py:42-43`). Live values: `java_lcg` → 0, `java_lcg_hybrid` → 1 (verified by import). |

### 5.1 Tally

- **Genuinely per-seed (3):** `seeds`, `forward_matches`, `reverse_matches`.
- **Per-seed but derived from the two above (1):** `score`.
- **Trial-level aggregates, constant across every row of a trial+mode (10):**
  `forward_count`, `reverse_count`, `bidirectional_count`, `intersection_count`,
  `intersection_ratio`, `intersection_weight`, `bidirectional_selectivity`,
  `forward_only_count`, `reverse_only_count`, `survivor_overlap_ratio`.
- **Run/config constants (6):** `window_size`, `offset`, `trial_number`, `skip_min`,
  `skip_max`, `skip_range`.
- **Categorical labels (2):** `skip_mode`, `prng_type`.

So of 22 columns, **4 carry per-seed information** and 3 of those are functionally
2 independent numbers (`forward_matches`, `reverse_matches`) plus their mean.

### 5.2 Constants and defaults in the writers

- `convert_survivors_to_binary.py` uses `.get(field, 0.0)` / `.get(field, 0)` for **every**
  metadata field (`:134-171`). A survivor JSON missing any of them yields a zero column, not
  an error.
- `convert_survivors_to_binary.py:179` defaults `skip_mode` to `'constant'` and `:184`
  resolves `prng_type` → `prng_base` → literal `'java_lcg'`.
- The canonical path takes **no** defaults: `_check_key_set`
  (`utils/canonical_arrays.py:254-280`) rejects both missing and extra keys, and Ruling G
  (`:36-40`) forbids any directional fallback or 0.0 default for the two rate fields.
- The empty-NPZ artifact writes all 22 arrays at length 0 with their frozen dtypes
  (`convert_survivors_to_binary.py:104-107`).
- `_flush_npz_incremental` (`window_optimizer_integration_final.py:296-305`) writes a
  **4-array** intermediate NPZ (`seeds`, `forward_match_rate`, `reverse_match_rate`,
  `score`) to `bidirectional_survivors_binary.npz` — not the 22-array contract. It is
  overwritten by the finalizer's certified generation at end of run
  (`:1948-1949`). A file matching this 4-array shape is present in the tree
  (`bidirectional_survivors_all.npz.flush.tmp.npz`).

---

## 6. `forward_matches` / `reverse_matches` at the source

### 6.1 The kernel computation

Constant-skip forward (`prng_registry.py:980-995`), and identically the constant-skip
reverse (`:3141-3160`):

```c
        int matches = 0;
        for (int i = 0; i < k; i++) {
            state = (a * state + c) & m;
            unsigned int output = (state >> 16) & 0xFFFFFFFF;
            if (((output % 1000) == (unsigned int)(residues[i] % 1000)) &&
                ((output %    8) == (unsigned int)(residues[i] %    8)) &&
                ((output %  125) == (unsigned int)(residues[i] %  125))) matches++;
            for (int s = 0; s < skip; s++) state = (a * state + c) & m;
        }
        float rate = ((float)matches) / ((float)k);
        if (rate > best_rate) { best_rate = rate; best_skip_val = skip; }
```

and the emission (`prng_registry.py:997-1002`):

```c
    if (best_rate >= threshold) {
        unsigned int pos = atomicAdd(survivor_count, 1);
        survivors[pos]    = seeds[idx];
        match_rates[pos]  = best_rate;
        best_skips[pos]   = (unsigned char)best_skip_val;
    }
```

Hybrid forward (`prng_registry.py:1060-1073`): `float match_rate = (float)matches / k;`
kept as the max over strategies, written to `match_rates[pos]`.
Hybrid reverse (`prng_registry.py:3230-3234`): `float rate = ((float)matches)/((float)k);`
written on the **first** strategy clearing threshold.

### 6.2 Host-side capture

`sieve_filter.py:317-324` (constant):

```python
                survivors = survivors_gpu[:count].get().tolist()
                rates     = match_rates_gpu[:count].get().tolist()
                skips     = best_skips_gpu[:count].get().tolist()
                for i, rate in enumerate(rates):
                    if rate >= min_match_threshold:
                        all_survivors.append(survivors[i])
                        all_match_rates.append(rate)
                        all_best_skips.append(skips[i])
```

`sieve_filter.py:332-341` builds `{'seed', 'family', 'match_rate', 'matches', 'total',
'best_skip'}`, where `matches = int(rate * k)` (`:333`).

`sieve_gpu_worker.py:310-316` is the equivalent for the persistent worker, emitting the
`slim_v1` tuple `(seed, match_rate, None, [best_skip])`.

### 6.3 Into the record

`window_optimizer_integration_final.py:143-147`:

```python
                    seed = survivor.get('seed', survivor.get('id'))
                    if seed is not None:
                        rate = float(survivor.get('match_rate', 0.0))
                        if seed not in records or rate > records[seed]['match_rate']:
                            records[seed] = {'seed': seed, 'match_rate': rate}
```

then `:844-845` builds `forward_map` / `reverse_map`, and `:914-919`:

```python
            fwd_rate = forward_map[seed]
            rev_rate = reverse_map[seed]
            accumulator['bidirectional'].append({
                'seed': seed,
                'forward_match_rate': fwd_rate,             # v3.0: per-seed
                'reverse_match_rate': rev_rate,             # v3.0: per-seed
```

### 6.4 Into the NPZ

Canonical writer — strict rename, no fallback (`utils/canonical_arrays.py:156-160`):

```python
_RENAMED_SOURCE_FIELDS: Dict[str, str] = {
    "seeds":           "seed",
    "forward_matches": "forward_match_rate",
    "reverse_matches": "reverse_match_rate",
}
```

Standalone converter — cross-direction fallback
(`convert_survivors_to_binary.py:123-131`):

```python
    forward_matches = np.array([
        s.get('forward_match_rate', s.get('reverse_match_rate', 0.0))
        for s in survivors
    ], dtype=np.float32)

    reverse_matches = np.array([
        s.get('reverse_match_rate', s.get('forward_match_rate', 0.0))
        for s in survivors
    ], dtype=np.float32)
```

i.e. in the standalone converter a record missing `forward_match_rate` silently receives the
**reverse** rate in the forward column, and vice versa. `utils/canonical_arrays.py:36-40`
records that this same-direction alias tolerance belongs "ONLY to the explicitly historical
legacy seams".

### 6.5 Answer to the question posed

`forward_matches` / `reverse_matches` **are genuine per-seed quantities**, not trial-level
aggregates: each is `matches/k` computed by the GPU thread that owns that seed, maximised
over the skip search (constant mode) or over strategies (hybrid forward) / taken from the
first qualifying strategy (hybrid reverse).

Two properties of that quantity follow directly from the code:

1. **The value space is coarse.** `rate = matches / k` with integer `matches ∈ [0, k]`,
   so at most `k + 1` distinct values exist. `k` is the window size, bounded by
   `distributed_config.json` `search_bounds.window_size` = `[6, 50]` with default 12. With
   a threshold of 0.30 and `k = 12`, only the 9 values `{4/12 … 12/12}` are attainable.
2. **The converter's variance check is calibrated against that.**
   `convert_survivors_to_binary.py:194-195`:
   ```python
       if n > 0 and fwd_unique < max(3, n * 0.10):
           print(f"⚠️  WARNING: Low variance ({fwd_unique} unique values for {n} survivors) - check Step 1 integration version")
   ```
   With `k ≤ 50` the unique count cannot exceed 51, so for any survivor population larger
   than ~510 the warning fires regardless of whether the per-seed rates are correct.

`score` is the arithmetic mean of the two (`:920`), so it takes at most `2k + 1` distinct
values and is likewise per-seed.

---

## 7. Kernel coverage reality

### 7.1 The registry

`KERNEL_REGISTRY` (`prng_registry.py:3729-4132`) has **44 entries** (verified by import):
11 base families × 4 variants (`base`, `base_hybrid`, `base_reverse`,
`base_hybrid_reverse`). The 11 bases, derived by stripping the three suffixes:

`java_lcg, lcg32, minstd, mt19937, pcg32, philox4x32, sfc64, xorshift128, xorshift32,
xoshiro256pp`, plus `xorshift64`.

### 7.2 Hardcoded parameter branches

`sieve_gpu_worker.py:217-304`:

| Branch | Line | Families |
|---|---|---|
| `xorshift32` | `:217` | 1 |
| `pcg32` | `:221` | 1 |
| `lcg32` | `:223` | 1 |
| `java_lcg`, `java_lcg_reverse` | `:227` | 2 |
| `java_lcg_hybrid`, `java_lcg_hybrid_reverse` | `:232` | 2 (own launch + `continue` at `:298`) |
| `minstd` | `:299` | 1 |
| `xorshift128` | `:302` | 1 |

Seven `family_name ==` / `in` branches covering six base families
(`java_lcg`, `lcg32`, `minstd`, `pcg32`, `xorshift128`, `xorshift32`).

`sieve_filter.py:274-309` has the equivalent set, plus a generic
`elif 'hybrid' in prng_family:` catch-all at `:294`.

### 7.3 The five uncovered families

`mt19937`, `philox4x32`, `sfc64`, `xorshift64`, `xoshiro256pp` have no branch in either
dispatcher.

**In this legacy/current path they do not raise `NotImplementedError`.** A grep for
`NotImplementedError` across `sieve_gpu_worker.py`, `sieve_filter.py`,
`prng_registry.py`, `persistent_worker_coordinator.py` and `zmq_sqlite_coordinator.py`
returns nothing. What happens instead: the family falls past every `elif`, reaches
`kernel_args.append(cp.int32(offset))` (`sieve_gpu_worker.py:304`, `sieve_filter.py:311`)
and launches with the base 11 arguments plus `offset`. Whether that matches the target
kernel's signature depends on the kernel — e.g. `xorshift64_flexible_sieve`
(`prng_registry.py:3832`) and `mt19937_full_sieve` (`:3769`) were not read for this survey,
so no claim is made about the outcome. **[inferred]** the launch is at minimum
signature-unchecked; CuPy `RawKernel` does not validate arity against the compiled
signature.

### 7.4 What is actually reachable

TFM sieves `java_lcg` only. `prng_base` reaches the sieve as
`window_optimizer.py --prng-type` → `optimize_window(prng_base=...)` →
`run_bidirectional_test(prng_base=...)` (`window_optimizer_integration_final.py:517`,
default `'java_lcg'`). The four derived identities used per trial are constructed at
`:797`, `:826`, `:937`, `:961`.

So in production exactly **4 of the 44 registry entries** are ever compiled:

| Reachable entry | Registry line | Kernel |
|---|---|---|
| `java_lcg` | `prng_registry.py:3957-3969` | `java_lcg_flexible_sieve` (`:958-1004`) |
| `java_lcg_reverse` | `:3908-3913` | `java_lcg_reverse_sieve` (`:3115-3169`) |
| `java_lcg_hybrid` | `:3970-3985` | `java_lcg_hybrid_multi_strategy_sieve` (`:1005-1081`) |
| `java_lcg_hybrid_reverse` | `:3914-3920` | `java_lcg_hybrid_reverse_sieve` (`:3170-3244`) |

The remaining **40 entries** are never compiled by the Step 2 path, though they are all
still enumerated: they populate `PRNG_TYPE_ENCODING` (`utils/prng_encoding.py:42-43`,
pinned at 44 by `tests/test_prng_encoding.py` per the docstring at `:22-24`), the
`--list-prngs` output (`sieve_filter.py:751-755`), and `BASE_PRNG_FAMILIES`
(`utils/canonical_arrays.py:188-191`). So the registry's size is load-bearing for the uint8
encoding even where the kernels are not executed.

### 7.5 Code that appears unreachable from Step 2

Described, not judged.

1. **`reverse_sieve_filter.py`** (311 lines). Routed only via
   `payload.search_type == 'reverse_sieve'` (`coordinator.py:749-754`, `:802-806`), which
   is produced by `_create_reverse_sieve_jobs` — called only from test scripts
   (`test_xorshift32_hybrid.py:119`, `test_forward_reverse_alignment.py:112`,
   `test_alignment_proper.py:125`, `test_distributed_reverse_sieve.py:82`,
   `test_26gpu_hybrid_full.py:110`, `_test_dist_coord.py:30`). `run_bidirectional_test`
   never sets that search type. Additional observations about the file:
   - its own header (`:7-17`) warns it "should NOT be run directly" and directs the reader
     to `run_bidirectional_test`;
   - it does **not** reverse the residue array — `residues = cp.array(draws, ...)`
     (`:160`), with no `[::-1]`;
   - its kernel call passes 11 arguments in the order
     `(seeds, skips, residues, survivors, rates, used_skips, count, n, k, threshold,
     offset)` (`:169-174`), whereas `java_lcg_reverse_sieve` declares 12 in a different
     order (`prng_registry.py:3117-3121`) — the registry kernels take `skip_min`/`skip_max`
     and take no candidate-skip array;
   - `run_hybrid_reverse_sieve` is a bare `pass` with the comment
     `# Keep as-is — not used` (`:200-202`).
2. **`coordinator.py:2009-2200`** — the "Traditional Static Distribution Mode" body after
   the unconditional early return at `:2005-2007`.
3. **`coordinator._create_sieve_jobs`** (`:2266-2375`) — builds a payload with
   `'hybrid': getattr(args, 'hybrid', False)` (`:2280`, `:2347`) and full strategy loading
   (`:2301-2324`). Reached only from the static path at `:2062`.
4. **`sieve_filter.py:294-309`** — the generic `elif 'hybrid' in prng_family:` inside
   `run_sieve` appends `max_misses, tolerances, n_strategies` to the **constant-kernel**
   argument list (which carries `best_skips`, not `skip_sequences`/`strategy_ids`).
   `execute_sieve_job` routes any `_hybrid` family to `run_hybrid_sieve` instead
   (`:583-600`), so this branch is not entered on the live path.
5. **`sieve_filter.py:601-665`** — the two-phase hybrid path, entered only when
   `use_hybrid` is true **and** `'_hybrid' not in family_name` (`:583`). The coordinator
   derives `hybrid` from `'_hybrid' in prng_type` (`coordinator.py:744`, `:942`), so the two
   conditions are mutually exclusive on this path.
6. **`estimate_background_thresholds`** (`adaptive_thresholds.py:9-93`) is imported by
   `sieve_filter.py:38` and `sieve_gpu_worker.py:69` and is called nowhere in the tree
   (grep for the symbol returns only the definition and the two imports, plus
   `sieve_filter_INTEGRATED.py:27`).
7. **`sieve_filter.py:766-787`** — the `save_forward_sieve_results` block in `main()`.
   Reached only when `sieve_filter.py` runs as `__main__`, which it does on the live path;
   it is wrapped in a bare `except Exception` printing "Note: New results format
   unavailable".
8. **`window_optimizer_integration_final.py:1982-2005`** — `save_bidirectional_sieve_results`
   is called with `forward_survivors=[]`, `reverse_survivors=[]`, `intersection=[]`
   (`:1985-1987`), i.e. three empty lists, inside a `try/except` that prints and continues.

---

## 8. Thresholds and filtering

### 8.1 Where the numbers come from

Live values, `distributed_config.json`:

```
search_bounds.forward_threshold  = {min: 0.30, max: 0.75, default: 0.30}
search_bounds.reverse_threshold  = {min: 0.30, max: 0.75, default: 0.30}
search_bounds.window_size        = {min: 6, max: 50, default: 12}
search_bounds.skip_min           = {min: 0, max: 10}
search_bounds.skip_max           = {min: 10, max: 250}
sieve_defaults.min_match_threshold = 0.01
sieve_defaults.phase1_threshold    = 0.01
sieve_defaults.phase2_threshold    = 0.01
sieve_defaults.window_size         = 512
sieve_defaults.skip_range          = [0, 20]
sieve_defaults.prng_families       = ["mt19937"]
```

`SearchBounds.from_config` reads the `search_bounds` block
(`window_optimizer.py:132-151`). The `sieve_defaults` block is read by the coordinator as a
fallback whenever `args` lacks the attribute (`coordinator.py:1341-1347`, `:738-743`,
`:936-941`) — on the Step 2 path `args` always supplies them, so `sieve_defaults` values
(including `window_size: 512` and `prng_families: ["mt19937"]`) act as unused fallbacks.

### 8.2 The path a threshold takes to the kernel

```
bounds.default_forward_threshold                       window_optimizer.py:128 / from_config :149
  → test_config(ft=…)                                  window_optimizer_integration_final.py:1709
    → run_bidirectional_test(forward_threshold=ft)     :1728
      → Args.threshold = forward_threshold             :775
        → coordinator._sieve_config['min_match_threshold'] = getattr(args,'threshold')
                                                       coordinator.py:1343
          → job['min_match_threshold']                 coordinator.py:739 / :937
            → sieve_filter: min_match_threshold        sieve_filter.py:534
              → cp.float32(min_match_threshold)        sieve_filter.py:270
                → kernel `float threshold`             prng_registry.py:963 / :3120
```

The reverse threshold takes the same route via `reverse_args.threshold = reverse_threshold`
(`window_optimizer_integration_final.py:827`) and `:959` for the hybrid reverse.

### 8.3 Three filtering points, in order

1. **In-kernel.** `if (best_rate >= threshold)` (`prng_registry.py:997`, `:3162`); hybrid
   `if (best_match_rate >= threshold)` (`:1069`) and `if (rate >= threshold)` (`:3231`). A
   seed below threshold is never written to the survivor buffer.
2. **Host re-filter, same threshold.** `sieve_filter.py:321` `if rate >= min_match_threshold`;
   `sieve_filter.py:474` `if rates[i] >= min_match_threshold`;
   `sieve_gpu_worker.py:314` `if rate >= threshold`; `sieve_gpu_worker.py:288`
   `if rate >= hybrid_threshold`. This is a redundant second pass over the same predicate.
3. **Intersection.** `forward_set & reverse_set` (§4). No threshold applies here.

There is no fourth filter: the finalizer's L2/L3
(`utils/run_finalizer.py:714-746`, `:752-808`) *select* among candidates, they do not
reject on score magnitude.

### 8.4 The hybrid threshold is a different number

Legacy path, `sieve_filter.py:587`:

```python
                        phase2_threshold = coerce_threshold(job.get('phase2_threshold', 'auto'), 0.50)
```

`coordinator.py:1349-1350` sets `_sieve_config['phase1_threshold'] = None` and
`['phase2_threshold'] = None`; `coordinator.py:745-746` / `:943-944` copy those into the
job. `job.get('phase2_threshold', 'auto')` therefore returns `None` (the key is present),
and `coerce_threshold(None, 0.50)` returns the default `0.50`
(`adaptive_thresholds.py:106-107`).

PWC path, `persistent_worker_coordinator.py:1118`: `phase2_threshold: float = 0.5` is the
declared default of `run_sieve_pass`, and `run_trial_persistent` never passes the argument
(call sites `:1699-1711`, `:1726-1738`). The value goes into the job at `:1228`, and
`sieve_gpu_worker.py:256-258` resolves:

```python
                phase2_raw = job.get('phase2_threshold', None)
                hybrid_threshold = coerce_threshold(phase2_raw, threshold) if phase2_raw is not None else threshold
```

`phase2_raw` is `0.5`, so `hybrid_threshold = 0.5`.

**Both backends therefore run the variable-skip forward and reverse passes at a fixed
`0.50`**, while the constant-skip passes run at the trial's forward/reverse threshold
(default `0.30`). The `reverse_args_hybrid.threshold = reverse_threshold` assignment at
`window_optimizer_integration_final.py:959` propagates to `min_match_threshold`, which the
single-phase hybrid branch does not consult.

### 8.5 Optuna-suggested thresholds do not reach the sieve

`window_optimizer_bayesian.py:437-456` suggests both thresholds and puts them on the config:

```python
            forward_threshold = trial.suggest_float('forward_threshold',
                                                   bounds.min_forward_threshold,
                                                   bounds.max_forward_threshold)
            ...
            config = WindowConfig(
                ...
                forward_threshold=round(forward_threshold, 2),
                ...
            )
            result = objective_function(config, optuna_trial=trial)
```

`objective` (`window_optimizer.py:480-482`) forwards only `config`, `seed_start`,
`seed_count`, `optuna_trial`. The installed `test_config`
(`window_optimizer_integration_final.py:1707-1734`) declares:

```python
        def test_config(config,
                        ss=seed_start, sc=seed_count,
                        ft=bounds.default_forward_threshold,
                        rt=bounds.default_reverse_threshold,
                        optuna_trial=None):
```

and passes `forward_threshold=ft, reverse_threshold=rt` (`:1728-1729`). Since `ft`/`rt` are
never supplied by the caller, every trial uses `bounds.default_*` (0.30/0.30 live).
`config.forward_threshold` / `config.reverse_threshold` are never read by
`run_bidirectional_test`.

The parallel-partition path does the same explicitly: `_worker_obj` suggests `ft`/`rt`
(`window_optimizer_integration_final.py:1274-1279`) and builds the config with them
(`:1280-1286`), while `_local_test` passes
`forward_threshold=_local_bounds.default_forward_threshold` /
`reverse_threshold=_local_bounds.default_reverse_threshold` (`:1243-1244`).

`WindowOptimizer.test_configuration`'s docstring states the opposite:
"Thresholds are now taken from config.forward_threshold and config.reverse_threshold"
(`window_optimizer.py:450`). That method body is replaced at
`window_optimizer_integration_final.py:1736`.

The suggested values are still recorded: they appear in `config.description()`
(`window_optimizer.py:99-101`), in the Optuna study's params, and in the saved best-config
(`window_optimizer.py:756`, `window_optimizer_bayesian.py:212`).

### 8.6 Pruning and skip gates

- **Forward-zero prune** (constant): `window_optimizer_integration_final.py:815-822` —
  raises `optuna.TrialPruned()` when the forward pass returns 0 survivors, but only if
  `enable_pruning` is set (`[S145-R1]`, `:814`) and Optuna is importable (`:816-817`).
  PWC equivalent: early return with a fully-shaped empty v2 result
  (`persistent_worker_coordinator.py:1627-1646`).
- **Hybrid-forward-zero skip** (`[S147 Q0]`):
  `window_optimizer_integration_final.py:953-955` and
  `persistent_worker_coordinator.py:1719-1723`. This *skips* the hybrid reverse pass, it
  does not prune the trial — constant results are preserved.
- **Objective**: `BidirectionalCountScorer.score` returns
  `float(result.bidirectional_count)` (`window_optimizer.py:285-286`), and
  `bidirectional_count` on the returned `TestResult` is the **combined**
  constant + variable total (`window_optimizer_integration_final.py:1036-1042`,
  `[S124]`), while `forward_count`/`reverse_count` on the same `TestResult` are
  **constant-mode only** (`:1039-1040`). `TestResult`'s own docstring notes this asymmetry
  (`window_optimizer.py:222-226`).

---

## 9. Selection and merge after the sieve

Included because it determines which sieve output actually reaches the artifact.

- **L2** (`utils/run_finalizer.py:690-746`): one winner per seed, key
  `(float32(score), -trial_number, mode_rank)` where `mode_rank = 1` for `'constant'`
  (`:708-711`). Comparison is deliberately in float32 so sub-float32 differences are exact
  ties falling through to the trial tiebreak (`:697-701`). A same-seed/same-trial/same-mode
  pair raises `AccumulatorConsistencyError` (`:736-743`).
- **L3** (`:752-808`): merge against the certified prior generation's arrays. Strictly
  `new_scores > prior_scores` replaces; equal or lower **retains the prior row byte for
  byte** (`:794`). Retained prior rows are copied by index from the prior's typed arrays
  (`:805`), never rebuilt.
- **Final order** (`:811-827`): global seed-ascending, with a strict-increase assertion.
- Publication is deliberately outside every `try/except`
  (`window_optimizer_integration_final.py:1906-1914`): a finalizer rejection propagates out
  of `optimize_window` and the previously certified generation stays current.

---

## 10. Observations

Descriptive only. Each is a statement about what the code does.

**O1 — "Reverse" is a reversed target, not a reversed generator.**
Every `*_reverse_sieve` kernel inspected iterates the PRNG forward; direction is created by
`residues[::-1]` on the host (`sieve_filter.py:232`, `:395`, `sieve_gpu_worker.py:189`).
The whitepaper's §4 predicate uses `G(s,−i)` (`docs/BIDIRECTIONAL_SIEVE_MATHEMATICAL_WHITEPAPER.md:58`).
Two registry descriptions state the implemented behaviour explicitly —
"Fixed skip **forward** validation" (`prng_registry.py:4099`, `:4118`) — while the java_lcg
ones say "fixed skip **backward** validation" (`:3911`, `:3917`) for kernels whose bodies are
forward recurrences.

**O2 — Forward and reverse passes over the same seed generate the same sequence.**
Follows from O1 plus the identical `a`, `c`, `m`, offset pre-advance and skip loop in
`prng_registry.py:958-1004` and `:3115-3169`. The whitepaper's independence assumption
(`:61-62`) and the squared-collapse result (`:79`) are stated about a different construction.

**O3 — 18 of 22 NPZ columns carry no per-seed information.** §5.1. Ten are trial-level
aggregates repeated verbatim on every row of a trial+mode; six are run/config constants; two
are categorical labels.

**O4 — `intersection_count` and `bidirectional_count` are the same number.**
`window_optimizer_integration_final.py:896` and `:898`;
`utils/canonical_records.py:228-229`. The duplication is annotated as deliberate at
`utils/canonical_records.py:203-204`.

**O5 — `bidirectional_selectivity` contains no intersection term.**
`len(fwd) / max(len(rev), 1)` (`:903`, `utils/canonical_records.py:234`) — it is a ratio of
the two population sizes, unbounded above, and `utils/canonical_arrays.py:227-236`
explicitly declines to ceiling it at 1.

**O6 — `forward_matches` has at most `k+1` attainable values, and the converter's variance
warning is scaled to `n`.** §6.5. With `window_size ≤ 50`
(`distributed_config.json` `search_bounds.window_size.max`) and the check
`fwd_unique < max(3, n * 0.10)` (`convert_survivors_to_binary.py:194`), the warning fires
for any survivor population above roughly 510 rows.

**O7 — Optuna's suggested thresholds never reach a kernel.** §8.5. They are suggested
(`window_optimizer_bayesian.py:437-444`), stored on `WindowConfig`, printed in
`description()`, and recorded in the study; the sieve receives
`bounds.default_forward_threshold` / `default_reverse_threshold` every trial
(`window_optimizer_integration_final.py:1709-1710`, `:1243-1244`).

**O8 — `window_optimizer.py:450` docstring contradicts the live behaviour.**
"Thresholds are now taken from config.forward_threshold and config.reverse_threshold" —
the method carrying that docstring is replaced at
`window_optimizer_integration_final.py:1736`.

**O9 — Variable-skip passes run at a hardcoded 0.50.** §8.4. Legacy via
`coerce_threshold(None, 0.50)` (`sieve_filter.py:587`); PWC via the `phase2_threshold: float
= 0.5` default (`persistent_worker_coordinator.py:1118`) resolved at
`sieve_gpu_worker.py:256-258`. The constant-skip passes run at 0.30.

**O10 — The forward hybrid kernel has no `offset` parameter; the reverse hybrid does.**
`prng_registry.py:1007-1012` vs `:3172-3177`. So window offset is applied in the constant
forward, constant reverse and hybrid reverse passes, and not in the hybrid forward pass.

**O11 — `sieve_filter.run_hybrid_sieve` appends a trailing `offset` argument for the forward
hybrid; `sieve_gpu_worker` does not.** `sieve_filter.py:432-462` builds the argument list as
the 13 common arguments, then `[a, c]` for a `java_lcg` family (`:451-452`), then
`cp.int32(offset)` (`:461`) — **16 arguments**.
`java_lcg_hybrid_multi_strategy_sieve` declares **15** parameters, ending at
`float threshold, unsigned long long a, unsigned long long c`
(`prng_registry.py:1008-1012`). `sieve_gpu_worker.py:259-269` builds the same call without
the trailing offset and launches at `:280`, then `continue`s at `:298` so the shared
`kernel_args.append(cp.int32(offset))` at `:304` is not reached. The Zeus-local PWC chunk
path runs `sieve_filter.py` (`persistent_worker_coordinator.py:1051-1055`) while remote
chunks run `sieve_gpu_worker.py`, so the two arms of one PWC pass build different argument
lists.

**O12 — `estimate_background_thresholds` is imported by both sieve entry points and never
called.** `sieve_filter.py:38`, `sieve_gpu_worker.py:69`; definition at
`adaptive_thresholds.py:9-93`. `sieve_gpu_worker.py:70-72` also defines a fallback
`coerce_threshold` for the ImportError case.

**O13 — The forward hybrid kernel hardcodes an initial skip estimate of 5.**
`prng_registry.py:1027` `int expected_skip = 5;`. `skip_min` / `skip_max` are not
parameters of either hybrid kernel, so the trial's configured skip range does not constrain
the variable-skip passes at all.

**O14 — Kernel-local `[2048]` arrays are bounded differently in the two hybrid kernels.**
Forward guards `draw_idx < k && draw_idx < 2048` (`prng_registry.py:1029`) and
`i < k && i < 2048` (`:1064`, `:1075`). Reverse writes `skip_seq[i]` for `i < k` with no
2048 guard (`:3216`, `:3226`) and copies `for (int i = 0; i < k; i++)` at `:3236`.
The comment at `:3196` records the buffer was widened from 512 to 2048 in S170. Given
`search_bounds.window_size.max = 50`, `k` cannot approach 2048 on the live path.

**O15 — `threads.append(thread)` is commented out in the parallel dispatcher.**
`coordinator.py:1648`: `#                 #             threads.append(thread)`, so the list
built at `:1642` stays empty and the join loop at `:1664-1665` iterates zero times.
Completion is instead gated by the poll loop at `:1651-1659` and `work_queue.join()` at
`:1662`; the threads are non-daemon (`:1646`).

**O16 — `run_complete_pipeline.py` passes flags the sieve scripts do not accept.**
`--lottery-data`, `--seed-start`, `--seed-end`, `--threshold` at `:66-81` vs
`sieve_filter.py:747-749` (`--job-file`, `--gpu-id`, `--list-prngs`).

**O17 — `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` is a fragment with a duplicated section.**
128 lines containing only §14 and §15, with §15 present twice verbatim
(`:38-83`, `:85-128`). Its §14 line-number reference ("lines 230, 385", `:22`) no longer
matches the file (`sieve_filter.py:326`, `:481`); the quoted guard text still matches.

**O18 — `reverse_sieve_filter.py` is a second reverse engine that the bidirectional path
never invokes, and its kernel call does not match the registry signature.** §7.5 item 1.

**O19 — `bidirectional_survivors.json` is no longer survivor data.**
`window_optimizer_integration_final.py:1956-1977` writes a generation-summary object. Its
own `note` field says "this file decides no winner and is not independently deduplicated"
(`:1958-1961`). Several consumers still reference it as a survivor file:
`run_complete_pipeline.py:95`, `:104`, `:112`; `generate_full_scoring_jobs.py:24`;
`generate_step3_scoring_jobs.py:15`; `full_scoring_worker.py:513`;
`verify_pruning_s118.py:31`. `forward_survivors.json` and `reverse_survivors.json` are
likewise count-only stubs (`:1870-1879`).

**O20 — `_flush_npz_incremental` writes a 4-array file to the 22-array contract's filename.**
`window_optimizer_integration_final.py:300-305` writes `seeds`, `forward_match_rate`,
`reverse_match_rate`, `score` to `bidirectional_survivors_binary.npz` — note the key names
are the *record* names, not the contract's `forward_matches` / `reverse_matches`. It also
clears `accumulator["bidirectional"]` after each flush (`:310`, `[S166]`). It is not called
from the legacy in-line accumulator block (only from `:424` and `:501`), so the legacy path
accumulates the full candidate list in memory until finalization.

**O21 — The standalone converter's cross-direction fallback.**
`convert_survivors_to_binary.py:123-131` fills a missing `forward_match_rate` from
`reverse_match_rate` and vice versa. The canonical writer forbids this
(`utils/canonical_arrays.py:36-40`).

**O22 — `sieve_defaults` in `distributed_config.json` carries values inconsistent with the
live search bounds.** `window_size: 512` vs `search_bounds.window_size.max: 50`;
`min_match_threshold: 0.01` vs `search_bounds.forward_threshold.min: 0.30`;
`prng_families: ["mt19937"]` vs the java_lcg-only sieve. These are used only as
`getattr(args, …, sieve_defaults…)` fallbacks (`coordinator.py:1341-1347`), and `args`
always supplies them on the Step 2 path.

**O23 — The local PWC result normalizer defaults a missing match rate to 0.5.**
`persistent_worker_coordinator.py:1076`:
`match_rates = [s.get("match_rate", 0.5) for s in raw_surv]`.

**O24 — `coordinator._sieve_config` is built in three places with different hybrid
semantics.** `:1339-1351` (dynamic path) hardcodes `'hybrid': False`,
`phase1_threshold: None`, `phase2_threshold: None`; `:2288-2299` (static path) reads
`getattr(args, 'hybrid', False)` and loads strategies; the per-job builders at `:744` and
`:942` ignore `_sieve_config['hybrid']` entirely and re-derive it from
`'_hybrid' in job.prng_type`. The last one is what the live path uses.

**O25 — `run_bidirectional_test`'s `Args` shim never sets `hybrid`, `phase1_threshold` or
`phase2_threshold`.** `window_optimizer_integration_final.py:765-786`. Every hybrid-related
job field on the legacy path therefore comes from the coordinator's own derivation or from
`None`.

**O26 — Threshold filtering is applied twice with the same predicate.**
In-kernel (`prng_registry.py:997`) and again on the host
(`sieve_filter.py:321`, `sieve_gpu_worker.py:314`).

**O27 — Legacy record emission iterates a `set`.**
`window_optimizer_integration_final.py:913` `for seed in bidirectional_constant:` and
`:1015` for the variable set, so within-trial row order is not reproducible. The canonical
PWC/ZMQ builder sorts (`utils/canonical_records.py:238`), and the finalizer sorts globally
(`utils/run_finalizer.py:817`).

**O28 — Non-canonical fields the kernels compute are discarded before the record.**
`best_skip`, `matches`, `total`, `strategy_id`, `strategy_name`, `skip_pattern`,
`skip_stats` are all built by `sieve_filter.py:334-341` / `:492-502` and dropped by
`extract_survivor_records` (`window_optimizer_integration_final.py:147`, `:158`). No NPZ
column carries any of them.

**O29 — `xoshiro256pp_hybrid_reverse` and `sfc64_hybrid_reverse` name a forward hybrid
kernel symbol.** `prng_registry.py:4105` → `'xoshiro256pp_hybrid_multi_strategy_sieve'`;
`:4124` → `'sfc64_hybrid_multi_strategy_sieve'`. Their `kernel_source` entries do define
those symbol names (`:1726-1729`, `:1867-1870`), and the sources differ from the
corresponding forward-hybrid sources, so the lookup resolves — but the two registry keys
compile kernels whose entry-point names are indistinguishable from the forward variants.
Neither entry is reachable from the Step 2 path (§7.4).

**O30 — Two `# NOTE`/warning comments in `sieve_gpu_worker.py` describe live invariants that
still hold.** `:228-229` ("hybrid variants handled separately below — do NOT add them here")
and `:234-238` (the forward/reverse hybrid signature difference) both match
`prng_registry.py:1012` and `:3177`.

**O31 — S155-era comments in `sieve_gpu_worker.py` describe an ROCR masking scheme the same
file's docstring contradicts.** `:145` says "Workers see all GPUs — no HIP/CUDA/ROCR
visibility masking in spawner" and `:146` "ROCR_VISIBLE_DEVICES not viable on this
CuPy/ROCm stack", while `:164-166`, `:422-423` say "ROCR_VISIBLE_DEVICES={gpu_id} in
spawner remaps assigned GPU to device index 0". Both sets of comments sit above the same
`cp.cuda.Device(0)` call (`:170`, `:424`).

---

## 11. Artifacts written by the Step 2 path

| File | Written at | Content |
|---|---|---|
| `results/window_opt_{forward,reverse}[_hybrid]_{W}_{O}_t{N}.json` | `window_optimizer_integration_final.py:801`, `:832`, `:942`, `:965` (passed as `output_file`) | Per-pass coordinator output path |
| `bidirectional_survivors_binary.npz` | `utils/run_finalizer` (`BINARY_NPZ_NAME`, `utils/run_finalizer.py:130`); intermediate 4-array form at `window_optimizer_integration_final.py:300-305` | Canonical 22-array Steps 2-6 input |
| `bidirectional_survivors_all.npz` | `utils/run_finalizer.py:129`; intermediate at `:296-297` | Accumulator NPZ |
| `bidirectional_survivors.json` | `window_optimizer_integration_final.py:1956-1977` | Post-success generation summary (no rows) |
| `forward_survivors.json`, `reverse_survivors.json` | `:1870-1877` | Count-only stubs |
| `window_optimization_results.json` | `window_optimizer.py` `save_results` via `:1797` | Optuna/search results |
| generation directory + sidecar | `utils/run_finalizer.finalize_run`, reported at `:1938-1949` | Certified generation, sha256-stamped |

---

## 12. Files read for this survey

`window_optimizer_integration_final.py`, `window_optimizer.py`, `window_optimizer_bayesian.py`
(partial), `coordinator.py` (partial), `sieve_filter.py`, `sieve_gpu_worker.py`,
`prng_registry.py` (registry block + java_lcg / xoshiro / sfc64 kernel sources),
`persistent_worker_coordinator.py` (partial), `reverse_sieve_filter.py` (partial),
`convert_survivors_to_binary.py`, `utils/canonical_arrays.py`, `utils/canonical_records.py`
(partial), `utils/run_finalizer.py` (partial), `utils/prng_encoding.py`,
`adaptive_thresholds.py` (partial), `run_complete_pipeline.py` (partial),
`distributed_config.json`, `docs/BIDIRECTIONAL_SIEVE_MATHEMATICAL_WHITEPAPER.md`,
`docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md`.

Not read, and therefore not described: the 40 unreachable kernel sources in
`prng_registry.py`, `zmq_sqlite_coordinator.py` internals, `distributed_worker.py`,
`hybrid_strategy.py`, and the entire `miner/` package (out of scope by instruction).
