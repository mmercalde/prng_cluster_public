# CLAUDE CODE REPORT — CPU/GPU GENERATOR CONTRACT, STAGE A (READ-ONLY) — 2026-11-13

**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_CPU_GPU_GENERATOR_CONTRACT_STAGE_A_20261113.md`
**Binding addendum:** `docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md` (§A, §B, §D applied)
**Package:** WP-1 of `docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md` · **Relay thread:** X-0002
**Host:** VM101 `zeus-ubuntu-vm`, user `michael`, venv `~/venvs/torch` · **Tree read:** worktree
`/home/michael/tfm_worktrees/wp1_stage_a`, detached at `0066b6e` (main HEAD at session start; the
brief's base `26d3147` is an ancestor). Every anchor below was read at `0066b6e` in this session.
**What this is:** a statement of what the code does and whether two pieces of code agree. It
judges no route by whether any "true seed" is found, makes no performance claim, and is not a hit rate.

---

## 0. First-step output (verbatim, 2026-11-13T18:53:32-08:00)

```
whoami=michael  hostname=zeus-ubuntu-vm
pgrep -af "range_miner_coordinator|watcher_agent|window_optimizer"
  -> matched only its own shell (pid 11214, the pattern-in-own-command self-match; CLAUDE.md rule 8)
ps -eo pid,user,args | grep -E "[r]ange_miner_coordinator|[w]atcher_agent|[w]indow_optimizer"
  -> no match (rc=1)                         => no coordinator / WATCHER / window optimizer running
ss -ltn | grep 5700                -> no match (rc=1)   => port 5700 unbound
git -C /home/michael/distributed_prng_analysis status --porcelain   -> (empty)
git worktree list
  /home/michael/distributed_prng_analysis  0066b6e [main]
  /home/michael/tfm_worktrees/rm_gates     fbd8264 (detached HEAD)
other process noted: pid 11175 `python3 -u scripts/beta_relay.py --request
  docs/exchange/to_beta/X-FINDINGS-PLAN_R2_ALPHA_REPLY.md` (the Beta relay named by Michael; not touched)
```

Worktree created: `git worktree add --detach /home/michael/tfm_worktrees/wp1_stage_a HEAD` →
`0066b6e (detached HEAD)`. The process check was repeated before the only compute step (Q7):
no match, port 5700 unbound. **Files in play: none.** No branch, tag or commit was created.
**No SQLite file was opened in this session** (addendum §A had nothing to apply to; §D.4).

---

## 1. Summary (null results first)

1. **CPU routes and the constant kernel use the same raw outputs only at skip `s = 0`.** Q7, java_lcg,
   284 seeds (84 saved + 200 synthetic): for `s ∈ {1, 5, 9}` the raw-index overlap with kernel stepping is
   **0 / 1,214,136** (training features), **0 / 303,576** (holdout_hits), **0 / 303,576**
   (holdout_quality) and **0 / 84** (Step 6 vote), on the saved sample; the synthetic sample gives the same zeros.
2. **`holdout_quality`, the training target, does not use kernel stepping even at `s = 0`:** 0 / 303,576
   raw indices identical. It compares holdout draw `j` (combined index `T+j`) with raw output `s_rec + j`.
3. **One argument, `skip`, is used with three meanings:** start position (`holdout_hits`, `skip=T`),
   the survivor's recorded skip used as a start position (`holdout_quality`), and zero (training features, Step 6).
   No CPU route uses an inter-draw gap.
4. **The survivor's own sieve skip never reaches Steps 3 or 6.** Constant `best_skips` and hybrid
   `skip_sequences` reach the worker payload and are discarded at Phase-5 assembly. The value used is the
   trial's lower search bound `skip_min` in Step 3 and `0` in Step 6. This is already recorded in `CHAPTER_6_AUDIT.md:71-78`.
5. **Training features, holdout_hits and the Step-6 vote form one consecutive stream** (raw index =
   combined-record index, `offset = train_history_len`). `holdout_quality` is the only route outside it.
6. **Projection defects:** the CPU batch fallback skips `% mod` (0.0 agreement with the mod route on
   12,800 values). `xorshift64_cpu` keeps the low 32 bits where its kernel compares the full 64-bit state
   (24 / 3,000 agree). `pcg32_hybrid` uses the next state's permutation (5 / 3,000; 3,000 / 3,000 shifted by one).
   `lcg32_hybrid` masks where the CPU reduces mod m (2 / 3,000).
7. **Five forward hybrid kernels declare `offset` and never read it:** `lcg32_hybrid`, `pcg32_hybrid`,
   `xorshift64_hybrid`, `mt19937_hybrid`, `xoshiro256pp_hybrid`. Two of them (`lcg32_hybrid` and
   `pcg32_hybrid`) are in Brief I's `PHASE_CAPABLE_VARIANTS`. This has no effect today because phase is pinned to 0.
8. **Step 6's dual-sieve route turns dict survivors into ints, which forces fresh feature extraction**
   over all N = 18,068 draws, not the T = 14,454 the model was trained on. The forward and reverse files
   the manifest names are now written as summary-only dicts.
9. **Q8 offers five options.** Option C, kernel stepping with the survivor's own skip, needs `miner/` and
   `utils/canonical_records.py` (no-touch) and is **DEFERRED**. Options A, B, B+, C′ and D edit no no-touch file.
10. **All of this is CPU-only.** It is not GPU execution evidence for the torch batch route that Step 3
    runs on the rigs.

---

## Q1 — Caller inventory

Search: `/bin/grep -rn --include=*.py -E "get_cpu_reference|\['cpu_reference'\]|\.get\('cpu_reference'|_cpu_func|_generate_sequence\(|extract_ml_features(_batch)?\(|_cpu_batch_generate|[a-z0-9]+_cpu\(|cpu_reference\b" .`
over the whole worktree (79 files), plus `SurvivorScorer(|batch_score|score_survivor(` and a pool-builder search (Q5).
"Production" = reachable from `STEP_SCRIPTS` (`agents/watcher_agent.py:403-411`) or the Chapter-13 retrain loop.

### Production callers

| # | file:line | generator called | `skip` passed | where the value comes from | what it means at this call |
|---|---|---|---|---|---|
| P1 | `survivor_scorer.py:116`, `:119-125` | `SurvivorScorer._generate_sequence` → `get_cpu_reference(prng_type)` → `*_cpu(seed, n, skip)`, then `% self.mod` | caller's `skip` | each caller below | whatever the caller means; the function treats it as a **start position** (advance `skip`, then `n` consecutive outputs, `prng_registry.py:170-183`) |
| P2 | `survivor_scorer.py:350-355` | `extract_ml_features(seed, lottery_history, …, skip)` → P1 with `n=len(lottery_history)` | caller's `skip`, default 0 | callers P5, P7, P8, P9 | start position |
| P3 | `survivor_scorer.py:547-581` (called `full_scoring_worker.py:375-381`) | `extract_ml_features_batch` → **`java_lcg_pytorch_gpu`** (`prng_registry.py:95-137`) via `get_pytorch_gpu_reference`, `skip=0`, `% mod` inside | `0` literal (`:575`) | hardcoded | **zero**: raw outputs 0..n-1. Step 3 training features, primary route for `java_lcg`/`java_lcg_hybrid` (only these two have `pytorch_gpu`, `prng_registry.py:3961`, `:3974`) |
| P4 | `survivor_scorer.py:577-581` → `:802-808` | `_cpu_batch_generate` → `cpu_func(seed, n, skip=0)`, **no `% mod`** | `0` literal (`:807`) | hardcoded | zero. The fallback when the torch route raises, and the **primary** route for every family without `pytorch_gpu` (`:580-581`) |
| P5 | `full_scoring_worker.py:439-446` | worker sequential fallback (after the batch path raises) → P2 with `train_history`, no `skip` | default `0` | signature default | zero |
| P6 | `full_scoring_worker.py:274-309` (called `:363-369`) | `compute_holdout_hits_batch` → `prng_func(seed, n_holdout, skip=offset)` (`:305`) | `train_history_len` (`:300`, "THIS IS THE LAW") | `len(train_history)` (`:367`) | **start position** = combined-history index of the first holdout draw (`continuation_phase`, `TB_RULING_WINDOW_ANCHOR_SEQUENCING.md:19`) |
| P7 | `full_scoring_worker.py:405-423`; same in the fallback at `:470-484` | `holdout_quality`: `_s111_extract_features_with_optional_skip` (`:68-79`) → P2 with `holdout_history` | `get_survivor_skip(meta)` (`:412`, `:473`) | `holdout_quality.py:26-56`: `skip_best` → `skip_min` → 0. `skip_best` has no producer (Q3), so this is `skip_min` | the **survivor's recorded skip** (in fact the trial's lower search bound), used as a **start position** |
| P8 | `prediction_generator.py:785-790` | Step 6, int survivor → P2 with the full `lottery_history` | `_get_survivor_skip(survivor, default_skip)` → `default_skip` for an int (`:530-553`) | `PredictionConfig.default_skip = 0` (`:109`) | start position (0) |
| P9 | `prediction_generator.py:839-849` | Step 6 vote → P1 `_generate_sequence(seed, next_idx+1, skip=survivor_skip)`, vote = `seq[next_idx]` | `survivor_skip` (`:795`) | `_get_survivor_skip` → `'skip'` / `metadata.skip` / `features.skip_hypothesis` → `default_skip`. No producer for any of the three (Q3), so 0 | start position. Vote = raw `survivor_skip + N` |
| P10 | `prediction_generator.py:864-893` (`extract_features_batch`) | P2 | dict: `_get_survivor_skip`; int: `default_skip` | as P8/P9 | start position. No caller of `extract_features_batch` was found inside `prediction_generator.py` (grep `extract_features_batch` → def only, `:864`); external callers not searched |
| P11 | `survivor_scorer.py:492-516` (`_vectorized_scoring_kernel`, used by `batch_score_vectorized` `:518-534`) | torch route `skip=0`; CPU fallback `cpu_func(…, skip=0)` **without `% mod`** (`:505-512`) | `0` | hardcoded | zero. **Not reached in production now:** Step 2 v4.x removed `SurvivorScorer` (`scorer_trial_worker.py:8`, `:103`). The only remaining mention is that file's docstring (`:36`) |
| P12 | `synthetic_draw_injector.py:227-238` (invoked by `draw_ingestion_daemon.py`, Chapter-13 synthetic mode) | `prng_func(true_seed, n=position+1, skip=0)`, value = `outputs[-1] % mod` | `0` | hardcoded | zero. Synthetic draw `position` = raw output `position` (consecutive stream, gap 0) |

Steps 2, 4 and 5 call no generator: Step 2 (`scorer_trial_worker.py:8`, `:103`); Step 4 (`adaptive_meta_optimizer.py` imports, grep); Step 5 reads the stored target `holdout_quality` (`meta_prediction_optimizer_anti_overfit.py:1479-1483`).
Chapter 13 diagnostics and attribution call none either (`grep _cpu|get_cpu_reference|_generate_sequence|extract_ml_features per_survivor_attribution.py chapter_13_*.py` → no hit).
`reinforcement_engine.py:597` calls P2 with no `skip` (start 0) on its non-dict branch. The search found it reachable from `anti_overfit_trial_worker.py`, `run_anti_overfit_optimizer.sh`, `steps/step2_execution_manager.py` and the `complete_whitepaper_workflow*` / `run_complete_pipeline.py` drivers. None of these is a current `STEP_SCRIPTS` entry, so its production reachability is **[UNVERIFIED]**.

### Excluded, with reason

| file(s) | reason |
|---|---|
| `prng_registry_before_fix_20251012_094113.py`, `prng_registry_before_device_function_20251012_09*.py` (2), `prng_registry_pre_registry.py`, `prng_registry_old.py`, `prng_registry_clean.py`, `prng_registry_before_simple_20251011_110611.py`, `prng_registry_before_mt_fix_20251011_101546.py` | dated or backup copies of the registry; nothing on the pipeline imports them |
| `backups/step6_restoration_20251223_213936/*`, `step6_restoration/*_PATCH.py`, `survivor_scorer_original.py` | backup and patch copies |
| `apply_s109_*`, `apply_s113_*`, `apply_s119_*` | one-shot patch corpus (forensic only, never re-executed) |
| `generate_ml_jobs.py` (`:341`, `:420`, `:778`) | legacy job generator. `:778` calls `score_survivor`, which `survivor_scorer.py` does not define (`grep "def score_survivor"` → no hit) |
| `ml_prng_cracker_distributed.py`, `generate_fingerprint_library*.py`, `enhanced_gpu_model_id.py`, `examine_ml_features.py`, `example_usage.py`, `validate_prediction_system.py`, `run_tests.py`, `test_*.py`, `verify_*.py`, `create_test_*`, `recreate_test_data.py`, `manual_kernel_test.py`, `train_one_kernel.py`, `complete_hybrid_alignment_test.py` | research tools and tests. All pass `skip=0` or are not on the pipeline |
| `build_pools.py` | a separate pool builder, listed under Q5 |

---

## Q2 — Stepping table, all 11 families

Columns: **init** (seed → state) · **offset** (phase pre-advance) · **pre** (advances before the first compared output) · **gap** (advances between compared outputs) · **proj** (raw value compared, before `% 1000`).
The constant kernel is the reference. Its stepping is `state ← init(seed); advance offset; advance skip; per draw: advance once, compare proj(state), advance skip`.
So draw `i` is raw output index **`offset + skip + i·(skip+1)`** (java_lcg `prng_registry.py:971-990`; philox as counters `:1535`, `:1542`).
Kernel offset loops: `:424` xorshift32, `:481` pcg32, `:533` lcg32, `:645` mt19937, `:694` xorshift64, `:974` java_lcg, `:1102` minstd, `:1236` xorshift128, `:1392` xoshiro256pp, `:1535` philox4x32, `:1987` sfc64.
CPU references: `:153-404`. CPU stepping for all 11 is `init; advance skip; then n consecutive outputs` → raw indices `skip … skip+n-1`.

### CPU reference vs constant kernel

| family (CPU ref) | init | offset | pre | gap | proj |
|---|---|---|---|---|---|
| xorshift32 (`xorshift32_cpu` :153) | AGREE `seed & M32` | DIFFER — no parameter | AGREE iff CPU skip = offset+skip | **DIFFER** 0 vs skip | AGREE `state` |
| pcg32 (:233) | AGREE | DIFFER | AGREE iff … | **DIFFER** | AGREE permute(oldstate) |
| lcg32 (:248) | AGREE `seed & M32`, (a·s+c) % m, registry a,c,m (`:3758-3762`) = CPU defaults | DIFFER | AGREE iff … | **DIFFER** | AGREE `state` |
| mt19937 (`mt19937_cpu_simple` :371, registry `:3770`) | AGREE init_genrand | DIFFER | AGREE iff … | **DIFFER** | AGREE tempered. Q2 check: `mt19937_cpu_simple` == `mt19937_cpu` 7000/7000 |
| xorshift64 (:217) | AGREE | DIFFER | AGREE iff … | **DIFFER** | **DIFFER**: CPU `state & 0xFFFFFFFF` (`:231`), kernel full 64-bit `state` (`:713`). Measured 24/3000 |
| java_lcg (:170) | AGREE `seed & (2^48−1)` | DIFFER | AGREE iff … (`:976-978` vs `:974-979`) | **DIFFER** 0 vs skip (`:987-989`) | AGREE `(state>>16) & M32` |
| minstd (:184) | AGREE `seed % m`, 0→1 (kernel m const 2147483647) | DIFFER | AGREE iff … | **DIFFER** | AGREE |
| xorshift128 (:198) | AGREE x=seed, y,z,w consts | DIFFER | AGREE iff … | **DIFFER** | AGREE `w` |
| xoshiro256pp (:267) | AGREE | DIFFER | AGREE iff … | **DIFFER** | AGREE 64-bit result |
| philox4x32 (:289) | AGREE key from seed; counter-based | DIFFER | AGREE: CPU counters start `skip`, kernel `offset+skip` | **DIFFER** CPU +1, kernel +skip+1 | AGREE `ctr[0]` |
| sfc64 (:315) | AGREE | DIFFER | AGREE iff … | **DIFFER** | AGREE `tmp` |

"AGREE iff" means the CPU's single `skip` can absorb `offset + skip` before the first draw. The CPU has no inter-draw gap, so the two agree on a whole window only when kernel `skip = 0`.
The proj column was checked empirically **only for xorshift64**. For the other ten, `q2_projection_check.py` compares the CPU with itself, so its 3000/3000 is tautological; agreement there rests on reading the kernels.
`tests/phase6/known_answer_reference.py` is an independent hand transcription of all four java_lcg variants (`constant_sieve`, `hybrid_forward_sieve`, `hybrid_reverse_sieve`). Its note 4 states this same CPU/kernel mismatch. Q7 cross-checks the java_lcg row against it.

### Forward hybrid kernel vs constant kernel

All 11 hybrids AGREE on **init**. All 11 DIFFER on **pre** and **gap**: these are a per-draw search, not a fixed stride. `expected_skip` starts at a hard 5 (java_lcg `:1027`). The search window is `[max(0, e−tol), e+tol]` (`:1033-1034`). On a hit, the hybrid advances `test_skip` and then one more.

| family | offset | re-centres on hit | advance on a miss | miss budget | proj vs constant |
|---|---|---|---|---|---|
| java_lcg (:1005) | no argument | yes | `search_max` (state left at backup+search_max; KAR `hybrid_forward_sieve` docstring) | `>=` | AGREE |
| xorshift32 (:862) | no argument | yes | `search_max` | `>=` | AGREE |
| minstd (:1136) | no argument | yes | `search_max` | `>=` | AGREE |
| xorshift128 (:1274) | no argument | yes | `search_max` | `>=` | AGREE |
| lcg32 (:2189) | **declared (`:2196`), never read** | yes | `search_max` | `>=` | **DIFFER**: `(state*a+c) & m` in uint32 (`:2227`, `:2230`) vs constant/CPU `(a·s+c) % m`. Measured 2/3000 vs `lcg32_cpu` |
| pcg32 (:2093) | **declared (`:2100`), never read** | yes | **0** (`state` untouched; candidates built from `state_backup`) | `>=` | **DIFFER**: permutes `output_state`, the new state (`:2137-2138`), not oldstate. Measured 5/3000 at the same advance count, 3000/3000 against `pcg32_cpu` one index later |
| xorshift64 (:2280) | **declared, never read** | yes | `search_max` | `>=` | AGREE with its constant kernel (full 64-bit). DIFFER from CPU, as above |
| mt19937 (:743) | **declared (`:779`), never read** | yes | `search_max + 1` (extract is in place) | `>=` | compares `output % 1000 == residues[i]` with no `% 1000` on the residue (`:820-821`). Equal for residues in [0,999] |
| xoshiro256pp (:1416) | **declared (`:1423`), never read** | **no** (stays 5) | 1 (restore, then `next()`, `:1475`) | `>` (`:1472`) | AGREE |
| philox4x32 (:1558) | no argument | **no** | 1 (`counter++`, `:1627`) | `>` | AGREE |
| sfc64 (:2011) | no argument | **no** | 1 | `>` | AGREE |

"Declared, never read": `data/wp1_stage_a/kernel_offset_scan.log` (AST scan of every `*_KERNEL` string, comments stripped). All 11 constant kernels and all 22 reverse kernels read `offset` (one use each). Five forward hybrids declare it and read it zero times.
This bears on Brief I's capability matrix. `PHASE_CAPABLE_VARIANTS` (`miner/range_miner_worker.py:142-153`) and v1.1 §3 list `lcg32_hybrid` and `pcg32_hybrid` as "phase input … delivered". The argument is delivered (AC1 arg-capture, `S172_WINDOW_ANCHOR_BRIEF_I_REPORT.md` §6), but those two kernel bodies never read it.
**Measured fact, no verdict:** with `generator_phase` pinned to 0 in v1, nothing that has run is affected. **Prediction:** if a nonzero phase is ever allowed on these two variants, the device stepping will not change, so the requested phase will not shape their survivors or pools. **Measurement that settles it:** a kernel-level test (not arg-capture) that runs `lcg32_hybrid` with phase 0 and phase k on a planted stream and compares the survivor sets. That needs a GPU and is outside this brief. `miner/` and `prng_registry.py` are no-touch; nothing is proposed.
Reverse hybrids (java_lcg `:3170-3244`) are a different algorithm again: offset is applied, the search is `[0, tol]`, state is restored on a miss, and the first qualifying strategy wins (KAR `hybrid_reverse_sieve`). No CPU route models any hybrid.

---

## Q3 — What the sieve records, and what arrives

| stage | constant mode | variable (hybrid) mode | anchor |
|---|---|---|---|
| kernel output | `best_skips[pos]` (uint8), the lowest skip achieving the best rate | `skip_sequences[pos*k+i]` (k entries), `strategy_ids[pos]` | `prng_registry.py:996-1002`; `:1062-1079` |
| worker payload | `[seed, rate, None, [best_skip]]` | `[seed, rate, strategy_id, skip_sequence]` | `miner/range_miner_worker.py:1272-1292`, `build_substripe_payload_bytes` `:1347-1365` |
| Phase-5 assembly | `[2]` and `[3]` validated, then **discarded**; only `seed`, `rate` enter the directional map | same | `miner/range_miner_npz_writer.py:786-823` (rates only, `:820-823`); `miner/assembly_shard_worker.py:78` ("fully validated … and then DISCARDED") |
| canonical record / NPZ | trial-level `skip_min`, `skip_max`, `skip_range` = the trial's **search bounds** from `WindowConfig`; `skip_mode` | same. The hybrid kernel ignores those bounds (skill §2.7 #4) | `utils/canonical_records.py:226-228`; context `window_optimizer_integration_final.py:1794` |
| Step 3 chunk metadata | NPZ fields incl. `skip_min/max/range` | same | `utils/survivor_loader.py:350`; `full_scoring_worker.py:176-199` |
| **Step 3 `get_survivor_skip`** | `skip_best` → **no producer** → `skip_min` (search lower bound) → 0 | same | `holdout_quality.py:47-56` |
| **Step 6 `_get_survivor_skip`** | `skip` → `metadata.skip` → `features.skip_hypothesis` → **no producer for any** → `default_skip` = 0 | same | `prediction_generator.py:530-553`, `:109` |

Absence searches. `skip_best`: `/bin/grep -rln skip_best .` (whole worktree, all file types) finds `holdout_quality.py` plus four docs only; `git log -S skip_best` finds `aaca35d` (S111) and `955d0fd` (docs); the main tree's `config_manifests/` and `agent_manifests/` JSON have none.
`skip_hypothesis`: `--include=*.py` finds only `prediction_generator.py:549`. Step 3's metadata dict (`full_scoring_worker.py:395-401`) has no `skip`.
Measured on the saved artifact (`survivors_with_scores.json`, 84 records, Mar 11 2026): 84/84 carry `features.skip_min = 5.0` and `skip_max = 56.0`; 0/84 carry `skip`, `metadata.skip` or `features.skip_hypothesis`. Calling the production `_get_survivor_skip` on them returns 0.
**Already governed, not new:** `docs/CHAPTER_6_AUDIT.md:71-78`, `:336-337` (no `skip_best` producer; `skip_min` is per-trial), and skill §2.2/§2.13 (`skip_sequences` discarded on the legacy path). This report adds that the miner path drops them at Phase-5 assembly, and that the constant `best_skip` is dropped the same way.
The certified current generation (`.s172_accumulator/current/bidirectional_survivors_binary.npz`, sha256 `aecc5d50…a43b`) has 22 arrays and **0 rows**, so no current record exists to inspect.

---

## Q4 — Training route: raw output indices

Index space: `daily3.json` (combined sessions, unfiltered) is split 80/20 with no session filter (`window_optimizer.py:1097-1104`, also `:1385-1387`).
Measured: N = 18,068 records, T = 14,454 train, H = 3,614 holdout. "Record `r`" means combined index `r`. Raw output `r` means the `(r+1)`-th generator output after `init(seed)`.

| route | record compared | raw output used | anchor |
|---|---|---|---|
| **training features**, batch route (`java_lcg`) | train `j`, `j ∈ [0,T)` | raw `j` (torch generator, `skip=0`, `% 1000`) | `full_scoring_worker.py:375-381` → `survivor_scorer.py:570-575` |
| training features, batch CPU fallback (`_cpu_batch_generate`) | train `j` | raw `j` **unreduced** (no `% mod`); compared directly with draws 0–999 | `survivor_scorer.py:577-581`, `:802-808` |
| training features, worker sequential fallback | train `j` | raw `j`, `% 1000` | `full_scoring_worker.py:439-446` → `survivor_scorer.py:355` |
| **holdout_hits** (legacy label, still emitted) | holdout `j` = record `T+j` | raw `T+j` | `full_scoring_worker.py:300`, `:305` |
| **holdout_quality** (the training target) | holdout `j` = record `T+j` | raw `s_rec + j`, `s_rec = skip_min` (5 on the saved artifact) | `full_scoring_worker.py:411-423` → `survivor_scorer.py:355` |
| `compute_holdout_hits` (single) | holdout `j` | raw `j` | `full_scoring_worker.py:231-270`. **No caller** (`grep "compute_holdout_hits("` → none outside the def) |

**Where they differ.** Training features and `holdout_hits` read one consecutive stream in which raw index equals combined record index (the `offset = train_history_len` law).
`holdout_quality` reads raw `s_rec … s_rec+H−1` (5…3618 on the saved artifact) for records 14,454…18,067. That is 14,449 outputs earlier than `holdout_hits` reads for the same draws, and inside the raw range the training features already used against train draws 5…3618.
None of the three uses an inter-draw gap. The batch route and its CPU fallback differ in projection only. Measured on 64×200 values: torch generator on `device=cpu` == `_generate_sequence(skip=0)` (True); `_cpu_batch_generate` agreement 0.0 (max value 4,294,929,170); `_cpu_batch_generate % 1000` agreement 1.0.
For every family other than java_lcg the fallback is the primary route. For java_lcg it runs only if the torch call raises.

**As a hit-rate prediction (Beta R2 §2 asked which target is affected).** Step 5 learns to rank seeds by agreement at raw positions [s, s+H). Step 6 votes at raw position N (Q5).
If the ranking signal is tied to raw position, a model trained on `holdout_quality` will rank on information from a different part of the stream than the one it votes from.
The measurement that settles it is WP-3's frozen-pool hit rate on unseen draws, comparing the current target against the Option D target in Q8.

---

## Q5 — Prediction route

**Vote index.** For a survivor with recorded skip `s` and history length `N`, Step 6 votes `seq[N]` of `_generate_sequence(seed, N+1, skip=s6)`, which is **raw output `s6 + N`** (`prediction_generator.py:839-849`).
`s6 = _get_survivor_skip(...)` is 0 for every survivor, whatever the sieve recorded (Q3). Under the manifest (`agent_manifests/prediction.json` `default_params.lottery_history = "daily3.json"`), N = 18,068, so the vote is **raw 18,068 regardless of `s`**.
That continues the training/`holdout_hits` stream (record N is the next draw). Kernel stepping for the same draw would be raw `s + N·(s+1)`. Measured raw-index identity: 84/84 and 200/200 at `s = 0`, 0/84 and 0/200 at `s ∈ {1, 5, 9}`.

**Stored versus fresh features (addendum §D.2).**

| survivor arrives as | features used for ranking | generator route | anchor |
|---|---|---|---|
| dict with `features` | **stored** Step-3 features (train route: raw 0…T−1 vs train draws) | none at ranking; vote via P9 | `prediction_generator.py:782-784` |
| int | **fresh** `extract_ml_features(seed, lottery_history, skip=default_skip)` over **all N draws**, raw 0…N−1. That is a longer history than the model was trained on (T) | P8 | `:785-790` |

`use_dual_sieve` defaults to True (`prediction_generator.py:112`). When reverse survivors are given, `compute_dual_sieve_intersection` returns **sorted ints** (`survivor_scorer.py:139-179`, `_coerce_seed_list` `:128-137`), so the manifest route (forward and reverse both passed) always takes the **fresh** branch (`prediction_generator.py:606-616`).
A saved-artifact comparison that feeds `survivors_with_scores.json` dicts with no reverse file exercises the **stored** branch.
For Q8: under any option that changes generation, the stored branch needs `survivors_with_scores.json` regenerated (a Step 3 re-run), and the fresh branch changes at Step 6 with no regeneration.

**Manifest inputs, observed (belongs to WP-2).** Step 1 now writes `forward_survivors.json` and `reverse_survivors.json` as summary dicts (`window_optimizer_integration_final.py:2945-2952`). Live main-tree files: 126 bytes, `{"survivor_count": 0, "note": …}`.
By reading, Step 6 loads such a file as a dict (`prediction_generator.py:1042-1050`). The intersection then calls `int('survivor_count')`, which raises `ValueError` (`survivor_scorer.py:136`) if the signal gate passes first. **Not executed this session — [UNVERIFIED by execution].**
The manifest lists `survivors_with_scores.json` as a required input but does not pass it to the script.

**Other pool builders.** Search: `--include=*.py` for `next_draw_prediction|prediction_pool|_build_prediction_pool|def generate_predictions`, and for `seq[next_idx]|[len(lottery_history)]`.
- `build_pools.py`: votes the value after `best_skip + 1` advances from the seed, i.e. **raw `best_skip`**, not raw N (`:210-226`). It supports xorshift32 only and raises `ValueError` for any other family (`:66-78`). Driven by `backtest_pools.py:98`, read by `watcher_kpi_baseline.py:7`; not in `STEP_SCRIPTS`.
- `evaluate_pools.py`, `chapter_13_diagnostics.py`: grade pools, generate nothing (no generator import).
- The Chapter-13 retrain re-runs Steps 3→5→6 (skill §8 records `chapter_13_triggers.py:630` carrying its own `STEP_SCRIPTS`; not re-read this session). It therefore uses the same routes.

---

## Q6 — Window anchor and generator phase

Brief I documents read this session:
- `docs/PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md` §1-§3, §4.7 (firewall)
- `docs/TB_RULING_WINDOW_ANCHOR_SEQUENCING.md:14`, `:19`, `:52-55` (`continuation_phase`)
- `docs/S172_WINDOW_ANCHOR_BRIEF_I.md:168`
- `docs/S172_WINDOW_ANCHOR_BRIEF_I_REPORT.md` §10-§11 (`:455-472`)
- commits `48a8705` and `69ca910`

Live code: `load_residue_window` applies the session filter, validates `0 ≤ window_anchor ≤ N_filtered − window_size` with no clamp, and slices `filtered[anchor : anchor+ws]` (`miner/range_miner_worker.py:867-931`).
`generator_phase` travels separately in the kernel's `offset` slot (`:295-310`) and is pinned to 0 in v1. The canonical array-4 `offset` now carries `window_anchor` (`utils/canonical_records.py:217-225`).
**Note:** the installed tfm-project-facts skill (v27.1) still says Brief I "does not exist". At `0066b6e` it has landed.

**After Brief I, a constant-mode survivor means:** `filtered[anchor + i]` matched raw `g + s + i·(s+1)` (g = 0) for at least a threshold fraction of `i < ws`.
**The CPU routes assume:** combined record `r` ↔ raw `r` (training, holdout_hits, Step 6), or raw `s_rec + j` (holdout_quality). There is no session filter and no anchor, and raw index 0 is tied to combined record 0.

To line up with a window that does not start at draw 0, a CPU route would have to know, per survivor or trial:

1. the **session set** (the sieve's index space is the filtered stream; the CPU's is the combined one), recorded as `sessions`;
2. `window_anchor` (array 4), recorded;
3. `generator_phase`, which is 0 in v1. It is recorded in generation metadata, not in the 22 arrays (`utils/canonical_records.py:219-224` comment). Its exact field was **not traced this session**;
4. the **survivor's own skip** (constant) or skip sequence (hybrid), which is **not available** (discarded, Q3);
5. a **mapping rule** from a combined-history draw outside the window to a raw index. Under constant stepping, the draw at filtered index `f` corresponds to raw `g + s + (f − anchor)·(s+1)`. That expression is defined only for `f ≥ anchor`; the sieve made no claim about records before the window. Draws from the other session have no defined position; this follows from the per-session ruling in skill §2.10b.

The Step-3 law `offset = train_history_len` is **firewalled**. It is not repealed or parameterized by Brief I, and any change needs its own ruling (`TB_RULING_WINDOW_ANCHOR_SEQUENCING.md:19`, `:55`). Brief I's report confirms `full_scoring_worker.py` was not touched (`S172_WINDOW_ANCHOR_BRIEF_I_REPORT.md:467-472`).

---

## Q7 — Size of the difference (plumbing only, not a hit rate)

`data/wp1_stage_a/q7_route_agreement.py`, CPU-only (`CUDA_VISIBLE_DEVICES=""`, `torch.cuda.is_available()` asserted False), java_lcg. Peak RSS 428 MiB, 94.3 s, completion sentinel present (`q7_run.log`).

- **Oracle.** Constant-kernel stepping, draw `j` → raw `g + s + j·(s+1)`, g = 0. Computed by affine jump.
- **Clean control.** The oracle equals the independent `known_answer_reference.generate_constant_stream` for 4 seeds × s ∈ {0,1,5,9} × g ∈ {0,3} × 40 draws: PASS.
- **Fault-injection control.** `java_lcg_cpu(skip=5)` vs oracle s = 5: draw 0 equal, sequence differs; detected: PASS.
- **Routes.** Called through the production functions: `SurvivorScorer._generate_sequence`, `get_cpu_reference`, `get_survivor_skip`, `PredictionGenerator._get_survivor_skip`. Signatures were printed with `inspect.signature` first.
- **Samples.** 84 seeds from `survivors_with_scores.json` (read-only) and 200 synthetic seeds (`random.Random(20261113)`).
- **What "agreement" counts.** For each draw, whether the route's `% 1000` value equals the oracle's, and whether the two raw indices are identical. Each route is compared in its own index space (combined history).

| route | s=0 value / index | s=1 | s=5 | s=9 | denominator (saved) |
|---|---|---|---|---|---|
| training features (raw j) | 1.0000 / all | 0.0010 / **0** | 0.0010 / **0** | 0.0010 / **0** | 1,214,136 |
| holdout_hits (raw T+j) | 1.0000 / all | 0.0011 / **0** | 0.0010 / **0** | 0.0010 / **0** | 303,576 |
| holdout_quality (raw s_route+j) | 0.0011 / **0** | 0.0010 / **0** | 0.0010 / **0** | 0.0010 / **0** | 303,576 |
| Step 6 vote, actual (`s6 = 0`) | 84/84 / 84 | 0/84 / **0** | 0/84 / **0** | 0/84 / **0** | 84 |
| Step 6 vote if `s6 = s` | 84/84 / 84 | 0/84 / **0** | 0/84 / **0** | 0/84 / **0** | 84 |

The synthetic sample (denominators 2,890,800 / 722,800 / 200) gives the same index zeros and value rates of 0.0009–0.0010. One synthetic Step-6 value matched at s = 1 (1/200) with a non-identical index.
The decisive column is **raw-index identity**. The value rates where indices differ are reported as counts; no conclusion is drawn from them.
**Limits.** java_lcg only. Constant-mode oracle only (no CPU route models a hybrid, Q2). The survivors' true sieve skip is **UNAVAILABLE** (discarded, Q3), so the table sweeps `s` instead. The torch route ran on `device=cpu`, which is not GPU execution evidence.

---

## Q8 — Proposed contract (not implemented)

**Named parameters**, one emitter, `emitted_j = reduce(project(raw[start_index + j·(gap+1)]))`:

- **`start_index`** (a): raw index of the first emitted output (advances before it). Constant kernel: `generator_phase + skip`. Today's CPU `skip` argument.
- **`gap`** (b): outputs discarded between emitted ones. A scalar, or a per-draw sequence for variable mode. Constant kernel: `skip`. Today's CPU: always 0.
- **`projection`** (c): a named raw→compared-value map: `cpu_native` (today's `*_cpu` return value), or `kernel_compare` (the kernel's compared value: full 64-bit state for xorshift64, next-state permutation for `pcg32_hybrid`, masked form for `lcg32_hybrid`).
- **`reduce_mod`**: the modulus, applied explicitly, never assumed. Today it is applied in P1 and P3 and silently skipped in P4 and P11.

**What each caller needs** (current value → the meaning it carries):

| caller | `start_index` | `gap` | `projection` / `reduce_mod` |
|---|---|---|---|
| P3/P4/P5 training features | 0 | 0 | cpu_native / 1000 (P4 today: none) |
| P6 holdout_hits | `T` (continuation_phase, firewalled) | 0 | cpu_native / 1000 |
| P7 holdout_quality | `s_rec` (= skip_min) | 0 | cpu_native / 1000 |
| P8/P10 Step 6 fresh features | `default_skip` (0) | 0 | cpu_native / 1000 |
| P9 Step 6 vote | `s6 + N` | n/a (single output) | cpu_native / 1000 |
| P12 synthetic injector | `position` | 0 | cpu_native / mod |
| build_pools (non-pipeline) | `best_skip` | n/a | xorshift32 only |

### Options for Stage B

| | intended change | target | features | Step 6 votes | retrain | files edited | must stay byte-identical |
|---|---|---|---|---|---|---|---|
| **A — document the current convention as intentional, change nothing** | write down that all CPU consumers model **one consecutive stream (gap 0) indexed from combined record 0**; that `skip` at P7 is a start position taken from `skip_min`; that Step 6 uses 0 | no | no | no | no | docs only (e.g. a contract section in `docs/CHAPTER_4_FULL_SCORING.md` / `CHAPTER_6`) | everything |
| **B — explicit-parameter refactor, current values** | new module (e.g. `generator_contract.py`) with `emit(family, seed, start_index, gap, n, projection, reduce_mod)`; P1–P12 pass today's values explicitly, so the meanings become visible; `prng_registry.py` untouched | no | no | no | no | new module; `survivor_scorer.py`, `full_scoring_worker.py`, `holdout_quality.py` (docstring), `prediction_generator.py` | every output of every route, including P4/P11's **unreduced** values (B keeps the defect as-is) |
| **B+ — projection repair only** | apply `reduce_mod` in P4/P11 | no (P7 uses P1) | **yes, wherever P4 runs**: every non-java family, and java_lcg after a torch failure | no | yes, for any model trained on P4 output | `survivor_scorer.py` | P1, P3, P5–P10 outputs; java_lcg torch-route features |
| **C — kernel stepping with each survivor's own skip** | `gap = best_skip` (constant), `gap = skip_sequence` (hybrid), `start_index = g + best_skip`, applied to P3–P9 | yes | yes | yes | yes | needs per-seed `best_skip`/`skip_sequence` carried through Phase-5 assembly and the record: `miner/range_miner_npz_writer.py`, `utils/canonical_records.py`, and a **22-array contract change** | — |
| **C′ — kernel stepping with the trial's `skip_min` as gap** | as C but with `gap = skip_min`, `start_index = skip_min`, and the continuation rewritten to `skip_min + r·(skip_min+1)` | yes | yes | yes | yes | `survivor_scorer.py`, `full_scoring_worker.py`, `prediction_generator.py`, new module | — ; **changes `continuation_phase` (P6), which needs its own Beta ruling**. `skip_min` is the trial's search lower bound, not the survivor's skip, so this tests a different hypothesis from C (owner's call) |
| **D — put `holdout_quality` on the continuation stream** | P7 `start_index = T` (matching P6 and P9), `gap = 0` | **yes** | no | no | yes | `full_scoring_worker.py` (`:412`, `:473` call sites) | features, holdout_hits, votes. Reverses S111's per-survivor-skip choice (`docs/S111_IMPLEMENTATION_PLAN_FINAL.md:481`), which Beta approved, so it needs a ruling |

**No-touch confirmation.** A, B, B+, C′ and D edit only `survivor_scorer.py`, `full_scoring_worker.py`, `holdout_quality.py`, `prediction_generator.py`, docs and a new module. None is on the no-touch list or in `GOVERNED_FILES` (`scripts/gate12_parity_gate.py:174-185`).
**Option C needs no-touch files (`miner/`, `utils/canonical_records.py`) and the frozen 22-array contract. It is DEFERRED** (proposal §0, "Deferred, not dropped").
`survivor_scorer.py` is in the rig deploy lists (`deploy_to_rigs.sh:17`), so any option except A needs a rig deploy before Step 3 next runs. That deploy is Michael-initiated.

**Compatibility controls for Stage B (addendum §D.3).** Each is an expected output with its raw indices, frozen before the change:

1. gap 0 and nonzero gap (s ∈ {1,5,9}) for every route, with expected indices per the Q4/Q5 tables;
2. nonzero `start_index`, and nonzero `generator_phase` in the oracle (g = 3, as in Q7's control);
3. missing skip metadata: `get_survivor_skip({})` → 0 and `{"skip_min": None}`; Step 6 int survivor; a dict survivor with no skip keys;
4. the **batch and every fallback route** run separately: P3 torch (CPU **and GPU** device), P4 `_cpu_batch_generate`, P5 worker sequential fallback (forced by making the batch raise), P11;
5. the old start-position caller P6 at `skip = T` must reproduce today's `holdout_hits` byte-for-byte under A, B, B+ and D;
6. the xorshift64, `pcg32_hybrid` and `lcg32_hybrid` projections (Q2) for any option that names `kernel_compare`;
7. the stored versus fresh Step 6 branch: one saved-artifact run with dicts and no reverse file, and one with ints.

Per option, "byte-identical" is checked as a diff of `survivors_with_scores.json` (features, `holdout_hits`, `holdout_quality`) and of Step 6 `predictions`, on one saved survivor file, before and after.
**CPU-only evidence is not GPU execution evidence.** P3 runs `java_lcg_pytorch_gpu` on CUDA (VM101 3080 Ti) and ROCm (rigs) in Step 3. Stage B needs a GPU run of P3 on both backends. That is Michael-initiated, and this session ran neither.

---

## UNAVAILABLE

- GPU execution of the P3 torch route on CUDA or ROCm (CPU-only session; no GPU, no rig SSH).
- Rig-deployed copies of `survivor_scorer.py` and `full_scoring_worker.py` (not compared; no rig SSH).
- The survivors' actual sieve skip (`best_skip`, `skip_sequence`): discarded at Phase-5 assembly, so Q7 sweeps `s` instead.
- Survivor rows in the current certified generation (0 rows).
- The effective Step 6 config on a real run: `reinforcement_engine_config.json` was absent from the main tree at the time of this check. A real run's value is not known; `PredictionConfig` defaults to `default_skip = 0`, `use_dual_sieve = True`.
- End-to-end execution of Step 6, of `extract_ml_features_batch` and of `compute_holdout_hits_batch`. Their generators were exercised; the full functions were not run.
- Hybrid kernel behaviour for the 10 non-java families: reading only, nothing executed.
- The exact metadata field that carries `generator_phase` post-Brief-I (Q6 item 3): not traced.
- `chapter_13_triggers.py:630` (cited from the skill, not re-read).
- The ser8 pre-repository archive and host state beyond `/etc/systemd/system` and `crontab -l`. The only related unit found is `daily3scraper.service`; there is no crontab.

## Commands used (all on VM101 as `michael`)

```
cat docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md docs/CLAUDE_CODE_INSTRUCTIONS_CPU_GPU_GENERATOR_CONTRACT_STAGE_A_20261113.md
pgrep -af "range_miner_coordinator|watcher_agent|window_optimizer"; ps -eo pid,user,args | /bin/grep -E "[r]ange_miner_coordinator|[w]atcher_agent|[w]indow_optimizer"; ss -ltn | grep 5700
git -C /home/michael/distributed_prng_analysis status --porcelain; git worktree list
git worktree add --detach /home/michael/tfm_worktrees/wp1_stage_a HEAD
sed -n / awk / /bin/grep -n over: prng_registry.py, survivor_scorer.py, full_scoring_worker.py, holdout_quality.py,
  prediction_generator.py, window_optimizer.py, window_optimizer_integration_final.py, miner/range_miner_worker.py,
  miner/range_miner_npz_writer.py, miner/range_miner_protocol.py, utils/canonical_records.py, utils/survivor_loader.py,
  agents/watcher_agent.py, agent_manifests/prediction.json, scorer_trial_worker.py, reinforcement_engine.py,
  synthetic_draw_injector.py, build_pools.py, scripts/gate12_parity_gate.py, tests/phase6/known_answer_reference.py,
  docs/BACKLOG.md §8.1, docs/CHAPTER_6_AUDIT.md, docs/CHAPTER_4_FULL_SCORING.md, docs/S111_IMPLEMENTATION_PLAN_FINAL.md,
  docs/PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md, docs/TB_RULING_WINDOW_ANCHOR_SEQUENCING.md,
  docs/S172_WINDOW_ANCHOR_BRIEF_I*.md, docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md,
  docs/exchange/from_beta/X-REVERSE-SIEVE_R2_BETA_REALIGNMENT.md
/bin/grep -rn --include=*.py -E "get_cpu_reference|\['cpu_reference'\]|...|cpu_reference\b" .        (Q1 inventory)
/bin/grep -rln "skip_best" . ; git log --oneline -S skip_best ; /bin/grep -rn --include=*.py skip_hypothesis .
python3 -I (read-only) on main-tree daily3.json, daily3_current.json, train_history.json, holdout_history.json,
  survivors_with_scores.json, forward_survivors.json, reverse_survivors.json, bidirectional_survivors_binary.npz
PYTHONDONTWRITEBYTECODE=1 python3 data/wp1_stage_a/kernel_offset_scan.py prng_registry.py | tee data/wp1_stage_a/kernel_offset_scan.log
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python3 -u data/wp1_stage_a/q7_route_agreement.py /home/michael/distributed_prng_analysis | tee data/wp1_stage_a/q7_run.log
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python3 -u data/wp1_stage_a/q2_projection_check.py | tee data/wp1_stage_a/q2_projection_check.log
```

Scratch artifacts (untracked, inside the worktree, not for commit): `data/wp1_stage_a/{kernel_offset_scan.py,.log, q7_route_agreement.py, q7_run.log, q7_stderr.log, q7_results.json, q2_projection_check.py,.log}`.
The first Q7 attempt failed at import (`ModuleNotFoundError: tests.phase6`). It was fixed by adding `tests/phase6` to `sys.path` and rerun; the logs are from the rerun.
**Main tree at close:** `git status --porcelain` empty. The only main-tree write this session is the worktree registration under `.git/worktrees/wp1_stage_a`, made by the brief's own command.
