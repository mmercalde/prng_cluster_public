# S172 — RUN-4 PROPOSAL: PINNED GEOMETRY, A/B AGAINST THE CERTIFIED GATE-12 PASS

**Status: PROPOSAL. Not authorized, not launched, nothing committed.** For Michael and Team Beta
before any rerun. Read-only investigation; the only artifacts produced are documents.

**Companions.** `S172_PHASE3_SURVIVOR_CAPACITY_CHARACTERIZATION.md` (the measurements this rests
on) · `RIG_CRASH_FORENSIC_20260822.md` (why run 3 is not a volume crash) · `LEADS.md` L-1, L-2.

**⚠ HEADLINE FINDING, AND IT CHANGES THE ASK.** The geometry pin **cannot** be expressed as a
launch parameter on the certified Gate-12 path. Every route requires either a **tracked-file
change** or **abandoning the certified launch harness**. §2 carries the source evidence. This is
disclosable, and it is the decision Beta actually has to make before the geometry is discussed.

---

## 1. THE PINNED CONFIGURATION

```
window_size            12          <- Optuna dimension, must be pinned
window_anchor (offset) 25          <- Optuna dimension, must be pinned
sessions               ["midday"]  <- Optuna dimension (session_idx = 1), must be pinned
forward_threshold      0.71        <- Optuna dimension, must be pinned
reverse_threshold      0.47        <- Optuna dimension, must be pinned
skip_min / skip_max    6 / 99      <- Optuna dimensions, must be pinned
generator_phase        0           <- ALREADY pinned in code; no action required
window_trials          1           <- single pinned draw, NO Optuna sampling
n_parallel             1           <- BINDING (D6.2 certified for n_parallel == 1 only)
prng_type              java_lcg    test_both_modes true
max_seeds              2147483648  miner_stripe_size 67108864  -> 32 macro-stripes/stage
worker_pool_size       25          use_range_miner true   use_persistent_workers false
--start-step 1 --end-step 1        MANDATORY
```

Every value is attempt 9's (`distributed_config_t1_554463d3`, launch commit `e9ca800`, tag
`gate12-passed-attempt9`), read from that run's own `trial_context` row — **except**
`generator_phase`, which is 0 under the Brief I v1 pin where attempt 9 ran 25.

### 1.1 Predicted envelope — check the actuals against these at run time

| quantity | predicted | attempt 9 actual | method |
|---|---:|---:|---|
| phase-3 survivors (`java_lcg_hybrid`) | **~96** | 59 | production kernel, 2²⁶ sample, 3 hits |
| phase-4 survivors (`java_lcg_hybrid_reverse`) | **~22,656** | 23,515 | production kernel, 2²⁴ sample, 177 hits |
| total staged bytes | **~1.25 MiB** | 2.02 MiB | 18 + 3.35k B/survivor at k=12 |
| shard files | **~5,693** (87.2% of 6,528) | 5,693 | geometry-determined, survivor-independent |
| peak coordinator inbound queue | **~2.8 MiB** | — | 1024 × 1e6 × r × 271 B |

**Both survivor figures are COMPUTED projections from a MEASURED rate**, taken on attempt 9's own
residue window (`sha256_residues` verified against `trial_context.residue_sha256 =
c761272da958ba34…`). They are predictions to be checked against actuals, not guarantees.

### 1.2 Every parameter against the validated bound it sits under

| parameter | value | validated bound it sits under | source of that bound |
|---|---|---|---|
| `window_size` | 12 | inside the §8 envelope (`k ≥ 10`); k=6 and k=8 are the excluded columns | capacity doc §8, measured grid |
| `window_anchor` | 25 | `0 ≤ anchor ≤ N_filtered − window_size = 8,503` (midday, 8,515 records); also ≤ the historical anchor ceiling of 100 | `load_residue_window` derived domain; `search_bounds.offset` |
| `sessions` | `["midday"]` | single-session — combined-session sequential sieving is **prohibited by default** | TB dataset-lifecycle ruling (§2.10b) |
| `forward_threshold` | 0.71 → M=9 | predicted 96 ≤ **774**, the largest phase-3 volume ever to complete four GPU stages (attempt 3) | capacity doc §5 |
| `reverse_threshold` | 0.47 → M=6 | predicted 22,656 ≤ **23,515**, the largest single-phase volume ever carried through a complete certified run (attempt 9 phase 4) | capacity doc §5 |
| `skip_min`/`skip_max` | 6 / 99 | inside `search_bounds` (`skip_min ≤ 10`, `skip_max ≤ 250`); **inert for survivor count** — hybrid kernels ignore sampled skip bounds, so this is provenance, not a control | `distributed_config.json`; `_load_strategies` (§2.7 instance 4) |
| `generator_phase` | 0 | the Brief I v1 pin, enforced at two independent seams | `build_stripe_assign_payload` + `assert_generator_phase_permitted` |
| total staged bytes | ~1.25 MiB | ≤ **2.02 MiB** (attempt 9); 0.007% of the 16 GiB `staging_high_water_bytes` | capacity doc §4.2, §5 |
| shard files | ~5,693 | identical to attempt 9; the bound is survivor-independent | capacity doc §4.1 |
| peak inbound queue | ~2.8 MiB | vs `MemTotal` 15,924.8 MiB — **three orders of magnitude** below the computed ceiling | capacity doc §4.4, L-2 |
| `n_parallel` | 1 | D6.2 is certified for `n_parallel == 1` **only** | §2.9 |

**Every parameter sits under a bound demonstrated by a run that completed.** No value in this
configuration is outside anything that has been shown to work.

## 2. ⚠ IT IS NOT A LAUNCH-PARAMETER PIN — VERIFIED AT SOURCE

`gate12_launch.sh` is tracked and clean at `69ca910`; the tree is clean-able today. **But the pin
itself cannot be delivered as a parameter.**

**The seven quantities to pin are Optuna search dimensions**, sampled inside the objective
(`window_optimizer_bayesian.py:529-549`): `window_size`, `offset`, `session_idx`, `skip_min`,
`skip_max`, `forward_threshold`, `reverse_threshold`. Pinning means either enqueuing them or
removing them from the search space.

**The mechanism that exists is S166 warm-start, and it is complete end to end:**

```
--warm-start-{window,offset,skip-min,skip-max,fwd-thresh,rev-thresh,session-idx}
  window_optimizer.py:1514-1528   (CLI; session-idx: 0=midday+evening, 1=midday, 2=evening)
    -> :1869-1875                 getattr(args, 'warm_start_*')
      -> window_optimizer_integration_final.py:2832-2848   _trial_history_ctx
         ("explicit warm-start params — override DB lookup")
        -> window_optimizer_bayesian.py:774-786            study.enqueue_trial(_ws_params)
           seven keys, exactly matching the seven suggest_* names
```

With `trials = 1`, the single trial **is** the enqueued one. The `round(…, 2)` at `WindowConfig`
construction (§2.36) is a no-op on 0.71 / 0.47.

**But Gate-12 launches go through WATCHER, not `window_optimizer.py` directly**
(`gate12_launch.sh:377` — `python3 agents/watcher_agent.py --run-pipeline --start-step 1
--end-step 1 --params '{…}'`), and **two independent filters block the warm-start keys there:**

1. **`_step1_declared_params`** (`agents/watcher_agent.py:1290-1314`) drops any `--params` key not
   present in the manifest's `default_params`. That manifest declares **32 keys, and none of the
   seven** `warm_start_*`.
2. **`_INTERNAL_ONLY_PARAMS`** (`agents/watcher_agent.py:1840-1847`) **strips all seven warm-start
   keys from the built command line even if declared** —
   *"[S167] warm_start_\* are not S114-S116 resume args — strip from CLI entirely."*

**A §2.15-class dead chain, found here and worth recording on its own.** The manifest's
`actions[0].args_map` **does** declare all seven `warm-start-*` CLI arguments — someone intended
them to be routable — while `watcher_agent.py:1840` strips them unconditionally, so those
args_map entries are **inert**. The same args_map declares `forward-threshold` and
`reverse-threshold`, whose own CLI help reads *"UNWIRED — passing this ABORTS the run
(`WINDOW_OPTIMIZER_THRESHOLD_OVERRIDE_UNWIRED`)"*. **Three inert-or-hazardous declarations in one
manifest.** Not repaired here.

### 2.1 The four routes, and what each costs

| route | tracked files edited | what it costs |
|---|---:|---|
| **A** — declare the 7 keys in `default_params` **and** narrow `_INTERNAL_ONLY_PARAMS` | **2** — `agent_manifests/window_optimizer.json` + `agents/watcher_agent.py` | edits WATCHER core and reverses an explicit S167 decision; the S167 rationale must be found and ruled on first |
| **B** — narrow `distributed_config.json` `search_bounds` to a single point | **1** | changes the **global** search space for every later run until reverted; `window_size.min` carries a TB ruling (`_s172_note`), so narrowing it is a governed edit, not a config tweak |
| **C** — add `config_file` to the manifest and use `--config-file` | **1** | `run_with_config` is a **different execution path** (`window_optimizer.py:1964`, "skips optimization") — not the path attempt 9 ran, which destroys the A/B |
| **D** — bypass WATCHER, call `window_optimizer.py` directly with `--warm-start-*` | **0** | abandons the certified launch harness and **G-PROD-SHAPE's "real WATCHER execution → manifest defaults" requirement** (§2.54), so the run could not serve as production-shape acceptance |

**No route is free.** My reading is that **A is the most honest** — it makes the pin explicit,
auditable and reusable, and it repairs a dead chain the manifest already advertises — but it is a
WATCHER-core change reversing a prior decision, so **it is a Beta call, not an Alpha one.** No
implementation is proposed here.

**Whatever is chosen, the tree must be clean at launch.** Clean-tree admission refused attempt 3
on exactly this, and it reads `git status --porcelain`, so **modified-tracked trips it too**. This
session leaves working-tree entries (this document and the other new ones, plus `BACKLOG.md`,
`CLAUDE.md` and the skill copy); all are Michael's to commit before any launch.

## 3. WHY THIS GEOMETRY — THE A/B ARGUMENT

**Run 4 is an acceptance run for Brief I. The strongest available design is a controlled
comparison against the only certified Gate-12 pass**, with exactly one variable moved.

```
attempt 9   e9ca800   k=12 anchor=25 midday tau 0.71/0.47 skip 6-99   generator_phase = 25
run 4       69ca910   k=12 anchor=25 midday tau 0.71/0.47 skip 6-99   generator_phase =  0
                                                                      ^^^^^^^^^^^^^^^^^^^^
```

Everything else — geometry, dataset, seed domain, stripe size, fleet, worker pool, backend — is
held. **Brief I is the single moved variable**, and `generator_phase` 25 → 0 is precisely the
change Brief I made: `RIG_CRASH_FORENSIC_20260822.md` §1–2 established that nothing else Brief I
touched can alter device execution — kernel bodies byte-identical (44/44, aggregate SHA
`cc75bddf70dd1345…`), arg tuple identical, launch pattern, allocation and per-launch timing
unchanged, ~3 µs of added pre-device work per sub-stripe.

**What run 4 CAN validate:** volume against prediction · four-stage plumbing end to end ·
completion through publication under the Brief I schema (`window_anchor` / `generator_phase` in
the payload, ledger, trial context, manifest metadata and canonical array 4) · the S145 coverage
and cursor path · the six pre-coordinator gates against the new required-key contract.

**What run 4 CANNOT validate, and this must not be claimed afterwards: population equivalence.**
Pinning phase to 0 starts the generator at a different trajectory point for the same anchor, so it
selects a **different set of seeds** even though the counts land within a few percent.
`PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md` says so in its own comparability
caveat: post-separation phase-zero populations are **not legitimate regression comparators** to
historical populations. **No run can validate that, ever.** Count agreement between ~96 and 59, or
~22,656 and 23,515, is evidence about *volume and plumbing* and about nothing else.

## 4. WHY PINNING IS REQUIRED, NOT PREFERRED

**Of the 64 measured forward cells in the reachable space, 35 (55%) predict a phase-3 survivor
count above the validated any-phase maximum of 23,515.** With `window_trials = 1`, a run gets
**one** draw. A free Optuna sample is therefore better than even money to land outside the
validated envelope, and the tail is not gentle: the reachable maximum is **137,973,376** at
`(k=6, τ=0.30)` — **10.5× run 3** and ~5,900× the validated maximum.

That corner is provably the worst case: `min kτ = 6 × 0.30 = 1.8 > 1` forces `M ≥ 2` everywhere,
and `M = 2` requires `k ≤ 2/0.30 = 6.67`, i.e. **k = 6 alone** — exactly one M=2 cell in the whole
space, and it is the maximum.

**`window_size = 6` has no safe threshold at all**: its minimum reachable forward count is 31,872,
at `τ = 0.75`, the top of the range. Run 3 drew `k = 6, τ = 0.35` — one cell away from the worst
in the space — on an ordinary sample.

**This is an argument for pinning run 4, not for changing the search space.** Whether the
production bounds should exclude `k = 6` is a separate question with its own TB-ruling history
(`window_size.min` was raised 2 → 6 by ruling, with a recorded rationale), and **nothing here
proposes touching it.**

## 5. WHAT THIS PROPOSAL ASKS FOR

1. **A ruling on the pinning route (§2.1, A–D).** This is the blocking decision and it precedes
   any discussion of the geometry.
2. Confirmation that the §1 configuration and its §1.2 bound-by-bound justification are accepted.
3. Acknowledgement of the §3 claim boundary: **volume, plumbing and completion — never population
   equivalence.**
4. Recording of the §2 dead chain (a manifest `args_map` declaring seven `warm-start-*` arguments
   WATCHER strips, plus two threshold arguments that abort the run) as a separate item.

**Not requested and not proposed:** any repair to L-1 or L-2, any change to `search_bounds`, any
crash-seeking ladder, and any launch. **Launching is Michael's.**
