# PLAN: the first multi-trial Step 1 search with `deliver_skip_range` on (sampled, not pinned)

Draft, uncommitted, planning only: **no run.** The owner said "please proceed" to planning at 2026-11-22 11:18.
Written by Claude (Team Alpha) in `/home/michael/tfm_worktrees/pwc-offset`.

## Summary (plain language)

1. Today's four runs pinned one geometry. This plans a search that samples window, anchor, sessions, thresholds and the skip range.
2. Launchable with no code change, except the derived launcher, which requires pins. Every switch check passes. Only TPE runs, unseeded.
3. The big risk is volume from loose thresholds, in both arms: k=12, τ=0.30 projects about 2.4M phase-3 survivors per 2^28, 47× the cap.
4. The volume gate checks one exact geometry, not a sampled search, and nothing enforces the cap during a run.
5. A fair on/off pair needs both arms to try the same tuples. Unseeded TPE can't; a fixed trial source can (a small, reviewed change).
6. Options: 10, 25 or 50 trials on the Stage 4 interval (about 30 min, 70 min or 2.3 h per arm), or 10 on [2^31, 2^32) (about 1 h).
7. None says anything about hit rate. That needs Steps 2–6 and about 1,400 unseen draws to tell 2% from 3% pools.
8. Recommended: narrow the bounds (this changes a recorded ruling) or gate each trial (it doesn't); fix the trial source; then O1 as a pair. Decisions D1–D7: §6.

## Prior record

All searches were run read-only against `/home/michael/distributed_prng_analysis` (main, `00e2ee7`) with
`/bin/grep`. Every hit listed was read in context.
`SC="docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_* docs/TB_RULING_REQUEST_* docs/PROPOSAL_* docs/CHAPTER_* docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md"`

| # | Command | What it returned, and what it decides |
|---|---|---|
| M1 | `/bin/grep -l -i -E "cap_survivors\|survivor_volume\|50,000\|50000" $SC docs/TFM_SESSION_HANDOFF_2026111*` | **`config_manifests/survivor_volume.cap`** sets 50,000 per phase, `compare: point`, `min_sample_seeds` 2^26. Its evidence field: owner 2026-11-19, raised from 25,000, "Raised only by the owner". The 11-19 handoff (`:31`, `:37`) records it. **`docs/S172_PHASE3_SURVIVOR_CAPACITY_CHARACTERIZATION.md`**: §1 measured phase-3 counts across every geometry run (k=6, τ=0.35: 13,146,485); §3.1 a projection table over the Optuna box (max 137,973,376 at k=6, τ=0.30); §4.4 the binding limit is coordinator RAM, not retention. ALREADY RECORDED |
| M1b | `/bin/grep -rn "estimate_volume(" --include=*.py . \| /bin/grep -v tests/` and `/bin/grep -n "survivor_volume\|volume_gate" agents/watcher_agent.py` | The only caller is `scripts/survivor_volume_gate.py:308`. **WATCHER does not call the gate.** The estimator's docstring (`utils/survivor_volume_estimator.py:15-16`) says "a later brief will call the same function once per Optuna trial"; that is not built. The gate refuses a range: "a parameter missing, unknown, or a range → REFUSE" (`scripts/survivor_volume_gate.py` docstring). ALREADY RECORDED |
| M2 | `/bin/grep -l -i -E "trials_per_chunk\|window_trials" $SC` | **`docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md`:**<br>§2.3 (`:184-201`): credit needs at least one COMPLETE trial of this chunk that swept every required phase over the whole interval; zero survivors are credited; a link is written after publication.<br>§2.4 (`:205`): one open search per family and mode set.<br>§2.7 (`:248-257`): a search owns one study; each chunk attempts `trials_per_chunk` trials; COMPLETE, PRUNED and FAIL each consume one; a used-up budget with no COMPLETE trial is uncredited; resume follows D6.2 with `n_parallel == 1`.<br>The manifest declares `window_trials` in `[10, 200]`, default 50 (`agent_manifests/window_optimizer.json:134-141`), but today's runs used 1, so the declared minimum is not enforced on this route (measured).<br>ALREADY RECORDED |
| M2b | `TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md` (V1/V3 of X-MINER-SKIP-SWITCH-2 R3) | Run 4's pins: k=12, anchor 25, midday, 0.71/0.47, skip 6–99, `window_trials=1` "produces exactly the pinned trial" (`:121`); all seven pins or none (`:123`, `:130`). ALREADY RECORDED. A sampled search is outside that ruling's pinned case |
| M3 | `/bin/grep -l -i -E "n_parallel *> *1\|parallel trials" $SC docs/PWC_BACKUP_OWNER_RULINGS_20261119.md` | **`docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:49-51`:** n_parallel > 1 is switched off by ruling on the miner path; parallel trials are a separate, parked job, targeting half-range trials. **`docs/BACKLOG.md:296-300`:** D6.2 is certified for `n_parallel == 1` only. The code refuses only `resume_checkpoint` with `n_parallel > 1` (`window_optimizer_integration_final.py:2744`). ALREADY RECORDED |
| M4 | `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md` §1 | "TFM is judged by one thing: whether its prediction pools hit more often than `k/1000` on draws the sieve never saw." Not by whether "the true seed" survives. ALREADY RECORDED; it governs §4 |
| M5 | `/bin/grep -l -i -E "holdout\|held-out\|unseen draw" $SC docs/HOLDOUT_HITS_IMPLEMENTATION_SUMMARY.md` | `HOLDOUT_HITS_IMPLEMENTATION_SUMMARY.md` (2025-12-31) frames holdout as telling "TRUE" from "LUCKY" seeds. **That is a state-recovery framing, which M4 rejects;** it is not used as the yardstick here. The pool metric is `exact_hits / pool_size` (`docs/CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md:417`; `chapter_13_llm_advisor.py:282-284`). ALREADY RECORDED |
| M6 | `/bin/grep -n "OptunaBayesianSearch(\|seed=" window_optimizer.py window_optimizer_bayesian.py`; `sed -n 515,572p window_optimizer.py` | Step 1 builds TPE with **`seed=None`** (`window_optimizer.py:434-436`). Only strategies whose `search()` matches the calling convention run (`require_supported_strategy`, `:543-570`); `random`, `grid` and `evolutionary` are refused as UNSUPPORTED. `OptunaRandomSearch` (`optuna_random_control`) exists in `window_optimizer_bayesian.py:969-1016`, but no Step 1 caller was found. ALREADY RECORDED |
| M7 | `/bin/grep -rl -i "deliver_skip_range\|constant_skip_delivery" docs/*.md` | Only `docs/TFM_SESSION_HANDOFF_20261121.md`. The switch's governing records are this directory (`PLAN.md`, `READY_TO_LAND.md`), the Beta threads (`docs/exchange/LEDGER.md`: X-SKIP-F1 R1, X-MINER-SKIP-SWITCH R1–R3, X-MINER-SKIP-SWITCH-2 R1–R3) and `docs/RESULT_SKIP_SWITCH_RIG_RUNS_20261122.md` (this worktree, draft). ALREADY RECORDED |
| M8 | the switch's own rules (`PLAN.md` §R2, §R3, §SW2, §SW3) | Search identity carries `constant_skip_delivery`. A study is tagged on its first switched trial. Unnamed auto-select skips tagged studies, and is refused for an on run. A named study must carry the arm's tag. One output root per arm, one publisher at a time; the marker must survive backup and restore. A sampling search validates the full domains under one rule shared by search open and the child. ALREADY RECORDED |
| M9 | D6.2 §4.3 (`docs/CLAUDE_CODE_INSTRUCTIONS_S172_D6_2_CHECKPOINT_RECONCILIATION.md:194-199`) | The owner approved, 2026-11-22 08:34, that `constant_skip_delivery` joins the run-context digest when on: "i approve as long as it doesnt break,confuse or in any other way slow down development". ALREADY RECORDED |
| M10 | `git check-ignore -v distributed_config.json` | Not ignored (`!*_config.json`), so **the sampling bounds are tracked**. Changing them in a run tree dirties it, and the clean-tree gate refuses. Narrowing the bounds needs a committed change |
| M11 | `sed -n 76,80p distributed_config.json` | `:78`, the S172 TB ruling, 2026-04-30: `window_size.min` 2 → 6, "Threshold bounds intentionally PRESERVED so Optuna can continue optimizing across [min, max]." ALREADY RECORDED, and it governs D2 |

**Labels:**
- **NEW:** this plan's options, sizing and comparison design.
- **ALREADY RECORDED:** M1–M11.
- **CHANGES A RULING:**
  - **D2's threshold narrowing:** it would reverse the S172 TB ruling recorded at `distributed_config.json:78` (2026-04-30: "Threshold bounds intentionally PRESERVED so Optuna can continue optimizing across [min, max]");
  - **raising the cap:** it is "Raised only by the owner";
  - **D6 changed to `n_parallel > 1`:** it would undo the parking ruling (M3).

**Not searched:** the host, the ser8 archives, the private repository, the rigs.

## 1. How a sampled switched-on search is launched today, and what refuses

**Launch path (no code change).** It is runbook C's sequence with no pins:
1. A fresh tree, staging, the ignored inputs and provisioning (P0–P3).
2. Params with the switch on, **no `warm_start_*` keys**, and `window_trials` N.
3. Search open (P8).
4. The derived launcher.

The params for the Stage 4 interval:
```json
{"prng_type": "java_lcg", "seed_start": 0, "max_seeds": 268435456, "miner_stripe_size": 67108864, "worker_pool_size": 25, "test_both_modes": true, "use_range_miner": true, "use_persistent_workers": false, "window_trials": N, "n_parallel": 1, "deliver_skip_range": true, "staging_dir": "/home/michael/<fresh staging>"}
```

**But the yesterday-derived launcher refuses this file.** It requires the seven `warm_start_*` keys
(`~/stage4_miner_launch.sh:100-106`). A sampled search therefore needs a launcher variant without that requirement,
a launch-path change that needs review. Or it goes through `gate12_launch.sh`'s own params, which point at the live
staging. **[decision D4]**

**What each check does with a sampled, switched-on search** (code at `00e2ee7`):

| Check | Outcome for this search |
|---|---|
| CLI and WATCHER switch validation | passes (`true`) |
| `presented_configuration` | unpinned, so it can sample. It validates the full domains `skip_min [0,10]`, `skip_max [10,250]` and `10 ≤ 250`: **passes**. It records `constant_skip_delivery: trial_range` and bounds kind `bounds` in the search space |
| Child `_step1_skip_delivery_bounds_check` | the same rule: **passes**, and hands the checked `SearchBounds` to the sampler |
| Exact pin | not applied (no pins), so Optuna samples `skip_min`, then `skip_max ∈ [max(skip_min, 10), 250]` |
| Study | fresh: tagged at trial 0. `--resume-study` with no name: **refused** for an on run. A named study must be tagged `trial_range` |
| Output root | a fresh root is marked `trial_range`; a baseline root is refused |
| Budget preflight (`_step1_budget_preflight`, `:1655-1668`) | refuses a resumed study whose COMPLETE count already meets `window_trials` |
| Strategy | only `bayesian` (TPE, `seed=None`) runs. `random`, `grid` and `evolutionary` are refused as UNSUPPORTED (M6) |
| `n_parallel > 1` | **parked by ruling** (M3). It is not refused by the code unless resuming a checkpoint, and the NP2 path loads its own bounds, not the child's checked object (`:2861`). The switch's per-trial wall at the coordinator still applies. Keep `n_parallel = 1` |
| Survivor-volume gate | **cannot be run on this search:** a range is refused (M1b). WATCHER does not call it, and nothing per trial does |
| Survivor cap at run time | **not enforced.** No runtime code reads `survivor_volume.cap`. Above it, a run continues until something else gives way; the capacity record names coordinator RAM (M1 §4.4) [UNVERIFIED for these volumes] |
| TRSE Rule A | active with today's `trse_context.json`, which narrows the window ceiling from 50 to 32 (all of today's WATCHER logs, about `:121`). The sampled window is then `[6, 32]` |

**What a sampled trial varies:**
- `window_size` `[6, 32]`;
- `offset` (anchor) `[0, 100]`;
- `session_idx` over three session sets;
- `forward_threshold` and `reverse_threshold` `[0.30, 0.75]`;
- `skip_min` `[0, 10]` and `skip_max` `[10, 250]`.

## 2. Sized options

### 2.1 Time per trial

**Measured inputs** (today, k=12, 25 workers, 4 stripes per phase on 2^28; `RESULT_SKIP_SWITCH_RIG_RUNS_20261122.md` §6):

| Arm | Hypotheses | Phase-1 stripe | Phase-2 stripe | Trial |
|---|---|---|---|---|
| B | 17 | 5.9–6.0 s | 3.7 s | 97.4 s |
| C | 33 | 6.6–6.7 s | 4.9–5.7 s | 114.5 s |
| D | 251 | 80.6–83.1 s | about 67 s | 242.9 s |

**A linear fit in the hypothesis count H, through B and D:**
- **The fit:** phase 1 ≈ 5.9 + 0.323·(H−17) s and phase 2 ≈ 3.7 + 0.27·(H−17) s, so **trial ≈ 97 + 0.59·(H−17) s**.
- **Check:** it predicts C at 107 s, against 114.5 s measured, and D at 236 s, against 243 s.
- **Status:** a fit to three single runs, not a law.

**What H a sampled trial gets.** TPE's first trials are random (startup). Uniform `skip_min ∈ [0,10]` and
`skip_max ∈ [max(skip_min,10), 250]` gives a mean H of about 126, and so **≈ 162 s per trial**. The range runs
from about 88 s (H = 1) to 243 s (H = 251).

**Not in the fit:**
- **Window size:** every measurement is at k=12, and per-hypothesis work grows with k (`k·(skip+1)` steps). A k=32
  trial could take up to about 2.7× longer in phases 1–2 [UNVERIFIED; time did not follow the step count at
  k=12].
- **Survivor-heavy trials:** the hybrid phases' time with large survivor counts is unmeasured, and so is staging
  time.
- **Launch overhead:** about 2.5 min per launch (gates, fleet, release) on top of the trials, from today's logs:
  the launch stamp to the first claim, plus publication.

### 2.2 Options

| Option | Interval | Trials | Expected wall time (both arms ×2) | Notes |
|---|---|---|---|---|
| **O1** | Stage 4, `[2^31, 2^31+2^28)` | 10 | ≈ 30 min (18–43) per arm | the smallest that explores more than TPE's random startup (`n_startup_trials` 5) |
| **O2** | Stage 4 | 25 | ≈ 70 min (40–104) per arm | TPE has 20 model-guided trials |
| **O3** | Stage 4 | 50 | ≈ 2.3 h (1.3–3.4 h) per arm | the manifest default `window_trials` |
| **OL** | `[2^31, 2^32)`, one chunk of 2^31 | 10 | ≈ 1 h per arm [model] | 32 stripes per phase over 25 workers is 2 waves, so a phase takes ≈ 2× its 2^28 time, not 8× [UNVERIFIED]. The pinned Run L found 23,568 in phase 4 on this interval, already half the cap at Run 4's thresholds |

### 2.3 Survivor-volume risk with loose thresholds

**Hybrid phases 3–4: the dominant risk, the same in both arms.** The skip range does not reach them (the hybrid
kernel takes no skip bound). From the capacity record's projection table (M1 §3.1: measured rate, projected), phase
3 per 2^31 seeds is:

| k \ τ | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.60 |
|---|---|---|---|---|---|---|
| 6 | 137,973,376 | 13,119,744 | 13,119,744 | 13,119,744 | 13,119,744 | 828,928 |
| 12 | 18,864,000 | 2,593,536 | 2,593,536 | 281,472 | 281,472 | 1,536 |
| 20 | 6,657,408 | 193,664 | 193,664 | 3,200 | 3,200 | 128 |
| 30 | 913,536 | 37,376 | 5,504 | 128 | 128 | <128 |

- **Scaled to 2^28,** the cap of 50,000 corresponds to 400,000 per 2^31.
  - `k = 12, τ = 0.30` is about **2.36M per 2^28, ≈ 47× the cap**.
  - Every `k ≤ 10` cell with `τ ≤ 0.50` is over the cap, and so is `k = 12` at `τ ≤ 0.40`.
- **A uniformly sampled startup trial lands over the cap often** [the fraction depends on the exact box; not
  computed].
- **The table is midday, anchor 75, phase 3 only.** Phase 4 was measured separately (§3.2 of that record); at today's
  pins it was 2995 on 2^28.

**Constant phases 1–2: the only volume the switch changes.** This is a model, computed here.
- **The model:** the per-seed survival probability is `1 − (1 − P(≥M of k at p = 1/1000))^H`, with `M = ⌈kτ⌉`.
- **Inputs, all assumed and none measured:** independent uniform outputs, and M taken from kτ (the float32
  boundary lead in `docs/LEADS.md` §2.36 can move M by one at exact cells).
- **Results:**
  - **At `τ = 0.30, k = 12` (M = 4):** about 2 survivors per 2^28 at H = 17, 17 at H = 126 and 33 at H = 251.
    Negligible.
  - **At `τ = 0.30, k = 6` (M = 2):** about 68k at H = 17, already over the cap with the switch off; 506k at
    H = 126 and 1.0M at H = 251. **Over the cap by up to 20×, scaled by the switch.**

**What the volume gate can do here:**
- **Per geometry only.** The gate takes exact values. It can be run on chosen corners of the box before launch, for
  example the loosest threshold at the smallest k with the widest skip. Those corners are known to fail.
- **Per sampled trial, only with new code.** That needs the estimator's planned per-trial caller: refuse or prune a
  trial whose projection exceeds the cap, before its sieve. It costs about 30–50 s of VM101 time per trial (B's
  gate took about 29 s and D's about 49 s for four phases).
- **What a sampled search can do instead:**
  - (a) run with bounds narrowed so that every corner passes the gate; a committed bounds change (M10); **or**
  - (b) build per-trial gating; **or**
  - (c) run with the box as it is and accept that a trial may exceed the cap. Nothing then stops it before RAM or
    staging does.

  **[decision D2]**

## 3. The paired switch-off arm, and what would be compared

**Why a pair is needed.** The switch's only effect is on phases 1–2 (G-ON-PAYLOAD; the hybrid payloads are
byte-identical).
- **On:** a sampled trial's phases 1–2 search its own `[skip_min, skip_max]`.
- **Off:** they search `[0, 16]`, whatever was sampled. That is the F-1 baseline.

So "what the switch changes" is defined per parameter tuple, and the two arms must try the **same tuples**.

**Today, they would not.** TPE has `seed=None` (M6) and adapts to each arm's own scores. Two searches diverge from
trial 1 and drift further after startup.

**How to make them try the same tuples** [decision D3]:
- **(a) A seeded TPE.** Expose a sampler seed. Both arms then get identical startup trials, but they diverge once
  the scores differ. That is partly like-for-like.
- **(b) A fixed trial source.** A seeded RandomSampler, or an explicit list of N tuples enqueued in both arms. The
  arms are then identical by construction, which is fully like-for-like. `OptunaRandomSearch` exists, but Step 1
  does not reach it (M6).
- **Either is a code change** on the launch path, a before-commit review class.

**The off arm:** the same search in a separate tree, root and study, with `deliver_skip_range` absent, the same
interval, trial count, trial source and `n_parallel = 1`. One publisher at a time.

**What would be compared, per tuple:**

| Field | Rule | Why |
|---|---|---|
| Phases 3–4 counts and sets, `bidirectional.variable` | must be equal | the hybrid path takes no skip bound |
| Phase 1–2 counts | superset only when the sampled range contains `[0, 16]` (`skip_min = 0`, `skip_max ≥ 16`); otherwise they may differ either way | each arm searches skips the other does not |
| NPZ rows going to Step 2, and the objective value (`BidirectionalCountScorer`) per trial | report the difference | this is what the switch actually changes downstream |
| The best trial each arm's search selects | report it | — |
| Credit, the swept intervals | must be equal | — |

- **Phases 1–2 keep no seed lists** after a run (`extract_*.log:5-6`). Seed-level comparison of the constant sets
  would need those lists recorded first [decision D5].

## 4. What each option can and cannot say about hit rate

**Can:**
- whether the switch runs correctly when sampled;
- the survivor volumes and stripe times per tuple, in both arms;
- which tuples the switch changes in phases 1–2;
- whether the objective picks different geometries.

**Cannot:** anything about pool hit rate against `k/1000`.
- Step 1's outputs are survivors and a chosen geometry. A survivor count is load, not quality (estimator docstring,
  M4).
- No option here produces a prediction pool, and none scores one on draws the sieve never saw.

**The measurement that would answer it** (not planned here):
1. For each arm, carry its Step 1 output through Steps 2–6 to prediction pools of fixed size k.
2. For a sequence of future draws, never used by any sieve or training, count `exact_hits / pool_size`
   (`CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md:417`).
3. Compare each arm to `k/1000` and to each other, with binomial intervals.

**The size it needs.** At k = 20 the chance level is 2%. Telling a 2% pool from a 3% pool (one-sided α = 0.05,
power 0.8) needs about **1,400 unseen draws** per arm. That is about two years at two draws a day.
- **Computed:** from the normal approximation.
- **Assumed:** independence between draws.
- **No Step 1 option shortens that.** Only the pool evaluation does.

## 5. Stop conditions (for whichever option is approved)

1. Every stop condition of runbooks A–D. Any gate or refusal; `Prior rows` ≠ 0; no `credited`; a rig losing cards;
   no WATCHER line for 20 min; `free -m` below 2 GiB on VM101. **Never SIGTERM a live coordinator.**
2. **Any trial's phase count above 50,000.** Stop at the end of that trial if the run can be stopped cleanly, and
   report it. With per-trial gating built, that trial is refused before its sieve instead.
3. Coordinator RSS on VM101 above 8 GiB, a safe margin on 15.5 GiB with no swap (CLAUDE.md rule 9). **Expected peak
   RSS is unknown for survivor-heavy trials;** state it before launch from the gate's corner projections.
4. A lease expiry, requeue or retry in any phase. Today's maximum was 3 renewals with no expiry.
5. A trial with `skip_min = 0, skip_max ≥ 16` whose phase 1–2 count is below the off arm's for the same tuple
   (fixed trial source only). That contradicts the superset rule.
6. Phases 3–4 differing between the arms for the same tuple.
7. The off arm's study or root showing `trial_range`, or the on arm's not showing it.
8. More than one Step 1 publisher at a time on one root, or an arm started in the other's tree.

## 6. Decisions that are the owner's (yes/no, with my recommendation)

| # | Decision | Recommendation |
|---|---|---|
| **D1** | Run a sampled switched-on search at all before the volume risk (§2.3) is handled? | **No.** Handle D2 first |
| **D2** | Narrow the tracked sampling bounds (raise `window_size.min` and/or `forward/reverse_threshold.min`) so every corner passes the gate, as a committed change? | **Yes,** as the smaller change (my recommendation). The alternative is to build per-trial gating (estimator per trial, refuse before the sieve), which is new launch-path code needing Beta review. Raising the 50,000 cap is not recommended: it is "Raised only by the owner", and RAM is the binding limit. **But narrowing the threshold bounds CHANGES A RULING.** `distributed_config.json:78` (`window_size._s172_note`) records the S172 TB ruling of 2026-04-30, which raised `window_size.min` from 2 to 6, and adds: "Threshold bounds intentionally PRESERVED so Optuna can continue optimizing across [min, max]." Raising `forward/reverse_threshold.min` reverses that decision. Raising `window_size.min` further goes past what the ruling set. **Per-trial gating changes no ruling:** the bounds stay as preserved, and a trial is refused only on its own projected volume. The conflict is recorded here, and the choice is the owner's |
| **D3** | Add a fixed trial source (a seeded RandomSampler or an enqueued tuple list) so the on and off arms try the same tuples? | **Yes.** Without it the pair compares two different searches, not the switch |
| **D4** | A launcher variant for unpinned params (yesterday's requires the seven pins), reviewed before use? | **Yes**, a one-line change to its params check, reviewed |
| **D5** | Record phase 1–2 seed lists for this search, so the constant sets can be compared seed by seed? | **No** for O1, where counts suffice at the expected volumes. Revisit if counts differ |
| **D6** | Keep `n_parallel = 1` (parallel trials stay parked, M3)? | **Yes** |
| **D7** | Which option, once D1–D4 are settled? | **O1 (10 trials, Stage 4 interval) as a pair**, then decide O2 or OL from its timings and volumes |

## 7. What I could not verify

- **Stripe time at window sizes other than 12,** and for survivor-heavy hybrid trials. §2's times are a linear fit
  to three single runs at k=12.
- **What the coordinator does with millions of survivors** in one trial on VM101: RAM, staging, finalizer. The
  capacity record computes RAM as the binding limit; it has not been measured at these volumes on the current code.
- **Whether `window_trials`' declared manifest range is enforced** on any route. It was not, on today's pinned
  route.
- **Whether `OptunaRandomSearch`** can be reached from Step 1 with small changes.
- **How TPE's startup and model-guided trials distribute** over this box in practice. §2.1's mean H assumes uniform
  startup sampling.
- **That OL's two-wave estimate holds** with 32 stripes per phase on 25 workers. It was not measured.
- **The fraction of uniformly sampled trials over the cap.** §2.3 names the cells that are over; the fraction was
  not computed.
- **Rig state.** None was read from this session.
