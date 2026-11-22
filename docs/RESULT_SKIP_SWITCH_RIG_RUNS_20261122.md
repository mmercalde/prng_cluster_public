# RESULT: RANGE-MINER constant-skip delivery switch, first rig runs A, B, C, D (2026-11-22)

A plain record of what ran, what it showed, and what is open.
- **Written by:** Claude (Team Alpha) in `/home/michael/tfm_worktrees/pwc-offset`. Draft, uncommitted.
- **Sources:** the four run trees, `~/distributed_prng_analysis/logs/skipsw_session/`,
  `data/gate12_evidence/parity_00e2ee7_20261122.json` and the lead's run-D blocks in `/tmp/skd/`, all read read-only.
  The run trees are `~/skipsw_run_off`, `~/skipsw_run_on`, `~/skipsw_run_on32` and `~/skipsw_run_on250`.
- **(lead-reported):** items the Alpha lead reported that this session could not read.
- **No verdicts.** Measured facts, their anchors and their limits only.

**Null result first.** In all four runs, constant phases 1–2 found **0 survivors**. A ran with the switch off, B with
it on at `[0, 16]`, C at `[0, 32]` and D at `[0, 250]`. Every compared output was the same in all four:
- phase rows `0 / 0 / 13 / 2995`;
- an empty bidirectional set;
- a 0-row NPZ with artifact `aecc5d504784ecca…`, identical to yesterday's.

**No survivor population, and so no downstream input, was changed by the switch at these ranges.** That is a
statement about one interval, `[2^31, 2^31 + 2^28)`, at one geometry.

**Hit rate was not measured.** Nothing here says anything about pool hit rate against `k/1000` on unseen draws, which
is how TFM is judged. These runs tested an engineering property: that the switch is inert when off, and that it
delivers the requested range when on.

## Prior record

| # | Command (worktree `pwc-offset`; `SC` = the standard scope of CLAUDE.md "Prior record first") | Returned |
|---|---|---|
| W1 | `/bin/grep -l -i -E "deliver_skip_range\|constant_skip_delivery\|skip.switch" $SC` | no hits: the scope holds no earlier record of the switch's rig behaviour. **NEW** |
| W2 | `git show main:docs/LEADS.md \| /bin/grep -n -i skip` | `:74`, the float32 threshold-boundary lead (constant path included). It records that `k=12, τ=0.71` is **not** on an affected boundary. ALREADY RECORDED; not engaged by these runs |
| W3 | `/bin/grep -rl -i -E "\[0, ?32\]\|skip 0-32\|0\.\.32" $SC docs/exchange` | no hits. **NEW** |
| W4 | `/bin/grep -l -i -E "stripe time\|per-stripe\|seconds per stripe\|kernel time" $SC` | `docs/CHAPTER_2_SOURCE_MAP_v1.md:180` (per-stripe effective values, not timing). No earlier `[0, 16]`, `[0, 32]` or `[0, 250]` stripe-time measurement. **NEW** |

**Governing, ALREADY RECORDED, and read:**
- the switch's plan and landing record (`docs/drafts/miner_skip_switch_v1/PLAN.md`, `READY_TO_LAND.md`);
- the Beta threads X-SKIP-F1, X-MINER-SKIP-SWITCH R1–R3 and X-MINER-SKIP-SWITCH-2 R1–R3 (`docs/exchange/LEDGER.md`);
- yesterday's result (`docs/RESULT_PWC_BACKUP_STAGE4_20261121.md` §2, §4).

**Not searched:** the host, the ser8 archives, the private repository, the rigs.

## 1. What was run, and from what

- **Code:**
  - landed `f916532` (09:30, 47 files, `owner_land`) and `00e2ee7` (09:34, the coordinator and the WATCHER manifest,
    hand-landed), per `git log` in the main tree;
  - the four core files at `00e2ee7` hash as recorded before landing: coordinator `b57f3c5a…`, integration
    `5ec53da4…`, bayesian `3cf0f0f1…`, search ledger `3d7e8f63…`.
- **Rig parity:** `data/gate12_evidence/parity_00e2ee7_20261122.json` has 30 rows, all `MATCH`. That is 10 governed
  files on `rrig6600`, `rrig6600b` and `rrig6600c`, collected 17:35:55Z. `miner/range_miner_coordinator.py` is
  `b57f3c5a…` on all three.
- **Runbooks:** `/tmp/skipsw_runbook.md` (A, B) and `/tmp/skipsw_runbook_c.md` (C). The lead ran them "with no
  deviation" (lead-reported).
- **Run D's blocks are not a separate runbook.** The lead derived them from runbook C by `sed`: `on32` → `on250`, skip
  max `32` → `250`, `c_vs_a` → `d_vs_a`. They are in `/tmp/skd/b01.sh`…`b18.sh`.
  - **Checked here:** each of the 18 blocks equals the matching block of `/tmp/skipsw_runbook_c.md` after those
    substitutions.
  - **Four blocks differ further**, and only in their remaining literals:
    - `b06`: the params assertion `== 250`;
    - `b09`: a comment's `[0,250]`;
    - `b10`: the search reason, "run D … skip [0,250] … owner 2026-11-22 11:03";
    - `b16`: the rule `warm_start_skip_max is 250`.
  - The rule script keeps runbook C's labels ("C"), so in D's rule file "C" means D.
- **Shared geometry:** `java_lcg`, both modes, interval `[2147483648, 2415919104)`, window 12, offset 25, session idx 1
  (`["midday"]`), thresholds 0.71/0.47, `window_trials` 1, `n_parallel` 1, `worker_pool_size` 25, stripe
  `67108864`.
- **What varies:**

  | Arm | Switch | Skip |
  |---|---|---|
  | A | off (key absent) | `[0, 16]` |
  | B | on | `[0, 16]` |
  | C | on | `[0, 32]` |
  | D | on | `[0, 250]` |

- Each arm had its own run tree at `00e2ee7`, its own staging directory, its own `prng_analysis.db` and its own one
  search.

## 2. Run A: switch off (`~/skipsw_run_off`, WATCHER log `logs/gate12_20261122_100017.log`)

- **Search:** `s1s-d85eabde…` (`bundle_off/bundle.json` `search_id`). Miner run `distributed_config_t1_2e5ab19a`.
- **Credit and publication** (log lines, in order):
  - `:278` credited, trials `[0]`, every required phase over `[2147483648, 2415919104]`;
  - `:280` `GENERATION PUBLISHED 0 rows`, `Prior rows: 0`;
  - `:290` certified, coverage `2a463179b32419cc`;
  - `:291` linked.
- **Generation:** `gen-20261122T180353276401Z-…`.
- **No `[SKIP-DELIVERY]` line in the log.**
- **No marker:** `.s172_accumulator/skip_delivery_arm` is absent in `~/skipsw_run_off`.
- **Phase rows:** `0 / 0 / 13 / 2995`; credit granted, swept `[2147483648, 2415919104]` in all four phases; artifact
  `aecc5d504784ecca` (`logs/skipsw_session/extract_off.log:2-4`).

## 3. Run B: switch on, `[0, 16]` (`~/skipsw_run_on`, `logs/gate12_20261122_100717.log`)

- **Search:** `s1s-463e03fa…`, opened with `constant_skip_delivery = trial_range` (lead-reported, runbook P8). Miner
  run `distributed_config_t1_c03d9a72`.
- **Switch lines:**
  - `:86` `[SKIP-DELIVERY] ON: RANGE-MINER phases 1-2 sieve each trial's own [skip_min, skip_max]`;
  - `:96` the root-arm banner;
  - `:125` `[SKIP-DELIVERY] ON: phases 1-2 sieve skip [0, 16] (baseline [0, 16])`.
- **Marker:** `~/skipsw_run_on/.s172_accumulator/skip_delivery_arm` reads `trial_range`.
- **Credit and publication:**
  - `:283` credited;
  - `:285` published 0 rows, `Prior rows: 0`;
  - `:295` certified, coverage `078c5a7e815326ea`;
  - `:296` linked.
- **Phase rows:** `0 / 0 / 13 / 2995`; artifact `aecc5d504784ecca` (`extract_on.log:2-4`).
- **Skip evidence:** the bundle provenance records `constant_skip_range_requested [0, 16]` and `effective [0, 16]`
  (`bundle_on/bundle.json`).
- **Survivor-volume gate:** `PROCEED` under the 50,000 cap. Phase 1 projection 0 (95% `[0.0, 14.8]`, 3.7 s); phase 2
  0 (3.4 s); phase 3 24.0 (6 sampled hits, 8.1 s); phase 4 2995 (13.5 s). Source:
  `~/skipsw_run_on/logs/stage4/volume_gate.log:6-10`.

## 4. Run C: switch on, `[0, 32]` (`~/skipsw_run_on32`, `logs/gate12_20261122_104258.log`)

- **Search:** `s1s-7e904b16…`. Miner run `distributed_config_t1_d660c7e5`.
- **Switch lines:**
  - `:96` the banner;
  - `:125` `[SKIP-DELIVERY] ON: phases 1-2 sieve skip [0, 32] (baseline [0, 16])`;
  - `:126` the WARNING `33 skip hypotheses per seed against the baseline's 17 (1.94x)`.
- **Marker:** reads `trial_range`.
- **Credit and publication:**
  - `:285` credited;
  - `:287` published 0 rows, `Prior rows: 0`;
  - `:297` certified, coverage `86a29519ee69ac36`;
  - `:298` linked.
- **Phase rows:** `0 / 0 / 13 / 2995`; artifact `aecc5d504784ecca` (`extract_on32.log:2-4`).
- **Skip evidence:** the bundle provenance records `constant_skip_range_requested [0, 32]` and `effective [0, 32]`.
- **Survivor-volume gate:** `PROCEED`. Phase 1 0 (3.9 s); phase 2 0 (3.7 s); phase 3 4.0 (**1** sampled hit, 8.3 s);
  phase 4 2995 (14.7 s). Source: `~/skipsw_run_on32/logs/stage4/volume_gate.log:6-10`.
  - **The phase-3 sampled hit count differs between B's gate (6) and C's (1).** Phase 3 is hybrid, so the trial's
    skip range does not reach it; the real runs' phase 3 was 13 in both B and C.
  - Why the gate's sample differs is not established. One possibility is that the sampled block depends on the
    geometry digest, which includes `skip_max` [UNVERIFIED].
- **No lease expiry, requeue or retry:** none of these strings appears in any of the three logs (searched for
  `lease expir`, `LEASE_EXPIRED`, `requeue`, `retry`).
- **After the run:** no workers left, and the live state unchanged (lead-reported).

## 4a. Run D: switch on, `[0, 250]` (`~/skipsw_run_on250`, `logs/gate12_20261122_110738.log`)

- **Owner:** 11:03, "proceed-before you run is there any monitoring that we need to be concerned about ?" (§ Owner's
  words). The monitoring state the lead checked in answer is §7a.
- **Params:** `~/skipsw_run_on250/logs/stage4/params.json` is C's params with `"warm_start_skip_max": 250` and
  staging `/home/michael/skipsw_staging_on250`.
- **Search:** `s1s-01be63b666fe…`. Miner run `distributed_config_t1_0a3aa6e6`.
- **Switch lines** (`/tmp/skd/runD_post.log:1-4`, quoting the WATCHER log):
  - `:96` the banner;
  - `:125` `[SKIP-DELIVERY] ON: phases 1-2 sieve skip [0, 250] (baseline [0, 16])`;
  - `:126` the WARNING `251 skip hypotheses per seed against the baseline's 17 (14.76x)`.
- **Marker:** reads `trial_range`. Arm A's tree still has no marker.
- **Credit and publication:**
  - `:310` credited;
  - `:312` published 0 rows;
  - `:317` `Prior rows: 0`;
  - `:322` certified, coverage `4b71dcc31478d010`;
  - `:323` linked.
  - The search status afterwards is `search_complete` (`runD_post.log:36-44`).
- **Phase rows:** `0 / 0 / 13 / 2995`; artifact `aecc5d504784ecca`; generation `gen-20261122T191333731207Z-…`
  (`extract_on250.log:2-4`).
- **Receipt:** `receipt_on250.json`, sha256 `a71d51b7…`, lag 7.8 s (`runD_post.log:52-55`).
- **Survivor-volume gate:** `PROCEED`. Phase 1 0 (95% `[0.0, 14.8]`, **19.0 s**); phase 2 0 (**8.7 s**); phase 3
  12.0 (3 sampled hits, 8.0 s); phase 4 2995 (13.4 s). Source: `~/skipsw_run_on250/logs/stage4/volume_gate.log:6-10`.
  The sampled phase-3 hits are 6, 1 and 3 for B, C and D; see §4.
- **No lease expiry:** none of `lease expir`, `EXPIRED`, `requeue`, `retry`, `GRACE` or `Traceback` appears in the log.
- **No workers left:** all three rigs `no workers` (`runD_post.log:46-51`).
- **Comparator:** `NOT COMPARABLE`, 2 DIFFERENT / 51 EQUAL / 10 NOT COMPARED. The only differing inputs are
  `param.deliver_skip_range` and `param.warm_start_skip_max` 16/250 (`compare_on250_vs_off.txt:4`, `:12`, `:70`).
- **D vs A rule check:** `ALL RULES HOLD` (`rules_on250_vs_off.txt:1-18`). Phases 3–4, the variable set and credit
  are equal; phase 1–2 counts and NPZ rows are 0 ≥ 0; the NPZ is bit-equal at 0 rows.
- **Lead-reported, not readable from this session:**
  - `LIVE UNCHANGED`;
  - all rigs at 8/8 cards with 0 fault-pattern kernel lines after the run.

## 5. The comparisons

**The engine comparator gives `NOT COMPARABLE` in all four pairs, by design.** It refuses any pair whose inputs differ
(`scripts/compare_engine_outputs.py:455-456`).

| Pair | DIFFERENT | EQUAL | NOT COMPARED | The only differing inputs | File |
|---|---|---|---|---|---|
| A vs yesterday | 1 | 51 | 10 | `run.repository_commit`: `6eafad9…` / `00e2ee7…` | `compare_off_vs_yesterday.txt:17`, `:69` |
| B vs A | 1 | 52 | 10 | `param.deliver_skip_range`, in B only | `compare_on_vs_off.txt:4`, `:69` |
| C vs A | 2 | 51 | 10 | `param.deliver_skip_range`; `param.warm_start_skip_max` 16 / 32 | `compare_on32_vs_off.txt:4`, `:12`, `:70` |
| D vs A | 2 | 51 | 10 | `param.deliver_skip_range`; `param.warm_start_skip_max` 16 / 250 | `compare_on250_vs_off.txt:4`, `:12`, `:70` |

In each pair the comparator's own rows for the four `phase_counts.*.survivor_rows` are EQUAL.

**Our output checks.** These are written for these runs, outside the repository (`/home/michael/skipsw_outputs_equal.py`,
`/home/michael/skipsw_outputs_c_vs_a.py`). **They are not the comparator.**
- **A vs yesterday:** `OUTPUTS EQUAL` (`equal_off_vs_yesterday.txt:1-9`). Inputs differ only in
  `run.repository_commit`. Phases, bidirectional, phase counts and credit are equal, and the NPZ is equal (22
  arrays, 0 rows).
- **B vs A:** `OUTPUTS EQUAL` (`equal_on_vs_off.txt:1-9`). Inputs differ only in `param.deliver_skip_range`.
- **C vs A:** `ALL RULES HOLD` (`rules_on32_vs_off.txt:1-18`).
  - Inputs differ in exactly the two expected keys.
  - Phases 3–4, `bidirectional.variable` and credit are equal.
  - Phase 1–2 counts C ≥ A (0 ≥ 0). NPZ rows C ≥ A (0, 0), bit-equal at 0 rows.
- **What the checks could not test:** phases 1–2 keep no seed lists in either bundle (`phases.1/2` null;
  `extract_*.log:5-6`). So C's "superset of A" was checked by count only, and with both counts 0 it is trivially
  true.

## 6. Phase timings: `[0, 32]` and `[0, 250]` against `[0, 16]`, measured on the rigs

From the `[S172-BP] burst_exact stage=N` line (stage start) and each stripe's `[H1H2] stripe_rx` line
(`disposition: stripe_complete`).
- **Phase 1:** the coordinator's `age_since_claim_s`, the time from claim to completion.
- **Phases 2–4:** `age_since_claim_s` is `null`, so the figures are seconds from the stage's `burst_exact` line to
  each stripe's completion.

There are 4 stripes per phase and one run per arm.

| Phase | A `[0,16]` off | B `[0,16]` on | C `[0,32]` on | D `[0,250]` on | C / B | D / B | Log lines (A; B; C; D) |
|---|---|---|---|---|---|---|---|
| 1 `java_lcg` | 9.14–9.20 s | 5.85–6.02 s | 6.57–6.69 s | **80.62–83.08 s** | ≈ 1.12 | ≈ 13.8 | `:160-163`; `:162-165`; `:162-165`; `:177-181` |
| 2 `java_lcg_reverse` | 3.62–3.69 s | 3.66–3.70 s | 4.86–5.66 s | **66.41–67.56 s** | ≈ 1.33–1.53 | ≈ 18 | `:168-171`; `:170-173`; `:171-175`; `:198-201` |
| 3 `java_lcg_hybrid` | 20.7–30.1 s | 19.9–28.4 s | 24.6–31.0 s | 27.7–33.9 s | (not skip-dependent) | (not skip-dependent) | `:179-184`; `:182-186`; `:185-188`; `:210-215` |
| 4 `java_lcg_hybrid_reverse` | 9.8–10.7 s | 9.0–10.5 s | 9.4–10.9 s | 11.5–13.3 s | (not skip-dependent) | (not skip-dependent) | `:192-195`; `:194-200`; `:197-201`; `:223-227` |

**Total trial time** (bundle `duration_s`): A 102.6 s, B 97.4 s, C 114.5 s, D 242.9 s.

**VM101 gate sampling seconds** for the same 2^26 seeds, which is not a rig measurement:

| Phase | B | C | D |
|---|---|---|---|
| 1 | 3.7 s | 3.9 s | 19.0 s |
| 2 | 3.4 s | 3.7 s | 8.7 s |

**Heartbeat lease renewals per stripe** (`heartbeats_renewed` in each `stripe_rx` row):

| Arm | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|---|---|---|---|---|
| A, B | 0 | 0 | 1 | 0 |
| C | 0 | 0 | 1–2 | 1 |
| D | **2** | **3** | 1 | 1 |

**D is the first observation of lease renewal in phases 1–2.** Every D phase 1 and phase 2 stripe was renewed two or
three times. None expired, none was requeued, and the longest stripe, 83.1 s, is about 28% of the 300 s
`compute_lease_timeout`.

**What D adds about scaling, as measured facts only:**
- **Phase 1, D/B ≈ 13.8×.** That is close to the hypothesis ratio, 251/17 = 14.76.
- **It is far from the forward kernel's LCG-step ratio.** By `12 + 13·skip` per skip that is 410,887 against 1,972
  steps per seed, ≈ 208×, the figure this session's runbook C would have projected. The measured times do not follow
  that step count.
- **Whether the kernel the worker actually executes is the one read** (`prng_registry.py` `JAVA_LCG_KERNEL`) was not
  established [UNVERIFIED]. Nor was what accounts for the gap.
- **C/B (≈ 1.12)** lies below both ratios. At small ranges, fixed per-stripe costs plausibly dominate, but that is not
  established.

**Limits of these numbers:**
- **Small and unrepeated:** 4 stripes per phase, one run per arm.
- **The noise is larger than the switch's effect:** the two `[0, 16]` runs differ from each other by ≈ 1.5× in
  phase 1 (A 9.2 s, B 6.0 s). That is a larger gap than B-to-C. A was the first run after the fleet launch, and
  whether that explains it is not established [UNVERIFIED].
- **Wall-clock stripe times, not kernel times:** they include transfer, staging and protocol work. How much of a
  stripe is the skip loop was not measured.
- **The projection did not hold.** This session's runbook projected ≈ 3.68× for phases 1–2, from the kernel's step
  count (`12 + 13·skip` per skip: 7,260 against 1,972 steps per seed). The measured rig wall-clock ratio was ≈ 1.1×
  for phase 1 and ≈ 1.3–1.5× for phase 2. The step count is a code reading; the stripe time is what was measured.
  The difference between them is unexplained.
- **Heartbeats:** in A, B and C no phase 1–2 stripe was renewed. Every one finished within one 30 s heartbeat
  interval. **D supplies the first phase 1–2 renewals** (see above).
- **D's phase-2 figures are measured from the stage start, not from claim.** Three renewals in about 67 s suggests the
  claim, or the first heartbeat, came before the `burst_exact` line; this is not established.

## 7. Limits (all of them)

1. **The comparator is `NOT COMPARABLE` by design** for every pair here (§5). The equality statements come from our
   own output checks, which are stricter than the comparator on nulls and are not reviewed by Beta.
2. **Phases 1–2 found 0 survivors in A, B, C and D.** So:
   - the delivered range changed no output;
   - the superset rule and the constant bidirectional set were exercised only at 0;
   - nothing in these runs tests the switch's effect on a non-empty constant population.
3. **No hit-rate evidence of any kind.**
4. **One interval, one geometry, one family, one run per arm.**
5. **Wide ranges, partly measured now.**
   - **Measured:** D ran `[0, 250]`, which contains every skip of `[6, 99]`. The survivor-volume gate said PROCEED
     (phase 1–2 projections 0), and the real run found 0.
   - **Still unmeasured:**
     - `[6, 99]` itself;
     - any range or interval where phases 1–2 produce survivors, which is when staging volume and the 50,000 cap
       would actually be exercised;
     - phase 1–2 stripe times above 83 s;
     - lease behaviour beyond three renewals per stripe;
     - any rig not in the three-rig, 25-worker shape used here.
6. **The phase-2 reverse kernel's loop** was not read (C runbook §6).
7. **Lead-reported, not read here:**
   - that the lead ran with no deviation;
   - C's no-workers-left check;
   - C's and D's `LIVE UNCHANGED` output (A's `live_state_before.txt` was read; neither final `diff` was saved to a
     file);
   - D's post-run rig state: 8/8 cards, 0 fault-pattern kernel lines;
   - the 11:03 monitoring state (§7a).

## 7a. Monitoring state before run D (lead-checked at 11:03; lead-reported, not readable from this session)

- **Telemetry:** inactive.
- **Polling:** no polling process on the hosts or the CT100s.
- **netconsole:** not loaded.
- **Kernel console level:** 7, set by the bring-up `dmesg -n 7`.
- **The launcher's concurrency sampler:** not run. This one is consistent with the launcher, which prints
  `concurrency sampler: NOT RUN for Stage 4` (`~/stage4_miner_launch.sh:505`, from which every arm's launcher was
  derived).
- **What this means for the timings:** none of §6's numbers was taken with telemetry, polling or a sampler running.
  It also means no rig-side fault capture beyond the kernel console was active during D.

## 8. Open (not decisions, not verdicts)

- **A non-empty constant population:**
  - whether a geometry or interval with known phase 1–2 survivors should be run with the switch on and off. That
    would test the superset rule and the delivered range on real rows;
  - phases 1–2 keep no seed lists (`extract_*.log:5-6`), so a seed-level superset check would need those lists
    recorded.
- **Timing:**
  - **CLOSED by run D:** whether to measure phase 1–2 stripe time at a range long enough to need heartbeat renewal.
    D measured 80.6–83.1 s (phase 1) and about 67 s (phase 2), with two or three renewals per stripe and no expiry
    (§6).
  - the A/B phase-1 gap (9.2 against 6.0 s, same range) is unexplained;
  - why phase-1 time scales with the hypothesis count (D/B ≈ 13.8, against 14.76) and not with the read kernel's LCG
    step count (≈ 208×) is open. Which kernel source the worker runs was not read.
- **The volume gate's phase-3 sample** varies between geometries that differ only in skip (§4).
- **Cleanup:** the four run trees and the four staging directories stay as evidence. Their removal is the owner's
  call.

## Owner's words today (2026-11-22, PST, verbatim as relayed to this session)

| Time | Subject | Words |
|---|---|---|
| 08:28 | rig-6600 resets closed: failed PSU breakout board | commit `d58e120` (owner's record; no quote relayed) |
| 08:34 | D6.2 §4.3 amendment: `constant_skip_delivery` joins the run-context digest when on | "i approve as long as it doesnt break,confuse or in any other way slow down development" |
| 08:37 | one more Beta round on the SW2 close-out | "sure" |
| 09:03 | fix the X-MINER-SKIP-SWITCH-2 R2 Q4 LIVE objection and send R3 | "sounds like the fix is required- lets get it doen and on to testing. please" |
| 09:28 | landing | "you are lead developer- if this is what gets us to testing- yes" |
| 09:46 | rig runs A (off) and B (on, `[0, 16]`) | "approved-into the breech once more" |
| 10:30 | run C | "run c at your descretion" |
| 11:03 | run D (switch on, `[0, 250]`) | "proceed-before you run is there any monitoring that we need to be concerned about ?" |
| 11:18 | planning the multi-trial switched search (planning only, no run) | "please proceed" |

Earlier, 2026-11-21 21:05, on opening X-MINER-SKIP-SWITCH-2: "Whichever makes the most sense. Remember the rigs are
off".

The Alpha lead accepted the X-MINER-SKIP-SWITCH-2 R3 review despite its relay `BOUNDARY VIOLATION`, whose cause is
recorded in `docs/drafts/miner_skip_switch_v1/READY_TO_LAND.md`. That was the lead's decision, not an owner ruling.
