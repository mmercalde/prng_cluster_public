# S172 — HYBRID SURVIVOR-VOLUME CAPACITY CHARACTERIZATION

*(Phases 3 and 4. Originally phase-3 only; the reverse-hybrid half was added 2026-08-27 — §3.2.)*

**Scope.** What survivor volume the Optuna-reachable search space can produce, why, and which
cluster resource binds first. **Read-only; no fleet.** All GPU measurement was one local RTX 3080
Ti on VM101 through the production `SieveExecutor.execute` path — no coordinator, no rigs, no
pipeline. Nothing committed.

**⚠ JUSTIFICATION NOTE.** Survivor volume is **not** a crash hypothesis. `RIG_CRASH_FORENSIC_
20260822.md` refuted the rig-side mechanisms: nothing has ever been spooled to rig disk
(`INLINE_BYTE_LIMIT` 48 MiB vs a 603 KB largest-ever shard), worker RSS is ~10.6 MiB per CT100,
and the load correlation inverted — `rrig6600b` carried the most volume and survived,
`rrig6600c` the least and died. This document stands on its own merits: run 3 produced 1,597×
attempt 9's phase-3 survivors and 31× the previous all-time high, and the cluster ceiling in that
regime was uncharacterized.

**⚠ CORRECTION TO `RIG_CRASH_FORENSIC_20260822.md` §3.2.** That section reported run 3 at "82% of
the file ceiling" inside a volume discussion. The number is right; the framing was wrong, and §4
below shows why. **The file-count retention bound is survivor-independent.** Attempt 3, with
**774** survivors, reached **91.9%** — higher than run 3's 81.9% at 13.1 million.

```
Verification-integrity controls (VIR-1…6):
- execution proof:     every survivor count below was produced by the PRODUCTION kernel
                       (`_get_kernel('java_lcg_hybrid')`) through the production
                       `SieveExecutor.execute`, on residue windows whose `sha256_residues`
                       digests MATCH the ledger `trial_context.residue_sha256`.
- clean control:       six historical geometries reproduced from their own ledger rows.
- audit claim scope:   phase-3 (`java_lcg_hybrid`, forward) survivor volume and the resources it
                       consumes. NOT a claim about why two hosts stopped on 2026-08-22.
- searched surfaces:   prng_registry.py, miner/range_miner_worker.py,
                       miner/range_miner_coordinator.py, sieve_gpu_worker.py, hybrid_strategy.py,
                       distributed_config.json, agent_manifests/window_optimizer.json,
                       three ledgers, logs/gate12_20260822_182119.log, live VM101 /proc/meminfo.
- unavailable surfaces: CT100 hosts (unreachable 2026-08-27); rig-side worker logs.
- MEASURED vs COMPUTED: every table is labelled. See §6.
```

---

## 1. EVERY GEOMETRY EVER RUN, AND ITS PHASE-3 SURVIVOR COUNT — **MEASURED**

Twelve trials exist across the three ledgers. Six reached phase 3. `M` is the **effective**
integer match requirement (§2); `residue_sha` verified against the ledger for every reconstructed
window.

| run | date | k (`window_size`) | anchor | sessions | τ (`forward_threshold`) | **M** | skip | max phase | **phase-3 survivors** |
|---|---|---:|---:|---|---:|---:|---|---:|---:|
| `689f3cd9` att 1 | 08-09 | 20 | 44 | evening | 0.45 | 10 | 2–66 | 2 | — |
| `abc63f71` att 2 | 08-10 | 13 | 69 | evening | 0.47 | 7 | 9–84 | 3 | **44,331** |
| `d606edbe` att 3 | 08-10 | 22 | 0 | evening | 0.47 | 11 | 7–229 | 4 | **774** |
| `c8939b64` att 4 | 08-11 | 15 | 71 | evening | 0.31 | 5 | 0–26 | 2 | — |
| `7e0d020b` att 5 | 08-12 | 12 | 49 | evening | 0.46 | 6 | 10–55 | 3 | **276,439** |
| `db0393b0` att 6 | 08-15 | 8 | 13 | midday+evening | 0.39 | 4 | 0–49 | 2 | — |
| `36bf30e3` att 7 | 08-16 | 29 | 49 | midday | 0.39 | 12 | 3–228 | 2 | — |
| `5c010902` att 8 | 08-16 | 15 | 82 | midday+evening | 0.60 | 9 | 3–204 | 2 | — |
| **`554463d3` att 9** | 08-17 | 12 | 25 | midday | 0.71 | 9 | 6–99 | 4 | **59** |
| `eed23c7f` run 1 | 08-22 | 20 | 58 | evening | 0.64 | 13 | 5–175 | 4 | **1** |
| `35ba80aa` run 2 | 08-22 | 22 | 26 | midday+evening | 0.69 | 16 | 5–218 | 1 | — |
| **`8cb116da` run 3** | 08-22 | 6 | 75 | midday | 0.35 | **3** | 10–98 | 4 | **13,146,485** |

Sorted by M, the six phase-3 points are monotone and span seven orders of magnitude:

```
M=3  (k=6,  τ=0.35)   13,146,485
M=6  (k=12, τ=0.46)      276,439
M=7  (k=13, τ=0.47)       44,331
M=9  (k=12, τ=0.71)           59
M=11 (k=22, τ=0.47)          774      <- larger k lifts C(k,M); see §2
M=13 (k=20, τ=0.64)            1
```

**`skip_min`/`skip_max` do not appear** because they cannot: the hybrid kernel's tolerances come
from the five fixed `hybrid_strategy` presets, never from the sampled bounds
(`miner/range_miner_worker.py:1098-1109`, `_load_strategies`, and the assign payload carries no
`strategies` key). That is §2.7 instance 4, visible here as data.

## 2. WHY k=6 / τ=0.35 YIELDS 13.1M — DERIVED FROM THE KERNEL

Source: `JAVA_LCG_HYBRID_KERNEL` in `prng_registry.py` (forward hybrid; **materially different
from the reverse hybrid** — do not reason about phase 3 from the phase-4 kernel).

**(a) Per draw the kernel scans a WINDOW of candidate skips, not one.**

```c
int search_min = (expected_skip > skip_tolerance) ? (expected_skip - skip_tolerance) : 0;
int search_max = expected_skip + skip_tolerance;
for (int test_skip = search_min; test_skip <= search_max; test_skip++) { ... }
```

Each `test_skip` produces a distinct output tested against `residues[draw]` under the CRT triple,
which is **exactly equivalent to `% 1000`** (`CHAPTER_2_BIDIRECTIONAL_SIEVE.md` §6). So per draw,
`P(match) = 1 − (1 − 1/1000)^W` where `W = search_max − search_min + 1`. `expected_skip` starts at
5 and **adapts to the found skip** (`expected_skip = test_skip`), so W runs from 11 (tolerance 5)
to 101 (tolerance 50, adapted). `best_match_rate` is the **max over all five strategies**, so the
widest preset dominates. Fitting the six historical points gives an effective **q ≈ 0.072**.

**(b) Survival requires an ABSOLUTE match count, and that is the whole story.**

```c
float match_rate = (float)matches / k;
...
if (best_match_rate >= threshold)
```

so a seed survives iff `matches ≥ M`, where **M is the smallest integer with `m/k ≥ τ`** — and
the survivor rate is `≈ P(Bin(k, q) ≥ M) ≈ C(k,M)·q^M`.

**The count is exponential in M, the absolute match requirement — not in the ratio τ, and not in
k.** Measured proof (§3): at k=6, thresholds 0.35, 0.40, 0.45 and 0.50 all produce **byte-identical
survivor counts (102,498 in a 2²⁴ sample)** because all four map to M=3; 0.70 and 0.75 likewise
coincide at M=5. τ is not the control variable. **M is.**

**(c) The arithmetic of the 13.1M.**

| | k | τ | M | q^M | measured survivors |
|---|---:|---:|---:|---:|---:|
| run 3 | 6 | 0.35 | **3** | 3.7×10⁻⁴ | 13,146,485 |
| attempt 9 | 12 | 0.71 | **9** | 5.2×10⁻¹¹ | 59 |

`q^3 / q^9 = q^{-6} ≈ 7.2×10⁶`, tempered by `C(6,3)/C(12,9) = 20/220 = 0.091` → predicted ratio
≈ 6.5×10⁵ against an observed 2.2×10⁵. Same order across a 220,000× gap.

**Answer to "the CRT match requirement, the window length, or their interaction":** the
interaction, expressed as `M = ⌈kτ⌉`. Each additional required match costs a factor of `1/q ≈ 14`.
A short window is dangerous **specifically because it makes M small** — k=6 with τ=0.35 needs only
three matches, the lowest M ever run.

**(d) One quantization seam found while validating this.** The kernel's `>=` is float32; the host
post-filter `if rate >= hybrid_threshold` (`range_miner_worker.py:1281`) compares the
float32-widened rate against a **float64** threshold. When `kτ` is an exact integer `m` and `m/k`
is not exactly representable in float32, the host **rejects** what the kernel admitted and the
effective requirement becomes `m+1`. Confirmed by histogram in three grid cells (k=20/τ=0.35:
`7/20 = 0.34999999404 < 0.35`, so M=8 not 7, and 9,575 kernel-admitted seeds became 1,513;
k=20/τ=0.45; k=10/τ=0.70). Harmless in effect, but it means **the effective M is set by the host,
not the kernel.** Related to but distinct from the open §2.36 Optuna raw→canonical item — that one
is on threshold provenance, this one is on the survivor filter. Recorded, not actioned.

## 3. IS IT PREDICTABLE? — YES ANALYTICALLY IN STRUCTURE, EXACTLY BY A 7-SECOND PROBE

**The closed form is a scaling law, not a capacity bound.** `r ≈ C(k,M)·q^M` with q ≈ 0.072
tracks the six historical points within a factor of ~2.5 over eight orders of magnitude. That is
enough to reason with and **not** enough to size a cluster from.

**Exact prediction is cheap and already done.** Running the production kernel over a 2²⁶ sample
(1/32 of the domain, one macro-stripe) on a single GPU reproduced every observed count:

| geometry | sample hits / 2²⁶ | projected @2³¹ | **observed** | ratio | seconds |
|---|---:|---:|---:|---:|---:|
| run 3 (k=6, τ=0.35) | 409,568 | 13,106,176 | 13,146,485 | 0.997 | 7.4 |
| att 5 (k=12, τ=0.46) | 8,708 | 278,656 | 276,439 | 1.008 | 4.9 |
| att 2 (k=13, τ=0.47) | 1,359 | 43,488 | 44,331 | 0.981 | 5.3 |
| att 3 (k=22, τ=0.47) | 22 | 704 | 774 | 0.910 | 9.4 |
| att 9 (k=12, τ=0.71) | 3 | 96 | 59 | Poisson-consistent | 5.0 |
| run 1 (k=20, τ=0.64) | 0 | <32 | 1 | consistent | 8.4 |

**This is the cheapest empirical probe, and it needs no fleet:** one GPU, one process,
`SieveExecutor.execute` against a stub resolver, 5–10 s per geometry. It is exact because it *is*
the production kernel, not a model of it.

### 3.1 Predicted-count table across the Optuna-reachable space — **MEASURED rate, COMPUTED projection**

Bounds are `window_size ∈ [6,50]`, `forward_threshold ∈ [0.30, 0.75]`
(`distributed_config.json:search_bounds`). Sample 2²⁴ per cell, scaled ×128 to 2³¹; midday,
anchor 75. `M` is effective (§2d).

| k \ τ | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.60 | 0.70 | 0.75 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **6** | **137,973,376** | 13,119,744 | 13,119,744 | 13,119,744 | 13,119,744 | 828,928 | 31,872 | 31,872 |
| **8** | 33,417,088 | 33,417,088 | 3,365,376 | 3,365,376 | 3,365,376 | 241,024 | 12,288 | 12,288 |
| **10** | 64,692,864 | 8,972,544 | 8,972,544 | 956,032 | 956,032 | 71,808 | 128 | 128 |
| **12** | 18,864,000 | 2,593,536 | 2,593,536 | 281,472 | 281,472 | 1,536 | <128 | <128 |
| **16** | 11,034,112 | 1,838,848 | 241,664 | 25,088 | 25,088 | 256 | <128 | <128 |
| **20** | 6,657,408 | 193,664 | 193,664 | 3,200 | 3,200 | 128 | <128 | <128 |
| **30** | 913,536 | 37,376 | 5,504 | 128 | 128 | <128 | <128 | <128 |
| **50** | 22,784 | 128 | <128 | <128 | <128 | <128 | <128 | <128 |

**The maximum of the reachable space is 137,973,376 at (k=6, τ=0.30), and that maximum is
provable, not merely observed.** M=1 would require `kτ ≤ 1`, but `min kτ = 6 × 0.30 = 1.8 > 1`, so
**M ≥ 2 everywhere in the box**, and M=2 requires `k ≤ 2/0.30 = 6.67`, i.e. **k = 6 alone**. There
is exactly one M=2 cell and it is the global maximum. *This is precisely what the TB ruling that
raised `window_size.min` from 2 to 6 purchased — the `_s172_note` in `distributed_config.json`
records W=2/3 giving 39%/53% survival by chance alone.*

**The reachable maximum is 10.5× run 3.** Run 3 was not near the top of the space.

### 3.2 PHASE 4 — `java_lcg_hybrid_reverse` — **MEASURED**

The reverse hybrid is a **different kernel**, not a mirrored one, and it is systematically less
permissive. From `JAVA_LCG_HYBRID_REVERSE_KERNEL` versus `JAVA_LCG_HYBRID_KERNEL`:

| | forward (phase 3) | reverse (phase 4) |
|---|---|---|
| candidate window per draw | `[expected_skip-T, expected_skip+T]`, **up to 2T+1** | `[0, T]`, **exactly T+1** |
| `expected_skip` adaptation | yes (`expected_skip = test_skip`) | **none** |
| state after a missed draw | advanced by `search_max` | **restored** to `state_backup` |
| miss cutoff | `consecutive_misses >= max_misses` | `> max_misses` |
| a strategy that trips the cutoff | still contributes its partial `match_rate` | **contributes nothing** (`if (!failed)`) |
| across the 5 strategies | **max** of all five | **first** that passes, then `return` |

Validation against the four historical phase-4 counts (2²⁴ sample, real residue windows,
digest-verified):

| run | k | τ_rev | M | projected @2³¹ | **observed** | ratio |
|---|---:|---:|---:|---:|---:|---:|
| att 9 `554463d3` | 12 | 0.47 | 6 | 22,656 | 23,515 | 0.963 |
| run 3 `8cb116da` | 6 | 0.68 | 5 | 3,712 | 3,801 * | 0.977 |
| att 3 `d606edbe` | 22 | 0.47 | 11 | <128 | 6 | consistent |
| run 1 `eed23c7f` | 20 | 0.57 | 12 | <128 | 0 | consistent |

*\* run 3's phase 4 completed 24 of 32 stripes; its raw 2,851 is scaled ×32/24 for comparison.*

**Phase-4 predicted counts across the reachable space** (same method, midday, anchor 75; τ here is
`reverse_threshold`):

| k \ τ | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.60 | 0.70 | 0.75 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **6** | **70,432,256** | 4,774,144 | 4,774,144 | 4,774,144 | 4,774,144 | 180,352 | 3,712 | 3,712 |
| **8** | 12,379,136 | 12,379,136 | 780,416 | 780,416 | 780,416 | 31,360 | 256 | 256 |
| **10** | 24,681,216 | 2,174,080 | 2,174,080 | 137,472 | 137,472 | 6,144 | <128 | <128 |
| **12** | 4,762,112 | 385,536 | 385,536 | 24,320 | 24,320 | <128 | <128 | <128 |
| **16** | 1,842,560 | 171,520 | 11,008 | 768 | 768 | <128 | <128 | <128 |
| **20** | 705,152 | 6,528 | 6,528 | 128 | 128 | <128 | <128 | <128 |
| **30** | 21,760 | 128 | <128 | <128 | <128 | <128 | <128 | <128 |
| **50** | <128 | <128 | <128 | <128 | <128 | <128 | <128 | <128 |

**Phase 4 is 0.1×–0.5× phase 3 at the same (k, M), and the gap widens with M** — 0.51× at M=2,
0.36× at M=3, 0.22× at M=4, 0.12× at M=5 (k=6). The narrower candidate window and the `!failed`
gate are the mechanism.

**Consequence for capacity: phase 3 is the binding phase.** Its maximum (137,973,376) is ~2× the
reverse maximum (70,432,256), and the two phases retain simultaneously, so a worst-case trial
stages both. **Run 3 died in phase 4, but the volume it had already delivered was phase 3's.**

---

## 4. WHAT SATURATES RETENTION — **COMPUTED**, and the answer is "nothing survivor-driven"

There are **three** distinct staging bounds and only one is volume-sensitive.

### 4.1 File-count bound (6,528, derived) — survivor-INDEPENDENT

`staging_high_water_files = None` in the manifest ⇒ derived. `preflight_plans` for run 3 records
`required_files = 6528 = 1088 + 1088 + 2176 + 2176`, i.e. `32 stripes × 34` (constant) and
`32 × 68` (hybrid) per stage. **That is a count of sub-stripes. It has no survivor term.**

Measured file counts against the bound, across every run:

| run | phase-3 survivors | shard files | % of 6,528 |
|---|---:|---:|---:|
| `d606edbe` att 3 | **774** | 5,999 | **91.9%** |
| `554463d3` att 9 | **59** | 5,693 | 87.2% |
| `eed23c7f` run 1 | **1** | 5,632 | 86.3% |
| `8cb116da` run 3 | **13,146,485** | 5,348 | 81.9% |
| `7e0d020b` att 5 | 276,439 | 3,968 | 60.8% |
| `35ba80aa` run 2 | — | 968 | 14.8% |

Survivor counts spanning 1 → 13,146,485 produce file counts spanning 5,632 → 5,348 — **inverted,
and driven entirely by how many stages completed.** **No survivor count saturates this bound.**

### 4.2 Byte bound (16 GiB) — volume-sensitive, and NOT reachable

`staging_high_water_bytes = 17,179,869,184` (`agent_manifests/window_optimizer.json`). Measured
payload cost from the run-3 ledger: **38.17 B per survivor** at k=6, plus a ~166 B envelope per
file.

```
saturating survivor count = (17,179,869,184 - 6,528x166) / 38.17 = 450,059,878 survivors
run 3 actual staged        =        502,925,197 B  =  2.93% of the bound
M=2 corner, BOTH hybrid phases = 2 x 137,973,376 x 38.17
                           =     10,532,887,524 B  = 61.3% of the bound
```

Saturation needs `r = 0.2096` per phase; the **provable maximum reachable r is 0.0642** (§3.1).
**The byte bound cannot be saturated from inside the Optuna-reachable space — peak utilisation is
~61%.** (k=6 is also the worst case on bytes: per-survivor cost rises ≈ `18 + 3.35k` B, fitted to
four ledger points, but survivor count falls far faster.)

### 4.3 Deferred bound (2,201) — capped by geometry

`[S172-BP] derived_bound … bound=2201` for the hybrid stages (`burst_conservative 2176 +
resume_margin 25`). Run 3 peaked at `deferred_high_water=1494` (67.9%), `pause_events=0`,
`capacity_invariant_terminations=0`. A trial cannot defer more frames than exist
(`32 × 68 = 2,176`), so volume can raise *occupancy* but cannot breach it by construction.

### 4.4 **The binding constraint is coordinator RAM — an ALLOCATION limit, not retention** — COMPUTED

`miner/range_miner_coordinator.py:9602`:

```python
inbound: "_queue.Queue" = _queue.Queue(maxsize=1024)
```

The reader decodes each frame and puts the **decoded message object** on this queue
(`:11284`). **The queue is bounded by COUNT and unbounded in BYTES**, and the byte weight of one
entry is linear in the survivor rate. Measured expansion, `json.loads` of a real payload:
**6.2× over the wire, ≈271 B per survivor decoded** (237 B when anchored to the ledger's measured
38.17 B wire cost).

```
decoded bytes with a full queue = 1024 x seed_cap x r x B_per_survivor
VM101 MemTotal = 15,924.8 MiB  (live read)   SwapTotal = 0

                       r         full queue        at run-3 occupancy (550/1024)
run 3       0.00612    1.4 - 1.6 GiB     0.74 - 0.85 GiB
M=2 corner  0.06425   14.5 - 16.6 GiB    7.8  - 8.9  GiB

r_crit (full queue == MemTotal) = 0.0602 - 0.0688
                                = 129.2M - 147.8M survivors @2^31
max reachable r                 = 0.0642 = 138.0M survivors
```

**The reachable maximum straddles the computed exhaustion point.** At 271 B/survivor the corner
*exceeds* it; at 237 B it sits at 93% of it. Even at run 3's own observed queue occupancy — 550 of
1,024, not a full queue — the corner puts **7.8–8.9 GiB of decoded envelopes in one Python queue**
on a 15.55 GiB box with **zero swap**, before counting the staging executors, the Step-1 host
process (which holds cupy), or anything else resident.

**So: retention capacity is not the binding constraint, and the cluster limit in this regime is an
arithmetic result — but it is a memory limit on VM101, not a retention limit.** The mechanism is
that the only admission control on ingress counts frames and ignores their size.

**⚠ AND IT IS AN ALLOCATION LIMIT, NOT AN ARCHITECTURAL ONE (owner correction, 2026-08-27).**
VM101 is a Proxmox VM. Its memory is configurable and **the Zeus host has 64 GB**. Every figure
above is a property of the *current allocation* of 15,924.8 MiB. **Raising the allocation moves
the ceiling above the reachable maximum:** roughly **17 GiB** clears a full queue at
`r_max = 0.0642`, and ~32 GiB leaves 2× headroom — both well inside 64 GB. Two consequences:

- **The allocation must not be raised before run 4.** Run 4 is a Brief I acceptance run; changing
  VM101's memory beforehand introduces a second variable into it. **Wait until Brief I closes.**
- **`SwapTotal = 0` is what makes the failure mode hard.** Adding swap converts an immediate OOM
  kill into degradation that telemetry could catch. That is a separate change from allocation
  size, with its own trade-off, and it is **not proposed here**.

**The design observation survives the correction, and it is the part for Beta.** A queue bounded
by count and unbounded in bytes is worth naming even when the box can be made bigger: **ingress
admission control cannot see volume at all**, so the safe operating region is a function of an
ungoverned resource (host RAM allocation) rather than of any governed bound. Every volume-aware
bound in the system — files, bytes, deferred frames — sits *downstream* of the one place that has
none. Raising the allocation moves the ceiling; it does not give the system a way to know where
the ceiling is. **Flagged, not fixed.** Registered as `LEADS.md` L-2 and `BACKLOG.md` §20.

**Assumptions, stated plainly.** (i) The queue reaches high occupancy — run 3 reached 53.7% at
1/10th the volume, and higher volume makes filling *more* likely, not less, since staging slows as
payloads grow; a run that never fills the queue never reaches this ceiling. (ii) `seed_cap` 1e6
(AMD); the Zeus worker's 2.5e6 cap makes its frames 2.5× heavier. (iii) 237–271 B/survivor is the
survivor list alone; the enclosing message adds more. (iv) `MemTotal`, not available memory.
**This ceiling has never been observed. It is computed.**

## 5. LAST KNOWN-GOOD OPERATIONAL POINT — **MEASURED**, stated conservatively

| claim | status |
|---|---|
| **59 phase-3 survivors** (attempt 9) completed all four stages **and publication**, 128/128, zero lease expiries | **MEASURED — the highest-confidence known-good point** |
| **1 phase-3 survivor** (run 1) completed all four stages and the run, on the current binary | **MEASURED** |
| **774 phase-3 survivors** (attempt 3) completed **all four GPU stages**; the run then failed at publication on the unrelated dirty-tree admission defect | **MEASURED — the largest volume that has ever completed four GPU stages** |
| 44,331 (attempt 2) and 276,439 (attempt 5) completed **phase 3 only**; both died at the stage 3→4 admission boundary for separately diagnosed reasons | **MEASURED — not evidence for or against volume** |
| 13,146,485 (run 3) completed phase 3; two hosts stopped 1–6 s into phase 4 | **MEASURED** |

**⚠ REFINEMENT — 774 is the phase-3 maximum, not the any-phase maximum.** Attempt 9 carried
**23,515 survivors through PHASE 4** in the same run that passed Gate 12 and published, and run 3
carried 3,801 (scaled) through phase 4 before it died. The per-phase picture:

| run | phase-3 | phase-4 | outcome |
|---|---:|---:|---|
| att 9 `554463d3` | 59 | **23,515** | four stages **+ publication**, 128/128 |
| att 3 `d606edbe` | **774** | 6 | four GPU stages; failed later at publication (dirty tree) |
| run 1 `eed23c7f` | 1 | 0 | four stages, complete |

**The validated envelope, stated correctly:**
- **≤ 23,515 survivors in any single phase**, demonstrated end-to-end through publication
  (attempt 9, phase 4);
- **≤ 774 in phase 3 specifically**, demonstrated through four GPU stages (attempt 3);
- **≤ 2.02 MiB total staged bytes** across a whole run (attempt 9).

Everything above those either failed for a separately diagnosed reason (attempts 2 and 5, both at
the stage 3→4 admission boundary) or is run 3. **There is a gap of three orders of magnitude
between the largest validated single-phase volume (23,515) and run 3's phase 3 (13.1M) that no run
occupies.**

## 6. MEASURED vs COMPUTED — the register

**MEASURED** (production kernel or live system, this session):
- all survivor counts in §1, §3, §3.1 and §3.2 — production `java_lcg_hybrid` and
  `java_lcg_hybrid_reverse` kernels, residue windows digest-verified against the ledger;
- the phase-4 validation set: four historical counts reproduced at 0.963x / 0.977x / consistent /
  consistent (§3.2);
- the M-collapse (identical counts across thresholds sharing an M) and the float32 host-filter
  seam — histogrammed;
- 38.17 B/survivor on the wire (run-3 ledger); 6.2× decode expansion and 271 B/survivor decoded
  (`tracemalloc`); file counts and `[S172-BP]` series (ledgers + run-3 log);
- VM101 `MemTotal 15,924.8 MiB`, `SwapTotal 0`; `inbound` maxsize 1024 (source).

**COMPUTED, never validated:**
- the 137,973,376 reachable maximum — a **measured rate** scaled from a 2²⁴ sample to 2³¹;
- 450,059,878 survivors to saturate the 16 GiB byte bound;
- `r_crit = 0.0602–0.0688` (129.2M–147.8M survivors) for coordinator memory exhaustion **at the
  CURRENT VM101 allocation**, under the four assumptions in §4.4 — an allocation limit, not an
  architectural one (§4.4);
- the ~17 GiB allocation that would clear the reachable maximum, and the ~32 GiB that would leave
  2x headroom.

**None of the computed ceilings is a validated ceiling, and none should be quoted as one.** The
distance between the last four-stage success (774) and the computed ceiling (~10⁸) is five orders
of magnitude of untested space.

## 7. WHAT THIS DOES AND DOES NOT LICENSE

**No crash-seeking ladder is designed or proposed here, and none should be run.** The ceiling is
computable, so a ladder would walk into a failure it can already predict.

**Both of the ceilings originally proposed as candidates are withdrawn on this evidence** — the
file-count retention bound (survivor-independent, §4.1) and the byte retention bound (peaks at
~61% utilisation, §4.2). What replaces them is an allocation-dependent memory limit (§4.4) that
can be moved, and a design observation that cannot (§4.4, `LEADS.md` L-2).

If validation is later wanted, the shape the arithmetic points at is **one** point, well below the
computed limit, phase-3 only, with per-host and coordinator resource sampling armed before
dispatch — and the §2.28 sampler-ordering rule applies verbatim. **That is a description of a
shape, not a proposal, and it needs owner authorization and a Beta ruling like any other run.**

The cheap, fleet-free follow-ups that need neither:
1. ✅ **DONE 2026-08-27 — §3.2.** The phase-4 probe was run; the reverse hybrid is 0.1x-0.5x the
   forward at the same (k, M), and phase 3 is confirmed as the binding phase.
2. Decide whether the count-bounded, byte-unbounded `inbound` queue (§4.4) is a defect worth a
   brief. It is the only resource with no volume-aware admission control anywhere on the path.
3. Record the float32/float64 host-filter seam (§2d) alongside the open §2.36 quantization item.

## 8. RUN-4 ENVELOPE — a region, not a number

**The envelope criterion is `predicted count ≤ 23,515` in BOTH hybrid phases** — the largest
single-phase volume ever carried through a complete certified run (attempt 9, phase 4, §5).
Applied to the measured grids of §3.1 and §3.2:

| k | min `forward_threshold` | forward count there | min `reverse_threshold` | reverse count there |
|---:|---:|---:|---:|---:|
| **6** | **NONE — k=6 exceeds the envelope at every reachable τ** | (min 31,872 at τ=0.70) | 0.70 | 3,712 |
| **8** | 0.70 | 12,288 | 0.70 | 256 |
| **10** | 0.70 | 128 | 0.60 | 6,144 |
| **12** | 0.60 | 1,536 | 0.60 | <128 |
| **16** | 0.60 | 256 | 0.40 | 11,008 |
| **20** | 0.45 | 3,200 | 0.35 | 6,528 |
| **30** | 0.40 | 5,504 | 0.30 | 21,760 |
| **50** | 0.30 | 22,784 | 0.30 | <128 |

**Three properties of this region worth stating:**

1. **`window_size = 6` has no safe threshold.** Its minimum reachable forward count is 31,872, at
   `τ = 0.75`, the top of the search range. The whole k=6 column is outside the envelope.
   k=8 is inside only at `τ ≥ 0.70`.
2. **Forward binds, not reverse.** Phase 4 is 0.1×–0.5× phase 3 at the same (k, M), so the
   forward threshold is the controlling parameter — but both must be checked, because the two are
   sampled independently and attempt 9 is the proof: `τ_fwd = 0.71` gave 59 while `τ_rev = 0.47`
   gave 23,515 in the same trial.
3. **55% of the measured forward cells (35 of 64) fall outside the envelope.** **Run 4 therefore
   cannot be a free Optuna sample.** A single unconstrained draw has a better-than-even chance of
   landing outside the validated region, and one draw is all a `window_trials = 1` run gets. The
   geometry must be **pinned**, not sampled — and pinning is a launch-parameter decision, not a
   code change.

### The geometry I would pick, and why

**Attempt 9's own: `window_size = 12`, `window_anchor = 25`, sessions `["midday"]`,
`forward_threshold = 0.71`, `reverse_threshold = 0.47`, skip `6–99`.**

Re-predicted at `generator_phase = 0` on attempt 9's real residue window (digest-verified):

```
phase 3   k=12  tau_fwd=0.71  M=9    ~96 survivors @2^31      (attempt 9 observed 59)
phase 4   k=12  tau_rev=0.47  M=6    ~22,656 survivors        (attempt 9 observed 23,515)
total hybrid survivors  ~22,750      staged bytes ~1.25 MiB
peak inbound queue at that rate      ~2.8 MiB   (vs 15,924.8 MiB MemTotal)
```

**Why this one and not a lower-volume cell:**

- It is **inside the validated envelope by construction** — it *is* the point that validated it.
- It makes run 4 a **direct A/B against the only certified Gate-12 pass**, isolating Brief I as
  the one changed variable. Every other candidate geometry adds a second.
- It still produces a **non-degenerate survivor population** (~22,750). Several envelope-safe
  cells predict `<128` or `0`, which would exercise the phase-4 delivery path with nothing in it
  and prove very little about the thing that broke.
- Its resource footprint is three orders of magnitude below every computed ceiling in §4.

**The one thing this does NOT reproduce, and it must be said:** `generator_phase` is pinned to 0
where attempt 9 ran 25, so the survivor *population* is a different set of seeds even though the
*counts* land within a few percent. Per the design's own comparability caveat
(`PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md`), post-separation phase-zero
populations are **not** legitimate regression comparators to historical ones. Run 4 can validate
**volume, plumbing and completion**; it cannot validate **population equivalence**, and no run can.

**This is a recommendation, not an authorization.** Launching is Michael's, and the geometry pin
is a launch-parameter choice that Beta may want to rule on.
