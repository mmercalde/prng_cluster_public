# RIG CRASH FORENSIC — 2026-08-22 phase-4 host hang

**Run of record:** `distributed_config_t1_8cb116da` ("run 3"), launched 2026-08-22 18:21:19,
trial context created 18:23:00, trial terminal 18:42:01, finalized 18:42:41.
**Comparators:** `distributed_config_t1_eed23c7f` ("run 1", 14:34:44, clean 128/128) ·
`distributed_config_t1_35ba80aa` ("run 2", 17:40:58, phase 1 only) ·
`distributed_config_t1_554463d3` (Gate-12 attempt 9, 2026-08-17, PASSED at `e9ca800`).
**Code under test:** `69ca910` (Window-Anchor Brief I). Attempt 9 ran `e9ca800`.

**Status:** OPEN. This document records measurements and eliminations. **No root cause is
claimed.** Read-only investigation; nothing committed, no fleet action.

```
Verification-integrity controls (VIR-1…6):
- execution proof:      every table below is a live query against a named ledger or log file,
                        re-runnable; the arg-tuple and kernel-constant results are executable
                        captures from two `git archive` trees, not readings of source.
- clean control:        run 1 (same binary, same generator_phase, clean 128/128) and Gate-12
                        attempt 9 (different binary, PASSED) are both carried in every table.
- fault-injection ctrl: n/a — this is a forensic reconstruction, not a detector.
- completion sentinel:  each section states its own denominator.
- unavailable-observer: CT100 memory/CPU/disk = UNAVAILABLE (all three CTs "No route to host",
                        probed 2026-08-27). Rig-side worker logs = NOT EXAMINED. Host kernel
                        ring = UNAVAILABLE (unprivileged LXC, §2.17). netconsole = EMPTY, and
                        an empty netconsole cannot distinguish "no event" from "not active".
                        None of these is reported as clean.
- audit claim scope:    what the code executes on a rig between e9ca800 and 69ca910; and what
                        the 2026-08-22 runs measurably placed on the hosts. NOT a claim about
                        why the hosts stopped.
- searched surfaces:    git (both commits, blob + AST), miner_staging/miner_ledger.db,
                        miner_ledger_archive/miner_ledger_pre_separation_9runs.db,
                        miner_forensic_20260823/ledger/miner_ledger.db,
                        logs/gate12_20260822_182119.log, logs/netconsole_*.log,
                        live VM101 /proc/meminfo and df, live CT100 SSH probe.
- unavailable surfaces: CT100 hosts (unreachable), Proxmox host consoles, rig worker logs.
- governance trail searched: TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE, TB_RULING_FIELD6_
                        IMPLEMENTATION, TB_RULING_BRIEF_I_*, PROPOSAL_WINDOW_ANCHOR_*.
- chapters searched:    CHAPTER_2 §6 (CRT lanes), not otherwise material here.
```

---

## 1. LAUNCH MECHANICS ARE EXONERATED — `e9ca800` → `69ca910`

Established at source, in both directions, for the `java_lcg_hybrid_reverse` (phase-4) path.

**Kernel bodies — byte-identical.** `prng_registry.py` is the *same git blob* at both commits
(`53dabe95eee423377e0bd964770ce6607a147432`), as is `sieve_gpu_worker.py` (`724ffbb5be…`).
Independently, resolving all 44 `KERNEL_REGISTRY` entries through their module-level
`*_KERNEL` string constants at each commit gives 44 → 44 entries, 44 distinct constants, zero
unresolved, zero differing, and an identical aggregate digest:

```
SHA256 over (name:const:sha) for all 44, key-sorted
  e9ca800  cc75bddf70dd1345cc8eaf77c4ede6785d7f51ea216a9153329e3b491f337aeb
  69ca910  cc75bddf70dd1345cc8eaf77c4ede6785d7f51ea216a9153329e3b491f337aeb
java_lcg_hybrid_reverse -> JAVA_LCG_HYBRID_REVERSE_KERNEL, 3027 bytes, sha 6936e6c47b2ffa2c…
```

This is verified against `e9ca800` directly, not against Brief I's own baseline.

**Arg construction — same tuple.** The arg list was *built by executing* `resolve_builder
('java_lcg_hybrid_reverse')` in both trees under an identical trial context. The two
14-element tuples are element-for-element identical in count, order and dtype:

```
1-7   BufferArg  seeds(uint64) residues(uint32) survivors(uint64) match_rates(float32)
                 skip_sequences(uint32) strategy_ids(uint32) survivor_count(uint32)
8-9   ScalarArg  int32 n_seeds, int32 k
10-11 BufferArg  strategy_max_misses(int32) strategy_tolerances(int32)
12-13 ScalarArg  int32 n_strategies, float32 threshold
14    ScalarArg  int32   <- the frozen trailing phase slot
```

The only difference in the whole capture is the `BuildContext` field name
(`offset` → `generator_phase`). `_hybrid_prefix`, `_constant_prefix`,
`materialize_kernel_args`, `_gpu_launch`, `resolve_builder`, `_load_strategies`,
`partition_stripe`, `select_seed_cap`, `sha256_residues` and `_best_effort_gpu_cleanup` are all
AST-identical between the commits.

**Launch pattern and concurrency — unchanged.** `miner/range_miner_worker.py:1266-1268` at HEAD
is byte-identical to `e9ca800`:

```python
threads = 256
blocks = (n_seeds + threads - 1) // threads
self._gpu_launch(kernel, blocks, threads, kernel_args)
```

One launch per sub-stripe, default stream, no explicit stream or event object anywhere on the
path. `VramCaps` is identical (`amd 2_000_000 / amd_hybrid 1_000_000 / nvidia 5_000_000 /
nvidia_hybrid 2_500_000`). Sub-stripe geometry is confirmed from the ledgers, not only the
code — `expected_substripes` per stripe:

| | phase 1/2 (constant) | phase 3/4 (hybrid) |
|---|---|---|
| attempt 9 `e9ca800` | 34 (AMD) / 14 (Zeus) | 68 (AMD) / 27 (Zeus) |
| run 3 `69ca910` | 34 / 14 | 68 / 27 |

**Allocation and memory — unchanged.** No cupy allocation, buffer, pinned-memory or transfer
call was touched; the allocation block and the `finally` teardown in `SieveExecutor.execute`
are line-for-line identical. The one memory-adjacent edit is in `load_residue_window`, where
`start = max(0, min(offset, n - window_size))` became a validated range check that raises. The
slice length is `window_size` either way, and the old clamp **never fired in production**:
`N_filtered` is 8,515 midday / 9,553 evening, so `derived_max ≥ 8,465` for any
`window_size ≤ 50`, while the anchor is bounded at 100. Same `start`, same residues, same
`residue_sha256`.

**Timing-adjacent work — measured, negligible, and it does not move the device seam.** The
Brief I guards sit between `resolve_builder` and `ResidueResolver.resolve`, i.e. **before** the
lazy `import cupy` and before `with cp.cuda.Device(...)`. The device is first touched at the
same code point. Measured on VM101, 200k iterations each:

```
execute() guard block (reject_legacy_offset_key + require_generator_phase
                       + capability + policy)          1.538 us / sub-stripe
ResidueResolver.resolve added required-key reads       1.485 us / sub-stripe
                                            TOTAL      3.023 us / sub-stripe
                                        per stripe    205.6  us  (68 sub-stripes)
                       per 32-stripe phase-4 stage      6.579 ms  (fleet-wide)
```

Against 4.5–6.7 s of worker compute per stripe that is ~5×10⁻⁵ %. The other two candidates are
not on a hot path: `MinerLedger._init_db`'s new `PRAGMA table_info(trial_context)` migration
wall runs **once**, from `MinerLedger.__init__` (`miner/range_miner_coordinator.py:1205`) — it
does *not* ride the per-query connection open that MP-1 charged for the drain starvation; and
`_pump_deferred` changed by six code lines, all in the end-of-pass field-6 instrument, none in
the per-entry loop, the memoization, the sweep or the `_admission_lock` scope.

**Conclusion for §1: nothing Brief I did can shift kernel dispatch timing, occupancy,
allocation or launch count on the rigs.**

## 2. THE ONE DEVICE-SIDE DIFFERENCE — trailing `int32` 25 → 0

The tuple is the same; **the value in slot 14 is not.** At `e9ca800` it was
`payload.get("offset", 0)` — the Optuna-sampled offset. At `69ca910` it is
`require_generator_phase(payload)`, pinned to 0 at two seams
(`build_stripe_assign_payload` raises on nonzero; `assert_generator_phase_permitted` raises on
nonzero). From `trial_context`:

| run | commit | trial | slot-14 value |
|---|---|---|---|
| attempt 9 `554463d3` | `e9ca800` | W12, midday, `offset_val=25` | **25** |
| run 1 `eed23c7f` | `69ca910` | W20, evening, `window_anchor=58` | **0** |
| run 2 `35ba80aa` | `69ca910` | W22, both, anchor 26 | **0** |
| run 3 `8cb116da` | `69ca910` | W6, midday, anchor 75 | **0** |

In `JAVA_LCG_HYBRID_REVERSE_KERNEL` that argument drives
`for (int o = 0; o < offset; o++) state = (a*state+c)&m;`, executed once per strategy. With
`n_strategies = 5` (the fixed `hybrid_strategy` presets — they do **not** derive from
`skip_min`/`skip_max`), attempt 9 ran 125 extra LCG iterations per thread and the 08-22 runs
ran none: at k=12 that is **≈0.6% less arithmetic per thread, uniform across the warp, no new
divergence**, i.e. marginally *faster*.

**⚠ COMPARABILITY CAVEAT (binding, and it is the design's own).** The pin does not merely
subtract work — it starts the generator at a different trajectory point for the same anchor,
so it **changes which seeds survive**. Per `PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_
SEPARATION_v1_1.md`, post-separation phase-zero populations are **not legitimate regression
comparators** to historical populations. Therefore: it cannot be asserted that run 3's trial
would have produced the same survivor count at `e9ca800`, and the survivor figures in §3 are
not a pre/post code comparison. They are a statement about what each run actually placed on
the hardware.

## 3. WHAT 13.1M SURVIVORS ACTUALLY MEANT ON THE HOSTS

### 3.1 The headline table

| run | phase-3 survivors | phase-4 survivors | phase-4 outcome |
|---|---:|---:|---|
| attempt 9 (`e9ca800`, PASSED) | **59** | 23,515 | 32/32 done |
| run 1 (clean) | **1** | 0 | 32/32 done |
| **run 3 (hung)** | **13,146,485** | 2,851 | 24 done / 8 cancelled |

### 3.2 Bytes serialized and staging file count (ledger `shards`, joined to `stripes`)

| run · phase | shards | survivors | staged bytes | MiB | max shard | avg shard |
|---|---:|---:|---:|---:|---:|---:|
| attempt 9 · p3 | 1,889 | 59 | 316,819 | 0.30 | 302 B | 168 B |
| attempt 9 · p4 | 1,848 | 23,515 | 1,478,862 | 1.41 | 2,293 B | 800 B |
| run 1 · p3 | 1,848 | 1 | 306,645 | 0.29 | 268 B | 166 B |
| run 1 · p4 | 1,848 | 0 | 306,544 | 0.29 | 167 B | 166 B |
| **run 3 · p3** | **2,094** | **13,146,485** | **502,221,629** | **478.96** | **603,612 B** | **239,838 B** |
| run 3 · p4 | 1,278 | 2,851 | 357,797 | 0.34 | 744 B | 280 B |

Derived:

- **Payload cost ≈ 38.2 bytes per survivor** on the wire and on disk
  (`(502,221,629 − 2,094×166) / 13,146,485`). Envelope floor is ~166 B per zero-survivor shard.
- **Run 3's phase 3 alone is 1,597× attempt 9's entire phase 3** (478.96 MiB vs 0.30 MiB) and
  **1,652× run 1's**. Whole-run staged bytes: run 3 **479.6 MiB** vs attempt 9 **2.02 MiB**
  (237×). The previous all-time high anywhere in the ledger corpus is attempt 5's phase 3 at
  **15.49 MiB** — run 3 is **31× that**.
- **Sustained aggregate delivery in phase 3: 2,094 shards over 234.2 s = 8.94 shards/s carrying
  ~2.05 MiB/s.** Attempt 9's phase 3 moved 0.30 MiB over 89.4 s ≈ 3.4 KiB/s. **~600×.**
- **Retention: 5,333 shards verified-and-enqueued, 479.6 MiB, `local_cleanup_status = none` for
  every one** — nothing was released — against a derived `required_files = 6528`
  (`preflight_plans`, `high_water_mode = derived`). That is **82% of the file ceiling**.
  **⚠ CORRECTED 2026-08-27 — the number is right, the framing was wrong.** The file-count bound
  is `stripes x sub-stripes` and carries **no survivor term**, so 82% is not a volume signal.
  Attempt 3, with **774** survivors, reached **91.9%**; attempt 9, with **59**, reached 87.2%.
  Survivor volume cannot saturate this bound. The volume-sensitive bound is
  `staging_high_water_bytes` (16 GiB), of which run 3 used **2.93%**. See
  `S172_PHASE3_SURVIVOR_CAPACITY_CHARACTERIZATION.md` §4.
  `[S172-BP] summary`: `deferred_high_water=1494` against `bound_in_force=2201` (**68%**),
  `inbound_qsize_high_water=550`, `staging_jobs_completed=5333` at `4.661/s`,
  `pause_events=0`, `capacity_timeout_terminations=0`, `capacity_invariant_terminations=0`,
  `inbound_saturation_events=0`.

### 3.3 Where the bytes actually went — the rigs wrote nothing to disk

**`remote_spool_path` is EMPTY for all 5,348 run-3 shards, and for every shard of every run in
the corpus (0 spooled, all inline).** `INLINE_BYTE_LIMIT = 48 MiB`
(`miner/range_miner_worker.py:1344`) and the largest shard ever produced is 603,612 B, so no
payload has ever taken the spool path. The route is:

```
rig worker   json.dumps(survivors)  ->  one inline frame per sub-stripe, sendall
wire         25 TCP sockets  ->  coordinator _conn_reader_loop decode
coordinator  bounded `inbound` queue -> staging executors -> 5,333 .json files on VM101
             (/home/michael/miner_staging/)
```

**Consequences, and they narrow the field sharply:**

- **Rig-side disk I/O from the 479 MiB is ZERO.** The CT100 spool directory
  (default `/dev/shm/prng/miner`, i.e. tmpfs/RAM) was never used. Any hypothesis resting on rig
  disk or rig tmpfs exhaustion is **refuted by the ledger**.
- The rig-side cost is JSON serialization plus socket writes: ~229 KB of JSON per sub-stripe,
  68 times per stripe.
- The 479 MiB lands on **VM101**, whose measured capacity is `MemTotal 15,924.8 MiB`,
  `SwapTotal 0.0 MiB` (live read, 2026-08-27), 377 GiB free on `/`.

### 3.4 Worker memory footprint — measured, and it is not the story

Peak per-worker host RSS for the survivor list is the resident sub-stripe only (one at a time,
serially). Measured by `tracemalloc` on the real tuple shape `(seed, rate, strategy_id,
list[k])`:

| shape | n | k | resident | per survivor |
|---|---:|---:|---:|---:|
| run 3, rig sub-stripe (average) | 6,029 | 6 | **1.33 MiB** | 232 B |
| run 3, largest sub-stripe (Zeus) | 15,588 | 6 | 3.30 MiB | 222 B |
| attempt 9 phase-4 sub-stripe | ~13 | 12 | ~0 | 170 B |

**≈10.6 MiB per CT100** with all 8 workers at their average sub-stripe simultaneously.
**Worker RAM exhaustion is not a viable mechanism at this volume.**

GPU memory moved the *wrong* way for a compute-stress reading: `skip_sequences_gpu` is
`n_seeds × k × 4 B`, so at the AMD hybrid cap of 1,000,000 seeds run 3 (k=6) allocated 24 MB
against attempt 9's 48 MB (k=12) — **run 3 used less VRAM, not more.**

### 3.5 What CT100 RAM and host I/O this implies — **UNAVAILABLE**

CT100 `MemTotal`, core count and filesystem headroom were **not obtained**: all three CTs
answered `No route to host` on 2026-08-27. Per VIR-5 and §2.17 this is recorded as
**UNAVAILABLE, never as clean or as "sufficient."** What §3.3/§3.4 *do* establish is that the
two candidate rig-side consumers — spool disk and worker RSS — were near zero, so the CT100
figure, when obtained, is unlikely to be the discriminator on its own.

## 4. THE REFRAMING — what the data supports, and three things it refutes

**Supported.** The stress axis unique to run 3 is **survivor serialization, delivery and
staging volume, not GPU compute load.** Phase 4 in run 3 produced only 2,851 survivors and its
completed stripes ran *faster* than attempt 9's:

| run | phase-4 avg stripe elapsed | max |
|---|---:|---:|
| attempt 9 | 21.64 s | 31.00 s |
| run 1 | 24.44 s | 33.82 s |
| run 3 | **13.61 s** | 24.54 s |

Coordinator-side, the volume is visible exactly where §3 predicts:
`msg_seconds_per_frame = 0.069153` in run 3 against attempt 9's 0.0013 s — **~50×** — while
`iteration_max` stayed at 6.83 s with `drain_stop_count_guard = 0`, i.e. **the MP-1/R-1…R-4
drain remedy held under 479 MiB and did not starve the control plane.**

**Refuted — record these, they matter more than the hypothesis they weaken.**

1. **Rig disk / tmpfs I/O: refuted.** Zero shards spooled, ever (§3.3).
2. **Per-host volume does not rank with survival.** `rrig6600b` carried the **most**
   (816 shards, 4,933,414 survivors, 179.9 MiB) and **survived**; `rrig6600c` carried the
   **least** (544 shards, 3,285,769 survivors, 121.3 MiB) and **died**. A per-host resource
   exhaustion mechanism has to explain that inversion, and none currently does.
3. **"Long idle before phase 4": refuted.** Run 3's phase-3 staging tail (last shard received
   → last shard verified) was **117.2 s**; attempt 9's was **189.9 s** — *longer*. Run 3's
   pre-phase-4 quiet period was not anomalous.

**⚠ The stated reason for demoting voltage sag does not hold, and sag is NOT demoted here.**
Two objections:

- The shorter phase-4 stripe times are measured **only on the two surviving hosts**
  (`rrig6600b`, Zeus). The dead hosts contributed no `elapsed_s`. They are not evidence about
  the load on `rrig6600` or `rrig6600c`.
- A supply sag is a **ramp/transient** phenomenon, not a sustained-load one. The relevant event
  is the current step when 24 rig GPUs simultaneously resume after ~2 minutes idle — which is
  exactly where the failure sits (§5). Short steady-state stripe times neither confirm nor
  refute it.

Sag and volume are therefore recorded as **co-leading, not ranked.**

## 5. THE FAILURE, RECONSTRUCTED FROM THE LEDGER AND THE COORDINATOR LOG

```
18:26:09 – 18:30:03   phase 3 GPU span (234.2 s). All three rigs + Zeus complete 32/32.
                      478.96 MiB / 13,146,485 survivors delivered inline.
18:30:03 – 18:32:00   staging tail; last phase-3 shard verified 18:32:00.
18:32:01              phase 4 dispatched; all 25 workers resume GPU work simultaneously.
18:32:01 – 18:32:25   rrig6600b:gpu0-7 each complete a FULL 68-sub-stripe stripe.
                      zeus completes 8 stripes (216 shards).
18:32:02 – 18:32:07   rrig6600 emits 7 shards; rrig6600c emits 8 shards.
                      ALL 15 FAIL STAGING. Both hosts then emit nothing, ever again.
18:37:01 – 18:38:31   rrig6600b + zeus take the reassigned stripes (attempt 1) and finish them.
18:42:01              [F1/F2] TRIAL TERMINAL class=compute_lease_expiry
                      stripe st3_s5, worker rrig6600:gpu4, attempt 1 — second failure,
                      hybrid retry exhausted -> trial aborted.
18:42:04              teardown: 25 READER_EXIT records.
```

**Three signatures, all first-order:**

1. **Every staging failure in the entire run — all 15 — belongs to the two hosts that died, and
   all fall inside a five-second window (18:32:02–18:32:07).** `rrig6600b` and Zeus have zero.
   The payloads were tiny (0–4 survivors, 166–373 B), so this is not a size effect.
2. **`WORKER_DISCONNECTED` count for the whole run is ZERO.** All 25 sockets remained
   ESTABLISHED from the coordinator's view for the **ten minutes** between the last byte from
   the dead hosts (18:32:07) and teardown (18:42:04). At teardown, five connections
   (`conn9`–`conn12`, `conn14`) exited `TRANSPORT_ERROR`; the other twenty exited
   `SHUTDOWN_STOP`.
   **This eliminates worker-process death.** A crashed or OOM-killed worker closes its socket
   and the coordinator sees a reasoned EOF; the Attempt-6 Part-A provenance machinery exists
   precisely to record that, and it recorded nothing. A host frozen at or below the kernel's
   TCP stack holds the socket open and silent — which is what was observed.
3. **The event is synchronised across two independent hosts.** Both went silent within five
   seconds of each other, 1–6 s after the fleet-wide phase-4 resume. Simultaneity across
   independent machines is better explained by a **shared** resource — power, switch,
   coordinator, hypervisor host — than by per-host exhaustion, which would be expected to track
   per-host volume and demonstrably does not (§4 refutation 2).

## 6. PREMISE CORRECTIONS

- **`rrig6600c` did NOT claim zero phase-4 stripes.** It produced **8 phase-4 shards at attempt
  0 between 18:32:05 and 18:32:07** (`rrig6600c:gpu0,2,3,4,5,6,7`). The `stripes` table shows no
  `rrig6600c` in `claimed_by` for phase 4 only because those stripes were **reassigned at
  attempt 1** and `claimed_by` was overwritten (`st3_s16`→`rrig6600:gpu2`,
  `st3_s17`→`rrig6600:gpu0`, `st3_s18/20/21/23/24`→`rrig6600b`). `rrig6600c` was alive and
  executing phase 4 for roughly two seconds before it stopped. The constraint is therefore
  *stronger* than "it never got work": **it died mid-stripe, in lockstep with `rrig6600`.**
- **The two hosts that stopped are `rrig6600` and `rrig6600c`.** `rrig6600b` completed all its
  phase-4 work and picked up seven reassigned stripes afterwards.

## 7. HYPOTHESIS REGISTER

| # | hypothesis | status | what would settle it |
|---|---|---|---|
| H1 | Brief I changed device execution / launch timing | **ELIMINATED for launch mechanics** (§1). One value changed (§2), incapable of altering dispatch, occupancy or allocation. | — |
| H2 | Shared-circuit voltage sag under the synchronised 24-GPU phase-4 resume | **OPEN, co-leading.** Not demoted; the demotion argument is refuted in §4. Fits: synchronicity, sub-TCP freeze, no worker-side error. Does not by itself explain the 15 staging failures preceding the silence. | Per-rig instrumented supply logging across a phase boundary; circuit map for the three rigs. |
| H3 | Host resource exhaustion / I/O–memory pressure under 13.1M-survivor staging | **OPEN, co-leading, NEW.** Fits: run 3 is 31× any prior staged volume and the only run ever to approach the retention ceiling. Constrained by: rig disk I/O = zero (§3.3), worker RSS ≈ 10.6 MiB/CT (§3.4), and per-host volume inverted against survival (§4). | CT100 `MemTotal`/`MemAvailable`/`nproc`/`df`, currently **UNAVAILABLE**; a per-host resource series sampled across the phase-3→4 boundary. |
| H4 | Worker-process death or OOM-kill | **ELIMINATED.** Zero `WORKER_DISCONNECTED`; sockets held open and silent for 10 minutes (§5.2). | — |
| H5 | Network partition / switch event | **OPEN, under-weighted so far.** Would produce the same sub-TCP silence and would explain synchronicity across two hosts. | Switch counters/logs; `rrig6600` and `rrig6600c` port topology vs `rrig6600b`. |
| H6 | Boot-selector confusion (rig returned as Ubuntu) | **NOT APPLICABLE to the 18:32 event** — the hosts were running as Proxmox and serving GPU work seconds earlier. Retained only as the standing rule (§6 of the skill) that a rebooted rig is not a recovered rig. | — |

## 8. WHAT THE NEXT EXPERIMENT MUST BE DESIGNED AROUND

**The number is 13,146,485 survivors / 478.96 MiB / 2,094 shards in one phase — and it is
reachable from the ordinary Optuna search space.** Run 3 sampled `window_size = 6` with
`forward_threshold = 0.35`: a six-draw window needing three CRT matches. Nothing bounded it.
`window_size = 6` was equally reachable at `e9ca800`; no bound moved in Brief I.

Design targets, in priority order:

1. **Reproduce the volume deliberately and instrument the hosts**, rather than waiting for
   Optuna to sample it again. A short-window / low-threshold phase-3 configuration is the
   forcing function. It needs owner authorization and a live fleet; it is not proposed here.
2. **Sample per-host resource series across the phase-3 → phase-4 boundary** (`MemAvailable`,
   load, socket counts, `rocm-smi` per-GPU state) at ≥1 Hz, on all three rigs, armed before
   dispatch — the §2.28 sampler-ordering lesson applies verbatim.
3. **Instrument the phase transition itself**, since the failure is 1–6 s after a synchronised
   resume, not during sustained load.
4. **Explain the 15 staging failures.** They are the earliest observable, they are perfectly
   correlated with the two dead hosts, and the `shards` table carries no error column for them.
   That is an observability gap worth closing before the next run.
5. **Consider a governed bound on reachable phase-3 survivor volume** — not proposed, not
   designed; recorded because run 3 reached 82% of the derived retention ceiling by ordinary
   sampling, and the next such trial has no headroom.

## 9. INCIDENTAL — field-6 falsifiers are populated in a production artifact

Run 3's `[S172-BP] summary` carries `deferred_distinct_attempts_high_water=27` and
`pump_liveness_probes_high_water=118` — integers, not the `UNOBSERVED` sentinel. The `d8b21e3`
repair works as specified.

**This is recorded for Beta, not claimed as discharging the obligation.** §2.49's mandated
phrasing governs statements about R-3's scaling model until the falsifiers are observed in the
**Phase-7 soak**, and run 3 is a Gate-12-shaped run, not the soak. Separately,
`deferred_distinct_attempts_high_water = 27` exceeds the 25-worker frozen cohort; §2.40 records
that this quantity "REFUTES the guarantee outright if it exceeds the cohort." Whether retained
frames from reassigned attempts legitimately push the distinct-key count above the live cohort
is **not resolved here** and is flagged as requiring interpretation before anyone reads 27 as
either a refutation or a benign artifact.

---

# CORRECTION ADDENDUM — 2026-08-28

**Appended, not merged.** The original text above is unchanged and is left standing as written on
2026-08-22. This addendum corrects **one line** of it. Everything else in the document, including
its audit-claim scope and every other unavailable-observer entry, is unaffected.

## The corrected line

The unavailable-observer block states:

> `netconsole = EMPTY, and an empty netconsole cannot distinguish "no event" from "not active".`

**That is incorrect as to "EMPTY."** `logs/netconsole_all_rigs.log` contains **11 packets dated
2026-08-22**, from all three Proxmox hosts:

```
19:25:50  .121/.155/.163   NC-TEST3 $(hostname) $(date +%T)
19:26:09  .121             NCPROOF-pve-rig6600
19:26:11  .155             NCPROOF-pve-rig6600b
19:26:14  .163             NCPROOF-pve-rig6600c
19:52:59  .155             watchdog: watchdog0: watchdog did not stop!   (x2)
19:52:59  .155             systemd-shutdown[1]: Failed to finalize DM devices, ignoring.
20:33:53  .121             systemd-shutdown[1]: Failed to finalize DM devices, ignoring.
20:33:57  .163             systemd-shutdown[1]: Failed to finalize DM devices, ignoring.
```

## What the packets are

**All 11 are post-incident operator activity.** Per the operator (2026-08-28): he always shuts the
rigs down after a crash and did so on the evening of 2026-08-22. The `NC-TEST3`/`NCPROOF` pair is
his arm-verification test; the `systemd-shutdown` lines are that cleanup. The `watchdog did not
stop!` line on `.155` belongs to an orderly systemd shutdown path and is **not** a fault
indication.

They are **not** evidence about the incident, and nothing in the original document's incident
analysis changes because of them.

## What this corrects, precisely

The original line offered a disjunction it could not resolve. The `NCPROOF` packets resolve one
half of it:

| branch | disposition |
|---|---|
| *"not active"* | **CLOSED.** The senders were provably armed and delivering on all three hosts. |
| *"no event"* during the freeze itself | **STANDS**, and is now the correct and only reading. |

**The corrected line should be read as:** *netconsole was ARMED and delivering (proven by
`NCPROOF` on all three hosts at 19:26); it captured no packet during the freeze window; the 11
packets it did capture that day are post-incident operator activity.*

## What this does NOT change

- **The 2026-08-22 mid-run freeze remains UNDETERMINED.** An armed-but-silent netconsole is
  evidence that no kernel-level message reached the wire — **not** evidence of a healthy host.
- The document's audit-claim scope ("NOT a claim about why the hosts stopped") is unaffected.
- Every other unavailable-observer entry — CT100 memory/CPU/disk, rig-side worker logs, host
  kernel ring under unprivileged LXC — stands exactly as written.
- No conclusion, exoneration or root cause anywhere in the original document is altered.

## Freeze window, tightened

A by-product of the correction, recorded as a bound and not as a hypothesis. Kernel monotonic
uptime stamps in the capture give implied boot times of 18:52:41 (`.155`), 18:58:18 (`.163`) and
18:58:55 (`.121`) — all **after** the run log's last write at 18:42:04 — and uptime runs
continuous from the 19:25 arming to each shutdown (wall-vs-uptime drift <0.07 s on all three, so
no second reboot in that window). The hosts therefore booted post-incident, were armed at 19:25,
and were shut down deliberately.

**The freeze is bounded to 18:42-18:52 and left no netconsole trace.** Cause not investigated
here, not closed, no hypothesis offered.

*Full lead register entry: `docs/LEADS.md` L-3. Re-arm procedure: `docs/RUNBOOK_NETCONSOLE_REARM.md`.*
