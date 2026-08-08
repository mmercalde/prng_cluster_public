# CLAUDE CODE REPORT — WHAT IS STEP 1 *FOR*? (read-only, historical)

**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_STEP1_PURPOSE_LINEAGE.md`
**Host:** VM101, `~/distributed_prng_analysis`, HEAD `8bbe79e` (working tree dirty per the
brief's own git status; no file in the tree was modified by this pass other than this report).
**Date:** 2026-08-08. **Read-only:** nothing launched, no port bound, no commit, no production
edit. Search order followed as bound: governance trail → chapters → code.

```
Verification-integrity controls (VIR-1…6):
- execution proof:        every quote re-read from the live file this session; every DB figure
                          from a read-only SQLite connection opened this session
- clean control:          n/a (no detector built)
- fault-injection:        n/a (no detector built)
- completion sentinel:    §7 "END OF REPORT"
- unavailable-observer:   §6 records the one surface I could not reach
- audit claim scope:      Step-1 stated purpose, its lineage across PWC and RANGE-MINER, and
                          the existence of a regime-change re-run mechanism
- searched surfaces:      docs/PROJECT_FILE_CATALOG.md (index, read first) · governance trail
                          (TB_RULING_*, TB_RULING_REQUEST_*, PROPOSAL_*, TEAM_ALPHA_*) ·
                          CHAPTER_1 / CHAPTER_2 / CHAPTER_13 (+ v1_1) / CHAPTER_5 / CHAPTER_6 ·
                          the whitepaper · docs/instructions.txt + Cluster_operating_manual.txt
                          (pre-repository operating manual) · SESSION_CHANGELOG_* ·
                          full git history incl. `git show <initial>:<file>`, `git log --follow`,
                          `git log -S` · live source (window_optimizer*, agents/watcher_agent.py,
                          agents/watcher_dispatch.py, chapter_13_*, miner/*, PWC, finalizer) ·
                          live gitignored config (distributed_config.json,
                          agent_manifests/window_optimizer.json) via python json, not grep ·
                          read-only prng_analysis.db · on-disk artifacts incl. out-of-repo
                          generation roots
- unavailable surfaces:   ser8 pre-repository archives (not reachable from VM101 this session);
                          the rigs (not queried — read-only brief, nothing rig-side was claimed)
- governance trail searched: YES — see §1, §2, §4, §5
- chapters searched:      YES — 1, 2, 13 (both revisions), 5, 6, 11, 14
```

**Standing caution honoured (skill §1.2):** every dated artifact quoted below is quoted for what
it *decided* or *stated at the time*, and every present-tense claim carries its own live anchor
obtained this session.

---

## 0. Answer in one page

**Step 1 was never, in any source, a per-draw seed-discovery step.** Not in the earliest
operating manual, not in the whitepaper, not in the chapters, not in any TB ruling, not in the
live manifest, and not in the code. It is a **global, monotonically-advancing sweep of a
contiguous seed range for one PRNG family**, whose *declared* deliverable is an **optimal window
configuration** and whose *actually consequential* deliverable — the certified 22-array survivor
generation — was never added to the manifest that declares Step 1's outputs.

The owner's working description is right about the shape of the flow and right that a
regime-change re-run exists. It is wrong on two mechanical points, and both matter for a seed
geometry decision:

| owner's clause | verdict |
|---|---|
| "For each draw, Step 1 goes through the space" | **Not per-draw.** Global, keyed on `prng_type` alone, frontier advances across runs and never rewinds. **No evidence found** for per-draw intent in any source. |
| "discovers the seed(s)" | Correct in substance, but Step 1's *declared* primary output is the window config; survivors leave through a channel the manifest does not declare. |
| "which are sent to the rest of the pipeline — fingerprints extracted, ML learns, pools built" | Correct, via the certified 22-array NPZ generation. |
| "ML finds which seeds are relevant" | Correct — Steps 3/5/6 rank the survivor pool. |
| "if ML decides a regime change, it re-runs the seed space — or something like that" | **A mechanism exists**, documented *and* implemented, but it is **human-gated at three separate points**, it re-runs **Steps 1→6 as configured** rather than "the seed space", and it inherits the global frontier — so a regime-shift rerun would sweep the *next* uncovered range, not re-examine the range that produced the current survivors. Two links in it are also inert (§5). |

The single most decision-relevant fact this pass turned up, for choosing a seed geometry: **the
sieve window is anchored at the OLDEST end of the dataset and cannot presently reach recent
draws.** `offset` is a head-relative index bounded `[0, 100]` and `window_size` is bounded
`[6, 50]`, so every sieve window lives inside `data[0:150]` of an 18,068-record, date-ascending
dataset — the 2000–2001 era. Details and anchors in §4.1.

---

## 1. Q1 — What did Step 1 do BEFORE the persistent-worker transports?

### 1.1 The earliest authoritative statement of purpose

The project predates its repository (initial commit `0101306`, 2025-11-29). The earliest
*authoritative* statement of Step 1's purpose is in the pre-repository operating manual, tracked
into the repo at `2b15002` (2025-12-07) as `docs/instructions.txt`:

> `docs/instructions.txt:4366-4370`
> ```
> ## WINDOW OPTIMIZER - Finding Optimal Sieve Parameters
>
> ### Purpose
> The window optimizer is NOT a sieve itself - it's a meta-tool that finds the BEST
> window parameters (window_size, offset, skip_range) to use with your forward/reverse
> residue sieve filters.
> ```

and, immediately under a heading naming the whitepaper as its source:

> `docs/instructions.txt:4372-4376`
> ```
> ### Strategy Context (from your whitepaper)
> 1. Forward Sieve: Uses %8, %125, %1000 residue filters + PRNG pattern matching
> 2. Reverse Sieve: Validates backward consistency through historical draws
> 3. Window Optimizer: Finds which window configuration minimizes false positives
> ```

The same file, in a different section, states the goal in the opposite direction:

> `docs/instructions.txt:3437-3439`
> ```
> WINDOW OPTIMIZER - COMPLETE DOCUMENTATION
> What It Does:
> Intelligently searches for the optimal window configuration that maximizes bidirectional
> survivors from forward + reverse sieve intersection.
> ```

> `docs/instructions.txt:4416-4417`
> ```
> ### Interpreting Results
> Lower bidirectional survivor count = Better configuration
> ```

**This contradiction is in the earliest source itself and is reported as found, not
adjudicated.** "Maximize survivors" and "lower survivor count is better" cannot both be the
objective. The contradiction was later settled *in favour of neither extreme*: the live target
is a **band**, 1K–10K bidirectional survivors
(`baselines/baseline_window_thresholds.json` → `expected_survivor_band`, cited at
`docs/CHAPTER_1_WINDOW_OPTIMIZER.md:587-588`), with the whitepaper's reasoning as the WHY —
an exact sieve leaves `{s*}` with "no ranking, no gradients, and **no learning signal**"
(`docs/BIDIRECTIONAL_SIEVE_MATHEMATICAL_WHITEPAPER.md:118-123`).

### 1.2 The code as it existed at the initial commit

`git show 0101306:window_optimizer.py` (header dates the file 2025-11-15, two weeks before the
repo existed):

```
Window Optimizer - WITH VARIABLE SKIP SUPPORT
Version: 2.0
Date: 2025-11-15
...
The key feature: This runs REAL sieves on all 26 GPUs!
```

`git show 0101306:window_optimizer_integration_final.py`:

```
Window Optimizer Integration - WITH VARIABLE SKIP SUPPORT
Version: 2.0
...
ACCUMULATES ALL BIDIRECTIONAL SURVIVORS WITH RICH METADATA
Saves ALL survivors from ALL trials with window metadata for temporal diversity
```

**So both deliverables existed at the initial commit.** The optimizer module frames itself as a
parameter search; the integration module frames itself as a survivor accumulator.

### 1.3 The chapter

> `docs/CHAPTER_1_WINDOW_OPTIMIZER.md:80-83`
> ```
> The Window Optimizer is **Step 1** of the 6-step pipeline. It performs two critical functions:
>
> 1. **Parameter Optimization:** Uses Bayesian optimization (Optuna TPE) to find optimal window parameters
> 2. **Survivor Generation:** Runs real sieves across all 26 GPUs and accumulates survivors
> ```

### 1.4 Which was primary — answered from the executable declaration

`agent_manifests/window_optimizer.json` is the file WATCHER actually reads. Read live this
session (it is gitignored under `.gitignore:41` `*.json` but force-added and tracked):

| manifest key | live value |
|---|---|
| `description` | `"Step 1: Bayesian window optimization using Optuna TPE. Runs real forward/reverse sieves across 26 GPUs to find optimal window_size, offset, and skip_range parameters."` |
| `primary_output` | `"optimal_window_config.json"` |
| `success_condition` | `["optimal_window_config.json"]` |
| `outputs` | `["bidirectional_survivors.json", "optimal_window_config.json"]` |

**The configuration is primary by declaration — unambiguously.** Step 1 *succeeds* if and only
if a window config was written. The survivor set is a secondary listed output.

**And the declaration is now wrong about the survivors.** The file the manifest names as the
survivor output is a summary with no seeds in it:

> `docs/CHAPTER_1_WINDOW_OPTIMIZER.md:1749`
> `bidirectional_survivors.json` — *"**post-success SUMMARY of the certified generation** —
> generation IDs and sha256s, **no seeds**"*, quoting the source in place:
> *"It is NO LONGER the canonical Steps 2-6 input… Steps 2-6 consume the canonical NPZ"*
> (`window_optimizer_integration_final.py:2628-2632`).

> `docs/CHAPTER_1_WINDOW_OPTIMIZER.md:1739-1746`
> *"The canonical Step-1 → Steps-2–6 carrier is the **certified NPZ generation**, produced by
> `utils.run_finalizer.finalize_run` (`window_optimizer_integration_final.py:2603`). It is the
> one output that matters, and the pre-correction table had no row for it."*

**Divergence, stated plainly:** the artifact that carries Step 1's actual value to the rest of
the pipeline is **not declared in Step 1's manifest at all** — not in `outputs`, not in
`primary_output`, not in `success_condition`. A Step-1 run that produced a config and no
survivors would be recorded as a success.

### 1.5 Exhaustive or sampled? Per-draw or global?

**Exhaustive within a contiguous range, and global.** Three independent anchors:

1. **Earliest manual** — the invocation is `seed_start` + `seed_count`, a contiguous block, with
   no draw parameter of any kind:
   > `docs/instructions.txt:4005-4009`
   > ```
   > results = coordinator.optimize_window(
   >     dataset_path='daily3.json',           # Your data file
   >     seed_start=0,                         # Start at seed 0
   >     seed_count=1_000_000_000,             # Test 1 billion seeds (match your working test)
   >     prng_base='java_lcg',                 # PRNG type
   > ```
2. **Live manifest defaults** — `seed_start = 0`, `max_seeds = 1073741824` (2³⁰), `prng_type =
   java_lcg`. There is no draw, date, session-window or dataset-position parameter in
   `default_params`. (Read live from `agent_manifests/window_optimizer.json`.)
3. **Live partitioning** — `assign_stripes` calls `partition_macro_stripes(total_seeds,
   config.miner_stripe_size, base_start)` and documents itself as *"Partition total_seeds into
   **contiguous** macro-stripes"* (`miner/range_miner_coordinator.py:2245-2260`). Every seed in
   `[seed_start, seed_start + total_seeds)` is visited. Nothing samples.

The *sampling* in Step 1 is over **window/threshold configurations** (Optuna TPE over
`window_size`, `offset`, `session_idx`, `skip_min`, `skip_max`, `forward_threshold`,
`reverse_threshold` — `window_optimizer_bayesian.py:529-550`), never over seeds.

**Answer to Q1:** the deliverable was **both**, from the first commit; the **optimal window
configuration was primary** and remains primary by every executable declaration; and the seed
range was **swept exhaustively over a contiguous global block per PRNG family**, never sampled
and never per-draw.

---

## 2. Q2 — Did the goal change when the project pivoted to PWC (SSH → TCP → ZMQ)?

### 2.1 The transports themselves: purely mechanical. Evidence.

Each transport change states its own scope, and every one of them is about process lifecycle
and connection management. None mentions Step 1's output, deliverable, or semantics.

| change | governing document / commit | stated scope |
|---|---|---|
| PWC (SSH) lifecycle | `docs/PROPOSAL_PWC_LIFECYCLE_FIX_S156_v2_0.md:1-45` | *"the 'persistent' worker pool is being recreated every trial instead of once per optimization session, and that repeated spawn/teardown cycle is the most likely reason rrig6600c hits the multi-process ROCm/PageTables crash cliff"* (Beta-confirmed root cause, quoted in the proposal) |
| ZMQ + SQLite | `docs/PROPOSAL_ZMQ_SQLITE_COORDINATOR_S158D_v1_0.md:1-8`, `:11-24` | *"Scope: New standalone files only… Risk: Low — purely additive, activated by flag, PWC untouched"*; the fault is *"Zeus babysits 24 SSH processes instead of GPUs doing pure compute."* |
| TCP-PWC | `db693db`, `935a04b` (2026-04-03), `716e641`, `6432a48` (2026-04-04) | commit subjects are transport/default/parser/manifest wiring only |

`git log --follow --since=2026-03-01 --until=2026-07-08 -- window_optimizer_integration_final.py`
returns 40 commits. Read as a set, they are: transport plumbing, VRAM/OOM repair, checkpoint
durability, warm-start, kernel-arg fixes, threshold repair. **None of them changes what Step 1
delivers.**

### 2.2 But two real scope changes landed inside that window — under their own governance

This is the honest answer to "did the goal move": **the transports did not move it; two
separately-governed changes did, and they happen to sit inside the same calendar window.**

| # | change | commit(s) | governing document | what moved |
|---|---|---|---|---|
| A | **Step 1 becomes the producer of the Steps-2–6 carrier** | `9ea8464` (2026-01-25) *"fix: Step 1 now auto-generates NPZ for Step 2 (Team Beta approved)"*; `3c3fc3d` (2026-01-25) *"NPZ auto-conversion in ALL execution paths"* | `docs/PROPOSAL_NPZ_Auto_Conversion_Step2.md` (2026-01-19) — *"A gap exists in the pipeline where Step 2.5 **expects** `bidirectional_survivors_binary.npz` but nothing **creates** it automatically."* | Step 1 acquires responsibility for the downstream carrier artifact. **Predates PWC** by ~7 weeks. |
| B | **Survivors become cumulative across runs, not per-run** | `3940517` + `ad5ab8d` (2026-03-15) *"progressive sweep framework"* / *"NPZ accumulator validated"* | `docs/PROPOSAL_S145_R1_Progressive_Empirical_Sweep.md` — TB approved *"Cross-session survivor accumulation"* and *"Merge by best per-seed `score`"* (`:13-14`) | The deliverable changes from "this run's survivors" to "a permanent accumulated pool." **Concurrent with PWC but caused by neither.** |

Two further enrichments in the same period, both narrower: `c6fde66` (2026-03-13, S140b) added
Step-1 trial history + warm-start + a downstream feedback loop; `S103`/`S104` (Feb 2026) turned
survivor records from trial-level aggregates into per-seed match rates plus the seven
intersection fields (`docs/CHAPTER_1_WINDOW_OPTIMIZER.md:88-101`).

**S145-R1 is also where the boundaries of the goal were explicitly *refused* by Beta** — this is
the clearest statement in the whole trail of what Step 1 is *not* for:

> `docs/PROPOSAL_S145_R1_Progressive_Empirical_Sweep.md:20-22`
> ```
> | "Complete 32-bit sweep" claim for java_lcg | ❌ Rejected |
> | "Practically sufficient coverage" conclusion | ❌ Rejected — deferred to post-sweep |
> | "Step 1 retired permanently" | ❌ Rejected |
> ```

> `:38-42`
> ```
> **What this is not:**
> - Not a mathematically exhaustive sweep of java_lcg (state space is 2^48)
> - Not proof that the CA ADM seeds only in 0→2^32
> - Not grounds to retire Step 1 permanently
> ```

**Answer to Q2:** the SSH → TCP → ZMQ pivots were **purely mechanical** — no commit and no
governing document in that lineage touches Step 1's deliverable, scope or semantics. Step 1's
deliverable *did* move twice in the surrounding period, once **before** the pivot (`9ea8464`,
NPZ carrier) and once **during but independently of** it (`3940517`/`ad5ab8d`, S145-R1
cross-session accumulation). Attributing either to the transport work would be wrong.

---

## 3. Q3 — Did the goal carry over into RANGE-MINER, or was some of it lost?

### 3.1 The contract carried over. Verified on disk.

The interface obligation is the 22-array NPZ, not value-matching against PWC (skill §0.7; Beta
retired PWC from certifying authority 2026-07-31). Read this session from the D6 release-grade
generation root at `/home/michael/d6_release_grade_20260729/generation_root/`:

```
bidirectional_survivors_binary.npz : 22 arrays, 319 survivors
  bidirectional_count, bidirectional_selectivity, forward_count, forward_matches,
  forward_only_count, intersection_count, intersection_ratio, intersection_weight,
  offset, prng_type, reverse_count, reverse_matches, reverse_only_count, score,
  seeds, skip_max, skip_min, skip_mode, skip_range, survivor_overlap_ratio,
  trial_number, window_size
```

All seven S104 intersection fields present; per-seed `forward_matches`/`reverse_matches`
present. **The contract carried.**

### 3.2 The brief's example — CONFIRMED

`StripeCompleteMessage.elapsed_s` is produced by the worker off a real clock:

> `miner/range_miner_worker.py:1341-1345`
> ```python
>             substripes_done=len(subs),
>             ...
>             elapsed_s=round(time.time() - t0, 3),
> ```

and the coordinator's handler passes six arguments to the ledger, none of them `elapsed_s`:

> `miner/range_miner_coordinator.py:5903-5905`
> ```python
>             self.ledger.record_stripe_complete(
>                 run_id, msg.stripe_id, attempt, bound_worker_id,
>                 msg.substripes_done, msg.survivors_total)
> ```

`grep -n "elapsed_s" miner/range_miner_coordinator.py` returns **no hits**. The value is decoded
and discarded. Confirmed.

### 3.3 Others of that shape — four more, plus one systemic

**Method:** for each field of each `miner/range_miner_protocol.py` dataclass, locate the worker
producer and then every coordinator reference. A field with a producer and no consumer is the
shape.

| # | field(s) | worker produces | coordinator consumes | verdict |
|---|---|---|---|---|
| 1 | `StripeCompleteMessage.elapsed_s` | `range_miner_worker.py:1345` | — | **DROPPED** (the brief's case) |
| 2 | `RegisterMessage.gpu_name`, `.vram_bytes`, `.gpu_id` | real device query `range_miner_worker.py:1163-1167`, sent `:1242-1245` | `register_worker(worker_id, hostname, backend, capabilities, node_config, admission_reason)` — `range_miner_coordinator.py:5745-5748`. `gpu_name`/`vram_bytes` appear **nowhere** in the coordinator | **DROPPED** |
| 3 | `MinerHeartbeatMessage.stripes_done`, `.stripes_error`, `.busy` | counters maintained `:1227-1228`, `:1327`, `:1352`; sent `:1272-1275` | heartbeat handler reads **only** `msg.current_stripe_id` — `range_miner_coordinator.py:5819-5840` | **DROPPED** |
| 4 | `StripeErrorMessage.error`, `.traceback` | `:1356-1361`, `traceback.format_exc()` | `handle_stripe_failure(run_id, stripe_id, retryable, eligible_workers, …)` — `range_miner_coordinator.py:4144-4148`, called at `:5909-5911` with `retryable=msg.retryable` only. No `msg.error` / `msg.traceback` reference exists in the coordinator | **DROPPED** |
| 5 | `MinerStatusMessage` (whole message: `state`, `progress`, `sub_index`, `stats`) | worker answers a status query in full — `:1439-1452` | the dispatcher returns early: `if mt not in ("sub_stripe_result", "stripe_complete", "stripe_error"): return` — `range_miner_coordinator.py:5841-5842`. The coordinator also **never constructs** a `MinerStatusMessage`, so the query is never sent | **DEAD CHANNEL — both directions** |

**Not on this list, deliberately:** `effective_threshold` on both message types **is** captured
(`range_miner_coordinator.py:5858-5865` → `record_substripe_effective` /
`record_stripe_complete_effective` at `:4486-4492`). `RegisterMessage.capabilities` **is**
captured (`:5747`). Those are the D6 provenance legs and they work.

### 3.4 The systemic version — the whole per-GPU telemetry surface is absent from the miner

The five drops above are not five independent oversights. They are the field-level trace of one
missing subsystem.

**PWC has it:**
- per-chunk wall clock into a dashboard writer —
  `persistent_worker_coordinator.py:1347` (`elapsed = time.time() - t0`) →
  `:1397` `self._progress_writer.log_gpu_result(hostname, gpu_id, gpu_type, seeds_in_chunk, elapsed, success=True)`
- worker error text and traceback surfaced —
  `:1519-1523` (`err_msg = res.get('message') or res.get('error', 'unknown')`;
  `tb = res.get('traceback', '')`; `self.logger.warning(f"  Worker traceback:\n{tb}")`)

**Who uses `log_gpu_result`:** `coordinator.py`, `persistent_worker_coordinator.py`,
`zmq_sqlite_coordinator.py`, `progress_display.py` — i.e. **all three legacy backends**.

**`grep -rn "ProgressWriter\|log_gpu_result\|progress_writer" miner/*.py` returns nothing.**
The miner path has no ProgressWriter integration at all.

**Why this is the right framing, and what it is not:** the miner's contract is the 22 arrays,
and the 22 arrays say nothing about telemetry — so this is **capability loss outside the
contract, not a contract violation.** But it is not cosmetic either, for two reasons that are
already on the record:

- `docs/PROPOSAL_S172_RANGE_MINER_v1_4_5.md:36` makes *"end-to-end throughput … a
  release-blocking Phase 6 dimension"*, with per-path throughput required on the same hardware
  (`:224-288`). The per-GPU timing that would decompose an end-to-end number is exactly what
  item 1 discards.
- skill §2.17's substitute fault-detection for the soak — GPU kernel logs being `UNAVAILABLE` on
  the unprivileged CT100s — names *"repeated lease expiries per identity"* and worker liveness
  as the surrogate signals. Items 3 and 4 are the per-worker error counter and the per-failure
  cause, dropped at the coordinator boundary.

### 3.5 One thing that looks like a miner loss and is not — stated so it is not re-reported

`skip_sequences` (which kills the three dead ML features `skip_mean`/`skip_std`/`skip_entropy`)
is **not** lost by the miner relative to PWC. Both engines produce it on the GPU
(`miner/range_miner_worker.py:917-924, :962-964`; `sieve_gpu_worker.py:254-286`). The kill is on
the **shared host side** — `extract_survivor_records()`
(`window_optimizer_integration_final.py:125`) reduces each survivor to `{seed, match_rate}`.
That is skill §2.2 and `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:943-949`, already governed.

**One non-obvious lineage detail that is worth carrying, because it changes where the fix
goes.** The standing statement — *"reviving them requires no kernel change, only that the host
stop discarding the sequence"* — is true of the **PWC** path, where `skip_sequences` reach the
parent process intact (`persistent_worker_coordinator.py:1541`). On the **miner** path they do
not reach the parent at all: they are validated and then discarded at the spool→projection
boundary inside the miner, under an explicit locked Beta ruling —

> `miner/range_miner_npz_writer.py:358-366`
> *"LOCKED DEFINITION (Team Beta D5 ruling, finding F1) … `merge_validated_spools` consumes only
> seed and match_rate per survivor … `strategy_id` and the ragged `skips` are fully validated
> inside `read_and_validate_spool` and then DISCARDED … They never cross a process boundary
> because canonical assembly never observes them — which is also what lets D5's artifact codec
> run `allow_pickle=False` with no object arrays."*

So on the certifying engine, the skip-output revival is **not** a host-only change; it needs the
D5 projection reopened. Not a defect — a governed decision — but the cheap-half framing does not
transfer to the engine that is now authoritative.

**Answer to Q3:** the 22-array contract carried over intact. What did not carry over is the
**operational-observability surface**: five worker-computed fields are decoded and discarded at
coordinator boundaries (§3.3), and the ProgressWriter/per-GPU-throughput subsystem present in
all three legacy backends is entirely absent from the miner (§3.4). The `elapsed_s` case named
in the brief is the smallest visible symptom of that, not an isolated one.

---

## 4. Q4a — Is a sweep per-draw or global?

### 4.1 Documented intent: global. No per-draw intent found anywhere.

The tracker keys on `prng_type` alone, and says why in its own docstring:

> `database_system.py:330-336`
> ```python
> def get_next_seed_start(self, prng_type: str, chunk_size: int) -> int:
>     """
>     [S140] Seed Coverage Tracker — returns the next uncovered seed_start
>     for a given prng_type across ALL prior runs.
>
>     Queries MAX(seed_range_end) from exhaustive_progress for this prng_type.
> ```
> `:348` — `'SELECT MAX(seed_range_end) FROM exhaustive_progress WHERE prng_type = ?'`

and WATCHER states the intent in the imperative:

> `agents/watcher_agent.py:1662-1666`
> ```python
>         # [S140] SEED COVERAGE TRACKER — Step 1 only
>         # Reads MAX(seed_range_end) for this prng_type from exhaustive_progress.
>         # Advances seed_start between pipeline runs so we never re-search covered ranges.
>         # If DB lookup fails for any reason, defaults to 0 — run proceeds normally.
>         # Invariant: new seed range forces fresh study (resume_study=False, study_name='')
> ```

The table's own schema confirms there is no per-draw dimension to key on. Read live, read-only:

```
PRAGMA table_info(exhaustive_progress):
  search_id, prng_type, mapping_type, seed_range_start, seed_range_end,
  seeds_completed, best_score, best_seed, last_updated
```

No draw column. No dataset column. No date column.

**Live coverage state** (read-only connection to `prng_analysis.db`, this session):

```
prng_type   rows   MIN(seed_range_start)   MAX(seed_range_end)
java_lcg     15              0              16,106,127,360
```

The frontier has advanced to **≈16.1 billion — 3.75 × 2³²**, i.e. past the entire S145-R1
target range. Under the live manifest (`max_seeds = 1073741824`), the *next* WATCHER-driven
Step-1 run would sweep `[16,106,127,360, 17,179,869,184)`.

**"Is the sweep per-draw" — no evidence found.** The governance trail, the chapters, the
whitepaper, `instructions.txt`, the manifest and the code contain **no statement, requirement,
schema field, parameter or ruling** describing per-draw seed discovery. The per-draw language
that does exist in the corpus is about something else entirely: `skip` (gaps *between* observed
draws — `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:480-486`) and per-draw false-positive rates
(`:724-737`). The owner's "for each draw" is, as far as the repository is concerned, not a thing
Step 1 has ever been specified to do.

### 4.2 Do survivors carry across runs? Yes — approved, and still implemented, by a different mechanism than the one that was approved.

Approved: `docs/PROPOSAL_S145_R1_Progressive_Empirical_Sweep.md:13-14`
(*"Cross-session survivor accumulation ✅ Approved"* / *"Merge by best per-seed `score`
✅ Approved"*), with the stated rationale *"Accumulated permanent survivor population — nothing
is ever discarded"* (`:45`).

Implemented, 2026-03: a JSON/NPZ accumulator merging on best per-seed score
(`docs/PROPOSAL_S145_R1_Progressive_Empirical_Sweep.md:178-192`; the per-trial NPZ checkpoint
descendant is still live at `window_optimizer_bayesian.py:224-292`, merging against a prior
`bidirectional_survivors_all.npz` at `:250-265`).

Implemented **today**: the same *policy* under a different, frozen *mechanism* — the D3.5
finalizer's chained generations:

> `utils/run_finalizer.py:752-762`
> ```
> def _l3_merge(...)
>     """Merge L2 winner arrays against the certified prior's arrays.
>         new score >  prior score  -> replace with the new row
>         new score == prior score  -> RETAIN PRIOR, byte-for-byte, every array
>         new score <  prior score  -> RETAIN PRIOR, byte-for-byte, every array
>     STRICT GREATER-THAN ONLY.
> ```

**S145-R1's approved merge policy survived verbatim; its implementation was replaced.** Worth
stating because the S145-R1 document reads as current and its code section is not.

### 4.3 Anchored or sliding? **Anchored — and anchored at the OLDEST end.**

This is the question the brief says "determines which reading can be true", so the anchors are
given in full.

**The window is a head-relative slice of the session-filtered, date-ascending record list:**

> `miner/range_miner_worker.py:641-650`
> ```python
>     if sessions:
>         data = [e for e in data if e.get("session") in sessions]
>     n = len(data)
>     if n < window_size:
>         raise ResidueResolutionError(...)
>     start = max(0, min(int(offset), n - window_size))
>     window = data[start:start + window_size]
> ```

This is the **shared residue authority** — the same function the coordinator side calls
(`:679-681`: *"the worker's default loader IS the canonical session-aware derivation the
coordinator side also calls"*), so there is no second implementation to check.

**`offset` is confirmed as a head-relative index, not an offset from the present:**

> `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:790-795`
> | source | definition |
> | old Chapter §3.1 | "time offset from current draw" |
> | **host code** | **head-relative array index into the session-filtered draw list** |
>
> `:801-803` — *"Host, as a data index: `start = max(0, min(int(offset), n - window_size));
> window = data[start:start + window_size]`"*

**The dataset is date-ascending and appends at the tail:**
`docs/PROJECT_FILE_CATALOG.md` / skill §2.14 (measured on VM101 2026-08-02): 18,068 records,
`2000-01-01 evening` → `2026-02-26 midday`, canonical order **(date ascending, evening before
midday)**, and *"the stored file MATCHES it."*

**The live bounds** — read this session from `distributed_config.json` (the authority;
`window_optimizer.py:85-87` merges it over code defaults, config wins per key):

```
window_size  min=6,  max=50,  default=12
offset       min=0,  max=100
skip_min     min=0,  max=10
skip_max     min=10, max=250
forward_threshold  min=0.3, max=0.75, default=0.3
reverse_threshold  min=0.3, max=0.75, default=0.3
```

**Two consequences follow, and both bear on the seed-geometry decision:**

1. **Appending draws does not invalidate prior survivors.** For any `offset` that already
   satisfied `offset ≤ n − window_size`, growing `n` leaves `data[start:start+window_size]`
   byte-identical. So the reading *"anchored → survivors filter down, reuse valid"* is the one
   the code supports. The competing reading — *"Optuna tunes `offset`, so old survivors may not
   be valid"* — is true only **across different `offset` values**, i.e. between two trials or two
   runs that picked different offsets, not because new draws arrived. Those are different
   claims and only the second is real.

2. **The sieve window cannot presently reach recent data.** With `offset ∈ [0,100]` and
   `window_size ∈ [6,50]`, every reachable window lies inside `data[0:150]` of 18,068 records.
   After session filtering (`sessions` is itself an Optuna dimension, `session_idx` at
   `window_optimizer_bayesian.py:535`) that is the **earliest ~150 draws — the 2000–2001 era**,
   which per skill §2.14 is exactly the evening-only era (1,038 of the 1,040 single-session
   dates fall in 2000–2002).

   I am reporting this as an **observed property of the live bounds**, not as a defect and not
   as a recommendation — the brief forbids proposing anything, and §0.4's standing rule applies:
   there may be a document explaining why `offset.max` is 100 that I did not find. What I can
   say is that `distributed_config.json` carries `_calibration_note` and `_s172_note` for
   `window_size` and **no note of any kind for `offset`**, and
   `docs/CHAPTER_1_WINDOW_OPTIMIZER.md:581-583` states those `_note` keys are *"the only in-repo
   record of why"* these values are what they are. So for `offset.max = 100` there is **no
   in-repo record of why**. If a seed geometry is about to be chosen, this is the bound to ask
   about first.

**Answer to Q4a:** documented intent is **global**, keyed on `prng_type`, monotonically
advancing, explicitly *"so we never re-search covered ranges."* Survivors are meant to **carry
across runs** (S145-R1, approved; implemented today by the finalizer's strict-greater-than L3
merge). The window is **anchored**, so the carry-forward reading is the coherent one. Per-draw
re-derivation: **no evidence found, in any source.**

---

## 5. Q4b — Does a regime-change re-sweep trigger exist?

**Yes. It is documented, it is implemented, and it is human-gated at three points. Two of its
links are inert, and one structural coupling makes the automatic arm hard to fire.**

### 5.1 Documented intent — Chapter 13, three places

> `docs/CHAPTER_13_LIVE_FEEDBACK_LOOP.md:117-120`
> ```
> | Category | Steps | Trigger |
> | **Static** | 1, 2, 4 | Run once; re-run only on regime shift |
> | **Dynamic** | 3, 5, 6 | Re-run as part of learning loop |
> ```
> `:122` — *"**Key insight:** The system learns by weighting survivors, not by endlessly
> searching new ones."*

> `docs/CHAPTER_13_LIVE_FEEDBACK_LOOP.md:504-511`
> ```
> ### 10.4 Regime Shift (Full Pipeline)
> Trigger Steps 1→6 only when:
> - Window decay > 0.5
> - Survivor churn > 0.4
> - LLM flags structural drift with confidence > 0.8
> - Manual operator override
> ```

> `:755-762` (§15.2 Step Interaction Matrix) — Step 1: *"Re-invoke only on regime shift"*;
> `:766-780` (§15.3) — the `requires_regime_reset() → run_steps([1,2,3,4,5,6])` sketch.

And the countervailing statement, in the same chapter — Step 1 is on the never-autonomous list:

> `docs/CHAPTER_13_LIVE_FEEDBACK_LOOP.md:166-172`
> ```
> ## 5. What Remains Stable
>
> These are **never** subject to autonomous modification:
>
> | Component | Reason |
> |-----------|--------|
> | Step 1 (Window Optimizer) | Defines search space; expensive; structural |
> ```

**These are not contradictory:** §5 forbids autonomous *modification of Step 1's parameters*;
§10.4 permits a governed *re-invocation of Step 1 as configured*. The distinction is exactly
whitepaper §9 / skill §0.5 — autonomy adjusts parameters within bounds, never structure — and
here it does not even adjust parameters.

### 5.2 Implemented — the live chain, end to end

| # | stage | anchor |
|---|---|---|
| 1 | compute `window_decay` (relative hit-rate decline) and `survivor_churn` (relative survivor-count change) | `chapter_13_diagnostics.py:549-559`, returned `:571-578` |
| 2 | flag `HIGH_WINDOW_DECAY` / `HIGH_SURVIVOR_CHURN` against thresholds `0.5` / `0.4` | `:654-658` |
| 3 | `REGIME_SHIFT_POSSIBLE` iff **both** | `:664-665` |
| 4 | `actions["rerun_step_1"] = True` and `human_review_required = True` | `:694-696`, reinforced `:723-724` |
| 5 | independently, the trigger evaluator raises `TriggerType.REGIME_SHIFT` → `TriggerAction.FULL_PIPELINE` — from the raw metrics at `chapter_13_triggers.py:290-302` **and** from the flag at `:313-319`; REGIME_SHIFT is top of the priority order `:339-347` | |
| 6 | `FULL_PIPELINE → [1, 2, 3, 4, 5, 6]` | `chapter_13_triggers.py:454-455` |
| 7 | written to an approval-request file, `requires_approval=True  # v1: always require approval` | `:369`, `:431-448` |
| 8 | **human runs `python3 chapter_13_triggers.py --approve`** → `execute_learning_loop(steps)` → `self.watcher_agent.run_pipeline(1, 6, params)` | `:485`, `:536-537`, `:612-616` |

A second, independent live route reaches the same place:

> `agents/watcher_dispatch.py:413-416`
> ```
>     Supported request_types:
>         "selfplay_retrain"  → dispatch_selfplay()
>         "learning_loop"     → dispatch_learning_loop()
>         "pipeline_rerun"    → dispatch_learning_loop(scope="full")
> ```
> `:269-272` — `elif scope == "full": steps = [1, 2, 3, 4, 5, 6]`; each executed via
> `self.run_step(step)` at `:314`.

That route is under active governance — `docs/TB_RULING_S179_IMPLEMENTATION_AUTH.md:214-223`
names all three request types and `:265` requires a generic fail-closed governance gate on the
consumer *"immediately after loading and identifying the request"*; `:288-300` requires the same
at the `dispatch_learning_loop()` chokepoint. **S179 is the live authority** and I did not
attempt to determine whether its conditions have landed — that is catalog §7 gap 8's open
question, not this brief's.

### 5.3 Three things that are inert or self-limiting — reported as observations, with status

**(a) `rerun_step_1` has no consumer.** `grep -rn "rerun_step_1"` over the whole tree returns
five hits: the initializer (`chapter_13_diagnostics.py:684`), the setter (`:695`), the approval
check (`:723`), a console print (`:850`), and the Chapter-13 doc's example JSON
(`docs/CHAPTER_13_LIVE_FEEDBACK_LOOP.md:404`). **Nothing reads it to decide anything.** The
working path is the parallel `TriggerAction.FULL_PIPELINE` route in §5.2, which does not consult
it. Same shape as skill §2.13's `Advisor → strategy_recommendation.json → WATCHER` row: a field
that is emitted and validated and never applied.

**(b) `pipeline_rerun` has no producer.** The only writer into `watcher_requests/` is
`request_selfplay()` — `chapter_13_triggers.py:799-804` — and it hardcodes
`"request_type": TriggerType.SELFPLAY_RETRAIN.value` (`:788`). `grep -rn "pipeline_rerun"` finds
the string only in `agents/watcher_dispatch.py` (`:416`, `:530`) and in the S179 ruling. **The
full-pipeline branch is reachable only by a hand-written request file or the
`--dispatch-learning-loop full` CLI** (`agents/watcher_agent.py:3389-3392`) — which is precisely
the "manual operator override" arm of §10.4, and precisely the *"hand-created or stale
`learning_loop` or `pipeline_rerun` request"* S179 §3 requires the generic gate to catch.

**(c) The automatic arm is self-limiting, and I could not find that this has been ruled on.**
`REGIME_SHIFT_POSSIBLE` requires **both** `window_decay > 0.5` **and** `survivor_churn > 0.4`
(`chapter_13_diagnostics.py:664-665`). `survivor_churn` is computed as
`abs(len(survivors) - prev_survivor_count) / prev_survivor_count` (`:556-559`) over
`survivors_with_scores.json`, the **Step-3** output (`:54`, `:23`). In the steady state the
learning loop re-runs Steps 3→5→6 over the *same* survivor pool, so the count is stable and
`survivor_churn ≈ 0` — and a >40% change in the survivor count is essentially what a Step-1
re-run *produces*. The trigger that would re-run Step 1 therefore depends on a signal that
mainly moves when Step 1 has already re-run.

The code says as much about its own churn metric, in place:

> `chapter_13_diagnostics.py:555-557`
> ```python
>         # Survivor churn: how many survivors changed?
>         prev_survivor_count = prev_health.get("survivor_count", len(survivors))
>         # Simplified: would need survivor ID tracking for proper churn
> ```

**Status check before calling this a finding (skill §1.1):** the KPI governance chain
S176→S177→S178→S179 covers these thresholds, and it explicitly parks this one —
`docs/WATCHER_KPI_CALIBRATION_FINDINGS_S176.md:107` records
`survivor_churn_threshold` as **NEEDS-DATA (Type-2)**, and `:219-221` states it *"cannot be
judged without recorded performance/advisor series and [is] out of scope for this analytic
pass."* `watcher_kpi_baseline.py:288-289` classes `window_decay` under
`"needs_full_pipeline_run"`. So the **threshold value** is governed-and-open. The **structural
coupling** described above is not addressed in S176–S179 or anywhere else I searched; I state it
as an observation with anchors, not as a new defect claim, and it should be checked against
Beta's record before being treated as either.

### 5.4 What the trigger would actually do — the part that matters for seed geometry

If approved, it runs `watcher.run_pipeline(1, 6)`, which enters Step 1 through the **same**
`run_step(1)` path as any other run — including the S140 coverage tracker at
`agents/watcher_agent.py:1667-1701`, which will advance `seed_start` to the frontier
(currently 16,106,127,360) and force a fresh Optuna study unless `study_name` is explicitly set.

**So a regime-shift rerun does not re-examine the seed space that produced the current
survivors. It sweeps the next uncovered block.** That is a faithful description of the
mechanism, not a criticism of it — but it is the specific point where the owner's "it re-runs
the seed space" and the implementation part company, and it is directly load-bearing for a seed
geometry choice.

**Answer to Q4b:** the mechanism **exists** — this is not a "no evidence found" case. It is
documented (Chapter 13 §3.2, §10.4, §15.2/15.3), implemented end to end (§5.2), and gated on
human approval at every route. `rerun_step_1` is inert, `pipeline_rerun` has no automated
producer, and the automatic arm's second precondition is structurally hard to satisfy without a
Step-1 rerun having already happened.

---

## 6. The three layers, where they diverge

Per the brief: *"where those three diverge, say so plainly, because that divergence is the
actual deliverable of this task."*

| # | (i) documented intent | (ii) what was implemented | (iii) what runs today | divergence |
|---|---|---|---|---|
| 1 | Step 1 delivers an **optimal window configuration**; survivors are a second output (`instructions.txt:4368`; manifest `primary_output`/`success_condition`) | Step 1 also became the **producer of the Steps-2–6 carrier** (`9ea8464`), later the certified 22-array generation | The certified generation is *"the one output that matters"* (`CHAPTER_1:1739-1746`) | **The manifest still declares neither the NPZ nor the generation.** Step 1 can succeed by its own success condition without delivering the thing downstream needs. |
| 2 | Survivors accumulate permanently, merged by best per-seed score (S145-R1, TB-approved) | JSON→NPZ accumulator, 2026-03 | Finalizer `_l3_merge`, strict `>`, byte-identical prior retention (`run_finalizer.py:752-762`) | **Policy survived; mechanism was replaced.** S145-R1 reads as current and its §5.1 code block is not. |
| 3 | Sweep is a **global frontier** per `prng_type`, *"so we never re-search covered ranges"* | `exhaustive_progress` keyed on `prng_type`; WATCHER advances `seed_start` | Frontier at **16,106,127,360**; next run starts there | **No divergence** — intent, implementation and live state agree. This is the cleanest chain in the report. |
| 4 | RANGE-MINER must produce *all* data the remaining steps require (skill §0.7; the 22 arrays) | 22-array contract implemented and Phase-6 certified for the miner/finalizer path | Contract holds (§3.1) | **No contract divergence.** But the per-GPU telemetry surface present in all three legacy backends is absent from the miner (§3.3–3.4) — out-of-contract capability loss. |
| 5 | Step 1 re-runs on regime shift, human-approved (Ch13 §10.4) | Full chain implemented (§5.2) | Reachable; two links inert; automatic arm self-limiting (§5.3) | **Documented behaviour is broader than reachable behaviour.** |
| 6 | Sieve window sized/positioned by Optuna over a governed range | `offset` = head-relative index into a date-ascending list | `offset ∈ [0,100]`, `window_size ∈ [6,50]` → window confined to `data[0:150]` of 18,068 | **The reachable window is the oldest ~0.8% of the dataset**, and `offset.max` carries no `_note` and no in-repo rationale (§4.3). |
| 7 | Certifying engine is RANGE-MINER; PWC retired from certifying authority 2026-07-31 (skill §0.7) | Miner is flag-selected (`--use-range-miner`) | Live manifest `default_params`: `use_range_miner = False`, `use_persistent_workers = True`, `pwc_transport = "tcp"` | **A WATCHER-driven Step 1 launched today would take the PWC-TCP path, not the certifying engine.** Stated as read from the manifest; the gate-12 production shape is the miner and is Beta-held, so this may simply be the pre-flip state — but it is what the file says right now. |

**Two further live-state facts, recorded without interpretation:** the repo root currently has
**no** D3.5 finalizer-owned compatibility symlinks and **no** `bidirectional_survivors_all.npz`
or `bidirectional_survivors_binary.npz` (only a leftover
`bidirectional_survivors_all.npz.flush.tmp.npz` and a `bidirectional_survivors_binary.meta.json`);
`optimal_window_config.json` is present only as `optimal_window_config.json.stale_1786149572`.
The last release-grade generation is out of tree at
`/home/michael/d6_release_grade_20260729/generation_root/` — **319 survivors, seeds
2,568 → 7,983,890**, i.e. entirely below 8 × 10⁶ while the coverage frontier stands at
1.6 × 10¹⁰. That is one run's certified output, not the accumulated pool, and it is offered as
a measurement, not an argument.

**Unavailable surface (VIR-1/VIR-6):** the ser8 pre-repository archives were not reachable from
VM101 this session. Everything in §1 about the pre-repository era rests on
`docs/instructions.txt` / `Cluster_operating_manual.txt` (identical 95,105-byte copies at repo
root and in `docs/`) and on `git show <initial-commit>:<file>`. If an earlier design note for
Step 1 exists, ser8 is where it would be, and this report does not cover it.

---

## 7. What was NOT found — stated plainly

- **No evidence found** of any source — governance, chapter, whitepaper, manual, manifest or
  code — describing Step 1 as a **per-draw** seed-discovery step, or of any seed-coverage key
  that includes a draw, date or dataset dimension.
- **No evidence found** of a ruling, proposal or chapter addressing the `survivor_churn` /
  Step-1-rerun coupling in §5.3(c). S176 parks the *threshold* as NEEDS-DATA; the coupling
  itself appears nowhere I searched.
- **No evidence found** of any in-repo rationale for `offset.max = 100` — the two `_note` keys in
  `distributed_config.json` cover `window_size` only.
- **No evidence found** of a producer for `pipeline_rerun` requests, or of any consumer for
  `rerun_step_1`.
- **No evidence found** that the transport pivots (SSH → TCP → ZMQ) changed Step 1's deliverable,
  scope or semantics.

Nothing in this report is a proposal. No fix, change, or next step is recommended or implied.

**END OF REPORT**
