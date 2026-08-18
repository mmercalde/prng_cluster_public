# SESSION CHANGELOG — 2026-08-17 — GATE-12 ATTEMPT 9 (PRODUCTION ACCEPTANCE RUN)

**Purpose:** the production acceptance run for the R-1…R-4 drain starvation remedy
(`e9ca800`, Beta-certified). Predecessor measurement pass: MP-1 (`168b6f1` / forensic
`c403a37`), run `logs/gate12_20260816_160503.log`.
**Authority:** Michael-initiated. Beta's acceptance question is answered against
**MP-1's own fields** as the oracle, per instruction.
**Host:** VM101, user `michael`, `~/venvs/torch`.
**Status:** **RUN PASSED.** Two acceptance fields are reported **OPEN** — the criterion
is ambiguous as to normalization and the ruling is Beta's, not Alpha's. One field is
**UNOBSERVABLE** for an instrumentation reason diagnosed below.

```
git status --porcelain  AT LAUNCH  (empty)      HEAD e9ca800ae65b44bc555f1402a9102932ba2e72ca
git status --porcelain  AT END     ?? .s172_accumulator/
                                   ?? bidirectional_survivors_all.npz
                                   ?? bidirectional_survivors_binary.npz
                                   (all three are attempt 9's OWN outputs, written
                                    after publication — see §6)
```

**Nothing committed by the agent. Nothing pushed. No mid-run intervention of any kind.**

---

## 0. Result

| | |
|---|---|
| Run stamp | `20260817_181819` |
| Run nonce | `gate12-20260817_181819-46500` (fresh; all prior nonces stay burned) |
| Run id | `distributed_config_t1_554463d3` |
| Stripes | **128 / 128** over **4 stages** |
| Stages | `java_lcg/1`, `java_lcg_reverse/2`, `java_lcg_hybrid/3`, `java_lcg_hybrid_reverse/4` |
| Serve loop | 738.162 s |
| Step 1 | **PASSED** (exit 0) |
| Saturation verdict | **SATISFIED** (criterion 1 AND criterion 2) |
| Coverage | certified `[0, 2,147,483,648)` java_lcg `{constant,variable}` |
| coverage_id | `c6f28aedf7af12cd` |
| Post-run cursor | `covered_seed_count=2147483648`, `certified_interval_count=1` |
| Lease expiries | **0** |
| ERROR / Traceback | **0** |

**No §21 credit composed from attempts 1–8 or any D6/MP-1 run.** The pre-run certified
cursor read `OPEN, next_seed_start=0, covered_seed_count=0, certified_interval_count=0` —
a genuinely fresh domain, verified before launch and recorded in the evidence bundle.

### Pre-flight and in-run gates

Verified before launch, read-only: HEAD `e9ca800`; `git status --porcelain` empty; all three
rigs reachable by key auth (`rrig6600`, `rrig6600b`, `rrig6600c`); **rig code-parity 30/30
MATCH, 0 MISMATCH, 0 UNAVAILABLE**.

In-run, all PASS: clean-tree admission · GPU fail-close 24/24 · rig code-parity · pre-dispatch
clean-tree assertion · worker-log sentinel 25/25 · worker liveness.

Launched detached under `setsid` (pid 46500, **pgid 46500, sid 46500**) — its own session, so
nothing in the operator's tooling shared a process group with the fleet. No timeout wrapper was
used anywhere; the completion watch polled `/proc/<pid>` existence and never called `kill`.
Fleet exited clean: 0 residual workers on all three rigs.

### Saturation evidence (sampler verdict, `logs/gate12_20260817_181819_verdict.txt`)

376 samples, **0 UNOBSERVED**. Peak 25 simultaneous compute-active workers with 7 queued at the
same instant. 4 qualifying simultaneity windows, **all 4 showing turnover**; witness = window 1
(2026-08-17T18:20:05→18:20:15, pending 7→1, drained 6, transitions 6, 5 reaching done).

---

## 1. Abort-signal clarification (it did NOT fire)

The abort condition was **`STEP 2: Scorer Meta-Optimizer (run #N)` actually executing**.

The run log **does** contain, at 18:32:29:

```
[INFO] Triggering Step 2: Scorer Meta-Optimizer
```

**This is not the abort condition and must not be read as one.** It is the evaluator's
follow-up-agent message, emitted after Step 1 passed. `--end-step 1` stopped the pipeline
immediately afterwards (`Pipeline execution finished`, 18:32:30). Evidence that Step 2 never ran:

- the executing form `STEP 2: Scorer Meta-Optimizer (run #` is **absent** from every artifact;
- `run_scorer_meta_optimizer.sh` was **never invoked** — no occurrence anywhere in the log;
- the pipeline summary prints `⬜ Step 2: Scorer Meta-Optimizer` (not run).

The TB-prohibited converter was therefore never reached and the D3.5 finalizer-owned symlink was
never touched. This is exactly the trap §2.25 of the launch script documents, and the
`--start-step 1 --end-step 1` guard held.

---

## 2. Method note — the oracle had to be calibrated before it could be trusted

`phase_attribution.exclusive_s` in the `[S172-SL] window` series is **cumulative per
(phase, thread)**, not a per-window delta. Verified monotonic non-decreasing on MP-1. Summing it
across windows inflates the figure by roughly the window count (it yields 163,283 s for a
17-minute run).

Read correctly — final value per thread, summed across the four staging threads — MP-1 gives:

```
miner-staging_0  908.049 s      miner-staging_2  913.258 s
miner-staging_1  910.571 s      miner-staging_3  908.575 s
TOTAL           3640.452 s   <-- reproduces Beta's cited ~3,640 s exactly
```

Two further independent agreements confirm the extraction is measuring the intended quantity:

- MP-1 pump per pass, extracted: **1.4632 s**; R-1…R-4 changelog's measured `pump` mean: **1.461 s**.
- MP-1 serve-thread `staging` final cumulative 681.153 s == per-window `phases.staging` summed
  681.153 s (two independent accountings agreeing).

MP-1's terminal per-frame message cost is cited in the remedy changelog as **2.14 s**; the window
series here gives last-window 1.958 s and peak 2.499 s, i.e. the same terminal shape.

---

## 3. The six acceptance fields, against MP-1

### ✅ Field 3 — per-frame message cost: the build-up is GONE

| | MP-1 | Attempt 9 |
|---|---|---|
| first window | 0.004707 s/frame | 0.005453 s/frame |
| last window | **1.958172 s/frame** | **0.001323 s/frame** |
| peak | 2.499061 s | 0.323323 s |
| `phase_max.msg` max | 2.880111 s | 0.642240 s |

MP-1 climbs ~416× from first to last. Attempt 9 **ends below where it started**. No build-up.

### ✅ Field 4 — drain passes service materially more than one frame

| | MP-1 | Attempt 9 |
|---|---|---|
| frames / passes | 2471 / 1771 | 6421 / 3283 |
| frames per pass | **1.3953** | **1.9558** |
| `conns_serviced_max` | 0 … 25 | 0 … 25 |
| drain stop reasons | deadline 399, empty 1372 | deadline 468, empty 2815 |

### ✅ Field 5 — the 22/25 zero-service terminal geometry is GONE

Zero-service windows were classified **idle vs starved** rather than counted alike: a window with
no arrivals and an empty queue has nothing to service and is not starvation.

| | MP-1 | Attempt 9 |
|---|---|---|
| windows with service | 76 | 44 |
| zero-service, nothing arrived (idle) | 13 | 28 |
| **zero-service WITH work present (starved)** | **1** | **0** |
| trailing windows `serviced=0` AND queue non-empty | 1 | **0** |
| final `inbound_qsize` | **369** | **0** |

MP-1's terminal five windows: `25 live / 3 serviced`, 5 frames per 5 passes, queue pinned at
340→369 and **rising**, then `25 live / 0 serviced` with 369 still queued. Attempt 9's final
window: `25 live / 25 serviced`, 50 arrived, 25 frames drained, **queue 0**.

### ⚠ Field 1 — pump exclusive on `miner-staging_*`: **OPEN, Beta to rule**

| | MP-1 | Attempt 9 | Δ |
|---|---|---|---|
| absolute total | 3640.5 s | **2394.1 s** | **−34.2%** |
| per wall-second (of 4 threads) | 3.756 | 3.244 | −13.6% |
| calls (4 threads) | 2,488 | 11,386 | +358% |
| **seconds per pass** | 1.4632 s | **0.2103 s** | **−85.6% (7.0×)** |

**Alpha does not call this met.** Beta's wording was that the figure "must COLLAPSE from
~3,640 s". In **absolute** terms it fell 34% — a reduction, not a collapse. The collapse is in
**per-pass cost** (7.0×), with 4.6× more passes, which is the shape R-1 was designed to produce
(probe once per distinct live key, not once per deferred entry). Which normalization the
criterion binds to is Beta's ruling.

Useful context for that ruling — the R-3 bench predicted **0.1364 s per pump pass at 1,700
deferred entries**. Attempt 9 ran at `deferred_high_water=1679`, almost exactly the bench shape,
and observed **0.2103 s**, i.e. **1.54× the bench prediction**. The remedy behaved in the
predicted direction and within the same order, short of the bench figure.

### ⚠ Field 2 — serve-thread staging vs msg: **OPEN, Beta to rule**

| | MP-1 | Attempt 9 |
|---|---|---|
| `staging` (serve thread) | 681.2 s = **70.3%** of loop | **159.9 s = 21.7%** of loop |
| `msg` | 689.2 s = 71.1% of loop | 181.7 s = 24.6% of loop |
| **staging / msg ratio** | **0.988** | **0.880** |

Absolute staging cost fell **76.5%** and its share of the serve loop fell from 70.3% to 21.7%.
But the **ratio** staging:msg barely moved (0.988 → 0.880), because both shrank together.
Whether "cease dominating msg" binds to the ratio or to the share of the loop is Beta's ruling.
**Alpha does not call this met.**

### ✖ Field 6 — the two R-1 high-waters: **UNOBSERVABLE from this run**

`deferred_distinct_attempts_high_water` and `pump_liveness_probes_high_water` appear in **no
artifact** — not the run log, not `~/miner_staging`, not `results/`.

Diagnosis (live source at `e9ca800`):

1. Both keys are initialized in `_bp` — `miner/range_miner_coordinator.py:3719-3720`.
2. Both are updated in the pump under `_bp_lock` — `:7978-7982`.
3. `staging_backpressure_metrics()` returns them: `out = dict(self._bp)` — `:7228-7229`.
4. **`log_staging_backpressure_summary`'s format string omits both keys** — `:7284`. The emitted
   `[S172-BP] summary` line carries 20 fields; neither of these is among them.
5. The returned dict reaches only the in-memory trial result
   (`"staging_backpressure": bp_metrics`, `:10602` / `:10620`), which is never persisted.

**The values were computed during the run and discarded.** Both fields are new in `e9ca800`, so
MP-1 has no baseline for them either — the comparison Beta asked for is not available from any
existing run. The one-line fix is to add both keys to the `:7284` format string; **Alpha has made
no code change**, as the run was an acceptance run and the tree must stay as launched.

Note for whoever rules on this: the code's own comment (`:3714-3721`) binds the reading — both are
high-waters over pump passes and therefore **lower bounds** on the true resident maxima. A value
above the cohort (25) **refutes** the guarantee outright; a small value **corroborates rather than
proves**. Any future report must not over-read a low value.

---

## 4. Back-pressure summary, both runs

| | MP-1 | Attempt 9 |
|---|---|---|
| `staging_jobs_completed` | 1,246 | **5,693** |
| `staging_jobs_per_sec` | 1.285 | **7.712** (**6.0×**) |
| `deferred_high_water` | 739 | **1,679** |
| `inbound_qsize_high_water` | 553 | 358 |
| `bound_in_force` | 1,113 | 2,201 |
| `pause_events` | 0 | 0 |
| `capacity_timeout_terminations` | 0 | 0 |

Attempt 9 carried **2.3× the deferred population** at **6× the staging throughput**, with a lower
inbound high-water. MP-1 failed after 2 stages / 64 stripes; attempt 9 completed 4 stages /
128 stripes.

---

## 5. Artifacts

```
logs/gate12_20260817_181819.log                    run log (the [S172-SL]/[S172-BP]/[H1H2] series)
logs/gate12_20260817_181819_verdict.txt            sampler saturation verdict — SATISFIED
logs/gate12_20260817_181819_concurrency.tsv        per-sample occupancy TSV (376 samples)
logs/gate12_20260817_181819_evidence.txt           pre-flight authority evidence + gate verdicts
logs/gate12_20260817_181819_source_digests.json    rig code-parity, 30/30
logs/gate12_20260817_181819_liveness.json          worker liveness gate evidence
logs/gate12_20260817_181819_fleet.log              fleet dispatch, 25 workers
logs/gate12_20260817_181819_sampler.log            sampler stdout
logs/gate12_attempt9_launcher_20260817_181819.log  launcher stream (gates + tail of run log)
```

---

## 6. Tree state at end — action required before the next launch

The run left three untracked entries, all of them **attempt 9's own outputs, written after
publication**. They did not affect this run: the clean-tree predicate was evaluated at admission
and again pre-dispatch, both empty, and D3.5 had already passed by the time they were written.

```
?? .s172_accumulator/
?? bidirectional_survivors_all.npz
?? bidirectional_survivors_binary.npz
```

They **will** refuse the next launch — the clean-tree admission gate refuses on exactly this
state, and it is what refused attempt 3.

### Disposition: all three belong in `.gitignore`, none in the commit

**What they actually are** (this decides it — none of the three is a data file):

```
bidirectional_survivors_all.npz    -> .s172_accumulator/current/bidirectional_survivors_all.npz     [symlink]
bidirectional_survivors_binary.npz -> .s172_accumulator/current/bidirectional_survivors_binary.npz  [symlink]
.s172_accumulator/current          -> generations/gen-20260818T013219550933Z-step1_java_lcg_0--6af51dd2…54b   [symlink]
```

The real data lives in the hash-named immutable generation directory, beside its
`provenance.json`. The two root entries are **pointers**, and the pointer target embeds a
per-run timestamp and a 64-hex sidecar digest — it changes on **every** run.

**How prior runs' outputs were handled** — the precedent is consistent, and it has one
superseded step that must not be misread:

| commit | date | what happened |
|---|---|---|
| `a9623ee` | 2026-03-07 | S127 — untracks runtime outputs, adds `bidirectional_survivors_binary.npz`, `optimal_window_config.json`, `watcher_decisions.jsonl` to `.gitignore` |
| `006623c` | 2026-03-07 | S127 fix — **reverts** binary.npz: *"real survivor data, must stay in git"*. At that commit it was mode `100644`, a **regular 4,524-byte data file** |
| `ad5ab8d` | 2026-03-15 | S145-R1 v2 — adds the `!bidirectional_survivors_all.npz` negation, when that path too was a real tracked accumulator file |
| `46a3828` | 2026-07-25 | **S172 Phase 5 D3.5 — removes both paths from tracking** and replaces them with symlinks into the immutable chain-authenticated generation store |

The `006623c` ruling is real and must be acknowledged: binary.npz was deliberately restored to
tracking. **It was superseded by D3.5.** That ruling was about a regular file holding survivor
data; D3.5 changed what the path *is*, and the D3.5 commit itself untracked it. Under D3.5 the
system of record for survivor data is the generation chain (authenticated by the sidecar digest
in the directory name, validating recursively to a clean-start root), **not git**.

`optimal_window_config.json` and `watcher_decisions.jsonl` — untracked by the same S127 commit and
never reverted — remain ignored today (`.gitignore:115`, `:116`). That is the surviving pattern
for this class.

**Why not commit them:**

- committing a symlink stores a pointer that is stale the moment the next run swaps `current`, so
  every run would dirty the tree again — the failure this section exists to close;
- the target name is per-run and machine-local, so the committed pointer is meaningless in a fresh
  clone;
- `.s172_accumulator/` is the finalizer's own namespace. `.gitignore:48-51` already documents
  `.s172_checkpoint/` as *"never an artifact, never committed"* and names it as *"deliberately
  separate from the finalizer's `.s172_accumulator/`"* — the accumulator is the finalizer-owned
  sibling and was simply never given its own rule;
- §2.25 of the launch script records that Step 2 fails by `mv`-ing a regular file onto the
  D3.5 finalizer-owned symlink → `PublicationError`. Tracking these paths as git content invites
  exactly that regular-file-over-symlink confusion.

**Why they are not ignored today** — a gap, not a decision. Nothing ignores them
(`git check-ignore` returns no match for all three). `.gitignore:45`'s
`!bidirectional_survivors_all.npz` is **inert and stale**: it sits inside the `*.json` block, and
`*.json` cannot match a `.npz`, so it negates a rule that never applied. It dates from `ad5ab8d`,
four months before D3.5 changed what the path is. When D3.5 untracked the two paths and created
`.s172_accumulator/`, `.gitignore` was never updated to match.

**Recommended patch (NOT applied — Michael's call, nothing committed):**

```gitignore
# [S172 Phase-5 D3.5] finalizer-owned publication store. `generations/` holds the
# immutable chain-authenticated generations; `current` is the atomic pointer. The
# two repo-root NPZ entries are SYMLINKS into it, retargeted every run, so they are
# pointers and not artifacts. Supersedes 006623c, which restored a REGULAR survivor
# data file to tracking before D3.5 changed what these paths are.
.s172_accumulator/
bidirectional_survivors_all.npz
bidirectional_survivors_binary.npz
```

and delete the stale, inert `!bidirectional_survivors_all.npz` at `.gitignore:45`.

**Consequence to accept knowingly:** with this patch, survivor NPZ data is no longer in git at
all. It lives in `.s172_accumulator/generations/`, authenticated by the chain. That is the D3.5
design, but it does mean the box's generation store — not the repository — is what must be backed
up. If survivor data is wanted in git, that is a separate decision and it conflicts with the
symlink publication model rather than being achievable by committing these three paths.

Nothing was committed, ignored or deleted by the agent.
