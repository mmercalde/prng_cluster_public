# CLAUDE CODE REPORT — S172 Staging Back-Pressure Remediation (B + D + A + C)

**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_S172_STAGING_BACKPRESSURE_REMEDIATION.md`
**Authority:** Team Beta ruling *"STAGING DEFERRED-QUEUE BACK-PRESSURE"* (2026-08-05), issued
against `docs/TEAM_ALPHA_DEFERRED_QUEUE_NOTE.md`.
**Host:** VM101 (`zeus-ubuntu-vm`, 192.168.3.177), repo `~/distributed_prng_analysis`, base
commit `7c4f11b`. All commands under `source ~/venvs/torch/bin/activate`.

**Nothing was committed, nothing was pushed, no pipeline was launched.**
**Order step 6 (the production-shape run) was NOT executed** — Beta's ruling forbids it until
these gates are reviewed. `tests/gate_s172_prod_shape.py` was not invoked.

---

## 0. Headline

| | |
|---|---|
| New suite `tests/test_s172_staging_backpressure.py` | **19/19 green** |
| Red-first discrimination | **18 of 19 red pre-fix**; the 19th is a non-regression control that MUST be green both sides (§4.1) |
| `tests/test_s172_staging_partb.py` (regression) | **24/24 green** |
| `tests/test_s172_phase4_coordinator.py` (regression) | **63/63 green** |
| Anchors in the brief | **all verified against live source; no drift** (§1) |
| Worker changes | **none were necessary** |

Three items are **flagged for Beta** and are called out where they arise: the §1.4 lease
exemption (made under the authority of gates 1–2, not on Alpha's initiative), the §1.5 default
of 600.0 s, and the §1.6 invariant disposition.

---

## 1. Anchor verification — read BEFORE a line was written

Every `file:line` the brief cites was opened in live source at `7c4f11b` first. **No anchor had
drifted.** Two are recorded precisely rather than silently accepted:

| brief anchor | live at `7c4f11b` | |
|---|---|---|
| `coordinator.py:2729` deferred-overflow `_on_staging_failed(..., True, ..., "staging deferred queue full — dispatch back-pressure")` | exactly, `:2729-2730` | ✅ |
| `:2721` D1b non-retryable (`False`) | exact | ✅ |
| `:2842` `StagingHashMismatch` · `:2846` `StagingTimeout` · `:2863` `StagingConfigurationError` · `:2868` generic | exact | ✅ |
| `:2060` StripeComplete reconciliation mismatch | exact | ✅ |
| `_conn_reader_loop:4137` | exact | ✅ |
| `_release_admission:2658` | exact | ✅ |
| `_serve_dispatch` heartbeat/lease renewal `:4275-4282` | renewal block `:4274-4282` | ✅ |
| `compute_lease_timeout = 300.0` at `:225` | exact | ✅ |
| `process_lease_expiry` `:3997` | `:3997` is the **serve-loop call site**; the definition is `:3099`. The brief's wording ("the serve loop's `process_lease_expiry` (`:3997`)") is correct as written | ✅ |
| constant-phase `fail_trial` `:3059-3066` | exact | ✅ |
| `expected_substripes_for:364` | exact | ✅ |
| `build_coordinator (:4684)` | `def` is at `:4682`; `:4684` is its first kwarg (`staging_dir`) — the brief says "`build_coordinator (:4684)` kwargs", which is accurate | ✅ |
| `range_miner_worker.py:1120-1126` `_sendall` — blocking loop, **no socket timeout** | exact; property confirmed | ✅ |
| `VramCaps`, `range_miner_worker.py:461-479` | class `:460-469`, `select_seed_cap` `:472-479` — the cited span covers both | ✅ |
| manifest key `:267` · kebab mapping `:56` | exact | ✅ |

**One naming discrepancy, reported not adapted around:** the brief names the target dataclass
`MinerCoordinatorConfig` (§3 route diagram, gate 10). **No such symbol exists anywhere in the
tree** — `/bin/grep -rn MinerCoordinatorConfig` hits only the brief itself. The live class is
**`CoordinatorConfig`** (`range_miner_coordinator.py:210`). Implemented against
`CoordinatorConfig`; G10 asserts on it by name.

---

## 2. Implementation-order compliance (1 → 5; 6 NOT run)

### Order step 1 — §0, the classification law (Beta D)

**Exactly ONE call site removed**, as specified. The deferred-overflow branch of
`enqueue_staging` no longer calls `_on_staging_failed` at all; it is replaced by the §1 pause
mechanism, with §1.6's invariant as its only terminal.

Proven surgical by **G-MATRIX-DIFF-a**, which parses the pre-change file
(`git show HEAD:miner/range_miner_coordinator.py`) and the working tree, extracts every
`self._on_staging_failed(...)` call through `ast.unparse` (normalized, so comments and line
moves cannot mask an argument change), and asserts:

* 7 call sites before, **6 after**;
* the single removed one is precisely the `"deferred queue full"` call;
* **no surviving call site differs** (`surviving_changed == []`);
* `_on_staging_failed`, `handle_stripe_failure` and `_handle_stripe_failure_locked` are
  **AST-identical** pre/post.

That last point drove a real design decision. The first implementation put the reader-resume
inside `_release_admission`, `_on_staging_failed` and the staging-job success tail — and
G-MATRIX-DIFF-a **caught it**, because `_on_staging_failed` is out of scope. The resume was
moved **inside `_pump_deferred`'s `finally`**, which is strictly better: every pre-existing
capacity-release caller becomes a resume point, in the "pump first, then resume" order Beta
asked for, with **zero edits to any out-of-scope method**.

**G-MATRIX-DIFF-b** then drives all six surviving classifications row by row against a frozen
expected table and asserts identical action, reason and retry accounting.

### Order step 2 — §1, per-connection pause/resume (B)

Implemented **in `_conn_reader_loop`**, per connection, with the C4 §1c rationale preserved in
the docstring. Only `sub_stripe_result` is gated; `register`/`heartbeat`/`stripe_complete`/
`stripe_error` pass through when they are the decoded frame.

* `resume_event: threading.Event` per paused connection; `pending_envelope` is a **scalar
  local** holding at most one already-decoded message.
* `staging_can_accept()` — True iff a staging slot is free **or**
  `len(_deferred) < bound − resume_margin`. Lock-free by design (Beta permits the approximate
  read; §2's margin covers the decode race).
* **Hysteresis:** pause at the bound, resume at `bound − resume_margin`, with
  `resume_margin = max(1, live connection count)` — documented as both the transition/
  already-decoded margin *and* the anti-thrash gap, because one envelope per connection is
  exactly how many can slip past the check.
* **Resume trigger:** `_pump_deferred`'s `finally` calls `_resume_paused_connections()`, which
  wakes readers in **pause-entry order (FIFO)**, one per confirmed capacity observation.
* **Per-connection, never global:** `_paused_connections` is keyed by raw socket; the accept
  loop and other readers are untouched. The serve loop reads pause state only for §1.4 and §1.5.

**Exactly-once and fencing (§1.3):** the held envelope is **not** dispatched while paused —
`record_substripe_result` runs only when the serve loop processes it after resume, so the
existing dedup insert and the existing L1 `accept_stripe_message` fence govern it unchanged.
**No second dedup layer was added**, and G6 proves it structurally: the reader loop calls none
of `record_substripe_result`, `accept_stripe_message`, `handle_stripe_failure`,
`_on_staging_failed`, `fail_trial`.

**One signature change:** `_conn_reader_loop` gained an optional `worker_by_sock=None`
parameter so a pausing reader can name the identity it is pausing — which is what §1.4 keys on.
Default `None` keeps every pre-existing caller valid (Phase-4 63/63 confirms).

### Order step 3 — §2, derived burst bound (A)

The constant **64 is deleted**. Two pure module-level functions, both resolving caps through
`advertised_effective_cap` (the coordinator's one cap path, which wraps the worker's own
`select_seed_cap`) so the two sides cannot diverge:

```
staging_burst_bound_exact(assignments)                     -> 116   (recorded assignment)
staging_burst_bound_conservative(slots, eligible_workers, phase, caps) -> 136   (4-slot AMD)
```

Runtime uses the **conservative** bound, installed by `derive_staging_deferred_bound(...)` at
stage setup immediately after `assign_stripes`, from the resolved execution set, the actual
stripe spans, the phase and per-worker caps:
`staging_deferred_bound = burst_bound_conservative + resume_margin`.

`staging_deferred_max` became `Optional[int] = None` — an **operator override** (None ⇒
derived). An override **below** the derived bound logs a WARNING naming both numbers (G9b
proves the warning names `64` and `138`).

**Deviation from the brief's literal signature, flagged:** the brief writes
`staging_burst_bound_conservative(slots, eligible_workers, phase, caps)` where `slots` reads as
a count — but the stripe span appears nowhere in that signature and cannot be invented. `slots`
is therefore **a sequence of per-slot stripe spans** (`[67_108_864] * 4` for the four-slot
case), which is also more honest: `partition_macro_stripes` can produce a final macro-stripe
shorter than `miner_stripe_size`, and the runtime call site passes the real spans. **A bare int
is refused with an explicit error** rather than paired with an assumed stripe size.

### Order step 4 — §3, configuration route (C)

All four controls wired through the complete route Beta specified, following Part B's
`staging_dir` pattern exactly:

```
agent_manifests/window_optimizer.json  default_params + args_map + param_docs
  -> window_optimizer.py  argparse flag -> call-site kwarg -> coordinator.<attr>
  -> window_optimizer_integration_final.py  getattr(coordinator, ...)
  -> run_trial_miner -> build_coordinator -> CoordinatorConfig
```

**Values stay at today's defaults** (`staging_workers=4`, `staging_queue_depth=2`) — Beta did
not rule a new number. `staging_deferred_max` ships as `null` (⇒ derived, the production shape;
WATCHER skips null-valued params so no flag is emitted, which is the intended behaviour and is
documented in `param_docs`). `staging_capacity_timeout` ships at `600.0`.

The manifest was edited with **targeted string edits only** — the diff is purely additive with
no escaped-unicode churn, so this is not a whole-file re-serialization (§7's `2389b61`
mechanism).

**A finding this surfaced.** Pre-fix, passing `staging_capacity_timeout=` (or any of the four)
to `run_trial_miner` was a **silent no-op**: `run_trial_miner` ends in `**kwargs`, so the
unknown keyword was swallowed and the run proceeded on the old defaults with no error. That is
the §2.15 dead-parameter shape in a fourth place, and it is why the pre-fix G11 run **hung
until `serve_trial timeout`** instead of failing on an unknown argument.

### Order step 5 — §4, metrics

Structured, grep-stable, every line prefixed `[S172-BP]`, through the existing logger:

`pause` · `resume` / `pause_aborted` / `reader_exit` · `derived_bound` · `burst_exact` ·
`lease_exempt` · `capacity_timeout` · `CAPACITY INVARIANT VIOLATED` · `summary`.

The trial-terminal `summary` carries: inbound-queue occupancy high-water · deferred occupancy
high-water vs the derived bound · paused-connection count/identities · per-pause duration and
cumulative pause time · staging jobs completed and jobs/sec over the trial · capacity-timeout
and capacity-invariant terminations. It is **also returned** on `serve_trial`'s result dict as
`staging_backpressure`, so the series is observable on the run and not only in a log.

### Order step 6 — **NOT RUN**

No production-shape trial. No fleet launch. `gate_s172_prod_shape.py` untouched.

---

## 3. Decisions flagged for Beta

### 3.1 §1.4 — the lease exemption (made under the authority of gates 1–2)

**The ruling is silent on leases; its gates are not satisfiable without this.** Heartbeats are
the only compute-lease renewal path (`_serve_dispatch:4275-4282`, `compute_lease_timeout =
300.0` at `:225`) and they ride the **same ordered TCP stream** as results. A
coordinator-paused connection therefore stops delivering renewals, and `process_lease_expiry`
routes the expiry into the matrix with `lease_expiry=True` — which **skips the non-retryable
branch** and lands on the constant-phase `fail_trial` (`:3059-3066`). **Any pause longer than
300 s would red Beta gates 1–2 through that door.**

Implemented: `process_lease_expiry` skips stripes whose claiming worker's connection is in
**coordinator-initiated pause**. Narrow by construction — membership in `_paused_connections`
is the only qualifier and only the coordinator writes it; the pause is itself bounded by §1.5;
an **unpaused** worker's genuine silence still expires.

**This is made under the authority of gates 1–2, not on Alpha's initiative. Beta may ratify or
amend it.** Evidence: **G-LEASE** (paused exempt, unpaused still expires, trial survives) plus
**G-MUT-LEASE**, which removes only the exemption, proves the mutated path executed, and shows
the credited assertion reds — the paused worker's expiry reaches the matrix and kills the trial
exactly as pre-fix.

### 3.2 §1.5 — the capacity-timeout default (600.0 s)

`staging_capacity_timeout: float = 600.0`, proposed as the same class as `staging_timeout`.
**Flagged for Beta**: this is a proposal, not a measurement. Measured from the **oldest**
currently-paused connection, so a long series of short pauses cannot trip it. On expiry the
coordinator calls **`fail_trial` directly** with
`coordinator_staging_capacity_timeout: staging did not release capacity within <T>s; N
connections paused (...)` — leading with the root cause (Part B convention), never through
`handle_stripe_failure`. The predicate is **latched**, so a reader thread and the serve loop can
never disagree about whether the bounded wait was exceeded.

### 3.3 §1.6 — coordinator-side overflow as an invariant

With the derived bound (≥ conservative burst + margin) and the reader pause holding traffic on
the wire, the serve-loop-side deferred-overflow branch is unreachable. If reached anyway, the
implementation logs `ERROR [S172-BP] CAPACITY INVARIANT VIOLATED` with the **full arithmetic**
(deferred, bound in force, derived bound, override, resume margin, conservative burst, slots,
phase, eligible workers, retained bytes, byte high-water, staging workers, queue depth) and
terminates via a direct `fail_trial(reason="coordinator_staging_capacity_invariant: ...")`.
**This disposition is Alpha's reading of Beta's "an invariant, not a matrix event"; flagged for
amendment.**

It is reachable today only by an operator override far below the derived bound — which is
exactly the case §2 warns about and then honours. `tests/test_s172_phase4_coordinator.py`
gate 56 does precisely that, and was updated accordingly (§5).

---

## 4. Gates — red-then-green evidence

Suite: `tests/test_s172_staging_backpressure.py`, CPU-only, modeled on
`tests/test_s172_staging_partb.py`. Gates run against the **real** per-connection reader loop
over **real** framed sockets (`socket.socketpair()` + `MinerFramedSocket`), the **real** staging
admission semaphore, the **real** serve loop (`run_trial_miner` with no `_serve` seam), and the
**real** ledger. No substitute reader, no stubbed predicate.

**GREEN (post-fix):** `/tmp/s172_bp_FINAL2.log` — `19/19 checks green`,
`COMPLETION SENTINEL: PASS`.

### 4.1 Red-first, per gate

The shipped suite imports symbols the pre-fix module lacks, so a bare pre-fix run dies at import
and proves only "a file changed". A driver
(`<scratchpad>/redfirst.py`) installs placeholders for the missing new surfaces so **every gate
executes and reds for its own reason**, run against the stashed pre-fix tree.

**RED evidence:** `/tmp/s172_bp_REDFIRST_FINAL.log` (and the earlier
`/tmp/s172_bp_REDFIRST.log`).

| gate | pre-fix red | why |
|---|---|---|
| G-MATRIX-DIFF-a | BEHAVIOURAL | `expected exactly 6 after the removal, found 7` — naming the deferred-queue call |
| **G-MATRIX-DIFF-b** | **green pre-fix — BY DESIGN** | it is the **non-regression control**: it asserts the six surviving classifications behave *identically*, so it must be green on both sides. A red here would mean the removal was not surgical. Recorded explicitly rather than fudged. |
| G-LAW | BEHAVIOURAL | `a retryable capacity failure is still routed through the matrix` |
| G1, G2, G3, G4, G5, G6, G7 | ABSENT | `CoordinatorConfig.__init__() got an unexpected keyword argument 'staging_capacity_timeout'` |
| G8 | BEHAVIOURAL | `no pending_envelope holder found in the reader loop` |
| G9 | ABSENT | `staging_burst_bound_exact does not exist` |
| G9b | ABSENT | `'RangeMinerCoordinator' object has no attribute 'derive_staging_deferred_bound'` |
| G10 | BEHAVIOURAL | `hop 1a: manifest default_params lacks staging_workers` |
| G11 | BEHAVIOURAL | **`serve_trial never terminated`** — the exact hang the bounded timeout exists to close, because the pre-fix `**kwargs` swallowed the knob |
| G-LEASE | ABSENT | as G1 |
| G-MUT-PAUSE, G-MUT-LEASE | ABSENT | as G1 |
| G-METRICS | ABSENT | as G1 |

**18 of 19 discriminate.** The 19th is the control described above.

### 4.2 Beta §5 mapping

| gate | proves | result |
|---|---|---|
| **G1** | saturating staging on a **phase-1** stripe fails nothing: no `_handle_stripe_failure_locked` entry, no retry consumed, no cancellation, no L1 fence against the valid attempt, trial alive | ✅ |
| **G2** | zero retry budget consumed — on **phase 3**, where a retry actually exists; `current_attempt` and `phase_degraded` unchanged, and the Q3 reassignment is still available afterwards | ✅ |
| **G3** | the paused peer stalls (0 frames delivered) while a second connection's heartbeat + stripe_complete flow to completion; and nothing A wrote is lost — all three frames arrive **in order** on resume | ✅ |
| **G4** | a **real** capacity release resumes it: the holder is a genuine `enqueue_staging` submission whose fetch blocks, released by `_submit_with_slot`'s own completion callback. Nothing external pokes the coordinator | ✅ |
| **G5** | every accepted sub-stripe staged exactly once across pause/resume — 4 ledger shard rows, `sub_index` 0..3 | ✅ |
| **G6** | no duplicate rows (same logical shard delivered twice ⇒ 1 row), no stale acceptance, and **no second dedup layer** (AST over the live reader) | ✅ |
| **G7** | a superseded attempt (`staging_generation`++ while paused) cannot resume-and-publish: fence drops it, 0 shard rows for the dead attempt, nothing published | ✅ |
| **G8** | `_deferred ≤ derived bound + margin`, retained bytes ≤ high-water, and **≤ 1 envelope per connection** — structurally (the holder is a scalar, never a container) and behaviourally (12 frames across 2 connections ⇒ exactly 2 pause records) | ✅ |
| **G9** | `staging_burst_bound_exact` = **116**; `staging_burst_bound_conservative` = **136**; plus the distinction asserted (conservative dominates exact), phase-awareness (hybrid bounds higher), a CUDA-only pool bounding lower (so the max-over-workers term is real), and the constant 64 gone from `_defer_locked`'s executable body | ✅ |
| **G9b** | the runtime bound is `136 + margin(2) = 138`, derived from live state; an override of 64 is honoured **and warns naming both numbers** | ✅ |
| **G10** | full manifest→`CoordinatorConfig` route for all four controls, injected end-to-end and observed; plus proof the values are **load-bearing** (semaphore sized `7+5`, bound `321`, timeout `42.5`) | ✅ |
| **G11** | forced capacity timeout terminates with a reason **leading** `coordinator_staging_capacity_timeout:`, and `_handle_stripe_failure_locked` was **never entered on the way to that decision** | ✅ |
| **G12** | fleet, production-shape — **NOT RUN** (Michael-initiated only, after Beta review) | ⏸ |
| **G-LEASE** | a pause past `compute_lease_timeout` does not expire the paused worker's lease; an unpaused worker's genuine silence still does | ✅ |
| **G-MATRIX-DIFF** | the six out-of-scope callers behave byte-identically (AST + behavioural) | ✅ |
| **G-MUT-PAUSE / G-MUT-LEASE** | mutation evidence — each mutant is proven to **execute** and to **red the credited assertion** | ✅ |
| **G-LAW / G-METRICS** | classification law asserted structurally; the `[S172-BP]` series complete and grep-stable | ✅ |

### 4.3 Two measurement defects found and fixed in the gates themselves

Both are worth recording because each would have produced a **falsely green** gate:

1. **G-METRICS was measuring nothing.** The `[S172-BP]` series is emitted at INFO; the module
   logger inherits the root's WARNING, so a handler alone captured only the warnings. The first
   run reported the captured line kinds as `['operator']` — i.e. it saw the override warning and
   no pause/resume/summary at all. Fixed with a `_capture_bp` helper that raises the logger
   level and restores it.
2. **G11 was dying of the wrong thing, twice.** First the loopback worker advertised default
   caps against a coordinator configured with `seed_cap_nvidia=10`, so `_validate_caps`
   **quarantined** it at registration and the trial died of admission, not capacity. Then, with
   that fixed, the reader outran the serve loop and read all four results while the gate was
   still open, so nothing paused and the §1.6 **invariant** fired instead of the §1.5 timeout.
   Fixed by advertising matching caps and by pacing the worker's sends (a real GPU worker
   computes between sub-stripes). G11 now additionally asserts
   `capacity_invariant_terminations == 0`, so it can never again pass on the wrong mechanism.

Also hardened: G4 and G11 use `TemporaryDirectory(ignore_cleanup_errors=True)`, because staging
threads may still be writing at block exit and a teardown `OSError` would **replace** a real
assertion failure with `Directory not empty` — which is exactly what masked G11's true pre-fix
red in the first red-first run.

### 4.4 Regression

| suite | result | log |
|---|---|---|
| `tests/test_s172_staging_partb.py` | **24/24 green**, sentinel PASS | `/tmp/bp_regress_partb_v2.log` |
| `tests/test_s172_phase4_coordinator.py` | **63/63 green** | `/tmp/bp_regress_phase4_v3.log` |
| `tests/test_s172_staging_backpressure.py` | **19/19 green**, sentinel PASS | `/tmp/s172_bp_FINAL2.log` |

**Phase-4 Gate 22 note.** With the new suite present as an untracked file, Gate 22 reds with
`unexpected changed .py files: {'tests/test_s172_staging_backpressure.py'}`. This is the known
untracked-`.py` sensitivity — **expected during development, not a regression, and not a reason
to widen Gate 22; the answer is to commit the file.** Verified by removing the new file and
re-running: **63/63, all checks green.** HEAD before any of this work was also 63/63, so the
one behavioural gate change (gate 56, §5) is the only Phase-4 delta.

---

## 5. One pre-existing gate updated, deliberately

`tests/test_s172_phase4_coordinator.py::gate56_bounded_deferred_queue` asserted, verbatim:

```python
# excess was back-pressured via the matrix (hybrid reassign) — trial runs on.
assert coord.ledger.get_trial("run")["state"] == "running"
```

**That is precisely the contract Beta D removes.** The bound itself is unchanged and still
proven by the gate; what changed is the **disposition of an overflow**. The gate now asserts the
§1.6 disposition — trial aborted, `capacity_invariant_terminations >= 1` — **and additionally**
that no stripe consumed a retry or was marked `phase_degraded`, which is what "did not enter the
matrix" looks like from the ledger. Its docstring records what changed and why, and notes that
the branch is reachable there only because the gate sets an explicit override far below the
derived bound.

No other pre-existing gate was touched.

---

## 6. The derived-bound arithmetic for the recorded assignment

From `docs/TEAM_ALPHA_DEFERRED_QUEUE_NOTE.md` §2, measured from
`/home/michael/miner_staging/miner_ledger.db` — not inferred:

```
stripe_span = miner_stripe_size = 67,108,864

ROCm  (--seed-cap-amd 2,000,000):     ceil(67,108,864 / 2,000,000) = 34
CUDA  (--seed-cap-nvidia 5,000,000):  ceil(67,108,864 / 5,000,000) = 14

EXACT (the assignment that actually happened, 3 x ROCm + 1 x CUDA):
    34 + 14 + 34 + 34 = 116          <- staging_burst_bound_exact

CONSERVATIVE (4 simultaneously admitted slots, any AMD worker eligible for each):
    4 x max(34, 14) = 4 x 34 = 136   <- staging_burst_bound_conservative

RUNTIME bound = 136 + resume_margin
    resume_margin = live connection count (2 in G9b)  -> 138
    the old constant was 64; the run died with 65 pending shards, to the unit.
```

**116 vs 136 is preserved verbatim in code comments (`range_miner_coordinator.py`, the §2
block), in the manifest `param_docs`, and in G9**, which asserts both numbers, asserts the
per-stripe decomposition `[34, 14, 34, 34]` (so four equal terms cannot coincidentally sum to
116), and asserts that the conservative bound **dominates** the exact one.

---

## 7. Files changed — COMPLETE

Michael: build the `git add` list from **this section**, never from recall.

### Modified (5)

| file | what changed |
|---|---|
| `miner/range_miner_coordinator.py` | §0 removal of the deferred-overflow `_on_staging_failed` call site; §1 pause/resume in `_conn_reader_loop` (+ optional `worker_by_sock` param) and the pause registry/`staging_can_accept`/resume/metrics block; §1.4 lease exemption in `process_lease_expiry`; §1.5 `staging_capacity_timeout` config + latched predicate + reason builder + serve-loop terminal check; §1.6 invariant branch; §2 `phase_family_probe` / `applicable_seed_cap` / `staging_burst_bound_exact` / `staging_burst_bound_conservative` / `derive_staging_deferred_bound` / `staging_deferred_bound` and `staging_deferred_max` → `Optional[int] = None`; `_defer_locked` now consults the derived bound; `_pump_deferred` resumes paused readers in its `finally`; §3 four kwargs on `build_coordinator` and `run_trial_miner`; §4 metrics + `staging_backpressure` on the serve result; `OrderedDict` import |
| `window_optimizer.py` | §3 hop 2 — four params on the run function, four `coordinator.<attr>` assignments, four argparse flags, four call-site kwargs, one `[S172-BP]` startup print |
| `window_optimizer_integration_final.py` | §3 hop 3 — four `getattr(coordinator, ...)` reads on the `run_trial_miner` call |
| `agent_manifests/window_optimizer.json` | §3 hop 1 — four `default_params`, four `args_map` kebab entries, four `param_docs` entries. **Purely additive; targeted edits, no re-serialization** |
| `tests/test_s172_phase4_coordinator.py` | gate 56 updated to the §1.6 disposition (§5 above) — the only pre-existing gate touched |

### Added (2, untracked)

| file | |
|---|---|
| `tests/test_s172_staging_backpressure.py` | the new acceptance suite (19 gates). **Committing this also clears the Phase-4 Gate 22 red** (§4.4) |
| `docs/CLAUDE_CODE_REPORT_S172_STAGING_BACKPRESSURE.md` | this report |

The brief itself, `docs/CLAUDE_CODE_INSTRUCTIONS_S172_STAGING_BACKPRESSURE_REMEDIATION.md`, is
also untracked and presumably belongs in the same commit.

### Explicitly NOT changed

* `miner/range_miner_worker.py` — **no worker change was necessary.** The property the design
  depends on (`_sendall:1120-1126` is a blocking loop with no socket timeout, so a full TCP
  buffer parks the worker's mining thread mid-`_send`) is already true.
* Seed caps and stripe geometry — **E is REJECTED**; nothing touched them.
* The six out-of-scope `_on_staging_failed` callers and the matrix itself — AST-identical,
  gated.
* `tests/gate_s172_prod_shape.py` — not run, not edited.

---

## 8. Verification-integrity controls (VIR-1…6)

* **execution proof:** every suite writes a `COMPLETION SENTINEL` line and a `N/M checks green`
  count; logs at `/tmp/s172_bp_FINAL2.log`, `/tmp/bp_regress_partb_v2.log`,
  `/tmp/bp_regress_phase4_v3.log`, `/tmp/s172_bp_REDFIRST_FINAL.log`. Gates were run with
  `python3 -u ... | tee`, never piped to `tail`.
* **clean control:** all three suites green with the change in place; Phase-4 was **63/63 at
  `7c4f11b` before any edit** (measured, by stashing), so the single behavioural delta is
  attributable.
* **fault-injection control:** 18 of 19 new gates proven red pre-fix, each with its own recorded
  reason (§4.1). Plus two live mutants (G-MUT-PAUSE, G-MUT-LEASE) that prove the mutated path
  **executed** and **reds the credited assertion** — the brief's mutation-evidence rule.
* **completion sentinel:** `COMPLETION SENTINEL: PASS — S172 staging back-pressure CPU gates
  green`.
* **unavailable-observer behavior:** G12 / G-PROD-SHAPE reports **NOT RUN**, never PASS. The
  suite's own footer states it and names why.
* **audit claim scope:** the coordinator staging back-pressure path and its configuration route,
  CPU-only, on VM101 at `7c4f11b`. **No claim is made about production-shape behaviour** — that
  is order step 6, and it has not run.
* **searched surfaces:** `miner/range_miner_coordinator.py` · `miner/range_miner_worker.py` ·
  `miner/range_miner_protocol.py` · `window_optimizer.py` ·
  `window_optimizer_integration_final.py` · `agent_manifests/window_optimizer.json` (live file —
  gitignored, read directly) · `tests/test_s172_staging_partb.py` ·
  `tests/test_s172_phase4_coordinator.py` · `git show HEAD:miner/range_miner_coordinator.py` ·
  `git status --porcelain`.
* **governance trail searched:** `docs/CLAUDE_CODE_INSTRUCTIONS_S172_STAGING_BACKPRESSURE_REMEDIATION.md`
  (read in full first) · `docs/TEAM_ALPHA_DEFERRED_QUEUE_NOTE.md` (read in full) ·
  `docs/CLAUDE_CODE_INSTRUCTIONS_STAGING_DIR_PART_B.md` (via the Part B suite it governs) ·
  `docs/` listing swept for back-pressure / deferred-queue artifacts. The `tfm-project-facts`
  skill was loaded before any code was read.
* **unavailable surfaces:** the live 25-daemon fleet and the CT100 rigs (order step 6 forbidden
  this session) · Proxmox host kernel logs on `.121`/`.155`/`.163` · `dmesg` on all hosts ·
  `docs/CLAUDE_CODE_CORRECTION4_S172_PHASE4_OVERLOAD.md`, `TB_BINDING_RULINGS_S172_PHASE4.md`
  and `TEAM_ALPHA_REVIEW_S172_PHASE4_REV5.md` were **not re-read this session** — they are
  quoted here only as the deferred-queue note quotes them, and are relayed as its claims, not
  re-derived as mine (§1.2).
* **chapters searched:** none were relevant to a coordinator transport/capacity change; none is
  claimed.

**Not established:** whether anything downstream of the staging queue works end to end. The
deferred-queue note's §0 caveat still stands — *the trial has still not completed* — and only
order step 6 can retire it.

---

## 9. Where this disagrees with the brief

Reported rather than improvised around, per the brief's closing instruction:

1. **`MinerCoordinatorConfig` does not exist** (§3 route, gate 10). Implemented against
   `CoordinatorConfig`. — §1.
2. **`staging_burst_bound_conservative(slots, ...)`** cannot derive a stripe span from the
   stated signature. `slots` is implemented as a sequence of per-slot spans; a bare int is
   refused explicitly. — §2, order step 3.
3. **§0's "every other `_on_staging_failed` caller must be byte-identical"** is stated about
   *callers*. G-MATRIX-DIFF-a additionally holds `_on_staging_failed` itself byte-identical,
   which is stricter — and it caught a first implementation that would have edited it. Recorded
   in case Beta intended only the call sites. — order step 1.
4. **`build_coordinator` is at `:4682`, not `:4684`** (`:4684` is its first kwarg); the
   definition of `process_lease_expiry` is at `:3099`, while `:3997` is its serve-loop call
   site. Both brief citations read correctly as written; noted for precision only. — §1.
