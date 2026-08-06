# CLAUDE CODE INSTRUCTIONS — S172 Staging Back-Pressure Remediation (B + D + A + C)

**Authority:** Team Beta ruling *"STAGING DEFERRED-QUEUE BACK-PRESSURE"* (2026-08-05), issued
against `docs/TEAM_ALPHA_DEFERRED_QUEUE_NOTE.md`. The ruling is **binding**, including its
implementation order (§4) and its twelve acceptance gates (§5).

**Host:** VM101 (`zeus-ubuntu-vm`, 192.168.3.177), repo `~/distributed_prng_analysis`.
`source ~/venvs/torch/bin/activate` before every test command.
Long suites: `python3 -u <suite> | tee /tmp/<name>.log` — **never pipe to `tail`**.

**Hard constraints:**
- **NO `git commit`, NO `git push`, NO pipeline launch.** Michael commits and dual-pushes.
- **No production-shape trial.** Beta: *"Do not commit or launch another production-shape trial
  until the new gates are green and submitted for Beta review."* Order step 6 happens only after
  Beta reviews the report this brief requires.
- **E is REJECTED.** No seed-cap or stripe-geometry change of any kind, for any reason.
- Read every anchor below in live source **before** writing a line. If any anchor has drifted
  from what this brief states, STOP and report the drift instead of adapting silently.

**Files in scope:**
- `miner/range_miner_coordinator.py` (primary)
- `miner/range_miner_worker.py` (read-only reference — no worker changes expected; report if one
  becomes necessary rather than making it)
- `window_optimizer.py`, `window_optimizer_integration_final.py`,
  `agent_manifests/window_optimizer.json` (C wiring route)
- `tests/test_s172_staging_partb.py` (pattern reference), new suite
  `tests/test_s172_staging_backpressure.py`
- Report: `docs/CLAUDE_CODE_REPORT_S172_STAGING_BACKPRESSURE.md`

---

## 0. The classification law (Beta D — binding, implement FIRST)

> **Coordinator staging back-pressure is a waiting state, not a stripe failure state.**
> No capacity-wait event may enter the phase-specific worker retry matrix.

**Order step 1 touches exactly ONE call site.** The deferred-overflow branch of
`enqueue_staging` — currently
`self._on_staging_failed(run_id, stripe_id, True, eligible_provider, "staging deferred queue
full — dispatch back-pressure")` (`range_miner_coordinator.py:2729`) — is **removed** and
replaced by the pause mechanism of §1.

**Every other `_on_staging_failed` caller is OUT OF SCOPE and must be byte-identical after this
work.** Justification, so nobody "improves" them:

| caller | event | why it stays |
|---|---|---|
| `:2721` | D1b — attempt cannot fit `staging_high_water_bytes` at all | a **permanent configuration impossibility**, not a transient wait; Part B's narrow non-retryable precedent governs |
| `:2842` | `StagingHashMismatch` | worker-output defect; C2:130-132 + documented Defect-5 ruling |
| `:2846` | `StagingTimeout` | staging-job failure, matrix-governed per C2 |
| `:2863` | `StagingConfigurationError` | Part B binding ruling, *"narrow by construction"* |
| `:2868` | generic transient fetch/IO | C2 retryable class |
| `:2060` | StripeComplete reconciliation mismatch | definitive structural worker-output failure |

**The one permitted trial-terminal path for capacity** is a bounded infrastructure timeout
(§1.5): `fail_trial(run_id, reason="coordinator_staging_capacity_timeout: ...")` called
**directly** — never through `handle_stripe_failure` / the matrix. (Beta §1: *"the terminal
reason must be a coordinator/infrastructure condition"*.)

Prove step 1 by a gate that saturates staging on a phase-1 stripe and asserts:
no `_handle_stripe_failure_locked` entry, no retry consumed, no cancellation, no L1 fence
activation against the valid attempt, trial not failed (Beta gates 1, 2).

## 1. B — per-connection pause/resume (order step 2)

### 1.1 Where the gate lives, and why

The pause is implemented **in `_conn_reader_loop`** (`:4137`), per connection. Rationale you
must preserve in comments: by the time `enqueue_staging` discovers saturation, the payload is
already decoded into coordinator RAM; gating at the reader keeps subsequent payloads **on the
wire / at the worker** (C4 §1c's stated property — the worker's `_sendall` at
`range_miner_worker.py:1120-1126` is a blocking loop with no socket timeout, so a full TCP
buffer parks the worker's mining thread harmlessly mid-`_send`).

### 1.2 Mechanism

Per accepted connection, alongside the existing reader thread state:
- `resume_event: threading.Event` (initially set)
- `pending_envelope: Optional[msg]` — capacity **exactly one** (Beta: *"retaining at most one
  bounded pending envelope per connection is acceptable"*)

Reader loop becomes:

```
msg = cfs.recv_msg()
if msg.message_type == "sub_stripe_result" and not coordinator.staging_can_accept():
    hold msg as pending_envelope
    pause_started = now; register (worker/conn) in coordinator._paused_connections
    wait on resume_event (loop with short timeout, honoring reader_stop and
        the capacity timeout of §1.5)
    on resume: deregister, put pending_envelope to inbound, clear it
else:
    put msg to inbound   (unchanged path)
```

- **Only `sub_stripe_result` is gated.** `register`/`heartbeat`/`stripe_complete`/
  `stripe_error` pass through *when they are the decoded frame*. (TCP is ordered: frames queued
  **behind** a held result stay on the wire — that is the point, and it is why §1.4 exists.)
- `staging_can_accept()` — new coordinator method: True iff a staging slot is free **or**
  `len(_deferred) < staging_deferred_bound − resume_margin`. Read under `_admission_lock` or as
  an approximate lock-free read — either is acceptable because §2's bound carries an explicit
  per-connection margin covering decode races (Beta: *"a documented bounded margin for
  transition and already-decoded messages"*).
- **Hysteresis:** pause at the bound, resume at `bound − resume_margin` (default
  `resume_margin = live connection count`), to prevent pause/resume thrash. Document the
  constant.
- **Resume trigger:** every capacity-release point that already calls `_pump_deferred`
  (`_release_admission:2658`, `_submit_with_slot` completion path) additionally sets the
  `resume_event` of paused connections **after** pumping, when `staging_can_accept()` holds.
  Fairness: resume in pause-entry order (FIFO over `_paused_connections`).
- **Per-connection, never global** (Beta B.3): pausing one reader must not touch any other
  connection's reader, the accept loop, or the serve loop. The serve loop remains single-
  threaded and untouched by pause state except for §1.4 and §1.5 checks.

### 1.3 Exactly-once and fencing across pause/resume (Beta gates 5–7)

The held envelope has **not** been dispatched: `record_substripe_result` runs only when the
serve loop processes it after resume, so the existing dedup insert and the existing L1
`accept_stripe_message` fence govern it unchanged. **Do not add a second dedup layer.** If the
attempt was legitimately superseded or cancelled while paused (`staging_generation` moved), the
existing fence drops the envelope on resume — that is correct and is Beta gate 7. Gate it.

### 1.4 Lease exemption — REQUIRED BY BETA'S OWN GATES, and flagged for ratification

**The ruling is silent on leases; its gates are not satisfiable without this.** Heartbeats are
the only compute-lease renewal path (`_serve_dispatch:4275-4282`,
`compute_lease_timeout = 300.0` at `:225`) and they ride the **same ordered TCP stream** as
results. A paused connection therefore stops delivering renewals, and the serve loop's
`process_lease_expiry` (`:3997`) routes the expiry into the matrix with `lease_expiry=True` —
which skips the non-retryable branch and lands on the constant-phase `fail_trial`
(`:3059-3066`). Any pause longer than 300s reds Beta gates 1–2 through this door.

**Therefore:** `process_lease_expiry` / its `state='claimed'` scan must **skip** stripes whose
claiming worker's connection is currently in **coordinator-initiated pause** (membership in
`_paused_connections`), because the coordinator caused the silence and knows it. The pause is
itself bounded by §1.5, so this cannot exempt a stripe forever. On resume, queued heartbeats
process normally and renewal restarts.

**Record this design decision prominently in the report** so Beta can ratify or amend the
mechanism — it is made under the authority of gates 1–2, not on Alpha's initiative.

### 1.5 The bounded capacity timeout (Beta §1 + gate 11)

New config `staging_capacity_timeout: float` (default: propose 600.0, same class as
`staging_timeout`; flag the default for Beta). Measured from the **oldest** currently-paused
connection's pause-entry time. On expiry, the coordinator terminates the trial via **direct**
`fail_trial(run_id, reason="coordinator_staging_capacity_timeout: staging did not release
capacity within <T>s; N connections paused")` — the reason string leads with the root cause
(Part B convention) and the event **never** enters the matrix. Paused readers observe
`reader_stop`/trial-terminal and exit; held envelopes are discarded (the trial is terminal).

### 1.6 Coordinator-side overflow after B — an invariant, not a matrix event

With §2's derived bound (≥ burst + margin), the serve-loop-side deferred-overflow branch is
mathematically unreachable. If it is reached anyway, sizing was wrong: log ERROR with the full
arithmetic and terminate via direct
`fail_trial(reason="coordinator_staging_capacity_invariant: ...")` — infra classification,
never the matrix. Flag this disposition for Beta in the report.

## 2. A — derived burst bound (order step 3)

Delete the meaning of the constant `64`. Implement **two pure functions** (module level,
unit-testable, no I/O):

```
def staging_burst_bound_exact(assignments) -> int
    # Σ over actual (stripe_span, worker) assignments:
    #     ceil(stripe_span / applicable_seed_cap(worker, phase))
    # MUST return 116 for the recorded 2026-08-05 assignment
    # (34 + 14 + 34 + 34: three ROCm cap 2,000,000 + one CUDA cap 5,000,000
    #  over stripe_span 67,108,864).

def staging_burst_bound_conservative(slots, eligible_workers, phase, caps) -> int
    # Beta's formula: Σ over simultaneously admitted stripe slots:
    #     max over workers eligible for that slot:
    #         ceil(stripe_span / applicable_seed_cap(worker, phase))
    # MUST return 136 for the conservative four-slot all-AMD-cap case.
```

- `applicable_seed_cap` must honor the phase (hybrid caps are tighter — `VramCaps`,
  `range_miner_worker.py:461-479`; use the coordinator's existing
  `advertised_effective_cap` path so the two sides cannot diverge, cf.
  `expected_substripes_for:364`).
- **Runtime uses the conservative bound**, computed at trial/stage setup from the resolved
  execution set, stripe geometry, phase and per-worker caps —
  `staging_deferred_bound = burst_bound_conservative + resume_margin` (the documented
  transition/decoded-envelope margin, §1.2). **Never a hand-maintained constant** (Beta A).
- `staging_deferred_max` in the dataclass becomes an **optional operator override** of the
  derived value (None ⇒ derived). If overridden below the derived bound, log a WARNING naming
  both numbers.
- **Preserve Beta's 116-vs-136 distinction verbatim in code comments and tests** — 136 is the
  conservative pre-assignment bound, 116 the exact count for that heterogeneous assignment.
  Beta singled this out.

## 3. C — configuration route (order step 4)

Wire **`staging_workers`**, **`staging_queue_depth`**, **`staging_deferred_max`** (as the
override of §2) and **`staging_capacity_timeout`** through the complete route Beta specified:

```
agent_manifests/window_optimizer.json default_params
    → window_optimizer.py call site
    → window_optimizer_integration_final.py
    → build_coordinator (:4684) kwargs
    → MinerCoordinatorConfig
```

Follow the exact pattern Part B used for `staging_dir` (manifest key `:267`, kebab mapping
`:56`). **Values stay at today's defaults** (`staging_workers=4`, `staging_queue_depth=2`):
Beta explicitly did **not** rule a new number — *"tune after measurement."* The deliverable is
reachability, proven end-to-end (gate 10), not new values.

## 4. Metrics (order step 5)

Through the existing logger, structured and grep-stable, emitted at pause/resume/completion
and in the trial-terminal summary: inbound-queue occupancy high-water · deferred occupancy
high-water vs derived bound · paused-connection count and identities · per-pause duration and
cumulative pause time · staging jobs completed/sec over the trial · capacity-timeout
terminations. Prefix every line `[S172-BP]` so `gate_s172_prod_shape.py` and operators can
extract the series.

## 5. Acceptance gates — new suite `tests/test_s172_staging_backpressure.py`

CPU-only, modeled on `tests/test_s172_staging_partb.py` (real relationships, never fabricated
values; every negative gate asserts on error text and type; **every gate proven RED against
pre-fix behavior first** — the Part B discipline). Map Beta §5 one-to-one:

| gate | proves |
|---|---|
| G1 | saturating staging during a phase-1 stripe fails nothing (no matrix entry, trial alive) |
| G2 | capacity wait consumes zero retry budget (`current_attempt` unchanged, `phase_degraded` unchanged) |
| G3 | paused connection's peer stalls while a second connection's traffic flows to completion |
| G4 | capacity release resumes the paused connection without operator action |
| G5 | every accepted sub-stripe staged exactly once across pause/resume (ledger row count) |
| G6 | no duplicate ledger rows, no stale-attempt acceptance across pause/resume |
| G7 | a superseded attempt (`staging_generation`++ while paused) cannot resume-and-publish — fence drops it |
| G8 | deferred retained bytes and pending envelopes bounded: ≤ derived bound + margin, ≤ 1 envelope/connection |
| G9 | `staging_burst_bound_exact` = **116** for the recorded assignment; `staging_burst_bound_conservative` = **136** for the four-slot AMD-cap case |
| G10 | full manifest→coordinator route proven for all four §3 controls (value injected at the manifest is observed in `MinerCoordinatorConfig`) |
| G11 | forced capacity timeout terminates with reason leading `coordinator_staging_capacity_timeout`, and `_handle_stripe_failure_locked` was never entered |
| G12 | (fleet — `gate_s172_prod_shape.py`, **Michael-initiated only, after Beta review**) production-shape run proceeds past this boundary with no `MinerIngressError` from missing phase completion |

Additional gates this brief requires beyond Beta's list:
- **G-LEASE**: a pause held past `compute_lease_timeout` does **not** expire the paused
  worker's lease (§1.4), and an *un*-paused worker's genuine silence still does (the exemption
  is narrow).
- **G-MATRIX-DIFF**: the six out-of-scope `_on_staging_failed` callers of §0 behave
  byte-identically pre/post (drive each; assert identical matrix outcome), so the removal is
  proven surgical.
- Mutation evidence rule applies: for at least the pause gate and the lease gate, prove the
  mutant (e.g., gate removed / exemption removed) actually executes the mutated path and reds
  the credited assertion — not merely that a file changed.

## 6. Report

`docs/CLAUDE_CODE_REPORT_S172_STAGING_BACKPRESSURE.md`, containing: implementation-order
compliance (1→5, with 6 explicitly NOT run); every gate's red-then-green evidence with log
paths; the §1.4 lease-exemption design decision and §1.5/§1.6 default/disposition flags for
Beta; the derived-bound arithmetic for the recorded assignment; a **complete "Files changed"
section** (Michael builds the `git add` list from it — never from recall); and a VIR
declaration naming searched and unavailable surfaces. One review round is the target: read
every anchor before drafting, and if the code disagrees with this brief anywhere, report the
disagreement — do not improvise around it.
