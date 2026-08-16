# SESSION CHANGELOG — 2026-08-16 — MP-1 DRAIN ATTRIBUTION

**Brief:** `~/dashboard_work/CCODE_BRIEF_MP1_DRAIN_ATTRIBUTION_v1_0.md` v1.0
**Origin:** Team Beta ruling on the attempt-7 forensic, 2026-08-16 — **MEASUREMENT BEFORE REMEDY.**
**Report:** `~/dashboard_work/MP1_DRAIN_ATTRIBUTION.md` (staged for Michael, not committed)
**Host:** VM101, user `michael`, `~/venvs/torch`.

```
git status --porcelain   AT START   (empty)          HEAD 2c38f8cbe01e67cc66e7204bf4f35b09da5ed1d1
git status --porcelain   AT END      M miner/range_miner_coordinator.py
                                    ?? tests/test_s172_mp1_drain_attribution.py
```

**Nothing committed. Nothing pushed. Nothing deployed. Nothing launched.**

---

## What this session did

**READ-ONLY OBSERVATION ONLY.** No behaviour change, no remedy — including the candidate cause
identified in source (below). The attempt-7 forensic confirmed the drain-starvation MECHANISM and
explicitly did not claim a CAUSE; MP-1 makes the cause measurable.

Extended the attempt-6-certified `[S172-SL]` serve-loop seam (no parallel system) with:

1. **Three-level per-iteration attribution, every level summing to a NAMED remainder**
   * L1 `iteration = accept + admission + drain + deadline + stage_setup + schedule + dispatch +
     expiry + advance + loop_remainder`
   * L2 `drain = msg + drain_remainder`
   * L3 `msg = staging + pump + msg_remainder`   ← `staging` and `pump` are NEW
   * Per-iteration residual **total, maximum and the wall instant of the maximum** at each level,
     plus `remainder_negative_{loop,drain,msg}` so a clamped negative is counted, never hidden.
   * `iteration_max_parts` — the FULL per-segment profile of the single worst iteration, captured
     when it becomes the worst. Nine independent maxima can come from nine different iterations and
     cannot be read as a profile of any one of them.
2. **`PhaseCharge`** — a re-entrant, THREAD-KEYED phase charge reporting inclusive AND exclusive
   time. Exclusive time is what makes L3 partition when `_pump_deferred` nests inside
   `enqueue_staging` on one thread. Charges carry `thread_name`, so pump time paid by
   `miner-staging_*` is never summed into the serve loop's share (R1-A's rule, coordinator side).
3. **Per-connection drain service census** — per frame, the connection and its **1-based position
   within its drain pass**; per pass, frames, **distinct connections serviced** and **connections
   live**. The census is built from the LIVE CONNECTION SET, so *"connection X was never serviced"*
   is a measured `OK`/0 row with `None` positions, never an absent row. Three-valued:
   `OK` / `UNAVAILABLE` (live set unreadable → `live_count = None`, never 0) / `NO_OBSERVATION`
   (the `UNOBSERVED` bucket for frames whose socket was already reaped — named, never dropped).
4. **`[S172-SL] window`** — ONE structured record per 10 s (monotonic limit, ~25 rows, 0.1 lines/s)
   carrying phase DELTAS, the three remainders, the drain-pass census, the connection census and the
   frame-class census. A build-up is a derivative and needs a series; totals and maxima cannot show
   one. A final window is forced at trial terminal so the window containing the terminal event is
   not discarded.
5. **`[S172-SL] iteration_profile`** — one terminal JSON record: the worst iteration's parts map,
   the three remainder maxima with their instants, `phase_attribution`, `frame_classes_run`.
6. **Drain message-class census** — see the heartbeat finding below.
7. `[S172-SL] summary` gains 23 fields, **appended** to the same grep-stable line (57 fields). Every
   certified field keeps its name, value and position. `STRIPE_RX_SUMMARY` and `ACTIVE_STRIPES` are
   **unchanged** — they are the baseline that made attempt 7 legible.

---

## Findings reported, NOT fixed (per the brief)

### F-A — `heartbeats_accepted = 0` run-wide is currently UNINTERPRETABLE

Every existing heartbeat counter is keyed by STRIPE, and a heartbeat reaches a stripe's counters
only if it carries a `current_stripe_id`: `frame_stripe_id()` returns it, so an empty one makes
`note_stripe_frame_enqueued` return before counting; `note_stripe_frame_dequeued` returns on the
same condition; `_serve_dispatch`'s heartbeat branch guards its accounting with `if _hb_stripe:`.

**A heartbeat with no stripe id is invisible to the entire H1/H2 inventory — arrival, dequeue and
acceptance alike — and reads exactly like a heartbeat that was never sent.**

MP-1 adds a message-class census taken at DEQUEUE, keyed by class only, separating
`heartbeat_with_stripe` from `heartbeat_without_stripe`. **MP-1 does not claim which reading
obtains.**

**BOUNDARY (Beta certification correction, 2026-08-16): the with/without-stripe classification
separates "never sent" from "sent and uncounted" ONLY WHERE ARRIVAL-SIDE EVIDENCE EXISTS.** The
census observes frames that reached the DRAIN, so a drain-side zero is consistent with *never sent*
and with *sent, enqueued, never dequeued*. Four-way split, three decisive branches:

* `heartbeat_without_stripe > 0` → decisive; arrival proves the frames were sent, and the
  stripe-keyed inventory's blindness to them is a second, independent measurement defect.
* `heartbeat_with_stripe > 0` with `heartbeats_accepted` still 0 → decisive; the identity fence
  dropped them (already WARN-logged per drop).
* `heartbeat == 0` **with** `STRIPE_RX_SUMMARY.frames_enqueued > 0` → decisive; the same drain
  starvation, no second defect.
* `heartbeat == 0` **with no arrival-side evidence** — the stripeless class, for which
  `note_stripe_frame_enqueued` returns before counting, so the enqueue inventory is blind by
  construction → **UNDECIDABLE from coordinator artifacts alone; narrowed, not closed.** Resolving
  it needs the preserved rig-side worker logs, an external surface.

Beta's open question is therefore **closed as measurable within that stated boundary** — not
reported as closed outright.

### F-B — a candidate cause chain, every link anchored, none changed

`_try_admit_locked` (`miner/range_miner_coordinator.py:4744`) is a **serialization gate** — at most
ONE stripe attempt may be admitted to staging at a time; every other worker's sub-results defer.
**This is governed, Beta-approved design ("Correction 6 … Beta's recommended approach A", stated in
that function's own docstring), NOT a defect** — §0.4's standing rule applies.

The chain it makes available:

```
25 workers stream concurrently -> one attempt admitted -> the rest DEFER
 -> `_deferred` grows (attempt 7: deferred_high_water 247)
 -> `_pump_deferred` (:7631) runs on EVERY staging-job completion and, holding
    `_admission_lock`, scans the WHOLE deferred list with ~2 ledger queries per entry
    (`_attempt_live_locked`, :4681) -> its lock-held cost is O(len(_deferred)) and GROWS
 -> the serve loop takes that same lock for EVERY sub_stripe_result, inside
    `enqueue_staging` -> inside `msg` -> inside `drain`
 -> `msg` per frame grows -> fewer frames per pass -> inbound backlog grows
 -> late-index stripes never reached -> enqueued=45, dequeued=0, lease expiry
```

Consistent with every attempt-7 observable, **but consistency is not measurement** — which is
exactly why Beta ruled measurement before remedy. MP-1 measures each link separately
(`staging_total` on the serve thread · `phase_attribution` pump rows by thread · the window series ·
`msg_seconds_per_frame` · `drain_passes_partial` · per-connection `frames_window`/`position_min`).
**REFUTATION IS A CONJUNCTION (Beta certification correction, 2026-08-16): the chain is refuted only
if BOTH the serve thread's `staging` exclusive time AND the `pump` attribution on the
`miner-staging_*` threads stay small and flat.** A growing pump cost on the staging-executor threads
starves the drain through `_admission_lock` contention inside `msg` even while serve-thread
`staging` exclusive stays small — exclusive accounting gives each thread only its own share, so a
single-sided test would let the chain survive its own refutation. Both sides are measured and stay
separable (`staging_total`/`_max` per window vs `phase_attribution` rows keyed by `thread_name`).
No remedy proposed; the
candidates differ materially in concurrency properties and the choice is Beta's (§2.26 precedent).

---

## Gates

`tests/test_s172_mp1_drain_attribution.py` — **NEW, 38/38 green.** RED/scope arms pinned to the FULL
SHA `2c38f8cbe01e67cc66e7204bf4f35b09da5ed1d1`, with the anchor verified pre-MP-1 before any arm is
credited (probes both the ABSENCE of every MP-1 surface and the PRESENCE of the pre-MP-1
`NESTED_SEGMENTS = ('msg',)`); a drifted anchor terminates `AnchorUnavailable`, which never accepts.

Every measured field carries a clean control, a fault-injection control and a stated wrong-input
that reds it. Four production-class mutants (`phase_charge` nulled → `D1` red; `NESTED_SEGMENTS`
emptied; `close_current_iteration` neutralised; sub-phase deltas withheld → `msg_remainder`
2.5 → 10.0), each recovering.

```
ANCHOR-AUTHENTIC · A1 L1 sums · A2 L2/L3 sum · A3 declaration matches derived · A4 remainder named
not silent · A5 negative clamp counted · A6 profile is ONE iteration · A7 rate UNOBSERVED not zero
B1 counts vary · B2 zero is MEASURED · B3 UNAVAILABLE not zero · B4 unresolved is named
B5 position discriminates head/tail · B6 window resets, totals survive · B7 census detached (R3-1)
B8 partial coverage counted · B9 unknown live count -> verdict withheld · B10 stripeless heartbeat
visible · B11 class window vs run
C1 keyed by thread · C2 exclusive excludes nested · C3 cannot perturb its caller
D1 real dispatch charges (+mutant) · D2 delta read lands in _sl, L3 sums · D3 pump thread-keyed
D4 wiring in serve_trial (AST) · D5 window rate-limited · D6 window is a SERIES · D7 summary
additive · D8 summary never raises
E1 14 no-touch surfaces byte-identical · E2 AST scope proof · E3 no control flow reached
E4 monotonic only · E5 no new wall read in serve_trial · E6 never raises · E7 one line per window
E8 window carries the census
```

### Regression at FINAL state (sequential; concurrent S172 suites flake on a free-space race)

```
attempt-6 remediation      78/78   D6 integration   82/82   D6 liveness      59/59
GPU gate                    9/9    clean-tree       31/31   back-pressure    50/50
F1/F2 active lease         16/16   H1/H2            62/62   F1 lease-origin  18/18
MP-1 drain attribution     38/38   (NEW)
phase-4 coordinator        62/63   <- Gate 22 development-state red, see below
```

**Phase-4's single red is Gate 22**, which builds `changed_py` from `git status --porcelain`
including UNTRACKED files. `miner/range_miner_coordinator.py` is already inside the gate's allowlist
(`tests/test_s172_phase4_coordinator.py:2395`), so the production change passes; the only offender
is the new untracked harness. **The allowlist was NOT widened** — Beta rejected that permanently —
and the gate self-clears on a clean committed tree.

## AST scope proof

**11 changed, 23 added, 0 removed — exactly as declared in the harness** (declared in advance;
a proof that merely lists what changed proves nothing about intent).

Changed: `ServeLoopTiming.{__init__, tick, close_current_iteration, note_drain_stop, _record,
metrics}` · `RangeMinerCoordinator.{__init__, log_serve_loop_timing_summary, _serve_dispatch,
_pump_deferred, serve_trial}`.

**No-touch, byte-identical vs `2c38f8c` (14):** `claim_stripe` · `schedule_pending_stripes` ·
`renew_lease` · `_renew_active_lease` · `process_lease_expiry` · `_handle_stripe_failure_locked` ·
`_execution_set_expected_workers` · `enqueue_staging` · `_defer_locked` · `_conn_reader_loop` ·
`dispatch_inbound_result` · `_release_resume_credit_exact` · `_resume_paused_connections` ·
`staging_capacity_timeout_expired` — plus the §4.3 bounded-admission block as a subtree.

`enqueue_staging` stays byte-identical because the staging charge is at the CALL SITE in
`_serve_dispatch`, not inside a certified capacity surface. `compute_lease_timeout` untouched
(mask-not-fix, ruled twice). `time.time()` sites in `serve_trial`: **2 → 2, unchanged** (the R1.3
lesson: the first F1 implementation put the wall-clock read at the call site while the class-level
gate stayed green).

---

## Files

| file | state |
|---|---|
| `miner/range_miner_coordinator.py` | MODIFIED (+1028/−13); governed file — full pin + parity 30/30 required before the MP-1 run |
| `tests/test_s172_mp1_drain_attribution.py` | NEW, 38 gates |
| `docs/SESSION_CHANGELOG_20260816_MP1_DRAIN_ATTRIBUTION.md` | this file |
| `~/dashboard_work/MP1_DRAIN_ATTRIBUTION.md` | the deliverable report (not in the repo) |

`miner/range_miner_worker.py` is **untouched**.

**fallback parity:** code=[current], env=[ok] — no dependency changed; the instrument uses only
`threading`, `time.perf_counter` and `json`, all already imported by the module.

---

## Next (Beta's sequence, not started)

Michael reviews → Beta certifies → Michael commits + dual-pushes → clean tree → deploy the governed
file (full pin, parity 30/30) → fresh nonce → the MP-1 run: a real Gate-12 attempt under the frozen
§2.29 shape, which may PASS (§21 evidence gathered normally) or fail at the same wall (in which case
the attribution answers the cause question). **The remedy fork stays open until MP-1's run reports.**
