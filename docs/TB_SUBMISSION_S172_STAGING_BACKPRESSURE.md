# TEAM ALPHA → TEAM BETA — SUBMISSION FOR GATE REVIEW
## S172 Staging Back-Pressure Remediation (per Beta ruling of 2026-08-05)

**Requested disposition:** approval of the remediation and its gates, plus rulings on the four
consolidated items in §4, so the work can be committed and order step 6 (the production-shape
rerun) can be authorized.

**Status:** implementation complete in the mandated order (steps 1–5); step 6 NOT run; no
trial launched. **Disclosure:** the remediation was committed at `4b1aad6` and dual-pushed at
the owner's direction ahead of this review — a sequence deviation from your hold (*"do not
commit … until the new gates are green and submitted for Beta review"*), recorded in the
commit message itself. Fix-forward applies to any amendment your review requires. The
operative hold — **no production-shape trial until your ruling** — is intact.

---

## 1. What was implemented, against your ruling

**D (binding classification law):** exactly ONE `_on_staging_failed` call site removed — the
deferred-overflow branch (old `:2729`). A coordinator capacity wait can no longer enter the
phase-specific retry matrix on any path. The six surviving `_on_staging_failed` callers (D1b
cannot-fit, hash mismatch, StagingTimeout, StagingConfigurationError, transient IO,
reconciliation mismatch) are proven **AST-identical and behaviourally identical** by gate
G-MATRIX-DIFF — each remains governed by its existing ruling.

**B (required mechanism):** per-connection ingress pause implemented in `_conn_reader_loop`.
Only `sub_stripe_result` frames are gated; `register`/`heartbeat`/`stripe_complete`/
`stripe_error` pass through when they are the decoded frame. At most ONE already-decoded
envelope is held per connection, in the reader thread's local — never in a queue. Subsequent
payloads stay on the wire / at the worker (the worker's `_sendall` is a blocking loop with no
socket timeout, so a full TCP buffer parks its mining thread harmlessly). Resume is FIFO in
pause-entry order, one reader per confirmed capacity observation (no thundering herd), with
hysteresis (pause at bound, resume below bound − margin). The resume trigger lives in
`_pump_deferred`'s `finally`, which makes every pre-existing capacity-release caller a resume
point with **zero edits to any out-of-scope method**. Deferred payloads claim a freed slot
before any paused reader is released, per your ordering.

**A (derived bound):** the constant 64 is deleted. Two pure functions:
`staging_burst_bound_exact` → **116** for the recorded 2026-08-05 assignment
(34 + 14 + 34 + 34); `staging_burst_bound_conservative` → **136** for the conservative
four-slot AMD-cap case (your formula: Σ over slots of max over eligible workers). The
distinction is preserved in signatures, docstrings, gate G9, and a per-stage
`[S172-BP] burst_exact ... exact=%d conservative=%d` log line in every run. The runtime bound
(conservative + a documented margin = live connection count, covering the decode-race window
you allowed) is derived at stage setup from the resolved execution set's **actual assignment
spans** — a short final macro-stripe is not rounded up — with cap resolution through
`advertised_effective_cap`, the single existing path shared with `expected_substripes_for`, so
bound and scheduler cannot diverge. `staging_deferred_max` survives only as an optional
operator override (None ⇒ derived); an override below the derived bound WARNS naming both
numbers.

**C (reachability, values unchanged):** `staging_workers`, `staging_queue_depth`,
`staging_deferred_max`, and the new `staging_capacity_timeout` wired through the complete
route: manifest `default_params` + `args_map` → CLI → `run_bayesian_optimization` →
coordinator attributes → integration `getattr` reads → `run_trial_miner` explicit parameters →
`build_coordinator` → `CoordinatorConfig`. Proven end-to-end by G10. Defaults unchanged (4 / 2
/ None / 600.0) per *"tune after measurement."* Note the route needed the manifest declaration,
not just signatures: WATCHER's step-scoped filter drops undeclared keys
(`agents/watcher_agent.py:1290-1314`).

**Metrics (§4 of the order):** `[S172-BP]`-prefixed structured series — inbound occupancy
high-water, deferred occupancy high-water vs bound, paused count/identities, per-pause and
cumulative pause durations, staging jobs/sec, capacity-timeout and capacity-invariant
termination counters — emitted at pause/resume and as a trial-terminal summary attached to the
serve result.

## 2. Gate evidence — two independent hosts

New suite `tests/test_s172_staging_backpressure.py`, CPU-only, every gate proven RED against
pre-fix behaviour first (red table in the Claude Code report §4), mutation evidence on the two
critical gates (G-MUT-PAUSE, G-MUT-LEASE: the mutant executes the mutated path and reds the
credited assertion).

| your gate | suite gate | result |
|---|---|---|
| 1 saturation fails nothing (phase 1) | G1 | PASS |
| 2 zero retry budget consumed | G2 | PASS |
| 3 pause is per-connection; peers flow | G3 | PASS |
| 4 capacity release auto-resumes | G4 | PASS |
| 5 exactly-once across pause/resume | G5 | PASS |
| 6 no duplicate rows / stale acceptance | G6 | PASS |
| 7 superseded attempt cannot resume-publish | G7 | PASS |
| 8 bounded retention (≤ bound + margin; ≤ 1 envelope/conn) | G8 | PASS |
| 9 derived sizing = 116 exact / 136 conservative | G9 (+G9b override-warn) | PASS |
| 10 full configuration route | G10 | PASS |
| 11 forced timeout → coordinator infra reason, not matrix | G11 | PASS |
| 12 production-shape rerun | **NOT RUN** — order step 6, awaiting this review |
| (Alpha) lease exemption narrow | G-LEASE | PASS |
| (Alpha) six survivors identical | G-MATRIX-DIFF a+b | PASS |
| (Alpha) law: no capacity path reaches matrix | G-LAW | PASS |

**Results: 19/19 on VM101 (canonical) and 19/19 in Alpha's sandbox** — a second, different
host. Regression: `test_s172_staging_partb.py` 24/24 and `test_s172_phase4_coordinator.py`
63/63 on VM101; in Alpha's sandbox both suites were additionally run against a clean worktree
at unpatched HEAD and the pass/fail lists diffed — **the only behavioural difference the patch
introduces anywhere is Gate 22 observing the untracked new test file** (the documented
condition that clears at commit; 63/63 pre-verified with the file absent). All other sandbox
reds are identical at HEAD: a pre-existing environmental assumption (suites inherit the 16 GiB
`staging_high_water_bytes` default, refused by Part B's own validation on hosts with a small
`$TMPDIR`) — recorded as a hygiene item, not introduced by this work. One instance of the same
assumption in the NEW suite (G11) was caught by Alpha's independent execution and fixed
(one line, test-only, disclosed in the review §3); this is why the suite is 79 KB where the
report's copy was one line shorter.

Also verified: `py_compile` clean on all three shared host files; zero new module-scope
imports — the PWC and ZMQ launch paths import exactly what they imported before this diff.

**Gate 56 of the phase-4 suite was updated** (disclosed): it asserted *"excess was
back-pressured via the matrix — trial runs on"*, the precise classification your D removed.
Its bound-proof is retained; its disposition now asserts the §1.6 invariant termination, plus
new assertions that no stripe consumed a retry and none was marked `phase_degraded`.

## 3. One deviation from your text, disclosed

Your B.1 says *"stop reading additional result traffic."* Implemented as: the gate is
consulted per decoded frame, and only a `sub_stripe_result` that cannot be accepted pauses the
connection (holding that one envelope). Frames physically behind a held result remain unread —
TCP is ordered — which is the wire-level property you named. We read this as operationally
equivalent to your clause and it is what makes the one-envelope allowance meaningful.

## 4. Items requiring your ruling

1. **§1.4 lease exemption — ratify or amend.** Your ruling is silent on leases; gates 1–2 are
   unsatisfiable without this. Heartbeats are the only compute-lease renewal path and ride the
   same ordered TCP stream as results, so a paused connection stops delivering renewals; at
   `compute_lease_timeout = 300 s`, `process_lease_expiry` would route the expiry into the
   matrix (`lease_expiry=True` skips the non-retryable branch → constant-phase `fail_trial`) —
   the same death through a different door, redding your gates 1–2 for any pause > 300 s.
   Implemented: stripes whose claiming worker is in **coordinator-initiated** pause are skipped
   by the expiry scan. Narrow by construction: membership in the pause registry is the only
   qualifier and only the coordinator writes it; the pause is bounded by §1.5; G-LEASE proves
   an unpaused worker's genuine silence still expires; G-MUT-LEASE proves removal reds the gate.
2. **§1.5 default** — `staging_capacity_timeout = 600.0 s` (same class as `staging_timeout`),
   measured from the oldest paused connection's entry time, latched once observed, terminal via
   direct `fail_trial` with a `coordinator_staging_capacity_timeout:` reason naming the root
   cause first. The timeout's existence and classification are your ruling; the value is a
   proposal.
3. **§1.6 disposition** — with the derived bound (≥ conservative burst + margin) and the reader
   pause, deferred overflow is mathematically unreachable; reaching it means the SIZING was
   wrong. Implemented as an ERROR log carrying the full arithmetic and a direct
   `fail_trial(coordinator_staging_capacity_invariant: ...)` — never the matrix. This is
   Alpha's reading of your "creating another unbounded coordinator queue is not [acceptable]"
   plus D; amend if you intend a different terminal shape.
4. **New finding for the record (§2.7 class, fourth instance):** pre-fix, passing any of the
   four capacity controls to `run_trial_miner` was a **silent no-op** — swallowed by its
   `**kwargs` tail. It is why the pre-fix G11 hung until `serve_timeout` rather than erroring.
   Now explicit parameters. Submitted for the register.

## 5. Attached artifacts

1. `CLAUDE_CODE_REPORT_S172_STAGING_BACKPRESSURE.md` — implementation report: order
   compliance, red-first evidence per gate, falsely-green gate corrections (§4.3), the four
   brief disagreements, complete files-changed (§7).
2. `TEAM_ALPHA_REVIEW_S172_STAGING_BACKPRESSURE.md` — Alpha's independent-execution review:
   differential HEAD-vs-patched proof, the G11 fix, brief-disagreement adjudication.
3. `s172_bp.patch` — the complete tracked-file diff (5 files, +997/−65).
4. `tests/test_s172_staging_backpressure.py` — the new suite (corrected copy, 19/19 both hosts).

On approval: any amendments you require land fix-forward on `4b1aad6`, then the
owner-initiated production-shape rerun at the recorded 4-stripe/25-daemon shape (your gate 12 /
order step 6), then the Phase-7 soak. §0 acknowledged as still in force.
