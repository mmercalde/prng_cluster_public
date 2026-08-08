# TEAM ALPHA → TEAM BETA — F1–F5 AMENDMENT DELTA (review + resubmission)

**Per your ruling of 2026-08-06 (HOLD — TARGETED FIX-FORWARD).** Delta against `4b1aad6`
(tip `42bdbb1`), fix-forward, no history rewrite. Two files changed, exactly as your §8.8
scope requires: `miner/range_miner_coordinator.py` (+453/−59 incl. one Alpha review fix,
§3) and `tests/test_s172_staging_backpressure.py` (+820/−12). Zero changes to worker code,
seed caps, stripe geometry, or `gate_s172_prod_shape.py`. Nothing committed; nothing
launched; gate 12 and the soak remain unrun per your hold.

**Alpha's disposition: APPROVE for your delta review.** Independently executed in full on a
second host, differential-proven against the `42bdbb1` baseline.

## 1. The five corrections (§8.1) — all landed, each with your required gate (§8.2)

**F1 — ingress resume credit.** `_resume_paused_connections` is now exactly one call to
`_grant_resume_credit`: at most ONE wake per capacity-release invocation, granted under
`_pause_lock` in the same critical section as the capacity observation, FIFO-oldest
unsignaled reader only, and the wake RESERVES the observation — no further grant (by event
or by poll) until the woken reader hands its envelope to `inbound` or exits. The second
herd door your gate would have caught — every paused reader's 50 ms defensive poll — is
closed by `_try_self_resume`: head-only, credit-free-only, takes the credit itself; the
lost-wakeup protection the poll exists for is preserved.
Gates: **G-RESUME-CREDIT-a** (mechanism: one release → one wake; non-head poll denied),
**G-RESUME-CREDIT-b** (your seven-step script with real reader threads: two paused, one
unit released, one resume path invoked, exactly the FIFO-first resumes, the second REMAINS
paused across a settling window, second unit → second resumes, FIFO order asserted).
**G-MUT-RESUME-CREDIT**: both mutants (the old loop restored; the headless poll restored)
executed and red the gate.

**F2 — resume grace (your §3 mechanism, as specified).** `deregister_paused_connection`
with `reason=="resume"` records `_capacity_resume_grace[worker_id] = now +
compute_lease_timeout` (a coordinator dict under `_pause_lock`; the reader's no-ledger rule
stands). `process_lease_expiry` skips actively-paused OR grace-live workers and prunes
expired entries in the same pass. The heartbeat branch clears the grace — gated on
`renew_lease`'s OWN boolean, stricter than your wording in the fail-closed direction: a
renew that did not land has not restored the lease, so the bridge stays up to its bound
(disclosed deviation, §2.3). Cleared on connection drop and at trial-terminal.
Gates: **G-LEASE-HANDOFF** (pause past the lease deadline; resume with the heartbeat still
queued behind the envelope; expiry scan in that window → zero matrix entries; heartbeat →
renewal + grace cleared; separately, a resumed worker that never heartbeats expires after
the bound). **G-MUT-LEASE-HANDOFF**: grace-recording removed → executed → red.

**F3 — timeout evidence snapshot.** The latch and the snapshot (`latched_at`,
`oldest_since`, `paused_count`, `worker_ids`) are taken in the SAME `_pause_lock` critical
section as the oldest-pause read that decided the timeout — which also closed a latent
double-latch race in the pre-amendment code (the latch check now re-runs under the lock).
`staging_capacity_timeout_reason` and the metrics use the snapshot whenever it exists.
Gate: **G-TIMEOUT-SNAPSHOT** — the reader observes the latch, deregisters and exits before
the serve loop terminates; the abort reason still names the triggering worker and a
nonzero count.

**F4 — registered workers only.** The pause condition now requires
`worker_by_sock.get(rawsock) is not None` — written only at registration, so it IS the
bound-worker predicate. An unbound result under saturation is not paused and not held; it
flows to the EXISTING serve-loop identity rejection unchanged. No new rejection logic.
Gate: **G-BOUND-PAUSE** — under saturation, a pre-registration result creates no pause
record, no grace, no timeout attribution, and dies in the existing identity path.

**F5 — sizing fails closed.** Stage setup materializes spans/eligible/exact-rows once at
entry; ANY derivation exception (`except Exception` — the brief's `(ValueError,
TypeError)` was itself a hole; a `KeyError` from a malformed cap record escaped it)
terminates via direct `fail_trial("coordinator_staging_sizing: ...")` before any result
traffic, never the matrix, never the smaller on-demand fallback —
`_derive_bound_from_current_state` is now commented as NOT a production fallback and no
production path reaches it after a failed derivation. Your item-3 ratification detail is
implemented: `_defer_locked` records WHICH bound tripped and the §1.6 invariant reason
carries one of three explicit phrases (derived-count / operator-override-count /
retained-bytes high-water).
Gates: **G-BOUND-DERIVATION-FAILURE** (malformed cap at stage setup → `coordinator_staging_
sizing:` terminal before staging, no matrix entry, no retry, never ran on the fallback) and
**G-BOUND-TRIP-PHRASE** (the reason names the tripped bound).

## 2. Evidence per your §8

1. **Corrections:** §1 above; coordinator only.
2. **Gates:** nine new (your five plus G-RESUME-CREDIT split a/b, the two required mutation
   gates, and G-BOUND-TRIP-PHRASE), appended to the suite. **Eight of nine reds are
   BEHAVIOURAL against a `4b1aad6` worktree** — e.g. F1's red shows one freed unit resuming
   the ENTIRE fleet; F2's shows a coordinator-caused silence entering the matrix during the
   handoff; F3's shows `0 connections paused (none)` about a timeout one worker caused;
   F5's reproduces your §6 end-to-end (fallback → stage dispatched → results returned →
   `ValueError` escaping `serve_trial` after real traffic).
3. **Mutation evidence:** G-MUT-RESUME-CREDIT and G-MUT-LEASE-HANDOFF, both proven to
   execute the mutated path and red the credited assertion.
4. **Suite green both hosts:** **28/28 on VM101, five consecutive full runs; 28/28 on the
   independent Alpha host** (fresh clone of `42bdbb1` + this delta).
5. **`test_s172_staging_partb.py`:** 24/24 on VM101; on the Alpha host 23/24 with the
   single red IDENTICAL at the `42bdbb1` baseline (the recorded pre-existing
   16 GiB-vs-`$TMPDIR` environmental assumption).
6. **`test_s172_phase4_coordinator.py`:** 63/63 on VM101 with the production diff applied;
   Alpha-host differential vs clean `42bdbb1` shows the ONLY change to be Gate 22 observing
   the modified suite file (the documented commit-clears condition; also isolated on VM101
   two ways — clean tree → 63/63, production diff only → 63/63). Gate 54's red is identical
   both sides (environmental). Gate 22 itself was not touched or widened.
7. **AST evidence:** G-MATRIX-DIFF-a/b re-run green — the six `_on_staging_failed`
   survivors and all three matrix methods are AST-identical to BOTH `7c4f11b` and
   `4b1aad6`, plus a new assertion that this amendment changed no survivor. (Disclosure:
   the gate's original `HEAD:` baseline red on its own success when `4b1aad6` became HEAD;
   pinning the hashes is the repair. The gate is now commit-independent.)
8. **No out-of-scope changes** — files-changed is exactly the two; zero new module-scope
   imports.

## 3. Alpha review fix, disclosed (production, 15 lines)

Claude Code flagged without actioning: `staging_backpressure_metrics()` can RAISE at the
trial-terminal summary via the on-demand derivation. Alpha traced reachability: after F5, a
sizing failure leaves `_derived_deferred_bound = None`, and the terminal summary then calls
`staging_deferred_bound()` → the on-demand path → the SAME malformed caps that failed
stage setup — raising at the summary and MASKING the honest `coordinator_staging_sizing`
terminal. That is the F3 disease relocated to the reporting layer. Fix: `bound_in_force`
is computed under `try/except`; on failure it reports `None` plus a `bound_in_force_error`
field, and the summary format is widened to render it. Reporting degrades; it never
overwrites the terminal truth. Verified by the full 28/28 run on the Alpha host;
**VM101 must re-run the suite after applying `alpha_review_fix_terminal_summary.patch`**
(the canonical combined delta `s172_bp_amendment_full.patch` already includes it).

## 4. Deviations register (all in the fail-closed direction)

1. F5 catches `Exception`, not the brief's `(ValueError, TypeError)` — §1/F5.
2. F2's grace-clear is gated on `renew_lease`'s boolean, not merely on reaching the call.
3. F1 enforces reader-exit ordering (deregister BEFORE credit release) — a grant can only
   target a live registry entry, so no grant can land on a record being removed and wedge
   the fleet. Not in your ruling or the brief; a liveness necessity, commented in-source.
4. §3's terminal-summary guard — Alpha-added this round.

## 5. Requested disposition

Approve the delta; on approval Michael commits the amendment (clearing Gate 22),
dual-pushes, and — per your §9 — the 4-stripe/25-daemon production-shape trial and the
Phase-7 soak await your explicit authorization.
