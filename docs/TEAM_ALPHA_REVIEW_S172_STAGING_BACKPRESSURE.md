# TEAM ALPHA REVIEW — S172 Staging Back-Pressure Remediation

**Reviewing:** the uncommitted VM101 working tree per
`docs/CLAUDE_CODE_REPORT_S172_STAGING_BACKPRESSURE.md`, against
`docs/CLAUDE_CODE_INSTRUCTIONS_S172_STAGING_BACKPRESSURE_REMEDIATION.md` and the binding Beta
ruling of 2026-08-05.

**Disposition: APPROVE for Beta review, with ONE Alpha review fix applied (test-only, one
line, disclosed in §3). Nothing is committed; no production-shape trial has run.**

---

## 1. Independent execution — not a re-reading of the report

Every claim below was **executed in Alpha's sandbox** on a fresh clone of `7c4f11b` with the
patch applied — a second, different host from VM101, which is itself additional evidence.

| suite | VM101 (report) | Alpha sandbox | note |
|---|---|---|---|
| `test_s172_staging_backpressure.py` | 19/19 | **18/19 → 19/19 after §3's fix** | the single red was environmental (G11), see §3 |
| `test_s172_staging_partb.py` | 24/24 | 23/24 | **pre-existing** env sensitivity — identical red at unpatched HEAD, see §4 |
| `test_s172_phase4_coordinator.py` | 63/63 | 56/63 vs **57/63 at HEAD** | differential = Gate 22's documented untracked-file mechanism ONLY; all other reds identical at HEAD (env), see §4 |

**Differential method:** both regression suites were run twice in the same environment — once
on the patched tree, once on a clean `git worktree` at HEAD — and the pass/fail lists diffed
line-by-line. **The only behavioural difference the patch introduces in any suite is Gate 22
observing the untracked new test file**, which is the standing documented condition and clears
at commit (Claude Code pre-verified 63/63 with the file absent).

Additionally verified: `py_compile` clean on all three shared host files; **zero new
module-scope imports** in `window_optimizer.py` / `window_optimizer_integration_final.py`, so
the PWC and ZMQ launch paths import exactly what they imported before — "standalone" is
measured, not assumed, for this diff.

## 2. Conformance to the ruling

- **§0 / Beta D:** exactly ONE `_on_staging_failed` call site removed (the deferred-overflow
  branch, old `:2729`). G-MATRIX-DIFF proves the six surviving callers **AST-identical and
  behaviourally identical** — Alpha's brief asked for behavioural identity; the implementation
  additionally holds `_on_staging_failed` itself identical, which is stricter and is what
  caught its own first (wrong) resume placement.
- **Beta B:** pause implemented per-connection in `_conn_reader_loop`; only `sub_stripe_result`
  gated; at most one decoded envelope held, in the reader thread's local, never in a queue;
  resume in FIFO pause-entry order, one reader per confirmed capacity observation; the resume
  trigger lives in `_pump_deferred`'s `finally`, which converts **every pre-existing
  capacity-release caller into a resume point with zero out-of-scope edits**.
- **Beta A:** the constant 64 is deleted. Two pure functions produce **116** (exact, recorded
  assignment) and **136** (conservative four-slot AMD-cap) — Beta's distinction preserved in
  signatures, docstrings, gate G9, and a per-stage `[S172-BP] burst_exact` log line so it is
  visible in every run log. Cap resolution goes through `advertised_effective_cap` — the single
  existing path shared with the scheduler — so bound and `expected_substripes_for` cannot
  diverge. The runtime bound derives from the **actual assignment spans** at stage setup, so a
  short final macro-stripe is not rounded up (exceeds the brief).
- **Beta C:** all four controls wired manifest → `args_map` → CLI → integration →
  `build_coordinator` → `CoordinatorConfig`, values unchanged, proven end-to-end by G10.
  The implementation caught that the brief's route was insufficient: adding only signatures
  would have left the params dropped by WATCHER's step-scoped filter
  (`agents/watcher_agent.py:1290-1314`) — declared in the manifest accordingly.
- **Order §4 respected; step 6 NOT run.** Metrics per §4, `[S172-BP]`-prefixed, plus a
  trial-terminal summary attached to the serve result.

## 3. Alpha review fix (disclosed) — G11 environment independence

G11 drives the real `run_trial_miner` path (good — it is the gate that caught the pre-fix
`**kwargs` swallow) but omitted an explicit `staging_high_water_bytes`, inheriting the 16 GiB
default. Part B's own validation **correctly** refuses a 16 GiB mark on a filesystem with less
free space, so the gate's verdict depended on the host having ≥16 GiB free in `$TMPDIR` — it
red in Alpha's sandbox (9.94 GiB free) for a reason unrelated to what it proves.

**Fix:** one line — `staging_high_water_bytes=64 * 1024 * 1024` in G11's `run_trial_miner`
call, with an in-place comment. Test-only; no production code touched. 19/19 after the fix in
the sandbox; **VM101 must re-run the suite over the corrected file before commit** to record
19/19 on the canonical host as well.

## 4. Hygiene finding for the record (pre-existing, NOT this change)

The **committed** Part B suite (one gate) and phase-4 suite (Gate 54 and several others) carry
the same 16-GiB-default assumption and red identically at unpatched HEAD on any host with a
small `$TMPDIR`. Not introduced by this work, not a blocker, passes on VM101 — recorded here so
it lands in `docs/BACKLOG.md` rather than being rediscovered as a "regression" by the next
environment that differs.

## 5. Brief-disagreement adjudication

Claude Code reported four disagreements with Alpha's brief rather than working around them.
**All four are adjudicated in Claude Code's favour**, including the first — the brief named a
`MinerCoordinatorConfig` class that does not exist (live: `CoordinatorConfig`); Alpha's error,
made by writing a name without verifying it, the exact defect class the brief warned against.
The refusal of a bare int in `staging_burst_bound_conservative(slots=...)` (no invented stripe
size) is the right reading of "derived from real geometry."

## 6. Items for Beta's ruling, consolidated

1. **§1.4 lease exemption** — required by gates 1–2 (heartbeats share the ordered TCP stream
   with results; a pause > 300 s otherwise reaches the constant-phase `fail_trial` through
   `process_lease_expiry`). Narrow: coordinator-initiated pauses only, bounded by §1.5,
   G-LEASE proves an unpaused worker's genuine silence still expires, G-MUT-LEASE proves the
   gate reds when the exemption is removed. **Ratify or amend.**
2. **§1.5 default** `staging_capacity_timeout = 600.0` (same class as `staging_timeout`).
3. **§1.6 disposition** — post-derivation deferred overflow is a sizing-invariant violation
   terminated by direct `fail_trial` (`coordinator_staging_capacity_invariant:` + full
   arithmetic), never the matrix. Gate 56 updated accordingly, bound-proof retained, with new
   assertions that no stripe consumed a retry.
4. **New finding, §2.7 class (fourth instance):** pre-fix, passing any of the four controls to
   `run_trial_miner` was a **silent no-op** — swallowed by `**kwargs`. It is why the pre-fix
   G11 run hung until `serve_timeout` instead of erroring. Now explicit parameters; recommend
   adding to the §2.7 register at next skill revision.

## 7. Sequence from here

1. Corrected suite file → VM101 (overwrites the working-tree copy) → re-run on VM101 → 19/19.
2. This review + the Claude Code report + the diff → **Beta**.
3. On Beta's approval: Michael commits from the report's §7 list (the new suite's commit also
   clears Gate 22), dual-pushes.
4. **Michael-initiated** production-shape rerun at the recorded 4-stripe/25-daemon shape —
   G12/order step 6 — then the 50-trial soak. §0 remains in force: the deferred queue is not
   proven to be the last defect; the next boundary is reached by running.
