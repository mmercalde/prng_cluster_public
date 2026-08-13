# SESSION CHANGELOG — 2026-08-11 — F1 LEASE-ORIGIN REPAIR + SERVE-LOOP INSTRUMENTATION

**Host:** VM101 `zeus-ubuntu` (192.168.3.177), user `michael`, `~/venvs/torch`.
**Base:** `213bfff`, clean tree verified at session start.
**Brief:** `~/dashboard_work/CCODE_BRIEF_F1_LEASE_ORIGIN_REPAIR_v1_1.md` (Beta scope ruling
2026-08-11, which **withdrew** the earlier "mechanism is free" language and prescribed the shape).
**Full report:** `~/dashboard_work/F1_LEASE_ORIGIN_REPAIR.md`.
**Nothing committed, nothing pushed, attempt 5 NOT launched.**

**⚠ CONTINUED 2026-08-12 — R1 CORRECTION.** Beta's RRR: **RETURN FOR NARROW R1 ONLY**. The
**F1 lease-origin repair itself is ACCEPTED and was not altered** — all four defects were in the new
`ServeLoopTiming` instrumentation. R1 brief:
`~/dashboard_work/CCODE_BRIEF_R1_SERVELOOP_INSTRUMENTATION_v1_0.md`. See **§9** below; the report is
**amended, not rewritten** (its §11–§13). Suite is now **18/18**, battery re-run at post-R1 final
state, everything else unchanged. Still nothing committed, nothing pushed, attempt 5 not launched.

---

## 1. The defect, re-derived rather than relayed

Gate-12 attempt 4 (`distributed_config_t1_c8939b64`). `serve_trial` captures `now = time.time()`
once per iteration and passed that same value into `schedule_pending_stripes(..., now=now)`, which
resolved it once at entry and stamped every `claim_stripe` of the pass with
`now + compute_lease_timeout`.

Measured this session from the frozen bundle (digests re-verified: 11/11, 29/29, 13/13 OK):

- stage 2 opened **19:15:42.900** with `claimed=25 queued=7`;
- the sampler brackets the last two backlog claims to **19:21:45.667 → 19:21:47.676**
  (`queued_pending` 2 → 0, `rrig6600:gpu1` and `zeus-ubuntu-vm:gpu0` newly compute-active);
- `__st1_s30` and `__st1_s31` both carry `lease_expires_at = 19:22:13.373838` — **identical to the
  microsecond**, i.e. one shared origin of `19:17:13.373838 + 300.0`.

⇒ the iteration's clock was **272.3 s** old at the scheduling pass; both stripes were born with
**~26–28 s of a nominal 300 s lease (~91 % consumed)**; both produced **zero shards**; the
constant-mode matrix then failed the trial correctly on a lease that had never been granted.

**Still unexplained and NOT fixed by this work:** what made that one iteration take ~4.5 minutes.
The coordinator log is empty 19:15:42.900 → 19:22:52.014 and the worker logs are 90–328 bytes each.
That is why the instrumentation exists.

## 2. The repair (Beta's prescribed mechanism, not free choice)

`miner/range_miner_coordinator.py`, **+286 / −7**, one file.

- **`schedule_pending_stripes`** — the per-pass `now = time.time() if now is None else now` is
  DELETED; `claim_now = time.time() if now is None else now` is read **immediately before each**
  `claim_stripe` (`:3224/:3227`). Per-claim, inside the loop, so the invariant survives a slow
  scheduler as well as a stale caller. `now` had no other consumer in the method.
- **`serve_trial` maintenance pass (`:7259`)** — no longer passes `now=`.
- **`assign_stripes` initial handoff (`:3092`)** — forwards the caller's argument **as given**
  (`injected_now`, still `None` in production) instead of its own defaulted value. `add_stripe`'s
  creation timestamp is unchanged. *Judgement call, flagged, one-line reversible.*
- The **injected-clock seam is retained**: an explicit `now` is honoured verbatim, so every
  existing gate's deterministic lease arithmetic is unchanged.

**Scope containment, by AST:** 248 → 257 qualified defs, **nothing removed, exactly three existing
defs changed** (`assign_stripes`, `schedule_pending_stripes`, `serve_trial`); the 9 additions are
the instrumentation. `schedule_pending_stripes` remains the **only** compute-lease creation site
(§2.27 invariant intact — `_renew_active_lease` is renewal and reads its own fresh clock).

## 3. Six-site audit — Beta's expected result CONFIRMED against source

Computed by AST from both `213bfff` and the working tree, not transcribed. Pre → post:

| site | callee | purpose | modified |
|---|---|---|---|
| `:6602`→`:6829` | `fail_trial` | staging-capacity-timeout terminal timestamp | NO |
| `:6785`→`:7013` | `fail_trial` | worker-admission-timeout terminal timestamp | NO |
| `:6855`→`:7089` | `fail_trial` | retention-sizing failure terminal timestamp | NO |
| `:6878`→`:7112` | `fail_trial` | preflight-provenance failure terminal timestamp | NO |
| `:6966`→`:7204` | `fail_trial` | staging-sizing failure terminal timestamp | NO |
| `:6999`→`:7259` | `schedule_pending_stripes` | **compute-lease origin** | **YES** |

`fail_trial → submit_abort → abort_trial`: `now` reaches only `create_trial(..., now)` and
`mark_trial_aborted(..., now, ...)`. **No deadline arithmetic on that path.**

**Beta's correction to Alpha verified, not trusted:** the maintenance loop calls
`process_lease_expiry(run_id, eligible)` — two positional args, **no `now`** — on both sources.

## 4. Stop-and-report rule: NOT triggered *(⚠ Beta ruled this TOO BROAD — see §9)*

No second independent timing defect exists, for a structural reason worth recording:

> The lease was the **only** site where a stale clock computes a **future deadline** that a
> **different, fresh** clock later evaluates (`process_lease_expiry` reads `time.time()` itself), so
> staleness produced a **premature fail-closed termination**. Every other consumer compares the
> stale `now` against a **past** timestamp, so staleness can only make a check fire **late** —
> a liveness cost, never a false terminal. Unchanged by this patch.

## 5. Serve-loop instrumentation (Beta-authorized, non-behavioural)

`ServeLoopTiming` (`:2513`) + `log_serve_loop_timing_summary` (`:5050`) → one grep-stable
`[S172-SL] summary` per trial beside `[S172-BP] summary`, also returned as `serve_loop_timing`.
Segments: `iteration · accept · drain · msg · deadline · stage_setup · schedule · dispatch ·
expiry · advance · exit`, plus `unattributed_total`.

- **`time.perf_counter()` only** (gated by AST) — a timing instrument must not share a clock with
  the lease seam, and a wall-clock step must not manufacture a high-water.
- **`loop_now_age_max`** = the age of the shared `now` **at the scheduling pass** — the quantity
  that was 272.3 s in attempt 4, kept after the repair so the delay cannot hide behind the fix.
- **Maxima, not means** (a stage is ~4,300 iterations at a 0.1 s poll), with
  `iteration_max_at` so a reader can locate the outlier in the log.
- Cannot raise, cannot mask a primary terminal reason (gated).

> **⚠ §5 IS SUPERSEDED IN THREE PLACES BY §9 (R1, 2026-08-12).** `iteration` did **not** account for
> every path — the terminal iteration was never recorded (R1.1). `unattributed_total` was **not**
> "time inside no named segment" — nested `msg` was subtracted twice (R1.2). "`perf_counter` only"
> was true of the class but **false as wired** — the call site read `time.time()` (R1.3). All three
> are fixed; read §9 before citing this section.

## 6. Gates — `tests/test_s172_f1_lease_origin.py`, 13/13 *(⚠ now 18/18 — §9)*

L1 (RED pinned pre-repair → 27.7 s residue / GREEN → full 300 s from the claim) · L2 (two late
claims, two distinct origins) · L3 (no premature constant failure; RED expires at claim+30) ·
L4 (constant expiry at claim+301 still terminal; hybrid first-expiry retry unchanged) · L5
(one-active invariant, `LeaseInvariantError` still raised) · L6 (**both halves of the repair
reverted one at a time — each mutant killed**, plus the anchor self-protection) · L7 (six-site
audit computed from both sources; serve-loop clock capture unchanged; live `serve_trial` proves the
admission window still fires with `serve_timeout=None`; instrumentation monotonic + inert).

**RED-arm provenance:** every RED arm runs the committed pre-repair source **pinned to `213bfff`**,
integrity-checked for both defect surfaces, reporting **UNAVAILABLE** rather than a pass if the
anchor drifts. Deterministic — a scripted `_Clock` replaces the module clock; no sleeps.

## 7. Regression battery — final state, sequential

phase-4 **63/63** · F1/F2 **16/16** · Defect A **29/29** · back-pressure **50/50** ·
admission-liveness **16/16** · resolved-execution-set **34/34** · elapsed-roundtrip **6/6** ·
new lease-origin suite **13/13**.

- **Gate 22** reds at 62/63 while `tests/test_s172_f1_lease_origin.py` is untracked — the known
  ruled behaviour (§2.33). **Allowlist NOT widened.** The 63/63 is the same tree with the new suite
  held aside, `git status --porcelain` = ` M miner/range_miner_coordinator.py`. Committing the file
  clears it. `test_s172_phase5_d5_process_sharded.py`'s `NR` arm will inherit the same red until
  then.
- **`test_s172_admission_binding.py` is 11/20 — PRE-EXISTING.** Differential-worktree proof
  (`git worktree add --detach <scratch> 213bfff`, same venv/host): pass/fail lists **identical** on
  base and patched, so nothing is chargeable to this change. Reads as the `localhost.gpu_count
  2 → 1` correction (`f255912`) against gates written for a two-identity local set. Not a timing
  defect, not in scope, not touched. Worktree removed and pruned.
- **Not re-run, therefore not re-credited:** the Phase-5 D-series, `d6_1`, `d6_2`, phase 1/2/3,
  `phase6_p05_dataset_authority`, `threshold_propagation`, `process_sharded_import_gate`,
  `staging_partb`, `gate12_gpu_gate`, `gate12_concurrency_sampler`
  (`gate12_cleantree_admission` ran 31/31 earlier this session).

## 8. Files

```
 M miner/range_miner_coordinator.py                       +381/-7   (was +286/-7 pre-R1)
?? tests/test_s172_f1_lease_origin.py                     NEW (L1-L7 + R1, 18 gates, 8 mutants)
?? docs/SESSION_CHANGELOG_20260811_F1_LEASE_ORIGIN.md     NEW (this file)
   ~/dashboard_work/F1_LEASE_ORIGIN_REPAIR.md             AMENDED, not rewritten (§11-§13)
```

*(`docs/TFM_PROJECT_FACTS_SKILL.md` is also modified in the tree; it was already modified at session
start and is not part of this work.)*

Frozen evidence bundles were opened **read-only** (`file:…?immutable=1`) and nothing in them was
modified.

**Next per the brief:** Michael reviews → **Beta certifies** → Michael commits and dual-pushes →
prelaunch battery → attempt 5. Gate 12 and Phase 7 remain HELD.

---

## 9. R1 CORRECTION — 2026-08-12 — serve-loop instrumentation only

**Beta disposition: RETURN FOR NARROW R1 ONLY.** ACCEPTED and untouched: the lease seam, production
no longer injecting `now=`, the `assign_stripes` `injected_now` judgement call (**APPROVED** — it
extends the invariant to initial claims rather than relying on geometry insertion being "fast
enough"), L1–L6, the one-active invariant, constant/hybrid semantics, the six-site audit, and
`test_s172_admission_binding.py` 11/20 as **not chargeable** to this amendment.

**All four defects were in the new instrumentation.**

| # | defect | fix |
|---|---|---|
| **R1.1** | `tick()` records an iteration only when the NEXT one begins, so the **terminal** iteration was never recorded. **Attempt 4's terminal iteration had already aged 272.3 s when it reached the scheduling pass; that same iteration later terminated the trial** — the instrument would have missed the event it was built for. The report's *"accounts for EVERY path"* was false as implemented. | `close_current_iteration()` — idempotent, non-raising, clock-free when already closed — called as the **first statement** of `serve_trial`'s `finally`, before the exit timer. Invariant: `iteration_count == iterations`. |
| **R1.2** | `msg` is timed **inside** `drain` and the residual subtracted **both**, so dispatch time was charged twice and `unattributed_total` was not "loop time inside no named segment" — erring toward **zero** exactly when the loop was busiest with messages. | `NESTED_SEGMENTS = ("msg",)`; the subtraction is over non-overlapping **top-level** segments. Nested totals still reported. |
| **R1.3** | The class read only `perf_counter`, but the **call site** computed `time.time() - now`, so the instrument as a whole used wall time and the L7 gate — inspecting only the class — could not see it. | Age derived from the monotonic mark `ServeLoopTiming` already holds; the loop's `now` survives as the wall-time **label** only. **No production wall-clock read added:** `serve_trial` performs 2 `time.time()` calls, the same as pinned (it was 3 before R1). |
| **R1.4** | `PINNED_COMMIT = "213bfff"` abbreviated. | `"213bffff512f0e360c40974cbfc9e787c5b005f0"`. Fails closed either way; a permanent governance anchor should not be abbreviated. |
| **R1.5** | Report/docstring claim *"one `time.time()` per **successful** claim and never otherwise"* is too strong — the read precedes `claim_stripe`, so a `False` return still consumed one. | Restated: **one read per claim attempt reaching `claim_stripe`**. **No credited assertion depended on the stronger wording** (L1-GREEN/L2 run passes where every attempt succeeds); prose corrected in both places. |

**⚠ THE R1.1 GATE ENCODED THE DEFECT AND TURNED GREEN** — `iterations == 2 and iteration_count == 1`
declared the missing terminal iteration to be correct behaviour. Fresh instance of the skill §2.30
pattern. Per Beta's instruction the other instrumentation gates were checked for the same shape:
`l7_non_lease_consumers_behaviour` asserted only field *names* and now also asserts the R1.1
invariant end-to-end off the **real** `serve_trial`'s returned `serve_loop_timing`. One unrequested
gate-integrity fix: the runner picked anchor-dependent arms by **substring match on display name**,
so a new pinned-source gate could have run without the anchor and reported FAIL where VIR-3 requires
UNAVAILABLE — dict membership is now the rule.

**Two record-only precision corrections from Beta's certification (prose only — no code changed, no
test re-run, no timer moved):**
1. **272.3 s is the AGE of the iteration's shared `now` at the scheduling pass, NOT the measured
   full duration of that production iteration** — the full duration was longer and is **unmeasured**,
   which is the gap this instrument exists to close. Every "the 272.3 s iteration WAS the terminal
   one" phrasing is restated as *"attempt 4's terminal iteration had already aged 272.3 s when it
   reached the scheduling pass; that same iteration later terminated the trial."* Using 272.3 s as
   the deterministic long terminal iteration **in the R1.1 test remains correct** — there it is a
   scripted duration, not a claim about the run.
2. **`stage_setup` is `assign_stripes` + deferred-bound derivation / associated stage setup — it
   does NOT cover the retention preflight.** Verified against live source: the timer starts at
   `:7205`, after `preflight_trial_retention` (`:7156-7204`) returns, so **retention-preflight time
   falls into `unattributed_total`**. **Beta explicitly does NOT want the timer moved.** Benign for
   the residual: preflight runs once per trial, at the first stage.

**Gates: 18/18** (13 + 5 R1 arms, 8 mutants). Every R1 mutant is applied to the **production class
or production source text**, asserts it was applied, executes the mutated path and kills the
credited assertion. Determinism: `_MonoClock` scripts `perf_counter` and **raises on exhaustion**,
with full consumption asserted.

**THE LEASE SEAM IS BYTE-UNCHANGED BY R1.** Per-definition AST digest vs pinned: **242 → 252 defs,
none removed, the same three changed** (`assign_stripes`, `schedule_pending_stripes`, `serve_trial`)
as before R1. Source-segment diff vs pinned gives exactly three non-docstring lines in each lease
function — **the accepted repair and nothing else**. Anchors:
`schedule_pending_stripes d7e8a1d3…7461` · `assign_stripes 567a41fe…586d`. Diffstat vs `213bfff` is
now **+381 / −7**; deletions are still **7** because every R1 edit lands inside a region this patch
had already added.

**RECORDED, NOT REPAIRED — terminal timestamp staleness** (Beta's separate finding, arising from
Alpha's own six-site audit):

```
CONFIRMED structurally · separate provenance/observability defect
NOT authorized in this patch · NOT an Attempt-5 launch blocker · record for later governance work
```

If a serve-loop iteration is 272 s stale and one of the five `fail_trial(now=now)` branches fires
late within it, the **durable terminal timestamp** is stale by the same amount — *actual terminal
decision time ≠ durable `finalized_at`*, potentially by minutes. **This corrects Alpha's "no second
timing defect" conclusion, which Beta ruled too broad:** the reasoning held for elapsed-time
*comparisons* (staleness makes them fire late, never early) but not for the five values **persisted
as terminal time**. **No code was written for it.**

**Battery re-run at post-R1 final state, sequentially:** phase-4 **63/63** · F1/F2 **16/16** ·
Defect A **29/29** · back-pressure **50/50** · admission-liveness **16/16** ·
resolved-execution-set **34/34** · elapsed-roundtrip **6/6** · lease-origin **18/18** ·
`admission_binding` **11/20 — identical, still pre-existing**. Gate 22 unchanged: 62/63 with the
untracked suite present, 63/63 with it held aside; **allowlist NOT widened**.
