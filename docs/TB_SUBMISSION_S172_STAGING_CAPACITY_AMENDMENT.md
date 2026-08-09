# TEAM ALPHA → TEAM BETA — S172 STAGING-CAPACITY AMENDMENT + `elapsed_s` PERSISTENCE

**Implements:** the *"S172 GATE-12 STAGING-CAPACITY DEADLOCK"* ruling (2026-08-07) §§2-6, and the
*"STEP-1 SEARCH GEOMETRY…"* ruling (2026-08-08) **R4**.

**Base:** `c7058d8`. **Nothing committed, nothing pushed, nothing launched.** Gate 12 and the
Phase-7 soak remain held. The seed-domain/cursor amendment is a separate work item, not started —
per your §12, they are not merged.

**Results, verified on two hosts:** `test_s172_staging_backpressure.py` **42/42** (VM101 ×3, and
independently on Alpha's host from a fresh clone of `c7058d8` + this patch);
`test_s172_elapsed_roundtrip.py` **6/6** both hosts; `test_s172_staging_partb.py` 24/24 VM101.
Red-first: **all 7 new arms red at `c7058d8`, each for its own distinct reason**, while all 35
pre-existing gates stayed green. **All 53 pre-existing gate functions are assertion-identical by
AST.**

---

## 1. What landed

**§1.1 — release on successful commit (Option C).** `commit_trial` now discharges every
trial-owned reservation **only after** the sink's successful return, **reusing
`ack_by_event_id`** — the path Alpha reported yesterday as having zero production callers. It was
built for exactly this and never wired. Failure retains everything, preserving D1.1's retry
contract, via a new `commit_cleanup_status` column rather than overloading the abort one.

**§1.2 — derived whole-trial bound.** All three `4096` sites → `None` = derive. The bound reuses
`staging_burst_bound_conservative` per stage rather than restating the formula, and the preflight
sits **above `assign_stripes`**, so an impossible ceiling dispatches nothing. **1,028 appears
nowhere in the code.**

**§1.3 — both high-waters routed** manifest → `args_map` → argparse → coordinator → factories →
config. The stale `getattr(..., 512)` is gone.

**§1.4 — the invisible blocker, found and closed.** `_run_staging_job:4358` was
`while True: … except StagingBackPressure: sleep(0.02)` — **a 50 Hz retry loop with no clock at
all.** That is what consumed the failed run's ~19 minutes while `staging_capacity_timeout_expired()`
watched only paused connections and saw nothing. Executor waits now register under the **same
lock** as the pause registry, so *"oldest across both classes"* is one atomic read.

**Part 2 — `elapsed_s`,** structurally separable: `tests/test_s172_elapsed_roundtrip.py` imports
nothing from the Part 1 suite, builds its own ledger, and touches no capacity surface. Six arms
including two Alpha considers load-bearing: a genuine `0.0` must persist as `0.0` (otherwise a
gate proving "absent → NULL" also passes an implementation that stores NULL unconditionally), and
a hostile replay carrying a *different* measurement must not overwrite the first.

---

## 2. Three items for your ruling — with Alpha's positions

### 2.1 Gate 37 — Alpha recommends SUPERSESSION, not a permanent red

Phase-4 Gate 37 asserts a staged file **still exists on disk after a successful commit**. Option C
§1.1 requires a successful commit to **delete** staged files. Both cannot hold.

**The evidence is behavioural, not argued:** the isolation run logs
`'[S172-CAP] trial … committed — released 1 trial-owned reservation(s), deleted 1 staged file(s)'`
immediately before the assertion fires. Gate 37 passes at `c7058d8` and reds now.

Claude Code left it red and reported it, per Alpha's instruction — editing a gate's assertion to
match one's own new behaviour is how a suite stops being independent evidence.

**Alpha's position: mark Gate 37 superseded by Option C §1.1 and authorise the replacement
assertion**, on the exact precedent of **Beta D superseding Gate 56** (skill §2.19), which you
ratified for the same reason: the *contract* changed, not the gate's correctness. **Reason for
preferring supersession over leaving it red:** a red gate that nobody is permitted to fix becomes
standing noise, and standing noise trains readers to skip reds. Phase-4 currently ships 61/63
(Gate 37 + Gate 22's untracked-`.py` condition); under supersession it returns to 62/63 pre-commit
and 63/63 after.

**Requested:** authorise the replacement assertion — that the manifest path was published and the
file is correctly absent post-commit — as a scoped, disclosed edit.

### 2.2 ~~The derived bound is 816, the failed run consumed 1,028~~ — **RETRACTED**

> **⚠ RETRACTED IN FULL. Team Beta corrected this section (ruling 2026-08-08 §3); the
> conclusion is DELETED, not amended. See `docs/CLAUDE_CODE_REPORT_S172_STAGING_CAPACITY_R1.md`
> §3.**
>
> This section derived 816 files from a **4-macro-stripe** geometry, mislabelled that as "the
> recorded gate-12 geometry", and inferred that 1,028 **"implies roughly five planned
> macro-stripes"**. That inference was factually wrong.
>
> The real 2026-08-07 gate-12 production geometry is `max_seeds = 1,073,741,824` over
> `miner_stripe_size = 67,108,864` = **16 macro-stripes per stage**; stage 0 consumed 504 files
> and stage 1 consumed 524, totalling the observed 1,028 against the 512 ceiling. **1,028 is
> stages 0 and 1 of a sixteen-stripe run** and implies nothing about a five-stripe plan.
>
> The 4-stripe / 116-exact figure belongs to the **2026-08-05 staging-back-pressure fixture**,
> not to gate 12. Both now carry their true provenance in the suite, and the real 16-stripe
> geometry has its own regression.
>
> The escalation this section raised — that the ledger recorded nothing about a run's *planned*
> geometry — was nonetheless upheld: Beta ruled the preflight plan **must** be persisted
> (R1 §5), and it now is.
>
> **Do not cite the 816 figure or the five-stripe inference for anything.**

**Requested:** should the planned geometry (stripe count, spans, per-phase expected sub-stripes,
and the derived requirement) be **recorded explicitly at stage setup**, so a derived bound is
auditable after the fact? Alpha proposes no mechanism; it is a small addition to the same preflight
that already computes the number.

### 2.3 No authoritative maximum shard-size contract exists — Alpha accepts and reports

Report item 9, per your §4.1. `INLINE_BYTE_LIMIT = 48 MiB` is the **inline-vs-spool selector** —
exceeding it is what *routes* a shard to the unbounded spool path — **not a maximum.** No
authoritative bound exists anywhere.

**Alpha's position: nothing to decide; this is the situation you anticipated.** No byte bound was
invented, and the byte ceiling remains runtime-enforced and protected by the §1.4 unified timeout,
exactly as *"a guessed byte bound is worse than an explicitly runtime-only bound"* requires.

---

## 3. Two scope deviations, disclosed and adjudicated by Alpha as correct

1. **`miner/range_miner_protocol.py` was needed outside the brief's expected file list.** The
   "absent field stores NULL, not 0.0" requirement is **unsatisfiable while the dataclass default
   is `0.0`** — an older peer omitting the key would decode as a genuine zero measurement, making
   "not reported" and "measured zero" indistinguishable in the ledger forever. The default is now
   `None`. Alpha judges this within R4's scope-item 3 (persist the worker-reported value) rather
   than an expansion of it.
2. **`effective_high_water_files()`'s no-plan branch degrades rather than fails closed.** Failing
   closed on that branch reds **8 existing gates** that construct coordinators without a stage
   plan. The fail-closed requirement is honoured where it was ruled — **before the first stripe is
   dispatched**, at the preflight — and the no-plan branch falls back to the configured value for
   bare-API and gate contexts that never dispatch. Alpha judges this consistent with §3.1; flagged
   so you can rule otherwise.

## 4. Not authorized by R4, and not included

`gpu_name`, `vram_bytes`, `gpu_id`, heartbeat counters, `StripeError.error`/`traceback`, and
`MinerStatusMessage` remain absent from the miner path. Per your R4 these *"can be handled
separately rather than being smuggled into the one-column amendment"* — **they were not.** Alpha
will submit them separately if you wish them addressed.

## 5. Measurement caveat, recorded in source and in the gate docstring

`elapsed_s` is **stripe service time** — sufficient for per-stripe and per-worker rate
calculations and sizing. It is **not** aggregate cluster wall-clock throughput, because concurrent
worker intervals overlap. **Fleet-level figures require an overlap-aware makespan denominator, not
a sum or average of per-stripe rates.** Alpha withdrew a ~12.8M seeds/sec estimate yesterday for
precisely this class of error and has recorded the caveat where the next consumer will read it.

## 6. Requested disposition

Approve the amendment; rule on §2.1 (Gate 37 supersession), §2.2 (recording planned geometry), and
§2.3 (accepted as reported). On approval Michael commits — which also clears Gate 22's untracked
condition — and dual-pushes. **The seed-domain/cursor amendment follows as a separate submission;
gate 12 remains held pending both.**
