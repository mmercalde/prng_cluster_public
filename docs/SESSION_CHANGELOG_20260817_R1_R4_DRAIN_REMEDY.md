# SESSION CHANGELOG — 2026-08-17 — R-1…R-4 DRAIN STARVATION REMEDY

**Briefs:** `~/dashboard_work/CCODE_BRIEF_R1_DRAIN_REMEDY_v1_0.md` · `…_R2_CACHED_POSITIVE_v1_0.md` ·
`…_R3_CAPACITY_BOUNDARY_v1_0.md` · `…_R4_AMENDMENT_IRREVERSIBILITY_v1_0.md`
**Origin:** Team Beta, on the MP-1 drain attribution (`c403a37`) — measurement first, then remedy.
**Report:** `~/dashboard_work/R1_DRAIN_REMEDY.md` (staged for Michael)
**Status:** **Beta CERTIFIED R-1 through R-4**; commit, deployment and a fresh-nonce acceptance run
are authorized. This changelog is written **before** the commit.
**Host:** VM101, user `michael`, `~/venvs/torch`.

```
git status --porcelain   AT START   (R-1)   (empty)      HEAD c403a373d21f2bee894ad0a5e45d2135e6da162f
git status --porcelain   AT END     (R-4)    M miner/range_miner_coordinator.py
                                             M tests/test_s172_h1h2_instrumentation.py
                                            ?? tests/test_s172_r1_drain_remedy.py

miner/range_miner_coordinator.py  sha256 = 1fd8284e1219e00902ce5dc6d9bc43fdcc92533a7f46ea746025140398d4df46
                                  (identical before and after R-4 — R-4 changed no production code)
```

**Nothing committed by the agent. Nothing pushed, deployed or launched.**

---

## 0. Result

MP-1 named the cause: `_pump_deferred` evaluated `_attempt_live_locked` **once per deferred entry**,
under `_admission_lock`, on **every** staging-job completion — and `MinerLedger._conn` opens a NEW
sqlite3 connection plus 3 PRAGMAs per query (~0.54 ms uncontended). Liveness is a **per-attempt**
property being paid for **per-frame**.

Measured on this box, reconstructing the MP-1 pathological shape (25 attempts × 68 frames = 1,700
deferred entries), one pump pass:

```
predecessor  1975.0 ms   3401 ledger reads
R-1            30.6 ms     51 reads  (67x)   <- but stages on cached positives:      BLOCKED by Beta
R-2           115.6 ms    185 reads  (18x)   <- but refuses a frame at the boundary: BLOCKED by Beta
R-3           136.4 ms    233 reads  (15x)   <- both closed; this is what ships
```

**3,401 → 233 ledger reads · 1,975 ms → 136 ms · 14.6×**, with both correctness blockers Beta raised
against the faster intermediate forms closed rather than accepted.

The reconstruction lands inside the measured band with **no fitted parameter**: 1.92 s at the
terminal backlog against MP-1's measured `pump` mean 1.461 s / terminal per-frame 2.14 s.

---

## 1. The chain, round by round

### R-1 — per-key collapse

A **pass-scoped, positive-only** memo (`live_keys`) inside `_pump_deferred`. The first entry of a
`(run_id, stripe_id, attempt)` key probes the ledger; later frames of that same key reuse the
observation. **A key observed DEAD is never recorded** — the caching is one-directional, so no drop
ever rests on a reused observation, and the memo cannot outlive the pass.

Three candidates in the brief were **killed on live source**, not adopted:

* **R-1a "stop at the first admissible entry"** — the scan admits one **ATTEMPT**, not one **ENTRY**,
  and it also performs dead-entry GC. Early exit changes both.
* **R-1b's shape "move the ledger I/O out from under the lock"** — the objective was adopted, the
  shape killed: it introduces a new interleaving, and the required re-validation is itself a
  lock-held read.
* **R-1c "do not pump on every completion"** — killed on the F1 contract: a dropped pump is a
  dropped wake, which changes resume-credit semantics.

**What was NOT changed to buy the speedup** — and this was Beta's binding constraint:

* The **one-attempt-at-a-time staging invariant stands.** No extra attempt is let through
  `_try_admit_locked`; `|_admitted| ≤ 1` is gated (G1a–G1c). Beta implicated the **cost** of
  enforcing serialization, never the policy.
* `_defer_locked`, the capacity bound, resume credit, the `_bp` counter semantics, the pump call
  **rate**, and `_deferred_retained_bytes()` are all untouched (byte-identity pins in the scope
  proof).
* **Two definitions changed — `_pump_deferred` and `__init__` — 0 added, 0 removed.** `__init__`
  gains only two `_bp` seed values.

### R-2 — memo invalidation at the grant, not at submission

Beta's blocker: a cached positive surviving past a grant, so a later frame of the same attempt could
stage after a death the predecessor would have caught.

**The fix goes further than Beta's literal wording, and Beta ratified the extension**
(*"Alpha's extension beyond my literal wording is correct"*). Discarding on successful
*submission* is insufficient, because `_on_done` releases staging slots **without**
`_admission_lock`. Invalidation is therefore placed at **`_try_admit_locked` returning True** —
before the slot acquire, so the admitted attempt re-probes on its next frame regardless of which
branch the acquire takes. Proven discriminating by mutant M10b, which passes the weaker gate (R2-1)
and reds the stronger one (R2-1b).

Cost, stated rather than hidden: 51 → 185 reads. That is what the correctness closure costs.

### R-3 — end-of-pass sweep for the capacity boundary

I had disclosed R-2's residual as *"GC latency in the conservative direction."* **Beta required it
measured, and the measurement changed the answer: I retracted the label.** Retained frames of a key
that died mid-pass are not a GC delay — `_deferred` is a **bounded capacity surface**, so the
predecessor could accept a new frame at the boundary where R-2 refuses it. A trial-fatal
`derived_count_bound` refusal, not latency.

Closure: an **O(K) end-of-pass sweep**. After the main scan, each key still in `live_keys` is
re-probed once; every retained frame of a key found dead is retired. Properties, each gated rather
than argued:

* **R3-1** capacity closure — predecessor and R-3 accept the same frame at the boundary.
* **R3-2** the sweep is a **no-op under quiescence**, so every prior differential stays valid.
  (Beta's condition — where predecessor and R-3 do *not* differ, the invariant making them identical
  is **demonstrated by the gate**, not inferred.)
* **R3-3** per-key complexity preserved: reads sit in `probes … 2·probes + 1`.

R-3 retains **61 ≤ 62** entries versus the predecessor — the only direction that cannot manufacture
a refusal.

### R-4 — two proof gates, no code change

Beta requested **no** production change; the digest above proves byte-identity to R-3.

**R4-1 — H1/H2 supersession, paired-amendment model.** The certified gate pinned
`src.count("released.append(") == 2`, correct for its source anchor; R-3 legitimately adds a third
departure class. **I declined to route the sweep's charge through `released.extend(...)` to restore
the count** — that would have gamed a detector into blindness. Beta agreed `2 → 3` is the wrong
amendment (same brittle shape, fails on the next legitimate class), and authorized replacing the
count with a **property gate**: *every frame leaving `_deferred` is charged exactly once, after the
lock is released and before staging submission.* Non-vacuity is **earned**: each `released.append`
site is AST-deleted from live source in turn and the property re-evaluated —

```
LIVE            -> no violations
site 0 deleted  -> 5 violations, first: dead-in-main-scan   ('s0',0,0) charged 0 times
site 1 deleted  -> 4 violations, first: resumed-into-ready  ('s0',0,0) charged 0 times
site 2 deleted  -> 5 violations, first: dead-in-r3-sweep    ('s1',0,0) charged 0 times
```

One-to-one with the three classes: every charge is load-bearing **and** every class is exercised.

**R4-2 — R3-4, same-attempt irreversibility.** R-3 takes one negative observation and drops every
retained frame of that key; the safety argument rests on a swept key never becoming live again at
the **same** `(run_id, stripe_id, attempt)`. That was source-reviewed, not gated. R3-4 gates it:
one `claim_stripe` caller (`schedule_pending_stripes`), a pending-only selector, requeue **advances**
the attempt, reclaim touches nothing terminal — and a mutant whose selector also yields FAILED rows
**resurrects the key**, redding arm 4.

**G4b was NOT weakened.** The ledger primitive's non-monotonicity stands — `claim_stripe`'s SQL does
accept `failed → claimed`, and the R3-4 mutant is built from exactly that capability. The two gates
are complementary: R3-4 proves the **production scheduler** does not exercise it for a swept key.
If a same-attempt reclamation path is ever added, R3-4 reds and G4b stays green.

---

## 2. Verification

* **Battery `tests/test_s172_r1_drain_remedy.py` — 44/44** (new, untracked). Anchored to
  `PINNED_COMMIT = c403a373d21f2bee894ad0a5e45d2135e6da162f`.
* **Differential oracle**: the verbatim **pre-patch** pump reconstructed via `git show` and `exec`'d
  against the production module's `__dict__` as globals (the A8-B2 lesson), so old and new are
  compared in one process against the same fixtures.
* **AST scope proof**: per-definition SHA-256 with docstrings blanked — exactly two definitions
  changed, none added, none removed. New callees are a declared, shape-pinned allowlist
  (`set`/`add`/`discard`/`len`), not a weakened gate.
* **Mutants** M1, M3, M4, M5, M7, M8, M8b, M9, M10, M10b, M11 — each reds a named gate.
* **Complexity falsifiability on the acceptance run**: two `_bp` fields,
  `deferred_distinct_attempts_high_water` and `pump_liveness_probes_high_water`, computed **outside**
  `_admission_lock` and **inline** (MP-1's scope proof asserts the added-definition set exactly, so
  no helper method). At the MP-1 shape they read **25** and **116** against a population of **1,700**.
  `deferred_distinct_attempts_high_water > 25` **refutes** the `K ≤ fleet size` bound outright.

### Regression (sequential — concurrent runs flake Part B G-VAL-6 on a free-space race)

| suite | result |
|---|---|
| `test_s172_staging_backpressure` (S172-BP) | **50/50** |
| `test_s172_staging_partb` | **24/24** |
| `test_s172_f1_f2_active_lease` | **16/16** |
| `test_s172_h1h2_instrumentation` | **62/62** (with the R-4 property gate) |
| `test_s172_mp1_drain_attribution` (MP-1, the oracle) | **38/38** |
| `test_s172_attempt6_remediation` | **78/78** — flake note below |
| `test_s172_admission_liveness` | **16/16** |
| `test_gate12_gpu_gate` | **9/9** |
| `test_gate12_cleantree_admission` | **31/31** |
| `test_s172_phase4_coordinator` | **62/63** — Gate 22 only |
| `test_s172_phase5_d6_production_adapter` | **0/9 — red at the base, not chargeable** |

**Phase-4 Gate 22** lists **both** edited test files: the new battery (untracked) *and*
`test_s172_h1h2_instrumentation.py` (**modified tracked**). Expected per §2.33 — the detector reads
`git status --porcelain`, so "committed once ⇒ clears forever" is false. It self-clears on a clean
committed tree; the allowlist was **not** widened.

**D6 0/9 is pre-existing.** Attributed by differential worktree at `c403a37` — identical failure
list, zero differential. Root cause: `_build_run` reads `rec["expected_substripes"]` straight off
`assign_stripes`, which since **F1** no longer claims, so that key is filled only for placed
stripes. A stale fixture against the F1 scheduler split, not a production defect. **The briefs'
expected "D6 82/82" is stale** — the suite has been red since F1 landed. Needs its own pass.

---

## 3. Carried forward — reported, NOT fixed

**1. Production docstring debt — `miner/range_miner_coordinator.py:7741`.**
*(R-4's brief cites `:7441`; the live line is **7741** — verified.)* Inside `_pump_deferred`'s
docstring:

> *"A key observed DEAD is never recorded, so every entry that is DROPPED is dropped on its own
> fresh, under-lock `_attempt_live_locked` call — byte-identical to the old loop."*

R-3's sweep makes the *"its own"* clause false. Beta required production byte-identity and said to
stop and report if anything needed a code change, so this is the report. Docstring-only, no
behaviour, no digest-bearing logic. **Recommended wording, to ride the first commit that legitimately
touches this file:**

> *"A key observed DEAD is never recorded, so no drop ever rests on a REUSED observation. R-3's
> end-of-pass sweep deliberately retires every retained frame of a key on ONE fresh negative probe;
> what holds per-entry is that the observation behind it is never reused."*

The corresponding sentence in the **test** header was corrected under R4-3 — it now states the two
things that are exactly true (no drop rests on a reused observation; R-3 retains ≤ predecessor) and
defers irreversibility to R3-4 instead of asserting it in prose.

**2. `attempt-6` RXP-1/4 is a pre-existing flake.** One run reported `77/78` —
`RXP-1/4 mutual exclusivity: _inject_E7 produced 'SHUTDOWN_STOP', already produced by
'SHUTDOWN_STOP'`. Five runs on the **unchanged** tree: **77/78, 78/78, 78/78, 78/78, 78/78** —
non-deterministic. Attribution does not rest on the tally: R-4 changes no production code (digest
identical) and `test_s172_attempt6_remediation.py` imports **neither** edited test file, so a
worker-session injection race is unreachable from a test-only diff. Not fixed — out of scope.

**Still explicitly out of scope, per the R-1 brief** (Beta tracks these on separate branches):
the **heartbeat disposition** defect (`heartbeats_accepted = 0` across 59 records — all 550
heartbeats reached the drain with zero identity mismatches and zero fence drops, so they were
disposed of after arrival, not lost in transit; **the faster drain does not make this moot**), and
the **missing `STRIPE_RX_SUMMARY`** for `st1_s30` (a clock skew — the expiry report uses the
iteration's shared `now` while `process_lease_expiry` reads a fresher clock; shorter iterations make
the miss *rarer*, which is worse than leaving it alone. **A reduced miss rate on the next run must
not be read as closure.**)

---

## 4. Acceptance oracle — predictions, stated in advance

MP-1 stays live. Every prediction reads off fields that already exist; none needs new instrument code.

| field | MP-1 (broken) | prediction |
|---|---|---|
| `pump` exclusive, staging ×4 | ~3,640 s | **< 100 s** |
| `pump` calls | 2,492 | **unchanged ±20%** (R-1c killed — the rate is deliberately untouched) |
| serve loop blocked in `staging` | 681.2 s | **< 60 s** |
| per-sub-result lock block | 0.366 s | **< 0.02 s** |
| `msg_seconds_per_frame`, first → last window | 0.005 → 2.14 s (~400×) | **no build-up: last within 5× of first** |
| staging share of window | 0.6% → 100% | **< 25% terminal** |
| `drain_passes_partial` | 1,762/1,771 (99.5%) | **< 20%** |
| `conns_serviced` / positions | 1 (position 1 only) | **> 1, positions spread above 1** |
| `capacity_invariant_terminations` / `capacity_timeout_terminations` / `pause_events` | 0/0/0 | **0/0/0** — any non-zero is a regression that outranks every timing gain |
| `deferred_distinct_attempts_high_water` (NEW) | not measured | **≤ 25** — above 25 **refutes** the bound |
| `pump_liveness_probes_high_water` (NEW) | not measured | **≈ 116** at gate-12 geometry — tracking the deferred **population** would mean the memo is not working |

**What would show the remedy FAILED** (§6.1 of the report, stated before the run so it cannot be read
charitably): `pump` exclusive still in the thousands; `msg_seconds_per_frame` still building ~400×
while `pump` drops (next suspect: `_deferred_retained_bytes()` under the lock in `_defer_locked`,
measured 0.879 ms at 1,700 entries and deliberately left alone); `drain_passes_partial` still ~99%;
probes tracking the population; or any capacity/pause counter going non-zero.

---

## 5. Files changed (build the `git add` list from HERE, not from recall)

```
miner/range_miner_coordinator.py         modified — _pump_deferred (R-1 memo, R-2 discard-on-authority,
                                                   R-3 end-of-pass sweep) and __init__ (2 _bp seed
                                                   values). 2 changed, 0 added, 0 removed.
tests/test_s172_r1_drain_remedy.py       NEW      — 44-gate R-1/R-2/R-3/R-4 battery (untracked)
tests/test_s172_h1h2_instrumentation.py  modified — E2 count gate superseded by the R-4 property gate
docs/SESSION_CHANGELOG_20260817_R1_R4_DRAIN_REMEDY.md   NEW — this file
```

Next, per Beta and Michael only: **commit → archive → deploy ten files → parity 30/30 → fresh nonce
→ attempt 9.**
