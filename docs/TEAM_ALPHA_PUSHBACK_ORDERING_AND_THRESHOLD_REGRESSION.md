# TEAM ALPHA → TEAM BETA — pushback on Rulings 20 and 21, and a live regression that outranks both

**Re:** `daily3` consumer-contract rulings. Alpha requests reconsideration of **Ruling 20**
(canonical ordering) on the grounds that its central premise is **factually incorrect**, and
partial re-grounding of **Ruling 21**. Alpha also reports a **live regression** found by the
threshold-path audit that outranks both in priority.

---

## 1. Ruling 20 — the premise is wrong: midday and evening are different PRNGs

Beta's argument rests on one stated premise:

> *"The combined array is treated as a temporal PRNG-output sequence… Step 3 derives PRNG
> advancement from array position… evening-before-midday is not a harmless serialization
> choice. It inverts two events on every dual-session date."*

That reasoning is valid **only if the combined array is one generator's output stream.**
It is not.

**External authoritative evidence:** *California State Lottery — Daily & SuperLotto Plus
Draw Procedures*, effective 2021-06-09 (in evidence).

- §II: *"A random number generation (RNG) program is used to select the primary and
  alternate draw equipment which will be used for the draw."* Equipment is selected **per
  draw session**, recorded on the Draw Games Certification, and verified by the external
  auditor.
- Draw Timing table: **midday D3** is a separate session (live draw 1:00:10pm) from
  **evening D3/D4/F5/DD** (live draw 6:30:05pm), each with its own draw-room entry,
  pre-test, equipment selection, and certification.
- §V: separate pre-test and `[Start New Draw Session]` per session; draw specs
  (`CA Daily 3 03:00-09r`) entered per session.

**Midday and evening are drawn on independently selected machines, each with its own RNS.**
They are two generators, not two samples from one stream.

### What follows

Interleaving two independent generators by wall-clock time does not produce a coherent
advance sequence in **either** generator. It produces a sequence belonging to neither.
Whether evening or midday appears first within a date is **immaterial to a PRNG that never
emitted the other draw.**

The session flags exist for precisely this reason. Michael's design intent: keep one file,
flag the sessions, and separate the streams when analysis requires it — which is what
`load_residue_window(..., sessions, ...)` already does and what the split files are for.
**The combined file is a container of two lineages, not one timeline.**

### The cost Beta's ruling would impose, for no benefit

Ruling 20 requires a new immutable lineage, a correction/migration manifest, a new
input-manifest digest, a **clean computational accumulator lineage**, and re-derivation of
all combined-session configs and artifacts — including discarding the tuning work behind
`optimal_window_config.json`.

Alpha's position: that migration buys **nothing**, because it imposes a false chronological
unity on two independent sources. Alpha requests Ruling 20 be **withdrawn or re-scoped**.

### What Alpha agrees does matter

**Single-session coherence.** Within `sessions=["midday"]` (or `["evening"]`) the sequence
must be strictly chronological, gap-honest, and duplicate-free — *that* is where array
position can carry PRNG meaning. Alpha proposes replacing Ruling 20 with:

> Ordering is normative **within a session stream**: strictly ascending by date, no
> duplicates, no reordering of history. Combined-session ordering is a **presentation
> detail** and carries no PRNG-advance semantics, because the two sessions are drawn on
> independently selected equipment.

Consumers should still stop silently normalizing — Beta is right that
`backtest_pools.py:33` re-sorting and `validate_survivors.py:170` reversing are defects.
Alpha accepts that half of Ruling 20: **validate, do not normalize.** The disagreement is
only about which order is canonical and whether a migration is warranted.

### An open question Alpha cannot answer and does not assert

If the two sessions are different machines, what does a **combined-session** sieve window
mean? A window spanning both mixes two generators' outputs. This may be intentional (both
machines run the same IGT RNG family, so the search targets the algorithm rather than one
seed stream), or combined-session analysis may be an artifact of convenience. **Alpha does
not know and will not assert.** It bears directly on Ruling 21's re-optimization
instruction and should be settled deliberately.

---

## 2. Ruling 21 — the conclusion may stand, but the grounding must change

Alpha does **not** contest that `offset` semantics are ambiguous and that the `0…100`
search bound confined every combined-session trial to the head of a 26-year dataset. Those
are real, and explicit `window_anchor` in config is an improvement.

But two of Beta's supporting points need correction:

1. Beta argued March 2000 is unintended partly because *"the reverse sieve reverses that
   ordered window to look backward in time."* Per the `tfm-project-facts` skill §2.4 and
   the Reverse Sieve Epiphany strategy, the reverse sieve **does not walk backward in
   time** — the kernels iterate the PRNG **forward**; direction comes from `residues[::-1]`
   on the host, because most PRNGs are not invertible without full state. Beta's temporal
   framing does not support the ordering conclusion.
2. **Re-optimization cannot be specified until §1's open question is settled.** Re-running
   the window optimizer under tail-relative semantics on a *combined* sequence would
   re-tune against a mixed-generator window.

Alpha therefore requests Ruling 21 be **retained as to the config schema** (explicit
`window_anchor`, separate `window_offset_draws` from `holdout_prng_offset` — that naming
separation is a genuinely good catch) and **deferred as to re-optimization** until the
combined-vs-single-session question is answered.

---

## 3. The finding that outranks both — a live, silently reverted regression

`docs/THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md` (audit complete at HEAD `91f0521`).
Verdict on skill §2.7: **CONFIRMED — and it is a regression, not a never-fixed defect.**

- **Fixed** 2026-04-30 in `3fdf434` ("S172: Fix Optuna threshold-drop bug").
- **Silently reverted** 2026-07-07 by `2389b61` (S172 Phase 0, PRNG_TYPE_ENCODING v3.2),
  which rewrote `window_optimizer_integration_final.py` **from a pre-fix copy**.
  `git log -S "getattr(config, 'forward_threshold'"` returns exactly those two commits —
  added, then removed. **The Phase 0 commit message never mentions thresholds.** It also
  reverted `run_bidirectional_test` defaults `0.50 → 0.01`.
- **Where it drops:** `window_optimizer.py:481-482` passes three positional args;
  `window_optimizer_integration_final.py:2264-2265` defaults `ft`/`rt` to
  `bounds.default_*`; `config.forward_threshold` is **never read**. Route B
  (`--n-parallel>1`) is worse: `:1835-1841` builds a `WindowConfig` with the sampled
  values, then `:1798-1799` **explicitly overrides them with the defaults**. `3fdf434`
  never covered that call site.
- **Empirical proof:** `optuna_studies/window_opt_1778552567.db` records trial 1 at
  `forward_threshold=0.73, reverse_threshold=0.31`; the same pair appears in
  `optimal_window_config.json → suggested_params`. **Neither value ever reached a kernel.
  Runtime is 0.30/0.30. TPE has been optimising a dead dimension.**

### This corrects Alpha's own brief, in Beta's favour and against it

**Constant-skip four-path parity is NOT at risk** — Alpha's stated worry was wrong. All
four backends (legacy/PWC/ZMQ/miner) read the *same two parameters* of
`run_bidirectional_test`; the drop is **above** that function, so all four run at 0.30
together. **Do not "fix the miner to match the oracle" on this axis.** D6 is not defeated;
it is out of scope. What is corrupted is **provenance and optimisation**, not path
agreement.

**Variable skip is where the oracle actually breaks — and it is a *three-way* split, not a
two-way one.** Verified hop by hop in the audit §7/F6:

| route | hybrid threshold | mechanism |
|---|---|---|
| **PWC** | **0.50** | `run_sieve_pass(… phase2_threshold: float = 0.5 …)` (`persistent_worker_coordinator.py:1119`); both hybrid call sites (`:1699-1710`, `:1726-1739`) pass `threshold`/`reverse_threshold` but **never** `phase2_threshold`, so the default stands → `sieve_gpu_worker.py:258` receives `0.5`, not `None` → `hybrid_threshold = 0.50` |
| **Legacy coordinator** | **0.30** | `coordinator.py:2280` reads `getattr(args, 'hybrid', False)`; the integration's `Args` class (`window_optimizer_integration_final.py:1320-1341`) **never sets** a `hybrid` attribute → `use_hybrid=False` → `:2298` sets `phase2_threshold=None` → worker falls through to `hybrid_threshold = threshold` |
| **Miner** | **pinned equal**, fail-closed at both ends (D6) |

Grep confirms **no caller anywhere** supplies `phase2_threshold` to `run_sieve_pass`.

**A separate latent inconsistency in the legacy route:** `coordinator.py:744` independently
sets `'hybrid': '_hybrid' in job.prng_type` → **True**, so the hybrid *kernel* is selected
while its *threshold key* is `None`. The two hybrid signals are derived from different
sources.

**Consequence for bounded Phase 6:** with `test_both_modes=True` and the same trial config,
PWC filters hybrid survivors at `0.50` while the miner and the legacy coordinator filter at
`0.30`. PWC returns **strictly fewer** hybrid survivors, and because PWC is the
authoritative comparator, **the miner will look like it is over-producing.** This is the
broken-oracle scenario — living in the hybrid threshold key, not the constant one. It is a
**bounded-Phase-6 blocker.**

**TRSE produces no threshold candidates at all** — regime/window quantities only. Rule A
moves `bounds.max_window_size`; Rules B/C are logged-only. The one tool that calibrates
sieve thresholds (`ca_d3_threshold_calibration.py`) only **prints** recommendations telling
a human to hand-edit `default_forward_threshold` — which is why the drop stayed invisible
for four months. `expected_skip = 5` is confirmed hardcoded in the live hybrid kernels,
with **no `skip_min`/`skip_max` in their signatures** (constant kernels have them).

**Note:** `s172_threshold_patch.py` is still in the tree and its FIX 2 anchor still matches
live text — but re-running it alone would leave Route B broken. That is a separate fix
brief.

**VIR-6 scope:** repo-scoped verdict complete at `91f0521`. Deployed copies verified
byte-identical on rrig6600 CT100 `.122` **only**; `.156` and `.164` have no
`~/distributed_prng_analysis` and no `sieve_gpu_worker.py` within `find -maxdepth 3` —
declared **UNAVAILABLE**, not clean. Systemd/cron not searched; no provisioning-scoped
claim is made.

---

## 4. Rulings requested

1. **Withdraw or re-scope Ruling 20.** The premise that the combined array is one PRNG
   sequence is contradicted by the CA Lottery draw procedures. Alpha proposes normative
   ordering **within a session stream**, with combined-session order treated as
   presentation. Alpha accepts the *validate-don't-normalize* half.
2. **Retain Ruling 21's config-schema requirements** (explicit `window_anchor`;
   `window_offset_draws` vs `holdout_prng_offset`); **defer re-optimization** until §1's
   open question is answered.
3. **Answer the open question:** is combined-session analysis intended, given two
   independently selected machines? This determines whether re-optimization is
   combined-sequence or per-session.
4. **Rule on priority.** Alpha's recommendation: the threshold regression (§3) is a live
   correctness defect actively corrupting an optimisation loop and blocking bounded
   Phase 6 on the hybrid key. It should precede the schema-freeze work. A regression
   silently reintroduced by an unrelated commit is also, on its own, a process finding —
   a fix landed and was reverted with no gate detecting it for four months.
5. **Ruling 22 (freeze/provenance), Ruling 23 (split files as bound derived views), and
   the D1/D2/D8 corruption blockers are accepted without objection** and are not contested
   here.
