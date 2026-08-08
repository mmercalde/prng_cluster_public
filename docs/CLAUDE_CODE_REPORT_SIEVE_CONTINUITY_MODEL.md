# CLAUDE CODE REPORT — WHAT CONTINUITY DOES THE SIEVE ASSUME? (READ-ONLY)

**Date:** 2026-08-08 · **Host:** VM 101 `zeus-ubuntu` (`192.168.3.177`) · **Tree:**
`/home/michael/distributed_prng_analysis` · **HEAD:** `8bbe79e`
**Type:** read-only investigation. Nothing was launched, nothing was edited, nothing committed.
**Search order followed:** governance trail → chapters → code (binding, per brief).

---

## 0. SUMMARY — the five answers in one place

| Q | answer in one line |
|---|---|
| **Q1** | **One continuous generator state, carried across the entire window, with the observed values separated by a skip.** The state is initialised once per (seed, skip-hypothesis) pass and thereafter only ever advanced — no kernel reinitialises, reseeds or resets it between observations. |
| **Q2** | **Reseed/breakpoint/regime concepts EXIST in the trail and in code — but none of them reaches the sieve's continuity model.** Machine identity and A/B-RNG selection: **no evidence found** anywhere. |
| **Q3** | Skip is an **abstraction over several physical causes at once** — pre-test outputs, other games co-drawn in the same session, and "session overhead"/"dual-RNG stride". Stated most explicitly in the calibration harness. **Constant mode applies it uniformly across the window; hybrid mode applies it per-gap** — and `skip_min`/`skip_max` bind **only** the constant mode. |
| **Q4** | **Yes, mechanically, and nothing in code prevents it.** The `sessions` filter is *not* the mechanism that avoids the question — it is a no-op when both sessions are selected. Governance **does** address the combined case (prohibited by default, non-certifying); it does **not** address a window spanning daily power-cycles *within* one session stream. That second case: **no evidence found.** |
| **Q5** | **No.** No statement of the form *"we assume the sequence is produced by X, seeded at Y, advancing by Z"* exists. The whitepaper states X and Z formally **with no gap term at all**; Chapter 2 §5.1 supplies the gap model and then **explicitly declines** to assert single-stream continuity. The only complete, unambiguous continuity model in the project is **implicit in the kernel**. |

**The single most load-bearing sentence found in the whole search** — Chapter 2 §5.1,
`docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:421-427`:

> **What these procedures do and do not establish.** They establish equipment selection, an
> unpublished pre-test, and co-drawn evening games — that is, outputs which are **consumed and
> not published** between the values an observer sees. They do **not** establish that every
> omitted output belongs to **one uninterrupted PRNG state stream.**

The chapter states the gap, states that the gap does not establish continuity — and the kernel
assumes continuity anyway. That is not a contradiction anyone hid; §5.1 was written (2026-08-01)
precisely to stop the code being read as the design. It is, however, the exact seam the owner is
asking about.

---

## Q1 — What continuity does the sieve assume between two consecutive observed Daily 3 values?

### Answer

**Consecutive outputs of one continuous generator stream, separated by a skip.** Not "consecutive
outputs" (skip = 0 is only one point in the searched range) and not "separated by an unknown skip"
in the sense of unmodelled — the skip is an explicit, searched quantity. What is *not* modelled,
anywhere, is any possibility that the two values came from **different** state trajectories.

### The kernel — the only place the model is complete and unambiguous

`prng_registry.py:972-999` (`java_lcg_flexible_sieve`, the constant-skip forward kernel, read live
this session):

```c
:972    for (int skip = skip_min; skip <= skip_max; skip++) {
:973        unsigned long long state = seed & m;
:974        for (int o = 0; o < offset; o++)  state = (a * state + c) & m;   // pre-advance
:977        for (int s = 0; s < skip;   s++)  state = (a * state + c) & m;   // burn before draw 0
:980        int matches = 0;
:981        for (int i = 0; i < k; i++) {
:982            state = (a * state + c) & m;
:983            unsigned int output = (state >> 16) & 0xFFFFFFFF;
:984-986        if (three-lane test) matches++;
:987            for (int s = 0; s < skip; s++)  state = (a * state + c) & m; // burn between draws
            }
:992        float rate = ((float)matches) / ((float)k);
:993-996    if (rate > best_rate) { best_rate = rate; best_skip_val = skip; }
        }
```

**The continuity assumption is `:973` and the absence of any counterpart to it inside the `i`
loop.** `state` is assigned once per skip hypothesis and, from there to the end of the window,
appears only on the left of `state = (a * state + c) & m`. There is no reinitialisation, no
reseed, no branch on `i`, no session or date input, and no per-position parameter of any kind. The
window is one unbroken trajectory by construction.

Chapter 2 records the same loop and names the property that matters here —
`docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:253-257`:

> ### 2.3 Skip burn placement
> `skip` states are burned **before the first draw and between every subsequent pair** — not once
> up front. This is the single most consequential detail in the kernel loop…

and `:461-465` gives the units:

| Skip | Meaning |
|---|---|
| 0 | Every PRNG output is published |
| 1 | Every other output is published |
| N | Every (N+1)ᵗʰ output is published |

### The whitepaper — states the model with **no gap term whatsoever**

`docs/BIDIRECTIONAL_SIEVE_MATHEMATICAL_WHITEPAPER.md:31`:

> - \(G(s,i)\) is the PRNG output of seed \(s\) at position \(i\)

and `:16-17`:

> - Observed draws: \( D = (d_1, d_2, \dots, d_n) \)
> - Candidate seeds: \( s \in S \), where \(|S| = 2^{32}\) (conceptually)

**Verified live this session:** `/bin/grep -inE "skip|gap|session|reseed|machine"` over the
whitepaper returns **exactly one line — `:9`, "machine learning".** The mathematical foundation
of the system contains no occurrence of *skip*, *gap*, *session* or *reseed*. Its predicate
`G(s,i) = d_i` identifies observation index `i` with generator step `i`: **the whitepaper's model
is the strictest possible continuity assumption — every published draw is the very next generator
output.** Everything the sieve actually does about gaps is a departure from the document that is
cited as its mathematical basis, and the whitepaper was never updated to cover it.

*(This is consistent with `docs/SKIP_SEMANTICS_SEARCH_v1.md:145`, which recorded the same zero-hit
result on 2026-08-01. Re-derived here rather than relayed.)*

### Chapter 2 — states the gap model, and refuses the continuity claim

`docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:394`:

> **The published draw sequence is not an uninterrupted PRNG output stream.**

then `:421-427` (quoted in §0 above), then `:429-434`:

> **These are therefore physically motivated *candidate gaps* supporting skip as a detector —
> not proven state advances.** The observable sequence contains real structural discontinuities
> of unknown and varying size, and skip models them. **It is a physical property of the data
> source, not a tuning convenience.**

**So the documented intent and the implemented behaviour differ in strength, not in direction.**
Chapter 2 says the omitted outputs are *candidate* gaps and does not claim they belong to one
stream. The kernel *requires* that they do — a burn of `skip` steps is only meaningful if the
next observed value comes from the same trajectory `skip` advances later.

### The hybrid (variable-skip) kernel — same continuity, different stride rule

`prng_registry.py:1027-1052` (forward hybrid, `java_lcg_hybrid_multi_strategy_sieve`):

```c
:1027   int expected_skip = 5;
:1029   for (int draw_idx = 0; draw_idx < k && draw_idx < 2048; draw_idx++) {
:1030       unsigned long long state_backup = state;
:1033       int search_min = (expected_skip > skip_tolerance) ? (expected_skip - skip_tolerance) : 0;
:1034       int search_max = expected_skip + skip_tolerance;
:1035       for (int test_skip = search_min; test_skip <= search_max; test_skip++) {
:1036           state = state_backup;
:1037-1039       for (int j = 0; j < test_skip; j++) state = (a * state + c) & m;
:1040           unsigned long long temp_state = (a * state + c) & m;
:1042-1044       if (three-lane test) {
:1046               actual_skip   = test_skip;
:1047               expected_skip = test_skip;      // re-centre
:1049               state = temp_state;
:1050               break;
                }
            }
```

`state` is again initialised once (`:1022`, per strategy) and carried forward across every
observation. The only difference from constant mode is that the stride is re-chosen per gap.

**One implemented behaviour that no document in the tree describes — reported as an observation.**
On a **miss** (the `test_skip` loop runs to completion without a hit), `state` is left at
`state_backup` advanced **`search_max` = `expected_skip + skip_tolerance`** times, because
`state = state_backup` is at the *top* of the loop body (`:1036`) and nothing restores it
afterwards. `temp_state` is discarded. So a missed observation:

- advances the stream by `expected_skip + skip_tolerance`, not by `expected_skip`; and
- records `actual_skip = expected_skip` (`:1030`, `:1052`) — a value that neither matched
  anything nor equals the advance actually taken.

The **reverse** hybrid does the opposite: `prng_registry.py:3218` `else { state = state_save; }`
restores the state on every failed trial, so on a miss the stream **does not advance at all** and
`skip_seq[i] = 0` (`:3225`). **Forward and reverse hybrid passes of the same trial therefore use
different continuity rules on a miss.** `docs/SKIP_SEMANTICS_SEARCH_v1.md:237-243` recorded the
*recorded-value* half of this ("misses record a fabricated value"); the **state-advance** half
appears in no document I found. Flagged as an observation about the continuity model, not as a
defect claim — whether it matters is the owner's and Beta's call.

---

## Q2 — Does any part of the system model a per-draw machine change, an A/B RNG choice, or a per-session power cycle / reseed?

**Split answer. The concepts exist and were reasoned about; none of them reaches the sieve.**

### 2a — Machine identity / per-draw equipment selection: **NO EVIDENCE FOUND**

Searched: the governance trail, all chapters, all `*.py` in the tree, and the dataset itself.
There is no field, parameter, config key, feature, or dataset column representing which machine
produced a draw, and no code path that branches on one.

The dataset cannot carry it — verified live this session:

```
daily3.json  →  18,068 records
key sets     →  Counter({('date', 'draw', 'session'): 18068})     # exactly one shape
sessions     →  Counter({'evening': 9553, 'midday': 8515})
```

`session` is the only partitioning attribute that exists. The residue loader reads exactly two
things per record — `session` for filtering and `draw`/`full_state` for the value
(`miner/range_miner_worker.py:642`, `:650`). **`date` is never read by the sieve path at all.**

### 2b — A/B RNG per machine: **observed in analysis, never modelled**

Three trail hits, all observations, none implemented:

`docs/SESSION_CHANGELOG_20260226_S112.md:136-142` — the fullest reading of the procedures anywhere
in the project:

> ### 7. CA Daily 3 Official Procedures Analysis
> Reviewed official California Lottery draw procedures (June 2021):
> **Key PRNG-relevant findings:**
> - **Dual RNG system** (RNG A + RNG B) — redundancy with potential switching
> - **Pre-test before every live draw** — consumes PRNG state (3+ digits)
> - **"New session" per draw** — potential reseed at session start
> - **Operator + Auditor login** sequence may consume additional RNG calls
> - **"Build Animation" setting** — unknown RNG state impact
> - **Reboot/alternate ADM protocol** — explicit regime changes on malfunction
> - **Twice daily** (midday + evening) — consistent timing structure

`docs/TODO_MASTER_S120.md:76-77` (carried verbatim into S122/S125b/S126/S127):

> Context: 85 real survivors, W8_O43 confirmed optimal. Regime boundaries at draw
> counts 3 and 8 suggest dual RNG systems / pre-test draws / session resets.

`docs/TRSE_v1_15_SPEC.md:265` — proposed, not built:

> | Dual RNG switching detection | Add `rng_switch_detected: bool` alongside `regime_type` |

**Note for the record:** S112's bullet reads *"**New session** per draw — potential reseed at
session start"*, and its §7 line `:181` says *"Official procedures confirm: new session per draw"*.
That is **closer to the brief's per-draw reading than to the per-session reading the later
governance chain adopted.** See Observation 1 in §6.

### 2c — Reseed / stream discontinuity: **DESIGNED, PARTLY TRANSPORTED, NEVER EXECUTED**

This is the strongest positive finding for Q2. The concept is first-class in the strategy layer —
`hybrid_strategy.py:19-23`, present since the initial commit `0101306` (2025-11-29):

```python
:19    max_consecutive_misses: int  # How many misses before declaring breakpoint
:20    skip_tolerance: int          # Search window around expected skip (±tolerance)
:21    enable_reseed_search: bool   # Search for new seed at breakpoints
:22    skip_learning_rate: float    # How fast to adapt expected skip (0.0-1.0)
:23    breakpoint_threshold: float  # Match rate drop that indicates breakpoint
```

Two of the five presets are named for it — `:35` `'Strict Continuous (No Reseed)'`, `:53`
`'Aggressive Reseed Detection'` — and `docs/instructions.txt:1200-1204` documents the third preset
as *"Best for: Patterns with potential reseeding"*.

**Traced end to end this session. There is no consumer.**

| hop | anchor | state |
|---|---|---|
| defined | `hybrid_strategy.py:21,23` | ✅ |
| serialised into the worker payload — **legacy coordinator only** | `coordinator.py:2343-2345`, `:2502-2504` | ✅ |
| certifying miner path | `_hybrid_prefix`, `miner/range_miner_worker.py:177-193` — returns 13 elements, **none of them a reseed or breakpoint field** | ✗ not even transported |
| kernel signature | `prng_registry.py:1010-1012` — `int* strategy_max_misses, int* strategy_tolerances, int n_strategies, float threshold, unsigned long long a, unsigned long long c`; reverse hybrid `:3176-3178` | ✗ no such parameter exists |

`/bin/grep -rn "enable_reseed_search\|breakpoint_threshold"` over the tree returns only the
definition, the two legacy serialisation sites, two test fixtures and the docs. **`breakpoint`
appears in no kernel body.** The only miss-related mechanism that actually executes is
`max_consecutive_misses`, which **aborts the draw loop** (`prng_registry.py:1055-1057` forward;
`:3222-3224` reverse) — it terminates the hypothesis rather than modelling a discontinuity within
it, and because `match_rate` divides by the full `k` (`:1057`) an early abort simply lowers the
score.

This is the **same shape** as skill §2.7's recurring defect class, and `skip_learning_rate` (the
third field in that block) is already recorded there as instance 6. `enable_reseed_search` and
`breakpoint_threshold` are, as far as I can establish, **not currently recorded in §2.7** —
`docs/SKIP_SEMANTICS_SEARCH_v1.md:279-283` named all three together as "transported but never
reach a kernel", and only `skip_learning_rate` was carried into the skill. Reported as a
completeness note, not a new finding.

### 2d — Regime segmentation (TRSE): **real, executes, statistical not physical**

`trse_step0.py` clusters draw-derived features and reports `current_regime`, `regime_age`,
`regime_stable`, `switch_rate` (`:50-56`, `:280-293`). This is the project's one *executing*
model of stream discontinuity — but it is derived from the published values by clustering, carries
no machine, session-boundary or power-cycle concept, and **does not touch the sieve's continuity
model.** Its only effect on Step 1 is Rule A, the window-size ceiling
(`window_optimizer_bayesian.py:620-626`); Rules B (skip) and C (offset) are **logged only**
(`:600-602`, `:638-641`), with the in-code reason at `:601-602` being that the advisory fields are
unreliable. Skill §2.7 5b additionally records Rule A's own wiring as incomplete.

### 2e — `reseed_probability`: **live, but a statistical proxy, and downstream of the sieve**

`step6_restoration/models/global_state_tracker.py:316-320`:

```python
:316   high_variance_count = sum(1 for v in metrics.values() if v > 1.0)
:317   reseed_prob = high_variance_count / len(marker_numbers) if marker_numbers else 0.0
:319   metrics['reseed_probability'] = float(reseed_prob)
```

It is the fraction of "marker numbers" whose appearance-gap coefficient of variation exceeds 1.0
(`:304-314`) — computed over the published history, as one of the 14 `global_*` run-context ML
features. It is an *inference from output statistics*, not a model of a physical reseed, and it
is consumed at Steps 3/5, never by a kernel.

### 2f — Survivor interpretation does acknowledge reseeds

`docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:373-377`:

> Survivors may be the true seed, one of several true seeds, **a partial match valid before a
> reseed event**, or a near-consistent neighbour admitted on purpose.

Also `docs/CHAPTER_6_ANTI_OVERFIT_TRAINING.md:50` (*"Partial match | Valid before a reseed event,
may fail afterward"*) and `:684` (*"Reseeding events may invalidate some"*). **So a reseed is
modelled in how survivors are *interpreted*, and nowhere in how they are *produced*.**

---

## Q3 — What do `skip_min` / `skip_max` actually represent, in the model's own words?

### Answer: an abstraction over several causes at once — explicitly so

The most direct statement in the tree is in the calibration harness, not in a chapter.
`ca_d3_threshold_calibration.py:10-12`:

> We don't know CA's exact skip rate, dual-RNG stride, or session overhead.
> We DO know our own PRNG completely.

and `:28-35`:

> **SCENARIOS**
> Each scenario is a draw_skip value representing a plausible CA machine
> state advancement between live draw outputs:
> ```
>   skip=0   : consecutive outputs, no inter-draw overhead
>   skip=3   : ≈ one pre-test cycle for a 3-digit draw
>   skip=5   : ≈ pre-test + small session overhead
>   skip=10  : ≈ pre-test + larger overhead or dual-RNG stride
>   skip=20  : ≈ heavier session overhead scenario
> ```

Sibling harness `ca_d3_window_calibration.py:296`:

> `--draw-skip ... help="Fixed draw_skip representing CA machine overhead (default: 5)"`

Findings doc `docs/THRESHOLD_CALIBRATION_FINDINGS_S148.md:115` repeats the mapping
(*"| D | 10 | ≈ pre-test + larger overhead or dual-RNG stride |"*). Both harnesses date to
`e051ee2`, 2026-03-19.

**So, in the model's own words, one skip unit is one generator output that was consumed and not
published — whatever consumed it.** The named causes span all three of the brief's candidate
categories: unobserved draws from other games in the same session, generator outputs consumed
internally (pre-test, login, animation build), and equipment-level strides (dual-RNG). It is an
abstraction over both, and the harness says outright that the true decomposition is unknown.

Chapter 2 §5.1 names the same causes at design level (`:401-408`): the unpublished pre-test
session; per-session equipment selection; and *"The evening session draws Daily 3, Daily 4,
Fantasy 5 and Daily Derby together. Other games' outputs sit between the Daily 3 values an
observer can see."*

`docs/instructions.txt:1182-1183` gives the parameter gloss:

```
:1182  - `--skip-min INT`: Minimum skip value in pattern (default: 0)
:1183  - `--skip-max INT`: Maximum skip value in pattern (default: 16)
```

**Two readings, at two stages — documented, and still unreconciled.** Chapter 2 §5.7
(`:561-579`) states it and I re-verified both anchors:

| stage | reading | source |
|---|---|---|
| **input** (into the sieve's search) | *"Minimum/Maximum skip value **in pattern**"*, documented hybrid default `[0,16]` | `docs/instructions.txt:1182-1183` |
| **output** (sieve → Step-3 scoring) | *"Minimum/Maximum gap that **worked**"*; *"Tight skip range = stronger hypothesis"* | `docs/PROPOSAL_ML_Architecture_Remediation_v2_0.md:150-158` |

with `config_manifests/feature_registry.json` and `config_manifests/parameter_registry.json:160,166`
disagreeing about which is authoritative. This is governed and open, not a new finding.

### Uniform across the window, or per-gap?

**Both — depending on the mode, and the brief's bounds bind only the first.**

| mode | application | anchor |
|---|---|---|
| **constant** | **UNIFORM.** One `skip` for the entire window; the kernel sweeps `[skip_min, skip_max]` and keeps the best rate and the skip that achieved it | `prng_registry.py:972` (`for skip = skip_min … skip_max`), `:987` (same `skip` burned between every pair), `:993-996` |
| **hybrid** | **PER-GAP.** A stride is chosen independently at each observation from `[expected_skip − tolerance, expected_skip + tolerance]`, re-centring on each hit | `prng_registry.py:1033-1035`, `:1046-1047` |

Chapter 2 §5.3 `:479-485` confirms the hybrid mechanics and, critically, that **no pattern is
supplied**: `skip_sequences` is an **output** (`:1075-1077`), and the per-draw window is seeded by
a hardcoded `expected_skip = 5` (`:1027`) whose ancestor comment reads `// Initial guess`
(`prng_registry_pre_registry.py:696`).

**The bounds `skip_min ∈ [0,10]`, `skip_max ∈ [10,250]` in the brief are live, and they reach the
constant kernel only.** Verified this session:

- bounds are real and match the brief — `distributed_config.json` `search_bounds`:
  `skip_min {"min":0,"max":10}`, `skip_max {"min":10,"max":250}`, `window_size {"min":6,"max":50,"default":12}`;
  loaded at `window_optimizer.py:167-170`, sampled at `window_optimizer_bayesian.py:538-543`.
- they reach the constant kernel — `_constant_prefix` emits both,
  `miner/range_miner_worker.py:171-172`.
- they **die before the hybrid kernel** — `_hybrid_prefix`, `miner/range_miner_worker.py:179-193`,
  returns 13 elements and no skip bound. **GOVERNED** — skill §2.7 #4, Chapter 2 §5.4 `:496-517`,
  status OPEN. Reported here as status, not as a finding.

**Consequence for this brief's question:** when the optimizer is in hybrid mode, the sampled
`[skip_min, skip_max]` describes a gap model the kernel never receives, and the *effective* gap
model is `expected_skip = 5 ± strategy_tolerance` over the five fixed presets
(`hybrid_strategy.py:35-77`, tolerances `{5, 20, 5, 10, 50}`) — a fixed sweep that is **not** an
Optuna dimension.

---

## Q4 — Can a window legitimately span a session boundary?

### 4a — Mechanically: **yes, and nothing prevents it**

The whole window-construction path is six lines. `miner/range_miner_worker.py:641-650`:

```python
:641   if sessions:
:642       data = [e for e in data if e.get("session") in sessions]
:643   n = len(data)
:644   if n < window_size:
:645-647       raise ResidueResolutionError(...)
:648   start = max(0, min(int(offset), n - window_size))
:649   window = data[start:start + window_size]
:650   return [int(entry.get("full_state", entry["draw"])) for entry in window]
```

This is **the** canonical derivation — its own docstring says so (`:605`, `:611-619`: *"`sessions`
is a first-class INPUT, applied here and ONLY here"*), and both the worker and the coordinator side
call it, by design.

Three properties follow directly:

1. **When `sessions = ['midday','evening']` the filter at `:642` is a no-op** — every record passes,
   and the window is a contiguous slice of the combined container. Under the canonical order
   (date ascending, **evening before midday** within a date — skill §2.14) a 24-record window is
   ~12 calendar days of alternating evening/midday records.
2. **No date-continuity check exists anywhere on this path.** `date` is never read. Calendar gaps,
   the 2019-01-25 evening-only anomaly, the 1,038 single-session dates of 2000-2002, and every
   overnight power-down are all invisible by construction.
3. **The session filter therefore does not avoid the question — it only *narrows* it.** Even
   `sessions=['midday']` yields a window whose consecutive records are one calendar day apart, i.e.
   separated by a `[Shut Down]` and a fresh power-on per the procedures the brief cites, and by a
   fresh equipment selection. The kernel treats them as one trajectory either way (Q1).

**The prohibited configuration is still reachable.** `window_optimizer.py:182-186` — re-verified at
HEAD this session, unchanged:

```python
:182   self.session_options = [
:183       ['midday', 'evening'],  # Both sessions
:184       ['midday'],              # Midday only
:185       ['evening']              # Evening only
:186   ]
```

and Optuna samples across all three — `window_optimizer_bayesian.py:535-537` (`session_idx =
trial.suggest_int('session_idx', 0, len(bounds.session_options) - 1)`), applied at `:557`.

**Live evidence that combined windows were in fact produced.** `optimal_window_config.json` does
**not exist at HEAD**; the surviving artifact `optimal_window_config.json.stale_1786149572`
(untracked, read-only) carries:

```json
"window_size": 21, "offset": 66, "skip_min": 10, "skip_max": 209,
"sessions": ["midday", "evening"], "prng_type": "java_lcg",
"completed_at": "2026-05-11T19:24:23.712210"
```

A 21-record combined window ≈ 10.5 calendar days ≈ 21 power-ups. Dated 2026-05-11, so it
**predates** the ruling below.

### 4b — Governance: the combined case **is** addressed; the within-session case is **not**

Team Beta ruling 2026-07-30/31, as recorded in `docs/BACKLOG.md:109-116`:

> Midday and evening use **independently selected equipment** (draw procedures §II). There is
> therefore **no evidentiary basis for advancing one PRNG state through interleaved records.**
> Ordering is normative **within a session stream**; combined-container order carries **no
> PRNG-advance meaning.**
>
> Consequences already in force: the chronological-reorder migration was **cancelled**; combined-
> session sequential sieve is **non-certifying and prohibited by default**; production
> re-optimization is **per-session**.

Origin of the withdrawal — `docs/TEAM_ALPHA_PUSHBACK_ORDERING_AND_THRESHOLD_REGRESSION.md:34-35`:

> **Midday and evening are drawn on independently selected machines, each with its own RNS.**
> They are two generators, not two samples from one stream.

and `:39-42`:

> Interleaving two independent generators by wall-clock time does not produce a coherent
> advance sequence in **either** generator. It produces a sequence belonging to neither.

Chapter 1 §8.3.1 carries the ruling **and** the unremedied gap —
`docs/CHAPTER_1_WINDOW_OPTIMIZER.md:1121-1133`:

> | **Session scope** | Production re-optimization is **per-session**. Combined-session
> *sequential* sieving is **non-certifying and prohibited by default** | Team Beta ruling,
> 2026-07-30/31 |
>
> **Why per-session.** Midday and evening draws use **independently selected equipment**, so there
> is no evidentiary basis for advancing one PRNG state through interleaved records. …
>
> > **Known gap, reported not resolved — re-verified OPEN at `81ef3f1`.** The sampler can still
> > select the prohibited mode… **An autonomous run can therefore currently select a configuration
> > that cannot be certified.**

**I re-verified that gap at HEAD `8bbe79e` this session** (anchors in §4a): `session_options` still
offers the combined option first, and the sampler still reaches it. This is a **status**, already
governed and flagged — not a new finding.

### 4c — The part the sources do **not** answer: **NO EVIDENCE FOUND**

**The brief's question has two halves, and only one is governed.**

- *Does one seed explain draws across a **session-type** boundary (midday↔evening)?* — **Answered:
  no.** Ruled, above.
- *Does one seed explain draws across a **power-cycle** boundary within one session stream — i.e.
  consecutive evening draws on consecutive days?* — **No evidence found.** No ruling, no chapter
  section, no proposal, no code comment addresses it. The per-session remedy is stated purely in
  terms of *which records may be interleaved*, never in terms of *how long one trajectory may be
  assumed to persist*. Under the procedures the brief cites, every adjacent pair inside a
  single-session window still spans a `[Shut Down]` (§VII.8), a fresh power-on (§V.2-3), a fresh
  equipment selection and a pre-test — and the sieve, per Q1, models all of that as `skip` burns on
  one unbroken trajectory.

**The nearest thing to a treatment is empirical, not architectural.**
`docs/SESSION_CHANGELOG_20260226_S112.md:169-184` reports the discovery that real-data windows
optimise at **W8** against W256-1024 on synthetic, concludes *"real-world lottery PRNGs operate in
short-lived regimes, not as one continuous seed stream"* (`:170-171`), and lists as its fifth item
of evidence *"Official procedures confirm: new session per draw, pre-tests, dual RNG, reboot
protocols — all create regime boundaries"* (`:178-179`). That is an observation that short windows
work; it is not a statement about whether the window may cross a boundary, and it produced no
constraint on `window_size`, whose live ceiling is **50** (`distributed_config.json`
`search_bounds.window_size.max`).

---

## Q5 — Is the model documented anywhere as an explicit physical hypothesis?

### Answer: **NO.** No statement of the requested form exists.

I searched the governance trail, every `CHAPTER_*`, the whitepaper, `instructions.txt`,
`Cluster_operating_manual.txt`, the proposals and the changelog corpus for a sentence of the form
*"we assume the draw sequence is produced by X, seeded at Y, advancing by Z."* **There is no such
sentence.** The four nearest approaches each supply part of it and are explicit about not supplying
the rest:

**1. The whitepaper supplies X and Z — and no gap term.** `:4-6`, `:31`, quoted at Q1. Its model is
*published draw `i` = generator output `i`*. It is the only fully specified continuity statement in
the documentation, and it is **stricter than anything the system implements** (skip = 0 always).
It contains no *Y* (seeding) and, per the live grep, no occurrence of skip/gap/session/reseed.

**2. Chapter 2 §5.1 supplies the gap model and then explicitly refuses the continuity claim.**
`docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:394`, `:401-408`, `:421-434` — quoted in full at §0 and Q1.
Its own citation status is `UNAVAILABLE` (`:397-399`):

> Per the *California State Lottery Daily & SuperLotto Plus Draw Procedures* (effective
> 2021-06-09) — **citation `UNAVAILABLE`**: the PDF is **not in the repo** and was **not read
> this session**. The statements below are corrected from Team Beta's ruling text and are **not
> verified at source.**

**3. Chapter 2 §5.6 states the epistemic goal, and states that it was never previously written
down.** `:531-550`:

> **The goal was never to reverse state. It is to extract a fingerprint.** … Variable skip
> therefore exists to **find the windows where coherent skip structure surfaces** — the
> fingerprint glimpse … **Variable skip is a detector, not a fitting procedure.** It is not trying
> to recover the generator's state…

with its own provenance disclaimer at `:534-538` (*"It is **not** a historically discovered
repository statement** — no document in the repository ever asserted it"*) and a NOT-FOUND row at
`:559`. **This is the closest the project comes to a stated hypothesis, and it is deliberately a
statement about what the search is *for*, not about what physically produced the data.**

**4. S112 states a physical hypothesis — in a changelog, as a discovery.**
`docs/SESSION_CHANGELOG_20260226_S112.md:128-131`:

> Interpretation: Real lottery PRNG operates in short-lived **regimes**, not
> as one continuous stream. The ADM (Automated Draw Machine) likely reseeds
> periodically — new session per draw, pre-test draws consuming RNG state,
> occasional reboots and alternate machine switches.

This is the most complete physical statement I found anywhere: it names the machine, the reseed,
the session, the pre-test and the machine switch. **It is in a session changelog, it is hedged
("likely"), it was never promoted into a chapter, a proposal or a ruling, and nothing in the sieve
was built or changed to represent it.** Its architectural follow-through (`:186-196`) points
exclusively at *downstream* components — GlobalStateTracker features, Chapter 13 retrain triggers,
Chapter 14 diagnostics — and its one sieve-side row reads *"Constant/Variable Skip | ✅ Built |
`test_both_modes: true` already handles variable skip between draws"*, which treats skip as already
covering the regime question.

### Therefore, stated plainly, as the brief asks

**The model exists only implicitly in kernel behaviour.** The one place a complete, unambiguous
continuity model is expressed is `prng_registry.py:973` and the absence of any reinitialisation in
the loop that follows it. Every document either states a stricter model with no gaps (whitepaper),
states the gaps and declines to claim continuity (Chapter 2 §5.1), states the search's purpose
rather than the physics (§5.6), or states the physics in a changelog that changed nothing
(S112).

**This bears directly on the owner's decision.** Read against the sources rather than the code, the
bounds are **scaffolding with a physically motivated *direction* but no physically derived
*magnitude***:

- `skip_max = 250` has **no derivation anywhere in the tree.** The only documented default is
  `16` (`instructions.txt:1183`), the only enumerated physical scenarios reach `20`
  (`ca_d3_threshold_calibration.py:35`), and the only empirical figure is the observed winning
  range `S5-56` (S112 `:114`). The live ceiling is 250.
- `skip_min ∈ [0,10]` bounds the *minimum* gap, which under the pre-test model should be at least
  one; nothing in the tree argues for the value 10.
- `window_size ≤ 50` has no boundary-derived justification; the only relevant empirical result
  (W8, and W2/W3 on the PA dataset — `docs/SESSION_CHANGELOG_20260314_S143.md:120-124`) points the
  other way and was never made a constraint.

I am not proposing a change to any of them; that is outside this brief. I am reporting that the
governance trail contains a **rationale** for each bound's existence and **no derivation** of any
bound's value.

---

## 6. Observations — real-world conditions the model does not represent

Per the brief: **observations, not defects.** Whether any of them matters is the owner's and
Beta's call.

**Observation 1 — "per draw" vs "per session" rests on a gloss, and the ruling chain rests on the
gloss.** The brief and the governance trail quote §II **identically**:

- brief `:31-33`: *"A random number generation (RNG) program is used to select the primary and
  alternate draw equipment which will be used for **the draw**."*
- `docs/TEAM_ALPHA_PUSHBACK_ORDERING_AND_THRESHOLD_REGRESSION.md:24-27`: same sentence verbatim,
  followed by *"Equipment is selected **per draw session**"* — which is Alpha's gloss, not the
  quoted text.

The whole per-session line — Ruling 20's withdrawal, "combined-session sieve non-certifying",
"production re-optimization is per-session" — descends from that gloss. **If §II selects equipment
per *draw*, then a single-session stream is itself a mixture over machines, and per-session
scoping is not a sufficient remedy — it is the same error at finer granularity.** Note also that
S112 `:139` and `:181`, reading the same document eleven months earlier, recorded *"'New session'
per draw"* and *"new session per draw"* — the per-draw reading. I cannot adjudicate: the PDF is
**not in the repository** (`docs/PROJECT_FILE_CATALOG.md:769`; `docs/BACKLOG.md`) and was not
available to me. **This is the single highest-leverage open item the search surfaced**, and it is
already tracked as the standing request to commit the source document
(`docs/SKIP_SEMANTICS_SEARCH_v1.md:160-163`).

**Observation 2 — the two session streams have structurally different gap sizes, and one pair of
bounds serves both.** Per the brief, evening draws four games (`03:00-09r`, `04:00-09r`,
`05:01-39u`, `03:01-12u 03:00-09r`) in one session; midday draws Daily 3 alone. The between-draw
consumption is therefore structurally larger and differently composed for evening than for midday.
`skip_min`/`skip_max` are one pair applied to whichever stream `sessions` selects, and the
sampler cannot condition one on the other (`window_optimizer_bayesian.py:535-543` — `session_idx`
and the skip bounds are sampled independently). **No document addresses this.**

**Observation 3 — the A/B RNG pair is not represented.** Two RNGs per machine (brief §V.4) would
mean the observable stream may interleave two trajectories even on one machine in one session.
Nothing in the model can express that; skip can only represent *consumption from one trajectory*.
Noted in S112 and TODO_MASTER as an observation; never modelled. See Q2b.

**Observation 4 — power-down/power-up between sessions is not represented.** §VII.8 `[Shut Down]`
and §V.2-3 power-on. The kernel carries one state across every observation in the window (Q1). Skip
can represent *outputs consumed*; it cannot represent *state discarded*.

**Observation 5 — the pre-test count is asymmetric between the brief and the trail.** The brief says
*"A pre-test draw is run before every official draw (§V.14)"*; the trail says *"One automatic
pre-test session runs before an automatic Daily draw … Additional pre-test draws run only when an
anomaly requires them"* (`docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:401-404`), itself a 2026-08-01
correction of an earlier Alpha misreading that had propagated through a chapter, a sibling chapter,
a source map, the skill and three Beta submissions (`:410-419`). Both readings support skip's
existence; they imply different magnitudes. Same root cause as Observation 1.

**Observation 6 — forward and reverse hybrid disagree on what a miss does to the stream.** See Q1.
Forward advances by `expected_skip + tolerance` and records `expected_skip`; reverse restores the
state and records `0`. Not found in any document.

**Observation 7 — represented conditions, for balance.** Two of the brief's facts *are* covered by
the model: the unpublished pre-test and the co-drawn evening games are exactly what skip was built
to represent (Chapter 2 §5.1, `ca_d3_threshold_calibration.py:29-35`). The model is not blind to
the procedures; it represents the two causes that reduce to "outputs consumed on one trajectory"
and none of the three that do not.

---

## 7. Governed statuses re-encountered — reported as status, not as findings

Per skill §1.1, a defect that is known, escalated and mid-remediation is a status. Each of these
was reached during the search and is **not** offered as a discovery:

| status | governance anchor |
|---|---|
| `skip_min`/`skip_max` never reach the hybrid kernels | skill §2.7 #4 · Chapter 2 §5.4 `:496-517` · `docs/HYBRID_SKIP_BOUND_AUDIT.md` — **OPEN** |
| forward hybrids ignore `offset` | skill §2.7 #5 · Chapter 2 §7.3 `:836-840` — **OPEN** |
| `offset` is one scalar doing two jobs, coherent only at `skip=0` | Chapter 2 §7.2-7.3 `:797-834` — observed inconsistency, not a repair |
| `skip_learning_rate` configured 0.2-0.7, kernel hard-adapts at 1.0 | skill §2.7 #6 · `prng_registry.py:1047` — **OPEN** |
| `skip_sequences` discarded at the host, killing 3 ML features | skill §2.2 · `window_optimizer_integration_final.py:125` (`extract_survivor_records`) |
| combined-session option still sampler-reachable after the ruling | Chapter 1 §8.3.1 `:1127-1133` — known gap, re-verified OPEN at HEAD |
| the CA draw-procedures PDF is not in the repository | `docs/PROJECT_FILE_CATALOG.md:769` · `docs/BACKLOG.md` — open backlog item |

One completeness note that is **not** in the skill: `enable_reseed_search` and
`breakpoint_threshold` (Q2c) are the same dead-transport class as `skip_learning_rate`, which *is*
recorded as §2.7 instance 6. All three were named together in
`docs/SKIP_SEMANTICS_SEARCH_v1.md:279-283`; only one was carried forward.

---

## 8. Verification-integrity controls (VIR-1…6)

- **execution proof:** every quotation carries a `file:line` obtained this session on VM 101 at
  HEAD `8bbe79e` by `Read`, `sed -n` or `/bin/grep -n`. No line number is recalled. Dated findings
  relayed from prior reports (`SKIP_SEMANTICS_SEARCH_v1.md`, `HYBRID_SKIP_BOUND_AUDIT.md`,
  Chapter 2's own verification) were **re-derived against live source** before being restated —
  specifically the whitepaper zero-hit grep, the `_hybrid_prefix` element list, the kernel
  signatures, `session_options`, and the dataset schema.
- **clean control:** the constant-skip path is the built-in negative control throughout. It runs
  through the same files, the same `BuildContext` and the same builder, and it *does* deliver
  `skip_min`/`skip_max` to the kernel (`miner/range_miner_worker.py:171-172`). Every "does not
  reach" claim in this report is made against a sibling that demonstrably does.
- **fault-injection control:** the search method was validated against known-present targets before
  any absence claim — `enable_reseed_search` (present in `hybrid_strategy.py`, absent from every
  kernel) and `reseed_probability` (initially returning nothing under a `--include=*.py` grep, then
  **found** at `step6_restoration/models/global_state_tracker.py:319` when the surface was widened
  to `*.json` as well). That near-miss is recorded deliberately: the first grep would have
  supported a false absence claim.
- **completion sentinel:** all searches ran to completion; none was truncated or timed out. Every
  `/bin/grep` hit that underpins a claim in this report was opened and read in context, not counted.
- **unavailable-observer behavior:** the CA draw-procedures PDF is **not in the repository** and was
  not read. Every statement about it in this report is a statement about **what the repository says
  about it**, explicitly attributed, never a claim at source. The rigs were not contacted; no
  kernel was executed; no GPU was used. Reported as UNAVAILABLE, not as clean.
- **audit claim scope:** the VM 101 working tree at HEAD `8bbe79e` (tracked + untracked), plus the
  gitignored `daily3.json`, `distributed_config.json` and `optimal_window_config.json.stale_*`.
  **Repo-and-host scoped. NOT cluster-scoped, NOT source-document scoped.**
- **searched surfaces:** `docs/` in full (591 files) — governance trail (`TB_RULING_*`,
  `TB_RULING_REQUEST_*`, `PROPOSAL_*`, `TEAM_ALPHA_*`), all `CHAPTER_*`, `BACKLOG.md`,
  `PROJECT_FILE_CATALOG.md`, the `SESSION_CHANGELOG_*` corpus, `instructions.txt`,
  `TODO_MASTER_*`, `TRSE_*`; the whitepaper; all `*.py` in the tree including `miner/`, `utils/`,
  `step6_restoration/`, the calibration harnesses and the `apply_s*.py` patch corpus;
  `config_manifests/*.json` and `distributed_config.json` via `/bin/grep` and `json.load` (the
  shell `grep` wrapper ignores `*.json` — memory `grep-wrapper-ignores-json`); the live
  `daily3.json`; `git log` on the calibration harnesses and `hybrid_strategy.py`.
- **unavailable surfaces:** the CA *Daily & SuperLotto Plus Draw Procedures* PDF (not in repo);
  the three rigs (not contacted); kernel execution (none); the public clone (not fetched); Optuna
  study DBs, `scoring_chunks/*.json` and NPZ artifacts (not mined); host systemd/cron; ser8
  pre-repository archives; out-of-band discussion.
- **governance trail searched (`TB_RULING*`, `PROPOSAL*`, `TEAM_ALPHA*`):** YES — first, per the
  binding search order. Load-bearing hits: `TEAM_ALPHA_PUSHBACK_ORDERING_AND_THRESHOLD_REGRESSION.md`
  (Ruling 20 withdrawal), `BACKLOG.md:105-120` (the ruling as recorded),
  `PROPOSAL_ML_Architecture_Remediation_v2_0.md:150-158`, `SKIP_SEMANTICS_SEARCH_v1.md`,
  `HYBRID_SKIP_BOUND_AUDIT.md`, `TRSE_INTEGRATION_PLAN_S121.md`, `TRSE_v1_15_SPEC.md:265`,
  `PROJECT_FILE_CATALOG.md:127,769`.
- **chapters searched:** `CHAPTER_2_BIDIRECTIONAL_SIEVE.md` (full §1, §2, §3, §4, §5, §7, §11),
  `CHAPTER_1_WINDOW_OPTIMIZER.md` §8.3.1, `CHAPTER_1_AUDIT_v1.md` C-6, `CHAPTER_2_SOURCE_MAP_v1.md`,
  `CHAPTER_6_ANTI_OVERFIT_TRAINING.md` (reseed references), `CHAPTER_12_WATCHER_AGENT.md`
  (strategy-dict fields).
- **termination:** **PASS** (VIR-3). All five questions are answered from primary sources with
  quoted anchors. Three answers are, in whole or in part, **"no evidence found"** — Q2a (machine
  identity), Q4c (the within-session power-cycle question), and Q5 (an explicit physical
  hypothesis) — and each is scoped to the searched surfaces above rather than asserted about the
  repository at large.

**Nothing was proposed and nothing was implemented, per the brief.**
