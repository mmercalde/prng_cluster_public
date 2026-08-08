# CLAUDE CODE REPORT — ATTACK PLAN RE-EVALUATED UNDER THE BLACK-BOX FRAMING

**Date:** 2026-08-08 · **Host:** VM 101 `zeus-ubuntu` (`192.168.3.177`) · **Tree:**
`/home/michael/distributed_prng_analysis` · **HEAD:** `746b545`
**Type:** read-only re-evaluation. Nothing launched, no production code, no configuration changed,
nothing committed. One file written.
**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_ATTACK_PLAN_BLACKBOX_REEVAL.md`
**Under review:** `docs/CLAUDE_CODE_REPORT_ATTACK_PLAN_FROM_PROCEDURES.md` (mine, this session)
**Skill:** `tfm-project-facts` **v18**, 2026-08-08 — including **§2.20**, which did not exist when
the report under review was written.

---

## 0. HEADLINE

**Two of my three load-bearing negative conclusions were wrong, and one of them was wrong twice.**

| item | verdict |
|---|---|
| **C.4** trajectory identity | **VOID as an objection** → CHANGES into a measurement |
| **C.5** residue-width / H-B | **VOID.** Wrong under mimicry, **and** the repo had already resolved it — I did not search |
| **E8** PRNG family | **VOID as a limitation** — with one narrow residue that is real |
| **E1–E10** | 6 VOID as mimicry limits · 3 CHANGE character · 1 SURVIVES (trivially) |
| **D ranking** | reordered; **D.4 VOID as proposed**; **D.2 materially CHANGED against my own framing** |

**C.5 is the serious one.** Skill v18 §7 now records it as one of four inherited-absence-claim
failures on 2026-08-08 — instance (d), *"carried forward an 'unable-to-succeed' concern about Daily
3's three-selection spec that the repo had already resolved."* It is mine. I asserted the sieve
could search for an object that does not exist, while `survivor_scorer.py:426-428` sat committed
saying **"Daily 3 = three independent Z10 draws; score each digit position directly."** The repo
had not only considered H-B — it had **implemented** it, in the place where it belongs, and left a
comment explaining why the CRT lanes stay.

**I did not search for it.** My VIR-6 declaration named `docs/` and the governance trail as searched
surfaces, and it was true for the questions I asked — but I never asked *"has the repo already
engaged the three-selection structure?"* An absence claim is only as good as the question behind
it, and mine was never posed.

---

## 1. THE FRAME, STATED SO THE VERDICTS FOLLOW FROM IT

Verified live this session, not relayed:

| statement | anchor, read live |
|---|---|
| TFM is functional mimicry of surface output, **not** seed recovery, **not** state reconstruction | `docs/PIPELINE_BEHAVIOUR_MODEL.md:1092-1095` (§15.1 heading: *"TFM is functional mimicry — black-box, not state recovery"*) |
| *"ML does not guess. It refines a space already reduced from 2³² to 10⁴."* | `docs/BIDIRECTIONAL_SIEVE_MATHEMATICAL_WHITEPAPER.md:160-167` (§10) |
| Loose thresholds produce a **manifold** `S_near = {s : d(s,s*) ≤ ε}`; exact sieves give *"No ranking · No gradients · No learning signal"* | whitepaper `:116-131` (§7) |
| *"A survivor is a scored candidate, not a verdict."* | `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:373-377` |
| the claim is that survivorship + ranking beats the `k/1000` baseline | `evaluate_pools.py:36-41` (`random_prob = k/1000.0`; `lift = actual_prob / random_prob`) |
| `KERNEL_REGISTRY` = **44 entries / 11 base families** | `prng_registry.py:3729`, counted by AST this session: `java_lcg, lcg32, minstd, mt19937, pcg32, philox4x32, sfc64, xorshift128, xorshift32, xorshift64, xoshiro256pp` |

### 1.1 The single sentence that dissolves most of my Part C

`docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:373-377`, read live:

> Survivors may be the true seed, one of several true seeds, **a partial match valid before a
> reseed event**, or a near-consistent neighbour admitted on purpose. **Deciding which is Step 3
> and Step 5's job, not Step 2's.**

**Chapter 2 enumerates the mixture case as a legitimate survivor category.** My C.4 argued that a
window spanning two sources produces an object the search cannot contain. Chapter 2 had already
named that object, given it a category, and assigned its adjudication to a later step. My argument
did not contradict the design — it contradicted a design decision it had not read.

### 1.2 The empirical fact the brief requires me to account for

**[verified live]** `docs/TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md:36-47`: a real run produced
**6,739 bidirectional survivors**, of which **98.8% sit at the floor** of
`bidirectional_selectivity` (≤1.011) and **only ~81 seeds** are above it.

**Under my original framework, applied literally, those 6,739 are all false positives** — because
my framework's success criterion was "did we find the trajectory that produced the values", and by
that criterion nothing there qualifies.

**Under the actual design they are the deliverable of Step 1.** A population dominated by marginal
survivors is exactly what whitepaper §7 asks for: *"Looser thresholds produce a manifold of
near-consistent seeds… These seeds share structured deviations that ML can learn to rank."* A
population with 98.8% at one value has variance in the other 89 features and is the input to
ranking, not the answer.

**I will not spin this as vindication of the current configuration.** The same 98.8% figure is the
evidence in a **governed** ruling that `bidirectional_selectivity` **cannot serve as the primary
quality signal for this dataset** (`:49-50`) — the v4.0→v4.4 objective lineage (skill §2.17c).
Reported as status, not as a finding.

### 1.3 One conflict I am flagging and not re-opening

Skill v18 **§2.20** states: *"the machine is selected **per draw at random** from a pool."* Skill
v18 **§7** lists as Alpha error (a): *"asserted midday/evening were separate machines — the CA
procedures §II specify per-draw random equipment selection."*

My Part A **§A.3.1** concluded the opposite — **per draw session** — on four documentary premises
(one sealed room entry/exit per session; §II.1's *"the draw(s)"*; §V.8's single *"game set"*;
§VII.2's single `[Run LIVE (#) Draw]`).

**The brief says Part A stands and is not in scope. I am not re-litigating it.** I flag it because
a reader holding both documents will hit the contradiction. **Under the black-box framing it is not
load-bearing either way** — owner correction 3 says a two-source mixture does not break a pool, and
per-draw selection is simply a *finer-grained* mixture than per-session. Both readings land in the
same place: the manifold spans sources. **This is a documentation-consistency item for the owner
and Beta, not a technical dependency of anything below.**

---

# 2. ITEM-BY-ITEM

## Item 1 — C.4, the trajectory-identity argument → **VOID as an objection; CHANGES into a measurement**

### What I claimed

> *"`skip` can represent an output consumed; it cannot represent a different trajectory… Under two
> independent machine states, a window of `n` observations lies within one trajectory with
> probability `2^-(n-1)`… No threshold, no bidirectionality and no seed count recovers a window
> that was never generated by one trajectory."*

### Why it is wrong

**I computed the wrong probability.** `2^-(n-1)` is `P(the window was produced by a single source
trajectory)`. The quantity that decides whether the sieve returns survivors is

```
P( ∃ (seed, skip, offset) in the searched family whose emitted sequence
    matches the observed window at rate ≥ τ )
```

**These are different questions, and only the first depends on how the data was produced.** The
second is a property of the *family's expressiveness against a given string*. An interleaved string
is a string; it is not less matchable than any other string of the same length. My argument
silently substituted a question about provenance for a question about fit.

The design says this in its own words. `prng_registry.py:972-989` (read live) sweeps every
`skip ∈ [skip_min, skip_max]` for every seed and keeps the argmax; survival is `best_rate ≥
threshold` (`:997`), with `threshold` sampled in `[0.30, 0.75]`
(`distributed_config.json → search_bounds.forward_threshold`). **Nothing in that predicate refers
to the source.** It cannot: it has never seen one.

### The brief's three-way question, answered

> *"is a mimic that matches an interleaved series (a) impossible, (b) possible but rarer, or
> (c) exactly the object the manifold is supposed to contain?"*

**(c), with (b) as a quantitative rider that is a measurement, not a derivation.**

- **(c)** is settled by `CHAPTER_2:373-377` — *"a partial match valid before a reseed event"* is an
  enumerated survivor category. A mimic that tracks source A across part of a window and drifts
  where source B interleaves **is** a near-consistent neighbour, and whitepaper §7 admits it
  deliberately: exact sieves give *"no learning signal."*
- **(b)** is real but is a *quality* statement, not an *existence* statement. An interleaved series
  should be harder for a single-trajectory family to fit tightly, so the manifold should skew
  toward lower match rates. **That is a measurable property of the survivor population, and it is
  now D.3's job** (§3).
- **(a) is what I claimed, and it is false.**

### What survives of C.4

One sentence, narrowed to the state-recovery frame it belongs to:

> If the goal were to identify the machine's trajectory, a per-session (or per-draw) source
> mixture would defeat it, because no single trajectory produced the series.

**That goal is explicitly not TFM's** (`PIPELINE_BEHAVIOUR_MODEL.md:1094-1095`). So the sentence is
true and inapplicable. **C.4 as written in my report should be read as VOID.**

---

## Item 2 — C.5, the residue-width / H-B claim → **VOID**

### What I claimed

> *"If H-B holds (each digit is its own selection), then no single generator output equals the
> published three-digit value under any correct model of the machine, and the current kernel's test
> can never be satisfied by the true generator, at any seed, any skip, any window, any threshold.
> This is the strict sense of 'mathematically unable to succeed'."*

### Reason 1 — it is wrong under mimicry

**The published value is a three-digit number.** `daily3.json` stores exactly that
(`{"date": …, "session": …, "draw": 390}`, all 18,068 values in `0…999`, measured this session).
A mimic is required to **emit matching values**, not to assemble them the way the machine did.

The kernel's predicate `(output % 1000) == (residues[i] % 1000)`
(`prng_registry.py:984`) asks: *does this candidate emit the published number?* That question is
well-posed and satisfiable **regardless of how many selections the machine made**, because the
machine's assembly process is not an operand. My claim required the predicate to be a statement
about the machine. It is a statement about the candidate.

**This is deliberate, and Chapter 2 says so.** `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:695-699`,
read live — the lane table:

| Lane | Role as built |
|---|---|
| **mod 1000** | **full published value** |
| mod 8 | low three bits |
| mod 125 | the coprime complement |

*"full published value"* is the design's own description. The kernel sieves the **published
artifact**, not a reconstruction of the draw mechanism.

### Reason 2 — the repo had already engaged H-B, and I did not look

**`survivor_scorer.py:426-428`, read live this session:**

```python
:426   # Digit-wise agreement features (S119) — CA Lottery spec 03:00-09r
:427   # Daily 3 = three independent Z10 draws; score each digit position directly.
:428   # Additive alongside CRT lanes — do not remove CRT until ablation confirms redundancy.
:429   _h = float(((pred // 100) % 10 == (act // 100) % 10).float().mean().item())
:430   _t = float(((pred // 10)  % 10 == (act // 10)  % 10).float().mean().item())
:431   _o = float(((pred)        % 10 == (act)        % 10).float().mean().item())
```

and the vectorised twin at **`:616-621`**, which additionally produces
`_edc = _hd + _td + _od` — `expected_digit_match_count`, range 0.0–3.0.

Indexed at **`docs/S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md:145,152`**:
*"Digit features are documented as S119 / CA Lottery spec 03:00-09r, additive alongside CRT lanes."*

**The repo cites the same spec string I derived Part A's H-B from — `03:00-09r` — and states the
same conclusion I did: three independent Z10 draws.** It then places that structure where it
belongs under mimicry: **in the scorer's feature vector**, as `hundreds_digit_agreement` /
`tens_digit_agreement` / `ones_digit_agreement` / `expected_digit_match_count`, so the ML can learn
from digit-level structure — **without** changing what the sieve is allowed to admit. That is
strictly better than my proposal, because it adds information without narrowing the manifold.

`:428` even pre-empts the follow-on error: *"do not remove CRT until ablation confirms redundancy."*

### Verdict

**VOID, on both grounds.** My original conclusion was wrong. The residue-width mismatch I described
is not a mismatch — it is the difference between what the sieve admits (published value) and what
the scorer measures (digit structure), and both halves are built.

**Effect on D.4:** its entire premise is gone. See §3.

---

## Item 3 — E8, the PRNG family → **VOID as a limitation**

### What I recorded

> *"TFM's `java_lcg` choice has no support in this document… no PRNG family can be confirmed or
> excluded from it."*

### Re-evaluated

The first clause is **true and irrelevant**; the second is **true and irrelevant**. Both are
artifacts of a state-recovery standard, under which the family is a hypothesis about the machine
and the document's silence is a missing corroboration.

Under mimicry the family is a **basis** — the function class the search draws candidates from.
Verified live: `KERNEL_REGISTRY` holds **44 entries across 11 base families**
(`prng_registry.py:3729`, AST-counted this session). The owner's correction states `java_lcg` was
selected **empirically** — initial testing flagged it as most probable — and that if it stops
fitting, another family is tried. **A basis is chosen for fit, not for correspondence.** A
document that does not name the vendor's algorithm therefore constrains nothing.

### What would still make the family choice load-bearing — the narrow residue

**One thing, and it is a fitting property, not an identification property:**

> **Expressiveness.** If no parameterisation in the chosen family can track the observed surface
> well enough that ranked survivors beat the `k/1000` baseline (`evaluate_pools.py:36-41`), the
> family is the wrong basis — regardless of what produced the data.

That is empirical, measurable, and already has the machinery to measure it. It is also why
**"PRNG-family authority" is reserved to humans** (skill §2.13). So: **E8 is void as a limitation
on the method; the family remains a real engineering choice with a real failure mode, and the
failure mode is "poor fit", never "wrong identification."**

---

## Item 4 — E1–E10 re-marked

| # | item | limit on **mimicry**? | limit on **identification**? | verdict |
|---|---|---|---|---|
| **E1** | which of two machines produced a value | **No** | Yes — total | **VOID** as a mimicry limit. A mimic is not required to name a source it never claimed |
| **E2** | which RNG, A or B | **No** | Yes — total; not even recorded | **VOID** as a mimicry limit. Same reasoning, one level finer |
| **E3** | pre-test values, never published | **No** | Yes | **CHANGES.** Not a limit — it is *the quantity `skip` parameterises*. Skill §0.4 and `ca_d3_threshold_calibration.py:28-35` name it as skip's purpose |
| **E4** | invalid live draws, retained not published | **No** | Yes | **CHANGES.** Same class as E3 — a stride perturbation, which variable skip exists to absorb (skill §0.4: *"Variable skip is a detector, not a fitting procedure"*) |
| **E5** | perturbation rate unobservable | **Partly** | Yes | **CHANGES.** It does not limit mimicry; it forbids deriving a skip *prior* from the document and therefore **argues for keeping the skip search wide** — the opposite of what my D.2 proposed. See §3 |
| **E6** | `u`-game selection variance invisible | **No** | Yes | **CHANGES.** Same class as E3/E4 |
| **E7** | generator state across `[Shut Down]` | **No** | Yes | **VOID** as a mimicry limit. Whether the machine reseeded does not bear on whether the family can fit the values. It bears on what the manifold contains — the C.4 reframe |
| **E8** | algorithm / vendor unnamed | **No** | Yes | **VOID** — item 3 |
| **E9** | evening game order unstated | **No** | No | **SURVIVES**, trivially — P2 already proved the session gap is order-invariant. It was never a limit |
| **E10** | 19.1% of records governed (**3,447 of 18,068**, confirmed) | **No** | Yes | **CHANGES** — below |

### E10, re-evaluated

The measurement stands (the brief confirms **3,447 / 18,068 = 19.1%** on or after 2021-06-09).
**What I inferred from it was frame-dependent and is now wrong:**

> ~~"The document licenses claims about roughly 19% of the dataset."~~

Under mimicry, **the document licenses claims about the *document-derived skip prior*, not about
the data's usability.** A 2003 draw is a perfectly good mimicry target; it is a published
three-digit number. What the era boundary bounds is the validity of `skip = 9` **as a derived
value** — outside 2021-06-09+ the session inventory is not established, so 9 is an extrapolation
(my own P3/P6 said this).

**Restated correctly:**

> The 19.1% figure bounds where a **document-derived prior** may be asserted. It does **not** bound
> where mimicry may be attempted, and it does **not** make the other 80.9% less useful — only
> differently governed.

---

# 3. REVISED PART D RANKING

| rank | approach | was | now | executable within governed constraints? |
|---|---|---|---|---|
| **1** | **D.3** — measure what the manifold contains | 3 | **1** | **Yes — Stage 1 is free, read-only, no cluster time** |
| **2** | **D.2** — skip-pinned session-scoped sieve | 2 | **2**, with its rationale **inverted** | **Yes — existing bounds, no new code** |
| **3** | **D.1** — reach document-governed draws | **1** | **3** | No — F-4 coupling |
| **—** | **D.4** — per-selection residue semantics | 4 | **VOID as proposed** | — |

## D.3 → rank 1. Character changed: from *test of a fatal objection* to *measurement of the manifold*

**Old hypothesis:** *"survivor yield falls off as `2^-(n-1)`, proving the mixture defeats the
search."* That hypothesis is void with C.4.

**New hypothesis, falsifiable:** *the survivor manifold has measurable internal structure that
reflects source mixing — clustering in `best_skip`, in match-rate distribution, and in which window
positions each survivor tracks — and that structure is learnable signal rather than noise.*

**Why rank 1.** It is free, it is read-only, it needs no launch and no Beta hold cleared, and it
measures precisely the object the frame says matters. **The measurement partly exists already:**
`TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md:36-47` reports the `bidirectional_selectivity` distribution
over 6,739 survivors (98.8% at floor, ~81 above). That is one axis of manifold composition, already
measured, already ruled on. The remaining axes — `best_skip` distribution, `forward_matches` vs
`reverse_matches` per seed — are in the frozen 22-array NPZ (skill §2.3: `seeds`,
`forward_matches`, `reverse_matches`, `score` are the four per-seed columns).

**Cost.** Stage 1: **zero cluster cost.** Stage 2 (a window-length sweep) is unchanged at ~6× a
standard trial and remains confounded by F-4 — so Stage 1 stands alone.

**Caveat, stated up front:** `forward_matches`/`reverse_matches` are **absent from the Step-3 merge
list** (skill §2.3, TB: *"possibly the most consequential finding in the trace"*). So Stage 1 can
measure them off the NPZ, but they do not currently reach the ML. That is a **governed** schema
question, reported as status.

## D.2 → rank 2, and **I had its rationale backwards**

**This is a self-correction the brief did not ask for.** My report sold D.2 partly on efficiency:

> ~~"pinning skip_min=9, skip_max=10 reduces the kernel's inner loop from up to 251 skip values to
> 2 — a ~125× reduction in per-seed work… this approach is CHEAPER than the status quo."~~

**Under mimicry that is an argument against it, not for it.** Whitepaper §7 (`:116-131`) is explicit
that the search must produce a **manifold**, because *"Exact sieves eliminate all variance… No
ranking, no gradients, no learning signal."* Narrowing the skip sweep from 251 values to 2
**shrinks the hypothesis space by ~125× and shrinks the manifold with it.** Fewer survivors is not
a saving — it is less learning signal, which is the precise failure mode §0.3 exists to prevent.

**D.2 therefore is not a production configuration and I should not have implied it could be.**

**What it still is, and this is a real thing:** a **diagnostic arm** — a controlled test of whether
the document-derived value 9 is favoured over neighbours. Requirements unchanged and still binding:
a matched control arm (e.g. `{17,18}`) on the same window, session-scoped, never combined
(skill §2.10b), and the honest null caveats from my original report (era mismatch, the evening-era
`skip = 7` variant).

**And E5 now cuts against it directly:** the document forbids deriving a perturbation rate, so the
skip distribution cannot be derived — only searched. **Pinning a searched dimension to a derived
point estimate is exactly the move E5 says is unavailable.** D.2 is a hypothesis test about a
prior, run *beside* production, never *as* production.

## D.1 → rank 3

**Why it drops.** I ranked it 1 on this reasoning:

> ~~"Until this changes, no approach in this report is actually testing the document."~~

True — and no longer the point. **Testing the document was never the objective; mimicking the
surface is.** Under mimicry, older draws are **merely differently governed, not less useful**
(item 4, E10). D.1 remains the only way to test the *document-derived prior* on data the document
governs, which is worth doing — but it is a validation of a prior, not a prerequisite for the
method.

**Its blocker is unchanged and is confirmed harder by v18.** Chapter 2 **F-4** (`:1133`,
**CONFIRMED, not repaired**): `offset` drives both the host residue slice **and** the device
pre-advance from one scalar, coherent only at `skip = 0`. Skill v18 **§2.20** adds detail my report
did not have: `offset.max = 100` **has no `_note` and no in-repo rationale**; the
`agent_manifests/window_optimizer.json` block declares `max: 2000` but has **no `args_map` entry
and no CLI route**, so it is **inert**; and the manifest description *"Time offset from current
draw position"* is **wrong** (host code: head-relative index). **This is a design item, not a bound
edit** — and skill v18 §7 records "proposed `n − window_size`" as an Alpha error for exactly this
reason. I am not proposing a value.

## D.4 → **VOID as proposed**

Its premise was C.5, which is void. Worse, it **proposes something that partly exists** —
self-check #5. The digit-level structure I proposed building a new kernel for is already extracted
as four scorer features (`survivor_scorer.py:426-432`, `:616-621`), on the *correct* side of the
boundary: **in the feature vector, where it adds information, not in the sieve predicate, where it
would narrow the manifold.**

**Withdrawn.** If digit-level structure is ever wanted *in the sieve*, `:428`'s instruction governs
the sequencing — *"do not remove CRT until ablation confirms redundancy"* — and sieve
strategy/mathematics is reserved human authority (skill §2.13).

---

# 4. §6 — WHAT THE BLACK-BOX FRAMING DOES **NOT** RESCUE

Per the brief: no softening. Each of these is stated with the force it had, or more.

## 4.1 The analysis window still sees 150 of 18,068 records — **REAL, and governed**

The reframe changes *why* it matters, not *whether*. **It is no longer "we cannot test the
document"** (a state-recovery concern). **It is "the fit evidence for every survivor comes from a
150-record slice at the oldest end of an 18,068-record trajectory."**

`offset ∈ [0,100]`, `window_size ∈ [6,50]` (`distributed_config.json → search_bounds`, live), and
`offset` slices from the oldest end (`miner/range_miner_worker.py:648-649`). Max reachable index
**149**.

**Already recorded, twice** — reported as status, not as a finding:
`docs/DAILY3_CONSUMER_CONTRACT_v1.md:198-212` (*"The production sieve analyses draws from March
2000"*) and skill v18 §2.20 (`offset.max = 100` has no derivation; the manifest route is inert).

**And the index convention that makes it consequential is a Beta law**, which I did not have when I
wrote the original report — `docs/DAILY3_CONSUMER_CONTRACT_v1.md:185`:

```
offset = train_history_len          # "OFFSET DERIVED - THIS IS THE LAW (per Team Beta)"
```

with `:189-196` stating that the raw array order **is** the generator's output sequence and the
array index **is** the advance count — and that `prediction_generator.py:839`
(`next_idx = len(lottery_history)`) makes the same assumption for the forward prediction position.
Verified live: `prediction_generator.py:839,846,849` regenerate each survivor's sequence to
`next_idx + 1` and read `seq[next_idx]`.

**So the architecture is coherent and ruled:** one index convention spans sieve window, holdout
label and prediction position. **The residue is a legitimate open question, and I state it as a
question, not a defect:** a survivor's *fit evidence* sits at indices ≤149 while its *quality
label* and its *vote* are read ~18,000 positions later on the same trajectory. Whether that
distance is benign is **empirical** — it is what `evaluate_pools.py`'s lift-vs-random measures —
and it is not something I can settle from a document. **I searched for a governance record of the
distance specifically and found the index convention ruled but no adjudication of the span. That is
a statement about my search, not about the repository** (skill §1.1, `INCOMPLETE` marker).

## 4.2 The F-4 window-anchor coupling — **REAL, unchanged, and a design item**

`docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:1133`, **CONFIRMED, not repaired.** One scalar drives the
history slice and the generator pre-advance; coherent only at `skip = 0`; *"belongs in the future
hybrid input-semantics design, not a standalone arithmetic patch."* **Mimicry does not touch this**
— it is an internal coherence defect in the search's own parameterisation, independent of what the
survivors mean.

## 4.3 The dead dimensions are **WORSE** under mimicry, not better

**This is the one place the reframe makes a finding stronger.**

| chain | state | anchor |
|---|---|---|
| Optuna `skip_min`/`skip_max` → hybrid kernel | ✗ dies at `_hybrid_prefix`; `expected_skip = 5` hardcoded | skill §2.7 #4, §2.13; `prng_registry.py:1027` (read live) |
| Optuna `offset` → forward hybrid | ✗ dies in kernel args | skill §2.7 #5 |
| `skip_learning_rate` → kernel | ✗ kernel hard-adapts at 1.0 | skill §2.7 #6 |

Under a state-recovery reading a dead dimension is a tuning inefficiency. **Under mimicry it is
worse: the manifold's shape *is* the product**, and these dimensions are precisely the ones that
shape it. A sampler steering `skip_min`/`skip_max` into a hybrid kernel that ignores them is not
mis-tuning a filter — it is **failing to vary the thing the ML is supposed to learn from**, while
recording that it varied it. Skill §0.5's *"dead dimension… an autonomous agent would 'learn' into
a void"* is the same point.

**All three are governed and OPEN.** Reported as status.

## 4.4 The combined-session container order — **REAL, governed**

**[measured live, this session]** All **8,514** same-date adjacent pairs in `daily3.json` are
`(evening, midday)` — anti-chronological, since midday (live by 1:10pm) precedes evening (by
6:40pm). `load_residue_window` preserves stored order and never re-sorts
(`miner/range_miner_worker.py:641-650`).

**Mimicry does not rescue this.** A mimic tracks *a sequence*; if the sequence handed to it is
time-reversed within each date, the mimic is fitting an artifact of storage order. Owner correction
3 covers *mixtures*; it does not cover *misordering*. Already governed —
`docs/DAILY3_CONSUMER_CONTRACT_v1.md:418` (assumption 3) and skill §2.10b's prohibition — and
**strengthened** by the reframe, because the whole method rests on the array index being the
advance count (§4.1's Beta law).

## 4.5 My own D.2 efficiency argument — **withdrawn, and it was mine**

Named here rather than buried in §3: I framed a ~125× reduction in kernel work as a benefit. Under
whitepaper §7 it is a manifold contraction. **Do not carry that framing forward.**

## 4.6 What I am NOT claiming is rescued

- I am not claiming the current configuration is well-tuned. §4.3 says three of its dimensions are
  dead and §4.1 says its window is pinned to 2000–2003 with no derivation for the bound.
- I am not claiming survivors are meaningful. `TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md:49-50` rules
  that the one selectivity signal measured **cannot serve as the primary quality signal for this
  dataset**. That ruling is live.
- I am not claiming mimicry succeeds. The claim the design makes is that survivorship + ranking
  **beats `k/1000`** (`evaluate_pools.py:36-41`). Whether it does is a measurement, and this report
  performed none.

---

# 5. VERIFICATION-INTEGRITY CONTROLS (VIR-1…6)

- **execution proof:** every code and document anchor above was read live on VM 101 at `746b545`
  this session, by `awk`-with-line-numbers or `/bin/grep -n`. The `KERNEL_REGISTRY` count (44
  entries / 11 base families) was produced by **AST parse**, not by grep. Dataset figures were
  computed against the live gitignored `daily3.json`.
- **clean control:** the brief's supplied anchors were treated as **claims to verify, not facts to
  cite** (skill §7, binding). All were confirmed: `S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md:152`
  ✅ · `survivor_scorer.py:616-617` ✅ · `:426-428` ✅ · `CHAPTER_2:373-377` ✅ ·
  `PIPELINE_BEHAVIOUR_MODEL.md:1094-1095` ✅ · whitepaper `:158-167` ✅ · `prng_registry.py:3729` ✅.
- **fault-injection control:** not applicable — a re-evaluation with no detector. None was written,
  executed or claimed.
- **completion sentinel:** all six brief items addressed; every E-row marked; every D-approach
  re-ranked or voided.
- **unavailable-observer behavior:** §4.1 declares an `INCOMPLETE` search result explicitly rather
  than converting it to an absence claim.
- **audit claim scope:** claims about my own prior report, about the framing documents, and about
  the live tree. **No claim about the operator's equipment, and no new claim about the procedures
  document** — Parts A and B were not re-derived, per the brief.
- **searched surfaces:** live VM 101 tree at `746b545` — `prng_registry.py`, `survivor_scorer.py`,
  `prediction_generator.py`, `evaluate_pools.py`, `miner/range_miner_worker.py`,
  `distributed_config.json` · live gitignored `daily3.json` · **`docs/` searched by content, not by
  filename**, including a targeted search for prior treatment of the fit-window/prediction-index
  span (§4.1) which **found the governing record and stopped me making it a finding**.
- **governance trail searched:** `TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md` (the 6,739-survivor
  measurement) · `DAILY3_CONSUMER_CONTRACT_v1.md` §4.2–§4.3 and the Beta offset law at `:185` ·
  `S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md` · `CLAUDE_CODE_REPORT_PIPELINE_OVERVIEW.md`
  §6 hop 7.
- **chapters searched:** `CHAPTER_2_BIDIRECTIONAL_SIEVE.md` (§5.1 survivor semantics `:373-377`,
  §6.2 lane table `:695-699`, §12 findings F-4 `:1133`) · whitepaper §7 `:116-131`, §10 `:158-167` ·
  `PIPELINE_BEHAVIOUR_MODEL.md` §15.1.
- **unavailable surfaces:** ser8 pre-repository archives · rig-deployed source · **live survivor
  NPZs and Optuna study rows — no `results/*.npz` or `generations/*` present on this host at this
  time**, so D.3 Stage 1 remains **proposed, not performed**, and the 6,739 figure is relayed from
  a governance document with its date attached (skill §1.2) · Beta ruling texts external to the
  tree.

---

# 6. WHAT THIS REPORT IS NOT

- **Not an authorization or a request for one.** Nothing launched. Beta holds gate 12 and the
  Phase-7 soak (skill §8).
- **Not an implementation or a proposal to implement.** D.4 is withdrawn; D.2 is reclassified as a
  diagnostic; nothing is proposed for the production configuration.
- **Not a re-derivation of Parts A or B.** They stand, per the brief.
- **Not a re-opening of §A.3.1** (per-draw vs per-session equipment selection). §1.3 flags the
  conflict with skill v18 §2.20/§7 for the owner and Beta and takes no position, because the frame
  makes it non-load-bearing.
- **Not a claim that the method works.** §4.6 says exactly what is unclaimed.

# 7. THE ONE-LINE ANSWER

**My original Part C was a state-recovery critique wearing a candidate-filter caveat.** C.4 computed
the probability that the data came from one trajectory when the question was whether the family can
fit the data; C.5 asserted an unsatisfiable predicate while the repo had already implemented the
three-selection structure as scorer features and deliberately kept the sieve on the published value.
**Both are void.** What survives the reframe is narrower and sharper: the window is pinned to 150 of
18,068 records by a bound with no derivation and a structural coupling (F-4) that forbids editing
it; three tuned dimensions never reach their kernels — **and that last one matters more under
mimicry than it did under my original framing, because the manifold's shape is the product.**
