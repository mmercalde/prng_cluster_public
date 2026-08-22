# Chapter 2: The Bidirectional Sieve (Step 2)

## TFM Pipeline — Operating Guide

**Version:** 4.2.0 — closed
**Status:** **CLOSED at `81ef3f1`, 2026-08-02** — verified-and-bounded, not finished. Restored
from `d14dcdd`, audited at `eed3904`, corrected at `e50e35f`, closed at `81ef3f1`. See **§14**
for the closure statement, what remains open, and the closure sentinel.
**Subject:** Step 2 — the bidirectional residue sieve, and RANGE-MINER, the engine that
now executes it
**Authority for this pass:** `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_1_AND_2_CLOSURE.md` (REV1).
Prior passes: `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_2_RESTORE.md` (REV1); reconnaissance
`docs/CHAPTER_2_SOURCE_MAP_v1.md`

---

## 0. What this chapter is, and how it was produced

### 0.1 The recovery

This chapter was destroyed, not lost. At `d14dcdd` it was 743 lines (§1–14). The commit
`248e48c` — *"chore: move CHAPTER docs to docs/ folder"* — copied a **34-line root-level
fragment over the 743-line chapter** and deleted the root file, leaving §14 alone. Two later
commits appended §15 twice verbatim, producing the 128-line fragment that stood until now.

The content below §1–§6 and §9 was recovered with:

```
git show d14dcdd:docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md
```

**This is the same defect class the project has already named once** — the stale-copy
overwrite that silently reverted the threshold fix at `2389b61`
(`docs/TFM_PROJECT_FACTS_SKILL.md` §2.7 #2). Two known instances, same mechanism, both
invisible in the commit message.

### 0.2 Recovered is not verified

**Every recovered line is pre-S172.** The original chapter described `sieve_filter.py` /
`GPUSieve` as *the* engine; RANGE-MINER did not exist when it was written. This pass therefore
did three different things to three different parts of it:

| § | disposition | basis |
|---|---|---|
| 1–4 | **recovered, verified, corrected in four places** | live kernels + whitepaper |
| 5 | **recovered and EXTENDED** — §5.1 and §5.6 are new and exist nowhere else | live source + the physical model |
| 6 | **recovered, verified, one central claim corrected** | live kernel text |
| 7 | **new** — the `offset` disposition Chapter 1's audit deferred here | live source |
| old 7–13 | **re-scoped, not restored** — superseded engine; replaced by §8, which cites | source map §1 |
| old 14 | **retained, corrected, re-scoped** → §9 | live source + live hosts |
| old 15 | **superseded** → §10 | PWC retirement, 2026-07-31 |

Section 12 lists what this pass changed and why. Section 13 is the verification declaration.

### 0.3 Boundary — whitepaper vs chapter

`docs/BIDIRECTIONAL_SIEVE_MATHEMATICAL_WHITEPAPER.md` (167 lines) says **why** bidirectional
sieving works. **This chapter says what the system does when it runs one.** Mathematics is
cited here, never restated. Where the two genuinely diverge, §3.5 names the divergence and
does not resolve it — that is a question for Team Beta, not for a description of as-built
behaviour.

### 0.4 Terminology

This is **TFM — Triangulated Functional Mimicry**: functional mimicry of PRNG surface output.
It is **not seed recovery**, and §5.6 is the section that explains why that distinction is
load-bearing rather than cosmetic. Where a cited document's own title uses other wording, the
title is reproduced exactly so it remains findable.

---

## 1. Mathematical Foundation

### 1.1 The observable-data problem

The draw operator runs an internal PRNG and publishes only a reduced value. For the family TFM
actually sieves — `java_lcg` — the live kernel is explicit about the widths
(`prng_registry.py:958-1004`):

```
internal LCG state      48 bits    state = (a*state + c) & 0xFFFFFFFFFFFF     HIDDEN
extracted output        32 bits    output = (state >> 16) & 0xFFFFFFFF        HIDDEN
published draw           0..999    compared as output % 1000                  VISIBLE
```

**Correction against the recovered text.** The original §1.1 described a *32-bit internal
state* reduced mod 1000. That is wrong for `java_lcg`: the state is **48-bit**, and the 32-bit
quantity is the *extracted output*, not the state. The recovered arithmetic downstream is
unaffected — it was always about the output space — but the width statement was not.

`a = 25214903917`, `c = 11` — **two anchor corrections at closure**
(`81ef3f1`, 2026-08-02):

- **The default-params anchor was wrong.** `prng_registry.py:1004` is the closing `'''` of the
  `JAVA_LCG_KERNEL` source string, not a parameter block. The `java_lcg` `default_params` live in
  the registry dictionary at **`:3963-3966`** (and `java_lcg_hybrid`'s at `:3976-3979`). Cite the
  entry by key — `PRNG_REGISTRY['java_lcg']['default_params']` — because the registry dict is far
  from the kernel strings and moves independently of them.
- **The count was wrong: it is two kernels, not three.** `a` is hardcoded in exactly **two**
  kernel bodies — `java_lcg_reverse_sieve` (`:3125-3126`) and `java_lcg_hybrid_reverse_sieve`
  (`:3182-3183`). Verified by counting every occurrence of the literal in the file: six sites,
  of which two are the CUDA hardcodes above, two are Python `kwargs.get('a', 25214903917)`
  defaults (`:113`, `:172`), and two are the registry `default_params` entries (`:3964`,
  `:3977`). **Both hardcoded kernels are reverse kernels**; the forward kernels take `a`/`c` as
  arguments. That is the same forward/reverse asymmetry §3.4 describes as the ABI consequence,
  and it is the reason the number matters — a third such kernel would have meant a forward path
  was also frozen, which it is not.

**The seed domain is not 2³².** The kernels mask the seed to 48 bits (`state = seed & m`,
`prng_registry.py:973`, `:3132`). The recovered summary's "starting seed space:
4,300,000,000 (2³²)" describes the *output* space. What is actually scanned is a
**configured range**, partitioned as contiguous macro-stripes over
`[base_start, base_start + total_seeds)` (`miner/range_miner_coordinator.py`
`partition_macro_stripes`) — not the whole domain.

**One as-built detail that is inert in production but has a known, deliberate purpose.** The
residue loader resolves each observed value as `entry.get("full_state", entry["draw"])`
(`miner/range_miner_worker.py:650`). **The live dataset carries no such field** — verified this
session: 0 of 18068 records in `daily3.json`, whose keys are exactly `{date, session, draw}`.

**`full_state` is not a "forward-compatibility seam."** It is the deliberate **synthetic
known-answer / multi-modulo validation hook** identified in the **Wall C ruling**. Synthetic
fixture generators plant the exact kernel output in that field precisely so a known-answer
test can bind unambiguously:

- `create_synthetic_full_state.py:27` writes `"full_state": int(state)`, its docstring stating
  the dataset carries full 32-bit state values *"not just mod 1000"* so that **"the multi-modulo
  sieve … work[s] correctly"** (`:3-4`). Also `variable_skip_dataset.py:92`,
  `create_synthetic_first30.py:20`.
- `tests/phase6/known_answer_reference.py:409-411`: *"Planting these as `full_state` guarantees
  the seed matches at rate 1.0 at that skip."*
- `tests/phase6/known_answer_gate.py:211-216`: *"`full_state` carries the exact 32-bit kernel
  output so the plant matches unambiguously"* — and records that `draw` must still be written
  because the canonical derivation evaluates `entry["draw"]` **eagerly** and raises `KeyError`
  without it.

**This is the second time the field has been mischaracterised in this chapter's lineage.** Its
purpose is known and documented; **inert in the live dataset is not the same as unexplained.**

**What it changes, stated precisely.** It changes the **residue source** — which value is fed
in as the observed residue — **not the comparison width.** **The sieve does not compare full
32-bit values.** The active predicate reduces both sides modulo **1000, 8 and 125** in every
kernel (§6.2), so a `full_state` residue is reduced exactly as a `draw` residue is. The
match test is unaffected either way (§6.3).

### 1.2 The collision space

For any single published draw, roughly **4.3 million** distinct 32-bit outputs reduce to it:

```
2³² / 1000 = 4,294,967,296 / 1000 ≈ 4.3 × 10⁶ collisions per draw
```

One draw constrains almost nothing. Filtering across many draws is the entire mechanism.

### 1.3 Sequential filtering — and the regime this system does *not* run in

Each additional draw multiplies the constraint. **At exact match** (τ = 1):

| After draw | Calculation | Expected random survivors |
|---|---|---|
| 1 | 2³² / 1000 | ~4,300,000 |
| 2 | 4.3M / 1000 | ~4,300 |
| 3 | 4,300 / 1000 | ~4.3 |
| 4 | 4.3 / 1000 | ~0.004 |
| N | 2³² / 1000ᴺ | → 0 |

The recovered text presented this table, and its 10⁻¹¹⁹¹ figure for N = 400, as *the* operating
characteristic of the sieve. **It is not.** It is the τ = 1 limit — the whitepaper's §6, which
states it for n = 50 as ≈ 10⁻³⁰⁰.

**TFM deliberately does not run at τ = 1.** Whitepaper §7, and
`docs/TFM_PROJECT_FACTS_SKILL.md` §0.3:

> Exact sieves eliminate *all* variance. Survivors = {s\*}. No ranking, no gradients,
> **no learning signal.**

Loose thresholds admit a *manifold* of near-consistent seeds that share structured deviations
ML can rank. **Loose thresholds are a mathematical necessity, not sloppiness.** They are why
thresholds are Optuna-tuned per direction, and why a threshold silently pinned to a constant is
a serious defect rather than a cosmetic one — a class that has now occurred six times
(skill §2.7).

**Read §1.3 and §1.4 as the bound the sieve is measured against, never as the population it
produces.** A reader who takes 10⁻¹¹⁹¹ as the operating false-positive rate will conclude every
survivor is the true seed, which §4.3 corrects.

### 1.4 General probability formula (τ = 1 only)

```
Expected random survivors = 2³² / 1000ᴺ
```

| Draws (N) | Expected random survivors, τ = 1 |
|---|---|
| 4 | 0.004 |
| 10 | 4.3 × 10⁻²¹ |
| 100 | 4.3 × 10⁻²⁹¹ |
| 400 | 4.3 × 10⁻¹¹⁹¹ |

For the loose-threshold regime the system actually runs, the governing statement is the
whitepaper's §5 — `P(B) ≈ P(F)²`, squaring the exponent — not this table.

---

## 2. Forward Sieve

### 2.1 What it does

The forward sieve walks the observed window from the **oldest** position toward the newest,
scoring each candidate seed on how many positions it reproduces.

### 2.2 What the kernel actually computes

The recovered §2.2 gave a set-intersection pseudocode (`survivors = survivors ∩ matching_seeds`
per draw). **The kernel does not do that.** It never materialises per-draw candidate sets. Each
thread owns one seed, walks the whole window once per skip hypothesis, and keeps a **match
rate** (`prng_registry.py:958-1004`):

```c
for (int skip = skip_min; skip <= skip_max; skip++) {
    unsigned long long state = seed & m;
    for (int o = 0; o < offset; o++) state = (a*state + c) & m;   // pre-advance
    for (int s = 0; s < skip;   s++) state = (a*state + c) & m;   // burn before first draw
    int matches = 0;
    for (int i = 0; i < k; i++) {
        state = (a*state + c) & m;
        unsigned int output = (state >> 16) & 0xFFFFFFFF;
        if (/* three-lane test — see §6 */) matches++;
        for (int s = 0; s < skip; s++) state = (a*state + c) & m; // burn between draws
    }
    float rate = ((float)matches) / ((float)k);
    if (rate > best_rate) { best_rate = rate; best_skip_val = skip; }
}
if (best_rate >= threshold) { /* emit seed, best_rate, best_skip_val */ }
```

Four properties that the set-intersection sketch obscures and that matter downstream:

1. **The output is a rate, not a boolean.** `match_rates` is one of only four NPZ columns
   carrying per-seed information (skill §2.3). The sieve is a *scorer* with a threshold, not a
   set filter.
2. **Skip is maximised over, not fixed.** One seed is tested at every skip in
   `[skip_min, skip_max]` and keeps its **best** rate and the skip that achieved it
   (`:992-995`). This is why a survivor is a *pair* (§5.5).
3. **Ties resolve to the lowest skip** — the comparison is `rate > best_rate`, strictly
   greater, so the first skip achieving a rate wins. Transcribed and pinned by the Phase 6
   known-answer reference (`tests/phase6/known_answer_reference.py:80-86`).
4. **The rate is float32 and compared against a float32 threshold.** Doing that arithmetic in
   float64 puts boundary survivors on the wrong side of `>=`
   (`tests/phase6/known_answer_reference.py:72-78`).

### 2.3 Skip burn placement

`skip` states are burned **before the first draw and between every subsequent pair** — not once
up front. This is the single most consequential detail in the kernel loop, because it is
exactly where the CPU reference in the registry disagrees (§12, F-6).

---

## 3. Reverse Sieve

### 3.1 What it does

The reverse sieve scores the same seeds against the **time-reversed** observation window.

### 3.2 Key insight: same PRNG, different direction

> **"Reverse" refers to the ORDER of the target data, NOT to inverting the PRNG.**

This is the most misread fact in the subject area, and it is correct in the recovered text.
Confirmed against live source this session:

- **The host reverses the residues.** `residues[::-1] if reverse else residues`
  (`miner/range_miner_worker.py:888`; legacy `sieve_filter.py:232`, `:395`;
  PWC `sieve_gpu_worker.py:189`).
- **The reverse kernel iterates the generator forward.** `java_lcg_reverse_sieve`
  (`prng_registry.py:3115-3169`) is the same recurrence `state = (a*state + c) & m`, step for
  step, as the forward kernel (`:3143` vs `:982`). **There is no modular inverse and no
  backward recurrence anywhere in the tree.**
- **Direction is a name test.** `is_reverse_family()` is a plain
  `family_name.endswith("_reverse")` (`miner/range_miner_worker.py:116`).

Most PRNGs are not invertible without full state. The reverse sieve is a **time-reversed
target**, not a time-reversed generator.

**Do not "fix" this.** It is listed under looks-like-a-bug-isn't (skill §2.6).

### 3.3 Why reverse matters

| Failure mode | Caught by |
|---|---|
| Early match, late divergence | Reverse sieve |
| Late match, early divergence | Forward sieve |
| Consistency in one direction only | Bidirectional intersection |

### 3.4 The ABI consequence

Because every fixed-skip reverse kernel **hardcodes its generator parameters in the kernel
body** (`prng_registry.py:3125-3126`) instead of taking them as arguments, forward and reverse
constant kernels **do not share an argument layout**. The reverse-constant ABI is
`_constant_prefix + int32(offset)` = **12 args with no family tail**; only the forward constant
branch carries a family-specific tail (`miner/range_miner_worker.py:205-212`). Arity varies by
variant and family: reverse-constant 12, reverse-hybrid 14, `java_lcg` forward-hybrid 15,
`lcg32` forward-hybrid 17.

This contract is documented **only** in that module's own comments (source map G-4). It is
load-bearing for anyone reasoning about the sieve.

### 3.5 One genuine divergence from the whitepaper — stated, not resolved

Whitepaper §4 (`:57-59`) defines the reverse predicate as

```
R(s) = (1/n) Σ 1[ G(s, −i) = d_{n+1−i} ] ≥ τ_r
```

— a generator evaluated at a **negative index**, i.e. a backward step. **The implementation
evaluates `G(s, i)` forward against a reversed residue array.** The draw term matches; the
generator term does not.

The consequence, stated descriptively: for one seed the forward and reverse passes generate the
**identical** output sequence and differ only in what that sequence is compared against.
Whitepaper §4's independence premise (`:61-62`) — on which §5's squaring of the exponent
(`:79`) rests — is stated about a construction in which they would not be identical.

**This chapter does not assert the statistical consequence either way.** That is a mathematics
question and mathematics is the whitepaper's side of the boundary (§0.3). It is recorded here
as an open item (§12, F-7) so that a reader meets it deliberately rather than by inference.

**A related documentation hazard, not to be inherited:** two registry descriptions say "fixed
skip **backward** validation" (`prng_registry.py:3911`, `:3917`) for kernels whose bodies are
forward recurrences, while others correctly say "forward" (`:4099`, `:4118`). The kernel text
governs.

---

## 4. Bidirectional Intersection

### 4.1 The core principle

```
bidirectional_survivors = forward_survivors ∩ reverse_survivors
```

A seed survives bidirectionally iff it clears the forward threshold on the forward-ordered
window **and** the reverse threshold on the reversed window.

### 4.2 The mechanism is a set intersection and nothing more

Stated precisely, because the recovered text left it implicit: the combination is
`forward_set & reverse_set`. There is **no joint gate, no re-verification of the surviving
pair, and no combined-rate threshold.** The two passes are independent scored runs whose seed
sets are intersected. `intersection_count` duplicating `bidirectional_count` is deliberate
(skill §2.6).

### 4.3 What survivors mean — corrected

The recovered §4.3 asserted:

> **"Survivors are NOT false positives."** At 10⁻¹¹⁹¹ probability, they exist because they
> actually match the PRNG behavior … they represent the true seed.

**This is false in the regime TFM operates in, and it contradicts the system's own design.**
It is true only at τ = 1, which §1.3 establishes the system deliberately avoids. At the loose
thresholds the sieve is *required* to use, the survivor population is by construction a
**manifold of near-consistent seeds** (whitepaper §7) — admitting non-true seeds is the
*purpose*, because a population of exactly one has no variance and therefore no learning
signal.

The corrected statement:

> **A survivor is a scored candidate, not a verdict.** The sieve reduces 2³²-scale output space
> to a survival-conditioned population with enough internal variance for ML to rank
> (whitepaper §8). Survivors may be the true seed, one of several true seeds, a partial match
> valid before a reseed event, or a near-consistent neighbour admitted on purpose. **Deciding
> which is Step 3 and Step 5's job, not Step 2's.**

This correction is the reason `forward_matches` and `reverse_matches` matter: they are the only
independent per-seed sieve signal, and they are **absent from the Step-3 merge list** — flagged
as possibly the most consequential finding in the descriptive trace (skill §2.3). The miner
emits both regardless.

---

## 5. Skip/Gap Handling

> **This section is the reason this deliverable exists.** §5.1 and §5.6 are written here for
> the first time; they existed nowhere in the repository. Read §5.1 before forming any opinion
> about whether `skip_min`/`skip_max` should exist.

### 5.1 Why skip exists — the physical model

**The published draw sequence is not an uninterrupted PRNG output stream.**

Per the *California State Lottery Daily & SuperLotto Plus Draw Procedures* (effective
2021-06-09) — **citation `UNAVAILABLE`**: the PDF is **not in the repo** and was **not read
this session**. The statements below are corrected from Team Beta's ruling text and are **not
verified at source.**

1. **One automatic pre-test session runs before an automatic Daily draw** on the selected
   equipment (§V: Pre-Test via `[Start Draw Session]`). **Additional pre-test draws are run
   only when an anomaly requires them.** Pre-test outputs are generated, verified and
   certified — and **never published.**
2. **Draw equipment is selected per session** by an RNG program, auditor-verified (§II).
   Midday and evening are separate sessions with separate equipment selection.
3. **The evening session draws Daily 3, Daily 4, Fantasy 5 and Daily Derby together.** Other
   games' outputs sit between the Daily 3 values an observer can see.

> **Corrected 2026-08-01 — an error Alpha introduced and propagated.** This chapter previously
> stated *"two pre-test draws run before every live draw."* That is **unsupported and appears
> incorrect for automatic Daily draws.** The **"two test draws" language applies to manual
> SuperLotto Plus equipment** — a different draw type — and Alpha misread the document. The
> claim propagated into Chapter 1, the Chapter 2 source map, the project-facts skill and three
> Beta submissions.
>
> **Only the count was wrong.** One unpublished pre-test session still produces unpublished
> outputs; per-session equipment selection and co-drawn evening games are unaffected.
> **Skip remains physically motivated.**

**What these procedures do and do not establish.** They establish equipment selection, an
unpublished pre-test, and co-drawn evening games — that is, outputs which are **consumed and
not published** between the values an observer sees. They do **not** establish that every
omitted output belongs to **one uninterrupted PRNG state stream.** The published values and
the unpublished ones are nowhere shown to form a single continuous advance sequence from a
single generator; midday and evening are independently selected equipment, so combined-container
order carries no PRNG-advance meaning at all (skill §2.10b).

**These are therefore physically motivated *candidate gaps* supporting skip as a detector —
not proven state advances.** The observable sequence contains real structural discontinuities
of unknown and varying size, and skip models them. **It is a physical property of the data
source, not a tuning convenience.** This is the same epistemic standard §5.6 applies: variable
skip is a **detector** looking for windows where coherent structure surfaces, not a
reconstruction of generator state.

#### 5.1.1 Why this paragraph is in this chapter

In one session, **Team Alpha, Team Beta and Claude Code independently recommended deleting
`skip_min`/`skip_max`** — a cornerstone of the design. All three inferred design intent from
current hybrid kernel signatures, which are themselves the defect (§5.4). None of them was
wrong to read the code; the code genuinely does not consume those bounds on the hybrid path.
**They were wrong because the document explaining why skip exists did not exist to be read.**
Michael stopped the removal.

**This section is that document.** A future reader who reaches the hybrid kernel signature and
concludes "these parameters are unused, remove them" has re-derived a conclusion that has
already been made and already been rejected. **The correct action on finding skip bounds unwired
is to wire them in, not to remove them** (`docs/HYBRID_SKIP_BOUND_AUDIT.md` §7).

*Open item: the source PDF is **not in the repo** and could not be read this session. Its
absence is the root cause of both the near-removal of `skip_min`/`skip_max` **and** the
two-pre-test misreading corrected above — a claim that survived a chapter, a sibling chapter, a
source map, a skill and three submissions because no reader could check it against the source.
It remains an open backlog item, and it is the reason the citation above is marked
`UNAVAILABLE`.*

### 5.2 Constant skip mode

A fixed stride between consecutive observed values.

| Skip | Meaning |
|---|---|
| 0 | Every PRNG output is published |
| 1 | Every other output is published |
| N | Every (N+1)ᵗʰ output is published |

**22 of 22 constant kernels declare `int skip_min, int skip_max`** and search the whole range
per seed, keeping the best (`prng_registry.py:963` signature, `:972` loop, `:992-995`
maximisation). The bounds reach the kernel: `_constant_prefix` emits them at
`miner/range_miner_worker.py:171-172`.

### 5.3 Variable skip mode (hybrid)

The stride varies between observations — e.g. `[5, 5, 3, 7, 5, 5, 8, 4]`.

**What the hybrid kernel actually does** (`prng_registry.py:1005-1081`), correcting a common
misreading:

- **No pattern is supplied and none is generated.** `skip_sequences` is an **output**
  (`:1054` records the per-draw stride, `:1075-1077` emits it).
- It runs a **greedy per-draw adaptive search**: from a running estimate `expected_skip`, it
  tries every stride in `[expected_skip − tolerance, expected_skip + tolerance]` (`:1033-1035`)
  and, on a hit, **re-centres the estimate on the stride that hit** (`:1048`).
- `expected_skip` is **hardcoded to 5** (`:1027`). The ancestor file still carries
  `// Initial guess` (`prng_registry_pre_registry.py:696`) — **a guess, not a constant.**
- `strategy_tolerances` is the **half-width of the per-draw matching window** (`:1023`), not a
  generation parameter. `strategy_max_misses` is a consecutive-miss abort (`:1022`, `:1055-1058`).
- **No coherence scoring exists.** The only score is `match_rate`.
- Forward hybrid scans all strategies and keeps the best (`:1061-1067`); **reverse hybrid
  returns on the first strategy clearing the threshold** (`:3239`) and does not maximise at all.

The five documented strategies — Strict Continuous, Lenient Continuous, Aggressive Reseed,
Balanced Hybrid (default), Extreme Tolerance — are strategy *parameterisations* of that one
algorithm, not distinct algorithms.

### 5.4 The defect: hybrid kernels do not execute the requested skip semantics

**0 of 22 hybrid kernels declare `skip_min`/`skip_max`** (`docs/HYBRID_SKIP_BOUND_AUDIT.md`;
signature at `prng_registry.py:1007-1012` — 15 params ending
`float threshold, unsigned long long a, unsigned long long c`).

The sampled bounds survive **eight hops** — argparse, config, coordinator, ledger, manifest,
payload, worker unpack, and the arg-build context — and then **die one call before launch**:

```
miner/range_miner_worker.py   skip_range unpacked from payload
                          →   BuildContext.skip_min / skip_max
                          →   _hybrid_prefix()  (:177-193)  ← never emits them
```

Verified at HEAD this session: `_hybrid_prefix` (`:179-193`) returns 13 elements, none of which
is a skip bound, while `_constant_prefix` (`:162-174`) emits both at `:171-172`. **The asymmetry
is in the argument builder, not in the payload.**

**Consequence, stated plainly: hybrid optimization results are non-certifying.** A trial's skip
range cannot constrain a hybrid pass; the search space advertises a dimension the kernel does
not read. This is skill §2.7 #4 and remains **OPEN** — described here, not repaired.

### 5.5 Survivor identity

**A survivor is a (seed, skip-hypothesis) pair — not a seed.**

```json
{ "seed": 244139, "skip": 5, "skip_mode": "constant", "match_rate": 0.98 }
```

This was §5.4 of the deleted chapter and is one of the three corroborations for §5.6. It is
also why §2.2's per-seed skip maximisation matters: the emitted `best_skips` column *is* the
hypothesis half of the pair.

### 5.6 Design intent — the fingerprint framing

> **What this section is.** It records **Michael's governing design intent as design doctrine**,
> accepted by Team Beta, and gives it a permanent home. **It is not a historically discovered
> repository statement** — no document in the repository ever asserted it. The corroboration
> table below shows earlier artifacts that are **consistent with** the doctrine and were
> produced under it; it does **not** show the doctrine being recovered from them. **Doctrine
> being recorded, not evidence being reported** — the NOT-FOUND row stands, and stays.

**The goal was never to reverse state. It is to extract a fingerprint.**

The published sequence exposes only fragments of PRNG state before other outputs interleave
(§5.1). Variable skip therefore exists to **find the windows where coherent skip structure
surfaces** — the fingerprint glimpse — and to produce survivors with *varied* skip structure so
that tree and neural models have something to rank on.

**Variable skip is a detector, not a fitting procedure.** It is not trying to recover the
generator's state; it is trying to locate the intervals in which the observable stream briefly
behaves coherently, and to characterise them well enough that a model can tell a strong
hypothesis from a weak one.

This framing is corroborated on three of four elements and was **written down nowhere**:

| element | corroboration |
|---|---|
| variable skip yields a *characterised* hypothesis, not a state | the (seed, skip) survivor pair, §5.5 — deleted §5.4 at `d14dcdd` |
| skip structure is meant to become model-rankable features | the Oct-2025 output spec: `skip_pattern` + `pattern_stats {mean_skip, variance, std_dev}` per survivor, `docs/instructions.txt:1236-1247` |
| skip bounds are a *search* concept at the input stage | `--skip-min` / `--skip-max` "…value **in pattern**", `docs/instructions.txt:1182-1183` |
| **the framing itself** | **NOT FOUND anywhere in the repository. This section is its home.** |

### 5.7 `skip_min`/`skip_max` are documented — in two readings, at two stages

Not a contradiction. The same names do two different jobs at two pipeline stages
(`docs/SKIP_SEMANTICS_SEARCH_v1.md`):

| stage | reading | source |
|---|---|---|
| **input** (Step 1 → 2) | *"Minimum/Maximum skip value **in pattern**"* — an element-wise bound on the discovered sequence; documented hybrid default `[0, 16]` | `docs/instructions.txt:1182-1183` (verified this session) |
| **output** (Step 2 → 3) | *"Minimum/Maximum gap that **worked**"* — an ML feature describing what the sieve found; *"Tight skip range = stronger hypothesis"* | `docs/PROPOSAL_ML_Architecture_Remediation_v2_0.md:150-158` (verified this session) |

**Two registries currently disagree** about which reading is authoritative:
`config_manifests/feature_registry.json` says "found during" (output);
`config_manifests/parameter_registry.json:160,166` says "for sieve search" (input). One is
wrong; correcting it belongs to whichever change settles the semantics.

**These two readings have different costs.** The **output** reading needs no kernel change at
all — it is blocked only by the host discarding `skip_sequences` (§8.6). The **input** reading is
§5.4 and needs the hybrid ABI wired. Conflating them is what makes the sampler-comparison
sequencing error possible (§11.4).

---

## 6. Three-Lane CRT Architecture

> **Restored from the deleted §6.** This was the **only prose explanation in the project's
> history** of a test that is live in every kernel and had been undocumented since `248e48c`.
> One of its central claims does not survive verification; §6.4 corrects it.

### 6.1 The construction

```
1000 = 8 × 125        gcd(8, 125) = 1     ← coprime
```

By the Chinese Remainder Theorem, agreement mod 1000 decomposes into agreement mod 8 and
agreement mod 125.

### 6.2 The test, as it exists in the kernel

Live and verbatim at `prng_registry.py:984-986` (`java_lcg_flexible_sieve`, forward constant),
`:1042-1044` (`java_lcg_hybrid_multi_strategy_sieve`, forward hybrid), `:3146-3148`
(`java_lcg_reverse_sieve`, reverse constant) — all three **re-verified unchanged at closure**
(`81ef3f1`, 2026-08-02):

```c
if (((output % 1000) == (unsigned int)(residues[i] % 1000)) &&
    ((output %    8) == (unsigned int)(residues[i] %    8)) &&
    ((output %  125) == (unsigned int)(residues[i] %  125))) matches++;
```

> The block above is byte-exact for `:984-986` and `:3146-3148`. The hybrid form at `:1042-1044`
> is the same conjunction with the loop index named `draw_idx` instead of `i`, and it opens a
> brace rather than ending in `matches++;` because the hybrid body also resets
> `consecutive_misses`. Same test, different surrounding block.

#### 6.2.1 The count — settled

**The chapter previously claimed "39 occurrences … one per kernel". That number is withdrawn.
It is not reproducible by any method, and the "one per kernel" gloss was also wrong.**

**The number is 43**, and it is 43 of the registry's **44** kernels — not one per kernel.

**Method (this is the method the number must be reproducible by).** A lane test is counted where
a line compares `output % 1000` against a `residues[…] % 1000` term **and the following two
lines carry the mod-8 and mod-125 conjuncts of the same comparison.** Structural, not textual:
it counts the conjunction the section is about, and is indifferent to casts, spacing, index
naming and whether the body ends in `matches++;` or opens a brace. Reproduce it with:

```python
lines = open('prng_registry.py').read().split('\n')
hits = [i+1 for i, l in enumerate(lines)
        if '% 1000' in l and '% 8' in '\n'.join(lines[i:i+3])
                         and '% 125' in '\n'.join(lines[i:i+3])]
len(hits)   # 43
```

**Machine-readable form, so the gate can check it rather than a reader having to.** The block
below is what `tests/test_chapter2_content_gate.py` parses and re-derives against live
`prng_registry.py` on every run. If the registry gains or loses a kernel, **this block goes stale
and the gate goes red** — which is the point. Do not hand-edit it to make a gate pass; re-run the
method and fix the prose with it.

<!-- BEGIN LANE TEST COUNT — machine-checked against prng_registry.py -->
```
    source_file:            prng_registry.py
    method:                 structural-3line-conjunction
    lane_test_count:        43
    total_kernels:          44
    single_lane_exception:  mt19937_hybrid_multi_strategy_sieve
    cast_variant_count:     31
    index_split:            residues[i]=30, residues[draw_idx]=13
```
<!-- END LANE TEST COUNT -->

> **Read `cast_variant_count` and `index_split` as corroboration, not as competing answers.**
> 30 + 13 = 43 is the same set counted another way; 31 is a formatting subset of it. They are
> recorded because each was, at some point, mistaken for the answer.

**Why three counts existed, and what each one actually measures.** All three are reproducible;
they simply do not measure the same thing.

| Count | What it measures | Verdict |
|---|---|---|
| **43** | complete three-lane conjunctions | **the right number** — it counts the test §6.2 prints |
| 31 | the subset written with the `(unsigned int)` cast on the residue side | a **formatting** variant — 12 of the 43 omit the cast; C integer promotion makes it semantically inert |
| 30 + 13 | the same 43, partitioned by loop-index name (`residues[i]` in constant kernels, `residues[draw_idx]` in hybrid kernels) | **the same 43**, split by an incidental naming difference |
| 39 | — | **matches no method.** Withdrawn |

31 was the strict-pattern count that the corrections session could not reconcile; it is a real
count of a real thing, but the thing is a coding-style split with no bearing on the sieve. 30+13
is 43 arrived at by a different route, which is corroboration rather than a competing answer.

**The 44th kernel — the exception that "one per kernel" hid.** `mt19937_hybrid_multi_strategy_sieve`
(`prng_registry.py:773`) is the only kernel in the registry that does **not** run the three-lane
test. It reduces once and compares once:

```c
unsigned int draw = output % 1000;          // :820
if (draw == residues[draw_idx]) {           // :821
```

This is worth stating precisely because §6.3 proves the three-lane test is **exactly equivalent**
to this single comparison. The mt19937 kernel is therefore not a defect and not a missing lane —
it is the same predicate written without the two redundant conjuncts, and it is evidence that the
redundancy in the other 43 is a convention rather than a requirement. It is also **outside TFM's
sieve path**: TFM sieves `java_lcg` only (`CLAUDE.md` §7), and mt19937 is one of the five
families the miner raises `NotImplementedError` for.

> **Why the chapter now names the method and not just the number.** The withdrawn 39 survived
> because it was carried as a bare figure with no stated way to re-derive it, so no later reader
> could tell whether it was stale, mis-transcribed, or measuring something else. **A number in a
> chapter must be reproducible by the method the chapter names** — the snippet above is that
> method, and re-running it is the whole audit.

| Lane | Role as built |
|---|---|
| mod 1000 | full published value |
| mod 8 | low three bits |
| mod 125 | the coprime complement |

### 6.3 Lane disagreement = prune — true, and vacuous

The recovered §6.3 called this "algebraic necessity, not heuristic". **The algebra is right and
the emphasis is misleading.** Because 1000 = 8 × 125 with gcd(8, 125) = 1, CRT gives:

```
x ≡ y (mod 8)  ∧  x ≡ y (mod 125)   ⟺   x ≡ y (mod 1000)
```

So the mod-8 and mod-125 conjuncts are **implied by** the mod-1000 conjunct. **No lane can
disagree once lane 1 agrees.** The three-lane test is exactly equivalent to
`(output % 1000) == (residues[i] % 1000)` alone.

Verified two ways this session: by the CRT argument above, which holds for all integers
regardless of residue magnitude (so it is unaffected by the `full_state` residue source of
§1.1 — which changes the input value, never the modulus); and by
exhaustive check over `x ∈ [0, 4000) × d ∈ [0, 1000)` — **0 cases in which the three-lane test
and the mod-1000 test differ.**

### 6.4 Triple-validation power — **corrected**

The recovered §6.4 claimed:

> Single mod 1000 match: ~0.1% false positive rate per draw.
> Triple validation: ~0.00001% false positive rate per draw.
> **Effectively requires full 32-bit state match.**

**This is incorrect, by roughly four orders of magnitude.** A redundant conjunct adds no
filtering power. The per-draw false-positive rate under the three-lane test is exactly the
mod-1000 rate:

```
P(random output matches one draw) = 1/1000 = 0.1%
```

and the test emphatically does **not** require a full 32-bit state match — ~4.3 million
distinct outputs still pass each draw (§1.2). The sieve's actual filtering power comes from
**sequential accumulation across k draws** (§1.3) and from **bidirectional intersection**
(§4), not from the lane decomposition.

**The redundancy is independently confirmed in-repo**, which corrects the source map's
expectation that §6 was the only surviving account: the Phase 6 known-answer reference states
it explicitly at `tests/phase6/known_answer_reference.py:66-70` —

> *"Because 1000 = 8 x 125 with gcd(8,125) = 1, the mod-1000 test already implies the other
> two; the redundancy is transcribed verbatim anyway rather than 'simplified', because the job
> of a reference is to mirror the specification, not to improve it."*

That reference is the one used by the Miner Known-Answer Transfer Gate, so the gate's 8/8
exact-set equality result is consistent with the lanes being redundant — the gate would produce
the same populations either way.

### 6.5 Why this is documented, not removed

**Do not remove the lanes on the strength of §6.4.** Three reasons, in order of weight:

1. **It is out of scope here.** This is a documentation pass; the brief prohibits repairs.
2. **The transcription is deliberate on the reference side** and would have to change in
   lockstep on both sides, under a gate, to preserve byte-identity guarantees.
3. **The redundancy may encode an intended architecture that was never built.** A genuine
   three-lane CRT sieve would compute candidate sets *per lane* and CRT-recombine them — a
   different and potentially much faster construction. What the kernel contains is the
   *consistency identity* of that design, evaluated redundantly, not the design itself.
   Whether that architecture was intended and abandoned is **not determinable from the
   surfaces available** and is recorded as an open question (§12, F-3).

**What §6 correctly establishes and what it does not:** the lane decomposition is a valid CRT
identity (§6.1–6.3, correct as recovered). The claim of *added filtering power* (§6.4) is not
supported. A reader should carry the first and discard the second.

### 6.6 A separate and legitimate use of lanes

Lane agreement **is** load-bearing elsewhere, where it is measured rather than ANDed:
`survivor_scorer.py:421-424` and `:612-614` compute `lane_agreement_8`, `lane_agreement_125`
and their mean `lane_consistency` as **graded ML features** over predictions vs actuals. There
the lanes are *not* redundant — partial agreement mod 8 with disagreement mod 125 is a real,
informative state that a boolean mod-1000 test cannot express. **The lane concept is sound; its
use as a conjunctive gate in the kernel is what is redundant.**

---

## 7. `offset` — one name, incompatible meanings

`docs/CHAPTER_1_AUDIT_v1.md` C-2 found `offset` carries **three incompatible definitions**,
could not settle the collision from Chapter 1's surfaces, and deferred it here. This chapter
picks the implemented meaning and states it.

### 7.1 The definitions in circulation

| source | definition |
|---|---|
| old Chapter §3.1 | "time offset from current draw" |
| host code | **head-relative array index** into the session-filtered draw list |
| `docs/instructions.txt:1181` | "temporal alignment (**PRNG steps** to skip before sequence)" |
| `config_manifests/parameter_registry.json:38-43` | advance seeds by **`offset*(skip+1)`** before testing |

### 7.2 What the code did — both of them, from one value

**AS AT THE CHAPTER-2 AUDIT ANCHOR** the certifying path used the **same scalar for two
different jobs**, verified at that HEAD. This is the historical finding and it is recorded
here unchanged:

- **Host, as a data index:** `start = max(0, min(int(offset), n - window_size)); window =
  data[start:start + window_size]` (`miner/range_miner_worker.py`, `load_residue_window`),
  read from `payload.get("offset", 0)` in `ResidueResolver.resolve`.
- **Device, as a generator pre-advance:** `ScalarArg(ctx.offset, "int32")` (`_offset_tail`),
  read from the same `payload.get("offset", 0)` in the sub-stripe path, consumed by the
  kernel as `for (o = 0; o < offset; o++) state = step(state)`
  (`prng_registry.py:974-976`).

**So `offset` simultaneously shifted which records were observed and how far the generator
was pre-advanced.** That is the defect this chapter recorded, and the verdict at the audit
anchor stands as written: **F-4 CONFIRMED, NOT REPAIRED.**

#### 7.2.1 What the code does now — the approved separation

**Status: repair implemented by Window-Anchor Brief I; acceptance pending.** The disposition
above is HISTORICAL and is not rewritten; this subsection records the subsequent authorized
change, and it does not claim acceptance.

Governing artifacts: `docs/PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md`
(design gate CLOSED) · `docs/TB_RULING_WINDOW_ANCHOR_V1_1_DESIGN_GATE_CLOSED.md` ·
`docs/S172_WINDOW_ANCHOR_BRIEF_I.md`.

The one scalar is split into two, and neither is reconstructed from the other:

| name | means ONLY | lives |
|---|---|---|
| `window_anchor` | which observed records form the residue window — `filtered_data[anchor : anchor+window_size]` | **host**, residue construction |
| `generator_phase` | how many generator-state advances precede the first comparison | **device**, the existing kernel `offset` argument where one exists |

- **Host:** `load_residue_window(path, window_size, sessions, window_anchor)` validates the
  anchor against a derived domain and **raises** — the silent clamp is gone. `derived_max` is
  computed on the POST-session-filter count, so a single-session trial cannot address past the
  end of its own sequence.
- **Device:** `_generator_phase_tail` emits `ScalarArg(ctx.generator_phase, "int32")` in the
  **unchanged position and dtype** — the kernel ABI is frozen byte-for-byte for all 44
  registry entries. The anchor reaches **no** kernel argument on any variant.
- The retired `offset` key is **hard-rejected**, never mapped to either successor: which of
  the two roles a historical value meant is not recoverable from the value.
- `generator_phase` is **pinned to 0 in v1** at both the coordinator's public assign-payload
  validation and the worker execution seam, and is carried explicitly so every artifact
  records the phase that ran rather than leaving it to be inferred.

### 7.3 The consequence, described

That coupling is **self-consistent only at `skip = 0`**. When each observed draw consumes one
PRNG output, shifting the window by one record and pre-advancing by one step keep the two
aligned. At `skip = N`, each observed draw consumes `N+1` outputs, so a one-record window shift
should correspond to a `(skip+1)`-step pre-advance — **which is exactly the formula
`parameter_registry.json` specifies and the kernel does not implement.** The kernel advances
`offset` steps flat.

**What this settles, and what it does not.** It settles Chapter 1 audit **C-2 as an observed
inconsistency**: `parameter_registry.json` is not merely an outlier description — it describes
the alignment the other two definitions would need in order to be jointly coherent at non-zero
skip. **It does not settle the repair, and must not be read as specifying one.**

**`offset*(skip+1)` is not a general fix.** It is well defined only for **constant** skip.
Under **variable** skip the per-record consumption varies by construction, so **no single
`(skip+1)` multiplier exists** — the correct pre-advance for a window shift depends on the
particular stride sequence, which is an *output* of the search, not an input to it (§5.3).

**F-4 therefore belongs inside the future hybrid input-semantics design, not a standalone
arithmetic patch.** Applying a flat `offset*(skip+1)` in isolation would harden constant-skip
semantics into a path whose hybrid half still has no defined input-bound meaning (§5.4, §5.7).
The two must be decided together.

> **SUBSEQUENT DISPOSITION — repair implemented by Window-Anchor Brief I; acceptance
> pending.** That design is the window-anchor / generator-phase separation, and it resolved
> F-4 by SPLITTING the scalar rather than by multiplying it: no `offset*(skip+1)` was applied
> anywhere. The paragraph above is the analysis as written at the audit anchor and is not
> revised. See §7.2.1.

**Additionally, forward hybrid kernels take no `offset` at all** — the `java_lcg` forward hybrid
signature ends `float threshold, unsigned long long a, unsigned long long c` with no offset
parameter (`prng_registry.py:1007-1012`; builder comment `miner/range_miner_worker.py:219`).
On that path the window shifts and the generator does not pre-advance whatsoever. That is
skill §2.7 #5, **OPEN**.

**Described, not repaired**, per the brief.

---

## 8. The engine today: RANGE-MINER

> The deleted §7–§13 documented `sieve_filter.py` / `GPUSieve` as *the* engine: component flow,
> ROCm prelude, class methods, `run_sieve`, `run_hybrid_sieve`, CLI and integration points.
> **That engine is superseded and those sections are not restored.** This section states the
> current architecture and **cites** the authoritative documents rather than duplicating them.

### 8.1 Why the engine changed

PWC suffered silent hard resets and `GCVM_L2_PROTECTION_FAULT` on the RX 6600 XT rigs at
full-fleet saturation, traced to launch-storm behaviour. After weeks of failed debugging the
project pivoted to **RANGE-MINER: persistent per-GPU daemons** (skill §0.7).

The replacement is an **interface** contract — the frozen 22-array NPZ survivor bundle — not a
"match PWC's values" contract. **The remaining steps must not be able to tell which engine
produced their input.**

### 8.2 Structure

| Module | Role |
|---|---|
| `miner/range_miner_coordinator.py` | stripe ledger, state machine, macro partitioning, L8 reconciliation, §6.8 phase table, threshold payload + provenance enforcement |
| `miner/range_miner_worker.py` | READY handshake, sub-stripe loop, per-family kernel ABI builders, residue-window authority, inline-vs-spool transport |
| `miner/range_miner_npz_writer.py` | Phase-5 assembly: spool validation, canonical replay, trial assembly |
| `miner/assembly_backends.py` | frozen two-backend interface (`serial_reference` \| `process_sharded`) |
| `miner/step1_ingress.py` | Step-1 accumulator ingress + certified-path resolution |
| `miner/range_miner_protocol.py` | 8 message types, length-prefixed JSON framing |

### 8.3 Seed-domain partitioning

The **coordinator** partitions the domain into contiguous macro-stripes with **no gap and no
overlap**; a macro-stripe may exceed one GPU's capacity. The **worker** then partitions its one
assigned macro-stripe into GPU-safe sub-stripes at runtime, with the cap branching on backend
(`rocm` → AMD caps, `cuda` → NVIDIA caps). The coordinator sizes `expected_substripes` using the
**same** cap the worker will partition with.

Completion is proved, not assumed: a stripe is complete only when sub-stripes done == expected
== distinct sub-indices, seed counts sum, survivor counts sum, **and** the sub-stripe ranges
tile the parent exactly. Gap or overlap → not complete.

Anchors: `miner/range_miner_coordinator.py` `partition_macro_stripes`,
`advertised_effective_cap`, `expected_substripes_for`, `_coverage_exact`,
`evaluate_stripe_completion`; `miner/range_miner_worker.py:472` `select_seed_cap`, `:493`
`partition_stripe`. See `docs/CHAPTER_2_SOURCE_MAP_v1.md` Source 6 for the line-level table.

### 8.4 Residue-window authority

**One derivation function, shared by parent and worker, session-filtered:**
`load_residue_window()` (`miner/range_miner_worker.py:602-650`), reached from the coordinator
side via `_miner_residues_for_config()`
(`window_optimizer_integration_final.py:273`, consumed at `:1218`).

Its docstring records the D6 defect it closes (`:611-626`): the parent used to derive residues
*without* the session filter while the worker applied it, so every single-session trial died on
the `residue_sha256` check. It says explicitly: **"Do NOT reintroduce a second session-filter
implementation on either side."**

Identity is by **content, not pathname**: `sha256_residues()` (`:555-559`) hashes compact JSON
of the int list.

Dataset-side contract — index = position in the PRNG output stream, `offset` slices from the
oldest end, and `load_residue_window` named as "the correctness-critical consumer":
`docs/DAILY3_CONSUMER_CONTRACT_v1.md` §4.1–§4.4, §7. **Cite, do not re-derive.**

### 8.5 Thresholds

**Cite, do not re-derive:** `docs/THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md` (384 lines) and
`docs/S172_THRESHOLD_PROPAGATION_REPAIR_REPORT.md` (419 lines, commit `8a55a68`).

Post-`8a55a68` live state: one canonical resolver `resolve_directional_threshold()`
(`window_optimizer_integration_final.py:214`) used by both routes. The **parent** resolves
direction per stripe via the §6.8 phase table and stamps `min_match_threshold` into the
assignment payload; the **worker does not choose a threshold and does not know about
forward/reverse**. Contradictory `min_match_threshold` / `phase2_threshold` pairs **fail
closed**.

Provenance is a **triple — requested / payload / effective** — recorded per sub-stripe and per
stripe, with parent-side fail-closed enforcement: effective MUST be present, all sub-stripes of
a stripe MUST agree, and disagreement or absence is a **certification failure**, not a warning.

### 8.6 Assembly, the NPZ contract, and one live feature loss

Assembly runs behind a **frozen two-backend interface that fails closed** — no silent default.
`serial_reference` remains the production default; `process_sharded` is implemented,
**available and UNPROMOTED**, and parallelises *only* spool-local validation — the parent alone
owns merge, dedup and intersection.

The carrier is the **frozen 22-array NPZ contract** (`utils/canonical_arrays.py`
`CANONICAL_ARRAY_CONTRACT`; record side `utils/canonical_records.py`). **Only four columns carry
per-seed information:** `seeds`, `forward_matches`, `reverse_matches`, `score`.

> **Naming trap.** `EXPECTED_NPZ_KEYS` — named in `CLAUDE.md` §6 Phase 5 — **exists in the tree
> only as a forbidden-token string** in `tests/test_s172_phase4_coordinator.py`. There is no
> symbol by that name. The contract wall lives under the `utils/canonical_arrays.py` /
> `utils/canonical_records.py` names. Do not write code or docs around a symbol that does not
> exist.

**Where skip structure is lost.** `extract_survivor_records()`
(`window_optimizer_integration_final.py:125-166`) reduces each survivor to
`{'seed', 'match_rate'}` and **discards `skip_sequences`**. That single discard is what kills the
three dead skip features `skip_mean`, `skip_std`, `skip_entropy` — whose producer *exists on the
GPU* (§5.3) and whose Oct-2025 ancestor spec is `pattern_stats` (§5.6). **Reviving them requires
no kernel change**, only that the host stop discarding the sequence. This is the **output**
reading of §5.7, and it is the cheap half.

### 8.7 The finalizer is frozen

`utils/run_finalizer.py` — `_l2_sort_key`, `_select_l2_winners`, L3 merge, global seed-ascending
sort. **Import it; never fork it** (skill §4). Same-trial/same-mode collision raises
`AccumulatorConsistencyError`. Generations chain; input identity is a lineage invariant.

---

## 9. Inter-Chunk GPU Cleanup (legacy engine, added 2026-01-26)

> Retained from the recovered §14, with corrected anchors and re-scoped applicability. This is a
> real historical fault-mode record on a path that still runs, and **deleting a documented fix
> invites its removal from the code.**

### 9.1 Problem

Step-1 forward sieves process seeds in chunks (~19K seeds/chunk). At 500K seeds ≈ 26 chunks,
VRAM fragmentation accumulated without cleanup and produced intermittent GPU hangs:

```
Error: HW Exception by GPU node-11... reason: GPU Hang
```

| Step | Chunks/invocation | Cleanup frequency | Result |
|---|---|---|---|
| Step 1 | ~26 | once at exit | **GPU hangs** |
| Step 2.5 / 3 | 1 | every invocation | stable |

### 9.2 Fix

Inter-chunk cleanup in **both** forward-sieve loops of `sieve_filter.py`. The guard text is
unchanged from the original record; **the line numbers moved** — the recovered §14 said
"lines 230, 385"; the calls are now at **`sieve_filter.py:326-327` and `:481-482`**, with the
end-of-run call at `:788` and the helper defined at `:90` (all verified this session):

```python
if chunk_start + chunk_size < seed_end:
    _best_effort_gpu_cleanup()
```

`gc.collect()` was added to `_best_effort_gpu_cleanup()`.

### 9.3 Validation as recorded

20/20 benchmark trials with 0 GPU hangs; all 26 GPUs healthy post-run; <5% overhead.

### 9.4 Scope — this is not a current mitigation for the certifying engine

**§9 describes the legacy chunked engine.** RANGE-MINER's persistent per-GPU daemons exist
*precisely because* launch-storm behaviour caused `GCVM_L2_PROTECTION_FAULT` on the rigs
(§8.1), and Phase 6.0 recorded **no GPU reset and no `GCVM_L2` fault** on either platform. Do
not read §9 as describing how the current engine avoids GPU hangs — it describes how the
superseded one was patched to.

### 9.5 The ROCm prelude — dead on every live rig

The recovered §8 documented the ROCm environment prelude as a critical, load-bearing
mitigation. It is still present (`sieve_filter.py:23-35`) and its hostname list has since been
extended to three entries:

```python
if HOST in ("rig-6600", "rig-6600b", "rig-6600c"):
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    os.environ.setdefault("HSA_ENABLE_SDMA", "0")
```

**The guard never fires.** Queried from VM 101 this session, the live worker hostnames are
`rrig6600`, `rrig6600b`, `rrig6600c` — CT100 is created with the rig's canonical hostname so
that `socket.gethostname()` *is* the coordinator identity. **None of the three matches the
tuple** (`rrig6600` ≠ `rig-6600`).

**Harmless today**, and this is why it went unnoticed: the current ROCm stack on all three rigs
needs **no HSA/GFX overrides** (skill §6). But the branch is dead, and
`docs/DOCUMENTATION_AUDIT_20260131.md:93-99` — which rated Chapter 2 a "LOW / single line" fix,
namely *add `rig-6600c`* — proposed a change in the **same wrong naming convention**, so
applying it would not have made the guard fire either.

Recorded as a finding (§12, F-5). **Not repaired here**, and **not a Phase-7 blocker.**

**Disposition for any future repair — recorded so the obvious fix is not applied by reflex:**

1. **Do NOT rename the hosts to match the tuple.** Making the guard fire would **activate
   obsolete ROCm environment overrides** (`HSA_OVERRIDE_GFX_VERSION`, `HSA_ENABLE_SDMA`) that
   the current rigs **reportedly do not need** (skill §6: all three CTs run cupy 13.5.1 on
   gfx1032 with **no HSA/GFX overrides**). The "one-line fix" is a behaviour change to a
   working fleet, disguised as a typo correction.
2. **First decide whether the prelude is still supported at all.** If it is **obsolete**,
   remove it **with its historical explanation preserved** — the GCVM_L2 fault chase is why it
   exists and that reasoning must not be deleted with the code (skill §0.4 standing rule).
3. **If it is retained, key it from an explicit platform/profile property** — a declared
   rig-profile or platform capability — **and test it.** **Never from another handwritten
   hostname tuple.** A hostname tuple is what produced this dead branch, and the
   `DOCUMENTATION_AUDIT_20260131.md` proposal would have reproduced the same failure mode in
   the same wrong convention.

Note also that the audit entry rated Chapter 2's condition against the 743-line version,
before anyone noticed §1–13 were gone.

---

## 10. Non-certifying diagnostic paths

> Replaces the recovered §15, which was present **twice verbatim** in the fragment
> (`:38-83` and `:85-128`) — the same duplication flagged for the sibling chapter in
> `docs/CHAPTER_1_AUDIT_v1.md`.

**Team Beta retired PWC/ZMQ from certifying authority on 2026-07-31** (skill §0.7, §3). It
remains a flag-selectable, non-certifying diagnostic backend reached via
`--use-persistent-workers`. Its hybrid path is **additionally quarantined** as
`PWC_HYBRID_THRESHOLD_CONTRACT_UNCERTIFIED`, because it filters at a hardcoded `0.50` — the
defect is made loud rather than repaired, since the path cannot certify anything either way
(skill §2.7 #3).

**Do not cite PWC results as comparators.** The S146 numbers in the deleted §15 (313
bidirectional survivors, 666 in the accumulator) are historical operating records of a path
that no longer carries authority, and are not restated here as current.

`sieve_filter.py`, `sieve_filter_INTEGRATED.py`, `reverse_sieve_filter.py` and its backups are
likewise not the certifying path. `docs/CHAPTER_2_SOURCE_MAP_v1.md` §7 inventories all of them
by name. **None of them is to be deleted** — a standing ruling of 2026-07-31 leaves the known
duplicates alone deliberately; the inventory exists so a reader does not mistake one for
current.

---

## 11. What is certified, and what is not

### 11.1 Bounded Phase 6 — CERTIFIED and CLOSED (`d98298c`, TB ruling 2026-08-02)

- **Wall A** — the complete consumer chain: frozen 22-array bundle → validation → Step-2 load
  without fallback → dict conversion → Step-3 chunks → real GPU scorer, with **value-by-value**
  metadata comparison, closing the "keys present but values defaulted" class.
- **Wall B** — repetition, assembly-backend equivalence, current CUDA/ROCm equivalence, and
  **node-assignment independence across two different ROCm rig pairs**. All five arms reproduced
  `artifact_sha256 0e0092fe…c4b0`.
- **Miner Known-Answer Transfer Gate** — all four active TFM variants through their real
  `SieveExecutor.execute` ABI paths; **8/8 populations exact-set equal**, zero missing, extra or
  mismatched; F5–F7 prove reference independence by rejecting three wrong semantics.

### 11.2 The scope limit — stated explicitly

**Wall A and Wall B used constant-skip generations.** **Hybrid worker semantics are covered by
the transfer gate, not by a four-phase Wall-A consumer run.** The scratch generations are **not**
release-grade; future publication still uses `--release-grade`.

### 11.3 What is therefore *not* proven about Step 2

- **Hybrid/variable-skip certification is blocked** — §5.4's skip bounds do not reach the
  kernel. Optuna constant-skip exploration **may resume**; hybrid exploration is
  **non-certifying only**.
- **Combined-session sequential sieving is non-certifying and prohibited by default.** Midday
  and evening use independently selected equipment (§5.1), so there is **no evidentiary basis
  for advancing one PRNG state through interleaved records**. Ordering is normative *within a
  session stream*; combined-container order carries **no PRNG-advance meaning**. Production
  re-optimization is **per-session**.
- **Path coverage is partial.** `docs/S172_SIEVE_PATH_VERIFICATION_SCOPE.md` frames it as four
  sieve paths × 6 covered families = 24 variants, and warns these are "two DIFFERENT claims —
  do not conflate."

### 11.4 A sequencing correction Team Beta issued against Alpha

The certifying four-phase TPE-vs-random sampler comparison **cannot** be scheduled merely
*"after the skip-output work."* The approved skip-output work (§8.6) restores `skip_mean` /
`skip_std` / `skip_entropy`; **it does not connect `skip_min`/`skip_max` to the hybrid
kernels** — that is the separate, unresolved input-bound interpretation of §5.7. The comparison
must wait until **either** hybrid search-input bounds have defined effective semantics, **or**
the comparison uses an explicitly phase-aware search space that does not pretend dead hybrid
dimensions are active. **Completing skip-output alone does not remove the dead-dimension
caveat.**

---

## 12. Audit findings from this pass

Findings are numbered F-1…F-8. **None is repaired here** — this is a documentation deliverable
and repairs are out of scope by the brief.

| # | Finding | Anchor | Disposition |
|---|---|---|---|
| **F-1** | **§6.4's triple-validation claim is wrong by ~4 orders of magnitude.** The three-lane test is CRT-redundant with mod 1000; per-draw FP is 1/1000, not 1e-7, and it does not require a full 32-bit state match | `prng_registry.py:984-986`, `:1042-1044`, `:3146-3148`; corroborated `tests/phase6/known_answer_reference.py:66-70` | **corrected in §6.4.** Lanes **not** to be removed (§6.5) |
| **F-2** | **§4.3's "survivors are NOT false positives" is false** in the loose-threshold regime the system requires, and contradicts whitepaper §7 | whitepaper `:116-131`; skill §0.3 | **corrected in §4.3** |
| **F-3** | The lane redundancy may be the residue of an intended lane-parallel CRT architecture that was never built; not determinable from available surfaces | — | **open question**, §6.5 |
| **F-4** | **`offset` drives a host data-slice and a device pre-advance from one scalar**; coherent only at `skip = 0`. `parameter_registry.json`'s `offset*(skip+1)` is the alignment the kernel does not implement | `miner/range_miner_worker.py:648-649`, `:694`, `:874`, `:196-197`; `prng_registry.py:974-976` | **CONFIRMED** at the audit anchor. Settles Chapter 1 audit C-2 as an **observed inconsistency — NOT the repair.** No single `offset*(skip+1)` multiplier exists under variable skip. **Belonged in the future hybrid input-semantics design, not a standalone arithmetic patch** — that design is the window-anchor / generator-phase separation. **SUBSEQUENT: repair implemented by Window-Anchor Brief I; acceptance pending** (§7.2.1). The CONFIRMED verdict above is historical and is not rewritten. Described, §7.3 |
| **F-5** | **The ROCm prelude hostname guard is dead on every live rig** — tuple says `rig-6600*`, live hostnames are `rrig6600*`. `DOCUMENTATION_AUDIT_20260131.md:93-99`'s proposed one-line fix used the same wrong convention | `sieve_filter.py:23-35`; live `hostname` from all three CT100s this session | **CONFIRMED dead legacy branch; NOT a Phase-7 blocker.** Harmless today (no overrides needed). **Do not rename the hosts** — that activates obsolete overrides. A later repair must first decide whether the prelude is supported, and if retained **key it from an explicit platform/profile property, never another hostname tuple.** §9.5 |
| **F-6** | §1.1's "32-bit internal state" is wrong for `java_lcg` — the state is 48-bit; 32 bits is the extracted output | `prng_registry.py:969`, `:983` | **corrected in §1.1** |
| **F-7** | Whitepaper §4's `G(s,−i)` assumes a backward step; the implementation is forward-against-reversed-residues, so forward and reverse generate identical sequences for one seed | whitepaper `:57-62`, `:79`; `prng_registry.py:3143`; `miner/range_miner_worker.py:888` | **named, not resolved**, §3.5 — Beta's side of the boundary |
| **F-8** | Residues resolve as `entry.get("full_state", entry["draw"])`; no live record carries `full_state` (0 of 18068) | `miner/range_miner_worker.py:650`; `daily3.json` inspected this session; `create_synthetic_full_state.py:3-4`, `:27`; `tests/phase6/known_answer_gate.py:211-216`; `tests/phase6/known_answer_reference.py:409-411` | **Inert in the live dataset, purpose known.** It is the **synthetic known-answer / multi-modulo validation hook** (Wall C ruling) — **not** a "forward-compatibility seam." It changes the **residue source, not the comparison width**: the predicate still reduces mod 1000/8/125. Recorded §1.1 |

### 12.1 Open items this chapter inherits but does not own

- **The CA draw-procedures PDF is not in the repo** (§5.1) — now demonstrated to be load-bearing:
  it is the root cause of **both** the two-pre-test misreading corrected in this pass **and** the
  near-removal
  of `skip_min`/`skip_max`.
- **No `TB_RULING_*` document exists for the 2026-07-30/31 session-stream rulings** (source map
  G-2); §11.3 cites a skill summary for a binding constraint. Every other adjudicated area has a
  ruling file.
- **`docs/STEP2_BIDIRECTIONAL_SIEVE_DESCRIPTIVE_TRACE.md` is untracked** and would be lost by a
  clean checkout. It is a legitimate prior-art input for the legacy/PWC half and should be
  tracked before being cited stably. **Its §8.5/O7 — "Optuna thresholds never reach a kernel" —
  is superseded by `8a55a68`; do not inherit that claim.**
- **`java_lcg_cpu` non-zero-skip mismatch** — the registry CPU reference applies `skip` once
  before generating; the kernel applies it between every draw (§2.3). **They agree only at
  `skip = 0`.** TB: separate bounded audit before Phase 7, **no fix authorized**. A known-answer
  reference built on it would validate the wrong semantics.
- **The 44-entry registry vs 4 compiled variants** — registry size is load-bearing for the
  uint8 `prng_type` encoding even though only 4 entries are ever compiled in production.
- **`forward_matches`/`reverse_matches` are absent from the Step-3 merge list** (§4.3) — needs a
  governed schema decision.

---

## 13. Verification declaration (VIR-1…6)

**execution proof (VIR-1).** Every `file:line` in this chapter was obtained on VM 101 at
`eed3904` this session, by `Read` or `/bin/grep -n` against the working tree, except where
explicitly attributed to a cited audit. **Anchors were re-verified rather than carried over:**
`docs/CHAPTER_2_SOURCE_MAP_v1.md` surveyed at `73dbacf`, and several `miner/` anchors have since
moved (e.g. `load_residue_window` 538→602, `residues[::-1]` 813→888,
`resolve_directional_threshold` 210→214, `_miner_residues_for_config` 290→273). All anchors
cited here are HEAD-current. The CRT redundancy (§6.3) was additionally established by
executed arithmetic, not only by argument.

**clean control (VIR-2).** Sections verified **correct and unchanged** from the recovered text,
with no correction required:

| § | verified correct as recovered | basis |
|---|---|---|
| 1.2 | collision space, 2³²/1000 ≈ 4.3 × 10⁶ | arithmetic |
| 1.3 table, 1.4 table | the τ = 1 arithmetic itself is right — only its framing as the operating regime was wrong | arithmetic; whitepaper §6 |
| 2.1 | forward = oldest → newest | `prng_registry.py:981-990` |
| 2.3 (recovered 2.3) | per-seed GPU loop shape: advance, extract, compare, skip | `prng_registry.py:981-989` |
| 3.1, 3.3 | reverse = newest → oldest; the failure-mode table | `:3142-3155` |
| **3.2** | **"reverse" = order of data, not PRNG inversion — verbatim correct** | `miner/range_miner_worker.py:888`, `:116`; `prng_registry.py:3143` |
| 4.1 | `bidirectional = forward ∩ reverse` | design + §4.2 |
| 4.2 | eliminates directional bias | whitepaper §5 |
| 5.2 | constant-skip semantics table (skip=N → every (N+1)ᵗʰ) | `prng_registry.py:972-990` |
| 5.3 (recovered) | five hybrid strategies; variable-skip motivation | `:1021-1067` |
| 5.5 (recovered 5.4) | survivor = (seed, skip) pair | `:992-1002` |
| 6.1 | 1000 = 8 × 125, gcd = 1, CRT decomposition | arithmetic |
| 6.2 | the three lanes and their roles | `:984-986` |
| 6.3 | "lane disagreement ⇒ impossible" — algebraically true (emphasis corrected, claim not) | CRT |
| 9.1–9.3 (recovered 14) | fault mode, root-cause table, guard text, 20/20 validation | `sieve_filter.py:326-327`, `:481-482` |

Sections **corrected**: 1.1 (F-6), 1.3/1.4 framing, 2.2 (mechanism), 4.3 (F-2), 6.4 (F-1),
9.2 anchors, 9.5 (F-5). Sections **re-scoped, not restored**: recovered 7–13 → §8. Sections
**superseded**: recovered 15 → §10. Sections **new**: 5.1, 5.6, 7.

**fault-injection / positive control (VIR-2).** **Not applicable, and stated rather than
omitted.** This pass ran no detector, gate or harness that could pass vacuously. The one
executed check (§6.3's exhaustive CRT comparison) is a mathematical identity test whose
negative result is the finding, not a pass.

**completion sentinel (VIR-3).** See below.

**unavailable-observer (VIR-5).** Nothing in this chapter is established by pipeline execution.
Every claim about runtime behaviour is traced by source, and is labelled as such.

**audit claim scope (VIR-6).**

*Searched surfaces:* VM 101 working tree at `eed3904` — `prng_registry.py` (java_lcg forward
constant/hybrid, reverse constant/hybrid), `sieve_filter.py`, `miner/range_miner_worker.py`,
`window_optimizer_integration_final.py`, `survivor_scorer.py`,
`tests/phase6/known_answer_reference.py`, `daily3.json` (parsed, all 18068 records),
`docs/` (whitepaper, source map, Chapter 1 + its audit, instructions.txt, ML remediation
proposal); `git show d14dcdd:` for the recovered chapter. `/bin/grep` used throughout so
`.json` and gitignored files were included. **One live-system check:** `hostname` over SSH from
VM 101 to all three CT100 workers (`.122`, `.156`, `.164`) — this is what established F-5, and
it is the one claim here that a repository-only reader could not have made.

*Unavailable surfaces — declared, not assumed clean:*

1. ~~**`miner/`, `agents/` and `window_optimizer.py` are owned by a concurrent Resolved Execution
   Set session.**~~ **DISCHARGED at closure (`81ef3f1`, 2026-08-02).** The obligation this
   surface created was: *"if that session lands changes, re-verify §5.4, §7.2 and §8.4."*
   **That session landed** — the Resolved Execution Set (`63e627f`) and admission binding
   (`eff6616`). The re-verification was performed:

   - `git diff --stat eed3904..81ef3f1` over `miner/`, `agents/` and `window_optimizer.py` shows
     four files changed — `agents/watcher_agent.py`, `miner/dataset_authority.py`,
     `miner/range_miner_coordinator.py`, `window_optimizer.py`. **`miner/range_miner_worker.py`
     is not among them**, and `prng_registry.py` and `window_optimizer_integration_final.py` are
     unchanged over the same span.
   - Every anchor in §5.4, §7.2 and §8.4 was nonetheless re-read at `81ef3f1` rather than
     inferred from the diff: `prng_registry.py:1007-1012`, `:974-976`;
     `miner/range_miner_worker.py:196-197`, `:648-649`, `:602-650`;
     `window_optimizer_integration_final.py:273`. **All exact, none moved.**

   The three sections are therefore **verified unchanged**, not merely unchallenged. This pass
   again edited only `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md`.
2. **Deployed source on the rigs was not compared against VM 101.** Every kernel claim here is a
   claim about the VM 101 tree. The hostname query is the sole system-scoped fact. **Repo ≠
   system.**
3. **No runtime values.** No GPU, sieve, miner, WATCHER or pipeline execution. Threshold, skip
   and partition behaviour is traced by source, never measured.
4. **The 40 non-java_lcg registry kernels were not read in full.** Only `java_lcg`,
   `java_lcg_reverse`, `java_lcg_hybrid`, `java_lcg_hybrid_reverse` were opened end to end.

   **Amended at closure (`81ef3f1`, 2026-08-02).** This surface previously read: *"The '39
   lane-test occurrences' count (§6.2) is a grep count over the live registry, not 39
   individually read kernels."* Both halves are now superseded:

   - **The number is 43, not 39** (§6.2.1), and 39 is withdrawn as unreproducible.
   - **It is no longer a bare grep count.** The count is structural — a mod-1000 comparison
     whose following two lines carry the mod-8 and mod-125 conjuncts — and §6.2.1 publishes the
     four-line program that produces it, so the figure is re-derivable without reading 44
     kernels. Each of the 43 sites was additionally resolved to its owning kernel, which is how
     the 44th (`mt19937_hybrid_multi_strategy_sieve`, `:773`) was identified as the single-lane
     exception and the old "one per kernel" gloss was retired.

   **What remains genuinely unavailable:** the *bodies* of the 40 non-java_lcg kernels are still
   unread. This chapter asserts that each contains the lane conjunction and nothing about what
   else it does. That is sufficient for §6, which is a claim about one predicate, and
   insufficient for any claim about those families' correctness — which this chapter does not
   make, and which is moot for TFM (java_lcg only).
5. **The CA draw-procedures PDF is not in the repo**, and was **not read in this pass either**.
   §5.1's citation is marked **`UNAVAILABLE`**. It was originally transcribed from the project's
   own record of it (skill §0.4) — **and that record was itself wrong about the pre-test count**,
   which is precisely how the error propagated unchecked. §5.1 is now corrected from **Team
   Beta's ruling text**, which is **not** the same as verification at source and is not
   presented as such.
6. **Team Beta's ruling texts** exist outside the repo except where transcribed.

---

### Completion sentinel

```
STATUS:  PASS
```

`PASS` is claimed for the **declared scope**: the chapter is restored from `d14dcdd`, §1–4 and
§14 are verified against live source with corrections stated, §5 and §6 are restored and
extended, §7–13 are re-scoped with citations rather than duplication, and eight findings are
recorded with anchors. **It is not a claim that Step 2 is fully verified** — §11.3 states
exactly what remains unproven, and §12.1 lists six inherited open items this chapter does not
own.

> The sentinel above belongs to the **restoration and correction** pass (`eed3904`). The
> **closure** pass carries its own, in §14 below. They are not the same claim and are not
> merged.

---

## 14. Closure statement

### 14.1 Verified against

**Commit `81ef3f1`, 2026-08-02**, on VM 101 (`192.168.3.177`), working tree, venv
`~/venvs/torch`. The chapter's body was written against `eed3904`; this pass re-verified it
against `81ef3f1` rather than assuming the interval was quiet.

### 14.2 What is verified this pass

**The one open item is closed.** §6.2's unreproducible "39 occurrences" is withdrawn and
replaced by **43**, with the counting method published as executable code (§6.2.1) so the figure
is re-derivable rather than trusted. The three candidate counts are reconciled — 31 measures a
formatting variant, 30+13 is the same 43 by another route, and 39 measures nothing. The "one per
kernel" gloss is retired: it is 43 of 44 kernels, with
`mt19937_hybrid_multi_strategy_sieve` (`prng_registry.py:773`) the single-lane exception.

**Two §1.1 anchor errors corrected** — `prng_registry.py:1004` is the closing `'''` of a kernel
string, not the `default_params` block (`:3963-3966`); and `a`/`c` are hardcoded in **two**
kernel bodies (`:3125-3126`, `:3182-3183`), not three, both of them reverse kernels.

**The chapter's own re-verification obligation is discharged.** VIR-6 unavailable-surface #1
made §5.4, §7.2 and §8.4 conditional on a concurrent session landing. It landed (`63e627f`,
`eff6616`). All anchors in those three sections were re-read at `81ef3f1` and **none moved**;
`miner/range_miner_worker.py`, `prng_registry.py` and `window_optimizer_integration_final.py`
are unchanged over `eed3904..81ef3f1`.

**Clean control (VIR-2) — verified correct and unchanged, no edit required:**

| § | re-verified at `81ef3f1` | basis |
|---|---|---|
| 6.2 code block | the three-lane conjunction, byte-exact | `prng_registry.py:984-986`, `:3146-3148`; `:1042-1044` modulo the `draw_idx` index name |
| 5.4 | hybrid kernel signature and the skip-semantics defect | `prng_registry.py:1007-1012` |
| 7.2 | both `offset` consumers — host slice and device pre-advance | `miner/range_miner_worker.py:648-649`, `:196-197`; `prng_registry.py:974-976` |
| 8.4 | residue-window authority | `miner/range_miner_worker.py:602-650`; `window_optimizer_integration_final.py:273` |
| 6.1, 6.3 | CRT construction and the redundancy argument | arithmetic; unchanged |
| §13 VIR-2 table (19 rows) | carried forward from `eed3904`; the files backing it are unchanged over the interval | `git diff --stat eed3904..81ef3f1` |

**Fault-injection control (VIR-3): `NOT_APPLICABLE`, and stated rather than omitted.** This is a
documentation pass, and **no executable gate covers this chapter** — `tests/` contains
`test_chapter1_p0_corrections.py` and no Chapter 2 equivalent, verified this session. Chapter 1's
edits were gated and run; Chapter 2's could not be. **That asymmetry is itself an open
governance item** (§14.3), not a clean result.

### 14.3 What remains open, and where it is tracked

**Nothing found this pass was repaired.** Every item below is carried, not closed.

| Open item | Where tracked | Disposition |
|---|---|---|
| **F-3** — the lane redundancy may be residue of an unbuilt lane-parallel CRT architecture | §6.5 | **open question**, not determinable from available surfaces |
| **F-4** — `offset` drives host slice and device pre-advance from one scalar | §7.3; Chapter 1 §3.1.2 | **CONFIRMED, not repaired AT THIS ANCHOR.** Settles Chapter 1 audit C-2 as an **observed inconsistency, not the repair.** No single `offset*(skip+1)` multiplier exists under variable skip; belonged in the future **hybrid input-semantics design**, not a standalone arithmetic patch. **SUBSEQUENT: repair implemented by Window-Anchor Brief I; acceptance pending** — see §7.2.1. The anchor verdict is preserved as recorded |
| **F-5** — ROCm prelude hostname guard dead on every live rig | §9.5 | **CONFIRMED dead legacy branch, NOT a Phase-7 blocker.** Harmless today. **Do not rename the hosts** — that activates obsolete overrides. Any repair must key from an explicit platform/profile property, never another hostname tuple |
| **F-7** — whitepaper `G(s,−i)` vs forward-against-reversed-residues | §3.5 | **named, not resolved** — Beta's side of the boundary |
| The six §12.1 inherited items | §12.1 | inherited, **not owned** by this chapter: the missing CA PDF, the absent `TB_RULING_*` for the 2026-07-30/31 stream, the untracked descriptive trace, the `java_lcg_cpu` non-zero-skip mismatch (**no fix authorized**), the 44-entry registry vs 4 compiled variants, and `forward_matches`/`reverse_matches` absent from the Step-3 merge list |
| **No executable gate covers this chapter** | this section | new this pass. Chapter 1 has `tests/test_chapter1_p0_corrections.py`; Chapter 2 has no equivalent, so its claims are protected by review only. Recorded as an observation for the gate owner, **not a proposal** |
| VIR-6 surfaces 2, 3, 5, 6 | §13 | still `UNAVAILABLE`: rig-deployed source never compared against VM 101; no runtime values; the CA PDF unread; Beta ruling texts external |
| VIR-6 surface 4, residual half | §13 | the **bodies** of the 40 non-java_lcg kernels remain unread. The lane-count claim no longer depends on reading them; nothing else about those families is asserted |

### 14.4 What this chapter is NOT

- **Not a proof that Step 2 is verified.** §11.3 states exactly what is unproven. The chapter
  documents **the sieve as built**, including where as-built diverges from the whitepaper (§3.5)
  and from its own parameter registry (§7.3).
- **Not a certification of hybrid worker semantics.** Hybrid is covered by the **Phase-6 transfer
  gate**, *not* by a four-phase Wall-A consumer run. §5.4's defect — hybrid kernels do not execute
  the requested skip semantics — is **described and open**, and no run in the record has certified
  the hybrid path end to end.
- **Not a claim about the other 40 registry families.** TFM sieves `java_lcg` only; the miner
  raises `NotImplementedError` for the five uncovered families. §6.2.1's count is a claim about
  one predicate's occurrence, not about those kernels' correctness.
- **Not an operator runbook.** No procedure here is written to be executed.
- **Not a system-scoped document.** Every kernel claim is a claim about the **VM 101 tree**. The
  only system-scoped fact in the chapter is the `hostname` query that established F-5.

### 14.5 Closure sentinel

```
CHAPTER 2 CLOSURE:  PASS
```

**`PASS` means verified-and-bounded, not finished.** It is claimed for exactly this scope: the
§6.2 count is settled and reproducible by a published method, the two §1.1 anchor errors are
corrected, the chapter's declared re-verification obligation is discharged against live source,
and everything still open is enumerated in §14.3 with where it is tracked. It is **not** a claim
that Step 2 is verified, that the hybrid path is certified, or that any F-item was repaired.

**Files changed by this pass:** `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` only.

---

## Version History

```
Version 4.2.0 — 2026-08-02  CLOSURE PASS (verified against 81ef3f1)
- §6.2  SETTLED: the "39 occurrences of the lane test" claim is WITHDRAWN as unreproducible.
        The number is 43, in 43 of the registry's 44 kernels. New §6.2.1 publishes the
        counting method as executable code so the figure is re-derivable, reconciles the
        three candidate counts (31 = a (unsigned int)-cast formatting variant; 30+13 = the
        same 43 split by loop-index name; 39 = no method), and retires the "one per kernel"
        gloss by naming the exception: mt19937_hybrid_multi_strategy_sieve (:773) tests
        mod 1000 only, at :820-821. Not a defect — §6.3 already proves the three-lane test
        is exactly equivalent to that single comparison.
- §1.1  CORRECTED: prng_registry.py:1004 is the closing ''' of JAVA_LCG_KERNEL, not the
        default_params block. Correct anchor :3963-3966 (java_lcg), :3976-3979 (hybrid).
        Prefer the dict key PRNG_REGISTRY['java_lcg']['default_params'].
- §1.1  CORRECTED: a/c are hardcoded in TWO kernel bodies, not three — java_lcg_reverse_sieve
        (:3125-3126) and java_lcg_hybrid_reverse_sieve (:3182-3183). Both are REVERSE kernels;
        the forward kernels take a/c as arguments (the §3.4 ABI asymmetry).
- §13   VIR-6 surface 1 DISCHARGED: the concurrent Resolved Execution Set session landed
        (63e627f, eff6616). §5.4, §7.2 and §8.4 anchors were re-read at 81ef3f1 — none moved;
        range_miner_worker.py, prng_registry.py and window_optimizer_integration_final.py are
        unchanged over eed3904..81ef3f1.
- §13   VIR-6 surface 4 AMENDED: superseded by §6.2.1. The residual unavailability — the
        BODIES of the 40 non-java_lcg kernels are still unread — is retained explicitly.
- §14   NEW: closure statement. Verified against / what is verified / what remains open and
        where tracked / what the chapter is NOT / closure sentinel. Records that NO executable
        gate covers this chapter (Chapter 1 has one; Chapter 2 has no equivalent) as an open
        governance observation.
- NO code, tests, config or manifests touched. No F-item repaired. Nothing removed.

Version 4.1.0 — 2026-08-01  THREE FACTUAL CORRECTIONS (Beta closure conditions)
- §5.1  CORRECTED: "two pre-test draws before every live draw" was WRONG and Alpha-introduced.
        One automatic pre-test SESSION for automatic Daily draws; additional pre-test draws
        only on anomalies. The "two test draws" language applies to MANUAL SuperLotto Plus
        equipment. Only the count was wrong — skip remains physically motivated.
        Citation marked UNAVAILABLE (PDF not in repo, not read).
- §5.1  QUALIFIED: the procedures establish equipment selection, an unpublished pre-test and
        co-drawn games — NOT that every omitted output belongs to one uninterrupted PRNG
        state stream. Now stated as physically motivated CANDIDATE GAPS supporting skip as a
        DETECTOR, not proven state advances — matching §5.6's epistemics.
- §1.1  CORRECTED: full_state is NOT a "forward-compatibility seam" (2nd mischaracterisation).
        It is the deliberate synthetic known-answer / multi-modulo validation hook of the
        Wall C ruling. It changes the RESIDUE SOURCE, not the comparison width — the sieve
        does NOT compare full 32-bit values; the predicate reduces mod 1000/8/125. §6 checked
        and found consistent, not contradictory.
- §5.6  REFRAMED as design doctrine corroborated by earlier artifacts, not a historically
        discovered repository statement. NOT-FOUND table retained.
- §7.3  F-4 disposition: settles Chapter 1 C-2 as an OBSERVED INCONSISTENCY, not the repair;
        no single offset*(skip+1) exists under variable skip; belongs in the future hybrid
        input-semantics design.
- §9.5  F-5 disposition: confirmed dead legacy branch, not a Phase-7 blocker. DO NOT rename
        hosts (activates obsolete ROCm overrides); a later repair keys from an explicit
        platform/profile property, never another handwritten hostname tuple.
- §12   F-4, F-5, F-8 dispositions updated. No repairs. No code, tests or config touched.

Version 4.0.0 — 2026-08-01  RESTORE-AND-AUDIT
- RECOVERED §1-14 from d14dcdd (743 lines), destroyed by stale-copy overwrite at 248e48c
- §5.1  NEW: the physical model of why skip exists (pre-test draws, per-session equipment,
        co-drawn evening games) — existed nowhere in the repository
- §5.6  NEW: Michael's fingerprint framing — the goal was never to reverse state
- §6    RESTORED and verified; §6.4's triple-validation claim CORRECTED (CRT-redundant)
- §4.3  CORRECTED: "survivors are NOT false positives" is false at loose thresholds
- §1.1  CORRECTED: java_lcg state is 48-bit, not 32-bit
- §7    NEW: settles the open half of Chapter 1 audit C-2 (offset)
- §8    RE-SCOPED: recovered §7-13 replaced by RANGE-MINER, citing not duplicating
- §9    RETAINED from §14, anchors corrected (230/385 → 326-327/481-482), scope narrowed
- §9.5  NEW: ROCm prelude hostname guard is dead on every live rig
- §10   SUPERSEDES the twice-duplicated §15 (PWC retired from certifying authority)

Version 3.0.0 — 2025-12-30   (recovered) mathematical foundation, forward/reverse
                             clarification, probability calculations, bidirectional power
Version 2.3.1 — 2025-10-29   (recovered) fixed control flow in execute_sieve_job
Version 2.3   — 2025-10-29   (recovered) fixed hardcoded 512 buffer in run_hybrid_sieve
```

---

*End of Chapter 2: The Bidirectional Sieve*
