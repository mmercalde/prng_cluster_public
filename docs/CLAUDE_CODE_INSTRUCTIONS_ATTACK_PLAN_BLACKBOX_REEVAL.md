# CLAUDE CODE INSTRUCTIONS — RE-EVALUATE THE ATTACK PLAN UNDER THE BLACK-BOX FRAMING (READ-ONLY)

**Host:** VM101, repo `~/distributed_prng_analysis`.

## CONSTRAINT — READ-ONLY. NO LAUNCHING.

Pipeline runs are MICHAEL-INITIATED ONLY; Beta holds gate 12 and the Phase-7 soak. Do not start
`watcher_agent.py`, `window_optimizer.py`, the fleet script, any worker, or bind 5700. No commits,
no production edits. Permitted: reading, git history, read-only DB reads. Write one file.

## Why this task exists

`docs/CLAUDE_CODE_REPORT_ATTACK_PLAN_FROM_PROCEDURES.md` (yours, this session) is thorough and its
Part A/B derivations are accepted. **But the owner has identified a framing error that may
invalidate parts of C, D and E.**

The report repeatedly evaluates approaches against the standard *"can this find the true
generator's actual trajectory?"* — a **state-recovery** standard. That is not what TFM does, and
the repo is explicit:

> `docs/PIPELINE_BEHAVIOUR_MODEL.md:1094-1095` — **TFM = Triangulated Functional Mimicry:
> functional mimicry of PRNG surface output. It is NOT seed recovery and NOT state
> reconstruction.**

> `docs/BIDIRECTIONAL_SIEVE_MATHEMATICAL_WHITEPAPER.md:158-167` — *"ML does not guess. It refines
> a space already reduced from 2³² to 10⁴."*

> `CHAPTER_2:373-377` — *"A survivor is a scored candidate, not a verdict."* The population is a
> **manifold by design**; near-misses are admitted deliberately because a population of exactly
> one has no variance and therefore no learning signal (whitepaper §7).

**The black-box framing, stated so it can be applied consistently:** TFM does not claim to
identify, reproduce or reverse the machine. It searches a *chosen generator family* for
parameterisations whose **surface output** tracks the observed published values far better than
chance. A survivor is a **mimic**, not the machine. What is extracted from survivors — how they
survived — is what carries downstream signal.

**One empirical fact the re-evaluation must account for:** the system **has** produced survivors
and the project has recorded anomalies. Any analytical framework that concludes the approach
"cannot succeed" must explain those results rather than ignore them. If your re-evaluation reaches
such a conclusion, say what the existing survivors then are.

## THREE OWNER CORRECTIONS — these are settled, do not re-open them

Treat these as given. They are the frame, not questions:

1. **The target is a PREDICTION POOL hit rate of ~65-85%, not 100% and not a unique seed.** No
   survivor is required to explain the whole series. The deliverable is a *ranked pool* graded on
   hits. Any argument of the form "no single trajectory can account for all observations" is
   therefore not an objection to the method — it is a statement about pool composition.
2. **The generator family is a SUBSTRATE FOR MIMICRY, not a hypothesis about the machine.** We do
   not care which PRNG produced the data if the heuristics can be learned and the surface output
   mimicked. **`java_lcg` was selected empirically** — initial testing flagged it as most probable
   — and it is **one of 44 entries in `KERNEL_REGISTRY`** (`prng_registry.py:3729`; verified this
   session: 44 top-level entries = 11 base families × {base, `_reverse`, `_hybrid`,
   `_hybrid_reverse`}; the families are `xorshift32, pcg32, lcg32, mt19937, xorshift64, java_lcg,
   minstd, xorshift128, xoshiro256pp, philox4x32, sfc64`). **If `java_lcg` stops fitting, another
   family is tried.** The document's silence about the vendor's generator is therefore not a
   constraint on the method at all.
3. **A two-source mixture does not break a pool.** If sessions alternate between two machines,
   some survivors mimic one source and some the other, and the pool spans both. That is the
   manifold behaving as designed (`CHAPTER_2:373-377`; whitepaper §7 on variance), not a defect.

## The task — re-evaluate, do not re-derive

**Part A and Part B stand. Do not redo them.** The session inventory (10 game-draws/day), the
derived `skip = 9` (order-invariant, session-scoped), the `skip = 7` evening-era variant, the
`skip = 39` H-B variant, and the P2b alternating-skip corollary are accepted as derived and are
not in scope for revision.

For each item below, state whether the original conclusion **SURVIVES**, **CHANGES**, or is
**VOID** under the black-box framing, with reasoning and anchors.

### 1. C.4 — the trajectory-identity argument
You concluded that per-session machine re-selection, power cycles and A/B RNG are **not** absorbed
by selectivity, because *"`skip` can represent an output consumed; it cannot represent a different
trajectory,"* and that a window of `n` observations lies within one trajectory with probability
`2^-(n-1)`.

**Re-evaluate:** that argument assumes the goal is to find the trajectory that *actually produced*
the values. Under mimicry, a survivor need not be the source trajectory — it needs to be a
parameterisation whose output tracks the observations. **Does a two-machine mixture prevent a
single mimic from tracking a session-scoped series, or does it merely change what a survivor
means?** Consider explicitly: if the observed series interleaves two sources, is a mimic that
matches it (a) impossible, (b) possible but rarer, or (c) exactly the object the manifold is
supposed to contain? Cite the whitepaper's noise-suppression argument and §7's variance rationale.

### 2. C.5 — the residue-width / H-B claim
You concluded that if each Daily 3 digit is its own selection, *"the current kernel's test can never
be satisfied by the true generator, at any seed, any skip, any window, any threshold."*

**Re-evaluate under mimicry.** The published value is a three-digit number regardless of how the
machine assembled it. A mimic is required to **emit matching values**, not to assemble them the
same way. **Does H-B actually invalidate the mod-1000 predicate, or only the claim that a survivor
is the machine?** Note also the owner's correction, which your report did not reach: the repo has
already engaged the three-selection structure — **digit features per S119 / spec `03:00-09r`,
additive alongside the CRT lanes** (`S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md:152`;
`survivor_scorer.py:616-617`, `:426-428`). Does that change D.4's rank?

### 3. E8 — the PRNG family
You recorded that *"TFM's `java_lcg` choice has no support in this document."*

**Re-evaluate:** under mimicry, is the document's silence on the generator family a limitation at
all? State plainly whether E8 is a real constraint on the method or an artifact of the
state-recovery standard. If the family is a **basis for mimicry** rather than an identification
claim, say what — if anything — would still make the family choice load-bearing.

### 4. E-series generally
Re-read E1–E10 and mark which are genuine limits on **mimicry** and which are limits only on
**identification**. E10's measurement is confirmed (measured on VM101 this session: **3,447 of
18,068 records = 19.1%** on or after 2021-06-09) — the number stands; re-evaluate only what it
implies.

### 5. The Part D ranking
Re-rank D.1–D.4 under the black-box framing if the ranking changes. Keep the requirement that at
least one approach is executable within current governed constraints. Specifically reconsider:
- whether D.1 (reaching document-governed draws) is still rank 1 — the procedures describe the
  **current-era** process, but under mimicry, do older draws become *less* useful, or merely
  *differently* governed?
- whether D.3 (two-machine mixture) changes character — its Stage 1 is free and read-only, and
  it may now be a **measurement of what the manifold contains** rather than a test of a fatal
  objection.
- whether D.4 falls further given item 2.

### 6. What the black-box framing does NOT rescue
For balance and honesty: name anything in your original report that remains a genuine problem under
mimicry. **Do not soften findings to fit the new frame.** If the era mismatch (C.3), the F-4
window-anchor coupling, or anything else is still real, say so with the same force as before.

## Report

`docs/CLAUDE_CODE_REPORT_ATTACK_PLAN_BLACKBOX_REEVAL.md`. Structure it as SURVIVES / CHANGES /
VOID per item, then the revised ranking, then §6. Every claim anchored. **"No evidence found" and
"my original conclusion was wrong" are both acceptable and preferred over hedging.** Do not
propose or implement anything.
