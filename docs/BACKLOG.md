# BACKLOG — tracked, not blocking Phase 7

**Purpose.** Everything below is known, deliberate to defer, and **not** a Phase-7 blocker. It is
written down so it is not rediscovered as a surprise finding in a later session, and so nobody
re-derives it from scratch at cost.

**Currency:** HEAD `09bbfbf` (2026-08-02). Every anchor in this file was re-read at source when
the file was written. An anchor is a claim with an expiry date — re-verify before acting.

**What is NOT here.** The hard Phase-7 prerequisites live in the skill's §8 approved sequence:
D6.2, D6.3, the `process_sharded` import gate, and 6-P2. This file is the *remainder*.

---

## 1. Unaudited chapters — the downstream half of the pipeline

**Chapters 3, 5, 6, 8 and 13 have had no scrutiny.** That is Step 3 scoring, Step 5 training,
Step 6 prediction, the PRNG registry, and the feedback loop — i.e. everything after the sieve.

Two facts set the prior for what an audit would find:

- **Chapter 1's audit found 9 of 41 claims accurate.** The base rate for an unaudited chapter in
  this project is not "mostly right with a few stale lines."
- **The one Step-3 finding we have surfaced sideways.** `forward_matches` / `reverse_matches` are
  absent from the Step-3 merge list — Beta called it *possibly the most consequential finding in
  the trace* — and it was found by an audit that was looking at something else. It was not found
  by anyone auditing Step 3, because nobody has.

`forward_matches` / `reverse_matches` are two of only four columns in the 22-array NPZ contract
that carry per-seed information (`seeds`, `forward_matches`, `reverse_matches`, `score`). They are
the **only independent per-seed sieve signal**. Their absence from the merge list needs a governed
schema decision; the miner keeps emitting both regardless, so nothing is being lost at the
producer — the loss is at the consumer.

**Implication worth stating plainly:** the audited half of the pipeline is the half that feeds the
unaudited half. Confidence in Step 2 does not transfer downstream.

---

## 2. Skip-output work — approved, sequenced after Phase 6

**The change.** Stop discarding `skip_sequences` at `window_optimizer_integration_final.py:147`,
and revive `skip_mean` / `skip_std` / `skip_entropy` — three of the five dead feature placeholders.
The producer already exists on the GPU. This requires **no kernel change**; only that the host stop
throwing the sequence away.

Verified at source (HEAD `09bbfbf`): `…final.py:140-151` builds each record as
`{'seed': seed, 'match_rate': rate}` and nothing else. The discard is not conditional and not
recent.

Lineage: the Oct-2025 output spec (`instructions.txt:1230-1245`) declares `skip_pattern` and
`pattern_stats: {mean_skip, variance, std_dev}` per survivor — the literal ancestor of the three
reviving features. This is a restoration, not an invention.

**The trap, and it is the whole difficulty.** Misses write a **fabricated `actual_skip`**.
Statistics computed naively over the returned sequence would therefore be statistics over
fabricated values. Every statistic must be taken **over valid hits only, behind an explicit
validity mask.**

**The open question that decides the shape of the work:** whether a hit mask reaches the host is
**[UNVERIFIED]**. If it does, this is a host-only change. If it does not, it becomes a kernel
change — a materially different piece of work with a different approval path. *Resolve this
before scoping, not during.*

**Naming, fixed:** `skip_search_*` for inputs, `observed_skip_*` for outputs. The whole
`skip_min`/`skip_max` confusion came from one pair of names doing two jobs at two pipeline stages.
Do not repeat it.

---

## 3. Sampler-comparison sequencing — a correction Beta issued against Alpha

The certifying four-phase TPE-vs-random comparison **cannot** be scheduled merely "after the
skip-output work." Alpha proposed that; Beta rejected it.

**Why.** The approved skip-output work (§2) retains observed sequences and restores three output
statistics. **It does not connect `skip_min` / `skip_max` to the hybrid kernels** — that is the
separate, unresolved **input-bound** interpretation, where the values die at `_hybrid_prefix`
(`range_miner_worker.py:177-193`). 22/22 constant kernels declare skip bounds; 0/22 hybrid do.

**Therefore the comparison waits for either:**
- hybrid search-input bounds with **defined effective semantics**; or
- an **explicitly phase-aware search space that does not pretend dead hybrid dimensions are
  active.**

Skip-output may proceed first. **Completing it alone does not remove the dead-dimension caveat.**
A sampler comparison run over a search space containing knobs connected to nothing measures the
sampler's behaviour in a void, and certifies nothing.

**Status quo:** TPE remains the production default *by status quo*. The five-seed run is a valid
constant-skip datapoint and useful directional evidence — **not** a certification of superiority,
and **not** authority for autonomous sampler selection.

---

## 4. Three `[WATCHER][RETRY]` log lines carrying the Chain C defect

`:1725`, `:1729`, `:1733`. The defect itself is fixed; these three call sites still emit the old
form. Cosmetic in effect, misleading in a soak log — which is exactly when someone will read them
and draw a wrong conclusion. Cheap to fix; not fixed yet.

---

## 5. Session-separated dataset authority

**Beta's ruling, and the part that is easy to misremember:** the combined publication was a
**provenance ruling**, not a finding that combined midday/evening records suit one PRNG model.

Midday and evening use **independently selected equipment** (draw procedures §II). There is
therefore **no evidentiary basis for advancing one PRNG state through interleaved records.**
Ordering is normative **within a session stream**; combined-container order carries **no
PRNG-advance meaning.**

Consequences already in force: the chronological-reorder migration was **cancelled**; combined-
session sequential sieve is **non-certifying and prohibited by default**; production
re-optimization is **per-session**.

What remains open is the *authority* question — which artifact is the session-scoped authority a
per-session run resolves against, given the pointer manifest currently addresses the combined
publication.

---

## 6. The non-terminating multi-stripe loopback — **[UNVERIFIED]**

A loopback run sized `total_seeds = 2 × miner_stripe_size` **does not terminate**; shards sit at
`staging_status='pending'`.

**What is established:** it reproduces with **no execution set frozen** and never enters the
admission-binding path, so it is independent of `eff6616`.

**What is NOT established:** whether this is a fixture limitation or a production defect. That
distinction is the whole question and it has not been answered.

**Why it is on this list rather than dismissed.** "Multi-stripe loopback does not terminate" sits
uncomfortably close to the §4.3 hang class — a trial that neither completes nor fails. That class
cost us an unbounded hang once. Better seen now than surfaced during a soak.

Beta has been asked (submission §6, ruling 3) whether this gets its own bounded investigation
before Phase 7. **Awaiting ruling.**

---

## 7. `dataset_provenance/*.json` never pruned

Same class as D6.3 — unbounded growth of run-scoped state with no retention policy. Newly found,
not yet scoped. Beta's D6.3 constraint applies by analogy and should be assumed to govern here
too: **never remove active, unresolved or audit-retained state merely for exceeding an age or
count threshold.**

---

## 8. Two read-only audits Beta wants before Phase 7

Both are **investigations, not repairs. No fix is authorised for either.**

### 8.1 `java_lcg_cpu` non-zero-skip mismatch

`prng_registry.py:170-183` applies skip **once before generating**; the kernel applies it
**between every draw** (`:987-989`). **They agree only at `skip=0`.**

Two call sites pass **non-zero** skip into that CPU path, both re-read at source:
- `survivor_scorer.py:124` — `self._cpu_func(seed=int(seed), n=n, skip=skip)`
- `full_scoring_worker.py:305` — `prng_func(seed, n_holdout, skip=offset)`, where `offset` is
  `train_history_len` and therefore routinely non-zero

**The audit question is reachability**: do these sites receive non-zero skip in practice, and if
so what does the resulting mismatch corrupt.

**Wall C caution, and the reason this matters beyond a local bug:** building the known-answer
reference on `java_lcg_cpu` would validate the **wrong semantics** — in the very deliverable whose
purpose is catching semantic error.

*Context: Michael reports all 44 PRNGs were validated through the sieves during pipeline
development — constant forward/reverse and hybrid variable-skip. An inventory is establishing what
exists before anything is scoped as new. §0.4's standing rule applies: absence of a working
implementation is not evidence of absent intent.*

### 8.2 Sampler provenance guard

`run_optimization()` trusts caller-supplied `sampler_class` / `sampler_module` / `optuna_version`
and **does not verify them against the actual sampler object.**

**Nothing already submitted is invalidated** — the existing TPE and Random wrappers are correctly
labelled. The exposure is forward-looking: **a fail-before-study guard is required before direct
use of the neutral core, or registration of any additional sampler.**

Note that `sampler` and `sampler_metadata` are already **required and keyword-only with no
default** — deliberately, so a caller cannot get TPE by omission and then report the run as
something else. **An unlabelled run is not a control.** The guard completes that intent; it does
not introduce it.

---

## 9. Small, verified, unfixed

- **`.gitignore:42` dead negation.** Line 41 is `*.json`; line 42 is
  `!config_*.json        # Keep config JSONs (safe & important)`. **`.gitignore` has no
  trailing-comment syntax** — a `#` is only a comment when it is the first character of the line.
  The pattern is therefore the entire line including spaces and comment text, and it negates
  nothing. Lines 43-44 (`!*_config.json`, `!schema_*.json`) have no trailing comment and do work.
  *This is why the `.json` extension is load-bearing for published dataset artifacts, and why they
  must not be named `*_config.json` or `schema_*.json`.*
- **The CA draw-procedures PDF is not in the repo.** It is the cited source for the physical model
  behind skip (§0.4 of the skill — pre-test sessions, per-session equipment selection, evening
  multi-game draws). Citation status is `UNAVAILABLE` until it lands. Everything derived from it is
  currently uncheckable by anyone reading only the repository.
- **Two doc-generator defects** — carried forward, not re-characterised here.
- **Chapter 1 §17.3 has 14 open items; Chapter 2 §14.3 has ten.** Enumerated in the chapters
  themselves; both chapters are closed as *verified and bounded*, which is not the same as
  *finished*.

---

## 10. Standing reminders that keep costing us

- **The repository is not the system (VIR-6).** systemd units, cron, host config and deployed
  uncommitted files are invisible to every repo-scoped gate. Alpha once reported "no scraper
  invoker exists" from a clone while an enabled boot-triggered unit sat on the host.
- **Gitignored files are invisible to every repo-scoped search.** `agent_manifests/trse.json` —
  the file *causing* TRSE F1 — had no git history at all until `93918f5`.
- **A keyword hit is not a finding until the surrounding text is read.** Four absence claims were
  falsified in one session; the last was made after a grep that **reached the exact line and did
  not read it.**
- **Cited is not read.** F6's specification sat in `TRSE_INTEGRATION_PLAN_S121.md`, tracked, cited
  repeatedly, unopened.
