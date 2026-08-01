# CLAUDE_CODE_INSTRUCTIONS_CHAPTER_2_SOURCE_GATHERING.md — REV1

**Chapter 2 (Bidirectional Sieve) — source gathering for a future reconstruction.**

**THIS IS RECONNAISSANCE, NOT AUTHORSHIP. Do not write Chapter 2. Do not change any code,
config or documentation. Do not commit.** The deliverable is a **source map** that lets a
later pass write the chapter quickly and correctly.

**Base:** current `main` on VM 101. Claude Code as `michael`, venv `~/venvs/torch`. You do NOT
commit, push, or run WATCHER. STOP at the gate.

**⚠️ CONCURRENCY — read before starting.** Another Claude Code session is performing the
Chapter 1 P0 remediation **right now**. It is actively editing:

```
window_optimizer.py                      ← code changes pending
docs/CHAPTER_1_WINDOW_OPTIMIZER.md       ← in progress
scripts/extract_search_bounds_snapshot.py
persistent_worker_coordinator.py         ← possible
```

**Do not read those files expecting a stable state, and do not touch them at all.** If your
work appears to require one, note it as a dependency and move on. This brief is scoped to
avoid them.

---

## 0. Why this exists

Team Beta ruled on `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md`:

```
Chapter 2 status: MISSING CORE CONTENT
repair type:      reconstruction
not:              stale-text correction
```

The file contains only **§14**, an inter-chunk GPU-cleanup note. **The bidirectional sieve —
the computational core of TFM — has no chapter.** Beta placed the reconstruction *after*
Phase 6-P0 but *before* full Phase 6 certification, "because Chapter 2 describes the
computational core that certification is meant to validate."

This pass front-loads the reading so the reconstruction itself is fast and grounded.

## 1. The question

> For each of the nine derivation sources Beta named, **what is the authoritative live source,
> where is it, and what does it currently say?**

Not: what should the chapter say. **Where would a writer get it, and is that source
trustworthy?**

## 2. Beta's nine required derivation sources

Map each to concrete `file:line` locations and summarise what a writer would find:

1. **Current forward and reverse residue construction**
2. **Host-side reverse ordering** — the `residues[::-1]` mechanism; kernels iterate the PRNG
   *forward* (skill §0.2). Confirm this is still true in live source.
3. **Constant versus hybrid kernel semantics** — the 22/22 vs 0/22 skip-bound split, and what
   `skip_sequences` / `strategy_tolerances` actually do in the hybrid path.
4. **Threshold and skip propagation** — post-`8a55a68` for thresholds; **cite**
   `THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md` and `HYBRID_SKIP_BOUND_AUDIT.md` rather than
   re-deriving them.
5. **Session-stream rules** — per-session ordering normative, combined-container order
   carrying no PRNG-advance meaning, combined-session sequential sieve non-certifying. Cite
   `DAILY3_CONSUMER_CONTRACT_v1.md` and the TB rulings.
6. **Seed-domain partitioning**
7. **RANGE-MINER execution** — coordinator, worker, stripe/sub-stripe lifecycle.
8. **D5/D6 assembly and provenance contracts** — `serial_reference` vs `process_sharded`, the
   22-array NPZ contract, the finalizer, threshold provenance.
9. **Independent bounded known-answer controls** — Beta's Phase 6 "Wall C". What exists today
   that could serve as a known-answer reference, if anything?

## 3. Also establish

- **What `docs/STEP2_BIDIRECTIONAL_SIEVE_DESCRIPTIVE_TRACE.md` contains** (currently
  untracked). Is it usable prior material for the reconstruction, partially usable, or
  superseded? This may be the single highest-value input and nobody has assessed it.
- **The relationship to `BIDIRECTIONAL_SIEVE_MATHEMATICAL_WHITEPAPER.md`.** The whitepaper
  covers the *mathematics* (survival probability, the squared exponent, why loose thresholds
  are required). The chapter should cover the *implementation*. **Draw the boundary
  explicitly** so the reconstruction neither duplicates the whitepaper nor leaves a gap.
- **What §14 (the surviving fragment) actually documents**, and whether it should be retained
  in the reconstruction or superseded.
- **Which existing audits already cover parts of this ground**, so the reconstruction cites
  rather than repeats: the threshold audit, the skip-bound audit, the consumer contract, the
  Chapter 1 audit, `TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md`.
- **Any stale or duplicate sieve documentation** that a future reader might mistake for
  current — name it, do not delete it.

## 4. Out of scope

- **Do not write any part of Chapter 2.**
- Do not change code, tests, config or documentation.
- **Do not touch `window_optimizer.py`, `docs/CHAPTER_1_WINDOW_OPTIMIZER.md`,
  `scripts/extract_search_bounds_snapshot.py`, or `persistent_worker_coordinator.py`** — the
  concurrent P0 session owns them.
- Do not re-derive the threshold path, the skip-bound findings, the dataset consumer
  contract, or Chapter 1 — **cite them**.
- Do not run the sieve, any GPU kernel, WATCHER, or the pipeline.
- Do not modify `distributed_config.json` (bare-metal addresses are deliberate).

## 5. Verification-integrity controls (VIR-1…6)

- **execution proof** — every source location carries a `file:line` anchor read this session.
- **clean control (VIR-2)** — state which of the nine sources you located **completely** vs.
  partially. A map with silent gaps is worse than one that names them.
- **fault-injection control** — n/a for read-only reconnaissance; **say so** rather than
  omitting it.
- **completion sentinel (VIR-3)** — end with explicit `PASS | FAIL | UNAVAILABLE | INCOMPLETE`
  and a per-source coverage table. A source not located is `INCOMPLETE`, never silently
  absent.
- **unavailable-observer (VIR-5)** — anything you cannot establish without executing something
  is `UNAVAILABLE`, not assumed.
- **audit claim scope (VIR-6)** — declare searched and unavailable surfaces. The rigs are
  powered on but **deployed kernel source has not been verified against VM 101** — if you do
  not check it, say so.

## 6. Deliverable

`docs/CHAPTER_2_SOURCE_MAP_v1.md`:

1. **Nine-source table** — Beta's source → authoritative live location(s) `file:line` → what
   it currently says → completeness (COMPLETE / PARTIAL / NOT FOUND).
2. **`STEP2_BIDIRECTIONAL_SIEVE_DESCRIPTIVE_TRACE.md` assessment** — usable / partially
   usable / superseded, with reasoning.
3. **Whitepaper-vs-chapter boundary** — proposed division of mathematics from implementation.
4. **§14 disposition** — retain or supersede.
5. **Citation inventory** — which existing documents the reconstruction should cite rather
   than duplicate.
6. **Gaps** — anything in Beta's nine sources with **no** authoritative source in the repo.
   *This is the most valuable output: it tells the reconstruction pass what must be written
   from live code rather than assembled from documents.*
7. **Stale/duplicate sieve documentation inventory** — named, not deleted.
8. **Coverage table + completion sentinel.**

Then STOP for Team Alpha review.
