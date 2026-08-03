# CLAUDE_CODE_INSTRUCTIONS_CHAPTER_3_ALIGNMENT_AUDIT.md — REV1

**Read-only audit. NO FIX IS AUTHORIZED. Nothing is to be repaired, rewired or refactored.**

**Base:** `575378e`. Claude Code on **VM101** as `michael`, venv `~/venvs/torch`.

**Question:** does `docs/CHAPTER_3_SCORER_META_OPTIMIZER.md` (958 lines) describe what the code
actually does today, and does its stage still align with Steps 1–2 **after RANGE-MINER replaced the
Step-2 sieve engine**?

**Deliverable:** `docs/CHAPTER_3_ALIGNMENT_AUDIT.md`. **Then STOP.**

---

## 0. Two things to know before starting

### 0.1 The base rate

**Chapter 1's audit found 9 of 41 claims accurate.** That is the prior for an unaudited chapter in
this project — not "mostly right with a few stale lines." **Verify claim by claim against live
source. A claim that merely sounds plausible is not verified.**

### 0.2 ⚠ Chapter 3 is NOT Step 3, and the numbering conflicts

`CHAPTER_3_SCORER_META_OPTIMIZER.md` is titled **"Scorer Meta-Optimizer (Step 2.5)."** But:

```python
# agents/watcher_agent.py:398-406
STEP_MANIFESTS = {
    1: "window_optimizer.json",
    2: "scorer_meta.json",      # <- WATCHER's step 2 IS the scorer meta-optimizer
    3: "full_scoring.json",
    ...
}
```

Meanwhile the project-facts pipeline reads *"Step 2 Bidirectional Sieve → survivors [RANGE-MINER
replaces this engine]"*, and the README uses conceptual stages where **sieve = 2** and
**scorer = 2.5**.

**Both schemes exist. They conflict.** And the bidirectional sieve does **not** occupy WATCHER step
2 at all — `run_bidirectional_test` lives in `window_optimizer_integration_final.py`, i.e. inside
**Step 1**, which is where the miner, the accumulator and the D3.5 finalizer are all driven from.

**Q1 is to settle this**, because every downstream alignment question depends on which numbering a
given document is speaking.

---

## 1. Q1 — the numbering, resolved from source

- Which stage does **WATCHER step 2** actually execute? Read `scorer_meta.json` and trace what it
  invokes.
- Where does the **bidirectional sieve** execute, by step index? Trace from `STEP_MANIFESTS`
  through to `run_bidirectional_test`.
- Is the conceptual-vs-executable mapping **documented anywhere**, or is it folklore? **If it is
  folklore, say so** — that is a finding in itself.
- Does Chapter 3's own text ever conflate the two?

## 2. Q2 — is Chapter 3's described execution path still live?

Chapter 3 documents a **pull architecture** with SSH-dispatched jobs, `scorer_trial_worker.py`,
`generate_scorer_jobs.py`, and a v3.1 → v3.2 change from long SSH commands to params-in-file. **Both
files still exist in the tree.** That is not the same as being reachable.

For each: **producer → artifact → consumer.** Is it invoked from a manifest, from WATCHER, from a
script, or from nothing? **Real code with no producer is not "wired"** — and the reverse also
applies: a file existing proves nothing about whether it runs.

**Note the era.** That architecture is PWC/SSH-vintage, and **PWC is retired from certifying
authority** (Beta, 2026-07-31). If Chapter 3 describes a path that is now non-certifying or dead,
that is the headline finding, not a footnote.

## 3. Q3 — THE SEAM. What does this stage consume, and did the producer change under it?

**This is the core question.** RANGE-MINER replaced the Step-2 sieve engine. Its contract is the
**22-array NPZ survivor bundle**, and the remaining steps *"must not be able to tell which engine
produced it."*

- **What does the scorer meta-optimizer actually read?** The 22-array NPZ? The certified accumulator
  generation? `bidirectional_survivors.json`? Something else?
- **Is what it reads still produced?** Note `bidirectional_survivors.json` **as survivor data is
  listed SUPERSEDED**, and the root `bidirectional_survivors_*.npz` names are now
  **finalizer-owned compatibility symlinks** (D3.5) that fail closed if a regular file appears.
- **Does it resolve the dataset through `daily3_current.json`, or open `daily3.json` directly?**
  The latter is a **legacy compatibility alias, permanently frozen**, and anything still reading it
  is **intentionally stale**.
- **Does it read `full_state`, or `entry["draw"]`, or the falsy-zero idiom `entry.get("draw") or …`?**
  The last one silently drops the **22 legitimate zero-draw records**.

## 4. Q4 — `forward_matches` / `reverse_matches`

Of the 22 NPZ columns, **only four carry per-seed information**: `seeds`, `forward_matches`,
`reverse_matches`, `score`. The latter two are the **only independent per-seed sieve signal**, and
they are **absent from the Step-3 merge list** — Beta called that *possibly the most consequential
finding in the trace*.

**Where does Chapter 3's stage sit relative to that loss?** Does it see those columns, pass them
through, drop them, or never touch them? **Does Chapter 3 claim to use them?**

**Report what you find. Do not fix it** — it needs a governed schema decision.

## 5. Q5 — claim-by-claim verification

Walk Chapter 3's substantive claims and mark each **ACCURATE · STALE · FALSE · UNVERIFIABLE**, with
a `file:line` anchor obtained **in this audit** for every one. Pay particular attention to:

- feature counts (**91 extracted / 89 trained** is current; *"~62"* and
  `full_scoring_worker.py`'s *"50 features"* are **superseded**);
- the training/holdout split described in §7 and the sampled-seeds correction in §4;
- GPU-vectorized scoring and adaptive memory batching (§8, §9) — do they match live code?;
- the CLI in §11 and the `scripts_coordinator.py` integration in §12;
- version history claims (§4) — do the described versions match what shipped?

**An unverifiable claim is `UNVERIFIABLE`, not `ACCURATE`.** Silence is never a pass.

## 6. Explicitly OUT of scope

- **Any fix, anywhere.** This audit produces findings, not patches.
- **The `java_lcg_cpu` non-zero-skip mismatch** (`survivor_scorer.py:124`,
  `full_scoring_worker.py:305`) — Beta scoped that as its **own bounded audit with no fix
  authorized**. If Chapter 3 touches it, **note the contact point and move on.**
- **D3.0-B and `convert_survivors_to_binary.py`** — open, and **Beta prohibits invoking the legacy
  converter.** Do not run it. Reading it is fine.
- Chapters 5, 6, 8, 13.

## 7. ⚠ Operational constraint — the soak launches next

**Your report file will be untracked, which reds Phase-4 Gate 22 and dirties the tree.** Item 5's
clean-tree preflight **rejects a dirty tree at finalization**, so an uncommitted report would make
the Phase-7 soak fail hours in, at publication.

**Write only `docs/CHAPTER_3_ALIGNMENT_AUDIT.md`, create no other file, and tell Michael in your
final line that it must be committed before the soak launches.**

Do NOT commit, do NOT push, do NOT run WATCHER, do NOT run the pipeline.

## 8. Report structure

Findings first, ordered by consequence — **not** a walkthrough of the chapter. For each: the claim,
the live anchor, the verdict, and what depends on it. Then the Q1–Q5 answers. Then a short list of
what a future session should not have to rediscover.

---

## Verification-integrity controls (VIR-1…6)

- **execution proof:** every verdict carries a `file:line` anchor obtained **in this audit**. A
  claim marked accurate without an anchor is not accurate, it is unchecked.
- **clean control / fault injection:** `NOT_APPLICABLE` — read-only audit, no detector under
  validation. **Write `NOT_APPLICABLE`, never `PASS`.**
- **completion sentinel:** `PASS | FAIL | UNAVAILABLE | INCOMPLETE`. **A question you could not
  answer makes the report `INCOMPLETE` for that question** — which is a good outcome and far better
  than a plausible guess.
- **audit claim scope:** state whether each finding is **repo-scoped** or verified on the **live
  host**. The repository is not the system: systemd units, cron and deployed uncommitted files are
  invisible to a repo-scoped search.
- **searched surfaces / unavailable surfaces:** enumerate both explicitly. **Gitignored files are
  invisible to every repo-scoped search** — check `git check-ignore` before concluding a config or
  manifest is absent.
