# CLAUDE_CODE_INSTRUCTIONS_2389B61_AUDIT_AND_WARMSTART_RESTORE.md — REV1

**Two parts. Part A is a read-only audit and comes FIRST. Part B is a bounded restoration.**

**Base:** HEAD `fbe5ff4` or later. Claude Code on **VM101** as `michael`, venv `~/venvs/torch`,
**run from `/home/michael/distributed_prng_analysis`.**

**Do NOT commit or push.** Do not touch `miner/`. Do not run the pipeline, WATCHER, or any scraper.

---

## 0. Why — `2389b61` has now reverted TWO independent fixes, three months apart

**Commit `2389b61`**, titled *"feat(s172): Phase 0 — shared PRNG_TYPE_ENCODING v3.2
(registry-derived)"*, is already recorded in the project-facts skill as the canonical instance of a
defect class: **a whole-block overwrite that silently reverts unrelated work, with a commit message
that never mentions it.**

**Known revert #1** — the Optuna threshold fix from `3fdf434`. Rewritten from a pre-fix copy during
unrelated encoding work. **Found four months later, by a targeted audit.**

**Known revert #2 — discovered 2026-08-04, at soak launch.** `2389b61` **deleted all seven
warm-start parameters** from `optimize_window()` in `window_optimizer_integration_final.py`:

```
- warm_start_window, warm_start_offset, warm_start_skip_min, warm_start_skip_max,
- warm_start_fwd_thresh, warm_start_rev_thresh, warm_start_session_idx
```

They had been **added deliberately three months earlier** by **`8cb2ada`**, titled
*"fix(s166): warm-start params flow end-to-end"*, whose own message reads:

> *"window_optimizer_integration_final.py: add 6 warm_start params to `optimize_window()` signature
> + pass into `_trial_history_ctx`; window_optimizer.py: add `--warm-start-*` CLI args + wire
> through; agents/watcher_agent.py: remove `warm_start_*` from `_INTERNAL_ONLY_PARAMS`."*

**`window_optimizer.py:816` still passes `warm_start_window`.** The receiving signature no longer
accepts it. **Step 1 dies at `TypeError` before a single trial runs** — which is exactly what
happened at 16:11:50 on 2026-08-04, three seconds into a 50-trial soak.

> **⚠ Two reverts are known. Nobody has ever diffed `2389b61` in full against what it should have
> touched. Part A does that.**

**Part A comes first because Part B's scope depends on it.** If `2389b61` reverted a third thing,
restoring only warm-start leaves the pipeline broken in a way the next launch discovers.

---

# PART A — audit `2389b61` in full. READ-ONLY.

## A1. Establish what the commit claimed to do

Read the full commit message and its stat. **State what it says it changes.**

## A2. Establish what it actually did

`git show 2389b61` in full. For **every** file it touched, classify each hunk:

| class | meaning |
|---|---|
| **IN-SCOPE** | serves the stated purpose (PRNG_TYPE_ENCODING v3.2 / registry-derived encoding) |
| **⚠ OUT-OF-SCOPE REVERT** | removes or reverts something unrelated to the stated purpose |
| **INCIDENTAL** | formatting, imports, whitespace — no behavioural effect |

**For every OUT-OF-SCOPE REVERT, establish:**
1. **what was removed**, quoted;
2. **which commit ADDED it** — `git log -S "<removed text>" -- <file>`;
3. **what that commit's message said it was doing**;
4. **is the removal still in effect at HEAD**, or was it restored later? **Check HEAD, do not
   assume.**

## A3. The specific question

**Is `warm_start_*` the only out-of-scope revert besides the threshold fix, or are there more?**

Answer it with evidence. **"I found no others" is only acceptable with the searches that establish
it** — name the files examined and the method.

## A4. Blast radius

For each still-in-effect revert: **what breaks, and is it reachable?** A signature mismatch that
kills Step 1 at launch is different from a dead parameter nothing reads. **Say which each is.**

## A5. Report Part A before starting Part B

**STOP after Part A and report.** If A3 finds a third revert, Part B's scope changes and Alpha will
re-brief. **Do not begin Part B on your own judgement.**

---

# PART B — restore the warm-start parameters. AFTER Part A is reported and cleared.

## B1. This is a RESTORATION, not a design decision

**Put back exactly what `8cb2ada` added.** Do not redesign, rename, reorder, or "improve" it. Use
`git show 8cb2ada -- window_optimizer_integration_final.py` as the source of truth.

**Note the count discrepancy and resolve it from source:** `8cb2ada`'s message says **6 params**;
`2389b61` removed **7** (including `warm_start_session_idx`). **Establish which is correct at the
call site** — `window_optimizer.py:816` and its surroundings — and restore what the caller actually
passes. **Report the discrepancy either way.**

## B2. Where it goes

The signature at `window_optimizer_integration_final.py:1926`, which currently ends at
`resume_checkpoint: str = ''` (D6.2 hop 3).

**⚠ `resume_checkpoint` MUST remain.** It is part of certified D6.2 (`18a2419`) and its position is
load-bearing for hop 3 of the operator route. **Restoring warm-start must not disturb it, its
default, or its comment block.**

Also restore the `_trial_history_ctx` population that `8cb2ada` added — **the signature alone
without the context write is a parameter that goes nowhere.**

## B3. Prove the restoration end to end

**Not just "it imports."** Establish, with evidence:

1. `inspect.signature(coordinator.optimize_window)` contains every restored param — **use
   `inspect`, per the project's own rule about checking signatures before calling production
   methods**;
2. the exact call at `window_optimizer.py:816` **no longer raises `TypeError`** — construct the
   call, do not run the pipeline;
3. the values **reach `_trial_history_ctx`** — that is what `8cb2ada` was for;
4. **`resume_checkpoint` still works** — D6.2's `G-RESUME-SURFACE` and `G-RESUME-ROUTE` must still
   pass.

## B4. Non-regression

Run and report: **D6.2** (31 gates) · **D6.1** · **D3.25** · **D3.5** · **Phase 3** · **Phase 4** ·
**D5**. **D6.2 is certified — if any of its gates red, STOP and report; do not adjust the gate.**

All commands with `source ~/venvs/torch/bin/activate`. Long suites:
`python3 -u <suite> | tee /tmp/<name>.log` — **never pipe to `tail`.**

---

## What NOT to do

- **Do not fix anything Part A finds beyond warm-start.** Report it; Alpha re-briefs.
- **Do not touch `resume_checkpoint`, `_l2_sort_key`, `_select_l2_winners`,
  `CANONICAL_ARRAY_CONTRACT`, `utils/prng_encoding`, `canonical_map_hash`, or the three finalizer
  validators.**
- Do not re-execute `2389b61` or any `apply_s*.py` — **the patch corpus is forensic only.**
- **Do not commit or push.**

## Report

**Part A:** the hunk classification table · every out-of-scope revert with its adding commit and
that commit's message · the A3 answer with its searches · the A4 blast radius.

**Part B (only if cleared):** what was restored and from where · the 6-vs-7 resolution · the four
B3 proofs · the non-regression table · confirmation `resume_checkpoint` is untouched.

Then **STOP** for Team Alpha review.

⚠ Part B dirties the tree. **Tell Michael it must be committed before any soak launch.**

---

## Verification-integrity controls (VIR-1…6)

- **execution proof:** quote the `git show` hunks your classification rests on. **A classification
  without quoted diff is unverifiable.**
- **clean control:** for Part B, `inspect.signature()` **before and after** — show both.
- **fault-injection control:** after restoring, **remove one restored param and confirm the
  `TypeError` returns.** A restoration that cannot be broken was not proven to be load-bearing.
- **completion sentinel:** `PASS | FAIL | UNAVAILABLE | INCOMPLETE`. **Part A `PASS` means the
  commit was fully classified — NOT that it is clean.** State that distinction.
- **audit claim scope:** repo-scoped at the stated HEAD, plus git history.
- **searched surfaces:** every file `2389b61` touched · `git log -S` for each removed block ·
  `window_optimizer.py` · the skill's §2.7 record of this commit · **`docs/` and the governance
  trail** (VIR-6 addendum — `8cb2ada`'s intent is recorded in its own message and possibly in
  Chapter 1 §8.1/§11).
- **unavailable surfaces:** name anything you could not reach.
