# CLAUDE_CODE_INSTRUCTIONS_WARMSTART_RESTORE_PART_B.md — REV1

**Part B is CLEARED, with scope set by Part A's findings and the owner's ruling on `min_workers`.**

**Base:** HEAD (current). Claude Code on **VM101** as `michael`, venv `~/venvs/torch`,
**run from `/home/michael/distributed_prng_analysis`.**

**Part A is complete and reported.** This brief supersedes Part B of
`CLAUDE_CODE_INSTRUCTIONS_2389B61_AUDIT_AND_WARMSTART_RESTORE.md`.

---

## 0. Scope — one restoration and one documentation correction

Part A found **three** out-of-scope reverts in `2389b61`. **Only one is being restored here.**

| revert | disposition |
|---|---|
| **H4/H6 — the 7 warm-start params** | **RESTORE (§1).** Hard blocker; killed Step 1 at 16:11:50 on 2026-08-04 |
| **H3 — `min_workers`** | **DO NOT RESTORE (§2).** Owner ruling; **documentation correction only** |
| **H2 — threshold `0.50→0.01`** | **Leave alone.** Dead path in production; already governed as F3 in `THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md:32`. Both production callers pass thresholds explicitly |

## 0.1 Part A's forensic finding — carry it, it changes how to read this commit

> **The overwrite source was not any committed revision.** Every ancestor touching the file
> (`e8a69f5`, `a6bc546`, `3fdf434`, `ca06f8c`) places the S166 clear **after** the flush print with
> three comment lines; `2389b61` writes it **before**, with two. No ancestor and no `apply_s*.py`
> produces that arrangement. `docs/window_optimizer_integration_final.py` — a stale duplicate, last
> touched `7313a43` 2026-05-03 — is close but still carries the warm-start params.

**So the pasted copy was an OUT-OF-TREE WORKING FILE that had absorbed some later fixes and not
others.** That is why the reverts scatter across April and May instead of truncating cleanly at one
date.

**Consequence: you cannot predict what else this commit broke from dates.** The skill currently
records it as *"rewritten from a pre-fix copy"* — **that is not accurate and is corrected in §3.**

---

## 1. Restore the warm-start parameters

### 1.1 Seven, not six — Part A resolved it

`8cb2ada` added **6**; `a6bc546` added **`warm_start_session_idx`** a day later;
**`window_optimizer.py:816-822` passes all 7 today.** Restore **all seven.**

**Source of truth:** `git show 8cb2ada -- window_optimizer_integration_final.py` and
`git show a6bc546 -- window_optimizer_integration_final.py`. **Restore what they added. Do not
redesign, rename, reorder or "improve" it.**

### 1.2 Both halves — signature AND context write

Restore the params in `optimize_window`'s signature **and** the `_trial_history_ctx` population
`8cb2ada` added. **A signature without the context write is a parameter that goes nowhere** — which
is exactly the dead-parameter pattern this project keeps finding.

### 1.3 ⚠ `resume_checkpoint` is untouchable

Part A confirms it **"sits directly below them at the call site and is present and untouched in
HEAD's signature."**

It is part of **certified D6.2 (`18a2419`)** and is **hop 3 of 3** of the operator route. **Its
presence, position, default and comment block must all survive unchanged.**

## 1.4 Prove it — four checks, not "it imports"

1. **`inspect.signature(coordinator.optimize_window)`** contains all seven — show it **before and
   after**. *(The project's own rule: `inspect.signature()` before calling production methods.)*
2. The exact call at **`window_optimizer.py:816-822` no longer raises `TypeError`** — construct the
   call; **do not run the pipeline.**
3. The values **reach `_trial_history_ctx`.**
4. **Fault injection:** remove one restored param and confirm the `TypeError` returns. **A
   restoration that cannot be broken was not proven load-bearing.**

## 1.5 Non-regression

**D6.2 (31 gates) · D6.1 · D3.25 · D3.5 · Phase 3 · Phase 4 · D5.**

**D6.2 is certified — if any of its gates red, STOP and report. Do not adjust the gate.**

`source ~/venvs/torch/bin/activate` before every command. Long suites:
`python3 -u <suite> | tee /tmp/<name>.log` — **never pipe to `tail`.**

---

## 2. `min_workers` — DO NOT RESTORE. Correct the documents instead.

### 2.1 Owner ruling and its rationale

**Michael, as owner:** the guard's original purpose was **to ensure the whole cluster was being
utilised** — during the **PWC SSH and TCP** era, when a crashed worker's share was **picked up by
the remaining workers**, so a run could silently proceed short-handed and merely take longer.

**It was a UTILISATION check, not a correctness gate.**

**RANGE-MINER does not have that shape.** Stripes are claimed per worker against a ledger, not
redistributed by slack-picking, so the failure mode the guard was written against does not exist on
the certifying path. **PWC is retired from certifying authority** (skill §0.7).

**Therefore: not a defect. Not restored. Not a Phase-7 blocker.**

### 2.2 The documents are what is wrong

Part A: **`docs/FLEET_STATE_REQUIREMENTS_v1.md` §2.2 — the fleet analysis Beta ruled on — and skill
§2.11 mechanism 3 both ASSERT THE CHAIN WORKS**, tracing it only as far as the coordinator
attribute (*"threaded at `window_optimizer.py:1510` (`pwc_min_workers`) and `:1617`
(`min_workers`)"*) and **stopping one hop short of the deleted line.**

Their stated conclusion — **`"23 < 24 → RuntimeError, run refused"`** — **is FALSE at HEAD, and has
been since 2026-07-07.**

**A false guarantee in a Beta-reviewed document is worse than no guarantee** — a reader relies on it.

**Correct both, in this pass:**

- **`docs/FLEET_STATE_REQUIREMENTS_v1.md` §2.2** — state plainly that the refusal **is not in effect
  at HEAD**; name `2389b61` as the cause and the date; record that it was a **PWC-era utilisation
  check, not a correctness gate**, and that **the owner has ruled it not-to-be-restored** because
  RANGE-MINER claims stripes per worker against a ledger rather than redistributing by slack-pick.
  **Do not delete the original analysis** — mark it superseded and keep it.
- **`docs/TFM_PROJECT_FACTS_SKILL.md` §2.11 mechanism 3** — same correction, one paragraph, with the
  §2.2 cross-reference.

**⚠ Skill edits require the three-location rule** (skill §7): commit + dual-push, `cp` to
`~/.claude/skills/tfm-project-facts/SKILL.md` with a `.bak-vN` first, re-upload to Settings, verify
in a fresh session. **Michael performs steps 2–4 — flag it in your report; do not attempt them.**

---

## 3. Correct the skill's record of `2389b61` itself

Skill §2.7 records the overwrite as *"rewritten from a pre-fix copy."* **Part A establishes that is
not what happened** (§0.1).

**Correct it to:** an **out-of-tree working file** that had absorbed some later fixes and not
others, hence **non-contiguous** damage across April–May. **Record that three out-of-scope reverts
are now known** (threshold, warm-start, `min_workers`), that **two were found only by targeted
audit and one by a launch failure**, and that **date-based reasoning about this commit's blast
radius does not work.**

---

## What NOT to do

- **Do not restore `min_workers`.** Do not touch `2389b61`'s threshold hunk.
- **Do not touch** `resume_checkpoint`, `_l2_sort_key`, `_select_l2_winners`,
  `CANONICAL_ARRAY_CONTRACT`, `utils/prng_encoding`, `canonical_map_hash`, or the three finalizer
  validators.
- **Do not re-execute `2389b61` or any `apply_s*.py`** — the patch corpus is **forensic only**.
- Do not run the pipeline, WATCHER or any scraper. Do not touch `miner/`.
- **Do not commit or push.**

## Report

The seven restored params and their source commits · `inspect.signature()` before and after · the
four §1.4 proofs including the fault injection · the non-regression table with D6.2 explicitly
called out · confirmation `resume_checkpoint` is byte-unchanged · the two §2.2 document corrections
· the §3 skill correction · **and a reminder that the skill's three-location rule needs Michael's
steps 2–4.**

Then **STOP** for Team Alpha review.

⚠ This dirties the tree. **It must be committed before any soak launch.**

---

## Verification-integrity controls (VIR-1…6)

- **execution proof:** quote `inspect.signature()` output before and after — **not a description of
  it.**
- **clean control:** the pre-restoration `TypeError` reproduced, then absent after.
- **fault-injection control:** §1.4.4 — remove one param, confirm the `TypeError` returns.
- **completion sentinel:** `PASS | FAIL | UNAVAILABLE | INCOMPLETE`. **`PASS` means warm-start is
  restored and proven — NOT that `2389b61`'s blast radius is fully known.** Three reverts are known;
  §0.1 says dates cannot bound the rest.
- **audit claim scope:** repo-scoped at the stated HEAD, plus git history.
- **searched surfaces:** `8cb2ada`, `a6bc546`, `2389b61`, `window_optimizer.py:816-822`,
  `window_optimizer_integration_final.py`, `docs/FLEET_STATE_REQUIREMENTS_v1.md`,
  `docs/TFM_PROJECT_FACTS_SKILL.md`, `docs/THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md`.
- **unavailable surfaces:** name anything you could not reach.
