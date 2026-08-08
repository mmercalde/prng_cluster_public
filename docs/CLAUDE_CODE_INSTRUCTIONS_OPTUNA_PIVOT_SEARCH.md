# CLAUDE CODE INSTRUCTIONS — LOCATE THE OPTUNA → FULL-SEED-SWEEP PIVOT (READ-ONLY)

**Host:** VM101, repo `~/distributed_prng_analysis`.

## CONSTRAINT — READ-ONLY, NO LAUNCHING

Pipeline runs are MICHAEL-INITIATED ONLY. Do NOT start `watcher_agent.py`,
`window_optimizer.py`, the fleet script, any worker, or bind 5700. Do NOT commit, push, or
edit production code. Permitted: reading files, `git log`/`log -S`/`show`/`grep`, reading the
governance trail in `docs/`, read-only DB reads. Write ONLY your report.

## The question

Michael's recollection: **Step 1 originally sampled window configurations with Optuna over
partial seed ranges; after establishing the system could handle it, the project PIVOTED to
sweeping the FULL 2³² seed space for a specific draw** — the goal being to find *every* seed
that can produce that draw, rather than optimizing a window config against a sampled range.

**What is actually wired today contradicts that** (Alpha verified from the public mirror):

- `agent_manifests/window_optimizer.json` default_params: `max_seeds = 1073741824` (2³⁰ —
  a QUARTER of 2³²), `strategy = bayesian`, `window_trials = 3`, `prng_type = java_lcg`.
- Tonight's live `EXEC CMD` (`logs/gate12_prodshape_20260807_180116.log`):
  `--strategy bayesian --trials 8 --max-seeds 1073741824 --seed-start 16106127360`.
- `agents/watcher_agent.py:1662-1700` — the `[S140] SEED COVERAGE TRACKER` reads
  `MAX(seed_range_end)` from `exhaustive_progress` and ADVANCES `seed_start` between runs so
  ranges are never re-searched. The cursor is at **16,106,127,360 ≈ 3.75 × 2³²**.
- No `2**32` / `4294967296` / `4_294_967_296` constant exists in the Step-1 path; the only
  "exhaustive" hits are legacy menu modules (`modules/performance_analytics.py`,
  `modules/advanced_research.py`) unrelated to Step 1.

Alpha's only memory of the full-sweep frame is from a **Fantasy 5 (California) proposal**
analysis (~July 2026) — "full 2³² seed sweep in Step 1, no Optuna" — which may have been
scoped to F5 and never back-ported to the daily3 / `java_lcg` path. **That is a hypothesis,
not a finding. Confirm or refute it with evidence.**

## What to determine — evidence, not inference

1. **Was the pivot ever IMPLEMENTED?** Search the full history, not just HEAD:
   - `git log -S "4294967296" --oneline --all`, same for `4_294_967_296`, `2**32`,
     `full_sweep`, `exhaustive_sweep`, `no_optuna`, `full_space`.
   - `git log --all --oneline --grep="exhaustive" --grep="full sweep" --grep="2\^32"
     --grep="full seed" -i`
   - Look for any Step-1 code path that bypasses Optuna entirely (a sweep loop rather than
     `strategy.search(objective, ...)`), in current or deleted files.
   - Check whether `--strategy` accepts a non-Bayesian exhaustive value
     (`window_optimizer.py` argparse) and whether any such strategy class exists.
2. **Was it ever RULED ON or DECIDED?** The governance trail is authoritative and
   intent-indexed — search `docs/` (including `PROJECT_FILE_CATALOG.md`), any
   `TB_RULING_*`, `CLAUDE_CODE_CORRECTION*`, `TEAM_ALPHA_*`, and the skill
   (`docs/TFM_PROJECT_FACTS_SKILL.md`, now v16) for a decision to move from sampled Optuna
   trials to a full-space sweep. Distinguish carefully between:
   - a decision RULED and IMPLEMENTED,
   - a decision RULED but NOT implemented,
   - a PROPOSAL/analysis only (e.g. the Fantasy 5 document, if it exists in the repo),
   - no such decision at all.
3. **What is the intended seed space for `java_lcg` on daily3?** `java.util.Random` has a
   **48-bit** state (multiplier `0x5DEECE66D`). Determine from the kernel/seeding code what
   the sweep space actually is — 2³², 2⁴⁸, or an unbounded cursor — and whether
   `seed_start = 16,106,127,360` is meaningful or is walking past the end of the real space.
   Cite `file:line` for the seeding/kernel constraint.
4. **What does the coverage DB say?** Read-only: the `exhaustive_progress` table — which
   PRNG types, what ranges, what total coverage. Is there a recorded upper bound?

## Report

`docs/CLAUDE_CODE_REPORT_OPTUNA_TO_FULL_SWEEP_PIVOT.md`. For each of the four questions:
the answer, the evidence (commit hashes, `file:line`, doc paths, query output), and — where
the answer is "no evidence found" — say exactly that rather than inferring. If the pivot WAS
implemented and later reverted, name the reverting commit; if it was ruled but never built,
name the ruling. Do not propose or implement anything.
