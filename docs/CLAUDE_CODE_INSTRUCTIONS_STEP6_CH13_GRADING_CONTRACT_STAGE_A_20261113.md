# CLAUDE CODE INSTRUCTIONS — STEP 6 → CHAPTER 13 GRADING CONTRACT, STAGE A (READ-ONLY) — 2026-11-13

**From:** Team Alpha · **Owner:** Michael · **Run on:** VM101, as `michael`, venv
`source ~/venvs/torch/bin/activate` · **Model:** Opus
**Base commit:** `26d3147`. Anchors below were read at that commit on the public mirror.
**Re-read every one on VM101 before relying on it.**
**Package:** WP-2 of `docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md`.
**Review class:** this stage produces a document (Beta after commit). Stage B is code that
produces results that count (Beta before commit) and is **not part of this brief**.

**Why.** TFM is judged by pool hit rate against `k/1000` on draws the pipeline never saw
(`CLAUDE.md`, "The yardstick"). That number can only be measured if the pool Step 6 writes
can be read and graded against the draw it was made for. At the base commit the default
path, the schema and the draw identity do not line up. This stage maps exactly what is
written and what is read, so one contract can be chosen.

---

## 0. Rules for this session

- **Read-only investigation. You edit no existing file.** You do not commit or push.
- **Work in a worktree.** `git worktree add --detach /home/michael/tfm_worktrees/wp2_stage_a HEAD`.
  Write only inside it. You may **read** the main tree at `/home/michael/distributed_prng_analysis`
  (gitignored JSON exists only there).
- **Files you may create, all inside the worktree:**
  - `docs/CLAUDE_CODE_REPORT_STEP6_CH13_GRADING_CONTRACT_STAGE_A.md`
  - scratch scripts and logs under `data/wp2_stage_a/` (untracked scratch inside the worktree, not for commit)
- **First step, before anything else:** check for a live run (`pgrep -af
  "range_miner_coordinator|watcher_agent|window_optimizer"`, `ss -ltn | grep 5700`) and list
  files in play (`git -C /home/michael/distributed_prng_analysis status --porcelain`,
  `git worktree list`). Put the output at the top of the report. If a run is live, do no
  heavy CPU work. If a file in play is not on the no-touch list below, stop and report.
- **No-touch files (do not edit, move, or run with side effects):** everything under `miner/`;
  `prng_registry.py`; `sieve_gpu_worker.py`; the rig parity set (`GOVERNED_FILES` in
  `scripts/gate12_parity_gate.py`); `window_optimizer*.py`; `coordinator.py`;
  `agents/watcher_agent.py`; `agent_manifests/`; `scripts/gate12_*`;
  `scripts/launch_fleet_manual.sh`; `tests/gate_s172_prod_shape.py`; `utils/run_finalizer.py`;
  `utils/canonical_records.py`; `utils/seed_coverage_ledger.py`; `distributed_config.json`.
  Reading them is expected.
- **Do not run Step 6, Chapter 13, WATCHER or any pipeline step.** Functions may be extracted
  and called on fixtures in a scratch script, as Beta did. No GPU, no fleet, no rig SSH.
- **No live state is written.** Do not create, overwrite or delete anything under
  `predictions/`, or any `prediction_pool.json`, `confidence_map.json`, `lottery_history.json`
  or diagnostics file in the main tree.
- **Yardstick.** This brief asks what is written and what is read. It makes no claim about
  whether TFM predicts.
- **Absence claims.** Any statement that something is missing, unused or never happened
  carries the search that produced it (command and scope, including whether gitignored files
  and the host filesystem were covered). What you could not read is `UNAVAILABLE`.

## 1. What Alpha and Beta read at `26d3147`

| Anchor | What it shows |
|---|---|
| `prediction_generator.py:623-627` | `generate_predictions` builds the pool through `_build_prediction_pool` |
| `prediction_generator.py:653-658`, `:708-713` | `predictions` is built as a list of integers |
| `prediction_generator.py:904-907`, `:944-946` | writes `predictions/next_draw_prediction.json` |
| `chapter_13_diagnostics.py:50` | `DEFAULT_PREDICTION_POOL = "prediction_pool.json"` |
| `chapter_13_diagnostics.py:105-118` | documented schema is a list of objects; a missing file loads as `{"predictions": [], "pool_size": 0}` |
| `chapter_13_diagnostics.py:139-168` | the outcome is the last entry in the history file |
| `chapter_13_diagnostics.py:242-244` | `pred.get("value", …)` on each prediction |
| `chapter_13_diagnostics.py:774-782` | diagnostics load predictions, then "latest draw" |
| `chapter_13_diagnostics.py:1017`, `:1047` | CLI override `--predictions` |
| `chapter_13_orchestrator.py:280-282` | `generate_diagnostics()` called with no arguments |
| `agents/watcher_agent.py:2288` | WATCHER expects `prediction_pool.json` from Step 6 (**read only; never edit**) |
| `docs/CHAPTER_CORRECTIONS_REPORT_20261111.md:28`, `:645` | the registered finding |

Beta's test (`X-REVERSE-SIEVE_R2_BETA_REALIGNMENT.md` §3): the unchanged
`compute_prediction_validation`, extracted without importing the pipeline, returned one hit
on object-format input and raised `AttributeError: 'int' object has no attribute 'get'` on
the equivalent integer-format input.

## 2. Questions to answer

**Q1 — Everything Step 6 writes.** Every output path (canonical and archive), and the full
schema of each: field names, types, and an example. Where are the pools of 20, 100 and 300
produced, and are they one file or several? Can a pool contain the same value twice?

**Q2 — Everything that reads a Step 6 output.** Search for both filenames
(`next_draw_prediction.json`, `prediction_pool.json`) across tracked files, gitignored files
in the main tree, manifests, shell scripts and systemd units. For each reader: file:line, the
path it expects, the schema it expects.

**Q3 — Reproduce Beta's control.** Extract the grading function and run it on (a) an
object-format pool with one known hit, (b) the same pool as plain integers, (c) a real
`next_draw_prediction.json` if one exists on the host (read-only copy into scratch). Report
what each returns or raises.

**Q4 — Draw identity.** What does Step 6 record about which draw a pool targets: date,
session (midday/evening), draw id, history length, data cutoff, generation time? What does
Chapter 13 use to choose the outcome it grades against? Can the two be matched today? If a
new draw is appended to the history after a pool is made, which draw does Chapter 13 grade?

**Q5 — Missing, stale and malformed inputs.** For each: what happens today (error, empty
pool graded as a miss, silent default)? Cite the line.

**Q6 — History.** Is there any record on this host of a pool being graded against a real
draw: diagnostics outputs, Chapter 13 logs, KPI files? Name the directories searched. Report
`FOUND` with paths and dates, or `UNAVAILABLE`. Do not conclude "never" from a tracked-tree
search alone.

**Q7 — Other consumers of the grade.** Who reads Chapter 13's diagnostics output, and which
fields do they need (hit count, hit rank, confidence, pool size)?

**Q8 — Proposed contract. Do not implement.** One schema; the path or paths; a required
target-draw identity field; the rule that a missing, stale or malformed input is an error and
never a miss; how duplicates are counted (`k` = distinct values). Give at least two options
for where the change lands (Step 6 side only; Step 6 and Chapter 13), and for each list the
files edited and every reader from Q2 that would be affected. `agents/watcher_agent.py` and
`agent_manifests/` are no-touch: if an option needs them, say so and mark that part deferred.
Name the positive and negative controls Stage B must pass.

## 3. Deliverable

`docs/CLAUDE_CODE_REPORT_STEP6_CH13_GRADING_CONTRACT_STAGE_A.md`, in this order: the first-step
output; a ten-line summary; Q1–Q8, each with `file:line` anchors read in this session; a
list of everything `UNAVAILABLE`; the commands used.

## 4. Stop gate

Stop when the report is written. Do not start Stage B. Tell Michael the report path. He
commits it; it then goes to Beta through `scripts/beta_relay.py` as thread **X-0003**.

## 5. Not in scope

Any code change. Running any pipeline step. The selfplay and attribution paths in
Chapter 13. Any performance claim.
