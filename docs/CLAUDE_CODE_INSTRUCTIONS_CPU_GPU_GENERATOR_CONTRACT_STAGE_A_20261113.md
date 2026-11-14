# CLAUDE CODE INSTRUCTIONS — CPU/GPU GENERATOR CONTRACT, STAGE A (READ-ONLY) — 2026-11-13

**From:** Team Alpha · **Owner:** Michael · **Run on:** VM101, as `michael`, venv
`source ~/venvs/torch/bin/activate` · **Model:** Opus
**Base commit:** `26d3147`. Anchors below were read at that commit on the public mirror.
**Re-read every one on VM101 before relying on it.**
**Package:** WP-1 of `docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md`.
**Review class:** this stage produces a document (Beta after commit). Stage B is code that
feeds results that count (Beta before commit) and is **not part of this brief**.

**Why.** TFM is judged by pool hit rate against `k/1000` on draws the pipeline never saw
(`CLAUDE.md`, "The yardstick"). The training target and the Step 6 vote are both built by
running each surviving candidate on the CPU. The sieve selected those candidates on the GPU
with different stepping, and the CPU callers pass one argument, `skip`, with more than one
meaning. Before anything is changed, the owner needs to see exactly what each route does.
The owner has authorised a fix on the condition that it breaks nothing; this stage is how
that condition is met.

---

## 0. Rules for this session

- **Read-only investigation. You edit no existing file.** You do not commit or push.
- **Work in a worktree.** `git worktree add --detach /home/michael/tfm_worktrees/wp1_stage_a HEAD`.
  Write only inside it. You may **read** the main tree at `/home/michael/distributed_prng_analysis`
  (gitignored JSON such as manifests and survivor files exists only there).
- **Files you may create, all inside the worktree:**
  - `docs/CLAUDE_CODE_REPORT_CPU_GPU_GENERATOR_CONTRACT_STAGE_A.md`
  - scratch scripts and logs under `data/wp1_stage_a/` (untracked scratch inside the worktree, not for commit)
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
- **No fleet, no GPU, no rig SSH.** CPU-only on VM101. Nothing is installed or upgraded.
- **No live state is written.** Survivor artifacts, ledgers and study databases are opened
  read-only (SQLite with `mode=ro`).
- **Yardstick.** Do not judge the sieve. Do not use "the generating candidate must survive"
  as a test, and do not describe any route as state recovery. This brief asks what the code
  does, and whether two pieces of code agree.
- **Absence claims.** Any statement that something is missing, unused or untested carries the
  search that produced it (command and scope). What you could not read is `UNAVAILABLE`.

## 1. What Alpha read at `26d3147`

| Anchor | What it shows |
|---|---|
| `prng_registry.py:170-183` | `java_lcg_cpu(seed, n, skip)`: advances `skip` once, then emits `n` consecutive outputs |
| `prng_registry.py:971-995` | constant `java_lcg` kernel: advance `offset`, advance `skip`, then per draw: one step, compare, advance `skip` again |
| `prng_registry.py:4195` | `get_cpu_reference` |
| `survivor_scorer.py:119-125` | `_generate_sequence` → `self._cpu_func(seed, n, skip)` |
| `survivor_scorer.py:350-355` | `extract_ml_features(seed, lottery_history, …, skip)` generates `len(lottery_history)` outputs |
| `full_scoring_worker.py:228-270` | `compute_holdout_hits`: no skip at all |
| `full_scoring_worker.py:299-309` | batch `holdout_hits`: `skip = train_history_len` ("OFFSET DERIVED — THIS IS THE LAW (per Team Beta)") |
| `full_scoring_worker.py:411-423` | `holdout_quality`: `skip_val = get_survivor_skip(meta)`, history = holdout draws only |
| `holdout_quality.py:26`, `:61` | `get_survivor_skip`, `compute_holdout_quality` |
| `meta_prediction_optimizer_anti_overfit.py:1479-1482` | training target is `holdout_quality` |
| `prediction_generator.py:752-800`, `:839-848` | Step 6: features per survivor, then `seq = _generate_sequence(seed, next_idx + 1, skip=survivor_skip)`, vote = `seq[next_idx]`, `next_idx = len(lottery_history)` |
| `tests/phase6/known_answer_reference.py` | independent CPU transcription of the four `java_lcg` kernel variants |
| `docs/BACKLOG.md` §8.1 | the registered investigation |

Beta's review of this topic: `X-REVERSE-SIEVE_R2_BETA_REALIGNMENT.md` §2 (delivered by hand
2026-11-13; ask Michael for the path if it is not under `docs/exchange/from_beta/`).

## 2. Questions to answer

**Q1 — Caller inventory.** List every production caller of a `*_cpu` generator,
`get_cpu_reference`, or `SurvivorScorer._generate_sequence` / `extract_ml_features`. For each:
file:line, what value is passed as `skip`, where that value comes from, and what it **means**
at that call (start position, gap between draws, the survivor's recorded skip, or other).
List backup and dated copies separately as excluded, with the reason.

**Q2 — Stepping table, all 11 families.** For each family, and for the CPU function, the
constant kernel and the hybrid kernel, state: how the seed becomes the initial state; how
`offset` is applied; any advance before the first draw; the advance between draws; the
output projection (shift, mask, modulus). Mark each cell AGREE or DIFFER against the constant
kernel. Use `tests/phase6/known_answer_reference.py` where it already covers a variant.

**Q3 — What the sieve records, and what arrives.** Per survivor, what does Step 1 record
about skip (constant: best skip; hybrid: a sequence)? Which of those fields reach Step 3
(`get_survivor_skip`) and Step 6 (`_get_survivor_skip`), and what is used when a field is
absent?

**Q4 — Training route.** For `holdout_quality`, `holdout_hits`, and the training-window
features: which outputs of the candidate's stream are compared with which draws? State it as
raw output indices. Say plainly where the three differ from each other.

**Q5 — Prediction route.** Which raw output index does Step 6 use as the next-draw vote for
a survivor with recorded skip `s` and history length `N`? Is there any other route that
builds a pool? Show the search.

**Q6 — Window anchor and generator phase.** Brief I separated the window anchor from the
generator phase. Cite the Brief I documents and say what the CPU routes would have to know
to line up with a window that does not start at draw 0.

**Q7 — Size of the difference (plumbing only).** CPU-only, small. For a sample of candidates
(a saved survivor artifact opened read-only if one exists, otherwise synthetic seeds), and
for each route in Q4 and Q5, report how often the route's output at a draw equals the
kernel-semantics output at that same draw. This measures agreement between two pieces of
code. It is **not** a hit rate and says nothing about TFM's performance.

**Q8 — Proposed contract. Do not implement.** Propose named parameters that separate
(a) start position, (b) gap between draws, (c) output projection. For each caller in Q1 say
which it needs. Give at least two options for Stage B, including "document the current
convention as intentional and change nothing", and for each option list what changes:
the training target, the features, the Step 6 votes, whether models must be retrained, and
which files are edited. Confirm that no option edits a no-touch file; if one must, say so
and mark it deferred.

## 3. Deliverable

`docs/CLAUDE_CODE_REPORT_CPU_GPU_GENERATOR_CONTRACT_STAGE_A.md`, in this order: the first-step
output; a ten-line summary; Q1–Q8, each with `file:line` anchors read in this session; a
list of everything `UNAVAILABLE`; the commands used.

## 4. Stop gate

Stop when the report is written. Do not start Stage B. Tell Michael the report path. He
commits it; it then goes to Beta through `scripts/beta_relay.py` as thread **X-0002**.

## 5. Not in scope

Any code change. Any change to what the sieve does. Any performance claim. The reverse
sieve's design. Steps 2 and 4.
