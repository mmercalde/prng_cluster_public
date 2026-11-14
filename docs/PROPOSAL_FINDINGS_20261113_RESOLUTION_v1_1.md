# PROPOSAL — RESOLVING THE 2026-11-13 FINDINGS, AND AGENT ASSIGNMENT (v1.1)

**Author:** Team Alpha (Claude) · **Date:** 2026-11-13 · **Owner:** Michael
**Supersedes:** v1.0 (same date, not committed).
**Executed by:** Claude Code sessions on VM101 · **Reviewed by:** Codex (Team Beta) through `scripts/beta_relay.py`
**Base:** public mirror at `26d3147`. Live VM101, the rigs and the private repo were **not** read.
Every line number below is from the mirror and must be re-read on VM101 before use.

## What changed from v1.0

| Change | Source |
|---|---|
| WP-1 is **authorised by the owner**, with the condition "it breaks nothing". Stage B waits for the Stage A contract and the owner's sign-off on it | Michael, 2026-11-13 |
| WP-1 no longer says "one CPU function with kernel-identical stepping". `skip` is used with two meanings by two callers; the contract comes first | Beta R2 §2, verified at source |
| WP-2 covers path **and** schema, target-draw identity, and missing input as an error | Beta R2 §3, verified at source |
| WP-3 adopts Beta's measurement rules and no longer waits on WP-2 | Beta R2 §4 |
| WP-4a moved to deferred | Michael, 2026-11-13 |
| WP-8 (alternative reverse configuration) stays deferred until WP-3 has a baseline | Michael / Alpha, 2026-11-13 |
| WP-7 gains a correction to `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md:97-98`; lead L-5 reworded | Beta R2 §5, verified |
| New §5: agent assignment for **both** tracks, RANGE-MINER first | Michael, 2026-11-13 |

---

## 0. RANGE-MINER ISOLATION (binding on every findings package)

| Rule | Detail |
|---|---|
| **No-touch files** | Everything under `miner/`; `prng_registry.py`; `sieve_gpu_worker.py`; the rig parity set (`GOVERNED_FILES` in `scripts/gate12_parity_gate.py`); `window_optimizer.py`, `window_optimizer_bayesian.py`, `window_optimizer_integration_final.py`; `coordinator.py`; `agents/watcher_agent.py`; `agent_manifests/`; `scripts/gate12_*`; `scripts/launch_fleet_manual.sh`; `tests/gate_s172_prod_shape.py`; `utils/run_finalizer.py`; `utils/canonical_records.py`; `utils/seed_coverage_ledger.py`; `distributed_config.json` |
| **No fleet** | No run, no rig SSH, no deploy, no GPU. CPU-only on VM101 |
| **No live state** | Nothing writes, moves or deletes in `~/miner_staging`, `~/miner_ledger_archive`, `.s172_accumulator/`, `.s172_checkpoint/`, published generations or Optuna databases. SQLite files are opened read-only (`mode=ro`) |
| **Separate working tree** | Each session writes only inside its own `git worktree` under `/home/michael/tfm_worktrees/`. It may **read** the main tree (gitignored JSON lives only there). The main tree stays clean for the launch gate and Gate 22 |
| **Check before starting** | First step of every session: look for a live run (coordinator or WATCHER process, port 5700) and list files in play (`git status --porcelain`, `git worktree list`). A file in play that is not on the no-touch list is reported and the session stops |
| **Landing** | Michael commits and dual-pushes, between RANGE-MINER launches |
| **Deferred, not dropped** | Any fix that needs a no-touch file waits. See §3 |

## 1. The yardstick

Pool hit rate against `k/1000` on draws the pipeline never saw (`CLAUDE.md`, `AGENTS.md`,
`docs/PITFALL_STATE_RECOVERY_YARDSTICK.md`). Per Beta R2 §4, "never saw" means unseen by
the sieve **and** by training, target construction, Optuna selection and threshold tuning
for that forecast. No package uses "the generating candidate must survive" as a test.
Acceptance is **plumbing agreement** or **hit rate**, nothing else.

## 2. Findings packages that can start now

### WP-1 — What `skip` means on the CPU side (owner-authorised)

**Registered:** `docs/BACKLOG.md` §8.1; `docs/TODO_CONSOLIDATED.md:65`, `:164`, `:238`.

**At source:**
- `prng_registry.py:170-183` — `java_lcg_cpu` advances `skip` once, then emits consecutive outputs.
- `prng_registry.py:973-990` — the constant kernel advances `offset`, then `skip`, then `skip` again after every draw.
- `full_scoring_worker.py:299-305` — `holdout_hits` passes `skip = train_history_len`: a **start position**.
- `full_scoring_worker.py:411-423` — `holdout_quality` passes the **survivor's own skip** through `survivor_scorer.py:350-355`. This is the training target (`meta_prediction_optimizer_anti_overfit.py:1479-1482`).
- `prediction_generator.py:839-848` — Step 6 takes output index `len(lottery_history)` with `skip = survivor_skip`.

One argument carries two meanings. Changing the CPU function alone would move the first
caller's alignment (Beta R2 §2). So:

- **Stage A (read-only):** brief `CLAUDE_CODE_INSTRUCTIONS_CPU_GPU_GENERATOR_CONTRACT_STAGE_A_20261113.md`. Produces the caller inventory, the per-family stepping table, and a proposed contract that separates start position, gap between draws, and output projection.
- **Owner sign-off** on the contract.
- **Stage B (code):** new module; `prng_registry.py` untouched; old functions stay. Before/after run of Step 3 scoring and Step 6 on a saved survivor file, every difference listed and explained. Beta reviews before commit.
- **Consequence to know:** holdout scores and Step 6 votes may change, and models trained on the old target would need retraining. `full_scoring_worker.py` is deployed to the rigs for Step 3, so Stage B needs a rig deploy before Step 3 next runs. RANGE-MINER runs end at Step 1.

### WP-2 — Step 6 → Chapter 13 grading contract

**Registered:** `docs/CHAPTER_CORRECTIONS_REPORT_20261111.md:28`, `:645`.

**At source:**
- Step 6 writes `predictions/next_draw_prediction.json` (`prediction_generator.py:904-907`), with `predictions` as a list of integers (`:653-658`, `:708-713`).
- Chapter 13 defaults to `prediction_pool.json` (`chapter_13_diagnostics.py:50`), is called without an override (`chapter_13_orchestrator.py:280-282`), and calls `.get()` on each prediction (`:242-244`).
- A missing file loads as an empty pool (`:114`). The outcome is the last entry in the history file (`:139-168`), with no check that it is the draw the pool targeted.
- Beta ran the unchanged grading function: object input returned one hit; integer input raised `AttributeError`.

- **Stage A (read-only):** brief `CLAUDE_CODE_INSTRUCTIONS_STEP6_CH13_GRADING_CONTRACT_STAGE_A_20261113.md`.
- **Stage B (code):** one contract for path, schema, target-draw identity and error handling. `agents/watcher_agent.py` is not edited. Beta reviews before commit.

### WP-3 — The hit-rate instrument

New files only. Reads frozen Step 6 outputs and recorded outcomes; launches nothing.
Rules fixed in the file header before any run (Beta R2 §4):
`k` = number of **distinct** values in the pool; one membership hit per draw; expected hits
= sum of each draw's `k/1000`; primary pool size, period and abstention rule declared in
advance; uncertainty reported; per-draw records kept; missing or stale inputs are errors,
not misses. Acceptance: chance on uniform random pools; the known count on a fixture.
**It makes no claim about TFM.** Beta reviews before commit.

### WP-6 — New leads (documents only)

`docs/LEADS.md` entries L-4 to L-9 as in v1.0, with **L-5 reworded**: "in constant mode each
pass picks its own best skip (`prng_registry.py:971-995`); for one fixed generated sequence
`F + R ≤ k + c`, where `c` counts positions whose forward and reversed targets agree
(Beta R2 §5)". No claim that a bidirectional survivor must use two skips.

### WP-7 — Documentation fixes

`docs/DOC_FIX_LIST_20261113.md` groups A, B, D and C2–C6, plus:
- `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md:97-98` — replace the false bound with Beta's, and remove the attribution to Beta.
- Commit the Epiphany and Functional Mimicry PDFs under `docs/reference/` so Beta can read them (searched `git ls-files` and `git log --all` by name at `26d3147`: not present).
- Commit Beta's two X-REVERSE-SIEVE replies under `docs/exchange/from_beta/` (R1 is untracked on VM101 per Beta's own status line; R2 was delivered by hand).

## 3. Deferred until RANGE-MINER is finished, or Michael schedules it

| Item | Why it waits |
|---|---|
| **WP-4a** — per-seed forward/reverse match counts into Step 3 (`survivor_scorer.py:774-779`) | Owner, 2026-11-13 |
| **WP-4b** — skip statistics dropped by the host | `window_optimizer_integration_final.py:125` is no-touch |
| **WP-5** — Step 1 objective has no noise comparison | `window_optimizer_bayesian.py:150-163` is no-touch |
| **WP-8** — alternative reverse configuration behind a flag | Touches the miner; needs WP-3's baseline to be judged. Beta has no evidence it scores higher; neither does Alpha — it is a prediction |
| **C1** — two "backward validation" strings | `prng_registry.py:3911`, `:3917`; rig parity set |

## 4. How each findings package runs

1. Alpha writes the brief. 2. A Claude Code session (Opus) does Stage A in its worktree and
stops. 3. Michael commits the report; it goes to Beta through the relay. 4. Michael signs
off the contract. 5. Stage B, Beta before commit. 6. Michael commits, between launches.

## 5. Agent assignment — both tracks, RANGE-MINER first

RANGE-MINER items are from `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md` §3.A, §3.B, §4.2.
A1 (verifier scoping) is done (`b10e1ef`). At most four agents at once (same plan, §4.1).

| # | Track | Task | Brief | State at `26d3147` |
|---|---|---|---|---|
| 1 | RANGE-MINER | **G2 + G3** — the two refusal gates for agent-launched runs | `docs/CLAUDE_CODE_INSTRUCTIONS_DELEGATED_RUN_GATES_20261111.md` (exists, base `cf9edaa`) | Not built: `scripts/env_freeze_gate.py`, `scripts/run_authorization_gate.py` and the report are absent from the mirror. Run it in a worktree; re-read its anchors at HEAD |
| 2 | RANGE-MINER | **A3 + A4** — read the coverage cursor and the attempt-9 bidirectional count; settle Run 4's seed range; list files in play | `CLAUDE_CODE_INSTRUCTIONS_RANGE_MINER_VM101_READS_20261113.md` (new) | Not found in the mirror |
| 3 | Findings | WP-1 Stage A | new, this delivery | — |
| 4 | Findings | WP-2 Stage A | new, this delivery | — |
| next | RANGE-MINER | **A2** — survivor-volume limit: design, then implementation (Beta before commit) | **Not written** (`ls docs \| grep -i VOLUME` returns nothing). Alpha writes it next | — |
| owner | RANGE-MINER | Power-on day: rigs up, `boot-proxmox`, restricted agent account (G1), environment baseline for G2, netconsole re-arm → Run 4 | completion plan §4.2 Waves 1–2 | Needs a date |

## 6. Decisions

**Recorded 2026-11-13:** WP-1 authorised (condition: breaks nothing). WP-4a deferred.
WP-8 waits for a baseline. The no-touch list is verified by each session's first step
rather than by the owner by hand.

**Still needed:** a power-on date for the rigs.
