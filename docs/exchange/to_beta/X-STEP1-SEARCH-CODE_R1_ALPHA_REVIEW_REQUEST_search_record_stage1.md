# X-STEP1-SEARCH-CODE · R1 · REVIEW_REQUEST · Step 1 search record, stage 1 (code)

## Prior record

**Searches run** (worktree `/home/michael/tfm_worktrees/step1_search` at base `7673924`,
2026-11-14):
- The brief's Prior record: the 19 governing records of
  `docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md`, plus the brief's own searches. They are
  carried by reference.
- `/bin/grep -rlnE "base_start" docs/`.
- `/bin/grep -nE "seed_start|base_start|seed offset|non-zero start|nonzero start"` over
  `docs/BACKLOG.md`, `docs/TODO_CONSOLIDATED.md`, `docs/LEADS.md` and
  `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md`.
- `/bin/grep -rlnE "sqlite:////home/michael/distributed_prng_analysis" --include=*.py .`.
- `/bin/grep -rnE "d6_production_adapter|d6_threshold_path|test_chapter1_p0_corrections" docs/*.md`.
- `git log` on `distributed_config.json` and `tests/test_s172_admission_binding.py`.

**Documents that already decide this subject:**

| Document | What it settles |
|---|---|
| `docs/CLAUDE_CODE_INSTRUCTIONS_STEP1_SEARCH_RECORD_STAGE1_20261114.md` | This work: S1–S9, the tests, this review |
| `docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md` | The design (owner-accepted 2026-11-14). No second design round |
| `docs/exchange/from_beta/X-STEP1-SEARCH_R1_BETA_REVIEW.md` | Beta's design review. Q4 LIVE accepted in v1.1 §2.3. Q11 groups = this work's gate list |
| Owner rulings 2026-11-14, in chat, recorded in the report §0 | The stray studies; the storage path fix; the dataset copies; the staging probe; the miner finding (two guards; no `miner/` edit; a separate `base_start` brief before Run 4) |

**Labels for what this request raises:**

| Item | Label |
|---|---|
| S1–S9 | ALREADY RECORDED (the brief) |
| The RANGE-MINER path sweeps from 0 whatever `seed_start` is | NEW. It contradicts `docs/CLAUDE_CODE_REPORT_STEP1_PURPOSE_LINEAGE.md:213-215` |
| The PWC and ZMQ paths also take no start | NEW, by reading (`persistent_worker_coordinator.py:1276-1280`, `zmq_sqlite_coordinator.py:831-832`) |
| The storage path fix in `window_optimizer_bayesian.py` | NEW. Owner-ruled 2026-11-14 |
| The Route-A clean control is red until its scheduled refresh | ALREADY RECORDED as a step (v1.1 §6, §8 item 4) |

**Not searched:** the private repository, ser8, the project PDFs, VM101's live databases. The
parallel worktree `/home/michael/tfm_worktrees/volume_limit` was not read, per the brief.

---

```
id / round      X-STEP1-SEARCH-CODE / R1
type            REVIEW_REQUEST
from / to       Team Alpha (Claude Code)  ->  Team Beta (Codex)
date            2026-11-14
review class    BEFORE-COMMIT (WATCHER and the Step 1 launch path)
base_commit     76739247ef46a0f921ed89d684a209800923b2d7 (7673924), plus the uncommitted
                worktree changes listed below
answers         (opening packet)
```

**The files are in a git worktree and are NOT committed.** Read them at the absolute paths
below. Michael commits after this review, or decides without it.

| File (absolute path) | Status |
|---|---|
| `/home/michael/tfm_worktrees/step1_search/utils/step1_search_ledger.py` | new |
| `/home/michael/tfm_worktrees/step1_search/agents/watcher_agent.py` | modified |
| `/home/michael/tfm_worktrees/step1_search/window_optimizer_integration_final.py` | modified |
| `/home/michael/tfm_worktrees/step1_search/window_optimizer.py` | modified |
| `/home/michael/tfm_worktrees/step1_search/database_system.py` | modified |
| `/home/michael/tfm_worktrees/step1_search/agent_manifests/window_optimizer.json` | modified |
| `/home/michael/tfm_worktrees/step1_search/window_optimizer_bayesian.py` | modified (storage path fix only) |
| `/home/michael/tfm_worktrees/step1_search/tests/test_step1_search_record_stage1.py` | new |
| `/home/michael/tfm_worktrees/step1_search/tests/test_step1_bayes_storage_path.py` | new |
| `/home/michael/tfm_worktrees/step1_search/tests/test_seed_domain_cursor_amendment.py` | modified (fixture changes, report §8) |
| `/home/michael/tfm_worktrees/step1_search/tests/test_watcher_followup_repairs.py` | modified (F-1 fixture change, report §8) |
| `/home/michael/tfm_worktrees/step1_search/tests/test_s172_threshold_propagation.py` | modified (fixture change, report §8) |
| `/home/michael/tfm_worktrees/step1_search/docs/CLAUDE_CODE_REPORT_STEP1_SEARCH_RECORD_STAGE1.md` | new (the report; read it first) |

`git -C /home/michael/tfm_worktrees/step1_search diff 7673924` shows every change to a
tracked file.

The main tree's HEAD moved during this work, to `35f5503` (the survivor-volume gate, landed by
the owner). None of the ten files that commit touches is a file under review:
`git diff --name-only 7673924 35f5503` and this worktree's changes do not overlap.

## 1. What changed

The report gives each item with `file:line` (§7) and the evidence (§5, §6, §8, §9). In short:

- A new append-only **search ledger** (`utils/step1_search_ledger.py`), in the history ledger's
  database, with its own schema version.
- **WATCHER** schedules Step 1 from the one open search's first gap, including 0, with three
  durable results. It refuses before dispatch when the dataset or any configuration field
  differs from the record. The operator alone opens, closes and reads a search, at the CLI.
- **The credit gate (S7)** sits in the caller, before `finalize_run`. The chunk-open event, the
  link and study ownership (S8, S9) are recorded alongside it.
- `step1_search_id` is carried on all three hops.
- Two owner-ruled guards (report §6):
  - **G-MINER-START**, before dispatch: a RANGE-MINER Step 1 run at `seed_start != 0` refuses.
  - S7 credits only when the **swept interval equals the requested one**.

## 2. Questions

Allowed answers: `NO OBJECTION` · `OBJECTION` (with Severity `LIVE` or `THEORETICAL`) ·
`SUGGESTION` · `NEEDS EVIDENCE`.

1. Credit (S7): can anything be published as certified, or any coverage row be written,
   without a COMPLETE trial of this chunk that finished every required phase?
2. Scheduling (S3): is the start taken from the open search's first gap in every case,
   including 0, and are the three results distinct and durable?
3. Authority (S2, S4): is there any route by which a search is opened, or pins are
   supplied, other than the operator's command line?
4. The search ledger (S1): append-only in fact, one open search enforced in one
   transaction, and not a second certification door?
5. The link (S8): derived from the publication witness, idempotent, and safe when it fails
   after publication?
6. Refusals (S4): dataset and each configuration field, before dispatch, naming both values?
7. Parameters (S6): does `step1_search_id` arrive on all three hops from a WATCHER-shaped
   params dict?
8. Existing suites: is every changed fixture justified, and is any obsolete fallback kept?
9. Untouched: the coverage ledger, the finalizer, `miner/`, the kernels, the 22-array
   contract, S167, Route-A containment, D6.2 and threshold provenance.
10. The stage boundary: is anything left to stage 2 that Run 4 needs to be safe?
11. Any governing record this work missed.
12. (Owner, 2026-11-14) Is production behaviour from the main tree unchanged by the path fix
    in `window_optimizer_bayesian.py`?
13. (Owner, 2026-11-14) Is the finding correct (report §6: the RANGE-MINER path sweeps from 0
    whatever `seed_start` is), and do the two guards make a wrong-interval credit impossible
    until the `base_start` brief lands?
