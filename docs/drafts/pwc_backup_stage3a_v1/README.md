# PWC backup — Stage 3a: the PWC-only items of the Stage 3 plan

**Date:** 2026-11-20 · **Host:** VM101, as `michael` · **Model:** Opus, one session, no sub-agents.
**Worktree:** `/home/michael/tfm_worktrees/pwc-offset`, detached at main `b2efdef` (Stage 2b landed).
**Status:** uncommitted. Nothing was committed, pushed, relayed to Beta or deployed. No rigs, no GPU, no live ledger.

**File split (owner, 2026-11-20).**
- This stage may edit:
  - `persistent_worker_coordinator.py`;
  - files under `persistent/`;
  - `sieve_filter.py`, in functions PWC reaches only;
  - `scripts/prove_unchanged_outside.py` and its test;
  - new PWC test files.
- Agent 1 (Stage 3b) owns `window_optimizer*.py`, `agent_manifests/` and the launch path. None of those were touched.
- Plan read, read-only: `/home/michael/tfm_worktrees/pwc-backup/docs/drafts/pwc_backup_stage3_v1/PLAN.md`
  (S3-11, S3-13).

**Yardstick.** This stage is judged by one thing: whether PWC, on the routes that can carry it, builds its sieve
input the way RANGE-MINER does where RANGE-MINER is correct. That covers the same window, from one read of the
dataset, and phase 0. Nothing here measures hit rate. Nothing here judges either engine by whether any seed is found.
PWC's yardstick question, hit rate against `k/1000` on unseen draws, stays open for Stage 4.

---

## S3-13 is withdrawn (owner, 2026-11-20 21:32-21:35 PST)

**The owner's words, verbatim:**
- "Well range miner must be fixed none the less"
- "Please do whatever method is the quickest and least recoding"

**Why.** On the RANGE-MINER route the trial's skip range does not reach phases 1-2. The assign payload carries no
`skip_range`, so the worker falls back to `[0, 16]` (`miner/range_miner_worker.py:1293`). That is a **recorded open
finding, not a design choice**: `docs/TFM_SESSION_HANDOFF_20261114.md` item 7, F-1 (`:76-80`). S3-13 would have copied
that gap into PWC, and it would have to be undone once RANGE-MINER is fixed.

This decision **supersedes the 16:45 ruling for S3-13 only**. The 16:45 ruling was "Yes, PWC follows range miner.
Unless you have a other way to verify they both return the same results".

**What was removed** (`S3-13_removed.diff`):
- `persistent_worker_coordinator.py`: the `PWC_CONSTANT_SKIP_RANGE = (0, 16)` block and its use in
  `run_trial_persistent`. **The file is now byte-identical to `b2efdef`** (sha256 `ceca9be7d4aaea51…`, both sides), so
  all 13 `skip_range` / `_skip_range` lines are as at `b2efdef`. PWC sends the trial's `[skip_min, skip_max]` again.
- `tests/test_pwc_stage3a.py`:
  - the drift test `G-SKIP-DRIFT` and its helpers;
  - `G-SKIP-TRIAL`;
  - the mutants M-SKIP-1 to M-SKIP-4.

  `G-S311-JOBS` now passes the trial range `[6, 99]` and checks that the jobs carry it unchanged.

**What stays:** S3-11 (already in `b2efdef`), Beta Finding 2 (one dataset read) and Beta Q8 (the proof-tool changes).

The RANGE-MINER fix is scoped separately, read-only: `SCOPE_MINER_SKIP_RANGE_FIX.md`.

---

## Prior record

### Searches run (worktree at `b2efdef`; `/bin/grep`)

Scope `S` = `docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_*.md docs/TB_RULING_REQUEST_*.md docs/PROPOSAL_*.md
docs/CHAPTER_*.md docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md`, plus `/bin/grep -rl` over
`docs/exchange/` and `docs/drafts/`:

```
/bin/grep -lIiE "skip_range.{0,40}\[0, ?16\]|\[0, ?16\].{0,40}skip|constant skip range|16:45|F4\b"   $S ; docs/exchange/ ; docs/drafts/
/bin/grep -lIiE "two.read|double.read|one dataset read|second read"                               (same)
/bin/grep -lIiE "PWC.STAGE3|Stage 3a|pwc_backup_stage3|keep-block|allow-added"                     (same)
/bin/grep -lIiE "HYBRID_SKIP_BOUND|skip bounds"                                                    (same)
/bin/grep -n "skip_range" miner/*.py ; /bin/grep -nE "skip_range|_skip_range" persistent_worker_coordinator.py
```

The `F4\b` alternative matches widely. Every hit with a skip-range line was read; the deciding ones are below.

### Documents that decide or record this subject

| Document | What it settles | Label |
|---|---|---|
| Owner, 2026-11-20 21:32-21:35 PST (above) | S3-13 withdrawn; RANGE-MINER is to be fixed | Owner decision. **Supersedes the 16:45 ruling for S3-13 only**. Not yet in a repository document |
| Owner, 2026-11-21 (relayed by the lead) | Narrow the Q8 header check: flag a non-inert header part only when it is new or differs from the base's; keep the order check strict | Owner instruction. Changes this stage's own 2026-11-20 tool rule, which no ruling covered. Not yet in a repository document |
| Owner ruling 2026-11-20 16:45 PST (F4); PLAN.md S3-13 option (A) | PWC's constant phases follow RANGE-MINER's skip range | **Superseded for S3-13** by the 21:32-21:35 decision |
| `docs/TFM_SESSION_HANDOFF_20261114.md:76-80` (item 7, F-1); `docs/drafts/BRIEF_PWC_BACKUP_STAGE2_STAGE3_20261119.md:49` | RANGE-MINER's constant phases never receive the trial's skip range; they use `[0, 16]` | ALREADY RECORDED, open. The fix is scoped in `SCOPE_MINER_SKIP_RANGE_FIX.md` |
| `docs/HYBRID_SKIP_BOUND_AUDIT.md` | Where skip values travel; the recorded skip stamp is in the frozen NPZ and is an ML feature | ALREADY RECORDED. Unchanged |
| `docs/exchange/from_beta/X-PWC-STAGE2B_R1_BETA_REVIEW.md:47` (Finding 2), `:67` (Q8) | The two-read window race; four limits of the proof tool | ALREADY RECORDED (carried to Stage 3 by the lead). Fixed here |
| `docs/drafts/pwc_backup_stage2b_v1/README.md`, "Beta round 1" | S3-11's coordinator side already landed in 2b; the SSH route keeps 0-only (owner) | ALREADY RECORDED |
| `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md` | Default rule: PWC does what RANGE-MINER does | ALREADY RECORDED |

### Labels for what this stage raises

| Item | Label |
|---|---|
| S3-13 | **Withdrawn** (owner 21:32-21:35). No code |
| F2: one dataset read for a new-key job | NEW code; fixes a recorded finding |
| F2 side effect: a dataset shorter than the window now gives "Dataset has only N entries" (the miner's separate refusal, `miner/range_miner_worker.py:909-912`) instead of `PWC_WINDOW_ANCHOR_OUT_OF_DOMAIN` | NEW; follows the miner |
| Q8: four new checks in `prove_unchanged_outside.py` | NEW tool code |
| S3-11 | ALREADY DONE in Stage 2b; no new code |

### Not searched

Rigs (off), ser8, the private repository, claude.ai project files and PDFs, Proxmox. Agent 1's worktree: only
PLAN.md was read.

---

## 1. Moving to `b2efdef`

- **Before moving:** all 29 paths `b2efdef` changed relative to `0caeaf9` were byte-identical in this tree, and no
  local-only file existed.
- **Backup:** `logs/pre_b2efdef/tree_files.tgz`.
- **Move:** `git checkout -f --detach b2efdef`. The status was clean afterwards.

## 2. The changes

### S3-11, the coordinator side of the offset split: already complete, no code

Landed in Stage 2b:
- TCP and local jobs carry `window_anchor` and `generator_phase` 0, and no `offset`;
- the worker's echo is checked;
- SSH keeps 0-only.

`run_trial_persistent` already takes `seed_start`, `pwc_coordinator_address` and `min_workers`, which S3-2 passes.
`G-S311-JOBS` re-checks the wire: anchor 25, phase 0, no `offset`, and the skip range carried as given.

### Beta Finding 2: one dataset read for a new-key job

`sieve_filter.py`, `execute_sieve_job`, new-key branch only:
- the dataset is parsed once and session-filtered once;
- then come the "fewer entries than the window" refusal, the domain check (never clamped) and the window slice
  `[anchor : anchor + window_size]`, all from that one list, as `miner/range_miner_worker.py:904-932` does;
- the second read through `load_draws_from_daily3` is gone from this branch.

**Old-style jobs are unchanged.** The legacy `coordinator.py` and ZMQ still use `load_draws_from_daily3`, clamp
included.

Checks:
- `G-F2-ONE-READ` reproduces Beta's fixture: ten rows, width 8, anchor 2, and the file shrinks after the first read.
  Now there is exactly one read and the window is rows `[2:10]` of the ten-row content.
- `G-F2-BASE-CONTROL` shows `b2efdef` read the file twice.
- `G-F2-EQUALS-MINER` compares 9 windows with the miner's `load_residue_window`.
- `G-F2-LEGACY-UNCHANGED` shows that old-style jobs give identical sieve calls and results to `b2efdef`.

### Beta Q8: the proof tool's gaps

`scripts/prove_unchanged_outside.py`:

| Gap (Beta `:67`) | Now |
|---|---|
| An added name whose statement runs code at import (`PIN = os.environ.setdefault(...)`) | `--allow-added` statements must be inert. A value must be a literal. An added def may have no decorators, only literal defaults and inert annotations. An added class needs an inert header, and its body may hold only defs, literal assignments, a docstring or `pass`. Otherwise: `allowed-added-import-time-evaluation` |
| An added name shadowing an existing import (`time = 0` after `import time`) | An added name must be bound **nowhere** at module level in the base (imports, aliases, `from` imports, assignments, `for`/`with`/`except` targets, defs, classes, inside module-level `if`/`try` too). It must also be bound exactly once in the new file. Otherwise: `allowed-added-name-invalid` |
| A changed definition header evaluated at import | When an allowed function's header changes, its parameters are compared **by name with the base** (narrowed 2026-11-21, below). A non-inert default or annotation is flagged only if the parameter is new or its AST differs from the base's: `allowed-header-import-time-evaluation`. The same applies to a non-inert return annotation that differs. Any change to the decorator list flags every decorator. Reordering existing parameters, or changing their kind (positional, keyword-only, `*args`, `**kwargs`), is always caught: `allowed-header-parameter-order-changed`. Classes keep the whole-header check |
| Allowing a big shared function hides a change in its miner branch | `--keep-block 'QUALNAME::HEADER'`, e.g. `run_bidirectional_test::if _use_miner:`. The block must be found exactly once on each side and be AST-identical. Its **preamble** must be identical too: every statement before it in the function, plus the function's signature and decorators |

**Not covered:**
- effects through other imported modules;
- an allowed change's effect on unchanged callers;
- walrus targets at module level;
- comments and formatting (`--strict-bytes` reports those).

Tests: 49/49, with 20 new cases (13 for Q8, 7 for the narrowing). The breakages were each caught: five of the Q8
checks on 2026-11-20, and six of the narrowed check on 2026-11-21 (`mutants_prove_unchanged_outside.txt`). Stage 2b's
real diffs still pass.

#### The header check narrowed (owner, 2026-11-21)

**The false positive.** Agent 1 combined both halves, and the 2026-11-20 rule flagged `run_bayesian_optimization` in
`window_optimizer.py`.
- **What it flagged:** the default `staging_high_water_bytes: int = 16 * 1024 ** 3`.
- **Why that was wrong:** the default is unchanged since `b2efdef` (`:756` → `:759`). The header changed only because
  Stage 3b appended one parameter, `pwc_coordinator_address`. The old rule checked every non-inert part of a changed
  header, not only the parts that changed.
- **Correction to the request:** the function is in `window_optimizer.py`, not `window_optimizer_integration_final.py`.
- **The new rule:**
  - parameters are matched by name with the base;
  - a non-inert default or annotation is flagged only when it is new or differs from the base's;
  - the parameters present on both sides must keep their relative order and their kind;
  - the header-change trigger now includes parameter names and kinds, so a pure swap of default-less parameters is
    seen.

**Tests added:**
- (a) an unchanged non-inert default with a new trailing inert parameter, keyword-only and positional: PASS;
- (b) a changed non-inert default: caught;
- (c) a new parameter with a non-inert default or annotation: caught;
- (d) reordered parameters, including a pure swap: caught;
- a kind change: caught;
- a decorator added next to an unchanged non-inert default: caught.

**Mutants** H1-H6 (`logs/stage3a_tool_mutants2/`, applied to a copy) were all detected. H1 restores the old rule, and
the (a) tests catch it.

**On the real file.** The check was run on a read-only copy of agent 1's `window_optimizer.py` (`logs/stage3a_tool_mutants2/`;
nothing written to that worktree), allowing its four changed functions.
- **Result:** UNCHANGED OUTSIDE ALLOWED, exit 0. `run_bayesian_optimization` has 0 import-time and 0 order issues.
- **For comparison:** the old rule gives exactly the reported false positive (`prove_unchanged_outside_demo.txt`).

**Still not caught:** a *removed* parameter. It is listed as `removed_parameters` in the report and the text output,
but it is not a failure. That is the same as the 2026-11-20 rule, and it is not an import-time effect.

## 3. Proof that RANGE-MINER is unaffected

`prove_unchanged_outside_demo.txt`, base `b2efdef`:

| File | Allowed | Result |
|---|---|---|
| `sieve_filter.py` | `execute_sieve_job` | exit 0; 31/31 top-level statements and the other 10 definitions identical |
| `persistent_worker_coordinator.py` | nothing | exit 0; **byte-identical** to `b2efdef` |

**Import proof** (`import_proof_*.txt`): a clean `python3 -I` import of both modules RANGE-MINER's launch imports
(`window_optimizer_integration_final.py:52-58`), `b2efdef` against new, under both hostnames. Environment changes,
CuPy calls and module sets are identical. **HOLDS.**

Untouched: none of the ten files in `scripts/gate12_parity_gate.py:174-185`, nothing under `miner/`, no
`window_optimizer*.py`, `agent_manifests/`, launcher or kernel.

## 4. Tests

### Focused (`focused_summary.tsv`; one at a time, ssh stubbed, CPU only)

| Suite | Result | Run | Same exit code as the before-pass on `b2efdef`? |
|---|---|---|---|
| `tests/test_prove_unchanged_outside.py` | 49/49 | 2026-11-21, after the narrowing | yes (0) |
| `tests/test_pwc_stage3a.py` (new) | 6/6 checks, 3/3 mutants | 2026-11-21, after the narrowing | new |
| `tests/test_pwc_stage2b_anchor_phase.py` | 35/35 | 2026-11-21, after the narrowing (it uses the tool through `scripts/pwc_sieve_filter_import_proof.py`) | yes |
| `tests/test_pwc_backup_stage2.py`, `tests/test_pwc_backup_step1.py`, `tests/test_pwc_stage1.py`, `tests/test_pwc_cache_reset.py`, `test_persistent_worker_harness.py`, `tests/test_s172_phase5_d3_25_candidate_ingress.py` | rc 0 | 2026-11-20, after the S3-13 removal | yes |

Both import proofs were regenerated after the narrowing. Both still HOLD and are identical to the earlier files.

**Why the 2026-11-21 narrowing needed no new full pass.** The change is confined to
`scripts/prove_unchanged_outside.py` and its own test. A repository-wide search found only two other references to
the tool (`/bin/grep -rl prove_unchanged_outside --include=*.py --include=*.sh --include=*.json`, outside `logs/` and
`data/`):
- `scripts/pwc_sieve_filter_import_proof.py`, whose only suite caller is `tests/test_pwc_stage2b_anchor_phase.py`.
  That suite was re-run (35/35).
- `tests/test_prove_unchanged_outside.py` itself.

No production module, launcher, miner file or other suite imports it. So none of the other 73 suites in the full pass
can change. The full after-pass below ran on the 2026-11-20 tool (`9c97789b…`), and its result stands for every file
except these two.

### Full before / after

How the passes are run (the Stage 2b method):
- **Runner:** `logs/pwc_stage3a_v1/run_suites.sh`, one suite at a time, `timeout 1200` each.
- **Environment:** CPU only, ssh/scp/rsync stubbed.
- **Suites:** the 2b list of 73 plus the two suites tracked at `b2efdef`: 75 in all, never
  `tests/test_s172_staging_partb.py`. "After" adds `tests/test_pwc_stage3a.py`.

- **"Before", clean `b2efdef`: VALID.** The Stage 3a files were parked in `logs/stage3a_parked/` and restored, with
  hashes, status and diff identical afterwards. 21 of 75 suites are red at base, the same set as in Stage 2b's passes.
- **The first "after" (now `logs/pwc_stage3a_v1/after_with_S3-13/`): SUPERSEDED.** It ran with S3-13 still in place.
  For the record, it gave 74 of 75 identical, Gate 22 the only difference; FAIR-1 passed in both passes. Its tally is
  kept as `before_after_tally_with_S3-13.md` and `before_after_failing_checks_with_S3-13.txt`.
- **"After", final code (S3-13 removed), `logs/pwc_stage3a_v1/after/`: DONE.** It ran 2026-11-20 22:57-23:43
  PST, after the owner said the slot was free. It was the only full pass running, and "before" was not re-run.
  - **Code:** HEAD `b2efdef`. The four changed `.py` files in `after/code_sha256.txt` hash the same as this package
    (`sieve_filter.py` `50a977bf…`, `scripts/prove_unchanged_outside.py` `9c97789b…`,
    `tests/test_prove_unchanged_outside.py` `d72128d3…`, `tests/test_pwc_stage3a.py` `5355af90…`).
    *The tool and its test have since been narrowed (2026-11-21): now `c208d641…` and `3292acda…`. They were
    re-run focused (§4, "Why the 2026-11-21 narrowing needed no new full pass").*
    `persistent_worker_coordinator.py` hashes the same as `b2efdef` (`ceca9be7…`).
  - **Result:** 76 suites (75 plus `tests/test_pwc_stage3a.py`). Per `compare_checks.py`, **73 of 75 have the same exit
    code and the same set of failing checks as "before"**. The two that differ:

    | Suite | Before | After | Cause |
    |---|---|---|---|
    | `tests/test_s172_phase4_coordinator.py` | rc 0 | rc 1: Gate 22 | Git-status check (`:2417`). It names exactly this package's four uncommitted `.py` files and nothing else. Same as Stage 2b (`docs/drafts/pwc_backup_stage2b_v1/README.md:243`); clears on commit |
    | `tests/test_s172_attempt6_remediation.py` | rc 1, 76/78 | rc 1, 75/78: FAIR-1 also | Wall-clock bound: `T_cp = 0.820s exceeds the recurrence 0.787s`. **Run alone on the same code** (`logs/pwc_stage3a_v1/after_fair1_rerun/`): FAIR-1 PASS, 76/78, the same two failures as "before" (RXP-3/6, RXP-3/7). It also flaked in Stage 2b's after-passes. The suite exercises only `miner/range_miner_coordinator.py`, which 3a does not change |

  - **The other 7 rows the tally text marks as differing** (`before_after_tally.md`, same=66, differ=9) have the same
    exit code and failing set:
    - `test_s172_admission_stall` (timed out at 1200 s on both sides) and `test_watcher_llm_integration`: only
      timings in the text differ;
    - `test_prove_unchanged_outside`: 29/29 becomes 42/42, the new cases;
    - `phase5_d1_engine`, `d1_workflow`, `d2_directional_uniqueness`, `d5_process_sharded`: these nest the phase-4
      suite. Their nested check now stops at Gate 22 rather than at the phase-3 count; their own gate counts are
      unchanged (e.g. D5 7/25 on both sides).
  - **New:** `tests/test_pwc_stage3a.py` rc 0, 6/6 checks, 3/3 mutants.
  - **Hygiene:**
    - `~/miner_staging` mtime was unchanged after every suite (`2026-11-19 23:24:44`), and its listing was identical
      before and after, apart from the parent `..` entry;
    - `optuna_studies/` and the main-tree status were unchanged;
    - the 25 stubbed network calls are identical to "before", apart from one temp-dir name;
    - nothing ran on a rig and no GPU was used.
  - Files: `before_after_tally.md`, `before_after_failing_checks.txt`.

## 5. Real-world consequences on this cluster

- **Skip range:** none from this stage. PWC sends the trial's sampled range, as at `b2efdef`. RANGE-MINER still sieves
  `[0, 16]` on phases 1-2 (F-1, open), so the two engines still differ on that point until the miner is fixed.
- **F2 changes the rig-side `sieve_filter.py`.** A redeploy brings it to the rigs, and until then they keep the
  two-read form. On this cluster the dataset is not replaced during a run (Beta found no evidence of it, `:47`).
- **A dataset shorter than the window:** the error text changes, and it is still a refusal before any sieve.

## Files to land (exact, final; 20 paths)

**Modified (3):**
- `sieve_filter.py`
- `scripts/prove_unchanged_outside.py`
- `tests/test_prove_unchanged_outside.py`

**New (1):** `tests/test_pwc_stage3a.py`

**`docs/drafts/pwc_backup_stage3a_v1/` (16):**
- `README.md`
- `BETA_REQUEST_SECTION_STAGE3A_DRAFT.md`
- `SCOPE_MINER_SKIP_RANGE_FIX.md`
- `S3-13_removed.diff`
- `pwc_backup_stage3a.diff`
- `prove_unchanged_outside_demo.txt`
- `import_proof_sieve_filter.txt`, `import_proof_persistent_worker_coordinator.txt`
- `mutants_prove_unchanged_outside.txt`
- `focused_summary.tsv`, `focused_tests_test_pwc_stage3a.py.txt`, `focused_tests_test_prove_unchanged_outside.py.txt`
- `before_after_tally.md`, `before_after_failing_checks.txt` (the final pass)
- `before_after_tally_with_S3-13.md`, `before_after_failing_checks_with_S3-13.txt` (superseded, kept for the record)

Not to land: `logs/`, `data/`.

## Not verified

- **Hardware:** any GPU run, and the rigs.
- **Hit rate:** the effect of F2, Stage 4.
- **FAIR-1's cause:** not investigated beyond the solo re-run. It is a wall-clock bound in a miner-only suite.
- **Owner decisions:** the 21:32-21:35 decision exists only as relayed text and this README.
