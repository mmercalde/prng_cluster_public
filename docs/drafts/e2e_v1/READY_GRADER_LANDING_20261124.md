# READY · e2e-grader branch, landing-ready check (2026-11-24)

Task: `~/distributed_prng_analysis/logs/e2e_session/TASK_GRADER_LANDING_READY_20261124.txt` (lead, 18:00).
Worktree `~/tfm_worktrees/e2e-grader`, branch `e2e-grader`. Nothing committed, pushed or landed.

## Prior record

**Searches run** (on main `abdf9f7`, read with `git grep`, so files ignored by `.gitignore` are not searched):

```
git grep -l -e <P> abdf9f7 -- docs/PROJECT_FILE_CATALOG.md 'docs/TB_RULING_*' 'docs/PROPOSAL_*' \
  'docs/CHAPTER_*' 'docs/*_AUDIT.md' docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md docs/exchange/
  P = e2e_grader | force-added | REGISTRY_REPAIR | tests/fixtures
git grep -n -i 'cheap screen\|e2e_grader\|c7fe48bf' abdf9f7 -- docs scripts tests
git log --format='%h %s' -- docs/phase6_evidence/known_answer_gate.json        (main tree)
git log --format='%h %s' -- tests/fixtures/pre_d5_range_miner_npz_writer.py.frozen
```

**Documents found that decide or record the subject:**
- **Pinning the grader:** `docs/drafts/anchor_v1/DESIGN_CHEAP_SCREEN_20261123.md:85` and `:371` (TODO-SCREEN-1). The grader is pinned by sha256 `c7fe48bf…` and must be repinned when it lands.
- **Force-adding past `.gitignore:41` (`*.json`):**
  - commit `d98298c`, whose message says "evidence records force-added past .gitignore:41";
  - `docs/PROJECT_FILE_CATALOG.md:917`, on manifests force-added and tracked;
  - `docs/TODO_CONSOLIDATED.md:442`;
  - `docs/BACKLOG.md:220`, which warns that gitignored files are invisible to repo-scoped searches.
- **Frozen reference sources in `tests/fixtures/`:**
  - `tests/fixtures/pre_d5_range_miner_npz_writer.py.frozen`, added in `bdd36b4`;
  - `tests/fixtures/*_control_*.txt`.
- The terms `e2e_grader` and `REGISTRY_REPAIR` appear nowhere in the searched scope. The branch's own drafts are not on main.

**Labels:**

| Item | Label |
|---|---|
| Missing fixtures cause the failures in the lead's main check | NEW (lead's check today) |
| Force-adding the `.json` files | ALREADY RECORDED as practice (`d98298c`; catalog `:917`) |
| Reference source placed in `tests/fixtures/` | ALREADY RECORDED as practice (`bdd36b4`) |
| Grader hash `c7fe48bf…` unchanged | ALREADY RECORDED (cheap-screen design `:85`, `:371`) |

**Not searched:** the host, the ser8 archives, the private repository, project files outside the repository, and other worktrees, except the `LEADS.md` ID check below.

## Plain summary

- **The cause.** On main, the lead's check failed because three test inputs were never part of the branch:
  - the reference grader sits in untracked `data/`;
  - the registry-repair decisions file is a `.json`, so `.gitignore:41` silently drops it;
  - two other `.json` files are named in tracked `SHA256SUMS` files, so `sha256sum -c` would fail on a fresh tree.

  The 8 mutants the lead saw as "SURVIVED_OR_UNCREDITED" were caused by the same missing inputs. With their baseline gates failing, the runner credits nothing: M39 reports `ran=False`, and RR1–RR6 sit on a suite that crashed.
- **The fix.** Test files and fixtures only. One test line changed:
  - `tests/test_e2e_grader.py:662-663`: the default reference path is now `tests/fixtures/e2e_grader_a4d96fe7.py.txt`, a byte copy with sha256 `a4d96fe7…` (checked by the test at `:701`). The `E2E_GRADER_REFERENCE` override is kept.
  - The five `.json` files stay at their current paths and are listed for **force-add** (`git add -f`), following `d98298c`. I did not move the decisions file into `tests/fixtures/`, for three reasons:
    - the record `REGISTRY_REPAIR_20261124.md` and `repair_record.json` cite it at its current path;
    - `repair_record.json` is covered by the tracked `SHA256SUMS`;
    - one copy cannot drift from a second one.
  - **No change** to `e2e_grader.py` (sha256 `c7fe48bf102d…`, verified in the check tree), `compare_settings.py` (`097b05bb…`) or `switch_check.py` (`dc1ce61e…`).
- **Proof.** I made a fresh detached worktree of main `abdf9f7` (`git worktree add --detach ~/tfm_worktrees/_landing_check_grader abdf9f7`), copied in **only** the 30 files below that existed when the tests ran, and ran the five suites one after another. Settings: `taskset -c 0-5`, `nice -n 10`, `python3 -B` with `PYTHONDONTWRITEBYTECODE=1`, no card. **All pass.** The tree held no `__pycache__` afterwards, and its only tracked change was `docs/LEADS.md`. The worktree was then removed (`git worktree remove --force`); `git worktree list` no longer shows it. The add and the remove wrote git's worktree bookkeeping under `~/distributed_prng_analysis/.git/worktrees/`, which the task asked for, as the lead's own check did.

| Suite (in the check tree) | Result | Log (this worktree) | sha256 |
|---|---|---|---|
| `tests/test_e2e_grader.py` | **36/36** | `logs/e2e_grader/landing_check_test_e2e_grader.log` | `4f21826f…` |
| `tests/test_compare_settings.py` | **9/9** | `logs/e2e_grader/landing_check_test_compare_settings.log` | `16e4f833…` |
| `tests/test_switch_check.py` | **12/12** | `logs/e2e_grader/landing_check_test_switch_check.log` | `5ff51df3…` |
| `tests/test_registry_repair.py` | **5/5** | `logs/e2e_grader/landing_check_test_registry_repair.log` | `bbf3bfda…` |
| `tests/mutate_e2e_grader.py` | **87/87 killed and credited**; baseline gates pass: True; 94.1 s | `logs/e2e_grader/landing_check_mutate_e2e_grader.log` | `39e157fc…` |

The copied file list: `logs/e2e_grader/landing_list_20261124.txt` (sha256 `a8b0e72d…`).

- **Outside dependencies:**
  - `test_switch_check.py` and `test_registry_repair.py` read the main repository's git objects (`git -C ~/distributed_prng_analysis show 41ad509:…`). The commit is on main, so landing does not change this.
  - The suites write only under the tree's own `data/e2e_grader_selftest/`.

## Landing file list (repo-relative)

N = new on main, M = modifies a tracked file, **-f** = must be force-added (`.gitignore:41`).

| # | Path | N/M | Note |
|---|---|---|---|
| 1 | `docs/LEADS.md` | M | appends L-9 only; see below |
| 2 | `docs/drafts/e2e_v1/READY_E2E_GRADER_20261123.md` | N | |
| 3 | `docs/drafts/e2e_v1/RESULT_E2E_FOLLOWUP_RUNH_20261123.md` | N | |
| 4 | `docs/drafts/e2e_v1/RESULT_E2E_GRADE_RUNH_20261123.md` | N | |
| 5 | `docs/drafts/e2e_v1/READY_GRADER_LANDING_20261124.md` | N | this note, written after the proof; no test reads it |
| 6 | `docs/drafts/loop_v1/AUTONOMY_SWITCH_MAP_20261123.md` | N | |
| 7 | `docs/drafts/loop_v1/READY_LOOP_GRADE_20261123.md` | N | |
| 8 | `docs/drafts/loop_v1/READY_SWITCH_CHECK_20261123.md` | N | |
| 9 | `docs/drafts/loop_v1/REGISTRY_REPAIR_20261124.md` | N | |
| 10 | `docs/drafts/loop_v1/REGISTRY_REPAIR_DECISIONS_20261124.json` | N **-f** | **read by `tests/test_registry_repair.py:31`**; sha256 `9cdffbb53d15…` |
| 11 | `docs/drafts/loop_v1/REGISTRY_REPAIR_OPTION_CLASSES_20261124.json` | N **-f** | evidence for item 3 of the repair record; sha256 `ac62885136ac…` |
| 12 | `docs/drafts/loop_v1/SWITCH_CHECK_20261123.json` | N **-f** | machine companion of `SWITCH_CHECK_20261123.md`, 3.8 MB; no test reads it; may be left out at the owner's choice |
| 13 | `docs/drafts/loop_v1/SWITCH_CHECK_20261123.md` | N | |
| 14 | `docs/drafts/loop_v1/registry_repair_20261124/SHA256SUMS` | N | names #17 |
| 15 | `docs/drafts/loop_v1/registry_repair_20261124/parameter_registry.py.repair.patch` | N | |
| 16 | `docs/drafts/loop_v1/registry_repair_20261124/parameter_registry.py.repaired` | N | read by `test_registry_repair.py:30`, `mutate_e2e_grader.py:302` |
| 17 | `docs/drafts/loop_v1/registry_repair_20261124/repair_record.json` | N **-f** | in #14; sha256 `3469d957d6d5…` |
| 18 | `docs/drafts/loop_v1/switch_patches_20261123/SHA256SUMS` | N | names #21 |
| 19 | `docs/drafts/loop_v1/switch_patches_20261123/parameter_registry.py.patch` | N | |
| 20 | `docs/drafts/loop_v1/switch_patches_20261123/parameter_registry.py.patched` | N | |
| 21 | `docs/drafts/loop_v1/switch_patches_20261123/manifest_fragments.json` | N **-f** | in #18; sha256 `46e5560f2793…` |
| 22 | `docs/exchange/to_beta/X-E2E-GRADER_R1_ALPHA_REVIEW_REQUEST_grader_first_form.md` | N | |
| 23 | `scripts/diagnostics/compare_settings.py` | N | unchanged, `097b05bb…` |
| 24 | `scripts/diagnostics/e2e_grader.py` | N | **unchanged, `c7fe48bf…`** (pinned by the screen) |
| 25 | `scripts/diagnostics/switch_check.py` | N | unchanged, `dc1ce61e…` |
| 26 | `tests/fixtures/e2e_grader_a4d96fe7.py.txt` | N | **new fixture**, sha256 `a4d96fe7ac44…`, byte copy of `data/e2e_grader_selftest/reference/…`; `*.txt` is not ignored (checked with `git check-ignore`) |
| 27 | `tests/mutate_e2e_grader.py` | N | |
| 28 | `tests/test_compare_settings.py` | N | |
| 29 | `tests/test_e2e_grader.py` | N | **changed this task** (`:662-663`, reference path); sha256 now `29cc7f5d…` |
| 30 | `tests/test_registry_repair.py` | N | |
| 31 | `tests/test_switch_check.py` | N | |

`scripts/diagnostics/` and `tests/fixtures/e2e_grader_a4d96fe7.py.txt` do not exist on main. A plain `git add` of rows 10, 11, 12, 17 and 21 adds nothing and prints nothing about it, which is how #10 was lost. After staging, `git status --ignored` or `git ls-files` should show all five.

The record `READY_LOOP_GRADE_20261123.md:56` still names the old reference path under `data/`. It is a dated record and is left as written. This note records the move.

## docs/LEADS.md compared with main

`git diff abdf9f7 -- docs/LEADS.md` shows **one change: 73 added lines, nothing removed**. It is the drafted **L-9** (run H prediction pools against k/1000), appended after L-7's last line. Main did not change this file between `41ad509` and `abdf9f7`. Three things to settle when it lands:

1. **Numbering.** Main has L-1 to L-7 (`git show abdf9f7:docs/LEADS.md | grep '^## L-'`). L-8 exists only as an uncommitted draft in `~/tfm_worktrees/pwc-offset/docs/LEADS.md:656`. If e2e-grader lands first, main jumps from L-7 to L-9. Renumbering is the lead's or the owner's call, as the entry's own note says.
2. **Draft note.** The quoted note at the head of L-9 ("DRAFT, worktree `e2e-grader` only, not landed") will be wrong once the entry lands. I did not edit it; the decision is the lead's.
3. **Paths.** L-9 points to the grade outputs at `~/tfm_worktrees/e2e-grader/data/e2e_runH_grade/` and `…_followup/`. Those are outside git, so the next item matters.

## Other ignored or untracked files in this worktree

| Path | Ignored or untracked | Keep? |
|---|---|---|
| `data/e2e_runH_grade/` (6.3 MB) and `data/e2e_runH_grade_followup/` (11 MB) | untracked (`data/`) | **Yes.** These are the counted run H grade and the 14 follow-up grades, cited by L-9 and the two RESULT records. Keeping them means archiving them, for example to `~/tfm_worktrees/_records/`, at the owner's choice. Not git: `data/` is never tracked on main (`git ls-tree abdf9f7 data` is empty). |
| `data/switch_check_trial/` (18 MB): `classify_options.py`, `rr_head.md`, `rr_tail.md`, `rr_end.md`, `table20.md`, `item3.md`, `unregistered_174.txt`, plus the run folders `t1`, `t2`, `repair_run1`, `repair_run2` | untracked | **Yes for the generators** (about 100 KB): `REGISTRY_REPAIR_20261124.md` and the classes JSON are produced by them. The run folders can be regenerated. |
| `logs/e2e_grader/*.log` (300 KB) | ignored (`**/logs/`) | **Yes.** Every test claim in the branch's records cites one of them. Archive them with the run H outputs. |
| `data/e2e_grader_selftest/` (863 MB) | untracked | No. Self-test scratch, regenerated by every run. The reference grader once kept there is now fixture #26. |
| `scripts/diagnostics/__pycache__/` | ignored | No, and **not mine**. It holds `e2e_grader.cpython-310.pyc` (2026-11-23 20:44, reported before) and `compare_settings.cpython-310.pyc` (2026-11-24 17:44, the time of the lead's in-worktree suite run, `_review/grader_*.out`). Left in place for the lead. |
| `data/switch_check_trial/__pycache__/orig_41ad509_parameter_registry.py.cpython-310.pyc` (2026-11-23 19:04) | was untracked | Mine: written when I imported the original registry copy before I adopted `-B`. **Removed this task** by its exact path (`rm` then `rmdir`). It was in no pinned tree. |

No other ignored file in the worktree holds work.
