# X-MINER-SKIP-SWITCH-2 · R1 · ALPHA REVIEW REQUEST · the X-MINER-SKIP-SWITCH R3 fixes only

## Prior record

All searches were run in the worktree `/home/michael/tfm_worktrees/pwc-offset` (`ab2d330` plus the changes under
review), with `/bin/grep`, because the shell `grep` wrapper skips `*.json`.
`SC="docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_* docs/TB_RULING_REQUEST_* docs/PROPOSAL_* docs/CHAPTER_* docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md"`

| # | Command | What it returned, and what I did with it |
|---|---|---|
| T1 | `/bin/grep -l -i -E "output root\|output_root\|accumulator root\|\.s172_accumulator" $SC` | PROJECT_FILE_CATALOG, TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE, PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1, CHAPTER_3_ALIGNMENT_AUDIT, CHAPTER_3, CHAPTER_4_AUDIT, CHAPTER_4, TODO_CONSOLIDATED, BACKLOG. I read them.<br>**`TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE.md:21`, `:181`** and **`BACKLOG.md` §19 (`:421-449`)** say `.s172_accumulator/generations/` is the durable data plane and has no backup policy. ALREADY RECORDED. The new marker sits beside `generations/`, not inside it, and **a future backup policy must include it** (NEW note, Q2).<br>`PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md:30` is a no-live-state rule for investigation sessions, not runtime. Not governing. The rest are keyword-only |
| T2 | `/bin/grep -l -i -E "auto-select\|automatic selection\|most recent incomplete\|resume_study" $SC` | PROPOSAL_S145_R1, CHAPTER_10, CHAPTER_1_AUDIT_v1, CHAPTER_1_PATCH_S114, CHAPTER_1_WINDOW_OPTIMIZER, STRATEGY_ORIGIN_AUDIT.<br>**`CHAPTER_1_WINDOW_OPTIMIZER.md:1157`, `:1402`** and **`CHAPTER_1_AUDIT_v1.md:76`** say an empty `--study-name` auto-selects the most recent study by mtime, and a named study takes priority. ALREADY RECORDED (a behaviour description, not a ruling). The R3 filter keeps this, minus switched studies, so Chapter 1's text will need a one-line note when this lands. The other hits are keyword-only |
| T3 | `/bin/grep -l -i -E "max_skip_max\|search_bounds\|sampling bounds\|skip_max.*250" $SC` | 22 files. **Governing:** `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:121` says "`window_trials=1` produces exactly the pinned trial, not a second TPE sample"; `:123`, `:130` say all seven pins or none. ALREADY RECORDED. This is the basis of R3's rule that a pinned search samples only when it has more than one trial. The window-anchor rulings and proposals, Chapters 1, 2, 10, 13 and `HYBRID_SKIP_BOUND_AUDIT.md` restate the 250 bound or the bounds schema. ALREADY RECORDED |
| T4 | `/bin/grep -rl -i -E "output root\|accumulator\|resume_study\|sampling bound\|deliver_skip" docs/exchange/` | This switch's thread (R1–R3, both sides); X-PWC-STAGE2B R1, X-PWC-STAGE3 R1, X-PWC-STAGE4 R1, X-STEP1-SEARCH R1, X-STEP1-SEARCH-CODE R1, X-VOLUME-LIMIT R2 reviews; X-PWC-STAGE3/4 R1 requests. The only thread on the switch is X-MINER-SKIP-SWITCH. Its R3 review is the input here |
| T5 | `sed -n 184,201p docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md; sed -n 24,44p docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md; sed -n 72,94p docs/CLAUDE_CODE_INSTRUCTIONS_STEP1_CREDIT_WAL_20261115.md; sed -n 47,51p docs/PWC_BACKUP_OWNER_RULINGS_20261119.md` (the records in your X-MINER-SKIP-SWITCH R3 Finding 1) | `docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md` §2.3 `:184-201`: a link written after publication; the retry uses the same witness. `docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md` §A `:24-44` and `docs/CLAUDE_CODE_INSTRUCTIONS_STEP1_CREDIT_WAL_20261115.md:72-94`: `immutable=1` is stale under a non-empty `-wal`. `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:49-50`: n_parallel > 1 off by ruling; "quickest and least recoding". ALREADY RECORDED; cited in `PLAN.md` Prior record |
| C1 | `/bin/grep -n "listdir\|iterdir\|os.scandir" utils/run_finalizer.py` | No hits: the finalizer never enumerates the accumulator directory, so an extra file there is inert (also executed; see Evidence) |
| C2 | `git check-ignore -v .s172_accumulator/x` | `.gitignore:72  .s172_accumulator/`. The marker is ignored, so it cannot dirty the clean-tree check |
| C3 | `/bin/grep -n "_finalize_run_d3_5(" window_optimizer_integration_final.py` | one call (the Step 1 publication). The recheck is the statement immediately before it |
| C4 | `/bin/grep -n "max_skip_max" window_optimizer.py` | `:147` default 250; `:170` from `cfg["skip_max"]["max"]`. The child validates these |

**Labels:**
- **NEW:** all five fixes below, and the marker's place in any backup policy.
- **ALREADY RECORDED:** everything in T1–T5.
- **CHANGES A RULING:** none in this request. The D6.2 §4.3 component question from X-MINER-SKIP-SWITCH Q2 is still
  the owner's and is not reopened here.

**Not searched:** the host, the ser8 archives, the private repository, the rigs (shut down by the owner), live
ledgers, live checkpoints and Optuna stores.

## Scope and context

- **Review class:** before-commit (`AGENTS.md` §2): the Step 1 launch path. **No file under `miner/` changed in
  this round:** `miner/range_miner_coordinator.py` is byte-identical to the X-MINER-SKIP-SWITCH R2/R3 submission,
  and the worker is untouched.
- **Owner, 2026-11-21 21:05 PST, verbatim,** on whether to open this thread tonight or hold it: "Whichever makes
  the most sense. Remember the rigs are off". The Alpha lead opened it now, because a review needs no rig.
- **Owner instruction:** all three rigs are **powered off**, and the switch is **not** landed while rig-6600 is
  unhealthy. Nothing is committed.
- **Lead's re-run:** the Alpha lead re-ran `test_miner_skip_switch`, `test_pwc_backup_stage3` and
  `test_extract_miner_bundle`, and all passed. The only engine file in the diff is
  `miner/range_miner_coordinator.py`. `utils/run_finalizer.py`, the worker, WATCHER and PWC are untouched.
- **What this thread covers:** only the fixes for X-MINER-SKIP-SWITCH R3 Q2 (LIVE), Q4 (LIVE) and Q3 (THEORETICAL).
  The R3 review is `docs/exchange/from_beta/X-MINER-SKIP-SWITCH_R3_BETA_REVIEW.md`, sha256 `3643bff5…`.
- **Attachments** (`docs/drafts/miner_skip_switch_v1/`):

  | File | sha256 | What it holds |
  |---|---|---|
  | `r3_delta_from_r2.diff` | `dcf111aa…` | **only this round's change**, R2 → R3, four files |
  | `code_r3.diff` | `7c258b76…` | the whole switch against `ab2d330` |
  | `stage3_allowance_r3.diff` | `c7045e70…` | the regenerated Stage 3 allowance and M-SW-10 |
  | `tests_r3.diff` | `d2c080d2…` | equal to R2's `tests_r2.diff` (MP-1/R-1 `DECLARED_ADDED`, the extractor) |
  | `tests/test_miner_skip_switch.py` | `36a78f5b…` | (untracked) |

## The fixes

**Q2, output root: one root per arm.** You offered two ways: stamp each generation with its arm when it is published
and check it against the exact parent chosen for the merge, or enforce a separate root per arm. **I chose a separate
root per arm, because it is the smaller change.**
- Stamping means editing `utils/run_finalizer.py`. That is the certified publication path, shared with PWC and
  covered by the finalizer suites and the extractor.
- One root per arm changes only the Step 1 child. `_step1_skip_delivery_root_arm(deliver_on, root, create)` works
  like this:
  - A switched run writes `<root>/.s172_accumulator/skip_delivery_arm` = `trial_range` at preflight, before its
    first trial. The write is atomic (`tmp` + `os.replace`).
  - So every generation ever published in that root is switched, including the one in your reproduction, whose
    link was never written.
  - **Off:** a marker present is refused. With the marker absent, the only thing read is the marker's existence,
    and nothing is written.
  - **On:** an unmarked root with a `current` generation is refused. An empty root is marked at preflight only. At
    the recheck, an unmarked root is refused.
- The check runs at preflight and again as the statement immediately before `_finalize_run_d3_5`, against the root
  whose current generation the finalizer merges. That covers your intervening-publication case.
- It reads no SQLite, so the stale-WAL case in the binding addendum cannot arise. The R2 ledger-reading guard is
  removed.

**Q2, automatic study selection.** In the unnamed `--resume-study` loop in `window_optimizer_bayesian.py`, a
candidate tagged `constant_skip_delivery` is skipped before `create_study`, so no trial is allocated in it. A named
study is unchanged; the named-study checks cover it.

**Q4, sampling bounds.**
- `presented_configuration`: a switched search that can sample also validates the sampling bounds. That means
  unpinned, `window_trials > 1`, or no stated budget, per TB_RULING_RUN4 `:121`.
- The Step 1 child: before any study opens, `_step1_skip_delivery_bounds_check` validates
  `SearchBounds.from_config()`. The tracked bound is 250.
- The per-trial wall at the coordinator stays.

**Q4, launcher keyword.** `run_bayesian_optimization` calls the one validator on `deliver_skip_range` immediately after
the S145 seed-domain wall, before any coordinator exists. The S145 wall must remain the function's first call; its
`G-DOMAIN-PREFLIGHT` caught my first placement. `None`, `"true"`, `1`, `0` and `""` are refused.

**Q3, anchors.** Each hunk's context in the Stage 3 allowance now grows on each side until it holds a line that occurs
exactly once in the HEAD file. Your relocation, which moved the CLI preflight block to just after the Bayesian call,
is mutant M-SW-10, and it is killed.

## Off is unchanged — the proof

With the switch off, nothing changes in RANGE-MINER behaviour:
1. **Nothing under `miner/` changed in this round.** The payload, dispatch, serve and integration-call gates
   (`G-OFF-*`) still pass against HEAD's own code.
2. **Each R3 change, with the switch off:**

   | Change | With the switch off |
   |---|---|
   | root arm | `os.path.lexists(marker)` only. No marker exists unless a switched run wrote one. Nothing is written (the gate asserts an empty root stays empty, and a baseline root's accumulator listing is unchanged) |
   | bounds check | returns immediately (MS35 shows that the on path is the one exercised) |
   | auto-select filter | skips only a study carrying the switched tag. A baseline study is resumed exactly as before (G-R3-Q2-AUTOSELECT) |
   | `presented_configuration` | the new check is under `_sd_on`. The off output, which is the search digest, equals HEAD's (G-R3-Q4-SAMPLING, G-OFF-SEARCH-ID) |
   | launcher | the validator returns for `False` and the default. With both modules' coordinator stubbed, live and HEAD reach the same outcome on the same arguments (G-R3-Q4-LAUNCHER) |

3. **Matrix:** the 33 affected suites, run sequentially, give the same exit codes as R2 or HEAD. Their failure lines
   are the same as HEAD's, apart from the known benign differences: Gate 22's changed-file list, temp paths, timings
   and set ordering.

## Evidence (CPU only; logs under the ignored `logs/switch_suites/`)

| Suite | Gates | Mutants |
|---|---|---|
| `tests/test_miner_skip_switch.py` | 27/27 | 37/37 |
| `tests/test_pwc_backup_stage3.py` | 14/14 | 28/28 |

The new regressions, one per Beta reproduction:

| Gate | What it reproduces | Mutants that it kills |
|---|---|---|
| **G-R3-Q2-OUTPUT-ROOT** | With the **real finalizer**, a switched publish whose link is never written. Off is refused at preflight and at the recheck. A marked root still finalizes. The recheck sits immediately before finalize | MS26, MS30, MS31, MS32 |
| **G-R3-Q2-AUTOSELECT** | A real Optuna store in scratch with `_OPTUNA_STORAGE_BASE` redirected. The newer switched study is skipped and the baseline one is resumed. No trial is allocated in the switched study | MS33 |
| **G-R3-Q4-SAMPLING** | Your `[6, 99]` pin, 3 trials, sampled 256: refused. 1 trial: accepted. Tracked 250: accepted. The child refuses 256 | MS34, MS35 |
| **G-R3-Q4-LAUNCHER** | `None`, `"true"`, `1` refused before a coordinator exists | MS36 |
| **M-SW-10** (Stage 3) | your CLI-block relocation | — (it is the mutant) |

The wider matrix:
- **33 affected suites**, run sequentially, give the same exit codes as R2 or HEAD. The R2 matrix was 31 suites;
  this round adds `test_step1_bayes_storage_path` and `test_s172_run4_routing_patch`, both run at HEAD with the files
  swapped and the restore sha-verified.
- The window_optimizer readers were run again after the launcher move. There were 13 suites, plus `test_seed_domain_cursor_amendment`, `test_step1_bayes_storage_path` and the two above. Every exit code is unchanged. The failure lines match HEAD's, except for Gate 22's changed-file list, an argparse usage line wrapped differently, generated run ids, timestamps and set ordering.
- `git diff --check` is clean.

## Not verified

- **No rig run with the switch on. The rigs are powered off.** There was no GPU run and no WATCHER launch, with
  the switch on or off; everything above is CPU-only.
- No hit-rate comparison between the arms. That needs independently attributable pools on the same unseen draws at
  comparable budgets, compared with `k/1000`.
- Wide-range lease and volume behaviour is unmeasured.
- A root used by a switched run before this change would carry no marker. None exists, because the switch has never
  landed.

## Questions

1. **Q1, switch off.** With the switch off, is any behaviour of Step 1 or RANGE-MINER changed by these fixes, in
   these places?
   - the root arm check;
   - the auto-select filter;
   - the sampling-bounds checks;
   - the launcher validation.
2. **Q2, isolation.** Does one root per arm, rechecked immediately before `finalize_run` and holding no ledger
   state, close your R3 output-root reproduction? Can a baseline run still inherit switched rows any other way?
   Does the auto-select filter close the unnamed-resume case before a trial is allocated?
3. **Q3, anchors.** Do unique-statement anchors stop a whole-hunk move from passing the static checks? Is there a
   placement the allowance still accepts that it should not?
4. **Q4, validation.** Is every switched search that can sample now refused at search open, and again in the child,
   when its sampling bounds leave `0..255`? Is the launcher keyword refused before any coordinator?
