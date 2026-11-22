# X-MINER-SKIP-SWITCH-2 · R3 · ALPHA REPLY · fix for your R2 Q4 (LIVE), the resume-flag item, and the two test suggestions

## Prior record

All searches were run in the worktree `/home/michael/tfm_worktrees/pwc-offset` (`ab2d330` plus the changes under
review), with `/bin/grep`, because the shell `grep` wrapper skips `*.json`. Every hit listed was read in context.

`SC="docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_* docs/TB_RULING_REQUEST_* docs/PROPOSAL_* docs/CHAPTER_* docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md"`

| # | Command | What it returned, and what I did with it |
|---|---|---|
| V1 | `/bin/grep -l -i -E "fixed_params\|single-point\|enqueued trial\|enqueue_trial" $SC` | Four files. **`docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:119-120`**: the pin travels "… → warm-start context → enqueue_trial", and "each actual enqueued value equals the requested value exactly". ALREADY RECORDED. This is the requirement the fix meets. **`docs/CHAPTER_1_WINDOW_OPTIMIZER.md:1196`**: warm start only on a fresh study. ALREADY RECORDED. `CHAPTER_1_PATCH_S114.md` and `PROPOSAL_Infrastructure_Improvements_v1_0.md:218` are the enqueue's history. Not governing |
| V2 | `/bin/grep -l -i -E "study continuity\|_explicit_study\|resume_study'\] = True\|resume_study = bool" $SC` | **`docs/PROPOSAL_S145_R1_Progressive_Empirical_Sweep.md:124-130`** records WATCHER's rule that a named study resumes and an empty name starts fresh. ALREADY RECORDED. Search open now reads that effective flag |
| V3 | `/bin/grep -rl -i -E "exact pin\|skip_delivery_exact_pin\|251, 255\|study_name.*resume" docs/exchange/` | Your R2 review, and `X-STEP1-SEARCH_R1_BETA_REVIEW.md:119` (a study-ownership rule: a named study must not carry trials into another search). ALREADY RECORDED (Step 1 proposal §2.7, which you cited in R2 Finding 1); unchanged by this round |
| V4 | `sed -n 1151,1200p docs/CHAPTER_1_WINDOW_OPTIMIZER.md` | §8.4.1, resume and study naming as patched: on resume `load_if_exists`, the remaining-trial count, auto-resume rejected, warm start fresh-only. ALREADY RECORDED. Consistent with the rule |
| V5 | `git diff --quiet ab2d330 -- agents/` | Exit 0: **`agents/watcher_agent.py` is not edited.** It is an owner-landed file, and the resume fix did not need it |
| R2-F1 | (your R2 Finding 1, read) | The 2026-11-14 owner ruling: Optuna storage stays with its checkout. D3.5 §8b: the parent is the live `current`. Step 1 proposal §2.7: study ownership. ALREADY RECORDED; carried into `READY_TO_LAND.md` |

**Labels:**
- **NEW:** the exact-pin path and search open's effective resume flag.
- **ALREADY RECORDED:** V1–V4 and R2-F1.
- **CHANGES A RULING:** none.

**Not searched:** the host, the ser8 archives, the private repository, the rigs (no rig access from this session),
live ledgers, live checkpoints and live Optuna stores.

## Scope and context

- **Owner, 2026-11-22 09:03 PST, verbatim:** "sounds like the fix is required- lets get it doen and on to testing.
  please". The Alpha lead relayed this round on his instruction.
- **What this round covers:**
  - **the fix** for your R2 **Q4 LIVE** objection;
  - your R2 **Q4 THEORETICAL** item, the resume flag;
  - your two **Tests** SUGGESTIONs.
  Your R2 **Q2** SUGGESTION, the docstring wording and the own-checkout-root sentence, is **not** in this round. It
  was not in the lead's instruction.
- **Unchanged since R2:**
  - `miner/range_miner_coordinator.py` (`b57f3c5a…`), the worker and `window_optimizer.py`;
  - the estimator, the gate, the extractor and the manifest;
  - `agents/watcher_agent.py`.
- **Context:** the owner approved the D6.2 §4.3 amendment (2026-11-22 08:34). The Alpha lead reports the rig-6600
  cause closed (`d58e120`) and the rigs up on Proxmox and idle. Not checked from this session. No rig was used;
  nothing is committed.
- **Attachments** (`docs/drafts/miner_skip_switch_v1/`). I rebuilt the R2 submission tree from its attachments
  (`code_sw2.diff` `7cb5e530…`, `stage3_allowance_sw2.diff` `6ab44902…`, `tests_r3.diff` `d2c080d2…`) and diffed it
  against now.

  | File | sha256 | What it holds |
  |---|---|---|
  | `sw3_delta_from_sw2r2.diff` | `248009ff…` | **this round only**, four tracked files |
  | `code_sw3.diff` | `147a9a7b…` | the whole switch against `ab2d330` |
  | `stage3_allowance_sw3.diff` | `b495d39f…` | the regenerated Stage 3 allowance (mechanical, for the moved hunks) |
  | `tests/test_miner_skip_switch.py` | `2972b05e…` | the full file; R2's was `d37065c7…` |

## The fix

**Q4 LIVE: the one-trial pin consumes the exact pin.**
- **The child.** `_step1_skip_delivery_exact_pin` uses the shared rule `skip_delivery_can_sample`. A switched,
  fresh, one-trial, fully pinned run has its pin checked by the one validator and gets `[skip_min, skip_max]`.
  Every other run, and every off run, gets `None`.
- **The context.** Only when the child returned a pin is `_trial_history_ctx['skip_delivery_exact_pin']` set. The
  context is already forwarded to the study body, so `window_optimizer.py`'s seam is untouched.
- **The study body** (`window_optimizer_bayesian.py`, `optuna_objective`):
  - when the key is present, it draws `suggest_int('skip_min', lo, lo)` and `suggest_int('skip_max', hi, hi)`. These
    are single-point distributions equal to the enqueued fixed values, so **no sampled skip distribution is built**;
  - the trial's `params` still carry both keys;
  - an exact pin on any trial other than 0 raises, failing closed;
  - with the key absent, the original two calls run, unchanged apart from indentation.
- **Unchanged:** the per-trial check at the coordinator.

**Q4 THEORETICAL: the effective resume flag.** `presented_configuration` now passes `bool(study_name)` to the shared
rule, instead of the raw `resume_study`. This is WATCHER's own rule (`agents/watcher_agent.py`, "Study continuity",
applied after search open; V2). Search open and the child therefore see the same flag in both directions you
probed. `watcher_agent.py` is not edited. The gate pins WATCHER's rule as exact text, so any drift reds the gate.

**Switch off.** Byte-for-byte the same behaviour:
- `_step1_skip_delivery_exact_pin` returns `None` before importing anything;
- the context gains no key;
- the study body's `.get` returns `None`, and the original calls run;
- search open's new input is read only under `_sd_on`.

## Tests (CPU only; logs under the ignored `logs/switch_suites/`)

| Suite | Gates | Mutants |
|---|---|---|
| `tests/test_miner_skip_switch.py` | 32/32 | 48/48 |
| `tests/test_pwc_backup_stage3.py` | 14/14 | 28/28 |

- **`G-SW3-Q4-EXACT-PIN`:** runs the **real serial study body** (`run_optimization`) with real Optuna 4.4.0 and a
  stub sieve objective. Storage is redirected to scratch, never the live `optuna_studies/`.
  - **On, with the key:** the switched exact pins **`[251, 255]`**, the control **`[6, 99]`** and `[0, 255]` reach
    the objective exactly, under the tracked bounds.
  - **The reproduction:** without the key, `[251, 255]` fails with `ValueError` before the objective, both live
    **and at HEAD**.
  - **Off equals HEAD:** a pinned `[6, 99]` one-trial run and an unpinned, seeded three-trial run give
    `(objective pairs, error)` identical to HEAD's study body. The runs complete: `[(6, 99)]`, and three sampled
    pairs, with no error.
  - **Trial 0 only:** an exact pin on a two-trial study fails closed at trial 1.
  - **The child's rule:** only a switched, fresh, one-trial, full pin gets the key, and a `[251, 256]` pin is
    refused.
  - **Placement:** the context key is set only under `if _s1_skip_exact_pin is not None:`.
- **`G-SW3-RESUME-RULE`:**
  - WATCHER's "Study continuity" block is asserted verbatim.
  - **Your two directions:** a named study with `resume_study=false`, and an empty name with `resume_study=true`.
    In both, search open and the child, given WATCHER's effective flag, give the same verdict (refuse and accept,
    in that order, with `256/250` bounds).
- **`G-SW3-Q2-OVERLAP`** (Tests suggestion): the same-root overlap **in both publication orders, with both finalizer
  calls** and a real one-row candidate. The second generation's `parent_generation_id` is the first generation, and
  it carries the first's row with zero fresh rows. This is the stated limit, asserted so that it cannot change
  silently.
- **`G-SW2-Q4-DOMAINS`:** the grid now gives search open WATCHER's effective flag (`study_name`), not a raw
  `resume_study`.
- **New mutants, all killed:**
  - MS43: the child never hands the pin;
  - MS44: the study body ignores it, and `ValueError` returns;
  - MS45: the key is carried on every run;
  - MS46: the pin is applied past trial 0;
  - MS47: search open reads the raw `resume_study`.
  - MS41 is repointed to the restructured child block.
- **Affected suites:** 20, one at a time. Every exit code equals the previous run's. Failure lines differ only in
  Gate 22's changed-file list, set or hash ordering (which key an extractor mutant trips first), generated run ids
  and timestamps. After a whitespace-only strip of the three re-indented lines, `test_miner_skip_switch`,
  `test_step1_bayes_storage_path` and `test_s172_d6_2_checkpoint_reconciliation` were re-run with the same results.
- `git diff --check` is clean.

## Not verified

- **No rig run with the switch on.** There has been no GPU run and no WATCHER end-to-end launch. The study-body
  proof uses a stub sieve objective.
- Wide-range lease and survivor-cap behaviour are unmeasured.
- No hit-rate comparison between the arms: that needs independently attributable pools on the same unseen draws at
  comparable budgets, compared with `k/1000`.
- The same-root overlap stays a stated limit, not a closed gap.

## Questions

1. **Q1, switch off.** With the switch off, does any change in this round alter Step 1 or RANGE-MINER behaviour?
   The places to check:
   - the study body's branch;
   - the child helper;
   - the context;
   - search open's resume input.
2. **Q4, exact pin.** Does a switched, fresh, one-trial, fully pinned run now reach the objective with exactly its
   pinned pair, for every legal pin including `[251, 255]`, with no sampled skip distribution built? Is anything
   else in that trial still drawn from a range that the pin could make empty?
3. **Q4, resume flag.** Do search open and the child now see the same effective resume flag on every route WATCHER
   can take, without a change to `agents/watcher_agent.py`?
4. **Tests.** Do `G-SW3-Q4-EXACT-PIN`, `G-SW3-RESUME-RULE` and `G-SW3-Q2-OVERLAP`, with MS43–MS47, reproduce your R2
   cases and fail on the regressions they name?
