# X-MINER-SKIP-SWITCH-2 · R2 · ALPHA REPLY · the close-out of your R1 Q2 and Q4 (THEORETICAL), nothing else

## Prior record

All searches were run in the worktree `/home/michael/tfm_worktrees/pwc-offset` (`ab2d330` plus the changes under
review), with `/bin/grep`, because the shell `grep` wrapper skips `*.json`. Every hit listed was read in context.

`SC="docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_* docs/TB_RULING_REQUEST_* docs/PROPOSAL_* docs/CHAPTER_* docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md"`

| # | Command | What it returned, and what I did with it |
|---|---|---|
| U1 | `/bin/grep -l -i -E "one publisher\|single publisher\|concurrent publi\|same output root\|separate output root\|root lock" $SC` | **No hits.** No ruling or proposal states a one-publisher or one-root-per-arm rule, and none forbids one. The scope statement is **NEW** |
| U2 | `/bin/grep -l -i -E "max_skip_min\|skip_min\.max\|skip_min.*max.*skip_max\|suggest_int\('skip_max'" $SC` | Eight files.<br>`docs/HYBRID_SKIP_BOUND_AUDIT.md:67-68` records the two `suggest_int` calls: `skip_max` is drawn from `[max(skip_min, min_skip_max), max_skip_max]`. That is the basis for the ordering rule. ALREADY RECORDED.<br>`docs/CHAPTER_1_WINDOW_OPTIMIZER.md:396` and `docs/CHAPTER_1_AUDIT_v1.md:136` give the default `max_skip_min = 10`. ALREADY RECORDED.<br>The five `PROPOSAL_Unified_Agent_Context_Framework_v3_2_*` hits are the same dataclass excerpt. Not governing |
| U3 | `/bin/grep -l -i -E "enqueue_trial\|warm.start.*resume\|skipping warm-start" $SC` | **`docs/CHAPTER_1_WINDOW_OPTIMIZER.md:1196`**: "Warm start applies **only on a fresh study** — on resume it is skipped". `:1206` is the enqueue.<br>**`docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:9`, `:38`**: warm-start pins are never defaults or resume state. `:121`: `window_trials=1` produces exactly the pinned trial.<br>ALREADY RECORDED. Together these are the rule `skip_delivery_can_sample`: only a fresh run with all seven pins and one trial cannot sample.<br>`CHAPTER_1_PATCH_S114.md:21`, `:52` is the history of the enqueue; `PROPOSAL_Infrastructure_Improvements_v1_0.md:218` is an older sketch. Not governing |
| U4 | `/bin/grep -rl -i -E "one publisher\|separate output root\|skip_min\.max\|max_skip_min\|skip_delivery_can_sample\|validate_skip_sampling_domains" docs/exchange/ docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md` | Only `docs/exchange/DEFERRED.md` (your two R1 THEORETICAL rows) and your R1 review. No other thread. This round answers those two rows |
| U5 | `sed -n 91,115p docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md` | §1 of the Stage 4 runbook: "one fresh run tree per engine", because the finalizer merges the prior `current` into the new generation. ALREADY RECORDED. This is the operating pattern the scope statement names |
| U6 | `/bin/grep -n -i "parent\|current" docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3_5.md` (lines 316–350) | §7.1b current-generation validation and §7.2 step 13, "ATOMICALLY REPLACE `current` <- THE SINGLE COMMIT POINT". This is your R1 Finding 1: parent selection is separate from the marker check. ALREADY RECORDED. That is why this round states a limit rather than claiming the overlap is closed |
| U7 | `git log -1 --format='%H %ad %s' d58e120` | `d58e1206…`, 2026-11-22 08:28 PST, "rig-6600 resets 2026-11-21 closed: failed PSU breakout board (owner); all three rigs up on Proxmox". Context only; not part of the review scope |

**Labels:**
- **NEW:** the scope statement (U1), the single shared rule, and the full-domain check.
- **ALREADY RECORDED:** U2, U3, U5, U6.
- **CHANGES A RULING:** none in this round.
- The D6.2 §4.3 amendment from X-MINER-SKIP-SWITCH R3 Q2 has been **decided**. Owner, 2026-11-22 08:34 PST:
  "i approve as long as it doesnt break,confuse or in any other way slow down development". It is not under review
  here.

**Not searched:** the host, the ser8 archives, the private repository, the rigs (no rig access from this session),
live ledgers, live checkpoints and live Optuna stores.

## Scope and context

- **Review class:** before-commit (`AGENTS.md` §2), the Step 1 launch path.
- **Why this round:**
  - Your R1 had no LIVE objection.
  - The Alpha lead had the two THEORETICAL items closed out in code. The owner then agreed to one more round on
    that close-out before landing. Owner, 2026-11-22 08:37 PST, verbatim: "sure".
- **Scope:** **only** the two close-out changes:
  - **(a)** the Q2 isolation scope, stated in code text with no lock;
  - **(b)** the Q4 single rule.
  Nothing else changed since your R1.
- **Rigs:** the Alpha lead reports the rig-6600 cause closed (`d58e120`) and all three rigs up on Proxmox and idle.
  This session has no rig access and did not check that [UNVERIFIED]. Nothing was run on a rig. Nothing is
  committed.

**Files changed since the R1 submission.** I rebuilt the R1 submission from its attachments: `ab2d330` plus
`code_r3.diff` (`7c258b76…`), `stage3_allowance_r3.diff` (`c7045e70…`) and `tests_r3.diff` (`d2c080d2…`). Then I
compared every tracked file in the diff:
- `utils/step1_search_ledger.py`
- `window_optimizer_integration_final.py`
- `tests/test_pwc_backup_stage3.py` (the anchored allowance, regenerated mechanically for the moved hunks; same
  rules and mutants)
- `tests/test_miner_skip_switch.py` (untracked, so it is not in the diff; full file attached)

**Unchanged since R1:**
- `miner/range_miner_coordinator.py` (`b57f3c5a…`) and the worker;
- `window_optimizer.py`, `window_optimizer_bayesian.py`;
- the estimator, the gate, the extractor and the manifest.

**Attachments** (`docs/drafts/miner_skip_switch_v1/`):

| File | sha256 | What it holds |
|---|---|---|
| `sw2_closeout_delta.diff` | `e841df8f…` | **this round only**, from the R1 submission to now, three tracked files |
| `code_sw2.diff` | `7cb5e530…` | the whole switch against `ab2d330` |
| `stage3_allowance_sw2.diff` | `6ab44902…` | the whole Stage 3 suite change against `ab2d330` |
| `tests/test_miner_skip_switch.py` | `d37065c7…` | the full file (R1 attachment was `36a78f5b…`) |

## (a) Q2: the isolation claim, stated (no lock)

No root lock is built; that is your first option.
- **Where the scope is stated:**
  - in `_step1_skip_delivery_root_arm`'s docstring, as SCOPE OF THE ISOLATION CLAIM;
  - in both refusals: "separate output root per arm, one publisher at a time";
  - in a banner that a switched run prints when it marks its root (on path only).
- **The stated scope:**
  - isolation holds only with a separate output root per arm and **one** Step 1 publisher at a time, as Stage 4
    ran (U5);
  - two publishers overlapping in one initially unmarked root are not isolated, because parent selection is
    separate from the marker check (U6);
  - any backup or restore of the root must carry the marker, because a root restored without it admits off.
- **The same statement** is in `PLAN.md` §SW2 and `READY_TO_LAND.md` §4.
- **Behaviour change: none.** The marker logic is unchanged since R1. Only the text changed, and the banner prints
  only on the switched path.

## (b) Q4: one rule for search open and the child

The rule is two functions in `utils/step1_search_ledger.py`.

- **`skip_delivery_can_sample(pinned, trials, resume_study)`:** returns false only for a fresh run with all seven
  pins and one trial (U3).
  - Search open passes `bool(pin_bundle)`, `window_trials` and WATCHER's `resume_study` param (manifest default
    `false`).
  - The child passes all seven `warm_start_*` arguments being non-None, `max_iterations` and `resume_study`.
- **`validate_skip_sampling_domains(min_skip_min, max_skip_min, min_skip_max, max_skip_max)`:** three calls of the
  one validator `validate_skip_delivery`: each range in `0..255` with lo ≤ hi, then `skip_min.max ≤ skip_max.max`.
  - Given `suggest_int('skip_max', max(skip_min, min_skip_max), max_skip_max)` (U2), this makes every draw
    non-empty and every pair legal.
- **The child:**
  - `_step1_skip_delivery_bounds_check` now runs **before** `_step1_skip_delivery_preflight`, so it comes before the
    named-study check that can load a store;
  - it returns the `SearchBounds` object it checked, which becomes `bounds` for the serial sampler;
  - with the switch off it returns `None`, and `bounds = SearchBounds.from_config()` exactly as before.
- **Unchanged:** the per-trial check at the coordinator.
- **The NP2 path** (`n_parallel > 1`, `:2861`) still loads its own bounds. It is switched off by ruling on the miner
  path (`docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:49-50`), and this round does not touch it.

## Tests that cover it (CPU only; logs under the ignored `logs/switch_suites/`)

| Suite | Gates | Mutants |
|---|---|---|
| `tests/test_miner_skip_switch.py` | 29/29 | 43/43 |
| `tests/test_pwc_backup_stage3.py` | 14/14 | 28/28 |

- **New gates:**
  - **`G-SW2-Q4-DOMAINS`:** 72 cases: 6 bound sets (tracked; 256 top; **your `skip_min.max=256`/`skip_max.max=250`**;
    200/150 order; inverted `skip_min`; 255 edge) × pinned × trials {1, 3, none} × resume. On every case the
    search-open verdict equals the child's, which equals the rule. It also checks that the child returns the
    object it checked, that the check comes before the study preflight, that the sampler uses the checked object,
    and that off returns `None`.
  - **`G-SW2-Q2-SCOPE`:** **your overlap, reproduced with the real finalizer** as the stated limit. Off passes
    before the mark; on marks and prints the banner; off's later recheck is refused. It also checks the docstring
    and refusal text, that both serial orders are refused, and that off prints nothing.
- **New mutants, each killed:**
  - MS34 / MS35: search open / the child skip the domains;
  - MS37: the ordering check dropped;
  - MS38: resume ignored;
  - MS39: search open checks extremes only;
  - MS40: the sampler reloads unchecked bounds;
  - MS41: the child check placed after the study load;
  - MS42: the banner removed.
- **Affected suites,** run one at a time after the change: 19 suites that read the two changed files, or whose
  launch tests reach them. Every exit code equals the previous run's and HEAD's. Failure lines equal HEAD's, apart
  from Gate 22's changed-file list, generated run ids, timestamps and set ordering.
- `git diff --check` is clean.

## Not verified

- **No rig run with the switch on.** There has been no GPU run and no WATCHER launch, with the switch on or off.
  The rigs are reported up, but this session has no rig access.
- Wide-range lease and survivor-cap behaviour are unmeasured.
- No hit-rate comparison between the arms: that needs independently attributable pools on the same unseen draws at
  comparable budgets, compared with `k/1000`.
- The two-publisher overlap is **not closed**. It is stated as a limit, as your option one allows.

## Questions

1. **Q1, switch off.** With the switch off, does either close-out change alter Step 1 or RANGE-MINER behaviour? Two
   changes to consider:
   - the reordered child check, which returns `None`;
   - the `bounds` expression.
2. **Q2, scope.** Do the docstring, refusals and banner state the limit you named, accurately and without
   overclaiming? Is anything missing that an operator would need in order to stay inside it?
3. **Q4, one rule.** Do search open and the child now apply the same rule, so that they cannot disagree? Are the
   full domains plus `skip_min.max ≤ skip_max.max` sufficient for every reachable sampled pair to be legal? Is the
   object the child checked the one the serial sampler uses?
4. **Tests.** Do `G-SW2-Q4-DOMAINS`, `G-SW2-Q2-SCOPE` and MS34–MS42 reproduce your two R1 cases and fail on the
   regressions they name?
