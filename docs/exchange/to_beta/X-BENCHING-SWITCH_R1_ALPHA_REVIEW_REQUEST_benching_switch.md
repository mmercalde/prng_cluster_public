# X-BENCHING-SWITCH  R1  REVIEW_REQUEST  benching package: no family benched by default, flags that WATCHER keeps in view


- **Review class:** BEFORE-COMMIT. The package changes:
  - WATCHER's post-Step-5 health check (`training_health_check.py`);
  - the Step 5 compare's family list (`meta_prediction_optimizer_anti_overfit.py`);
  - `watcher_policies.json`;
  - `.gitignore`;
  - WATCHER itself (`agents/watcher_agent.py`, 9 added lines, landed by hand by the owner);
  - one new module (`why_report_flags.py`).

  Owner rulings 2026-11-26 07:05-07:14 (`logs/e2e_session/OWNER_RULING_benching_switch_20261126.txt`): "yes - but put it
  in front of beta first". Your review is not binding. The owner decides.
- **Base commit:** `933a4a0` (`933a4a08ac38fa1800f025905019793eca7c9022`), main.
- **Main has moved since the package was built:** main is now `7f28109` (Gemini relay landing, five files: `scripts/gemini_relay.py`, `tests/test_gemini_relay.py`, `docs/exchange/GEMINI_RELAY_PROTOCOL.md`, `CLAUDE.md`, `AGENTS.md`). None of them is a file under review. The lead proves the package on a fresh copy of `7f28109` before asking the owner to land.
- **Rounds:** up to three (`RELAY_PROTOCOL.md` §2.5).
- **Allowed answers per question:** `NO OBJECTION` · `OBJECTION` · `SUGGESTION` · `NEEDS EVIDENCE`.
- **Please keep to real-world cases on this system as it is today:**
  - VM 101, with WATCHER run by the owner under `nohup`;
  - `watcher_policies.json` with this package (`benching: off`, `expiry_mode` absent, `grading_rules` absent);
  - the tracked skip registry (`neural_net` at 1);
  - `diagnostics_outputs/` as WATCHER's cwd sees it.
- **Label every objection `LIVE` or `THEORETICAL`** (`AGENTS.md` §5). A theoretical case is welcome, but please say it
  is one.

**This package flips a default, by ruling.** Before it, 3 consecutive critical grades benched a family: `SKIP_MODEL`,
and the next compare left the family out. After it, at the default `benching: off`, no family is benched:
- the count is kept;
- every check at or over the limit raises a why-report flag;
- WATCHER lists the open flags at the start of every run and notifies the owner once per new flag.

`benching: on` keeps the old skip. The owner gives his explicit yes to the default change at landing.

**The files are NOT on main.** Read them in
`~/.local/state/api-agents/worktrees/20261126-074830-pkg-benching-g2-watcher-933a4a0/<path>`.

## Prior record

**1. Searches run** (this worktree at `933a4a0`, 2026-11-26, using `/bin/grep` so that `.json` files are not skipped).
The scope list follows `CLAUDE.md` "Prior record first":
```
S="docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_* docs/TB_RULING_REQUEST_* docs/PROPOSAL_* docs/CHAPTER_* \
   docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md docs/exchange/"
/bin/grep -nEi 'model_skip|max_consecutive_critical|SKIP_MODEL|why-report|why_report|WHY-1|benching|benched|skip registry' $S   # (A)
/bin/grep -rnE 'grade_scope|all_families|worst_family|G-2\b|grade set by|CD-06' $S                                         # (B)
/bin/grep -rlI -i 'why_report_flag\|why-report flag\|why_report_needed\|notify_telegram\|flag reader' $S docs/drafts/loop_v1/
/bin/grep -rnEi 'telegram|cluster_notify' $S
/bin/grep -rnEi 'unit W\b|LL-8|why-report.*WATCHER|WATCHER.*why-report' $S docs/drafts
/bin/grep -rnE 'repository_tree_clean|gitignore' $S
git check-ignore -v diagnostics_outputs/why_report_flags.jsonl diagnostics_outputs/training_health_log.jsonl ; git ls-files diagnostics_outputs
```
The full hit lists are in the three READY notes in `docs/drafts/loop_v1/`: `READY_BENCHING_SWITCH_20261126.md` (A),
`READY_G2_GRADE_SCOPE_FIX_20261126.md` (B) and `READY_BENCHING_PACKAGE_20261126.md` (the package), each under
"Prior record". Hits on "benchmark", and the unrelated "G-2" gap label in `CHAPTER_2_SOURCE_MAP_v1.md:485`, were read
and ruled out.

**2. Documents found that decide, authorise, schedule or record the subject:**
- **The owner rulings:**
  - `logs/e2e_session/OWNER_RULING_benching_switch_20261126.txt`: 07:05-07:07 benching, 07:10 the look-out, 07:14 G-2;
  - the 2026-11-24 10:47-10:50 ruling, quoted at `docs/drafts/loop_v1/READY_LAND_CH14_DIAG_20261125.md:167-184`;
  - `CLAUDE.md` "The learning loop" item 3; `AGENTS.md:54`.
- **TODO-WHY-1:** `docs/LEADS.md:885-897`. The automatic why-report is deferred. Until it exists, the lead runs the
  cycle by hand.
- **The hours-mode defect:** `docs/drafts/loop_v1/READY_LOOP_DIAG_2_20261123.md:229`.
- **Your earlier positions:** `docs/exchange/from_beta/X-CH14-DIAG_R1_BETA_REVIEW.md:97` (count three);
  `X-CH14-DIAG_R3_BETA_REVIEW.md:109-117` (CD-09, the explicit error for a bad value, activation left to the owner).
- **The removal path as audited:** `docs/CHAPTER_14_AUDIT.md:163`. The registry history:
  `docs/CHAPTER_14_TRAINING_DIAGNOSTICS.md:2010-2016`.
- **CD-06 `grade_scope`:** `docs/drafts/ch14_v1/PLAN_CH14_DIAGNOSTICS.md:173` and `:302`. These are outside the required
  list. They do not say whose evidence a promoted grade carries.
- **Telegram:**
  - `docs/CHAPTER_12_WATCHER_AGENT.md:788` (escalation sends CRITICAL);
  - `docs/PROJECT_FILE_CATALOG.md:505` (`TELEGRAM_NOTIFICATION_SYSTEM_REFERENCE.md`);
  - notification classes A/B/C at `agents/watcher_agent.py:136-147`.
- **The clean-tree refusal:** `docs/PROJECT_FILE_CATALOG.md:166`; the `.gitignore` ruling block, `:814-825`.
- **The hand-landed set:** `scripts/owner_land.py:86-87`.

**3. Labels:**

| Item | Label |
|---|---|
| Benching switch, default `off`, the flag | ALREADY RECORDED (rulings of 11-24 and 11-26); the build is NEW |
| Hours-expiry restart | defect ALREADY RECORDED; the fix is NEW |
| G-2 fix | NEW (owner yes 07:14) |
| `flag_id`, `critical_since_known`, `.gitignore` entries | NEW |
| Flag reader and attach record; WATCHER hook | NEW (ruled in principle 07:10) |
| Automatic why-report (unit W / TODO-WHY-1) | ALREADY RECORDED, deferred, **not built** |
| CHANGES A RULING | none |

**4. Not searched:**
- the host outside the worktree;
- ser8 archives;
- the private repository;
- project files outside the repository: the LL-7/LL-8 plan with its unit W, and the text of
  `TELEGRAM_NOTIFICATION_SYSTEM_REFERENCE.md`.

`logs/e2e_session/` is read from the main checkout.

## Files under review

The list with each file's sha256 is in `docs/drafts/loop_v1/MANIFEST_BENCHING_PACKAGE_20261126.txt`.

| # | Path | N/M | Note |
|---|---|---|---|
| 1 | `training_health_check.py` | M +161 −27 | switch, hours fix, G-2, flag with `flag_id` |
| 2 | `meta_prediction_optimizer_anti_overfit.py` | M +19 −2 | **Step 5 compare reader** `:162-198` only |
| 3 | `watcher_policies.json` | M +2 −1 | `benching: "off"` |
| 4 | `.gitignore` | M +5 | three runtime records `:126-130` |
| 5 | `why_report_flags.py` | N 266 | reader, attach, WATCHER hooks, CLI |
| 6 | `agents/watcher_agent.py` | M +9, **hand-landed** | diff `DRAFT_WATCHER_WHY_FLAGS_20261126.diff` |
| 7-12 | `tests/test_benching_switch.py`, `bench_switch_mutants.py`, `test_g2_grade_scope_fix.py`, `test_why_report_flags.py`, `why_flags_mutants.py`, `test_ch14_diag_wave2.py`, `ch14_wave2_mutants.py` | N / M | tests and mutants |

## What was run (CPU only, suites one at a time, logs `data/pkg/logs/final/`)

| Suite | Result |
|---|---|
| benching suite / mutants | 26 passed / clean + 17 of 17 killed |
| G-2 suite | 19 passed |
| reader + WATCHER suite / mutants | 13 passed / clean + 10 of 10 killed |
| ch14 wave1 / wave2 | 37 / 37 passed; mutants 25/25 / 36/36 |
| land-defaults single / compare | 86/86 gates, 9/9 mutants / 29/29 gates, 2/2 mutants |
| `switch_check.py` | rc 0, stale 0, `benching` row S02121; 5 new CLI options of `why_report_flags.py` "not registered" |

## Known limits (stated so you do not have to find them)

1. **A flag stays closed once a why-report is attached**, even if the same run of criticals continues. Only WATCHER's
   log says "repeat".
2. **The flag, attach and notified files are uncapped.**
3. **The CD-11 health log has the same `.jsonl` exposure.** It is off by default and has not been changed.
4. **The tracked `model_skip_registry.json`** is rewritten on every critical, at base as well. Not changed.
5. **The reader's CLI options are not registered.** It is an operator tool, not a step script.
6. **No automatic why-report** (unit W).

**Lead decision on known limit 1 (2026-11-26):** it changes after this round. A new flag will be raised when the family reaches another `max_consecutive_critical` criticals after the attached report, so a family that keeps failing cannot go quiet. Not in the files you are reading.

## Questions

**Q1. No family is benched by default.** At `benching` absent or `off`, can any path still drop a family from the
compare, or return `SKIP_MODEL`, because of consecutive criticals? The paths:
- the legacy single, multi and fixed paths of `check_training_health`;
- `is_model_skipped`; `get_skip_status`;
- the compare reader, including its import-failure fallback.

**Q2. `benching: on` equals `933a4a0`** under `until_reset`: actions, registry contents, family list and `--status`.

**Q3. The hours restart.** Is the restart at `training_health_check.py:1106-1112` right, and is it complete?

**Q4. G-2 together with the flag.** Under `grade_scope: all_families`, a promoted family's result carries:
- its own report, issues, fixes and metrics (`:843-871`);
- the flag, raised for the promoted family, not for the winner (`test_g2_promotion_plus_why_flag`).

Is that the right family to flag, and is anything still reported from the wrong family?

**Q5. Flag identity.** `flag_id` is the family plus `critical_since`. If that is unknown, it is the first `flagged_at`,
kept in the registry entry (`:1119-1123`, `:1161`, `:1177`). The id is cleared by a reset (`:1220`) and by the hours
restart (`:1111`). `critical_since_known` is False when the count predates the record. Does a realistic sequence give
either of these:
- two ids for one run of criticals;
- one id for two runs?

Examples: a hand reset, a switch flipped mid-run, a corrupt registry.

**Q6. The WATCHER hook.**
- Does the hand-landed diff keep Steps 5 and 6 unchanged? The suite compares `run_pipeline(5, 6)` with the WATCHER at
  `933a4a0` and finds the same `run_step` calls, params, notifications, stdout, log and return value with no flags;
  `run_step` and `_build_retry_params` are source-identical.
- Can a reader or notification failure stop the pipeline? Can one be lost silently?
- Is one notification per new `flag_id`, plus one reminder per run, the right volume for a human?

**Q7. The tree stays clean.** With the `.gitignore` entries, can the first flag, attach or notification still dirty the
tree and trip `scripts/gate12_cleantree_gate.py`?

**Q8. Tests.** Does each credited gate fail when it should, and can any pass vacuously? Does any test write outside its
`--basetemp` or `--out`?

Please label every objection `LIVE` or `THEORETICAL`, as `AGENTS.md` §5 requires.
