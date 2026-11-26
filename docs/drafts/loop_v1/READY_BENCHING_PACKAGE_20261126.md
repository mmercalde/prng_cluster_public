# READY · benching package: benching switch, hours-expiry fix, G-2 fix, why-report flag reader, WATCHER hook (2026-11-26)

> **After Beta R1 (2026-11-26):** this package was moved onto main `7f28109` and given one fix pass after Beta's R1
> review. See the section **After Beta R1** near the end; it supersedes the line numbers, counts and test results
> above it where they differ. Fix-pass worktree:
> `~/.local/state/api-agents/worktrees/20261126-090310-fixpass-benching-7f28109`, detached at `7f28109`.

Worktree `~/.local/state/api-agents/worktrees/20261126-074830-pkg-benching-g2-watcher-933a4a0`, detached at main `933a4a0`
(`933a4a08ac38fa1800f025905019793eca7c9022`). **Prepared, not landed.** Nothing was committed or pushed. No pipeline was
launched. CPU only: no GPU, no rig, no ssh. Beta goes first: the draft request
`X-BENCHING-SWITCH_R1_ALPHA_REVIEW_REQUEST_benching_switch.md` is in this folder and **has not been sent**. The owner
lands the package. `agents/watcher_agent.py` is landed **by hand** from `DRAFT_WATCHER_WHY_FLAGS_20261126.diff`.

The package combines:
- **A**, the benching switch with its hours fix (`READY_BENCHING_SWITCH_20261126.md`, this folder);
- **B**, the G-2 fix (`READY_G2_GRADE_SCOPE_FIX_20261126.md`, this folder);
- the lead's review fixes (a)-(d);
- the WATCHER flag reader.

The owner's rulings are in `logs/e2e_session/OWNER_RULING_benching_switch_20261126.txt`: 07:05-07:10 for benching and the
look-out, 07:14 for G-2.

## Prior record

**1. Searches run** (this worktree, 2026-11-26, using `/bin/grep` so that `.json` files are not skipped):
```
S="docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_* docs/TB_RULING_REQUEST_* docs/PROPOSAL_* docs/CHAPTER_* \
   docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md docs/exchange/"
/bin/grep -rlI -i 'why_report_flag\|why-report flag\|why_report_needed' $S docs/drafts/loop_v1/
/bin/grep -rlI -i 'benching' $S docs/drafts/loop_v1/
/bin/grep -rlI -i 'TODO-WHY-1' $S docs/drafts/loop_v1/
/bin/grep -rlI -i 'grade_scope.*all_families\|G-2' $S docs/drafts/loop_v1/
/bin/grep -rlI -i 'notify_telegram' $S ; /bin/grep -rnEi 'telegram|cluster_notify' $S
/bin/grep -rnEi 'unit W\b|LL-8|why-report.*WATCHER|WATCHER.*why-report' $S docs/drafts
/bin/grep -rnE 'repository_tree_clean|gitignore' $S
git check-ignore -v diagnostics_outputs/why_report_flags.jsonl diagnostics_outputs/training_health_log.jsonl
git ls-files diagnostics_outputs
```
A's and B's own searches are in their READY notes ("Prior record") and are not repeated here.

**2. Documents found** that decide, authorise, schedule or record the subject:
- `logs/e2e_session/OWNER_RULING_benching_switch_20261126.txt`. Benching off by default; flag the family for a human or
  AI to check; Beta before landing (07:05-07:07). "something needs to look out for this - something in the pipeline"
  (07:10). Stage 1 is WATCHER, stage 2 is TODO-WHY-1 (07:10). Yes to G-2 (07:14).
- The 2026-11-24 10:47-10:50 ruling, quoted at `docs/drafts/loop_v1/READY_LAND_CH14_DIAG_20261125.md:167-184`. Also
  `CLAUDE.md` "The learning loop" item 3, and `AGENTS.md:54`.
- `docs/LEADS.md:885-897`: TODO-WHY-1. "Until TODO-WHY-1's trigger and queue are built, the lead runs the attribution
  cycle by hand."
- `docs/drafts/loop_v1/READY_LOOP_DIAG_2_20261123.md:229`: the hours-mode defect.
- `docs/exchange/from_beta/X-CH14-DIAG_R1_BETA_REVIEW.md` and `_R3_BETA_REVIEW.md`: Beta's earlier positions on CD-09 and
  benching (cited in A's request).
- `docs/CHAPTER_12_WATCHER_AGENT.md:788` and `docs/PROJECT_FILE_CATALOG.md:505`: Telegram as human attention routing
  (`TELEGRAM_NOTIFICATION_SYSTEM_REFERENCE.md`). The notification classes are documented at
  `agents/watcher_agent.py:136-147` (CRITICAL / DEGRADED / INFO).
- `docs/PROJECT_FILE_CATALOG.md:166` and `:814-825`: the clean-tree refusal (`repository_tree_clean is False`) and the
  ruling-bearing `.gitignore` block.
- `scripts/owner_land.py:86-87`: `agents/watcher_agent.py` is hand-landed.
- The LL-7/LL-8 plan's unit W: named in the brief. **Not read** (it is outside this worktree; see 4).

**3. Labels**

| Item | Label |
|---|---|
| Benching switch, default `off`, flag file | ALREADY RECORDED (rulings of 11-24 and 11-26 07:05-07:07); the build is A's |
| Hours-expiry restart | defect ALREADY RECORDED (`READY_LOOP_DIAG_2:229`); the fix is NEW (A) |
| G-2 fix | NEW; owner yes 07:14 (B) |
| `.gitignore` entries for the three records | NEW |
| `flag_id`, `critical_since_known`, `why_flag_first_at` | NEW |
| Flag reader and attach record (`why_report_flags.py`) | NEW; ruled in principle at 07:10 |
| WATCHER hook: list at run start, one reminder per run, one notification per new `flag_id` | NEW; ruled in principle at 07:10 |
| Automatic why-report run | ALREADY RECORDED as TODO-WHY-1 / unit W; **not built** |
| CHANGES A RULING | none |

**4. Not searched:** the host outside this worktree, ser8 archives, the private repository, and project files outside the
repository. That includes the LL-7/LL-8 plan (unit W) and `TELEGRAM_NOTIFICATION_SYSTEM_REFERENCE.md` itself.

## What changed (file:line in this worktree)

**`training_health_check.py`** (+161 −27)
- **From A:**
  - the switch: `:114-126`, `benching_mode` `:149`, and in `_wave2_settings` `:173`;
  - `skip_active` off branch `:963`;
  - `_severity_to_action` / `_check_skip_registry` now return a 3-tuple `:1075-1147`;
  - the hours restart `:1106-1112`;
  - `_attach_skip_fields` `:1149`;
  - `_raise_why_flag` `:1167`;
  - `get_skip_status` `:1419`; the CLI `--status` text `:1716`.
- **From B (G-2), in `_check_training_health_fixed`:**
  - `family_reports` `:815`;
  - `winner_name, promoted` `:835`;
  - the promoted family's own report, grade, issues, fixes and metrics `:843-855`;
  - `winner_model_type`, `grade_set_by` and `grade_note` `:869-871`.
  - A's `_attach_skip_fields` follows at `:872`, so the result carries both changes.
- **Review fix (b), `critical_since_known`:** `:1184`. It is False when the count began before `critical_since` was
  recorded. `critical_since` stays None in that case.
- **Review fix (c), `flag_id`:** `why_flag_id` `:1161`, set at `:1177`. It is the model family plus `critical_since`. If
  that is unknown, it is the family plus `why_flag_first_at`, the first `flagged_at` of this run of criticals, kept in the
  registry entry (`:1119-1123`).
  - `reset_skip_registry` clears both keys (`:1220`).
  - The hours restart also clears them (`:1111`). This happens only when off has run, so the `on` registry is unchanged.
  - A flag line is still appended on every critical at or over the limit (unchanged).

**`meta_prediction_optimizer_anti_overfit.py`** (+19 −2, A): the compare reader `:162-198`.

**`watcher_policies.json`** (+2 −1, A): `benching: "off"` `:168`.

**`.gitignore`** (+5), review fix (a): `:126-130`. It ignores `why_report_flags.jsonl`, `why_report_attachments.jsonl` and
`why_report_notified.jsonl` under `diagnostics_outputs/`.

**`why_report_flags.py`** (new, 266 lines):
- `read_flags` `:89` groups by `flag_id`. It derives an id for older lines, and counts unreadable lines instead of hiding
  them.
- `open_flags` `:128`. A flag is open until a why-report is attached.
- `attach` `:133` is append-only. It refuses an unknown id, a missing report file, an empty `--by`, or a second report.
- `summary_lines` `:154`.
- The WATCHER hooks are `watcher_run_start` `:175` and `watcher_on_health` `:196`. Each catches every exception, logs it
  at ERROR with its traceback, and returns.
- CLI `list [--all] [--json]` and `attach FLAG_ID REPORT_PATH --by NAME [--note]` `:227`.

**`agents/watcher_agent.py`** (+9, **hand-landed**, `DRAFT_WATCHER_WHY_FLAGS_20261126.diff`):
- the guarded import `:115-120`. If the import fails, both hooks become a function that logs ERROR at every call;
- the call at run start `:3035`, straight after "Starting pipeline";
- the call in `_handle_training_health` `:2581`.

**Tests and tools:**
- `tests/test_benching_switch.py`: 18 tests from A, plus 8 for review fixes (b)-(d).
- `tests/bench_switch_mutants.py`: A's 10 mutants (M06's anchor moved), plus M11-M17.
- `tests/test_g2_grade_scope_fix.py`, from B. One change: the base-equality registry comparison ignores the benching keys
  `critical_since` and `why_flag_first_at`. The benching suite tests those keys.
- `tests/test_why_report_flags.py` (new, 13 tests) and `tests/why_flags_mutants.py` (new, 10 mutants).
- A's edits to `tests/test_ch14_diag_wave2.py` and `tests/ch14_wave2_mutants.py`.

## The default change, in plain words

Before this package, a model family graded critical 3 times in a row was benched: WATCHER got `SKIP_MODEL` and the next
compare left the family out. After it, at the default `benching: off`:
- the family is never benched;
- the health check answers `RETRY`, and WATCHER retries Step 5 at most `max_retries` times, then goes on to Step 6;
- every check from the 3rd critical on appends a flag line;
- WATCHER sends the owner one notification the first time a `flag_id` appears.

At the start of every pipeline run, WATCHER prints and logs the open flags above everything else and sends one reminder,
until a why-report is attached with `python3 why_report_flags.py attach`. `benching: on` restores the old skip. Under
`until_reset` it is identical to `933a4a0` (tested). Under `hours` it now restarts the count after a skip expires.

## What is unchanged

- **WATCHER's Step 5 and Step 6 argv.** `run_step`, `_build_retry_params` and `_get_max_training_retries` are
  source-identical to `933a4a0` (`test_argv_builders_unchanged`).
- **WATCHER with no flags.** `run_pipeline(5, 6)` makes the same `run_step` calls with the same params, sends the same
  notifications, and produces the same stdout, log and return value as the WATCHER at `933a4a0`. This holds on a PROCEED
  path and on a RETRY×3-then-PROCEED path (`test_no_flags_identical_to_base`). Mutant WM8 shows the test notices one
  extra blank line.
- **Legacy grading and all four G-2 base-equality modes** are identical except for the benching keys (B's suite).
- **The CH14-DIAG landing defaults:** `test_ch14_diag_land_defaults.py` passes all gates (below).

## Hit rate: prediction and settling measurement (not measured)

*Corrected after Beta R1, additional finding 1 (`docs/exchange/from_beta/X-BENCHING-SWITCH_R1_BETA_REVIEW.md`,
"Additional finding"). The first version said pools could change only through the compare's family list. That was
wrong: it left out the Step 5 retry path.*

- **What changes, two ways.**
  1. *Which families the Step 5 compare trains.* With benching off, a family at or over the limit stays in the list
     (`meta_prediction_optimizer_anti_overfit.py:184-206`).
  2. *Which Step 5 attempt supplies Step 6's model.* At the limit the base returned `SKIP_MODEL`
     (`git show 7f28109:training_health_check.py`, `_check_skip_registry`). With benching off the action is `RETRY`
     (`training_health_check.py:1158-1162`). So WATCHER can run one more Step 5 training attempt, within its retry limit
     (`agents/watcher_agent.py:3196-3259`, `watcher_policies.json:131-135`), and Step 6 can then use that attempt's model
     instead of the earlier attempt's.
- **Consequence:** pools can differ even when the same family wins the compare, because Step 6 may use a different
  attempt's model.
- **No direction is inferred.** Nothing here says whether hit rate goes up or down.
- **Today:** the tracked registry holds `neural_net` at 1 (`diagnostics_outputs/model_skip_registry.json`). Two more
  critical checks would take it to 3 with one retry already used. From there the two arms part: the base advances to
  Step 6 on `SKIP_MODEL`, and benching off can retry once more.
- **Not affected:** G-2, the reader and the WATCHER hook change reporting only. They cannot move pools. One exception
  is the R1 F3 re-flag: it raises a flag and a notification only, and the action stays `RETRY` either way.
- **Measurement that would settle it:** a paired walk-forward on back draws (`scripts/diagnostics/e2e_walk_forward.py`).
  - Arms: `benching: off` and `on`, each on its own actual retry path.
  - Per run, record which Step 5 attempt supplied Step 6's model.
  - Score pool hits against `k/1000` on draws the sieve never saw.
- [UNVERIFIED] whether the driver can set a policy key per arm and record the supplying attempt.

## Test results

VM 101, CPU only, run sequentially by `data/pkg/run_all.sh` with nothing else running. Logs are in
`data/pkg/logs/final/`.

| Suite | Result |
|---|---|
| `tests/test_benching_switch.py` | **26 passed** |
| `tests/bench_switch_mutants.py` | **18/18**: the clean copy passes; 17/17 mutants killed on their credited tag |
| `tests/test_g2_grade_scope_fix.py` | **19 passed** |
| `tests/test_why_report_flags.py` | **13 passed** |
| `tests/why_flags_mutants.py` | **11/11**: the clean copy passes; 10/10 killed |
| `tests/test_ch14_diag_wave1.py` / `wave2.py` | **37 / 37 passed** |
| `tests/ch14_wave1_mutants.py` / `ch14_wave2_mutants.py` | **25/25 / 36/36 killed** |
| `test_ch14_diag_land_defaults.py --part single` (log inside `--out`) | **86/86 gates, 9/9 mutants** |
| `test_ch14_diag_land_defaults.py --part compare` (log inside `--out`) | **29/29 gates, 2/2 mutants** |
| `scripts/diagnostics/switch_check.py` (`--commit 933a4a0`, this worktree linked) | rc 0; stale specs 0; `model_skip_rules.benching` row S02121 (added, default `off`); **new switches for the patch 5**: the CLI options of `why_report_flags.py` (S02129-S02135), class "not registered" |

The mutants the brief asked for:
- WM1 removes the open-flag listing → `WSTART` red.
- WM2 removes the new-flag notify → `WNEW` red. WM3 removes the run reminder → `WREMIND` red.
- WM4 makes a reader failure raise → `WFAIL` red.
- WM5 and WM6 remove the WATCHER calls.
- WM7 notifies every repeat.
- WM9 drops a `.gitignore` line, checked in a scratch repository.
- WM10 makes the reader ignore attachments.

Every test that needs a notifier uses a fake one (a list). No Telegram message was sent.

## Exposure of other records to the clean-tree gate (review fix (a), reported, not changed)

- **CD-11 `diagnostics_outputs/training_health_log.jsonl` has the same exposure.** `git check-ignore` does not match it
  (`.gitignore` ignores `*.json`, not `*.jsonl`). Its switch defaults to `off` (`training_health_check.py:107`), so
  nothing is written today. The first run with `training_health_log: on` would leave an untracked file, and the
  clean-tree gate would refuse the next Step 1. Not changed here.
- **Pre-existing, wider:** `diagnostics_outputs/model_skip_registry.json` is **tracked** (`git ls-files`). Every critical
  check rewrites it (`_check_skip_registry`), so the tree is dirty under any `benching` setting, including at `933a4a0`.
  This package adds keys to that file but does not create the exposure. Not changed.

## What I did not handle

1. **The automatic why-report** (TODO-WHY-1 / unit W). Not built.
2. ~~A flag attached while its run of criticals continues stays closed.~~ **Handled after Beta R1 (F3):** after
   another `max_consecutive_critical` criticals with no reset, a NEW flag is raised. See "After Beta R1".
3. **The flag file and the notified record are uncapped** (one line per critical at or over the limit).
4. **switch_check lists 5 new CLI options** of `why_report_flags.py` as "not registered". It is an operator CLI, not a
   step script WATCHER calls, so no `ParameterSpec` or manifest entry was made. The lead decides.
5. **Notification wording** uses the tag `[WATCHER][WHY-REPORT]`, not one of the documented classes
   (CRITICAL / DEGRADED / INFO). `TELEGRAM_NOTIFICATION_SYSTEM_REFERENCE.md` was not read.
6. **The two items in the exposure section above** (CD-11 log, tracked registry): reported only.

## After Beta R1 (fix pass, 2026-11-26)

Beta's review: `docs/exchange/from_beta/X-BENCHING-SWITCH_R1_BETA_REVIEW.md` (main `b4f95f2`). Its positions: Q1, Q2 and
Q7 NO OBJECTION; Q3 OBJECTION · THEORETICAL; Q4 OBJECTION · THEORETICAL; Q5 OBJECTION · LIVE; Q6 OBJECTION · LIVE; Q8
SUGGESTION; one additional finding. Fix pass in the worktree
`~/.local/state/api-agents/worktrees/20261126-090310-fixpass-benching-7f28109`, detached at main `7f28109`. Nothing was
committed or pushed. No pipeline was launched. CPU only. No real notification was sent.

**Rebase.** The 18 manifest files were copied unchanged from the package worktree; all 18 sha256 matched its MANIFEST
before copying. Main moved `933a4a0` → `7f28109` through five files (`AGENTS.md`, `CLAUDE.md`,
`docs/exchange/GEMINI_RELAY_PROTOCOL.md`, `scripts/gemini_relay.py`, `tests/test_gemini_relay.py`). None of them is in
the package. For each of the 7 modified files the blob at `933a4a0` equals the blob at `7f28109` (`git rev-parse
<rev>:<path>`), so nothing was merged by hand. Main has since moved to `b4f95f2`. That commit adds only exchange records
(Beta's review, the sent request, `DEFERRED.md`, `LEDGER.md`), none of which is a package file. The suites' base
revision is now `7f28109`.

| Beta | Position | Done | Where |
|---|---|---|---|
| Q1 default off | NO OBJECTION | nothing | — |
| Q2 `on` + `until_reset` | NO OBJECTION | nothing | — |
| Q3 hours fallback | OBJECTION · THEORETICAL | **fixed (F5)**, 9 lines: the import-failure fallback lets an expired timed skip go, keeps an active one, and keeps one whose `last_critical` is unreadable (as `skip_active`) | `meta_prediction_optimizer_anti_overfit.py:163,173-174,192-200`; test `tests/test_benching_switch.py:595` (expired / active / unreadable); mutant M20 |
| Q4 promoted provenance | OBJECTION · THEORETICAL | **fixed (F4)**: in the promoted case `report_path`, `report_source` (`all_families_grade_setter_report`) and `freshness` are the grade setter's own; the winner's are kept as `winner_report_path`, `winner_report_source`, `winner_freshness`; the CD-11 health log carries the same (`grade_set_by`, `winner_*`) | `training_health_check.py:872-880`, `:1032-1036`; test `tests/test_g2_grade_scope_fix.py:175` (Beta's `stale_report_action: grade` fixture); mutants `winner_report_path`, `winner_freshness`, `health_log_no_winner_fields` (`:280-291`) |
| Q4, retry LLM | (same) | **recorded, not fixed**: the retry LLM opens the canonical report independently (`agents/watcher_agent.py:2701-2734`); WATCHER is hand-landed and out of this pass's scope | — |
| Q5 interrupted write | OBJECTION · LIVE | **fixed (F1)**: `_write_skip_registry` writes a temp file in the same directory, fsyncs it, then `os.replace`. It is the only writer of `SKIP_REGISTRY_PATH`, used by the check and by `reset_skip_registry`. The temp name is git-ignored | `training_health_check.py:1185-1199`, `:1152`, `:1290`; `.gitignore:131-132`; test `tests/test_benching_switch.py:560` (json.dump interrupted part-way, check and reset); mutants M18, M19 |
| Q5, formula | (same) | **fixed (F1)**: the flag line records `why_flag_first_at`. `flag_id_of` uses the writer's chain, `critical_since` → `why_flag_first_at` → `flagged_at` | `training_health_check.py:1202`, `:1244`; `why_report_flags.py:102`; test `tests/test_why_report_flags.py:549` (known and unknown `critical_since`); mutants WM19, WM20 |
| Q6 failed send | OBJECTION · LIVE | **fixed (F2)**, details below | `why_report_flags.py:193-278`, `:313-337`; `agents/watcher_agent.py:2581,3035` |
| Q7 clean tree | NO OBJECTION | nothing | — |
| Q8 tests | SUGGESTION | **done (F6)**: (1) a WATCHER equality case whose third health result is healthy, asserting that it is consumed; (2) a `git ls-files` tracked-path assertion next to the check-ignore gate, with a tracked control; (3) every suite run with `PYTHONDONTWRITEBYTECODE=1` exported before Python starts, plus `-B`; (4) the snapshot helper asserts git's exit status | `tests/test_why_report_flags.py:349,366`, `:198-205`; `data/fixpass/run_all.sh`; `tests/test_benching_switch.py:113-115` |
| Additional finding 1 | NEW | **corrected (F7)**: the hit-rate section above now covers the Step 5 retry path | this file, "Hit rate" |
| Q6 cadence, re-flag (lead decision, announced to Beta) | — | **built (F3)**, details below | `training_health_check.py:1133-1140`, `:1160-1161`, `:1208-1225`, `:1246`; `why_report_flags.py:127,183-185,238-241` |

**F2, the sender (Beta Q6).**
- `why_report_flags.send_notification` (`:207`) runs the script named in `watcher_policies.json`
  `notifications.script_path`. The default is `/usr/local/bin/cluster_notify.sh`, the same as `notify_telegram`.
- It waits at most 15 s (`NOTIFY_TIMEOUT_S`, `:59`) and returns `(ok, detail)`. `ok` is true only on exit status 0. A
  spawn error, a non-zero exit, a timeout and `telegram_enabled: false` are all failures. It never raises.
- `_notify_new` (`:255`) appends one record per attempt to `why_report_notified.jsonl`, with `status: sent | failed`.
  "notification sent" is logged only on success. A failure is logged at ERROR as "notification NOT sent ... tried again
  at the next health check or run start".
- `_retry_failed` (`:270`) runs in both hooks. It retries every open flag whose attempts all failed.
- A record with no `status` (written before R1) counts as sent.
- WATCHER passes no callback now: `_why_on_health(health, logger)` `:2581` and `_why_run_start(logger)` `:3035`. The
  diff is still 9 added lines, and `notify_telegram` is not touched.
- **Limit, stated plainly.** The installed `cluster_notify.sh` exits 0 in every case: its header says "Silent on
  failure (exit 0 always)", and it ends with `exit 0` after `curl ... >/dev/null 2>&1` (`/usr/local/bin/cluster_notify.sh`,
  outside the repo). So exit 0 proves the script ran, **not that Telegram delivered**. This pass detects a missing or
  unrunnable script, a timeout, and a policy that has notifications off. It cannot see delivery without changing that
  script. The script is outside the repo and was not touched.
- **Critical path:** each attempt is synchronous. At most 15 s, and only when a flag is new or pending, plus the one
  reminder at run start when flags are open.
- Tests run the real sender with `why_report_flags.subprocess` replaced by a fake for every test (autouse
  `_no_real_notify`, `tests/test_why_report_flags.py:67-102`). Covered: spawn error, non-zero exit, timeout, success, the
  policy path and `telegram_enabled: false` (`:449`, `:460`); ERROR log, `failed` record, and a retry at the same flag's
  next check, at another health check, and at run start (`:478`); the same through WATCHER (`:509`). Mutants WM11-WM18.

**F3, re-flag after an attached why-report.**
- A segment of a run of criticals starts at `why_flag_base` (`training_health_check.py:1133-1140`).
- When the current flag has a why-report attached (`_why_flag_attached`, which reads `why_report_attachments.jsonl`,
  `:1208`) and the family is critical again, a new segment starts. That critical sets a new `critical_since`, and
  `why_flag_prev` is set to the attached flag.
- Inside the segment, criticals below `max_consecutive_critical` return `RETRY` with no flag (`:1160-1161`).
- At the limit, a NEW `flag_id` is raised with `previous_flag_id` (`:1246`). WATCHER sends one NEW notification that
  names the flag it follows (`why_report_flags.py:238-241`).
- A reset clears all four keys (`_WHY_FLAG_ENTRY_KEYS`, `:1181`). Under `benching: on` none of them is written.
- The count itself never resets on an attachment.
- Test: `tests/test_why_report_flags.py:566`. Flag, attach, two criticals → no flag and no notification; the third → a
  new open flag linked to the first, and one notification; the fourth → a repeat. A control without an attachment keeps
  one id. Mutants WM21-WM23.

**What this pass did not handle.**
- The retry LLM's independent report read (Q4; `agents/watcher_agent.py:2701-2734`).
- Delivery confirmation through `cluster_notify.sh` (F2 limit above).
- The repeated ERROR plus `failed` record at every health check while notifications are off or the script is missing.
  This is bounded by the number of checks, and the notified record is uncapped.
- Whether the walk-forward driver can set `benching` per arm and record the supplying attempt [UNVERIFIED].
- The draft request copy in this folder differs in its header from the sent copy landed at `b4f95f2`
  (`docs/exchange/to_beta/`). The lead may drop the draft from the landing list.

**Test results after the fix pass.** VM 101, CPU only, run one at a time by `data/fixpass/run_all.sh` with nothing else
running. Logs are in `data/fixpass/final/logs/`; the land-defaults logs are inside their `--out`.

| Suite | Result |
|---|---|
| `tests/test_benching_switch.py` | **30 passed** |
| `tests/bench_switch_mutants.py` | **21/21**: clean copy passes, 20/20 killed on their credited tag |
| `tests/test_g2_grade_scope_fix.py` | **23 passed** (7 in-suite mutants killed) |
| `tests/test_why_report_flags.py` | **32 passed** |
| `tests/why_flags_mutants.py` | **24/24**: clean copy passes, 23/23 killed on their credited tag |
| `tests/test_ch14_diag_wave1.py` / `wave2.py` | **37 / 37 passed** |
| `tests/ch14_wave1_mutants.py` / `ch14_wave2_mutants.py` | **25/25 / 36/36 killed** |
| `test_ch14_diag_land_defaults.py --part single` | **86/86 gates, 9/9 mutants**; peak RSS 637 MB (`/usr/bin/time -v`) |
| `test_ch14_diag_land_defaults.py --part compare` | **29/29 gates, 2/2 mutants**; peak RSS 588 MB |
| `tests/test_gemini_relay.py` (a script; `pytest` collects nothing from it, rc 5) | **33/33 PASS, `COMPLETION SENTINEL: PASS`** run as `python3 -B -u` |
| `scripts/diagnostics/switch_check.py --commit 7f28109` | rc 0; 0 stale specs; `model_skip_rules.benching` row S02123 (added, `off`); the `why_report_flags.py` CLI options S02131-S02137 "not registered" (operator CLI, as before) |

WATCHER diff: 9 added lines, 0 removed; `DRAFT_WATCHER_WHY_FLAGS_20261126.diff` (34 lines) is byte-identical to `git diff
7f28109 -- agents/watcher_agent.py`.
