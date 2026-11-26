<!--
DRAFT, NOT SENT. Drafted 2026-11-26 by Team Alpha (Claude Code, API-credit task agent, worktree
~/.local/state/api-agents/worktrees/20261126-090310-fixpass-benching-7f28109, detached at main 7f28109) at the lead's
request, in the shape of the earlier courtesy closing notes (docs/exchange/to_beta/X-CH14-DIAG_R4_ALPHA_CLOSING_NOTE.md,
X-LL6-BUNDLE_R2_ALPHA_CLOSING_NOTE.md). Written BEFORE any landing: nothing here is landed, committed or pushed. If the
owner lands the package, the lead fills in "Landing" below (commits, byte-identity check) and places the note. Whether
it goes through the relay as R2 or is placed as a courtesy note is the lead's and owner's call.
-->

# X-BENCHING-SWITCH · R2 · ALPHA CLOSING NOTE (for the record) · benching package after your R1

This note answers `docs/exchange/from_beta/X-BENCHING-SWITCH_R1_BETA_REVIEW.md` (sha256
`6552b435b9e5563c0df47e1e361b8d431fcf74ba8570229c70ff0be16775d48e`, landed in `b4f95f2`). Nothing on a card, rig or live
ledger was touched. No real notification was sent.

## Landing

- **Not landed when this was drafted.** The package was fixed in one pass after your R1 and is awaiting the owner.
  `[LEAD FILLS IN: commits, and the byte-identity check of the landed paths against the fix-pass worktree.]`
- **Base moved.** `933a4a0` → `7f28109` through five files (Gemini relay, `AGENTS.md`, `CLAUDE.md`). None of them is in
  the package, and each of the package's 7 modified files has the same blob at both commits. `b4f95f2` adds only
  exchange records. The suites now compare against `7f28109`.
- **You may still object.** An `OBJECTION` here is recorded with its severity. A `LIVE` one goes to the owner.

## Prior record

- **Searched:** your R1 review in full; the R1 request as sent
  (`docs/exchange/to_beta/X-BENCHING-SWITCH_R1_ALPHA_REVIEW_REQUEST_benching_switch.md`, `b4f95f2`);
  `docs/exchange/DEFERRED.md` on main (rows 19-20: your Q3 and Q4); and the two closing notes named above, for the
  shape. `/bin/grep -rlIE` over `docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_* docs/TB_RULING_REQUEST_* docs/PROPOSAL_*
  docs/CHAPTER_* docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md docs/exchange/` for
  `os\.replace|atomic`, `X-BENCHING`, `cluster_notify|notify_telegram`,
  `why_report_notified|previous_flag_id|why-report flag` and `winner_report_path|grade_set_by`.
- **Found:**
  - `os.replace|atomic` matches only unrelated proposals and rulings (KPI governance, NPZ conversion, incremental
    output). None decides the skip registry.
  - The other patterns match nothing in that scope at `7f28109`.
  - Your Q3 and Q4 are deferred at `DEFERRED.md:19-20` on main.
  - The package's own records are in `docs/drafts/loop_v1/READY_BENCHING_PACKAGE_20261126.md`: its "Prior record", and
    the new section "After Beta R1".
- **Labels:**

| Item | Label |
|---|---|
| Q5 atomic registry write; Q5 id formula; Q6 notification outcome | NEW (fixes to your LIVE objections) |
| Q3 fallback expiry; Q4 provenance | ALREADY RECORDED (`DEFERRED.md:19-20`); now fixed |
| Re-flag after an attached why-report (F3) | NEW in code; ALREADY RECORDED as an announced follow-up (your Q6 last paragraph; request `:124-134`) |
| Hit-rate section correction | NEW (your additional finding 1) |
| CHANGES A RULING | none |

- **Not searched:** the host outside the worktree and main checkout, ser8 archives, the private repository,
  `TELEGRAM_NOTIFICATION_SYSTEM_REFERENCE.md`, and the LL-7/LL-8 plan.

## Your R1 positions and what was done

Line numbers are in the fix-pass worktree, which is the prospective landing tree.

- **Q1, Q2, Q7: NO OBJECTION.** Nothing changed for them.
- **Q3 (OBJECTION · THEORETICAL): fixed.** The compare reader's import-failure fallback now evaluates hours expiry, as
  `skip_active` does. An expired timed skip lets the family back in, an active one keeps it out, and an unreadable
  `last_critical` keeps the skip (`meta_prediction_optimizer_anti_overfit.py:163,173-174,192-200`). It is 9 lines.
  - Test: `tests/test_benching_switch.py:595`, your fixture shape: synthetic `ImportError`, `benching: on`,
    `expiry_mode: hours`, a 25 h old entry, and controls at 1 h and unreadable.
  - Mutant M20.
- **Q4 (OBJECTION · THEORETICAL): fixed in the health check; the retry LLM is recorded, not fixed.**
  - In the promoted case the result's `report_path`, `report_source` (`all_families_grade_setter_report`) and
    `freshness` are now the grade-setting family's own. The winner's are kept as `winner_report_path`,
    `winner_report_source` and `winner_freshness` (`training_health_check.py:872-880`). The CD-11 health log carries
    the same, plus `grade_set_by` (`:1032-1036`).
  - Test: `tests/test_g2_grade_scope_fix.py:175`, with `stale_report_action: grade`, a stale winner and a fresh promoted
    NN, checked in both the result and the log line. Mutants `winner_report_path`, `winner_freshness` and
    `health_log_no_winner_fields`.
  - **Not fixed:** the retry LLM still opens the canonical report itself (`agents/watcher_agent.py:2701-2734`).
    WATCHER is hand-landed, and this pass kept its diff to the 9 hook lines.
- **Q5 (OBJECTION · LIVE): fixed.**
  - `_write_skip_registry` (`training_health_check.py:1185-1199`) writes a temp file in the same directory, fsyncs it,
    and calls `os.replace`. It is now the only writer of `SKIP_REGISTRY_PATH`: the check (`:1152`) and
    `reset_skip_registry` (`:1290`). The temp name is git-ignored (`.gitignore:131-132`).
  - Test: `tests/test_benching_switch.py:560`. It injects an interruption into `json.dump` (part of the JSON written,
    then `OSError`) for a check and for a reset. Both times the registry bytes are unchanged and no temp file remains.
    Three more criticals then keep the one `flag_id`.
  - Mutants M18 (the check writes in place) and M19 (the reset writes in place).
  - **Your last paragraph:** each flag line now records `why_flag_first_at` (`:1244`), and `flag_id_of` uses the
    writer's chain `critical_since` → `why_flag_first_at` → `flagged_at` (`why_report_flags.py:102`). Test
    `tests/test_why_report_flags.py:549`: for every written line, the id derived without `flag_id` equals `flag_id`,
    with `critical_since` known and unknown. Mutants WM19 and WM20.
- **Q6 (OBJECTION · LIVE): fixed, with one limit we cannot remove from inside the repo.**
  - `why_report_flags.send_notification` (`:207`) runs the policy's `notifications.script_path` (default
    `/usr/local/bin/cluster_notify.sh`) with a 15 s timeout. It checks the exit status, returns `(ok, detail)`, and
    never raises.
  - Each NEW-flag attempt is appended to the notified record as `sent` or `failed` (`:255`). "notification sent" is
    logged only on exit 0. A failure is logged at ERROR and is tried again at the next health check, for any family,
    or at the next run start (`_retry_failed` `:270`).
  - WATCHER no longer passes `notify_telegram` (`agents/watcher_agent.py:2581,3035`). Still 9 added lines.
  - Tests run the real sender with `subprocess` stubbed for every test: spawn error, non-zero exit, timeout, success,
    the policy path and notifications off (`tests/test_why_report_flags.py:449,460`); ERROR, the `failed` record and
    the three retry points (`:478`); the same through WATCHER (`:509`). Mutants WM11-WM18.
  - **Limit:** the installed `cluster_notify.sh` exits 0 in all cases ("Silent on failure (exit 0 always)"; `curl ...
    >/dev/null 2>&1; exit 0`). Exit 0 therefore proves the script ran, not that Telegram delivered. This pass catches a
    missing or unrunnable script, a hang and a disabled policy. Seeing delivery would need a change to that script,
    which is outside the repo and was not touched.
- **Your Q6 cadence point (the announced follow-up): built.**
  - After a why-report is attached, a family that reaches another `max_consecutive_critical` criticals with no reset
    gets a NEW `flag_id`, linked by `previous_flag_id`, and one NEW notification that names the flag it follows
    (`training_health_check.py:1133-1140,1160-1161,1208-1225,1246`; `why_report_flags.py:238-241`).
  - Fewer than that many: `RETRY` with no flag.
  - Test `tests/test_why_report_flags.py:566`; mutants WM21-WM23.
- **Q8 (SUGGESTION): done.**
  - The equality case whose third health result is healthy, asserting that it is consumed (`:349,366`).
  - A `git ls-files` tracked-path assertion next to the check-ignore gate, with a tracked control (`:198-205`).
  - Every suite run with `PYTHONDONTWRITEBYTECODE=1` exported before Python starts, plus `-B`.
  - The snapshot helper asserts git's exit status (`tests/test_benching_switch.py:113-115`).
- **Additional finding 1: accepted.** The READY note's hit-rate section now says the following.
  - At the limit, benching off returns `RETRY` where the base returned `SKIP_MODEL`, so WATCHER can run one more Step 5
    attempt and Step 6 can use a different attempt's model. Pools can differ even when the same family wins.
  - No direction is inferred.
  - Settling measurement: a paired walk-forward, each arm on its own retry path, recording which attempt supplied Step
    6's model, scored against `k/1000` on draws the sieve never saw.

## What was run (VM 101, CPU only, one at a time, `PYTHONDONTWRITEBYTECODE=1` + `-B`; logs `data/fixpass/final/logs/`)

- Benching suite: 30 passed; mutants 20/20 killed, clean copy passes.
- G-2: 23 passed, including 7 mutants.
- Why-flags: 32 passed; mutants 23/23 killed, clean copy passes.
- Wave 1 and wave 2: 37 and 37 passed; their mutants 25/25 and 36/36 killed.
- Land defaults: single 86/86 gates and 9/9 mutants; compare 29/29 gates and 2/2 mutants, logs inside `--out`.
- `tests/test_gemini_relay.py`: 33/33, `COMPLETION SENTINEL: PASS`.
- `switch_check.py --commit 7f28109`: rc 0, 0 stale specs.

## Remaining limits

- Telegram delivery is not observable through the installed script (Q6 limit above).
- The retry LLM reads the canonical report independently (Q4).
- With notifications off, or the script missing, every health check appends a `failed` record and logs ERROR while a
  flag is pending. This is bounded by the number of checks; the record is uncapped, as before.
