# X-BENCHING-SWITCH · R1 · Team Beta review

2026-11-26 Pacific · BEFORE-COMMIT · package base `933a4a08ac38fa1800f025905019793eca7c9022` · main checkout `7f281095c6e1b264473db17c2b8cce6ebee691c1`.

`git status --porcelain` at intake, main checkout

```text
?? docs/exchange/to_beta/X-BENCHING-SWITCH_R1_ALPHA_REVIEW_REQUEST_benching_switch.md
```

`git status --porcelain` at intake, package worktree

```text
 M .gitignore
 M agents/watcher_agent.py
 M meta_prediction_optimizer_anti_overfit.py
 M tests/ch14_wave2_mutants.py
 M tests/test_ch14_diag_wave2.py
 M training_health_check.py
 M watcher_policies.json
?? data/
?? docs/drafts/loop_v1/DRAFT_WATCHER_WHY_FLAGS_20261126.diff
?? docs/drafts/loop_v1/MANIFEST_BENCHING_PACKAGE_20261126.txt
?? docs/drafts/loop_v1/READY_BENCHING_PACKAGE_20261126.md
?? docs/drafts/loop_v1/READY_BENCHING_SWITCH_20261126.md
?? docs/drafts/loop_v1/READY_G2_GRADE_SCOPE_FIX_20261126.md
?? docs/drafts/loop_v1/X-BENCHING-SWITCH_R1_ALPHA_REVIEW_REQUEST_benching_switch.md
?? tests/bench_switch_mutants.py
?? tests/test_benching_switch.py
?? tests/test_g2_grade_scope_fix.py
?? tests/test_why_report_flags.py
?? tests/why_flags_mutants.py
?? why_report_flags.py
```

## Prior record

- Searched both checkouts with `rg` for benching, skip expiry, consecutive criticals, flag identity, grade scope and promotion, Telegram failure and delivery, and the clean-tree gate. Checked the Beta rulings, proposals, chapter audits, consolidated TODO, leads, loop drafts and prior exchange. Read the November 24 primary ruling cited through the READY notes and the November 26 owner record directly.
- **ALREADY RECORDED** · Default-off benching, the WATCHER look-out, the separate G-2 fix ahead of the fixed-grader flip, and hand-run why-reports pending TODO-WHY-1. The request covers these rulings. `logs/e2e_session/OWNER_RULINGS_pending_list_20261124.txt:20–28`, `logs/e2e_session/OWNER_RULING_benching_switch_20261126.txt:2–7`, `CLAUDE.md:43–63`, `docs/LEADS.md:889–897`.
- **ALREADY RECORDED** · Hours mode re-benched immediately after expiry, the legacy compare grades the canonical NN report regardless of the score winner, and the skip registry is tracked. The package also records the uncapped logs, closure after attachment, the default-off CD-11 log and the operator CLI registration status. `docs/drafts/loop_v1/READY_LOOP_DIAG_2_20261123.md:225–231`, `docs/CHAPTER_14_AUDIT.md:159–186,228`, `docs/drafts/loop_v1/READY_BENCHING_PACKAGE_20261126.md:194–213`, request `:124–134`.
- **NEW** · This implementation, the notification and identity failures in Q5–Q6, the remaining opt-in issues in Q3–Q4, and the evidence qualifications below. **CHANGES A RULING** · None.
- Outside the search were the private repository, ser8 archives, the external LL-7/LL-8 and Telegram reference documents, and live or remote run state.

## Positions

| Question | Position | Severity |
|---|---|---|
| Q1 | NO OBJECTION | — |
| Q2 | NO OBJECTION | — |
| Q3 | OBJECTION | THEORETICAL — With hours mode and benching both enabled, an import failure makes the compare keep an expired skip active. |
| Q4 | OBJECTION | THEORETICAL — Fixed grading with all-families promotion can identify the promoted family while returning the winner’s report path and freshness. |
| Q5 | OBJECTION | LIVE — An interrupted registry write can split one uninterrupted series of critical grades into two open IDs under default policy. |
| Q6 | OBJECTION | LIVE — A failed Telegram spawn is recorded as a successful new-flag notification and suppresses the next new-flag attempt. |
| Q7 | NO OBJECTION | — |
| Q8 | SUGGESTION | — |

## Reasoning

### Q1 · Default-off behavior

The legacy single, legacy multi and fixed paths all reach the same critical-action code. At or above the limit it returns `RETRY` with a flag when benching is off. Both registry readers use `skip_active`, which returns false even above the limit. The compare removes only entries whose skip is active. Its import fallback also requires explicit `on` before excluding a family. The absent-key and explicit-off tests exercise over-limit entries and include an `on` control that changes the dispatched family list. The legacy scope of which report is graded remains the recorded B6 limitation. `training_health_check.py:149–154,360–377,460–491,852–872,942–966,1067–1072,1135–1155,1371–1423`, `meta_prediction_optimizer_anti_overfit.py:148–207`, `tests/test_benching_switch.py:220–277,475–498,524–554`.

### Q2 · `on` with `until_reset`

The base comparison exercises critical actions, a reset, registry contents, the status dictionary and the compare family list. It requires the base to actually bench a family. From the tracked registry shape, the writer and readers preserve the former behavior. I also compared the actual `--status` stdout for the same over-limit fixture, and it matched byte for byte. If an operator first collects flags with `off` and later switches to `on`, an existing entry can still carry the two identity fields until reset. Those fields are ignored by the skip decision and status display. `tests/test_benching_switch.py:284–351`, `training_health_check.py:952–966,1105–1146,1214–1224,1406–1423,1707–1717`, base `training_health_check.py:909–927,1042–1076`.

### Q3 · Timed restart

On the normal path an expired skip at count three restarts at one, yielding `RETRY`, `RETRY`, `SKIP_MODEL` across three more critical checks. The active-window control preserves the base result, and unreadable `last_critical` preserves the existing skip with a diagnostic note. The remaining gap is in the compare fallback. With a synthetic `ImportError` and a valid hours policy, the normal reader dispatched four families from an expired entry while the fallback dispatched three. The fallback reads the mode switch and threshold and never evaluates elapsed time, so the omitted family cannot generate the new critical that would restart its counter. A focused fallback test should cover an expired entry alongside the existing absent/off/on cases. `training_health_check.py:942–975,1105–1117`, `meta_prediction_optimizer_anti_overfit.py:160–201`, `tests/test_benching_switch.py:358–380,475–498`.

### Q4 · Promoted-family evidence

The promoted family is the right family to flag. Its grade drives the action and counter. Its report supplies the issues, fixes, metrics and archive, and the returned result names both families. The top-level `report_path`, `report_source` and `freshness` still come from the winner. In a scratch fixture using the explicit `stale_report_action: grade` policy, the result named the promoted NN as the grade setter while pointing at the CatBoost file and carrying CatBoost’s stale run ID. The optional health log uses those same fields. The retry LLM also opens the canonical report independently, which can belong to a different family. Before enabling fixed all-families grading, retain the winner’s provenance under explicit winner fields and test that the grade-setting report and its freshness reach consumers together. `training_health_check.py:775–830,835–872,1009–1025`, `agents/watcher_agent.py:2701–2734`, `tests/test_g2_grade_scope_fix.py:124–161`, `tests/test_benching_switch.py:539–554`.

### Q5 · Flag identity

Healthy sequential writes preserve one ID through repeat criticals and a policy flip within the same streak. An explicit reset removes the anchor. In a scratch copy seeded from the tracked registry, I generated an open flag, replaced the registry with the truncated JSON that an interrupted truncate-and-write can leave, then supplied three more critical reports with zero intervening successes or resets. Both IDs stayed open for the same series. On malformed JSON the next check starts from an empty registry, writes a new first-critical timestamp, and leaves the old flag intact. A temporary file followed by atomic replacement would preserve the previous registry through an interrupted write. Add a check that injects an interruption and verifies the prior identity survives. A smaller contract mismatch is that the fallback anchor uses the first flagging check’s `last_critical` time. The actual `flagged_at` is sampled separately, so the advertised identity formula differs from the reader’s formula for an ID-less record. `training_health_check.py:1097–1133,1158–1185,1214–1226`, `why_report_flags.py:81–86,96–125`, `tests/test_benching_switch.py:423–461`.

### Q6 · WATCHER hook and notification

The hand-landed diff matches the worktree and adds the import fallback and the two hook calls. The no-flag tests exercise both steps, the retry builders match the base, and the hooks catch reader and raised callback errors. The benching policy changes the health action at the threshold, so package-level runs can take an additional Step 5 retry even with identical WATCHER builders. See Finding 1. `agents/watcher_agent.py:115–120,2577–2615,3034–3035,3196–3259`, `why_report_flags.py:175–222`, `tests/test_why_report_flags.py:287–317,353–375`.

The production notifier catches spawn exceptions silently, drops child stdout and stderr, and never checks the child’s exit status. The new-flag hook then logs “owner notified” and appends the ID to the deduplication record. I injected a spawn error into the actual notifier body and called the hook twice. The first call logged success and persisted the ID. The second returned `repeat` without another launch attempt. A later reminder can still be attempted if the durable flag is open, with the same silent transport failure behavior. Use a notification path that surfaces the result of the attempted dispatch, logs failures and leaves failures eligible for retry. Tie the ledger and message wording to the outcome WATCHER can observe. Keep the transport check off the pipeline’s critical path. `agents/watcher_agent.py:135–154`, `why_report_flags.py:175–193,200–222`.

One new alert per ID plus one consolidated reminder per owner-started run is a reasonable cadence for this setup. It is in addition to Step 5’s existing degraded messages. The already announced follow-up for persistent criticals after an attachment is relevant to cadence and remains open. `why_report_flags.py:185–189,205–218`, `agents/watcher_agent.py:3209–3217,3237–3249`, request `:124–134`.

### Q7 · Clean-tree gate

Git matched all three exact output paths to the added rules, and I checked the index separately. None of those three paths is tracked. Their creation under the repository-root WATCHER cwd leaves them out of porcelain. The health check that first emits a flag also updates the already tracked registry, so the complete operation can still dirty the checkout and cause the next clean-tree gate to refuse. That is the recorded exposure at the base. The attach command checks an existing report file and writes the attachment record. A separately created report remains subject to Git’s ordinary rules. `.gitignore:128–130`, `training_health_check.py:1125–1139,1196–1203`, `why_report_flags.py:133–151,212–218`, `scripts/gate12_cleantree_gate.py:108–134`, `window_optimizer_integration_final.py:115–122`.

### Q8 · Test credit and isolation

The dispatch tests verify actual stubbed family calls and a positive exclusion control. The equality tests check that both steps ran, and the run-start and new-alert tests verify their expected calls. Three credit limits matter here.

- The named `retry_then_proceed` equality fixture supplies three critical results before its healthy result. With the two-retry limit, Step 6 follows exhaustion and WATCHER never consumes that healthy result. Keep the exhaustion case and add one where the third health result is healthy. The notification tests measure calls to a list callback. A notifier failure regression needs to exercise the production callback with its subprocess boundary stubbed. `tests/test_why_report_flags.py:254–262,287–304,339–350`, `watcher_policies.json:131–135`, `agents/watcher_agent.py:3201–3249`.
- The ignore gate uses `--no-index`, so it can pass for a path that was force-added and would dirty the tree. Add the independent tracked-path assertion to the gate. The suite’s clean negative control for the neighboring file still has value. `tests/test_why_report_flags.py:146–156`, `tests/why_flags_mutants.py:67–88`.
- I found no application fixture write outside the configured scratch paths in the package runs. Alpha’s aggregate runner exports `PYTHONDONTWRITEBYTECODE` before starting Python and keeps its own console logs in a separate directory. Use that environment setting or `-B` in the standalone commands too. Setting the environment variable after Python starts does not stop the current interpreter from writing import caches. The benching snapshot also filters out bytecode from porcelain, scopes file metadata checks to selected directories, and accepts Git stdout without checking its exit status. Treat it as a check of those paths and that invocation, with enforcement of the after-snapshot in fixture teardown if individual tests are run. `data/pkg/run_all.sh:1–11`, `tests/test_benching_switch.py:19–27,81–127,557–595`, `tests/test_why_report_flags.py:14–23,48–57`.

## Additional finding

1. **NEW · The package’s hit-rate prediction omits Step 5 retries.** The READY note says pools can change only when a family retained by benching off would have won a compare. Starting from the tracked count of one, two critical checks can take the count to three with one retry already consumed. The base then advances to Step 6 on `SKIP_MODEL`. With benching off, WATCHER can run a third training attempt and use its artifacts for Step 6. The same family can win both attempts and still produce different pools. I infer no direction for hit rate. The settling measurement is paired walk-forward with each arm using its actual retry path, recording which attempt supplied Step 6’s model and scoring hits against `k/1000` on draws the sieve never saw. `docs/drafts/loop_v1/READY_BENCHING_PACKAGE_20261126.md:152–162`, `diagnostics_outputs/model_skip_registry.json:1–8`, base `training_health_check.py:1053–1076`, package `training_health_check.py:1135–1146`, `agents/watcher_agent.py:3196–3259`, `meta_prediction_optimizer_anti_overfit.py:341–351`.

## Verification

All 18 hashes matched the package manifest, the nine-line WATCHER change matched the hand-landed diff, and the five paths changed on main since the base were separate from the package’s code and test files. In the activated Torch environment, I reran the benching, G-2 and why-report suites under `/tmp` (26, 19 and 13 passed), the four Wave 2 CD-09 tests, and both package mutation runners (17/17 and 10/10 killed, with clean controls passing). I ran scratch probes for CLI parity, interrupted registry state, failed notification spawn, promoted report provenance and expired hours-mode fallback. The compare and WATCHER tests stub their step and notification dispatch.

I read Alpha’s recorded Wave 1, Wave 2, landing-defaults and switch-check results. Those broader reruns, a real Telegram delivery check, an end-to-end run on the prospective landing tree and a walk-forward hit-rate result are **UNAVAILABLE** from this review.
