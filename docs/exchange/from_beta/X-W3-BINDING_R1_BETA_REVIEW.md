# X-W3-BINDING R1 · Team Beta design review

2026-11-27, Pacific. Base read `14815a794a23f1872e478c5bd202d9ecd411f74b` in both checkouts. The design is the untracked `docs/drafts/unit_w/DESIGN_FLAG_MODEL_BINDING_20261127.md` in Alpha's `w3_binding_design` worktree. Its SHA-256 at review was `0f2eed9d79ccc8152635b5957b5480aa46a1182cfb2210628ffff7b23468a438`.

`git status --porcelain` at intake in the main checkout

```text
?? docs/exchange/to_beta/X-W3-BINDING_R1_ALPHA_REVIEW_REQUEST_flag_model_binding.md
```

`git status --porcelain` at intake in the design worktree

```text
?? docs/drafts/unit_w/BETA_REQUEST_DRAFT_X-W3-BINDING_R1.md
?? docs/drafts/unit_w/DESIGN_FLAG_MODEL_BINDING_20261127.md
```

## Prior record

- I checked Alpha's prior record and used `rg -l -i` and `rg -n -i` for CD-05/CD-07, run identity, compare artifacts, promoted grades, repeat selection, checkpoint fallback and hash latency across the project file catalog, Beta rulings and ruling requests, proposals, chapters and audits, consolidated TODO, backlog, leads, exchange, drafts and local session rulings. I read the W plan §§6.4–6.9, Alpha's design §§1–8, the W2 closing note and READY record, and the learning-loop section. The rigs, Proxmox, private remote, ser8 archives and live run state were outside this review.
- **Finding 1 · ALREADY RECORDED · Omitted governing record.** The owner's G-2 ruling and its closeout require the promoted family's own evidence to drive the health result, down to report path, source and freshness. The flag's digest and family must follow that same report at the point where the flag is raised. The cited CD-05 row also specifies a run-scoped per-family tree report alongside the canonical copy. The design cites that row for tree report coverage but omits the run-scoped path, which bears directly on Q1 and Q2. `logs/e2e_session/OWNER_RULING_benching_switch_20261126.txt:7`, `docs/drafts/loop_v1/READY_G2_GRADE_SCOPE_FIX_20261126.md:34–44,53–62`, `docs/drafts/loop_v1/X-BENCHING-SWITCH_R2_ALPHA_CLOSING_NOTE_benching_switch.md:65–73`, `docs/drafts/ch14_v1/PLAN_CH14_DIAGNOSTICS.md:187`.
- **ALREADY RECORDED.** CD-07's dispatch stamp, the legacy compare directory discrepancy, the archive sidecar's canonical path, the runner's latest-nonempty-digest rule and legacy live fallback, and synchronous pre-append hashing cost. The earlier repeat-with-null observation is a status. Its application to a W3 occurrence explicitly marked unbound is a design choice. `docs/drafts/ch14_v1/PLAN_CH14_DIAGNOSTICS.md:174`, `docs/CHAPTER_6_AUDIT.md:179–186`, `docs/exchange/from_beta/X-W2-HOOK_R2_BETA_REVIEW.md:68–76`.
- **NEW.** The producer's exact report/checkpoint pairing, the precedence of an explicit unbound occurrence, the compare-child exit gate and the proof needed for the new writes and pre-append I/O.
- **CHANGES A RULING.** Alpha's option A changes the accepted flag line on disk. Replacing the tree report's checkpoint digest with a producer-authored pair in the sidecar also needs Michael's agreement under the W2 closing order. The prior order names both the report and the sidecar. `docs/exchange/to_beta/X-W2-HOOK_R2_ALPHA_CLOSING_NOTE.md:15–16`, `docs/drafts/unit_w/READY_W2_20261127.md:57–63,75–83`.

## Positions

| Question | Position | Severity |
| --- | --- | --- |
| Q1 | OBJECTION | LIVE — Two supported Step 5 invocations using the same output directory can let one writer certify the other writer's checkpoint under its own run ID. |
| Q2 | OBJECTION | LIVE — A newly unbound flag occurrence can select a different attempt's checkpoint through the current runner. |
| Q3 | NO OBJECTION | — |
| Q4 | NO OBJECTION | — |
| Q5 | SUGGESTION | — |
| Q6 | SUGGESTION | — |
| Q7 | NEEDS EVIDENCE | — |

## Q1 · Bind the pair at the final-model writer

The missing link is producer provenance for the **pair**. Under C9, A can write its tree report and serialize its checkpoint, B can overwrite the shared same-family checkpoint, and A can then hash the path and publish a sidecar pairing A's report with B's bytes. The resulting family, run ID and digests all agree, including on a flag-time rehash. A scratch interleaving demonstrated that the proposed equalities admit that pair. The base uses shared checkpoint names and copies those paths after each compare child into its archive. The direct WATCHER pipeline CLI enters `run_pipeline` without passing through the daemon PID check. Section 5's same-second test covers a replacement **after** the pair was recorded. Test a replacement **before** the writer records it, including across different S88 seconds. `meta_prediction_optimizer_anti_overfit.py:224–229,259–269,309–318,1295–1372,1903–1926,2826–2839`, `agents/watcher_agent.py:3310–3325,4191–4217`, design §§4 C9/C11, 5.

Capture both digests from artifacts the final-model writer owns, using run-local staging or an enforced single writer from report/checkpoint creation through digest capture. For trees the report builder receives the fitted object used by the checkpoint writer, so an authoritative pair can preserve the current public write order. For NN, require equality between the worker report's checkpoint digest and the archived/bound checkpoint. C7 writes that value and §§3/C13 never require it. Give only the final-model path authority to publish the pair. A report-count gate is a useful regression check, and a same-run trial report substituted before publication should also fail. Keep one authoritative report/checkpoint pair. If the identity index mirrors fields from the sidecar, verify the sidecar digest and compare any mirrored fields used in the decision. The NN report includes round history and the worker restores the best state before saving. The claim is that the report grades the training run that produced the named final checkpoint. `meta_prediction_optimizer_anti_overfit.py:1903–1926,1962–1964,2694–2703,2826–2839,2958–2969`, `train_single_trial.py:704–728,752–795`, design §§3, 4 C7/C9–C13, 5.

## Q2 · Missing cases

- **New unbound occurrence after a bound repeat, or on its own.** The real base reader picked the old digest when I appended an explicit unbound repeat with the same flag ID. An unbound-only fixture yielded no digest, which sends `run_for_flag` to the live checkpoint. That behavior is recorded for W2. With W3, it can attach a report to the same flag group for a checkpoint other than the one behind the latest occurrence. Have the latest occurrence decide for W3, leave a newly unbound flag open with its reason and preserve the recorded historical fallback for legacy-only flags. Gate bound A → unbound B and unbound B alone, with an otherwise usable and graded substitute present, and assert neither attaches. Also gate a new occurrence arriving during report generation. The existing attachment check covers group openness, and W2 already records that race. `training_health_check.py:1142–1177,1240–1243`, `scripts/diagnostics/why_report_runner.py:202–216,561–571,622–634`, `why_report_flags.py:219–247`, `docs/exchange/from_beta/X-W2-HOOK_R2_BETA_REVIEW.md:74–76`.
- **Compare child writes a complete current pair and then exits nonzero.** The parent archives before using the return code and ends with exit zero. C11's three checks accept a current pair even in that case. Make `returncode == 0` a condition for publishing a bindable compare identity, independent of the existing winner logic. Use a stub that writes all matching artifacts and exits 1. The writes-nothing fixture cannot exercise this case. `meta_prediction_optimizer_anti_overfit.py:308–325,328–351,369–379`, design §§4 C11, 5.
- **Fixed grading promotes a non-winner.** Carry the exact digest, path, original family and run ID with each parsed report and select the grade setter's tuple before `_severity_to_action` can raise the flag. The result's promoted path is set later. Gate this with a healthy winner and critical non-winner on `fixed + all_families`, and require a bound flag with the promoted report's digest and checkpoint. A canonical legacy NN fixture exercises a different route. `training_health_check.py:811–843,846–869,881–894,1171–1177`, design §§4 C12–C13, 5.

## Q3 · One identity per dispatched attempt

Yes. Stamp the result returned by `run_step` with the ID and start time it generated, including when it loads the result from disk. The health retry calls `run_step` again and a genuinely skipped or Step 6-only invocation has no Step 5 report to grade. Carry the output-directory hint on that result too if the W2 unbound message still uses it. The family count can span attempts under the existing policy. Each flag occurrence should name the checkpoint of the attempt that drove its grade, subject to the proof in Q1. `agents/watcher_agent.py:2005–2013,2468–2508,2516–2535,3123–3130,3187–3239`, `training_health_check.py:1142–1177`.

## Q4 · Diagnostics level

Yes. Thread a supplied dispatch stamp independently of diagnostics level. On the supported Step 5 routes at legacy, only NN emits a final report, so only NN is eligible to bind at that level. Preserve the legacy NN collector when threading the stamp. Its constructor takes a separate branch from the full collector, and the fallback emitter builds another instance. A fallback report without producer proof remains unbound. `training_diagnostics.py:142–181`, `train_single_trial.py:191–207,650–668,787–806,1008–1042`, `meta_prediction_optimizer_anti_overfit.py:2809–2842`.

## Q5 · Compatibility and failure behavior

- Preserve full-mode CLI parity when no identity was supplied. Full already generates a diagnostics ID internally and forwards it to children. Trigger W3 writes from the explicit binding request, with a compare-parent/child test that proves the pre-existing full-mode ID alone cannot enable them. Keep provenance and the existing model readers and score keys intact. Run a before/after comparison of the old artifacts and child argv at both levels. `training_diagnostics.py:142–181`, `meta_prediction_optimizer_anti_overfit.py:94–139,269–278,369–376,1359–1372`, `prediction_generator.py:310–329`, `chapter_13_orchestrator.py:600–609`, `models/inference_bundle.py:256–275`.
- Make each new identity write best-effort. Inject a read error and an identity-record write error and verify the existing Step 5 exit, winner restore, and flag/health action, with a missing or unbound identity as appropriate. A handoff file should become visible only when its record is complete. The tree report already catches failures and the legacy compare restore precedes the existing summary writes. `meta_prediction_optimizer_anti_overfit.py:353–379,2817–2842,3382–3400`, `training_health_check.py:257–272,1304–1316`.
- Reconcile the no-ID promise and the stale gate. The base fixed grader can yield stale using the compare pointer alone, so C5 can change the no-ID registry reset. With an expected ID, a missing report stamp currently falls through to timestamp or mtime. I verified an unstamped same-family report with a post-start mtime reads as fresh. Decide whether WATCHER-issued strict checks reject unstamped reports and pin the outcome explicitly in the gate. For fixed grading, test a current single-model report alongside a leftover full-compare pointer. The pointer presently picks the report path before freshness is checked, so the new run ID can produce a stale outcome without evaluating the current report. `training_health_check.py:317–325,790–822,916–946,950–961`, `agents/watcher_agent.py:2585–2596`, design §§4 C5, 5.

## Q6 · Storage

I favor **A if Michael amends the ruling**. Binding status, reason and digest go in the same append as the occurrence. I found no tracked runtime parser broken by additive keys. `read_flags` extracts named fields and still counts one repeat per line. The runner already reads the digest and directory. It still needs the occurrence rule in Q2 and an explicit handoff of the checkpoint path or basename in C15. Its archive match is a search by digest. The LL-8 spec ultimately loads the matched frozen block's model directory and checks the loaded digest and family. Include a gate that reaches that handoff. `why_report_flags.py:76–92,117–139`, `scripts/diagnostics/why_report_runner.py:202–257,269–307,508–516,561–571`, `scripts/diagnostics/attribution_cycle.py:495–504`.

Under the current ruling, **B** preserves the primary flag line. Join only the exact occurrence. A crash or failure between the appends needs an explicit unbound outcome for a W3 occurrence. Specify how the reader distinguishes that missing peer from a legacy flag, and show an interrupted-append test before relying on B. `docs/drafts/unit_w/READY_W2_20261127.md:65–83`, design §4.1.

## Q7 · Cost before Step 6

Synchronous streaming SHA-256 is acceptable given a measured budget and a bounded failure path. The flag append and Step 6 both wait for the check. Cheap checks reduce frequency, and a byte cap limits data read, while a stalled read can still exceed any wall-time target. Measure cold and warm reads on representative and largest supported artifacts, including total added Step 5 publication time and worst-case pre-append time. If the read cannot finish within an agreed Step 6 budget, use a deadline the health check can enforce and append one unbound occurrence with a named reason. Gate missing, unreadable and timed-out reads and assert the existing action and one append. Actual artifact sizes and timing are **UNAVAILABLE** here, so I would set no numerical cap from this review. `agents/watcher_agent.py:3187–3204`, `training_health_check.py:1304–1316`, `models/inference_bundle.py:168–173`, design §§4 C13/C14, 8.

## Verification

On the base, in the activated Torch environment with CUDA disabled, `tests/test_g2_grade_scope_fix.py` passed **23/23** with its temporary directory under `/tmp`. Two scratch checks exercised the real flag reader and freshness function. The mixed-pair scratch check exercised the proposed equality rule with synthetic bytes. It did not execute W3 code. I read the remaining dispatch, producer, reader and runner paths. W3 code, installed owner-patch gates, live timing and walk-forward outcomes are **UNAVAILABLE** for this design review. The review concerns the accuracy of the advisory flag and its report. Pool hit rate was unmeasured.
