# X-W2-HOOK R2 · Team Beta review

Date 2026-11-27. Base commit `5f6c1924a214d78a5ce9b1a5aaf91d9e9e894222`. Read the submitted worktree and the owner patch at 08:30 PST.

`git status --porcelain` in `/home/michael/tfm_worktrees/w2_watcher_hook` at intake

```text
 M scripts/diagnostics/why_report_runner.py
 M tests/bench_switch_mutants.py
 M tests/test_why_report_flags.py
 M training_health_check.py
?? data/
?? docs/drafts/unit_w/
?? tests/test_why_report_auto.py
?? why_report_auto.py
```

`git status --porcelain` in `/home/michael/distributed_prng_analysis` at intake

```text
 M docs/exchange/LEDGER.md
?? docs/exchange/from_beta/X-W2-HOOK_R1_BETA_REVIEW.md
?? docs/exchange/to_beta/X-W2-HOOK_R1_ALPHA_REVIEW_REQUEST_watcher_why_report_hook.md
?? docs/exchange/to_beta/X-W2-HOOK_R2_ALPHA_REVIEW_REQUEST_watcher_why_report_hook.md
```

| Question | Position | Severity |
| --- | --- | --- |
| Q1 | OBJECTION | LIVE — with automation enabled, a safety halt inside the step loop still invokes the report runner. |
| Q2 | OBJECTION | LIVE — a default-off flag can still name another run's checkpoint, and a normal first import can write a new cache file. |
| Q3 | OBJECTION | LIVE — the default compare route can publish a new winner in the canonical directory while the flag hashes an older model in the requested directory. |
| Q4 | OBJECTION | LIVE — an ordinary stale diagnostics file can produce a flag naming the latest same-family checkpoint. |
| Q5 | NO OBJECTION | — |
| Q6 | SUGGESTION | — |
| Q7 | OBJECTION | LIVE — the suite can certify a clean replay of the patch while the installed owner files lack the hook. |
| Q8 | OBJECTION | LIVE — a successful attachment with a failed attempt append generates an alert saying the report failed with reason `None`. |

## Prior record

I checked Alpha's searches and read the cited W1 exchange, the W2 READY note and owner patch, the Chapter 14 owner rulings and learning-loop section, and the relevant W plan sections. I searched `docs/`, including the exchange, drafts, `TB_RULING_*`, `PROPOSAL_*`, the chapter audits, consolidated TODO, backlog and leads, for compare output routing, CD-07, flag identity, halt and runner isolation, and summary and notification behavior. The rigs, Proxmox, ser8 archives, private remote, and live run state were outside this review.

**Finding 1 · ALREADY RECORDED · Omitted governing record.** Chapter 6 B2 already records that the default compare route drops `--output-dir`, and CD-07 already records the missing WATCHER run-identity handoff. The W plan also lists run ID among the flag inputs, requires checkpoint binding and off-state file parity, and specifies a child-process boundary for the runner. These govern Q1–Q4. `docs/CHAPTER_6_AUDIT.md:131–139,179–186`, `docs/drafts/ch14_v1/PLAN_CH14_DIAGNOSTICS.md:174`, `docs/exchange/to_beta/X-CH14-DIAG_R1_ALPHA_REVIEW_REQUEST_ch14_diag_landing.md:155–159`, `~/tfm_worktrees/_staging/ll78/PLAN_LAND_LL7_LL8_20261126.md:196–216`.

**ALREADY RECORDED.** The trigger is W2's scope. The queue comes after real reports. W1's hash checks and flag-time gap, W2's latest-line-with-a-hash rule, winner-only lookup in the supplied directory, and serial per-flag cost are recorded. `logs/e2e_session/OWNER_RULING_ll78_plan_20261126.txt:2–5`, `docs/exchange/from_beta/X-W1-RUNNER_R1_BETA_REVIEW.md:62`, `docs/drafts/unit_w/READY_W2_20261127.md:35–46,281–287,425–431`.

**NEW.** The compare override and stale-report paths now have checkpoint attribution consequences. The scratch probes also exposed a halt path into the hook, off-state bytecode, and a false failure notification. The release gate's use of a reconstructed patch leaves the landed hook unverified. **CHANGES A RULING** has no entries.

## Reasons

### Q1 · Result, exceptions and WATCHER state

On the normal path the hook runs after Step 6, progress cleanup and `_pipeline_running = False`. Its return is discarded. Both guards catch ordinary `Exception`s, and the inner `finally` restores `sys.dont_write_bytecode` even on an injected `SystemExit`. That injected `SystemExit` escapes both guards. I found no current enabled-path LL-8 exit that propagates that way because LL-8 runs as a child. The runner's Python orchestration, policy reads, artifact scans, hashes and attempt handling now run synchronously inside WATCHER. This differs from the plan's runner process boundary. The runner can also insert the repo into `sys.path` if absent, and its imports remain cached. `docs/drafts/unit_w/W2_OWNER_LAND.patch:30–39`, `why_report_auto.py:87–129`, `scripts/diagnostics/why_report_runner.py:80–86,531–563,593–609,692–720`, `~/tfm_worktrees/_staging/ll78/PLAN_LAND_LL7_LL8_20261126.md:211–216`.

**Verified halt path.** In the synthetic WATCHER driver, the startup safety check passed and the first in-loop check refused. WATCHER dispatched zero steps and still called the stubbed runner once. The `break` at the safety check flows to the same epilogue as successful completion. The daemon shutdown handler also clears `_pipeline_running`, and the epilogue can then begin an entire batch before the daemon returns to its loop. Gate the hook on the intended completion conditions and cover both an in-loop halt and graceful shutdown. Keep the planned process boundary for the enabled runner if it is required to survive a runner process failure. `agents/watcher_agent.py:3078–3108,3267–3273,3426–3436,3534–3567`, `docs/drafts/unit_w/W2_OWNER_LAND.patch:30–39`, `scripts/diagnostics/why_report_runner.py:426–450,692–708`.

### Q2 · Off or missing

The patched epilogue imports `why_report_auto` before reading the switch. A scratch first import with bytecode enabled and policy absent wrote `__pycache__/why_report_auto.cpython-310.pyc`. The runner stayed unloaded. `auto_run` reads the policy and returns early for off or missing keys, so it writes no report or attempt in those cases. Independently of the switch, the health call can load the checkpoint resolver, hash a checkpoint, add fields to the returned flag and its on-disk line, and log an identity warning. It can also persist the wrong hash in Q3/Q4. The no-flag path and the tested flag path with `models_dir=None` are narrower than off-state parity for a real Step 5 flag. If Alpha intends flag enrichment with automation off, record that as an explicit change to the plan's parity expectation and test it as such. `docs/drafts/unit_w/W2_OWNER_LAND.patch:21–38`, `why_report_auto.py:28–41,53–74,87–105`, `training_health_check.py:1240–1258,1293–1306`, `tests/test_why_report_auto.py:523–549`, W plan §6.6 item 5.

### Q3 · Requested directory versus actual publication

**Verified compare route.** Step 5 defaults to comparison. WATCHER builds the argv from the manifest and permitted overrides and stores the requested directory. The S88 CLI handoff drops that directory and restores the winning files under `models/reinforcement`. With training children stubbed, comparison published a new CatBoost winner there. The real `_flagged_model` returned an older CatBoost hash from the supplied custom directory (`bbbb9c8e3beb` versus new `d1e435370ade`). The underlying output routing is Chapter 6 B2. W2 is where that routing becomes a wrong checkpoint identity. Alpha's custom-path test verifies transmission to argv and health, with the child dispatch stubbed. It cannot establish where compare writes. `agent_manifests/reinforcement.json:173–179`, `agents/watcher_agent.py:2032–2041,2420–2435`, `docs/drafts/unit_w/W2_OWNER_LAND.patch:13–28`, `meta_prediction_optimizer_anti_overfit.py:224–226,259–269,308–325,351–363,3281–3307`, `tests/test_why_report_auto.py:878–885,952–964`.

**Verified blank override.** An empty string omitted `--output-dir` while passing `models_dir=""` to the stubbed health check. On the single-family path Step 5 would use its argparse default, and `_flagged_model` would interpret the empty string as WATCHER's cwd. A JSON null similarly omits the argument and suppresses identity enrichment. Use the completed Step 5 publication for provenance and cover compare, omitted and explicit single-family paths. `agents/watcher_agent.py:2424–2435,4180–4202`, `meta_prediction_optimizer_anti_overfit.py:3193–3195,3341–3350`, `training_health_check.py:1245–1253,1293–1294`.

For the stock serial path, I found no read of a prior invocation's `_step5_output_dir` caused by a genuine freshness skip or a resume at Step 6. The skip returns `skipped=True`, the health block requires Step 5 and excludes that result, and a subsequent dispatched Step 5 overwrites the value before running. The health retry re-enters Step 5 with copied original params plus its modifications. The value can remain on the instance after a skip or a run without Step 5. The new test for a missing Step 5 uses a fresh instance and a stub reporting success without `skipped=True`. An invocation-local result would make the association explicit where the health check consumes it. `agents/watcher_agent.py:2005–2012,2467–2512,2661–2670,3093–3104,3123–3127,3171–3197,3201–3236`, `docs/drafts/unit_w/W2_OWNER_LAND.patch:5–26`, `tests/test_why_report_auto.py:896–899,967–975`.

### Q4 · Checkpoint and diagnostics can describe different runs

**Verified stale report.** I gave the real health check a critical report stamped for run A and a same-family checkpoint from run B. With only `models_dir`, it raised a flag carrying B's SHA. With `expected_run_id` naming B, it returned `stale` and raised no flag. The legacy health guard depends on caller run context and the owner patch supplies only the directory. For example, WATCHER can omit `--enable-diagnostics` via a false override and retain the old canonical report. A matching family and a readable checkpoint establish neither the producing run nor which checkpoint received the grade. Use the producing run identity to bind the diagnostics and checkpoint before asserting the flag's hash. `agents/watcher_agent.py:2032–2041,2424–2435,3191–3197`, `docs/drafts/unit_w/W2_OWNER_LAND.patch:21–28`, `meta_prediction_optimizer_anti_overfit.py:249–251,270–278,3209–3211`, `training_health_check.py:250–252,311–319,898–930,1240–1258,1293–1306`.

The hash is synchronous and precedes the append, so its I/O adds latency before a flag and before Step 6. A stalled read can block there without a deadline. An ordinary resolution or hash exception yields the null fields and the existing append is still attempted. For a promoted non-winner the family check correctly rejects the winner sidecar. The S88 archives contain family-specific directories, but their copied sidecars may still point to canonical paths. In scratch a relocated sidecar selected the extant written path first and the local copy only after that path disappeared. Require same-run artifact provenance before claiming identity from any archive. `training_health_check.py:823–863,1240–1258,1293–1306,1539–1545`, `meta_prediction_optimizer_anti_overfit.py:259–260,314–325,351–360`, `models/inference_bundle.py:156–165`.

### Q5 · From a named SHA to an attached report

For stable inputs, the latest appended line carrying a SHA controls. A scratch repeat with null SHA left the previous named SHA selected, as the lead's rule specifies. The three lookups compare file bytes to the requested digest. Block selection uses that digest, the spec requires it, LL-8 verifies the loaded bundle's digest and family, and attachment follows a matching child result and a report hash check. I found no normal substitution after the named SHA is selected. Q3/Q4 concern the upstream truth of the SHA. The run-time lookup of flags without a named SHA remains the recorded W1 behavior. The flag snapshot is also taken before running LL-8, and attachment checks group openness without rechecking the named line. A repeat appended by a concurrent publisher in that interval can make the report stale relative to the new line. No concurrent-publisher result was verified. `scripts/diagnostics/why_report_runner.py:202–257,269–307,508–516,561–571,610–634`, `scripts/diagnostics/attribution_cycle.py:482–504`, `why_report_flags.py:219–247`.

### Q6 · Landing and process restart

Land the cohort together and start a fresh WATCHER process from the finished tree before relying on W2. A process importing the patch against old health code can raise `TypeError` before the health function's own exception guard. A running base process retains its old `run_pipeline` and imported health function after disk files change. It therefore continues to emit legacy flags and cannot run the new hook. Conversely, a patched process keeps the eagerly imported health function, while its hook can import the auto module later and read policies and dynamically load the runner from disk. A commit records a coherent cohort on disk. The in-memory function references change on process restart. `agents/watcher_agent.py:101–113,3191–3199,3267–3273`, `docs/drafts/unit_w/W2_OWNER_LAND.patch:21–39`, `why_report_auto.py:44–50,53–60,90–105`, `training_health_check.py:210–214`, `docs/drafts/unit_w/READY_W2_20261127.md:50–57`.

### Q7 · The historical patch gate and the installed code gate

Pinning a comparison to `5f6c192` is valid for historical replay. `W2.owner_patch` recreates that base in a scratch clone, applies the archived patch and uses the scratch code for the hook and Step 5 gates. It computes `same_owner` and prints it, while its pass predicate ignores it. Alpha's phase A demonstrates the difference directly. The gate passed with `same_owner=False` and phase A was 19/19 green. A post-landing green replay therefore cannot detect a subsequent removal of the installed hook. The flag-suite source check deliberately accepts the base or the fully patched state. Keep replay as a pre-landing gate and exercise the installed WATCHER directly as the post-landing gate, with removal of the installed hook as a negative control. `tests/test_why_report_auto.py:721–727,741–770,826–844,941–991`, `tests/test_why_report_flags.py:375–404`, `~/tfm_worktrees/_proof/w2_5f6c192_logs/A_auto.log:12,38`.

### Q8 · Report outcome and notification

**Verified false failure alert.** W1 can attach a successful report and then fail to append its attempt line, returning `status="ok"` and `attempt_record_error`. Feeding that shape through the real W2 hook with the notifier stubbed produced an ERROR and a failure notification with `f: None`. The recipient needs the true outcome and the failed operation. Preserve the attachment result and name the attempt write error in the alert. Also, the ordered failed-flag summary is emitted at the next invocation's start. The newly failed flags get errors and a notification at the end of the current call. If the §6.9 summary is meant for that same invocation, the current hook does not emit it. `scripts/diagnostics/why_report_runner.py:622–643`, `why_report_auto.py:106–125`, `agents/watcher_agent.py:3034–3035`, `why_report_flags.py:253–269,377–387`, `~/tfm_worktrees/_staging/ll78/PLAN_LAND_LL7_LL8_20261126.md:223–225`.

## Verification

I activated the Torch venv, matched all seven submitted hashes to Alpha's manifest, applied the owner patch in `/tmp/beta-w2-r2.8JKMy6/src`, and ran the submitted W2 suite on CPU. It passed **19/19 gates and 17/17 mutants**, including the nested flag and benching suites. I then ran the described targeted probes with synthetic files, stubbed step dispatch, stubbed compare children and stubbed notifications. The scratch suite log is `/tmp/beta-w2-r2.8JKMy6/auto.log`. I read Alpha's A/B logs and chain for the W1 and other suite totals, and rechecked the submitted hashes and worktree status before writing this review. Real pipeline and daemon rollout evidence, live flag-match rates, and post-landing main evidence are **UNAVAILABLE**. This review measures diagnostic binding and WATCHER behavior. It contains no held-out hit-rate measurement.
