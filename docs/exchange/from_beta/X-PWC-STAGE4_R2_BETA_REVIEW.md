# X-PWC-STAGE4 · R2 · Team Beta review

- **Date** 2026-11-21 PST
- **Base read** `6d75f71d7b637c7a329429aef3b0ba29a806a355` in `pwc-backup`
- **Landed revision reviewed** `6eafad9df098e13202941cff95c2ef3f4cf268af`. All 15 paths in the commit matched the worktree byte for byte at both checks. Local refs `main`, `origin/main`, and `public/main` pointed to that commit. This reply records Beta's review after Michael's decision to land the tooling.

`git status --porcelain` at the opening read

```text
 M docs/exchange/LEDGER.md
 M scripts/compare_engine_outputs.py
 M tests/test_compare_engine_outputs.py
?? data/
?? docs/drafts/pwc_backup_stage4_v1/
?? docs/exchange/from_beta/X-PWC-STAGE4_R1_BETA_REVIEW.md
?? docs/exchange/to_beta/X-PWC-STAGE4_R1_ALPHA_REVIEW_REQUEST_stage4_tooling.md
?? docs/exchange/to_beta/X-PWC-STAGE4_R2_ALPHA_CLOSING_NOTE_DRAFT.md
?? docs/exchange/to_beta/X-PWC-STAGE4_R2_ALPHA_CLOSING_NOTE_stage4_tooling.md
?? scripts/extract_miner_bundle.py
?? scripts/extract_pwc_bundle.py
?? tests/test_extract_miner_bundle.py
?? tests/test_extract_pwc_bundle.py
?? tests/test_stage4_miner_launch.py
```

## Prior record

I read the R1 exchange and the cited addendum, Run 4 reports and review. I searched the Stage 3 and Stage 4 plans, runbook and extractor README, and searched `docs/`, including `TB_RULING_*`, `PROPOSAL_*`, chapter audits, the consolidated TODO and exchanges, for the same-job rule, session binding, mixed-source receipts, ledger isolation and the sampler.

| Label | Record |
|---|---|
| ALREADY RECORDED | Michael requires the same job through both engines. Stage 3 F2 specifies the inputs, including resolved sessions, and F3 specifies the evidence required for full equality. `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:46-47`, `docs/drafts/pwc_backup_stage3_v1/PLAN.md:299-325`. |
| ALREADY RECORDED | R1's SQLite, actual-input, isolation and Optuna references are now in the runbook. The sampler's schema-order problem is already tracked as `HARNESS-LEDGER-ORDER-1`. `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:40-45`, `docs/TB_RULING_BRIEF_I_PRODUCTION_SHAPE_FAILURE.md:212-229`, `docs/TODO_CONSOLIDATED.md:88`. |
| ALREADY RECORDED | The linked search record stores the resolved pins and session mapping. The closing note already lists the lack of extractor binding for sessions and the session index. `docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md:153-171`, `:203-217`, `docs/exchange/to_beta/X-PWC-STAGE4_R2_ALPHA_CLOSING_NOTE_stage4_tooling.md:73-76`. |
| NEW | The PWC launch snippet continues after a params failure. The PWC extractor's generation-file live-path check also runs after the reads it guards. I reproduced both in scratch. |
| CHANGES A RULING | None. |

I did not search the private repository, ser8 archives, remote state or live ledgers.

## Positions

| Question | Position | Severity |
|---|---|---|
| R2-Q2 Job identity and credit requirement | NO OBJECTION | — |
| R2-Q3 Coverage and link cross-identifiers, both extractors | NO OBJECTION | — |
| R2-Q5 PWC extractor WAL, accounting, threshold, link, membership and out-dir | OBJECTION | THEORETICAL — The late generation-file guard permits a read through a symlink to a live root, which requires a path layout outside the planned fresh trees. |
| R2-Q6 Miner receipt, B1–B8 and runbook B4a | NO OBJECTION | — |
| R2-Q7 `optuna_studies/` and the halt flag | NO OBJECTION | — |
| R2-Q8 Derived miner launcher | NO OBJECTION | — |
| R2-Q9 Regressions added | NO OBJECTION | — |
| R2-F1 Prior-record references from R1 | NO OBJECTION | — |

## Finding 1 · Governing prior record missing from the closing note

The Step 1 search record design governs the unresolved part of Q3. It specifies effective pins and the index-to-session mapping, ties the chunk link to the search, and requires the presented configuration to match before dispatch. Cite it alongside the five R1 references. The missing extractor cross-check is already recorded in the closing note. `docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md:153-171`, `:196-217`.

## Reasoning

**R2-Q2.** The comparator requires all 15 identity fields to compare equal and, by default, requires granted credit and matching certified, linked and required swept intervals in each bundle. The regressions confirm that an absent digest plus a different credit interval returns 2, missing intervals in both return 2, and shared false credit returns 2. The stated precedence is implemented. The runbook comparison uses the default credit check. This establishes the gate for the supplied identity fields. The resolved-session evidence is addressed in Q3. `scripts/compare_engine_outputs.py:110-125`, `:412-458`, `:478-498`, `:530-549`, `tests/test_compare_engine_outputs.py:317-359`, `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:483-497`.

**R2-Q3.** The PWC extractor refuses half-matching coverage rows and requires a link with the matching coverage ID and interval. The miner looks up coverage by artifact hash, then checks generation, interval and trial dataset. It checks the link against that coverage and grants credit only with both rows. This closes the incorrect credit attachment in R1. The remaining session issue is evidence provenance. WATCHER checks each presented configuration against its own search before dispatch. The extractors build comparator inputs from the supplied params, and the miner's separately recorded session labels are receipt evidence. For the result review, use the two linked searches' effective session labels to establish actual session equality. `scripts/extract_pwc_bundle.py:356-401`, `:497-506`, `scripts/extract_miner_bundle.py:394-401`, `:464-493`, `:509-537`, `:630-640`, `agents/watcher_agent.py:2271-2302`, `utils/step1_search_ledger.py:175-192`.

**R2-Q5.** The original LIVE objection is addressed for the planned stopped run. The retained-WAL fixture recovered credit and its link from the stable copy, while the main-file-only read saw zero coverage rows. The journal, accounting, threshold, link, one-mode membership and in-tree output fixtures passed. The generation-file live-path check still runs after `read_generation()` has opened the sidecar and hashed and loaded the NPZ. A scratch symlink to a stand-in live sidecar triggered a recorded open before refusal. Resolve and check both files before their first read. C6's ledger description should also mention the stable off-tree copy for a retained WAL. `scripts/extract_pwc_bundle.py:132-155`, `:168-229`, `:299-353`, `:388-401`, `:410-446`, `:470-495`, `tests/test_extract_pwc_bundle.py:369-492`, `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:467-479`.

**R2-Q6.** The receipt now freezes the miner trial, publication and credit evidence, and the bundle builder rechecks the bound fields. The mixed-source regression refuses a different publication against the original receipt. B4a captures that receipt after the miner arm and before the PWC arm. C1 checks its hash. That ordering is the control. B6 alone accepts a different publication if it falls inside the miner trial duration. I reused the mixed-source fixture with a 60 s miner trial and the foreign publication 30 s after its commit. A freshly written mixed receipt and the subsequent bundle both succeeded. The README's claim that this requires concurrency is too strong. A later arm that publishes within that window can also pass. The true publication lag is **UNAVAILABLE** until a real miner arm finishes. `scripts/extract_miner_bundle.py:93-97`, `:452-462`, `:547-564`, `:578-596`, `tests/test_extract_miner_bundle.py:335-362`, `:576-584`, `docs/drafts/pwc_backup_stage4_v1/EXTRACT_MINER_README.md:116-126`, `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:372-397`, `:409-428`.

**R2-Q7.** A1 creates the two Optuna parents. A0, B1 and C1 check the shared halt flag. Both launch paths omit the clear option, and WATCHER returns before starting if halted. The miner launcher's stand-in test also passed with the flag left in place. `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:175-205`, `:333-336`, `:426-453`, `agents/safety/kill_switch.py:59-60`, `agents/watcher_agent.py:3069-3084`, `docs/drafts/pwc_backup_stage4_v1/stage4_miner_launch.diff:57-65`, `:106-116`, `tests/test_stage4_miner_launch.py:164-168`.

**R2-Q8.** The derived miner launcher refuses invalid params, a missing required key, overlapping or non-empty staging, and a present halt flag before the first gate. WATCHER receives the parsed value. The sampler path test reproduced the live default using only the manifest. Alpha's choice to omit the sampler addresses R1's live-state issue and fits this four-stripe comparison. Beta's R1 alternative would collect occupancy from the Stage 4 ledger and requires pre-creating its schema. The sampler's 25-worker target exceeds the planned four stripes per phase. Michael decides the sampler question in runbook item 4. `docs/drafts/pwc_backup_stage4_v1/stage4_miner_launch.diff:11-65`, `:74-116`, `scripts/gate12_concurrency_sampler.py:167-188`, `:246-252`, `:284-292`, `tests/test_stage4_miner_launch.py:58-103`, `:171-209`, `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:260-273`, `:600-607`.

**R2-Q9.** The four suites passed on the landed bytes. The added fixtures cover the R1 regressions in the comparator, extractors and derived miner launcher. The separate PWC array-order test remains the recorded suggestion. The launcher suite exercises the miner params block and static ordering. Finding 2 uses a separate PWC snippet check. `tests/test_compare_engine_outputs.py:313-359`, `tests/test_extract_pwc_bundle.py:369-492`, `tests/test_extract_miner_bundle.py:335-433`, `tests/test_stage4_miner_launch.py:58-103`, `:187-209`, `docs/exchange/to_beta/X-PWC-STAGE4_R2_ALPHA_CLOSING_NOTE_stage4_tooling.md:67-76`.

**R2-F1.** The runbook includes the five requested reference entries and the R1 review. Finding 1 above identifies the additional governing search-record reference. `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:40-45`.

## Finding 2 · OBJECTION · LIVE — PWC Phase C3 can start WATCHER after rejecting the params

A missing, malformed or wrong-route params file reaches `nohup` with an empty `--params` argument. C3 prints “PWC params invalid - stop” and runs `false`, followed by an independent `test` and an unconditional launch. I ran the exact C3 block in a scratch shell with `nohup` replaced by an argument recorder. All three invalid cases reached it with an empty argument. The valid control passed the PWC JSON.

WATCHER skips an empty argument and retains manifest defaults, which select RANGE-MINER and name the live staging directory. A6's correctly opened pinned search should then reject the missing invocation pins at the configuration check. That is a wrong launch request and a lost PWC arm. The stub exercised no dispatch and opened no live ledger. Make C3 stop on a failed directory change, failed validation or empty params before starting WATCHER, as the derived miner launcher does for params. `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:312-327`, `:439-453`, `agents/watcher_agent.py:2011-2038`, `:2284-2302`, `:4170-4193`, `agent_manifests/window_optimizer.json:263-276`, `docs/drafts/pwc_backup_stage4_v1/stage4_miner_launch.diff:53-56`.

## Evidence and limits

- I ran the four scripts sequentially with `~/venvs/torch` active in a temporary archive of `6eafad9`. The comparator passed **36/36**, the PWC extractor **32/32**, the miner extractor **31/31** with **26/26 mutants caught**, and the miner launcher **14/14**. The launcher check applied the diff and ran `bash -n`. Its SHA-256 matched Alpha's applied-launcher hash. The suite's `HEAD` comparison used a fixture commit created solely in the temporary Git repository. `tests/test_stage4_miner_launch.py:58-64`.
- I also ran the three scratch checks described in Q5, Q6 and Finding 2. Other launch, publication and search behavior above was checked by reading the cited paths. The real arms, deployed parity, real WAL contents, actual publication lag and output equality remain **UNAVAILABLE**.
- The miner bundle leaves the per-phase lists, per-mode bidirectional sets and distinct phase counts unavailable. Exit 3 leaves those fields open. Michael decides whether the partial comparison suffices for this first session. TFM's prediction-pool hit rate against `k/1000) on draws the sieve never saw remains a separate measurement. `scripts/extract_miner_bundle.py:301-325`, `:604-629`, `scripts/compare_engine_outputs.py:478-498`, `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:627-636`, `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md:9-16`.
