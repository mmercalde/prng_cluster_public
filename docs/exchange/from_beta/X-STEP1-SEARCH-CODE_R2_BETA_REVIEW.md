# X-STEP1-SEARCH-CODE · R2 · BETA REVIEW

## Prior record

**Searches and reads in this session:**

- Read both checkouts' `AGENTS.md`, Alpha's R2 reply, the R1 request and Beta review, the implementation report, the stage-1 brief and the relevant search-record v1.1 sections, including its carried governing-record table. R2 incorporates R1's searches by reference; I checked that referenced section rather than treating “none new” as an absent record.
- Searched `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md` and `docs/exchange/` for failed jobs, partial sweeps, publication/link retry, pending-write failure, stranded searches, accumulated-dataset compatibility and schema definitions. Followed the design review's recovery requirement and the recorded Q6 deferral. Checked the completion/delegated-run constraints at relevant passages. Search displays were bounded; this is not an exhaustive reread of every hit or every carried record.
- Read the changed ledger and integration paths, their callers and tests, and the actual legacy coordinator's accounting and partition construction. Compared all thirteen packet hashes with the worktree and with the saved R1 hashes. Reviewed the tracked diff's scope against `7673924`.

| Record / item | Label | Disposition |
|---|---|---|
| Stage-1 brief S1–S9; v1.1 §§2.2–2.7; Beta's design and code R1 reviews | ALREADY RECORDED | Governing implementation and retry requirements. |
| R1 Q1: credit despite failed jobs | ALREADY RECORDED | R2 correction verified below; LIVE objection closed. |
| R1 Q5: publication succeeds but restart cannot repair the missing link | ALREADY RECORDED | R2 repairs the case where the pending row committed. Its disclosed pending-write residual still reproduces the stranded-search outcome. This is a remaining status of Q5, not a newly discovered finding. |
| R1 Q4: definition-level schema validation | ALREADY RECORDED | Suggestion adopted and verified. |
| R1 Q6: accumulated-dataset restriction; miner start guard; deferred budgets; Route-A refresh | ALREADY RECORDED | Existing limitations remain; this review adds no launch authority or new deferral. |

**No additional governing document omitted from the request's carried record was located.**

**Not searched/read:** live databases, live Optuna stores or accumulated generations; rigs or Proxmox hosts; ser8; private/remote repositories; project PDFs; the parallel `volume_limit` worktree. No external research and no complete reread of all nineteen carried governing records.

## Review header

- **Thread / round:** X-STEP1-SEARCH-CODE / R2.
- **Date / host:** 2026-11-14, VM101, as `michael`.
- **Class:** BEFORE-COMMIT, Step 1 launch path.
- **Reviewed base/worktree HEAD:** `76739247ef46a0f921ed89d684a209800923b2d7`, plus the uncommitted packet in `/home/michael/tfm_worktrees/step1_search`.
- **Main-tree HEAD:** `35f55032b587a2995049fa7f1ea13febf182b074`.
- **Authority:** Beta records positions; Michael decides. This is not run authorization.

Main-tree `git status --porcelain` at initial read:

```text
 M docs/exchange/DEFERRED.md
 M docs/exchange/LEDGER.md
?? docs/exchange/from_beta/X-STEP1-SEARCH-CODE_R1_BETA_REVIEW.md
?? docs/exchange/to_beta/X-STEP1-SEARCH-CODE_R1_ALPHA_REVIEW_REQUEST_search_record_stage1.md
?? docs/exchange/to_beta/X-STEP1-SEARCH-CODE_R2_ALPHA_REPLY.md
```

Worktree `git status --porcelain` at initial read:

```text
 M agent_manifests/window_optimizer.json
 M agents/watcher_agent.py
 M database_system.py
 M tests/test_s172_threshold_propagation.py
 M tests/test_seed_domain_cursor_amendment.py
 M tests/test_watcher_followup_repairs.py
 M window_optimizer.py
 M window_optimizer_bayesian.py
 M window_optimizer_integration_final.py
?? data/
?? docs/CLAUDE_CODE_REPORT_STEP1_SEARCH_RECORD_STAGE1.md
?? tests/test_step1_bayes_storage_path.py
?? tests/test_step1_search_record_stage1.py
?? utils/step1_search_ledger.py
```

**Anchor convention:** code/test paths below refer to the reviewed worktree. Line anchors were read this session. Governing documents and exchange files are in the main checkout.

## Positions

| Question | Position | Severity |
|---|---|---|
| 1 — R1 Q1: complete-interval evidence | NO OBJECTION | — |
| 2 — R1 Q5: publication/link recovery | OBJECTION | LIVE — a transient failure of the new pending INSERT after the history commit still strands a published one-trial chunk, with both supported child restart forms refusing recovery. |
| 3 — R1 Q4: schema definitions | NO OBJECTION | — |
| 4 — previously unopposed R1 positions | NO OBJECTION | — |

## 1 — R1 Q1 closed within the stated route contracts

The legacy guard requires integer accounting, at least one planned job, all planned jobs successful, zero permanent failures and no nonempty failed-job list (`window_optimizer_integration_final.py:1406-1423`). Each phase is evaluated separately before receiving an interval (`:2261-2279`). The coordinator supplies those counters from its progress state (`coordinator.py:2193-2222`); its job plan offsets and clips every partition to the requested range (`:2308-2316`, `:2365-2379`), with the caller supplying that start and count (`window_optimizer_integration_final.py:1973-1978`). This is accounting evidence, rather than merely evidence that the call returned.

**Executed:** Alpha's real legacy evidence-producer/publication case passed. I additionally ran six isolated cases through the real legacy `run_bidirectional_test` and publication caller over `[1000,2000)`: a successful four-phase control with zero bidirectional survivors; one partial job failure in each phase separately; and a skipped fourth phase. The control produced one coverage row and one link. Every incomplete case raised `Step1NoCreditError` and produced neither. The intercepted coordinator calls received start 1000/count 1000. No worker or sieve ran. The phase-3 failure also caused phase 4 to be skipped, as the existing branch specifies (`:2159-2177`); that case does not isolate phase 3 from phase 4.

PWC and ZMQ now explicitly return empty `swept_intervals` (`window_optimizer_integration_final.py:1397-1403`, `:1899-1904`, `:1961-1965`), which cannot satisfy required-phase credit (`utils/step1_search_ledger.py:1004-1021`). The unit gate and branch-source checks passed. **The real PWC/ZMQ builders were not executed.** The miner's committed/contiguous-done-stripe evidence path remains present (`window_optimizer_integration_final.py:1426-1470`); its CPU fixture passed, without a miner launch.

I found no route that still earns credit on the R1 failed-job evidence. This closes R1 Q1. These checks measure traversal accounting, not hit rate or predictive value.

## 2 — R1 Q5 partly fixed; the disclosed pending-write residual remains LIVE

**Verified correction:** with a committed pending row, failure of `record_chunk_link` is repaired by a fresh process through WATCHER's real `run_step(1)`: one coverage row, one repaired link, zero dispatched commands, no replacement generation. The child-preflight repair and stale-chunk refusal also passed. These are the executed gates at `tests/test_step1_search_record_stage1.py:921-996`, exercising lookup repair (`utils/step1_search_ledger.py:579-588`) and child repair before cursor/budget checks (`window_optimizer_integration_final.py:1519-1528`, `:1542-1569`).

**No weaker repair door located for legitimately recorded pending rows.** Pending creation performs the original witness/coverage comparisons and the same search/chunk checks as direct linking (`utils/step1_search_ledger.py:828-856`, `:901-937`). Repair reads the certified history row by coverage id, checks generation against the pending row, reuses `_link_problems`, and copies the persisted counts (`:939-973`). The immutable pending row has the same append-only protection as the other tables (`:63-64`, `:322-350`, `:444-448`). The generation/mode/interval/family/non-witness refusal cases also passed when directed at pending creation. The repair does not require a new caller to assert publication fields.

**Remaining measured failure:** the history call precedes the new pending call (`window_optimizer_integration_final.py:3448-3452`, `:3466-3476`), and the pending row commits in a separate transaction (`utils/step1_search_ledger.py:858-885`, `:914-918`). I injected `sqlite3.OperationalError` at `_insert_idempotent` only for `T_PENDING`, after the real pending validation, then removed the fault. This exercised the real optimization, finalizer, coverage write and pending-validation path, with only the sieve intercepted:

```text
after failure:       coverage=1, pending=0, links=0, gap=0
fresh WATCHER:       one captured command for the same gap; no actual dispatch
fresh selector:      state=gap, gap=0
fresh child retry:   Step1SearchPreflightError (search already owns its study)
named-study resume:  Step1ChunkBudgetError (completed=1, budget=1)
after both retries:  coverage=1, pending=0, links=0, gap=0; same generation
objective calls on both retries: 0
```

The fresh WATCHER check supplied matching pruning configuration; its command was intercepted. Repair returns immediately when no pending row exists (`utils/step1_search_ledger.py:945-951`). The subsequent refusals are the study-ownership and consumed-budget walls (`window_optimizer_integration_final.py:1558-1569`, `:1489-1502`). They correctly prevent rerunning or inventing credit, but cannot recover this publication.

Alpha explicitly discloses this residual in the R2 reply. I am not reporting it as a new finding. It means the broader R1 Q5 recovery requirement is only partially met. A successful preceding write to the same database does not make the next transaction infallible; this test establishes the consequence of a transient failure, not its observed frequency on live state. No hostile process, different configuration or new deployment is required for this failure mode.

**Required correction:** make the checked publication-to-chunk recovery information durable or reconstructible even when this pending INSERT fails after the history row. Preserve the existing certification door, original link validation and budget refusal. Add the failed-pending-write case, followed by recovery in a fresh process, alongside the successful-pending/failed-link control. The acceptance outcome remains exactly one coverage row and link, zero new trials and no replacement generation. This is the existing v1.1 §2.3 / S8 retry requirement, not stage-2 budget work.

## 3 — R1 Q4 suggestion adopted

Definition comparison now covers each required table and both append-only triggers using `sqlite_master.sql`, in addition to version and column checks (`utils/step1_search_ledger.py:460-491`). It compares whitespace-normalized stored DDL against the module's own definitions (`:348-354`). The fresh-schema control and same-name/ineffective-trigger and wrong-column-type refusals passed (`tests/test_step1_search_record_stage1.py:811-833`); removing the definition check made the mutant red. No remaining objection to the requested definition-level check. This is validation of the module's required schema objects, not a claim that every possible unrelated schema object is prohibited.

## 4 — the other R1 NO OBJECTION positions remain unchanged

The saved R1 snapshot confirms only the ledger, integration file, new search suite and report changed for R2. The other nine source/test/manifest hashes are identical to R1. The expanded search suite passed its scheduling, identity, authority, configuration, parameter-hop and interval-guard cases. Repair inserts only a validated link; it neither opens a search nor supplies stored pins (`utils/step1_search_ledger.py:939-973`). The tracked diff still does not edit the protected coverage ledger, finalizer, `miner/`, kernels or registry.

Accordingly R1 Q2, Q3 and Q7–Q13 remain NO OBJECTION within their R1 limits. In particular, Q10's agreement with the stage split remains conditional on completing stage 1; it does not defer the remaining Q5 repair gap. R1 Q6 remains the recorded THEORETICAL objection, as Alpha acknowledges; its accumulated-dataset restriction is not closed by this round.

## Execution and limits

All executions activated `~/venvs/torch`, disabled bytecode writes and set `CUDA_VISIBLE_DEVICES=''`. Source files were read in place. The suite's scratch assignment and fresh-child invocation were redirected **in memory** to `/tmp/beta_step1_r2/`; assertions and repository sources were not edited. The fresh child used the same redirected harness and write audit. Python audit hooks refused file mutations and SQLite connections outside `/tmp`. No live database was opened and no production pipeline, worker, fleet, rig connection, installation, commit or push occurred. WATCHER commands were captured by its test dispatch stub.

| Check actually run by Beta | Result |
|---|---|
| Revised stage-1 suite | 37/37 gates; 30/30 mutants detected; completion sentinel PASS |
| Real legacy four-phase control and individual partial failures | Control credited; each incomplete case refused with no coverage or link |
| Pending creation with wrong witnesses | Generation, modes, interval, family and non-witness refused |
| Pending INSERT failure, then fresh-process lookup and both child restart forms | Remaining stranded publication reproduced |
| Full remaining regression set, GPU execution, Run 4, refreshed Route-A provenance | UNAVAILABLE in this review; Alpha's reported regression is not an independent Beta pass |

Harnesses and logs: `/tmp/beta_step1_r2/run_suite.py`, `suite.log`, `check_phases.py`, `phases.log`, `check_residual.py`, `fresh_lookup.py`, `residual_final.log`. The suite's labelled clean-tree/finalizer intercept and sieve stubs remain test limits. Nonfatal missing-schema diagnostics from the legacy wrapper are in the logs; publication and ledger outcomes were checked separately.

All thirteen packet hashes matched at the initial read and remained unchanged at the final recheck. Main-tree status before writing was unchanged from the header; this review is the only repository file I created.

## Findings outside the numbered questions

None. The pending-write residual is already disclosed and is addressed under Q5. **R1 Q1 is closed; R1 Q5 remains the one LIVE objection.**
