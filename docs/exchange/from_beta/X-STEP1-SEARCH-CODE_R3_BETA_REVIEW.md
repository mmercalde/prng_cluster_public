# X-STEP1-SEARCH-CODE · R3 · BETA REVIEW

## Prior record

**Searches and reads in this session:**

- Read both checkouts' `AGENTS.md` (identical), Alpha's R3 reply, the R1 request, the R2 reply and Beta's R1/R2 reviews. R3 incorporates R1's searches by reference; I checked that section. Read the stage-1 brief, relevant v1.1 proposal sections and its carried governing-record table, the implementation report's relevant sections, the D6.2 identity addendum, and relevant completion/delegated-run constraints.
- Searched `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md` and `docs/exchange/` for study identity, reconstruction, pending-write failure, stranded publications, publication/link retry and accumulated-dataset compatibility. Checked `LEDGER.md` and `DEFERRED.md`. Search displays were bounded; the focused recovery search is saved in `/tmp/beta_step1_r3/prior_search.log`.
- Read the reconstruction, pending/direct-link checks, history certification door, study binding, chunk-open, child preflight, WATCHER lookup and relevant tests. Checked the tracked diff's scope against `7673924`, all thirteen R3 hashes, and the saved R2 hashes.

| Record / item | Label | Disposition |
|---|---|---|
| Stage-1 brief S1–S9; v1.1 §§2.2–2.7; Beta design and R1/R2 code reviews | ALREADY RECORDED | Governing implementation and recovery requirements. |
| R2 Q2 / R1 Q5 pending-write residual | ALREADY RECORDED | Existing LIVE objection; R3 correction verified below and objection closed. |
| Reconstruction and search-chunk study provenance | NEW | R3 implementation of that existing recovery requirement. |
| R3's disclosed outside-search reuse of the search's own study | ALREADY RECORDED | A limit of the binding, addressed under Q2; not a new finding. |
| R1 Q6 accumulated-dataset restriction; stage-2 budgets; miner-origin guard; Route-A refresh | ALREADY RECORDED | Existing statuses remain. No new deferral or launch authority. |

**No additional governing document omitted from the request's carried record was located.**

**Not searched/read:** live databases, live Optuna stores or accumulated generations; rigs or Proxmox hosts; ser8; private/remote repositories; project PDFs; the parallel `volume_limit` worktree. No external research or complete reread of all nineteen governing records.

## Review header

- **Thread / round:** X-STEP1-SEARCH-CODE / R3.
- **Date / host:** 2026-11-14, VM101, as `michael`.
- **Class:** BEFORE-COMMIT, Step 1 launch path.
- **Reviewed base/worktree HEAD:** `76739247ef46a0f921ed89d684a209800923b2d7`, plus the uncommitted packet in `/home/michael/tfm_worktrees/step1_search`.
- **Main-tree HEAD:** `35f55032b587a2995049fa7f1ea13febf182b074`.
- **Authority:** Beta records review positions; Michael decides. This is not run authorization.

Main-tree `git status --porcelain` at initial read:

```text
 M docs/exchange/DEFERRED.md
 M docs/exchange/LEDGER.md
?? docs/exchange/from_beta/X-STEP1-SEARCH-CODE_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-STEP1-SEARCH-CODE_R2_BETA_REVIEW.md
?? docs/exchange/to_beta/X-STEP1-SEARCH-CODE_R1_ALPHA_REVIEW_REQUEST_search_record_stage1.md
?? docs/exchange/to_beta/X-STEP1-SEARCH-CODE_R2_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-STEP1-SEARCH-CODE_R3_ALPHA_REPLY.md
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

**Anchor convention:** source/test paths below refer to the reviewed worktree. All cited source lines were read this session.

## Positions

| Question | Position | Severity |
|---|---|---|
| 1 — recovery after either pending-write or link-write failure | NO OBJECTION | — |
| 2 — reconstruction validation and chunk/search binding | NO OBJECTION | — |
| 3 — earlier NO OBJECTION positions | NO OBJECTION | — |

## 1 — both post-history failure cases recover; R1 Q5 is closed

**Verified by execution:** the full revised suite passed **39/39 gates and detected 33/33 mutants**, with its completion sentinel PASS. Both fresh-process WATCHER cases passed: a failed link after a committed pending row, and an injected `sqlite3.OperationalError` at the pending INSERT after its real validation. Each finished with exactly one history row and one link, zero dispatched commands and no replacement generation. A stale child invocation ran zero trials. These cases exercise the real optimization/finalizer/history caller with the sieve intercepted (`tests/test_step1_search_record_stage1.py:925-999`).

The durable chunk-open precedes the first sieve and records the actual Optuna study (`window_optimizer_integration_final.py:1590-1606`, `:3122`). Search publication now passes that study to the existing history door (`:3448-3456`). Repair runs before WATCHER chooses the gap (`utils/step1_search_ledger.py:582-595`) and before the child's cursor and budget refusals (`window_optimizer_integration_final.py:1522-1528`, `:1542-1569`). Thus the old refusal walls remain useful after recovery.

I additionally injected the pending failure and invoked the child directly, without a WATCHER lookup between failure and retry. Its preflight reconstructed the link and refused the stale interval with first gap 1000: one history row, one link, zero objective calls and the same generation directories. The suite independently covers the corresponding direct-child control for a failed link (`tests/test_step1_search_record_stage1.py:1035-1056`).

The pending path preserves recorded trial counts (`utils/step1_search_ledger.py:975-979`). Reconstruction writes **-1 for every trial count**, accurately marking unavailable counts rather than asserting one completed trial (`:1033-1037`). This does not implement stage-2 trial accounting.

“No caller-asserted field” here means no new repair-time assertions: repair reads persisted records. The initial study provenance is caller-supplied at publication, as the unchanged door explicitly permits (`utils/seed_coverage_ledger.py:881-907`). This is not a claim that the publication witness itself authenticates the study name.

## 2 — no weaker supported binding located; disclosed limit retained

The direct and pending writes require the publication witness, compare its interval, family, modes, generation and artifact digest with the history row, and check search/chunk compatibility (`utils/step1_search_ledger.py:819-853`, `:907-943`). The pending repair retrieves that exact history id and checks its generation against the validated pending record (`:960-979`).

Without a pending row, reconstruction considers only chunks with neither pending nor link. It reads certified history for the search's family, requires exact interval, modes, dataset digest and chunk study, excludes rows recorded before chunk-open and coverage ids already linked anywhere, and refuses multiple candidates. It reuses `_link_problems` and the idempotent insertion path, deriving generation from the selected immutable history row (`utils/step1_search_ledger.py:999-1037`; family filtering: `utils/seed_coverage_ledger.py:994-1018`). Timestamp parsing handles both ledgers' UTC formats (`utils/step1_search_ledger.py:369-377`). It creates a search link, not another certified history row.

**Executed beyond the suite:** synthetic certified-row fixtures separately excluded wrong dataset, modes, interval, family, another study, absent study, a pre-open row and an otherwise matching row already linked to another search. The matching control linked once; repeating repair added nothing. The suite's two-candidate refusal also passed (`tests/test_step1_search_record_stage1.py:1002-1032`). I additionally carried ambiguity through real WATCHER `run_step(1)`: `step1_search_lookup_failed`, zero dispatches and zero links, as its exception branch specifies (`agents/watcher_agent.py:2199-2212`).

**Limit, already disclosed:** matching provenance cannot distinguish a manual outside-search publication using this search's own study, interval, family, modes and dataset after chunk-open. An isolated synthetic row with exactly those values linked; I did not execute such a manual run. This follows from the actual predicates (`utils/step1_search_ledger.py:1013-1019`) and the child's explicit outside-search path (`window_optimizer_integration_final.py:1510-1511`). Therefore I do not give a universal claim against arbitrary manual reuse.

For supported search execution, a study owned by another search is refused, and the interval identifies the chunk within the search (`utils/step1_search_ledger.py:749-788`). WATCHER requires its selected search id to reach the command (`agents/watcher_agent.py:2428-2445`). No supported cross-search/chunk misbinding was found. The manual same-study case needs the additional operator action Alpha disclosed; it does not leave the measured R2 transient-write failure LIVE on the supported path.

## 3 — earlier NO OBJECTION positions stand within their recorded limits

Hash comparison with the saved R2 packet confirms only the ledger, integration file, search suite and report changed. The other nine files are identical. The tracked diff still excludes the protected history ledger, finalizer, `miner/`, kernels and registry.

The credit decision still requires COMPLETE state, evidence from this chunk, every required phase and exact interval (`utils/step1_search_ledger.py:1055-1086`), and executes before finalization (`window_optimizer_integration_final.py:3360-3393`). The legacy accounting and PWC/ZMQ no-credit evidence remain (`:1397-1423`). Definition-level schema validation remains (`utils/step1_search_ledger.py:467-494`). Their relevant revised-suite gates passed.

Accordingly the earlier NO OBJECTION positions, including R2 Q1 and Q3, stand. **R1 Q6 remains the recorded THEORETICAL objection/open accumulated-dataset requirement** in `DEFERRED.md`; this round does not close it. Stage-2 budget work and Route-A refresh retain their previous status. R1 Q5 / R2 Q2 is now closed; no LIVE objection remains in this review.

## Execution and limits

All execution activated `~/venvs/torch`, disabled bytecode writes and set `CUDA_VISIBLE_DEVICES=''`. The test scratch path and fresh-child harness invocation were redirected in memory to `/tmp/beta_step1_r3/`; assertions and repository sources were not edited. Parent and fresh-child audit hooks refused file mutations and database connections outside `/tmp`. Dispatch, notifications and the sieve remained intercepted. Synthetic witnesses above are ledger-level fixtures, not claims of actual GPU work.

| Check actually run by Beta | Result |
|---|---|
| Revised stage-1 battery, including fresh-process recovery controls | 39/39 gates; 33/33 mutants detected; completion sentinel PASS |
| Additional reconstruction exclusions and replay | Passed |
| Failed pending INSERT followed directly by child preflight | Repaired; stale interval refused; zero new objective calls/generations |
| Ambiguous history through WATCHER | Lookup failure; zero dispatches/links |
| Full remaining regression set, real PWC/ZMQ builders, GPU execution, Run 4, refreshed Route-A provenance | UNAVAILABLE in this review; Alpha's reported results are not independent Beta passes |

Harnesses/logs: `/tmp/beta_step1_r3/run_suite.py`, `suite.log`, `check_binding.py`, `binding.log`. The suite retains its labelled clean-tree intercept; nonfatal missing-schema diagnostics appear in its log. Publication and ledger assertions passed separately. No live database, pipeline, worker, fleet, rig connection, installation, commit or push occurred.

All thirteen R3 packet hashes matched initially and at final recheck. Main/worktree status before writing remained as recorded above. This review is the only repository file I created.

## Findings outside the numbered questions

None. Recorded limitations remain statuses, not new findings.
