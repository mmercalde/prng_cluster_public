# X-STEP1-SEARCH-CODE · R1 · BETA REVIEW

## Prior record

**Searches run in this session:**

- Read `AGENTS.md`, the request, the worktree's implementation report, the stage-1 brief, v1.1 of the search-record proposal and Beta's X-STEP1-SEARCH R1 design review. Compared the worktree's `AGENTS.md` with the main tree's: identical.
- Enumerated governing documents with `rg --files docs`; searched `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md` and `docs/exchange/` for failed jobs/chunks, incomplete/partial sweeps, sweep evidence, publication/link retry, cross-dataset accumulation and schema validation. Also searched the Chapter 1 audit, corrections report and Step 1 Stage A report. Search displays were bounded; this does not claim an exhaustive reading of every hit.
- Checked the brief/proposal's carried prior record against the completion plan, delegated-run restrictions, Route-A containment and R2 sequencing. Followed source references into the coverage ledger, finalizer, dataset authority, legacy coordinator, PWC and RANGE-MINER assignment path. Searched production callers of `record_chunk_link` and construction/recovery of publication witnesses.
- Read the full tracked production diff against `7673924`, both new suites, relevant changed fixture passages and the production anchors cited below. Checked the twelve source/test/manifest hashes in Alpha's landing list: all match the reviewed files.

| Governing record / item | Label | Disposition |
|---|---|---|
| Stage-1 brief S1–S9 and v1.1 §§2.1–2.7 | ALREADY RECORDED | Implementation requirements, including the proposal's precedence when the brief differs. |
| Beta design review Q4 and Q11 | ALREADY RECORDED | Completed traversal is required; failure/retry evidence must exercise the production path. |
| Route-A ruling and R2; S167; D6.2 and threshold constraints carried by the brief/proposal | ALREADY RECORDED | Remain constraints; no new launch authority. |
| Cross-dataset accumulator/input-wall problem | ALREADY RECORDED | Stage A P7 and Beta design review Q6 already record it; v1.1 §2.6 requires a restriction pending the wall. Q6 below is its current status, not a new finding. |
| Miner origin-0 defect, two temporary guards, Route-A refresh, stage-2 budget limitation | ALREADY RECORDED | Disclosed by this request/report; not rediscovered findings. |
| New evidence can certify failed legacy jobs; ordinary retry cannot repair a failed publication link | NEW | Implementation gaps measured below under Q1 and Q5. |
| Search scheduling replaces the historical cursor's scheduling authority | CHANGES A RULING | Already accepted in v1.1; this review does not reopen that design. |

**No additional governing record omitted from the request's carried record was located.** In particular, the dataset restriction was included by reference; the issue is its implementation, not a missing record.

**Not searched/read:** live databases or Optuna stores, live accumulator artifacts, rigs, Proxmox hosts, ser8, private/remote repositories, project PDFs or the contents of the parallel `volume_limit` worktree. No external research. No complete reread of all 19 carried records or all chapters.

## Review header

- **Thread / round:** X-STEP1-SEARCH-CODE / R1.
- **Date / host:** 2026-11-14, VM101, as `michael`.
- **Class:** BEFORE-COMMIT, WATCHER and Step 1 launch path.
- **Reviewed base:** `76739247ef46a0f921ed89d684a209800923b2d7`, plus the uncommitted files in `/home/michael/tfm_worktrees/step1_search`.
- **Main-tree HEAD:** `35f55032b587a2995049fa7f1ea13febf182b074`.
- **Authority:** Beta records review positions. Michael decides. This is not run authorization.

Main-tree `git status --porcelain` at initial read:

```text
?? docs/exchange/to_beta/X-STEP1-SEARCH-CODE_R1_ALPHA_REVIEW_REQUEST_search_record_stage1.md
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

**Anchor convention:** all code/test/manifest paths below are relative to the reviewed worktree, not the main tree. Governing `docs/` references are available in the main tree. Line numbers were read in this session.

## Positions

| Question | Position | Severity |
|---|---|---|
| Q1 — credit | OBJECTION | LIVE — permanent legacy job failures can produce a COMPLETE trial, a certified coverage row and search completion without a completed sweep. |
| Q2 — scheduling | NO OBJECTION | — |
| Q3 — authority | NO OBJECTION | — |
| Q4 — ledger | SUGGESTION | — |
| Q5 — publication link | OBJECTION | LIVE — a transient link failure after a one-trial publication strands the search because both normal restart and named-study resume refuse before link repair. |
| Q6 — refusals | OBJECTION | THEORETICAL — the missing accumulated-dataset check requires a different dataset from the retained generation, which is outside the same-dataset Run 4 described in this packet. |
| Q7 — three parameter hops | NO OBJECTION | — |
| Q8 — fixtures and fallback | NO OBJECTION | — |
| Q9 — protected surfaces | NO OBJECTION | — |
| Q10 — stage boundary | NO OBJECTION | — |
| Q11 — governing records | NO OBJECTION | — |
| Q12 — storage-path behavior | NO OBJECTION | — |
| Q13 — miner finding and interval guards | NO OBJECTION | — |

## Q1 — S7 accepts requested work as evidence of completed work on other routes

**Measured:** in an isolated search over `[0,1000)`, constant mode, one Optuna trial, the real legacy `run_bidirectional_test` received this intercepted result on **both** forward and reverse calls:

```python
{"metadata": {"total_jobs": 1, "successful_jobs": 0, "failed_jobs": 1},
 "results": [], "failed_jobs": [("j", "worker timeout")]}
```

The real optimization/publication path nevertheless wrote one certified coverage row, one link, and returned `search_complete` on selection. No sieve, worker or GPU ran. The interception substitutes a result shape that the actual coordinator returns after permanent failures (`coordinator.py:2193-2222`, `:2244-2249`).

**Cause:** the legacy route extracts surviving records, without checking failed/missing jobs (`window_optimizer_integration_final.py:125-164`, `:1986-1987`, `:2017`), then stamps phases 1 and 2 with the **requested** interval simply because those calls returned (`:2223-2240`). The pure credit decision checks that supplied evidence against the requested interval and accepts it (`utils/step1_search_ledger.py:874-897`); publication and coverage then follow (`window_optimizer_integration_final.py:3326-3360`, `:3405-3409`). Optuna COMPLETE is not a second proof of successful jobs.

The PWC/ZMQ additions have the same evidentiary weakness: they manufacture every required phase's interval from the handed arguments (`window_optimizer_integration_final.py:1397-1400`, `:1869-1872`, `:1930-1933`). PWC reports `failed_chunks` at pass level (`persistent_worker_coordinator.py:1508-1548`), but its trial builder consumes survivor maps and emits `pruned=False` without carrying that completion accounting (`:1725-1727`, `:1766-1768`, `:1881-1898`). This PWC consequence is a source inference; the executed end-to-end counterexample is the legacy route. The PWC hybrid quarantine remains in place (`persistent_worker_coordinator.py:195-217`); this is not a request to lift it.

**Required correction:** accept phase evidence only when the route establishes successful completion of the whole interval. If a route cannot supply that evidence within this brief, refuse certification for that route. Add an executed failed-job/partial-completion case through the real evidence producer and publication caller, with a successful zero-survivor sweep as the paired control. The existing suite replaces `run_bidirectional_test` with preconstructed evidence (`tests/test_step1_search_record_stage1.py:292-335`), so its green S7 cases do not establish this property.

This is a traversal-accounting defect, not a verdict about a filter or predictive value. The CPU check measures no hit rate; false coverage makes subsequent hit-rate attribution and the claim of completed traversal unreliable.

## Q2 — the search cursor is used, including zero

The selector uses the imported first-gap calculation over linked intervals inside the search range (`utils/step1_search_ledger.py:517-558`). WATCHER assigns that result unconditionally, including zero, clips the chunk at the declared end and supplies the selected id (`agents/watcher_agent.py:2292-2297`). Lookup failure, no open search and completion have separate result forms (`:2195-2242`); only the two clean-stop values bypass pipeline escalation (`:3122-3137`). Completion is recorded as an append-only event (`utils/step1_search_ledger.py:546-550`).

The search suite's three-state cases, F5a and Run 4 command-shape case passed, as did the changed cursor suite. Q1 concerns whether credit feeding that cursor is valid, not its gap arithmetic.

## Q3 — no new production opening or pin authority found

The new CLI handles opening and exits before pipeline execution (`agents/watcher_agent.py:801-880`, `:3941-3943`). Opening reads the frozen dataset and the existing operator pin split; it does not reconstruct pins from the ledger (`:813-815`, `:842-868`). The run path still injects pins only from the invocation bundle (`:2031-2033`) and compares that presented configuration against the record (`:2274-2277`).

The no-opening-from-params/environment/record and no-inherited-pins cases passed. The Python ledger API is an implementation surface; I found no added autonomous production caller of its opener.

## Q4 — append-only and transactional opening; qualify “exact schema”

The module's writes are inserts, its own connections enable recursive triggers, and its UPDATE/DELETE triggers abort (`utils/step1_search_ledger.py:332-334`, `:412-415`). The one-open check, search insert and opening event share `BEGIN IMMEDIATE` (`:625-669`). The suite's append-only, racing-opener and schema cases passed. The new module does not write the historical coverage table; its link requires the existing witness and checks its family/modes/range/generation against the chunk and search (`:774-815`).

**Suggestion:** strengthen or qualify the exact-schema claim. Validation currently compares column **names** and the presence of trigger **names** (`:440-457`), not column types/constraints or trigger bodies. A same-named ineffective trigger would satisfy that check. I did not establish such a malformed schema on this system, so this is not a LIVE objection or a claim that the fresh schema is ineffective. Add a same-name/wrong-definition case if exact definition validation is intended.

## Q5 — helper idempotence passes, production recovery does not

The link derives publication fields and checks consistency; retrying the helper with the same objects is idempotent (`utils/step1_search_ledger.py:781-837`). A failure at the production call propagates after publication, leaving the linked-interval cursor unmoved (`window_optimizer_integration_final.py:3423-3428`; `utils/step1_search_ledger.py:517-528`). Those parts are correct.

**Measured missing behavior:** after a one-trial chunk published successfully and its link raised an injected transient `Step1LinkError`, the database held one coverage row, zero links and first gap 0. After removing the fault:

- Repeating the ordinary fresh invocation raised `Step1SearchPreflightError`: the search already owns the first study (`window_optimizer_integration_final.py:1528-1539`).
- Supplying that actual study name with resume enabled raised `Step1ChunkBudgetError`: completed=1, budget=1 (`:1466-1479`, called at `:2340-2341`).

Neither route reaches link repair. The only production `record_chunk_link` call found is after a new optimization and finalization (`:3424`). The existing test avoids this problem by retaining the artifact and coverage in a test-local `held` dictionary and directly calling the ledger helper twice (`tests/test_step1_search_record_stage1.py:815-847`). That proves helper idempotence, not retry after the child exits.

**Required correction:** provide a durable way to recover and validate the same publication witness/coverage/chunk/trial counts and retry only the missing link before the no-new-trials refusal. Keep S7 and the budget refusal for genuinely unevidenced work. Prove recovery through the supported production entry point in a fresh process, with zero additional objective calls, no replacement generation and exactly one link. This implements the already-required retry contract in v1.1 §2.3; it does not require the stage-2 per-chunk budget redesign.

## Q6 — comparisons work against the search; the recorded accumulator restriction is absent

WATCHER compares the frozen run's version/digest against the search and compares the declared configuration field by field, naming presented and recorded values (`agents/watcher_agent.py:2258-2289`; `utils/step1_search_ledger.py:148-219`). The child repeats dataset/search identity checks (`window_optimizer_integration_final.py:1499-1511`). The suite's configuration mutations and moved-pointer case passed.

**ALREADY RECORDED status:** neither comparison establishes that the selected search's dataset equals the dataset behind the accumulated generation. The finalizer still loads and merges the prior generation without a dataset input (`utils/run_finalizer.py:1636-1641`, `:1662-1678`). v1.1 §2.6 explicitly requires refusal pending the input wall; its §9 schedules the cross-dataset decision before the first search on newer draws. The stage-1 brief gives the proposal precedence where they differ.

An isolated check published an empty generation on fixture A, froze a scratch fixture B with one changed draw and a different version/digest, opened a search on B, and ran one stubbed COMPLETE sweep. It published and linked B with A as its parent. This verifies absence of the restriction; it does **not** claim stale nonempty rows were measured or that the live Run 4 dataset has changed. No live dataset/accumulator was read to establish that condition. Hence THEORETICAL for this packet's same-dataset Run 4; it holds nothing. Preserve the restriction as an explicit open requirement before any different-dataset search is dispatched, rather than treating search/run equality as its closure.

## Q7 — all three hops are present

The manifest declares the default/argument mapping (`agent_manifests/window_optimizer.json:64`, `:282`, `:341`); the CLI parses and forwards it (`window_optimizer.py:1524`, `:952`, `:1941`); integration accepts it (`window_optimizer_integration_final.py:2275`). WATCHER rejects a dropped/mismatched built argument (`agents/watcher_agent.py:2428-2446`). The executed three-hop and dropped-id cases passed.

## Q8 — changed fixtures are justified; scheduled red controls remain statuses

The changed cursor fixtures explicitly open isolated searches for dispatch cases, replace an obsolete history-COMPLETE expectation and add the paired history-COMPLETE/open-search case (`tests/test_seed_domain_cursor_amendment.py:305-326`, `:425-466`, `:733-792`, `:1649-1671`). The follow-up fixture tests both clean outcomes and uses lookup failure as the escalating near miss (`tests/test_watcher_followup_repairs.py:186-205`, `:316-355`). The threshold fixture adds the real chunk state to extracted closures without an Optuna trial (`tests/test_s172_threshold_propagation.py:248-254`, `:342-344`, `:850-852`; early-return hooks at `window_optimizer_integration_final.py:1563-1564`, `:1579-1580`). Their relevant executed checks passed.

No obsolete historical-cursor fallback was found in the changed scheduling block (`agents/watcher_agent.py:2167-2310`). Alpha's reported Route-A fixture failures pending the scheduled evidence refresh are **ALREADY RECORDED**, not fresh implementation findings or passing controls. That refresh and the post-commit clean-tree checks still need their own evidence before Run 4, as v1.1 §8 and Route-A R2 require.

## Q9 — protected code was not edited

The tracked diff contains the nine listed modified files, plus the separately inspected new ledger and tests. No diff touches the coverage ledger, finalizer, `miner/`, kernels, registry or array contract. The new gate is in the caller before finalization (`window_optimizer_integration_final.py:3306-3360`); existing checkpoint preparation and threshold resolution remain in the call path (`:3007-3018`, `:3068-3069`). Pin authority remains as described under Q3. This is a source/diff position, not a claim that I independently reran every D6.2 or GPU regression suite. The S7 behavioral restriction on NP2 is disclosed in Alpha's report, not an undisclosed change.

## Q10 — no objection to the agreed split, subject to completing stage 1

Run 4's proposed search has one chunk and one trial (v1.1 §2.9; `tests/test_step1_search_record_stage1.py:58-62`, `:1249-1274`). The disclosed stage-2 per-chunk budget work is not needed to make that shape truthful. Q1 and Q5 are stage-1 completion gaps; they must not be deferred under that label. Q6 remains the recorded different-dataset restriction.

The temporary miner guard still deliberately blocks the proposed nonzero-start Run 4 (`window_optimizer.py:698-703`, `:807-808`). The separate `base_start` work and Route-A evidence refresh remain prerequisites already disclosed by Alpha. This review neither removes that guard nor treats Run 4 as launch-ready.

## Q11 — no omitted governing record located

See Prior record. No new finding about a missed governing document. The known constraints remain statuses even where this implementation has not closed them.

## Q12 — the storage-path fix preserves main-tree behavior

The only Bayesian-file changes are the import-time checkout base and the two URL expressions (`window_optimizer_bayesian.py:23`, `:674`, `:693`). The AST-based main-tree comparison passed for absolute, relative and dot-relative `__file__`, both URLs, and the import-time/chdir and intercepted Optuna-call checks also passed. All four storage mutants were detected. This supports unchanged URL behavior when imported from the main checkout; it does not claim behavior is unchanged when deliberately running another checkout, which is the purpose of the fix.

## Q13 — the origin-0 finding and temporary interval guards are correct within their stated scope

By source, the miner call passes `total_seeds` without a start (`window_optimizer_integration_final.py:1688-1698`); `serve_trial` calls `assign_stripes` without `base_start` (`miner/range_miner_coordinator.py:10324-10326`), whose default is 0 (`:3993-4000`). That supports Alpha's finding. I did not rerun Alpha's fake-worker experiment or a real sweep.

The CLI/wrapper guard precedes coordinator creation (`window_optimizer.py:807-815`); its executed refusal/mutant passed. Independently, the miner evidence reader requires committed trial evidence and contiguous `done` stripe rows (`window_optimizer_integration_final.py:1403-1442`), and S7 requires each required phase's interval to equal the request (`utils/step1_search_ledger.py:879-891`). The interval-mismatch and miner-evidence suite cases passed. These guard the stated wrong-origin miner credit until the separate brief lands. Q1 shows why this must not be widened into a claim that every route proves complete traversal.

## Execution and limits

All test/check executions used `source ~/venvs/torch/bin/activate`, `PYTHONDONTWRITEBYTECODE=1` and `CUDA_VISIBLE_DEVICES=''`. All created state was under `/tmp`; no live ledger was opened. Dispatch/sieves were intercepted. No production pipeline, worker, fleet, rig connection, installation, commit or push occurred; pipeline-method tests intercepted dispatch.

| Check actually run by Beta | Result |
|---|---|
| Search-record stage-1 suite | 33/33 gates; 25/25 mutants detected |
| Storage-path suite | 3/3 gates; 4/4 mutants detected |
| Changed cursor suite | 41/41 |
| Changed WATCHER follow-up suite | 17/17 |
| Changed threshold Route-A gate / M1 | Sampled 0.73/0.31 forwarded; mutation to 0.30/0.30 detected |
| Failed legacy jobs through real optimization/publication | Counterexample confirmed: 1 coverage row, 1 link, search complete |
| Production retry after injected link failure | Counterexample confirmed: both restart forms blocked; coverage=1, links=0, gap=0 |
| Different-dataset scratch search | Known restriction absent: 1 objective call, linked publication with A parent |
| Full remaining regression set, actual GPU/kernel execution, Run 4, refreshed Route-A provenance | UNAVAILABLE in this review; not reported as pass or zero |

Harness details and logs are in `/tmp/beta_step1_review/`: `suite.log`, `storage.log`, `cursor2.log`, `followup.log`, `threshold_fixture.log`, `reproduce_final.log` and their runners. To obey Beta's write restriction, the search suite's scratch assignment was redirected **in memory** to `/tmp`; the follow-up suite's initial cwd was redirected there. Repository sources and assertions were not edited. The cursor suite was rerun from its documented checkout cwd after a scratch-cwd attempt could not resolve relative manifests. An audit hook rejected Python file mutations outside `/tmp`; bytecode writes were disabled. The finalizer clean-tree intercept and the stubbed sweeps are the same labelled limits as the supplied integration harness. The failed-job check additionally used the real legacy evidence producer. Nonfatal missing-schema diagnostics from the legacy integration wrapper appear in its log; publication and database outcomes were measured independently.

Final source-hash recheck: all twelve reviewed source/test/manifest files and the report were unchanged from the review snapshot; all twelve still match Alpha’s landing hashes. The main-tree status before writing remained the initial request-only status. After writing, its only additional entry is this review.

## Findings outside the numbered questions

None. Q1 and Q5 are new implementation objections within the requested scope. Q6, the miner-start defect, deferred budgets and scheduled red controls are recorded statuses, not additional findings. **Q1 and Q5 are the two LIVE objections.**
