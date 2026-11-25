## Prior record

**X-E2E-GRADER-LANDED · R1 · Team Beta · 2026-11-24 PST**  
Review class: after-commit courtesy. Base read: `81c93867d19c68bbf36f3649859454b722206448`.

`git status --porcelain` at initial read:

```text
?? docs/exchange/to_beta/X-E2E-GRADER-LANDED_R1_ALPHA_REVIEW_REQUEST_grader_as_landed.md
```

**Finding 1 — ALREADY RECORDED governing material omitted from the request's prior record.** The request supplies searches, so the review proceeds. The landed grader cites three additional governing sources at `scripts/diagnostics/e2e_grader.py:17-25`. The November 13 addendum §E–F specifies the value, identity, pending and duplicate controls. The reverse-sieve realignment §4 supplies the measurement rules WP-3 adopts. S176 Phase C specifies causal walk-forward. The November 24 selfplay contract amendment also governs how development evidence can be used in the learning loop. Record these alongside WP-3 and Q5.  
`docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md:115-135`, `docs/exchange/from_beta/X-REVERSE-SIEVE_R2_BETA_REALIGNMENT.md:112-127`, `docs/TB_RULING_S176_WATCHER_KPI.md:692-727`, `docs/CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md:10-26`.

- **Searches run.** Repeated Alpha's `git grep -l -i -e e2e_grader -e WP-3 HEAD` over the catalog, rulings, ruling requests, proposals, chapters and audits, consolidated TODO, backlog, leads and exchange. Repeated the `DEFERRED.md` check. Searched `docs/` with `rg -i` for grader and Poisson-binomial terms, duplicate draws and identities, stale pools, cutoff, dataset publication and development mode. Read the applicable matches, the superseded request and the requested landing note in the local `e2e-grader` worktree. Read the Step 6 generator on main and the vote writer from the local `e2e-vote` ref for input provenance.
- **ALREADY RECORDED.** WP-3 and Beta's Q5 establish the measurement contract. The dataset identity contract requires unique `(date, session)` pairs, and the historical scraper duplicate problem is documented. The pending-to-new-digest limitation and the combined-index ordering concern are also recorded in the superseded request. `docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md:78-86`, `docs/exchange/from_beta/X-FINDINGS-PLAN_R2_BETA_REVIEW.md:69-87`, `docs/exchange/DEFERRED.md:8`, `docs/DAILY3_CONSUMER_CONTRACT_v1.md:435-449`, `docs/PROPOSAL_Unified_Agent_Context_Framework_v3_2_10.md:326-334`, `docs/exchange/to_beta/X-E2E-GRADER_R1_ALPHA_REVIEW_REQUEST_grader_first_form.md:114-128`.
- **NEW.** The measured consequence in the landed grader is that duplicate outcome identities at separate indices contribute separate graded rows. The landing and this courtesy review are also new. **CHANGES A RULING:** none.
- **Not searched.** Ser8 archives, private remote branches, rigs, ledgers, external project files and host material beyond the named local worktree context and frozen Run H inputs.

## Positions

| Question | Position | Severity |
|---|---|---|
| Q1 — Counting | NO OBJECTION | — |
| Q2 — Refusals | OBJECTION | LIVE — A frozen snapshot with a repeated date and session can pass the grader and count one physical draw twice, and the repository records duplicates from the existing scraper. |
| Q3 — Development mode | NO OBJECTION | — |

## Q1 — Counting

Step 6 emits integers in ranking order and can repeat values. The grader validates the raw values and derives `k_t` from the distinct set. It scores one membership hit per graded row, including outcome `000`, and sums `k_t / draw_space` over those rows. Its Poisson-binomial calculation groups equal probabilities into binomials, convolves them and sums the mass from the observed hit count upward. That gives the upper tail under the independent uniform null up to floating-point rounding. The known-count, duplicate-vote, zero and brute-force probability gates passed.  
`prediction_generator.py:835-850`, `scripts/diagnostics/e2e_grader.py:270-300`, `scripts/diagnostics/e2e_grader.py:622-644`, `scripts/diagnostics/e2e_grader.py:719-731`, `scripts/diagnostics/e2e_grader.py:1207-1216`, `tests/test_e2e_grader.py:164-223`, `tests/test_e2e_grader.py:428-446`.

I checked the frozen Run H A1 midday pool records. The 904 pools have distinct sizes 18, 19 and 20 on 15, 143 and 746 rows, respectively. This gives `Σk_t = 17,907` and `E = 17.907`. Using the previously reported 22 hits, the landed tail and an independent 45-digit decimal calculation both gave `0.192472067094678` to the shown precision. I used the recorded hit count for this arithmetic and did not replay the actual outcomes.  
`docs/drafts/e2e_v1/RESULT_E2E_GRADE_RUNH_20261123.md:70-110`, `scripts/diagnostics/e2e_grader.py:622-644`.

## Q2 — Refusals

**The grader accepts a duplicate outcome identity at two combined indices.** The dataset loader validates the draw number and appends each row without checking date/session uniqueness. The vote writer takes target metadata from that same indexed dataset. The grader then matches each record to its indexed row and checks duplicate pool records by `(arm_id, combined_index)`. Both indices therefore pass the identity and duplicate-record checks when the pools and snapshot carry matching hashes.  
`scripts/diagnostics/e2e_grader.py:304-326`, `scripts/diagnostics/e2e_grader.py:494-517`, `scripts/diagnostics/e2e_grader.py:570-604`, `scripts/diagnostics/e2e_grader.py:1207-1216`, `e2e-vote:scripts/diagnostics/e2e_vote_harness.py:287-301`, `e2e-vote:scripts/diagnostics/e2e_vote_harness.py:740-764`.

**Verified.** I copied the outcome row at index 250 to index 251 in a synthetic dataset, bound the pool records to the resulting snapshot and gave both targets a three-value pool containing the outcome. The grader exited 0 with `controls_ok=true`. It emitted two hits for the same date/session and included both rows in the 200 graded draws and the 0.600 expected hits. The current root dataset and the Run H snapshot were byte-identical and each had 18,068 unique date/session pairs. The fixture demonstrates the failure on a bad snapshot, with no affected Run H result observed.  
`scripts/diagnostics/e2e_grader.py:304-326`, `scripts/diagnostics/e2e_grader.py:1207-1216`, `scripts/diagnostics/e2e_grader.py:1239-1245`, `scripts/diagnostics/e2e_grader.py:1336-1342`.

**Why LIVE.** The scraper appends every parsed row, and the earlier full scrape recorded 721 duplicate entries before deduplication. A snapshot can carry the correct hash and row count while retaining a repeat. The publication verifier's row checks cover the count and endpoints. The scheduled scraper is held pending 6-P2, so the realistic exposure is a manually refreshed dataset frozen for grading with a repeated row. A repeat can raise or lower the reported hit rate according to the extra row's membership result and adds `k_t / 1000` to the reported expectation. It also gives the Poisson-binomial model an extra trial for the same outcome. Reject repeated `(date, session)` at dataset load, with both row indices in the error. A fixture with a pool for each index and matching snapshot digests should exit 2 before creating outputs. Any change to the grader bytes needs the walk-forward and Step 1 screen pins updated together.  
`daily3_scraper.py:44-69`, `daily3_scraper.py:87-100`, `docs/PROPOSAL_Unified_Agent_Context_Framework_v3_2_10.md:326-334`, `scripts/verify_dataset_publication.py:181-233`, `docs/TODO_CONSOLIDATED.md:129`, `docs/DAILY3_CONSUMER_CONTRACT_v1.md:435-449`, `scripts/diagnostics/e2e_grader.py:719-731`, `scripts/diagnostics/e2e_grader.py:1390-1396`, `utils/step1_screen.py:118-127`, `e2e-map:scripts/diagnostics/e2e_walk_forward.py:533-546`.

The standard refusal gates passed for malformed pool values or JSONL, stale target and binding fields, duplicate records for one index under the default rule, and a missing eligible pool. The declared alternate rules choose before grading. A future index goes to the pending bucket with `hit=null` and is outside the graded denominator. A pool bound to the old digest will be refused after a dataset refresh, as already recorded.  
`scripts/diagnostics/e2e_grader.py:434-452`, `scripts/diagnostics/e2e_grader.py:485-552`, `scripts/diagnostics/e2e_grader.py:576-618`, `scripts/diagnostics/e2e_grader.py:1255-1264`, `tests/test_e2e_grader.py:227-368`, `docs/exchange/to_beta/X-E2E-GRADER_R1_ALPHA_REVIEW_REQUEST_grader_first_form.md:114-116`.

## Q3 — Development mode

Before writing or printing, development mode clears the arm verdict fields, empties the summary's counted-positive list and replaces every arm headline with `DEVELOPMENT, NOT A COUNTED VERDICT.` The report title, summary, per-draw and optional detail rows carry the mode, and the output checksum list includes the marker file. The CLI refuses a development-declared block in counted mode. I ran the existing R22 gates plus a two-arm development case with a significant paired comparison. Both arm headlines carried the development prefix and all counted verdict fields stayed null or empty. The comparison reported its statistic in the development summary without issuing a verdict.  
`scripts/diagnostics/e2e_grader.py:1030-1062`, `scripts/diagnostics/e2e_grader.py:1076-1106`, `scripts/diagnostics/e2e_grader.py:1118-1144`, `scripts/diagnostics/e2e_grader.py:1175-1198`, `scripts/diagnostics/e2e_grader.py:1284-1312`, `scripts/diagnostics/e2e_grader.py:1332-1342`, `tests/test_e2e_grader.py:756-827`.

## Findings outside the numbered questions

2. **Landing record location.** The request cites `docs/drafts/e2e_v1/READY_GRADER_LANDING_20261124.md` as context. I found it in the local `e2e-grader` worktree. It is absent from base `81c9386`, although the note lists itself for landing and says it was written after the test run. The landed reference fixture is present and passes its pinned-hash and equivalence gate.  
   `/home/michael/tfm_worktrees/e2e-grader/docs/drafts/e2e_v1/READY_GRADER_LANDING_20261124.md:73-83`, `tests/test_e2e_grader.py:662-664`, `tests/test_e2e_grader.py:698-745`.

## Checks and limits

- Activated `~/venvs/torch`. On VM101 CPU, `tests/test_e2e_grader.py` passed **36/36**. The mutation runner's `--target grader` selection killed and credited **47/47** mutants with all baseline gates passing. I redirected suite fixtures and the mutation runner's output directory to temporary directories under `docs/exchange/from_beta/` and removed them afterwards. The other mutation targets were outside this review and were not rerun. `tests/test_e2e_grader.py:31-35`, `tests/mutate_e2e_grader.py:317-350`, `tests/mutate_e2e_grader.py:354-391`.
- The landed grader hash was `c7fe48bf102daa42b284713f4b2c2508d7f21c9613a01f063be443c881fd9ace` at both checks. The reviewed source and tracked fixtures matched HEAD. Read the prior Run H report and inspected its frozen pool metadata and dataset identities. A fresh grade of the actual outcomes and a pipeline or fleet run were outside this review.
- A single frozen block measures membership on that block. The learning loop includes repeated extraction and training across new draws, with walk-forward on untouched back draws as the promotion gate. This review concerns instrument behavior. `AGENTS.md:38-54`, `CLAUDE.md:43-66`, `docs/CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md:10-26`.
