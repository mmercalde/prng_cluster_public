# X-STEP1-SEARCH · R1 · BETA REVIEW

## Prior record

**Searches run in this session:**

- Read `AGENTS.md`, the complete request and proposal, and the relevant sections of the Stage A report and X-RUN4-RANGE R1 review.
- Enumerated relevant files with `rg --files docs`; searched `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md` and `docs/exchange/` for search identity, zero/completed trials, chunk budgets, strict-greater merge, session scope and input-manifest identity. Also searched `CHAPTER_1_AUDIT_v1.md`, the corrections report and the Chapter 10, 12 and 13 audits. Some broad search displays were bounded/truncated; this is not a claim to have read every matching document.
- Followed the checkpoint references into the D6.2 reconciliation instructions, bounded-repair instructions and identity addendum; searched their resume, ordinal, context and budget passages.
- Read the cursor amendment and R1/R2 instructions; completion-plan F1–F4 and sequencing; delegated-run restrictions; Route-A R2; S167; S176 walk-forward requirements; S179 consumer/chokepoint requirements and BOOTSTRAP; fact-file §2.10b; BACKLOG §5; and Chapter 13's rerun policy.
- Followed threshold provenance into Chapter 1 §7.2.1, the threshold-path audit and threshold-repair instructions. Read the dataset publication schema's lineage/input-wall passages.
- Read current source and harness anchors cited below, including the requested WATCHER, ledger, database, optimizer and publication passages. Searched the current dataset-freeze entry points and session-selection paths.

**Finding 1 — governing records missing from the proposal's table.** These are **ALREADY RECORDED** constraints, not new implementation defects:

| Omitted record | Effect |
|---|---|
| `docs/CLAUDE_CODE_INSTRUCTIONS_S172_D6_2_CHECKPOINT_RECONCILIATION.md:167-208` and `docs/CLAUDE_CODE_INSTRUCTIONS_S172_D6_2_BOUNDED_REPAIR.md:72-105`, `:128-146` | Govern Q4/Q11's within-chunk resume: explicit checkpoint run id, context binding, distinct Optuna numbers and record ordinals, proof an existing study was loaded, and the `n_parallel == 1` certification boundary. Listing the identity addendum alone does not capture these requirements. |
| `docs/CHAPTER_1_WINDOW_OPTIMIZER.md:810-828` and `docs/CLAUDE_CODE_INSTRUCTIONS_S172_THRESHOLD_REPAIR.md:49-51`, `:102-130` | Govern Q7/Q9/Q11's claim to record what ran: one threshold resolver, requested/payload/effective provenance, execution-based controls, and evidence before classifying historical study values as executed. These records do not establish a current threshold regression. |

Other governing positions in the proposal's table are **ALREADY RECORDED**, including the dataset input wall, per-session rule, warm-start containment and historical cursor law. Moving scheduling to a search cursor **CHANGES A RULING**, as Alpha declares. The zero-trial history disposition and the concrete budget/lifecycle requirements discussed below concern the **NEW** design; Stage A P3/P4 themselves remain recorded statuses.

**Not searched:** live or archived SQLite databases, Optuna stores, attempt 9's retained artifacts, host services, ser8, rigs, Proxmox hosts, remote/private repositories or external sources. No complete Chapter 1 reread. No test suite or optimizer/publication execution. I did not verify attempt 9's import fields or any live S179 implementation.

## Review header

- **Thread / round:** X-STEP1-SEARCH / R1.
- **Date / host:** 2026-11-14, VM101.
- **Base read:** `eac05ec90454d97953b361f1e8bc9ad2a1f16949`.
- **Request:** `docs/exchange/to_beta/X-STEP1-SEARCH_R1_ALPHA_REVIEW_REQUEST_search_record_and_rerun.md`.
- **`git status --porcelain` at initial read:** empty output. It was still empty immediately before writing this review. A final read-time comparison of the inspected production sources, cursor harness, proposal and request against that base showed no diff.
- **Class / authority:** design review before code. Beta records positions; Michael decides. This is not implementation certification or run authorization.

## Positions

| Question | Position | Severity |
|---|---|---|
| Q1 — earlier scheduling objection | NO OBJECTION | — |
| Q2 — cursor law and keep-list | NO OBJECTION | — |
| Q3 — separate tables | NO OBJECTION | — |
| Q4 — credit and budget | OBJECTION | LIVE — the proposed zero-trial history rule can certify an untraversed interval on the already-recorded exhausted-study path and make historical COMPLETE false. |
| Q5 — opening authority | NO OBJECTION | — |
| Q6 — dataset and session | SUGGESTION | — |
| Q7 — Optuna reach and containment | SUGGESTION | — |
| Q8 — ML row provenance | NO OBJECTION | — |
| Q9 — search 0 import / Run 4 | NEEDS EVIDENCE | — |
| Q10 — F4/F5 wording | SUGGESTION | — |
| Q11 — measurements | SUGGESTION | — |
| Q12 — missed governing records | SUGGESTION | — |

## Q1 — the proposed separation answers the earlier objection

Yes, at the design level. An explicitly opened search can start inside the governed domain even after the historical union is COMPLETE. That supplies the rerun capability requested in X-RUN4-RANGE Q2 without erasing history (`docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_0.md:160-185`).

The earlier implementation fact remains: WATCHER currently queries the historical cursor and returns before command construction on COMPLETE (`agents/watcher_agent.py:2052-2085`). Closure of that implementation objection needs the changed route and intercepted-dispatch evidence, followed by the separately authorized production evidence. This review does not mark that work done.

## Q2 — retain the keep-list and distinguish the two COMPLETE states

No objection to the scheduling amendment in §3. The pure first-gap calculation already represents COMPLETE without an out-of-domain next seed (`utils/seed_coverage_ledger.py:576-600`). Applying it to properly validated search-linked intervals preserves that arithmetic. The current history selector and mode-containment predicate can remain unchanged (`utils/seed_coverage_ledger.py:994-1055`).

Keep these implementation obligations explicit:

- Derive range, family, executed modes and generation from the existing certified publication witness; validate the link against that row and the selected search. The present certification door derives those fields from the artifact (`utils/seed_coverage_ledger.py:862-907`). A caller-supplied link must not independently assert a different interval or mode set.
- Assign the selected search's first gap even when it is zero. Current WATCHER only overwrites a supplied start when the cursor is positive (`agents/watcher_agent.py:2087-2113`); preserving that branch unchanged would not establish the stronger new scheduling claim. This cursor-zero limitation is already recorded in the R2 instructions at `:145-155`.
- Keep the pre-dispatch domain wall and fail-closed lookup (`agents/watcher_agent.py:2055-2066`, `:2131-2142`). Distinguish no open search, completed search and lookup failure in durable results.
- Represent opening, superseding and completion with history-preserving lifecycle events. Define the unique open-search rule transactionally. A missing or conflicting selector must not become a fresh search at zero.
- After successful artifact publication, make history/link recording idempotent and recoverable. A link-write failure must surface and leave the search unadvanced; retry must use the same publication witness. The existing hook already raises after a coverage-write failure rather than pretending publication was undone (`window_optimizer_integration_final.py:3033-3060`).

No reset or deletion is needed. Q4's proposed history-credit rule is the exception to an otherwise sound keep-list statement.

## Q3 — prefer the sibling tables

No objection; prefer this option. Adding `search_id` to `certified_coverage` conflicts with its exact column assertion (`utils/seed_coverage_ledger.py:688-693`, `:732-757`) and needlessly changes the established history object.

The sibling design still needs enforced references, immutable bindings, schema/version validation and duplicate-write behavior. It must not become an alternate certification door. Preserve the history ledger tests; see Q11 about WATCHER integration tests whose scheduling authority is intentionally changing.

## Q4 — zero completed trials do not establish traversal

**Do not adopt “zero completed trials still gets a certified history row because the seeds were traversed” as an unconditional rule.** On the exhausted-study path, the current budget calculation can give zero objective calls (`window_optimizer_bayesian.py:688-699`, `:804-829`). The no-result return exists (`:855-869`), and integration obtains candidates and supplies the configured interval to the finalizer without a completed-trial gate at those call sites (`window_optimizer_integration_final.py:2923`, `:3001-3011`); the history write follows publication (`:3056-3060`). Publication alone is not evidence that this invocation executed that interval.

This is a source-path inference about the already-recorded P4 mechanism, **not an executed reproduction**. M1 should settle its actual behavior. The NEW objection is that §2.3 would preserve the false-history possibility deliberately, despite making historical COMPLETE the F4 acceptance evidence.

Use this disposition:

- No evidenced completed traversal of the chunk under the required modes: no certified history credit and no search credit. Retain the attempt, diagnostic publication and failure/skip reason in non-certifying audit records if needed.
- A completed traversal yielding zero bidirectional survivors is different: zero survivors do not invalidate traversal credit. Require execution evidence, not a positive survivor count.
- Zero **new** trials on a resume need not mean zero work: a verified recovered chunk may already contain the required completed work. Define whether it is eligible for finalization, and prove that work belongs to this chunk. Never borrow the prior chunk's study-wide completed count as its evidence.

Per-chunk budgets are the right direction for P3/P4, but specify a durable chunk identity and budget baseline/target **before its first trial**, plus the checkpoint handle and actual Optuna-trial assignments. Recomputing the baseline after a crash grants another budget. Counting the whole study credits earlier chunks. Neither is a within-chunk resume.

Also define the budget unit. `study.optimize(..., n_trials=...)` receives the number of trials to attempt, while the current subtraction counts only COMPLETE trials (`window_optimizer_bayesian.py:688-691`, `:806-829`). State how PRUNED and FAIL consume budget, separately from the requirement for an evidenced completed traversal. Distinguish a budget-exhausted chunk from an intentionally under-budget publication. Keep Optuna numbers and persisted record ordinals separate, as the omitted D6.2 ruling requires; current integration explicitly maintains that separation (`window_optimizer_integration_final.py:2720-2744`).

## Q5 — the routes are consistent only within their existing authority

No objection to explicit opening through the stated routes. New draws alone must not open a search; an open gap makes already authorized work due, and does not supply authorization itself.

The existing human CLI already populates the privileged pin channel (`agents/watcher_agent.py:3919-3932`). Merely adding explicit search selection/opening there does not create a new caller of that channel. It needs the planned scoped routing review and evidence refresh, not another permission request to use the existing operator seam.

If search continuation, import, the approved-request route or another helper reconstructs stored pins and populates `_operator_pin_params` or `_pin_bundle`, **that is a new authority-bearing production caller** and needs the explicit governance review required by Route-A R2 (`docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:13`, `:48-50`). Stored configuration is not operator authority.

S179 requires both the generic request-consumer gate and the execution-chokepoint recheck (`docs/TB_RULING_S179_IMPLEMENTATION_AUTH.md:225-300`), with triggers audit-only under BOOTSTRAP (`:525-532`). A request marked approved is not a substitute. The present CLI approval branch loads status/steps and calls `run_pipeline` (`agents/watcher_agent.py:3845-3878`); I did not verify a working S179 gate on this host. Agent launch authority also remains subject to the delegated-run rules; Beta's own no-launch contract remains unchanged.

## Q6 — freeze the actual input and record the session stream

Finishing a search against its original immutable dataset is sound. Appended draws neither erase its past traversal nor make its scores current. A mismatch refusal is appropriate. The record should preserve dataset version/digest and the authoritative input-manifest identity, with lineage/provenance sufficient to reproduce the input (`docs/TFM_PROJECT_FACTS_SKILL.md:638-648`).

Specify the actual route to the pinned version. WATCHER currently freezes the pointer alias (`agents/watcher_agent.py:748-763`) and separately rewrites dataset parameters (`:1932-1935`); the optimizer child freezes its own supplied path (`window_optimizer.py:1822-1839`). Merely comparing a search to today's pointer can strand an unfinished search after an append. Passing a retained named version must also make WATCHER preflight, the child, checkpoint identity and the publication hook agree on that version. The resolver supports an explicit version (`miner/dataset_authority.py:548-563`); this design has not yet demonstrated the complete route.

For session scope, retain the semantic session label/set and its ordering/filter rule, the source publication identity, the resolved session-input identity, and each trial's actual session selection. An integer `session_idx` alone is insufficient: the optimizer maps it through `bounds.session_options` (`window_optimizer_bayesian.py:535-557`). Bind that mapping, too. Credit for one selected stream must not imply a sweep of every declared session option.

Do not decide BACKLOG §5 by adding a field: the authority for the session-scoped artifact remains open (`docs/BACKLOG.md:104-120`). Combined-session sequential input remains non-certifying by default; this review supplies no exception and no new interpretation of the operational Run-4 claim boundary.

The cross-dataset accumulator issue is **ALREADY RECORDED**, not a new finding. Deferring its implementation is acceptable only with an enforceable restriction: refuse incompatible-input publication until the input wall or another owner-decided disposition is implemented. A new search id does not license merging unlike inputs. The present finalizer merges the prior generation (`utils/run_finalizer.py:1636-1642`), and its sidecar construction carries the generation chain and range but no dataset binding in that payload (`:1662-1700`). Same-dataset Run 4 still needs its import evidence.

## Q7 — keys are useful; enforce study ownership and separate evaluations

The proposed reach is compatible with S167 if it remains passive. Preserve fresh searches and the explicit operator pin seam; do not make `repeat_of` executable warm-start authority (`docs/SESSION_CHANGELOG_20260424_S167_S168_S169.md:19-51`).

Add an explicit study-ownership rule: continuing chunks may use their search's study; a new search gets a distinct study identity, with collision/mismatch refusal. An arbitrary existing `study_name` must not silently carry prior trials into another search. Current resume loads the supplied study and accepts it when it has completed trials (`window_optimizer_bayesian.py:665-699`); the new search record needs the compatibility check that this path does not itself provide. Define the content-hash preimage/version and explicit repeat identity so an identical experiment can be reopened while retries remain idempotent.

For write-back, append a separate evaluation record referencing search, evaluated generation/pool, target draws and information cutoff; leave trial evidence immutable. Search-level hit rate is an aggregate observation, not proof every trial caused those hits, especially with accumulated rows from several searches. The present writer updates all rows sharing a run id (`database_system.py:484-499`); replacing its key alone does not establish causal trial attribution.

A later carry-forward proposal must name its source/configuration mapping, distinguish resume from cross-search reuse, obtain review of any new privileged caller, preserve dataset/session compatibility and resource bounds, and demonstrate prospective benefit against the fresh-search baseline on unseen draws with comparable budgets. Outcomes used to choose the carry-forward policy cannot also be its final evaluation. S176's information cutoff already applies to window selection and threshold selection (`docs/TB_RULING_S176_WATCHER_KPI.md:692-727`). No numerical lift is established here.

## Q8 — the generation-chain derivation is sound within its stated limits

Yes, for the origin of the **retained winning row**. The actual merge retains the prior row on equal/lower score and replaces it only on strict float32 improvement; it copies every array from the selected source (`utils/run_finalizer.py:757-807`). A newly appearing seed or a strictly increased retained score therefore identifies a current-generation winner. Otherwise trace the parent. Parent generation and artifact/sidecar hashes are recorded (`utils/run_finalizer.py:1662-1667`).

This requires every relevant ancestor and its search link to remain available and verified. An absent legacy link means unknown origin, not “the current search.” The derivation does not identify every search that encountered a seed or a losing equal-score candidate.

I do not require finalizer edits or a new array now. A versioned external origin index written after publication is a reasonable convenience, but it must implement the same strict-greater rule and have a recoverable publication/link boundary. At minimum add a tie/replacement/inheritance test and document retention requirements. ML must not consume guessed provenance when a link is missing.

## Q9 — import is conditional; dataset and numeric pins alone are insufficient

NEEDS EVIDENCE for attempt 9's actual import. Neither Alpha's proposal nor this review opened the archive. “Search 0” should be a display label for an ordinarily identified import, not a bypass of schema, validation or credit rules.

The import evidence should retain:

- Archive identity/path, hashes and read-only query/extraction procedure; exact source run, trial, range and phase-completion records.
- The existing `coverage_id`, artifact/sidecar hashes, generation and repository revision, cross-checked against the certified publication, without rewriting it.
- Frozen dataset version/digest, session mapping and actual stream, effective window/anchor and generator-phase semantics, skip geometry, directional thresholds, executed modes, and trial completion/budget evidence attributable to this interval.
- Actual study identity, or an explicit unknown. No invented study or trial settings. Record which imported facts were observed, derived or unavailable, and the idempotent import/opening event.

Compare the **effective** configuration and permitted operational scope, not just equal pin numbers. The Run-4 ruling expressly describes historical pre-advance 25 becoming independent `generator_phase=0`, and limits the comparison's meaning (`docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:154-179`). The new record must expose that semantic difference. Continuing search 0 under a changed configuration needs an explicit disposition; do not quietly hash both as identical.

If the evidence cannot support a credited continuation, keep attempt 9 as historical traversal and report the proposed separate-search/range consequence before launch. Do not invent its lower-half search credit merely to obtain the desired Run-4 start. All archive reads must obey the immutable read-only rule.

## Q10 — F4 is clear; split the two F5 milestones

No objection to F4 meaning historical traversal, provided every newly credited interval has real traversal evidence as in Q4. It establishes operational coverage, not predictive benefit or completeness across windows/sessions.

Separate F5 into (a) the intercepted-command capability test, required before Run 4, and (b) later production confirmation by the soak. Otherwise making the soak part of “finished” conflicts with the retained completion-plan sequence, which puts Brief II and the soak after finished (`docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:52-65`, `:96-101`). The owner can adopt this wording without moving the soak ahead of Brief II.

## Q11 — keep M1–M7's intent; expand the implementation evidence

They are a useful starting set, not sufficient evidence for the code that will implement this design. Before submitting that code for review:

1. **Run M1 before edits**, in isolated state with objective/dispatch interception, and report actual calls, candidate/publication outcome and credit writes separately. Replace M3 with Q4's evidence-based disposition; test zero work, completed zero-survivor work, and valid recovered work.
2. **Budget/resume:** two chunks in one study each receive the declared budget; crash/restart within chunk 2 preserves its baseline and recovered evidence. Include PRUNED/FAIL states and the D6.2 integrated-resume, missing-study and NP2 boundary controls. Report study numbers, record ordinals and chunk assignments separately.
3. **Identity/isolation:** same-configuration explicit repeat gets a distinct search; retry does not; study reuse across searches is refused. Reopening/superseding preserves old records and produces at most one unambiguous open selector.
4. **Publication/link failures:** failed publication grants neither credit; failed history/link recording surfaces after publication and recovers idempotently; wrong-generation/dataset/mode links and schema/read errors fail closed. Include gaps/overlaps and the exact domain endpoints.
5. **Effective input/configuration:** named version A remains usable after pointer movement to B; wrong bytes/digest refuse before dispatch. Independently mutate session mapping, search-space digest, pins, sampler/pruning configuration and declared budget. Prove WATCHER-shaped params arrive at the optimizer; prove recorded effective values agree with the existing execution evidence. Preserve the threshold resolver and relevant regression controls.
6. **Authority/freshness:** M5 plus no-open/complete-search clean stops; no lookup error becomes a default search; no trigger bypasses BOOTSTRAP/S179; an unpinned invocation following a pinned one inherits neither pins nor authority. Refresh Route-A controls against the changed target.
7. **Provenance/import:** retained, new, strictly improved and tied rows across multiple searches; missing ancestor/link is unknown or refused. Test import replay and rejection of missing or incompatible import evidence. Scope evaluation write-back to the recorded evaluation identity.

**M7 needs a stated compatibility plan.** Leaving the ledger module/schema unchanged is reasonable. The existing suite also tests WATCHER scheduling: its COMPLETE arm stubs `get_certified_cursor` and expects that historical answer to stop dispatch (`tests/test_seed_domain_cursor_amendment.py:675-704`). That is the authority being amended. An unedited green run alone cannot prove both historical reporting and the new selected-search behavior. Keep the historical ledger assertions intact; explicitly identify any integration-fixture changes needed for the new search selector, and add the paired case “historical COMPLETE + open search → command.” Do not preserve an obsolete scheduling fallback merely to keep the old fixture green.

These are CPU-only evidence requirements; none calls for a fleet run in this design review. Existing suites that mutate production files must be isolated by Alpha; Beta cannot run them in this shared tree under its write restriction.

## Q12 — correct the prior-record inventory

Add Finding 1's checkpoint and threshold-provenance records, labeled ALREADY RECORDED, with their effect on resume, import and measurement design. The dataset publication schema corroborates the already-listed input-wall ruling (`docs/DATASET_PUBLICATION_SCHEMA_v1.md:107-115`); it is not another new defect or a newly discovered decision.

## Verification and limits

I ran read-only repository searches, numbered source/document reads, `git rev-parse HEAD`, `git status --porcelain` and the scoped comparison against the base. I ran no tests and opened no database. M1–M7, the import, routing refresh, publication/recovery behavior and S179 host behavior are **UNAVAILABLE in this review**, not passes. No pipeline, worker, fleet connection, package change, commit or push occurred. The only repository write is this review.

All behavioral conclusions above are source-verified facts or explicitly identified design inferences. No additional new implementation defect is claimed beyond the questions reviewed. The zero-trial proposal objection affects truthful operational coverage; it says nothing about a filter's predictive quality. TFM's predictive criterion remains pool hits above `k/1000` on draws unseen by selection and scoring.
