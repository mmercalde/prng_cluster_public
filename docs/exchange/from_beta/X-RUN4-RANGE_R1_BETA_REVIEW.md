# X-RUN4-RANGE · R1 · BETA REVIEW

## Prior record

**Searches run in this session:**

- Read `AGENTS.md`, the complete request, and `docs/CLAUDE_CODE_REPORT_RANGE_MINER_VM101_READS.md`.
- `rg -n -i 'coverage|complete|seed.start|interval|route.a|certif|repeat|second half|soak|session' docs/TB_RULING_* docs/PROPOSAL_* docs/CHAPTER_*_AUDIT.md docs/TODO_CONSOLIDATED.md`.
- `rg -l -i 'seed.domain.complete|after complete|coverage identity|per.geometry|re.sweep|routing certification|seed.domain sweep terminus' docs`.
- Searched the S145 amendment instructions, R1/R2 instructions, submissions and reports for `provenance|partition|dataset|session|geometry|window`; read the relevant passages.
- Searched `docs/PIPELINE_BEHAVIOUR_MODEL.md` and the fact file for coverage, cursor, regime reruns and session policy; read the purpose-lineage report's initial conclusions and provenance.
- Read the Run-4 routing/geometry ruling and final R2 routing review, the Run-4 proposal's comparison section, completion-plan §§2–3, soak requirements, delegated-run rules, findings-plan isolation rules, and the state-recovery pitfall.
- Read current source and test anchors cited below; inspected `git log -- agents/watcher_agent.py` and `git diff 3e2537b HEAD -- agents/watcher_agent.py`.

**Finding 1 — governing record omitted from the request.** These are **ALREADY RECORDED** positions, not newly discovered implementation defects:

| Record missed | Effect on this review |
|---|---|
| `docs/CLAUDE_CODE_INSTRUCTIONS_SEED_DOMAIN_CURSOR_R1.md:7-16`, corroborated by `docs/TB_SUBMISSION_SEED_DOMAIN_CURSOR_R1.md:34-39` and `docs/CLAUDE_CODE_REPORT_SEED_DOMAIN_CURSOR_R1.md:51-57` | Beta explicitly accepted `dataset_sha256` as provenance, **not a v1 partition key**. Q2 therefore encounters an actual recorded decision, not merely an absent dataset predicate. |
| `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:12-24` | Governs the final routing certificate, privileged authority seams, unpinned control, mutation evidence and post-commit provenance check. Relevant to Q4. |
| `docs/CLAUDE_CODE_INSTRUCTIONS_WATCHER_FOLLOWUP_REPAIRS_20261111.md:57-78`, with implementation report `docs/CLAUDE_CODE_REPORT_WATCHER_FOLLOWUP_REPAIRS.md:123-153` | COMPLETE stops the pipeline cleanly without advancing. This repair does not create a route for another optimization. Relevant to Q2/Q4. |
| `docs/CLAUDE_CODE_REPORT_STEP1_PURPOSE_LINEAGE.md:45-62`, also summarized in the already-listed fact file at `:1128-1136` | The tension between a regime rerun and a globally advancing frontier was already recorded. The precise post-COMPLETE/soak ordering conflict is a further consequence; the broader rerun mismatch is not NEW. |

Other governing items checked: completion plan F2/F4/A3/C2; S145-R1's rejection of permanent Step-1 retirement; fact-file §2.10b's dataset/session policy; Route A's limited operational claim boundary. All are **ALREADY RECORDED**. A changed coverage partition or explicit rerun policy **CHANGES A RULING**. The recommendation on Q1's upper-half comparison is a **NEW** clarification for Michael to adopt if he agrees.

**Not searched:** ser8, rigs, Proxmox hosts, external/private archives or remote repositories. The original standalone August terminus ruling was not located; its amendment/R1 records were read. I did not independently reopen attempt 9's generation or miner archive. Its zero-row/per-phase counts remain attributed to Alpha's reads report. I did not trace optimizer stripe planning or exercise a finalizer/publication. No claim here establishes actual upper-half fleet execution or hit rate.

## Review header

- **Thread / round:** X-RUN4-RANGE / R1
- **Date / host:** 2026-11-13, VM101, user `michael`; local checks around 20:00 PST.
- **Base read:** `8a314e54a4342aede84fbe232c74f20425159b63`.
- **Request:** `docs/exchange/to_beta/X-RUN4-RANGE_R1_ALPHA_REVIEW_REQUEST_run4_seed_range_and_complete_state.md`.
- **`git status --porcelain` at initial read:** empty output. It remained empty after the checks, before writing this review. No tracked-source drift was detected.
- **Authority:** Beta records review positions; Michael decides. This review authorizes no launch or production change.

## Positions

| Question | Position | Severity |
|---|---|---|
| Q1 — upper-half interval and Run-4 claim boundary | NO OBJECTION | — |
| Q2 — permanent family-wide Step-1 refusal | OBJECTION | LIVE — completing the presently planned second-half sweep would prevent the planned soak and later Step-1 optimization for this family. |
| Q3 — ordering | SUGGESTION | — |
| Q4 — refreshing routing evidence | SUGGESTION | — |
| Q5 — missed governing record | SUGGESTION | — |

## Q1 — same scale can establish the limited operational claims

**Yes, conditionally.** A completed `[2³¹, 2³²)` run with the stated pinned geometry, dataset, stripe size, worker pool and production route can establish volume class, four-phase plumbing, publication, coverage advancement and production-shape completion. Those are the claims listed in `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:168-179`. The completion plan explicitly describes the second half as another attempt-9-shaped run (`:61`) and leaves the interval decision open at A3 (`:77`).

It cannot be described as an experiment with only `generator_phase` changed. The original proposal explicitly held the seed domain constant (`docs/S172_RUN4_PROPOSAL_PINNED_GEOMETRY.md:134-159`); the interval now changes too. Record both intervals and narrow the comparison accordingly. Neither matching counts nor mismatching counts isolate the effect of phase, establish population equivalence, or measure predictive value.

The expected counts are an envelope to check, not a guaranteed result on a different interval. Apply the existing volume-limit requirement to the **actual requested range and residue window**, as specified in `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:154-165`. This review supplies no upper-half survivor-volume measurement.

**Verified locally:** the live ledger still contains exactly the attempt-9 certified row, `[0, 2147483648)`, both modes. Pure production cursor functions give OPEN at `2147483648`. With that value supplied to the intercepted provenance harness, current WATCHER builds `--seed-start 2147483648 --max-seeds 2147483648`, with all seven pins unchanged. This follows the cursor call and overwrite at `agents/watcher_agent.py:2052-2054`, `:2087-2089`; the exact endpoint passes the production preflight at `:2131-2135`. Execution below argv construction remains untested here.

## Q2 — COMPLETE must finish a traversal, not retire optimization

**No: permanent family-wide refusal is not the intended learning behavior.** The owner's position in this request governs; permanent retirement was also rejected in `docs/PROPOSAL_S145_R1_Progressive_Empirical_Sweep.md:22`, `:39-41`.

The implementation fact is unambiguous. Identity canonicalization returns family and required mode set (`utils/seed_coverage_ledger.py:271-320`). Record selection tests family and mode containment, with no dataset, session, window or threshold predicate (`:994-1019`, `:1029-1055`). WATCHER returns before building a command on COMPLETE (`agents/watcher_agent.py:2068-2085`), then `run_pipeline` stops without advancing (`:2902-2909`). Dataset changes therefore cannot reopen this cursor. The local simulated-COMPLETE check confirms that no argv is built. This is an operational scheduling objection, not a claim about a filter's predictive value.

The smallest correct coverage statement is:

> The retained publication certifies traversal of this seed interval for the recorded family and executed modes under that run's actual inputs and search configurations. It does not certify that every window, threshold or session was tried, that its scores remain valid on another dataset, or that future optimization is unnecessary.

The ledger binds range, dataset digest and artifact identity (`utils/seed_coverage_ledger.py:608-624`, `:822-840`); the current cursor deliberately computes a coarser historical union. Keep these two meanings explicit.

**Dataset and session:** yes, a claim that a *particular search is complete or reusable* must distinguish the frozen dataset version and selected session stream, as well as its relevant search configuration. The fact-file lifecycle rule (`docs/TFM_PROJECT_FACTS_SKILL.md:632-648`) supports that requirement. However, stale scores do not erase the historical fact that a seed interval was traversed. Nor does “per-session” mean every invocation must automatically receive a fresh namespace: distinguish midday/evening selection from an execution/study identifier.

**Suggested narrow design for Alpha to specify:** preserve the append-only historical traversal ledger and its first-gap/COMPLETE result; introduce an explicit optimization/rerun policy so historical COMPLETE does not prohibit a newly authorized Step-1 search inside `[0, 2³²)`. Bind each search to frozen dataset identity, selected session, declared configuration/search space and a stable search identity, with an explicit way to repeat a search. A resume continues that identity; a new optimization does not silently inherit permanent completion from another search. Exact schema/API choices belong in Alpha's amendment.

If Alpha instead proposes making dataset/session/configuration part of the coverage cursor key, that is also reviewable, but must explicitly amend the v1 decision that dataset hash is provenance only (`docs/CLAUDE_CODE_INSTRUCTIONS_SEED_DOMAIN_CURSOR_R1.md:16`; executable counterpart `utils/seed_coverage_ledger.py:884-888`). Dataset partitioning alone does not enable another geometry or repeated search on an unchanged dataset. Session partitioning alone has the same limitation.

Retain domain bounds, publication-before-credit, append-only history and fail-closed lookup. Do not reset/delete old credit or silently make a new cursor scope look like the old one. State whether plan F4 still means historical traversal coverage or a newly scoped search; changing that definition also needs Michael's decision. Production changes require Alpha's proposal and Beta's before-commit review. No such change is reviewed here.

## Q3 — choose (c), then use the planned sequence

**Recommend (c): settle the coverage/rerun rule before treating either alternative as the agreed route through Run 4 and the soak.** The required capability is already decided by the owner: Step 1 must be runnable again. What remains is its concrete policy and evidence, not another request for permission to preserve learning.

- **(a) under current code:** a successful second-half publication would make the historical union COMPLETE; the subsequent WATCHER Step-1 invocation would build no command (`utils/seed_coverage_ledger.py:583-592`; `agents/watcher_agent.py:2068-2085`). This does not deliver the soak.
- **(b):** moving a 50-trial invocation ahead of COMPLETE does not resolve future reruns. Nor is the soak automatically Run 4: Run 4 has one pinned trial; the soak requires 50 trials, a high/low survivor mix and cleanup evidence (`docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_7_SOAK.md:96-124`). Reusing some future soak evidence would need an explicit revised acceptance protocol and would change the completion plan's order.
- **After the amendment is settled and verified:** retain the planned Run 4 over the upper half, review its publication/production-shape evidence, then Brief II and the soak in the plan's order (`docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:57-65`, `:96-101`).

This recommendation does not invalidate an independently owner-authorized upper-half operational run under Q1. Such a run would still not settle Q2 or make the following soak executable without the amendment.

## Q4 — refresh the evidence against the current target

**A provenance rerun is necessary but insufficient by itself to renew the whole routing certificate.** It proves the positive pinned route; it does not re-establish the unpinned control, partial/unusable-value refusal, unauthorized-origin rejection or invocation isolation covered by the R2 review (`docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:12-20`).

Use a scoped refresh: record the current commit/digests, review the two-commit delta, rerun relevant routing/containment regression controls against that target, and capture the intended command with a freshly read real cursor value. The diff from `3e2537b` changes failure evaluation, clean handling of COMPLETE and CLI dry-run refusal; it does not edit the pin-routing/run-step construction. Relevant current anchors are `agents/watcher_agent.py:968-995`, `:2902-2909`, `:3652-3670`, and the CLI authority seam at `:3919-3932`. This supports a delta review rather than reopening the Route-A design. Any subsequent Q2 implementation requires another refresh of the affected evidence.

Preserve the frozen cursor-zero oracle for its original unpinned comparison; add a separate real-cursor case. Its deterministic stub is deliberate (`tests/fixtures/run4_routing_clean_control.py:13-18`, `:79-90`). The prelaunch script likewise fixes its own cursor at zero (`tests/run4_prelaunch_provenance_check.py:136-152`). Running that file unchanged cannot establish the actual launch interval. Capture effective start/end, mode set, seven values and provenance argv explicitly.

**Evidence this session:** original prelaunch assertions 10/10; separate in-memory harness variants for OPEN at the measured `2³¹` and hypothetical COMPLETE both passed. Fleet, dataset freeze and subprocess execution were intercepted, so these are command-construction checks, not production preflight or renewed full certification.

**UNAVAILABLE under Beta's write contract:** the complete existing routing mutation suite was not run. Its M1b arm writes and restores `agents/watcher_agent.py` (`tests/test_s172_run4_routing_patch.py:738-760`), which Beta may not do. Alpha can supply compliant isolated certification evidence; this review does not count the historical 20/20 as a current pass.

## Q5 — add the records in finding 1

The decisive omission is the explicit **dataset-provenance-only** decision. Add it and relabel a dataset-partition proposal **CHANGES A RULING**. Also add final Route-A R2, the COMPLETE clean-stop specification/report, and the previously recorded rerun/frontier tension. No searched source supplies a complete, currently implemented post-COMPLETE optimization policy. That is a bounded search result, not a claim that no external ruling exists.

## Checks and limits

- Ran `/tmp/x_run4_range_beta_check.py` after activating `~/venvs/torch`, with `PYTHONDONTWRITEBYTECODE=1`. Final exit 0 and `BETA_CHECK_COMPLETE: PASS`; output is `/tmp/x_run4_range_beta_check.log`. An initial audit-hook type mismatch stopped before opening SQLite; corrected the temporary check and reran successfully.
- Opened only `file:/home/michael/distributed_prng_analysis/prng_analysis.db?mode=ro&immutable=1`, after checking for non-empty WAL/journal files. Database hash/stat were unchanged before and after. No ledger wrapper was constructed. Mode queries and hypothetical interval union used production pure functions.
- Executed the existing prelaunch harness assertions and two AST-adjusted **test-fixture-only** variants in memory. Production source was untouched. Network connections, subprocess launches, repository writes and other SQLite connections had rejecting audit tripwires. The harness intercepted dispatch; no pipeline or worker was launched.
- Read the two-commit WATCHER diff. No full routing mutation suite, GPU check, fleet preflight, optimizer execution, finalizer, or prediction evaluation ran. Those are not passes.
- Current WATCHER SHA-256: `bff9e22b488a7eadc2918210a2933eb11fcbf1c7ff46da8a284ce6b40dc71238`.

No additional new defect is asserted beyond the questions reviewed. Finding 1 corrects the prior-record inventory. Attempt 9's zero rows remain an observation at its recorded geometry; this review draws no hit-rate conclusion from them. TFM's predictive criterion remains pool hits against `k/1000` on unseen draws.
