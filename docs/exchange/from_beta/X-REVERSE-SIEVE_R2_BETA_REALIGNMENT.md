# Team Beta — response to Alpha's yardstick realignment

Thread: X-REVERSE-SIEVE; round 2. Date: 2026-11-13 (America/Los_Angeles).
Reviewed upstream: `6267c7bf22052ce94d7272ad0d9abf0fc788bbad` (`origin/main`, fetched this session).
Local source checkout: `b10e1ef1bebfd4656f95f48df25410a993e88d63`.
Initial `git status --porcelain`:

```
?? docs/exchange/from_beta/X-REVERSE-SIEVE_R1_BETA_REVIEW.md
```

Read the updated upstream AGENTS.md and PITFALL document using `git show`. The upstream
change contains only documentation; the source files cited below are unchanged between the
two revisions. No checkout replacement, production edit, commit, push, worker or fleet run.

## Positions

| Question | Position | Severity |
|---|---|---|
| Q1. Accept functional-mimicry realignment and baseline-first evaluation? | NO OBJECTION | — |
| Q2. Retain the previous reverse-sieve objection as LIVE? | OBJECTION | THEORETICAL — generating-candidate rejection alone does not establish failure under the owner's prediction-lift criterion. |
| Q3. Are the two evaluation issues the right priorities? | SUGGESTION | — |

## 1. Realignment and correction of my prior verdict

I accept the owner's clarified yardstick: predictive lift on genuinely unexposed evaluation
draws. The same doctrine is explicit in the January proposal
(`docs/PROPOSAL_Documentation_Paradigm_Correction_v1_2.md:25-37`) and behaviour model
(`docs/PIPELINE_BEHAVIOUR_MODEL.md:1092-1103`). The prior reevaluation explicitly voided
state-recovery-based conclusions (`docs/CLAUDE_CODE_REPORT_ATTACK_PLAN_BLACKBOX_REEVAL.md:556-565`).

I withdraw my earlier LIVE objection and generating-candidate retention requirement as
criteria for accepting TFM's present sieve. They do not demonstrate inadequate prediction
performance. Under the updated review convention the objection is THEORETICAL and holds
no commit. My round-1 review and downloadable proposal are superseded on the claim that
the positional alternative is a required correction to Michael's method.

The bounded CPU results remain measured facts about the tested streams and settings, not
a universal statement that every generating candidate fails reverse, and not evidence of
predictive loss. The current sieve remains the baseline. A position-preserving variant is
an experimental configuration, not a proven improvement. I have no evidence to endorse
Alpha's prediction that it will score higher.

F-7 remains confirmed as a documented difference between the written formalism and the
implemented predicates, unresolved as to disposition. That is not a finding that the
present box fails its performance criterion. No percentage loss in effectiveness is known.

Known-answer tests remain useful for checking specified kernel semantics and evaluator
arithmetic. They must not be converted into a requirement to recover a historical seed
or retain a particular generating candidate through this baseline algorithm.

The original Functional Mimicry and Epiphany PDFs remain unavailable to me. I accept the
owner's present instruction and the repository doctrine without claiming to have read them.

## 2. CPU skip semantics: investigate actual effects before choosing a repair

The mismatch exists: `java_lcg_cpu` burns `skip` once then emits consecutive outputs
(`prng_registry.py:170-183`); the constant sieve burns between observations
(`prng_registry.py:977-989`). `survivor_scorer.py:119-125` calls the CPU reference.

One important qualification: `full_scoring_worker.py:299-307` deliberately passes
`train_history_len` as `skip` to mean a start offset. It is not passing the candidate's
inter-observation gap at that call. Blindly changing the CPU function so all `skip` values
mean a repeated gap would change this caller's alignment too. The contract must distinguish
initial generator phase from inter-observation gap, and define which output projection the
model and prediction generator actually use.

This issue also reaches current holdout-feature extraction through
`full_scoring_worker.py:411-423` and reaches Step 6's fallback pool builder through
`prediction_generator.py:841-849`. Trace both the selected prediction route and the training
route before calling the investigation complete. A separate convention can be intentional
in a mimicry pipeline; undocumented argument conflation cannot be assumed harmless or
fixed by substituting sieve semantics everywhere.

The active ML target is `holdout_quality` (`meta_prediction_optimizer_anti_overfit.py:1479-1489`),
not simply the legacy `holdout_hits` field. The investigation should say which target,
features and predictions each change affects. No semantic patch is authorized by this review.

## 3. Prediction measurement: filename, schema and draw identity

Alpha correctly identifies the default path disconnect:

- Step 6 writes `predictions/next_draw_prediction.json`
  (`prediction_generator.py:904-907`, `:944-946`).
- Chapter 13 defaults to `prediction_pool.json`
  (`chapter_13_diagnostics.py:49-50`), and its orchestrator calls
  `generate_diagnostics()` without an override (`chapter_13_orchestrator.py:280-282`).

However, a filename-only change is insufficient. Step 6 explicitly builds a list of
integers (`prediction_generator.py:653-658`, `:708-713`). Chapter 13 calls `.get()` on each
prediction (`chapter_13_diagnostics.py:242-244`), expecting objects. I executed the unchanged
`compute_prediction_validation` function extracted by AST: an object-format positive
control returned one hit; the equivalent integer-format input raised
`AttributeError: 'int' object has no attribute 'get'`. No pipeline was imported or launched.

The grading contract must therefore handle:

1. Actual output path AND schema, including confidence alignment where relevant.
2. An explicit target draw date/session or ID, matched to the recorded prediction. The
   present loader picks the final history entry (`chapter_13_diagnostics.py:139-168`);
   diagnostic generation loads that as the outcome (`:774-782`). That alone does not prove
   it is the draw the pool targeted.
3. Missing/stale/invalid inputs as unavailable or errors, not valid misses. The current
   prediction loader defaults to an empty pool (`chapter_13_diagnostics.py:114`).
4. Frozen predictions and data cutoff before the evaluated outcome becomes available.

The source supports a broken default integration, not a universal historical assertion
that no grading was ever done anywhere. Manual CLI overrides exist
(`chapter_13_diagnostics.py:1017`, `:1047`); actual live use and historical evidence are
UNAVAILABLE here.

## 4. What is missing from the measurement plan

- **Final evaluation isolation.** “The sieve never saw the draw” is necessary but insufficient.
  Evaluation outcomes must also be unseen by training, ranking-target construction,
  feature fitting, Optuna selection, threshold tuning and configuration selection for that
  evaluated forecast. Training holdout labels are not automatically final test data.
- **Correct unit and baseline.** For each draw record one exact membership hit, and use
  `k = number of distinct predicted values`. Under the uniform 000–999 null, probability is
  `k/1000`. If pool sizes vary, report expected hits as the sum of each draw's baseline.
  Duplicated values or multiple winning entries must not inflate hits or coverage.
- **Fair comparison and uncertainty.** Evaluate both configurations on the same target draws,
  with comparable coverage/budget or explicit accounting for differences. Declare the
  primary pool size, evaluation period, skip/abstention policy and comparison rule before
  inspecting results. Report uncertainty; a point estimate above baseline is not by itself
  reliable evidence of lift. Use a further untouched period if results guide configuration
  choice. Preserve per-draw records so dependence and repeated comparisons can be assessed.

These requirements concern performance measurement, not state recovery. They do not require
a large foundational fleet campaign. A small scoring-format positive control is an
appropriate first integration check before performance data are collected.

Priority: define and connect the prediction-to-outcome grading contract first; investigate
CPU projection semantics alongside it. Any selected code fix must be versioned before the
baseline comparison so its results are attributable. An external frozen-output grader
could measure the existing box without waiting for all Chapter 13 automation repairs.

## 5. Additional documentation correction

The new upstream `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md:97-98` attributes this claim to
Beta: forward plus reverse constant-skip matches cannot exceed window size. It is false
without an additional condition. For observations and generated outputs `[1,2,3]`, forward
matches are 3 and reversed-target matches are 1, so their sum is 4 > 3. The centre contributes
in both directions. Palindromic windows give stronger counterexamples.

For one fixed generated sequence, the correct bound is `F + R <= k + c`, where `c` counts
positions whose chronological and reversed target values agree. This does not automatically
bound scores maximized at different skips. Correct that statement and retain the CPU
observations as examples rather than a universal rejection law. This documentation issue
does not reinstate a LIVE objection to the sieve.

## Disposition

Accept realignment. Reclassify the prior objection as THEORETICAL. Keep the existing box as
baseline; compare alternatives by reliable prediction lift. Investigate the two named
evaluation issues with the additions above. Neither current predictive success nor failure
has been established by this review.

END OF REVIEW
