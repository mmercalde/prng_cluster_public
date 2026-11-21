<!--
DRAFT. NOT RELAYED. The Stage 3b section of the combined Stage 3 Beta request.
The lead assembles the request with agent 2's Stage 3a section beside this one, under a single header (id/round,
review class, base commit), a combined Prior record, the owner's framing and fixed facts, and one positions table.
Question numbers here are 3b-Q1 … 3b-Q8 so they cannot collide with 3a's.
Base for this section: main b2efdef, worktree /home/michael/tfm_worktrees/pwc-backup (uncommitted).
-->

## Stage 3b — the shared launch side (worktree `/home/michael/tfm_worktrees/pwc-backup`, base `b2efdef`)

### Prior record (3b items; the combined record above covers the governance search)

| Document | What it settles | Label |
|---|---|---|
| `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:20-23`, `:14` | Ruling 2 (PWC earns credit) and ruling 3 (no seed-start refusal), 2026-11-19 20:57 PST; default rule "PWC does what RANGE-MINER does" | **Fixed facts** (owner) |
| Owner ruling 2026-11-20 15:53 PST, verbatim: "Yes, range miner is the new default" (rulings file `:46`, this worktree) | RANGE-MINER is the Step 1 default; PWC only by its flag | **Fixed fact** (owner). **CHANGES A RULING:** `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:436`, `:503` (default only after Phase 7); `docs/TODO_CONSOLIDATED.md:93`. Not yet in a committed rulings file |
| Owner ruling 2026-11-20 20:24 PST, "Yes": finish the PWC backup first, then re-enable n_parallel > 1 as its own job (rulings file `:49`, this worktree) | The parallel path is out of Stage 3; S3-6 (the self-killing `pkill`) and S3-7 (the partition dial address) go with that job | **Fixed fact** (owner) |
| Owner, 2026-11-20 (this session): edit only the named PWC lines of three existing suites; add one keyword to `agents/watcher_agent.py` | The test edits and the WATCHER edit in this section | **Fixed facts** (owner) |
| `docs/exchange/from_beta/X-PWC-STAGE2B_R1_BETA_REVIEW.md:34`, `:59-61` (your Q5) | An incomplete PWC trial enters Optuna COMPLETE and scores | ALREADY RECORDED. Addressed here |
| `docs/exchange/from_beta/X-PWC-STAGE2B_R1_BETA_REVIEW.md:67` (your Q8) | Allowing all of `run_bidirectional_test` would hide a miner-branch change; keep a separate comparison of that subtree and its preamble | ALREADY RECORDED. Done here (statement-level, with mutants) |
| `miner/step1_ingress.py:192-200`; `window_optimizer_integration_final.py:1350` | RANGE-MINER refuses a trial with no committed assembly before ingress | The behaviour PWC now copies |
| `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R1.md:14`, `:94`; `agents/watcher_agent.py:511-519` | "`default_params` untouched" concerns the seven Run-4 warm-start keys | Not in conflict: engine choice is not one of them. Please confirm |
| `docs/CHAPTER_12_WATCHER_AGENT.md:1220`; `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_7_SOAK.md:20`; `docs/BACKLOG.md:298-300` | n_parallel > 1 SWITCHED-OFF-BY-RULING on the miner path | ALREADY RECORDED |

### Files under review (absolute paths; `git -C /home/michael/tfm_worktrees/pwc-backup diff b2efdef`)

| File | Change |
|---|---|
| `/home/michael/tfm_worktrees/pwc-backup/docs/drafts/pwc_backup_stage3b_v1/README.md` | the package README; read it first |
| `/home/michael/tfm_worktrees/pwc-backup/window_optimizer_integration_final.py` | the `if _use_pw:` block of `run_bidirectional_test`; seven new PWC-only helpers (`:1406-1484`) |
| `/home/michael/tfm_worktrees/pwc-backup/window_optimizer.py` | the seed-start guard narrowed to ZMQ; a dial-address parameter, attribute (PWC runs only) and CLI flag; `require_fleet` for PWC at the dataset gate |
| `/home/michael/tfm_worktrees/pwc-backup/agent_manifests/window_optimizer.json` | the default flip (15:53 ruling); one null `pwc_coordinator_address` key |
| `/home/michael/tfm_worktrees/pwc-backup/agents/watcher_agent.py` | one keyword: `require_fleet=bool(final_params.get("use_persistent_workers"))` |
| `/home/michael/tfm_worktrees/pwc-backup/tests/test_s172_miner_seed_start.py`, `.../tests/test_step1_search_record_stage1.py`, `.../tests/test_s172_phase1_scaffolding.py` | named PWC lines only |
| `/home/michael/tfm_worktrees/pwc-backup/tests/test_pwc_backup_stage3.py` | new |
| `/home/michael/tfm_worktrees/pwc-backup/test_persistent_worker_harness.py` | **T20 only** (owner yes, 2026-11-20): its mock PWC result now carries a complete `pwc_trial_completion_v1` record; AST proof that nothing else changed: `t20_only_proof.txt` |

Evidence is in `docs/drafts/pwc_backup_stage3b_v1/`:
- `pwc_backup_stage3b.diff`, which applies to `git archive b2efdef` and reproduces all 10 files byte for byte;
- `miner_subtree_statement_comparison.txt`;
- `watcher_identical_except_one_call.txt`;
- `old_suites_pwc_lines_only_proof.txt`;
- `focused_*.txt`.

### What changed

- **Ruling 3.** The PWC seed-start refusal is removed in both places:
  - `assert_step1_route_seed_start_supported` refuses ZMQ only;
  - the integration's PWC call to it is gone, and `run_trial_persistent` gets `seed_start`.
- **The PWC call** now also passes `pwc_coordinator_address` (a new `--pwc-coordinator-address` flag with no default;
  RANGE-MINER takes its coordinator host the same way, `scripts/launch_fleet_manual.sh:147`, `:239-241`) and the
  coordinator's ready-gate count. That count was never passed, so it was 1.
- **Your Q5.** Before `_build_test_result_from_pw`, the PWC block raises `PwcTrialIncompleteError` unless:
  - the completion record has schema `pwc_trial_completion_v1`;
  - the trial is complete;
  - every required pass is `complete` and ran on TCP.

  It raises `PwcThresholdProvenanceError` unless every required pass has a verified threshold record. Nothing reaches
  the accumulator, and Optuna sees an exception, not a COMPLETE score, as with RANGE-MINER's `MinerIngressError`.
- **Ruling 2.** The credit evidence is `_step1_pwc_sweep_evidence`: the swept intervals come from PWC's own record,
  and only when the checks above all hold. `credit_decision` is unchanged.
- **15:53 ruling.** The manifest defaults to `use_range_miner: true` and `use_persistent_workers: false`. The new
  `pwc_coordinator_address: null` is skipped by WATCHER (`if value is None`), so it never reaches a command line.
- **PWC fails closed on a missing provisioning manifest, as RANGE-MINER does:**
  - on the CLI: `run_start_dataset_gate(require_fleet=…)`;
  - in WATCHER: `resolve_absent_fleet_status(require_fleet=…)`, the one-keyword edit.
- **Not done, by the 20:24 ruling:** S3-6 and S3-7, both in the n_parallel > 1 path.

### Real-world scenarios for 3b

- **3b-S1. A WATCHER Step 1 with no engine in its params.**
  - Before: it went to PWC, which refused at startup (no dial address; sizing outside the stable envelope).
  - Now: it goes to RANGE-MINER, with RANGE-MINER's preconditions: the fleet, a provisioning manifest, and admission
    by the manifest's `worker_pool_size`.
- **3b-S2. Run 4 as launched today** (`gate12_launch.sh:465-473`, both engine flags explicit). The merged params,
  argv, coordinator attributes and `run_trial_miner` arguments are identical to `b2efdef` (G-MANIFEST,
  G-MINER-SETTINGS, G-MINER-JOBS).
- **3b-S3. A PWC run chosen by the flag.**
  - It must pass `use_persistent_workers: true` **and** `use_range_miner: false`, or the engine mutex refuses
    (`window_optimizer.py:1770-1782`).
  - It also needs `--pwc-coordinator-address` and chunk sizing inside the stable envelope; without them PWC refuses
    at startup.
  - With them, a seed start of 2³¹ is dispatched as such.
- **3b-S4. One-sided update.**
  - VM101 is updated while the rigs keep older PWC workers: a worker that does not echo what Stage 2/2b check makes
    its pass incomplete, so the trial is refused (not scored) and earns no credit.
  - Agent 2's coordinator (3a) and this integration (3b) are meant to land together. An older coordinator without
    `pwc_completion` is refused by the same check.
- **3b-S5. A PWC trial that loses a chunk** (timeout, or a worker of a different version). It raises before the
  accumulator, so Optuna marks the trial failed, as it would for a RANGE-MINER trial without a committed assembly.
  Whether that stops the whole study is the same as for the miner path.
- **3b-S6. A missing provisioning manifest on a PWC run.** It refuses before dispatch on both launch routes, as the
  miner does. ZMQ and legacy are unchanged.

### Questions (allowed answers: NO OBJECTION · OBJECTION · SUGGESTION · NEEDS EVIDENCE; LIVE/THEORETICAL on every OBJECTION)

**3b-Q1. RANGE-MINER, byte for byte.** Our evidence:
- statement-level AST of `run_bidirectional_test`: everything outside `if _use_pw:` is identical, including
  `if _use_miner:` and the preamble;
- the whole-file definition proof;
- the inside-function proof for the four edited `window_optimizer.py` functions;
- WATCHER identical except one keyword;
- run-time equality of the coordinator attributes and the `run_trial_miner` arguments for a Run-4-shaped launch.

By your own check: does any line of 3b change what a RANGE-MINER run does, through the CLI or through WATCHER?

**3b-Q2. The new default (15:53 ruling, a fixed fact).** Name any realistic WATCHER, CLI or agent launch on this
cluster that relied on the old PWC default and now does something unsafe on RANGE-MINER, rather than refusing.
Example: admission by the manifest default `worker_pool_size` 8. Also confirm that the Route-A "`default_params`
untouched" rule does not cover engine selection.

**3b-Q3. Choosing PWC.** Is "PWC by flag" safe as built? Name any route where PWC is selected without the dial
address or without a valid sizing, and does something other than refuse at startup.

**3b-Q4. Your Q5.** Name any path by which a PWC trial with incomplete coverage, a non-TCP pass or an unverified
threshold still reaches the accumulator or an Optuna score. Is raising (the RANGE-MINER behaviour) the right outcome
for the study?

**3b-Q5. Credit.** Does `_step1_pwc_sweep_evidence`, with the unchanged `credit_decision`, credit exactly the
complete, all-TCP, threshold-verified PWC trials, including a zero-survivor complete trial and constant-only trials?
Name a case where it credits more, or less.

**3b-Q6. One-sided deployment.** With VM101 updated and the rigs on an older PWC worker or `sieve_filter.py`, or the
reverse, name any sequence that yields a scored or credited partial result.

**3b-Q7. The ready gate.** The pipeline now passes the coordinator's `pwc_min_workers`, 24 on a full run, where PWC
used 1. Is there a real run on this cluster that this breaks, beyond refusing a short fleet before dispatch?

**3b-Q8. Test edits.** In the three existing suites only the named PWC checks changed, and in `test_persistent_worker_harness.py` only T20. Our evidence:
`old_suites_pwc_lines_only_proof.txt`, a whole-file comparison plus a list-entry diff. M9 is retired as an invalid
experiment, because `run_with_config` has no ZMQ flag and its guard can no longer fire. By your own comparison:
- did any RANGE-MINER gate, mutant or assertion change?
- is retiring M9 right?

### Test evidence (3b)

- **Focused:**
  - `tests/test_pwc_backup_stage3.py`: 14/14 gates, 18/18 mutants;
  - `tests/test_compare_engine_outputs.py`: 13/13;
  - the PWC suites and the D3.25 suite pass;
  - 7 shared or miner suites keep exactly their baseline failing checks: no pointer manifest in a worktree; no GPU;
    Gate 22 naming only uncommitted new files.
- **Full before/after pass (2026-11-20, `before_after_tally.md`, `before_after_failing_checks.txt`):** 77 suites, before on a clean `b2efdef`, after with the parked work restored (42/42 hashes, status and diff identical). 73 of 75 common suites have identical failing-check sets. Four differ: (1) `test_persistent_worker_harness.py` T20 failed: its mock PWC result had no completion record and expected to be scored; under Q5 it is refused by design (`window_optimizer_integration_final.py:2072`). The owner approved editing T20 only; after the edit the suite passes 25/25 (`t20_rerun_test_persistent_worker_harness.txt`); (2) `test_gate12_cleantree_admission.py` W-NO-WEAKENING and (3) `test_s172_phase4_coordinator.py` Gate 22 compare the working tree with HEAD and red on any uncommitted edit (`tests/test_gate12_cleantree_admission.py:941-951`, `tests/test_s172_phase4_coordinator.py:1626`); by reading, both clear once the files are committed (not executed); (4) `test_s172_miner_seed_start.py` G-R5-ROUTES fails at the same point under its new label. The new suites run green: `test_pwc_backup_stage3.py` 14/14 + 18/18, `test_compare_engine_outputs.py` 13/13.
- **UNAVAILABLE:** GPU runs, rigs, a real WATCHER run on the new default, and hit rate (Stage 4).
