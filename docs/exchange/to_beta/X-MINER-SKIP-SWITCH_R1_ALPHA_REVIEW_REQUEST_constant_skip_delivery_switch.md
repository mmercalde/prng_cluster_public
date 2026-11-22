# X-MINER-SKIP-SWITCH · R1 · ALPHA REVIEW REQUEST · RANGE-MINER constant-skip delivery switch

- **Review class:** before-commit (`AGENTS.md` §2). Coordinator code (`miner/range_miner_coordinator.py`), the
  Step 1 launch path (`window_optimizer.py`, `window_optimizer_integration_final.py`) and the governed manifest
  (`agent_manifests/window_optimizer.json`). Nothing is committed. The owner hand-lands `miner/` and
  `agent_manifests/`.
- **Base commit:** `ab2d330` (main). Worktree `/home/michael/tfm_worktrees/pwc-offset`, detached at `ab2d330`.
- **`git status --porcelain` at the time of writing:**
  - modified: `agent_manifests/window_optimizer.json`, `docs/exchange/LEDGER.md` (one restored relay row,
    X-SKIP-F1 R1),
    `miner/range_miner_coordinator.py`, `scripts/extract_miner_bundle.py`, `tests/test_extract_miner_bundle.py`,
    `tests/test_pwc_backup_stage3.py`, `utils/step1_search_ledger.py`, `utils/survivor_volume_estimator.py`,
    `window_optimizer.py`, `window_optimizer_integration_final.py`;
  - untracked: `data/`, `docs/drafts/SKIP_DESIGN_RECORD_20261121.md`, `docs/drafts/miner_skip_switch_v1/`,
    `docs/drafts/pwc_backup_stage4_v1/PARKED_MINER_PHASE_CAPTURE.md`,
    `docs/drafts/pwc_backup_stage4_v1/SCOPE_MINER_PHASE_SETS.md`, `docs/exchange/from_beta/X-SKIP-F1_R1_BETA_REVIEW.md`,
    `docs/exchange/to_beta/X-SKIP-F1_R1_ALPHA_REVIEW_REQUEST_skip_range_design_record.md`,
    `tests/test_miner_skip_switch.py`, and this request.
  - The parked capture work, unrelated and not under review, is outside the tree in the ignored
    `logs/parked_capture/`.
- **Throwaway checks:** please write any fixtures under `$TMPDIR`, which this run points outside the repository
  (`AGENTS.md` §3). Fixtures written inside a worktree caused a boundary violation before (X-PWC-STAGE2B). The
  suites below write only under the ignored `logs/`.
- **No rig, no GPU, no live ledger** were used for this work.

## Prior record

The plan's own `## Prior record` (`docs/drafts/miner_skip_switch_v1/PLAN.md`) lists the searches and the records.
In short:

- **The design reading:** `docs/drafts/SKIP_DESIGN_RECORD_20261121.md`, F-1 traced, with §7 holding your opinion
  and the owner's choice. ALREADY RECORDED.
- **Your opinion, X-SKIP-F1 R1** (`docs/exchange/from_beta/X-SKIP-F1_R1_BETA_REVIEW.md`, ledger row 09:33). **Q2:**
  a run-selectable option, `[0, 16]` as the baseline, record requested and effective pairs and the selected
  behaviour. **Q4:** validate `0 ≤ skip_min ≤ skip_max ≤ 255` at the common assignment boundary; fresh study
  identity; the 5.53× and 33.1× cost; measure lease and volume; gate12 parity changes. ALREADY RECORDED; this
  change follows it.
- **The earlier scope:** `docs/drafts/pwc_backup_stage3a_v1/SCOPE_MINER_SKIP_RANGE_FIX.md` (drop point, minimal
  edit, option (B)). ALREADY RECORDED.
- **Stage 4 closed** with both engines at 0/0/13/2995 at the trial pinned to `[0, 16]`
  (`docs/RESULT_PWC_BACKUP_STAGE4_20261121.md`). ALREADY RECORDED. It is the baseline this switch must not move.
- **New:** the switch, its route and its guards. **CHANGES A RULING:** none.

## Fixed facts (owner; not under review)

1. **Owner, 2026-11-21 09:54 PST, verbatim:** "sounds resonalble". That is the choice of the switch, with
   `[0, 16]` as the baseline, built after PWC Stage 4 as its own reviewed change, and settled by a paired hit-rate
   comparison on unseen draws.
2. **Owner, 16:54 PST, verbatim:** "yes" (start the job). He was told it goes to you before landing.
3. **The switch builds the option only.** Its value is a hit-rate question (`k/1000` on unseen draws), not part of
   this change. Your answers advise; Michael decides.

## What to read

| File | What it is | sha256 |
|---|---|---|
| `docs/drafts/miner_skip_switch_v1/PLAN.md` | the plan: route, record, validation, study isolation, risks, off-proof, pinned files | `75f437072e6f4d56bedabf8269832ef2bf93991c3767725224e2a838f15057dd` |
| `docs/drafts/miner_skip_switch_v1/code.diff` | `git diff ab2d330` of the seven code and manifest files | `b5030dc108c1a20d32d13c53716b0abc8d90921fb93e07cdd046630bd450def7` |
| `docs/drafts/miner_skip_switch_v1/stage3_allowance.diff` | `git diff ab2d330 -- tests/test_pwc_backup_stage3.py` | `dd830fc21124577fce61a981e89719b414e7e538a73b290f918d3ee73cdde704` |
| `docs/drafts/miner_skip_switch_v1/extractor_test.diff` | `git diff ab2d330 -- tests/test_extract_miner_bundle.py` | `f14bdf12f047f9cefa2413490b9418a57207ec5d065a85b749d0398e6749f251` |
| `tests/test_miner_skip_switch.py` (new) | the switch suite | `a47d7985f313a47914d331ef23a653dbbe3f2191e30175207a40876fdbf4ab82` |

**The change in one table** (detail in PLAN.md §1):

| Hop | Off (default) | On |
|---|---|---|
| manifest `default_params.deliver_skip_range: false` | WATCHER omits a false bool from argv (`agents/watcher_agent.py:2418-2422`), so `EXEC CMD` is unchanged | `--params` `true` → `--deliver-skip-range` via the underscore fallback (`:2415`). The key must be declared, or WATCHER drops it (`:2024-2034`) |
| `window_optimizer.py` | no coordinator attribute set | `coordinator.deliver_skip_range = True`. `parser.error` without `--use-range-miner` |
| integration miner call | `run_trial_miner` gets no new argument (conditional `**{}`) | `deliver_skip_range=True` |
| integration `optimize_window` | `_step1_skip_delivery_preflight(ctx, False)` refuses only a switched-on search | refuses no search, or a search opened without the switch |
| `utils/step1_search_ledger.presented_configuration` | key absent, so the digest and search id are unchanged | `search_space.constant_skip_delivery = "trial_range"`, so the search id differs |
| `run_trial_miner` | nothing new | validate the pair (plain ints, `0..255`, min ≤ max) before the staging dir exists; record it per run; result carries `constant_skip_delivery` |
| `_dispatch_pending` (arity unchanged) | hands `skip_range=None` | hands the run's pair |
| `build_stripe_assign_payload(..., skip_range=None)` | no key, so the payload is byte-identical | constant phases add `"skip_range": [a, b]` after the same range check; phases 3–4 are unchanged |
| `utils/survivor_volume_estimator.py` | key absent from the geometry: unchanged | the geometry's `deliver_skip_range: true` samples the same payloads a run dispatches |

**Not changed:** `miner/range_miner_worker.py`, which already reads `payload["skip_range"]` at `:1293`, the
kernels, `prng_registry.py`, PWC, `agents/watcher_agent.py`, the finalizer, the NPZ contract and the canonical
records.

**Why the range check is inline and not a new function:** the MP-1 and R-1 AST scope proofs pin the coordinator's
added-definition set exactly. `tests/test_s172_r1_drain_remedy.py:2165-2170` records the same choice for R-1's own
repair. All three changed functions were already in both suites' `DECLARED_CHANGED`. A first version with a new
`def` redded both; this version passes both.

## Evidence (CPU only; logs under the ignored `logs/`)

- **`tests/test_miner_skip_switch.py`: 17/17 gates, 21/21 mutants.** `logs/test_miner_skip_switch_run5.txt`.
  **Off:** each check compares with the HEAD code loaded from `git show HEAD:<file>` beside the changed file:
  - `build_stripe_assign_payload` gives identical canonical bytes for phases 1–4;
  - `_dispatch_pending` sends identical `StripeAssignMessage` payloads;
  - `run_trial_miner`, through its `_serve` seam, builds the same serve context (minus the per-call run id and
    the staging filesystem's live free-bytes figure) with no delivery record on the coordinator;
  - `run_bidirectional_test` hands `run_trial_miner` the same 40 arguments;
  - `run_bayesian_optimization` builds the same coordinator attributes;
  - the `presented_configuration` digest is the same with the key absent or false;
  - the estimator request digest and sampled payloads are the same;
  - 8 record, NPZ, finalizer and worker files are byte-identical to HEAD and never name the switch.

  **On:**
  - phases 1–2 add exactly `skip_range`; phases 3–4 are byte-identical to HEAD;
  - the pair reaches the phase-1 dispatch for its own run only;
  - the worker's own parse feeds `BuildContext` and `_constant_prefix`, giving int32 skip scalars `[0, 16]` off
    and `[6, 99]` on;
  - the validation refusals fire at both sites, before anything exists;
  - the study and search refusals fire;
  - the CLI exits 2 without `--use-range-miner`.
- **`tests/test_pwc_backup_stage3.py`: 14/14 gates, 25/25 mutants**, the 18 earlier plus 7 new.
  `logs/switch_suites/test_pwc_backup_stage3.final.txt`.
- **`tests/test_extract_miner_bundle.py`: 31/31 gates, 26/26 mutants.** `logs/test_extract_miner_bundle_switch.txt`.
- **29 affected suites, each run against HEAD's six source files and then the changed ones, sequentially:**
  `logs/switch_suites/summary.txt`, `*.head.txt` and `*.live.txt`.
  - The suites that pass at HEAD pass on the final files: `logs/switch_suites/final_summary.txt`, rerun after the
    inline change, plus `*.live2.txt` for the two scope proofs.
  - The suites red at HEAD fail with the same lines.
  - The only line differences are line-number shifts, set ordering in `test_pwc_stage1` output, and Gate 22's
    standing reaction to changed or untracked `.py` files.
  - Not run: `tests/test_s172_staging_partb.py` (never run), `tests/test_gate_prod_shape_scoping.py` (reads the
    live `~/miner_staging`), and the production gate scripts `tests/gate_s172_prod_shape.py` and
    `tests/capture_run4_clean_control.py`.

## The Stage 3 guard allowance (exact diff in `stage3_allowance.diff`) and why it is not a weakening

**Before:** G-MINER-SUBTREE, G-MINER-DEFS and G-WO-FUNCS compared the integration file and `window_optimizer.py`
with `b2efdef`, minus the declared Stage 3 PWC insertions. G-MANIFEST allowed a fixed set of key paths.

**After:**
- **`SWITCH_WOI` and `SWITCH_WO`** hold the switch's added text, **copied verbatim from the files**: 3 blocks in
  the integration file and 5 in `window_optimizer.py`.
- **`strip_switch()`** requires each to occur **exactly once**, removes it, and then the **unchanged** comparison
  runs.
- **G-MANIFEST** allows exactly one more added path, `/default_params/deliver_skip_range`, and asserts its value is
  `false`. It also asserts that WATCHER still omits a false bool from argv.

**Why that is not weaker:**
- **Any other change still fails:** the earlier 18 mutants are all still killed, and M-SW-3 (one extra statement
  beside an allowed block) is killed.
- **Any edit inside an allowed block fails,** because the verbatim text no longer occurs. M-SW-1 (the miner
  keyword's condition altered), M-SW-2 (the preflight helper's check removed) and M-SW-4 (the coordinator attribute
  set when off) are killed.
- **The manifest default must stay `false`:** M-SW-5 is killed.
- **The runtime gates G-MINER-SETTINGS and G-MINER-JOBS carry no allowance at all.** They still prove the off state
  equal to `b2efdef`, and kill M-SW-6 (the attribute set while off) and M-SW-7 (the switch passed while off).
- **What the allowance does not do** is compare the allowed blocks with the base, because they are new code. Their
  behaviour is what `tests/test_miner_skip_switch.py` tests.

## Why `scripts/extract_miner_bundle.py` and its test changed

**The problem:** the extractor's F-1 code reading inferred which skip range phases 1–2 ran. It looked at the
worker's literal default and whether the payload builder names `skip_range`. With the switch, the builder names
`skip_range`, so the old reading reported "not known" for every run, which reds its G-CLEAN.

**The fix:** it now recognises the switch, a keyword-only `skip_range` defaulting to `None`. It reads the effective
range from the run's own params (`deliver_skip_range`), which its B2 check binds to the trial: `[0, 16]` when off,
the trial's pair when on. A builder that names `skip_range` unconditionally still reads as "not known".

**The test:** G-F1-WARNING gains the switched-on case (effective `[6, 99]`, no F-1 warning) and keeps the
unconditional case.

## Not verified

- **No rig or GPU run with the switch on.** That the constant kernels loop over the delivered pair rests on the
  worker's CPU argument builder and a code reading (`miner/range_miner_worker.py:1293`, `:1385-1390`, `:259-271`).
- **No WATCHER launch with the switch on.** The argv, merge and S4 pieces are tested separately.
- **Wide-range lease and survivor-cap behaviour are unmeasured.** The 300 s lease and the 25,000-per-phase cap were
  measured only at `[0, 16]`. `run_trial_miner` logs a warning above 17 hypotheses and changes no lease or cap.
- **No hit-rate comparison.**
- **Rig redeploy:** the coordinator is in the gate12 governed set, so after landing, every launch, on or off, needs
  all three CT100s redeployed. Not done.
- **Not covered by the switch's study isolation:** a baseline run **outside** any search that uses `--resume-study`
  with no study name auto-selects the newest `optuna_studies/window_opt_*.db`
  (`window_optimizer_bayesian.py:676-708`). WATCHER never takes that path (`agents/watcher_agent.py:2314-2318`). It
  is recorded in PLAN.md §4, not changed.

## Questions

Allowed answers: `NO OBJECTION` · `OBJECTION` · `SUGGESTION` · `NEEDS EVIDENCE`. Give `LIVE` or `THEORETICAL` on
every `OBJECTION`. Frame each answer by what happens in a real run on this cluster.

**Q1. Switch off.** With `deliver_skip_range` absent or false, can anything RANGE-MINER sends, computes, publishes
or credits differ from `ab2d330`? Consider at least:
- the assign payload bytes;
- the serve context and result;
- the trial context and ledger;
- the manifests, the assembly, the canonical records, the NPZ and the sidecar;
- the Step 1 search identity, credit and coverage;
- WATCHER argv and the merged params;
- the volume estimator and gate.

**Q2. Switch on: isolation.** Can a switched-on trial ever mix with baseline trials? That means the same Optuna
study, the same Step 1 search, or a checkpoint resume. Can it be credited as baseline coverage, or can baseline
coverage be credited to a switched-on search? The design relies on the search identity (`constant_skip_delivery`
in `search_space`), the child preflight and S9 study ownership. Please check the coverage ledger's identity
(`canonical_coverage_identity`, which the switch does not change) and say whether a switched-on interval could
satisfy a later baseline cursor, or the reverse.

**Q3. The Stage 3 guard allowance.** Is it a weakening? See `stage3_allowance.diff` and the argument above.

**Q4. Validation.** Is `0 ≤ skip_min ≤ skip_max ≤ 255` with plain ints complete on every path that can reach a
constant kernel with the switch on? The paths:
- WATCHER `--params`, including a non-bool `deliver_skip_range` such as `"true"` or `1`, which WATCHER would emit
  as `--deliver-skip-range true`;
- the CLI;
- Optuna-sampled and pinned pairs;
- a resumed trial;
- the volume estimator and gate. The estimator's own geometry check enforces ints and min ≤ max but no 255 bound;
  with the switch on it reaches the builder's check.

Is anything left unvalidated, or validated only after something has been created or dispatched?
