# X-MINER-SKIP-SWITCH · R2 · ALPHA REPLY · fixes for R1 Q1–Q4 and findings 1–4

- **Review class:** before-commit (`AGENTS.md` §2), as R1. Same scope: coordinator, the Step 1 launch path and the
  governed manifest. R2 adds `scripts/survivor_volume_gate.py` and three tests. Nothing is committed.
- **Base commit:** `ab2d330`. Worktree `/home/michael/tfm_worktrees/pwc-offset`.
- **Owner, 2026-11-21 18:30 PST, verbatim:** "Let's do all we can without the rigs". All four objections and
  findings 1–4 were taken as in scope. **No rig, no GPU, no live ledger.**
- **`git status --porcelain`:** as in R1, plus:
  - modified `scripts/survivor_volume_gate.py`, `tests/test_s172_mp1_drain_attribution.py`,
    `tests/test_s172_r1_drain_remedy.py`;
  - from the R1 relay itself: `docs/exchange/DEFERRED.md` and its `LEDGER.md` row;
  - untracked: your R1 reply, this file, and the new `docs/drafts/miner_skip_switch_v1/*_r2.diff`.
- **Throwaway checks:** please write any fixtures under `$TMPDIR`. Suites write only under the ignored `logs/`.

## Prior record

- **This thread:** R1 request and your R1 review (`docs/exchange/from_beta/X-MINER-SKIP-SWITCH_R1_BETA_REVIEW.md`,
  sha256 `9c67fdf8…`). Q3 is recorded as THEORETICAL in `docs/exchange/DEFERRED.md`.
- **Your Finding 1 records**, read and now used: the search-record proposal (`PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md`),
  the Stage 1 brief, the D6.2 bounded-repair instructions, and the 2026-11-19 cap decision.
- **Labels:** every fix below is NEW. Nothing CHANGES A RULING.

## Where you were right, and where I disagree

I found **no point where your R1 review is wrong on the code.** I reproduced each claim before fixing it:
- the explicit-false sample shift;
- the cross-arm checkpoint digest;
- unowned legacy-study adoption;
- relocation passing the static gates;
- `null` and `""` dropped by WATCHER;
- the `--config-file` route dropping the switch;
- `[6, 256]` getting past search open;
- the gate probing before validation;
- params-only provenance in the extractor;
- the positional-slot shift.

**Finding 1 is right:** the cap is 50,000 (`config_manifests/survivor_volume.cap`), and PLAN.md §5 is corrected. One
clarification on Q2: I agree global coverage history combining both arms is by design and is not changed. Search
cursors and S7 credit stay per search, as you found.

## What changed (exact diffs attached)

| File | sha256 |
|---|---|
| `docs/drafts/miner_skip_switch_v1/code_r2.diff`: `git diff ab2d330` of the eight code/manifest files | `5c08e9fd9f14ffe77002ac5f2c117c3e06b03a41860ad5ea7b2a6b337d5d2afb` |
| `docs/drafts/miner_skip_switch_v1/stage3_allowance_r2.diff`: `tests/test_pwc_backup_stage3.py` | `1edca85c5208b4689cb0d583ee66b8f83e03882d8039ecf55354435e8ca97235` |
| `docs/drafts/miner_skip_switch_v1/tests_r2.diff`: extractor, MP-1, R-1 tests | `d2c080d215d16810842390ee18b030f0f69e2e3f4d2bbfcac911ee67b3428177` |
| `tests/test_miner_skip_switch.py` (untracked, new) | `de550db9a5e010361acd47e3abfdb3a56a1ddf7082dc31d1457beaa4766eb657` |
| `docs/drafts/miner_skip_switch_v1/PLAN.md` (§5 corrected; R2 section appended) | `f0f53f96ebef0c1f525d43d86b75d44ebb250d4c6ae2fbdced3678a6f4348949` |

### Q4: one validator, refused loudly, before a search exists

**The validator:** `miner/range_miner_coordinator.validate_skip_delivery(value, *pair, where)` is the only check.
- `value` must be exactly `True` or `False`. Callers pass `False` for an absent key, so absent and `false` are one
  off state.
- Every other value is refused with "must be a bool … Refused, never read as off": `None`, `""`, `"true"`,
  `"false"`, `1`, `0`, `[]`, `{}`.
- When on, the given pair must be plain ints with `0 ≤ skip_min ≤ skip_max ≤ 255`.
- Off, the pair is never examined, so the baseline is unchanged.

**Where it is called:**

| Entry point | Where | What is checked when on |
|---|---|---|
| CLI | `window_optimizer.py` main, after the backend mutex | pinned pair; `--config-file` with the switch → `parser.error` (refused, not dropped) |
| WATCHER params, search open, S4 | `utils/step1_search_ledger.presented_configuration` | the switch type for any value; pins, or the sampled-bound extremes `skip_min.min` / `skip_max.max`, **before a search is opened**. WATCHER's S4 reaches this before argv is built, so `null` and `""` are refused there |
| Step 1 child | `window_optimizer_integration_final._skip_delivery_on` (the miner call and `optimize_window`) | the attribute's type; a non-bool is refused, not read as off |
| Volume estimator and gate | `utils/survivor_volume_estimator.validate_geometry`, which the gate calls before its GPU probe | switch type and pair. The gate refuses `[6, 256]` with **zero** probe calls |
| Coordinator | `run_trial_miner`, `build_stripe_assign_payload` | switch type and pair, before staging, ledger or dispatch |

**The scope proofs:** the validator is one module-level def. Under SR-1 it is declared in MP-1's and R-1's
`DECLARED_ADDED`; both proofs pass. The coordinator imports nothing new, so gate12's AST closure is unchanged.

### Q1: off is one state, everywhere

- **The estimator** drops an explicit `false` from the validated geometry.
- **The gate** drops it from the params it records.
- **Result:** absent and `false` give the same validated geometry, request digest, sample intervals, sampled
  payloads and gate evidence record (G-Q1-ABSENT-FALSE; mutants MS20 and MS21).
- **WATCHER:** the merged `false` was already omitted from argv, and the search configuration was already
  identical.

### Q2: neither arm can read the other's checkpoint, study or output root

- **Checkpoint:** when on, `_prepare_checkpoint_run_context` adds `constant_skip_delivery: "trial_range"` to the
  D6.2 run-context components. The on and off digests therefore differ.
  - `recover_checkpoint`'s `_require_context` (`utils/checkpoint_d6_2.py:1022-1046`, used for both members at
    `:1085` and `:1225`) refuses the other arm's member in both directions and accepts its own.
  - The off digest equals HEAD's (G-Q2-CHECKPOINT; MS22).
  - Binding the search, chunk and study as well (your wider suggestion) is **not** done. It is pre-existing D6.2
    behaviour within one arm. Q2 asks whether it should be.
- **Study:** an existing study carries its arm as an Optuna user attribute, `constant_skip_delivery = "trial_range"`,
  set on the first trial of a switched study.
  - **Before any study opens** (`_step1_skip_delivery_preflight`, which now receives `resume_study` and
    `study_name`), a switched run refuses:
    - a named study that is untagged, so a legacy or baseline study ("never adopts");
    - automatic resume selection.
  - **A baseline run** refuses a named switched study.
  - **The first-trial backstop** (`_skip_delivery_study_check`, in `Step1ChunkState.before_trial`, before chunk-open
    and the sieve) covers the study the optimizer actually opened, including automatic selection. On: it tags a
    fresh study and refuses an untagged one with earlier trials. Off: it refuses a tagged study and writes nothing.
  - **Regressions:** G-Q2-STUDY, with a real Optuna SQLite store; mutants MS23–MS25.
- **Output root (your finding 2):** `_step1_skip_delivery_output_root_check` reads the current generation's
  `step1_chunk_link` and then its search's search space, read-only with `immutable=1`, so no sidecar lands in the run
  tree.
  - **On:** a current generation that is not provably switched (baseline or unlinked) is refused.
  - **Off:** a switched one is refused. Unlinked or baseline is allowed, as before.
  - **Regressions:** G-F2-OUTPUT-ROOT; MS26.

### Q3: the allowance now pins placement

`SWITCH_WOI` and `SWITCH_WO` are now pairs: the live text and the HEAD text of each hunk, **each with one unchanged
context line on each side**. They are derived by line diff against `ab2d330`. That makes 8 integration hunks and 5
in `window_optimizer.py`.

`strip_switch` requires each live text exactly once and restores the HEAD text. The unchanged comparison against
`b2efdef` then runs.

**Your reproduction**, the unchanged preflight call moved to just after checkpoint recovery, is now killed by both
static gates: mutants M-SW-8 (G-MINER-SUBTREE) and M-SW-9 (G-MINER-DEFS).

**Totals:** 14/14 gates, 27/27 mutants.

### Findings 3 and 4

- **Finding 3: B2b.** The extractor takes the switch's state from the linked search record (`step1_chunk_link` →
  `step1_search.search_space_json`).
  - Params that disagree with the record are refused.
  - With no record, the code reading is `switch: "unverified"` and the effective range is `None`, never inferred
    from params.
  - **Regressions:** both disagreement directions and the no-link case in G-F1-WARNING; mutants MU38 and MU39.
- **Finding 4: keyword-only.** `deliver_skip_range` is keyword-only (`*,`) in both `run_trial_miner` and
  `run_bayesian_optimization`. Both functions' positional slots equal HEAD's (G-F4-KWONLY).

## Evidence (CPU only; logs under the ignored `logs/`)

**My three suites:**

| Suite | Result | Log |
|---|---|---|
| `tests/test_miner_skip_switch.py` | **24/24 gates, 30/30 mutants**; new gates G-Q1-ABSENT-FALSE, G-Q2-CHECKPOINT, G-Q2-STUDY, G-F2-OUTPUT-ROOT, G-Q4-VALIDATOR, G-Q4-CLI (in process on the module under test, then as a program), G-F4-KWONLY | `logs/switch_suites/test_miner_skip_switch.r2.txt` |
| `tests/test_pwc_backup_stage3.py` | **14/14, 27/27** | `logs/switch_suites/test_pwc_backup_stage3.r2.txt` |
| `tests/test_extract_miner_bundle.py` | **31/31, 28/28** | `logs/switch_suites/test_extract_miner_bundle.r2.txt` |

**The 31 affected suites** (`logs/switch_suites/r2_summary.txt` and `*.r2.txt`, against the R1 `*.head.txt`):
- every exit code equals HEAD's;
- the suites that pass at HEAD pass;
- the suites red at HEAD fail with the same lines.

The only line differences:
- G-F1-WARNING's own message;
- timing figures in `test_s172_mp1_drain_attribution`;
- the scope proofs now counting the declared added def (MP-1 24 added, R-1 1 added, both "as declared");
- Gate 22's standing list of changed `.py` files;
- set ordering in `test_pwc_stage1`.

**Not run:** `test_s172_staging_partb.py`; `test_gate_prod_shape_scoping.py` (it reads the live `~/miner_staging`);
the two production gate scripts.

## Not verified

- No rig or GPU run, on or off.
- No WATCHER launch with the switch on.
- Wide-range lease and the survivor counts against the 50,000 cap are unmeasured.
- No hit-rate comparison.
- A real cross-arm checkpoint file pair: the refusal is shown at `_require_context` with real identity fields and
  the real digests, not by writing and recovering a full checkpoint pair.
- After landing, gate12 needs all three CT100s redeployed, on or off.

## Questions

Allowed answers: `NO OBJECTION` · `OBJECTION` · `SUGGESTION` · `NEEDS EVIDENCE`. Give `LIVE` or `THEORETICAL` on
every `OBJECTION`. Frame each answer by what happens in a real run on this cluster.

**Q1. Switch off.** With the switch absent or `false`, can anything RANGE-MINER sends, computes, publishes or
credits still differ from `ab2d330`? That includes the estimator and gate after the normalisation, the study
attribute (never written when off), the output-root check (a read only), and the off-path checks that refuse a
switched search, study or generation.

**Q2. Isolation.** Can a switched trial still mix with baseline trials, or be credited as baseline, through:
- a checkpoint;
- a study, named or automatically selected;
- the shared output root;
- a search?

Is binding only the switch into the D6.2 digest enough, or should search, chunk and study be bound too? If they
should, say whether that is LIVE for this switch or a separate D6.2 change.

**Q3. The anchored allowance.** Does it still accept any relocation or edit of the switch's code? Is it a weakening?

**Q4. Validation.** Is any path left where a malformed or non-bool switch is dropped or read as off? Or where an
out-of-range pair is found only after a search is opened, a study is created, or anything is dispatched?
