<!--
DRAFT. NOT RELAYED. The Stage 3a section of the combined Stage 3 Beta request.
The lead assembles the request with agent 1's Stage 3b section beside this one, under a single header (id/round,
review class, base commit), a combined Prior record, the owner's framing and fixed facts, and one positions table.
Question numbers here are 3a-Q1 … 3a-Q6 so they cannot collide with 3b's.
Base for this section: main b2efdef, worktree /home/michael/tfm_worktrees/pwc-offset (uncommitted).
-->

## Stage 3a — the PWC-only items (worktree `/home/michael/tfm_worktrees/pwc-offset`, base `b2efdef`)

### Fixed facts for 3a (owner; not under review)

- **S3-13 is withdrawn.** Owner, 2026-11-20 21:32-21:35 PST, verbatim: "Well range miner must be fixed none the less"
  and "Please do whatever method is the quickest and least recoding".
  - This supersedes his 16:45 ruling ("Yes, PWC follows range miner. Unless you have a other way to verify they both
    return the same results") **for S3-13 only**.
  - The reason: the miner's constant phases not receiving the trial's skip range is a recorded open finding, F-1
    (`docs/TFM_SESSION_HANDOFF_20261114.md:76-80`; `miner/range_miner_worker.py:1293` falls back to `[0, 16]`).
    Copying it into PWC would be undone once RANGE-MINER is fixed.
  - PWC therefore sends the trial's skip range exactly as at `b2efdef`. `persistent_worker_coordinator.py` is
    byte-identical to `b2efdef`.
  - The miner fix is a separate future change, scoped read-only in
    `docs/drafts/pwc_backup_stage3a_v1/SCOPE_MINER_SKIP_RANGE_FIX.md`. It is not part of this review.
- **The default rule** for anything unruled: PWC does what RANGE-MINER does.

### Prior record (3a items; the combined record above covers the governance search)

| Document | What it settles | Label |
|---|---|---|
| Owner, 2026-11-20 21:32-21:35 PST (above) | S3-13 withdrawn | Owner decision. Supersedes 16:45 for S3-13 only |
| `docs/TFM_SESSION_HANDOFF_20261114.md:76-80` (F-1) | The miner's phases 1-2 never see the trial's skip range | ALREADY RECORDED, open (miner side) |
| `docs/exchange/from_beta/X-PWC-STAGE2B_R1_BETA_REVIEW.md:47` (Finding 2), `:67` (Q8) | Your two-read finding and the proof-tool limits | ALREADY RECORDED. Addressed here |
| `docs/drafts/pwc_backup_stage2b_v1/README.md` "Beta round 1" | S3-11's coordinator side landed in 2b; SSH keeps 0-only | ALREADY RECORDED |

### Files under review (absolute paths; `git -C /home/michael/tfm_worktrees/pwc-offset diff b2efdef`)

| File | Change |
|---|---|
| `/home/michael/tfm_worktrees/pwc-offset/docs/drafts/pwc_backup_stage3a_v1/README.md` | the package README; read it first |
| `/home/michael/tfm_worktrees/pwc-offset/sieve_filter.py` | `execute_sieve_job`, new-key branch only: one dataset read |
| `/home/michael/tfm_worktrees/pwc-offset/scripts/prove_unchanged_outside.py`, `.../tests/test_prove_unchanged_outside.py` | four new checks, including `--keep-block` |
| `/home/michael/tfm_worktrees/pwc-offset/tests/test_pwc_stage3a.py` | new |

`persistent_worker_coordinator.py` is **not** changed (byte-identical to `b2efdef`).

Evidence is in `docs/drafts/pwc_backup_stage3a_v1/`:
- `prove_unchanged_outside_demo.txt`, `import_proof_*.txt`;
- `mutants_prove_unchanged_outside.txt`, `focused_*`;
- `S3-13_removed.diff`;
- `before_after_tally.md`, `before_after_failing_checks.txt` (final after-pass).

### What changed

- **Finding 2.** A new-key job derives the short-file refusal, the domain check and the window from one parsed,
  session-filtered list, as `miner/range_miner_worker.py:904-932` does. Old-style jobs (legacy `coordinator.py`, ZMQ)
  are unchanged.
- **Q8.** `prove_unchanged_outside.py` now refuses:
  - added names whose statement runs code at import;
  - added names already bound in the base (imports included) or rebound in the new file;
  - changed allowed headers that evaluate non-inert code. Narrowed by the owner on 2026-11-21:
    - parameters are compared by name with the base;
    - a non-inert default or annotation is flagged only when it is new or its AST differs from the base's;
    - any change to the decorator list flags every decorator;
    - reordering or re-kinding existing parameters is always caught (`allowed-header-parameter-order-changed`).

    The earlier rule flagged Stage 3b's unchanged `staging_high_water_bytes = 16 * 1024 ** 3` default in
    `window_optimizer.py` `run_bayesian_optimization` (`:756` → `:759`), where only a trailing parameter was added.

  New `--keep-block QUALNAME::HEADER` pins one block and its preamble inside an allowed function.
- **S3-11.** No new code. It landed in 2b; `G-S311-JOBS` re-checks the wire.

### Real-world scenarios for 3a

- **3a-S1. Rigs still on the `b2efdef` `sieve_filter.py` after VM101 is updated.** F2 lives in the rig copy, so the
  rigs keep the two-read form until a redeploy. On this cluster the dataset is not replaced during a run.
- **3a-S2. A dataset shorter than the window.** Still refused before any sieve, now with the miner's "only N entries"
  refusal instead of the out-of-domain code.
- **3a-S3. Stage 3b allows a large shared function**, e.g. `run_bidirectional_test`. `--keep-block` lets the proof
  pin the miner branch and everything before it in that function.
- **3a-S4. A PWC trial whose sampled skip range is not `[0, 16]`.** PWC sieves the sampled range on phases 1-2, as at
  `b2efdef`. RANGE-MINER sieves `[0, 16]` (F-1, open). So the engines still differ there, by the owner's choice,
  until the miner is fixed. Stage 4's comparison must pin `skip [0, 16]` or wait for the miner fix.

### Questions (allowed answers: NO OBJECTION · OBJECTION · SUGGESTION · NEEDS EVIDENCE; LIVE/THEORETICAL on every OBJECTION)

**3a-Q1. RANGE-MINER.** Both shared modules RANGE-MINER's VM101 launch imports
(`window_optimizer_integration_final.py:52-58`) are covered:
- `sieve_filter.py` changed only inside `execute_sieve_job`;
- `persistent_worker_coordinator.py` is byte-identical to `b2efdef`;
- the clean-process import proof is identical under both hostnames.

By your own check: does any line of 3a change a RANGE-MINER run?

**3a-Q2. Finding 2.**
- Is there any remaining path in which a PWC new-key job's checked domain and its window come from different reads?
- Is there one in which the window differs from `load_residue_window` on the same file?

**3a-Q3. Old-style jobs.** Do the legacy `coordinator.py` and ZMQ engines see any change from 3a? Our evidence:
`G-F2-LEGACY-UNCHANGED` gives identical sieve calls and results to `b2efdef` at offsets 0, 1 and 5, the last one
clamped.

**3a-Q4. The proof tool.** With the four new checks, what import-time or call-time change in a shared module would it
still not report?

**3a-Q4b. The narrowed header check.** It accepts a non-inert default that is AST-identical to the same-named
parameter's in the base, on the ground that the base already evaluated it at import.
- Is there a case where such a default evaluates differently in the new file? One example: a name it reads was rebound
  elsewhere in the module. Every module-level statement outside the allowed definitions is still compared.
- A removed parameter is reported but not failed. Should it fail?

**3a-Q5. `--keep-block`.** Is the preamble rule the right granularity for pinning a miner branch inside a function
Stage 3b changes? It covers the block itself, every statement before it in the function, and the signature and
decorators.

**3a-Q6. Test edits.** `tests/test_prove_unchanged_outside.py` changed one PASS case: an added decorator on an
allowed function is now a CATCH case. Does that weaken what the suite guarded?

### Test evidence (3a)

- **Focused, after the S3-13 removal:**
  - `tests/test_pwc_stage3a.py`: 6/6 checks, 3/3 mutants;
  - `tests/test_prove_unchanged_outside.py`: 49/49 after the 2026-11-21 narrowing, with 7 new cases. Caught: 5
    breakages of the Q8 checks and 6 of the narrowed check (H1 restores the old rule);
  - the narrowed tool on a read-only copy of Stage 3b's `window_optimizer.py`, allowing its four changed functions:
    UNCHANGED OUTSIDE ALLOWED, 0 header issues;
  - the narrowing is tool-only, so no full pass was re-run. Nothing but its own test and
    `scripts/pwc_sieve_filter_import_proof.py` references the tool, and that script's suite
    (`tests/test_pwc_stage2b_anchor_phase.py`) was re-run: 35/35;
  - every PWC suite exits as on `b2efdef`.
- **Full before (clean `b2efdef`):** valid, 75 suites.
- **After, final code** (S3-13 removed; `before_after_tally.md`, `before_after_failing_checks.txt`): 73 of 75 suites
  have the same exit code and the same failing-check set as on clean `b2efdef`. The two that differ:
  - `tests/test_s172_phase4_coordinator.py` Gate 22, the git-status check. It names exactly the four uncommitted
    `.py` files of this package, as in Stage 2b, and clears on commit;
  - `tests/test_s172_attempt6_remediation.py` FAIR-1, a wall-clock bound (0.820 s against 0.787 s). Re-run alone on
    the same code it PASSES with the "before" set (76/78). It is a miner-only suite on a file 3a does not change.

  `tests/test_pwc_stage3a.py` is new: rc 0. `~/miner_staging`, `optuna_studies/` and the main tree were unchanged.
  The earlier after-pass, which ran with S3-13, is superseded and kept as `*_with_S3-13.*`.
- **UNAVAILABLE:** GPU runs, rigs, and hit rate (Stage 4).
