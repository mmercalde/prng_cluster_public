# CLAUDE CODE INSTRUCTIONS — BINDING ADDENDUM AFTER BETA'S X-FINDINGS-PLAN R1 — 2026-11-13

**From:** Team Alpha · **Owner:** Michael · **Base commit:** `fbd8264`
**Answers:** `docs/exchange/from_beta/X-FINDINGS-PLAN_R1_BETA_REVIEW.md`

**This addendum is binding on these four documents. Where it and one of them disagree, the
addendum wins. Read it before the brief you were given.**

- `docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md` (§0, §2)
- `docs/CLAUDE_CODE_INSTRUCTIONS_RANGE_MINER_VM101_READS_20261113.md`
- `docs/CLAUDE_CODE_INSTRUCTIONS_CPU_GPU_GENERATOR_CONTRACT_STAGE_A_20261113.md`
- `docs/CLAUDE_CODE_INSTRUCTIONS_STEP6_CH13_GRADING_CONTRACT_STAGE_A_20261113.md`

**Why.** Beta raised two LIVE objections (Q4, Q5): all four documents tell a session to open
SQLite files with `mode=ro`. On a WAL-mode database in a writable directory that creates
`-wal` and `-shm` files beside the original, which is a write to live state. Alpha reproduced
it (SQLite 3.53.1: a closed, checkpointed WAL database gained both sidecars after one
`mode=ro` SELECT, and they stayed after close; with `immutable=1` nothing was created). The
repository already records this hazard and its remedy at
`tests/gate_s172_prod_shape.py:137-162` (`open_ledger`). Alpha's instruction was wrong.

---

## A. Reading SQLite files — replaces every `mode=ro` instruction

1. **Never open a live SQLite file with `mode=ro` alone, and never with a plain
   `sqlite3.connect(path)`.**
2. **Before opening,** without touching SQLite: confirm no coordinator, WATCHER or window
   optimizer process is running; `stat` the database and any `-wal`, `-shm` or `-journal`
   beside it; record names, sizes and mtimes in the report.
3. **`-wal` absent or zero bytes:** open with `file:<path>?mode=ro&immutable=1`.
4. **`-wal` non-empty:** the original is `UNAVAILABLE`. Do not open it. An immutable read
   would show stale data (Beta's fixture: old value through immutable, new value through the
   writer). If the content is needed, copy the database **and** its sidecars into scratch
   inside your worktree while no writer is running, show the source was stable across the
   copy (sizes and mtimes before and after), and query only the copy.
5. **Never** checkpoint, vacuum, delete, rename or repair a source file or sidecar.
6. **After reading,** `stat` the source directory again and show in the report that nothing
   beside the original was created or changed.
7. **Do not construct production database helpers against an original** to obtain a value.
   `CoverageLedger.__init__` runs `init_schema` on a writer connection
   (`utils/seed_coverage_ledger.py:704-720`), and the database wrapper builds that object
   lazily (`database_system.py:350`). Use plain SQL through an immutable connection or on a
   scratch copy. Pure functions may be imported and fed data you have already read.

## B. Isolation — additions to the no-touch list and the rules

- **Add to no-touch:** the root launch harness `gate12_launch.sh`. `scripts/gate12_*` does not
  name it.
- **Create no shared git ref.** No branch, no tag, no commit, in any worktree. A detached
  worktree is invisible to the relay snapshot; a new branch is not
  (`scripts/beta_relay.py:248-249`).
- **Gate 22 and the clean-tree gate are judged from the main tree only.** Do not widen any
  allowlist to admit worktree scratch.
- **The start-of-session process check is not a guarantee.** Repeat it immediately before any
  step heavier than reading files, and stop if a run has started.
- **Record of fact:** the G2/G3 session (`docs/CLAUDE_CODE_INSTRUCTIONS_DELEGATED_RUN_GATES_20261111.md`)
  builds new code; it is not a read-only session. It ran on 2026-11-13 in
  `/home/michael/tfm_worktrees/rm_gates`, wrote nothing in the main tree, wrote no launch
  record or baseline outside temp directories, and made one read-only ssh probe of `rrig6600`
  that returned `No route to host`.

## C. RANGE-MINER reads brief — additions

**R1, cursor.** Report per canonical PRNG base **and** per required mode set
(`{constant, variable}`): the qualifying record identities, the normalized union of certified
intervals, the status, and `next_seed_start` (`utils/seed_coverage_ledger.py:447-452`,
`:1049-1052`). `covered_seed_count` is not in general the first gap.

**R2, Run 4's range.** Trace the **effective** start after WATCHER, which can replace the
requested start with the cursor's `next_seed_start` or refuse because the domain is complete
(`agents/watcher_agent.py:2068-2088`). Also record **which command** Run 4 is: the root
harness's `--params` block carries no warm-start keys (`gate12_launch.sh:377-383`); the Run-4
Step-1 command with them is recorded at
`docs/SESSION_CHANGELOG_20260828_RUN4_ROUTING_PATCH.md:211-218`. Report the intended
operator command, its seven pinned values, the effective seed interval, the manifest and
configuration, and the dataset path, hash, session and window. Do not assume a Gate-12 launch
is the pinned Run-4 invocation.

**R3, counts.** Canonical records exist only for the forward/reverse intersection
(`utils/canonical_records.py:209-211`); one-direction populations are aggregate metadata.
Publication selects winners and merges the prior generation before counting rows
(`utils/run_finalizer.py:1613-1696`). Report separately, where available:
`raw_candidate_count`, `l2_winner_count`, `prior_row_count`, `final_row_count`, the parent
generation identity, and per-mode attempt totals. Do **not** call a cumulative row count
"attempt-9 yield", do not infer one-direction populations from row membership, and do not
report an unavailable total as zero. A comparison with any noise estimate is valid only on the
same count definition and settings.

**Add a closing section, "Not launch readiness".** This report does not establish GPU health,
rig parity, netconsole proof, the restricted agent account, the environment baseline, an
authorization, a volume limit, or a clean tree at launch time. List each as outside its
evidence.

## D. WP-1 Stage A brief — additions

1. **Q1 and Q4 must also trace the batch route and every fallback as separate production
   routes:** `extract_ml_features_batch` (`full_scoring_worker.py:375`); its vectorized
   generator called with `skip=0` (`survivor_scorer.py:570-575`); its CPU fallback
   `_cpu_batch_generate`, also `skip=0` (`survivor_scorer.py:577-578`, `:802-807`); and the
   worker's sequential fallback, which calls single-seed extraction
   (`full_scoring_worker.py:438-446`). An inventory limited to CPU-generator callers would
   miss the vectorized route.
2. **Q5 must separate stored features from fresh extraction in Step 6:** dictionary survivors
   carry stored features; integer survivors are re-extracted
   (`prediction_generator.py:781-790`). Say which route a saved-artifact comparison exercises
   and which artifacts would have to be regenerated under each option in Q8.
3. **Q8 must name exact compatibility controls for Stage B,** not only an agreement rate:
   expected outputs and indices for zero and non-zero gap, non-zero start and phase, missing
   skip metadata, the batch and fallback routes, and the old start-position caller. For each
   option list what is intended to change and what must stay byte-identical. CPU-only evidence
   is not GPU execution evidence; say so.
4. This brief needs no SQLite read. If you find one is needed, section A applies.

## E. WP-2 Stage A brief — additions

1. **Q8's contract must state both distinct `k` and one membership hit per target draw,** and
   name the numerator and denominator of the aggregate. Today the grading loop counts once per
   prediction (`chapter_13_diagnostics.py:242-249`) while breadth is de-duplicated (`:260-263`);
   Beta's fixture returned `exact_hits=2` for two copies of the winning value. Either preserve
   the existing fields' meaning or add a clearly named membership field. Do not silently
   reinterpret a diagnostic field. Q7 must list which consumers read `exact_hits`.
2. **Q4 must cover forecast eligibility, not only target identity:** forecast and
   configuration identity, data cutoff, and an outcome-independent rule for choosing among
   several artifacts that target the same draw.
3. **Q8's controls must include:** duplicate values; out-of-range or wrong-type values; a stale
   target; a future target with no outcome yet (pending, **not** a miss); repeated grading of
   one draw; missing and malformed artifacts; and a valid draw of `000`.

## F. For the later WP-3 instrument brief (no session acts on this now)

Validate integers in `[0, 999]`; bind each frozen pool to its target, cutoff and
configuration; reject duplicate target records or resolve them by a declared rule; record
abstentions and their denominator; specify the uncertainty calculation for varying `k`;
judge the uniform-random control against a declared tolerance, not exact equality.
