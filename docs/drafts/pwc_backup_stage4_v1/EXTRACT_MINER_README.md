# extract_miner_bundle.py: RANGE-MINER run → `engine_output_bundle_v1`

Stage 4 preparation. Code only, CPU only. Worktree `/home/michael/tfm_worktrees/pwc-offset` at `6d75f71`; engine
anchors are at that commit. New files only:
- `scripts/extract_miner_bundle.py`
- `tests/test_extract_miner_bundle.py`
- this README

**Revisions:**
- **r1:** the first extractor.
- **r2:** aligned to agent 1's comparator `454eb7b3…`: the null rule and `build_inputs()`.
- **r3 (this one, 2026-11-21 afternoon):** answers Beta's X-PWC-STAGE4 R1 Q6, Q3 and Q9, and aligns to agent 1's
  Q2 comparator `040f77b6…` (12:47, read read-only at `~/tfm_worktrees/pwc-backup/scripts/compare_engine_outputs.py`).

**This extractor needs agent 1's Stage 4 comparator.** It imports `build_inputs()`, and it reads
`F2_REQUIRED_INPUTS` when present. Against the `6d75f71` comparator it refuses with exit 2.

## Prior record

- **Searches (r1, unchanged):**
  - `/bin/grep -rlI -i "engine_output_bundle\|extract_miner_bundle\|extractor per engine" docs/ scripts/ tests/`;
  - `/bin/grep -lI -i "per-phase survivor\|NOT COMPARED"` over `docs/PROJECT_FILE_CATALOG.md`, `TODO_CONSOLIDATED.md`,
    `BACKLOG.md`, `LEADS.md`, `docs/TB_RULING_*` and `docs/PROPOSAL_*`;
  - `ls docs/exchange/{to,from}_beta | grep -i "stage4\|extract\|compare"`.
- **Added for r3:**
  - `grep -n "set_user_attr" window_optimizer*.py`;
  - `grep -n "step1_sweep_evidence"` over `window_optimizer*.py`, `agents/watcher_agent.py` and `utils/*.py`;
  - prints of the miner run id in `window_optimizer_integration_final.py`.
  These were the search for an existing record binding a miner trial to its generation; see Q6.
- **Documents found:**
  - `docs/drafts/pwc_backup_stage3_v1/PLAN.md` §F, the design;
  - `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:47`;
  - agent 1's comparator docstring: null rule, agreed sources, exit codes, `F2_REQUIRED_INPUTS`;
  - Beta's `X-PWC-STAGE4_R1_BETA_REVIEW.md`, read read-only in agent 1's tree;
  - agent 1's `SESSION_RUNBOOK.md:126-145` (the pins and the two params JSONs), read for the params shape only.
- **Labels:**
  - the receipt and binding checks are **NEW**;
  - per-phase sets not kept and F-1 are **ALREADY RECORDED** (r1/r2; `docs/drafts/SKIP_DESIGN_RECORD_20261121.md`);
  - nothing **CHANGES A RULING**.
- **Not searched:** the host, the ser8 archives, the private repository, the rigs, `~/miner_staging`. The rest of
  agent 1's runbook was not read.

## Use: two steps

```
# step 1 — right after the RANGE-MINER Step 1 exits, BEFORE the PWC arm starts
python3 -u scripts/extract_miner_bundle.py --staging-dir S --output-root T --params-json P \
    --write-receipt /home/michael/distributed_prng_analysis/data/stage4/miner_receipt.json

# step 2 — any time later
python3 -u scripts/extract_miner_bundle.py --staging-dir S --output-root T --params-json P \
    --out-dir /home/michael/distributed_prng_analysis/data/stage4/bundle_miner \
    --receipt /home/michael/distributed_prng_analysis/data/stage4/miner_receipt.json
    [--run-id <miner trial>] [--ledger-db <run ledger; default T/prng_analysis.db>]
```

- **S** is the coordinator `staging_dir`; **T** is the Step 1 CWD, the `finalize_run` output root; **P** is the
  WATCHER `--params` JSON.
- **Exit codes:** 0 means written; 2 means refused, with the reason on stderr and nothing written.
- **Both steps run the same reads and the same binding checks B1–B8.**
- **Step 2 also refuses** if any identity bound in the receipt has changed.
- **Neither step writes into S or T.** The receipt and the out-dir must lie outside both, and a receipt is never
  overwritten.
- **Ledger reads:** an immutable read, or a byte copy when the `-wal` is non-empty, as before.

## Beta X-PWC-STAGE4 R1: what changed, point by point

### Q6 (LIVE): the mixed-source pair. Beta is right.

**What Beta found:**
- r2 bound the miner ledger to the publication only by the NPZ trial ordinal (r2 `:376-379`).
- The swept-vs-generation interval was only a warning (r2 `:382-384`).
- So a miner ledger paired with another run's publication, both trial 0, gave a miner-labelled bundle.

**What the run records:** nothing durably ties a miner trial to the generation it fed.
- The miner `run_id` reaches Step 1 only in `step1_sweep_evidence`, which is held in memory
  (`window_optimizer_integration_final.py:1769-1773`, `:2022`).
- `TestResult.to_dict` carries no run id (`window_optimizer.py:274-284`).
- The sidecar `run_id` is `step1_<prng>_<seed_start>` (`window_optimizer_integration_final.py:3579`), which is the
  same for both arms.
- No print names the miner run id on the success path.

So the binding has to be **captured**. That is the receipt, plus checks that make a mixed pair fail even when the
receipt is written for the wrong pair.

**The receipt** (`stage4_miner_run_receipt_v1`) records:
- the real paths of S, T, the run ledger and P;
- the sha256 of P;
- the miner `run_id`, `trial_number`, `created_at` and `finalized_at`;
- a sha256 over the trial's ledger rows (trial, `trial_context`, stripes, shards). This is content-based, so a
  later WAL checkpoint does not change it;
- the `generation_id`, generation dir, artifact and sidecar sha256;
- the `coverage_id` and the link;
- the F2 job identity.

**The binding checks** (each is a refusal):

| Check | What it binds | Anchor |
|---|---|---|
| **B1** | params `use_range_miner: true`, and params `staging_dir` = `--staging-dir`. The runbook's miner params carry it (`SESSION_RUNBOOK.md:137`) | — |
| **B2** | the params pins `warm_start_window/_offset/_skip_min/_skip_max/_fwd_thresh/_rev_thresh` and `prng_type` = the trial's `trial_context`; `test_both_modes` = the generation's executed modes. `warm_start_session_idx` is not bound, because the run records the session labels, not the index or the option list | `coordinator:1384-1399` |
| **B3** | every NPZ row's `trial_number`, `window_size`, `offset` (= window anchor), `skip_min`, `skip_max` = this trial's | `utils/canonical_records.py:215-228` |
| **B4** | every NPZ row's `forward_count`/`reverse_count` = this trial's per-phase `survivor_rows`, for that row's mode | `canonical_records.py:234-235` |
| **B5** | every phase's swept interval = the generation's `[seed_start, seed_end_exclusive)`. This was a warning in r2 | — |
| **B6** | `0 ≤ sidecar created_at − trial finalized_at ≤ trial duration`. Both clocks are the same process: `run_finalizer.py:1573`, `coordinator:1990-1992`. A generation published by a later run of the same job lags by at least that run's execution | — |
| **B7, B8** | see Q3 | — |

**Result on Beta's case** (gate G-MIXED-SOURCE):
- **Setup:** A and B run the same job, both trial 0. B differs by one seed in phase 4, so its variable
  bidirectional set loses seed 1250.
- **A's ledger with B's publication, against A's receipt:** refused. The receipt's `generation_id`,
  `artifact_sha256` and the other bound fields differ.
- **A receipt written for the mixed pair:** refused by B6, once A finished well before B published.
- **The correctly paired arms:** exit 1, differing only in the NPZ.

**The limit, stated plainly:**
- If both arms publish within one trial duration of the miner commit, B6 cannot separate them. That would take
  concurrent runs, which the runbook forbids.
- In that case, if the engines also agree on every count and context value, B1–B5 and B7–B8 pass too.
- Then only a receipt **written at the right time** binds the pair. **The runbook must run step 1 between the
  miner arm and the PWC arm.** That is agent 1's file, so I have not edited it.

**B6 risk [UNVERIFIED on a real run]:** the publication lag of a real Stage 4 trial is not measured. A very short
trial, such as PLAN F4's zero-survivor control, could finish faster than Step 1's wrap-up. B6 would then refuse
a genuine pair. It fails closed, and the refusal names both numbers. If it fires, the lead should decide whether
B6 needs a different bound.

### Q3 (LIVE): coverage and link cross-identifiers. Beta is right.

**What Beta found:** r2 selected `generation_id` from the coverage row and `coverage_id` from the link row, then
never compared them (r2 `:408-431`). r3 checks both:
- **B7:** the `certified_coverage` row for this artifact must carry **this** `generation_id`. It must also carry the
  generation's interval and the trial's `dataset_sha256`. In r2 the interval and dataset checks were warnings.
- **B8:** a `step1_chunk_link` row for this generation must name **that** `coverage_id` and the same interval. A
  link with no coverage row is refused.

**Alignment to agent 1's definition:** `credit.granted` is now true only when both rows exist and are bound (the
comparator docstring's `credit` line). A coverage row without a link gives `granted: false`. In r2 that gave
`true`. Gate G-COVERAGE-CROSS-ID forges each mismatch: another generation, another interval, another dataset,
another coverage id, another link interval, a link with no coverage.

### Q9: regressions. Beta is right; the matrix lacked them.

New gates:
- G-MIXED-SOURCE;
- G-COVERAGE-CROSS-ID;
- G-NON-DONE: a committed trial with a stripe not `done` (r2 branch `:243-244`, untested);
- G-PHASE-SET (r2 `:237-239`, untested), with phases {1..5} and {1,2} against a both-modes generation;
- G-TRIAL-ORDINAL (r2 `:376-379`, untested);
- G-PINS (B2), G-NPZ-CONTEXT (B3), G-NPZ-COUNTS (B4), G-SWEPT-INTERVAL (B5), G-LAG (B6), G-RECEIPT;
- G-PARAMS, extended with missing and foreign `staging_dir` (B1).

Each new check has a mutant.

### Q2: alignment to agent 1's job-identity rule (comparator `040f77b6…`)

- **The F2 job identity in the bundle** is `inputs`, via `build_inputs()`. `F2_REQUIRED_INPUTS` names 15 keys there:
  `run.*` from the sidecar and the coverage row, and `param.*` from the params pins.
- **The extractor flags missing keys:** if any is missing or null, it adds `WARNING F2 job-identity inputs missing
  or null [...]` to `provenance.self_checks`, because the comparator then gives exit 2.
- **B2 binds those `param.*` pins to the trial's own record,** so the F2 identity the miner bundle presents is
  what the miner ran, not only what was requested.
- **`provenance.job_identity`** also carries the F2 rows read from the miner's own records. It is ignored by the
  comparator; it is there for the reader:
  - seed range, window, anchor, sessions, dataset;
  - generator phase, thresholds;
  - the constant skip range as requested, and as the code says it ran (F-1);
  - phases, family;
  - strategies and transport, which are null because they are not recorded or not applicable.
- **The comparator's Stage 4 credit requirement** (granted, and certified = linked = swept = run interval) is met by
  a correctly credited run. The gates show exit 1 or exit 3 with credit required, not exit 2.

### Not mine to change

- **Q5** (the PWC extractor's WAL handling) cites this file's copy-then-read as the model
  (`extract_miner_bundle.py:121-167` at r2); it is unchanged.
- **Q1, Q4, Q7 and Q8** concern the comparator, the PWC extractor, the runbook and the launcher.

## Where each piece comes from

| Bundle piece | Source | Anchor |
|---|---|---|
| `inputs` | `build_inputs(params, provenance.json, certified_coverage.dataset_sha256)`. The dataset comes only from a B7-bound row | comparator (agent 1) `build_inputs`; `seed_coverage_ledger.py:646-667` |
| `phases.N`, `bidirectional.*` | **null** (NOT COMPARED). The seed→rate maps live only in the in-memory assembly; the spools are deleted at commit; the NPZ keeps one row per seed | `range_miner_npz_writer.py:223-245`, `:1296-1302`; `coordinator:8340-8342`; `run_finalizer.py:714-749` |
| `phase_counts.N` | `survivor_rows` = the sum of `stripes.survivors_total`, checked against the current attempt's shards (a NULL or mismatched count is null, never 0); `distinct_survivors` is null | `coordinator:1128-1170`; `range_miner_worker.py:2479` |
| `credit` | `granted` (B7 and B8 both bound), `certified_interval`, `linked_interval`, and `swept_intervals` from stripes (all `done`, contiguous, = the generation's interval) | `window_optimizer_integration_final.py:1614-1628`; `step1_search_ledger.py:306` |
| `npz` | the parent-free generation's `bidirectional_survivors_binary.npz`, sha256 checked against the sidecar, copied byte for byte. A wrong array order is copied as is and flagged; the PWC extractor refuses it instead (`extract_pwc_bundle.py:142-143`) | `run_finalizer.py:124-131`, `:752` |
| allowed to differ | the miner `run_id`, `finished_at`, `duration_s`, `worker_ids`, `job_ids`, `geometry` (engine keys plus stripes and plan), `logs`, `provenance` (receipt and sha256, bindings, `job_identity`, the F-1 code reading, `not_available`, `self_checks`) | — |

**F-1 is unchanged.**
- The worker default `[0,16]` (`range_miner_worker.py:1293`) and a payload builder with no `skip_range`
  (`coordinator:9216`) give `WARNING F-1` when the trial requested another range.
- The agreed inputs carry the **pins**, so two arms pinned to `[0,16]` agree on it.

## Tests: `tests/test_extract_miner_bundle.py` (only this suite was run)

**Run against a copy of agent 1's comparator placed in this tree for the test only:**
1. `scripts/compare_engine_outputs.py` (HEAD `434fa3ba…`, clean) was overwritten with agent 1's `040f77b6…`.
2. The suite printed and checked that sha256 (G-COMPARATOR).
3. The file was restored with `git checkout --`; it is `434fa3ba…` again.

**Results:**
- **Run 7: 31/31 gates PASS; 26/26 mutants KILLED, each by its own check.** Output:
  `logs/test_extract_miner_bundle_run7.txt` (ignored).
- **Runs 5 and 6** exposed fixture errors, not extractor errors:
  - the forged coverage row needs `'CERTIFIED'`;
  - the wrong-order gate needed its rewritten artifact certified, because `run.dataset_sha256` is a required F2 input;
  - three mutants (MU3, MU5, MU26) were first killed by a different check, so their gates were narrowed until only
    the intended check could see the fault.
  Run 5 also showed agent 1's comparator had moved from `454eb7b3…` to `040f77b6…` mid-task.

**How the synthetic runs are built:** with the real writers (`MinerLedger`, `finalize_run` over
`build_mode_records`, `CoverageLedger.record_publication`, `step1_search_ledger._DDL`). Forged coverage and link
rows are plain INSERTs into the real schemas.

**Gates:**
- **Clean and comparison:** G-COMPARATOR, G-CLEAN, G-NULL-NEVER-EQUAL (exit 3, and nulls still NOT COMPARED
  against a bundle that has the lists).
- **Beta's three points:** G-MIXED-SOURCE, G-COVERAGE-CROSS-ID, G-RECEIPT.
- **Credit and inputs:** G-F1-WARNING, G-CREDIT, G-PARAMS, G-PINS, G-LIVE-PATH.
- **Missing or bad records:** G-MISSING-GENERATION, G-NULL-COUNT, G-PHASE-GAP (an interior gap), G-NON-DONE,
  G-PHASE-SET, G-TRIAL-ORDINAL, G-NPZ-CONTEXT, G-NPZ-COUNTS, G-SWEPT-INTERVAL, G-LAG, G-UNCOMMITTED.
- **The NPZ:** G-TRUNCATED-NPZ, G-MODIFIED-NPZ, G-WRONG-ORDER.
- **Ledger edge cases:** G-RETRY, G-WAL, G-PRIOR-GENERATION, G-MULTI-TRIAL.
- **Output and CLI:** G-OUT-DIR, G-CLI (receipt exit 0; bundle without receipt exit 2; bundle exit 0; no
  overwrite).

**Mutants:**
- **MU1–MU14 (from r2):** empty lists for nulls, sha check, re-save, contiguity, attempts, parent, WAL,
  uncommitted, NULL as 0, a claimed distinct count, the dataset source, the params engine check, live paths.
- **MU15–MU26 (new):**
  - B1, B2, B3 trial ordinal, B3 context, B4, B5, B6, B7 generation id, B8 link;
  - the receipt comparison;
  - non-done stripes;
  - the phase set.

## Not verified

- **No real run was extracted.** Layouts are from the code at `6d75f71` and the real writers driven by the test.
- **B6's real publication lag** is not measured (see the Q6 risk).
- **Agent 1's comparator may still change;** these results hold for `040f77b6…`.
- **The PWC extractor (`extract_pwc_bundle.py`, now 12:50) was not run against this one.**
