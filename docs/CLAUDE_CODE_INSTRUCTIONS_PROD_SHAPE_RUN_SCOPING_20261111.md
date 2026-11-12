# CLAUDE CODE INSTRUCTIONS — MAKE `G-PROD-SHAPE` RUN-SCOPED — 2026-11-11

**From:** Team Alpha · **Owner:** Michael · **Run on:** VM101, as `michael`, in
`~/distributed_prng_analysis`, venv `source ~/venvs/torch/bin/activate` · **Model:** Opus
**Base commit:** `b6cb789`. Anchors were read at that commit by Alpha.
`tests/gate_s172_prod_shape.py` and `miner/` are unchanged since `a15b524`.
**Review class: BEFORE-COMMIT.** This is the verifier for runs that count, and §3 changes what
one of its legs asserts. Michael does not commit until Beta has reviewed or Michael has decided.

**Why.** Finished condition F1 in `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md`: the run verifier
must be able to pass on a clean run. Today it cannot pass for any run on this host
(`docs/S172_WINDOW_ANCHOR_BRIEF_I_CLOSURE_STEP23_REPORT.md` §4.1): sections B and C read the
whole shared ledger and the whole shared staging directory, so every run inherits the rows of
seven aborted runs and 512 orphan files from a run with no ledger trace.

---

## 0. Rules for this session

- **You do not commit or push.** Stop at the gate in §6 and report.
- **You may write only these files:**
  - `tests/gate_s172_prod_shape.py`
  - `tests/test_gate_prod_shape_scoping.py` (new)
  - `docs/CLAUDE_CODE_REPORT_PROD_SHAPE_RUN_SCOPING.md` (new)
- **Nothing under `miner/` is edited.** The coordinator is read, never changed.
- **The live ledger and staging directory are opened read-only.** No row is updated, no file is
  moved or deleted. The 512 orphan files stay exactly where they are.
- **Use the Edit tool only.**
- Other Claude Code sessions are editing `docs/CHAPTER_*.md`, `agents/watcher_agent.py` and
  `agents/watcher_dispatch.py`. Leave those alone.
- **No fleet and no pipeline launch.**
- Logs go under `data/prod_shape_scoping/` (untracked, not committed).
- Investigate first (§1). If §1 contradicts §2 or §3, stop and report before editing.

## 1. Investigate (read-only; answers go in the report)

1. **Where the live ledger and staging directory are.** The fact file names
   `/home/michael/miner_staging/miner_ledger.db` and warns that the same-named file in the
   project root is stale. Confirm against `agent_manifests/window_optimizer.json`
   `default_params.staging_dir`.
2. **The trials in the ledger:** `run_id`, `state`, and shard count per run. Identify attempt 9
   (`distributed_config_t1_554463d3`) and confirm its state.
3. **Staged file naming.** The coordinator builds names as `{safe_run}__{stripe_id}_a…`
   (`miner/range_miner_coordinator.py:4373`). State how `safe_run` is derived from `run_id` and
   confirm that the 512 orphans all carry `distributed_config_t1_25e4f207`.
4. **Which of the other leak patterns carry a run id** in their names: `*.tmp`, the
   `.s172_staging_probe*` files and provisional manifests (verifier `:209-229`).
5. **Remote versus inline delivery.** Alpha's reading of the source, to be confirmed or refuted:
   - A worker spools a result only when the payload reaches `INLINE_BYTE_LIMIT = 48 MiB`
     (`miner/range_miner_worker.py:1344`, `:2475-2489`). Otherwise it is sent inline.
   - The coordinator enqueues `kind="inline"` or `kind="remote"` accordingly (`:11845`, `:11859`).
   - `_delete_remote` runs only for `kind == "remote"` (`:4527`), and is the only path that
     moves `remote_delete_status` off `'none'`.
   - So for an inline shard, `'none'` is the correct final value: no remote file ever existed.
   Measure on the live ledger, read-only:
   - shards with `remote_spool_path IS NOT NULL`, in total and for attempt 9;
   - `MAX(size_bytes)` over all shards.
   State whether `remote_spool_path` is NULL for every inline shard. If any shard has a
   non-NULL path, list its `remote_delete_status`.
6. **Every reference to `gate_s172_prod_shape`** in `scripts/launch_fleet_manual.sh`,
   `tests/test_s172_staging_partb.py`, `tests/test_s172_staging_backpressure.py` and
   `miner/range_miner_coordinator.py`. For each: is it a name in a comment, or does something
   execute or parse the verifier?
7. **Baseline before any edit.** Save the full output of:
   ```
   python3 -u tests/test_s172_staging_partb.py        | tee data/prod_shape_scoping/base_partb.log
   python3 -u tests/test_s172_staging_backpressure.py | tee data/prod_shape_scoping/base_bp.log
   ```
   and of the unmodified verifier against the live ledger, using the newest log you can find
   for attempt 9 as `--log`. If no such log exists, say so and record section A as UNAVAILABLE.

## 2. Change 1 — scope sections B and C to one run

**Required behaviour.**
- A new argument `--run-id`. Sections B and C judge **that run only**.
- Without `--run-id` the verifier prints the trials it found and ends with an overall verdict
  of **UNAVAILABLE**, never PASS. A verdict about "the ledger" is no longer offered.
- A `--run-id` that is not in `trials` is FAIL.
- **Section B.** Every query is restricted to the run: `trials`, `stripes`, `shards` and
  `reservations` all carry `run_id`. B1 becomes "this trial is `committed`". The existing
  `len(shards) > 0` guards stay, so a run with no shards cannot pass vacuously.
- **Section C.** A leaked file counts against the run only if its name carries the run's id
  (per §1.3 and §1.4). Patterns that carry no run id keep their current whole-directory check.
- **Residue from other runs is reported, not hidden.** One extra line per section, for example
  `B-OTHER  23389 shards from 8 other runs not judged` and
  `C-OTHER  512 staged files from 1 other run not judged`. These lines are informational and do
  not change the verdict. They are printed every time, including when the count is zero.
- Section A (anti-fabrication) and section D (artifact) are not changed.

## 3. Change 2 — B7 judges only shards that had a remote file

**This changes an assertion, so it is conditional on §1.5.** Make it only if §1.5 confirms that
`remote_spool_path` is NULL for every inline shard and that `_delete_remote` is the sole writer
of a non-`'none'` status. If §1.5 finds otherwise, stop and report; do not edit B7.

**Required behaviour.**
- B7 passes when every shard of the run **with a non-NULL `remote_spool_path`** has
  `remote_delete_status != 'none'`.
- Shards delivered inline are exempt.
- The evidence string always states both counts, for example
  `0 remote-spooled, 5693 inline; remote_delete_status values: ['none']`.
- A run with zero shards still fails, as now.

**Not changed:** the coordinator. Whether a 48 MiB threshold is the right spool boundary is not
this brief's question.

## 4. Gates (in the new test file)

All fixtures are built in a temp directory: a small SQLite ledger with the four tables, and a
staging directory. Table definitions are copied from the coordinator's DDL (`:1248-1330`) by
reading the source at test time, not retyped.

| Gate | Fixture | Must hold |
|---|---|---|
| S-RED | Run T (committed, all shards acked and cleaned, inline) plus run X (aborted, shards `enqueued`) plus 3 staged files named for run X. **Pre-change verifier** from `git show b6cb789:tests/gate_s172_prod_shape.py` | B5, B6 and C2 FAIL. Reproduces the fault |
| S-A | Same fixture, live verifier, `--run-id T` | B1 to B8 and C1 to C4 PASS; `B-OTHER` and `C-OTHER` lines present with the right counts |
| S-B | Same, plus one staged file named for run T | C2 FAIL |
| S-C | Same as S-A but one of T's shards has `phase5_status='enqueued'` | B5 FAIL |
| S-D | Same as S-A but T's trial state is `aborted` | B1 FAIL |
| S-E | `--run-id` naming a run not in `trials` | FAIL |
| S-F | No `--run-id` | Overall verdict UNAVAILABLE; exit status is not the PASS status |
| S-G | T has a reservation with `status='held'`; X has none | B8 FAIL |
| S-H | X has a held reservation; T has none | B8 PASS for T |
| B7-A | T: all shards inline (`remote_spool_path` NULL), status `'none'` | B7 PASS, evidence shows `0 remote-spooled` |
| B7-B | T: one shard with a remote path and status `'none'` | B7 FAIL |
| B7-C | T: one shard with a remote path and status `'deleted'`, the rest inline | B7 PASS |
| B7-D | T: zero shards | B7 FAIL |
| MUT-1 | Remove the `run_id` restriction from the shards query | S-A red |
| MUT-2 | Remove the run-id filter from C2 | S-A red |
| MUT-3 | Restore the old B7 condition | B7-A red |
| MUT-4 | Make the no-`--run-id` path return PASS | S-F red |

Then one check against real data, read-only, reported but not a gate:

- **LIVE-1.** The live verifier against the live ledger with
  `--run-id distributed_config_t1_554463d3`. Report every leg. Expected from the STEP23 report:
  B5 and B6 at 5693/5693. Report B7's two counts and section C's `C-OTHER` line. Section D's
  result is reported as found.

## 5. Verification block

- **Execution proof:** each gate prints its name and verdict; the suite prints
  `COMPLETION SENTINEL: PASS` only when every gate ran and passed.
- **Clean controls:** S-A, S-H, B7-A, B7-C.
- **Fault-injection controls:** S-RED, S-B, S-C, S-D, S-G, B7-B and the four mutants. Mutants
  are applied to a copy of live source. A mutant that fails to apply is an error, not a
  detection.
- **Unavailable-observer behaviour:** an unreadable ledger or staging directory is UNAVAILABLE,
  never PASS and never a zero count.
- **Regression:** re-run the two §1.7 suites after the edit and diff against the baseline.
- **Scope proof:** `git status --porcelain` shows only the three files in §0, `data/`, and
  files owned by the other sessions. State the sha256 of the live ledger before and after
  LIVE-1 to show it was not written.

## 6. Stop gate and report

Stop without committing. Write `docs/CLAUDE_CODE_REPORT_PROD_SHAPE_RUN_SCOPING.md` with:

1. The §1 answers with `file:line` anchors and the measured counts.
2. A plain statement on §1.5: is `'none'` the correct final value for an inline shard, yes or
   no, and the evidence.
3. The diff, the gate table with verdicts, and the mutant evidence.
4. LIVE-1's full output.
5. The before/after regression table.
6. Anything in this brief the source or the ledger contradicted.
7. Open items found and not fixed. Expected at least: what should happen to the 512 orphan
   files; whether any remote spool files exist on the rigs (cannot be checked with the fleet
   off).
8. The exact `git add` line for Michael, built from `git status`.
