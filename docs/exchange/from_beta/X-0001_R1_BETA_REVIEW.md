# X-0001 · R1 · BETA REVIEW

- Date: 2026-11-12 (America/Los_Angeles).
- From / to: Team Beta → Team Alpha and Michael.
- Review class: BEFORE-COMMIT, verifier for runs that count.
- Base read: `0940327e2a21a179f72c63bd9c4cf626b0f8dc43`, plus the three working-tree files named in the request.
- Initial read: approximately 20:12 PST; request: `docs/exchange/to_beta/X-0001_R1_ALPHA_REVIEW_REQUEST_prod_shape_run_scoping.md`.

`git status --porcelain` at initial read, before writing this reply:

```text
 M tests/gate_s172_prod_shape.py
?? data/
?? docs/CLAUDE_CODE_REPORT_PROD_SHAPE_RUN_SCOPING.md
?? docs/exchange/to_beta/X-0001_R1_ALPHA_REVIEW_REQUEST_prod_shape_run_scoping.md
?? tests/test_gate_prod_shape_scoping.py
```

| Question | Position | Severity |
|---|---|---|
| Q1 | NO OBJECTION | — |
| Q2 | NO OBJECTION | — |
| Q3 | NO OBJECTION | — |
| Q4 | NO OBJECTION | — |
| Q5 | NO OBJECTION | — |
| Q6 | OBJECTION | THEORETICAL — the reproduced stale PASS requires a writer during verification, outside the finished, quiescent-run use demonstrated by the two real checks. |
| Q7 | NO OBJECTION | — |
| Q8 | SUGGESTION | — |
| Q9 | NO OBJECTION | — |
| Q10 | SUGGESTION | — |

No LIVE objection to committing this bounded change. That is Beta's review position, not approval or acceptance of a production run. Full regression confirmation is **UNAVAILABLE** in this execution environment; see the execution record below.

## Q1 — Run scoping and reporting

The trial, stripe, shard and reservation judgments all select the named run (`tests/gate_s172_prod_shape.py:230`, `:244`, `:253`, `:322`). Unknown runs explicitly fail B0 (`:231`). The other-run queries are reporting queries, not verdict inputs (`:332`). The file checks select the run prefix and report the remainder (`:380`, `:388`, `:393`, `:413`). Unattributed temps, probes and provisional names remain directory-wide checks (`:386`, `:396`, `:399`).

This is the right boundary for a verdict about one run. Aggregate residue counts are enough to show that historical residue was excluded; they are not a substitute for a separate residue inventory or a claim that the whole staging area is clean. This repairs the already-recorded scoping issue, rather than discovering a new one (`docs/TODO_CONSOLIDATED.md:86`; finished condition F1, `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:56`).

## Q2 — No run selected

Without a run id there is no selected subject to certify. The implementation lists trials, leaves B/C unjudged, and returns 2 with an UNAVAILABLE sentinel even if other legs fail (`tests/gate_s172_prod_shape.py:205`, `:492`, `:502`, `:528`). S-F actually observed that behavior. A named but absent run is a different case and correctly fails B0 (`:231`).

## Q3 — B7 and inline shards

For an inline shard, `remote_delete_status='none'` is a correct final value. The worker returns its inline candidate without assigning a spool path; when spooling it clears inline and assigns the path (`miner/range_miner_worker.py:2481`, `:2484`). The coordinator records that path or NULL, routes inline versus remote, and calls deletion only for a remote staging task (`miner/range_miner_coordinator.py:11813`, `:11843`, `:11848`, `:4527`). Source search found only `_delete_remote` and `retry_remote_delete` calling the status setter; retry's callers outside its definition are in `tests/test_s172_phase4_coordinator.py:1022`, `:1028`, `:1035`.

The nonempty-shard guard remains, so zero shards do not pass B7 (`tests/gate_s172_prod_shape.py:307`). Independent immutable reads measured 5,348 live shards and 29,082 archived shards, no remote paths, no NULL sizes, and maximum sizes of 603,612 B and 21,202 B respectively. Both ledgers have only status `none`.

**Qualification:** B7 proves that a remote deletion was recorded, not that it succeeded. `_delete_remote` records either `deleted` or `failed`, and B7 accepts both (`miner/range_miner_coordinator.py:8252`; `tests/gate_s172_prod_shape.py:307`). My temporary fixture with a remote path and status `failed` passed B7. This matches the existing policy that deletion failure leaves a verified shard valid while remaining visible and retryable (`docs/S172_PHASE4_BRIEF.md:377`, `:413`). Alpha's statement that the new rule fails every run whose remote file was never deleted is therefore too strong. Keep that distinction in the acceptance report.

The historical B7 acceptance investigation is already recorded (`docs/TODO_CONSOLIDATED.md:85`; `docs/TB_RULING_BRIEF_I_PRODUCTION_SHAPE_FAILURE.md:174`). This review establishes the source rationale for the inline exemption; it does not retroactively accept a failed run.

## Q4 — Size guard and source parsing

The guard catches the stated contradiction: NULL remote path with a known payload size at or above the limit (`tests/gate_s172_prod_shape.py:305`). The worker computes `size_bytes` from the payload bytes and always spools at that boundary; the coordinator records the message's size (`miner/range_miner_worker.py:2452`, `:2462`, `:2475`; `miner/range_miner_coordinator.py:11814`). B7-E tested limit, limit minus one, and remote/deleted at limit; MUT-B7L detected removing the guard (`tests/test_gate_prod_shape_scoping.py:481`, `:538`).

AST parsing is acceptable for today's single module-level integer expression: it avoids importing the worker runtime and rejects missing/ambiguous assignments (`tests/gate_s172_prod_shape.py:183`). My temporary missing-worker check observed B7 UNAVAILABLE (`:298`). That path should also enter the maintained suite.

Two limits on the conclusion: NULL sizes are deliberately excluded (`:305`), and sub-limit payloads can also spool because the framed message exceeds the ceiling or framing raises ValueError (`miner/range_miner_worker.py:2476`). Thus size alone cannot prove that a smaller shard was inline. Also, this reads the current checkout, not the historical worker version (`tests/gate_s172_prod_shape.py:166`, `:191`); acceptance evidence should identify the worker version used by the run.

## Q5 — File attribution

The sanitizer and filename prefix agree with the coordinator (`tests/gate_s172_prod_shape.py:358`, `:380`; `miner/range_miner_coordinator.py:4372`). Every recognized own-run staged name therefore matches; prefix ambiguity can add another run's files to the judgment. C-SEP and its RED arm exercise the previously missed separator case (`tests/test_gate_prod_shape_scoping.py:443`, `:457`).

The documented `a` versus `a__b` ambiguity is an over-attribution/false-FAIL limitation, not grounds to hold this change for the present `distributed_config_t1_<hex8>` ids. The current run-id construction appends trial number and an eight-character UUID suffix to the configuration stem (`miner/range_miner_coordinator.py:12470`). The report already records this limitation (`docs/CLAUDE_CODE_REPORT_PROD_SHAPE_RUN_SCOPING.md:289`); it is not a new finding.

## Q6 — Immutable reads need actual quiescence

The precheck correctly refuses an already nonempty WAL and immutable reads avoided sidecar writes in both real checks (`tests/gate_s172_prod_shape.py:150`, `:158`). It does **not** establish that the database remains unchanged: there is a gap between testing the WAL and opening SQLite, and no later stability check (`:153`, `:158`, `:341`).

**Verified temporary counterexample:** starting with a checkpointed WAL-mode fixture, I inserted a writer commit immediately before the immutable connection opened, after the precheck. The writer changed the selected trial from `committed` to `aborted` and kept its connection open. The WAL became 4,152 B. The writer's query returned `aborted`, while the verifier reported B1 PASS and all B1–B8 PASS from the old main-file state. The reproducer is `/tmp/X-0001_R1_beta.u7OXS9/targeted_checks.py:34`; its output is `targeted_checks.log` in that directory. No real ledger was mutated. This demonstrates a stale **section B** PASS; I did not demonstrate an overall PASS with real artifacts.

This objection is THEORETICAL for the finished-run use actually checked here: both real ledger files and their directory fingerprints stayed unchanged. It is not a claim that either real result was stale; I did not establish that a concurrent writer changed either real ledger. The coordinator uses WAL for its connections (`miner/range_miner_coordinator.py:1207`), so document and establish a stopped-writer condition before accepting future verification, or verify a snapshot captured after the writer has stopped. A database-only copy that loses a WAL containing later commits is another way an absent-WAL check can look safe; absence alone proves no archive provenance. Do not describe C-8 as sufficient protection against concurrent writes. The theoretical objection holds no commit. See Q10 for why an immutable connection's journal-mode report must not be used to infer the file's persistent journal mode.

## Q7 — Missing observations

UNAVAILABLE correctly distinguishes missing evidence from an observed failed invariant. Missing ledger and staging inputs produce unavailable legs (`tests/gate_s172_prod_shape.py:150`, `:224`, `:370`, `:506`); neither can yield PASS, because unavailable legs prevent it (`:538`). Other observed failures can still make the overall sentinel FAIL (`:533`). My temporary absent-ledger and absent-directory checks both observed the intended unavailable legs.

## Q8 — RED arms, mutants, and a masked assertion

The new suite passed 29/29. Its mutants require an exact replacement target and executed hit sentinels (`tests/test_gate_prod_shape_scoping.py:98`, `:128`). The observed REDs were specific: MUT-1 changed B5/B6, MUT-2 changed C2, MUT-3 changed B7, MUT-4 changed the no-selection verdict, MUT-5 caused ledger legs to be judged despite a nonempty WAL, MUT-6 created fixture sidecars, MUT-C1 missed a PID temp, and MUT-B7L missed an oversize inline shard (`:507`, `:517`, `:527`, `:549`, `:559`, `:568`, `:578`, `:538`). Those are useful controls.

**Name: S-E.** Its B0 assertion is specific, but its overall FAIL/exit-1 assertion is masked. Every fixture uses an absent log and empty generation directory (`tests/test_gate_prod_shape_scoping.py:162`, `:164`, `:252`), while S-E checks B0 plus the overall exit/sentinel (`:311`). I changed only the in-memory verifier's final failure collection to ignore B0. S-E's exact predicate still passed because A/D supplied unrelated failures (`tests/gate_s172_prod_shape.py:524`, `:533`; temporary reproducer `targeted_checks.py:75`).

Suggestion: add an otherwise-passing control so unknown-run failure alone must determine the overall verdict. Also assert exit 2 specifically in S-F, which currently accepts any nonzero code (`tests/test_gate_prod_shape_scoping.py:318`). These are test-strengthening suggestions; I found no wrong-reason detection among the listed mutants' named target legs.

## Q9 — Bind acceptance evidence before accepting Run 4

No objection to recording A's linkage as open and repairing it separately before Run 4's result is accepted. Section A examines the supplied log without a run-id argument; `main` passes those inputs separately (`tests/gate_s172_prod_shape.py:76`, `:474`, `:488`, `:499`). This is already disclosed (`docs/CLAUDE_CODE_REPORT_PROD_SHAPE_RUN_SCOPING.md:260`). Committing B/C scoping does not close it. See additional finding 1 for the corresponding artifact linkage.

## Q10 — Investigate the archive rewrite before Run 4

Yes: perform a bounded local provenance investigation before Run 4, preserve the current files and manifests, and record the result or explicitly unresolved attribution. This is a suggestion about evidence handling, not a hold on the verifier commit or a demand to restore any file.

Independently verified: the manifest records `4846d34fdbdc1455bd275bff1f04fef9b2dfc352d6b17d4d70f7176ca2013336` and preserved committed-state sidecars (`logs/brief_i_evidence/LEDGER_ARCHIVE_MANIFEST.json:7`, `:11`). The archive copy now hashes to `c5d942a232da599ab694096a373ea614e66ed63a70170d27439fd489249e7d46`, has no sidecars, reports journal mode `delete`, and passes `PRAGMA integrity_check`. The displaced original's main-file hash still matches the manifest and its WAL is 4,148,872 B. I hashed those original bytes but did not open its database or read its WAL state.

**Correction to the supplied explanation:** `PRAGMA journal_mode` above was queried through an immutable connection. All three original database headers (live, archive copy, displaced original) have read/write version bytes `[2, 2]`, the WAL header values. A freshly created, known WAL-mode temporary fixture also reports `delete` through an immutable connection while retaining `[2, 2]` in its header. This was reproduced by `targeted_checks.py:26`; the original header-only reads are at `:87`. Thus the immutable PRAGMA result does not establish conversion of the archive file to DELETE mode. Alpha's claimed mode switch is unsupported and conflicts with the header evidence (`docs/CLAUDE_CODE_REPORT_PROD_SHAPE_RUN_SCOPING.md:218`, `:237`). Correct that account; the verified hash mismatch and missing sidecars remain to investigate.

**Not independently verified:** Alpha's original-plus-WAL row-digest comparison and attribution of the rewrite. Those remain reported evidence (`docs/CLAUDE_CODE_REPORT_PROD_SHAPE_RUN_SCOPING.md:224`, `:237`). A matching row count, integrity check, or modification time does not identify the writer or establish unchanged row content. Keep the archived run informational pending provenance disposition.

## Execution record

All Python execution used the activated environment and suppressed repository bytecode writes:

```bash
source ~/venvs/torch/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=.
python -u tests/test_gate_prod_shape_scoping.py
python -u tests/test_s172_staging_partb.py
python -u tests/test_s172_staging_backpressure.py
```

The three suites ran sequentially. Full outputs are in `/tmp/X-0001_R1_beta.u7OXS9/`, named after each suite with `.log` extensions.

| Check | Observed result | Review classification |
|---|---|---|
| Scoping suite | 29/29 PASS, completion sentinel PASS, exit 0 | Verified |
| Part B regression | 23/24 green, suite sentinel FAIL, exit 1 | Full regression UNAVAILABLE: G-ROUTE-2 could not access writable live staging |
| Backpressure regression | 24/52 green, suite sentinel FAIL, exit 1 | Full regression UNAVAILABLE: 28 checks stopped with socket PermissionError/EPERM |

G-ROUTE-2 calls the validator on the manifest's live path (`tests/test_s172_staging_partb.py:600`). It stopped at the validator's access check, before the file-writing probe (`miner/range_miner_coordinator.py:12191`, `:12195`). That probe would violate Beta's read-only live-state contract, so I did not attempt to enable or rerun it. The socket checks use local fixtures, including a loopback listener (`tests/test_s172_staging_backpressure.py:1155`), but this sandbox forbids their sockets. I did not alter tests or seek an execution bypass. These are unavailable checks, not passing regressions and not demonstrated defects in this patch. Alpha's reported 24/24 and 52/52 were not reproduced here.

The independent throwaway checks described in Q3/Q4/Q6/Q7/Q8 ran successfully from `/tmp/X-0001_R1_beta.u7OXS9/targeted_checks.py`. Production modules, source, report, manifest and review-history documents were otherwise only read. No pipeline, worker, fleet connection, installation, commit, push or live-state mutation was performed.

### Two real read-only checks after hardening

```bash
python -u tests/gate_s172_prod_shape.py \
  --run-id distributed_config_t1_8cb116da \
  --log logs/gate12_20260822_182119.log
python -u tests/gate_s172_prod_shape.py \
  --ledger /home/michael/miner_ledger_archive/miner_ledger_pre_separation_9runs.db \
  --run-id distributed_config_t1_554463d3 \
  --log logs/gate12_attempt9_launcher_20260817_181819.log
```

| Leg | Live run 3 | Archived attempt 9 (informational) |
|---|---|---|
| A1–A8 | All PASS | All PASS |
| B1 | FAIL: aborted | PASS: committed |
| B2 | PASS: 120/128 done | PASS: 128/128 done |
| B3 | PASS: 5,333/5,348 verified | PASS: 5,693/5,693 verified |
| B4 | PASS: phases 1,2,3,4 | PASS: phases 1,2,3,4 |
| B5 | FAIL: 0/5,348 acked | PASS: 5,693/5,693 acked |
| B6 | FAIL: 0/5,348 cleaned | PASS: 5,693/5,693 cleaned |
| B7 | PASS: 0 remote, 5,348 inline, 0 oversize inline | PASS: 0 remote, 5,693 inline, 0 oversize inline |
| B8 | PASS: 0 held / 5,333 reservations | PASS: 0 held / 5,693 reservations |
| B-OTHER | 0 shards / 0 runs | 23,389 shards / 8 runs |
| C1–C4 | All PASS | All PASS against today's directory only |
| C-OTHER | 0 staged + 0 temps / 0 runs | 0 staged + 0 temps / 0 runs |
| D* | FAIL: no generation directory | FAIL: no generation directory |
| Overall | 17 pass / 4 fail / 0 unavailable; FAIL, exit 1 | 20 pass / 1 fail / 0 unavailable; FAIL, exit 1 |

Both B7 outputs named the worker limit, 50,331,648 B. Full outputs: `live1a.log` and `live1b.log` in the temporary evidence directory. Neither run received an overall PASS.

### Before/after fingerprints

Before and after both verifier executions, recorded each ledger's SHA-256, its WAL/SHM presence and hashes, and each containing directory's sorted entries, sizes, modes and modification times. The complete JSON records are `live_fingerprint_before.json` and `live_fingerprint_after.json` in the temporary evidence directory. They are identical.

| File | SHA-256, identical before and after |
|---|---|
| Live `miner_ledger.db` | `7bd1021310043bce39c4b3822c1eac0e7a604cd9f81d7d28112433e3e21a5c2c` |
| Live `miner_ledger.db-wal` (0 B) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| Live `miner_ledger.db-shm` | `fd4c9fda9cd3f9ae7c962b0ddf37232294d55580e1aa165aa06129b8549389eb` |
| Archive `miner_ledger_pre_separation_9runs.db` | `c5d942a232da599ab694096a373ea614e66ed63a70170d27439fd489249e7d46` |

Directory listings, identical before and after (sizes in bytes):

```text
/home/michael/miner_staging/
  miner_ledger.db                          6434816
  miner_ledger.db-shm                        32768
  miner_ledger.db-shm.pregate12_20260809       32768
  miner_ledger.db-wal                            0
  miner_ledger.db-wal.pregate12_20260809           0
  miner_ledger.db.pregate12_20260809         917504
  threshold_provenance.json                    512
/home/michael/miner_ledger_archive/
  displaced_original/                        4096
  miner_ledger_pre_separation_9runs.db     33099776
  archive database -wal / -shm: ABSENT
```

The three reviewed working-tree files and the worker/coordinator sources retained the same hashes at the later read. Reviewed verifier SHA-256: `fe47e81c0f0209494473e7753a10400d49984bd6f0265a1ac5ec82cb267f90df`; suite: `685f7be44554fa7d7c164c79e30668bb62038fddc7b8c6f7fb58fe83ecfcefbd`; report: `090ab77059dd66af8d1ecc1ccc341cab8ff3da4c7b9f8399368dc67d0254168a`.

## Additional findings

1. **Section D also lacks run attribution.** It selects the newest generation by default, then the newest NPZ beneath the chosen directory, and checks contract/load-back without comparing it to `--run-id` (`tests/gate_s172_prod_shape.py:511`, `:428`, `:433`, `:438`, `:445`). Inference: a different run's structurally valid artifact could satisfy D. This is unchanged code and is not a hold on this bounded patch. Include artifact identity alongside Q9's log identity before accepting a production result. I searched `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md` and the exchange/report material before recording this; the supplied report explicitly records A's gap but does not establish D's linkage. No overall wrong-PASS artifact reproducer was run.
2. **Known statuses, not new defects:** historical C cannot certify a past directory state, B2/B3/B4 remain existential assertions, and C4 lacks a demonstrated producer filename. The report already records these (`docs/CLAUDE_CODE_REPORT_PROD_SHAPE_RUN_SCOPING.md:254`, `:261`, `:265`); source reads confirm the relevant predicates (`tests/gate_s172_prod_shape.py:246`, `:255`, `:263`, `:372`, `:397`). They remain outside this change's claimed proof.

This reply is the only repository file written by Beta for this round. Michael decides disposition.
