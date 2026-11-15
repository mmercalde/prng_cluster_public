## Prior record

**Searches and reads in this session:**

- Read `AGENTS.md`, the R1 request, the complete implementation report, the Step-1 credit-WAL brief, addendum §A/§B, and the tracked diff against `c47680a`.
- Ran `rg -n -i '\bwal\b|immutable|sidecar|quiescen|readonly_shm|non.empty.*ledger|last connection'` across `docs/PROJECT_FILE_CATALOG.md`, `docs/TB_RULING*`, `docs/PROPOSAL*`, `docs/CHAPTER*`, `docs/*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/LEADS.md` and `docs/exchange/`. Read relevant SQLite/credit contexts; unrelated model-sidecar hits were not governing SQLite records. Narrowed the governing search to WAL/immutable/quiescence terms. Logs: `/tmp/beta_step1_credit_wal/record_hits.txt`, `governing_wal.txt`.
- Searched those governing scopes and the search-record exchange for `R-LIVE-WRITER`, `G-EVID-WAL`, stale-positive/checkpoint fixtures, `repeat_of`, study ownership and opening reasons. Read the relevant prior reviews and search proposal. Read the relay snapshot protocol and implementation, source credit/launch/preflight paths, coordinator teardown and stripe mutation sites, and the affected test functions. Test-status search: `/tmp/beta_step1_credit_wal/test_record_search.txt`.
- Read Alpha's saved suite summary and Q2 fixture log. Checked SQLite's primary [WAL documentation](https://sqlite.org/wal.html) and [WAL file format](https://sqlite.org/fileformat2.html#walformat).

| Record / item | Label | Disposition |
|---|---|---|
| Step-1 credit-WAL brief; addendum §A; X-FINDINGS-PLAN R1/R2; X-0001 R1 Q6 | ALREADY RECORDED | Source isolation, WAL visibility and quiescence requirements. |
| X-MINER-STALL R1:132 blanket nonempty-WAL refusal | CHANGES A RULING | This implementation changes that recorded behavior; Michael decides. |
| Search-record v1.1:170,301; X-STEP1-SEARCH-CODE R3 | ALREADY RECORDED | Repeat identity grants no credit; study/budget and publication requirements remain. |
| `docs/exchange/RELAY_PROTOCOL.md` §3 | ALREADY RECORDED | Governs Q8's snapshot claim; omitted from the packet's governing-record tables. Finding 1 is this record omission, not a new relay defect. |
| Run-4 cause, copy behavior and disclosed part3 failure | ALREADY RECORDED | Packet claims independently checked below. |
| Replacement R-LIVE-WRITER does not reproduce part3's stale-positive control | NEW | Measured test limitation, addressed under Q7. |

**Not searched/exercised:** private repository, ser8, rigs, Proxmox, live ledgers, live Optuna/search state, or project files outside the repository other than the expressly supplied frozen evidence bundle. No complete reread of all project documents. GPU execution, production launch and hit-rate measurement are outside this review.

## Review header

- **Thread / round:** X-STEP1-CREDIT-WAL / R1.
- **Date / host / user:** 2026-11-15, VM101, `michael`.
- **Class:** BEFORE-COMMIT; code deciding whether a run counts.
- **Base and worktree HEAD read:** `c47680a24a5634445b8548f5db00b0150feb17f4`, plus the uncommitted packet in `/home/michael/tfm_worktrees/step1-credit-wal`.
- **Authority:** Beta records review positions. Michael decides. This review credits no run and authorizes no launch.

Initial `git status --porcelain`, unchanged before writing this review:

```text
 M window_optimizer_integration_final.py
?? data/
?? docs/CLAUDE_CODE_REPORT_STEP1_CREDIT_WAL.md
?? docs/exchange/to_beta/X-STEP1-CREDIT-WAL_R1_ALPHA_REVIEW_REQUEST_credit_on_nonempty_wal.md
?? tests/test_step1_credit_wal.py
```

Reviewed SHA-256 values, unchanged on the final reread at 12:26 PST:

```text
ceea5d67331edf6aaa6fc8193bc204a718709020259e77fb24f8140dd5caca09  window_optimizer_integration_final.py
21454278ccf7806dbb4ab1710c86471111dd065096e864967c977ae743664036  tests/test_step1_credit_wal.py
01e845650975b62287fcef6d7be3cda4d98d7ff2f90b2796f2ef2f26bad301a8  docs/CLAUDE_CODE_REPORT_STEP1_CREDIT_WAL.md
```

## Positions

| Question | Position | Severity |
|---|---|---|
| 1 — Cause | SUGGESTION | — |
| 2 — No new write | NO OBJECTION | — |
| 3 — No stale or torn credit | NO OBJECTION | — |
| 4 — Writers have stopped | NO OBJECTION | — |
| 5 — Addendum §A | SUGGESTION | — |
| 6 — Fail-closed | NO OBJECTION | — |
| 7 — Tests | SUGGESTION | — |
| 8 — Scratch location | SUGGESTION | — |
| 9 — Q2 owner route | NO OBJECTION | — |
| 10 — Q3/Q4 by reading | SUGGESTION | — |
| 11 — Governing record | SUGGESTION | — |

**No LIVE objection found.** The suggestions below include wording for the owner disposition and preservation of the existing stale-positive test. The disclosed failing suite remains a failure until its intended assertion change is decided and implemented.

## 1 — Cause

**The Run-4 mechanism is confirmed. Narrow “every run” to runs where the sampler remains connected through the last writing connection's close.** The sampler opens `mode=ro` and leaves `conn` bound between iterations (`scripts/gate12_concurrency_sampler.py:202-211`, `:1098`). The harness starts it first and schedules its supervisor's termination after WATCHER exits (`gate12_launch.sh:364-395`).

Independently ran the real sampler script against a temporary real-MinerLedger fixture. Three checks between samples found database/WAL/SHM descriptors open. After the fixture's writers closed, WAL size was 931,152 B: the base reader returned `None`, while the new reader returned all four intervals. After terminating the read-only sampler, the WAL remained 931,152 B and the base reader still returned `None`. Evidence: `/tmp/beta_step1_credit_wal/extra.log`.

The universal wording is too strong: the harness passes `--max-seconds 7200`, enforced by the sampler, which can also stop after 180 seconds without runnable stripes (`gate12_launch.sh:367`; `scripts/gate12_concurrency_sampler.py:1057-1059`, `:1087-1089`, `:1140-1147`). These exits can precede a later last writer close. They do not explain away Run 4 or remove the need for this repair.

## 2 — No new source write

**No source-content or sidecar-write route found in the supported configuration.** The helper stats the sources and uses `copyfile` with destinations inside a newly created private scratch directory; its sole SQLite connection names the copy, and cleanup targets that scratch (`window_optimizer_integration_final.py:1440-1473`). The other branch keeps the immutable original connection (`:1515-1523`). No production ledger wrapper is constructed by this reader (`:1483-1527`).

The new suite independently verified unchanged source filenames, sizes, hashes and mtimes, both with and without source SHM (`tests/test_step1_credit_wal.py:208-241`). The direct-original `mode=ro` mutant changed the fixture directory and was detected. This conclusion assumes the normal system-temp configuration; Q8 addresses the configurable scratch location. It does not assert protection against an unrelated process writing the source.

## 3 — Snapshot correctness

**No wrong-credit path found for a stable, intact database/WAL pair at this terminal single-coordinator boundary.** Both source stamps bracket both copies; detected changes return no rows, and read/copy errors reach the outer no-evidence handler (`window_optimizer_integration_final.py:1451-1465`, `:1524-1530`). Source SHM is not copied or consulted (`:1440`, `:1452-1461`). A missing or stale source SHM therefore cannot supply an old end mark to the copy reader.

In addition to the 12-gate suite, independently exercised a recycled WAL with **3 current-generation frames and 328 older-salt frames**. The latest committed phase-1 `staging` state remained uncredited. With a subsequent uncommitted transaction spilling dirty pages into that WAL, the reader still refused phase 1 and left source fingerprints unchanged. Evidence: `extra_checks.py` / `extra.log` under `/tmp/beta_step1_credit_wal/`.

Correct the helper's explanatory sentence at `window_optimizer_integration_final.py:1434-1436`: uncommitted frames can have valid salt/checksums; lack of a committed transaction boundary excludes them. Older-generation tails fail generation validation. SQLite documents these separate mechanisms in its [WAL format](https://sqlite.org/fileformat2.html#walformat). The implementation appropriately delegates recovery to SQLite.

The stamps are a copy-stability check, not a lock or a freshness guarantee extending through the later query (`window_optimizer_integration_final.py:1454-1465`). An independent writer changing this run after the stamps, or the already disclosed timestamp-restoration case, lies outside the supported stopped-writer premise. The broader quiescence limitation is **ALREADY RECORDED** in X-0001 R1 Q6, not a new LIVE finding here.

## 4 — Terminal boundary

**Adequate for the presently supported route, with the limits above.** The evidence read follows the synchronous miner call (`window_optimizer_integration_final.py:1922`, `:1939-1940`). Coordinator teardown stops and joins readers, drains both executors with `wait=True`, and only then builds its terminal return (`miner/range_miner_coordinator.py:10660-10684`, `:10733-10757`). This is stronger evidence than merely observing per-operation SQLite connections.

A `WHERE run_id=?` predicate alone would not prove that committed rows cannot change: generic setters can update the selected run (`miner/range_miner_coordinator.py:1694`, `:2233`). The relevant argument is the completed serve path and drained writers, under the one-coordinator use. The stability check is a useful additional refusal, as R-MOVING demonstrates; it does not replace that premise (`tests/test_step1_credit_wal.py:336-356`).

## 5 — Addendum wording

**The design keeps §A's purposes, but it is an adaptation rather than literal compliance with every procedural sentence.** §A:28 requires no running coordinator/WATCHER/window optimizer; §A:34-36 says database and sidecars, scratch inside the worktree. Here the optimizer is executing the check, scratch is system temp, and SHM is intentionally rebuilt.

Use the already flagged owner disposition to state that narrow exception explicitly: after terminal miner teardown, automatically copy the database and WAL to private system scratch, require source stability, rebuild the index only on the copy, and fail closed on errors. Keep §A's outside-reader prohibitions intact. No additional broad relaxation is needed beyond an owner decision that expressly covers these details and the changed nonempty-WAL credit behavior.

## 6 — Refusals

**No unintended credit relaxation found.** Missing run id, false `committed`, absent database, nonempty rollback journal and unreadable evidence still refuse (`window_optimizer_integration_final.py:1484-1492`, `:1509-1530`). Complete/contiguous phase extraction is unchanged (`:1531-1548`); the downstream decision still requires this chunk's evidence, COMPLETE state, every required phase and exact requested intervals (`utils/step1_search_ledger.py:1055-1086`).

The intentional change permits complete committed WAL evidence even with an open idle writer or absent SHM. It does not use connection presence as evidence of phase completion (`window_optimizer_integration_final.py:1525`, `:1536-1545`). Independent missing-phase, claimed/staging, gap, false-committed, corrupt-header, journal and moving-source controls all refused. The WAL-empty pre-open race remains the recorded unchanged limitation, not a repaired case.

## 7 — Tests and the old gate

**The gates and mutants exercise behavior. R-LIVE-WRITER does not fully carry the original stale-positive fixture.** The new assertions check actual intervals/credit, source fingerprints and cleanup; the four mutants produced the intended failures (`tests/test_step1_credit_wal.py:208-258`, `:336-356`, `:463-470`).

However, the shared new fixture has **zero stripe rows in its immutable main-file view** (`tests/test_step1_credit_wal.py:244-254`). R-LIVE-WRITER starts from that fixture and commits a staging update (`:283-297`). I replaced only the copy helper with an immutable-original reader: **R-LIVE-WRITER still passed**, refusing because all four phases were missing. The original part3 fixture instead checkpoints complete rows into the main file before committing a staging update; it proves that immutable reading would wrongly return phase 1 as complete (`tests/test_s172_small_repairs_part3.py:169-184`). Under that same mutant its safety assertion failed with phase 1 credited (`:191-192`). Both observations are in `extra.log`.

**Recommendation:** retain the old fixture and its phase-1 safety assertion, and retire/reword only its blanket `ev is None` / `UNAVAILABLE` requirements (`tests/test_s172_small_repairs_part3.py:193-194`) after Michael's decision. Alternatively make R-LIVE-WRITER checkpoint its complete baseline and explicitly show the stale-positive immutable control. G-STALE already kills the blanket immutable mutant, but that does not make the two fixtures equivalent. This is a test-coverage suggestion, not a demonstrated production wrong-credit defect.

## 8 — Scratch location

**Private system temp is technically suitable; record the policy exception in Q5's owner disposition.** It avoids repository dirt and directs SQLite's recovery/index writes away from source files (`window_optimizer_integration_final.py:1449-1473`). The clean predicate tests repository porcelain (`:119-122`), and the relay inventories repository files and the configured live ledger, not arbitrary `/tmp` contents (`scripts/beta_relay.py:247-283`). No gate/allowlist widening is needed.

Do not describe cleanup as unconditional: `rmtree(..., ignore_errors=True)` is best effort, and a killed process cannot execute `finally` (`window_optimizer_integration_final.py:1472-1473`). The normal and refusal cleanups passed. Also, `mkdtemp` uses the configured temporary directory; a future `TMPDIR` pointing inside live staging would violate the intended location boundary (`:1449`). That configuration was not established here and is not a LIVE objection. The report's quoted external “box rule 7” was not independently available as a separate governing file; §A's explicit worktree wording is available and requires the stated disposition.

## 9 — Owner retry route

**The commands and operational order are sound for the stated unchanged configuration and dataset.** Child study ownership and consumed-budget checks run before objective dispatch (`window_optimizer_integration_final.py:1636-1647`, `:1572-1580`, `:2486-2492`). Closing the old search, then opening an explicit repeat, is supported; a closed identical identity cannot reopen itself, and only one search may be open for the selector (`utils/step1_search_ledger.py:679-691`). WATCHER routes the close/open/status options and `repeat_of` as reported (`agents/watcher_agent.py:833-846`, `:871-882`).

Independently passed five existing stage-1 controls: exhausted-budget refusal, same-id retry, explicit repeat, close/open and study ownership. Read, but did not independently replay, Alpha's full exact-Run-4 WATCHER/child dry run. No live search command was executed. Landing first is the sensible operational order; it avoids spending another attempt with the same credit refusal. Closing earlier is not itself a code dependency. Re-read the actual status/new identity before launch; the predicted repeat id remains conditional on unchanged identity inputs (`utils/step1_search_ledger.py:661-667`).

## 10 — Report §§6–7

**The practical conclusions stand, with two qualifications.** A later normal WAL-aware coordinator connection can use the committed frames, and a retained nonempty WAL need not be truncated or deleted to read them. But default autocheckpointing is PASSIVE and can be incomplete while readers hold snapshots; physical file length alone does not prove successful checkpoint progress. Replace the unconditional future checkpoint sentence in report §6 with that conditional account. The coordinator sets WAL/busy-timeout/synchronous but no custom checkpoint setting in `_conn` (`miner/range_miner_coordinator.py:1207-1213`); SQLite describes the default and its reader limitation in [WAL §§3.1–3.2](https://sqlite.org/wal.html#automatic_checkpoint). Apply Q1's sampler-lifetime qualification to the universal recurrence wording too.

Section B's refusal is confirmed **by reading**: nonempty WAL returns no connection, records B* unavailable, and prevents an overall PASS; other failed legs can make the overall result FAIL rather than INCOMPLETE (`tests/gate_s172_prod_shape.py:150-156`, `:224-227`, `:533-542`). This reader repair does not repair that verifier. Its disposition remains necessary before claiming a full production-shape acceptance; this review accepts no run.

## 11 — Governing-record addition and findings

1. **Prior-record omission — ALREADY RECORDED:** add `docs/exchange/RELAY_PROTOCOL.md` §3, especially `:64-81`, to the governing-record table for Q8. It specifies the repository and live-ledger snapshot boundary. The matching implementation was read at `scripts/beta_relay.py:247-283`. This supplements the already cited addendum; it is not a new relay flaw or a new credit requirement.

No further unasked production defect found. The Q7 test limitation and Q1/Q10 wording corrections are addressed under their requested questions, not duplicated as findings.

## Execution and limits

All checks activated `~/venvs/torch`, disabled bytecode writes and exposed no CUDA devices. Test `SCRATCH` assignments were redirected **in memory** to `/tmp/beta_step1_credit_wal/`; assertions and repository sources were unchanged. Python audit guards refused mutations and database connections outside `/tmp`, including in the real sampler child. No active coordinator/WATCHER/optimizer process was listed before execution.

| Independently executed | Result |
|---|---|
| New credit-WAL suite | 12/12 gates; 4/4 mutants detected; PASS |
| part3 evidence-open and evidence-WAL functions | OPEN passed; WAL failed only the disclosed blanket refusal, returning phase 2 and withholding phase 1 |
| Real sampler process with real MinerLedger fixture | Persistent between-sample descriptors; base refused during and after read-only last close; new reader returned all four intervals |
| Old-generation tail and uncommitted spilled transaction | Latest committed incomplete phase remained uncredited; source unchanged during evidence read |
| Immutable mutant against replacement vs original stale-positive fixture | Replacement passed; original caught stale positive |
| Five selected existing search-record controls | 5/5 passed |
| Frozen Run-4 ledger, copied again into `/tmp` | All four phases `[2147483648,4294967296)`; copy fingerprints unchanged; scratch removed |
| Supplied frozen ledger files vs `SHA256SUMS.txt` | 3/3 hashes matched |

Harnesses/logs: `/tmp/beta_step1_credit_wal/run_checks.py`, `checks.log`, `extra_checks.py`, `extra.log`, `search_checks.py`, `search.log`. The selected search checks emitted the already disclosed nonfatal missing `schemas/field_mappings.json` diagnostic and still passed their assertions.

**Read only:** Alpha's complete base/patched regression summary (part3 3/3 vs 2/3; stage-1 39/39 and 33/33 mutants on both; seed-start 10/10 and 11/11 mutants on both). Full independent reruns of those remaining batteries are **UNAVAILABLE in this review**, not additional Beta passes. Production-shape execution, live retry, fleet behavior and predictive hit rate are also **UNAVAILABLE**. Only this review was written inside the repository. No live ledger/search state, halt file, pipeline, worker, remote machine, package, commit or push was touched.
