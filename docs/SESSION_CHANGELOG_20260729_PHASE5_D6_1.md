# SESSION_CHANGELOG — 2026-07-29 — S172 Phase-5 **D6.1**

**Incremental NPZ atomic flush and durability repair.**

**Base:** `4c421fc` (D6 closed and pushed; release-grade generation certified at
`b08c2c5`). Implemented on VM 101 as `michael`, venv `~/venvs/torch`.
**Not committed, not pushed, WATCHER not run** — stopped at the gate for Team
Alpha review.

**Status: implementation complete, all gates green — but D6.1 CHANGED SHAPE
mid-flight and Team Beta must rule on the scope change.** See §2.

---

## 0. Behavioural change in one sentence (Beta §2)

**D6.1 changes the flush from an always-failing attempt into a real provisional
snapshot write.** Before D6.1 `_flush_npz_incremental` failed on every single
invocation since S152 and printed a swallowed warning, so nothing was ever
written to disk. After D6.1 it actually writes: a real, fsynced, atomically
replaced, transaction-identified pair of four-field NPZ members under
`.s172_checkpoint/<run_id>/`.

That is a genuine change in observable behaviour, not a no-op repair, and it is
why the hygiene work in §12 exists — code paths and test suites that previously
"called the flush" without consequence now materialise files.

The snapshot remains **non-authoritative** (see §6.3): provisional, four-field,
never consumed by the finalizer or Steps 2-6.

---

## 1. What was actually broken

Beta's framing was correct and, if anything, understated: **incremental
durability did not exist.** The S152 helper failed on *every* invocation since
it was written, and a broad `except Exception` converted each failure into an
ignorable stdout warning.

All four briefed defects were reproduced at source before any edit.

| # | Defect | Verification |
|---|---|---|
| **D1** | `.npz` suffix bug, ×2 (`:295`, `:300`). Temp name lacked `.npz`, so `savez_compressed` appended one and wrote `...flush.tmp.npz`; `os.replace` then targeted a name that never existed. | Reproduced on numpy 1.22.0: file created was `bidirectional_survivors_all.npz.flush.tmp.npz`; `os.replace` raised `FileNotFoundError`. |
| **D2** | Broad `except Exception` (`:318`) turned every failure — including total outage — into one non-fatal warning. | Read at source. This is why D1 went unnoticed since S152. |
| **D3** | List-clear inside the same `try`, after both replaces. Only *accidentally* protective: the exception fired at the first `os.replace`, so the clear never ran. Fixing D1 makes the ordering load-bearing. | Read at source; now gated, not assumed. |
| **D4** | The S166 comment asserted the survivors were already safely persisted — a guarantee that has never once held. | Read at source; corrected. |

### 1.1 A fifth defect the brief did not anticipate

**The briefed repair, applied as written, would have permanently broken
generation publication.**

The helper wrote to relative `bidirectional_survivors_all.npz` and
`bidirectional_survivors_binary.npz`. **Since D3.5 those two names are
compatibility SYMLINKS owned by the finalizer**
(`utils/run_finalizer.py:1400-1404`, `_bootstrap_root_aliases`), pointing into
`.s172_accumulator/current/`. The finalizer **fails closed** if a regular file
appears at either path:

> *"The historical root artifacts were explicitly removed under Ruling F: a
> regular file reappearing at those paths means something wrote outside the
> finalizer, and the run must stop rather than overwrite it."*

Proven by reproduction, not inferred:

```
after bootstrap: bidirectional_survivors_all.npz  islink=True -> .s172_accumulator/current/...
after repaired flush:                             islink=False isfile=True
next finalize:   PublicationError: ... exists as a regular file or directory,
                 not the expected compatibility symlink
```

Root cause: **S152's checkpoint predates D3.5, which reassigned those paths out
from under it.** The helper is a vestige of the pre-D3.5 world, still called
once per trial, aimed at names it no longer owns. The collision stayed
invisible for the same reason D1 did — the write never succeeded.

Two further consequences of simply "making the write work":

- `window_optimizer_integration_final.py:1868` feeds the **in-memory list** to
  the D3.5 finalizer, which requires all **24** `CANONICAL_RECORD_FIELDS`. The
  checkpoint persists **4**. Enabling the S166 clear would truncate the
  certified generation's raw-candidate input; 20 of 24 fields are
  unrecoverable.
- The merge step would read a certified 22-array artifact through the symlink
  and rewrite it as a 4-array file.

---

## 2. ⚠️ SCOPE CHANGE — Team Beta ruling required

D6.1 was mandated as **"repair the flush in place."** That is not achievable
safely. The delivered work is instead:

> **relocate the checkpoint to its own namespace, and defer the in-memory
> clear.**

Michael (Alpha) confirmed this direction during the session. **Beta has not
ruled on it.** Two items are recorded explicitly, per Alpha's instruction:

### (a) The S166 in-memory OOM clear remains UNIMPLEMENTED and is DEFERRED as
### its own tracked item — it must not be silently dropped

`_FLUSH_CLEAR_IN_MEMORY = False`. The clear is *not* implemented, only
positioned and gated.

- **Why deferred:** a 4-array checkpoint cannot reconstruct the 24 canonical
  fields the D3.5 finalizer consumes from the in-memory list. Clearing today
  would silently truncate certified generations.
- **What it blocks:** unbounded in-memory candidate growth. **This blocks
  Phase 7 soak safety exactly as the flush defect did** — a long multi-trial
  soak accumulates every candidate for the run's whole lifetime with no
  bound. S166 added the clear for a real OOM on Zeus; that protection has
  never actually been active, so the exposure is pre-existing, not new — but
  it is now *known*, and it is a soak blocker in its own right.
- **What enabling it requires:** (1) a checkpoint carrying all 24
  `CANONICAL_RECORD_FIELDS`, and (2) a finalizer read-back path that rebuilds
  raw candidates from the checkpoint. Both are new capability, not defect
  repair.
- **Mitigation now:** the *ordering* property is fully gated with the flag
  forced on (`G-CLEAR-AFTER`, mutant M2), so enabling it later is a one-line
  change against a gate that already proves the clear can only run after both
  replaces succeed.

### (b) The path collision is a defect **in the D6.1 brief itself**

The brief specified requirement 3 ("the in-memory candidate list clears ONLY
after both replaces have succeeded") and the in-place repair of D1 **without
checking that `window_optimizer_integration_final.py:1868` feeds the in-memory
list to a finalizer requiring all 24 `CANONICAL_RECORD_FIELDS`**, and without
checking that the two target paths are finalizer-owned symlinks.

Had D6.1 been implemented exactly as briefed, the first successful flush would
have replaced both symlinks with regular 4-array files and **every subsequent
`finalize_run` would have raised `PublicationError` — generation publication
permanently broken** — while the clear silently truncated the certified
generation's input.

This is the same failure mode D4 documents: **a stated guarantee that was never
checked against the code.** It is recorded here as a brief-authoring defect, not
an implementation surprise.

---

## 3. What changed

### `window_optimizer_integration_final.py`

- **Checkpoint namespace** — `.s172_checkpoint/incremental_survivors_all.npz`
  and `.../incremental_survivors_binary.npz`. Never the finalizer-owned root
  names. *Nothing can depend on the old location: the write has never once
  succeeded, so no consumer has ever observed a file there.*
- **D1 fixed via the write mechanism, not the name** — `savez_compressed` is
  handed an **open file handle**, so numpy writes to exactly the path given.
  See §4 for why this mechanism was chosen.
- **Real durability** — each temp is `fflush` + `fsync`ed before `os.replace`,
  and the checkpoint directory is fsynced after. `os.replace` is atomic for the
  *directory entry* only; without fsync a power-loss crash can leave the renamed
  file truncated. "Atomic" without durability is not a checkpoint.
- **Sequential-atomic with self-repair** — both temps written to completion
  first, then the two replaces back-to-back. **Not** claimed to be jointly
  atomic.
- **D2 fixed** — three-tier failure contract (§5).
- **D3 fixed** — clear gated behind `_FLUSH_CLEAR_IN_MEMORY`, positioned
  strictly after both replaces.
- **D4 fixed** — the false guarantee is gone; the documentation now states the
  property the code actually keeps.
- **Crash-orphan collection** — stale temps are purged by the next flush, but
  **only** when the owning pid is dead (`os.kill(pid, 0)`). `optimize_window`
  can run partition workers in parallel against one CWD, and a blind `*.tmp`
  sweep would delete a live sibling's in-flight write.
- `_flush_last_count = current_count` on success (restoring the pre-S166
  semantics, correct now that the list is not cleared).
- Observability: `_flush_success_count`, `_flush_failure_count`,
  `_flush_last_error`.

### `tests/test_s172_phase5_d6_production_adapter.py` — cadence gate re-anchored

Two assertions in `g_flush_cadence` pinned the **broken** behaviour and had to
move (brief §2.8 / G-CADENCE anticipated this):

- **(1)** demanded the whole helper be byte-identical to `2a6e0f8`. Replaced
  with a verbatim pin of the **entry gate** only — the cadence rule D6 actually
  owns — cross-checked against the frozen commit so the oracle itself cannot
  drift.
- **(5)** asserted the flush *fails* ("proceeds past the gate" measured on
  stdout, because the NPZ never landed). Now pins the **successful** flush: the
  checkpoint files exist, and the finalizer-owned root names are untouched.

### `.gitignore`
- `.s172_checkpoint/` — run-local crash-recovery state, never an artifact.

### `tests/test_s172_d6_1_flush_durability.py` — new, 13 gates / 7 mutants

---

## 4. Alpha's implementation calls

**Suffix mechanism — open file handle** (Beta left the mechanism free, gated the
property). Both candidates were validated empirically; the handle was chosen
because it is the only one that **also enables `fsync` on the exact descriptor
numpy wrote**. Naming the temp `...tmp.npz` would have required reopening the
file to fsync it, adding a window where the "durable" temp is not durable.

The gate pins the **property**, not the mechanism: `G-SUFFIX` first asserts the
temp name does *not* end in `.npz` — deliberately the D1-prone shape — then
asserts the file numpy creates equals the path `os.replace` consumes. The
property therefore holds for any future temp name, including one that
reintroduces the exact bug. Mutant **M1** restores the un-suffixed *name* form
and `G-SUFFIX` reds.

**Compression — `savez_compressed` kept** (Call B). Documented loudly at the
call site, naming D5 §6.7.A and mutant M6a, so nobody "harmonizes" the two.
`G-COMPRESSION-CONTRACT` proves the contracts stay **separate**: the checkpoint
is DEFLATED, *and* the D5 artifact writer still emits `ZIP_STORED` and still
carries M6a's exact mutation anchor.

**Sequential-atomic, not jointly atomic** (Call A). The pair *can* be
inconsistent; the claim is only that it is **detectable** and **self-repairing**
— gated in `G-CRASH-RESTART`, not asserted in a comment.

---

## 5. Exception-handling contract (D2)

**Non-fatal to the trial** (the established contract, docstring `:252`) **but
never silent**, in three tiers:

| Tier | Trigger | Behaviour |
|---|---|---|
| **Expected / recoverable** | unreadable prior checkpoint | stdout `Warning`, drop the merge, continue from memory, **flush still succeeds**. This is the normal post-crash self-repair path, not an incident. |
| **Write failure** (`OSError`) | ENOSPC / EACCES / EIO | **stderr `ERROR`** + traceback, `_flush_failure_count += 1`, **all candidates retained** |
| **Unexpected** (any other) | contract/programming error | **stderr `UNEXPECTED ERROR`** + traceback, counted, all candidates retained |

A checkpoint failure never kills the trial, but a human sees stderr and a
soak/WATCHER can observe the counters. Mutant **M5** re-broadens the handler to
the pre-D6.1 single stdout warning; `G-VISIBLE-FAILURE` reds.

---

## 6. The three crash points

| Crash point | What a restart observes | Repair |
|---|---|---|
| **(a) before any replace** | Both finals at their **complete prior** content. No temps (orphans collected by the next flush). All candidates still in memory. | Nothing to repair |
| **(b) between the two replaces** | **Mixed pair**: `_all` advanced, `_binary` did not. Each file is **individually complete and loadable** (`zipfile.testzip()` clean). Detected by **transaction identity** — see §6.1. All candidates still in memory. | **Self-repairs the SNAPSHOT on the next flush** — readable member A ∪ memory, both rewritten. Not accumulator resume. Gated. |
| **(c) after both replaces** | Consistent, complete pair. Candidates still in memory (clear disabled). | Nothing to repair; **replaying the flush is idempotent** (merge-by-seed dedup), so no double-counting. |

### 6.1 ⚠️ CORRECTION — "detectable by seed-set comparison" was WRONG

**REV2, after Beta's review.** D6.1's first report claimed crash point (b) was
*"detectable by seed-set comparison."* **That claim is false and is retracted.**
Beta's counterexample disproves the general case:

> Old pair holds seed 42 @ score 0.40. The new transaction holds seed 42 @ 0.90.
> A crash after replacing member A leaves **A = 0.90 / B = 0.40 with identical
> seed sets `{42}`** — seed-set comparison reports *agreement* across two
> different transactions. The same hole exists whenever only the match rates
> change.

This is the second time in this workstream a stated guarantee was not checked
against the code — the same failure mode as D4 and as the §1.1 path collision.

**Fix — transaction identity on both members.** This does **not** require
D6.2's 24-field schema. Both members now carry, produced from **one transaction
descriptor built before either temp is written**:

| Field | Type | Purpose |
|---|---|---|
| `checkpoint_schema_version` | str | marks the interim four-field format so D6.2 can tell it apart |
| `checkpoint_id` | str | unique per transaction |
| `checkpoint_sequence` | int | monotonic within a run |
| `run_id` | str | stable run identity |
| `logical_candidate_count` | int | rows in this transaction |
| `four_field_content_digest` | str | sha256 over **all four** fields, seed-sorted canonical order |

Detection compares **transaction identity, never seed sets**
(`_flush_identity_differs`). Rows are written in seed-sorted order so the digest
depends on content alone, not dict insertion order.

**Load contract, all five outcomes gated:**

| On load | Result |
|---|---|
| matching identity + digest | **accept** (`consistent`) |
| mismatched identity / sequence | **interrupted replacement** |
| matching seeds, differing field digest | **inconsistency detected** |
| one member unreadable | **pair incomplete**, repaired where available state permits |
| neither valid | **recovery fails visibly**; in-memory records untouched |

A member carrying **no identity block** (pre-D6.1 or foreign) is **refused**,
not guessed at.

`G-TRANSACTION-IDENTITY` asserts *both* halves: that seed-set comparison
genuinely cannot see the difference — so the gate has teeth — and that the
production detector classifies it as an interrupted replacement. Mutant **M8**
reverts detection to seed-set-only and the gate reds with exactly Beta's
scenario.

### 6.2 Beta's `.s172_checkpoint/` path conditions — all six applied

| # | Condition | How |
|---|---|---|
| 1 | Git-ignored | `.gitignore: .s172_checkpoint/` |
| 2 | **Not CWD-dependent** | resolved from a stable root — `PRNG_CHECKPOINT_ROOT`, else this module's own directory. A mid-run `os.chdir` cannot move or fork the snapshot. |
| 3 | **Run-isolated** | `.s172_checkpoint/<run_id>/`, run id stable per process (`PRNG_CHECKPOINT_RUN_ID` overridable), so consecutive/concurrent runs cannot collide |
| 4 | Same filesystem | temp and destination in the same directory; asserted by `st_dev` at runtime, fail-closed |
| 5 | **Never a finalizer alias** | `_flush_assert_not_alias` checks basename **and** realpath against both aliases, before anything is written |
| 6 | Explicit schema version | `s172-d6.1-four-field-v1`, carried inside every member |

All six are gated by `G-PATH-CONDITIONS`.

**Two consequences worth recording, neither a defect:**

1. With no `PRNG_CHECKPOINT_ROOT` set, the stable root is this module's
   directory — i.e. **the repo tree**. That is what conditions 1 and 2 ask for
   together (stable *and* git-ignored), and `git status` stays clean. It does
   mean snapshot state now materialises where pre-D6.1 nothing ever appeared,
   because the write never succeeded.
2. **`tests/test_s172_phase5_d3_25_candidate_ingress.py` now writes a real
   snapshot** when run: several of its gates drive `_build_test_result_from_pw`
   with a live accumulator rather than the flush spy, so the repaired helper
   actually runs. D3.25 stays **13/13** and its file is untouched (brief §4);
   the output is run-isolated and git-ignored. Recorded rather than "fixed",
   because silencing it would mean editing a protected harness.

**Open follow-up for Beta — retention.** Nothing prunes
`.s172_checkpoint/<run_id>/`. Each process creates one directory and none are
ever removed, so a Phase-7 soak or a WATCHER-driven series accumulates them
indefinitely. Run isolation is what Beta required and is implemented; a
retention/GC policy is new capability and is deliberately **not** in D6.1.

### 6.3 ⚠️ CLAIMS NARROWED TO BETA'S CONTRACT

Wording was corrected in the **code, comments, and this changelog**. D6.1
repairs **writing, visibility, per-file atomic replacement, and isolation** of
the **non-authoritative four-field incremental snapshot**. It does **NOT**
provide:

- full accumulator resume,
- finalizer reconstruction,
- S166 in-memory memory protection.

**The in-memory list remains the finalizer's authoritative source.** The
snapshot is **non-authoritative until D6.2**. It is **not** a canonical
accumulator checkpoint, must **not** be used to reconstruct finalizer input,
and its merge-by-seed is **provisional snapshot maintenance only** — it decides
no winner; D3.5's explicit L2 key remains the sole authority.

`G-COMMENT-TRUTH` now enforces this in both directions: it fails if the code
reasserts the retracted seed-set claim or the false S166 guarantee, and fails if
the three scope disclaimers are missing.

---

## 7. Gate results — `tests/test_s172_d6_1_flush_durability.py`

**15/15 green · 8/8 mutants killed** under the four-part rule
(applies-once · mutated-path-executes · detector-clean-unmutated ·
fails-from-the-injected-defect).

```
G-SUFFIX                 the temp target cannot be .npz-rewritten by NumPy
G-ATOMIC-ACCUM/BINARY    complete prior or complete new, never partial
G-CLEAR-AFTER            the list clears only after BOTH replaces succeed
G-RETAIN-ON-FAIL         zero candidate loss at four injection points
G-NO-TEMP-LEAK           no temp survives success or any failure
G-CUMULATIVE             exact cumulative counts, dedup + prior merge intact
G-CRASH-RESTART          three crash points, detected + snapshot self-repair
G-TRANSACTION-IDENTITY   mixed pair caught when the seed sets are IDENTICAL
G-PATH-CONDITIONS        ignored, CWD-free, run-isolated, same-fs, versioned
G-CADENCE                entry gating unchanged; pins SUCCESSFUL flush
G-VISIBLE-FAILURE        failures are surfaced, tiered and counted
G-NO-ALIAS-COLLISION     finalizer-owned root paths never written
G-COMPRESSION-CONTRACT   snapshot compressed, D5 artifact ban intact
G-COMMENT-TRUTH          no retracted claim reasserted; disclaimers present
G-MUTANTS                mutation proof (8 mutants)
```

Mutants: **M1** un-suffixed temp name → G-SUFFIX · **M2** clear before the
replaces → G-CLEAR-AFTER · **M3** clear on failed write → G-RETAIN-ON-FAIL ·
**M4** temps leaked → G-NO-TEMP-LEAK · **M5** handler re-broadened →
G-VISIBLE-FAILURE · **M6** prior-merge dropped → G-CUMULATIVE · **M7**
snapshot aimed back at the root names → G-NO-ALIAS-COLLISION · **M8** pair
detection reverted to seed-set-only → G-TRANSACTION-IDENTITY.

M8's recorded kill signature is Beta's counterexample verbatim: *"a mixed pair
with identical seed sets was classified 'consistent', not an interrupted
replacement."*

**Harness note.** The mutant runner swaps the *source* every gate builds from
(`_active`), so a mutated module is what the gate actually executes. The first
draft passed a mutant module as an argument while the gates constructed their
own from production — **M2 survived vacuously** and exposed it. Fixed; the
survivor is what caught it.

---

## 8. Non-regression

Captured **before any edit** at `4c421fc`, re-run after. All green, both times:

| Suite | Before | After |
|---|---|---|
| D1.1 | 18/18 | 18/18 |
| D1.0 | 8/8 | 8/8 |
| D2 | 7/7 | 7/7 |
| D0 | 12/12 | 12/12 |
| D3.0 | 10/10 | 10/10 |
| D3 columnizer | 10/10 | 10/10 |
| **D3.25** | **13/13** | **13/13** |
| D3.5 | 60/60 | 60/60 |
| D4 | 8/8 | 8/8 |
| D5 (18 mutants) | 24/24 | 24/24 |
| D6 3.A (16 mutants) | 9/9 | 9/9 |
| D6 threshold (11 mutants) | 17/17 | 17/17 |
| Phase 3 | 17/17 | 17/17 |
| Phase 4 | 63/63 | 63/63 |
| **D6.1 (new, 7 mutants)** | — | **13/13** |

**D3.25 unchanged at 13/13.** Its flush gate replaces the helper with a spy and
never executes the body, so the repair cannot reach it — the one-flush-per-trial
cadence invariant is untouched by construction, and `G-CADENCE` re-asserts it
independently by AST (exactly one call site in each of
`_build_test_result_from_pw` and `_build_test_result_from_miner`).

**D5's artifact compression ban is untouched** — D5 never scans
`window_optimizer_integration_final.py`; its ban is scoped to the artifact
writer (`miner/assembly_shard_worker.py:308`, `np.savez(fh, **payload)`), which
is unmodified and still emits `ZIP_STORED`.

---

## 9. Scope discipline

**Not touched:** PWC/ZMQ ingress · the D3.25 four-map contract · `TestResult`
shape · the D6 threshold/provenance/residue work · the certified-artifact NPZ
contract (D5 §6.7.A). `serial_reference` remains default; `process_sharded`
unpromoted.

**Exploratory ROCm launch test: NOT RUN.** Beta permitted one as optional and
explicitly non-certifying. Skipped to hold D6.1's scope — the repair is CPU-only
and the test cannot satisfy any Phase 6.0 acceptance criterion. It remains
available ahead of Phase 6.0.

---

## 10. Fallback parity

`fallback parity: code=[not re-checked this session], env=[not re-checked this
session]` — Zeus runs one OS at a time and `.127` was not booted. D6.1 adds no
new dependency (`os`, `numpy`, `glob`, `re`, `zipfile` are all stdlib/existing).

---

## 12. Pre-commit hygiene verification (Beta)

**Question:** does every test that exercises the live `_flush_npz_incremental`
set `PRNG_CHECKPOINT_ROOT` to a per-test temporary directory and clean it up,
so the suite never accumulates `.s172_checkpoint/<run_id>/` in the repository
or in a shared production checkpoint root?

**Answer: it did NOT. Isolation was missing in D3.25 and has been added.**

Five suites reference the helper or its callers. Audited individually:

| Suite | Reaches the live helper? | Isolation |
|---|---|---|
| `test_s172_d6_1_flush_durability.py` | yes, throughout | **already present** — `_in_tmp()` sets root + run id per gate and restores both |
| `test_s172_phase5_d6_production_adapter.py` | yes, in `g_flush_cadence` | **already present** — added with the D6.1 cadence re-anchor |
| `test_s172_phase5_d3_25_candidate_ingress.py` | **YES — `ingest()` drives the real adapter, no spy** | **ADDED this pass** — `_isolated_checkpoint_root()`, per invocation |
| `test_s172_phase5_d6_threshold_path.py` | **no** — its G13 leg raises on the provenance wall before the flush and asserts the accumulator stays empty | not needed; file left untouched (Beta-approved work) |
| `test_s172_phase4_coordinator.py` | **no** — its three references are gate-22 registration comments, not calls | not needed |

**Sentinel audit.** Each suite was run with `PRNG_CHECKPOINT_ROOT` pointed at an
empty sentinel directory standing in for a shared production root. A suite that
fails to isolate writes into the sentinel; one that isolates overrides it.

```
suite                                        rc   sentinel entries   repo .s172_checkpoint   gates
d3_25_candidate_ingress                      0    0                  no                      13/13
d6_threshold_path                            0    0                  no                      17/17
d6_production_adapter                        0    0                  no                       9/9
d6_1_flush_durability                        0    0                  no                      15/15
phase4_coordinator                           0    0                  no                      63/63
```

**Negative control** — the audit has teeth. The pre-isolation D3.25 (from
`HEAD`) run against the same sentinel:

```
pre-isolation D3.25 -> sentinel entries: 4
  <sentinel>/.s172_checkpoint/
  <sentinel>/.s172_checkpoint/zeus-ubuntu-vm-74765-1785429666/
```

So the leak was real and the fix is what closes it, rather than the audit
passing vacuously. This also confirms §0: the leak exists *because* the flush
now genuinely writes.

**Tree verification (post-run):**

- **No stray `.s172_checkpoint/` anywhere in the repo tree** — `find . -name
  .s172_checkpoint` returns nothing.
- **Finalizer-owned aliases:** *absent* from this working tree. No
  `.s172_accumulator/` exists at any depth, so no generation has been published
  here and the run-root aliases have never been created — there was nothing for
  the snapshot to clobber. The nine files matching those two names are
  **regular files inside archived `logs/S174_*_bundle/local/`** run artifacts
  from 2026-05, unrelated to `_bootstrap_root_aliases`. The symlink property
  itself is proven live by `G-NO-ALIAS-COLLISION`, which bootstraps both
  aliases, flushes twice, and re-asserts `os.path.islink` on each plus a clean
  `_bootstrap_root_aliases` re-run.

One stale artifact was found and removed during this pass: a pre-correction
snapshot pair at `.s172_checkpoint/` (flat, no `<run_id>/`, no identity block)
left by an earlier D6.1 run before run isolation existed. The current loader
classifies exactly that shape as `unrecoverable` and refuses it rather than
misinterpreting it — gated in `G-TRANSACTION-IDENTITY`.

---

## 11. Review gate

**STOPPED for Team Alpha review. Not committed, not pushed.**

Team Beta must rule on the §2 scope change — **"repair the flush in place"** →
**"relocate the checkpoint to its own namespace and defer the clear"** — and on
whether the deferred S166 clear (§2a) is tracked as a Phase-7 soak blocker.

After Alpha + Beta pass: Michael commits and dual-pushes, then Phase 6.0
(paired CUDA/ROCm smoke) runs against the post-D6.1 codebase.
