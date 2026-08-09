# CLAUDE CODE INSTRUCTIONS — S145 SEED-DOMAIN / CURSOR AMENDMENT, REVISION 1

**Host:** VM101, repo `~/distributed_prng_analysis`. The amendment is **uncommitted in the working
tree** at base `4dd5535`. `source ~/venvs/torch/bin/activate` before every test. Long suites:
`python3 -u <suite> | tee /tmp/<name>.log` — never `| tail`.

**Authority:** Team Beta ruling *"S145 SEED-DOMAIN / COVERAGE-CURSOR AMENDMENT"* (2026-08-08).
**RETURN FOR NARROW REVISION — architecture accepted, commit not authorized.** Three blockers.

**APPROVED AND CLOSED — do not reopen, do not re-argue, do not "improve":** the `[0, 2^32)` shared
terminus · legacy-tracker deauthorization (rows retained, zero certified progress) · the first-gap
cursor · the explicit `COMPLETE` / `next_seed_start = None` state · the append-only ledger design
(Beta: *"a strong solution to the exact clobber class that damaged the old tracker"* — content-derived
identity, bare INSERT, triggers, `recursive_triggers=ON`) · **keeping the publication producer
wired** · raising when a post-publication ledger write fails · WATCHER failing closed on cursor
lookup failure · `dataset_sha256` as provenance, **not** a v1 partition key.

**Beta §12 — make ONLY these changes.** Do not touch: staging-capacity implementation · S172 gates ·
sieve mathematics · Optuna search policy · legacy rows · dataset authority · new telemetry ·
Phase-7 work.

**Hard constraints:** no commit, no push, **no pipeline launch, no fleet launch, no port 5700
bind.** Gate 12 is HELD — Beta: *"Do not run Gate 12 on the production fleet under the current
patch."* If a fix appears to need a file outside the amendment's change set, STOP and report.

**Base verification:** working tree still carries the amendment (4 modified + 4 untracked:
`utils/seed_coverage_ledger.py`, `tests/test_seed_domain_cursor_amendment.py`, and the two docs);
`python3 -u tests/test_seed_domain_cursor_amendment.py` → **29/29**. Untracked runtime residue
(WAL sidecars, `*.stale_*`) is expected and not a stop condition.

---

## BLOCKER A — THE CERTIFICATION DOOR IS BYPASSABLE (Beta §1)

**Beta's most important finding, and our own suite demonstrates the defect.**

`CoverageLedger.record_publication()` is documented as accepting the canonical `RunArtifactResult`
but **accepts any object exposing an `artifact_sha256` attribute**, while every other coverage
field is supplied independently by the caller. Two proofs from the submitted suite itself:

1. the "successful publication" arm uses **`_FakeArtifact`**, not a real `RunArtifactResult`, and
   expects it to certify coverage;
2. the mutation gate calls the **public** `record_certified_interval()`, fabricates a
   never-published billion-seed interval, and **proves the fabricated row advances the
   authoritative cursor.**

That second case is **not a hypothetical mutant — it is an existing production API bypass**, and it
violates the governing law: *starting a run is not coverage; receiving results is not coverage; a
provisional database write is not coverage; canonical publication is the evidence wall.*

**Required correction — ONE production certification door:**

```
RunArtifactResult → record_publication() → certified_coverage
```

1. **Derive publication identity FROM the artifact.** D3.5's frozen `RunArtifactResult` already
   carries `run_id`, `prng_base`, `skip_modes_executed`, `seed_start`, `seed_count`,
   `seed_end_exclusive`, `artifact_sha256`, `generation_id`, `repository_commit`, and
   publication/provenance identity — and it is constructed **only after** the canonical publication
   commit succeeds. **Do not independently accept caller versions of fields the witness already
   possesses.** Target shape:

   ```python
   record_publication(artifact: RunArtifactResult, *, dataset_sha256, study_identity=None)
   ```

   At minimum, **reject an object that is not the canonical result type or cannot satisfy the
   complete frozen result contract.**
2. **Restrict the raw writer** to an internal implementation seam — `_record_certified_interval(...)`
   — and remove it from the public/exported coverage API. Beta: *"Python underscore privacy is not
   a security boundary; that is not the point. This is an authority boundary."*

**Required gates:** `_FakeArtifact("f"*64)` **refused** as a witness · a real `RunArtifactResult`
**succeeds** · caller-supplied values **cannot substitute** a different `run_id`, range, PRNG
identity, mode set, commit or generation from what the artifact says · a **repo/source scan**
proving the only production path creating certified coverage is `record_publication` · the previous
"mutant raw writer" becomes a **test-only/internal bypass**, not a supported public authority.

## BLOCKER B — CANONICAL COVERAGE SCOPE (Beta §2, answering Alpha's open question)

Beta ruled on the partitioning question Alpha raised and **rejected both options Alpha framed**:
not raw `prng_type` alone, and **not a simple `(prng_type, skip_mode)` scalar key either.**

**The correct coverage identity is `prng_base` + the required executed-mode set**, because Step 1
can execute *constant only* or *constant + variable*, and those are **distinct searches** —
`test_both_modes` runs the base PRNG for constant skip and the hybrid variant for variable skip. A
range searched only under constant skip cannot certify that the variable-skip search happened.

**Required law — store the canonical identity already present in D3.5:**

```
prng_base
skip_modes_executed      # from run configuration; authoritative even if a mode produced zero survivors
```

**Cursor inclusion uses SET CONTAINMENT:**

```
record.prng_base == requested.prng_base
AND requested_modes ⊆ record.skip_modes_executed
```

| certified record | requested | counts? |
|---|---|---|
| `{constant}` | `{constant}` | **YES** |
| `{constant}` | `{constant, variable}` | **NO** |
| `{constant, variable}` | `{constant}` | **YES** |
| `{constant, variable}` | `{constant, variable}` | **YES** |
| `{variable}` | `{constant}` | **NO** |

Beta: this is better than treating `java_lcg_hybrid` as an unrelated coverage namespace. **Use the
finalizer's canonical run identity — base family plus executed mode set — rather than inventing
another one in the ledger.**

**The `prng_type` / `prng_base` inconsistency is NO LONGER BACKLOG.** Alpha filed it as such; Beta
overruled: *"It is not backlog anymore once this table becomes authoritative."* Alpha's own
observation is the reason — WATCHER can query `java_lcg_hybrid` while the publication hook records
`java_lcg`, splitting one logical search into incompatible namespaces. **Resolve it here** by
canonicalizing both sides to `prng_base = java_lcg` with the appropriate `required_modes`.
**Do not preserve the ambiguity into a brand-new authority table.**

## BLOCKER C — THE PRE-DISPATCH WALL MISSES A THIRD EXECUTION PATH (Beta §4)

The report claimed the wall covers *"both entry points — WATCHER and the direct CLI."* **Too
broad.** `window_optimizer.py` has a third direct execution mode: **`run_with_config()`**, which
iterates ranges and invokes `run_bidirectional_test()` directly. Current topology:

```
WATCHER              → wall ✅
direct Bayesian      → wall ✅
--config-file / run_with_config → sieve execution, NO S145 wall ❌
```

The finalizer still catches an invalid range later, but **§7 exists precisely to prevent GPU work
before discovering the interval was illegal.**

**Required correction — preflight the ENTIRE requested plan, not one chunk.** Because that function
runs multiple iterations, validating only the first is insufficient: at `max_seeds = 2^30` with
`iterations = 5`, **the fifth interval escapes `[0, 2^32)`** — the command must be rejected before
the first GPU iteration rather than after ~4 billion seeds.

```python
for iteration in range(iterations):
    start = iteration * max_seeds
    assert_seed_domain_preflight(start, max_seeds)
```

…before initializing or dispatching any sieve work. Equivalent total-plan arithmetic is acceptable
**provided its gate proves every generated interval is covered.**

**Required gates:** config-mode exact-bound success · config-mode final-seed success · config-mode
`2^30 × 4` success · config-mode `2^30 × 5` **fail with zero dispatch** · structural/behavioural
proof that **no `run_bidirectional_test` executes before plan validation.**

### C.2 — Remove the pre-wall type coercions (Beta §5, rides with C)

The wall deliberately rejects `bool`, `float`, `str` and other non-`int` types (`_require_int`).
But the new WATCHER/direct wrappers **coerce with `int(...)` before invoking the wall**, defeating
that contract:

```
wall(True) → reject       BUT    int(True) → 1 → wall(1) → accept
                                 int(1.9)  → 1 → accepted
```

**Pass the value to the wall in its authoritative form.** Normal argparse operation already yields
integers, so this should not disturb normal CLI use. The suite currently checks the strict helper
directly but never drives malformed values **through the real wrappers**.

**Required gates:** wrapper-level negative arms for at least `True`, `1.5`, `"0"` — or whatever
malformed values are genuinely reachable through each programmatic entry point.

---

## VERIFICATION BEFORE RESUBMISSION

- `tests/test_seed_domain_cursor_amendment.py`: all prior gates green **plus** the new
  publication-witness, canonical-scope, config-mode-plan and wrapper-type arms;
- **red-first evidence for all three blockers** against the current submitted patch;
- staging suites **unaffected and re-run sequentially** (Beta §10: keep them sequential for
  certification until the `G-VAL-6` race gets its own amendment) — 50/50, 24/24, 6/6, 60/60;
- phase-4 evaluated in the clean/committed model; state which method;
- **zero diff against the staging-capacity implementation and its suites** — confirm
  programmatically, as the last report did;
- no gate-12 production run; no Phase-7 soak.

## REPORT

`docs/CLAUDE_CODE_REPORT_SEED_DOMAIN_CURSOR_R1.md`:

1. Per-blocker implementation notes with `file:line`.
2. The `record_publication` signature as built, and **exactly which fields are derived from the
   artifact versus still accepted from the caller** (and why, for any that remain).
3. How the raw writer is restricted, and how the repo/source scan gate proves no production
   certification bypass exists.
4. The canonical coverage identity as stored, the containment predicate, and **how the
   `prng_type`/`prng_base` mismatch was canonicalized** on both the WATCHER-query and
   publication-hook sides.
5. The whole-plan preflight for `run_with_config`, and the proof no dispatch precedes it.
6. Red-first and mutation evidence per new arm.
7. Files changed — expect the same set plus test additions. Anything else justified.
8. Any disagreement with this brief **reported, not worked around.**
