# CLAUDE_CODE_INSTRUCTIONS_S172_D6_1_FLUSH_DURABILITY.md — REV1

**S172 — `D6.1: incremental NPZ atomic flush and durability repair.`**

Team Beta mandated this as a standalone high-priority defect: non-blocking for the
committed D6 adapter, but **blocking** extended Phase 6 benchmark runs, the Phase 7
multi-trial soak, and any WATCHER-controlled long-running execution. Beta's framing
is the operative one: **incremental durability does not currently exist** — if a run
terminates before finalization, the intended NPZ checkpoint is absent. That affects
crash recovery and soak safety.

**Base:** `9e0bfe9` (D6 closed, release-grade generation certified at `b08c2c5`).
Claude Code on VM 101 as `michael`, venv `~/venvs/torch`. Implement and iterate;
do **NOT** commit, push, or run WATCHER. STOP at the gate for Team Alpha review.

---

## 0. What is actually broken — read this before writing anything

Alpha read the helper at source (`window_optimizer_integration_final.py:243-319`).
It is **four defects, not one**. Verify each yourself before changing it.

**D1 — the `.npz` suffix bug (the known one, present twice).**
```python
_tmp = _ACCUM_NPZ + ".flush.tmp"                    # "...all.npz.flush.tmp"
_np_flush.savez_compressed(_tmp, seeds=..., score=...)   # writes "...flush.tmp.npz"
_os_flush.replace(_tmp, _ACCUM_NPZ)                 # FileNotFoundError
```
NumPy appends `.npz` when the filename lacks it, so `os.replace` targets a name that
was never created. Identical bug on the binary write (`_tmp_bin`, `:300-305`).

**D2 — the broad `except Exception` swallows everything** (`:318`). Every failure
becomes a non-fatal warning. That is why this has never been noticed: the helper has
*always* failed and always printed a warning nobody treated as fatal.

**D3 — the list-clear is inside the same `try`, after both replaces** (`:307-310`).
Today that is *accidentally* protective — the exception fires at the first
`os.replace`, so `accumulator["bidirectional"] = []` never runs and no candidates are
lost. **Fixing D1 naively makes this ordering load-bearing.** Also note
`_flush_last_count = 0` is reset immediately before the clear; if anything throws
between the first successful replace and the clear, on-disk and in-memory state
diverge (survivable via merge-by-seed dedup, but it must be gated, not left to luck).

**D4 — the S166 comment asserts a guarantee that has never held.** "*data is safe in
NPZ*" is the stated justification for clearing the in-memory list. Since the write
has always failed, that claim has never once been true. Correct the comment as part
of the repair — a load-bearing comment documenting a false guarantee is itself a
defect.

**Cadence context:** `_FLUSH_EVERY` (env `PRNG_FLUSH_EVERY`, default 10) and
`_flush_last_count` gate entry (`:259-260`). The D3.25 gate currently pins **exactly
one `_flush_npz_incremental` call per trial** — that cadence invariant must survive
unchanged.

---

## 1. Implementation calls (Alpha-decided; Beta delegated implementation choice)

**Call A — two writes: sequential-atomic, not falsely "jointly atomic."**
Write **both** temp files first, then perform the two `os.replace` calls
back-to-back. Do **not** claim joint atomicity — true joint atomicity across two
files needs a directory swap or a manifest, which is out of scope. Instead:
- gate that a crash **between** the two replaces leaves a *detectably* inconsistent
  pair, and that the **next flush repairs it** via the existing merge-by-seed dedup;
- document this explicitly as sequential-atomic-with-self-repair, not atomic.
*Rationale: an honest, provable property beats an atomicity claim the code cannot
keep — and this project has already been bitten once by a comment asserting a
guarantee that did not exist (D4 above).*

**Call B — keep `savez_compressed`.**
This is a **checkpoint**, not a certified artifact; compression is the right tradeoff
for a file rewritten every N survivors. But **document loudly** why it differs from
D5's §6.7.A ban on compressed NPZ, which applies to *artifacts* and is enforced by
D5's M6a mutant (reds on `compress_type=8`). Add a comment at the call site naming
the distinction so nobody later "harmonizes" the two and reds a D5 gate.
*Rationale: the two contracts are genuinely different; the hazard is conflation, and
the cure is documentation, not uniformity.*

**Suffix handling (mechanism free, property gated).** Beta noted an open file handle
to `np.savez()` bypasses filename rewriting; naming the temp with `.npz` already
present (e.g. `...all.npz.flush-<pid>.tmp.npz`) also works. **Choose either after
reading the call site, state why, and gate the *property*** — "the temp target cannot
trigger NumPy's implicit `.npz` suffix behaviour" — not the mechanism.

---

## 2. Required behaviour after the repair

1. The temp target **cannot** trigger NumPy's implicit `.npz` suffix rewrite.
2. Each write is atomic: a complete temp file is `os.replace`d onto the final name.
3. **The in-memory candidate list clears ONLY after both replaces have succeeded.**
4. **A failed write retains ALL in-memory candidates** — no data loss on any failure
   path.
5. Temporary files are removed on **every** path, success and failure.
6. Repeated flushes preserve **exact cumulative counts** (merge-by-seed dedup,
   highest score wins, prior-NPZ merge intact).
7. **Crash/restart behaviour is explicit**: state and gate what a restart sees after
   (a) a crash before any replace, (b) between the two replaces, (c) after both.
8. The **cadence gate is updated to pin successful flush behaviour** rather than the
   current failed attempt — and the D3.25 one-call-per-trial invariant still holds.
9. **Narrow the exception handling (D2).** A genuine write failure must be *visible*,
   not silently swallowed. Keep the helper non-fatal to the trial if that is the
   established contract — but distinguish "expected, recoverable" from "unexpected"
   and make the latter loud. State the chosen contract explicitly.
10. Correct the S166 comment (D4) so it describes the guarantee that now actually
    holds.

## 3. Gates — `tests/test_s172_d6_1_flush_durability.py`

Each gate must FAIL on wrong behaviour; oracles hand-transcribed. Cover:

- **G-SUFFIX:** the temp path cannot be suffix-rewritten by NumPy (assert the file
  NumPy actually creates equals the path `os.replace` consumes).
- **G-ATOMIC-ACCUM / G-ATOMIC-BINARY:** each final NPZ is either the complete prior
  content or the complete new content — never partial.
- **G-CLEAR-AFTER:** the in-memory list clears **only** after both replaces succeed.
- **G-RETAIN-ON-FAIL:** inject a failure at each of — before first write, between
  write and replace, between the two replaces, after both — and assert **zero
  candidate loss** in every case.
- **G-NO-TEMP-LEAK:** no temp file remains after success or any failure path.
- **G-CUMULATIVE:** repeated flushes preserve exact cumulative counts; merge-by-seed
  dedup and prior-NPZ merge behave as before.
- **G-CRASH-RESTART:** the three crash points above, each asserting what a restart
  observes and that the next flush self-repairs an inconsistent pair.
- **G-CADENCE:** `_FLUSH_EVERY` / `_flush_last_count` entry gating unchanged; the
  **D3.25 one-flush-per-trial invariant holds**; the gate now pins **successful**
  flush, replacing the assertion that pinned the failed attempt.
- **G-VISIBLE-FAILURE:** an unexpected write failure is surfaced, not swallowed.
- **G-COMPRESSION-CONTRACT:** the checkpoint may be compressed **and** D5's artifact
  ban is untouched — assert D5's own gate still reds on a compressed *artifact*
  (i.e. prove the two contracts remain separate).

**Mutants (four-part kill rule: applies once; mutated path executes; reaches the
crediting assertion; fails from the injected defect):**
1. restore the un-suffixed temp name → G-SUFFIX reds.
2. move the list-clear before the replaces → G-CLEAR-AFTER / G-RETAIN-ON-FAIL red.
3. clear the list on a failed write → G-RETAIN-ON-FAIL reds.
4. leave the temp file behind → G-NO-TEMP-LEAK reds.
5. re-broaden the exception handler to swallow everything → G-VISIBLE-FAILURE reds.
6. drop the prior-NPZ merge → G-CUMULATIVE reds.

## 4. Scope — do NOT touch

- The **D3.25 flush-cadence invariant** (one call per trial) — preserve exactly.
- PWC/ZMQ ingress, the D3.25 four-map contract, `TestResult` shape.
- The D6 threshold path, provenance enforcement, residue authority — all just
  Beta-approved and committed.
- The certified-artifact NPZ contract (D5 §6.7.A) — the checkpoint is a separate
  contract; do not converge them.
- `serial_reference` remains default; `process_sharded` unpromoted.

## 5. Non-regression

Capture green at `9e0bfe9` **before any edit**, in the venv: D1.1 18/18 · D4 8/8 ·
D5 24/24 (18 mutants) · D6 3.A 9/9 (16 mutants) · D6 threshold/provenance 17/17
(11 mutants) · Phase 4 63/63 · Phase 3 17/17 · D0 12/12 · D1.0 8/8 · D2 7/7 ·
D3.0 10/10 · D3 10/10 · D3.25 13/13 · D3.5 60/60. After the repair: all still green
plus the new D6.1 gate. **D3.25 must stay 13/13** — it owns the cadence invariant.

## 6. Optional, explicitly exploratory

Beta permits a **non-certifying** preliminary ROCm launch test during D6.1
development, to catch basic environment failures ahead of Phase 6.0. If run, label
it **exploratory** in all output and the report; it **cannot** satisfy any Phase 6.0
acceptance criterion. Do not let it expand D6.1's scope.

## 7. Report

Diff + gate results + the pre-edit non-regression baseline. State explicitly: the
suffix mechanism chosen and why; the exception-handling contract chosen; the three
crash-point behaviours; confirmation the D3.25 cadence invariant is unchanged and
13/13; and confirmation D5's artifact compression ban is untouched. Then STOP for
Team Alpha review.

**Do not commit.** After Alpha + Beta pass: Michael commits, then Phase 6.0 (paired
CUDA/ROCm smoke) runs against the post-D6.1 codebase.
