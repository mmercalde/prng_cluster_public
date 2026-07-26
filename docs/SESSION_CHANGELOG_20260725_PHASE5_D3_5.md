# SESSION_CHANGELOG_20260725_S179 — S172 Phase 5 D3.5 (shared run finalizer)

Spec: `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3_5.md` (REV3.1, Beta-approved).
Frozen against HEAD `70cd6f0`. **No commit, no push, no WATCHER** (per instruction).

## Correction round (Team Beta ruling on harness expiry) — RESOLVED

The §13 stop condition reported below was ruled on: the "must not modify
D3/D3.0 tests" clause **yields**. D3.5 is the planned expiry event for the
inline-writer migration seam. Neither the closure nor a snapshot of its source is
retained anywhere.

| Item | Result |
|---|---|
| (1) D3 harness — inline half of C8 retired | **10/10** |
| (2) D3.0 harness — 7 inline consumers retired/reframed | **10/10** |
| (3) F18 replaced with the binding strengthened form + added to F26 | **51/51, 27 mutants red** |
| (4) Gate-22 whitelist annotated with the retirement reason | Phase 4 **63/63** |

**Post-migration non-regression, full D3.5 tree:** D3.5 51/51 · D3.25 13/13 ·
**D3 10/10** · **D3.0 10/10** · D2 7/7 · D1.1 18/18 · D1.0 8/8 · D0 12/12 ·
Phase 4 63/63 · Phase 3 17/17.

**Byte-identity vs `70cd6f0`** (prohibited to modify — all three confirmed
IDENTICAL, and absent from `git status`):

```
utils/canonical_arrays.py        78e0e20463a273c229785657e69a94346cfb1fe79150fbdac065e3ce9697be80
utils/prng_encoding.py           a4889e891fd180f78ad6af36daeb0c10d818cca16b343be7d022d03128d62f03
convert_survivors_to_binary.py   565b088ba8e400483938d2cddcaec867846e6d2eaaf9e6c3541eac20fe55a600
```

**Historical attribution isolation (proof a), reproduced.** A symlink farm of the
live D3.5 tree with `window_optimizer_integration_final.py` **and both original
harnesses** restored from `70cd6f0` runs **D3 10/10 and D3.0 10/10** — confirming
the only cause of the earlier red was the removed closure, not any D3.5 change to
the modules those gates certify.

**Prohibitions observed.** AST scan of both harnesses and the integration file
finds **no executable reference** to `_survivors_to_arrays`,
`_inline_survivors_to_arrays`, `run_inline_writer` or `load_inline_writer` — the
remaining textual mentions are prose in the retirement notes. No closure retained
as dead code, no embedded source, no compatibility module, no skipped checks, no
weakened standalone-writer assertion.

### (3) F18 — the binding strengthened form

Mutates BOTH anchors: `seed_end_exclusive` computed in fixed-width unsigned
arithmetic AND removal of the interval-ordering check that would otherwise reject
the wrapped result incidentally. Boundary case `seed_start = 2**32 - 10`,
`seed_count = 100`, wrapped end `90`.

The gate demonstrates the defective implementation **certifies** a generation
whose sidecar claims the sweep `[4294967286, 90)` — an inverted interval, so the
artifact is empty and the false claim is then inherited by the next run as a
chain-authenticated parent (asserted).

Attribution is enforced by construction: the candidate list is **empty** (no
candidate validation, no L2, no ordering, no identity wall can contribute), the
metadata is otherwise well-formed and the tree clean (no parameter/sidecar
rejection), and every publication step is untouched (no unrelated publication
failure). Production's rejection message must name the **unwrapped** end
`4294967386` — a value a fixed-width implementation never computes. The registered
mutant's red signature is exactly:

```
killed by F18: AssertionError: F18: expected CoverageValidationError, nothing raised
```

Permanently registered in F26's mutation set (now 27 mutants, all red).

---

## Status: implementation + gates COMPLETE

| Item | Result |
|---|---|
| (A) `utils/run_finalizer.py` | NEW, complete |
| (B) integration replacement + swallow-wrapper fix + legacy dedup retirement | complete |
| (C) gates F1-F51 + F26 mutation set | **51/51 green, 26/26 mutants red** |
| Blocking non-regression | 7 of 9 green; **D3 and D3.0 red** — see §13 below |

## Pre-edit baseline (captured green at `70cd6f0` BEFORE any edit)

D3.25 13/13 · D3 10/10 · D3.0 10/10 · D2 7/7 · D1.1 18/18 · D1.0 8/8 · D0 12/12 ·
Phase 4 63/63 · Phase 3 17/17.

## Post-edit non-regression

Green: D3.25 13/13, D2 7/7, D1.1 18/18, D1.0 8/8, D0 12/12, Phase 4 63/63,
Phase 3 17/17. Red: **D3 and D3.0**, both at import time, both for the same
single cause (below).

## §13 STOP CONDITION 1 — the inline block cannot be replaced without touching
## the must-not-modify list

`tests/test_s172_phase5_d3_columnizer.py:318` and
`tests/test_s172_phase5_d3_0_encoding_contract.py:258` both call
`load_inline_writer()` at module import, which extracts the inline
`_survivors_to_arrays` closure from `window_optimizer_integration_final.py` by
AST line-range and drives it as a live parity oracle. D3.5 §0/§2 requires that
inline block to be REPLACED, which deletes that closure, so both harnesses now
abort before running a single check:

```
AssertionError: _survivors_to_arrays not found in window_optimizer_integration_final.py
```

The three spec clauses cannot all hold simultaneously:

* §0/§2 — replace the inline run-finalization block (deletes `_survivors_to_arrays`);
* §2 — **must NOT modify** D3/D3.25 tests;
* §12 — D3 10/10 and D3.0 10/10 are **blocking** non-regression.

D3's own module docstring anticipated this: *"the existing `_survivors_to_arrays`
closure ... stay in place and in use **until D3.5**"*. The harnesses were not
updated to match.

**Cause isolated, non-destructively.** A symlink farm of the current working tree
with ONLY `window_optimizer_integration_final.py` restored from `70cd6f0` runs
**D3 10/10 and D3.0 10/10 green**. So neither `utils/canonical_arrays.py` nor
`convert_survivors_to_binary.py` — the modules those two gates actually certify —
is affected by D3.5; the sole cause is the retired closure.

**Not remediated here, deliberately.** Editing either harness violates §2;
retaining a dead `_survivors_to_arrays` purely to satisfy a gate would keep the
retired legacy columnizer alive against §10. Team Beta ruling requested on which
of the three clauses yields.

## (A) `utils/run_finalizer.py` — NEW

One public entry point, `finalize_run()`, on the frozen §8 signature, returning
the frozen `RunArtifactResult`. Binding pipeline order implemented exactly:
validate EVERY raw candidate through D3 → coverage → run-identity wall → L2
(record domain) → `records_to_arrays(winners)` → prior load/validate → L3 (ARRAY
domain) → global seed-ascending sort → `validate_array_bundle` → publication.

* **Reuses, never reimplements** — `records_to_arrays`, `validate_array_bundle`,
  `BASE_PRNG_FAMILIES`, `CANONICAL_ARRAY_CONTRACT`, `encode/decode_*`.
* **No subprocess anywhere in the module.** `repository_commit` /
  `repository_tree_clean` arrive as arguments; the git query lives in the caller
  (`_repository_state`), which is why the frozen signature takes them.
* Errors derive from `RunFinalizerError(RuntimeError)`, **not** `ValueError` —
  load-bearing, so a fail-closed rejection can never be mistaken for a fallback
  candidate by a legacy `except ValueError` arm.
* Publication follows §7.2 steps 1-15 verbatim, including [D3] hashing the
  **reopened stored** sidecar bytes, [D1] the hash-bound directory name
  `<generation_id>--<sidecar_sha256>`, [C4] first-generation alias bootstrap
  before the commit, and [D4] `PublicationDurabilityError` for a post-swap
  fsync failure.
* Filesystem primitives are routed through named module-level seams (`_mkdir`,
  `_write_npz`, `_fsync_file`, `_fsync_dir`, `_write_and_fsync_bytes`,
  `_atomic_rename`, `_replace_symlink`) so F30/F32/F51 can instrument the
  publication order rather than assert it from source alone.

**No coverage database was read or written.** `utils/run_finalizer.py` contains
no reference to `prng_analysis.db`, `exhaustive_progress`, sqlite, or any
coverage table; the only invariant it proves is the LOCAL one of §6.

## (B) `window_optimizer_integration_final.py`

* The inline S145-R1 NPZ-accumulator block (`70cd6f0:1799-2020`) is replaced by
  a single `finalize_run` call.
* **§11 [B4] fix:** the finalizer call sits **outside** every `try/except`. The
  broad swallow at `70cd6f0:2004` and its `convert_survivors_to_binary.py`
  subprocess fallback are **deleted**, as is the tagged-ValueError re-raise arm.
  Only non-canonical diagnostics remain inside a swallowing wrapper.
* **§10:** `deduplicate_survivors` (`70cd6f0:1684-1700`) is **removed**, not
  bypassed. `bidirectional_survivors.json` is now a post-success summary of the
  certified generation and no longer describes itself as the canonical Steps 2-6
  input; the canonical NPZ is authoritative.
* New helper `_repository_state()` returns `(commit, tree_clean)` truthfully; a
  dirty tree makes the finalizer refuse to certify (§7.3 / F37).

## (C) `tests/test_s172_phase5_d3_5_finalizer.py` — NEW, 51/51

Independent hand-transcribed oracles throughout (22 array names/order/dtypes, 23
sidecar keys, layout names, identity ids, every tie outcome). No production
constant is asserted against itself.

**F26 mutation proof: 26 mutants, all RED, each attributable.** Two of them
mutate the integration file (swallowed integration exception → F31; score-only
legacy dedup left active → F36); the rest mutate `utils/run_finalizer.py`.

Two harness-side corrections were made during bring-up and are worth flagging:

1. **F15/F25 source gates** were initially blunt substring searches and red on
   the finalizer's own docstrings, which deliberately NAME the prohibited things
   ("populates neither `MinerTrialAssembly.binary_npz_path`...", "the
   `convert_survivors_to_binary.py` subprocess..."). They now assert over
   **executable references** (non-docstring string constants, `ast.Name` /
   `ast.Attribute` / import references) instead of prose.
2. **F18** — the spec asks to mutate the addition to fixed-width unsigned and
   prove the wrap is rejected. Finding: with the interval-ordering check intact,
   a uint32 wrap is **always** rejected incidentally, because the wrapped end is
   necessarily below `start`. To prove the Python-int arithmetic is genuinely
   load-bearing, F18 also builds a second mutant with the ordering check
   removed; it then **certifies a generation carrying the false claim
   `[4294967286, 90)`** (no seed can satisfy an inverted interval, so the leak is
   an empty generation with false coverage metadata that the next run inherits as
   a certified parent). Production rejects the identical call.

## Files

| File | Change |
|---|---|
| `utils/run_finalizer.py` | NEW |
| `tests/test_s172_phase5_d3_5_finalizer.py` | NEW |
| `window_optimizer_integration_final.py` | inline block replaced; swallow wrapper fixed; legacy dedup removed |
| `tests/test_s172_phase4_coordinator.py` | gate-22 registration only (two new paths) |

Untouched, as required: `utils/canonical_arrays.py`, `utils/canonical_records.py`,
D3/D3.25 tests, `persistent_worker_coordinator.py`, `zmq_sqlite_coordinator.py`,
`miner/*`, `prng_analysis.db`, WATCHER.

Fallback parity: not re-checked this session (no phase boundary reached — D3.5 is
blocked on the §13 ruling above).
