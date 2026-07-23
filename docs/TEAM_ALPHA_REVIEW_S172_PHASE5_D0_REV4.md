# TEAM_ALPHA_REVIEW_S172_PHASE5_D0_REV4.md

**S172 RANGE-MINER — Phase 5 Deliverable D0, correction round 4 (REV4)**
**Resubmission for Team Beta binding verification**

**Reviewer:** Team Alpha (orchestrator)
**Prior verdict:** Beta REJECT of REV3 — one code blocker (`window_size`/`offset`
fabrication in `run_trial_miner`) + a packaging defect (Gate 20 hang, missing
`prng_registry.py`). Everything else in REV3 was ACCEPTED.
**This round:** the two remaining fabricated fields fixed, a third-order coercion
site (that Beta's fix would otherwise have converted into a raw `TypeError`) fixed,
Gate B4 corrected (Beta's vacuity catch), Gate B5 added, and a genuinely standalone
archive built and proven. Team Alpha audited the **entire** `run_trial_miner` +
`serve_trial` context path against source — not just the named fields.
**Tree base:** HEAD **`833507e`**.
**Gates:** D0 **12/12** · Phase 4 **63/63** (clean tree) · Phase 3 **17/17**.
Both suites also verified green **from the extracted archive** (not the working tree).
**Archive:** `s172_phase5_d0_rev4.tar.gz` (27 members)
**SHA-256:** `d81c4302369ae7ed420e17f730e2915a37d97219bca874ceb84149c9a11c7460`

---

## 1. Accepted in prior rounds (unchanged — for context)

Absent-`trial_context` publish guard; compare-and-insert conflict + two-process
concurrency probe; `skip_min`/`skip_max` omission fix (REV3); the 13 Phase-4 gate
context-seeding adaptations (Beta-approved); durable table; gate-22 whitelist;
`prng_base`/`WindowConfig` direct-access fixes.

## 2. REV4 blocker resolution — `window_size`/`offset` fabrication

**Beta's finding:** `run_trial_miner` declared `kwargs.get("window_size", 1)` /
`kwargs.get("offset", 0)`, so an omitting caller had the omission fabricated into a
valid-looking `1`/`0` before `build_trial_context_from_serve`'s guard.

**Fix (same shape as the accepted skip fix):**
- Signature: `window_size: Optional[int] = None, offset: Optional[int] = None`.
- Context build: `"window_size": window_size` / `"offset": offset` — the old
  `kwargs.get(..., 1/0)` lines are removed. An omitted value flows through as `None`
  and hits the guard, which raises `MinerMetadataError`.
- Real `_use_miner` call site already passes `config.window_size`/`config.offset`
  (direct attribute access), so production is unchanged.

## 3. Third-order coercion site fixed (would otherwise have masked the fix)

`serve_trial` previously ran, **before** the guard:
```python
window_size = int(context.get("window_size", 1))
offset      = int(context.get("offset", 0))
```
Even with the signature fixed, an omitted value arriving as `None` would hit
`int(None)` → a raw **`TypeError`**, not the clean `MinerMetadataError`. REV4
reorders so `build_trial_context_from_serve(...)` + `set_trial_context(...)` run
**first** (guard fires on `None` before any coercion), then the window params are
read from the **validated projection** (`trial_ctx["window_size"]` /
`["offset"]` / `["sessions"]`) — already int-coerced and guaranteed non-None —
never re-fabricated from raw `context`. `compute_dataset_sha256` was moved up
accordingly (the context build needs it). This is the site that would have produced
a REV5 finding; it is closed.

## 4. Full-audit — the fail-closed set is now CLOSED

Team Alpha independently re-verified every `_SERVE_CONTEXT_REQUIRED` field against
the actual context dict in the diff (not the summary table):

| field | source in `run_trial_miner` | can be silently omitted? |
| --- | --- | --- |
| trial_number | required positional param | No → `TypeError` |
| prng_base | required positional param | No → `TypeError` |
| forward_threshold | required positional param | No → `TypeError` |
| reverse_threshold | required positional param | No → `TypeError` |
| skip_min | `Optional[int] = None` (REV3) | No → guard raises |
| skip_max | `Optional[int] = None` (REV3) | No → guard raises |
| window_size | `Optional[int] = None` (REV4) | No → guard raises |
| offset | `Optional[int] = None` (REV4) | No → guard raises |
| sessions | `kwargs.get("sessions")`, **not** in `_SERVE_CONTEXT_REQUIRED` | N/A — optional, `None → []` |

**No mandatory field can be fabricated by an omitting caller.** The set is complete.

## 5. Gates — B4 corrected, B5 added (both non-vacuous)

- **Gate B4 (corrected):** Beta observed REV3's B4 explicit-zero success case omitted
  `window_size`/`offset`, so it "passed for the wrong reason." B4 now supplies
  `window_size`/`offset` in every case and omits exactly one skip field at a time.
- **Gate B5 (new):** through the **actual `run_trial_miner()` entry point** — omitted
  `window_size` → `MinerMetadataError` (no context row, no stripes, no manifests);
  omitted `offset` → same; companion: explicit `offset=0` (window supplied) still
  persists. Verified non-vacuous against a reverted copy (old code fabricates
  `window_size=1`, persists, no raise → B5 fails against it).
- **3 real-serve Phase-4 gates** (37, 57, `_run_serve_thread`) now pass explicit
  `skip_min=0, skip_max=0, offset=0` — the Blocker analogue of the accepted skip=0.

D0 harness now **12/12** (7 original + B1 + B2 + B3 + B4-corrected + B5).

## 6. Packaging — standalone archive, Gate 20 hang fixed (empirically)

REV3 hung at Gate 20 because the archive omitted `prng_registry.py` and other repo
deps. REV4 was built by **extract-run-add-repeat** until both suites run to
completion from the extracted dir alone. The run required, beyond `prng_registry.py`:
`utils/survivor_loader.py`, the PWC/ZMQ coexistence modules (`persistent/`,
`integration/`), `window_optimizer.py`, `agent_manifests/window_optimizer.json`,
the phase-0/1/2/3 test files, and `sieve_gpu_worker.py` (+ guarded
`adaptive_thresholds.py`/`hybrid_strategy.py`, lazily imported by
`range_miner_worker.py:719` in phase-3 gate 12).

**Archive:** 27 members, SHA-256
`d81c4302369ae7ed420e17f730e2915a37d97219bca874ceb84149c9a11c7460`.

**Proof of standalone execution** (from the extracted dir, `fatal: not a git
repository` — genuinely detached):
```
$ PYTHONPATH=. python3 tests/test_s172_phase5_d0.py          → 12/12
$ PYTHONPATH=. python3 tests/test_s172_phase4_coordinator.py → 63/63
```
Confirmed deterministic across two independent fresh extracts.

## 7. IMPORTANT — where to read what (diff vs. archive)

Three of D0's four files are **tracked**; the harness is **untracked**, so the
`git diff` does NOT contain it:
- **`tmp/d0_rev4.diff`** (in the archive) — the three tracked files
  (`range_miner_coordinator.py`, `window_optimizer_integration_final.py`,
  `tests/test_s172_phase4_coordinator.py`), verified byte-current with the working
  tree.
- **`tests/test_s172_phase5_d0.py`** (a full file member in the archive, not in the
  diff) — the D0 harness with gates B1–B5. **Read it from the archive**, not the diff.

## 8. Run instructions (Beta)

```
sha256sum s172_phase5_d0_rev4.tar.gz
# expect d81c4302369ae7ed420e17f730e2915a37d97219bca874ceb84149c9a11c7460
mkdir d0_verify && tar xzf s172_phase5_d0_rev4.tar.gz -C d0_verify && cd d0_verify
PYTHONPATH=. python3 tests/test_s172_phase5_d0.py          # expect 12/12
PYTHONPATH=. python3 tests/test_s172_phase4_coordinator.py # expect 63/63
```
(The two-process conflicting-context probe from REV2 is unchanged; the concurrency
core was accepted.)

## 9. Recommendation

Team Alpha recommends **approval of D0 REV4**. Both remaining fabricated fields are
fixed to the accepted fail-closed shape; the full audit proves no mandatory field can
be silently omitted (set closed); the third-order `serve_trial` coercion that would
have masked the fix is repaired; B4 is de-vacuoused and B5 added, both proven
non-vacuous; the archive is genuinely standalone with execution proof. Suites green
on a clean tree and from the extracted archive.

On approval, Michael commits the four D0 files by name (pull-first onto current HEAD,
dual-push); the `.gitignore models/` guard is already committed (`e5fb370`).

**Submitted for binding verification.** Archive SHA-256
`d81c4302369ae7ed420e17f730e2915a37d97219bca874ceb84149c9a11c7460`.
