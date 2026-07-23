# CLAUDE_CODE_CORRECTION_S172_PHASE5_D0_REV2.md

**S172 RANGE-MINER — Phase 5 D0, correction round (Beta REJECT → resubmit)**

Team Beta issued a binding **REJECT** on D0 with **two narrow integrity blockers**.
No architectural redesign — the D0 architecture, durable projection, restart
reconstruction, phase semantics, provenance propagation, and interim-manifest
behavior are all approved. The gate-22 whitelist is approved (keep it
path-specific — no glob, no broad `tests/`). Fix exactly the two blockers below,
add the required gates, rerun all three suites green, STOP and report. Do **not**
commit, push, or run WATCHER.

**Working rules unchanged:** read live source before every claim; each new gate must
FAIL on the wrong behavior; exercise the real lifecycle (no mocks that assert the
comment); STOP at the gate and report.

**The meta-point behind both blockers (internalize it):** D0's contract is that
invalid or inconsistent metadata **fails closed BEFORE Phase 5 publication** — not
that the bad path "happens to be unreachable in production." A guard that only works
because a bad input never arrives is an assumption, not a guard. Both fixes convert
"unreachable in practice" into "raises on contact."

---

## Blocker 1 — `INSERT OR IGNORE` silently accepts a *conflicting* context

**Defect:** `set_trial_context` uses `INSERT OR IGNORE`, which prevents *mutation*
but does not enforce *semantic consistency*. Failure case:

```
run_id R created with window_size=20
run_id R accidentally re-served with window_size=40
INSERT OR IGNORE silently keeps window_size=20
new work may run under the 40 config while manifests publish the 20 config
→ internally inconsistent provenance, no error raised
```

**Required behavior — compare-and-insert, transactionally protected:**

```python
existing = get_trial_context(run_id)      # inside the same transaction/lock
if existing is None:
    insert(new_context)
elif canonicalize(existing) == canonicalize(new_context):
    return                                 # legitimate restart/replay: idempotent no-op
else:
    raise MinerMetadataError(
        f"conflicting immutable trial context for run_id={run_id}"
    )
```

Requirements:
- The read-compare-insert must be **transactionally protected** (under the existing
  `self._write_lock` AND a single DB transaction) so two concurrent initializations
  of the same `run_id` cannot race between the `get` and the `insert`. Do not read
  outside the lock and write inside it.
- `canonicalize()` must compare the **semantic** context, not raw row bytes: the
  same field set `get_trial_context` returns (11 trial-global + provenance),
  normalized so an identical replay compares equal — e.g. `sessions` compared as the
  same decoded list regardless of JSON key spacing, numeric fields compared by value
  not string. Round-trip through the same JSON encode/decode both sides so an
  identical replay is guaranteed equal.
- On conflict, the **original row is unchanged**, and the raise happens **before any
  stripe work** (this is called in `serve_trial` before assignment/dispatch —
  verify the ordering still holds after the change).
- Drop reliance on `INSERT OR IGNORE` for the conflict path; you may still use it as
  the concurrency-safe insert primitive *inside* the transaction as long as a losing
  concurrent insert is then detected by re-reading and comparing (so a race resolves
  to either "identical → no-op" or "conflict → raise", never silent divergence).

**Gate B1 (add to `tests/test_s172_phase5_d0.py`):** exercise all three conditions
against the real ledger:
1. **First insertion** → succeeds, row present.
2. **Identical replay** → succeeds as an idempotent no-op (row unchanged, no raise).
3. **One-field mutation** → raises `MinerMetadataError`; original row unchanged;
   **zero stripes assigned; zero manifests published.**
   Test at least mutations of `window_size`, `dataset_sha256`, and `prng_base`.

---

## Blocker 2 — fallbacks defeat the mandatory-metadata guard

D0's guard checks `is None`, but the production seam substitutes concrete values for
missing inputs *before* the guard sees them, so a missing mandatory field arrives as
"present but wrong" and passes. Remove every such substitution on the production
seam.

### 2a — `prng_base` fallback (this is the one Team Alpha wrongly closed)

**Defect** in `serve_trial`:
```python
"prng_base": context.get("prng_base") or family_name,
```
`prng_base` is mandatory. Substituting a concrete *variant* (`family_name`, e.g.
`java_lcg_hybrid_reverse`) for a missing base converts a missing-field error into
apparently-present-but-semantically-malformed metadata. The downstream
`utils/prng_encoding` hard-fail is **not** an acceptable backstop — D0's contract
requires the failure *before* Phase 5 publication, not eventually at NPZ conversion.

**Required:**
```python
prng_base = context["prng_base"]          # required-key access, no fallback
# then explicitly reject None / "" / invalid base:
if prng_base is None or (isinstance(prng_base, str) and prng_base.strip() == ""):
    raise MinerMetadataError("prng_base missing/empty in trial context (fail-closed).")
```
Do not fall back to `family_name` anywhere in the context-build path.

### 2b — numeric `.get(..., 0)` / `getattr(..., 0)` defaults

**Confirmed against source** (`window_optimizer.py:85-91`): `WindowConfig` declares
`window_size, offset, sessions, skip_min, skip_max` as **required** constructor
fields (no defaults). Only `forward_threshold` (0.40) and `reverse_threshold` (0.45)
carry dataclass defaults. So direct attribute access on the real production type is
safe and a malformed substitute object raises `AttributeError` loudly instead of
coercing to `0`.

**At the call site** (`window_optimizer_integration_final.py` `_use_miner`), replace:
```python
skip_min = getattr(config, 'skip_min', 0)
skip_max = getattr(config, 'skip_max', 0)
# (and likewise for window_size/offset/sessions if defaulted)
```
with direct access:
```python
skip_min          = config.skip_min
skip_max          = config.skip_max
window_size       = config.window_size
offset            = config.offset
sessions          = config.sessions
forward_threshold = config.forward_threshold
reverse_threshold = config.reverse_threshold
```

**At the context boundary** (`serve_trial`'s `set_trial_context({...})` build),
replace every `context.get(k, 0)` / `context.get(k, 0.0)` for a mandatory field with
**required-key access** `context[k]` (or a single explicit required-field validator
that raises `MinerMetadataError` listing any missing mandatory key). **Important
nuance:** the two thresholds *do* have `WindowConfig` dataclass defaults, so relying
on the `config` object cannot enforce "threshold present." Beta's requirement is that
the **context-boundary** access reject a missing threshold — so `forward_threshold`/
`reverse_threshold` must be enforced as required **keys of the context dict**
(`context["forward_threshold"]` / required-field validator), not read from the
`WindowConfig` object where a default would silently satisfy them.

`trial_number`: if it legitimately has no meaningful value in some path, decide with
Team Alpha whether it is mandatory; if mandatory, required-key access, no `-1`
default. (Beta's blocker names skip bounds and thresholds explicitly; treat
`trial_number` consistently with the mandatory list — flag if you find a path where
it can't be supplied rather than defaulting it silently.)

**Gate B2 (add to `tests/test_s172_phase5_d0.py`):** each of these must fail
**before stripe creation**:
- missing `prng_base` → raise
- empty `prng_base` (`""`) → raise
- missing `skip_min` → raise
- missing `skip_max` → raise
- missing `forward_threshold` → raise
- missing `reverse_threshold` → raise

For each: assert `MinerMetadataError` raised; **no trial context inserted; no stripe
assigned; no manifest published.**

Additionally assert **`threshold_used` correctness**: equals `forward_threshold` for
forward phases (1, 3) and `reverse_threshold` for reverse phases (2, 4).

---

## Rerun + report

- D0 harness (existing 7 + new B1 three-condition + new B2 six-missing +
  threshold_used assertion): **all green**
- Phase 4: **63/63**
- Phase 3: **17/17**

STOP and report with the actual four-file diff. Do not commit/push/WATCHER. Team
Alpha re-reviews adversarially (holding the fail-closed *contract*, not
reachability), then resubmits the four-file patch to Beta for binding verification.

## Commit-hygiene note (for the eventual commit, not this step)

Per Beta: `.gitignore models/` is a **separate one-line housekeeping commit**, NOT
folded into the D0 feature commit. Do not mix repository hygiene with the Phase-4
seam correction. (Michael handles commits; this is recorded so the correction doc
and the review packet agree.)
