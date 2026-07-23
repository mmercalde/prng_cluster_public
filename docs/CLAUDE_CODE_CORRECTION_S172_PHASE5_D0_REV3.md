# CLAUDE_CODE_CORRECTION_S172_PHASE5_D0_REV3.md

**S172 RANGE-MINER — Phase 5 D0, correction round 3 (Beta REJECT REV2 → REV3)**

Team Beta ran the REV2 archive: SHA matched, extraction path-safe, compiled, D0 harness
**9/9**, and the **independent two-process SQLite concurrency probe PASSED** (compare-
and-insert accepted). But Beta reproduced **two remaining fail-closed violations** via
the real publish/serve path, plus a packaging discrepancy. Fix exactly these three,
add the two required gates, rerun green, STOP and report. Do **not** commit/push/WATCHER.

**Accepted by Beta (do NOT touch):** gate-22 whitelist; durable context table;
compare-and-insert conflict handling; two-process concurrency; `prng_base`→`family_name`
fallback removal; direct `WindowConfig` attribute access at the call site.

**The meta-point (again, sharper this time):** a guard that only works because the bad
input "never arrives in production" is not a guard. REV2 fixed the *context-build*
fallbacks but left two paths where a missing input is silently converted to a
valid-looking value BEFORE the guard sees it. Both fixes below move the failure to the
point of contact. Read live source before every claim; each gate must FAIL on the wrong
behavior; drive the REAL lifecycle, not a validator in isolation.

---

## Blocker 1 — an empty manifest still reaches Phase 5 when the durable context row is absent

**Beta's reproduction (definitive):** with a verified shard staged but **no
`trial_context` row**, `publish_attempt()` runs:
```python
trial_ctx = self.ledger.get_trial_context(run_id)
trial_metadata = (derive_trial_metadata(trial_ctx, stripe)
                  if trial_ctx is not None else None)
```
→ `trial_metadata=None` → `_build_manifest(..., trial_metadata=None)` emits
`"trial_metadata": {}` → `phase5_sink.publish_shard(manifest)`. Result:
`published_count 1, trial_metadata {}`. **An empty manifest reaches Phase 5.**

Gate D0-6 never tested this: it tested incomplete-context rejection at
`set_trial_context()` and corrupted `family_name` *after a valid context exists* — never
publication with the durable row **completely absent**. The `if trial_ctx is not None`
branch with a legacy-`{}` fallback is the hole. The comment claiming "production always
persists context first, so `{}` can't reach Phase 5" was an assertion, never enforced.

**Required fix (in `publish_attempt`, BEFORE building or publishing any manifest):**
```python
trial_ctx = self.ledger.get_trial_context(run_id)
if trial_ctx is None:
    raise MinerMetadataError(
        f"missing durable trial context for run_id={run_id!r}; "
        "refusing Phase 5 publication"
    )
trial_metadata = derive_trial_metadata(trial_ctx, stripe)
```
Remove the `... if trial_ctx is not None else None` fallback on the publish path. The
missing-context case now fails closed instead of leaking `{}`.

**Note on the interim `_finalize_stage` manifest:** it legitimately still calls
`_build_manifest` with no metadata and keeps `{}` — that path is NOT published to
Phase 5. The fix is specifically in the **publish** path (`publish_attempt`), not in
`_build_manifest`'s `None` handling. Confirm from source that `_finalize_stage`'s
manifest never reaches `publish_shard`; if it does, that is a second leak — STOP and
report.

**Gate B3 (add to `tests/test_s172_phase5_d0.py`):** stage a **verified shard with NO
`trial_context` row**, attach a real mock Phase 5 sink, call `publish_attempt()`, assert:
- `MinerMetadataError` raised
- `sink.published == []`
- `coordinator.enqueued == []` (no shard marked enqueued)
- the shard is **not** marked enqueued in the ledger

Drive the REAL staging + publish path (mirror `_run_phase_to_publish`), not a mock that
asserts the comment. The gate must FAIL against the current code (which publishes `{}`).

---

## Blocker 2 — `skip_min`/`skip_max` still silently default to 0 in `run_trial_miner()`

**Defect:** `run_trial_miner()` declares
```python
skip_min: int = 0,
skip_max: int = 0,
```
and always inserts those into the serve context. A caller that **omits** the fields has
the omission converted to a valid-looking `0` **before** it reaches
`build_trial_context_from_serve`'s missing-field guard. Same fallback-masking REV2 was
supposed to remove — one layer up, at the entry point instead of the context builder.
(REV2 correctly fixed the *call site* to pass `config.skip_min`; it left the *callee
signature* defaulting.)

**Required fix — fail-closed defaults at the entry point:**
```python
skip_min: Optional[int] = None,
skip_max: Optional[int] = None,
```
(or keyword-only required args). Then the existing context validator rejects an omitted
value instead of seeing a fabricated `0`. Any synthetic/serve-path test that
legitimately needs zero must now pass `skip_min=0, skip_max=0` **explicitly**.

When building the serve context inside `run_trial_miner`, do not coerce `None` to `0`
before the guard — pass the value through so `build_trial_context_from_serve`'s
required-key/None check fires. (If `run_trial_miner` currently does
`"skip_min": skip_min` into the context dict, a `None` will now correctly reach the
guard; confirm no intermediate `int(... or 0)` swallows it.)

**Gate B4 (add to `tests/test_s172_phase5_d0.py`):** call the **actual
`run_trial_miner()` entry point** (not only `build_trial_context_from_serve`) with
`skip_min`/`skip_max` **omitted**, assert it fails with `MinerMetadataError` **before
serving or stripe creation** — no trial context inserted, no stripe assigned, no
manifest published. A companion assertion: the same call **with** `skip_min=0,
skip_max=0` explicit succeeds (proves zero is still a legitimate explicit value, only
omission fails).

---

## Packaging fix — include the actual changed file

The REV2 archive's `tar tzf` listed only 9 members and
**`window_optimizer_integration_final.py` was absent** — its patch was present only in
`tmp/d0_rev2.diff`. The corrected REV3 archive must contain the **actual changed file**,
not just its diff. Beta's requirement: the complete four-file source package.

**REV3 archive must contain (verify with `tar tzf` AND a member count):**
```
miner/range_miner_coordinator.py            # changed (Blockers 1+2 + prior)
window_optimizer_integration_final.py       # changed (call site) — WAS MISSING, include it
tests/test_s172_phase5_d0.py                # changed (+ B3, B4 gates)
tests/test_s172_phase4_coordinator.py       # changed (gate-22 whitelist)
# transitive deps for standalone execution:
miner/range_miner_worker.py
miner/range_miner_protocol.py
miner/__init__.py
utils/prng_encoding.py
utils/__init__.py
tmp/d0_rev3.diff                            # full 4-file diff vs HEAD 0c3166a
```
After building, run `tar tzf <archive> | wc -l` and confirm all four changed files are
listed by name — do not trust that the tar command's arguments all made it in.

---

## Rerun + report

- D0 harness (existing 9 + B3 missing-context-publish + B4 omitted-skip-via-entrypoint):
  **all green**
- Phase 4: **63/63**
- Phase 3: **17/17**

STOP and report with the actual four-file diff AND a `tar tzf` member listing proving
`window_optimizer_integration_final.py` is in the archive. Do not commit/push/WATCHER.
Team Alpha re-reviews adversarially (this round: trace the missing-context publish path
and the entry-point signature, not just the context builder), then resubmits the
complete archive + hash for Beta binding verification.

## Commit-hygiene note (unchanged, for the eventual commit)

`.gitignore models/` is a SEPARATE one-line housekeeping commit, NOT folded into D0. New
S177 stray `watcher_kpi_metricC_deterministic_v2_1.py` stays untracked. Stage D0 by name;
never `git add -A`.
