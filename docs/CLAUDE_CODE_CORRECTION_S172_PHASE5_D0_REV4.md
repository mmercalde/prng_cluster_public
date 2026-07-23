# CLAUDE_CODE_CORRECTION_S172_PHASE5_D0_REV4.md

**S172 RANGE-MINER — Phase 5 D0, correction round 4 (Beta REJECT REV3 → REV4)**

Team Beta ran the REV3 archive: SHA matched, path-safe, D0 **11/11**, two-process
probe passed, missing-context fix passed, omitted-skip fix passed, **and the 13
Phase-4 test adaptations are APPROVED**. One code blocker remains plus one packaging
defect. Fix exactly these, correct the one gate Beta exposed as accidentally
vacuous, rebuild a genuinely standalone archive, rerun green, STOP and report. Do
**not** commit/push/WATCHER.

**Now accepted by Beta (do NOT touch):** absent-`trial_context` publish guard;
compare-and-insert conflict + two-process concurrency; omitted `skip_min`/`skip_max`
fix; the 13 Phase-4 gate context-seeding adaptations; the durable table; gate-22
whitelist; `prng_base`/`WindowConfig` fixes.

**The meta-point, final form:** `run_trial_miner` fabricates a default for MORE than
one mandatory field. REV3 fixed `skip_min`/`skip_max` but left `window_size` and
`offset` fabricating `1`/`0`. **This correction audits EVERY field the function
defaults, not just the two Beta named** — the fix is complete only when no mandatory
field in `_SERVE_CONTEXT_REQUIRED` can be fabricated by an omitting caller. Read live
source; verify the full audit; each gate must FAIL on the wrong behavior.

---

## Blocker — `run_trial_miner()` fabricates `window_size` and `offset`

**Beta's reproduction:** with `window_size` and `offset` omitted (skip explicit),
the trial proceeds and persists `window_size=1, offset=0`. Live location
`range_miner_coordinator.py:3758-3760`:
```python
"window_size": kwargs.get("window_size", 1),
"sessions":    kwargs.get("sessions"),
"offset":      kwargs.get("offset", 0),
```

### Required fix — same shape as the accepted skip fix

Make `window_size` and `offset` **explicit fail-closed parameters**:
```python
window_size: Optional[int] = None,
offset:      Optional[int] = None,
```
and pass them through **unchanged** to the serve context (no `or 1` / `or 0` / `int(... or 0)` coercion):
```python
"window_size": window_size,
"offset":      offset,
```
`_SERVE_CONTEXT_REQUIRED` already lists both, so `build_trial_context_from_serve`
rejects omission (None) before stripe creation. The real `_use_miner` call site
already passes `config.window_size`/`config.offset`, so production is unchanged.

### MANDATORY full-audit (do this, don't just fix the two named)

Enumerate **every** field `run_trial_miner` puts into the serve `context` via a
default (`kwargs.get(k, <default>)`, `.get(k, <default>)`, or a signature default),
and cross-check against `_SERVE_CONTEXT_REQUIRED`:
`(trial_number, window_size, offset, skip_min, skip_max, prng_base,
forward_threshold, reverse_threshold)`.
- Any **mandatory** field that can be fabricated by an omitting caller → convert to
  fail-closed (Optional=None passthrough, or required arg). Report the complete list
  you found and fixed.
- `sessions` is **intentionally optional** (not in `_SERVE_CONTEXT_REQUIRED`;
  None→[]): leave it defaulting. Do not fail-close it.
- Operational params (`serve_poll`, `serve_timeout`, `serve_read_deadline`,
  `worker_pool_size`, caps, timeouts) are **not** mandatory metadata: leave them.
- Confirm `trial_number`, `prng_base`, `forward_threshold`, `reverse_threshold` are
  explicit params that cannot be silently omitted; if any carries a fabricating
  default, fix it too and report it.

If the audit finds a mandatory field beyond `window_size`/`offset`, fix it in the
same pass — do NOT leave it for a REV5.

---

## Gate correction — B4 is accidentally vacuous (Beta's catch)

Beta observed: **Gate B4's explicit-zero success case omits both `window_size` and
`offset`, yet successfully persists a context** — so B4 currently *demonstrates the
hole* while claiming to prove fail-closed. Fix B4 so its success case supplies the
now-mandatory `window_size`/`offset` explicitly, and its failure cases prove omission
of each fails closed.

## Gate B5 (new) — `window_size`/`offset` fail closed via the real entry point

Through the **actual `run_trial_miner()` entry point** (not
`build_trial_context_from_serve` in isolation):
- omitted `window_size` → `MinerMetadataError`, no `trial_context` row, no stripes,
  no manifests;
- omitted `offset` → same;
- companion success: explicit `offset=0` (and `window_size` supplied) still persists
  a valid context — proving `0` remains a legitimate explicit value, only omission
  fails.

Verify non-vacuous against a reverted copy (old code persists `1`/`0` with no raise →
B5 must fail against it).

---

## Packaging — make the archive genuinely standalone (REV3 Gate 20 hang)

Beta's REV3 run **hung at Gate 20 before worker registration** because the archive
omitted `prng_registry.py` "and other repository dependencies." A readable archive is
not enough — Beta *executes* the Phase-4 command, so it must import and run clean.

**Determine the REAL runtime dep set empirically on 101 — do NOT guess:**
1. Build a candidate archive, extract to a scratch dir, and run BOTH advertised
   commands from that dir:
   ```
   PYTHONPATH=. python3 tests/test_s172_phase5_d0.py
   PYTHONPATH=. python3 tests/test_s172_phase4_coordinator.py
   ```
2. On each `ModuleNotFoundError` / `ImportError`, add the missing repo file, rebuild,
   re-run. Repeat until **both suites run to completion** (11/11 and 63/63) from the
   extracted archive alone — no reliance on the surrounding checkout.
3. Known missing: `prng_registry.py` (imported by `range_miner_worker.py` and
   `utils/prng_encoding.py`). There may be more that Gate 20's worker-registration
   path pulls — the empirical run is authoritative, not this list.

**Archive verification (report all three):**
- `tar tzf <archive> | wc -l`
- `tar tzf <archive>` full member listing (must show all four changed files by name
  + `prng_registry.py` + every dep the empirical run required)
- `sha256sum <archive>`
- **Proof of standalone execution:** paste the two test runs executed from the
  extracted scratch dir (not the working tree), showing 11/11 and 63/63 — this is
  what proves Gate 20 no longer hangs.

---

## Rerun + report

- D0 harness (existing 11 + corrected B4 + new B5): **all green** (expect 12/12)
- Phase 4: **63/63** (clean tree)
- Phase 3: **17/17**
- Standalone-archive proof: both suites green from the extracted archive dir.

STOP and report: the four-file diff, the full-audit field list (what defaults you
found + fixed), the archive member listing + SHA, and the extracted-dir test output.
Do not commit/push/WATCHER. Team Alpha re-reviews (this round: audit the WHOLE
`run_trial_miner` context build, and confirm standalone execution), then resubmits.

## Commit-hygiene note (unchanged)

`.gitignore models/` already committed (`e5fb370`). D0 stages its four files by name,
pull-first onto current HEAD, dual-push — only after Beta binding approval. Never
`git add -A`.
