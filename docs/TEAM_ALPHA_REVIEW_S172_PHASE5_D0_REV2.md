# TEAM_ALPHA_REVIEW_S172_PHASE5_D0_REV2.md

**S172 RANGE-MINER — Phase 5 Deliverable D0, correction round (REV2)**
**Resubmission for Team Beta binding verification**

**Reviewer:** Team Alpha (orchestrator)
**Prior verdict:** Beta REJECT — two narrow integrity blockers (no redesign).
**This round:** both blockers fixed, required gates added, all three suites green.
Team Alpha re-reviewed adversarially against live source, holding the **fail-closed
contract** (not "unreachable in practice"). Recommending approval.
**Base:** HEAD `0c3166a`, working tree on VM 101, uncommitted.
**Gates:** D0 **9/9** (7 original + B1 + B2) · Phase 4 **63/63** · Phase 3 **17/17**.
**Correction doc applied:** `docs/CLAUDE_CODE_CORRECTION_S172_PHASE5_D0_REV2.md`.

---

## 1. Blocker resolutions

### Blocker 1 — `INSERT OR IGNORE` accepted a conflicting context silently → FIXED

`set_trial_context` is now **compare-and-insert under `self._write_lock` in one DB
transaction**:
- read existing → if absent, `INSERT OR IGNORE` (concurrency-safe primitive) → re-read
- canonicalize existing vs. new; equal → idempotent no-op; unequal → raise
  `MinerMetadataError`, original row untouched.

`INSERT OR IGNORE` is **no longer relied on for immutability** — it survives only as
the insert primitive inside the txn, with a losing cross-process insert caught by the
re-read + compare. Two new module helpers: `_canonicalize_trial_context()` (semantic,
JSON sorted-key round-trip) and `_trial_context_row_to_ctx()` (row → the same dict
`get_trial_context` returns, so both sides compare field-for-field).

### Blocker 2 — fallbacks defeated the mandatory-metadata guard → FIXED

- **2a `prng_base`:** new `build_trial_context_from_serve()` uses required-key access
  `context["prng_base"]` and explicitly rejects `None`/`""`. **No `family_name`
  fallback anywhere.** The `serve_trial` call site now builds context through this
  function before any stripe assignment.
- **2b numerics:** call site (`window_optimizer_integration_final.py`) switched from
  `getattr(config, 'window_size', 1)` etc. to **direct** `config.window_size` /
  `config.skip_min` / `config.offset` / `config.sessions` — a malformed config raises
  `AttributeError` loudly instead of coercing to `1`/`0`/`None`. Context-boundary
  (`_SERVE_CONTEXT_REQUIRED`) uses **required-key access** for every mandatory field.
- **Threshold nuance honored:** because `forward_threshold`/`reverse_threshold` carry
  `WindowConfig` dataclass defaults, "threshold present" is enforced as a **required
  key of the context dict** (`_SERVE_CONTEXT_REQUIRED` includes both), NOT read from
  the `WindowConfig` object where a default would silently satisfy it.

---

## 2. Team Alpha adversarial re-review (against live source, fail-closed contract)

The prior Alpha review closed Blocker 2 too leniently ("unreachable in production").
This round holds Beta's stricter bar: **fails closed on contact**, not "the bad path
never arrives." Four checks, each traced to source:

**Check 1 — canonicalize is semantic, complete, and does NOT leak the timestamp —
PASS.** `_canonicalize_trial_context` builds from **only the 11 semantic fields**;
`created_at` is **absent** from the canonical form, so a same-run_id re-serve at a
different time canonicalizes equal (restart-idempotency holds). Every field is
type-coerced and `json.dumps(sort_keys=True)`, so numeric-string form / key spacing
cannot cause a false conflict. Both compare sides use the same function over the same
field set, so a mutation in any field changes the string → conflict raises; no field
is omitted from the compare.

**Check 2 — the transaction closes the race — PASS.** read → `INSERT OR IGNORE` →
re-read → compare → commit all execute inside `with self._write_lock: with
self._conn() as conn:`. The lock spans get-through-insert, so two concurrent
same-run_id inits cannot interleave. Losing cross-process insert is caught by the
re-read + compare → resolves to no-op or raise, never silent divergence.

**Check 3 — thresholds enforced at the context-dict boundary, not the WindowConfig
object — PASS.** `_SERVE_CONTEXT_REQUIRED` includes `forward_threshold` and
`reverse_threshold`; `build_trial_context_from_serve` raises on any missing key
before building. `prng_base` required-key with None/empty reject, no family fallback.

**Check 4 — B1-cond3 and B2 assert zero stripes/manifests via the REAL lifecycle —
PASS.** The harness records through `_MockSink.publish_shard` on the real coordinator
publish surface, and `_run_phase_to_publish` drives the genuine
assign→stage(`build_substripe_payload_bytes` + `record_substripe_result`)→
`record_stripe_complete`→`finalize_stripe`→publish path (asserts exactly one manifest
publishes on the happy path). On the failure paths:
- Gate B1 (`gateB1_compare_and_insert_conflict`): identical replay asserts row
  unchanged; each of window_size/dataset_sha256/prng_base mutation asserts
  `MinerMetadataError` + original row unchanged + **`all_stripes(run_id) == []`** +
  **`sink.published == []`**.
- Gate B2 (six missing-field cases): each asserts `MinerMetadataError` before stripe
  creation + **`sink.published == []`**.
- Gate D0-6 publish-path: `sink2.published == []` ("no manifest, and no `{}`, may
  reach Phase 5").
- Gates fail on the wrong behavior — the harness explicitly notes the pre-D0 `{}`
  publish makes Gate 1 fail against the old code.

---

## 3. One assumption to name (not a defect)

The Blocker-1 race protection assumes `self._write_lock` is **process-local** and that
**SQLite's own file locking** handles a genuinely cross-process second writer. In that
case the in-txn `INSERT OR IGNORE` + re-read + compare is the cross-process safety net
(a losing insert is detected, not assumed to have won). Within one process the lock
alone already serializes, so the `OR IGNORE` + re-read is belt-and-suspenders there.
Flagged so Beta can probe the two-process case directly rather than infer it; the
design handles it, but the assumption is worth stating.

---

## 4. Gate-22 whitelist (previously approved) — unchanged this round

The only change to the approved Phase-4 harness remains the single path-specific
allowlist entry `"tests/test_s172_phase5_d0.py"` (approved by Beta last round,
kept path-specific — no glob, no broad `tests/`). Claude Code confirms it did not
touch gate 22 this round. PWC/ZMQ/pwc_protocol untouched-assertions intact.

**File-race note (informational):** during this run an external S177 file
(`watcher_kpi_metricC_deterministic_v2_1.py`) appeared in the tree at 16:46 from the
other agent's workstream. Claude Code relocated only the pre-existing S177 files to
confirm 63/63 and restored them exactly; the D0 change-set is clean against gate 22.
This is a commit-hygiene item (two live workstreams share the tree), not a D0 defect —
see §5.

---

## 5. Files & commit hygiene (for Michael, post-approval)

Four-file D0 surface:
- `miner/range_miner_coordinator.py` (cumulative D0; this round Blockers 1+2)
- `window_optimizer_integration_final.py` (Blocker 2b call site)
- `tests/test_s172_phase5_d0.py` (new; B1 + B2 gates + D0-4 update)
- `tests/test_s172_phase4_coordinator.py` (gate-22 whitelist, prior round, untouched)

Staging discipline (two agents' untracked work + `models/gpt-oss-120b/` coexist):
1. `.gitignore models/` as a **SEPARATE one-line housekeeping commit** (Beta ruling —
   NOT folded into the D0 feature commit).
2. Stage D0 **strictly by name** — never `git add -A`. Note the new S177 stray
   `watcher_kpi_metricC_deterministic_v2_1.py` must also stay untracked.
3. Verify with `git status` + `git diff --cached` (only D0 staged; S176/S177 +
   `models/` still untracked).
4. Commit: `feat(s172): Phase 5 D0 — manifest metadata seam + durable context`
5. Dual-push: `git push origin main && git push public main`.

---

## 6. Recommendation

Team Alpha recommends **approval of D0 REV2**. Both blockers are fixed to the
prescribed shape; the four adversarial checks pass against live source; the new gates
drive the real publish lifecycle and assert zero-stripe/zero-manifest on every
fail-closed path; non-regression holds (63/63, 17/17). §3 names the one assumption for
Beta's probe. On approval, Michael commits per §5 (gitignore as its own commit), then
D1 begins in a fresh Claude Code session.

**Submitted for binding verification.** Four-file diff available on VM 101
(`/tmp/d0_rev2.diff` + `tests/test_s172_phase5_d0.py`).
