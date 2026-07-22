# ADDENDUM — WATCHER KPI Governance States v1.2 (to Proposal v1.1)

**Date:** 2026-07-21
**Session:** S179 (S178 follow-up — TB ruling `TB_RULING_S178_KPI_GOVERNANCE.md`, four amendments + §7)
**Author:** Team Alpha
**Status:** DRAFT ADDENDUM — for Team Beta code-review before any implementation is scoped.
**Authority:** Recommend-only. This document changes nothing. No thresholds are selected;
no autonomous enforcement is enabled; `watcher_policies.json` and all runtime code are
UNCHANGED. Every code location below is a **read-only trace on tree `0c3166a`**, not an edit.
**Basis:** `docs/TB_RULING_S178_KPI_GOVERNANCE.md` — *"ARCHITECTURE APPROVED IN PRINCIPLE —
FOUR MANDATORY AMENDMENTS BEFORE IMPLEMENTATION"* — resolving the four remaining issues on
`docs/PROPOSAL_WATCHER_KPI_GOVERNANCE_STATES_v1_1.md`.
**Verified tree head:** `git rev-parse HEAD == 0c3166a630be321809f415bb28af28e319d0fe1b`
(matches the head TB reviewed against; local working tree is dirty from a concurrent S172 lane
that touches none of the files traced here).

> **Relationship to v1.1.** This is an ADDENDUM. v1.1 is unchanged and remains the governing
> architectural direction; TB approved its enforcement location (§2.1), governance-filtering
> order (§2.2), fail-closed unmapped-trigger behavior (§2.3), canonical Hit@K source / Option A
> (§2.4), hard-vs-soft separation (§2.5), and derived null rates (§2.6). **v1.1 + this addendum
> together form the implementation contract.** Approved v1.1 content is referenced, not restated.
> Where this addendum supersedes specific v1.1 wording, it says so explicitly (see §4).
>
> Scope discipline is unchanged: still **no** numerical thresholds are selected. This addendum
> corrects the four control-flow / metric-contract gaps TB named, plus the §7 override policy.

---

## 1. SELFPLAY request + execution governance gates (TB §3)

The v1.1 primary gate (`Chapter13Orchestrator.run_cycle()` after `evaluate_triggers()`, before
LLM/validation/approval/routing — v1.1 §2.1) governs the normal cycle. The SELFPLAY
request path is **separate** and must be gated independently at two locations.

### 1.1 Trace basis (read-only, tree `0c3166a`)

- `TriggerManager.request_selfplay()` — `chapter_13_triggers.py:762`. It builds a request dict
  (`status="pending"` `:790`; `requires_watcher_approval=True` `:795`), then **writes the
  actionable request file** at `chapter_13_triggers.py:799-804`:
  ```python
  requests_dir = Path("watcher_requests")          # :799
  requests_dir.mkdir(exist_ok=True)                # :800
  request_file = requests_dir / f"{request_id}.json"
  with open(request_file, 'w') as f:               # :803
      json.dump(request, f, indent=2)              # :804
  ```
  and records it in trigger history (`self._record_trigger(...)`, `:812-815`).
- **No live caller.** `request_selfplay()` and its decision helper `should_request_selfplay()`
  (`chapter_13_triggers.py:819`) have **zero runtime callers** on tree `0c3166a`.
  `run_cycle()` (`chapter_13_orchestrator.py:251`) evaluates triggers (`:359`) and routes only to
  `request_approval()` (`:415/:422/:429/:440`); it never invokes the SELFPLAY path.
  **Consequence:** `request_selfplay()` is itself the sole entry point AND the sole writer, so the
  pre-creation gate cannot be delegated to a caller — it must live inside the function.

### 1.2 Pre-creation gate — insertion point `chapter_13_triggers.py:799`

Insert the governance resolution at the **top of `request_selfplay()`, immediately before the
`Path("watcher_requests")` write block (`:799`)**. Resolve the SELFPLAY governance entry
(`SELFPLAY_RECOMMENDED` registry mapping → enforcement state, via the v1.1 §2.2 candidate/filter
structure and the §2.3 fail-closed default for an unmapped trigger) and branch with **TB §3's
per-mode behavior, verbatim**:

| Enforcement | Behavior (TB §3, verbatim intent) |
|---|---|
| `audit_only` | record a **hypothetical** SELFPLAY recommendation; **do NOT create a pending watcher request**; `dispatched = false`; `approval_requested = false` |
| `shadow` | record **candidate and review metadata**; **do NOT create an executable request** (see §2) |
| `active` | create the **normal WATCHER-authorized request** (proceed to the `:799-804` write) |

Only the `active` branch reaches the file write. An unmapped/unknown SELFPLAY governance entry
resolves to `audit_only` and fails closed (v1.1 §2.3). Every branch emits a `KPI_EVALUATED`
ledger event (§7) with the resolved enforcement and the (hypothetical | candidate | dispatched)
disposition, so the decision is auditable even when no request file is written.

### 1.3 Consumer-side stale-request re-check — two insertion points

WATCHER must **re-read current governance state before authorizing or executing an existing
SELFPLAY request** (TB §3), giving the same stale-request protection v1.1 proposes for normal
retraining approvals. There are **two consumer routes**, so the re-check is placed at both:

1. **File-driven route (primary):** `_scan_watcher_requests()` (`agents/watcher_dispatch.py:555`,
   glob `:575`) → `process_chapter_13_request()` (`agents/watcher_dispatch.py:403`) → selfplay
   branch (`:508-518`) → `dispatch_selfplay()` (`:517`).
   **Insert the governance re-read in `process_chapter_13_request()` just before the request-type
   route branch (before `:508`):** if the current SELFPLAY state is not `active`, archive the
   request as `BLOCKED_BY_GOVERNANCE` (§7 event) and return without dispatching — fail closed.
2. **Defense-in-depth (authoritative chokepoint):** the top of `dispatch_selfplay()`
   (`agents/watcher_dispatch.py:88`), **beside the existing halt-flag safety gate (`:125-128`)**.
   This is the single point BOTH routes funnel through, and it also covers the direct CLI route in
   §1.4 that never touches `process_chapter_13_request()`. If SELFPLAY state is not `active`,
   refuse and emit `BLOCKED_BY_GOVERNANCE`.

### 1.4 Additional bypass paths found (Task 1.4)

- **Request creation:** within code, `request_selfplay()` (`chapter_13_triggers.py:799-804`) is the
  **sole** creator of a pending `watcher_requests/*.json`. The retrain path writes
  `pending_approval.json` (`APPROVAL_REQUEST_FILE`, `chapter_13_triggers.py:57`) via
  `request_approval()` (`:419`) and is governed by the v1.1 §2.1 primary gate — not a SELFPLAY
  bypass. Archive writes target `watcher_requests/archive/` and are excluded from the consumer glob
  (`agents/watcher_dispatch.py:577-578`). **No other writer creates pending request files.**
- **Consumer bypass (flag for TB):** the CLI `--dispatch-selfplay`
  (`watcher_agent.py:1838-1842`) builds a request dict inline and calls `dispatch_selfplay()`
  **directly**, skipping any `watcher_requests/` file AND `process_chapter_13_request()`. This
  confirms `dispatch_selfplay()` (not the file consumer) is the correct authoritative re-check
  location (§1.3.2), and — per Michael's S179 direction — **`--dispatch-selfplay` is classified as
  a privileged manual override under §8**, with the same flag/reason/identity/ledger requirements
  as `--execute`. Without the override flag it must respect governance state.

---

## 2. `shadow` contract (TB §4)

Adopt TB §4's definition **verbatim**. A `shadow` trigger:

```text
evaluate a candidate calibrated threshold
record whether it would have fired
record its hypothetical action
count false alarms, overlap and recovery
optionally generate a non-executable human-review artifact
never create an executable approval request
never dispatch pipeline work
```

**audit_only vs shadow distinction (exactly as TB framed it):**

```text
audit_only:
    collect raw metric and legacy-trigger observations;
    no candidate calibrated policy is asserted.        (raw observation)

shadow:
    evaluate a specific candidate policy;
    record what that candidate would have done;
    still prohibit execution.                          (candidate-policy evaluation)
```

The optional human-review artifact is **non-executable** — it may carry the candidate policy,
would-have-fired flags, and the false-alarm/overlap/recovery counts, but it is never a
`watcher_requests/*.json` and is never consumed by `_scan_watcher_requests()`. `shadow` at the
SELFPLAY path therefore records candidate + review metadata and stops (§1.2, `shadow` branch).

---

## 3. Per-metric state consistency (TB §4)

v1.1 validates `global_state` against `enforcement`; TB requires **each metric's own `state`** to
be validated too, as two tables. Any inconsistent combination **fails closed to `audit_only`**.

**Table A — metric state → allowed enforcement:**

| metric `state` | allowed `enforcement` |
|---|---|
| `BOOTSTRAP` | `audit_only` only |
| `CALIBRATING` | `audit_only` or `shadow` |
| `GOVERNED` | `audit_only`, `shadow`, or `active` |

**Table B — per-metric state ≤ global lifecycle ceiling:**

| `global_state` | max per-metric state |
|---|---|
| `BOOTSTRAP` | no metric may be `CALIBRATING` or `GOVERNED` |
| `CALIBRATING` | no metric may be `GOVERNED` |
| `GOVERNED` | individual metrics may remain in any lower state |

**Rejected example (TB §4, verbatim):**

```json
{
  "global_state": "GOVERNED",
  "metric": { "state": "BOOTSTRAP", "enforcement": "active" }
}
```

This violates Table A (`BOOTSTRAP` permits `audit_only` only). Validation must reject it and force
the metric to `audit_only` — never honor `active`. Both tables are enforced at policy-load /
governance-resolution time; a failure in either collapses the offending metric to `audit_only`
and emits a configuration warning (v1.1 §2.3 semantics).

---

## 4. Deterministic ranking contract (TB §5)

**Supersedes** v1.1's *"ties broken by rank index"* wording and its reliance on the NumPy default
sort. Trace basis (`prediction_generator.py`, tree `0c3166a`):

- `:836` `ranked_idx = np.argsort(predicted_quality)[::-1]` — the default `argsort` is **not** a
  documented deterministic tie-break for a governance artifact.
- `:841` `for idx in ranked_idx[:pool_size]:` — truncates the ranked indexes to `pool_size` raw
  rows **before** producing the saved prediction (TB §2.4).
- `:849` the emitted outcome is `seq[next_idx]` per survivor; duplicate **outcome** values across
  survivors are **not** collapsed — the first `pool_size` raw rows are assumed to be distinct
  predictions.

**Required contract:**

1. **Explicit deterministic ordering** (all three keys, in order):
   ```text
   primary   : predicted_quality  DESC
   secondary : stable survivor identifier  ASC
   tertiary  : original source index  ASC
   ```
2. **Persist both** `raw_rank` (position in the full deterministic ranked list) and `unique_rank`
   (position among first-occurrence unique outcomes).
3. **Duplicate-collapse rule (operational, not merely stored):** the **first occurrence in the
   deterministic ranked order wins** and is assigned the next `unique_rank`; later occurrences of
   the same outcome value are **retained only as provenance** and are **not** assigned another
   `unique_rank`.
4. **Walk, don't truncate-and-assume:** the generator continues through the **raw ranked survivor
   list until it has collected K UNIQUE outcome values**, rather than taking the first 300 raw rows
   and assuming they are 300 unique predictions. This is the additional ranked artifact TB §2.4
   requires before Hit@100 / Hit@300 are real measurements (Option A: preserve full Step-6 ranked
   output, do not derive from the truncated 20-entry pool).

---

## 5. Hit@K unavailability when fewer than K unique outcomes exist (TB §5)

When the deterministic walk (§4.4) yields fewer than K unique outcomes:

```text
requested_k       = 300
unique_k          = 217
hit300_available  = false
hit300            = null           # never label a 217-entry pool as Hit@300
```

A **separate, explicitly-distinct** observation MAY be recorded:

```text
hit_at_available_k
available_k = 217
null_rate   = available_k / draw_space        # e.g. 217 / draw_space
```

`hit_at_available_k` is **NOT** the Hit@K KPI and must never be substituted for it in any
governance decision; it is a diagnostic of pool depth only. (Consistent with the v1.1 §2.6 derived
null contract `p_null = unique_K / draw_space`.)

---

## 6. KPI ledger locking / durability correction (TB §6.1)

**Strike the `PIPE_BUF` claim.** `PIPE_BUF`-bounded atomicity applies to **pipes / FIFOs**, not to
`O_APPEND` writes on a **regular** JSONL file, so it is the wrong basis for the append-only ledger.
Replace with TB §6.1's cooperating-lock protocol:

```text
one cooperating lock protocol, shared by ALL writers:
    fcntl.flock(LOCK_EX)
    O_APPEND
    one encoded write while holding the lock
    flush() + os.fsync() before releasing the lock   # when durable audit persistence is required
```

All writers use the **same** lock. The **torn-final-line recovery rule is retained** (a reader that
encounters an incomplete final line discards it and recovers the prior complete records).

---

## 7. Evaluation identity + lifecycle event model (TB §6.2–6.3)

**Evaluation identity** (replaces the incomplete `draw_id + prediction_artifact_fingerprint` key):

```text
evaluation_id = f(
    draw_id,
    prediction_artifact_fingerprint,
    governance_policy_fingerprint,
    evaluator_schema_version,
    source_commit
)
```

An **exact match on all five components is idempotent** (suppress the duplicate). A change to **any**
component (governance-policy change, source-code change, trigger-schema change, state transition,
corrected diagnostic implementation) is a **new evaluation revision** — the second evaluation is
recorded, not suppressed, so audit history is preserved.

**One canonical append-only event ledger**, with multiple lifecycle events (approval and execution
happen *after* the initial evaluation, so a single row cannot hold their final values):

```text
KPI_EVALUATED
APPROVAL_REQUESTED
APPROVAL_APPROVED
APPROVAL_REJECTED
DISPATCH_STARTED
DISPATCH_COMPLETED
DISPATCH_FAILED
BLOCKED_BY_GOVERNANCE
```

Each later event references `evaluation_id + draw_id + request_id`. The original `KPI_EVALUATED`
event is **never overwritten** — subsequent state is expressed as **new appended events**.
(The SELFPLAY gate §1 and the manual-override §8 both emit into this same ledger:
`KPI_EVALUATED` at the gate, `BLOCKED_BY_GOVERNANCE` at a fail-closed consumer re-check, and the
override ledger event at §8.)

---

## 8. Privileged manual-execution override policy (TB §7)

Two direct-execution paths bypass the autonomous-trigger approval round-trip. Neither is removed
(both are explicit human operations), but **both are classified as privileged overrides** in the
implementation contract:

1. **`--execute` → `execute_standalone()`** — `chapter_13_triggers.py:630` (invoked from `main()`
   at `chapter_13_triggers.py:932`). Runs the selected pipeline steps with no approval request.
2. **`--dispatch-selfplay`** — `watcher_agent.py:1838-1842` (calls `dispatch_selfplay()` directly;
   S179 addition per Michael, and consistent with the §1.4 bypass finding).

**Both** require, as a privileged manual override:

```text
explicit --manual-governance-override        # without it, direct execution respects governance state
mandatory --reason
operator identity
source commit
policy fingerprint
requested steps
ledger event                                 # appended to the §7 canonical ledger
```

Without `--manual-governance-override`, direct execution **still respects governance state** (i.e.
it fails closed exactly as an autonomous trigger would when the metric/global state prohibits
`active`). This prevents either command from becoming an undocumented bypass during later autonomous
operation. **Explicitly flagged for TB:** `--dispatch-selfplay` is newly folded into this override
policy in v1.2 (it was not named in TB §7, which cited only `--execute`).

---

## Appendix A — TB §11 coverage checklist (Task 0)

| TB §11 item | Ruling ref | Addendum § | Insertion point / basis (file:line, tree `0c3166a`) |
|---|---|---|---|
| 1. SELFPLAY request + execution gates | §3 | §1 | pre-create `chapter_13_triggers.py:799`; consumer `agents/watcher_dispatch.py` pre-route (<`:508`) + `dispatch_selfplay` top (`:88`, beside `:125-128`) |
| 2. Exact `shadow` behavior | §4 | §2 | TB §4 verbatim; audit_only(raw) vs shadow(candidate) |
| 3. Per-metric state consistency | §4 | §3 | Tables A + B; fail-closed to `audit_only`; TB rejected example |
| 4. Deterministic ranking + duplicate collapse | §5 | §4 | supersedes v1.1; `prediction_generator.py:836,:841,:849` |
| 5. Hit@K unavailable (<K unique) | §5 | §5 | `hit{K}_available=false` / `hit{K}=null`; optional `hit_at_available_k` |
| 6. JSONL locking/durability | §6.1 | §6 | strike PIPE_BUF; `flock(LOCK_EX)`+`O_APPEND`+write+`fsync` |
| 7. Evaluation identity + lifecycle events | §6.2–6.3 | §7 | 5-component id; 8 lifecycle events; never overwrite |
| 8. Privileged manual override | §7 | §8 | `execute_standalone()` `chapter_13_triggers.py:630`/`:932`; `--dispatch-selfplay` `watcher_agent.py:1838-1842` |

Analyzer defects (TB §9–§10) are addressed in `watcher_kpi_metricC_deterministic_v2_2.py`, not in
this proposal (see the S179 changelog for the v2.1→v2.2 verification matrix).

## Appendix B — implementation-files / test-plan delta (adds to v1.1 §10 only)

These are the *additional* items v1.2 introduces on top of v1.1's §10 plan (no v1.1 item is
removed or restated):

- **`chapter_13_triggers.py`** — pre-creation SELFPLAY gate inside `request_selfplay()` (§1.2);
  `execute_standalone()` privileged-override guard (§8).
- **`agents/watcher_dispatch.py`** — consumer-side stale re-check in `process_chapter_13_request()`
  (§1.3.1) and `dispatch_selfplay()` (§1.3.2).
- **`watcher_agent.py`** — `--dispatch-selfplay` privileged-override guard (§8).
- **`prediction_generator.py`** — deterministic 3-key ordering + `raw_rank`/`unique_rank` +
  unique-walk to K (§4); the additional ranked artifact (Option A, v1.1 §2.4).
- **KPI ledger module** — `flock(LOCK_EX)`+`O_APPEND`+`fsync` protocol (§6); 5-component
  `evaluation_id` + 8-event lifecycle, append-only, never-overwrite (§7).
- **Governance policy validator** — per-metric Table A + Table B, fail-closed to `audit_only` (§3);
  `shadow` runtime path (§2).
- **Test-plan additions:** (a) SELFPLAY audit_only/shadow/active branch tests — assert no
  `watcher_requests/*.json` written in audit_only/shadow, and `dispatched=false`/
  `approval_requested=false` in audit_only; (b) stale-request re-check at both consumer points
  fails closed when state ≠ active (incl. the `--dispatch-selfplay` CLI route); (c) per-metric
  state tables reject the TB example and every Table-A/B violation; (d) deterministic-ranking
  reproducibility + duplicate-collapse + unique-walk to K, and `hit{K}_available=false` when
  unique_k < K; (e) ledger concurrency: N cooperating writers under one `flock`, torn-final-line
  recovery, and idempotency on exact 5-component match vs new revision on any component change;
  (f) both manual-override commands fail closed without `--manual-governance-override` and emit the
  ledger event with it. Each gate constructed to FAIL on the pre-amendment behavior.

---

*End of v1.2 Addendum. Recommend-only; changes nothing. v1.1 + this addendum = the implementation
contract. Submit with Analyzer v2.2 for Team Beta binding verification.*
