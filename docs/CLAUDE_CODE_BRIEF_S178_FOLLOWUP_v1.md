# Claude Code Brief — S178 Follow-Up: Proposal v1.2 Addendum + Analyzer v2.2 — v1

**Runs on:** VM101, as `michael`, from `/home/michael/distributed_prng_analysis`.
**Context:** Team Beta approved Proposal v1.1 as the governing architectural
direction (ruling: `docs/TB_RULING_S178_KPI_GOVERNANCE.md` — read FIRST; §11 is
your deliverable list). Implementation is paused only for a short v1.2
ADDENDUM (four amendments + the §7 override policy) and Analyzer v2.2 (five
fixes). This is a narrow round — no rewrite, no runtime implementation.

---

## CONCURRENT-SESSION RULES (unchanged from last session; still in force)

Another agent may be active on this tree (S172 workstream).

- **Write-set is EXACTLY three files:**
  `docs/PROPOSAL_WATCHER_KPI_GOVERNANCE_STATES_v1_2_ADDENDUM.md`,
  `watcher_kpi_metricC_deterministic_v2_2.py`,
  `docs/SESSION_CHANGELOG_YYYYMMDD_S<N>.md`. Plus append-only memory. Nothing else.
- Never touch the S172 lane (`miner/`, `tests/test_s172*`,
  `window_optimizer_integration_final.py`, S172 docs/briefs).
- All runtime files READ-ONLY (`chapter_13_*.py`, `watcher_agent.py`,
  `watcher_policies.json`, `prediction_generator.py`, manifests).
- Transient tree churn → pause and re-check before concluding anything.
- No state-altering git commands. Read-only git is fine.
- No commits/pushes, no pipeline runs, no policy edits. Read source before
  every claim (file:line). Use /bin/grep for searches that must cover .json.

---

## Task 0 — Read the ruling; build the amendment checklist

Read `docs/TB_RULING_S178_KPI_GOVERNANCE.md` in full. Build a checklist mapping
the 8 addendum items (§11) + 5 analyzer fixes to where each will be addressed.
It becomes the addendum's coverage appendix.

---

## Task 1 — Trace the SELFPLAY path (Amendment 1 evidence) — READ-ONLY

The one new source trace this round. Pin with file:line:

1. `request_selfplay()` — where it writes the pending JSON under
   `watcher_requests/`, how the request is marked as requiring WATCHER
   approval, and how it lands in trigger history. Quote the write site.
2. The exact insertion point for the pre-creation governance gate (in
   `request_selfplay()` or its caller — state which and why).
3. The WATCHER-side SELFPLAY consumer: where an existing SELFPLAY request is
   authorized/executed, and where the stale-request governance re-check goes.
4. Confirm whether any OTHER request-creation path exists that bypasses both
   the orchestrator gate and `request_selfplay()` (grep for writes into
   `watcher_requests/`). If found, report it — TB will want it gated too.

---

## Task 2 — Write the v1.2 Addendum

`docs/PROPOSAL_WATCHER_KPI_GOVERNANCE_STATES_v1_2_ADDENDUM.md` — an ADDENDUM
to v1.1 (v1.1 untouched; do not restate approved content, reference it).
Header: same authority block (DRAFT for TB review; recommend-only; changes
nothing) + explicit statement that v1.1 + this addendum together form the
implementation contract. Cover TB §11's eight items exactly:

1. **SELFPLAY gates** (from Task 1): pre-creation gate with TB's per-mode
   behavior verbatim (audit_only → record hypothetical, NO pending request,
   dispatched=false, approval_requested=false; shadow → record candidate +
   review metadata, no executable request; active → normal WATCHER-authorized
   request), plus the consumer-side stale-request re-check. File:line for both
   insertion points. Include any additional bypass path found in Task 1.4.
2. **Shadow contract**: adopt TB §4's definition verbatim — evaluate a
   specific candidate calibrated policy, record would-have-fired +
   hypothetical action, count false alarms/overlap/recovery, optional
   non-executable human-review artifact, never create executable requests,
   never dispatch. State the audit_only-vs-shadow distinction exactly as TB
   framed it (raw observation vs candidate-policy evaluation).
3. **Per-metric state consistency**: the metric-state → allowed-enforcement
   rules AND the metric-state ≤ global-state ceiling, both as validation
   tables; any inconsistent combination fails closed to audit_only; include
   TB's rejected example.
4. **Deterministic ranking contract**: explicit sort keys (predicted_quality
   desc → stable survivor id asc → source index asc), persist raw_rank +
   unique_rank, duplicate-collapse rule (first occurrence in deterministic
   order wins; later duplicates = provenance only), generator walks the raw
   ranked list until K UNIQUE outcomes are collected (no truncate-and-assume).
   Note this supersedes v1.1's "ties broken by rank index" wording and the
   np.argsort assumption (cite prediction_generator.py lines).
5. **Hit@K unavailability**: fewer than K unique → hit{K}_available=false,
   hit{K}=null; optional separate hit_at_available_k observation with
   available_k and its derived null rate, explicitly NOT the Hit@K KPI.
6. **Ledger locking correction**: strike the PIPE_BUF claim (wrong for
   regular files — applies to pipes/FIFOs); replace with TB's protocol:
   single cooperating fcntl.flock(LOCK_EX) + O_APPEND + one encoded write
   under the lock + flush/fsync before release for durable audit records;
   all writers share the lock; torn-final-line recovery retained.
7. **Evaluation identity + lifecycle events**: identity = draw_id +
   prediction_artifact_fingerprint + governance_policy_fingerprint +
   evaluator_schema_version + source_commit (exact match idempotent; any
   component change = new revision). Lifecycle events KPI_EVALUATED /
   APPROVAL_REQUESTED / APPROVAL_APPROVED / APPROVAL_REJECTED /
   DISPATCH_STARTED / DISPATCH_COMPLETED / DISPATCH_FAILED /
   BLOCKED_BY_GOVERNANCE, each referencing evaluation_id + draw_id +
   request_id; original evaluation event never overwritten.
8. **Privileged manual override** (§7): --execute path classified as
   privileged override — requires --manual-governance-override + mandatory
   --reason + operator identity + source commit + policy fingerprint +
   requested steps + a ledger event; without the flag, direct execution
   respects governance state. Cite execute_standalone() location.

Close with the Task 0 coverage appendix and an updated implementation-files/
test-plan delta (only what these amendments add to v1.1's §10 plan).

---

## Task 3 — Write Analyzer v2.2

`watcher_kpi_metricC_deterministic_v2_2.py` (v2.1 untouched). TB §11's five
fixes:

1. **Threshold-shape-aware sensitivity**: the assumed_healthy_sensitivity
   block must use the same shape logic as the primary analysis — fires on
   hit+miss → p=1.0; fires on miss only → p=1-assumed_rate; fires on neither
   → p=0.0 (never fires; report null + status, consistent with the primary
   verdict). Derive waiting time from that probability.
2. **Explicit provenance**: --repo-root argument; resolve via
   `git -C <root> rev-parse HEAD` + `git -C <root> status --porcelain`.
3. **Record**: analyzed_repo_root, analyzed_source_commit,
   analyzed_tree_dirty, policy_file_path, policy_file_sha256,
   analyzer_file_sha256.
4. **Fail-fatal** when provenance cannot be resolved (no silent null). If an
   explicitly non-authoritative mode is offered (e.g. --no-provenance), it
   must mark the findings authoritative=false.
5. **Reject Boolean collapse_threshold** (isinstance(x, bool) check before
   float coercion — True currently passes as 1.0).

Verify: re-run the full v2.1 validation matrix (all previously-passing checks
still pass; nulls still ≈1.0204 / 5.3146); reproduce TB's two §9
contradiction cases and show they now agree with the primary shape (threshold
=0 → sensitivity reports never-fires; threshold=1 → wait = 1 draw at any
rate); test provenance from inside the repo (records commit + dirty flag),
from outside (fatal), and with --repo-root pointing at a non-repo (fatal);
test Boolean threshold rejected. Save findings to
`watcher_kpi_metricC_v2_2_findings.json` (strict JSON; verify it parses; run
with --repo-root pointing at this tree so provenance is real).

---

## Task 4 — Changelog and stop

`docs/SESSION_CHANGELOG_YYYYMMDD_S<N>.md`: checklist coverage, v2.1→v2.2 diff
summary, Task 1 trace summary, any concurrent-session observations. Deliver
the three files for Michael → Team Beta. **Stop after delivering.** No
implementation, no policy edits, no walk-forward.

## Checkpoint

After Task 1, report the SELFPLAY trace (both insertion points + any bypass
paths found) to Michael and WAIT for confirmation before writing the addendum
— same pattern as the S178 session's gate-placement checkpoint.
