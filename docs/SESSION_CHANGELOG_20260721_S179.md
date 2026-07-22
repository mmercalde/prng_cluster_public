# SESSION CHANGELOG — 2026-07-21 — S179

**Lane:** WATCHER KPI Governance (S178 follow-up). Concurrent S172 lane active on the same
tree throughout (see below). **Recommend-only; nothing committed, no policy edited, no pipeline
or WATCHER run.**

**Basis:** `docs/TB_RULING_S178_KPI_GOVERNANCE.md` — Proposal v1.1 *ARCHITECTURE APPROVED IN
PRINCIPLE — four mandatory amendments*; Analyzer v2.1 *REVISION REQUIRED — two logic/provenance
defects*. Verified tree head `0c3166a630be321809f415bb28af28e319d0fe1b` (matches TB's review head).

## Deliverables (write-set: exactly three files + memory)

1. `docs/PROPOSAL_WATCHER_KPI_GOVERNANCE_STATES_v1_2_ADDENDUM.md` — v1.2 addendum (8 items + §7).
2. `watcher_kpi_metricC_deterministic_v2_2.py` — Analyzer v2.2 (5 fixes).
3. `docs/SESSION_CHANGELOG_20260721_S179.md` — this file.
   Verification artifact (analyzer output, sanctioned by Task 3): `watcher_kpi_metricC_v2_2_findings.json`.

## TB §11 coverage checklist (Task 0)

| Item | Where | Item | Where |
|---|---|---|---|
| 1 SELFPLAY gates | Addendum §1 | 6 JSONL locking | Addendum §6 |
| 2 shadow contract | Addendum §2 | 7 eval identity + lifecycle | Addendum §7 |
| 3 per-metric state | Addendum §3 | 8 manual override | Addendum §8 |
| 4 deterministic ranking | Addendum §4 | A1–A5 analyzer fixes | Analyzer v2.2 |
| 5 Hit@K unavailable | Addendum §5 | | |

## Task 1 — SELFPLAY trace (read-only, confirmed with Michael before writing)

- **Writer / pre-creation gate:** `request_selfplay()` `chapter_13_triggers.py:762`; write site
  `:799-804` (`status=pending` `:790`, `requires_watcher_approval=True` `:795`, trigger history
  `:812-815`). **No live caller** — `run_cycle()` (`chapter_13_orchestrator.py:251`) routes only to
  `request_approval()`; `should_request_selfplay()` (`:819`) is uncalled. Gate therefore lives
  **inside** `request_selfplay()` at `:799` (sole entry = sole writer), branching per TB §3
  (audit_only → hypothetical, no request, dispatched/approval_requested=false; shadow → candidate
  + review metadata, no executable request; active → normal request).
- **Consumer re-check:** `process_chapter_13_request()` `agents/watcher_dispatch.py:403` pre-route
  (before `:508`), + defense-in-depth at `dispatch_selfplay()` top (`:88`, beside halt-flag gate
  `:125-128`).
- **Bypass paths (Task 1.4):** `request_selfplay()` is the **sole** creator of pending
  `watcher_requests/*.json` (retrain path uses `pending_approval.json` via `request_approval()`
  `:419`, governed by v1.1 §2.1; archive writes to `watcher_requests/archive/`, excluded from the
  consumer glob `:577-578`). **CLI `--dispatch-selfplay`** (`watcher_agent.py:1838-1842`) calls
  `dispatch_selfplay()` directly, bypassing the file consumer → per Michael's S179 direction, now
  classified as a **privileged manual override** alongside `--execute` (Addendum §8), flagged for TB.
- **Insertion points confirmed by Michael** before the addendum was written (both as traced; plus
  the `--dispatch-selfplay` override addition).

## v2.1 → v2.2 diff summary (Analyzer)

| # | Fix (TB) | Change |
|---|---|---|
| 1 | §9 threshold-shape sensitivity | assumed-rate collapse block now reuses the primary `(fires_on_hit, fires_on_miss)` shape: both→p=1.0; miss-only→p=1−assumed_rate; neither→p=0.0. Was unconditional `1/p_miss`. |
| 2 | §10 explicit provenance | new `--repo-root`; resolve via `git -C <root> rev-parse HEAD` + `status --porcelain` (not the cwd). |
| 3 | §10 record fields | probe now records `analyzed_repo_root, analyzed_source_commit, analyzed_tree_dirty, policy_file_path, policy_file_sha256, analyzer_file_sha256`. |
| 4 | §10/§11.4 fail-fatal | unresolved authoritative provenance is FATAL (no silent null); `--no-provenance` marks `authoritative=false`. |
| 5 | §10 reject bool threshold | `_validate_rate()` rejects `bool` before `float()` (bool is an int subclass; `float(True)=1.0` no longer passes). |

Unchanged from v2.1: all §8 confirmed validations, the §4 edge cases, strict-JSON (`allow_nan=False`),
the null arithmetic, and the wording discipline. v2.1 kept untouched for the audit trail.

## Verification matrix (Analyzer v2.2) — all green

- **Primary nulls preserved:** pool=20, draw_space=1000, thr=0.01, misses=5 → collapse `1.0204`,
  miss-run `5.3146` (unchanged).
- **TB §9 case A** (`collapse_threshold=0`): assumed-rate → shape "does not fire", fire_prob 0.0,
  exp_draws `null` / `infinite_never_fires` (was 1.25 in v2.1).
- **TB §9 case B** (`collapse_threshold=1`): assumed-rate → fire_prob 1.0, wait `1.0` draw at rates
  0.2/0.5/0.9 (was 1.25 in v2.1).
- **Provenance:** inside repo → real commit `0c3166a`, `analyzed_tree_dirty=true`; no `--repo-root`
  (authoritative) → FATAL; `--repo-root` non-repo / missing dir → FATAL; `--no-provenance` →
  `authoritative=false`, commit null, succeeds.
- **Boolean threshold:** JSON `hit_rate_collapse_threshold=true` → FATAL in v2.2; negative control
  confirms v2.1 accepted it as 1.0 (rc=0).
- **§8 regression guards** (pool≤0, pool>draw_space, draw_space≤0, misses<1, thr>1, fire_horizon≤0,
  non-integer misses 5.5) all still FATAL; degenerate assumed miss-run: hit0→5.0 @ draw 5,
  hit1→`null`/`infinite_never_fires`; strict JSON parses with no `Infinity`/`NaN` tokens.
- **Authoritative findings** written to `watcher_kpi_metricC_v2_2_findings.json` (live
  `watcher_policies.json`, `--repo-root .`); strict-parse OK; six provenance fields present.

## Concurrent-session observations

- The S172 lane was active on this tree throughout. Observed mid-session edits to
  `tests/test_s172_phase5_d0.py` (added B3/B4 REV3 gates + `run_trial_miner`/`socket` imports) and a
  dirty working tree (git status porcelain non-empty; reflected as `analyzed_tree_dirty=true` in the
  findings). **None of the S172 files overlap this lane's write-set;** all runtime files here were
  read-only. No churn observed in the Chapter-13 / WATCHER / KPI files traced.

## Stop

Delivered the three files (+ findings artifact) for Michael → Team Beta. No implementation, no
policy edits, no walk-forward, no commit/push. Awaiting TB binding verification of v1.2 + v2.2.
