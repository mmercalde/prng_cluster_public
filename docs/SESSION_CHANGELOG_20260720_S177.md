# Session Changelog — 2026-07-20 — S177 (S176 TB-ruling follow-up, Items 1–3)

**Session:** S177
**Author:** Team Alpha (Claude)
**Track:** WATCHER KPI Validation — TB ruling follow-up (`CLAUDE_CODE_BRIEF_S176_FOLLOWUP_v1.md`)
**Ruling:** `docs/TB_RULING_S176_WATCHER_KPI.md`
**Status:** Recommend-only. No commits, no policy edits, no config-bug fixes, no cluster runs. Delivered to Michael → Team Beta.

---

## Summary

Executed the three work items the TB ruling unblocks. Item 1 (two verifications)
was reviewed and accepted by Michael before flowing into the proposal: D1/D2 are
RESOLVED on `main` (removed from the defect list; stale map logged as separate P3),
and `minimum_hit_rate` deprecation precondition is met (zero runtime consumers). Item
2 produced a null-stated deterministic analyzer v2 with all twelve TB §8 corrections.
Item 3 drafted the BOOTSTRAP→CALIBRATING→GOVERNED governance-states proposal.

## Item 1 — verifications (accepted by Michael)

- **1a — D1/D2: RESOLVED.** Working tree `HEAD == origin/main == 0c3166a`; `git diff
  0c3166a` empty and status clean for all relevant files (no uncommitted divergence).
  Authoritative output gate is manifest-driven and correct: `check_output_freshness()`
  (`watcher_agent.py:419`) → `manifest.get("primary_output")` (`:406`); Step 3
  `full_scoring.json:12` → `survivors_with_scores.json`; Step 4 `ml_meta.json:10` →
  `reinforcement_engine_config.json`; Step 4 gate is file-existence, not R². Both
  defects removed from the P1 list.
  - **P3 cleanup logged separately (outside KPI work):** the stale hardcoded
    `step_files` map in `_find_results()` (`watcher_agent.py:1317-1322`) still lists
    `full_scoring_results.json` / `optimal_ml_config.json`, but it is a non-gating,
    None-tolerant telemetry loader (caller `:1289` falls through to success), so not a
    defect — recommend aligning it to the manifests or replacing with a manifest lookup.
- **1b — `minimum_hit_rate`: zero runtime consumers confirmed on `0c3166a`.** Sole
  non-doc/non-artifact occurrence is the config definition `watcher_policies.json:74`.
  Deprecation precondition met (TB Q3 / §2.6).

## ⚠️ Methodology note — the tool `grep` honors `.gitignore` (silent `*.json` skip)

The `grep` provided in this environment is a **`ugrep` wrapper invoked with
`--ignore-files`**, so it respects `.gitignore`. `.gitignore:41` is `*.json` (only
`config_*.json` / `*_config.json` / `schema_*.json` re-included). Consequently every
"repo-wide" grep run through the tool **silently skipped almost all `.json` files**,
including `watcher_policies.json` and the manifests. The 1b search was redone with
**`/bin/grep`** (no ignore) to become genuinely definitive. Retroactive correction:
the S176 "grep definitive" phrasing was valid for `.py` (never ignored) but overstated
for `.json` coverage; the conclusion (no `.json` consumer beyond the definition) still
holds under `/bin/grep`. **Guidance for future sessions: use `/bin/grep` (or
`command grep`) for any completeness-critical `.json` search on this box.**

## Item 2 — deterministic analyzer v2 (TB §8, all twelve points)

`watcher_kpi_metricC_deterministic_v2.py` (v1 kept untouched for the audit trail),
run against real `watcher_policies.json` → `watcher_kpi_metricC_v2_findings.json`.
Verified: fail-loud with no policy source; structural validation (`pool_size>0`,
`draw_space>0`, `max_misses>=1`); at the uniform null (Hit@20 = 0.02) collapse
expected-draws-to-first-false-fire = 1.02 and 5-miss-run = 5.31, both ≤ the
documented 1000-draw false-alarm horizon → `FIRES-WITHIN-HORIZON-AT-NULL`.

**v1 → v2 diff (5 lines):**
1. Policy values (`collapse_threshold`, `max_misses`) now read from `--policies` or
   explicit flags with **fail-loud** if neither — no hardcoded `0.01/20/5/0.05`.
2. Verdict criterion replaced `mean_gap > max_misses` with a defined
   **false-alarm-horizon / expected-waiting-time** test (`--false-alarm-horizon`).
3. `chance_hit_probability` → `uniform_null_hit_probability`; all findings stated **at
   the uniform null**, no "healthy TFM" wording; `--assumed-healthy-hit-rate` kept as a
   clearly-separated optional sensitivity input.
4. Added structural validators (§8.1–8.3) and an explicit **unique-pool-size assumption**
   report (§8.9).
5. Output now states only **two** live triggers consume metric C and `minimum_hit_rate`
   is a configured **target** not a trigger (§8.11), and frames Metric-A/Metric-C as
   **complementary views of one Bernoulli event**, not independent evidence (§8.12).

## Item 3 — governance-states proposal (document only)

`docs/PROPOSAL_WATCHER_KPI_GOVERNANCE_STATES_v1_0.md`: three-state model with per-trigger
transitions; `kpi_governance` schema separating prospective-outcome (Hit@K/Lift@K) from
pool-structure metrics (weight shares, unique count, breadth, entropy, effective size,
duplicates, stability); audit-only trigger mode (dispatch-site state gate; structural
gates mapped to existing checks with a flagged GAP for schema/finite/normalized);
per-draw KPI recording plumbing hooked at `generate_diagnostics()` + trigger-history
writer; `minimum_hit_rate` deprecation path; a metric-name uniqueness audit flagging
**"hit rate"** and **"coverage"** as overloaded; explicit non-scope (Phase C deferred
post-S172-Phase-7, no thresholds, no autonomous enforcement).

## Files delivered (for Michael → Team Beta)

- `watcher_kpi_metricC_deterministic_v2.py` — v2 analyzer (CPU, read-only)
- `watcher_kpi_metricC_v2_findings.json` — v2 run against real policy
- `docs/PROPOSAL_WATCHER_KPI_GOVERNANCE_STATES_v1_0.md` — governance-states proposal
- `docs/SESSION_CHANGELOG_20260720_S177.md` — this file
- (v1 `watcher_kpi_metricC_deterministic.py` unchanged — audit trail)

## Hard-rule compliance

- No `git commit`/`push`; no `watcher_policies.json` change; no config-bug fix (P3
  logged only); no pipeline/cluster/walk-forward run. All tools CPU-only, read-only
  (write only findings). Live source read + cited before every claim; `/bin/grep` used
  for the definitive `.json` search.

## Fallback parity

`fallback parity: code=[current — HEAD==origin/main==0c3166a], env=[n/a — no dep change; CPU-only read-only session]`

## Not done (deliberately)

- Phase A implementation, Phase C walk-forward, Track A/Task 3 — not started, per brief.
- No thresholds selected; proposal awaits TB code review before any implementation.
