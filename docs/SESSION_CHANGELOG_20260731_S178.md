# SESSION_CHANGELOG_20260731_S178.md

**Chapter 1 remediation, first tranche — P0 items 1–5.**

| field | value |
|---|---|
| Authority | `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_1_P0_CORRECTION.md` REV1 |
| Findings | `docs/CHAPTER_1_AUDIT_v1.md` (`db9782a`), §6 correction list |
| Threshold history | `docs/THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md` — **cited, not re-derived** |
| Base commit | `0c47fe3` (VM 101 `192.168.3.177`, `main`, user `michael`, venv `~/venvs/torch`) |
| Date | 2026-07-31 |
| Type | **3 code changes + 2 documentation corrections.** Not a docs-only pass |
| Committed | **NO.** Held at the gate for Team Alpha review |

---

## 1. Files changed

| file | change |
|---|---|
| `window_optimizer.py` | items 1, 2, 3 — three fail-closed behaviours + single-authority threshold resolution |
| `window_optimizer_integration_final.py` | item 2 adjunct — `strategy_map.get(name, RandomSearch())` → fail closed. **Two hunks only**; `resolve_directional_threshold` and every backend gate byte-identical |
| `docs/CHAPTER_1_WINDOW_OPTIMIZER.md` | items 4, 5 + P0.3/P0.4/P0.5 |
| `scripts/extract_search_bounds_snapshot.py` | **NEW** — programmatic bounds snapshot with `repository_commit` + `configuration_digest` |
| `tests/test_chapter1_p0_corrections.py` | **NEW** — 6 gates + 6 mutants, sentinel `PASS` |
| `tests/test_s172_phase4_coordinator.py` | Gate-22 coexistence allow-list registration + rationale |

Untouched, verified by diff: `miner/`, `sieve_gpu_worker.py`, `prng_registry.py`,
`persistent/pwc_protocol.py`, `persistent_worker_coordinator.py`,
`zmq_sqlite_coordinator.py`, `distributed_config.json`, the two ruled-stale duplicates.

## 2. Item 1 — dead threshold override flags fail closed

`--forward-threshold` / `--reverse-threshold` were declared at
`window_optimizer.py:1063-1066` and never referenced after `parse_args()` (dead dimension
D-4). **Chosen form: fail closed, flags retained in argparse** — see §10 of the report for
the reasoning. Diagnostic `WINDOW_OPTIMIZER_THRESHOLD_OVERRIDE_UNWIRED`, `rc=2`, raised
before the backend mutex and long before `MultiGPUCoordinator` is constructed. The four
conditions under which the flags may return are recorded in-source.

## 3. Item 2 — unsupported strategies fail closed

`--strategy random|grid|evolutionary` abort with `WINDOW_OPTIMIZER_STRATEGY_UNSUPPORTED`,
naming the missing kwargs. The gate is **derived from live `inspect.signature`**, not a
hardcoded list of broken names, so repairing a strategy clears it with no edit. A Bayesian
request with Optuna unavailable now raises `WINDOW_OPTIMIZER_BAYESIAN_UNAVAILABLE` instead
of silently becoming random search. An unrecognised strategy name no longer falls through
to `RandomSearch()`. **Nothing was deleted** (§0.4).

## 4. Item 3 — D-4 metadata reports what executed

The invented `0.72` / `0.81` constants are gone from both sites
(`agent_metadata.suggested_params` and `run_with_config`'s `run_bidirectional_test` call).
Resolution goes through the single `resolve_directional_threshold()` authority. Absent an
authoritative value the field is **omitted**; a value is **never clamped**; `0.0` survives
because `is None` is the sole fallback trigger. New provenance field `executed_thresholds`
separates observed/executed from proposal.

**Behaviour change to declare:** `--config-file` mode previously sieved at `0.72` / `0.81`
whenever the config file carried no thresholds — which `optimal_window_config.json` never
does. It now sieves at the governed `distributed_config.json` default (`0.30` / `0.30`),
the same value the Bayesian route's backends receive.

## 5. Items 4–5 — documentation

- §3.1: skip definition kept **verbatim**; *why skip exists* physical model added with the
  draw-procedures citation; Team Beta's **DEFECT** callout added verbatim; standing rule
  (wire-in, not removal) stated. New §3.1.1 records dead dimensions D-1…D-4 (P0.4).
- §3.2/§4.1/§4.2/§4.3/§10.1: every numeric bound replaced with live authority plus a
  **programmatically extracted** snapshot carrying `generated_at`, `repository_commit` and
  `configuration_digest`. Precedence rule stated; both `_note` provenance fields carried over.
- §7.2.1 added: `resolve_directional_threshold()` documented as an invariant with the
  `3fdf434` → `2389b61` → `8a55a68` regression history (P0.3).
- §2.1/§12.1/§12.3 rewritten: certified NPZ generation named canonical;
  `bidirectional_survivors.json` demoted to summary; forward/reverse marked count-only;
  flat record shape corrected; the non-existent `timestamp` field dropped (P0.5).

**`CHAPTER_1_PATCH_S114.md` was NOT merged and NOT revived.**

## 6. Verification

- New harness `tests/test_chapter1_p0_corrections.py`: **12/12**, sentinel **PASS**.
- Non-regression, 16 suites: green before and after. **D6-threshold 17/17**,
  **threshold-propagation 5/5**, Phase 4 63/63.
- The harness is side-effect free: `train_history.json`, `holdout_history.json` and
  `trse_context.json` are byte-identical before and after a run.

## 7. Not done in this tranche (retained, not waived)

P1/P2 items 6–17 · hybrid skip wire-in · `run_with_config` writing `[]` survivor files ·
`window_optimizer.py` `logger.warning` with no `logging` import · Optuna sampling the
prohibited combined-session mode · `agent_manifests/window_optimizer.json`
`search_strategy.choices` still advertising all four strategies ·
`optimal_window_config.json` still carrying no top-level threshold keys.

**STOP — held at the gate for Team Alpha review. Nothing committed, nothing pushed.**
