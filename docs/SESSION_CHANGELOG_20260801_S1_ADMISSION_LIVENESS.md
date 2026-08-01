# SESSION_CHANGELOG_20260801_S1 — §4.3 admission-liveness repair (Beta Ruling 1)

**Base:** `f6bd944` on VM 101 (`192.168.3.177`), venv `~/venvs/torch`, as `michael`.
**Authority:** `docs/CLAUDE_CODE_INSTRUCTIONS_ADMISSION_LIVENESS_REPAIR.md` (REV1);
trace in `docs/FLEET_STATE_REQUIREMENTS_v1.md` §4.3.
**Status:** implemented, gated, non-regression clean. **NOT committed** (Claude does not
commit; Michael commits and dual-pushes).

---

## 1. The change

`assign_stripes`, `_dispatch_pending`, `process_lease_expiry` and the stage advance all sat
behind one guard at `miner/range_miner_coordinator.py:3715`
(`len(eligible) >= expected_workers and stage_idx < len(workflow_stages)`), while
`serve_timeout` defaults to `None`. A worker loss that crossed the threshold therefore
stopped lease expiry from being processed at all — the Blocker-3 matrix was unreachable in
exactly the situation it exists for.

The threshold test was **moved, not removed**, and split in two:

| file:line | change |
|---|---|
| `miner/range_miner_coordinator.py:159` | `DEFAULT_WORKER_ADMISSION_TIMEOUT = 180.0` — the PWC readiness window (`persistent_worker_coordinator.py:826`, `:864`) |
| `miner/range_miner_coordinator.py:3558-3580` | resolve + fail-closed validate the window, **first statement region of `serve_trial`**, before the dataset digest / trial context / listen socket. `None`/0/negative/inf → `ValueError` |
| `miner/range_miner_coordinator.py:3673-3674` | `admission_stage_idx` / `admission_started_at` |
| `miner/range_miner_coordinator.py:3806` | outer guard becomes `if stage_idx < len(workflow_stages):` |
| `miner/range_miner_coordinator.py:3809-3846` | **ADMISSION (bounded)** — re-arm keyed on `admission_stage_idx != stage_idx`; on expiry `fail_trial` naming run id, stage, family/phase, expected and eligible counts |
| `miner/range_miner_coordinator.py:3858+` | **MAINTENANCE (unbounded)** — dispatch, lease expiry, stage advance and commit run for an assigned stage regardless of eligible count |
| `miner/range_miner_coordinator.py:4477` | `worker_admission_timeout` into the serve context |
| `miner/__init__.py:20,24` | export the default so the call site imports it |
| `window_optimizer_integration_final.py:74,1273` | one `getattr(coordinator, ...)` passthrough; no literal at the call site |

Unchanged and asserted so by `G-FORBIDDEN-ABSENT`: the Blocker-3 matrix (byte-identical),
`expected_workers` (bound once, from `worker_pool_size`), `worker_pool_size` code sites,
`serve_timeout`'s `None` default.

## 2. Gates — `tests/test_s172_admission_liveness.py`, **16/16 green ×3 runs**

New harness (Phase-4's 63/63 is a pinned figure; it is registered there, not grown).
Beta's six gates + the churn gate + a structural re-arm gate + a forbidden-change gate,
each behavioural scenario also run against an AST-located one-line **mutant** that restores
the outer guard: **5/5 hang gates red under the mutant, healthy control green.**

Live arm passes `serve_timeout=None`, so no wall clock exists that could end a run — a pass
is structurally a terminal decision by the code under test. The harness budget is a FAILURE
signal only.

## 3. Non-regression — all green

D6-threshold **17/17** · threshold-propagation **5/5 gates, 3/3 mutants** ·
Chapter1-P0 **12/12** · P0.5 dataset authority **38/38 with `--fleet`** ·
Phase 4 **63/63** · Phase 3 **17/17** · D0 12/12 · D1.0 8/8 · D1.1 18/18 · D2 7/7 ·
D3.25 13/13 · D3.5 60/60 · D4 8/8 · D5 24/24 · D6 9/9 · D6.1 15/15 · encoding 8/8.

Gate 22 and `G-MINER-UNCHANGED` registered by **appending**; P0.5's strengthening of
`G-MINER-UNCHANGED` (threshold-token grep over registered diffs) is intact and now applies
to a superset of files.

## 4. Finding raised, NOT fixed (out of scope)

A **pre-existing, intermittent stall in the Phase-5 staging admission path**, found while
building the healthy control. In a multi-stage trial where the second stage's first
sub-stripe result arrives seconds after the first stage's staging completed, the second
shard sits at `staging_status='pending'` forever; the stripe stays `staging`, the trial
neither commits nor fails.

**Reproduced at byte-`HEAD` (`f6bd944`), 5 of 6 runs**, in a module built from
`git show HEAD:miner/range_miner_coordinator.py` — i.e. with none of this repair present.
Not caused by, and not in scope for, this deliverable. Needs its own investigation and a
Beta ruling; `_try_admit_locked` (`:2512`) serialises attempts and `_admitted` is retired
only inside `_pump_deferred` (`:2663`), which is driven by events that may all have fired
before the later shard arrives. Whether a real (acking) Phase-5 sink masks it in production
is exactly the open question — the fixture's `_StubSink` never acks.

`fallback parity: code=[not re-checked this session], env=[not re-checked this session]`
