# SESSION_CHANGELOG — 2026-08-01 — P0.5 closure condition (Beta Q2)

**Authority:** `docs/CLAUDE_CODE_INSTRUCTIONS_P0_5_Q2_CLOSURE.md` REV1
**Base:** `6bf89ef` (clean). **Not committed, not pushed** — Michael commits.
**Scope:** the single closure condition Beta named at its conditional
acceptance of P0.5 (`d4ff1e4`), and nothing else.

## The defect

A missing provisioning manifest recorded `UNAVAILABLE` and the run proceeded.
Beta: *"Recording `UNAVAILABLE` and proceeding violates the authority
boundary."* Four conditions — manifest **missing · unreadable · invalid ·
empty** — must be fatal for a miner-backed run before any coordinator
construction or dispatch. And `UNAVAILABLE` means *a required verification was
attempted and could not be completed*, so a path that never needed fleet
verification must record `NOT_APPLICABLE` instead of borrowing the word.

## The change

| file:line | change |
|---|---|
| `miner/dataset_authority.py:107-124` | `FLEET_STATUS_PASS / UNAVAILABLE / NOT_APPLICABLE` — the vocabulary, with Beta's definition of each in the docstrings |
| `miner/dataset_authority.py:964-1057` | `load_provisioning_nodes()` hardened: **unreadable** (`OSError`) and **invalid** (unparseable JSON, non-object top level, unsupported schema, non-list `datasets`/`nodes`, unusable node entry) now raise `DatasetProvisioningError`, chained, naming the absolute path. They were fatal before too — as a bare `OSError`/`JSONDecodeError`/`KeyError` escaping outside both callers' `except` clauses |
| `miner/dataset_authority.py:1060-1194` | `resolve_absent_fleet_status()` — the one place that decides what an unusable manifest costs. Raises for `miner_backed` (and for `require_fleet`); `NOT_APPLICABLE` when the caller declares `remote_execution=False`; `UNAVAILABLE` otherwise, including unknown |
| `miner/dataset_authority.py:1196-1262` | `run_start_dataset_gate(..., miner_backed=, remote_execution=)` routes both the missing and empty cases through it, before `write_run_provenance` |
| `window_optimizer.py:1468-1490` | declares `miner_backed=args.use_range_miner` and `remote_execution=True` (both sieve entry points construct `MultiGPUCoordinator`, `:756` / `:1079`) |
| `agents/watcher_agent.py:1490-1515` | the inline absent/empty branches replaced by the shared decision; `remote_execution=None` (unknown) per step |
| `tests/test_s172_phase6_p05_dataset_authority.py` | gates 33-37 (live gate renumbered 38) |
| `tests/test_s172_phase4_coordinator.py:2085-2113` | gate-22 registration rationale for this tranche |

## Verification

`38/38` P0.5 incl. `--fleet` (all three CT100s PASS on target) · Phase 4
`63/63` · threshold-propagation `5/5` · D6 threshold-path `17/17`.

**Gate 34 proves Beta's condition 3 by absence, not by the raise.** The real
`window_optimizer.main()` runs `--use-range-miner` in a child with tripwires on
`MultiGPUCoordinator`, `run_bayesian_optimization`, `run_with_config`,
`fleet_preflight`, `provision_node_dataset`, `write_run_provenance`, every
`subprocess` entry point, `socket.connect/bind`, `os.fork/posix_spawn/system`
and `multiprocessing.Process.start`. Measured: **no tripwire fired · no
descendant PID (read from `/proc`) · no new entry under the repo root ·
`SystemExit(2)` naming the manifest path**. Only the *input* is patched (where
the manifest is looked for); the dataset-authority path is the production one.

**VIR-2 (gate 37):** reverting `resolve_absent_fleet_status` to the pre-closure
behaviour turns both detectors red — gate 33 stops raising, and the reverted
CLI run fires `write_run_provenance` then `run_bayesian_optimization`, i.e. it
records UNAVAILABLE and proceeds to dispatch, exactly the defect Beta named.

**Successful-manifest path unchanged (gate 36):** with a usable manifest the
new decision function is unreachable — replaced by a raiser, the run still
passes — and the provenance is identical field-for-field with `miner_backed`
true and false. No re-certification trigger.

## Explicitly not done

No Q1 local-run bypass. `range_miner_coordinator.py:3714-3737` untouched.
No Q3 bootstrap contract, no Q4 pruning, no Q5 freshness, no skip work.

fallback parity: code=[current], env=[ok] (no dependency change).
