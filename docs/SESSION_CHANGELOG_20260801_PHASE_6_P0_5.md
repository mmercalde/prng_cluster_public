# SESSION_CHANGELOG_20260801_PHASE_6_P0_5.md

**S181 — Phase 6-P0.5: the behavioural cutover.**
Authority: `docs/CLAUDE_CODE_INSTRUCTIONS_PHASE_6_P0_5_IMPLEMENTATION.md` (REV1).
Base: `2042a18` (≥ the brief's `09a7ebc`). Host: VM 101 (`192.168.3.177`) as `michael`,
venv `~/venvs/torch`. **Not committed, not pushed, WATCHER not run — stopped at the gate.**

P0 (`131787d`) created files and changed no running code. P0.5 is the inverse: it makes the
published pointer manifest authoritative. Every behavioural change lands in one deliverable so
the first post-publication distributed certification has one cause to attribute.

---

## Task zero — the contract amendment (Beta, P0 ruling §2)

`docs/PROVISIONING_CONTRACT_AMENDMENT.md` applied to
`docs/RUNTIME_DATASET_PROVISIONING_CONTRACT.md`, all four changes, documentation only:

| edit | change |
|---|---|
| 1 | `**Status:**` block replaced — obligations attributed to **6-P0.5**, with the ratified 6-P0 / 6-P0.5 phase-boundary table |
| 2 | `expected_sha256` marked **SUPERSEDED** as a statically configured value; identity is run-scoped, frozen at run start from the pointer |
| 3 | §3 failure table gains Beta's classification correction — `DatasetProvisioningError(ResidueError)`, chained, path + node |
| §5.1 | new subsection recording the fleet state verified 2026-08-01 |

## Code — the eight required behaviours

| # | requirement | where |
|---|---|---|
| 1 | WATCHER resolves the pointer | `agents/watcher_agent.py:477` (`p05_freeze_dataset`), `:495` (`p05_resolve_dataset_path`), `:543`, `:1473-1527` |
| 2 | one-time run-start freeze (manifest/version, absolute path, sha256, size, record count) | `miner/dataset_authority.py:576` (`freeze_run_dataset`); `window_optimizer.py:1443-1484` |
| 3 | dispatch the absolute immutable path, never the bare alias | `window_optimizer.py:1484`; `agents/watcher_agent.py:1483`; `miner/dataset_authority.py:459` (`resolve_dataset_path`) |
| 4 | fail before first worker dispatch | `miner/dataset_authority.py:904` (`fleet_preflight`); gate runs before `MultiGPUCoordinator` construction and before `Popen` |
| 5 | per-node provisioning + on-target verification | `miner/dataset_authority.py:704` (`verify_node_dataset`), `:828` (`provision_node_dataset`); `scripts/provision_dataset_fleet.py` |
| 6 | run provenance recording the frozen values | `miner/dataset_authority.py:1015` (`write_run_provenance`); `dataset_provenance/*.json` |
| 7 | pointer movement mid-run must not alter that run | the freeze + `miner/range_miner_coordinator.py:85` (`resolve_dataset_sha256`) |
| 8 | validate the pointer names a permitted version-stamped filename | `miner/dataset_authority.py:253` (`validate_version_filename`) |

**§2.1 — the scope defect.** `range_miner_coordinator.py:3499` (now `:3537`) derived
`dataset_sha256` per **trial**. A scrape between two Optuna trials changed the bytes under a
study and every downstream check stayed self-consistent against a different dataset, with no
error anywhere. `serve_trial` (`:3537`) and the assign-payload builder (`:3439`) now resolve
through `resolve_dataset_sha256` (`:85`), which answers from the run-start freeze when the path
is the frozen one and falls back to the exact pre-P0.5 hashing behaviour otherwise — so no
existing caller changes meaning.

**Beta §3 — the exception correction.** `DatasetProvisioningError(ResidueError)` added at
`miner/range_miner_worker.py:523`. `_sha256_file` (`:586`) and `load_residue_window` (`:602`)
chain the original `FileNotFoundError`/`OSError` through `_dataset_absent` (`:575`) and name
the absolute path and the node (`_node_identity`, `:562`). The bare `FileNotFoundError` that
escaped mid-run is closed.

## Fleet

Provisioned through one path, **including `rrig6600`** — a provisioning step that skips a node
it believes correct cannot detect the case it exists for.

| node | before | after |
|---|---|---|
| `rrig6600` `.122` | version file **ABSENT** (held only the alias) | **PASS**, digest re-derived on target |
| `rrig6600b` `.156` | **ABSENT** | **PASS**, digest re-derived on target |
| `rrig6600c` `.164` | **ABSENT** | **PASS**, digest re-derived on target |

The brief recorded `.122` as "matches" — true of `daily3.json`, but the dispatched path after
P0.5 is the **version file**, which `.122` did not have either. All three were absent by the
measure that now matters; the absent-dataset fault-injection control was captured against all
three before provisioning.

## Verification

- **P0.5 harness** `tests/test_s172_phase6_p05_dataset_authority.py` — **33/33** with the
  live-fleet gate, 32/32 without. Every §7 negative path, each fault injected into a tempfile
  publication tree.
- **Non-regression, all green:** D1.1 18/18 · D4 8/8 · D5 24/24 · D6 3.A 9/9 (16 mutants) ·
  **D6-threshold 17/17** · D6.1 15/15 (8 mutants) · **threshold-propagation 5/5** (3/3 mutants) ·
  Chapter1-P0 12/12 · Phase 3 17/17 · **Phase 4 63/63**.

**Two scope tripwires required registration** — both are per-deliverable git-status whitelists,
and both are extended by registering with a rationale, never by loosening the predicate:

- **gate 22** (`tests/test_s172_phase4_coordinator.py:1602`) — the three new `.py` files
  registered. Checked first: the last commit to touch that file was `131787d`, this same
  lineage, so the block was appended to rather than rewritten.
- **`G-MINER-UNCHANGED`** (`tests/test_s172_threshold_propagation.py:536`) — this gate asserts
  the *threshold repair* left `miner/` alone. P0.5 is a different deliverable that necessarily
  changes `miner/`, so its three files are registered. The gate's actual subject is left
  enforced and was **strengthened**: the kernel/executor surface (`sieve_gpu_worker.py`,
  `prng_registry.py`, `pwc_protocol.py`) must still be byte-identical, any *other* `miner/`
  file is still a red, and a new check greps the registered files' diff for threshold tokens
  so the registration is verified rather than trusted.
- Published artifacts byte-unmodified — `daily3.json`, the version file and the pointer all
  re-hashed at the end of the harness (gate 31), not merely asserted.

## §4 — preflight/dispatch divergence

**Closed for the dataset; the underlying pattern is narrowed, not removed.** Preflight
(`get_step_io_from_manifest`) and dispatch both now resolve to the same absolute immutable
path, and an absolute path has no CWD dependence. Proven from `CWD=/tmp`: the dispatched value
is `/home/michael/distributed_prng_analysis/daily3-…json` and resolves, where the old bare
string would have resolved to `/tmp/daily3.json`. The `Popen` at `agents/watcher_agent.py:1948`
still carries no `cwd=`, and two other manifest params (`output`, `trse_context`) are still
relative and therefore still CWD-dependent — out of scope here, and flagged.

## Flagged for Alpha/Beta

1. **Preflight freshness** now compares against the version file's mtime (Aug 1) rather than
   the alias's (Mar 4), so steps 0 and 1 read STALE and will re-execute on the next WATCHER
   launch. Soft, not blocking. Arguably correct under Beta's "append-only does not make prior
   scores valid" ruling; the alternative reintroduces two resolution bases.
2. **Missing provisioning manifest** is recorded `UNAVAILABLE` and the run proceeds. Whether
   its absence should instead hard-fail a miner-backend run is a governance decision the brief
   does not settle.
3. `dataset_provisioning.json` is gitignored per contract §1, so a fresh clone has no fleet
   definition until one is placed.
4. **A `window_optimizer.py` run now verifies every node in the manifest**, so a local
   single-GPU run refuses to start while any rig is down. That is the fail-closed reading of
   requirements 4/5; if Beta wants local smokes unblocked, the knob has to be governed rather
   than assumed.
5. `dataset_provenance/*.json` is never pruned — same class as the D6.3 checkpoint-pruning
   blocker, not addressed here.

## Out of scope, untouched

Hybrid skip wire-in · RandomSampler arm · any new publication · `daily3.json` / version file /
pointer manifest · the split files · the falsy-zero droppers · `.gitignore`.
