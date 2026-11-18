# Cache policy v5 (DRAFT, nothing landed): fresh per-run per-worker CuPy cache for counted runs

v5 answers the four must-fix items in `docs/exchange/from_beta/X-CACHE-POLICY_R1_BETA_REVIEW.md`. It also
puts the two before-first-counted-run items into the freeze. Thread: **X-CACHE-POLICY**, R2. The base is v4
(`docs/drafts/cache_policy_v4/`); everything Beta accepted in R1 is unchanged.

**Nothing here is landed.**
- `scripts/launch_fleet_manual.sh`, `gate12_launch.sh` and `/home/michael/run4_launch.sh` are not edited.
- Nothing ran on a rig or a GPU, and no real ssh was used.

## Files

| File | sha256 | Applies to / lands as |
|---|---|---|
| `launch_fleet_manual.diff` | `331d0830026b190f34165fba38080b713b4afc88cea1359b1e2e1e921c0f8ef9` | `scripts/launch_fleet_manual.sh` @ `048fd47` (`21ff9399…`) |
| `gate12_launch.diff` | `5e2f7afd7954e65769dbb1fc805f868fb3bc6a4040dd9202aa72801d45e1819c` | `gate12_launch.sh` @ `048fd47` (`ec9ae1e9…`) |
| `run4_launch.diff` | `cc0d349128c34015322f880af403a53b9444f762c74c2a86d5c107851cc4defe` | Route A `/home/michael/run4_launch.sh` (outside the repo, `9cd3aebf…`) |
| `scripts/cache_records_verify.py` | `774161853209fbd8ece40b61f12c4ecef4eac2a9fc1cd0f0e69b7c4d3021c3d8` | new `scripts/cache_records_verify.py` |
| `scripts/gate12_cache_join_gate.py` | `9668a479a12a7802ad84cad90b977f36e5d1c3b6a4dd11481f5786b3fa920c6f` | new `scripts/gate12_cache_join_gate.py` |
| `scripts/cache_footprint.py` | `2e13cc037ad53c4d537866f7361874ac0f9f9b6e54635fba9c46adf6a56f1f08` | new `scripts/cache_footprint.py` (post-run, read-only) |
| `tests/test_cache_policy_v5.py` | `c569ff0b542e8dd35c2243076f156c4a0953505ad49027ea5e8df3424ce58214` | tests and mutants (CPU only) |
| `tests/fixtures/run4_launch.sh` | `5ca23e3ca47f16ee5ce1cc3b8d75d4ee7bee545f13b25c3e25db94887901c1b6` | the patched Route A, a local fixture (= `run4_launch.diff` applied to `9cd3aebf…`; checked by the suite) |

## Results (2026-11-17, VM101, CPU only)

- **Pre-landing**, `python3 -B docs/drafts/cache_policy_v5/tests/test_cache_policy_v5.py`: **109/109, in 3
  consecutive runs.** The suite builds a landed-layout tree in a temporary directory from the committed
  files, the diffs and the new scripts.
- **Post-landing**, `… --landed <repo> [--route-a <installed run4_launch.sh>]`: **108/108.** This ran against
  a simulated landed tree: copies of the committed launcher and `gate12_launch.sh` with the diffs applied, the
  three scripts under `scripts/`, and a patched copy of the installed Route A. It has one check fewer because
  the fixture-equals-baseline check only runs pre-landing.
- **Mutants**, `… mutants`: **19/19 detected.** v4's L2 is no longer listed: it is still equivalent, as
  accepted in R1.
- **Cleanup:** fixtures live under the worktree's `logs/cachev5_*` and are removed afterwards. The stand-ins
  are killed by recorded PID. After every run there were 0 leftover fixtures and 0 stand-ins.

## R1 must-fix → v5

| R1 item | v5 | Test(s) |
|---|---|---|
| **1. Status-check both git reads; refuse before dispatch** | Entry points, policy block: `GIT_HEAD=$(git rev-parse --verify HEAD)` must succeed and match `^[0-9a-f]{40}$`. `GIT_STATUS_OUT=$(git status --porcelain)` must succeed, and the dirty count is taken from the captured output. Otherwise `GATE-12 REFUSED`, exit 1, before the fleet call. `prov_hash` now checks `sha256sum`'s own exit status as well as the 64-hex shape. `cache_footprint.py` is added to the hashed files | E2E-ENTRY on **both** entry points: `git_head_fails`, `git_status_fails` and `hash_rc_fails` (a `sha256sum` that prints a valid hash, then exits 1) are each refused before dispatch, with no `dispatch_*.log`. Mutants E1, E2, E3 |
| **2. NUL-safe environment check, exactly one complete value, startup and parked; argv tokens intact; non-UTF-8 handled** | **Startup** (CT post-check, rc 15): `LC_ALL=C grep -zc '^CUPY_CACHE_DIR='` counts the NUL-terminated entries that start with the name. `LC_ALL=C grep -zxcF "CUPY_CACHE_DIR=$WC"` counts entries equal to the whole expected entry. Both counts must be 1 and both greps must return ≤ 1; the environment is never converted to lines. The record prints `env_CUPY_CACHE_DIR=$WC` only after that exact match. **Parked** (join gate): the probe sends cmdline, environ and the sentinel grep as base64 sections, each with its own exit status. The environment is split on NUL as bytes: exactly one entry may start with `CUPY_CACHE_DIR=`, and its value bytes must equal the path. The cmdline is split on NUL: exactly one `--run-nonce` token, and the next token equals the nonce exactly. The liveness `argv` gets the same adjacency check. No text decode happens before comparison | U-STARTUP (the extracted CT fragment on fixture environ files): correct and non-UTF-8+correct pass. Refused: missing, name inside another value, embedded newline, duplicate (same or different value), suffix, unreadable, read error after a correct-looking count. U-JOIN: the same environment cases, plus nonce as a longer token, nonce split across tokens, `--run-nonce` twice, unreadable cmdline/environ, a read error after correct output, transport failing after correct output. E2E: stand-ins exec'd with a wrong, missing, name-inside or multiline value give exit 15; a non-UTF-8 unrelated variable passes. Mutants S1–S3, J1–J4 |
| **3. Invalidate the records TSV if anything fails after the rename** | `publish()` tracks `renamed`. On any exception after `os.replace` (e.g. the directory `fsync`), it removes the official file. If that also fails, it renames it to `<out>.INVALID`; if both fail, it says so. rc 2 every time, with "failure after publication" in stderr | U-VERIFY: a directory-`fsync` failure after the rename gives rc 2 and nothing at the official path. With `unlink` also failing: rc 2, official path absent, `.INVALID` present. Mutant V3 |
| **4. A harness against the landed scripts with a local Route A fixture; fix the SSH stub `/tmp` rewrite** | `--landed ROOT [--route-a PATH]` runs every check against `ROOT/scripts/*.py`, `ROOT/scripts/launch_fleet_manual.sh` and `ROOT/gate12_launch.sh` as installed. Route A comes from `--route-a` or the shipped `tests/fixtures/run4_launch.sh`. The default mode builds the same layout from the diffs. **Stub fix:** the transport shields the fixture's own absolute path before rewriting, and rewrites `/tmp` and the cache root only at the start of a path token, followed by a delimiter. The reverse mapping also decodes and re-encodes the join probe's base64 payloads | Post-landing run 108/108. E2E `stub transport: a fixture path containing '/tmp/'` (the fixture root is under `…/tmp/deep/`) passes clean. The Route A fixture is checked against `run4_launch.diff` applied to the installed baseline |

## Before the first counted run, now in the freeze

| Item | v5 |
|---|---|
| **Nonce- and PID-matched sentinel line with timestamp, per rig slot** | The join gate's probe greps `SESSION_SENTINEL` from the liveness row's absolute `log_path`. Exactly one line must parse as JSON with `run_nonce` == the nonce, `pid` == the parked pid and `worker_id` == the slot; its leading timestamp must parse as ISO-8601 UTC (`%Y-%m-%dT%H:%M:%S.%fZ`). It refuses on none, two or more, another nonce, pid or worker, a bad timestamp, or an unreadable log. On PASS it writes `logs/gate12_<stamp>_cache_sentinels.tsv` (`worker_id endpoint pid run_nonce sentinel_epoch t_prep startup_s sentinel_line`), and the entry points copy it into `logs/gate12_<stamp>_cache_freeze/` under `SHA256SUMS`. `startup_s = sentinel − t_prep` is the start-up measurement. Tests: U-JOIN (7 sentinel cases), E2E over real stand-ins (4 rows, `startup_s ≥ 0`), E2E-ENTRY clean (the sentinels TSV is in the verified freeze). Mutants J5, J6, E5 |
| **Footprint from the TSV paths, apparent vs allocated** | `scripts/cache_footprint.py --cache-tsv <frozen TSV> --out <json>`, run after the workers exit; the entry points print the exact command. For each TSV row, over read-only ssh: `du -s --apparent-size --block-size=1` → `apparent_bytes`, `du -s --block-size=1` → `allocated_bytes` (disk blocks), the entry count, and per CT the number of `cupy_cache_run_*` directories retained beside it. Units are stated in the JSON. Any unmeasured path gives verdict `INCOMPLETE` and exit 1. The output is opened with `"x"`. Tests: U-FOOT, where both figures match local `du`, the labels are present, retained directories are counted, and a missing path gives `INCOMPLETE`. Mutant F1 |

## Not done, stated plainly

- **Not run:**
  - the full `gate12_launch.sh` / Route A end to end, i.e. the real clean-tree, GPU, parity, sentinel and
    liveness gates and the coordinator. The entry-point tests run the policy block through the fleet call, with
    stub sentinel and liveness gates, then the real join gate and freeze up to the sampler boundary;
  - anything on the rigs. The rigs are off.
- **The footprint step is not in either entry point's control flow.** Both end in a live `tail -f`, so it runs
  as a separate post-run command, which they print. Its JSON belongs beside the freeze; it is not part of
  `SHA256SUMS`, because it is written after the run.
- **Zeus local cache:** unchanged, `/tmp/cupy_cache_local_gpu<N>`, shared across runs (accepted in R1).
- **Before the first counted run** (from R1, still owed): the clean committed harness, the installed Route A
  checked against `run4_launch.diff` (the `--landed … --route-a` mode does this), and the normal GPU and
  parity gates.
