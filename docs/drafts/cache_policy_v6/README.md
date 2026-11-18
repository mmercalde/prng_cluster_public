# Cache policy v6 (DRAFT, nothing landed): fresh per-run per-worker CuPy cache for counted runs

v6 fixes X-CACHE-POLICY R2 items 1–3 (`docs/exchange/from_beta/X-CACHE-POLICY_R2_BETA_REVIEW.md`) and applies
the theoretical item 4. Items 5 and 6 are recorded below as first-counted-run checklist entries. Thread:
**X-CACHE-POLICY**, R3 (the final round). The base is v5 (`docs/drafts/cache_policy_v5/`). Everything Beta
accepted in R1 and R2 is unchanged: **the three diffs and `cache_records_verify.py` are byte-identical to v5.**

**Nothing here is landed.**
- `scripts/launch_fleet_manual.sh`, `gate12_launch.sh` and `/home/michael/run4_launch.sh` are not edited.
- Nothing ran on a rig or a GPU, and no real ssh was used.

## Files

| File | sha256 | v6 vs v5 | Applies to / lands as |
|---|---|---|---|
| `launch_fleet_manual.diff` | `331d0830026b190f34165fba38080b713b4afc88cea1359b1e2e1e921c0f8ef9` | unchanged | `scripts/launch_fleet_manual.sh` @ `048fd47` (`21ff9399…`) |
| `gate12_launch.diff` | `5e2f7afd7954e65769dbb1fc805f868fb3bc6a4040dd9202aa72801d45e1819c` | unchanged | `gate12_launch.sh` @ `048fd47` (`ec9ae1e9…`) |
| `run4_launch.diff` | `cc0d349128c34015322f880af403a53b9444f762c74c2a86d5c107851cc4defe` | unchanged | Route A `/home/michael/run4_launch.sh` (outside the repo, `9cd3aebf…`) |
| `scripts/cache_records_verify.py` | `774161853209fbd8ece40b61f12c4ecef4eac2a9fc1cd0f0e69b7c4d3021c3d8` | unchanged | new `scripts/cache_records_verify.py` |
| `scripts/gate12_cache_join_gate.py` | `e2d323ce4a13b9dbcdd901bfbc869bb4adbf4796a083e2c393065fd11c51ce8d` | item 4 | new `scripts/gate12_cache_join_gate.py` |
| `scripts/cache_footprint.py` | `1673a8155b184b7152f80619a1f825548f8a859b76f10eb7f52458b4c045339b` | item 2 | new `scripts/cache_footprint.py` (post-run, read-only) |
| `tests/test_cache_policy_v6.py` | `d4808a528a1288334c1924eb2cbfcc14b974caeb7f73da64ecbbc6d9ad3590dd` | item 1 + new cases | `tests/test_cache_policy_v6.py` |
| `tests/fixtures/run4_launch.sh` | `5ca23e3ca47f16ee5ce1cc3b8d75d4ee7bee545f13b25c3e25db94887901c1b6` | unchanged | `tests/fixtures/run4_launch.sh` (no name clash with the existing `tests/fixtures/`) |

## Results (2026-11-18, VM101, CPU only)

- **Pre-landing**, run from the worktree root with `python3 -B docs/drafts/cache_policy_v6/tests/test_cache_policy_v6.py`:
  **112/112, in 3 consecutive runs.**
- **Landed, installed under `tests/`** (R2 item 1). This used an ordinary writable checkout at
  `logs/landed_v6_*`, a `git init` with the patched launcher and `gate12_launch.sh`, the three scripts under
  `scripts/`, and the test and fixture under `tests/`. Route A was the patched baseline copied outside that tree.
  From the checkout root it ran
  `python3 -B tests/test_cache_policy_v6.py --landed <checkout> --route-a <checkout>/ext/run4_launch.sh`:
  **111/111.**
  - Scratch was created at `<checkout>/logs/cachev6_*`.
  - Neither `/home/michael/tfm_worktrees/logs` nor `/home/logs` exists afterwards.
  - The one check fewer is the pre-landing fixture-equals-baseline check.
- **Mutants**, `… mutants`: **22/22 detected.** v5's 19 plus three new ones: J7 (encoder status discarded),
  J8 (environ filter rc 2 tolerated) and F2 (footprint timeout not caught).
- **Cleanup:** after every run there were 0 leftover fixtures, 0 stand-ins and 0 stub sleeps.

## R2 → v6

| R2 item | v6 | Test(s) |
|---|---|---|
| **1. Scratch path independent of the draft layout; rerun installed under `tests/`** (blocked landing) | `--scratch-root DIR`, default `./logs` of the **current directory**, created with `mkdir -p`; fixtures go in a `cachev6_*` temporary directory there. The draft folder and the repository root are computed from the file's position **only in pre-landing mode**, where the diffs are needed; `--landed` never uses them. The test prints its scratch path and mode at start | The landed run above, with the test installed under `tests/` in a separate checkout |
| **2. Footprint: SSH timeout per row** | `--timeout-s` (default 30). `subprocess.TimeoutExpired` is caught inside the per-row measurement. The row gets `status: TIMEOUT` and a `reason` naming endpoint, path and timeout. The remaining rows are still measured, the JSON is written with verdict `INCOMPLETE`, and the exit is 1 | U-FOOT: one CT's transport stalls (`exec sleep 60`, `--timeout-s 2`) and the other is good. Result: good row `OK` with bytes equal to local `du`, stalled row `TIMEOUT` with endpoint and path, `INCOMPLETE`, exit 1. Mutant F2 |
| **3. README vs provenance** | **I chose the smaller change: correct the README.** v5's README said `cache_footprint.py` was "added to the hashed files"; the entry-point diffs never did that, and v6 leaves the diffs unchanged. Provenance hashes the entry point and `launch_fleet_manual.sh`, `cache_records_verify.py`, `gate12_cache_join_gate.py`, `gate12_worker_liveness_gate.py` and `gate12_sentinel_gate.py`, plus git HEAD (which pins the committed `cache_footprint.py` bytes, as Beta notes), the dirty-line count, policy, slots and nonce. The footprint script runs after the run, outside the launch, so it has no per-launch digest | — (no code change) |
| **4. (THEORETICAL) Keep the encoder status; filter on the CT** | Each probe section captures `PIPESTATUS` as an array: an encoder exit ≠ 0 makes the section rc 90 whatever the reader returned. environ is now `LC_ALL=C grep -z -e '^CUPY_CACHE_DIR=' -- /proc/<pid>/environ` on the CT: NUL-framed whole entries, only those starting with the name. grep rc 1 means no such entry, which the exactly-one check refuses; rc 2 and rc 90 refuse as read failures. The Python NUL split and exactly-one, byte-equal comparison are unchanged | U-JOIN: base64 exits 1 after valid output → refused; the CT filter's grep exits 2 after printing the correct entry → refused; all v5 environ cases (missing, name inside another value, embedded newline, duplicate, suffix, non-UTF-8 + correct) still behave as before. Mutants J7, J8 |

## First counted run: checklist entries (R2 items 5 and 6; with R1's still-owed items)

These are **operational**, for Michael's first authorised counted run. None is satisfied by the CPU tests.

1. **Clean committed harness.** Land, commit, deploy as the existing process requires. This package adds no
   file to the ten-file CT parity list; if governed worker bytes change in the same commit, the committed-HEAD
   deployment and 30/30 verification apply.
2. **Installed Route A: cache-gate coverage.**
   `python3 -B tests/test_cache_policy_v6.py --landed <repo> --route-a /home/michael/run4_launch.sh` from the
   repo root. It must be green.
3. **Installed Route A: parameters, separately (R2 item 5).** The landed test reads only Route A's syntax and
   cache-gate slice; Beta showed `warm_start_offset` 25 → 26 still passes. So before the launch:
   - (a) record `sha256sum /home/michael/run4_launch.sh`;
   - (b) `diff` it against `tests/fixtures/run4_launch.sh` (the reviewed patched fixture), which must show only
     the approved `--params` block;
   - (c) check that block value by value against `docs/RUN4_LAUNCH_CHECKLIST.md` (the `RUN4_PARAMS` line at
     `:320`, rules at `:306`, `:376`);
   - (d) run the checklist's own Route A pre-launch provenance check.
4. **GPU and parity gates** as normal.
5. **Evidence.** 24 verified rig rows in `cache_records_<nonce>.tsv`, a passing join (24 joined, Zeus as local)
   and `sha256sum -c SHA256SUMS` passing in `logs/gate12_<stamp>_cache_freeze/`, which includes
   `gate12_<stamp>_cache_sentinels.tsv`, before the sampler and coordinator. Verify the freeze before
   accepting the result.
6. **Post-run footprint.** After the workers exit, run the printed `scripts/cache_footprint.py` command. The
   JSON goes beside the freeze (not in `SHA256SUMS`) and must say `COMPLETE`; `INCOMPLETE` is reported as
   such.
7. **Timing report (R2 item 6).** In the first run's timing report, give:
   - per-slot `startup_s` (sentinel − `t_prep`, from the frozen sentinels TSV);
   - the configured `STAGGER`;
   - the time from first parked worker to actual release;
   - **the cache join and freeze duration** (24 serial SSH probes, 20 s timeout each, so up to 480 s in the worst
     case);
   - the maximum `SESSION_RELEASED.waited_s` across the fleet, against the 900 s per-worker parking deadline;
   - any `SESSION_RELEASE_TIMEOUT`.

   Live headroom is unknown until then.

## Not done, stated plainly

- **Not run:**
  - the full `gate12_launch.sh` / Route A end to end (real clean-tree, GPU, parity, sentinel, liveness,
    coordinator);
  - anything on the rigs, which are off.
- **The worst-case join time.** 24 × 20 s = 480 s, inside the 900 s parking deadline, but it adds to the
  sentinel and liveness gates. It is measured on the first run (item 7), not shortened here.
- **Zeus local cache:** unchanged, `/tmp/cupy_cache_local_gpu<N>`, shared across runs (accepted in R1).
