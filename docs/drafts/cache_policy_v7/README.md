# Cache policy v7 (DRAFT, nothing landed): fresh per-run per-worker CuPy cache for counted runs

v7 fixes the X-CACHE-POLICY R3 landing blocker (`docs/exchange/from_beta/X-CACHE-POLICY_R3_BETA_REVIEW.md` Q1):
the package broke the existing D6 suite `tests/test_s172_d6_integration_repair.py`. v7 also fixes the R3 Q2
leftover (draft paths resolved at load time) and folds R3 Q3's three checklist corrections into the
first-counted-run checklist below. Review thread: **X-CACHE-POLICY-D6**, R1 (new; X-CACHE-POLICY used its three
rounds).

**Unchanged from v6, byte for byte:**
- the launcher, `gate12_launch.sh` and Route A diffs;
- the three scripts;
- the Route A fixture.

Beta's R3 found items 2–4 of R2 holding on these files.

**Nothing here is landed.**
- `scripts/launch_fleet_manual.sh`, `gate12_launch.sh`, `/home/michael/run4_launch.sh` and the real
  `tests/test_s172_d6_integration_repair.py` (`8c1c09dc…`) are not edited.
- Nothing ran on a rig or a GPU, and no real ssh was used.

## Files

| File | sha256 | v7 vs v6 | Applies to / lands as |
|---|---|---|---|
| `launch_fleet_manual.diff` | `331d0830026b190f34165fba38080b713b4afc88cea1359b1e2e1e921c0f8ef9` | unchanged | `scripts/launch_fleet_manual.sh` @ `048fd47` (`21ff9399…`) |
| `gate12_launch.diff` | `5e2f7afd7954e65769dbb1fc805f868fb3bc6a4040dd9202aa72801d45e1819c` | unchanged | `gate12_launch.sh` @ `048fd47` (`ec9ae1e9…`) |
| `run4_launch.diff` | `cc0d349128c34015322f880af403a53b9444f762c74c2a86d5c107851cc4defe` | unchanged | Route A `/home/michael/run4_launch.sh` (outside the repo, `9cd3aebf…`) |
| **`test_s172_d6_integration_repair.diff`** | `6b4e35f6da911272e785e28006c18df9f82d561097cdc9081e0c377c1c0a7e67` | **new** | `tests/test_s172_d6_integration_repair.py` @ `048fd47` (`8c1c09dc…`) |
| `d6/test_s172_d6_integration_repair.py` | `b1a2da0573ec3ad93fc5bb6f9991a6c9299f42a513596ef83c60abc520f32850` | new | the D6 file with that diff applied (for reading) |
| `scripts/cache_records_verify.py` | `774161853209fbd8ece40b61f12c4ecef4eac2a9fc1cd0f0e69b7c4d3021c3d8` | unchanged | new `scripts/cache_records_verify.py` |
| `scripts/gate12_cache_join_gate.py` | `e2d323ce4a13b9dbcdd901bfbc869bb4adbf4796a083e2c393065fd11c51ce8d` | unchanged | new `scripts/gate12_cache_join_gate.py` |
| `scripts/cache_footprint.py` | `1673a8155b184b7152f80619a1f825548f8a859b76f10eb7f52458b4c045339b` | unchanged | new `scripts/cache_footprint.py` (post-run, read-only) |
| `tests/test_cache_policy_v7.py` | `d37528e1124e5b131c79d42286b60a0a6d4900c74a5e6fd4423a5f44c881e6ab` | no draft paths at load | `tests/test_cache_policy_v7.py` |
| `tests/fixtures/run4_launch.sh` | `5ca23e3ca47f16ee5ce1cc3b8d75d4ee7bee545f13b25c3e25db94887901c1b6` | unchanged | `tests/fixtures/run4_launch.sh` |

## Results (2026-11-18, VM101, CPU only, real loopback listener)

**The landed tree.** It was a local `git clone --shared` of this repository, checked out at `048fd47` and kept
under the worktree's `logs/`, with no commit made in it. It received the launcher, `gate12_launch.sh` and D6
diffs (`patch --fuzz=0`), the three scripts in `scripts/`, and the v7 test and fixture in `tests/`. Route A was
the installed baseline patched with `run4_launch.diff`, held outside that tree. `TMPDIR` pointed inside the
tree. The full history is present, so D6's pinned anchors are readable.

| Suite | Tree | Result |
|---|---|---|
| D6, `python3 -u tests/test_s172_d6_integration_repair.py` | **base `048fd47`, unpatched** (control) | **82/82** green, PASS |
| D6, unmodified | base + v6 diffs + scripts (Beta's blocker, reproduced here) | **71/82**. 11 FAILED: WS-2a, WS-2e, WS-5, WS-5b, WS-7 (literal), LCH-2, LCH-3, LCH-7, LCH-6b, M5, M6 |
| **D6, repaired (v7 diff)** | **landed v7 tree** | **84/84** green, PASS, **3 of 3 consecutive runs**. 84 = 82 − the 1 literal check + 2 WS-7 path checks + WS-2f |
| **v7 cache suite, `--landed <tree> --route-a <patched Route A>`** | **landed v7 tree** | **111/111** ALL GREEN |
| v7 cache suite, pre-landing | drafts | **112/112**, 3 of 3 consecutive runs |
| v7 cache suite, `mutants` | drafts | **22/22 detected** |
| v7 cache suite installed at **`/tmp/checkout/tests/`** | copy of the landed `scripts/`, `tests/`, `gate12_launch.sh` | `--landed /tmp/checkout --help` rc 0. `--landed /tmp/checkout --route-a …` **111/111**, scratch at `/tmp/checkout/logs/cachev7_*`. Pre-landing mode there exits with a stated reason instead of `IndexError`. The directory was removed afterwards |
| WS-2f teeth | landed v7 tree with the launcher's `shared` path mutated to `/tmp/cupy_cache_gpu0` for every GPU | D6 **82/84**: WS-2f and the WS-7 path check FAILED, as they should |

- **Leftovers:** after every run there were 0 stub workers or stand-ins and 0 temporary directories. The clone
  trees were removed after the runs.
- **Logs:** `logs/cache_policy_v7_evidence/`. The `logs/` directory is git-ignored, so these files are not
  part of the package:
  - `d6_base_048fd47_unpatched.log`;
  - `d6_v6_unrepaired.log`;
  - `d6_v7_landed_run{1,2,3}.log`;
  - `v7_landed.log`.

## The D6 repair (R3 Q1)

| R3 asked | v7 |
|---|---|
| Make the D6 stand-ins exercise the shared diagnostic path | `run_launcher(…, cache_policy="shared")` sets `RIG_CUPY_CACHE_POLICY=shared` for every Part B/E/M arm; `None` leaves it unset. The pinned pre-fix anchors do not read the variable, so it is inert for every RED arm. D6's `STUB_SSH_EXEC` already relocates `/tmp/cupy_cache_gpu<N>`, so the shared path is executed, not only accepted |
| Preserve the historical red controls | Unchanged: WS-0/1d/3, LCH-0/1, M4/M5/M6 and every other arm. Their bytes, anchors and pins are untouched; the diff touches only the docstring, `run_launcher`, WS-2 (adds WS-2f) and WS-7's cache entry |
| Check the per-worker path for that policy (replacing WS-7's literal) | **Static (WS-7):** the launcher's shared branch is `else echo "/tmp/cupy_cache_gpu$1"; fi`, and the remote command carries `CUPY_CACHE_DIR=$WCACHE`. **Executed (WS-2f, new):** while the fleet is parked, each remote worker's own `/proc/<pid>/environ` (PID from its ACK) holds exactly one `CUPY_CACHE_DIR`, equal to its own relocated `cupy_cache_gpu<N>`. The directory exists and the values are distinct per worker. RED when a worker has none, two, another worker's, or a per-run directory |
| Keep the v6 default-policy and counted-path checks | Unchanged in `tests/test_cache_policy_v7.py`: clean `per_run`, exit 14, exit 15, bad record, shared and Zeus controls, both entry points through join and freeze |
| Rerun D6 and the v6 suite on the resulting tree | The table above |

## The R3 Q2 leftover: landed mode resolves no draft paths

v6's README said nothing about the scratch location depends on the file's position. That held, but the draft
folder and repository root were still computed at import, so a shallow install raised `IndexError` before
argument parsing.
- In v7 the only path resolved at import is the file's own directory.
- `draft_paths()` runs only in pre-landing mode, when `--landed` is absent. If the draft layout is absent it
  exits with a stated reason.
- `--landed ROOT` never resolves a draft path, at any depth. Tested from `/tmp/checkout/tests/` above.

## First counted run: checklist (R3 Q3 corrections folded in)

These steps are operational, for Michael's first authorised counted run, and are not covered by the CPU tests.
**The governing document is `docs/RUN4_LAUNCH_CHECKLIST.md`, which is carried forward in full by reference.**
Nothing here replaces any of its steps. The cache-policy additions are marked **[cache]**.

1. **The whole Run 4 checklist, in order, by reference.** That includes:
   - Michael's authorisation and decision before step 13;
   - rigs on in Proxmox (§1);
   - VM101 tree clean with HEAD recorded (§2, `:89`);
   - no workers running (§3);
   - GPU gate 8/8 per rig (§4);
   - the parity clean control (§5);
   - coordinator deploy from committed HEAD (§6);
   - parity gate 30/30 (§7);
   - netconsole re-armed with one proof line per host (§8);
   - the dataset frozen and present (§9);
   - the survivor-volume gate (§10);
   - Michael opening the search, with the pinned open/status check (§11, `:304`, `:337`);
   - the Route A pre-launch check on the committed tree (§12, `:348`);
   - the launch (§13).

   Clean committed launch, GPU and parity admission, and committed-HEAD deployment with 30/30 verification apply
   if governed bytes change (`docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:55`, `:118`, `:175`;
   `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:55`, `:79`, `:317`). This package adds no file to the ten-file
   CT parity list.
2. **[cache] Landed tests on the committed tree**, from the repo root:
   - `python3 -u tests/test_s172_d6_integration_repair.py` must give 84/84;
   - `python3 -B tests/test_cache_policy_v7.py --landed <repo> --route-a /home/michael/run4_launch.sh` must be
     green.
3. **[cache] Route A comparison (corrected per R3 Q3.1).** Compare the installed wrapper with the **landed
   `gate12_launch.sh`**: `diff /home/michael/run4_launch.sh <repo>/gate12_launch.sh` must be limited to the
   `--params` block. Then check that block value by value and type by type against `RUN4_PARAMS`
   (`docs/RUN4_LAUNCH_CHECKLIST.md:320`; 17 values). Keep as separate evidence:
   - the installed wrapper's `sha256sum`;
   - the landed test result from item 2;
   - the checklist's committed-tree Route A pre-launch check (§12, `:348`, `:376`).

   The shipped `tests/fixtures/run4_launch.sh` is **not** the comparison base: it already carries the Run 4
   parameter block.
4. **[cache] Timing sources, arranged before dispatch (R3 Q3.3).**
   - Record the cache join and freeze **start and end times on VM101**.
   - Preserve all **25** workers' nonce-matched park, release and timeout evidence before any later dispatch
     reuses the logs. That includes each rig's remote `/tmp/minerlogs/gpu<N>.log` and Zeus's local log.
   - The pre-release freeze already holds the 24 rig start-up measurements (`cache_sentinels.tsv`) and the
     liveness evidence that includes Zeus. It precedes the release events, so the release-side evidence must
     be collected separately.
5. **[cache] Pre-sampler evidence:**
   - 24 verified rig rows;
   - a passing join (24 joined, Zeus recorded as local);
   - `sha256sum -c SHA256SUMS` passing in `logs/gate12_<stamp>_cache_freeze/`.
6. **[cache] Post-run footprint.** After the workers exit, run the printed `scripts/cache_footprint.py`
   command. **Retain and hash the footprint JSON in the whole-run evidence bundle.** An `INCOMPLETE`
   collection leaves the footprint evidence outstanding and is reported as such.
7. **[cache] Timing report.** Give:
   - per-slot `startup_s`;
   - the configured `STAGGER`;
   - the elapsed time from first park to release;
   - the join and freeze duration;
   - the maximum successful worker-local `SESSION_RELEASED.waited_s` against the 900 s deadline;
   - any `SESSION_RELEASE_TIMEOUT`;
   - any missing worker evidence.

   The 480 s figure (24 × 20 s) is only the SSH timeout budget of the join probes; the freeze, the other gates
   and the coordinator bind also consume parking time. Live timing and footprint are unknown until this run.
8. **Acceptance, by reference (§15, `:510`, `:533`, `:552`; `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:145`).**
   - A completed run and publication.
   - A run-scoped `G-PROD-SHAPE` PASS.
   - Search and coverage confirmation.
   - The hashed whole-run evidence and closure report, which include the verified cache freeze, the 24 rig
     joins with Zeus separate, and the footprint JSON.
   - Evidence review before acceptance, unless Michael decides directly.
   - For a delegated run, the append-only session record.

## Not done, stated plainly

- **Not run:** the full live `gate12_launch.sh` / Route A (real clean-tree, GPU, parity, sentinel, liveness,
  coordinator), and anything on the rigs, which are off.
- **The landed tree was a clone with uncommitted patches, not a commit.** D6's HI checks measure a porcelain
  *delta*, so this matches a pre-commit landing. A post-commit rerun is checklist item 2.
- **Zeus local cache:** unchanged, `/tmp/cupy_cache_local_gpu<N>`, shared across runs (accepted in R1).
