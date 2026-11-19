# PWC fresh CuPy cache per run, v1 (DRAFT, nothing landed)

**Prior record:** LEADS L-7 (warm cache faults, cold runs clean). The v7 launcher rule is in `9034e18`, and the owner extended it to PWC on 2026-11-18. NEW: this patch. Not searched: ser8, the private repository, the rigs (off).

**Files.** `persistent_worker_coordinator.diff` (`2c5238d8…`) applies to main `fe763fe` (`fc89a373…`) with `--fuzz=0`; `tests/test_pwc_cache_reset.py` (`8e0beea7…`) lands at that path.
- Not touched: `persistent/pwc_worker_service.py:305`, a `setdefault` that never fires on the rigs because both start paths set `CUPY_CACHE_DIR`.

**Governed.** `persistent_worker_coordinator.py` is not in the parity gate's file list (`scripts/gate12_parity_gate.py:175-184`). It is coordinator code, which is a Beta before-commit review class (CLAUDE.md "Beta reviews"); it has not been reviewed.

**Mechanism.** Both rig start paths give worker N `CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_<N>`; the later assignment overrides the `${HOME}/.cache/cupy` default in `ROCM_ENV_VARS`. One ssh call per rig, before any worker starts:
`for g in 0..pool-1; rm -rf || exit 3; mkdir || exit 4; n=$(ls -A) || exit 5; [ -z "$n" ] || exit 5; echo PWC_CACHE_RESET_OK`
- The rig is accepted only if the exit status is 0 **and** the last output line is the marker. An exception is a refusal.
- **TCP path:** the reset runs after the stale-worker kill. A refused rig is skipped, so no script is written, nothing is launched, and the online and ready barriers do not count it.
- **SSH-PWC path:** the reset runs after the S156 reap, before the first spawn. A refused rig gets quarantined handles, the same state as a failed spawn.
- **Not changed:** Zeus's local `/tmp/cupy_cache_zeus_gpu<N>`, the same as v7 left Zeus.

**Results** (2026-11-18, VM101 CPU). The simulated landed tree is a `git clone --shared` of main checked out at `fe763fe`, under `logs/pwc_cache_v1/tree`, with nothing committed. The six existing tests that touch the file ran before and after the patch:

| Test | Before | After |
|---|---|---|
| `test_s149_rocr_isolation_harness.py` | rc 1, 5 failed | identical |
| `test_persistent_worker_harness.py` | rc 1, 21/25 (T13, T14, T19, T20) | identical apart from a MagicMock id |
| `tests/test_s172_threshold_propagation.py` | rc 1, gates 3/5 | identical |
| `tests/test_s172_phase5_d3_25_candidate_ingress.py` | 13/13 | 13/13 |
| `tests/test_s172_resolved_execution_set.py` | 34/34 | 34/34 |
| `tests/test_s172_phase4_coordinator.py` | 63/63 | 63/63 * |

\* Gate 22 reds on any untracked `.py`, including the new test file. Run with that file set aside, the suite is 63/63; once the file is committed, Gate 22 no longer counts it.
- **New test**, run from the landed location as `python3 -u tests/test_pwc_cache_reset.py`: **21/21, rc 0**.
- **Mutants: 5/5 caught.** TCP call removed (4 failed), SSH call removed (3), exit status ignored (1), marker ignored (2), `ls` status ignored (1). Not run: a real rig (off) or real ssh.

## Known limit (recorded at landing, 2026-11-19)

Beta X-PWC-CACHE-RESET R1 raised, and R2 accepted as a recorded limit: on the TCP path a stale
worker from an earlier coordinator can reconnect and be served with a warm cache if BOTH the
kill and the cache reset for that rig fail (for example the rig refuses SSH while its TCP link
still works). The same gap exists on main without this patch. PWC is the backup path and is not
in use. TO-DO if PWC is ever used: close the reconnect gap, and run one warm-cache versus
fresh-cache PWC test. Review record: docs/exchange/from_beta/X-PWC-CACHE-RESET_R2_BETA_REVIEW.md.
