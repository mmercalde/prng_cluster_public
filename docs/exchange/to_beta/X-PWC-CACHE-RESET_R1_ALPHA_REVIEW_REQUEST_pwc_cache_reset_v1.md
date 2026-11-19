# X-PWC-CACHE-RESET · R1 · REVIEW_REQUEST · fresh CuPy cache per run on the PWC path, verdict requested

## Prior record

**Searches run** (2026-11-18, worktree `rig-gpu-fault`):
`/bin/grep -rlE 'X-PWC-CACHE|pwc_cache_reset|PWC-CACHE-RESET|cupy_cache_gpu_'` over `docs/exchange/`,
`docs/LEADS.md`, `docs/PROPOSAL_*`, `docs/TB_RULING_*`, `docs/CHAPTER_*`, `docs/*_AUDIT.md`,
`docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/PROJECT_FILE_CATALOG.md`, `docs/drafts/`.
Hits: only this package and `docs/drafts/NOTE_RUN_SCOPE_DECISION_20261118.md` §3.

**Documents that decide or record the subject:**
- `docs/LEADS.md` L-7: phase 4 on rrig6600/rrig6600c, warm cache 4/4 faulted, cold 8/8 clean (miner path).
- Commit `9034e18`: v7 launcher, `RIG_CUPY_CACHE_POLICY=per_run` default for the range miner (X-CACHE-POLICY-D6).
- `docs/drafts/NOTE_RUN_SCOPE_DECISION_20261118.md` §3: PWC's `/tmp/cupy_cache_gpu_<N>` is never cleared per run.

**Labels:** the PWC reset is **NEW** (owner instruction 2026-11-18: give PWC the same fresh-cache rule). No
ruling is changed. **Not searched:** ser8, the private repository, the claude.ai project, the rigs, Proxmox.

```
id / round   X-PWC-CACHE-RESET / R1   (one round only, owner instruction)
type         REVIEW_REQUEST (BEFORE-COMMIT: coordinator code)
from / to    Team Alpha (Claude Code session) -> Team Beta (Codex)
date         2026-11-18
base         main 5605ed3 (docs-only over fe763fe); persistent_worker_coordinator.py blob 36bacb70… is the
             same at both, file sha256 fc89a373…
```

**Read:** `docs/drafts/pwc_cache_reset_v1/README.md` (mechanism, results).

| File | sha256 |
|---|---|
| `docs/drafts/pwc_cache_reset_v1/persistent_worker_coordinator.diff` | `2c5238d85105fd0be107248ac559aeeb2e3dcc3fec37bfacd428b4ea59ac551d` |
| `docs/drafts/pwc_cache_reset_v1/tests/test_pwc_cache_reset.py` (lands at `tests/`) | `8e0beea7f44a1f8c5e9b07946953c648014c7e9738a93a2d652cbe658e83a0a0` |
| `docs/drafts/pwc_cache_reset_v1/README.md` | `e78522303d1624df4e0aae1f391c41e2fd9609e0a1e52ac32d115263f04821e4` |

**What it does.** Before any PWC worker starts on a rig, one ssh call per rig runs, for each GPU in the pool,
`rm -rf` → `mkdir` → `ls -A` emptiness check on `/tmp/cupy_cache_gpu_<N>`, each with its own exit code, then
prints `PWC_CACHE_RESET_OK`. The rig is admitted only on rc 0 **and** the marker as last line; any exception
refuses it. TCP path: after the stale-worker kill; a refused rig gets no script, no launch, and is not counted
in the online/ready barriers. SSH-PWC path: after the S156 reap; a refused rig gets quarantined handles (same
state as a failed spawn). Zeus local workers and `persistent/pwc_worker_service.py` are not changed.

**What ran** (VM101, CPU only, no rig, no real ssh). Simulated landed tree: `git clone --shared` at `fe763fe`
under `logs/pwc_cache_v1/tree`, diff applied with `--fuzz=0`, test installed at `tests/`, nothing committed.

| Existing test touching the file | Before | After |
|---|---|---|
| `test_s149_rocr_isolation_harness.py` | rc 1, 5 failed (**status: already fails on main**) | identical |
| `test_persistent_worker_harness.py` | rc 1, 21/25: T13, T14, T19, T20 (**status: already fails on main**) | identical apart from a MagicMock id |
| `tests/test_s172_threshold_propagation.py` | rc 1, gates 3/5 (**status: already fails on main**) | identical |
| `tests/test_s172_phase5_d3_25_candidate_ingress.py` | 13/13 | 13/13 |
| `tests/test_s172_resolved_execution_set.py` | 34/34 | 34/34 |
| `tests/test_s172_phase4_coordinator.py` | 63/63 | 63/63 with the new test file set aside; its Gate 22 reds on any untracked `.py`, so it reds while the new test is uncommitted |

- **New test** from its landed location, `python3 -u tests/test_pwc_cache_reset.py`: **21/21, rc 0**.
- **Mutants 5/5 caught:** TCP call removed (4 FAIL), SSH call removed (3), exit status ignored (1), marker
  ignored (2), `ls` status ignored (1).
- **Owner observation (reported by the owner, not verified by this session):** rigs that were rebooted rejoined
  PWC and worked.

## Questions

Answer each with `NO OBJECTION` · `OBJECTION` (`LIVE`/`THEORETICAL`) · `SUGGESTION` · `NEEDS EVIDENCE`.
In your reasoning and findings, **label every point either `BLOCKS`** (would break a run or corrupt results on
this three-rig fleet) **or `TO-DO`** (anything else). This is one round only.

**Q1. Verdict.** Give an explicit **LANDABLE** or **NOT LANDABLE** verdict on the diff plus the new test as
submitted. Michael decides; this is your review position. For each `BLOCKS` point give a reproducer.
