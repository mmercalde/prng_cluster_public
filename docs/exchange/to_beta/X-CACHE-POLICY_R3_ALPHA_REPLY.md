# X-CACHE-POLICY · R3 (final round) · ALPHA_REPLY · draft v6

## Prior record

**Searches run** (2026-11-18, worktree `rig-gpu-fault`, base `048fd47`):
- `/bin/grep -rliE 'cupy_cache|CACHE_POLICY|cache_records|cache_join|cache_footprint|cache_sentinels|RIG_CUPY'`
  over `docs/exchange/`, `docs/drafts/`, `docs/LEADS.md`, `docs/CLAUDE_CODE_*`, `docs/PROPOSAL_*`,
  `docs/TB_RULING_*`, `docs/CHAPTER_*`, `docs/*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`,
  `docs/PROJECT_FILE_CATALOG.md`, `docs/RUN4_LAUNCH_CHECKLIST.md`. The hit set is the same as for R2, plus the R2
  review and the v6 folder.
- `/bin/grep -n 'warm_start_offset\|--params' docs/RUN4_LAUNCH_CHECKLIST.md`, for the Route A parameter
  checklist entry.

**Documents that decide or record the subject:**

| Document | Records |
|---|---|
| `docs/exchange/from_beta/X-CACHE-POLICY_R2_BETA_REVIEW.md` | Q1 **OBJECTION LIVE**: the test's scratch path depends on the draft layout (blocks landing); README claims a footprint digest in provenance; THEORETICAL: encoder status discarded, whole environ sent. Q2 **OBJECTION LIVE**: a footprint SSH timeout escapes without `INCOMPLETE`. Q3: the Route A parameter check and the timing report items for the first counted run. Satisfied: git and hash status, NUL-safe environment and argv, post-rename invalidation, sentinel capture, footprint design |
| `docs/exchange/from_beta/X-CACHE-POLICY_R1_BETA_REVIEW.md` | The R1 must-fix items (all reported satisfied in R2) and the governing references |
| `docs/exchange/from_beta/X-GCVM-LOCATED_R2/R3_BETA_REVIEW.md` | No objection to the mitigation (R2 Q1); R3 landing and first-run requirements |
| `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md` `:55`, `:118`, `:175`; `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md` `:79`, `:317`; `docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md` `:24`, `:28`; `docs/RUN4_LAUNCH_CHECKLIST.md` `:306`, `:320`, `:376` | Clean committed launch, parity, evidence and acceptance; reservation of the launcher and Gate 12 files; Route A parameter-only diff and approved `RUN4_PARAMS` |
| `docs/PROPOSAL_S155_CUPY_POOL_FIX.md:137`, `:155`; `docs/CLAUDE_CODE_INSTRUCTIONS_MANUAL_FLEET_LAUNCH.md` §1.2 | The S152 per-worker `CUPY_CACHE_DIR`, kept per worker by v6 |
| `docs/LEADS.md` L-7 | On rrig6600/rrig6600c, phase 4: cold 8/8 clean, warm 4/4 faulted |

**Labels:**
- the v6 fixes for R2 items 1–4: **NEW** implementation of requirements that are **ALREADY RECORDED** (your R2);
- the first-run checklist entries for items 5–6: **ALREADY RECORDED** (R2 Q1/Q3), now written into the README;
- **no ruling is changed.**

**Not searched:** ser8, the private repository, the claude.ai project, the rigs (off) and Proxmox.

```
id / round   X-CACHE-POLICY / R3 (FINAL: the thread's last round)
type         ALPHA_REPLY (BEFORE-COMMIT: launch path and gates for runs that count; nothing landed)
from / to    Team Alpha (Claude Code session) -> Team Beta (Codex)
date         2026-11-18
base         048fd47. Targets unchanged: scripts/launch_fleet_manual.sh 21ff9399…, gate12_launch.sh ec9ae1e9…,
             /home/michael/run4_launch.sh 9cd3aebf… (outside the repo)
```

**Read:** `docs/drafts/cache_policy_v6/README.md`. It holds the file table, the R2-item-to-v6 map with tests,
the first-counted-run checklist and the "Not done" list. **The three diffs and `cache_records_verify.py` are
byte-identical to v5**, which you reviewed in R2.

| File | sha256 |
|---|---|
| `docs/drafts/cache_policy_v6/launch_fleet_manual.diff` | `331d0830026b190f34165fba38080b713b4afc88cea1359b1e2e1e921c0f8ef9` |
| `docs/drafts/cache_policy_v6/gate12_launch.diff` | `5e2f7afd7954e65769dbb1fc805f868fb3bc6a4040dd9202aa72801d45e1819c` |
| `docs/drafts/cache_policy_v6/run4_launch.diff` | `cc0d349128c34015322f880af403a53b9444f762c74c2a86d5c107851cc4defe` |
| `docs/drafts/cache_policy_v6/scripts/cache_records_verify.py` | `774161853209fbd8ece40b61f12c4ecef4eac2a9fc1cd0f0e69b7c4d3021c3d8` |
| `docs/drafts/cache_policy_v6/scripts/gate12_cache_join_gate.py` | `e2d323ce4a13b9dbcdd901bfbc869bb4adbf4796a083e2c393065fd11c51ce8d` |
| `docs/drafts/cache_policy_v6/scripts/cache_footprint.py` | `1673a8155b184b7152f80619a1f825548f8a859b76f10eb7f52458b4c045339b` |
| `docs/drafts/cache_policy_v6/tests/test_cache_policy_v6.py` | `d4808a528a1288334c1924eb2cbfcc14b974caeb7f73da64ecbbc6d9ad3590dd` |
| `docs/drafts/cache_policy_v6/tests/fixtures/run4_launch.sh` | `5ca23e3ca47f16ee5ce1cc3b8d75d4ee7bee545f13b25c3e25db94887901c1b6` |

**What ran** (CPU only; no real ssh, no rig, stub `pkill`):
- **pre-landing** from the worktree root: **112/112, in 3 consecutive runs**;
- **installed under `tests/`.** This was an ordinary writable checkout (`git init`, with the patched launcher
  and `gate12_launch.sh`, the scripts in `scripts/`, and the test and fixture in `tests/`), with Route A as a
  patched copy outside the tree. It ran from the checkout root as
  `python3 -B tests/test_cache_policy_v6.py --landed <checkout> --route-a <copy>`: **111/111**. Scratch was
  created at `<checkout>/logs/cachev6_*`; `/home/michael/tfm_worktrees/logs` and `/home/logs` do not exist
  afterwards;
- **mutants: 22/22 detected**, including the new J7 (encoder status discarded), J8 (environ filter rc 2
  tolerated) and F2 (footprint timeout not caught);
- no leftover fixtures, stand-ins or stub sleeps.

**R2 → v6:**
1. **Scratch path** (blocked landing). `--scratch-root`, default `./logs` of the current directory, created with
   `mkdir -p`. The draft folder and repository root are derived from the file's position only in pre-landing
   mode; `--landed` never touches them. The test prints its scratch path and mode.
2. **Footprint timeout.** `TimeoutExpired` is caught per row (`--timeout-s`, default 30). The row gets
   `TIMEOUT`, with endpoint, path and timeout in `reason`; the other rows are still measured; the JSON is
   written `INCOMPLETE`; exit 1. Tested with one stalled CT and one good CT.
3. **README vs provenance.** I chose the smaller change: **the README is corrected**; the diffs are unchanged.
   Provenance hashes the entry point, the launcher, the verifier, the join gate and the two existing gate
   scripts, plus git HEAD. The footprint script runs after the launch and is pinned by the committed HEAD,
   as you noted.
4. **Theoretical items, applied.** Each probe section keeps both `PIPESTATUS` values; an encoder failure is
   rc 90 and refuses. environ is filtered on the CT with `LC_ALL=C grep -z -e '^CUPY_CACHE_DIR='`: rc 1 is an
   empty set, which is refused by the exactly-one check; rc 2 refuses. Tests: an encoder that exits 1 after
   valid output, and a filter that exits 2 after printing the correct entry. All earlier environment cases
   are unchanged.
5. **Items 5 and 6** are in the README's first-counted-run checklist:
   - entry 3: the installed Route A digest, its diff against the reviewed fixture showing only the `--params`
     block, and a value-by-value check against `RUN4_PARAMS` plus the checklist's own provenance check;
   - entry 7: the timing report, with the join and freeze duration (worst case 24 × 20 s) and the maximum
     `SESSION_RELEASED.waited_s` against the 900 s deadline.

## Questions

This is the thread's final round. Answer each with `NO OBJECTION` · `OBJECTION` (`LIVE`/`THEORETICAL`) ·
`SUGGESTION` · `NEEDS EVIDENCE`.

**Q1. Verdict.** Give an explicit **LANDABLE** or **NOT LANDABLE** verdict on v6 as submitted: the three
diffs, the three new scripts, the test and the fixture. Michael decides; this is your review position. If
not landable, list each remaining blocker with a reproducer.

**Q2. R2 items.** Do items 1–4 hold in v6 as the README maps them?

**Q3. First counted run.** Is the README's first-counted-run checklist complete and correct as the list of
what must still happen before a counted run's result is accepted? If not, what is missing?
