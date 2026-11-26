# ADDENDUM to RESULT_SKIP_SWITCH_RIG_RUNS_20261122.md: `/tmp` evidence lost in the Zeus crash (2026-11-22)

- **What this is:** written by Claude (Team Alpha) in `/home/michael/tfm_worktrees/pwc-offset`, uncommitted.
- **The record is not edited.** The landed record (`docs/RESULT_SKIP_SWITCH_RIG_RUNS_20261122.md`, main `41ad509`) is
  byte-identical to this worktree's copy (`cmp`).
- **Reads:** everything below was read from disk after the restart. Nothing was run on a rig or a GPU.

## 1. What was lost

Zeus crashed at about 12:26, and VM101 restarted at 12:30, both lead-reported. `/tmp` was wiped.

**Six `/tmp` paths cited in the record no longer exist** (`test -e`):

| Cited as | Record line | What it was |
|---|---|---|
| `/tmp/skipsw_runbook.md` | `:50` | runbook for runs A and B |
| `/tmp/skipsw_runbook_c.md` | `:50` | runbook for run C, and the source of run D's blocks |
| `/tmp/skd/` | `:6` | the lead's run-D blocks |
| `/tmp/skd/b01.sh` … `b18.sh` | `:53` | those 18 blocks |
| `/tmp/skd/runD_post.log` | `:144` and the run-D section | the lead's post-run-D output |

**Also gone, but not cited by the record:**
- `/tmp/sk` and `/tmp/skc` (the lead's A/B and C block directories);
- `/tmp/skip_land_cmd.sh` and `/tmp/skip_handland.txt` (the landing commands; both were used, as `f916532` and `00e2ee7`);
- this session's scratchpad.

## 2. Where the re-saved copies are

| Copy | Original | Body sha256 | Status |
|---|---|---|---|
| `docs/drafts/miner_skip_switch_v1/RUNBOOK_RIG_RUNS_A_B.md` | `/tmp/skipsw_runbook.md` | `36a659b087518233…` | **EXACT** |
| `docs/drafts/miner_skip_switch_v1/RUNBOOK_RIG_RUN_C.md` | `/tmp/skipsw_runbook_c.md` | `f90eb1bd525224f8…` | **EXACT** |

- **How they were rebuilt:** by replaying this session's own `Write` call and the edits that followed it, from the
  session transcript.
- **Why "exact":** each replay equals the sha256 recorded when that runbook was handed to the lead. Those hashes are
  `36a659b0…` and `f90eb1bd…`, the same values quoted in this session's reports before the runs.
- **The header:** each copy carries a 5-line header and a blank line. The body below it is the original, byte for
  byte.

## 3. How run D's blocks were derived

The record states this (`:53-60`, the run-D blocks bullet). The 18 blocks were runbook C's 18 bash blocks after
three `sed` substitutions: `on32` → `on250`, skip max `32` → `250`, and `c_vs_a` → `d_vs_a`.
- **What was checked before the crash:** each block equalled runbook C's block under those rules.
- **Four blocks differed further, only in their remaining literals:**
  - `b06`: the params assertion `== 250`;
  - `b09`: a comment's `[0,250]`;
  - `b10`: the search reason;
  - `b16`: the rule label `warm_start_skip_max is 250`.
- **They can be rebuilt** by applying those rules to `RUNBOOK_RIG_RUN_C.md`. The copy would be a reconstruction, not
  the lead's files.
- **What survives of what they produced,** in `/home/michael` and not under `/tmp` (checked now):
  - `skipsw_launch_on250.sh` differs from the reviewed launcher `stage4_miner_launch.sh` (sha256 `fe964b04…`) only in
    its `cd` line, as do the off, on and on32 launchers;
  - `skipsw_outputs_d_vs_a.py` (sha256 `3405033d…`) equals `skipsw_outputs_c_vs_a.py` (`caf950d0…`) under the same
    substitutions (`diff` empty).

## 4. What `runD_post.log` held, and where those facts survive

| The record's citation | What it showed | Where it survives |
|---|---|---|
| `runD_post.log:1-4` | the switch lines | `~/skipsw_run_on250/logs/gate12_20261122_110738.log:86`, `:96`, `:125`, `:126` |
| `:5-9` | credit and publication | the same log, `:310`, `:312`, `:317`, `:322`, `:323` |
| `:36-44` | the search status after the run (`search_complete`) | `~/skipsw_run_on250/logs/stage4/search_status_after.log:8` |
| `:52-55` | the receipt (sha256 `a71d51b7…`, lag 7.8 s) | `logs/skipsw_session/receipt_on250.log:1-2`, `receipt_on250.sha256` |
| `:56-71` | the extract output | `logs/skipsw_session/extract_on250.log` |
| `:45` | the marker `trial_range` | `~/skipsw_run_on250/.s172_accumulator/skip_delivery_arm` |
| `:46-51` | **the post-run rig check** (all three rigs `no workers`) | **survives nowhere.** It is now lead-reported only |

## 5. What evidence survives (read now)

**`~/distributed_prng_analysis/logs/skipsw_session/`** (37 entries):
- **Bundles:** `bundle_off`, `bundle_on`, `bundle_on32` and `bundle_on250`, each with `bundle.json` and
  `survivors.npz`.
- **Comparator outputs:** `compare_off_vs_yesterday.txt`, `compare_on_vs_off.txt`, `compare_on32_vs_off.txt` and
  `compare_on250_vs_off.txt`.
- **Our check outputs:** `equal_off_vs_yesterday.txt`, `equal_on_vs_off.txt`, `rules_on32_vs_off.txt` and
  `rules_on250_vs_off.txt`.
- **Extract logs:** `extract_{off,on,on32,on250}.log`.
- **Receipts:** `receipt_{off,on,on32,on250}.{json,log,sha256}`. Each `.json` re-hashes to its recorded sha256:

  | Receipt | sha256 |
  |---|---|
  | `receipt_off` | `55d8d7d6…` |
  | `receipt_on` | `a4d274d3…` |
  | `receipt_on32` | `c2631a42…` |
  | `receipt_on250` | `a71d51b7…` |

- **Run copies:** `run_{off,on,on32,on250}/`, each with its `.sha256` manifest. **Re-checked now with
  `sha256sum -c`: 0 failures in all four** (90, 94, 94 and 94 entries).
- `live_state_before.txt`.

**The four run trees,** all at `00e2ee7`, with a clean `git status` (0 lines):

| Tree | WATCHER log | Marker |
|---|---|---|
| `~/skipsw_run_off` | `logs/gate12_20261122_100017.log` | **none** |
| `~/skipsw_run_on` | `logs/gate12_20261122_100717.log` | `skip_delivery_arm` |
| `~/skipsw_run_on32` | `logs/gate12_20261122_104258.log` | `skip_delivery_arm` |
| `~/skipsw_run_on250` | `logs/gate12_20261122_110738.log` | `skip_delivery_arm` |

- **Also in every tree:**
  - `logs/stage4/` (params, launcher output, provisioning log and receipt, search open and status, the clean-tree
    check);
  - in B, C and D only, the volume-gate log, record and params;
  - `.s172_accumulator/` (`current`, `generations`);
  - `prng_analysis.db`.
- **The four staging directories** `~/skipsw_staging_{off,on,on32,on250}` each still hold `miner_ledger.db` and
  `threshold_provenance.json`.
- **Outside the evidence directory, in `/home/michael`:**
  - the four launchers;
  - the base launcher `stage4_miner_launch.sh` (`fe964b04…`);
  - the three check scripts `skipsw_outputs_equal.py`, `skipsw_outputs_c_vs_a.py` and `skipsw_outputs_d_vs_a.py`.

## 6. Effect on the record

- **Every measured fact in the record is still anchored** in surviving files, except one: run D's post-run rig check
  (`runD_post.log:46-51`, "no workers" on all three rigs), which is now lead-reported only.
- **The citations to `/tmp/skipsw_runbook*.md`** resolve to the re-saved copies in §2.
- **The citations to `/tmp/skd/`** resolve to the derivation rule in §3, plus the surviving outputs in §4.
