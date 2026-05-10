# SESSION_CHANGELOG — 2026-05-09 — S174 D1 Forensic Run

## Outcome

D1 reproduced target failure: **VALID_POST_COMPLETION_FAULT** (per Team Beta ruling).
Subsequent reboots of rrig6600 + rrig6600c crashed. rrig6600b (stock) unverified at close.

## Work completed

- S174 hard ready-gate patch + commit `ca06f8c` (validated this session)
- Repo cleanup commit `eae758b` (42 stale backup files removed)
- launch_s174.py (Python launcher, 4-slice TB-reviewed build, 2521 lines)
  - Slice 1: argparse + paths + provenance + preflight
  - Slice 2: subprocess + signals + sentinel classification
  - Slice 3: observation window + bundle assembly
  - Slice 4: semantic classification (8 paths)
  - All TB-required fixes applied
- Positive smoke test: VALID_CLEAN, exit 0
- Negative gate test: READY_GATE_FAILED, exit 5
- D1 launched + completed: child exit 0, 425M seeds, 10:40 compute
- Forensic bundles preserved (Zeus 461 KB + ser8 41 MB)

## D1 forensic finding

- Compute phase clean (8500 fwd + 8500 rev chunks, 17000 total, ~1.4M s/s)
- ~1 sec after optimizer completion, rrig6600 SMU breakdown
- Netconsole evidence: response:0xFFFFFFFF / TransferTableSmu2Dram /
  Failed to retrieve enabled ppfeatures
- Cascade: rrig6600 → unreachable; rrig6600b workers 8→0; rrig6600c workers 8→1
- Launcher classified VALID_CLEAN due to FAULT_KEYWORDS gap
- Team Beta reclassified: VALID_POST_COMPLETION_FAULT

## Known bug surfaced

- launch_s174.py FAULT_KEYWORDS missing RDNA2 SMU strings
- crash_forensic_daemon.py stale-log-pointer (memory line 24, confirmed in production)

## Cluster state at session close (2026-05-09 ~23:30 PDT)

- rrig6600: rebooted, crashed
- rrig6600c: rebooted, crashed
- rrig6600b: not verified at session close, last known healthy 22:55
- Common factor in failed rigs: cwsr_enable=0 mcbp=0 kernel modparam (S166)
- rrig6600b is stock — does not have cwsr_enable=0
- **Hypothesis for tomorrow: revert cwsr_enable=0 on rrig6600 + rrig6600c to match stock rrig6600b config**

## Patch queue (TB-approved, NOT YET WRITTEN)

1. FAULT_KEYWORDS extension — 5 strings:
   response:0xFFFFFFFF, Failed to retrieve enabled ppfeatures,
   TransferTableSmu2Dram, GetEnabledSmuFeaturesHigh, GetEnabledSmuFeaturesLow
2. crash_forensic_daemon stale-log fix (Option B preferred:
   launcher writes /tmp/active_run.json, daemon reads each cycle)
3. launch_s174.py --reclassify <bundle_dir> mode (additive,
   preserves raw artifacts)

## D2 status

BLOCKED. Per Team Beta. Not to be run until:
1. rrig6600 + rrig6600c recovered
2. cwsr_enable=0 hypothesis tested
3. D1 reclassified via Patch 3
4. TB clearance

## Files

- ~/Downloads/S174_D1_zeus_bundle.tar.gz (461 KB)
- ~/Downloads/ser8_crash_forensics_D1_20260509_225522.tar.gz (41 MB)
- launch_s174.py on Zeus at ~/distributed_prng_analysis/launch_s174.py (committed ca06f8c)
- python3_with_venv.sh wrapper on Zeus at ~/distributed_prng_analysis/python3_with_venv.sh

## First moves tomorrow

1. Check rrig6600b reachability + GPU count
2. Inspect /etc/modprobe.d/amdgpu* on all 3 rigs to confirm cwsr_enable=0 location
3. If rrig6600b stock + healthy, revert cwsr_enable=0 on rrig6600 + rrig6600c
4. Reboot recovered rigs, verify 8/8/8
5. THEN start patch work (FAULT_KEYWORDS first)
