# SESSION CHANGELOG — 2026-11-16 — outside recorder, Runs A/B/C, GCVM fault located (Claude Code, worktree `rig-gpu-fault`)

**Base:** live tree `048fd47`. **Nothing committed** (Michael commits). Tests and runs were launched by Alpha
under delegated authority; this session launched nothing on a GPU.

## Prior record

Each new document opens with its own Prior record:
- `docs/CLAUDE_CODE_RUNBOOK_RIG_FAULT_RECORDER_AB.md`
- `docs/CLAUDE_CODE_REPORT_GCVM_LOCATED_20261116.md`
- `docs/CLAUDE_CODE_PLAN_RIG_ARM_V1_V2.md`

No ruling was changed.

## Done

| Item | Files |
|---|---|
| Outside recorder: plan/arm/disarm/stream/snap/auto/compare/counts; read-only snapshot allow-list; tracefs-only settings (16 events, mono clock, 8 MB/CPU). Fixes: ssh stdin, the disarm restore parse, the counts marker parse, the auto file switch after an aborted launch, the auto T7 exit | `scripts/rig_fault_recorder.py`, `tests/test_rig_fault_recorder.py` (6/6) |
| Fault-address mapper: scoped verdicts; process attribution (thread + explicit VM pid); PTE `dst` decode; TGID vs TID; `--all-lines` | `scripts/rig_fault_map.py` |
| Runbook and host settings list | `docs/CLAUDE_CODE_RUNBOOK_RIG_FAULT_RECORDER_AB.md` |
| Run A (16:23, run #1, clean) and Run B (16:46, run #2, **fault 16:56:20.382**, same three cards) recorded and analysed; evidence secured with SHA-256 | `logs/rig_fault_rec/20261116_AB*/` |
| GCVM report R2 with Beta's accepted corrections; load-to-launch gaps for all 24 workers | `docs/CLAUDE_CODE_REPORT_GCVM_LOCATED_20261116.md` |
| Beta thread X-GCVM-LOCATED R1: three LIVE objections, accepted by Alpha | `docs/exchange/{to,from}_beta/X-GCVM-LOCATED_R1_*` |
| L-7 third repeat and new follow-ups | `docs/LEADS.md` |
| Next arms V1/V2 and cache portability | `docs/CLAUDE_CODE_PLAN_RIG_ARM_V1_V2.md` |
| Run C (V2) recorder set up | `logs/rig_fault_rec/20261116_C/` |

## Host actions taken (all approved by Alpha)

- `systemctl stop rig-power-telemetry.service` on all three hosts, at 16:16 and again checked 17:3x.
  `amdgpu-powercap-min` untouched.
- tracefs `arm` on all three hosts at 16:16; on `.121` and `.163` again at 17:38 after their reboot.
- One process stopped by exact PID: pid 96263 on `.155`, my own orphaned `cat trace_pipe`, at 17:40.

## Disclosures

- `.155` trace events ~17:08–17:40 were lost to the orphan reader (idle period).
- Run C's first T0 markers never reached the stream on `.121` and `.163`; T0 was retaken. The holder of
  `trace_pipe` on `.121` and `.163` at 17:38:27 was not identified.
- The secured evidence's `SHA256SUMS.txt` lists itself and omits `fault_map.txt` (Beta R1 finding). To be
  regenerated without self-reference when evidence is next frozen.

## Fallback parity

`fallback parity: code=[behind: uncommitted worktree changes], env=[not checked]`

## Evening update (17:00–18:50 PST), written before VM101 power-off

**Runs** (launched by Alpha; recorded by this session's outside recorder):

| Run | Launch | Boot position on rrig6600/rrig6600c | Cache | Outcome |
|---|---|---|---|---|
| C (V2) | 17:46:06 | run #1 | warm, pre-filled from rrig6600b | **FAULT** 17:55:48.69: rrig6600c 16:00.0 + 19:00.0 (new), rrig6600 19:00.0; 06:00.0 did not fault |
| D | 18:10:03 | run #1 (boot 18:00) | emptied (cold) | clean, credited |
| E (V1) | 18:28:47 | run #2, same boot | emptied again (cold) | clean, credited; Alpha's cache listings show 64 `.hsaco` per rig created 02:29–02:38:13 UTC, i.e. a cold compile in every phase |

**Tally, phase 4** (single events per cell):

| | cold | warm |
|---|---|---|
| run #1 | clean 4/4 | fault 1/1 |
| run #2+ | clean 1/1 | fault 3/3 |

**Evidence:**
- `logs/rig_fault_rec/20261116_AB_secured_165728/`
- `logs/rig_fault_rec/20261116_C_secured_175651/` (`MANIFEST.sha256`)
- `logs/rig_fault_rec/20261116_D/` (stream for D and E, and runD), `logs/rig_fault_rec/20261116_E/`
- Alpha's copy `/home/michael/runCDE_evidence_20261116/`
- Analysis helpers copied to `logs/rig_fault_rec/*.py` and `gaps.json`

**Code changes** (uncommitted):
- `scripts/rig_fault_recorder.py`:
  - the remote reader exits on SSH stdin EOF (fixes orphan `trace_pipe` holders);
  - T7 is snapped before the loop exits;
  - the `auto` console/coordinator file switch.
- `scripts/rig_fault_map.py`: process attribution, `dst` decode, TGID/TID, scoped verdicts, `--all-lines`.
- `tests/test_rig_fault_map.py` (2/2) and `tests/test_rig_fault_recorder.py` (6/6).

**Host actions:**
- Arm on `.121`/`.163` at 17:38 and 18:0x (after reboots).
- Stopped by exact PID, all my own orphaned `cat trace_pipe` readers from VM101: `.155` 96263 (17:40);
  `.121` 1669, `.155` 159145, `.163` 1589 (18:0x).
- Telemetry was confirmed stopped throughout.
- Stream and `auto` were stopped at owner power-off. The rigs were not contacted afterwards.

**Disclosures:**
- `.155` events ~17:08–17:40 were lost (idle).
- Run C's first T0 markers on `.121`/`.163` were lost; T0 was retaken.
- Run D's automatic T7 was lost to the `auto` ordering bug; replaced by `T7_manual` ~25 min late.
- The first watchdog version of the reader fix failed its local test and was replaced before any rig use.

## Status of Alpha's items 1–5 (offline request 18:4x)

1. **Run C verdicts and gaps: DONE** (report §9). 06:00.0 did not fault in Run C.
2. **Phase-4 gaps D and E against A/B/C: DONE** (report §10 table).
3. **E compiled in every phase: DONE** from the trace (dispatch→code 280–1216 ms, all phases). It is
   corroborated by Alpha's cache listings (02:29–02:38:13 UTC). Those listings are in
   `/home/michael/runCDE_evidence_20261116/` and have **not been read by this session**.
4. **Why phases 1–3 survive a warm load: DONE as candidates** (report §10): scratch reuse in phase 4,
   longest idle, code-pair stacking, the stale-address read in C. Killing observations are stated. **PENDING:**
   reread register E29 (`HSA_NO_SCRATCH_RECLAIM`) against the scratch candidate; compute the idle times for
   C, D and E.
5. **Launcher cache-reset draft: DONE, not landed, not sent.**
   - Draft: `docs/drafts/DRAFT_launcher_cupy_cache_reset.diff` (default byte-identical, checked).
   - Request: `docs/exchange/to_beta/X-CUPY-CACHE-RESET_R1_ALPHA_REVIEW_REQUEST_launcher_cache_reset.md`.
   - **PENDING:** Alpha's go to relay it to Beta; its gates and mutants are not written yet.

**Other pending:**
- The GCVM report R2 has not been re-reviewed by Beta; Runs C–E are not yet in any Beta thread.
- The secured evidence manifest for A/B still lists itself (Beta R1 finding); regenerate at the next freeze.
- The stream design fix is untested beyond Runs D/E.
- The register's cold-start counter-examples (C-15, C-34, C-47, C-86) still need explaining against L-7.
- Nothing is committed. Files for the owner:
  - `scripts/rig_fault_recorder.py`, `scripts/rig_fault_map.py`;
  - `tests/test_rig_fault_recorder.py`, `tests/test_rig_fault_map.py`;
  - `docs/CLAUDE_CODE_RUNBOOK_RIG_FAULT_RECORDER_AB.md`, `docs/CLAUDE_CODE_REPORT_GCVM_LOCATED_20261116.md`;
  - `docs/CLAUDE_CODE_PLAN_RIG_ARM_V1_V2.md`, `docs/LEADS.md`, this changelog;
  - `docs/drafts/DRAFT_launcher_cupy_cache_reset.diff`;
  - the X-GCVM-LOCATED and X-CUPY-CACHE-RESET exchange files.

## 2026-11-17 addendum

- **Runs F, G, H** (one boot; caches emptied before each; launched by Alpha): all clean, recorded T0–T7 + POST.
  Trace loss 0 on all hosts. Report §11; LEADS L-7.
- **Host actions this day:**
  - tracefs `arm` on all three hosts (~17:00);
  - snapshots and markers only;
  - read-only reads of `/usr/src/amdgpu-6.12.12-2194681.22.04` on `.155`.
  - No telemetry, power, driver or module change.
- **Still running** at the end of this entry: the stream (`20261117_F/stream.pid`) and tracing armed. Not yet
  disarmed, awaiting Alpha.
- **Offline results:**
  - all-run phase-4 gap table (A–H);
  - per-lane private arrays per phase;
  - fault placement against scratch (none inside);
  - the installed KFD map/unmap/flush ordering (§11).
- **Evening 11-17:**
  - **Disarmed** tracing on all three hosts: 0 events, clock `local`, buffer 1410 KB, no readers. `tracing_on`
    is left at 0 where a fresh boot has 1; the next boot restores it.
  - Stream (pid 8742) stopped. No further rig contact.
  - Hardware notes written: report §12, decode report §1.3 link row corrected, LEADS.
- **11-17 late (offline):**
  - Beta R2 received: Q1 no objection; Q2 LIVE objection (listing error read as empty); Q3 driver-side flush
    record; Q4 no objection; finding on the Run C manifest.
  - Then, as Alpha directed:
    - draft v3 + gates/mutants (report §13);
    - Run C `MANIFEST_SUPPLEMENT_20261117.md` (originals untouched);
    - CPU corrected in `TRIANGULATED_FUNCTIONAL_MIMICRY_VERIFIED_v1_0.md`, `CHAPTER_13_IMPLEMENTATION_PROGRESS_v1_6.md`
      and `SELFPLAY_ARCHITECTURE_PROPOSAL_v1_0.md` (historical changelogs left as written);
    - report §12/§13, LEADS.
  - `scripts/launch_fleet_manual.sh` not touched; no Beta round 3 sent; nothing committed or pushed.
- **11-17:** X-GCVM-LOCATED R3 sent (final round); Beta: OBJECTION LIVE on draft v3 (record verifier fails
  open; regex join), with landing and first-counted-run requirements. Recorded in LEADS. Nothing committed.
