# SESSION CHANGELOG — 2026-11-15 — rig GPU fault (Claude Code, worktree `rig-gpu-fault`)

**Base:** `3579088`. **Nothing committed** (Michael commits). **Nothing run on a rig GPU. No host
write.** One CT write and one VM101 GPU touch are disclosed below.

## Prior record
See the Prior record sections of the four documents below; each opens with its own search log.
No owner ruling was changed.

## What was done

| Item | Document / file |
|---|---|
| T1 evidence table, T2 config diff, T3 recorder, T4 plan (X-RIG-GPU-FAULT R1–R3; out of rounds) | `docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md` |
| Torch arm set, escalation ladder, fleet P-vs-N design and budget; probe + stop controller (X-RIG-GPU-PROBE R1 reviewed; controller R2 fixes built, **R2 parked** at Alpha's priority change) | `docs/CLAUDE_CODE_PLAN_RIG_GPU_PROBE.md`; `scripts/rig_seam_probe.py`, `scripts/rig_seam_probe.cleanup.sha256`, `scripts/rig_seam_controller.py`, `tests/test_rig_seam_probe_controller.py` |
| Recorder: passive mode (no SMU reads), lifetime flock | `scripts/rig_gpu_recorder.py` |
| **No-torch miner worker (`--cleanup-torch off`)**, launcher variables, gates + mutants (X-MINER-NO-TORCH R1) | `miner/range_miner_worker.py`, `scripts/launch_fleet_manual.sh`, `tests/test_miner_no_torch.py`, `docs/CLAUDE_CODE_REPORT_MINER_NO_TORCH.md` |
| LEADS: L-3 ruling (Beta's wording), L-6 (rig GPU fault lead, torch history, two-runtime measurement) | `docs/LEADS.md` |
| Brief saved verbatim | `docs/CLAUDE_CODE_INSTRUCTIONS_RIG_GPU_FAULT_20261115.md` |
| Beta threads | `docs/exchange/{to,from}_beta/X-RIG-GPU-FAULT_R1..R3*`, `X-RIG-GPU-PROBE_R1*`, `X-MINER-NO-TORCH_R1*` |

## Key measured findings
- **Two HIP/HSA runtimes load in one rig-worker-environment process** once torch is imported:
  the system ROCm 6.4.3 copy plus the torch wheel's HIP 6.3 copy. This was measured at the
  library level with no device visible. GPU initialisation by both was not shown.
- Boot-time netconsole arming failed on every retained boot inspected (`vmbr0`); arming has been
  manual.
- PCIe AtomicOp requests to the root are not enabled on 7 of 8 slots per rig (PCH), identically
  on all three rigs.
- rrig6600b `19:00.0` SMU `0xFFFFFFFF` at 13:47:28: one failed metrics read while idle.

## Disclosures
- **CT write:** an empty `/dev/shm/.none` was created and deleted on the `.164` CT by a leftover in
  a read-only check command.
- **VM101 GPU touch:** mutant M11 of the probe suite reached the real torch in the test venv and
  very likely called `torch.cuda.is_available/synchronize/empty_cache` on VM101's 3080 Ti, twice,
  with no kernels. Guards are now in both suites: a refusing stub torch and hidden devices.

## Fallback parity
`fallback parity: code=[behind: this worktree's uncommitted changes], env=[not checked this session]`
