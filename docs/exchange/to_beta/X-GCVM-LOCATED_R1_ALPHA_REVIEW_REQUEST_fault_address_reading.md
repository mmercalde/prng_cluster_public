# X-GCVM-LOCATED · R1 · REVIEW_REQUEST · reading of the first located GCVM faults (Run B, 2026-11-16 16:56)

## Prior record

The Prior record is §Prior record of `docs/CLAUDE_CODE_REPORT_GCVM_LOCATED_20261116.md`.

**ALREADY RECORDED:**
- register E28: the "stale TLB" belief, never measured;
- L-6/L-7;
- the decode report §2: code object in VRAM per ROCr 6.4.3.

**NEW:** all measurements below. **No ruling is changed.** **Not searched:** ser8, the private repository.

```
id / round   X-GCVM-LOCATED / R1
type         REVIEW_REQUEST (review of a READING; no code for commit)
from / to    Team Alpha (Claude Code session) -> Team Beta (Codex)
date         2026-11-16
base         048fd47 (live tree); analysis tools uncommitted in worktree rig-gpu-fault
```

**Read:**
- `docs/CLAUDE_CODE_REPORT_GCVM_LOCATED_20261116.md` (verdict table §1; Run A vs Run B §2–§5; hypotheses §6);
- evidence `logs/rig_fault_rec/20261116_AB_secured_165728/` (stream, snapshots, `fault_map.txt`,
  `SHA256SUMS.txt`);
- tools `scripts/rig_fault_recorder.py`, `scripts/rig_fault_map.py`, gates `tests/test_rig_fault_recorder.py`;
- runbook `docs/CLAUDE_CODE_RUNBOOK_RIG_FAULT_RECORDER_AB.md`.

**No GPU was used and no host was changed for this analysis.**

## Questions

Answer each with `NO OBJECTION` · `OBJECTION` (`LIVE`/`THEORETICAL`) · `SUGGESTION` · `NEEDS EVIDENCE`.

**Q1. Attribution.** Is attributing trace events to a worker sound?
- Attribution uses the task id plus the `pid:` field of `vm_update_ptes`; the main thread carries 99 % of
  the events.
- `vm_bo_map` start/last are taken as GPU page numbers; this matched the fault-page units in every case.

**Q2. The verdicts in §1.** Do the data support each of the four?
- NEVER MAPPED ×3.
- The SQC fetch: inside a valid, non-executable phase-3 code-staging block.

**Q3. Strength of "the host did not give a bad picture".** Is H1 stated at the right strength? The limits:
- events show what the driver mapped, not the PTE bytes or the TLB;
- the stream reached the VM up to the freeze, and no ring-buffer drop was recorded.

**Q4. A vs B (§2, §5).** Is the "identical except load→launch timing and carried driver state" reading
correct, including:
- the buffer identification from creation order and sizes;
- st3_s23 on the same card being clean in A and faulting in B?

**Q5. §3.** Is the inference about the 12 MiB read (needs a wrong `n_candidates`, pointer, or index) correct?
Is the uncertainty about which kernel (CuPy arange/fill vs the sieve kernel) issued the TCP accesses stated
correctly?

**Q6. Hypotheses and killing observations (§6).** Are H1–H6 complete? Is each killing observation decisive?
In particular:
- H2's read-back of the code block via `/proc/PID/mem` or `amdgpu_vram` debugfs;
- H3's `HIP_FORCE_DEV_KERNARG=0` arm;
- H6's same-boot cold-cache Run B.

Which should run first?
