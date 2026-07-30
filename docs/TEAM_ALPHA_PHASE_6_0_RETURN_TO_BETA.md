# TEAM ALPHA → TEAM BETA — Phase 6.0 return package

**Re:** `Phase 6.0 — single-rig ROCm miner smoke, paired against an identical CUDA
control`. Base `8e2f5bf`. **Nothing committed, nothing pushed, WATCHER never
touched.** Stopped at the review gate.

**Headline: the RANGE-MINER production path ran on an AMD RX 6600 XT under ROCm and
produced a byte-identical certified artifact to the NVIDIA CUDA run.** This is the
first execution on the hardware class the entire rearchitecture was motivated by.

---

## 1. Acceptance — met in full, and exceeded

**All 22 canonical arrays `np.array_equal → True`**, canonical order identical across
CUDA and ROCm. Counts identical: **forward 398,156 / reverse 383 / bidirectional
319**.

**Stronger than the brief required:** the certified NPZ `artifact_sha256` is
**identical across all three artifacts** —

```
0e0092feeb02e22d28557ddf4d8e421941d6117bcc0448d7f7323ec402c1c4b0

  · D6 release-grade CUDA generation      (commit b08c2c5, 2026-07-29)
  · Phase 6.0 CUDA control                (commit 8e2f5bf, 2026-07-30)
  · Phase 6.0 ROCm generation, RX 6600 XT (commit 8e2f5bf, 2026-07-30)
```

One hash equality subsumes the count, ordering, and field-for-field checks
simultaneously — against an artifact certified a day earlier, on different silicon,
at a different commit.

The third-comparison figures were taken from the **committed**
`docs/D6_RELEASE_GRADE_SMOKE_20260729.log`, not from the brief's summary — i.e.
checked against the artifact of record. **No CUDA-side regression**, so nothing to
report under that clause.

## 2. Non-regression and CUDA-path integrity

All **15 §7 gates green before and after**. `phase4_coordinator` reports **63/63**
including `[PASS] Gate 22: coexistence`, and it ran **after** the final harness edit,
so it validated the delivered tree.

The CUDA default path is proven unchanged **two ways**: structurally (+449/−13, the
`Popen` call verbatim inside `if target is None:`, default-path stdout preserved) and
empirically (pristine and extended harness produce a byte-identical certified
artifact).

Deployment: CT100 clone at `8e2f5bf` — identical source **proven by construction**
via `git rev-parse`, not asserted.

## 3. §4 ROCm health evidence — gap identified, then closed by Michael

**The container could not produce the dmesg diff.** CT100 is an unprivileged LXC:
`dmesg` denied, `/dev/kmsg` absent, `journalctl -k` empty, no passwordless sudo, no
root key auth to the Proxmox host at `.121`. The `amdgpu` driver lives in the **host**
kernel.

**Alpha deliberately did not run it anyway.** Inside the container it returns empty
regardless of what the GPU did, so it would have produced a green line that meant
nothing — the vacuous pass §4 explicitly warns against. It was reported as a gap with
the exact command, and the failure class was evidenced through surfaces that *do* move
on a fault: PCIe replay count, KFD topology, DRI inventory, VRAM, a post-run kernel
launch proving context was not lost, and a 14-pattern log scan. All clean. RAS
counters were flagged **unavailable** on consumer RDNA2 rather than "clean", so their
absence is not misread as a passing check.

**Michael ran it from the Proxmox host. The gap is now closed and the log is clean.**

Boot-time init at 11:46:39 (SMU, KFD topology, ring assignments, HMM 8176 MB), then
exactly three pairs of `amdgpu: Freeing queue vital buffer … queue evicted` at
**12:57:10, 12:59:48, 13:02:32** — one pair per run, in the run window. That is normal
HSA compute-queue teardown on process exit: **clean worker exit observed from the
host side.**

**Absent throughout — including the fault class this rearchitecture exists to avoid:**
no `GPU reset` / `amdgpu_device_gpu_recover`; **no `GCVM_L2_PROTECTION_FAULT`**; no
`VM_L2` / vm fault; no ring timeout; no SDMA error; no soft recovery; no MES failure.
(`EDAC ie31200: No ECC support` is the host CPU memory controller, unrelated.)

Incidental corroboration of the SKU correction: `active_cu_number 32` — 32 CUs is the
**RX 6600 XT**; the plain RX 6600 has 28.

## 4. Artifact classification

Both runs used **scratch repository mode deliberately** (§5 forbids a second
release-grade generation; symmetric modes keep the comparison clean). §3.B's
clean-committed-repository leg is satisfied CT-side at `8e2f5bf`.

The ROCm generation is classified, in all output and evidence, as:

> **ROCm platform-validation certified generation — non-authoritative**

The D6 CUDA generation (`gen-20260730T002104136270Z-step1_java_lcg_0`, commit
`b08c2c5`) remains the authoritative release-grade artifact.

## 5. Findings raised

1. **SKU wrong in more places than the brief stated** — and `CLAUDE.md` has **zero**
   occurrences of "RX 6600", so that part of the brief is itself incorrect. Nothing
   edited; it is descriptive metadata, not functional. Recommend a separate docs
   correction pass.
2. **`daily3.json` is gitignored — `git clone` alone cannot stand up a rig.** The
   dataset was scp'd and its identity enforced cryptographically at runtime. **This
   will recur in Phases 6 and 7**; rig provisioning needs a documented data step, not
   just a clone.
3. **Extending the harness was the only gate-safe option** — Gate 22 whitelists it; a
   new `.py` would have redded §7.
4. **Harness defect found and fixed:** the cleanup `pkill -f` matched its own shell and
   killed it before reporting. See §6 — Alpha requests a ruling on the general case.

## 6. Ruling requested — verification integrity as a standing principle

Finding 4 is the **third instance in three deliverables** of the same failure shape: a
check that was not checking, presenting as a check that passed.

| # | deliverable | the non-check | how it presented |
|---|---|---|---|
| 1 | D6.1 | mutant harness passed a mutated module as an argument while gates built their own from production | **M2 "survived"** — proved nothing |
| 2 | D6.1 | `_flush_npz_incremental` failed on every run for months | a **non-fatal warning** nobody treated as failure |
| 3 | 6.0 | cleanup `pkill -f` killed the shell mid-verification | **silence** — identical to a quiet pass |

Each was found by accident: a suspicious survivor, a log line read closely, a
truncated report. The four-part mutation rule and D6.1's positive-control requirement
already encode the principle, but they are applied **per-gate, ad hoc, when someone
remembers**.

Alpha proposes a standing principle for Beta to formalise:

> **Every verification step must be able to demonstrate that it ran. A step that
> produces no output on failure is not a verification.** Where a check can be vacuous
> — a detector that never fires, a harness that mutates the wrong object, a cleanup
> that can kill its own reporter — the gate must include a **negative control**
> proving the check fails when it should.

Alpha notes this precedent already exists in the codebase and works: D6.1's sentinel
audit used an empty sentinel directory as a negative control and caught a genuine
pre-isolation leak, and Phase 6.0's `G-TRANSACTION-IDENTITY` asserts *both* that
seed-set comparison sees agreement *and* that the detector reports the interruption.
The request is to make it a rule rather than a habit.

## 7. Deliverables

`docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md`,
`docs/SESSION_CHANGELOG_20260730_PHASE_6_0_ROCM_PARITY.md`, and the harness extension
(1 tracked file). The rig kernel-log limitation and the SKU facts were also recorded
to memory, since both recur in Phases 6 and 7.

## 8. Alpha disposition

Phase 6.0 acceptance is met in full and exceeded by three-way artifact-hash identity.
The §4 kernel-log gap is closed from the host with a clean result. Requesting Beta's
pass, plus a ruling on §6.

Sequence per Beta's D6.1 disposition:
`D6.1 ✅ → Phase 6.0 (this) → bounded Phase 6 → D6.2 → D6.3 → Phase 7.`
