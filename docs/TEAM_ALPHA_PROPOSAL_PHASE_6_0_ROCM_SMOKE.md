# TEAM ALPHA → TEAM BETA — proposal: Phase 6.0, single-rig ROCm miner smoke

**Re:** S172 — inserting a cheap ROCm validation step before full Phase 6, and a
status correction on the `rrig6600` migration blocker.

**Status:** D6 closed and pushed (certification record `20d6493`, generation
certified at `b08c2c5`). This is a scope proposal, not a deliverable. Requesting
Beta's ruling before any work starts.

---

## 1. Status correction — the Proxmox migration blocker is cleared

Beta's D6 runway stated the `rrig6600` Proxmox migration was **not** required for
the D6 single-GPU smoke but **was** required before the Phase-7 26-GPU saturation
soak. Michael reports that work is complete: **`rrig6600b` and `rrig6600c` are no
longer mid-conversion; the AMD rigs are fully integrated.**

That removes the stated Phase-7 infrastructure blocker and, more importantly for
this proposal, means ROCm hardware is reachable **now** rather than after a
conversion effort.

## 2. The gap this proposal addresses

Everything S172 has certified to date — D0 through D6, including the release-grade
generation — has run on **one NVIDIA RTX 3080 Ti under CUDA/cupy**. The AMD rigs
have not executed the miner at all.

This matters because the entire RANGE-MINER rearchitecture exists to solve a
**ROCm-side failure**: PWC's silent hard resets / L2 protection faults on the
`rx 6600` rigs at full-fleet saturation, traced to launch-storm behaviour that the
persistent-daemon design replaces. The thesis under test is "persistent per-GPU
daemons hold PWC-class throughput at saturation without the fault." **No test so
far has touched the hardware, driver stack, or failure mode that thesis is about.**

The current plan reaches ROCm only at **Phase 7** (50-trial, 26-GPU WATCHER soak),
which is also the most expensive and most scaffolding-dependent test in the
programme. Alpha's concern: a ROCm-specific defect in the miner worker — kernel
ABI/prefix differences, `amdgpu` memory behaviour, persistent-daemon lifecycle
under ROCm, or the `hipMemcpy` path — would be discovered *after* building the
full multi-node apparatus that assumes ROCm works, and while 26 GPUs are under
load. That is the worst place to find it and the hardest place to isolate it.

## 3. Proposal — Phase 6.0: single-rig, single-GPU ROCm miner smoke

Insert a small step **between D6.1 and full Phase 6**: the exact ROCm analogue of
D6's 3.B acceptance smoke, on **one RX 6600 on one rig**.

**Scope (deliberately minimal — no new infrastructure):**
- one rig, one GPU, ROCm 6.4.3 / kernel 6.8.0-107 / `amdgpu-dkms 6.12.12` pinned;
- `serial_reference` backend, `java_lcg`, small-but-real seed window;
- real `range_miner_worker.py` process, real ROCm kernel execution (not cupy/CUDA);
- asymmetric non-default thresholds (e.g. `forward=0.31 / reverse=0.47`) so the D6
  provenance chain is exercised on ROCm too;
- acceptance identical to 3.B: certified generation, 22-array bundle validated,
  Step-2 loader read-back `fallback_used=False`, and
  `requested == payload == effective` read off the **ROCm** executor.

**Explicitly NOT in scope:** multi-node coordination, the four-path comparison,
throughput/promotion measurement, saturation, WATCHER. Those remain Phase 6/7.

**What it does not require:** the second RTX 3080 Ti, passwordless SSH to all CT100
workers, VM-101 DHCP reservation, or any Phase-6 prerequisite. It needs one rig up
and reachable, which is now the case.

## 4. Why this is worth a step of its own

- **Cheapest possible discovery point.** If the miner cannot execute on ROCm, that
  is a small single-rig failure today instead of a compound failure inside Phase 7.
- **It de-risks the Phase-6 investment.** Phase 6's four-path verify and the
  §17 promotion benchmark both assume the miner runs correctly on the rigs. This
  validates that assumption before the scaffolding is built on top of it.
- **Provenance parity.** D6 proved the threshold reaches a **CUDA** kernel. The
  provenance contract should be demonstrated on the **ROCm** kernel path as well,
  since that is the production fleet. The `effective_threshold` read-back is
  executor-specific by construction, so this is genuinely new evidence, not a
  repeat.
- **It tests the actual thesis one step earlier.** Not saturation — but the first
  evidence that the persistent-daemon miner functions on the hardware the pivot
  was for.

## 5. Sequencing Alpha proposes

```
D6.1  — incremental NPZ atomic flush and durability repair   (Beta-mandated, blocks soak)
6.0   — single-rig ROCm miner smoke                          (this proposal)
6     — four-path verify + §17 promotion benchmark
7     — 26-GPU saturation soak (50 trials, WATCHER)
```

D6.1 stays first: it is Beta-mandated and a durability defect is worse to carry
into any multi-trial work. 6.0 then runs before the Phase-6 apparatus is built.

Alpha has no objection to running 6.0 in parallel with D6.1 if Beta prefers, since
they touch different code (flush helper vs. worker execution path) — but sequential
is cleaner for attribution if either surfaces something.

## 6. Ruling requested

1. Approve or reject inserting **Phase 6.0** as scoped above.
2. If approved: confirm the acceptance set (Alpha proposes 3.B parity plus the
   ROCm-side provenance chain), and whether a certified generation from a rig
   should be treated as release-grade (it would be certified against the same
   repository commit, executed on different silicon) or explicitly as
   platform-validation evidence only.
3. Confirm whether 6.0 runs before or in parallel with D6.1.

Alpha's recommendation: approve, sequential after D6.1, and treat the resulting
generation as **platform-validation evidence** rather than a second release-grade
artifact — one release-grade generation per commit avoids ambiguity about which
artifact is authoritative.
