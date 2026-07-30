# SESSION CHANGELOG — 2026-07-30 — S172 Phase 6.0 ROCm parity

**Outcome: acceptance met. Stopped at the Team Alpha review gate. Nothing committed or pushed.**

Full evidence: `docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md`.

## What was done

First execution of the RANGE-MINER production path on AMD hardware — one RX 6600 XT in CT100
`rrig6600` — paired against an identical CUDA control on VM 101's RTX 3080 Ti.

* Deployed the repo to CT100 by `git clone` at **`8e2f5bf`** (docs-only above the brief's
  `3823b56`; used on both sides). CT tracked-clean, zero untracked.
* Extended `tests/smoke_s172_phase5_d6_zeus_single_gpu.py` (+449/−13) with an additive
  `--rocm-remote` target and a `--compare` 22-array comparator. CUDA default path unchanged.
* Ran the D6-pinned trial on both platforms: 8M seeds, stripe 4M, cap 1M, window 3,
  thresholds 0.31/0.47, java_lcg, serial_reference, constant-skip phases 1+2.

## Results

* **All 22 canonical arrays `np.array_equal` → True**, canonical order identical.
* Counts identical on both platforms and equal to the D6 release-grade run:
  **forward 398,156 / reverse 383 / bidirectional 319**.
* **The certified NPZ `artifact_sha256` is identical across all three** — D6 release-grade CUDA
  at `b08c2c5`, Phase 6.0 CUDA, and Phase 6.0 ROCm: `0e0092fe…c1c4b0`. Stronger than the
  required field-for-field equality.
* Threshold provenance `requested == payload == effective` on both, `validated=True`.
* Step-2 loader `fallback_used=False`, 319 rows, npz v3, both.
* ROCm health: no change in PCIe replay count, KFD topology, DRI nodes or VRAM; post-run kernel
  probe succeeded; 0 hits across 14 fault patterns; 0 spool residue; 0 surviving processes.
* No env overrides set on the rig — verified positive finding.

## Non-regression

All 15 §7 gates green **before** and **after** the edit (D1.1 · D1.0 · D0 · D2 · D3.0 · D3 ·
D3.25 · D3.5 · D4 · D5 · D6 3.A · D6-threshold · D6.1 · Phase 3 · Phase 4). The CUDA default
path was additionally proven unchanged empirically: pristine-harness and extended-harness CUDA
runs produced a **byte-identical certified artifact**.

## Findings raised (not fixed — out of scope per §6)

1. **SKU:** cards are **RX 6600 XT** (DID `0x73ff`, VBIOS `113-123XT145W201222`).
   `distributed_config.json:16,27,38` says "RX 6600", as do `README.md`,
   `parameter_registry.json`, `distributed_config_1rig.json` and others — broader than the
   brief's "three places". **`CLAUDE.md` contains zero occurrences**; the brief is wrong on that
   point. Nothing edited.
2. **`daily3.json` is gitignored** (`.gitignore:41`), so `git clone` alone cannot stand up a
   rig; it was scp'd. Rig-provisioning runbooks need an explicit dataset step. Identity is
   still enforced at runtime via mandatory `dataset_sha256` + `residue_sha256`.
3. **Host kernel-log access missing:** CT100 is an unprivileged LXC — `dmesg` denied,
   `/dev/kmsg` absent, `journalctl -k` empty, no passwordless sudo, and no root key auth to the
   Proxmox host `192.168.3.121`. The §4 dmesg before/after diff is reported as a **gap** with
   the exact command to close it, rather than satisfied by an empty in-container buffer that
   would have read as a false pass.
4. **Harness defect found and fixed:** the remote-cleanup check's `pkill -f` pattern matched its
   own shell and killed it before reporting, producing a silently empty evidence line.

## Fallback parity

`fallback parity: code=[not assessed this session], env=[not assessed this session]` — Phase 6.0
did not touch `.127`, and the §5 two-pass review was not triggered (no boot of the bare-metal
target). Flagged so the omission is explicit rather than assumed.
