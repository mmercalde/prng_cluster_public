# TFM ↔ LLM Cluster Infrastructure (cross-reference)

TFM runs on the same Proxmox VE + LXC infrastructure that hosts the local LLM
cluster. This is a **pointer document** — it does not duplicate the canonical
docs, it points to them so there is one source of truth.

## Canonical documentation (NOT here — in the `rx6600-llm-inference` repo)
- **`docs/RUNBOOK.md`** — Proxmox/CT operations: device visibility, the verified
  serving recipe, memory/crash safety, snapshots, teardown.
- **`docs/DISTRIBUTED.md`** — multi-rig RPC research (facts/history).
- **`docs/POC_Federated_RPC.md`** — the gated plan to validate multi-rig federation.
- Install-side detail (Proxmox + amdgpu-dkms) lives in the `proxmox-gpu-passthrough` repo.

## TFM-relevant facts (summary only — verify against the canonical docs)
- The rigs run **Proxmox VE 8.4 + LXC**. The compute/LLM stack lives in a
  container (rig-6600c: **CT100**, IP `192.168.3.192`; Proxmox host `192.168.3.163`).
- **The container runs BOTH driver paths on the same 8 GPUs**: ROCm/HIP
  (CuPy/PyTorch/TF — what TFM uses) AND Vulkan/RADV (llama.cpp LLM serving).
  TFM's ROCm workload and any LLM serving coexist on the same hardware.
- **GPU passthrough** is path-based (`/dev/kfd` + `/dev/dri` renderD128–135) and
  reboot-durable. ROCm userspace uses `HSA_OVERRIDE_GFX_VERSION=10.3.0`
  (gfx1032 spoofed as gfx1030).
- **Snapshots are the safety net**: risky changes go in a container snapshot and
  roll back via `pct rollback` — no rig is a fixed reference anymore. This
  replaces the old "golden reference rig / full disk clone" restore model.
- **Memory caution shared with the LLM stack**: the rigs have only 7.7 GB host
  RAM. Do not run a large mmap'd process and a heavy GPU job simultaneously
  (see RUNBOOK §2 — the one recorded crash was host-RAM exhaustion).

## Boundary note
TFM infrastructure and the LLM-serving stack are **isolated workloads sharing
hardware**, not merged systems. TFM's coordinator/worker architecture, ZMQ+SQLite
transport, and PRNG pipeline are documented in this repo (`distributed_prng_analysis`);
the LLM cluster is documented in `rx6600-llm-inference`. Keep the two docsets
separate; this file is the only bridge.
