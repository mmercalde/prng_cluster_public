# CLAUDE.md — TFM (distributed_prng_analysis) on-box operating contract

This file is read automatically by Claude Code at the start of every session.
It is the source of truth for how an agent operates on this repo. Keep it
committed; update it and any pasted Project-instructions copy together.

TFM = Triangulated Functional Mimicry. This is a white-hat PRNG-analysis
research system (functional mimicry of PRNG surface output — **not** seed
recovery, and the word "lottery" is never used; always "TFM"). Team Alpha
(Claude) implements; **Team Beta is a separate approval authority whose
rulings are binding.** Claude is never Team Beta and never speaks for it.

---

## 0. Run-as and identity (READ FIRST)

- **Run as `michael`, never root.** The TFM tree lives under
  `/home/michael/distributed_prng_analysis`. root has no tree and the wrong
  SSH keys. If reached via `qm guest exec` (which runs as root), wrap in
  `su - michael -c '...'`.
- **Canonical dev box = VM 101 (`zeus-ubuntu`)** at `192.168.3.177`, running
  under the Zeus Proxmox host. All development happens here.
- **`.127` bare-metal Ubuntu = FROZEN FALLBACK. Do not develop on it.** It is a
  boot-selector alternative to Proxmox (Zeus runs one OS at a time), kept as a
  clean recovery target. Touching it turns it into a second drifting working
  copy — the exact confusion this contract exists to prevent.

## 1. Hard rules (non-negotiable)

1. **Never commit or push from an agent sandbox.** Claude edits and tests;
   **Michael commits and dual-pushes.** Configure the deny-list so `git commit`,
   `git push`, and pipeline launches are blocked or always-ask.
2. **Dual-push is Michael's, and always both remotes:**
   `git push origin main && git push public main`
   (`origin` = private `mmercalde/prng_cluster_project`,
   `public` = `mmercalde/prng_cluster_public`).
3. **Never launch the pipeline autonomously.** `watcher_agent.py --run-pipeline`
   spins 26 GPUs. Always Michael-initiated, always `nohup`, never `tmux`.
4. **No hardcoding.** No values baked from memory or a stale doc — only from a
   live check or a committed source file.
5. **Verify bugs before fixing. Never restore from backup — fix forward.**
6. **Read live code before patching.** Clone the public repo fresh each session;
   for files that may differ from live, verify on the box before editing.
7. **Harness discipline (Team Beta enforced):** `inspect.signature()` before
   calling production methods; smoke-test one cycle; synthetic triggers for
   gated paths; test behavioral correctness, not variable assignment.
8. Every session: write `SESSION_CHANGELOG_YYYYMMDD_SN.md`, commit to `docs/`.

## 2. Environment

- **VM 101 venv:** `~/venvs/torch/bin/activate`. For optimizer invocations use
  the wrapper `--optimizer-python ~/distributed_prng_analysis/python3_with_venv.sh`
  (the bare `~/venvs/torch/bin/python3` symlink lacks isolation).
- Non-interactive SSH does not auto-activate the venv.
- **Rigs:** `~/rocm_env/bin/activate`.
- **Deploy rule:** `mkdir -p` any new remote dir BEFORE the scp that fills it
  (SSH is stateless; order matters).
- **scp to 101 uses ABSOLUTE paths** (`/home/michael/...`), not `~` — the
  SFTP-backed scp on ser8 does not expand `~` reliably.

## 3. Network topology (verified 2026-07-22)

**Every machine in the cluster is a boot-selector**, not a one-way migration.
Each boots its **default target — bare Ubuntu on its original IP** — *or*, as an
alternate, **Proxmox**, under which the workload runs in a VM/CT at a different
address. Both address sets are valid; **never simultaneously** (a machine runs one
OS at a time). This is the same model already understood for Zeus (`.127`
bare-metal *or* `.128` Proxmox → VM 101), now applied uniformly to the rigs.

| Machine | Default boot — bare Ubuntu | Alternate — Proxmox host | Workload endpoint under Proxmox |
|---|---|---|---|
| Zeus | `192.168.3.127` (alias `zeus`, tunnel `rzeus`; FROZEN) | `.128` (`pzeus-lan`/`pve-zeus`, root) | **VM 101 `192.168.3.177`** (canonical dev box) |
| rrig6600 | `192.168.3.120` | `.121` (`pve-rig6600`) | **CT100 `192.168.3.122`** |
| rrig6600b | `192.168.3.154` | `.155` (`pve-rig6600b`) | **CT100 `192.168.3.156`** |
| rrig6600c | `192.168.3.162` | `.163` (`pve-rig6600c`) | **CT100 `192.168.3.164`** |

- **VM 101 (canonical dev):** `192.168.3.177`, **DHCP** (`ipv4.method: auto`,
  lease from `.10`, addr flagged `dynamic` — verified 2026-07-22). **Pin static
  before it becomes the permanent box.**
- **Windows VM 100:** `192.168.3.138`, RDP only. GPU allocation: the Proxmox host
  console uses a **dedicated 1660 Ti**; VM 100 holds **one** 3080 Ti *only while
  running*; the **second 3080 Ti is unassigned and free** to pass through to VM 101
  for rig testing.
- **The worker endpoint depends on which target is booted.** When a rig boots its
  default bare-Ubuntu target, its endpoint is the original IP (`.120`/`.154`/`.162`);
  when it boots Proxmox, its endpoint is the CT100 address (`.122`/`.156`/`.164`).
  `distributed_config.json` currently holds the **bare-metal** addresses because
  those match the **default** boot target — this is **not a bug** and must not be
  "corrected."
- **Current state (point-in-time, 2026-07-22):** all three rigs are booted into
  **Proxmox** (live ping sweep from VM 101: `.120`/`.154`/`.162` DOWN,
  `.122`/`.156`/`.164` UP). In this state the config's bare-metal addresses are not
  reachable. This is a **boot state, not a defect**, and is distinct from the
  *default* boot target the config is written against.
- **CT IPs are STATIC** (`host = rig + 1, CT100 worker = host + 1`; RUNBOOK_v1.6_PATCH
  FIX 1 — the old runbook wrongly used +10). CT100 gets the rig's canonical hostname
  (`pct create --hostname rrig6600c`) so `socket.gethostname()` reports the rig name
  for coordinator identity (`docs/S172_INFRASTRUCTURE_INTERFACE_v1_0.md`).
- **michael→CT100 SSH key auth is verified working to all three CTs (2026-07-22)** —
  `ssh -o BatchMode=yes michael@<ip> hostname` returns `rrig6600`/`rrig6600b`/
  `rrig6600c`. The checked-in `zeus-proxmox-build/dotfiles/ssh_config` remains STALE
  for migrated rigs (still bare-metal `.154`/`.162`); do not edit it here.
- **The three rigs are identical clones**, so two CTs sharing an SSH host-key
  fingerprint is **expected, not a MITM signal.**
- **Making the rig endpoint selectable** (e.g. `rig_profile: baremetal | proxmox`)
  is the correct long-term fix and satisfies the no-hardcoding rule, but it is
  **Phase 6 prep and explicitly NOT done.**

*Verified 2026-07-22 by:* live ping sweep from VM 101, `ssh BatchMode` key-auth
check to all three CTs, `pct config 100` on rrig6600 (device binding + hostname +
static IP), and Proxmox UI (`pve-rig6600` at `.121:8006`, CT 100 running). *Not
re-checked this session:* the default bare-Ubuntu boot targets (all rigs are
currently in Proxmox) and `distributed_config.json` contents.

## 4. GPU / rig regimes (the frozen-vs-evolving split)

Library hell was already paid for once (GCVM_L2 fault chase). The lesson's
scope is **runtime environments**, not the dev box:

- **Rigs = PINNED AND FROZEN.** ROCm userspace + modparams locked; snapshot,
  don't touch. The miner is infrastructure-neutral (hostname identity,
  auto-detect output, standard ROCm interface) *specifically so it runs against
  frozen rigs without the rigs ever changing.*
- **VM 101 = evolving (it's the dev box) but ENV-CAPTURED.** Every dependency
  change gets committed to a reproducible artifact (`requirements`/setup), so
  "frozen" and "current" are both one command away — never a hand-rebuilt box.
- **Fallback endgame:** once all rigs are migrated, the real fallback becomes a
  `qm snapshot` of 101 at a known-good phase boundary (environment-complete by
  definition), retiring the `.127`-drift question entirely.

## 5. Fallback-parity review (read-and-report ONLY)

Purpose: keep `.127` a *proven* fallback, not an assumed one. Two layers:
**code** (self-healing via git — one `git pull` from current no matter how far
101 pivots) and **environment** (rots silently if 101's dep changes go
uncaptured — this is the layer that actually breaks a fallback).

**Trigger:** at each phase boundary / TB milestone, not on a calendar. Add one
changelog line: `fallback parity: code=[current|behind N], env=[ok|needs X]`.

**Two-pass, because Zeus runs one OS at a time** (101 and `.127` are never up
simultaneously):

1. On 101 (up now): record `git rev-parse HEAD`, `pip freeze`, run the current
   phase harness.
2. Separately, after booting `.127`: `git fetch` + report commits-behind, diff
   its `pip freeze` against 101's, run the same harness after a pull.
3. Produce a parity report: code ✅/behind-N, env diffs, harness pass/fail,
   dated.

**Remediation is NOT part of the review.** Do not `pip install` / pull-to-patch
`.127` to "fix" drift — that is an unreproducible hand-modification (the frozen
lesson's anti-pattern). If the review finds a missing dep, capture it in the
committed `requirements`/setup artifact and re-provision from that.

## 6. S172 RANGE-MINER — current phase plan

Spec: `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md` (frozen at `1f6c0c5`).
All acceptance gates §11.B/C/E are release blockers.

| Phase | Status | Artifact |
|---|---|---|
| 0 PRNG_TYPE_ENCODING | ✅ `2389b61` | shared registry-derived encoding |
| 1 scaffolding | ✅ `8d0183f` | miner pkg, argparse gate, WATCHER v1.8.0 |
| 2 protocol | ✅ `e0c9d1c` | `miner/range_miner_protocol.py` (8 msg types) + harness |
| 3 worker daemon | **NEXT** | `miner/range_miner_worker.py` — READY handshake, sub-stripe loop, 7 hardcoded branches / 6 base families, `NotImplementedError` for the 5 uncovered |
| 4 coordinator | — | stripe assignment, per-family VRAM caps (TB Q2), one-retry-then-fail-trial (TB Q3) |
| 5 dedup + NPZ contract wall | — | EXPECTED_NPZ_KEYS fail-hard (§12.1) |
| 6 PWC-vs-miner NPZ byte-identity | — | acceptance §11.A–E (needs GPU + ≥2 rigs) |
| 7 WATCHER soak + pool=8 regression | — | §11.K (3-rig / 24-GPU) |

**Phase 3 prerequisites (GPU/rig-dependent):**
- CUDA smoke test is satisfiable on 101 (3080 Ti, 12 GB, confirmed).
- Rig smoke tests need **michael→CT100 SSH key auth** — **verified working to all
  three CTs (2026-07-22)**; see §3.
- Pin VM 101 `.177` static before it becomes the permanent box (still **DHCP** as of
  2026-07-22 — verified).

**Protocol invariants Phase 3 must honor** (from Phase 2, `pwc_protocol` parity):
- Length-prefixed JSON framing (4-byte big-endian + compact UTF-8), 64 MB cap.
- `from_dict()` filters unknown kwargs via `dataclasses.fields()` (TB blocker-A);
  unknown `message_type` → `ValueError` (Phase 0 hard-fail).
- **All dataclass envelope fields carry defaults** — the public-clone
  `persistent/pwc_protocol.py` violates this (`worker_id` lacks a default) and
  cannot import on CPython ≥3.7; the miner sidesteps it. Do not copy that
  envelope verbatim.

## 7. Coverage facts (Phase 3 kernel branches)

`sieve_gpu_worker.py:208-306` has 7 hardcoded `family_name ==` branches covering
**6 base families**: java_lcg (+ `_reverse`, and a separate hybrid branch),
lcg32, minstd, pcg32, xorshift128, xorshift32. The **5 uncovered** families —
mt19937, philox4x32, sfc64, xorshift64, xoshiro256pp — must raise a clean
`NotImplementedError` naming the family and launch **no kernel**. TFM sieve
targets java_lcg only. Hybrid kernels have a different signature (extra
`skip_sequences`, and `a,c` vs `offset`) — replicate verbatim.
