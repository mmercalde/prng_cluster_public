# SESSION_CHANGELOG_20260722_S180 — Topology documentation correction (boot-selector model)

**Session:** S180
**Date:** 2026-07-22
**Team:** Alpha (Claude Code, VM 101 `zeus-ubuntu` 192.168.3.177, user `michael`)
**Task doc:** `docs/CLAUDE_CODE_INSTRUCTIONS_TOPOLOGY_DOC_CORRECTION_v2.md`
**Scope:** DOCUMENTATION ONLY — no `.py`/`.sh`/`.json` edits, no IP changed anywhere.
**Pre-edit HEAD:** `4c697a8 feat(s172): add Phase 5 D0 durable trial metadata seam`

---

## 1. What was wrong, and the correction

`CLAUDE.md` §3 modeled the rigs as **one-way migrated** to Proxmox
(`Migrated: yes/no` column, "rrig6600 — NO, still bare-metal"). That is the wrong
data model. **Every machine in the cluster is a boot-selector**: it boots its
**default target — bare Ubuntu on its original IP** — *or*, as an alternate,
**Proxmox**, under which the workload runs in a VM/CT at a different address. Both
address sets are valid, never simultaneously. This is the same model §3 already
applied correctly to Zeus (`.127` bare-metal *or* `.128` Proxmox → VM 101); it is
now applied uniformly to the rigs.

| Machine | Default boot — bare Ubuntu | Alternate — Proxmox host | Workload endpoint under Proxmox |
|---|---|---|---|
| Zeus | `192.168.3.127` (`zeus`/`rzeus`; FROZEN) | `.128` (`pzeus-lan`/`pve-zeus`, root) | **VM 101 `192.168.3.177`** (dev box) |
| rrig6600 | `192.168.3.120` | `.121` (`pve-rig6600`) | **CT100 `192.168.3.122`** |
| rrig6600b | `192.168.3.154` | `.155` (`pve-rig6600b`) | **CT100 `192.168.3.156`** |
| rrig6600c | `192.168.3.162` | `.163` (`pve-rig6600c`) | **CT100 `192.168.3.164`** |

## 2. Why the proposed IP swap was REJECTED

A previous scoping attempt ("CT100 IP Migration Cleanup") assumed the bare-metal IPs
were dead and proposed swapping `.120→.122`, `.154→.156`, `.162→.164` across config
**and code**. **That swap was NOT performed and is rejected.**

The cluster is **dual-boot**. `distributed_config.json` (and other config/code)
holding `.120`/`.154`/`.162` is **not a bug** — those are the correct endpoints for
the **default** boot target. Hardcoding the CT100 addresses would bind the config to
one boot mode and **break the other** (the default bare-Ubuntu boot). The correct
long-term fix is a **selectable** endpoint (`rig_profile: baremetal | proxmox`),
which is **Phase 6 prep and explicitly out of scope** here. This session changed
**documentation context only**; no IP was substituted in any file.

## 3. §1 verified facts recorded (established 2026-07-22, evidence)

- **All three rigs currently booted into Proxmox** — live ping sweep from VM 101:
  `.120` DOWN, `.122` UP · `.154` DOWN, `.156` UP · `.162` DOWN, `.164` UP.
- **rrig6600's Proxmox side is installed and running** — `pve-rig6600` at `.121:8006`,
  CT 100 running (uptime ~1.5 h, task history). The old "NO — still bare-metal"
  claim was FALSE and was removed.
- **michael→CT100 SSH key auth works to all three CTs** — `ssh -o BatchMode=yes
  michael@<ip> hostname` returns `rrig6600`/`rrig6600b`/`rrig6600c`. The stale
  "not yet set up (open Phase-3 item)" line was updated to done (2026-07-22).
- **CT device binding correct** (`pct config 100`, rrig6600): `dev0: /dev/kfd,gid=104`,
  `/dev/dri` bind, cgroup2 `c 226:* rwm`, `hostname rrig6600`, `ip .122/24`, gw `.10`.
- **ROCm functional across all 24 GPUs** — llama RPC run across all three rigs under
  Proxmox. (No `/sys/class/drm/card*` device-count "verification" — misleading.)
- **The three rigs are identical clones** — two CTs sharing an SSH host-key
  fingerprint is expected, not a MITM signal.
- **GPU allocation:** Proxmox host console on a dedicated 1660 Ti; Windows VM 100
  (`.138`) holds one 3080 Ti only while running; the **second 3080 Ti is unassigned
  and free** to pass through to VM 101 for rig testing.

## 4. Stage 1 results (the one open check + state confirmation)

- **(a) VM 101 `.177` addressing = DHCP.** `nmcli` connection "Wired connection 1":
  `ipv4.method: auto`, no static `ipv4.addresses`, `DHCP4.OPTION` present
  (`dhcp_server_identifier = 192.168.3.10`), live `ip -4 addr` flagged `dynamic`.
  → Still DHCP; the **"pin static before it becomes the permanent box"** to-do is
  retained (in §3 and the §6 Phase 3 prerequisites).
- **(b) Boot state unchanged since §1:** ping sweep from VM 101 —
  `DOWN .120 / .154 / .162`, `UP .122 / .156 / .164`. Matches §1 exactly. (A DOWN
  bare-metal address here is a boot state, not a defect.)

## 5. Files touched (all `.md`, additive context only)

| File | Change |
|---|---|
| `CLAUDE.md` §3 | Rewrote to the boot-selector model; **removed the `Migrated: yes/no` column**; added worker-endpoint-depends-on-boot statement + point-in-time "all in Proxmox" note; key-auth → done; GPU line reworded (2nd 3080 Ti free); DHCP confirmed + pin-static retained; identical-clones / shared-host-key note; verified stamp → 2026-07-22 with basis. |
| `CLAUDE.md` §6 | Phase 3 prerequisites: key auth → verified working (2026-07-22); DHCP status confirmed. |
| `docs/COMPLETE_OPERATING_GUIDE_v2_0.md` | Added boot-selector note after the `IP Addresses:` list; no IP changed. |
| `docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md` | Added boot-selector note after the `distributed_config.json` sample; no IP changed. |
| `docs/S172_INFRASTRUCTURE_INTERFACE_v1_0.md` | Added boot-selector note near the coordinator-visible IP row; no IP changed. |

**Scope verification (Stage 4):**
- `git diff --stat` — only the 4 `.md` files above. No `.py`/`.sh`/`.json` touched.
- No-IP-substitution check — every dotted octet on a removed line reappears on an
  added line (equal-or-greater count); additions only expand context (e.g. `.10`
  DHCP server). No line removes one machine's IP and adds a different one.

## 6. Explicitly NOT done (out of scope / open items)

- **`rig_profile` selectable endpoints (`baremetal | proxmox`)** — the real fix for
  the config's single-mode addresses. **Phase 6 prep — NOT done.**
- **2nd 3080 Ti passthrough to VM 101** for rig testing — hardware is free but
  passthrough is **NOT done.**
- **Pin VM 101 `.177` static** — still DHCP as of 2026-07-22; **NOT done.**
- **No config/code edited:** `distributed_config.json`, `ml_coordinator_config.json`,
  coordinators, launchers, health/deploy scripts — untouched by design.
  `zeus-proxmox-build/dotfiles/ssh_config` remains stale for migrated rigs
  (reported, not edited). `zmq_sqlite_coordinator.py:_get_zeus_ip()` left as-is
  (UDP routing lookup, not a defect).

## 7. Fallback parity

`fallback parity: code=current, env=ok` — documentation-only session on VM 101;
no dependency or runtime change, so no `.127` env drift introduced.

---

**STOP — awaiting Michael's review.** No commit, no push, no WATCHER/pipeline run
performed. Michael commits and dual-pushes (`git push origin main && git push public main`).
