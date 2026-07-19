# SESSION CHANGELOG — 2026-07-19 — Telegram Boot Notify: Proxmox Hosts + CT100s

**Scope:** Extend the cluster boot-notification system (Telegram) to the migrated
Proxmox topology. All three rigs now boot into Proxmox; the bare-metal notify
installs on the dormant TIMETEC Ubuntu drives no longer fire.

---

## Result

Boot notify deployed and verified 🟢 8/8 OK on all six new targets:

| Target | IP | Role | Status |
|---|---|---|---|
| pve-rig6600 | 192.168.3.121 | Proxmox host | ✅ |
| pve-rig6600b | 192.168.3.155 | Proxmox host | ✅ |
| pve-rig6600c | 192.168.3.163 | Proxmox host | ✅ |
| rrig6600 (CT100) | 192.168.3.122 | LXC worker | ✅ |
| rrig6600b (CT100) | 192.168.3.156 | LXC worker | ✅ |
| rrig6600c (CT100) | 192.168.3.164 | LXC worker | ✅ |

Message format matches the original bare-metal pattern (🟢 ONLINE, host, IP,
time, 🟢/🔴 GPU line) with one addition: a `(HOST)` / `(CT)` role label so the
two notifies per rig are distinguishable.

## Files deployed (per node)

- `/etc/cluster-boot-notify.conf` — credentials + `EXPECTED_GPUS=8`
  (hosts: `root:root 600`; CTs: `root:michael 640` per S159G convention)
- `/usr/local/bin/cluster_boot_notify.sh` — v2 AMD variant (see below)
- `/etc/systemd/system/cluster-boot-notify.service` — oneshot,
  `After=network-online.target`, enabled

Installer/patch scripts (delivered via ser8 → scp to hosts → `pct push` to CTs):
- `install_boot_notify_amd.sh` — full install, credentials via env, idempotent
- `update_boot_notify_v2.sh` — script-only patch (AMD-only count + emoji)

## Key changes vs bare-metal script

1. **AMD-only GPU count.** Counts render nodes whose PCI vendor is `0x1002`
   via `/sys/class/drm/renderD*/device/vendor`, with lspci fallback. Required
   because the Biostar boards expose an Intel iGPU render node — a naive
   renderD count reported 9/8 MISSING on pve-rig6600b/c and rrig6600c CT.
   (pve-rig6600 shows only 8 render nodes — its iGPU is inactive.)
2. **HOST/CT role label** via container detection (`/run/systemd/container`
   or `container=` in pid 1 environ).
3. Credentials sourced from the production conf in VM101
   (`/etc/cluster-boot-notify.conf` — WATCHER's `cluster_notify.sh` reads the
   same file; this is why the Proxmox host .128 has no conf).

## Bug found & fixed: rrig6600 CT100 gateway

CT100 on rrig6600 was created with `gw=192.168.3.1` instead of the correct
`192.168.3.10` (per runbook, matches other CTs). LAN traffic worked (never
touches the gateway) but all internet egress — including DNS and the Telegram
API — silently failed. Fixed:

```
pct set 100 --net0 name=eth0,bridge=vmbr0,gw=192.168.3.10,hwaddr=BC:24:11:EC:7A:49,ip=192.168.3.122/24,type=veth
```

One-off typo from the most recent migration; b and c were correct.

## Open items (not blockers)

- **CT100 onboot:** rrig6600b's CT was found stopped mid-session. Verify
  `pct config 100 | grep onboot` on all three hosts; set `pct set 100 --onboot 1`
  if workers should auto-start after host boot.
- **CT timezones:** CT messages timestamp in UTC (+00:00); hosts in -07:00.
  Cosmetic. `pct exec 100 -- timedatectl set-timezone <zone>` if desired.
- **VM101 conf:** `EXPECTED_GPUS=1` — bump to 2 when the second 3080 Ti is
  passed through.
- Dormant bare-metal Ubuntu installs still carry the old notify — harmless
  (fires only if a rig is deliberately booted back to bare metal).

## Docs updated

- `docs/TELEGRAM_NOTIFICATION_SYSTEM_REFERENCE.md` → v2.0 (node table,
  Proxmox deployment section, iGPU gotcha, gateway post-mortem).
