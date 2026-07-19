# Telegram Notification System — Cluster Reference
**TFM Distributed PRNG Analysis Cluster**
**Last updated: 2026-07-19 | v2.1 — Proxmox topology deployment (incl. pzeus host)**
**Supersedes: 2026-04-03 S159G edition (bare-metal topology)**

---

## Overview

The cluster uses a Telegram bot (`@Cluster_Prng_bot`) to deliver real-time status
notifications to the operator. There are **two separate notification systems** that
share the same bot but serve different purposes:

| System | Fires when | Lives on |
|---|---|---|
| **Boot Notify** | Node powers on / reboots | Proxmox rig hosts + CT100 workers + VM101 |
| **WATCHER Runtime** | Pipeline events (CRITICAL, DEGRADED, INFO) | Zeus (VM101) only |

**2026-07-19 topology note:** all rigs now boot into Proxmox. Boot notify runs
in TWO places per rig — the Proxmox host (box survived POST + amdgpu load) and
CT100 (worker container up, all 8 render nodes visible). Messages carry a
`(HOST)` / `(CT)` role label. The bare-metal installs on the dormant TIMETEC
Ubuntu drives remain but only fire if a rig is booted back to bare metal.

---

## 1. Boot Notification System

### Purpose
Confirms each rig is alive after a boot or reboot. Reports hostname, IP, timestamp,
and GPU count. Fires once per boot via systemd.

### Files (on each Proxmox host AND inside each CT100)

```
/etc/cluster-boot-notify.conf          ← credentials (BOT_TOKEN + CHAT_ID)
/usr/local/bin/cluster_boot_notify.sh  ← the notification script
/etc/systemd/system/cluster-boot-notify.service  ← systemd unit that fires it
```

### Credentials file
```
/etc/cluster-boot-notify.conf
```
Owner/perms: CT100s and VM101 = `root:michael 640` (root writes, michael
reads). Proxmox hosts = `root:root 600` (no michael user on hosts).

Contents:
```bash
BOT_TOKEN=<your_bot_token>
CHAT_ID=<your_chat_id>
EXPECTED_GPUS=8        # rigs; VM101 uses its passthrough GPU count
```

**To view (hosts):**
```bash
ssh root@192.168.3.121 'cat /etc/cluster-boot-notify.conf'   # pve-rig6600
ssh root@192.168.3.155 'cat /etc/cluster-boot-notify.conf'   # pve-rig6600b
ssh root@192.168.3.163 'cat /etc/cluster-boot-notify.conf'   # pve-rig6600c
```

**To view (CT100, via host):**
```bash
ssh root@192.168.3.155 'pct exec 100 -- cat /etc/cluster-boot-notify.conf'
```

**Production credential source of truth:** VM101 —
`ssh michael@192.168.3.177 'cat /etc/cluster-boot-notify.conf'`
(WATCHER's `cluster_notify.sh` sources this same file, which is why the
Proxmox Zeus host at .128 has no conf.)

**To fix permissions if broken (S159G lesson):**
```bash
ssh rrig6600b 'sudo chown root:michael /etc/cluster-boot-notify.conf && sudo chmod 640 /etc/cluster-boot-notify.conf'
```

### Boot notify script
```
/usr/local/bin/cluster_boot_notify.sh
```
Permissions: `755` (executable by all)

Sources the conf file, reads hostname/IP/GPU count, sends curl POST to Telegram API.
Edit this file to change the boot message format.

**v2 (2026-07-19, AMD/Proxmox variant):**
- GPU count = render nodes with PCI vendor `0x1002` only, via
  `/sys/class/drm/renderD*/device/vendor` (lspci fallback). REQUIRED: the
  Biostar boards expose an Intel iGPU render node; a naive renderD count
  reads 9/8 on pve-rig6600b/c and rrig6600c CT. (pve-rig6600's iGPU is
  inactive — it shows 8 either way.)
- Adds `(HOST)` / `(CT)` role label via container detection.
- Same script deployed to hosts and CTs; role/hostname self-label.

**pzeus host variant (`install_boot_notify_pzeus.sh`):** counts NVIDIA
VGA/3D devices via lspci — driver-independent, so vfio-bound passthrough
GPUs (no host driver, invisible to nvidia-smi and /sys/class/drm) are
still counted. pzeus expects 3: 2x RTX 3080 Ti (19:00→VM100, 68:00→VM101)
+ 1x GTX 1660 Ti (1a:00).

### Systemd service
```
/etc/systemd/system/cluster-boot-notify.service
```

Check status:
```bash
ssh rrig6600b 'systemctl is-enabled cluster-boot-notify.service'
ssh rrig6600b 'systemctl is-active cluster-boot-notify.service'
```

View last boot journal:
```bash
ssh rrig6600b 'journalctl -u cluster-boot-notify.service -b 0 --no-pager'
```

Manual trigger (test without rebooting):
```bash
ssh rrig6600b 'bash /usr/local/bin/cluster_boot_notify.sh'
```

### Sample Telegram output
```
🟢 ONLINE
Host: rig-6600b
IP: 192.168.3.154
Time: 2026-04-03T07:19:34-07:00
🟢 GPUs: 8/8 OK
```

---

## 2. WATCHER Runtime Notification System

### Purpose
Delivers pipeline event alerts to the operator during autonomous operation.
Advisory only — never blocks or alters pipeline control flow.

### Files (on Zeus only)

```
/usr/local/bin/cluster_notify.sh                        ← runtime notify script (called by WATCHER)
~/distributed_prng_analysis/agents/watcher_agent.py     ← notify_telegram() function (line 129)
~/distributed_prng_analysis/watcher_policies.json       ← controls which events notify
```

### notify_telegram() — watcher_agent.py line 129
```python
def notify_telegram(message: str):
    """Send a Telegram notification via cluster_notify.sh.
    Session 77: Best-effort, non-blocking, silent on failure.
    """
    try:
        subprocess.Popen(
            ["/usr/local/bin/cluster_notify.sh", message],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass  # best-effort, never block
```

Fire-and-forget — WATCHER never waits for delivery confirmation.

### Notification classes — watcher_policies.json

```json
"notifications": {
    "telegram_enabled": true,
    "script_path": "/usr/local/bin/cluster_notify.sh",
    "classes": {
        "critical": {
            "enabled": true,
            "description": "Human intervention required — always notify",
            "can_disable": false
        },
        "degraded": {
            "enabled": true,
            "description": "Autonomy self-corrected — always notify",
            "can_disable": false
        },
        "info": {
            "enabled": false,
            "description": "Pipeline completed cleanly — optional",
            "can_disable": true
        }
    },
    "info_on_complete": false
}
```

| Class | Fires when | Can disable? | Default |
|---|---|---|---|
| CRITICAL | Pipeline halted, human required | ❌ No | ✅ On |
| DEGRADED | WATCHER self-corrected an issue | ❌ No | ✅ On |
| INFO | Pipeline completed cleanly | ✅ Yes | ❌ Off |

### To enable completion (INFO) notifications
Edit `watcher_policies.json` on Zeus:
```json
"info": { "enabled": true },
"info_on_complete": true
```

### To disable Telegram entirely (testing only)
```json
"telegram_enabled": false
```

### Events that trigger CRITICAL
- Pipeline HALTED (e.g. missing config file)
- File validation failed
- Step exceeded retry limit
- Kill switch activated

### Events that trigger DEGRADED
- WATCHER auto-corrected a step failure
- Model fallback triggered
- Worker recovery

---

## 3. Node Summary

| Node | IP | Boot Notify | Runtime Notify | Conf |
|---|---|---|---|---|
| ser8 | 192.168.1.229 | ❌ N/A (workstation) | ❌ N/A | N/A |
| pzeus (Proxmox host) | 192.168.3.128 | ✅ v2 2026-07-19 (lspci NVIDIA count, vfio-safe) | ❌ N/A | /etc/cluster-boot-notify.conf (600, EXPECTED_GPUS=3) |
| VM101 zeus-ubuntu-vm | 192.168.3.177 | ✅ (P2V-inherited, nvidia-smi count) | ✅ WATCHER (S77) | /etc/cluster-boot-notify.conf (EXPECTED_GPUS=2 — reports 🔴 1/2 until 2nd 3080 Ti reassigned from VM100 win11) |
| pve-rig6600 (host) | 192.168.3.121 | ✅ v2 2026-07-19 | ❌ N/A | /etc/cluster-boot-notify.conf (600) |
| pve-rig6600b (host) | 192.168.3.155 | ✅ v2 2026-07-19 | ❌ N/A | /etc/cluster-boot-notify.conf (600) |
| pve-rig6600c (host) | 192.168.3.163 | ✅ v2 2026-07-19 | ❌ N/A | /etc/cluster-boot-notify.conf (600) |
| rrig6600 CT100 | 192.168.3.122 | ✅ v2 2026-07-19 | ❌ N/A | /etc/cluster-boot-notify.conf (640) |
| rrig6600b CT100 | 192.168.3.156 | ✅ v2 2026-07-19 | ❌ N/A | /etc/cluster-boot-notify.conf (640) |
| rrig6600c CT100 | 192.168.3.164 | ✅ v2 2026-07-19 | ❌ N/A | /etc/cluster-boot-notify.conf (640) |
| VM100 win11-gpu | (VM) | ❌ (Windows — no bash/systemd; Task Scheduler candidate) | ❌ N/A | N/A |
| bare-metal Zeus | 192.168.3.127 | ✅ dormant (nvidia-smi, EXPECTED_GPUS=3; fires only when booted bare-metal) | — | FROZEN FALLBACK, hands-off |
| bare-metal rigs (TIMETEC) | .120/.154/.162 | ⚠ dormant (fires only if booted bare-metal) | ❌ | old install retained |

---

## 4. Common Troubleshooting

### Node not sending boot notification (Proxmox era)
```bash
# Host:
ssh root@192.168.3.155 'journalctl -u cluster-boot-notify.service -b 0 --no-pager'
# CT100 (via host):
ssh root@192.168.3.155 'pct exec 100 -- journalctl -u cluster-boot-notify.service -b 0 --no-pager'
# Manual test:
ssh root@192.168.3.155 'bash /usr/local/bin/cluster_boot_notify.sh'
ssh root@192.168.3.155 'pct exec 100 -- bash /usr/local/bin/cluster_boot_notify.sh'
```
If a CT test fails with `curl: (28) Resolving timed out` — check the CT
gateway (see §5b) and that the CT is actually running (`pct status 100`).

### Rig not sending boot notification (legacy bare-metal commands)
```bash
# Step 1: check the service ran
ssh rrig6600b 'journalctl -u cluster-boot-notify.service -b 0 --no-pager'

# Step 2: check conf exists and is readable
ssh rrig6600b 'ls -la /etc/cluster-boot-notify.conf'

# Step 3: fix permissions if owner drifted (most common cause)
ssh rrig6600b 'sudo chown root:michael /etc/cluster-boot-notify.conf && sudo chmod 640 /etc/cluster-boot-notify.conf'

# Step 4: manual test
ssh rrig6600b 'bash /usr/local/bin/cluster_boot_notify.sh'
```

### WATCHER not sending alerts
```bash
# Check script exists on Zeus
ls -la /usr/local/bin/cluster_notify.sh

# Check telegram_enabled in policies
grep telegram_enabled ~/distributed_prng_analysis/watcher_policies.json

# Manual test
/usr/local/bin/cluster_notify.sh "test message"
```

### Change bot token or chat ID
```bash
# On each rig that needs updating:
ssh rrig6600b 'sudo nano /etc/cluster-boot-notify.conf'
# Update BOT_TOKEN= and/or CHAT_ID= lines
# Save, then test:
ssh rrig6600b 'bash /usr/local/bin/cluster_boot_notify.sh'
```

---

## 5b. 2026-07-19 Post-Mortem — CT100 Gateway Typo (rrig6600)

**Symptom:** rrig6600 CT100 boot notify: `curl: (28) Resolving timed out`.
LAN traffic (SSH, coordinator) unaffected.
**Root cause:** CT created with `gw=192.168.3.1`; correct LAN gateway is
`192.168.3.10` (per runbook; other CTs correct). All internet egress from the
CT silently failed — invisible until something needed the outside world.
**Fix:** `pct set 100 --net0 name=eth0,bridge=vmbr0,gw=192.168.3.10,hwaddr=<same>,ip=192.168.3.122/24,type=veth`
**Prevention:** after creating any CT, verify `pct config <id> | grep net0`
gateway matches 192.168.3.10, and test egress: `pct exec <id> -- getent hosts api.telegram.org`.

---

## 5. S159G Post-Mortem — Permissions Drift Incident

**Date:** 2026-04-03
**Symptom:** rrig6600b boot notification did not arrive after morning reboot
**Root cause:** `/etc/cluster-boot-notify.conf` ownership drifted from `root:michael`
during S159G `/etc/environment` deployment work. File existed but was unreadable
by michael, causing `cluster_boot_notify.sh` to exit silently with no message sent.
**Fix:** `sudo chown root:michael /etc/cluster-boot-notify.conf && sudo chmod 640`
**Prevention:** After any sudo work on a rig involving `/etc/` files, verify conf
ownership has not drifted: `ls -la /etc/cluster-boot-notify.conf`

---

*v2.1 updated 2026-07-19 (Proxmox topology deployment — see
SESSION_CHANGELOG_20260719_TELEGRAM_PROXMOX.md). Original: Session S159G.
Store at: `docs/TELEGRAM_NOTIFICATION_SYSTEM_REFERENCE.md`*
