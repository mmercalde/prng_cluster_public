# Telegram Notification System — Cluster Reference
**TFM Distributed PRNG Analysis Cluster**
**Last updated: 2026-04-03 | Session S159G post-mortem**

---

## Overview

The cluster uses a Telegram bot (`@Cluster_Prng_bot`) to deliver real-time status
notifications to the operator. There are **two separate notification systems** that
share the same bot but serve different purposes:

| System | Fires when | Lives on |
|---|---|---|
| **Boot Notify** | Rig powers on / reboots | Each rig (rrig6600/b/c) |
| **WATCHER Runtime** | Pipeline events (CRITICAL, DEGRADED, INFO) | Zeus only |

---

## 1. Boot Notification System

### Purpose
Confirms each rig is alive after a boot or reboot. Reports hostname, IP, timestamp,
and GPU count. Fires once per boot via systemd.

### Files (on each rig — rrig6600, rrig6600b, rrig6600c)

```
/etc/cluster-boot-notify.conf          ← credentials (BOT_TOKEN + CHAT_ID)
/usr/local/bin/cluster_boot_notify.sh  ← the notification script
/etc/systemd/system/cluster-boot-notify.service  ← systemd unit that fires it
```

### Credentials file
```
/etc/cluster-boot-notify.conf
```
Owner: `root:michael` | Permissions: `640` (root writes, michael reads)

Contents:
```bash
BOT_TOKEN=<your_bot_token>
CHAT_ID=<your_chat_id>
```

**To view on any rig:**
```bash
ssh rrig6600  'cat /etc/cluster-boot-notify.conf'
ssh rrig6600b 'cat /etc/cluster-boot-notify.conf'
ssh rrig6600c 'cat /etc/cluster-boot-notify.conf'
```

**To edit on a rig (example rrig6600b):**
```bash
ssh rrig6600b 'sudo nano /etc/cluster-boot-notify.conf'
```

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

| Node | IP | Boot Notify | Runtime Notify | Conf Location |
|---|---|---|---|---|
| ser8 | 192.168.1.229 | ❌ N/A (workstation) | ❌ N/A | N/A |
| Zeus | 192.168.3.127 | ❌ N/A | ✅ WATCHER (S77) | via watcher_policies.json |
| rrig6600 | 192.168.3.120 | ✅ Active | ❌ N/A | /etc/cluster-boot-notify.conf |
| rrig6600b | 192.168.3.154 | ✅ Active | ❌ N/A | /etc/cluster-boot-notify.conf |
| rrig6600c | 192.168.3.162 | ✅ Active | ❌ N/A | /etc/cluster-boot-notify.conf |

---

## 4. Common Troubleshooting

### Rig not sending boot notification
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

*Document generated Session S159G. Store at: `docs/TELEGRAM_NOTIFICATION_SYSTEM_REFERENCE.md`*
