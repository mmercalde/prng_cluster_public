#!/usr/bin/env python3
"""
apply_s152_ssh_keepalive.py
============================
Patch: Harden SSH keepalive in persistent_worker_coordinator.py

PROBLEM
-------
Workers on remote rigs sit idle between trials (can be 5-15 minutes).
The SSH spawn command uses ServerAliveInterval=30 but no ServerAliveCountMax.

If the remote sshd has ClientAliveInterval set (e.g. reverted to 300s after
reboot), the server kills idle SSH sessions before the next trial starts.
All 8 workers on the rig die simultaneously — exactly the pattern we see.

FIX
---
1. Add ServerAliveCountMax=10 to SSH spawn command.
   With ServerAliveInterval=30 and ServerAliveCountMax=10:
   Client will probe every 30s and tolerate up to 10 missed responses
   before declaring the connection dead = 300s client-side tolerance.
   This keeps the connection alive even if sshd ClientAliveInterval reverts.

2. Add ConnectTimeout=10 to fail fast on dead hosts.

Files patched
-------------
  persistent_worker_coordinator.py

Backup: persistent_worker_coordinator.py.bak_s152_ssh_keepalive
"""

import shutil
import sys
from pathlib import Path

DRY_RUN = "--dry-run" in sys.argv

TARGET = Path("persistent_worker_coordinator.py")
BACKUP = Path("persistent_worker_coordinator.py.bak_s152_ssh_keepalive")

OLD_SSH_CMD = '''\
        ssh_cmd = [
            "ssh",
            "-q",                              # suppress SSH banners/warnings
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            "-o", "ServerAliveInterval=30",
            f"{node.username}@{node.hostname}" if node.username else node.hostname,'''

NEW_SSH_CMD = '''\
        ssh_cmd = [
            "ssh",
            "-q",                              # suppress SSH banners/warnings
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=10",    # [S152] probe 10× before giving up = 300s tolerance
            "-o", "ConnectTimeout=10",         # [S152] fail fast on dead hosts
            f"{node.username}@{node.hostname}" if node.username else node.hostname,'''


def apply():
    src = TARGET.read_text()

    if "ServerAliveCountMax" in src:
        print("⚠️  ServerAliveCountMax already present — patch already applied. Aborting.")
        return

    if OLD_SSH_CMD not in src:
        print("❌ SSH command anchor not found — check for prior changes.")
        return

    patched = src.replace(OLD_SSH_CMD, NEW_SSH_CMD, 1)

    if DRY_RUN:
        print("=== DRY RUN — no files written ===")
        print(f"  ServerAliveCountMax=10 present: {'ServerAliveCountMax=10' in patched}")
        print(f"  ConnectTimeout=10 present:      {'ConnectTimeout=10' in patched}")
        return

    shutil.copy2(TARGET, BACKUP)
    print(f"✅ Backup: {BACKUP}")
    TARGET.write_text(patched)
    print(f"✅ Patched: {TARGET}")
    print()
    print("Verification:")
    print(f"  ServerAliveCountMax=10: {'ServerAliveCountMax=10' in patched}")
    print(f"  ConnectTimeout=10:      {'ConnectTimeout=10' in patched}")
    print()
    print("Effect: SSH client probes every 30s, tolerates 10 missed probes = 300s")
    print("Workers survive idle periods between trials even if sshd ClientAliveInterval reverts.")


if __name__ == "__main__":
    apply()
