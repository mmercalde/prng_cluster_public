#!/usr/bin/env python3
"""
trial3_reverse_watcher.py
Runs on Zeus. Watches sweep_run1_production.log for Trial 3 reverse pass start.
When detected, SSH into rrig6600c and dump full diagnostic snapshot every 10 seconds.
Deploy on Zeus: python3 trial3_reverse_watcher.py &
"""

import subprocess
import time
import sys
import os
import threading
from datetime import datetime

LOG_FILE = os.path.expanduser("~/distributed_prng_analysis/logs/sweep_run1_production.log")
DIAG_LOG = os.path.expanduser("~/distributed_prng_analysis/logs/trial3_reverse_diag.log")
RIG_C = "192.168.3.162"
INTERVAL = 5  # seconds between snapshots on rrig6600c

def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"{ts} [WATCHER] {msg}"
    print(line, flush=True)
    with open(DIAG_LOG, 'a') as f:
        f.write(line + "\n")

def ssh_cmd(host, cmd, timeout=10):
    """Run command on remote host, return output."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
             f"michael@{host}", cmd],
            capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip(), result.returncode
    except Exception as e:
        return f"ERROR: {e}", -1

def snapshot_rig_c():
    """Dump full diagnostic snapshot from rrig6600c."""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # RAM
    mem, _ = ssh_cmd(RIG_C, "free -m")
    # PageTables  
    page, _ = ssh_cmd(RIG_C, "grep -E 'PageTables|Committed_AS|MemAvailable|Slab' /proc/meminfo")
    # Worker processes
    workers, _ = ssh_cmd(RIG_C, "pgrep -af sieve_gpu_worker | head -10")
    # Per-worker VmSize
    worker_mem, _ = ssh_cmd(RIG_C, """
        for pid in $(pgrep -f sieve_gpu_worker); do
            echo "PID=$pid $(grep -E 'VmSize|VmRSS|VmPeak' /proc/$pid/status 2>/dev/null | tr '\\n' ' ')"
        done
    """)
    # GPU VRAM
    vram, _ = ssh_cmd(RIG_C, "source ~/rocm_env/bin/activate && rocm-smi --showmemuse 2>/dev/null | head -20")
    # dmesg tail
    dmesg, _ = ssh_cmd(RIG_C, "dmesg | tail -5")
    # OOM log
    oom, _ = ssh_cmd(RIG_C, "grep -a 'total-vm\\|OOM killer\\|Killed process' /var/log/syslog | tail -3")

    with open(DIAG_LOG, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"{ts} [SNAPSHOT] rrig6600c Trial 3 reverse pass diagnostic\n")
        f.write(f"--- RAM ---\n{mem}\n")
        f.write(f"--- MEMINFO ---\n{page}\n")
        f.write(f"--- WORKERS ---\n{workers}\n")
        f.write(f"--- WORKER_MEM ---\n{worker_mem}\n")
        f.write(f"--- VRAM ---\n{vram}\n")
        f.write(f"--- DMESG ---\n{dmesg}\n")
        f.write(f"--- OOM_LOG ---\n{oom}\n")
        f.write(f"{'='*60}\n")

def monitor_rig_c():
    """Continuously snapshot rrig6600c until it goes down or we stop."""
    log(f"Starting rrig6600c snapshot loop (interval={INTERVAL}s)")
    consecutive_failures = 0
    while True:
        out, rc = ssh_cmd(RIG_C, "uptime")
        if rc != 0:
            consecutive_failures += 1
            log(f"rrig6600c SSH FAILED (attempt {consecutive_failures}) — rc={rc}")
            if consecutive_failures >= 3:
                log("rrig6600c UNREACHABLE — likely crashed! Check /var/log/syslog after reboot")
                break
        else:
            consecutive_failures = 0
            log(f"rrig6600c alive: {out}")
            snapshot_rig_c()
        time.sleep(INTERVAL)

def watch_log():
    """Watch production log for Trial 3 reverse pass start."""
    log(f"Watching {LOG_FILE} for Trial 3 reverse pass...")
    
    trial_count = 0
    reverse_started = False
    
    # Start from end of file
    try:
        with open(LOG_FILE, 'r') as f:
            f.seek(0, 2)  # seek to end
            
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.5)
                    continue
                
                line = line.strip()
                
                # Count completed trials
                if 'NEW BEST' in line or ('Trial' in line and 'config saved' in line):
                    trial_count += 1
                    log(f"Trial {trial_count} completed: {line}")
                
                # Detect reverse pass start
                if 'Reverse' in line and 'sieve' in line.lower() and not reverse_started:
                    current_trial = trial_count + 1
                    log(f"Reverse pass detected for Trial {current_trial}: {line}")
                    
                    if current_trial >= 3:
                        log("*** TRIAL 3 REVERSE PASS STARTING — launching rrig6600c monitor ***")
                        reverse_started = True
                        # Launch monitoring in separate thread
                        t = threading.Thread(target=monitor_rig_c, daemon=True)
                        t.start()
                        t.join()  # wait for it to finish (crash or complete)
                        log("Trial 3 reverse pass monitor finished")
                        return
                    else:
                        log(f"Trial {current_trial} reverse pass — not monitoring yet")
                
                # Reset reverse flag between trials
                if 'Forward Sieve' in line:
                    reverse_started = False
                    
    except FileNotFoundError:
        log(f"Log file not found: {LOG_FILE}")
        sys.exit(1)
    except KeyboardInterrupt:
        log("Interrupted by user")

if __name__ == "__main__":
    log("Trial 3 reverse pass watcher starting")
    log(f"Diagnostic output: {DIAG_LOG}")
    watch_log()
    log("Watcher complete")
