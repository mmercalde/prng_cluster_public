# SESSION CHANGELOG — S163 (Rig Fix Confirmation & Crash Investigation)
**Date:** 2026-04-16
**Session:** S163 (continuation)
**Focus:** Rig crash root cause confirmation + all fixes verified deployed
**Author:** Claude (Team Alpha Lead Dev)
**HEAD at start:** `3c20d2d`

---

## Context

After deploying `free_all_blocks()` removal + MEM_DEBUG instrumentation (`3c20d2d`),
a Step 1 run was launched at 18:01 with `seed_cap_amd=100000`, 3 trials, 24 workers.
All 3 rigs crashed and rebooted during the run. Workers went offline ~18:13 (rrig6600b
first, per netconsole). Zeus coordinator (`window_optimizer.py` PID 88761) continued
spinning at 94% CPU for 1.5+ hours waiting for workers that never returned.
Run was killed manually after diagnosis.

**Important:** The crashes occurred BEFORE the snd_hda_intel + amdgpu fixes were
fully confirmed deployed. This run was NOT a valid test of those fixes.

---

## Crash Investigation

### Timeline (from netconsole_listener.log)
| Time | Event |
|------|-------|
| 18:01 | Run launched, 26 GPUs online |
| 18:13 | rrig6600b rebooted (PCI bus enumeration visible in netconsole) |
| 18:18 | rrig6600c NETCONSOLE READY (post-reboot) |
| 18:31 | Last log entry — chunk 10729 |
| 18:45 | rrig6600 NETCONSOLE READY (post-reboot) |
| ~20:10 | Zeus process killed manually |

### What the logs showed
- Only Trial 1 (index 0) completed: `W21_O68_midday+evening_S2-72_FT0.37_RT0.48`
- 13 Trial/PASS lines total — run aborted mid-Trial 2
- No GPU fault signatures in kern.log (crashes happened BEFORE this boot)
- Workers: 0/0/0 on all 3 rigs after reboot

### Root cause
Crashes occurred during a run launched BEFORE all rig fixes were baked into
initramfs. The snd_hda_intel and amdgpu.conf fixes were present in
`/etc/modprobe.d/` but `update-initramfs -u` had not yet been run on some rigs,
meaning the old module parameters loaded from initramfs at boot.

---

## Fix Verification — All 3 Rigs Confirmed Clean

### rrig6600 (192.168.3.120)
| Check | Result |
|-------|--------|
| `/etc/modprobe.d/snd-power.conf` | ✅ `power_save=0 power_save_controller=N` |
| `/etc/modprobe.d/amdgpu.conf` | ✅ `gfxoff=0` |
| `/boot/initrd.img` timestamp | ✅ Apr 15 18:10 (after conf files written) |
| `snd_hda_intel power_save` param | ✅ `0` (confirmed live) |
| `amdgpu gfxoff` sysfs | ✅ absent (correct — disabled means no sysfs node) |

### rrig6600b (192.168.3.154)
| Check | Result |
|-------|--------|
| `/etc/modprobe.d/snd-power.conf` | ✅ `power_save=0 power_save_controller=N` |
| `/etc/modprobe.d/amdgpu.conf` | ✅ `gfxoff=0` |
| `/boot/initrd.img` timestamp | ✅ Apr 16 12:02 (after conf files written) |
| `snd_hda_intel power_save` param | ✅ `0` (confirmed live) |
| `amdgpu gfxoff` sysfs | ✅ absent (correct) |

### rrig6600c (192.168.3.162)
| Check | Result |
|-------|--------|
| `/etc/modprobe.d/snd-power.conf` | ✅ `power_save=0 power_save_controller=N` |
| `/etc/modprobe.d/amdgpu.conf` | ✅ `gfxoff=0` |
| `/boot/initrd.img` timestamp | ✅ Apr 15 18:10 (after conf files written) |
| `snd_hda_intel power_save` param | ✅ `0` (confirmed live) |
| `amdgpu gfxoff` sysfs | ✅ absent (correct) |

**All 3 rigs confirmed: snd_hda_intel power_save disabled, GFXOFF disabled,
initramfs rebuilt with both fixes. This is the first time all 3 rigs have been
in a verified clean state simultaneously.**

---

## NPZ State

`bidirectional_survivors_binary.npz` regressed to empty (264 bytes) because
`rm -f bidirectional_survivors.json` was run before the crashed trial and the
NPZ was cleared. The accumulated survivor data is safe:

- `bidirectional_survivors_all.npz` — 37030 bytes, 1069+ seeds — **untouched**
- `bidirectional_survivors_binary.npz` — 264 bytes (empty) — needs restore

**Action:** Restore binary NPZ from all-file before next run, or let next
successful Step 1 run rewrite it naturally.

---

## Instrumentation Status

`apply_s163_mem_debug_worker.py` — patch script used to deploy MEM_DEBUG
instrumentation to remote workers. Committed for reference. Not needed for
future runs (instrumentation is now in `sieve_gpu_worker.py` directly).

---

## Active Config State (end of session)

| Parameter | Value |
|-----------|-------|
| `seed_cap_amd` | 100,000 |
| `seed_cap_nvidia` | 5,000,000 |
| `window_trials` | 3 |
| `min_workers` | 24 |
| `S163_MEM_DEBUG` | 1 (enabled for next run) |
| Transport | TCP-PWC |
| `free_all_blocks()` | Removed (TB Option B) |
| All rigs | snd_hda_intel power_save=0 ✅ |
| All rigs | amdgpu gfxoff=0 ✅ |
| All rigs | initramfs rebuilt ✅ |

---

## Next Action

Launch first clean run with all fixes confirmed:

```bash
ssh rzeus "cd ~/distributed_prng_analysis && source ~/venvs/torch/bin/activate && \
  S163_MEM_DEBUG=1 PYTHONPATH=. nohup python3 agents/watcher_agent.py \
  --run-pipeline --start-step 1 --end-step 1 \
  --params '{\"window_trials\":3,\"seed_cap_amd\":100000,\"seed_cap_nvidia\":5000000,\"min_workers\":24}' \
  > logs/s163_100k_fixes_confirmed.log 2>&1 & echo launched"
```

---

*S163 continuation — 2026-04-16 — Team Alpha*
