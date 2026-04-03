# Team Beta Update — S159G
## Environment Propagation Root Cause Found
**Date:** 2026-04-02  
**Submitted by:** Team Alpha  
**Follows:** TB_SUBMISSION_S159G_RIG6600_CRASHES.md

---

## What We Did

Following TB's P0 ruling to verify `HSA_ENABLE_SDMA=0` actually reaches live worker PIDs, we conducted a systematic investigation of the worker environment across all 3 rigs.

---

## Step 1 — Check Live Worker PID Environment

We checked the actual environment of a running worker PID on each rig:

**rrig6600** (crashed rig):
```
cat /proc/<worker_pid>/environ | tr '\0' '\n' | grep -E 'HSA|SDMA|ROCR|POWER'
→ HSA_OVERRIDE_GFX_VERSION=10.3.0
```
Only ONE variable. Missing: `HSA_ENABLE_SDMA`, `ROCR_VISIBLE_DEVICES`, 
`HSA_ENABLE_RUNTIME_POWER_MGMT`, `AMDGPU_NO_POWER_PROFILE`.

**rrig6600b** (stable rig):
```
→ ROCR_VISIBLE_DEVICES=0
→ HSA_ENABLE_SDMA=0
→ AMDGPU_NO_POWER_PROFILE=1
→ HSA_ENABLE_RUNTIME_POWER_MGMT=0
→ HSA_OVERRIDE_GFX_VERSION=10.3.0
```
Full environment — all 5 variables present.

**rrig6600c** (survived this run):
```
→ ROCR_VISIBLE_DEVICES=0
→ HSA_ENABLE_SDMA=0
→ AMDGPU_NO_POWER_PROFILE=1
→ HSA_ENABLE_RUNTIME_POWER_MGMT=0
→ HSA_OVERRIDE_GFX_VERSION=10.3.0
```
Full environment — all 5 variables present.

**Conclusion:** rrig6600 workers are missing 4 critical ROCm env vars that rrig6600b 
and rrig6600c workers have.

---

## Step 2 — Check Systemd Unit Files

We examined the actual transient systemd unit files on rrig6600b and rrig6600c:

```ini
[Service]
Type=exec
Restart=always
RestartSec=2s
Environment="ROCR_VISIBLE_DEVICES=0" "CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_0" \
            "HSA_OVERRIDE_GFX_VERSION=10.3.0" "HSA_ENABLE_SDMA=0" \
            "HSA_ENABLE_RUNTIME_POWER_MGMT=0" "AMDGPU_NO_POWER_PROFILE=1"
ExecStart="/bin/bash" "-lc" "cd /home/michael/distributed_prng_analysis && \
           exec /home/michael/rocm_env/bin/python -u zmq_sqlite_worker.py ..."
```

The coordinator's `--setenv` flags ARE correctly making it into the systemd unit 
`[Service] Environment=` section on rrig6600b and rrig6600c. Systemd injects these 
directly into the process environment.

**We could not check rrig6600's unit file because rrig6600 had no active workers** — 
it crashed mid-run and the coordinator does not re-spawn workers on crashed rigs. 
When rrig6600 rebooted it rejoined via ZMQ job pulling but had no systemd units.

---

## Step 3 — Why Did rrig6600 Not Get the Env Vars?

This is the remaining open question. Two hypotheses:

### Hypothesis A — Stale Worker Launch (Most Likely)
The workers we checked on rrig6600 earlier in the session were from a **previous run** 
launched before the `HSA_ENABLE_SDMA=0` and other power management env vars were added 
to the coordinator in S159F (commit 561428b). Those workers were still running from the 
old coordinator launch and never got the new env vars.

**Evidence:** The coordinator commit adding power management env vars was `561428b`. 
If rrig6600's workers were launched before that commit was deployed, they would have 
the old env. rrig6600b and rrig6600c were launched fresh after the commit.

### Hypothesis B — Coordinator Code Path Difference
Some code path in `zmq_sqlite_coordinator.py` is treating rrig6600 differently from 
rrig6600b and rrig6600c when building the systemd-run command. The coordinator source 
shows `--setenv` flags at lines 514-519 for AMD rigs — but if rrig6600 was being 
treated as a different node type or hitting a different launch path, it could receive 
a different command.

---

## Step 4 — What Needs To Be Verified Next

TB's acceptance test: **"Does a live worker PID on rrig6600 show HSA_ENABLE_SDMA=0 
in its environment after a fresh coordinator launch?"**

This requires:
1. Kill all current workers on all rigs
2. Fresh coordinator launch with current code (commit 561428b+)
3. Immediately after rrig6600 workers start, check:
```bash
ssh rrig6600 "systemctl --user cat zmq-worker-gpu0.service | grep Environment"
ssh rrig6600 "cat /proc/$(pgrep -f zmq_sqlite_worker | head -1)/environ | tr '\0' '\n' | grep -E 'HSA|SDMA|ROCR|POWER'"
```

If rrig6600 workers show full env → the stale-launch hypothesis was correct, 
and `HSA_ENABLE_SDMA=0` was always the fix, just never verified until now.

If rrig6600 workers still show incomplete env → there is a coordinator code path 
bug treating rrig6600 differently.

---

## Current Run Status

- rrig6600 is UP (ping responding)
- No new crash events in netconsole since 18:04
- rrig6600b and rrig6600c showing only `perf: interrupt took too long` — 
  performance sampling adjustment, not crashes
- Run appears to be completing on rrig6600b and rrig6600c only

---

## Updated TB Questions

1. Should we do a fresh full-cluster launch now to verify rrig6600 gets the full 
   env vars in a clean start?
2. If the fresh launch confirms rrig6600 gets full env, should we consider the 
   crash resolved pending one more full-cluster stability run?
3. If the fresh launch still shows missing env on rrig6600, should we patch the 
   coordinator to explicitly verify env propagation before accepting worker registration?

---

*Team Alpha Update — S159G — 2026-04-02*
