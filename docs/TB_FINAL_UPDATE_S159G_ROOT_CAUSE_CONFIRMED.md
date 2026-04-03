# Team Beta Final Update — S159G
## Root Cause Confirmed: rrig6600 Workers Missing Critical ROCm Env Vars
**Date:** 2026-04-02  
**Submitted by:** Team Alpha  
**Status:** Root cause confirmed — fix identified — pending implementation

---

## Summary

We have confirmed the root cause of rrig6600 crashes. It is an environment
propagation failure. rrig6600 workers consistently run without 4 critical ROCm
protective variables that rrig6600b and rrig6600c workers always have.

---

## The Evidence — Side by Side

We checked live worker PID environments on all 3 rigs during an active run:

**rrig6600 (crashes):**
```
HSA_OVERRIDE_GFX_VERSION=10.3.0   ← only this
```

**rrig6600b (stable):**
```
ROCR_VISIBLE_DEVICES=0
HSA_ENABLE_SDMA=0
AMDGPU_NO_POWER_PROFILE=1
HSA_ENABLE_RUNTIME_POWER_MGMT=0
HSA_OVERRIDE_GFX_VERSION=10.3.0
```

**rrig6600c (stable):**
```
ROCR_VISIBLE_DEVICES=0
HSA_ENABLE_SDMA=0
AMDGPU_NO_POWER_PROFILE=1
HSA_ENABLE_RUNTIME_POWER_MGMT=0
HSA_OVERRIDE_GFX_VERSION=10.3.0
```

This is consistent across EVERY run we tested — including a fresh launch today.

---

## Why It Happens

The coordinator launches workers via `systemd-run --user` with `--setenv` flags
for all 5 variables. On rrig6600b and rrig6600c this works correctly — the
transient systemd unit files contain the full `Environment=` block and workers
inherit all 5 vars.

On rrig6600, the coordinator also reports `systemd-run worker active` in the
log — so the launch command is identical. But the env vars do NOT reach the
worker PID. Only `HSA_OVERRIDE_GFX_VERSION` from `/etc/environment` survives.

The most likely explanation is that rrig6600's `bash -lc` login shell startup
sequence is interfering with the systemd-injected environment before `exec`
hands off to python. Specifically:

- rrig6600's `.bashrc` has `if [ -z "$VIRTUAL_ENV" ] && [ -n "$PS1" ]` guard
  before sourcing `rocm_env/bin/activate` — the `$PS1` check means activate
  is never sourced in non-interactive shells
- rrig6600b's `.bashrc` has `if [ -d "$HOME/rocm_env" ]` — no `$PS1` check,
  activate always sourced
- The virtualenv activate script or the login shell startup on rrig6600 is
  clearing or not preserving the systemd-injected vars

The net result: rrig6600 workers run the GPU kernel without `HSA_ENABLE_SDMA=0`,
exposing them to the SDMA engine bug that causes the GPU VM fault cascade
seen in netconsole.

---

## Additional Confirmation — Trial Continued After Crash

When rrig6600 crashed mid-run, the ZMQ SQLite coordinator successfully:
1. Detected expired leases
2. Redistributed chunks to rrig6600b and rrig6600c
3. Completed the forward and reverse sieves without data loss

This confirms the coordinator architecture is crash-resilient. But rrig6600
continued crashing on every subsequent run because the env var defect is
structural — it reproduces on every fresh launch.

---

## The Fix

The simplest and most robust fix is to add the 4 missing variables to
`/etc/environment` on rrig6600. This file is read by PAM for every process
on the system, bypassing the entire bash/systemd/activate chain entirely:

```bash
sudo tee -a /etc/environment << 'ENVEOF'
HSA_ENABLE_SDMA=0
HSA_ENABLE_RUNTIME_POWER_MGMT=0
AMDGPU_NO_POWER_PROFILE=1
ROCR_VISIBLE_DEVICES=0
ENVEOF
```

Note: `ROCR_VISIBLE_DEVICES=0` limits to GPU 0 per worker — this is handled
per-worker by the coordinator's `--setenv` flags. Adding it globally to
`/etc/environment` may conflict with multi-GPU worker assignments. TB guidance
requested on whether to include `ROCR_VISIBLE_DEVICES` in `/etc/environment`
or only the 3 power/SDMA variables.

**Safe set for `/etc/environment`:**
```
HSA_ENABLE_SDMA=0
HSA_ENABLE_RUNTIME_POWER_MGMT=0
AMDGPU_NO_POWER_PROFILE=1
```

`ROCR_VISIBLE_DEVICES` should remain coordinator-managed via `--setenv`.

---

## TB Questions

1. Confirm: add `HSA_ENABLE_SDMA=0`, `HSA_ENABLE_RUNTIME_POWER_MGMT=0`,
   `AMDGPU_NO_POWER_PROFILE=1` to `/etc/environment` on rrig6600 only, or
   all 3 rigs for consistency?

2. Should we also fix the `.bashrc` `[ -n "$PS1" ]` guard on rrig6600 to
   match rrig6600b (unconditional activate), as a belt-and-suspenders fix?

3. After the `/etc/environment` fix is applied and rrig6600 is rebooted,
   one clean full-cluster stability run — if rrig6600 survives, S159G is
   resolved?

---

*Team Alpha Final Update — S159G — 2026-04-02*
