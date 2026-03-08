# Session Changelog — S128
**Date:** 2026-03-08
**Commit:** pre-patch (baseline established this session)
**Tag:** n/a (S128 was diagnostic only, no code changes committed)
**Status:** CLOSED

---

## Summary

S128 was a diagnostic and baseline-establishment session. No code was modified or committed. The primary deliverable was a confirmed aggregate throughput baseline for the residue sieve pipeline, used as the reference point for all subsequent S129B/S130 work.

---

## Phase C Baseline Run

**Command:**
```bash
python3 coordinator.py --method residue_sieve \
  --seed-cap-amd 2000000 --seed-cap-nvidia 5000000 \
  -s 200000000 daily3.json
```

**Result:**
- Total seeds: 200,000,000
- Duration: ~235.5s
- Aggregate sps: **849,469 sps**
- Jobs: 100/100 complete, 0 failures

This number became the S128 baseline. All subsequent throughput comparisons reference 849,469 sps.

---

## Investigation: seed_cap parameter flow

S128 confirmed that `seed_cap_nvidia` and `seed_cap_amd` do not flow through the WATCHER → `window_optimizer.py` → `MultiGPUCoordinator` call chain. `window_optimizer_integration_final.py` instantiates `MultiGPUCoordinator` with no cap arguments — constructor defaults apply silently. The CLI defaults only take effect when `coordinator.py` is run as `__main__`. Both params are absent from `window_optimizer.json` manifest, so WATCHER would drop them silently (S97 class failure pattern). This was documented as a carry-forward item for a future patch.

---

## Carry-Forward Items Identified

- Seed cap patch to `window_optimizer_integration_final.py` (4 instantiation sites) and `agent_manifests/window_optimizer.json` — not yet implemented
- `apply_caps.py` — needs to be run with final measured values

---

## Rig Status at Session Close

| Rig | IP | Status |
|-----|----|--------|
| rrig6600 | 192.168.3.120 | ✅ HEALTHY — 8× RX 6600 XT |
| rrig6600b | 192.168.3.154 | ✅ HEALTHY — 8× RX 6600 |
| rrig6600c | 192.168.3.162 | ⚠️ PARTIAL — 8× RX 6600, i5-8400T ~50% throughput deficit |
