# DOCUMENTATION AUDIT — January 31, 2026

## Summary

**Status: 8 project knowledge documents are STALE**

Every document predates the rig-6600c integration (Jan 30-31, 2026) and the GPU redistribution. Production code is correct (all 22 files patched), but documentation will mislead any future Claude session or human reader.

**Actual Cluster State (Production):**

| Node | IP | GPUs | Type | max_concurrent |
|------|----|------|------|----------------|
| Zeus | localhost | 2 | RTX 3080 Ti | 2 |
| rig-6600 | 192.168.3.120 | 8 | RX 6600 | 8 |
| rig-6600b | 192.168.3.154 | 8 | RX 6600 | 8 |
| rig-6600c | 192.168.3.162 | 8 | RX 6600 | 8 |
| **Total** | | **26** | | **26** |

**ROCm version**: 6.4.3 (docs say 5.7)

---

## File-by-File Discrepancies

---

### 1. `instructions.txt`

| # | What | Says | Should Say |
|---|------|------|------------|
| 1 | ROCm prelude hostname list | `["rig-6600", "rig-6600b"]` | `["rig-6600", "rig-6600b", "rig-6600c"]` |
| 2 | GPU counts | 12 per rig | 8 per rig |
| 3 | Test expected output | `# Expected: 12` | `# Expected: 8` |
| 4 | rig-6600c test commands | Missing entirely | Add test for 192.168.3.162 |
| 5 | Hardware Architecture total | "26 GPUs" (correct total, wrong distribution) | "26 GPUs across 4 nodes (2+8+8+8)" |
| 6 | distributed_config.json example | 2 AMD nodes, 12 GPUs each | 3 AMD nodes, 8 GPUs each |
| 7 | Zeus python_env | `/home/michael/venvs/tf/bin/python` | `/home/michael/venvs/torch/bin/python` |
| 8 | rig-6600c IP | Not mentioned | 192.168.3.162 |
| 9 | ROCm prelude "Critical Files" list | 3 files | 22 files have ROCm prelude (list subset OK but note it) |

---

### 2. `Cluster_operating_manual.txt`

| # | What | Says | Should Say |
|---|------|------|------------|
| 1 | Hardware Architecture | "rig-6600: 12x RX 6600, rig-6600b: 12x RX 6600" | "rig-6600: 8x, rig-6600b: 8x, rig-6600c: 8x" |
| 2 | Total GPUs | "26 GPUs, ~285.69 TFLOPS" | "26 GPUs across 4 nodes" |
| 3 | ROCm prelude | `["rig-6600", "rig-6600b"]` | Add `"rig-6600c"` |
| 4 | Test commands | Only rig-6600 and rig-6600b | Add rig-6600c (192.168.3.162) |
| 5 | Expected GPU count | `# Expected: 12` | `# Expected: 8` |
| 6 | rig-6600c | Not mentioned anywhere | Add as 4th node throughout |
| 7 | ROCm version | Implied 5.7 | 6.4.3 |

---

### 3. `COMPLETE_OPERATING_GUIDE_v1.1.md`

| # | What | Says | Should Say |
|---|------|------|------------|
| 1 | Version header | "v1.1.0 December 2025" | Needs v1.2.0 January 2026 |
| 2 | Title line | "26-GPU Cluster Architecture" | Still 26, but update subtitle to show 4 nodes |
| 3 | Hardware table | rig-6600: 12, rig-6600b: 12 | rig-6600: 8, rig-6600b: 8, rig-6600c: 8 |
| 4 | Network IPs | Only 192.168.3.120 and .154 | Add 192.168.3.162 |
| 5 | SSH keys section | "rig-6600 and rig-6600b" | Add rig-6600c |
| 6 | ROCm prelude everywhere | `["rig-6600", "rig-6600b"]` | Add `"rig-6600c"` |
| 7 | "Files Requiring ROCm Prelude" | 6 files listed | Note that 22 files now have prelude |
| 8 | Software Dependencies | "ROCm 5.7+" | "ROCm 6.4.3" |

---

### 4. `CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md`

| # | What | Says | Should Say |
|---|------|------|------------|
| 1 | Hardware Topology diagram | 3 nodes, 3rd is "(planned)" | 4 nodes, rig-6600c is live |
| 2 | GPU counts in diagram | "12× RX 6600" per rig | "8× RX 6600" per rig |
| 3 | VRAM per rig | "96GB VRAM" (12×8GB) | "64GB VRAM" (8×8GB) |
| 4 | Total VRAM | "216GB VRAM" | "216GB VRAM" (still correct: 24×8+2×12=216) |
| 5 | Performance table | 2 rig rows with "12× RX 6600" | 3 rig rows with "8× RX 6600" |
| 6 | TFLOPS per rig | ~108 (12×8.93) | ~71.4 (8×8.93) |
| 7 | Total TFLOPS | ~285 | ~282 (still close enough) |
| 8 | ROCm prelude | `["rig-6600", "rig-6600b"]` | Add `"rig-6600c"` |
| 9 | PER_NODE_CONCURRENCY dict | zeus:2, rig-6600:1, rig-6600b:1 | Add rig-6600c:1 (or update all to 8 per production config) |
| 10 | ROCm version | "ROCm 5.7" | "ROCm 6.4.3" |
| 11 | Ramdisk deployment loop | 3 nodes (localhost, .120, .154) | 4 nodes (add .162) |
| 12 | Ramdisk verification expected | "3 nodes × 2 files" | "4 nodes × 2 files" |
| 13 | max_concurrent_script_jobs | 12 per rig | 8 per rig |
| 14 | Validated Configuration | `max_concurrent_script_jobs: 12` | `max_concurrent_script_jobs: 8` |

---

### 5. `CHAPTER_2_BIDIRECTIONAL_SIEVE.md`

| # | What | Says | Should Say |
|---|------|------|------------|
| 1 | ROCm prelude | `("rig-6600", "rig-6600b")` | Add `"rig-6600c"` |

---

### 6. `README.md`

| # | What | Says | Should Say |
|---|------|------|------------|
| 1 | Node table | rig-6600: 12×, rig-6600b: 12× | rig-6600: 8×, rig-6600b: 8×, rig-6600c: 8× |
| 2 | Planned node | "rig-6600xt (planned)" | Remove — rig-6600c is the actual 4th node |
| 3 | ROCm activation | `source ~/tf/bin/activate` | `source ~/rocm_env/bin/activate` |
| 4 | "Scale across 26 GPUs" | Correct total | Still correct |

---

### 7. `CHAPTER_3_SCORER_META_OPTIMIZER.md`

| # | What | Says | Should Say |
|---|------|------|------------|
| 1 | No explicit hostname refs | N/A — references coordinator/config | Likely OK, but verify |

---

### 8. `CHAPTER_12_ADDENDUM_v1_3_0.md` / `CHAPTER_12_WATCHER_AGENT.md`

| # | What | Says | Should Say |
|---|------|------|------------|
| 1 | Any hostname refs to check | Need to verify | Verify no stale rig-6600/rig-6600b-only references |

---

## Cross-Cutting Issues (Affect ALL Docs)

### A. ROCm Prelude Pattern
Every occurrence of:
```python
if HOST in ["rig-6600", "rig-6600b"]:
```
Must become:
```python
if HOST in ["rig-6600", "rig-6600b", "rig-6600c"]:
```

**Affected docs**: instructions.txt, Cluster_operating_manual.txt, COMPLETE_OPERATING_GUIDE_v1.1.md, CHAPTER_9, CHAPTER_2

### B. GPU Count Pattern
Every occurrence of "12" GPUs per rig must become "8".
Every occurrence of "12× RX 6600" must become "8× RX 6600".

**Affected docs**: ALL except CHAPTER_3

### C. Node Count Pattern
Every "3-node" reference must become "4-node".
Every loop/list over (zeus, rig-6600, rig-6600b) must add rig-6600c.

**Affected docs**: ALL

### D. ROCm Version
5.7 → 6.4.3

**Affected docs**: CHAPTER_9, COMPLETE_OPERATING_GUIDE

---

## Priority Order for Updates

1. **CRITICAL** — `instructions.txt` (Claude reads this first every session)
2. **CRITICAL** — `Cluster_operating_manual.txt` (operational reference)
3. **HIGH** — `COMPLETE_OPERATING_GUIDE_v1.1.md` (comprehensive guide)
4. **HIGH** — `CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md` (cluster-specific)
5. **MEDIUM** — `README.md` (external-facing)
6. **LOW** — `CHAPTER_2_BIDIRECTIONAL_SIEVE.md` (single line)
7. **LOW** — Remaining chapters (verify and patch)

---

## What IS Correct (No Changes Needed)

- `distributed_config.json` — Fixed and committed (4d979e0)
- Production code — All 22 files patched with rig-6600c
- `scripts_coordinator.py` — ROCM_HOSTNAMES includes rig-6600c
- `run_step3_full_scoring.sh` — Reads nodes dynamically from config
- Total GPU count of 26 — Coincidentally still correct (was 2+12+12, now 2+8+8+8)
