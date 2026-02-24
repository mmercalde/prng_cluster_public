# SESSION CHANGELOG - January 23, 2026

## Summary

**CRITICAL FIX:** NPZ v3.0 metadata preservation - resolves silent ML feature corruption.

---

## Issue Discovered

### NPZ Metadata Loss Bug

**Severity:** High (silent data corruption)  
**Root Cause:** Process evolution oversight during NPZ/ramdisk optimization

**Timeline:**
| Date | Event |
|------|-------|
| Jan 3, 2026 | NPZ format created for Step 2 (3 arrays: seeds, forward_matches, reverse_matches) |
| Jan 3, 2026 | v1.9.1 comment added: "metadata already in chunk file" — true at the time |
| Jan 21-22, 2026 | Step 3 switched from JSON to NPZ for ramdisk optimization |
| Jan 23, 2026 | Bug discovered: 14 of 47 ML features silently zeroed |

**Impact:**
- `bidirectional_survivors.json` contains **22 fields** per survivor
- `bidirectional_survivors_binary.npz` (v2.0) contained only **3 arrays**
- **19 metadata fields were silently dropped**
- **14 ML features always computed as 0.0** in Step 3

**Affected Features (were always 0.0):**
```
skip_min, skip_max, skip_range
forward_count, reverse_count, bidirectional_count
intersection_count, intersection_ratio, intersection_weight
forward_only_count, reverse_only_count
survivor_overlap_ratio, bidirectional_selectivity
```

---

## Resolution

### NPZ v3.0 - Full Metadata Preservation

**Files Modified:**

| File | Version | Change |
|------|---------|--------|
| `convert_survivors_to_binary.py` | v2.0 → v3.0 | Save all 22 fields to NPZ |
| `utils/survivor_loader.py` | v1.0 → v2.0 | Detect NPZ version, reconstruct full objects from v3 |
| `bidirectional_survivors_binary.npz` | regenerated | Now contains 22 arrays |

**NPZ v3.0 Format:**

```python
# Core arrays (v1.0 compatibility)
seeds              # uint32
forward_matches    # float32
reverse_matches    # float32

# Metadata arrays (v3.0 addition)
window_size        # int32
offset             # int32
trial_number       # int32
skip_min           # int32
skip_max           # int32
skip_range         # int32
forward_count      # float32
reverse_count      # float32
bidirectional_count      # float32
intersection_count       # float32
intersection_ratio       # float32
intersection_weight      # float32
bidirectional_selectivity # float32
forward_only_count       # float32
reverse_only_count       # float32
survivor_overlap_ratio   # float32
score              # float32
skip_mode          # uint8 (encoded: 0=constant, 1=variable)
prng_type          # uint8 (encoded: 0=java_lcg, 1=java_lcg_reverse, ...)
```

**File Size Comparison:**

| Format | Size | Arrays |
|--------|------|--------|
| JSON (source) | 57.9 MB | 22 fields |
| NPZ v2.0 (old) | 0.6 MB | 3 arrays |
| NPZ v3.0 (new) | 733 KB | 22 arrays |

**Compression ratio:** 80.8x (vs JSON)

---

## Verification Results

```
Arrays in NPZ: 22
Expected:      22

✅ All critical metadata fields present

Sample values (first survivor):
  seed:               0
  skip_min:           8
  skip_max:           230
  forward_count:      48754.0
  bidirectional_count:48754.0

✅ Metadata fields contain real data (not all zeros)
```

**Note:** `forward_only_count` and `reverse_only_count` are correctly zero because bidirectional survivors are the intersection (passed both sieves). No forward-only or reverse-only seeds exist in this dataset by definition.

---

## survivor_loader.py v2.0 Features

- **Auto-detect NPZ version:** Distinguishes v1 (3 arrays) from v3 (22 arrays)
- **Full object reconstruction:** `_array_to_dict()` now reconstructs all 22 fields
- **Backward compatible:** Works with both v1 and v3 NPZ files
- **New `npz_version` field:** `SurvivorData` now reports detected version
- **New convenience function:** `get_survivor_metadata()` returns `{seed: metadata_dict}`

---

## Deployment Steps Completed

1. ✅ Patched `convert_survivors_to_binary.py` → v3.0
2. ✅ Patched `utils/survivor_loader.py` → v2.0
3. ✅ Regenerated NPZ with all 22 fields
4. ✅ Verified metadata presence and non-zero values

---

## Remaining Steps

1. Copy NPZ to remote rigs:
   ```bash
   scp bidirectional_survivors_binary.npz 192.168.3.120:~/distributed_prng_analysis/
   scp bidirectional_survivors_binary.npz 192.168.3.154:~/distributed_prng_analysis/
   ```

2. Copy updated loader to remote rigs:
   ```bash
   scp utils/survivor_loader.py 192.168.3.120:~/distributed_prng_analysis/utils/
   scp utils/survivor_loader.py 192.168.3.154:~/distributed_prng_analysis/utils/
   ```

3. Test Step 3:
   ```bash
   PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 3 --end-step 3
   ```

4. Git commit:
   ```bash
   git add convert_survivors_to_binary.py utils/survivor_loader.py bidirectional_survivors_binary.npz
   git commit -m 'fix(npz): preserve all 22 metadata fields (v3.0)
   
   CRITICAL FIX: NPZ v2.0 silently dropped 19 metadata fields,
   causing 14/47 ML features to be zeroed in Step 3.
   
   - convert_survivors_to_binary.py: v2.0 → v3.0 (22 arrays)
   - utils/survivor_loader.py: v1.0 → v2.0 (version detection)
   - Regenerated bidirectional_survivors_binary.npz
   
   Team Beta ruling: January 23, 2026'
   ```

5. Update documentation:
   - INSTRUCTIONS_NPZ_ADDITION.md
   - CHAPTER_4_FULL_SCORING.md (reference NPZ v3.0)

---

## Team Beta Position

✔ Confirmed bug with clear root cause  
✔ No architectural rollback required  
✔ NPZ remains correct long-term format  
✔ Fix is localized, safe, and testable  
✔ Backward compatible (loader handles v1 and v3)

---

## Git Commit

```
fix(npz): preserve all 22 metadata fields (v3.0) - Jan 23, 2026

Files: convert_survivors_to_binary.py, utils/survivor_loader.py
```
