# SESSION CHANGELOG - 2026-01-25

## Manifest Parameter Precedence Fixes

### Root Cause
CLI testing bypasses manifests. Parameters tuned via CLI were never synced to `agent_manifests/*.json`, causing WATCHER runs to use stale defaults.

### Manifests Fixed

| File | Parameter | Old | New | Impact |
|------|-----------|-----|-----|--------|
| `window_optimizer.json` | max_seeds | 10,000,000 | 100,000 | 2hr timeout → 40min |
| `window_optimizer.json` | trials | 50 | 50 | (verified correct) |
| `full_scoring.json` | chunk_size | 5,000 | 1,000 | OOM fix for 7.7GB rigs |
| `scorer_meta.json` | sample_size | 25,000 | 450 | 4.5× throughput improvement |

### Documentation Added

**Chapter 12 - CRITICAL LESSON:**
> "When fixing params via CLI testing, ALWAYS update the manifest too!"

**Chapters 4, 9, 12:**
- Manifest parameter precedence explanation
- Table showing CLI vs WATCHER config sources
- Debugging checklist

### Step 3 Results (Post-Fix)
- 100/100 jobs completed
- 0 failures (was 2 OOM failures before fix)
- 99,941 survivors scored
- 64 features per survivor
- Runtime: 5:20

### Files Modified
- `agent_manifests/window_optimizer.json`
- `agent_manifests/full_scoring.json`
- `agent_manifests/scorer_meta.json`
- `CHAPTER_12_WATCHER_AGENT.md`
- `CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md`
- `CHAPTER_4_FULL_SCORING.md`

### Git Commits
- `fix: chunk_size 5000→1000 in full_scoring manifest (OOM fix)`
- `fix: Sync manifests + document CLI/manifest lesson`
