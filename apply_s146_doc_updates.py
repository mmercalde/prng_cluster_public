#!/usr/bin/env python3
"""
apply_s146_doc_updates.py — S147 documentation patch script
Applies S146 PWC invariants, hybrid kernel signatures, dashboard fixes
to 5 chapter documents on Zeus.

Usage:
    python3 apply_s146_doc_updates.py --dry-run
    python3 apply_s146_doc_updates.py
"""
import os
import shutil
import sys
import argparse

DOCS = os.path.expanduser("~/distributed_prng_analysis/docs")
ROOT = os.path.expanduser("~/distributed_prng_analysis")
DRY_RUN = False


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def backup(path):
    bak = path + ".bak_s147"
    if not os.path.exists(bak):
        shutil.copy(path, bak)
        print(f"  BAK  {os.path.basename(bak)}")
    else:
        print(f"  BAK  (already exists, skipping copy) {os.path.basename(bak)}")


def section_marker(section_text):
    """
    Return a marker string that IS actually written into the document — the
    section's own top-level (`## `) heading.

    This exists because the previous idempotency guard tested `label in content`,
    and `label` (e.g. "CHAPTER_1 PWC S146 kernel invariants") is a caller-side
    identifier that is NEVER written into any document. The guard could therefore
    never fire, and every run appended the section again. That is how
    docs/CHAPTER_1_WINDOW_OPTIMIZER.md acquired a verbatim duplicate of its
    S146 kernel-invariants section.
    """
    for line in section_text.strip().splitlines():
        if line.startswith("## "):
            return line.strip()
    return None


def apply_append(filepath, label, section_text):
    """Append section_text at end of file (after a final newline)."""
    path = os.path.join(DOCS, filepath)
    if not os.path.exists(path):
        print(f"  SKIP {label}: file not found at {path}")
        return False

    content = read_file(path)
    marker = section_marker(section_text)
    if marker is None:
        # Fail closed: without a marker we cannot prove idempotency, so refuse to
        # append rather than risk another duplicate.
        print(f"  SKIP {label}: section has no '## ' heading — cannot verify idempotency")
        return False
    if marker in content:
        print(f"  SKIP {label}: already patched (marker present: {marker!r})")
        return False

    before = len(content.splitlines())
    new_content = content.rstrip("\n") + "\n\n" + section_text.strip() + "\n"
    after = len(new_content.splitlines())

    if DRY_RUN:
        print(f"  DRY  {label}: would append {after - before} lines ({before} → {after})")
        return True

    backup(path)
    write_file(path, new_content)
    print(f"  OK   {label}: appended {after - before} lines ({before} → {after})")
    return True


def apply_replace(filepath, label, old_text, new_text):
    """Replace exactly one occurrence of old_text with new_text."""
    path = os.path.join(DOCS, filepath)
    if not os.path.exists(path):
        print(f"  SKIP {label}: file not found at {path}")
        return False

    content = read_file(path)
    count = content.count(old_text)
    if count == 0:
        print(f"  SKIP {label}: anchor text not found (0 matches)")
        return False
    if count > 1:
        print(f"  WARN {label}: anchor matches {count} times — skipping (ambiguous)")
        return False

    before = len(content.splitlines())
    new_content = content.replace(old_text, new_text, 1)
    after = len(new_content.splitlines())

    if DRY_RUN:
        print(f"  DRY  {label}: would replace anchor ({before} → {after} lines)")
        return True

    backup(path)
    write_file(path, new_content)
    print(f"  OK   {label}: replaced anchor ({before} → {after} lines)")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Patch 1: CHAPTER_9 — append PWC S146 architecture section
# ─────────────────────────────────────────────────────────────────────────────
CH9_SECTION = """
---

## Persistent Worker Coordinator (PWC) — Architecture Invariants (S146)

**Files:** `persistent_worker_coordinator.py`, `sieve_gpu_worker.py`

### Overview

The Persistent Worker Coordinator (PWC) is the production Step 1 sieve execution backend
when `--use-persistent-workers` is active. Workers are kept alive between trials, accepting
job payloads over SSH. Zeus dispatches local jobs; remote rigs receive jobs via `sieve_gpu_worker.py --persistent`.

### S146 Validated Architecture

| Parameter | Value | Notes |
|-----------|-------|-------|
| `worker_pool_size` | **4** | Validated stable. Do NOT increase to 8. |
| `JOB_TIMEOUT_S` | **600** | (not 300) — allow time for large seed ranges |
| `_localhost_semaphore` | `threading.Semaphore(2)` | Zeus has 2 GPUs; mirrors coordinator.py line 269 |

### Hybrid Kernel Signature Invariants (CRITICAL)

GPU kernel arg tails differ between forward and reverse hybrid kernels.
These **must not be conflated** — they have different argument tails:

| Kernel direction | Tail args | Notes |
|-----------------|-----------|-------|
| Forward hybrid  | `..., threshold, unsigned long long a, unsigned long long c` | a,c passed explicitly |
| Reverse hybrid  | `..., threshold, int offset` | a,c hardcoded inside kernel |

**Rule:** `sieve_gpu_worker.py` must split the hybrid `elif` into separate forward and
reverse branches and pass the correct tail for each. Sending `a, c` to a reverse hybrid
kernel → crash.

### Threshold Routing (Hybrid)

- Hybrid kernels and hybrid post-filter **must use `phase2_threshold`**, not `min_match_threshold`.
- `hybrid_threshold = coerce_threshold(phase2_raw, threshold) if phase2_raw is not None else threshold`
- Post-filter: `if rate >= hybrid_threshold:` (NOT `if rate >= threshold:`)

### Strategy Dict Requirement

`sieve_filter.py` calls `StrategyConfig(**s)` which requires all 6 fields:

```
name, max_consecutive_misses, skip_tolerance,
enable_reseed_search, skip_learning_rate, breakpoint_threshold
```

Both `persistent_worker_coordinator.py` and `sieve_gpu_worker.py` must send the full
`StrategyConfig.to_dict()` result — never a partial dict.

### Dashboard Integration (ProgressWriter)

PWC must call these on every chunk/trial to drive live dashboard throughput:

```python
# After every successful chunk:
self._progress_writer.log_gpu_result(hostname, gpu_id, gpu_type, seeds_in_chunk, elapsed)

# After every completed trial:
self._progress_writer.update_trial_stats(trial_id, survivors_found, ...)
```

Dashboard: `45.32.131.224:5002`

### S146 Validation Result

3 trials, 10M seeds, all 4 sieve passes — zero errors:

| Pass | Status |
|------|--------|
| java_lcg forward | ✅ |
| java_lcg_reverse | ✅ |
| java_lcg_hybrid forward | ✅ |
| java_lcg_hybrid_reverse | ✅ |
"""

# ─────────────────────────────────────────────────────────────────────────────
# Patch 2: CHAPTER_1 — append PWC S146 kernel invariants note
# ─────────────────────────────────────────────────────────────────────────────
CH1_SECTION = """
---

## Persistent Worker Mode — S146 Kernel Invariants

When `--use-persistent-workers` is active, Step 1 dispatches sieve jobs via
`persistent_worker_coordinator.py` → `sieve_gpu_worker.py`. The following invariants
were validated in S146 and must be preserved in any future modifications:

### Hybrid Kernel Arg Tails (CRITICAL)

```
Forward hybrid:  kernel_args = (..., threshold, a, c)
Reverse hybrid:  kernel_args = (..., threshold, offset)   # a,c hardcoded in kernel
```

These are **not interchangeable**. Passing `(threshold, a, c)` to a reverse hybrid kernel
causes an immediate crash.

### Threshold Invariant

Hybrid families use `phase2_threshold` for both kernel invocation and post-filter check.
Base threshold (`min_match_threshold`) is used only for constant-skip families.

### int32 Casts

All scalar kernel args must be explicitly cast: `cp.int32(n_seeds)`, `cp.int32(k)`,
`cp.int32(skip_min)`, `cp.int32(skip_max)`. ROCm/CuPy requires explicit types.

### Count Clamp (defensive)

```python
count = min(int(survivor_count_gpu[0].get()), n_seeds)
```

Applied to both hybrid and non-hybrid extraction paths to prevent buffer overrun on
corrupt kernel writes.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Patch 3: CHAPTER_2 — append PWC sieve execution path section
# ─────────────────────────────────────────────────────────────────────────────
CH2_SECTION = """
---

## 15. Persistent Worker Execution Path (S146)

### Two Sieve Execution Backends

As of S146, Step 1 supports two sieve execution backends:

| Mode | Flag | Backend path |
|------|------|-------------|
| Default (legacy) | (none) | `coordinator.py` → `sieve_filter.py` |
| Persistent workers | `--use-persistent-workers` | `PWC` → `sieve_gpu_worker.py --persistent` |

The persistent worker path keeps sieve workers alive between trials, eliminating
SSH process spawn overhead on every chunk.

### Persistent Worker Call Chain

```
watcher_agent.py
  └─► window_optimizer_integration_final.py  (use_persistent_workers=True)
        └─► run_trial_persistent()   (persistent_worker_coordinator.py:669)
              └─► PersistentWorkerCoordinator
                    Zeus:    execute_local_sieve_job()  ──► sieve_filter.py
                    Remote:  _dispatch_to_worker()       ──► sieve_gpu_worker.py --persistent
```

### Hybrid Routing in sieve_gpu_worker.py

`sieve_gpu_worker.py` handles four sieve pass types:

| Pass type | Kernel family field | Arg tail |
|-----------|---------------------|----------|
| Constant skip forward | `prng_families` (base) | standard |
| Constant skip reverse | `prng_families` (reverse) | standard |
| Hybrid forward | `prng_families` (hybrid) | `threshold, a, c` |
| Hybrid reverse | `prng_families` (hybrid_reverse) | `threshold, offset` |

The hybrid forward and reverse branches are implemented as **separate elif blocks** —
they must not share kernel_args construction.

### S146 Validation

All 4 pass types validated on live hardware (3 trials, 10M seeds, Zeus + 3 rigs):
313 bidirectional survivors found (274 constant + 40 variable skip).
666 total in NPZ accumulator after S146 preprod run.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Patch 4: CHAPTER_12 — append Step 1 PWC note at end
# ─────────────────────────────────────────────────────────────────────────────
CH12_SECTION = """
---

## 12. Step 1 Execution Path — Persistent Worker Mode (S146)

### Updated Step 1 Dispatch

When WATCHER executes Step 1 with `--use-persistent-workers`, the sieve backend
is `persistent_worker_coordinator.py` (PWC) rather than the legacy `coordinator.py` path.

**Manifest flag:** `window_optimizer.json` must include `"use_persistent_workers": true` in
the `args_map` section (or pass `--use-persistent-workers` in the launch args).

### WATCHER Step 1 Confidence Threshold

WATCHER issues PROCEED on Step 1 when:
- Bidirectional survivor count > 0
- WATCHER confidence = 1.00 (no anomalies)

S146 preprod run: 313 survivors → confidence 1.00 → PROCEED.

### PWC Architecture Invariants for WATCHER Integration

| Invariant | Value |
|-----------|-------|
| `worker_pool_size` | 4 (not 8) |
| `JOB_TIMEOUT_S` | 600 |
| Localhost semaphore | `threading.Semaphore(2)` |
| Strategy dict fields | All 6 required (`name`, `max_consecutive_misses`, `skip_tolerance`, `enable_reseed_search`, `skip_learning_rate`, `breakpoint_threshold`) |

These invariants are enforced in `persistent_worker_coordinator.py` and
`sieve_gpu_worker.py` as of commit `7e4ae02` (S146).
"""

# ─────────────────────────────────────────────────────────────────────────────
# Patch 5: COMPLETE_OPERATING_GUIDE — insert PWC procedures before end marker
# ─────────────────────────────────────────────────────────────────────────────
COG_OLD = """---

**— End of Document —**"""

COG_NEW = """---

## 11. Persistent Worker Coordinator (PWC) Operating Procedures (S146)

### Overview

The Persistent Worker Coordinator (`persistent_worker_coordinator.py`) is the production
sieve backend for Step 1. Workers persist between trials, reducing SSH spawn overhead.

### Launch with PWC

```bash
# Via WATCHER (recommended)
PYTHONPATH=. python3 agents/watcher_agent.py \\
  --run-pipeline --start-step 1 --end-step 1 \\
  --params '{"use_persistent_workers": true}'

# Via sweep script
bash sweep_run1.sh          # uses manifest: max_seeds=1B, trials=50
bash sweep_preprod.sh       # uses manifest: max_seeds=50M, trials=5  (validation)
```

### Validated Operating Parameters

| Parameter | Value | Do NOT change |
|-----------|-------|---------------|
| `worker_pool_size` | 4 | Increasing to 8 causes instability |
| `JOB_TIMEOUT_S` | 600 | 300 is too short for large seed ranges |
| `_localhost_semaphore` | `Semaphore(2)` | Zeus has 2 GPUs — must match GPU count |

### Hybrid Kernel Invariants

Forward hybrid kernel expects: `..., threshold, a, c`
Reverse hybrid kernel expects: `..., threshold, offset` (a,c hardcoded)

These are distinct. Mixing them causes an immediate crash at kernel launch.
Threshold for hybrid families is always `phase2_threshold`, not `min_match_threshold`.

### Dashboard Monitoring

Dashboard: `http://45.32.131.224:5002`

PWC writes to `/tmp/cluster_progress.json` via `ProgressWriter`. Live per-node
throughput is updated after every chunk via `log_gpu_result()`. Trial survivor counts
are updated after each trial via `update_trial_stats()`.

If dashboard shows 0 seeds/sec during a run, check that `log_gpu_result()` is being
called in the PWC chunk completion path.

### Kill All Workers

```bash
ssh rzeus "pkill -f 'watcher_agent.py'; pkill -f 'window_optimizer.py'"
ssh rrig6600  "pkill -f sieve_gpu_worker 2>/dev/null"
ssh rrig6600b "pkill -f sieve_gpu_worker 2>/dev/null"
ssh rrig6600c "pkill -f sieve_gpu_worker 2>/dev/null"
```

### S146 Validation Summary

Pre-production run: 3 trials, 10M seeds, all 4 sieve passes clean.
313 bidirectional survivors (274 constant-skip + 40 variable-skip).
WATCHER confidence: 1.00 — PROCEED.
NPZ accumulator: 666 seeds.

---

**— End of Document —**"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global DRY_RUN
    parser = argparse.ArgumentParser(description="Apply S146 documentation updates")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change, no writes")
    args = parser.parse_args()
    DRY_RUN = args.dry_run

    if DRY_RUN:
        print("=== DRY RUN MODE — no files will be modified ===\n")

    results = []

    print("── Patch 1: CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md ──")
    results.append(apply_append("CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md",
                                "CHAPTER_9 PWC S146", CH9_SECTION))

    print("── Patch 2: CHAPTER_1_WINDOW_OPTIMIZER.md ──")
    results.append(apply_append("CHAPTER_1_WINDOW_OPTIMIZER.md",
                                "CHAPTER_1 PWC S146 kernel invariants", CH1_SECTION))

    print("── Patch 3: CHAPTER_2_BIDIRECTIONAL_SIEVE.md ──")
    results.append(apply_append("CHAPTER_2_BIDIRECTIONAL_SIEVE.md",
                                "CHAPTER_2 PWC S146 sieve path", CH2_SECTION))

    print("── Patch 4: CHAPTER_12_WATCHER_AGENT.md ──")
    results.append(apply_append("CHAPTER_12_WATCHER_AGENT.md",
                                "CHAPTER_12 PWC S146 Step1 path", CH12_SECTION))

    print("── Patch 5: COMPLETE_OPERATING_GUIDE_v2_0.md ──")
    results.append(apply_replace("COMPLETE_OPERATING_GUIDE_v2_0.md",
                                 "COMPLETE_OPERATING_GUIDE PWC S146",
                                 COG_OLD, COG_NEW))

    ok = sum(1 for r in results if r)
    skip = sum(1 for r in results if not r)
    print(f"\n{'DRY RUN ' if DRY_RUN else ''}Summary: {ok} applied, {skip} skipped")

    if not DRY_RUN and ok > 0:
        print("\nNext step — commit and dual-push:")
        print("  cd ~/distributed_prng_analysis")
        print("  git add docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md \\")
        print("          docs/CHAPTER_1_WINDOW_OPTIMIZER.md \\")
        print("          docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md \\")
        print("          docs/CHAPTER_12_WATCHER_AGENT.md \\")
        print("          docs/COMPLETE_OPERATING_GUIDE_v2_0.md")
        print("  git commit -m 'docs(S147): apply S146 PWC invariants to 5 chapter files'")
        print("  git push origin main && git push public main")

    sys.exit(0 if skip == 0 else 1)


if __name__ == "__main__":
    main()
