# S172 Phase 5 D6 — RELEASE-GRADE CERTIFICATION RECORD

**Date:** 2026-07-29 (run 17:20–17:21 local)
**Status:** Beta Step 3 complete — release-grade certified generation produced from
the clean real repository. Raw evidence:
`docs/D6_RELEASE_GRADE_SMOKE_20260729.log`; artifact tree preserved at
`~/d6_release_grade_20260729/` (copied out of `/tmp/d6_zeus_smoke`, which does not
survive reboot).

---

## 1. Repository provenance — release-grade, no snapshot

```
repository_mode              : release-grade  (REAL repository, no snapshot taken)
repo_root                   : /home/michael/distributed_prng_analysis
repository_commit           : b08c2c5a5c51c6abb57b272709c799b073f2cbb9
tracked_tree_clean (GOVERNS): True    [git status --porcelain --untracked-files=no]
clean_including_untracked   : False   [WOI._repository_state — information only]
TRACKED-DIRTY paths         : 0  (none — these would BLOCK the run)
UNTRACKED paths             : 9  (permitted, recorded, waived per tracked-clean-only policy)
```

Untracked paths recorded in the evidence (none affect the certified source):
`CLAUDE_CODE_BRIEF_S176_FOLLOWUP_v1.md`, `CLAUDE_CODE_BRIEF_S177_RESUBMISSION_v1.md`,
`CLAUDE_CODE_BRIEF_S178_FOLLOWUP_v1.md`, `CLAUDE_CODE_BRIEF_WATCHER_KPI_VALIDATION_v1.md`,
`docs/STEP2_BIDIRECTIONAL_SIEVE_DESCRIPTIVE_TRACE.md`, `tmp/d0_rev3.diff`,
`tmp/d0_rev4.diff`, `tmp/d2_evidence/nr_baseline_pre_edit.txt`,
`tmp/d2_evidence/writer_pre_mutation_sha.txt`.

Machine-readable evidence sidecar:
`release_grade_repository_state.json` (commit, cleanliness policy,
tracked_tree_clean, full untracked list, generation id/dir/sha256s).

**This generation is certified against the project's own commit** — not a
throwaway snapshot. The prior 2026-07-28 scratch-repo run remains valid *path*
evidence only.

## 2. Commit lineage certified

```
fc37fb5  D6 A — production adapter, Step-1 miner ingress, approved backend seam, adapter gates 9/9
2be51d5  D6 B — directional threshold propagation, effective-threshold provenance + parent-side
                fail-closed enforcement, shared residue authority, threshold gates 17/17 (11 mutants)
1f7f99a  docs — changelogs, correction brief, autonomy signposts, Alpha/Beta review record
b08c2c5  D6   — --release-grade smoke mode certifying against the real repository commit  ← CERTIFIED AT
```

## 3. Threshold provenance — requested == payload == effective

```
requested : forward=0.31  reverse=0.47
payload   : {'1': [0.31], '2': [0.47]}
effective : {'1': [0.31], '2': [0.47]}
phase->dir: {'1': 'forward', '2': 'reverse'}
```

All three legs agree for both directions, and the **effective leg is read back off
the real CUDA executor**, not recomputed from configuration. Phase→direction
mapping confirms no swap. This is the evidence `WindowConfig` alone could not
provide (Beta §2/§3), and the four provenance conditions are enforced parent-side
fail-closed before `commit_trial`, assembly, ingress, accumulator mutation, and
`finalize_run`.

## 4. Survivor counts by direction — physical evidence at the GPU

```
forward       (phase 1, threshold 0.31) : 398,156
reverse       (phase 2, threshold 0.47) :     383
bidirectional (intersection)            :     319
```

Pre-fix comparison (both directions silently at the hardcoded 0.25):
`forward 398,156 / reverse 398,226` — near-identical. The three-order-of-magnitude
divergence is only producible if the two thresholds reach the kernel
independently. 319 of 383 reverse survivors (83%) also cleared forward; reverse at
0.47 performs nearly all the selection (0.005% of 8M) while forward at 0.31 passes
~5%.

## 5. Certified artifact

```
generation_id     : gen-20260730T002104136270Z-step1_java_lcg_0
generation_dir    : .../generations/gen-20260730T002104136270Z-step1_java_lcg_0--99ba444b…f3d8
artifact_sha256   : 0e0092feeb02e22d28557ddf4d8e421941d6117bcc0448d7f7323ec402c1c4b0
sidecar_sha256    : 99ba444b22e19d2dbb64cdb92d287b4359e30e21627da94bd7f54fbba8b4f3d8
raw_candidates    : 319   l2_winners: 319   prior_rows: 0   final_rows: 319
paths             : all 4 certified paths exist on disk
```

- **22-array validation:** 22 arrays, order matches the frozen oracle,
  `validate_array_bundle()` **passed**, rows=319.
- **Sidecar:** 32 keys, schema `s172.d3_5.provenance.v1.1`, encoding
  `s172.phase0.encoding.v1`; carries `repository_commit: b08c2c5…` and
  `repository_tree_clean: True` **inside the artifact's own provenance**;
  `canonical_map_hash: 55686c64…a191`.
- **Step-2 loader:** `utils.survivor_loader.load_survivors format=npz
  npz_version=3 count=319 **fallback_used=False**`.

The generation directory is chain-addressed by its own sidecar hash.
`prior_rows: 0` — this is generation one; no prior accumulator to chain from.

## 6. Real-silicon evidence

Trial completed in **23.2 s** for 8,000,000 seeds (stripe 4,000,000, substripe cap
1,000,000, window 3), real GPU worker `miner/range_miner_worker.py` (pid 32719,
cupy on device 0), coordinator on `127.0.0.1:34263`.

`nvidia-smi` bracketing: idle 22 W / 264 MiB before → **97 W / 539 MiB with the
worker process resident** during the CUDA sieve → 264 MiB after, clean teardown,
no leaked GPU memory. Driver 550.163.01, CUDA 12.4, RTX 3080 Ti 12288 MiB.

## 7. Known defect observed, as expected (D6.1)

```
[S152-FLUSH] Warning: incremental flush failed (non-fatal):
[Errno 2] No such file or directory:
'bidirectional_survivors_all.npz.flush.tmp' -> 'bidirectional_survivors_all.npz'
```

This is the tracked `_flush_npz_incremental` defect, firing exactly as diagnosed
(temp filename lacks `.npz`, NumPy appends it, `os.replace` targets the wrong
name). **Benign for this run:** the failure precedes the candidate-list clear, so
all 319 candidates reached the finalizer and the generation is complete and
correct. Recorded here so its appearance in the release-grade log is not read as a
new surprise. It remains **`D6.1 — incremental NPZ atomic flush and durability
repair`**: non-blocking for the D6 commit, **blocking for extended Phase 6
benchmark runs, Phase 7 multi-trial soak, and WATCHER-controlled long-running
execution.**

## 8. Clean-tree verification (Beta Step 2, run before this smoke)

On the committed tree, in the `~/venvs/torch` venv:

```
D1.1                         : green (18/18)
D4                           : green (8/8)
D5                           : green (24/24, 18 mutants; internal NR chain covers
                               D2 7/7, D3 10/10, D3.0 10/10, D3.25 13/13, D3.5 60/60)
D6 production adapter (3.A)  : green (9/9, 16 mutants)
D6 threshold/provenance gate : green (17/17, 11 mutants)
```

## 9. Disposition

Beta Step 3 is complete. The release-grade certified generation exists, is bound to
commit `b08c2c5`, validates as a 22-array bundle, and is read back by the Step-2
loader without fallback — with the Optuna-tunable per-direction thresholds proven
to have reached the CUDA kernel unchanged.

Remaining per Beta §10: commit this certification record (Step 4), push `origin`
and `public`, then open **`D6.1 — incremental NPZ atomic flush and durability
repair`** before any extended Phase 6/7 or WATCHER soak execution.
