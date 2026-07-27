# S172 Phase 5 D5 — Claude Code kickoff (VM101)

You are Claude Code on VM 101 (`michael@192.168.3.177`), working in
`~/distributed_prng_analysis` as user `michael` (NOT root). You implement and
iterate; you do **not** commit, push, or run WATCHER. When gates and
non-regression are green, STOP and report for Team Alpha review.

## Authority

Your complete instructions are `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D5.md`
(REV1). Read it in full before writing anything. Spec authority:
`docs/PROPOSAL_S172_RANGE_MINER_v1_4_5.md` §6.7.A/B/C, §17. This is the binding
Team Beta D5 ruling (option A + sampled RSS-sum, findings F1–F4 locked).

**Frozen against HEAD `3e8580a`.** First command: `git rev-parse HEAD` and
confirm `3e8580a`. If it differs, STOP and report — do not proceed on a moved
base.

## The one sentence that governs every decision

D5 parallelizes only spool-local validation. D1.1 remains the sole authority for
validation semantics and global assembly semantics; workers produce ordered,
lossless validated-spool artifacts, while the parent alone performs deterministic
global merge, duplicate attribution, intersection, enrichment, and final
assembly. If any change moves a global-assembly responsibility into a worker, it
is wrong — STOP.

## Order of work

1. **Capture the baseline green FIRST, before any edit.** D4 8/8
   (`PYTHONPATH=. python3 tests/test_s172_phase5_d4_serial_backend.py`), D3.5
   60/60, D3.25 13/13, D3 10/10, D3.0 10/10, D2 7/7, D1.1 18/18, D1.0 8/8, D0
   12/12, Phase 4 63/63, Phase 3 17/17. Paste the outputs.

2. **Commit 1 — extraction ONLY** (`miner/range_miner_npz_writer.py`). Extract
   the per-spool validator and the inline merge verbatim; make `assemble_trial`
   a thin serial wrapper; add `ValidatedSpoolProjection`. No multiprocessing, no
   codec, no perf change, no incidental cleanup. **Proof: D1.1 18/18 and every
   downstream suite stay green with ZERO test edits.** If a D1.1 test needs
   changing to pass, the extraction changed behavior — STOP, do not touch the
   test.

3. **Commit 2 — `process_sharded` backend** (`miner/assembly_backends.py` +
   `miner/assembly_shard_worker.py`). Workers call the extracted validator;
   parent calls the extracted merge. spawn canonical (forkserver only if a test
   proves no inherited GPU state; never fork). Sampled concurrent-tree RSS-sum
   peak_rss at 25 ms. 1/2/4/6/8-process benchmark. Full D5 gate
   (`tests/test_s172_phase5_d5_process_sharded.py`) including every mutant under
   the four-part kill rule.

## Hard rules

- **Read live source before every claim.** This seam is where the last four Team
  Beta rejections happened; do not reason from memory.
- **Reuse, never reimplement.** No second copy of validation, map construction,
  dedup, intersection, columnization, ordering or publication.
- **Two commits, in order, never folded.**
- `MinerResultManifest` / `SpoolMeta` do **not** exist — the manifest is a plain
  `dict`. Keep `read_and_validate_spool(run_id, manifest: Dict[str, Any])`; do
  not invent a dataclass wrapper.
- A backend produces a `MinerTrialAssembly` and STOPS. Do not import or call
  `finalize_run`. NPZ path fields stay `None`.
- Every gate must FAIL on wrong behavior; oracles hand-transcribed, never
  imported from the module under test.
- No `git commit` / `git push` / `watcher_agent --run-pipeline` (deny-ruled).

## Report

Per commit: diff + status + full command/output evidence + the pre-edit
baseline. Commit 1: explicit confirmation D1.1 18/18 + downstream stayed green,
no test edits. Commit 2: per-mutant red signature with four-part attribution, the
1/2/4/6/8 benchmark table with canonical `peak_rss`, and confirmation no D0–D4
production module or test changed beyond the Commit-1 extraction. Then STOP.
