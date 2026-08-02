# SESSION_CHANGELOG_20260801_S184 — BOUNDED PHASE 6

**Host:** VM 101 (`zeus-ubuntu`, 192.168.3.177), as `michael`, venv `~/venvs/torch`.
**Base commit:** `76e8eaf` (unchanged — nothing committed or pushed from the sandbox).
**Authority:** `docs/CLAUDE_CODE_INSTRUCTIONS_BOUNDED_PHASE_6.md` (REV1).
**Submission:** `docs/TEAM_ALPHA_BOUNDED_PHASE_6_SUBMISSION.md`.

---

## Outcome

| Deliverable | Sentinel |
|---|---|
| Wall A — interface and consumer (§1) | **PASS** — 7/7 legs, 5/5 fault injections rejected |
| Wall B — determinism and platform (§2) | **PASS** — 4 fresh comparisons IDENTICAL, 1 leg CITED |
| §3 Miner Known-Answer Transfer Gate | **PASS** — 8/8 populations exact-set equal, 8/8 faults rejected, 5/5 worker-path controls |
| §4 RandomSampler control arm | **PASS (NON-CERTIFYING)** — neutral entrypoint + 10 arms, dead-dimension caveat stands |
| §6 two ordered corrections | **DONE** |
| §8 non-regression | **PASS — 22/22 suites exit 0** |

Certified generation reproduced in every Wall-B arm:
`artifact_sha256 0e0092feeb02e22d28557ddf4d8e421941d6117bcc0448d7f7323ec402c1c4b0`, 319 rows,
forward 398,156 / reverse 383 / bidirectional 319 — the same digest as the authoritative D6
generation and both Phase 6.0 runs, reproduced today at `76e8eaf` with the current post-P0.5
worker, on CUDA and on ROCm, across three physical machines.

---

## Files added

```
tests/phase6/known_answer_reference.py     450 lines  independent stdlib reference (json/hashlib/struct only)
tests/phase6/known_answer_gate.py          932 lines  §3 transfer gate
tests/phase6/wall_ab_gate.py               999 lines  §1 + §2
tests/phase6/sampler_control_arm.py        511 lines  §4
docs/phase6_evidence/known_answer_gate.json
docs/phase6_evidence/wall_ab.json
docs/phase6_evidence/sampler_control_arm.json
docs/TEAM_ALPHA_BOUNDED_PHASE_6_SUBMISSION.md
docs/SESSION_CHANGELOG_20260801_S184_BOUNDED_PHASE_6.md   (this file)
```

## Files modified

| File | Change |
|---|---|
| `window_optimizer_bayesian.py` | §4: TPE-sampler construction moved OUT of the study body; the body becomes `run_optimization(..., sampler, sampler_metadata)` with both new arguments **required and keyword-only**. `search()` stays the thin TPE entrypoint (signature and behaviour unchanged, `multivariate=True` preserved). New `OptunaRandomSearch` = the operator-selected RandomSampler arm. New `describe_sampler()` and `SAMPLER_ENTRYPOINTS`. The result dict's `strategy` key now reports the sampler that actually chose the points instead of a hardcoded `'optuna_bayesian'`. No search-space, objective, warm-start or threshold change. **No autonomous sampler selection** — reserved authority, not built. |
| `tests/test_s172_phase4_coordinator.py` | Gate 22 `allowed` set **appended** (nothing rewritten) with the S184 registration block: `window_optimizer_bayesian.py` plus the four new `tests/phase6/` paths, each with rationale. **No gate added, so the 63/63 tally is unchanged.** |
| `reverse_kernel_test_results.txt` | §6: marked **SUPERSEDED — NOT VALIDATION EVIDENCE** in place. All 20 original rows preserved verbatim. Header records: all results `BOTH ZERO`, no positive control, **prohibited from citation under VIR-2**, and names the replacement gate. |
| `quick_test_all_22.sh` | §6 durability: it **generated and truncated** the superseded file, which would have undone Beta's order on the next run. Output now goes to a timestamped file; the superseded record is never touched. Flagged in the submission as an Alpha judgment call. |
| `test_ALL_46_prngs_10M.sh` | §6: header `46 → 44`, both `(12) → (11)` category comments, `$SUCCESS/46` → `/44`. Header block records the verification and states the script is a liveness sweep, **not** known-answer evidence. Filename deliberately unchanged. |
| `docs/KNOWN_ANSWER_VALIDATION_INVENTORY.md` | §6: three `[S184 CORRECTION]` notes striking the false "names 46 variants" / "would hard-fail on two names that no longer resolve" claims. |
| `docs/TEAM_ALPHA_WALL_C_SUBMISSION.md` | §6: one `[S184 CORRECTION]` note striking the same false claim. |

**`miner/`, `sieve_gpu_worker.py`, `prng_registry.py` and `persistent/pwc_protocol.py` are
BYTE-UNCHANGED**, so `G-MINER-UNCHANGED` needed no new registration and P0.5's strengthening of it
is intact.

---

## Verified facts established this session

* **`test_ALL_46_prngs_10M.sh` contains 44 valid registry names**, 11 in each of four categories,
  covering `KERNEL_REGISTRY` exactly (set difference empty both ways). Beta was right; Alpha's
  earlier "two invalid names / would hard-fail" claim was **false** and is struck in two documents.
* **`process_sharded` is not broken in production.** Its spawn children re-import `__main__`, and
  `assembly_shard_worker`'s §6.7.A guard refuses a worker holding a GPU context. Measured:
  `window_optimizer_integration_final` imports cupy at module level (**True**), but
  `window_optimizer` — the real Step-1 `__main__` — does **not** (imports WOI lazily), so a
  production spawn child stays cupy-free. The first Wall-B run failed here; the defect was in the
  gate, and the gate is now cupy-free at import time. **Alpha recommends (has not implemented) a
  gate asserting `cupy not in sys.modules` after importing `window_optimizer` as `__main__` —
  currently an implicit, untested precondition of a whole backend.**
* **Rig source was stale.** `rrig6600` carried the repo at `8e2f5bf` (pre-P0.5 worker);
  `rrig6600b`/`rrig6600c` carried **no source at all**, only the frozen dataset. The brief's
  dataset claim was correct — all three hold `513648160d35…68f6`, verified on target — but the
  *code* had never been deployed. Current source is now deployed to all three at
  `/home/michael/distributed_prng_analysis` and verified by per-file sha256 **on target**.
* **`load_residue_window` requires `draw` even when `full_state` is present** —
  `entry.get("full_state", entry["draw"])` evaluates `entry["draw"]` eagerly. Found by running it.
* The **§3 gate's own first version passed vacuously**: two reverse populations compared
  empty-against-empty because the planted seed sat outside the bounded range. Fixed by
  `_assert_plant_in_scope`, which makes that arrangement INCOMPLETE rather than PASS.

---

## Open items

1. **Stray repository extraction in `$HOME` on all three rigs.** Alpha's first deploy omitted a
   `cd` and left ~837 stray top-level entries per rig. The correct deployment was done separately
   and verified, so **no evidence in this session depended on the stray copy**. The cleanup command
   was denied by the sandbox; the exact command for Michael is in the submission §10. Caveat: the
   extraction overwrote any same-named `$HOME` file, including top-level dotfiles the repository
   carries (`.tmux.conf`, `.gitignore`, `.hash_local.txt`, `.hash_remote.txt`, `.recovery`).
2. **§4 sequencing** — Alpha recommends Beta sequence the *certifying* sampler comparison after
   the skip-output work, because `skip_min`/`skip_max` remain dead on the hybrid path.
3. **The implicit `process_sharded` precondition** — see above; recommended, not implemented.
4. **`quick_test_all_22.sh` output-path change** — Alpha judgment call, submitted for confirmation.

---

## Fallback parity

`fallback parity: code=[not checked this session], env=[not checked this session]` — the
`.127` bare-metal fallback was not booted (Zeus runs one OS at a time and VM 101 was up
throughout), so the two-pass review in CLAUDE.md §5 could not be performed. Declared rather than
assumed.

---

**Not committed, not pushed. WATCHER not run. The pipeline not launched. STOP at the gate.**
