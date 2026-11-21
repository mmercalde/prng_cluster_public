# SCOPE — RANGE-MINER: deliver the trial's skip range to phases 1-2 (F-1)

**Date:** 2026-11-20 · **Status:** scope only, read-only. **No miner or pinned file was edited.** This is a separate
future change, not part of Stage 3a. Read at main `b2efdef`, worktree `/home/michael/tfm_worktrees/pwc-offset`.

**Owner, 2026-11-20 21:32-21:35 PST, verbatim:**
- "Well range miner must be fixed none the less"
- "Please do whatever method is the quickest and least recoding"

**Yardstick.** The fix is judged by the prediction pools' hit rate against `k/1000` on unseen draws, not by whether any
seed or "the true stream" survives (CLAUDE.md). It changes what phases 1-2 sieve, so its effect is a hit-rate question
(Beta, `docs/exchange/from_beta/X-VOLUME-LIMIT_R1_BETA_REVIEW.md` Q1; settling measurement below).

---

## Prior record

**Searches run** (`/bin/grep`, worktree at `b2efdef`):

```
grep -n "skip_range|skip_min|skip_max" miner/range_miner_coordinator.py miner/range_miner_worker.py
grep -n "def build_stripe_assign_payload|def workflow_phase_semantics|def workflow_stages_for|_dispatch_pending(" miner/range_miner_coordinator.py
git log --oneline -S'"skip_range"' -- miner/range_miner_coordinator.py      # no commit
git log --oneline -S'skip_range' -- miner/range_miner_worker.py             # dbe3d0e (Phase 3) only
grep -ln "build_stripe_assign_payload" tests/*.py ; grep -n "_dispatch_pending" tests/test_s172_attempt6_remediation.py
grep -n "skip" utils/survivor_volume_estimator.py ; sed -n 172,186p scripts/gate12_parity_gate.py
/bin/grep -n "M6a|2.4" docs/HYBRID_SKIP_BOUND_AUDIT.md ; sed -n 370,395p docs/TEAM_ALPHA_BOUNDED_PHASE_6_SUBMISSION.md
/bin/grep -n "F-1" docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md docs/TFM_SESSION_HANDOFF_20261114.md
```

**Found, and read:**

| Document | What it says | Label |
|---|---|---|
| `docs/TFM_SESSION_HANDOFF_20261114.md:76-80`, item 7 | "The skip range never reaches phases 1 and 2 on the RANGE-MINER route … Skip is a cornerstone of the design. This is the owner's decision, after Run 4" | ALREADY RECORDED (F-1). The owner decided 2026-11-20 21:32-21:35: fix it |
| `docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md:92`, `:99`, `:360-369` | F-1 found by reading. `build_stripe_assign_payload` has no `skip_range` key and `git log -S` finds none. The two documents below are "Contradicted at source". What would settle the hit-rate question: "a flag-selectable delivered-skip configuration, compared with the baseline on unseen-draw pool hit rate" | ALREADY RECORDED |
| `docs/HYBRID_SKIP_BOUND_AUDIT.md` §2.4 (`:104-120`), rows M1-M6a | The sampled bounds are "carried" through trial entry (M1), coordinator (M2), ledger/manifest (M3), worker unpack (M4: `payload.get("skip_range", [0, 16])`) and `BuildContext` (M5), and reach the constant kernel (M6a: `_constant_prefix` → `ScalarArg(ctx.skip_min/skip_max, "int32")`) | ALREADY RECORDED, **wrong at M3→M4**: the payload never carried them. **What it says the system should do: the trial's sampled `skip_min`/`skip_max` reach the constant kernels' `for (skip = skip_min; skip <= skip_max; skip++)` loop.** M6b: the hybrid ABI has no skip bound, by design |
| `docs/TEAM_ALPHA_BOUNDED_PHASE_6_SUBMISSION.md:382` | "effective skip semantics: constant only. Sampled `skip_min`/`skip_max` are placed in the payload's `skip_range` and reach `for (int skip = skip_min; skip <= skip_max; skip++)`" | ALREADY RECORDED. It describes a harness that drove `SieveExecutor` with its own payloads, not the coordinator's builder. **It says the same as the audit: sampled bounds into the constant kernels through `skip_range`** |
| `docs/exchange/from_beta/X-VOLUME-LIMIT_R1_BETA_REVIEW.md` Q1 | F-1 treated as an already-disclosed status; NO OBJECTION to the gate | ALREADY RECORDED |
| CLAUDE.md, "The yardstick" step 4 | "Propose any alternative as a flag-selectable configuration. The current algorithm stays as the baseline." | Standing rule. See "Design choice for the owner" below |

**Not searched:** rigs (off), deployed rig bytes, ser8, the private repository, PDFs.

---

## 1. Exactly where the skip range is dropped (RANGE-MINER route, `b2efdef`)

| # | file:line | What happens | Skip range |
|---|---|---|---|
| 1 | `window_optimizer_integration_final.py:1904-1905` | `run_trial_miner(..., skip_min=config.skip_min, skip_max=config.skip_max, …)` | carried |
| 2 | `miner/range_miner_coordinator.py:12399-12400`, `:12568-12569` | `run_trial_miner` parameters into the serve context | carried |
| 3 | `miner/range_miner_coordinator.py:9552` (`build_trial_context_from_serve`), `:2734-2735` | the immutable trial context carries `int(skip_min)`, `int(skip_max)` | carried |
| 4 | `miner/range_miner_coordinator.py:1391-1392` (ledger schema), `:1872-1878` (`set_trial_context`); `get_trial_context` `:1898-1907` | persisted to and read back from the ledger | carried (record only) |
| 5 | `miner/range_miner_coordinator.py:10498-10504` (`serve_trial` loop) | calls `self._dispatch_pending(run_id, fam, ph, …, window_anchor, generator_phase, residues, trial_number, trial_ctx["forward_threshold"], trial_ctx["reverse_threshold"])`. **`trial_ctx` holds `skip_min`/`skip_max` here and does not pass them** | **dropped here** |
| 6 | `miner/range_miner_coordinator.py:11994-12036` (`_dispatch_pending`) | the signature has no skip parameter; it calls `build_stripe_assign_payload(...)` at `:12018-12023` | absent |
| 7 | `miner/range_miner_coordinator.py:9216-9367` (`build_stripe_assign_payload`), dict at `:9331-9358` | the payload holds `dataset`, `dataset_sha256`, `window_size`, `sessions`, `window_anchor`, `generator_phase`, `residue_sha256`, `min_match_threshold`, `phase2_threshold`. **No `skip_range`.** The phase is resolved through `workflow_phase_semantics` (`:2744-2767`: 1 forward/constant, 2 reverse/constant, 3 forward/variable, 4 reverse/variable). The stage list is `workflow_stages_for` (`:12046-…`) | absent |
| 8 | `miner/range_miner_worker.py:1293` (`SieveExecutor.execute`) | `skip_min, skip_max = tuple(payload.get("skip_range", [0, 16]))` | **default `[0, 16]` used** |
| 9 | `miner/range_miner_worker.py:1388` → `BuildContext`; `_constant_prefix` `:259-271` (`:270-271` the two `int32` scalars) | `[0, 16]` reaches the constant kernels' loop (`prng_registry.py:421`, `:478`: `for (int skip = skip_min; skip <= skip_max; skip++)`) | `[0, 16]` |

The variable-skip kernels (phases 3-4) take no skip bounds (`_hybrid_prefix`; audit M6b). They are not affected by any
change here.

**Also through the same builder:** `utils/survivor_volume_estimator.py:510-515` builds its sampling payloads with
`build_stripe_assign_payload`. Its own note (`:661-664`) records that the sampled skip "do[es] not reach the sampled
kernels — exactly as in a run".

---

## 2. The smallest change that carries it into the phase 1-2 payload

**One pinned file, three small edits, and no worker change** (the worker already reads `payload["skip_range"]`, `:1293`):

1. **`build_stripe_assign_payload`** (`range_miner_coordinator.py:9216`) gains a **keyword-only, required**
   `skip_range` argument with no default, in the same style as `phase`/`generator_phase`.
   - It validates two non-negative plain ints with `min ≤ max`, and refuses otherwise with `MinerMetadataError`.
   - When `_skip_mode == "constant"` (phases 1-2, from the `workflow_phase_semantics` call already at the top of the
     function), it adds `"skip_range": [int(skip_min), int(skip_max)]` to the dict.
   - For phases 3-4 the payload stays **byte-identical**. The hybrid kernels have no skip bound, so this keeps the
     variable-skip payload and its provenance records unchanged.

   About 12 lines.
2. **`_dispatch_pending`** (`:11994`): **no signature change.** The skip range comes from the persisted trial context
   for `run_id` (`self.ledger.get_trial_context(run_id)`, the same source the resume path uses at `:4316`), read once
   per run and cached. It then passes `skip_range=(ctx["skip_min"], ctx["skip_max"])` to the builder at `:12018`. A
   missing context or missing fields refuses with `MinerMetadataError`, fail-closed, never `[0, 16]`. About 8 lines.
   - **Why not a new positional parameter:** `tests/test_s172_attempt6_remediation.py:3447-3475` runs pinned
     `serve_trial` source that calls `_dispatch_pending` with 14 positional arguments, through an instance-scoped
     bridge. Brief I's added `generator_phase` already forced that bridge once. Keeping the arity unchanged leaves it
     working.
   - The alternative is passing `trial_ctx["skip_min"]`/`["skip_max"]` at `:10498` through a new parameter. That is
     fewer lines in the coordinator, but it means editing the pinned-source bridge in a miner test.
3. **`utils/survivor_volume_estimator.py:510-515`**, not pinned: pass `skip_range=(g["skip_min"], g["skip_max"])`
   and correct the note at `:661-664`. Without this the estimator raises, because the argument is required. That is
   the intended effect of making it required: no caller can silently keep `[0, 16]`.

**Not changed:**
- `miner/range_miner_worker.py`, so its gate12 digest is unchanged;
- the kernels and `prng_registry.py`;
- the NPZ contract: `skip_min`/`skip_max`/`skip_range` are already recorded from the trial context, so for constant
  rows the stamp now equals what was sieved;
- PWC, which already sends the trial's range (S3-13 withdrawn).

**Optional, a second step and not the smallest:** make the worker refuse a constant-phase payload without
`skip_range`, instead of defaulting to `[0, 16]` at `:1293`. That edits a second pinned file and needs a rig worker
redeploy.

### Design choice for the owner (not settled by "quickest")

- **(A) Direct fix:** always deliver the sampled range. This is the smallest change, as above, and it is what the
  audit, the Phase-6 submission and the owner's words describe.
- **(B) Flag-selectable:** a trial-level `deliver_skip_range` switch, with the current `[0, 16]` behaviour kept as the
  baseline. This is what CLAUDE.md's yardstick step 4 and the volume report's settling measurement ask for. It adds a
  manifest/CLI field, which is agent 1's files and WATCHER plumbing, so it is more recoding.

The owner's "quickest and least recoding" points to (A). (B) is listed because the standing rule names it. The
hit-rate comparison below can still be run under (A), by pinning a trial to `skip [0, 16]` as the baseline arm.

---

## 3. Pinned files and gate12 digests

| File | In `scripts/gate12_parity_gate.py:174-185`? | Changed by (A)? | Digest |
|---|---|---|---|
| `miner/range_miner_coordinator.py` | **yes** | yes | **changes** |
| `miner/range_miner_worker.py` | yes | no (unless the optional worker step) | unchanged |
| `prng_registry.py`, `sieve_gpu_worker.py`, `miner/range_miner_protocol.py`, `miner/dataset_authority.py`, `miner/__init__.py`, `adaptive_thresholds.py`, `execution_set.py`, `hybrid_strategy.py` | yes | no | unchanged |
| `utils/survivor_volume_estimator.py` | no | yes | — |

The coordinator runs on VM101, but gate12's governed set is the worker's **statically reachable** import closure
(`scripts/gate12_parity_gate.py:43-60`), and it includes `miner/range_miner_coordinator.py`. Gate12 compares each rig's
deployed bytes with the local tree. **After the change, gate12 refuses every launch until all three CT100s hold the
new coordinator file.**

---

## 4. Tests it needs (all miner-side; editing miner tests needs owner approval, with line-limited diffs as in Stage 2)

**New gates:**
- phase 1 and 2 payloads carry `skip_range == [ctx.skip_min, ctx.skip_max]` for several sampled ranges, including
  `(0, 16)`, `(6, 99)` and `(3, 3)`;
- phase 3 and 4 payloads are byte-identical to `b2efdef`'s;
- the builder refuses a missing argument, a non-int, a negative value or `min > max`;
- `_dispatch_pending` refuses when the trial context is missing;
- end to end, in process, with no GPU: the kernel-argument capture (the existing G-CAP / arg-capture harness style)
  shows `ScalarArg(skip_min)`/`ScalarArg(skip_max)` equal to the trial's values on constant stripes;
- the volume estimator samples the delivered range.

**Mutants:**
- the key dropped;
- the bounds swapped;
- the range added to variable phases;
- the context read replaced by `[0, 16]`.

**Existing callers to update** (they call `build_stripe_assign_payload` without the new required argument):
- `tests/test_miner_no_torch.py`
- `tests/test_s172_mp1_drain_attribution.py`
- `tests/test_s172_phase4_coordinator.py`
- `tests/test_s172_phase5_d6_threshold_path.py`
- `tests/test_s172_r1_drain_remedy.py`
- `tests/test_s172_window_anchor_brief_i.py`

Each gets one added keyword argument. Tests that compare payload key sets, or pin coordinator function bodies by AST
against a base commit, must be re-pinned. For example, the `PINNED_SAME` list in `tests/test_s172_admission_stall.py`
does not list these two functions, but other AST pins must be checked [UNVERIFIED for the full set].

**Proof:** `scripts/prove_unchanged_outside.py --base <base> --file miner/range_miner_coordinator.py --allow
RangeMinerCoordinator.build_stripe_assign_payload RangeMinerCoordinator._dispatch_pending`. Add `--keep-block` where a
large function is allowed.

**Full before/after pass:** `tests/test_s172_staging_partb.py` stays excluded.

---

## 5. Rig redeploy and run-level consequences

- **Redeploy:** `scp` of `miner/range_miner_coordinator.py` only, to each CT100's `script_path` (`.122`, `.156`,
  `.164`; `mkdir -p` first, absolute paths), then `scripts/gate12_parity_gate.py` must pass. No worker restart is
  needed for behaviour (the worker is unchanged), but gate12 refuses until the bytes match.
- **Work per seed on phases 1-2 changes with the range width.** The constant kernels loop over every skip in
  `[skip_min, skip_max]`. At Run 4's pinned geometry (skip 6-99, `docs/TODO_CONSOLIDATED.md:79`) that is 94 values
  instead of 17, about **5.5× the phase 1-2 kernel work**, before any survivor-count effect [arithmetic, not measured].
  Stripe sizing, lease timeouts and run duration were all measured at `[0, 16]`.
- **The survivor volume gate was measured at `[0, 16]`.** The cap of 25,000 (owner decision 2026-11-14) and Run 4's
  phase counts must be re-measured with the delivered range before a counted run (`utils/survivor_volume_estimator.py`
  follows automatically once edit 3 lands).
- **`best_skips` is `uint8`** (`miner/range_miner_worker.py:266`, `:1382`). A sampled `skip_max` above 255 would not
  fit. Check the search space's upper bound before the change [UNVERIFIED here].
- **Hit rate:** unknown in direction. **The settling measurement:** the same trials with `skip [0, 16]` (baseline)
  and with the delivered range, scored by prediction-pool hit rate against `k/1000` on unseen draws (the volume
  report's Beta Q1 note).
- **PWC:** already sends the trial's range, so after this change the two engines sieve the same phase 1-2 range again
  (Stage 4's comparison no longer needs to pin `skip [0, 16]`).

## 6. Open questions for the owner

1. **(A) or (B)** (section 2).
2. **Trial context:** read from the ledger by `run_id` (no signature change, the attempt6 bridge untouched), or pass
   `trial_ctx` at `:10498` (fewer lines, but editing the pinned-source bridge)?
3. **The worker:** keep the `[0, 16]` default at `:1293` for legacy payloads, or make it refuse (a second pinned file,
   a rig redeploy)?
4. **Before Run 4 or after:** the change raises phase 1-2 work about 5.5× at Run 4's geometry and invalidates the
   volume-gate measurement.
