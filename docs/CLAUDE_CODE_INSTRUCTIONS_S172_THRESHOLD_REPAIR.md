# CLAUDE_CODE_INSTRUCTIONS_S172_THRESHOLD_REPAIR.md — REV2

**S172 — optimizer threshold propagation repair (Beta Ruling 24 items 1–2, Priority-0;
plus PWC hybrid quarantine per Beta comparator ruling §5).**

**REV2 supersedes REV1.** Beta has since **retired PWC/ZMQ from certifying authority** —
they are flag-selectable, non-certifying diagnostic paths. Consequently **repair 3 is no
longer a RANGE-MINER Phase 6 blocker**, and this brief's scope shrinks accordingly. What
replaces it is a quarantine requirement: a known-wrong PWC hybrid route must not remain
silently runnable.

**Base:** current `main` (audit at `50eb69b`; comparator ruling at `8e8e6f3`). Claude Code
on VM 101 as `michael`, venv `~/venvs/torch`. Implement and iterate; do **NOT** commit,
push, or run WATCHER. STOP at the gate for Team Alpha review.

**P0 operational note — for Michael, not Claude Code:** no new Optuna runs and no
variable-skip certification runs until this lands. **Preserve existing Optuna study
databases** — they are regression evidence (Beta P1). Do not delete or overwrite them.

---

## 0. The defect — one hop, one file, two places

From `docs/THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md`:

```
Optuna suggest ─► WindowConfig ─► objective() ─►╳ test_config ─► run_bidirectional_test ─► backend ─► kernel
  (0.73/0.31)     (0.73/0.31)     (0.73/0.31)   ▲   (0.30/0.30)     (0.30/0.30)            (0.30)     float32(0.30)
                                                │
        THE DROP: window_optimizer.py:481-482 passes 3 positional args;
        window_optimizer_integration_final.py:2264-2265 defaults ft/rt to bounds.default_*;
        config.forward_threshold is never read.   (Route B: explicit override at :1798-1799.)
```

**Everything downstream of `run_bidirectional_test` is faithful on all backends.** The drop
is **above the backend split**, so it affects the miner too — this is an *optimizer* repair,
not a PWC repair.

**Regression history:** `3fdf434` (2026-04-30) fixed Route A. `2389b61` (2026-07-07)
silently reverted it while doing unrelated PRNG-encoding work — a **stale-copy overwrite**;
its commit message never mentions thresholds. `git log -S "getattr(config,
'forward_threshold'"` returns exactly those two commits, added then removed. `2389b61` also
reverted `run_bidirectional_test` defaults `0.50 → 0.01` (live at
`window_optimizer_integration_final.py:1074-1075`).

`s172_threshold_patch.py` is still in the tree and its FIX 2 anchor still matches live text
— **do not simply re-run it.** It never covered Route B.

**Provenance caveat (Beta):** the May 11 study is **INDETERMINATE**, not proven poisoned —
it falls between the April fix and the July revert and has no run-bound commit provenance.
Do not characterise historical studies as invalid without evidence; classify by interval.

---

## 1. Repair 1 — Route A (single-process, default) — **MANDATORY**

`test_config` (`window_optimizer_integration_final.py:2264-2265`) must read the sampled
values off the config instead of defaulting to `bounds.default_*`. Restore the `3fdf434`
shape: `ft=None` / `rt=None` in the signature with a `getattr(config, 'forward_threshold',
…)` / `reverse_threshold` fallback resolved **at call time**.

`window_optimizer.py:481-482` passes three positional args — determine whether the fix
belongs at the caller, the callee, or both, and **state why**.

## 2. Repair 2 — Route B (`--n-parallel > 1`) — **MANDATORY**

Never covered by `3fdf434`: `:1835-1841` builds a `WindowConfig` with the sampled values,
then `:1798-1799` **explicitly overrides them with the defaults**. Remove that override.
Confirm no other call site does the same.

## 3. PWC hybrid — implement **or** quarantine (Beta §5)

Beta: *"optional must not mean that a known-wrong PWC hybrid route remains silently
runnable."* One of these must hold before PWC hybrid execution is permitted:

**Option A — implement.** Remove PWC's implicit `phase2_threshold=0.5`
(`persistent_worker_coordinator.py:1119`; the two hybrid call sites at `:1699-1710` and
`:1726-1739` never pass it) and propagate the directional threshold, with a bounded
requested/payload/effective gate proving it reached the kernel.

**Option B — quarantine.** PWC hybrid mode **fails closed** with a clear
`PWC_HYBRID_THRESHOLD_CONTRACT_UNCERTIFIED` error.

**Alpha's recommendation: Option B.** PWC is no longer a certifying path; implementing a
provenance gate for a diagnostic route is work that buys nothing Phase 6 needs. Quarantine
is smaller, safer, and makes the defect loud rather than latent. **If you find Option A is
genuinely trivial — a one-line propagation with no new gate surface — say so and propose
it; otherwise implement B.**

PWC **constant-skip** diagnostic comparisons may still run, non-certifying, with exact
input/config provenance.

**Do NOT change the miner to imitate PWC's `0.50`** (Beta, twice explicit). The miner pins
both keys equal with fail-closed enforcement and is correct.

**Also report** (fix only if small and safe, else defer): `coordinator.py:744` independently
sets `'hybrid': '_hybrid' in job.prng_type` → `True`, so the legacy route selects the hybrid
*kernel* while its *threshold key* is `None` — two hybrid signals from different authorities.

---

## 4. Gates — `tests/test_s172_threshold_propagation.py`

Use **distinctive asymmetric values `forward=0.73 / reverse=0.31`** — the exact pair the
audit found stranded in `optuna_studies/window_opt_1778552567.db`. **Never `0.30`**, which
is indistinguishable from the defect.

| gate | asserts |
|---|---|
| G-ROUTE-A | `n_parallel=1`: sampled `0.73/0.31` reach `run_bidirectional_test` |
| G-ROUTE-B | `n_parallel>1`: sampled values survive; the `:1798-1799` override is gone |
| G-KERNEL | the value reaching the kernel is `0.73`/`0.31`, **read at the executor**, not inferred from config |
| G-MINER-UNCHANGED | the miner's D6 path is behaviourally unchanged; its provenance enforcement and fail-closed gates still hold |
| G-PWC-HYBRID | Option A: effective == directional threshold. Option B: PWC hybrid raises `PWC_HYBRID_THRESHOLD_CONTRACT_UNCERTIFIED` |

**Mutants** (four-part kill rule, VIR-2 — execution proof, clean control, fault-injection
control, detector independence):
1. restore the `bounds.default_*` default in `test_config` → G-ROUTE-A reds
2. restore the Route B override → G-ROUTE-B reds
3. (Option A) restore PWC's `0.5` → G-PWC-HYBRID reds · (Option B) remove the quarantine
   guard → G-PWC-HYBRID reds

**Verification-integrity controls (VIR-1…6):**
- **execution proof** — effective threshold read **at the executor**, not recomputed from
  config (the D6 provenance pattern)
- **clean control** — a correctly-propagated run passes
- **fault-injection control** — mutants 1–3
- **completion sentinel** — explicit `PASS | FAIL | UNAVAILABLE | INCOMPLETE`
- **unavailable-observer behavior** — a backend that cannot be exercised is `UNAVAILABLE`,
  not clean
- **audit claim scope** — repo-scoped unless deployed copies are checked; at audit time
  rigs `.156` and `.164` had no `~/distributed_prng_analysis`

---

## 5. Scope — do NOT touch

- **Hybrid skip bounds** (`expected_skip = 5` hardcoded; no `skip_min`/`skip_max` in the
  hybrid kernel signatures). A **kernel signature change** — deferred, see §6.
- Study↔commit provenance binding; the replacement-resistant regression gate — §6.
- The miner's D6 threshold/provenance/residue work; PWC/ZMQ ingress; the D3.25 contract;
  `TestResult` shape; D5's artifact contract; the dataset schema work.
- **Do not delete or overwrite existing Optuna study databases.**
- Do not build Phase 6's known-answer fixtures here — that is Beta's Wall C, its own item.

## 6. Deferred — name these in the report, do not build them

- **Hybrid skip-bound dead dimension** — hybrid kernels ignore sampled `skip_min`/`skip_max`
  and start from hardcoded `expected_skip = 5`. Either pass them in or remove them from
  hybrid optimisation and provenance. **Historical variable-skip trials are independently
  suspect for this reason**, separately from the threshold defect. Beta lists this at
  revised-Phase-6 step 2.
- **Study↔commit provenance binding** — bind each Optuna study to repository commit, tree
  cleanliness, dataset identity and execution route. *This is the gap that made the May 11
  study indeterminate rather than decidable.*
- **Replacement-resistant regression gate** — a test that survives **whole-block
  replacement**. Text-anchor validation alone is insufficient: `2389b61` reverted the fix by
  overwriting the file from a stale copy and nothing detected it for over two months.

## 7. Non-regression

Capture green before any edit and again after: D1.1 · D1.0 · D0 · D2 · D3.0 · D3 · D3.25 ·
D3.5 · D4 · D5 · D6 3.A · D6-threshold · D6.1 · Phase 3 · Phase 4. **D6-threshold 17/17 must
stay green** — this pass must not disturb the miner's already-correct path.

## 8. Report

Per hop: what was dropped, what now propagates, `file:line`. Whether the Route A fix landed
at caller, callee, or both, and why. **Which PWC hybrid option you took and why.** The gate
matrix and mutant results. Confirmation the miner is behaviourally unchanged. The
`coordinator.py:744` disposition. The three deferred items named. Then STOP.
