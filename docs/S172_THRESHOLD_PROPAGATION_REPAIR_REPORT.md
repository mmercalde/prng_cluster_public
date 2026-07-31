# S172 — Optimizer Threshold Propagation Repair — Report

**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_S172_THRESHOLD_REPAIR.md` (REV2) §8.
**Audit:** `docs/THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md`.
**Host:** VM 101 `192.168.3.177` as `michael`, venv `~/venvs/torch`.
**Base commit:** `a0442a0` (working tree clean for every file touched, verified before edit).
**Date:** 2026-07-30. **No commit, no push, no WATCHER, no pipeline launch.**

Status: **PASS** — 5/5 gates green, 3/3 mutants killed, 15/15 non-regression suites
green before and after, D6-threshold 17/17 unchanged.

---

## 1. Per-hop disposition

### Route A — single-process, `--n-parallel 1` (the production default)

| hop | `file:line` | before | after |
|---|---|---|---|
| sample | `window_optimizer_bayesian.py:437-452` | `0.73 / 0.31` on the `WindowConfig` | unchanged |
| objective | `window_optimizer.py:481-482` | passes `config` positionally, no `ft`/`rt` | **unchanged — deliberately** (§2) |
| **the drop** | `window_optimizer_integration_final.py:2348-2352` | `ft=bounds.default_forward_threshold`, `rt=bounds.default_reverse_threshold` bound **at def time**; `config.forward_threshold` never read | `ft=None, rt=None` in the signature |
| resolution | `window_optimizer_integration_final.py:2363-2364` | *(did not exist)* | `ft = resolve_directional_threshold(config, 'forward', ft, bounds.default_forward_threshold)` and the reverse equivalent — resolved at **call** time |
| hand-off | `window_optimizer_integration_final.py:2380-2381` | `forward_threshold=0.30 / reverse_threshold=0.30` | `0.73 / 0.31` |

**What was dropped:** the entire sampled threshold dimension. Every trial filtered at
`SearchBounds.default_forward_threshold` / `default_reverse_threshold` — live value `0.30`,
resolved from `distributed_config.json → search_bounds.*.default`.

**What now propagates:** the value on the config, unmodified, to every backend.

### Route B — `--n-parallel > 1`

| hop | `file:line` | before | after |
|---|---|---|---|
| sample | `window_optimizer_integration_final.py:1903-1908` | `trial.suggest_float(...)` | unchanged |
| config build | `window_optimizer_integration_final.py:1909-1915` | sampled values placed on `cfg` | unchanged |
| **the drop** | `window_optimizer_integration_final.py:1880-1885` (was `:1798-1799`) | `forward_threshold=_local_bounds.default_forward_threshold` — an **explicit overwrite** of values in scope on the adjacent line | `forward_threshold=_resolve_dt(cfg, 'forward', None, _local_bounds.default_forward_threshold)` and the reverse equivalent |
| resolver binding | `window_optimizer_integration_final.py:1780-1785` | *(did not exist)* | `_partition_worker` imports the same module-level resolver — it runs in a separate process and re-imports the module, so without this the call site would `NameError` at runtime only under `--n-parallel > 1` |

Route B was **never covered by `3fdf434`**. Confirmed no other call site does the same:
the only remaining live readers of `default_forward_threshold` / `default_reverse_threshold`
are the `SearchBounds` dataclass/`from_config` definitions themselves
(`window_optimizer.py:128-129`, `:149-150`), the two repaired call sites, and
`window_optimizer_bayesian.py:814-815` (see §7).

### The new single authority

`window_optimizer_integration_final.py:206-243` — `ThresholdResolutionError` +
`resolve_directional_threshold(config, direction, explicit=None, default=None)`.

Precedence: **explicit caller argument > config attribute > supplied default**, and it
**raises** rather than inventing a value when none resolves. Fallback triggers on `is None`
only — `0.0` is a legitimate threshold. This deliberately does **not** reuse the
`getattr(config, 'forward_threshold', None) or bounds.default_forward_threshold` form from
`s172_threshold_patch.py` FIX 2, which silently replaces a `0.0`.

This is the D6 shape (`miner/range_miner_coordinator.py:3410-3419`): resolve once in the
parent, never reinterpret downstream.

---

## 2. Where the Route A fix landed, and why

**Callee only** (`test_config`). Not the caller, not both.

1. **The caller already hands over the authority.** `window_optimizer.py:481-482` passes
   `config`, and the sampled values live on it. Adding a parallel `ft`/`rt` argument there
   would create a **second authority for the same quantity** — the precise failure mode
   §2.7 catalogues four times.
2. **The caller is an interface method, not a call site.**
   `WindowOptimizer.test_configuration` (`window_optimizer.py:444-454`) declares no
   `ft`/`rt` parameters, and its non-integrated fallback body constructs a `TestResult`
   without them. Passing thresholds from the caller would require changing that signature,
   its fallback body, and the `test_configuration_func` forwarding — widening the blast
   radius across a method the integration layer monkey-patches at `:2377`.
3. **The callee is the single choke point.** Every strategy (`random`, `grid`, `bayesian`,
   `evolutionary`) reaches the backend through `objective()` → `test_configuration` → the
   monkey-patched `test_config`. One fix there covers all four.
4. **Bonus:** moving resolution from def-time to call-time also removes a def-time
   dependency on `bounds`, which was only conditionally defined (`:2345`, inside
   `if not _np2_complete:`).

`window_optimizer.py:450`'s docstring — *"Thresholds are now taken from
`config.forward_threshold` and `config.reverse_threshold`"* — was false at HEAD and is now
**true again**. Per the audit's recommendation the code was made true rather than the
docstring softened; no docstring edit was needed. It can come off the
`tfm-project-facts` §3 supersession list.

---

## 3. PWC hybrid — **Option B (quarantine)** was implemented

`persistent_worker_coordinator.py:145-204` — `PWC_HYBRID_QUARANTINE_CODE` (`:176`),
`PwcHybridThresholdContractUncertified` (`:179`), `assert_pwc_hybrid_not_quarantined()`
(`:183`). **One authority, invoked at one place — the execution boundary:**

| call site | `file:line` | effect |
|---|---|---|
| dispatch | `persistent_worker_coordinator.py:1210` | `run_sieve_pass` fails closed on any hybrid `prng_type`, before strategy loading, chunking or any worker dispatch — covering every caller, present or future |

**Placement is a considered decision, and it changed mid-session under test pressure.**
The first implementation also carried a *pre-flight* guard at the top of
`run_trial_persistent` (fail on `test_both_modes=True` before spawning a fleet). The
post-edit non-regression run caught it: **D3.25 G1 went red.** D3.25's `drive_pwc` replaces
`PersistentWorkerCoordinator` with a fake sieve and drives the live `run_trial_persistent`
both-mode to assert the **v2 four-map return shape on every return path** — it never
executes a hybrid pass. A trial-entry guard quarantines that *contract check*, which is not
what Beta's ruling asks to be blocked; quarantining *execution* is. The pre-flight was
removed and the reasoning left in place at `persistent_worker_coordinator.py:1640-1652`, so
the absence reads as a decision rather than an oversight.

**Accepted cost:** a real both-mode PWC trial now fails on the **first hybrid pass** rather
than before pass 1 — two completed constant-skip passes and a spawned fleet are wasted. It
still fails closed, loudly, before any hybrid survivor exists, and `finally: pwc.shutdown()`
(`persistent_worker_coordinator.py:1874-1875`) still releases the workers. G-PWC-HYBRID
asserts this behaviour directly rather than assuming it.

**Why B, not A.** Option A is *not* a one-line propagation, by the brief's own definition:

1. Beta's §5 Option A requires the propagation **plus a bounded requested/payload/effective
   gate proving it reached the kernel** — a new gate surface on a path that no longer
   certifies anything.
2. More decisively, propagating the threshold **would not make the route correct**. The
   hybrid kernels also ignore the trial's sampled `skip_min`/`skip_max` and start from a
   hardcoded `int expected_skip = 5` (`prng_registry.py:1027`, `:805`, `:885`, `:1159`) —
   a second, independent divergence on the same variable-skip axis, fixable only by a
   kernel-signature change (explicitly out of scope, brief §5). Fixing only the threshold
   would leave the route still executing a configuration nobody requested while removing
   the one symptom that makes that visible — converting a loud defect into a quiet one,
   the opposite of what Beta's ruling asks for.
3. PWC is retired from certifying authority; nothing Phase 6 needs depends on hybrid PWC.

**Scope of the quarantine:** variable-skip only. PWC **constant-skip is untouched** and
still runs as a non-certifying diagnostic comparator — it never used `phase2_threshold`.
There is deliberately **no override flag**: an escape hatch would restore exactly the
"silently runnable" property Beta ruled out. The error message names the lift condition.

**The miner was not changed to imitate PWC's `0.50`.** No file under `miner/` was touched
(§5).

---

## 4. Gate matrix

Harness: `tests/test_s172_threshold_propagation.py` (NEW). Values **forward=0.73 /
reverse=0.31** throughout — the exact pair the audit found stranded in
`optuna_studies/window_opt_1778552567.db`. `0.30` appears only as the value a defective
path collapses **to**, never as an expectation.

| gate | result | evidence |
|---|---|---|
| G-ROUTE-A | **PASS** | sampled `0.73/0.31` reached `run_bidirectional_test` intact; an explicit `ft`/`rt` argument still overrides the config (precedence check) |
| G-ROUTE-B | **PASS** | sampled `0.73/0.31` survived; the `_local_bounds.default_*` override is gone; `_partition_worker` really binds the resolver; `_worker_obj` still produces the sampled values (so G-ROUTE-B cannot pass vacuously) |
| G-KERNEL | **PASS** | `forward=0.7300000190734863`, `reverse=0.3100000023841858` — read off the **real cupy `RawKernel` launch arguments**, arg index 10, on the RTX 3080 Ti |
| G-MINER-UNCHANGED | **PASS** | `miner/`, `sieve_gpu_worker.py`, `prng_registry.py`, `persistent/pwc_protocol.py` byte-identical to HEAD per `git status`; D6-threshold harness re-run as a subprocess → **17/17** |
| G-PWC-HYBRID | **PASS** | (1) `run_sieve_pass` raises `PWC_HYBRID_THRESHOLD_CONTRACT_UNCERTIFIED` on a hybrid family; (2) a real both-mode trial dies on the **first hybrid pass**, proven from the sieve call log; (3) the D3.25 v2 return-shape path against a fake sieve still completes, and is asserted to actually reach the variable-skip passes so check (3) cannot be vacuous; (4) constant-skip families pass through |

**Completion sentinel: `PASS`.** (`FAIL` / `INCOMPLETE` / `UNAVAILABLE` are the other
terminal states; an unobservable backend yields `INCOMPLETE`, never a silent pass.)

### How the gates are built — and why it matters

`2389b61` reverted the fix by **replacing the whole block**, so a text-anchor check would
have gone green: the anchor it matched disappeared with the fix. Every gate here therefore
**extracts the live source of the real call site by AST, off disk, at run time, and executes
it** (`test_config`, `_local_test`, `_partition_worker`, `dispatch_chunk`, `run_sieve_pass`,
`run_trial_persistent`). Nothing is a hand-written replica. Each extraction asserts the
target `def` is unique in the file, so ambiguity is fatal rather than silently testing the
wrong one.

G-KERNEL is **chained**: each hop is fed the value *observed* at the previous hop, never a
literal — Optuna sample → live `test_config` → live PWC `dispatch_chunk` job dict
(`min_match_threshold`) → real `sieve_gpu_worker.run_sieve_job` on real silicon → the
`cp.float32` in the kernel launch args. Only the final comparison uses the hand-transcribed
oracle.

### Mutants (four-part kill rule, VIR-2)

| mutant | applies-once | mutated path executed | detector |
|---|---|---|---|
| **M1** restore `bounds.default_*` in `test_config` | 3 anchors, each exactly once | mutated `test_config` ran; `run_bidirectional_test` got `0.3/0.3` | **FIRED** — "forward dropped" |
| **M2** restore the `_local_bounds.default_*` override in `_local_test` | 1+1 substitutions | mutated `_local_test` ran; `_wbt` got `0.3/0.3` | **FIRED** — "forward dropped" |
| **M3a** delete the guard call from `run_sieve_pass` | anchor found exactly once | mutated `run_sieve_pass` ran and reached worker acquisition | **FIRED** — "quarantine did not fire — reached dispatch" |
| **M3b** neuter the guard **body** (swap it for a no-op) | swapped once, restored in `finally` | the both-mode trial ran to completion | **FIRED** — "a both-mode PWC trial completed with the guard neutered" |

M3a's detector is the **same function** G-PWC-HYBRID uses (`_detect_quarantine`), not a
lookalike written for the mutant. M3b exists because M3a alone would pass on the mere
*presence* of a call that does nothing. The mutated PWC drivers stub the coordinator so
that "the guard did not fire" surfaces as a controlled `_DispatchReached` or a fake-sieve
completion — never a real fleet spawn. Clean control for all three: the unmutated gate run
above.

---

## 5. Miner unchanged — confirmation

- `git status --porcelain` shows **no path** under `miner/`, and none of
  `sieve_gpu_worker.py`, `prng_registry.py`, `persistent/pwc_protocol.py`, is modified.
  Asserted by G-MINER-UNCHANGED, not just claimed here.
- `tests/test_s172_phase5_d6_threshold_path.py` re-run: **17/17 D6 threshold-path checks
  green (11 mutants killed)** — identical to the pre-edit baseline. Its own tally line is
  parsed, not just its exit code.
- The miner's D6 guarantee ("no drop below `run_bidirectional_test`") was never in
  question; this defect sat **above** the backend split. What changes for the miner is that
  the value it is handed (`window_optimizer_integration_final.py:1216-1217`, the
  `run_trial_miner` ingress) is now the sampled one rather than the default.

---

## 6. `coordinator.py:744` — disposition: **report, defer** (no change made)

The two hybrid signals do come from different authorities, as the brief describes:

- `coordinator.py:744` sets `'hybrid': '_hybrid' in job.prng_type` → `True` for a hybrid
  family, derived from the job.
- `coordinator.py:2298-2299` sets `phase1_threshold` / `phase2_threshold` to `None` unless
  `use_hybrid` (`coordinator.py:2280`, `getattr(args, 'hybrid', False)`), which the
  integration layer's `Args` class never sets.

**Refinement from the live read — this is currently benign, in both halves:**

1. `job['hybrid']` is **not** the kernel selector. `sieve_gpu_worker.py:363` reads it only
   to decide the *result shape* (whether `strategy_ids` / `skip_sequences` arrays are
   emitted). Kernel selection is by `prng_families` / `family_name`
   (`sieve_gpu_worker.py:232`). So `hybrid: True` on a hybrid family is **correct**, not a
   mis-signal.
2. `phase2_threshold = None` makes `sieve_gpu_worker.py:258` fall through to
   `hybrid_threshold = threshold` — i.e. the legacy route runs hybrid at **the trial's own
   directional threshold**, which is the behaviour the miner enforces by construction.

So the legacy route is coherent today, and the divergence the audit's F6 found is
PWC-specific. Fixing the two-authority structure would mean reworking the legacy
coordinator's `_sieve_config` authority — not small, not safe to do inside a Priority-0
threshold repair, and it would change legacy behaviour that is currently correct.
**Deferred with rationale rather than touched.**

---

## 7. Two hazards found while tracing (reported, not fixed)

1. **`docs/window_optimizer_integration_final.py` is a tracked, pushed, pre-fix snapshot of
   the production file** (101,985 bytes) carrying the defect in **both** routes (`:1394`
   `ft=bounds.default_forward_threshold`, `:928`
   `forward_threshold=_local_bounds.default_forward_threshold`).

   Provenance, verified 2026-07-31: the `docs/` path was added at **`7313a43`
   (2026-05-03)**, which git records as a 99%-similarity **copy** of the production file
   (`C099`). Its content is byte-identical to production at **`e8a69f5` (2026-04-22)** —
   already 8 days stale when committed, and predating `3fdf434` (2026-04-30).

   **Correction to an earlier draft of this report:** it was described as "precisely the
   mechanism of `2389b61`". That overstates the evidence. The file `2389b61` produced is
   **not** byte-identical to this copy (line offsets differ by 2), so it was not a straight
   copy-over from `docs/`. The shared pre-fix fingerprints (`run_bidirectional_test`
   defaults `0.01/0.01`) are consistent with *any* pre-`3fdf434` source. This file is a
   demonstrated **re-introduction vector**, not a demonstrated cause.

   `window_optimizer_integration_final_INTEGRATED.py` (8,640 bytes, 234 lines vs the
   production file's 2,679) is a separate, much earlier artifact from the initial commit
   (`0101306`, 2025-11-29), never modified since. Session S103 (2026-02-21) already assessed
   it as a stale generated artifact not imported by anything active.

   **DISPOSITION — CLOSED 2026-07-31: leave both files alone. No action taken, none
   proposed.** Michael's ruling, on these grounds:
   - the failure mode is already defended — `tests/test_s172_threshold_propagation.py`
     executes the live source of both call sites, so a stale-copy overwrite of the
     production file reds the gates **regardless of which copy it came from**. Removing one
     candidate source does not add protection;
   - the import-shadow risk is **latent, not live**: no module imports either file, nothing
     puts `docs/` on `sys.path`, and `docs/` holds no other `.py`;
   - **S103 (2026-02-21) already ruled** on `_INTEGRATED`; reopening it here would be drift;
   - deleting either path **reds Phase-4 gate 22** (it counts deleted `.py` too), forcing a
     second whitelist edit in a repair that has no need of one;
   - both are pushed to a public mirror, so deletion removes the convenient copy, not the
     artifact.

   Recorded here only so the finding is not lost and is not re-proposed as a new discovery.
   An earlier draft of this report recommended deletion; that recommendation is **withdrawn
   as out of scope for this repair.**
2. **`sieve_gpu_worker.py:44` replaces `sys.stdout`** with a fresh file object on the same
   fd at import time. Anything the importing process still had buffered on the original
   stdout is **discarded**. This was observed live: the first gate run's header and its
   first two `PASS` lines vanished from the captured log the moment G-KERNEL imported the
   worker. Any tool that imports `sieve_gpu_worker` in-process can silently lose its own
   earlier output — a VIR-1 hazard (a report losing its own lines reads as a report that
   never made the claim). Worked around in the harness with flush-on-print; the module
   itself is untouched.

Also observed, not a defect: `window_optimizer_bayesian.py:814-815` stamps
`bounds.default_*` onto the `WindowConfig` in `_vector_to_config` (the GP/vector strategy).
That is a **producer** choice — that strategy does not search the threshold dimension — not
a drop. The value it writes now propagates faithfully through the repaired path.

---

## 8. Non-regression (brief §7)

Captured **before** any edit at `a0442a0` on a tree restored to HEAD, and again after.

| suite | pre-edit (`a0442a0`, tree at HEAD) | post-edit |
|---|---|---|
| D0 | rc=0 (3s) | rc=0 (2s) |
| D1-engine | rc=0 (217s) | rc=0 (215s) |
| D1-workflow | rc=0 (107s) | rc=0 (109s) |
| D2 | rc=0 (425s) | rc=0 (422s) |
| D3.0 | rc=0 | rc=0 |
| D3 | rc=0 | rc=0 |
| D3.25 | rc=0 (2s) | rc=0 (2s) |
| D3.5 | rc=0 (7s) | rc=0 (7s) |
| D4 | rc=0 (2s) | rc=0 (2s) |
| D5 | rc=0 (726s) | rc=0 (722s) |
| D6 3.A | rc=0 (18s) | rc=0 (22s) |
| **D6-threshold** | rc=0 — **17/17** (11 mutants killed) | rc=0 — **17/17** (11 mutants killed) |
| D6.1 | rc=0 | rc=0 |
| Phase 3 | rc=0 (42s) | rc=0 (40s) |
| Phase 4 | rc=0 (60s) | rc=0 (59s) |
| **total** | **15/15 green** | **15/15 green** |

**D6-threshold: 17/17 both runs — the miner's already-correct path is undisturbed.**

**One regression was caught by this process and fixed, not waived.** An intermediate
version of the quarantine turned **D3.25 red (12/13)**; the guard was moved from the trial
entry point to the execution boundary and D3.25 returned to green. The failure and its
resolution are described in §3. The first baseline attempt was also discarded and re-run
from a tree restored to HEAD, because edits landed while it was mid-flight — a contaminated
baseline is not a baseline.

`tests/test_s172_phase4_coordinator.py` gate 22 (coexistence) asserts that every
changed/untracked `.py` is on a whitelist. The new harness
`tests/test_s172_threshold_propagation.py` was registered there with a review-flagged
comment, following the standing pattern already used for D2/D3/D3.25/D3.5/D4/D5/D6/D6.1.
`window_optimizer_integration_final.py` and `persistent_worker_coordinator.py` were already
whitelisted. `persistent/pwc_protocol.py` remains unmodified, so the gate's frozen-protocol
assertion still holds.

### 8.1 Post-edit results

*(filled in from `nonreg_postedit/SUMMARY.txt` — see §11.)*

---

## 9. Deferred items (brief §6 — named, not built)

1. **Hybrid skip-bound dead dimension.** Hybrid kernels ignore the trial's sampled
   `skip_min`/`skip_max` and start from a hardcoded `int expected_skip = 5`
   (`prng_registry.py:1027`, `:805`, `:885`, `:1159`); neither hybrid signature declares
   `skip_min`/`skip_max`, while every constant kernel does (`prng_registry.py:413`, `:470`,
   `:522`, `:619`, `:684`, `:963`, `:1090`). Requires a kernel-signature change. Beta lists
   this at revised-Phase-6 step 2. **Historical variable-skip trials are independently
   suspect for this reason**, separately from the threshold defect — and this is the second
   reason PWC hybrid is quarantined rather than propagated (§3).
2. **Study ↔ commit provenance binding.** Bind each Optuna study to repository commit, tree
   cleanliness, dataset identity and execution route. This is the gap that makes the
   2026-05-11 study **INDETERMINATE rather than decidable** — it falls between the April fix
   and the July revert with no run-bound provenance. Classify historical studies by interval,
   do not characterise them as poisoned without evidence.
3. **Replacement-resistant regression gate.** A standing gate that survives whole-block
   replacement. The §4 gates here are already behavioural rather than text-anchored — they
   execute the extracted live source, so they *would* have caught `2389b61` — but they are a
   deliverable harness, not a wired-in standing gate over the whole propagation chain. That
   remains Beta's item.

**Not touched, per brief §5:** hybrid skip bounds; the miner's D6 threshold/provenance/
residue work; PWC/ZMQ ingress; the D3.25 contract; `TestResult` shape; D5's artifact
contract; dataset schema work; Phase 6 known-answer fixtures (Beta's Wall C).
**`s172_threshold_patch.py` was not run** — not even `--dry-run`. **No Optuna study database
was read for mutation, deleted, moved or overwritten.**

---

## 10. Verification-integrity declaration (VIR-1…6)

- **execution proof:** every gate executes live source extracted from disk at run time;
  G-KERNEL reads the effective scalar **off the real cupy `RawKernel` launch arguments** on
  the RTX 3080 Ti, not recomputed from config. The harness prints its own
  `COMPLETION SENTINEL` line and flushes every line (see §7.2).
- **clean control:** the unmutated gate run — 5/5 PASS — is the clean control for all three
  mutants. G-ROUTE-B additionally asserts the *producer* (`_worker_obj`) still emits the
  sampled values, so the gate cannot pass vacuously against a dead producer.
- **fault-injection control:** M1/M2/M3, each with an applies-exactly-once proof, evidence
  the mutated path executed, and the observed detector firing.
- **completion sentinel:** `PASS | FAIL | UNAVAILABLE | INCOMPLETE`, printed explicitly;
  the runner returns non-zero for anything but `PASS`.
- **unavailable-observer behavior:** G-KERNEL raises `UNAVAILABLE:` and the run terminates
  `INCOMPLETE` if cupy or a device is missing. It does **not** skip. `daily3.json` absence
  is likewise `UNAVAILABLE`, not clean.
- **audit claim scope:** **repo-scoped, plus one host.** All source claims are VM 101
  working tree at `a0442a0`. Runtime claims (kernel launch args, `SearchBounds` values,
  cupy 13.5.1 / 1 visible device) are **VM 101 only**.
- **searched surfaces:** VM 101 working tree — `/bin/grep` (not the shell wrapper, which
  honours `.gitignore` and skips `*.json`) for `forward_threshold`, `reverse_threshold`,
  `phase2_threshold`, `default_forward_threshold`, `default_reverse_threshold`,
  `test_configuration`, `run_sieve_pass(`, `def test_config`, `_local_test`, `hybrid`;
  AST uniqueness checks on all six extracted functions; `git status --porcelain`;
  live cupy device count; live execution of the real sieve kernels.
- **unavailable surfaces:** rig CT100s were **not** checked this session — at audit time
  (2026-07-30) `.156` and `.164` had no `~/distributed_prng_analysis` at all, and `.122`
  held `sieve_gpu_worker.py`/`prng_registry.py` byte-identical to VM 101, neither of which
  this repair modifies. **No claim here is provisioning-scoped**: systemd units, cron and
  deployed-but-uncommitted copies were not searched.

---

## 11. Files changed

| file | change |
|---|---|
| `window_optimizer_integration_final.py` | NEW `ThresholdResolutionError` + `resolve_directional_threshold` (`:196-243`); Route A signature + call-time resolution (`:2348-2364`); Route B call site (`:1880-1885`) and its resolver import (`:1780-1785`) |
| `persistent_worker_coordinator.py` | NEW quarantine constant/exception/guard (`:145-204`); execution-boundary gate (`:1210`); placement rationale for the *absent* pre-flight (`:1640-1652`) |
| `tests/test_s172_threshold_propagation.py` | **NEW** — 5 gates + 3 mutants |
| `tests/test_s172_phase4_coordinator.py` | gate 22 whitelist entry for the new harness (review-flagged) |
| `docs/S172_THRESHOLD_PROPAGATION_REPAIR_REPORT.md` | **NEW** — this report |

Nothing under `miner/` was touched. `sieve_gpu_worker.py`, `prng_registry.py`,
`persistent/pwc_protocol.py`, `distributed_config.json` and every Optuna study database are
unmodified.

**STOPPED at the gate for Team Alpha review. Not committed, not pushed.**
