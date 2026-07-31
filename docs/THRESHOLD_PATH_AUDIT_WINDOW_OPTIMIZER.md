# Threshold Path Audit — Window-Optimizer / PWC Route

**Question (falsifiable):** in the window-optimizer / PWC route, does a configured or
calibrated forward/reverse threshold actually reach the CUDA/ROCm sieve kernel, or is it
dropped at some hop in favour of a default?

**Verdict on the `tfm-project-facts` §2.7 claim** ("Optuna thresholds never reach the sieve
— `test_config` default args `ft`/`rt` are never supplied, so every trial runs at 0.30/0.30
while the study records suggested values"):

> ## **CONFIRMED — and it is a REGRESSION, not a never-fixed defect.**

The bug was fixed on 2026-04-30 in commit `3fdf434` ("S172: Fix Optuna threshold-drop bug")
and **silently reverted on 2026-07-07** by commit `2389b61` (S172 Phase 0, PRNG_TYPE_ENCODING
v3.2), which rewrote `window_optimizer_integration_final.py` from a pre-`3fdf434` copy. The
live tree at HEAD carries the *pre-fix* code. `window_optimizer.py` kept its half of the
same patch, so the two files now disagree.

- Audit host: **VM101 `192.168.3.177`** (`zeus-ubuntu-vm`), run as `michael`.
- Repo state: `/home/michael/distributed_prng_analysis`, HEAD `91f0521`, working tree clean
  for every file cited below (all cited files are tracked and unmodified).
- Date: 2026-07-30. **No code was changed by this audit.**

---

## 1. Executive summary

| # | Finding | Status |
|---|---|---|
| **F1** | Optuna's sampled `forward_threshold`/`reverse_threshold` are dropped at `test_config` (single-process path). Every trial filters at the config default **0.30 / 0.30**. | **CONFIRMED — live** |
| **F2** | The `n_parallel > 1` path drops them too, and does so *explicitly* — it builds a `WindowConfig` carrying the sampled values, then overrides with `_local_bounds.default_*`. | **CONFIRMED — live** |
| **F3** | F1/F2 were fixed in `3fdf434` and reverted in `2389b61`. Two companion defensive fixes (`run_bidirectional_test` defaults `0.50 → 0.01`) were reverted in the same commit. | **CONFIRMED — regression** |
| **F4** | `window_optimizer.py:450` docstring asserts the fixed behaviour ("Thresholds are now taken from `config.forward_threshold`…"). It is false at HEAD — the file's own half of the patch survived, the integration file's half did not. | **CONFIRMED** |
| **F5** | §2.7 instance 2 — hybrid kernels hardcode `expected_skip = 5`, and `skip_min`/`skip_max` are absent from the hybrid kernel signatures. | **CONFIRMED — live** |
| **F6** | §2.7 instance 3 — variable-skip at `0.50` while constant runs `0.30`. **Route-dependent, not universal:** true on the **PWC** route only. The legacy-coordinator route runs hybrid at the *same* value as constant; the **miner** route pins them equal by construction. | **CONFIRMED for PWC; REFUTED for legacy + miner** |
| **F7** | **TRSE produces no threshold candidates at all** — it calibrates window/regime quantities. Only Rule A is applied, and it moves `bounds.max_window_size`, nothing else. | **CHANGED — the premise does not apply** |
| **F8** | **Phase-6 impact is the opposite of the one feared for constant skip, and real for variable skip.** All four backends receive the *same* poisoned scalar from one shared function parameter, so constant-skip four-path comparison stays consistent (all at 0.30). Variable-skip diverges: **PWC 0.50 vs miner 0.30 vs legacy 0.30**. | see §6 |

**The empirical smoking gun.** The most recent Optuna study on this box,
`optuna_studies/window_opt_1778552567.db` (2026-05-11 19:24), records trial 1 as
`forward_threshold = 0.73`, `reverse_threshold = 0.31`. The code path that executed it can
only have filtered at `0.30 / 0.30`. The same pair is written out to
`optimal_window_config.json` → `agent_metadata.suggested_params` as
`forward_threshold: 0.73, reverse_threshold: 0.31` — a value no kernel ever used.

---

## 2. Hop-by-hop trace — Route A (single-process, `--n-parallel 1`, the default)

`window_optimizer.py:1082` sets `--n-parallel` default `1`, so this is the production path
unless a launcher overrides it.

| Hop | Producer (`file:line`) | Consumer (`file:line`) | Value carried | Intact? |
|---|---|---|---|---|
| 1 | `window_optimizer_bayesian.py:437-442` — `trial.suggest_float('forward_threshold', …)` / `('reverse_threshold', …)` over `[0.30, 0.75]` | `window_optimizer_bayesian.py:445-452` — `WindowConfig(… forward_threshold=round(ft,2), reverse_threshold=round(rt,2))` | Optuna sample, e.g. `0.73 / 0.31` | ✅ |
| 2 | `window_optimizer_bayesian.py:456` — `result = objective_function(config, optuna_trial=trial)` | `window_optimizer.py:480-482` — `objective()` | `config` still carries the sample | ✅ |
| 3 | `window_optimizer.py:481-482` — `self.test_configuration(config, seed_start, seed_count, optuna_trial=optuna_trial)` — **3 positional args, no `ft`/`rt`** | `window_optimizer_integration_final.py:2262-2266` (bound at `:2291`, `optimizer.test_configuration = test_config`) | — | ❌ **DROP** |
| 4 | `window_optimizer_integration_final.py:2264-2265` — `ft=bounds.default_forward_threshold`, `rt=bounds.default_reverse_threshold` | function body | **`0.30 / 0.30`** | ❌ default wins |
| 4a | `window_optimizer_integration_final.py:2259` — `bounds = SearchBounds.from_config()` → `window_optimizer.py:133-150` → `distributed_config.json` `search_bounds.forward_threshold.default` | `SearchBounds.default_forward_threshold` | `0.3` (live-resolved, §3) | ✅ (of the wrong quantity) |
| 5 | `window_optimizer_integration_final.py:2283-2284` — `forward_threshold=ft, reverse_threshold=rt` | `run_bidirectional_test`, `window_optimizer_integration_final.py:1067-1080` | `0.30 / 0.30` | — |
| 6a | `window_optimizer_integration_final.py:1330` — `Args.threshold = forward_threshold`; `:1382` `reverse_args.threshold = reverse_threshold` | `coordinator.execute_distributed_analysis` (`:1352`, `:1387`) | `0.30 / 0.30` | — |
| 6b | `window_optimizer_integration_final.py:1235-1236` — `forward_threshold=…, reverse_threshold=…` | `persistent_worker_coordinator.run_trial_persistent:1548-1549` | `0.30 / 0.30` | — |
| 6c | `window_optimizer_integration_final.py:1293-1294` | `zmq_sqlite_coordinator.run_trial_zmq_sqlite` | `0.30 / 0.30` | — |
| 6d | `window_optimizer_integration_final.py:1149-1150` | `miner.run_trial_miner` (D6 ingress) | `0.30 / 0.30` | — |
| 7 (PWC) | `persistent_worker_coordinator.py:1614` `threshold=forward_threshold`; `:1655` `threshold=reverse_threshold` | `run_sieve_pass:1110-1123` | `0.30 / 0.30` | — |
| 8 (PWC) | `persistent_worker_coordinator.py:1222` — `"min_match_threshold": threshold` | job dict → worker | `0.30` | — |
| 9 | `sieve_gpu_worker.py:156` — `threshold = coerce_threshold(job.get('min_match_threshold'), 0.25)` | `sieve_gpu_worker.py:215` — `cp.float32(threshold)` into the kernel; `:314` `if rate >= threshold` | **`float32(0.30)` reaches the CUDA/ROCm kernel** | — |

**Net:** the scalar that reaches the kernel on this route is always
`SearchBounds.default_forward_threshold` / `default_reverse_threshold`. Optuna's sample never
travels past hop 3.

### Binding proof (VIR-2 controls)

A standalone probe replicating *only* the call/def pair at
`window_optimizer.py:481` and `window_optimizer_integration_final.py:2262` (no pipeline code
imported, no GPU):

```
CLEAN CONTROL (live signature)   : optuna sampled 0.73/0.31 -> kernel got 0.3/0.3
POSITIVE CONTROL (3fdf434 form)  : optuna sampled 0.73/0.31 -> kernel got 0.73/0.31
```

The positive control is the exact patched body from `3fdf434`; it recovers the sampled
values, which establishes the probe is not vacuous.

---

## 3. Runtime value — live resolution, not read from a doc

```
$ python3 -c "from window_optimizer import SearchBounds; b=SearchBounds.from_config(); ..."
default_forward_threshold = 0.3
default_reverse_threshold = 0.3
min/max fwd = 0.3 0.75
min/max rev = 0.3 0.75
```

Source: `distributed_config.json` → `search_bounds.forward_threshold = {"min":0.3,"max":0.75,"default":0.3}`
(same for `reverse_threshold`).

Note the layered irony: `3fdf434`'s FIX 1 raised the *code* fallbacks to `0.50`
(`window_optimizer.py:123-129`, `:149-150`, `:177-178` all survive at HEAD), but the JSON
`default` key is `0.3` and takes precedence, so the raised code floor never fires. The
`0.30` the skill recorded is exact.

Unrelated key, listed to avoid confusion: `distributed_config.json` → `sieve_defaults`
carries `min_match_threshold: 0.01`, `phase1_threshold: 0.01`, `phase2_threshold: 0.01`.
These are the *legacy-coordinator* fallbacks (`coordinator.py:2285-2286`) and are not
consulted on the window-optimizer route, because `Args` always sets `threshold` explicitly.

---

## 4. Hop-by-hop trace — Route B (`--n-parallel > 1`)

| Hop | `file:line` | Value | Intact? |
|---|---|---|---|
| 1 | `window_optimizer_integration_final.py:1829-1834` — `ft = trial.suggest_float('forward_threshold', …)`, `rt = trial.suggest_float('reverse_threshold', …)` | Optuna sample | ✅ |
| 2 | `window_optimizer_integration_final.py:1835-1841` — `cfg = WindowConfig(… forward_threshold=round(ft,2), reverse_threshold=round(rt,2))` | sample on the config | ✅ |
| 3 | `window_optimizer_integration_final.py:1842` — `result = _local_test(cfg, optuna_trial=trial)` | — | ✅ |
| 4 | `window_optimizer_integration_final.py:1798-1799` — `forward_threshold=_local_bounds.default_forward_threshold`, `reverse_threshold=_local_bounds.default_reverse_threshold` | **`0.30 / 0.30`** | ❌ **DROP — explicit override** |

Route B is worse than Route A: the sampled values are present on `cfg`, in scope, on the
adjacent line, and are discarded by an explicit assignment rather than by a defaulted
parameter. `3fdf434` never touched this call site.

---

## 5. Comparison — the miner's post-D6 path (the contrast case)

| Hop | `file:line` | Behaviour |
|---|---|---|
| resolve | `miner/range_miner_coordinator.py:3410-3419` — `resolved_threshold = float(forward_threshold)` / `float(reverse_threshold)`, direction-resolved per stripe via the §6.8 phase table | one canonical resolution in the parent |
| payload | `miner/range_miner_coordinator.py:3444-3445` — `"min_match_threshold": resolved_threshold`, `"phase2_threshold": resolved_threshold` | both keys pinned **equal** |
| parent gate | `miner/range_miner_coordinator.py:3449-3453` — raises `MinerMetadataError` if the two ever differ | fail-closed at the producer |
| worker gate | `miner/range_miner_worker.py:789-797` — re-validates on receipt, raises `ThresholdContractError` on a contradictory pair | fail-closed at the consumer |
| kernel | `miner/range_miner_worker.py:784`, `:858-863`, `:913` — `effective_threshold` read back **off the real executor** | requested / payload / effective recorded separately (`range_miner_coordinator.py:1644-1652`) |
| metadata | `miner/range_miner_coordinator.py:1539-1540` — `threshold_used` selected per direction | per-stripe provenance |

The miner honours whatever it is handed and proves it did. **But on the window-optimizer
route it is handed `0.30 / 0.30`** (hop 6d above) — D6's guarantee is "no drop below
`run_bidirectional_test`", and the drop audited here is *above* it. D6 is not defeated; it
is simply out of scope of this defect.

---

## 6. Phase-6 four-path impact — corrected from the brief's premise

The brief's stated worry was that PWC would run at a hardcoded value while the miner honours
its configured one, making the miner look defective. That is **not** what the code does for
constant skip:

- All four backends (legacy / PWC / ZMQ / miner) read the *same two function parameters* of
  `run_bidirectional_test` (`window_optimizer_integration_final.py:1074-1075`), fanned out at
  `:1149-1150`, `:1235-1236`, `:1293-1294`, `:1330`/`:1382`.
- Therefore **constant-skip four-path parity is not threatened by this defect.** All four
  paths run at `0.30 / 0.30` together. Do not "fix the miner to match the oracle" — there is
  nothing to reconcile on this axis.
- What *is* corrupted is **provenance and optimisation**: the study, `step1_trial_history`,
  the `optimal_window_config.json` `suggested_params` block, and any KPI or governance
  record derived from them all assert threshold values that were never executed. Optuna's
  TPE has been optimising a dimension with **zero effect on the objective** for every trial
  since 2026-07-07 (and before 2026-04-30).

**Variable skip is a genuine four-path divergence** — see §7, F6.

---

## 7. The other two open §2.7 instances

### F5 — hybrid kernels hardcode `expected_skip = 5` — **CONFIRMED**

`prng_registry.py` live at HEAD, inside the CUDA source strings:

| kernel | signature | hardcode |
|---|---|---|
| `java_lcg_hybrid_multi_strategy_sieve` | `prng_registry.py:1007-1013` — params end `… n_strategies, float threshold, unsigned long long a, unsigned long long c` | `prng_registry.py:1027` — `int expected_skip = 5;` |
| `java_lcg_hybrid_reverse_sieve` | `prng_registry.py:3172-3178` — params end `… n_strategies, float threshold, int offset` | (uses `skip_tolerance` search from `try_skip = 0`) |
| further hybrid families | `prng_registry.py:805`, `:885`, `:1159` | `int expected_skip = 5;` |

Neither hybrid signature declares `skip_min` or `skip_max`. Contrast the **constant** kernels,
which do: `prng_registry.py:413`, `:470`, `:522`, `:619`, `:684`, `:963`, `:1090` all carry
`int skip_min, int skip_max` and loop `for (int skip = skip_min; skip <= skip_max; skip++)`.

So on the hybrid path the trial's `skip_min`/`skip_max` are transported (they ride in
`skip_range` on the job, `persistent_worker_coordinator.py:1223`) but are **not kernel
parameters**; the kernel starts from a fixed guess of 5 and searches
`±strategy_tolerances[strat_id]` (`prng_registry.py:813-814`). The skip window Optuna
explores does not constrain the hybrid kernel.

### F6 — variable-skip at `0.50` vs constant `0.30` — **CONFIRMED for PWC only**

Mechanism, PWC route:

1. `persistent_worker_coordinator.py:1119` — `run_sieve_pass(… phase2_threshold: float = 0.5 …)`.
2. The two hybrid call sites — `persistent_worker_coordinator.py:1699-1710` (forward) and
   `:1726-1739` (reverse) — pass `threshold=forward_threshold` / `reverse_threshold` but
   **never pass `phase2_threshold`**, so the `0.5` default stands.
3. `persistent_worker_coordinator.py:1229` — `"phase2_threshold": phase2_threshold` → job.
4. `sieve_gpu_worker.py:257-258` — `phase2_raw = job.get('phase2_threshold'); hybrid_threshold = coerce_threshold(phase2_raw, threshold) if phase2_raw is not None else threshold`.
   `phase2_raw` is `0.5`, not `None`, so **`hybrid_threshold = 0.50`**.
5. `sieve_gpu_worker.py:266` / `:277` — `cp.float32(hybrid_threshold)` into the hybrid kernel;
   `:288` — `if rate >= hybrid_threshold`.

Grep confirms no caller anywhere supplies `phase2_threshold` to `run_sieve_pass`
(`/bin/grep -rn "run_sieve_pass(" --include=*.py .` — the only live callers are
`persistent_worker_coordinator.py:1610`, `:1651`, `:1699`, `:1726`, plus test doubles).

**Not universal — the other two routes behave differently:**

- **Legacy coordinator route:** `coordinator.py:2280` — `use_hybrid = getattr(args, 'hybrid', False)`.
  The integration's `Args` class (`window_optimizer_integration_final.py:1320-1341`) never sets
  a `hybrid` attribute, so `use_hybrid` is `False`, so `coordinator.py:2298` sets
  `phase2_threshold = None`. `sieve_gpu_worker.py:258` then falls through to
  `hybrid_threshold = threshold`. Hybrid runs at **the same value as constant** (`0.30`).
  (Note `coordinator.py:744` independently sets `'hybrid': '_hybrid' in job.prng_type` → `True`,
  so the hybrid *kernel* is selected while its threshold key is `None` — the two hybrid
  signals are derived from different sources.)
- **Miner route:** pinned equal by construction and enforced at both ends — §5.

**Consequence for bounded Phase 6:** on the variable-skip axis the four paths are
**not** comparable as-is. With `test_both_modes=True` and the same trial config, PWC filters
hybrid survivors at `0.50` while the miner and the legacy coordinator filter at `0.30`. PWC
will return strictly fewer hybrid survivors, and — because PWC is the authoritative
comparator — the miner will look like it is over-producing. **This is the "broken oracle"
scenario the brief anticipated; it lives in the hybrid threshold key, not the constant one.**

---

## 8. TRSE / Step 0 — what it actually calibrates (F7)

| Artifact | Producer | What it contains | Consumer | Reaches a sieve threshold? |
|---|---|---|---|---|
| `trse_context.json` | `trse_step0.py:135` (`DEFAULT_OUTPUT`) | keys: `regime_type`, `regime_type_confidence`, `regime_stable`, `recommended_window_size`, `window_coherence_ceiling`, `w3_w8_ratio`, `skip_entropy_profile`, `dominant_offset_lag`, `confirmed_windows`, … — **no forward/reverse threshold key** | `window_optimizer_bayesian.py:25-27` (`_load_trse_context`), applied at `:495-535`; mirrored in the NP2 worker at `window_optimizer_integration_final.py:1760-1790` | **No** |
| `trse_boundary_candidates.json` | `trse_entropy_probe.py` | `meta / series / stats / boundary_candidates / candidate_count`; the only "threshold" is `z_threshold` (default `DEFAULT_Z = 2.0`, `trse_entropy_probe.py:44`) — a z-score flag for regime boundaries | no live consumer found in the searched surface | **No** |
| — | `trse_calibration_probe.py` | recalibration of **regime-classification** cutoffs (`:34`, `:168`, `:362`) — the `classify_regime_type` decision boundary, not a sieve match rate | advisory/stdout | **No** |
| — | `step0_heuristic_validation.py` | contains **no** occurrence of the string `threshold` (295 lines, searched) | — | **No** |

TRSE's only *applied* effect on Step 1 is **Rule A**, `window_optimizer_bayesian.py:508-517`:
when `regime_type == 'short_persistence'` and `regime_type_confidence >= 0.70` and
`regime_stable`, it lowers `bounds.max_window_size` to `min(32, …)`. **Rule B (skip bounds)
and Rule C (offset) are logged only, disabled per TB S121** (`:522-534`). Nothing in TRSE
touches `min_forward_threshold`, `default_forward_threshold`, or any threshold reaching a
kernel.

**Separately: the one tool that *does* calibrate sieve thresholds has no automated consumer.**
`ca_d3_threshold_calibration.py` runs the real GPU sieve worker and derives conservative /
permissive threshold baselines — then **prints** them (`:388-392`) with the instruction to
hand-edit `persistent_worker_coordinator.py → threshold`, `window_optimizer.py →
default_forward_threshold`, `WindowSearchBounds → min_forward_threshold`. It writes no
artifact any code reads (its only file write is a temp draw file, `:328-329`). So "calibrated
threshold value" exists as a human recommendation only, and even a hand-applied one would
land on `default_forward_threshold` — i.e. on the value that *does* reach the kernel, which
is why the drop has stayed invisible.

---

## 9. Where exactly the value is dropped

```
Optuna suggest ──► WindowConfig ──► objective() ──►╳ test_config ──► run_bidirectional_test ──► backend ──► kernel
  (0.73/0.31)      (0.73/0.31)      (0.73/0.31)    ▲  (0.30/0.30)      (0.30/0.30)              (0.30)      float32(0.30)
                                                   │
                    THE DROP: window_optimizer.py:481-482 passes 3 positional args;
                    window_optimizer_integration_final.py:2264-2265 defaults ft/rt to
                    bounds.default_*; config.forward_threshold is never read.
                    (Route B: explicit override at :1798-1799.)
```

Everything downstream of `run_bidirectional_test` is faithful on all four backends. The
defect is a single hop, in one file, in two places.

---

## 10. Regression forensics

```
3fdf434  2026-04-30  S172: Fix Optuna threshold-drop bug + restore full cluster + window_size.min=6
2389b61  2026-07-07  feat(s172): Phase 0 — shared PRNG_TYPE_ENCODING v3.2 (registry-derived)
                     └─ 30 insertions, 35 deletions in window_optimizer_integration_final.py
```

`git log -S "getattr(config, 'forward_threshold'" -- window_optimizer_integration_final.py`
returns exactly those two commits: `3fdf434` added the marker, `2389b61` removed it.

`2389b61` reverted, in one file:

| reverted | from | to |
|---|---|---|
| `test_config` signature + body (FIX 2) | `ft=None` / `rt=None` + `getattr(config, …)` fallback | `ft=bounds.default_forward_threshold` / `rt=bounds.default_reverse_threshold` |
| `run_bidirectional_test` defaults (FIX 1.F) | `forward_threshold: float = 0.50` / `reverse_threshold: float = 0.50` | `= 0.01` / `= 0.01` (live at `window_optimizer_integration_final.py:1074-1075`) |

`2389b61`'s commit message describes only the PRNG-encoding work and names
`window_optimizer_integration_final.py` as "inline writer" patched — the threshold revert was
not intended and not mentioned. **This is a stale-copy overwrite**, the classic failure mode
for a file edited both by a patch script and by hand.

Corroborating artifacts still on disk (untracked backups, not evidence by themselves but
consistent): `window_optimizer_integration_final.py.s172_bak_20260430_203809`,
`window_optimizer.py.s172_bak_20260430_203809`, `distributed_config.json.s172_threshold_fix_bak`.

The idempotent re-application tool is still in the tree and still committed:
`s172_threshold_patch.py` (FIX 2 anchors at `:150-176`, post-condition check at `:283-300`).
Its FIX 2 anchor matches the live text exactly — it would re-apply cleanly. **Applying it is a
fix action and is explicitly out of scope for this read-only audit.**

---

## 11. Verification-integrity declaration (VIR-1…6)

- **execution proof:** every claim above carries a `file:line` obtained this session by
  `/bin/grep` / `sed` against the live tree at HEAD `91f0521` on VM101. The runtime value
  `0.30/0.30` was obtained by importing `SearchBounds.from_config()` on the box, not read
  from a doc. The recorded-vs-executed gap was read out of
  `optuna_studies/window_opt_1778552567.db` with `sqlite3`.
- **clean control:** the binding probe's positive-control arm (the `3fdf434` body) returns
  `0.73/0.31`, proving the probe distinguishes the two forms rather than always printing the
  default.
- **fault-injection control:** the probe injects the *fixed* code as the counterfactual
  (inverted-polarity injection: the defect is live, the fix is the injected variant). No
  defect was injected into any repo file.
- **completion sentinel:** this report reaching §11 with all eight findings resolved to
  CONFIRMED / REFUTED / CHANGED. No hop in §2 or §4 is left as "not traced".
- **unavailable-observer behavior:** two rig CT100s could not be checked (below); they are
  reported as **UNAVAILABLE for the deployed-copy question**, not as agreeing.
- **audit claim scope:** **repo + partial system.** The source-level verdict (F1–F7) is
  repo-scoped and complete. The deployed-copy question is system-scoped and only **1 of 3**
  rigs was observable.
- **searched surfaces:**
  - VM101 working tree `/home/michael/distributed_prng_analysis` at HEAD `91f0521` — full
    recursive `/bin/grep` (not the shell `grep` wrapper, which honours `.gitignore` and skips
    `*.json`) for `forward_threshold`, `reverse_threshold`, `phase2_threshold`, `expected_skip`,
    `skip_min`/`skip_max`, `suggest_float`, `def test_config`, `run_sieve_pass(`,
    `trse_context`, `trse_boundary_candidates`.
  - git history of `window_optimizer_integration_final.py` (`log -S`, `show`, `merge-base`).
  - live Python import of `window_optimizer.SearchBounds`; live `sqlite3` read of the newest
    Optuna study DB; live JSON parse of `distributed_config.json`,
    `optimal_window_config.json`, `trse_context.json`, `trse_boundary_candidates.json`.
  - process table (`ps -eo`) — confirmed no pipeline/optimizer run in flight during the audit
    (only `netconsole_listener.py`), so nothing read here was mid-write.
  - deployed copies on **rrig6600 CT100 `192.168.3.122`**: `sieve_gpu_worker.py` and
    `prng_registry.py` are **byte-identical** to VM101
    (`9ace0d83…4846`, `32435fb8…c0b7`), so F5/F6 hold on the executing host, not only in the repo.
- **unavailable surfaces:**
  - **rrig6600b CT100 `192.168.3.156`** and **rrig6600c CT100 `192.168.3.164`** — SSH key auth
    works, but `~/distributed_prng_analysis` does not exist on either; `find ~ -maxdepth 3
    -name sieve_gpu_worker.py` returned nothing. Their home dirs hold `rocm_env` and
    llama.cpp work only. **Whether a drifted worker copy exists elsewhere on those hosts is
    UNVERIFIED** — declaring the absence I searched for, at the depth I searched it.
  - Zeus bare-metal `.127` — not booted (Zeus runs one OS at a time); not inspected.
  - Proxmox hosts `.121/.155/.163` — not inspected (no root key auth; not needed for this
    question).
  - Windows VM 100 — out of scope.
  - **Not searched:** systemd units, cron, or any provisioning surface. No claim here is
    provisioning-scoped.

---

## 12. What this audit did **not** do

No file was modified. `s172_threshold_patch.py` was **read, not run** — not even with
`--dry-run`. No pipeline, GPU job, or optimizer run was launched. No commit, no push.

## 13. Recommended follow-ups (for a separate *fix* brief — not performed here)

1. **Restore FIX 2** at `window_optimizer_integration_final.py:2262-2266` *and* fix Route B at
   `:1798-1799` (which `3fdf434` never covered). Re-running `s172_threshold_patch.py` alone
   would leave Route B broken.
2. **Decide the variable-skip threshold contract before bounded Phase 6.** Either give PWC's
   hybrid call sites an explicit `phase2_threshold=<the direction's threshold>` (matching the
   miner's pinned-equal contract), or freeze `0.50` as intended and make the miner and legacy
   route match. Comparing the four paths on variable skip before this is decided will produce
   a divergence that is an artifact of the oracle.
3. **Add a regression gate** that fails if the value reaching the kernel differs from
   `config.forward_threshold`/`reverse_threshold` — the D6 requested/payload/effective
   read-back pattern (`miner/range_miner_coordinator.py:1644-1652`) already exists and is the
   obvious model. A pure text-anchor check would not have caught `2389b61`, because the whole
   block was replaced; a behavioural gate would have.
4. **Treat every threshold value recorded before this is fixed as non-executed** — the study
   DBs, `step1_trial_history`, and `optimal_window_config.json`
   `agent_metadata.suggested_params` all assert values no kernel used.
5. **`window_optimizer.py:450`** — the docstring is currently false; it is already on the
   `tfm-project-facts` §3 supersession list, and the correct resolution is to make the code
   true rather than to soften the docstring.
