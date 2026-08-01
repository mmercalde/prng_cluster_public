# TRSE / Step 0 — Audit v1

**Session:** Claude Code on VM 101 (`michael@192.168.3.177`), `~/distributed_prng_analysis`, venv `~/venvs/torch`
**Base commit:** `9933ba2` (branch `main`)
**Date:** 2026-07-31
**Authority:** `docs/CLAUDE_CODE_INSTRUCTIONS_TRSE_STEP0_AUDIT.md` (REV1)
**Design-intent authority:** `docs/TRSE_v1_15_SPEC.md`, `docs/TRSE_INTEGRATION_PLAN_S121.md`
**Scope:** AUDIT ONLY. No code, config, or documentation was changed. Nothing was committed.

---

## 0. Headline

TRSE's **analytical core is sound and its one applied rule is correctly wired end-to-end.**
Rule A reaches Optuna's search space and demonstrably narrows it. Rules B and C are
**ADVISORY-BY-DESIGN**, disabled by an explicit Team Beta ruling — **not** dropped wires, and
**not** instances of the §2.7 defect class.

The defects found are **not** in what TRSE computes. They are in **how Step 0 is launched and
how its artifact ages**:

1. WATCHER's Step 0 command is **malformed** — it passes three arguments `trse_step0.py` does
   not accept. Exit code 2, proven by execution this session. Masked by `skip_on_fail`.
2. The freshness sentinel that decides whether Step 0 re-runs is **bumped by Step 1's own
   write-back**, producing a self-perpetuating "always fresh" state. The live context was
   computed **2026-03-13** and is still being applied.

Both are launch/lifecycle faults. Neither corrupts the regime mathematics.

---

## 1. What TRSE is

**TRSE = Temporal Regime Segmentation Engine**, pipeline **Step 0**, the head of the chain
(`agents/watcher_agent.py:388`, `:399`, `:410`).

It reads the draw history (`daily3.json`, 18,068 draws), slides windows across it, extracts
per-window entropy and digit-transition features (`trse_step0.py:153`, `:162`, `:185`), and
KMeans-clusters those windows at three scales — W200/W400/W800 (`:276`, `:302`) — to answer:
**which temporal regime is the sequence currently in, how long has it been there, and how
stable is it?**

v1.15 adds three analyses that *test for* structure the CA analysis already established
(`TRSE_v1_15_SPEC.md:15-21`):

| Function | `trse_step0.py` | Produces |
|---|---|---|
| `classify_regime_type` | `:397` | `regime_type`, `regime_type_confidence`, `duality_score`, `w3_w8_ratio`, `window_density_profile` |
| `analyze_skip_entropy` | `:483` | `skip_entropy_profile` |
| `detect_offset_periodicity` | `:538` | `dominant_offset_lag` |

**Purpose (spec-cited):** TRSE does **not** search for seeds. It is a *classifier* that
conditions Step 1's search space — "This is a *classification* based on signal shape, not a
brute-force seed search" (`TRSE_v1_15_SPEC.md:59-61`). Its entire sanctioned influence is
"limited to Step 1 SearchBounds narrowing only — Steps 2-6 unchanged"
(`agent_manifests/trse.json`, `notes`), corroborated by `TRSE_INTEGRATION_PLAN_S121.md:123-128`.

This is consistent with `docs/THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md:36` (F7): **TRSE
produces no threshold candidates.** That fact is cited, not re-derived.

---

## 2. Invocation model — the decisive question

**Verdict: BOTH — a registered pipeline step that is OPT-IN, plus a manual tool.**
It is **not** reached by a default pipeline run.

### 2.1 Step 0 is fully registered

| Surface | Anchor | State |
|---|---|---|
| `STEP_SCRIPTS[0] = "trse_step0.py"` | `agents/watcher_agent.py:388` | ✅ present |
| `STEP_MANIFESTS[0] = "trse.json"` | `agents/watcher_agent.py:399` | ✅ present |
| `STEP_NAMES[0]` | `agents/watcher_agent.py:410` | ✅ present |
| timeout override `{0: 1}` | `agents/watcher_agent.py:2813` | ✅ present |
| `agent_manifests/trse.json` | on disk, 2026-03-07 | ✅ present (**untracked** — see F5) |
| `PIPELINE_STEPS[0]` | `agents/pipeline/pipeline_step_context.py:65-77` | ✅ present |
| `step: int = Field(ge=0, le=6)` | `agents/full_agent_context.py:42` | ✅ allows 0 |
| `pipeline_step … ge=0` | `agents/manifest/agent_manifest.py:156` | ✅ allows 0 |
| `STEP_NAMES[0]` (display) | `agents/progress_display.py:409` | ✅ present |

All five files named in `TRSE_INTEGRATION_PLAN_S121.md:169-177` were delivered.

### 2.2 …but the default entry point starts at Step 1

```
run_pipeline(self, start_step: int = 1, end_step: int = 6, …)   agents/watcher_agent.py:2003
    """start_step: First step to run (1-6)"""                   agents/watcher_agent.py:2008
--start-step … default=1, help="Starting step for pipeline (1-6)"  agents/watcher_agent.py:2718-2723
```

The loop is `while … self.current_step <= end_step` seeded from `start_step`
(`:2031`, `:2040`). With the default, **Step 0 is never entered.** The module's own usage line
documents `--run-pipeline --start-step 1` (`:36`). Reaching Step 0 requires an explicit
`--start-step 0`, or the auto-approve dispatch path where `start_step = min(steps_to_run)`
(`:2576`) and the request happens to include 0.

### 2.3 It has been run that way — historically

- `watcher_decisions.jsonl`: **36** Step-0 decision records, 2026-03-07 → 2026-03-15.
- `logs/pa_step0_step1.log:14-34` — a real `--start-step 0` run.
- `logs/step01_200trial_5M.log:17`, `logs/step01_200trial_fresh.log:17`,
  `logs/step01_200trial_5M_s140b.log:17` — "Running Step 0: trse_step0.py".

**No Step-0 activity after 2026-03-15** in any searched surface.

### 2.4 The manual surface

`trse_step0.py:872` (`if __name__ == "__main__"`) with all-defaulted args (`:811-816`) — a
bare `python3 trse_step0.py` works. The other three modules are **manual-only**:

| Module | Callers found in code | Status |
|---|---|---|
| `trse_calibration_probe.py` | **none** | manual; output "for TB submission" (`:491`, `:540`) |
| `trse_entropy_probe.py` | **none** | manual |
| `step0_heuristic_validation.py` | **none** (only an untracked brief) | manual |

No cron entry, no systemd unit, and no `.sh` invokes any TRSE module (checked `crontab -l`,
`systemctl list-unit-files`, `--include=*.sh/.service/.timer`). The only TRSE-adjacent unit,
`daily3scraper.service`, is `disabled` (already known — skill §2.9).

**Consequence for classification:** TRSE is operator-elected. Rules B/C stopping at a human is
therefore *coherent with the invocation model*, not anomalous.

---

## 3. Output → consumer table

`trse_context.json` — written `trse_step0.py:868` (`save_context`, `:795-800`).
Loaded by `window_optimizer_bayesian.py:495` via `_load_trse_context` (`:25-47`); mirrored in
the NP2 partition-worker path at `window_optimizer_integration_final.py:1814-1850`.

| Output | Written | Read by | Effect | Class |
|---|---|---|---|---|
| `regime_type` | `trse_step0.py:397`→`:740` | `window_optimizer_bayesian.py:497` | Rule A gate | **ACCURATE — wired** |
| `regime_type_confidence` | `:397` | `:498` | Rule A gate (≥0.70) | **ACCURATE — wired** |
| `regime_stable` | `:623` block | `:499` | Rule A gate | **ACCURATE — wired** |
| `trse_version` | `:708` | `:39-43` | version guard, requires ≥1.15 | **ACCURATE — wired** |
| `w3_w8_ratio` | `:397` | `:501` | **printed only** (`:505`) | ADVISORY-BY-DESIGN |
| `recommended_window_size` | `:740` (echoes CLI arg `:859`) | `:500` `_rec_ws` | **loaded, never used** | **DEFECT — dead (F6)** |
| `skip_entropy_profile` | `:483` | `:524` | log line only (`:526`) | **ADVISORY-BY-DESIGN** |
| `dominant_offset_lag` | `:538` | `:530` | log line only (`:532`) | **ADVISORY-BY-DESIGN** |
| `confirmed_windows` | `window_optimizer.py:789-794` (**Step 1 writes back**) | **nothing** | none | **DEFECT — write-only (F3)** |
| `window_coherence_ceiling` / `window_confidence` | `:740` block, `null` | — | none | ADVISORY-BY-DESIGN (spec `:194` "TB's forward hooks") |
| `current_regime`, `regime_age`, `silhouette`, `switch_rate`, `regime_counts`, `scales`, `regime_entropy_profile`, `current_window_features`, `duality_score`, `window_density_profile`, `n_draws`, `elapsed_seconds`, `timestamp` | `:708-760` | `:503-505` prints `current_regime` only | diagnostic record | ADVISORY-BY-DESIGN |

**Other artifacts:**

| Artifact | Producer | Consumer | Class |
|---|---|---|---|
| `trse_boundary_candidates.json` (18,149 B, 2026-03-06) | `trse_entropy_probe.py:440` | **none found** | ADVISORY-BY-DESIGN |
| `trse_boundary_candidates_wide.json` (2026-03-06) | `trse_entropy_probe.py:453` — second pass at W800/S100, auto-triggered when `candidate_count > 8` (`:448`) | **none found** | ADVISORY-BY-DESIGN |
| `probe_results.json` (2,427 B, 2026-03-07) | `trse_calibration_probe.py:537` | **none** — explicitly "Submit … to TB for review" (`:540`) | ADVISORY-BY-DESIGN |
| `trse_entropy_probe.png` (2026-03-06, **git-tracked**) | `trse_entropy_probe.py:393` | human | ADVISORY-BY-DESIGN |
| `step0_validation_findings.json` | `step0_heuristic_validation.py:288` | **none** | ADVISORY-BY-DESIGN |

The two boundary-candidate files are **not** duplicates (differing md5) despite equal byte
size — they are the narrow and wide passes.

`density_proxy` is additionally imported out of `trse_step0.py` by `w8_correlation_test.py:50-58`
(a manual test harness). `machine_fingerprint_probe.py:94` only mentions TRSE in a printed string.

---

## 4. Rule A / B / C disposition

### Rule A — window ceiling from `regime_type` → **APPLIED, CORRECTLY WIRED**

`window_optimizer_bayesian.py:507-517`:

```python
if (_regime_type == 'short_persistence' and _type_conf >= 0.70 and _regime_stable):
    new_max = max(bounds.min_window_size + 1, min(32, bounds.max_window_size))
    bounds.max_window_size = new_max        # :514
```

**Does it reach Optuna?** Yes — verified hop by hop:

1. `SearchBounds` is a plain **mutable** `@dataclass` (`window_optimizer.py:107-108`, not
   frozen, not a namedtuple) — the in-place write at `:514` is sound, and the code comments so.
2. `bounds` is the `search()` parameter (`window_optimizer_bayesian.py:387`); `optuna_objective`
   closes over it (`:417`).
3. Trials sample `trial.suggest_int('window_size', bounds.min_window_size, bounds.max_window_size)`
   at **`:420-422`**.
4. Rule A runs at `:507-517`, the study is created at `:537+` — **mutation precedes every trial.**

**Live values:** `distributed_config.json` `search_bounds.window_size = {min: 6, max: 50}`;
live context has `regime_type=short_persistence`, `regime_type_confidence=0.8275`,
`regime_stable=true` → gate passes → ceiling **50 → 32**.

**Rule A vs. the S172 `window_size.min = 6` ruling: NO CONFLICT.** They are orthogonal —
the ruling raises `min`, Rule A lowers `max`. Result `[6, 32]`, a strict subset of `[6, 50]`,
and `max(min+1, …)` at `:512` structurally prevents Rule A from ever crossing below `min`.
Neither overwrites the other. `distributed_config.json` was not modified.

The partition-worker path (`n_parallel`) applies the identical rule against its own
`_local_bounds` (`window_optimizer_integration_final.py:1814-1850`) — the S139B patch **was
applied** (see §7).

### Rule B — skip bounds → **ADVISORY-BY-DESIGN (deliberately disabled)**

`window_optimizer_bayesian.py:523-527` logs and applies nothing. This is **not** a dropped
wire. Three independent pieces of evidence:

1. **In-code rationale**, `:491-494`:
   > "Rules B (skip) and C (offset) are logged only — not applied. Shuffle test (S121)
   > confirmed density_proxy measures digit bias, not temporal correlation, so skip/offset
   > advisory fields are unreliable as hard bounds constraints."
2. **The governing ruling**, `docs/SESSION_CHANGELOG_20260307_S122.md:56`:
   > "**Rules B and C (LOGGED ONLY):** Skip bounds and offset bounds disabled per TB + S121
   > shuffle test result (density_proxy measures digit bias, not temporal correlation)."
3. **The producer's own docstring**, `trse_step0.py:487-489`:
   > "TB caution: skip range cannot be reliably inferred from observed draw values alone…
   > This field is **ADVISORY only**."

`TRSE_v1_15_SPEC.md:216-228` does specify Rule B as *applying* bounds — that spec text is
**SUPERSEDED** by the later S121/S122 shuffle-test ruling. The code reflects current intent.

### Rule C — offset prior → **ADVISORY-BY-DESIGN, and inert on live data**

`window_optimizer_bayesian.py:529-533`, same ruling. Additionally the guard never fires:
live `dominant_offset_lag = {dominant_lag: 52, lag_strength: 0.0044, confident: false}`.
`trse_step0.py:541-542` warns "FFT on integer draw values (mod 1000) is fragile."

This is **exactly the graceful degradation the spec predicted** — `TRSE_v1_15_SPEC.md:326-328`:
> "If items 2-3 fail validation on real data (e.g. dominant_lag comes back as 17 instead of
> 43), those fields will be set `confident=False` and Step 1 will ignore them — **which is the
> correct fallback behavior.**"

Spec validation criterion 3 (`:316`, `dominant_lag` in [35,55]) is met at 52; criterion 1
(`:314`, `short_persistence`) is met. Criterion 2 (`:315`) is **not** — see F4.

---

## 5. Dead-dimension inventory

| # | Dimension | Dies at | Class |
|---|---|---|---|
| D1 | `confirmed_windows` | written `window_optimizer.py:792-794`; **zero readers** — `grep -c confirmed_windows trse_step0.py` = **0** | **DEFECT** (F3) |
| D2 | `recommended_window_size` | CLI `trse_step0.py:814` → context `:740` → `window_optimizer_bayesian.py:500` `_rec_ws` → **never referenced again** | **DEFECT** (F6) |
| D3 | `--window-size`, `--stride` in `agent_manifests/trse.json` `default_params` | rejected by `trse_step0.py:807-817` argparse | **DEFECT** (F1) |
| D4 | `skip_entropy_profile`, `dominant_offset_lag`, `w3_w8_ratio`, diagnostics | stop at stdout / the JSON record | **ADVISORY-BY-DESIGN** |
| D5 | `window_coherence_ceiling`, `window_confidence` | emitted `null`, never populated | **ADVISORY-BY-DESIGN** — spec `:194` names them TB forward hooks |
| D6 | probe artifacts (`trse_boundary_candidates*.json`, `probe_results.json`, `.png`, `step0_validation_findings.json`) | human / TB submission | **ADVISORY-BY-DESIGN** |

**No TRSE value is declared as an agent-tunable parameter.** `grep -rn tunable agents/*.py`
returns nothing, and no TRSE key appears in the `parameter_application` surface. The §0.5
"declared-but-disconnected knob" failure mode is **absent** here — with the narrow exception of
D2, which is manifest-declared and consumer-dropped.

---

## 6. Spec-vs-code conflicts

| # | Spec says | Code does | Which reflects intent |
|---|---|---|---|
| S1 | Rules B & C apply bounds (`TRSE_v1_15_SPEC.md:216-240`) | logged only (`:523-533`) | **CODE.** Spec **SUPERSEDED** by S121 shuffle test / TB ruling (`S122.md:56`) |
| S2 | Rule A ceiling = `rec_ws * 4` (`TRSE_INTEGRATION_PLAN_S121.md:94`) | hardcoded `min(32, …)` (`:513`) | **CODE.** Spec §4 (`:210`) already revised to a flat 32; the plan text is **STALE**. Side effect: makes D2 dead |
| S3 | Step 0 is "standalone, optional, passive" (`SPEC:280`) | registered as a WATCHER step | **BOTH** — the plan (`:26-45`) supersedes that line; the manifest keeps it optional via `skip_on_fail` |
| S4 | manifest `default_params` include `window_size`, `stride` (`PLAN:63-72`) | argparse accepts neither | **NEITHER** — v1.15 replaced single-scale with fixed W200/W400/W800 (`SPEC:154-158`), so the params are **STALE** leftovers of v1.1. The manifest was never updated to match. **This is F1** |
| S5 | `consistent_with_known_skip` distinguishes real skip structure (`SPEC:92-96`) | near-vacuous overlap test (`:519-521`) | **SPEC** — the check does not implement the described discrimination. **F4** |
| S6 | "TRSE uses these on subsequent runs" (`apply_s136_doc_updates.py:320-323`) | TRSE never reads `confirmed_windows` | **CODE** — the doc string is **CONTRADICTED-BY-CODE**. **F3** |

---

## 7. Patch-script application status

Per the brief's warning, each was checked against live source rather than assumed.

| Script | Applied? | Evidence |
|---|---|---|
| `apply_s139_window_max_50.py` | **APPLIED** | `distributed_config.json` `window_size.max = 50` |
| `apply_s139b_trse_partition_fix.py` | **APPLIED** | `window_optimizer_integration_final.py:1814` carries the `S139B` marker and the full Rule A block |
| `apply_s140b_trial_history.py` | **APPLIED** | `trse_context_file` threading present at `window_optimizer.py:390`, `:469`, `:538`, `:662`; `window_optimizer_bayesian.py:392` |
| `apply_s142b_np2_terminal.py` | **APPLIED** | `window_optimizer_integration_final.py:1692` carries the `S123 TRSE thread` signature |
| `fix_step1_timeout.py` | **CONSISTENT** | Step 0 timeout is 1 min at `agents/watcher_agent.py:2813`; script states "Step 0 timeout: 1 min (TRSE — unchanged)" (`:135`) |
| `apply_s136_doc_updates.py` | **DOC-ONLY** | its `confirmed_windows` claim (`:320-323`) is contradicted — F3 |

`docs/window_optimizer_integration_final.py` and the `*.s17*_bak` / `*.s170_*_bak` files are
**stale duplicates**, excluded from all verdicts. Their retention is a standing ruling
(2026-07-31); no change is proposed.

---

## 8. Prioritised findings

### F1 — CRITICAL · DEFECT · WATCHER's Step 0 command is malformed and cannot run

`agent_manifests/trse.json` `default_params` declares `window_size`, `stride` and
`trse_context`. WATCHER emits **every** `default_params` key as `--key value`
(`agents/watcher_agent.py:1496-1512`; `trse.json` has no `args_map`, so the
underscore→hyphen fallback at `:1499` applies). `trse_step0.py:807-817` uses strict
`parse_args` and defines none of the three.

**Execution proof (this session, VM 101, run from an empty scratchpad CWD so the repo could
not be written):**

```
$ python3 trse_step0.py --lottery-data daily3.json --output trse_context.json \
    --window-size 400 --stride 50 --k-clusters 5 --recommended-window-size 8 \
    --trse-context trse_context.json
trse_step0.py: error: unrecognized arguments: --window-size 400 --stride 50 --trse-context trse_context.json
EXIT_CODE=2
```

Nothing was created; the repo was untouched.

**Why it is invisible:** `trse.json` sets `"skip_on_fail": true`, so a Step-0 failure is
recorded as `action: proceed`. `watcher_decisions.jsonl` holds **17** such entries
(2026-03-09 01:02 → 2026-03-10 01:19), all reading *"Step 0 output absent — downstream step
runs with defaults"*. Step 0 fails **silently by design of the skip policy**.

**Why it is currently dormant:** WATCHER's freshness check short-circuits Step 0 before the
command is ever built (`agents/watcher_agent.py:1347`, `:1357-1364`) — see
`logs/pa_step0_step1.log:22`: *"Step 0: Fresh … skipping (output is fresh)"*. The defect fires
**only when `trse_context.json` is older than `daily3.json`** — i.e. precisely when new draws
have arrived and Step 0 would do useful work.

**Note:** a fresh clone has **no** `agent_manifests/trse.json` (F5) and would therefore invoke
`python3 trse_step0.py` with no params — which **succeeds**. The manifest is the fault.

### F2 — HIGH · DEFECT · Self-perpetuating staleness lock on `trse_context.json`

Two mechanisms combine:

1. **Step 1 writes back into Step 0's output.** `window_optimizer.py:793-794` reopens
   `trse_context.json` in `"w"` and rewrites it to append a `confirmed_windows` entry. This
   **bumps the mtime**.
2. **That mtime is the freshness sentinel.** `check_output_freshness` (`agents/watcher_agent.py:495-524`)
   compares `primary_output` mtime against `required_inputs` mtime — from the manifest, exactly
   `trse_context.json` vs `daily3.json`.

So every Step 1 run refreshes the sentinel for a Step 0 that never re-ran.

**Live proof:**

| Quantity | Value |
|---|---|
| `trse_context.json` internal `timestamp` (real compute time) | **2026-03-13T12:10:27Z** |
| `trse_context.json` file mtime | **2026-05-01 16:05:11** |
| last `confirmed_windows[-1].timestamp` | **2026-05-01T16:05:11.723533** (matches mtime to the microsecond) |
| `daily3.json` mtime | 2026-03-04 16:58 |

The mtime is owned by Step 1's write-back, not by Step 0. The regime analysis in force is
**~4.5 months old**.

**Compounding:** `_load_trse_context` (`window_optimizer_bayesian.py:25-47`) has a **version
guard but no staleness guard**. A stale context is applied unconditionally. The failure
sequence after new draws land is therefore: Step 0 goes stale → executes → **F1** exits 2 →
`skip_on_fail` proceeds → Step 1 loads the **old** context and applies Rule A from it → Step 1
writes back → mtime now exceeds the new `daily3.json` → Step 0 reports "fresh" forever after.

**Consequence:** Rule A would narrow the window ceiling on the basis of a regime that may no
longer hold, with no operator-visible signal. Note the direction of risk is bounded — Rule A
only narrows `max` to 32 and cannot violate `min` — but the input is unverified.

### F3 — MEDIUM · DEFECT · `confirmed_windows` is write-only, and destroyed on re-run

Written at `window_optimizer.py:789-794` (capped to the last 50, `:792`). **Zero readers** —
`grep -c "confirmed_windows" trse_step0.py` = **0**, and the only other repo hits are the
writer, its stale backups, and documentation.

`apply_s136_doc_updates.py:320-323` documents a feedback loop that does not exist:
> "`trse_context.json` as `confirmed_windows`. TRSE uses these on subsequent runs"

**CONTRADICTED-BY-CODE.** The stated purpose at `window_optimizer.py:770` — "Builds a
regime→window lookup table over multiple runs" — is unrealised.

**Second-order:** `save_context` (`trse_step0.py:795-800`, called `:868`) writes the freshly
built context dict **without merging** the existing file. A successful Step 0 run therefore
**erases the 20 accumulated entries** (currently spanning 2026-03-14 → 2026-05-01). Latent
only because F1/F2 prevent Step 0 from running.

### F4 — MEDIUM · LATENT HAZARD · `consistent_with_known_skip` is near-vacuous

`trse_step0.py:519-521`:

```python
overlap = (p5 <= known_hi) and (p95 >= known_lo)     # p5 <= 56 and p95 >= 5
```

This is an **overlap** test between the 5th/95th percentile band and `[5, 56]`, not a
consistency test. It is true for almost any real distribution.

**Live values:** `gap_range_min = 27`, `gap_range_max = 773`, against
`known_skip_range = [5, 56]` → **`consistent_with_known_skip: true`**. A measured band
14× wider than the known range is reported as consistent. `draw_gap_mean` is 334.7 — the spec's
worked example expects ~31.2 (`SPEC:73`).

This **fails spec validation criterion 2** (`SPEC:315`: `gap_range_min≈5, gap_range_max≈56`),
and the docstring's own expectation (`:489`: "consistent_with_known_skip=False is expected and
acceptable") is contradicted by the implementation returning `true`.

**No current consequence** — Rule B is disabled, so nothing consumes it. **The hazard is
forward-looking:** if Rule B is ever enabled, `SPEC:219-227` would clamp
`skip_min ∈ [0,15]`, `skip_max ∈ [46,66]` on this vacuous signal. That matters now because
**hybrid skip bounds are the next approved task** (skill §2.7 #4, §8) — anyone re-enabling
Rule B while wiring skip bounds would import a false-positive gate. Flagging, not fixing.

### F5 — LOW · VIR-6 · `agent_manifests/trse.json` is untracked and invisible to git

`git check-ignore -v` → `.gitignore:41:*.json`. The other **6** manifests are force-added and
tracked; `trse.json` is **not** (`git ls-files agent_manifests/` returns 6 entries, without it).

Consequences: no history for the manifest that causes F1; no repo gate can see it; a fresh
clone has no Step 0 manifest at all, so `get_step_io_from_manifest(step=0)` would raise and
`check_output_freshness` would return a **HARD** failure (`agents/watcher_agent.py:503-505`).
The live host and the repository disagree about Step 0's configuration.

### F6 — LOW · DEFECT · `recommended_window_size` is loaded then dropped

An operator-settable knob (`trse_step0.py:814`, manifest `default_params`) that is **echoed,
not computed** — `trse_step0.py:740` writes back the CLI input verbatim. Step 1 loads it into
`_rec_ws` (`window_optimizer_bayesian.py:500`) and **never references it again**; Rule A uses a
hardcoded `32` (`:513`) instead of the planned `rec_ws * 4` (`PLAN:94`).

This is a small, low-consequence instance of the §0.5 pattern: a declared parameter that
survives argparse, manifest, artifact and consumer-load, then dies one statement before use.
Per §0.4 the correct disposition is **wire-in or explicit retirement by ruling — not silent
removal**.

---

## 9. VIR-2 clean control — outputs verified CORRECTLY WIRED

Stated explicitly, as required. Each was traced producer → artifact → consumer this session:

1. **Rule A gate inputs** — `regime_type`, `regime_type_confidence`, `regime_stable`:
   produced `trse_step0.py:397`/`:740`, read `window_optimizer_bayesian.py:497-499`, all three
   evaluated in the live gate `:508-510`. ✅
2. **Rule A → Optuna search space** — `bounds.max_window_size` mutated `:514`, consumed by
   `trial.suggest_int('window_size', …)` `:420-422`, ordering verified (mutation `:514` precedes
   study creation `:537`), mutability verified (`window_optimizer.py:107-108`, unfrozen
   dataclass). ✅ **This is the one applied output and it works.**
3. **Rule A in the NP2 partition path** — independently re-applied against `_local_bounds`,
   `window_optimizer_integration_final.py:1814-1850`. ✅
4. **Version guard** — `trse_version` written `:708`, enforced `window_optimizer_bayesian.py:39-43`,
   live value `1.15.1` ≥ 1.15. ✅
5. **Passive-absence fallback** — `_load_trse_context` returns `None` on missing/invalid
   (`:30-35`, `:45-47`); Step 1 prints and proceeds with defaults (`:534-535`). Matches
   `SPEC:286`. ✅
6. **`--trse-context` CLI** — `window_optimizer.py:1084-1087`, threaded `:1176` → `:538` →
   `:662` → `window_optimizer_bayesian.py:392` → `:495`. Four hops, all present. ✅
7. **`trse_context` in the Step-1 manifest** — `agent_manifests/window_optimizer.json:250`. ✅
8. **Agent-layer Step-0 registration** — all 9 surfaces in §2.1 present and consistent. ✅
9. **Rule A vs. S172 `window_size.min = 6`** — verified non-conflicting; `max(min+1, …)` at
   `:512` makes inversion structurally impossible. ✅
10. **`skip_on_fail` semantics** — declared `agent_manifests/trse.json`, honoured
    `agents/watcher_agent.py:659`, `:670`. Behaves as specified (it is the *masking* of F1 that
    is the problem, not the mechanism). ✅

**Fault-injection control:** **n/a — this is a read-only audit**, stated rather than omitted
per the brief §6. The one positive control available without mutating state was the F1
argparse reproduction (§8), which is a genuine executed failure, not an inferred one.

---

## 10. Coverage table

| Brief §3 item | Status | Where |
|---|---|---|
| 1 What TRSE is | **COMPLETE** | §1 |
| 2 Invocation model | **COMPLETE** | §2 |
| 3 Output → consumer trace | **COMPLETE** | §3 |
| 4 Rule A / bounds / S172 agreement | **COMPLETE** | §4, §9.9 |
| 5 Rules B and C | **COMPLETE** | §4 |
| 6 JSON artifacts + mtimes | **COMPLETE** | §3, F2 |
| 7 Agent-layer integration | **COMPLETE** | §2.1, §5 (no tunable declared) |
| 8 Dead dimensions, classified | **COMPLETE** | §5 |
| 9 Spec vs. implementation | **COMPLETE** | §6 |
| Patch-script application status | **COMPLETE** | §7 |
| VIR-2 clean control | **COMPLETE** | §9 |

### Searched surfaces (VIR-6)

VM 101 working tree at `9933ba2`: all `*.py` (`/bin/grep`, not the ugrep wrapper, so
gitignored `*.json` **were** searched); `agent_manifests/`; `distributed_config.json`;
live untracked artifacts (`trse_context.json`, `trse_boundary_candidates*.json`,
`probe_results.json`, `trse_entropy_probe.png`, `daily3.json`) incl. mtimes and content;
`watcher_decisions.jsonl` (694 records); `watcher_history.json`; `logs/`; `docs/`;
`git ls-files` / `git check-ignore`; `crontab -l`; `systemctl list-unit-files`;
`--include=*.sh/*.service/*.timer`. One controlled execution (`trse_step0.py` argparse, F1).

### Unavailable / not examined (VIR-5, VIR-6)

- **The rigs (`.122/.156/.164`) and the Proxmox hosts were not inspected.** TRSE is CPU-only
  and runs on VM 101, so this is not believed material — but it is **UNAVAILABLE**, not
  assumed clean.
- **Runtime behaviour of Rule A was not executed.** Rule A's wiring is proven by static trace
  plus dataclass-mutability verification; no Optuna study was run (running Step 1 is out of
  scope). Classified **verified-by-trace**, not verified-by-execution.
- **`agent_manifests/trse.json` has no git history** (untracked) — when `window_size`/`stride`
  entered `default_params` cannot be dated. The 2026-03-09/10 `skip_on_fail` burst is
  *consistent with* F1 but not proven to be F1; the manifest mtime (2026-03-07) precedes it.
- **Whether Step 0 was ever invoked after 2026-03-15** could not be established beyond the
  searched logs and decision journal; older logs may have been rotated.
- No claim is made about uncommitted state on any host other than VM 101.

---

## 11. Completion sentinel

```
VERIFICATION-INTEGRITY CONTROLS (VIR-1…6)
- execution proof:            F1 reproduced live (exit 2); all verdicts carry file:line read this session
- clean control:              §9 — 10 outputs verified correctly wired, incl. Rule A → Optuna
- fault-injection control:    n/a — read-only audit (declared, not omitted)
- completion sentinel:        below
- unavailable-observer:       §10 — 5 surfaces declared UNAVAILABLE, none assumed correct
- audit claim scope:          VM 101 repo + live host artifacts only
- searched surfaces:          §10
- unavailable surfaces:       §10

RESULT: FAIL
```

**FAIL** — the audit itself completed with full coverage (no item `INCOMPLETE`), but the
audited subject does not pass: **F1 (Step 0 cannot execute under WATCHER)** and
**F2 (staleness lock; a 2026-03-13 context still in force)** are live defects at the head of
the pipeline.

**Assessment against the brief's framing:** TRSE is *substantially* fine, as expected. Rules
B and C are **ADVISORY-BY-DESIGN** under an explicit TB ruling and must **not** be repaired.
The §2.7 "tuned parameter never reaches the kernel" pattern is **not** present in TRSE's
analytical path; the only instance is **F6**, minor and cosmetic in effect. The real defects
are launch-path and artifact-lifecycle faults, and they are worth a repair brief:
F1 and F2 together mean **Step 0 has not meaningfully run since 2026-03-15**, while Step 1
continues to apply Rule A from its output every run.

Per §0.4, no removal or simplification is proposed anywhere in this report.

---

**AUDIT COMPLETE — STOPPING FOR TEAM ALPHA REVIEW.**
No code, configuration, or documentation was modified. Nothing was committed or pushed.

*Team Alpha (Claude Code, VM 101) — TRSE / Step 0 audit v1, base `9933ba2`, 2026-07-31*
