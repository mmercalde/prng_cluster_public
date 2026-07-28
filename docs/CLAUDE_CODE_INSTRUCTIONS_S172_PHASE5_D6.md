# CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D6.md — REV1

**S172 RANGE-MINER — Phase 5, Deliverable D6: the production integration adapter,
plus the Zeus single-GPU certified-generation smoke gate.**

**Frozen against HEAD `2a6e0f8`** (D5 closed, dual-pushed). D6 is the first
deliverable that touches **real silicon and real Step-1/Step-2 wiring**;
D0–D5 were contract-and-equivalence work on synthetic fixtures. Authority:
`docs/PROPOSAL_S172_RANGE_MINER_v1_4_5.md`, the D5 chain, and this brief.

**Audience:** Claude Code on VM 101 (`michael@192.168.3.177`),
`~/distributed_prng_analysis`, as `michael`. You implement and iterate; you do
NOT commit, push, or run WATCHER. STOP at the gate for Team Alpha review.

---

## 0. Read this first — most of the wiring already exists

Verified at `2a6e0f8`. **D6 is not wiring from scratch.** The Phase-4 seam is
already in `window_optimizer_integration_final.py`:

- `run_bidirectional_test(...)` has a **`use_range_miner=True` branch** (`:461`)
  that calls `run_trial_miner(...)`, driving the real coordinator's `serve_trial`
  with the full production parameter set (caps, hybrid caps, stripe/substripe,
  staging high-water, bind `0.0.0.0:5700`, window params by direct attribute
  access). That path runs today.
- `_build_test_result_from_miner(...)` (`:388`) builds the Step-1 `TestResult`
  for the miner path, deliberately **detached** from `_build_test_result_from_pw`
  (routing miner output through the PWC/ZMQ D3.25 ingress wall is forbidden — the
  miner already emits canonical 24-field records inside Phase-5 assembly).

**What that path does NOT do yet — this is exactly D6's scope, stated by the
author in the `_build_test_result_from_miner` docstring:**

- It **appends no candidates**: "No candidate is appended: D6 owns miner
  candidate ingress." `serve_trial` returns run/stripe/manifest state and none of
  the population keys, so today every count reads `+0` and nothing is
  accumulated. This is preserved verbatim (including flush cadence) until D6.
- Certification status is explicit: PWC/ZMQ both-mode candidate output was
  **certified at D3.25**; **miner both-mode run-level candidate output is
  `uncertified until D6`.**

So D6 closes exactly one gap: **turn the running-but-inert miner trial into a
certified accumulator generation, and feed its real candidates back into the
Step-1 accumulator + `TestResult` shape** — without rerunning normalization and
without touching the PWC/ZMQ ingress.

---

## 1. The stored artifacts D6 adapts (do not rebuild any of them)

- **`MinerTrialAssembly`** (`range_miner_npz_writer.py:211`) — its docstring is
  the contract: "Stable across D1–D6 … **D6 adapts it back into the legacy
  Step-1 accumulator shape.**" D6 appends off `canonical_records_constant` /
  `canonical_records_variable` (both `list[dict]`, ascending seed) **straight off
  the stored assembly, WITHOUT rerunning normalization** (D3.25 REV3 §4). The
  four maps, `bidirectional_constant/_variable`, and `directional_counts` are
  already populated.
- **`finalize_run(...) -> RunArtifactResult`** (`utils/run_finalizer.py:345`) —
  produces the immutable certified generation: `generation_dir`, `all_npz_path`,
  `binary_npz_path` (the finalizer's own paths — note it populates **neither**
  `MinerTrialAssembly.binary_npz_path` nor `all_npz_path`; both stay permanently
  `None` per Beta Ruling E). The certified prior is a 22-array NPZ bundle.
- **Selected assembly backend** — `get_assembly_backend(name)`
  (`miner/assembly_backends.py`), **default `serial_reference`**;
  `process_sharded` selectable but unpromoted (Phase 6 owns promotion).

**Fail-closed is already designed in:** `MinerTrialAssembly` "fails closed on
`None` where it needs a path"; every assembly failure is a producer/contract
defect that fails closed. D6 must preserve this — where publication results are
required and absent, D6 raises, never fabricates a zero/empty result.

---

## 2. D6 scope — the production path, end to end

Wire the full lifecycle behind `use_range_miner=True`, default backend
`serial_reference`:

```
real RANGE-MINER trial (serve_trial, real coordinator)
  → coordinator / staged spool lifecycle (already runs)
  → selected assembly backend  → MinerTrialAssembly
  → shared run finalizer (finalize_run)  → immutable certified generation
  → adapt canonical_records_* back into the Step-1 accumulator
  → existing Step-1 TestResult / optimizer return shape
```

Concretely:

1. **Candidate ingress (the core gap).** After the miner trial produces its
   stored `MinerTrialAssembly`, append its real candidates into the accumulator
   from `canonical_records_constant` / `_variable` directly — no re-normalization,
   no PWC/ZMQ ingress. Advance `forward_count` / `reverse_count` /
   `bidirectional_count` from the assembly's real populations, replacing today's
   `+0`. Preserve the existing threshold-gated flush cadence exactly.
2. **Finalization.** Drive the stored assembly through `finalize_run` to produce
   the certified generation; carry the finalizer's `RunArtifactResult` paths.
   `MinerTrialAssembly.binary_npz_path` / `all_npz_path` stay `None` (Ruling E) —
   the certified paths come from `RunArtifactResult`, never from the assembly.
3. **Step-1 return shape unchanged.** `_build_test_result_from_miner` returns the
   same `TestResult` fields Step 1 already consumes; only the counts become real.
   Do not alter the `TestResult` contract or the PWC/ZMQ builders.
4. **Backend selection** threads through as a config/attribute
   (`assembly_backend`, default `serial_reference`), resolved via
   `get_assembly_backend`. `process_sharded` selectable, never default.

**Out of scope for D6:** multi-node infrastructure, the four-path Phase-6
comparison, throughput promotion, the `rrig6600` Proxmox migration. D6 is
single-node production wiring + one real-GPU smoke.

---

## 3. The D6 gate — including the Zeus single-GPU real-silicon smoke

Two tiers.

### 3.A Adapter gates (fixtures / mocked coordinator, no GPU)

- **G-INGRESS:** a stored `MinerTrialAssembly` with known `canonical_records_*`
  appends exactly those candidates into the accumulator, with real forward /
  reverse / bidirectional counts — and the appended records are **identical** to
  the assembly's (no re-normalization, byte-for-byte record equality).
- **G-NO-PWZ-INGRESS:** the miner path never calls the PWC/ZMQ D3.25 ingress /
  four-map normalizer (AST + runtime).
- **G-FINALIZE:** the stored assembly drives `finalize_run` to a certified
  generation; `RunArtifactResult` paths exist and point at the 22-array bundle;
  `MinerTrialAssembly.binary_npz_path` / `all_npz_path` remain `None`.
- **G-FAILCLOSED:** when a required publication result is absent (assembly `None`,
  or a required path `None`), D6 **raises** — it never returns a zero/empty
  `TestResult`. Assert the specific fail-closed exception, not a silent `+0`.
- **G-TESTRESULT:** the returned `TestResult` matches the Step-1 contract fields;
  the PWC/ZMQ builders are untouched (AST).
- **G-BACKEND-DEFAULT:** with no backend specified, `serial_reference` is
  selected; `process_sharded` is reachable only by explicit selection.
- **G-FLUSH-CADENCE:** the threshold-gated flush fires on the same cadence as the
  pre-D6 path (the docstring's "flush cadence does not shift" invariant).

### 3.B Zeus single-GPU certified-generation smoke (real silicon) — the headline

A **Zeus-only, single RTX 3080 Ti** real trial, needing **none** of the
multi-node work. It must prove the entire path on real hardware:

```
CUDA sieve (real 3080 Ti, java_lcg)
  → sub-stripe result / spool
  → coordinator publication
  → Phase-5 assembly (serial_reference)
  → 22-array validation
  → certified generation (finalize_run)
  → Step-2 loader successfully reads it back
```

Requirements:
- runs on VM 101's passed-through 3080 Ti (`nvidia-smi` sees 12 GB); venv
  `~/venvs/torch`; `--optimizer-python ~/distributed_prng_analysis/python3_with_venv.sh`.
- small but real seed window (real CUDA sieve, not a mock), `java_lcg`,
  `serial_reference` backend.
- **acceptance:** a certified generation directory is produced, its 22-array
  bundle passes validation, and the **Step-2 loader reads it back successfully**
  — this is the **first certified accumulator generation on real silicon**.
- capture: `nvidia-smi` at run, the generation dir + sidecar, the 22-array
  validation result, and the Step-2 load confirmation.

This gate is a real-hardware smoke, not a throughput benchmark — no s/s target
here (that's Phase 6). It proves the path *runs and certifies*, once, on a GPU.

---

## 4. Non-negotiable rules

1. Read live source before every claim — this is a production seam; the miner
   branch, `_build_test_result_from_miner`, `finalize_run`, and the
   `MinerTrialAssembly` contract are all already written and must be reused, not
   reimplemented.
2. Do not touch the PWC/ZMQ ingress, the D3.25 four-map contract, or the
   `TestResult` shape. Coexistence with PWC/ZMQ must hold (the miner gate is
   behind `use_range_miner` and touches neither).
3. `serial_reference` is the default backend; `process_sharded` selectable,
   unpromoted.
4. Fail closed on absent publication results — never a fabricated zero result.
5. Every gate must FAIL on wrong behavior; oracles hand-transcribed.
6. No `git commit`/`push`/`watcher --run-pipeline` from the sandbox.

## 5. Non-regression

Capture green at `2a6e0f8` before any edit: D5 24/24 (18 mutants), D4 8/8, D1.1
18/18, D2 7/7, D3 10/10, D3.0 10/10, D3.25 13/13, D3.5 60/60, Phase 4 63/63,
Phase 3 17/17. After D6: all still green, plus the D6 adapter gates (3.A) and the
Zeus single-GPU smoke (3.B). The PWC/ZMQ paths must remain byte-for-byte
unaffected (their builders untouched).

## 6. Report

Per the D5 house style: the adapter diff + gate results (3.A), then the **Zeus
single-GPU smoke evidence (3.B)** — `nvidia-smi`, the certified generation dir +
sidecar, 22-array validation, and the Step-2 load-back confirmation, i.e. the
first certified generation on real silicon. Confirm `serial_reference` default,
PWC/ZMQ untouched, and fail-closed behavior. Then STOP for Team Alpha review.

---

## Appendix — Phase 6 / Phase 7 prerequisites (NOT D6 work; tracked so they run in parallel)

Per Beta's runway, these are prepared **alongside** D6 but are not gated by it:

**Phase 6 infra (before the four-path verify):**
1. second RTX 3080 Ti passed through to VM 101;
2. passwordless non-interactive SSH `michael` → CT100 workers;
3. VM 101 given a stable IP / DHCP reservation;
4. remaining items in `PHASE6_PREREQS.md`.
The `rrig6600` Proxmox migration is **not** needed for the D6 single-GPU smoke,
but **is** required before the Phase-7 26-GPU saturation soak.

**Phase 6 acceptance (after D6):** identical low- and high-survivor workloads
through all four paths (PWC · ZMQ+SQLite · miner+serial_reference ·
miner+process_sharded); all 22 arrays `np.array_equal`; NPZ→dict sorted-by-seed
match; PWC authoritative; disagreements localized, never voted away. High-survivor
miner gate: median of ≥3 warmed runs, ≥500k seeds/s end-to-end, ≥25% of
low-survivor throughput on identical hardware, no OOM/swap/queue-growth/abandoned-
spool/accounting-mismatch, with separate timings for GPU / transfer / SHA /
parse-columnize / merge / final-write / validation / total.

**Promotion:** `process_sharded` becomes default only if the real high-survivor
Phase-6 trial shows ALL of: ≥20% median end-to-end improvement, identical final
arrays, ≤50% additional peak host RSS, zero swap. The D5 synthetic benchmark
(~1.6× faster at ~2–3× RAM) already suggests it fails the RAM criterion, so the
expected outcome is `serial_reference` stays default / `process_sharded` stays
available for tuning — but Phase 6 must measure the real workload before locking
that.

**Phase 7 (after Phase 6):** 50-trial WATCHER soak, ≥5 high-survivor + ≥5
low-survivor controls, constant + hybrid modes, per-trial cleanup verification,
no monotonic RSS / spool backlog / temp-file growth.
