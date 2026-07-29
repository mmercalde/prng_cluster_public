# SESSION CHANGELOG — 2026-07-27 — S172 Phase 5 D6 (production integration adapter + Zeus single-GPU smoke)

**Status:** IMPLEMENTED AND GATED ON VM 101, **not committed, not pushed** —
stopped at the gate for Team Alpha review per the D6 brief. Claude does not
commit, push, or run WATCHER.

**Base:** HEAD `a22216c`; code byte-identical to the D6 freeze point `2a6e0f8`
(`ed487d9` and `a22216c` are docs-only). Verified before any edit:
`git diff --name-only 2a6e0f8 a22216c` returns `docs/` paths only.

**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D6.md` (REV1).

---

## What D6 delivers

D6 closes exactly one gap, the one the pre-D6 code named in its own docstring:
**the miner path ran but appended nothing and returned `+0` for every count.**

Two facts from live source made the shape of the fix:

1. `_build_test_result_from_miner` read `serve_trial`'s dict with
   `.get(..., set()/{})`, and `serve_trial` returns run/stripe/manifest state
   and **no population keys** — so every count was structurally zero.
2. The `_use_miner` gate passed **no `phase5_sink`** to `run_trial_miner`, so
   the coordinator's L6 boundary was wired to `None` and the trial performed
   **no Phase-5 assembly at all**. There was nothing to ingest even in
   principle.

D6 wires both: the gate now builds an `AssemblingPhase5Sink` around the
configured assembly backend and passes it in; the builder fetches the **stored**
`MinerTrialAssembly` by `run_id` and appends
`canonical_records_constant` / `_variable` **as they stand** — no
re-normalization, no PWC/ZMQ ingress — advancing the real directional counts.

**The finalizer is NOT called a second time.** `finalize_run` is already wired
at RUN level over the accumulator (`window_optimizer_integration_final.py:1812`),
and the D3.5 author's comment at `:1710` states the shared finalizer is what
"every backend (legacy in-process sieve, PWC, ZMQ and — via D6 — the range
miner) now goes through." Feeding the accumulator IS how the miner reaches a
certified generation; adding a per-trial `finalize_run` would mint one
generation per trial and break the run-level provenance chain. The brief's §2
pipeline diagram lists finalization before accumulator adaptation; live code has
it after, once per run. **Live code wins, per rule §4.1.** Flagged for Alpha.

---

## Files changed

| File | Δ | What |
|---|---|---|
| `window_optimizer_integration_final.py` | +122/−~30 | the `use_range_miner` gate builds+passes the sink; `_build_test_result_from_miner` ingests the stored assembly |
| `miner/step1_ingress.py` | **NEW**, 288 | the D6 adapter: backend resolution, sink construction, fail-closed assembly fetch, accumulator ingress, certified-path extraction |
| `miner/range_miner_npz_writer.py` | +37/−2 | `AssemblingPhase5Sink(backend=None)` + a `_assemble` seam; `None` keeps pre-D6 behaviour verbatim |
| `tests/test_s172_phase5_d6_production_adapter.py` | **NEW**, 1304 | the 3.A gate harness — 9 checks, 16 mutants |
| `tests/smoke_s172_phase5_d6_zeus_single_gpu.py` | **NEW**, 400 | the 3.B real-silicon smoke |
| `tests/test_s172_phase5_d3_25_candidate_ingress.py` | +233/−87 | G13 updated from its interim contract to the D6 one (see below) |
| `tests/test_s172_phase4_coordinator.py` | +43 | gate-22 coexistence whitelist, D6 entry |

**Untouched, and asserted so:** `_build_test_result_from_pw`, the PWC and ZMQ
gates in `run_bidirectional_test`, and `_flush_npz_incremental` are **byte-identical
to `2a6e0f8`** — D6's G-TESTRESULT and G-FLUSH-CADENCE diff them against
`git show 2a6e0f8:…` and a mutant that edits either one reds.

---

## The one pre-existing gate D6 had to update: D3.25 G13

G13 asserted the miner path appends nothing, returns an all-zero `TestResult`,
and "consumed canonical records that only D6 may append." That was D3.25's own
stated **interim** state — its certification note reads *"miner both-mode
run-level candidate output **uncertified until D6**."* D6 certifies it, so those
three assertions are updated. Everything G13 exists to guard is unchanged and
now stronger:

- **isolation** — the miner path still never reaches the four-map normalizer,
  the v2 ingress wall, or the PWC builder (AST + runtime);
- **flush cadence** — still exactly one `_flush_npz_incremental` per invocation,
  same label, same accumulator, none for a `None` accumulator;
- **never a fabricated zero** — the same bare `serve_trial` dict G13 has always
  used now **raises** instead of returning zeros, so the zero deltas it asserted
  still hold, by refusal rather than by inertness.

Its two bite proofs became three: revert-to-inert, drop-the-flush, and
fabricate-a-zero. **D3.25: 13/13, 17 mutants.**

---

## 3.A adapter gates — 9/9, 16 mutants killed

`python tests/test_s172_phase5_d6_production_adapter.py`

| Gate | Result |
|---|---|
| G-INGRESS | ✅ assembly records → accumulator, object-identical, real counts |
| G-INGRESS/builder | ✅ same through the production `_build_test_result_from_miner` |
| G-NO-PWZ-INGRESS | ✅ AST (no import/reference) + runtime tripwire over the D3.25 ingress |
| G-FINALIZE | ✅ certified 22-array generation; Ruling E holds (both assembly NPZ paths still `None`) |
| G-FAILCLOSED | ✅ 7 absent-result shapes raise; accumulator untouched every time |
| G-TESTRESULT | ✅ Step-1 contract fields; PWC builder byte-identical to `2a6e0f8` |
| G-BACKEND-DEFAULT | ✅ `serial_reference` default; `process_sharded` explicit-only; backend reaches assembly |
| G-FLUSH-CADENCE | ✅ one flush/trial, same label, threshold rule unmodified |

Fixtures drive the **real** producer surface (coordinator → ledger → staging →
sink → commit), oracles are hand-transcribed literals, and mutants are swapped
into the **production** module namespace so every kill is attributable.

Mutants: M1 drop-variable-records · M2 swap-mode-order · M3 re-normalize ·
M4 count-drops-variable · M5 missing-count-defaults-zero · M6
absent-result-fabricates-empty · M7 tolerate-missing-path · M8
process_sharded-as-default · M9 sink-discards-backend · M10
writer-ignores-backend-seam · M11 adapter-imports-normalizer · M12
builder-calls-v2-wall · M13 PWC-builder-edited · M14 flush-twice · M15
threshold-gate-removed · M16 certified-paths-off-wrong-object.

---

## 3.B Zeus single-GPU certified-generation smoke — ACCEPTANCE REACHED

`python tests/smoke_s172_phase5_d6_zeus_single_gpu.py`
(venv `~/venvs/torch`, VM 101, RTX 3080 Ti 12288 MiB)

**Configuration:** `java_lcg`, `test_both_modes=False` (phases 1+2, constant
skip), seeds `[0, 8,000,000)`, stripe 4,000,000, sub-stripe cap 1,000,000,
window 3, threshold 0.25, backend `serial_reference` (unconfigured → default).

**Chain, all real:** CUDA sieve in a separate `miner/range_miner_worker.py`
process on device 0 (cupy + the production `sieve_gpu_worker` kernels) →
**16 staged sub-stripe spools, 30 MB** → coordinator staging/verification and
publication → Phase-5 assembly (`serial_reference`) → D6 ingress → 22-array
validation → `finalize_run` → Step-2 loader.

| Evidence | Value |
|---|---|
| trial wall time | **42.9 s** (8M seeds, 2 phases) |
| forward / reverse | 398,156 / 398,226 |
| bidirectional candidates | **217,557** |
| generation_id | `gen-20260728T013443255510Z-step1_java_lcg_0` |
| artifact_sha256 | `e6e270ff26ee3e50742f6e73818163ce126fb79f3e554ab94d87659dfda6c50b` |
| sidecar_sha256 | `54c310f90d3bcce7953eaaad8449f7e52030f45e72250bcdacc259a1233fe277` |
| raw / L2 winners / prior / final | 217,557 / 217,557 / 0 / **217,557** |
| 22-array bundle | names+order match the frozen oracle; `validate_array_bundle()` passed |
| sidecar | 32 keys, `s172.d3_5.provenance.v1.1`, encoding `s172.phase0.encoding.v1` |
| **Step-2 loader** | `utils.survivor_loader.load_survivors` → `format=npz`, `npz_version=3`, `count=217,557`, `fallback_used=False` |

**This is the first certified accumulator generation produced on real silicon.**

`nvidia-smi` was captured before, during and after. Idle baseline was P8 / 20 W /
264 MiB; during the run the GPU sat at **P2 / 94 W / 539 MiB** with the worker
process resident (272 MiB). The per-sample utilisation figure landed between
kernel launches, so the load evidence is the power/perf state, the resident
worker, and 30 MB of survivor spool that only real kernel execution produces —
stated that way rather than as a utilisation claim.

### Three harness provisions, stated so no claim overreaches

1. **The optimizer loop** — one trial is driven directly instead of running
   `optimize_window`'s search. The trial itself is the production call.
2. **The bind address** — 127.0.0.1 on an ephemeral port, not production's
   `0.0.0.0:5700`. A single-GPU Zeus smoke needs no fixed port and no second host.
3. **Repository identity** — `finalize_run` **refuses a dirty tree** (§7.3), and
   an agent sandbox may not commit. The harness snapshots the exact source under
   test (HEAD's tracked files + the seven working-tree `.py` files) into a
   throwaway git repo, commits *there*, and passes that root to the same
   `_repository_state` helper. The recorded SHA
   (`425c75efb471b2144fda54aaa76b7873a121e2a3`) identifies a tree byte-identical
   to what ran — **it is not this project's commit.** The release-grade
   generation is the one produced after Michael commits.

---

## Non-regression

Captured green at `2a6e0f8` before any edit, and re-run after.

| Suite | Baseline | After D6 |
|---|---|---|
| Phase 3 worker | 17/17 | 17/17 |
| Phase 4 coordinator | 63/63 | 63/63 |
| D0 | 12/12 | 12/12 |
| D1.0 workflow | 8/8 | 8/8 |
| D1.1 engine | 18/18 | 18/18 |
| D2 directional uniqueness | 7/7 | 7/7 |
| D3.0 | 10/10 | 10/10 |
| D3 columnizer | 10/10 | 10/10 |
| D3.25 | 11/13 → **13/13** | 13/13, 17 mutants |
| D3.5 finalizer | 60/60 | 60/60 |
| D4 | 8/8 | 8/8 |
| D5 | 24/24 (18 mutants) | 24/24 |
| **D6 (new)** | — | **9/9 (16 mutants)** + 3.B acceptance |

D1.1 staying 18/18 **with no test edit** is the proof that the
`AssemblingPhase5Sink(backend=None)` seam changed no behaviour.

### D5 writer-freeze exception (Beta §7.7 / 4A), recorded verbatim

> "D6 introduces one approved post-D5 extension to `AssemblingPhase5Sink`: an
> optional assembly-backend seam whose `None` path is the exact pre-D6 behavior."

The D5 writer must therefore **no longer be described as unconditionally frozen
after D6**. It carries exactly one approved extension — the optional
assembly-backend seam above — and no other. `None` remains the default and is
byte-for-byte the pre-D6 path, which is what D1.1 staying 18/18 with no test edit
demonstrates.

---

## Findings raised, deliberately NOT fixed (out of D6 scope)

1. **`_flush_npz_incremental` has never written its NPZ.** The temp name
   `bidirectional_survivors_all.npz.flush.tmp` does not end in `.npz`, so
   `np.savez_compressed` appends one and writes `…flush.tmp.npz`; the following
   `os.replace(_tmp, …)` raises `FileNotFoundError` into the helper's own broad
   `except`, which prints a non-fatal warning. Observed live during the 3.B run:
   `[S152-FLUSH] Warning: incremental flush failed (non-fatal): [Errno 2] …`.
   **Consequence, and it is benign:** because the write fails *before*
   `accumulator["bidirectional"] = []`, the accumulator is never cleared, so
   every candidate still reaches the finalizer. Since D3.5 the incremental NPZ
   is not canonical anyway. **Not fixed:** repairing the temp name would change
   flush behaviour, which G-FLUSH-CADENCE forbids and D6 does not own. G-FLUSH-CADENCE
   pins the current behaviour so a future change is deliberate.
2. **Coordinator/worker seed-cap coupling is a hard abort.** A worker whose
   *advertised* `seed_caps` differ from the central config in **any** family is
   quarantined ("seed_cap 'amd'=2000000 != central config 500000"); a quarantined
   sole worker leaves no eligible worker and the trial **aborts**. Hit during
   3.B bring-up. In production both sides come from the same resolved §12.4
   configuration, so this is a hand-driven-run hazard, not a defect — but it is
   sharp, and Phase 6/7 multi-node bring-up will meet it.
   **D6's fail-closed behaviour was validated by it:** the aborted trial produced
   `MinerIngressError`, not a silent zero-candidate trial.
3. **Residue derivation is asymmetric between coordinator and worker.** The
   coordinator computes residues via `_get_residues_for_config` → no session
   filter; the worker re-derives them through `ResidueResolver` **with** the
   payload's `sessions` filter and verifies `residue_sha256`. On `daily3.json`
   with `sessions=['midday','evening']` the filter is a no-op, so they agree —
   but a **single-session** config would diverge and fail the trial with
   `ResidueVerificationError`. Phase 6/7 concern.
4. **Directional thresholds do not reach the kernel.**
   `build_stripe_assign_payload` carries no `min_match_threshold`, so the
   executor uses its 0.25 default regardless of the configured
   forward/reverse thresholds. The 3.B run therefore used 0.25 on **both**
   sides, and the harness records 0.25 in the trial metadata so the manifest
   stays truthful. Phase 6/7 concern.

---

## Contract compliance

- ✅ `serial_reference` is the default; `process_sharded` selectable only with an
  explicit `pool_size`, never promoted.
- ✅ PWC/ZMQ ingress, the D3.25 four-map contract and the `TestResult` shape are
  untouched — asserted by AST, by byte-diff against `2a6e0f8`, and by mutant.
- ✅ `MinerTrialAssembly.binary_npz_path` / `all_npz_path` stay `None` (Ruling E);
  certified paths come only from `RunArtifactResult`.
- ✅ Fail closed on every absent publication result; no fabricated zero.
- ✅ Every gate fails on wrong behaviour (16 mutants, four-part rule).
- ✅ No `git commit` / `git push` / `watcher --run-pipeline` from the sandbox.

**Fallback parity:** code=current (`a22216c`, uncommitted D6 work in tree),
env=ok (no new dependency; cupy 13.5.1 and numpy already captured).
