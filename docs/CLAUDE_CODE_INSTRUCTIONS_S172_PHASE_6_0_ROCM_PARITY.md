# CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_6_0_ROCM_PARITY.md — REV1

**S172 — `Phase 6.0: single-rig ROCm miner smoke, paired against an identical CUDA
control.`**

Beta approved Phase 6.0 with one required addition: schema-valid ROCm output alone
does not establish computational parity, because a platform-specific kernel defect
could still produce a structurally valid generation. Acceptance therefore requires
**field-for-field equality of all 22 canonical arrays across CUDA and ROCm** for the
same deterministic trial.

**This is the first execution of the RANGE-MINER production path on the hardware the
entire rearchitecture was motivated by.** Everything certified through D6.1 ran on
CUDA; the L2/reset failures that killed PWC were on the RX 6600 fleet.

**Base:** `3823b56` (D6.1 closed and pushed). Claude Code on VM 101 as `michael`.
Implement and iterate; do **NOT** commit, push, or run WATCHER. STOP at the gate.

---

## 0. Verified prerequisites — do not re-litigate, but DO re-confirm at run time

Checked live on 2026-07-30 from VM 101. Re-confirm each in the run banner; do not
assume they still hold.

| requirement | verified state |
|---|---|
| rig reachable, key auth | `rrig6600` @ **`192.168.3.122`** (CT100), `ssh -o BatchMode=yes` succeeds from VM 101 |
| GPUs bound into the LXC | `/dev/kfd` present; `card0-7` + `renderD128-135` present |
| ROCm sees them | `/opt/rocm/bin/rocm-smi` enumerates 8 × **AMD Radeon RX 6600 XT**, device id `0x73ff`, rev `0xc1` |
| cupy under ROCm | **13.5.1**, `getDeviceCount() == 8`, real kernel executes (`arch 103` = gfx1032, arithmetic verified) |
| ROCm runtime | `runtimeGetVersion() == 60443484` (6.4.43484) |
| env overrides | **NONE required or set** — empty in both non-interactive and login environments; a kernel compiles and runs regardless. `HSA_OVERRIDE_GFX_VERSION` is unnecessary: ROCm 6.4 supports gfx1032 natively. |
| source deployment | CT reaches GitHub (`git ls-remote` returned `3823b56cd8637b…`) |
| repo on CT | **ABSENT** — `~/distributed_prng_analysis` does not exist. Deployment is a task, not a given. |
| ROCm venv on CT | `~/rocm_env` present (note: **not** `~/venvs/torch`, which is VM 101's) |

**SKU correction — carry this into all evidence.** The cards are **RX 6600 XT**, not
"RX 6600". `distributed_config.json`, `CLAUDE.md` §3, and the Phase 6.0 proposal all
say RX 6600. Record the true device identity; do not propagate the wrong SKU into a
certification record. Do **not** edit those files in this deliverable — report the
discrepancy for a separate docs correction.

**Topology note (already correct, do not "fix"):** `distributed_config.json` holds
**bare-metal** rig addresses (`.120/.154/.162`) because those match the *default*
boot target. The rigs are currently booted into Proxmox, so the live worker endpoint
is CT100 at `.122/.156/.164` (`host = rig+1`, `CT100 = host+1`). `CLAUDE.md` §3 states
explicitly this is **not a bug and must not be corrected.** Phase 6.0 addresses CT100
directly and changes no config.

---

## 1. Deployment (first task)

Clone the repo onto CT100 at the **exact** commit under test:

```
~/distributed_prng_analysis on 192.168.3.122, at commit 3823b56
```

Clone rather than copy: `git rev-parse HEAD` on the CT then **proves** the source
commit matches VM 101's, satisfying Beta's identical-source requirement by
construction rather than by trust. Verify and record the CT-side `git rev-parse HEAD`
and tracked-tree cleanliness.

Confirm the ROCm venv can import what the worker needs (`cupy`, numpy, and any
worker import) **from the deployed tree**, using `~/rocm_env/bin/python`.

## 2. The paired trial — identical on both platforms

Run the **same bounded trial twice**. Every one of these must be identical across the
two runs; any divergence invalidates the comparison:

```
source commit          seed start and count      draw data + residue window
sessions               window size and offset    skip mode / range
forward threshold      reverse threshold         PRNG family (java_lcg)
assembly backend (serial_reference)              constant-skip phases 1 and 2
```

- **CUDA control:** VM 101, RTX 3080 Ti, `~/venvs/torch`.
- **ROCm subject:** CT100 `rrig6600`, **one** RX 6600 XT (device 0), `~/rocm_env`.

Reuse the existing `tests/smoke_s172_phase5_d6_zeus_single_gpu.py` harness — it
already parameterises `--seed-start/--seed-count/--stripe-size/--seed-cap/
--window-size/--forward-threshold/--reverse-threshold` and passes `--gpu-id` /
`--device-index` to the worker. Extend it for a remote ROCm target rather than
writing a second harness; keep the CUDA default path byte-identical in behaviour.

**Use non-default asymmetric thresholds** (`forward=0.31 / reverse=0.47`) so the D6
provenance chain is exercised on ROCm too. A `0.25/0.25` run would prove nothing
about the corrected path.

**Pinned trial parameters — use the D6 release-grade run's values exactly:**

```
--seed-start 0        --seed-count 8000000     --stripe-size 4000000
--seed-cap 1000000    --window-size 3
--forward-threshold 0.31   --reverse-threshold 0.47
java_lcg · serial_reference · constant-skip phases 1 and 2 · daily3.json
```

Beta required only that the two Phase-6.0 runs be identical **to each other**;
matching D6's parameters is Alpha's addition, and it buys a **third comparison for
free**: the ROCm output can also be checked against the already-certified
release-grade CUDA generation (`gen-20260730T002104136270Z-step1_java_lcg_0`,
commit `b08c2c5`), whose expected counts are **forward 398,156 / reverse 383 /
bidirectional 319**. Those are known-good numbers — a mismatch is immediately
legible rather than needing a fresh baseline to interpret. **Every** parameter above
must match, not just the seed count.

**Expect the ROCm run to take longer** than the CUDA run's 23.2 s: one RX 6600 XT is
a much smaller card than an RTX 3080 Ti. That is expected and is **not** a finding —
Phase 6.0 makes no throughput claim.

**Bounded by design:** one rig, one GPU, one persistent worker, constant-skip
forward+reverse only. **No** multi-node coordination, **no** backend promotion,
**no** saturation claim, **no** WATCHER.

## 3. Acceptance — 3.B parity PLUS cross-platform equality

**3.B-parity evidence, on the ROCm side:**
- clean committed repository (CT-side commit + tracked-clean recorded);
- certified generation produced;
- frozen-order **22-array** validation (`validate_array_bundle()`);
- Step-2 loader read-back with **`fallback_used=False`**;
- asymmetric non-default thresholds;
- `requested == payload == effective`, with the **effective value returned by the
  ROCm executor** (not recomputed from config).

**Cross-platform equality — the required addition:**
- `np.array_equal(cuda_array, rocm_array)` for **every one of the 22 canonical
  arrays**;
- forward survivor count, reverse survivor count, bidirectional count — all equal;
- canonical record **ordering** identical;
- threshold provenance identical;
- Step-2 loader metadata identical.

Report the comparison as an explicit **22-row matrix**, not a summary boolean. If any
array differs, report *which*, the first differing index, and both values — a
divergence is a finding to localize, never to average away.

**Third comparison (Alpha addition, free given the pinned parameters):** also check
both runs' counts against the D6 release-grade CUDA generation — **forward 398,156 /
reverse 383 / bidirectional 319**. Agreement is corroboration against a
previously-certified artifact; disagreement on the *CUDA* side would indicate a
regression since `b08c2c5` rather than a platform difference, and must be
distinguished from a ROCm-only divergence. Report all three sets side by side.

## 4. ROCm identity and health evidence (Beta-required)

Capture in the raw evidence:
- hostname and GPU index;
- **RX 6600 XT** device identity (name, device id, rev, GUID);
- ROCm/HIP runtime and driver versions;
- kernel and `amdgpu-dkms` versions;
- the worker's **explicit ROCm/HIP backend identity** (prove it is the ROCm path, not
  a CUDA fallback);
- environment overrides used (expected: **none** — record that as a positive finding);
- clean worker exit;
- **no GPU reset, no L2 protection fault, no VM fault, and no new relevant `amdgpu`
  kernel error during the run** — capture `dmesg`/kernel log before and after and
  diff. *This is the failure class the entire rearchitecture exists to avoid; its
  absence must be evidenced, not assumed.*
- no abandoned spool or temporary artifact on either host;
- no `.s172_checkpoint/` residue outside the run-isolated namespace.

Timing may be recorded **for context only**. No throughput conclusion, no promotion
decision, and no saturation claim belongs to Phase 6.0.

## 5. Artifact classification — non-authoritative

The ROCm run uses the normal finalizer and produces a technically certified
generation. Classify it in **all** output and evidence as:

> **ROCm platform-validation certified generation — non-authoritative**

Do **not** call it a second release-grade generation. The D6 CUDA generation
(`gen-20260730T002104136270Z-step1_java_lcg_0`, certified at `b08c2c5`) remains the
authoritative release-grade artifact. Phase 6.0 proves platform equivalence on
different silicon; it does not replace or compete with that artifact.

## 6. Scope — do NOT touch

`distributed_config.json` (deliberately bare-metal — see §0); `CLAUDE.md` §3;
the D6/D6.1 implementation; PWC/ZMQ ingress; the D3.25 contract; `TestResult` shape;
D5's artifact contract. `serial_reference` stays default; `process_sharded` stays
unpromoted. Do not implement retention/GC (that is **D6.3**) or the 24-field
checkpoint (**D6.2**). Do not enable the S166 clear.

## 7. Non-regression

Capture green at `3823b56` before any edit and again after: D1.1 · D1.0 · D0 · D2 ·
D3.0 · D3 · D3.25 · D3.5 · D4 · D5 · D6 3.A · D6-threshold · **D6.1** · Phase 3 ·
Phase 4. If the harness is extended for a remote target, the CUDA default path must
remain behaviourally unchanged — prove it.

## 8. Report

Deployment evidence (CT commit + cleanliness); the identical trial configuration used
on both platforms; both platforms' hardware/backend identity; requested/payload/
effective thresholds for each; forward/reverse/bidirectional counts for each; the
**22-array `np.array_equal` matrix**; both generation IDs and artifact SHA-256 values;
both Step-2 load-backs with `fallback_used=False`; the ROCm kernel-log health check
(before/after diff); spool and temp cleanup result; and the SKU discrepancy note.
Then STOP for Team Alpha review.

**Do not commit.** After Alpha + Beta pass: Michael commits the evidence record.
