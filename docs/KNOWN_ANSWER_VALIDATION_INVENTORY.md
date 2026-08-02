# KNOWN-ANSWER VALIDATION INVENTORY — what already exists for PRNG-through-sieve correctness

**Brief:** one falsifiable question — *does known-answer correctness validation for the sieve
already exist in this repository, from the 44-PRNG pipeline development?*

**Answer: YES, substantially — and it is materially more than "nothing."** Michael's report is
confirmed in outline: every registry PRNG was driven through the sieves during pipeline
development, in constant-forward, constant-reverse and hybrid variable-skip modes, and
independent reference implementations were written and *debugged against known answers*. What
the evidence does **not** support is the stronger reading that all 44 were **known-answer
verified**. Most of the 44-wide sweeps checked *liveness* or *differential behaviour*, not a
planted seed. The known-answer core is real, is independent, and is narrower than the sweeps.

**Scope of this document:** read-only inventory. Nothing was changed, nothing committed. **No
harness was executed this session** — every currency claim below is static analysis and is
labelled as such.

- **Repo state:** `/home/michael/distributed_prng_analysis` on VM101, `HEAD = b510c40`,
  working tree dirty (6 modified / 2 untracked, all S1 admission-liveness work, unrelated).
- **Live registry read by import** under `~/venvs/torch` (the one dynamic check performed).

---

## 0. Verdict against Team Beta's Wall C

Wall C requires *"bounded independent known-answer controls — a reference that does NOT call the
miner's coordinator, backend or finalizer."*

| Wall C requirement | Status | Anchor |
|---|---|---|
| An independent reference implementation exists | ✅ **EXISTS**, three separate ones | §1.1, §1.2 |
| It avoids the miner's coordinator / backend / finalizer | ✅ **YES** — all predate RANGE-MINER and import none of it | §3 |
| It is a *known-answer* test (planted seed must be recovered) | ✅ **EXISTS** but narrow | §1.2, §1.3 |
| It covers all 44 registry variants | ❌ **NO** — see §2 | §2 |
| It matches the **production** java_lcg kernel definition | ❌ **NO** — the fully-independent one does not | §1.2 |
| Fixtures still present | ❌ **NO** — gitignored, never committed, absent | §1.1 |
| It has a fault-injection (positive) control per VIR-2 | ❌ **NO** — and one committed result set is vacuous | §1.5 |
| It terminates in PASS/FAIL | ⚠️ **PARTIAL** — most print and exit 0 regardless | §4 |

**Net:** Wall C is roughly *two-thirds pre-built*. The remaining third is real but is
**adaptation of existing assets**, not greenfield work. Scoping it as new work would have
discarded a working independent bidirectional-sieve reference, a kernel-semantics-correct
fixture generator family, and a direct-RawKernel harness. See §6.

---

## 1. What exists

### 1.1 Independent fixture generators — kernel-matching, registry-free

Sixteen top-level generators re-implement the PRNG **inline** and emit a known-seed draw file.
They import **nothing** — no registry, no coordinator, no engine.

| File | Families | Modes |
|---|---|---|
| `generate_all_test_data.py` (183 L) | java_lcg, minstd, xorshift128 | forward constant **and** forward variable |
| `create_java_lcg_test.py` / `_variable_test.py` | java_lcg | constant / variable |
| `create_minstd_test.py`, `create_pcg32_test.py`, `create_pcg32_hybrid_test.py`, `create_lcg32_hybrid_test.py`, `create_xorshift128_test.py`, `create_xorshift64_test.py` / `_hybrid_` / `_variable_`, `create_sfc64_test.py` / `_variable_`, `create_xoshiro256pp_test.py` / `_variable_`, `create_philox4x32_test.py` / `_1234.py` | 9 further families | constant / variable |
| `regenerate_all_tests_1234.sh` | all of the above | re-seeds every generator to `SEED = 1234`, regenerates, scp's to rigs |

**Skip semantics are kernel-correct.** `generate_all_test_data.py:39-42` applies the skip
*between every draw* (`for _ in range(skip): step()` then `step()` → emit), which is exactly the
kernel's loop at `prng_registry.py:987-989`. These generators are **not** contaminated by the
`java_lcg_cpu` divergence (§5) because they never call it.

**The fixtures themselves are gone.** `.gitignore:41` is a blanket `*.json`; the outputs
(`test_multi_prng_*.json`, `test_*_hybrid.json`) were **never committed** (`git log --all
--diff-filter=A` over those globs returns nothing) and are **absent from the working tree**. Two
were caught in an archive and later deleted:
`archives/cleanup_20251130_073217/backups/test_sfc64_known_seed.json.backup_20251021` and
`…/test_xoshiro256pp_known_seed.json.backup_20251021` (removed at `27b740b`). The *generators*
survive, so the fixtures are **regenerable** — this is a recoverable gap, not a lost asset.

### 1.2 `pa_sieve_validation_harness.py` — the only fully independent known-answer sieve control

352 lines, "Team Alpha S143". This is the single most Wall-C-shaped artifact in the repo.

- Its own `java_lcg_forward` / `java_lcg_reverse` (`:47-79`).
- **Its own CPU brute-force sieve** — `cpu_forward_sieve` / `cpu_reverse_sieve` (`:84-107`),
  docstring *"Slow but provably correct."*
- **Tier 1 — pure synthetic:** plants `KNOWN_SEED = 3_141_592`, skip 7, window 8, and asserts the
  seed is recovered **as a bidirectional survivor**. Real PASS/FAIL (`:196-203`).
- **Tier 2 — injection:** plants `2_718_281` inside 200 real PA draws, must still be recovered.
- **Tier 3 — real PA vs random baseline**, plus an explicit forward≠reverse independence check.
- Imports: `sys, json, time, random, pathlib, typing`. **No coordinator, no registry, no engine,
  no miner.**

**But its PRNG is not the production one.** Two divergences, both verified by reading:

| | `pa_sieve_validation_harness.py` | production `JAVA_LCG_KERNEL` |
|---|---|---|
| seed → state | `(seed ^ 0x5DEECE66D) & MASK` — Java scramble (`:57`) | `seed & m` — **no scramble** (`prng_registry.py:973`) |
| output extraction | `state >> (48-31)` i.e. `>>17` (`:63`) | `(state >> 16) & 0xFFFFFFFF` (`prng_registry.py:983`) |
| reverse | true modular inverse, steps **backward** (`:67-79`) | iterates **forward** on host-reversed residues (skill §0.2) |

So it is a **self-consistent closed loop**: its sieve is validated against its own generator. It
proves *the bidirectional sieve method recovers a planted seed*; it does **not** certify the
production kernel's java_lcg. That is the single highest-value fix in §6.

### 1.3 Engine-routed known-answer tests — real assertions, wrong executor for Wall C

These genuinely check `planted_seed in survivors` and branch PASS/FAIL:

- `test_all_prngs_properly.py:104-115` — `match = [s for s in survivors if s['seed'] ==
  test['seed']]` → `✅ PASSED` / `❌ FAILED`. Covers mt19937 constant, xorshift32 constant,
  xorshift32_hybrid variable.
- `test_comprehensive_prngs.py` (227 L) — 4 configurations: mt19937 constant, mt19937 hybrid
  variable, xorshift32 constant, xorshift32 hybrid variable.
- `complete_hybrid_alignment_test.py`, `test_hybrid_alignment.py`,
  `test_forward_reverse_alignment.py`, `test_distributed_reverse_sieve.py`,
  `test_reverse_kernel.py`, `test_xorshift32_*` family — same pattern.

All of them instantiate `coordinator.MultiGPUCoordinator` and call `_create_sieve_jobs` /
`execute_gpu_job`. **They call a coordinator, so they do not satisfy Wall C's independence
clause** — even though the coordinator in question is the legacy 26-GPU one, not the miner's.

Their *reference* side is sound: they call `prng_registry.*_cpu` with **`skip=0`** and perform
the skip striding themselves in host Python (`test_comprehensive_prngs.py:78-82`,
`test_all_prngs_properly.py:43-48`). See §5 — this is why the `java_lcg_cpu` bug did not
contaminate them.

### 1.4 `manual_kernel_test.py` — direct RawKernel, no coordinator

110 lines. Compiles `cp.RawKernel(config['kernel_source'], config['kernel_name'])` directly
(`:32-33`), hand-marshals all 17 kernel args, launches, and checks `if 54321 in survivors`
(`:101`). **No coordinator, no engine.** Covers `xorshift32_hybrid` only, and prints rather than
exiting non-zero. This is the correct structural template for a miner-era kernel known-answer
control.

### 1.5 The 44-wide sweeps — liveness and differential, NOT known-answer

This is where the "all 44 were tested" claim needs qualifying.

- **`test_ALL_46_prngs_10M.sh`** — runs all 46 named variants against real `daily3.json` at 10 M
  seeds and greps output for `COMPLETED`. **No planted seed, no expected answer.** This is a
  liveness/smoke sweep. It is also stale: it names **46** variants; the live registry has **44**
  (§2).
- **`quick_test_all_22.sh`** — runs forward vs reverse for 20 pairs and asserts only that the
  survivor **counts differ**. Differential, not known-answer.
- **Its committed result — `reverse_kernel_test_results.txt` — is vacuous.** All 20 rows read
  `Forward=0, Reverse=0, ⚠️ BOTH ZERO`. Zero survivors everywhere; the detector could not have
  distinguished a correct kernel from a dead one. Under **VIR-2** this is a
  vacuous-capable detector run with **no clean control and no fault-injection control**. It must
  not be cited as evidence of kernel correctness.
- **`test_all_hybrids.sh`** — prints `✅ $prng: Completed` unconditionally and closes with
  *"Note: Hybrid tests show 0 survivors with constant skip=5 test data / This is CORRECT
  behavior"*. Documents its own vacuity.

### 1.6 Scorer tests — deliberately circular, not sieve

`run_tests.py` header: *"Uses YOUR actual prng_registry.py to generate test data / This ensures
100% accuracy by using the SAME implementation for both test data generation AND scoring."* It
validates `survivor_scorer.py` (Step 3), generating and checking with the *same* function by
design. `test_local_system.py:176-200` likewise, across 5 families. **Neither is a sieve test and
neither is an independent control.**

### 1.7 Miner-era tests (`tests/`) — structural only

- `tests/test_s172_phase3_worker.py` asserts kernel-**argument** shapes: builder coverage, arg
  counts (`len(rh) == 14`), dtypes (`mat[12].dtype == np.float32`), buffer-name presence. No
  numbers are computed from a PRNG.
- The **E9 golden** (`tests/test_s172_phase5_d3_0_encoding_contract.py:126-161`) is a
  *columnizer* golden: hand-transcribed 22-array output from three synthetic records with
  **invented** match rates (`0.25`, `0.75`, `0.5`). It validates encoding and dtype packing, not
  sieve arithmetic.
- `tests/fixtures/` contains exactly one file: `pre_d5_range_miner_npz_writer.py.frozen`.

**There is no numeric PRNG-through-sieve known-answer test anywhere under `tests/`.**

### 1.8 Recovered from git history

- **`fix_cpu_reference.py`** (present at `a076602^`, deleted at `a076602`) — fixes a real uint64
  overflow in the `xoshiro256pp_reverse` CPU reference (`rotl(s0 + s3, 23)` →
  `rotl((s0 + s3) & 0xFFFFFFFFFFFFFFFF, 23)`), and its verification block states the expected
  output verbatim: *"Should be: `[808, 187, 219]`"*. **This is direct evidence that the CPU
  references were exercised against known answers during the 44-PRNG campaign, and that doing so
  found a genuine bug.** It corroborates Michael's account.
- The two `*_known_seed.json` fixtures (§1.1), for exactly the two families whose *reverse* CPU
  references exist (§2) — a coherent story, not a coincidence.

---

## 2. Coverage — which of the 44, and which modes

**The live registry has 44 entries, not 46.** Read by import this session:

```
KERNEL_REGISTRY count: 44
  constant forward: 11    hybrid forward: 11    constant reverse: 11    hybrid reverse: 11
```

11 base families × 4 modes. `docs/STEP2_BIDIRECTIONAL_SIEVE_DESCRIPTIVE_TRACE.md:507` is
correct at 44. **These are stale at 46:** `docs/CHAPTER_8_PRNG_REGISTRY.md:36,64,962,1036`,
`docs/TRIANGULATED_FUNCTIONAL_MIMICRY_VERIFIED_v1_0.md:51,64,136,139,277,564,646,658`,
`docs/PROJECT_FILE_CATALOG.md:126,377`. `test_ALL_46_prngs_10M.sh` would hard-fail on the two
names that no longer resolve.

**CPU-reference coverage is asymmetric — 26 of 44:**

| Mode | Count | Has CPU reference |
|---|---|---|
| constant forward | 11 | **11 / 11** |
| hybrid forward | 11 | **11 / 11** |
| constant reverse | 11 | **2 / 11** |
| hybrid reverse | 11 | **2 / 11** |

The only reverse variants with a CPU reference are `sfc64_reverse`,
`sfc64_hybrid_reverse`, `xoshiro256pp_reverse`, `xoshiro256pp_hybrid_reverse`. **Eighteen reverse
variants have no CPU reference at all** — no `java_lcg_reverse`, no `mt19937_reverse`, none of
the rest.

**Consequence for the brief's question:** *reverse* known-answer validation could not have been
performed from the registry for 18 of 22 reverse variants. The reverse-mode testing that did
happen (`quick_test_all_22.sh`) was therefore **differential, not known-answer** — which is
exactly what its vacuous all-zero result set shows.

This is less damaging than it looks, because per skill §0.2 reverse kernels **iterate forward**
on host-reversed residues; a forward reference plus `residues[::-1]` *is* the correct reverse
reference. But **no committed code does that reversal**, so the control does not currently exist.

**Honest coverage summary.** All 44 names have been **run**. A subset — roughly the 11 forward
families in constant and variable mode — has been checked against a **planted seed**. Reverse
modes were checked **differentially only**. "All 44 known-answer verified" overstates it; "the
sieves were driven with every PRNG in constant, reverse and hybrid modes" is accurate.

---

## 3. Method — the decisive question

Three distinct reference strategies were used. **None round-trips through the miner** — the
miner did not exist when any of this was written.

| Strategy | Independence | Used by |
|---|---|---|
| **(i)** Inline re-implementation in the generator | Independent of registry **and** engine. Kernel-matching skip semantics. | `generate_all_test_data.py`, all `create_*_test.py` |
| **(ii)** `prng_registry.*_cpu` at `skip=0` + host-side striding | Independent of the GPU kernel, coordinator, backend, finalizer — but co-located with the kernels in one file | `test_comprehensive_prngs.py`, `test_all_prngs_properly.py`, `manual_kernel_test.py`, `complete_hybrid_alignment_test.py` |
| **(iii)** Fully independent generator **and** independent brute-force sieve | Independent of everything | `pa_sieve_validation_harness.py` |

**Answer to the decisive question: the references were independent.** No harness computed its
expected answer by running the engine and reading its output back. Strategy (iii) is independent
end to end. Strategy (i) is independent of everything but uses the legacy coordinator as
*executor*. Strategy (ii) shares a file with the kernels but is a separate implementation.

**The residual caveat is executor, not reference.** Wall C's clause is *"does not call the
miner's coordinator, backend or finalizer"* — satisfied by all three, trivially. But (i) and (ii)
do call the **legacy** `MultiGPUCoordinator`, so if Beta's intent is "no orchestrator at all",
only (iii) and `manual_kernel_test.py` qualify unmodified.

---

## 4. Currency — does it still run, and against which engine?

**The load-bearing finding: the miner executes the same kernel source as the legacy sieve.**

```
miner/range_miner_worker.py:837   from sieve_gpu_worker import _get_kernel, coerce_threshold
miner/range_miner_worker.py:841   kernel, config = _get_kernel(family)
sieve_gpu_worker.py:129-130       config = get_kernel_info(prng_family)
                                  kernel = cp.RawKernel(config['kernel_source'], config['kernel_name'])
```

`get_kernel_info` reads `prng_registry.KERNEL_REGISTRY`. So **RANGE-MINER compiles and launches
the identical kernel source strings the 44-PRNG campaign exercised.** Any known-answer result
about kernel *arithmetic* transfers to the miner unchanged. This is a much stronger currency
position than "the work predates RANGE-MINER" suggests.

**What does not transfer:** everything the miner added around the kernel — argument marshalling
(`_constant_prefix` / `_hybrid_prefix`), residue-window resolution, per-direction threshold
resolution, stripe assignment, dedup, and NPZ finalization. No legacy harness touches any of it.

**Would each still execute today?** (static analysis; nothing was run)

| Asset | Verdict |
|---|---|
| `pa_sieve_validation_harness.py` | **Likely runs as-is.** Pure Python, no GPU, no imports beyond stdlib. Tier 1 is unconditional; Tiers 2–3 need `pa_pick3.json`, which **is present** at top level. |
| `generate_all_test_data.py`, `create_*_test.py` | **Likely run as-is.** Pure Python, self-contained. Regenerate the missing fixtures. |
| `manual_kernel_test.py` | Needs CuPy + a GPU. VM101 has a 3080 Ti. Kernel arg list is hand-written against the current signature — **verify before trusting**; arg counts have moved (`tests/test_s172_phase3_worker.py` asserts 12/13/14-length prefixes). |
| coordinator-routed harnesses (§1.3) | `MultiGPUCoordinator` still exists (`coordinator.py:231, :436, :1061, :2265`), but `distributed_config.json` holds the **bare-metal** rig addresses `.120/.154/.162`, and the rigs are currently in **Proxmox** (CT100s at `.122/.156/.164`). They would not reach a rig unbooted. Fixtures also absent. |
| `regenerate_all_tests_1234.sh` | scp targets `.120` / `.154` — stale endpoints, and only two of three rigs. |
| `test_ALL_46_prngs_10M.sh` | Names 46 variants against a 44-entry registry → hard-fails. Also spins 26 GPUs; **out of scope, never to be launched by an agent.** |

---

## 5. The `java_lcg_cpu` question

**The divergence is real, and confirmed numerically this session.**

- `prng_registry.py:170-181` — `java_lcg_cpu` applies `skip` **once** before generating, then
  emits `n` **consecutive** outputs.
- `JAVA_LCG_KERNEL` (`prng_registry.py:958`+) — applies `offset` (`:974-976`), then `skip` once
  (`:977-979`), then `skip` again **between every draw** (`:987-989`).

Measured (`java_lcg`, seed 12345, 5 draws, mod 1000):

| skip | `java_lcg_cpu` | kernel semantics |
|---|---|---|
| 0 | `[875, 331, 694, 737, 468]` | `[875, 331, 694, 737, 468]` — **identical** |
| 3 | `[737, 468, 925, 202, 265]` | `[737, 265, 282, 985, 776]` — **diverges after element 0** |

They agree at `skip=0` and at element 0 only. The brief's characterisation is exact.

**Was the original validation aware of this?** No document says so — but **every harness is
constructed so as to be immune to it**, and that construction is too consistent to be accidental:

- Every registry-based harness calls the CPU reference with **`skip=0`** and does the skip
  striding itself in host Python: `test_comprehensive_prngs.py:78-82`,
  `test_all_prngs_properly.py:43-48`, `manual_kernel_test.py:20-26`,
  `complete_hybrid_alignment_test.py:13-21`, `test_forward_reverse_alignment.py:29`,
  `test_hybrid_alignment.py:20`, `test_reverse_kernel.py:36,112`.
- The independent generators (§1.1) never call `java_lcg_cpu` at all — they re-implement java_lcg
  inline with kernel-matching semantics.
- `pa_sieve_validation_harness.py` never imports `prng_registry`.

**Therefore: the divergence invalidates none of the sieve validation found.** `java_lcg_cpu` was
not used with `skip > 0` to build a sieve expectation anywhere in the repo. A different reference
was used — three of them.

**Adjacent live finding, outside this brief's scope but flagged as required.** Two *production*
consumers do pass a non-zero skip straight into the divergent CPU reference:

- `survivor_scorer.py:124` — `raw = self._cpu_func(seed=int(seed), n=n, skip=skip)`
- `full_scoring_worker.py:305` — `predictions = prng_func(seed, n_holdout, skip=offset)`

These are Step-3 scoring paths evaluating **kernel-produced survivors** with **non-kernel skip
semantics**. That is a producer/consumer semantic mismatch on the live path. It is *not* a Wall C
item and I have not traced its consequences — recording it here so it is not lost. It warrants
its own investigation brief.

---

## 6. The gap — what would genuinely need doing

**Not nothing, but far less than new work.** The following already exist and are reusable:

- an independent, kernel-semantics-correct fixture-generator pattern (§1.1);
- a complete independent bidirectional CPU sieve with planted-seed PASS/FAIL and a random
  baseline (§1.2);
- a direct-RawKernel, coordinator-free invocation template (§1.4);
- and — decisively — the **same kernel source** under the miner as under the legacy sieve (§4).

Genuinely missing, in priority order:

1. **Align the independent reference to the production kernel.** `pa_sieve_validation_harness.py`
   must drop the Java seed-scramble and use `>>16` not `>>17` (§1.2), and its reverse must become
   *forward iteration over reversed residues* rather than modular-inverse backward stepping
   (skill §0.2). This is a ~20-line change to an existing 352-line harness. **Highest value per
   unit of work in the whole gap.**
2. **Fixtures must be committed or generated in-test.** Current fixtures are gitignored
   (`.gitignore:41`) and gone. Either commit them under a non-`.json` extension / an explicit
   negation, or generate deterministically inside the test. Note the `.json` extension is
   load-bearing elsewhere (skill §2.10) — do not collide with that.
3. **Cover the miner's own path.** Nothing today compares a **miner-produced** survivor set to an
   independently computed one. This is the actual Wall C deliverable and the one thing with no
   precursor: plant a seed, run it through the miner, and check recovery against the aligned
   reference from (1).
4. **Add a fault-injection (positive) control — VIR-2.** `reverse_kernel_test_results.txt` is the
   cautionary artifact: 20/20 `BOTH ZERO`, no clean control, no positive control, reported as a
   test result. Every new control needs a deliberately-wrong seed or perturbed residue that
   **must** fail, or it can pass vacuously.
5. **Reverse-mode reference for 18 of 22 reverse variants** (§2) — or, cheaper and correct,
   derive reverse expectations from the forward reference plus `residues[::-1]`, which needs no
   new PRNG code at all.
6. **PASS/FAIL exit codes.** Most legacy harnesses print and exit 0 regardless
   (`test_all_hybrids.sh` prints ✅ unconditionally). VIR-3 requires termination in
   `PASS | FAIL | UNAVAILABLE | INCOMPLETE`.

**Bounding recommendation.** TFM's sieve targets **java_lcg only** (CLAUDE.md §7). Bounding Wall C
to java_lcg × {constant forward, constant reverse, hybrid forward, hybrid reverse} is defensible,
matches "bounded" in Beta's own scoping, and is **largely assemblable from parts already in the
tree**. A 44-wide known-answer suite is not required by Wall C and is not recommended as a
precondition for Phase 6.

---

## 7. Verification-integrity controls (VIR-1…6)

- **Execution proof:** every claim carries a `file:line` anchor read this session. The registry
  counts in §2 and the numeric divergence in §5 come from a live `python3` import under
  `~/venvs/torch` on VM101, output pasted into this report.
- **Clean control:** n/a — read-only inventory, no detector authored.
- **Fault-injection control:** n/a — nothing executed. **Explicitly noted as a gap in §6.4** for
  the work this inventory scopes.
- **Completion sentinel:** this section; report written to `docs/KNOWN_ANSWER_VALIDATION_INVENTORY.md`.
- **Unavailable-observer behavior:** every surface I could not reach is listed below rather than
  treated as empty. **No absence claim in this document rests on an unsearched surface.**
- **Audit claim scope:** the **repository working tree at `b510c40` on VM101, plus its full git
  history across all branches**. Claims are repo-scoped. **The repository is not the system
  (VIR-6)** — a harness could exist on a rig, in a systemd unit, or in an uncommitted file and be
  invisible here.
- **Searched surfaces:** full top-level listing (≈900 entries); `tests/` (26 files) and
  `tests/fixtures/`; `scripts/`; `miner/`; `utils/`; `config_manifests/`; `docs/` including all
  163 `SESSION_CHANGELOG_*.md` and the Chapter series; `git log --all --diff-filter=D` for
  deleted files; targeted recovery of `fix_cpu_reference.py` from `a076602^`; live
  `KERNEL_REGISTRY` by import; `/bin/grep` used throughout for `.json` coverage (the shell `grep`
  wrapper honours `.gitignore` and would have silently skipped `*.json`).
- **Unavailable surfaces — searched-and-unreachable, or not searched:**
  - the three rig CT100s (`192.168.3.122` / `.156` / `.164`) — **no remote filesystem inspected**;
  - `.127` frozen bare-metal Zeus — not booted, Zeus runs one OS at a time;
  - host-side systemd units, cron, and any deployed-but-uncommitted files (VIR-6);
  - `archives/cleanup_20251130_073217/` contents beyond the deleted-path listing;
  - the private `origin` remote beyond what this clone's history contains;
  - **no harness was executed** — all "would it run today" verdicts in §4 are static analysis,
    not observed behaviour, and are labelled as such.

---

## 8. Recommended next step

Do **not** scope Wall C as new work. Scope it as **adaptation**, in this order: align
`pa_sieve_validation_harness.py` to the production java_lcg kernel definition (§6.1) → restore
fixtures deterministically (§6.2) → extend to the miner path (§6.3) → add the fault-injection
control (§6.4).

Separately, and **not** as part of Wall C: open an investigation brief on the
`survivor_scorer.py:124` / `full_scoring_worker.py:305` skip-semantics mismatch (§5).

**STOP** — per brief. No changes made, nothing committed, no pipeline launched.
