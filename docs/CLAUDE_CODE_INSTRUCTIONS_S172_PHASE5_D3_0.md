# CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3_0.md

**S172 RANGE-MINER — Phase 5, Deliverable D3.0: legacy seam correction —
canonical PRNG/skip encoding + rectangular 22-array empty output**

**Audience:** Claude Code on VM 101 (`michael@192.168.3.177`), in
`~/distributed_prng_analysis`. You write and iterate; you do NOT commit, push,
or run WATCHER. When the gate + non-regression are green, STOP and report.

**Frozen against HEAD `2d37b77`.** Authority: the Team Beta D3 pre-write ruling
(Ruling A — canonical encoding mandatory, no fallback, no Phase-6 divergence
whitelist; Ruling B — rectangular 22-array empty output) and the Team Beta
`randu` disposition ruling, absorbed herein.

---

## 0. Why D3.0 exists

Phase 0 replaced inline PRNG encoding tables with a registry-derived canonical
module that hard-fails on unknown identities. **That fix never reached either
NPZ writer.** Verified at `2d37b77`: `utils/prng_encoding.py` is imported by
exactly two files in the repo (`tests/test_prng_encoding.py`,
`miner/range_miner_npz_writer.py`). Both NPZ writers still carry a local
12-entry table with a silent `.get(..., 0)` fallback:

- `convert_survivors_to_binary.py:30-38` (`PRNG_TYPE_ENCODING`,
  `SKIP_MODE_ENCODING`), used at `:127-136`
- `window_optimizer_integration_final.py:1715` (`_PRNG_ENC`), used at `:1758`
  — the **live** Step-1 NPZ producer

Measured divergence, canonical (44 entries) vs legacy (12):

```
7 shared keys DISAGREE on value:
    java_lcg_reverse, lcg32, minstd_reverse, mt19937,
    mt19937_reverse, xorshift128, xorshift128_reverse
java_lcg_hybrid: canonical=1, legacy=MISSING -> .get(...,0) collapses to java_lcg=0
legacy-only keys absent from the registry: randu(10), randu_reverse(11)
```

TFM runs java_lcg (0 in both), but **phases 3/4 emit `java_lcg_hybrid`** —
canonical 1, legacy silently 0. So every historical hybrid NPZ carries a
mislabelled `prng_type` column feeding the ML feature surface. This is the
exact bug Phase 0 was created to fix, still live where it does the most damage.

Second defect: the two writers disagree with each other on the empty case.
`convert_survivors_to_binary.py:64` writes **one** array (`seeds=[]`); the
inline writer keeps all 22 rectangular (S163-KARG-NPZ, `:1833-1860`). Ruling B:
the one-array form is a defect, not an alternate valid representation.

**D3.0 is a narrow seam correction. It does NOT restructure the accumulator.**

## 1. Non-negotiable working rules

1. **Read live source before every claim.** Every cite above was verified at
   `2d37b77`; re-verify before depending on it.
2. **Each gate must FAIL on wrong behavior** — the pre-fix tree MUST fail the
   hybrid-encoding and rectangular-empty gates (§4 requires you to demonstrate
   this).
3. **Behavior-preserving except where specified.** The ONLY intended behavior
   changes are: correct numeric IDs for non-`java_lcg` identities, hard-fail on
   unknown identities, and rectangular empty output. Non-empty, known-identity
   output must be otherwise unchanged.
4. STOP at the gate. No commit/push/WATCHER. Do not begin D3.

## 2. Scope — two production files + one new harness

**Modify:**
- `convert_survivors_to_binary.py` — encoding + empty case
- `window_optimizer_integration_final.py` — encoding ONLY (`_PRNG_ENC` at
  `:1715` and its use at `:1758`). **Do not touch** the accumulator merge,
  supersede logic, backfill, sort, or dual write.

**Create:**
- `tests/test_s172_phase5_d3_0_encoding_contract.py`

Gate-22 whitelist: register the new test path AND note that two production
files changed (pre-authorized registration only; if gate 22's structure cannot
express a production-file change, STOP and report rather than altering gate
logic).

**Explicitly OUT of scope:** any accumulator restructuring; any change to L2
`deduplicate_survivors`; any change to the candidate-ingress adapter
`_build_test_result_from_pw` (that is **D3.25**); the columnizer refactor (that
is **D3**); migration of existing NPZ artifacts (that is **D3.5**).

## 3. Required corrections

### 3.1 Canonical encoding, both writers

1. `from utils.prng_encoding import encode_prng_type, encode_skip_mode`
   (verify the exact exported names before writing calls — use
   `inspect.signature`).
2. **Delete** both local tables: `PRNG_TYPE_ENCODING` / `SKIP_MODE_ENCODING`
   in `convert_survivors_to_binary.py:30-38` and `_PRNG_ENC` in
   `window_optimizer_integration_final.py:1715`.
3. Replace every `.get(value, 0)` encode with a direct canonical call. **No
   fallback, no alias, no compatibility mapping, no preservation of legacy IDs
   10/11.**
4. Unknown identities propagate the canonical `ValueError` unchanged — do NOT
   wrap it in a local exception type (per the standing D1.1 Ruling B: the
   canonical module's hard-fail is the validation decision and must not be
   shadowed).
5. **Resolution before encode is the caller's job** (the canonical docstring
   says so explicitly): where the legacy code reads
   `s.get('prng_type', s.get('prng_base', 'java_lcg'))`, keep that resolution
   step, then pass the resolved string to `encode_prng_type`. Do not pass a
   bare `prng_base` and hope.
6. `skip_mode` uses `encode_skip_mode` the same way. **Verified: canonical
   `{constant: 0, variable: 1}` is identical to the legacy table**, so this is
   a pure source-of-truth change with no numeric effect — assert that in a
   gate rather than assuming it silently.

### 3.2 Rectangular empty output (`convert_survivors_to_binary.py` only)

`:60-66` currently writes one array and returns. Replace with all 22 arrays,
each **length zero with its exact frozen dtype**:

| dtype | arrays |
|---|---|
| `uint32` (1) | `seeds` |
| `int32` (6) | `window_size`, `offset`, `trial_number`, `skip_min`, `skip_max`, `skip_range` |
| `float32` (13) | `forward_matches`, `reverse_matches`, `score`, `forward_count`, `reverse_count`, `bidirectional_count`, `intersection_count`, `intersection_ratio`, `intersection_weight`, `bidirectional_selectivity`, `forward_only_count`, `reverse_only_count`, `survivor_overlap_ratio` |
| `uint8` (2) | `skip_mode`, `prng_type` |

The six `*_count` arrays are `float32` despite being logically integers —
**reproduce that exactly**; it is the existing on-disk contract.

An empty artifact must be structurally indistinguishable from a non-empty one
except for length: exactly 22 keys, same key set, same dtypes, all lengths 0.

## 4. Gate — `tests/test_s172_phase5_d3_0_encoding_contract.py`

**Independent oracle (the G9 lesson, binding):** the harness must contain its
own hand-transcribed expectations — the 22 array names, their order, their
dtypes, and the specific numeric IDs asserted below. It must **not** import
`PRNG_TYPE_ENCODING`/`SKIP_MODE_ENCODING` and then use the same constant as
its expected result. Where a canonical ID is asserted, write the integer
literal.

Checks (each must fail on wrong behavior):

1. **E1** `java_lcg` encodes to the same ID through both corrected writers and
   the canonical module (integer literal `0`).
2. **E2** `java_lcg_hybrid` encodes to canonical `1`, **not** `0`. *(Fails
   pre-fix — the legacy table lacks the key and `.get(...,0)` collapses it.)*
3. **E3** at least one shared key whose legacy value differed now matches
   canonical — assert `java_lcg_reverse == 3` (legacy had `1`).
4. **E4** unknown `prng_type` raises `ValueError` (not `0`) from both writers.
5. **E5** unknown `skip_mode` raises `ValueError`.
6. **E6** `randu` and `randu_reverse` each raise `ValueError`. Per the Team
   Beta ruling these are *unsupported and unreachable through the current
   registry-backed kernel producer path* — do NOT assert they were never
   historically emitted, and do NOT preserve IDs 10/11.
7. **E7** `skip_mode` numeric output is unchanged by the refactor:
   `constant → 0`, `variable → 1` (integer literals).
8. **E8** empty input to `convert_survivors_to_binary` produces exactly **22**
   keys, the exact key set, exact dtypes, all lengths `0`. *(Fails pre-fix —
   one array.)*
9. **E9** non-empty behavior otherwise unchanged: for a fixture of
   known-identity `java_lcg` records, every one of the 22 arrays is
   `np.array_equal` to the pre-fix output (capture the pre-fix output first —
   see below).
10. **E10** the inline writer's `prng_type` column for a mixed
    constant+hybrid fixture now carries `{0, 1}` rather than `{0}` collapsed.

**Pre-fix capture requirement:** before editing production, run the harness
against the unmodified tree and record which checks fail — **E2, E8, E10 must
fail pre-fix** (and E3/E4/E5/E6 as applicable). Save that output as evidence.
For E9, capture the pre-fix 22-array output on the java_lcg fixture and
compare the post-fix output against it.

**Blocking non-regression:** D2 7/7, D1.1 18/18, D1.0 8/8, D0 12/12, Phase 4
63/63, Phase 3 17/17. Capture the baseline green at `2d37b77` BEFORE any edit.

## 5. Stop conditions

- canonical `skip_mode` values turn out NOT to match the legacy table (they
  were verified to match — if your check disagrees, STOP);
- correcting the encoding requires touching the accumulator, `_survivors_to_arrays`,
  or `_build_test_result_from_pw` — STOP and report the coupling;
- gate 22 cannot express a production-file change without altering its logic —
  STOP and report;
- a non-regression suite reds because a *fixture* carries a now-invalid
  identity — STOP and report (that is a real finding about existing test data,
  not something to work around by restoring a fallback);
- any gate passes only by weakening it.

## 6. Report

Diff + status, full command/output evidence, the **pre-fix failure capture**
(E2/E8/E10 red before the edit), and the E9 pre/post array-equality evidence.
Then STOP for Team Alpha review.
