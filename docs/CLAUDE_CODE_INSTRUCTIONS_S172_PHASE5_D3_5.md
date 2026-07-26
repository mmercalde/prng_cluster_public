# CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3_5.md — REV3.1

**REV3.1 changelog** — absorbs the Team Beta REV3 final review (architecture
approved; one blocking provenance correction + three clarifications):
**[D1]** the **chain tip** is now authenticated — the generation directory is
named `<generation_id>--<sidecar_sha256>`, making the atomic `current` pointer
the trust anchor for the newest generation, which by definition has no child to
vouch for it; **[D2]** automatic prior selection frozen across all four cases —
omitting the optional argument must not silently start a new lineage;
**[D3]** one sidecar write/fsync/hash order frozen (hash the **reopened stored
bytes**, resolving a REV3 contradiction between §7.2 and §7.3);
**[D4]** post-swap failure defined — `PublicationDurabilityError`, no
`RunArtifactResult`, recovery validation on the next invocation. Gates F48-F51.
**After these amendments deployment is automatically approved.**

**REV3 changelog** — absorbs the Team Beta REV2 pre-deployment review (approved
in principle; four narrow blockers, no redesign): **[C1]** `sidecar_sha256`
removed from the sidecar payload — a file cannot contain its own hash; it lives
in `RunArtifactResult` and in the **child's** `parent_sidecar_sha256`;
**[C2]** run identity bound to every current and prior row — a valid candidate
of a *different* `prng_base` must not be silently labelled by the sidecar;
**[C3]** prior selection pinned to the live `current` pointer plus **recursive**
chain validation to a clean-start root — otherwise the parent hashes are
recorded metadata, not an enforced chain; **[C4]** first-generation alias
bootstrap frozen, so F29 holds on generation 1. Plus **[C5]** the prior's full
numeric-domain validation. All REV2 rulings remain approved.

**S172 RANGE-MINER — Phase 5, Deliverable D3.5: the shared run finalizer —
L2 winner selection, L3 array-domain merge, immutable-generation publication**

**REV2 changelog** — absorbs the Team Beta pre-deployment review (NOT APPROVED
as written; four architectural blockers): **[B1]** L3 moves to the **22-array
domain after** D3 columnization — a 22→24 reconstruction of the prior is
impossible because D3 deliberately drops `sessions` and `prng_base`;
**[B2]** **every raw candidate** validates through D3 before L2, so a malformed
*losing* candidate cannot vanish; **[B3]** publication becomes an immutable
**generation directory** with a single `current`-pointer commit — two root-file
renames cannot be atomic together; **[B4]** the finalizer must **not** sit
inside the live caller's swallow-and-fallback wrapper. Plus: exact public API
(§8), exact coverage arithmetic (§6), data-**and**-metadata hash chain (§7.2),
prior-generation validation checklist (§9), retirement of the legacy score-only
deduplicator (§10), gates F27-F37.

**Audience:** Claude Code on VM 101 (`michael@192.168.3.177`). You write and
iterate; you do NOT commit, push, or run WATCHER. When gates + non-regression
are green, STOP and report.

**Frozen against HEAD `70cd6f0`.** Authority: Team Beta scoping ruling
(Questions A/B/C), pre-deployment review (blockers 1-4 + §§5-10), Ruling D (tie
semantics), Ruling F (`docs/PROVENANCE_DISPOSITION_ACCUMULATOR_20260725.md`
REV2 — CLOSED, clean start).

---

## 0. What D3.5 is, and the corrected pipeline

D3.5 replaces the inline run-finalization block in
`window_optimizer_integration_final.py` with a **shared finalizer used by every
backend** (legacy, PWC, ZMQ, and — via D6 — miner).

**The pipeline order is binding [B1][B2]:**

```text
all raw current-run 24-field candidates
    -> STRICTLY VALIDATE EVERY RAW CANDIDATE           (D3, before anything else)
    -> validate current-run coverage
    -> L2 winner selection                              (RECORD domain)
    -> records_to_arrays(L2 winners)                    (D3 columnization)
    -> load + validate certified prior 22-array bundle
    -> L3 merge                                         (ARRAY domain)
    -> global seed-ascending array ordering
    -> validate_array_bundle(final arrays)
    -> immutable-generation publication
    -> RunArtifactResult
```

**Why L3 is array-domain [B1]:** the certified prior is a **22-array** NPZ. The
canonical 24-field record contains two fields the arrays do not carry —
`sessions` and `prng_base` — which D3 drops by contract. Reconstructing
24-field prior records is therefore impossible without inventing data. Merging
in the array domain also makes equal/lower prior retention natural: retained
rows are **copied directly from their existing typed arrays**, never
reconstructed or re-encoded.

### 0.1 This is newly certified construction, not a port

Team Beta's binding framing. The historical accumulator **never functioned as a
cross-run mechanism**: exactly one genuine accumulation event
(`Net new: +352`, 2026-03-15) in five months of logs; every other row-count
change came from the failure path overwriting the whole file. There is **no
proven-good production history to regress against.** Every rule below is gated
on its own merits — "the old code did it this way" is not evidence.

### 0.2 The structural defect being designed out

```text
OLD:  one mutable file, edited in place
      merge raises -> fallback writer REPLACES it -> lineage destroyed silently
NEW:  immutable generations + parent hash chain + single pointer commit
      any failure -> previous certified generation still current, nothing published
```

**No fallback writer may be invoked from the finalizer under any circumstance.**

---

## 1. Non-negotiable working rules

1. **Read live source before every claim.** Cites verified at `70cd6f0`.
2. **Each gate must FAIL on wrong behavior** — §11's mutation set is the Rule-2
   evidence.
3. **Independent oracles.** Expected key sets, orders, dtypes, tie outcomes and
   sidecar fields are hand-transcribed. Do **not** import a production constant
   and assert against it (D1.1 G9 / D3.0 E8 / D3 C1 lesson).
4. **Synthetic priors only.** No real prior exists (Ruling F: clean start) and
   none is required.
5. STOP at the gate. No commit/push/WATCHER.

## 2. Scope

**Create:** `utils/run_finalizer.py` (finalizer, `RunArtifactResult`, sidecar
read/write, generation-directory machinery);
`tests/test_s172_phase5_d3_5_finalizer.py`.

**Modify:** `window_optimizer_integration_final.py` (replace the inline block;
fix the swallow wrapper per §5; retire the legacy dedup's authority per §10);
`tests/test_s172_phase4_coordinator.py` (gate-22 registration only).

Verified sites at `70cd6f0`: `deduplicate_survivors` def `:1684`, L2 call
`:1735`, JSON write `:1769-1780`, columnize `:1878`, L3 merge `:1891-1895`,
sort `:1960-1961`, dual write `:1968`/`:1971`, summary `:1973-1978`, tagged
re-raise `:1986`, **broad swallow `:2004`**.

**Must NOT modify:** `utils/canonical_arrays.py`; `utils/canonical_records.py`;
D3/D3.25 tests; `persistent_worker_coordinator.py`; `zmq_sqlite_coordinator.py`;
`miner/*`; `prng_analysis.db` or any coverage table; WATCHER. Discovering a
required change there triggers **STOP**.

**Reuse, do not reimplement:** `records_to_arrays` (`utils/canonical_arrays.py:480`),
`validate_array_bundle` (`:543`), `CANONICAL_RECORD_FIELDS`
(`utils/canonical_records.py:115`).

## 3. Raw-candidate validation before L2 **[B2]**

A malformed *losing* candidate must fail the run, not vanish during selection.
Example that must fail:

```text
seed X: valid record, score 0.9
seed X: record MISSING `sessions`, score 0.4     <- loses L2, but must still fail the run
```

Required sequence:

1. **materialize the candidate iterable exactly once** (it may be a generator;
   D3's `records_to_arrays` consumes it in one pass);
2. pass the **complete raw candidate list** through D3's strict
   `records_to_arrays` — this is the 24-field validation wall;
3. discard or reuse those temporary arrays;
4. only then perform coverage validation and L2.

**Do not reimplement the 24-field validator in D3.5.**

## 4. L2 — batch winner selection (record domain)

For candidates sharing a seed within one run's batch (Ruling D, binding):

```text
1. highest canonical float32 score
2. then lowest trial_number
3. then constant before variable — ONLY as a tiebreak within the same trial
```

- Comparison domain is **`float32`**: convert with `np.float32(...)` before
  comparing. Two Python floats differing only beyond `float32` precision are an
  **exact tie**. Never compare pre-rounding Python floats and store the rounded
  value.
- A **same-trial, same-mode** collision for one seed is impossible after D1/D2
  and must raise a dedicated accumulator-consistency error.
- Exactly one record per seed; the result is **independent of input order**.

## 5. L3 — merge in the array domain **[B1]**

For one L2 winner row versus one certified prior row with the same seed:

```text
new score >  prior score  -> replace with the new row
new score == prior score  -> RETAIN PRIOR, byte-for-byte, every array
new score <  prior score  -> RETAIN PRIOR, byte-for-byte, every array
```

- Strict greater-than only. **Equal retains prior** — the L2 tiebreakers must
  never displace this.
- **No combined generic max-sort** over `prior + raw candidates`. Order is
  fixed: validate all raw → L2 → columnize winners → *then* L3 against prior
  arrays.
- Prior rows with no matching current seed are retained unchanged, copied
  directly from their existing typed arrays.
- Final output is **globally seed-ascending**.

### 5.1 Prior policy (binding, Ruling F)

```text
no prior generation                     -> start with an empty accumulator
prior + valid matching sidecar          -> validate (§9), then merge
prior without sidecar                   -> FAIL CLOSED
hash / schema / encoding mismatch       -> FAIL CLOSED
historical pre-D3 artifact              -> PROHIBITED, never import or migrate
```

**No filename-based trust.** The archived 20,949-row artifact gets no
exception; it is forensic evidence only.

## 6. Coverage validation — exact **[Question B ruling]**

D3.5 proves a **local** invariant: every candidate seed lies within the
interval declared for the current run. **One contiguous interval only** — the
live caller supplies `seed_start` and `seed_count`; a later deliverable may
generalize to multiple intervals.

Freeze these checks, all in **Python integer** arithmetic:

```text
seed_start          : int, bool rejected, 0 <= seed_start < 2**32
seed_count          : int, bool rejected, seed_count > 0
seed_end_exclusive  : seed_start + seed_count   (Python ints)
                      seed_start < seed_end_exclusive <= 2**32
candidate seed      : int, bool rejected, seed_start <= seed < seed_end_exclusive
```

**Do not perform the addition in `np.uint32`** — that permits wraparound.

**The `uint32` domain wall.** The frozen artifact stores `seeds: uint32`, so the
representable domain is `[0, 2**32)`. Java LCG's 48-bit internal state does
**not** silently expand this schema. A 48-bit domain requires a separately
governed revision (`seeds uint64`, new schema version, new readers, new sidecar
contract, new Phase-6 baseline) and is **out of scope**.

### 6.1 Explicitly outside D3.5

Do **not**: read or update `prng_analysis.db`; decide whether global coverage is
continuous; repair gaps in `exhaustive_progress`; trust historical coverage
rows; coordinate survivor and coverage writes in one cross-store transaction; or
prevent intentionally non-contiguous sweeps. A separate coverage-ledger
deliverable owns gap detection, overlap handling, consolidation and resume
policy.

## 7. Publication — one atomic commit point **[B3]**

Two independent root-file renames cannot guarantee that a reader never observes
one without the other. Publication therefore uses an **immutable generation
directory** with a single pointer swap.

### 7.1 Layout

```text
.s172_accumulator/
├── generations/
│   └── <generation_id>--<sidecar_sha256>/     <- hash-bound name [D1]
│       ├── bidirectional_survivors_all.npz
│       ├── bidirectional_survivors_binary.npz
│       └── provenance.json
└── current -> generations/<generation_id>--<sidecar_sha256>
```

Example: `generations/run-abc123--8f91...e27/`, whose `provenance.json` records
`generation_id = run-abc123` and — per [C1] — does **not** contain its own hash.

**Why the name carries the hash [D1].** Every historical generation is
authenticated by its child's `parent_sidecar_sha256`. The **newest** generation
has no child, so without this its `provenance.json` could be modified and the
next run would hash whatever it found, with no authoritative expected value.
Binding the hash into the pointer target makes the atomic `current` swap the
trust anchor for the live tip.

The two NPZ names must contain **byte-identical payloads** (hard link where the
filesystem allows; otherwise write one and copy before hashing). Root
compatibility names remain as static symlinks through `current`:

```text
bidirectional_survivors_all.npz    -> .s172_accumulator/current/bidirectional_survivors_all.npz
bidirectional_survivors_binary.npz -> .s172_accumulator/current/bidirectional_survivors_binary.npz
```

### 7.1a First-generation alias bootstrap **[C4]**

For F29 to hold on the **first** generation, both root aliases must already
exist as **dangling symlinks before `current` is committed** — otherwise they
would require separate post-commit filesystem changes and the single-commit
property would be false on generation 1.

Before creating the temporary generation directory:

```text
for each root alias (bidirectional_survivors_all.npz,
                     bidirectional_survivors_binary.npz):
    absent                        -> create the exact static symlink
    exact expected symlink        -> accept
    regular file / directory /
    wrong-target symlink          -> FAIL CLOSED

fsync output_root after alias creation
```

Only then proceed to §7.2. At the step-12 swap both aliases become valid
simultaneously.

**No existing regular file may be silently replaced.** This matters especially
because the historical root artifacts were explicitly removed under Ruling F —
a regular file reappearing at those paths indicates something wrote outside the
finalizer and must stop the run.

A failure after bootstrap but before the `current` commit leaves only harmless
dangling aliases; no generation is accepted.

### 7.1b Current-generation validation **[D1]**

Before trusting `current`, in this order:

```text
1. read the symlink WITHOUT following an arbitrary external target
2. require its target to be a DIRECT CHILD of .s172_accumulator/generations
3. parse <generation_id> and <expected_sidecar_sha256> from the target name
4. hash the stored provenance.json
5. require the actual hash == the hash embedded in the pointer target
6. require sidecar.generation_id == the parsed generation_id
```

Any mismatch, an escape outside `generations/`, or a non-directory target
**fails closed**. Older generations remain authenticated through their
children's `parent_sidecar_sha256`; this rule covers the tip.

### 7.2 Publication sequence (binding order) **[D3]**

```text
 1. create .tmp-<generation_id> under generations/ on the SAME filesystem
 2. write both NPZ names
 3. validate both and verify byte-identical
 4. hash the canonical NPZ
 5. serialize the canonical sidecar bytes (no sidecar_sha256 field)
 6. write all sidecar bytes, flush, fsync provenance.json
 7. REOPEN and read the STORED bytes; compute sidecar_sha256 from them
 8. fsync NPZ file data
 9. fsync the temporary generation directory
10. atomically rename .tmp-<id> -> generations/<generation_id>--<sidecar_sha256>
11. fsync generations/
12. create a temporary `current` symlink
13. ATOMICALLY REPLACE `current`          <- THE SINGLE COMMIT POINT
14. fsync .s172_accumulator/
15. only now construct RunArtifactResult
```

**Hash the reopened file, never the pre-write memory buffer [D3]** — the value
must describe the bytes actually stored. The final directory name is only known
after step 7, which is why the rename is step 10.

Any failure **before step 13** leaves the prior generation active. An
unreferenced complete generation may be removed or retained for diagnosis but is
neither accepted nor returned. Temporary and final directories must share a
filesystem; `EXDEV` or equivalent **fails closed**.

### 7.2a Failure after the commit point **[D4]**

Step 13 is the logical commit; a step-14 failure cannot honestly be reported as
"nothing published." Freeze:

```text
failure BEFORE current replacement
    -> previous current remains selected, nothing published

successful current replacement
    -> logical publication COMMITTED

failure while fsyncing the parent directory AFTER replacement
    -> raise a dedicated PublicationDurabilityError
    -> do NOT return RunArtifactResult
    -> do NOT invoke any fallback
    -> the next invocation performs recovery validation (§7.1b) before proceeding
```

The next invocation may accept the new current generation **only if** its
directory, artifact, sidecar and hash-bound pointer all validate.

### 7.3 Sidecar — data **and** metadata hash chain

`parent_artifact_sha256` alone links the payload but does not authenticate the
parent's provenance metadata. The chain must cover both:

```text
generation_id
artifact_sha256
parent_generation_id            (null on clean start)
parent_artifact_sha256          (null on clean start)
parent_sidecar_sha256           (null on clean start)
repository_commit
repository_tree_clean
artifact_schema_version
sidecar_schema_version
encoding_contract_version
canonical_map_hash
row_count
run_id, prng_base, skip_modes_executed
seed_start, seed_count, seed_end_exclusive
raw_candidate_count, l2_winner_count, prior_row_count, final_row_count
created_at
```

Freeze **separate** constants for `artifact_schema_version`,
`sidecar_schema_version` and `encoding_contract_version` — do not overload one
generic `schema_version`.

**`canonical_map_hash`** is SHA-256 over canonical UTF-8 JSON:

```python
json.dumps(
    {"encoding_version": ENCODING_VERSION,
     "prng_type_encoding": PRNG_TYPE_ENCODING,
     "skip_mode_encoding": SKIP_MODE_ENCODING},
    sort_keys=True, separators=(",", ":"), ensure_ascii=True,
).encode("utf-8")
```

(from `utils/prng_encoding.py`, which already defines `ENCODING_VERSION`, the
registry-derived PRNG map and the fixed skip-mode map).

**`sidecar_sha256` is NOT a sidecar field [C1].** A file cannot contain its own
hash: writing the field changes the bytes the field describes. It lives in
`RunArtifactResult` and, for the next generation, in `parent_sidecar_sha256`.
Binding sequence:

```text
1. construct the canonical sidecar payload WITHOUT sidecar_sha256
2. serialize deterministically (below)
3. write and fsync provenance.json
4. compute SHA-256 over the FINAL STORED BYTES
5. return that hash in RunArtifactResult.sidecar_sha256
6. the child generation records it as parent_sidecar_sha256
```

Frozen sidecar serialization — identical form to `canonical_map_hash`:

```python
json.dumps(payload, sort_keys=True, separators=(",", ":"),
           ensure_ascii=True).encode("utf-8")
```

**`skip_modes_executed` comes from run configuration, never inferred from
survivor rows** — an executed mode may legitimately produce zero survivors.

**A certified generation must record `repository_tree_clean=True`.** The first
certified production baseline must not claim a commit SHA while running
uncommitted source.

## 8. Public API — frozen, do not invent

```python
@dataclass(frozen=True)
class RunArtifactResult:
    generation_id: str
    generation_dir: Path
    all_npz_path: Path
    binary_npz_path: Path
    sidecar_path: Path
    artifact_sha256: str
    sidecar_sha256: str
    parent_generation_id: str | None
    parent_artifact_sha256: str | None
    parent_sidecar_sha256: str | None
    repository_commit: str
    repository_tree_clean: bool
    artifact_schema_version: str
    sidecar_schema_version: str
    encoding_contract_version: str
    canonical_map_hash: str
    run_id: str
    prng_base: str
    skip_modes_executed: tuple[str, ...]
    seed_start: int
    seed_count: int
    seed_end_exclusive: int
    raw_candidate_count: int
    l2_winner_count: int
    prior_row_count: int
    final_row_count: int
    created_at: str
    elapsed_seconds: float


def finalize_run(
    candidates: Iterable[Mapping[str, object]],
    *,
    output_root: Path,
    run_id: str,
    prng_base: str,
    skip_modes_executed: Sequence[str],
    seed_start: int,
    seed_count: int,
    repository_commit: str,
    repository_tree_clean: bool,
    prior_generation_dir: Path | None = None,
) -> RunArtifactResult:
    ...
```

`RunArtifactResult` is created **only after** step 12 succeeds. **No partially
successful write may produce one.** `MinerTrialAssembly.binary_npz_path` /
`all_npz_path` remain **deprecated and permanently `None`** — D3.5 must not
populate them.

## 8a. Run-identity wall — current and prior **[C2]**

D3 validates each record internally but does **not** prove a candidate belongs
to the run identity the sidecar will claim. Without this wall the following
passes: `finalize_run(prng_base="java_lcg", ...)` fed a candidate whose
`prng_base` is `xorshift32` — both individually valid, the generation falsely
labelled.

**Current-run wall, enforced before L2:**

```text
prng_base            : a forward, non-hybrid canonical base family
skip_modes_executed  : nonempty, no duplicates, values limited to
                       {constant, variable}, canonical stored order
                       (constant, then variable)
every candidate      : candidate.prng_base == run prng_base
                       candidate.skip_mode in skip_modes_executed
```

A mode listed in `skip_modes_executed` may legitimately produce **zero** rows —
the executed-mode set therefore still comes from configuration, never from
candidate inference.

**Prior wall — decoding successfully is NOT sufficient.** For every prior row:

```text
skip_mode == constant  -> prng_type ID must encode sidecar.prng_base
skip_mode == variable  -> prng_type ID must encode sidecar.prng_base + "_hybrid"
```

Valid-but-inconsistent IDs **fail closed**.

## 8b. Prior selection and recursive chain validation **[C3]**

`prior_generation_dir` alone permits merging against a stale or detached
generation, forking the lineage. Production selection is pinned to the live
pointer:

All four cases are frozen [D2] — a caller must not start a new lineage merely
by omitting the optional argument:

```text
current absent  + prior omitted           -> clean start
current absent  + prior supplied          -> FAIL CLOSED
current present + prior omitted           -> AUTOMATICALLY use current's target
current present + matching prior supplied -> use it
current present + nonmatching prior       -> FAIL CLOSED
```

In every case where `current` is present it must first pass §7.1b
current-generation validation.

Synthetic tests may build a temporary `current` pointer; they must **not**
bypass this rule.

**Recursive validation.** A prior claiming a parent is followed until a
clean-start root is reached. For every link verify:

```text
parent_generation_id matches the directory
parent_artifact_sha256 matches the parent NPZ
parent_sidecar_sha256 matches the parent provenance.json
no missing generation
no repeated generation ID
no cycle
clean-start root has ALL parent_* fields null
```

Without this, the parent hashes are recorded metadata rather than an enforced
provenance chain — which is precisely the property whose absence made the
historical accumulator uncertifiable (Ruling F).

## 9. Prior-generation validation

Before L3, a supplied prior generation must pass **all** of:

```text
generation directory complete
sidecar has the exact required key set and types
artifact hash matches sidecar
sidecar hash matches the supplied parent reference
artifact_schema_version matches
sidecar_schema_version matches
encoding_contract_version matches
canonical_map_hash matches
prng_base matches the current accumulator identity
validate_array_bundle succeeds
seeds strictly increasing and unique
skip_mode IDs decode successfully
prng_type IDs decode successfully
mode/type identity consistency per 8a's prior wall
row_count matches the sidecar
```

**Full numeric-domain validation of the prior [C5].** `validate_array_bundle`
confirms keys, order, dimensions and dtypes; it does **not** establish semantic
domains. The finalizer must additionally verify:

```text
forward_matches, reverse_matches, score  : finite, within [0, 1]
the six count arrays                     : finite, nonnegative, integer-valued
intersection_ratio, survivor_overlap_ratio,
intersection_weight, bidirectional_selectivity
                                         : finite, nonnegative
```

(`bidirectional_selectivity` may exceed 1 — apply only the bounds above.) This
validation lives in `utils/run_finalizer.py`; **do not modify D3.**

Different `skip_modes_executed` sets across generations are **allowed**;
different `prng_base` values are **not**.

## 10. Retire the legacy deduplicator's authority **[Beta §9]**

Verified at `70cd6f0`: `deduplicate_survivors` (`:1684-1700`) selects by
`lexsort((-scores, seeds))` — seed ascending, score descending — with **no
trial_number or mode key**. Its output `bidirectional_deduped` feeds **both**
`bidirectional_survivors.json` (`:1771`) and the NPZ path (`:1878`). If D3.5
replaces only the NPZ path, the JSON and the canonical NPZ can disagree on the
winner for the same seed.

After D3.5:

- the legacy deduplicator must **not** determine any canonical or full
  bidirectional survivor output;
- the canonical NPZ generation is **authoritative**;
- `bidirectional_survivors.json` becomes a **post-success summary**, or is
  generated from the **exact L2 winners** the shared finalizer selected — not
  independently deduplicated;
- its comment must no longer describe it as the canonical Steps 2-6 input.

Forward/reverse diagnostic summaries may remain outside the canonical
publication transaction.

## 11. Fail-closed propagation through the live caller **[B4]**

Verified at `70cd6f0`: the integration block catches `except Exception as
_accum_err:` (`:2004`), prints a warning, runs the
`convert_survivors_to_binary.py` subprocess fallback, and continues toward a
successful return. Only tagged `[S163-KARG-NPZ]` ValueErrors re-raise
(`:1986`). Installing a fail-closed finalizer inside that wrapper would yield:

```text
finalizer rejects prior / publication fails
  -> warning printed
  -> optimization STILL RETURNS SUCCESS      <- violates D3.5
```

**Binding correction — choose one:**

- move the finalizer invocation **outside** the broad non-canonical-output
  `try/except`; **or**
- catch the dedicated finalizer exception **only to add context**, then
  immediately **re-raise**.

The optimizer must not return `results` after canonical finalization fails.

## 12. Gate — `tests/test_s172_phase5_d3_5_finalizer.py`

Independent hand-written oracles throughout.

**L2/L3 semantics (Ruling D T1-T8):**
- **F1** new/new unequal → higher score wins (`trial 8 constant 0.70` vs
  `trial 3 variable 0.80` → trial 3 variable).
- **F2** equal score, different trials **and** modes → **lower trial_number
  wins** (`trial 3 variable 0.80` vs `trial 8 constant 0.80` → trial 3
  variable). *Load-bearing: rejects a global mode-first rule.*
- **F3** equal score, same trial, different modes → constant wins.
- **F4** equal after `float32` conversion → tie; lower trial_number wins.
- **F5** prior/new unequal → new greater replaces prior **in the array domain**.
- **F6** prior/new exact tie → prior retained **byte-for-byte in every one of
  the 22 arrays**.
- **F7** same-trial/same-mode collision → accumulator-consistency error.
- **F8** ordering independence → shuffle repeatedly; identical output.

**Publication and crash safety:**
- **F9** publication failure → prior artifact **and** sidecar byte-identical
  afterward.
- **F10** merge failure → no canonical artifact written; prior untouched.
- **F11** sidecar write failure → no accepted new generation.
- **F12** parent-artifact-hash mismatch → fail closed.
- **F13** prior without sidecar → fail closed.
- **F14** an uncertified historical NPZ is **never** imported — assert both the
  raise **and** that no row of it reaches the output.
- **F15** no fallback subprocess or legacy writer invoked — assert at **source
  level** (the finalizer body references neither `convert_survivors_to_binary`
  nor any subprocess spawn) **and behaviorally** via a spy that fails the gate
  if either is called.

**Coverage and domain:**
- **F16** candidate below `seed_start` → fail.
- **F17** candidate at/above `seed_end_exclusive` → fail.
- **F18** `seed_start + seed_count` overflow → fail. **Mutate the
  implementation to use fixed-width unsigned addition and prove the wrap is
  rejected by the oracle.**
- **F19** declared coverage outside `[0, 2**32)` → fail.
- **F20** candidate seed outside `[0, 2**32)` → fail.
- **F21** valid non-zero range succeeds; coverage recorded in both
  `RunArtifactResult` and the sidecar.

**Contract:**
- **F22** the published artifact passes `validate_array_bundle`, has exactly the
  frozen 22 keys in frozen order/dtypes, and is globally seed-ascending.
- **F23** clean start → `parent_*` fields all `null`; row count == L2 winner
  count.
- **F24** sidecar `artifact_sha256` matches the published file's actual hash.
- **F25** `MinerTrialAssembly` path fields remain `None`.

**REV2 additions:**
- **F27** malformed **losing** raw candidate → finalization fails before L2; no
  generation published.
- **F28** L3 operates on current-winner arrays + prior arrays; **no 22→24
  reconstruction** anywhere (source-level assertion plus a behavioral check that
  retained prior rows are byte-identical to their source arrays).
- **F29** artifact, sidecar and both root aliases become visible through **one**
  `current`-pointer swap.
- **F30** NPZ fsync, sidecar fsync and directory fsync all invoked **before**
  the pointer swap (instrument/spy the fsync calls and assert ordering).
- **F31** finalizer failure **propagates through `optimize_window`** — exception
  reaches the caller, no success result returned, `current` unchanged.
- **F32** `current` remains unchanged after **every** injected failure point
  (parameterize across steps 1-11).
- **F33** parent **sidecar** hash mismatch → fail closed.
- **F34** duplicate or unsorted prior seeds → fail closed.
- **F35** invalid prior `skip_mode` / `prng_type` IDs → fail closed.
- **F36** the legacy score-only deduplicator cannot decide a canonical winner
  (source **and** behavioral).
- **F37** dirty repository state (`repository_tree_clean=False`) cannot produce
  a certified generation.

**REV3 additions:**
- **F38** clean first-generation bootstrap → both aliases created as dangling
  symlinks before the commit; both valid immediately after it.
- **F39** a conflicting **regular file** at a root alias path → fail closed,
  never silently replaced; likewise a symlink pointing at the wrong target.
- **F40** failure after bootstrap but before the `current` commit → only
  harmless dangling aliases remain; no generation accepted.
- **F41** candidate with a `prng_base` differing from the run identity → fail.
- **F42** candidate whose `skip_mode` is absent from `skip_modes_executed` →
  fail. Separately: a mode listed in `skip_modes_executed` producing **zero**
  rows succeeds.
- **F43** prior row with valid-but-identity-inconsistent mode/type IDs (e.g.
  `skip_mode=constant` carrying `<base>_hybrid`) → fail closed.
- **F44** detached prior — `prior_generation_dir` not resolving to `current`'s
  target → fail closed. Also: `current` absent with a prior supplied → fail.
- **F45** recursive chain failures each fail closed: missing ancestor; modified
  ancestor sidecar; generation cycle; repeated generation ID; a clean-start root
  with non-null `parent_*`.
- **F46** malformed prior numeric domains each fail closed: NaN directional
  rate; fractional count; negative metric; `score` above 1.
- **F47** the sidecar payload contains **no** `sidecar_sha256` key, and
  `RunArtifactResult.sidecar_sha256` equals SHA-256 over the final stored bytes
  of `provenance.json`.

**REV3.1 additions:**
- **F48** modify the live `current` generation's `provenance.json` after
  publication → the next finalization **fails closed** (the stored hash no
  longer matches the hash embedded in the pointer target).
- **F49** malformed `current` target each fails closed: wrong embedded sidecar
  hash; a target escaping `generations/`; a non-directory target; a target whose
  parsed `generation_id` disagrees with `sidecar.generation_id`.
- **F50** `current` exists and `prior_generation_dir` is **omitted** → current's
  target is automatically loaded and merged (not a silent clean start).
- **F51** inject failure in the **post-swap** directory fsync (step 14) →
  dedicated `PublicationDurabilityError`; **no** `RunArtifactResult`; no
  fallback; the next invocation performs §7.1b recovery validation and may
  accept the generation only if directory, artifact, sidecar and hash-bound
  pointer all validate.

**F26 mutation proof** — kill each of: generic max-sort over
`prior + raw candidates`; equal-score L3 replacing instead of retaining; L2
mode-before-trial ordering; Python-float instead of `float32` comparison
domain; in-place mutation of the prior; restored fallback-writer call; skipping
`validate_array_bundle` before publication; sidecar written before the artifact
hash; accepting a prior without a sidecar; dropping the local coverage check;
**L3-before-columnization**; **validating only L2 winners**; **separate
artifact/sidecar root-file renames**; **pointer swap before directory fsync**;
**swallowed integration exception**; **omitted parent-sidecar hash**;
**score-only legacy dedup left active**; **`sidecar_sha256` written into the
sidecar payload**; **candidate identity unchecked against run identity**;
**prior accepted without resolving `current`**; **parent chain recorded but not
recursively validated**; **aliases created after the `current` commit instead of
before**; **ancestors validated but the current-tip hash check omitted**;
**generation directory named without the sidecar hash**; **sidecar hashed from
the in-memory buffer instead of the reopened stored bytes**; **prior silently
omitted when `current` exists (clean start instead of merge)**. Report each red
signature and confirm
each kill is attributable to its intended gate.

**Blocking non-regression:** D3.25 13/13, D3 10/10, D3.0 10/10, D2 7/7, D1.1
18/18, D1.0 8/8, D0 12/12, Phase 4 63/63, Phase 3 17/17. Baseline captured green
at `70cd6f0` **before** any edit.

## 13. Stop conditions

- the inline block cannot be replaced without touching the must-not-modify list;
- a legacy backend's candidate shape cannot satisfy §3's validation without
  weakening it;
- an L2/L3 rule as specified contradicts an existing green gate — report the
  contradiction, adjust neither;
- crash-safe publication cannot be achieved on the target filesystem without a
  fallback writer, or temp and final directories cannot share a filesystem;
- any gate passes only by weakening it.

## 14. Report

Diff + status, full command/output evidence, the pre-edit baseline, mutation
evidence with per-mutant red signatures and attribution, and explicit
confirmation that no coverage database was read or written. Then STOP for Team
Alpha review.
