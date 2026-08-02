# CLAUDE_CODE_INSTRUCTIONS_S172_D6_2_CHECKPOINT_RECONCILIATION.md — REV5

**S172 — `D6.2: 24-field checkpoint, canonical reconciliation, and finalizer resume path.`**

**Base:** HEAD `9470750`. Claude Code on **VM101** as `michael`, venv `~/venvs/torch`. Implement
and iterate; do **NOT** commit, push, or run WATCHER. STOP at the gate.

**Status.** Beta ratified the asymmetric architecture, the digest split, the duplicate matrix, the
recovery direction, the context components, the provenance semantics and the parity definition.
REV5 closes five bounded items. **Beta's sequence: REV5 → implement and certify D6.2 → begin the
Phase-7 soak.**

---

## 0. Settled — do not reopen

- **Member A is a marker / compatibility stub. Member B is the sole recovery payload. Loss or
  corruption of B is unrecoverable.** A must never be described or consumed as an accumulator
  backup. Pair validation is still required before clearing memory.
- **`canonical_state_digest` and `member_content_digest` are separate identities** (§3).
- **`seed` is uint32** — ruled `a63c361` (Seed-Domain v1.1): 48-bit internal state, uint32
  canonical storage, `high16=0` stratum, 1 part in 65,536; at window ≥ 3 all 65,536 high-state
  classes produce distinct sequences, so no reduction exists. **Functional mimicry, not state
  reversal — honest stratum labelling, not a uint64 migration.** The current `uint64` write
  (`:851`) is over-wide storage of already-constrained values.
- **D6.2 does not restore the optimizer execution cursor.**
- **The multi-stripe protocol investigation does not block the Phase-7 soak** and is not part of
  D6.2 certification. Tracked in `docs/BACKLOG.md` §6.

---

## 1. Why this exists

1. The checkpoint persists 4 arrays under `s172-d6.1-four-field-v1` (`:347`); the D3.5 finalizer
   consumes the in-memory list and requires **24** `CANONICAL_RECORD_FIELDS`. 4 cannot make 24.
2. **The S166 guarantee was doubly false** — *"data is safe in NPZ"* justified the clear, but the
   write always failed *and* 4 fields could never have backed it. `_FLUSH_CLEAR_IN_MEMORY = False`
   (`:437`).
3. The current merge policy is `s.get("score", 0.0) > seen[seed].get("score", 0.0)`, not
   `_l2_sort_key` / `_select_l2_winners` (`utils/run_finalizer.py:688-747`, frozen, Ruling D).

---

## 2. Schema

### 2.1 Import, never transcribe

| authority | location |
|---|---|
| `CANONICAL_ARRAY_CONTRACT` | `utils/canonical_arrays.py:98-123` |
| `utils.prng_encoding` codec | `encode_/decode_skip_mode`, `encode_/decode_prng_type` |
| `canonical_map_hash()` | `utils/run_finalizer.py:486` — **exported** |

The checkpoint stores **record** field names. `forward_match_rate` / `reverse_match_rate` are
renamed only in the **array** domain (`canonical_arrays.py:158-162`) — **do not apply that rename
here.**

### 2.2 The 24 fields — RESTORED (Beta correction 5)

Order is `CANONICAL_RECORD_FIELDS` (`utils/canonical_records.py:114-123`). **Verify against HEAD
and report any drift.**

| # | record field | storage dtype | note |
|---|---|---|---|
| 1 | `seed` | `uint32` | §0 — settled |
| 2 | `forward_match_rate` | `float32` | unit interval |
| 3 | `reverse_match_rate` | `float32` | unit interval |
| 4 | `score` | `float32` | the L2 comparison domain |
| 5 | `window_size` | `int32` | |
| 6 | `offset` | `int32` | |
| 7 | `skip_min` | `int32` | |
| 8 | `skip_max` | `int32` | |
| 9 | `skip_range` | `int32` | |
| 10 | `sessions` | **CSR — §2.4** | `list[str]`; not an array |
| 11 | `trial_number` | `int32` | L2 tiebreak **and** replay key |
| 12 | `prng_base` | **derived — §2.3** | not stored |
| 13 | `skip_mode` | `uint8` | `encode_skip_mode` |
| 14 | `prng_type` | `uint8` | `encode_prng_type` |
| 15 | `forward_count` | `float32` | integral-valued, float32 by frozen schema |
| 16 | `reverse_count` | `float32` | ″ |
| 17 | `bidirectional_count` | `float32` | ″ |
| 18 | `intersection_count` | `float32` | ″ — duplicates `bidirectional_count` **deliberately** |
| 19 | `intersection_ratio` | `float32` | nonnegative, **not** ceilinged at 1 |
| 20 | `forward_only_count` | `float32` | integral-valued |
| 21 | `reverse_only_count` | `float32` | integral-valued |
| 22 | `survivor_overlap_ratio` | `float32` | nonnegative |
| 23 | `bidirectional_selectivity` | `float32` | **may legitimately exceed 1** — `len(fwd)/max(len(rev),1)` |
| 24 | `intersection_weight` | `float32` | nonnegative |

**Reuse `canonical_arrays.py:229-238`'s bound sets. No generic `<= 1` ceiling.**

### 2.3 `prng_base` — derived

`skip_mode == "constant" → prng_type == prng_base`; `"variable" → prng_type == prng_base +
"_hybrid"`. Enforced at ingress (`canonical_arrays.py:306-364`). Version-guarded by §3.

### 2.4 `sessions` — CSR

`sessions_values` (`<U`, flat, record order) + `sessions_offsets` (`int64`, length
`records + 1`). `[]` is legal and round-trips as `[]`. **A scalar string is never a session
list** — `canonical_sessions` fails closed on one because the legacy
`getattr(config, 'sessions', 'all')` fallback **fabricated a session name**; the decoder must never
reconstruct a scalar into `[scalar]`.

### 2.5 Other

`savez_compressed` retained (**D5 §6.7.A untouched**). Checkpoint stays under
`.s172_checkpoint/<run_id>/`, never a finalizer-owned path. Exact dtypes.

---

## 3. The two digests — EXACT PREIMAGES (Beta correction 3)

**For every included array, hash: a domain separator, then field name, exact dtype, exact shape,
contiguous bytes.** *(D6.1's digest at `:513-528` omits shape — two differently-shaped arrays with
identical bytes would collide.)*

### 3.1 `canonical_state_digest` — shared

Covers **only the complete canonical record state**, in **deterministic global-seed order**, in
**canonical record-field order** (§2.2), with **`sessions_values` and `sessions_offsets` in fixed
declared positions**.

**It covers no identity field.** Stored in both members. **B recomputes and verifies it; A binds
its marker to it and does not claim to recompute it.**

**Equivalent canonical states assembled in different arrival or flush orders must produce the same
digest.** This is a correctness property, not a nicety — the checkpoint is written after arbitrary
interleavings.

### 3.2 `member_content_digest` — per member

Covers **every persisted field of that member EXCEPT `member_content_digest` itself** — a field
cannot hash itself.

**Decide and state explicitly whether the remaining identity fields are included.** Alpha's
recommendation: **include them**, so the digest detects tampering with the identity block, and
compute it last after every other field is fixed. **State the choice in the report; do not leave
it implicit.**

### 3.3 The identity block

| key | notes |
|---|---|
| `checkpoint_schema_version` | **update** — the four-field marker must change |
| `checkpoint_id` · `checkpoint_sequence` · `run_id` · `logical_candidate_count` | unchanged |
| `encoding_version` | `prng_encoding.ENCODING_VERSION` |
| `canonical_map_hash` | **import from `run_finalizer`; do not reimplement** |
| `run_context_digest` | §4.3 |
| `canonical_state_digest` | §3.1 |
| `member_content_digest` | §3.2 — **expected to DIFFER between members** |

**A normal installed pair agrees on:** checkpoint id, sequence, run id, `run_context_digest`,
schema and encoding identities, `logical_candidate_count`, `canonical_state_digest`. **Their
`member_content_digest` values differ by design** — assert the difference is tolerated and
agreement is **not** required.

**Why `canonical_map_hash` and not the version string:** `tests/test_prng_encoding.py` pins
`len(PRNG_TYPE_ENCODING) == 44`. **Renaming a key preserves the count and `ENCODING_VERSION` while
renumbering every id after it alphabetically.** A member whose `encoding_version` **or**
`canonical_map_hash` differs **fails before decoding.**

---

## 4. Resume (Beta corrections 1 and 2)

### 4.1 The selector is a RUN ID. One API, not two.

`resume_checkpoint: str = ''` — **a checkpoint run id.** Empty means no resume.

Resolved **exclusively** beneath `.s172_checkpoint/<run_id>/`. Required:
- **no absolute paths**;
- **no `..` traversal**;
- **no newest-directory discovery**, anywhere, at any layer;
- **no mutable path in `run_context_digest`**;
- **reject any resolved directory escaping the checkpoint root, including through a symlink**
  (compare `realpath`, as `_flush_assert_not_alias` (`:478`) already does for the finalizer
  aliases).

### 4.2 The operator route — THREE HOPS, all required

**Adding the method parameter alone leaves the resume path dead.** The value must survive:

| # | hop | anchor | if missed |
|---|---|---|---|
| 1 | `agent_manifests/window_optimizer.json` → `default_params` | 24 keys today; add `resume_checkpoint: ""` | **WATCHER's step-scoped filter silently DROPS the key** (`agents/watcher_agent.py:1290-1314` — *"if key in declared"*) |
| 2 | `coordinator.optimize_window(...)` explicit kwargs | `window_optimizer.py:790-810` | never reaches the method |
| 3 | `optimize_window` signature | `window_optimizer_integration_final.py:1695-1710` | rejected as unexpected kwarg |

**This is the `Advisor → strategy_recommendation.json → WATCHER` dead-chain pattern and the TRSE F1
manifest drift.** A gate must prove the value **arrives at the method from a WATCHER-shaped params
dict**, not merely that the parameter exists.

### 4.3 `run_context_digest`

Components, exactly these: frozen dataset identity **and** digest · repository commit
(`_repository_state`, `:101`) · `prng_base` · **ordered** executed skip modes · `seed_start`,
`seed_count`, `seed_end` · execution-set `set_id` (`execution_set.py:299`) **or canonical null when
inapplicable**.

**Encoding:** versioned canonical JSON — sorted keys, fixed separators, `ensure_ascii` — then
SHA-256. `_canonical_json_bytes` (`run_finalizer.py:480`) is the pattern; reuse or mirror exactly.

**Excluded and gated as excluded: PID, timestamp, mutable path, newest-directory inference.**
*(The default run id embeds pid and time at `:448` — that must not leak in.)*

**Rejection happens BEFORE categorical decoding. Gates mutate every component independently** —
one combined mutation is not evidence for six components.

### 4.4 The combination matrix — THE TRIAL-NUMBER COLLISION (Beta correction 1)

**`trial_number` is part of the replay key `(seed, trial_number, skip_mode)`.** A checkpoint-only
resume with a **fresh** Optuna study restarts trial numbering at zero, so a new record can collide
with a recovered one on the same key with different canonical contents — which §6.1 correctly
raises as corruption. **A restart would manufacture corruption.**

| `resume_checkpoint` | `resume_study` | behaviour |
|---|---|---|
| no | no | normal fresh run |
| no | **yes** | existing Optuna behaviour, unchanged |
| **yes** | **yes** | continuing optimization allowed **only after proving the resumed study's next trial number exceeds every recovered `trial_number`** |
| **yes** | no | **must not begin new trials.** May reconstruct/finalize the recovered accumulator **if that surface exists**; otherwise **reject before optimization with a specific error** |

**Never silently offset or rewrite Optuna trial numbers** — that creates a second trial-number
authority and false provenance.

**"Independent controls" means neither argument aliases or implicitly enables the other. It does
NOT mean every combination may continue optimization.**

Gates: the unsafe **checkpoint + fresh study** continuation is **rejected before a new candidate is
admitted**; the matching resumed study **begins above the recovered trial namespace**; **no
same-key collision can be created merely by restart.**

### 4.5 Durable resumed-run provenance

Record durably, at minimum: **recovered checkpoint run id · checkpoint id and sequence ·
`canonical_state_digest` · recovered canonical-record count.** State where it is persisted.

**`raw_candidate_count` — use this wording:** *the records supplied to the finalizer by the resumed
execution.* **Neither** the original process's raw count **nor** a cumulative count across all
pre-compaction observations. **No sidecar-field parity is claimed.**

### 4.6 Sequence initialization

The next `checkpoint_sequence` exceeds **the highest structurally valid sequence observed in either
member, including a discarded newer A marker** (§5). **A sequence extracted from an otherwise
invalid member is NOT a structurally valid sequence.**

---

## 5. Mixed-pair recovery (Beta corrections 3 and 4)

Replacement is **A first, then B** (`:875-876`), so a legitimate crash leaves **A at n+1, B at
n** — A newer **and** unrecoverable. The blanket "higher valid sequence" rule is wrong here.

**Member A cases, disambiguated** — *agreement with a missing A is impossible, so "all invariant
fields agree" cannot be the test for every case*:

| A's state | required behaviour |
|---|---|
| **missing or unreadable** | validate **B against the caller-supplied run id and context**; recover B |
| **readable, identity block matches, but fails its `member_content_digest`** | recover B and **repair the pair** |
| **structurally valid identity block that CONFLICTS with B or the requested context** | **fail closed** |
| **valid newer uncommitted marker, invariants match** | **discard it, recover B, initialize the repaired sequence above A** |

Plus, unchanged:

| state | behaviour |
|---|---|
| consistent A/B transaction | recover B |
| **B missing or invalid** | **fail closed regardless of A** |
| any context / schema / encoding disagreement | **fail closed** |
| equal sequence, different `checkpoint_id` | **fail closed** |

**Fail-closed means: do NOT clear in-memory state.** **Recovery installs and validates a fresh pair
before optimization continues**, sequenced per §4.6.

---

## 6. Reconciliation

### 6.1 Replay normalization, then the canonical authority

1. **canonicalize both records into checkpoint storage domains** (float32 score, uint8 codes, …);
2. **collapse a bit-identical 24-field replay** before winner selection;
3. `(seed, trial_number, skip_mode)` matches but **any canonical field differs** → raise
   `AccumulatorConsistencyError`;
4. pass the remainder to **`_select_l2_winners`**.

**Step 2 is replay normalization, not a second winner policy** — state it that way.

### 6.2 The canonical rule

Highest **float32** score → lowest `trial_number` → constant-before-variable **within one trial
only**. Order-independent. Comparing pre-rounding float64 while storing the rounded value **is the
defect this converts away.**

**Import `_select_l2_winners` / `_l2_sort_key`.** Private, not in `__all__` — import anyway, or
extract to a shared module used by **both** call sites. **Never fork.**

---

## 7. Finalizer parity and pre-clear validation

### 7.1 Parity

**Exact equality of all 22 canonical arrays · identical global seed order · identical canonical
NPZ/artifact digest · truthful resumed-run provenance.** Generation id, creation time, elapsed
time, sidecar hash and `raw_candidate_count` differ **necessarily** — losers are discarded by
construction.

### 7.2 Three protections before any clear

Every newly observed raw record passes the walls `finalize_run` applies before L2
(`utils/run_finalizer.py:1606-1611`, in order):

1. `_validate_raw_candidates` (`:665`) — strict 24-field validation
2. `_validate_candidate_coverage` (`:558`) — declared seed-coverage
3. `_validate_candidate_identity` (`:634`) — run-identity wall

**Reuse or extract; do not duplicate.** All three are private — same rule as §6.2.
`_validate_raw_candidates`'s docstring already states §9's compaction invariant: *a malformed
**losing** candidate must fail the run, not vanish during selection.*

---

## 8. Enabling the clear

```
construct cumulative canonical state
write both temporary artifacts
fsync/close as required
validate both temporary artifacts
replace destination A
replace destination B
validate the installed pair
only then clear the flushed in-memory entries
```

A mutant clearing **between the two replaces** must fail. D6.2 turns `_FLUSH_CLEAR_IN_MEMORY` on
and proves the finalizer still receives complete 24-field input **via the resume path, not the
truncated stump.**

---

## 9. Gates — `tests/test_s172_d6_2_checkpoint_reconciliation.py`

### 9.1 Duplicate matrix — key identity decides

| case | behaviour |
|---|---|
| bit-identical 24-field replay | collapsed by normalization; idempotent |
| changed match rates, **same** trial/mode | **corruption → raises** |
| changed match rates, **distinct** trial/mode | canonical selector |
| changed **non-key provenance**, same trial/mode | **corruption → raises** |
| constant vs variable **within one trial** | **mode tiebreak** |
| **different trials** | **trial-number tiebreak, before mode** |
| float64-only difference, **distinct trial numbers** | float32 tie → lower trial_number wins |
| restart-replay duplicate | idempotent, no double-count |

**Every row expecting canonical winner selection uses distinct `(trial_number, skip_mode)`.**

### 9.2 Digests
- **G-DIGEST-SPLIT** · **G-DIGEST-PREIMAGE:** `member_content_digest` **excludes itself**; the
  stated identity-field decision is what the code does.
- **G-STATE-ORDER-PERMUTATION:** the **same canonical state assembled in permuted arrival/flush
  order yields the identical `canonical_state_digest`.** **An order permutation, not a shape
  mutant.**
- **G-MEMBER-DIGEST-SCOPE:** a shape-only change reds it.
- **G-IDENTITY-BIND:** `encoding_version` and `canonical_map_hash` mismatch **fails before
  decoding**.

### 9.3 Resume
- **G-RESUME-ROUTE:** all three §4.2 hops — including that a **WATCHER-shaped params dict**
  carrying `resume_checkpoint` reaches the method rather than being filtered out.
- **G-SELECTOR-CONFINEMENT:** absolute path · `..` · symlink escape · newest-directory discovery —
  **each rejected**.
- **G-COMBINATION-MATRIX:** all four §4.4 rows; **checkpoint + fresh study rejected before a new
  candidate is admitted**; resumed study **begins above the recovered trial namespace**; **no
  same-key collision creatable by restart**; **Optuna trial numbers never offset or rewritten.**
- **G-CONTEXT-DIGEST:** canonical JSON; **every component mutated independently**, each rejected
  before decoding; PID / timestamp / mutable path **absent** from the preimage.
- **G-CURSOR-NOT-CLAIMED** · **G-RESUME-PROVENANCE** (§4.5's four fields).

### 9.4 Recovery
- **G-RECOVERY-MATRIX:** all four §5 A-cases **plus** the four unchanged rows, each its own case.
- **G-SEQUENCE-INIT:** exceeds the highest **structurally valid** sequence in either member,
  including a discarded newer A; **a sequence read from an otherwise invalid member does not
  count**.
- **G-STUB-HONESTY:** **no path describes or consumes A as an accumulator backup.**

### 9.5 Schema
- **G-STORAGE-DOMAIN** with a **non-float32-representable** input · **G-CSR-STRICT** (int64, 1-D,
  length `records + 1`, first offset zero, monotonic, final offset `== len(sessions_values)`, no
  out-of-range slices) · **G-SESSIONS-CASES** (`[]`, `[""]`, ordered multi-session, non-ASCII,
  all-empty proving `sessions_values` stays Unicode and does not default to float64) ·
  **G-ENCODING-AUTHORITY** (AST: no literal maps, no transcribed dtype table) · **G-PARITY** ·
  **G-PRE-CLEAR-WALLS** · **G-CLEAR-SAFE** · **G-CADENCE** · **G-COMPRESSION-CONTRACT** ·
  **G-NO-SYMLINK-COLLISION.**

### 9.6 Mutants

Four-part kill rule; prove each red comes **from its injected defect**; swap the source every gate
builds from.

Inline `score >` policy · float64 comparison · drop the trial_number tiebreak · swallow the
same-trial/same-mode collision · clear between the two replaces · drop a transaction-identity
field · write to a finalizer-owned path · hardcode the skip_mode map · changed encoding-map hash ·
**recover the newer A instead of B** · **a malformed losing candidate disappearing during
compaction** · reintroduced newest-directory inference · omit shape from the member digest ·
**include `member_content_digest` in its own preimage** · **permute assembly order and expect a
different state digest** · **allow checkpoint + fresh study to continue**.

---

## 10. Non-regression

**No Wall A/B rerun required** (Beta). Capture green before any edit: **D3.25 (13/13), D3.5
(60/60), D6.1, Phase 3 (17/17), Phase 4 (63/63)**, plus D1.1 · D1.0 · D0 · D2 · D3.0 · D3 · D4 ·
D5 · D6 3.A · D6-threshold. After: all green plus D6.2.

**D5 is now 25 gates** following the import gate. **Note:** Phase-4 Gate 22 builds `changed_py`
from `git status --porcelain`, so **any uncommitted new file reds it and propagates to D5's `NR`
arm.** Expect that during development; it is not a regression.

All commands on **VM101**, `source ~/venvs/torch/bin/activate` first.

---

## 11. Scope — do NOT touch

D6 threshold/provenance/residue work; PWC/ZMQ ingress; the D3.25 four-map contract; `TestResult`
shape; D5's artifact contract; `serial_reference` as default. Do not modify `_l2_sort_key`,
`_select_l2_winners`, `CANONICAL_ARRAY_CONTRACT`, `utils/prng_encoding`, `canonical_map_hash`, or
the three §7.2 validators — **reused, not revised.** Do not modify `_RusageChildrenSampler`
(tracked separately in BACKLOG).

---

## 12. Report

§2.2 table as verified at HEAD (report drift) · `sessions` encoding and `[]` round-trip ·
`prng_base` derivation both modes · **both digest preimages exactly, including the identity-field
decision** · the order-permutation result · normalization-then-`_select_l2_winners` stated as
replay normalization · which path each duplicate row exercises · **all three resume hops, with
evidence the value survives WATCHER's filter** · **all four combination-matrix rows** · the
`run_context_digest` components and encoding · **all eight recovery rows** · §7.1 parity properties
demonstrated and the explicit statement that **sidecar-field equality is not claimed** ·
confirmation the execution cursor is not claimed · gate/mutant counts · D3.25 and D3.5 unchanged.
Then STOP for Team Alpha review.

---

## Verification-integrity controls (VIR-1…6)

- **execution proof:** each gate prints its name and a non-trivial assertion count; the parity gate
  reports compared artifact digests; the recovery and combination gates name the row under test.
- **clean control:** an uninterrupted reference run passes every recovery row's healthy branch; a
  normal fresh run passes with both resume controls empty.
- **fault-injection control:** §9.6, four-part kill rule on each.
- **completion sentinel:** `PASS | FAIL | UNAVAILABLE | INCOMPLETE`; only `PASS` accepts.
- **unavailable-observer behavior:** D6.2 should carry **no fleet dependency**; if one appears,
  report it as a finding. With rigs down a fleet-dependent arm is `UNAVAILABLE`, never `PASS`.
- **audit claim scope:** repo-scoped, `9470750`.
- **searched surfaces:** tracked repo at `9470750`.
- **unavailable surfaces:** host state on VM101 and the rigs; uncommitted local modifications; the
  live `KERNEL_REGISTRY` contents if changed since the clone.
