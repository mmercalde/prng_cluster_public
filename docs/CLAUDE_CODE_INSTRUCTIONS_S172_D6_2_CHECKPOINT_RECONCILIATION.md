# CLAUDE_CODE_INSTRUCTIONS_S172_D6_2_CHECKPOINT_RECONCILIATION.md — REV4

**S172 — `D6.2: 24-field checkpoint, canonical reconciliation, and finalizer resume path.`**

**Base:** HEAD `9470750`. Claude Code on **VM101** as `michael`, venv `~/venvs/torch`. Implement
and iterate; do **NOT** commit, push, or run WATCHER. STOP at the gate.

**Status.** Beta **ratified the asymmetric-member architecture** (REV3 §0.2) and **accepted both
owner rulings**. REV4 closes five bounded items Beta identified. Beta's sequence: issue REV4 →
implement and certify D6.2 → begin the Phase-7 soak.

---

## 0. Disposition

### 0.1 What REV4 fixes, and whose defect it was

**Four of the five are consequences of Alpha's own accepted contest.** Removing member-payload
duplication was the right call and Beta ratified it — but it broke the digest model, and REV3 did
not follow through:

REV3 §3 required **one widened content digest covering all persisted fields**, while §4.2 required
**each member to validate its own content digest**. **With asymmetric members those cannot be the
same digest** — member A does not contain member B's payload. Beta caught a contradiction Alpha
introduced.

Likewise the blanket *"select the higher valid sequence"* rule is **wrong under A-first/B-second
replacement**, and `G-RESTART-{A,B,C,D}` was shorthand Alpha carried from REV1 and never expanded.

### 0.2 Settled — do not reopen

- **`seed` is uint32.** Ruled `a63c361` (TB-approved, Seed-Domain v1.1): 48-bit internal state,
  uint32 canonical storage, the `high16=0` stratum, 1 part in 65,536. At window ≥ 3 all 65,536
  high-state classes produce distinct sequences; no reduction exists. **Functional mimicry, not
  state reversal — honest stratum labelling, not a uint64 migration.** The current `uint64` write
  (`:851`) is over-wide storage of already-constrained values. **No decision, no truncation
  exposure.**
- **The multi-stripe protocol observation does not block the Phase-7 soak** and is **not part of
  D6.2 certification.** Tracked in `docs/BACKLOG.md` §6.
- **Asymmetric members are the architecture.** Member A is a transaction marker / compatibility
  stub; **member B is the sole recovery payload; loss or corruption of B is unrecoverable.**
  **A must never be described or consumed as an accumulator backup.** Pair validation is still
  required before clearing memory.

---

## 1. Why this exists

1. The checkpoint persists 4 arrays under `s172-d6.1-four-field-v1` (`:347`); the finalizer
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

The checkpoint stores **record** field names. `forward_match_rate`/`reverse_match_rate` are renamed
only in the **array** domain (`canonical_arrays.py:158-162`) — **do not apply that rename here.**

### 2.2 The 24 fields

As REV2 §2.2, confirmed by Beta against the live authorities. `seed` → uint32 (§0.2); fields 2-9,
11, 15-24 take their contract dtype; `skip_mode`/`prng_type` → uint8 via the codec; `sessions` →
CSR (§2.4); `prng_base` derived (§2.3). **Reuse `canonical_arrays.py:229-238`'s bound sets** — no
generic `<= 1` ceiling, since `bidirectional_selectivity` may legitimately exceed 1.

### 2.3 `prng_base` — derived

`skip_mode == "constant" → prng_type == prng_base`; `"variable" → prng_type == prng_base +
"_hybrid"`. Enforced at ingress (`canonical_arrays.py:306-364`). Version-guarded by §3.

### 2.4 `sessions` — CSR

`sessions_values` (`<U`, flat, in record order) + `sessions_offsets` (`int64`, length
`records + 1`). `[]` is legal and round-trips as `[]`. **A scalar string is never a session list** —
`canonical_sessions` fails closed on one because the legacy `getattr(config, 'sessions', 'all')`
fallback **fabricated a session name**; the decoder must never reconstruct a scalar into
`[scalar]`.

### 2.5 Other

`savez_compressed` retained (**D5 §6.7.A untouched** — a durable run checkpoint is not a worker
transport artifact). Checkpoint stays under `.s172_checkpoint/<run_id>/`, never a finalizer-owned
path. Exact dtypes.

---

## 3. Identity — TWO DIGEST LAYERS (Beta correction 2)

**These are distinct and must never be conflated.**

### Layer 1 — shared canonical-state identity

**`canonical_state_digest`** — computed over the **complete canonical 24-field state**, stored in
**both** members.
- **Member B recomputes and verifies it** against its own payload.
- **Member A merely binds its transaction marker to the expected state.** A cannot recompute it and
  must not pretend to.

### Layer 2 — per-member integrity

**`member_content_digest`** — independently computed over **that member's actual payload**,
covering **each field's name, dtype, shape, and contiguous bytes**.

*(D6.1's digest at `:513-528` covers name, dtype and bytes — **shape is missing**, so two
differently-shaped arrays with identical bytes would collide. Add shape.)*

### The identity block

| key | notes |
|---|---|
| `checkpoint_schema_version` | **update** — the four-field marker must change |
| `checkpoint_id` · `checkpoint_sequence` · `run_id` · `logical_candidate_count` | unchanged |
| `encoding_version` | `prng_encoding.ENCODING_VERSION` |
| `canonical_map_hash` | **import from `run_finalizer`; do not reimplement** |
| `run_context_digest` | §4.2 |
| `canonical_state_digest` | Layer 1 |
| `member_content_digest` | Layer 2 — **expected to DIFFER between members** |

**A normal installed pair must agree on:** checkpoint id, sequence, run id, `run_context_digest`,
schema and encoding identities, `logical_candidate_count`, and `canonical_state_digest`. **Their
`member_content_digest` values differ by design** — a gate must assert that difference is
tolerated and that agreement is *not* required.

**Why `canonical_map_hash` and not the version string alone:** `tests/test_prng_encoding.py` pins
`len(PRNG_TYPE_ENCODING) == 44`. **Renaming or replacing a key preserves the count and preserves
`ENCODING_VERSION` while renumbering every id after it alphabetically.** A member whose
`encoding_version` or `canonical_map_hash` differs **fails before decoding.**

---

## 4. Resume — surface, context, provenance (Beta corrections 5)

### 4.1 The caller surface, named and traced

**Production entrypoint:** `MultiGPUCoordinator.optimize_window()`, monkey-patched by
`add_window_optimizer_to_coordinator()` (`window_optimizer_integration_final.py:1683`), signature
`:1697-1710`. Reached from WATCHER via
`agents/watcher_agent.py --run-pipeline --start-step 1`.

**Add one parameter:** `resume_checkpoint: str = ''` — an explicit checkpoint run id or handle.
Empty means **no resume**. **There is no inference of any kind.**

**⚠ The trap, and it is easy to fall into.** That signature **already carries `resume_study: bool`
and `study_name: str`** (`:1704-1705`) — **Optuna study resume, a different thing.**
`resume_study` restores the optimizer's study; `resume_checkpoint` restores accumulator state.
**D6.2 does NOT restore the optimizer's execution cursor** and must not claim to.

**Required: do not overload, alias, imply, or couple the two.** A gate must assert that
`resume_checkpoint` is independent of `resume_study` — each works with the other set either way —
and the report must state the distinction in those terms.

### 4.2 `run_context_digest` — canonical construction

Components, **exactly these**:
- frozen dataset identity **and** digest;
- repository commit (`_repository_state`, `:101`);
- `prng_base`;
- **ordered** executed skip modes;
- `seed_start`, `seed_count`, `seed_end`;
- execution-set `set_id` (`execution_set.py:299`), or **canonical null when inapplicable**.

**Encoding:** versioned canonical JSON — **sorted keys, fixed separators, `ensure_ascii`** — then
SHA-256. `_canonical_json_bytes` (`run_finalizer.py:480`) is the existing pattern; reuse or mirror
it exactly.

**Excluded, and gated as excluded: PID, timestamp, mutable path, and any newest-directory
inference.** *(The default run id embeds pid and time at `:448` — that value must not leak into
this digest.)*

**Rejection is BEFORE categorical decoding.** A differing context never reaches the decoder.

**Gates must mutate every component independently and prove rejection for each.** One combined
mutation is not evidence for six components.

### 4.3 Durable resumed-run provenance

Record, durably, at minimum: **recovered checkpoint run id · checkpoint id and sequence ·
`canonical_state_digest` · recovered canonical-record count.** State where it is persisted.

**`raw_candidate_count` — the precise definition, and use this wording:** *the records supplied to
the finalizer by the resumed execution.* It is **neither** the original process's raw count **nor**
a cumulative count across all pre-compaction observations. **No sidecar-field parity is claimed.**

### 4.4 Sequence initialization

The next `checkpoint_sequence` must exceed **the highest structurally valid sequence observed in
either member — including a discarded newer A marker** (§5).

---

## 5. Mixed-pair recovery matrix (Beta correction 3)

**Replaces `G-RESTART-{A,B,C,D}` entirely.** The blanket "higher valid sequence" rule is **wrong
here**: replacement is **A first, then B** (`:875-876`), so a legitimate crash leaves **A at n+1
and B at n** — A is newer and unrecoverable.

| state | required behaviour |
|---|---|
| A newer (n+1), B older (n) and valid | **A is an uncommitted marker — discard it, recover B** |
| B newer and valid; A older, missing or corrupt | **recover B**, but only when all invariant context fields agree |
| consistent A/B transaction | **recover B** |
| A invalid, B valid | **recover B and repair the pair** |
| **B missing or invalid** | **fail closed regardless of A** |
| any context / schema / encoding disagreement | **fail closed** |
| equal sequence, different `checkpoint_id` | **fail closed** |

**Fail-closed means: do NOT clear in-memory state.**

**Recovery must install and validate a fresh pair before optimization continues**, with its
sequence set per §4.4.

---

## 6. Reconciliation

### 6.1 Replay normalization, then the canonical authority

Binding order:
1. **canonicalize both records into checkpoint storage domains** (float32 score, uint8 codes, …);
2. **collapse a bit-identical 24-field replay** before winner selection;
3. `(seed, trial_number, skip_mode)` matches but **any canonical field differs** → raise
   `AccumulatorConsistencyError`;
4. pass the remainder to **`_select_l2_winners`**.

**Step 2 is replay normalization, not a second winner policy** — state it that way in the report.

### 6.2 The canonical rule

Highest **float32** score → lowest `trial_number` → constant-before-variable **within one trial
only**. Order-independent. Comparing pre-rounding float64 while storing the rounded value **is the
defect this converts away.**

**Import `_select_l2_winners` / `_l2_sort_key`.** Private, not in `__all__` — import anyway, or
extract to a shared module used by **both** call sites. **Never fork.**

---

## 7. Finalizer parity and pre-clear validation

### 7.1 Parity, accurately

A checkpoint of canonical winners **cannot** reconstruct the original raw candidate list — losers
are discarded by construction. **Parity means:** exact equality of all **22 canonical arrays** ·
**identical global seed order** · **identical canonical NPZ/artifact digest** · **truthful
resumed-run provenance**. Generation id, creation time, elapsed time, sidecar hash and
`raw_candidate_count` differ **necessarily**.

### 7.2 Three protections before any clear

Every newly observed raw record passes the same walls `finalize_run` applies before L2
(`utils/run_finalizer.py:1606-1611`, in order):

1. `_validate_raw_candidates` (`:665`) — strict 24-field validation
2. `_validate_candidate_coverage` (`:558`) — declared seed-coverage
3. `_validate_candidate_identity` (`:634`) — run-identity wall

**Reuse or extract; do not duplicate.** All three are private — same rule as §6.2.

`_validate_raw_candidates`'s docstring already states §8's compaction invariant: *a malformed
**losing** candidate must fail the run, not vanish during selection.* **Existing precedent, not a
new demand.**

---

## 8. Enabling the clear

Only after §2-§7 hold. Beta's ordering, verbatim:

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

### 9.1 Duplicate matrix (Beta correction 4) — key identity decides

| case | required behaviour |
|---|---|
| bit-identical 24-field replay | collapsed by normalization; idempotent |
| changed match rates, **same** trial/mode | **corruption → raises** |
| changed match rates, **distinct** trial/mode | **canonical selector** |
| changed **non-key provenance**, same trial/mode | **corruption → raises** |
| constant vs variable **within one trial** | **mode tiebreak** |
| **different trials** | **trial-number tiebreak, before mode** |
| float64-only difference, **distinct trial numbers** | float32 tie → lower trial_number wins |
| restart-replay duplicate | idempotent, no double-count |

**Any row expecting canonical winner selection must use distinct `(trial_number, skip_mode)`
identities.** With the same trial and mode, a changed rate or provenance field **is** corruption.
This removes the last opportunity to recreate REV2's contradiction.

### 9.2 Digest and identity
- **G-DIGEST-SPLIT:** `canonical_state_digest` agrees across members; **`member_content_digest`
  differs and that difference is tolerated**; B recomputes and verifies the state digest; **A does
  not claim to.**
- **G-MEMBER-DIGEST-SCOPE:** the per-member digest covers **name, dtype, shape and contiguous
  bytes** — prove a shape-only change reds it.
- **G-IDENTITY-BIND:** `encoding_version` and `canonical_map_hash` both present; a mismatch on
  either **fails before decoding**.

### 9.3 Resume
- **G-RESUME-SURFACE:** `resume_checkpoint` exists on `optimize_window`, defaults to no-resume, and
  is **independent of `resume_study`** (both combinations exercised).
- **G-NO-INFERENCE:** **no newest-directory inference anywhere** (AST + runtime).
- **G-CONTEXT-DIGEST:** canonical JSON, sorted keys, fixed separators; **every component mutated
  independently**, each proving rejection **before decoding**; PID / timestamp / mutable path
  **absent** from the digest input.
- **G-CURSOR-NOT-CLAIMED:** nothing asserts optimizer-cursor restoration.
- **G-RESUME-PROVENANCE:** the four §4.3 fields durable and correct.

### 9.4 Recovery
- **G-MIXED-PAIR-MATRIX:** all seven §5 rows, each as its own case.
- **G-SEQUENCE-INIT:** next sequence exceeds the highest structurally valid sequence seen in either
  member, **including a discarded newer A marker**.
- **G-STUB-HONESTY:** A documented as a marker/stub; **no path describes or consumes it as an
  accumulator backup.**

### 9.5 Schema
- **G-STORAGE-DOMAIN:** equality **in declared storage domains**, with a deliberately
  **non-float32-representable** input proving the expected canonicalization.
- **G-CSR-STRICT:** int64, 1-D offsets, length `records + 1`, first offset zero, monotonic, final
  offset `== len(sessions_values)`, no out-of-range slices.
- **G-SESSIONS-CASES:** `[]`, `[""]`, ordered multi-session, non-ASCII, and **all-empty proving
  `sessions_values` stays a Unicode array rather than defaulting to float64.**
- **G-ENCODING-AUTHORITY:** AST — no literal skip_mode map, no literal prng_type map, no
  transcribed dtype table in the checkpoint path.
- **G-PARITY** (§7.1) · **G-PRE-CLEAR-WALLS** (§7.2) · **G-CLEAR-SAFE · G-CADENCE ·
  G-COMPRESSION-CONTRACT · G-NO-SYMLINK-COLLISION.**

### 9.6 Mutants

Four-part kill rule; prove each red comes **from its injected defect**; swap the source every gate
builds from (the D6.1 vacuous-mutant lesson).

Inline `score >` policy · float64 comparison · drop the trial_number tiebreak · swallow the
same-trial/same-mode collision · clear between the two replaces · drop a transaction-identity
field · write to a finalizer-owned path · hardcode the skip_mode map · **changed encoding-map
hash** · **incorrect mixed-pair freshness selection (recover the newer A)** · **a malformed losing
candidate disappearing during checkpoint compaction** · reintroduced newest-directory inference ·
**omit shape from the member digest.**

---

## 10. Non-regression

Beta confirmed **no Wall A/B rerun is required.** Capture green before any edit: **D3.25 (13/13),
D3.5 (60/60), D6.1, Phase 3 (17/17), Phase 4 (63/63)**, plus D1.1 · D1.0 · D0 · D2 · D3.0 · D3 ·
D4 · D5 · D6 3.A · D6-threshold. After: all green plus D6.2.

All commands on **VM101**, `source ~/venvs/torch/bin/activate` first.

---

## 11. Scope — do NOT touch

D6 threshold/provenance/residue work; PWC/ZMQ ingress; the D3.25 four-map contract; `TestResult`
shape; D5's artifact contract; `serial_reference` as default. Do not modify `_l2_sort_key`,
`_select_l2_winners`, `CANONICAL_ARRAY_CONTRACT`, `utils/prng_encoding`, `canonical_map_hash`, or
the three §7.2 validators — **reused, not revised.**

---

## 12. Report

The §2.2 table as verified at HEAD (report drift) · the `sessions` encoding and its `[]`
round-trip · `prng_base` derivation both modes · **the two digest layers and where each is computed
and verified** · the normalization-then-`_select_l2_winners` path, stated as replay normalization
· which path each duplicate row exercises · **the `resume_checkpoint` surface and its independence
from `resume_study`** · the exact `run_context_digest` component list and encoding · **all seven
mixed-pair rows** · the §7.1 parity properties demonstrated and the explicit statement that
**sidecar-field equality is not claimed** · confirmation the execution cursor is not claimed ·
gate/mutant counts · D3.25 and D3.5 unchanged. Then STOP for Team Alpha review.

---

## Verification-integrity controls (VIR-1…6)

- **execution proof:** each gate prints its name and a non-trivial assertion count; the parity gate
  reports compared artifact digests, not a boolean; the mixed-pair gate names the row it is on.
- **clean control:** an uninterrupted reference run passes every recovery row's healthy branch.
- **fault-injection control:** §9.6, four-part kill rule on each.
- **completion sentinel:** `PASS | FAIL | UNAVAILABLE | INCOMPLETE`; only `PASS` accepts.
- **unavailable-observer behavior:** D6.2 should carry **no fleet dependency**; if one appears,
  report it as a finding rather than working around it. With rigs down a fleet-dependent arm is
  `UNAVAILABLE`, never `PASS`.
- **audit claim scope:** repo-scoped, `9470750`.
- **searched surfaces:** tracked repo at `9470750`.
- **unavailable surfaces:** host state on VM101 and the rigs; uncommitted local modifications; the
  live `KERNEL_REGISTRY` contents if changed since the clone.
