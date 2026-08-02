# S172_D6_2_IMPLEMENTATION_REPORT.md

**S172 Phase-5 D6.2 — 24-field checkpoint, canonical reconciliation, and the finalizer
resume path.**

Report per REV5 §12 plus addendum §6. Implemented on **VM101** as `michael`, venv
`~/venvs/torch`. **Nothing committed, nothing pushed, WATCHER not run.**

**Base:** HEAD `16d42db` (`git pull` → already up to date).
**Governing documents:** `docs/CLAUDE_CODE_INSTRUCTIONS_S172_D6_2_CHECKPOINT_RECONCILIATION.md`
(REV5) as amended by `docs/CLAUDE_CODE_INSTRUCTIONS_S172_D6_2_REV5_BINDING_ADDENDUM.md`
(BINDING — where the two differ, the addendum wins).

**Completion sentinel: `PASS`** for D6.2 itself. Four non-regression suites carry a
single red each, all of which bottom out in Phase-4 Gate 22's untracked-`.py`
sensitivity — the condition the brief predicted and forbade widening (§13.3).

---

## 0. Files changed

| file | status | what |
|---|---|---|
| `utils/checkpoint_d6_2.py` | **NEW** | schema, both digests, CSR sessions, run-id grammar, confinement, `run_context_digest`, nine-row recovery, reconciliation, write transaction |
| `tests/test_s172_d6_2_checkpoint_reconciliation.py` | **NEW** | 29 gates, 377 assertions, 23 mutants |
| `window_optimizer_integration_final.py` | modified | flush rewritten onto the 24-field payload; clear enabled; run context; resume; finalizer input |
| `window_optimizer_bayesian.py` | modified | the two trial-namespace checks |
| `window_optimizer.py` | modified | `--resume-checkpoint`, hop 2 of the operator route, floor forwarded to the study body |
| `agent_manifests/window_optimizer.json` | modified | hop 1: `resume_checkpoint` in `default_params` + `args_map` + `param_docs` |
| `tests/test_s172_d6_1_flush_durability.py` | modified | **ported** to the D6.2 payload — §13.1 |
| `tests/test_s172_phase5_d6_production_adapter.py` | modified | G-FLUSH-CADENCE ported — §13.1 |
| `tests/test_s172_phase5_d3_5_finalizer.py` | modified | one integration-mutant anchor re-anchored — §13.1 |
| `tests/test_s172_process_sharded_import_gate.py` | modified | constant derivation follows an import alias — §13.1 |

---

## 1. §2.2 — the 24 fields AS VERIFIED AT HEAD. Zero drift.

Storage dtypes are **derived**, never transcribed: `CANONICAL_ARRAY_CONTRACT`
(`utils/canonical_arrays.py:98-123`) is walked back through the frozen
`_SOURCE_FIELD_BY_ARRAY` rename map, so the checkpoint stores **record** field names.
The array-domain renames (`seed→seeds`, `*_match_rate→*_matches`) are **not** applied
here, per §2.1.

`G-SCHEMA-24` hand-transcribes REV5 §2.2's table independently and compares. **Result:
the field order equals `CANONICAL_RECORD_FIELDS` at HEAD exactly, and all 22 typed
dtypes match the specified table. No drift to report.**

| # | record field | storage dtype | note |
|---|---|---|---|
| 1 | `seed` | `uint32` | §0 settled |
| 2–4 | `forward_match_rate`, `reverse_match_rate`, `score` | `float32` | unit interval |
| 5–9 | `window_size`, `offset`, `skip_min`, `skip_max`, `skip_range` | `int32` | |
| 10 | `sessions` | **CSR** | §2.4 — not an array |
| 11 | `trial_number` | `int32` | L2 tiebreak **and** replay key |
| 12 | `prng_base` | **derived** | §2.3 — not stored |
| 13 | `skip_mode` | `uint8` | `encode_skip_mode` |
| 14 | `prng_type` | `uint8` | `encode_prng_type` |
| 15–24 | `forward_count` … `intersection_weight` | `float32` | |

24 canonical fields → **22 typed columns + 2 CSR arrays**. Bound sets are reused from
`canonical_arrays.py:229-238` through the imported `_validate_raw_candidates`; there is
no generic `<= 1` ceiling anywhere in D6.2 (`bidirectional_selectivity` may exceed 1).

### `sessions` — CSR encoding and the `[]` round-trip

`sessions_values` (`<U`, flat, record order) + `sessions_offsets` (`int64`, length
`records + 1`). `G-SESSIONS-CASES` round-trips five shapes: `[]`, `[""]`, ordered
multi-session with a repeat, non-ASCII (`"mediodía"`), and an all-empty set.

* **`[]` round-trips as `[]`** — a list, not `None`, not a scalar.
* **all-empty keeps `sessions_values` Unicode** (`dtype.kind == "U"`, shape `(0,)`), so
  it cannot default to `float64`.
* **A scalar string is never a session list, in either direction.** `canonical_sessions`
  fails closed on `"all"` at encode; the decoder cannot express a scalar at all, because
  a CSR row is a slice. `G-CSR-STRICT` rejects eight structural violations
  independently (dtype, 2-D, length, first offset, monotonicity, final offset,
  out-of-range slice, non-Unicode values).

### `prng_base` — derivation in both modes

`constant → prng_type == prng_base`; `variable → prng_type == prng_base + "_hybrid"`.
`derive_prng_base` is the inverse used when rebuilding a record from storage and **fails
closed** on either inconsistent pair (hybrid type in constant mode, non-hybrid type in
variable mode). Both directions are gated.

---

## 2. §3 + addendum §1 — BOTH DIGEST PREIMAGES, EXACTLY AS BUILT

Every array contributes: **domain separator · field name · exact dtype · exact shape ·
contiguous bytes.** Shape is in the preimage on purpose — D6.1's digest
(`:513-528`) omitted it, so two differently-shaped arrays with identical bytes would
have collided. `G-MEMBER-DIGEST-SCOPE` proves shape, dtype and field name are each load-
bearing.

### 2.1 `canonical_state_digest` — shared, content only

Domain separator `s172.d6.2.canonical-state.v1\x00`. Rows are **globally seed-sorted
before the arrays are constructed**. Then, in this **fixed physical order** — which is
*derived* from `CANONICAL_RECORD_FIELDS` by exactly two rules (`sessions` expands in
place to its two CSR arrays; `prng_base` is dropped), not transcribed:

```
 1 seed                      9 skip_range              17 bidirectional_count
 2 forward_match_rate       10 sessions_values         18 intersection_count
 3 reverse_match_rate       11 sessions_offsets        19 intersection_ratio
 4 score                    12 trial_number            20 forward_only_count
 5 window_size              13 skip_mode               21 reverse_only_count
 6 offset                   14 prng_type               22 survivor_overlap_ratio
 7 skip_min                 15 forward_count           23 bidirectional_selectivity
 8 skip_max                 16 reverse_count           24 intersection_weight
```

This is byte-identical to addendum §1's list. **`prng_base` is NOT separately hashed** —
it is reconstructed from `prng_type` + `skip_mode` and adds no information.

It **covers no identity field** (proved: changing `checkpoint_id`/`checkpoint_sequence`
leaves it unchanged). It is stored in **both** members; **B recomputes and verifies it**,
**A binds its marker to it and does not claim to recompute it** — A does not persist the
state.

`G-STATE-ORDER-PHYSICAL` does not read the constant; it **instruments the hasher and
captures the order the code actually emits**.

### 2.2 `member_content_digest` — per member. **THE ADDENDUM §1 DECISION, STATED**

Domain separator `s172.d6.2.member-content.v1\x00`.

**Identity fields ARE included** — per addendum §1, which decided what REV5 §3.2 left to
Alpha. That inclusion covers `canonical_state_digest`. It excludes **only itself**.

The **fixed field order**, never dictionary or NPZ iteration order:

```
identity, in IDENTITY_KEYS order:
   1 checkpoint_schema_version    6 encoding_version
   2 checkpoint_id                7 canonical_map_hash
   3 checkpoint_sequence          8 run_context_digest
   4 run_id                       9 canonical_state_digest
   5 logical_candidate_count     10 member_role
                                 11 member_content_digest  <- EXCLUDED
then the member's payload arrays:
   member A:  seed, score                       (2 arrays)
   member B:  the 24 state arrays, in the fixed physical order above
```

* **Computed LAST**, after every other field is fixed: `build_identity` leaves the field
  empty and `seal_member` fills it — gated.
* **Order independence proved by construction**: `G-DIGEST-PREIMAGE` hands a
  **reversed** identity mapping and a **reversed** payload mapping and requires the
  **identical** digest.
* **Every other identity field is proved to be inside it** — each of the ten is tampered
  independently and must change the digest.
* **The two members' digests DIFFER by design**, and `validate_installed_pair` asserts
  the difference is tolerated: agreement is never required, and *equal* digests are
  treated as an error (one member would not be what it claims).

### 2.3 The order-permutation result

`G-STATE-ORDER-PERMUTATION` (an **order permutation, not a shape mutant**): four
arrival orders of the same four records, three different **flush-boundary splits** of
the same state, and three direct permutations handed straight to
`canonical_state_arrays` — **all ten produce one digest.** The direct arm matters: it
proves the global seed sort is a property of the array construction rather than merely
inherited from `_select_l2_winners` happening to emit ascending seeds (mutant M19 is
credited to exactly that arm).

### 2.4 The identity block — one addition beyond §3.3's table

`member_role` (`marker_stub` / `recovery_payload`) is a D6.2 **addition** to §3.3's
table, and is declared here rather than left implicit. Reason: the two members are
asymmetric by design, so the role belongs inside the *digested* identity rather than
being inferable only from a file name that a careless copy could swap. It is excluded
from `TRANSACTION_INVARIANT_KEYS` (the set a normal installed pair must agree on)
because, like `member_content_digest`, it differs by design.

`canonical_map_hash` is **imported** from `utils.run_finalizer`, never reimplemented.
`G-IDENTITY-BIND` proves a member declaring a different `encoding_version` **or**
`canonical_map_hash` fails **before decoding** — and it proves it against a pair that is
internally consistent and mutually agreeing (both members resealed), so the rejection is
attributable to the identity bind and not to some other wall.

---

## 3. §6 — reconciliation, stated as replay normalization

`reconcile(recovered, new)`:

1. **canonicalize both sides into the checkpoint storage domains** — float32 score and
   rates, Python ints for integer columns, categoricals round-tripped through
   `utils/prng_encoding`, `sessions` a fresh `list[str]`. This is what makes a record
   read off disk and a record just produced in memory bit-comparable;
2. **collapse a bit-identical 24-field replay.** *This is replay normalization, not a
   second winner policy.* It removes an exact duplicate of something already recorded —
   the restart case — and decides nothing between two records that differ;
3. `(seed, trial_number, skip_mode)` matches but **any** canonical field differs →
   `AccumulatorConsistencyError`, naming the differing fields;
4. the remainder goes to **`_select_l2_winners`**, imported from `utils.run_finalizer`.

`_l2_sort_key` and `_select_l2_winners` are **imported, never forked** — `G-ENCODING-
AUTHORITY` walks the AST of `utils/checkpoint_d6_2.py` and requires that neither name
(nor `canonical_map_hash`, nor any of the three §7.2 validators) is redefined there.

### §9.1 duplicate matrix — which path each row exercises

| row | path exercised | result |
|---|---|---|
| bit-identical 24-field replay | step 2, replay normalization | collapsed; idempotent on re-reconcile |
| changed match rates, **same** trial/mode | step 3 | `AccumulatorConsistencyError` |
| changed match rates, **distinct** trial/mode | step 4, `_l2_sort_key` component 1 | higher float32 score wins |
| changed **non-key provenance**, same trial/mode | step 3 | `AccumulatorConsistencyError` |
| constant vs variable **within one trial** | step 4, component 3 | constant wins |
| **different trials** | step 4, component 2 **before** 3 | lower-trial *variable* beats higher-trial *constant* |
| float64-only difference, distinct trials | step 1 → float32 tie → component 2 | lower `trial_number` wins |
| restart-replay duplicate | step 2 | idempotent, no double count |

Every row expecting canonical winner selection uses **distinct `(trial_number,
skip_mode)`**.

---

## 4. §4.2 — ALL THREE RESUME HOPS, with the WATCHER-filter evidence

`G-RESUME-ROUTE` proves the value **arrives**, not merely that the parameter exists.

**Hop 1 — `agent_manifests/window_optimizer.json`.** `default_params` went **24 → 25**
keys with `resume_checkpoint: ""`; `args_map` gained `"resume-checkpoint":
"resume_checkpoint"`; `param_docs` documents it. The gate:

* asserts the key is in `default_params`;
* AST-checks that WATCHER's **live** `allowed_params` is still derived from
  `default_params` (`agents/watcher_agent.py:1551`) — otherwise the gate would be
  measuring the wrong thing;
* drives the **real** merge with a WATCHER-shaped params dict containing both
  `resume_checkpoint` and an undeclared control key: the first survives, **the control
  key is dropped** (so the filter is proven to still filter);
* builds the command through the same `args_map` reverse-lookup the live builder uses
  and requires `--resume-checkpoint gate-run-77` in the argv;
* AST-checks that `window_optimizer.py`'s argparse really declares
  `--resume-checkpoint`, so the emitted command would not abort.

**Hop 2 — the explicit kwargs.** AST over the **live call sites**: both
`run_bayesian_optimization(...)` and `coordinator.optimize_window(...)` pass
`resume_checkpoint=`.

**Hop 3 — the method signature.** AST over the live `def optimize_window(self,
dataset_path, …)`: `resume_checkpoint` is a parameter.

**And it is CONSUMED.** The gate feeds `"not/a/run/id"` through
`_prepare_checkpoint_run_context` and requires the rejection to name the value — a
parameter that is accepted and ignored is a dead chain, which is precisely the
`Advisor → strategy_recommendation.json → WATCHER` pattern this hop list exists for.

**Manifest `version` deliberately left at `1.8.0`**: `tests/test_s172_phase1_scaffolding.py:160`
pins it, D6.2 requires no bump, and bumping it would red a live NR gate for no reason.

---

## 5. §4.1 + addendum §3 — the selector

`resume_checkpoint: str = ''` is a **checkpoint run id**. Empty means no resume.

**Addendum §3 grammar (an additional wall, not a replacement):** the whole string must
match `[A-Za-z0-9._-]+`; `.` and `..` are rejected **explicitly** even though the grammar
admits the characters; non-`str` is rejected. `G-RUNID-GRAMMAR` accepts 3 and rejects 24,
including `/`, `\`, empty, `.`, `..`, spaces, `:`, `;`, `*`, `~`, NUL and newline.

**Realpath and symlink-escape checks remain mandatory.** `G-SELECTOR-CONFINEMENT`
rejects `/etc`, `../../etc`, `..`, `a/b`, and a **symlinked run directory pointing
outside the checkpoint root** (compared on `realpath`, the same way
`_flush_assert_not_alias` already compares for the finalizer aliases).

**No newest-directory discovery, anywhere, at any layer.** The gate walks the AST of
`utils/checkpoint_d6_2.py` and rejects `getmtime`/`getctime` **and any directory
enumeration at all**; in `window_optimizer_integration_final.py` it scopes the rule to
the checkpoint surface and permits exactly one enumeration — the pid-keyed stale-temp
sweep, which collects a crashed run's orphans and decides no path.

The **writer** goes through the same `resolve_checkpoint_dir`, so the grammar and the
containment check apply to it too, not only to an operator-supplied selector.

---

## 6. §4.3 — `run_context_digest`

Versioned canonical JSON — sorted keys, `separators=(",",":")`, `ensure_ascii` — then
SHA-256, mirroring `run_finalizer._canonical_json_bytes` exactly. The preimage:

```json
{"dataset":{"filename":…,"sha256":…,"version_id":…},
 "execution_set_id":null,
 "prng_base":"java_lcg",
 "repository_commit":…,
 "run_context_digest_version":"s172-d6.2-run-context-v1",
 "seed_count":…,"seed_end":…,"seed_start":…,
 "skip_modes_executed":["constant","variable"]}
```

Components, exactly these: **frozen dataset identity and digest** · **repository commit**
(`_repository_state`) · **`prng_base`** · **ordered executed skip modes** ·
**`seed_start` / `seed_count` / `seed_end`** · **execution-set `set_id` or canonical
null**.

* The dataset contributes `version_id`, bare `filename` and `sha256` — taken from the
  P0.5 `FrozenDataset` when this process froze one, else derived from the file. **The
  absolute path is deliberately excluded**: §4.3 excludes a mutable path, and the path is
  not part of what makes two runs the same run — the digest is.
* `execution_set_id` comes from `active_execution_set()` (the **consumer** API; `None`
  reads are counted, per Beta's freeze-after-read retraction) and is the canonical
  `null` when no set is frozen.

**Excluded and gated as excluded:** PID, timestamp, mutable path, newest-directory
inference. `G-CONTEXT-DIGEST` asserts the default run id — which embeds pid and wall
time at `:448` — does not appear in the preimage.

**Every component mutated INDEPENDENTLY.** Eleven mutations
(`dataset.version_id`, `dataset.sha256`, `dataset.filename`, `repository_commit`,
`prng_base`, skip-mode **order**, skip-mode **set**, `seed_start`, `seed_count`,
`seed_end`, `execution_set_id`) each change the digest, all eleven digests are distinct,
and each is **rejected before categorical decoding**. One combined mutation was not
accepted as evidence for the rest.

---

## 7. §4.4 + addendum §4 — ALL FOUR COMBINATION ROWS

| `resume_checkpoint` | `resume_study` | implemented behaviour |
|---|---|---|
| no | no | normal fresh run; recovers nothing, claims no resume provenance |
| no | **yes** | existing Optuna behaviour, untouched; the checkpoint path is not consulted |
| **yes** | **yes** | continues **only** above the recovered trial namespace (two checks, §7.1) |
| **yes** | no | **rejected before optimization**, with a specific error |

**Row 4 is a rejection, and here is why it is not a reconstruct-and-finalize.** REV5
§4.4 permits reconstructing/finalizing the recovered accumulator *"if that surface
exists"*. **It does not exist in this entrypoint**: `optimize_window` has no path that
finalizes without running `optimizer.optimize(...)`, so the specified fallback is
`reject before optimization with a specific error`, which is what is implemented. The
error names the mechanism (a fresh study restarts trial numbering; `trial_number` is part
of the replay key; a restart would **manufacture** the corruption §6.1 raises) and names
the remedy (pass `--resume-study` as well).

**"Independent controls" is honoured in the narrow sense stated:** neither argument
aliases or implicitly enables the other. Nothing in the resume path sets `resume_study`,
and `resume_study` alone never touches the checkpoint.

The gate seeds its checkpoint under the **production** context, so if the guard were
removed the resume would otherwise succeed — that is what makes mutant M22 attributable
to the guard rather than to a context disagreement.

### 7.1 The two trial-namespace checks — with the enqueued case exercised

**Check 1 — pre-flight over the loaded study**, before `study.optimize` is entered
(`window_optimizer_bayesian.py`, immediately before `_trials_to_run`). Scans
**nonterminal** trials (`WAITING`, `RUNNING`) and rejects any at or below the recovered
maximum.

**Check 2 — at the very top of `optuna_objective`**, before any `suggest_*`, before
`objective_function(...)`, before dispatch or candidate admission. Trials are obtained
through `study.optimize(...)`, not ask/tell, so this is the first point `trial.number` is
readable.

**Numbers are never rewritten or offset.** The gate AST-scans
`window_optimizer_bayesian.py` for any assignment to `.number` and requires none.

**The enqueued warm-start case is exercised, not asserted.** The gate builds a real
Optuna study, calls `study.enqueue_trial(...)` — the S166 warm-start seam, confirmed
still present at `window_optimizer_bayesian.py:725` — producing a genuine `WAITING`
trial numbered 0, then **executes the live pre-flight block extracted by AST** against a
recovered maximum of 3. It is rejected. Mutant **M23** narrows the scan to `COMPLETE`
trials and the gate reds. Clean control: the same study with a floor below the enqueued
number passes.

Check 2 is likewise executed as extracted live source: trials 0, 3 and 5 are rejected at
floor 5; 6 passes; and with no resume the guard is inert.

### 7.2 ⚠ FINDING — `trial.number` is **not** the number that lands in the record

Addendum §4 locates the checks on `trial.number`. In this codebase the record's
`trial_number` comes from a **different quantity**: `trial_counter['count']`
(`window_optimizer_integration_final.py`, `test_config`), a process-local 1-based ordinal
that restarts every run — while `trial.number` is the study-scoped 0-based Optuna
number. They are not the same value.

Consequence: **the two specified checks alone would not close the replay-key collision
§4.4 exists to prevent.** A resumed study continues Optuna's numbering, but the record
ordinal would restart at 1 and collide with recovered trial 1 under different canonical
contents.

Both checks are implemented **exactly as specified** (they are binding, and check 1 is
the only thing that catches an enqueued trial). In addition, on a resume the **record
ordinal is initialized from the recovered maximum**, so the first new trial is
`recovered_max + 1`. This is **not** offsetting or rewriting an Optuna trial number — no
Optuna number is read, written or shifted; it is the local record ordinal resuming its
own history instead of pretending the recovered trials never happened — and it does not
restore the optimizer execution cursor. `G-TRIAL-NAMESPACE` pins it.

**This is submitted for Beta's ruling.** If Beta prefers the record's `trial_number` to
become `trial.number` outright, that is a larger change touching D1/D3.25 record
semantics and is not in D6.2's scope.

---

## 8. §5 + addendum §2 — ALL NINE RECOVERY ROWS

Each is its own case in `G-RECOVERY-MATRIX`, and the gate asserts nine **distinct**
outcome labels were reached.

| # | state | implemented behaviour | verified |
|---|---|---|---|
| 1 | A missing **or** unreadable | validate B against the caller-supplied run id + context; recover B; repair pair; next = B+1 | both variants; and B is refused when validated against a different run id |
| 2 | A readable, identity matches, fails its `member_content_digest` | recover B, repair the pair | payload of A tampered; A's sequence does **not** raise the next one (§4.6) |
| 3 | A structurally valid but **conflicts** with B or the requested context | **fail closed** | A resealed with a different `run_id` |
| 4 | A a valid **newer** uncommitted marker, invariants match | discard A, recover B, sequence **above A** | A resealed at seq 7 over B at 1 → recovered id is B's, next = 8, discard recorded |
| 5 | **B valid and newer; A valid but older; invariants agree** | **recover B, repair pair, sequence above B** | A resealed at seq 0 over B at 1 → next = 2, repair required |
| 6 | consistent A/B transaction | recover B; **no repair needed** | the clean control |
| 7 | **B missing or invalid** | **fail closed regardless of A** | both missing and corrupt variants |
| 8 | any context / schema / encoding disagreement | **fail closed** | mismatched `run_context_digest` |
| 9 | equal sequence, different `checkpoint_id` | **fail closed** | A resealed with a different id at the same sequence |

**Row 5 is present and is its own case.** It was not argued away as unreachable.

**Fail-closed means: do NOT clear in-memory state.** `recover_checkpoint` cannot reach
the in-memory list at all — it takes a directory and returns records — and
`G-PRE-CLEAR-WALLS` / `G-CLEAR-SAFE` prove the accumulator survives every rejection on
the flush side.

**Recovery installs and validates a fresh pair before optimization continues**
(`_install_repaired_checkpoint_pair`), on rows 1, 2, 4 and 5; row 6 has nothing to repair.

### §4.6 sequence initialization

The next sequence exceeds the **highest structurally valid** sequence observed in either
member, **including a discarded newer A marker**. `G-SEQUENCE-INIT` proves both halves:
a valid newer A at 12 gives next = 13; the same field forged to 99 in a member that then
fails its own digest gives next = 6 (B at 5) — **a sequence read from an otherwise
invalid member does not count.**

> This second half was a **real defect the gate caught**: the first implementation used
> `max(seq_b, probe_sequence) + 1` on row 2, which would have let a single flipped byte
> in A's stored sequence push the run's numbering anywhere. Fixed before the suite went
> green.

### G-STUB-HONESTY

Member A stores **exactly** `seed`, `score` and its 11 identity fields — verified against
the on-disk archive. **A alone never produces a recovery** (B deleted → fail closed). No
source text in either module describes A as an accumulator backup, and mutant **M20**
(A sealed with B's full payload) reds this gate.

---

## 9. §7 — parity and the three pre-clear walls

### 9.1 Parity properties demonstrated

`G-PARITY` runs two arms over the same 24 raw candidates (cross-trial and cross-mode
duplicates for the same seeds; no same-trial/same-mode duplicate, which would be
corruption):

* **arm A** — everything stays in memory, the pre-D6.2 behaviour;
* **arm B** — checkpoint + **clear** every third record, then the finalizer is fed
  `_checkpoint_finalizer_input(...)`.

Both go through the real `_select_l2_winners → records_to_arrays → _l3_merge →
_sort_by_seed` chain. Demonstrated:

* **exact equality of all 22 canonical arrays**, value-by-value **and** dtype-by-dtype;
* **identical global seed order**;
* **identical canonical artifact digest** — `a61102a2f5ac…` in both arms;
* the arm B run really did clear (its in-memory list is shorter than the raw input) and
  really did checkpoint (sequence ≥ 4), so the arm is not vacuous.

**`raw_candidate_count` differs — 24 vs 4 — by design.** Its meaning is REV5 §4.5's
wording verbatim: *the records supplied to the finalizer by the resumed execution* —
**neither** the original process's raw count **nor** a cumulative count across all
pre-compaction observations.

**NO SIDECAR-FIELD PARITY IS CLAIMED.** Generation id, creation time, elapsed time,
sidecar hash and `raw_candidate_count` differ necessarily; losers are discarded by
construction.

### 9.2 Three protections before any clear

Every **newly observed raw record** passes the walls `finalize_run` applies before L2,
**in its order** — `_validate_raw_candidates` → `_validate_candidate_coverage` →
`_validate_candidate_identity` — all three **imported** from `utils.run_finalizer`, none
duplicated. They run **before reconciliation**, because reconciliation compacts losers
away: `G-PRE-CLEAR-WALLS` feeds a **malformed LOSING candidate** (missing `sessions`,
lower score, higher trial) and requires the run to fail rather than the record to vanish.
Each of the three walls is red independently; the clean control is green and does clear.

The same three walls run over the un-checkpointed tail inside
`_checkpoint_finalizer_input`, so a malformed loser there cannot vanish either.

### 9.3 The clear

`_FLUSH_CLEAR_IN_MEMORY = True`. §8's order is implemented literally: construct
cumulative state → write both temps (open-handle `savez_compressed` + fsync) → **validate
both temps** → replace A → replace B → fsync dir → **validate the installed pair** → only
then clear.

`G-CLEAR-SAFE` injects a failure at the write, at replace #1 and at replace #2, and
requires every candidate retained in all three; mutant **M21** clears between the two
replaces and the gate reds. `G-CADENCE` proves the threshold gate is unchanged (below
→ silent and nothing written; at → one flush; nothing new → silent; next batch → seq 2).

**`savez_compressed` retained** — both members are `ZIP_DEFLATED`. D5 §6.7.A is
untouched and the gate asserts `_assert_stored_uncompressed` is still where it was. Do
not harmonize the two.

---

## 10. §4.5 — durable resumed-run provenance

Persisted to `.s172_checkpoint/<run_id>/resume_provenance.json` — a sibling of the
members inside the run-isolated directory, written with the same fsync-then-atomic-
replace discipline (provenance that a crash can lose is not provenance). Never a
finalizer-owned path.

Carries, at minimum: **recovered checkpoint run id · checkpoint id · checkpoint sequence
· `canonical_state_digest` · recovered canonical-record count**, plus the recovery row,
the next sequence, any discarded newer-A sequence, the run-context digest and its
components, and the two explicit statements below.

It is **also echoed into `bidirectional_survivors.json`**, the finalizer's post-success
summary, so a reader of the certified generation finds it without knowing the checkpoint
directory exists.

---

## 11. Execution cursor — NOT claimed

**D6.2 does not restore the optimizer execution cursor.** The provenance records
`optimizer_execution_cursor_restored: false`; the resume path prints it; the module
header states it in the scope note; and `G-CURSOR-NOT-CLAIMED` asserts no provenance key
implies otherwise. A resumed run recovers the accumulated canonical state and continues
optimization under its own trial namespace — where the *search* had got to remains
entirely Optuna's.

---

## 12. Gate and mutant counts

**D6.2 — `tests/test_s172_d6_2_checkpoint_reconciliation.py`: 29/29 gates green, 377
assertions, 23/23 mutants killed. RESULT: PASS.**

Gates: G-SCHEMA-24 · G-STATE-ORDER-PHYSICAL · G-DIGEST-SPLIT · G-DIGEST-PREIMAGE ·
G-MEMBER-DIGEST-SCOPE · G-STATE-ORDER-PERMUTATION · G-IDENTITY-BIND · G-DUPLICATE-MATRIX
· G-RECOVERY-MATRIX · G-SEQUENCE-INIT · G-STUB-HONESTY · G-STORAGE-DOMAIN · G-CSR-STRICT
· G-SESSIONS-CASES · G-ENCODING-AUTHORITY · G-NO-SYMLINK-COLLISION ·
G-COMPRESSION-CONTRACT · G-PRE-CLEAR-WALLS · G-CLEAR-SAFE · G-CADENCE · G-PARITY ·
G-RUNID-GRAMMAR · G-SELECTOR-CONFINEMENT · G-RESUME-ROUTE · G-COMBINATION-MATRIX ·
G-CONTEXT-DIGEST · G-CURSOR-NOT-CLAIMED · G-RESUME-PROVENANCE · G-TRIAL-NAMESPACE.

Mutants — all 16 from REV5 §9.6 and all 7 added by addendum §5, each under the four-part
kill rule (mutation applies exactly once · the mutated source is the one loaded · the
detector passes against production · the red comes from the injected defect):

M1 inline `score >` · M2 float64 storage domain · M3 trial-number tiebreak dropped ·
M4 same-key collision swallowed · M5 identity field dropped · M6 finalizer-owned path ·
M7 skip_mode map hardcoded · M8 encoding-map hash no longer binds · M9 recover newer A ·
M10 malformed loser vanishes · M11 newest-directory inference · M12 shape omitted ·
M13 member digest in its own preimage · M14 state digest in dict order · M15 `prng_base`
hashed into the state · M16 identity field excluded from the member digest · M17 run id
with `/` · M18 recovery row 5 deleted · M19 order-permutation invariance broken ·
M20 member A given B's payload · M21 clear between the replaces · M22 checkpoint + fresh
study continues · M23 enqueued trial at/below the recovered maximum executes.

**Three mutants were re-credited during the run** because the originally-nominated gate
could not see the defect and the kill would have been vacuous — each is documented inline:
M2 → G-STORAGE-DOMAIN (the frozen `_l2_sort_key` casts to float32 itself, so L2 cannot
see a float64 store); M20 → G-STUB-HONESTY (the member digests still differ, so the split
detector cannot see it); M8 required resealing **both** members (resealing one is caught
by the A-vs-B run-invariant comparison, a different wall).

---

## 13. Non-regression

### 13.1 Suites I had to change, and exactly what changed

Four suites pinned facts D6.2 is **required** to invert (§1, §3.3, §8). Each is
**re-pointed at the replacement property, never relaxed.** These changes need Beta's
ratification — REV5 §10 asks for D6.1 green, and D6.1 as written pins the opposite of
three things REV5 mandates. That conflict is unresolvable as specified.

**`tests/test_s172_d6_1_flush_durability.py` — ported (15/15, 8 mutants).**
Harness: candidates became full canonical 24-field records (the three walls now validate
them); a run context is installed at the single `_run` choke point; member A's seed
column is read as `seed`. Assertions re-pointed:

* `_FLUSH_CLEAR_IN_MEMORY is False` → `is True`, with the reason (REV5 §8);
* `"four-field" in _CHECKPOINT_SCHEMA_VERSION` → `not in`, **plus** an assertion that the
  version is *imported* rather than restated (REV5 §3.3: the marker must change);
* `_flush_inspect_pair` / `_PAIR_*` / `four_field_content_digest` → a thin
  `_pair_status` helper that classifies through the **live**
  `recover_checkpoint`, reimplementing none of the decision. G-TRANSACTION-IDENTITY's
  property is unchanged — Beta's counterexample (identical seed sets, changed score,
  mixed pair) is still constructed and still must not be classified consistent;
* G-CUMULATIVE's higher-score re-observation now carries a **distinct trial number** (one
  replay key with two contents is corruption, not a competition);
* G-VISIBLE-FAILURE's "corrupt prior member warns on stdout" tier is **gone** — D6.2
  never reads member A as data, so there is no failed-merge condition to report. It is
  replaced by a stronger arm: **no run context is a loud, fail-closed condition that
  clears nothing**;
* G-COMMENT-TRUTH: four D6.1 phrases retired as now-false (`NON-AUTHORITATIVE, FOUR-FIELD
  INCREMENTAL SNAPSHOT`, `PROVISIONAL SNAPSHOT MAINTENANCE ONLY`, `THE IN-MEMORY LIST
  REMAINS THE FINALIZER'S AUTHORITATIVE SOURCE`) and asserted **absent**; six required
  phrases retained or added.
* Mutants M2/M3/M5/M6/M7 re-anchored onto the live text (same injected defect, same
  credited gate). **M8 replaced**: its subject (`_flush_inspect_pair` reverted to
  seed-set-only) is in a different module now, so its mutant moved to the D6.2 suite
  (M9, M18) and this slot became "the fail-closed run-context guard removed".

**`tests/test_s172_phase5_d6_production_adapter.py` — G-FLUSH-CADENCE ported (9/9, 16
mutants).** Installs a run context, uses canonical records, and now **pins the clear**
plus the reachability of the cleared candidates through
`_checkpoint_finalizer_input`. D6's own property — one flush per trial, after the append,
with the pre-D6 label — is untouched.

**`tests/test_s172_phase5_d3_5_finalizer.py` — one anchor (60/60).** The F36 integration
mutant wrapped `_raw_candidates_d3_5 = survivor_accumulator['bidirectional']`, which
moved. Same injected defect (a score-only legacy dedup around the finalizer's input),
same gate.

**`tests/test_s172_process_sharded_import_gate.py` — one helper (7/7).** It derives
`_CHECKPOINT_DIRNAME` / `_CHECKPOINT_ROOT_ENV` from Step-1's source by AST and requires a
module-scope assignment; D6.2 **imports** them instead. `_step1_constant` now follows an
import alias to the defining module. Its rule — *read from source, never restate* — is
intact; the value simply comes one hop further along.

### 13.2 Final results

| suite | result |
|---|---|
| **D6.2 (new)** | **29/29, 377 assertions, 23 mutants — PASS** |
| D6.1 (ported) | **15/15**, 8 mutants |
| D6 3.A | **9/9**, 16 mutants |
| D6-threshold | **17/17**, 11 mutants |
| D3.5 | **60/60** |
| D3.25 | **13/13** |
| D3 columnizer | **10/10** |
| D3.0 | **10/10** |
| D0 | **12/12** |
| D4 | **8/8** |
| process_sharded import gate | **7/7** |
| Phase 1 | **6/6** |
| Phase 2 | **6/6** |
| Phase 3 | **17/17** |
| 6-P0.5 dataset authority | **37/37** |
| Chapter 1 P0 corrections | **12/12** |
| `test_prng_encoding` | **8/8** |
| Phase 4 | 62/63 — **Gate 22 only** |
| D1.1 | 17/18 — its NR arm only |
| D1.0 | 7/8 — its NR arm only |
| D2 | 6/7 — its NR arm only |
| D5 | 24/25 — its NR arm only |

**D3.25 and D3.5 are unchanged** (13/13 and 60/60), as §12 asks.

### 13.3 The five reds are one cause, and it is the expected one

Phase-4 Gate 22 builds `changed_py` from `git status --porcelain`, so **any uncommitted
new file reds it**. It names exactly the two new files:

```
AssertionError: unexpected changed .py files:
  {'utils/checkpoint_d6_2.py', 'tests/test_s172_d6_2_checkpoint_reconciliation.py'}
```

D1.1's only red is "Phase 4 exited 1"; D1.0's is the same; D2's is "D1.1 exited 1"; D5's
is "D1.1 exited 1". **Proved, not assumed:** with the two new files temporarily moved
aside, `tests/test_s172_phase4_coordinator.py` returns **63/63 checks green, exit 0**
(files restored immediately afterwards). Nothing else regressed.

**Gate 22 was not edited.** This is expected, is not a regression, and is not a reason to
widen the gate.

**No Wall A/B rerun performed** — Beta stated none is required.

---

## 14. Open items for Team Alpha / Team Beta

1. **§7.2 — `trial.number` vs the record's `trial_number`.** They are different
   quantities here. Both specified checks are implemented; the record ordinal is
   additionally continued from the recovered maximum. Needs a ruling.
2. **§13.1 — four NR suites edited.** REV5 §10 requires D6.1 green while REV5 §1/§3.3/§8
   require the three facts D6.1 pins to become false. The conflict is unresolvable as
   specified; the resolution taken is *port, don't relax*, itemised above.
3. **Member A's seed column is named `seed`, not `seeds`.** Addendum §1 says *"`seeds`
   and `score`"*; REV5 §2.1 says the checkpoint stores **record** field names and *"do
   not apply that rename here"*. I read the addendum as confirming **which two fields**
   rather than re-opening the naming domain, and used `seed` for consistency with member
   B and with §2.1. **This is the one place the two documents could be read as differing
   and I had to choose.** Flagging it explicitly.
4. **`member_role` added to the identity block** beyond §3.3's table (§2.4). Declared,
   with reasoning.
5. **Row 4 of the combination matrix is a rejection**, because the
   reconstruct/finalize-only surface REV5 §4.4 conditions on does not exist in this
   entrypoint (§7).
6. **The trial-namespace floor travels on an attribute seam** (`strategy._resume_trial_floor`),
   the pattern S149/S152 already established for `_survivor_accumulator`, rather than as a
   new `OPTIMIZE_FORWARDED_KWARGS` entry — that tuple is AST-gated against the live
   `strategy.search(...)` call and is also what `strategy_contract_gap` measures the three
   gated strategies against, so widening it would change an unrelated contract. The value
   is read and **enforced**, not advisory, and is gated end to end.

---

## Verification-integrity controls (VIR-1…6)

- **execution proof:** every gate prints its name and a non-trivial assertion count (377
  total); the parity gate reports the compared artifact digests; the recovery and
  combination gates name the row under test.
- **clean control:** an uninterrupted reference run passes every recovery row's healthy
  branch (row 6); a normal fresh run passes with both resume controls empty; every mutant
  carries a positive control requiring its detector to pass against unmutated source.
- **fault-injection control:** 23 mutants, four-part kill rule on each; three re-credited
  where the nominated gate could not see the defect, documented inline and in §12.
- **completion sentinel:** `PASS | FAIL | UNAVAILABLE | INCOMPLETE`; the runner prints
  one and only `PASS` accepts. **D6.2 = PASS.**
- **unavailable-observer behavior:** D6.2 carries **no fleet dependency** — nothing in the
  suite contacts a rig, a GPU or a coordinator, and none of it needs one. No arm is
  `UNAVAILABLE`.
- **audit claim scope:** repo-scoped, this working tree at HEAD `16d42db` plus the
  uncommitted changes listed in §0.
- **searched surfaces:** the tracked repo; the two new untracked files; the live
  `agent_manifests/window_optimizer.json`.
- **unavailable surfaces:** host state on VM101 and the rigs; the live `KERNEL_REGISTRY`
  if changed since HEAD; any deployed uncommitted files outside this tree.

---

**STOP — for Team Alpha review. Not committed, not pushed, WATCHER not run.**
