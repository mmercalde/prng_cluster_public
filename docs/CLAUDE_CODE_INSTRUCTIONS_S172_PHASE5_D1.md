# CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D1.md — REV5

**S172 RANGE-MINER — Phase 5, Deliverable D1: Phase-4 corrections (D1.0:
workflow + terminal-race) + shared four-population assembly engine + concrete
`Phase5Sink` (D1.1)**

**REV5 changelog (APPROVED by Team Beta — first D1 revision authorized for
implementation; incorporates the two approval-note implementation
requirements: W6 runs in an isolated, timeout-terminable subprocess; the lock
prohibition is specifically the coordinator `_lifecycle_lock`, not the
ledger's internal `_write_lock`):** absorbs the Team Beta REV4 review (REJECT — the REV2/REV4
lock-based abort correction deadlocks the existing retry matrix):
**[TB-D1-DL]** `handle_stripe_failure` holds `_lifecycle_lock` (:2851) through
`_handle_stripe_failure_locked`, whose non-retryable / constant-phase /
no-alternate-hybrid / second-hybrid branches call `fail_trial` (:2966) =
`submit_abort(...).result()` — a synchronous wait on the cleanup-executor
thread, which under the REV4 shape would block forever on the same
non-reentrant-across-threads lock. The fix is now **CAS-result disambiguation
plus terminal-state re-read** using the ledger's existing atomic
`state='running'` transitions — NO `_lifecycle_lock` in `abort_trial`;
**[TB-D1-W5R2]** W5-R revised: commit completes while abort is paused after
its stale read (no lock-blocking expectation); **[TB-D1-W6]** new W6 gate
proving retry-matrix-triggered synchronous abort cannot deadlock. All other
REV4 content approved and retained.

**REV4 changelog:** **[TB-D1-W5R]** deterministic stale-read race gate;
**[TB-D1-G13C]** G13 dual-copy provenance mutation; **[TB-D1-G16C]** G16
final state always tombstoned. **REV3:** **[TB-D1-C1]** terminal-race
correction joins D1.0; **[TB-D1-GC1/GC2]** concurrency wording + G4 split;
**[TB-D1-PV]** container validation. Prior marks **[TB-D1-1..7]**,
**[TB-D1-DEC1/2]**, **[TB-D1-B1..B5]**, **[TB-D1-API]** retained.

**Audience:** Claude Code, running on VM 101 (`michael@192.168.3.177`), inside
`~/distributed_prng_analysis`. You write and iterate the implementation and its
harness here. You do **not** commit, push, or run WATCHER. When each
deliverable's gate is green you STOP and report; Team Alpha reviews the actual
files against live source, Team Beta reviews, Michael commits + dual-pushes.

**Frozen against repo HEAD `7f2a010`** (D0 at `4c697a8`). D1.1 implements
against the post-D1.0 coordinator state. Spec authority:
`docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5.md` §D1,
`docs/PROPOSAL_S172_RANGE_MINER_v1_4_5.md` §6.7.B/C, frozen `v1_4_4`
§4.2/§4.3/§6.8, and all three Team Beta D1 rulings absorbed herein.

---

## 0. Non-negotiable working rules

1. **Read live source before every claim.** Every line cite here was verified
   against HEAD `7f2a010` — re-verify each before depending on it. Both D1.0
   defects sat in plain sight (`workflow_stages_for` :3591-3600; `abort_trial`'s
   handling of a `False` from `mark_trial_aborted` :2997-3005) and were found
   only by auditing behavior against contract, not by reading lines in
   isolation. Audit every claim against what production actually does.
2. **Each gate must FAIL on absent or wrong behavior.** Construct the
   adversarial case, not the happy path.
3. **No test-only shortcuts.** The happy path of each harness drives the REAL
   lifecycle: real staged spool files, real `Phase5Sink.publish_shard` from the
   real publish surface (`publish_attempt`, coordinator:1913), real
   `coordinator.commit_trial` / `coordinator.abort_trial` reaching the sink.
   Direct sink-method calls are permitted ONLY for defense-in-depth gates the
   post-D1.0 coordinator cannot produce, and each must be labeled a **direct
   sink invariant-break probe** in a comment **[TB-D1-GC1]**.
4. **STOP and report at each gate.** D1.0 and D1.1 are separately gated; do not
   chain. Do not commit, push, or invoke WATCHER.
5. **Do not invent field semantics.** Every canonical-record field reproduces
   the live Step-1 meaning exactly (§6). If a required value cannot be produced
   without inventing a formula → **STOP** (proposal-amendment territory).
6. **`utils/prng_encoding.py` is the single source of truth** for PRNG/skip
   encoding validation. D1 stores **strings** in records (numeric encoding is
   D3) but validates them through the canonical module (§5.4).

---

## 1. Scope — two staged deliverables, four files

**D1.0 (first, separately gated and reviewed) [TB-D1-B1, TB-D1-C1]:**
- modify `miner/range_miner_coordinator.py` — exactly TWO narrow corrections:
  1. `workflow_stages_for` (:3591-3600) — §2.1;
  2. `abort_trial` terminal decision (:2974-3005) — §2.2;
  nothing else in the coordinator changes;
- create `tests/test_s172_phase5_d1_workflow.py` — the D1.0 gates W1-W6
  (standalone; must run before the writer module exists — do not import it).

**D1.1 (only after D1.0 review):**
- create `miner/range_miner_npz_writer.py` — `MinerTrialAssembly` (§7), the
  backend-independent assembly engine (§5), `AssemblingPhase5Sink` implementing
  the coordinator's `Phase5Sink` protocol (coordinator:1287-1306) (§4), the D1
  exception types (§8);
- create `tests/test_s172_phase5_d1_engine.py` — Gate D1.1 (§9). Reuse the
  real-lifecycle scaffolding pattern of `tests/test_s172_phase5_d0.py`.

**Explicitly OUT of scope (Beta-bound):**
- any coordinator change beyond the two D1.0 corrections;
- any change to `window_optimizer_integration_final.py` (production wiring +
  adapter = **D6**);
- NPZ writing of any kind — no `np.savez*`, no 22-array construction (**D3/D4**);
- backend parallelism / process pools (**D5**);
- physical temp-file cleanup + benchmark instrumentation (**D7**);
- durable cross-process sink reconstruction (**not provided by D1** — §4.0
  [TB-D1-B2]);
- any new locking system in the coordinator — AND no `_lifecycle_lock`
  acquisition inside `abort_trial`: the terminal-race fix is ledger-CAS +
  re-read only (§2.2) **[TB-D1-DL]**;
- commit / push / WATCHER (deny-ruled on 101; do not work around).

---

## 2. D1.0 — two narrow Phase-4 corrections

### 2.1 Correction 1 — constant runs are bidirectional  **[TB-D1-B1]**

**Defect (verified :3591-3600):** `workflow_stages_for(base, False)` returns
`[(base, 1)]` — forward-constant ONLY. A real `test_both_modes=False` trial
executes no P2 reverse pass and can never produce a constant bidirectional
population. Legacy Step 1 ("PART 1: CONSTANT SKIP TEST (Always runs)",
`window_optimizer_integration_final.py:561+`) ALWAYS runs forward **and**
reverse for constant; only variable is gated by `test_both_modes`. The
production `_use_miner` call site passes the real `test_both_modes` through, so
this is a producer defect.

**Fix (this exact shape):**

```python
if test_both_modes:
    return [(prng_base, 1),
            (f"{prng_base}_reverse", 2),
            (f"{prng_base}_hybrid", 3),
            (f"{prng_base}_hybrid_reverse", 4)]
return [(prng_base, 1),
        (f"{prng_base}_reverse", 2)]
```

Update the docstring ("constant is always bidirectional; hybrid pair only when
test_both_modes"). No other change to this function.

### 2.2 Correction 2 — abort/commit terminal-race fencing  **[TB-D1-C1]**

**Defect (verified):** `commit_trial` makes its durable terminal decision under
`_lifecycle_lock` (:2936-2949) then delivers to the sink outside it.
`abort_trial` (:2974-3005) uses NO lock: it pre-reads state (:2989-2990),
early-returns on `committed` only from that **stale** read, then calls
`mark_trial_aborted` — whose `UPDATE … WHERE state='running'` (:887-894)
correctly returns `False` when a concurrent commit already won — **but a
`False` is then treated as "already aborted, retry the discharge"**: the sink
`abort_trial` is called and staged files are deleted regardless (:3001-3009).
The ledger's mutual exclusivity (both transitions `WHERE state='running'`,
:890/:929) is sound; the defect is `abort_trial` conflating
`False-because-aborted` (must retry the discharge idempotently) with
`False-because-committed` (must refuse).

Failure outcome A (commit's sink delivery first): sink installs the assembly,
abort then clears + tombstones it, commit delivery records `"done"` → a
committed trial with no retrievable result. Failure outcome B (abort's sink
delivery first): sink tombstoned, staged spools deleted, commit delivery
records `"failed"` → redelivery can never recover. Both violate terminal
mutual exclusivity, commit retryability, and L7 staged-file ownership.

**Fix — CAS-result disambiguation plus terminal-state re-read. NO
`_lifecycle_lock` acquisition in `abort_trial` (Team Beta binding
[TB-D1-DL]), this exact shape:**

```python
self.ledger.create_trial(run_id, -1, now)
trial = self.ledger.get_trial(run_id)

if trial is not None and trial["state"] == "committed":
    return {"event": None, "cleanup": "refused", "first": False,
            "refused": "already_committed"}

abort_event_id = f"{run_id}:abort"

if trial is not None and trial["state"] == "aborted":
    first = False
else:
    first = self.ledger.mark_trial_aborted(run_id, abort_event_id, now)
    if not first:
        # The initial read may now be stale. Determine which terminal
        # transition actually won the atomic state='running' race.
        trial = self.ledger.get_trial(run_id)
        if trial is not None and trial["state"] == "committed":
            return {"event": None, "cleanup": "refused", "first": False,
                    "refused": "already_committed"}
        if trial is None or trial["state"] != "aborted":
            raise RuntimeError(
                f"unexpected terminal transition for {run_id!r}")

if first:
    self.ledger.cancel_active_stripes(run_id)
```

The sink discharge and everything after it are unchanged.

**Why no lock — the deadlock [TB-D1-DL], verified:**
`handle_stripe_failure` (:2835) acquires `_lifecycle_lock` (:2851) and holds
it through `_handle_stripe_failure_locked`, whose branches (non-retryable,
constant-phase, no-alternate-hybrid-worker, second-hybrid-failure) call
`fail_trial` (:2966) — which is `submit_abort(...).result()`: a synchronous
wait for the **cleanup-executor thread** to finish `abort_trial`. Under the
REV4 locked shape, that thread would block acquiring `_lifecycle_lock` while
the dispatch/staging thread cannot release it until `.result()` returns.
`threading.RLock` is reentrant only for the SAME thread — the executor thread
cannot inherit the caller's ownership. Permanent deadlock, not a timing race.

**Why the CAS fix is race-safe:** the ledger already provides terminal mutual
exclusion — both `mark_trial_aborted` and `mark_trial_committed` are atomic
`UPDATE … WHERE state='running'` transitions (:887-894, :922-933). The defect
was never the absence of an atomic terminal decision; it was `abort_trial`
failing to distinguish WHY that atomic update returned `False`. The commit
side already handles losing the CAS: `if not mark_trial_committed(...):
raise TrialAborted` (:2945-2947). With the re-read added on the abort side:

```text
commit wins:  abort reads running -> commit atomically running->committed ->
              abort's mark_trial_aborted() returns False -> abort re-reads
              committed -> refuses without calling sink.abort_trial()
abort wins:   abort atomically running->aborted -> commit's
              mark_trial_committed() fails -> commit raises TrialAborted
prior abort:  abort sees aborted -> first=False -> sink discharge retried
              idempotently
```

All three required behaviors hold: commit/abort terminal exclusivity; retry
of an incomplete abort discharge; **no deadlock when `fail_trial` is invoked
while `_lifecycle_lock` is held.** The refused-return shape reuses the
existing one (:2991-2995). Docstring updated to describe the CAS + re-read
decision. No other change to this method; no new locking system anywhere; no
coordinator `_lifecycle_lock` acquisition inside `abort_trial` (the ledger
methods legitimately acquire their own internal `_write_lock` — the prohibited
operation is specifically the coordinator lifecycle lock from the
cleanup-executor thread).

### 2.3 Gate D1.0 (`tests/test_s172_phase5_d1_workflow.py`)

Real lifecycle, recording sink, each check failing on the wrong behavior:

- **W1:** `workflow_stages_for(base, False) == [(base,1), (f"{base}_reverse",2)]`
  and the `True` branch unchanged (4 stages) — asserted for at least two
  distinct bases (never hardcode one family).
- **W2 producer gate:** drive a real `test_both_modes=False` run through the
  real serve/publish surface → the sink receives P1 **and** P2 manifests
  (explicit `forward/constant` and `reverse/constant` identities) and **zero**
  P3/P4 manifests.
- **W3:** commit reaches the sink with a complete constant bidirectional
  input: both directional populations non-empty for a fixture whose seed
  populations intersect (manifest-coverage assertion; full assembly is D1.1).
- **W4 non-regression:** Phase 4 **63/63** (including the 4-stage assertion at
  test_s172_phase4_coordinator.py:2287-2288, unaffected), Phase 3 **17/17**,
  D0 harness green.
- **W5 terminal-race fencing [TB-D1-C1-W5, TB-D1-W5R]** — three subcases,
  real coordinator, events-based synchronization, **no sleeps**. Note the
  discrimination boundary: W5-A and W5-B pause AFTER a durable terminal
  transition, so they pass even the defective pre-fix coordinator (its stale
  pre-read then correctly sees the terminal state, and pre-fix commit's
  decision already sits under `_lifecycle_lock`). They pin the ordinary
  orderings. **Only W5-R proves the race correction itself.**
  - **W5-A ordinary commit-first ordering:**
    1. start real `coordinator.commit_trial()` (worker thread);
    2. pause inside the recording sink's `commit_trial()` — the durable
       committed transition has already occurred;
    3. while paused, call real `coordinator.abort_trial()`;
    4. assert: abort returns `cleanup == "refused"` with
       `refused == "already_committed"`; sink `abort_trial()` call count is
       **zero**; **no staged spool was deleted**; after releasing the pause,
       commit delivery becomes `"done"`.
  - **W5-B ordinary abort-first ordering:**
    1. start real `coordinator.abort_trial()` (worker thread);
    2. pause inside the recording sink's `abort_trial()` — the durable
       aborted transition has already occurred;
    3. call real `coordinator.commit_trial()`;
    4. assert: commit raises `TrialAborted`; sink `commit_trial()` call count
       is **zero**; after releasing the pause, abort cleanup completes
       normally (`cleanup == "done"`).
  - **W5-R stale-read race exclusion — the discriminating gate [TB-D1-W5R2]:**

    *Instrumentation:* temporarily wrap `ledger.get_trial` so that:
    - only the abort worker thread is intercepted;
    - only its FIRST relevant `get_trial(run_id)` call is intercepted;
    - the real method is called first;
    - assert the returned state is `running`;
    - signal event `abort_read_running`;
    - block on event `release_abort_read`;
    - then return the already-read `running` row.

    This reproduces the exact vulnerable interval: *abort has observed
    `running` but has not yet called `mark_trial_aborted`.*

    *Sequence (CAS semantics — commit is NOT expected to block):*
    1. create a running trial with staged files and a pausable recording sink;
    2. start real `coordinator.abort_trial()` in thread A;
    3. wait on `abort_read_running` (abort has read `running` and is paused);
    4. start real `coordinator.commit_trial()` in thread B;
    5. while abort remains paused, assert:
       - commit **completes** with `delivery == "done"`;
       - final ledger state is `committed`;
       - sink `commit_trial()` call count is exactly **one**;
       - sink `abort_trial()` call count remains **zero**;
    6. release `release_abort_read`;
    7. abort receives its stale `running` row and calls
       `mark_trial_aborted()`;
    8. that update returns `False` (commit won the atomic
       `state='running'` race);
    9. the corrected abort **re-reads** the ledger, sees `committed`, and
       refuses;
    10. assert: abort returns `cleanup == "refused"` with
        `refused == "already_committed"`; sink `abort_trial()` call count is
        **zero**; **no staged spool was deleted**; the committed sink result
        remains intact.

    *Why this fails against pre-fix HEAD:* after `mark_trial_aborted()`
    returns `False`, the defective code does not inspect the winning terminal
    state — it proceeds to `sink.abort_trial()` and staged-file deletion.
    Pre-fix HEAD therefore fires BOTH sink terminal deliveries for one run
    (and destroys the committed result); the corrected implementation fires
    only commit's. That is the exact defect §2.2 removes — proven without any
    lock acquisition.
- **W6 — locked retry-matrix failure reaches synchronous abort without
  deadlock [TB-D1-W6]:**
  1. create a running trial with a phase-1 or phase-2 stripe and staged
     files; attach a recording sink whose `abort_trial()` returns normally;
  2. invoke real
     `coordinator.handle_stripe_failure(run_id, stripe_id, retryable=False,
     eligible_workers=[])` in a worker thread / future with a **bounded
     completion timeout** (the timeout is the failure detector, not a
     synchronization mechanism);
  3. assert: the call **completes** rather than hanging; the returned action
     is `fail_trial`; final trial state is `aborted`; sink `abort_trial()`
     was called exactly **once**; abort cleanup status is complete
     (`"done"`).
  4. additionally cover at least one **constant-phase `retryable=True`**
     case (that branch independently calls `fail_trial` while holding the
     same lock) with the same completion + single-discharge assertions.
  This gate fails by timeout under the REV4 locked-abort design, because
  `handle_stripe_failure` already owns `_lifecycle_lock` while waiting on
  the cleanup executor's `Future.result()` — it proves the §2.2 CAS fix
  composes with the existing failure matrix.
  **Harness requirement (Team Beta binding):** run W6's deliberate deadlock
  detector in an **isolated subprocess that can be terminated on timeout**. A
  plain `ThreadPoolExecutor` `future.result(timeout=...)` can report a
  timeout while leaving the caller thread blocked holding `_lifecycle_lock`
  and the cleanup-executor thread blocked waiting for it — those surviving
  threads can prevent the test process from exiting. The timeout must
  terminate the isolated child process and make the parent harness fail
  cleanly.

**STOP after D1.0.** Team Alpha reviews, Team Beta reviews, Michael commits.
D1.1 begins only against the corrected producer.

---

## 3. Verified input contract (post-D1.0; re-verify each cite)

**Manifest shape** — `_build_manifest` (coordinator:2038-2069): `event_id,
run_id, stripe_id, workflow_phase, attempt, sub_index, local_spool_path,
expected_size, expected_sha256, trial_metadata`, plus **top-level**
`dataset_sha256`/`residue_sha256` deliberately lifted from the metadata
(:2064-2068). Every Phase-5-published manifest carries a complete, validated
`trial_metadata` — `publish_attempt` (:1913-1961) fails closed on a missing
durable trial-context row and runs `validate_trial_metadata` (:1551) before
any sink call.

**Metadata field sets** (:1352-1378): 9 trial-global, 2 provenance, 6
phase-specific (`workflow_phase, family_name, prng_type, direction, skip_mode,
threshold_used`). `prng_type = prng_base` (constant) / `prng_base + "_hybrid"`
(variable); `threshold_used` = the directional threshold; direction/skip_mode
are explicit strings from `workflow_phase_semantics` (:1475) via
`derive_trial_metadata` (:1501).

**Spool payload** — `build_substripe_payload_bytes`
(`miner/range_miner_worker.py:881-899`): compact sorted-key UTF-8 JSON:

```json
{"schema_version":"s172_substripe_v1","stripe_id":"...","sub_index":N,
 "seed_start":S,"seed_count":C,"survivors":[[seed,match_rate,strategy_id_or_null,[skip...]], ...]}
```

Constant passes emit `(seed, rate, null, [best_skip])`; hybrid passes emit
`(seed, rate, strategy_id, skip_sequence)`. `strategy_id`/`skip_sequence` are
NOT canonical-record fields and NOT 22-array fields.

**Producer guarantees the sink relies on (each verified; concurrency wording
per Team Beta [TB-D1-GC1]):**
- `publish_attempt` fires only when a whole attempt reconciles; a failed
  attempt publishes nothing before retry (~:2885). At most **one accepted
  attempt** per stripe reaches the sink.
- **Publication happens once. [TB-D1-B2]** `publish_shard` is called once per
  verified shard, then the shard is marked enqueued (:1954-1957). On failed
  commit delivery the coordinator later calls ONLY
  `phase5_sink.commit_trial(event)` again (:2941-2960); **there is no manifest
  replay step.**
- `commit_trial` (:2928-2965): a sink raise → the coordinator **returns**
  `{"delivery":"failed", "error":...}` (it does NOT propagate), records the
  failed status durably, and redelivers the SAME event
  (`event_id = f"{run_id}:commit"`) on the next call. Commit after abort
  raises `TrialAborted` at the coordinator (:2936-2939) — it never reaches
  the sink.
- `abort_trial` runs on the cleanup executor (`submit_abort`, :3026-3032) and
  calls the sink synchronously; staged files are deleted only after the sink
  returns successfully (L7).
- **Terminal concurrency:** at HEAD `7f2a010`, abort could race a commit-time
  sink delivery because `abort_trial` failed to disambiguate WHY
  `mark_trial_aborted` returned `False` (stale pre-read; no terminal-state
  re-read). **D1.0 corrects this via the ledger's atomic CAS + re-read
  (§2.2).** Against the post-D1.0 coordinator, commit and abort sink
  deliveries for one run are mutually exclusive. The sink retains its own
  lock for internal thread safety and defense-in-depth against direct or
  malformed callers — it must NOT be described or relied upon as the
  mechanism that repairs coordinator terminal exclusivity.

**Trial-context canonicalization** — `_canonicalize_trial_context`
(:1409-1433): the 11-field canonical semantic form (9 trial-global + 2
provenance; sessions normalized `None → []`; typed coercions). Reuse it —
never write a second canonicalizer.

**Workflow authority** — `workflow_stages_for` (:3591, post-D1.0) is the
producer's family/phase authority. Import it; never reproduce the suffix table
by hand **[TB-D1-B5]**.

---

## 4. `AssemblingPhase5Sink` — lifecycle semantics

### 4.0 Restart / retryability contract — exact wording  **[TB-D1-B2]**

> A failed assembly is retryable only while the same sink instance retains its
> accumulated manifests. Coordinator commit redelivery reuses those retained
> manifests; it does not republish them. Cross-process sink reconstruction is
> not provided by D1 and must not be claimed.

Consequences, binding on §4.3: assembly failure deletes ONLY temporary
assembly/result state; it **must retain the accumulated manifests** needed for
redelivery. Successful commit is the only point at which those inputs may
later become eligible for logical release (physical staged-file lifecycle
remains the coordinator's, per L2/L7; temp cleanup is D7).

### 4.1 Synchronization + ownership  **[TB-D1-B3, TB-D1-GC1]**

```text
AssemblingPhase5Sink owns one threading.RLock.

publish_shard, commit_trial, abort_trial, and result access (get_assembly)
all synchronize through that lock.

commit_trial holds the lock through assembly and atomic result installation.
abort_trial waits for an active assembly to finish, then clears state and
returns.

publish_shard stores a canonical deep copy of the manifest, never the caller's
mutable dictionary reference.
```

Purpose of the lock, post-D1.0: **internal thread safety and defense-in-depth
against direct or malformed callers.** Legitimate coordinator flow never
invokes sink commit and sink abort concurrently for one run (§3 terminal
concurrency); the sink lock is NOT the mechanism that provides that mutual
exclusivity — §2.2 is. The deep copy closes the aliasing hole: the coordinator
retains and returns the same mutable manifest dicts (`self.enqueued`,
:1956-1958), so a caller-side mutation after publication must not alter the
sink's future assembly input. A sorted-key JSON round-trip is an acceptable
deep-copy mechanism and doubles as the canonical comparison form.

### 4.2 `publish_shard(manifest)`

1. **No spool I/O at publish time.** Publish stores the (deep-copied) manifest
   only; all spool reads happen at commit-time assembly (gate-instrumented).
2. **Tombstoned `run_id` → silently ignore**, zero spool opens
   **[TB-D1-DEC2]** ("harmlessly ignore later stale manifests",
   coordinator:1287-1297).
3. Re-run `validate_trial_metadata(manifest["trial_metadata"])` (reuse the
   coordinator's function).
4. **Replay / slot-conflict rules [TB-D1-2] — exact, no additions:**
   - same `event_id` + canonically identical manifest → idempotent no-op;
   - same `event_id` + different content → raise `ManifestReplayConflict`;
   - different `event_id`, same `(run_id, stripe_id, sub_index)` → raise
     `ManifestReplayConflict`, **even when bytes and SHA are identical**;
   - "canonically identical" = equal after the sorted-key JSON round-trip.
5. **Post-commit rules [TB-D1-API]:** once a run has successfully committed —
   - an exact already-known shard replay remains an idempotent no-op;
   - a NEW event or NEW logical shard for that committed run raises
     `AssemblyStateError`.
6. Accumulate keyed by `run_id`.

### 4.3 `commit_trial(event)` — atomic install  **[TB-D1-3]**

> Assembly is stored exactly once **successfully**. A successful duplicate
> commit event is an idempotent no-op. If assembly raises, no completed result
> and no consumed-commit marker is stored, and redelivery of the same commit
> event attempts assembly again.

Mechanics, under the sink lock throughout:
1. assemble into **local temporary state** (run the §5 engine over the
   retained manifests);
2. complete **all** validation (§5.1-§5.5) while still in temporary state;
3. install the finished `MinerTrialAssembly` into the run-result store;
4. only then record the consumed `event_id`.

On any raise: delete the temporary assembly/result state ONLY — **retain the
accumulated manifests** (§4.0) — then re-raise. The coordinator converts the
raise to `delivery:"failed"` and redelivers the same event.

Duplicates & invariants:
- consumed `event_id` replayed → no-op, **zero** spool opens, zero map
  construction (instrumented; see G4-B — the real coordinator never replays a
  `"done"` delivery to the sink, :2941-2943, so this is sink-level
  defense-in-depth **[TB-D1-GC2]**);
- a **different** commit `event_id` for an already-committed run → raise
  `AssemblyStateError`; must NOT trigger replacement assembly **[TB-D1-API]**
  (the coordinator only ever emits `{run_id}:commit`, :2934);
- commit for a **tombstoned** run → raise `AssemblyStateError` (post-D1.0,
  reachable only by direct call — the coordinator raises `TrialAborted`
  first);
- commit with **zero retained manifests** → raise `AssemblyStateError` (stop
  condition §10 if it fires on a legitimately driven fixture).

### 4.4 `abort_trial(event)` — synchronous discharge  **[TB-D1-DEC2, TB-D1-B3]**

Takes the sink lock (waiting for any active assembly to finish), then on
return ALL of the following hold:
- accumulated manifests for the `run_id`: discarded;
- any partial or completed assembly: discarded;
- the sink holds **no reference** to any trial-owned staged path;
- the `run_id` is tombstoned; later stale `publish_shard` calls are ignored
  with zero spool opens.

Idempotent: aborting an unknown or already-aborted `run_id` is a successful
no-op.

### 4.5 `get_assembly` — the frozen D6 accessor  **[TB-D1-API]**

```python
def get_assembly(self, run_id: str) -> Optional[MinerTrialAssembly]: ...
```

Synchronized by the sink lock. Returns `None` before successful commit and
after abort — never a partial result. D6 fails closed on `None`. This
signature is frozen now so D6 has a stable surface.

---

## 5. The assembly engine (backend-independent)

One module-level function (suggested: `assemble_trial(run_id, manifests) ->
MinerTrialAssembly`) so D4/D5 later call the SAME derivation **[TB-R2]**. The
sink's `commit_trial` is its only D1 caller.

### 5.1 Per-manifest identity validation — before grouping  **[TB-D1-5, TB-D1-B5]**

For EVERY manifest, each raising `PhaseIdentityError` on mismatch:

```text
manifest["run_id"]         == the run_id passed to assemble_trial
manifest["workflow_phase"] == trial_metadata["workflow_phase"]
manifest["dataset_sha256"] == trial_metadata["dataset_sha256"]     # lifted copies agree
manifest["residue_sha256"] == trial_metadata["residue_sha256"]
(direction, skip_mode)     == workflow_phase_semantics(workflow_phase)
family_name                == the family workflow_stages_for(prng_base, True)
                              maps to this workflow_phase   # imported authority,
                                                            # never a hand-built table
prng_type == prng_base                    when skip_mode == "constant"
prng_type == prng_base + "_hybrid"        when skip_mode == "variable"
threshold_used == forward_threshold       when direction == "forward"
threshold_used == reverse_threshold       when direction == "reverse"
```

Direction and skip_mode remain the **explicit manifest strings** and are the
grouping values. `workflow_phase_semantics` and `workflow_stages_for` are
consistency oracles only — never silent substitutes.

### 5.2 Cross-manifest consistency — 11 fields  **[TB-D1-1]**

All manifests of one `run_id` must agree on the 11-field canonical trial
context (9 trial-global + `dataset_sha256` + `residue_sha256`), canonicalized
with the coordinator's `_canonicalize_trial_context` semantics (:1409-1433).
Mixed hashes or mixed trial-global values → `AssemblyConsistencyError`.

**Phase-set completeness [TB-D1-B1]:** the set of workflow phases present must
be exactly `{1,2}` or `{1,2,3,4}`. Anything else — `{1}`, `{1,3}`, `{1,2,3}`,
`{2}`, … — is an incomplete directional pairing and raises
`AssemblyConsistencyError`. D1 never declares an absent reverse population
legitimate and never fabricates one. (Every executed pass yields ≥1 manifest:
each completed stripe publishes its verified shards, and sub-stripes spool a
payload even with zero survivors — so phase absence means the pass did not
run.)

### 5.3 Spool read + identity + payload validation  **[TB-D1-6, TB-D1-B5, TB-D1-PV]**

Per manifest, at commit-time only:
1. read staged bytes; verify count == `expected_size`, SHA-256 ==
   `expected_sha256` (defense-in-depth);
2. parse; **container validation [TB-D1-PV]** — JSON decode errors and ALL
   container/type failures become `SpoolIdentityError`, never a raw
   `TypeError`/`KeyError`/quirk escape:
   - parsed JSON root must be a `dict`;
   - all mandatory payload keys must exist (`schema_version, stripe_id,
     sub_index, seed_start, seed_count, survivors`);
   - `survivors` must be a `list`;
   - each survivor entry must be a `list` of exactly four elements;
   - `skip_sequence` must itself be a `list`;
3. assert identity:
   - `schema_version == "s172_substripe_v1"`;
   - payload `stripe_id` == manifest `stripe_id`;
   - payload `sub_index` is an **integer excluding bool**, THEN equal to the
     manifest value (`True == 1` in Python — equality alone does not exclude a
     Boolean payload identity **[TB-D1-PV]**);
4. assert payload semantics (`bool` excluded everywhere —
   `isinstance(x, bool)` fails every integer check below):
   - `seed_start`: integer, not bool;
   - `seed_count`: nonnegative integer, not bool;
   - every seed: integer (not bool) AND within
     `[seed_start, seed_start + seed_count)`;
   - `match_rate`: numeric (not bool), **finite**, within `[0.0, 1.0]`;
   - `strategy_id`: `None` or integer (not bool);
   - every skip value: integer (not bool).
   Any failure → `SpoolIdentityError`. A correctly hashed but misassociated,
   malformed, or semantically invalid spool must never enter a directional
   map.

### 5.4 Directional maps + duplicate invariant  **[TB-R1, TB-D1-7]**

Group by the explicit `(direction, skip_mode)` strings into the four passes;
build `forward_map_constant, reverse_map_constant, forward_map_variable,
reverse_map_variable` (`seed → match_rate`). While inserting, a seed already
present in the SAME directional population raises immediately:

```python
class DirectionalDuplicateError(Exception):
    # STRUCTURED ATTRIBUTES — real attributes D2 asserts on directly:
    run_id; workflow_phase; direction; skip_mode; seed
    first_stripe; first_sub_index; first_attempt; first_match_rate
    dup_stripe;   dup_sub_index;   dup_attempt;   dup_match_rate
```

A duplicate is a producer/coverage defect, never a dedup opportunity; Phase 5
must NOT resolve it by max match_rate (that rule belongs only to the D3
cross-trial accumulator boundary, `v1_4_4` §4.3). Track
`(stripe_id, sub_index, attempt, match_rate)` provenance per inserted seed.

Post-D1.0 emptiness rules: variable maps are empty exactly when phases `{3,4}`
are absent (`test_both_modes=False`); a pass whose shards all carry empty
survivor lists yields an empty map. There is no legitimate lone-direction pass
(§5.2).

Validate every distinct `prng_type` via
`utils/prng_encoding.encode_prng_type` and every `skip_mode` via
`encode_skip_mode` (hard-fail on unknown). Discard the numeric; strings stay
in records, numeric arrays are D3.

### 5.5 Intersections + canonical enrichment

```text
bidirectional_constant = keys(forward_map_constant) & keys(reverse_map_constant)
bidirectional_variable = keys(forward_map_variable) & keys(reverse_map_variable)
```

Per mode, with `F = set(fwd_map)`, `R = set(rev_map)` — semantics frozen from
the live constant block (`window_optimizer_integration_final.py:652-694`) and
variable block (`:756-796`):

```python
forward_count             = len(fwd_map)          # == record count, guaranteed by §5.4
reverse_count             = len(rev_map)
bidirectional_count       = len(F & R)
intersection_count        = len(F & R)            # same value, distinct field (live :660-661 / :773-774)
intersection_ratio        = len(F & R) / max(len(F | R), 1)
forward_only_count        = len(F - R)
reverse_only_count        = len(R - F)
survivor_overlap_ratio    = len(F & R) / max(len(F), 1)
bidirectional_selectivity = len(F) / max(len(R), 1)
intersection_weight       = len(F & R) / max(len(F) + len(R), 1)
skip_range                = skip_max - skip_min
# per surviving seed:
score                     = (fwd_map[seed] + rev_map[seed]) / 2.0
```

`window_size, offset, skip_min, skip_max, sessions, trial_number, prng_base`
from the consistency-checked trial-global metadata; `skip_mode, prng_type`
from the mode's phase-specific metadata; `sessions` normalized `None → []`
(D0 canonical semantics).

---

## 6. The frozen 24-field canonical record  **[TB-D1-4]**

One module-level constant; every record has exactly these keys in exactly this
order (live insertion order at `window_optimizer_integration_final.py:683-694`
= seed/rates/score + `metadata_base` :652-676; hybrid identically :785-796 +
:756-780):

```python
CANONICAL_RECORD_FIELDS = (
    "seed", "forward_match_rate", "reverse_match_rate", "score",
    "window_size", "offset", "skip_min", "skip_max", "skip_range", "sessions",
    "trial_number", "prng_base", "skip_mode", "prng_type",
    "forward_count", "reverse_count", "bidirectional_count",
    "intersection_count", "intersection_ratio",
    "forward_only_count", "reverse_only_count",
    "survivor_overlap_ratio", "bidirectional_selectivity", "intersection_weight",
)
```

- `canonical_records_constant` / `canonical_records_variable`: one record per
  surviving bidirectional seed of that mode, **ascending seed order**;
- **`threshold_used` is NOT a 25th field** — validated (§5.1) and gate-tested,
  but manifest identity/validation metadata only;
- no extra keys, no missing keys, no reordering — gate-enforced (G9).

## 7. `MinerTrialAssembly`  **[TB-D1-DEC1]**

```python
@dataclass
class MinerTrialAssembly:
    run_id: str
    bidirectional_constant: set          # set[int]
    bidirectional_variable: set
    forward_map_constant: dict           # dict[int, float]
    reverse_map_constant: dict
    forward_map_variable: dict
    reverse_map_variable: dict
    canonical_records_constant: list     # list[dict], ascending seed
    canonical_records_variable: list
    directional_counts: dict
    timing: dict                         # at least {"assembly_s": float}
    # D3/D4 populate ONLY after the corresponding artifact is successfully
    # written AND validated, via dataclasses.replace() (or an equivalent
    # explicit update) — never by mutating mid-way through a failed write.
    # None means "not produced yet"; an empty string would falsely claim a
    # path exists. The D6 adapter fails closed on None where it needs a path.
    binary_npz_path: Optional[str] = None
    all_npz_path: Optional[str] = None
```

Stable cross-deliverable result object — do NOT split. `directional_counts`
keys (ints): `forward_constant, reverse_constant, forward_variable,
reverse_variable, bidirectional_constant, bidirectional_variable`.

## 8. Exceptions (module-level in `range_miner_npz_writer.py`)

| Exception                   | Raised when |
| --------------------------- | ----------- |
| `ManifestReplayConflict`    | §4.2.4 conflicts (event_id content conflict; logical-slot conflict even with identical bytes) |
| `AssemblyConsistencyError`  | §5.2 mixed 11-field context; incomplete phase set (not `{1,2}` / `{1,2,3,4}`) |
| `PhaseIdentityError`        | §5.1 any identity check fails (incl. run_id + lifted-provenance agreement) |
| `SpoolIdentityError`        | §5.3 size/sha/JSON-decode/container/schema/stripe_id/sub_index/tuple-shape/semantic failure |
| `DirectionalDuplicateError` | §5.4 duplicate seed within one directional population (structured attributes mandatory) |
| `AssemblyStateError`        | commit for tombstoned run (direct probe); different commit event_id for committed run; commit with zero retained manifests; new event/slot published to a committed run |

All are producer/contract defects: fail closed, never resolve silently, never
catch-and-continue inside the engine.

## 9. Gate D1.1 — `tests/test_s172_phase5_d1_engine.py`

Fixture: real (post-D1.0) coordinator + ledger + real staged spools through
the real serve/publish surface (D0 harness pattern), a **two-mode,
multi-stripe, multi-sub-stripe** run with hand-computable populations: per
mode at least one seed in `F∩R`, one in `F−R`, one in `R−F`, distinct match
rates so `score` averaging is observable. Instrument spool opens (wrap/count
engine file opens).

1. **G1 hand-computed assembly:** four maps, both intersections, EVERY §5.5
   derived field equal hand-computed expected values, both modes. Mis-grouping
   tripwire: perturb one manifest's direction/phase in a fixture copy → the
   equality gate must fail, and the perturbed run must raise
   `PhaseIdentityError`.
2. **G2:** `get_assembly(run_id)` is `None` before commit.
3. **G3:** successful real `coordinator.commit_trial` → `delivery == "done"`,
   `get_assembly` returns exactly one complete `MinerTrialAssembly`.
4. **G4 — split [TB-D1-GC2]:**
   - **G4-A real coordinator duplicate:** re-call real
     `coordinator.commit_trial` after `"done"` → coordinator returns the
     duplicate no-op (`duplicate: True`, delivery `"done"`); the sink's
     commit-call count does **not** increase; zero spool opens, zero map
     construction. (Proves coordinator idempotence — the coordinator never
     replays a `"done"` delivery to the sink, :2941-2943.)
   - **G4-B direct sink replay probe (labeled):** call `sink.commit_trial()`
     directly with the exact successfully consumed event → no raise; zero
     spool opens; zero map construction; the stored assembly remains the SAME
     object. (Proves the sink-level idempotence contract of §4.3.)
5. **G5 failed-assembly retryability — real coordinator sequence:** corrupt
   one staged spool's bytes after publish, then:
   1. call real `coordinator.commit_trial(run_id)`;
   2. assert returned `delivery == "failed"` (the coordinator catches — the
      call does NOT raise);
   3. assert `event_id == f"{run_id}:commit"`;
   4. assert `get_assembly(run_id) is None` and no consumed-event marker;
   5. assert accumulated manifests are retained;
   6. repair the spool;
   7. call real `coordinator.commit_trial(run_id)` again;
   8. assert the SAME event_id, `delivery == "done"`, and one completed
      result.
6. **G6:** exact `event_id` + canonically identical manifest replay → no-op.
7. **G7 (direct sink invariant-break probes, labeled):** same `event_id` +
   different content → `ManifestReplayConflict`; different `event_id`, same
   `(run_id, stripe_id, sub_index)`, identical bytes+SHA →
   `ManifestReplayConflict`; post-commit NEW event/slot →
   `AssemblyStateError`; post-commit different commit `event_id` →
   `AssemblyStateError`, no replacement assembly (stored result unchanged,
   zero spool opens).
8. **G8 abort/tombstone — split [TB-D1-B4]:**
   - real `coordinator.abort_trial(run_id)` → sink state cleared, tombstoned;
     a stale `publish_shard` after abort is ignored with zero spool opens;
     `get_assembly(run_id) is None`;
   - real `coordinator.commit_trial(run_id)` after abort → raises
     `TrialAborted` **at the coordinator** (never reaches the sink);
   - direct `sink.commit_trial(event)` after tombstone → `AssemblyStateError`
     (direct sink invariant-break probe, labeled).
9. **G9:** every canonical record has exactly the frozen 24 keys, in the
   frozen order, lists ascending by seed; both modes.
10. **G10:** `threshold_used` validated per direction (flip one manifest's
    value → `PhaseIdentityError`) AND absent from every record's key set.
11. **G11 identity matrix:** individually corrupt each of — `direction`,
    `skip_mode`, `prng_type`, `family_name`, manifest-vs-metadata
    `workflow_phase`, manifest `run_id`, lifted `dataset_sha256`, lifted
    `residue_sha256`, `threshold_used` — each raises `PhaseIdentityError`;
    the oracles never silently substitute.
12. **G12 spool identity + semantics + containers [TB-D1-B5, TB-D1-PV]:**
    mismatched payload `stripe_id`; mismatched `sub_index`; wrong
    `schema_version`; malformed tuple shape; bool seed; out-of-range seed;
    non-finite match_rate; match_rate > 1.0; bool strategy_id; bool skip
    value; **root is not a dict; `survivors` is not a list; payload
    `sub_index` is bool; `skip_sequence` is not a list; invalid JSON** → each
    raises `SpoolIdentityError`, nothing enters any map. **For every case
    where the payload remains valid JSON, recompute `expected_size` and
    `expected_sha256` after modifying it** so the gate exercises the
    validator, not merely the digest check (the invalid-JSON case necessarily
    also recomputes size/sha over the broken bytes).
13. **G13 [TB-D1-G13C]:** cross-manifest provenance divergence — on ONE
    manifest of the run, mutate **both copies together** so the manifest stays
    internally valid but its canonical 11-field context disagrees with the
    others:
    ```python
    manifest["dataset_sha256"] = different_hash
    manifest["trial_metadata"]["dataset_sha256"] = different_hash
    ```
    (and, separately, the same for `residue_sha256`) →
    `AssemblyConsistencyError`. Mutating only the nested copy belongs to G11,
    where §5.1's lifted-vs-nested check fires `PhaseIdentityError` FIRST —
    those single-copy cases stay in G11 and must not be duplicated here. Also:
    a phase set of `{1}` and of `{1,2,3}` (recording-sink fixtures) →
    `AssemblyConsistencyError` **[TB-D1-B1]**.
14. **G14:** engine-level duplicate seed within one directional population →
    `DirectionalDuplicateError` with ALL structured attributes populated and
    correct (assert attributes, not message text). Full producer-level
    adversarial fixture is D2.
15. **G15 empty-pass legitimacy (post-D1.0):** a `test_both_modes=False` run
    (phases exactly `{1,2}`) → constant maps/records populated, variable
    maps/sets/records empty, no error; a pass whose shards carry empty
    survivor lists → empty map, derived fields consistent with the
    `max(..., 1)` guards, no error.
16. **G16 ownership + concurrency [TB-D1-B3, TB-D1-GC1]:**
    - mutate the caller-owned manifest dict (and its nested
      `trial_metadata`) after `publish_shard` → the sink's stored input and
      subsequent assembly are unaffected;
    - **(direct sink invariant-break probe, labeled — the post-D1.0
      coordinator never produces this interleaving; W5 proves that):**
      concurrent direct `sink.commit_trial()` / `sink.abort_trial()` for one
      run → abort cannot return while a spool read is active (assert ordering
      via the instrumented reader + events). **Final state after BOTH direct
      calls have completed [TB-D1-G16C]** — abort-last always wins: if abort
      takes the lock first, it tombstones and commit raises
      `AssemblyStateError`; if commit takes the lock first, it installs a
      result, then abort removes it and tombstones. A stored assembly may
      exist transiently before abort returns, but never after the synchronous
      abort completes. Assert:
      - the run is tombstoned;
      - `get_assembly(run_id) is None`;
      - no manifests remain; no temporary state remains; no staged-path
        reference remains;
      - commit either completed before abort or raised `AssemblyStateError`
        after abort;
      - no torn mixed state occurred;
    - after abort returns: no manifest, no result, no temporary state, no
      staged-path reference remains in the sink (introspect sink state).

**Non-regression (blocking):** Phase 4 **63/63**, Phase 3 **17/17**, D0
harness, and the D1.0 workflow gates (W1-W6) all green throughout. Any red
STOPS work.

## 10. Explicit stop conditions (STOP and report, do not code around)

- a canonical-record field cannot be produced without inventing a formula;
- a directional population proves incomplete/truncated/unassociable at read
  time in a way §5.2/§5.3 does not already classify;
- `AssemblyStateError` (zero-manifest commit) or `AssemblyConsistencyError`
  (phase-set) fires against a legitimately driven real-lifecycle fixture —
  report, do not relax;
- **The terminal-race fix must use the ledger's existing atomic
  `state='running'` transition and post-failure state re-read. Do not acquire
  `_lifecycle_lock` inside `abort_trial`, because `fail_trial` may invoke
  abort through the cleanup executor while the caller already holds that
  lock** [TB-D1-DL] — if the CAS + re-read shape cannot be implemented as
  specified, report; do not reach for a lock;
- W5's instrumentation (the pausable recording sink, or W5-R's thread-specific
  first-call `ledger.get_trial` interception) cannot deterministically hold
  its window open — report the obstacle, do not substitute sleeps for
  synchronization and do not weaken W5-R back into an ordinary-ordering gate;
- reusing the coordinator's canonicalizer/validators/`workflow_stages_for`
  requires changing coordinator code beyond the two D1.0 fixes — report the
  exact obstacle;
- any gate only passes by relaxing it to match a stub.

## 11. Kickoff

Implement **D1.0 first** (both §2 corrections + gates W1-W6), iterate to green
(`source ~/venvs/torch/bin/activate; PYTHONPATH=. python3
tests/test_s172_phase5_d1_workflow.py`), **STOP and report**. Only after
review clearance: implement D1.1 (`miner/range_miner_npz_writer.py` + engine
harness), iterate to green, STOP and report before D2.
