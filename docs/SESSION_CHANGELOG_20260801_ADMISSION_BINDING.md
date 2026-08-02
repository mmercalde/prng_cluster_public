# SESSION_CHANGELOG_20260801_ADMISSION_BINDING.md

**Subject:** the two repairs Team Beta required before Phase-7 closure, plus Q1's executable half.
**Authority:** `docs/CLAUDE_CODE_INSTRUCTIONS_ADMISSION_BINDING_REPAIR.md` (REV1).
**Base:** `a836595`, clean tree. VM 101 (`zeus-ubuntu-vm`, 192.168.3.177), venv `~/venvs/torch`.
**Not committed, not pushed, WATCHER not run** — Claude Code stops at the gate.

Beta **accepted** fleet identity and consumer unification at `63e627f` but **withheld Phase-7
closure** pending exactly these. This session is both, plus repair C, and nothing else.

---

## Repair A — the freeze-after-read property was FALSE as implemented

**What Alpha claimed** (`docs/TEAM_ALPHA_EXECUTION_SET_AND_CHAPTER2_SUBMISSION.md` §1.3): that
`freeze_execution_set()` refusing an already-read set made Beta's ordering requirement
*"structurally impossible to violate rather than merely documented."*

**Why it was false.** `active_execution_set()` incremented `_READS` **only inside**
`if _ACTIVE is not None`. A consumer could read `None`, take the legacy path — which every
consumer helper has (`if s is None: <pre-existing behaviour>`) — and the set could still be
frozen afterwards. Alpha asserted the property without tracing the counter's trigger condition.
The **live entrypoints were and are correctly ordered**, so the full-fleet evidence stands; the
stronger structural claim did not.

### What now enforces it — the code path, including the `None` case

`execution_set.py:804-839`

```python
def active_execution_set() -> Optional[ResolvedExecutionSet]:
    global _READS
    with _LOCK:
        _READS += 1          # :838 — UNCONDITIONAL; fires on a None read too
        return _ACTIVE
```

`execution_set.py:735-780` — `freeze_execution_set()` reads that counter **before** installing
anything:

```python
    with _LOCK:
        if _ACTIVE is not None:          # :761 — idempotent branch, taken FIRST
            if _ACTIVE.set_id() == s.set_id():
                return _ACTIVE           # :763 — never consults _READS
            raise ExecutionSetError("... FROZEN for this run ...")
        if _READS:                       # :770
            raise ExecutionSetError("cannot freeze ... after it has already been read ...")
        _ACTIVE = s                      # :777
```

So: an empty read reaches `_READS += 1` at `:838`, and the next freeze reaches `if _READS:` at
`:770` and refuses. Re-freezing an identical set returns at `:763`, **before** `_READS` is
consulted, which is why consumption cannot break WATCHER/CLI re-entrancy.

### The private, non-consuming peek

`execution_set.py:783-801` — `_peek_execution_set()`. Private, returns `_ACTIVE` under the lock,
does **not** touch `_READS`. Its one caller is the code that **owns** the freeze:

`agents/watcher_agent.py:1337` — `WatcherAgent._ensure_execution_set` is re-entered on every
step and must ask *"have I already frozen one this process?"*. That is a decision about
**whether to freeze**, not about **how to run**. Counting it would make the resolver trip its own
guard on step 1 and refuse the very freeze it exists to perform. Gate A4 asserts by AST over the
live source that `_ensure_execution_set` calls `_peek_execution_set` and **not**
`active_execution_set`, and that no consumer helper reaches the peek.

### Three gates (+ two controls)

| gate | requirement | result |
|---|---|---|
| A1 | empty consumer read → later freeze **refused** | PASS |
| A2 | clean resolve/freeze before any read → **passes** | PASS |
| A3 | identical re-freeze after consumption → **still idempotent** | PASS |
| A4 | the peek is silent, private, and used by the owner only | PASS |
| A5 | **fault injection** — restore the `None`-read exemption → A1 goes RED | PASS |

`tests/test_s172_resolved_execution_set.py::g_resolve_once_read_then_freeze` **encoded the hole**
(it asserted *"a read of an EMPTY set must not block a later freeze"* and forged `XS._READS = 1`
to reach the refusal). It is corrected in place — the tally stays **34/34**.

### The committed claim is corrected, not deleted

`docs/TEAM_ALPHA_EXECUTION_SET_AND_CHAPTER2_SUBMISSION.md` §1.3 now carries a **RETRACTION**
block above the original paragraph, which is left standing (`G-COMMENT-TRUTH`): what was
claimed, why it was false, what was actually true and remains true, and what holds now.

---

## Repair B — admission bound to the frozen set

**The defect.** The set recorded an admission count while `serve_trial` derived
`expected_workers` independently from `context["worker_pool_size"]`. Two frozen run facts about
one run, free to disagree — a local two-GPU set still waited for eight workers that the set
itself refuses to admit.

### Semantics

`execution_set.py:660-704`, inside `resolve_execution_set()`:

```
requested_admission = int(admission_count) if admission_count is not None else None
identity_count      = sum(n.gpu_count for n in nodes)        # :674 — the SAME tuple
                                                             #        contains_worker() tests
effective_admission = min(requested_admission, identity_count)   # :693
```

Both are recorded — `requested_admission_count` (`:198`) and the effective `admission_count` —
and both are in `content()`, hence in **`set_id`**: a run that asked for 8 and was clamped to 2
is not the same run as one that asked for 2.

### The four clamp cases

| case | expected | result |
|---|---|---|
| full 26-GPU set, default request 8 | **8**, unchanged | PASS |
| local Zeus set, default request 8, two GPUs | **2** | PASS |
| local set, explicit request 1 | **1** (no clamp — it is `min`, not "the node's GPU count") | PASS |
| zero / negative / zero-capacity | **fail during resolution** | PASS |

Zero-capacity is injected at the source the resolver reads (`gpu_count` in a temp
`distributed_config.json`), so the refusal is produced by the real path, not a hand-built object.

### Visibly logged, and in provenance

```
[EXEC-SET] ADMISSION CLAMPED: requested 8, but the resolved set contains 2 worker
identities (localhost:2) — this run admits 2. The clamp is recorded in provenance
(requested_admission_count/admission_count) and in set_id.
[EXEC-SET] resolved: execution set c1998493a4fa backend=miner profile=proxmox PARTIAL
nodes=['localhost'] gpus=2 remote=False admission=2 (CLAMPED from requested 8; 2 worker
identities in the set)
```

`describe()` carries the clamp, so it appears at resolution, at freeze, in the CLI banner and in
WATCHER's log line. `to_provenance()` adds `requested_admission_count`, `admission_clamped` and
`worker_identity_count`.

### Where `expected_workers` comes from now

`miner/range_miner_coordinator.py:3693-3696`, in `serve_trial`'s preamble:

```python
expected_workers, _admission_source = _execution_set_expected_workers(
    int(context.get("worker_pool_size", 1) or 1))
logger.info("[ADMISSION] run %s: expected_workers=%d (source=%s)",
            run_id, expected_workers, _admission_source)
```

`_execution_set_expected_workers` (`:178`) → `execution_set.admission_expectation` (`:952`).
With a set frozen the **set's effective `admission_count` is returned**; the context value stays
the REQUEST and is no longer a second answer. With **no** set frozen the context value is
returned unchanged — the pre-existing behaviour every Phase-4 loopback gate runs on, and a branch
production cannot take (both entrypoints freeze before any coordinator exists).

Read back off a real run: `[ADMISSION] run run-admission_t5_…: expected_workers=2
(source=execution_set(c1998493a4fa))`.

`window_optimizer.py:1496` previously said the count was *"recorded, never re-imposed … not so
that anything downstream is overridden by it."* That comment **was** the defect; it is corrected
in place.

**PWC scope note:** `min_workers` is recorded (and clamped for the record) in the set, but the
PWC still reads its own value. Binding it is not in this brief.

---

## Repair C — Q1's executable half, over the REAL `serve_trial`

`G-LOCAL` calls `fleet_preflight()` directly. Every gate below drives
`RangeMinerCoordinator.serve_trial` against loopback workers on genuine framed sockets with
**`serve_timeout=None`** — nothing but the code's own terminal decision can end a run, and a run
that does not decide is `still-hung`, a failure.

| # | requirement | result |
|---|---|---|
| 1 | `--execution-set-nodes localhost` resolves **two** eligible identities | PASS — `('zeus-ubuntu-vm:gpu0','zeus-ubuntu-vm:gpu1')`, `remote_execution=False` |
| 2 | the default pool request is **bounded to effective admission 2** | PASS |
| 3 | **two local workers cause stage assignment** without waiting for eight | PASS — assigned + committed in 0.4 s, `expected_workers=2 (source=execution_set(…))` |
| 4 | an **unlisted third worker remains quarantined** | PASS — dispatched nothing; and with 2 connected but only 1 listed, admission still reports **1 admitted** |
| 5 | missing local capacity reaches the **existing** bounded admission failure | PASS — `…phase 1) expected 2 eligible worker(s), 1 admitted after 6.0s (worker_admission_timeout=6.0s)` |
| 6 | **full-fleet / default-eight unchanged** | PASS — 8 listed workers → committed at `expected_workers=8`; 2 workers → still refuses, **naming 8** |
| — | **fault injection** — unbind admission → C3 goes RED | PASS |
| — | anti-vacuity summary — 4 real `serve_trial` runs, `serve_timeout=None` in every one | PASS |

Gate 4's second arm is the discriminating one: two well-formed workers are connected, only one is
listed, and admission still reports **1 admitted** — the stranger did not enter the eligible pool.
No Wall A/B rerun (Beta, explicit). Phase 6 not re-run — certified at `d98298c`.

---

## Confirmed unchanged (gate-asserted, not asserted in prose)

- `DEFAULT_WORKER_ADMISSION_TIMEOUT = 180.0` — value and constant.
- `serve_timeout` default `None`, in both the serve loop and the runner context.
- The **Blocker-3 matrix** — `handle_stripe_failure`, `_handle_stripe_failure_locked`,
  `_pick_other_worker`, `process_lease_expiry` **byte-identical** to `HEAD` (AST segment compare).
- `distributed_config.json` addresses (`localhost/.120/.154/.162`) — deliberate, CLAUDE.md §3.
- `expected_workers` still **bound exactly once**, at `serve_trial`'s **top level** (never inside
  the loop), from the requested `worker_pool_size`; `worker_pool_size` **code-site count
  unchanged** (the §4.3 unit-semantics check), which is why the new `[ADMISSION]` log line does
  not re-read the context key.

## Files changed

| file | change |
|---|---|
| `execution_set.py` | A: unconditional read counting + private peek. B: clamp, both counts, `set_id`, `describe()`, `to_provenance()`, `admission_expectation()` |
| `miner/range_miner_coordinator.py` | `_execution_set_expected_workers` + the one `serve_trial` binding |
| `agents/watcher_agent.py` | resolver owner switched to the non-consuming peek |
| `window_optimizer.py` | comment-only correction (`G-COMMENT-TRUTH`) |
| `docs/TEAM_ALPHA_EXECUTION_SET_AND_CHAPTER2_SUBMISSION.md` | §1.3 retraction, §1.4 supersession note |
| `tests/test_s172_resolved_execution_set.py` | the one gate that encoded the false property — corrected, tally still 34 |
| `tests/test_s172_admission_liveness.py` | gate 2 amended for the authorized binding change (still: one binding, in the preamble, from `worker_pool_size`) |
| `tests/test_s172_phase4_coordinator.py` | gate-22 registration, **appended** |
| `tests/test_s172_admission_binding.py` | **NEW** — 20 gates (A/B/C + 2 fault injections) |

## Verification-integrity controls (VIR-1…6)

- **execution proof:** effective `admission_count` read back from `to_provenance()`, and
  `expected_workers` read off the `[ADMISSION]` line the coordinator emitted during four real
  `serve_trial` runs — not from source.
- **clean control:** full-fleet default-8 unchanged (B1, C6 arm 1); clean resolve/freeze before
  any read still passes (A2).
- **fault-injection control:** A5 restores the `None`-read exemption → A1 reds. C-MUTANT reverts
  `_execution_set_expected_workers` to the raw context value → C3 reds.
- **completion sentinel:** **PASS**.
- **unavailable-observer:** none of these gates was skipped; nothing is assumed.
- **audit claim scope:** repairs A, B, C only. **Searched surfaces:** `execution_set.py`, the six
  consumers, `serve_trial` and `run_trial_miner`, both production entrypoints, the four cited
  gate suites, and an AST sweep of the whole tree for import-time consumer reads (result: NONE).
  **Unavailable surfaces:** real rig GPUs (no `serve_trial` run against CT100 daemons this
  session); Wall A/B (not required); Phase 6 (closed at `d98298c`).

## Non-regression — all green

resolved execution set **34/34** · P0.5 **38/38 `--fleet`** (live-fleet gate included) ·
admission liveness **16/16** · admission binding **20/20 (new)** · Phase 4 **63/63** ·
threshold-propagation 5/5 (3 mutants killed) · Chapter1-P0 12/12 · D6-threshold 17/17 (11
mutants) · D6 3.A 9/9 (16 mutants) · D6.1 15/15 (8 mutants) · D1.1 18/18 · D1.0 8/8 · D0 12/12 ·
D4 8/8 · D5 24/24 · Phase 3 · Phase 2 6/6.

Wall A/B not re-run (Beta, explicit). Phase 6 not re-run — certified and closed at `d98298c`.

## Observation, out of scope, [UNVERIFIED]

A **multi-stripe** loopback run (2 workers, `total_seeds = 2 × miner_stripe_size`) does not
terminate: shards sit at `staging_status='pending'`. It **reproduces with no execution set frozen
and this repair's code path never entered**, so it is independent of admission binding —
pre-existing behaviour of the loopback fixture (`_FakeWorker` answers one sub-stripe per stripe)
and/or the staging executor. Whether it is a fixture limitation or a production defect is
**not established here**. Recorded rather than absorbed; the new harness uses the single-stripe
sizing every existing miner suite uses, and says so in-file.
