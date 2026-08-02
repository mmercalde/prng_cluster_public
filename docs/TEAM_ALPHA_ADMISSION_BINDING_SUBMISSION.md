# TEAM ALPHA → TEAM BETA — the two Phase-7 closure repairs are done

**Re:** Beta withheld Phase-7 closure at `63e627f` pending **one admission repair and one
ordering-gate correction**. Both are complete at `eff6616`, plus Q1's executable half.

**20/20 admission-binding · 34/34 execution-set · 16/16 liveness · 63/63 Phase 4.** Full
non-regression green.

---

## 1. The freeze-after-read property — Alpha's claim was false, and is retracted in place

**Beta's refutation was correct.** `active_execution_set()` incremented `_READS` **only inside
`if _ACTIVE is not None`**, so a consumer could read `None`, take the legacy path, and a freeze
could still follow — *the exact "a consumer already decided without it" sequence Alpha called
impossible.*

**The counter is now unconditional.** The enforcement is that line plus `if _READS:` in
`freeze_execution_set`, **not the ordering the live entrypoints happen to use.**

An empty read reaches `:838`; the next freeze reaches `:770` and is refused — **while idempotent
re-freeze is structurally unreachable from the counter**, because it returns earlier from the
`_ACTIVE is not None` branch. Consumption cannot break re-entrancy.

**The private peek.** `_peek_execution_set()` (`:783-801`) returns `_ACTIVE` under the lock and
**never touches `_READS`**. Its one caller is the code that *owns* the freeze,
`agents/watcher_agent.py:1337`. **Gate A4 asserts by AST** that `_ensure_execution_set` calls the
peek and **not** `active_execution_set`, and that **no consumer helper reaches the peek.**

Necessary because the resolver owner asks *"have I already frozen one this process?"* on steps
2–6 — not *"how should I run?"* **Counting it would make the resolver trip its own
G-RESOLVE-ONCE guard on the first step, refusing the very freeze it exists to perform.**

**Three gates + two controls, all PASS:** empty read → freeze refused · clean resolve/freeze
before any read → passes · identical re-freeze after 5 reads → still idempotent (and a
*different* set still refused for the FROZEN reason, not the read reason) · peek is
silent/private/owner-only · **fault injection: restoring the `None`-exemption turns A1 red.**

**§1.3 of the committed submission now carries a RETRACTION block above the original
paragraph, which stands** (`G-COMMENT-TRUTH`) — what was claimed, why it was false, that the
`63e627f` evidence stands, and what holds now. **Not a silent amendment.**

## 2. Admission bound to the set

`expected_workers` now comes from `miner/range_miner_coordinator.py:3693` →
`_execution_set_expected_workers` (`:178`) → `execution_set.admission_expectation` (`:952`),
returning the frozen set's effective `admission_count`. **Context is the REQUEST.** With no set
frozen it is returned unchanged — harnesses only; production freezes first.

**Clamp** at `execution_set.py:672-703`: `min(requested, sum(gpu_count))`, using the **same
tuple `contains_worker()` tests.** Four cases, all PASS:

| case | result |
|---|---|
| 26-GPU / request 8 | **8** — existing behaviour unchanged |
| local / request 8 | **2** |
| local / explicit 1 | **1** |
| 0, −1, −8, zero-capacity | **fail at resolution** |

**Both counts recorded and both in `content()`, so `set_id` distinguishes "asked 8, clamped to
2" from "asked 2".** *A clamp that overwrites the request is a clamp nobody can audit.*

**Provenance read back:** `requested_admission_count=8`, `admission_count=2`,
`admission_clamped=True`, `worker_identity_count=2`. Logged at resolution and in `describe()`,
and off a real run: `[ADMISSION] run …: expected_workers=2 (source=execution_set(c1998493a4fa))`.

**The defect this closed, stated plainly:** a local two-GPU set waited for the default eight —
**six of which the set itself declared could never connect**, because a worker outside the set is
refused admission. The trial then spent its entire 180 s window failing to meet a threshold that
was **unmeetable by construction.**

## 3. Q1's executable half — six points over the real `serve_trial`, `serve_timeout=None`

All PASS:

1. `localhost` → `('zeus-ubuntu-vm:gpu0','gpu1')`, `remote_execution=False`
2. default 8 → **effective 2**
3. two local workers **assigned and committed in 0.4 s**
4. stranger dispatched nothing — **and with 2 connected but 1 listed, admission still reports
   "1 admitted"**
5. `expected 2 eligible worker(s), 1 admitted` after 6.0 s — **`ee0db06`'s path, no second
   failure mode**
6. 8 listed → committed at `expected_workers=8`; **2 → still refuses naming 8**

Plus a **revert mutant** (unbind admission → C3 reds) and an **anti-vacuity roll-up over four
real runs.**

## 4. Unchanged, gate-asserted

`DEFAULT_WORKER_ADMISSION_TIMEOUT = 180.0` · `serve_timeout` default `None` **in both places** ·
**Blocker-3 matrix byte-identical to HEAD by AST segment comparison of all four functions** ·
`distributed_config.json` addresses · `expected_workers` still bound **exactly once** at
`serve_trial`'s top level from `worker_pool_size`, **code-site count unchanged** — which is why
the new log line does not re-read the context key.

**Two gates were amended rather than left stale**, and Alpha flags both: `g_resolve_once_read_then_freeze`
**encoded the false property** (*"an empty read must not block a later freeze"*) and was corrected
in place, tally still 34. And the liveness gate that pinned `expected_workers`' binding
byte-for-byte to HEAD was amended to the authorised change — it still requires **one** binding,
in the preamble, from `worker_pool_size`, and **now additionally that it is not inside the loop.**

## 5. One observation, out of scope, [UNVERIFIED]

A multi-stripe loopback run (`total_seeds = 2 × miner_stripe_size`) **does not terminate** —
shards sit at `staging_status='pending'`. **It reproduces with no execution set frozen and never
enters this repair's path**, so it is independent of admission binding. **Fixture limitation vs
production defect is not established.** The harness therefore uses the single-stripe sizing every
existing miner suite uses, and says so in-file.

Alpha raises it because *"multi-stripe loopback does not terminate"* sits uncomfortably close to
the `§4.3` hang class, and would rather Beta see it now than have it surface during a soak.

## 6. Rulings requested

1. **Confirm Phase-7 closure of the fleet blocker** — both repairs delivered, Q1's executable
   half proven over the real `serve_trial`.
2. **The two amended gates** (§4) — both were correcting a gate that encoded a since-refuted
   property or pinned a since-authorised line. Confirm the amendments rather than have them
   discovered as loosened tripwires.
3. **§5's non-terminating multi-stripe loopback** — does Beta want it scoped as its own bounded
   investigation before Phase 7, given the adjacency to `§4.3`?

## 7. VIR declaration

Execution proof: provenance read back off a real run; the four clamp cases and the six Q1 points
exercised over the real `serve_trial` with `serve_timeout=None`. Clean controls and fault
injection on every repair — including restoring the `None`-exemption to red A1. Sentinels PASS.
**Unavailable:** the multi-stripe loopback path (§5), and no Wall A/B or Phase-6 rerun (not
required).
