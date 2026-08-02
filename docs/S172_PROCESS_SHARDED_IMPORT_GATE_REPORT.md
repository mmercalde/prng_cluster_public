# S172_PROCESS_SHARDED_IMPORT_GATE_REPORT.md

**Deliverable:** the `process_sharded` CPU-only import gate (Beta-REQUIRED hardening).
**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PROCESS_SHARDED_IMPORT_GATE.md` REV1 + the
mid-session ADDENDUM (contamination guard; mutant must red for the right reason).
**Box:** VM101 `zeus-ubuntu-vm` `192.168.3.177`, user `michael`, venv `~/venvs/torch`
(Python 3.10.12, torch 2.5.1+cu121, cupy 13.5.1).
**Scope honoured:** test-side only. **No production file was modified.** `assert_cpu_only`
and `_FORBIDDEN_GPU_MODULES` were exercised, never revised.
**Not done, per contract:** no commit, no push, no `watcher_agent.py --run-pipeline`.

**Status: PASS — STOPPED for Team Alpha review.**

**Headline:** import gate **7/7 green, 3/3 mutants red**. D5 **24/24 → 24/25**, all 24
pre-existing gates still green; the one red is an uncommitted-file artifact (Finding 2), not
a regression. Checkpoint census **25 → 25, zero created**. Four findings in §8 — Finding 2
needs a decision, Finding 1 is the one place I departed from the brief's literal wording.

---

## 0. Base commit, and one piece of drift that is not drift

The brief is based on `55daf4b`. Session start was `285cbd7`. **HEAD moved to `9470750`
mid-session** (Michael committed `docs: D6.2 identity addendum ... + D6.3 read-only
investigation brief`).

That commit is **docs-only** — 2 files, +220 lines, both under `docs/`. Verified:

```
git diff --stat 285cbd7 HEAD
 docs/CLAUDE_CODE_INSTRUCTIONS_S172_D6_3_RETENTION_INVESTIGATION.md | 129 +++++
 docs/TEAM_ALPHA_D6_2_IDENTITY_ADDENDUM.md                          |  91 +++++
```

The two audited files are **byte-identical across `55daf4b` → `285cbd7` → `9470750`**:

```
git diff --stat 55daf4b HEAD -- tests/test_s172_phase5_d5_process_sharded.py \
                                miner/assembly_shard_worker.py
(empty)
```

So every line anchor in the brief resolves exactly, and all measurements below are valid
for `55daf4b` and for HEAD alike.

---

## 1. The four gaps — all four CONFIRMED at HEAD, zero drift

Each anchor read at source, not from the brief.

| Gap | Anchor | Line at HEAD | Verdict |
|---|---|---|---|
| 2.1 probe duplicates the forbidden list | `:1436` | `"gpu_modules": sorted(m for m in ("torch", "cupy") if m in sys.modules),` | **CONFIRMED** |
| — production authority | `assembly_shard_worker.py:170` | `_FORBIDDEN_GPU_MODULES: Tuple[str, ...] = ("torch", "cupy")` | **CONFIRMED** |
| 2.2 probe never invokes the guard | `:1483` | `assert probe["gpu_modules"] == [], (` | **CONFIRMED** |
| 2.3 child never imports the Step-1 surface | `:1441-1446` | `_run_probe` → `concurrent.futures.ProcessPoolExecutor(..., mp_context=ctx)` | **CONFIRMED** |
| 2.4 runtime injection covers one module | `:1488` / `:1517` | `injected = "torch" not in sys.modules` … `assert not ({"torch", "cupy"} & set(names))` | **CONFIRMED** |
| — guard + docstring | `assembly_shard_worker.py:231-244` (docstring `:232-238`) | `def assert_cpu_only() -> None:` | **CONFIRMED** |

`g_no_gpu` occupies `:1480-1517` (the brief's `~:1480-1518` includes the trailing blank).
The 2.4 block is `:1486-1496` counting its leading comment; the injection statement itself
is `:1488`. Both are presentation details, not drift.

**Gap 2.1 restated as measured:** the probe's literal and production's tuple agree *today*
— that is exactly the defect. A third forbidden module added to production would leave the
probe checking two, and the gate would stay green while the invariant widened past it.

---

## 2. What "the real Step-1 module surface" is — and why it is not the host module

### 2.1 The surface used

**`miner` + `miner.step1_ingress`** — and these two names are **never written in the gate**.
They are AST-derived at run time from Step-1's own source by `_step1_surface()`, which
extracts every *module-scope* import of the miner package from
`window_optimizer_integration_final.py`:

- `:74` `from miner import DEFAULT_WORKER_ADMISSION_TIMEOUT, run_trial_miner`
- `:84` `from miner.step1_ingress import MinerIngressError, build_assembling_sink, ...`

Both sit inside `try/except ImportError` at module level; the extractor descends into `try`
/ `if` / `with` but **never into a function or class body**, because an import inside a
function does not run at import time and so cannot put a library into a fresh interpreter's
`sys.modules`. (`:294` `from miner.range_miner_worker import load_residue_window` is
function-scope and correctly excluded — it is reached transitively anyway.)

Derived rather than named so the surface **cannot silently narrow**: if Step-1 gains a third
miner import, the gate widens automatically and keeps covering all of it.

Those two roots transitively load **all 9 miner modules**, measured in-child:

```
miner, miner.assembly_backends, miner.assembly_shard_worker, miner.dataset_authority,
miner.range_miner_coordinator, miner.range_miner_npz_writer, miner.range_miner_protocol,
miner.range_miner_worker, miner.step1_ingress
```

This is **exactly the chain `assert_cpu_only`'s docstring makes its claim about**, confirmed
at source:

- `miner/range_miner_npz_writer.py:50` → imports the coordinator  *("the D1.1 engine…")*
- `miner/range_miner_coordinator.py:54` → imports `range_miner_worker`  *("…which imports the coordinator, which imports that module")*

`G-SURFACE-GUARD` asserts all six load-bearing members are present, so "nothing leaked"
can never degenerate into "nothing was imported".

### 2.2 FINDING — the brief's named module cannot be the surface

The brief calls `window_optimizer_integration_final` "the real Step-1 surface". Taken
literally that requirement is **unsatisfiable against a clean tree**, and I did not
implement it literally. Measured:

```
import window_optimizer_integration_final   →  cupy present in sys.modules
window_optimizer_integration_final.py:53   from sieve_filter import load_draws_from_daily3
sieve_filter.py:52                         import cupy as cp        # module scope
```

Importing the host module and then calling `assert_cpu_only()` **reds on the unmutated
tree** — the clean control required by §3 would fail. This is not a defect: Step-1 *is* the
GPU sieve host, and it legitimately holds a GPU context. The guard's claim is about the
**miner subgraph** Step-1 pulls in, not about Step-1's own interpreter.

Rather than assume that distinction, **`G-HOST-BOUNDARY` measures it**: it imports the host
module in a fresh interpreter, records which forbidden modules arrive and the file each came
from, and asserts **none of them originated inside `miner/`**. If a GPU library ever reaches
the interpreter *from the miner package*, that arm reds and the real finding surfaces
immediately.

Also relevant to a future reader: Step-1's two miner imports are wrapped in
`except ImportError`, so a broken miner package would be **silently swallowed** there. The
gate's child imports the surface **directly**, where an ImportError is loud and is
classified `INCOMPLETE` — never a pass.

---

## 3. Placement, and how the new gates are reached

**A new file: `tests/test_s172_process_sharded_import_gate.py`** (7 gates, 3 mutants,
standalone-runnable, own sentinel).

Chosen over extending `g_no_gpu` because gap 2.1 requires asserting that **no
GPU-module-name literal appears anywhere in the new gate**. The existing `g_no_gpu` arms
legitimately contain those literals (`:1488`, `:1493`, `:1517` — their AST arms are written
*against* them), so the assertion could not be made total inside that file without either
weakening it to a line range or editing arms the brief freezes. In a separate file the
assertion is total: `G-NO-DUP-LIST` scans the file's own source.

**Reached from D5's aggregate** by one added `_check` row in `main()`, with
`g_import_gate()` calling the existing `_run_suite(...)` helper against
`"7/7 import-gate checks green"`. Run as a subprocess so the new file's own sentinel and
mutation evidence stay intact.

**It is placed immediately before `NR`, not beside `G-NO-GPU`** — semantically it belongs
next to the arm it extends, and that is where I first put it, but that placement **reds
G-RSS deterministically**. See Finding 4: `RUSAGE_CHILDREN.ru_maxrss` is a process-lifetime
high-water mark, and this gate deliberately spawns GPU-library-importing children. The
ordering constraint is recorded in a comment at the call site so it is not silently undone.

**The D5 edit is purely additive — zero deletions.** Verified two ways:

```
git diff HEAD -- tests/test_s172_phase5_d5_process_sharded.py | grep '^-' | grep -v '^---'
(no output)

diff <(git show HEAD:…d5_process_sharded.py | sed -n '1427,1518p') \
     <(sed -n '1427,1518p' …d5_process_sharded.py)
IDENTICAL: _probe_child, _run_probe, g_spawn, g_no_gpu byte-for-byte unchanged
```

**No existing arm was changed to accommodate a new one.** §5's stop-condition was not
triggered by the gate itself — but see Finding 2 in §9.

---

## 4. The fresh-interpreter invocation

Not a multiprocessing child: a `spawn` child is prepared from the parent's state and
re-imports the parent's `__main__`, which muddies any claim about what a clean interpreter
pulls in. This is a bare interpreter that imports only what it is told on `argv`.

```
argv : [sys.executable, "<tmp>/s172_import_gate_runner_*/cpu_only_child.py",
        "miner", "miner.step1_ingress"]
cwd  : /home/michael/distributed_prng_analysis
env  : PYTHONPATH             = <repo>                       (clean control)
                              = <mutant tree>:<repo>         (mutant runs)
       PYTHONDONTWRITEBYTECODE = 1
       PRNG_CHECKPOINT_ROOT    = <tmp sandbox>                (ADDENDUM §1)
timeout: 300 s → AssertionError naming the timeout; a timeout is INCOMPLETE, never a pass
```

The child script lives in its **own** directory, never inside a candidate import tree, so
`sys.path[0]` can never shadow the tree under test — resolution is decided purely by
`PYTHONPATH`.

**Execution proof (VIR-1).** Every child emits, before and around its assertion:
`CHILD_PID`, `CHILD_EXE`, `SURFACE_REQUESTED`, `FORBIDDEN_RESOLVED` (read from production),
`WORKER_FILE`, one `IMPORTED` line per surface module *with the file it resolved from*,
`MINER_MODULES`, `PRESENT` + `PRESENT_FILE`, `CHECKPOINT_ROOT`, `REACHED_ASSERTION`, then
`GUARD` / `GUARD_TYPE` / `GUARD_MRO` / `GUARD_MESSAGE`, then `SENTINEL`.

`_assert_executed()` applies to **every** child and rejects: a missing sentinel
(`INCOMPLETE`), `UNAVAILABLE` (a finding — this gate has no fleet dependency), a child PID
equal to the parent's, a different surface than requested, and — critically — a
`FORBIDDEN_RESOLVED` that differs from production's tuple, which would mean the gate and the
authority had drifted.

---

## 5. Results

### 5.1 The gates — 7/7 green

| Gate | Closes | Result |
|---|---|---|
| `G-SURFACE` | 2.3 | surface derived from Step-1 source, non-empty, every member on disk |
| `G-SURFACE-GUARD` | **2.2 + 2.3** | fresh interpreter imports the surface, **invokes production `assert_cpu_only()`** → exit 0, `SENTINEL=PASS` |
| `G-NO-DUP-LIST` | **2.1** | no GPU-module-name literal anywhere in the gate |
| `G-RUNTIME-INJECTION` | **2.4** | every forbidden module injected in turn |
| `G-MUTANTS` | §3 | 3 module-scope-import mutants, four-part rule |
| `G-HOST-BOUNDARY` | finding §2.2 | host's GPU context measured, provably not from `miner/` |
| `G-NO-CONTAMINATION` | ADDENDUM §1 | census before == after |

### 5.2 Gap 2.1 — the forbidden list is read, never restated

`G-NO-DUP-LIST` scans the gate's **own source** four ways and requires all clean:
AST string constants, identifiers/attributes/import aliases, **comment tokens** (via
`tokenize` — comments are not in the AST, so an AST-only pass would miss a latent copy of
the authority in prose), and the embedded child-script source.

It then proves it is **not vacuous**: the scanner is run against a synthesised
`f"import {_FORBIDDEN[0]}"` and must detect it. Without that arm the check would pass on any
file, including one that duplicates the list.

The single point where a forbidden name enters the file is
`_FORBIDDEN = tuple(ASW._FORBIDDEN_GPU_MODULES)`. Resolved value at run time: **2 modules**.

### 5.3 Gap 2.4 — per-module runtime injection, **2/2**

Iterating the production tuple, not a literal pair. For **each** name in turn: inject a
sentinel, assert `ShardArtifactError` fires, assert the message **names the injected module**
*and* carries the holding pid, restore `sys.modules` in a `finally` **in every case**, then
assert the sentinel did not leak.

| Injected | Guard fired | Message names module | Restored |
|---|---|---|---|
| `_FORBIDDEN[0]` (`torch`) | ✅ `ShardArtifactError` | ✅ | ✅ |
| `_FORBIDDEN[1]` (`cupy`) | ✅ `ShardArtifactError` | ✅ | ✅ |

Negative controls: `assert_cpu_only()` passes **before** the loop and **after** it, so a
leaked sentinel would red the arm rather than silently poison later tests.

### 5.4 §3 — the mutants: 3/3 red, each for the right reason

Injection is built as `f"import {module_name}"` from the production tuple (no literal), into
a **copy** of the miner package in a temp tree; the live file is never touched. The copy wins
only because its directory is prepended to the child's `PYTHONPATH`.

| Mutant | Target | Exit | Exception | Message names module |
|---|---|---|---|---|
| `M-WORKER-TORCH` | `miner/range_miner_worker.py` | **3** | `ShardArtifactError` | ✅ `['torch']` |
| `M-WORKER-CUPY` | `miner/range_miner_worker.py` | **3** | `ShardArtifactError` | ✅ `['cupy']` |
| `M-COORD-TORCH` | `miner/range_miner_coordinator.py` | **3** | `ShardArtifactError` | ✅ `['torch']` |

Sample signature:

```
exit=3 ShardArtifactError: assembly shard worker (pid 13008) has GPU module(s)
['torch'] in sys.modules — assembly is CPU-only work and a worker must never
hold a GPU context (§6.7.A)
```

`range_miner_worker` is mutated because it is the module the guard's docstring names by
hand; the coordinator is mutated to prove the gate covers the **graph**, not one file.

**ADDENDUM §2 — red for the right reason, enforced not assumed.** Each mutant asserts:
`returncode != 0`; `SENTINEL == FAIL` (not `INCOMPLETE`); `GUARD_TYPE ==
ShardArtifactError.__name__`; `ProcessShardedAssemblyError` in the exception MRO;
**`ImportError` and `ModuleNotFoundError` explicitly absent from the MRO**; the injected
module named in the message; `REACHED_ASSERTION == 1` (rules out a collection/loader death);
and a timeout raises rather than passing.

**Four-part kill rule, per mutant:**

1. **applies once** — module-scope pre-check via `_module_scope_imports` (a *function-scope*
   import of the same library is **not** a pre-existing mutation: it is the arrangement the
   docstring blesses, and the injected defect is precisely the *lift* of such an import to
   module scope); injected text asserted to occur exactly once; re-parsed and confirmed to be
   a single module-scope `Import` node.
2. **mutated path executed** — child reports at least one surface module resolved from the
   mutant tree, `WORKER_FILE` under the mutant tree, and `PRESENT == [injected module]`.
3. **reached the credited assertion** — `REACHED_ASSERTION=1`, and the same runner passes
   clean in `G-SURFACE-GUARD` (the positive control).
4. **red from the injected defect** — guaranteed by (3) plus the type/MRO/message assertions.

**Clean control, same invocation style:** unmutated tree → `exit=0`, `SENTINEL=PASS`,
`PRESENT` empty, 9 miner modules loaded. A **post-mutation control** re-runs the live tree
after all three mutants and must still pass, proving no mutant escaped its temp copy.

Two real defects were caught by these controls during development and fixed, rather than
banked as kills:
- the first vacuity pre-check walked the whole AST and so mistook `range_miner_worker`'s
  *legitimate function-scope* GPU imports for a pre-existing mutation;
- the first injection point landed **above** `from __future__ import annotations`, producing
  a `SyntaxError`. The harness refused to credit it — it reported `INCOMPLETE`, exactly as
  the four-part rule requires. That is itself evidence the rule is load-bearing here.

### 5.5 ADDENDUM §1 — contamination guard and census

Step-1 fixes `_FLUSH_RUN_ID_DEFAULT` from `(hostname, pid, epoch)` **at import time**
(`window_optimizer_integration_final.py:448`), and every fresh interpreter is a new pid — so
an unredirected flush would scatter run-id directories into the real checkpoint root, where
they would be indistinguishable from production ones.

**Guard:** every child gets `PRNG_CHECKPOINT_ROOT=<tempfile dir>`. The env-var name and
`.s172_checkpoint` are **AST-read from Step-1's source** (`_CHECKPOINT_ROOT_ENV`,
`_CHECKPOINT_DIRNAME`), not hardcoded. They are read by AST rather than by import,
deliberately: importing Step-1's host into the test process would pull a GPU library into the
**parent** and break `G-RUNTIME-INJECTION`'s negative controls.

**Redirect proven to take effect, not merely requested:** the child discovers any
`_flush_checkpoint_root` resolver by attribute on the modules it imported, calls it, and
reports the resolved path. `G-HOST-BOUNDARY` asserts it equals the sandbox.
Evidence: `checkpoint redirect honoured in-child — resolved root == sandbox for 1 resolver(s)`.

**Census of the real root** `/home/michael/distributed_prng_analysis/.s172_checkpoint`:

| | count |
|---|---|
| **before** | **25** |
| **after** | **25** |
| **created by this work** | **0** |
| removed | 0 |

`G-NO-CONTAMINATION` brackets the whole run, asserts created == removed == 0, and also
asserts the sandbox is a real directory **outside the repo tree**.

Independently corroborated: the newest directory under the real root is
`zeus-ubuntu-vm-80774-1785633414`, mtime **2026-08-01 18:17** — the day *before* this
session. No directory carries any child pid from this work. **Import alone does not flush**;
the guard is defence in depth, and it held.

### 5.6 D5 before / after

| | measurement |
|---|---|
| **D5 before** | **24/24 green**, 0 FAIL — measured at clean HEAD, new file moved out of the tree |
| **D5 after** | **24/25 green** — 25 gates (24 pre-existing + `G-IMPORT-GATE`). **All 24 pre-existing gates green.** The single red is `NR`, whose sole root cause is Finding 2 (the new file is untracked, so Phase-4 Gate 22 reds). |
| import gate | **7/7 green**, 3/3 mutants red |

**§5's non-regression condition is met:** every one of the 24 gates that existed before this
work is green after it, `G-IMPORT-GATE` itself is green, and the count rose 24 → 25 by
addition only. The `NR` red is a **working-tree artifact of an uncommitted file**, not a
regression — it disappears the moment the file is committed (Finding 2), and it did not
appear in the 24/24 baseline precisely because the file was absent from the tree then.

Verified that `NR` has no other cause — the complete set of failure headers across the
entire nested run is exactly two, and the second is caused by the first:

```
--- Gate 22: coexistence (use_range_miner, PWC/ZMQ) ---
    AssertionError: unexpected changed .py files: {'tests/test_s172_process_sharded_import_gate.py'}
--- NR: D1.1 18/18, D2 7/7, ... ---
    AssertionError: tests/test_s172_phase5_d1_engine.py exited 1
```

`G-RSS` and `G-MUTANTS` (18/18) are green in this run — see Finding 4 for the intermediate
run in which they were not, and why.

---

## 6. Verification-integrity sentinels

| Control | Result |
|---|---|
| execution proof (VIR-1) | ✅ every child echoes resolved tuple, imports + resolved files, pid, assertion marker |
| clean control (VIR-2) | ✅ unmutated tree exit 0 / `PASS`, same invocation style; plus a post-mutation control |
| fault injection (VIR-3) | ✅ 3 module-scope mutants + 2/2 runtime injections |
| completion sentinel (VIR-4) | ✅ `PASS \| FAIL \| UNAVAILABLE \| INCOMPLETE`; only `PASS` accepts; a child dying without a sentinel is `INCOMPLETE` |
| unavailable-observer (VIR-5) | ✅ **no arm reported `UNAVAILABLE`** — no fleet dependency; all rigs irrelevant to this gate |
| audit claim scope (VIR-6) | repo-scoped; `tests/test_s172_phase5_d5_process_sharded.py` + `miner/assembly_shard_worker.py`, byte-identical `55daf4b`→HEAD |

**Unavailable surfaces:** none blocked this work. The runtime import graph was established by
execution on VM101, which is the only way to establish it.

---

## 7. Files touched

| File | Change |
|---|---|
| `tests/test_s172_process_sharded_import_gate.py` | **NEW** — 7 gates, 3 mutants |
| `tests/test_s172_phase5_d5_process_sharded.py` | **+1 `_check` row, +1 `g_import_gate()`. Purely additive, zero deletions.** |
| production files | **none** |

---

## 8. Findings

### Finding 1 — the brief's Step-1 surface is unsatisfiable as literally written

`window_optimizer_integration_final` holds `cupy` at import time via
`sieve_filter.py:52`. Importing it then calling `assert_cpu_only()` reds on a clean tree, so
§3's mandatory clean control could not pass. Resolved by using the **miner-side surface
Step-1 imports** — which is the graph the guard's own docstring describes — and by adding
`G-HOST-BOUNDARY` to *measure* the exclusion instead of assuming it. Full detail in §2.2.
**No production change is proposed; this is Step-1 behaving correctly.**

### Finding 2 — Phase-4 Gate 22 reds while the new test file is uncommitted

`tests/test_s172_phase4_coordinator.py:1607-2363` (`gate22_coexistence`) computes
`changed_py` from `git status --porcelain`, which **includes untracked files**, and asserts
`changed_py <= allowed`. The new file is not in `allowed`, so it reds:

```
AssertionError: unexpected changed .py files: {'tests/test_s172_process_sharded_import_gate.py'}
```

That propagates: Gate 22 → Phase-4 → D1.0 → D2 → **D5's `NR` arm**.

Notes:
- `tests/test_s172_phase5_d5_process_sharded.py` **is** already in `allowed`, so my edit to
  it is not implicated.
- `git add` does **not** help — staged files still appear in `--porcelain`.
- This is the known standing sensitivity, not a new defect.
- Registering each new test `.py` in Gate 22's `allowed` set with a rationale comment is the
  **documented, established pattern** (`tests/test_chapter2_content_gate.py` and ~50 others
  are listed there).

**I did not make that edit.** Gate 22 is an existing arm in another suite, and the brief's
§5 instruction is to stop and report rather than change an existing arm to accommodate a new
one. **Two resolutions, Team Alpha's call:**

1. **Commit the file** (Michael's step anyway) — `--porcelain` goes clean and Gate 22 passes
   with no code change at all. Preferred.
2. Register `"tests/test_s172_process_sharded_import_gate.py"` in Gate 22's `allowed` set.

### Finding 3 — my first baseline was self-contaminated (disclosed, corrected)

The first baseline run reported 23/24. That red was **caused by my own work**: I created the
new file while that run was still inside its `NR` phase, and Gate 22 (Finding 2) saw it. I
discarded that measurement, moved the new file out of the tree, confirmed
`git status --porcelain` was clean, and re-measured: **24/24 green**. The 24/24 figure is the
one reported in §5.6. Recorded here because a 23/24 line in a log is otherwise
indistinguishable from a real regression.

---

### Finding 4 — G-RSS has a latent ordering fragility (pre-existing; exposed, then avoided)

`_RusageChildrenSampler.__exit__` (`tests/test_s172_phase5_d5_process_sharded.py:2107-2119`)
reads `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss`. Its docstring describes this as "the
maximum of any SINGLE reaped child", and G-RSS then asserts
`tree.peak_rss > rusage.peak_rss` (`:2163`).

`ru_maxrss` for `RUSAGE_CHILDREN` is in fact a **process-lifetime high-water mark over every
child the process has ever reaped** — it is not scoped to the `with` block. So the assertion
silently depends on **no earlier child in the whole D5 run having exceeded G-RSS's own
two-child tree-sum (~339 MiB)**.

Measured directly on VM101:

```
RUSAGE_CHILDREN before any child :        0 MiB
after a trivial child            :       10 MiB
after ONE torch-importing child  :      378 MiB
after another trivial child      :      378 MiB   <- high-water PERSISTS
```

With the import gate placed beside `G-NO-GPU`, its GPU-importing children raised the mark to
384 MiB and G-RSS red with `tree-sum 339 MiB <= RUSAGE_CHILDREN 384 MiB`, which in turn made
mutant **M8** survive (its detector *is* G-RSS). Both were **deterministic consequences of
ordering, not flakes**, and both cleared once the gate moved after `G-MUTANTS`.

**Handled entirely on my side — no existing arm was modified.** The gate now runs immediately
before `NR`, with the constraint documented at the call site.

**Non-blocking, but worth tracking:** any future D5 arm that reaps a large child before
G-RSS will red it the same way. A scope-correct fix would be to have
`_RusageChildrenSampler` record `ru_maxrss` on `__enter__` as well and compare the
*delta* — but that is an edit to an existing D4/D5 arm and therefore explicitly out of scope
here. Flagged for Team Alpha, not actioned.

---

## 9. What a reviewer should check first

1. **§2.2 / Finding 1** — the Step-1 surface decision. It is the one place I departed from
   the brief's literal wording, and the departure is forced: the literal reading cannot pass
   its own mandatory clean control.
2. **Finding 2** — needs a decision (commit the file, or register it in Gate 22's `allowed`
   set). Until then D5 reads 24/25.
3. **Finding 4** — the G-RSS ordering coupling. Worth knowing about before any future arm is
   added to D5, and worth deciding whether the scope-correct `__enter__`-delta fix should be
   scheduled.
4. **§5.4's two development defects** — both were caught by the four-part rule refusing to
   credit a wrong-reason red, which is the behaviour Beta asked for.

**STOPPED for Team Alpha review. Nothing committed, nothing pushed.**
