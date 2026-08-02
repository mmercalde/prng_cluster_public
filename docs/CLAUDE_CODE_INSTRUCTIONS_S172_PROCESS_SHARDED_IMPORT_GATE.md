# CLAUDE_CODE_INSTRUCTIONS_S172_PROCESS_SHARDED_IMPORT_GATE.md — REV1

**S172 — the `process_sharded` CPU-only import gate. Beta-REQUIRED hardening.**

**Base:** HEAD `55daf4b`. Claude Code on **VM101** as `michael`, venv `~/venvs/torch`. Implement
and iterate; do **NOT** commit, push, or run WATCHER. STOP at the gate.

**Scope: test-side only.** No production file is modified. `assert_cpu_only` and
`_FORBIDDEN_GPU_MODULES` are the surfaces being *exercised*, not revised.

**Why this can run now:** it depends on no pending Beta ruling. Beta stated it is **required
hardening, not a Phase-6 blocker** — the real `process_sharded` arm passed. It runs in parallel
with D6.2's approval cycle.

---

## 0. Beta's requirement, verbatim in substance

A gate that uses a **fresh spawned interpreter**, a **real Step-1 module surface**, invokes the
**production `assembly_shard_worker.assert_cpu_only()`** (*do not duplicate its forbidden list*),
covers **both `torch` and `cupy`**, plus **a mutant introducing a module-level GPU import that
proves it reds**.

## 1. What already exists — read it first

`tests/test_s172_phase5_d5_process_sharded.py::g_no_gpu` (`:1480-1518`) already covers four things,
and this brief must not duplicate them:

1. a spawned-child probe asserting no GPU module is present;
2. in-process fault injection proving `assert_cpu_only` fires;
3. an AST assertion that `assert_cpu_only()` is the **first statement** of `validate_spool_shard`,
   so it cannot be reached only after a GPU context is built;
4. an AST assertion that neither the worker nor the backend module imports `torch`/`cupy` at
   module scope.

**Read `g_no_gpu` in full before writing anything.** The new gate extends it; it does not replace
it, and the existing arms stay.

## 2. The four gaps — each anchored, each verified at `55daf4b`

### 2.1 The probe duplicates the forbidden list — the defect Beta's wording targets

`:1436` builds its answer from a **hardcoded literal**:

```python
"gpu_modules": sorted(m for m in ("torch", "cupy") if m in sys.modules),
```

Production holds the authority at `miner/assembly_shard_worker.py:170`:

```python
_FORBIDDEN_GPU_MODULES: Tuple[str, ...] = ("torch", "cupy")
```

**The two agree today.** That is precisely the problem: adding a third forbidden module to
production would leave the probe silently checking two, and the gate would stay green while the
invariant it exists to protect had widened past it. Beta's *"do not duplicate its forbidden list"*
is not a new constraint — **it is a correction to what is already there.**

**Required:** the child imports and reads `_FORBIDDEN_GPU_MODULES` from production. **AST-assert
that no GPU-module-name literal appears anywhere in the new gate.**

### 2.2 The probe inspects `sys.modules`; it does not invoke the production guard

`:1483` asserts `probe["gpu_modules"] == []` — a **restatement** of what `assert_cpu_only` checks,
computed independently in the test. If `assert_cpu_only` were deleted, weakened, or made
conditional, this arm would still pass.

**Required:** the child **calls `assembly_shard_worker.assert_cpu_only()`** and reports the
outcome, including the exception type and message on failure.

### 2.3 The child does not import the real Step-1 module surface

`_run_probe` (`:1441-1446`) spawns via `ProcessPoolExecutor` with an mp context. A `spawn` child
re-imports the test module and reaches `assembly_shard_worker` — but **not**
`window_optimizer_integration_final`, the real Step-1 surface.

That surface is the one that matters. `assert_cpu_only`'s own docstring
(`assembly_shard_worker.py:232-238`) states the invariant it is asserting: *"`range_miner_worker`
imports torch/cupy only INSIDE its kernel functions, so importing the D1.1 engine — which imports
the coordinator, which imports that module — pulls in no GPU library."* **That is a claim about the
Step-1 import graph, and nothing currently tests it end to end.** A refactor that lifts one
`import torch` to module scope anywhere in that chain would break the invariant while every
existing arm stayed green.

**Required:** a **fresh interpreter via `sys.executable` `subprocess`** — not a multiprocessing
child, which inherits the parent's already-populated `sys.modules` view in ways that muddy the
claim — which:
1. imports the real Step-1 module surface;
2. then calls the production `assert_cpu_only()`;
3. exits non-zero with a diagnostic on failure.

State in the report exactly which module(s) constitute "the real Step-1 module surface" as
implemented, and why.

### 2.4 Runtime fault injection covers `torch` only

`:1486-1496` injects a sentinel for `torch` and asserts the guard fires:

```python
injected = "torch" not in sys.modules
if injected:
    sys.modules["torch"] = sentinel
```

**`cupy` is never injected at runtime.** It appears only in the AST arm (`:1517`). So the runtime
half of the guard is proven for one of two forbidden modules.

**Required:** inject **each** name in `_FORBIDDEN_GPU_MODULES` **in turn** — iterating the
production tuple, not a literal pair — and assert the guard fires and **names the injected
module** in its message. Restore `sys.modules` in a `finally` in every case; a leaked sentinel
would poison every later test in the process.

## 3. The mutant

Beta requires a mutant **introducing a module-level GPU import** that proves the gate reds.

- Inject `import torch` (or `cupy`) at **module scope** into a **copy** of a module in the Step-1
  import chain — never the live file.
- Run the fresh-interpreter gate of §2.3 against the mutated tree.
- **Required outcome:** non-zero exit, and the failure must be `ShardArtifactError` naming the
  module — **not** an `ImportError`, a timeout, or a collection error. A gate that reds for the
  wrong reason is not a proven gate.
- **Four-part kill rule:** prove the mutant applied, that the mutated import path executed, that
  the gate reached its assertion, and that the red came **from the injected defect**.

**Clean control:** the same runner against the unmutated tree must pass, in the same invocation
style. Both results in the report.

## 4. Placement

Add to the existing D5 suite as new gates, or a new file — Claude Code's call. If a new file:
`tests/test_s172_process_sharded_import_gate.py`, and it must be added to whatever aggregate the
D5 suite is invoked through, or it will never run. **State which choice was made and how the new
gates are reached.**

## 5. Non-regression

D5 must stay green at its current count, with the existing `g_no_gpu` arms **unchanged**. If any
existing arm has to change to accommodate the new ones, stop and report — that is a finding.

All commands on **VM101**, `source ~/venvs/torch/bin/activate` first.

## 6. Report

The four gaps confirmed or refuted at HEAD (they were verified at `55daf4b`; report drift); which
modules constitute the Step-1 surface and why; the fresh-interpreter invocation used; the
per-module injection results; the mutant's exit code and exception type with clean control
alongside; gate/mutant counts; D5's before/after counts. Then STOP for Team Alpha review.

---

## Verification-integrity controls (VIR-1…6)

- **execution proof:** the fresh interpreter prints the resolved `_FORBIDDEN_GPU_MODULES` it read
  from production and the modules it imported, so a child that silently did nothing is
  distinguishable from one that passed.
- **clean control:** unmutated tree passes, same invocation style.
- **fault-injection control:** §3's module-level-import mutant, plus §2.4's per-module runtime
  injection.
- **completion sentinel:** `PASS | FAIL | UNAVAILABLE | INCOMPLETE`; only `PASS` accepts. A
  subprocess that dies without printing its sentinel is **INCOMPLETE**, never `PASS`.
- **unavailable-observer behavior:** this gate has **no fleet dependency** — it must pass with all
  rigs down. If any arm reports `UNAVAILABLE`, that is a finding.
- **audit claim scope:** repo-scoped, `tests/test_s172_phase5_d5_process_sharded.py` and
  `miner/assembly_shard_worker.py` at `55daf4b`.
- **searched surfaces:** tracked repo at `55daf4b`.
- **unavailable surfaces:** host state on VM101; uncommitted local modifications; the runtime
  import graph as deployed, which only execution on VM101 can establish.
