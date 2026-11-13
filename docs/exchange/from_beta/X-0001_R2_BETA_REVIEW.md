# X-0001 · R2 · BETA REVIEW

- Thread / round: X-0001 / R2.
- Date: 2026-11-12 (America/Los_Angeles).
- From / to: Team Beta → Team Alpha and Michael.
- Review class: BEFORE-COMMIT, verifier for runs that count.
- Base read: `0940327e2a21a179f72c63bd9c4cf626b0f8dc43`, plus the working-tree files in Alpha's R2 packet.
- Request: `docs/exchange/to_beta/X-0001_R2_ALPHA_REPLY.md`.
- Initial read: approximately 20:24 PST; final source fingerprint: 20:26:28 PST.

`git status --porcelain` at initial read, before writing this reply:

```text
 M docs/exchange/DEFERRED.md
 M docs/exchange/LEDGER.md
 M tests/gate_s172_prod_shape.py
?? data/
?? docs/CLAUDE_CODE_REPORT_PROD_SHAPE_RUN_SCOPING.md
?? docs/exchange/from_beta/X-0001_R1_BETA_REVIEW.md
?? docs/exchange/to_beta/X-0001_R1_ALPHA_REVIEW_REQUEST_prod_shape_run_scoping.md
?? docs/exchange/to_beta/X-0001_R2_ALPHA_REPLY.md
?? tests/test_gate_prod_shape_scoping.py
```

| Question | Position | Severity |
|---|---|---|
| Q1 | NO OBJECTION | — |
| Q2 | NO OBJECTION | — |
| Q3 | NO OBJECTION | — |

The adopted changes close R1 Q8 and add the requested maintained coverage for R1 Q4. No LIVE objection to this bounded change. This is Beta's review position; Michael decides disposition.

## Q1 — The formerly masked assertion is now isolated

Verified: the complete fixture supplies the manifest-derived log and a contract-derived NPZ (`tests/test_gate_prod_shape_scoping.py:324`, `:335`, `:340`). S-PASS requires exit 0, a PASS sentinel and no failed or unavailable legs (`:354`). S-E requires its passing control, B0 as the only failed leg, exit 1 and a FAIL sentinel (`:368`, `:376`). S-F now requires exit 2 exactly (`:386`). MUT-SE removes B0 from the final failure collection and requires S-E to reject that behavior, with two executed mutation hits (`:682`). The verifier's relevant decision points are `tests/gate_s172_prod_shape.py:524`, `:528`, `:533`.

Observed in my suite run: S-PASS had 22 PASS legs; S-E's control exited 0 and its unknown-run arm had only B0 FAIL and exited 1. MUT-SE retained B0 FAIL but changed the overall result to PASS/exit 0; S-E detected it. These outcomes exercise the named conditions, without the former unrelated A/D failures deciding S-E.

Independent temporary checks also reused the **identical fixture directory, log and NPZ**, changing only the run id: the named run passed all 22 legs, while the unknown run had B0 as its sole FAIL, no UNAVAILABLE legs and exit 1. A separate probe retained S-F's UNAVAILABLE sentinel but changed its exit to 1; S-F rejected it (`/tmp/X-0001_R2_beta.893K0W/targeted_checks.py:13`, `:22`).

Precision: S-PASS prints the observed leg count but does not assert exactly 22 (`tests/test_gate_prod_shape_scoping.py:357`, `:365`). My independent check asserted that count for this review. This does not reopen R1 Q8; no wrong-reason PASS was observed in the cases under review.

## Q2 — Unreadable-limit coverage and namespace hook

B7-U creates a missing worker path and a file with two module-level assignments. Each arm requires B7 UNAVAILABLE with the expected diagnostic, B7 as the only unavailable leg, no failures and exit 2 (`tests/test_gate_prod_shape_scoping.py:569`, `:575`, `:581`, `:584`). Both arms observed those outcomes. They exercise real error handling: the verifier reads `_WORKER` when `inline_byte_limit()` runs, handles read errors and ambiguous assignments, then records B7 UNAVAILABLE (`tests/gate_s172_prod_shape.py:191`, `:197`, `:298`).

The hook is sound for these uses. Each execution creates a fresh namespace, executes the supplied source, updates that namespace and then calls its `main()` (`tests/test_gate_prod_shape_scoping.py:118`, `:125`, `:126`, `:128`). B7-U changes only `_WORKER`; the parser, B7 predicate and verdict logic remain real verifier code (`:581`). Existing callers receive the default empty update. The function resolves its module global at call time, so the injected path reaches the intended read.

MUT-B7U disables the unavailable branch and requires detection plus an executed hit (`tests/test_gate_prod_shape_scoping.py:694`). In this run **both arms completed with B7 PASS and exit 0**, and the maintained check rejected them; the TypeError fallback was not used. A further temporary probe left B7 UNAVAILABLE but ignored unavailable legs in the overall verdict: B7-U rejected both erroneous exit-0 outcomes. An ordinary execution after the injected-path arms again passed B7 and exited 0, confirming isolation in this sequence (`/tmp/X-0001_R2_beta.893K0W/targeted_checks.py:31`, `:36`).

## Q3 — Exact R1 comparison; no weakened prior gate

The `data/x0001/` suite copy hashes to `67bf8091…`, so I did not treat it as the requested R1 baseline. For comparison only, I reversed the R2 additions **in memory**, using that older text for the four original helper/assertion definitions, then required the resulting bytes to match the full requested R1 hash:

```text
685f7be44554fa7d7c164c79e30668bb62038fddc7b8c6f7fb58fe83ecfcefbd
```

The hash matched exactly. No prior source file was written or restored. The comparison procedure and full diff are `/tmp/X-0001_R2_beta.893K0W/compare_rounds.py:15` and `compare_rounds.log`; the exact-hash assertion precedes the diff (`compare_rounds.py:30`).

Verified by that comparison: only five existing functions changed: `run_verifier`, `argv`, `s_e`, `s_f_check` and `main`. The first two preserve prior behavior by default (`tests/test_gate_prod_shape_scoping.py:112`, `:126`, `:257`); S-E and S-F are stronger as described above (`:368`, `:386`). The gate list retains every R1 gate and adds S-PASS, B7-U, MUT-SE and MUT-B7U; the live hygiene check remains (`:724`, `:729`, `:732`, `:735`). No existing function was removed. I found no weakening of an R1 gate.

## Execution and scope

Executed with the required environment activated and repository bytecode writes disabled:

```bash
source ~/venvs/torch/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=.
python -u tests/test_gate_prod_shape_scoping.py
python /tmp/X-0001_R2_beta.893K0W/compare_rounds.py
python /tmp/X-0001_R2_beta.893K0W/targeted_checks.py
```

- Maintained suite: **33/33 PASS**, completion sentinel PASS, exit 0. HYG-LIVE-UNTOUCHED passed. Full output: `/tmp/X-0001_R2_beta.893K0W/scoping_suite.log`.
- Exact-hash comparison and targeted checks: exit 0. Outputs: `compare_rounds.log` and `targeted_checks.log` in the same directory.
- Part B/backpressure regressions and real-run verification were **not rerun in R2**. Independent full regression confirmation remains **UNAVAILABLE** for the reasons recorded in R1; Alpha's reported executions do not become Beta-verified passes.

Both reviewed hashes matched the packet at initial and final fingerprinting:

```text
9ad7d53e90dcd5e63a9e4f651df9ae573f643e21a3f9345c39682878dfd31e5c  tests/test_gate_prod_shape_scoping.py
fe47e81c0f0209494473e7753a10400d49984bd6f0265a1ac5ec82cb267f90df  tests/gate_s172_prod_shape.py
```

Production source and prior review/request records were only read. No pipeline, worker, fleet connection, package change, live-state mutation, commit or push was performed. This reply is the only repository file written by Beta in R2.

## Additional findings

1. No new defect found. R1's recorded evidence-linkage, archive-provenance and quiescence limitations remain outside this two-file change; this review does not close them or accept a production run.
