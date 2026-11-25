# READY · TASK_SWITCH_REGISTER + ADDENDUM_1 (window g, worktree e2e-grader) · 2026-11-23

## Plain summary

Following the 19:00 addendum, I did **not** build a parallel register. The switches belong in the system that already exists:
- `agents/registry/parameter_registry.py` (Pydantic `ParameterSpec`) for what a switch is and its legal bounds;
- `agent_manifests/*.json` (`default_params`, `args_map`) for what WATCHER can pass.

**What I delivered:**
1. **The map.** How a switch is known, reached, proposed, applied, proved and stopped, with file and line, and which links work today. Chain A is the working template.
2. **The checker.** It classes every switch found in code, on main and in the worktrees:

   | Class | All rows | Step-script switches |
   |---|---|---|
   | Registered and reachable | 17 | **8** |
   | Registered but unreachable | 1 | **1** |
   | Not registered | 1,913 | **174** |

   Each unregistered switch gets the exact `ParameterSpec` and manifest entry that would register it.
3. **A draft patch set.** Not applied. It registers today's 281 new switches as 275 specs in the existing registry format, with defaults unchanged, plus manifest fragments for the three step manifests.
4. **The gaps.** 19 to-dos with their recorded IDs (B1, B3, B12, B17, S1, S3, LL-15, LL-16, LL-18, D6 tripwire) and the NEW ones marked.

Everything ran on CPU, 2 cores at most, under `nice`. Nothing touched the main tree, nothing was committed, and nothing was applied to a governed or hand-landed file.

## Prior record

It is in full in `AUTONOMY_SWITCH_MAP_20261123.md` §Prior record: the searches, the documents found, the labels, and what was not searched.

## Files (all untracked in this worktree)

| File | sha256 |
|---|---|
| `docs/drafts/loop_v1/AUTONOMY_SWITCH_MAP_20261123.md` (map §1, checker §2, patch §3, gaps §4) | `3a945503165313ce634070ed2ea19bf4bc10fcfcb64316f85373c4a5b290bee5` |
| `docs/drafts/loop_v1/SWITCH_CHECK_20261123.md` / `.json` (generated; do not edit) | `59e740f9…cb77` / `68b4068a…c307` |
| `docs/drafts/loop_v1/switch_patches_20261123/` | see its `SHA256SUMS`: `.patch` `7654fa9b…`, `.patched` `9f52ed5f…`, `manifest_fragments.json` `46e5560f…` |
| `scripts/diagnostics/switch_check.py` | `ff59eb3863efb62ad769ea8d47026bebfb4fa2fd461902a3339033e901a40966` |
| `tests/test_switch_check.py` | `418a8f0897b28e3f38aaa47e78b393d6c25b9ca68f2930383191a82e84f4b148` |
| `tests/mutate_e2e_grader.py` (now also runs the switch-check mutants; `--target sc` runs only those) | `340492e2b5f4f83421c7b0a158328b5643da9d6607ea48d6f91a519a2ddbebea` |

## Tests

| Suite | Result | Log |
|---|---|---|
| `test_switch_check.py` | **10/10** | `logs/e2e_grader/sc_selftest_run3.log` |
| Full mutation suite | **74/74 killed and credited**, all baselines pass; 121.5 s on 2 workers | `logs/e2e_grader/all_mutants_run5.log` |
| Real run | 16 s, 80 MB, one core | `logs/e2e_grader/switch_check_real_run2.log` |

The mutation suite covers grader 47, `compare_settings` 11 and switch check 16.

**Two bugs found and fixed during the build, both in my checker:**
- **Basename match:** the registry lookup matched by basename, so `backups/` copies were classed as registered. It now requires the registered path.
- **Positional style:** `arg_style: positional` (step 2) and the key WATCHER pops (`output_file`) were not modelled. Both now are.

## Not verified

- **Run-record evidence:** only the records on disk count.
- **Same-idea-under-two-names (311):** these are candidates, not findings.
- **The patch:** checked for syntax, Pydantic import, unchanged existing specs and valid defaults; not run inside the agents.
- **Shell wrappers:** their forwarding to the Python they call is not traced.

## What I ran, and a stray cache I made (lead note, after the 19:25 reboot)

**What I ran in this task, all as michael on VM101 CPU:**

| Run | Where | Result |
|---|---|---|
| `switch_check.py` | Real run at 19:12:58, outputs in this folder | – |
| `switch_check.py` trials | `data/switch_check_trial/t1`, `t2` | – |
| `tests/test_switch_check.py` | Fixture git repo and 2 worktrees under `data/e2e_grader_selftest/sc_run_*` | – |
| `tests/mutate_e2e_grader.py` | Full suite plus `--target sc` | – |
| Pydantic import check of the patched registry | – | 29 → 304 specs |

- Nothing ran on the GPU, and nothing ran in the main tree, which was read only through `git show` and `git cat-file`.
- **Bytecode cache:** the Pydantic check loaded `switch_patches_20261123/parameter_registry.py.patched` with a `SourceFileLoader` from inside this folder, at 19:12:59. That wrote `switch_patches_20261123/__pycache__/parameter_registry.py.cpython-310.pyc`, which tripped another agent's pin check. **That was my error: I imported code from inside a worktree that other agents pin.**
- **Removal:** I removed that file and its folder by exact path at 19:51. The folder's `SHA256SUMS` re-verifies OK, and no `__pycache__` remains in this worktree outside `data/`.
- **Left in place:** 408 `__pycache__` folders under this worktree's `data/`, from my mutation and fixture runs (the mutated copies there were imported). They are mine and can be removed on request.
- **From now on:** any import check runs on a copy outside the pinned worktree, with `PYTHONDONTWRITEBYTECODE=1`, under `taskset -c 4,5 nice -n 15`.

**Reboot at 19:25:**
- My task had finished at 19:17:59, and nothing of mine was in flight, so nothing was re-run.
- Every deliverable re-verified against the hashes recorded above. So did the run H counted-grade and follow-up declarations.
- My survival check used a `pgrep -f` pattern that matched its own shell. It only listed processes and stopped nothing, but it breaks box rule 8; I will use recorded PIDs from now on.

## Owner questions (one line each, real choices only)

1. Land the registry patch as drafted? It changes no default, only what the registry knows.
2. Fix or drop the 20 stale specs (B1) first?
3. Should the learning-loop drivers (walk-forward, e2e vote and grade) get a WATCHER manifest or dispatch route (G6), so their switches become reachable at all?
4. Does the Chapter 14 S1 boundary on the six Step-5 LLM fields still stand after your 10:40 statement that Beta holds no longer bind? The draft patch leaves them out.
5. Should an applier (B12 / LL-16) be built behind `apply_mode human_approval`, which would lift S3, or stay held?
