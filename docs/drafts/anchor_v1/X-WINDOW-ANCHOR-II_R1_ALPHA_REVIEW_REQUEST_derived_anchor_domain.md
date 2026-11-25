# X-WINDOW-ANCHOR-II · R1 · ALPHA REVIEW REQUEST · flag-selectable derived window-anchor domain (DRAFT, NOT SENT)

**DRAFT.** Not in `docs/exchange/to_beta/`, not relayed. The owner decides whether and when it goes (one round).
On sending, copy unchanged to `docs/exchange/to_beta/` and run
`python3 scripts/beta_relay.py --request docs/exchange/to_beta/X-WINDOW-ANCHOR-II_R1_ALPHA_REVIEW_REQUEST_derived_anchor_domain.md`.

## Prior record

Searches and documents: `docs/drafts/anchor_v1/STATUS_AND_PLAN_ANCHOR.md` "Prior record" (same session, same
tree). The decisive entries:

| Document | What it settles | Label for this request |
|---|---|---|
| `docs/PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md` §4.2, §4.3, §7 AC3; `docs/TB_RULING_WINDOW_ANCHOR_V1_1_DESIGN_GATE_CLOSED.md` | derived domain `{min 0, max_cap null}`, no clamp, phase 0, AC3 | ALREADY RECORDED; implemented |
| `docs/TB_RULING_WINDOW_ANCHOR_BRIEF_I_CODE_REVIEW.md` (M8 section) | M8 must mutate the real control-era ceiling 100 → 149 in Brief II | ALREADY RECORDED; M8 now has a production site and is DETECTED |
| `docs/TB_RULING_BRIEF_I_PRODUCTION_SHAPE_FAILURE.md:6-7` | Brief I NOT ACCEPTED; Brief II BLOCKED | **CHANGES A RULING** in sequence. Owner direction 2026-11-23 10:32 asks for this work; the owner's explicit word on the block is Q1 of the status document |
| v1.1 §4.3 "`search_bounds.offset` is REMOVED … outright" | old key removal | **CHANGES A RULING**: kept, because the task requires the bounded behaviour to stay default until the owner rules |
| `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md`; WATCHER Route A (`agents/watcher_agent.py:511-519`, `:690-711`) | operator-only seven-key pin | ALREADY RECORDED; unchanged |
| Skip-delivery switch `f916532`/`00e2ee7` (X-MINER-SKIP-SWITCH, -2) | pattern for a default-off Step-1 setting | ALREADY RECORDED; copied |

Not searched: host outside the worktree, ser8, private repo, project files, rigs.

```
id / round      X-WINDOW-ANCHOR-II / R1
type            REVIEW_REQUEST
from / to       Team Alpha (Claude Code)  ->  Team Beta (Codex)
date            2026-11-23 (draft)
review class    before-commit: the Step 1 launch path for runs that count (window_optimizer.py,
                window_optimizer_bayesian.py, window_optimizer_integration_final.py,
                utils/step1_search_ledger.py) and the hand-landed manifest
                agent_manifests/window_optimizer.json. No miner/ file, no kernel, no worker.
base commit     41ad509 (main). Worktree /home/michael/tfm_worktrees/anchor-brief2.
```

**Files under review:** `docs/drafts/anchor_v1/code.diff` (5 tracked files, +217/−12, sha256
`2abdfa0db37cea005fc40f02cad0aa30014a82fef9522bf55776d398000e031b`), new `utils/window_anchor_domain.py`
(`a9fc243d04b21813549fc5982219f3438f2197f95cfb6f0f532d5024ffc58e01`), new `tests/test_window_anchor_brief2.py`
(`db06efd7ad9b42ac20b8c1a3431ea2789e5ac16911dc0cd711170410ebd5c256`). Hashes as of the round-2 refresh; re-take them at send.
**Owner ruling 2026-11-23 11:25:** Brief II goes ahead before Brief I acceptance (lifts
`TB_RULING_BRIEF_I_PRODUCTION_SHAPE_FAILURE.md:6-7`); the CHANGES-A-RULING row above is ruled.

## What changes

A second setting, `anchor_holdout_guard` (bool, default false, refused unless the derived domain is on), caps the
anchor so the window ends before the first record of Step 1's 80/20 holdout for that session filter
(`READY_ANCHOR_BRIEF2.md` point 4). The first setting, `derived_anchor_domain` (bool, default false), is routed manifest `default_params` + `args_map` →
`--derived-anchor-domain` → `run_bayesian_optimization(*, derived_anchor_domain)` → `optimize_window(derived_anchor_domain)`.

- **Off:** WATCHER omits a false bool from argv; the CLI forwards nothing (conditional `**{}`); no strategy
  attribute is set; the Optuna objective runs its original two `suggest_int` calls; no search key is added. A
  study or search tagged by a derived run refuses a bounded run; untagged ones (all that exist) are unaffected.
- **On:** `resolve_anchor_domain` reads the dataset once, checks its sha256 against the run's frozen dataset,
  counts `N_filtered` per session option, and proves each count against `miner.range_miner_worker.load_residue_window`
  (anchor `N-1` loads, `N` is refused). The objective draws `window_size`, then `session_idx`, then `offset`
  (legacy name = window anchor) in `[min, effective_max(window_size, sessions)]`, and passes every choice,
  sampled or enqueued by the operator pin, through `check_window_choice` before `WindowConfig` exists. Refusals
  name anchor, domain, sessions and dataset sha256. Refused with `--config-file`, non-RANGE-MINER routes,
  `--strategy` ≠ bayesian, `n_parallel` ≠ 1, `--resume-checkpoint`.
- **Not changed:** `miner/`, kernels, `generator_phase` (stays the v1 pin 0 at `window_optimizer_integration_final.py:2180`),
  NPZ arrays, canonical records, the Step 3 law, `WindowConfig.offset`'s name, `distributed_config.json`,
  `parameter_registry.py`, `agents/watcher_agent.py`.

## Evidence (CPU only)

- `tests/test_window_anchor_brief2.py`: **18/18 gates, 12/12 mutants DETECTED** (applied, executed, credited
  assertion reached), 71.6 s wall, peak RSS 224.5 MB. Off-path identity is proved against HEAD's own code loaded from
  `git show HEAD:<file>`: a seeded TPE study gives identical trial parameters, and `presented_configuration` is
  identical with the key absent or false, pinned and unpinned.
- Differential battery, 17 suites that import the changed modules, live vs a `git archive HEAD` copy, run
  sequentially: no suite changes verdict. Pre-existing reds are identical on both sides (stub DB in the Run 4
  routing suite, missing pointer manifest in a fresh worktree, the RXP-1 flake, Gate 22 on uncommitted `.py`).
  Table: `docs/drafts/anchor_v1/READY_ANCHOR_BRIEF2.md` §Tests.

## Questions (real-world cases on this cluster only)

Allowed answers: `NO OBJECTION` · `OBJECTION` (with `LIVE` or `THEORETICAL`) · `SUGGESTION` · `NEEDS EVIDENCE`.

**Q1. A Run-4-style pinned acceptance run with the setting off.** The owner relaunches the committed Route-A Run 4
command (anchor 25, window 12, midday, `window_trials=1`, `n_parallel=1`). Does anything in this diff change its
argv, its Step 1 search id/digest, its study, its trial parameters or its published artifact? (We claim no:
false bool omitted at `agents/watcher_agent.py` bool branch; G-DEFAULT-SEARCH covers the pinned case.)

**Q2. A derived run that pins anchor 7,000 midday, window 12.** Today the same pin runs with only an Optuna
`UserWarning` (status doc §C). With the setting on it is validated against `[0, 8503]` and runs. Is there any
consumer between `WindowConfig` and the Phase-5 publication that still assumes `offset ≤ 100` (e.g. a cache key,
a result filename, a TRSE `confirmed_windows` entry, the canonical array 4 int32 range)? We found none by reading
(`utils/canonical_arrays.py:206` is int32).

**Q3. The arm separation.** A derived study is tagged `window_anchor_domain=derived` at its first trial; the
search space gains `window_anchor_domain` and `window_anchor_bounds`. A bounded WATCHER run that names a derived
search or study is refused before the sieve. Is the separation sufficient for the D6.2 checkpoint path, given the
setting refuses `--resume-checkpoint` rather than tagging the run context?

**Q4. The count proof.** `N_filtered` is counted with the same predicate as the worker and then proved against
the worker's own function at two anchors. Is that an acceptable reading of "one session-filter authority", or do
you require the count to come out of `miner/range_miner_worker.py` (a governed file, so a parity redeploy)?

**Q5. Holdout reach.** With the derived domain on and the guard off, a midday trial may place its window at filtered
index ≥ 6,707, inside `holdout_history.json`. With the guard on, the maximum is `N_train − w` (midday 6,695 at W12).
Is the guard's `N_train` (same session predicate over `data[:step1_holdout_split_point(len)]`) the right reading of
"before the first holdout record" for each session filter, given the split is over the combined history?

**Q6. TPE with a moving `offset` range.** Measured (`READY_ANCHOR_BRIEF2.md` point 3): with the switch on,
`offset` leaves the multivariate intersection search space and is sampled by independent TPE after startup (21/24
trials), on Optuna 4.4.0 with `multivariate=True`. Does anything in the S172 study-continuity or resume rules
assume every Step 1 parameter is in the joint space?
