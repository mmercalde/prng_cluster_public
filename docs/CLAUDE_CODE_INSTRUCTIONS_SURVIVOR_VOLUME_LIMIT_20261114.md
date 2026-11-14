# CLAUDE CODE INSTRUCTIONS — SURVIVOR-VOLUME LIMIT (plan item A2, finished condition F3)

## Prior record

**Searches run (public mirror, HEAD `b2d2822`, 2026-11-14):**

- `grep -lIiE "volume[- ]limit|volume cap|survivor[- ]volume|pre-run sample|volume-aware|projected survivors" docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_* docs/TB_RULING_REQUEST_* docs/PROPOSAL_* docs/CHAPTER_* docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md docs/exchange/*.md docs/exchange/*/*.md`
- `grep -rnIiE "survivor_volume|volume_gate|volume_cap|projected_survivors" --include=*.py --include=*.sh --include=*.json .` outside `docs/` (no hit: no such code exists).
- `ls docs | grep -iE "capacity|volume|envelope|RUN4"`.
- `grep -rnI "2²⁶ sample\|2²⁴ sample\|177 hits\|3 hits" docs/*.md`.
- Read: `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md` §7.3;
  `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md` §2, §3.A, review rule 3 (`:322`);
  `docs/S172_PHASE3_SURVIVOR_CAPACITY_CHARACTERIZATION.md` header, §3, §5, §7;
  `docs/S172_RUN4_PROPOSAL_PINNED_GEOMETRY.md` §1.1, §1.2;
  `docs/exchange/to_beta/X-0000_R2_ALPHA_REPLY.md` Q4;
  `docs/exchange/from_beta/X-RUN4-RANGE_R1_BETA_REVIEW.md` Q1;
  `docs/CLAUDE_CODE_INSTRUCTIONS_DELEGATED_RUN_GATES_20261111.md` §2–§4;
  `scripts/run_authorization_gate.py` header.

**Documents that already decide this subject:**

| Document | What it settles | Label |
|---|---|---|
| `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md` §2 F3, §3.A A2, G4, `:322` | A survivor-volume limit in the launch path: a pre-run sample predicts volume, the launch is refused above a cap. Reviewed before commit | ALREADY RECORDED. This brief builds it |
| `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md` §7.3 | The method: a sample of the seed range through the production kernel on the run's own residue window, projected per phase; REFUSE above the cap. Starting cap 23,515 in one phase. Raised only by the owner, on evidence from completed runs | ALREADY RECORDED as a proposal. Its review question on the cap (§13 Q3) was never sent to Beta |
| `docs/S172_PHASE3_SURVIVOR_CAPACITY_CHARACTERIZATION.md` §3, §5 | The method was run by hand on 2026-08-27: one local RTX 3080 Ti on VM101, `SieveExecutor.execute`, a stub resolver, 5–10 s per geometry. It reproduced six historical counts. Validated envelope: 23,515 in one phase (attempt 9, phase 4) | ALREADY RECORDED. This brief turns that hand method into a gate |
| `docs/S172_RUN4_PROPOSAL_PINNED_GEOMETRY.md` §1.1 | Run 4 predicted by hand: about 96 in phase 3 (2²⁶ sample, 3 hits) and about 22,656 in phase 4 (2²⁴ sample, 177 hits), on attempt 9's residue window, for `[0, 2³¹)` | ALREADY RECORDED |
| `docs/exchange/to_beta/X-0000_R2_ALPHA_REPLY.md` Q4 | Alpha's position: the limit is a hard precondition for the first run with sampled settings and for any agent-launched run. It is not a precondition for the pinned Run 4. Beta did not reassess Q4 in R2 | ALREADY RECORDED. Not changed here |
| `docs/exchange/from_beta/X-RUN4-RANGE_R1_BETA_REVIEW.md` Q1 | Run 4 now takes `[2³¹, 2³²)`. The hand prediction was for the other half. Apply the volume requirement to the actual range and residue window. No upper-half measurement exists | ALREADY RECORDED. M-B below supplies that measurement |
| `docs/LEADS.md` L-2 · `docs/BACKLOG.md` §20 · `docs/TODO_CONSOLIDATED.md:100` | The coordinator's inbound queue has no volume-aware bound | ALREADY RECORDED. A separate item. Not built here |

**Items this brief raises:**

| Item | Label |
|---|---|
| The cap equals the count the pinned geometry already produced (23,515), and Run 4 is predicted at about 22,656. A sample's noise is larger than that margin. The gate could refuse Run 4 on noise | NEW. Measured in M-B and reported. The cap's number and the comparison rule stay the owner's (§2, the cap file) |
| The cap lives in an owner-committed file, `config_manifests/survivor_volume.cap`, under the same ownership rule as the run authorization files | NEW. The `.cap` extension is deliberate: `.gitignore:41` ignores `*.json` |
| The gate is a standalone script. Wiring it into a launcher is G4 and is not in this brief | NEW scoping. `gate12_launch.sh` is not edited |
| **Owner, 2026-11-14:** a sample that finds too many survivors points at the settings, so feed the estimate back to the search instead of only refusing the run. Before each trial, estimate the volume for the settings Optuna picked; skip a trial over the cap and record why, so the search steers away | NEW. **Not built here.** It changes the Step 1 trial loop and is its own brief after Run 4, BEFORE-COMMIT class, due before the 50-trial soak. This brief only makes the estimator callable for one trial's settings (§2) |
| The measured region behind that idea: `docs/S172_PHASE3_SURVIVOR_CAPACITY_CHARACTERIZATION.md` §8. Volume is set by window size and threshold together; window 6 is over the cap at every reachable threshold; 35 of 64 measured forward cells are over it | ALREADY RECORDED |

**Not searched:** VM101's live state, ser8 archives, the private repository, the project
PDFs. Line anchors were read at `b2d2822`. **Re-read each one before relying on it.**

---

**From:** Team Alpha · **Owner:** Michael · **Run on:** VM101, as `michael`, venv
`source ~/venvs/torch/bin/activate` · **Model:** Opus
**Base commit:** the commit that lands this brief (its parent is `b2d2822`). Record
`git rev-parse HEAD` in your first-step output.
**Review class:** BEFORE-COMMIT (completion plan `:322`).
**Report:** `docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md`, in your worktree.

**Why.** Run 3 chose its settings by sampling and produced 13,146,485 survivors in one
phase; nobody had predicted it. The largest volume ever carried through a complete run is
23,515. The limit predicts a run's volume before it launches and refuses a run above the cap.

**The yardstick (CLAUDE.md).** TFM is judged by prediction-pool hit rate against `k/1000` on
unseen draws. A survivor count is a load figure here. It says nothing about quality.

## 0. Rules for this session

- **No commit, push, branch or tag.** Michael lands everything.
- **Worktree:** `git worktree add --detach /home/michael/tfm_worktrees/volume_limit HEAD`.
  Write only inside it. The one exception is the review request file, §4.
- **No fleet. No SSH to any rig. No coordinator, no WATCHER, no pipeline step.**
- **One local GPU on VM101, for the sample only.** Check `nvidia-smi` first. Use the GPU
  only if it is present and has no compute process on it. If it is absent or busy, M-B is
  `UNAVAILABLE`: say so, build the gate against the injected sampler, and continue.
- **No writes to live state:** the live database, the coverage ledger, the miner ledger,
  the staging directory, the halt file. The sample uses a stub resolver and writes its
  output under the worktree.
- **Not edited:** everything under `miner/`, `prng_registry.py`, the kernels,
  `scripts/run_authorization_gate.py`, `scripts/env_freeze_gate.py`, `gate12_launch.sh`,
  `agents/watcher_agent.py`, `scripts/owner_land.py`, `scripts/beta_relay.py`.
- **Long commands** run under `nohup`, never tmux.
- **Another session is working in parallel** in `/home/michael/tfm_worktrees/step1_search`
  on different files. Do not read from or write to that directory.
- **First step:** `pgrep -af beta_relay.py`; `git rev-parse HEAD`; `git status --porcelain`
  in the main tree; create the worktree; activate the venv; `nvidia-smi`. Record the output.

## 1. M-B — measure first

Reproduce the hand method of 2026-08-27 (capacity record §3), then apply it to Run 4 as it
will launch.

1. **Control.** Attempt 9's geometry on `[0, 2³¹)`: reproduce the recorded sample (phase 3,
   2²⁶ seeds, 3 hits; phase 4, 2²⁴ seeds, 177 hits) on a residue window whose
   `sha256_residues` equals `trial_context.residue_sha256 = c761272da958ba34…`. If the
   control does not reproduce, stop and report.
2. **Run 4, lower half,** `[0, 2³¹)`, with Run 4's configuration as the built command gives
   it (`docs/SESSION_CHANGELOG_20260828_RUN4_ROUTING_PATCH.md:212-222`; window 12, anchor 25,
   `generator_phase 0`, sessions `["midday"]`, thresholds 0.71 / 0.47, skip 6–99). Report
   whether its residue window and its counts equal the control's. The ruling says the two
   populations are not regression-equivalent
   (`docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:154-179`).
3. **Run 4, upper half,** `[2³¹, 2³²)`: the range Run 4 will take. All four phases. Sample
   at least 2²⁶ seeds per phase, and 2²⁸ for phase 4.

For each: dataset file and sha256, residue digest, the sample's seed intervals, hits per
phase, the projection, a 95% interval on it, seconds, GPU, commit. Count what a run counts:
survivors after the host filter (capacity record §2d). Give the kernel-admitted count
beside it. Label every figure MEASURED or COMPUTED.

The sample goes through the production path: the same payload builder
(`miner/range_miner_coordinator.py:9216`), kernel selection and `SieveExecutor.execute`
(`miner/range_miner_worker.py:1112`) a real stripe uses. Re-implement none of it.

## 2. The gate

`scripts/survivor_volume_gate.py`. It predicts and decides. It launches nothing.

**Input.** `--params` (a JSON object or `@file`), in the convention of
`scripts/run_authorization_gate.py`: `seed_start`, `seed_end_exclusive`, and the run's
geometry as exact values (PRNG family, window, anchor, generator phase, sessions,
thresholds, skip range, dataset file). `--record-out PATH` for the evidence record.

**The cap file.** `config_manifests/survivor_volume.cap`, JSON text:

```
{"cap_survivors_per_phase": 23515,
 "compare": "point",
 "min_sample_seeds": 67108864,
 "evidence": "attempt 9 phase 4, run distributed_config_t1_554463d3"}
```

`compare` is `point` (the projection) or `upper95` (the upper end of its 95% interval).
Create the file in the worktree with these values. It becomes the owner's when he lands it.

**Verdicts.** Vocabulary and exit codes as the other gates: 0 = PROCEED, 1 = REFUSE.

| Situation | Verdict |
|---|---|
| Every phase's compared figure is at or below the cap | PROCEED |
| Any phase's compared figure is above the cap | REFUSE, naming the phase, the figure, the interval and the cap |
| The cap file is absent, unparseable or missing a field | REFUSE |
| The cap file is not tracked by git, or differs from `HEAD`; `git` cannot run | REFUSE |
| A geometry parameter is missing, or is a range instead of an exact value | REFUSE. Sampled settings are judged one trial at a time, not as a range |
| The sample is smaller than `min_sample_seeds`, or any of it lies outside the requested range | REFUSE |
| The residue window the gate sampled on differs from the one the payload builder gives for these parameters | REFUSE |
| No GPU, or the GPU has a compute process on it | UNAVAILABLE, which is REFUSE |
| The sampler raises, times out or returns a malformed result | ERROR, which is REFUSE |

**The sample.** Drawn from inside the requested range, in several blocks spread across it,
chosen by a fixed rule from the parameters so that the same request gives the same sample.
Projection = hits × (range size ÷ sample size), per phase. Zero hits gives a projection of
0 and an upper bound that is not 0; report both.

**The evidence record** (JSON at `--record-out`): the parameters; dataset sha256; residue
digest; the sample's intervals; hits, projection and interval per phase; the cap file's
sha256 and contents; the verdict and its reason; kernel identity; GPU; repository commit;
UTC time.

**The sampler is injectable,** so the tests run on CPU with no GPU.

**The estimator is a function, and the gate is one caller of it.** Put the estimate in an
importable function that takes one exact geometry and a seed range and returns hits,
projection and interval per phase, with no command-line parsing, no cap and no verdict
inside it. The gate script calls it and applies the cap. A later brief will call the same
function from the Step 1 trial loop, once per trial, for the settings Optuna picked. Do not
write that caller, and do not edit `window_optimizer*.py`. Say in the report what the
function costs per call (seconds, GPU memory) and whether it can run on a fleet GPU as well
as the local one.

**A survivor count is load, not quality.** Nothing the estimator or the gate writes may be
read as a score. The evidence record names its verdict as a load verdict.

## 3. Tests

`tests/test_survivor_volume_gate.py`, CPU only, injected sampler, throwaway git repository
for the cap-file ownership cases (as `tests/test_run_authorization_gate.py` does it).

| Case | Expect |
|---|---|
| V-A | All phases under the cap: PROCEED, exit 0, record written |
| V-B | One phase exactly at the cap: PROCEED. One above by 1: REFUSE naming that phase |
| V-C | `compare: upper95`, projection under the cap and upper bound over it: REFUSE. The same with `compare: point`: PROCEED |
| V-D | Zero hits in every phase: PROCEED, upper bounds reported and not 0 |
| V-E | Cap file absent · unparseable · field missing · untracked · modified in the worktree: REFUSE each |
| V-F | A geometry parameter missing · given as a range: REFUSE each |
| V-G | Sample below the minimum · a sample block outside the range: REFUSE each |
| V-H | Residue digest mismatch: REFUSE |
| V-I | No GPU · GPU busy: UNAVAILABLE, exit 1 |
| V-J | Sampler raises · times out · returns a malformed result: ERROR, exit 1 |
| V-K | The same request twice: the same sample intervals |
| V-L | Projection arithmetic at `[0, 2³¹)`, `[2³¹, 2³²)` and a range that is not a power of two |
| V-M | The estimator called directly as a function, for one exact geometry, with the injected sampler: same figures as the gate reports for the same request; no cap file read; no verdict returned |
| V-MUT-1 | UNAVAILABLE mapped to PROCEED: V-I red |
| V-MUT-2 | `>` changed to `>=` at the cap: V-B red |
| V-MUT-3 | The projection scaled by the wrong range: V-L red |
| V-MUT-4 | The cap-file ownership check removed: V-E red |

Show each mutant red in the report. If M-B ran, add one live case: the gate on Run 4's
upper-half request with the real sampler, and report the verdict under `point` and under
`upper95`.

## 4. Review by Beta, carried by this session

When the code and tests are done and you have stopped editing:

1. Write the report (§5) in the worktree.
2. `pgrep -af beta_relay.py`. If a relay is running, wait 60 seconds and check again.
3. Write the request file in the **main tree**, the only write you make outside the
   worktree: `/home/michael/distributed_prng_analysis/docs/exchange/to_beta/X-VOLUME-LIMIT_R1_ALPHA_REVIEW_REQUEST_survivor_volume_gate.md`.
   Layout as `AGENTS.md` §5: it opens with `## Prior record`; review class BEFORE-COMMIT;
   base commit; every file under review by absolute worktree path, marked new or modified;
   a line saying the files are in a worktree and are not committed; then the questions
   below, unchanged, each with the allowed answers.
4. Start the relay from the main tree at once, as `docs/exchange/RELAY_PROTOCOL.md` §2
   says. Write nothing in the main tree while it runs.
5. Act on Beta's positions by the protocol's table. At most three rounds. You never
   overrule Beta and never argue it into agreement. You stop when you disagree with a LIVE
   objection, when a fix is outside §2, or when Beta cannot read the worktree.

**Questions for Beta (use as written):**

1. Is the sample the production path (payload builder, kernel selection, executor, residue
   window, host filter), with nothing re-implemented?
2. Is the sample drawn from the requested range, and is the projection arithmetic right?
3. The decision rule: `point` against `upper95`, the minimum sample, zero hits. Is a
   refusal or a pass ever decided by noise in a way the record does not show?
4. Does every failure end in REFUSE, never PROCEED?
5. The cap file: is the ownership rule the same as the run authorization files', and
   sound?
6. Can running the sample itself harm anything (the GPU on VM101, live state)?
7. The cap of 23,515 (`docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md` §7.3; its §13 Q3 was
   never sent): what evidence supports it, and what would support a different number?
8. The estimator as a function: can a per-trial caller use it later without a rewrite, and
   does anything in it treat a survivor count as a score?
9. Any governing record this work missed.

## 5. Report

`docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md`, opening with `## Prior record`. Then:
first-step output; M-B with its three tables; what the gate does, with `file:line`; the
test table with counts and the mutants shown red; the live case if it ran; Beta's last
position per question and what was done; anything unresolved with both positions; what you
did not verify.

Put this at the top of the report, in plain words, for the owner: Run 4's projected
survivors per phase on `[2³¹, 2³²)`, the interval, and whether the gate passes it under
each comparison rule.

End with the landing list: every new file with its sha256.

## 6. Stop gates

Stop and report, without working around it, when: the control in M-B does not reproduce;
the production path cannot be driven without editing a file listed as not edited; a step
would write to live state; Beta holds a LIVE objection you disagree with or whose fix is
outside §2; the third round ends; Beta cannot read the worktree.
