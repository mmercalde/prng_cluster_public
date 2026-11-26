# NOTE · Can Step 1 (RANGE-MINER) run on VM101's 3080 Ti alone? (2026-11-24, read only)

Task: `logs/e2e_session/TASK_SINGLE_CARD_SIEVE_CHECK_20261124.txt`. Nothing was launched: no sieve, no card, no rig, no `/dev/shm`. CPU work was reading on cores 6,7 at nice 15.

## Plain summary

**Yes for the engine, no for the launch harness.**

The engine has run on this card alone: the D6 "Zeus single-GPU" smoke, 2026-07-29, commit `b08c2c5`, made a certified generation on the 3080 Ti.

Step 1 has a declared setting for a one-node fleet, `--execution-set-nodes localhost`. It clamps admission to 1 worker, so it needs no new code on a **direct `window_optimizer.py` run**.

Two routes cannot use it today:
- **WATCHER**: the setting is not in the manifest. Recorded owner item, `TODO_CONSOLIDATED.md:213`.
- **`gate12_launch.sh`**, run H's harness: its GPU, parity, launch, sentinel and liveness steps are all about the three rigs, and none of them has a setting.

Three things block it **today** whatever the route:
- the card is in use by the walk-forward;
- run H's settings break the 50,000 survivor cap even at 2²⁴ seeds, so each window needs the owner's one-off;
- the macro view's anchor domain and screen are unlanded code.

Estimated (not measured) one-card cost per window at run H's settings:
- 2²⁸ seeds: about 200-240 s, **about 400 a day**;
- 2²⁶: about 65 s, about 1,300 a day;
- 2²⁴: about 30 s, about 2,900 a day.

One 2-4 minute card measurement would settle these. PWC is **not** a simpler route.

## Prior record

```
P='single.card\|single-card\|one-card\|one card\|single worker\|one worker\|1-worker\|one-worker\|localhost worker\|VM101 card\|101 card\|3080 Ti.*miner\|miner.*3080'
grep -il "$P" docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_* docs/TB_RULING_REQUEST_* docs/PROPOSAL_* docs/CHAPTER_* \
     docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md docs/exchange/*/* \
     docs/CLAUDE_CODE_REPORT_* docs/SESSION_CHANGELOG_* docs/PHASE6_PREREQS.md     (main tree, 41ad509)
grep -rln -i "zeus-only\|single-GPU smoke\|D6 smoke" docs;  grep -rn -- "--execution-set-nodes" docs tests
```

| Found | Says | Label |
|---|---|---|
| `docs/D6_RELEASE_GRADE_CERTIFICATION_RECORD.md`; log `docs/D6_RELEASE_GRADE_SMOKE_20260729.log`; `tests/smoke_s172_phase5_d6_zeus_single_gpu.py:1-60` | One 3080 Ti, the real miner worker, the production `run_bidirectional_test(use_range_miner)`; 8 M seeds, window 3, certified generation. The harness drives one trial and binds to loopback | **ALREADY RECORDED** (engine on one card) |
| `docs/PHASE6_PREREQS.md:60-64`, `:134-147`, `:266-273` | Zeus-only D6 smoke path; `localhost.gpu_count` 2 → 1 recommended | **ALREADY RECORDED**; now 1 (`distributed_config.json:6`) |
| `docs/TODO_CONSOLIDATED.md:213`; `docs/CHAPTER_1_WINDOW_OPTIMIZER.md:1437`, `:1624`, `:2271` | `--execution-set-nodes` / `--rig-profile` are not in the manifest `args_map`, so a WATCHER run takes the full fleet; owner (manifest) | **ALREADY RECORDED** (status) |
| Run H `logs/stage4/volume_gate.log` (`~/skipsw_run_thr3030`) | 3080 Ti kernel times at run H's settings; phase 3 REFUSES the cap | **ALREADY RECORDED**; used in §2 |
| OWNER 13:02, 13:03, 11:11 (task file) | macro view first; "not spread too thin"; work around the rigs | the request |

Not searched: the host beyond the files named, ser8, the private repository, project files outside the repository. I found no log of a full `window_optimizer.py main()` run with a localhost-only set (searched `logs/*.log`, `logs/*/*.log` for `EXEC-SET`) [UNVERIFIED that one never ran].

## 1. Gates a one-card run meets (main tree, 41ad509)

| Gate | Refuses with the rigs absent? | Existing setting |
|---|---|---|
| Execution set: `window_optimizer.py:1625-1640` (flag), `:1821-1890` (resolve + freeze); `execution_set.py` `resolve_execution_set` | Full fleet by default. `localhost` set: resolves; the local name must equal `socket.gethostname()` | **`--execution-set-nodes localhost`**. Admission = min(request, 1 identity) |
| Miner admission: `miner/range_miner_coordinator.py:560-584`, `:9630` (`expected_workers` from the set); `:9473-9490` (bounded admission, default 180 s) | No, once one local worker connects | same set |
| Dataset authority P0.5: `window_optimizer.py:1946-1960` | No. It verifies only the set's nodes; `remote_execution` is derived False | — |
| `require_fleet`: `window_optimizer.py:1954` | PWC only; not the miner | — |
| Step 1 search id: `window_optimizer_integration_final.py:1676-1677` | No on a direct run (no search credit or ledger). WATCHER requires it (S6) | — |
| Stripe planning: `CoordinatorConfig` (`seed_cap_nvidia` 5 M), `:4053-4079`, `:1040` | No. 16 stripes run in turn on one worker, as the D6 smoke did (8 sub-stripes on one card) | — |
| **WATCHER**: `agents/watcher_agent.py:1769-1772` (step-scoped filter), `:1821` | **Refuses** in effect: `execution_set_nodes` is not declared, so it is dropped and the full fleet resolves | **None.** Smallest: manifest `default_params` `"execution_set_nodes": ""` + `args_map` entry (default `""` keeps argv unchanged). Hand-landed, owner; already TODO `:213` |
| **gate12 GPU gate**: `scripts/gate12_gpu_gate.py:24-30` | **Refuses**: all three rigs must report a full count | **None** (by design: a rig saturation claim) |
| **gate12 parity**: `scripts/gate12_parity_gate.py:366-393` | **Refuses**: probes every remote node of the full set; no node option | **None** |
| **Fleet launch**: `scripts/launch_fleet_manual.sh:246-273` (every node), `:503-507` (truncated → exit 1); `gate12_launch.sh:204`, `:324`, `:352`, `:399`, `:434`, `:509` (rig CT addresses written in) | **Refuses** | **None** |
| Survivor-volume gate: `scripts/survivor_volume_gate.py` (operator block b09, not inside Step 1) | Run H's settings **refuse** at any seed count tried (§3) | Owner one-off (run H 14:30) |
| VM101 card rule: GPU only with `sunshine` (memory); lead rule, no GPU job while VM100 is gaming | **Today: busy** (walk-forward) | — |
| Clean tree for a certified generation (`finalize_run` refuses a dirty tree; D6 record §1) | Brief II and the screen are unlanded, so main can sieve only anchors 0-100 with the count objective | — |

**Proposal only, nothing built.** The smallest change for a harnessed one-card run would be a default-off `EXECUTION_SET_NODES` in `launch_fleet_manual.sh` and `--execution-set-nodes` in `gate12_parity_gate.py`. Both would resolve through the same `resolve_execution_set`. A localhost-only set has no remote parity targets, because the local tree is the reference (`gate12_parity_gate.py:371-372`), so the gate would have to record "no remote nodes". The gate12 GPU gate stays as it is: it certifies a rig run, and a one-card run is not one.

The route that needs no change is a direct `window_optimizer.py --use-range-miner --execution-set-nodes localhost …`, with the local worker started by the launcher's own local command (`launch_fleet_manual.sh:344-351`). This is the D6 smoke's shape, run through `main()` [UNVERIFIED end to end].

## 2. Cost per window on the one card (run H's settings: window 12, skip 0-16, thresholds 0.30, both modes)

Measured on this card, from run H's volume gate (production sieve path, sampled):

| Phase | Seeds | Time |
|---|---|---|
| 1 | 2²⁶ | 3.2 s |
| 2 | 2²⁶ | 3.7 s |
| 3 | 2²⁶ | 9.9 s |
| 4 | 2²⁸ | 14.8 s |

Measured on the fleet (run H, 25 workers): trial 130.8 s; per-phase stripe max/sum 5.7/22.6, 3.5/14.1, 46.4/180.3, 16.6/61.5 s. Only 4 of the 25 workers did any work: 2²⁸ seeds make 4 stripes per phase. So the fleet gave run H little over one card.

The D6 smoke (one card, 8 M seeds, window 3) took 19.9 s from trial creation to finalisation (smoke ledger). A forward sub-stripe of 1 M seeds with about 50 k survivors took about 1.7 s, against about 0.1 s with about 50 survivors. So writing survivors costs about 30 µs each.

| Seeds | Kernel (linear in seeds) | Survivor writing (phase 3 + 4) | Fixed per trial | **Per window** | **Per day** |
|---|---|---|---|---|---|
| 2²⁸ | 82 s | about 95 s (2.37 M + 0.60 M) | 20-60 s | **about 200-240 s** | **about 360-430** |
| 2²⁶ | 20.5 s (phases 1-3 measured) | about 24 s | about 20 s | **about 65 s** | **about 1,300** |
| 2²⁴ | about 5 s | about 6 s | about 20 s | **about 30 s** | **about 2,900** |

Every per-window figure is **[UNVERIFIED]**: linear scaling, the survivor-write rate taken from a window-3 smoke, and fixed cost between the smoke's and run H's 58.6 s non-sieve share (130.8 − 72.2 s). The screen adds about 0.3 s (hits). The 59 s publication is paid once per run.

**One measurement settles it:** one direct one-card trial at run H's settings, at 2²⁴ and then 2²⁶, about 2-4 minutes of card time, timed from the miner ledger's `trials.created_at` / `finalized_at` and the stripes table. It needs the card free and the volume one-off.

## 3. What a smaller seed range does (prediction)

- **Survivor count falls in proportion to the seeds.** Phase 3 measured 592,050 in 2²⁶ against 2.37 M projected for 2²⁸.
  - Run H's 12,156 final survivors at 2²⁸ predict about 3,000 at 2²⁶ and about 760 at 2²⁴.
  - Phase 3 at 2²⁴ is about 148 k, still **above the 50,000 cap**: the volume gate refuses run H's settings at every size in the table.
- **Screen noise under the null does not grow.** `screen_hits` uses the top 20 voters and the exact null over 600 draws, whatever the survivor count. `screen_rank` with about 760 voters leaves at least 240 values tied last on each draw; the exact null accounts for the ties, but the resolution falls.
- **The signal can be lost.** A smaller range is one slice of seed space, not a random sample. If a window's effect sits in particular seeds, a 2²⁴ slice sees it with probability about 1/16 (one sixteenth of 2²⁸). So more good windows score like bad ones (false negatives); a good score is still a good score. If the effect is spread over many seeds, the expected score holds but varies more from slice to slice.
- **Test:** screen the same 5-10 windows on two disjoint 2²⁴ slices and on 2²⁶, and compare how well the scores agree (rank correlation).

## 4. PWC as a one-card route

**No.**
- PWC can address the local card. It uses the same execution-set node list (`persistent_worker_coordinator.py:987-995`), local GPU slots (`:963-965`), and `--min-workers` (default 24, `window_optimizer.py:1597`) set to 1.
- But the derived anchor domain and the screen refuse every route except RANGE-MINER (Brief II, `window_optimizer.py` `--derived-anchor-domain` refusal list). PWC's own anchor check limits the SSH route to anchor 0 (`persistent_worker_coordinator.py:434-445`).
- PWC is a non-certifying diagnostic engine (Beta retirement, 2026-07-31).
- It needs the same launch work and gives less.

## Questions (owner / lead)

- **Q1.** May a first one-card trial run as a direct `window_optimizer.py --execution-set-nodes localhost` launch (not WATCHER, not gate12), labelled development only?
- **Q2.** Run H's settings break the 50,000 cap at every seed count: one standing one-off for the macro view, or tighter thresholds?
- **Q3.** The card is the walk-forward's: does the 2-4 minute measurement go before or after it?
