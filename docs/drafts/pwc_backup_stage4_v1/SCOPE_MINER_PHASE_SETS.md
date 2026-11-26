# Scope: let a RANGE-MINER run keep its four per-phase seed→rate sets for the comparator

Read-only scoping note, 2026-11-21, after Stage 4. No code was written; there was no rig contact; the live ledger
was not touched.
- **Anchors** are at the main tree's HEAD `6eafad9`, the Stage 4 run trees' commit.
- **Evidence** is read from `~/distributed_prng_analysis/logs/stage4_session/`.

**The question:** what is the smallest change that lets a RANGE-MINER run keep its four per-phase (seed, rate)
sets, without changing what RANGE-MINER computes, publishes or credits? Keeping them would close the 10 fields
Stage 4 reported NOT COMPARED.

## Prior record

- **Searches:**
  - the plan's gap 3 and its decision row: `grep -n -i "gap" docs/drafts/pwc_backup_stage4_v1/REDEPLOY_AND_SESSION_PLAN.md`;
  - the runbook's expectation: `SESSION_RUNBOOK.md:499`;
  - the PWC adapter's count-only accumulation: `window_optimizer_integration_final.py:1256-1258`;
  - the miner adapter: `:1272-1365`;
  - the miner sink: `miner/range_miner_npz_writer.py:223-245`, `:1136`, `:1296-1302`;
  - the staged-spool deletion at commit: `miner/range_miner_coordinator.py:8329-8345`, `:8755-8767`;
  - the rig pins: `scripts/gate12_parity_gate.py:174-185`, `scripts/pwc_parity_gate.py:107-131`;
  - the child environment: `agents/watcher_agent.py:2889-2901`;
  - the clean-tree rule: `window_optimizer_integration_final.py:119-120`, and `git check-ignore logs/ data/`.
- **Found:**
  - **Gap 3 is ALREADY RECORDED.** "Per-phase survivor sets are not persisted by either engine"
    (`REDEPLOY_AND_SESSION_PLAN.md:370-378`). The decision is left open: "a capture hook (an integration change, so
    Beta review) or 'not compared'" (`:412`).
  - The Stage 4 result is that open decision, measured: exit 3, 10 fields NOT COMPARED (`compare.txt`).
  - My extractor README r3 records why the miner bundle writes null.
- **Labels:**
  - **NEW:** the options and the recommendation below.
  - **Nothing here CHANGES A RULING.**
- **Not searched:** the host, the ser8 archives, the private repository, the rigs.

## The measured facts

- **Interval:** Stage 4 swept `[2^31, 2^31 + 2^28)`, both engines. Per-phase rows were **0 / 0 / 13 / 2,995**
  (`bundle_pwc/bundle.json` `phase_counts`; miner `extract_miner.log`). The NPZ had zero bidirectional rows.
- **The miner holds the four maps only in memory.** It holds them **in full**, inside the committed
  `MinerTrialAssembly`: `forward_map_constant`, `reverse_map_constant`, `forward_map_variable` and
  `reverse_map_variable`, each `dict[int, float]` (`range_miner_npz_writer.py:223-245`).
- **The Step 1 adapter already has that object.** `require_assembly(...)` returns it at
  `window_optimizer_integration_final.py:1351-1352`. It is then used only for `ingest_assembly` (`:1354`) and
  dropped.
- **No duplicates:** assembly refuses a duplicate seed within a direction (`DirectionalDuplicateError`,
  `range_miner_npz_writer.py:930`). So `distinct_survivors` = `len(map)` is exact once the map is kept.
- **The staged spools are deleted at commit.** `commit_trial`'s cleanup calls `ack_by_event_id`, which deletes the
  staged file and releases its reservation in the same step (`coordinator:8340-8344`, `:8759-8767`).
- **PWC keeps its lists in its pass files** only up to 100,000 survivors per pass; above that it keeps a summary
  only (`scripts/extract_pwc_bundle.py:15-17`, citing `persistent_worker_coordinator.py:2517-2532`). At Stage 4's
  size every PWC list was kept.

## Options

### A. Capture in the Step 1 miner adapter, behind an environment flag (recommended)

**What it does:** right after `ingest_assembly` (`window_optimizer_integration_final.py:1354`), when
`TFM_MINER_PHASE_CAPTURE_DIR` is set, it writes one file per trial. The file holds, for each phase:
- the sorted seeds (`uint32`) and the rates (`float32`);
- the row count;
- a sha256 over the canonical bytes;
- the trial's `run_id` and `trial_number`, so the extractor's receipt can bind it.

It reads the assembly and nothing else.
- **Placement:** after the accumulator append, so append order and flush cadence are untouched.
- **The write:** atomic (temp file, then rename), into the capture directory only.

**Files touched:**
- `window_optimizer_integration_final.py`: one guarded block of about 30–40 lines. This file is **not** in either
  rig pin (`gate12_parity_gate.py:174-185`, `pwc_parity_gate.py:107-131`) and is not under `miner/`,
  `agents/watcher_agent.py` or `agent_manifests/`. It is the Step 1 launch path, so it is a **before-commit Beta
  review** class (CLAUDE.md, Beta reviews).
- `scripts/extract_miner_bundle.py` and its test (mine): read the file, check its sha256, bind it to the receipt,
  and fill `phases.1-4`, `bidirectional.*` (the per-mode intersections) and `distinct_survivors`. That closes all
  10 fields.

**Rig files:** none change, so **no redeploy**. The capture runs on VM101, inside the Step 1 process.

**How it stays off:**
- Unset or empty means the block returns before doing anything.
- An environment variable needs **no** change to `window_optimizer.py`'s parser, `agent_manifests/window_optimizer.json`
  or WATCHER, because WATCHER copies its environment into the Step 1 child (`watcher_agent.py:2889`, `env=env`
  at `:2901`). The operator exports it before launch; `gate12_launch.sh` is unchanged.
- **The trade-off:** a CLI flag would be the project's usual explicit route. It would mean touching
  `window_optimizer.py` and, for WATCHER launches, the governed manifest `args_map`, which is a wider change for a
  diagnostic.
- **Either way the capture must record that it was on.** Its file's existence, plus a line in the Step 1 log, say
  so; off is the absence of both.

**Where it may write:**
- The directory must be **outside the run tree, or under the ignored `logs/`** (`.gitignore:85`). `data/` is **not**
  ignored (`git check-ignore data/x` gives no match), and `finalize_run` refuses a dirty tree
  (`window_optimizer_integration_final.py:119-120`).
- A capture into `data/` would therefore stop publication. That would change what RANGE-MINER publishes, so the
  block must refuse such a directory up front.

**If the capture fails** (disk, permissions): log it loudly and continue. The run's outputs do not depend on it.
The extractor then finds no file and keeps the nulls, NOT COMPARED, as today. It never fakes an empty set.

**Cost:**
- **Memory:** the assembly already holds the maps, so the extra is one serialization buffer, about 8 bytes per
  entry as NumPy arrays.
- **Disk at 2^28 (measured shape):** 3,008 entries, about 24 KB as NPZ or about 90 KB as JSON.
- **Disk at 2^31:** about 8× if survivors spread evenly [UNVERIFIED], so about 24,000 entries, about 190 KB NPZ or
  about 0.7 MB JSON. That is consistent with the survivor-volume measurement at `[0,16]` over 2^31: 0/0/60/23,568
  (`docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md:13-18`).
- **Worst case:** no larger than the spool volume the run already staged for the same trial, which is bounded by
  `staging_high_water_bytes`, default 16 GiB (`coordinator:604`). Looser thresholds would raise phases 1–2; the
  write is linear in the map sizes the run already held.

**How to prove the output is byte-identical with the flag off (and on):**
1. **Diff confinement:** `scripts/prove_unchanged_outside.py` on the integration file shows the only change is the
   guarded block.
2. **The existing suites:** `tests/test_s172_phase5_d6_production_adapter.py` and
   `test_s172_phase5_d6_threshold_path.py`, unchanged and green, one at a time.
3. **A new gate:** the same synthetic committed trial through `_build_test_result_from_miner`, with the variable
   unset and then set. It must give:
   - identical accumulator records and counters;
   - an identical `TestResult`;
   - identical flushed-NPZ bytes;
   - an identical `finalize_run` artifact sha256 and sidecar, apart from `created_at`.
   With the variable unset, a filesystem snapshot must show **no new file anywhere**.
4. **Mutants:**
   - the capture mutating a map in place (sorting the dict or popping a key) must be caught by the record identity
     check;
   - the capture running when unset must be caught by the snapshot;
   - a capture directory inside the run tree must be refused.
5. **The parity pins:** the gate12 and PWC parity gates' expected digests are unchanged, because no pinned file
   changed.
6. **Optional, on a rig:** repeat the Stage 4 miner arm with capture on. All 22 arrays must be EQUAL to the banked
   run (artifact `aecc5d50…`) through the comparator. Whether the NPZ bytes themselves repeat across runs is
   [UNVERIFIED]; the arrays are the test.

### B. Keep the staged spools (skip deletion at commit), behind a flag (not recommended)

**Files:** `miner/range_miner_coordinator.py`, where deletion and release are one step in `ack_by_event_id`
(`:8340-8344`).

**Pinned:** yes. It is in the gate12 `GOVERNED_FILES` (`gate12_parity_gate.py:180`), so its digest changes and the
parity gate fails until the rigs are **redeployed**, even though the rigs never execute the coordinator. It is
also `miner/`, which the owner lands by hand.

**Behaviour:** it changes staging behaviour. Reservations are released when the file is deleted. Keeping files
while releasing reservations breaks the disk accounting behind the high-water bound (`coordinator:604`), and
keeping both changes capacity for later trials.

**Cost:** the spools are JSON per shard, so disk is larger than A's. It also needs a second reader that
re-implements the assembly's spool validation. **It changes the run, so it fails the brief.**

### C. Capture inside the miner sink at commit, behind a flag (second choice)

**Files:** `miner/range_miner_npz_writer.py` (`AssemblingPhase5Sink`, `:1136`). It is not in the gate12 pin, so
**no redeploy**. It is `miner/`, so the owner lands it by hand.

**The objection:** it runs inside commit delivery. An exception there marks `commit_delivery_status` failed and
retains the staging (`coordinator:8674-8690`), so a capture fault could change the commit outcome unless it is
swallowed.

**Compared with A:** the same data and cost as A, with more coupling. Choose it only if the capture must also
cover miner use outside Step 1.

### D. The cheaper alternative: an order-independent digest per phase

**What it is:** the sha256 of each phase's (seed, rate) set over a canonical sorted encoding (`uint32` seed LE ‖
`float32` rate bits). Sorting makes it independent of arrival order, so a multiset hash is not needed. It is 32
bytes per phase.

**Where it would come from:**
- **PWC:** computable today in the extractor from the pass files, with no PWC change, but only for passes of
  100,000 survivors or fewer.
- **Miner:** **it still needs A's hook.** The maps exist only in memory, so a digest does not shrink the code
  surface; it only shrinks the bytes, which are trivial already (about 190 KB at 2^31).
- **Comparator:** it would need a new field, which is agent 1's change.

**What it loses:** localization. A differing digest names no seed. That goes against the comparator's rule that a
difference is localized, not summarized (`compare_engine_outputs.py`, docstring "Its method is kept").

**Digest plus seed lists without rates:** this localizes membership but not rates, and F3 requires each seed's
rate equal (PLAN.md §F3). The rates add only 4 bytes each.

**Verdict:** D is not cheaper where it matters. Use the digest as an **integrity check inside A's file**, not
instead of it.

## Recommendation

**Option A:**
- one guarded block in `window_optimizer_integration_final.py`, enabled by `TFM_MINER_PHASE_CAPTURE_DIR` and off by
  default;
- writing outside the run tree, or under `logs/`;
- with a per-phase sha256 inside the file;
- plus the extractor change on my side.

**Why:**
- no rig file changes, so no redeploy;
- no governed manifest or WATCHER file is touched;
- disk use is a few hundred KB at 2^31, and memory is already paid;
- it closes all 10 NOT COMPARED fields.

**What it needs:**
- Beta review before commit, as the Step 1 launch path;
- the flag-off proof in steps 1–5 above.

**Open for the owner:**
- **The flag route.** The environment variable is smallest. A CLI flag is the explicit project route, at the cost of
  touching the governed manifest `args_map`.
- **The PWC side.** PWC keeps lists only up to 100,000 per pass. A run with larger phase-1/2 populations would
  leave PWC's phases NOT COMPARED, with the asymmetry reversed. Stage 4's size is well inside that limit.
