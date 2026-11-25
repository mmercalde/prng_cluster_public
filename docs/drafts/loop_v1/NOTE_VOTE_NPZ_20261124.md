# NOTE — run H full vote matrix as an NPZ (TASK_MISS_VOTE_NPZ_20261124)

DEVELOPMENT / DESCRIPTIVE, NOT COUNTED. Written 2026-11-24 by the miss-report agent, worktree
`e2e-vote`. Nothing committed.

## Prior record

- Searched: `/bin/grep -rln "vote matrix\|vote_npz\|VOTE_NPZ" docs/` in this worktree. One hit:
  `docs/drafts/loop_v1/READY_LOOP_MISS_20261123.md`. That note records the matrix that the miss
  report built in memory (run3, 12,156 × 3,614 int16, 54.7 s on 6 workers;
  `data/e2e_loop_miss/run3/run_record.json` `vote_matrix`). It saved no file and no sha256.
- Request: `~/distributed_prng_analysis/logs/e2e_session/TASK_MISS_VOTE_NPZ_20261124.txt`
  (sha256 `9462fa9f…807a`), for ch14-learn READY_LOOP_LEARN_W2 §4 Q2. I have not read that
  note.
- Labels: the NPZ file is NEW. The matrix it holds is ALREADY RECORDED (the READY note above,
  §1). Nothing here changes a ruling.
- Not searched: the catalog, rulings, LEADS and exchange directories (this note makes no
  finding), the main tree, the host, ser8 archives, the private repository.

## The file

`data/e2e_vote_npz/runH_full_vote.npz`, 87,990,486 bytes,
**sha256 `cf18242b661c91ab9e2ba1e23d46326b0faf40bb639d3a16fa66940e20e5708e`**.
It was written with `np.savez`, uncompressed.

| key | dtype | shape | meaning |
|---|---|---|---|
| `V` | int16 | (12156, 3614) | `V[i, j]` = the module's `seq[combined_index[j]]` for survivor row `i` |
| `seeds` | int64 | (12156,) | the seed of row `i`, in Step 3 file order (not ranked order); 12,156 unique |
| `combined_index` | int64 | (3614,) | 14454 … 18067 (spec `vote_range` [14454, 18068)) |

- The sha256 of `V`'s raw bytes is `1520495b…6e1381`, recomputed after reloading. The values
  run from 0 to 999.
- Every row's skip is 0 (`skips_distinct: [0]`), so `V` needs no skip column.
- The run record is `data/e2e_vote_npz/runH_full_vote.npz.record.json` (`73c48d47…bd3`). It
  holds the code shas, the signatures and the timing.

## How it was built

- **Builder:** `scripts/diagnostics/e2e_vote_npz.py` (`4c2f3499…9ead`, new).
- **What it calls:** `e2e_miss_report`'s own `rebuild_ranking` and `full_vote`, through module
  import with no copies (miss report `ec7c7ad1…`, unchanged).
- **Inputs:** the same spec as miss run3, `data/e2e_loop_miss/spec_run1.json` (`248310d0…`).
  The builder repeats that run's arm loop (`e2e_miss_report.py:515-551`).
- **Checks, fail-closed:**
  - Every declared pool record must equal `V[order[:N], t − t0]`. This held for 5,421 A1
    records and 10,842 A0 records, 16,263 of 16,263.
  - A0 must have the same rows and skips as A1, and it did.
- **Nothing outcome-related was read:** no grade file and no draw value.
- **Resources:** `taskset -c 6,7`, `nice -n 15`, 2 forked workers, `CUDA_VISIBLE_DEVICES=""`,
  so the card was not touched. Building the matrix took 91.7 s and the whole run took 1:43.
  Peak RSS was 1.03 GB in the main process and 0.47 GB in the largest child. The log is
  `logs/e2e_vote_npz/run1.log` (`66ccb66d…d96d`).

## Limits

- Run3 recorded no sha256 for its matrix, so this file cannot be compared to it byte for byte.
  What shows they are the same matrix is the 16,263 pool-record check, together with the code
  and spec shas recorded above.
- Correction to my Addendum 3 report: it said no `__pycache__` existed. In fact, gitignored
  `__pycache__` directories dated 2026-11-23 are in this worktree, under `scripts/diagnostics/`,
  `tests/`, `utils/`, `models/`, `integration/`, the root, and
  `data/e2e_loop_miss/provenance/`. I have not removed them.
