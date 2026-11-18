# Cache policy v4 (DRAFT, nothing landed): fresh per-run per-worker CuPy cache for counted runs

Answers every Beta X-GCVM-LOCATED R3 requirement. Thread for review: **X-CACHE-POLICY**.

**Nothing here is landed:**
- `scripts/launch_fleet_manual.sh`, `gate12_launch.sh` and `/home/michael/run4_launch.sh` are **not** edited;
  the diffs apply to copies.
- The two new scripts live here, at the paths they would take under `scripts/`.

## Files

| File | sha256 | Applies to / lands as |
|---|---|---|
| `launch_fleet_manual.diff` | `16988e5d…` | `scripts/launch_fleet_manual.sh` @ `048fd47` (`21ff9399…`) |
| `gate12_launch.diff` | `20e4b0e8…` | `gate12_launch.sh` @ `048fd47` (`ec9ae1e9…`) |
| `run4_launch.diff` | `ba3b5968…` | Route A `/home/michael/run4_launch.sh` (outside the repo, `9cd3aebf…`) |
| `scripts/cache_records_verify.py` | `644199c8…` | new `scripts/cache_records_verify.py` |
| `scripts/gate12_cache_join_gate.py` | `100209d3…` | new `scripts/gate12_cache_join_gate.py` |
| `tests/test_cache_policy_v4.py` | `b66a61ca…` | tests and mutants (CPU only) |

Run:
- `python3 -B docs/drafts/cache_policy_v4/tests/test_cache_policy_v4.py`: **65/65**, 3 consecutive runs.
- `… mutants`: **14/15 detected**; L2 is equivalent, see below.

## What v4 does (against Beta R3)

**Before landing:**

| R3 requirement | v4 |
|---|---|
| One complete cache record and one complete ACK per expected rig slot; literal fields; host and prep time required; record bound to its dispatch endpoint | `cache_records_verify.py`. Each slot needs exactly one `[cache] OK` line whose fields are **exactly** the ten keys. Each is compared by **string equality**: tag, host == that dispatch's worker host, gpu, policy, path, created, entries, env == path; t_prep must be a decimal time and pid digits. It also needs exactly one **whole-line** `[dispatch] started pid=<d> gpu=<d>`, whose gpu equals the slot's gpu and whose pid equals the record's pid. Expected slots come from the **fleet config** (`RIG_SLOTS`), not from the dispatch arrays. Expectations must be unique per (host, gpu) and per (endpoint, path) |
| Fail if the output cannot be written in full; publish only after complete success | All slots are checked first. Then the TSV goes to a temp file (`open "x"`), is flushed and `fsync`ed, then `os.replace`, then the directory is `fsync`ed. Any exception, including an interrupt, gives rc 2 and removes the temp file. A defect gives rc 1 and publishes nothing. A pre-existing TSV for the tag gives rc 3 (written once). The launcher refuses the fleet on any nonzero rc, before "all … dispatched" |
| A test for each R3 defect | U-VERIFY: unavailable TSV path; unwritable output directory; interrupted write (rename fails); interrupt signal during publish; dotted tag mismatch (`caseX1` vs `case.1`); dotted root mismatch (`/tmp/cacheYroot` vs `/tmp/cache.root`); missing host; empty host; missing t_prep; non-decimal t_prep; wrong-GPU ACK; a diagnostic containing `started pid=` instead of an ACK; a full ACK embedded in a longer line; absent ACK; duplicate ACK; duplicate record; extra field; host suffix; path suffix; record host not the dispatch host; record pid ≠ ACK pid; env ≠ path; `policy=shared` in a record; expectation count ≠ slots; duplicate expectations. **Clean control:** equal paths on two CTs are accepted, and the TSV is published exactly once |
| An executable test of the edited launcher with stubbed transport; exit 14, exit 15 and a bad record each stop the fleet before the coordinator; shared and Zeus as controls | E2E runs the **patched launcher itself**: 2 rigs × 2 GPUs + 1 local Zeus GPU, stub `ssh` (each fake CT has its own `/tmp` and cache root, mapped back in returned text), inert worker stand-ins that keep their argv, stub `pkill`. **Clean:** rc 0, 4 rows, every stand-in's **real** `/proc/<pid>/environ` carries its fresh, empty path. **exit 14** (pre-existing cache dir), **exit 15** (stand-in exec's with another `CUPY_CACHE_DIR`) and a **bad record** (wrong CT hostname): each gives launcher rc ≠ 0 and no TSV. **shared:** rc 0, no `[cache]` lines, no TSV, the old per-GPU directory used. **Zeus:** the local stand-in carries `/tmp/cupy_cache_local_gpu0` and has no record. **E2E-GATE12** runs the patched `gate12_launch.sh` **and** `run4_launch.sh` slice (guard → provenance → fleet call → abort) with the patched launcher. Each fault exits 1 and never reaches the next gate (sentinel), hence never the coordinator or release; clean reaches it |
| `per_run` mandatory on the counted path, including Route A; an exported `shared` cannot pass | Both entry points, before the fleet call: an **inherited** `RIG_CUPY_CACHE_POLICY` other than `per_run` → `GATE-12 REFUSED`, exit 1, before the launcher runs (tested for both). Then `export RIG_CUPY_CACHE_POLICY=per_run`. `shared` remains for explicit diagnostic runs of the launcher alone |

**Before the first counted run** (implemented in the drafts; first exercised on the first verified run):

| R3 requirement | v4 |
|---|---|
| Join each worker's cache record to its current-nonce, sentinel-correlated parked liveness row; refuse on missing, mismatched or ambiguous | `gate12_cache_join_gate.py`, wired in **both** entry points right after the worker liveness gate, before the sampler and coordinator. Literal join on nonce (liveness `run_nonce`, record tag, and `--run-nonce <nonce>` in the parked argv), endpoint, worker id (`<host>:gpu<N>`), gpu and pid. The liveness row must be status OK; its sentinel correlation is reused. Exactly the expected rig count on both sides; duplicates refuse. Zeus is recorded separately as local. U-JOIN covers a clean join and refusals for: different parked pid, different nonce in argv, liveness nonce differs, liveness not OK, missing record, record tag differs, wrong / missing / duplicated parked `CUPY_CACHE_DIR`, unreadable environment, transport failing after printing the right value. E2E runs the join over the **real** parked stand-ins |
| Re-read `CUPY_CACHE_DIR` from the parked process | Same gate. A read-only `/proc/<pid>/cmdline` check (must carry `--run-nonce <nonce>`), then `/proc/<pid>/environ` over SSH. It needs exactly one `CUPY_CACHE_DIR`, equal to the startup path. Any SSH rc ≠ 0 refuses, even if the output looked right |
| Record launcher and entry-point versions | Both entry points write `logs/gate12_<stamp>_provenance.txt` **before the fleet starts**: sha256 of the entry point and of `launch_fleet_manual.sh`, `cache_records_verify.py`, `gate12_cache_join_gate.py`, `gate12_worker_liveness_gate.py`, `gate12_sentinel_gate.py`; `git rev-parse HEAD`; dirty-line count; policy, rig slots and nonce. Any missing or unhashable file → `GATE-12 REFUSED` (tested) |
| Freeze the TSV, dispatch logs and parked evidence | After a passing join: `logs/gate12_<stamp>_cache_freeze/` gets the records TSV, the expect TSV, all `dispatch_*.log`, the liveness JSON, the join JSON and the provenance file, plus `SHA256SUMS`. All of this happens before the sampler, because the dispatch logs are reused by the next launch. A failed freeze aborts like a failed join (rc 97) |
| Measure start-up time and cache footprint on the first verified rig run | **Plan:** per worker, start-up = sentinel time − `t_prep` (both in the frozen evidence: `cache_records_<nonce>.tsv`, sentinel log). Compare with the existing start-up window and stagger. Footprint = `du -sb /tmp/cupy_cache_run_<nonce>_gpu<N>` on each CT after the run (read-only), and the count of retained `cupy_cache_run_*` directories per CT. Recorded in the run report; cleanup of old run directories stays with the operating procedure, outside worker start-up |

## Not done, stated plainly

- **Mutant L2 survives.** L2 sizes the expectation from the dispatch arrays instead of the fleet config. It is
  equivalent under the current launcher, whose `DISPATCHED != TOTAL` guard already refuses a short dispatch
  before verification. It is kept as defence in depth.
- **Not run:**
  - the full `gate12_launch.sh` / `run4_launch.sh` end to end (clean-tree, GPU, parity, sentinel and liveness
    gates, coordinator). The tests run the entry-point **slice** from the policy guard through the fleet-call
    abort, plus the join gate separately;
  - anything on the rigs.
- **Not covered:** the Zeus local cache stays `/tmp/cupy_cache_local_gpu<N>` (shared across runs), as before.
- **Redeploy after landing:** the join gate and verifier are VM101-side files. They are not in the CT parity
  closure; only the launcher-side bytes change. Provenance pins them per run.
