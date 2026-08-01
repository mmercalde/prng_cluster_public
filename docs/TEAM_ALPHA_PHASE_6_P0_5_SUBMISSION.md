# TEAM ALPHA → TEAM BETA — Phase 6-P0.5 complete: the dataset authority cutover

**Re:** Phase 6-P0.5. Committed `d4ff1e4`, pushed. Changelog:
`docs/SESSION_CHANGELOG_20260801_PHASE_6_P0_5.md`.

**Beta's P0 procedural exception was explicitly not precedent, and Alpha did not treat it as
one** — this brief was written to Beta's confirmed §3 scope before implementation began.

**Four items require a ruling (§6). One is operational and affects day-to-day work.**

---

## 1. What changed

The pointer manifest is now authoritative. `daily3.json` is a **legacy compatibility alias**,
per Beta's P0 ruling §1.

| # | requirement | where |
|---|---|---|
| 1 | WATCHER resolves the pointer | `agents/watcher_agent.py:477`, `:495`, `:1473-1527` |
| 2 | one-time run-start freeze — manifest/version, absolute path, sha256, size, record count | `miner/dataset_authority.py:576`; `window_optimizer.py:1443-1484` |
| 3 | dispatch the absolute immutable path, never the bare alias | `window_optimizer.py:1484`; `agents/watcher_agent.py:1483` |
| 4 | fail before first worker dispatch | `miner/dataset_authority.py:904` — before `MultiGPUCoordinator` construction and before `Popen` |
| 5 | per-node provisioning + **on-target** verification | `miner/dataset_authority.py:704`, `:828`; `scripts/provision_dataset_fleet.py` |
| 6 | run provenance recording the frozen values | `miner/dataset_authority.py:1015` |
| 7 | pointer movement mid-run must not alter that run | freeze + `miner/range_miner_coordinator.py:85` |
| 8 | pointer must name a permitted version-stamped filename | `miner/dataset_authority.py:253` |

**+482 / −9 across seven files**, plus three new. Almost purely additive: new paths added,
existing behaviour left in place.

### 1.1 The scope defect fixed along the way

`range_miner_coordinator.py:3499` derived `dataset_sha256` **per trial**. A scrape between two
Optuna trials changed the bytes under a study, and **every downstream check stayed
self-consistent against a different dataset, with no error anywhere.** Now resolved through
`resolve_dataset_sha256` (`:85`), which answers from the run-start freeze when the path is the
frozen one and **falls back to the exact pre-P0.5 hashing behaviour otherwise** — so no existing
caller changes meaning. This is what makes Beta's requirement 7 mean anything.

### 1.2 Beta's exception correction, implemented as ruled

`DatasetProvisioningError(ResidueError)` at `miner/range_miner_worker.py:523`. The subclassing
is **not** category-smuggling: the coordinator's failure matrix routes that hierarchy to
`stripe_error(retryable=False)`, and a provisioning fault genuinely is not retryable —
*retrying a stripe against a node with no dataset produces the same failure more slowly.*
Original exceptions chained; absolute path **and node** named.

**A path the brief did not anticipate:** `load_residue_window`'s classification looks
unreachable from the worker (the digest check touches the file first) — but the **coordinator**
calls it directly via `_miner_residues_for_config`, where it *is* the first touch.

## 2. A correction to Alpha's own brief

Alpha recorded `rrig6600` `.122` as **matching**. That was true of `daily3.json` — but **the path
dispatched after P0.5 is the version file, which `.122` did not have either.**

**All three nodes failed the absent-dataset control**, captured against real state before
provisioning. That is a stronger fault-injection result than the brief designed for: Alpha
expected only two nodes could exercise it.

All three then provisioned **through one path, no special-casing** — a provisioning step that
skips a node it believes correct cannot detect the case it exists for — and verified **on
target**: `513648160d35…68f6`, PASS/PASS/PASS.

## 3. Evidence

- **P0.5 harness 33/33 with the live-fleet gate** (`--fleet`), 32/32 without. Every negative
  path fault-injected into a tempfile publication tree: pointer missing · unparseable · names
  the alias · absolute path · traversal · six non-conforming names · target absent · digest,
  size and record-count mismatch · schema mismatch · unpublished alias · absent node · digest
  mismatch on node · unreachable node → **UNAVAILABLE** · **pointer moved mid-run → run
  unaffected**.
- **Non-regression all green:** D1.1 18/18 · D4 8/8 · D5 24/24 · D6 3.A 9/9 ·
  **D6-threshold 17/17** · D6.1 15/15 · **threshold-propagation 5/5** · Chapter1-P0 12/12 ·
  Phase 3 17/17 · **Phase 4 63/63**. P0 verifier still 20/20.
- **Published artifacts byte-unmodified** — `daily3.json`, the version file and the pointer all
  **re-hashed at the end of the harness** (gate 31), not asserted.
- **Gate 32 asserts hybrid skip is NOT wired in** — scope compliance as a test rather than a
  claim in a report.

### 3.1 Two scope tripwires required registration

Both are per-deliverable git-status whitelists; both were extended **by registering with a
rationale, never by loosening the predicate**.

- **gate 22** — three new `.py` files registered. The last commit to touch that file was
  `131787d`, the same lineage, so the block was **appended to** rather than rewritten.
- **`G-MINER-UNCHANGED`** — this gate asserts the *threshold repair* left `miner/` alone. P0.5
  is a different deliverable that necessarily changes `miner/`. Rather than widen it, Alpha
  **strengthened** it: the kernel/executor surface (`sieve_gpu_worker.py`, `prng_registry.py`,
  `pwc_protocol.py`) must still be byte-identical, **any other `miner/` file still reds**, and a
  new check greps the registered files' diff for threshold tokens. *A registration is a claim,
  so verify it rather than trust it.*

## 4. §4 divergence — closed for the dataset, narrowed structurally

Preflight and dispatch now resolve to the **same absolute path**, and an absolute path has no
CWD dependence. **Proven from `CWD=/tmp`:** the dispatched value resolves correctly where the
old bare string would have become `/tmp/daily3.json`.

**Not closed generally.** The `Popen` at `agents/watcher_agent.py:1948` still carries no `cwd=`,
and two other manifest params — `output` and `trse_context` — remain relative and therefore
CWD-dependent. Reported, out of scope here.

Adopting Beta's causality correction: this is a **pre-existing latent authority defect exposed
during P0**, not one P0 created.

## 5. Contract amendment applied (Beta P0 ruling §2)

`docs/RUNTIME_DATASET_PROVISIONING_CONTRACT.md` — phase attribution corrected to **6-P0.5** with
the ratified boundary table; static `expected_sha256` marked **SUPERSEDED** in favour of
run-scoped frozen identity; §3 failure table carries Beta's classification correction; new §5.1
records the fleet state. Documentation only.

## 6. Rulings requested

**Q1 — the operational one. A local single-GPU run now refuses while any rig is down.**
`window_optimizer.py` verifies **every** node in the provisioning manifest. That is the
fail-closed reading of requirements 4 and 5, and Alpha implemented it deliberately — but it is
how most local testing has been done, including much of this week's.

**Alpha is not proposing a bypass.** If one is wanted it must be governed, not assumed. Alpha
also notes Michael's recollection that a full-fleet requirement predates P0.5 via a GPU-count
check with Telegram notification — **a separate investigation is establishing whether P0.5
tightened an existing constraint or created a new one**, and whether the dataset check
(per-node) and the GPU check (per-GPU) agree. Alpha will report before requesting any
relaxation.

**Q2 — a missing provisioning manifest records `UNAVAILABLE` and the run proceeds.** Should its
absence instead hard-fail a miner-backend run? The brief did not settle it. Alpha leans toward
hard-fail for a miner run and permissive for non-fleet paths, but has no mandate.

**Q3 — `dataset_provisioning.json` is gitignored** per contract §1, so a fresh clone has **no
fleet definition** until one is placed. Same class as the `daily3.json` gap: the fleet works
until someone starts from a clean checkout.

**Q4 — `dataset_provenance/*.json` is never pruned.** Same class as the D6.3 checkpoint-pruning
blocker; not addressed here.

**Q5 — preflight freshness.** It now compares against the **version file's** mtime (Aug 1)
rather than the alias's (Mar 4), so steps 0 and 1 read STALE and re-execute on the next WATCHER
launch. Soft, non-blocking. Arguably correct under Beta's *"append-only does not make prior
scores valid"* ruling; the alternative reintroduces two resolution bases. Alpha recommends
leaving it.

## 7. Out of scope, untouched

Hybrid skip wire-in (**gate 32 asserts this**) · RandomSampler arm · any new publication ·
`daily3.json` / version file / pointer manifest · the split files · the falsy-zero droppers ·
`.gitignore`.

**Session splits remain unversioned and unbound.** Beta: accepting the combined publication is a
**provenance** ruling, not a finding that combined midday/evening records are analytically
appropriate for one PRNG model. **Session-separated dataset authority remains open work** and
Alpha does not treat it as settled.

## 8. VIR declaration

Execution proof: digests re-derived **on target**, frozen values read back from run provenance.
Clean control: full three-node run resolves, freezes, dispatches. Fault injection: the §3
negative table, plus the absent-dataset case captured against **real** pre-provisioning state on
all three nodes. Completion sentinel: **PASS** (33/33 live). Unavailable: none — the rigs were
reachable throughout. Scope: repo + VM 101 + the three CT100 workers.
