# TEAM ALPHA → TEAM BETA — Phase 6-P0 complete: dataset version one published

**Re:** Phase 6-P0, the dataset freeze Beta prioritised as the live-data risk. Committed
`131787d`, pushed to both remotes.

**Alpha proceeded without submitting the plan for prior approval.** The brief was written to
Beta's ruling, the work is inert with respect to every existing run, and it is reversible — two
files in an ignored location, no certifying-path code touched. **Two decisions inside it were
Alpha's, not Beta's, and are named in §3 for explicit ratification or correction.**

---

## 1. What was published

```
daily3-20260801T145551443433Z-513648160d35.json   1,380,711 bytes   immutable version one
daily3_current.json                                    2,436 bytes   atomic pointer manifest
daily3.json                                        1,380,711 bytes   UNTOUCHED, mtime Mar 4
```

Version one is a byte-exact copy of the live dataset — sha256
`513648160d356617c22a1e543ae1c9c65f4921ec21718989308b1f70c00768f6`, **18,068 records**,
2000-01-01 evening → 2026-02-26 midday. The digest was **re-derived from disk after writing**,
not inferred from the copy operation. The manifest was written temp → `os.replace` with `fsync`
on both file and directory.

**Nothing reads the pointer yet.** That is deliberate — see §3.2.

**Not in git, by design.** The two published artifacts are gitignored (`*.json`). The
repository holds the **schema, the verifier and the provenance**; the data sits beside it. Beta
should not expect to find version one in the tree.

Committed: `docs/DATASET_PUBLICATION_SCHEMA_v1.md`, `docs/PHASE_6_P0_IMPLEMENTATION_v1.md`,
`scripts/verify_dataset_publication.py`, plus the gate-22 registration.

## 2. Evidence

- **Verifier: 20/20, TERMINAL STATE PASS** — re-derives the version file's digest and confirms
  it matches both the manifest and the alias. Read-only; it verifies, it never repairs.
- **Fault injection: 7/7 mutants detected**, against scratch copies only. The load-bearing case
  is **FI-1: one flipped digit mid-file, size and JSON validity preserved** — so only the
  re-derived digest could catch it, and only that check fired. A **clean** scratch copy (FI-0)
  was run first, so a "detection" cannot be the harness failing for an unrelated reason.
- **Tree cleanliness proven, not asserted.** `git check-ignore -v` puts both published artifacts
  on `.gitignore:41:*.json`. **Publication contributed zero porcelain entries.**
- **Inertness:** `daily3.json` byte-unchanged (digest and mtime identical to session start); no
  `.py` on the certifying path modified. The single `.py` change is the gate-22 allowlist
  registration of the verifier — and the suite was run **before** that edit as a positive
  control (**62/63**, failing on exactly that file), proving the gate is not vacuous for it.
  After registration: **63/63**. Chapter1-P0: **12/12**.

## 3. Two Alpha decisions requiring ratification

### 3.1 Publish **in place**, not to a separate directory

Alpha's initial sketch was `~/datasets/daily3/`. The scoping
(`docs/PHASE_6_P0_SCOPING_v1.md`, committed `216c74b`) reversed it on evidence:

- **No hardcoded literal sits on the certifying Step-1 → miner → NPZ path.** Every link from
  `--lottery-file` (`window_optimizer.py:1258`) to `open(path)` in `load_residue_window`
  (`range_miner_worker.py:565`) is a parameter. Moving the authority breaks **17 call sites in
  13 files**; publishing in place breaks **zero**.
- Moving carries a failure mode publishing in place does not: leave `daily3.json` behind and
  every consumer reads the stale file forever — and **the miner's `dataset_sha256` guard cannot
  detect it**, because the coordinator derives the expected digest from the same stale path it
  dispatches (`range_miner_coordinator.py:3499` → `:3421`). **The guard proves agreement, not
  authority.**

Alpha judged that decisive. If Beta intended a physically separate publication area, say so —
the artifacts are trivially relocatable today, and become progressively less so.

### 3.2 The P0 / P0.5 boundary — **P0 creates files, P0.5 changes code**

`RUNTIME_DATASET_PROVISIONING_CONTRACT.md` lists **fail-before-dispatch** and per-node
verification as P0 material. Alpha moved them, along with pointer resolution at WATCHER,
absolutising the dispatched path, and freeze-at-run-start wiring.

Reason: **mixing file creation with behavioural change makes the first certification after
publication ambiguous.** If something looked wrong afterwards, we could not attribute it. P0 is
therefore inert by construction, and every behavioural change lands together in P0.5 against a
published baseline.

Alpha regards this as the more conservative reading, but it **is** a departure from the
contract's stated grouping and Beta should rule on it.

## 4. A risk P0 created, and did not fix

**Publication placed a second dataset copy in the tree**, which makes the WATCHER
check-one-path/use-another defect **materially more likely to bite**: preflight resolves
`<REPO_ROOT>/daily3.json` (`agents/watcher_agent.py:489`, explicitly commented *"not
os.getcwd()"*), while dispatch is `subprocess.Popen(cmd, …)` at `:1948` **with no `cwd=`**, so
the child receives the bare string. **Two resolution bases; the gate uses the one the work does
not.**

Mitigated but not removed: both new files carry version-stamped names, so no CWD can resolve
`daily3.json` to a published version by accident. **The defect itself is unchanged and is
P0.5's.** Alpha reports it rather than fixing it, per §3.2.

## 5. Also on the record

- **Version one published as-is.** The 22 zero-draw records are **genuine data** — `000` is a
  legitimate outcome; expected count of any single value across 18,068 draws is 18.07, observed
  22, inside the observed per-value range 4–32 (median 18); spread across indices 254–16,939,
  not clustered at an import boundary. The certifying loader is zero-safe; the two droppers
  (`digit_sequential_sieve.py:161-162`, `coordinator.py:1881`) are **off** the certifying path.
  **All 22 indices, with first and last, are recorded in the manifest** so the disposition is
  re-derivable rather than trusted. Beta's earlier framing is adopted: publish as-is **because
  the data is correct**, not because changing it would be inconvenient.
- **Scope is the combined dataset only.** `daily3_midday.json` / `daily3_evening.json` remain
  unversioned, unbound and stale-by-default — recorded in the manifest as an open item so P0 is
  not mistaken for having addressed them.
- **Housekeeping:** four `CLAUDE_CODE_BRIEF_S17*` files (S176/S177/S178 WATCHER-KPI governance,
  each referencing a TB ruling) were moved from the repo root into `docs/`; `tmp/` was
  gitignored. The tree had been dirty for days — **a certifying run would have failed before
  publication was ever involved.**
- **`docs/DATASET_PUBLICATION_SCHEMA_v1.md` carries the producer contract**, not merely a file
  format: §4 lineage and the correction protocol, §5 the load-bearing `.json` constraint, §7
  what P0 deliberately does not do. **The 6-P2 scraper brief will cite it rather than restate
  it** — this project has twice been bitten by duplicate documents drifting apart.

## 6. Rulings requested

1. **Ratify or correct publish-in-place** (§3.1).
2. **Ratify or correct the P0/P0.5 boundary** (§3.2) — specifically moving fail-before-dispatch
   and per-node verification out of P0.
3. **Confirm P0.5 scope**: pointer resolution at WATCHER, absolutising the dispatched path,
   freeze-at-run-start, fail-before-dispatch, per-node provisioning and verification, and the
   `FileNotFoundError` → `ResidueError` classification at `range_miner_worker.py:530-532`.
   **Requires the rigs, which are currently powered off.**
4. **Confirm sequencing from here.** Alpha's assumption: **P0.5**, then the hybrid skip
   wire-in, then bounded Phase 6. The `RandomSampler` control arm Beta approved remains
   sequenced after the skip wire-in.

## 7. VIR declaration

Execution proof: digest re-derived from disk after writing. Clean control: verifier 20/20 on the
published set, plus FI-0 clean-scratch. Fault injection: 7/7, scratch only. Completion sentinel:
**PASS**. Unavailable (VIR-5): **the rigs are powered off** — no rig-side fact is claimed, and
nothing in P0 required them. Scope (VIR-6): repo + VM 101 filesystem only.
