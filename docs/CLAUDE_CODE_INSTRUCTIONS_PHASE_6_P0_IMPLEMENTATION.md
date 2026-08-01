# CLAUDE_CODE_INSTRUCTIONS_PHASE_6_P0_IMPLEMENTATION.md — REV1

**Phase 6-P0: publish dataset version one and its pointer manifest, in place.**

**P0 CREATES FILES. P0 DOES NOT CHANGE RUNNING CODE.** That boundary is the deliverable's
defining constraint — see §0.2. If a change to a `.py` file on the certifying path seems
necessary, **stop and report** rather than crossing it.

**Base:** `216c74b` (scoping report committed). Claude Code on VM 101 as `michael`, venv
`~/venvs/torch`. Implement and iterate; you do **NOT** commit, push, or run WATCHER. STOP at the
gate for Team Alpha review.

**The rigs are off.** Nothing in P0 requires them. Anything rig-side is P0.5.

---

## 0. Two decisions already made — do not revisit

### 0.1 Publish **in place** (option b), not `~/datasets/`

`docs/PHASE_6_P0_SCOPING_v1.md` §4 established this with evidence, and it reversed Alpha's
initial sketch:

- **No hardcoded literal sits on the certifying Step-1 → miner → NPZ path.** Every link from
  `--lottery-file` (`window_optimizer.py:1258`, `required=True`) to `open(path)` in
  `load_residue_window` (`range_miner_worker.py:565`) is a parameter. Moving the authority
  breaks **17 call sites in 13 files**; publishing in place breaks **zero**.
- Moving has a failure mode publishing in place does not: leave `daily3.json` at the root and
  every consumer reads the stale file forever — and **the miner's `dataset_sha256` guard cannot
  catch it**, because the coordinator derives the expected digest from the same stale path it
  dispatches (`range_miner_coordinator.py:3499` → `:3421`). *The guard proves agreement, not
  authority.*

**So `daily3.json` stays exactly where it is** and becomes the pointer's resolved target.

### 0.2 P0 creates files; P0.5 changes code

Pointer resolution at WATCHER, absolutising the dispatched path, freeze-at-run-start wiring and
the `FileNotFoundError` → `ResidueError` classification are all **cheap and high-value** — and
all deferred, deliberately. Mixing file creation with behavioural change makes the first
certification after publication ambiguous. **P0 must be inert with respect to every existing
run.**

---

## 1. What to create

### 1.1 The publication schema — freeze it before writing anything

The first version file's name is **permanent**. Document the schema in the deliverable before
creating the file.

```
daily3-<UTC>Z-<sha256[:12]>.json
```

**The `.json` extension is load-bearing.** `.gitignore:41` is `*.json`; that is what keeps
published files invisible to the clean-tree check at `run_finalizer.py:1589`. A version file
with any other extension would dirty the tree and **block certification**.

### 1.2 The version file

A **byte-exact copy** of the current `daily3.json` — sha256 `513648160d35…68f6`, 18,068
records, 1,380,711 bytes — published beside it. Verify the copy's digest equals the source's
after writing.

**Do not modify the content.** Scoping §6 confirmed the 22 zero-draw records are **genuine
data** (expected 18.07 by chance, observed 22, spread across indices 254–16,939, not clustered
at an import boundary), the certifying loader is zero-safe, and the two droppers
(`digit_sequential_sieve.py:161-162`, `coordinator.py:1881`) are **off** the certifying path.
Publish as-is **because the data is correct** — not because changing it would be inconvenient.

### 1.3 The pointer manifest

Atomically replaced (write temp → `os.replace`), so a reader sees the old or the new manifest,
never a partial one. Carry **inside the JSON**:

```
version_id · filename · sha256 · size_bytes · record_count
first_draw (date, session) · last_draw (date, session)
published_utc · dataset_lineage_id · predecessor_sha256 (null for version one)
notes: the 22 zero-draw records, and that they are genuine
```

**No sidecar files.** A `.sha256`, `.txt` or `README.md` companion in the publication directory
each dirties the tree and blocks certification (scoping §5.2). The digest lives **in** the
manifest.

**Do not name it `*_config.json` or `schema_*.json`** — existing `.gitignore` negations
un-ignore those patterns, which would defeat §1.1.

### 1.4 The read-only verifier — part of P0, not deferred

A script that re-derives the version file's digest and confirms it matches **both** the manifest
and the alias (`daily3.json`).

Scoping §8.1.6: *"without it P0 has published something nothing has ever checked."* This is P0's
clean control. It must be **read-only** — it verifies, it never repairs.

### 1.5 The correction protocol — documentation only

Record that a correction opens a **new lineage** and the old lineage is preserved and
audit-retained. **No enforcement code in P0.**

## 2. Verify the tree stays clean — the acceptance test that matters most

After publishing, `git status --porcelain` must show **no new untracked non-ignored files**.
Prove it, do not assert it:

- run `git check-ignore -v` against every file created and paste the result;
- run `git status --porcelain` before and after publication and diff the output.

**Known condition:** the tree is **already dirty** — four untracked `CLAUDE_CODE_BRIEF_S17*.md`
files plus `tmp/`. That is pre-existing and **not P0's to fix**; a certifying run would fail
today regardless. Report the before-state so your after-state is interpretable, and **do not
delete or commit those files.**

## 3. Out of scope — do not cross these lines

- **Do not modify any `.py` file on the certifying path.** No pointer resolution, no path
  absolutising, no freeze-at-run-start, no `ResidueError` classification. All P0.5.
- **Do not move, rename or modify `daily3.json`.**
- **Do not touch `.gitignore`** — including the dead `:42 !config_*.json` negation, which needs
  a governed decision because fixing it un-ignores an unknown set of files.
- **Do not fix the D1 falsy-zero droppers.** Non-certifying, and must not be bundled with a data
  publication.
- **Do not fix the §2.1 WATCHER check-one-path/use-another defect** (preflight resolves
  `<REPO_ROOT>/daily3.json` at `watcher_agent.py:489`; dispatch is `Popen(cmd, …)` at `:1948`
  with no `cwd=` and the child gets the bare string). Real, reported, **not P0**.
- **Do not touch `daily3_midday.json` / `daily3_evening.json`.** P0 publishes the **combined
  file only**; the split files remain unversioned and unbound — an open item, not addressed
  here.
- Do not SSH to the rigs. Do not run WATCHER, the sieve, any GPU kernel, or the pipeline.

## 4. Verification-integrity controls (VIR-1…6)

- **execution proof** — the version file's digest is re-derived after writing, not assumed from
  the copy operation.
- **clean control (VIR-2)** — the §1.4 verifier passing on a correctly published set.
- **fault-injection control (VIR-2)** — this one is required and is easy to skip: **corrupt a
  scratch copy** (flip a byte, truncate a record) and show the verifier **fails**. A verifier
  that has only ever seen good input is unproven. Do this against a scratch directory, never
  against the published artifacts.
- **completion sentinel (VIR-3)** — explicit `PASS | FAIL | UNAVAILABLE | INCOMPLETE`.
- **unavailable-observer (VIR-5)** — the rigs are off; anything rig-side is `UNAVAILABLE`, not
  assumed.
- **audit claim scope (VIR-6)** — declare searched and unavailable surfaces.

## 5. Non-regression

P0 changes no running code, so a full suite pass is not the point — **proving inertness is.**

- `tests/test_s172_phase4_coordinator.py` — 63/63. If gate 22 flags a new `.py` (the verifier),
  register it in the allowlist **with rationale**, per the established pattern.
- `tests/test_chapter1_p0_corrections.py` — 12/12.
- **State explicitly that no existing run behaviour changed**, and how you know: no `.py` on the
  certifying path modified, `daily3.json` byte-unchanged (compare digest to the value in §1.2),
  nothing consuming the manifest yet.

## 6. Report

The frozen schema and the reasoning for each field. The published filenames with digests.
`git check-ignore` output for every created file, and the before/after `git status --porcelain`.
The verifier's clean-control and fault-injection results. Explicit confirmation that no `.py` on
the certifying path was modified and that `daily3.json` is byte-unchanged. The P0.5 and P2
deferrals restated so the boundary is on the record. Then STOP. **Do not commit.**
