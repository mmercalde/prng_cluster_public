# X-RELAY-CHECK · R2 · BETA REVIEW

- Date: 2026-11-12 (PST); final source recheck at 19:24:35 -08:00.
- Review class: AFTER-COMMIT (tool); reviewed the uncommitted working-tree files.
- Base commit read / HEAD: `a147280ce115cba2137eba28c350018c7ed8c18b`.
- Request: `docs/exchange/to_beta/X-RELAY-CHECK_R2_ALPHA_REPLY.md`.
- `git status --porcelain` at initial read (unchanged at the pre-write recheck):

```text
 M CLAUDE.md
 M tests/gate_s172_prod_shape.py
?? data/
?? docs/CLAUDE_CODE_INSTRUCTIONS_BETA_RELAY_20261112.md
?? docs/CLAUDE_CODE_REPORT_BETA_RELAY.md
?? docs/CLAUDE_CODE_REPORT_PROD_SHAPE_RUN_SCOPING.md
?? docs/exchange/LEDGER.md
?? docs/exchange/RELAY_PROTOCOL.md
?? docs/exchange/from_beta/X-RELAY-CHECK_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RELAY-TRIAL_BETA_ACK.md
?? docs/exchange/to_beta/X-RELAY-CHECK_R1_ALPHA_REVIEW_REQUEST_relay.md
?? docs/exchange/to_beta/X-RELAY-CHECK_R2_ALPHA_REPLY.md
?? scripts/beta_relay.py
?? tests/test_beta_relay.py
?? tests/test_gate_prod_shape_scoping.py
```

| Question | Position |
|---|---|
| Q1 | OBJECTION |

## Q1 — Yes; the qualified verdict is accurate, but §7 still omits persistent-change paths

The R1 ignored-file fix is verified. The snapshot enumerates ignored paths, applies the exclusions, records small-file hashes or large-file size/mtime, and compares those records (`scripts/beta_relay.py:217`, `scripts/beta_relay.py:224`, `scripts/beta_relay.py:256`, `scripts/beta_relay.py:303`). The existing RL-Q/RL-R/RL-S/RL-T checks and MUT-6/MUT-7 all passed in this review; their cases and mutations are defined at `tests/test_beta_relay.py:267`, `tests/test_beta_relay.py:274`, `tests/test_beta_relay.py:282`, `tests/test_beta_relay.py:295`, and `tests/test_beta_relay.py:340`.

The revised exit-0 wording, “no change was detected by the snapshot,” matches the comparison and verdict selection (`docs/exchange/RELAY_PROTOCOL.md:77`; `scripts/beta_relay.py:495`, `scripts/beta_relay.py:514`, `scripts/beta_relay.py:524`). Already acknowledged gaps remain possible, including writes to excluded ignored paths (`docs/exchange/RELAY_PROTOCOL.md:119`); RL-S directly verified that case (`tests/test_beta_relay.py:282`). Those disclosed limitations are statuses, not new defects.

However, **§7 is not a complete description of the remaining gaps, and §3's coverage claims need further qualification.** Four additional fake-child probes each returned:

```text
code=0 verdict=REVIEW RECEIVED violations=[]
```

Each changed object remained changed after return. Each probe used a separate temporary repository, a successful fake child, and the existing valid reply. None depended on concurrent writes or changing something back.

### 1. Retargeting an untracked, non-ignored symlink

**Verified:** before the relay snapshot, create `target-a.txt` and `target-b.txt` with identical bytes, and an untracked `pointer -> target-a.txt`. The fake child replaces only the link with `pointer -> target-b.txt`. The changed link survives, but the relay reports exit 0.

Untracked entries use `p.is_file()` and `_sha_file(p)`, whose ordinary `open()` follows a symlink; the link text/type is not recorded (`scripts/beta_relay.py:188`, `scripts/beta_relay.py:249`, `scripts/beta_relay.py:254`). In contrast, the ignored-file recorder explicitly records `os.readlink(p)` (`scripts/beta_relay.py:225`).

This is a persistent symlink-target change, not a permission change or an empty-directory change. §7 does not list it (`docs/exchange/RELAY_PROTOCOL.md:119`). The symlink-target statement in §3 appears within the ignored-file bullet; it should not be read as protecting untracked links (`docs/exchange/RELAY_PROTOCOL.md:55`, `docs/exchange/RELAY_PROTOCOL.md:58`).

Fixture: `/tmp/beta_relay_test_5d8lwlky/case01/repo`.

### 2. Editing a tracked file already marked assume-unchanged

**Verified:** before the first snapshot, set `git update-index --assume-unchanged tracked.py` in the fixture. The fake child changes `tracked.py` from `x = 1` to `x = 999`. The content remains changed, with exit 0. The child does not change the index or another `.git/` file.

The tracked-file hash set is selected entirely by `git diff HEAD --name-only`; the other tracked checks also depend on Git status/diff (`scripts/beta_relay.py:238`, `scripts/beta_relay.py:240`, `scripts/beta_relay.py:243`). In this fixture Git did not report the changed file, so it never entered that hash set (`scripts/beta_relay.py:247`).

The statement “sha256 of every tracked file that differs from HEAD” therefore overstates the implementation (`docs/exchange/RELAY_PROTOCOL.md:54`). §7's `.git/` limitation describes changes inside `.git/`; this probe instead changes an ordinary tracked source file while the pre-existing index flag stays in place (`docs/exchange/RELAY_PROTOCOL.md:125`).

Fixture: `/tmp/beta_relay_test_5d8lwlky/case02/repo`.

### 3. Editing a file inside an untracked nested Git repository

**Verified:** before the first snapshot, initialize an untracked `nested/` repository and create `nested/payload.txt` containing `before`. The fake child changes that file to `after`. The content remains changed, with exit 0.

In this fixture, the outer Git enumeration yields `nested/` rather than its descendant files. The untracked recorder stores the directory as `<not-a-file>` and never walks its contents (`scripts/beta_relay.py:250`, `scripts/beta_relay.py:254`). Comparing that unchanged marker cannot expose the payload edit (`scripts/beta_relay.py:298`).

This is an edit to an existing nonempty repository's ordinary payload file, outside its `.git/`. It is not covered by §7's empty-directory or `.git/` entries (`docs/exchange/RELAY_PROTOCOL.md:124`, `docs/exchange/RELAY_PROTOCOL.md:125`). The untracked-file coverage wording should describe this enumeration limit (`docs/exchange/RELAY_PROTOCOL.md:55`).

Fixture: `/tmp/beta_relay_test_5d8lwlky/case03/repo`.

### 4. Changing an existing halt file's contents

**Verified on a temporary fixture only:** create the fixture halt file with `before` before the first snapshot; have the fake child replace its contents with `after`. Exit 0; contents remain changed. The live `/tmp/agent_halt` was not created or written.

Only `p.exists()` is recorded and compared (`scripts/beta_relay.py:262`, `scripts/beta_relay.py:308`). §3 accurately says the check is existence-only (`docs/exchange/RELAY_PROTOCOL.md:59`), but §7's outside-repository exclusion expressly excepts the halt file without stating that its contents are unchecked (`docs/exchange/RELAY_PROTOCOL.md:128`). Add that distinction to §7; this case does not require expanding the intended existence-only check.

Fixture: `/tmp/beta_relay_test_5d8lwlky/case04/repo`.

### Requested disposition

At minimum, qualify §3 and add these gaps to §7. If detecting these changes is intended, extend the corresponding records/enumeration and add fixture gates instead. This objection concerns incomplete coverage disclosure; it does not demand replacing the documented snapshot approach with comprehensive prevention. Michael decides the disposition.

## Evidence and scope

Read `scripts/beta_relay.py`, `docs/exchange/RELAY_PROTOCOL.md`, the supporting test suite, the R1 review, and the R2 request. Before recording the additional gaps, searched `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*_AUDIT.md`, and `docs/TODO_CONSOLIDATED.md`, plus the exchange and relay reports, for relay, symlink, assume-unchanged, skip-worktree, nested-repository, halt-content, and filemode records. No prior record of these four relay-specific gaps was found in that search.

Ran:

```bash
source ~/venvs/torch/bin/activate
PYTHONDONTWRITEBYTECODE=1 python3 -u tests/test_beta_relay.py
```

Observed **30/30 PASS**, `COMPLETION SENTINEL: PASS`, exit 0. That includes the ignored-file gates, all seven mutation checks, and HYG-LIVE. The suite's live-state comparison is at `tests/test_beta_relay.py:396`; it reported unchanged observed halt/ledger state. The suite's initial fixture root was `/tmp/beta_relay_test_pmu3aeuc`, removed by its own cleanup (`tests/test_beta_relay.py:402`).

Separately ran a throwaway Python heredoc in the same activated environment with `PYTHONDONTWRITEBYTECODE=1`. It loaded the helpers through `runpy.run_path('tests/test_beta_relay.py')`, loaded the current relay source through `load_module`, and used `Fx.run` with `GOOD` plus the following child mutations. These helpers write fixtures under `/tmp` and inject a fake launcher (`tests/test_beta_relay.py:29`, `tests/test_beta_relay.py:54`, `tests/test_beta_relay.py:74`, `tests/test_beta_relay.py:101`, `tests/test_beta_relay.py:112`).

```python
# Probe 1: equal-byte targets and initial link established before run_relay.
p = pathlib.Path('pointer'); p.unlink(); p.symlink_to('target-b.txt')

# Probe 2: assume-unchanged flag established before run_relay.
pathlib.Path('tracked.py').write_text('x = 999\n')

# Probe 3: nested repository and initial payload established before run_relay.
pathlib.Path('nested/payload.txt').write_text('after\n')

# Probe 4: existing fixture halt file established before run_relay.
pathlib.Path(sys.argv[2]).write_text('after\n')
```

For each probe, asserted that the inspected target/content differed before versus after, `r.code == 0`, and `r.violations == []`. All four assertions passed. Fixtures remain under `/tmp/beta_relay_test_5d8lwlky` for inspection. These are verified behavior in controlled fixtures; their presence in the live working tree is not asserted.

No real Codex child, live relay, pipeline, fleet run, or worker was launched. No rigs were contacted. Real CLI acceptance of the new flags and sandbox enforcement were not exercised; RL-U checks argv construction only (`tests/test_beta_relay.py:302`). All review-created repository content is confined to this response.

Source hashes matched the request at initial read and the pre-write recheck:

```text
f8bd5122ed573b77b12e0e42bed8dceae1e5938eb5924e008f8cfa91763c0606  scripts/beta_relay.py
7dc72e62cc9027deb2e0316ca168ad26c2834e48986313120998d4ea8ff584fc  docs/exchange/RELAY_PROTOCOL.md
06eac146348c7be084895c285b29303156433bada0011a2d41a9b3d7c20c6275  tests/test_beta_relay.py
```

## Additional findings

None outside Q1's scope; the four numbered gaps above answer Q1.
