# X-OWNER-LAND · R1 · BETA REVIEW

## Prior record

**Thread / round:** X-OWNER-LAND / R1. **Date:** 2026-11-14 (PST).
**Class:** Tool, AFTER-COMMIT; review before first real use. **Base read:**
`472cf94c0f44f4a9c0439dd61a7bb7b0dcae3f33` (HEAD throughout this review).
**`git status --porcelain` on entry:** empty output. Empty again immediately before writing this review.
Beta records positions; Michael decides. This is not an approval.

**Searches actually run:**

- `rg -n -i 'owner_land|landing script|git_push\.sh|push_to_claude|commit list|no.touch|hand.landed|dual.push|inbox'`
  over `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*`, `docs/*_AUDIT.md`,
  `docs/TODO_CONSOLIDATED.md`, `docs/exchange/`, `docs/BACKLOG.md`, `docs/LEADS.md`,
  `CLAUDE.md` and `AGENTS.md`; then over `docs/PROJECT_FILE_CATALOG.md`,
  `docs/CLAUDE_CODE_REPORT_*`, `docs/CLAUDE_CODE_INSTRUCTIONS_*` and
  `docs/TFM_PROJECT_FACTS_SKILL.md`.
- Before recording additional defects, searched all `docs/` Markdown for
  `owner_land|stale.plan|cached.hash|staged.*(rollback|restore)|symlink.*(land|dest)|landing.*(hash|race)|uncommitted.*(overwrit|modify)|unpushed`.
  Broadened with `cached|rehash|sha256|index|staged|self|landing|commit` over rulings,
  proposals, chapter audits, TODO and exchange records. Also searched the owner-land
  documents and earlier relay reviews for symlink, index flags and hook records.
- Read both code files in full; the brief and report; `AGENTS.md`; relevant `CLAUDE.md`
  rules; the relay protocol; completion-plan collision rules; Findings isolation rules;
  binding-addendum §B; delegated-fleet proposal §§2–4; and handoff §§2, 4, 6–7.
  Inspected launch-harness admission/pre-dispatch/WATCHER lines, the parity tuple,
  ignore rules, both legacy push scripts and local deny entries.
- Ran `git status --porcelain`, `git rev-parse HEAD`, `git log -5 --oneline`,
  `sha256sum` on both code files, local remote-tracking divergence counts,
  `git ls-files --stage`, `git ls-files -v`, and `git check-ignore` for local settings.
  Enumerated agent environment **names only**.

**Governing records and labels:**

| Record / item | Label and effect |
|---|---|
| `AGENTS.md` §§2–5; `CLAUDE.md:82-88`; completion plan `:136-142` | ALREADY RECORDED: owner commits and dual-pushes; Beta reviews within its write boundary. |
| Findings proposal `:24-34`; binding addendum `:46-56` | ALREADY RECORDED: isolation, full no-touch list, landing between launches, repeated process checks. |
| `docs/exchange/RELAY_PROTOCOL.md:28-29`, `:108-136` | ALREADY RECORDED: exclusive relay interval, owner decisions and ledger provenance. |
| Owner-land brief §1; report §§2, 6–7 | ALREADY RECORDED: intended guards, wider-list decision, agent-guard limits, legacy scripts and unavailable real-use evidence. |
| Request Q2–Q6 | ALREADY RECORDED in this request; the reproductions below confirm status rather than claim discovery. |
| Delegated-fleet proposal `:44-48`, `:55-63`, `:80-82` | ALREADY RECORDED governing constraints omitted from the request's governing-record table; see Finding 1. |
| Handoff `:110-113`, `:161-170` | ALREADY RECORDED: narrower deny entry, longer list, refusal stubs and expected-file landings; omitted sections of a document the request cites. These are not evidence of a new owner decision on Q8–Q10. |
| Cached exchange hashes during the prompt; loss of a pre-existing index on rollback; ability to land the script itself | NEW measurements in this review. No owner-land record of these defects was found in the searches above. |
| Any proposed alteration to an owner ruling | None proposed here. Q8–Q10 remain recommendations for Michael. |

**Not searched / not exercised:** ser8, rigs, Proxmox, private GitHub contents, external
project files, shell aliases, real remote tips over the network, live run databases or
staging contents. No production CLI, pipeline, worker, fleet launcher or relay was run.
Local settings were read, not edited. No installed package was changed.

## Finding 1 — governing record omitted from the request (Q12)

The request's search terms find the delegated-fleet proposal's introductory dual-push
reference, but its governing-record table misses the provisions relevant to Q6:
`docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:55-59` explicitly includes both standalone
prelaunch gates and `scripts/launch_fleet_manual.sh`, with committed-source requirements;
`:81-82` forbids tracked edits during a run session and gate weakening. Its `:44-48`
requires enforcing controls rather than treating an instruction as a control. Add these
sections to the prior record. This is a record omission, not a newly discovered fleet defect
or permission for Beta to launch anything.

Also carry forward handoff `docs/TFM_SESSION_HANDOFF_20261113.md:110-113` and `:163-170`:
they already record the exact narrow deny/list/stub recommendations and expected-file
landing discipline. The request cites this document's other sections.

## Positions

| Question | Position | Severity |
|---|---|---|
| Q1 | OBJECTION | LIVE — exchange files can change during confirmation and be committed under their old displayed hashes, and rollback can discard existing staged content. |
| Q2 | OBJECTION | LIVE — ordinary unpushed work, including the script's own partial-push outcome, can be sent without appearing in the landing plan. |
| Q3 | OBJECTION | LIVE — a normal uncommitted edit to a planned modify destination is silently overwritten on success. |
| Q4 | OBJECTION | LIVE — exchange mode can publish a changed request or reply whose bytes disagree with the relay's recorded provenance. |
| Q5 | SUGGESTION | — |
| Q6 | OBJECTION | LIVE — the existing shell launch harness can be active after admission and before WATCHER while the landing process guard reports no match. |
| Q7 | NO OBJECTION | — |
| Q8 | SUGGESTION | — |
| Q9 | SUGGESTION | — |
| Q10 | SUGGESTION | — |
| Q11 | OBJECTION | LIVE — the committed script lacks the history and launch-interval checks needed for the proposed first landing. |
| Q12 | SUGGESTION | — |

## Evidence run and limits

Activated `source ~/venvs/torch/bin/activate`. Created an independent local clone with
`git clone --local --no-hardlinks` under `/tmp/beta-owner-land-r1.BzwGzE/checkout`;
no linked worktree or shared refs were created in the reviewed checkout. Ran there:

```
PYTHONDONTWRITEBYTECODE=1 python3 -u tests/test_owner_land.py
```

**33/33 PASS; COMPLETION SENTINEL: PASS.** This includes the existing mutants and
hygiene gate. The first attempted suite command preceded completion of the clone and
exited 2 with “can't open file”; it executed no tests. The completed run above is the
reported result.

Additional throwaway checks used the suite's existing `Fx`, injected dependencies and
module loader (`tests/test_owner_land.py:56-63`, `:83-115`, `:151-167`), never the
production CLI. Every test commit/push was confined to fixture repositories and their
local bare remotes. Reproducer and results remain at:

- `/tmp/beta-owner-land-r1.BzwGzE/review_checks.py`
- `/tmp/beta-owner-land-r1.BzwGzE/review-check-results.json`
- `/tmp/beta-owner-land-r1.BzwGzE/existing-suite.log`

| Check | Observed result |
|---|---|
| Prior local commit, then normal manifest | Exit 0; origin advanced by two commits; `earlier.md` was pushed but never printed. |
| Dirty planned `README.md` | Exit 0; owner's uncommitted bytes replaced by incoming `readme v2`. |
| Exchange reply edited after its valid sha256 was recorded in ledger | Exit 0; committed reply hash differed from the ledger. |
| Exchange reply edited by injected confirmation reader | Exit 0; printed hash remained the old reply's hash, while the changed reply was committed and pushed. |
| Planned file had distinct index/worktree contents; pre-commit hook failed | Exit 5; worktree bytes restored, but original index bytes replaced with HEAD's `readme v1`. |
| Ignored new JSON destination | Prompt called once; exit 5 from `git add`; destination removed by rollback. |
| Shell harness/fleet argument lists | Both unmatched; direct Python coordinator-suite form matched; pytest form unmatched. |
| Manifest modifying committed `scripts/owner_land.py`, with real fixture `self_path` | Exit 0; replacement script committed and pushed. |

The exchange checks installed full, correct request/reply sha256 values in the fixture
ledger **before** changing the reply; they do not rely on the suite's short placeholder
hashes (`tests/test_owner_land.py:36-40`).

**UNAVAILABLE:** real `ssh -t`, human use of the real `input()`, actual GitHub pushes,
process discovery against a live relay/run, and an atomic exclusion test with a real
launcher. Those are not passes. The additional checks measure injected fixture cases;
operational consequences below are inferences from those measurements and the cited source.

## Reasoning by question

### Q1 — brief coverage, second pass and rollback

Most listed checks are present in the shared path (`scripts/owner_land.py:542-554`),
confirmation is mandatory (`:586-593`), named paths are added and the staged **path set**
is checked (`:624-629`), and normal failed-copy/commit cases restore copied worktree
bytes and modes (`:603-623`, `:630-652`). The passing existing suite supports these
covered cases; it is not evidence that all states are covered.

Two gaps prevent an unqualified “in every mode”:

1. **Exchange hashes are cached.** The plan records `sha256_file` once (`:301-303`).
   `check_entries` compares the saved `e.sha`, not a new read of exchange bytes
   (`:499-510`). The second pass reuses the same plan (`:595`); exchange entries then skip
   copying and its post-copy hash check (`:606-608`, `:619-623`). `git add` stages whatever
   bytes exist then (`:624-629`). The fixture printed hash `452b7b64…`, committed
   `996ebd55…`, and returned 0. Re-read planned bytes on both passes and verify staged
   content against the approved hashes before commit. Add a prompt-change exchange gate.
   For copied manifest/worktree sources, the post-copy check currently prevents this
   particular stale-source case from successfully committing (`:622-623`), though the
   refusal comes after confirmation.
2. **Rollback does not restore a pre-existing index.** Planned dirty paths are admitted
   (`:438-442`). The script backs up worktree bytes/modes only (`:610`), then resets
   planned index entries to HEAD (`:635`). A staged version different from both HEAD and
   worktree is lost from the index even when the worktree is correctly restored
   (`:636-643`). Simplest fix: refuse any pre-existing staged change, in every mode.
   Alternatively preserve and restore the exact prior index entries. Add a staged-state
   rollback gate, including exchange mode; the existing rollback fixtures begin without
   such staged changes (`tests/test_owner_land.py:391-412`).

The demonstrated **unlisted file reaching a push** is Q2's earlier commit. The demonstrated
**unshown content reaching the new commit** is the changed exchange reply above. The path-set
check alone proves neither historical push scope nor byte identity (`scripts/owner_land.py:626-629`,
`:657`). I did not demonstrate a new, unlisted path entering the newly created commit under the
normal fixture configuration.

### Q2 — remote history

Agree with Alpha: refuse unless HEAD equals both remote-tracking `main` tips, and refuse
missing tips, before confirmation and on recheck. Checking only remote names
(`scripts/owner_land.py:524-526`) does not constrain the history sent by `push … main`
(`:657`). The script's own partial-push result leaves exactly the kind of unequal state
that must be finished before the next landing (`:658-664`). Do not repair or push earlier
work automatically.

**Local status, not a defect finding:** this checkout's HEAD equalled both `origin/main`
and `public/main` as stored locally: divergence counts `0 0` for each. No fetch was made;
this does not verify the actual remote tips. Alpha's “as last fetched” check limits
accidental outgoing history; it cannot promise a network push will succeed. Retain the
existing explicit partial-push handling (`:658-664`).

### Q3 — dirty modify destinations

Agree. In manifest/worktree modes, require both the destination's worktree content and
its index state to agree with HEAD before replacing it. Compare file type/mode as well
where Git tracks them. Source-side `modify` validation only establishes destination
existence/tracking and that incoming bytes differ from HEAD
(`scripts/owner_land.py:494-510`); `check_dirty` exempts the planned destination
(`:438-442`). Successful `copyfile` overwrites it (`:619`), and its saved prior version
is used only on failure (`:630-643`). The fixture confirmed silent loss on success.
Exchange mode intentionally lands in-place edits, so apply the Q1 index safeguard there
rather than requiring every exchange worktree byte to equal HEAD.

### Q4 — relay hashes

Agree before first exchange use. The code reads thread/round/verdict, not the request/reply
hash columns (`scripts/owner_land.py:93`, `:306-324`). The relay's provenance requirement
is recorded at `docs/exchange/RELAY_PROTOCOL.md:125-136`. Resolve the exact round's request
and reply unambiguously, verify both complete sha256 values, and refuse missing, malformed,
ambiguous or mismatched evidence. A request already committed at HEAD still needs its hash
checked. Extra same-thread files selected by `:297-300` need their own approved plan hashes;
the latest ledger row does not certify every such file.

Keep this distinct from Q1: a matching ledger at plan creation does not protect against
bytes changing while confirmation waits. Both checks are needed. Do not treat unsuccessful
relay verdicts as agreement; their disposition remains Michael's under
`docs/exchange/RELAY_PROTOCOL.md:95-103`, `:115`.

### Q5 — ignored destinations

Adopt the pre-confirmation check. This is a usability suggestion, not a demonstrated
incorrect commit: the ignored-file fixture returned 5 and removed the copied file.
The cause is explicit-path `git add` (`scripts/owner_land.py:624-625`) followed by existing
rollback (`:630-652`), with this repository's `*.json` rule (`.gitignore:41`). Check new
untracked destinations using Git's actual ignore rules, including negations; do not add
`-f` automatically or mistake a tracked ignored filename for a new-file refusal.

### Q6 — process list

Agree with adding **both** shell launcher names and correcting the pytest example.
Recognize a launcher run directly and as a shell's script argument; simply adding names
to `label()` is insufficient for `bash gate12_launch.sh`, because the current classifier
returns immediately for non-Python executables (`scripts/owner_land.py:405-409`). Keep
shell `-c` text and editor arguments out of substring matching (`:388-390`;
`tests/test_owner_land.py:364-373`). Add gates for both direct and shell invocation forms.

The relevant current interval is concrete: admission (`gate12_launch.sh:133`), another
clean-tree assertion (`:246`), and WATCHER creation much later (`:377`). The intervening
shell is the process the existing classifier misses. A clean commit can change HEAD in
that interval; cleanliness alone is not an unchanged-HEAD proof.

**Otherwise sufficient only within a stated scope:** the existing classifier covers the
named relay/coordinator/WATCHER/optimizer programs under its ordinary direct/Python/module
forms (`scripts/owner_land.py:391-424`;
`tests/test_owner_land.py:349-361`). I found no further required program-name omission for
those forms. That is not a guarantee of all run activity: pytest, wrappers, detached
workers, and port occupancy are not equivalent to those forms. The report already records
that port 5700 is not checked (`docs/CLAUDE_CODE_REPORT_OWNER_LAND.md:254-258`). Do not
claim that two `/proc` samples provide atomic exclusion: enumeration and copying are
separate operations (`scripts/owner_land.py:110-122`, `:595-619`). Retain the operational
rule that no new run starts during a landing. If automatic mutual exclusion is required,
a shared mechanism honored by both paths needs separate launch-path review; this review
does not authorize a launch-script edit.

### Q7 — observed Codex environment names

Names observed in this session's tool shell:

```
CODEX_APPLY_PATCH_PRESERVE_LINE_ENDINGS
CODEX_CI
CODEX_MANAGED_BY_NPM
CODEX_MANAGED_PACKAGE_ROOT
CODEX_SANDBOX_NETWORK_DISABLED
CODEX_SESSION_ID
CODEX_THREAD_ID
CODEX_TOOL_CALL_ID
CODEX_VERSION
```

Yes: `CODEX_` covers every Codex-prefixed name observed here; the guard checks presence,
not values (`scripts/owner_land.py:97-98`, `:357-363`). This shell also inherited Claude
markers; inheritance does not establish which program originally set them. This is a
measurement of this session, not a promise that every Codex deployment exports these names.

### Q8 — full hand-landed set

Recommend the full Findings list plus root `gate12_launch.sh`, while retaining the current
parity/self-protection entries. The present check implements the brief's shorter set
(`scripts/owner_land.py:84-87`, `:461-476`), so the mismatch is already recorded policy
scope, not a newly found implementation failure. Use the exact enumerated optimizer
filenames and the `scripts/gate12_*` family from
`docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md:28`; include the addendum's root
harness (`docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md:48-49`).
Michael decides whether that agent-edit list also becomes the tool's refusal list.
See Finding 3 for the script's own path.

### Q9 — rule, deny entry and limits

Support the owner-only rule and the narrower `Bash(*scripts/owner_land.py*)` entry. Clarify
the test exception: permitted fixture tests call the injected API
(`tests/test_owner_land.py:160-167`); the prohibition on “anything that invokes it” should
not accidentally outlaw that stated exception. Beta must still run the suite outside the
reviewed checkout because of its write boundary.

The report's §7.2 is **not complete as an access-control claim**. The dependencies are
injectable (`scripts/owner_land.py:125-136`), and `run()` is callable independently of
`main()`'s defaults (`:702-727`), as the authorized test harness itself demonstrates.
A textual Bash deny pattern does not establish coverage of imports, renamed copies,
relative invocation from `scripts/`, or other execution tools. These are limitations to
document, not routes Beta attempted against a real checkout. The TTY/environment/word
checks are useful accidental-use guards, not authentication of Michael
(`:350-363`, `:586-593`). An absolute “impossible for an agent” claim is not established.

**Local status:** `.claude/settings.json:3-9` currently lists the direct git/pipeline deny
patterns and Git ask rule, without an owner-land entry. No settings were changed. No
claim is made that the proposed wildcard was tested by Claude Code's permission engine.
The local `.claude/settings.local.json` is untracked and ignored by `.gitignore:41`;
`git status --porcelain` was empty. Its mere existence does not cause `DIRTY-PATH` here.

### Q10 — old scripts

Prefer refusal stubs over deletion: print that Michael uses `scripts/owner_land.py`, exit
nonzero, and do **not** forward to it. This leaves an intelligible failure for old commands
and references. The unsafe bodies are directly verified at `git_push.sh:19-21` and
`push_to_claude.sh:3-5`. Deletion would also remove those execution paths; this is an owner
choice, not a correctness objection between the two options.

### Q11 — first real use

**Do not use the script as committed.** Of Q2–Q6, fix **Q2 and Q6 before even the proposed
two-new-document manifest landing**. Q2 applies to every push (`scripts/owner_land.py:657`);
Q6 applies throughout launch preparation (`gate12_launch.sh:133`, `:246`, `:377`).

The remaining requirements are mode-specific:

- **Q3 before any manifest/worktree `modify` entry.** A strictly new-file first manifest
  does not exercise the overwrite path (`scripts/owner_land.py:489-498`, `:619`).
- **Q4 and Q1's fresh exchange-hash check before exchange mode.** Neither is exercised by
  a strictly new-file manifest (`:301-303`, `:606-608`).
- **Q1's index safeguard before accepting pre-staged planned content in any mode.** A
  first manifest should have an empty index (`:624-635`).
- **Q5 is recommended, not a prerequisite for two non-ignored `.md` destinations.** Its
  measured failure rolled back (`:630-652`).

Prefer one correction batch with regression gates for all of these, rather than relying
on remembered mode restrictions. After that, two new document files are a reasonable
bounded first use **by Michael**, with the real prompt and both real push results recorded.
The currently unavailable real-terminal/remote checks can be earned during that use;
they need not be claimed as pre-existing passes. No Beta launch, commit or push is requested.

### Q12 — prior record

Add Finding 1's delegated-fleet provisions and the omitted handoff sections. The searches
found no further owner-land ruling that decides Q8–Q10 for Michael. The narrower deny
entry, wider no-touch list and refusal stubs are already recorded recommendations.

## Additional findings

**2. NEW — rollback loses pre-existing staged content (LIVE; included in Q1).** The fixture
kept different staged and worktree versions of a planned file, then forced the existing
pre-commit failure. Exit 5 restored the worktree version but reset the index to HEAD.
The causal lines are `scripts/owner_land.py:610`, `:635-643`. Refuse pre-staged content or
restore its exact index entries. This remains relevant to exchange mode even after Q3
refuses dirty copied destinations.

**3. NEW — the script can land a replacement of itself.** The request's closing explanation
and report `docs/CLAUDE_CODE_REPORT_OWNER_LAND.md:107`, `:217-218` say self-updates must be
hand-landed. Verified otherwise: a manifest targeting `scripts/owner_land.py` with different
bytes returned 0 and pushed the replacement in the fixture, with `self_path` pointing to
the original committed fixture script. `check_self_identity` checks the running file
**before** copying (`scripts/owner_land.py:529-539`, `:595`); the path is absent from
`HAND_LANDED_FILES` (`:86`), and replacement happens later (`:619`, `:629`). Recommendation:
explicitly refuse `SELF_REL` as a destination if hand-landing self-updates is the intended
rule, and add a gate with a correctly committed starting script. The existing `G-SELF`
only tests a wrong running path (`tests/test_owner_land.py:486-489`). This was an explicitly
planned replacement, not an unlisted-file commit; it does not by itself block the bounded
document-only first use after Q2/Q6 are fixed.

**4. NEW — cached exchange plan hash (LIVE; included in Q1).** The confirmation reader
changed only a planned reply after a correctly hashed ledger and printed plan were built.
Exit 0 committed the changed bytes while displaying the old digest. Causal lines:
`scripts/owner_land.py:301-303`, `:499-510`, `:595`, `:606-608`, `:624-629`. This is distinct
from an already stale ledger at plan creation (Q4). Add a second-pass content check and
staged-content verification, with a dedicated prompt-change gate.

No reviewed source changed during this review. Both code sha256 values remained exactly
those in the request. The only repository file written by Beta is this review.
