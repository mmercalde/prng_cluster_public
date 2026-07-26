# TEAM_ALPHA_REVIEW_S172_PHASE5_D3_5.md — REV2 (completed review)

**Subject:** Team Alpha completed review of the D3.5 implementation
**Spec:** `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3_5.md` REV3.1 + the Team
Beta harness-expiry ruling (D3/D3.0 migration gates retired, F18 strengthened)
**Base:** HEAD `70cd6f0`
**Supersedes:** REV1, which deferred artifact review pending the §1 spec ruling.
**Verdict: APPROVED — recommend Team Beta review for commit. No correction round
required. Two commit-scope items need an explicit Beta decision (§6); neither is
an implementation defect.**

---

## 1. Beta's required byte-identity proof — satisfied twice over

The three components the retired gates still certify are **byte-identical** to
`70cd6f0`:

```text
utils/canonical_arrays.py          e3033e1ee523...   IDENTICAL
utils/prng_encoding.py             cfd86900bcc6...   IDENTICAL
convert_survivors_to_binary.py     23ee88c847db...   IDENTICAL
```

Stronger than the hashes: **none of the three appears in the diff as a modified
file.** The 11 textual matches are prose inside the harness retirement notes.
They cannot have changed.

## 2. Diff scope

```text
M  window_optimizer_integration_final.py            integration + swallow + dedup retirement
M  tests/test_s172_phase5_d3_columnizer.py          migration-gate retirement
M  tests/test_s172_phase5_d3_0_encoding_contract.py migration-gate retirement
M  tests/test_s172_phase4_coordinator.py            gate-22 registration
D  bidirectional_survivors_all.npz                  Ruling F -- see 6.1
D  bidirectional_survivors_binary.npz               Ruling F -- see 6.1
?? utils/run_finalizer.py                           new
?? tests/test_s172_phase5_d3_5_finalizer.py         new
```

Matches Beta's scope amendment exactly, plus the two Ruling-F deletions. Nothing
in the must-not-modify list was touched.

## 3. The harness retirement did not weaken live assertions

**Beta's prohibition was "do not weaken the remaining standalone-writer
assertions." Verified satisfied:** `run_convert_writer` is retained in every
check that had it — E7's and E9's loops simply lost their `("inline", ...)`
tuple, and C8 lost only its `legacy_inline` branch. Every assertion *about the
standalone writer* is unchanged.

What changed in character: **parity** assertions (convert vs inline) were
retired and canonical-module assertions added where the check concerns the
encoding contract rather than one writer — E1-E3 gained
`encode_prng_type(...) == <literal id>`, E4-E6 gained the canonical raise.

**Honest note for the record:** E4-E6's new canonical raise is largely
**redundant with Phase 0's own `tests/test_prng_encoding.py`**, so the *net*
coverage of those three checks decreased by one writer. That is unavoidable once
the writer is deleted — there is no third writer to substitute — not a defect in
the edit. The alternative (keeping dead production code alive to feed an
assertion) is strictly worse, consistent with Beta's ruling.

Prohibition sweep independently confirmed: no executable reference to
`_survivors_to_arrays`, `_inline_survivors_to_arrays`, `run_inline_writer` or
`load_inline_writer` survives in either harness or the integration file.

## 4. Mechanical verification (Team Alpha sandbox, pristine `70cd6f0`)

Independent reproduction on the migrated tree:

```text
D3.5   51/51 green
D3     10/10 green   (revised harness)
D3.0   10/10 green   (revised harness)
```

Claude Code additionally reported the pre-edit baseline green, the historical
attribution isolation (both **original** harnesses 10/10 with only
`window_optimizer_integration_final.py` restored from `70cd6f0`, proving the
earlier reds had exactly one cause), and 27 mutants red with attribution.

**Team Alpha independent mutants, targeting the newest and least-reviewed code —
the chain-tip authentication:**

| mutant | result |
|---|---|
| **MT1** drop the tip sidecar-hash comparison (7.1b step 5) | **killed** — F48, F49, F51 red |
| **MT2** accept absolute pointer targets (remove the `isabs` guard) | **survives — proven redundant, not a gap** |
| **MT3** drop the sidecar hash from the final generation directory name | **killed** — F5, F6, F12, F18 red |

**MT2 resolved rather than reported as a gap.**
`PurePosixPath('/generations/foo').parts` is `('/', 'generations', 'foo')` —
length 3 — so step 2's direct-child check already rejects every absolute target.
The `isabs` guard is redundant defence-in-depth with a clearer error message; its
removal is behaviourally inert, which is why no gate moves. Not a defect, not a
missing gate.

MT3 is the informative kill: removing the hash from the directory name breaks
`_parse_generation_dir_name`, cascading through the whole prior-loading chain.
The hash-bound name is load-bearing, exactly as REV3.1 [D1] intends.

## 5. Implementation faithfulness

- **7.1b chain-tip validation**: all six steps, in order. Two hardenings
  **beyond** spec: absolute-target rejection and
  `os.path.islink(generation_dir)` rejection, so a symlinked generation
  directory cannot masquerade as a real one. Both endorsed.
- **Sidecar**: `sidecar_sha256` deliberately **absent** from the key set
  (`utils/run_finalizer.py:142-157` documents why), living in
  `RunArtifactResult` and the child's `parent_sidecar_sha256` — [C1] satisfied.
- **Publication**: `final_dir = generations_dir / f"{generation_id}--{sidecar_sha256}"`
  (`:1361`) — the name cannot be formed before the hash exists, which
  structurally enforces the REV3.1 ordering (rename after hashing).
- **`PublicationDurabilityError`** is a distinct type documented against [D4]
  (`:232`), so a post-commit fsync failure cannot be reported as "nothing
  published".
- **Error taxonomy**: everything derives from `RunFinalizerError(RuntimeError)`,
  not `ValueError` — deliberate, so a fail-closed rejection cannot be swallowed
  by an upstream `except ValueError` and mistaken for a fallback candidate. This
  applies the D3.0 tagged-error lesson without being asked. Endorsed.
- **No subprocess** in the module; git state arrives as arguments, which is why
  the frozen signature takes `repository_commit` / `repository_tree_clean`.
- **No coverage-database access** — no reference to `prng_analysis.db`,
  `exhaustive_progress` or sqlite anywhere in the module.
- **11 satisfied**: the `:2004` broad swallow and its
  `convert_survivors_to_binary.py` subprocess fallback are deleted; the
  finalizer call sits outside every `try/except`.
- **10 satisfied**: `deduplicate_survivors` removed outright;
  `bidirectional_survivors.json` demoted to a post-success summary.

## 6. Two commit-scope items requiring an explicit Beta decision

Neither is an implementation defect; both are consequences of Ruling F the
disposition did not address.

### 6.1 The deleted accumulator artifacts were git-tracked

`git status` shows them as ` D`, not as untracked removals:

```text
 D bidirectional_survivors_all.npz
 D bidirectional_survivors_binary.npz
```

Ruling F's `rm -f` therefore deleted **tracked** files, and the D3.5 commit will
record their removal from version control. Team Alpha considers this correct and
desirable — a rejected artifact should not remain in the repository — but it
should be **explicit and intentional in the commit**, not an incidental side
effect. Requested: Beta's confirmation that the deletions belong in the D3.5
commit rather than a separate Ruling-F cleanup commit.

This also adds to the provenance record: the accumulator was version-controlled,
so its historical content remains recoverable from git history if forensic need
ever arises beyond the `archive/` copy.

### 6.2 `archive/` is untracked — does the forensic archive enter git?

```text
?? archive/
```

The Ruling-F disposition specified "archived, not deleted" and recorded the
hashes, but did not say whether the archive is an on-disk artifact or a committed
one. It holds ~190 KB across four `.npz` files. Team Alpha's view: **leave it
untracked**, since git history already preserves the tracked originals (6.1) and
committing rejected data invites future confusion about its status. Requested:
Beta's ruling either way, so the decision is recorded rather than defaulted.

## 7. Recommendation

Approve for commit. Suggested scope: `utils/run_finalizer.py`,
`tests/test_s172_phase5_d3_5_finalizer.py`, the four modified files, the two
Ruling-F deletions (pending 6.1), the REV3.1 brief, this memo, and the session
changelog — which Claude Code wrote as
`docs/SESSION_CHANGELOG_20260725_S179.md` and should be renamed to
`SESSION_CHANGELOG_20260725_PHASE5_D3_5.md` per the phase convention.

— Team Alpha (Claude), 2026-07-25
