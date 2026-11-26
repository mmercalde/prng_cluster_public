# Beta R1 on X-REGISTRY-REPAIR: positions and proposed responses (nothing changed yet)

Task: `logs/e2e_session/TASK_REG_BETA_SEND_20261125.txt` item 4. **Nothing has been changed in response to the review.** This note proposes; the lead and owner decide.

**What this note covers.** It points at Beta's file rather than replacing it (`RELAY_PROTOCOL.md` §5). Read the review itself for Beta's words and evidence.

| | |
|---|---|
| **Review file** | `docs/exchange/from_beta/X-REGISTRY-REPAIR_R1_BETA_REVIEW.md` (main tree, uncommitted), sha256 `760e356a9c4549919d8674b000cfa524984ec98bf5e2b843cf05c7cbfa1be163` |
| **Request** | `docs/exchange/to_beta/X-REGISTRY-REPAIR_R1_ALPHA_REVIEW_REQUEST_registry_parts2_3.md`, sha256 `2e8a6857295e18994ceb66012fd43588cf728eff3a20f507176dd9e789b7a282`; byte-identical to `docs/drafts/loop_v1/DRAFT_X-REGISTRY-REPAIR_R1_REQUEST.md` in this worktree |
| **Relay** | `VERDICT: REVIEW RECEIVED (exit 0)`, 419.5 s; log `logs/beta_relay/X-REGISTRY-REPAIR_R1.log` (main) and this worktree's `logs/registry_repair_land/beta_relay_R1.log` |
| **Ledger** | one row appended to `docs/exchange/LEDGER.md` (main, modified, uncommitted) |
| **Left for the owner route** | the three exchange files in the main tree, uncommitted, as the task says |

**Before sending:** `pgrep -af '[e]2e_walk_forward_selftest'` was empty (rc 1), and main was clean at `579dbb7`. The bracket form keeps the pattern from matching the shell that runs it (box rule 8). Nothing in this worktree or the main tree was edited while the relay ran.

## Positions

| Q | Beta's position | Severity |
|---|---|---|
| Q1: the 20 drops (part 1, landed) | **NO OBJECTION** | — |
| Q2: the seven specs | **NO OBJECTION** | — |
| Q3: defaults after the manifest declarations | **NO OBJECTION** | — |
| Q4: explicit `schema_check=fail` | **NO OBJECTION** | — |
| Q5: R10 and the tests | **SUGGESTION** | — |
| Prior-record Finding 1 | (a finding, not a position) | — |

**There is no `OBJECTION`, so nothing holds the commit** (`RELAY_PROTOCOL.md` §2.4). Nothing went to `DEFERRED.md`.

## Each point, with my proposed response

### Finding 1: governing records missing from the request's prior record

**Beta's point.** The request did not cite:
- `TASK_SWITCH_REGISTER_ADDENDUM_1.txt:1-4` (existing registry, registration distinct from reach, defaults unchanged);
- `docs/CLAUDE_CODE_INSTRUCTIONS_CHAIN_C_TRUTHFULNESS_HOTFIX.md:33-45, 54-57`, the direct source of the Step 5 reporting boundary, which bars adding the six diagnostic allowlist fields;
- `docs/TODO_SELFPLAY_AND_LLM_AUTONOMY.md` B3 (`:133-139`, `:158-163`).

The LL-3 declarations sit outside the six-field list.

**Proposed response: accept.**
- It is a citation gap, not a change in substance. The READY note did list the addendum (`:31-36`), but the request did not, and the Chain C hotfix instruction was not searched at all.
- I would add all three to the prior record of any next document on this subject, and to the READY note's "Prior record" before part 2 lands.
- It also confirms the boundary holds: none of the six allowlisted names is declared.

### Q1: NO OBJECTION

**Beta's points.** No correction under the original keys. Row 5 has a successor under its own script key (`generate_scorer_jobs.py --trials`). `--batch-size` is progress-only. `--dropout` is a single override. The kept-spec mismatches remain a status.

**Proposed response: record it.** Nothing to do. The 4 kept-spec mismatches stay an owner question (READY note, Questions 1).

### Q2: NO OBJECTION, with one concrete caution

**Beta's points:**
1. All seven specs match the parser.
2. LL-3's further restrictions live in the resolver. An allow-only request (`allow_target_columns_as_features` with no `exclude_features` list) routes, and is then refused by the resolver before fitting.
3. **"Use a JSON boolean for the WATCHER value."** The registry's BOOL validator accepts the *string* `"false"` (`parameter_registry.py:130-139`). WATCHER passes a non-bool value as `--flag value` (`agents/watcher_agent.py:2412-2428`), so the string `"false"` becomes `--allow-target-columns-as-features false`. A flag-only argparse option then rejects the extra token. That is a fail-closed stop, not a wrong run.

**Proposed response: record, no change to parts 2 and 3.**
- The manifest command already writes JSON `false` (`OWNER_MANIFEST_COMMANDS.txt` command 2), so the declared default is safe.
- **NEW for the backlog:** a mismatch between `validate_value` (accepts `'true'`/`'false'`/`'yes'`/`'1'` for BOOL) and WATCHER's builder (only a real `bool` is treated as a flag). It matters once an applier (G7) feeds validated values to WATCHER. The fix belongs there: coerce BOOL to a real `bool` before `params`. Alternatively, tighten `validate_value` for BOOL, but that changes shared code outside this ruling, so it is an owner question.
- I'd add one line to the READY note's caveats.

### Q3: NO OBJECTION

**Beta's points.** Beta ran both owner commands on copies from `579dbb7`: the bytes match, and the default argv is unchanged. Beyond argv:
- WATCHER's debug list of loaded keys grows;
- the switch checker's inventory sees the new keys;
- the step-scoped filter admits explicit overrides;
- no retry proposal is affected;
- no gate pins either manifest's key count or hash.

**Proposed response: record it.** It matches the READY note's "what changes at default: nothing in argv". The debug-log line growing is worth one line in the READY note.

### Q4: NO OBJECTION, with a condition

**Beta's points.** An explicit strict request is acceptable as a fail-closed option. "Any use of strict mode as a default or as a comparison requirement would need evidence from the intended arms, including A0."

**Proposed response: record the condition as a standing rule for LL-1.** `schema_check=fail` stays operator-chosen. It may not become a default, or a requirement of a walk-forward comparison, until it has been run against the intended arms, A0 included. I'd add that sentence to the READY note's LL-1 caveat and propose it to the lead for `docs/LEADS.md` / the handoff, not as a code change.

### Q5: SUGGESTION, three items

R10's refusals and the byte-for-byte regeneration were confirmed. So was R8's unchanged registry output without `--register`. The note text for omitted boolean defaults changed, and the report gained base metadata; both are stated in the READY note. Three gaps remain.

**5a. A duplicate registration passes all seven gates.**
- **What Beta found:** a second `schema_check` spec with only `off` allowed passed every gate. `A-LL-ADDONLY` builds a dict, so duplicate names collapse. `A-LL-CODE` reads the first match (`get_param` returns the first, `parameter_registry.py:200-218`). With the order reversed, `A-LL-CODE` fails, but `A-LL-ADDONLY` still passes.
- **Proposed response: adopt before part 2 lands.** It is inside my own files. Add a gate to `tests/test_registry_ll_register.py`: exactly one spec per normalised name per script, for every registry in the file, not only the LL ones. Add a mutant that appends a duplicate. The same check belongs in `switch_check.py` R10's post-patch check (`set(new_specs) - set(old_specs)` also collapses duplicates), with a refusal and a mutant.
- No registry content changes: the current file has no duplicates (Beta verified the 7/7 run).

**5b. `A-LL-MANIFEST` tests strings, not behaviour.**
- **What Beta found:** the gate finds WATCHER's skip rules as source strings. It would pass with a different owner command, or after a regression that keeps those strings. The real evidence was the separate argv probe.
- **Proposed response: adopt.** Move the argv comparison into the test.
  - The test would apply the two exact owner commands to temporary copies of the manifests, then build the Step 5 and Step 6 argv through the real `run_step` with the probe's stubs.
  - It would assert that the default argv is identical and that one non-default value per key reaches argv.
  - The stub set is the one `tests/fixtures/run4_routing_clean_control.py` already uses.
- Cost: it imports `agents.watcher_agent`, so it is heavier than the current gate. It needs the dataset-pointer files only through the stubs. And `argv_probe.py` moves from `data/` into the landing set, as `tests/fixtures/…`.

**5c. R10 accepts argument shapes the registry cannot represent.**
- **What Beta found:** with `chapter_13_triggers.py --steps` (`nargs`), R10 succeeded and produced an `INT` spec whose default `[3, 5, 6]` its own validator rejects. The scanner drops `nargs` (`switch_check.py:280-309`) and then assigns a scalar type (`:976-1007`).
- **Proposed response: adopt.**
  - Keep `nargs` (and `action` values other than `store`/`store_true`/`store_false`) in the parsed row.
  - Make R10 refuse such options, with a reason, until the registry can represent them.
  - Add a refusal gate and a mutant in `tests/test_switch_check.py`.
- R8's old blind patch path has the same weakness (it is how `switch_patches_20261123` would have typed list options). I would apply the refusal to R10 only, and list the R8 case as NEW.

**Is it in scope?** 5a, 5b and 5c are all inside part 2's own files (`tests/test_registry_ll_register.py`, `tests/test_switch_check.py`, `tests/mutate_e2e_grader.py`, `scripts/diagnostics/switch_check.py`). None touches the registry content, the manifests or pipeline code. The protocol says a SUGGESTION is adopted "only if it is inside the brief's stated scope" (§2.4). The task says change nothing yet, so this needs the lead's go. With it, the work would be:
1. the three fixes plus gates and mutants;
2. a re-run of the two suites and the mutant runner, on a fresh tree of `579dbb7` with part 2 copied in;
3. new sha256s for the changed test and checker files.

The registered registry file `bea4fcfe…` would not change.

## One thing for the owner: the checker's fixture suite makes a git commit

**Fact.** Beta marked the checker fixture suite `UNAVAILABLE`, partly because its fixture creates a Git commit.

`tests/test_switch_check.py:195-197` builds a throwaway fixture repository under `data/e2e_grader_selftest/<run>/fixture_<pid>/main`. It runs `git init`, `git add -A` and `git -c user.name=fixture … commit -q -m fixture`, then adds two worktrees of it. It has been like this since it landed (`76234eb`, 2026-11-24).

**I ran that suite today** in this worktree, in both check trees, and indirectly through the mutant runner's `sc` target (each mutant builds its own fixture). The count:
- **129 fixture repositories, one commit each, are in this worktree** (`find data/e2e_grader_selftest -type d -path '*fixture_*/main/.git' | wc -l` = 129).
- **About 92 more were in the two check trees**, now removed: two `sc` mutant runs of 13 baseline and 30 mutant gate processes each, which is 86, plus 6 suite runs. This figure is estimated from the run counts, not counted.

All are in new throwaway repositories under the ignored `data/`. None is in the TFM repository, and none was pushed.

**It still falls under the task rule** "agents never commit, anywhere, scratch clones included" if that rule covers a test fixture repository. I did not ask before running it, and I should have flagged it then.

**Question for the owner.** Do fixture-repository commits made by a test count? If they do:
- the suite needs another way to give the checker a commit to read, for example a fixture tarball of a pre-built repository;
- this worktree's `data/e2e_grader_selftest/` (6 run directories, 129 fixture repositories) holds them, to be removed or kept as you decide.

I have not touched them.

## What stays open (for the owner)

1. Whether to adopt Q5 5a, 5b and 5c before part 2 lands (my proposal: yes, all three; the registry file itself does not change).
2. The fixture-commit question above.
3. Carried over from the READY note: the 4 kept specs that disagree with the code, the agent-name mapping, the 5 recommended registrations, and the LL-2 route (`DESIGN_LL2_WATCHER_ROUTE_20261125.md`).
