# X-RELAY-CHECK · R1 · BETA REVIEW

- Date: 2026-11-12 (PST).
- Review class: AFTER-COMMIT (tool); reviewed working-tree files.
- Base commit read / HEAD: `a147280ce115cba2137eba28c350018c7ed8c18b`.
- `git status --porcelain` at initial read:

```text
 M CLAUDE.md
 M tests/gate_s172_prod_shape.py
?? data/
?? docs/CLAUDE_CODE_INSTRUCTIONS_BETA_RELAY_20261112.md
?? docs/CLAUDE_CODE_REPORT_PROD_SHAPE_RUN_SCOPING.md
?? docs/exchange/LEDGER.md
?? docs/exchange/RELAY_PROTOCOL.md
?? docs/exchange/from_beta/X-RELAY-TRIAL_BETA_ACK.md
?? docs/exchange/to_beta/X-RELAY-CHECK_R1_ALPHA_REVIEW_REQUEST_relay.md
?? scripts/beta_relay.py
?? tests/test_beta_relay.py
?? tests/test_gate_prod_shape_scoping.py
```

| Question | Position |
|---|---|
| Q1 | OBJECTION |

## Q1 — Yes: a persistent change to an ignored file escapes detection

**Verified path:** a fake Beta child writes a valid reply and creates `logs/hidden.txt` in a temporary repository where `logs/` is ignored. The actual `run_relay()` returns `REVIEW RECEIVED`, exit 0, with `violations=[]`; the outside file remains present containing `outside change`.

The snapshot obtains untracked files using `git ls-files --others --exclude-standard`, which excludes ignored files (`scripts/beta_relay.py:214`). Its other repository content checks use Git status and diff (`scripts/beta_relay.py:202`, `scripts/beta_relay.py:204`, `scripts/beta_relay.py:207`); there is no independent walk of ignored repository contents in that snapshot. The separate recursive walk covers only `from_beta/` (`scripts/beta_relay.py:231`). After comparison (`scripts/beta_relay.py:446`), an empty violation list, successful child, and valid reply reach `EXIT_RECEIVED` (`scripts/beta_relay.py:465`, `scripts/beta_relay.py:474`).

This is a **known limitation, not a newly discovered defect**: `docs/exchange/RELAY_PROTOCOL.md:94` explicitly records ignored files and most `/tmp` writes as unseen. Nevertheless, Q1 asks whether *any* such path exists, so its answer is OBJECTION. The unqualified “nothing outside” meaning at `docs/exchange/RELAY_PROTOCOL.md:53` exceeds that coverage. At minimum, qualify the verdict as no detected changes within the snapshot's coverage; comprehensive enforcement would require additional controls.

## Evidence and scope

Read both requested files in full, plus the fixture helpers in `tests/test_beta_relay.py:47`, `tests/test_beta_relay.py:64`, `tests/test_beta_relay.py:89`, and `tests/test_beta_relay.py:100`. Searched the specified ruling, proposal, chapter-audit and consolidated-TODO records and relay documents for recorded limitations before reporting this position.

Ran a throwaway Python check after `source ~/venvs/torch/bin/activate`, with `PYTHONDONTWRITEBYTECODE=1`. Loaded the existing fixture helpers using `runpy.run_path` without invoking their suite main, loaded a temporary copy of the relay, and injected the existing valid-reply script plus:

```python
pathlib.Path('logs/hidden.txt').write_text('outside change')
```

Fixture: `/tmp/beta_relay_test_qig52ugi/case01/repo`. Observed:

```text
verdict: REVIEW RECEIVED code: 0 violations: []
outside content: outside change
```

All fixture writes, including its Git initialization and initial commit, were confined to `/tmp`; no live relay, pipeline, worker, rig connection, or live-state mutation was performed. The existing full suite was not run; no full-suite pass is claimed. `git check-ignore logs/beta_relay/example.log` also confirmed that the real repository ignores that example log path; no real log was created by this review.

Hashes rechecked before writing this reply were unchanged:

```text
b7f13effa4ce701dc659da7317d61d37c41aeea8fbab0a7cd697b78443a84b20  scripts/beta_relay.py
b8bdc360bc7ef872b6525c91acfdaa026224a48f76fe6efc0f79d5d8ce0d87b9  docs/exchange/RELAY_PROTOCOL.md
```

## Additional findings

None outside Q1's scope.
