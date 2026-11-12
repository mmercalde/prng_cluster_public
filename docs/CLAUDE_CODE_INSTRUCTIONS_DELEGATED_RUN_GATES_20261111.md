# CLAUDE CODE INSTRUCTIONS — TWO REFUSAL GATES FOR AGENT-LAUNCHED RUNS — 2026-11-11

**From:** Team Alpha · **Owner:** Michael · **Run on:** VM101, as `michael`, in
`~/distributed_prng_analysis`, venv `source ~/venvs/torch/bin/activate` · **Model:** Opus
**Base commit:** `cf9edaa`. Anchors were read at that commit by Alpha.
**Review class:** AFTER-COMMIT (tools). Both gates go to Beta before the first agent-launched
run uses them.
**Specification:** `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md` §5.2, §7.1, §7.2.

**Why.** The owner has ruled that agents may launch runs, with two absolute limits: agents never
update the system, and never launch outside what the owner authorized. A limit with no gate
that refuses is not a control. This brief builds the two gates as standalone tools. It does
**not** wire them into any launch script; that is a later brief.

---

## 0. Rules for this session

- **You do not commit or push.** Stop at the gate in §6 and report.
- **You may write only these files, all new:**
  - `scripts/env_freeze_gate.py`
  - `scripts/run_authorization_gate.py`
  - `tests/test_env_freeze_gate.py`
  - `tests/test_run_authorization_gate.py`
  - `docs/CLAUDE_CODE_REPORT_DELEGATED_RUN_GATES.md`
- **No existing file is edited.** In particular not `gate12_launch.sh`,
  `scripts/gate12_parity_gate.py`, `scripts/launch_fleet_manual.sh` or `execution_set.py`.
- **You do not create a baseline and you do not create an authorization.** Those are the
  owner's acts. Tests build their own in a temp directory.
- **Nothing is installed, upgraded or removed on any host.** Every probe is read-only.
- **The fleet is off.** Do not try to power, wake or reboot anything. A rig that cannot be
  reached is the expected result today and is reported as UNAVAILABLE.
- Other sessions are editing `docs/CHAPTER_*.md` and `tests/gate_s172_prod_shape.py`. Leave
  those alone.
- Logs go under `data/delegated_run_gates/` (untracked, not committed).
- The phase-4 Gate 22 red caused by new untracked `.py` files is expected. Do not widen its
  allowlist.

## 1. Investigate (read-only; answers go in the report)

1. **How `scripts/gate12_parity_gate.py` resolves its targets and probes them:**
   `gate_targets` (`:366`), `build_probe_script` (`:409`), `parse_probe_output` (`:443`),
   `probe_rig` (`:492`), the BEGIN/END framing that makes a truncated listing UNAVAILABLE, the
   injectable `runner`, and the exit codes (`:150-151`). The new gates reuse these by import
   where the function fits unchanged. List which you import and which you could not, and why.
2. **What the local node is** in the resolved execution set, and how to probe VM101 itself
   without ssh.
3. **What can be read without `sudo`** on VM101, and the equivalent command for a CT100 worker,
   for each of these facts. Give the exact command and a sample of its output on VM101:
   - kernel release, and the kernel command line;
   - the Python version of the environment the worker runs in (`~/venvs/torch` on VM101,
     `~/rocm_env` on the rigs);
   - the full `pip freeze` of that environment;
   - the installed OS package list (`dpkg-query -W`);
   - the GPU driver version (NVIDIA on VM101; `amdgpu` on the rigs);
   - the CUDA or ROCm version.
   A CT100 worker is an unprivileged container, so its kernel release is the Proxmox host's.
   Say so in the report: that is how host kernel changes become visible without host access.
4. **`.gitignore`:** `*.json` (`:41`), `*.log` (`:11`) and `logs/` (`:85`) are ignored. Confirm,
   and name a tracked location and file extension for the baseline and the authorizations that
   is **not** ignored. Proposed: `config_manifests/env_freeze_baseline.cfg` and
   `config_manifests/run_authorizations/<CLASS>.auth`, both holding JSON text.

## 2. Gate G2 — environment freeze (`scripts/env_freeze_gate.py`)

**Purpose.** Refuse a launch if anything on any host in the execution set differs from the
owner's baseline. This is the enforcement of "agents never update the system".

**Subcommands.**

- `capture --out <path>`: probe every host in the resolved execution set, plus the local node,
  and write the baseline. It refuses to write if any host is UNAVAILABLE, and refuses to
  overwrite an existing file. Its help text says it is run by the owner only.
- `verify --phase pre|post [--evidence-json <path>]`: probe the same hosts and compare.

**What a host record holds.** For each fact in §1.3: the value, and for the two long lists
(`pip freeze`, OS packages) both the sha256 and the full sorted list, so a refusal can name the
package that changed and not only say that something did.

**Verdict rules.** Same vocabulary and exit codes as the parity gate: 0 = proceed, 1 = REFUSE.

| Condition | Verdict |
|---|---|
| Every fact on every host equals the baseline | PROCEED |
| Any fact differs on any host | REFUSE, naming host, fact, baseline value and live value; for a list, the added, removed and changed entries |
| A host in the execution set is missing from the baseline, or the reverse | REFUSE |
| A host cannot be reached, times out, or returns truncated output | UNAVAILABLE, which is REFUSE |
| A probe ran and its output cannot be parsed | ERROR, which is REFUSE |
| The baseline file is absent | REFUSE |
| The baseline file is not tracked by git, or differs from `HEAD` | REFUSE. Only the owner commits, so a tracked, unmodified baseline is the owner's |
| `git` cannot run | UNAVAILABLE, which is REFUSE |

`--phase post` uses the same rules; its report line reads `VIOLATION` in place of `REFUSE`, and
the exit status is still 1.

**Design constraints.**
- Probes are one script per host passed as a single ssh argument, framed with BEGIN and END
  lines, as the parity gate does.
- `local HEAD`, timestamps and hostnames are printed as context and never enter the comparison,
  except the hostname check the parity gate already makes.
- The comparison function takes two host records and nothing else, so nothing printed for
  context can affect a verdict.

## 3. Gate G3 — run authorization (`scripts/run_authorization_gate.py`)

**Purpose.** Refuse any launch that the owner has not authorized in writing.

**The authorization file.** One per run class, JSON text, tracked in git, at the location
settled in §1.4:

```
{
  "run_class": "RUN4",
  "params": { "<name>": <exact value> | {"min": <n>, "max": <n>}, ... },
  "seed_start_min": <int>, "seed_end_max_exclusive": <int>,
  "max_launches": <int>,
  "max_wall_clock_s": <int>,
  "window_local": {"tz": "America/Los_Angeles", "start_hour": <0-23>, "end_hour": <0-24>},
  "expires": "YYYY-MM-DD"
}
```

**Subcommands.**

- `verify --class <CLASS> --params <json or @file> [--evidence-json <path>]`
- `consume --class <CLASS> --run-nonce <nonce>`: records one launch. Called by a launcher only
  after every gate has passed.
- `show --class <CLASS>`: prints the authorization and launches used.

**Verdict rules.**

| Condition | Verdict |
|---|---|
| File for the class absent, unparseable, or missing a required field | REFUSE |
| File not tracked by git, or differs from `HEAD` | REFUSE |
| `run_class` in the file does not equal `--class` | REFUSE |
| Today is after `expires` | REFUSE |
| Current local time is outside `window_local` | REFUSE |
| Launches already recorded for this class ≥ `max_launches` | REFUSE |
| A launch parameter is not named in `params` | REFUSE. An unknown parameter is never assumed harmless |
| A parameter named in `params` is absent from the launch | REFUSE |
| A value differs from an exact value, or lies outside a `min`/`max` | REFUSE, naming the parameter |
| The launch's seed range is not inside `[seed_start_min, seed_end_max_exclusive)` | REFUSE |
| `--end-step` is not 1, or `n_parallel` is not 1, when either is supplied | REFUSE |
| Everything holds | PROCEED |

**Launch record.** Append-only JSON lines under `logs/run_authorizations/<CLASS>.jsonl`
(`logs/` is ignored, so the clean-tree gate is unaffected). Each line: class, nonce, UTC time,
the sha256 of the authorization file, and the parameters. A nonce already recorded is REFUSE.
A record whose authorization hash differs from the current file does not count toward
`max_launches` and is listed in the output.

**Known limit, to be stated in the report and not solved here.** The launch record lives on
VM101 under the same account the agent runs as, so it can be deleted. The count is therefore
cross-checkable, not tamper-proof. Say what would make it so.

**Design constraints.** The clock is injectable. Comparison of numbers is exact for integers;
floats are compared after `float()` on both sides with no rounding. Booleans are compared as
booleans, not as strings.

## 4. Gates (tests)

Both suites build everything in a temp directory, including a throwaway git repository for the
tracked-and-unmodified checks. No test contacts a rig. Probe output is supplied through the
injectable runner.

**`tests/test_env_freeze_gate.py`**

| Gate | Fixture | Must hold |
|---|---|---|
| E-A | Three rigs and the local node, all equal to baseline | PROCEED, exit 0 |
| E-B | One rig's kernel release differs | REFUSE naming that rig and both values |
| E-C | One package version differs in one rig's `pip freeze` | REFUSE naming the package, old and new version |
| E-D | A package added on one rig; a package removed on another | REFUSE listing each as added or removed |
| E-E | GPU driver version differs on the local node | REFUSE |
| E-F | One rig times out | UNAVAILABLE and exit 1, never PROCEED |
| E-G | One rig's output is cut before END | UNAVAILABLE and exit 1 |
| E-H | Baseline file absent | REFUSE |
| E-I | Baseline present but untracked | REFUSE |
| E-J | Baseline tracked, then modified in the worktree | REFUSE |
| E-K | A rig is in the execution set and not in the baseline | REFUSE |
| E-L | `capture` with one rig UNAVAILABLE | Writes nothing, exit 1 |
| E-M | `capture` onto an existing file | Writes nothing, exit 1 |
| E-N | Context fields (timestamp, `local HEAD`) differ between baseline and live | PROCEED. Context never decides |
| E-MUT-1 | Comparison made to ignore `pip freeze` | E-C red |
| E-MUT-2 | UNAVAILABLE mapped to PROCEED | E-F red |
| E-MUT-3 | Tracked-and-unmodified check removed | E-I and E-J red |

**`tests/test_run_authorization_gate.py`**

| Gate | Fixture | Must hold |
|---|---|---|
| A-A | Valid authorization; launch parameters equal | PROCEED |
| A-B | One exact parameter differs | REFUSE naming it |
| A-C | A launch parameter not named in the authorization | REFUSE |
| A-D | A named parameter missing from the launch | REFUSE |
| A-E | A bounded parameter at `min`, at `max`, just below, just above | PROCEED, PROCEED, REFUSE, REFUSE |
| A-F | Expired yesterday; expires today | REFUSE; PROCEED |
| A-G | Clock outside the window; at the start hour; at the end hour | REFUSE; PROCEED; REFUSE |
| A-H | `max_launches = 2` with 0, 1, 2 launches recorded | PROCEED, PROCEED, REFUSE |
| A-I | Same nonce consumed twice | Second is REFUSE |
| A-J | Seed range starts below the minimum; ends past the maximum | REFUSE; REFUSE |
| A-K | Authorization untracked; tracked then modified | REFUSE; REFUSE |
| A-L | `--class` differs from the file's `run_class` | REFUSE |
| A-M | Boolean `true` in the file, string `"true"` in the launch | REFUSE |
| A-N | A launch record made under a different authorization hash | Not counted, and listed |
| A-MUT-1 | Unknown parameters allowed | A-C red |
| A-MUT-2 | Expiry check removed | A-F red |
| A-MUT-3 | Launch count check removed | A-H red |

Then two live checks, reported but not gates:

- **LIVE-E.** `env_freeze_gate.py verify --phase pre` on VM101 as it stands. Expected: REFUSE,
  because no baseline exists. Report the exact output.
- **LIVE-P.** Run the local-node probe alone and print the record it produces for VM101, so the
  owner can see what a baseline will contain.

## 5. Verification block

- **Execution proof:** each gate prints its name and verdict; each suite prints
  `COMPLETION SENTINEL: PASS` only when every gate ran and passed.
- **Clean controls:** E-A, E-N, A-A.
- **Fault-injection controls:** every REFUSE and UNAVAILABLE gate above, and the six mutants.
  Mutants are applied to a copy of live source. A mutant that fails to apply is an error.
- **Unavailable-observer behaviour:** UNAVAILABLE is never PROCEED and never a zero.
- **Scope proof:** `git status --porcelain` shows only the five files in §0, `data/`, and files
  owned by the other sessions. No file under `config_manifests/` or `logs/` was created outside
  a temp directory.

## 6. Stop gate and report

Stop without committing. Write `docs/CLAUDE_CODE_REPORT_DELEGATED_RUN_GATES.md` with:

1. The §1 answers, with the exact probe commands and sample output.
2. Both gate tables with verdicts and the mutant evidence.
3. LIVE-E and LIVE-P output.
4. **For the owner, to run on power-on day:** the exact `capture` command, and a filled-in
   example authorization for class `RUN4` built from
   `docs/S172_RUN4_PROPOSAL_PINNED_GEOMETRY.md` §1, with the three owner-set fields
   (`max_wall_clock_s`, `window_local`, `expires`) left as `OWNER TO SET`.
5. What wiring the gates into a launch script will need, and which existing suites pin
   `gate12_launch.sh`.
6. Anything in this brief the source contradicted.
7. Open items, including the launch-record limit in §3.
8. The exact `git add` line for Michael, built from `git status`.
