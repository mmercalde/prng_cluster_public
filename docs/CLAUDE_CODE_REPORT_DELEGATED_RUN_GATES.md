# CLAUDE CODE REPORT — TWO REFUSAL GATES FOR AGENT-LAUNCHED RUNS — 2026-11-13

**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_DELEGATED_RUN_GATES_20261111.md` (written at `cf9edaa`)
**Worked at:** `fbd8264` (HEAD), in worktree `/home/michael/tfm_worktrees/rm_gates`
(`git worktree add --detach … HEAD`, Team Alpha override 1). Nothing was written in the main tree.
**Run as:** `michael` on VM101 (`zeus-ubuntu-vm`), venv `~/venvs/torch`.
**Status:** STOPPED AT THE §6 GATE. Nothing committed or pushed. No baseline and no authorization
were created outside temp directories. Nothing was installed. No host was powered, woken or rebooted.
**Review class:** AFTER-COMMIT (tools). Both gates go to Beta before the first agent-launched run uses them.

## Headline (null results first)

- **LIVE-E: REFUSE, exit 1.** No baseline exists on VM101 (`baseline ABSENT`). This is the expected result.
- **Fleet: 0 of 3 rigs reachable.** `rrig6600` (`192.168.3.122`) gave `UNAVAILABLE (ssh_transport_failure)`,
  `No route to host`. The fleet is off, as the brief expected. The other two rigs were not probed live.
- **Every rig-side command and sample below is `[UNVERIFIED today]`.** Rig values come from the
  2026-07-30 CT100 session record and from `tests/smoke_s172_phase5_d6_zeus_single_gpu.py:246-250`.
- **Suites:** `test_env_freeze_gate.py` passed **17/17** (14 gates + 3 mutants).
  `test_run_authorization_gate.py` passed **18/18** (14 gates + 1 extra gate + 3 mutants).
  Both print `COMPLETION SENTINEL: PASS`.
- **A finding that affects the RUN4 authorization (§4, §6):** the RUN4 §1 pinned values cannot
  be authorized under their §1 names. The real launch carries them as the seven Route-A
  `warm_start_*` keys in `--params`. The example below uses the real launch keys.

## Anchor drift, `cf9edaa` → `fbd8264` (Team Alpha override 2)

`git diff --stat cf9edaa HEAD` over every anchored file (`scripts/gate12_parity_gate.py`,
`execution_set.py`, `gate12_launch.sh`, `scripts/launch_fleet_manual.sh`, `.gitignore`,
`docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md`, `docs/S172_RUN4_PROPOSAL_PINNED_GEOMETRY.md`)
shows **only `.gitignore` changed**, with four lines appended at `:164-167`
(`data/prod_shape_scoping/`, `data/x0001/`). Every line anchor in the brief still resolves:

| Brief anchor | At HEAD | Drift |
|---|---|---|
| `gate_targets` `:366` | `scripts/gate12_parity_gate.py:366` | none |
| `build_probe_script` `:409` | `:409` | none |
| `parse_probe_output` `:443` | `:443` | none |
| `probe_rig` `:492` | `:492` | none |
| exit codes `:150-151` | `:150-151` (`EXIT_PROCEED = 0`, `EXIT_REFUSE = 1`) | none |
| `.gitignore` `*.log :11`, `*.json :41`, `logs/ :85` | `:11`, `:41`, `:85` | none |

Stale anchor outside the brief, noted but not touched: the comment at `gate12_launch.sh:221` says
`logs/` is ignored at `.gitignore:62`. It is at `:85`.

---

## 1. §1 answers

### 1.1 How the parity gate resolves and probes targets, and what was reused

- **Targets** (`gate12_parity_gate.py:366-406`). `execution_set.resolve_execution_set(backend="miner", …)`
  joins `rig_profiles_config.json` with `distributed_config.json`. `remote_nodes()` then **drops the
  local node**. `script_path` is read from `distributed_config.json`, keyed by `config_hostname`.
- **Probe** (`:409-436`). One shell script hashes every governed file. It exits 3 if the `cd` fails
  and is framed by `TFM-PARITY-BEGIN` and `TFM-PARITY-END`.
- **Parse** (`:443-489`). A missing BEGIN is `ERROR`. A missing END is `UNAVAILABLE`
  (`truncated_probe_output`), which is how a truncated listing becomes UNAVAILABLE. Unknown or
  malformed rows are `ERROR`.
- **probe_rig** (`:492-552`). The script travels as **one ssh argv element**, with
  `-n -o ConnectTimeout=5 -o BatchMode=yes`. A `TimeoutExpired` or any exception is UNAVAILABLE.
  `rc 255` is UNAVAILABLE (`ssh_transport_failure`). `rc 3` is UNAVAILABLE (`no_such_directory`).
  Any other non-zero rc is ERROR.
- **Injectable runner.** It has the shape `runner(cmd, timeout) -> CompletedProcess` and defaults to `_run` (`:439`).
- **Exit codes** (`:150-151`). 0 = proceed, 1 = REFUSE.

**Imported unchanged** (by `scripts/env_freeze_gate.py`, re-exported to `run_authorization_gate.py`):

- `PROBE_OK`, `PROBE_UNAVAILABLE`, `PROBE_ERROR`, which come from `preflight_check` through the parity gate;
- `EXIT_PROCEED`, `EXIT_REFUSE`, `SSH_CONNECT_TIMEOUT`, `SSH_TRANSPORT_FAILURE_STATUS`;
- `_run`, `_utcnow` and `local_head`, which is context only;
- `PROBE_TIMEOUT`, as the base for git calls.

**Not imported, and why:**

| Function | Why it could not be used unchanged |
|---|---|
| `gate_targets` | It drops the local node (`remote_nodes()`, `:393`), which G2 must probe. It also does not return `python_env` or `config_hostname`, so the environment path cannot be joined afterwards. `env_freeze_gate.env_targets` performs the **same join** over `resolved.nodes` and adds `python_env` from `distributed_config.json`. |
| `build_probe_script` | It is hard-wired to hashing `GOVERNED_FILES` under `script_path`. G2 probes different facts. |
| `parse_probe_output` | It accepts only `HOST`/`FILE` rows; anything else is `unknown_row` → ERROR. G2 keeps the same discipline (BEGIN missing → ERROR, END missing → UNAVAILABLE, never repair) with its own row set and two framed list blocks. |
| `probe_rig` | It calls `build_probe_script` internally and has no parameter for the script. G2's `probe_host` copies its structure exactly: one argv element, the same ssh options, the same rc-255 rule, and timeout → UNAVAILABLE. It also adds a local branch (`bash -c`, no ssh). |

**Deviation:** `PROBE_TIMEOUT` is `4 × PG.PROBE_TIMEOUT`, which is 60 s against 15 s. `pip freeze`
plus `dpkg-query` over about 2.4k packages needs more headroom than hashing ten files. A timeout is still UNAVAILABLE.

### 1.2 The local node, and probing it without ssh

From a live, read-only `resolve_execution_set(backend="miner")` at HEAD (default profile `proxmox`):

```
ResolvedNode(node_id='localhost', endpoint='localhost', config_hostname='localhost',
             worker_hostname='zeus-ubuntu-vm', ssh_user='michael', local=True, gpu_count=1, ...)
rrig6600  -> 192.168.3.122 · rrig6600b -> 192.168.3.156 · rrig6600c -> 192.168.3.164
```

The local node is `node_id='localhost'` with `local=True`. Its `distributed_config.json`
`python_env` is `/home/michael/venvs/torch/bin/python`; the rigs use
`/home/michael/rocm_env/bin/python`. `resolve_execution_set` itself raises if
`socket.gethostname()` differs from `worker_hostname` (`execution_set.py:619-631`).

G2 probes the local node by running **the same framed script** as `["bash", "-c", script]`
through the same injectable runner. Parsing and comparison are identical; only the transport
differs, and the rc-255 transport rule is not applied locally. The hostname check covers every
host, the local node included.

### 1.3 Facts readable without `sudo`

Everything runs as `michael`. No command needs root. The probe script is `build_probe_script` in
`scripts/env_freeze_gate.py`. `PY` is the node's `python_env` from `distributed_config.json`.

| Fact | Command (VM101 and CT100 alike) | VM101 sample (live, 2026-11-13) | CT100 expectation `[UNVERIFIED today]` |
|---|---|---|---|
| kernel release | `uname -r` | `6.8.0-149-generic` | the **Proxmox host's** kernel, e.g. `6.8.12-9-pve` (2026-07-30) |
| kernel cmdline | `cat /proc/cmdline` | `BOOT_IMAGE=/boot/vmlinuz-6.8.0-149-generic root=UUID=e8a40b49-… ro quiet splash pcie_aspm=off acpi_rev_override=1 vt.handoff=7` | the host's cmdline: LXC does not virtualise `/proc/cmdline` `[UNVERIFIED on these CTs]` |
| Python version | `"$PY" -c 'import platform;print(platform.python_version())'` | `3.10.12` | `3.10.4` (rrig6600, rrig6600b) and `3.10.12` (rrig6600c), per `docs/TODO_CONSOLIDATED.md:94` |
| pip freeze | `"$PY" -m pip freeze --all` | 269 entries, sha256 `346dce19…562761` (266 without `--all`: pip, setuptools and wheel are added) | same command, `rocm_env` |
| OS packages | `dpkg-query -W -f='${binary:Package}\t${db:Status-Abbrev}${Version}\n'` | 2426 entries, sha256 `452bfe0d…7cdebf`; first row `accountsservice	ii 22.07.5-2ubuntu1.6` | the **container's own** dpkg database (CT userspace, ROCm packages), not the host's |
| GPU driver | NVIDIA: `/proc/driver/nvidia/version` → `Kernel Module` field. amdgpu: `cat /sys/module/amdgpu/version`, falling back to `srcversion` | `550.163.01` (`nvidia-smi` agrees); amdgpu `ABSENT` | `6.12.12` (2026-07-30); host-side module, visible read-only through `/sys/module` |
| CUDA / ROCm | CUDA: `/usr/local/cuda/version.json` → `cuda.version`. ROCm: `head -1 /opt/rocm/.info/version` | CUDA `12.4.1` (`nvcc` reports `12.4.131`); ROCm `ABSENT` | ROCm `6.4.3-128` (2026-07-30); CUDA `ABSENT` |

Full LIVE-P output is in §3.

**On CT100 kernel visibility.** A CT100 worker is an unprivileged LXC container. It has no kernel
of its own, so `uname -r` inside it reports the **Proxmox host's** kernel release, and
`/sys/module/amdgpu/version` reports the host-side driver. This is how a host kernel or driver
change becomes visible to G2 **without any access to the Proxmox host**, which VM101 lacks: root
on `.121` refuses key auth (2026-07-30 record).

**Library versions inside the env.** The CUDA version that torch and CuPy actually use lives in
their wheels, for example `torch==2.5.1+cu121`. CuPy reports runtime 12090. `pip freeze` captures
both, so a wheel swap is caught even though `/usr/local/cuda` stays at 12.4.1.

### 1.4 `.gitignore` and the tracked locations

Confirmed at HEAD: `*.log` is at `:11`, `*.json` at `:41` and `logs/` at `:85`. `git check-ignore -v` reports:

```
.gitignore:85:logs/   logs/run_authorizations/RUN4.jsonl
.gitignore:11:*.log   data/delegated_run_gates/x.log
(no match)            config_manifests/env_freeze_baseline.cfg
(no match)            config_manifests/run_authorizations/RUN4.auth
```

The proposed locations are adopted:

- baseline: **`config_manifests/env_freeze_baseline.cfg`**
- authorizations: **`config_manifests/run_authorizations/<CLASS>.auth`**

Both hold JSON text and **neither is ignored**. `config_manifests/` already exists and is tracked:
`config_manifests/feature_registry.json` is force-added despite `*.json`. The launch record goes in
`logs/run_authorizations/<CLASS>.jsonl`, which is ignored, so the clean-tree gate is unaffected.

---

## 2. What was built

### G2 — `scripts/env_freeze_gate.py`

- **`capture --out <path>`.** Owner only, and the help text says so. It refuses before probing if
  the target file exists, and opens it with `open(…, "x")` so it can never overwrite. It refuses
  and writes nothing if any host is not `OK` or answers under the wrong hostname.
- **`verify --phase pre|post [--evidence-json P] [--rig-profile P]`.** It checks the baseline is
  tracked and byte-identical to HEAD (`git cat-file -e HEAD:<rel>`, then `git cat-file blob`
  compared byte for byte against the file). It parses those **same bytes**. It validates the
  schema, and that each list's sha256 matches its entries. Then it probes every host and compares.
  `post` prints `VIOLATION` in place of `REFUSE`; the exit is still 1.
- **Host records.** `kernel_release`, `kernel_cmdline`, `python_version`, `gpu_driver_nvidia`,
  `gpu_driver_amdgpu`, `cuda_version` and `rocm_version`, plus `pip_freeze` and `os_packages`.
  Each list is stored as `{sha256, entries}` with the entries sorted.
- **Context.** `captured_at`, `local_head`, `rig_profile`, and per host `endpoint`, `hostname`
  and `collected_at` live under `context`. `compare_host_records(baseline, live)` takes two
  records and nothing else. Refusals for lists name every added, removed and changed entry
  (`changed numpy: 1.26.4 -> 2.0.0`).
- **Fail-closed paths.** An unrecognised probe status, an unresolvable execution set, an
  unparseable baseline, or git being unable to run are all REFUSE.
- `--rig-profile` was added for parity with `gate12_parity_gate.py:700`. Its default is the
  profile map's own `default_profile`.

### G3 — `scripts/run_authorization_gate.py`

- **`verify --class C --params JSON|@file [--evidence-json P]`.** It applies the brief §3 table
  verbatim, plus one extra rule (A-X1 below).
- **`consume --class C --run-nonce N --params JSON|@file`.** It takes an exclusive `flock` on the
  record, **re-runs the full verification** inside the lock, refuses a duplicate nonce, then
  appends `{class, nonce, utc, auth_sha256, params}` and `fsync`s. `--params` was added here; see §6.
- **`show --class C`.** It prints the authorization, its sha256, the launches counted, and any
  records not counted.
- **Comparison rules.** Integers compare exactly. If either side is a float, both are converted
  with `float()`, with no rounding. A boolean only equals a boolean, so `True` never equals `1`
  or `"true"`. A string never equals a number. Lists are compared element by element.
  `NaN` and `Infinity` are rejected at parse time.
- **Bounds and window.** Bounds are inclusive. The window is `[start_hour, end_hour)` in
  `window_local.tz`, and `expires` is compared against the date in that same zone.
- **Seed range.** The reserved launch keys `seed_start` and `seed_end_exclusive` state the range,
  and both are required. **Beyond the brief (A-X1):** when the launch also carries `max_seeds`,
  as the real launch does (`gate12_launch.sh:381`), `seed_end_exclusive` must equal
  `seed_start + max_seeds`. Without this a launcher could state one range and run another.
- **`end_step` and `n_parallel`** must be 1 whenever they are supplied, in any of the spellings
  `end_step`, `--end-step` or `end-step`, whatever the authorization says.
- **Name validation.** Class names must match `[A-Z0-9_]{1,32}` (no path traversal) and nonces
  must match `[A-Za-z0-9._:-]{1,128}`.

### Gate tables

Evidence is in `data/delegated_run_gates/test_env_freeze_gate.log` and
`data/delegated_run_gates/test_run_authorization_gate.log`. Both suites were run sequentially.

**`tests/test_env_freeze_gate.py` — 17/17, `COMPLETION SENTINEL: PASS`**

| Gate | Fixture | Verdict observed | Result |
|---|---|---|---|
| E-A | 3 rigs + local, all equal to baseline | PROCEED, exit 0 | PASS |
| E-B | `rrig6600b` kernel `6.8.12-9-pve` → `6.8.12-10-pve` | REFUSE `rrig6600b: kernel_release baseline='6.8.12-9-pve' live='6.8.12-10-pve'` | PASS |
| E-C | numpy version changed on `rrig6600` | REFUSE `changed numpy: 1.26.4 -> 2.0.0` | PASS |
| E-D | package added on `rrig6600`, one removed on `rrig6600c` | REFUSE `added requests 2.32.3` / `removed cupy-rocm-6-0 13.0.0` | PASS |
| E-E | local NVIDIA driver `550.163.01` → `560.35.03` | REFUSE `localhost: gpu_driver_nvidia` | PASS |
| E-F | `rrig6600b` times out | `UNAVAILABLE (reason=timeout)`, exit 1, no PROCEED | PASS |
| E-G | `rrig6600c` output cut before END | `UNAVAILABLE (reason=truncated_probe_output)`, exit 1 | PASS |
| E-H | baseline absent | REFUSE `baseline ABSENT` | PASS |
| E-I | baseline present, untracked | REFUSE `baseline UNTRACKED` | PASS |
| E-J | baseline tracked, then only its context modified in the worktree | REFUSE `baseline MODIFIED` | PASS |
| E-K | `rrig6600d` in the execution set, not in the baseline | REFUSE `in the execution set but not in the baseline` | PASS |
| E-L | `capture` with `rrig6600` returning ssh rc 255 | writes nothing, exit 1 | PASS |
| E-M | `capture` onto an existing file | file bytes unchanged, exit 1 | PASS |
| E-N | committed baseline with different `captured_at`, `local_head`, `collected_at` and `endpoint` | PROCEED | PASS |
| E-MUT-1 | comparison skips `pip_freeze` | E-C **red**: the mutant PROCEEDs | PASS (killed) |
| E-MUT-2 | UNAVAILABLE branch → `continue` | E-F **red**: the mutant PROCEEDs | PASS (killed) |
| E-MUT-3 | tracked check replaced by a plain file read | E-I **red** and E-J **red**: the mutant PROCEEDs | PASS (killed) |

E-J modifies **only** context, so the file still parses and still matches live state. The
tracked-and-unmodified check is the only thing that can refuse it, which is why E-MUT-3 turns it
red. A version that changed a fact would be refused by the comparison anyway and would not test the check.

**`tests/test_run_authorization_gate.py` — 18/18, `COMPLETION SENTINEL: PASS`**

The clock is injected and fixed at Fri 2026-11-13 10:30 America/Los_Angeles. The window is 9-17.

| Gate | Fixture | Verdict observed | Result |
|---|---|---|---|
| A-A | valid authorization, launch equal | PROCEED | PASS |
| A-B | `forward_threshold` 0.71 → 0.72 | REFUSE naming `'forward_threshold'` | PASS |
| A-C | launch adds `generator_phase` | REFUSE `'generator_phase' is not named` | PASS |
| A-D | launch drops `window_anchor` | REFUSE `'window_anchor' is absent` | PASS |
| A-E | `worker_pool_size` bound [24,25]: 24 / 25 / 23 / 26 | PROCEED / PROCEED / REFUSE / REFUSE | PASS |
| A-F | expires 2026-11-12 / 2026-11-13 | REFUSE `expired` / PROCEED | PASS |
| A-G | 08:59 / 09:00 / 17:00 local | REFUSE / PROCEED / REFUSE | PASS |
| A-H | `max_launches=2`; 0, 1, 2 consumed | PROCEED / PROCEED / REFUSE `launches exhausted` | PASS |
| A-I | same nonce consumed twice | second REFUSE `already recorded` | PASS |
| A-J | start 99 < min 100; end 1001 > max 1000 | REFUSE / REFUSE | PASS |
| A-K | untracked; tracked then modified | REFUSE `UNTRACKED` / REFUSE `MODIFIED` | PASS |
| A-L | file content says `SWEEP`, `--class RUN4` | REFUSE `run_class 'SWEEP' in the file != --class 'RUN4'` | PASS |
| A-M | `true` in the file, `"true"` in the launch | REFUSE naming `'test_both_modes'` | PASS |
| A-N | one record under a different authorization sha256, `max_launches=1` | PROCEED; `NOT COUNTED … nonce=old-auth-run` listed | PASS |
| A-X1 *(beyond brief)* | `seed_end_exclusive` ≠ `seed_start + max_seeds` | REFUSE | PASS |
| A-MUT-1 | unknown-parameter loop emptied | A-C **red**: the mutant PROCEEDs | PASS (killed) |
| A-MUT-2 | expiry `if` → `if False` | A-F **red**: the mutant PROCEEDs on the expired file | PASS (killed) |
| A-MUT-3 | launch-count `if` → `if False` | A-H **red**: the mutant PROCEEDs with 0 remaining | PASS (killed) |

**Mutant evidence.**

- Each mutant is a string replacement on a copy of live source. The replacement must match
  **exactly once**; otherwise the mutant counts as an error, which is a FAIL. The copy is loaded
  through `importlib` from the temp directory.
- Kill reasons were checked separately, not just the red/green bit
  (`data/delegated_run_gates/mutant_output_check_env.log` and `…_auth.log`). Every killed mutant
  printed `PROCEED` where the live gate refuses, so no kill came from a crash.

**Clean controls:** E-A, E-N and A-A. **Unavailable-observer behaviour:** UNAVAILABLE is never
PROCEED and never a zero. E-F and E-G exit 1, and the evidence rows keep `status=UNAVAILABLE`
with the reason.

**Scope proof.** `git status --porcelain --untracked-files=all` in the worktree shows only the
four new `.py` files, plus this report once written. All logs and JSON under
`data/delegated_run_gates/` are ignored (`*.log`, `*.json`). Nothing exists under `logs/` or
`config_manifests/` outside the temp directories, which live under `data/delegated_run_gates/`
and are removed at the end of each suite. The main tree shows `M docs/exchange/LEDGER.md` and
`?? docs/exchange/from_beta/X-FINDINGS-PLAN_R1_BETA_REVIEW.md`. **Neither is from this session.**

**Peak RSS:** under 100 MB. The largest object is about 2.4k dpkg lines per host.

---

## 3. LIVE-E and LIVE-P

**LIVE-E** (`data/delegated_run_gates/live_e.log`, evidence in `live_e_evidence.json`):

```
$ python3 -u scripts/env_freeze_gate.py verify --phase pre --evidence-json data/delegated_run_gates/live_e_evidence.json
ENV-FREEZE GATE — VERIFY --phase pre
local HEAD : fbd82644543a81dbda10d124e6a16070ee900e81   [CONTEXT ONLY]
ENV-FREEZE GATE : REFUSE — 1 reason(s):
  * baseline ABSENT: config_manifests/env_freeze_baseline.cfg does not exist
exit=1
```

**LIVE-P** (`data/delegated_run_gates/live_p.log`; the full record including every list entry is
in `live_p_full_record.json`). This is the local node probed alone, twice:

```
status OK hostname zeus-ubuntu-vm expected zeus-ubuntu-vm
two consecutive probes: records identical = True
{
  "cuda_version": "12.4.1",
  "gpu_driver_amdgpu": "ABSENT",
  "gpu_driver_nvidia": "550.163.01",
  "kernel_cmdline": "BOOT_IMAGE=/boot/vmlinuz-6.8.0-149-generic root=UUID=e8a40b49-c19d-41d9-bc96-cc00f9852634 ro quiet splash pcie_aspm=off acpi_rev_override=1 vt.handoff=7",
  "kernel_release": "6.8.0-149-generic",
  "os_packages": {"n_entries": 2426, "sha256": "452bfe0ded6501a72041d122b84cfc1e64ee0290dbe32aa00a28aa2b217cdebf",
                  "first_3": ["accountsservice\tii 22.07.5-2ubuntu1.6", "acl\tii 2.3.1-1", "acpi-support\tii 0.144"]},
  "pip_freeze":  {"n_entries": 269, "sha256": "346dce192b41f572732848d201d5875ef156671f1eadac09b1e924464f562761",
                  "first_3": ["Cython==3.2.2", "Flask==3.1.2", "Jinja2==3.1.6"]},
  "python_version": "3.10.12",
  "rocm_version": "ABSENT"
}
rig probe: {'node_id': 'rrig6600', 'endpoint': '192.168.3.122', 'status': 'UNAVAILABLE',
            'reason': 'ssh_transport_failure', 'stderr': 'ssh: connect to host 192.168.3.122 port 22: No route to host'}
```

The rig line is an extra, read-only check that the real ssh path classifies an off rig as
UNAVAILABLE. It is not a gate.

---

## 4. For the owner, on power-on day

**Capture the baseline** after `boot-proxmox` on all three rigs and the CT100s are up. Then, on VM101:

```bash
cd ~/distributed_prng_analysis && source ~/venvs/torch/bin/activate
mkdir -p data/delegated_run_gates
python3 -u scripts/env_freeze_gate.py capture --out config_manifests/env_freeze_baseline.cfg \
  2>&1 | tee data/delegated_run_gates/capture_$(date +%Y%m%d_%H%M%S).log
echo "capture rc=${PIPESTATUS[0]}"     # must be 0; on 1 nothing was written
python3 -u scripts/env_freeze_gate.py verify --phase pre   # must PROCEED only after you commit it
git add config_manifests/env_freeze_baseline.cfg   # then commit and dual-push as usual
```

- It captures under the profile map's default profile (`proxmox` today). For the other boot
  target, pass `--rig-profile baremetal` and keep a separate baseline. The verify location is
  fixed, so changing boot target means re-capturing.
- Expect a file of roughly 300-400 KB: four hosts, each with its full dpkg and pip lists.
- `rrig6600c`'s Python 3.10.12, against 3.10.4 on the other two rigs, will be **recorded, not
  repaired** (proposal §4 item 3).

**Example authorization for `RUN4`** → `config_manifests/run_authorizations/RUN4.auth`.

The keys are the **real launch `--params` keys**:

- the attempt-9 block at `gate12_launch.sh:378-383`;
- the seven Route-A keys (`docs/S172_RUN4_ROUTING_PATCH_BRIEF_ROUTE_A.md:173-174`), carrying the
  §1 values;
- `start_step` and `end_step` from the WATCHER argv.

`seed_start` is reserved for the seed-range rule, so it is not listed in `params`. The JSON
structure below was checked with `validate_auth` and `check`: with the owner fields filled
temporarily it PROCEEDs against the matching launch
(`data/delegated_run_gates/run4_example_check.log`, `refusals: []`).

```json
{
  "run_class": "RUN4",
  "params": {
    "use_persistent_workers": false,
    "use_range_miner": true,
    "worker_pool_size": 25,
    "max_seeds": 2147483648,
    "miner_stripe_size": 67108864,
    "test_both_modes": true,
    "prng_type": "java_lcg",
    "window_trials": 1,
    "n_parallel": 1,
    "warm_start_window": 12,
    "warm_start_offset": 25,
    "warm_start_skip_min": 6,
    "warm_start_skip_max": 99,
    "warm_start_fwd_thresh": 0.71,
    "warm_start_rev_thresh": 0.47,
    "warm_start_session_idx": 1,
    "start_step": 1,
    "end_step": 1
  },
  "seed_start_min": 0,
  "seed_end_max_exclusive": 2147483648,
  "max_launches": 1,
  "max_wall_clock_s": "OWNER TO SET",
  "window_local": "OWNER TO SET — {\"tz\": \"America/Los_Angeles\", \"start_hour\": H, \"end_hour\": H}",
  "expires": "OWNER TO SET — YYYY-MM-DD"
}
```

**How each value was derived:**

- **Pinned values:** RUN4 proposal §1 (`docs/S172_RUN4_PROPOSAL_PINNED_GEOMETRY.md:18-31`).
- **Session:** `sessions ["midday"]` maps to `warm_start_session_idx: 1`. That route defaults to 0
  (midday+evening) if omitted (Route-A brief `:208-215`).
- **`generator_phase`** is pinned in code, not a parameter (§1: "ALREADY pinned in code"), so it
  is deliberately absent. A launch that supplies it is refused.
- **Seed range:** `[0, 2³¹)` comes from `"seed_start": 0, "max_seeds": 2147483648` at
  `gate12_launch.sh:381`, and from `partition_macro_stripes(..., base_start=0)` at
  `miner/range_miner_coordinator.py:761-783`. §1 itself states only `max_seeds`.
- **`max_launches: 1`** is my proposal for "Run 4", one run. It is not an owner-set field in the
  brief, so change it if you intend retries.
- **Owner fields:** the gate rejects the three `OWNER TO SET` strings as malformed (REFUSE) until
  they are replaced with an integer, an object and a date.
- **Window limit:** the window must satisfy `0 <= start_hour < end_hour <= 24`. A window that
  crosses midnight is not supported (§7).

**`[UNVERIFIED]`: the exact Route-A launch line for Run 4 is not committed anywhere I found.**
`gate12_launch.sh` still carries the attempt-9 block without the `warm_start_*` keys. If the
real RUN4 launch adds or drops any key, this file must change with it, because unknown and
missing keys both REFUSE.

---

## 5. Wiring the gates into a launch script (a later brief)

1. **Placement.** In `gate12_launch.sh`, place G3 `verify` and G2 `verify --phase pre` beside the
   clean-tree, GPU and parity gates. They must come **before** the clean-slate `pkill`, the
   fleet dispatch (`launch_fleet_manual.sh`), the sampler and the coordinator. Use the existing
   `${PIPESTATUS[0]}` pattern (`gate12_launch.sh:159-169`, `:192-198`), never `if ! … | tee`.
2. **One source for the parameters.** The parameters G3 checks and the parameters WATCHER runs
   with must be **the same bytes**. Write the params JSON once to a file under `logs/`. Pass it to
   G3 as `--params @file`, adding `start_step`, `end_step` and
   `seed_end_exclusive = seed_start + max_seeds`, and feed WATCHER from the same file. Whether
   `watcher_agent.py --params` accepts a file is `[UNVERIFIED]`. If it does not, G3 needs the
   same literal string, extracted by the script rather than retyped.
3. **`consume`** runs last, after every gate has passed and immediately before the `nohup … watcher_agent.py` line, using `--run-nonce "$STAMP"`.
4. **G2 `--phase post`** runs from the run-exit supervisor (`gate12_launch.sh` §4.5) and goes into
   the evidence bundle as `VIOLATION` or PROCEED. Its exit status must reach the session's
   stop-condition logic (proposal §8).
5. **`max_wall_clock_s`** is validated by G3 but **enforced by nothing**. The supervisor needs a
   deadline that stops the run, and that is the launcher's job.
6. **Suites that pin `gate12_launch.sh`** must be re-run, and each new invocation string must not
   disturb them:
   - `tests/test_gate12_gpu_gate.py:306-330`: line order of the GPU gate relative to the parity
     gate, `pkill -f`, `launch_fleet_manual.sh`, the sampler and `watcher_agent.py`. Also `:375`,
     which requires `PIPESTATUS[0]`.
   - `tests/test_s172_d6_liveness_gate.py:786-800`, Part D: the order of the sentinel, liveness,
     sampler, coordinator and `--phase release` invocations, found by exact string.
   - `tests/test_gate12_cleantree_admission.py:24, 86, 230-247`: C1 **executes the real script**
     end to end in a sandboxed HOME. It extracts the `GATE12-CONFIG-ROTATION` region verbatim
     (markers at `gate12_launch.sh:225-227`) and pins `HEAD:gate12_launch.sh`.
   - `tests/test_gate12_concurrency_sampler.py:1244` only mentions the script.
7. **Cleanliness.** The two `.auth`/`.cfg` files are tracked, so the clean-tree gate is
   unaffected as long as they are committed. `logs/run_authorizations/` is ignored.

---

## 6. Where the brief and the source disagree

1. **`consume` has no `--params`** (brief §3, subcommands), but the launch record must hold "the
   parameters" (brief §3, launch record). I added a required `--params` to `consume`, and
   `consume` re-verifies under the lock so a count can never pass `max_launches` between
   `verify` and `consume`.
2. **The RUN4 §1 names are not launch parameters.** The brief asks for an authorization "built
   from §1". The RUN4 proposal's own §2 says the pin "cannot be delivered as a parameter"; the
   Route-A patch (`3e2537b`) routes it as `warm_start_*` keys. An authorization keyed on
   `window_size`, `forward_threshold` and the other §1 names would REFUSE every real RUN4 launch
   (A-D: named parameter absent). The example in §4 uses the real keys.
3. **Seed range.** The brief's verdict table needs "the launch's seed range", but the launch states
   `seed_start` + `max_seeds` and no end. This is handled by the reserved keys plus the
   consistency rule A-X1.
4. **`gate_targets` is not reusable unchanged** for G2, because it drops the local node and
   `python_env` (see §1.1). The brief anticipated this possibility ("where the function fits unchanged").
5. **Nothing contradicted on `.gitignore`.** The anchors hold. The only stale `.gitignore`
   reference is `gate12_launch.sh:221` (`:62`), which is not the brief's.
6. **The local hostname is `zeus-ubuntu-vm`.** Both the profile map and `socket.gethostname()`
   say so. CLAUDE.md calls VM 101 `zeus-ubuntu`; that is the VM name, not the hostname. This is
   not a defect, but anyone transcribing a hostname should know it.

---

## 7. Open items

1. **The launch record is not tamper-proof (brief §3 known limit).**
   `logs/run_authorizations/<CLASS>.jsonl` lives on VM101 under `michael`, the account the agent
   runs as. Deleting it resets the count. It is **cross-checkable, not tamper-proof**. Any of the
   following would make it tamper-evident or tamper-proof:
   - an append-only file owned by a separate account (`chattr +a`, which needs root, so it is the owner's act);
   - a remote append-only log on a host the agent cannot write to;
   - the owner committing the record after each run, so `git` history is the counter;
   - **the cheapest independent cross-check:** count the runs already recorded in the miner
     ledger since the authorization's commit date, and refuse if the ledger shows more runs
     than the record does.
2. **"Tracked and equal to HEAD" proves the file is the owner's only while the agent cannot commit.**
   - The deny-list on `git commit` is therefore part of both gates' threat model.
   - The agent can still **check out an older commit** that holds an older, looser authorization.
     Records under a different hash would then not count, so an older file effectively resets
     the count. Its `expires` limits the damage.
   - **Hardening option** (not built): require `HEAD` to equal `origin/main`, or the file to equal
     `origin/main:<rel>`, or run G3 after the clean-tree gate on the canonical tree.
3. **Module parameters are not baselined.** Proposal §4 item 1 forbids changes to "kernel
   parameters or module parameters". G2 covers the kernel cmdline but not
   `/sys/module/amdgpu/parameters/*` (for example `ppfeaturemask`, cited at
   `deploy_docs_20260118.sh:100`). They are readable without root. Adding them is a one-row
   extension; it was not done because the brief fixed the fact list.
4. **There is no rig-environment digest beyond `pip freeze`.** Proposal §7.2 says "digests of the
   rig environment". `pip freeze` catches version changes but not a file edited in place inside
   `site-packages`. A sha256 manifest of `rocm_env` would catch that, at the cost of a slower probe.
5. **`/proc/cmdline` inside an unprivileged LXC** is assumed to be the host's.
   `[UNVERIFIED on these CTs]` Check it on power-on day; whatever it shows is what gets baselined.
6. **A window that crosses midnight is refused as malformed** (`start_hour < end_hour` is
   required). If the owner wants overnight windows, that is a small, testable extension.
7. **`max_wall_clock_s` is not enforced** (§5 item 5).
8. **Phase-4 Gate 22 will be red** until the four new `.py` files are committed. This is expected
   per the brief, and the allowlist was not widened.
9. **Beta review is required before first use** (AFTER-COMMIT class).

---

## 8. `git add` for Michael

The files are in the worktree `/home/michael/tfm_worktrees/rm_gates`, detached at `fbd8264`.
Built from `git status --porcelain --untracked-files=all` there:

```bash
git -C /home/michael/tfm_worktrees/rm_gates add \
  scripts/env_freeze_gate.py scripts/run_authorization_gate.py \
  tests/test_env_freeze_gate.py tests/test_run_authorization_gate.py \
  docs/CLAUDE_CODE_REPORT_DELEGATED_RUN_GATES.md
```

Commit there, then bring the commit onto `main` with `git -C ~/distributed_prng_analysis cherry-pick <sha>`
(or copy the five files into the main tree and add them there). Remove the worktree afterwards
with `git worktree remove /home/michael/tfm_worktrees/rm_gates`.
