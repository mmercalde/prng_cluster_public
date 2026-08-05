# CLAUDE_CODE_INSTRUCTIONS_MANUAL_FLEET_LAUNCH.md — REV1

**Bring up 25 worker daemons by hand and run the Phase-7 soak against them.**

**Base:** HEAD `2d3e34c` or later. Claude Code on **VM101** as `michael`,
**run from `/home/michael/distributed_prng_analysis`.**

**⚠ THIS BRIEF AUTHORISES LAUNCHING PROCESSES ON THE RIGS AND STARTING THE SOAK.**
It does **not** authorise committing, pushing, editing production code, or building a supervisor.

---

## 0. Why this is being done by hand

Your own investigation established: **RANGE-MINER has no worker-launch mechanism, has never had
one, and has never run more than 3 daemons.** The 25-daemon fleet the frozen execution set
describes **has never existed as running processes.**

A supervisor is a **new deliverable** and goes to Team Beta separately. **This brief is a one-off
manual launch**, for two reasons:

1. **It answers a question nobody can answer from source: does a 25-daemon fleet actually hold up?**
   The S157 JIT-cache race and the S155 VA-space/OOM risk at 8 workers/rig are **untested on this
   path** — the only evidence is PWC's OOM. **The supervisor should be designed against observed
   behaviour, not assumed behaviour.**
2. It unblocks the soak measurement — the RAM series with `_FLUSH_CLEAR_IN_MEMORY = True` — which
   is what Phase 7 exists to produce.

**No outcome is expected.** *"The fleet does not survive 8 workers/rig"* is a real and valuable
result. **Do not tune, retry silently, or work around a failure to make the soak start** — record
it and stop.

---

## 1. The three hazards a hand launch must handle

All three are yours, from today's investigation. **A naive loop hits all of them.**

### 1.1 `--device-index 0` is MANDATORY on every ROCR worker

`main()` defaults `device_index = gpu_id` when `--device-index` is omitted
(`miner/range_miner_worker.py:1498`). Under `ROCR_VISIBLE_DEVICES=N` the visible device count is
**1**, so **seven of eight workers would silently try to bind an index that no longer exists.**

- **`--gpu-id N`** = the **logical** identity → `rrig6600:gpuN`, and **must match the frozen set.**
- **`--device-index 0`** = the **physical** bind under ROCR.

**They deliberately differ. Pass both, explicitly, every time.**

### 1.2 Per-worker `CUPY_CACHE_DIR`

**S157: 8 workers sharing one JIT cache directory race.** Never hit, because 8/rig has never run.
**Give each worker its own.**

### 1.3 Cold-start stagger

`warm_gpu` **explicitly delegates the stagger to the spawner**, and there is no spawner. **Space the
launches.** Use ~3 s between workers on a rig; report what you used.

### 1.4 Ordering — the coordinator must be listening FIRST

Workers `connect()` with **no retry** (`:1232`). A worker started before the coordinator binds
`:5700` dies immediately.

**Therefore: start the soak first, then bring the fleet up inside the 180 s admission window.**
`worker_admission_timeout` is `getattr(coordinator, 'worker_admission_timeout', 180.0)` —
**no CLI flag, no manifest key. It cannot be shortened.**

---

## 2. Pre-flight — before anything is launched

1. `git status --porcelain` **EMPTY.** *(Item 5's clean-tree wall rejects at publication.)*
2. No stale workers anywhere: `pgrep -af range_miner_worker` on VM101 **and each rig.**
   **If any are running, STOP and report** — a leftover daemon holds a `worker_id` in the frozen set.
3. Nothing already bound to `:5700` on VM101: `ss -tln '( sport = :5700 )'`.
4. Record free RAM on VM101 and each rig, **and free VRAM per GPU** (`rocm-smi --showmeminfo vram`).
   **§1.2/§1.3's risks are memory risks — the baseline is the evidence.**

## 3. Launch — soak first, then the fleet

**Step 1 — the soak** (this is the certified-parameter command; do not alter it):

```bash
cd /home/michael/distributed_prng_analysis
source ~/venvs/torch/bin/activate
PYTHONPATH=. nohup python3 agents/watcher_agent.py \
  --run-pipeline --start-step 1 --end-step 1 \
  --params '{"use_range_miner": true, "use_persistent_workers": false, "worker_pool_size": 25, "n_parallel": 1, "window_trials": 50}' \
  > logs/phase7_soak.log 2>&1 &
echo $! > logs/phase7_soak.pid
```

**Step 2 — wait for the port**, then launch. Do not sleep blindly:

```bash
for i in $(seq 1 60); do ss -tln '( sport = :5700 )' | grep -q 5700 && break; sleep 1; done
```

**Step 3 — VM101's 3080 Ti** (no ROCR, `~/venvs/torch`), then **8 per rig** (`~/rocm_env`) with
`ROCR_VISIBLE_DEVICES=$N`, `CUPY_CACHE_DIR` **per worker**, `--host 192.168.3.177`,
`--gpu-id $N --device-index 0`, each to its own log, ~3 s apart.

**Confirm the `--host` value from `distributed_config.json` / the execution set rather than
trusting this brief.** VM101's own worker may use `127.0.0.1`; the rigs need the LAN address.

## 4. Verify registration — the ~15 s gate, not the 180 s timeout

**Poll from the moment the first worker starts:**

```bash
ss -tn '( sport = :5700 )' | grep -c ESTAB          # 1 per registered worker
sqlite3 <ledger> "select worker_id,status from workers"   # eligible vs quarantined
```

**Target: 25 established, 25 eligible.**

**If the count stalls short for ~15 s, STOP and report — name the missing `worker_ids`.** Do not
wait out the 180 s. **The difference between "24 registered" and "25 registered" is the entire
diagnosis** you would otherwise wait three minutes to not learn.

**Locate the ledger path from source; do not guess it.**

## 5. Then monitor — this is the actual soak

Poll every 60 s and **log a SERIES, not a latest value**:

| what | why |
|---|---|
| **coordinator RSS** | **THE HEADLINE RESULT.** First run ever with `_FLUSH_CLEAR_IN_MEMORY = True`. Monotonic growth across trials = S166 still broken |
| trials completed / failed | progress |
| established worker connections | a worker dying mid-soak is an abort trigger |
| free RAM on VM101 and rigs | **VM101 has NO SWAP — the failure mode is an OOM kill, not swapping** |
| per-GPU VRAM on one rig | §1.2/§1.3 risk surface |

**Abort triggers — capture state, stop, report:** a worker dies without the coordinator recording a
completion · established count drops · coordinator RSS grows monotonically across ≥5 trials ·
any OOM in `dmesg` on VM101 · a trial that neither completes nor fails.

**Report `UNAVAILABLE`, never `PASS`, for host-kernel GPU faults** — `.121/.155/.163` are
unreadable from VM101; that exception is owner-authorised and Beta-acknowledged.

## 6. Shutdown

**When the soak ends or you abort: stop all 25 daemons** (`SIGTERM` — the worker installs a handler
at `:1511`) and confirm none survive on any host. **A leftover daemon poisons the next run's
admission.**

## 7. Report — final message; create NO file

The exact commands used, per host · the stagger interval · registration timeline (how long to 25,
or where it stalled and which `worker_ids` were missing) · **the RSS series with trial numbers** ·
trials completed · every abort criterion evaluated · **whether 8 workers/rig held up: VRAM, RAM,
JIT-cache behaviour, and any evidence of the S157 race or S155-class OOM** · shutdown confirmation.

**Then STOP.** Do not commit, push, or build a supervisor.

---

## Verification-integrity controls (VIR-1…6)

- **execution proof:** quote the registration poll output at 25, and the RSS series as a table.
  **"It ran fine" is not a result.**
- **clean control:** the pre-flight of §2 — a fleet-free, port-free, clean-tree starting state.
- **fault-injection control:** `NOT_APPLICABLE` — a live run, not a detector. **Write
  `NOT_APPLICABLE`, never `PASS`.**
- **completion sentinel:** `PASS | FAIL | UNAVAILABLE | INCOMPLETE`. **A soak that stops early is
  `INCOMPLETE`.** **`PASS` requires 50 trials AND the RSS series — not a green WATCHER line:
  WATCHER scored a crashed step 1.0000 twice today because `file_exists` passed on a stale
  artifact.**
- **audit claim scope:** live fleet — VM101 plus three CT100 workers.
- **searched surfaces:** name every host you touched.
- **unavailable surfaces:** the Proxmox host kernel logs on `.121/.155/.163`; rig `~/.bash_history`
  (established absent today).
