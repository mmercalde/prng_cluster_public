# CLAUDE CODE REPORT — rig GPU fault: decode, first 50 ms, cross-host timing, next tests

**Direction:** Alpha, 2026-11-15 after the 18:28 fault. This is a read-only pass with no GPU
use and no host write. It also covers Alpha's four additions and Beta's informal comments
(owner-relayed; not a relay round). No Beta round.
**Base:** live tree `048fd47`. Rig access ended at the owner's 19:08 shutdown; copies are in
`logs/rig_gpu_fault_20261115/offline_1908/` (`SHA256SUMS.txt`).
**Status:** analysis only. No root cause is claimed.

---

## Prior record

**Searches run** (2026-11-15):
- `/bin/grep -niE 'pre-?compil|warm-?up|warmup|prewarm|eager|warm_gpu|kernel cache|cupy_cache|CUPY_CACHE'`
  over `/home/michael/tfm_worktrees/gpu-fault-history/docs/REGISTER_GPU_FAULT_PRIOR_WORK.md`.
- The same register: Table 1 (`:2755-2867`, GCVM rows) and the uptime/boot strings.
- `/bin/grep -nE 'worker_pool_size|admission_count|expected_workers'` over `miner/`,
  `execution_set.py` and `gate12_launch.sh`, plus `/home/michael/run4_launch.sh`.
- `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md` (read `:130-210`).
- `git show 22bb145`; `docs/LEADS.md`.
- The prior-record search for the no-torch report (`docs/CLAUDE_CODE_REPORT_MINER_NO_TORCH.md`).

**Documents that decide or record the subject:**

| Document | Records |
|---|---|
| `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:132-152` | Run-4 geometry APPROVED, including "same stripe geometry / worker pool" (`:149`) |
| Register E27, E30, E28, E29 (`:1081-1129`) | Reverse-sieve warm-up (`8d91311`, "worse or no effect"), kernel pre-compile at start (`22bb145`, "made crashes worse", reverted `ac32cd9`), no-pool and `HSA_NO_SCRATCH_RECLAIM` ("worse") |
| Register `:179`, `:424-425` | A warm-up barrier "didn't help" (01-23) |
| Register `:153`, `:633`, `:2503` | Per-worker `CUPY_CACHE_DIR` (S152, `c06e4e4`) "eliminated rrig6600c crash" |
| Register Table 1 | GCVM rows with uptime: C-15 81 s, C-34 166 s, C-47 161 s, C-86 275 s; B-59 and B-61 with no run active |
| `docs/LEADS.md` L-6 | Lead; now carries the 18:28 refutation |

**Labels:**

| Item | Label |
|---|---|
| Two-runtime hypothesis refuted as the sole cause (18:28) | NEW (recorded in L-6 and both reports) |
| Fault decode, PASID = 32776 + 9·gpu-id, stripe maps, host-wide zero results | NEW |
| Run-#2-after-boot pattern (today) | NEW, filed as L-7; the record contradicts it as a general rule (§1.6) |
| `22bb145` "pre-compile" constructed objects only and compiled nothing (§5) | NEW reading of an ALREADY RECORDED entry |
| Hold-out with `worker_pool_size` 22 | **CHANGES A RULING** (`TB_RULING_RUN4…:149`); needs the owner |
| Preload flag | NEW proposal; the default is unchanged |

**Not searched:** ser8 archives, the private repository, Zeus logs; the rigs after 19:08.

---

## 1. The recorded first-fault lines, per card

Source: main-tree `logs/netconsole_all_rigs.log` (13:21 at `:359-380`, 18:28 at `:855-890`).
Each line carries the VM101 listener's receipt time and, in brackets, the host's kernel uptime.

### 1.1 Status decode

Fields for gfx10 `GCVM_L2_PROTECTION_FAULT_STATUS`:
- MORE_FAULTS [0];
- WALKER [3:1];
- PERMISSION [7:4];
- MAPPING [8];
- CID [17:9];
- RW [18];
- VMID [23:20].

These are checked against the driver's own decode on the same lines. The meaning of the
individual PERMISSION bits is **[UNVERIFIED]**: no gc_10_3 header is on the box.

| Incident | Card (host) | Worker | Host pid | PASID | Address (page) | Status | CID | RW | PERM | WALKER/MAP | VMID |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 13:21 | `19:00.0` (.121) | rrig6600:gpu7 | 895553 | 32839 | `0x70eceb58a000` | `0x00841011` | TCP 0x8 | 1 (write) | 0x1 | 0/0 | 8 |
| 13:21 | `16:00.0` (.163) | rrig6600c:gpu6 | 892773 | 32830 | `0x779ec88c1000` | `0x00801231` | SQC inst 0x9 | 0 | 0x3 | 0/0 | 8 |
| 13:21 | `06:00.0` (.163) | rrig6600c:gpu1 | 891759 | 32785 | `0x77ae98c80000` | **`0xFFFFFFFF`** on the first printed line (register reads all-ones) | — | — | — | — | — |
| 18:28 | `19:00.0` (.121) | rrig6600:gpu7 | 833288 | 32839 | `0x789d1d01f000` | `0x00841051` | TCP 0x8 | 1 | 0x5 | 0/0 | 8 |
| 18:28 | `16:00.0` (.163) | rrig6600c:gpu6 | 831759 | 32830 | `0x78010f611000` | `0x00801231` | SQC inst 0x9 | 0 | 0x3 | 0/0 | 8 |
| 18:28 | `06:00.0` (.163) | rrig6600c:gpu1 | 830758 | 32785 | `0x705f453e9000` | `0x80001281` | SQC inst 0x9 | 0 | 0x8 | 0/0 | **0** (bit 31 set) |

**Facts:**
- Every first printed line has **MORE_FAULTS=1**.
- Each host's first line is preceded by `gmc_v10_0_process_interrupt: 3 callbacks suppressed`
  at both incidents. These are therefore the first **printed** faults, not necessarily the first
  faults.
- Within ~90–150 ms every card's status reads `0xFFFFFFFF` (WALKER 7, MAPPING 1, CID 0x1ff): the
  register reads all-ones, so the device has stopped answering on PCIe.
- `06:00.0` at 18:28 reports VMID 0 and bit 31 in the status, while the interrupt entry says
  vmid 8. That is inconsistent, and the cause is unresolved.
- SQC lines repeat **the same page** on each card (`16:00.0`: `0x78010f611000` twice; `06:00.0`:
  `0x705f453e9000` twice). The TCP lines on `19:00.0` change address on every line.
- No mapping error and no walker error occurred: the page walk completed and a permission check
  failed.

### 1.2 PID → gpu-id → stripe

Host PIDs are host-namespace PIDs and the processes are gone, so pid → worker is made through
the card instead:
- **Card → gpu-id.** The launcher sets `ROCR_VISIBLE_DEVICES=$N` (`scripts/launch_fleet_manual.sh:328`).
  The CT KFD node order by `location_id` is `03,06,0a|09,0d,10,13,16,19`
  (`logs/rig_gpu_fault_20261115/ct_kfd_topology_*.txt`). So `06:00.0`=gpu1, `16:00.0`=gpu6 and
  `19:00.0`=gpu7.
- **Independent check.** The PASIDs fit **32776 + 9·gpu-id** on both hosts at both incidents
  (gpu1 32785, gpu6 32830, gpu7 32839). That is consistent with each worker process opening the
  KFD device plus all 8 render nodes (9 PASIDs), in launch order [mechanism: inference].
- **Stripes.** These come from the first `ACTIVE_STRIPES` snapshot after dispatch (13:21:38.676
  and 18:28:37.356). Rig stripes `st3_s0..s24` were still first claims; Zeus was on its next
  stripe.

| Worker | 13:21 stripe | 18:28 stripe | st3_sN seed_start = (32+N)·2²⁶ |
|---|---|---|---|
| rrig6600:gpu7 | st3_s1 | st3_s7 | different ranges |
| rrig6600c:gpu6 | st3_s20 | st3_s21 | different |
| rrig6600c:gpu1 | st3_s23 | st3_s14 | different |

### 1.3 What the faulting workers share that the other 21 do not

| Property | Shared by the 3 and not the other 21? | Evidence |
|---|---|---|
| Stripe or seed range | **No.** At 18:28, st3_s1, s20 and s23 (the 13:21 faulting stripes) ran on rrig6600:gpu0, rrig6600c:gpu7 and rrig6600b:gpu0, which did not fault. The 18:28 faulted st3_s14 later completed on rrig6600b:gpu6 (`offline_1908/ct_192.168.3.156.tar`, `tmp/minerlogs/gpu6.log` STRIPE_BEGIN 02:33:29Z) | §1.2 |
| Dispatch rank within host (stripe index taken as claim order; there is no per-claim log line) | **No.** 13:21: gpu7 2nd of 8 on rrig6600; gpu6 4th, gpu1 7th on rrig6600c. 18:28: gpu7 8th; gpu1 3rd, gpu6 7th | snapshots |
| Card | **Yes, the same three physical cards at both incidents.** `16:00.0` also carries the same SQC signature on 04-04 and 04-17 (L-6) | §1.1 |
| VBIOS | **No.** `19:00.0` = `113-D53201-R66XTG` (4 such cards on rrig6600); `16:00.0` = `113-EXT800296-L04` (as all 8 rrig6600b cards and 6 others on rrig6600c); `06:00.0` = `113-EXT800291-L03` (unique) | `logs/rig_gpu_fault_20261115/card_attrs_20261115_1900.txt`; `offline_1908/host_*_cards_and_boot.txt` |
| VRAM | **No.** All 24 cards report 8573157376 | same |
| Slot / upstream bridge | **No common position.** Each card sits behind its own switch pair; the three faulting cards are at three different positions | same |
| PCIe AtomicOps ReqEn | **No.** The three faulting cards are `ReqEn-`, as are 18 non-faulting cards; the three `ReqEn+` cards (03:00.0) did not fault today | same |
| Link | **No.** All x16 16 GT/s (cap) | `run4_attempt2…/hostlogs/gpu_config_*.txt` |
| Kernel cache directory | Per worker: `CUPY_CACHE_DIR=/tmp/cupy_cache_gpu$N` (`launch_fleet_manual.sh:329`), the same scheme for all 24 | §3 |
| Cached binary | Not comparable for `.122`/`.164`: their caches were cleared at the 18:41 CT restart (§3) | — |
| CuPy cache environment | Identical by launcher source. No live process could be read: no workers were running at 18:53 | `offline_1908/ct_*_listing.txt` |
| Host | **Yes**: all three are on rrig6600 and rrig6600c, not rrig6600b | — |
| Run since host boot | **Yes, today**: both faulting runs were run #2 of a boot (§1.6) | — |

### 1.4 Host-wide effect

At the 9-s snapshot, the 16 workers on rrig6600 and rrig6600c had returned **0** phase-4
sub-stripe results, at both incidents. rrig6600b had returned 180 (13:21) and 190 (18:28).
Whole hosts stopped before any of their cards completed one phase-4 sub-stripe; a warm one takes
about 300 ms on rrig6600b.

### 1.5 Firmware

Alpha reports identical firmware packages and `dimgrey_cavefish` rlc/smc md5 on all three hosts.
Package lists and md5 sums are re-captured in `offline_1908/host_*_cards_and_boot.txt`.

### 1.6 Run since boot (Alpha addition)

Boots are from `journalctl --list-boots` (`offline_1908/host_*_cards_and_boot.txt`):
- `.121`: 08:27:40 and 13:53:14;
- `.163`: 08:27:25 and 13:53:02;
- `.155`: 08:27:27, one boot all day.

| Run | .121 / .163 run # on boot | .155 run # | Cache state on .122/.164 | Outcome |
|---|---|---|---|---|
| attempt 1 (09:05, phase 4 ≈09:15) | **1** | 1 | cold (compiled at first use) | clean |
| attempt 2 (13:11, phase 4 13:21:29) | **2** | 2 | warm | **fault** |
| attempt 3 (17:49, phase 4 17:59:01) | **1** | 3 | cold | clean, credited |
| confirm 1 (18:18, phase 4 18:28:29) | **2** | 4 | warm | **fault** |

Two qualifications:
- "No other GPU run between boot and attempt 1/3" rests on the evidence directories. On `.156`
  the earliest cache entry is 09:06:33 (attempt 1), so no earlier run happened on that CT's boot.
  For `.122`/`.164` it is **[UNVERIFIED]**.
- rrig6600b was warm from its run #2 onward and never faulted.

**Against the record** (register Table 1): at least six GCVM events occurred where a warm cache
was impossible:
- C-15 at uptime 81 s, C-34 at 166 s, C-47 at 161 s, C-86 at 275 s (runs started minutes after
  a boot, with per-worker `/tmp` caches);
- B-59 and B-61 with **no run active**.

So a warm cache is **not necessary** for GCVM across the record. The other ~37 GCVM rows carry no
boot time in the register; the per-row check is OPEN (L-7).

## 2. A rig worker's first 50 ms after its first phase-4 stripe (`048fd47`)

**Beta's point (1) is verified.**
- `sieve_gpu_worker.py:126-133` `_get_kernel` only builds `cp.RawKernel(source, name)`, caches it
  in a dict and logs "Compiled kernel".
- CuPy compiles, or reads the cached binary, loads the module and gets the function at the first
  invocation.
- **Measured:** in attempt 1 (cold), `.156` workers logged phase-4 STRIPE_BEGIN at
  17:15:16.064–.081Z, and the phase-4 `.hsaco` files were written at 17:15:16.36–.79Z (cache
  mtimes). The compile ran 0.3–0.7 s after "Compiled kernel" was printed, inside the first launch.

**Order of operations:**

| # | What happens | file:line |
|---|---|---|
| 0 | Phase 3's last sub-stripe ends: `finally` → `_gpu_cleanup_dispatch()` → gc, `deviceSynchronize`, `free_all_blocks` ×2 (the CuPy-only path with `off`) | `miner/range_miner_worker.py:1460`, `:1121-1142` |
| 1 | STRIPE_ASSIGN received; `handle_stripe` builds sub-stripes and emits STRIPE_BEGIN (no GPU) | `:2360-2430` |
| 2 | `execute`: builder and generator-phase guards; `ResidueResolver.resolve` (host, cached dataset) | `:1258-1275` |
| 3 | `import cupy` (already imported); `_get_kernel(family)` builds a RawKernel object only (**no compile, no load**) | `:1278-1283`; `sieve_gpu_worker.py:126-133` |
| 4 | Device allocations. The pools were emptied at step 0, so these are fresh device allocations and mappings: residues `cp.array` (H2D), seeds `cp.arange` (an elementwise kernel, from cache), survivors, match_rates, count (`cp.zeros`), strategy arrays (H2D), strategy_ids, skip_sequences | `:1329-1360` |
| 5 | Builder and `materialize_kernel_args` (host) | `:1392-1393` |
| 6 | `_gpu_launch` → `kernel((blocks,),(threads,),args)` → **first invocation**. CuPy `_compile_with_cache_hip` computes the key; on a hit it reads `<sha1>.hsaco`, checks its content hash and calls `mod.load(binary)` (hipModuleLoadData: the ROCr loader allocates and maps the code object), then gets the function and launches | `:1251-1252`, `:1395-1397`; CT cupy 13.5.1 `cupy/cuda/compiler.py:873`, key `:912`/`:922`, verified load `:937-945` |
| 7 | `survivor_count_gpu[0].get()` blocks the host until the kernel finishes | `:1399` |

**Between obtaining the kernel object (3) and its first launch (6)** there is no explicit
`gc.collect` and no `free_all_blocks`; nothing goes out of scope except host temporaries. There is
no module unload in our code. CuPy keeps the module alive through the RawKernel's function object
(Beta: `function.pyx:210-265`, not read here). Python's automatic generational GC can run at any
allocation, but no cupy object referenced by the kernel is unreachable.

**Concurrently:** the heartbeat thread (`:2339`, started `:2712`) sends frames and makes no CuPy
calls (`/bin/grep 'cp\.'` finds none after `:1727` in the daemon).

**What could unmap, or never map, kernel code.** These are named, not shown:
- (i) The code object's page-table entries are written or invalidated after the dispatch is
  queued: a load→launch ordering or TLB-invalidation race inside ROCr/KFD/amdgpu at step 6.
- (ii) A fresh hipMalloc/free at steps 0 and 4 changes the same VM's page tables while step 6 maps
  code. The register's E29 belief (scratch reclaim invalidating instruction pages) is this class;
  `HSA_NO_SCRATCH_RECLAIM=1` was "worse".
- (iii) Scratch (private segment) for the new kernel is allocated at the first dispatch.
- (iv) A card-local defect that corrupts translation under a load step.

The status says the walk completed and the permission check failed. That is consistent with
(i)–(iii), where the PTE exists but is not yet or no longer valid, and does not separate them.

**The phase-4 kernel is not compiled fresh in the run** (Alpha (a)):
- All four sources are static raw strings with no interpolation: `prng_registry.py:958`
  (java_lcg), `:3115` (reverse), `:1005` (hybrid), `:3170` (hybrid_reverse), registered at
  `:3957-3972`, `:3908-3916`.
- `RawKernel` is built with no options (`sieve_gpu_worker.py:130`).
- The cache key is SHA-1 of `(arch, options, hiprtc version, backend)` + preprocessed empty-file
  base + source + extra source (`compiler.py:912-922`).
- On `.156` no cache entry has been written since attempt 1 (latest 17:15:16Z), so attempts 2 and
  3 and confirm 1 loaded the phase-4 binary from cache.

**`stage=after_warm_gpu`** (`:2123`, `:2307`) runs after `warm_gpu` (`:1718-1727`). That executes
`cp.arange`, `a+1` and `.sum()`: three small CuPy-generated kernels, the three cache entries
written at worker start (`df5f…`, `7ead…`, `42ec…` at 09:06:33–:54 on `.156`, 3 s apart). **It does
not include the phase-4 kernel or any sieve kernel.**

## 3. Kernel cache (Alpha (c); Beta (2))

**Where the caches live and whether they survive a reboot:**
- **Per worker:** `/tmp/cupy_cache_gpu$N`, 8 per CT (`launch_fleet_manual.sh:327-329`).
- `/tmp` is on the CT rootfs (ext4, `pve-vm--100--disk--0`). It **does not survive a CT start**:
  observed absent on `.122` and `.164` after their 18:41 restarts.
- `~/.cupy/kernel_cache` (the default; not used by fleet workers) is persistent.

**Comparisons:**
- **`.156`**: the 8 worker caches each hold 8 entries; **every entry is byte-identical across the
  8 workers** (sha256). The phase-4 object `46d35db4….hsaco` is 6511 B, sha256 `b8081567…`,
  written 17:15:16.36–.79Z in attempt 1, with no entry written since.
- **`.122`, `.164`**: today's caches were destroyed by the CT restart, so they **cannot be
  compared**. The phase-1/phase-2 objects in each CT's persistent default cache (`190aa0bf…`,
  `757a409d…`, compiled independently in July/August) are byte-identical across all three CTs.
  That supports deterministic compilation across the CTs. Equality of the phase-4 object on
  `.122`/`.164` is **[UNVERIFIED]**.
- **Preserved:** tars of all three CTs' cache trees and `/tmp/minerlogs`, with per-file size,
  mtime and sha256: `offline_1908/ct_*.tar`, `ct_*_listing.txt`. Arch and compiler options are
  inside the key. Their explicit values (`gfx1032`, the hiprtc version) are **not separately
  recorded**; that is a gap for the logging in §5.

## 4. How two hosts fault within 0.4 ms (Alpha; Beta (3))

**What the timestamps are.**
- The bracketed values are each host's kernel uptime clock.
- The leading values are the VM101 listener's **receipt** clock.

The 0.34 ms (18:28) and 0.63 ms (13:21) gaps are **receipt** gaps. An attempt to place the events
on the hosts' own clocks fails:

| Host | NCPROOF (17:49) receipt vs kernel | Fault kernel stamp | Projected wall time | Receipt |
|---|---|---|---|---|
| `.121` | 17:49:14.5696 / [14193.7676] | [16548.6730] | 18:28:29.475 | 18:28:29.449 |
| `.163` | 17:49:15.0924 / [14202.5268] | [16556.8153] | 18:28:29.381 | 18:28:29.449 |

The kernel clocks drift by 26 ms and 68 ms against VM101 in 39 min (11–29 ppm), so kernel stamps
**cannot** order the hosts at better than tens of ms.

**Real uncertainty.** The event-time gap is the receipt gap plus the difference in printk→UDP
send delay plus the difference in LAN delay on the two paths. Netconsole normally transmits in
printk context, so this is plausibly sub-ms to a few ms, but it is **not measured**.

**Why the events can coincide with no cross-host coupling.**
- The coordinator sends all 25 phase-4 stripes in one burst (`burst_exact … claimed=25` at
  18:28:29.404).
- rrig6600b:gpu6 logged STRIPE_BEGIN at 18:28:29.402 on its own clock.
- Every worker then runs the identical deterministic §2 sequence on the same software. On a
  warm run this has no compile, so every card reaches its first phase-4 instructions at about
  the same offset after a common trigger.
- The observed offset is +39 ms (13:21) and +46 ms (18:28) after the coordinator's dispatch line.
- On a cold run the compile (0.3–0.7 s, 8 workers on 4 cores) delays and spreads the first
  launches.

**No shared power, PCIe or bus links the two hosts.** The simultaneity is explained by the common
trigger; the faulting itself is card- and host-specific (§1.3). This explanation is an
inference from the sequence. It is consistent with L-7, which it does not establish.

**Unobservable today** (Beta (3)):
- *Does each SQC fault address fall inside the loaded code object?* **Unobservable.** No log
  records the process's mappings or the code object's load address. The no-torch self-check
  reads `/proc/self/maps` but logs only runtime-library paths (`runtime_identity`,
  `range_miner_worker.py:1145-1166`). The faulting workers' logs were lost with the CT `/tmp`.
  The needed record is `/proc/self/maps` written to the log right before and right after the
  first invocation of each phase kernel, plus the code object's GPU load range.

  ROCr's code-object load address is not exposed through CuPy. A ROCm debug-agent or HIP
  `AMD_LOG_LEVEL` record might show it; whether either prints the load range on 6.4.3 is
  **[UNVERIFIED]**, and both are environment changes.
- *Did module load succeed; was anything unloaded or invalidated before launch?* **Unobservable**
  for the faulting workers: their logs are gone, and CuPy does not log a successful load. A failed
  load raises in CuPy and would be a StripeError; the coordinator recorded no error frames, only
  0 frames or 1 heartbeat for the faulting workers (§1.4).

## 5. Next tests

### 5.1 Primary (Beta's preference): move phase-4 load earlier, one worker at a time

**Flag:** `--kernel-preload {off,at_prepare}`, default `off`, which is byte-unchanged
(launcher variable `RIG_KERNEL_PRELOAD`, appended only when non-default, as `RIG_CLEANUP_TORCH`).

**With `at_prepare`:** after `warm_gpu` in `prepare()`/`register()` and before the sentinel and
REGISTER, for each family of the workflow (`java_lcg`, `java_lcg_reverse`, `java_lcg_hybrid`,
`java_lcg_hybrid_reverse`):
1. `_get_kernel(f)`.
2. Force the module load **without a launch** by reading `RawKernel.attributes`. This triggers
   compile-or-cache-read, `hipModuleLoadData` and get-function, and runs no kernel.
3. Log, per family:
   - the cache path, key and sha256 of the `.hsaco`;
   - load success and wall time;
   - the function attributes (`num_regs`, `local_size_bytes`, `max_threads_per_block`);
   - arch and the hiprtc version;
   - a `/proc/self/maps` dump before and after.
4. Log `/proc/self/maps` again before the first launch of every phase.

**One worker at a time:** the launcher already starts workers 3 s apart
(`launch_fleet_manual.sh:150`), and a warm preload of 4 modules takes ms. All preloads therefore
finish, serially per host, before any worker is released.

**Unchanged:** cards, binary (the same cache files), seeds, geometry, `worker_pool_size` 25,
admission and credit. Claim order is not pinnable today. It already varies run to run, and §1.3
shows the fault does not follow it.

**Predictions:**

| Result at phase-4 dispatch | Reading |
|---|---|
| The fault disappears on a run #2 | a load-at-launch (step 6) is necessary for it |
| Same cards, same offset | load timing is excluded; allocation, launch and the kernel remain |
| It moves to preload time | the load itself faults on these cards |

**Power:** needs repeats on run #2s. Today's two run #2s both faulted, a single-arm rate with
denominator 2. A null needs several clean run #2s before it means anything.

**Reconciled with the register:**
- **`22bb145` "pre-compile … made crashes worse" did not pre-compile anything.** Its loop called
  `_get_kernel(fam)` (`git show 22bb145`), and `_get_kernel` at that commit is the same
  object-only constructor (`22bb145:sieve_gpu_worker.py:126-133`). No compile and no module load
  happened at start. The register also records that `sieve_gpu_worker.py` was the SSH-PWC path,
  not the TCP-PWC path then in use (E28/E30 caveat, TBS163:38-40). The worse outcome cannot be
  attributed to eager loading.
- **E27's reverse-sieve warm-up** was a `cp.zeros(1)` context warm-up, not a kernel load
  ("worse or no effect", post-DKMS).
- **The 01-23 "warm-up barrier didn't help"** concerned a different failure class (OOM era).

So the record holds **no test of eager module load**, and the proposed flag is a first test, not
a repeat.

### 5.2 Secondary: three-card hold-out (rrig6600:gpu7, rrig6600c:gpu1, rrig6600c:gpu6)

**What must change:**
- The execution set selects whole nodes only (`execution_set.py:537-612`, `declared_nodes`).
  Admitted identities are `host:gpu0..gpu{count-1}`, from `distributed_config.json` gpu counts.
  There is **no per-GPU exclusion**. A declared exclusion list (e.g. `declared_excluded_workers`)
  would be needed in `resolve_execution_set`, and so in `is_admitted_worker` and
  `admission_expectation` (→ 22).
- The launcher must skip those identities (`launch_fleet_manual.sh` loop, `:218` TOTAL).
- `run4_launch.sh` sentinel and liveness counts (`--threshold 25` `:366`; 25/25 gates `:277-291`)
  must change.
- `worker_pool_size` 25 → 22 (`run4_launch.sh:379`).
- **Stripe geometry does not change:** `max_seeds/miner_stripe_size` gives 32 stripes per stage,
  and slots=32 is independent of the pool.

**Does the search record accept it?**
- The pinned geometry approval includes "same stripe geometry / **worker pool**"
  (`TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:149`). A 22-worker run **changes that ruling**:
  it needs the owner, and a Beta position.
- The run's execution-set id would differ (`[EXEC-SET] … admission=25` in the run log). Whether
  Step 1 credit compares the set id against the Attempt-9/Run-4 record is **[UNVERIFIED]**; it was
  not traced in this pass.

**Beta's caution:** the hold-out also lowers per-host load and alters timing (7 and 6 workers on
two hosts).

### 5.3 Variant: swap which card serves which worker identity (no pool change)

Permute `ROCR_VISIBLE_DEVICES` per identity in the launcher, e.g. identity gpu7 → card N. The
change is one map variable, default identity. Pool, admission, geometry and credit stay at 25.

This tests card against identity/launch position/PASID order:
- the fault follows the physical cards → card;
- it follows the identities → launch position or claim order.

The record already shows the fault does **not** follow stripes (§1.3). A stripe-assignment swap
would need new coordinator claim pinning, and today's data already answers it.

## 6. Records and limits

- Rig copies (before the 19:08 shutdown): `logs/rig_gpu_fault_20261115/offline_1908/`.
  - CT tars: kernel caches, `/tmp/minerlogs` (`.156` only; confirm 1), `~/.cupy/kernel_cache`.
  - Host journals: boot 0 full, boot −1 kernel. The previous-boot journals stop at 13:18:56 and
    18:23:42, before the faults, so netconsole is the only fault record.
  - Per-card sysfs: VBIOS, VRAM, link, bridge path, subsystem, revision, `lspci -vv` excerpts.
  - Firmware md5s.
- Worker logs for attempts 1–3 are in the run evidence directories; `.122`/`.164` logs for the
  two faulting runs do not exist.
- **Lead status:**
  - L-6: OPEN, with the two-runtime arm refuted as the sole cause.
  - L-7: OPEN.
  - No lead was closed.
