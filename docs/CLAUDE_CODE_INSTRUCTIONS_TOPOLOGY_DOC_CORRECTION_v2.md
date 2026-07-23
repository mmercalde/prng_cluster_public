# CLAUDE_CODE_INSTRUCTIONS_TOPOLOGY_DOC_CORRECTION_v2.md

**Task:** Correct the cluster topology **documentation** to the boot-selector model.
**For:** Claude Code on VM 101 (`zeus-ubuntu`, 192.168.3.177), user `michael`,
working dir `/home/michael/distributed_prng_analysis`.
**Scope:** DOCUMENTATION ONLY. **No `.py`, `.sh`, or `.json` file may be edited.**
**Supersedes:** v1 of this doc, and the earlier "CT100 IP Migration Cleanup" draft
(whose core proposal is REJECTED — see §0).

---

## 0. Why this task exists, and what NOT to do

A previous scoping attempt assumed the rigs were **one-way migrated** to Proxmox and
that the bare-metal IPs were dead, and proposed swapping `.120→.122`, `.154→.156`,
`.162→.164` across config and code.

**That premise is wrong. DO NOT PERFORM THAT SWAP.**

Every machine in the cluster is a **boot-selector**: bare Ubuntu on the original IP
(the default boot target) **or** Proxmox, under which the workload runs in a VM/CT at
a different address. Both address sets are valid — never simultaneously.

| Machine | Default boot — bare Ubuntu | Alternate — Proxmox host | Workload endpoint under Proxmox |
|---|---|---|---|
| Zeus | `192.168.3.127` (alias `zeus`, tunnel `rzeus`) | `.128` (`pzeus-lan`/`pve-zeus`, root) | **VM 101 `192.168.3.177`** (canonical dev box) |
| rrig6600 | `192.168.3.120` | `.121` (`pve-rig6600`) | **CT100 `192.168.3.122`** |
| rrig6600b | `192.168.3.154` | `.155` (`pve-rig6600b`) | **CT100 `192.168.3.156`** |
| rrig6600c | `192.168.3.162` | `.163` (`pve-rig6600c`) | **CT100 `192.168.3.164`** |

Consequences that define this task:

1. **Config/code holding `.120`/`.154`/`.162` is NOT a bug.** Those are the correct
   endpoints for the default boot target. Changing them would hardcode one boot mode
   and break the other.
2. The defect is a **data-model error in one documentation section**: `CLAUDE.md` §3
   models rig migration as one-way (`Migrated: yes/no`) when the rigs are
   boot-selectors — a model that same section already applies correctly to Zeus.
3. Making rig endpoints **selectable** (e.g. `rig_profile: baremetal | proxmox`) is
   the correct long-term fix and satisfies the no-hardcoding rule, but it is
   **Phase 6 prep and explicitly OUT OF SCOPE here.**

## 1. Verified facts to document (established 2026-07-22 — do not re-derive)

These were checked live by the operator + Team Alpha. Record them as verified with
that date; do not present them as unverified or re-litigate them.

- **All three rigs are currently booted into Proxmox.** Live ping sweep from VM 101:
  `.120` DOWN, `.122` UP · `.154` DOWN, `.156` UP · `.162` DOWN, `.164` UP.
- **rrig6600's Proxmox side is installed and running** — Proxmox host `pve-rig6600`
  at `.121:8006`, CT 100 `rrig6600` running (observed uptime ~1.5 h, start/shutdown
  task history). The `CLAUDE.md` claim "**NO — still bare-metal**" is FALSE and must
  be removed.
- **michael→CT100 SSH key auth now WORKS on all three CTs.** Verified with
  `ssh -o BatchMode=yes michael@<ip> hostname` returning `rrig6600`, `rrig6600b`,
  `rrig6600c`. The `CLAUDE.md` line "michael→CT100 key auth is NOT yet set up (open
  Phase-3 item)" is stale and must be updated to **done (2026-07-22)**.
- **CT device binding is correct** (`pct config 100` on rrig6600): `dev0:
  /dev/kfd,gid=104`, `/dev/dri` bind mount, `lxc.cgroup2.devices.allow: c 226:* rwm`,
  `hostname: rrig6600` (so `socket.gethostname()` reports the rig name for
  coordinator identity), `ip 192.168.3.122/24`, gw `.10`.
- **ROCm is functional in the CTs across all 24 GPUs** — llama RPC has been run
  across all three rigs under Proxmox. Do not "verify" GPU counts with
  `/sys/class/drm/card*`; that counts display nodes and is misleading.
- **The three rigs are identical clones.** Two CTs share an SSH host-key fingerprint
  as a result. This is expected, not a MITM signal — note it so a future reader
  doesn't misread it.
- **GPU allocation:** the Proxmox host console uses a **dedicated 1660 Ti**;
  Windows VM 100 (`.138`) holds **one** 3080 Ti **only while running**; the **second
  3080 Ti is unassigned and free** to pass through to VM 101 for rig testing.

## 2. Ground rules

1. **Do NOT commit, push, or run WATCHER/the pipeline.** Michael commits and
   dual-pushes after review.
2. **DO NOT EDIT ANY `.py`, `.sh`, OR `.json` FILE.** If you believe one needs
   changing, STOP and report. This task cannot change runtime behavior.
3. **Do not change any IP address in any file.** You are adding *context* about which
   endpoint applies in which boot state — not substituting addresses.
4. **Read each file region before editing.** No blind sed.
5. **Leave the historical record alone:** `SESSION_CHANGELOG_*`, TB rulings,
   proposals, `backups/`, `old_results/`, one-shot `apply_*`/`patch_*`/`fix_*`.
   Stale topology there is history, not a bug.
6. Do not invent verification. §1 facts are established; the only open check is
   Stage 1.

## 3. Stage 1 — the one genuinely open question (plus a state confirmation)

**(a) Is VM 101 `.177` DHCP or static?** `CLAUDE.md:68` says "DHCP — pin static
before relying on it" and `:150` repeats it. Determine the actual configured method
and report it; do not guess.
```bash
nmcli -t -f GENERAL.DEVICE,IP4.ADDRESS,ipv4.method device show 2>/dev/null \
  || cat /etc/netplan/*.yaml 2>/dev/null
```

**(b) Confirm the boot state has not changed** since §1 was recorded (rigs could have
been rebooted to Ubuntu in the interim):
```bash
for ip in 192.168.3.120 192.168.3.154 192.168.3.162 \
          192.168.3.122 192.168.3.156 192.168.3.164; do
  if ping -c1 -W2 "$ip" >/dev/null 2>&1; then echo "UP   $ip"; else echo "DOWN $ip"; fi
done
```
Expected: bare-metal set DOWN, CT set UP. **A DOWN address in either set is a boot
state, not a defect** — never report it as an error. If the result contradicts §1,
say so and describe the actual state.

Record `git log --oneline -1` before edits.

## 4. Stage 2 — `CLAUDE.md` §3 (the actual defect)

Rewrite §3 using the boot-selector model already applied to Zeus in that section.
Required properties:

- Every machine presented as **default boot (bare Ubuntu, original IP)** *or*
  **Proxmox (host → VM/CT workload endpoint)**. Use the §0 table's content.
- **The `Migrated: yes/no` column is REMOVED** — it encodes the wrong model.
  "Proxmox is available and running on this rig" is not "this rig exclusively runs
  Proxmox."
- Explicit statement that **the worker endpoint depends on which target is booted**,
  and that `distributed_config.json` currently holds the **bare-metal** addresses
  because those match the default boot target.
- **Current state (2026-07-22): all three rigs are booted into Proxmox**, so the
  config's bare-metal addresses are not reachable in the present state. State this
  as a point-in-time observation, clearly distinguished from the *default* boot.
- Note that selectable endpoints (`rig_profile`) are **Phase 6 prep, not done.**

Line edits in the same section:

- **Key auth line (~87):** replace "michael→CT100 key auth is NOT yet set up (open
  Phase-3 item)" with **verified working to all three CTs (2026-07-22)**. Keep the
  note that the checked-in `zeus-proxmox-build/dotfiles/ssh_config` is stale for
  migrated rigs, unless you verify otherwise (do not edit that file — §7).
- **GPU line (~71):** current text (`Windows VM 100 | .138 | RDP only; holds the 2nd
  3080 Ti`) reads as though the second card is permanently taken. Reword per §1: host
  console on a dedicated 1660 Ti; VM 100 holds one 3080 Ti *while running*; the second
  3080 Ti is free to pass through to VM 101 for rig testing.
- **VM 101 addressing (~68, ~150):** update per Stage 1(a). If static, say so and drop
  the "pin static" to-do; if still DHCP, keep it.
- Add a one-line note that the rigs are **identical clones**, so shared SSH host-key
  fingerprints between CTs are expected.

Update the section's "verified" stamp to **2026-07-22** with a one-line basis naming
what was checked (ping sweep, key auth, `pct config`, Proxmox UI) and what was not.

## 5. Stage 3 — Living reference docs: ADD context, CHANGE NO IPs

These are living reference (not history) and cite bare-metal IPs. Those IPs are
**correct for the default boot target** — **do not change them.** Add a short
boot-selector note near the first topology reference in each, pointing at
`CLAUDE.md` §3 as authority:

| File | First topology reference |
|---|---|
| `docs/COMPLETE_OPERATING_GUIDE_v2_0.md` | ~line 70 (`IP Addresses:` list) |
| `docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md` | ~lines 594, 642 (config sample / node loop) |
| `docs/S172_INFRASTRUCTURE_INTERFACE_v1_0.md` | ~line 26 (coordinator-visible IP row) |

Suggested note (adapt per file):

> **Boot-selector note:** each rig is dual-boot. The addresses shown are the
> **default** bare-Ubuntu endpoints. When a rig is booted into Proxmox instead, its
> worker endpoint is the CT100 address (`.122`/`.156`/`.164`). As of 2026-07-22 all
> three rigs are running under Proxmox. See `CLAUDE.md` §3.

Do **not** rewrite example commands, config samples, or output transcripts — they
illustrate the default target and remain valid.

## 6. Stage 4 — Prove no behavioral change, then changelog

1. **Scope check:**
   ```bash
   git status --short
   git diff --stat
   ```
   The changed-file list must contain **only `.md` files**. Any `.py`, `.sh`, or
   `.json` = scope violation: revert it and report.
2. **No-IP-substitution check:**
   ```bash
   git diff -U0 -- '*.md' | grep -E '^[-+].*192\.168\.3\.' | sort
   ```
   Every `-`/`+` must be additive context. A line removing `.120` and adding `.122`
   (or similar) is a scope violation.
3. Write `docs/SESSION_CHANGELOG_YYYYMMDD_SNNN.md` (next session number after the
   latest in `docs/`) recording: the boot-selector model; **why the proposed IP swap
   was REJECTED** (dual-boot; the swap would hardcode one mode and break the other);
   the §1 verified facts with their evidence; Stage 1 results; files touched; pre-edit
   HEAD; and open items — explicitly listing **`rig_profile` selectable endpoints as
   Phase 6 prep (not done)** and **2nd 3080 Ti passthrough to VM 101 (not done)**.
4. **STOP and report.** Michael reviews, commits, dual-pushes.

## 7. Out of scope (do not touch)

- Any `.py`, `.sh`, or `.json` — including `distributed_config.json`,
  `ml_coordinator_config.json`, `scripts_coordinator.py`, `launch_s174.py`,
  `s172_prelaunch_check.py`, `chunk_size_config.py`,
  `steps/step2_execution_manager.py`, `zmq_sqlite_coordinator.py`,
  `web_dashboard.py`, and every `*_health*.sh` / `deploy_*.sh`.
- `zmq_sqlite_coordinator.py:_get_zeus_ip()` — it uses a **UDP** socket, so
  `connect()` sends no packets and the target need not exist or be reachable. It is a
  local-IP routing lookup, not a connection. **Not a defect; leave it.**
- `~/.ssh/config` on VM 101 and `zeus-proxmox-build/dotfiles/ssh_config` — report if
  stale, do not edit.
- The rigs, CTs, and Proxmox hosts themselves — PINNED AND FROZEN. This session edits
  documentation in the repo tree on VM 101 only.
- Any `rig_profile` / selectable-endpoint implementation — **Phase 6 prep.**
- GPU-count "verification" — ROCm functionality is already proven (llama RPC across
  24 GPUs). Do not add device-count checks.
