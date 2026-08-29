# RUNBOOK — netconsole re-arm on power-on

**Status: STUB, RECONSTRUCTED — not yet executed end to end against a live rig.**
**Created 2026-08-28.** Written so power-on day is paste-and-go rather than rediscovery.

**Provenance of every command below.** There is no `install_netconsole_zeus.sh` in the tree and no
`modprobe netconsole` line recorded anywhere in `docs/`. The receiver half is read from committed
source (`netconsole_listener.py`). The **sender half is reconstructed from the observed 2026-08-22
`NCPROOF` packets**, which prove the shape that worked but not the exact command that produced it.
**Sender steps are therefore marked `[RECONSTRUCTED]` and must be corrected in place the first
time they are actually run.** See `docs/LEADS.md` L-3.

---

## 0. Facts this runbook depends on

| fact | value | source |
|---|---|---|
| receiver host | VM101 `192.168.3.177` | `netconsole_listener.py` binds `0.0.0.0` |
| receiver port | **6667** (UDP) — *not* the 6666 default | `netconsole_listener.py:16` |
| capture file | `logs/netconsole_all_rigs.log` (appended) | `netconsole_listener.py:14-15` |
| senders | the **Proxmox hosts** `.121` / `.155` / `.163` | 2026-08-22 packets came from these |
| **NOT** senders | the CT100s `.122` / `.156` / `.164` | unprivileged LXC: cannot load kernel modules or write `/sys/kernel/config` |
| access gap | **no root key auth from VM101 to `.121`** | recorded gap; sender steps need host console or an interactive root login |

**Why the sender is host-side.** netconsole is a kernel module writing to a UDP socket. An
unprivileged LXC container has no permission to `modprobe` or to write
`/sys/kernel/config/netconsole/`. Attempting any of §2 from inside a CT will fail, and that
failure is expected, not a defect.

---

## 1. Receiver — on VM101 (safe, non-destructive)

```bash
# Is it already armed? Expect one UNCONN row owned by python3.
ss -lunp | grep 6667

# If nothing is listening, start it. NEVER pipe to tail (buffers; a live run
# looks identical to a hang). nohup, never tmux.
cd /home/michael/distributed_prng_analysis
nohup python3 -u netconsole_listener.py >> logs/netconsole_listener.log 2>&1 &
echo $! > logs/netconsole_listener.pid       # record the PID; do not pkill by pattern

# Confirm
ss -lunp | grep 6667
tail -2 logs/netconsole_all_rigs.log          # expect a fresh "LISTENER STARTED" line
```

The listener appends, so history is preserved across restarts. `LISTENER STARTED` lines are
bookkeeping, not received packets — do not read them as sender liveness.

---

## 2. Sender — on EACH Proxmox host `.121`, `.155`, `.163`  `[RECONSTRUCTED]`

**Prerequisite:** the rig is powered on **and booted to Proxmox**. A plain reboot returns it as
bare Ubuntu on `.120`/`.154`/`.162`; `boot-proxmox` is required. **Never reboot a host without
asking.**

Run as root on the host (console or interactive login — key auth is not available):

```bash
# Identify the interface carrying 192.168.3.0/24 and the gateway MAC.
ip -4 addr show | grep 192.168.3
IFACE=$(ip -4 route get 192.168.3.177 | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1); exit}')
DSTMAC=$(ip neigh show 192.168.3.177 | awk '{print $5; exit}')
SRCIP=$(ip -4 route get 192.168.3.177 | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1); exit}')
echo "iface=$IFACE src=$SRCIP dstmac=$DSTMAC"

# Arm. Format: src-port@src-ip/dev,tgt-port@tgt-ip/tgt-mac
modprobe netconsole netconsole=6665@${SRCIP}/${IFACE},6667@192.168.3.177/${DSTMAC}

# Verify the module took
grep ^netconsole /proc/modules
```

**If `ip neigh` shows no entry for `.177`,** ping VM101 once first to populate the ARP cache —
netconsole needs a literal destination MAC and will not resolve one itself.

**Persistence across reboot is NOT configured and is NOT assumed.** The command above arms the
running kernel only. Whether the 2026-08-22 arming persisted is unknown; treat every power-on as
requiring §2 again until a persistence mechanism is deliberately installed and proven.

---

## 3. Prove it — the `NCPROOF` step (this is the acceptance gate)

This reproduces exactly what was observed working on 2026-08-22. **Do not skip it.** An unproven
sender is the ambiguity that cost the 2026-08-22 forensics a usable observer.

On each host:

```bash
echo "NCPROOF-$(hostname)" > /dev/kmsg
```

On VM101, confirm all three arrived — **and note the quoting lesson**: the 2026-08-22 capture
records the literal text `NC-TEST3 $(hostname) $(date +%T)`, i.e. that attempt was sent inside
single quotes and the shell never expanded it. Use double quotes, or `> /dev/kmsg` as above.

```bash
grep -a NCPROOF logs/netconsole_all_rigs.log | tail -5
```

**Expected — three lines, one per host, each naming its own hostname:**

```
<ts> 192.168.3.121: [ uptime] NCPROOF-pve-rig6600
<ts> 192.168.3.155: [ uptime] NCPROOF-pve-rig6600b
<ts> 192.168.3.163: [ uptime] NCPROOF-pve-rig6600c
```

**Fewer than three lines = that host's sender is NOT armed.** Do not start a run until all three
appear. Record the proof timestamps in the session changelog.

---

## 4. Reading a capture afterwards

```bash
# per-sender packet counts (the file can contain binary; -a is required)
grep -a -oE '192\.168\.3\.[0-9]+' logs/netconsole_all_rigs.log | sort | uniq -c | sort -rn

# real packets only, excluding listener bookkeeping
grep -a -E '192\.168\.3\.[0-9]+:' logs/netconsole_all_rigs.log | tail -40
```

**The bracketed number is kernel monotonic uptime, and it is load-bearing evidence.** If
wall-clock delta between two packets from one host equals their uptime delta, that host did not
reboot in between; a discontinuity means it did. Subtracting uptime from wall-clock gives the
implied boot time. This is what resolved L-3's residual question — it is worth doing routinely.

---

## 5. What this runbook does NOT cover

- **Persistence across reboot.** Not configured, not proven, deliberately out of scope here.
- **GPU power caps and off-host power telemetry** — the other two Run-4 infrastructure hardening
  items in the TB R2 ruling's post-commit step 3. Separate work, not netconsole.
- **Any claim that an armed netconsole will capture the next freeze.** The 2026-08-22 freeze
  produced **no** netconsole packet while the senders were armed. Arming buys the ability to
  distinguish "no event reached the wire" from "no observer" — it does not guarantee an event.

---

## 6. First-execution checklist (delete these lines once done)

- [ ] §2 executed against a live host; **correct the `[RECONSTRUCTED]` commands in place** to
      whatever actually worked, and drop the marker.
- [ ] Record whether arming survived a reboot; if not, decide persistence deliberately.
- [ ] Confirm `.155` behaves like `.121`/`.163` (it shut down 41 min earlier on 2026-08-22 —
      almost certainly operator ordering, but unverified).
- [ ] Promote this stub to a full runbook and cross-reference it from the skill.
