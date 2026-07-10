# RUNBOOK: Proxmox VE + amdgpu-dkms Install for RX 6600 Rigs
**Version:** 1.3
**Date:** 2026-07-08
**Verified on:** rrig6600c (first conversion)
**Applies to:** rrig6600, rrig6600b (identical hardware — Biostar TB360-BTC Pro 2.0, 8× RX 6600)
**Result achieved:** amdgpu-dkms 6.12.12 built + loaded on Proxmox VE 8.4 / kernel 6.8.12-9-pve; all 8 GPUs enumerated (`/dev/kfd` + renderD128–135).

---

## 0. Decisions locked in (from rrig6600c trial)

- **Proxmox VE 8.4-1**, NOT 9.x. Reason: 8.4 ships kernel 6.8, which is inside
  ROCm 6.4.3's supported range. 9.x ships kernel 7.0 (unverified with our
  gfx1032-spoofed stack). SHA256 of ISO:
  `d237d70ca48a9f6eb47f95fd4fd337722c3f69f8106393844d027d28c26523d8`
- **Driver = amdgpu-dkms 6.12.12** (matches the pinned rig version; fixes the
  8-concurrent-procs-per-GPU crash). Pulled from AMD's jammy repo — works on
  Debian bookworm despite the distro label.
- **IP scheme:** Proxmox host = rig IP + 1 (matches pve-zeus .127→.128).
  rrig6600 .120→**.130** · rrig6600b .154→**.164** · rrig6600c .162→**.163**.
- **Boot:** Proxmox on the ORICO (2nd drive); Ubuntu stays default via
  BIOS boot-order (TIMETEC first) + `boot-proxmox` GRUB one-shot from Ubuntu.

---

## 1. Drive prep (per rig)

1. Wipe the ORICO fully on a workstation (kills stale EFI that hijacks boot order):
   ```
   sudo wipefs -a /dev/sdX
   sudo sgdisk --zap-all /dev/sdX
   ```
   (Confirm sdX is the ORICO by size/model — 476.9G USB/SATA — never a system drive.)
2. Install the blank ORICO in the rig alongside the existing TIMETEC (Ubuntu).

## 2. Proxmox install (needs monitor — video notes below)

3. Write the 8.4 ISO to USB: `sudo dd if=proxmox-ve_8.4-1.iso of=/dev/sdX bs=4M status=progress oflag=sync`
4. **Video gotcha (Biostar mining board):** onboard HDMI is dead; use a
   monitor on **GPU #1**. The graphical AND terminal installers black-screen
   after selection because amdgpu mode-switches. The install still ran to
   completion once selected — but if it won't show, edit the boot entry (`e`)
   and append `nomodeset` (or `nomodeset amdgpu.dc=0 video=efifb:off`).
   After install, the rig runs headless — never needs local video again.
5. Installer answers: filesystem **ext4** (NOT ZFS — protects 8 GB RAM);
   target disk = the **ORICO** (verify by model, not the TIMETEC);
   IP `<rig-ip + 1>/24`, gateway `192.168.3.10`, DNS `1.1.1.1`,
   hostname `pve-rig6600<x>.local`.
6. Pull the USB on reboot. Confirm web UI at `https://<host-ip>:8006/`.
7. Reassemble: TIMETEC to its M.2 slot; **BIOS boot order = TIMETEC first**,
   ORICO second. Boots Ubuntu by default.

## 3. boot-proxmox one-shot (run from Ubuntu, on-site first time)

8. ```
   sudo cp /etc/default/grub /etc/default/grub.backup
   echo 'GRUB_DISABLE_OS_PROBER=false' | sudo tee -a /etc/default/grub
   ```
9. os-prober will NOT find Proxmox (separate ESP). Use a direct chainload.
   Get the Proxmox ESP UUID: `sudo blkid /dev/sda2` (the ORICO's ~1G EFI part).
10. Append to `/etc/grub.d/40_custom` (substitute the real UUID):
    ```
    menuentry 'Proxmox VE' {
        insmod part_gpt
        insmod fat
        insmod chain
        search --no-floppy --fs-uuid --set=root <ESP-UUID>
        chainloader /EFI/proxmox/shimx64.efi
    }
    ```
11. `sudo update-grub` then verify: `grep "Proxmox VE" /boot/grub/grub.cfg`
12. Alias: `echo "alias boot-proxmox='sudo grub-reboot \"Proxmox VE\" && sudo reboot'" >> ~/.bashrc && source ~/.bashrc`
13. Test on-site: `boot-proxmox` → confirm host UI at `:8006` → reboot →
    confirm it auto-returns to Ubuntu. (GRUB_DEFAULT="saved" makes the
    one-shot revert cleanly.)

## 4. Proxmox repos (run as root on the Proxmox host)

14. Disable enterprise (no subscription), add free repo:
    ```
    sed -i 's/^deb/#deb/' /etc/apt/sources.list.d/pve-enterprise.list
    sed -i 's/^deb/#deb/' /etc/apt/sources.list.d/ceph.list 2>/dev/null
    echo "deb http://download.proxmox.com/debian/pve bookworm pve-no-subscription" > /etc/apt/sources.list.d/pve-no-subscription.list
    apt-get update
    ```

## 5. amdgpu-dkms driver (THE key step — run as root on Proxmox host)

15. Install kernel headers + toolchain:
    ```
    apt-get install -y proxmox-headers-$(uname -r) build-essential
    ```
16. Install AMD's installer .deb (copy from a rig or keep a copy in the backup repo):
    ```
    apt-get install -y /root/amdgpu-install_6.4.60403-1_all.deb
    ```
17. This adds AMD's jammy repos (`/etc/apt/sources.list.d/amdgpu.list` +
    `rocm.list`). They work on bookworm. Refresh + verify version:
    ```
    apt-get update
    apt-cache policy amdgpu-dkms      # candidate must be 1:6.12.12.60403-...
    ```
18. Build the DKMS module against the pve kernel:
    ```
    apt-get install -y amdgpu-dkms amdgpu-dkms-firmware
    dkms status                       # expect: amdgpu/6.12.12-... 6.8.12-9-pve: installed
    ```
19. Load + verify all 8 GPUs:
    ```
    modprobe amdgpu
    ls -la /dev/kfd                   # must exist
    ls /dev/dri/                      # card0 (Intel) + card1-8 + renderD128-135
    dmesg | grep -i amdgpu | tail     # rings init, no errors
    ```
    ("Cannot find any crtc or sizes" = harmless, no monitor on the card.)

## 6. Persist across reboot + rig tuning (run as root on Proxmox host)

20. Autoload at boot:
    ```
    echo "amdgpu" >> /etc/modules-load.d/amdgpu.conf
    ```
21. **CRITICAL — remove the installer's `nomodeset` AND set the rigs' proven
    kernel cmdline.** The Proxmox installer persists `nomodeset` (from the
    install-time video workaround) in `/etc/default/grub.d/installer.cfg`.
    `nomodeset` disables KMS, which cripples amdgpu's SMU telemetry sysfs →
    `cat /sys/class/hwmon/hwmonN/temp1_input` returns "Operation not
    permitted" for ALL cards (even as root on the host). This kills the
    rocm-smi failure canary (see §12). The rigs' bare-metal Ubuntu cmdline
    has NO nomodeset plus a full amdgpu tuning set; replicate it:
    ```
    # remove the installer's nomodeset injection
    mv /etc/default/grub.d/installer.cfg /root/installer.cfg.disabled
    # set the proven cmdline (copied from a working bare-metal rig's /proc/cmdline)
    sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="quiet"/GRUB_CMDLINE_LINUX_DEFAULT="quiet pcie_aspm=off pci=noaer amdgpu.ppfeaturemask=0xffff7fff amdgpu.gfxoff=0 amdgpu.runpm=0 amdgpu.aspm=0 pci=assign-busses,hpbussize=0x33 iommu=pt amdgpu.dc=0 amdgpu.lockup_timeout=30000 amdgpu.gpu_recovery=1"/' /etc/default/grub
    update-grub
    # verify: no nomodeset, params present
    grep -h "vmlinuz.*root" /boot/grub/grub.cfg | head -1
    ```
    Note: `amdgpu.runpm=0` (disables BACO runtime PM) is the key telemetry
    fix; `nomodeset` removal is the other half. Proxmox boots off the Intel
    iGPU, so nomodeset was never needed post-install. Since this uses GRUB
    (not systemd-boot on this ext4/UEFI install), `update-grub` is correct.
    The `ppfeaturemask`/`gfxoff` here on cmdline supersede the modprobe.d
    version in the next step (redundant but harmless).
22. (Optional/legacy) modprobe.d tuning — now redundant with step 21's
    cmdline, kept for reference:
    ```
    echo "options amdgpu gfxoff=0 ppfeaturemask=0xffff7fff" > /etc/modprobe.d/amdgpu.conf
    update-initramfs -u
    ```
    (Firmware warnings for Vega/Navi12/Aldebaran/etc. are harmless — that's
    firmware for GPUs we don't have. Navi 23 firmware is present.)
23. Pin the driver so it never auto-upgrades (matches rig hold policy):
    ```
    apt-mark hold amdgpu-dkms amdgpu-install
    ```
24. Reboot and verify: amdgpu auto-loads (§5 step 19) AND telemetry works:
    ```
    for h in 1 2 3 4 5 6 7 8; do echo -n "hwmon$h: "; cat /sys/class/hwmon/hwmon$h/temp1_input 2>&1; done
    ```
    Must show 8 real temps (e.g. 39000 = 39°C), NOT "Operation not permitted".
    (Local console may go black after nomodeset removal — expected, host is
    headless; judge success by network reachability + this temp check.)

---

## 7. Still TODO (not yet done on rrig6600c — LXC arm of trial)

- Build CT100 (rig-identity LXC), bind `/dev/kfd` + `/dev/dri/*` + `/dev/shm/prng`.
- Restore ROCm 6.4.3 **userspace** into the container from the rig backup
  tarball (`rig-6600b_rocm_env_*.tar.gz`) — userspace goes in the container,
  NOT on the Proxmox host. Host only provides the kernel driver (done above).
- `rocm-smi` inside CT100 must show 8× RX 6600.
- Run a miner-dispatched Step 1 job (RAM headroom is the 8 GB question).

## 8. Known open issues (from S172 docs — not blockers, but track)

- amdgpu-dkms has a host RSS leak (~5.4 GB/hr under load, ROCm #5915).
  In-kernel driver lacks the leak but can't handle 8 concurrent procs/GPU.
  Unresolved driver dilemma — carried over from bare metal, not caused by Proxmox.
- Cold-start 8-GPU power spike (ROCm #5238): keep warm-up/power-cap practice.

---

**Notes on the host/container boundary (reconciliation §8 item 4):**
The Proxmox HOST owns: amdgpu kernel driver, `/dev/kfd` + `/dev/dri` nodes,
firmware, modprobe tuning. The CONTAINER owns: ROCm 6.4.3 userspace, HIP
runtime, rocm_env venv, all HSA_OVERRIDE_GFX_VERSION=10.3.0 / HSA_ENABLE_SDMA=0
env vars (set per-worker). Keep this boundary clean across upgrades.

---

## 9. LXC container with GPU access (CT100) — VERIFIED on rrig6600c

Result achieved 2026-07-08: a privileged LXC saw all 8 GPUs and `rocminfo`
opened all 8 gfx1030 compute agents (the gfx1032->gfx1030 spoof) with RW
access, from inside the container.

### 9.1 Create the container
```
pveam update
pveam download local ubuntu-22.04-standard_22.04-1_amd64.tar.zst
pct create 100 local:vztmpl/ubuntu-22.04-standard_22.04-1_amd64.tar.zst \
  --hostname rrig6600c \       # MUST be the rig name (socket.gethostname() → coordinator identity)
  --cores 4 --memory 4096 --swap 512 \
  --rootfs local-lvm:20 \
  --net0 name=eth0,bridge=vmbr0,ip=dhcp \   # real CT100: use rig identity IP
  --unprivileged 0 \           # privileged = simplest reliable GPU passthrough
  --features nesting=1
```

### 9.2 Bind the GPU devices — USE THE PATH-BASED `dev0:` SYNTAX
Append to /etc/pve/lxc/100.conf. The `dev0:` syntax (Proxmox 8.2+) resolves
`/dev/kfd` BY PATH at container start, so it auto-adapts to the floating kfd
major (see GOTCHA 2) and sets the render group (fixes GOTCHA 1). Do NOT
hardcode the kfd major in a cgroup rule.
```
dev0: /dev/kfd,gid=104                              # 104 = render GID; path-based, self-resolving
lxc.cgroup2.devices.allow: c 226:* rwm             # DRI major 226 is STABLE, safe to hardcode
lxc.mount.entry: /dev/dri dev/dri none bind,optional,create=dir
```
Get the render GID with: `getent group render | cut -d: -f3` (was 104 here).
`pct stop 100 && pct start 100` after editing.

### 9.3 Prove GPU access inside the container
```
pct exec 100 -- ls /dev/dri/                 # card0-7 + renderD128-135 present
# install rocm-smi + rocminfo (test tools only; real workload uses rocm_env)
pct exec 100 -- bash -c "mkdir -p /etc/apt/keyrings && \
  wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor > /etc/apt/keyrings/rocm.gpg && \
  echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.4.3 jammy main' > /etc/apt/sources.list.d/rocm.list && \
  apt-get update -qq && apt-get install -y rocminfo rocm-smi-lib"
pct exec 100 -- /opt/rocm/bin/rocm-smi                       # lists 8 GPUs
pct exec 100 -- bash -c "HSA_OVERRIDE_GFX_VERSION=10.3.0 rocminfo | grep -c gfx1030"  # =16 (8 GPUs x2 lines)
```
Telemetry inside the CT: works automatically via the container's default /sys
mount once the HOST telemetry is fixed (§6 step 21). NO hwmon bind needed:
```
pct exec 100 -- bash -c "for h in 1 2 3 4 5 6 7 8; do echo -n \"hwmon\$h: \"; cat /sys/class/hwmon/hwmon\$h/temp1_input 2>&1; done"
```
Must show 8 real temps — that's the canary working inside the sandbox.

## 10. GOTCHAS discovered on rrig6600c (both now SOLVED by §9.2 dev0: syntax)

**GOTCHA 1 — render group name mismatch.**
Host `/dev/kfd` + render nodes are group `render` (GID 104). Inside a fresh
Ubuntu 22.04 container, GID 104 maps to the name `ssl-cert`. ROCm refuses RW
access unless the user is in that GID. Symptom:
`Unable to open /dev/kfd read-write: Operation not permitted / root is not
member of "ssl-cert" group`. SOLVED by `gid=104` in the `dev0:` line (§9.2),
which sets the correct group on the device. (Old manual fix was
`usermod -aG ssl-cert root`.)

**GOTCHA 2 — kfd major number is NOT stable across boots.**
`/dev/kfd` uses a dynamically-allocated major (seen as 236, 237, 238 on
successive boots — it depends on module load order). Hardcoding it in a
`lxc.cgroup2.devices.allow` rule means the container silently loses kfd access
whenever the major shifts. SOLVED by the path-based `dev0: /dev/kfd` syntax
(§9.2) — Proxmox resolves the device by path at each container start, so the
major can float freely. Verified across a host reboot: container still
enumerated all 8 GPUs with zero config change. The DRI major (226) IS stable,
so its cgroup rule is fine to hardcode.

## 11. Userspace (the part backups save)

The DKMS driver rebuilds cleanly from AMD's repo (§5). The ROCm **userspace**
(CuPy-for-ROCm especially) does NOT — there is no pip wheel for cupy-rocm at
6.4.3, and building from source is the original "many failures" ordeal.
DO NOT rebuild it. Restore the rig backup tarball into the container:
`rig-6600b_rocm_env_*.tar.gz` → `/home/michael/rocm_env` in CT100, then
validate with a real CuPy matmul + a Step 1 job. Host provides the driver;
container provides this restored userspace.

## 12. GPU telemetry canary (REQUIRED — hard-won, verified on rrig6600c)

On these rigs the `rocm-smi` "Expected integer value from monitor, but got
"" " error (and unreadable `temp1_input`) is a validated dual-purpose signal:
(1) a per-card hardware/SMU failure, and (2) — more commonly — feedback that
the running code pushed the cards too hard or skipped required due diligence.
This matters most for the agent-sandbox use case: an agent writing GPU code is
the most likely actor to trigger it, and it's the primary signal the code
mistreated the hardware.

Root cause of it being broken on a fresh Proxmox install: `nomodeset` (from
the installer video workaround) + missing amdgpu tuning params. Fixed by §6
step 21 (remove nomodeset, set `amdgpu.runpm=0` etc.). After that fix, a
healthy host shows real per-card temps.

**For CT100:** telemetry works AUTOMATICALLY inside the container via its
default read-only /sys mount, once the HOST fix (§6 step 21) is applied. NO
hwmon sysfs bind is needed — verified: with the host cmdline fixed, CT100's
rocm-smi/`cat temp1_input` showed real per-card temps with zero extra config.
NO separate monitoring system needed — running rocm-smi IS the check; a bad
card stands out because it shows the error while the other 7 show numbers.

**Debugging note (methodology that found the fix):** three wrong theories were
ruled out first — idle power state, "SMU driver if version not matched" dmesg
line (a red herring present on the WORKING rig too), and kernel lockdown (was
`[none]`). The cause was found by comparing `/proc/cmdline` on the broken
Proxmox host vs. a working bare-metal rig. Always compare against the known-
good rig before theorizing.
