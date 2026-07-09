# RUNBOOK: Proxmox VE + amdgpu-dkms Install for RX 6600 Rigs
**Version:** 1.1
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
21. Match the rigs' tuning (gfxoff must be modprobe.d — kernel cmdline is
    silently ignored per TB incident report):
    ```
    echo "options amdgpu gfxoff=0 ppfeaturemask=0xffff7fff" > /etc/modprobe.d/amdgpu.conf
    update-initramfs -u
    ```
    (Firmware warnings for Vega/Navi12/Aldebaran/etc. are harmless — that's
    firmware for GPUs we don't have. Navi 23 firmware is present.)
22. Pin the driver so it never auto-upgrades (matches rig hold policy):
    ```
    apt-mark hold amdgpu-dkms amdgpu-install
    ```
23. Reboot and re-run step 19 checks to confirm amdgpu auto-loads.

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

### 9.2 Bind the GPU devices (append to /etc/pve/lxc/100.conf)
```
lxc.cgroup2.devices.allow: c 226:* rwm       # DRI (card* + renderD*)
lxc.cgroup2.devices.allow: c <KFD_MAJOR>:* rwm   # see GOTCHA 2
lxc.mount.entry: /dev/kfd dev/kfd none bind,optional,create=file
lxc.mount.entry: /dev/dri dev/dri none bind,optional,create=dir
```
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
pct exec 100 -- bash -c "HSA_OVERRIDE_GFX_VERSION=10.3.0 rocminfo | grep gfx"  # 8× gfx1030 agents
```

## 10. GOTCHAS discovered on rrig6600c (both bite the container)

**GOTCHA 1 — render group name mismatch.**
Host `/dev/kfd` + render nodes are group `render` (GID 104). Inside a fresh
Ubuntu 22.04 container, GID 104 maps to the name `ssl-cert`. ROCm refuses RW
access unless the user is in that GID. Symptom:
`Unable to open /dev/kfd read-write: Operation not permitted / root is not
member of "ssl-cert" group`. Quick fix: `usermod -aG ssl-cert root`. Proper
fix for production CT100: rename GID 104 to `render` inside the container so
it matches rig convention.

**GOTCHA 2 — kfd major number is NOT stable across boots.**
`/dev/kfd` major was 236 on one boot, 237 on the next. If the cgroup allow
rule hardcodes the wrong major, the container binds the device but ROCm gets
"Operation not permitted" on kfd. For a one-off test, check `ls -la /dev/kfd`
and use the current major. **For production CT100, do NOT hardcode** — use a
Proxmox lxc hookscript that reads the live kfd major at container start and
writes the correct `lxc.cgroup2.devices.allow` rule, OR migrate to the newer
`dev0:` passthrough syntax which resolves the device dynamically.

## 11. Userspace (the part backups save)

The DKMS driver rebuilds cleanly from AMD's repo (§5). The ROCm **userspace**
(CuPy-for-ROCm especially) does NOT — there is no pip wheel for cupy-rocm at
6.4.3, and building from source is the original "many failures" ordeal.
DO NOT rebuild it. Restore the rig backup tarball into the container:
`rig-6600b_rocm_env_*.tar.gz` → `/home/michael/rocm_env` in CT100, then
validate with a real CuPy matmul + a Step 1 job. Host provides the driver;
container provides this restored userspace.
