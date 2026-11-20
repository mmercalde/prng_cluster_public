from mcp.server.fastmcp import FastMCP
import subprocess
import sys

mcp = FastMCP("cluster-status")

# Updated 2026-11-19 for the current layout: all three rigs dual-boot
# bare Ubuntu (fallback) and Proxmox (where runs happen). In Proxmox the
# workers live in container CT100 on each rig. Zeus runs VM101 (coordinator).
#
# Rules this file follows:
#   - No rocm-smi or any other GPU monitoring read. Card counts come from
#     the PCI bus only (lspci), which does not talk to the cards.
#   - Nothing here reboots a rig or stops a container while work is running.
#   - No power switching. Outlets are handled elsewhere (rigpower.py on VM101).

# --- Ping layer: raw IPs (ping does not read ~/.ssh/config) -------------
# Verified against SER8 ~/.ssh/config 2026-11-19.
# Scheme: Proxmox host = Ubuntu IP + 1, CT100 worker = Ubuntu IP + 2.
PING_HOSTS = {
    # Zeus
    "zeus (ubuntu)":     "192.168.3.127",
    "pve-zeus":          "192.168.3.128",
    "vm101":             "192.168.3.177",
    # rig-6600
    "rig6600 (ubuntu)":  "192.168.3.120",
    "pve-rig6600":       "192.168.3.121",
    "rig6600 CT100":     "192.168.3.122",
    # rig-6600b
    "rig6600b (ubuntu)": "192.168.3.154",
    "pve-rig6600b":      "192.168.3.155",
    "rig6600b CT100":    "192.168.3.156",
    # rig-6600c
    "rig6600c (ubuntu)": "192.168.3.162",
    "pve-rig6600c":      "192.168.3.163",
    "rig6600c CT100":    "192.168.3.164",
    # other
    "kamrui":            "192.168.3.152",
}

# --- SSH layer: aliases resolved by ~/.ssh/config -----------------------
# (user, key, port all come from the config -- IPs never hardcoded here)
# rig name -> (alias when booted into Ubuntu, alias when booted into Proxmox)
RIGS = {
    "rig6600":  ("rig6600",  "pve-rig6600"),
    "rig6600b": ("rig6600b", "pve-rig6600b"),
    "rig6600c": ("rig6600c", "pve-rig6600c"),
}
PVE_HOSTS = ["pve-zeus", "pve-rig6600", "pve-rig6600b", "pve-rig6600c"]
RIG_PVE_HOSTS = [pve for _, pve in RIGS.values()]
WORKER_CT = 100          # worker container on every rig
COORDINATOR_VM = 101     # coordinator VM on pve-zeus
VM101_ALIAS = "vm101"

SSH_OPTS = ["-o", "ConnectTimeout=5", "-o", "BatchMode=yes"]

# Process patterns that mean "work is running".
WORKER_PATTERN = "python.*(range_miner|worker)"
COORDINATOR_PATTERN = "window_optimizer.py|miner/range_miner|persistent_worker_coordinator"

CARD_COUNT_CMD = "lspci -d 1002: | grep -c VGA"


def _run(alias: str, cmd: str, timeout: int = 30):
    """Run cmd over ssh. Returns (reachable, output)."""
    try:
        r = subprocess.run(
            ["ssh", *SSH_OPTS, alias, cmd],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"(timed out after {timeout}s)"
    out = (r.stdout + r.stderr).strip()
    # ssh itself exits 255 when it cannot connect or log in
    if r.returncode == 255:
        return False, out or "(ssh could not connect)"
    return True, out if out else f"(exit code {r.returncode}, no output)"


def _ssh(alias: str, cmd: str, timeout: int = 30) -> str:
    return _run(alias, cmd, timeout)[1]


def _count(text: str):
    """First line of text as an int, or None if it is not a number."""
    try:
        return int(text.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def _rig_mode(rig: str):
    """Which system the rig is booted into: 'proxmox', 'ubuntu' or None."""
    ubuntu_alias, pve_alias = RIGS[rig]
    if _run(pve_alias, "true", timeout=10)[0]:
        return "proxmox"
    if _run(ubuntu_alias, "true", timeout=10)[0]:
        return "ubuntu"
    return None


def _ct_workers(pve_alias: str):
    """Worker processes inside CT100. Returns (ct_status, count or None)."""
    status = _ssh(pve_alias, f"pct status {WORKER_CT}", timeout=15)
    if "running" not in status:
        return status, 0
    out = _ssh(pve_alias,
               f"pct exec {WORKER_CT} -- pgrep -c -f '{WORKER_PATTERN}'",
               timeout=20)
    return status, _count(out)


def _coordinator_procs():
    """Coordinator processes on VM101. Returns count, or None if unknown."""
    ok, out = _run(VM101_ALIAS, f"pgrep -c -f '{COORDINATOR_PATTERN}'",
                   timeout=15)
    return _count(out) if ok else None


@mcp.tool()
def ping_rigs() -> str:
    """Ping every address in the TFM cluster (Ubuntu boots, Proxmox hosts,
    worker containers, VM101); report UP or DOWN for each. A rig answers
    on its Ubuntu address OR its Proxmox address, never both."""
    results = []
    for name, ip in PING_HOSTS.items():
        r = subprocess.run(
            ["ping", "-c", "1", "-W", "2", ip], capture_output=True,
        )
        status = "UP" if r.returncode == 0 else "DOWN"
        results.append(f"{name:18s} {ip:15s} {status}")
    return "\n".join(results)


@mcp.tool()
def gpu_status(rig: str) -> str:
    """Show a rig's state without touching the GPUs: which system it is
    booted into, how many AMD cards are on the PCI bus (expect 8), whether
    the worker container is running, and how many worker processes are
    running. Does NOT run rocm-smi. rig: rig6600, rig6600b, or rig6600c."""
    if rig not in RIGS:
        return f"Unknown rig '{rig}'. Options: {list(RIGS)}"
    ubuntu_alias, pve_alias = RIGS[rig]
    mode = _rig_mode(rig)
    if mode is None:
        return f"{rig}: not reachable on its Ubuntu or its Proxmox address"
    if mode == "ubuntu":
        cards = _ssh(ubuntu_alias, CARD_COUNT_CMD, timeout=15)
        return (f"{rig}: booted into bare Ubuntu (fallback), not Proxmox\n"
                f"AMD cards on bus: {cards}")
    cards = _ssh(pve_alias, CARD_COUNT_CMD, timeout=15)
    ct_status, workers = _ct_workers(pve_alias)
    workers_txt = "unknown" if workers is None else str(workers)
    return (f"{rig}: booted into Proxmox\n"
            f"AMD cards on bus: {cards}\n"
            f"CT{WORKER_CT}: {ct_status}\n"
            f"worker processes in CT{WORKER_CT}: {workers_txt}")


@mcp.tool()
def reboot_rig(rig: str) -> str:
    """Reboot a rig (rig6600, rig6600b, rig6600c). Refuses while worker
    processes are running on it.
    - Booted into Ubuntu: reboots INTO Proxmox.
    - Booted into Proxmox: plain reboot; the rig comes back in Ubuntu, so
      call this again to return it to Proxmox, then start CT100."""
    if rig not in RIGS:
        return f"Unknown rig '{rig}'. Options: {list(RIGS)}"
    ubuntu_alias, pve_alias = RIGS[rig]
    mode = _rig_mode(rig)
    if mode is None:
        return (f"{rig}: not reachable, no reboot issued. A frozen rig needs "
                f"an outlet power cycle, which this tool does not do.")
    if mode == "proxmox":
        _, workers = _ct_workers(pve_alias)
        if workers is None:
            return (f"REFUSED: could not tell whether workers are running "
                    f"on {rig}. No reboot issued.")
        if workers > 0:
            return (f"REFUSED: {workers} worker process(es) running in "
                    f"CT{WORKER_CT} on {rig}. No reboot issued.")
        _run(pve_alias, "reboot", timeout=15)
        return (f"Reboot issued to {rig} (Proxmox). It will come back in "
                f"Ubuntu; call reboot_rig again to return it to Proxmox, "
                f"then pve_start {pve_alias} {WORKER_CT}.")
    workers = _count(_ssh(ubuntu_alias, f"pgrep -c -f '{WORKER_PATTERN}'",
                          timeout=15))
    if workers is None:
        return (f"REFUSED: could not tell whether workers are running "
                f"on {rig}. No reboot issued.")
    if workers > 0:
        return (f"REFUSED: {workers} worker process(es) running on {rig} "
                f"(Ubuntu). No reboot issued.")
    ok, out = _run(ubuntu_alias,
                   'sudo -n grub-reboot "Proxmox VE" && sudo -n reboot',
                   timeout=15)
    if ok and out and "exit code" not in out and "closed" not in out.lower():
        return f"{rig} (Ubuntu) answered: {out}"
    return (f"Reboot into Proxmox issued to {rig}. Allow about two minutes, "
            f"then pve_start {pve_alias} {WORKER_CT}.")


@mcp.tool()
def pve_list(pve_host: str) -> str:
    """List all CTs and VMs (with status) on a Proxmox host.
    pve_host: pve-zeus, pve-rig6600, pve-rig6600b, or pve-rig6600c."""
    if pve_host not in PVE_HOSTS:
        return f"Unknown Proxmox host '{pve_host}'. Options: {PVE_HOSTS}"
    return _ssh(pve_host, "echo '== CTs =='; pct list; echo '== VMs =='; qm list")


@mcp.tool()
def pve_start(pve_host: str, vmid: int) -> str:
    """Start a CT or VM by vmid on a Proxmox host. Workers are CT 100 on
    each rig; the coordinator is VM 101 on pve-zeus.
    pve_host: pve-zeus, pve-rig6600, pve-rig6600b, or pve-rig6600c."""
    if pve_host not in PVE_HOSTS:
        return f"Unknown Proxmox host '{pve_host}'. Options: {PVE_HOSTS}"
    vmid = int(vmid)
    return _ssh(pve_host, f"pct start {vmid} 2>/dev/null || qm start {vmid}")


@mcp.tool()
def pve_stop(pve_host: str, vmid: int) -> str:
    """Gracefully shut down a CT or VM by vmid on a Proxmox host. Refuses
    to stop a rig's worker container while workers are running, and refuses
    to stop VM 101 while a coordinator is running on it."""
    if pve_host not in PVE_HOSTS:
        return f"Unknown Proxmox host '{pve_host}'. Options: {PVE_HOSTS}"
    vmid = int(vmid)
    if pve_host in RIG_PVE_HOSTS and vmid == WORKER_CT:
        _, workers = _ct_workers(pve_host)
        if workers is None:
            return (f"REFUSED: could not tell whether workers are running "
                    f"in CT{WORKER_CT} on {pve_host}. Nothing stopped.")
        if workers > 0:
            return (f"REFUSED: {workers} worker process(es) running in "
                    f"CT{WORKER_CT} on {pve_host}. Nothing stopped.")
    if pve_host == "pve-zeus" and vmid == COORDINATOR_VM:
        procs = _coordinator_procs()
        if procs is None:
            return ("REFUSED: could not reach VM101 to check for a running "
                    "coordinator. Nothing stopped.")
        if procs > 0:
            return (f"REFUSED: {procs} coordinator process(es) running on "
                    f"VM101. Nothing stopped.")
    return _ssh(pve_host, f"pct shutdown {vmid} 2>/dev/null || qm shutdown {vmid}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        # Read-only check from a terminal: python3 cluster_status.py --selftest
        print(ping_rigs())
        for _rig in RIGS:
            print()
            print(gpu_status(_rig))
        for _host in PVE_HOSTS:
            print()
            print(f"--- {_host} ---")
            print(pve_list(_host))
    else:
        mcp.run()
