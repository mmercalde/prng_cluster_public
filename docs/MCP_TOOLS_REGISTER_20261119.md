# MCP tool register (started 2026-11-19)

Written by Claude (Team Alpha) at Michael's request, so there is one place that lists every MCP control around the cluster and nobody finds one by surprise. Add a row here whenever one is created, moved or plugged into another program.

An MCP server is a small program that gives an AI assistant a few actions it can take by itself. It does nothing until an assistant that has it configured calls it.

## Register

| Name | File | Machine | Plugged into | Used by TFM pipeline? |
|---|---|---|---|---|
| `cluster-status` | `~/mcp/cluster_status.py` (backup of the July version: `~/mcp/cluster_status.py.bak_20261119`) | SER8 | Claude desktop app on SER8 (`~/.config/Claude/claude_desktop_config.json`) | No |
| `freecad-zeus`, `freecad-local` | not read | SER8 | Claude desktop app on SER8 (same config file) | No (CAD, unrelated to TFM) |

Reference copy of `cluster_status.py` in this repo: `docs/drafts/ser8_mcp_cluster_status/cluster_status.py` (sha256 `687e982e28271943c4e873727520b4d5c2e192fed3d41f760a04ed547c21c90c`). The live file is the one on SER8; nothing runs the repo copy.

Checked 2026-11-19:

- The desktop app's config lists exactly three servers: `freecad-zeus`, `freecad-local`, `cluster-status` (output pasted by Michael).
- DeepSeek Harness on VM101 (`~/.dsh`, one profile `web`) has no MCP servers configured (read on VM101 by Claude).
- Nothing in the repo, on VM101 or in the project docs refers to `cluster_status.py`.
- A Claude Code session in `~/phone-agent-install` on SER8 read or edited the file at some point. It is not configured to run it.
- NOT checked: Claude Code's own MCP list on SER8 (`claude mcp list`), Zeus bare-metal, the rigs, the phone.

## `cluster-status`: what it does

Six actions. It reaches machines through the SSH names in SER8's `~/.ssh/config`.

| Action | What it does | Changes anything? |
|---|---|---|
| `ping_rigs` | Pings every address (Ubuntu boots, Proxmox hosts, CT100 containers, VM101) | No |
| `gpu_status` | One rig: which system it is booted into, AMD card count from the PCI bus, CT100 status, worker process count | No |
| `pve_list` | Lists containers and VMs on a Proxmox host | No |
| `pve_start` | Starts a container or VM | Yes |
| `pve_stop` | Shuts down a container or VM. Refuses a rig's CT100 while workers run; refuses VM101 while a coordinator runs or if VM101 cannot be checked | Yes |
| `reboot_rig` | Reboots a rig. Refuses while workers run. From Ubuntu it reboots into Proxmox; from Proxmox it is a plain reboot (rig returns in Ubuntu). Does nothing on an unreachable rig | Yes |

Rules built in: no `rocm-smi` or any other GPU monitoring read; no power switching (outlets are `rigpower.py` on VM101 only).

Read-only check from a terminal on SER8: `python3 ~/mcp/cluster_status.py --selftest`. Run by Michael 2026-11-19 20:26 PST: all three rigs in Proxmox, 8 cards, CT100 running, 0 workers; Zeus VM100 and VM101 running.

## History

- July 2026: built and deployed to SER8 (file comment: verified 2026-07-13). Old layout: no rig-6600 Proxmox host, GPU status via `rocm-smi` on the Ubuntu addresses, plain reboot.
- 2026-11-19: rewritten by Claude for the current layout, installed by Michael. Tested by Claude against simulated rigs only, then by Michael's self-test on SER8. The desktop app needs a restart to load it.

## Owner decisions (2026-11-19)

- Leave it as is: plugged into the Claude desktop app on SER8 only.
- Not wired into DeepSeek Harness. If that is ever done, Claude's suggestion is a read-only copy (ping, status, list).
- Claude (Team Alpha) does not use it for runs. Runs go through VM101's terminal, `rigpower.py` and SSH.
- Michael's stated aim: no "rats nest of mcp controls that no one knows about". Keep this register current.
