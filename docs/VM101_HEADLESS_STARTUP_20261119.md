# VM101 headless start-up (set up 2026-11-19)

Written by Claude (Team Alpha). What makes VM101 usable by Claude after a boot with nobody at the screen, why it was needed, and how to undo it. All files are on VM101, outside the repo.

## VM101 auto-start with Zeus (owner asked for this note, 2026-11-19)

**Goal: VM101 starts by itself when Zeus boots into Proxmox, the same as VM100.** One command on the Proxmox host (`ssh pve-zeus`), run by Michael because Claude cannot reach the Proxmox host from VM101:

```
qm set 101 --onboot 1
```

**Status: DONE on 2026-11-20 at 12:57 PST** (set by Michael; correction 2026-11-20). The earlier status line said it was done on 2026-11-19 at 20:33 PST; that was wrong, and the setting was not on then. Not verified by Claude: the config was not read, and auto-start has not yet been proven by a Zeus boot. Check with `qm config 101 | grep onboot`. UNVERIFIED: that VM100 starts by itself because it has this setting on; that is inferred from what Michael sees.

**Discard: enabled on 2026-11-20** (set by Michael), so blocks freed inside VM101 can be returned to the Proxmox thin pool. Not verified by Claude: which disk entry carries `discard=on`, and whether a trim (`fstrim`) has run since. Check on the Proxmox host with `qm config 101 | grep discard`.

## Routine

1. Zeus must be in Proxmox (`sudo boot-proxmox` from bare Ubuntu if needed).
2. VM101 should now start by itself with Proxmox (see above). If it does not, on the Proxmox host (`ssh pve-zeus`): `qm start 101`.
3. Wait about two minutes. VM101 logs in to its desktop automatically, unlocks the login keyring, starts the web terminal at `http://127.0.0.1:7681` and opens Chrome. Chrome's Claude extension then connects (browser deviceId `337d9ee9-34bf-4c11-b6a3-c487c96374e1`).

The start-up is tied to VM101 booting, not to an SSH login. Nobody needs to SSH into VM101.

Proven on 2026-11-19 at 13:01 PST: reboot, no SSH login, no typing; `/tmp/kr_unlock.log` said `UNLOCKED` and the browser connected.

## Why it was needed (measured on 2026-11-19)

- The desktop logs in automatically, so no password is entered, so the GNOME login keyring stays locked (`org.freedesktop.Secret.Collection Locked` returned `true`).
- Chrome waits behind the keyring's on-screen unlock box. Its window title stays "Untitled", no page loads, and the Claude extension never connects.
- Starting Chrome from SSH did not help for the same reason.
- `gnome-keyring-daemon --replace --unlock` did NOT unlock the existing keyring here, and replacing the daemon mid-session left Chrome asking a different service. Do not use it. A reboot cleared that state.

## What was installed

| File | Purpose |
|---|---|
| `~/.config/autostart/claude-vm101.desktop` | Desktop start-up entry. `Exec=bash -c "$HOME/bin/vm101_startup.sh"` |
| `~/bin/vm101_startup.sh` | Waits for the keyring control socket, runs the unlock, starts `~/bin/claude_terminal_vm101.sh`, waits 5 s, opens Chrome on the terminal page. Logs: `/tmp/kr_unlock.log`, `/tmp/ct.log` |
| `~/bin/kr_unlock.py` | 9 lines. Sends the password to the keyring's control socket (`/run/user/<uid>/keyring/control`), the same route a password login uses. Prints `UNLOCKED`, `DENIED` or `FAILED`. Reads the password from standard input, or asks for it if run by hand |
| `~/.config/kr_pass` | Michael's VM101 login password, mode 600, created by Michael. Claude has not seen it and must not read it |
| `~/claude-vm101.desktop.bak_20261119` | The first version of the start-up entry (no unlock step) |

## Owner decision and trade-off

Michael chose this (option A) over giving the keyring an empty password (option B). Either way, anyone who gets into the `michael` account on VM101 can reach the saved passwords.

## Maintenance

- If the VM101 login password changes, Michael re-creates the file:
  `(umask 077; IFS= read -rsp "Password: " P; printf '%s' "$P" > ~/.config/kr_pass; echo)`
  then checks `ls -l ~/.config/kr_pass` shows `-rw-------` and a size above 0.
- To unlock by hand from SSH: `python3 ~/bin/kr_unlock.py`, then restart Chrome:
  `pkill chrome; sleep 2; DISPLAY=:0 nohup google-chrome http://127.0.0.1:7681 >/dev/null 2>&1 &`
- To check the keyring from SSH:
  `export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$(id -u)/bus; gdbus call --session --dest org.freedesktop.secrets --object-path /org/freedesktop/secrets/collection/login --method org.freedesktop.DBus.Properties.Get org.freedesktop.Secret.Collection Locked`
- To undo everything: `rm ~/.config/autostart/claude-vm101.desktop ~/.config/kr_pass`.

## Still manual after a reboot

- tmux session `tfm` starts with one shell window. Agents are restarted with
  `tmux new-window -d -n inv -c /home/michael/tfm_worktrees/rig-gpu-fault 'claude --model opus --continue'`.
- `source ~/venvs/torch/bin/activate` and `export RUN4_PARAMS=...` in window 0 before any run.
- `cp ~/lastmsg.py /tmp/lastmsg.py`.

## Notes

- Several Chrome browsers can be connected to the account at once. VM101 is the one with the deviceId above; confirm with `hostname` (`zeus-ubuntu-vm`) on the terminal page before doing anything.
- The `connectedAt` time in the browser list is not reliable evidence of which machine a browser is on.
- Sunshine runs on VM101, so Moonlight can show the desktop if the screen is ever needed.
