# Rig power control via Tuya cloud (Merkury WW119 strip): reference

Set up 2026-11-16. Updated 2026-11-18. Owner: Michael. Written by Claude (Team Alpha lead). No secrets are recorded in this document.

Repo copy of the claude.ai project doc `claude/RIG_POWER_CONTROL_TUYA_REFERENCE_20261116.md`, as of 2026-11-18.

## What it is

The three rigs and Zeus are plugged into one Merkury WW119 Smart Multi-Port Wall Tap (a Tuya device, in the SmartLife app). Each outlet can be read and switched through Tuya's cloud with the `tinytuya` Python library.

Route: VM101 or SER8 -> tinytuya -> Tuya Cloud -> strip. Needs Internet. Local (LAN) discovery of the strip did not work from SER8; cause unresolved, not needed for the cloud route.

## GPU power supplies are separate and switched by hand (owner, 2026-11-18)

- A rig outlet powers the rig's motherboard only. The GPUs are fed by a separate server PSU with breakout boards.
- **The GPU PSUs are switched by hand, by Michael, at the rigs. They can be OFF.** On 2026-11-18 they were off when Alpha switched the rig outlets on; the rigs booted with no power to the cards.
- **Rule: no bring-up until Michael says the GPU PSUs are on.** A rig that booted without GPU power needs a restart after the GPU PSU comes on before its cards show up.
- A true cold start of the cards means switching the GPU PSU off by hand.
- **Planned by Michael:** a 12 V signal wire from each rig's own PSU to the remote-on input of its GPU PSU, so the rig outlet alone turns the GPUs on and off. (He first considered a separate Tuya wifi switch.) When this is wired, record it here and drop the "wait for Michael" rule.

## Outlet map (read from the Tuya service 2026-11-16 15:56 PST, names as set in the app)

| Outlet name | Switch code | Who may switch it |
|---|---|---|
| rig-6600c | `switch_1` | Alpha, via the helper |
| rig-6600 | `switch_2` | Alpha, via the helper |
| Zeus | `switch_3` | NEVER by an agent. VM101 runs on Zeus |
| rig-6600b | `switch_4` | Alpha, via the helper |
| Merkury WW119 (main) | whole strip | NEVER by an agent |

**Map confirmed by real switch actions:** power cycles on 2026-11-16 (16:59 and 17:58 PST), bring-up on 2026-11-17, and on/off of all three rig outlets on 2026-11-18 (07:55 on, 08:18 off). Michael sees each switch action as a Telegram notification.

Other entries visible in Google Home: "Miner Fan" (no longer connected, ignore), "iHome WW117" (offline, unrelated).

## Identifiers (not secret)

- Device ID: `eb5c4919d81933a017qnfu`
- Tuya developer project: "Rig Power", Western America data center, tinytuya region `us`
- Phone app: SmartLife (also linked to Google Home and Alexa)

## Where things live

| Machine | Python environment | Config directory | Notes |
|---|---|---|---|
| SER8 | `~/smart-plugs` | `~` (`tinytuya.json`, `devices.json`) | Set up first; read test passed |
| VM101 | `~/smart-plugs` (tinytuya 1.21.1) | `~/tuya-config` (mode 700; files mode 600) | Read test passed 15:56 PST; helper lives here |

Credentials are in `tinytuya.json` (Access ID and Access Secret) and `devices.json` (local keys). They are outside the repo and must stay out of Git, agent prompts and chat.

## The helper on VM101: `~/tuya-config/rigpower.py`

The only way Alpha or an agent uses the plugs. Run it with `~/smart-plugs/bin/python`.

```
rigpower.py status              # read-only: name, switch code, ON/OFF for all four
rigpower.py cycle rig-6600c     # OFF, wait 12 s, ON, then status
rigpower.py on rig-6600b        # ON, then status
rigpower.py off rig-6600b       # OFF and stay off, then status (added 2026-11-18 at Michael's request)
```

Built-in safeguards:

- Accepts only `rig-6600`, `rig-6600b`, `rig-6600c`. Tested: `cycle Zeus`, `cycle switch_3` and `off zeus` are all refused.
- Before every action it re-reads the outlet names from the Tuya service and refuses if the name-to-switch map differs from the table above (for example after renaming or re-pairing).
- `off` refuses if the rig still answers on the network (its Ubuntu or Proxmox address pings). Shut the rig down cleanly first, then switch the outlet off.
- The version before the `off` command is kept as `~/tuya-config/rigpower.py.bak_20261118`.

## Rules of use

1. **Bring-up (`on`): only with Michael's explicit go for the bring-up itself, and only after he says the GPU PSUs are on.** Choosing a run option is not a go to power the rigs. (Alpha got this wrong on 2026-11-18.)
2. **`cycle`: only on a rig confirmed frozen** (host does not answer, kernel recorder shows the fault or goes silent).
3. **`off`: only after a clean shutdown** (`sudo -n poweroff` on the Ubuntu address, or `poweroff` on Proxmox), when Michael has asked for the rigs to be off.
4. Tell Michael which rig and when, each time.
5. After `on` or `cycle`: the rig boots Ubuntu; run `boot-proxmox` (`sudo -n grub-reboot "Proxmox VE" && sudo -n reboot`), then re-arm netconsole (`dmesg -n 7`), stop `rig-power-telemetry.service`, and `pct start 100`.
6. If the helper refuses or the cloud call fails, stop and message Michael; he taps the outlet in his app.

## Known behaviour

- **rig-6600 does not always start when its outlet comes on.** Seen on 2026-11-17 (needed a `cycle`) and again on 2026-11-18 (never answered after `on`; on that day the GPU PSUs were also off). Michael put the first case down to timing. If it does not answer about two minutes after `on`, `cycle` it once.

## Open items

- The API secret appeared in a setup screenshot on 2026-11-16: rotating it is optional hygiene (low risk; Michael's call). If rotated, update `tinytuya.json` on SER8 and VM101.
- Tuya's IoT Core free allowance can expire; when it does only cloud calls stop. Check quota and expiry.
- Record the 12 V signal-wire change here when Michael has wired it.
- Google Home web (home.google.com) is an alternative control path; not set up on VM101's browser and not needed while this works.
