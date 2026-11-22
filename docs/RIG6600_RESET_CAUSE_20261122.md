# rig-6600 resets, 2026-11-21: cause found by the owner

**Status: closed (owner, 2026-11-22).**

- **Symptom:** silent resets and a reboot loop on both Ubuntu and Proxmox while idle, boots lasting 4 to 7 minutes,
  no error in any log. Facts as captured: `docs/TFM_SESSION_HANDOFF_20261121.md` (Rigs) and its addendum.
- **Cause (owner, 2026-11-22 08:06):** a failed PSU breakout board. Owner: "Bad psu breakout board. I bought three of
  these and all three failed."
- **Fix:** the breakout board is out. At 08:27 the rig was running on a spare PSU; the owner's replacement is an
  Apevia SPIRIT600W Spirit 600W ATX power supply.
- **After the fix (single reads, no polling):** rig-6600 ran 18 minutes in Ubuntu with 8 of 8 cards (yesterday's
  failing boots lasted 4 to 7 minutes), then was moved to Proxmox at 08:24; CT100 running, 8 of 8 cards.
  rig-6600b and rig-6600c were brought up on Proxmox at 08:11-08:13, CT100 running, 8 of 8 cards each.
- **Not the cause:** TFM software, logging, PCIe link speed. The owner had ruled out PCIe speed on 2026-11-21.
- **Limit:** no load run has been made since the fix; the evidence is idle uptime only.
- **Lesson:** resets with no logged error on an idle rig point at power hardware first. Check the PSU breakout
  board before software.
- **Record gap:** what ended the July 2026 rig-6600 resets was not found in the repo, the project docs or the
  searchable chats.
