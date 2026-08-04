# SER8_ARCHIVE_INVENTORY.md — REV1

**Completion sentinel: `UNAVAILABLE`.**

**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_SER8_ARCHIVE_INVENTORY.md` (REV1).
**Run from:** VM101 (`192.168.3.177`), `/home/michael/distributed_prng_analysis`, HEAD `b72c560`, tree clean.
**Date:** 2026-08-03.

---

# 0. RESULT IN ONE LINE

**ser8 is reachable from VM101 and is up, but VM101 holds no credential ser8 accepts. No
directory listing was obtained, no file was opened, and no inventory exists.** Per the brief's
own completion-sentinel rule — *"If the ser8 key does not work, the whole report is `UNAVAILABLE`
— say so and stop. Do not substitute guesses from filenames seen in a screenshot"* — this report
terminates here.

**Nothing below is an inventory.** §5–§8 record that each required output is unobtainable, and
preserve the frame so the next session can execute §3 without re-deriving it.

---

# 1. WHAT WAS ATTEMPTED — full access trace

Every command below was run on **VM101** as `michael`. All are read-only or connectivity probes.
**No write, move, delete or rename was attempted against ser8**, and no such command was
constructed.

| # | attempt | result |
|---|---|---|
| 1 | `ssh rser8` (the only ser8-ish alias in `~/.ssh/config` → `45.32.131.224:2005`) | `No route to host` — this is the **Vultr relay**, not a LAN path to ser8 |
| 2 | `ping -c2 192.168.1.229` (the address the brief §6 names) | 100% packet loss |
| 3 | `ssh michael@192.168.1.229` | `No route to host` |
| 4 | `/dev/tcp/192.168.1.229/22` | `No route to host` |
| 5 | `ip -4 addr` / `ip route` | VM101 has **one** interface, `192.168.3.177/24`, default gw `192.168.3.10`. **No route to `192.168.1.0/24` exists at all** |
| 6 | `getent hosts ser8.local` | **`192.168.3.24`** — mDNS resolves ser8 onto VM101's *own* subnet |
| 7 | `avahi-resolve -a 192.168.3.24` | **`SER8.local`** — reverse-confirms the identity |
| 8 | `ip neigh` | `192.168.3.24 … REACHABLE` — already live in the ARP table |
| 9 | `ssh michael@192.168.3.24` (agent + default keys) | `Permission denied (publickey,password)` |
| 10 | `ssh -i ~/.ssh/id_rsa -o IdentitiesOnly=yes michael@192.168.3.24` | `Permission denied (publickey,password)` |
| 11 | `ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes michael@192.168.3.24` | `Permission denied (publickey,password)` |
| 12 | `ssh -vv …` auth trace | Host is live: `OpenSSH_8.9p1 Ubuntu-3ubuntu0.16`. Both keys **offered and refused**. Server offers `publickey,password` only |
| 13 | `find ~ -maxdepth 3` for `*.pem`, `id_*`, `*ser8*`, `*readonly*`, `*_ro` | Only `id_rsa` / `id_ed25519` (both refused above) and unrelated syncthing certs. **No restricted read-only key exists on VM101** |

---

# 2. TWO FINDINGS ABOUT ACCESS (these are real, and they are not the inventory)

### 2.1 The address in the brief is stale — ser8 is on VM101's own subnet

The brief §6 and `docs/TELEGRAM_NOTIFICATION_SYSTEM_REFERENCE.md:227` both give ser8 as
**`192.168.1.229`**. VM101 has no interface, route or gateway path to `192.168.1.0/24`; that
address is unreachable by construction, not by outage.

ser8 answers on **`192.168.3.24`**, forward- and reverse-resolved by mDNS (`ser8.local` /
`SER8.local`), live in the ARP table, running OpenSSH 8.9p1.

**This is recorded, not corrected.** Per brief §6 (*no fixes*) and §0 (*a ser8 document is
presumed superseded, this is not an audit*), `TELEGRAM_NOTIFICATION_SYSTEM_REFERENCE.md` was
**not edited**. The address there is `N/A (workstation)` in a notification-coverage table, so
nothing executable reads it — but a future brief that copies it will fail the same way this one
did.

### 2.2 The credential gap is directional, and the direction that was never provisioned is this one

VM101's `~/.ssh/authorized_keys` carries **`michael@SER8`** (twice) and **`ser8-master-key`**.
So **ser8 → VM101 is provisioned and has been for a long time** — which matches every historical
reference in the repo, all of which describe ser8 as the *source* of an `scp` push
(`SESSION_81_HANDOFF.md:88`, `S111_IMPLEMENTATION_PLAN_FINAL.md:388`,
`DOCUMENTATION_UPDATES_S71.md:90`, and the working agreement in the skill §7).

**VM101 → ser8 has never been provisioned.** The brief's premise — *"reads ser8 over the
restricted read-only key"* — describes a credential that does not exist on this box. This is not
a key that stopped working; it is a key that was never installed here.

**The guard the brief describes (§1: restricted key, chrooted SFTP, read-only by construction)
therefore also does not exist.** That matters for §9: the shortest unblock is *not* the correct
one.

---

# 3. EXECUTION PROOF — the three counts the brief requires

The brief's VIR block requires three separate counts. All three are zero, and **zero here means
"the surface was never listed", not "the surface was listed and held nothing":**

| count | value | meaning |
|---|---|---|
| files **matched** by §2 patterns | **0** | no directory listing was ever obtained |
| files **inventoried** (first ~40 lines read) | **0** | no file on ser8 was opened |
| files **skipped** by pattern | **0** | nothing to skip — nothing was enumerated |

**No file on ser8 was opened, in whole or in part. No media file was opened. No file outside the
§2 patterns was listed, counted or described** — the §1 privacy scope was never placed under
strain, because no listing was ever returned.

---

# 4. §3.1 INVENTORY

`UNAVAILABLE`. No rows. See §3.

---

# 5. §3.2 PRIORITY LIST — the frame, prepared and unanswerable

`docs/PIPELINE_BEHAVIOUR_MODEL.md` §17 was read at HEAD `b72c560`. It marks **six** individual
behaviours `INCOMPLETE` — a code anchor with no WHY found. That is the priority list, reproduced
verbatim below so the next session starts from it rather than re-deriving it.

**The ser8 column cannot be filled by this pass.** Every entry is `UNAVAILABLE` — *a required
verification was attempted and could not complete* — never `NOT_APPLICABLE`, and never a guess.

| # | behaviour with a code anchor and no WHY | anchor | §17 "where to look next" | ser8 candidate file |
|---|---|---|---|---|
| I-1 | `build_pools.py` defaults `--prng-type` to `xorshift32` and reads `results/multi_gpu_analysis_*.json`, not `survivors_with_scores.json` — a different lineage from Step 6 | `build_pools.py:169, 89` | predates the java_lcg focus; `PREDICTION_STRATEGIES_DOCUMENTED.md` + the S1xx changelogs | **`UNAVAILABLE`** |
| I-2 | `backtest_pools.py` is the only in-repo caller of `build_pools.py`/`evaluate_pools.py`; how the 20/100/300 pools are produced in production is unstated | `backtest_pools.py:98, 120` | Chapter 7's unread half (§7–§12); `NOTE_Step7_Not_Required_for_Autonomy.md` | **`UNAVAILABLE`** |
| I-3 | `prediction_generator.py` writes a second history copy beside the canonical output | `prediction_generator.py:945-951` | Chapter 7 §8 (Output Formats) — not read in that pass | **`UNAVAILABLE`** |
| I-4 | `full_scoring_worker.py`'s sequential path merges 6 fields where the batch path merges 18 — the defect is governed, but no document says why the sequential list was ever shorter | `full_scoring_worker.py:451-455` vs `survivor_scorer.py:772-782` | Chapter 4 is **unaudited**; the S1xx changelog for the sequential-fallback introduction | **`UNAVAILABLE`** |
| I-5 | `ml_coordinator_config.json` is tracked and names a 26-GPU fleet no mechanism in Beta's six-mechanism table references | `CHAPTER_3_ALIGNMENT_AUDIT.md` F9 | `FLEET_STATE_REQUIREMENTS_v1.md` covers six mechanisms and does not include this one | **`UNAVAILABLE`** |
| I-6 | `chapter_13_orchestrator.py` derives `run_id` as `f"step1_{prng}_{seed_start}"` for the downstream write-back; the identity convention is undocumented in anything opened there | `chapter_13_orchestrator.py:300-304` | Chapter 13 §14 (Outputs) and §18 — not read in that pass | **`UNAVAILABLE`** |

**Read the fourth column before waiting on ser8.** §17 names a specific in-repo destination for
**all six** — and for I-2, I-3, I-5 and I-6 that destination is a *tracked file at HEAD*, openable
on VM101 right now. Only I-1 and I-4 point partly at the changelog corpus. **ser8 is the surface
of last resort for these six, not the first one**, and this pass reached none of them either way.

**The behaviour model's own §17 preamble stands unchanged:** each of these is a statement about
*that* search, not about the repository — and the explanation very likely sits in a surface that
pass did not open. Those surfaces are the 168-file changelog corpus, eleven unaudited chapters,
`instructions.txt` (152K, opened only at its two skip anchors), `Cluster_operating_manual.txt`
(96K, unopened), the two PDFs and the `.docx` — **all reachable from VM101 today, and ser8 is
not.**

---

# 6. §4 THE CA DRAW-PROCEDURES PDF

**`UNAVAILABLE` — identity NOT confirmed.**

The brief asks for confirmation from **filename and size only**. Neither was obtained: no
directory listing was returned. The brief supplies a partial name fragment; **that fragment is
deliberately not repeated here as a finding**, because reporting it back would be exactly the
substitution the completion-sentinel rule forbids — a filename asserted from a prior description
rather than observed.

The repo's standing position is unchanged and unweakened by this pass: the PDF is
**citation-only** (`PROJECT_FILE_CATALOG.md` §7), the skill carries it as `UNAVAILABLE`, and
Chapter 2 §5.1 / Chapter 1 §3.1.2 cite a document the repo does not contain. **Still an open
item.**

---

# 7. §5 AMBIGUOUS NAMES — for Michael

**Empty.** This section exists to list filenames that *might* be project-related but were **not
opened** under the §1 privacy rule. No filename on ser8 was ever seen, so there is nothing to
rule on.

**This emptiness is not a clean bill.** Under VIR-5, unobservable is not clean.

---

# 8. §3.3 THE `apply_s*.py` PATCH CORPUS

**`UNAVAILABLE`.** Count, session-number range, and any comparison against the repo's own
one-shot patch corpus (`PROJECT_FILE_CATALOG.md` §4.8) all require a listing that was never
obtained. **No gap in the change record is claimed in either direction.**

---

# 9. WHAT UNBLOCKS THIS — Michael's decision, two options that are not equivalent

**Do not treat these as interchangeable.** They differ in exactly the property the brief's §1
was written to guarantee.

### Option A — provision the restricted read-only key the brief assumes (matches the design)

Create a dedicated keypair for this purpose, authorize it on ser8 **with restrictions**, and
place the private half on VM101. The brief's model is a restricted key plus chrooted SFTP, which
makes read-only a property of the *server*, not of the agent's good behaviour.

An `authorized_keys` entry of this shape on ser8 enforces it host-side:

```
restrict,command="/usr/lib/openssh/sftp-server -R",from="192.168.3.177" ssh-ed25519 AAAA…<new key> vm101-archive-ro
```

`-R` puts the SFTP subsystem itself in read-only mode; `restrict` disables port/agent/X11
forwarding and PTY allocation; `from=` pins the source. **A write is then refused by ser8**,
which is what "read-only by construction" means and what §1 promised.

### Option B — authorize VM101's existing key (fast, and it discards the guard)

Appending VM101's existing public key to ser8's `~/.ssh/authorized_keys`:

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIN6G6tt7Xwv5IXMB0U2LndW5e8tfOzPEIpKig2kdVETb michael@Michael
```

grants **unrestricted shell and full write access to ser8**, including everything §1 lists as
out of scope — the unrelated engineering projects, the personal files, the media. Read-only
would then rest entirely on an agent choosing not to write, on a host whose `~/Downloads/` the
brief describes as mixed with voicemails and personal photographs. **Recommend against.**

**Either option is Michael's to execute on ser8.** No key was generated, no `authorized_keys`
line was written, and nothing was installed anywhere by this pass.

### Also worth settling regardless

The address. `192.168.1.229` is unreachable from VM101 by routing, not by outage, and it is what
both the brief and `TELEGRAM_NOTIFICATION_SYSTEM_REFERENCE.md:227` carry. `192.168.3.24` is
where ser8 actually answers. Whether that is a re-IP, a second NIC, or a documentation error
predating a network change is **not established here** — only that the working address today is
`192.168.3.24`.

---

# 10. VERIFICATION-INTEGRITY CONTROLS (VIR-1…6)

- **execution proof:** the thirteen-step access trace in §1, each row an actual command run on
  VM101 this session with its actual output. The three coverage counts are in §3: **0 matched /
  0 inventoried / 0 skipped**, with the explicit reading that zero means *never enumerated*, not
  *enumerated and empty*.
- **clean control:** `NOT_APPLICABLE` — inventory, not a detector.
- **fault-injection control:** `NOT_APPLICABLE` — same reason.
- **completion sentinel:** **`UNAVAILABLE`.** The required surface was reachable at the network
  layer and refused at the authentication layer. Per VIR-1 an inaccessible surface is never a
  pass, and per VIR-3 only `PASS` accepts. **Nothing in this report may be cited as evidence
  about what ser8 does or does not contain.**
- **unavailable-observer behaviour:** the observer could not observe, and says so. **No filename,
  count, size, subject line or verdict was inferred from the brief's own description, from a
  screenshot, or from any prior session's account of ser8.** §6 declines to echo back a filename
  fragment it was handed, for exactly this reason.
- **audit claim scope:** ser8 reachability and authentication from VM101 only. **No claim is made
  about ser8's contents.** The repo was read at HEAD `b72c560` only to reproduce the §17
  `INCOMPLETE` frame in §5.
- **searched surfaces:** VM101 network configuration (`ip addr`, `ip route`, `ip neigh`), mDNS
  (`getent`, `avahi-resolve`), `~/.ssh/` in full, a bounded `find` over `~` for private-key
  material, `~/.bash_history`, and repo-wide `/bin/grep` for ser8 references.
  `docs/PIPELINE_BEHAVIOUR_MODEL.md` §17 was read at `b72c560`.
- **unavailable surfaces:** **ser8 in its entirety** — `~/Downloads/` (pattern-scoped) and
  `~/Downloads/PRNG/`, both **never listed**. `docs/PROJECT_FILE_CATALOG.md` was **not** consulted
  for "in git?" determinations, because there were no candidate files to determine anything
  about. Everything on ser8 outside §2's patterns remains deliberately unsearched, as it should
  be.

---

**Tree impact:** this file is untracked and dirties the working tree. Per the brief, **it must be
committed before the Phase-7 soak launches, or before a running soak reaches publication** —
`Gate 22` reds on stray untracked files, and `run_finalizer.py:1589` runs a clean-tree check at
publication. `.gitignore:41` is `*.json`, so a `.md` deliverable is **not** exempt.

*Nothing was imported. Nothing was written to ser8. No commit, no push.*
