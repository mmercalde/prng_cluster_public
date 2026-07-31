# TEAM ALPHA → TEAM BETA — dataset lifecycle: four findings, rulings requested

**Re:** the runtime dataset provisioning contract (Beta Phase 6.0 ruling §5.1),
committed at `5ea5312`. While preparing the implementation, Alpha established that the
draw dataset is **not immutable**, which invalidates one field of the contract as
written and exposes three further gaps.

**Nothing has been implemented or changed.** All investigation was read-only.

---

## Finding A — a fixed `expected_sha256` cannot work *(contract defect, Alpha's)*

The contract lists `expected_sha256` as a manifest field, which assumes an immutable
dataset. Michael reports **two scraper variants — one appends, one rewrites — selected
by argument.** Both defeat a fixed digest:

- **append** — the digest changes on every scrape; a hardcoded value fails every run
  after the first, and the predictable "fix" is to delete the check.
- **rewrite** — worse: prior content is not guaranteed stable. A re-scrape may
  reorder, re-deduplicate, or correct an earlier draw, so today's file is not
  necessarily yesterday's plus one row.

**The invariant is not immutability; it is fleet consistency:**

> Every node participating in a run computed against **byte-identical** draw data, and
> the run records which version that was.

**Proposed correction — freeze at run start.** The parent computes the dataset digest
**once** at run start; that value enters the run manifest; every node must prove its
copy matches **that** digest **before dispatch**. The run proceeds on the frozen
version even if a scrape lands mid-run; the next run picks up the new data.

The failure prevented is concrete and expensive to diagnose: one rig on yesterday's
file and two on today's yields divergent survivor counts with **no signal anywhere in
the pipeline** explaining why — the shape of defect that would be misattributed to
ROCm or the coordinator. Phase 6.0's cross-platform comparison would have been
meaningless under that condition.

## Finding B — no data-version provenance *(gap; out of §5.1 scope)*

The certified generation sidecar records `repository_commit` and
`repository_tree_clean` — **which code** produced the artifact. It records **nothing
about which data** the artifact was computed against.

Tolerable while the dataset is static. **Under daily autonomous scraping it becomes
permanent and irreversible:** a shelf of certified generations, each computed against a
different silently-superseded dataset version, none reproducible, with no way after the
fact to determine what any of them used.

The D6 release-grade artifact is byte-reproducible **only because `daily3.json` happens
not to have changed since** (mtime Mar 4). Nothing enforces or records that.

**Proposed correction:** record the frozen dataset digest in the generation sidecar
alongside `repository_commit`, pinning each artifact to a specific **code** version
*and* a specific **data** version.

**Alpha has not implemented this and does not propose folding it into provisioning.**
It changes the finalizer's provenance record — D3.5 territory, frozen — and quietly
widening a deliverable is the exact scope failure flagged on D6.1. Raised for Beta to
scope and sequence.

## Finding C — an enabled, boot-triggered scraper unit whose target does not exist

```
/etc/systemd/system/daily3scraper.service    enabled · boot-triggered · Restart=always
User=michael · WorkingDirectory=/home/michael/distributed_prng_analysis
enable symlink dated Sep 11 2025
```

**The target `run_daily3scraper.py` does not exist** — not on disk, not in
`git ls-files`, and never in git history (`git log --all` empty for that path). Per
boot the unit starts, fails with ENOENT, exits 2, and restarts 5× until systemd's
start limit stops it. The journal is persistent and retains 100 boots
(2026-05-02 → 2026-07-30): **3,297 lines, seven distinct message shapes, all variants
of that loop, no successful execution in the retained window.** Corroborated by draw
mtimes showing no automated writes (`daily3.json` Mar 4, `lottery_history.json`
Feb 15).

**Michael confirms this unit is intentional** — pre-wired infrastructure awaiting its
implementation, and a more capable `run_daily3scraper.py` is planned. It is therefore
**not proposed for removal.** Two properties Beta should nonetheless hold:

1. **Anything later placed at that exact path executes automatically at next boot,
   under `Restart=always`, with no repo-side review gate.** The path lives inside the
   working directory, so an ordinary file creation arms it.
2. **The unit is invisible to every repo-scoped audit** — which is exactly the gap that
   produced Alpha's initial incorrect finding (see D).

Alpha recommends the eventual `run_daily3scraper.py` be treated as a reviewed
deliverable with its own gates, not a convenience script — it will be the only
component in the system that executes autonomously at boot without human invocation.

## Finding D — repo-scoped audits cannot answer system-scoped questions *(process)*

Alpha initially reported **PASS — no scraper invoker exists.** That was wrong. The
repo evidence was correct; the **scope** was not. Alpha searched a clone and answered a
question about the system.

At Michael's insistence a live-host search was run. It **confirmed all five of Alpha's
repo-side claims** and then found Finding C outside the tree.

**Generalisation:** the repository is not the system. Provisioning state — systemd
units, cron, host configuration, deployed-but-uncommitted files — is invisible to every
gate this project has built, all of which read git. A repo-scoped answer to a
system-scoped question must be labelled partial. **This is VIR-5 applied to auditing
itself: unsearched is `UNAVAILABLE`, not clean.**

Alpha proposes adding to the verification standard:

> **VIR-6 — Scope declaration.** Any audit answering an existential question ("does
> anything invoke X", "is anything scheduled", "does any consumer depend on Y") must
> declare the surfaces it searched **and** those it could not. A repository-scoped
> search may not be reported as a system-scoped result. Host provisioning, scheduler
> wiring, and deployed-but-uncommitted files are distinct surfaces and must be named
> explicitly as searched or `UNAVAILABLE`.

## Confirmed clean (live-host search, exhaustive)

Repo: no module writes `daily3.json` (30+ hits all read-only); `pa_pick3_scraper.py`
has no callers outside docs; `draw_ingestion_daemon.py` is a filesystem observer with
no network imports, appending only to `lottery_history.json`; Chapter 13's
`STEP_SCRIPTS` is a closed six-entry map; **a second byte-identical `STEP_SCRIPTS` map
exists at `watcher_agent.py:314-321`** (consumed at `:1158`) — also closed, also no
scraper; the `agent_manifests/` string-keyed script indirection
(`manifest_loader.py:98-111` → `command_builder.py:61`) was enumerated exhaustively —
nine scripts, no scraper; WATCHER's dispatch is a closed three-way route
(`selfplay_retrain | learning_loop | pipeline_rerun`) with no scrape verb and no
data-driven extension; all 13 GBNF grammars contain no scrape/fetch/ingest token;
**Chapter 14 code does not exist** (docs only); all 298 subprocess/exec/import call
sites checked; no WATCHER scheduler thread exists.

Host: no crontab for `michael` or root; `/etc/cron.*` stock only; 16 stock timers, no
user timers; 117 enabled units of which three are project-related
(`daily3scraper.service`, `netconsole-listener.service`, `cluster-boot-notify.service`);
no autostart; no shell-profile hooks; no scraper process running.

`/home/michael/daily3_scraper.py` **does exist** (4,334 B, Sep 7 2025) alongside
`fantasy5_scraper.py` and a `cluster_controller/` pair — none invoked by anything; the
only cross-reference is `preprocess_daily3.py:10` printing an instruction for a human
to run it.

## UNAVAILABLE — unsearched, not clean (VIR-5)

Journal before 2026-05-02 (unit predates retention by ~8 months — whether
`run_daily3scraper.py` ever existed and ran successfully cannot be stated); the three
rig CT100s and bare-metal `.127`, not swept for their own cron/timer wiring; Windows
VM 100, RDP-only; git history of other branches for arbitrary deleted invokers; and a
whole-file semantic read of all 741 `.py` files — the method was keyword plus
exhaustive call-site enumeration, so a caller naming a scraper through a fully computed
string with no keyword on the line would evade it (assessed low-probability given the
manifest and dispatch enumerations returned closed, but not zero).

## Rulings requested

1. **Approve the freeze-at-run-start correction** to the provisioning contract
   (replace fixed `expected_sha256` with a run-scoped frozen digest; retain
   fail-before-dispatch).
2. **Scope and sequence Finding B.** Alpha recommends a separate deliverable, plausibly
   near **D6.2** which already touches provenance and the finalizer; numbering and
   ordering deferred to Beta.
3. **Confirm whether data-version provenance is a Phase 7 prerequisite.** Alpha's view:
   yes — a multi-day autonomous soak is precisely where unrecorded dataset drift becomes
   unrecoverable, and it is the first context in which scraping actually runs daily.
4. **Rule on `run_daily3scraper.py`'s status** — Alpha recommends a reviewed deliverable
   with gates, given it will execute at boot with `Restart=always` and no review gate.
5. **Adopt VIR-6** (scope declaration) or direct otherwise.

## Fleet status (for sequencing)

All three rigs up, key auth from VM 101, `rocm_env` present, 8 GPUs each:
`192.168.3.122 rrig6600` · `192.168.3.156 rrig6600b` · `192.168.3.164 rrig6600c`.
`daily3.json` is present only on `.122` (hand-copied during Phase 6.0), so the
provisioning implementation has three real nodes to verify against.
