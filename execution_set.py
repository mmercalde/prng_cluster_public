"""S172 — the RESOLVED EXECUTION SET: one frozen fleet authority per run.

Authority: `docs/CLAUDE_CODE_INSTRUCTIONS_RESOLVED_EXECUTION_SET.md` (Team Beta
fleet ruling), evidence: `docs/FLEET_STATE_REQUIREMENTS_v1.md`.

THE PROBLEM THIS EXISTS TO REMOVE
---------------------------------
There was no single required fleet state. SIX mechanisms decided independently,
at three granularities, across two disjoint address sets, and which ones applied
depended on the backend flag and on whether the run came through WATCHER or the
CLI:

  P0.5 dataset preflight   node        .122/.156/.164  (CT100)
  legacy test_connectivity node        .120/.154/.162  (bare metal)
  PWC ready gate           GPU         .120/.154/.162
  WATCHER GPU health       GPU         .120/.154/.162  (non-blocking by design)
  boot notify              GPU         host-local, Telegram-only, exit 0
  miner expected_workers   worker      whoever connects

Three point at bare metal, one at the CT100s, two name no fixed set at all. The
rigs are booted into Proxmox, so P0.5 passes and the three bare-metal checks
*structurally cannot*. Beta ruled that **none of the six defines the fleet** and
that a single run-scoped resolved set must — with all six becoming CONSUMERS.
None of them is deleted here. They are re-pointed.

THE ONE IDEA
------------
A run resolves the execution set **once**, after backend and rig-profile
selection and **before** dataset verification, GPU verification, coordinator
construction or dispatch, and then FREEZES it. Every later consumer reads that
frozen value. A topology change, a config edit or a worker that dials in late
cannot alter a run in progress, because nothing in the run re-resolves.

This is deliberately the same shape as the P0.5 dataset freeze
(`miner/dataset_authority.py`): resolve once at run start, freeze, and let every
consumer read the freeze rather than the mutable source. The fleet definition now
gets the treatment the dataset identity already got.

WHAT IS EXPLICIT AND WHAT IS NEVER INFERRED
-------------------------------------------
Beta, verbatim:

  > A partial set must be explicit and frozen before the run — never inferred
  > from which workers happened to answer.

and, on the miner's registration path:

  > unknown miner workers must not become eligible merely because they connected.

So: this module **never probes anything** to decide membership. It reads declared
config (`rig_profiles_config.json` joined against `distributed_config.json` and
`dataset_provisioning.json`), takes an explicit node selection if one is given,
and produces a closed set. Reachability is then the CONSUMERS' problem — and a
declared node that does not answer is a loud failure, which is the point. A
worker outside the set that connects anyway is registered-but-ineligible
(quarantined), never silently admitted.

WHAT IS DELIBERATELY NOT CHANGED
--------------------------------
`distributed_config.json`'s bare-metal addresses (deliberate, CLAUDE.md §3);
the `worker_admission_timeout` / `serve_timeout=None` split from `ee0db06`;
`expected_workers` and `worker_pool_size` semantics; the Blocker-3 matrix;
WATCHER GPU health staying non-blocking; boot notify staying Telegram-only and
`exit 0`; P0.5's `UNAVAILABLE` / `NOT_APPLICABLE` vocabulary, in which
UNAVAILABLE means *a required verification was attempted and could not complete*
and an unknown topology keeps the over-constrained reading.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

RESOLVER_VERSION = 1

PROFILE_MAP_NAME = "rig_profiles_config.json"
DISTRIBUTED_CONFIG_NAME = "distributed_config.json"
PROVISIONING_MANIFEST_NAME = "dataset_provisioning.json"

SUPPORTED_PROFILE_MAP_SCHEMA_VERSIONS = frozenset({1})

#: The backends the mutex in `window_optimizer.main()` chooses between, plus the
#: default legacy path. The resolved set records which one it was resolved FOR,
#: because the answer to "what must be verified" is not the same for all of them.
BACKEND_MINER = "miner"
BACKEND_PWC = "pwc"
BACKEND_ZMQ = "zmq"
BACKEND_LEGACY = "legacy"
VALID_BACKENDS = frozenset({BACKEND_MINER, BACKEND_PWC, BACKEND_ZMQ, BACKEND_LEGACY})

PROFILE_BAREMETAL = "baremetal"
PROFILE_PROXMOX = "proxmox"


class ExecutionSetError(RuntimeError):
    """The execution set could not be resolved, or was used incoherently.

    Deliberately a hard error on every path. A run that cannot establish WHICH
    machines it is running on is in exactly the position P0.5 refused for the
    dataset: it cannot verify what it cannot name.
    """


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


# ===========================================================================
# The resolved set
# ===========================================================================

@dataclass(frozen=True)
class ResolvedNode:
    """One logical node, resolved to ONE endpoint for this run.

    `endpoint` is already profile-selected: a consumer reads it and never has to
    know that a second address for this machine exists. That is the whole
    mechanism by which two disjoint address sets stop producing two disjoint
    verdicts.
    """
    node_id: str                    # logical identity: rrig6600, localhost
    endpoint: str                   # THIS run's address for it
    config_hostname: str            # its key in distributed_config.json
    worker_hostname: str            # what socket.gethostname() reports there
    ssh_user: str
    local: bool
    gpu_count: int
    gpu_type: str
    rig_profile: str

    def gpu_identities(self) -> Tuple[str, ...]:
        """The per-GPU identities this node contributes: `<worker_hostname>:gpu<n>`.

        Identical in form to the miner's own `worker_id`
        (`miner/range_miner_worker.py`: f"{socket.gethostname()}:gpu{gpu_id}"),
        because it must be comparable to it without translation — a membership
        test that needs a mapping layer is a membership test that will drift.
        """
        return tuple(f"{self.worker_hostname}:gpu{i}" for i in range(self.gpu_count))

    def to_provenance(self) -> Dict[str, Any]:
        d = asdict(self)
        d["gpu_identities"] = list(self.gpu_identities())
        return d


@dataclass(frozen=True)
class ResolvedExecutionSet:
    """The frozen, run-scoped fleet authority. Beta's required contents.

    Carries: backend · rig profile · logical nodes and endpoints · worker/GPU
    identities · local-vs-remote · admission count · dataset-verification
    targets. `set_id` is a digest over exactly the deciding content (never the
    timestamp or the invoker), so "WATCHER and the CLI produce the identical set
    for identical inputs" is a checkable claim rather than an assurance.

    ADMISSION IS TWO NUMBERS, DELIBERATELY (Beta, admission-binding brief §1)
    ------------------------------------------------------------------------
    `requested_admission_count` is what the run ASKED for (`worker_pool_size`
    for the miner, `min_workers` for the PWC). `admission_count` is what the
    set IMPOSES:

        admission_count = min(requested, count of selected worker identities)

    Both are recorded because a clamp that overwrites the request is a clamp
    nobody can audit. The defect this closes: the set recorded one count while
    `_serve_clients()` derived `expected_workers` independently from
    `context["worker_pool_size"]` — two frozen run facts about the same run that
    could disagree, so a local two-GPU set still sat waiting for eight workers
    that the set itself said would never exist.
    """
    backend: str
    rig_profile: str
    nodes: Tuple[ResolvedNode, ...]
    remote_execution: bool
    admission_count: Optional[int]           # EFFECTIVE — what the miner imposes
    partial: bool
    declared_nodes: Optional[Tuple[str, ...]]
    invoked_by: str
    # Defaulted, and therefore placed after the non-defaulted fields rather than
    # beside `admission_count` where it belongs semantically: a frozen dataclass
    # cannot order a defaulted field before a bare one, and every construction
    # site (the resolver, the gates) is keyword-based, so position carries no
    # meaning here. `None` = no admission count was requested for this backend.
    requested_admission_count: Optional[int] = None
    resolver_version: int = RESOLVER_VERSION
    resolved_utc: str = field(default_factory=_utc_now_iso)
    resolved_on: str = field(default_factory=socket.gethostname)
    sources: Tuple[str, ...] = ()

    # -- membership -------------------------------------------------------
    def node_ids(self) -> Tuple[str, ...]:
        return tuple(n.node_id for n in self.nodes)

    def endpoints(self) -> Tuple[str, ...]:
        return tuple(n.endpoint for n in self.nodes)

    def config_hostnames(self) -> Tuple[str, ...]:
        return tuple(n.config_hostname for n in self.nodes)

    def worker_ids(self) -> Tuple[str, ...]:
        out: List[str] = []
        for n in self.nodes:
            out.extend(n.gpu_identities())
        return tuple(out)

    def gpu_count(self) -> int:
        return sum(n.gpu_count for n in self.nodes)

    def remote_nodes(self) -> Tuple[ResolvedNode, ...]:
        return tuple(n for n in self.nodes if not n.local)

    def contains_worker(self, worker_id: Optional[str]) -> bool:
        """G-NO-INFERENCE, in one line.

        A worker that connects is admitted only if the SET already named it. It
        does not become eligible by having answered — that is the exact defect
        Beta named on the miner's registration path.
        """
        return bool(worker_id) and worker_id in self.worker_ids()

    def admission_clamped(self) -> bool:
        """Did the set have to reduce what the run asked for?

        True only when a request existed AND the set's worker identities could
        not satisfy it. Read back from provenance rather than inferred from a
        log line.
        """
        return (self.requested_admission_count is not None
                and self.admission_count is not None
                and self.admission_count < self.requested_admission_count)

    def contains_node(self, node_id_or_endpoint: str) -> bool:
        return (node_id_or_endpoint in self.node_ids()
                or node_id_or_endpoint in self.endpoints()
                or node_id_or_endpoint in self.config_hostnames())

    # -- consumer views ---------------------------------------------------
    def dataset_verification_targets(self) -> List[Any]:
        """The P0.5 node list, as `dataset_authority.NodeSpec`s.

        This is Beta's Q1 refinement, and it arrives through the resolver
        exactly as required rather than by special-casing P0.5 or weakening
        `require_fleet`: a run whose set is one local node verifies that node; a
        run whose set is three rigs verifies three. **The set decides, not a
        flag.** A one-node set that fails still refuses — nothing here makes a
        failing node passable, it only decides which nodes are this run's.
        """
        from miner.dataset_authority import NodeSpec       # lazy: avoids a cycle
        return [
            NodeSpec(node_id=n.node_id, ssh_address=n.endpoint,
                     ssh_user=n.ssh_user, local=n.local)
            for n in self.nodes
        ]

    # -- identity ---------------------------------------------------------
    def content(self) -> Dict[str, Any]:
        """Exactly the deciding fields — what two resolvers must agree on."""
        return {
            "resolver_version": self.resolver_version,
            "backend": self.backend,
            "rig_profile": self.rig_profile,
            "remote_execution": self.remote_execution,
            # BOTH are deciding content, so BOTH are in `set_id`: a run that
            # asked for 8 and was clamped to 2 is not the same run as one that
            # asked for 2, even though they admit the same number of workers.
            # The clamp is therefore in the set's identity, not only in its log.
            "requested_admission_count": self.requested_admission_count,
            "admission_count": self.admission_count,
            "partial": self.partial,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "endpoint": n.endpoint,
                    "config_hostname": n.config_hostname,
                    "worker_hostname": n.worker_hostname,
                    "ssh_user": n.ssh_user,
                    "local": n.local,
                    "gpu_count": n.gpu_count,
                    "gpu_type": n.gpu_type,
                }
                for n in self.nodes
            ],
        }

    def set_id(self) -> str:
        blob = json.dumps(self.content(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def to_provenance(self) -> Dict[str, Any]:
        return {
            "execution_set_schema_version": 1,
            "set_id": self.set_id(),
            "resolver_version": self.resolver_version,
            "backend": self.backend,
            "rig_profile": self.rig_profile,
            "remote_execution": self.remote_execution,
            "requested_admission_count": self.requested_admission_count,
            "admission_count": self.admission_count,
            "admission_clamped": self.admission_clamped(),
            "worker_identity_count": len(self.worker_ids()),
            "partial": self.partial,
            "declared_nodes": list(self.declared_nodes or ()),
            "invoked_by": self.invoked_by,
            "resolved_utc": self.resolved_utc,
            "resolved_on": self.resolved_on,
            "sources": list(self.sources),
            "node_count": len(self.nodes),
            "gpu_count": self.gpu_count(),
            "worker_ids": list(self.worker_ids()),
            "nodes": [n.to_provenance() for n in self.nodes],
        }

    def describe(self) -> str:
        scope = "PARTIAL" if self.partial else "full"
        # The clamp is VISIBLE wherever the set is described — and the set is
        # described at resolution, at freeze, by the CLI banner and by WATCHER's
        # log line. A run bounded to fewer workers than it asked for must never
        # have to be inferred from a worker count that quietly never arrives.
        if self.admission_clamped():
            adm = (f" admission={self.admission_count}"
                   f" (CLAMPED from requested "
                   f"{self.requested_admission_count}; "
                   f"{len(self.worker_ids())} worker identities in the set)")
        elif self.admission_count is not None:
            adm = f" admission={self.admission_count}"
        else:
            adm = ""
        return (f"execution set {self.set_id()[:12]} "
                f"backend={self.backend} profile={self.rig_profile} "
                f"{scope} nodes={list(self.node_ids())} "
                f"gpus={self.gpu_count()} remote={self.remote_execution}{adm}")


# ===========================================================================
# The profile map — the JOIN, cross-checked against both existing sources
# ===========================================================================

def default_profile_map_path(repo_root: Optional[str] = None) -> str:
    return os.path.join(repo_root or _repo_root(), PROFILE_MAP_NAME)


def _read_json(path: str, what: str) -> Any:
    if not os.path.exists(path):
        raise ExecutionSetError(
            f"{what} MISSING at {path}. The execution set cannot be resolved, so "
            f"this run cannot establish which machines it runs on.")
    try:
        with open(path, "r") as f:
            return json.load(f)
    except OSError as exc:
        raise ExecutionSetError(
            f"{what} UNREADABLE at {path}: {exc}") from exc
    except ValueError as exc:
        raise ExecutionSetError(
            f"{what} INVALID (not parseable JSON) at {path}: {exc}") from exc


def _load_config_nodes(config_path: str) -> Dict[str, Dict[str, Any]]:
    """`distributed_config.json` keyed by hostname. Addresses are READ, never written."""
    doc = _read_json(config_path, "distributed_config.json")
    out: Dict[str, Dict[str, Any]] = {}
    for nc in doc.get("nodes", []):
        if isinstance(nc, dict) and nc.get("hostname"):
            out[str(nc["hostname"])] = nc
    return out


def _load_provisioning_endpoints(manifest_path: str,
                                 dataset_logical_name: str = "daily3",
                                 ) -> Optional[Dict[str, str]]:
    """`dataset_provisioning.json` node_id -> ssh_address, or None when absent.

    Absent is a legitimate answer (the manifest is gitignored, so a fresh clone
    has none) and is NOT fatal here — P0.5 already owns what an unusable
    manifest costs a run, including the miner-backed hard fail. This function
    exists only so the cross-check below can run when the file IS there.
    """
    if not os.path.exists(manifest_path):
        return None
    doc = _read_json(manifest_path, "dataset_provisioning.json")
    for entry in doc.get("datasets", []):
        if not isinstance(entry, dict):
            continue
        if entry.get("dataset_logical_name") != dataset_logical_name:
            continue
        out: Dict[str, str] = {}
        for n in entry.get("nodes", []) or []:
            if isinstance(n, dict) and n.get("node_id") and n.get("ssh_address"):
                out[str(n["node_id"])] = str(n["ssh_address"])
        return out
    return {}


@dataclass(frozen=True)
class ProfileMap:
    path: str
    default_profile: str
    profiles: Tuple[str, ...]
    nodes: Tuple[Dict[str, Any], ...]


def load_profile_map(path: Optional[str] = None) -> ProfileMap:
    """Read and validate the rig-profile join table.

    Validation is strict on shape and on the two things that silently rot: an
    endpoint declared for a profile that does not exist, and a node declared
    here that `distributed_config.json` does not describe. Both are refused.
    """
    p = os.path.abspath(path or default_profile_map_path())
    doc = _read_json(p, "rig profile map")

    if not isinstance(doc, dict):
        raise ExecutionSetError(
            f"rig profile map INVALID at {p}: top level is "
            f"{type(doc).__name__}, expected an object.")

    ver = doc.get("manifest_schema_version")
    if ver not in SUPPORTED_PROFILE_MAP_SCHEMA_VERSIONS:
        raise ExecutionSetError(
            f"rig profile map INVALID at {p}: declares "
            f"manifest_schema_version={ver!r}; supported: "
            f"{sorted(SUPPORTED_PROFILE_MAP_SCHEMA_VERSIONS)}")

    profiles = doc.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ExecutionSetError(
            f"rig profile map INVALID at {p}: 'profiles' must be a non-empty list.")
    profiles_t = tuple(str(x) for x in profiles)

    default_profile = doc.get("default_profile")
    if default_profile not in profiles_t:
        raise ExecutionSetError(
            f"rig profile map INVALID at {p}: default_profile="
            f"{default_profile!r} is not one of {list(profiles_t)}.")

    nodes = doc.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ExecutionSetError(
            f"rig profile map INVALID at {p}: 'nodes' must be a non-empty list.")

    seen = set()
    clean: List[Dict[str, Any]] = []
    for n in nodes:
        if not isinstance(n, dict):
            raise ExecutionSetError(
                f"rig profile map INVALID at {p}: a node entry is "
                f"{type(n).__name__}, expected an object.")
        for required in ("node_id", "config_hostname", "worker_hostname", "endpoints"):
            if not n.get(required):
                raise ExecutionSetError(
                    f"rig profile map INVALID at {p}: node {n.get('node_id')!r} "
                    f"is missing required field {required!r}.")
        node_id = str(n["node_id"])
        if node_id in seen:
            raise ExecutionSetError(
                f"rig profile map INVALID at {p}: duplicate node_id {node_id!r}.")
        seen.add(node_id)
        eps = n["endpoints"]
        if not isinstance(eps, dict):
            raise ExecutionSetError(
                f"rig profile map INVALID at {p}: node {node_id!r} 'endpoints' is "
                f"{type(eps).__name__}, expected an object.")
        for prof in profiles_t:
            if not eps.get(prof):
                raise ExecutionSetError(
                    f"rig profile map INVALID at {p}: node {node_id!r} declares no "
                    f"endpoint for profile {prof!r}. Both topologies are retained "
                    f"(Beta ruling 3); a node that cannot be addressed under a "
                    f"declared profile makes that profile unresolvable.")
        for prof in eps:
            if prof not in profiles_t:
                raise ExecutionSetError(
                    f"rig profile map INVALID at {p}: node {node_id!r} declares an "
                    f"endpoint for unknown profile {prof!r}; declared profiles are "
                    f"{list(profiles_t)}.")
        clean.append(n)

    return ProfileMap(path=p, default_profile=str(default_profile),
                      profiles=profiles_t, nodes=tuple(clean))


def _cross_check(pmap: ProfileMap,
                 config_nodes: Dict[str, Dict[str, Any]],
                 provisioning: Optional[Dict[str, str]]) -> List[str]:
    """Refuse a join table that disagrees with either file it joins.

    FLEET_STATE_REQUIREMENTS_v1 §5.4-4: *"A node in the provisioning manifest
    that is not in distributed_config.json (or vice versa). Nothing compares the
    two files."* Something does now, and it is the one place that has both in
    hand. A mismatch is fatal rather than resolved by preference, because
    preferring one silently recreates the divergence this module exists to end.
    """
    problems: List[str] = []
    for n in pmap.nodes:
        node_id = str(n["node_id"])
        cfg_host = str(n["config_hostname"])
        if cfg_host not in config_nodes:
            problems.append(
                f"node {node_id!r}: config_hostname {cfg_host!r} is not a node in "
                f"{DISTRIBUTED_CONFIG_NAME}")
            continue
        bare = n["endpoints"].get(PROFILE_BAREMETAL)
        if bare and not bool(n.get("local")) and bare != cfg_host:
            problems.append(
                f"node {node_id!r}: baremetal endpoint {bare!r} != "
                f"{DISTRIBUTED_CONFIG_NAME} hostname {cfg_host!r}. "
                f"{DISTRIBUTED_CONFIG_NAME}'s addresses are the bare-metal profile "
                f"and are deliberate (CLAUDE.md §3) — fix this map, not that file.")
        if provisioning is not None and node_id in provisioning:
            declared = n["endpoints"].get(PROFILE_PROXMOX)
            if declared != provisioning[node_id]:
                problems.append(
                    f"node {node_id!r}: proxmox endpoint {declared!r} != "
                    f"{PROVISIONING_MANIFEST_NAME} ssh_address "
                    f"{provisioning[node_id]!r}")
    return problems


# ===========================================================================
# The resolver — ONE function, called by both WATCHER and the CLI
# ===========================================================================

def resolve_execution_set(
    *,
    backend: str,
    invoked_by: str,
    rig_profile: Optional[str] = None,
    declared_nodes: Optional[Sequence[str]] = None,
    admission_count: Optional[int] = None,
    profile_map_path: Optional[str] = None,
    config_path: Optional[str] = None,
    provisioning_manifest_path: Optional[str] = None,
    repo_root: Optional[str] = None,
) -> ResolvedExecutionSet:
    """Resolve the run's execution set. Pure function of DECLARED inputs.

    Nothing here is probed, pinged, discovered or inferred: the same arguments
    against the same files produce the same `set_id` in WATCHER and in the CLI,
    which is what makes G-SAME-RESOLVER a test rather than a promise.

    `declared_nodes` is the ONLY way to get a partial set, and naming a subset
    IS the explicit declaration Beta required. A name that the map does not
    describe is a hard error — never a silently dropped node, because a set that
    quietly shrank is exactly a set that was inferred.
    """
    root = repo_root or _repo_root()
    if backend not in VALID_BACKENDS:
        raise ExecutionSetError(
            f"unknown backend {backend!r}; expected one of {sorted(VALID_BACKENDS)}")

    cfg_path = os.path.abspath(config_path
                               or os.path.join(root, DISTRIBUTED_CONFIG_NAME))
    prov_path = os.path.abspath(provisioning_manifest_path
                                or os.path.join(root, PROVISIONING_MANIFEST_NAME))

    pmap = load_profile_map(profile_map_path or default_profile_map_path(root))
    config_nodes = _load_config_nodes(cfg_path)
    provisioning = _load_provisioning_endpoints(prov_path)

    problems = _cross_check(pmap, config_nodes, provisioning)
    if problems:
        raise ExecutionSetError(
            "FLEET DEFINITION INCOHERENT — the rig profile map disagrees with the "
            "files it joins, so no single fleet can be resolved:\n  "
            + "\n  ".join(problems)
            + f"\nsources: {pmap.path}, {cfg_path}"
            + (f", {prov_path}" if provisioning is not None else ""))

    profile = rig_profile or pmap.default_profile
    if profile not in pmap.profiles:
        raise ExecutionSetError(
            f"unknown rig profile {profile!r}; declared profiles are "
            f"{list(pmap.profiles)} ({pmap.path})")

    by_id = {str(n["node_id"]): n for n in pmap.nodes}

    if declared_nodes is not None:
        wanted = [str(x).strip() for x in declared_nodes if str(x).strip()]
        if not wanted:
            raise ExecutionSetError(
                "an explicit execution set was requested but names no nodes. A "
                "partial set must be EXPLICIT (Beta); an empty declaration is not "
                "a declaration.")
        unknown = [w for w in wanted if w not in by_id]
        if unknown:
            raise ExecutionSetError(
                f"execution set names node(s) {unknown} that the rig profile map "
                f"does not describe (known: {sorted(by_id)}). A named node is "
                f"never silently dropped — a set that quietly shrank is a set that "
                f"was inferred.")
        selected = [by_id[w] for w in wanted]
        partial = len(selected) < len(pmap.nodes)
        declared_t: Optional[Tuple[str, ...]] = tuple(wanted)
    else:
        selected = list(pmap.nodes)
        partial = False
        declared_t = None

    nodes: List[ResolvedNode] = []
    for n in selected:
        node_id = str(n["node_id"])
        cfg_host = str(n["config_hostname"])
        cfg = config_nodes[cfg_host]
        local = bool(n.get("local"))
        worker_hostname = str(n["worker_hostname"])
        if local:
            # A renamed box must be a LOUD error. Silently keeping a stale name
            # would make every local worker fail the membership test at
            # registration and present as "nobody showed up" — the least
            # diagnosable failure this design can produce.
            live = socket.gethostname()
            if live != worker_hostname:
                raise ExecutionSetError(
                    f"node {node_id!r} declares worker_hostname "
                    f"{worker_hostname!r} but this machine reports {live!r} "
                    f"(socket.gethostname()). The miner worker identity is "
                    f"f'{{socket.gethostname()}}:gpu{{id}}', so a local worker "
                    f"would never match the resolved set. Update {pmap.path}.")
        nodes.append(ResolvedNode(
            node_id=node_id,
            endpoint=str(n["endpoints"][profile]),
            config_hostname=cfg_host,
            worker_hostname=worker_hostname,
            ssh_user=str(n.get("ssh_user") or cfg.get("username") or "michael"),
            local=local,
            gpu_count=int(cfg.get("gpu_count", 0)),
            gpu_type=str(cfg.get("gpu_type", "unknown")),
            rig_profile=profile,
        ))

    # remote_execution is DERIVED FROM THE SET, never declared alongside it.
    # §3 of the brief: `remote_execution=False` is a topology statement, not a
    # bypass — a local run that still drives the 26-GPU coordinator performs
    # remote execution and must not declare otherwise. Deriving it removes the
    # possibility of declaring otherwise.
    remote_execution = any(not n.local for n in nodes)

    # =======================================================================
    # ADMISSION BINDING — the set decides how many workers this run waits for
    # =======================================================================
    # Beta (admission-binding brief §1): the set recorded one count while
    # `_serve_clients()` derived `expected_workers` independently from
    # `context["worker_pool_size"]`. Two frozen facts about one run, free to
    # disagree — and they did: a local two-GPU set still waited for the default
    # eight workers, i.e. for six workers the set itself said did not exist.
    #
    #     effective = min(requested pool size, selected worker identities)
    #
    # "selected worker identities" is exactly the membership the admission test
    # uses (`contains_worker` -> `worker_ids()`), not a parallel capacity
    # notion, so the count that gates admission and the identities that pass it
    # are derived from the same tuple.
    #
    # FAIL DURING RESOLUTION, not at admission time, for zero / negative /
    # zero-capacity: those are unsatisfiable before anything is allocated, and
    # the 180s bounded-admission window (ee0db06) exists to bound a fleet that
    # is LATE, not to discover a fleet that is arithmetically impossible.
    requested_admission = (int(admission_count)
                           if admission_count is not None else None)
    identity_count = sum(n.gpu_count for n in nodes)
    effective_admission = requested_admission
    if requested_admission is not None:
        if requested_admission <= 0:
            raise ExecutionSetError(
                f"admission count {requested_admission} is not positive. The "
                f"admission precondition is `len(eligible) >= expected_workers`, "
                f"so a non-positive count is satisfied by an EMPTY pool: the "
                f"stage would be assigned with no workers at all "
                f"(`assign_stripes` then raises 'requires at least one worker'), "
                f"and the bounded admission window would never have a job to do. "
                f"Refused at RESOLUTION, before that window is ever armed.")
        if identity_count <= 0:
            raise ExecutionSetError(
                f"the resolved execution set contains NO worker identities "
                f"(nodes={[n.node_id for n in nodes]}, gpu counts="
                f"{[n.gpu_count for n in nodes]}), but this run requests an "
                f"admission count of {requested_admission}. A set with zero "
                f"capacity cannot admit anybody: every worker that connects is "
                f"outside the set by construction. Refused at RESOLUTION.")
        effective_admission = min(requested_admission, identity_count)
        if effective_admission != requested_admission:
            logger.warning(
                "[EXEC-SET] ADMISSION CLAMPED: requested %d, but the resolved "
                "set contains %d worker identit%s (%s) — this run admits %d. "
                "The clamp is recorded in provenance "
                "(requested_admission_count/admission_count) and in set_id.",
                requested_admission, identity_count,
                "y" if identity_count == 1 else "ies",
                ", ".join(f"{n.node_id}:{n.gpu_count}" for n in nodes),
                effective_admission)

    sources = [pmap.path, cfg_path]
    if provisioning is not None:
        sources.append(prov_path)

    resolved = ResolvedExecutionSet(
        backend=backend,
        rig_profile=profile,
        nodes=tuple(nodes),
        remote_execution=remote_execution,
        admission_count=effective_admission,
        requested_admission_count=requested_admission,
        partial=partial,
        declared_nodes=declared_t,
        invoked_by=str(invoked_by),
        sources=tuple(sources),
    )
    logger.info("[EXEC-SET] resolved: %s", resolved.describe())
    return resolved


# ===========================================================================
# The freeze — run-scoped, set-once, read by every consumer
# ===========================================================================

_LOCK = threading.RLock()
_ACTIVE: Optional[ResolvedExecutionSet] = None
_READS = 0
_FROZEN_AT: Optional[str] = None


def freeze_execution_set(s: ResolvedExecutionSet) -> ResolvedExecutionSet:
    """Install the run's set. Once. Before anything reads it.

    Two refusals, and they are the two gates:

      * G-RESOLVE-ONCE — freezing AFTER a consumer has already read is refused,
        **including when what the consumer read was `None`.** A set that arrives
        after a decision was made did not govern that decision, and pretending
        otherwise is how "resolved before dataset verification, GPU verification,
        coordinator construction and dispatch" becomes a comment instead of a
        property. The empty read is the load-bearing case: a consumer that read
        `None` took the legacy path, which IS a decision made without the set.
        The enforcement is `_READS += 1` unconditionally in
        `active_execution_set()` plus the `if _READS:` test below — not the
        ordering the current entrypoints happen to use.
      * G-FROZEN — freezing a DIFFERENT set is refused. Re-freezing the
        identical set (same `set_id`) is idempotent and harmless: WATCHER and
        the CLI resolving the same inputs in the same process must not be a
        failure.
    """
    global _ACTIVE, _FROZEN_AT
    if not isinstance(s, ResolvedExecutionSet):
        raise ExecutionSetError(
            f"freeze_execution_set() expects a ResolvedExecutionSet, got "
            f"{type(s).__name__}")
    with _LOCK:
        if _ACTIVE is not None:
            if _ACTIVE.set_id() == s.set_id():
                return _ACTIVE
            raise ExecutionSetError(
                "the execution set is FROZEN for this run and cannot be replaced.\n"
                f"  frozen: {_ACTIVE.describe()}\n"
                f"  offered: {s.describe()}\n"
                "A topology or config change does not alter a run in progress "
                "(Beta: the set is frozen before the run).")
        if _READS:
            raise ExecutionSetError(
                f"cannot freeze the execution set after it has already been read "
                f"{_READS} time(s): a consumer has already decided without it. The "
                f"set must be resolved AFTER backend and rig-profile selection and "
                f"BEFORE dataset verification, GPU verification, coordinator "
                f"construction and dispatch.")
        _ACTIVE = s
        _FROZEN_AT = _utc_now_iso()
        logger.info("[EXEC-SET] FROZEN for this run: %s", s.describe())
        return s


def _peek_execution_set() -> Optional[ResolvedExecutionSet]:
    """RESOLVER-OWNER ONLY: is a set already frozen? — WITHOUT counting a read.

    Private (leading underscore) and deliberately not exported through any
    consumer helper. The ONE legitimate caller is the code that OWNS the freeze
    and needs an idempotency check before performing it —
    `WatcherAgent._ensure_execution_set`, which is re-entered on every step and
    must return the already-frozen set rather than resolve a second one.

    That check is not a consumer decision. It decides *whether to freeze*, not
    *how to run*, so counting it would make the resolver trip the very guard it
    exists to arm: with `active_execution_set()` now counting `None` reads
    (see below), the owner's own "is one frozen yet?" probe would register as
    "a consumer already decided without the set" and refuse the freeze that was
    about to happen. Anything that reads the set to DECIDE ANYTHING uses
    `active_execution_set()` and is counted.
    """
    with _LOCK:
        return _ACTIVE


def active_execution_set() -> Optional[ResolvedExecutionSet]:
    """The frozen set, or None. **Every call counts as a consumer read.**

    None means no set was resolved in this process — a direct harness call, a
    unit test, a module imported standalone. Consumers treat None as "behave
    exactly as before this work existed", which is what keeps every pre-existing
    suite green. Both PRODUCTION entry points (the `window_optimizer.py` CLI and
    `WatcherAgent.run_step`) always freeze one, so None never occurs on a real run.

    WHY THE COUNTER FIRES ON `None` — a RETRACTION (Beta, admission-binding brief §0)
    ---------------------------------------------------------------------------
    This function used to increment `_READS` only inside `if _ACTIVE is not None`,
    and the submission claimed on that basis that freezing after a read was
    structurally impossible. **That claim was false**, and Beta traced why: with
    the counter behind the non-`None` test, a consumer could read `None`, take
    the legacy path, and the set could still be frozen afterwards — precisely the
    "a consumer already decided without it" sequence `freeze_execution_set`
    refuses. The empty read is not the harmless case; it is THE case that matters,
    because a consumer that read `None` did not merely fail to learn the fleet, it
    went and behaved as though no fleet authority existed.

    So the counter is now unconditional, and `G-RESOLVE-ONCE` is a property of
    this line rather than of the order the live entrypoints happen to use:

        _READS += 1          <- fires for a None read too
        return _ACTIVE

    The freeze path (`freeze_execution_set`, above) reads that counter BEFORE
    installing anything, and the idempotent re-freeze of an identical set returns
    earlier still — from the `_ACTIVE is not None` branch, which is not reached
    via the counter at all — so consumption cannot break re-entrancy.
    """
    global _READS
    with _LOCK:
        _READS += 1
        return _ACTIVE


def require_execution_set(context: str = "") -> ResolvedExecutionSet:
    s = active_execution_set()
    if s is None:
        raise ExecutionSetError(
            f"no resolved execution set is frozen{' for ' + context if context else ''}. "
            f"The fleet for this run was never established.")
    return s


def clear_execution_set() -> None:
    """Test-only teardown. Production has one set per process, and no way back."""
    global _ACTIVE, _READS, _FROZEN_AT
    with _LOCK:
        _ACTIVE = None
        _READS = 0
        _FROZEN_AT = None


class execution_set_scope:
    """`with execution_set_scope(s):` — freeze for a block, then release.

    For harnesses and gates only. A production run freezes once and never
    releases, because a run has exactly one fleet.
    """

    def __init__(self, s: Optional[ResolvedExecutionSet]):
        self._s = s

    def __enter__(self) -> Optional[ResolvedExecutionSet]:
        clear_execution_set()
        if self._s is not None:
            freeze_execution_set(self._s)
        return self._s

    def __exit__(self, *exc) -> bool:
        clear_execution_set()
        return False


# ===========================================================================
# Consumer helpers — the ONE way each of the six reads the set
# ===========================================================================

def filter_config_nodes(node_dicts: Sequence[Dict[str, Any]],
                        *, consumer: str) -> List[Dict[str, Any]]:
    """Re-point a `distributed_config.json` node list at the resolved set.

    Shared by the legacy coordinator, the PWC and the WATCHER preflight checker,
    because all three build their node list from the same file and all three
    were therefore hard-wired to the bare-metal addresses. Two things happen and
    only these two:

      1. nodes outside the set are DROPPED — the set decides the fleet;
      2. the surviving node's `hostname` is rewritten to the set's profile
         endpoint — the set decides the address.

    `distributed_config.json` itself is never written. With no set frozen the
    list is returned untouched, so every pre-existing caller behaves exactly as
    it did before.
    """
    s = active_execution_set()
    if s is None:
        return list(node_dicts)
    by_cfg_host = {n.config_hostname: n for n in s.nodes}
    out: List[Dict[str, Any]] = []
    for nc in node_dicts:
        host = str(nc.get("hostname", ""))
        rn = by_cfg_host.get(host)
        if rn is None:
            logger.info("[EXEC-SET] %s: node %s not in the resolved execution set "
                        "— dropped", consumer, host or "<unnamed>")
            continue
        repointed = dict(nc)
        repointed["hostname"] = rn.endpoint
        if rn.endpoint != host:
            logger.info("[EXEC-SET] %s: node %s re-pointed %s -> %s (profile=%s)",
                        consumer, rn.node_id, host, rn.endpoint, rn.rig_profile)
        out.append(repointed)
    if not out:
        raise ExecutionSetError(
            f"{consumer}: the resolved execution set {s.set_id()[:12]} selected no "
            f"node from {DISTRIBUTED_CONFIG_NAME} (set nodes: "
            f"{list(s.node_ids())}). Refusing rather than running against an empty "
            f"fleet.")
    return out


def is_admitted_worker(worker_id: Optional[str]) -> Tuple[bool, Optional[str]]:
    """G-NO-INFERENCE for the miner registration path.

    Returns `(admitted, refusal_reason)`. With no set frozen every worker is
    admitted, which is the pre-existing behaviour and keeps the Phase-4 gates
    and loopback harnesses unchanged. With a set frozen, a worker the set does
    not name is refused ADMISSION — not the connection: it still registers, and
    the refusal is durably recorded on the worker row, exactly as a capability
    inconsistency already is. An unknown worker must not become eligible merely
    because it connected.
    """
    s = active_execution_set()
    if s is None:
        return True, None
    if s.contains_worker(worker_id):
        return True, None
    return False, (
        f"worker {worker_id!r} is NOT in the resolved execution set "
        f"{s.set_id()[:12]} (profile={s.rig_profile}, nodes={list(s.node_ids())}). "
        f"Registered but INELIGIBLE: membership is declared before the run, never "
        f"earned by connecting.")


def admission_expectation(context_pool_size: int,
                          *, consumer: str = "miner") -> Tuple[int, str]:
    """How many workers must be admitted before a stage is assigned.

    Returns `(expected_workers, source)`. THE SET IS THE AUTHORITY when one is
    frozen — Beta's requirement, and the whole point of the repair: with a set
    frozen, `context["worker_pool_size"]` must not remain a parallel authority
    for the same quantity. It stays the REQUEST (the set was resolved from it,
    which is why the two normally agree); what it stops being is a second answer.

    With NO set frozen the context value is returned unchanged. That is not a
    bypass — it is the pre-existing behaviour, and it is what every Phase-4
    loopback gate and every direct-harness call runs on. Production cannot take
    that branch: both entrypoints freeze a set before any coordinator exists.

    A frozen set whose `admission_count` is None (a backend that declares no
    admission semantics) also falls back, and says so in `source`, because
    inventing a count the set never carried would be the same defect pointed the
    other way.
    """
    ctx = int(context_pool_size)
    s = active_execution_set()
    if s is None:
        return ctx, "context(no execution set frozen)"
    if s.admission_count is None:
        logger.info("[EXEC-SET] %s: the frozen set %s carries no admission "
                    "count (backend=%s); expected_workers stays %d from context.",
                    consumer, s.set_id()[:12], s.backend, ctx)
        return ctx, f"context(set {s.set_id()[:12]} carries no admission count)"
    if s.admission_count != ctx:
        logger.warning(
            "[EXEC-SET] %s: expected_workers BOUND TO THE FROZEN SET: %d "
            "(requested %s via worker_pool_size=%d; set %s has %d worker "
            "identities across %s). The context value is the REQUEST, not a "
            "second authority — a run must not wait for workers its own frozen "
            "fleet says do not exist.",
            consumer, s.admission_count, s.requested_admission_count, ctx,
            s.set_id()[:12], len(s.worker_ids()), list(s.node_ids()))
    else:
        logger.info("[EXEC-SET] %s: expected_workers=%d, bound to the frozen "
                    "set %s (agrees with the requested pool size).",
                    consumer, s.admission_count, s.set_id()[:12])
    return int(s.admission_count), f"execution_set({s.set_id()[:12]})"


def execution_set_provenance() -> Optional[Dict[str, Any]]:
    """The frozen set as a provenance object, or None. Read back, not assumed."""
    s = active_execution_set()
    return s.to_provenance() if s is not None else None
