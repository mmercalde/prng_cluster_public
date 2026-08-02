"""S172 Phase 6-P0.5 — the runtime dataset authority.

Phase 6-P0 (commit `131787d`) published an immutable dataset version and an atomic
pointer manifest and changed **no** running code. Nothing read the pointer. This
module is what makes the pointer authoritative, and it is the single place where
that authority lives.

Contract documents (binding):

  * `docs/DATASET_PUBLICATION_SCHEMA_v1.md`     — the frozen publication schema
  * `docs/RUNTIME_DATASET_PROVISIONING_CONTRACT.md` — provisioning + fail-before-dispatch
  * `docs/CLAUDE_CODE_INSTRUCTIONS_PHASE_6_P0_5_IMPLEMENTATION.md` §2 — the eight
    required behaviours this module exists to provide

THE ONE IDEA
------------
A run resolves the pointer **once**, at run start, into an immutable
`FrozenDataset`. Every later consumer — the coordinator's per-assignment digest,
the fleet verification, the run provenance record — reads **that frozen value**
and never touches the pointer again. A scrape that lands mid-run moves the
pointer; the run in progress does not notice, because nothing in the run re-reads
it. That is requirement 7, and it is the reason `freeze` is a verb here rather
than a comment.

WHY THE FREEZE IS RUN-SCOPED AND NOT TRIAL-SCOPED
-------------------------------------------------
`range_miner_coordinator.serve_trial` derived `dataset_sha256` per **trial**
(`compute_dataset_sha256(dataset_path)`). Two consecutive Optuna trials in one
study could therefore observe two different datasets, with no error anywhere: the
digest simply changed between trials and every downstream check remained
self-consistent against the wrong thing. A study split across two datasets is not
a detectable condition after the fact — the NPZ carries whichever digest was
current when its trial ran. Moving that derivation to run scope is what makes
requirement 7 mean anything, and it is why `run_frozen_dataset_sha256()` exists.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
It never writes, moves, renames or repairs a dataset, a version file, the pointer
manifest, or the `daily3.json` compatibility alias. Publication is P0 (done) and
P2 (the scraper). This module **reads and verifies**; the only file it creates is
its own run-provenance record, and the only file it transfers is a copy of an
already-published immutable version onto a worker node.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import shlex
import socket
import subprocess
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# ===========================================================================
# Constants — the published layout (DATASET_PUBLICATION_SCHEMA_v1 §1, §2)
# ===========================================================================

#: The pointer manifest. NOT a dataset and NOT a symlink — Beta's ruling
#: (project facts §2.10) requires a manifest so the digest travels with the name.
POINTER_MANIFEST_NAME = "daily3_current.json"

#: The legacy compatibility alias. Every pre-P0 consumer opens this. After P0.5
#: it is never *dispatched* — see `resolve_dataset_path`.
LEGACY_ALIAS_NAME = "daily3.json"

#: The version filename grammar, schema §2: `daily3-<UTC>Z-<sha256[:12]>.json`
#: where <UTC> is ISO-8601 basic `YYYYMMDDThhmmssffffff` (8 + T + 12 digits).
#: Anchored at both ends: this is an allowlist, not a search.
VERSION_FILENAME_RE = re.compile(
    r"^daily3-(?P<stamp>\d{8}T\d{12})Z-(?P<digest12>[0-9a-f]{12})\.json$"
)

#: Manifest schema versions this reader understands. A future schema 2 must fail
#: loudly here rather than be read with schema-1 assumptions.
SUPPORTED_MANIFEST_SCHEMA_VERSIONS = frozenset({1})

#: Fields the pointer manifest must carry for a run to be frozen against it.
REQUIRED_MANIFEST_FIELDS = (
    "manifest_schema_version",
    "version_id",
    "filename",
    "sha256",
    "size_bytes",
    "record_count",
)

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")

#: Default provisioning manifest (RUNTIME_DATASET_PROVISIONING_CONTRACT §1).
#: `*.json` so `.gitignore:41` keeps it out of `git status --porcelain`, and
#: therefore out of `repository_tree_clean` / the finalizer certification wall.
PROVISIONING_MANIFEST_NAME = "dataset_provisioning.json"

#: Where run provenance records land. `*.json` inside, so the directory never
#: shows up as untracked content and cannot dirty the tree.
RUN_PROVENANCE_DIRNAME = "dataset_provenance"

# ---------------------------------------------------------------------------
# Fleet-verification vocabulary (Beta, P0.5 closure ruling)
# ---------------------------------------------------------------------------
#: Every node verified on target.
FLEET_STATUS_PASS = "PASS"

#: **A required verification was ATTEMPTED and could not be completed.** That is
#: the whole meaning of the word, and it is why it is fatal for a miner-backed
#: run: "we needed it and could not get it" is not "we did not need it".
FLEET_STATUS_UNAVAILABLE = "UNAVAILABLE"

#: **No fleet verification was ever required** — this path performs no remote
#: execution, so there is no worker dataset to establish. Introduced by Beta's
#: P0.5 closure ruling precisely so that a path which never needed the check
#: stops borrowing `UNAVAILABLE`, which would make a genuine unverifiable fleet
#: indistinguishable from a run that had no fleet at all.
FLEET_STATUS_NOT_APPLICABLE = "NOT_APPLICABLE"


# ===========================================================================
# Exceptions
# ===========================================================================

class DatasetAuthorityError(Exception):
    """Base for every dataset-authority failure raised before dispatch."""


class PointerResolutionError(DatasetAuthorityError):
    """The pointer manifest is missing, unreadable, unparseable, or incomplete."""


class PointerValidationError(DatasetAuthorityError):
    """The pointer resolved, but names something it is not permitted to name.

    §2.2: the pointer selects among *published versions*. It is not a general
    path parameter. A bare alias, an absolute path, a traversal, or any filename
    outside the version grammar is refused here.
    """


class DatasetIdentityError(DatasetAuthorityError):
    """The on-disk version file does not match the identity the manifest claims."""


class DatasetFreezeError(DatasetAuthorityError):
    """A second, conflicting freeze was attempted inside one run.

    Not a defensive nicety: this is the audible form of the split-study failure
    the run-scoped freeze exists to prevent.
    """


def _dataset_provisioning_error():
    """`DatasetProvisioningError`, imported lazily.

    It lives in `miner.range_miner_worker` because that is where `ResidueError`
    lives and Beta's correction (P0 ruling §3) requires the provisioning error to
    sit **inside** the residue hierarchy so the coordinator's existing
    `stripe_error(retryable=False)` control flow is preserved — while *not*
    flattening a missing dataset into an undifferentiated residue error.

    Imported inside the function so that importing this module from
    `window_optimizer.py` or `agents/watcher_agent.py` does not drag in the
    worker module.
    """
    from miner.range_miner_worker import DatasetProvisioningError
    return DatasetProvisioningError


# ===========================================================================
# Primitives
# ===========================================================================

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def sha256_file(path: str) -> str:
    """Streaming sha256. Byte-for-byte the same derivation as
    `range_miner_coordinator.compute_dataset_sha256` and
    `range_miner_worker._sha256_file`, so all three compare like against like."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_records(path: str) -> int:
    """Record count of a dataset file — the semantic count the schema carries.

    A top-level JSON array is required: the publication schema's `record_count`
    and the P2 publication-prefix wall are both **record-sequence** notions
    (project facts §2.10 — a byte-prefix test is invalid for a JSON array).
    """
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise DatasetIdentityError(
            f"dataset {path!r} is not a top-level JSON array "
            f"(got {type(data).__name__}); record_count is undefined for it"
        )
    return len(data)


def local_node_identity() -> str:
    """The node name used in every failure message.

    `socket.gethostname()` deliberately: CT100 is created with the rig's
    canonical hostname (`pct create --hostname rrig6600c`) precisely so that
    hostname *is* coordinator identity (docs/S172_INFRASTRUCTURE_INTERFACE_v1_0).
    """
    try:
        return socket.gethostname()
    except Exception:                                    # pragma: no cover
        return "<unknown-host>"


# ===========================================================================
# The frozen identity
# ===========================================================================

@dataclass(frozen=True)
class FrozenDataset:
    """The run's dataset identity, resolved once and immutable thereafter.

    `frozen=True` is load-bearing, not decoration: the whole point is that no
    consumer can mutate what a later consumer reads.
    """
    dataset_logical_name: str
    path: str                       # ABSOLUTE — requirement 3
    filename: str
    sha256: str
    size_bytes: int
    record_count: int
    resolution_source: str          # "pointer" | "explicit_version" | "explicit"
    version_id: Optional[str] = None
    dataset_lineage_id: Optional[str] = None
    published_utc: Optional[str] = None
    predecessor_sha256: Optional[str] = None
    manifest_path: Optional[str] = None
    manifest_sha256: Optional[str] = None
    first_draw: Optional[Dict[str, Any]] = None
    last_draw: Optional[Dict[str, Any]] = None
    frozen_utc: str = ""
    frozen_on_node: str = ""

    def identity_key(self) -> tuple:
        """What must not change within one run."""
        return (self.path, self.sha256, self.size_bytes, self.record_count)

    def to_provenance(self) -> Dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        vid = self.version_id or "<unversioned>"
        return (f"{vid} sha256={self.sha256[:12]}… "
                f"{self.size_bytes} bytes / {self.record_count} records "
                f"at {self.path}")


# ===========================================================================
# Pointer resolution + validation (requirements 1 and 8)
# ===========================================================================

def validate_version_filename(filename: Any, *, pointer_path: str) -> str:
    """Requirement 8 / §2.2 — the pointer names a *permitted* version file.

    Every rejection below is a refusal, never a repair. The pointer selects among
    published versions; anything else is a category error, and the run stops.
    """
    if not isinstance(filename, str) or not filename:
        raise PointerValidationError(
            f"pointer manifest {pointer_path!r}: 'filename' must be a non-empty "
            f"string, got {filename!r}"
        )

    # A directory component of ANY kind is refused before the grammar check, so
    # the error names the real problem instead of a confusing grammar mismatch.
    if os.path.isabs(filename) or filename != os.path.basename(filename) \
            or "/" in filename or "\\" in filename or filename in (".", ".."):
        raise PointerValidationError(
            f"pointer manifest {pointer_path!r}: 'filename' must be a bare "
            f"filename inside the publication directory, not a path — got "
            f"{filename!r}. The pointer selects among published versions; it is "
            f"not a general path parameter (schema §2, P0.5 §2.2)."
        )

    if filename == LEGACY_ALIAS_NAME:
        raise PointerValidationError(
            f"pointer manifest {pointer_path!r} names the legacy compatibility "
            f"alias {LEGACY_ALIAS_NAME!r}. The alias is mutable and carries no "
            f"version identity — it can never be the pointer's target "
            f"(P0.5 §2.2, requirement 3)."
        )

    if not VERSION_FILENAME_RE.match(filename):
        raise PointerValidationError(
            f"pointer manifest {pointer_path!r}: {filename!r} does not match the "
            f"frozen version grammar daily3-<YYYYMMDDThhmmssffffff>Z-<sha256[:12]>.json "
            f"(DATASET_PUBLICATION_SCHEMA_v1 §2)"
        )
    return filename


def _require_containment(publication_dir: str, resolved: str, pointer_path: str) -> None:
    """The resolved target must sit inside the publication directory.

    Checked on `realpath` of both sides, so a symlink planted in the publication
    directory cannot carry the target outside it. The bare-filename check above
    already refuses traversal syntactically; this refuses it semantically too,
    because the two are not the same guarantee.
    """
    real_dir = os.path.realpath(publication_dir)
    real_target = os.path.realpath(resolved)
    if os.path.dirname(real_target) != real_dir:
        raise PointerValidationError(
            f"pointer manifest {pointer_path!r}: target {resolved!r} resolves to "
            f"{real_target!r}, which is outside the publication directory "
            f"{real_dir!r}. Refusing (P0.5 §2.2)."
        )


def read_pointer_manifest(pointer_path: str) -> Dict[str, Any]:
    """Read + structurally validate the pointer manifest. No dataset touched yet."""
    if not os.path.exists(pointer_path):
        raise PointerResolutionError(
            f"pointer manifest not found: {pointer_path!r} on node "
            f"{local_node_identity()!r}. Phase 6-P0 publishes it; a tree without "
            f"it is not a published tree."
        )
    try:
        with open(pointer_path, "r") as f:
            manifest = json.load(f)
    except json.JSONDecodeError as exc:
        raise PointerResolutionError(
            f"pointer manifest {pointer_path!r} is not parseable JSON: {exc}"
        ) from exc
    except OSError as exc:
        raise PointerResolutionError(
            f"pointer manifest {pointer_path!r} is unreadable on node "
            f"{local_node_identity()!r}: {exc}"
        ) from exc

    if not isinstance(manifest, dict):
        raise PointerResolutionError(
            f"pointer manifest {pointer_path!r} must be a JSON object, got "
            f"{type(manifest).__name__}"
        )

    missing = [k for k in REQUIRED_MANIFEST_FIELDS if k not in manifest]
    if missing:
        raise PointerResolutionError(
            f"pointer manifest {pointer_path!r} is missing required field(s): "
            f"{', '.join(missing)}"
        )

    schema_version = manifest["manifest_schema_version"]
    if schema_version not in SUPPORTED_MANIFEST_SCHEMA_VERSIONS:
        raise PointerResolutionError(
            f"pointer manifest {pointer_path!r} declares "
            f"manifest_schema_version={schema_version!r}; this reader supports "
            f"{sorted(SUPPORTED_MANIFEST_SCHEMA_VERSIONS)}. Refusing to read a "
            f"future schema with present-schema assumptions."
        )

    sha = manifest["sha256"]
    if not isinstance(sha, str) or not _SHA256_HEX_RE.match(sha):
        raise PointerResolutionError(
            f"pointer manifest {pointer_path!r}: 'sha256' must be 64 lowercase "
            f"hex characters, got {sha!r}"
        )
    return manifest


def resolve_pointer(pointer_path: str, *, verify_record_count: bool = True) -> FrozenDataset:
    """Resolve `daily3_current.json` to a verified `FrozenDataset` (requirement 1).

    Verification here is deliberate and complete: the digest, the size and the
    record count are all **re-derived from the file on disk** and compared with
    what the manifest claims. This is ~1.4 MB of hashing and one JSON parse, once
    per run. The alternative — trusting the manifest — would make the frozen
    identity an assertion rather than a measurement, and every downstream check
    would then be verifying the run against a claim instead of against the data.
    """
    pointer_path = os.path.abspath(pointer_path)
    publication_dir = os.path.dirname(pointer_path)
    manifest = read_pointer_manifest(pointer_path)

    filename = validate_version_filename(manifest["filename"], pointer_path=pointer_path)

    # Schema §2 invariant: `filename == version_id + ".json"`, so neither can
    # drift from the other. A manifest violating it is internally inconsistent
    # and there is no correct way to choose which field wins.
    version_id = manifest["version_id"]
    if f"{version_id}.json" != filename:
        raise PointerValidationError(
            f"pointer manifest {pointer_path!r}: version_id {version_id!r} and "
            f"filename {filename!r} disagree; the schema requires "
            f"filename == version_id + '.json' (§2)"
        )

    # The digest prefix in the name is a convenience copy — but if it disagrees
    # with the authoritative full digest the manifest is self-contradictory.
    digest12 = VERSION_FILENAME_RE.match(filename).group("digest12")
    manifest_sha = manifest["sha256"]
    if manifest_sha[:12] != digest12:
        raise PointerValidationError(
            f"pointer manifest {pointer_path!r}: filename digest prefix "
            f"{digest12!r} disagrees with the manifest sha256 "
            f"{manifest_sha[:12]!r} (schema §2)"
        )

    target = os.path.join(publication_dir, filename)
    if not os.path.exists(target):
        raise PointerResolutionError(
            f"pointer manifest {pointer_path!r} names {filename!r}, which does "
            f"not exist at {target!r} on node {local_node_identity()!r}"
        )
    _require_containment(publication_dir, target, pointer_path)

    # ---- re-derive identity from the bytes on disk --------------------------
    actual_size = os.path.getsize(target)
    claimed_size = int(manifest["size_bytes"])
    if actual_size != claimed_size:
        raise DatasetIdentityError(
            f"published version {target!r} is {actual_size} bytes, manifest "
            f"claims {claimed_size}. Size is the cheap truncation check and it "
            f"failed; refusing before any digest work."
        )

    actual_sha = sha256_file(target)
    if actual_sha != manifest_sha:
        raise DatasetIdentityError(
            f"published version {target!r} sha256 {actual_sha} does not match "
            f"the pointer manifest's {manifest_sha}. The immutable version file "
            f"has been altered, or the manifest points at the wrong bytes."
        )

    claimed_records = int(manifest["record_count"])
    if verify_record_count:
        actual_records = count_records(target)
        if actual_records != claimed_records:
            raise DatasetIdentityError(
                f"published version {target!r} holds {actual_records} records, "
                f"manifest claims {claimed_records}"
            )
    else:
        actual_records = claimed_records

    return FrozenDataset(
        dataset_logical_name="daily3",
        path=target,
        filename=filename,
        sha256=actual_sha,
        size_bytes=actual_size,
        record_count=actual_records,
        resolution_source="pointer",
        version_id=version_id,
        dataset_lineage_id=manifest.get("dataset_lineage_id"),
        published_utc=manifest.get("published_utc"),
        predecessor_sha256=manifest.get("predecessor_sha256"),
        manifest_path=pointer_path,
        manifest_sha256=sha256_file(pointer_path),
        first_draw=manifest.get("first_draw"),
        last_draw=manifest.get("last_draw"),
        frozen_utc=_utc_now_iso(),
        frozen_on_node=local_node_identity(),
    )


def resolve_dataset_path(
    dataset_arg: str,
    *,
    pointer_manifest: Optional[str] = None,
    allow_unpublished_alias: bool = False,
    verify_record_count: bool = True,
) -> FrozenDataset:
    """Turn a `--lottery-file` argument into a verified, absolute `FrozenDataset`.

    Three cases, and the first is the one requirement 3 is about:

    1. **the bare alias** `daily3.json` — resolve the pointer beside it and
       return the *immutable version* instead. The alias is mutable, carries no
       version identity, and is exactly what must never be dispatched. If no
       pointer manifest sits beside it the run is refused: an alias in a
       published tree with no pointer is not a situation with a safe default.
    2. **an explicit version-stamped file** — validated against the grammar and
       used as given. Deliberately NOT re-resolved through the pointer: a run
       that was handed a specific immutable version must keep using it even if
       the pointer moves (requirement 7). This is the case WATCHER's children hit.
    3. **any other dataset** (`pa_pick3.json`, a harness temp file) — frozen on
       its own path/digest with `version_id=None`. Requirement 3 forbids
       dispatching the bare `daily3.json`; it does not make every other dataset
       in the project unusable.

    `allow_unpublished_alias` exists for harnesses that build a synthetic tree
    containing a `daily3.json` and no publication. It is never set on the
    certifying path.
    """
    abs_path = os.path.abspath(dataset_arg)
    basename = os.path.basename(abs_path)
    parent = os.path.dirname(abs_path)

    # ---- case 1: the legacy alias ------------------------------------------
    if basename == LEGACY_ALIAS_NAME:
        pointer = pointer_manifest or os.path.join(parent, POINTER_MANIFEST_NAME)
        if os.path.exists(pointer):
            frozen = resolve_pointer(pointer, verify_record_count=verify_record_count)
            logger.info(
                "[P0.5] pointer resolved: %s -> %s", LEGACY_ALIAS_NAME, frozen.describe()
            )
            return frozen
        if not allow_unpublished_alias:
            raise PointerResolutionError(
                f"refusing to dispatch the bare compatibility alias {abs_path!r}: "
                f"no pointer manifest at {pointer!r} on node "
                f"{local_node_identity()!r}. The alias is mutable and version-less; "
                f"after Phase 6-P0.5 the authoritative dataset is the immutable "
                f"version the pointer names (requirement 3, §2.2)."
            )
        logger.warning(
            "[P0.5] %s frozen as an UNPUBLISHED alias (no pointer manifest at %s) — "
            "permitted only outside the certifying path", abs_path, pointer
        )

    # ---- case 2: an explicit published version -----------------------------
    elif VERSION_FILENAME_RE.match(basename):
        if not os.path.exists(abs_path):
            raise PointerResolutionError(
                f"dataset version {abs_path!r} does not exist on node "
                f"{local_node_identity()!r}"
            )
        size = os.path.getsize(abs_path)
        sha = sha256_file(abs_path)
        digest12 = VERSION_FILENAME_RE.match(basename).group("digest12")
        if sha[:12] != digest12:
            raise DatasetIdentityError(
                f"version file {abs_path!r} has sha256 {sha}, but its name claims "
                f"the content digest starts {digest12!r}. The file is not the "
                f"version its name says it is."
            )
        records = count_records(abs_path) if verify_record_count else -1
        pointer = pointer_manifest or os.path.join(parent, POINTER_MANIFEST_NAME)
        manifest_sha = sha256_file(pointer) if os.path.exists(pointer) else None
        return FrozenDataset(
            dataset_logical_name="daily3",
            path=abs_path,
            filename=basename,
            sha256=sha,
            size_bytes=size,
            record_count=records,
            resolution_source="explicit_version",
            version_id=basename[:-len(".json")],
            manifest_path=pointer if manifest_sha else None,
            manifest_sha256=manifest_sha,
            frozen_utc=_utc_now_iso(),
            frozen_on_node=local_node_identity(),
        )

    # ---- case 3 (and the permitted case-1 fallthrough) ---------------------
    if not os.path.exists(abs_path):
        raise PointerResolutionError(
            f"dataset {abs_path!r} does not exist on node "
            f"{local_node_identity()!r}"
        )
    return FrozenDataset(
        dataset_logical_name=os.path.splitext(basename)[0],
        path=abs_path,
        filename=basename,
        sha256=sha256_file(abs_path),
        size_bytes=os.path.getsize(abs_path),
        record_count=count_records(abs_path) if verify_record_count else -1,
        resolution_source="explicit",
        frozen_utc=_utc_now_iso(),
        frozen_on_node=local_node_identity(),
    )


# ===========================================================================
# The run-scoped freeze (requirements 2 and 7)
# ===========================================================================

_FREEZE_LOCK = threading.RLock()
_RUN_FREEZE: Optional[FrozenDataset] = None
_RUN_FREEZE_LABEL: Optional[str] = None


def freeze_run_dataset(frozen: FrozenDataset, *, run_label: str = "") -> FrozenDataset:
    """Install the run's dataset identity. Once. (Requirement 2.)

    Re-freezing the *same* identity is idempotent, because a run may legitimately
    reach the freeze through more than one entry point (WATCHER resolves, then
    the child process it launched resolves the absolute path it was handed).

    Re-freezing a **different** identity raises. That is the split-study
    condition — two datasets inside one run — and it is precisely what must never
    pass silently.
    """
    global _RUN_FREEZE, _RUN_FREEZE_LABEL
    with _FREEZE_LOCK:
        if _RUN_FREEZE is None:
            _RUN_FREEZE = frozen
            _RUN_FREEZE_LABEL = run_label or None
            logger.info("[P0.5] run dataset FROZEN: %s", frozen.describe())
            return _RUN_FREEZE

        if _RUN_FREEZE.identity_key() == frozen.identity_key():
            return _RUN_FREEZE

        raise DatasetFreezeError(
            "REFUSING a second, conflicting dataset freeze inside one run.\n"
            f"  already frozen : {_RUN_FREEZE.describe()}\n"
            f"  now offered    : {frozen.describe()}\n"
            "A run is bound to exactly one dataset identity from run start. Two "
            "identities in one run is the split-study failure the freeze exists "
            "to prevent (P0.5 §2.1, requirement 7)."
        )


def get_frozen_dataset() -> Optional[FrozenDataset]:
    """The run's frozen identity, or None if this process never froze one."""
    with _FREEZE_LOCK:
        return _RUN_FREEZE


def clear_frozen_dataset() -> None:
    """Drop the freeze. For harnesses and for a process that genuinely starts a
    new run in-process; never called on the certifying path mid-run."""
    global _RUN_FREEZE, _RUN_FREEZE_LABEL
    with _FREEZE_LOCK:
        _RUN_FREEZE = None
        _RUN_FREEZE_LABEL = None


def freeze_for_run(
    dataset_arg: str,
    *,
    run_label: str = "",
    pointer_manifest: Optional[str] = None,
    allow_unpublished_alias: bool = False,
    verify_record_count: bool = True,
) -> FrozenDataset:
    """Resolve + freeze in one call — the run-start entry point."""
    frozen = resolve_dataset_path(
        dataset_arg,
        pointer_manifest=pointer_manifest,
        allow_unpublished_alias=allow_unpublished_alias,
        verify_record_count=verify_record_count,
    )
    return freeze_run_dataset(frozen, run_label=run_label)


def run_frozen_dataset_sha256(dataset_path: str) -> Optional[str]:
    """The frozen digest for `dataset_path`, or None if this run has no freeze.

    This is the function that converts the coordinator's **per-trial** digest
    derivation into a **per-run** one. It returns the frozen digest only when the
    path being asked about is the frozen path, so a run that legitimately handles
    a second, different dataset (a harness, a non-frozen path) still gets the
    old compute-it-now behaviour rather than a silently wrong answer.
    """
    frozen = get_frozen_dataset()
    if frozen is None:
        return None
    if os.path.abspath(dataset_path) != frozen.path:
        return None
    return frozen.sha256


# ===========================================================================
# Provisioning + per-node verification (requirements 4 and 5)
# ===========================================================================

@dataclass(frozen=True)
class NodeSpec:
    """A worker node the dataset must be present on before dispatch."""
    node_id: str
    ssh_address: str
    ssh_user: str = "michael"
    local: bool = False             # verify on this box, no ssh

    def target(self) -> str:
        return f"{self.ssh_user}@{self.ssh_address}" if self.ssh_user else self.ssh_address


#: Per-node outcome. Terminates in one of the four VIR-3 sentinels.
@dataclass
class NodeVerification:
    node_id: str
    ssh_address: str
    dataset_path: str
    status: str                     # PASS | FAIL | UNAVAILABLE | INCOMPLETE
    digest: Optional[str] = None
    size_bytes: Optional[int] = None
    expected_digest: Optional[str] = None
    expected_size_bytes: Optional[int] = None
    message: str = ""
    provisioned: bool = False

    def to_provenance(self) -> Dict[str, Any]:
        return asdict(self)


def _run(cmd: Sequence[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout)


def _ssh_cmd(node: NodeSpec, remote: str, connect_timeout: int) -> List[str]:
    return [
        "ssh", "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={connect_timeout}",
        node.target(), remote,
    ]


def verify_node_dataset(
    frozen: FrozenDataset,
    node: NodeSpec,
    *,
    timeout: float = 120.0,
    connect_timeout: int = 10,
) -> NodeVerification:
    """Re-derive the digest **on the target node** and compare with the freeze.

    On the target, not on the sender — contract §4 step 3. Hashing the file you
    just sent, on the machine that will actually read it, is the only version of
    this check that catches a truncated or interrupted transfer. Hashing the
    source proves only that the source is intact, which was never in doubt.

    An unreachable node or an unrunnable checker is `UNAVAILABLE`, never a pass
    (VIR-5). The caller fails closed on both.
    """
    rec = NodeVerification(
        node_id=node.node_id,
        ssh_address=node.ssh_address,
        dataset_path=frozen.path,
        status="INCOMPLETE",
        expected_digest=frozen.sha256,
        expected_size_bytes=frozen.size_bytes,
    )

    quoted = shlex.quote(frozen.path)
    # One round trip: existence, size and digest together, so the three facts
    # describe the same instant rather than three different ones.
    remote = (
        f"if [ ! -f {quoted} ]; then echo ABSENT; exit 0; fi; "
        f"stat -c %s -- {quoted}; sha256sum -- {quoted} | cut -d' ' -f1"
    )

    try:
        if node.local:
            proc = _run(["bash", "-lc", remote], timeout)
        else:
            proc = _run(_ssh_cmd(node, remote, connect_timeout), timeout)
    except subprocess.TimeoutExpired:
        rec.status = "UNAVAILABLE"
        rec.message = (
            f"node {node.node_id} ({node.ssh_address}) did not answer the digest "
            f"check within {timeout}s; the dataset at {frozen.path} is "
            f"UNVERIFIABLE, which is not the same as clean (VIR-5)"
        )
        return rec
    except OSError as exc:
        rec.status = "UNAVAILABLE"
        rec.message = (
            f"could not run the digest check for node {node.node_id} "
            f"({node.ssh_address}): {exc}"
        )
        return rec

    if proc.returncode != 0:
        rec.status = "UNAVAILABLE"
        rec.message = (
            f"node {node.node_id} ({node.ssh_address}) unreachable or the digest "
            f"check failed (rc={proc.returncode}): "
            f"{(proc.stderr or proc.stdout).strip()[:400]}"
        )
        return rec

    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]

    if lines and lines[0] == "ABSENT":
        rec.status = "FAIL"
        rec.message = (
            f"dataset ABSENT on node {node.node_id} ({node.ssh_address}): "
            f"{frozen.path} does not exist. Expected version "
            f"{frozen.version_id or '<unversioned>'} "
            f"sha256={frozen.sha256} ({frozen.size_bytes} bytes)."
        )
        return rec

    if len(lines) < 2:
        rec.status = "INCOMPLETE"
        rec.message = (
            f"node {node.node_id} ({node.ssh_address}) returned an unusable "
            f"digest-check result for {frozen.path}: {proc.stdout.strip()[:400]!r}"
        )
        return rec

    try:
        rec.size_bytes = int(lines[0])
    except ValueError:
        rec.status = "INCOMPLETE"
        rec.message = (
            f"node {node.node_id}: unparseable size {lines[0]!r} for {frozen.path}"
        )
        return rec
    rec.digest = lines[1].strip()

    if rec.digest != frozen.sha256:
        rec.status = "FAIL"
        rec.message = (
            f"dataset DIGEST MISMATCH on node {node.node_id} "
            f"({node.ssh_address}) at {frozen.path}: "
            f"expected {frozen.sha256} ({frozen.size_bytes} bytes), "
            f"got {rec.digest} ({rec.size_bytes} bytes)"
        )
        return rec

    if rec.size_bytes != frozen.size_bytes:
        # Contract §3 calls this impossible and requires it be treated as a bug
        # in the checker rather than reconciled. Fail closed.
        rec.status = "FAIL"
        rec.message = (
            f"node {node.node_id}: digest matches but size differs "
            f"({rec.size_bytes} vs expected {frozen.size_bytes}) for "
            f"{frozen.path}. Contract §3: impossible — treat as a checker bug "
            f"and fail closed."
        )
        return rec

    rec.status = "PASS"
    rec.message = (
        f"node {node.node_id}: {frozen.path} verified on target — "
        f"sha256={rec.digest} ({rec.size_bytes} bytes)"
    )
    return rec


def provision_node_dataset(
    frozen: FrozenDataset,
    node: NodeSpec,
    *,
    timeout: float = 900.0,
    connect_timeout: int = 10,
) -> NodeVerification:
    """Place the frozen immutable version on `node`, then verify it **there**.

    The destination is `frozen.path` — the same absolute path the coordinator
    dispatches. That is not a convenience: an absolute path in the assignment
    payload only resolves on the worker if the file is at that exact path, so
    "where the file goes" and "what gets dispatched" are the same fact and must
    not be two independently configurable ones.

    Provisioning is unconditional. A node whose copy already matches is
    provisioned again anyway (P0.5 §5): a provisioning step that skips a node it
    believes is already correct is a provisioning step that cannot detect the
    case it exists for. `.122` is exactly that case — its copy was hand-placed
    during Phase 6.0 and *happens* to match.
    """
    rec = NodeVerification(
        node_id=node.node_id,
        ssh_address=node.ssh_address,
        dataset_path=frozen.path,
        status="INCOMPLETE",
        expected_digest=frozen.sha256,
        expected_size_bytes=frozen.size_bytes,
    )

    dest_dir = os.path.dirname(frozen.path)
    try:
        # `mkdir -p` BEFORE the copy that fills it — SSH is stateless and order
        # matters (CLAUDE.md §2 deploy rule).
        mk = _run(_ssh_cmd(node, f"mkdir -p -- {shlex.quote(dest_dir)}", connect_timeout),
                  timeout=connect_timeout + 20)
        if mk.returncode != 0:
            rec.status = "UNAVAILABLE"
            rec.message = (
                f"could not create {dest_dir} on node {node.node_id} "
                f"({node.ssh_address}): {(mk.stderr or mk.stdout).strip()[:300]}"
            )
            return rec

        # Absolute paths on both sides — the SFTP-backed scp does not expand `~`
        # reliably (CLAUDE.md §2).
        cp = _run(
            ["scp", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={connect_timeout}",
             frozen.path, f"{node.target()}:{frozen.path}"],
            timeout=timeout,
        )
        if cp.returncode != 0:
            rec.status = "UNAVAILABLE"
            rec.message = (
                f"transfer to node {node.node_id} ({node.ssh_address}) failed: "
                f"{(cp.stderr or cp.stdout).strip()[:300]}"
            )
            return rec
    except subprocess.TimeoutExpired:
        rec.status = "UNAVAILABLE"
        rec.message = (
            f"provisioning node {node.node_id} ({node.ssh_address}) timed out "
            f"after {timeout}s"
        )
        return rec
    except OSError as exc:
        rec.status = "UNAVAILABLE"
        rec.message = f"provisioning node {node.node_id} failed to launch: {exc}"
        return rec

    verified = verify_node_dataset(frozen, node,
                                   connect_timeout=connect_timeout)
    verified.provisioned = True
    return verified


def fleet_preflight(
    frozen: FrozenDataset,
    nodes: Sequence[NodeSpec],
    *,
    timeout: float = 120.0,
    connect_timeout: int = 10,
) -> List[NodeVerification]:
    """Verify every node, then fail closed if any is not `PASS` (requirement 4).

    Every node is checked before anything is raised, so one operator round trip
    reports the whole fleet rather than the first broken node. The raise itself
    is `DatasetProvisioningError`, which is a `ResidueError`, so the coordinator's
    existing non-retryable control flow is preserved — while the operational
    category (a provisioning problem, not a residue problem) survives in the
    type. That is Beta's P0-ruling §3 correction.
    """
    records = [
        verify_node_dataset(frozen, n, timeout=timeout, connect_timeout=connect_timeout)
        for n in nodes
    ]
    bad = [r for r in records if r.status != "PASS"]
    if bad:
        detail = "\n".join(f"  [{r.status}] {r.message}" for r in bad)
        raise _dataset_provisioning_error()(
            "FAIL BEFORE DISPATCH — the dataset is not verified on every node.\n"
            f"dataset: {frozen.describe()}\n"
            f"{len(bad)} of {len(records)} node(s) did not pass:\n{detail}\n"
            "No worker was dispatched, no GPU work started, no spool created "
            "(RUNTIME_DATASET_PROVISIONING_CONTRACT §2)."
        )
    return records


# ===========================================================================
# The provisioning manifest (contract §1)
# ===========================================================================

def default_provisioning_manifest_path(repo_root: Optional[str] = None) -> str:
    root = repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, PROVISIONING_MANIFEST_NAME)


def load_provisioning_nodes(
    manifest_path: Optional[str] = None,
    *,
    dataset_logical_name: str = "daily3",
) -> Optional[List[NodeSpec]]:
    """Read the fleet from the provisioning manifest.

    Returns `None` when no manifest exists — a distinct answer from "a manifest
    that declares zero nodes", and the caller must treat the two differently:
    absent means fleet verification is UNAVAILABLE and must be recorded as such,
    never quietly reported as clean. What each of those two answers *costs* is
    the caller's decision and lives in `resolve_absent_fleet_status()`, because
    only the caller knows whether this run has a fleet to verify.

    The other two of Beta's four conditions — **unreadable** and **invalid** —
    are decided here, because they are not a question of topology: a manifest
    that cannot be read or cannot be understood establishes nothing for anyone,
    so it is fatal on every path. They were already fatal before this change,
    but as a bare `OSError` / `JSONDecodeError` / `KeyError` escaping two frames
    below the gate — unclassified, and outside the `except` clauses both callers
    already had. They are now `DatasetProvisioningError`, chained to the
    original, naming the absolute path that was read.
    """
    path = os.path.abspath(manifest_path or default_provisioning_manifest_path())
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r") as f:
            doc = json.load(f)
    except OSError as exc:
        raise _dataset_provisioning_error()(
            f"provisioning manifest UNREADABLE at {path}: {exc}. The fleet this "
            f"run must verify cannot be established."
        ) from exc
    except ValueError as exc:            # json.JSONDecodeError is a ValueError
        raise _dataset_provisioning_error()(
            f"provisioning manifest INVALID (not parseable JSON) at {path}: "
            f"{exc}. The fleet this run must verify cannot be established."
        ) from exc

    if not isinstance(doc, dict):
        raise _dataset_provisioning_error()(
            f"provisioning manifest INVALID at {path}: top level is "
            f"{type(doc).__name__}, expected an object."
        )

    schema_version = doc.get("manifest_schema_version")
    if schema_version not in SUPPORTED_MANIFEST_SCHEMA_VERSIONS:
        raise _dataset_provisioning_error()(
            f"provisioning manifest INVALID at {path}: declares "
            f"manifest_schema_version={schema_version!r}; supported: "
            f"{sorted(SUPPORTED_MANIFEST_SCHEMA_VERSIONS)}"
        )

    datasets = doc.get("datasets", [])
    if not isinstance(datasets, list):
        raise _dataset_provisioning_error()(
            f"provisioning manifest INVALID at {path}: 'datasets' is "
            f"{type(datasets).__name__}, expected a list."
        )

    for entry in datasets:
        if not isinstance(entry, dict):
            raise _dataset_provisioning_error()(
                f"provisioning manifest INVALID at {path}: a 'datasets' entry "
                f"is {type(entry).__name__}, expected an object."
            )
        if entry.get("dataset_logical_name") != dataset_logical_name:
            continue
        declared = entry.get("nodes", [])
        if not isinstance(declared, list):
            raise _dataset_provisioning_error()(
                f"provisioning manifest INVALID at {path}: 'nodes' for "
                f"{dataset_logical_name!r} is {type(declared).__name__}, "
                f"expected a list."
            )
        nodes = []
        for n in declared:
            try:
                nodes.append(NodeSpec(
                    node_id=n["node_id"],
                    ssh_address=n["ssh_address"],
                    ssh_user=n.get("ssh_user", "michael"),
                    local=bool(n.get("local", False)),
                ))
            except (KeyError, TypeError, AttributeError) as exc:
                raise _dataset_provisioning_error()(
                    f"provisioning manifest INVALID at {path}: node entry "
                    f"{n!r} for {dataset_logical_name!r} is unusable ({exc!r}). "
                    f"Each node requires node_id and ssh_address."
                ) from exc
        return nodes
    return []


def resolve_absent_fleet_status(
    nodes: Optional[Sequence[NodeSpec]],
    *,
    manifest_path: str,
    miner_backed: bool,
    remote_execution: Optional[bool] = None,
    require_fleet: bool = False,
    context: str = "",
) -> str:
    """Decide what an unusable provisioning manifest COSTS this run.

    Called only when no per-node verification will run — `nodes` is `None`
    (manifest missing) or `[]` (manifest present, declares no nodes). The other
    two of Beta's four conditions raise inside `load_provisioning_nodes()` and
    never arrive here.

    Beta's P0.5 closure ruling, which this function is:

      > A missing, unreadable, invalid, or empty provisioning manifest means the
      > system cannot establish which worker datasets must be verified.
      > Recording `UNAVAILABLE` and proceeding **violates the authority
      > boundary.**

    So for a miner-backed run this raises, before any coordinator is
    constructed and before any worker is dispatched. For everything else the
    answer is a *status*, and which status is the second half of the ruling:
    `NOT_APPLICABLE` when the caller declares it performs no remote execution
    (there was never a fleet to verify), `UNAVAILABLE` otherwise — including
    when the caller does not know, because "unknown" must keep the
    over-constrained reading rather than quietly earn the clean one.

    NOTE (scope): `remote_execution=False` is a statement about **topology**,
    not a bypass. It is not, and must not become, the "local run verifies only
    the local node" refinement — that is Beta's Q1, explicitly not authorized.
    A local run that still drives the 26-GPU coordinator performs remote
    execution and must not declare otherwise.
    """
    if nodes:
        raise ValueError(
            "resolve_absent_fleet_status() called with a non-empty node list; "
            "run fleet_preflight() instead"
        )

    if nodes is None:
        condition = "MISSING — no manifest exists at that path"
    else:
        condition = ("EMPTY — the manifest exists but declares no nodes for "
                     "this dataset")

    if miner_backed or require_fleet:
        why = ("a miner-backed run" if miner_backed
               else "fleet verification was required")
        raise _dataset_provisioning_error()(
            "FAIL BEFORE DISPATCH — the provisioning manifest is "
            f"{condition}.\n"
            f"expected provisioning manifest: {manifest_path}\n"
            f"{('run: ' + context + chr(10)) if context else ''}"
            f"This is {why}: the system cannot establish which worker datasets "
            "must be verified, so it cannot verify them. Recording UNAVAILABLE "
            "and proceeding would violate the authority boundary (Beta, P0.5 "
            "closure ruling).\n"
            "No coordinator was constructed, no worker was dispatched, no GPU "
            "work started, no spool created "
            "(RUNTIME_DATASET_PROVISIONING_CONTRACT §2)."
        )

    if remote_execution is False:
        return FLEET_STATUS_NOT_APPLICABLE
    return FLEET_STATUS_UNAVAILABLE


# ===========================================================================
# The Resolved Execution Set — this module is a CONSUMER of it, not an author
# ===========================================================================
# P0.5 was the ONE mechanism of the six that had been updated for the Proxmox
# migration, so it alone verified the CT100 endpoints while three others still
# checked the bare-metal addresses. It is not being retired or loosened here:
# its fail-before-dispatch behaviour, its per-node on-target digest re-derivation
# and its Beta-ratified UNAVAILABLE / NOT_APPLICABLE vocabulary are untouched.
# What changes is only WHICH NODES ARE THIS RUN'S — which was never P0.5's
# decision to make, and is now the resolved set's.
#
# Imported lazily and defensively: `dataset_authority` must stay importable on
# its own (workers, harnesses, a bare clone), and a missing resolver must degrade
# to today's behaviour rather than break the dataset authority.

def _active_execution_set():
    try:
        from execution_set import active_execution_set
    except Exception:                                    # noqa: BLE001
        return None
    return active_execution_set()


def _active_execution_set_provenance():
    try:
        from execution_set import execution_set_provenance
    except Exception:                                    # noqa: BLE001
        return None
    return execution_set_provenance()


# ===========================================================================
# Run provenance (requirement 6)
# ===========================================================================

def run_provenance_record(
    run_label: str,
    frozen: FrozenDataset,
    node_records: Optional[Sequence[NodeVerification]] = None,
    *,
    fleet_status: str = "UNAVAILABLE",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The provenance object: the frozen values, read back rather than assumed."""
    doc: Dict[str, Any] = {
        "provenance_schema_version": 1,
        "phase": "6-P0.5",
        "run_label": run_label,
        "recorded_utc": _utc_now_iso(),
        "recorded_on_node": local_node_identity(),
        "frozen_dataset": frozen.to_provenance(),
        "fleet_status": fleet_status,
        "fleet": [r.to_provenance() for r in (node_records or [])],
    }
    # [Resolved Execution Set] The run's frozen fleet authority, READ BACK from
    # the freeze rather than reconstructed here — so the record shows what the
    # run actually verified against, and the "which nodes were these?" question
    # has one answer that survives the run. None when no set is frozen (a direct
    # harness call); both production entry points always freeze one.
    _xset_prov = _active_execution_set_provenance()
    if _xset_prov is not None:
        doc["execution_set"] = _xset_prov
    if extra:
        doc["extra"] = dict(extra)
    return doc


def write_run_provenance(
    run_label: str,
    frozen: FrozenDataset,
    node_records: Optional[Sequence[NodeVerification]] = None,
    *,
    fleet_status: str = "UNAVAILABLE",
    repo_root: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist the run provenance record and return its path.

    Written under `dataset_provenance/` with a `.json` name, so `.gitignore:41`
    keeps it out of `git status --porcelain` and it can never dirty the working
    tree or block certification at the finalizer wall.
    """
    root = repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, RUN_PROVENANCE_DIRNAME)
    os.makedirs(out_dir, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", run_label or "run")
    out_path = os.path.join(out_dir, f"{safe}.json")

    doc = run_provenance_record(run_label, frozen, node_records,
                               fleet_status=fleet_status, extra=extra)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path)
    logger.info("[P0.5] run provenance written: %s", out_path)
    return out_path


# ===========================================================================
# The run-start gate — what callers actually call
# ===========================================================================

def run_start_dataset_gate(
    dataset_arg: str,
    *,
    run_label: str,
    require_fleet: bool = False,
    miner_backed: bool = False,
    remote_execution: Optional[bool] = None,
    provisioning_manifest: Optional[str] = None,
    repo_root: Optional[str] = None,
    allow_unpublished_alias: bool = False,
    write_provenance: bool = True,
    execution_set: Optional[Any] = None,
) -> FrozenDataset:
    """Resolve, freeze, verify the fleet, record provenance — before any dispatch.

    Returns the `FrozenDataset` whose `.path` the caller must dispatch. Raises
    before returning if anything is wrong, which is the whole contract: nothing
    downstream of this call has to re-check, because nothing downstream of this
    call runs if the check failed.

    `require_fleet=False` with a manifest present still verifies the fleet — the
    flag only governs what happens when **no usable manifest exists**.

    `miner_backed=True` makes all four of Beta's conditions — manifest missing,
    unreadable, invalid, empty — fatal here, which is the P0.5 closure ruling.
    This is the last point at which nothing has been allocated, so the refusal
    is free: the caller has not constructed a coordinator and has not dispatched
    a worker, because the caller has not returned from this function.

    `remote_execution=False` is the caller declaring it has no fleet at all; the
    absence is then `NOT_APPLICABLE`, not `UNAVAILABLE`. Unknown (the default,
    `None`) keeps the over-constrained reading. See
    `resolve_absent_fleet_status()` — including why this is not a local-run
    bypass.

    `execution_set` (or the process-frozen one) supersedes the provisioning
    manifest as the source of VERIFICATION TARGETS — never as the source of
    leniency. See the block below.
    """
    frozen = freeze_for_run(
        dataset_arg,
        run_label=run_label,
        allow_unpublished_alias=allow_unpublished_alias,
    )

    # -----------------------------------------------------------------
    # WHICH NODES ARE THIS RUN'S? — the resolved execution set decides.
    # -----------------------------------------------------------------
    # Beta's Q1 refinement arrives HERE and only through the shared resolver:
    # a run whose set is one local node verifies THAT node; a run whose set is
    # three rigs verifies three. The set decides, not a flag — `require_fleet`
    # is not weakened and P0.5 is not special-cased. A one-node set that fails
    # still refuses: nothing below makes a failing node passable.
    #
    # `load_provisioning_nodes` is NOT retired — it remains the source when no
    # set is frozen (harnesses, direct calls, every pre-existing test), and the
    # resolver cross-checks the profile map against this same manifest, so the
    # two can no longer disagree silently.
    # SCOPE, precisely: the set decides WHICH NODES ARE VERIFIED. It does not
    # decide whether the provisioning authority boundary applies. Beta's P0.5
    # closure ruling — an unusable manifest is fatal for a miner-backed run,
    # because the system cannot establish which WORKER datasets must be
    # provisioned and verified — is preserved verbatim below and is still
    # reached before any node is contacted. The one case the set legitimately
    # answers on its own is a set with NO remote node: there is no worker
    # dataset to establish, so the remote-provisioning record is genuinely
    # NOT_APPLICABLE and the local node is verified directly. That is Q1, and it
    # is the only behaviour that changes.
    xset = execution_set if execution_set is not None else _active_execution_set()
    records: List[NodeVerification] = []

    # Raises (unreadable / invalid) before returning; None = missing, [] = empty.
    nodes = load_provisioning_nodes(provisioning_manifest)

    if xset is not None and (nodes or not xset.remote_execution):
        targets = xset.dataset_verification_targets()
        logger.info("[P0.5] verification targets from execution set %s "
                    "(profile=%s%s): %s",
                    xset.set_id()[:12], xset.rig_profile,
                    ", PARTIAL" if xset.partial else "",
                    [n.node_id for n in targets])
        records = fleet_preflight(frozen, targets)       # raises on any non-PASS
        fleet_status = FLEET_STATUS_PASS
        if write_provenance:
            write_run_provenance(run_label, frozen, records,
                                 fleet_status=fleet_status, repo_root=repo_root)
        return frozen

    if not nodes:
        manifest_path = os.path.abspath(
            provisioning_manifest or default_provisioning_manifest_path(repo_root)
        )
        # Raises for a miner-backed run — before any coordinator construction,
        # before any dispatch, and before any provenance is written.
        fleet_status = resolve_absent_fleet_status(
            nodes,
            manifest_path=manifest_path,
            miner_backed=miner_backed,
            remote_execution=remote_execution,
            require_fleet=require_fleet,
            context=run_label,
        )
        logger.warning(
            "[P0.5] provisioning manifest %s at %s — per-node dataset "
            "verification did NOT run for %s. Recorded as %s.",
            "missing" if nodes is None else "declares no nodes",
            manifest_path, frozen.dataset_logical_name, fleet_status,
        )
    else:
        records = fleet_preflight(frozen, nodes)     # raises on any non-PASS
        fleet_status = FLEET_STATUS_PASS

    if write_provenance:
        write_run_provenance(run_label, frozen, records,
                             fleet_status=fleet_status, repo_root=repo_root)
    return frozen
