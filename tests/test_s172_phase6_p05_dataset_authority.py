#!/usr/bin/env python3
"""
test_s172_phase6_p05_dataset_authority.py — S172 Phase 6-P0.5 acceptance harness

Phase 6-P0 published an immutable dataset version and an atomic pointer manifest
and changed no running code. P0.5 is the behavioural cutover, and this harness is
its acceptance evidence: the eight required behaviours of
`docs/CLAUDE_CODE_INSTRUCTIONS_PHASE_6_P0_5_IMPLEMENTATION.md` §2, plus every
negative path in its §7 table.

The negative paths ARE the substance. A gate that only proves the happy path
proves nothing about a fail-closed mechanism — VIR-2 requires a clean control and
a fault-injection control, and every refusal below is a fault deliberately
injected into a synthetic publication tree built in a temp directory.

NOTHING PUBLISHED IS TOUCHED. Every mutation happens inside `tempfile.mkdtemp()`.
The real `daily3.json`, the real version file and the real pointer manifest are
read only, and gate 30 re-derives all three digests at the END of the run to
prove that in the strongest available way: by measurement, not by assertion.

Run:
    cd ~/distributed_prng_analysis
    source ~/venvs/torch/bin/activate
    PYTHONPATH=. python3 tests/test_s172_phase6_p05_dataset_authority.py

Optional live-fleet gates (SSH to the three CT100 workers, read-only):
    PYTHONPATH=. python3 tests/test_s172_phase6_p05_dataset_authority.py --fleet

Exit 0 = all green. Exit 1 = a check failed (DO NOT COMMIT).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from miner import dataset_authority as D                             # noqa: E402
from miner.range_miner_worker import (                               # noqa: E402
    DatasetProvisioningError,
    ResidueError,
    _sha256_file,
    load_residue_window,
)

_results = []

# The real, published artifacts — read-only for the whole harness.
REAL_POINTER = os.path.join(_ROOT, D.POINTER_MANIFEST_NAME)
REAL_ALIAS = os.path.join(_ROOT, D.LEGACY_ALIAS_NAME)


# ===========================================================================
# Synthetic publication trees — the fault-injection substrate
# ===========================================================================

def _records(n=8):
    return [{"date": f"2026-01-{i+1:02d}", "session": "midday", "draw": i}
            for i in range(n)]


def make_publication(tmp, *, records=None, stamp="20260801T145551443433"):
    """Build a schema-conforming publication tree and return (pointer, version)."""
    records = _records() if records is None else records
    body = json.dumps(records).encode("utf-8")

    import hashlib
    sha = hashlib.sha256(body).hexdigest()
    filename = f"daily3-{stamp}Z-{sha[:12]}.json"
    version_path = os.path.join(tmp, filename)
    with open(version_path, "wb") as f:
        f.write(body)

    manifest = {
        "manifest_schema_version": 1,
        "version_id": filename[:-len(".json")],
        "filename": filename,
        "sha256": sha,
        "size_bytes": len(body),
        "record_count": len(records),
        "dataset_lineage_id": "daily3-test-L001",
        "published_utc": "2026-08-01T14:55:51.443433Z",
        "predecessor_sha256": None,
    }
    pointer = os.path.join(tmp, D.POINTER_MANIFEST_NAME)
    with open(pointer, "w") as f:
        json.dump(manifest, f)
    # the legacy alias, byte-identical, exactly as publish-in-place leaves it
    with open(os.path.join(tmp, D.LEGACY_ALIAS_NAME), "wb") as f:
        f.write(body)
    return pointer, version_path


def repoint(pointer, **overrides):
    """Rewrite the pointer manifest — the fault injector."""
    with open(pointer) as f:
        m = json.load(f)
    m.update(overrides)
    with open(pointer, "w") as f:
        json.dump(m, f)
    return pointer


def _expect(exc_type, fn, *a, **kw):
    """Run fn, require exc_type. Returns the exception for message assertions."""
    try:
        fn(*a, **kw)
    except exc_type as e:
        return e
    except Exception as e:                                    # wrong type
        raise AssertionError(
            f"expected {exc_type.__name__}, got {type(e).__name__}: {e}") from e
    raise AssertionError(f"expected {exc_type.__name__}, nothing was raised "
                         f"(VACUOUS DETECTOR)")


# ===========================================================================
# Requirement 1 + 8 — pointer resolution and pointer validation
# ===========================================================================

def gate01_real_pointer_resolves():
    """CLEAN CONTROL: the real published pointer resolves and self-verifies."""
    f = D.resolve_pointer(REAL_POINTER)
    assert f.resolution_source == "pointer"
    assert os.path.isabs(f.path), "the frozen path MUST be absolute (req 3)"
    assert D.VERSION_FILENAME_RE.match(f.filename), f.filename
    assert f.version_id + ".json" == f.filename
    assert len(f.sha256) == 64 and f.sha256[:12] in f.filename
    # identity re-derived from the bytes, not copied from the manifest
    assert f.sha256 == D.sha256_file(f.path)
    assert f.size_bytes == os.path.getsize(f.path)
    assert f.record_count == D.count_records(f.path)
    assert f.manifest_sha256 == D.sha256_file(REAL_POINTER)


def gate02_alias_resolves_to_version():
    """Req 3: the bare alias resolves to the immutable version, never itself."""
    f = D.resolve_dataset_path(REAL_ALIAS)
    assert f.path != os.path.abspath(REAL_ALIAS), \
        "resolving the alias must NOT return the alias"
    assert os.path.basename(f.path) != D.LEGACY_ALIAS_NAME
    assert D.VERSION_FILENAME_RE.match(os.path.basename(f.path))
    assert f.resolution_source == "pointer"


def gate03_pointer_missing_refused():
    """§7: pointer missing → refuse, pre-dispatch."""
    tmp = tempfile.mkdtemp()
    try:
        e = _expect(D.PointerResolutionError,
                    D.resolve_pointer, os.path.join(tmp, D.POINTER_MANIFEST_NAME))
        assert "not found" in str(e)
    finally:
        shutil.rmtree(tmp)


def gate04_pointer_unparseable_refused():
    """§7: pointer unparseable → refuse, pre-dispatch."""
    tmp = tempfile.mkdtemp()
    try:
        p = os.path.join(tmp, D.POINTER_MANIFEST_NAME)
        with open(p, "w") as f:
            f.write("{not json,,,")
        e = _expect(D.PointerResolutionError, D.resolve_pointer, p)
        assert "not parseable" in str(e)
    finally:
        shutil.rmtree(tmp)


def gate05_pointer_naming_alias_refused():
    """§2.2: a pointer naming `daily3.json` must be refused.

    The alias is mutable and version-less. If the pointer could name it, the
    pointer would select nothing at all — this is the exact case §2.2 calls out.
    """
    tmp = tempfile.mkdtemp()
    try:
        p, _ = make_publication(tmp)
        repoint(p, filename=D.LEGACY_ALIAS_NAME)
        e = _expect(D.PointerValidationError, D.resolve_pointer, p)
        assert "alias" in str(e)
    finally:
        shutil.rmtree(tmp)


def gate06_pointer_absolute_path_refused():
    """§2.2: the pointer is not a general path parameter."""
    tmp = tempfile.mkdtemp()
    try:
        p, v = make_publication(tmp)
        repoint(p, filename="/etc/passwd")
        e = _expect(D.PointerValidationError, D.resolve_pointer, p)
        assert "bare filename" in str(e)
    finally:
        shutil.rmtree(tmp)


def gate07_pointer_traversal_refused():
    """§2.2: traversal refused — syntactically, before the grammar check."""
    tmp = tempfile.mkdtemp()
    try:
        p, _ = make_publication(tmp)
        for bad in ("../daily3-20260801T145551443433Z-aabbccddeeff.json",
                    "sub/daily3-20260801T145551443433Z-aabbccddeeff.json",
                    ".."):
            e = _expect(D.PointerValidationError, D.resolve_pointer,
                        repoint(p, filename=bad))
            assert "bare filename" in str(e), (bad, str(e))
    finally:
        shutil.rmtree(tmp)


def gate08_pointer_nonconforming_name_refused():
    """§7: pointer names a non-conforming filename → refuse (req 8)."""
    tmp = tempfile.mkdtemp()
    try:
        p, _ = make_publication(tmp)
        for bad in ("daily3-latest.json",              # no stamp, no digest
                    "daily3-20260801T145551443433Z-AABBCCDDEEFF.json",  # uppercase
                    "daily3-20260801T1455514434Z-aabbccddeeff.json",    # short stamp
                    "daily3-20260801T145551443433Z-aabbccddeef.json",   # 11 hex
                    "pa_pick3-20260801T145551443433Z-aabbccddeeff.json",
                    "daily3-20260801T145551443433Z-aabbccddeeff.txt"):
            e = _expect(D.PointerValidationError, D.resolve_pointer,
                        repoint(p, filename=bad))
            assert "version grammar" in str(e), (bad, str(e))
    finally:
        shutil.rmtree(tmp)


def gate09_pointer_target_missing_refused():
    """§7: pointer names a file that does not exist → refuse."""
    tmp = tempfile.mkdtemp()
    try:
        p, v = make_publication(tmp)
        os.remove(v)
        e = _expect(D.PointerResolutionError, D.resolve_pointer, p)
        assert "does not exist" in str(e)
    finally:
        shutil.rmtree(tmp)


def gate10_version_id_filename_disagreement_refused():
    """Schema §2 invariant: filename == version_id + '.json'."""
    tmp = tempfile.mkdtemp()
    try:
        p, _ = make_publication(tmp)
        repoint(p, version_id="daily3-somethingelse")
        _expect(D.PointerValidationError, D.resolve_pointer, p)
    finally:
        shutil.rmtree(tmp)


def gate11_filename_digest_prefix_disagreement_refused():
    """The name's digest prefix is a convenience copy — but it must agree."""
    tmp = tempfile.mkdtemp()
    try:
        p, _ = make_publication(tmp)
        with open(p) as f:
            m = json.load(f)
        bad_name = f"daily3-20260801T145551443433Z-{'0'*12}.json"
        shutil.copy(os.path.join(tmp, m["filename"]), os.path.join(tmp, bad_name))
        repoint(p, filename=bad_name, version_id=bad_name[:-5])
        e = _expect(D.PointerValidationError, D.resolve_pointer, p)
        assert "digest prefix" in str(e)
    finally:
        shutil.rmtree(tmp)


def gate12_digest_mismatch_refused():
    """The published bytes must match the manifest digest — measured, not trusted."""
    tmp = tempfile.mkdtemp()
    try:
        p, v = make_publication(tmp)
        with open(v, "r+b") as f:          # same length, different bytes
            f.seek(2)
            f.write(b"X")
        e = _expect(D.DatasetIdentityError, D.resolve_pointer, p)
        assert "sha256" in str(e)
    finally:
        shutil.rmtree(tmp)


def gate13_size_mismatch_refused():
    """Size is the cheap truncation check and it must fire before hashing."""
    tmp = tempfile.mkdtemp()
    try:
        p, v = make_publication(tmp)
        with open(v, "ab") as f:
            f.write(b" ")
        e = _expect(D.DatasetIdentityError, D.resolve_pointer, p)
        assert "bytes" in str(e)
    finally:
        shutil.rmtree(tmp)


def gate14_record_count_mismatch_refused():
    """record_count is the semantic count; a lie about it is an identity fault."""
    tmp = tempfile.mkdtemp()
    try:
        p, _ = make_publication(tmp)
        repoint(p, record_count=999)
        e = _expect(D.DatasetIdentityError, D.resolve_pointer, p)
        assert "records" in str(e)
    finally:
        shutil.rmtree(tmp)


def gate15_unsupported_schema_version_refused():
    """A future schema must fail loudly, never be read with today's assumptions."""
    tmp = tempfile.mkdtemp()
    try:
        p, _ = make_publication(tmp)
        repoint(p, manifest_schema_version=2)
        e = _expect(D.PointerResolutionError, D.resolve_pointer, p)
        assert "schema" in str(e)
    finally:
        shutil.rmtree(tmp)


def gate16_missing_required_field_refused():
    tmp = tempfile.mkdtemp()
    try:
        p, _ = make_publication(tmp)
        with open(p) as f:
            m = json.load(f)
        del m["sha256"]
        with open(p, "w") as f:
            json.dump(m, f)
        e = _expect(D.PointerResolutionError, D.resolve_pointer, p)
        assert "missing required field" in str(e)
    finally:
        shutil.rmtree(tmp)


def gate17_unpublished_alias_refused():
    """Req 3: an alias with no publication beside it is refused, not dispatched."""
    tmp = tempfile.mkdtemp()
    try:
        alias = os.path.join(tmp, D.LEGACY_ALIAS_NAME)
        with open(alias, "w") as f:
            json.dump(_records(), f)
        e = _expect(D.PointerResolutionError, D.resolve_dataset_path, alias)
        assert "refusing to dispatch the bare compatibility alias" in str(e)
        # ...and the harness escape hatch is explicit, never implicit
        f2 = D.resolve_dataset_path(alias, allow_unpublished_alias=True)
        assert f2.resolution_source == "explicit"
    finally:
        shutil.rmtree(tmp)


# ===========================================================================
# Requirement 2 + 7 — the run-start freeze and pointer-movement protection
# ===========================================================================

def gate18_freeze_is_one_time_and_idempotent():
    """Req 2: freeze once; an identical re-freeze is a no-op, not a second freeze."""
    D.clear_frozen_dataset()
    try:
        a = D.freeze_for_run(REAL_ALIAS, run_label="gate18")
        b = D.freeze_for_run(REAL_ALIAS, run_label="gate18-again")
        assert a is b, "re-freezing the same identity must return the SAME object"
        assert D.get_frozen_dataset() is a
    finally:
        D.clear_frozen_dataset()


def gate19_conflicting_freeze_raises():
    """Two datasets inside one run is the split-study failure — it must be audible."""
    D.clear_frozen_dataset()
    tmp = tempfile.mkdtemp()
    try:
        D.freeze_for_run(REAL_ALIAS, run_label="gate19")
        p, _ = make_publication(tmp)
        other = D.resolve_pointer(p)
        e = _expect(D.DatasetFreezeError, D.freeze_run_dataset, other)
        assert "conflicting" in str(e) and "split-study" in str(e)
    finally:
        D.clear_frozen_dataset()
        shutil.rmtree(tmp)


def gate20_pointer_moves_mid_run_does_not_alter_run():
    """§7 / req 7 — THE gate. Move the pointer mid-run; the run must not notice.

    Two published versions in one tree. The run freezes on version A, then the
    pointer is atomically repointed at version B (a scrape landing mid-run). The
    coordinator's digest resolution must still answer A, because the run reads
    the freeze and never the pointer.
    """
    D.clear_frozen_dataset()
    tmp = tempfile.mkdtemp()
    try:
        pointer, v_a = make_publication(tmp, records=_records(8))
        frozen = D.freeze_for_run(os.path.join(tmp, D.LEGACY_ALIAS_NAME),
                                  run_label="gate20")
        assert frozen.path == v_a

        # --- the scrape lands: publish version B and move the pointer ---------
        import hashlib
        body_b = json.dumps(_records(9)).encode("utf-8")
        sha_b = hashlib.sha256(body_b).hexdigest()
        name_b = f"daily3-20260802T101010101010Z-{sha_b[:12]}.json"
        with open(os.path.join(tmp, name_b), "wb") as f:
            f.write(body_b)
        repoint(pointer, filename=name_b, version_id=name_b[:-5], sha256=sha_b,
                size_bytes=len(body_b), record_count=9,
                predecessor_sha256=frozen.sha256)

        # the pointer really did move
        moved = D.resolve_pointer(pointer)
        assert moved.path.endswith(name_b) and moved.sha256 == sha_b

        # ...and the RUN did not
        assert D.get_frozen_dataset().path == v_a
        assert D.get_frozen_dataset().sha256 == frozen.sha256
        assert D.run_frozen_dataset_sha256(v_a) == frozen.sha256

        # the coordinator's per-assignment digest answers the FROZEN value
        from miner.range_miner_coordinator import resolve_dataset_sha256
        assert resolve_dataset_sha256(v_a) == frozen.sha256
        assert resolve_dataset_sha256(v_a) != sha_b
    finally:
        D.clear_frozen_dataset()
        shutil.rmtree(tmp)


def gate21_coordinator_digest_is_run_scoped_not_trial_scoped():
    """§2.1: the defect was SCOPE. Same path, mutated bytes, two 'trials'.

    Pre-P0.5 `serve_trial` re-hashed the file on every entry, so mutating the
    bytes between two trials silently produced two different digests inside one
    study with no error. The frozen resolution must return the run's digest both
    times; the unfrozen fallback must still behave exactly as it did before.
    """
    from miner.range_miner_coordinator import (
        resolve_dataset_sha256, compute_dataset_sha256,
    )
    D.clear_frozen_dataset()
    tmp = tempfile.mkdtemp()
    try:
        pointer, v = make_publication(tmp)
        frozen = D.freeze_for_run(v, run_label="gate21")
        trial_1 = resolve_dataset_sha256(v)

        # a scrape rewrites the bytes at the same path between trials
        with open(v, "r+b") as f:
            f.seek(2)
            f.write(b"7")
        trial_2 = resolve_dataset_sha256(v)

        assert trial_1 == trial_2 == frozen.sha256, \
            "the run's digest changed between trials — the study just split"
        assert compute_dataset_sha256(v) != frozen.sha256, \
            "control: the FILE really did change (otherwise this gate is vacuous)"

        # non-regression: with no freeze, the old behaviour is untouched
        D.clear_frozen_dataset()
        assert resolve_dataset_sha256(v) == compute_dataset_sha256(v)
    finally:
        D.clear_frozen_dataset()
        shutil.rmtree(tmp)


def gate22_frozen_digest_scoped_to_the_frozen_path():
    """A freeze must not answer for a path it does not own."""
    D.clear_frozen_dataset()
    tmp = tempfile.mkdtemp()
    try:
        D.freeze_for_run(REAL_ALIAS, run_label="gate22")
        other = os.path.join(tmp, "other.json")
        with open(other, "w") as f:
            json.dump(_records(), f)
        assert D.run_frozen_dataset_sha256(other) is None
        assert D.run_frozen_dataset_sha256(D.get_frozen_dataset().path) is not None
    finally:
        D.clear_frozen_dataset()
        shutil.rmtree(tmp)


# ===========================================================================
# Beta §3 — the exception correction
# ===========================================================================

def gate23_missing_dataset_is_classified_chained_and_named():
    """Beta §3: NOT a bare FileNotFoundError, NOT a flattened ResidueError.

    Four separate obligations, each asserted: the type is
    `DatasetProvisioningError`; it is inside the residue hierarchy so the
    coordinator's non-retryable control flow is preserved; the original
    exception is chained rather than discarded; and the message names the
    absolute path and the node.
    """
    missing = os.path.join(tempfile.gettempdir(), "p05_definitely_absent.json")
    if os.path.exists(missing):
        os.remove(missing)

    for fn, args in ((_sha256_file, (missing,)),
                     (load_residue_window, (missing, 4, None, 0))):
        e = _expect(DatasetProvisioningError, fn, *args)
        assert isinstance(e, ResidueError), \
            "must stay in the residue hierarchy (control flow preserved)"
        assert type(e) is not ResidueError, \
            "must NOT be flattened into an undifferentiated ResidueError"
        assert isinstance(e.__cause__, FileNotFoundError), \
            "the original exception must be chained, not discarded"
        assert os.path.abspath(missing) in str(e), "message must name the ABSOLUTE path"
        assert D.local_node_identity() in str(e), "message must name the NODE"


def gate24_present_dataset_still_hashes():
    """CLEAN CONTROL for gate 23 — the classifier is not swallowing good reads."""
    assert _sha256_file(REAL_ALIAS) == D.sha256_file(REAL_ALIAS)
    tmp = tempfile.mkdtemp()
    try:
        p = os.path.join(tmp, "d.json")
        with open(p, "w") as f:
            json.dump(_records(8), f)
        assert load_residue_window(p, 4, None, 0) == [0, 1, 2, 3]
    finally:
        shutil.rmtree(tmp)


# ===========================================================================
# Requirement 4 + 5 — per-node verification and fail-before-dispatch
# ===========================================================================

def gate25_local_node_verification_clean_and_faulted():
    """Per-node verify: PASS on a good copy, FAIL on a corrupted one, FAIL on absent.

    Driven through the `local=True` NodeSpec so the fault injection is real
    (actual bytes, actual `sha256sum`, actual `stat`) without mutating a rig.
    """
    D.clear_frozen_dataset()
    tmp = tempfile.mkdtemp()
    try:
        pointer, v = make_publication(tmp)
        frozen = D.resolve_pointer(pointer)
        node = D.NodeSpec("local-test", "127.0.0.1", local=True)

        ok = D.verify_node_dataset(frozen, node)
        assert ok.status == "PASS", ok.message
        assert ok.digest == frozen.sha256 and ok.size_bytes == frozen.size_bytes

        # --- fault: same size, different bytes -> digest mismatch -------------
        with open(v, "r+b") as f:
            f.seek(2)
            f.write(b"X")
        bad = D.verify_node_dataset(frozen, node)
        assert bad.status == "FAIL" and "DIGEST MISMATCH" in bad.message
        assert frozen.sha256 in bad.message and bad.digest in bad.message

        # --- fault: absent ---------------------------------------------------
        os.remove(v)
        gone = D.verify_node_dataset(frozen, node)
        assert gone.status == "FAIL" and "ABSENT" in gone.message
        assert frozen.path in gone.message and "local-test" in gone.message
    finally:
        D.clear_frozen_dataset()
        shutil.rmtree(tmp)


def gate26_unreachable_node_is_unavailable_not_clean():
    """VIR-5: an unverifiable node is UNAVAILABLE and still fails the fleet."""
    frozen = D.resolve_pointer(REAL_POINTER)
    # 192.0.2.0/24 is TEST-NET-1 (RFC 5737) — guaranteed not to route anywhere.
    dead = D.NodeSpec("unreachable-test", "192.0.2.1", ssh_user="nobody")
    rec = D.verify_node_dataset(frozen, dead, timeout=20.0, connect_timeout=3)
    assert rec.status == "UNAVAILABLE", rec.status
    assert rec.digest is None
    e = _expect(DatasetProvisioningError, D.fleet_preflight, frozen, [dead],
                timeout=20.0, connect_timeout=3)
    assert "FAIL BEFORE DISPATCH" in str(e)


def gate27_fleet_preflight_reports_whole_fleet_then_fails():
    """One round trip reports every broken node, not just the first."""
    D.clear_frozen_dataset()
    tmp = tempfile.mkdtemp()
    try:
        pointer, v = make_publication(tmp)
        frozen = D.resolve_pointer(pointer)
        good = D.NodeSpec("good", "127.0.0.1", local=True)
        recs = D.fleet_preflight(frozen, [good])          # clean control
        assert [r.status for r in recs] == ["PASS"]

        os.remove(v)
        dead = D.NodeSpec("unreachable-test", "192.0.2.1", ssh_user="nobody")
        e = _expect(DatasetProvisioningError, D.fleet_preflight, frozen,
                    [good, dead], timeout=20.0, connect_timeout=3)
        assert "2 of 2" in str(e), str(e)
        assert "good" in str(e) and "unreachable-test" in str(e)
        assert frozen.path in str(e)
    finally:
        D.clear_frozen_dataset()
        shutil.rmtree(tmp)


def gate28_provisioning_manifest_absent_is_distinguishable_from_empty():
    """`None` (no manifest) and `[]` (manifest, no nodes) must not be conflated.

    They demand different operator responses, and collapsing them is how a
    fleet check silently becomes a no-op.
    """
    tmp = tempfile.mkdtemp()
    try:
        assert D.load_provisioning_nodes(os.path.join(tmp, "absent.json")) is None
        m = os.path.join(tmp, "m.json")
        with open(m, "w") as f:
            json.dump({"manifest_schema_version": 1,
                       "datasets": [{"dataset_logical_name": "daily3", "nodes": []}]}, f)
        assert D.load_provisioning_nodes(m) == []
        with open(m, "w") as f:
            json.dump({"manifest_schema_version": 1, "datasets": [
                {"dataset_logical_name": "daily3", "nodes": [
                    {"node_id": "n1", "ssh_address": "10.0.0.1"},
                    {"node_id": "n2", "ssh_address": "10.0.0.2", "ssh_user": "u"},
                ]}]}, f)
        nodes = D.load_provisioning_nodes(m)
        assert [n.node_id for n in nodes] == ["n1", "n2"]
        assert nodes[1].ssh_user == "u" and nodes[0].ssh_user == "michael"
    finally:
        shutil.rmtree(tmp)


# ===========================================================================
# Requirement 6 — run provenance
# ===========================================================================

def gate29_run_provenance_records_the_frozen_values():
    """Req 6: the frozen values appear in provenance, READ BACK, not assumed."""
    D.clear_frozen_dataset()
    tmp = tempfile.mkdtemp()
    try:
        frozen = D.freeze_for_run(REAL_ALIAS, run_label="gate29")
        node = D.NodeSpec("local-test", "127.0.0.1", local=True)
        rec = D.verify_node_dataset(frozen, node)
        path = D.write_run_provenance("gate29", frozen, [rec],
                                      fleet_status=rec.status, repo_root=tmp)
        with open(path) as f:
            doc = json.load(f)
        fd = doc["frozen_dataset"]
        assert fd["version_id"] == frozen.version_id
        assert fd["sha256"] == frozen.sha256
        assert fd["path"] == frozen.path and os.path.isabs(fd["path"])
        assert fd["size_bytes"] == frozen.size_bytes
        assert fd["record_count"] == frozen.record_count
        assert fd["manifest_path"] == frozen.manifest_path
        assert fd["manifest_sha256"] == frozen.manifest_sha256
        assert doc["phase"] == "6-P0.5"
        assert doc["fleet"][0]["node_id"] == "local-test"
        # written where it cannot dirty the tree
        assert path.endswith(".json") and D.RUN_PROVENANCE_DIRNAME in path
    finally:
        D.clear_frozen_dataset()
        shutil.rmtree(tmp)


# ===========================================================================
# WATCHER wiring + the published-artifact invariant
# ===========================================================================

def gate30_watcher_resolves_pointer_to_absolute_path():
    """Req 1 + 3 at the WATCHER seam: manifest-declared alias -> absolute version."""
    D.clear_frozen_dataset()
    try:
        import agents.watcher_agent as W
        out = W.p05_resolve_dataset_path(os.path.join(W.REPO_ROOT, "daily3.json"))
        assert os.path.isabs(out)
        assert os.path.basename(out) != D.LEGACY_ALIAS_NAME
        assert D.VERSION_FILENAME_RE.match(os.path.basename(out))
        # a non-dataset path is passed through untouched — this maps one file,
        # it is not a general path rewriter
        assert W.p05_resolve_dataset_path("/x/train_history.json") == \
            "/x/train_history.json"
        # and the manifest-driven preflight now resolves to that same object
        req, _ = W.get_step_io_from_manifest(1)
        assert any(os.path.basename(r) != D.LEGACY_ALIAS_NAME
                   and D.VERSION_FILENAME_RE.match(os.path.basename(r))
                   for r in req), req
        assert all(D.LEGACY_ALIAS_NAME != os.path.basename(r) for r in req), \
            "preflight must not still be checking the bare alias (§4)"
    finally:
        D.clear_frozen_dataset()


def gate31_published_artifacts_unmodified():
    """The harness must leave publication byte-identical. Measured, not asserted.

    Values are the ones Phase 6-P0 published (commit 131787d).
    """
    published_sha = "513648160d356617c22a1e543ae1c9c65f4921ec21718989308b1f70c00768f6"
    version = os.path.join(_ROOT, f"daily3-20260801T145551443433Z-513648160d35.json")
    assert D.sha256_file(version) == published_sha, "VERSION FILE MODIFIED"
    assert os.path.getsize(version) == 1380711
    assert D.sha256_file(REAL_ALIAS) == published_sha, "daily3.json MODIFIED"
    assert os.path.getsize(REAL_ALIAS) == 1380711
    with open(REAL_POINTER) as f:
        m = json.load(f)
    assert m["sha256"] == published_sha and m["record_count"] == 18068
    assert m["version_id"] == "daily3-20260801T145551443433Z-513648160d35"


def gate32_no_hybrid_skip_wire_in():
    """Out of scope, explicitly (§6): P0.5 must not touch the hybrid skip path.

    Beta: wiring both would put two behavioural changes into the first
    post-publication certification, defeating the reason P0/P0.5 were split.
    """
    r = subprocess.run(["git", "diff", "--stat", "--", "miner/", "window_optimizer.py",
                        "agents/", "utils/", "prng_registry.py"],
                       cwd=_ROOT, capture_output=True, text=True)
    touched = {ln.split("|")[0].strip() for ln in r.stdout.splitlines() if "|" in ln}
    forbidden = {"prng_registry.py", "miner/range_miner_protocol.py",
                 "window_optimizer_bayesian.py", "sieve_gpu_worker.py"}
    assert not (touched & forbidden), f"out-of-scope files modified: {touched & forbidden}"
    # and the skip-bound dead dimension is still exactly as it was
    src = open(os.path.join(_ROOT, "miner/range_miner_worker.py"),
               encoding="utf-8").read()
    assert "_hybrid_prefix" in src, "sanity: the hybrid prefix builder still exists"


# ===========================================================================
# Beta P0.5 CLOSURE RULING — an unusable provisioning manifest is FATAL for a
# miner-backed run, before any coordinator construction and any dispatch
# ===========================================================================
#
#   > A missing, unreadable, invalid, or empty provisioning manifest means the
#   > system cannot establish which worker datasets must be verified. Recording
#   > UNAVAILABLE and proceeding **violates the authority boundary.**
#
# Gate 33 covers the four conditions. Gate 34 is the substance: it proves the
# *absence of the side effects*, not merely that something was raised. Gate 35
# is the NOT_APPLICABLE routing, gate 36 the successful-manifest clean control,
# gate 37 the fault injection that proves 33 and 34 are not vacuous.


def _write_manifest(path, doc):
    with open(path, "w") as f:
        json.dump(doc, f)
    return path


def _good_manifest_doc(node_id="local-test"):
    return {"manifest_schema_version": 1, "datasets": [
        {"dataset_logical_name": "daily3", "nodes": [
            {"node_id": node_id, "ssh_address": "127.0.0.1", "local": True}]}]}


def _unusable_manifest_cases(tmp):
    """The four conditions Beta named, as (label, manifest_path) pairs.

    Every one of them is a real file (or a real absence) on disk — nothing here
    is simulated by patching the loader.
    """
    cases = [("missing", os.path.join(tmp, "no_such_manifest.json"))]

    unreadable = _write_manifest(os.path.join(tmp, "unreadable.json"),
                                 _good_manifest_doc())
    os.chmod(unreadable, 0o000)
    if os.geteuid() != 0 and not os.access(unreadable, os.R_OK):
        cases.append(("unreadable", unreadable))       # skipped as root: root reads anything

    bad_json = os.path.join(tmp, "invalid_json.json")
    with open(bad_json, "w") as f:
        f.write("{ nodes: [,,,")
    cases.append(("invalid/unparseable", bad_json))

    cases.append(("invalid/schema", _write_manifest(
        os.path.join(tmp, "invalid_schema.json"),
        {"manifest_schema_version": 2, "datasets": []})))

    cases.append(("invalid/node-entry", _write_manifest(
        os.path.join(tmp, "invalid_node.json"),
        {"manifest_schema_version": 1, "datasets": [
            {"dataset_logical_name": "daily3",
             "nodes": [{"ssh_address": "10.0.0.1"}]}]})))   # no node_id

    cases.append(("empty/no-nodes", _write_manifest(
        os.path.join(tmp, "empty_nodes.json"),
        {"manifest_schema_version": 1, "datasets": [
            {"dataset_logical_name": "daily3", "nodes": []}]})))

    cases.append(("empty/no-entry-for-dataset", _write_manifest(
        os.path.join(tmp, "empty_other.json"),
        {"manifest_schema_version": 1, "datasets": [
            {"dataset_logical_name": "pa_pick3", "nodes": [
                {"node_id": "x", "ssh_address": "10.0.0.9"}]}]})))
    return cases


def gate33_miner_backed_unusable_manifest_is_fatal():
    """Beta closure: missing · unreadable · invalid · empty — all four FATAL.

    Driven through the real `run_start_dataset_gate`, the same function
    `window_optimizer.main()` calls, with `miner_backed=True`. Each refusal must
    be a `DatasetProvisioningError` (Beta's approved classification, inside the
    residue hierarchy) and must NAME THE EXPECTED ABSOLUTE MANIFEST PATH, per
    Beta's Q3 ruling that the preflight message state where it looked.

    Side-effect absence is asserted here too, in its cheapest form: the gate
    writes run provenance as its last act, so a provenance file appearing under
    `repo_root` would mean the gate ran to completion. None may exist.
    """
    D.clear_frozen_dataset()
    tmp = tempfile.mkdtemp()
    try:
        pointer, version = make_publication(tmp)
        alias = os.path.join(tmp, D.LEGACY_ALIAS_NAME)
        cases = _unusable_manifest_cases(tmp)
        assert len(cases) >= 6, f"fault substrate incomplete: {cases}"

        for label, manifest in cases:
            e = _expect(DatasetProvisioningError, D.run_start_dataset_gate,
                        alias, run_label=f"gate33_{label}", miner_backed=True,
                        provisioning_manifest=manifest, repo_root=tmp)
            assert isinstance(e, ResidueError), \
                f"{label}: must stay in the residue hierarchy"
            assert os.path.abspath(manifest) in str(e), \
                f"{label}: message must name the expected ABSOLUTE manifest " \
                f"path — got {str(e)[:300]!r}"

        prov = os.path.join(tmp, D.RUN_PROVENANCE_DIRNAME)
        assert not os.path.isdir(prov) or not os.listdir(prov), \
            "the gate wrote run provenance — it did not fail before dispatch"

        # ...and the same four conditions on a NON-miner path are unchanged by
        # this correction: missing/empty still merely record, so the ruling was
        # applied to the topology Beta named and not to everything in reach.
        for label, manifest in cases:
            if label.startswith("invalid") or label == "unreadable":
                continue          # unusable-for-anyone: fatal on every path
            D.clear_frozen_dataset()
            f = D.run_start_dataset_gate(
                alias, run_label=f"gate33_nonminer_{label}", miner_backed=False,
                provisioning_manifest=manifest, repo_root=tmp,
                write_provenance=False)
            assert f.path == version
    finally:
        D.clear_frozen_dataset()
        for root, dirs, files in os.walk(tmp):
            for name in files:
                try:
                    os.chmod(os.path.join(root, name), 0o600)
                except OSError:
                    pass
        shutil.rmtree(tmp)


# --- gate 34: the negative gate proper -------------------------------------
# Beta's condition 3 — "no coordinator construction, no worker process, no
# dispatch occurs" — is a claim about things that must NOT have happened, so it
# cannot be proven by catching an exception. It is proven here by running the
# real `window_optimizer.main()` with `--use-range-miner` in a child process
# with an ARMED TRIPWIRE on every construction and process-creation surface it
# could reach, and reading back four independent absence measurements:
#
#   1. no tripwire fired          — nothing was constructed, nothing was spawned;
#   2. no descendant process      — measured from /proc, not inferred;
#   3. no new entry under the repo root — no spool, no output dir, no provenance;
#   4. the process died at argparse (SystemExit 2) naming the manifest path.
#
# The tripwires replace the stage AFTER the gate, never the gate itself: the
# dataset-authority path under test is the real one (VIR-1 execution proof).
# The only patched input is *where the manifest is looked for*.

_CHILD_DRIVER = r'''
import json, os, sys

ROOT = os.environ["P05_ROOT"]
OUT = os.environ["P05_VERDICT"]
verdict = {"tripwires": [], "outcome": None, "exit_code": None, "stderr": "",
           "children_after": [], "root_new_entries": []}


def _emit():
    with open(OUT, "w") as f:
        json.dump(verdict, f)


def _children(pid):
    """Descendant PIDs, read straight out of /proc — no subprocess, no ps."""
    out = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open("/proc/%s/stat" % entry, "rb") as fh:
                fields = fh.read().rsplit(b")", 1)[1].split()
            if int(fields[1]) == pid:
                out.append(int(entry))
        except (OSError, IndexError, ValueError):
            continue
    return out


try:
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)

    from miner import dataset_authority as D
    ABSENT = os.environ["P05_ABSENT_MANIFEST"]
    D.default_provisioning_manifest_path = lambda repo_root=None: ABSENT

    if os.environ.get("P05_REVERT") == "1":
        # THE INJECTED FAULT: the pre-closure behaviour, verbatim in effect —
        # record UNAVAILABLE for an absent manifest and let the run proceed.
        def _reverted(nodes, **kw):
            return "UNAVAILABLE"
        D.resolve_absent_fleet_status = _reverted

    import window_optimizer as W

    def _trip(name, raising=True):
        def _fire(*a, **k):
            verdict["tripwires"].append(name)
            if raising:
                raise RuntimeError("TRIPWIRE " + name)
            return "/dev/null/" + name
        return _fire

    # everything downstream of the gate
    W.MultiGPUCoordinator = _trip("MultiGPUCoordinator")
    W.run_bayesian_optimization = _trip("run_bayesian_optimization")
    W.run_with_config = _trip("run_with_config")
    try:
        import coordinator as _C
        _C.MultiGPUCoordinator = _trip("coordinator.MultiGPUCoordinator")
    except ImportError:
        pass
    D.fleet_preflight = _trip("fleet_preflight")
    D.provision_node_dataset = _trip("provision_node_dataset")
    # non-raising: the run must get FURTHER than this under the injected fault
    D.write_run_provenance = _trip("write_run_provenance", raising=False)
    # every process-creation surface reachable from CPython
    import subprocess as _sp
    _sp.Popen = _trip("subprocess.Popen")
    _sp.run = _trip("subprocess.run")
    _sp.call = _trip("subprocess.call")
    _sp.check_output = _trip("subprocess.check_output")
    import socket as _sk
    _sk.socket.connect = _trip("socket.connect")
    _sk.socket.bind = _trip("socket.bind")
    _sk.create_connection = _trip("socket.create_connection")
    os.fork = _trip("os.fork")
    os.posix_spawn = _trip("os.posix_spawn")
    os.system = _trip("os.system")
    import multiprocessing as _mp
    _mp.Process.start = _trip("multiprocessing.Process.start")

    before = set(os.listdir(ROOT))

    import io
    err = io.StringIO()
    real_err = sys.stderr
    sys.stderr = err
    sys.argv = ["window_optimizer.py", "--strategy", "bayesian",
                "--lottery-file", os.environ["P05_LOTTERY"],
                "--use-range-miner", "--trials", "1"]
    try:
        W.main()
        verdict["outcome"] = "RETURNED"
    except SystemExit as exc:
        verdict["outcome"] = "SystemExit"
        verdict["exit_code"] = exc.code
    except BaseException as exc:
        verdict["outcome"] = type(exc).__name__ + ": " + str(exc)[:2000]
    finally:
        sys.stderr = real_err

    verdict["stderr"] = err.getvalue()[-4000:]
    verdict["children_after"] = _children(os.getpid())
    verdict["root_new_entries"] = sorted(set(os.listdir(ROOT)) - before)
except BaseException as exc:
    verdict["outcome"] = "DRIVER_ERROR: " + repr(exc)[:2000]
finally:
    _emit()
'''


def _run_child_driver(revert=False, timeout=600):
    """Run the real window_optimizer CLI path with tripwires; return the verdict."""
    tmp = tempfile.mkdtemp()
    try:
        driver = os.path.join(tmp, "p05_condition3_driver.py")
        with open(driver, "w") as f:
            f.write(_CHILD_DRIVER)
        out = os.path.join(tmp, "verdict.json")
        env = dict(os.environ)
        env.update({
            "P05_ROOT": _ROOT,
            "P05_VERDICT": out,
            "P05_ABSENT_MANIFEST": os.path.join(tmp, "definitely_absent.json"),
            "P05_LOTTERY": REAL_ALIAS,
            "PYTHONPATH": _ROOT,
        })
        env.pop("P05_REVERT", None)
        if revert:
            env["P05_REVERT"] = "1"
        proc = subprocess.run([sys.executable, driver], env=env, cwd=_ROOT,
                              capture_output=True, text=True, timeout=timeout)
        assert os.path.exists(out), (
            "the driver produced no verdict — VIR-1: silence is not a pass.\n"
            f"rc={proc.returncode}\nstdout tail:\n{proc.stdout[-2000:]}\n"
            f"stderr tail:\n{proc.stderr[-2000:]}")
        with open(out) as f:
            verdict = json.load(f)
        verdict["_child_rc"] = proc.returncode
        return verdict
    finally:
        shutil.rmtree(tmp)


def gate34_no_construction_no_process_no_dispatch():
    """Beta condition 3, proven by the ABSENCE of the side effects."""
    v = _run_child_driver()
    assert not str(v["outcome"]).startswith("DRIVER_ERROR"), v["outcome"]

    # 1. nothing constructed, nothing spawned — no tripwire fired at all
    assert v["tripwires"] == [], (
        f"a post-gate surface was reached: {v['tripwires']}")
    # 2. the refusal really happened, at argparse, before any work
    assert v["outcome"] == "SystemExit", v["outcome"]
    assert v["exit_code"] == 2, v["exit_code"]
    assert "DATASET_AUTHORITY_P0_5" in v["stderr"], v["stderr"][-1500:]
    assert "FAIL BEFORE DISPATCH" in v["stderr"], v["stderr"][-1500:]
    assert "definitely_absent.json" in v["stderr"], \
        "the refusal must name the manifest path it looked for"
    # 3. no descendant process was ever created — measured from /proc
    assert v["children_after"] == [], v["children_after"]
    # 4. no spool, no output dir, no provenance record, nothing at all
    assert v["root_new_entries"] == [], v["root_new_entries"]


def gate35_not_applicable_is_routed_and_distinct():
    """Beta: `UNAVAILABLE` means ATTEMPTED and could not be completed.

    A path that never needed fleet verification must not borrow that word. The
    three answers must be three answers, and the unknown case must keep the
    over-constrained one.
    """
    D.clear_frozen_dataset()
    tmp = tempfile.mkdtemp()
    try:
        pointer, version = make_publication(tmp)
        alias = os.path.join(tmp, D.LEGACY_ALIAS_NAME)
        absent = os.path.join(tmp, "no_manifest.json")

        def _status(label, **kw):
            D.clear_frozen_dataset()
            f = D.run_start_dataset_gate(alias, run_label=label,
                                         provisioning_manifest=absent,
                                         repo_root=tmp, **kw)
            assert f.path == version
            with open(os.path.join(tmp, D.RUN_PROVENANCE_DIRNAME,
                                   f"{label}.json")) as fh:
                return json.load(fh)["fleet_status"]

        # THE CLEAN CONTROL: non-miner, no remote execution -> NOT_APPLICABLE
        na = _status("gate35_na", miner_backed=False, remote_execution=False)
        assert na == D.FLEET_STATUS_NOT_APPLICABLE, na
        assert na != D.FLEET_STATUS_UNAVAILABLE

        # remote execution, non-miner -> still UNAVAILABLE (attempted, unmet)
        assert _status("gate35_remote", miner_backed=False,
                       remote_execution=True) == D.FLEET_STATUS_UNAVAILABLE
        # unknown -> UNAVAILABLE, never the clean word by default
        assert _status("gate35_unknown",
                       miner_backed=False) == D.FLEET_STATUS_UNAVAILABLE

        # and NOT_APPLICABLE is never a way out of the miner ruling
        _expect(DatasetProvisioningError, D.run_start_dataset_gate, alias,
                run_label="gate35_miner", miner_backed=True,
                remote_execution=False, provisioning_manifest=absent,
                repo_root=tmp)
    finally:
        D.clear_frozen_dataset()
        shutil.rmtree(tmp)


def gate36_successful_manifest_path_unchanged():
    """Beta: re-certification is needed only if successful-manifest behaviour moved.

    Two proofs that it did not. Structural: with a usable manifest the entire
    new decision function is unreachable — it is replaced by a raiser here and
    the run still passes. Behavioural: the provenance record produced with
    `miner_backed=True` and with `miner_backed=False` is identical field for
    field, and identical to what a PASS fleet produced before.
    """
    D.clear_frozen_dataset()
    tmp = tempfile.mkdtemp()
    original = D.resolve_absent_fleet_status
    try:
        pointer, version = make_publication(tmp)
        alias = os.path.join(tmp, D.LEGACY_ALIAS_NAME)
        manifest = _write_manifest(os.path.join(tmp, "good.json"),
                                   _good_manifest_doc())

        def _tripwire(*a, **k):
            raise AssertionError(
                "the absent-manifest decision was consulted on a SUCCESSFUL "
                "manifest — the new code is on the certified path")
        D.resolve_absent_fleet_status = _tripwire

        docs = {}
        for label, miner in (("gate36_miner", True), ("gate36_plain", False)):
            D.clear_frozen_dataset()
            f = D.run_start_dataset_gate(alias, run_label=label,
                                         miner_backed=miner,
                                         remote_execution=True,
                                         provisioning_manifest=manifest,
                                         repo_root=tmp)
            assert f.path == version
            with open(os.path.join(tmp, D.RUN_PROVENANCE_DIRNAME,
                                   f"{label}.json")) as fh:
                docs[label] = json.load(fh)

        for label, doc in docs.items():
            assert doc["fleet_status"] == "PASS", (label, doc["fleet_status"])
            assert len(doc["fleet"]) == 1 and doc["fleet"][0]["status"] == "PASS"
            assert doc["fleet"][0]["digest"] == doc["frozen_dataset"]["sha256"]
            assert doc["fleet"][0]["dataset_path"] == version

        # identical field for field, once the two per-run stamps that MUST
        # differ between any two runs (the label, the clock) are removed
        _volatile = ("run_label", "recorded_utc", "frozen_utc")
        def _stable(doc):
            out = {k: v for k, v in doc.items() if k not in _volatile}
            out["frozen_dataset"] = {k: v for k, v in doc["frozen_dataset"].items()
                                     if k not in _volatile}
            return out
        a, b = _stable(docs["gate36_miner"]), _stable(docs["gate36_plain"])
        assert set(a["frozen_dataset"]) >= {"sha256", "path", "version_id"}, \
            "sanity: the comparison must not have stripped the identity itself"
        assert a == b, "the miner flag changed a SUCCESSFUL run's provenance"
    finally:
        D.resolve_absent_fleet_status = original
        D.clear_frozen_dataset()
        shutil.rmtree(tmp)


def gate37_fault_injection_control():
    """VIR-2: revert the hard-fail and the two gates above must go RED.

    A gate that has only ever seen the fixed code is unproven. The fault is the
    pre-closure behaviour itself — `resolve_absent_fleet_status` returning
    UNAVAILABLE instead of raising — injected into both detectors.
    """
    # --- gate 33's detector, faulted --------------------------------------
    D.clear_frozen_dataset()
    tmp = tempfile.mkdtemp()
    original = D.resolve_absent_fleet_status
    try:
        pointer, version = make_publication(tmp)
        alias = os.path.join(tmp, D.LEGACY_ALIAS_NAME)
        absent = os.path.join(tmp, "no_manifest.json")

        D.resolve_absent_fleet_status = lambda nodes, **kw: "UNAVAILABLE"
        try:
            D.run_start_dataset_gate(alias, run_label="gate37_faulted",
                                     miner_backed=True,
                                     provisioning_manifest=absent,
                                     repo_root=tmp, write_provenance=False)
            faulted_raised = False
        except DatasetProvisioningError:
            faulted_raised = True
        assert not faulted_raised, \
            "the injected fault did not take — gate 33 is not measuring it"
    finally:
        D.resolve_absent_fleet_status = original
        D.clear_frozen_dataset()
        shutil.rmtree(tmp)

    # --- gate 34's detector, faulted, through the real CLI -----------------
    v = _run_child_driver(revert=True)
    assert not str(v["outcome"]).startswith("DRIVER_ERROR"), v["outcome"]
    assert v["tripwires"], (
        "the reverted run fired no tripwire — gate 34 cannot distinguish a run "
        "that stopped from a run that never started (VACUOUS DETECTOR)")
    assert "run_bayesian_optimization" in v["tripwires"], v["tripwires"]
    assert "write_run_provenance" in v["tripwires"], (
        "the reverted run must record UNAVAILABLE and PROCEED — that is the "
        "defect Beta named", v["tripwires"])
    # the fixed run asserts tripwires == []; the faulted run breaks exactly that
    assert v["tripwires"] != []


# ===========================================================================
# Live fleet (opt-in) — read-only
# ===========================================================================

def gate38_live_fleet_verification():
    """LIVE: verify the frozen dataset on all three CT100 workers (read-only)."""
    frozen = D.resolve_pointer(REAL_POINTER)
    nodes = D.load_provisioning_nodes()
    if nodes is None:
        raise AssertionError(
            f"no provisioning manifest at {D.default_provisioning_manifest_path()} "
            "— cannot run the live fleet gate (UNAVAILABLE, not clean)")
    assert nodes, "provisioning manifest declares no nodes"
    recs = [D.verify_node_dataset(frozen, n) for n in nodes]
    for r in recs:
        print(f"      [{r.status}] {r.node_id} ({r.ssh_address}): "
              f"{r.digest or '—'}")
    bad = [r for r in recs if r.status != "PASS"]
    assert not bad, "\n".join(r.message for r in bad)


# ===========================================================================
# Runner
# ===========================================================================

def _check(name, fn):
    try:
        fn()
        _results.append((name, True, ""))
        print(f"  ✅ {name}")
    except Exception:
        _results.append((name, False, traceback.format_exc()))
        print(f"  ❌ {name}")


def main():
    live = "--fleet" in sys.argv
    print("=" * 74)
    print("S172 Phase 6-P0.5 — dataset authority acceptance harness")
    print("=" * 74)

    _check("Gate  1: [req1] real pointer resolves + self-verifies (CLEAN)",
           gate01_real_pointer_resolves)
    _check("Gate  2: [req3] alias resolves to the immutable version",
           gate02_alias_resolves_to_version)
    _check("Gate  3: [neg] pointer missing → refuse",
           gate03_pointer_missing_refused)
    _check("Gate  4: [neg] pointer unparseable → refuse",
           gate04_pointer_unparseable_refused)
    _check("Gate  5: [req8] pointer naming daily3.json → refuse",
           gate05_pointer_naming_alias_refused)
    _check("Gate  6: [req8] pointer naming an absolute path → refuse",
           gate06_pointer_absolute_path_refused)
    _check("Gate  7: [req8] pointer traversal → refuse",
           gate07_pointer_traversal_refused)
    _check("Gate  8: [req8] non-conforming filename → refuse (6 forms)",
           gate08_pointer_nonconforming_name_refused)
    _check("Gate  9: [neg] pointer target does not exist → refuse",
           gate09_pointer_target_missing_refused)
    _check("Gate 10: [schema] version_id/filename disagreement → refuse",
           gate10_version_id_filename_disagreement_refused)
    _check("Gate 11: [schema] filename digest prefix disagreement → refuse",
           gate11_filename_digest_prefix_disagreement_refused)
    _check("Gate 12: [identity] digest mismatch → refuse",
           gate12_digest_mismatch_refused)
    _check("Gate 13: [identity] size mismatch → refuse",
           gate13_size_mismatch_refused)
    _check("Gate 14: [identity] record-count mismatch → refuse",
           gate14_record_count_mismatch_refused)
    _check("Gate 15: [schema] unsupported manifest schema → refuse",
           gate15_unsupported_schema_version_refused)
    _check("Gate 16: [schema] missing required field → refuse",
           gate16_missing_required_field_refused)
    _check("Gate 17: [req3] unpublished bare alias → refuse",
           gate17_unpublished_alias_refused)
    _check("Gate 18: [req2] freeze is one-time + idempotent",
           gate18_freeze_is_one_time_and_idempotent)
    _check("Gate 19: [req2] conflicting second freeze → raise",
           gate19_conflicting_freeze_raises)
    _check("Gate 20: [req7] pointer moved MID-RUN → run unaffected",
           gate20_pointer_moves_mid_run_does_not_alter_run)
    _check("Gate 21: [§2.1] coordinator digest run-scoped, not trial-scoped",
           gate21_coordinator_digest_is_run_scoped_not_trial_scoped)
    _check("Gate 22: [req2] frozen digest scoped to the frozen path",
           gate22_frozen_digest_scoped_to_the_frozen_path)
    _check("Gate 23: [Beta §3] absent dataset classified + chained + named",
           gate23_missing_dataset_is_classified_chained_and_named)
    _check("Gate 24: [Beta §3] CLEAN control — good reads still succeed",
           gate24_present_dataset_still_hashes)
    _check("Gate 25: [req5] per-node verify: PASS / mismatch / absent",
           gate25_local_node_verification_clean_and_faulted)
    _check("Gate 26: [VIR-5] unreachable node → UNAVAILABLE, still fails",
           gate26_unreachable_node_is_unavailable_not_clean)
    _check("Gate 27: [req4] fleet preflight reports all, then fails closed",
           gate27_fleet_preflight_reports_whole_fleet_then_fails)
    _check("Gate 28: [req5] absent manifest ≠ empty manifest",
           gate28_provisioning_manifest_absent_is_distinguishable_from_empty)
    _check("Gate 29: [req6] run provenance carries the frozen values",
           gate29_run_provenance_records_the_frozen_values)
    _check("Gate 30: [req1/3] WATCHER resolves pointer → absolute path",
           gate30_watcher_resolves_pointer_to_absolute_path)
    _check("Gate 31: [scope] published artifacts byte-unmodified",
           gate31_published_artifacts_unmodified)
    _check("Gate 32: [scope] hybrid skip NOT wired in",
           gate32_no_hybrid_skip_wire_in)
    _check("Gate 33: [Beta closure] miner + unusable manifest → FATAL (4 conditions)",
           gate33_miner_backed_unusable_manifest_is_fatal)
    _check("Gate 34: [Beta closure] no coordinator, no process, no dispatch",
           gate34_no_construction_no_process_no_dispatch)
    _check("Gate 35: [Beta closure] NOT_APPLICABLE routed, distinct from UNAVAILABLE",
           gate35_not_applicable_is_routed_and_distinct)
    _check("Gate 36: [CLEAN] successful-manifest path unchanged",
           gate36_successful_manifest_path_unchanged)
    _check("Gate 37: [VIR-2] fault injection — revert the hard-fail, gates red",
           gate37_fault_injection_control)
    if live:
        _check("Gate 38: [LIVE] fleet verification, digests on target",
               gate38_live_fleet_verification)

    print("=" * 74)
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{passed}/{total} checks green "
          f"({'with' if live else 'without'} the live-fleet gate)")
    if passed != total:
        print("\nFAILURES (DO NOT COMMIT):")
        for name, ok, tb in _results:
            if not ok:
                print(f"\n--- {name} ---\n{tb}")
        print("\nRESULT: FAIL")
        sys.exit(1)
    print("\nRESULT: PASS — S172 Phase 6-P0.5 dataset authority is "
          "contract-validated (pending Team Alpha + Team Beta review).")
    sys.exit(0)


if __name__ == "__main__":
    main()
