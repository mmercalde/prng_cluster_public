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
# Live fleet (opt-in) — read-only
# ===========================================================================

def gate33_live_fleet_verification():
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
    if live:
        _check("Gate 33: [LIVE] fleet verification, digests on target",
               gate33_live_fleet_verification)

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
