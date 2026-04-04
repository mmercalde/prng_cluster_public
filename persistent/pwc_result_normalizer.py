"""
persistent/pwc_result_normalizer.py
=====================================
PWC Transport Adapter v1 — Result normalization shim.
TB-approved S159G proposal. Team Alpha implementation.

This is the contract wall.
Step 3 and all downstream steps must never see transport-native wire objects.
They must see only normalized legacy-compatible result objects.

TB ruling: "Internal wire format may evolve. External Step 1 result contract may not."
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict


def normalize_transport_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert TCP transport result messages into the existing PWC/coordinator
    result contract expected by current pipeline code.

    Supports:
      - result_inline: payload embedded in message
      - result_spooled: payload in spool file on disk
    """
    mtype = raw.get("message_type")

    if mtype == "result_inline":
        result = raw.get("result") or {}
    elif mtype == "result_spooled":
        result = _load_spooled_result(raw)
    else:
        raise ValueError(f"unsupported message_type for normalization: {mtype!r}")

    # Normalize to the legacy PWC result contract.
    # This is what PersistentWorkerCoordinator._dispatch_job() currently returns.
    normalized = {
        # Identity
        "job_id":       raw.get("job_id"),
        "lease_id":     raw.get("lease_id"),
        "attempt":      raw.get("attempt", 0),
        "worker_id":    raw.get("worker_id"),
        # Worker identity (legacy fields)
        "hostname":     result.get("hostname"),
        "gpu_id":       result.get("gpu_id", -1),
        # Result
        "status":       "ok" if result.get("success", False) else "error",
        "result":       result.get("payload", {}),
        "error":        result.get("error"),
        # Format hint for downstream slim_v1 fast path
        "result_format": result.get("result_format", "legacy_json"),
    }

    # Flatten slim_v1 parallel arrays if present — matches existing PWC behavior
    payload = normalized["result"]
    if isinstance(payload, dict) and payload.get("format") == "slim_v1":
        normalized["result"] = payload  # pass through, coordinator handles slim_v1

    return normalized


def _load_spooled_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load a spooled result from disk and validate checksum.
    Spooled results are used when payload exceeds inline_max_bytes.
    """
    spool_path = raw.get("spool_path", "")
    expected_sha256 = raw.get("sha256", "")

    if not spool_path or not os.path.exists(spool_path):
        return {
            "hostname": raw.get("worker_id", "").split(":")[0],
            "gpu_id": -1,
            "success": False,
            "error": f"spool file not found: {spool_path!r}",
            "payload": {},
            "result_format": "legacy_json",
        }

    try:
        # Validate checksum if provided
        if expected_sha256:
            import hashlib
            with open(spool_path, "rb") as f:
                actual = hashlib.sha256(f.read()).hexdigest()
            if actual != expected_sha256:
                return {
                    "hostname": raw.get("worker_id", "").split(":")[0],
                    "gpu_id": -1,
                    "success": False,
                    "error": f"spool checksum mismatch: expected {expected_sha256}, got {actual}",
                    "payload": {},
                    "result_format": "legacy_json",
                }

        with open(spool_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        # Clean up spool file after successful load
        try:
            os.unlink(spool_path)
        except Exception:
            pass

        return {
            "hostname": payload.get("hostname", raw.get("worker_id", "").split(":")[0]),
            "gpu_id":   payload.get("gpu_id", -1),
            "success":  True,
            "payload":  payload.get("result", payload),
            "result_format": payload.get("result_format", "legacy_json"),
        }

    except Exception as exc:
        return {
            "hostname": raw.get("worker_id", "").split(":")[0],
            "gpu_id": -1,
            "success": False,
            "error": f"spool load failed: {exc}",
            "payload": {},
            "result_format": "legacy_json",
        }
