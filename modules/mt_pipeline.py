
import os, json, time

def utc_now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _to_jsonable(obj):
    try:
        import numpy as _np
    except Exception:
        _np = None
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if _np is not None:
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, _np.ndarray):
            return [_to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, (int, float, str)) or obj is None or isinstance(obj, bool):
        return obj
    return str(obj)

def load_uint32_sequence(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sequence file not found: {path}")
    raw = open(path, "r", encoding="utf-8").read().strip()
    outputs = None
    meta = {"source": path}
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            outputs = [int(x) for x in data]
            meta["format"] = "json_list"
    except Exception:
        pass
    if outputs is None:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        try:
            outputs = [int(ln) for ln in lines]
            meta["format"] = "lines"
        except Exception as e:
            raise ValueError(f"Unrecognized sequence file format: {e}")
    cleaned = []
    for x in outputs:
        xi = int(x)
        if not (0 <= xi <= 0xFFFFFFFF):
            raise ValueError(f"Value out of 32-bit range: {x}")
        cleaned.append(xi)
    meta["count"] = len(cleaned)
    return cleaned, meta

def save_reconstruction_result(result: dict, out_path: str | None = None) -> str:
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    results_dir = os.path.abspath(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    if out_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        out_path = os.path.join(results_dir, f"mt_state_recon_{ts}.json")
    result.setdefault("job_type", "state_reconstruction")
    ver = result.get("verification") or {}
    mr = ver.get("match_rate")
    if isinstance(mr, (int, float)):
        result.setdefault("composite_score", float(mr))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(result), f, indent=2)
    return os.path.abspath(out_path)
