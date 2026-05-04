#!/usr/bin/env python3
"""
S170-LIVE-GPU-PROBE-v2

Fixes TCP-PWC static gpu_count bug:
- Coordinator probes live ROCm/CuPy GPU count before launching workers.
- If probe fails: fail closed, skip that ROCm node for this run.
- Worker validates isolated Device(0) before sending READY.
"""

from pathlib import Path
import ast
import sys

PWC = Path("persistent_worker_coordinator.py")
WORKER = Path("persistent/pwc_worker_service.py")
MARKER = "S170-LIVE-GPU-PROBE"

def die(msg, code=1):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)

def replace_once(src, old, new, label):
    n = src.count(old)
    if n != 1:
        die(f"{label}: anchor count={n}, expected 1", 10)
    return src.replace(old, new, 1)

def patch_pwc():
    src = PWC.read_text()

    if MARKER in src:
        print("[S170-LIVE-GPU-PROBE-v2] persistent_worker_coordinator.py already patched")
        return

    anchor = '''    def _is_rocm(self, node: WorkerNode) -> bool:
        gt = (node.gpu_type or "").lower()
        return ("rx" in gt) or ("amd" in gt) or ("rocm" in gt)

'''

    method = '''    def _probe_live_rocm_gpu_count(self, node: WorkerNode) -> Optional[int]:
        """
        [S170-LIVE-GPU-PROBE]
        Return live CuPy-visible ROCm GPU count for this rig.

        Fail-closed policy:
          - valid integer >= 0: use it
          - probe error/timeout: return 0, skip this node for this run

        Rationale: distributed_config.json is desired topology, not live truth.
        """
        if self._is_localhost(node.hostname) or not self._is_rocm(node):
            return node.gpu_count

        import subprocess as _probe_sp

        host = node.hostname
        user = node.username
        activate_path = node.python_env.replace("/bin/python", "/bin/activate")

        probe_py = "import cupy as cp\\nprint(cp.cuda.runtime.getDeviceCount())\\n"

        cmd = (
            "source " + activate_path + " && "
            "cd " + node.script_path + " && "
            "export HSA_OVERRIDE_GFX_VERSION=10.3.0 && "
            "export HSA_ENABLE_SDMA=0 && "
            "export HSA_ENABLE_DEBUG_TRAP=0 && "
            "export ROCM_PATH=/opt/rocm && "
            "export HIP_PATH=/opt/rocm/hip && "
            "export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/hip/lib:${LD_LIBRARY_PATH} && "
            "export PATH=/opt/rocm/bin:${PATH} && "
            "unset ROCR_VISIBLE_DEVICES && unset HIP_VISIBLE_DEVICES && unset CUDA_VISIBLE_DEVICES && "
            "python3 - <<'PY'\\n" + probe_py + "PY"
        )

        ssh_cmd = [
            "ssh", "-q",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=10",
            (user + "@" + host) if user else host,
            cmd,
        ]

        try:
            r = _probe_sp.run(ssh_cmd, capture_output=True, text=True, timeout=45)
            if r.returncode != 0:
                self.logger.error(
                    f"[S170-LIVE-GPU-PROBE] {host}: probe failed rc={r.returncode}; "
                    f"FAIL-CLOSED skipping node. stderr={r.stderr.strip()[:300]}"
                )
                return 0

            lines = [x.strip() for x in r.stdout.splitlines() if x.strip()]
            live = int(lines[-1])
            configured = int(node.gpu_count)

            if live != configured:
                self.logger.warning(
                    f"[S170-LIVE-GPU-PROBE] {host}: live ROCm GPU count "
                    f"{live}/{configured}; launching only live GPUs this run"
                )
            else:
                self.logger.info(
                    f"[S170-LIVE-GPU-PROBE] {host}: live ROCm GPU count OK {live}/{configured}"
                )

            return max(0, live)

        except Exception as e:
            self.logger.error(
                f"[S170-LIVE-GPU-PROBE] {host}: probe exception {e}; "
                f"FAIL-CLOSED skipping node"
            )
            return 0

'''

    src = replace_once(src, anchor, anchor + method, "insert live probe method")

    old = '''            pool = min(self.worker_pool_size, node.gpu_count)
            ssh_base = ["ssh", "-q",
'''

    new = '''            # [S170-LIVE-GPU-PROBE] Use live ROCm enumeration, not static config only.
            _live_gpu_count = self._probe_live_rocm_gpu_count(node)
            node.gpu_count = min(int(node.gpu_count), int(_live_gpu_count))
            pool = min(self.worker_pool_size, node.gpu_count)

            if pool <= 0:
                self.logger.error(
                    "[S170-LIVE-GPU-PROBE] " + host +
                    ": no live ROCm GPUs available; skipping node for this run"
                )
                continue

            ssh_base = ["ssh", "-q",
'''

    src = replace_once(src, old, new, "replace TCP pool sizing")

    ast.parse(src)
    PWC.write_text(src)
    print("[S170-LIVE-GPU-PROBE-v2] patched persistent_worker_coordinator.py")

def patch_worker():
    src = WORKER.read_text()

    if MARKER in src:
        print("[S170-LIVE-GPU-PROBE-v2] persistent/pwc_worker_service.py already patched")
        return

    old = '''                self._wait_for_init()
                self._import_sieve()
                self._send_ready()
'''

    new = '''                self._wait_for_init()
                self._import_sieve()
                self._validate_compute_device()  # [S170-LIVE-GPU-PROBE] fail before READY if isolated GPU invalid
                self._send_ready()
'''

    src = replace_once(src, old, new, "insert validation before READY")

    anchor = '''    def _send_ready(self) -> None:
        """S161 v2: notify coordinator we are compute-ready."""
'''

    method = '''    def _validate_compute_device(self) -> None:
        """
        [S170-LIVE-GPU-PROBE]
        Validate assigned isolated compute device before READY.

        With ROCR_VISIBLE_DEVICES=<physical_gpu>, CuPy should see exactly the
        isolated logical Device(0). If not, exit before the coordinator can use us.
        """
        try:
            import cupy as cp
            n = cp.cuda.runtime.getDeviceCount()
            if n < 1:
                raise RuntimeError(f"CuPy sees {n} devices after isolation")
            with cp.cuda.Device(0):
                x = cp.zeros((1,), dtype=cp.uint8)
                cp.cuda.runtime.deviceSynchronize()
                del x
            log.info(
                f"[{self.worker_id}] [S170-LIVE-GPU-PROBE] device validation OK "
                f"visible_devices={n} logical_device=0 rocm={self.use_rocm}"
            )
        except Exception as exc:
            log.error(
                f"[{self.worker_id}] [S170-LIVE-GPU-PROBE] device validation FAILED: {exc}"
            )
            self._emit_heartbeat(
                "exception",
                error=f"S170 device validation failed: {exc}",
            )
            sys.exit(42)

'''

    src = replace_once(src, anchor, method + anchor, "insert validation method")

    ast.parse(src)
    WORKER.write_text(src)
    print("[S170-LIVE-GPU-PROBE-v2] patched persistent/pwc_worker_service.py")

def main():
    if not PWC.exists():
        die(f"missing {PWC}")
    if not WORKER.exists():
        die(f"missing {WORKER}")

    patch_pwc()
    patch_worker()

    for path in (PWC, WORKER):
        ast.parse(path.read_text())
        print(f"[S170-LIVE-GPU-PROBE-v2] AST OK: {path}")

    print("[S170-LIVE-GPU-PROBE-v2] DONE")

if __name__ == "__main__":
    main()
