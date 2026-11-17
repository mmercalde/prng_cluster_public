#!/usr/bin/env python3
"""CPU-only gates for scripts/rig_fault_map.py (Beta X-GCVM-LOCATED R1 Q1/Q6). No host, no GPU.
G-OWNER  an event from a secondary thread is attributed to its process via the explicit VM pid field.
G-DST    the {..} continuation line after vm_update_ptes is attached as physical addresses.
G-TGID   the netconsole parser keeps the process pid (TGID) and the thread pid separately.
"""
import os, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import rig_fault_map as m  # noqa: E402

TRACE = ("1\t          python-100  [001] .....  1.000000: amdgpu_vm_update_ptes: pid:100 vm_ctx:0x1 start:0x0000000010 end:0x0000000011, flags:0x71, incr:4096, dst:\n"
         "2\t{0x1fec64000}\n"
         "3\t          python-101  [002] .....  1.100000: amdgpu_vm_update_ptes: pid:100 vm_ctx:0x1 start:0x0000000020 end:0x0000000021, flags:0x71, incr:4096, dst:\n"
         "4\t{0x1fec65000,0x1fec66000}\n"
         "5\t          python-101  [002] .....  1.100100: amdgpu_vm_bo_map: bo=00000000aa, start=20, last=20, offset=0000000000, flags=60\n"
         "6\t    kworker/u12:2-7  [000] .....  1.2: amdgpu_vm_flush: ring=sdma0, id=0, hub=0, pd_addr=0000300001\n")


def test_owner_and_dst():
    d = tempfile.mkdtemp()
    Path(d, "trace_h.log").write_text(TRACE)
    rows = m.trace_for(d, "h")
    tp = m.task_process(rows)
    assert tp.get(101) == 100, f"G-OWNER: thread 101 must belong to process 100, got {tp}"
    bo_map = [r for r in rows if r[4] == "amdgpu_vm_bo_map"][0]
    assert m.owner(bo_map, tp) == 100, "G-OWNER: a bo_map from the secondary thread is the process's"
    kw = [r for r in rows if r[4] == "amdgpu_vm_flush"][0]
    assert m.owner(kw, tp) is None, "G-OWNER: a kworker is never assigned by task name"
    pt = [r for r in rows if r[4] == "amdgpu_vm_update_ptes"]
    assert pt[0][6] == [0x1fec64000] and pt[1][6] == [0x1fec65000, 0x1fec66000], f"G-DST: {pt}"


def test_tgid():
    nc = ("2026-11-16 16:56:20.382286 192.168.3.163: [ 5784.976448] amdgpu 0000:16:00.0: amdgpu: [gfxhub] page fault (src_id:0 ring:24 vmid:8 pasid:32830)\n"
          "2026-11-16 16:56:20.382298 192.168.3.163: [ 5784.976463] amdgpu 0000:16:00.0: amdgpu:  in process python pid 211495 thread python pid 211508\n"
          "2026-11-16 16:56:20.382313 192.168.3.163: [ 5784.976468] amdgpu 0000:16:00.0: amdgpu:   in page starting at address 0x000076a2cd9fb000 from client 0x1b (UTCL2)\n"
          "2026-11-16 16:56:20.382324 192.168.3.163: [ 5784.976474] amdgpu 0000:16:00.0: amdgpu: GCVM_L2_PROTECTION_FAULT_STATUS:0x00801031\n")
    f = tempfile.NamedTemporaryFile("w", delete=False, suffix=".log"); f.write(nc); f.close()
    F = m.faults(f.name, "2026-11-16 16:50")
    os.unlink(f.name)
    assert F and F[0]["pid"] == 211495 and F[0]["tid"] == 211508, f"G-TGID: {F}"


if __name__ == "__main__":
    test_owner_and_dst(); print("test_owner_and_dst: PASS")
    test_tgid(); print("test_tgid: PASS")
