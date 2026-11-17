#!/usr/bin/env python3
"""CPU-only gates for scripts/rig_fault_recorder.py. No ssh, no host, no GPU.

G-ALLOW    the snapshot script reads only allow-listed files; no file that acts on read or
           touches registers/SMU/HWS can appear in it.
G-ARM      `arm` refuses without --i-have-go and never sshes; the arm script enables only the
           listed events, never NEVER_EVENTS; disarm turns every one off and restores state.
G-TRIG     the trigger machine takes T1..T7 at the right lines, once each, in order, never T5
           after phase-4 dispatch, and FAULT on a netconsole page-fault line.
G-PARSE    summarize() reads the idle sample's sections (kfd mem_limit, meminfo, cards).
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import rig_fault_recorder as r  # noqa: E402

DENY = ("amdgpu_evict", "amdgpu_gpu_recover", "amdgpu_test_ib", "amdgpu_benchmark",
        "amdgpu_preempt_ib", "amdgpu_regs", "amdgpu_pm_info", "amdgpu_sensors", "amdgpu_smu",
        "amdgpu_gfxoff", "amdgpu_vbios", "amdgpu_firmware_info", "amdgpu_wave", "amdgpu_gpr",
        "amdgpu_iomem", "amdgpu_vram ", "hang_hws", "hqds", "mqds", "power1", "hwmon",
        "pp_", "gpu_metrics", "force_sclk", "cu_occupancy\n")


def test_allow():
    s = r.build_snap_script(None)
    for d in DENY:
        assert d not in s, f"G-ALLOW: denied name {d!r} reachable from the snapshot script"
    assert "trace_marker" not in s, "G-ALLOW: an unarmed snapshot must write nothing"
    assert "! -name 'cu_occupancy*'" in s, "G-ALLOW: KFD cu_occupancy must be excluded"
    for w in (">", "tee "):
        for line in s.splitlines():
            if w in line and "/sys/" in line and "trace_marker" not in line:
                assert "2>/dev/null" in line or "2>&1" in line or "< /proc" in line, \
                    f"G-ALLOW: possible write in snapshot script: {line}"
    m = r.build_snap_script("SNAP T1 x")
    assert m.splitlines()[0].startswith("[ -w /sys/kernel/tracing/trace_marker ]"), "marker line"


def test_arm():
    called = []
    orig = r.ssh_bash
    r.ssh_bash = lambda *a, **k: called.append(a) or (0, "", 0, 0)
    try:
        rc = r.main(["arm", "--hosts", "h1"])
    finally:
        r.ssh_bash = orig
    assert rc == 2 and not called, "G-ARM: arm without --i-have-go must refuse and ssh nothing"
    a = r.arm_script("ptes")
    for e in r.NEVER_EVENTS:
        assert e not in a, f"G-ARM: never-event {e} enabled"
    for e in r.EVENTS_BASE + r.EVENTS_PTES:
        assert f"events/{e}/enable" in a
    assert "events/amdgpu/amdgpu_vm_update_ptes" not in r.arm_script("base")
    d = r.disarm_script()
    for e in r.EVENTS_BASE + r.EVENTS_PTES:
        assert f"echo 0 > $T/events/{e}/enable" in d, f"G-ARM: disarm leaves {e} on"
    assert "tracing_on" in d and r.STATE in d


def _stage(t, st, now_active, now_done):
    t.feed(f"x derived_bound stage phase={st + 1} family=f", now_active)
    t.feed(f'x "active_count": 25, "event": "ACTIVE_STRIPES", "stage_idx": {st},', now_active)
    return t.feed(f'x "active_count": 0, "event": "ACTIVE_STRIPES", "stage_idx": {st},', now_done)


def test_disarm_restore():
    """G-DISARM: the restore block parses the saved state (incl. an unexpanded buffer) in bash."""
    import subprocess, tempfile, re as _re
    d = r.disarm_script()
    blk = [l for l in d.splitlines() if l.startswith("if [ -e ")][0]
    for saved, want_clock, want_kb in (("clock=local buffer_kb=7 (expanded: 1408)", "local", "1408"),
                                       ("clock=global buffer_kb=4096", "global", "4096")):
        with tempfile.TemporaryDirectory() as td:
            st, T = os.path.join(td, "saved"), os.path.join(td, "T")
            os.mkdir(T)
            open(st, "w").write(saved + "\n")
            b = blk.replace(r.STATE, st)
            p = subprocess.run(["bash", "-c", f"T={T}\n{b}"], capture_output=True, text=True)
            assert p.returncode == 0 and not p.stderr, f"G-DISARM: {p.stderr}"
            assert open(os.path.join(T, "trace_clock")).read().strip() == want_clock
            assert open(os.path.join(T, "buffer_size_kb")).read().strip() == want_kb
            assert not os.path.exists(st), "G-DISARM: saved state not removed"
    assert ". " + r.STATE not in d, "G-DISARM: saved state must not be sourced"


def test_triggers():
    t = r.Triggers(t5_delay=45)
    assert t.feed('x "active_count": 0, "event": "ACTIVE_STRIPES", "stage_idx": 0,', 0) == [], \
        "G-TRIG: a zero before the stage was ever active must not fire"
    assert t.feed("GATE-12 SENTINEL  : PASS - 25/25 identities", 0) == ["T1"]
    assert t.feed("GATE-12 SENTINEL  : PASS again", 0) == [], "G-TRIG: T1 once only"
    assert _stage(t, 0, 1, 2) == ["T2"]
    assert t.feed('x "active_count": 0, "event": "ACTIVE_STRIPES", "stage_idx": 0,', 3) == [], "once"
    assert _stage(t, 1, 4, 5) == ["T3"]
    assert _stage(t, 2, 6, 100) == ["T4"]
    assert t.tick(120) == [], "G-TRIG: T5 too early"
    assert t.tick(146) == ["T5"], "G-TRIG: T5 at T4+45 s"
    assert t.tick(200) == [], "G-TRIG: T5 once"
    assert _stage(t, 3, 300, 400) == ["T6"]
    f = t.feed("2026-11-15 18:28:29.449135 192.168.3.121: [16548.673016] amdgpu 0000:19:00.0: "
               "amdgpu: [gfxhub] page fault (src_id:0 ring:40 vmid:8 pasid:32839)", 301)
    assert f == ["FAULT"] and t.faults == [("192.168.3.121", "0000:19:00.0")]
    t2 = r.Triggers(45)
    _stage(t2, 2, 0, 10)
    t2.feed("x derived_bound stage phase=4 family=f", 20)
    assert t2.tick(60) == ["T5_SKIPPED_PHASE4_ALREADY_DISPATCHED"], \
        "G-TRIG: never snapshot into the phase-4 dispatch"


def test_counts_markers():
    """G-COUNTS: intervals are keyed on the sanitised marker that trace_marker actually carries."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        f.write("1\t bash-1 [002] ..... 1.0: tracing_mark_write: TFMREC SNAP_T0_vm101_ns=5_\n"
                "2\t python-9 [001] ..... 2.0: amdgpu_vm_bo_map: bo=0, start=1\n"
                "3\t bash-1 [002] ..... 3.0: tracing_mark_write: TFMREC SNAP_T1_vm101_ns=6_console\n"
                "4\t python-9 [001] ..... 4.0: amdgpu_vm_bo_unmap: bo=0, start=1\n")
    c = r.trace_counts(Path(f.name))
    os.unlink(f.name)
    assert c == {"T0": {"amdgpu_vm_bo_map": 1}, "T1": {"amdgpu_vm_bo_unmap": 1}}, c


def test_parse():
    p = HERE.parent / "logs/rig_fault_rec/idle_sample_20261116/snaps/T0_IDLE/192.168.3.121.txt"
    if not p.exists():
        return
    s = r.summarize(p.read_text())
    assert s["kfd_sys_limit_M"] > 0 and len(s["cards"]) == 8 and "mem_MemFree_kB" in s
    assert "thp_enabled" in s and "thp_khugepaged_defrag" in s and "thp_defrag" in s


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print(f"{n}: PASS")
