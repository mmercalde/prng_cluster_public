#!/usr/bin/env python3
"""DRAFT gates + mutants for docs/drafts/DRAFT_launcher_cupy_cache_reset.diff (v3; Beta X-GCVM-LOCATED R2 Q2).

CPU only, no ssh, no rig, no write outside a temporary fixture under the worktree's logs/.
The draft diff is applied to a temporary COPY of scripts/launch_fleet_manual.sh; the real launcher
is never modified. Remote CT fragments are executed locally in bash against fixture directories, with
an inert `sleep` stand-in for the worker.

Gates (each with clean controls; a gate earns mutant credit only if its clean run passes):
  G-POLICY   unset -> per_run; shared opt-out keeps /tmp/cupy_cache_gpu<N>; per-run path distinct per GPU
             and per nonce; nonce-free launch gets a fresh launch tag; bad policy / bad tag / bad root -> 64.
  G-PREP     fresh dir -> "[cache] OK" with the stand-in's PID and its effective CUPY_CACHE_DIR; refusals
             with rc 14 AND a reason, before the stand-in starts: repeated nonce (exists), pre-existing file,
             symlink, mkdir failure, hidden entry, enumeration error; env mismatch -> rc 15, stand-in killed.
  G-RECORDS  verify_cache_records: 24 good logs -> 0 + a 24-row TSV; any missing, stale-tag, duplicate,
             path-mismatch, gpu-mismatch, pid-mismatch or shared record -> nonzero (fleet refused).
  G-WIRING   the patched launcher: prep precedes the generic mkdir; the generic mkdir excludes the per-run
             cache; CUPY_CACHE_DIR=$WCACHE; post-check precedes the launch ACK; verify_cache_records runs
             before "all ... daemons dispatched"; Zeus local cache untouched; shared mode renders the two
             changed remote lines byte-identical to the original launcher.
Run:  python3 -B docs/drafts/test_launcher_cupy_cache_policy.py            (gates)
      python3 -B docs/drafts/test_launcher_cupy_cache_policy.py mutants    (gates + mutants)
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DIFF = REPO / "docs/drafts/DRAFT_launcher_cupy_cache_reset.diff"
LAUNCHER = REPO / "scripts/launch_fleet_manual.sh"
BEGIN, END = "# >>> X-CUPY-CACHE-POLICY block", "# <<< X-CUPY-CACHE-POLICY block <<<"


def patched_copy(work: Path) -> Path:
    d = work / "lp"
    (d / "scripts").mkdir(parents=True)
    shutil.copy(LAUNCHER, d / "scripts/launch_fleet_manual.sh")
    p = subprocess.run(["patch", "-s", "-p1", "-d", str(d), "-i", str(DIFF)], capture_output=True, text=True)
    assert p.returncode == 0, f"draft diff does not apply: {p.stdout}{p.stderr}"
    return d / "scripts/launch_fleet_manual.sh"


def block(launcher: Path) -> str:
    s = launcher.read_text()
    i, j = s.index(BEGIN), s.index(END)
    return s[i:j + len(END)]


def bash(script, env=None, cwd=None, timeout=60):
    e = dict(os.environ, **(env or {}))
    for k in ("RUN_NONCE", "RIG_CUPY_CACHE_POLICY", "RIG_CUPY_CACHE_ROOT"):
        if k not in (env or {}):
            e.pop(k, None)
    return subprocess.run(["bash", "-c", script], env=e, cwd=cwd, capture_output=True, text=True, timeout=timeout)


# ------------------------------------------------------------------------------------------- G-POLICY
def g_policy(launcher: Path, work: Path):
    blk = block(launcher)
    probe = blk + '\necho "P=$RIG_CUPY_CACHE_POLICY T=$CACHE_TAG A=$(cache_path_for 3) B=$(cache_path_for 4)"\n'
    root = str(work / "ct")
    r = bash(probe, {"RUN_NONCE": "gate12-20261117_170045-1", "RIG_CUPY_CACHE_ROOT": root})
    assert r.returncode == 0, f"G-POLICY: clean run failed: {r.stderr}"
    out = r.stdout.strip()
    assert out.startswith("P=per_run "), f"G-POLICY: default must be per_run: {out}"
    assert f"A={root}/cupy_cache_run_gate12-20261117_170045-1_gpu3 " in out + " ", f"G-POLICY: path: {out}"
    assert f"B={root}/cupy_cache_run_gate12-20261117_170045-1_gpu4" in out, f"G-POLICY: per-GPU path: {out}"
    r2 = bash(probe, {"RUN_NONCE": "gate12-20261117_172033-2", "RIG_CUPY_CACHE_ROOT": root})
    assert "_172033-2_gpu3" in r2.stdout and r2.stdout != r.stdout, "G-POLICY: a new nonce must select a new path"
    r3 = bash(probe, {"RIG_CUPY_CACHE_ROOT": root})
    assert re.search(r"T=launch\d{8}_\d{6}_\d+ ", r3.stdout), f"G-POLICY: nonce-free launch tag: {r3.stdout}"
    r4 = bash(probe, {"RIG_CUPY_CACHE_POLICY": "shared"})
    assert "A=/tmp/cupy_cache_gpu3 B=/tmp/cupy_cache_gpu4" in r4.stdout, f"G-POLICY: shared path: {r4.stdout}"
    for env in ({"RIG_CUPY_CACHE_POLICY": "warm"}, {"RUN_NONCE": "bad tag;rm"},
                {"RIG_CUPY_CACHE_ROOT": "relative/dir"}, {"RIG_CUPY_CACHE_ROOT": "/tmp/a b"}):
        rb = bash(probe, env)
        assert rb.returncode == 64, f"G-POLICY: {env} must exit 64, got {rb.returncode}"


# --------------------------------------------------------------------------------------------- G-PREP
STUBS = {
    "mkdir_hidden": '#!/bin/bash\n/bin/mkdir "$@" && for a in "$@"; do case "$a" in -*) ;; *) : > "$a/.hidden";; esac; done\n',
    "find_fail": '#!/bin/bash\necho "find: injected enumeration error" >&2\nexit 1\n',
    # a racing creator: the target appears between the exists check and the creation
    "mkdir_race": '#!/bin/bash\nfor a in "$@"; do case "$a" in -*) ;; *) /bin/mkdir -p "$a";; esac; done\nexec /bin/mkdir "$@"\n',
}


def prep_script(blk, wc, tag="T1", nn=3, standin_env=None, pre=""):
    se = standin_env if standin_env is not None else wc
    return (blk + f'\n{pre}\nWC={wc}; TAG={tag}; NN={nn}; RCN=$RC_REMOTE_CACHE_NOT_FRESH; RCE=$RC_REMOTE_CACHE_ENV_MISMATCH\n'
            'eval "$CACHE_PREP_BODY"\n'
            f'env -i CUPY_CACHE_DIR={se} sleep 30 & worker_pid=$!\n'
            'echo "STANDIN pid=$worker_pid"; sleep 0.5\n'   # the launcher checks after its settle sleep
            'eval "$CACHE_POST_BODY"\n'
            'echo "AFTER_POST"; kill $worker_pid 2>/dev/null; exit 0\n')


def stub_path(work, name, as_name):
    d = work / f"stub_{name}"
    d.mkdir(exist_ok=True)
    f = d / as_name
    f.write_text(STUBS[name])
    f.chmod(0o755)
    return f"{d}:{os.environ['PATH']}"


def g_prep(launcher: Path, work: Path):
    blk = block(launcher)
    ct = work / "ct"
    ct.mkdir(exist_ok=True)
    wc = ct / "cupy_cache_run_T1_gpu3"
    r = bash(prep_script(blk, wc))
    assert r.returncode == 0 and "AFTER_POST" in r.stdout, f"G-PREP: fresh dir failed: rc={r.returncode} {r.stderr}"
    m = re.search(r"STANDIN pid=(\d+)", r.stdout)
    ok = [l for l in r.stdout.splitlines() if l.startswith("[cache] OK ")]
    assert len(ok) == 1 and f" pid={m.group(1)} " in ok[0] and ok[0].endswith(f"env_CUPY_CACHE_DIR={wc}") \
        and f" path={wc} created=exclusive entries=0 " in ok[0], f"G-PREP: OK record wrong: {ok}"
    assert wc.is_dir() and not any(wc.iterdir()), "G-PREP: the dir must exist and stay empty"

    def refused(script, reason, rc=14, env=None):
        rr = bash(script, env)
        assert rr.returncode == rc, f"G-PREP: {reason}: expected rc {rc}, got {rr.returncode}: {rr.stderr}"
        assert f"reason={reason}" in rr.stderr, f"G-PREP: {reason}: no visible reason: {rr.stderr}"
        if rc == 14:
            assert "STANDIN" not in rr.stdout, f"G-PREP: {reason}: the worker must not start"
        assert "[cache] OK" not in rr.stdout, f"G-PREP: {reason}: no OK record allowed"
        return rr

    refused(prep_script(blk, wc), "exists")                                  # repeated nonce
    f = ct / "cupy_cache_run_T2_gpu3"; f.write_text("x")
    refused(prep_script(blk, f, "T2"), "exists")                             # pre-existing file
    t = ct / "target"; t.mkdir(); ln = ct / "cupy_cache_run_T3_gpu3"; ln.symlink_to(t)
    refused(prep_script(blk, ln, "T3"), "exists")                            # symlink
    dangling = ct / "cupy_cache_run_T3b_gpu3"; dangling.symlink_to(ct / "nowhere")
    refused(prep_script(blk, dangling, "T3b"), "exists")                     # dangling symlink
    refused(prep_script(blk, ct / "no_parent" / "cupy_cache_run_T4_gpu3", "T4"), "mkdir_failed")
    refused(prep_script(blk, ct / "cupy_cache_run_T4r_gpu3", "T4r"), "mkdir_failed",
            env={"PATH": stub_path(work, "mkdir_race", "mkdir")})            # creation must be exclusive
    refused(prep_script(blk, ct / "cupy_cache_run_T5_gpu3", "T5"), "nonempty_1",
            env={"PATH": stub_path(work, "mkdir_hidden", "mkdir")})          # hidden entry
    rr = refused(prep_script(blk, ct / "cupy_cache_run_T6_gpu3", "T6"), "enumeration_failed",
                 env={"PATH": stub_path(work, "find_fail", "find")})         # enumeration error
    assert "entries=0" not in rr.stdout + rr.stderr, "G-PREP: an enumeration error must never read as entries=0"
    rr = refused(prep_script(blk, ct / "cupy_cache_run_T7_gpu3", "T7", standin_env="/tmp/cupy_cache_gpu3"),
                 "env_mismatch", rc=15)
    pid = re.search(r"STANDIN pid=(\d+)", rr.stdout).group(1)
    alive = subprocess.run(["kill", "-0", pid], capture_output=True).returncode == 0
    assert not alive, "G-PREP: the mismatched worker must be killed"


# ------------------------------------------------------------------------------------------ G-RECORDS
def g_records(launcher: Path, work: Path):
    blk = block(launcher)
    tag = "gate12-T"
    root = "/tmp"
    logs = work / "logs"; logs.mkdir()

    def mk(i, kind="ok"):
        p = logs / f"dispatch_rig_gpu{i}.log"
        path = f"{root}/cupy_cache_run_{tag}_gpu{i % 8}"
        pid = 1000 + i
        rec = (f"[cache] OK tag={tag} host=rrig h gpu={i % 8} policy=per_run path={path} created=exclusive "
               f"entries=0 t_prep=1.0 pid={pid} env_CUPY_CACHE_DIR={path}")
        lines = {"ok": [rec],
                 "missing": [],
                 "stale": [rec.replace(f"tag={tag}", "tag=gate12-OLD")],
                 "dup": [rec, rec],
                 "path": [rec.replace(f"path={path} ", f"path={path}x ")],
                 "gpu": [rec.replace(f" gpu={i % 8} ", f" gpu={(i + 1) % 8} ")],
                 "pid": [rec.replace(f" pid={pid} ", f" pid={pid + 7} ")],
                 "env": [rec.replace(f"env_CUPY_CACHE_DIR={path}", "env_CUPY_CACHE_DIR=/tmp/cupy_cache_gpu0")],
                 "shared": [rec.replace("policy=per_run", "policy=shared")]}[kind]
        p.write_text("\n".join(lines + [f"[dispatch] started pid={pid} gpu={i % 8}"]) + "\n")
        return p, path, i % 8

    def run(kinds):
        for f in logs.iterdir():
            f.unlink()
        arr_l, arr_c, arr_g, arr_b = [], [], [], []
        for i, k in enumerate(kinds):
            p, path, g = mk(i, k)
            arr_l.append(str(p)); arr_c.append(path); arr_g.append(str(g)); arr_b.append(f"rig:gpu{g}")
        q = lambda a: " ".join(f'"{x}"' for x in a)
        s = (blk + f'\nLOGDIR={work}\nREMOTE_DISPATCH_LOGS=({q(arr_l)})\nREMOTE_DISPATCH_CACHE=({q(arr_c)})\n'
             f'REMOTE_DISPATCH_GPU=({q(arr_g)})\nREMOTE_DISPATCH_LABELS=({q(arr_b)})\nverify_cache_records\n')
        return bash(s, {"RUN_NONCE": tag})

    r = run(["ok"] * 24)
    assert r.returncode == 0, f"G-RECORDS: 24 good records must pass: {r.stderr}"
    tsv = (work / f"cache_records_{tag}.tsv").read_text().splitlines()
    assert len(tsv) == 25 and all(l.endswith("\tOK") for l in tsv[1:]), f"G-RECORDS: TSV: {tsv[:3]}"
    for k in ("missing", "stale", "dup", "path", "gpu", "pid", "env", "shared"):
        kinds = ["ok"] * 24; kinds[17] = k
        rk = run(kinds)
        assert rk.returncode != 0 and "CACHE RECORD DEFECT" in rk.stderr, f"G-RECORDS: a '{k}' record must refuse the fleet"


# ------------------------------------------------------------------------------------------- G-WIRING
def g_wiring(launcher: Path, work: Path):
    s = launcher.read_text()
    orig = LAUNCHER.read_text()
    i_prep = s.index('"${CACHE_PREP}mkdir -p /tmp/minerlogs $REMOTE_RELEASE_DIR${SHARED_MKDIR} || exit $RC_REMOTE_MKDIR')
    assert "CUPY_CACHE_DIR=$WCACHE \\" in s[i_prep:i_prep + 400], "G-WIRING: CUPY_CACHE_DIR must be the verified path"
    j = s.index("${CACHE_POST}printf '[dispatch] started pid=%s gpu=%s\\n'")
    assert i_prep < j, "G-WIRING: post-check must precede the launch ACK"
    sel = s[s.index('WCACHE="$(cache_path_for "$N")"'):i_prep]
    assert 'CACHE_POST="$CACHE_POST_BODY"; SHARED_MKDIR=""' in sel, "G-WIRING: per_run must keep the cache out of mkdir -p"
    assert 'CACHE_PREP=""; CACHE_POST=""; SHARED_MKDIR=" $WCACHE"' in sel, "G-WIRING: shared must keep the old mkdir"
    v = s.index("if ! verify_cache_records; then")
    assert v < s.index('echo "[launch] all $DISPATCHED of $TOTAL daemons dispatched"'), "G-WIRING: verify before success"
    assert 'exit 1' in s[v:v + 400], "G-WIRING: a defect must exit nonzero"
    assert 'CUPY_CACHE_DIR="/tmp/cupy_cache_local_gpu$N"' in s, "G-WIRING: Zeus local cache must be unchanged"
    # shared mode: render the two changed remote lines and compare with the original launcher's lines
    N, REL, RCM = "3", "/tmp/minerlogs", "11"
    new_mk = f"mkdir -p /tmp/minerlogs {REL} /tmp/cupy_cache_gpu{N} || exit {RCM}"   # ${CACHE_PREP}=""; SHARED_MKDIR=" $WCACHE"
    old_mk = "mkdir -p /tmp/minerlogs $REMOTE_RELEASE_DIR /tmp/cupy_cache_gpu$N || exit $RC_REMOTE_MKDIR" \
        .replace("$REMOTE_RELEASE_DIR", REL).replace("$N", N).replace("$RC_REMOTE_MKDIR", RCM)
    assert '"mkdir -p /tmp/minerlogs $REMOTE_RELEASE_DIR /tmp/cupy_cache_gpu$N || exit $RC_REMOTE_MKDIR' in orig
    assert new_mk == old_mk, "G-WIRING: shared-mode mkdir line differs from the original"
    assert "CUPY_CACHE_DIR=/tmp/cupy_cache_gpu$N \\" in orig, "G-WIRING: original anchor moved"


GATES = {"policy": g_policy, "prep": g_prep, "records": g_records, "wiring": g_wiring}

MUTANTS = [
    ("M1-default-shared", 'RIG_CUPY_CACHE_POLICY="${RIG_CUPY_CACHE_POLICY:-per_run}"',
     'RIG_CUPY_CACHE_POLICY="${RIG_CUPY_CACHE_POLICY:-shared}"', "policy"),
    ("M2-validation-bypass", 'case "$RIG_CUPY_CACHE_POLICY" in per_run|shared) ;;',
     'case "$RIG_CUPY_CACHE_POLICY" in *) ;;', "policy"),
    ("M3-tag-omitted", 'echo "$RIG_CUPY_CACHE_ROOT/cupy_cache_run_${CACHE_TAG}_gpu$1"',
     'echo "$RIG_CUPY_CACHE_ROOT/cupy_cache_run_gpu$1"', "policy"),
    ("M4-gpu-omitted", 'echo "$RIG_CUPY_CACHE_ROOT/cupy_cache_run_${CACHE_TAG}_gpu$1"',
     'echo "$RIG_CUPY_CACHE_ROOT/cupy_cache_run_${CACHE_TAG}"', "policy"),
    ("M5-mkdir-p", 'mkdir "$WC" 2>/dev/null || cache_refuse mkdir_failed',
     'mkdir -p "$WC" 2>/dev/null || cache_refuse mkdir_failed', "prep"),
    ("M6-exists-check-removed", 'if [ -e "$WC" ] || [ -L "$WC" ]; then cache_refuse exists; fi', ':', "prep"),
    ("M7-nonempty-check-removed", '[ "${#ENTS}" -eq 0 ] || cache_refuse "nonempty_${#ENTS}"', ':', "prep"),
    ("M8-enumeration-status-ignored", 'ENTS=$(find "$WC" -mindepth 1 -maxdepth 1 -printf .) || cache_refuse enumeration_failed',
     'ENTS=$(find "$WC" -mindepth 1 -maxdepth 1 -printf .)', "prep"),
    ("M9-env-check-removed", 'if [ "$CE" != "$WC" ]; then', 'if false; then', "prep"),
    ("M10-records-ignore-count", 'if [ "$n" -ne 1 ]; then v="records=$n"', 'if false; then v="records=$n"', "records"),
    ("M11-records-ignore-pid", '! printf \'%s\' "$rec" | grep -q " pid=$ack env_CUPY_CACHE_DIR=$exp\\$"', 'false', "records"),
    ("M12-cache-back-in-mkdir-p", 'CACHE_POST="$CACHE_POST_BODY"; SHARED_MKDIR=""',
     'CACHE_POST="$CACHE_POST_BODY"; SHARED_MKDIR=" $WCACHE"', "wiring"),
    ("M13-verify-not-called", 'if ! verify_cache_records; then', 'if false; then', "wiring"),
    ("M14-ack-before-check", "${CACHE_POST}printf '[dispatch] started pid=%s gpu=%s\\n'",
     "printf '[dispatch] started pid=%s gpu=%s\\n'", "wiring"),
]


def run_gate(name, launcher):
    work = Path(tempfile.mkdtemp(dir=str(REPO / "logs"), prefix=f"cachegate_{name}_"))
    try:
        GATES[name](launcher, work)
        return True, ""
    except AssertionError as e:
        return False, str(e)
    except (ValueError, IndexError) as e:            # a wiring anchor that vanished is a gate failure
        return False, f"anchor missing: {e}"
    finally:
        subprocess.run(["chmod", "-R", "u+w", str(work)])
        shutil.rmtree(work, ignore_errors=True)


def main():
    (REPO / "logs").mkdir(exist_ok=True)
    base = Path(tempfile.mkdtemp(dir=str(REPO / "logs"), prefix="cachegate_base_"))
    launcher = patched_copy(base)
    ok = True
    for g in GATES:
        passed, why = run_gate(g, launcher)
        print(f"G-{g.upper()}: {'PASS' if passed else 'FAIL ' + why}")
        ok &= passed
    if len(sys.argv) > 1 and sys.argv[1] == "mutants":
        for mid, old, new, gate in MUTANTS:
            src = launcher.read_text()
            if src.count(old) != 1:
                print(f"{mid}: INVALID (pattern count {src.count(old)})"); ok = False; continue
            m = base / f"{mid}.sh"
            m.write_text(src.replace(old, new))
            passed, why = run_gate(gate, m)
            print(f"{mid}: {'SURVIVED' if passed else 'DETECTED'} by G-{gate.upper()}" + ("" if passed else f" ({why[:90]})"))
            ok &= not passed
    shutil.rmtree(base, ignore_errors=True)
    print("ALL GREEN" if ok else "NOT GREEN")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
