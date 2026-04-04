#!/usr/bin/env python3
"""
apply_s161_worker_clean_exit.py
================================
TB-approved fix: worker exits cleanly after repeated ECONNREFUSED
following a successful session (Option 2 from TB ruling).

Logic:
  - Track whether worker ever completed a successful session
  - After a successful session, count consecutive ECONNREFUSED
  - After MAX_REFUSED consecutive refusals, exit cleanly
  - This prevents stale reconnect storms at Gate 2 scale (24 workers)

Apply:
    python3 apply_s161_worker_clean_exit.py --dry-run
    python3 apply_s161_worker_clean_exit.py
"""
import argparse, ast, shutil, sys
from pathlib import Path

TARGET = Path("persistent/pwc_worker_service.py")
BACKUP = Path("persistent/pwc_worker_service.py.pre_s161_clean_exit")

OLD = '''    def run_forever(self) -> None:
        self._setup_env()
        self._import_sieve()

        while True:
            try:
                self._connect()
                log.info(f"[{self.worker_id}] connected to {self.host}:{self.port}")
                self._main_loop()
            except KeyboardInterrupt:
                log.info(f"[{self.worker_id}] interrupted")
                break
            except (ConnectionError, OSError) as exc:
                log.warning(
                    f"[{self.worker_id}] transport error: {exc} "
                    f"— reconnecting in {RECONNECT_DELAY_S}s"
                )
                self._close()
                time.sleep(RECONNECT_DELAY_S)
            except Exception as exc:
                log.error(
                    f"[{self.worker_id}] unexpected error: {exc}\\n{tb.format_exc()}"
                )
                self._close()
                time.sleep(RECONNECT_DELAY_S)'''

NEW = '''    def run_forever(self) -> None:
        self._setup_env()
        self._import_sieve()

        # S161: terminal reconnect policy (TB Gate 1 ruling)
        # After a successful session, exit cleanly on repeated ECONNREFUSED
        # instead of reconnecting forever. Prevents stale-process storms at scale.
        _had_session  = False
        _refused_count = 0
        _MAX_REFUSED  = 5  # exit after 5 consecutive refusals post-session

        while True:
            try:
                self._connect()
                log.info(f"[{self.worker_id}] connected to {self.host}:{self.port}")
                _refused_count = 0  # reset on successful connect
                self._main_loop()
                _had_session = True  # mark session complete after main_loop returns
            except KeyboardInterrupt:
                log.info(f"[{self.worker_id}] interrupted")
                break
            except (ConnectionError, OSError) as exc:
                err_str = str(exc)
                if _had_session and "Connection refused" in err_str:
                    _refused_count += 1
                    if _refused_count >= _MAX_REFUSED:
                        log.info(
                            f"[{self.worker_id}] coordinator gone after session "
                            f"({_refused_count} refused) — exiting cleanly"
                        )
                        break
                log.warning(
                    f"[{self.worker_id}] transport error: {exc} "
                    f"— reconnecting in {RECONNECT_DELAY_S}s"
                )
                self._close()
                time.sleep(RECONNECT_DELAY_S)
            except Exception as exc:
                log.error(
                    f"[{self.worker_id}] unexpected error: {exc}\\n{tb.format_exc()}"
                )
                self._close()
                time.sleep(RECONNECT_DELAY_S)'''

def apply(dry_run=False):
    content = TARGET.read_text(encoding="utf-8")
    count = content.count(OLD)
    if count == 0:
        print("ERROR: anchor not found")
        sys.exit(1)
    if count > 1:
        print(f"ERROR: {count} matches")
        sys.exit(1)
    print("OK anchor: [run_forever terminal reconnect policy]")
    if dry_run:
        print("DRY RUN — no files modified")
        return
    shutil.copy(TARGET, BACKUP)
    print(f"Backup: {BACKUP}")
    content = content.replace(OLD, NEW, 1)
    try:
        ast.parse(content)
    except SyntaxError as e:
        print(f"AST FAIL: {e}")
        sys.exit(1)
    print("AST OK")
    TARGET.write_text(content, encoding="utf-8")
    print(f"Written: {TARGET}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    apply(args.dry_run)
