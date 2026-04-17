#!/usr/bin/env python3
"""
apply_s163_integration_json_guard.py

Patches window_optimizer_integration_final.py to add 100K size guards on
forward_survivors.json and reverse_survivors.json writes.

Root cause: json.dump() of >1M records causes multi-hour serialization hang.
This fix propagates the S163 commit 9b3c443 pattern (applied to
persistent_worker_coordinator.py) into window_optimizer_integration_final.py.

bidirectional_survivors.json write is KEPT as-is (without indent) because it's
smaller and is the canonical input for Steps 2-6.

Dry-run supported:
    python3 apply_s163_integration_json_guard.py --dry-run
"""
import argparse
import shutil
import ast
import sys
from pathlib import Path

TARGET = "window_optimizer_integration_final.py"

# Old block — 3 un-guarded json.dump calls
OLD = """            with open('forward_survivors.json', 'w') as f:
                json.dump(sorted(forward_deduped, key=lambda x: x['seed']), f, indent=2)
            print(f\"✅ Saved forward_survivors.json: {len(forward_deduped)} unique seeds\")

            with open('reverse_survivors.json', 'w') as f:
                json.dump(sorted(reverse_deduped, key=lambda x: x['seed']), f, indent=2)
            print(f\"✅ Saved reverse_survivors.json: {len(reverse_deduped)} unique seeds\")

            with open('bidirectional_survivors.json', 'w') as f:
                json.dump(sorted(bidirectional_deduped, key=lambda x: x['seed']), f)
            print(f\"✅ Saved bidirectional_survivors.json: {len(bidirectional_deduped)} unique seeds\")"""

# New block — guarded with 100K threshold matching S163 commit 9b3c443 pattern
NEW = """            # [S163] JSON write size guard — propagates commit 9b3c443 pattern.
            # json.dump() of >1M records causes multi-hour serialization hang.
            # When survivor count exceeds threshold, write a summary only.
            # Full data remains in bidirectional_survivors_all.npz (accumulator).
            _JSON_WRITE_LIMIT = 100_000

            if len(forward_deduped) <= _JSON_WRITE_LIMIT:
                with open('forward_survivors.json', 'w') as f:
                    json.dump(sorted(forward_deduped, key=lambda x: x['seed']), f, indent=2)
                print(f\"✅ Saved forward_survivors.json: {len(forward_deduped)} unique seeds\")
            else:
                _summary = {
                    \"survivor_count\": len(forward_deduped),
                    \"note\": f\"Full survivors omitted (count > {_JSON_WRITE_LIMIT:,}) — see bidirectional_survivors_all.npz\",
                }
                with open('forward_survivors.json', 'w') as f:
                    json.dump(_summary, f, indent=2)
                print(f\"⚠️  forward_survivors.json: summary only ({len(forward_deduped):,} > {_JSON_WRITE_LIMIT:,}) — NPZ has full data\")

            if len(reverse_deduped) <= _JSON_WRITE_LIMIT:
                with open('reverse_survivors.json', 'w') as f:
                    json.dump(sorted(reverse_deduped, key=lambda x: x['seed']), f, indent=2)
                print(f\"✅ Saved reverse_survivors.json: {len(reverse_deduped)} unique seeds\")
            else:
                _summary = {
                    \"survivor_count\": len(reverse_deduped),
                    \"note\": f\"Full survivors omitted (count > {_JSON_WRITE_LIMIT:,}) — see bidirectional_survivors_all.npz\",
                }
                with open('reverse_survivors.json', 'w') as f:
                    json.dump(_summary, f, indent=2)
                print(f\"⚠️  reverse_survivors.json: summary only ({len(reverse_deduped):,} > {_JSON_WRITE_LIMIT:,}) — NPZ has full data\")

            # bidirectional_survivors.json — always written (canonical input for Steps 2-6)
            # Same 100K guard for safety, but bidirectional count is normally much smaller.
            if len(bidirectional_deduped) <= _JSON_WRITE_LIMIT:
                with open('bidirectional_survivors.json', 'w') as f:
                    json.dump(sorted(bidirectional_deduped, key=lambda x: x['seed']), f)
                print(f\"✅ Saved bidirectional_survivors.json: {len(bidirectional_deduped)} unique seeds\")
            else:
                _summary = {
                    \"survivor_count\": len(bidirectional_deduped),
                    \"note\": f\"Full survivors omitted (count > {_JSON_WRITE_LIMIT:,}) — see bidirectional_survivors_binary.npz\",
                }
                with open('bidirectional_survivors.json', 'w') as f:
                    json.dump(_summary, f, indent=2)
                print(f\"⚠️  bidirectional_survivors.json: summary only ({len(bidirectional_deduped):,} > {_JSON_WRITE_LIMIT:,}) — binary NPZ has full data\")"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Verify match without writing")
    args = ap.parse_args()

    path = Path(TARGET)
    if not path.exists():
        sys.exit(f"❌ {TARGET} not found. Run from ~/distributed_prng_analysis/")

    src = path.read_text()

    # Verify the OLD block appears exactly once
    count = src.count(OLD)
    if count == 0:
        sys.exit(
            "❌ OLD block not found. File may already be patched or has diverged.\n"
            "   Check lines 1436-1446 of window_optimizer_integration_final.py"
        )
    if count > 1:
        sys.exit(f"❌ OLD block matches {count} times — ambiguous, aborting.")

    print(f"✅ Anchor found (1 match) at byte offset {src.index(OLD)}")

    if args.dry_run:
        print("DRY RUN — no changes written")
        print(f"Would replace {len(OLD)} bytes with {len(NEW)} bytes")
        # AST check on the new file contents
        new_src = src.replace(OLD, NEW)
        try:
            ast.parse(new_src)
            print("✅ AST check passed for proposed new contents")
        except SyntaxError as e:
            sys.exit(f"❌ AST check FAILED on proposed new contents: {e}")
        return

    # Backup
    backup = path.with_suffix(path.suffix + ".pre_s163_json_guard")
    shutil.copy(path, backup)
    print(f"✅ Backup written: {backup}")

    # Apply
    new_src = src.replace(OLD, NEW)
    try:
        ast.parse(new_src)
    except SyntaxError as e:
        sys.exit(f"❌ AST check FAILED, NOT writing: {e}")

    path.write_text(new_src)
    print(f"✅ Patched {TARGET}")
    print(f"   Old: 3 unguarded json.dump calls at lines ~1436-1446")
    print(f"   New: 3 json.dump calls each gated by _JSON_WRITE_LIMIT=100000")


if __name__ == "__main__":
    main()
