#!/usr/bin/env python3
"""
S154 Fix: Explicit GPU array deletion to prevent virtual memory bloat OOM.

Root cause confirmed via kern.log:
  "Out of memory: Killed process 3008 (python) total-vm:41452828kB"

  sieve_gpu_worker.py allocates GPU arrays per job but never explicitly deletes
  them. With CUPY_CUDA_MEMORY_POOL_TYPE=none, CuPy allocates directly from OS
  and relies on Python GC — which is non-deterministic. Over hundreds of
  sequential jobs on a persistent worker, unfreed arrays accumulate and VM
  balloons to 41GB on a 7.7GB RAM machine → OOM killer fires → rig crashes.

Fix: Add explicit del of all GPU arrays + gc.collect() before _best_effort_gpu_cleanup().

Verified: patch string found exactly once, patched file parses as valid Python.
"""

import os, sys, shutil
from datetime import datetime

TARGET = os.path.expanduser('~/distributed_prng_analysis/sieve_gpu_worker.py')
BACKUP = TARGET + f'.bak_s154_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
DRY_RUN = '--dry-run' in sys.argv

OLD = '    _best_effort_gpu_cleanup()\n\n    total_tested'

NEW = (
    '    # [S154] Explicit GPU array deletion — prevents VM bloat OOM.\n'
    '    # Root cause: CUPY_CUDA_MEMORY_POOL_TYPE=none + non-deterministic GC\n'
    '    # causes unfreed CuPy arrays to accumulate → 41GB VM → OOM killer.\n'
    '    try: del seeds_gpu\n'
    '    except NameError: pass\n'
    '    try: del survivors_gpu\n'
    '    except NameError: pass\n'
    '    try: del match_rates_gpu\n'
    '    except NameError: pass\n'
    '    try: del best_skips_gpu\n'
    '    except NameError: pass\n'
    '    try: del survivor_count_gpu\n'
    '    except NameError: pass\n'
    '    try: del residues_gpu\n'
    '    except NameError: pass\n'
    '    try: del strategy_ids_gpu\n'
    '    except NameError: pass\n'
    '    try: del skip_sequences_gpu\n'
    '    except NameError: pass\n'
    '    import gc; gc.collect()\n'
    '    _best_effort_gpu_cleanup()\n\n'
    '    total_tested'
)

def main():
    if not os.path.exists(TARGET):
        print(f'ERROR: {TARGET} not found'); sys.exit(1)

    with open(TARGET, 'r') as f:
        content = f.read()

    count = content.count(OLD)
    if count != 1:
        print(f'ERROR: Expected 1 occurrence of target string, found {count}')
        sys.exit(1)

    new_content = content.replace(OLD, NEW, 1)

    # Validate syntax
    import ast
    try:
        ast.parse(new_content)
    except SyntaxError as e:
        print(f'ERROR: Patched file has syntax error: {e}')
        sys.exit(1)

    if DRY_RUN:
        print('DRY RUN — target string found exactly once, syntax valid. Safe to apply.')
        return

    shutil.copy2(TARGET, BACKUP)
    print(f'Backup: {BACKUP}')
    with open(TARGET, 'w') as f:
        f.write(new_content)
    print(f'✅ Patch applied: explicit GPU del + gc.collect() before _best_effort_gpu_cleanup()')
    print()
    print('Deploy to rigs:')
    print('  scp ~/distributed_prng_analysis/sieve_gpu_worker.py rrig6600:~/distributed_prng_analysis/')
    print('  scp ~/distributed_prng_analysis/sieve_gpu_worker.py rrig6600b:~/distributed_prng_analysis/')
    print('  scp ~/distributed_prng_analysis/sieve_gpu_worker.py rrig6600c:~/distributed_prng_analysis/')

if __name__ == '__main__':
    main()
