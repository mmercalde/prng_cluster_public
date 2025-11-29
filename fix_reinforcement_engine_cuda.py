#!/usr/bin/env python3
"""
fix_reinforcement_engine_cuda.py
Fixes hardcoded CUDA device references in reinforcement_engine.py that break 
CUDA_VISIBLE_DEVICES isolation.

THE BUGS:
1. Lines 103-104: Hardcodes 'cuda:0' instead of respecting CUDA_VISIBLE_DEVICES
2. Lines 115-119: Iterates over device_count() which returns physical count, not visible count
3. Lines 608-610: DataParallel with hardcoded device_ids=[0, 1]

THE FIX:
- Always use cuda:0 (the logical device that CUDA_VISIBLE_DEVICES maps correctly)
- Remove multi-GPU initialization that conflicts with env var isolation
- Disable DataParallel when running isolated jobs (single GPU per process)
"""

import sys
import shutil
from datetime import datetime

def fix_reinforcement_engine(filepath="reinforcement_engine.py"):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Backup
    backup_path = f"{filepath}.backup.cuda_fix.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy(filepath, backup_path)
    print(f"✅ Created backup: {backup_path}")
    
    # FIX 1: Replace the early CUDA initialization function (lines ~95-121)
    old_init_cuda = '''def init_cuda_context():
    """
    Initialize CUDA context early to prevent cuBLAS warnings
    This fixes the warning:
    "Attempting to run cuBLAS, but there was no current CUDA context!"
    """
    if torch.cuda.is_available():
        # Check if CUDA is already initialized (don't re-init if GPU already in use)
        try:
            device = torch.device('cuda:0')
            torch.cuda.set_device(device)
            # Force CUDA context creation with a dummy operation
            _ = torch.zeros(1).to(device)
        except RuntimeError as e:
            if "busy or unavailable" in str(e):
                # CUDA already initialized by another process/thread - that's fine
                return True
            else:
                raise
        # Initialize both GPUs if available
        if torch.cuda.device_count() > 1:
            for i in range(torch.cuda.device_count()):
                device_i = torch.device(f'cuda:{i}')
                _ = torch.zeros(1).to(device_i)
        return True
    return False'''
    
    new_init_cuda = '''def init_cuda_context():
    """
    Initialize CUDA context early to prevent cuBLAS warnings
    This fixes the warning:
    "Attempting to run cuBLAS, but there was no current CUDA context!"
    
    IMPORTANT: Always use cuda:0 (logical device). CUDA_VISIBLE_DEVICES handles
    the mapping to physical GPUs. Do NOT iterate over device_count() as that
    returns physical count, not visible count.
    """
    if torch.cuda.is_available():
        try:
            # Always use cuda:0 - CUDA_VISIBLE_DEVICES maps this to the correct physical GPU
            device = torch.device('cuda:0')
            torch.cuda.set_device(device)
            # Force CUDA context creation with a dummy operation
            _ = torch.zeros(1).to(device)
            return True
        except RuntimeError as e:
            if "busy or unavailable" in str(e):
                # CUDA already initialized by another process/thread - that's fine
                return True
            else:
                raise
    return False'''
    
    if old_init_cuda in content:
        content = content.replace(old_init_cuda, new_init_cuda)
        print("✅ Fixed init_cuda_context() - removed multi-GPU iteration")
    else:
        print("⚠️  Could not find exact init_cuda_context() block - may need manual fix")
    
    # FIX 2: Replace the _wrap_model DataParallel section
    old_wrap = '''        elif torch.cuda.device_count() > 1:
            self.logger.info(f"  🚀 Using {torch.cuda.device_count()} GPUs for training!")
            self.model = nn.DataParallel(self.model, device_ids=[0, 1])
            for i in range(torch.cuda.device_count()):
                self.logger.info(f"     GPU {i}: {torch.cuda.get_device_name(i)}")'''
    
    new_wrap = '''        elif torch.cuda.device_count() > 1 and os.environ.get('CUDA_VISIBLE_DEVICES') is None:
            # Only use DataParallel if CUDA_VISIBLE_DEVICES is not set (full GPU access)
            # When CUDA_VISIBLE_DEVICES is set, we're running isolated single-GPU jobs
            self.logger.info(f"  🚀 Using {torch.cuda.device_count()} GPUs for training!")
            device_ids = list(range(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
            for i in device_ids:
                self.logger.info(f"     GPU {i}: {torch.cuda.get_device_name(i)}")'''
    
    if old_wrap in content:
        content = content.replace(old_wrap, new_wrap)
        print("✅ Fixed _wrap_model() - DataParallel now respects CUDA_VISIBLE_DEVICES")
    else:
        print("⚠️  Could not find exact _wrap_model() DataParallel block - may need manual fix")
    
    # FIX 3: Make sure 'os' is imported (for os.environ check)
    if 'import os' not in content:
        # Add os import after other imports
        content = content.replace('import torch', 'import os\nimport torch')
        print("✅ Added 'import os'")
    
    # Write fixed file
    with open(filepath, 'w') as f:
        f.write(content)
    
    # Verify syntax
    print("\n=== Syntax check ===")
    import subprocess
    result = subprocess.run([sys.executable, '-m', 'py_compile', filepath],
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Python syntax OK")
    else:
        print(f"❌ Syntax error: {result.stderr}")
        print("Restoring backup...")
        shutil.copy(backup_path, filepath)
        return False
    
    print(f"\n✅ Fix applied successfully!")
    print(f"   Backup: {backup_path}")
    print(f"\n   Changes:")
    print(f"   - init_cuda_context(): Now only inits cuda:0 (respects CUDA_VISIBLE_DEVICES)")
    print(f"   - _wrap_model(): DataParallel disabled when CUDA_VISIBLE_DEVICES is set")
    print(f"\nDon't forget to push to remote nodes:")
    print(f"   scp reinforcement_engine.py 192.168.3.120:~/distributed_prng_analysis/")
    print(f"   scp reinforcement_engine.py 192.168.3.154:~/distributed_prng_analysis/")
    
    return True

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "reinforcement_engine.py"
    success = fix_reinforcement_engine(filepath)
    sys.exit(0 if success else 1)
