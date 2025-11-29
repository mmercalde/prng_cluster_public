#!/usr/bin/env python3
"""
Python-based updater for survivor_scorer.py
More reliable than bash sed commands
"""
import sys
import os
import re
from datetime import datetime

# Check if file exists
if not os.path.exists('survivor_scorer.py'):
    print("❌ ERROR: survivor_scorer.py not found in current directory")
    sys.exit(1)

print("=" * 70)
print("survivor_scorer.py PyTorch GPU Updater (Python)")
print("=" * 70)
print()

# Backup
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
backup_file = f"survivor_scorer.py.backup_{timestamp}"
os.system(f"cp survivor_scorer.py {backup_file}")
print(f"✅ Backup created: {backup_file}")

# Read original file
with open('survivor_scorer.py', 'r') as f:
    content = f.read()

print(f"✅ Read survivor_scorer.py")

# Step 1: Add imports after numpy import
import_block = """
# PyTorch GPU support imports
from prng_registry import (
    get_cpu_reference,
    get_pytorch_gpu_reference,
    has_pytorch_gpu,
    list_pytorch_gpu_prngs,
    get_kernel_info
)
"""

# Find numpy import and add after it
content = re.sub(
    r'(import numpy as np)',
    r'\1' + import_block,
    content,
    count=1
)

print("✅ Added imports from prng_registry")

# Step 2: Replace _vectorized_scoring_kernel method
new_method = '''    def _vectorized_scoring_kernel(self, seeds_tensor, lottery_history_tensor, device):
        """
        GPU-vectorized scoring kernel - uses PyTorch GPU from prng_registry.
        Version 2.2 - Now uses prng_registry PyTorch GPU implementations.
        """
        batch_size = seeds_tensor.shape[0]
        history_len = lottery_history_tensor.shape[0]
        predictions = None
        
        # Try PyTorch GPU path first
        if has_pytorch_gpu(self.prng_type):
            try:
                if hasattr(self, 'logger'):
                    self.logger.info(f"✅ Using PyTorch GPU for {self.prng_type}")
                
                prng_func = get_pytorch_gpu_reference(self.prng_type)
                prng_info = get_kernel_info(self.prng_type)
                prng_params = prng_info.get('default_params', {})
                
                predictions = prng_func(
                    seeds=seeds_tensor,
                    n=history_len,
                    mod=self.mod,
                    device=device,
                    skip=0,
                    **prng_params
                )
                
            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.error(f"❌ PyTorch GPU failed for {self.prng_type}: {e}")
                    self.logger.info(f"   Falling back to CPU...")
                predictions = None
        
        # CPU Fallback
        if predictions is None:
            if hasattr(self, 'logger'):
                available_gpu = list_pytorch_gpu_prngs()
                self.logger.info(
                    f"ℹ️  {self.prng_type} using CPU fallback "
                    f"(PyTorch GPU available for: {available_gpu})"
                )
            
            prng_func = get_cpu_reference(self.prng_type)
            prng_info = get_kernel_info(self.prng_type)
            prng_params = prng_info.get('default_params', {})
            
            seeds_cpu = seeds_tensor.cpu().numpy()
            predictions_cpu = np.zeros((batch_size, history_len), dtype=np.int64)
            
            for idx in range(batch_size):
                sequence = prng_func(
                    seed=int(seeds_cpu[idx]),
                    n=history_len,
                    skip=0,
                    **prng_params
                )
                predictions_cpu[idx] = sequence[:history_len]
            
            predictions = torch.tensor(predictions_cpu, dtype=torch.int64, device=device)
        
        # Compare on GPU
        matches = (predictions == lottery_history_tensor.unsqueeze(0))
        scores = matches.float().sum(dim=1) / history_len
        
        return scores
'''

# Find and replace the _vectorized_scoring_kernel method
# Pattern: from def _vectorized_scoring_kernel to the next def at same indentation
pattern = r'    def _vectorized_scoring_kernel\(self,.*?\n(?=    def [a-z_]|\nclass |\Z)'

if re.search(pattern, content, re.DOTALL):
    content = re.sub(pattern, new_method + '\n', content, count=1, flags=re.DOTALL)
    print("✅ Replaced _vectorized_scoring_kernel method")
else:
    print("⚠️  Could not find _vectorized_scoring_kernel method")
    print("   Method may need to be added manually")

# Write updated file
with open('survivor_scorer.py', 'w') as f:
    f.write(content)

print("✅ Wrote survivor_scorer.py")

# Verify syntax
print()
print("✅ Verifying syntax...")
result = os.system("python3 -c 'from survivor_scorer import SurvivorScorer' 2>/dev/null")

if result == 0:
    print("✅ survivor_scorer.py syntax OK")
    print()
    print("=" * 70)
    print("✅ UPDATE COMPLETE!")
    print("=" * 70)
    print()
    print(f"Backup: {backup_file}")
    print(f"Modified: survivor_scorer.py")
    sys.exit(0)
else:
    print("❌ Syntax error detected!")
    print("   Restoring from backup...")
    os.system(f"cp {backup_file} survivor_scorer.py")
    print("   Restored: survivor_scorer.py")
    sys.exit(1)
