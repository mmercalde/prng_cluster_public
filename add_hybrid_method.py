#!/usr/bin/env python3
"""Add run_hybrid_sieve method to sieve_filter.py"""

with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

# Find where to insert (after run_sieve method ends, around line 220)
insert_at = None
for i, line in enumerate(lines):
    if 'def run_sieve(' in line:
        # Find the end of this method
        indent_level = len(line) - len(line.lstrip())
        for j in range(i + 1, len(lines)):
            if lines[j].strip() and not lines[j].startswith(' ' * (indent_level + 1)) and not lines[j].strip().startswith('#'):
                insert_at = j
                break
        break

if not insert_at:
    print("❌ Could not find insertion point")
    exit(1)

print(f"✅ Inserting hybrid method at line {insert_at + 1}")

hybrid_method = '''
    def run_hybrid_sieve(
        self,
        prng_family: str,
        seed_start: int,
        seed_end: int,
        residues: List[int],
        strategies: List[Dict[str, Any]],
        min_match_threshold: float = 0.5,
        chunk_size: int = 100_000,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Run hybrid variable skip sieve using multiple strategies.
        Uses FULL MT19937 with adaptive skip detection.
        """
        if prng_family != 'mt19937_hybrid':
            raise ValueError(f"Hybrid sieve only supports mt19937_hybrid, got {prng_family}")
        
        with self.device:
            kernel, config = self._get_kernel(prng_family, None)
            k = len(residues)
            residues_gpu = cp.array(residues, dtype=cp.uint32)
            
            # Prepare strategy parameters
            n_strategies = len(strategies)
            strategy_max_misses = cp.array([s['max_consecutive_misses'] for s in strategies], dtype=cp.int32)
            strategy_tolerances = cp.array([s['skip_tolerance'] for s in strategies], dtype=cp.int32)
            
            all_survivors = []
            all_match_rates = []
            all_strategy_ids = []
            all_skip_sequences = []
            
            for chunk_start in range(seed_start, seed_end, chunk_size):
                chunk_end = min(chunk_start + chunk_size, seed_end)
                n_seeds = chunk_end - chunk_start
                
                # Allocate arrays
                seeds_gpu = cp.arange(chunk_start, chunk_end, dtype=cp.uint32)
                survivors_gpu = cp.zeros(n_seeds, dtype=cp.uint32)
                match_rates_gpu = cp.zeros(n_seeds, dtype=cp.float32)
                strategy_ids_gpu = cp.zeros(n_seeds, dtype=cp.uint32)
                skip_sequences_gpu = cp.zeros(n_seeds * 512, dtype=cp.uint32)
                survivor_count_gpu = cp.zeros(1, dtype=cp.uint32)
                
                # Launch kernel
                threads_per_block = 256
                blocks = (n_seeds + threads_per_block - 1) // threads_per_block
                
                kernel_args = [
                    seeds_gpu, residues_gpu, survivors_gpu,
                    match_rates_gpu, skip_sequences_gpu, strategy_ids_gpu,
                    survivor_count_gpu, n_seeds, k,
                    strategy_max_misses, strategy_tolerances, n_strategies,
                    cp.float32(min_match_threshold), cp.int32(offset)
                ]
                
                kernel((blocks,), (threads_per_block,), kernel_args)
                cp.cuda.Device().synchronize()
                
                # Collect survivors
                count = int(survivor_count_gpu[0].get())
                if count > 0:
                    survivors = survivors_gpu[:count].get().tolist()
                    rates = match_rates_gpu[:count].get().tolist()
                    strat_ids = strategy_ids_gpu[:count].get().tolist()
                    skip_seqs = skip_sequences_gpu.get().reshape(n_seeds, 512)
                    
                    for idx in range(count):
                        all_survivors.append(survivors[idx])
                        all_match_rates.append(rates[idx])
                        all_strategy_ids.append(strat_ids[idx])
                        all_skip_sequences.append(skip_seqs[idx].tolist()[:k])
            
            return {
                'survivors': all_survivors,
                'match_rates': all_match_rates,
                'strategy_ids': all_strategy_ids,
                'skip_sequences': all_skip_sequences,
                'total_tested': seed_end - seed_start
            }

'''

new_lines = lines[:insert_at] + [hybrid_method] + lines[insert_at:]

with open('sieve_filter.py', 'w') as f:
    f.writelines(new_lines)

print("✅ Added run_hybrid_sieve() method")

# Verify syntax
try:
    compile(''.join(new_lines), 'sieve_filter.py', 'exec')
    print("✅ File compiles!")
except SyntaxError as e:
    print(f"❌ Syntax error: {e}")
