#!/usr/bin/env python3
"""Add hybrid_variable_skip_sieve method to sieve_filter.py"""

# Read the file
with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

# Insert after line 223 (after run_sieve method ends, before execute_sieve_job)
insert_at = 223

hybrid_method = """
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
        \"\"\"Run hybrid multi-strategy variable skip sieve\"\"\"
        
        # Import strategy helper
        try:
            from hybrid_strategy import analyze_skip_pattern
        except ImportError:
            print("WARNING: hybrid_strategy module not found, using basic analysis", file=sys.stderr)
            def analyze_skip_pattern(pattern):
                import statistics
                return {
                    'min': min(pattern) if pattern else 0,
                    'max': max(pattern) if pattern else 0,
                    'avg': statistics.mean(pattern) if pattern else 0,
                    'variance': statistics.variance(pattern) if len(pattern) > 1 else 0,
                    'std_dev': statistics.stdev(pattern) if len(pattern) > 1 else 0,
                }
        
        # Only mt19937_hybrid supports this mode
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
            
            # Result containers
            all_survivors = []
            all_match_rates = []
            all_strategy_ids = []
            all_skip_sequences = []
            total_tested = 0
            start_time = time.time()
            
            # Process in smaller chunks for hybrid (more memory intensive)
            for chunk_start in range(seed_start, seed_end, chunk_size):
                chunk_end = min(chunk_start + chunk_size, seed_end)
                n_seeds = chunk_end - chunk_start
                
                # Allocate arrays
                seeds_gpu = cp.arange(chunk_start, chunk_end, dtype=cp.uint32)
                survivors_gpu = cp.zeros(n_seeds, dtype=cp.uint32)
                match_rates_gpu = cp.zeros(n_seeds, dtype=cp.float32)
                strategy_ids_gpu = cp.zeros(n_seeds, dtype=cp.uint32)
                skip_sequences_gpu = cp.zeros(n_seeds * 512, dtype=cp.uint32)  # 512 draws max
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
                    
                    for i in range(count):
                        if rates[i] >= min_match_threshold:
                            all_survivors.append(survivors[i])
                            all_match_rates.append(rates[i])
                            all_strategy_ids.append(strat_ids[i])
                            # Extract skip sequence for this survivor
                            skip_seq = skip_seqs[i, :k].tolist()
                            all_skip_sequences.append(skip_seq)
                
                total_tested += n_seeds
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Build detailed survivor records with skip patterns
            survivor_records = []
            for seed, rate, strat_id, skip_seq in zip(
                all_survivors, all_match_rates, all_strategy_ids, all_skip_sequences
            ):
                matches = int(rate * k)
                skip_stats = analyze_skip_pattern(skip_seq)
                
                survivor_records.append({
                    'seed': int(seed),
                    'family': 'mt19937_hybrid',
                    'match_rate': float(rate),
                    'matches': matches,
                    'total': k,
                    'strategy_id': int(strat_id),
                    'strategy_name': strategies[strat_id]['name'] if strat_id < len(strategies) else 'unknown',
                    'skip_pattern': skip_seq,
                    'skip_stats': skip_stats
                })
            
            return {
                'family': 'mt19937_hybrid',
                'seed_range': {'start': seed_start, 'end': seed_end},
                'survivors': survivor_records,
                'strategies_tested': n_strategies,
                'stats': {
                    'seeds_tested': total_tested,
                    'survivors_found': len(survivor_records),
                    'duration_ms': duration_ms,
                    'seeds_per_sec': total_tested / (duration_ms / 1000) if duration_ms > 0 else 0
                }
            }

"""

# Insert the method
new_lines = lines[:insert_at] + [hybrid_method] + lines[insert_at:]

# Write back
with open('sieve_filter.py', 'w') as f:
    f.writelines(new_lines)

print(f"✅ Added run_hybrid_sieve method at line {insert_at}")

# Test syntax
try:
    with open('sieve_filter.py', 'r') as f:
        compile(f.read(), 'sieve_filter.py', 'exec')
    print("✅ File syntax is valid!")
except SyntaxError as e:
    print(f"❌ Syntax error: {e}")
