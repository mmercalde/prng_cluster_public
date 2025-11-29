# Replace lines 115-175 in reverse_sieve_filter.py

    def run_reverse_sieve(self,
                          candidate_seeds: List[Dict],  # FIXED: Dict not int!
                          prng_family: str,
                          draws: List[int],
                          skip_range: Tuple[int, int] = (0, 20),  # Ignored - use candidate skip
                          min_match_threshold: float = 0.01,
                          offset: int = 0) -> Dict[str, Any]:
        """
        Run reverse sieve on candidate seeds using their EXACT skip values.
        """
        start_time = time.time()
        with self.device:
            n_candidates = len(candidate_seeds)
            k = len(draws)
            print(f"Reverse sieve: {n_candidates} candidates, {k} draws", file=sys.stderr)
            
            # Extract seeds and their specific skip values
            seeds_array = [c['seed'] if isinstance(c, dict) else c for c in candidate_seeds]
            skips_array = [c.get('skip', 0) if isinstance(c, dict) else 0 for c in candidate_seeds]
            
            # Prepare GPU arrays
            candidate_seeds_gpu = cp.array(seeds_array, dtype=cp.uint32)
            candidate_skips_gpu = cp.array(skips_array, dtype=cp.uint8)  # NEW: per-seed skips
            residues = cp.array(draws, dtype=cp.uint32)
            survivors = cp.zeros(n_candidates, dtype=cp.uint32)
            match_rates = cp.zeros(n_candidates, dtype=cp.float32)
            survivor_count = cp.zeros(1, dtype=cp.uint32)
            
            # Get kernel
            kernel, config = self._get_kernel(prng_family)
            
            # Launch kernel - FIXED: pass candidate_skips instead of skip_range
            threads_per_block = 256
            blocks = (n_candidates + threads_per_block - 1) // threads_per_block
            kernel(
                (blocks,), (threads_per_block,),
                (candidate_seeds_gpu, candidate_skips_gpu, residues, survivors, 
                 match_rates, survivor_count, cp.int32(n_candidates), cp.int32(k),
                 cp.float32(min_match_threshold), cp.int32(offset))
            )
            
            # Collect results
            count = int(survivor_count[0].get())
            survivor_records = []
            if count > 0:
                surv_cpu = survivors[:count].get()
                rates_cpu = match_rates[:count].get()
                for i in range(count):
                    survivor_records.append({
                        'seed': int(surv_cpu[i]),
                        'skip': int(skips_array[surv_cpu[i]]),  # Use original skip
                        'match_rate': float(rates_cpu[i])
                    })
            
            duration_ms = (time.time() - start_time) * 1000
            return {
                'family': prng_family,
                'survivors': survivor_records,
                'stats': {
                    'candidates_tested': n_candidates,
                    'survivors_found': count,
                    'duration_ms': duration_ms
                }
            }
