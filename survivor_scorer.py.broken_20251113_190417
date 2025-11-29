# function with many small operations. It's hard to fully GPU-accelerate
                    # without writing custom CUDA kernels for the whole thing.
                    # The parts that *are* accelerated (generate_sequence,
                    # score_survivor) are already using the GPU.

                    features = self.extract_ml_features(seed, lottery_history, window_metadata=meta)

                    all_results.append({
                        'seed': seed,
                        'features': features,
                        'score': features.get('score', 0.0)
                    })

                self.logger.info(f"GPU {gpu_id}: Finished processing.")

        except Exception as e:
            self.logger.error(f"FATAL ERROR on GPU {gpu_id}: {e}")
            import traceback
            traceback.print_exc()
            # Return partial results if any
            if not all_results:
                 # Create error results for all seeds
                all_results = [{'seed': s, 'features': {}, 'score': 0.0, 'error': str(e)} for s in seeds]

        return all_results

    def _score_gpu_shard(self, gpu_id: int,
                         seeds_shard: List[int],
                         lottery_history: List[int],
                         metadata_shard: Optional[List[Dict]] = None
                         ) -> List[Dict]:
        """
        Internal helper for ThreadPoolExecutor.
        Sets the device context and calls the batch scoring function.
        """
        try:
            if GPU_AVAILABLE:
                cp.cuda.Device(gpu_id).use()
                self.logger.info(f"Thread worker: Switched to GPU {gpu_id}")

            return self._batch_score_on_gpu(
                gpu_id=gpu_id,
                seeds=seeds_shard,
                lottery_history=lottery_history,
                window_metadata=metadata_shard
            )
        except Exception as e:
            self.logger.error(f"FATAL ERROR in thread for GPU {gpu_id}: {e}")
            import traceback
            traceback.print_exc()
            # Return empty results for this shard
            return [{'seed': s, 'features': {}, 'score': 0.0, 'error': str(e)} for s in seeds_shard]


    def _batch_score_dual_gpu(self, seeds: List[int],
                              lottery_history: List[int],
                              window_metadata: Optional[List[Dict]] = None
                              ) -> List[Dict]:
        """
        Performs batch scoring using two GPUs via THREADING.
        This fixes the CUDA deadlock issue.
        """

        midpoint = len(seeds) // 2
        seeds_gpu0 = seeds[:midpoint]
        seeds_gpu1 = seeds[midpoint:]

        meta_gpu0 = window_metadata[:midpoint] if window_metadata else None
        meta_gpu1 = window_metadata[midpoint:] if window_metadata else None

        self.logger.info(f"🚀 Using DUAL GPU mode (ThreadPoolExecutor) for {len(seeds)} seeds")
        self.logger.info(f"   GPU 0: {len(seeds_gpu0)} seeds")
        self.logger.info(f"   GPU 1: {len(seeds_gpu1)} seeds")

        results_gpu0 = []
        results_gpu1 = []

        timeout = 1800  # 30 minutes max (from your previous fix)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future0 = executor.submit(self._score_gpu_shard, 0, seeds_gpu0, lottery_history, meta_gpu0)
                future1 = executor.submit(self._score_gpu_shard, 1, seeds_gpu1, lottery_history, meta_gpu1)

                self.logger.info("Waiting for GPU 0...")
                results_gpu0 = future0.result(timeout=timeout)
                self.logger.info(f"✅ GPU 0 finished, {len(results_gpu0)} results.")

                self.logger.info("Waiting for GPU 1...")
                results_gpu1 = future1.result(timeout=timeout)
                self.logger.info(f"✅ GPU 1 finished, {len(results_gpu1)} results.")

        except concurrent.futures.TimeoutError:
            self.logger.error(f"❌ DUAL GPU SCORING TIMED OUT after {timeout} seconds")
            # Create error results for any seeds that didn't finish
            if not results_gpu0:
                results_gpu0 = [{'seed': s, 'features': {}, 'score': 0.0, 'error': 'Timeout'} for s in seeds_gpu0]
            if not results_gpu1:
                results_gpu1 = [{'seed': s, 'features': {}, 'score': 0.0, 'error': 'Timeout'} for s in seeds_gpu1]

        except Exception as e:
            self.logger.error(f"❌ DUAL GPU SCORING FAILED: {e}")
            if not results_gpu0:
                results_gpu0 = [{'seed': s, 'features': {}, 'score': 0.0, 'error': str(e)} for s in seeds_gpu0]
            if not results_gpu1:
                results_gpu1 = [{'seed': s, 'features': {}, 'score': 0.0, 'error': str(e)} for s in seeds_gpu1]

        # Combine results in the original order
        all_results = results_gpu0 + results_gpu1

        # Free memory on both GPUs
        if GPU_AVAILABLE:
            try:
                with cp.cuda.Device(0):
                    cp.get_default_memory_pool().free_all_blocks()
                with cp.cuda.Device(1):
                    cp.get_default_memory_pool().free_all_blocks()
                self.logger.info("Freed GPU memory pools.")
            except Exception as e:
                self.logger.warning(f"Could not free GPU memory: {e}")

        return all_results

# ============================================================================
# CLI INTERFACE (for testing)
# ============================================================================

def main():
    """CLI for testing the scorer"""
    parser = argparse.ArgumentParser(description='SurvivorScorer CLI Test')
    parser.add_argument('--seed', type=int, default=12345, help='Test seed')
    parser.add_argument('--history-file', type=str, default='synthetic_lottery.json', help='Lottery history JSON')
    parser.add_argument('--test-batch', action='store_true', help='Run a batch test')
    parser.add_argument('--dual-gpu', action='store_true', help='Use dual GPU for batch test')
    parser.add_argument('--count', type=int, default=1000, help='Number of seeds for batch test')

    args = parser.parse_args()

    try:
        with open(args.history_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                if 'draw' in data[0]:
                    lottery_history = [d['draw'] for d in data]
                elif 'number' in data[0]:
                    lottery_history = [d['number'] for d in data]
                else:
                    lottery_history = data # Assume list of ints
            else:
                lottery_history = data

        print(f"Loaded {len(lottery_history)} lottery draws from {args.history_file}")

    except Exception as e:
        print(f"Error loading {args.history_file}: {e}")
        return 1

    # Initialize scorer (now with config_dict=None by default)
    scorer = SurvivorScorer(prng_type='java_lcg', mod=1000)

    if args.test_batch:
        print(f"\n--- Testing Batch Score ---")
        print(f"Seeds: {args.count}")
        print(f"Dual GPU: {args.dual_gpu}")

        test_seeds = list(range(args.count))

        start_time = time.time()
        results = scorer.batch_score(
            test_seeds,
            lottery_history,
            use_dual_gpu=args.dual_gpu
        )
        end_time = time.time()

        print(f"\nBatch scoring complete in {end_time - start_time:.2f}s")
        print(f"Results: {len(results)}")
        if results:
            print("Sample result:")
            print(json.dumps(results[0], indent=2, default=str))

            # Check for errors
            errors = [r for r in results if 'error' in r]
            if errors:
                print(f"\nWARNING: {len(errors)} errors found in batch")
                print(f"Sample error: {errors[0]['error']}")

    else:
        print(f"\n--- Testing Single Seed ---")
        print(f"Seed: {args.seed}")

        start_time = time.time()
        features = scorer.extract_ml_features(args.seed, lottery_history)
        end_time = time.time()

        print(f"Feature extraction complete in {end_time - start_time:.4f}s")
        print(f"Total features: {len(features)}")
        print("\nSample features:")
        sample_keys = list(features.keys())[:5]
        for key in sample_keys:
            print(f"  {key}: {features[key]}")

        print("\n...")
        sample_keys = list(features.keys())[-5:]
        for key in sample_keys:
            print(f"  {key}: {features[key]}")

if __name__ == "__main__":
    main()
