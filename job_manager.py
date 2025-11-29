#!/usr/bin/env python3
"""
Job Management and Work Distribution Module
Handles dynamic job queue, work distribution, and progress tracking
"""

import queue
import threading
from dataclasses import dataclass

class DynamicJobQueue:
    """Thread-safe dynamic job queue with work stealing capabilities"""
    def __init__(self, gpu_optimizer):
        self._queue = queue.Queue()
        self._completed = []
        self._failed = []
        self._lock = threading.Lock()
        self._total_seeds_remaining = 0
        self._original_total_seeds = 0
        self.gpu_optimizer = gpu_optimizer
        self._job_counter = 0
        
    def populate_with_work(self, total_seeds: int, prng_types, mapping_types, samples: int, lmax: int, grid_size: int):
        """Populate queue with work chunks"""
        self._original_total_seeds = total_seeds
        self._total_seeds_remaining = total_seeds
        # Create smaller chunks for better work distribution
        base_chunk_size = max(2000, total_seeds // 50) # Dynamic chunk sizing
        current_seed = 1000
        for prng_type in prng_types:
            for mapping_type in mapping_types:
                seeds_left = total_seeds
                while seeds_left > 0:
                    chunk_size = min(base_chunk_size, seeds_left)
                    seed_list = list(range(current_seed, current_seed + chunk_size))
                    job_id = f"{prng_type}_{mapping_type}_dynamic_{self._job_counter}"
                    job_spec = {
                        'job_id': job_id,
                        'prng_type': prng_type,
                        'mapping_type': mapping_type,
                        'seeds': seed_list,
                        'samples': samples,
                        'lmax': lmax,
                        'grid_size': grid_size,
                        'mining_mode': False, # Will be set per worker
                        'search_type': 'correlation',
                        'attempt': 0
                    }
                    self._queue.put(job_spec)
                    current_seed += chunk_size
                    seeds_left -= chunk_size
                    self._job_counter += 1
                    
    def get_next_job_for_worker(self, worker):
        """Get next job optimized for specific worker"""
        try:
            job_spec = self._queue.get_nowait()
            # Optimize chunk size for this worker
            original_size = len(job_spec['seeds'])
            optimal_size = self.gpu_optimizer.calculate_optimal_chunk_size(worker.node.gpu_type, original_size)
            # Adjust job size if beneficial (only if significantly better)
            if optimal_size > original_size * 1.5:
                start_seed = job_spec['seeds'][0]
                new_seeds = list(range(start_seed, start_seed + min(optimal_size, self._total_seeds_remaining)))
                job_spec['seeds'] = new_seeds
            # Set mining mode based on worker
            job_spec['mining_mode'] = 'RX 6600' in worker.node.gpu_type
            return job_spec
        except queue.Empty:
            return None
            
    def complete_job(self, job_spec, result):
        """Mark job as completed"""
        with self._lock:
            if result.success:
                self._completed.append(result)
                self._total_seeds_remaining -= len(job_spec['seeds'])
            else:
                self._failed.append((job_spec, result))
                
    def get_stats(self):
        """Get current queue statistics"""
        with self._lock:
            return {
                'jobs_remaining': self._queue.qsize(),
                'jobs_completed': len(self._completed),
                'jobs_failed': len(self._failed),
                'seeds_remaining': self._total_seeds_remaining,
                'total_seeds': self._original_total_seeds,
                'completion_percentage': (1.0 - self._total_seeds_remaining / max(1, self._original_total_seeds)) * 100
            }
