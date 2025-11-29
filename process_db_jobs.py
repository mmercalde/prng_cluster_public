#!/usr/bin/env python3
"""
Database Job Processor - ML/AI Friendly
NO HARDCODED VALUES - Uses existing configuration systems
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class JobProcessorConfig:
    """Configuration for job processor - ML/AI friendly"""
    db_path: str
    batch_size: Optional[int] = None
    job_types: Optional[List[str]] = None
    priority_threshold: Optional[int] = None
    output_dir: str = "results/summaries"
    verbose: bool = True
    lottery_conversion_method: str = "hash"


class JobProcessor:
    """Process database jobs using existing PRNG configurations"""
    
    def __init__(self, config: JobProcessorConfig):
        self.config = config
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0
        }
        # Get supported PRNGs from the reconstructor itself
        self._initialize_prng_support()
    
    def _initialize_prng_support(self):
        """Initialize PRNG support by querying the reconstructor"""
        try:
            from enhanced_gap_aware_reconstruction import EnhancedGapAwarePRNGReconstructor
            reconstructor = EnhancedGapAwarePRNGReconstructor()
            self.supported_prngs = reconstructor.supported_algorithms
            if self.config.verbose:
                print(f"Loaded {len(self.supported_prngs)} supported PRNG types from reconstructor")
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Could not load PRNG support: {e}")
            self.supported_prngs = []
    
    def process_jobs(self) -> Dict[str, Any]:
        """Process pending jobs based on configuration"""
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()
        
        query = "SELECT job_id, search_type, prng_type, parameters FROM search_jobs WHERE status = 'pending'"
        params = []
        
        if self.config.job_types:
            placeholders = ','.join(['?' for _ in self.config.job_types])
            query += f" AND search_type IN ({placeholders})"
            params.extend(self.config.job_types)
        
        if self.config.priority_threshold is not None:
            query += " AND priority >= ?"
            params.append(self.config.priority_threshold)
        
        query += " ORDER BY priority DESC, created_at ASC"
        
        if self.config.batch_size:
            query += f" LIMIT {self.config.batch_size}"
        
        cursor.execute(query, params)
        pending_jobs = cursor.fetchall()
        
        if not pending_jobs:
            if self.config.verbose:
                print("No pending jobs found matching criteria")
            conn.close()
            return self.stats
        
        if self.config.verbose:
            print(f"\nProcessing {len(pending_jobs)} jobs")
            print("=" * 60)
        
        for job_id, search_type, prng_type, params_json in pending_jobs:
            if self.config.verbose:
                print(f"\n[{self.stats['total_processed'] + 1}] {job_id}")
                print(f"    Type: {search_type} / {prng_type}")
            
            self.stats['total_processed'] += 1
            
            try:
                parameters = json.loads(params_json)
                result = self._execute_job(job_id, search_type, prng_type, parameters)
                
                if result['success']:
                    cursor.execute("""
                        UPDATE search_jobs 
                        SET status = 'completed', completed_at = ?
                        WHERE job_id = ?
                    """, (datetime.now().isoformat(), job_id))
                    if self.config.verbose:
                        print(f"    ✅ Completed")
                    self.stats['successful'] += 1
                else:
                    cursor.execute("UPDATE search_jobs SET status = 'failed' WHERE job_id = ?", (job_id,))
                    if self.config.verbose:
                        print(f"    ❌ Failed: {result.get('error', 'Unknown')}")
                    self.stats['failed'] += 1
                
                conn.commit()
                
            except Exception as e:
                if self.config.verbose:
                    print(f"    ❌ Error: {e}")
                cursor.execute("UPDATE search_jobs SET status = 'failed' WHERE job_id = ?", (job_id,))
                conn.commit()
                self.stats['failed'] += 1
        
        conn.close()
        
        if self.config.verbose:
            self._print_summary()
        
        return self.stats
    
    def _execute_job(self, job_id: str, search_type: str, prng_type: str, parameters: Dict) -> Dict[str, Any]:
        """Route job to appropriate executor"""
        if search_type == 'state_reconstruction':
            return self._execute_state_reconstruction(job_id, prng_type, parameters)
        elif search_type == 'historical_analysis':
            return self._execute_historical_analysis(job_id, parameters)
        else:
            self.stats['skipped'] += 1
            return {'success': False, 'error': f'Unknown job type: {search_type}'}
    
    def _convert_lottery_data(self, sequence: List[int]) -> List[int]:
        """Convert lottery draws to 32-bit values if needed"""
        try:
            from analyze_my_lottery_data import convert_lottery_to_32bit
            
            # Auto-detect if conversion needed (lottery draws are typically 0-999)
            if max(sequence) < 1000:
                if self.config.verbose:
                    print(f"    → Converting lottery data (method: {self.config.lottery_conversion_method})")
                return convert_lottery_to_32bit(sequence, method=self.config.lottery_conversion_method)
            return sequence
        except Exception as e:
            if self.config.verbose:
                print(f"    ⚠️  Conversion warning: {e}")
            return sequence
    
    def _normalize_prng_type(self, prng_type: str) -> str:
        """Normalize PRNG type to match reconstructor's expected names"""
        normalized = prng_type.lower().strip()
        
        # Common aliases - check against actual supported list
        if normalized == 'mt':
            # Check which MT variant is supported
            if 'mt19937' in self.supported_prngs:
                return 'mt19937'
            elif 'mt19937_64' in self.supported_prngs:
                return 'mt19937_64'
        
        return normalized
    
    def _execute_state_reconstruction(self, job_id: str, prng_type: str, parameters: Dict) -> Dict[str, Any]:
        """Execute state reconstruction"""
        try:
            from enhanced_gap_aware_reconstruction import EnhancedGapAwarePRNGReconstructor
            
            known_sequence = parameters.get('known_sequence', [])
            if not known_sequence:
                return {'success': False, 'error': 'No known sequence in parameters'}
            
            # Convert lottery data if needed
            converted_sequence = self._convert_lottery_data(known_sequence)
            
            # Normalize PRNG type
            normalized_prng = self._normalize_prng_type(prng_type)
            
            if self.config.verbose:
                print(f"    → Reconstructing {normalized_prng} from {len(converted_sequence)} values")
            
            # Use the reconstructor
            reconstructor = EnhancedGapAwarePRNGReconstructor()
            sparse_outputs = [(i, val) for i, val in enumerate(converted_sequence)]
            result = reconstructor.reconstruct_with_gaps(normalized_prng, sparse_outputs)
            
            if result.get('success'):
                return {'success': True, 'result': result}
            else:
                return {'success': False, 'error': result.get('error', 'Reconstruction failed')}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _execute_historical_analysis(self, job_id: str, parameters: Dict) -> Dict[str, Any]:
        """Execute historical analysis"""
        try:
            from historical_analysis_real import create_historical_analysis_real
            
            data_file = parameters.get('data_file')
            if not data_file:
                return {'success': False, 'error': 'No data_file in parameters'}
            
            output_file = f"{self.config.output_dir}/{job_id}_summary.txt"
            
            if self.config.verbose:
                print(f"    → Analyzing {data_file}")
            
            search_id = create_historical_analysis_real(data_file, output_file)
            
            return {
                'success': True,
                'search_id': search_id,
                'output_file': output_file
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _print_summary(self):
        """Print processing summary"""
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print(f"  Total: {self.stats['total_processed']}")
        print(f"  ✅ Successful: {self.stats['successful']}")
        print(f"  ❌ Failed: {self.stats['failed']}")
        print(f"  ⚠️  Skipped: {self.stats['skipped']}")
        print("=" * 60)


def process_jobs(db_path: str, **kwargs) -> Dict[str, Any]:
    """Convenience function for backward compatibility"""
    config = JobProcessorConfig(db_path=db_path, **kwargs)
    processor = JobProcessor(config)
    return processor.process_jobs()


if __name__ == '__main__':
    import sys
    
    db_path = sys.argv[1] if len(sys.argv) > 1 else 'prng_analysis.db'
    config_args = {'db_path': db_path}
    
    if '--batch-size' in sys.argv:
        idx = sys.argv.index('--batch-size')
        config_args['batch_size'] = int(sys.argv[idx + 1])
    
    if '--job-types' in sys.argv:
        idx = sys.argv.index('--job-types')
        config_args['job_types'] = sys.argv[idx + 1].split(',')
    
    if '--conversion-method' in sys.argv:
        idx = sys.argv.index('--conversion-method')
        config_args['lottery_conversion_method'] = sys.argv[idx + 1]
    
    if '--quiet' in sys.argv:
        config_args['verbose'] = False
    
    config = JobProcessorConfig(**config_args)
    processor = JobProcessor(config)
    processor.process_jobs()
