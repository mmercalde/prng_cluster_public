#!/usr/bin/env python3
import sqlite3
import numpy as np
import json
import time
import hashlib
import itertools
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from database_system import get_database, SearchJob, CacheResult

@dataclass
class ExhaustiveSearchConfig:
    prng_type: str
    mapping_type: str
    target_sequence: List[int]
    samples_per_seed: int = 10000
    seed_chunk_size: int = 1000000  # 1M seeds per chunk
    max_seeds: int = 2**32  # Full 32-bit space
    priority: int = 1

@dataclass
class ParameterSweepConfig:
    base_prng: str  # 'lcg', 'mt', 'xorshift'
    parameter_ranges: Dict[str, List[Any]]  # e.g., {'multiplier': [1664525, 1103515245], 'increment': [1, 12345]}
    target_sequence: List[int]
    samples_per_combo: int = 10000
    priority: int = 2

@dataclass
class StateReconstructionConfig:
    prng_type: str
    known_sequence: List[int]
    sequence_length: int
    reconstruction_method: str  # 'bruteforce', 'algebraic', 'differential'
    priority: int = 3

@dataclass
class SpecificDrawAnalysisConfig:
    target_draw: List[int]
    search_depth: int = 100000  # How many seeds to test for this draw
    variant_analysis: bool = True  # Look for seed variants in other draws
    target_lottery: str = "daily3"
    priority: int = 2


@dataclass
class HistoricalAnalysisConfig:
    """Configuration for historical pattern analysis"""
    data_file: str
    top_n: int = 10  # Top N most frequent numbers
    bottom_n: int = 10  # Bottom N least frequent numbers
    streak_threshold: int = 2  # Minimum consecutive appearances
    entropy_window: int = 100  # Window size for entropy calculation
    enable_frequency: bool = True
    enable_gaps: bool = True
    enable_streaks: bool = True
    enable_temporal: bool = True

class AdvancedSearchManager:
    def __init__(self, db_path: str = "prng_analysis.db"):
        self.db = get_database(db_path)

    def safe_get_search_id(self, job):
        """Safely extract search_id from job parameters, handling both dict and string formats"""
        if not hasattr(job, 'parameters') or job.parameters is None:
            return None
        try:
            if isinstance(job.parameters, dict):
                return job.parameters.get('search_id')
            elif isinstance(job.parameters, str):
                return json.loads(job.parameters).get('search_id')
        except:
            pass
        return None

    def _generate_parameter_combinations(self, parameter_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all possible parameter combinations"""
        keys = parameter_ranges.keys()
        values = [parameter_ranges[key] for key in keys]
        combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        return combinations

    def _create_mt_reconstruction_jobs(self, search_id: str, config: StateReconstructionConfig) -> int:
        """Create Mersenne Twister state reconstruction jobs"""
        jobs_created = 0
        job_id = f"{search_id}_mt_recon"
        job = SearchJob(
            job_id=job_id,
            search_type="state_reconstruction",
            prng_type=config.prng_type,
            mapping_type="mod",
            seed_start=0,
            seed_end=10000,  # Limited seed range for reconstruction
            samples=config.sequence_length,
            parameters={
                'search_id': search_id,
                'known_sequence': config.known_sequence,
                'reconstruction_method': config.reconstruction_method
            },
            priority=config.priority
        )
        self.db.create_search_job(job)
        jobs_created += 1
        return jobs_created

    def _create_lcg_reconstruction_jobs(self, search_id: str, config: StateReconstructionConfig) -> int:
        """Create LCG parameter reconstruction jobs"""
        jobs_created = 0
        job_id = f"{search_id}_lcg_recon"
        job = SearchJob(
            job_id=job_id,
            search_type="state_reconstruction",
            prng_type=config.prng_type,
            mapping_type="mod",
            seed_start=0,
            seed_end=10000,
            samples=config.sequence_length,
            parameters={
                'search_id': search_id,
                'known_sequence': config.known_sequence,
                'reconstruction_method': config.reconstruction_method
            },
            priority=config.priority
        )
        self.db.create_search_job(job)
        jobs_created += 1
        return jobs_created

    def _create_bruteforce_reconstruction_jobs(self, search_id: str, config: StateReconstructionConfig) -> int:
        """Create brute force reconstruction jobs"""
        jobs_created = 0
        chunk_size = 100000
        max_seeds = 1000000
        for seed_start in range(0, max_seeds, chunk_size):
            seed_end = min(seed_start + chunk_size - 1, max_seeds - 1)
            job_id = f"{search_id}_bruteforce_{seed_start}_{seed_end}"
            job = SearchJob(
                job_id=job_id,
                search_type="state_reconstruction",
                prng_type=config.prng_type,
                mapping_type="mod",
                seed_start=seed_start,
                seed_end=seed_end,
                samples=config.sequence_length,
                parameters={
                    'search_id': search_id,
                    'known_sequence': config.known_sequence,
                    'reconstruction_method': config.reconstruction_method
                },
                priority=config.priority
            )
            self.db.create_search_job(job)
            jobs_created += 1
        return jobs_created

    def create_exhaustive_search(self, config: ExhaustiveSearchConfig) -> str:
        """Create exhaustive seed search jobs"""
        search_id = f"exhaustive_{config.prng_type}_{config.mapping_type}_{int(time.time())}"
        print(f"Creating exhaustive search: {search_id}")
        print(f" PRNG: {config.prng_type}")
        print(f" Mapping: {config.mapping_type}")
        print(f" Target sequence: {config.target_sequence}")
        print(f" Total seeds to test: {config.max_seeds:,}")
        print(f" Chunk size: {config.seed_chunk_size:,}")

        jobs_created = 0
        for seed_start in range(0, config.max_seeds, config.seed_chunk_size):
            seed_end = min(seed_start + config.seed_chunk_size - 1, config.max_seeds - 1)
            job_id = f"{search_id}_chunk_{seed_start}_{seed_end}"
            job = SearchJob(
                job_id=job_id,
                search_type="exhaustive_seed",
                prng_type=config.prng_type,
                mapping_type=config.mapping_type,
                seed_start=seed_start,
                seed_end=seed_end,
                samples=config.samples_per_seed,
                parameters={
                    'search_id': search_id,
                    'target_sequence': config.target_sequence,
                    'chunk_size': config.seed_chunk_size
                },
                priority=config.priority
            )
            self.db.create_search_job(job)
            jobs_created += 1

        print(f"Created {jobs_created} exhaustive search jobs")
        return search_id

    def create_parameter_sweep(self, config: ParameterSweepConfig) -> str:
        """Create parameter sweep jobs for LCG/custom PRNGs"""
        search_id = f"param_sweep_{config.base_prng}_{int(time.time())}"
        print(f"Creating parameter sweep: {search_id}")
        print(f" Base PRNG: {config.base_prng}")
        print(f" Parameters: {config.parameter_ranges}")

        # Generate all parameter combinations
        param_combinations = self._generate_parameter_combinations(config.parameter_ranges)
        jobs_created = 0
        for i, param_combo in enumerate(param_combinations):
            job_id = f"{search_id}_params_{i}"
            job = SearchJob(
                job_id=job_id,
                search_type="parameter_sweep",
                prng_type=config.base_prng,
                mapping_type="mod",  # Default, can be overridden
                seed_start=0,
                seed_end=10000,  # Test a range of seeds for each parameter combo
                samples=config.samples_per_combo,
                parameters={
                    'search_id': search_id,
                    'param_combo': param_combo,
                    'target_sequence': config.target_sequence
                },
                priority=config.priority
            )
            self.db.create_search_job(job)
            jobs_created += 1

        print(f"Created {jobs_created} parameter sweep jobs")
        return search_id

    def create_state_reconstruction(self, config: StateReconstructionConfig) -> str:
        """Create state reconstruction analysis"""
        search_id = f"state_recon_{config.prng_type}_{int(time.time())}"
        print(f"Creating state reconstruction: {search_id}")
        print(f" PRNG: {config.prng_type}")
        print(f" Known sequence length: {len(config.known_sequence)}")
        print(f" Method: {config.reconstruction_method}")

        if config.prng_type == "mt" and config.reconstruction_method == "algebraic":
            # Mersenne Twister state reconstruction
            jobs_created = self._create_mt_reconstruction_jobs(search_id, config)
        elif config.prng_type == "lcg" and config.reconstruction_method == "algebraic":
            # LCG parameter reconstruction
            jobs_created = self._create_lcg_reconstruction_jobs(search_id, config)
        else:
            # Brute force approach
            jobs_created = self._create_bruteforce_reconstruction_jobs(search_id, config)

        print(f"Created {jobs_created} state reconstruction jobs")
        return search_id

    def create_specific_draw_analysis(self, config: SpecificDrawAnalysisConfig) -> str:
        """Analyze specific draw and find all possible seeds"""
        search_id = f"draw_analysis_{hashlib.md5(str(config.target_draw).encode()).hexdigest()[:8]}_{int(time.time())}"
        print(f"Creating specific draw analysis: {search_id}")
        print(f" Target draw: {config.target_draw}")
        print(f" Search depth: {config.search_depth:,} seeds")
        print(f" Variant analysis: {config.variant_analysis}")

        # Phase 1: Find all seeds that can produce this specific draw
        jobs_created = 0
        for prng_type in ['mt', 'xorshift', 'lcg']:
            for mapping_type in ['mod', 'scale']:
                job_id = f"{search_id}_find_{prng_type}_{mapping_type}"
                job = SearchJob(
                    job_id=job_id,
                    search_type="specific_draw_find",
                    prng_type=prng_type,
                    mapping_type=mapping_type,
                    seed_start=0,
                    seed_end=config.search_depth,
                    samples=len(config.target_draw),
                    parameters={
                        'search_id': search_id,
                        'target_draw': config.target_draw,
                        'phase': 'find_seeds'
                    },
                    priority=config.priority
                )
                self.db.create_search_job(job)
                jobs_created += 1

        # Phase 2: If variant analysis requested, search for seed variants
        if config.variant_analysis:
            for prng_type in ['mt', 'xorshift', 'lcg']:
                job_id = f"{search_id}_variant_{prng_type}"
                job = SearchJob(
                    job_id=job_id,
                    search_type="variant_analysis",
                    prng_type=prng_type,
                    mapping_type="mod",
                    seed_start=0,
                    seed_end=10000,  # Smaller seed range for variant analysis
                    samples=len(config.target_draw),
                    parameters={
                        'search_id': search_id,
                        'target_draw': config.target_draw,
                        'phase': 'variant_analysis',
                        'target_lottery': config.target_lottery
                    },
                    priority=config.priority
                )
                self.db.create_search_job(job)
                jobs_created += 1

        print(f"Created {jobs_created} specific draw analysis jobs")
        return search_id

    def create_historical_analysis(self, data_file: str, output_file: str, config: HistoricalAnalysisConfig = None) -> str:
        """Create and execute historical analysis"""
        from historical_analysis_real import create_historical_analysis_real

        try:
            # Execute the real analysis
            search_id = create_historical_analysis_real(data_file, output_file, config)

            # Create job record for tracking
            job_id = f"{search_id}_historical"
            job = SearchJob(
                job_id=job_id,
                search_type="historical_analysis",
                prng_type="analysis",
                mapping_type="statistical",
                seed_start=0,
                seed_end=0,
                samples=1,
                parameters={
                    'search_id': search_id,
                    'data_file': data_file,
                    'output_file': output_file,
                    'completed': True
                },
                priority=2
            )
            self.db.create_search_job(job)
            print(f"Created and executed historical analysis job")
            return search_id

        except Exception as e:
            print(f"Historical analysis failed: {e}")
            raise e

    def create_temporal_analysis(self, lottery_name: str, date_range: Tuple[str, str]) -> str:
        """Create temporal analysis jobs for a date range"""
        search_id = f"temporal_{lottery_name}_{int(time.time())}"
        print(f"Creating temporal analysis: {search_id}")
        print(f" Lottery: {lottery_name}")
        print(f" Date range: {date_range[0]} to {date_range[1]}")

        jobs_created = 0
        job_id = f"{search_id}_temporal"
        job = SearchJob(
            job_id=job_id,
            search_type="temporal_analysis",
            prng_type="all",
            mapping_type="all",
            seed_start=0,
            seed_end=10000,
            samples=1000,
            parameters={
                'search_id': search_id,
                'lottery_name': lottery_name,
                'date_range': list(date_range)
            },
            priority=2
        )
        self.db.create_search_job(job)
        jobs_created += 1

        print(f"Created {jobs_created} temporal analysis jobs")
        return search_id

    def get_search_progress(self, search_id: str) -> Dict[str, Any]:
        """Get progress for a specific search"""
        progress = self.db.get_exhaustive_progress(search_id)

        # Get all pending jobs and filter by search_id manually
        jobs = self.db.get_pending_jobs()
        jobs = [job for job in jobs if self.safe_get_search_id(job) == search_id]

        total_seeds = sum(job.seed_end - job.seed_start + 1 for job in jobs)
        completed_seeds = sum(p['seeds_completed'] for p in progress)
        completion_percentage = (completed_seeds / total_seeds * 100) if total_seeds > 0 else 0

        best_score = min((p['best_score'] for p in progress if p['best_score'] is not None), default=None)
        best_seed = next((p['best_seed'] for p in progress if p['best_seed'] is not None), None)

        return {
            'search_id': search_id,
            'completion_percentage': completion_percentage,
            'total_seeds': total_seeds,
            'completed_seeds': completed_seeds,
            'best_score': best_score,
            'best_seed': best_seed,
            'jobs': len(jobs),
            'progress': progress
        }

    def get_all_searches(self) -> List[Dict[str, Any]]:
        """Get status of all searches"""
        with sqlite3.connect(self.db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT DISTINCT json_extract(parameters, '$.search_id') as search_id
                FROM search_jobs
                WHERE json_extract(parameters, '$.search_id') IS NOT NULL
            ''')
            search_ids = [row['search_id'] for row in cursor.fetchall()]

        searches = []
        for search_id in search_ids:
            progress = self.get_search_progress(search_id)
            # Get all pending jobs and filter for this search
            all_jobs = self.db.get_pending_jobs()
            search_jobs = [job for job in all_jobs if self.safe_get_search_id(job) == search_id]
            search_type = search_jobs[0].search_type if search_jobs else 'unknown'

            searches.append({
                'search_id': search_id,
                'search_type': search_type,
                'progress': progress
            })
        return searches

    def cancel_search(self, search_id: str) -> int:
        """Cancel all jobs for a search"""
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.execute('''
                UPDATE search_jobs
                SET status='cancelled', completed_at=?
                WHERE json_extract(parameters, '$.search_id')=?
            ''', (datetime.now().isoformat(), search_id))
            cancelled = cursor.rowcount
            conn.commit()
        return cancelled

    def export_search_results(self, search_id: str, output_file: str) -> bool:
        """Export search results to JSON file"""
        try:
            results = self.db.get_best_results()
            results = [r for r in results if json.loads(r.parameters).get('search_id') == search_id]

            if not results:
                print(f"No results found for search_id: {search_id}")
                return False

            export_data = {
                'search_id': search_id,
                'results': [
                    {
                        'prng_type': r.prng_type,
                        'mapping_type': r.mapping_type,
                        'seed': r.seed,
                        'samples': r.samples,
                        'composite_score': r.composite_score,
                        'chi2_score': r.chi2_score,
                        'lag5_score': r.lag5_score,
                        'computed_at': r.computed_at,
                        'node_id': r.node_id,
                        'runtime': r.runtime
                    }
                    for r in results
                ],
                'exported_at': datetime.now().isoformat()
            }

            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"Search results exported to {output_file}")
            return True
        except Exception as e:
            print(f"Error exporting search results: {e}")
            return False

# Command line interface for search management
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Advanced Search Manager for PRNG Analysis')
    parser.add_argument('--db', default='prng_analysis.db', help='Database file path')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Exhaustive search
    exhaustive_parser = subparsers.add_parser('exhaustive', help='Create exhaustive search')
    exhaustive_parser.add_argument('prng_type', help='PRNG type (mt, xorshift, lcg)')
    exhaustive_parser.add_argument('mapping_type', help='Mapping type (mod, scale)')
    exhaustive_parser.add_argument('target_sequence', help='Target sequence JSON file')
    exhaustive_parser.add_argument('--samples', type=int, default=10000, help='Samples per seed')
    exhaustive_parser.add_argument('--chunk-size', type=int, default=1000000, help='Seeds per chunk')
    exhaustive_parser.add_argument("--max-seeds", type=int, default=1000000, help="Maximum seeds to test (default: 1M)")
    exhaustive_parser.add_argument("--recent", type=int, help="Filter to recent N days")

    # Parameter sweep
    param_parser = subparsers.add_parser('parameter-sweep', help='Create parameter sweep')
    param_parser.add_argument('prng_type', help='PRNG type')
    param_parser.add_argument('param_file', help='Parameter ranges JSON file')
    param_parser.add_argument('target_sequence', help='Target sequence JSON file')

    # State reconstruction
    state_parser = subparsers.add_parser('state-reconstruction', help='Create state reconstruction')
    state_parser.add_argument('prng_type', help='PRNG type')
    state_parser.add_argument('sequence_file', help='Known sequence JSON file')
    state_parser.add_argument('--method', default='bruteforce', help='Reconstruction method')

    # Draw analysis
    draw_parser = subparsers.add_parser('draw-analysis', help='Analyze specific draw')
    draw_parser.add_argument('target_draw', help='Target draw values (comma-separated)')
    draw_parser.add_argument('--depth', type=int, default=100000, help='Search depth')
    draw_parser.add_argument('--variants', action='store_true', help='Enable variant analysis')

    # Historical analysis
    hist_parser = subparsers.add_parser('historical-analysis', help='Analyze historical patterns')
    hist_parser.add_argument('data_file', help='Lottery data file')
    hist_parser.add_argument('--output', help='Output file path')

    # Progress
    progress_parser = subparsers.add_parser('progress', help='Show search progress')
    progress_parser.add_argument('search_id', nargs='?', help='Specific search ID')

    # Cancel
    cancel_parser = subparsers.add_parser('cancel', help='Cancel search')
    cancel_parser.add_argument('search_id', help='Search ID to cancel')

    # Export
    export_parser = subparsers.add_parser('export', help='Export search results')
    export_parser.add_argument('search_id', help='Search ID to export')
    export_parser.add_argument('output_file', help='Output JSON file')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        exit(1)

    manager = AdvancedSearchManager(args.db)

    if args.command == 'exhaustive':
        with open(args.target_sequence, 'r') as f:
            target_seq = json.load(f)
        config = ExhaustiveSearchConfig(
            prng_type=args.prng_type,
            mapping_type=args.mapping_type,
            target_sequence=target_seq,
            samples_per_seed=args.samples,
            seed_chunk_size=args.chunk_size,
            max_seeds=args.max_seeds
        )
        search_id = manager.create_exhaustive_search(config)
        print(f"Created exhaustive search: {search_id}")

    elif args.command == 'parameter-sweep':
        with open(args.param_file, 'r') as f:
            param_ranges = json.load(f)
        with open(args.target_sequence, 'r') as f:
            target_seq = json.load(f)
        config = ParameterSweepConfig(
            base_prng=args.prng_type,
            parameter_ranges=param_ranges,
            target_sequence=target_seq
        )
        search_id = manager.create_parameter_sweep(config)
        print(f"Created parameter sweep: {search_id}")

    elif args.command == 'state-reconstruction':
        with open(args.sequence_file, 'r') as f:
            known_seq = json.load(f)
        config = StateReconstructionConfig(
            prng_type=args.prng_type,
            known_sequence=known_seq,
            sequence_length=len(known_seq),
            reconstruction_method=args.method
        )
        search_id = manager.create_state_reconstruction(config)
        print(f"Created state reconstruction: {search_id}")

    elif args.command == 'draw-analysis':
        target_draw = [int(x.strip()) for x in args.target_draw.split(',')]
        config = SpecificDrawAnalysisConfig(
            target_draw=target_draw,
            search_depth=args.depth,
            variant_analysis=args.variants
        )
        search_id = manager.create_specific_draw_analysis(config)
        print(f"Created draw analysis: {search_id}")

    elif args.command == 'historical-analysis':
        search_id = manager.create_historical_analysis(args.data_file, args.output)
        print(f"Created historical analysis: {search_id}")

    elif args.command == 'progress':
        if args.search_id:
            progress = manager.get_search_progress(args.search_id)
            print(f"Progress for {args.search_id}:")
            for key, value in progress.items():
                print(f"  {key}: {value}")
        else:
            searches = manager.get_all_searches()
            print("All searches:")
            for search in searches:
                print(f"  {search['search_id']}: {search['search_type']} - {search['progress']['completion_percentage']:.1f}% complete")

    elif args.command == 'cancel':
        cancelled = manager.cancel_search(args.search_id)
        print(f"Cancelled {cancelled} jobs")

    elif args.command == 'export':
        success = manager.export_search_results(args.search_id, args.output_file)
        if success:
            print("Export completed successfully")
        else:
            print("Export failed")

# === ADDED: Exact MT state reconstruction job creator ===
def create_mt_state_reconstruction_exact(self, sequence_file: str, debug: bool = False):
    """
    Enqueue an exact MT19937 state reconstruction job that consumes a known 32-bit sequence file.
    parameters = {"sequence_file": "...", "debug": bool}
    """
    job = {
        "search_type": "mt_state_recon_exact",
        "description": "Exact MT19937 reconstruction from known 32-bit outputs",
        "parameters": {"sequence_file": sequence_file, "debug": bool(debug)}
    }
    job_id = self.core.db.insert_job(job) if hasattr(self.core, "db") else None
    if hasattr(self.core, "logger"):
        self.core.logger.info(f"Enqueued mt_state_recon_exact job: {job_id} -> {sequence_file}")
    return job_id or "mt_state_recon_exact"
