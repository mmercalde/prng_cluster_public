#!/usr/bin/env python3

import sqlite3
import json
import hashlib
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

@dataclass
class CacheResult:
    prng_type: str
    mapping_type: str
    seed: int
    samples: int
    parameters: str  # JSON string of additional parameters
    chi2_score: float
    lag5_score: float
    composite_score: float
    full_results: str  # JSON string of complete results
    computed_at: str
    node_id: str
    runtime: float

@dataclass
class SearchJob:
    job_id: str
    search_type: str
    prng_type: str
    mapping_type: str
    seed_start: int
    seed_end: int
    samples: int
    parameters: Dict[str, Any]
    priority: int = 1
    status: str = 'pending'
    assigned_node: Optional[str] = None
    created_at: str = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

class DistributedPRNGDatabase:
    def __init__(self, db_path: str = "prng_analysis.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prng_type TEXT NOT NULL,
                    mapping_type TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    samples INTEGER NOT NULL,
                    parameters TEXT NOT NULL,
                    parameter_hash TEXT NOT NULL,
                    chi2_score REAL NOT NULL,
                    lag5_score REAL NOT NULL,
                    composite_score REAL NOT NULL,
                    full_results TEXT NOT NULL,
                    computed_at TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    runtime REAL NOT NULL,
                    UNIQUE(prng_type, mapping_type, seed, samples, parameter_hash)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS search_jobs (
                    job_id TEXT PRIMARY KEY,
                    search_type TEXT NOT NULL,
                    prng_type TEXT NOT NULL,
                    mapping_type TEXT NOT NULL,
                    seed_start INTEGER NOT NULL,
                    seed_end INTEGER NOT NULL,
                    samples INTEGER NOT NULL,
                    parameters TEXT NOT NULL,
                    priority INTEGER DEFAULT 1,
                    status TEXT DEFAULT 'pending',
                    assigned_node TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS exhaustive_progress (
                    search_id TEXT NOT NULL,
                    prng_type TEXT NOT NULL,
                    mapping_type TEXT NOT NULL,
                    seed_range_start INTEGER NOT NULL,
                    seed_range_end INTEGER NOT NULL,
                    seeds_completed INTEGER DEFAULT 0,
                    best_score REAL,
                    best_seed INTEGER,
                    last_updated TEXT NOT NULL,
                    PRIMARY KEY(search_id, prng_type, mapping_type, seed_range_start)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS lottery_draws (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lottery_name TEXT NOT NULL,
                    draw_date TEXT NOT NULL,
                    draw_number INTEGER,
                    winning_numbers TEXT NOT NULL,
                    number_hash TEXT NOT NULL,
                    metadata TEXT,
                    UNIQUE(lottery_name, draw_date, draw_number)
                )
            ''')
            
            # Create indices for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_cache_lookup ON cache_results(prng_type, mapping_type, seed, samples, parameter_hash)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_job_status ON search_jobs(status, priority)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_progress_search ON exhaustive_progress(search_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_lottery_hash ON lottery_draws(number_hash)')
    
    def get_parameter_hash(self, parameters: Dict[str, Any]) -> str:
        """Create hash of parameters for caching"""
        param_str = json.dumps(parameters, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def check_cached_result(self, prng_type: str, mapping_type: str, seed: int, 
                           samples: int, parameters: Dict[str, Any]) -> Optional[CacheResult]:
        """Check if result already exists in cache"""
        param_hash = self.get_parameter_hash(parameters)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT * FROM cache_results 
                WHERE prng_type=? AND mapping_type=? AND seed=? AND samples=? AND parameter_hash=?
            ''', (prng_type, mapping_type, seed, samples, param_hash))
            
            row = cursor.fetchone()
            if row:
                return CacheResult(
                    prng_type=row['prng_type'],
                    mapping_type=row['mapping_type'],
                    seed=row['seed'],
                    samples=row['samples'],
                    parameters=row['parameters'],
                    chi2_score=row['chi2_score'],
                    lag5_score=row['lag5_score'],
                    composite_score=row['composite_score'],
                    full_results=row['full_results'],
                    computed_at=row['computed_at'],
                    node_id=row['node_id'],
                    runtime=row['runtime']
                )
        return None
    
    def store_result(self, result: CacheResult, parameters: Dict[str, Any]):
        """Store computation result in cache"""
        param_hash = self.get_parameter_hash(parameters)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO cache_results 
                (prng_type, mapping_type, seed, samples, parameters, parameter_hash,
                 chi2_score, lag5_score, composite_score, full_results, 
                 computed_at, node_id, runtime)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.prng_type, result.mapping_type, result.seed, result.samples,
                result.parameters, param_hash, result.chi2_score, result.lag5_score,
                result.composite_score, result.full_results, result.computed_at,
                result.node_id, result.runtime
            ))
    
    def get_uncached_seeds(self, prng_type: str, mapping_type: str, 
                          seed_start: int, seed_end: int, samples: int,
                          parameters: Dict[str, Any]) -> List[int]:
        """Get list of seeds that haven't been computed yet"""
        param_hash = self.get_parameter_hash(parameters)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT seed FROM cache_results 
                WHERE prng_type=? AND mapping_type=? AND samples=? AND parameter_hash=?
                AND seed BETWEEN ? AND ?
            ''', (prng_type, mapping_type, samples, param_hash, seed_start, seed_end))
            
            cached_seeds = set(row[0] for row in cursor.fetchall())
        
        all_seeds = set(range(seed_start, seed_end + 1))
        uncached_seeds = list(all_seeds - cached_seeds)
        return sorted(uncached_seeds)
    
    def create_search_job(self, job: SearchJob) -> str:
        """Create a new search job"""
        if job.created_at is None:
            job.created_at = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO search_jobs 
                (job_id, search_type, prng_type, mapping_type, seed_start, seed_end,
                 samples, parameters, priority, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                job.job_id, job.search_type, job.prng_type, job.mapping_type,
                job.seed_start, job.seed_end, job.samples, json.dumps(job.parameters),
                job.priority, job.status, job.created_at
            ))
        
        return job.job_id
    
    def get_pending_jobs(self, limit: int = 10) -> List[SearchJob]:
        """Get pending jobs ordered by priority"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT * FROM search_jobs 
                WHERE status='pending' 
                ORDER BY priority DESC, created_at ASC 
                LIMIT ?
            ''', (limit,))
            
            jobs = []
            for row in cursor.fetchall():
                job = SearchJob(
                    job_id=row['job_id'],
                    search_type=row['search_type'],
                    prng_type=row['prng_type'],
                    mapping_type=row['mapping_type'],
                    seed_start=row['seed_start'],
                    seed_end=row['seed_end'],
                    samples=row['samples'],
                    parameters=json.loads(row['parameters']),
                    priority=row['priority'],
                    status=row['status'],
                    assigned_node=row['assigned_node'],
                    created_at=row['created_at'],
                    started_at=row['started_at'],
                    completed_at=row['completed_at']
                )
                jobs.append(job)
            
            return jobs
    
    def assign_job(self, job_id: str, node_id: str):
        """Assign job to a node"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE search_jobs 
                SET status='running', assigned_node=?, started_at=?
                WHERE job_id=?
            ''', (node_id, datetime.now().isoformat(), job_id))
    
    def complete_job(self, job_id: str):
        """Mark job as completed"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE search_jobs 
                SET status='completed', completed_at=?
                WHERE job_id=?
            ''', (datetime.now().isoformat(), job_id))
    
    def update_exhaustive_progress(self, search_id: str, prng_type: str, mapping_type: str,
                                 seed_range_start: int, seed_range_end: int, 
                                 seeds_completed: int, best_score: float = None, 
                                 best_seed: int = None):
        """Update progress for exhaustive search"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO exhaustive_progress
                (search_id, prng_type, mapping_type, seed_range_start, seed_range_end,
                 seeds_completed, best_score, best_seed, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                search_id, prng_type, mapping_type, seed_range_start, seed_range_end,
                seeds_completed, best_score, best_seed, datetime.now().isoformat()
            ))
    
    def get_exhaustive_progress(self, search_id: str) -> List[Dict]:
        """Get progress for exhaustive search"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT * FROM exhaustive_progress WHERE search_id=?
                ORDER BY seed_range_start
            ''', (search_id,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def store_lottery_draw(self, lottery_name: str, draw_date: str, draw_number: int,
                          winning_numbers: List[int], metadata: Dict = None):
        """Store lottery draw data"""
        numbers_str = json.dumps(winning_numbers)
        numbers_hash = hashlib.md5(numbers_str.encode()).hexdigest()
        metadata_str = json.dumps(metadata) if metadata else '{}'
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO lottery_draws
                (lottery_name, draw_date, draw_number, winning_numbers, number_hash, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (lottery_name, draw_date, draw_number, numbers_str, numbers_hash, metadata_str))
    
    def find_matching_draws(self, target_numbers: List[int], lottery_name: str = None) -> List[Dict]:
        """Find draws that match target numbers"""
        numbers_str = json.dumps(target_numbers)
        numbers_hash = hashlib.md5(numbers_str.encode()).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            if lottery_name:
                cursor = conn.execute('''
                    SELECT * FROM lottery_draws 
                    WHERE number_hash=? AND lottery_name=?
                ''', (numbers_hash, lottery_name))
            else:
                cursor = conn.execute('''
                    SELECT * FROM lottery_draws WHERE number_hash=?
                ''', (numbers_hash,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_best_results(self, prng_type: str = None, mapping_type: str = None, 
                        limit: int = 100) -> List[CacheResult]:
        """Get best results (lowest composite scores)"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = 'SELECT * FROM cache_results'
            params = []
            
            if prng_type or mapping_type:
                conditions = []
                if prng_type:
                    conditions.append('prng_type=?')
                    params.append(prng_type)
                if mapping_type:
                    conditions.append('mapping_type=?')
                    params.append(mapping_type)
                query += ' WHERE ' + ' AND '.join(conditions)
            
            query += ' ORDER BY composite_score ASC LIMIT ?'
            params.append(limit)
            
            cursor = conn.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                result = CacheResult(
                    prng_type=row['prng_type'],
                    mapping_type=row['mapping_type'],
                    seed=row['seed'],
                    samples=row['samples'],
                    parameters=row['parameters'],
                    chi2_score=row['chi2_score'],
                    lag5_score=row['lag5_score'],
                    composite_score=row['composite_score'],
                    full_results=row['full_results'],
                    computed_at=row['computed_at'],
                    node_id=row['node_id'],
                    runtime=row['runtime']
                )
                results.append(result)
            
            return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            # Cache statistics
            cursor = conn.execute('SELECT COUNT(*) FROM cache_results')
            total_cached = cursor.fetchone()[0]
            
            cursor = conn.execute('''
                SELECT prng_type, mapping_type, COUNT(*) as count 
                FROM cache_results 
                GROUP BY prng_type, mapping_type
            ''')
            cache_by_type = {f"{row[0]}-{row[1]}": row[2] for row in cursor.fetchall()}
            
            # Job statistics
            cursor = conn.execute('''
                SELECT status, COUNT(*) as count 
                FROM search_jobs 
                GROUP BY status
            ''')
            jobs_by_status = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Best scores
            cursor = conn.execute('''
                SELECT prng_type, mapping_type, MIN(composite_score) as best_score 
                FROM cache_results 
                GROUP BY prng_type, mapping_type
            ''')
            best_scores = {f"{row[0]}-{row[1]}": row[2] for row in cursor.fetchall()}
            
            return {
                'total_cached_results': total_cached,
                'cache_by_type': cache_by_type,
                'jobs_by_status': jobs_by_status,
                'best_scores': best_scores,
                'database_size_mb': self.get_db_size_mb()
            }
    
    def get_db_size_mb(self) -> float:
        """Get database file size in MB"""
        try:
            import os
            size_bytes = os.path.getsize(self.db_path)
            return size_bytes / (1024 * 1024)
        except:
            return 0.0

# Factory function for easy database access
_db_instance = None
_db_lock = threading.Lock()

def get_database(db_path: str = "prng_analysis.db") -> DistributedPRNGDatabase:
    """Get singleton database instance"""
    global _db_instance
    
    with _db_lock:
        if _db_instance is None:
            _db_instance = DistributedPRNGDatabase(db_path)
        return _db_instance
