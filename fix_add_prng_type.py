#!/usr/bin/env python3
"""Add prng_type to JobSpec dataclass"""

with open('coordinator.py', 'r') as f:
    content = f.read()

old = """                        @dataclass
                        class JobSpec:
                            job_id: str
                            seeds: List[int]
                            samples: int
                            lmax: int
                            grid_size: int
                            mining_mode: bool = False
                            search_type: str = 'correlation'
                            target_draw: Optional[List[int]] = None
                            analysis_type: str = 'statistical'
                            attempt: int = 0"""

new = """                        @dataclass
                        class JobSpec:
                            job_id: str
                            seeds: List[int]
                            samples: int
                            lmax: int
                            grid_size: int
                            prng_type: str = 'mt19937'
                            mapping_type: str = 'mod'
                            mining_mode: bool = False
                            search_type: str = 'correlation'
                            target_draw: Optional[List[int]] = None
                            analysis_type: str = 'statistical'
                            attempt: int = 0"""

if old in content:
    content = content.replace(old, new)
    
    # Also update the JobSpec instantiation
    old2 = """                        job = JobSpec(
                            job_id=job_spec.get('job_id', 'unknown'),
                            seeds=job_spec.get('seeds', []),
                            samples=job_spec.get('samples', 1000),
                            lmax=job_spec.get('lmax', 10),
                            grid_size=job_spec.get('grid_size', 50),
                            search_type='residue_sieve'
                        )"""
    
    new2 = """                        job = JobSpec(
                            job_id=job_spec.get('job_id', 'unknown'),
                            seeds=[job_spec['seeds'][0], job_spec['seeds'][-1] + 1],  # Convert to [start, end]
                            samples=job_spec.get('samples', 1000),
                            lmax=job_spec.get('lmax', 10),
                            grid_size=job_spec.get('grid_size', 50),
                            prng_type=job_spec.get('prng_type', 'mt19937'),
                            mapping_type=job_spec.get('mapping_type', 'mod'),
                            search_type='residue_sieve'
                        )"""
    
    content = content.replace(old2, new2)
    
    with open('coordinator.py', 'w') as f:
        f.write(content)
    print("✅ Added prng_type to JobSpec!")
else:
    print("❌ Pattern not found")
